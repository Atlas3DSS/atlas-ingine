#!/usr/bin/env python3
"""Quantize a Qwen3.8-Flash-Next (qwen4exp, 177B) GGUF checkpoint into a native Elastic atlas-ingine pack (#849).

Uses the exact Lloyd-Max codebooks and bit-packing shared by tools/quantize_elastic_ingine.py,
tools/quantize_elastic_qwen38.py, and native/src/backends/qwen4exp/qwen4exp_cuda_ops.cu:
  - nvfp4  (4.50 b/w): E2M1 + E4M3 scale/16 + FP32 outer factor
  - atlas3 (3.50 b/w): 8-centroid symmetric Lloyd-Max (12B / 32 elems) + E4M3 scale/16 + FP32 factor
  - atlas2 (2.50 b/w): 4-centroid symmetric Lloyd-Max (4B / 16 elems) + E4M3 scale/16 + FP32 factor
  - fp8    (8.00 b/w): E4M3FN + per-tensor FP32 weight_scale

Architecture residency split (177B total):
  - 4.95B dense backbone (HC, 36x GDN, 12x FullAttn+QSA, L1 PLE proj, MoE routers & shared experts,
    token_embd, output): stored in model-00001..dense shards (~3.1 GB GPU VRAM).
  - 120.80B routed MoE experts (48 layers x 512 experts, top-10 active, n_ff_exp=640, hidden=2560):
      * atlas-70g (~70.9 GB GPU VRAM, All-In-VRAM on 96 GB RTX PRO 6000 Blackwell):
          Boundary MoE=nvfp4, Core MoE=nvfp4, Dense linear=fp8, embed/output=nvfp4
      * atlas-56g (~56.7 GB GPU VRAM, All-In-VRAM on 80 GB A100/H100 & 96 GB Blackwell):
          Boundary MoE (L00..01, L46..47)=nvfp4, Core MoE (L02..45)=atlas3, Dense linear=fp8, embed/output=nvfp4
      * atlas-42g (~42.6 GB GPU VRAM, All-In-VRAM on 48 GB A6000/L40S; high-hit TieredExpertCache on 24-32 GB):
          Boundary MoE (L00..03, L44..47)=atlas3, Core MoE (L04..43)=atlas2, Dense linear=fp8, embed/output=nvfp4
  - 51.20B PLE n-gram table (per_layer_token_embd.weight, 20M x 2560, 0 GB VRAM):
      Stored in dedicated host-only ple-*.safetensors shards (nvfp4 ~28.8 GB or raw q8_0 ~54.4 GB),
      or omitted with --ple-format external for fast local testing against the source GGUF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

K_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
K_ATLAS2 = torch.tensor([-1.5, -0.453125, 0.453125, 1.5], dtype=torch.float32)
K_ATLAS3 = torch.tensor(
    [-2.0, -1.25, -0.703125, -0.2265625, 0.2265625, 0.703125, 1.25, 2.0],
    dtype=torch.float32,
)

PROFILES: dict[str, dict[str, Any]] = {
    "atlas-70g": {
        "boundary_layers": [0, 1, 2, 3, 44, 45, 46, 47],
        "boundary_moe": "nvfp4",
        "core_moe": "nvfp4",
        "dense_linear": "fp8",
        "embed_output": "nvfp4",
    },
    "atlas-56g": {
        "boundary_layers": [0, 1, 46, 47],
        "boundary_moe": "nvfp4",
        "core_moe": "atlas3",
        "dense_linear": "fp8",
        "embed_output": "nvfp4",
    },
    "atlas-42g": {
        "boundary_layers": [0, 1, 2, 3, 44, 45, 46, 47],
        "boundary_moe": "atlas3",
        "core_moe": "atlas2",
        "dense_linear": "fp8",
        "embed_output": "nvfp4",
    },
}

GGML_GEOM: dict[int, tuple[int, int]] = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    2: (32, 18),    # Q4_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    30: (1, 2),     # BF16
}

SCALAR_FMT: dict[int, tuple[str, int]] = {
    0: ("<B", 1),
    1: ("<b", 1),
    2: ("<H", 2),
    3: ("<h", 2),
    4: ("<I", 4),
    5: ("<i", 4),
    6: ("<f", 4),
    7: ("<?", 1),
    10: ("<Q", 8),
    11: ("<q", 8),
    12: ("<d", 8),
}


@dataclass
class GgufTensor:
    name: str
    dims: tuple[int, ...]
    type_id: int
    shard_path: Path
    data_offset: int
    byte_size: int

    @property
    def in_features(self) -> int:
        return self.dims[0]

    @property
    def total_rows(self) -> int:
        r = 1
        for d in self.dims[1:]:
            r *= d
        return r

    @property
    def row_bytes(self) -> int:
        be, bb = GGML_GEOM[self.type_id]
        assert self.dims[0] % be == 0
        return (self.dims[0] // be) * bb


def discover_gguf_shards(path: Path) -> list[Path]:
    if path.is_file():
        m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", path.name)
        if m:
            stem, total = m.group(1), int(m.group(3))
            return [path.parent / f"{stem}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
        return [path]
    files = sorted(
        p for p in path.iterdir()
        if p.is_file() and p.suffix == ".gguf"
        and not any(x in p.name.lower() for x in ("mmproj", "imatrix", "dflash", "mtp"))
    )
    if not files:
        raise FileNotFoundError(f"No .gguf shards found in {path}")
    return files


def read_gguf_set(path: Path) -> tuple[dict[str, Any], dict[str, GgufTensor]]:
    shards = discover_gguf_shards(path)
    kv: dict[str, Any] = {}
    tensors: dict[str, GgufTensor] = {}

    def rstr(f) -> str:
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", errors="replace")

    def rval(f, vt: int, keep: bool):
        if vt in SCALAR_FMT:
            fmt, sz = SCALAR_FMT[vt]
            raw = f.read(sz)
            return struct.unpack(fmt, raw)[0] if keep else None
        if vt == 8:
            n = struct.unpack("<Q", f.read(8))[0]
            if not keep:
                f.seek(n, 1)
                return None
            return f.read(n).decode("utf-8", errors="replace")
        if vt == 9:
            et, cnt = struct.unpack("<IQ", f.read(12))
            if et in SCALAR_FMT:
                fmt, sz = SCALAR_FMT[et]
                if not keep:
                    f.seek(cnt * sz, 1)
                    return None
                raw = f.read(cnt * sz)
                return list(struct.unpack(f"<{cnt}{fmt[1]}", raw))
            if not keep:
                for _ in range(cnt):
                    rval(f, et, False)
                return None
            return [rval(f, et, True) for _ in range(cnt)]
        raise ValueError(f"Unsupported GGUF value type {vt}")

    skip_keys = {
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.scores",
    }

    for sidx, sp in enumerate(shards):
        with sp.open("rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"Bad GGUF magic in {sp}")
            ver, n_t, n_kv = struct.unpack("<IQQ", f.read(20))
            align = 32
            for _ in range(n_kv):
                k = rstr(f)
                vt = struct.unpack("<I", f.read(4))[0]
                keep = (sidx == 0 and k not in skip_keys) or k == "general.alignment"
                v = rval(f, vt, keep)
                if k == "general.alignment" and v is not None:
                    align = int(v)
                if sidx == 0 and k not in skip_keys:
                    kv[k] = v
            metas = []
            for _ in range(n_t):
                name = rstr(f)
                nd = struct.unpack("<I", f.read(4))[0]
                dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
                tid, rel_off = struct.unpack("<IQ", f.read(12))
                metas.append((name, dims, tid, rel_off))
            pos = f.tell()
            data_start = ((pos + align - 1) // align) * align
            for name, dims, tid, rel_off in metas:
                be, bb = GGML_GEOM[tid]
                numel = 1
                for d in dims:
                    numel *= d
                bsz = (numel // be) * bb
                tensors[name] = GgufTensor(
                    name=name,
                    dims=dims,
                    type_id=tid,
                    shard_path=sp,
                    data_offset=data_start + rel_off,
                    byte_size=bsz,
                )
    return kv, tensors


def dequantize_rows_numpy(type_id: int, raw: bytes, n_rows: int, in_f: int) -> np.ndarray:
    """Vectorized GGML row dequantizer -> float32 [n_rows, in_f]."""
    if type_id == 0:  # F32
        return np.frombuffer(raw, dtype="<f4").reshape(n_rows, in_f).copy()
    if type_id == 1:  # F16
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(n_rows, in_f)
    if type_id == 30:  # BF16
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return u16.view(np.float32).reshape(n_rows, in_f).copy()

    buf = np.frombuffer(raw, dtype=np.uint8)
    if type_id == 2:  # Q4_0: 18B / 32
        nb = (n_rows * in_f) // 32
        blk = buf.reshape(nb, 18)
        d = blk[:, :2].copy().view("<f2").astype(np.float32)  # [nb, 1]
        qs = blk[:, 2:18]
        lo = (qs & 0x0F).astype(np.float32) - 8.0
        hi = (qs >> 4).astype(np.float32) - 8.0
        out = np.concatenate([lo, hi], axis=1) * d
        return out.reshape(n_rows, in_f)

    if type_id == 7:  # Q5_1: 24B / 32
        nb = (n_rows * in_f) // 32
        blk = buf.reshape(nb, 24)
        d = blk[:, 0:2].copy().view("<f2").astype(np.float32)
        m = blk[:, 2:4].copy().view("<f2").astype(np.float32)
        qh = blk[:, 4:8].copy().view("<u4")  # [nb, 1]
        qs = blk[:, 8:24]
        shifts0 = np.arange(16, dtype=np.uint32)[None, :]
        shifts1 = np.arange(16, 32, dtype=np.uint32)[None, :]
        xh0 = (((qh >> shifts0) & 1) << 4).astype(np.float32)
        xh1 = (((qh >> shifts1) & 1) << 4).astype(np.float32)
        lo = (qs & 0x0F).astype(np.float32) + xh0
        hi = (qs >> 4).astype(np.float32) + xh1
        out = np.concatenate([lo, hi], axis=1) * d + m
        return out.reshape(n_rows, in_f)

    if type_id == 8:  # Q8_0: 34B / 32
        nb = (n_rows * in_f) // 32
        blk = buf.reshape(nb, 34)
        d = blk[:, 0:2].copy().view("<f2").astype(np.float32)
        qs = blk[:, 2:34].view(np.int8).astype(np.float32)
        return (qs * d).reshape(n_rows, in_f)

    if type_id == 12:  # Q4_K: 144B / 256
        nb = (n_rows * in_f) // 256
        blk = buf.reshape(nb, 144)
        d = blk[:, 0:2].copy().view("<f2").astype(np.float32)[:, 0]
        dmin = blk[:, 2:4].copy().view("<f2").astype(np.float32)[:, 0]
        sc = blk[:, 4:16]
        qs = blk[:, 16:144]
        out = np.empty((nb, 256), dtype=np.float32)
        for grp in range(4):
            j0, j1 = 2 * grp, 2 * grp + 1
            if j0 < 4:
                s1 = (sc[:, j0] & 63).astype(np.float32)
                m1 = (sc[:, j0 + 4] & 63).astype(np.float32)
                s2 = (sc[:, j1] & 63).astype(np.float32)
                m2 = (sc[:, j1 + 4] & 63).astype(np.float32)
            else:
                s1 = ((sc[:, j0 + 4] & 0x0F) | ((sc[:, j0 - 4] >> 6) << 4)).astype(np.float32)
                m1 = ((sc[:, j0 + 4] >> 4) | ((sc[:, j0] >> 6) << 4)).astype(np.float32)
                s2 = ((sc[:, j1 + 4] & 0x0F) | ((sc[:, j1 - 4] >> 6) << 4)).astype(np.float32)
                m2 = ((sc[:, j1 + 4] >> 4) | ((sc[:, j1] >> 6) << 4)).astype(np.float32)
            q = qs[:, grp * 32 : (grp + 1) * 32]
            out[:, grp * 64 : grp * 64 + 32] = (d * s1)[:, None] * (q & 0x0F) - (dmin * m1)[:, None]
            out[:, grp * 64 + 32 : grp * 64 + 64] = (d * s2)[:, None] * (q >> 4) - (dmin * m2)[:, None]
        return out.reshape(n_rows, in_f)

    if type_id == 13:  # Q5_K: 176B / 256
        nb = (n_rows * in_f) // 256
        blk = buf.reshape(nb, 176)
        d = blk[:, 0:2].copy().view("<f2").astype(np.float32)[:, 0]
        dmin = blk[:, 2:4].copy().view("<f2").astype(np.float32)[:, 0]
        sc = blk[:, 4:16]
        qh = blk[:, 16:48]
        ql = blk[:, 48:176]
        out = np.empty((nb, 256), dtype=np.float32)
        u1, u2 = 1, 2
        for grp in range(4):
            j0, j1 = 2 * grp, 2 * grp + 1
            if j0 < 4:
                s1 = (sc[:, j0] & 63).astype(np.float32)
                m1 = (sc[:, j0 + 4] & 63).astype(np.float32)
                s2 = (sc[:, j1] & 63).astype(np.float32)
                m2 = (sc[:, j1 + 4] & 63).astype(np.float32)
            else:
                s1 = ((sc[:, j0 + 4] & 0x0F) | ((sc[:, j0 - 4] >> 6) << 4)).astype(np.float32)
                m1 = ((sc[:, j0 + 4] >> 4) | ((sc[:, j0] >> 6) << 4)).astype(np.float32)
                s2 = ((sc[:, j1 + 4] & 0x0F) | ((sc[:, j1 - 4] >> 6) << 4)).astype(np.float32)
                m2 = ((sc[:, j1 + 4] >> 4) | ((sc[:, j1] >> 6) << 4)).astype(np.float32)
            q = ql[:, grp * 32 : (grp + 1) * 32]
            h1 = ((qh & u1) != 0).astype(np.float32) * 16.0
            h2 = ((qh & u2) != 0).astype(np.float32) * 16.0
            out[:, grp * 64 : grp * 64 + 32] = (d * s1)[:, None] * ((q & 0x0F) + h1) - (dmin * m1)[:, None]
            out[:, grp * 64 + 32 : grp * 64 + 64] = (d * s2)[:, None] * ((q >> 4) + h2) - (dmin * m2)[:, None]
            u1 <<= 2
            u2 <<= 2
        return out.reshape(n_rows, in_f)

    if type_id == 14:  # Q6_K: 210B / 256
        nb = (n_rows * in_f) // 256
        blk = buf.reshape(nb, 210)
        ql = blk[:, 0:128]
        qh = blk[:, 128:192]
        sc = blk[:, 192:208].view(np.int8).astype(np.float32)
        d = blk[:, 208:210].copy().view("<f2").astype(np.float32)[:, 0]
        out = np.empty((nb, 256), dtype=np.float32)
        is_idx = np.arange(32) // 16  # 0..0 (16x), 1..1 (16x)
        for h in range(2):
            ql_h = ql[:, h * 64 : (h + 1) * 64]
            qh_h = qh[:, h * 32 : (h + 1) * 32]
            sc_h = sc[:, h * 8 : (h + 1) * 8]
            q1 = ((ql_h[:, 0:32] & 0x0F) | (((qh_h >> 0) & 3) << 4)).astype(np.float32) - 32.0
            q2 = ((ql_h[:, 32:64] & 0x0F) | (((qh_h >> 2) & 3) << 4)).astype(np.float32) - 32.0
            q3 = ((ql_h[:, 0:32] >> 4) | (((qh_h >> 4) & 3) << 4)).astype(np.float32) - 32.0
            q4 = ((ql_h[:, 32:64] >> 4) | (((qh_h >> 6) & 3) << 4)).astype(np.float32) - 32.0
            base = h * 128
            out[:, base + 0 : base + 32] = d[:, None] * sc_h[:, is_idx + 0] * q1
            out[:, base + 32 : base + 64] = d[:, None] * sc_h[:, is_idx + 2] * q2
            out[:, base + 64 : base + 96] = d[:, None] * sc_h[:, is_idx + 4] * q3
            out[:, base + 96 : base + 128] = d[:, None] * sc_h[:, is_idx + 6] * q4
        return out.reshape(n_rows, in_f)

    raise ValueError(f"Unsupported GGML type_id={type_id}")


def dequantize_rows_torch(type_id: int, raw: bytes, n_rows: int, in_f: int, dev: torch.device) -> torch.Tensor:
    """GPU-accelerated GGML row dequantizer -> float32 [n_rows, in_f] on dev."""
    if dev.type != "cuda":
        return torch.from_numpy(dequantize_rows_numpy(type_id, raw, n_rows, in_f))
    u8 = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).to(dev, non_blocking=False)
    if type_id == 0:  # F32
        return u8.view(torch.float32).reshape(n_rows, in_f)
    if type_id == 1:  # F16
        return u8.view(torch.float16).to(torch.float32).reshape(n_rows, in_f)
    if type_id == 30:  # BF16
        return u8.view(torch.bfloat16).to(torch.float32).reshape(n_rows, in_f)
    if type_id == 8:  # Q8_0: 34B / 32
        nb = (n_rows * in_f) // 32
        blk = u8.view(nb, 34)
        d = blk[:, 0:2].contiguous().view(torch.float16).to(torch.float32)
        qs = blk[:, 2:34].view(torch.int8).to(torch.float32)
        return (qs * d).reshape(n_rows, in_f)
    if type_id == 14:  # Q6_K: 210B / 256 (hot path for 120B UD-Q6_K_XL experts!)
        nb = (n_rows * in_f) // 256
        blk = u8.view(nb, 210)
        ql = blk[:, 0:128]
        qh = blk[:, 128:192]
        sc = blk[:, 192:208].view(torch.int8).to(torch.float32)
        d = blk[:, 208:210].contiguous().view(torch.float16).to(torch.float32)  # [nb, 1]
        out = torch.empty((nb, 256), dtype=torch.float32, device=dev)
        is_idx = (torch.arange(32, device=dev) // 16)
        for h in range(2):
            ql_h = ql[:, h * 64 : (h + 1) * 64].to(torch.int32)
            qh_h = qh[:, h * 32 : (h + 1) * 32].to(torch.int32)
            sc_h = sc[:, h * 8 : (h + 1) * 8]
            q1 = ((ql_h[:, 0:32] & 0x0F) | (((qh_h >> 0) & 3) << 4)).to(torch.float32) - 32.0
            q2 = ((ql_h[:, 32:64] & 0x0F) | (((qh_h >> 2) & 3) << 4)).to(torch.float32) - 32.0
            q3 = ((ql_h[:, 0:32] >> 4) | (((qh_h >> 4) & 3) << 4)).to(torch.float32) - 32.0
            q4 = ((ql_h[:, 32:64] >> 4) | (((qh_h >> 6) & 3) << 4)).to(torch.float32) - 32.0
            base = h * 128
            out[:, base + 0 : base + 32] = d * sc_h[:, is_idx + 0] * q1
            out[:, base + 32 : base + 64] = d * sc_h[:, is_idx + 2] * q2
            out[:, base + 64 : base + 96] = d * sc_h[:, is_idx + 4] * q3
            out[:, base + 96 : base + 128] = d * sc_h[:, is_idx + 6] * q4
        return out.reshape(n_rows, in_f)
    if type_id == 12:  # Q4_K: 144B / 256
        nb = (n_rows * in_f) // 256
        blk = u8.view(nb, 144)
        d = blk[:, 0:2].contiguous().view(torch.float16).to(torch.float32)[:, 0]
        dmin = blk[:, 2:4].contiguous().view(torch.float16).to(torch.float32)[:, 0]
        sc = blk[:, 4:16].to(torch.int32)
        qs = blk[:, 16:144].to(torch.int32)
        out = torch.empty((nb, 256), dtype=torch.float32, device=dev)
        for grp in range(4):
            j0, j1 = 2 * grp, 2 * grp + 1
            if j0 < 4:
                s1 = (sc[:, j0] & 63).to(torch.float32)
                m1 = (sc[:, j0 + 4] & 63).to(torch.float32)
                s2 = (sc[:, j1] & 63).to(torch.float32)
                m2 = (sc[:, j1 + 4] & 63).to(torch.float32)
            else:
                s1 = ((sc[:, j0 + 4] & 0x0F) | ((sc[:, j0 - 4] >> 6) << 4)).to(torch.float32)
                m1 = ((sc[:, j0 + 4] >> 4) | ((sc[:, j0] >> 6) << 4)).to(torch.float32)
                s2 = ((sc[:, j1 + 4] & 0x0F) | ((sc[:, j1 - 4] >> 6) << 4)).to(torch.float32)
                m2 = ((sc[:, j1 + 4] >> 4) | ((sc[:, j1] >> 6) << 4)).to(torch.float32)
            q = qs[:, grp * 32 : (grp + 1) * 32]
            out[:, grp * 64 : grp * 64 + 32] = (d * s1)[:, None] * (q & 0x0F).to(torch.float32) - (dmin * m1)[:, None]
            out[:, grp * 64 + 32 : grp * 64 + 64] = (d * s2)[:, None] * (q >> 4).to(torch.float32) - (dmin * m2)[:, None]
        return out.reshape(n_rows, in_f)
    if type_id == 13:  # Q5_K: 176B / 256
        nb = (n_rows * in_f) // 256
        blk = u8.view(nb, 176)
        d = blk[:, 0:2].contiguous().view(torch.float16).to(torch.float32)[:, 0]
        dmin = blk[:, 2:4].contiguous().view(torch.float16).to(torch.float32)[:, 0]
        sc = blk[:, 4:16].to(torch.int32)
        qh = blk[:, 16:48].to(torch.int32)
        ql = blk[:, 48:176].to(torch.int32)
        out = torch.empty((nb, 256), dtype=torch.float32, device=dev)
        u1, u2 = 1, 2
        for grp in range(4):
            j0, j1 = 2 * grp, 2 * grp + 1
            if j0 < 4:
                s1 = (sc[:, j0] & 63).to(torch.float32)
                m1 = (sc[:, j0 + 4] & 63).to(torch.float32)
                s2 = (sc[:, j1] & 63).to(torch.float32)
                m2 = (sc[:, j1 + 4] & 63).to(torch.float32)
            else:
                s1 = ((sc[:, j0 + 4] & 0x0F) | ((sc[:, j0 - 4] >> 6) << 4)).to(torch.float32)
                m1 = ((sc[:, j0 + 4] >> 4) | ((sc[:, j0] >> 6) << 4)).to(torch.float32)
                s2 = ((sc[:, j1 + 4] & 0x0F) | ((sc[:, j1 - 4] >> 6) << 4)).to(torch.float32)
                m2 = ((sc[:, j1 + 4] >> 4) | ((sc[:, j1] >> 6) << 4)).to(torch.float32)
            q = ql[:, grp * 32 : (grp + 1) * 32]
            h1 = ((qh & u1) != 0).to(torch.float32) * 16.0
            h2 = ((qh & u2) != 0).to(torch.float32) * 16.0
            out[:, grp * 64 : grp * 64 + 32] = (d * s1)[:, None] * ((q & 0x0F).to(torch.float32) + h1) - (dmin * m1)[:, None]
            out[:, grp * 64 + 32 : grp * 64 + 64] = (d * s2)[:, None] * ((q >> 4).to(torch.float32) + h2) - (dmin * m2)[:, None]
            u1 <<= 2
            u2 <<= 2
        return out.reshape(n_rows, in_f)
    return torch.from_numpy(dequantize_rows_numpy(type_id, raw, n_rows, in_f)).to(dev)


def read_gguf_tensor_f32(t: GgufTensor, row0: int = 0, n_rows: int | None = None, dev: torch.device | None = None) -> torch.Tensor:
    tot = t.total_rows
    cnt = (tot - row0) if n_rows is None else n_rows
    rb = t.row_bytes
    with t.shard_path.open("rb") as f:
        f.seek(t.data_offset + row0 * rb)
        raw = f.read(cnt * rb)
    if dev is not None and dev.type == "cuda":
        ten = dequantize_rows_torch(t.type_id, raw, cnt, t.in_features, dev)
    else:
        ten = torch.from_numpy(dequantize_rows_numpy(t.type_id, raw, cnt, t.in_features))
    if len(t.dims) == 1:
        return ten.reshape(t.dims[0])
    if len(t.dims) == 2:
        return ten  # [cnt, in_f]
    if n_rows is None and len(t.dims) == 3:
        return ten.reshape(t.dims[2], t.dims[1], t.dims[0])  # [E, out_f, in_f]
    return ten


def to_fp8_e4m3fn(t: torch.Tensor) -> torch.Tensor:
    try:
        return t.to(torch.float8_e4m3fn)
    except RuntimeError:
        return t.cpu().to(torch.float8_e4m3fn).to(t.device)


def fp8_to_f32(t: torch.Tensor) -> torch.Tensor:
    try:
        return t.to(torch.float32)
    except RuntimeError:
        return t.cpu().to(torch.float32).to(t.device)


def quantize_fp8_2d(w_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = w_f32.abs().amax().clamp(min=1e-12)
    scale = (amax / 448.0).to(torch.float32)
    q = to_fp8_e4m3fn((w_f32 / scale).clamp(-448.0, 448.0))
    return q.contiguous(), scale.reshape(())


def quantize_codebook_nd(
    w_f32: torch.Tensor, fmt: str, per_expert: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize [R, K] or [E, R, K] -> (codes_u8, scales_fp8_e4m3, scale2_f32)."""
    dev = w_f32.device
    orig_shape = w_f32.shape
    if per_expert:
        assert w_f32.ndim == 3
        E, R, K = orig_shape
    else:
        assert w_f32.ndim == 2
        E, R, K = 1, orig_shape[0], orig_shape[1]
        w_f32 = w_f32.unsqueeze(0)
    assert K % 32 == 0, f"K={K} must be divisible by 32"

    if fmt == "atlas2":
        cb, max_c = K_ATLAS2.to(dev), 1.5
    elif fmt == "atlas3":
        cb, max_c = K_ATLAS3.to(dev), 2.0
    elif fmt == "nvfp4":
        cb, max_c = K_E2M1.to(dev), 6.0
    else:
        raise ValueError(f"Unsupported elastic format: {fmt}")

    amax = w_f32.abs().amax(dim=(1, 2)).clamp(min=1e-12)  # [E]
    scale2 = (amax / (448.0 * max_c)).to(torch.float32)  # [E]

    if fmt in ("atlas2", "atlas3"):
        mids = 0.5 * (cb[:-1] + cb[1:])
    else:
        pos_cb = cb[:8]
        pos_mids = 0.5 * (pos_cb[:-1] + pos_cb[1:])

    q_all = torch.empty((E, R, K // 16, 16), dtype=torch.uint8, device=dev)
    sc_all = torch.empty((E, R, K // 16), dtype=torch.float8_e4m3fn, device=dev)

    chunk_e = 64 if E > 1 else 1
    chunk_r = 2048 if K >= 1024 else 131072
    for e0 in range(0, E, chunk_e):
        e1 = min(E, e0 + chunk_e)
        fac = scale2[e0:e1].view(e1 - e0, 1, 1, 1)
        for r0 in range(0, R, chunk_r):
            r1 = min(R, r0 + chunk_r)
            wb = (w_f32[e0:e1, r0:r1].view(e1 - e0, r1 - r0, K // 16, 16)) / fac
            sb = (wb.abs().amax(dim=-1).clamp(min=1e-12) / max_c).clamp(max=448.0)
            for _ in range(2):
                scaled = wb / sb.unsqueeze(-1).clamp(min=1e-12)
                if fmt in ("atlas2", "atlas3"):
                    idx = torch.bucketize(scaled, mids).to(torch.uint8)
                else:
                    mag_idx = torch.bucketize(scaled.abs(), pos_mids).to(torch.uint8)
                    idx = mag_idx | ((scaled < 0).to(torch.uint8) * 8)
                cval = cb[idx.to(torch.int64)]
                sb = ((wb * cval).sum(dim=-1) / (cval * cval).sum(dim=-1).clamp(min=1e-12)).clamp(min=0.0, max=448.0)
            sb_fp8 = to_fp8_e4m3fn(sb)
            scaled = wb / fp8_to_f32(sb_fp8).unsqueeze(-1).clamp(min=1e-12)
            if fmt in ("atlas2", "atlas3"):
                idx = torch.bucketize(scaled, mids).to(torch.uint8)
            else:
                mag_idx = torch.bucketize(scaled.abs(), pos_mids).to(torch.uint8)
                idx = mag_idx | ((scaled < 0).to(torch.uint8) * 8)
            q_all[e0:e1, r0:r1] = idx
            sc_all[e0:e1, r0:r1] = sb_fp8

    q_flat = q_all.reshape(E, R, K)
    if fmt == "nvfp4":
        q2 = q_flat.view(E, R, K // 2, 2)
        codes = (q2[..., 0] | (q2[..., 1] << 4)).contiguous()
    elif fmt == "atlas2":
        q4 = q_flat.view(E, R, K // 4, 4)
        codes = (q4[..., 0] | (q4[..., 1] << 2) | (q4[..., 2] << 4) | (q4[..., 3] << 6)).contiguous()
    else:
        q32 = q_flat.view(E, R, K // 32, 32)
        lo_b = (q32 & 3).view(E, R, K // 32, 8, 4)
        lo_bytes = lo_b[..., 0] | (lo_b[..., 1] << 2) | (lo_b[..., 2] << 4) | (lo_b[..., 3] << 6)
        hi_b = ((q32 >> 2) & 1).view(E, R, K // 32, 4, 8).to(torch.int32)
        shifts = torch.arange(8, dtype=torch.int32, device=dev)
        hi_bytes = (hi_b << shifts).sum(dim=-1).to(torch.uint8)
        codes = torch.cat([lo_bytes, hi_bytes], dim=-1).reshape(E, R, (K * 3) // 8).contiguous()

    if not per_expert:
        return codes.squeeze(0), sc_all.squeeze(0).contiguous(), scale2.squeeze(0).reshape(())
    return codes, sc_all.contiguous(), scale2.contiguous()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def link_or_copy(src_file: Path, dst_file: Path) -> None:
    if dst_file.exists():
        dst_file.unlink()
    try:
        os.link(src_file, dst_file)
    except OSError:
        import shutil
        shutil.copy2(src_file, dst_file)


def read_safetensors_keys(path: Path) -> list[str]:
    with path.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    return [k for k in hdr.keys() if k != "__metadata__"]


def is_valid_safetensors(path: Path) -> bool:
    """Verify file exists, has valid JSON header, and exact expected byte length."""
    try:
        if not path.is_file():
            return False
        sz = path.stat().st_size
        if sz < 16:
            return False
        with path.open("rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            if hlen == 0 or 8 + hlen > sz:
                return False
            hdr = json.loads(f.read(hlen))
        max_end = 0
        n_tensors = 0
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            n_tensors += 1
            offs = v.get("data_offsets")
            if not isinstance(offs, list) or len(offs) != 2:
                return False
            if offs[1] > max_end:
                max_end = int(offs[1])
        return n_tensors > 0 and (8 + hlen + max_end) == sz
    except Exception:
        return False


def save_safetensors_atomic(sd: dict[str, torch.Tensor], dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    save_file(sd, str(tmp))
    os.replace(tmp, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize Qwen3.8-Flash-Next GGUF into Elastic atlas-ingine safetensors pack(s)")
    ap.add_argument("--src", required=True, help="Source Qwen3.8-Flash-Next GGUF directory or first shard")
    ap.add_argument("--dst", default="", help="Destination Elastic atlas-ingine directory (single-profile mode)")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="atlas-56g")
    ap.add_argument("--profiles", default="",
                    help="Comma-separated multi-profile single-pass mode (e.g. atlas-70g,atlas-56g,atlas-42g)")
    ap.add_argument("--dst-root", default="",
                    help="Destination root for --profiles (writes <dst-root>/Qwen3.8-Flash-Next-<prof>-ingine)")
    ap.add_argument("--ple-format", choices=("nvfp4", "q8_0", "f32", "external"), default="nvfp4",
                    help="Host-only PLE table packaging: nvfp4 (~28.8 GB), q8_0 (~54.4 GB), f32, or external")
    ap.add_argument("--reuse-ple-dir", default="",
                    help="Optional existing Elastic pack dir to hardlink/copy pre-built ple-*.safetensors shards from")
    ap.add_argument("--copy-tokenizer-from", default="",
                    help="Optional directory to copy tokenizer.json / tokenizer_config.json / chat_template.jinja from")
    ap.add_argument("--resume", action="store_true",
                    help="Skip dense/layer/PLE shards that already exist and have valid safetensors headers on disk")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=0, help="CPU torch threads (0 = default)")
    ap.add_argument("--ple-chunk-rows", type=int, default=10_000_000,
                    help="Rows per host-only PLE shard (default 10M -> ~0.9 GB NVFP4 shard for dim=160)")
    ap.add_argument("--layers-limit", type=int, default=0, help="Debug only: process only first N layers (0 = all)")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    src = Path(args.src).resolve()
    dev = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    t0 = time.time()

    if args.profiles:
        prof_names = [p.strip() for p in args.profiles.split(",") if p.strip()]
        for p in prof_names:
            if p not in PROFILES:
                raise ValueError(f"Unknown profile {p}")
        dst_root = Path(args.dst_root or ".").resolve()
        dst_map = {p: dst_root / f"Qwen3.8-Flash-Next-{p}-ingine" for p in prof_names}
    else:
        if not args.dst:
            raise ValueError("--dst is required when --profiles is not specified")
        prof_names = [args.profile]
        dst_map = {args.profile: Path(args.dst).resolve()}

    for d in dst_map.values():
        d.mkdir(parents=True, exist_ok=True)

    kv, tensors = read_gguf_set(src)
    arch = kv.get("general.architecture", "qwen4exp")
    n_layers = int(kv[f"{arch}.block_count"])
    if args.layers_limit > 0:
        n_layers = min(n_layers, args.layers_limit)
    print(f"Qwen3.8-Flash-Next Elastic Quant: profiles={prof_names} layers={n_layers} ple={args.ple_format} dev={dev}", flush=True)

    config_json_txt = json.dumps({"model_type": "qwen4exp", "architectures": ["Qwen4ExpForCausalLM"], "gguf_kv": kv}, indent=2) + "\n"
    tok_src = Path(args.copy_tokenizer_from).resolve() if args.copy_tokenizer_from else None
    for d in dst_map.values():
        (d / "config.json").write_text(config_json_txt)
        if tok_src and tok_src.is_dir():
            import shutil
            for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt", "generation_config.json"):
                if (tok_src / fn).is_file():
                    shutil.copy2(tok_src / fn, d / fn)
        if isinstance(kv.get("tokenizer.chat_template"), str) and kv["tokenizer.chat_template"]:
            (d / "chat_template.jinja").write_text(kv["tokenizer.chat_template"])

    # Per-profile tracking state
    state: dict[str, dict[str, Any]] = {
        p: {
            "dst": dst_map[p],
            "wm": {},
            "qlayers": {},
            "shards": [],
            "gpu_bytes": 0,
            "ple_bytes": 0,
            "layer_formats": [],
        }
        for p in prof_names
    }

    def record_shard(p: str, fname: str, keys: list[str], b: int, digest: str, is_ple: bool) -> None:
        st = state[p]
        for k in keys:
            st["wm"][k] = fname
        if is_ple:
            st["ple_bytes"] += b
        else:
            st["gpu_bytes"] += b
        st["shards"].append({"filename": fname, "byte_size": b, "sha256": digest, "host_only": is_ple})

    # 1. Global + dense backbone tensors -> model-00001 (shared across profiles with identical dense/embed spec)
    p0 = prof_names[0]
    dense_sd: dict[str, torch.Tensor] = {}
    dense_qlayers: dict[str, dict[str, Any]] = {}
    embed_fmt = PROFILES[p0]["embed_output"]
    dense_fmt = PROFILES[p0]["dense_linear"]

    dense_fname = f"model-00001-of-{n_layers + 1:05d}.safetensors"
    dense_path0 = dst_map[p0] / dense_fname
    if args.resume and is_valid_safetensors(dense_path0):
        dense_keys = read_safetensors_keys(dense_path0)
        # Reconstruct dense_qlayers metadata deterministically from tensor names & profile spec
        for name in sorted(tensors.keys()):
            if name == "per_layer_token_embd.weight" or "_exps.weight" in name:
                continue
            if name.startswith("blk.") and int(name.split(".")[1]) >= n_layers:
                continue
            t = tensors[name]
            is_2d = len(t.dims) == 2
            is_embed_out = name in ("token_embd.weight", "output.weight")
            is_small_or_norm = (not is_2d) or (t.dims[0] % 32 != 0) or (t.dims[1] <= 64) or ("conv1d" in name) or ("hc_" in name) or ("ffn_gate_inp" in name)
            if is_embed_out:
                prefix = name[:-len(".weight")]
                dense_qlayers[prefix] = {"quant_algo": "FP8"} if embed_fmt == "fp8" else {"quant_algo": embed_fmt.upper(), "group_size": 16}
            elif not is_small_or_norm:
                prefix = name[:-len(".weight")]
                dense_qlayers[prefix] = {"quant_algo": "FP8"} if dense_fmt == "fp8" else {"quant_algo": dense_fmt.upper(), "group_size": 16}
        dense_b = dense_path0.stat().st_size
        dense_sha = sha256_file(dense_path0)
        print(f"  [RESUME] reused {dense_fname}: {dense_b / 1e9:.2f} GB ({len(dense_keys)} tensors)", flush=True)
    else:
        for name in sorted(tensors.keys()):
            if name == "per_layer_token_embd.weight" or "_exps.weight" in name:
                continue
            if name.startswith("blk.") and int(name.split(".")[1]) >= n_layers:
                continue
            t = tensors[name]
            is_2d = len(t.dims) == 2
            is_embed_out = name in ("token_embd.weight", "output.weight")
            is_small_or_norm = (not is_2d) or (t.dims[0] % 32 != 0) or (t.dims[1] <= 64) or ("conv1d" in name) or ("hc_" in name) or ("ffn_gate_inp" in name)

            if is_embed_out:
                w_f32 = read_gguf_tensor_f32(t, dev=dev).to(dev)
                prefix = name[:-len(".weight")]
                if embed_fmt == "fp8":
                    q, sc = quantize_fp8_2d(w_f32)
                    dense_sd[prefix + ".weight"] = q.cpu()
                    dense_sd[prefix + ".weight_scale"] = sc.cpu()
                    dense_qlayers[prefix] = {"quant_algo": "FP8"}
                else:
                    codes, sc, sc2 = quantize_codebook_nd(w_f32, embed_fmt, per_expert=False)
                    dense_sd[prefix + ".weight"] = codes.cpu()
                    dense_sd[prefix + ".weight_scale"] = sc.cpu()
                    dense_sd[prefix + ".weight_scale_2"] = sc2.cpu()
                    dense_qlayers[prefix] = {"quant_algo": embed_fmt.upper(), "group_size": 16}
                del w_f32
            elif is_small_or_norm:
                w_f32 = read_gguf_tensor_f32(t)
                dense_sd[name] = w_f32.to(torch.float32 if len(t.dims) == 1 or "conv1d" in name else torch.bfloat16).contiguous()
            else:
                w_f32 = read_gguf_tensor_f32(t, dev=dev).to(dev)
                prefix = name[:-len(".weight")]
                if dense_fmt == "fp8":
                    q, sc = quantize_fp8_2d(w_f32)
                    dense_sd[prefix + ".weight"] = q.cpu()
                    dense_sd[prefix + ".weight_scale"] = sc.cpu()
                    dense_qlayers[prefix] = {"quant_algo": "FP8"}
                else:
                    codes, sc, sc2 = quantize_codebook_nd(w_f32, dense_fmt, per_expert=False)
                    dense_sd[prefix + ".weight"] = codes.cpu()
                    dense_sd[prefix + ".weight_scale"] = sc.cpu()
                    dense_sd[prefix + ".weight_scale_2"] = sc2.cpu()
                    dense_qlayers[prefix] = {"quant_algo": dense_fmt.upper(), "group_size": 16}
                del w_f32

        save_safetensors_atomic(dense_sd, dense_path0)
        dense_keys = list(dense_sd.keys())
        dense_b = dense_path0.stat().st_size
        dense_sha = sha256_file(dense_path0)
        print(f"  wrote {dense_fname}: {dense_b / 1e9:.2f} GB ({len(dense_keys)} tensors, shared across {len(prof_names)} profiles)", flush=True)
        dense_sd.clear()
    for p in prof_names:
        if p != p0:
            link_or_copy(dense_path0, dst_map[p] / dense_fname)
        state[p]["qlayers"].update(dense_qlayers)
        record_shard(p, dense_fname, dense_keys, dense_b, dense_sha, is_ple=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Single-pass stream MoE expert layers 0..n_layers-1: read each layer ONCE, quantize each distinct format ONCE!
    for il in range(n_layers):
        tl = time.time()
        fmt_to_profs: dict[str, list[str]] = {}
        for p in prof_names:
            prof = PROFILES[p]
            tfmt = prof["boundary_moe"] if il in set(prof["boundary_layers"]) else prof["core_moe"]
            state[p]["layer_formats"].append(tfmt)
            fmt_to_profs.setdefault(tfmt, []).append(p)

        fname = f"model-{il + 2:05d}-of-{n_layers + 1:05d}.safetensors"
        # Check if every needed format shard for layer `il` already exists cleanly on disk (--resume)
        missing_fmts = [
            tfmt for tfmt, plist in fmt_to_profs.items()
            if not (args.resume and all(is_valid_safetensors(dst_map[p] / fname) for p in plist))
        ]
        if not missing_fmts:
            for tfmt, plist in fmt_to_profs.items():
                fpath0 = dst_map[plist[0]] / fname
                keys = read_safetensors_keys(fpath0)
                b = fpath0.stat().st_size
                dig = sha256_file(fpath0)
                for p in plist:
                    for suffix in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                        state[p]["qlayers"][f"blk.{il}.{suffix}"] = {
                            "quant_algo": tfmt.upper(),
                            "group_size": 16,
                            "per_expert_scale_2": True,
                        }
                    record_shard(p, fname, keys, b, dig, is_ple=False)
            print(f"  [L{il:02d}/{n_layers - 1:02d}] RESUMED ({len(fmt_to_profs)} fmts, {time.time() - tl:.1f}s)", flush=True)
            continue

        # Dequantize gate/up/down ONCE onto dev (GPU) and quantize for every missing format needed here.
        fmt_sds: dict[str, dict[str, torch.Tensor]] = {f: {} for f in missing_fmts}
        for suffix in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            tname = f"blk.{il}.{suffix}.weight"
            w_f32 = read_gguf_tensor_f32(tensors[tname], dev=dev).to(dev)
            prefix = f"blk.{il}.{suffix}"
            for tfmt in missing_fmts:
                codes, sc, sc2 = quantize_codebook_nd(w_f32, tfmt, per_expert=True)
                fmt_sds[tfmt][prefix + ".weight"] = codes.cpu()
                fmt_sds[tfmt][prefix + ".weight_scale"] = sc.cpu()
                fmt_sds[tfmt][prefix + ".weight_scale_2"] = sc2.cpu()
                del codes, sc, sc2
            del w_f32

        for tfmt, plist in fmt_to_profs.items():
            first_p = plist[0]
            fpath0 = dst_map[first_p] / fname
            if tfmt in fmt_sds:
                sd = fmt_sds[tfmt]
                save_safetensors_atomic(sd, fpath0)
                keys = list(sd.keys())
                sd.clear()
            else:
                keys = read_safetensors_keys(fpath0)
            b = fpath0.stat().st_size
            dig = sha256_file(fpath0)
            for p in plist:
                if p != first_p:
                    link_or_copy(fpath0, dst_map[p] / fname)
                for suffix in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
                    state[p]["qlayers"][f"blk.{il}.{suffix}"] = {
                        "quant_algo": tfmt.upper(),
                        "group_size": 16,
                        "per_expert_scale_2": True,
                    }
                record_shard(p, fname, keys, b, dig, is_ple=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        summary = ", ".join(f"{f}->{len(pl)}p" for f, pl in fmt_to_profs.items())
        print(f"  [L{il:02d}/{n_layers - 1:02d}] MoE ({summary}) in {time.time() - tl:.1f}s", flush=True)

    # 3. Host-only PLE table packaging (0 GB VRAM) — quantized ONCE and hardlinked across all profiles!
    eff_ple_fmt = args.ple_format
    if eff_ple_fmt == "nvfp4" and "per_layer_token_embd.weight" in tensors:
        if tensors["per_layer_token_embd.weight"].in_features % 32 != 0:
            eff_ple_fmt = "f32"
    ple_info: dict[str, Any] = {"format": eff_ple_fmt, "host_only": True}
    if eff_ple_fmt != "external" and "per_layer_token_embd.weight" in tensors:
        pt = tensors["per_layer_token_embd.weight"]
        tot_rows = pt.total_rows
        ple_dim = pt.in_features
        ple_info["rows"] = tot_rows
        ple_info["dim"] = ple_dim
        chunk_rows = int(args.ple_chunk_rows)
        ple_info["chunk_rows"] = chunk_rows
        n_ple_shards = (tot_rows + chunk_rows - 1) // chunk_rows
        ple_info["num_shards"] = n_ple_shards
        reuse_dir = Path(args.reuse_ple_dir).resolve() if args.reuse_ple_dir else None
        for si in range(n_ple_shards):
            tp = time.time()
            r0 = si * chunk_rows
            cnt = min(chunk_rows, tot_rows - r0)
            fname = f"ple-{si + 1:05d}-of-{n_ple_shards:05d}.safetensors"
            fpath0 = dst_map[p0] / fname
            if args.resume and is_valid_safetensors(fpath0):
                keys = read_safetensors_keys(fpath0)
            elif reuse_dir is not None and (reuse_dir / fname).is_file():
                link_or_copy(reuse_dir / fname, fpath0)
                if eff_ple_fmt == "q8_0":
                    keys = [f"ple.chunk_{si}.q8_0_raw"]
                elif eff_ple_fmt == "f32":
                    keys = [f"ple.chunk_{si}.f32"]
                else:
                    keys = [f"ple.chunk_{si}.weight", f"ple.chunk_{si}.weight_scale", f"ple.chunk_{si}.weight_scale_2"]
            else:
                if eff_ple_fmt == "q8_0":
                    with pt.shard_path.open("rb") as f:
                        f.seek(pt.data_offset + r0 * pt.row_bytes)
                        raw = f.read(cnt * pt.row_bytes)
                    sd = {f"ple.chunk_{si}.q8_0_raw": torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).reshape(cnt, pt.row_bytes)}
                elif eff_ple_fmt == "f32":
                    sd = {f"ple.chunk_{si}.f32": read_gguf_tensor_f32(pt, row0=r0, n_rows=cnt).contiguous()}
                else:
                    w_f32 = read_gguf_tensor_f32(pt, row0=r0, n_rows=cnt, dev=dev).to(dev)
                    codes, sc, sc2 = quantize_codebook_nd(w_f32, "nvfp4", per_expert=False)
                    del w_f32
                    sd = {
                        f"ple.chunk_{si}.weight": codes.cpu(),
                        f"ple.chunk_{si}.weight_scale": sc.cpu(),
                        f"ple.chunk_{si}.weight_scale_2": sc2.cpu(),
                    }
                    del codes, sc, sc2
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                save_safetensors_atomic(sd, fpath0)
                keys = list(sd.keys())
                sd.clear()
            b = fpath0.stat().st_size
            dig = sha256_file(fpath0)
            for p in prof_names:
                if p != p0:
                    link_or_copy(fpath0, dst_map[p] / fname)
                record_shard(p, fname, keys, b, dig, is_ple=True)
            print(f"  [PLE {si + 1}/{n_ple_shards}] {fname}: {b / 1e9:.2f} GB ({time.time() - tp:.1f}s)", flush=True)

    # 4. Write index, hf_quant_config.json, and atlas_elastic_manifest.json for every profile!
    for p in prof_names:
        st = state[p]
        d = st["dst"]
        tot = st["gpu_bytes"] + st["ple_bytes"]
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": tot}, "weight_map": st["wm"]}, indent=2) + "\n"
        )
        out_qcfg = {
            "producer": {"name": "atlas-ingine-elastic", "version": "1.0"},
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "quantized_layers": st["qlayers"],
            },
            "atlas_elastic": {
                "family": "qwen4exp",
                "model": "Qwen3.8-Flash-Next",
                "profile": p,
                "spec": PROFILES[p],
                "layer_formats": st["layer_formats"],
                "gpu_weight_bytes": st["gpu_bytes"],
                "ple_host_bytes": st["ple_bytes"],
                "ple": ple_info,
                "source": str(src),
            },
        }
        (d / "hf_quant_config.json").write_text(json.dumps(out_qcfg, indent=2) + "\n")
        (d / "atlas_elastic_manifest.json").write_text(
            json.dumps(
                {
                    "profile": p,
                    "gpu_weight_bytes": st["gpu_bytes"],
                    "ple_host_bytes": st["ple_bytes"],
                    "total_bytes": tot,
                    "shards": st["shards"],
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"DONE [{p}]: {d} | GPU weights={st['gpu_bytes'] / 1e9:.2f} GB | Host PLE={st['ple_bytes'] / 1e9:.2f} GB | "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()

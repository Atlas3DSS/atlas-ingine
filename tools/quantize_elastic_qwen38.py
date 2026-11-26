#!/usr/bin/env python3
"""Quantize a Qwen3.8-27B-NVFP4 checkpoint into a native Elastic atlas-ingine checkpoint (#849).

Uses the exact same Lloyd-Max codebooks and bit-packing as tools/quantize_elastic_ingine.py
and native/src/kernels/moe/dequant.cu / native/src/backends/qwen4exp/qwen4exp_cuda_ops.cu:
  - nvfp4  (4.50 b/w): E2M1 + E4M3 scale/16 + FP32 outer factor
  - atlas3 (3.50 b/w): 8-centroid symmetric Lloyd-Max (12B / 32 elems) + E4M3 scale/16 + FP32 factor
  - atlas2 (2.50 b/w): 4-centroid symmetric Lloyd-Max (4B / 16 elems) + E4M3 scale/16 + FP32 factor
  - fp8    (8.00 b/w): E4M3FN + per-tensor FP32 weight_scale

Tiered profiles for Qwen3.8-27B (64 layers: 48 Gated DeltaNet + 16 Full Attention, text-only):
  - Boundary layers: L00..03, L60..63 (8 layers -> higher precision tier)
  - Core layers:     L04..59          (56 layers)

  * atlas-14g (~13.5 GB text-only, fits 16 GB GPUs all-in-VRAM with room for KV):
      Boundary MLP=nvfp4, Core MLP=atlas3, Attn/GDN=fp8, embed_tokens=fp8, lm_head=nvfp4
  * atlas-11g (~10.8 GB text-only, fits 12 GB GPUs all-in-VRAM):
      Boundary MLP=atlas3, Core MLP=atlas2, Boundary Attn=fp8, Core Attn=nvfp4, embed/lm_head=nvfp4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

K_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
K_ATLAS2 = torch.tensor([-1.5, -0.453125, 0.453125, 1.5], dtype=torch.float32)
K_ATLAS3 = torch.tensor(
    [-2.0, -1.25, -0.703125, -0.2265625, 0.2265625, 0.703125, 1.25, 2.0],
    dtype=torch.float32,
)

PROFILES: dict[str, dict[str, str]] = {
    "atlas-17g": {
        "boundary_mlp": "nvfp4",
        "core_mlp": "atlas3",
        "boundary_attn": "fp8",
        "core_attn": "fp8",
        "embed": "fp8",
        "lm_head": "nvfp4",
    },
    "atlas-14g": {
        "boundary_mlp": "nvfp4",
        "core_mlp": "atlas3",
        "boundary_attn": "fp8",
        "core_attn": "nvfp4",
        "embed": "nvfp4",
        "lm_head": "nvfp4",
    },
    "atlas-11g": {
        "boundary_mlp": "atlas3",
        "core_mlp": "atlas2",
        "boundary_attn": "fp8",
        "core_attn": "nvfp4",
        "embed": "nvfp4",
        "lm_head": "nvfp4",
    },
}


def is_boundary_layer(layer_idx: int, num_layers: int = 64) -> bool:
    return layer_idx < 4 or layer_idx >= num_layers - 4


def decode_nvfp4_2d(codes_u8: torch.Tensor, scales_fp8: torch.Tensor, scale2: torch.Tensor) -> torch.Tensor:
    dev = codes_u8.device
    lut = K_E2M1.to(dev)
    lo = lut[(codes_u8 & 0x0F).to(torch.int64)]
    hi = lut[(codes_u8 >> 4).to(torch.int64)]
    r = codes_u8.shape[0]
    vals = torch.stack([lo, hi], dim=-1).reshape(r, -1, 16)
    sc = scales_fp8.to(torch.float32).unsqueeze(-1)
    fac = scale2.to(torch.float32).reshape(())
    return (vals * sc * fac).reshape(r, -1)


def decode_fp8_2d(w_fp8: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    return w_fp8.to(torch.float32) * weight_scale.to(torch.float32).reshape(())


def quantize_fp8_2d(w_f32: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = w_f32.abs().amax().clamp(min=1e-12)
    scale = (amax / 448.0).to(torch.float32)
    q = (w_f32 / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q.contiguous(), scale.reshape(())


def quantize_codebook_2d(
    w_f32: torch.Tensor, fmt: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize [R, K] float32 matrix -> (codes_u8, scales_fp8_e4m3, scale2_f32)."""
    dev = w_f32.device
    R, K = w_f32.shape
    assert K % 32 == 0, f"K={K} must be divisible by 32"
    if fmt == "atlas2":
        cb, max_c = K_ATLAS2.to(dev), 1.5
    elif fmt == "atlas3":
        cb, max_c = K_ATLAS3.to(dev), 2.0
    elif fmt == "nvfp4":
        cb, max_c = K_E2M1.to(dev), 6.0
    else:
        raise ValueError(f"Unsupported elastic format: {fmt}")

    amax = w_f32.abs().amax().clamp(min=1e-12)
    scale2 = (amax / (448.0 * max_c)).to(torch.float32)

    q_rows = torch.empty((R, K // 16, 16), dtype=torch.uint8, device=dev)
    sc_rows = torch.empty((R, K // 16), dtype=torch.float8_e4m3fn, device=dev)
    if fmt in ("atlas2", "atlas3"):
        mids = 0.5 * (cb[:-1] + cb[1:])
    else:
        pos_cb = cb[:8]
        pos_mids = 0.5 * (pos_cb[:-1] + pos_cb[1:])

    chunk_r = 2048
    for r0 in range(0, R, chunk_r):
        r1 = min(R, r0 + chunk_r)
        wb = (w_f32[r0:r1] / scale2).view(r1 - r0, K // 16, 16)
        sb = (wb.abs().amax(dim=-1).clamp(min=1e-12) / max_c).clamp(max=448.0)
        for _ in range(2):
            scaled = wb / sb.unsqueeze(-1).clamp(min=1e-12)
            if fmt in ("atlas2", "atlas3"):
                idx = torch.bucketize(scaled, mids).to(torch.uint8)
                cval = cb[idx.to(torch.int64)]
            else:
                mag_idx = torch.bucketize(scaled.abs(), pos_mids).to(torch.uint8)
                idx = mag_idx | ((scaled < 0).to(torch.uint8) * 8)
                cval = cb[idx.to(torch.int64)]
            sb = ((wb * cval).sum(dim=-1) / (cval * cval).sum(dim=-1).clamp(min=1e-12)).clamp(min=0.0, max=448.0)
        sb_fp8 = sb.to(torch.float8_e4m3fn)
        scaled = wb / sb_fp8.to(torch.float32).unsqueeze(-1).clamp(min=1e-12)
        if fmt in ("atlas2", "atlas3"):
            idx = torch.bucketize(scaled, mids).to(torch.uint8)
        else:
            mag_idx = torch.bucketize(scaled.abs(), pos_mids).to(torch.uint8)
            idx = mag_idx | ((scaled < 0).to(torch.uint8) * 8)
        q_rows[r0:r1] = idx
        sc_rows[r0:r1] = sb_fp8

    q_flat = q_rows.reshape(R, K)
    if fmt == "nvfp4":
        q2 = q_flat.view(R, K // 2, 2)
        codes = (q2[..., 0] | (q2[..., 1] << 4)).contiguous()
    elif fmt == "atlas2":
        q4 = q_flat.view(R, K // 4, 4)
        codes = (q4[..., 0] | (q4[..., 1] << 2) | (q4[..., 2] << 4) | (q4[..., 3] << 6)).contiguous()
    else:
        q32 = q_flat.view(R, K // 32, 32)
        lo_b = (q32 & 3).view(R, K // 32, 8, 4)
        lo_bytes = lo_b[..., 0] | (lo_b[..., 1] << 2) | (lo_b[..., 2] << 4) | (lo_b[..., 3] << 6)
        hi_b = ((q32 >> 2) & 1).view(R, K // 32, 4, 8).to(torch.int32)
        shifts = torch.arange(8, dtype=torch.int32, device=dev)
        hi_bytes = (hi_b << shifts).sum(dim=-1).to(torch.uint8)
        codes = torch.cat([lo_bytes, hi_bytes], dim=-1).reshape(R, (K * 3) // 8).contiguous()

    return codes, sc_rows.contiguous(), scale2.reshape(())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize Qwen3.8-27B-NVFP4 into native Elastic atlas-ingine checkpoint")
    ap.add_argument("--src", required=True, help="Source Qwen3.8-27B-NVFP4 directory")
    ap.add_argument("--dst", required=True, help="Destination Elastic atlas-ingine directory")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="atlas-14g")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-shard-gb", type=float, default=4.5, help="Target safetensors shard size in GB")
    ap.add_argument("--keep-vision", action="store_true", help="Keep model.visual.* tensors")
    ap.add_argument("--keep-mtp", action="store_true", help="Keep mtp.* tensors")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    prof = PROFILES[args.profile]
    print(f"Qwen3.8-27B Elastic Quant: profile={args.profile} spec={prof} device={dev}", flush=True)

    # Copy tokenizer, config, chat template, license
    for fn in os.listdir(src):
        if fn.endswith(".safetensors") or fn in ("model.safetensors.index.json", "hf_quant_config.json"):
            continue
        sp, dp = src / fn, dst / fn
        if sp.is_file():
            shutil.copy2(sp, dp)

    src_index = json.loads((src / "model.safetensors.index.json").read_text())
    src_qcfg = json.loads((src / "hf_quant_config.json").read_text())
    wm: dict[str, str] = src_index["weight_map"]
    src_qlayers: dict[str, dict] = src_qcfg["quantization"]["quantized_layers"]

    shard_cache: dict[str, dict[str, torch.Tensor]] = {}

    def get_tensor(k: str) -> torch.Tensor:
        sf = wm[k]
        if sf not in shard_cache:
            if len(shard_cache) >= 2:
                shard_cache.clear()
            shard_cache[sf] = load_file(str(src / sf), device="cpu")
        return shard_cache[sf][k]

    # Collect logical module prefixes
    all_keys = sorted(wm.keys())
    consumed: set[str] = set()
    new_qlayers: dict[str, dict] = {}
    out_shards: list[dict[str, torch.Tensor]] = [{}]
    out_shard_bytes = 0
    max_shard_bytes = int(args.max_shard_gb * 1e9)
    t0 = time.time()

    def add_tensors(tensors: dict[str, torch.Tensor]) -> None:
        nonlocal out_shard_bytes
        sz = sum(v.numel() * v.element_size() for v in tensors.values())
        if out_shards[-1] and out_shard_bytes + sz > max_shard_bytes:
            out_shards.append({})
            out_shard_bytes = 0
        out_shards[-1].update(tensors)
        out_shard_bytes += sz

    def target_fmt_for(prefix: str) -> str | None:
        if prefix == "model.language_model.embed_tokens":
            return prof["embed"]
        if prefix == "lm_head":
            return prof["lm_head"]
        if ".layers." in prefix:
            parts = prefix.split(".")
            l_idx = int(parts[parts.index("layers") + 1])
            bnd = is_boundary_layer(l_idx)
            if ".mlp." in prefix:
                return prof["boundary_mlp"] if bnd else prof["core_mlp"]
            if ".self_attn." in prefix or ".linear_attn." in prefix:
                if prefix in src_qlayers:
                    return prof["boundary_attn"] if bnd else prof["core_attn"]
        return None

    for k in all_keys:
        if k in consumed:
            continue
        if not args.keep_vision and k.startswith("model.visual."):
            consumed.add(k)
            continue
        if not args.keep_mtp and k.startswith("mtp."):
            consumed.add(k)
            continue

        if k.endswith(".weight"):
            prefix = k[: -len(".weight")]
            tfmt = target_fmt_for(prefix)
            sinfo = src_qlayers.get(prefix)
            sfmt = sinfo["quant_algo"].lower() if sinfo else "bf16"
            if tfmt is not None and tfmt != sfmt:
                # Requantize weight
                w_raw = get_tensor(k).to(dev)
                if sfmt == "nvfp4":
                    ws = get_tensor(prefix + ".weight_scale").to(dev)
                    ws2 = get_tensor(prefix + ".weight_scale_2").to(dev)
                    w_f32 = decode_nvfp4_2d(w_raw, ws, ws2)
                    for suf in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale"):
                        if prefix + suf in wm:
                            consumed.add(prefix + suf)
                elif sfmt == "fp8":
                    ws = get_tensor(prefix + ".weight_scale").to(dev)
                    w_f32 = decode_fp8_2d(w_raw, ws)
                    for suf in (".weight", ".weight_scale", ".input_scale"):
                        if prefix + suf in wm:
                            consumed.add(prefix + suf)
                else:
                    w_f32 = w_raw.to(torch.float32)
                    consumed.add(k)
                del w_raw

                pack: dict[str, torch.Tensor] = {}
                if tfmt == "fp8":
                    q, sc = quantize_fp8_2d(w_f32)
                    pack[prefix + ".weight"] = q.cpu()
                    pack[prefix + ".weight_scale"] = sc.cpu()
                    new_qlayers[prefix] = {"quant_algo": "FP8"}
                else:
                    codes, sc, sc2 = quantize_codebook_2d(w_f32, tfmt)
                    pack[prefix + ".weight"] = codes.cpu()
                    pack[prefix + ".weight_scale"] = sc.cpu()
                    pack[prefix + ".weight_scale_2"] = sc2.cpu()
                    new_qlayers[prefix] = {"quant_algo": tfmt.upper(), "group_size": 16}
                del w_f32
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                add_tensors(pack)
                continue

            # Passthrough quantized module with all sibling scale tensors
            if sinfo is not None:
                pack = {}
                for suf in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale"):
                    sk = prefix + suf
                    if sk in wm:
                        pack[sk] = get_tensor(sk)
                        consumed.add(sk)
                new_qlayers[prefix] = dict(sinfo)
                add_tensors(pack)
                continue

        # Plain scalar/norm/conv tensor passthrough
        add_tensors({k: get_tensor(k)})
        consumed.add(k)

    new_wm: dict[str, str] = {}
    shard_manifest = []
    total_bytes = 0
    n_shards = len(out_shards)
    for i, sd in enumerate(out_shards, start=1):
        fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        fpath = dst / fname
        save_file(sd, str(fpath))
        for k in sd:
            new_wm[k] = fname
        b = fpath.stat().st_size
        total_bytes += b
        shard_manifest.append({"filename": fname, "byte_size": b, "sha256": sha256_file(fpath)})
        print(f"  wrote {fname}: {b / 1e9:.2f} GB ({len(sd)} tensors)", flush=True)

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": new_wm}, indent=2) + "\n"
    )
    out_qcfg = {
        "producer": {"name": "atlas-ingine-elastic", "version": "1.0"},
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": None,
            "quantized_layers": new_qlayers,
        },
        "atlas_elastic": {
            "family": "qwen3_5",
            "model": "Qwen3.8-27B",
            "profile": args.profile,
            "spec": prof,
            "text_only": (not args.keep_vision and not args.keep_mtp),
            "source": str(src),
        },
    }
    (dst / "hf_quant_config.json").write_text(json.dumps(out_qcfg, indent=2) + "\n")
    (dst / "atlas_elastic_manifest.json").write_text(
        json.dumps({"profile": args.profile, "total_bytes": total_bytes, "shards": shard_manifest}, indent=2) + "\n"
    )
    print(f"DONE: {dst} ({total_bytes / 1e9:.2f} GB, {n_shards} shards, {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

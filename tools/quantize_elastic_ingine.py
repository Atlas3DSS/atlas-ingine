#!/usr/bin/env python3
"""Quantize an Inkling-2-Mini Hugging Face checkpoint into a native Elastic sub-4-bit atlas-ingine checkpoint.

Formats supported per physical layer (0..23) in hf_quant_config.json:
  - nvfp4  (4.50 b/w): native E2M1 + E4M3 scale/16 + FP32 factor/expert (passthrough if already nvfp4)
  - mxfp8  (8.25 b/w): native E4M3 + E8M0 scale/32 (passthrough if already mxfp8)
  - atlas3 (3.50 b/w): 3-bit symmetric Lloyd-Max codebook (group 16, E4M3 scale + FP32 factor, FP16-exact product)
  - atlas2 (2.50 b/w): 2-bit symmetric Lloyd-Max codebook (group 16, E4M3 scale + FP32 factor, FP16-exact product)

Loop-tiered profiles (matching Inkling-2-Mini physics):
  - Tier A: L00..05, L18..19 (8 single-pass layers, 8.15B experts)
  - Tier B: L06..17          (12 3x-looped core layers, 12.23B experts -> +1 precision tier)
  - Tier C: L20..23          (4 output tail layers, 4.08B experts -> +2 precision tiers)

  * atlas-10g: Tier A=atlas2, Tier B=atlas2, Tier C=atlas3 (~10.0 GB total weights, fits 12 GB GPUs)
  * atlas-12g: Tier A=atlas2, Tier B=atlas3, Tier C=nvfp4  (~12.1 GB total weights, fits 16 GB GPUs)
  * atlas-14g: Tier A=atlas3, Tier B=nvfp4,  Tier C=nvfp4  (~14.2 GB total weights, fits 16-24 GB GPUs)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

K_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
K_ATLAS2 = torch.tensor([-1.5, -0.453125, 0.453125, 1.5], dtype=torch.float32)
K_ATLAS3 = torch.tensor([-2.0, -1.25, -0.703125, -0.2265625, 0.2265625, 0.703125, 1.25, 2.0], dtype=torch.float32)

PROFILES: dict[str, tuple[str, str, str]] = {
    "atlas-10g": ("atlas2", "atlas2", "atlas3"),
    "atlas-12g": ("atlas2", "atlas3", "nvfp4"),
    "atlas-14g": ("atlas3", "nvfp4", "nvfp4"),
}


def layer_tier(l: int) -> int:
    if 6 <= l <= 17:
        return 1  # Tier B: 3x looped core
    if l >= 20:
        return 2  # Tier C: output tail
    return 0      # Tier A: single-pass prelude + post-loop entry


def decode_nvfp4(codes_u8: torch.Tensor, scales_fp8: torch.Tensor, factor_f32: torch.Tensor) -> torch.Tensor:
    # codes_u8: [E, R, K/2], scales_fp8: [E, R, K/16], factor_f32: [E]
    dev = codes_u8.device
    lut = K_E2M1.to(dev)
    lo = lut[(codes_u8 & 0x0F).to(torch.int64)]
    hi = lut[(codes_u8 >> 4).to(torch.int64)]
    vals = torch.stack([lo, hi], dim=-1).reshape(codes_u8.shape[0], codes_u8.shape[1], -1)  # [E, R, K]
    sc = scales_fp8.to(torch.float32).unsqueeze(-1)  # [E, R, K/16, 1]
    vals = vals.view(codes_u8.shape[0], codes_u8.shape[1], -1, 16) * sc
    vals = vals * factor_f32.view(-1, 1, 1, 1)
    return vals.reshape(codes_u8.shape[0], codes_u8.shape[1], -1)


def decode_mxfp8(codes_fp8: torch.Tensor, scales_u8: torch.Tensor) -> torch.Tensor:
    # codes_fp8: [E, R, K], scales_u8: [E, R, K/32] (E8M0)
    vals = codes_fp8.to(torch.float32).view(codes_fp8.shape[0], codes_fp8.shape[1], -1, 32)
    sc = torch.pow(2.0, scales_u8.to(torch.float32) - 127.0).unsqueeze(-1)
    return (vals * sc).reshape(codes_fp8.shape[0], codes_fp8.shape[1], -1)


def quantize_codebook(w_f32: torch.Tensor, fmt: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize [E, R, K] float32 weights into (codes_u8, scales_fp8, amax_f32, den_f32)."""
    dev = w_f32.device
    E, R, K = w_f32.shape
    assert K % 32 == 0, f"K={K} must be divisible by 32"
    if fmt == "atlas2":
        cb = K_ATLAS2.to(dev)
        max_c = 1.5
    elif fmt == "atlas3":
        cb = K_ATLAS3.to(dev)
        max_c = 2.0
    elif fmt == "nvfp4":
        cb = K_E2M1.to(dev)
        max_c = 6.0
    else:
        raise ValueError(f"Unsupported target format: {fmt}")

    # Per-expert outer factor: factor = amax / (448.0 * max_c) so block scale fits comfortably in E4M3 [0, 448].
    amax = w_f32.abs().amax(dim=(1, 2)).clamp(min=1e-12)  # [E]
    den = torch.full((E,), 448.0 * max_c, dtype=torch.float32, device=dev)
    factor = amax / den  # [E]

    # Normalize by per-expert factor -> blocks [E, R, K/16, 16]
    w_norm = (w_f32 / factor.view(E, 1, 1)).view(E, R, K // 16, 16)
    blk_max = w_norm.abs().amax(dim=-1).clamp(min=1e-12)  # [E, R, K/16]
    # Initial scale: blk_max / max_c, refined with 2 iterations of alternating least-squares (optimal MSE!)
    sc = (blk_max / max_c).clamp(max=448.0)
    # Process in chunks over experts to keep peak VRAM small (<2 GB)
    q_all = torch.empty((E, R, K // 16, 16), dtype=torch.uint8, device=dev)
    sc_all = torch.empty((E, R, K // 16), dtype=torch.float8_e4m3fn, device=dev)

    # Midpoints for fast nearest-centroid lookup (cb is sorted ascending for atlas2/atlas3)
    if fmt in ("atlas2", "atlas3"):
        mids = 0.5 * (cb[:-1] + cb[1:])  # [3] or [7]

    chunk_e = 16
    for e0 in range(0, E, chunk_e):
        e1 = min(E, e0 + chunk_e)
        wb = w_norm[e0:e1]  # [ce, R, nb, 16]
        sb = sc[e0:e1].clone()  # [ce, R, nb]
        for _ in range(2):
            scaled = wb / sb.unsqueeze(-1).clamp(min=1e-12)
            if fmt in ("atlas2", "atlas3"):
                idx = torch.bucketize(scaled, mids).to(torch.uint8)
                cval = cb[idx.to(torch.int64)]
            else:
                # NVFP4: map |x| to positive E2M1 [0..7], then attach sign bit 8
                pos_cb = cb[:8]
                pos_mids = 0.5 * (pos_cb[:-1] + pos_cb[1:])
                mag_idx = torch.bucketize(scaled.abs(), pos_mids).to(torch.uint8)
                sign = (scaled < 0).to(torch.uint8) * 8
                idx = mag_idx | sign
                cval = cb[idx.to(torch.int64)]
            # Optimal L2 scale update: s* = sum(wb * cval) / sum(cval^2)
            num = (wb * cval).sum(dim=-1)
            den_s = (cval * cval).sum(dim=-1).clamp(min=1e-12)
            sb = (num / den_s).clamp(min=0.0, max=448.0)
        # Snap scale to E4M3, then do final exact assignment against the snapped E4M3 scale!
        sb_fp8 = sb.to(torch.float8_e4m3fn)
        sb_eff = sb_fp8.to(torch.float32).unsqueeze(-1).clamp(min=1e-12)
        scaled = wb / sb_eff
        if fmt in ("atlas2", "atlas3"):
            idx = torch.bucketize(scaled, mids).to(torch.uint8)
        else:
            pos_cb = cb[:8]
            pos_mids = 0.5 * (pos_cb[:-1] + pos_cb[1:])
            mag_idx = torch.bucketize(scaled.abs(), pos_mids).to(torch.uint8)
            sign = (scaled < 0).to(torch.uint8) * 8
            idx = mag_idx | sign
        q_all[e0:e1] = idx
        sc_all[e0:e1] = sb_fp8

    q_flat = q_all.reshape(E, R, K)
    if fmt == "nvfp4":
        q2 = q_flat.view(E, R, K // 2, 2)
        codes = (q2[..., 0] | (q2[..., 1] << 4)).contiguous()
    elif fmt == "atlas2":
        q4 = q_flat.view(E, R, K // 4, 4)
        codes = (q4[..., 0] | (q4[..., 1] << 2) | (q4[..., 2] << 4) | (q4[..., 3] << 6)).contiguous()
    else:
        # ATLAS3: 32 elements -> 12 bytes: [lo0(4B), lo1(4B), hi01(4B)]
        # lo0 = 2-bit low bits of elem 0..15 (4 bytes), lo1 = elem 16..31 (4 bytes), hi01 = 1-bit high bits of 0..31 (4 bytes)
        q32 = q_flat.view(E, R, K // 32, 32)
        lo_bits = q32 & 3
        hi_bits = (q32 >> 2) & 1
        lo_b = lo_bits.view(E, R, K // 32, 8, 4)
        lo_bytes = lo_b[..., 0] | (lo_b[..., 1] << 2) | (lo_b[..., 2] << 4) | (lo_b[..., 3] << 6)  # [E, R, K/32, 8]
        hi_b = hi_bits.view(E, R, K // 32, 4, 8).to(torch.int32)
        shifts = torch.arange(8, dtype=torch.int32, device=dev)
        hi_bytes = (hi_b << shifts).sum(dim=-1).to(torch.uint8)  # [E, R, K/32, 4]
        codes = torch.cat([lo_bytes, hi_bytes], dim=-1).reshape(E, R, (K * 3) // 8).contiguous()

    return codes, sc_all.contiguous(), amax.contiguous(), den.contiguous()


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize Inkling-2-Mini HF checkpoint to native Elastic atlas-ingine format")
    ap.add_argument("--src", required=True, help="Source HF Inkling-2-Mini checkpoint directory")
    ap.add_argument("--dst", required=True, help="Destination Elastic atlas-ingine checkpoint directory")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="atlas-12g", help="Loop-tiered bit profile")
    ap.add_argument("--tier-a", default="", help="Override Tier A (L0-5,18-19) format: atlas2|atlas3|nvfp4|mxfp8")
    ap.add_argument("--tier-b", default="", help="Override Tier B (L6-17 looped) format")
    ap.add_argument("--tier-c", default="", help="Override Tier C (L20-23 tail) format")
    ap.add_argument("--no-vision", action="store_true", help="Omit vision encoder tensors (model.visual.*, saves ~0.64 GB on disk; run with --no-vision)")
    ap.add_argument("--no-mtp", action="store_true", help="Omit DSpark MTP speculative drafter (mtp.safetensors, saves ~0.64 GB on disk)")
    ap.add_argument("--text-only", action="store_true", help="Shorthand for --no-vision --no-mtp (minimal text-only checkpoint, saves ~1.27 GB on disk)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    omit_vision = bool(args.no_vision or args.text_only)
    omit_mtp = bool(args.no_mtp or args.text_only)

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ta, tb, tc = PROFILES[args.profile]
    if args.tier_a:
        ta = args.tier_a
    if args.tier_b:
        tb = args.tier_b
    if args.tier_c:
        tc = args.tier_c
    target_fmts = [ta if layer_tier(l) == 0 else (tb if layer_tier(l) == 1 else tc) for l in range(24)]
    print(f"Profile={args.profile} -> TierA={ta} TierB={tb} TierC={tc} on {dev}", flush=True)

    # Read source hf_quant_config.json to know source format per layer
    qcfg = json.loads((src / "hf_quant_config.json").read_text())
    src_fmts: dict[int, str] = {}
    for g in qcfg["model"]["parameter_groups"]:
        dt = g["compute_dtype"]
        for t in g["tensors"]:
            if t.endswith("mlp.experts.w13_weight"):
                l = int(t.split(".")[3])
                src_fmts[l] = dt

    # Copy metadata and auxiliary safetensors (e.g. mtp.safetensors for DSpark --spec), skipping weight shards.
    for fn in os.listdir(src):
        if (fn.startswith("model") and fn.endswith(".safetensors")) or fn in ("model.safetensors.index.json", "hf_quant_config.json"):
            continue
        if omit_mtp and fn == "mtp.safetensors":
            continue
        sp, dp = src / fn, dst / fn
        if dp.exists() or dp.is_symlink():
            dp.unlink()
        if sp.is_file():
            shutil.copy2(sp, dp)
        elif sp.is_dir():
            shutil.copytree(sp, dp, dirs_exist_ok=True)

    # Load shard map
    idx_path = src / "model.safetensors.index.json"
    if idx_path.exists():
        src_index = json.loads(idx_path.read_text())
        weight_map: dict[str, str] = src_index["weight_map"]
    else:
        raise RuntimeError(f"Expected {idx_path}")

    # Group tensors by shard so we process shard-by-shard, except expert tensors which we handle per layer
    layer_expert_keys: dict[int, set[str]] = {l: set() for l in range(24)}
    for k in weight_map:
        if ".mlp.experts." in k:
            l = int(k.split(".")[3])
            layer_expert_keys[l].add(k)

    # Cache loaded shards minimally
    shard_cache: dict[str, dict[str, torch.Tensor]] = {}

    def get_tensor(key: str) -> torch.Tensor:
        sf = weight_map[key]
        if sf not in shard_cache:
            if len(shard_cache) >= 2:
                shard_cache.clear()
            shard_cache[sf] = load_file(str(src / sf), device="cpu")
        return shard_cache[sf][key]

    new_weight_map: dict[str, str] = {}
    total_bytes = 0
    t0 = time.time()

    # 1. Copy all non-expert tensors verbatim into shard 00001
    non_expert: dict[str, torch.Tensor] = {}
    for sf in sorted(set(weight_map.values())):
        data = load_file(str(src / sf), device="cpu")
        for k, v in data.items():
            if ".mlp.experts." not in k:
                if omit_vision and k.startswith("model.visual."):
                    continue
                non_expert[k] = v
    ne_name = "model-00001-of-00025.safetensors"
    save_file(non_expert, str(dst / ne_name))
    for k, v in non_expert.items():
        new_weight_map[k] = ne_name
        total_bytes += v.numel() * v.element_size()
    print(f"[non-expert] wrote {len(non_expert)} tensors ({total_bytes / 1e9:.2f} GB) in {time.time() - t0:.1f}s", flush=True)
    non_expert.clear()

    # 2. Quantize each physical layer 0..23 into its own shard
    for l in range(24):
        tl = time.time()
        p = f"model.llm.layers.{l}.mlp.experts."
        sfmt = src_fmts[l]
        tfmt = target_fmts[l]
        out_tensors: dict[str, torch.Tensor] = {}
        if sfmt == tfmt:
            for k in sorted(layer_expert_keys[l]):
                out_tensors[k] = get_tensor(k)
        else:
            for wname in ("w13", "w2"):
                codes_in = get_tensor(p + f"{wname}_weight").to(dev)
                scales_in = get_tensor(p + f"{wname}_weight_scale").to(dev)
                if sfmt == "nvfp4":
                    amax_in = get_tensor(p + f"{wname}_weight_amax").to(dev).to(torch.float32)
                    den_in = get_tensor(p + f"{wname}_weight_outer_scale_denominator").to(dev).to(torch.float32)
                    fac_in = amax_in / (den_in if den_in.numel() > 1 else den_in.expand_as(amax_in))
                    w_f32 = decode_nvfp4(codes_in, scales_in, fac_in)
                elif sfmt == "mxfp8":
                    w_f32 = decode_mxfp8(codes_in, scales_in)
                else:
                    raise RuntimeError(f"Unsupported source format {sfmt} at layer {l}")
                del codes_in, scales_in
                q_codes, q_scales, q_amax, q_den = quantize_codebook(w_f32, tfmt)
                del w_f32
                out_tensors[p + f"{wname}_weight"] = q_codes.cpu()
                out_tensors[p + f"{wname}_weight_scale"] = q_scales.cpu()
                out_tensors[p + f"{wname}_weight_amax"] = q_amax.cpu()
                out_tensors[p + f"{wname}_weight_outer_scale_denominator"] = q_den.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        sh_name = f"model-{l + 2:05d}-of-00025.safetensors"
        save_file(out_tensors, str(dst / sh_name))
        sh_bytes = sum(v.numel() * v.element_size() for v in out_tensors.values())
        total_bytes += sh_bytes
        for k in out_tensors:
            new_weight_map[k] = sh_name
        print(f"[L{l:02d}] {sfmt} -> {tfmt} ({sh_bytes / 1e6:.1f} MB, {time.time() - tl:.1f}s)", flush=True)

    # 3. Write updated model.safetensors.index.json and hf_quant_config.json
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": new_weight_map}, indent=2) + "\n"
    )
    # Rebuild parameter_groups cleanly for expert tensors
    new_groups = []
    by_fmt: dict[str, list[str]] = {}
    for l in range(24):
        p = f"model.llm.layers.{l}.mlp.experts."
        by_fmt.setdefault(target_fmts[l], []).extend([p + "w13_weight", p + "w2_weight"])
    for g in qcfg["model"]["parameter_groups"]:
        kept = [t for t in g["tensors"] if ".mlp.experts." not in t and not (omit_vision and t.startswith("model.visual."))]
        if kept:
            g2 = dict(g)
            g2["tensors"] = kept
            new_groups.append(g2)
    for dt, tensors in by_fmt.items():
        new_groups.append({"compute_dtype": dt, "tensors": tensors})
    qcfg["model"]["parameter_groups"] = new_groups
    qcfg["atlas_elastic"] = {
        "profile": args.profile,
        "tier_a": ta,
        "tier_b": tb,
        "tier_c": tc,
        "layer_formats": target_fmts,
        "no_vision": omit_vision,
        "no_mtp": omit_mtp,
        "source": str(src),
    }
    (dst / "hf_quant_config.json").write_text(json.dumps(qcfg, indent=2) + "\n")
    print(f"DONE: {dst} ({total_bytes / 1e9:.2f} GB total, {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()

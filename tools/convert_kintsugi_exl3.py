#!/usr/bin/env python3
"""Convert Kintsugi-Inkling-2-Mini (EXL3/Escha W2 Map M2) + native Inkling-2-Mini-ingine
into a native atlas-ingine safetensors checkpoint (~9.0 GB text-only in VRAM with --dense-fp8,
<11.6 GB VRAM at 262K context with --kv-q4, preserving native vision + DSpark + JSON mode).

License & Attribution:
  - Inkling-2-Mini base weights: Thinking Machines (Apache-2.0)
  - Kintsugi-Inkling-2-Mini-12G.gguf EXL3 Map M2 experts: Apache-2.0 / Escha W2 (exllamav3 MIT)
  - atlas-ingine checkpoint format: Apache-2.0

EXL3 tensor layout stored in safetensors:
  - model.llm.layers.<l>.mlp.experts.w13_weight : U8  [128, N/16, K/16, 32*bits]  (N=2304, K=2304, stacked [gate; up])
  - model.llm.layers.<l>.mlp.experts.w13_rin    : F16 [128, 2304]
  - model.llm.layers.<l>.mlp.experts.w13_rout   : F16 [128, 2304]
  - model.llm.layers.<l>.mlp.experts.w2_weight  : U8  [128, N/16, K/16, 32*bits]  (N=2304, K=1152)
  - model.llm.layers.<l>.mlp.experts.w2_rin     : F16 [128, 1152]
  - model.llm.layers.<l>.mlp.experts.w2_rout    : F16 [128, 2304]
Note: GGUF escha_code is [128, K/16, N/16, 16*bits] uint16; transposing dims (1, 2) to
[128, N/16, K/16, 32*bits] uint8 makes every output tile's K-strip 100% contiguous for both
decode (moe.cu) and prefill (wgemm.cu).

Optional Q4_0 embedding (--embed-q4, default ON):
  - model.llm.embed.weight : U8 [201024, 1296]  (SoA per row: [1152B nibbles | 144B FP16 scales], 260.5 MB vs 926.3 MB BF16)
"""
import argparse
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path
import numpy as np


def parse_gguf(path: Path):
    f = open(path, "rb")
    magic, ver, nt, nkv = struct.unpack("<4sIQQ", f.read(24))
    assert magic == b"GGUF", f"Not a GGUF file: {magic}"

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8")

    def rval(t):
        fmts = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        if t in fmts:
            fmt = fmts[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        if t == 8:
            return rstr()
        if t == 9:
            et, ln = struct.unpack("<IQ", f.read(12))
            if et in fmts:
                f.seek(ln * struct.calcsize(fmts[et]), os.SEEK_CUR)
                return None
            for _ in range(ln):
                rval(et)
            return None
        raise ValueError(f"Unknown GGUF kv type {t}")

    meta = {}
    for _ in range(nkv):
        k = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        v = rval(t)
        if k in ("general.alignment", "inkling2.escha.map"):
            meta[k] = v

    tensors = {}
    for _ in range(nt):
        name = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
        tt, off = struct.unpack("<IQ", f.read(12))
        tensors[name] = (dims, tt, off)

    align = int(meta.get("general.alignment", 32))
    data_start = (f.tell() + align - 1) // align * align
    escha_map = json.loads(meta["inkling2.escha.map"])
    return f, data_start, tensors, escha_map


def parse_safetensors_dir(src_dir: Path):
    idx_path = src_dir / "model.safetensors.index.json"
    if idx_path.exists():
        wmap = json.loads(idx_path.read_text())["weight_map"]
        files = sorted(set(wmap.values()))
    else:
        files = ["model.safetensors"]
        wmap = None

    entries = {}
    for fname in files:
        fpath = src_dir / fname
        with open(fpath, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
            base = 8 + hlen
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            entries[k] = (fpath, base + s, e - s, v["dtype"], v["shape"])
    return entries


def quantize_embed_q4_0(bf16_bytes: bytes, rows: int, cols: int) -> bytes:
    """Quantize BF16 [rows, cols] to SoA Q4_0 U8 [rows, cols/2 + (cols/32)*2] = [rows, 1296].
    Deterministic signed max matching native Q4_0 KV cache."""
    assert cols % 32 == 0
    nb = cols // 32
    # Convert BF16 uint16 -> float32 by shifting left 16 bits
    u16 = np.frombuffer(bf16_bytes, dtype=np.uint16).reshape(rows, cols)
    out = np.empty((rows, cols // 2 + nb * 2), dtype=np.uint8)
    chunk = 4096
    for r0 in range(0, rows, chunk):
        r1 = min(r0 + chunk, rows)
        u32 = u16[r0:r1].astype(np.uint32) << 16
        x = u32.view(np.float32).reshape(r1 - r0, nb, 32)
        # Signed max magnitude element per 32-block
        idx = np.argmax(np.abs(x), axis=2, keepdims=True)
        mx = np.take_along_axis(x, idx, axis=2)[..., 0]  # [B, nb]
        d_f16 = (mx * np.float32(-0.125)).astype(np.float16)
        d = d_f16.astype(np.float32)
        inv_d = np.where(d != 0.0, 1.0 / d, 0.0).astype(np.float32)
        q = np.clip(np.rint(x * inv_d[:, :, None]), -8.0, 7.0).astype(np.int32) + 8
        q = q.astype(np.uint8)
        # Pack pairs (even low nibble, odd high nibble)
        packed = (q[:, :, 0::2] | (q[:, :, 1::2] << 4)).reshape(r1 - r0, cols // 2)
        out[r0:r1, : cols // 2] = packed
        out[r0:r1, cols // 2 :] = d_f16.view(np.uint8).reshape(r1 - r0, nb * 2)
    return out.tobytes()


def write_safetensors(out_path: Path, specs: list, write_fn):
    """Stream-write a safetensors file without holding all tensors in RAM.
    specs: list of (name, dtype, shape, nbytes)
    write_fn(name, fh): writes exact nbytes to fh.
    """
    header = {}
    off = 0
    for name, dtype, shape, nbytes in specs:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + nbytes]}
        off += nbytes
    hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    # Pad header to 8-byte alignment for clean mmap/pread alignment
    pad = (-len(hbytes)) % 8
    if pad:
        hbytes += b" " * pad
    with open(out_path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hbytes)))
        fh.write(hbytes)
        for name, dtype, shape, nbytes in specs:
            pos0 = fh.tell()
            write_fn(name, fh)
            wrote = fh.tell() - pos0
            assert wrote == nbytes, f"{name}: wrote {wrote} != expected {nbytes}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native-dir", type=Path, default=Path("/mnt/storage/models/Inkling-2-Mini-ingine"))
    ap.add_argument("--kintsugi-gguf", type=Path, default=Path("/mnt/storage/models/Kintsugi-Inkling-2-Mini/Kintsugi-Inkling-2-Mini-12G.gguf"))
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/storage/models/Inkling-2-Mini-atlas-exl3-9g"))
    ap.add_argument("--embed-bf16", action="store_true", help="Keep embed.weight in BF16 (default: Q4_0 saves 665 MB)")
    ap.add_argument("--no-vision", action="store_true", help="Omit vision encoder tensors (model.visual.*, saves ~0.64 GB on disk; run with --no-vision)")
    ap.add_argument("--no-mtp", action="store_true", help="Omit DSpark MTP speculative drafter (mtp.safetensors, saves ~0.64 GB on disk)")
    ap.add_argument("--text-only", action="store_true", help="Shorthand for --no-vision --no-mtp (10.08 GB text-only checkpoint, saves ~1.28 GB on disk)")
    args = ap.parse_args()
    omit_vision = bool(args.no_vision or args.text_only)
    omit_mtp = bool(args.no_mtp or args.text_only)

    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Parsing native checkpoint: {args.native_dir}")
    native = parse_safetensors_dir(args.native_dir)
    print(f"Parsing Kintsugi GGUF:     {args.kintsugi_gguf}")
    gf, g_base, gtensors, escha_map = parse_gguf(args.kintsugi_gguf)
    print(f"Escha map: {len(escha_map)} layers")

    # Separate text vs vision tensors (keep vision in its own shard so text shard is pure ~9.5 GB on disk!)
    skip_suffixes = (
        ".mlp.experts.w13_weight",
        ".mlp.experts.w13_weight_scale",
        ".mlp.experts.w13_weight_amax",
        ".mlp.experts.w13_weight_outer_scale_denominator",
        ".mlp.experts.w2_weight",
        ".mlp.experts.w2_weight_scale",
        ".mlp.experts.w2_weight_amax",
        ".mlp.experts.w2_weight_outer_scale_denominator",
    )

    text_specs = []
    vision_specs = []
    # Map name -> ("native" | "embed_q4" | "exl3_code" | "exl3_r", info)
    sources = {}

    for name in sorted(native.keys()):
        if any(name.endswith(s) for s in skip_suffixes):
            continue
        fpath, foff, nbytes, dtype, shape = native[name]
        if name == "model.llm.embed.weight" and not args.embed_bf16:
            rows, cols = shape
            qbytes = rows * (cols // 2 + (cols // 32) * 2)
            text_specs.append((name, "U8", [rows, cols // 2 + (cols // 32) * 2], qbytes))
            sources[name] = ("embed_q4", (fpath, foff, nbytes, rows, cols))
        elif name.startswith("model.visual."):
            if not omit_vision:
                vision_specs.append((name, dtype, shape, nbytes))
                sources[name] = ("native", (fpath, foff, nbytes))
        else:
            text_specs.append((name, dtype, shape, nbytes))
            sources[name] = ("native", (fpath, foff, nbytes))

    # Add EXL3 experts for all 24 layers
    w13_groups = {2: [], 3: []}
    w2_groups = {2: [], 3: []}
    for l in range(24):
        cfg = escha_map[str(l)]
        b13 = int(cfg["gate_up"])
        b2 = int(cfg["down"])
        p = f"model.llm.layers.{l}.mlp.experts."
        w13_groups[b13].append(p + "w13_weight")
        w2_groups[b2].append(p + "w2_weight")

        for proj, gname, K, N, bits in [
            ("w13", f"blk.{l}.ffn_gate_up_exps", 2304, 2304, b13),
            ("w2", f"blk.{l}.ffn_down_exps", 1152, 2304, b2),
        ]:
            # Code: GGUF [128, K/16, N/16, 16*bits] u16 -> transpose(0,2,1,3) -> [128, N/16, K/16, 32*bits] u8
            kt, nt_ = K // 16, N // 16
            tile_bytes = 32 * bits
            code_name = p + f"{proj}_weight"
            code_nbytes = 128 * nt_ * kt * tile_bytes
            text_specs.append((code_name, "U8", [128, nt_, kt, tile_bytes], code_nbytes))
            sources[code_name] = ("exl3_code", (gname + ".escha_code", kt, nt_, bits))

            # rin [128, K] F16, rout [128, N] F16
            rin_name = p + f"{proj}_rin"
            rout_name = p + f"{proj}_rout"
            text_specs.append((rin_name, "F16", [128, K], 128 * K * 2))
            text_specs.append((rout_name, "F16", [128, N], 128 * N * 2))
            sources[rin_name] = ("exl3_r", (gname + ".escha_rin", 128 * K * 2))
            sources[rout_name] = ("exl3_r", (gname + ".escha_rout", 128 * N * 2))

    def emit_tensor(name, fh):
        kind, info = sources[name]
        if kind == "native":
            fpath, foff, nbytes = info
            with open(fpath, "rb") as sf:
                sf.seek(foff)
                rem = nbytes
                while rem > 0:
                    buf = sf.read(min(rem, 16 * 1024 * 1024))
                    assert buf
                    fh.write(buf)
                    rem -= len(buf)
        elif kind == "embed_q4":
            fpath, foff, nbytes, rows, cols = info
            print(f"  Quantizing {name} [{rows}, {cols}] BF16 -> Q4_0 [{rows}, 1296]...")
            with open(fpath, "rb") as sf:
                sf.seek(foff)
                raw = sf.read(nbytes)
            fh.write(quantize_embed_q4_0(raw, rows, cols))
        elif kind == "exl3_code":
            gname, kt, nt_, bits = info
            dims, tt, off = gtensors[gname]
            expected_dims = (16 * bits, nt_, kt, 128)
            assert dims == expected_dims, f"{gname}: dims {dims} != {expected_dims}"
            tile_bytes = 32 * bits
            nbytes = 128 * kt * nt_ * tile_bytes
            gf.seek(g_base + off)
            raw = gf.read(nbytes)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(128, kt, nt_, tile_bytes)
            # Transpose (E, KT, NT, B) -> (E, NT, KT, B) contiguous
            tr = np.ascontiguousarray(arr.transpose(0, 2, 1, 3))
            fh.write(tr.tobytes())
        elif kind == "exl3_r":
            gname, nbytes = info
            dims, tt, off = gtensors[gname]
            gf.seek(g_base + off)
            raw = gf.read(nbytes)
            assert len(raw) == nbytes
            fh.write(raw)

    text_shard = "model-00001-of-00001.safetensors" if omit_vision else "model-00001-of-00002.safetensors"
    vision_shard = "model-00002-of-00002.safetensors"
    print(f"Writing text shard ({len(text_specs)} tensors) -> {args.out_dir / text_shard}")
    write_safetensors(args.out_dir / text_shard, text_specs, emit_tensor)
    if not omit_vision:
        print(f"Writing vision shard ({len(vision_specs)} tensors) -> {args.out_dir / vision_shard}")
        write_safetensors(args.out_dir / vision_shard, vision_specs, emit_tensor)
    gf.close()

    # Write model.safetensors.index.json
    weight_map = {}
    total_bytes = 0
    for name, _, _, nb in text_specs:
        weight_map[name] = text_shard
        total_bytes += nb
    if not omit_vision:
        for name, _, _, nb in vision_specs:
            weight_map[name] = vision_shard
            total_bytes += nb
    (args.out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2) + "\n"
    )

    # Update hf_quant_config.json with per-tensor exl3_2 / exl3_3 groups
    qcfg = json.loads((args.native_dir / "hf_quant_config.json").read_text())
    new_groups = []
    for grp in qcfg["model"]["parameter_groups"]:
        dt = grp.get("compute_dtype", "")
        if dt in ("nvfp4", "mxfp8", "atlas2", "atlas3", "exl3_2", "exl3_3"):
            continue
        ts = [t for t in grp["tensors"] if not any(t.endswith(s) for s in skip_suffixes)]
        if omit_vision:
            ts = [t for t in ts if not t.startswith("model.visual.")]
        if not args.embed_bf16:
            ts = [t for t in ts if t != "model.llm.embed.weight"]
        if not ts:
            continue
        grp2 = dict(grp)
        grp2["tensors"] = ts
        new_groups.append(grp2)
    if not args.embed_bf16:
        new_groups.append({
            "compute_dtype": "q4_0",
            "output_dtype": "bfloat16",
            "required_activation_quantization": None,
            "tensors": ["model.llm.embed.weight"],
            "weight_block_shape": [1, 32],
            "weight_transform": "q4_0",
        })
    for bits, dt in [(2, "exl3_2"), (3, "exl3_3")]:
        ts = sorted(w13_groups[bits] + w2_groups[bits])
        if ts:
            new_groups.append({
                "compute_dtype": dt,
                "output_dtype": "bfloat16",
                "required_activation_quantization": None,
                "tensors": ts,
                "weight_block_shape": [16, 16],
                "weight_transform": "exl3_mcg_had128",
            })
    qcfg["model"]["parameter_groups"] = new_groups
    qcfg["atlas_exl3"] = {
        "embed_q4": not args.embed_bf16,
        "no_vision": omit_vision,
        "no_mtp": omit_mtp,
    }
    (args.out_dir / "hf_quant_config.json").write_text(json.dumps(qcfg, indent=2) + "\n")

    # Copy metadata & symlink tiktoken (+ optional mtp.safetensors DSpark drafter)
    for item in [
        "chat_template.jinja", "config.json", "generation_config.json",
        "processor_config.json", "special_tokens_map.json", "tokenizer_config.json",
        "tokenizer.json", "LICENSE", "NOTICE",
    ]:
        src = args.native_dir / item
        if src.exists():
            shutil.copy2(src, args.out_dir / item)
    link_items = ["tiktoken"] if omit_mtp else ["mtp.safetensors", "tiktoken"]
    for link_item in link_items:
        src = args.native_dir / link_item
        dst = args.out_dir / link_item
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    text_sz = (args.out_dir / text_shard).stat().st_size
    vis_sz = 0 if omit_vision else (args.out_dir / vision_shard).stat().st_size
    print(f"Done in {time.time() - t0:.1f}s!")
    print(f"  Text shard:   {text_sz / 1e9:.3f} GB ({text_sz / (1024**3):.3f} GiB) [BF16 dense on disk -> ~{(text_sz - 1.146e9)/1e9:.3f} GB / {(text_sz - 1.146e9)/(1024**3):.3f} GiB in VRAM with --dense-fp8!]")
    if not omit_vision:
        print(f"  Vision shard: {vis_sz / 1e9:.3f} GB ({vis_sz / (1024**3):.3f} GiB)")
    else:
        print("  Vision shard: omitted (--no-vision / --text-only)")
    print(f"  DSpark MTP:   {'omitted (--no-mtp / --text-only)' if omit_mtp else 'included (mtp.safetensors)'}")


if __name__ == "__main__":
    main()

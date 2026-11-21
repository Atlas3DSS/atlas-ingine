"""Benchmark moe_up / moe_down (SPEC.md, area `moe`) at M = 1 and M = 8.

Weights must come from DRAM, not the 96 MB L2. Each timed sequence rotates over the layers of a group
(all resident at once) and, within a layer, over 16 routings: for M = 1 the 16 expert sets are disjoint
(all 128 experts, 576 MB NVFP4 / 1.05 GB MXFP8 per layer), for M = 8 they are random (~50 unique experts
per call). Bytes a call reads were last touched >= 15 calls earlier, after >= 540 MB of other traffic.

Rotate over as many layers as memory allows (`--group 0 1 2 3 4 5`): footprints beyond ~1 GB cost extra
(address translation, measured ~+7 us per up+down pair with 3 NVFP4 layers vs 1), which is what the
decode step sees with all 24 layers resident. The default groups fit the 6% per-process memory cap of a
shared GPU (one MXFP8 layer is 1.05 GB).

The calls are captured in a CUDA graph and replayed (as in the decode step); time per call = replay time /
calls, median over repeats. GB/s counts the touched expert bytes (codes + scales of every unique expert of
the call) of the op(s) timed. "up+down" launches both per call; with --pdl 1 they use programmatic
dependent launch (moe_down prefetches its first weights while moe_up drains).

    PYTHONPATH=engine python engine/bench/bench_moe.py [--group 3] [--group 0 1 2] [--group 21] [--pdl 0 1]
"""
from __future__ import annotations

import argparse
import gc
import os
import statistics
from pathlib import Path

import torch

from ingine import ext, model
from ingine import weights as W

CKPT = os.environ.get("INGINE_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models" / "Inkling-2-Mini"))
H, I, E, TOPK = W.HIDDEN, W.INTER, W.EXPERTS, W.TOP_K
PEAK = 1.79e12
SETS = 16


def expert_bytes(ex):
    """(w13 bytes, w2 bytes) of one expert, codes + scales."""
    per = lambda t: t[0].numel() * t.element_size()  # noqa: E731
    return per(ex.w13) + per(ex.w13_scale), per(ex.w2) + per(ex.w2_scale)


def routings(M: int, gen):
    if M == 1:
        perm = torch.randperm(E, generator=gen)
        return [perm[TOPK * i:TOPK * (i + 1)].view(1, TOPK).int().cuda() for i in range(SETS)]
    return [torch.stack([torch.randperm(E, generator=gen)[:TOPK] for _ in range(M)]).int().cuda()
            for _ in range(SETS)]


def load_group(layers):
    group = []
    for l in layers:
        torch.cuda.empty_cache()
        lay = W.load_layer(CKPT, l)
        t = {k: lay.t[k] for k in ("mlp_conv", "mlp_norm")}
        group.append((l, lay.experts, t))
        del lay
        gc.collect()
    return group


def time_graph(fn_calls, repeats):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for fn in fn_calls:
            fn()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(2):
                for fn in fn_calls:
                    fn()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    times = []
    for _ in range(repeats):
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0.record()
        g.replay()
        t1.record()
        t1.synchronize()
        times.append(t0.elapsed_time(t1) * 1e3 / (2 * len(fn_calls)))
    return statistics.median(times), min(times)


def bench_group(mod, group, Ms, pdls, repeats):
    rows = []
    gen = torch.Generator().manual_seed(sum(l for l, _, _ in group))
    for M in Ms:
        per_layer = []
        for l, ex, t in group:
            fmt = 0 if ex.fmt == "nvfp4" else 1
            x = model.rms_norm(torch.randn(M, H, generator=gen).cuda(), t["mlp_norm"]).to(torch.bfloat16)
            per_layer.append((ex, fmt, t["mlp_conv"], x, routings(M, gen)))
        wts = torch.full((M, TOPK), 1.0, device="cuda")
        act = torch.empty(M, TOPK, I, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(M, H, device="cuda")
        st = torch.zeros(3, H, device="cuda")
        no_hist = torch.empty(0, device="cuda")
        # calls ordered set-major, layer-minor: consecutive calls hit different layers
        order = [(ex, fmt, cw, x, idxs[i]) for i in range(SETS) for ex, fmt, cw, x, idxs in per_layer]
        uniq = statistics.mean(int(c[4].unique().numel()) for c in order)
        b13, b2 = expert_bytes(group[0][1])
        for pdl in pdls:
            def up(c, pdl=pdl):
                ex, fmt, _, x, i = c
                mod.moe_up(x, i, ex.w13, ex.w13_scale, ex.w13_factor, fmt, act, pdl=bool(pdl))

            def down(c, pdl=pdl):
                ex, fmt, cw, _, i = c
                mod.moe_down(act, i, wts, ex.w2, ex.w2_scale, ex.w2_factor, fmt, cw, st, out, no_hist, pdl=bool(pdl))

            def both(c):
                up(c)
                down(c)

            for name, fn, nbytes in (("moe_up", up, b13), ("moe_down", down, b2), ("up+down", both, b13 + b2)):
                med, best = time_graph([lambda c=c, fn=fn: fn(c) for c in order], repeats)
                gbs = uniq * nbytes / (med * 1e-6) / 1e9
                rows.append((",".join(str(l) for l, _, _ in group), group[0][1].fmt, M, pdl, name, med, best, uniq,
                             gbs, gbs * 1e9 / PEAK))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, nargs="+", action="append",
                    help="layers resident together, calls rotate over them (repeatable)")
    ap.add_argument("--M", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--pdl", type=int, nargs="+", default=[0, 1], help="programmatic dependent launch off/on")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--mem-fraction", type=float, default=0.06)
    args = ap.parse_args()
    groups = args.group or [[3], [0, 1, 2], [21]]
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    mod = ext.load("moe")
    print(f"{'layers':>8} {'fmt':>6} {'M':>2} {'pdl':>3} {'op':>9} {'median us':>10} {'min us':>8} {'experts':>8} "
          f"{'GB/s':>7} {'of 1.79TB/s':>11}")
    for layers in groups:
        group = load_group(layers)
        for r in bench_group(mod, group, args.M, args.pdl, args.repeats):
            print(f"{r[0]:>8} {r[1]:>6} {r[2]:>2} {r[3]:>3} {r[4]:>9} {r[5]:>10.1f} {r[6]:>8.1f} {r[7]:>8.1f} "
                  f"{r[8]:>7.0f} {100 * r[9]:>10.1f}%", flush=True)
        del group
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

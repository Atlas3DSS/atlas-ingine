"""attn_decode timing: a sliding layer (full 1024-key ring) and a full layer at 0 / 8K / 32K / 128K context.
Each measurement replays a CUDA graph of back-to-back calls that rotate over enough distinct caches that
L2 (96 MB) cannot serve the K/V (a single 512 MB cache read in order also defeats it); reports the
median per-call time over replays and the achieved bandwidth on the K/V bytes the call must read. As a
ceiling under the current GPU conditions (the GPU may be shared), it first measures torch.sum over a
256 MB BF16 tensor the same way.

    PYTHONPATH=engine python engine/bench/bench_attn.py [--reps 30] [--m 1]
"""
from __future__ import annotations

import argparse
import statistics

import torch

torch.cuda.set_per_process_memory_fraction(0.06)

from ingine import ext  # noqa: E402

HQ, HKV, D, WIN = 24, 8, 128, 1024
KV_BYTES_PER_POS = HKV * D * 2 * 2


def layer_tensors(full: bool, M: int, g):
    R = 8664 if full else 8600
    proj = (torch.randn(M, R, device="cuda", generator=g) * 0.5).bfloat16().float()
    q_scale = torch.rand(HQ, D, device="cuda", generator=g) + 0.5
    r_scale = torch.rand(HQ, 16, device="cuda", generator=g) + 0.5
    rel_proj = (torch.randn(16, 1024, device="cuda", generator=g) * 0.3).bfloat16().float()
    conv = torch.randn(*((4, 1024) if full else (1024, 4)), device="cuda", generator=g) * 0.5
    return proj, q_scale, r_scale, rel_proj, conv, conv.clone()


def reference_read_bw(reps: int) -> float:
    n = 64 * 2**20  # two 128 MB halves, alternated
    x = torch.empty(2, n, device="cuda", dtype=torch.bfloat16).normal_()
    out = torch.empty((), device="cuda")
    graph = torch.cuda.CUDAGraph()
    torch.sum(x[0], 0, dtype=torch.float32, out=out)
    with torch.cuda.graph(graph):
        for i in range(8):
            torch.sum(x[i % 2], 0, dtype=torch.float32, out=out)
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / 8)
    del x, graph
    torch.cuda.empty_cache()
    return 2 * n / statistics.median(times) / 1e3


def bench(full: bool, ctx: int, n_caches: int, reps: int, M: int = 1):
    g = torch.Generator(device="cuda").manual_seed(0)
    rows = ctx + M if full else WIN
    params = layer_tensors(full, M, g)
    caches = []
    for _ in range(n_caches):
        k = torch.empty(rows, HKV, D, device="cuda", dtype=torch.bfloat16).normal_(0.0, 0.1, generator=g)
        v = torch.empty(rows, HKV, D, device="cuda", dtype=torch.bfloat16).normal_(0.0, 1.0, generator=g)
        if full:
            kpos = torch.empty(0, dtype=torch.long, device="cuda")
        else:  # full ring: slot s holds the latest position < ctx congruent to s
            s = torch.arange(WIN, device="cuda")
            kpos = ctx - 1 - torch.remainder(ctx - 1 - s, WIN)
        ck = torch.zeros(3, 1024, device="cuda")
        caches.append((ck, ck.clone(), k, v, kpos))
    pos = torch.tensor([ctx], device="cuda")
    ws = torch.zeros(ext.load("attn").attn_workspace_bytes(M), dtype=torch.uint8, device="cuda")
    out = torch.empty(M, HQ * D, device="cuda", dtype=torch.bfloat16)
    mod = ext.load("attn")
    window = 0 if full else WIN
    calls = max(n_caches, 24)

    def run():
        for i in range(calls):
            mod.attn_decode(params[0], *params[1:], *caches[i % n_caches], pos, window, full, ws, out)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()  # warm-up (also builds nothing: the extension is loaded)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(3):
        graph.replay()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / calls)
    us = statistics.median(times)
    kv = (ctx if full else WIN) * KV_BYTES_PER_POS
    del caches, graph
    torch.cuda.empty_cache()
    return us, min(times), kv / us / 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--m", type=int, default=1)
    args = ap.parse_args()
    mod = ext.load("attn")
    print(f"attn_decode M={args.m}, active 16-block clusters (sliding/full): "
          f"{mod.attn_max_active_clusters(args.m, WIN)}/{mod.attn_max_active_clusters(args.m, 0)} (8 needed), "
          f"median of {args.reps} graph replays")
    print(f"reference: torch.sum over 128 MB BF16 reads at {reference_read_bw(args.reps):.0f} GB/s now")
    cases = [("full ctx 0", True, 0, 8), ("sliding ring 1024", False, 50000, 40), ("full 8K", True, 8192, 8),
             ("full 32K", True, 32768, 2), ("full 128K", True, 131072, 1)]
    for name, full, ctx, n in cases:
        us, best, gbs = bench(full, ctx, n, args.reps, args.m)
        kv_mb = (ctx if full else WIN) * KV_BYTES_PER_POS / 2**20
        print(f"{name:18s} K/V {kv_mb:7.1f} MiB  {us:8.2f} us (best {best:8.2f})  {gbs:7.0f} GB/s", flush=True)


if __name__ == "__main__":
    main()

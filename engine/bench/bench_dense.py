"""Benchmarks of the `dense` ops at decode shapes (SPEC.md, area `dense`).

Each op is captured in a CUDA graph as a chain of launches that rotates over enough distinct weight sets
(synthetic, real shapes) to exceed the 96 MB L2, so weights stream from DRAM as in a decode step; the graph
is replayed several times and the median per-launch time is reported with the achieved GB/s of weight
bytes (+ activations for mix). Launch-to-launch gaps inside a graph are included, as in the real step.
Every op is timed with pdl=False and pdl=True (programmatic dependent launch: consecutive launches overlap
the next kernel's weight prefetch with the previous kernel's tail). `--ops chain` times proj -> proj_conv
-> mix -> router -> embed sequences captured in one graph.

Run: PYTHONPATH=engine python engine/bench/bench_dense.py [--M 1] [--ops proj,head,chain,...]
"""
from __future__ import annotations

import argparse
import statistics

import torch

from ingine import ext

H = 2304
DRAM_GBPS = 1792.0
L2_BYTES = 96 << 20


def bench(fn_list, reps_bytes, repeats=15, inner=None):
    """fn_list: callables (one per weight set). Returns median µs per call."""
    n = len(fn_list)
    inner = inner or max(1, 2000 // n)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_list:  # warm-up (builds, first-call attribute setup)
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            for f in fn_list:
                f()
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / (inner * n))
    del g
    return statistics.median(times), min(times)


def copies(nbytes):
    return max(2, -(-2 * L2_BYTES // nbytes))  # enough sets for > 2x L2


def report(name, us, best, nbytes):
    gbps = nbytes / (us * 1e-6) / 1e9
    print(f"{name:40s} {us:8.2f} us (best {best:7.2f})  {nbytes / 1e6:8.2f} MB  {gbps:7.1f} GB/s "
          f"({100 * gbps / DRAM_GBPS:5.1f}% of {DRAM_GBPS:.0f})", flush=True)


def both(name, make, nbytes, inner=None):
    """make(pdl) -> list of callables; reports pdl off and on."""
    for pdl in (False, True):
        report(f"{name} pdl={'on' if pdl else 'off'}", *bench(make(pdl), nbytes, inner=inner), nbytes)


def rand_w(rows, cols, kind, mod):
    if kind == "fp8":  # built in 16K-row chunks: no full-size BF16 temporary
        q = torch.empty(rows, cols, device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.empty(rows, device="cuda")
        for a in range(0, rows, 16384):
            b = min(rows, a + 16384)
            q[a:b], s[a:b] = mod.fp8_rows(torch.randn(b - a, cols, device="cuda", dtype=torch.bfloat16).mul_(0.02))
        return q, s
    return torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16).mul_(0.02), torch.empty(0, device="cuda")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--ops", default="proj,proj_conv,router,mix,gates,embed,head,chain")
    args = ap.parse_args()
    torch.cuda.set_per_process_memory_fraction(0.046)  # ~1.45 GB (shared GPU)
    mod = ext.load("dense")
    M = args.M
    ops = args.ops.split(",")
    e = torch.empty(0, device="cuda")
    print(f"M = {M}; GPU shared with other jobs: numbers are medians of {15} graph replays")

    if "proj" in ops:
        for R in (8600, 8664):
            for kind in ("bf16", "fp8"):
                nb = R * H * (2 if kind == "bf16" else 1) + (R * 4 if kind == "fp8" else 0)
                ws = [rand_w(R, H, kind, mod) for _ in range(copies(nb))]
                segs = [(0, 6552, 0), (6552, 7576, 1), (7576, 8600, 2)] + ([(8600, 8632, 1), (8632, 8664, 2)] if R == 8664 else [])
                seg = torch.tensor(segs, dtype=torch.int32, device="cuda")
                xin = torch.randn(3, M, H, device="cuda").to(torch.bfloat16)
                out = torch.empty(M, R, device="cuda")
                both(f"proj R={R} {kind}",
                     lambda p: [lambda w=w, s=s: mod.proj(xin, w, s, seg, out, pdl=p) for w, s in ws], nb)
                del ws

    if "proj_conv" in ops:
        for kind in ("bf16", "fp8"):
            nb = H * 3072 * (2 if kind == "bf16" else 1) + (H * 4 if kind == "fp8" else 0)
            ws = [rand_w(H, 3072, kind, mod) for _ in range(copies(nb))]
            x = torch.randn(M, 3072, device="cuda").to(torch.bfloat16)
            cw = torch.randn(H, 4, device="cuda")
            st = torch.zeros(3, H, device="cuda")
            out = torch.empty(M, H, device="cuda")
            both(f"proj_conv {kind}",
                 lambda p: [lambda w=w, s=s: mod.proj_conv(x, w, s, cw, st, out, e, pdl=p) for w, s in ws], nb)
            del ws

    if "router" in ops:
        for dt in (torch.float32, torch.bfloat16):
            nb = 128 * H * (4 if dt == torch.float32 else 2)
            ws = [torch.randn(128, H, device="cuda").to(dt) for _ in range(copies(nb))]
            bias = torch.randn(128, device="cuda") * 0.01
            x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
            idx = torch.empty(M, 8, dtype=torch.int32, device="cuda")
            wts = torch.empty(M, 8, device="cuda")
            both(f"router {str(dt).split('.')[-1]}",
                 lambda p: [lambda w=w: mod.router(x, w, bias, idx, wts, pdl=p) for w in ws], nb)
            del ws

    if "mix" in ops:
        for nbk, Q, outs in ((0, 1, "bf16"), (1, 1, "both"), (5, 1, "bf16"), (5, 3, "bf16"), (2, 4, "f32")):
            blocks = [torch.randn(M, H, device="cuda") for _ in range(nbk)]
            ptrs = torch.tensor([b.data_ptr() for b in blocks], dtype=torch.int64, device="cuda")
            base, write = torch.randn(M, H, device="cuda"), torch.randn(M, H, device="cuda")
            r = torch.nn.functional.normalize(torch.randn(H, device="cuda"), dim=0)
            aw = torch.ones(1, device="cuda")
            part = torch.empty(M, H, device="cuda")
            q = torch.randn(Q, H, device="cuda")
            nw = torch.ones(Q, H, device="cuda")
            ob = torch.empty(Q, M, H, dtype=torch.bfloat16, device="cuda") if outs != "f32" else e
            of = torch.empty(Q, M, H, device="cuda") if outs != "bf16" else e
            skip = nbk == 0 and Q == 1
            nbytes = (nbk + 3) * M * H * 4 + Q * H * 4 * 2 + Q * M * H * (2 * (ob.numel() > 0) + 4 * (of.numel() > 0))
            both(f"mix blocks={nbk} Q={Q} out={outs}",
                 lambda p: [lambda: mod.mix(ptrs, base, write, r, aw, part, e if skip else q,
                                            nw if ob.numel() else e, ob, of, pdl=p)], nbytes)

    if "gates" in ops:
        for G in (3, 1):
            for fdt in (torch.float32, torch.bfloat16):
                fb = 4 if fdt == torch.float32 else 2
                nb = G * (64 * 2 * H * 2 + 2 * H * 64 * fb)
                sets = []
                for _ in range(copies(nb)):
                    sets.append((torch.randn(G, 64, 2 * H, device="cuda").to(torch.bfloat16),
                                 (torch.randn(G, H, 64, device="cuda") * 0.1).to(fdt),
                                 (torch.randn(G, H, 64, device="cuda") * 0.1).to(fdt)))
                h, msg = torch.randn(G, M, H, device="cuda"), torch.randn(G, M, H, device="cuda")
                nw = torch.ones(G, H, device="cuda")
                ob = torch.empty(G, M, H, dtype=torch.bfloat16, device="cuda")
                of = torch.empty(G, M, H, device="cuda") if G == 1 else e
                both(f"message_gates G={G} fg={str(fdt).split('.')[-1]}",
                     lambda p: [lambda d=d, f=f, g=g: mod.message_gates(h, msg, d, f, g, nw, ob, of, pdl=p)
                                for d, f, g in sets], nb)
                del sets

    if "embed" in ops:
        table = torch.randn(65536, H, device="cuda", dtype=torch.bfloat16)
        tok = torch.randint(0, 65536, (M,), device="cuda")
        enw = torch.ones(H, device="cuda")
        eo = torch.empty(M, H, device="cuda")
        both("embed", lambda p: [lambda: mod.embed(tok, table, enw, eo, pdl=p)], M * H * 2)
        del table

    if "head" in ops:
        V = 200058
        for kind in ("bf16", "fp8"):
            w, s = rand_w(V, H, kind, mod)
            nb = V * H * (2 if kind == "bf16" else 1) + (V * 4 if kind == "fp8" else 0)
            x = torch.randn(M, H, device="cuda").to(torch.bfloat16)
            out = torch.empty(M, V, device="cuda")
            both(f"head {kind}", lambda p: [lambda: mod.head(x, w, s, out, pdl=p)], nb, inner=50)
            del w, s
            torch.cuda.empty_cache()

    if "chain" in ops:
        # proj (full layer, FP8) -> proj_conv (FP8) -> mix (5 blocks, 3 queries) -> router (BF16) -> embed,
        # rotated over enough weight sets to stream from DRAM; reported per chain of 5 kernels
        R = 8664
        nb1 = R * H + R * 4 + H * 3072 + H * 4 + 128 * H * 2
        sets = [(rand_w(R, H, "fp8", mod), rand_w(H, 3072, "fp8", mod),
                 torch.randn(128, H, device="cuda").to(torch.bfloat16)) for _ in range(copies(nb1))]
        segs = torch.tensor([(0, 6552, 0), (6552, 7576, 1), (7576, 8600, 2), (8600, 8632, 1), (8632, 8664, 2)],
                            dtype=torch.int32, device="cuda")
        xin = torch.randn(3, M, H, device="cuda").to(torch.bfloat16)
        po = torch.empty(M, R, device="cuda")
        x = torch.randn(M, 3072, device="cuda").to(torch.bfloat16)
        cw = torch.randn(H, 4, device="cuda")
        st = torch.zeros(3, H, device="cuda")
        co = torch.empty(M, H, device="cuda")
        blocks = [torch.randn(M, H, device="cuda") for _ in range(5)]
        ptrs = torch.tensor([b.data_ptr() for b in blocks], dtype=torch.int64, device="cuda")
        r = torch.nn.functional.normalize(torch.randn(H, device="cuda"), dim=0)
        aw = torch.ones(1, device="cuda")
        part = torch.empty(M, H, device="cuda")
        q3, nw3 = torch.randn(3, H, device="cuda"), torch.ones(3, H, device="cuda")
        ob = torch.empty(3, M, H, dtype=torch.bfloat16, device="cuda")
        bias = torch.randn(128, device="cuda") * 0.01
        idx = torch.empty(M, 8, dtype=torch.int32, device="cuda")
        wts = torch.empty(M, 8, device="cuda")
        table = torch.randn(65536, H, device="cuda", dtype=torch.bfloat16)
        tok = torch.randint(0, 65536, (M,), device="cuda")
        enw = torch.ones(H, device="cuda")
        eo = torch.empty(M, H, device="cuda")

        def chain(p, wp, wc, wr):
            def f():
                mod.proj(xin, wp[0], wp[1], segs, po, pdl=p)
                mod.proj_conv(x, wc[0], wc[1], cw, st, co, e, pdl=p)
                mod.mix(ptrs, co, blocks[0], r, aw, part, q3, nw3, ob, e, pdl=p)
                mod.router(ob[0], wr, bias, idx, wts, pdl=p)
                mod.embed(tok, table, enw, eo, pdl=p)
            return f

        res = {}
        for p in (False, True):
            us, best = bench([chain(p, *s) for s in sets], nb1, inner=max(1, 400 // len(sets)))
            res[p] = us
            report(f"chain proj>proj_conv>mix>router>embed pdl={'on' if p else 'off'}", us, best, nb1)
        print(f"PDL saves {res[False] - res[True]:.2f} us per 5-kernel chain "
              f"({(res[False] - res[True]) / 5:.2f} us per kernel boundary)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end ingine speed: prefill tokens/s and greedy decode tokens/s for one request.

    PYTHONPATH=engine tools/.venv/bin/python engine/bench/bench_e2e.py --backend fast --tokens 512
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from ingine.engine import Engine, Sampling, Stats

MODEL_DIR = os.environ.get("INGINE_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models" / "Inkling-2-Mini"))

PROMPTS = {
    "chat": "Explain how a hash map handles collisions, with a short Python example.",
    "code": "Write a Python module implementing an LRU cache with O(1) get/put, type hints and unit tests.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--backend", default="fast", choices=("torch", "fast"))
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--ablation")
    ap.add_argument("--max-ctx", type=int, default=32768)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--dense-fp8", action="store_true")
    ap.add_argument("--kv-fp8", action="store_true", help="FP8 full-attention KV caches")
    ap.add_argument("--spec", action="store_true", help="DSpark speculative decoding")
    ap.add_argument("--spec-threshold", type=float, default=0.8,
                    help="verify the leading drafts whose draft probability exceeds this (0 = always all 7)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--long-prompt", type=int, default=0, help="also time prefill of a synthetic prompt of N tokens")
    args = ap.parse_args()

    t = time.time()
    eng = Engine(args.model_dir, max_ctx=args.max_ctx, ablation=args.ablation, backend=args.backend,
                 graph=not args.no_graph, dense_fp8=args.dense_fp8, spec=args.spec,
                 spec_threshold=args.spec_threshold, kv_fp8=args.kv_fp8)
    print(f"load+capture {time.time() - t:.1f}s, GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    stop = {eng.tokenizer.convert_tokens_to_ids("<|content_model_end_sampling|>")}
    results = {}
    for name, prompt in PROMPTS.items():
        text = eng.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                 reasoning_effort=args.effort)
        ids = eng.tokenizer(text, add_special_tokens=False)["input_ids"]
        rates = []
        for r in range(args.runs):
            eng.tokens_in_state = []  # force a full prefill each run
            st = Stats()
            out = list(eng.generate(ids, args.tokens, Sampling(temperature=args.temperature, seed=r),
                                    set() if r else stop, stats=st))
            rates.append((st.generated_tokens - 1) / st.decode_seconds)
            if r == 0:
                print(f"--- {name}: {eng.tokenizer.decode(out[:120])!r}")
        results[name] = {"decode_tok_s": statistics.median(rates), "runs": rates, "spec": st.extra}
        spec = ""
        if st.extra.get("rounds"):
            spec = f", {(st.extra['accepted'] + st.extra['rounds']) / st.extra['rounds']:.2f} tokens/verify (last run)"
        print(f"{name}: decode {statistics.median(rates):.1f} tok/s (runs {', '.join(f'{x:.1f}' for x in rates)}){spec}",
              flush=True)
    if args.long_prompt:
        ids = (eng.tokenizer(open(__file__).read(), add_special_tokens=False)["input_ids"] * 1000)[: args.long_prompt]
        eng.tokens_in_state = []
        torch.cuda.synchronize()
        t = time.time()
        eng._prefill(ids)
        torch.cuda.synchronize()
        dt = time.time() - t
        results["prefill"] = {"tokens": len(ids), "tok_s": len(ids) / dt}
        print(f"prefill {len(ids)} tokens: {len(ids) / dt:.0f} tok/s", flush=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()

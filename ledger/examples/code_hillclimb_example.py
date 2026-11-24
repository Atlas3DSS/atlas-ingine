#!/usr/bin/env python3
"""Minimal runnable example: host-verified Python code hillclimb updating a SQLite VerifiedLedger.

Usage:
  # Offline dry-run (no server needed; verifies ledger admission, failure->repair linking, and rendering):
  python3 ledger/examples/code_hillclimb_example.py --dry-run

  # Live against atlas-ingine or atlas-engine:
  python3 ledger/examples/code_hillclimb_example.py --base-url http://127.0.0.1:8003/v1 --model inkling-2-mini
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verified_ledger import VerifiedLedger, VerifyResult, call_openai_chat, run_verified_loop

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python(text: str) -> str:
    m = _CODE_FENCE_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def verify_top_k_sums(candidate: str) -> VerifyResult:
    """Host verifier: checks exact correctness + throughput (higher score = faster)."""
    code = extract_python(candidate)
    ns: dict = {}
    try:
        exec(compile(code, "<candidate>", "exec"), ns, ns)
    except Exception as exc:
        return VerifyResult(ok=False, error_signature=f"{type(exc).__name__}: {exc}", error_detail=str(exc))

    fn = ns.get("top_k_window_sums")
    if not callable(fn):
        return VerifyResult(
            ok=False,
            error_signature="MissingFunction: top_k_window_sums",
            error_detail="Define top_k_window_sums(nums: list[int], w: int, k: int) -> list[int].",
        )

    cases = [
        ([1, 2, 3, 4, 5], 2, 2, [9, 7]),
        ([5, -1, 3, 2, -4, 6], 3, 3, [7, 4, 4]),
        ([10], 1, 1, [10]),
    ]
    try:
        for nums, w, k, expected in cases:
            got = fn(nums, w, k)
            if got != expected:
                return VerifyResult(
                    ok=False,
                    error_signature="WrongAnswer",
                    error_detail=f"Input ({nums}, w={w}, k={k}): expected {expected}, got {got}",
                )
        big = list(range(20000))
        t0 = time.perf_counter()
        res = fn(big, 64, 5)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if len(res) != 5 or res[0] != sum(big[-64:]):
            return VerifyResult(ok=False, error_signature="LargeCaseMismatch", error_detail=f"Bad large output: {res}")
    except Exception as exc:
        return VerifyResult(ok=False, error_signature=f"{type(exc).__name__}: {exc}", error_detail=str(exc))

    score = 1000.0 / max(dt_ms, 0.05)
    return VerifyResult(
        ok=True,
        score=score,
        summary=f"All tests passed in {dt_ms:.2f} ms (use O(n) sliding window + heapq.nlargest).",
        tags=["sliding_window", "python", "topk"],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="")
    ap.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tmp_dir = None
    if args.db:
        db_path = Path(args.db)
    else:
        tmp_dir = tempfile.TemporaryDirectory(prefix="atlas_ledger_example_")
        db_path = Path(tmp_dir.name) / "example.sqlite"

    task_text = (
        "Write a fast Python function `top_k_window_sums(nums: list[int], w: int, k: int) -> list[int]` "
        "returning the k largest contiguous length-w window sums in descending order. Return only a Python code block."
    )

    with VerifiedLedger(db_path, env_version="py3-v1") as ledger:
        ledger.set_fact("runtime", f"Python {sys.version.split()[0]} stdlib only (no numpy)", category="environment")

        if args.dry_run:
            # Simulates iter 1 failing (wrong function name) -> iter 2 passing (O(n) sliding window)
            attempts = [
                "```python\ndef wrong_name(nums, w, k):\n    return []\n```",
                (
                    "```python\nimport heapq\n"
                    "def top_k_window_sums(nums, w, k):\n"
                    "    cur = sum(nums[:w])\n"
                    "    sums = [cur]\n"
                    "    for i in range(w, len(nums)):\n"
                    "        cur += nums[i] - nums[i - w]\n"
                    "        sums.append(cur)\n"
                    "    return heapq.nlargest(k, sums)\n```"
                ),
            ]
            idx = {"i": 0}

            def gen(_messages):
                out = attempts[min(idx["i"], len(attempts) - 1)]
                idx["i"] += 1
                return out, "stop"
        else:
            def gen(messages):
                return call_openai_chat(
                    base_url=args.base_url,
                    model=args.model,
                    messages=messages,
                    max_tokens=8192,
                    temperature=0.2,
                    reasoning_effort="low",
                )

        summary = run_verified_loop(
            ledger=ledger,
            task_id="top_k_window_sums",
            task_text=task_text,
            verifier=verify_top_k_sums,
            generate=gen,
            max_iters=2 if args.dry_run else 4,
            stop_on_first_pass=args.dry_run,
        )
        print(f"solved={summary.solved} iters={summary.iters_used} best_score={summary.best_score:.2f}")
        print("ledger_stats=", ledger.stats())
        print("\n--- Rendered ledger context for a follow-up task ---")
        print(ledger.render_context("Optimize sliding window top-k sums in Python"))
        if not summary.solved:
            return 1

    if tmp_dir is not None:
        tmp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

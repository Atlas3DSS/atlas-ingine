#!/usr/bin/env python3
"""End-to-end verification of `atlas-6g-homology` (<=6 GB VRAM, <=16 GB RAM):
1) Identity / who-you-are coherence probe
2) VerifiedLedger memory-recovery probe (facts + drift recovery + JSON mode)
3) Code generation coherence probe (parsed & executed)

Usage:
  python3 tools/verify_atlas_6g.py --base-url http://127.0.0.1:8016/v1 --model inkling-2-mini
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ledger.verified_ledger import VerifiedLedger, VerifyResult, call_openai_chat, run_verified_loop


def extract_python_block(text: str) -> str:
    if "```python" in text:
        part = text.split("```python", 1)[1]
        return part.split("```", 1)[0].strip()
    if "```" in text:
        part = text.split("```", 1)[1]
        return part.split("```", 1)[0].strip()
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8016/v1")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--effort", default="low")
    args = ap.parse_args()

    results: dict[str, object] = {}

    # 1. Identity & who-you-are coherence probe
    t0 = time.time()
    id_reply, id_finish = call_openai_chat(
        base_url=args.base_url,
        model=args.model,
        messages=[
            {"role": "system", "content": "You are Inkling-2-Mini running in Atlas 6 GB homology-offload mode. Be clear, direct, and accurate."},
            {"role": "user", "content": "Who are you, what are your strengths, and how do you handle long-term factual recall accurately? Answer in 3 concise sentences."},
        ],
        max_tokens=1024,
        temperature=0.0,
        reasoning_effort=args.effort,
    )
    id_ok = len(id_reply.strip()) >= 40 and ("inkling" in id_reply.lower() or "atlas" in id_reply.lower())
    results["identity_probe"] = {"ok": id_ok, "finish": id_finish, "seconds": round(time.time() - t0, 2), "reply": id_reply.strip()[:400]}
    print(f"[1/3 identity] ok={id_ok} ({time.time() - t0:.1f}s): {id_reply.strip()[:220]}", flush=True)

    # 2. VerifiedLedger memory recovery probe (with JSON output + environment drift)
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "atlas_6g_identity.sqlite"
        with VerifiedLedger(str(db), env_version="v1", max_context_chars=2500) as ledger:
            ledger.set_fact("agent_callsign", "Atlas-6G-Homology-Alpha", category="identity")
            ledger.set_fact("owner_rule", "Strictly stay under 6.0 GB VRAM and 16.0 GB system RAM", category="constraints")
            ledger.set_fact("legacy_port", "8099", category="service")
            ledger.bump_env_version("v2", "Service migrated from legacy_port 8099 to active_port 8016 with kv_mode=fp8.")
            ledger.set_fact("agent_callsign", "Atlas-6G-Homology-Alpha", category="identity")
            ledger.set_fact("active_port", "8016", category="service")
            ledger.set_fact("kv_mode", "fp8", category="service")

            ctx = ledger.render_context("Recover identity callsign, active service port, and kv_mode")
            t1 = time.time()
            mem_raw, mem_finish = call_openai_chat(
                base_url=args.base_url,
                model=args.model,
                messages=[
                    {"role": "system", "content": "Use ONLY the verified ledger context below. Output valid JSON with keys: callsign, active_port, kv_mode.\n\n" + ctx},
                    {"role": "user", "content": "Return your verified identity callsign, current active_port (not stale legacy port), and kv_mode as JSON."},
                ],
                max_tokens=1024,
                temperature=0.0,
                reasoning_effort=args.effort,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(mem_raw)
            mem_ok = (
                parsed.get("callsign") == "Atlas-6G-Homology-Alpha"
                and str(parsed.get("active_port")) == "8016"
                and str(parsed.get("kv_mode")).lower() == "fp8"
            )
            results["ledger_recovery"] = {"ok": mem_ok, "finish": mem_finish, "seconds": round(time.time() - t1, 2), "parsed": parsed}
            print(f"[2/3 ledger_recovery] ok={mem_ok} ({time.time() - t1:.1f}s): {parsed}", flush=True)

            # 3. Verified coding loop (coherence + execution verification)
            def code_verifier(candidate: str) -> VerifyResult:
                src = extract_python_block(candidate)
                ns: dict[str, object] = {}
                try:
                    exec(compile(src, "<candidate>", "exec"), ns)
                    fn = ns.get("merge_intervals")
                    assert callable(fn), "missing function merge_intervals"
                    assert fn([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
                    assert fn([[1, 4], [4, 5]]) == [[1, 5]]
                    assert fn([]) == []
                    return VerifyResult(ok=True, score=1.0, summary="merge_intervals passed all 3 unit tests")
                except Exception as exc:
                    return VerifyResult(ok=False, score=0.0, error_signature=type(exc).__name__, error_detail=str(exc))

            t2 = time.time()
            loop_res = run_verified_loop(
                ledger=ledger,
                task_id="merge_intervals",
                task_text="Write a clean Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges all overlapping intervals and returns them sorted by start.",
                verifier=code_verifier,
                generate=lambda msgs: call_openai_chat(
                    base_url=args.base_url,
                    model=args.model,
                    messages=msgs,
                    max_tokens=1536,
                    temperature=0.0,
                    reasoning_effort=args.effort,
                ),
                max_iters=2,
                stop_on_first_pass=True,
            )
            results["verified_code"] = {"ok": loop_res.solved, "iters": loop_res.iters_used, "seconds": round(time.time() - t2, 2)}
            print(f"[3/3 verified_code] ok={loop_res.solved} iters={loop_res.iters_used} ({time.time() - t2:.1f}s)", flush=True)

    all_ok = all(bool(v["ok"]) for v in results.values())
    print(json.dumps({"all_ok": all_ok, **results}, indent=2))
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

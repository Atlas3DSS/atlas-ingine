#!/usr/bin/env python3
"""Replay frozen #824 cache traces against a live native `ingine-serve --engine` server.

Measures actual GPU TTFT, OpenAI `usage.prompt_tokens_details.cached_tokens`, prompt/completion
token counts, and server-log model prefill / reuse / restored_bytes / CPU render metrics matched
by exact `chatcmpl-...` request ID.

Supports matched pre-optimization baseline runs for Stage 4 (#827):
  - `b1-cold`:    B=1 (`--max-batch 1 --park-mb 0`) with a cache-flush turn between trace turns
                  so every trace turn measures true cold GPU prefill (cached_tokens == 0).
  - `b1-warm`:    B=1 (`--max-batch 1`) native sequential replay (measures existing Extend /
                  Checkpoint / Parked reuse on GPU).
  - `batch-cold`: B>1 (`--max-batch N`, e.g. 4) sequential/interleaved replay against the
                  pre-optimization BatchGenerator (or post-Stage-2 `batch-warm` for #827).

Usage:
  python3 tools/cache_replay_baseline.py \
      --traces-dir results/cache_stage1_traces \
      --base-url http://127.0.0.1:18391 \
      --model inkling-2-mini \
      --mode b1-warm \
      --server-log /tmp/ingine_serve_b1.log \
      --output results/cache_stage1_gpu_b1_warm.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


LOG_PREFILL_RE = re.compile(
    r"(chatcmpl-[0-9a-f]+):\s+(?:slot=(\d+)\s+)?reuse=(\S+)\s+reused=(\d+)\s+"
    r"recomputed=(\d+)\s+restored_bytes=(\S+)\s+prefill=([0-9.]+)s"
)
LOG_DONE_RE = re.compile(
    r"(chatcmpl-[0-9a-f]+):\s+prompt=(\d+)\s+cached=(\d+)\s+recomputed=(\d+)\s+"
    r"reuse=(\S+)\s+restored_bytes=(\S+)\s+render_reused=(\d+)\s+render_ms=([0-9.]+)\s+"
    r"completion=(\d+)\s+finish=(\S+)\s+queued=([0-9.]+)s\s+ttft=(\S+)\s+decode=(.+)$"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_and_load_traces(traces_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = traces_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded: list[dict[str, Any]] = []
    for entry in manifest["traces"]:
        tpath = traces_dir / entry["file"]
        doc = json.loads(tpath.read_text(encoding="utf-8"))
        # Recompute canonical compact JSON hash over `turns` (matches nlohmann::ordered_json::dump()).
        canonical = json.dumps(doc["turns"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        actual_sha = sha256_bytes(canonical)
        expected_sha = entry["trace_sha256"]
        if actual_sha != expected_sha or doc.get("trace_sha256") != expected_sha:
            raise ValueError(
                f"Trace SHA-256 mismatch for {tpath}: expected {expected_sha}, got {actual_sha}"
            )
        loaded.append(doc)
    return manifest, loaded


def post_chat(
    base_url: str, payload: dict[str, Any], stream: bool, timeout_s: float = 300.0
) -> dict[str, Any]:
    body = dict(payload)
    body["stream"] = stream
    if stream:
        body["stream_options"] = {"include_usage": True}
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft_s: float | None = None
    req_id: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    chunks = 0

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        if not stream:
            data = json.loads(resp.read().decode("utf-8"))
            total_s = time.perf_counter() - t0
            req_id = data.get("id")
            choices = data.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason")
            usage = data.get("usage")
            return {
                "id": req_id,
                "finish_reason": finish_reason,
                "usage": usage,
                "ttft_ms": None,
                "total_ms": round(total_s * 1000.0, 2),
                "stream": False,
            }

        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            ev = json.loads(data_str)
            if req_id is None and ev.get("id"):
                req_id = ev["id"]
            if ev.get("usage") is not None:
                usage = ev["usage"]
            choices = ev.get("choices") or []
            if choices:
                ch = choices[0]
                delta = ch.get("delta") or {}
                has_token_delta = bool(
                    delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")
                )
                if ttft_s is None and has_token_delta:
                    ttft_s = time.perf_counter() - t0
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                chunks += 1

    total_s = time.perf_counter() - t0
    if ttft_s is None:
        ttft_s = total_s
    return {
        "id": req_id,
        "finish_reason": finish_reason,
        "usage": usage,
        "ttft_ms": round(ttft_s * 1000.0, 2),
        "total_ms": round(total_s * 1000.0, 2),
        "chunks": chunks,
        "stream": True,
    }


def parse_server_log(log_path: Path | None) -> dict[str, dict[str, Any]]:
    if not log_path or not log_path.exists():
        return {}
    by_id: dict[str, dict[str, Any]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOG_PREFILL_RE.search(line)
        if m:
            cid = m.group(1)
            rec = by_id.setdefault(cid, {})
            rec["slot"] = int(m.group(2)) if m.group(2) is not None else None
            rec["reuse"] = m.group(3)
            rec["reused_tokens"] = int(m.group(4))
            rec["recomputed_tokens"] = int(m.group(5))
            rec["restored_bytes"] = None if m.group(6) == "n/a" else int(m.group(6))
            rec["model_prefill_ms"] = round(float(m.group(7)) * 1000.0, 2)
        m2 = LOG_DONE_RE.search(line)
        if m2:
            cid = m2.group(1)
            rec = by_id.setdefault(cid, {})
            rec["log_prompt_tokens"] = int(m2.group(2))
            rec["log_cached_tokens"] = int(m2.group(3))
            rec["log_recomputed_tokens"] = int(m2.group(4))
            rec["reuse"] = m2.group(5)
            rec["restored_bytes"] = None if m2.group(6) == "n/a" else int(m2.group(6))
            rec["cpu_render_reused_tokens"] = int(m2.group(7))
            rec["cpu_render_ms"] = float(m2.group(8))
            rec["log_completion_tokens"] = int(m2.group(9))
            rec["log_finish_reason"] = m2.group(10)
            rec["queued_ms"] = round(float(m2.group(11)) * 1000.0, 2)
    return by_id


def flush_b1_state(base_url: str, model: str, nonce: int) -> None:
    """Issue a disjoint 1-token prompt to evict live/checkpoint state in B=1 --park-mb 0 cold mode."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"__CACHE_FLUSH_NONCE_{nonce}__"},
            {"role": "user", "content": f"flush {nonce}"},
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "enable_thinking": False,
    }
    post_chat(base_url, payload, stream=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay frozen #824 cache traces against native ingine-serve.")
    ap.add_argument("--traces-dir", required=True, type=Path)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["b1-cold", "b1-warm", "batch-cold", "batch-warm"])
    ap.add_argument("--server-log", type=Path, default=None)
    ap.add_argument("--git-rev", default="")
    ap.add_argument("--gpu-name", default="")
    ap.add_argument("--serve-flags", default="")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--non-streaming", action="store_true")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    manifest, traces = verify_and_load_traces(args.traces_dir)

    # Probe /health to record active server config (max_batch, max_queue).
    with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=10) as r:
        health = json.loads(r.read().decode("utf-8"))

    stream = not args.non_streaming
    trace_results: list[dict[str, Any]] = []
    flush_nonce = 0

    for doc in traces:
        if args.mode in ("b1-cold", "b1-warm"):
            flush_nonce += 1
            flush_b1_state(args.base_url, args.model, flush_nonce)

        rows: list[dict[str, Any]] = []
        for turn in doc["turns"]:
            if args.mode == "b1-cold":
                flush_nonce += 1
                flush_b1_state(args.base_url, args.model, flush_nonce)
            req_payload = dict(turn["request"])
            req_payload["model"] = args.model
            req_payload["max_tokens"] = args.max_tokens
            res = post_chat(args.base_url, req_payload, stream=stream)
            usage = res.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            prompt_tok = int(usage.get("prompt_tokens", 0))
            cached_tok = int(details.get("cached_tokens", 0))
            comp_tok = int(usage.get("completion_tokens", 0))
            rows.append(
                {
                    "turn": turn["turn"],
                    "label": turn["label"],
                    "lane": turn["lane"],
                    "request_id": res["id"],
                    "prompt_tokens": prompt_tok,
                    "cached_tokens": cached_tok,
                    "recomputed_tokens": max(0, prompt_tok - cached_tok),
                    "completion_tokens": comp_tok,
                    "finish_reason": res["finish_reason"],
                    "ttft_ms": res["ttft_ms"],
                    "total_ms": res["total_ms"],
                }
            )
        trace_results.append(
            {
                "name": doc["name"],
                "trace_sha256": doc["trace_sha256"],
                "turns": len(rows),
                "per_turn": rows,
            }
        )

    # Give spdlog a moment to flush then join exact server-log prefill/reuse telemetry by request_id.
    time.sleep(0.25)
    log_by_id = parse_server_log(args.server_log)

    for tr in trace_results:
        tot_prompt = tot_cached = tot_recomp = tot_comp = 0
        ttfts: list[float] = []
        prefills: list[float] = []
        reuse_counts: dict[str, int] = {}
        for row in tr["per_turn"]:
            lrec = log_by_id.get(row["request_id"] or "", {})
            row["reuse"] = lrec.get("reuse")
            row["model_prefill_ms"] = lrec.get("model_prefill_ms")
            row["restored_bytes"] = lrec.get("restored_bytes")
            row["cpu_render_reused_tokens"] = lrec.get("cpu_render_reused_tokens")
            row["cpu_render_ms"] = lrec.get("cpu_render_ms")
            tot_prompt += row["prompt_tokens"]
            tot_cached += row["cached_tokens"]
            tot_recomp += row["recomputed_tokens"]
            tot_comp += row["completion_tokens"]
            if row["ttft_ms"] is not None:
                ttfts.append(float(row["ttft_ms"]))
            if row["model_prefill_ms"] is not None:
                prefills.append(float(row["model_prefill_ms"]))
            if row["reuse"]:
                reuse_counts[row["reuse"]] = reuse_counts.get(row["reuse"], 0) + 1

        tr["total_prompt_tokens"] = tot_prompt
        tr["total_cached_tokens"] = tot_cached
        tr["total_recomputed_tokens"] = tot_recomp
        tr["total_completion_tokens"] = tot_comp
        tr["cache_hit_pct"] = round(100.0 * tot_cached / max(1, tot_prompt), 2)
        tr["mean_ttft_ms"] = round(sum(ttfts) / len(ttfts), 2) if ttfts else None
        tr["p95_ttft_ms"] = (
            round(sorted(ttfts)[min(len(ttfts) - 1, int(0.95 * len(ttfts)))], 2) if ttfts else None
        )
        tr["total_model_prefill_ms"] = round(sum(prefills), 2) if prefills else None
        tr["reuse_breakdown"] = reuse_counts
        print(
            f"[{args.mode}] trace={tr['name']:<28s} turns={tr['turns']:2d} "
            f"prompt={tot_prompt:7d} cached={tot_cached:7d} ({tr['cache_hit_pct']:5.1f}%) "
            f"mean_ttft={tr['mean_ttft_ms']}ms prefill_total={tr['total_model_prefill_ms']}ms "
            f"reuse={reuse_counts}"
        )

    report = {
        "schema": "atlas-ingine.cache_gpu_replay_baseline.v1",
        "mode": args.mode,
        "stream": stream,
        "git_rev": args.git_rev,
        "gpu_name": args.gpu_name,
        "serve_flags": args.serve_flags,
        "health": health,
        "pins": {
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "chat_template_sha256": manifest["chat_template_sha256"],
        },
        "traces": trace_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

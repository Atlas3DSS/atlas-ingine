#!/usr/bin/env python3
"""Replay frozen #824 cache traces against a live native `ingine-serve --engine` server.

Strict fail-loud measurement harness for matched pre-optimization baselines (#824) and Stage 4 (#827):
  - Verifies SHA-256 of every frozen trace AND (when `--model-dir` is given) the loaded model directory's
    `tokenizer.json` and `chat_template.jinja` against `manifest.json`.
  - Requires explicit OpenAI `usage` with `prompt_tokens > 0`, `completion_tokens >= 0`, and integer
    `usage.prompt_tokens_details.cached_tokens` in `[0, prompt_tokens]` on every turn.
  - Preserves a missing first token delta as `ttft_ms = None` (unmeasured) — never substitutes total latency.
  - Requires `--server-log` with a complete, unique 1-to-1 join of every `chatcmpl-...` request ID to both
    prefill and completion log lines, verifying per-turn agreement between API and server-log prompt/cached
    token counts and non-null `model_prefill_ms`.
  - In cold controls (`b1-cold`, `batch-cold`), rejects the arm if any turn reports `cached_tokens != 0` or
    `reuse != "none"`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_and_load_traces(
    traces_dir: Path, model_dir: Path | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = traces_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if model_dir is not None:
        tok_path = model_dir / "tokenizer.json"
        tmpl_path = model_dir / "chat_template.jinja"
        if not tok_path.is_file() or not tmpl_path.is_file():
            raise RuntimeError(f"Model dir missing tokenizer.json or chat_template.jinja: {model_dir}")
        got_tok = sha256_file(tok_path)
        got_tmpl = sha256_file(tmpl_path)
        if got_tok != manifest["tokenizer_sha256"]:
            raise ValueError(
                f"tokenizer.json SHA-256 mismatch: model_dir={got_tok}, manifest={manifest['tokenizer_sha256']}"
            )
        if got_tmpl != manifest["chat_template_sha256"]:
            raise ValueError(
                f"chat_template.jinja SHA-256 mismatch: model_dir={got_tmpl}, manifest={manifest['chat_template_sha256']}"
            )

    loaded: list[dict[str, Any]] = []
    for entry in manifest["traces"]:
        tpath = traces_dir / entry["file"]
        doc = json.loads(tpath.read_text(encoding="utf-8"))
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
    return {
        "id": req_id,
        "finish_reason": finish_reason,
        "usage": usage,
        "ttft_ms": round(ttft_s * 1000.0, 2) if ttft_s is not None else None,
        "total_ms": round(total_s * 1000.0, 2),
        "chunks": chunks,
        "stream": True,
    }


def validate_usage(res: dict[str, Any], label: str) -> tuple[str, int, int, int]:
    req_id = res.get("id")
    if not isinstance(req_id, str) or not req_id.startswith("chatcmpl-"):
        raise RuntimeError(f"[{label}] missing or invalid response id: {req_id!r}")
    usage = res.get("usage")
    if not isinstance(usage, dict):
        raise RuntimeError(f"[{label}] response missing required 'usage' object (id={req_id})")
    prompt_tok = usage.get("prompt_tokens")
    comp_tok = usage.get("completion_tokens")
    details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_tok, int) or isinstance(prompt_tok, bool) or prompt_tok <= 0:
        raise RuntimeError(f"[{label}] invalid prompt_tokens={prompt_tok!r} (id={req_id})")
    if not isinstance(comp_tok, int) or isinstance(comp_tok, bool) or comp_tok < 0:
        raise RuntimeError(f"[{label}] invalid completion_tokens={comp_tok!r} (id={req_id})")
    if not isinstance(details, dict):
        raise RuntimeError(f"[{label}] missing usage.prompt_tokens_details (id={req_id})")
    cached_tok = details.get("cached_tokens")
    if (
        not isinstance(cached_tok, int)
        or isinstance(cached_tok, bool)
        or cached_tok < 0
        or cached_tok > prompt_tok
    ):
        raise RuntimeError(
            f"[{label}] invalid cached_tokens={cached_tok!r} for prompt_tokens={prompt_tok} (id={req_id})"
        )
    return req_id, prompt_tok, cached_tok, comp_tok


def parse_server_log(log_path: Path) -> dict[str, dict[str, Any]]:
    if not log_path.is_file():
        raise RuntimeError(f"Server log file missing or unreadable: {log_path}")
    by_id: dict[str, dict[str, Any]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOG_PREFILL_RE.search(line)
        if m:
            cid = m.group(1)
            rec = by_id.setdefault(cid, {})
            if "prefill_seen" in rec:
                raise RuntimeError(f"Duplicate prefill log record for request_id={cid}")
            rec["prefill_seen"] = True
            rec["slot"] = int(m.group(2)) if m.group(2) is not None else None
            rec["prefill_reuse"] = m.group(3)
            rec["prefill_reused_tokens"] = int(m.group(4))
            rec["prefill_recomputed_tokens"] = int(m.group(5))
            rec["prefill_restored_bytes"] = None if m.group(6) == "n/a" else int(m.group(6))
            rec["model_prefill_ms"] = round(float(m.group(7)) * 1000.0, 2)
        m2 = LOG_DONE_RE.search(line)
        if m2:
            cid = m2.group(1)
            rec = by_id.setdefault(cid, {})
            if "done_seen" in rec:
                raise RuntimeError(f"Duplicate completion log record for request_id={cid}")
            rec["done_seen"] = True
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
    res = post_chat(base_url, payload, stream=False)
    validate_usage(res, f"flush_{nonce}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Replay frozen #824 cache traces against native ingine-serve.")
    ap.add_argument("--traces-dir", required=True, type=Path)
    ap.add_argument("--model-dir", type=Path, default=None, help="Verify tokenizer/template SHA-256 against manifest")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["b1-cold", "b1-warm", "batch-cold", "batch-warm"])
    ap.add_argument("--server-log", required=True, type=Path)
    ap.add_argument("--git-rev", default="")
    ap.add_argument("--gpu-name", default="")
    ap.add_argument("--gpu-uuid", default="")
    ap.add_argument("--serve-flags", default="")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--non-streaming", action="store_true")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    manifest, traces = verify_and_load_traces(args.traces_dir, args.model_dir)

    with urllib.request.urlopen(args.base_url.rstrip("/") + "/health", timeout=10) as r:
        health = json.loads(r.read().decode("utf-8"))

    stream = not args.non_streaming
    trace_results: list[dict[str, Any]] = []
    seen_request_ids: set[str] = set()
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
            req_id, prompt_tok, cached_tok, comp_tok = validate_usage(res, f"{doc['name']}:{turn['label']}")
            if req_id in seen_request_ids:
                raise RuntimeError(f"Duplicate response request_id observed across turns: {req_id}")
            seen_request_ids.add(req_id)
            rows.append(
                {
                    "turn": turn["turn"],
                    "label": turn["label"],
                    "lane": turn["lane"],
                    "request_id": req_id,
                    "prompt_tokens": prompt_tok,
                    "cached_tokens": cached_tok,
                    "recomputed_tokens": prompt_tok - cached_tok,
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

    # Allow spdlog flush, then enforce complete, unique, per-turn consistent server-log join.
    time.sleep(0.35)
    log_by_id = parse_server_log(args.server_log)
    is_cold_control = args.mode in ("b1-cold", "batch-cold")

    for tr in trace_results:
        tot_prompt = tot_cached = tot_recomp = tot_comp = 0
        ttfts: list[float] = []
        missing_ttft = 0
        prefills: list[float] = []
        reuse_counts: dict[str, int] = {}
        for row in tr["per_turn"]:
            cid = row["request_id"]
            lrec = log_by_id.get(cid)
            if not lrec or not lrec.get("prefill_seen") or not lrec.get("done_seen"):
                raise RuntimeError(
                    f"[{tr['name']}:{row['label']}] incomplete server log record for {cid}: {lrec!r}"
                )
            if lrec["log_prompt_tokens"] != row["prompt_tokens"]:
                raise RuntimeError(
                    f"[{cid}] prompt_tokens mismatch: api={row['prompt_tokens']} log={lrec['log_prompt_tokens']}"
                )
            if lrec["log_cached_tokens"] != row["cached_tokens"] or lrec["prefill_reused_tokens"] != row["cached_tokens"]:
                raise RuntimeError(
                    f"[{cid}] cached_tokens mismatch: api={row['cached_tokens']} "
                    f"log_done={lrec['log_cached_tokens']} log_prefill={lrec['prefill_reused_tokens']}"
                )
            if lrec["reuse"] != lrec["prefill_reuse"]:
                raise RuntimeError(
                    f"[{cid}] reuse path mismatch: prefill={lrec['prefill_reuse']} done={lrec['reuse']}"
                )
            if is_cold_control and (row["cached_tokens"] != 0 or lrec["reuse"] != "none"):
                raise RuntimeError(
                    f"Cold control violation in {args.mode} at {tr['name']}:{row['label']} ({cid}): "
                    f"cached_tokens={row['cached_tokens']}, reuse={lrec['reuse']}"
                )

            row["reuse"] = lrec["reuse"]
            row["model_prefill_ms"] = lrec["model_prefill_ms"]
            row["restored_bytes"] = lrec["restored_bytes"]
            row["cpu_render_reused_tokens"] = lrec["cpu_render_reused_tokens"]
            row["cpu_render_ms"] = lrec["cpu_render_ms"]
            tot_prompt += row["prompt_tokens"]
            tot_cached += row["cached_tokens"]
            tot_recomp += row["recomputed_tokens"]
            tot_comp += row["completion_tokens"]
            if row["ttft_ms"] is not None:
                ttfts.append(float(row["ttft_ms"]))
            else:
                missing_ttft += 1
            prefills.append(float(row["model_prefill_ms"]))
            reuse_counts[row["reuse"]] = reuse_counts.get(row["reuse"], 0) + 1

        tr["total_prompt_tokens"] = tot_prompt
        tr["total_cached_tokens"] = tot_cached
        tr["total_recomputed_tokens"] = tot_recomp
        tr["total_completion_tokens"] = tot_comp
        tr["cache_hit_pct"] = round(100.0 * tot_cached / max(1, tot_prompt), 2)
        tr["ttft_measured_turns"] = len(ttfts)
        tr["ttft_unmeasured_turns"] = missing_ttft
        tr["mean_ttft_ms"] = round(sum(ttfts) / len(ttfts), 2) if ttfts else None
        tr["p95_ttft_ms"] = (
            round(sorted(ttfts)[min(len(ttfts) - 1, int(0.95 * len(ttfts)))], 2) if ttfts else None
        )
        tr["total_model_prefill_ms"] = round(sum(prefills), 2)
        tr["reuse_breakdown"] = reuse_counts
        print(
            f"[{args.mode}] trace={tr['name']:<28s} turns={tr['turns']:2d} "
            f"prompt={tot_prompt:7d} cached={tot_cached:7d} ({tr['cache_hit_pct']:5.1f}%) "
            f"mean_ttft={tr['mean_ttft_ms']}ms prefill_total={tr['total_model_prefill_ms']}ms "
            f"reuse={reuse_counts}"
        )

    report = {
        "schema": "atlas-ingine.cache_gpu_replay_baseline.v2",
        "mode": args.mode,
        "stream": stream,
        "git_rev": args.git_rev,
        "gpu_name": args.gpu_name,
        "gpu_uuid": args.gpu_uuid,
        "serve_flags": args.serve_flags,
        "health": health,
        "pins": {
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "chat_template_sha256": manifest["chat_template_sha256"],
            "model_dir_verified": args.model_dir is not None,
        },
        "traces": trace_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

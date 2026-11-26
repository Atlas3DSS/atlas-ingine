#!/usr/bin/env python3
"""Stage 4 (#827) independent cache-off vs cache-on HTTP replay & parity acceptance harness.

Extends tools/cache_replay_baseline.py with:
  - Per-turn exact output capture (content + reasoning_content + tool_calls + SHA-256)
    so cache-off vs cache-on output parity is verified on every single frozen turn.
  - p50, p95, and mean TTFT, total prefill ms, sum(reused)/sum(prompt), and reuse vs
    theoretical multi-entry eligible token ceiling from results/cache_stage1_baseline.json.
  - Separate `compare` subcommand that produces a machine-verified paired report comparing
    (b1-cold vs b1-warm) and (batch-cold vs batch-warm) at identical checkpoint, build,
    GPU, batch shape, and flags.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from cache_replay_baseline import (
    flush_b1_state,
    parse_server_log,
    sha256_bytes,
    validate_joined_turn,
    validate_usage,
    verify_and_load_traces,
)


def post_chat_with_output(
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
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_deltas: list[Any] = []

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        if not stream:
            data = json.loads(resp.read().decode("utf-8"))
            total_s = time.perf_counter() - t0
            req_id = data.get("id")
            choices = data.get("choices") or []
            if choices:
                finish_reason = choices[0].get("finish_reason")
                msg = choices[0].get("message") or {}
                if msg.get("content"):
                    content_parts.append(str(msg["content"]))
                if msg.get("reasoning_content"):
                    reasoning_parts.append(str(msg["reasoning_content"]))
                if msg.get("tool_calls"):
                    tool_deltas.append(msg["tool_calls"])
            usage = data.get("usage")
        else:
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
                    c_piece = delta.get("content")
                    r_piece = delta.get("reasoning_content")
                    t_piece = delta.get("tool_calls")
                    if c_piece:
                        content_parts.append(str(c_piece))
                    if r_piece:
                        reasoning_parts.append(str(r_piece))
                    if t_piece:
                        tool_deltas.append(t_piece)
                    if ttft_s is None and (c_piece or r_piece or t_piece):
                        ttft_s = time.perf_counter() - t0
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
                    chunks += 1
            total_s = time.perf_counter() - t0

    output_obj = {
        "content": "".join(content_parts),
        "reasoning_content": "".join(reasoning_parts),
        "tool_calls": tool_deltas,
        "finish_reason": finish_reason,
    }
    output_sha = sha256_bytes(json.dumps(output_obj, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return {
        "id": req_id,
        "finish_reason": finish_reason,
        "usage": usage,
        "ttft_ms": round(ttft_s * 1000.0, 2) if ttft_s is not None else None,
        "total_ms": round(total_s * 1000.0, 2),
        "chunks": chunks,
        "stream": stream,
        "output": output_obj,
        "output_sha256": output_sha,
    }


def cmd_replay(args: argparse.Namespace) -> int:
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
            res = post_chat_with_output(args.base_url, req_payload, stream=stream)
            req_id, prompt_tok, cached_tok, comp_tok = validate_usage(res, f"{doc['name']}:{turn['label']}")
            if req_id in seen_request_ids:
                raise RuntimeError(f"Duplicate response request_id: {req_id}")
            seen_request_ids.add(req_id)
            rows.append({
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
                "output_sha256": res["output_sha256"],
                "output": res["output"],
            })
        trace_results.append({
            "name": doc["name"],
            "trace_sha256": doc["trace_sha256"],
            "turns": len(rows),
            "per_turn": rows,
        })

    time.sleep(0.35)
    log_by_id = parse_server_log(args.server_log)

    all_prompt = all_cached = all_recomp = all_comp = 0
    all_ttfts: list[float] = []
    all_prefill_ms = 0.0

    for tr in trace_results:
        tot_prompt = tot_cached = tot_recomp = tot_comp = 0
        ttfts: list[float] = []
        prefills: list[float] = []
        reuse_counts: dict[str, int] = {}
        for row in tr["per_turn"]:
            cid = row["request_id"]
            lrec = log_by_id.get(cid)
            validate_joined_turn(row, lrec, tr["name"], args.mode)
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
            prefills.append(float(row["model_prefill_ms"]))
            reuse_counts[row["reuse"]] = reuse_counts.get(row["reuse"], 0) + 1

        all_prompt += tot_prompt
        all_cached += tot_cached
        all_recomp += tot_recomp
        all_comp += tot_comp
        all_ttfts.extend(ttfts)
        all_prefill_ms += sum(prefills)

        tr["total_prompt_tokens"] = tot_prompt
        tr["total_cached_tokens"] = tot_cached
        tr["total_recomputed_tokens"] = tot_recomp
        tr["total_completion_tokens"] = tot_comp
        tr["cache_hit_pct"] = round(100.0 * tot_cached / max(1, tot_prompt), 2)
        tr["p50_ttft_ms"] = round(statistics.median(ttfts), 2) if ttfts else None
        tr["mean_ttft_ms"] = round(sum(ttfts) / len(ttfts), 2) if ttfts else None
        tr["p95_ttft_ms"] = round(sorted(ttfts)[min(len(ttfts) - 1, int(0.95 * len(ttfts)))], 2) if ttfts else None
        tr["total_model_prefill_ms"] = round(sum(prefills), 2)
        tr["reuse_breakdown"] = reuse_counts
        print(
            f"[#827 {args.mode}] trace={tr['name']:<28s} turns={tr['turns']:2d} "
            f"prompt={tot_prompt:7d} cached={tot_cached:7d} ({tr['cache_hit_pct']:5.1f}%) "
            f"p50={tr['p50_ttft_ms']}ms p95={tr['p95_ttft_ms']}ms prefill={tr['total_model_prefill_ms']}ms reuse={reuse_counts}",
            flush=True,
        )

    report = {
        "schema": "atlas-ingine.cache_stage4_accept_replay.v1",
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
        "overall": {
            "total_prompt_tokens": all_prompt,
            "total_cached_tokens": all_cached,
            "total_recomputed_tokens": all_recomp,
            "total_completion_tokens": all_comp,
            "cache_hit_pct": round(100.0 * all_cached / max(1, all_prompt), 2),
            "p50_ttft_ms": round(statistics.median(all_ttfts), 2) if all_ttfts else None,
            "mean_ttft_ms": round(sum(all_ttfts) / len(all_ttfts), 2) if all_ttfts else None,
            "p95_ttft_ms": round(sorted(all_ttfts)[min(len(all_ttfts) - 1, int(0.95 * len(all_ttfts)))], 2) if all_ttfts else None,
            "total_model_prefill_ms": round(all_prefill_ms, 2),
        },
        "traces": trace_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


def _normalize_tool_calls(tool_calls: Any) -> Any:
    """Normalize ephemeral per-request tool-call IDs (e.g. call_xxxx) while preserving all semantic fields."""
    norm = copy.deepcopy(tool_calls)
    id_re = re.compile(r"^call_[0-9A-Za-z_-]+$")
    seq = 0

    def walk(obj: Any) -> Any:
        nonlocal seq
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k == "id" and isinstance(v, str) and (id_re.match(v) or v.startswith("call_")):
                    seq += 1
                    out[k] = f"call_NORM_{seq}"
                else:
                    out[k] = walk(v)
            return out
        return obj

    return walk(norm)


def normalize_output_for_parity(output_obj: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": output_obj.get("content", ""),
        "reasoning_content": output_obj.get("reasoning_content", ""),
        "tool_calls": _normalize_tool_calls(output_obj.get("tool_calls", [])),
        "finish_reason": output_obj.get("finish_reason"),
    }


def _output_sha256(output_obj: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(output_obj, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _diff_fields(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    fields = []
    for key in ("content", "reasoning_content", "tool_calls", "finish_reason"):
        if a.get(key) != b.get(key):
            fields.append(key)
    return fields


def compare_pair(cold: dict[str, Any], warm: dict[str, Any], stage1_refs: dict[str, int]) -> dict[str, Any]:
    assert cold["git_rev"] == warm["git_rev"], "git_rev mismatch between cold and warm arms"
    assert cold["gpu_uuid"] == warm["gpu_uuid"], "gpu_uuid mismatch between cold and warm arms"
    cold_by_name = {t["name"]: t for t in cold["traces"]}
    warm_by_name = {t["name"]: t for t in warm["traces"]}
    assert set(cold_by_name) == set(warm_by_name)

    tot_turns = 0
    raw_exact_matches = 0
    id_norm_matches = 0
    tool_id_only_diffs = 0
    unresolved_mismatches: list[dict[str, Any]] = []
    raw_only_tool_id_diffs: list[dict[str, Any]] = []
    per_trace: list[dict[str, Any]] = []
    tot_stage1_ref = 0
    tot_cached = 0
    cold_turn0_ttfts: list[float] = []
    warm_turn0_ttfts: list[float] = []

    is_batch_arm = warm["mode"].startswith("batch-")
    arm_scope = "serial_batch_capacity4_slot0" if is_batch_arm else "serial_b1_engine"

    for name in sorted(cold_by_name):
        ct = cold_by_name[name]
        wt = warm_by_name[name]
        assert len(ct["per_turn"]) == len(wt["per_turn"])
        t_raw_match = 0
        t_norm_match = 0
        for cr, wr in zip(ct["per_turn"], wt["per_turn"]):
            tot_turns += 1
            assert cr["prompt_tokens"] == wr["prompt_tokens"], f"prompt_tokens mismatch on {name}:{cr['label']}"
            if cr["turn"] == 0 and cr["ttft_ms"] is not None and wr["ttft_ms"] is not None:
                cold_turn0_ttfts.append(float(cr["ttft_ms"]))
                warm_turn0_ttfts.append(float(wr["ttft_ms"]))
            raw_same = cr["output_sha256"] == wr["output_sha256"]
            c_norm = normalize_output_for_parity(cr["output"])
            w_norm = normalize_output_for_parity(wr["output"])
            norm_same = _output_sha256(c_norm) == _output_sha256(w_norm)
            if raw_same:
                raw_exact_matches += 1
                t_raw_match += 1
            if norm_same:
                id_norm_matches += 1
                t_norm_match += 1
                if not raw_same:
                    tool_id_only_diffs += 1
                    raw_only_tool_id_diffs.append({
                        "trace": name,
                        "label": cr["label"],
                        "prompt_tokens": cr["prompt_tokens"],
                        "warm_cached_tokens": wr["cached_tokens"],
                        "warm_reuse": wr["reuse"],
                        "diff_kind": "ephemeral_tool_call_id_only",
                    })
            else:
                unresolved_mismatches.append({
                    "trace": name,
                    "label": cr["label"],
                    "prompt_tokens": cr["prompt_tokens"],
                    "warm_cached_tokens": wr["cached_tokens"],
                    "warm_reuse": wr["reuse"],
                    "diff_fields": _diff_fields(c_norm, w_norm),
                    "cause_status": "unresolved",
                    "cold_output": cr["output"],
                    "warm_output": wr["output"],
                })
        s1_ref_tok = stage1_refs.get(name, 0)
        tot_stage1_ref += s1_ref_tok
        tot_cached += wt["total_cached_tokens"]
        per_trace.append({
            "trace": name,
            "turns": len(ct["per_turn"]),
            "raw_exact_output_matches": t_raw_match,
            "id_normalized_output_matches": t_norm_match,
            "prompt_tokens": wt["total_prompt_tokens"],
            "warm_cached_tokens": wt["total_cached_tokens"],
            "stage1_reference_reusable_tokens": s1_ref_tok,
            "reused_over_stage1_reference_pct": round(100.0 * wt["total_cached_tokens"] / max(1, s1_ref_tok), 2) if s1_ref_tok else None,
            "cold_prefill_ms": ct["total_model_prefill_ms"],
            "warm_prefill_ms": wt["total_model_prefill_ms"],
            "prefill_reduction_pct": round(100.0 * (1.0 - wt["total_model_prefill_ms"] / max(0.01, ct["total_model_prefill_ms"])), 2),
            "cold_p50_ttft_ms": ct["p50_ttft_ms"],
            "warm_p50_ttft_ms": wt["p50_ttft_ms"],
            "cold_p95_ttft_ms": ct["p95_ttft_ms"],
            "warm_p95_ttft_ms": wt["p95_ttft_ms"],
            "warm_reuse_breakdown": wt["reuse_breakdown"],
        })

    c_ov, w_ov = cold["overall"], warm["overall"]
    unresolved_field_counts: dict[str, int] = {}
    for m in unresolved_mismatches:
        for f in m["diff_fields"]:
            unresolved_field_counts[f] = unresolved_field_counts.get(f, 0) + 1

    return {
        "cold_mode": cold["mode"],
        "warm_mode": warm["mode"],
        "execution_concurrency": "single_client_serial",
        "arm_scope": arm_scope,
        "git_rev": warm["git_rev"],
        "gpu_name": warm["gpu_name"],
        "gpu_uuid": warm["gpu_uuid"],
        "total_turns": tot_turns,
        "raw_exact_output_matches": raw_exact_matches,
        "raw_exact_output_parity_pct": round(100.0 * raw_exact_matches / max(1, tot_turns), 2),
        "id_normalized_output_matches": id_norm_matches,
        "id_normalized_output_parity_pct": round(100.0 * id_norm_matches / max(1, tot_turns), 2),
        "ephemeral_tool_call_id_only_differences": tool_id_only_diffs,
        "unresolved_output_mismatches_count": len(unresolved_mismatches),
        "unresolved_mismatch_field_counts": unresolved_field_counts,
        "unresolved_mismatch_status": "unresolved" if unresolved_mismatches else "none",
        "raw_only_tool_id_differences": raw_only_tool_id_diffs,
        "unresolved_mismatches": unresolved_mismatches,
        "total_prompt_tokens": w_ov["total_prompt_tokens"],
        "total_warm_cached_tokens": w_ov["total_cached_tokens"],
        "sum_reused_over_sum_prompt_pct": w_ov["cache_hit_pct"],
        "stage1_reference_reusable_tokens": tot_stage1_ref,
        "stage1_reference_note": "Offline Stage 1 multi-entry reference estimate from results/cache_stage1_baseline.json (not a hard upper bound; live prefix caching can reuse additional shared prefix tokens beyond the offline estimate).",
        "reused_over_stage1_reference_pct": round(100.0 * tot_cached / max(1, tot_stage1_ref), 2) if tot_stage1_ref else None,
        "cold_total_prefill_ms": c_ov["total_model_prefill_ms"],
        "warm_total_prefill_ms": w_ov["total_model_prefill_ms"],
        "prefill_reduction_pct": round(100.0 * (1.0 - w_ov["total_model_prefill_ms"] / max(0.01, c_ov["total_model_prefill_ms"])), 2),
        "cold_p50_ttft_ms": c_ov["p50_ttft_ms"],
        "warm_p50_ttft_ms": w_ov["p50_ttft_ms"],
        "cold_p95_ttft_ms": c_ov["p95_ttft_ms"],
        "warm_p95_ttft_ms": w_ov["p95_ttft_ms"],
        "turn0_cold_mean_ttft_ms": round(sum(cold_turn0_ttfts) / max(1, len(cold_turn0_ttfts)), 2),
        "turn0_warm_mean_ttft_ms": round(sum(warm_turn0_ttfts) / max(1, len(warm_turn0_ttfts)), 2),
        "per_trace": per_trace,
    }


def cmd_compare(args: argparse.Namespace) -> int:
    s1 = json.loads(args.stage1_baseline.read_text(encoding="utf-8"))
    stage1_refs = {t["name"]: int(t["theoretical_multi_entry_reusable_tokens"]) for t in s1["traces"]}
    b1_pair = compare_pair(
        json.loads(args.b1_cold.read_text(encoding="utf-8")),
        json.loads(args.b1_warm.read_text(encoding="utf-8")),
        stage1_refs,
    )
    batch_pair = compare_pair(
        json.loads(args.batch_cold.read_text(encoding="utf-8")),
        json.loads(args.batch_warm.read_text(encoding="utf-8")),
        stage1_refs,
    )
    summary = {
        "schema": "atlas-ingine.cache_stage4_accept_summary.v2",
        "execution_concurrency": "single_client_serial",
        "b1_serial_cold_vs_warm": b1_pair,
        "batch_capacity4_serial_slot0_cold_vs_warm": batch_pair,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    keys = (
        "arm_scope",
        "total_turns",
        "raw_exact_output_matches",
        "id_normalized_output_matches",
        "ephemeral_tool_call_id_only_differences",
        "unresolved_output_mismatches_count",
        "unresolved_mismatch_field_counts",
        "total_warm_cached_tokens",
        "total_prompt_tokens",
        "sum_reused_over_sum_prompt_pct",
        "stage1_reference_reusable_tokens",
        "reused_over_stage1_reference_pct",
        "prefill_reduction_pct",
        "cold_p50_ttft_ms",
        "warm_p50_ttft_ms",
        "cold_p95_ttft_ms",
        "warm_p95_ttft_ms",
    )
    print(json.dumps({
        "b1_serial": {k: b1_pair[k] for k in keys},
        "batch_capacity4_serial_slot0": {k: batch_pair[k] for k in keys},
    }, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rep = sub.add_parser("replay")
    p_rep.add_argument("--traces-dir", required=True, type=Path)
    p_rep.add_argument("--model-dir", type=Path, default=None)
    p_rep.add_argument("--base-url", required=True)
    p_rep.add_argument("--model", required=True)
    p_rep.add_argument("--mode", required=True, choices=["b1-cold", "b1-warm", "batch-cold", "batch-warm"])
    p_rep.add_argument("--server-log", required=True, type=Path)
    p_rep.add_argument("--git-rev", default="")
    p_rep.add_argument("--gpu-name", default="")
    p_rep.add_argument("--gpu-uuid", default="")
    p_rep.add_argument("--serve-flags", default="")
    p_rep.add_argument("--max-tokens", type=int, default=16)
    p_rep.add_argument("--non-streaming", action="store_true")
    p_rep.add_argument("--output", required=True, type=Path)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--stage1-baseline", required=True, type=Path)
    p_cmp.add_argument("--b1-cold", required=True, type=Path)
    p_cmp.add_argument("--b1-warm", required=True, type=Path)
    p_cmp.add_argument("--batch-cold", required=True, type=Path)
    p_cmp.add_argument("--batch-warm", required=True, type=Path)
    p_cmp.add_argument("--output", required=True, type=Path)

    args = ap.parse_args()
    return cmd_replay(args) if args.cmd == "replay" else cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())

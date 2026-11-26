#!/usr/bin/env python3
"""Unit and live-SSE integration tests for tools/strix_halo_verify.py (#835 / #859)."""

from __future__ import annotations

import copy
import http.server
import json
import os
import re
import socketserver
import stat
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

import strix_halo_verify as shv  # noqa: E402


class _MockIngineHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen_prefixes: set[str] = set()
    request_log: list[dict[str, Any]] = []
    disable_prefix_cache: bool = False

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": "Qwen3.8-Flash-Next"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length).decode("utf-8"))
        messages = req.get("messages", [])
        max_toks = int(req.get("max_completion_tokens", 8))

        prompt_text = " ".join(m.get("content", "") for m in messages)
        prompt_tokens = max(1, len(prompt_text.split()))
        cached_tokens = 0
        if not self.disable_prefix_cache:
            if len(messages) >= 4:
                base_key = messages[1].get("content", "")
                if base_key in self.seen_prefixes:
                    cached_tokens = max(1, len(base_key.split()))
            elif len(messages) >= 2:
                self.seen_prefixes.add(messages[1].get("content", ""))

        self.request_log.append(
            {
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "max_completion_tokens": max_toks,
                "num_messages": len(messages),
            }
        )

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        time.sleep(0.002)
        n_emit = min(max_toks, 8)
        for i in range(n_emit):
            chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": f"tok{i} "}, "finish_reason": None}],
                "usage": None,
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
            if i + 1 < n_emit:
                time.sleep(0.001)

        final_chunk = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": n_emit,
                "total_tokens": prompt_tokens + n_emit,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
        }
        self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def _make_qualifying_bench() -> dict[str, Any]:
    """Construct a complete qualifying evidence fixture that beats all 11 owner target floors."""
    decode_rates = {
        "code": 105.0,
        "prose": 74.0,
        "edit": 128.0,
        "tool": 92.0,
        "multilang": 71.0,
    }
    decode_runs = {
        name: {
            "prompt_tokens": 96,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 96,
            "completion_tokens": 128,
            "ttft_s": 0.04,
            "decode_s": 127.0 / rate,
            "total_s": 0.04 + 127.0 / rate,
            "decode_tok_s": rate,
            "prefill_tok_s": 2400.0,
        }
        for name, rate in decode_rates.items()
    }
    prefill_runs = {
        "8192": {
            "target_tokens": 8192,
            "prompt_tokens": 8240,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 8240,
            "completion_tokens": 1,
            "ttft_s": 8240.0 / 2100.0,
            "prefill_tok_s": 2100.0,
        },
        "32768": {
            "target_tokens": 32768,
            "prompt_tokens": 32900,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 32900,
            "completion_tokens": 1,
            "ttft_s": 32900.0 / 1950.0,
            "prefill_tok_s": 1950.0,
        },
        "131072": {
            "target_tokens": 131072,
            "prompt_tokens": 131200,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 131200,
            "completion_tokens": 1,
            "ttft_s": 131200.0 / 1850.0,
            "prefill_tok_s": 1850.0,
        },
    }
    agent_runs = [
        {
            "target_ctx_tokens": 32768,
            "turn1_prime": {"prompt_tokens": 32800, "cached_tokens": 0, "completion_tokens": 8, "ttft_s": 16.0, "total_s": 16.2},
            "turn2_warm": {
                "prompt_tokens": 32960,
                "cached_tokens": 32700,
                "completion_tokens": 64,
                "ttft_s": 0.65,
                "total_s": 1.35,
                "cache_hit_ratio": 32700.0 / 32960.0,
            },
        },
        {
            "target_ctx_tokens": 65536,
            "turn1_prime": {"prompt_tokens": 65600, "cached_tokens": 0, "completion_tokens": 8, "ttft_s": 34.0, "total_s": 34.2},
            "turn2_warm": {
                "prompt_tokens": 65760,
                "cached_tokens": 65500,
                "completion_tokens": 64,
                "ttft_s": 0.80,
                "total_s": 1.60,
                "cache_hit_ratio": 65500.0 / 65760.0,
            },
        },
        {
            "target_ctx_tokens": 96000,
            "turn1_prime": {"prompt_tokens": 96200, "cached_tokens": 0, "completion_tokens": 8, "ttft_s": 50.0, "total_s": 50.2},
            "turn2_warm": {
                "prompt_tokens": 96360,
                "cached_tokens": 96000,
                "completion_tokens": 64,
                "ttft_s": 0.95,
                "total_s": 1.85,
                "cache_hit_ratio": 96000.0 / 96360.0,
            },
        },
    ]
    cold_run = {
        "measured": True,
        "target_tokens": 34816,
        "server_ready_s": 1.8,
        "cold_total_s": 16.5,
        "request": {
            "prompt_tokens": 34900,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 34900,
            "completion_tokens": 16,
            "ttft_s": 14.4,
            "total_s": 14.7,
        },
    }
    bench: dict[str, Any] = {
        "mode": "live_engine",
        "fake_smoke": False,
        "server_origin": "managed_local",
        "accelerator": {
            "backend": "hip",
            "selected_device_arch": "gfx1151",
            "source": "runtime_http",
            "conflict": False,
        },
        "model_identity": {
            "model_id": "Qwen3.8-Flash-Next",
            "models_listed": 1,
            "model_entry_matched": True,
            "content_sha256": "a" * 64,
            "source": "runtime_http",
            "conflict": False,
            "expected_model_sha256": "a" * 64,
        },
        "decode_runs": decode_runs,
        "prefill_runs": prefill_runs,
        "agent_runs": agent_runs,
        "cold_run": cold_run,
    }
    verified, notes = shv.extract_verified_metrics(bench)
    bench["summary_metrics"] = verified
    bench["eligibility_notes"] = notes
    return bench


def _write_mock_serve_launcher(path: Path) -> None:
    """Write a minimal executable Python HTTP server matching ingine-serve CLI flags for cold-spawn tests."""
    script = f"""#!{sys.executable}
import argparse, http.server, json, socketserver, sys, time

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): return
    def do_GET(self):
        if self.path == "/health":
            b = b'{{"status":"ok"}}'
        elif self.path == "/v1/models":
            b = b'{{"object":"list","data":[{{"id":"mock-strix"}}]}}'
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n).decode("utf-8"))
        msgs = req.get("messages", [])
        pt = max(1, len(" ".join(m.get("content", "") for m in msgs).split()))
        n_emit = min(int(req.get("max_completion_tokens", 8)), 8)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(n_emit):
            c = {{"id":"x","object":"chat.completion.chunk","choices":[{{"index":0,"delta":{{"content":"a "}},"finish_reason":None}}],"usage":None}}
            self.wfile.write(("data: " + json.dumps(c) + "\\n\\n").encode("utf-8"))
            self.wfile.flush()
            if i + 1 < n_emit:
                time.sleep(0.0005)
        u = {{"id":"x","object":"chat.completion.chunk","choices":[],"usage":{{"prompt_tokens":pt,"completion_tokens":n_emit,"total_tokens":pt+n_emit,"prompt_tokens_details":{{"cached_tokens":0}}}}}}
        self.wfile.write(("data: " + json.dumps(u) + "\\n\\n").encode("utf-8"))
        self.wfile.write(b"data: [DONE]\\n\\n")
        self.wfile.flush()

p = argparse.ArgumentParser()
p.add_argument("--model-dir")
p.add_argument("--host", default="127.0.0.1")
p.add_argument("--port", type=int, required=True)
p.add_argument("--engine", action="store_true")
p.add_argument("--fake", action="store_true")
a, _ = p.parse_known_args()
class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
S((a.host, a.port), H).serve_forever()
"""
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class TestStrixHaloVerify(unittest.TestCase):
    def test_floor_constants_match_cpp_vendor_policy(self) -> None:
        header = (REPO_ROOT / "native" / "src" / "core" / "vendor_policy.h").read_text(encoding="utf-8")
        cpp_map = {
            "decode_code_tok_s": "min_decode_code_tok_s",
            "decode_prose_tok_s": "min_decode_prose_tok_s",
            "decode_edit_tok_s": "min_decode_edit_tok_s",
            "decode_tool_tok_s": "min_decode_tool_tok_s",
            "decode_multilang_tok_s": "min_decode_multilang_tok_s",
            "prefill_8k_tok_s": "min_prefill_8k_tok_s",
            "prefill_32k_tok_s": "min_prefill_32k_tok_s",
            "prefill_131k_tok_s": "min_prefill_131k_tok_s",
            "agent_turn_30k_96k_s": "max_agent_turn_30k_96k_s",
            "agent_ttft_30k_96k_s": "max_agent_ttft_30k_96k_s",
            "cold_34k_s": "max_cold_34k_s",
        }
        self.assertEqual(len(shv.STRIX_HALO_TARGET_FLOOR), len(cpp_map))
        for rule in shv.STRIX_HALO_TARGET_FLOOR:
            field = cpp_map[rule.key]
            m = re.search(rf"double\s+{field}\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*;", header)
            self.assertIsNotNone(m, f"Missing {field} in vendor_policy.h")
            self.assertEqual(float(m.group(1)), rule.threshold, f"Drift on {rule.key}")

    def test_strict_scoring_and_fake_rejection(self) -> None:
        bench = _make_qualifying_bench()
        strix_report = {"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": bench}
        s_pass = shv.score_report(strix_report)
        self.assertEqual(s_pass["verdict"], "PASS_STRIX_HALO_FLOOR")
        self.assertTrue(s_pass["hardware_verified"])
        self.assertEqual(s_pass["passed"], 11)

        # Non-gfx1151 hardware cannot claim hardware_verified even if all 11 qualifying metrics pass.
        s_non = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": False}}, "bench": bench})
        self.assertEqual(s_non["verdict"], "PASS_FLOOR_NON_GFX1151")
        self.assertFalse(s_non["hardware_verified"])
        self.assertEqual(s_non["passed"], 11)

        # Even if rocminfo lists gfx1151, running on another backend/arch (e.g. cuda or gfx1100) must not verify.
        b_wrong_arch = copy.deepcopy(bench)
        b_wrong_arch["accelerator"] = {"backend": "hip", "selected_device_arch": "gfx1100", "source": "runtime_http"}
        s_wa = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_wrong_arch})
        self.assertEqual(s_wa["verdict"], "PASS_FLOOR_NON_GFX1151")
        self.assertFalse(s_wa["hardware_verified"])
        self.assertEqual(s_wa["passed"], 11)

        b_wrong_be = copy.deepcopy(bench)
        b_wrong_be["accelerator"] = {"backend": "cpu", "selected_device_arch": "cpu", "source": "runtime_http"}
        s_wb = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_wrong_be})
        self.assertEqual(s_wb["verdict"], "PASS_FLOOR_NON_GFX1151")
        self.assertFalse(s_wb["hardware_verified"])

        # Non-live_engine mode MUST block BOTH PASS_STRIX_HALO_FLOOR verdict AND hardware_verified while keeping all 11 rows.
        b_nonlive = copy.deepcopy(bench)
        b_nonlive["mode"] = "custom_non_live"
        b_nonlive["fake_smoke"] = False
        s_nl = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_nonlive})
        self.assertEqual(s_nl["verdict"], "UNMEASURED_NON_LIVE_MODE")
        self.assertFalse(s_nl["hardware_verified"])
        self.assertEqual(s_nl["passed"], 11)
        self.assertEqual(len(s_nl["rows"]), 11)
        self.assertTrue(all(r["measured"] is not None for r in s_nl["rows"]))

        # Unbound model identity or unbound selected accelerator preserves observations but blocks PASS_STRIX_HALO_FLOOR.
        b_unbound_model = copy.deepcopy(bench)
        b_unbound_model["model_identity"] = {
            "model_id": "Qwen3.8-Flash-Next",
            "content_sha256": None,
            "source": "missing",
            "expected_model_sha256": None,
        }
        # Even if preflight scanned a local INGPACK1 file with a matching hash, score_report MUST NOT accept it!
        s_um = shv.score_report(
            {
                "preflight": {"rocm": {"gfx1151_detected": True}, "model": {"pack_content_sha256": "a" * 64}},
                "bench": b_unbound_model,
            },
            expected_model_sha256="a" * 64,
        )
        self.assertEqual(s_um["verdict"], "PASS_FLOOR_UNBOUND_IDENTITY")
        self.assertFalse(s_um["hardware_verified"])
        self.assertIsNone(s_um["observed_model_sha256"])
        self.assertEqual(s_um["passed"], 11)

        b_unbound_dev = copy.deepcopy(bench)
        b_unbound_dev["accelerator"] = {
            "backend": None,
            "selected_device_arch": None,
            "source": "missing",
            "expected_backend": "hip",
            "expected_device_arch": "gfx1151",
        }
        s_ud = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_unbound_dev})
        self.assertEqual(s_ud["verdict"], "PASS_FLOOR_UNBOUND_IDENTITY")
        self.assertFalse(s_ud["hardware_verified"])
        self.assertEqual(s_ud["passed"], 11)

        # Non-authoritative source tags (e.g. "cli" or "local_file") cannot produce PASS_STRIX_HALO_FLOOR.
        b_cli_src = copy.deepcopy(bench)
        b_cli_src["accelerator"]["source"] = "cli"
        s_cli = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_cli_src})
        self.assertEqual(s_cli["verdict"], "PASS_FLOOR_UNBOUND_IDENTITY")
        self.assertFalse(s_cli["hardware_verified"])

        b_local_src = copy.deepcopy(bench)
        b_local_src["model_identity"]["source"] = "local_file"
        s_loc = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_local_src})
        self.assertEqual(s_loc["verdict"], "PASS_FLOOR_UNBOUND_IDENTITY")
        self.assertFalse(s_loc["hardware_verified"])
        self.assertIsNone(s_loc["observed_model_sha256"])

        # Model pack SHA-256 mismatch is caught explicitly while preserving all 11 observed rows.
        s_mm = shv.score_report(
            {"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": bench},
            expected_model_sha256="b" * 64,
        )
        self.assertEqual(s_mm["verdict"], "PASS_FLOOR_MODEL_IDENTITY_MISMATCH")
        self.assertFalse(s_mm["hardware_verified"])
        self.assertEqual(s_mm["passed"], 11)

        # Fake smoke mode MUST ALWAYS be rejected as UNMEASURED_FAKE_MODE even with full evidence.
        fake_bench = copy.deepcopy(bench)
        fake_bench["mode"] = "smoke_fake_only"
        fake_bench["fake_smoke"] = True
        s_fake = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": fake_bench})
        self.assertEqual(s_fake["verdict"], "UNMEASURED_FAKE_MODE")
        self.assertFalse(s_fake["hardware_verified"])
        self.assertEqual(s_fake["unmeasured"], 11)

        # Bare summary_metrics without raw evidence CANNOT pass.
        bare_report = {
            "preflight": {"rocm": {"gfx1151_detected": True}},
            "bench": {"mode": "live_engine", "fake_smoke": False, "summary_metrics": dict(bench["summary_metrics"])},
        }
        s_bare = shv.score_report(bare_report)
        self.assertEqual(s_bare["verdict"], "INCOMPLETE_UNMEASURED")
        self.assertFalse(s_bare["hardware_verified"])
        self.assertEqual(s_bare["unmeasured"], 11)

        # Boundary equality on every single floor metric must FAIL (strict > / <).
        def mutate_to_boundary(key: str, threshold: float) -> dict[str, Any]:
            b = copy.deepcopy(bench)
            if key.startswith("decode_"):
                suite = key[len("decode_") : -len("_tok_s")]
                b["decode_runs"][suite]["decode_tok_s"] = threshold
            elif key == "prefill_8k_tok_s":
                b["prefill_runs"]["8192"]["ttft_s"] = b["prefill_runs"]["8192"]["uncached_prompt_tokens"] / threshold
            elif key == "prefill_32k_tok_s":
                b["prefill_runs"]["32768"]["ttft_s"] = b["prefill_runs"]["32768"]["uncached_prompt_tokens"] / threshold
            elif key == "prefill_131k_tok_s":
                b["prefill_runs"]["131072"]["ttft_s"] = b["prefill_runs"]["131072"]["uncached_prompt_tokens"] / threshold
            elif key == "agent_turn_30k_96k_s":
                b["agent_runs"][-1]["turn2_warm"]["total_s"] = threshold
            elif key == "agent_ttft_30k_96k_s":
                b["agent_runs"][-1]["turn2_warm"]["ttft_s"] = threshold
            elif key == "cold_34k_s":
                b["cold_run"]["cold_total_s"] = threshold
            return b

        for rule in shv.STRIX_HALO_TARGET_FLOOR:
            s_bad = shv.score_report(
                {"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": mutate_to_boundary(rule.key, rule.threshold)}
            )
            self.assertEqual(s_bad["verdict"], "FAIL_STRIX_HALO_FLOOR", f"Boundary should fail for {rule.key}")
            self.assertFalse(s_bad["hardware_verified"])

    def test_short_diagnostic_64tok_warm_and_cold_cannot_pass_floor(self) -> None:
        """Short 64-token warm/cold diagnostic runs without prefix hits must stay UNMEASURED."""
        with tempfile.TemporaryDirectory() as td:
            serve_bin = Path(td) / "mock_serve.py"
            model_dir = Path(td) / "model"
            model_dir.mkdir()
            _write_mock_serve_launcher(serve_bin)

            bench = shv.run_benchmark_suite(
                base_url=None,
                serve_bin=serve_bin,
                model_dir=model_dir,
                model_id_override=None,
                fake_smoke=False,
                extra_serve_args=[],
                decode_max_tokens=8,
                prefill_lengths=[64, 128, 192],
                agent_lengths=[64],
                cold_tokens=64,
                startup_timeout_s=10.0,
            )
            # Even if decode and 8K/32K/131K prefill passed, short 64-tok warm/cold MUST remain UNMEASURED.
            bench["prefill_runs"] = _make_qualifying_bench()["prefill_runs"]
            score = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": bench})
            self.assertEqual(score["verdict"], "INCOMPLETE_UNMEASURED")
            self.assertFalse(score["hardware_verified"])

            by_key = {r["key"]: r for r in score["rows"]}
            self.assertEqual(by_key["agent_turn_30k_96k_s"]["status"], "UNMEASURED")
            self.assertEqual(by_key["agent_ttft_30k_96k_s"]["status"], "UNMEASURED")
            self.assertEqual(by_key["cold_34k_s"]["status"], "UNMEASURED")
            self.assertIsNone(bench["summary_metrics"]["agent_turn_30k_96k_s"])
            self.assertIsNone(bench["summary_metrics"]["cold_34k_s"])
            # Short diagnostics are still preserved cleanly for inspection.
            self.assertEqual(len(bench["diagnostic_metrics"]["agent_warm_turns"]), 1)
            self.assertTrue(bench["diagnostic_metrics"]["cold_run"]["measured"])
            self.assertLess(bench["diagnostic_metrics"]["cold_run"]["observed_prompt_tokens"], 500)

    def test_prefill_spoofed_keys_and_cached_prefill_rejected(self) -> None:
        # 1. Keys say 8192/32768/131072, but observed server prompt tokens are only 96 -> UNMEASURED.
        b_short = _make_qualifying_bench()
        for k in ("8192", "32768", "131072"):
            b_short["prefill_runs"][k]["prompt_tokens"] = 96
            b_short["prefill_runs"][k]["uncached_prompt_tokens"] = 96
            b_short["prefill_runs"][k]["ttft_s"] = 0.001
        s_short = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_short})
        self.assertEqual(s_short["verdict"], "INCOMPLETE_UNMEASURED")
        self.assertFalse(s_short["hardware_verified"])
        for k in ("prefill_8k_tok_s", "prefill_32k_tok_s", "prefill_131k_tok_s"):
            row = next(r for r in s_short["rows"] if r["key"] == k)
            self.assertEqual(row["status"], "UNMEASURED")

        # 2. Cached prefill (cached_tokens > 0) cannot satisfy cold prefill floor.
        b_cached = _make_qualifying_bench()
        b_cached["prefill_runs"]["32768"]["cached_tokens"] = 1000
        b_cached["prefill_runs"]["32768"]["uncached_prompt_tokens"] = 31900
        s_cached = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_cached})
        self.assertEqual(s_cached["verdict"], "INCOMPLETE_UNMEASURED")
        row_32k = next(r for r in s_cached["rows"] if r["key"] == "prefill_32k_tok_s")
        self.assertEqual(row_32k["status"], "UNMEASURED")
        self.assertIn("cached", row_32k["note"])

        # 3. Single 131K run alone cannot satisfy 8K or 32K bands.
        b_only131 = _make_qualifying_bench()
        b_only131["prefill_runs"] = {"131072": b_only131["prefill_runs"]["131072"]}
        s_only131 = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_only131})
        rows = {r["key"]: r["status"] for r in s_only131["rows"]}
        self.assertEqual(rows["prefill_131k_tok_s"], "PASS")
        self.assertEqual(rows["prefill_32k_tok_s"], "UNMEASURED")
        self.assertEqual(rows["prefill_8k_tok_s"], "UNMEASURED")

    def test_agent_span_and_warm_cache_observation_required(self) -> None:
        # 1. Zero cached_tokens at 32K/64K/96K -> UNMEASURED (no warm cache observed).
        b_nocache = _make_qualifying_bench()
        for r in b_nocache["agent_runs"]:
            r["turn2_warm"]["cached_tokens"] = 0
        s_nc = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_nocache})
        self.assertEqual(s_nc["verdict"], "INCOMPLETE_UNMEASURED")
        self.assertFalse(s_nc["hardware_verified"])

        # 2. Trivial partial cache hit (10 tokens < 80% of Turn 1) -> UNMEASURED.
        b_lowhit = _make_qualifying_bench()
        b_lowhit["agent_runs"][-1]["turn2_warm"]["cached_tokens"] = 10
        s_lh = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_lowhit})
        self.assertEqual(s_lh["verdict"], "INCOMPLETE_UNMEASURED")

        # 3. 32K only (missing upper 96K band) -> UNMEASURED.
        b_lowonly = _make_qualifying_bench()
        b_lowonly["agent_runs"] = [b_lowonly["agent_runs"][0]]
        s_lo = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_lowonly})
        self.assertEqual(s_lo["verdict"], "INCOMPLETE_UNMEASURED")

        # 4. 32K + 96K warm cached, but middle 64K missed cache -> UNMEASURED (no cherry-picking).
        b_midmiss = _make_qualifying_bench()
        b_midmiss["agent_runs"][1]["turn2_warm"]["cached_tokens"] = 0
        s_mm = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_midmiss})
        self.assertEqual(s_mm["verdict"], "INCOMPLETE_UNMEASURED")

    def test_cold_34k_requires_uncached_34k_prompt_and_managed_spawn(self) -> None:
        # Short 96-token cold run -> UNMEASURED.
        b_short = _make_qualifying_bench()
        b_short["cold_run"]["request"]["prompt_tokens"] = 96
        b_short["cold_run"]["request"]["uncached_prompt_tokens"] = 96
        s_short = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_short})
        self.assertEqual(s_short["verdict"], "INCOMPLETE_UNMEASURED")
        self.assertFalse(s_short["hardware_verified"])

        # Cached cold run -> UNMEASURED.
        b_cached = _make_qualifying_bench()
        b_cached["cold_run"]["request"]["cached_tokens"] = 512
        s_cached = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_cached})
        self.assertEqual(s_cached["verdict"], "INCOMPLETE_UNMEASURED")

        # External server (measured=False) -> UNMEASURED.
        b_ext = _make_qualifying_bench()
        b_ext["cold_run"] = {"measured": False, "reason": "UNMEASURED_EXTERNAL_SERVER", "cold_total_s": None}
        s_ext = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": b_ext})
        self.assertEqual(s_ext["verdict"], "INCOMPLETE_UNMEASURED")

    def test_ingpack_header_identity_reader(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            pack_path = model_dir / "model.ingpack"
            hdr = bytearray(256)
            hdr[0:8] = b"INGPACK1"
            hdr[8:12] = (1).to_bytes(4, "little")
            hdr[12:16] = (1).to_bytes(4, "little")  # kWeightPackFlagHasContentSha256
            hdr[96:128] = bytes.fromhex("11" * 32)
            hdr[128:160] = bytes.fromhex("22" * 32)
            pack_path.write_bytes(bytes(hdr))
            prov = shv.collect_model_provenance(model_dir, hash_packs=False)
            self.assertTrue(prov["has_ingpack"])
            self.assertEqual(prov["pack_header_sha256"], "11" * 32)
            self.assertEqual(prov["pack_content_sha256"], "22" * 32)

    def test_rocminfo_gfx1151_parser(self) -> None:
        sample = """
        Agent 1
          Name: AMD Ryzen AI Max+ 395
        Agent 2
          Name: gfx1151
          Marketing Name: AMD Radeon 8060S Graphics
        """
        targets = shv.parse_gfx_targets_from_rocminfo(sample)
        self.assertEqual(targets, ["gfx1151"])
        self.assertEqual(shv.parse_gfx_targets_from_rocminfo("Agent 1: gfx1100 and gfx942"), ["gfx1100", "gfx942"])

    def test_ctest_summary_parser(self) -> None:
        sample = """
        1/4 Test  #7: core_safetensors .................   Passed    0.01 sec
        2/4 Test #15: core_upload ......................   Passed    0.35 sec
        3/4 Test #47: serve_server .....................***Skipped   0.00 sec
        4/4 Test #59: core_vendor_portability ..........***Failed    0.02 sec
        """
        summary = shv.parse_ctest_summary(sample)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["skipped"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["total_observed"], 4)

    def test_live_sse_bench_and_scorecard(self) -> None:
        _MockIngineHandler.seen_prefixes = set()
        _MockIngineHandler.request_log = []
        _MockIngineHandler.disable_prefix_cache = False
        server = _ThreadedHTTPServer(("127.0.0.1", 0), _MockIngineHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            res = shv.run_benchmark_suite(
                base_url=f"http://127.0.0.1:{port}",
                serve_bin=None,
                model_dir=None,
                model_id_override=None,
                fake_smoke=False,
                extra_serve_args=[],
                decode_max_tokens=8,
                prefill_lengths=[128, 256, 384],
                agent_lengths=[256],
                cold_tokens=256,
                startup_timeout_s=10.0,
            )
        finally:
            server.shutdown()
            server.server_close()

        self.assertEqual(res["model_id"], "Qwen3.8-Flash-Next")
        self.assertEqual(set(res["decode_runs"].keys()), {"code", "prose", "edit", "tool", "multilang"})
        for name, dr in res["decode_runs"].items():
            self.assertEqual(dr["completion_tokens"], 8, f"decode {name} completion tokens")
            self.assertGreater(dr["decode_tok_s"], 0.0)
        for length in ("128", "256", "384"):
            self.assertEqual(res["prefill_runs"][length]["cached_tokens"], 0)
            self.assertGreater(res["prefill_runs"][length]["prefill_tok_s"], 0.0)
        self.assertEqual(len(res["agent_runs"]), 1)
        self.assertGreater(res["agent_runs"][0]["turn2_warm"]["cached_tokens"], 0)
        self.assertGreater(res["agent_runs"][0]["turn2_warm"]["cache_hit_ratio"], 0.5)
        self.assertFalse(res["cold_run"]["measured"])
        self.assertEqual(res["cold_run"]["reason"], "UNMEASURED_EXTERNAL_SERVER")

        # Short diagnostic lengths + external server must remain INCOMPLETE_UNMEASURED.
        report = {"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": res}
        score = shv.score_report(report)
        self.assertEqual(score["verdict"], "INCOMPLETE_UNMEASURED")
        self.assertFalse(score["hardware_verified"])
        md = shv.render_scorecard_markdown(report, score)
        self.assertIn("INCOMPLETE_UNMEASURED", md)
        self.assertIn("Decode — Edit", md)

        # CLI `score` exits 2 strictly, 0 with --allow-unmeasured.
        with tempfile.TemporaryDirectory() as td:
            rpt_path = Path(td) / "report.json"
            md_path = Path(td) / "scorecard.md"
            rpt_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(shv.main(["score", str(rpt_path), "--markdown-out", str(md_path)]), 2)
            self.assertEqual(shv.main(["score", str(rpt_path), "--markdown-out", str(md_path), "--allow-unmeasured"]), 0)
            self.assertTrue(md_path.exists())

    def test_false_identity_cli_override_external_pack_and_multimodel_binding(self) -> None:
        """Verify false-identity attacks are rejected while preserving all 11 raw observations."""
        sha_a = "a" * 64
        sha_b = "b" * 64

        def _make_handler(health_payload: dict[str, Any], models_payload: dict[str, Any]) -> type[_MockIngineHandler]:
            class _CustomHandler(_MockIngineHandler):
                def do_GET(self) -> None:  # noqa: N802
                    if self.path == "/health":
                        body = json.dumps(health_payload).encode("utf-8")
                    elif self.path == "/v1/models":
                        body = json.dumps(models_payload).encode("utf-8")
                    else:
                        self.send_response(404)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

            return _CustomHandler

        def _write_fake_ingpack(model_dir: Path, content_sha_hex: str) -> None:
            hdr = bytearray(256)
            hdr[0:8] = b"INGPACK1"
            hdr[8:12] = (1).to_bytes(4, "little")
            hdr[12:16] = (1).to_bytes(4, "little")
            hdr[96:128] = bytes.fromhex("11" * 32)
            hdr[128:160] = bytes.fromhex(content_sha_hex)
            (model_dir / "model.ingpack").write_bytes(bytes(hdr))

        # 1. CLI --selected-backend hip --selected-device-arch gfx1151 MUST NOT override contrary runtime metadata,
        #    and an external --base-url MUST NOT inherit a local --model-dir INGPACK1 hash.
        h1 = _make_handler(
            {"status": "ok", "backend": "cpu", "selected_device_arch": "zen5"},
            {"object": "list", "data": [{"id": "Qwen3.8-Flash-Next"}]},
        )
        server = _ThreadedHTTPServer(("127.0.0.1", 0), h1)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                mdir = Path(td) / "local_pack"
                mdir.mkdir()
                _write_fake_ingpack(mdir, sha_a)
                res = shv.run_benchmark_suite(
                    base_url=f"http://127.0.0.1:{port}",
                    serve_bin=None,
                    model_dir=mdir,
                    model_id_override=None,
                    fake_smoke=False,
                    extra_serve_args=[],
                    decode_max_tokens=4,
                    prefill_lengths=[64],
                    agent_lengths=[64],
                    cold_tokens=64,
                    startup_timeout_s=10.0,
                    expected_model_sha256=sha_a,
                    selected_device_arch="gfx1151",
                    selected_backend="hip",
                )
        finally:
            server.shutdown()
            server.server_close()

        # Runtime metadata stays authoritative ("cpu"/"zen5"); CLI flags remain expected assertions.
        self.assertEqual(res["accelerator"]["backend"], "cpu")
        self.assertEqual(res["accelerator"]["selected_device_arch"], "zen5")
        self.assertEqual(res["accelerator"]["expected_backend"], "hip")
        self.assertEqual(res["accelerator"]["expected_device_arch"], "gfx1151")
        # External server gave no runtime SHA -> observed content_sha256 stays None despite local INGPACK1 file!
        self.assertIsNone(res["model_identity"]["content_sha256"])
        self.assertEqual(res["model_identity"]["local_model_dir_pack_sha256"], sha_a)

        # Graft passing raw 11 metrics onto this captured identity to verify verdict rejects false CLI + local pack.
        q = _make_qualifying_bench()
        res["decode_runs"] = q["decode_runs"]
        res["prefill_runs"] = q["prefill_runs"]
        res["agent_runs"] = q["agent_runs"]
        res["cold_run"] = q["cold_run"]
        score_cli_conflict = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": res})
        self.assertEqual(score_cli_conflict["verdict"], "PASS_FLOOR_ACCELERATOR_MISMATCH")
        self.assertFalse(score_cli_conflict["hardware_verified"])
        self.assertEqual(score_cli_conflict["passed"], 11)

        # 2. External server reporting HIP gfx1151 but NO model hash + matching local INGPACK1 -> UNBOUND.
        h2 = _make_handler(
            {"status": "ok", "backend": "hip", "selected_device_arch": "gfx1151"},
            {"object": "list", "data": [{"id": "Qwen3.8-Flash-Next"}]},
        )
        server2 = _ThreadedHTTPServer(("127.0.0.1", 0), h2)
        port2 = server2.server_address[1]
        thread2 = threading.Thread(target=server2.serve_forever, daemon=True)
        thread2.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                mdir = Path(td) / "local_pack"
                mdir.mkdir()
                _write_fake_ingpack(mdir, sha_a)
                res2 = shv.run_benchmark_suite(
                    base_url=f"http://127.0.0.1:{port2}",
                    serve_bin=None,
                    model_dir=mdir,
                    model_id_override=None,
                    fake_smoke=False,
                    extra_serve_args=[],
                    decode_max_tokens=4,
                    prefill_lengths=[64],
                    agent_lengths=[64],
                    cold_tokens=64,
                    startup_timeout_s=10.0,
                    expected_model_sha256=sha_a,
                )
        finally:
            server2.shutdown()
            server2.server_close()

        self.assertIsNone(res2["model_identity"]["content_sha256"])
        res2["decode_runs"] = q["decode_runs"]
        res2["prefill_runs"] = q["prefill_runs"]
        res2["agent_runs"] = q["agent_runs"]
        res2["cold_run"] = q["cold_run"]
        score_ext_unbound = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": res2})
        self.assertEqual(score_ext_unbound["verdict"], "PASS_FLOOR_UNBOUND_IDENTITY")
        self.assertFalse(score_ext_unbound["hardware_verified"])
        self.assertEqual(score_ext_unbound["passed"], 11)

        # 3. Multi-model /v1/models binds metadata strictly to the requested model entry.
        h3 = _make_handler(
            {"status": "ok", "backend": "hip", "selected_device_arch": "gfx1151", "content_sha256": sha_a},
            {
                "object": "list",
                "data": [
                    {"id": "Model-A", "backend": "hip", "selected_device_arch": "gfx1151", "content_sha256": sha_a},
                    {"id": "Model-B", "backend": "hip", "selected_device_arch": "gfx1151", "content_sha256": sha_b},
                    {"id": "Model-NoHash", "backend": "hip", "selected_device_arch": "gfx1151"},
                ],
            },
        )
        server3 = _ThreadedHTTPServer(("127.0.0.1", 0), h3)
        port3 = server3.server_address[1]
        thread3 = threading.Thread(target=server3.serve_forever, daemon=True)
        thread3.start()
        try:
            url3 = f"http://127.0.0.1:{port3}"
            info_b = shv.detect_server_runtime_info(url3, "Model-B")
            self.assertTrue(info_b["model_entry_matched"])
            self.assertEqual(info_b["content_sha256"], sha_b)

            info_nohash = shv.detect_server_runtime_info(url3, "Model-NoHash")
            self.assertTrue(info_nohash["model_entry_matched"])
            # Must NOT inherit global /health hash sha_a when /v1/models lists multiple models.
            self.assertIsNone(info_nohash["content_sha256"])

            info_missing = shv.detect_server_runtime_info(url3, "Model-Missing")
            self.assertFalse(info_missing["model_entry_matched"])
            self.assertIsNone(info_missing["content_sha256"])
        finally:
            server3.shutdown()
            server3.server_close()

        # 4. Locally launched server: verified launch/loader telemetry binds identity; absent or conflicting telemetry fails closed.
        with tempfile.TemporaryDirectory() as td:
            mdir = Path(td) / "model"
            mdir.mkdir()
            _write_fake_ingpack(mdir, sha_a)

            # 4a. Managed launcher emitting verified loader telemetry -> passes when metrics pass.
            tel_bin = Path(td) / "mock_serve_telemetry.py"
            _write_mock_serve_launcher(tel_bin)
            src = tel_bin.read_text(encoding="utf-8")
            inject = (
                f'sys.stderr.write("loader: backend=hip selected_device_arch=gfx1151 '
                f'content_sha256={sha_a} loaded_model_dir=" + str(a.model_dir) + "\\n"); sys.stderr.flush()\n'
            )
            tel_bin.write_text(src.replace("S((a.host, a.port), H).serve_forever()", inject + "S((a.host, a.port), H).serve_forever()"), encoding="utf-8")
            res_local_ok = shv.run_benchmark_suite(
                base_url=None,
                serve_bin=tel_bin,
                model_dir=mdir,
                model_id_override=None,
                fake_smoke=False,
                extra_serve_args=[],
                decode_max_tokens=4,
                prefill_lengths=[64],
                agent_lengths=[64],
                cold_tokens=64,
                startup_timeout_s=10.0,
                expected_model_sha256=sha_a,
                selected_backend="hip",
                selected_device_arch="gfx1151",
            )
            self.assertEqual(res_local_ok["accelerator"]["source"], "managed_process_telemetry")
            self.assertEqual(res_local_ok["model_identity"]["source"], "managed_process_telemetry")
            self.assertEqual(res_local_ok["model_identity"]["content_sha256"], sha_a)
            res_local_ok["decode_runs"] = q["decode_runs"]
            res_local_ok["prefill_runs"] = q["prefill_runs"]
            res_local_ok["agent_runs"] = q["agent_runs"]
            res_local_ok["cold_run"] = q["cold_run"]
            s_ok = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": res_local_ok})
            self.assertEqual(s_ok["verdict"], "PASS_STRIX_HALO_FLOOR")
            self.assertTrue(s_ok["hardware_verified"])

            # 4d. Managed launcher where BOTH HTTP and stderr loader telemetry agree (including asymmetric
            #     backend-in-both + arch-in-stderr-only) -> canonical deduplicated source passes PASS_STRIX_HALO_FLOOR!
            both_bin = Path(td) / "mock_serve_both_agree.py"
            _write_mock_serve_launcher(both_bin)
            bsrc = both_bin.read_text(encoding="utf-8")
            bsrc = bsrc.replace(
                'b = b\'{"status":"ok"}\'',
                f'b = b\'{{"status":"ok","backend":"hip","content_sha256":"{sha_a}"}}\'',
            )
            binj = (
                f'sys.stderr.write("loader: backend=hip selected_device_arch=gfx1151 '
                f'content_sha256={sha_a}\\n"); sys.stderr.flush()\n'
            )
            both_bin.write_text(
                bsrc.replace("S((a.host, a.port), H).serve_forever()", binj + "S((a.host, a.port), H).serve_forever()"),
                encoding="utf-8",
            )
            res_both = shv.run_benchmark_suite(
                base_url=None,
                serve_bin=both_bin,
                model_dir=mdir,
                model_id_override=None,
                fake_smoke=False,
                extra_serve_args=[],
                decode_max_tokens=4,
                prefill_lengths=[64],
                agent_lengths=[64],
                cold_tokens=64,
                startup_timeout_s=10.0,
                expected_model_sha256=sha_a,
                selected_backend="hip",
                selected_device_arch="gfx1151",
            )
            self.assertEqual(res_both["accelerator"]["source"], "managed_process_telemetry+runtime_http")
            self.assertEqual(res_both["model_identity"]["source"], "managed_process_telemetry+runtime_http")
            self.assertFalse(res_both["accelerator"]["conflict"])
            self.assertFalse(res_both["model_identity"]["conflict"])
            res_both["decode_runs"] = q["decode_runs"]
            res_both["prefill_runs"] = q["prefill_runs"]
            res_both["agent_runs"] = q["agent_runs"]
            res_both["cold_run"] = q["cold_run"]
            s_both = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": res_both})
            self.assertEqual(s_both["verdict"], "PASS_STRIX_HALO_FLOOR")
            self.assertTrue(s_both["hardware_verified"])
            self.assertEqual(s_both["accelerator_source"], "managed_process_telemetry+runtime_http")
            self.assertEqual(s_both["model_identity_source"], "managed_process_telemetry+runtime_http")
            self.assertEqual(s_both["passed"], 11)

            # 4c. Managed launcher with conflicting HTTP vs stderr telemetry -> fails closed as mismatch!
            conflict_bin = Path(td) / "mock_serve_conflict.py"
            _write_mock_serve_launcher(conflict_bin)
            csrc = conflict_bin.read_text(encoding="utf-8")
            csrc = csrc.replace(
                'b = b\'{"status":"ok"}\'',
                'b = b\'{"status":"ok","backend":"cpu","selected_device_arch":"zen5"}\'',
            )
            cinj = (
                f'sys.stderr.write("loader: backend=hip selected_device_arch=gfx1151 '
                f'content_sha256={sha_a}\\n"); sys.stderr.flush()\n'
            )
            conflict_bin.write_text(
                csrc.replace("S((a.host, a.port), H).serve_forever()", cinj + "S((a.host, a.port), H).serve_forever()"),
                encoding="utf-8",
            )
            res_conf = shv.run_benchmark_suite(
                base_url=None,
                serve_bin=conflict_bin,
                model_dir=mdir,
                model_id_override=None,
                fake_smoke=False,
                extra_serve_args=[],
                decode_max_tokens=4,
                prefill_lengths=[64],
                agent_lengths=[64],
                cold_tokens=64,
                startup_timeout_s=10.0,
                expected_model_sha256=sha_a,
            )
            self.assertTrue(res_conf["accelerator"]["conflict"])
            self.assertEqual(res_conf["accelerator"]["source"], "conflict")
            res_conf["decode_runs"] = q["decode_runs"]
            res_conf["prefill_runs"] = q["prefill_runs"]
            res_conf["agent_runs"] = q["agent_runs"]
            res_conf["cold_run"] = q["cold_run"]
            s_conf = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": res_conf})
            self.assertEqual(s_conf["verdict"], "PASS_FLOOR_ACCELERATOR_MISMATCH")
            self.assertFalse(s_conf["hardware_verified"])
            self.assertEqual(s_conf["passed"], 11)

            # 4b. Managed launcher emitting NO telemetry -> stays unbound despite local INGPACK1 + CLI flags!
            silent_bin = Path(td) / "mock_serve_silent.py"
            _write_mock_serve_launcher(silent_bin)
            res_local_silent = shv.run_benchmark_suite(
                base_url=None,
                serve_bin=silent_bin,
                model_dir=mdir,
                model_id_override=None,
                fake_smoke=False,
                extra_serve_args=[],
                decode_max_tokens=4,
                prefill_lengths=[64],
                agent_lengths=[64],
                cold_tokens=64,
                startup_timeout_s=10.0,
                expected_model_sha256=sha_a,
                selected_backend="hip",
                selected_device_arch="gfx1151",
            )
            self.assertIsNone(res_local_silent["model_identity"]["content_sha256"])
            self.assertIsNone(res_local_silent["accelerator"]["backend"])
            res_local_silent["decode_runs"] = q["decode_runs"]
            res_local_silent["prefill_runs"] = q["prefill_runs"]
            res_local_silent["agent_runs"] = q["agent_runs"]
            res_local_silent["cold_run"] = q["cold_run"]
            s_sil = shv.score_report({"preflight": {"rocm": {"gfx1151_detected": True}}, "bench": res_local_silent})
            self.assertEqual(s_sil["verdict"], "PASS_FLOOR_UNBOUND_IDENTITY")
            self.assertFalse(s_sil["hardware_verified"])
            self.assertEqual(s_sil["passed"], 11)


if __name__ == "__main__":
    unittest.main()

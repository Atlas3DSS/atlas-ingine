#!/usr/bin/env python3
"""CPU unit & mock-server tests for tools/host_cache_mode_policy.py (#829 Requirement 3)."""
from __future__ import annotations

import http.server
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import host_cache_mode_policy as pol  # noqa: E402


def start_mock_worker(
    active_mode: str,
    supported_modes: list[str],
    model_id: str = "inkling-2-mini",
    max_model_len: int = 1048576,
    max_batch: int = 1,
    header_override: str | None = None,
    body_mode_override: str | None = None,
    cached_tokens: int = 0,
):
    received: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send(self, code: int, obj: dict, mode_hdr: str | None = None):
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if mode_hdr is not None:
                self.send_header("X-Ingine-Kv-Cache-Mode", mode_hdr)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            hdr = header_override if header_override is not None else active_mode
            if self.path == "/health":
                self._send(
                    200,
                    {
                        "status": "ok",
                        "model": model_id,
                        "kv_cache_mode": active_mode,
                        "supported_kv_cache_modes": supported_modes,
                        "kv_cache_mode_mutable": False,
                        "max_batch": max_batch,
                        "waiting": 0,
                        "running": False,
                    },
                    mode_hdr=hdr,
                )
            elif self.path == "/v1/models":
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model_id,
                                "object": "model",
                                "max_model_len": max_model_len,
                                "kv_cache_mode": active_mode,
                                "supported_kv_cache_modes": supported_modes,
                                "kv_cache_mode_mutable": False,
                            }
                        ],
                    },
                    mode_hdr=hdr,
                )
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            received.append(
                {
                    "headers": {k.lower(): v for k, v in self.headers.items()},
                    "body": body,
                }
            )
            hdr = header_override if header_override is not None else active_mode
            reported = body_mode_override if body_mode_override is not None else active_mode
            self._send(
                200,
                {
                    "id": "chatcmpl-mock-1",
                    "object": "chat.completion",
                    "model": model_id,
                    "kv_cache_mode": reported,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": f"ok:{reported}"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": max(128, cached_tokens),
                        "completion_tokens": 8,
                        "total_tokens": max(128, cached_tokens) + 8,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    },
                },
                mode_hdr=hdr,
            )

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, f"http://127.0.0.1:{port}", received


class TestHostCacheModePolicy(unittest.TestCase):
    def test_exact_kv_memory_geometry(self):
        # 1M tokens in Q4_0: 9.00 GiB per_pass vs 4.50 GiB last_full -> exact 4.50 GiB savings.
        pp_mib = pol.estimate_full_kv_mib(1048576, "q4_0", "per_pass")
        lf_mib = pol.estimate_full_kv_mib(1048576, "q4_0", "last_full")
        self.assertAlmostEqual(pp_mib / 1024.0, 9.0, places=6)
        self.assertAlmostEqual(lf_mib / 1024.0, 4.5, places=6)
        self.assertAlmostEqual((pp_mib - lf_mib) / 1024.0, 4.5, places=6)

    def test_default_per_pass_and_memory_triggered_last_full(self):
        w_pp = pol.WorkerCapability(
            worker_id="w-pp",
            base_url="http://127.0.0.1:9001",
            model_id="inkling-2-mini",
            active_kv_cache_mode="per_pass",
            supported_kv_cache_modes=("per_pass", "last_full"),
            max_model_len=1048576,
            kv_format="q4_0",
            available_kv_vram_mib=6144.0,  # fits 5.5 GiB headroom-adjusted
        )
        w_lf = pol.WorkerCapability(
            worker_id="w-lf",
            base_url="http://127.0.0.1:9002",
            model_id="inkling-2-mini",
            active_kv_cache_mode="last_full",
            supported_kv_cache_modes=("per_pass", "last_full"),
            max_model_len=1048576,
            kv_format="q4_0",
            available_kv_vram_mib=6144.0,
        )

        # Short context: fits per_pass easily -> MUST default to per_pass.
        req_short = pol.AdmissionRequest(
            task_id="t-short",
            session_id="s-1",
            cache_namespace="tenant-a",
            context_tokens=32768,
            expected_new_tokens=512,
            prefix_sha256="aaa",
        )
        dec_short = pol.select_worker_and_mode([w_pp, w_lf], req_short)
        self.assertTrue(dec_short.admitted)
        self.assertEqual(dec_short.selected_mode, "per_pass")
        self.assertEqual(dec_short.worker_id, "w-pp")
        self.assertEqual(dec_short.effective_cache_namespace, "tenant-a:kv_mode=per_pass")

        # 1M context on 6 GiB KV budget: per_pass needs 9.16 GiB (>5.5 GiB), last_full needs 4.66 GiB (<5.5 GiB) -> routes to w-lf!
        req_1m = pol.AdmissionRequest(
            task_id="t-1m",
            session_id="s-2",
            cache_namespace="tenant-a",
            context_tokens=1048000,
            expected_new_tokens=576,
            prefix_sha256="bbb",
            budgets=pol.AdmissionBudgets(max_nll_delta=0.02),
        )
        dec_1m = pol.select_worker_and_mode([w_pp, w_lf], req_1m)
        self.assertTrue(dec_1m.admitted)
        self.assertEqual(dec_1m.selected_mode, "last_full")
        self.assertEqual(dec_1m.worker_id, "w-lf")
        self.assertEqual(dec_1m.effective_cache_namespace, "tenant-a:kv_mode=last_full")

        # Same 1M context, but task requires exact per_pass quality -> MUST reject (cannot use last_full, per_pass OOM).
        req_exact = pol.AdmissionRequest(
            task_id="t-1m-exact",
            session_id="s-3",
            cache_namespace="tenant-a",
            context_tokens=1048000,
            expected_new_tokens=576,
            prefix_sha256="ccc",
            budgets=pol.AdmissionBudgets(require_exact_per_pass=True),
        )
        dec_exact = pol.select_worker_and_mode([w_pp, w_lf], req_exact)
        self.assertFalse(dec_exact.admitted)
        self.assertIn("quality_budget_requires_exact_per_pass", dec_exact.reason)

    def test_spec_batch_tp_workers_forbid_last_full(self):
        for kwargs in ({"spec_enabled": True}, {"max_batch": 16}, {"tp_size": 2}):
            w = pol.WorkerCapability(
                worker_id="w-restricted",
                base_url="http://127.0.0.1:9003",
                model_id="inkling-2-mini",
                active_kv_cache_mode="per_pass",
                supported_kv_cache_modes=("per_pass", "last_full"),
                **kwargs,
            )
            self.assertFalse(w.supports_mode("last_full"))
            self.assertTrue(w.supports_mode("per_pass"))

    def test_mode_switch_fail_closed_and_allowed_paths(self):
        w_lf = pol.WorkerCapability(
            worker_id="w-lf",
            base_url="http://127.0.0.1:9004",
            model_id="inkling-2-mini",
            active_kv_cache_mode="last_full",
            supported_kv_cache_modes=("per_pass", "last_full"),
            max_model_len=131072,
            kv_format="q4_0",
            available_kv_vram_mib=8192.0,
            prefill_tok_per_s=2000.0,
            decode_tok_per_s=50.0,
        )

        # 1. Raw switch without compatible checkpoint or replay -> BLOCKED!
        req_raw = pol.AdmissionRequest(
            task_id="t-sw-raw",
            session_id="s-sw",
            cache_namespace="ns1",
            context_tokens=8192,
            expected_new_tokens=128,
            prefix_sha256="deadbeef",
            preferred_mode="last_full",
            mode_switch=pol.ModeSwitchContext(prior_kv_cache_mode="per_pass"),
        )
        dec_raw = pol.evaluate_worker_for_mode(w_lf, req_raw, "last_full")
        self.assertFalse(dec_raw.admitted)
        self.assertIn("mode_switch_blocked_no_checkpoint_or_replay", dec_raw.reason)

        # 2. Wrong-mode checkpoint supplied -> BLOCKED!
        bad_ckpt = pol.CheckpointIdentity(
            model_id="inkling-2-mini",
            kv_format="q4_0",
            kv_cache_mode="per_pass",
            cache_namespace="ns1:kv_mode=per_pass",
            prefix_tokens=8192,
            prefix_sha256="deadbeef",
            checkpoint_path="/tmp/bad.ckpt",
        )
        req_bad_ckpt = pol.AdmissionRequest(
            task_id="t-sw-badckpt",
            session_id="s-sw",
            cache_namespace="ns1",
            context_tokens=8192,
            expected_new_tokens=128,
            prefix_sha256="deadbeef",
            preferred_mode="last_full",
            mode_switch=pol.ModeSwitchContext(prior_kv_cache_mode="per_pass", compatible_checkpoint=bad_ckpt),
        )
        dec_bad = pol.evaluate_worker_for_mode(w_lf, req_bad_ckpt, "last_full")
        self.assertFalse(dec_bad.admitted)
        self.assertIn("mode_switch_checkpoint_mode_mismatch", dec_bad.reason)

        # 3. Compatible last_full checkpoint on an HTTP worker WITHOUT matching loaded_checkpoint_digest -> BLOCKED!
        good_ckpt = pol.CheckpointIdentity(
            model_id="inkling-2-mini",
            kv_format="q4_0",
            kv_cache_mode="last_full",
            cache_namespace="ns1:kv_mode=last_full",
            prefix_tokens=8192,
            prefix_sha256="deadbeef",
            checkpoint_path="/tmp/good_lf.ckpt",
        )
        req_good_ckpt = pol.AdmissionRequest(
            task_id="t-sw-goodckpt",
            session_id="s-sw",
            cache_namespace="ns1",
            context_tokens=8192,
            expected_new_tokens=128,
            prefix_sha256="deadbeef",
            preferred_mode="last_full",
            mode_switch=pol.ModeSwitchContext(prior_kv_cache_mode="per_pass", compatible_checkpoint=good_ckpt),
        )
        dec_unloaded = pol.evaluate_worker_for_mode(w_lf, req_good_ckpt, "last_full")
        self.assertFalse(dec_unloaded.admitted)
        self.assertIn("mode_switch_checkpoint_restore_unavailable_require_authorized_replay", dec_unloaded.reason)

        # 3b. Compatible last_full checkpoint on a worker holding that exact verified loaded checkpoint -> ADMITTED!
        w_lf_loaded = pol.WorkerCapability(
            worker_id="w-lf-loaded",
            base_url="http://127.0.0.1:9004",
            model_id="inkling-2-mini",
            active_kv_cache_mode="last_full",
            supported_kv_cache_modes=("per_pass", "last_full"),
            max_model_len=131072,
            kv_format="q4_0",
            available_kv_vram_mib=8192.0,
            prefill_tok_per_s=2000.0,
            decode_tok_per_s=50.0,
            loaded_checkpoint_digest=good_ckpt.identity_digest(),
        )
        dec_good = pol.evaluate_worker_for_mode(w_lf_loaded, req_good_ckpt, "last_full")
        self.assertTrue(dec_good.admitted)
        self.assertEqual(dec_good.mode_switch_action, "compatible_checkpoint")
        self.assertFalse(dec_good.bypass_prefix_cache)
        self.assertEqual(dec_good.checkpoint_identity.kv_cache_mode, "last_full")
        self.assertNotEqual(bad_ckpt.identity_digest(), dec_good.checkpoint_identity.identity_digest())

        # 4. Authorized replay -> ADMITTED with bypass_prefix_cache=True, deadline checked!
        req_replay = pol.AdmissionRequest(
            task_id="t-sw-replay",
            session_id="s-sw",
            cache_namespace="ns1",
            context_tokens=8192,
            expected_new_tokens=128,
            prefix_sha256="deadbeef",
            preferred_mode="last_full",
            budgets=pol.AdmissionBudgets(deadline_seconds=10.0),
            mode_switch=pol.ModeSwitchContext(
                prior_kv_cache_mode="per_pass",
                replay_authorized=True,
                authorized_replay_tokens=8192,
            ),
        )
        dec_replay = pol.evaluate_worker_for_mode(w_lf, req_replay, "last_full")
        self.assertTrue(dec_replay.admitted)
        self.assertEqual(dec_replay.mode_switch_action, "authorized_replay")
        self.assertTrue(dec_replay.bypass_prefix_cache)

        # 5. Authorized replay exceeding deadline -> REJECTED!
        req_tight = pol.AdmissionRequest(
            task_id="t-sw-tight",
            session_id="s-sw",
            cache_namespace="ns1",
            context_tokens=8192,
            expected_new_tokens=128,
            prefix_sha256="deadbeef",
            preferred_mode="last_full",
            budgets=pol.AdmissionBudgets(deadline_seconds=1.0),
            mode_switch=pol.ModeSwitchContext(
                prior_kv_cache_mode="per_pass",
                replay_authorized=True,
                authorized_replay_tokens=8192,
            ),
        )
        dec_tight = pol.evaluate_worker_for_mode(w_lf, req_tight, "last_full")
        self.assertFalse(dec_tight.admitted)
        self.assertIn("deadline_budget_exceeded", dec_tight.reason)

    def test_live_mock_discovery_dispatch_and_sqlite_audit(self):
        srv_pp, url_pp, recv_pp = start_mock_worker("per_pass", ["per_pass", "last_full"])
        srv_lf, url_lf, recv_lf = start_mock_worker("last_full", ["per_pass", "last_full"])
        srv_bad, url_bad, _ = start_mock_worker(
            "last_full",
            ["per_pass", "last_full"],
            body_mode_override="per_pass",  # lies in completion body!
        )
        try:
            w_pp = pol.discover_worker_capability(
                "w-pp", url_pp, available_kv_vram_mib=6144.0, cache_namespace_header="X-Tenant-Scope"
            )
            w_lf = pol.discover_worker_capability(
                "w-lf", url_lf, available_kv_vram_mib=6144.0, cache_namespace_header="X-Tenant-Scope"
            )
            self.assertEqual(w_pp.active_kv_cache_mode, "per_pass")
            self.assertEqual(w_pp.supported_kv_cache_modes, ("per_pass", "last_full"))
            self.assertEqual(w_lf.active_kv_cache_mode, "last_full")

            with tempfile.TemporaryDirectory() as td:
                db = Path(td) / "host_policy.sqlite"
                with pol.HostCacheModeRegistry(db) as reg:
                    # Turn 1: short context -> per_pass worker, mode-scoped namespace header sent.
                    req1 = pol.AdmissionRequest(
                        task_id="task-1",
                        session_id="sess-42",
                        cache_namespace="acct-7",
                        context_tokens=16384,
                        expected_new_tokens=256,
                        prefix_sha256="p1",
                    )
                    out1 = reg.dispatch_chat_completion(
                        [w_pp, w_lf], req1, [{"role": "user", "content": "hi"}]
                    )
                    self.assertTrue(out1["admitted"])
                    self.assertEqual(out1["server_kv_cache_mode"], "per_pass")
                    self.assertEqual(recv_pp[-1]["headers"].get("x-tenant-scope"), "acct-7:kv_mode=per_pass")

                    # Turn 2 on SAME session jumps to 1M context (requires last_full) WITHOUT replay/ckpt -> BLOCKED!
                    req2_blocked = pol.AdmissionRequest(
                        task_id="task-2-blocked",
                        session_id="sess-42",
                        cache_namespace="acct-7",
                        context_tokens=1000000,
                        expected_new_tokens=256,
                        prefix_sha256="p2",
                    )
                    out2 = reg.dispatch_chat_completion(
                        [w_pp, w_lf], req2_blocked, [{"role": "user", "content": "big"}]
                    )
                    self.assertFalse(out2["admitted"])
                    self.assertIn("mode_switch_blocked", out2["decision"]["reason"])

                    # Turn 3 on SAME session with authorized replay -> routed to w_lf with namespace header omitted (fail-closed bypass).
                    req3_replay = pol.AdmissionRequest(
                        task_id="task-3-replay",
                        session_id="sess-42",
                        cache_namespace="acct-7",
                        context_tokens=1000000,
                        expected_new_tokens=256,
                        prefix_sha256="p2",
                        mode_switch=pol.ModeSwitchContext(
                            prior_kv_cache_mode="per_pass",
                            replay_authorized=True,
                            authorized_replay_tokens=1000000,
                        ),
                    )
                    out3 = reg.dispatch_chat_completion(
                        [w_pp, w_lf], req3_replay, [{"role": "user", "content": "big"}]
                    )
                    self.assertTrue(out3["admitted"])
                    self.assertEqual(out3["server_kv_cache_mode"], "last_full")
                    self.assertEqual(out3["decision"]["mode_switch_action"], "authorized_replay")
                    self.assertTrue(out3["decision"]["bypass_prefix_cache"])
                    self.assertNotIn("x-tenant-scope", recv_lf[-1]["headers"])
                    self.assertEqual(reg.get_session_mode("sess-42"), "last_full")

                    # Verify SQLite audit rows recorded everything, including end-to-end bound max_tokens.
                    self.assertEqual(recv_pp[-1]["body"]["max_tokens"], 256)
                    self.assertEqual(recv_lf[-1]["body"]["max_tokens"], 256)
                    conn = sqlite3.connect(str(db))
                    n_disp = conn.execute("SELECT COUNT(*) FROM dispatches").fetchone()[0]
                    n_res = conn.execute("SELECT COUNT(*) FROM results WHERE verified_mode_match = 1").fetchone()[0]
                    disp_max_toks = [r[0] for r in conn.execute("SELECT max_tokens FROM dispatches").fetchall()]
                    conn.close()
                    self.assertEqual(n_disp, 3)
                    self.assertEqual(n_res, 2)
                    self.assertEqual(disp_max_toks, [256, 256, 256])

                    # Conflicting explicit max_tokens vs req.expected_new_tokens -> fails closed with ValueError.
                    with self.assertRaisesRegex(ValueError, "max_tokens mismatch"):
                        reg.dispatch_chat_completion(
                            [w_pp], req1, [{"role": "user", "content": "hi"}], max_tokens=512
                        )

                    # Conflicting server response header vs body -> fail-closed exception.
                    w_bad = pol.discover_worker_capability("w-bad", url_bad, available_kv_vram_mib=6144.0)
                    req_bad = pol.AdmissionRequest(
                        task_id="task-bad",
                        session_id="sess-bad",
                        cache_namespace="acct-9",
                        context_tokens=1024,
                        expected_new_tokens=64,
                        prefix_sha256="pb",
                        preferred_mode="last_full",
                    )
                    with self.assertRaisesRegex(RuntimeError, "kv_cache_mode conflict"):
                        reg.dispatch_chat_completion([w_bad], req_bad, [{"role": "user", "content": "x"}])
        finally:
            for s in (srv_pp, srv_lf, srv_bad):
                s.shutdown()
                s.server_close()

    def test_verified_checkpoint_restore_and_max_tokens_binding(self):
        good_ckpt = pol.CheckpointIdentity(
            model_id="inkling-2-mini",
            kv_format="q4_0",
            kv_cache_mode="last_full",
            cache_namespace="acct-7:kv_mode=last_full",
            prefix_tokens=4096,
            prefix_sha256="ckpt-hash-1",
            checkpoint_path="/mnt/nvme/lf.ckpt",
        )
        srv_zero, url_zero, recv_zero = start_mock_worker(
            "last_full", ["per_pass", "last_full"], cached_tokens=0
        )
        srv_hit, url_hit, recv_hit = start_mock_worker(
            "last_full", ["per_pass", "last_full"], cached_tokens=4096
        )
        try:
            w_zero = pol.discover_worker_capability(
                "w-lf-zero",
                url_zero,
                available_kv_vram_mib=8192.0,
                loaded_checkpoint_digest=good_ckpt.identity_digest(),
            )
            w_hit = pol.discover_worker_capability(
                "w-lf-hit",
                url_hit,
                available_kv_vram_mib=8192.0,
                loaded_checkpoint_digest=good_ckpt.identity_digest(),
            )
            with tempfile.TemporaryDirectory() as td:
                db = Path(td) / "ckpt_verify.sqlite"
                with pol.HostCacheModeRegistry(db) as reg:
                    req_sw = pol.AdmissionRequest(
                        task_id="task-ckpt-sw",
                        session_id="sess-ckpt",
                        cache_namespace="acct-7",
                        context_tokens=4096,
                        expected_new_tokens=77,
                        prefix_sha256="ckpt-hash-1",
                        preferred_mode="last_full",
                        mode_switch=pol.ModeSwitchContext(
                            prior_kv_cache_mode="per_pass",
                            compatible_checkpoint=good_ckpt,
                        ),
                    )
                    # Server returns cached_tokens=0 -> MUST fail closed (never accept as checkpoint transition)!
                    with self.assertRaisesRegex(RuntimeError, "checkpoint restore unverified"):
                        reg.dispatch_chat_completion([w_zero], req_sw, [{"role": "user", "content": "hi"}])
                    self.assertEqual(recv_zero[-1]["body"]["max_tokens"], 77)
                    self.assertIsNone(reg.get_session_mode("sess-ckpt"))

                    # Server proves cached_tokens == 4096 >= prefix_tokens -> succeeds and binds max_tokens=77!
                    out_hit = reg.dispatch_chat_completion([w_hit], req_sw, [{"role": "user", "content": "hi"}])
                    self.assertTrue(out_hit["admitted"])
                    self.assertEqual(out_hit["bound_max_tokens"], 77)
                    self.assertEqual(recv_hit[-1]["body"]["max_tokens"], 77)
                    self.assertEqual(reg.get_session_mode("sess-ckpt"), "last_full")

                    # Replay path where buggy server returns cached_tokens > 0 -> MUST fail closed!
                    req_replay_bug = pol.AdmissionRequest(
                        task_id="task-replay-bug",
                        session_id="sess-replay-bug",
                        cache_namespace="acct-7",
                        context_tokens=4096,
                        expected_new_tokens=55,
                        prefix_sha256="ckpt-hash-2",
                        preferred_mode="last_full",
                        mode_switch=pol.ModeSwitchContext(
                            prior_kv_cache_mode="per_pass",
                            replay_authorized=True,
                            authorized_replay_tokens=4096,
                        ),
                    )
                    with self.assertRaisesRegex(RuntimeError, "replay prefix-cache violation"):
                        reg.dispatch_chat_completion([w_hit], req_replay_bug, [{"role": "user", "content": "hi"}])
        finally:
            for s in (srv_zero, srv_hit):
                s.shutdown()
                s.server_close()


if __name__ == "__main__":
    unittest.main()

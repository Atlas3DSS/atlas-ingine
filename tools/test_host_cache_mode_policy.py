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
    cached_tokens: object = 0,
    omit_cached_tokens: bool = False,
    loop_passes: int | None = None,
    models_loop_passes_override: int | None = None,
    explicit_null_loop_passes: bool = False,
    models_supported_override: list[str] | None = None,
    completion_loop_passes_override: object = ...,
    enforce_expected_loop_passes: bool = True,
    model_family: str | None = None,
    models_family_override: str | None = None,
):
    received: list[dict] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send(self, code: int, obj: dict, mode_hdr: str | None = None, lp_hdr: str | None = None):
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if mode_hdr is not None:
                self.send_header("X-Ingine-Kv-Cache-Mode", mode_hdr)
            if lp_hdr is not None:
                self.send_header("X-Ingine-Loop-Passes", lp_hdr)
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            hdr = header_override if header_override is not None else active_mode
            lp_hdr = str(loop_passes) if loop_passes is not None else None
            if self.path == "/health":
                health_obj = {
                    "status": "ok",
                    "model": model_id,
                    "kv_cache_mode": active_mode,
                    "supported_kv_cache_modes": supported_modes,
                    "kv_cache_mode_mutable": False,
                    "max_batch": max_batch,
                    "waiting": 0,
                    "running": False,
                }
                if explicit_null_loop_passes:
                    health_obj["loop_passes"] = None
                elif loop_passes is not None:
                    health_obj["loop_passes"] = loop_passes
                if model_family is not None:
                    health_obj["model_family"] = model_family
                self._send(200, health_obj, mode_hdr=hdr, lp_hdr=lp_hdr)
            elif self.path == "/v1/models":
                card_obj = {
                    "id": model_id,
                    "object": "model",
                    "max_model_len": max_model_len,
                    "kv_cache_mode": active_mode,
                    "supported_kv_cache_modes": (
                        models_supported_override if models_supported_override is not None else supported_modes
                    ),
                    "kv_cache_mode_mutable": False,
                }
                if explicit_null_loop_passes:
                    card_obj["loop_passes"] = None
                else:
                    card_lp = models_loop_passes_override if models_loop_passes_override is not None else loop_passes
                    if card_lp is not None:
                        card_obj["loop_passes"] = card_lp
                card_fam = models_family_override if models_family_override is not None else model_family
                if card_fam is not None:
                    card_obj["model_family"] = card_fam
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [card_obj],
                    },
                    mode_hdr=hdr,
                    lp_hdr=lp_hdr if models_loop_passes_override is None else None,
                )
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            req_headers = {k.lower(): v for k, v in self.headers.items()}
            received.append(
                {
                    "headers": req_headers,
                    "body": body,
                }
            )
            hdr = header_override if header_override is not None else active_mode
            if enforce_expected_loop_passes and loop_passes is not None:
                exp_lp = req_headers.get("x-ingine-expected-loop-passes")
                if exp_lp is not None and exp_lp != str(loop_passes):
                    self._send(
                        409,
                        {"error": {"message": f"worker loop_passes mismatch: expected={exp_lp}, server={loop_passes}"}},
                        mode_hdr=hdr,
                        lp_hdr=str(loop_passes),
                    )
                    return
            reported = body_mode_override if body_mode_override is not None else active_mode
            details = {} if omit_cached_tokens else {"cached_tokens": cached_tokens}
            ptoks = max(128, cached_tokens) if type(cached_tokens) is int else 128
            resp_obj: dict = {
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
                    "prompt_tokens": ptoks,
                    "completion_tokens": 8,
                    "total_tokens": ptoks + 8,
                    "prompt_tokens_details": details,
                },
            }
            resp_lp_hdr: str | None = None
            if completion_loop_passes_override is not ...:
                if completion_loop_passes_override is not None:
                    resp_obj["loop_passes"] = completion_loop_passes_override
                    resp_lp_hdr = str(completion_loop_passes_override)
            elif loop_passes is not None:
                resp_obj["loop_passes"] = loop_passes
                resp_lp_hdr = str(loop_passes)
            self._send(200, resp_obj, mode_hdr=hdr, lp_hdr=resp_lp_hdr)

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

        # 1. Raw switch without authorized replay -> BLOCKED!
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
        self.assertIn("mode_switch_blocked_no_authorized_replay", dec_raw.reason)

        # 2. HTTP checkpoint-only transition (no runtime HTTP restore receipt exists) -> BLOCKED!
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
        dec_ckpt = pol.evaluate_worker_for_mode(w_lf, req_good_ckpt, "last_full")
        self.assertFalse(dec_ckpt.admitted)
        self.assertIn("mode_switch_checkpoint_restore_unavailable_over_http_require_authorized_replay", dec_ckpt.reason)

        # Verify distinct identity digests across modes.
        pp_ckpt = pol.CheckpointIdentity(
            model_id="inkling-2-mini",
            kv_format="q4_0",
            kv_cache_mode="per_pass",
            cache_namespace="ns1:kv_mode=per_pass",
            prefix_tokens=8192,
            prefix_sha256="deadbeef",
            checkpoint_path="/tmp/good_lf.ckpt",
        )
        self.assertNotEqual(pp_ckpt.identity_digest(), good_ckpt.identity_digest())

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

    def test_replay_requires_observed_int_zero_and_registry_binds_stale_context(self):
        srv_pp, url_pp, _ = start_mock_worker("per_pass", ["per_pass", "last_full"], cached_tokens=0)
        srv_lf_ok, url_lf_ok, recv_ok = start_mock_worker("last_full", ["per_pass", "last_full"], cached_tokens=0)
        srv_lf_null, url_lf_null, _ = start_mock_worker("last_full", ["per_pass", "last_full"], cached_tokens=None)
        srv_lf_omit, url_lf_omit, _ = start_mock_worker(
            "last_full", ["per_pass", "last_full"], omit_cached_tokens=True
        )
        srv_lf_pos, url_lf_pos, _ = start_mock_worker("last_full", ["per_pass", "last_full"], cached_tokens=512)
        try:
            w_pp = pol.discover_worker_capability("w-pp", url_pp, available_kv_vram_mib=8192.0)
            w_ok = pol.discover_worker_capability("w-lf-ok", url_lf_ok, available_kv_vram_mib=8192.0)
            w_null = pol.discover_worker_capability("w-lf-null", url_lf_null, available_kv_vram_mib=8192.0)
            w_omit = pol.discover_worker_capability("w-lf-omit", url_lf_omit, available_kv_vram_mib=8192.0)
            w_pos = pol.discover_worker_capability("w-lf-pos", url_lf_pos, available_kv_vram_mib=8192.0)

            with tempfile.TemporaryDirectory() as td:
                db = Path(td) / "replay_strict.sqlite"
                with pol.HostCacheModeRegistry(db) as reg:
                    # Establish registry-confirmed prior session mode = per_pass.
                    req_init = pol.AdmissionRequest(
                        task_id="t-init",
                        session_id="sess-strict",
                        cache_namespace="acct-7",
                        context_tokens=1024,
                        expected_new_tokens=64,
                        prefix_sha256="h0",
                    )
                    out_init = reg.dispatch_chat_completion([w_pp], req_init, [{"role": "user", "content": "hi"}])
                    self.assertTrue(out_init["admitted"])
                    self.assertEqual(reg.get_session_mode("sess-strict"), "per_pass")

                    # Stale caller context claiming prior_kv_cache_mode="last_full" when registry has "per_pass" -> BLOCKED!
                    req_stale = pol.AdmissionRequest(
                        task_id="t-stale",
                        session_id="sess-strict",
                        cache_namespace="acct-7",
                        context_tokens=4096,
                        expected_new_tokens=77,
                        prefix_sha256="h1",
                        preferred_mode="last_full",
                        mode_switch=pol.ModeSwitchContext(prior_kv_cache_mode="last_full"),
                    )
                    out_stale = reg.dispatch_chat_completion([w_ok], req_stale, [{"role": "user", "content": "hi"}])
                    self.assertFalse(out_stale["admitted"])
                    self.assertIn("stale_session_mode_context", out_stale["decision"]["reason"])
                    self.assertEqual(reg.get_session_mode("sess-strict"), "per_pass")

                    # Authorized replay against servers returning null, missing, or positive cached_tokens -> ALL FAIL CLOSED!
                    req_replay = pol.AdmissionRequest(
                        task_id="t-replay",
                        session_id="sess-strict",
                        cache_namespace="acct-7",
                        context_tokens=4096,
                        expected_new_tokens=77,
                        prefix_sha256="h1",
                        preferred_mode="last_full",
                        mode_switch=pol.ModeSwitchContext(
                            prior_kv_cache_mode="per_pass",
                            replay_authorized=True,
                            authorized_replay_tokens=4096,
                        ),
                    )
                    for bad_worker in (w_null, w_omit, w_pos):
                        with self.assertRaisesRegex(RuntimeError, "replay prefix-cache violation"):
                            reg.dispatch_chat_completion([bad_worker], req_replay, [{"role": "user", "content": "hi"}])
                        # Session mode MUST NOT advance on failed replay verification.
                        self.assertEqual(reg.get_session_mode("sess-strict"), "per_pass")

                    # Authorized replay with observed integer cached_tokens == 0 -> SUCCEEDS and binds max_tokens=77!
                    out_ok = reg.dispatch_chat_completion([w_ok], req_replay, [{"role": "user", "content": "hi"}])
                    self.assertTrue(out_ok["admitted"])
                    self.assertEqual(out_ok["bound_max_tokens"], 77)
                    self.assertEqual(recv_ok[-1]["body"]["max_tokens"], 77)
                    self.assertEqual(reg.get_session_mode("sess-strict"), "last_full")
        finally:
            for s in (srv_pp, srv_lf_ok, srv_lf_null, srv_lf_omit, srv_lf_pos):
                s.shutdown()
                s.server_close()

    def test_loop_passes_discovery_sizing_identity_and_cross_depth_guard(self):
        # Exact #838 depth-aware memory math (reproduces L6 @ 8,192 total tokens, fp8 KV):
        # L6 has 14 full slots (224 MiB), 70 sliding visits (280 MiB rings + 0.546875 MiB ring_pos),
        # and 84 conv visits (6.3984375 MiB) -> 510.9453125 MiB total.
        self.assertAlmostEqual(pol.estimate_full_kv_mib(8192, "fp8", "per_pass", loop_passes=6), 224.0, places=9)
        self.assertAlmostEqual(
            pol.estimate_total_state_mib(8192, "fp8", "per_pass", loop_passes=6), 510.9453125, places=9
        )
        with self.assertRaises(ValueError):
            pol.estimate_full_kv_mib(8192, "fp8", "last_full", loop_passes=6)

        srv_l1, url_l1, _ = start_mock_worker("per_pass", ["per_pass"], loop_passes=1)
        srv_l3, url_l3, _ = start_mock_worker("per_pass", ["per_pass", "last_full"], loop_passes=3)
        srv_l6, url_l6, _ = start_mock_worker("per_pass", ["per_pass"], loop_passes=6)
        srv_legacy, url_legacy, _ = start_mock_worker("per_pass", ["per_pass", "last_full"], loop_passes=None)
        srv_bad_lf, url_bad_lf, _ = start_mock_worker("per_pass", ["per_pass", "last_full"], loop_passes=5)
        srv_mismatch, url_mismatch, _ = start_mock_worker(
            "per_pass", ["per_pass"], loop_passes=3, models_loop_passes_override=6
        )
        try:
            w_l1 = pol.discover_worker_capability("w-l1", url_l1, kv_format="fp8", available_kv_vram_mib=2048.0)
            w_l3 = pol.discover_worker_capability("w-l3", url_l3, kv_format="fp8", available_kv_vram_mib=2048.0)
            w_l6 = pol.discover_worker_capability("w-l6", url_l6, kv_format="fp8", available_kv_vram_mib=2048.0)
            w_legacy = pol.discover_worker_capability("w-legacy", url_legacy, kv_format="fp8", available_kv_vram_mib=2048.0)

            self.assertEqual(w_l1.loop_passes, 1)
            self.assertEqual(w_l3.loop_passes, 3)
            self.assertEqual(w_l6.loop_passes, 6)
            self.assertEqual(w_legacy.loop_passes, 3)  # legacy server omitting loop_passes defaults to 3
            self.assertFalse(w_l1.supports_mode("last_full"))
            self.assertFalse(w_l6.supports_mode("last_full"))
            self.assertTrue(w_l3.supports_mode("last_full"))

            with self.assertRaisesRegex(RuntimeError, "incompatible last_full"):
                pol.discover_worker_capability("w-bad-lf", url_bad_lf)
            with self.assertRaisesRegex(RuntimeError, "inconsistent loop_passes"):
                pol.discover_worker_capability("w-mismatch", url_mismatch)

            # L6 at 8,192 total tokens, fp8 KV, 350 MiB budget MUST BE REJECTED (510.9453125 > 350 MiB).
            req_budget = pol.AdmissionRequest(
                task_id="t-l6-budget",
                session_id="s-l6",
                cache_namespace="tenant-a",
                context_tokens=8064,
                expected_new_tokens=128,
                prefix_sha256="deadbeef",
                budgets=pol.AdmissionBudgets(memory_budget_mib=350.0),
            )
            dec_l6_tight = pol.evaluate_worker_for_mode(w_l6, req_budget, "per_pass")
            self.assertFalse(dec_l6_tight.admitted)
            self.assertIn("memory_budget_exceeded", dec_l6_tight.reason)
            self.assertAlmostEqual(dec_l6_tight.estimated_state_mib, 510.9453125, places=6)

            # L1, L3, L6 under sufficient budget MUST produce distinct checkpoint digests and cache namespaces.
            req_open = pol.AdmissionRequest(
                task_id="t-depth-id",
                session_id="s-id",
                cache_namespace="tenant-a",
                context_tokens=8064,
                expected_new_tokens=128,
                prefix_sha256="deadbeef",
                budgets=pol.AdmissionBudgets(memory_budget_mib=1024.0),
            )
            dec_l1 = pol.evaluate_worker_for_mode(w_l1, req_open, "per_pass")
            dec_l3 = pol.evaluate_worker_for_mode(w_l3, req_open, "per_pass")
            dec_l6 = pol.evaluate_worker_for_mode(w_l6, req_open, "per_pass")
            self.assertTrue(dec_l1.admitted and dec_l3.admitted and dec_l6.admitted)
            self.assertEqual(len({dec_l1.effective_cache_namespace, dec_l3.effective_cache_namespace, dec_l6.effective_cache_namespace}), 3)
            self.assertEqual(
                len(
                    {
                        dec_l1.checkpoint_identity.identity_digest(),
                        dec_l3.checkpoint_identity.identity_digest(),
                        dec_l6.checkpoint_identity.identity_digest(),
                    }
                ),
                3,
            )
            self.assertEqual(dec_l3.effective_cache_namespace, "tenant-a:kv_mode=per_pass")
            self.assertEqual(dec_l1.effective_cache_namespace, "tenant-a:kv_mode=per_pass:loop_passes=1")
            self.assertEqual(dec_l6.effective_cache_namespace, "tenant-a:kv_mode=per_pass:loop_passes=6")

            # Cross-depth session switch (L3 -> L6) in HostCacheModeRegistry MUST fail closed without authorized_replay.
            with tempfile.TemporaryDirectory() as td:
                db = Path(td) / "depth_switch.sqlite"
                with pol.HostCacheModeRegistry(db) as reg:
                    out_l3 = reg.dispatch_chat_completion([w_l3], req_open, [{"role": "user", "content": "hi"}])
                    self.assertTrue(out_l3["admitted"])
                    self.assertEqual(reg.get_session_state("s-id"), ("per_pass", 3))

                    out_l6_blocked = reg.dispatch_chat_completion([w_l6], req_open, [{"role": "user", "content": "hi"}])
                    self.assertFalse(out_l6_blocked["admitted"])
                    self.assertIn("mode_switch_blocked_no_authorized_replay", out_l6_blocked["decision"]["reason"])
                    self.assertEqual(reg.get_session_state("s-id"), ("per_pass", 3))

                    req_l6_replay = pol.AdmissionRequest(
                        task_id="t-depth-replay",
                        session_id="s-id",
                        cache_namespace="tenant-a",
                        context_tokens=8064,
                        expected_new_tokens=128,
                        prefix_sha256="deadbeef",
                        budgets=pol.AdmissionBudgets(memory_budget_mib=1024.0),
                        mode_switch=pol.ModeSwitchContext(
                            prior_kv_cache_mode="per_pass",
                            prior_loop_passes=3,
                            replay_authorized=True,
                            authorized_replay_tokens=8064,
                        ),
                    )
                    out_l6_ok = reg.dispatch_chat_completion([w_l6], req_l6_replay, [{"role": "user", "content": "hi"}])
                    self.assertTrue(out_l6_ok["admitted"])
                    self.assertEqual(out_l6_ok["server_loop_passes"], 6)
                    self.assertEqual(reg.get_session_state("s-id"), ("per_pass", 6))
        finally:
            for s in (srv_l1, srv_l3, srv_l6, srv_legacy, srv_bad_lf, srv_mismatch):
                s.shutdown()
                s.server_close()

    def test_depth_precondition_stale_restart_opaque_scopes_positional_and_discovery_strictness(self):
        # 1. Backwards-compatible positional 9th arg of WorkerCapability MUST bind spec_enabled=True (not loop_passes).
        w_pos = pol.WorkerCapability(
            "w-pos",
            "http://127.0.0.1:9999",
            "inkling-2-mini",
            "per_pass",
            ("per_pass", "last_full"),
            False,
            65536,
            1,
            True,  # 9th positional arg == spec_enabled
        )
        self.assertTrue(w_pos.spec_enabled)
        self.assertEqual(w_pos.loop_passes, 3)
        self.assertFalse(w_pos.supports_mode("last_full"))

        # 2. Opaque caller scopes with literal :loop_passes=<n> suffixes MUST stay distinct at the same worker depth.
        raw_scopes = ("tenant-a", "tenant-a:loop_passes=6", "tenant-a:kv_mode=per_pass:loop_passes=6")
        for depth in (1, 3, 6):
            scoped = [pol.mode_scoped_namespace(s, "per_pass", loop_passes=depth) for s in raw_scopes]
            self.assertEqual(len(set(scoped)), 3, f"Namespace collision at depth {depth}: {scoped}")
            digests = [
                pol.CheckpointIdentity(
                    model_id="inkling-2-mini",
                    kv_format="fp8",
                    kv_cache_mode="per_pass",
                    cache_namespace=ns,
                    prefix_tokens=8064,
                    prefix_sha256="deadbeef",
                    loop_passes=depth,
                ).identity_digest()
                for ns in scoped
            ]
            self.assertEqual(len(set(digests)), 3, f"Digest collision at depth {depth}")

        # 3. Discovery rejects explicit JSON null loop_passes and model-card supported_kv_cache_modes mismatches.
        srv_null_lp, url_null_lp, _ = start_mock_worker("per_pass", ["per_pass"], explicit_null_loop_passes=True)
        srv_card_bad_sup, url_card_bad_sup, _ = start_mock_worker(
            "per_pass", ["per_pass"], loop_passes=5, models_supported_override=["per_pass", "last_full"]
        )
        # 4. Stale retained L1 capability dispatched against server restarted at L6 (HTTP 409 precondition)
        #    AND against a misbehaving server ignoring precondition (returns L6 or omits depth for L1).
        srv_restarted_l6, url_restarted_l6, _ = start_mock_worker("per_pass", ["per_pass"], loop_passes=6)
        srv_lying_l6, url_lying_l6, _ = start_mock_worker(
            "per_pass",
            ["per_pass"],
            loop_passes=6,
            enforce_expected_loop_passes=False,
        )
        srv_omit_for_l1, url_omit_for_l1, _ = start_mock_worker(
            "per_pass",
            ["per_pass"],
            loop_passes=1,
            completion_loop_passes_override=None,
            enforce_expected_loop_passes=False,
        )
        srv_legacy_l3, url_legacy_l3, _ = start_mock_worker("per_pass", ["per_pass", "last_full"], loop_passes=None)
        try:
            with self.assertRaises(ValueError):
                pol.discover_worker_capability("w-null-lp", url_null_lp)
            with self.assertRaisesRegex(RuntimeError, "inconsistent supported_kv_cache_modes"):
                pol.discover_worker_capability("w-card-bad", url_card_bad_sup)

            # Retained L1 capability pointing at URL now serving L6 -> HTTP 409 BEFORE generation, no ok/session update!
            stale_l1_to_l6 = pol.WorkerCapability(
                worker_id="w-stale",
                base_url=url_restarted_l6,
                model_id="inkling-2-mini",
                active_kv_cache_mode="per_pass",
                supported_kv_cache_modes=("per_pass",),
                kv_format="fp8",
                available_kv_vram_mib=2048.0,
                loop_passes=1,
            )
            stale_l1_lying = pol.WorkerCapability(
                worker_id="w-lying",
                base_url=url_lying_l6,
                model_id="inkling-2-mini",
                active_kv_cache_mode="per_pass",
                supported_kv_cache_modes=("per_pass",),
                kv_format="fp8",
                available_kv_vram_mib=2048.0,
                loop_passes=1,
            )
            stale_l1_omit = pol.WorkerCapability(
                worker_id="w-omit",
                base_url=url_omit_for_l1,
                model_id="inkling-2-mini",
                active_kv_cache_mode="per_pass",
                supported_kv_cache_modes=("per_pass",),
                kv_format="fp8",
                available_kv_vram_mib=2048.0,
                loop_passes=1,
            )
            w_legacy = pol.discover_worker_capability(
                "w-legacy", url_legacy_l3, kv_format="fp8", available_kv_vram_mib=2048.0
            )
            req_350 = pol.AdmissionRequest(
                task_id="t-stale",
                session_id="s-stale",
                cache_namespace="tenant-a",
                context_tokens=8064,
                expected_new_tokens=128,
                prefix_sha256="deadbeef",
                budgets=pol.AdmissionBudgets(memory_budget_mib=350.0),
            )
            with tempfile.TemporaryDirectory() as td:
                db = Path(td) / "stale_depth.sqlite"
                with pol.HostCacheModeRegistry(db) as reg:
                    with self.assertRaises(Exception):
                        reg.dispatch_chat_completion([stale_l1_to_l6], req_350, [{"role": "user", "content": "hi"}])
                    self.assertIsNone(reg.get_session_state("s-stale"))

                    with self.assertRaisesRegex(RuntimeError, "loop_passes mismatch"):
                        reg.dispatch_chat_completion([stale_l1_lying], req_350, [{"role": "user", "content": "hi"}])
                    self.assertIsNone(reg.get_session_state("s-stale"))

                    with self.assertRaisesRegex(RuntimeError, "loop_passes omitted"):
                        reg.dispatch_chat_completion([stale_l1_omit], req_350, [{"role": "user", "content": "hi"}])
                    self.assertIsNone(reg.get_session_state("s-stale"))

                    conn = sqlite3.connect(str(db))
                    verified_ok = conn.execute(
                        "SELECT COUNT(*) FROM results WHERE status = 'ok' OR verified_mode_match = 1"
                    ).fetchone()[0]
                    conn.close()
                    self.assertEqual(verified_ok, 0)

                    # Legacy L=3 server omitting depth fields on /health, /v1/models, and completion STILL works!
                    out_leg = reg.dispatch_chat_completion([w_legacy], req_350, [{"role": "user", "content": "hi"}])
                    self.assertTrue(out_leg["admitted"])
                    self.assertEqual(out_leg["server_loop_passes"], 3)
                    self.assertEqual(reg.get_session_state("s-stale"), ("per_pass", 3))
        finally:
            for s in (srv_null_lp, srv_card_bad_sup, srv_restarted_l6, srv_lying_l6, srv_omit_for_l1, srv_legacy_l3):
                s.shutdown()
                s.server_close()

    def test_qwen38_non_inkling_geometry_and_fail_closed(self):
        # 1. Exact Qwen3.8-27B state geometry: 48 GDN layers fixed (151.50 MiB) + 16 BF16 full layers (64 KiB/tok).
        self.assertAlmostEqual(pol.QWEN38_GDN_FIXED_STATE_MIB, 151.5, places=6)
        self.assertAlmostEqual(pol.estimate_qwen38_state_mib(0, "bf16"), 151.5, places=6)
        self.assertAlmostEqual(pol.estimate_qwen38_state_mib(8192, "bf16"), 151.5 + 512.0, places=6)
        self.assertAlmostEqual(
            pol.estimate_qwen38_state_mib(8192, "gdn_f32_conv_bf16_kv_bf16"), 663.5, places=6
        )
        # Inkling-only options/formats must fail closed on Qwen.
        with self.assertRaises(ValueError):
            pol.estimate_qwen38_state_mib(8192, "q4_0")
        with self.assertRaises(ValueError):
            pol.estimate_qwen38_state_mib(8192, "fp8")
        with self.assertRaises(ValueError):
            pol.estimate_qwen38_state_mib(8192, "bf16", kv_cache_mode="last_full")
        with self.assertRaises(ValueError):
            pol.estimate_qwen38_state_mib(8192, "bf16", loop_passes=3)

        # 2. WorkerCapability refuses applying Inkling-2 model_family/geometry to Qwen model IDs.
        with self.assertRaises(ValueError):
            pol.WorkerCapability(
                worker_id="w-bad-fam",
                base_url="http://127.0.0.1:9999",
                model_id="Qwen3.8-27B-NVFP4",
                kv_format="q4_0",
                loop_passes=3,
                model_family="inkling2",
            )
        with self.assertRaises(ValueError):
            pol.WorkerCapability(
                worker_id="w-bad-qwen-lp",
                base_url="http://127.0.0.1:9999",
                model_id="Qwen3.8-27B-NVFP4",
                kv_format="bf16",
                loop_passes=3,
                model_family="qwen3_5",
            )
        with self.assertRaises(ValueError):
            pol.WorkerCapability(
                worker_id="w-bad-qwen-lf",
                base_url="http://127.0.0.1:9999",
                model_id="Qwen3.8-27B-NVFP4",
                active_kv_cache_mode="per_pass",
                supported_kv_cache_modes=("per_pass", "last_full"),
                kv_format="bf16",
                loop_passes=1,
                model_family="qwen3_5",
            )

        # 3. Unknown model family fails closed at admission (never silently applies Inkling state geometry).
        w_unknown = pol.WorkerCapability(
            worker_id="w-unk",
            base_url="http://127.0.0.1:9999",
            model_id="mystery-model",
            active_kv_cache_mode="per_pass",
            supported_kv_cache_modes=("per_pass",),
            kv_format="bf16",
            loop_passes=1,
            model_family="mystery_arch",
        )
        req_small = pol.AdmissionRequest(
            task_id="t-unk",
            session_id="s-unk",
            cache_namespace="tenant-q",
            context_tokens=1024,
            expected_new_tokens=64,
            prefix_sha256="abcd",
            budgets=pol.AdmissionBudgets(memory_budget_mib=4096.0),
        )
        dec_unk = pol.evaluate_worker_for_mode(w_unknown, req_small, "per_pass")
        self.assertFalse(dec_unk.admitted)
        self.assertIn("unsupported_model_family:mystery_arch", dec_unk.reason)

        # 4. Live mock Qwen worker discovery + admission + dispatch:
        #    At 8192 total tokens, Qwen state is 663.50 MiB whereas Inkling L=3 q4_0 would be 236.7 MiB.
        #    A 500 MiB budget MUST reject Qwen (proving Inkling geometry is not used); 700 MiB MUST admit.
        srv_qwen, url_qwen, recv_qwen = start_mock_worker(
            "per_pass",
            ["per_pass"],
            model_id="Qwen3.8-27B-NVFP4",
            max_model_len=8192,
            loop_passes=1,
            model_family="qwen3_5",
        )
        srv_qwen_bad_lp, url_qwen_bad_lp, _ = start_mock_worker(
            "per_pass",
            ["per_pass"],
            model_id="Qwen3.8-27B-NVFP4",
            max_model_len=8192,
            loop_passes=3,
            model_family="qwen3_5",
        )
        srv_qwen_bad_lf, url_qwen_bad_lf, _ = start_mock_worker(
            "per_pass",
            ["per_pass", "last_full"],
            model_id="Qwen3.8-27B-NVFP4",
            max_model_len=8192,
            loop_passes=1,
            model_family="qwen3_5",
        )
        srv_qwen_fam_mis, url_qwen_fam_mis, _ = start_mock_worker(
            "per_pass",
            ["per_pass"],
            model_id="Qwen3.8-27B-NVFP4",
            max_model_len=8192,
            loop_passes=1,
            model_family="qwen3_5",
            models_family_override="inkling2",
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "unsupported Inkling loop_passes"):
                pol.discover_worker_capability("w-q-bad-lp", url_qwen_bad_lp)
            with self.assertRaisesRegex(RuntimeError, "unsupported last_full"):
                pol.discover_worker_capability("w-q-bad-lf", url_qwen_bad_lf)
            with self.assertRaisesRegex(RuntimeError, "inconsistent model_family"):
                pol.discover_worker_capability("w-q-fam-mis", url_qwen_fam_mis)
            with self.assertRaises(ValueError):
                pol.discover_worker_capability("w-q-bad-fmt", url_qwen, kv_format="q4_0")

            w_qwen = pol.discover_worker_capability("w-qwen", url_qwen, available_kv_vram_mib=4096.0)
            self.assertEqual(w_qwen.model_family, "qwen3_5")
            self.assertEqual(w_qwen.kv_format, "bf16")
            self.assertEqual(w_qwen.loop_passes, 1)
            self.assertFalse(w_qwen.supports_mode("last_full"))

            req_500 = pol.AdmissionRequest(
                task_id="t-q500",
                session_id="s-q500",
                cache_namespace="tenant-q",
                context_tokens=8064,
                expected_new_tokens=128,
                prefix_sha256="deadbeef",
                budgets=pol.AdmissionBudgets(memory_budget_mib=500.0),
            )
            dec_500 = pol.evaluate_worker_for_mode(w_qwen, req_500, "per_pass")
            self.assertFalse(dec_500.admitted)
            self.assertAlmostEqual(dec_500.estimated_state_mib, 663.5, places=4)
            self.assertIn("memory_budget_exceeded", dec_500.reason)

            req_700 = pol.AdmissionRequest(
                task_id="t-q700",
                session_id="s-q700",
                cache_namespace="tenant-q",
                context_tokens=8064,
                expected_new_tokens=128,
                prefix_sha256="deadbeef",
                budgets=pol.AdmissionBudgets(memory_budget_mib=700.0),
            )
            with tempfile.TemporaryDirectory() as td:
                db = Path(td) / "qwen_policy.sqlite"
                with pol.HostCacheModeRegistry(db) as reg:
                    out = reg.dispatch_chat_completion([w_qwen], req_700, [{"role": "user", "content": "hi"}])
                    self.assertTrue(out["admitted"])
                    self.assertEqual(out["server_kv_cache_mode"], "per_pass")
                    self.assertEqual(out["server_loop_passes"], 1)
                    self.assertAlmostEqual(out["decision"]["estimated_state_mib"], 663.5, places=4)
                    self.assertEqual(recv_qwen[-1]["headers"].get("x-ingine-expected-loop-passes"), "1")
                    self.assertEqual(reg.get_session_state("s-q700"), ("per_pass", 1))
        finally:
            for s in (srv_qwen, srv_qwen_bad_lp, srv_qwen_bad_lf, srv_qwen_fam_mis):
                s.shutdown()
                s.server_close()


if __name__ == "__main__":
    unittest.main()

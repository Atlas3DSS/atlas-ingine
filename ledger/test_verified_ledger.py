#!/usr/bin/env python3
"""Unit tests for standalone ledger/verified_ledger.py."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verified_ledger import VerifiedLedger, VerifyResult, run_verified_loop


class TestVerifiedLedger(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="test_verified_ledger_")
        self.db = Path(self.tmp.name) / "ledger.sqlite"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_host_gate_and_winner_upgrade(self) -> None:
        with VerifiedLedger(self.db, env_version="v1") as led:
            # Failure must NOT enter winners
            r_fail = led.record_outcome(
                "t1", "compute sum", "bad code",
                VerifyResult(ok=False, error_signature="SyntaxError", error_detail="line 1"),
            )
            self.assertFalse(r_fail["winner_admitted"])
            self.assertTrue(r_fail["failure_logged"])
            self.assertEqual(led.stats()["active_winners"], 0)

            # Passing result enters winners and links prior error into repairs
            r_pass = led.record_outcome(
                "t1", "compute sum", "def f(x): return sum(x)",
                VerifyResult(ok=True, score=1.5, summary="Fixed syntax; 1.5x speedup"),
                recent_error_sig="SyntaxError",
            )
            self.assertTrue(r_pass["winner_admitted"])
            self.assertTrue(r_pass["repair_linked"])
            self.assertEqual(led.stats()["active_winners"], 1)
            self.assertEqual(led.stats()["active_repairs"], 1)

            # Lower score does not overwrite higher winner; higher score does
            self.assertFalse(led.record_winner("t1", "compute sum", "slower", 1.2))
            self.assertTrue(led.record_winner("t1", "compute sum", "faster", 2.8))

    def test_drift_invalidation(self) -> None:
        with VerifiedLedger(self.db, env_version="v1") as led:
            led.set_fact("orders_table", "orders(id, amt)", category="schema")
            led.record_winner("q1", "sum orders", "SELECT SUM(amt) FROM orders", 1.0)
            res = led.bump_env_version("v2", "orders renamed to sales_orders")
            self.assertEqual(res["stale_facts"], 1)
            self.assertEqual(res["stale_winners"], 1)
            st = led.stats()
            self.assertEqual(st["active_facts"], 0)
            self.assertEqual(st["stale_facts"], 1)
            self.assertEqual(st["active_winners"], 0)
            self.assertEqual(st["stale_winners"], 1)
            ctx = led.render_context("sum orders")
            self.assertIn("sales_orders", ctx)
            self.assertNotIn("SELECT SUM(amt) FROM orders", ctx)

    def test_budget_and_verified_loop(self) -> None:
        with VerifiedLedger(self.db, env_version="v1", max_context_chars=700) as led:
            for i in range(10):
                led.set_fact(f"fact_{i}", "x" * 120, category="schema")
            ctx = led.render_context("query")
            self.assertLessEqual(len(ctx), 700)

            calls = {"n": 0}

            def gen(_msgs):
                calls["n"] += 1
                return ("bad" if calls["n"] == 1 else "good_v2"), "stop"

            def ver(cand: str) -> VerifyResult:
                if cand == "bad":
                    return VerifyResult(ok=False, error_signature="BadOutput", error_detail="got bad")
                return VerifyResult(ok=True, score=3.0, summary="passed with good_v2")

            summary = run_verified_loop(led, "loop_task", "produce good", ver, gen, max_iters=3, stop_on_first_pass=True)
            self.assertTrue(summary.solved)
            self.assertEqual(summary.iters_used, 2)
            self.assertEqual(summary.best_score, 3.0)
            self.assertEqual(led.stats()["active_repairs"], 1)

    def test_cli_roundtrip(self) -> None:
        script = str(Path(__file__).resolve().parent / "verified_ledger.py")
        out1 = subprocess.check_output(
            [sys.executable, script, "--db", str(self.db), "add-fact", "--key", "k1", "--value", "v1"], text=True
        )
        self.assertTrue(json.loads(out1)["ok"])
        out2 = subprocess.check_output(
            [sys.executable, script, "--db", str(self.db), "inspect"], text=True
        )
        self.assertEqual(json.loads(out2)["active_facts"], 1)


if __name__ == "__main__":
    unittest.main()

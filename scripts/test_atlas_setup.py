#!/usr/bin/env python3
"""Unit and live-loopback HTTP tests for scripts/atlas_setup.py (stdlib only)."""
from __future__ import annotations

import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import atlas_setup as mod  # noqa: E402


def _make_hw(
    gpus: list[dict],
    *,
    ram_total_gb: float = 64.0,
    ram_free_gb: float = 48.0,
    disk_free_gb: float = 200.0,
    iommu_pt: bool = False,
    binaries: dict[str, str] | None = None,
    checkpoints: list[dict] | None = None,
    other_vendors: list[dict] | None = None,
) -> dict:
    return {
        "os": "Linux",
        "machine": "x86_64",
        "hostname": "test-host",
        "cpu_threads": 32,
        "ram_total_gb": ram_total_gb,
        "ram_free_gb": ram_free_gb,
        "disk_free_gb": disk_free_gb,
        "disk_total_gb": 500.0,
        "gpus": gpus,
        "other_vendors": other_vendors or [],
        "iommu": {"iommu_pt": iommu_pt, "iommu_off": False, "p2p_friendly": iommu_pt},
        "binaries": binaries if binaries is not None else {"install": "/fake/ingine-serve"},
        "checkpoints": checkpoints or [],
        "has_hf_cli": True,
    }


def _gpu(idx: int, name: str, vram_mib: int, cc: str, driver: str = "580.178.04") -> dict:
    return mod.parse_nvidia_smi_csv(f"{idx}, {name}, {vram_mib}, {vram_mib - 512}, {cc}, {driver}, 0000:01:00.0")[0]


class AutoTunerTests(unittest.TestCase):
    def test_parse_nvidia_smi_six_generations(self) -> None:
        csv = "\n".join(
            [
                "0, Tesla P100-PCIE-16GB, 16384, 16000, 6.0, 580.95.01, 0000:01:00.0",
                "1, Tesla V100-PCIE-32GB, 32768, 32000, 7.0, 580.95.01, 0000:02:00.0",
                "2, Tesla T4, 15360, 15000, 7.5, 580.95.01, 0000:03:00.0",
                "3, NVIDIA GeForce RTX 3090, 24576, 24000, 8.6, 580.95.01, 0000:04:00.0",
                "4, NVIDIA GeForce RTX 4090, 24576, 24000, 8.9, 580.95.01, 0000:05:00.0",
                "5, NVIDIA RTX PRO 6000 Blackwell, 97887, 96000, 12.0, 580.95.01, 0000:06:00.0",
            ]
        )
        gpus = mod.parse_nvidia_smi_csv(csv)
        self.assertEqual(len(gpus), 6)
        self.assertEqual([g["arch"] for g in gpus], ["sm_60", "sm_70", "sm_75", "sm_86", "sm_89", "sm_120a"])
        self.assertEqual([g["act_fp16_required"] for g in gpus], [True, True, True, False, False, False])

    def test_6gb_gpu_1m_context_uses_atlas_6g_and_paged_q4(self) -> None:
        hw = _make_hw([_gpu(0, "NVIDIA RTX A2000 6GB", 6138, "8.6")])
        plan = mod.build_plan(hw, tier_id="small", ctx=1048576)
        argv = plan["workers"][0]["argv"]
        self.assertIn("--atlas-6g", argv)
        self.assertIn("--kv-q4", argv)
        self.assertIn("--kv-paged", argv)
        self.assertNotIn("--kv-fp8", argv)
        self.assertNotIn("--spec", argv)
        self.assertFalse(plan["vision"])
        self.assertEqual(plan["array_bounds"]["max_array"], 1)

    def test_24gb_1m_small_runs_resident_q4_without_mutually_exclusive_kv_flags(self) -> None:
        hw = _make_hw([_gpu(0, "NVIDIA GeForce RTX 4090", 24576, "8.9")])
        plan = mod.build_plan(hw, tier_id="small", ctx=1048576)
        argv = plan["workers"][0]["argv"]
        self.assertIn("--dense-fp8", argv)
        self.assertIn("--kv-q4", argv)
        self.assertNotIn("--kv-fp8", argv)
        self.assertNotIn("--kv-paged", argv)
        self.assertLess(plan["estimated_vram_per_gpu_gb"], 21.0)

    def test_96gb_blackwell_batch_disables_spec_and_last_full_cleanly(self) -> None:
        hw = _make_hw([_gpu(0, "NVIDIA RTX PRO 6000 Blackwell", 97887, "12.0")])
        plan = mod.build_plan(hw, tier_id="large", ctx=65536, array_count=4, layout="batch", kv_cache_mode="last_full")
        argv = plan["workers"][0]["argv"]
        self.assertEqual(plan["layout"], "batch")
        self.assertIn("--max-batch", argv)
        self.assertIn("4", argv)
        self.assertNotIn("--spec", argv)
        self.assertEqual(plan["kv_cache_mode"], "per_pass")
        self.assertNotIn("last_full", argv)

    def test_last_full_b1_non_spec_valid(self) -> None:
        hw = _make_hw([_gpu(0, "NVIDIA GeForce RTX 3090", 24576, "8.6")])
        plan = mod.build_plan(hw, tier_id="small", ctx=262144, array_count=1, kv_cache_mode="last_full", spec=True)
        argv = plan["workers"][0]["argv"]
        self.assertIn("--kv-cache-mode", argv)
        self.assertIn("last_full", argv)
        self.assertNotIn("--spec", argv)

    def test_multi_v100_tp4_selects_fp16_build_and_warns_without_iommu_pt(self) -> None:
        gpus = [_gpu(i, "Tesla V100-PCIE-32GB", 32768, "7.0") for i in range(4)]
        hw = _make_hw(gpus, iommu_pt=False, binaries={})
        plan = mod.build_plan(hw, tier_id="large", ctx=65536, array_count=4, layout="tp")
        self.assertEqual(plan["layout"], "tp")
        self.assertEqual(plan["tp_size"], 4)
        self.assertEqual(plan["workers"][0]["cuda_visible_devices"], "0,1,2,3")
        self.assertIn("-DINGINE_TP_SIZE=4", plan["build_command"])
        self.assertIn("-DINGINE_ACT_FP16=ON", plan["build_command"])
        self.assertTrue(any("iommu=pt" in w for w in plan["warnings"]))

    def test_multi_gpu_dp_array_assigns_distinct_ports_and_devices(self) -> None:
        gpus = [
            _gpu(0, "NVIDIA GeForce RTX 3090", 24576, "8.6"),
            _gpu(1, "NVIDIA RTX PRO 6000 Blackwell", 97887, "12.0"),
        ]
        hw = _make_hw(gpus, iommu_pt=True)
        plan = mod.build_plan(hw, tier_id="large", ctx=65536, array_count=2, layout="dp", port=8100)
        self.assertEqual(plan["layout"], "dp")
        self.assertEqual(len(plan["workers"]), 2)
        self.assertEqual([w["port"] for w in plan["workers"]], [8100, 8101])
        self.assertEqual({w["cuda_visible_devices"] for w in plan["workers"]}, {"0", "1"})

    def test_cpu_fallback_and_multivendor_notice(self) -> None:
        hw = _make_hw([], binaries={}, other_vendors=[{"vendor": "amd_rocm", "detail": "RX 7900 XTX", "status": "staged (#835)"}])
        plan = mod.build_plan(hw)
        self.assertEqual(plan["layout"], "cpu_fake")
        self.assertIn("--fake", plan["workers"][0]["argv"])
        self.assertTrue(any("amd_rocm" in w for w in plan["warnings"]))

    def test_local_checkpoint_scanner_detects_mtp_and_tier(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "Inkling-2-Mini-exl3-9g"
            ckpt.mkdir()
            (ckpt / "config.json").write_text("{}", encoding="utf-8")
            (ckpt / "model-00001-of-00001.safetensors").write_bytes(b"\x00" * 16)
            (ckpt / "mtp.safetensors").write_bytes(b"\x00" * 16)
            info = mod.inspect_checkpoint_dir(ckpt)
            self.assertIsNotNone(info)
            assert info is not None
            self.assertTrue(info["has_mtp"])
            self.assertEqual(info["inferred_tier"], "small")


class HttpUiAndProxyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.received_chat: list[dict] = []

        class MockOpenAI(http.server.BaseHTTPRequestHandler):
            outer = self

            def log_message(self, fmt: str, *args) -> None:
                return

            def do_GET(self) -> None:
                if self.path == "/health":
                    body = b'{"status":"ok"}'
                elif self.path == "/v1/models":
                    body = b'{"object":"list","data":[{"id":"inkling-2-mini"}]}'
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n).decode("utf-8"))
                self.outer.received_chat.append(req)
                resp = json.dumps(
                    {
                        "id": "chatcmpl-mock",
                        "choices": [{"message": {"role": "assistant", "content": "Vision & text received!"}}],
                        "usage": {"completion_tokens": 12},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        self.mock_openai = http.server.ThreadingHTTPServer(("127.0.0.1", 0), MockOpenAI)
        self.mock_port = self.mock_openai.server_address[1]
        threading.Thread(target=self.mock_openai.serve_forever, daemon=True).start()

        hw = _make_hw([_gpu(0, "NVIDIA RTX PRO 6000 Blackwell", 97887, "12.0")])
        self.ctrl = mod.AtlasSetupController(ROOT, hw=hw)
        handler = mod.make_http_handler(self.ctrl, "127.0.0.1", self.mock_port)
        self.ui_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.ui_port = self.ui_server.server_address[1]
        threading.Thread(target=self.ui_server.serve_forever, daemon=True).start()

    def tearDown(self) -> None:
        self.ui_server.shutdown()
        self.ui_server.server_close()
        self.mock_openai.shutdown()
        self.mock_openai.server_close()
        self.ctrl.stop_all()

    def _req(self, path: str, body: dict | None = None) -> tuple[int, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        req = urllib.request.Request(f"http://127.0.0.1:{self.ui_port}{path}", data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read().decode("utf-8")
                ct = resp.headers.get("Content-Type", "")
                return resp.status, json.loads(raw) if "json" in ct else raw
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_ui_detect_plan_permission_gate_and_multimodal_chat(self) -> None:
        code, html_body = self._req("/")
        self.assertEqual(code, 200)
        self.assertIn("ATLAS INGINE", html_body)
        self.assertIn("circle-btn", html_body)

        code, det = self._req("/api/detect")
        self.assertEqual(code, 200)
        self.assertEqual(det["hardware"]["gpus"][0]["arch"], "sm_120a")

        code, plan = self._req("/api/plan", {"tier": "large", "ctx": 262144, "array_count": 1, "mode": "api"})
        self.assertEqual(code, 200)
        self.assertEqual(plan["ctx"], 262144)

        # Permission gate MUST reject unapproved download and launch with HTTP 403.
        code, err = self._req("/api/download", {"approved": False, "tier": "large"})
        self.assertEqual(code, 403)
        self.assertIn("approval", err["error"].lower())

        code, err = self._req("/api/launch", {"approved": False, "tier": "large"})
        self.assertEqual(code, 403)
        self.assertIn("approval", err["error"].lower())

        # Client-supplied forged argv must be ignored; server recomputes plan (missing weights -> 400).
        code, err = self._req(
            "/api/launch",
            {
                "approved": True,
                "tier": "large",
                "plan": {"binary_ready": True, "needs_download": False, "workers": [{"argv": ["/bin/echo", "pwn"]}]},
            },
        )
        self.assertEqual(code, 400)
        self.assertIn("missing", err["error"].lower())

        # Status probes our mock OpenAI server on loopback.
        code, st = self._req("/api/status")
        self.assertEqual(code, 200)
        self.assertEqual(st["state"], "ready")

        # Multimodal chat proxy rejects non-loopback endpoints and forwards loopback multimodal payloads.
        code, bad_ep = self._req("/api/chat", {"endpoint_url": "http://example.com:8000/v1", "messages": []})
        self.assertEqual(code, 400)

        code, chat = self._req(
            "/api/chat",
            {
                "endpoint_url": f"http://127.0.0.1:{self.mock_port}/v1",
                "model": "inkling-2-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                            {"type": "text", "text": "What is in this frame?"},
                        ],
                    }
                ],
            },
        )
        self.assertEqual(code, 200)
        self.assertEqual(chat["choices"][0]["message"]["content"], "Vision & text received!")
        self.assertEqual(len(self.received_chat), 1)
        self.assertEqual(self.received_chat[0]["messages"][0]["content"][0]["type"], "image_url")


class CliSmokeTests(unittest.TestCase):
    def test_cli_plan_json_and_setup_help(self) -> None:
        res = subprocess.run(
            [sys.executable, "-B", str(SCRIPTS_DIR / "atlas_setup.py"), "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        data = json.loads(res.stdout)
        self.assertIn("hardware", data)
        self.assertIn("plan", data)

        help_res = subprocess.run(
            ["bash", str(ROOT / "setup.sh"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(help_res.returncode, 0, help_res.stderr)
        self.assertIn("--auto", help_res.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)

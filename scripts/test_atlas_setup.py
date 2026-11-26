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
from unittest import mock
import urllib.error
import urllib.request
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import atlas_setup as mod  # noqa: E402


def _write_test_shard(path: Path, tensor: str = "w") -> None:
    header = json.dumps({tensor: {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(1))


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


def _gpu(
    idx: int,
    name: str,
    vram_mib: int,
    cc: str,
    driver: str = "580.178.04",
    free_mib: int | None = None,
) -> dict:
    free = (vram_mib - 512) if free_mib is None else free_mib
    return mod.parse_nvidia_smi_csv(f"{idx}, {name}, {vram_mib}, {free}, {cc}, {driver}, 0000:01:00.0")[0]


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
            (ckpt / "tokenizer.json").write_text("{}", encoding="utf-8")
            (ckpt / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
            _write_test_shard(ckpt / "model.safetensors")
            (ckpt / "mtp.safetensors").write_bytes(b"\x00" * 16)
            info = mod.inspect_checkpoint_dir(ckpt)
            self.assertIsNotNone(info)
            assert info is not None
            self.assertTrue(info["has_mtp"])
            self.assertTrue(info["usable_engine"])
            self.assertTrue(info["complete_fake_assets"])
            self.assertEqual(info["inferred_tier"], "small")
            ok, missing = mod.verify_checkpoint_dir(ckpt, require_weights=True)
            self.assertTrue(ok)
            self.assertEqual(missing, [])

    def test_partial_checkpoint_and_missing_index_shard_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            # Partial dir with only config + mtp (no model shard, no tokenizer/template)
            partial = Path(td) / "Inkling-2-Mini-partial"
            partial.mkdir()
            (partial / "config.json").write_text("{}", encoding="utf-8")
            (partial / "mtp.safetensors").write_bytes(b"\x00" * 16)
            ok, missing = mod.verify_checkpoint_dir(partial, require_weights=True)
            self.assertFalse(ok)
            self.assertIn("tokenizer.json", missing)
            self.assertIn("chat_template.jinja", missing)
            self.assertIn("model.safetensors or model.safetensors.index.json", missing)

            # Indexed checkpoint missing shard 2
            indexed = Path(td) / "Inkling-2-Mini-indexed"
            indexed.mkdir()
            (indexed / "config.json").write_text("{}", encoding="utf-8")
            (indexed / "tokenizer.json").write_text("{}", encoding="utf-8")
            (indexed / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
            (indexed / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}),
                encoding="utf-8",
            )
            (indexed / "model-00001.safetensors").write_bytes(b"\x00" * 16)
            ok2, missing2 = mod.verify_checkpoint_dir(indexed, require_weights=True)
            self.assertFalse(ok2)
            self.assertIn("model-00002.safetensors", missing2)

            # Fake assets alone suffice for CPU --fake mode
            ok_fake, _ = mod.verify_checkpoint_dir(indexed, require_weights=False)
            self.assertTrue(ok_fake)

    def test_empty_vram_and_truncated_shards(self) -> None:
        gpu = mod.parse_nvidia_smi_csv("0, RTX 3090, 24576, 0, 8.6, 580.95, 0000:01:00.0")[0]
        plan = mod.build_plan(_make_hw([gpu]), tier_id="small")
        self.assertFalse(plan["hardware_ready"])
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "model.safetensors"
            _write_test_shard(shard)
            self.assertTrue(mod._shard_complete(shard, ["w"]))
            self.assertFalse(mod._shard_complete(shard, ["other"]))
            shard.write_bytes(shard.read_bytes()[:-1])
            self.assertFalse(mod._shard_complete(shard, ["w"]))

    def test_free_vram_controls_sizing_and_per_gpu_dp_cap(self) -> None:
        # 24 GB card with only 9.5 GB free must size to free VRAM (--atlas-6g) and warn.
        busy_3090 = _gpu(0, "NVIDIA GeForce RTX 3090", 24576, "8.6", free_mib=9728)
        hw_busy = _make_hw([busy_3090])
        plan_busy = mod.build_plan(hw_busy, tier_id="small", ctx=65536)
        self.assertIn("--atlas-6g", plan_busy["workers"][0]["argv"])
        self.assertTrue(any("free VRAM" in w for w in plan_busy["warnings"]))

        # Heterogeneous DP: PRO 6000 (95 GB free) + RTX 3090 (23.5 GB free) with array=4 Large:
        # 3090 fits at most 1 Large worker; PRO 6000 holds the remaining workers (never 2 on 3090!).
        gpus = [
            _gpu(0, "NVIDIA GeForce RTX 3090", 24576, "8.6", free_mib=24064),
            _gpu(1, "NVIDIA RTX PRO 6000 Blackwell", 97887, "12.0", free_mib=97280),
        ]
        hw_het = _make_hw(gpus, binaries={"install": "/fake/install/ingine-serve"})
        plan_dp = mod.build_plan(hw_het, tier_id="large", ctx=65536, array_count=4, layout="dp")
        self.assertEqual(plan_dp["layout"], "dp")
        self.assertEqual(len(plan_dp["workers"]), 4)
        gpu0_workers = [w for w in plan_dp["workers"] if w["cuda_visible_devices"] == "0"]
        gpu1_workers = [w for w in plan_dp["workers"] if w["cuda_visible_devices"] == "1"]
        self.assertEqual(len(gpu0_workers), 1)
        self.assertEqual(len(gpu1_workers), 3)

    def test_mixed_sm_dp_uses_per_gpu_binaries_and_flags(self) -> None:
        gpus = [
            _gpu(0, "Tesla V100-PCIE-32GB", 32768, "7.0", free_mib=32256),
            _gpu(1, "NVIDIA GeForce RTX 3090", 24576, "8.6", free_mib=24064),
        ]
        hw = _make_hw(
            gpus,
            binaries={"sm70": "/fake/sm70/ingine-serve", "install": "/fake/install/ingine-serve"},
        )
        plan = mod.build_plan(hw, tier_id="large", ctx=262144, array_count=2, layout="dp")
        by_dev = {w["cuda_visible_devices"]: w for w in plan["workers"]}
        self.assertEqual(by_dev["0"]["binary_path"], "/fake/sm70/ingine-serve")
        self.assertEqual(by_dev["1"]["binary_path"], "/fake/install/ingine-serve")
        # 32 GB V100 uses --kv-fp8 at 262K; 24 GB 3090 uses --kv-q4 at 262K.
        self.assertIn("--kv-fp8", by_dev["0"]["argv"])
        self.assertIn("--kv-q4", by_dev["1"]["argv"])

    def test_tp_selects_actual_group_before_arch_and_flags(self) -> None:
        # 1x PRO 6000 (sm_120a) + 2x V100 (sm_70): TP=2 MUST pick the 2x V100 group and build for sm_70 FP16!
        gpus = [
            _gpu(0, "NVIDIA RTX PRO 6000 Blackwell", 97887, "12.0", free_mib=96000),
            _gpu(1, "Tesla V100-PCIE-32GB", 32768, "7.0", free_mib=32000),
            _gpu(2, "Tesla V100-PCIE-32GB", 32768, "7.0", free_mib=32000),
        ]
        hw = _make_hw(gpus, binaries={})
        plan = mod.build_plan(hw, tier_id="large", ctx=65536, array_count=2, layout="tp")
        self.assertEqual(plan["layout"], "tp")
        self.assertEqual(plan["workers"][0]["cuda_visible_devices"], "1,2")
        self.assertEqual(plan["workers"][0]["arch"], "sm_70")
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=70", plan["build_command"])
        self.assertIn("-DINGINE_ACT_FP16=ON", plan["build_command"])
        self.assertIn("-DINGINE_TP_SIZE=2", plan["build_command"])

    def test_legacy_and_arch_specific_binary_matching_and_repo_root_presets(self) -> None:
        # Turing sm_75 MUST NOT reuse sm70-only binary; must require legacy-fp16.
        hw_t4 = _make_hw([_gpu(0, "Tesla T4", 15360, "7.5")], binaries={"sm70": "/fake/sm70/ingine-serve"})
        plan_t4 = mod.build_plan(hw_t4, tier_id="small", ctx=65536)
        self.assertFalse(plan_t4["binary_ready"])
        self.assertIn("(cd native && cmake --preset legacy-fp16", plan_t4["build_command"])

        # RTX 3090 sm_86 MUST NOT reuse release (sm_120a-only) binary.
        hw_3090 = _make_hw([_gpu(0, "NVIDIA GeForce RTX 3090", 24576, "8.6")], binaries={"release": "/fake/release/ingine-serve"})
        plan_3090 = mod.build_plan(hw_3090, tier_id="large", ctx=65536)
        self.assertFalse(plan_3090["binary_ready"])
        self.assertIn("bash scripts/install.sh --build-only", plan_3090["build_command"])

        # A100 sm_80 is not in install preset (86;89;120a) -> explicit sm_80 build command.
        hw_a100 = _make_hw([_gpu(0, "NVIDIA A100-PCIE-80GB", 81920, "8.0")], binaries={"install": "/fake/install/ingine-serve"})
        plan_a100 = mod.build_plan(hw_a100, tier_id="large", ctx=262144)
        self.assertFalse(plan_a100["binary_ready"])
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=80", plan_a100["build_command"])

        # CPU --fake needs tokenizer+template assets and uses repo-root-valid (cd native && cmake --preset cpu-dev ...).
        hw_cpu = _make_hw([], binaries={})
        plan_cpu = mod.build_plan(hw_cpu)
        self.assertTrue(plan_cpu["needs_download"])
        self.assertIn("tokenizer.json", plan_cpu["download_files"])
        self.assertIn("chat_template.jinja", plan_cpu["download_files"])
        self.assertIn("(cd native && cmake --preset cpu-dev", plan_cpu["build_command"])

        # Verify every preset referenced in build_command exists in native/CMakePresets.json!
        presets_json = json.loads((ROOT / "native" / "CMakePresets.json").read_text(encoding="utf-8"))
        valid_presets = {p["name"] for p in presets_json["configurePresets"]}
        for p_name in ("cpu-dev", "sm70", "legacy-fp16", "install"):
            self.assertIn(p_name, valid_presets)


class SetupProcessTests(unittest.TestCase):
    def test_approved_cpu_download_requests_only_displayed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller = mod.AtlasSetupController()
            plan = mod.build_plan(_make_hw([]), model_dir=tmp)
            self.assertTrue(plan["needs_download"])
            self.assertTrue(plan["download_files"])
            with mock.patch.object(mod.shutil, "which", return_value="/bin/echo"):
                controller.start_download(plan, approved=True)
            try:
                proc = controller._download_proc
                self.assertIsNotNone(proc)
                self.assertEqual(proc.wait(timeout=3), 0)
                self.assertEqual(proc.args[3:-2], plan["download_files"])
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    controller.launch_workers(dict(plan, binary_ready=True), approved=True)
            finally:
                controller.stop_all()

    def test_worker_replacement_and_partial_launch_are_reaped(self) -> None:
        controller = mod.AtlasSetupController()
        worker = dict(worker_index=0, cuda_visible_devices="", host="127.0.0.1", port=1,
                      base_url="http://127.0.0.1:1/v1", shell="test worker",
                      argv=[sys.executable, "-c", "import time; time.sleep(30)"])
        plan = dict(binary_ready=True, needs_download=False, workers=[worker])
        children = []
        popen = subprocess.Popen

        def spawn(*args, **kwargs):
            proc = popen(*args, **kwargs)
            children.append(proc)
            return proc

        try:
            with mock.patch.object(mod.subprocess, "Popen", side_effect=spawn):
                controller.launch_workers(plan, approved=True)
                self.assertIsNone(children[0].poll())
                failing = dict(worker, worker_index=1, argv=["/does-not-exist/ingine-serve"])
                with self.assertRaises(FileNotFoundError):
                    controller.launch_workers(dict(plan, workers=[worker, failing]), approved=True)
            self.assertEqual(len(children), 2)
            self.assertTrue(all(proc.poll() is not None for proc in children))
            self.assertEqual(controller._worker_procs, [])
        finally:
            controller.stop_all()


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

        code, _ = self._req("/api/download", {"approved": "false", "tier": "large"})
        self.assertEqual(code, 403)
        code, _ = self._req("/api/launch", {"approved": True, "plan_id": "superseded", "tier": "large"})
        self.assertEqual(code, 409)

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
    def test_bootstrap_forwards_auto_arguments_from_an_existing_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "-C", tmp, "init", "-q"], check=True)
            subprocess.run(["git", "-C", tmp, "remote", "add", "origin",
                            "https://github.com/Atlas3DSS/atlas-ingine"], check=True)
            (Path(tmp) / "setup.sh").write_text("printf 'arg=%s\\n' \"$@\"\n")
            result = subprocess.run(["bash", str(ROOT / "get.sh"), "--auto", "--plan"],
                                    env={**os.environ, "INGINE_DIR": tmp}, stdin=subprocess.DEVNULL,
                                    start_new_session=True, capture_output=True, text=True, check=True)
            self.assertIn("arg=--auto\narg=--plan\n", result.stdout)

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

        inst_res = subprocess.run(
            ["bash", str(ROOT / "scripts" / "install.sh"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(inst_res.returncode, 0, inst_res.stderr)
        self.assertIn("--build-only", inst_res.stdout)

        # Verify repo-root preset listing works via `(cd native && cmake --list-presets)`.
        preset_res = subprocess.run(
            ["bash", "-c", "(cd native && cmake --list-presets)"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(preset_res.returncode, 0, preset_res.stderr)
        for p_name in ("install", "sm70", "legacy-fp16", "cpu-dev"):
            self.assertIn(f'"{p_name}"', preset_res.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)

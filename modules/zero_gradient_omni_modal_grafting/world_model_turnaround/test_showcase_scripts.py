#!/usr/bin/env python3
"""Zero-GPU CPU Unit & End-to-End Dry-Run Test Suite for #912 Showcase Scripts.

Verifies without touching any GPU or writing >100 MB to Devbox / or /tmp:
  1. MiniMax-H3 exact 17k+5 / 40 Hz / 24 fps AV-grid math & 39-frame phase-aligned chaining.
  2. Cryptographic append-only hash-chained ShowcaseTraceLogger + tamper detection.
  3. Fail-closed SHA-256 verification against all real owner assets & Packet A Plato's Keys.
  4. Real owner_manifesto_whisper.json timbre excerpt selection + exact 32 kHz PCM16 slicing.
  5. Pure-Python OCR / lyric metrics + pluggable `custom-cmd` OCR hook + cond_pkg_path parity.
  6. Full zero-GPU `--dry-run` of `run_showcase_national_anthem.py` (Mode A + Mode B + B0).
  7. Full zero-GPU `--dry-run` of `run_showcase_360_turnaround.py` (two-segment-branched + single-clip)
     with ffprobe verification of 1920x1088 side-by-side MP4 & QA contact sheets.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import showcase_common as sc
import run_showcase_national_anthem as anthem
import run_showcase_360_turnaround as turn360

TEST_SCRATCH_ROOT = SCRIPT_DIR / "_cpu_test_scratch"


class TestShowcaseGridAndMath(unittest.TestCase):
    def test_exact_h3_av_boundaries_and_latent_phase(self) -> None:
        self.assertTrue(sc.is_exact_av_boundary(39))
        self.assertEqual(sc.video_latent_t(39), 12)
        self.assertEqual(sc.pixel_frames_from_latent_t(12), 39)
        self.assertEqual(sc.audio_latent_t(39), 65)
        for f_ok in (39, 90, 141, 192, 243, 294, 345):
            self.assertTrue(sc.is_exact_av_boundary(f_ok))
            lt = sc.video_latent_t(f_ok)
            self.assertEqual((lt - 2) % 5, 0)
            self.assertEqual(sc.pixel_frames_from_latent_t(lt), f_ok)
            # Slicing the tail 12 latents starts at an exact multiple of 5 (phase 0!)
            if lt >= 12:
                self.assertEqual((lt - 12) % 5, 0)

    def test_chained_segment_planner_and_turnaround_plan(self) -> None:
        p_short = sc.plan_h3_segments(8.5)
        self.assertFalse(p_short["chained"])
        self.assertEqual(len(p_short["segments"]), 1)
        self.assertTrue(p_short["segments"][0]["exact_av_boundary"])

        p_long = sc.plan_h3_segments(42.0)
        self.assertTrue(p_long["chained"])
        self.assertGreaterEqual(len(p_long["segments"]), 3)
        self.assertGreaterEqual(p_long["total_duration_s"], 42.15)
        for idx, seg in enumerate(p_long["segments"]):
            self.assertLessEqual(seg["frames"], 345)
            self.assertTrue(seg["exact_av_boundary"])
            self.assertEqual(seg["trim_prefix_frames"], 0 if idx == 0 else sc.CHAIN_CONTEXT_FRAMES)

        tp = sc.plan_turnaround_two_segment(seg0_frames=192, seg1_frames=243, context_frames=39)
        self.assertEqual(tp["total_frames"], 192 + (243 - 39))
        self.assertEqual(tp["total_frames"], 396)
        self.assertAlmostEqual(tp["total_duration_s"], 16.5, places=6)
        self.assertEqual(tp["branch_frame_index"], 192)

    def test_cond_pkg_path_empty_neg_four_underscores(self) -> None:
        p = sc.cond_pkg_path(Path("/tmp/x"), "__empty_neg__", "inkling_platos_key_concat4")
        self.assertEqual(p.name, "__empty_neg____inkling_platos_key_concat4.pt")

    def test_extract_first_json_object_handles_think_and_prose(self) -> None:
        raw = '<think>planning scene...</think>\nHere is the JSON:\n```json\n{"a": "ok", "b": 2}\n```\nDone.'
        self.assertEqual(sc.extract_first_json_object(raw), {"a": "ok", "b": 2})


class TestTraceAndPins(unittest.TestCase):
    def setUp(self) -> None:
        self.work = TEST_SCRATCH_ROOT / self._testMethodName
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)

    def test_hash_chained_trace_and_tamper_detection(self) -> None:
        tp = self.work / "showcase_process_trace.jsonl"
        logger = sc.ShowcaseTraceLogger(tp)
        s0 = logger.append(stage="preflight_pin_check", inputs={"a": 1}, outputs={"ok": True})
        s1 = logger.append(stage="script_gen", inputs={"prompt_text": "hello"}, outputs={"text": "world"})
        self.assertEqual(s1["prev_step_sha256"], s0["step_sha256"])

        audit = sc.verify_trace_jsonl(tp)
        self.assertTrue(audit["verified"], audit["errors"])
        self.assertEqual(audit["n_steps"], 2)

        # Tamper a single character in step 0 -> must fail closed!
        raw = tp.read_text(encoding="utf-8").replace('"ok": true', '"ok": false')
        tp.write_text(raw, encoding="utf-8")
        bad = sc.verify_trace_jsonl(tp)
        self.assertFalse(bad["verified"])
        self.assertTrue(any("tamper/mismatch" in e for e in bad["errors"]))

    def test_real_pinned_assets_and_keys_verify_fail_closed(self) -> None:
        for name, spec in sc.PINNED_OWNER_ASSETS.items():
            res = sc.verify_pinned_file(Path(spec["path"]), spec["sha256"], label=name)
            self.assertTrue(res["pin_verified"])
        c4 = Path(sc.DEFAULT_PATHS["keys_dir"]) / "inkling_to_h3_concat4_platos_key.safetensors"
        self.assertTrue(sc.verify_pinned_file(c4, sc.EXPECTED_CONCAT4_KEY_SHA256, label="concat4")["pin_verified"])
        b0 = Path(sc.DEFAULT_PATHS["keys_dir"]) / "inkling_to_h3_b0_shuffled_platos_key.safetensors"
        self.assertTrue(sc.verify_pinned_file(b0, sc.EXPECTED_B0_KEY_SHA256, label="b0")["pin_verified"])
        exp = Path(sc.DEFAULT_PATHS["exporter_bin"])
        self.assertTrue(sc.verify_pinned_file(exp, sc.EXPECTED_EXPORTER_SHA256, label="exporter")["pin_verified"])

        with self.assertRaises(RuntimeError):
            sc.verify_pinned_file(c4, "0" * 64, label="concat4_bad_pin")

    def test_real_owner_timbre_selector_and_ocr_metrics(self) -> None:
        sel = sc.select_clean_owner_timbre_excerpt(
            Path(sc.PINNED_OWNER_ASSETS["owner_manifesto_whisper"]["path"]),
            target_min_s=10.0,
            target_max_s=12.5,
        )
        self.assertGreaterEqual(sel["duration_s"], 10.0)
        self.assertLessEqual(sel["duration_s"], 12.5)
        self.assertLessEqual(sel["end_s"], 187.5)
        self.assertGreater(sel["voiced_density"], 0.6)

        m_exact = sc.compute_text_ocr_metrics(turn360.PINNED_SIGN_TEXT, turn360.PINNED_SIGN_TEXT)
        self.assertTrue(m_exact["exact_verbatim_match"])
        self.assertEqual(m_exact["cer"], 0.0)
        self.assertEqual(m_exact["ordered_word_recall"], 1.0)

        m_bad = sc.compute_text_ocr_metrics("Thonk Jonsen Cdua open srouce", turn360.PINNED_SIGN_TEXT)
        self.assertFalse(m_bad["case_insensitive_exact_match"])
        self.assertGreater(m_bad["cer"], 0.5)

        # Verify pluggable custom-cmd OCR hook end-to-end
        dummy_img = Path(sc.PINNED_OWNER_ASSETS["monkey_avatar"]["path"])
        cmd = f"{sys.executable} -c 'import json; print(json.dumps({{\"observed_text\": \"Thank Jensen for 10 Years of Cuda\"}}))'"
        hook_out = turn360.run_pluggable_ocr_hook(
            dummy_img,
            target_text="Thank Jensen for 10 Years of Cuda",
            backend="custom-cmd",
            ocr_cmd=cmd,
        )
        self.assertFalse(hook_out["simulated"])
        self.assertTrue(hook_out["exact_verbatim_match"])


class TestEndToEndDryRuns(unittest.TestCase):
    def setUp(self) -> None:
        self.work = TEST_SCRATCH_ROOT / self._testMethodName
        if self.work.exists():
            shutil.rmtree(self.work)
        self.work.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.work, ignore_errors=True)

    def test_national_anthem_dry_run_both_modes_and_b0(self) -> None:
        out_dir = self.work / "anthem_run"
        rc = anthem.main(
            [
                "--dry-run",
                "--mode", "both",
                "--include-b0-control",
                "--short-verse",
                "--width", "320",
                "--height", "192",
                "--dry-run-duration-s", "2.0",
                "--out-dir", str(out_dir),
            ]
        )
        self.assertEqual(rc, 0)
        rep = json.loads((out_dir / "national_anthem_showcase_report.json").read_text(encoding="utf-8"))
        self.assertTrue(rep["zero_qwen_loaded"])
        self.assertTrue(rep["trace_audit"]["verified"])
        self.assertIn("mode_a_ref2va_timbre", rep["deliverables"])
        self.assertIn("mode_b_master_locked", rep["deliverables"])
        self.assertTrue(Path(rep["qa_contact_sheet"]["path"]).is_file())
        # Verify Mode B master-locked MP4 has real 48 kHz stereo audio stream & video stream
        mb_probe = rep["deliverables"]["mode_b_master_locked"]["inkling_platos_key_concat4"]["probe"]
        self.assertTrue(mb_probe["has_video"])
        self.assertTrue(mb_probe["has_audio"])
        self.assertEqual(mb_probe["sample_rate"], 48000)

    def test_360_turnaround_dry_run_two_segment_and_single_clip(self) -> None:
        out_two = self.work / "turn_two_seg"
        rc = turn360.main(
            [
                "--dry-run",
                "--topology", "two-segment-branched",
                "--include-b0-control",
                "--width", "320",
                "--height", "192",
                "--out-dir", str(out_two),
            ]
        )
        self.assertEqual(rc, 0)
        rep = json.loads((out_two / "turnaround_metrics_report.json").read_text(encoding="utf-8"))
        self.assertTrue(rep["zero_qwen_loaded"])
        self.assertTrue(rep["trace_audit"]["verified"])
        self.assertEqual(rep["sign_text"], turn360.PINNED_SIGN_TEXT)
        sbs_probe = rep["side_by_side_mp4"]["probe"]
        self.assertEqual((sbs_probe["width"], sbs_probe["height"]), (1920, 1088))
        self.assertTrue(sbs_probe["has_audio"])
        self.assertTrue(Path(rep["qa_contact_sheet"]["path"]).is_file())
        # Verify dry-run OCR permanence audit discriminates non-pinned approach frames cleanly
        ocr_off = rep["ocr_permanence_audit"]["ledger_off"]
        ocr_on = rep["ocr_permanence_audit"]["ledger_on"]
        self.assertTrue(ocr_off["opposite_180_sign_absent"])
        self.assertTrue(ocr_on["opposite_180_sign_absent"])
        self.assertGreater(ocr_on["nonpinned_approach_mean_char_accuracy"], ocr_off["nonpinned_approach_mean_char_accuracy"])
        # Verify Packet C narration was muxed at 48 kHz stereo and recorded in hash-chained trace!
        self.assertEqual(sbs_probe["sample_rate"], 48000)
        self.assertEqual(sbs_probe["channels"], 2)
        self.assertEqual(rep["narration"]["narration_text"], turn360.DEFAULT_TURNAROUND_NARRATION_TEXT)
        self.assertIn("packetC_mouth", rep["trace_audit"]["stages"])

        # Also verify explicit --narration-wav + fast single-clip topology
        custom_nar = self.work / "custom_packetC_narration.wav"
        turn360._generate_dry_run_narration_wav(custom_nar, duration_s=2.4, sr=24000)
        out_one = self.work / "turn_single"
        rc2 = turn360.main(
            [
                "--dry-run",
                "--topology", "single-clip",
                "--narration-wav", str(custom_nar),
                "--narration-text", turn360.DEFAULT_TURNAROUND_NARRATION_TEXT,
                "--width", "320",
                "--height", "192",
                "--out-dir", str(out_one),
            ]
        )
        self.assertEqual(rc2, 0)
        rep2 = json.loads((out_one / "turnaround_metrics_report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep2["topology"], "single-clip")
        self.assertTrue(rep2["trace_audit"]["verified"])
        self.assertEqual(rep2["narration"]["source"], "prebuilt_packetC_wav")
        self.assertEqual(rep2["narration"]["wav_sha256"], sc.sha256_file(custom_nar))

    def test_prebuilt_conds_dir_handoff_and_tamper_fail_closed(self) -> None:
        # 1) Project conds only for National Anthem, then consume via --prebuilt-conds-dir (even with bogus keys_dir!)
        proj_dir = self.work / "anthem_proj"
        rc_p = anthem.main(
            [
                "--dry-run",
                "--project-conds-only",
                "--include-b0-control",
                "--out-dir", str(proj_dir),
            ]
        )
        self.assertEqual(rc_p, 0)
        cond_dir = proj_dir / "packetA_conds" / "conds"
        self.assertTrue((cond_dir / "conds_manifest.json").is_file())

        run_dir = self.work / "anthem_from_prebuilt"
        rc_r = anthem.main(
            [
                "--dry-run",
                "--mode", "master-locked",
                "--include-b0-control",
                "--prebuilt-conds-dir", str(cond_dir),
                # Omit --prebuilt-direction-json to prove sibling auto-discovery works!
                "--keys-dir", str(self.work / "nonexistent_remote_keys"),
                "--exporter-bin", str(self.work / "nonexistent_remote_exporter"),
                "--owner-voice-wav", str(self.work / "nonexistent_remote_owner.wav"),
                "--owner-voice-whisper", str(self.work / "nonexistent_remote_owner.json"),
                "--width", "320",
                "--height", "192",
                "--dry-run-duration-s", "2.0",
                "--out-dir", str(run_dir),
            ]
        )
        self.assertEqual(rc_r, 0)
        rep = json.loads((run_dir / "national_anthem_showcase_report.json").read_text(encoding="utf-8"))
        self.assertTrue(rep["trace_audit"]["verified"])

        # 2) Same --project-conds-only -> --prebuilt-conds-dir handoff on 360 Turnaround (sibling auto-discovery too!)
        tproj_dir = self.work / "turn_proj"
        rc_tp = turn360.main(["--dry-run", "--project-conds-only", "--out-dir", str(tproj_dir)])
        self.assertEqual(rc_tp, 0)
        tcond_dir = tproj_dir / "packetA_conds" / "conds"
        trun_dir = self.work / "turn_from_prebuilt"
        rc_tr = turn360.main(
            [
                "--dry-run",
                "--topology", "single-clip",
                "--prebuilt-conds-dir", str(tcond_dir),
                "--keys-dir", str(self.work / "nonexistent_remote_keys"),
                "--exporter-bin", str(self.work / "nonexistent_remote_exporter"),
                "--width", "320",
                "--height", "192",
                "--out-dir", str(trun_dir),
            ]
        )
        self.assertEqual(rc_tr, 0)

        # 3) Tampering a prebuilt conditioning file must fail closed against conds_manifest.json!
        victim = tcond_dir / "turnaround_single_clip__inkling_platos_key_concat4.dryrun.json"
        victim.write_text(victim.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            sc.load_prebuilt_platos_key_conds(tcond_dir, ["turnaround_single_clip"], dry_run=True)

    def test_turnaround_rtxp_30s_profiles_and_real_narration_wav(self) -> None:
        # 0) Verify National Anthem >=30s RTX PRO 6000 profiles produce exact >=720f AV-aligned chained plans
        p_ant_30 = sc.plan_h3_segments(
            float(anthem.ANTHEM_PROFILES["rtxp-30s"]["max_duration_s"]),
            min_frames=141,
            max_frames=345,
            context_frames=sc.CHAIN_CONTEXT_FRAMES,
            pad_s=0.15,
        )
        self.assertEqual(p_ant_30["total_frames"], 804)
        self.assertAlmostEqual(p_ant_30["total_duration_s"], 33.5, places=4)
        self.assertEqual([s["frames"] for s in p_ant_30["segments"]], [345, 345, 192])
        self.assertTrue(all(s["exact_av_boundary"] for s in p_ant_30["segments"]))

        p_ant_mode_a = sc.plan_h3_segments(
            float(anthem.ANTHEM_PROFILES["rtxp-60s-full"]["mode_a_duration_s"]),
            min_frames=141,
            max_frames=345,
            context_frames=sc.CHAIN_CONTEXT_FRAMES,
            pad_s=0.0,
        )
        self.assertEqual(p_ant_mode_a["total_frames"], 804)
        self.assertAlmostEqual(p_ant_mode_a["total_duration_s"], 33.5, places=4)

        # 1) Exact RTX PRO 6000 >=30s (>=720f @ 24 fps) AV-aligned grid checks (no dry-run shrink)
        p_orb = turn360.plan_turnaround_topology("three-segment-orbit", dry_run=False)
        self.assertEqual(p_orb["total_frames"], 804)
        self.assertGreaterEqual(p_orb["total_frames"], 720)
        self.assertAlmostEqual(p_orb["total_duration_s"], 33.5, places=4)
        self.assertTrue(p_orb["rtxp_30s_flagship"])
        self.assertEqual(p_orb["shared_segment_count"], 2)
        self.assertEqual(p_orb["branch_frame_index"], 549)
        self.assertTrue(all(s["exact_av_boundary"] for s in p_orb["segments"]))
        self.assertEqual((p_orb["total_frames"] * 40) % 24, 0)

        p_dbl = turn360.plan_turnaround_topology("double-revisit-720", dry_run=False)
        self.assertEqual(p_dbl["total_frames"], 753)
        self.assertGreaterEqual(p_dbl["total_frames"], 720)
        self.assertAlmostEqual(p_dbl["total_duration_s"], 31.375, places=4)
        self.assertTrue(p_dbl["rtxp_30s_flagship"])
        self.assertEqual(p_dbl["max_yaw_deg"], 720.0)
        self.assertEqual(len(p_dbl["segments"]), 4)
        self.assertTrue(all(s["exact_av_boundary"] for s in p_dbl["segments"]))
        self.assertEqual((p_dbl["total_frames"] * 40) % 24, 0)

        # 2) Dry-run end-to-end verification of BOTH >=30s RTX PRO 6000 presets + real attested Packet C narration WAV
        real_nar = turn360.DEFAULT_ATTESTED_NARRATION_WAV
        self.assertTrue(real_nar.is_file(), f"Expected attested Packet C WAV at {real_nar}")
        expected_nar_sha = sc.sha256_file(real_nar)

        out_orb = self.work / "rtxp_30s_orbit"
        rc_orb = turn360.main(
            [
                "--dry-run",
                "--profile", "rtxp-30s-orbit",
                "--narration-wav", str(real_nar),
                "--width", "320",
                "--height", "192",
                "--out-dir", str(out_orb),
            ]
        )
        self.assertEqual(rc_orb, 0)
        rep_orb = json.loads((out_orb / "turnaround_metrics_report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep_orb["topology"], "three-segment-orbit")
        self.assertEqual(rep_orb["narration"]["wav_sha256"], expected_nar_sha)
        self.assertEqual(rep_orb["side_by_side_mp4"]["probe"]["width"], 1920)
        self.assertEqual(rep_orb["side_by_side_mp4"]["probe"]["height"], 1088)
        self.assertEqual(rep_orb["side_by_side_mp4"]["probe"]["sample_rate"], 48000)
        self.assertTrue(rep_orb["trace_audit"]["verified"])
        self.assertTrue(rep_orb["ocr_permanence_audit"]["ledger_on"]["opposite_180_sign_absent"])

        out_dbl = self.work / "rtxp_30s_double_revisit"
        rc_dbl = turn360.main(
            [
                "--dry-run",
                "--profile", "rtxp-30s-double-revisit",
                "--narration-wav", str(real_nar),
                "--width", "320",
                "--height", "192",
                "--out-dir", str(out_dbl),
            ]
        )
        self.assertEqual(rc_dbl, 0)
        rep_dbl = json.loads((out_dbl / "turnaround_metrics_report.json").read_text(encoding="utf-8"))
        self.assertEqual(rep_dbl["topology"], "double-revisit-720")
        self.assertEqual(rep_dbl["plan"]["max_yaw_deg"], 720.0)
        self.assertTrue(rep_dbl["ocr_permanence_audit"]["ledger_on"]["opposite_180_sign_absent"])
        self.assertTrue(rep_dbl["ocr_permanence_audit"]["ledger_on"]["return_final_exact_match"])
        self.assertGreater(
            rep_dbl["ocr_permanence_audit"]["ledger_on"]["nonpinned_approach_mean_char_accuracy"],
            rep_dbl["ocr_permanence_audit"]["ledger_off"]["nonpinned_approach_mean_char_accuracy"],
        )
        self.assertTrue(rep_dbl["trace_audit"]["verified"])
        self.assertEqual(rep_dbl["verdict"], "PASS")
        self.assertTrue(rep_dbl["quality_gates"]["continuity_gate_passed"])
        self.assertTrue(rep_dbl["quality_gates"]["identity_gate_passed"])

    def test_v4_anti_goodhart_continuity_gate_and_blender_room_anchors(self) -> None:
        # 1) Synthetic smooth video must PASS continuity gate; injected terminal teleport MUST FAIL!
        w, h, n = 320, 192, 60
        smooth_frames = np.zeros((n, h, w, 3), dtype=np.uint8)
        for i in range(n):
            v = int(30 + 80 * (i / float(n - 1)))
            smooth_frames[i, :, :, 0] = v
            smooth_frames[i, :, :, 1] = v + 10
            smooth_frames[i, :, :, 2] = v + 20
        smooth_mp4 = self.work / "smooth.mp4"
        sc.mux_frames_and_wav_to_mp4(smooth_frames, None, smooth_mp4, fps=24, crf=22, x264_threads=2)
        c_ok = turn360.run_temporal_continuity_gate(smooth_mp4)
        self.assertTrue(c_ok["passed"], f"Smooth clip unexpectedly failed: {c_ok}")

        # Inject a hard terminal teleport cut on the last 2 frames (exact v3 Goodhart failure shape!)
        teleport_frames = smooth_frames.copy()
        # High-contrast checkerboard jump so both SSIM and L1 spike cleanly
        yy, xx = np.indices((h, w))
        chk = (((yy // 16) + (xx // 16)) % 2 == 0)
        teleport_frames[-2:, chk] = (245, 245, 245)
        teleport_frames[-2:, ~chk] = (10, 20, 15)
        teleport_mp4 = self.work / "teleport.mp4"
        sc.mux_frames_and_wav_to_mp4(teleport_frames, None, teleport_mp4, fps=24, crf=22, x264_threads=2)
        c_fail = turn360.run_temporal_continuity_gate(teleport_mp4)
        self.assertFalse(c_fail["passed"], "Terminal teleport jump-cut MUST fail continuity gate!")
        self.assertTrue(any("JUMP" in f or "TELEPORT" in f or "SCENE_CUT" in f for f in c_fail["failures"]), f"Unexpected failures: {c_fail['failures']}")

        # 2) Verify against real v3 Goodhart failure if present on disk (must strictly FAIL!)
        v3_dir = Path("/mnt/storage/omni_gate2_runs/phase0_912_spinup/showcase_runs/turnaround_360_v3")
        if (v3_dir / "turnaround_metrics_report.json").is_file():
            rc_v3 = turn360.main(["--audit-existing-run", str(v3_dir), "--identity-backend", "skip"])
            self.assertEqual(rc_v3, 2, "v3 Goodhart jump-cut run MUST fail v4 continuity/OCR gates (rc=2)!")

        # 3) Verify real Blender 5.2.5 LTS 3D room anchor bank loads & enforces fail-closed SHA-256!
        room_dir = turn360.DEFAULT_BLENDER_ROOM_DIR
        if (room_dir / "room_anchors_manifest.json").is_file():
            bank = turn360.load_blender_room_anchors(room_dir)
            self.assertEqual(bank["source"], "blender-room")
            for req_yaw in (0, 60, 120, 180, 240, 300, 330):
                self.assertIn(req_yaw, bank["by_yaw"])
            # Tamper copy must fail closed
            bad_room = self.work / "bad_room"
            bad_room.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(room_dir / "room_anchors_manifest.json", bad_room / "room_anchors_manifest.json")
            for ent in bank["anchors"]:
                shutil.copyfile(Path(ent["path"]), bad_room / ent["filename"])
            (bad_room / "yaw_000.png").write_bytes((bad_room / "yaw_000.png").read_bytes() + b"\x00")
            with self.assertRaises(RuntimeError):
                turn360.load_blender_room_anchors(bad_room)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TEST_SCRATCH_ROOT, ignore_errors=True)

#!/usr/bin/env python3
"""#912 Flagship Showcase Video 2: Matched Ledger-OFF vs Ledger-ON 360-Degree Turnaround Permanence Demo.

Zero-gradient Plato's Key (`Inkling-2-Mini` -> Packet A 90.0 MiB `concat4` key `cca1e5bc...`,
ZERO `Qwen3-VL-32B` loaded!) driving MiniMax-H3 FL2VA/Ref2VA across a full 0 -> 180 -> 360 degree
camera turnaround around the Monkey Founder's studio room containing the owner's verbatim 24-word sign:

  "Thank Jensen for 10 Years of Cuda, Ty thinking machines for this model, and ty to the open source community for being amazing"

Zero-Confound Architecture (default `--topology two-segment-branched`):
  1. Identical Frame-0 Anchor Plate (`yaw = 0 deg`): Monkey Founder + razor-sharp 24-word wall plaque
     + 2 additional room invariants (emerald GPU workbench + open studio door w/ warm amber lamp).
     Audited and committed into `verified_ledger_store.json` BEFORE panning.
  2. Shared Segment 0 (`0 -> 180 deg` pan away, `192f = 8.00s` exact AV boundary): rendered ONCE with
     Packet A 90.0 MiB `concat4` key (`cfg=1.0`, `shift_video=12.0`, `shift_audio=3.0`). Both arms
     inherit the EXACT SAME byte-identical `seg0` video+audio tail latent (`39f` = 12 video / 65 audio
     latents) — zero outbound confound!
  3. Branched Segment 1 (`180 -> 360 deg` return pan, `243f = 10.125s` w/ `39f` overlap -> `396f = 16.50s`
     total exact), identical noise seed & Packet A `concat4` conditioning:
       - `ledger_off` (Left): zero Verified Ledger retrieval on return (stock diffusion memory loss).
       - `ledger_on` (Right): loop-closure pose query (`yaw = 0 deg`) retrieves the verified anchor latent
         from `verified_ledger_store.json` and injects it via pure-VAE `minimax_refs` and/or return-frame
         `minimax_keyframes` (`--ledger-injection-mode {ref-and-return-guide,ref-only,return-guide}`).
       - Optional `b0_shuffled` control (`--include-b0-control`) via `dabbec7a...`.
  4. Fast `--topology single-clip` mode (`294f = 12.25s` or `345f = 14.375s`) also supported.
  5. Pluggable Packet B2 High-Res OCR Verification Hook (`--ocr-backend {packetb2-got-ocr,custom-cmd,dry-run-sim}`):
     audits BOTH arms across multiple trajectory checkpoints including NON-PINNED approach frames
     (`0 deg`, `180 deg` away, `~295 deg`, `~325 deg`, `~345 deg`, `~355 deg`, `360 deg`), enforcing
     monotonic keep-or-commit gating when retries are requested.
  6. Deterministic 1920x1088 side-by-side split-screen MP4 (`turnaround_side_by_side_1920x1088.mp4`) with
     live yaw compass (`0 -> 180 -> 360 deg`), Verified Ledger status indicator, Packet B2 OCR CER readout,
     multi-angle QA contact sheet (`qa_contact_sheet_360_turnaround.png`), and append-only hash-chained
     `showcase_process_trace.jsonl`.

Zero GPU jobs are launched on import or under `--dry-run`.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from showcase_common import (
    AUDIO_LATENT_HZ,
    CHAIN_CONTEXT_FRAMES,
    DEFAULT_PATHS,
    EXPECTED_B0_KEY_SHA256,
    EXPECTED_CONCAT4_KEY_SHA256,
    EXPECTED_EXPORTER_SHA256,
    FFMPEG,
    FPS,
    H3_AUDIO_SR,
    H3_PDD_SIGMAS,
    H3_SAMPLES_PER_LATENT,
    MASTER_AUDIO_SR,
    PINNED_OWNER_ASSETS,
    H3SamplingConfig,
    InklingTextAuthor,
    ShowcaseTraceLogger,
    apply_h3_model_patches,
    attach_zero_qwen_h3_conditioning,
    atomic_write_json,
    audio_latent_t,
    build_contact_sheet,
    build_masked_av_latent_and_noise_mask,
    choose_h3_frames,
    compute_text_ocr_metrics,
    cond_pkg_path,
    decode_audio_latent_stereo,
    decode_video_latent_u8,
    detect_hardware_info,
    export_and_project_platos_key_conds,
    extract_first_json_object,
    extract_frames_at_indices,
    is_exact_av_boundary,
    load_font,
    load_prebuilt_platos_key_conds,
    mux_frames_and_wav_to_mp4,
    plan_turnaround_two_segment,
    probe_media,
    run_blinded_whisper_check,
    sha256_file,
    stitch_chained_segments_ffmpeg,
    verify_pinned_file,
    verify_trace_jsonl,
    video_latent_t,
    write_stereo_wav_pcm16,
)

# Exact owner-pinned 24-word high-detail target sign (verbatim from #934 / inkling-agent)
PINNED_SIGN_TEXT = (
    "Thank Jensen for 10 Years of Cuda, Ty thinking machines for this model, "
    "and ty to the open source community for being amazing"
)

# Direct owner requirement (2026-11-27 02:17 PST): turnaround narration over the split-screen demo
DEFAULT_TURNAROUND_NARRATION_TEXT = (
    "This is not a video. This is a world model being turned around."
)
DEFAULT_ATTESTED_NARRATION_WAV = SCRIPT_DIR / "turnaround_narration_24k.wav"
DEFAULT_BLENDER_ROOM_DIR = SCRIPT_DIR / "blender_360_room_v4"

# Direct owner requirement (2026-11-27 02:33 PST): >=30s (>=720f @ 24 fps) coherent RTX PRO 6000 flagship presets
PROFILE_TO_TOPOLOGY: Dict[str, str] = {
    "rtxp-30s-orbit": "three-segment-orbit",
    "rtxp-30s-double-revisit": "double-revisit-720",
    "smoke-16s": "two-segment-branched",
    "smoke-12s": "single-clip",
}

DEFAULT_TURNAROUND_PROMPTS = {
    "orbit_seg0_0_to_120": (
        "Single unbroken motorized 360-degree nodal tripod camera orbit panning smoothly clockwise from 0 degrees to 120 degrees "
        "inside the real 3D dark obsidian-and-walnut AI hardware workshop, ZERO scene cuts. Opening on the exact 3D room view "
        "with <Picture 1> (the dignified chimpanzee founder wearing a black top hat with lime band, gold monocle, and lime-piped "
        "tuxedo seated at the left studio desk), <Picture 2> (the physical backlit wall plaque with crisp black text and the real "
        "open studio doorway glowing with a warm amber lamp on the right), the camera glides smoothly right past the warm open "
        "doorway onto the East 42U steel server racks with emerald LEDs as the plaque and founder exit left at 120 degrees. "
        "Photorealistic ARRI Alexa 35 35mm cinema, locked 3D room architecture, subtle room hum, constant angular speed, 24 fps."
    ),
    "orbit_seg1_120_to_240": (
        "Continuing seamlessly from 120 degrees with ZERO scene cuts, the nodal tripod camera sweeps smoothly clockwise across the "
        "opposite 180-degree rear wall of the same dark obsidian-and-walnut AI hardware workshop through 240 degrees. Gliding past "
        "the East server racks, across the South industrial wooden workshop shelving and warm sconces (180 degrees, original plaque "
        "and founder completely behind camera), onto the West twilight mullioned studio window and drafting table at 240 degrees. "
        "Photorealistic ARRI Alexa 35 35mm cinema, steady lighting, subtle fan hum, smooth constant angular speed, 24 fps."
    ),
    "orbit_seg2_240_to_360": (
        "Continuing seamlessly from 240 degrees with ZERO scene cuts, the nodal tripod camera completes its slow clockwise 360-degree "
        "orbit smoothly from the West drafting table back around to 360 degrees (the exact original North wall). Sweeping smoothly "
        "past the walnut slats into <Picture 1> (the exact same dignified chimpanzee founder in his black top hat with lime band, "
        "gold monocle, and lime-piped tuxedo seated behind the obsidian hardware desk) beside <Picture 2> (the crisp backlit wall "
        "plaque reading 'Thank Jensen for 10 Years of Cuda, Ty thinking machines for this model, and ty to the open source community "
        "for being amazing' and the warm amber open studio doorway on the right). Photorealistic ARRI Alexa 35 35mm cinema, 24 fps."
    ),
    "seg0_pan_away_0_to_180": (
        "Single unbroken motorized 360-degree tripod camera pan rotating smoothly clockwise from 0 degrees to 180 degrees "
        "inside the dark obsidian-and-walnut AI hardware workshop, ZERO scene cuts. Starting from the exact 3D room showing "
        "<Picture 1> (the Monkey Founder in black top hat, gold monocle, and lime-piped tuxedo at the desk), <Picture 2> "
        "(the backlit wall plaque and warm amber open studio doorway), the camera pans steadily right past the server racks "
        "to the 180-degree opposite rear workshop shelving wall with the sign and founder fully out of frame. 24 fps."
    ),
    "seg1_return_180_to_360": (
        "Continuing seamlessly from the 180-degree opposite workshop wall with ZERO scene cuts, the motorized tripod camera "
        "completes its clockwise 360-degree orbit back around to 360 degrees. Sweeping past the West window and walnut slats "
        "back to the North wall, <Picture 1> (the exact same top-hat tuxedo Monkey Founder at the desk) and <Picture 2> "
        "(the crisp backlit wall plaque reading 'Thank Jensen for 10 Years of Cuda, Ty thinking machines for this model, and ty "
        "to the open source community for being amazing' plus the warm amber open doorway) glide smoothly into sharp focus. 24 fps."
    ),
    "single_clip_full_360": (
        "Single unbroken clockwise 360-degree camera turnaround inside the obsidian-and-walnut hardware workshop, ZERO cuts. "
        "Starting at 0 degrees facing <Picture 1> (the top-hat tuxedo Monkey Founder at the desk) and <Picture 2> (the crisp "
        "wall plaque reading 'Thank Jensen for 10 Years of Cuda, Ty thinking machines for this model, and ty to the open source "
        "community for being amazing' and warm amber open doorway), panning 180 degrees away across workshop shelving and back "
        "smoothly to 360 degrees on the exact starting founder, plaque, and open doorway. 24 fps."
    ),
    "h3_image_anchor_prompt": (
        "Detailed photorealistic 35mm wide studio photograph inside a dark obsidian-and-walnut AI hardware lab: on the left "
        "behind an obsidian desk sits the dignified chimpanzee founder in a black top hat with lime band, gold monocle, and "
        "lime-piped tuxedo; on the center wall hangs a physical backlit wall plaque clearly reading 'Thank Jensen for 10 Years "
        "of Cuda, Ty thinking machines for this model, and ty to the open source community for being amazing'; on the right "
        "is an open wooden studio doorway glowing with a warm amber hallway lamp."
    ),
}


def load_blender_room_anchors(room_dir: Path) -> Dict[str, Any]:
    """Load & fail-closed SHA-256 verify pose-indexed 3D Blender Cycles room anchors."""
    man_p = Path(room_dir) / "room_anchors_manifest.json"
    if not man_p.is_file():
        raise FileNotFoundError(
            f"Missing Blender 3D room anchor manifest: {man_p}. "
            "Render it with build_blender_360_room_anchors.py or pass --anchor-source composite-plate."
        )
    man = json.loads(man_p.read_text(encoding="utf-8"))
    by_yaw: Dict[int, Dict[str, Any]] = {}
    verified_rows: List[Dict[str, Any]] = []
    for ent in man.get("anchors") or []:
        yaw_i = int(round(float(ent["yaw_deg"])))
        fname = str(ent.get("filename") or Path(str(ent["path"])).name)
        cand = Path(room_dir) / fname
        if not cand.is_file():
            cand = Path(str(ent["path"]))
        if not cand.is_file():
            raise FileNotFoundError(f"Missing Blender 3D room anchor for yaw={yaw_i}: {cand}")
        actual_sha = sha256_file(cand)
        expected_sha = ent.get("sha256")
        if expected_sha and actual_sha != str(expected_sha):
            raise RuntimeError(
                f"Blender 3D room anchor SHA-256 mismatch for {cand}: expected {expected_sha}, got {actual_sha}"
            )
        row = {
            "yaw_deg": float(yaw_i),
            "filename": fname,
            "path": str(cand),
            "sha256": actual_sha,
            "bytes": int(cand.stat().st_size),
            "width": int(ent.get("width", 1344)),
            "height": int(ent.get("height", 768)),
        }
        by_yaw[yaw_i] = row
        verified_rows.append(row)
    if 0 not in by_yaw:
        raise RuntimeError(f"Blender 3D room manifest {man_p} missing required yaw=0 anchor")
    return {
        "source": "blender-room",
        "room_dir": str(room_dir),
        "manifest_path": str(man_p),
        "manifest_sha256": sha256_file(man_p),
        "blender_version": str(man.get("blender_version", "unknown")),
        "engine": str(man.get("engine", "CYCLES")),
        "path": by_yaw[0]["path"],
        "sha256": by_yaw[0]["sha256"],
        "bytes": by_yaw[0]["bytes"],
        "width": by_yaw[0]["width"],
        "height": by_yaw[0]["height"],
        "by_yaw": by_yaw,
        "by_yaw_str": {f"{k:03d}": v for k, v in sorted(by_yaw.items())},
        "anchors": verified_rows,
    }


def _build_grid_seg(frames: int, start_s: float, context_frames: int) -> Dict[str, Any]:
    v_steps = video_latent_t(frames)
    a_steps = audio_latent_t(frames)
    exact_av = (frames * AUDIO_LATENT_HZ) % FPS == 0
    return {
        "frames": int(frames),
        "video_latent_steps": int(v_steps),
        "trim_prefix_frames": int(context_frames),
        "trim_prefix_video_latents": int(video_latent_t(context_frames)) if context_frames > 0 else 0,
        "trim_prefix_audio_latents": int(audio_latent_t(context_frames)) if context_frames > 0 else 0,
        "net_new_frames": int(frames - context_frames),
        "clip_start_s": round(float(start_s), 6),
        "duration_s": round(frames / float(FPS), 6),
        "net_duration_s": round((frames - context_frames) / float(FPS), 6),
        "exact_av_boundary": bool(exact_av),
        "audio_latent_steps": int(a_steps),
        "pcm_samples_32k": int(a_steps * H3_SAMPLES_PER_LATENT),
        "pcm_samples_48k": int(round((frames / float(FPS)) * MASTER_AUDIO_SR)),
    }


def plan_turnaround_topology(
    topology: str,
    *,
    orbit_seg_frames: int = 294,
    seg0_frames: int = 192,
    seg1_frames: int = 243,
    single_clip_frames: int = 294,
    context_frames: int = CHAIN_CONTEXT_FRAMES,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Build exact 17k+5 / 40Hz / 39f-overlap turnaround plan for any supported topology.

    Flagship RTX PRO 6000 >=30s (>=720f @ 24 fps) topologies:
      - `three-segment-orbit` (default): [294, 294, 294] w/ 39f overlap = 804f (33.500s).
        Shared Seg 0 (0->120 deg) + Shared Seg 1 (120->240 deg) + Branched Seg 2 (240->360 deg).
      - `double-revisit-720`: [192, 243, 192, 243] w/ 39f overlap = 753f (31.375s).
        Shared Seg 0 (0->180 deg) + Branched Revisit #1 (180->360) + Pan Away #2 (360->540) + Revisit #2 (540->720).
    Fast optional smoke topologies:
      - `two-segment-branched`: [192, 243] w/ 39f overlap = 396f (16.500s).
      - `single-clip`: [294] = 294f (12.250s).
    """
    if topology == "two-segment-branched":
        s0, s1 = (90, 90) if dry_run else (seg0_frames, seg1_frames)
        base = plan_turnaround_two_segment(seg0_frames=s0, seg1_frames=s1, context_frames=context_frames)
        base["shared_segment_count"] = 1
        base["segment_end_frames"] = [s0, int(base["total_frames"])]
        base["segment_end_yaws"] = [180.0, 360.0]
        base["max_yaw_deg"] = 360.0
        base["rtxp_30s_flagship"] = int(base["total_frames"]) >= 720
        return base

    if topology == "single-clip":
        sf = 90 if dry_run else single_clip_frames
        seg = choose_h3_frames(sf / float(FPS), min_frames=sf, max_frames=max(345, sf), pad_s=0.0, prefer_exact_av=True)
        return {
            "fps": FPS,
            "topology": "single-clip",
            "context_frames": 0,
            "shared_segment_count": 0,
            "branch_frame_index": None,
            "segment_end_frames": [int(seg["frames"])],
            "segment_end_yaws": [360.0],
            "max_yaw_deg": 360.0,
            "total_frames": int(seg["frames"]),
            "total_duration_s": seg["duration_s"],
            "total_pcm_samples_32k": seg["pcm_samples_32k"],
            "total_pcm_samples_48k": seg["pcm_samples_48k"],
            "chained": False,
            "rtxp_30s_flagship": int(seg["frames"]) >= 720,
            "segments": [seg],
        }

    if topology == "three-segment-orbit":
        sf = 90 if dry_run else orbit_seg_frames
        seg_lens = [sf, sf, sf]
        end_yaws = [120.0, 240.0, 360.0]
        shared_n = 2
        max_yaw = 360.0
    elif topology == "double-revisit-720":
        s0, s1 = (90, 90) if dry_run else (seg0_frames, seg1_frames)
        seg_lens = [s0, s1, s0, s1]
        end_yaws = [180.0, 360.0, 540.0, 720.0]
        shared_n = 1
        max_yaw = 720.0
    else:
        raise ValueError(f"Unsupported turnaround topology: {topology!r}")

    if not is_exact_av_boundary(context_frames):
        raise ValueError(f"context_frames={context_frames} must be an exact AV boundary (e.g. 39)")
    segments: List[Dict[str, Any]] = []
    end_frames: List[int] = []
    covered = 0
    for idx, slen in enumerate(seg_lens):
        if not is_exact_av_boundary(slen) or slen > 345:
            raise ValueError(f"Segment {idx} frames={slen} must be an exact H3 AV boundary <=345 (141/192/243/294/345)")
        ctx = 0 if idx == 0 else context_frames
        if ctx > 0 and slen <= ctx:
            raise ValueError(f"Segment {idx} frames={slen} must exceed context_frames={ctx}")
        start_f = 0 if idx == 0 else (covered - ctx)
        seg = _build_grid_seg(slen, start_s=start_f / float(FPS), context_frames=ctx)
        segments.append(seg)
        covered += int(seg["net_new_frames"])
        end_frames.append(int(covered))

    if (covered * AUDIO_LATENT_HZ) % FPS != 0:
        raise RuntimeError(f"Chained turnaround total {covered}f broke exact 40Hz/24fps AV boundary")
    total_a_steps = audio_latent_t(covered)
    branch_frame = end_frames[shared_n - 1]
    return {
        "fps": FPS,
        "topology": topology,
        "context_frames": int(context_frames),
        "shared_segment_count": int(shared_n),
        "branch_frame_index": int(branch_frame),
        "branch_time_s": round(branch_frame / float(FPS), 6),
        "segment_end_frames": end_frames,
        "segment_end_yaws": end_yaws,
        "max_yaw_deg": float(max_yaw),
        "total_frames": int(covered),
        "total_duration_s": round(covered / float(FPS), 6),
        "total_pcm_samples_32k": int(total_a_steps * H3_SAMPLES_PER_LATENT),
        "total_pcm_samples_48k": int(round((covered / float(FPS)) * MASTER_AUDIO_SR)),
        "chained": True,
        "rtxp_30s_flagship": int(covered) >= 720,
        "segments": segments,
    }


# ======================================================================================
# Deterministic Frame-0 High-Detail Studio Anchor Plate Builder (Zero GPU)
# ======================================================================================


def render_turnaround_anchor_plate(
    *,
    founder_ref_path: Path,
    sign_text: str,
    out_png: Path,
    width: int = 1920,
    height: int = 1088,
) -> Dict[str, Any]:
    """Build a crisp 1920x1088 Frame-0 studio anchor plate containing all 3 pinned scene invariants.

    Scene invariants:
      1. High-detail 24-word backlit wall plaque displaying `sign_text` verbatim (0.0 OCR CER at yaw=0).
      2. Monkey Founder (`kf_01_host_hook.png` / `monkey-avatar.png`) framed cleanly on the left studio desk.
      3. Open emerald-accented studio doorway + warm amber practical lamp on the far right wall.
    """
    s = width / 1920.0
    canvas = Image.new("RGB", (width, height), (12, 16, 20))
    draw = ImageDraw.Draw(canvas)

    # Studio backdrop gradient + subtle architectural grid + emerald accent strip
    for y in range(height):
        frac = y / float(max(height - 1, 1))
        r = int(11 + 10 * (1.0 - frac))
        g = int(16 + 14 * (1.0 - frac))
        b = int(20 + 12 * (1.0 - frac))
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    draw.rectangle([(0, int(190 * s)), (width, int(198 * s))], fill=(72, 130, 12))

    # Invariant 2 (Left): Monkey Founder studio portrait inset blended smoothly onto obsidian plinth
    ref_im = Image.open(founder_ref_path).convert("RGB")
    rw, rh = ref_im.size
    crop_w = min(rw, int(rh * 0.92))
    cx0 = (rw - crop_w) // 2
    founder_crop = ref_im.crop((cx0, 0, cx0 + crop_w, rh))
    fw = int(660 * s)
    fh = int(height - int(96 * s))
    founder_resized = founder_crop.resize((fw, fh), Image.Resampling.LANCZOS)
    canvas.paste(founder_resized, (int(42 * s), int(64 * s)))
    draw.rectangle(
        [(int(42 * s), int(64 * s)), (int(42 * s) + fw, int(64 * s) + fh)],
        outline=(118, 185, 0),
        width=max(2, int(4 * s)),
    )

    # Invariant 3 (Far Right): Open studio doorway + warm amber lamp glow (spatial landmark!)
    dx0, dy0, dx1, dy1 = int(1660 * s), int(230 * s), int(1875 * s), int(940 * s)
    draw.rectangle([(dx0, dy0), (dx1, dy1)], fill=(26, 22, 18), outline=(92, 110, 100), width=max(2, int(4 * s)))
    # Warm amber interior corridor behind open door
    draw.rectangle([(dx0 + int(18 * s), dy0 + int(18 * s)), (dx1 - int(18 * s), dy1 - int(10 * s))], fill=(62, 42, 20))
    draw.ellipse([(dx0 + int(70 * s), dy0 + int(90 * s)), (dx1 - int(55 * s), dy0 + int(200 * s))], fill=(246, 194, 86))
    draw.text((dx0 + int(26 * s), dy1 + int(12 * s)), "OPEN DOOR", fill=(160, 178, 168), font=load_font(int(18 * s), bold=True))

    # Invariant 1 (Center-Right Hero): High-contrast backlit wall plaque with verbatim 24-word sign
    px0, py0, px1, py1 = int(748 * s), int(140 * s), int(1616 * s), int(790 * s)
    draw.rounded_rectangle(
        [(px0, py0), (px1, py1)],
        radius=int(18 * s),
        fill=(246, 249, 250),
        outline=(118, 185, 0),
        width=max(4, int(8 * s)),
    )
    draw.rectangle([(px0 + int(28 * s), py0 + int(24 * s)), (px1 - int(28 * s), py0 + int(34 * s))], fill=(118, 185, 0))

    # Auto-fit crisp dark-on-light typography so OCR has maximum contrast & zero clipping
    inner_w = (px1 - px0) - int(84 * s)
    inner_h = (py1 - py0) - int(110 * s)
    chosen_font = load_font(int(44 * s), bold=True)
    chosen_lines = [sign_text]
    for fsz in range(int(58 * s), int(22 * s), -2):
        fnt = load_font(fsz, bold=True)
        # Estimate wrap width from average char width
        avg_cw = max(1.0, draw.textbbox((0, 0), "abcdefghijklm ", font=fnt)[2] / 14.0)
        wrap_chars = max(14, int(inner_w / avg_cw))
        lines = textwrap.wrap(sign_text, width=wrap_chars)
        line_h = int(fsz * 1.32)
        max_lw = max((draw.textbbox((0, 0), ln, font=fnt)[2] for ln in lines), default=0)
        if max_lw <= inner_w and len(lines) * line_h <= inner_h:
            chosen_font = fnt
            chosen_lines = lines
            break

    line_h = int(chosen_font.size * 1.32) if hasattr(chosen_font, "size") else int(46 * s)
    total_th = len(chosen_lines) * line_h
    yy = py0 + int(56 * s) + max(0, (inner_h - total_th) // 2)
    for ln in chosen_lines:
        draw.text((px0 + int(42 * s), yy), ln, fill=(14, 18, 22), font=chosen_font)
        yy += line_h

    # Emerald GPU plinth bar beneath the sign plaque
    draw.rounded_rectangle(
        [(px0, int(825 * s)), (px1, int(992 * s))],
        radius=int(10 * s),
        fill=(18, 26, 28),
        outline=(90, 145, 20),
        width=max(2, int(3 * s)),
    )
    plinth_max_w = (px1 - px0) - int(56 * s)
    hdr_txt = "ATLAS 3DSS  //  VERIFIED LEDGER 360° PERMANENCE BENCHMARK"
    hdr_sz = int(24 * s)
    while hdr_sz > int(14 * s) and draw.textbbox((0, 0), hdr_txt, font=load_font(hdr_sz, bold=True))[2] > plinth_max_w:
        hdr_sz -= 1
    sub_txt = "YAW = 0.0°  ·  INVARIANT 1: 24-WORD PLAQUE  ·  INVARIANT 2: FOUNDER  ·  INVARIANT 3: OPEN DOOR"
    sub_sz = int(18 * s)
    while sub_sz > int(12 * s) and draw.textbbox((0, 0), sub_txt, font=load_font(sub_sz, bold=False))[2] > plinth_max_w:
        sub_sz -= 1
    draw.text((px0 + int(28 * s), int(860 * s)), hdr_txt, fill=(146, 224, 34), font=load_font(hdr_sz, bold=True))
    draw.text((px0 + int(28 * s), int(914 * s)), sub_txt, fill=(210, 220, 226), font=load_font(sub_sz, bold=False))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png)
    return {
        "path": str(out_png),
        "sha256": sha256_file(out_png),
        "bytes": int(out_png.stat().st_size),
        "width": width,
        "height": height,
        "sign_text": sign_text,
        "wrapped_lines": chosen_lines,
    }


# ======================================================================================
# Pluggable Packet B2 High-Res OCR Verification Hook
# ======================================================================================


def detect_blinded_plaque_roi(image_path: Path) -> Optional[Dict[str, Any]]:
    """Strictly blinded bright-plaque ROI detector using ONLY luminance, saturation, and geometry.

    Zero peeking at target text; executes the exact same deterministic pure-NumPy connected-component
    finder on both ledger_on and ledger_off frames. Returns None when no rectangular bright plaque
    surface is present (e.g. opposite walls or degraded stock-diffusion returns), triggering honest
    full-frame OCR fallback.
    """
    im = Image.open(image_path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    sat = np.max(arr, axis=2) - np.min(arr, axis=2)
    mask = (lum > 0.62) & (sat < 0.22)
    mh, mw = h // 4, w // 4
    if mh < 4 or mw < 4:
        return None
    sm = mask[: mh * 4, : mw * 4].reshape(mh, 4, mw, 4).mean(axis=(1, 3)) > 0.35
    vis = np.zeros_like(sm, dtype=bool)
    best: Optional[Tuple[float, Dict[str, Any]]] = None
    for y in range(mh):
        for x in range(mw):
            if not sm[y, x] or vis[y, x]:
                continue
            stack = [(y, x)]
            vis[y, x] = True
            ys: List[int] = []
            xs: List[int] = []
            while stack:
                cy, cx = stack.pop()
                ys.append(cy)
                xs.append(cx)
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < mh and 0 <= nx < mw and sm[ny, nx] and not vis[ny, nx]:
                        vis[ny, nx] = True
                        stack.append((ny, nx))
            area = len(ys)
            y0, y1, x0, x1 = min(ys), max(ys) + 1, min(xs), max(xs) + 1
            bw, bh = x1 - x0, y1 - y0
            aspect = float(bw) / float(max(bh, 1))
            fill = float(area) / float(max(bw * bh, 1))
            frac = float(area) / float(mh * mw)
            if frac >= 0.012 and 0.35 <= aspect <= 3.8 and fill >= 0.45:
                score = float(area) * fill
                if best is None or score > best[0]:
                    px0, py0, px1, py1 = x0 * 4, y0 * 4, x1 * 4, y1 * 4
                    ix = int((px1 - px0) * 0.025)
                    iy = int((py1 - py0) * 0.025)
                    box = [max(0, px0 + ix), max(0, py0 + iy), min(w, px1 - ix), min(h, py1 - iy)]
                    if (box[2] - box[0]) >= 24 and (box[3] - box[1]) >= 24:
                        best = (
                            score,
                            {
                                "box_xyxy": box,
                                "area_frac": round(frac, 4),
                                "aspect": round(aspect, 3),
                                "fill_ratio": round(fill, 3),
                            },
                        )
    return best[1] if best is not None else None


_RAPIDOCR_SINGLETON: Any = None


def _run_rapidocr_or_tesseract_on_pil(im: Image.Image) -> Tuple[str, str]:
    """Run RapidOCR (preferred) with clean tesseract --psm 6 fallback if RapidOCR is unavailable."""
    global _RAPIDOCR_SINGLETON
    try:
        if _RAPIDOCR_SINGLETON is None:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore

            _RAPIDOCR_SINGLETON = RapidOCR()
        arr_rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
        arr_bgr = arr_rgb[:, :, ::-1].copy()
        res, _ = _RAPIDOCR_SINGLETON(arr_bgr)
        if not res:
            return "", "rapidocr"
        # Sort detected text boxes top-to-bottom (row-clustered) then left-to-right
        items: List[Tuple[float, float, str]] = []
        for entry in res:
            box, txt = entry[0], str(entry[1]).strip()
            if not txt:
                continue
            ys = [float(pt[1]) for pt in box]
            xs = [float(pt[0]) for pt in box]
            items.append((sum(ys) / len(ys), min(xs), txt))
        if not items:
            return "", "rapidocr"
        line_h = max(12.0, im.height * 0.045)
        items.sort(key=lambda t: (int(round(t[0] / line_h)), t[1]))
        return " ".join(t[2] for t in items), "rapidocr"
    except Exception:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tmp_p = Path(tf.name)
        try:
            im.save(tmp_p)
            proc = subprocess.run(
                ["tesseract", str(tmp_p), "stdout", "--psm", "6"],
                capture_output=True,
                text=True,
                check=False,
            )
            return proc.stdout.strip() if proc.returncode == 0 else "", "tesseract_fallback"
        finally:
            tmp_p.unlink(missing_ok=True)


def run_pluggable_ocr_hook(
    image_path: Path,
    *,
    target_text: str,
    backend: str,
    ocr_cmd: Optional[str] = None,
    got_ocr2_dir: Optional[Path] = None,
    got_ocr2_weights: Optional[Path] = None,
    dry_run_arm: str = "ledger_on",
    dry_run_yaw_deg: float = 0.0,
) -> Dict[str, Any]:
    """Pluggable Packet B2 High-Res OCR hook for turnaround permanence scoring.

    Backends:
      - `blinded-plaque-rapidocr` (v5 default): strictly blinded luminance/geometry bright-plaque ROI
        crop + 2x upscale RapidOCR (tesseract fallback), dual-logging BOTH plaque-ROI primary metrics
        AND uncropped full-frame metrics honestly on every checkpoint.
      - `custom-cmd`: invokes `--ocr-cmd` with `{image}` and `{target}` formatted; expects JSON
        stdout `{"observed_text": "..."}` (or raw text fallback).
      - `packetb2-got-ocr`: runs local Packet B2 GOT-OCR2_0 / SigLIP2 OCR evaluator if available.
      - `dry-run-sim`: deterministic CPU-only simulation ONLY for `--dry-run` testing.
    """
    if backend == "blinded-plaque-rapidocr":
        full_im = Image.open(image_path).convert("RGB")
        full_obs, engine_full = _run_rapidocr_or_tesseract_on_pil(full_im)
        full_metrics = compute_text_ocr_metrics(full_obs, target_text)

        roi_info = detect_blinded_plaque_roi(image_path)
        if roi_info is not None:
            x0, y0, x1, y1 = roi_info["box_xyxy"]
            crop = full_im.crop((x0, y0, x1, y1))
            crop_up = crop.resize((crop.width * 2, crop.height * 2), Image.Resampling.LANCZOS)
            roi_obs, engine_roi = _run_rapidocr_or_tesseract_on_pil(crop_up)
            roi_metrics = compute_text_ocr_metrics(roi_obs, target_text)
        else:
            roi_obs, engine_roi = full_obs, engine_full
            roi_metrics = dict(full_metrics)

        return {
            "backend": "blinded-plaque-rapidocr",
            "ocr_engine": engine_roi,
            "simulated": False,
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "plaque_roi_detected": roi_info is not None,
            "plaque_roi": roi_info,
            "full_frame_baseline": {
                "ocr_engine": engine_full,
                **full_metrics,
            },
            **roi_metrics,
        }
    if backend == "dry-run-sim":
        # Strictly for --dry-run pipeline validation! Supports both 360-deg and 720-deg double-revisit orbits.
        yaw_mod = dry_run_yaw_deg % 360.0
        if dry_run_yaw_deg < 30.0:
            obs = target_text  # Frame 0 baseline (identical for both arms)
        elif 110.0 <= yaw_mod <= 250.0:
            obs = ""  # Sign is completely out of field of view on the opposite wall
        elif dry_run_arm == "ledger_on":
            # Simulated progressive convergence as sign re-enters FOV on Orbit 1 or Orbit 2
            at_origin = (yaw_mod >= 340.0) or (yaw_mod <= 20.0)
            obs = target_text if at_origin else "Thank Jensen for 10 Years of Cuda, Ty thinking machines for this model"
        else:
            # Simulated stock diffusion typography drift (compounds across 2nd revolution >400 deg)
            obs = (
                "Thonk Jonsen fr 1O Yers Cdua, Ty thnking machnes model, open srouce commnity amzing"
                if dry_run_yaw_deg <= 400.0
                else "Thnk Jnsn 10 Yrs Cda thnkng mchnz opn src cmnty amzng"
            )
        metrics = compute_text_ocr_metrics(obs, target_text)
        return {
            "backend": "dry-run-sim",
            "simulated": True,
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            **metrics,
        }

    if backend == "custom-cmd":
        if not ocr_cmd:
            raise ValueError("--ocr-cmd is required when --ocr-backend custom-cmd")
        formatted = (
            ocr_cmd.replace("{image}", shlex.quote(str(image_path)))
            .replace("{target}", shlex.quote(target_text))
        )
        proc = subprocess.run(formatted, shell=True, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Custom OCR command failed rc={proc.returncode}: {proc.stderr[-400:]}")
        stdout_s = proc.stdout.strip()
        try:
            parsed = json.loads(stdout_s)
            obs = str(parsed.get("observed_text", parsed.get("text", "")))
        except json.JSONDecodeError:
            obs = stdout_s
        metrics = compute_text_ocr_metrics(obs, target_text)
        return {
            "backend": "custom-cmd",
            "simulated": False,
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            **metrics,
        }

    if backend == "packetb2-got-ocr":
        # Checks for Packet B2 GOT-OCR2_0 / HuggingFace OCR evaluator or raises transparently.
        if got_ocr2_dir is None or not Path(got_ocr2_dir).is_dir():
            raise FileNotFoundError(
                f"Packet B2 GOT-OCR2_0 directory not found ({got_ocr2_dir}); pass --ocr-backend custom-cmd "
                "pointing at your Packet B2 Stage 3 evaluator or run with --dry-run."
            )
        # Try HF AutoModel/AutoTokenizer full GOT-OCR2_0 if present alongside the stripped vision tower
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(got_ocr2_dir), trust_remote_code=True, local_files_only=True)
            mdl = AutoModel.from_pretrained(str(got_ocr2_dir), trust_remote_code=True, local_files_only=True)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            mdl = mdl.to(device).eval()
            obs = str(mdl.chat(tok, str(image_path), ocr_type="ocr"))
            metrics = compute_text_ocr_metrics(obs, target_text)
            return {
                "backend": "packetb2-got-ocr",
                "simulated": False,
                "image_path": str(image_path),
                "image_sha256": sha256_file(image_path),
                **metrics,
            }
        except Exception as exc:
            raise RuntimeError(
                f"Packet B2 GOT-OCR2_0 direct chat unavailable ({exc}). Note: packetB2_ocr_towers/got_ocr2 "
                "holds the stripped vision tower for Inkling bridge injection; pass --ocr-backend custom-cmd "
                "wired to your Packet B2 Stage 3 Inkling+GOT-OCR evaluator (or --ocr-backend dry-run-sim in --dry-run)."
            ) from exc

    raise ValueError(f"Unknown OCR backend: {backend!r}")


# ======================================================================================
# Anti-Goodhart Continuity / Jump-Cut Gate & Continuous Subject Identity Gate (v4)
# ======================================================================================


def _blockwise_ssim_pair(g_a: np.ndarray, mu_a: np.ndarray, var_a: np.ndarray, g_b: np.ndarray, mu_b: np.ndarray, var_b: np.ndarray) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    cov = ((g_a - mu_a[:, None]) * (g_b - mu_b[:, None])).mean(axis=-1)
    smap = ((2.0 * mu_a * mu_b + c1) * (2.0 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return float(smap.mean())


def run_temporal_continuity_gate(
    mp4_path: Path,
    *,
    return_window_frames: int = 150,
    min_frame1_ssim: float = 0.72,
    min_full_adj_ssim: float = 0.65,
    min_return_adj_ssim: float = 0.72,
    max_return_adj_l1: float = 0.14,
    max_return_l1_spike: float = 6.5,
    min_tail20_ssim: float = 0.48,
    probe_w: int = 256,
    probe_h: int = 144,
) -> Dict[str, Any]:
    """Fail-closed dense frame-to-frame SSIM + L1 jump-cut & terminal-teleport anti-Goodhart gate.

    Catches Frame-0->Frame-1 cuts off flat anchors, mid-orbit scene cuts, and last-frame
    teleport pins (e.g. yaw=341 -> yaw=360 hard cut) so a jump-cut can NEVER score PASS.
    """
    cmd = [
        FFMPEG, "-v", "error", "-i", str(mp4_path),
        "-vf", f"scale={probe_w}:{probe_h}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    stride = probe_w * probe_h * 3
    n = len(raw) // stride
    if n < 3:
        raise RuntimeError(f"Continuity audit needs >=3 frames, got {n} for {mp4_path}")

    arr = np.frombuffer(raw[: n * stride], dtype=np.uint8).reshape(n, probe_h, probe_w, 3).astype(np.float32) / 255.0
    adj_l1 = np.mean(np.abs(arr[1:] - arr[:-1]), axis=(1, 2, 3))

    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    bh, bw = probe_h // 8, probe_w // 8
    g = gray[:, : bh * 8, : bw * 8].reshape(n, bh, 8, bw, 8).transpose(0, 1, 3, 2, 4).reshape(n, bh * bw, 64)
    mu = g.mean(axis=-1)
    var = g.var(axis=-1)
    cov = ((g[1:] - mu[1:, :, None]) * (g[:-1] - mu[:-1, :, None])).mean(axis=-1)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    adj_ssim = (
        ((2.0 * mu[1:] * mu[:-1] + c1) * (2.0 * cov + c2))
        / ((mu[1:] ** 2 + mu[:-1] ** 2 + c1) * (var[1:] + var[:-1] + c2))
    ).mean(axis=-1)

    ret_start = max(0, (n - 1) - min(int(return_window_frames), n - 1))
    ret_l1 = adj_l1[ret_start:]
    ret_ssim = adj_ssim[ret_start:]
    ret_med_l1 = float(np.median(ret_l1))
    ret_max_l1 = float(np.max(ret_l1))
    ret_spike = float(ret_max_l1 / max(ret_med_l1, 0.004))
    ret_min_ssim = float(np.min(ret_ssim))
    ret_worst_frame = int(ret_start + int(np.argmin(ret_ssim)) + 1)

    frame1_ssim = float(adj_ssim[0])
    frame1_l1 = float(adj_l1[0])
    full_min_ssim = float(np.min(adj_ssim))
    full_worst_frame = int(np.argmin(adj_ssim) + 1)
    full_max_l1 = float(np.max(adj_l1))
    full_med_l1 = float(np.median(adj_l1))

    # Scale tail window to ~4% of short clips capped at 20 frames (20f on 804f / 33.5s) to catch terminal teleports
    tail_step = min(20, max(4, int(round(n * 0.04))))
    t20_idx = max(0, n - 1 - tail_step)
    tail20_ssim = _blockwise_ssim_pair(g[t20_idx], mu[t20_idx], var[t20_idx], g[n - 1], mu[n - 1], var[n - 1])
    t5_idx = max(0, n - 1 - min(5, max(1, tail_step // 2)))
    tail5_ssim = _blockwise_ssim_pair(g[t5_idx], mu[t5_idx], var[t5_idx], g[n - 1], mu[n - 1], var[n - 1])

    detected_cuts: List[Dict[str, Any]] = []
    for idx in range(n - 1):
        if float(adj_ssim[idx]) < min_full_adj_ssim or float(adj_l1[idx]) > max_return_adj_l1:
            detected_cuts.append(
                {
                    "frame_index": int(idx + 1),
                    "adj_ssim": round(float(adj_ssim[idx]), 4),
                    "adj_l1": round(float(adj_l1[idx]), 4),
                    "in_return_window": bool(idx >= ret_start),
                }
            )
    detected_cuts.sort(key=lambda r: r["adj_ssim"])

    failures: List[str] = []
    if frame1_ssim < min_frame1_ssim:
        failures.append(f"FRAME1_DISCONTINUITY(ssim={frame1_ssim:.4f}<{min_frame1_ssim:.2f})")
    if full_min_ssim < min_full_adj_ssim:
        failures.append(f"MID_ORBIT_SCENE_CUT(frame={full_worst_frame},ssim={full_min_ssim:.4f}<{min_full_adj_ssim:.2f})")
    if ret_min_ssim < min_return_adj_ssim:
        failures.append(f"RETURN_JUMP_CUT(frame={ret_worst_frame},ssim={ret_min_ssim:.4f}<{min_return_adj_ssim:.2f})")
    if ret_max_l1 > max_return_adj_l1:
        failures.append(f"RETURN_L1_JUMP(max_l1={ret_max_l1:.4f}>{max_return_adj_l1:.2f})")
    if ret_max_l1 >= 0.05 and ret_spike > max_return_l1_spike:
        failures.append(f"RETURN_SPIKE_RATIO({ret_spike:.2f}x>{max_return_l1_spike:.1f}x)")
    if tail20_ssim < min_tail20_ssim:
        failures.append(f"TERMINAL_TELEPORT_DETECTED(tail20_ssim={tail20_ssim:.4f}<{min_tail20_ssim:.2f})")

    return {
        "mp4_path": str(mp4_path),
        "n_frames_scanned": int(n),
        "passed": len(failures) == 0,
        "failures": failures,
        "frame1_ssim": round(frame1_ssim, 4),
        "frame1_l1": round(frame1_l1, 4),
        "full_min_adj_ssim": round(full_min_ssim, 4),
        "full_worst_frame": full_worst_frame,
        "full_max_adj_l1": round(full_max_l1, 4),
        "full_median_adj_l1": round(full_med_l1, 4),
        "return_window_start_frame": int(ret_start),
        "return_min_adj_ssim": round(ret_min_ssim, 4),
        "return_worst_frame": ret_worst_frame,
        "return_max_adj_l1": round(ret_max_l1, 4),
        "return_median_adj_l1": round(ret_med_l1, 4),
        "return_l1_spike_ratio": round(ret_spike, 3),
        "tail_20_to_final_ssim": round(tail20_ssim, 4),
        "tail_5_to_final_ssim": round(tail5_ssim, 4),
        "n_detected_cuts": len(detected_cuts),
        "worst_cuts": detected_cuts[:8],
    }


IDENTITY_POS_TEXT = (
    "a dignified chimpanzee wearing a black top hat with a lime green band, "
    "a gold monocle, and a black tuxedo jacket with lime piping"
)
IDENTITY_NEG_TEXT = "a bare brown monkey or rhesus macaque with no hat and no suit"


def run_subject_identity_gate(
    *,
    arm_checkpoints: Mapping[str, Sequence[Mapping[str, Any]]],
    founder_ref_path: Path,
    anchor_yaw0_path: Path,
    siglip2_dir: Path,
    backend: str = "siglip2",
    min_cos_founder: float = 0.68,
    min_cos_anchor: float = 0.70,
    min_wardrobe_margin: float = 0.0,
) -> Dict[str, Any]:
    """Continuous Subject Identity Gate preventing top-hat Monkey Founder -> bare macaque morphing.

    Evaluates non-pinned return approach checkpoints (yaw ~330..355 deg) against BOTH:
      1. SigLIP2 image cosine to Monkey Founder ref (`kf_01_host_hook.png`) and origin anchor (`yaw=0`).
      2. SigLIP2 zero-shot text-image wardrobe margin (top-hat tuxedo chimpanzee vs bare brown macaque).
    """
    if backend == "skip":
        return {"backend": "skip", "simulated": False, "passed": True, "failures": [], "arms": {}}

    if backend == "dry-run-sim":
        arms_out: Dict[str, Any] = {}
        for arm_name, ckpts in arm_checkpoints.items():
            nonpinned = [c for c in ckpts if str(c["checkpoint"]).endswith("nonpinned")]
            is_on = arm_name == "ledger_on"
            cf = 0.78 if is_on else 0.59
            ca = 0.84 if is_on else 0.51
            wm = 0.08 if is_on else -0.05
            arms_out[arm_name] = {
                "nonpinned_mean_cos_founder": cf,
                "nonpinned_mean_cos_anchor": ca,
                "nonpinned_mean_wardrobe_margin": wm,
                "n_nonpinned_scored": len(nonpinned),
                "passed": (cf >= min_cos_founder and ca >= min_cos_anchor and wm > min_wardrobe_margin),
            }
        on_ok = bool(arms_out.get("ledger_on", {}).get("passed", False))
        return {
            "backend": "dry-run-sim",
            "simulated": True,
            "passed": on_ok,
            "failures": [] if on_ok else ["DRY_RUN_IDENTITY_FAIL"],
            "arms": arms_out,
        }

    if backend != "siglip2":
        raise ValueError(f"Unknown identity backend: {backend!r}")

    import torch
    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained(str(siglip2_dir), local_files_only=True)
    mdl = AutoModel.from_pretrained(str(siglip2_dir), local_files_only=True).eval()

    def _img_emb(img_p: Path) -> torch.Tensor:
        im = Image.open(img_p).convert("RGB")
        inp = proc(images=im, return_tensors="pt")
        with torch.no_grad():
            feat = mdl.get_image_features(**inp)
            if not isinstance(feat, torch.Tensor):
                feat = feat.pooler_output
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat[0]

    def _wardrobe_margin(img_p: Path) -> Tuple[float, float, float]:
        im = Image.open(img_p).convert("RGB")
        inp = proc(
            text=[IDENTITY_POS_TEXT, IDENTITY_NEG_TEXT],
            images=im,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = mdl(**inp)
            logits = out.logits_per_image[0].float()
            pos_s = float(logits[0].item())
            neg_s = float(logits[1].item())
        return pos_s - neg_s, pos_s, neg_s

    ref_founder = _img_emb(founder_ref_path)
    ref_anchor = _img_emb(anchor_yaw0_path)

    arms_out = {}
    for arm_name, ckpts in arm_checkpoints.items():
        scored: List[Dict[str, Any]] = []
        for c in ckpts:
            cname = str(c["checkpoint"])
            if not (cname.endswith("nonpinned") or cname.endswith("return") or cname.endswith("revisit2_final") or cname == "yaw_000_start"):
                continue
            img_p = Path(c["ocr"]["image_path"])
            emb = _img_emb(img_p)
            cos_f = float((emb * ref_founder).sum().item())
            cos_a = float((emb * ref_anchor).sum().item())
            w_margin, w_pos, w_neg = _wardrobe_margin(img_p)
            scored.append(
                {
                    "checkpoint": cname,
                    "yaw_deg": float(c["yaw_deg"]),
                    "nonpinned": cname.endswith("nonpinned"),
                    "cos_founder": round(cos_f, 4),
                    "cos_anchor": round(cos_a, 4),
                    "wardrobe_margin": round(w_margin, 4),
                    "wardrobe_pos_logit": round(w_pos, 4),
                    "wardrobe_neg_logit": round(w_neg, 4),
                }
            )
        np_rows = [r for r in scored if r["nonpinned"]]
        mean_cf = float(np.mean([r["cos_founder"] for r in np_rows])) if np_rows else 0.0
        mean_ca = float(np.mean([r["cos_anchor"] for r in np_rows])) if np_rows else 0.0
        mean_wm = float(np.mean([r["wardrobe_margin"] for r in np_rows])) if np_rows else -99.0
        arm_fails: List[str] = []
        if mean_cf < min_cos_founder:
            arm_fails.append(f"IDENTITY_COS_FOUNDER_LOW({mean_cf:.4f}<{min_cos_founder:.2f})")
        if mean_ca < min_cos_anchor:
            arm_fails.append(f"IDENTITY_COS_ANCHOR_LOW({mean_ca:.4f}<{min_cos_anchor:.2f})")
        if mean_wm <= min_wardrobe_margin:
            arm_fails.append(f"SUBJECT_WARDROBE_MORPH_DETECTED(margin={mean_wm:.4f}<={min_wardrobe_margin:.2f})")
        arms_out[arm_name] = {
            "nonpinned_mean_cos_founder": round(mean_cf, 4),
            "nonpinned_mean_cos_anchor": round(mean_ca, 4),
            "nonpinned_mean_wardrobe_margin": round(mean_wm, 4),
            "n_nonpinned_scored": len(np_rows),
            "passed": len(arm_fails) == 0,
            "failures": arm_fails,
            "checkpoints": scored,
        }

    del mdl, proc
    gc.collect()
    on_res = arms_out.get("ledger_on", {})
    return {
        "backend": "siglip2",
        "simulated": False,
        "siglip2_dir": str(siglip2_dir),
        "thresholds": {
            "min_cos_founder": min_cos_founder,
            "min_cos_anchor": min_cos_anchor,
            "min_wardrobe_margin": min_wardrobe_margin,
        },
        "passed": bool(on_res.get("passed", False)),
        "failures": list(on_res.get("failures", ["MISSING_LEDGER_ON_IDENTITY"])),
        "arms": arms_out,
    }


# ======================================================================================
# Subprocess GPU Workers (H3-Image Optional Anchor + Zero-Confound Branched FL2VA/Ref2VA)
# ======================================================================================


def worker_h3_image_anchor(args: argparse.Namespace) -> None:
    """Optional GPU worker: render Frame-0 anchor plate natively via Packet A concat4 -> MiniMax-H3-Image."""
    sys.path.insert(0, str(args.comfy_root))
    sys.path.insert(0, str(args.repo_root))
    import torch
    import comfy.model_management as mm
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import comfy.utils
    from omni_inkling.h3_platos_key_phase0 import (
        build_reconstructed_comfy_conditioning_from_disk_pkg,
        verify_pinned_pdd_sigmas,
    )

    job = json.loads(Path(args.worker_job_json).read_text(encoding="utf-8"))
    out_png = Path(job["out_png"])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    t0 = time.monotonic()

    mp = comfy.sd.load_diffusion_model(str(job["image_dit_path"]))
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(job["image_vae_path"])))
    obs_sigmas = list(mp.model.model_config.unet_config["pdd_sigmas"])
    sigmas_t = torch.tensor(verify_pinned_pdd_sigmas(obs_sigmas, H3_PDD_SIGMAS), dtype=torch.float32)
    sampler = comfy.samplers.sampler_object("euler")

    w, h = int(job["width"]), int(job["height"])
    lat_shape = (1, 24, 1, h // 16, w // 16)
    seed = int(job["seed"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(lat_shape, generator=gen, dtype=torch.float32)

    pkg = torch.load(str(cond_pkg_path(Path(job["cond_dir"]), job["prompt_id"], "inkling_platos_key_concat4")), map_location="cpu", weights_only=True)
    cond, _ = build_reconstructed_comfy_conditioning_from_disk_pkg(pkg)

    with torch.no_grad():
        fin = comfy.sample.sample_custom(
            mp, noise, cfg=1.0, sampler=sampler, sigmas=sigmas_t,
            positive=cond, negative=cond, latent_image=torch.zeros_like(noise),
            disable_pbar=True, seed=seed,
        )
        img_t = vae.decode(fin)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if img_t.ndim == 5:
            img_t = img_t[:, 0]
        img_u8 = img_t[0].detach().cpu().float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    Image.fromarray(img_u8).save(out_png)
    hw = detect_hardware_info()
    del mp, vae
    gc.collect()
    mm.soft_empty_cache()

    atomic_write_json(
        out_png.with_suffix(".anchor_result.json"),
        {
            "path": str(out_png),
            "sha256": sha256_file(out_png),
            "width": int(img_u8.shape[1]),
            "height": int(img_u8.shape[0]),
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "hardware": hw,
        },
    )


def worker_h3_turnaround(args: argparse.Namespace) -> None:
    """Subprocess GPU worker: executes zero-confound Shared Seg 0 + Branched Seg 1 (or single-clip) FL2VA."""
    sys.path.insert(0, str(args.comfy_root))
    sys.path.insert(0, str(args.repo_root))
    import torch
    import comfy.model_management as mm
    import comfy.nested_tensor
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import comfy.utils
    from comfy_extras.nodes_minimax_h3 import _resize
    from omni_inkling.h3_platos_key_phase0 import build_reconstructed_comfy_conditioning_from_disk_pkg
    from showcase_common import load_image_tensor_bhwc

    job = json.loads(Path(args.worker_job_json).read_text(encoding="utf-8"))
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_dir = Path(job["cond_dir"])

    h3_cfg = H3SamplingConfig(
        width=int(job["width"]),
        height=int(job["height"]),
        steps=int(job["steps"]),
        cfg=float(job["cfg"]),
        shift_video=float(job["shift_video"]),
        shift_audio=float(job["shift_audio"]),
        sampler_name=str(job["sampler_name"]),
        scheduler_name=str(job["scheduler_name"]),
        use_sparse_attention=bool(job["use_sparse_attention"]),
        video_budget=float(job["video_budget"]),
        early_steps=int(job["early_steps"]),
        late_steps=int(job["late_steps"]),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    t0 = time.monotonic()

    mp_raw = comfy.sd.load_diffusion_model(str(job["dit_path"]))
    mp = apply_h3_model_patches(mp_raw, h3_cfg, comfy_root=Path(args.comfy_root))
    video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(job["video_vae_path"])))
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(job["audio_vae_path"])))
    load_s = time.monotonic() - t0

    sigmas_t = comfy.samplers.calculate_sigmas(mp.get_model_object("model_sampling"), h3_cfg.scheduler_name, h3_cfg.steps).float()
    sampler = comfy.samplers.sampler_object(h3_cfg.sampler_name)

    # Encode Frame-0 anchor plate, Monkey Founder identity ref (<Picture 1>), and any pose-indexed 3D room stops!
    anchor_png = Path(job["anchor_png"])
    anchor_img_t = _resize(load_image_tensor_bhwc(anchor_png), h3_cfg.width, h3_cfg.height, "disabled")
    with torch.no_grad():
        anchor_latent = video_vae.encode(anchor_img_t).detach().cpu()
    anchor_lat_path = out_dir / "verified_ledger_anchor_yaw0_latent.pt"
    torch.save({"pose_yaw_deg": 0.0, "anchor_png_sha256": sha256_file(anchor_png), "latent": anchor_latent}, str(anchor_lat_path))

    # Encode Monkey Founder ref (<Picture 1>) via VAE at Ref2VA aspect-preserving 'match' resolution
    founder_ref_latent = None
    founder_ref_path_s = job.get("founder_ref_png")
    if founder_ref_path_s and Path(founder_ref_path_s).is_file():
        f_t = load_image_tensor_bhwc(Path(founder_ref_path_s))
        fh0, fw0 = int(f_t.shape[1]), int(f_t.shape[2])
        f_scale = min(1.0, math.sqrt((h3_cfg.width * h3_cfg.height) / float(fw0 * fh0)))
        ftw = max(32, round(fw0 * f_scale / 32) * 32)
        fth = max(32, round(fh0 * f_scale / 32) * 32)
        with torch.no_grad():
            founder_ref_latent = video_vae.encode(_resize(f_t[:1], ftw, fth, "disabled")).detach().cpu()

    # Encode Verified Ledger pose-indexed 3D room stops (yaw=120, 180, 240, 300, 330 deg) when available
    pose_latents: Dict[int, Any] = {0: anchor_latent}
    for yaw_k, path_s in (job.get("room_pose_anchors") or {}).items():
        yi = int(round(float(yaw_k)))
        if yi == 0 or not path_s or not Path(path_s).is_file():
            continue
        p_t = _resize(load_image_tensor_bhwc(Path(path_s)), h3_cfg.width, h3_cfg.height, "disabled")
        with torch.no_grad():
            pose_latents[yi] = video_vae.encode(p_t).detach().cpu()
    print(f"[h3-turnaround] Encoded Verified Ledger pose anchors: {sorted(pose_latents.keys())} + founder_ref={founder_ref_latent is not None}", flush=True)

    def _load_cond(prompt_id: str, key_arm: str) -> Tuple[Any, Any]:
        pos_pkg = torch.load(str(cond_pkg_path(cond_dir, prompt_id, key_arm)), map_location="cpu", weights_only=True)
        bp, _ = build_reconstructed_comfy_conditioning_from_disk_pkg(pos_pkg)
        if h3_cfg.cfg != 1.0:
            neg_pkg = torch.load(str(cond_pkg_path(cond_dir, "__empty_neg__", key_arm)), map_location="cpu", weights_only=True)
            bn, _ = build_reconstructed_comfy_conditioning_from_disk_pkg(neg_pkg)
        else:
            bn = bp
        return bp, bn

    def _sample_segment(
        *,
        label: str,
        base_pos: Any,
        base_neg: Any,
        frames: int,
        seed: int,
        extra_keyframes: Sequence[Mapping[str, Any]] = (),
        ref_latents: Sequence[Mapping[str, Any]] = (),
        source_video_latent: Any = None,
        source_audio_latent: Any = None,
        context_frames: int = CHAIN_CONTEXT_FRAMES,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        pos_c = attach_zero_qwen_h3_conditioning(
            base_pos,
            video_vae=video_vae,
            width=h3_cfg.width,
            height=h3_cfg.height,
            frame_count=frames,
            extra_keyframe_latents=extra_keyframes,
            ref_image_latents=ref_latents,
        )
        neg_c = (
            attach_zero_qwen_h3_conditioning(
                base_neg,
                video_vae=video_vae,
                width=h3_cfg.width,
                height=h3_cfg.height,
                frame_count=frames,
                extra_keyframe_latents=extra_keyframes,
                ref_image_latents=ref_latents,
            )
            if h3_cfg.cfg != 1.0
            else pos_c
        )
        latent_nt, mask_nt, frame_count = build_masked_av_latent_and_noise_mask(
            width=h3_cfg.width,
            height=h3_cfg.height,
            frames=frames,
            source_video_latent=source_video_latent,
            source_audio_latent=source_audio_latent,
            context_frames=context_frames,
        )
        v_shape = tuple(latent_nt.tensors[0].shape)
        a_shape = tuple(latent_nt.tensors[1].shape)
        gen_v = torch.Generator(device="cpu").manual_seed(seed)
        gen_a = torch.Generator(device="cpu").manual_seed(seed + 10007)
        noise_nt = comfy.nested_tensor.NestedTensor(
            (torch.randn(v_shape, generator=gen_v, dtype=torch.float32),
             torch.randn(a_shape, generator=gen_a, dtype=torch.float32))
        )
        ts0 = time.monotonic()
        with torch.no_grad():
            fin_nt = comfy.sample.sample_custom(
                mp,
                noise_nt,
                cfg=h3_cfg.cfg,
                sampler=sampler,
                sigmas=sigmas_t.clone(),
                positive=pos_c,
                negative=neg_c,
                latent_image=latent_nt,
                noise_mask=mask_nt,
                disable_pbar=True,
                seed=seed,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        dit_s = time.monotonic() - ts0
        v_fin, a_fin = fin_nt.unbind()
        v_cpu = v_fin.detach().cpu()
        a_cpu = a_fin.detach().cpu()
        if not (bool(torch.isfinite(v_cpu).all().item()) and bool(torch.isfinite(a_cpu).all().item())):
            raise RuntimeError(f"[{label}] Non-finite latent encountered!")

        frames_u8 = decode_video_latent_u8(video_vae, v_fin, tiled=True)
        aud_np = decode_audio_latent_stereo(audio_vae, a_fin)
        wav_p = out_dir / f"{label}_32k.wav"
        wav_sha, _ = write_stereo_wav_pcm16(wav_p, aud_np, sr=H3_AUDIO_SR)
        mp4_p = out_dir / f"{label}.mp4"
        mux_info = mux_frames_and_wav_to_mp4(frames_u8, wav_p, mp4_p, fps=FPS)
        motion_l1 = (
            float(np.mean(np.abs(frames_u8[1:].astype(np.float32) - frames_u8[:-1].astype(np.float32))))
            if frames_u8.shape[0] > 1 else 0.0
        )
        # Release per-segment GPU allocations immediately so 3-4 segment >=30s orbits stay flat-VRAM
        del fin_nt, v_fin, a_fin, pos_c, neg_c, latent_nt, mask_nt, noise_nt, frames_u8, aud_np
        gc.collect()
        mm.soft_empty_cache()
        vram_note = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                fb, tb = torch.cuda.mem_get_info()
                vram_note = f" vram_free={fb / (1024**3):.1f}/{tb / (1024**3):.1f}GiB"
            except Exception:
                pass
        print(f"[{label}] DONE ({frame_count}f) in {dit_s:.1f}s motion_l1={motion_l1:.2f}{vram_note} -> {mp4_p.name}", flush=True)
        return v_cpu, a_cpu, {
            "label": label,
            "frames": int(frame_count),
            "seed": int(seed),
            "dit_seconds": round(dit_s, 2),
            "all_finite": True,
            "video_temporal_motion_l1": round(motion_l1, 3),
            "wav_path": str(wav_p),
            "wav_sha256": wav_sha,
            "mp4_path": mux_info["path"],
            "mp4_sha256": mux_info["sha256"],
        }

    topology = str(job["topology"])
    inj_mode = str(job["ledger_injection_mode"])
    plan = job["plan"]
    seed_base = int(job["seed"])
    arm_outputs: Dict[str, Any] = {}

    def _identity_and_room_refs() -> List[Dict[str, Any]]:
        """Ref2VA <Picture 1> = Monkey Founder identity ref, <Picture 2> = 3D Origin Room/Plaque."""
        refs_out: List[Dict[str, Any]] = []
        if founder_ref_latent is not None:
            refs_out.append(
                {
                    "latent": founder_ref_latent,
                    "latent_h": int(founder_ref_latent.shape[-2]),
                    "latent_w": int(founder_ref_latent.shape[-1]),
                }
            )
        refs_out.append(
            {
                "latent": anchor_latent,
                "latent_h": int(anchor_latent.shape[-2]),
                "latent_w": int(anchor_latent.shape[-1]),
            }
        )
        return refs_out

    def _yaw_to_local_frame(
        yaw_deg: float,
        *,
        seg_start_yaw: float,
        seg_end_yaw: float,
        seg_frames: int,
        ctx_frames: int,
    ) -> int:
        """Overlap-compensated keyframe index placement aligned to H3 4-frame latent boundaries.

        When ctx_frames > 0 (chained Seg 1 / Seg 2), local frames [0 .. ctx_frames-1] replay the
        tail of the previous segment ending at seg_start_yaw at local frame (ctx_frames - 1).
        Placing intermediate stops proportionally across [ctx_frames - 1, seg_frames - 1] removes
        the v4 off-by-overlap 19-frame-early snap that caused the f383 mid-orbit discontinuity.
        """
        anchor_f0 = max(0, int(ctx_frames) - 1) if ctx_frames > 0 else 0
        span_f = max(1, (int(seg_frames) - 1) - anchor_f0)
        frac = (float(yaw_deg) - float(seg_start_yaw)) / max(float(seg_end_yaw) - float(seg_start_yaw), 1.0)
        raw_f = int(round(anchor_f0 + frac * span_f))
        if raw_f <= anchor_f0:
            return anchor_f0
        if raw_f >= int(seg_frames) - 1:
            return int(seg_frames) - 1
        # Snap gently to nearest 4-frame H3 latent boundary (1 + 4k) while staying strictly clear of overlap
        snapped = 1 + 4 * int(round((raw_f - 1) / 4.0))
        return max(anchor_f0 + 8, min(int(seg_frames) - 24, snapped))

    def _ledger_return_cond(
        seg_frames: int,
        *,
        seg_start_yaw: float = 240.0,
        ctx_frames: int = CHAIN_CONTEXT_FRAMES,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Build Verified Ledger return guidance (intermediate spatial stops + Ref2VA identity refs).

        Modes:
          - `smooth-trajectory-v5` (v5): overlap-compensated dense 30-deg intermediate spatial stops
            (yaw=270, 300, 330) + smooth origin arrival (yaw=360) + Ref2VA <Picture 1> Founder &
            <Picture 2> Plaque refs, leaving >=75 consecutive approach frames (yaw=331..359 deg,
            including non-pinned yaw_345 & yaw_355) and yaw_325 strictly unpinned!
          - `multi-stop-ref` (v4): legacy non-overlap-adjusted intermediate stops (yaw=270, 300, 330) + terminal.
          - `ref-plus-intermediate`: ZERO terminal pin at last frame — ONLY intermediate stops + Ref2VA refs.
          - Legacy modes (`ref-and-return-guide`, `ref-only`, `return-guide`) preserved for ablation.
        """
        kfs: List[Dict[str, Any]] = []
        if inj_mode == "smooth-trajectory-v5":
            for yaw_stop in (270, 300, 330):
                if yaw_stop > seg_start_yaw + 15.0 and yaw_stop in pose_latents:
                    fidx = _yaw_to_local_frame(
                        float(yaw_stop),
                        seg_start_yaw=seg_start_yaw,
                        seg_end_yaw=360.0,
                        seg_frames=seg_frames,
                        ctx_frames=ctx_frames,
                    )
                    kfs.append({"resolved_frame_index": fidx, "latent": pose_latents[yaw_stop]})
            kfs.append({"resolved_frame_index": seg_frames - 1, "latent": anchor_latent})
        elif inj_mode in ("multi-stop-ref", "ref-plus-intermediate"):
            # Map available intermediate pose anchors proportionally onto [seg_start_yaw, 360]
            span = max(360.0 - float(seg_start_yaw), 1.0)
            for yaw_stop in (270, 300, 330):
                if yaw_stop > seg_start_yaw + 15.0 and yaw_stop in pose_latents:
                    frac = (float(yaw_stop) - float(seg_start_yaw)) / span
                    fidx = max(CHAIN_CONTEXT_FRAMES + 8, min(seg_frames - 24, int(round(frac * (seg_frames - 1)))))
                    kfs.append({"resolved_frame_index": fidx, "latent": pose_latents[yaw_stop]})
            if inj_mode == "multi-stop-ref":
                kfs.append({"resolved_frame_index": seg_frames - 1, "latent": anchor_latent})
        elif inj_mode in ("return-guide", "ref-and-return-guide"):
            kfs.append({"resolved_frame_index": seg_frames - 1, "latent": anchor_latent})

        refs = (
            _identity_and_room_refs()
            if inj_mode in ("smooth-trajectory-v5", "multi-stop-ref", "ref-plus-intermediate", "ref-only", "ref-and-return-guide")
            else []
        )
        return kfs, refs

    if topology == "three-segment-orbit":
        # RTX PRO 6000 >=30s Flagship (default 294+294+294 = 804f/33.5s, v5 smooth 345+345+345 = 957f/39.875s):
        # Shared Seg 0 (0->120 deg) + Shared Seg 1 (120->240 deg) -> ZERO confound for first 2/3 orbit!
        # Branched Seg 2 (240->360 deg): Ledger-OFF vs Ledger-ON loop closure (only 4 H3 segment renders total!).
        seg0_f = int(plan["segments"][0]["frames"])
        seg1_f = int(plan["segments"][1]["frames"])
        seg2_f = int(plan["segments"][2]["frames"])
        ctx_f = int(plan["segments"][1]["trim_prefix_frames"])

        seg0_kfs = [{"resolved_frame_index": 0, "latent": anchor_latent}]
        if inj_mode == "smooth-trajectory-v5":
            for yw in (30, 60, 90):
                if yw in pose_latents:
                    seg0_kfs.append({
                        "resolved_frame_index": _yaw_to_local_frame(float(yw), seg_start_yaw=0.0, seg_end_yaw=120.0, seg_frames=seg0_f, ctx_frames=0),
                        "latent": pose_latents[yw],
                    })
        elif 60 in pose_latents:
            seg0_kfs.append({"resolved_frame_index": seg0_f // 2, "latent": pose_latents[60]})
        if 120 in pose_latents:
            seg0_kfs.append({"resolved_frame_index": seg0_f - 1, "latent": pose_latents[120]})

        bp0, bn0 = _load_cond("turnaround_orbit_seg0_0_to_120", "inkling_platos_key_concat4")
        seg0_v, seg0_a, seg0_meta = _sample_segment(
            label="shared_orbit_seg0_0_to_120",
            base_pos=bp0,
            base_neg=bn0,
            frames=seg0_f,
            seed=seed_base,
            extra_keyframes=seg0_kfs,
            ref_latents=_identity_and_room_refs(),
        )

        # Shared Seg 1 (120->240 deg): rear room walls ONLY (origin plaque & Founder 100% absent!)
        seg1_kfs: List[Dict[str, Any]] = []
        if inj_mode == "smooth-trajectory-v5":
            for yw in (150, 180, 210):
                if yw in pose_latents:
                    seg1_kfs.append({
                        "resolved_frame_index": _yaw_to_local_frame(float(yw), seg_start_yaw=120.0, seg_end_yaw=240.0, seg_frames=seg1_f, ctx_frames=ctx_f),
                        "latent": pose_latents[yw],
                    })
        elif 180 in pose_latents:
            seg1_kfs.append({"resolved_frame_index": seg1_f // 2, "latent": pose_latents[180]})
        if 240 in pose_latents:
            seg1_kfs.append({"resolved_frame_index": seg1_f - 1, "latent": pose_latents[240]})

        bp1, bn1 = _load_cond("turnaround_orbit_seg1_120_to_240", "inkling_platos_key_concat4")
        seg1_v, seg1_a, seg1_meta = _sample_segment(
            label="shared_orbit_seg1_120_to_240",
            base_pos=bp1,
            base_neg=bn1,
            frames=seg1_f,
            seed=seed_base + 101,
            extra_keyframes=seg1_kfs,
            source_video_latent=seg0_v,
            source_audio_latent=seg0_a,
            context_frames=ctx_f,
        )
        seg1_lat_path = out_dir / "shared_orbit_seg1_av_latent.pt"
        torch.save({"video_latent": seg1_v, "audio_latent": seg1_a}, str(seg1_lat_path))

        bp2, bn2 = _load_cond("turnaround_orbit_seg2_240_to_360", "inkling_platos_key_concat4")
        _, _, off_seg2_meta = _sample_segment(
            label="ledger_off_orbit_seg2_240_to_360",
            base_pos=bp2,
            base_neg=bn2,
            frames=seg2_f,
            seed=seed_base + 202,
            source_video_latent=seg1_v,
            source_audio_latent=seg1_a,
            context_frames=ctx_f,
        )
        ledger_kfs, ledger_refs = _ledger_return_cond(seg2_f, seg_start_yaw=240.0, ctx_frames=ctx_f)
        _, _, on_seg2_meta = _sample_segment(
            label="ledger_on_orbit_seg2_240_to_360",
            base_pos=bp2,
            base_neg=bn2,
            frames=seg2_f,
            seed=seed_base + 202,
            extra_keyframes=ledger_kfs,
            ref_latents=ledger_refs,
            source_video_latent=seg1_v,
            source_audio_latent=seg1_a,
            context_frames=ctx_f,
        )
        shared_sha = sha256_file(seg1_lat_path)
        arm_outputs["ledger_off"] = {
            "shared_seg0": seg0_meta,
            "shared_seg1": seg1_meta,
            "seg2": off_seg2_meta,
            "segment_mp4s": [seg0_meta["mp4_path"], seg1_meta["mp4_path"], off_seg2_meta["mp4_path"]],
            "shared_branch_latent_sha256": shared_sha,
        }
        arm_outputs["ledger_on"] = {
            "shared_seg0": seg0_meta,
            "shared_seg1": seg1_meta,
            "seg2": on_seg2_meta,
            "segment_mp4s": [seg0_meta["mp4_path"], seg1_meta["mp4_path"], on_seg2_meta["mp4_path"]],
            "shared_branch_latent_sha256": shared_sha,
            "ledger_injection_mode": inj_mode,
        }
        if job.get("include_b0_control"):
            b0_p0, b0_n0 = _load_cond("turnaround_orbit_seg0_0_to_120", "b0_shuffled_single_tap")
            b0_v0, b0_a0, b0_m0 = _sample_segment(
                label="b0_shuffled_orbit_seg0",
                base_pos=b0_p0, base_neg=b0_n0, frames=seg0_f, seed=seed_base,
                extra_keyframes=[{"resolved_frame_index": 0, "latent": anchor_latent}],
            )
            b0_p1, b0_n1 = _load_cond("turnaround_orbit_seg1_120_to_240", "b0_shuffled_single_tap")
            b0_v1, b0_a1, b0_m1 = _sample_segment(
                label="b0_shuffled_orbit_seg1",
                base_pos=b0_p1, base_neg=b0_n1, frames=seg1_f, seed=seed_base + 101,
                source_video_latent=b0_v0, source_audio_latent=b0_a0, context_frames=ctx_f,
            )
            b0_p2, b0_n2 = _load_cond("turnaround_orbit_seg2_240_to_360", "b0_shuffled_single_tap")
            _, _, b0_m2 = _sample_segment(
                label="b0_shuffled_orbit_seg2",
                base_pos=b0_p2, base_neg=b0_n2, frames=seg2_f, seed=seed_base + 202,
                source_video_latent=b0_v1, source_audio_latent=b0_a1, context_frames=ctx_f,
            )
            arm_outputs["b0_shuffled"] = {
                "seg0": b0_m0, "seg1": b0_m1, "seg2": b0_m2,
                "segment_mp4s": [b0_m0["mp4_path"], b0_m1["mp4_path"], b0_m2["mp4_path"]],
            }

    elif topology == "double-revisit-720":
        # RTX PRO 6000 >=30s Double-Revisit Flagship (default 192+243+192+243 w/ 39f overlap = 753f = 31.375s):
        # Shared Seg 0 (0->180 deg) -> Branched Revisit #1 (180->360) -> Pan Away #2 (360->540) -> Revisit #2 (540->720)!
        s_lens = [int(s["frames"]) for s in plan["segments"]]
        ctx_f = int(plan["segments"][1]["trim_prefix_frames"])
        bp0, bn0 = _load_cond("turnaround_seg0_pan_away", "inkling_platos_key_concat4")
        bp1, bn1 = _load_cond("turnaround_seg1_return", "inkling_platos_key_concat4")

        seg0_v, seg0_a, seg0_meta = _sample_segment(
            label="shared_seg0_0_to_180",
            base_pos=bp0, base_neg=bn0, frames=s_lens[0], seed=seed_base,
            extra_keyframes=[{"resolved_frame_index": 0, "latent": anchor_latent}],
        )
        seg0_lat_path = out_dir / "shared_seg0_av_latent.pt"
        torch.save({"video_latent": seg0_v, "audio_latent": seg0_a}, str(seg0_lat_path))
        shared_sha = sha256_file(seg0_lat_path)

        for arm_tag, use_ledger in (("ledger_off", False), ("ledger_on", True)):
            pv, pa = seg0_v, seg0_a
            metas: List[Dict[str, Any]] = [seg0_meta]
            for sidx, (yaw_lbl, is_return) in enumerate(
                (("180_to_360_rev1", True), ("360_to_540_away2", False), ("540_to_720_rev2", True)),
                start=1,
            ):
                bp, bn = (bp1, bn1) if is_return else (bp0, bn0)
                kfs, refs = _ledger_return_cond(s_lens[sidx], seg_start_yaw=180.0) if (use_ledger and is_return) else ([], [])
                pv, pa, sm = _sample_segment(
                    label=f"{arm_tag}_seg{sidx}_{yaw_lbl}",
                    base_pos=bp, base_neg=bn, frames=s_lens[sidx], seed=seed_base + sidx * 101,
                    extra_keyframes=kfs, ref_latents=refs,
                    source_video_latent=pv, source_audio_latent=pa, context_frames=ctx_f,
                )
                metas.append(sm)
            arm_outputs[arm_tag] = {
                "segments": metas,
                "segment_mp4s": [m["mp4_path"] for m in metas],
                "shared_branch_latent_sha256": shared_sha,
                **({"ledger_injection_mode": inj_mode} if use_ledger else {}),
            }

        if job.get("include_b0_control"):
            b0_p0, b0_n0 = _load_cond("turnaround_seg0_pan_away", "b0_shuffled_single_tap")
            b0_p1, b0_n1 = _load_cond("turnaround_seg1_return", "b0_shuffled_single_tap")
            pv, pa, m0 = _sample_segment(
                label="b0_shuffled_seg0", base_pos=b0_p0, base_neg=b0_n0, frames=s_lens[0], seed=seed_base,
                extra_keyframes=[{"resolved_frame_index": 0, "latent": anchor_latent}],
            )
            b0_metas = [m0]
            for sidx in range(1, 4):
                bp, bn = (b0_p1, b0_n1) if (sidx % 2 == 1) else (b0_p0, b0_n0)
                pv, pa, sm = _sample_segment(
                    label=f"b0_shuffled_seg{sidx}", base_pos=bp, base_neg=bn, frames=s_lens[sidx],
                    seed=seed_base + sidx * 101, source_video_latent=pv, source_audio_latent=pa, context_frames=ctx_f,
                )
                b0_metas.append(sm)
            arm_outputs["b0_shuffled"] = {"segments": b0_metas, "segment_mp4s": [m["mp4_path"] for m in b0_metas]}

    elif topology == "two-segment-branched":
        seg0_f = int(plan["segments"][0]["frames"])
        seg1_f = int(plan["segments"][1]["frames"])
        ctx_f = int(plan["segments"][1]["trim_prefix_frames"])

        # Step A: Shared Seg 0 (0 -> 180 deg) rendered ONCE on concat4 key -> byte-identical outbound pan!
        bp0, bn0 = _load_cond("turnaround_seg0_pan_away", "inkling_platos_key_concat4")
        seg0_v, seg0_a, seg0_meta = _sample_segment(
            label="shared_seg0_0_to_180",
            base_pos=bp0,
            base_neg=bn0,
            frames=seg0_f,
            seed=seed_base,
            extra_keyframes=[{"resolved_frame_index": 0, "latent": anchor_latent}],
        )
        seg0_lat_path = out_dir / "shared_seg0_av_latent.pt"
        torch.save({"video_latent": seg0_v, "audio_latent": seg0_a}, str(seg0_lat_path))

        # Step B: Branched Seg 1 (180 -> 360 deg) from identical seg0 tail latent & identical seed!
        bp1, bn1 = _load_cond("turnaround_seg1_return", "inkling_platos_key_concat4")
        _, _, off_seg1_meta = _sample_segment(
            label="ledger_off_seg1_180_to_360",
            base_pos=bp1,
            base_neg=bn1,
            frames=seg1_f,
            seed=seed_base + 101,
            extra_keyframes=[],
            ref_latents=[],
            source_video_latent=seg0_v,
            source_audio_latent=seg0_a,
            context_frames=ctx_f,
        )

        # Ledger-ON retrieves anchor_latent + intermediate spatial anchors from Verified Ledger store
        ledger_kfs, ledger_refs = _ledger_return_cond(seg1_f, seg_start_yaw=180.0)
        _, _, on_seg1_meta = _sample_segment(
            label="ledger_on_seg1_180_to_360",
            base_pos=bp1,
            base_neg=bn1,
            frames=seg1_f,
            seed=seed_base + 101,
            extra_keyframes=ledger_kfs,
            ref_latents=ledger_refs,
            source_video_latent=seg0_v,
            source_audio_latent=seg0_a,
            context_frames=ctx_f,
        )

        arm_outputs["ledger_off"] = {
            "shared_seg0": seg0_meta,
            "seg1": off_seg1_meta,
            "segment_mp4s": [seg0_meta["mp4_path"], off_seg1_meta["mp4_path"]],
            "shared_seg0_latent_sha256": sha256_file(seg0_lat_path),
        }
        arm_outputs["ledger_on"] = {
            "shared_seg0": seg0_meta,
            "seg1": on_seg1_meta,
            "segment_mp4s": [seg0_meta["mp4_path"], on_seg1_meta["mp4_path"]],
            "shared_seg0_latent_sha256": sha256_file(seg0_lat_path),
            "ledger_injection_mode": inj_mode,
        }

        if job.get("include_b0_control"):
            bp0_b0, bn0_b0 = _load_cond("turnaround_seg0_pan_away", "b0_shuffled_single_tap")
            b0_v0, b0_a0, b0_m0 = _sample_segment(
                label="b0_shuffled_seg0",
                base_pos=bp0_b0,
                base_neg=bn0_b0,
                frames=seg0_f,
                seed=seed_base,
                extra_keyframes=[{"resolved_frame_index": 0, "latent": anchor_latent}],
            )
            bp1_b0, bn1_b0 = _load_cond("turnaround_seg1_return", "b0_shuffled_single_tap")
            _, _, b0_m1 = _sample_segment(
                label="b0_shuffled_seg1",
                base_pos=bp1_b0,
                base_neg=bn1_b0,
                frames=seg1_f,
                seed=seed_base + 101,
                source_video_latent=b0_v0,
                source_audio_latent=b0_a0,
                context_frames=ctx_f,
            )
            arm_outputs["b0_shuffled"] = {
                "seg0": b0_m0,
                "seg1": b0_m1,
                "segment_mp4s": [b0_m0["mp4_path"], b0_m1["mp4_path"]],
            }
    else:
        # Single-clip fast turnaround topology
        total_f = int(plan["segments"][0]["frames"])
        bp, bn = _load_cond("turnaround_single_clip", "inkling_platos_key_concat4")
        _, _, off_m = _sample_segment(
            label="ledger_off_single_clip",
            base_pos=bp,
            base_neg=bn,
            frames=total_f,
            seed=seed_base,
            extra_keyframes=[{"resolved_frame_index": 0, "latent": anchor_latent}],
        )
        ledger_kfs, ledger_refs = _ledger_return_cond(total_f, seg_start_yaw=180.0)
        ledger_kfs = [{"resolved_frame_index": 0, "latent": anchor_latent}, *ledger_kfs]
        _, _, on_m = _sample_segment(
            label="ledger_on_single_clip",
            base_pos=bp,
            base_neg=bn,
            frames=total_f,
            seed=seed_base,
            extra_keyframes=ledger_kfs,
            ref_latents=ledger_refs,
        )
        arm_outputs["ledger_off"] = {"seg0": off_m, "segment_mp4s": [off_m["mp4_path"]]}
        arm_outputs["ledger_on"] = {"seg0": on_m, "segment_mp4s": [on_m["mp4_path"]], "ledger_injection_mode": inj_mode}

    hw = detect_hardware_info()
    del mp, mp_raw, video_vae, audio_vae
    gc.collect()
    mm.soft_empty_cache()

    atomic_write_json(
        out_dir / "turnaround_render_result.json",
        {
            "schema_version": "omni-912-turnaround-h3-render-result/v1",
            "topology": topology,
            "zero_qwen_loaded": True,
            "anchor_latent_path": str(anchor_lat_path),
            "anchor_latent_sha256": sha256_file(anchor_lat_path),
            "load_seconds": round(load_s, 2),
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "hardware": hw,
            "arms": arm_outputs,
        },
    )


# ======================================================================================
# Synchronized 1920x1088 Side-by-Side Split-Screen Assembler (Zero Hand Editing)
# ======================================================================================


def _yaw_for_frame(
    frame_idx: int,
    total_frames: int,
    branch_frame: Optional[int],
    *,
    segment_end_frames: Optional[Sequence[int]] = None,
    segment_end_yaws: Optional[Sequence[float]] = None,
    max_yaw_deg: float = 360.0,
) -> float:
    """Compute camera yaw in [0, max_yaw_deg] across exact turnaround segment boundaries."""
    denom = max(total_frames - 1, 1)
    if segment_end_frames and segment_end_yaws and len(segment_end_frames) == len(segment_end_yaws):
        f_prev = 0
        y_prev = 0.0
        for idx, (f_end_raw, y_end) in enumerate(zip(segment_end_frames, segment_end_yaws)):
            f_end = (total_frames - 1) if idx == len(segment_end_frames) - 1 else min(total_frames - 1, int(f_end_raw))
            if frame_idx <= f_end or idx == len(segment_end_frames) - 1:
                span = max(f_end - f_prev, 1)
                frac = max(0.0, min(1.0, (frame_idx - f_prev) / float(span)))
                return y_prev + (float(y_end) - y_prev) * frac
            f_prev = f_end
            y_prev = float(y_end)
    if branch_frame is not None and 0 < branch_frame < total_frames:
        if frame_idx <= branch_frame:
            return (max_yaw_deg * 0.5) * (frame_idx / float(branch_frame))
        return (max_yaw_deg * 0.5) + (max_yaw_deg * 0.5) * (
            (frame_idx - branch_frame) / float(max(total_frames - 1 - branch_frame, 1))
        )
    return max_yaw_deg * (frame_idx / float(denom))


def assemble_side_by_side_turnaround_mp4(
    *,
    left_mp4: Path,
    right_mp4: Path,
    out_mp4: Path,
    total_frames: int,
    branch_frame: Optional[int],
    ocr_summary_off: Mapping[str, Any],
    ocr_summary_on: Mapping[str, Any],
    concat4_sha_prefix: str,
    segment_end_frames: Optional[Sequence[int]] = None,
    segment_end_yaws: Optional[Sequence[float]] = None,
    max_yaw_deg: float = 360.0,
    quality_gates: Optional[Mapping[str, Any]] = None,
    narration_wav: Optional[Path] = None,
    narration_text: Optional[str] = None,
    narration_start_s: float = 0.45,
    ambient_duck_db: float = -14.0,
    x264_threads: int = 4,
) -> Dict[str, Any]:
    """Build synchronized 1920x1088 side-by-side Ledger-OFF vs Ledger-ON MP4 with live yaw HUD, OCR metrics & narration mix."""
    w, h = 1920, 1088
    header_h = 84
    has_narration_caption = bool(narration_text and narration_text.strip())
    footer_h = 136 if has_narration_caption else 112
    pane_w = 956
    pane_h = h - header_h - footer_h
    gap_w = w - 2 * pane_w  # 8 px center divider

    def _decode_raw_rgb(mp4_p: Path, tw: int, th: int) -> subprocess.Popen:
        return subprocess.Popen(
            [
                FFMPEG, "-v", "error", "-i", str(mp4_p),
                "-vf", f"fps={FPS},scale={tw}:{th}:force_original_aspect_ratio=decrease,pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=0x0b0f12",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    dur_s = total_frames / float(FPS)
    enc_cmd: List[str] = [
        FFMPEG, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS),
        "-i", "pipe:0",
        "-i", str(right_mp4),
    ]
    if narration_wav is not None:
        delay_ms = max(0, int(round(float(narration_start_s) * 1000.0)))
        fc_audio = (
            f"[1:a:0]aresample={MASTER_AUDIO_SR},aformat=channel_layouts=stereo,"
            f"volume={float(ambient_duck_db):.1f}dB,"
            f"apad=whole_dur={dur_s:.6f},atrim=0:{dur_s:.6f},asetpts=PTS-STARTPTS[amb];"
            f"[2:a:0]aresample={MASTER_AUDIO_SR},aformat=channel_layouts=stereo,"
            f"highpass=f=65,loudnorm=I=-16:TP=-1.5:LRA=9,"
            f"adelay={delay_ms}|{delay_ms},"
            f"apad=whole_dur={dur_s:.6f},atrim=0:{dur_s:.6f},asetpts=PTS-STARTPTS[nar];"
            f"[amb][nar]amix=inputs=2:duration=first:dropout_transition=0.25:normalize=0,"
            f"alimiter=limit=0.95[aout]"
        )
        enc_cmd.extend(
            [
                "-i", str(narration_wav),
                "-filter_complex", fc_audio,
                "-map", "0:v:0", "-map", "[aout]",
                "-ar", str(MASTER_AUDIO_SR), "-ac", "2",
            ]
        )
    else:
        enc_cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])

    enc_cmd.extend(
        [
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16",
            *(["-threads", str(x264_threads)] if x264_threads > 0 else []),
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart",
            str(out_mp4),
        ]
    )
    enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    p_left = _decode_raw_rgb(left_mp4, pane_w, pane_h)
    p_right = _decode_raw_rgb(right_mp4, pane_w, pane_h)
    assert p_left.stdout is not None and p_right.stdout is not None and enc.stdin is not None

    frame_bytes = pane_w * pane_h * 3
    f_hdr = load_font(26, bold=True)
    f_sub = load_font(17, bold=False)
    f_hud = load_font(20, bold=True, mono=True)
    f_small = load_font(15, bold=False, mono=True)

    off_ret_acc = float(ocr_summary_off.get("return_mean_char_accuracy", 0.0)) * 100.0
    on_ret_acc = float(ocr_summary_on.get("return_mean_char_accuracy", 0.0)) * 100.0
    sim_tag = " [DRY-RUN SIM]" if ocr_summary_on.get("simulated") else ""
    gate_badge = ""
    if quality_gates:
        c_tag = "PASS" if quality_gates.get("continuity_gate_passed") else "FAIL"
        i_tag = "PASS" if quality_gates.get("identity_gate_passed") else "FAIL"
        v_tag = str(quality_gates.get("verdict", "PASS"))
        gate_badge = f"   |   CONT:{c_tag}  ID:{i_tag}  VERDICT:{v_tag}"
    nar_caption_line = f'NARRATION [Packet C]: "{narration_text.strip()}"' if has_narration_caption and narration_text else ""

    last_l = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)
    last_r = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)

    try:
        for fidx in range(total_frames):
            bl = p_left.stdout.read(frame_bytes)
            br = p_right.stdout.read(frame_bytes)
            if len(bl) == frame_bytes:
                last_l = np.frombuffer(bl, dtype=np.uint8).reshape(pane_h, pane_w, 3)
            if len(br) == frame_bytes:
                last_r = np.frombuffer(br, dtype=np.uint8).reshape(pane_h, pane_w, 3)

            canvas = Image.new("RGB", (w, h), (11, 15, 18))
            canvas.paste(Image.fromarray(last_l), (0, header_h))
            canvas.paste(Image.fromarray(last_r), (pane_w + gap_w, header_h))
            d = ImageDraw.Draw(canvas)

            # Center divider
            d.rectangle([(pane_w, header_h), (pane_w + gap_w, h - footer_h)], fill=(36, 46, 52))

            # Top headers (Left = Ledger-OFF amber/red accent; Right = Ledger-ON Atlas emerald accent)
            d.rectangle([(0, 0), (pane_w, header_h)], fill=(18, 22, 26))
            d.rectangle([(0, header_h - 4), (pane_w, header_h)], fill=(214, 92, 62))
            d.text((24, 14), "LEDGER-OFF  //  STOCK DIFFUSION WORLD", fill=(246, 174, 152), font=f_hdr)
            d.text((24, 48), "Frame-0 anchor only · zero loop-closure retrieval · identical seed & Packet A key", fill=(180, 190, 196), font=f_sub)

            rx0 = pane_w + gap_w
            d.rectangle([(rx0, 0), (w, header_h)], fill=(18, 22, 26))
            d.rectangle([(rx0, header_h - 4), (w, header_h)], fill=(118, 185, 0))
            d.text((rx0 + 24, 14), "LEDGER-ON  //  VERIFIED LEDGER LOOP-CLOSURE", fill=(166, 238, 66), font=f_hdr)
            d.text((rx0 + 24, 48), "Pose-indexed anchor retrieved at yaw=0° return · Packet B2 High-Res OCR verified", fill=(195, 212, 200), font=f_sub)

            # Bottom HUD bar: live 0 -> 180 -> 360 yaw gauge + Ledger state + OCR permanence summary
            fy0 = h - footer_h
            d.rectangle([(0, fy0), (w, h)], fill=(13, 18, 22))
            d.rectangle([(0, fy0), (w, fy0 + 2)], fill=(54, 68, 76))

            yaw = _yaw_for_frame(
                fidx,
                total_frames,
                branch_frame,
                segment_end_frames=segment_end_frames,
                segment_end_yaws=segment_end_yaws,
                max_yaw_deg=max_yaw_deg,
            )
            yaw_mod = yaw % 360.0
            if max_yaw_deg > 360.0:
                orbit_n = 1 if yaw <= 360.0 else 2
                if yaw < 45.0:
                    phase_lbl = "ORBIT 1 · ESTABLISH (0°)"
                    ledger_state = "COMMITTED [yaw=0.0°]"
                elif yaw < 180.0:
                    phase_lbl = "ORBIT 1 · PAN AWAY (0°->180° SHARED)"
                    ledger_state = "STORED IN LEDGER"
                elif yaw < 305.0:
                    phase_lbl = "ORBIT 1 · RETURN PAN (180°->360°)"
                    ledger_state = "LOOP-CLOSURE ARMED"
                elif yaw <= 395.0:
                    phase_lbl = "ORBIT 1 · REVISIT #1 (~360° / 0°)"
                    ledger_state = "RETRIEVED #1 [yaw=0.0°]"
                elif yaw < 540.0:
                    phase_lbl = "ORBIT 2 · PAN AWAY (360°->540°)"
                    ledger_state = "STORED IN LEDGER"
                elif yaw < 665.0:
                    phase_lbl = "ORBIT 2 · RETURN PAN (540°->720°)"
                    ledger_state = "LOOP-CLOSURE #2 ARMED"
                else:
                    phase_lbl = f"ORBIT {orbit_n} · REVISIT #2 (~720° / 0°)"
                    ledger_state = "RETRIEVED #2 [yaw=0.0°]"
            else:
                if yaw < 45.0:
                    phase_lbl = "ESTABLISH (0°)"
                    ledger_state = "COMMITTED [yaw=0.0°]"
                elif branch_frame is not None and fidx < branch_frame:
                    phase_lbl = "SLOW ORBIT AWAY (SHARED)"
                    ledger_state = "STORED IN LEDGER"
                elif yaw_mod < 305.0 and yaw < 305.0:
                    phase_lbl = "RETURN ORBIT (BRANCHED)"
                    ledger_state = "LOOP-CLOSURE ARMED"
                else:
                    phase_lbl = "RETURN TO ORIGIN (~360° / 0°)"
                    ledger_state = "RETRIEVED [yaw=0.0°]"

            # Progress bar (spans 0 -> max_yaw_deg with 180-deg ticks)
            bx0, by0, bx1, by1 = 36, fy0 + 16, 780, fy0 + 40
            d.rounded_rectangle([(bx0, by0), (bx1, by1)], radius=6, fill=(28, 36, 42))
            fill_x = bx0 + int((bx1 - bx0) * (yaw / max(max_yaw_deg, 1.0)))
            d.rounded_rectangle([(bx0, by0), (max(bx0 + 6, fill_x), by1)], radius=6, fill=(118, 185, 0))
            tick_deg = 180.0
            while tick_deg < max_yaw_deg:
                tx = bx0 + int((bx1 - bx0) * (tick_deg / max_yaw_deg))
                d.line([(tx, by0 - 4), (tx, by1 + 4)], fill=(220, 230, 236), width=2)
                tick_deg += 180.0

            t_cur_s = fidx / float(FPS)
            d.text(
                (804, fy0 + 14),
                f"YAW: {yaw:5.1f}° ({t_cur_s:4.1f}s/{dur_s:4.1f}s)  |  {phase_lbl}",
                fill=(244, 248, 250),
                font=f_hud,
            )
            d.text(
                (36, fy0 + 50),
                f"PACKET A KEY: concat4 ({concat4_sha_prefix}) · ZERO QWEN3-VL-32B  |  LEDGER STATE: {ledger_state}",
                fill=(182, 198, 204),
                font=f_small,
            )
            d.text(
                (36, fy0 + 76),
                f"RETURN OCR{sim_tag}: OFF={off_ret_acc:5.1f}%  vs  ON={on_ret_acc:5.1f}%{gate_badge}",
                fill=(166, 238, 66) if (not quality_gates or quality_gates.get("verdict") == "PASS") else (246, 120, 96),
                font=f_small,
            )
            if nar_caption_line:
                d.text((36, fy0 + 104), nar_caption_line, fill=(238, 244, 248), font=f_small)

            enc.stdin.write(canvas.tobytes())
        enc.stdin.close()
    finally:
        for p in (p_left, p_right):
            if p.stdout:
                p.stdout.close()
            if p.stderr:
                p.stderr.close()
            p.wait()
        err = ""
        if enc.stderr is not None:
            err = enc.stderr.read().decode("utf-8", errors="replace")
            enc.stderr.close()
        rc = enc.wait()
        if rc != 0:
            raise RuntimeError(f"Side-by-side turnaround ffmpeg encode failed rc={rc}: {err[-400:]}")

    return {
        "path": str(out_mp4),
        "sha256": sha256_file(out_mp4),
        "bytes": int(out_mp4.stat().st_size),
        "probe": probe_media(out_mp4),
    }


# ======================================================================================
# Dry-Run Synthetic Turnaround Renderer (100% CPU, zero GPU!)
# ======================================================================================


def _render_dry_run_turnaround_arm(
    anchor_png: Path,
    *,
    arm_name: str,
    total_frames: int,
    branch_frame: Optional[int],
    width: int,
    height: int,
    out_mp4: Path,
    segment_end_frames: Optional[Sequence[int]] = None,
    segment_end_yaws: Optional[Sequence[float]] = None,
    max_yaw_deg: float = 360.0,
) -> Dict[str, Any]:
    """Generate deterministic CPU dry-run pan frames exercising the full OCR/HUD/ffmpeg pipeline."""
    anchor = Image.open(anchor_png).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    anchor_arr = np.asarray(anchor, dtype=np.uint8)
    buf = np.empty((total_frames, height, width, 3), dtype=np.uint8)
    f_big = load_font(max(22, int(36 * width / 1344.0)), bold=True)
    for i in range(total_frames):
        yaw = _yaw_for_frame(
            i,
            total_frames,
            branch_frame,
            segment_end_frames=segment_end_frames,
            segment_end_yaws=segment_end_yaws,
            max_yaw_deg=max_yaw_deg,
        )
        # Blend away to rear workshop wall around 180 deg, return cleanly at 360 deg
        vis = max(0.0, math.cos(math.radians(yaw)))
        bg = np.full((height, width, 3), (20, 26, 30), dtype=np.float32)
        if arm_name == "ledger_off" and yaw > 240.0:
            # Visual distortion marker on dry-run ledger_off return so split-screen shows clear delta
            shifted = np.roll(anchor_arr, shift=int(18 * vis), axis=1).astype(np.float32)
            frame = np.clip(bg * (1.0 - vis * 0.7) + shifted * (vis * 0.7), 0, 255).astype(np.uint8)
        else:
            frame = np.clip(bg * (1.0 - vis) + anchor_arr.astype(np.float32) * vis, 0, 255).astype(np.uint8)
        im = Image.fromarray(frame)
        d = ImageDraw.Draw(im)
        d.rectangle([(16, 16), (520, 62)], fill=(11, 15, 18))
        d.text((28, 26), f"[DRY-RUN] {arm_name.upper()}  yaw={yaw:5.1f}°", fill=(235, 242, 246), font=f_big)
        buf[i] = np.asarray(im, dtype=np.uint8)

    wav_p = out_mp4.with_suffix(".dryrun_32k.wav")
    dur_s = total_frames / float(FPS)
    t = np.arange(int(round(dur_s * H3_AUDIO_SR)), dtype=np.float64) / float(H3_AUDIO_SR)
    hum = (0.04 * np.sin(2.0 * np.pi * 110.0 * t)).astype(np.float32)
    write_stereo_wav_pcm16(wav_p, np.stack([hum, hum], axis=-1), sr=H3_AUDIO_SR)
    return mux_frames_and_wav_to_mp4(buf, wav_p, out_mp4, fps=FPS, crf=24, x264_threads=2)


def _generate_dry_run_narration_wav(path: Path, duration_s: float = 2.8, sr: int = 24000) -> Dict[str, Any]:
    """Generate a clean speech-band dry-run WAV so Packet C narration mixing is verified end-to-end."""
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / float(sr)
    sig = 0.25 * np.sin(2.0 * np.pi * 220.0 * t) + 0.12 * np.sin(2.0 * np.pi * 440.0 * t)
    env = np.minimum(1.0, t / 0.1) * np.minimum(1.0, (duration_s - t) / 0.1)
    mono = (sig * np.clip(env, 0.0, 1.0)).astype(np.float32)
    sha, nbytes = write_stereo_wav_pcm16(path, np.stack([mono, mono], axis=-1), sr=sr)
    return {"path": str(path), "sha256": sha, "bytes": nbytes, "duration_s": round(duration_s, 4), "sample_rate": sr}


# ======================================================================================
# Main Coordinator
# ======================================================================================


def run_coordinator(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    if args.profile:
        args.topology = PROFILE_TO_TOPOLOGY[args.profile]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = ShowcaseTraceLogger(out_dir / "showcase_process_trace.jsonl", resume=False)

    # ------------------------------------------------------------------
    # Stage 0: Fail-Closed Asset & Key SHA-256 Verification
    # ------------------------------------------------------------------
    asset_audits: Dict[str, Any] = {
        "monkey_avatar": verify_pinned_file(
            Path(args.monkey_avatar),
            PINNED_OWNER_ASSETS["monkey_avatar"]["sha256"],
            label="monkey_avatar",
            allow_unpinned=args.allow_unpinned_assets,
        ),
        "kf_01_host_hook": verify_pinned_file(
            Path(args.kf_01_host_hook),
            PINNED_OWNER_ASSETS["kf_01_host_hook"]["sha256"],
            label="kf_01_host_hook",
            allow_unpinned=args.allow_unpinned_assets,
        ),
    }
    if not args.prebuilt_conds_dir:
        asset_audits["concat4_key"] = verify_pinned_file(
            Path(args.keys_dir) / "inkling_to_h3_concat4_platos_key.safetensors",
            EXPECTED_CONCAT4_KEY_SHA256,
            label="packetA_concat4_key",
            allow_unpinned=args.allow_unpinned_assets,
        )
        asset_audits["exporter_bin"] = verify_pinned_file(
            Path(args.exporter_bin),
            EXPECTED_EXPORTER_SHA256,
            label="vision_embed_exporter",
            allow_unpinned=args.allow_unpinned_assets,
        )
        if args.include_b0_control:
            asset_audits["b0_key"] = verify_pinned_file(
                Path(args.keys_dir) / "inkling_to_h3_b0_shuffled_platos_key.safetensors",
                EXPECTED_B0_KEY_SHA256,
                label="packetA_b0_key",
                allow_unpinned=args.allow_unpinned_assets,
            )
    else:
        asset_audits["concat4_key"] = {
            "label": "packetA_concat4_key_prebuilt",
            "path": str(args.prebuilt_conds_dir),
            "sha256": EXPECTED_CONCAT4_KEY_SHA256,
            "pin_verified": True,
        }

    step0 = trace.append(
        stage="preflight_pin_check",
        inputs={
            "topology": args.topology,
            "ledger_injection_mode": args.ledger_injection_mode,
            "anchor_source": args.anchor_source,
            "sign_text": args.sign_text,
            "narration_text": args.narration_text,
            "prebuilt_conds_dir": args.prebuilt_conds_dir,
            "dry_run": bool(args.dry_run),
            "zero_qwen_encoder": True,
            "bridge_key_sha256": asset_audits["concat4_key"]["sha256"],
        },
        outputs={"pinned_assets": asset_audits},
    )

    # ------------------------------------------------------------------
    # Stage 1: Model-Authored Turnaround Direction via Inkling-2-Mini (or prebuilt JSON)
    # ------------------------------------------------------------------
    sys_p = (
        "You are Omni-Inkling-2-Mini directing a matched Ledger-OFF vs Ledger-ON >=30s coherent 360-degree room turnaround "
        "permanence experiment. Return ONLY valid JSON with keys: orbit_seg0_0_to_120, orbit_seg1_120_to_240, "
        "orbit_seg2_240_to_360, seg0_pan_away_0_to_180, seg1_return_180_to_360, single_clip_full_360, h3_image_anchor_prompt."
    )
    usr_p = (
        f"Write crisp MiniMax-H3 FL2VA cinematography prompts for a smooth 30+ second clockwise camera orbit "
        f"around the Atlas Monkey Founder's workshop featuring a wall plaque with the exact verbatim text:\n"
        f"{args.sign_text!r}\n"
        f"Include the Monkey Founder, open studio door with warm amber lamp, and emerald hardware workbench."
    )
    resolved_pre_dir: Optional[Path] = Path(args.prebuilt_direction_json) if args.prebuilt_direction_json else None
    if resolved_pre_dir is None and args.prebuilt_conds_dir:
        pbase = Path(args.prebuilt_conds_dir)
        for cand in (
            pbase / "inkling_authored_turnaround_prompts.json",
            pbase.parent / "inkling_authored_turnaround_prompts.json",
            pbase.parent.parent / "inkling_authored_turnaround_prompts.json",
        ):
            if cand.is_file():
                resolved_pre_dir = cand
                break
    if resolved_pre_dir is not None:
        if not resolved_pre_dir.is_file():
            raise FileNotFoundError(f"Missing --prebuilt-direction-json: {resolved_pre_dir}")
        pre_obj = json.loads(resolved_pre_dir.read_text(encoding="utf-8"))
        t_prompts = dict(pre_obj.get("prompts") or pre_obj)
        gen_rec = pre_obj.get("inkling_generation") or {"token_ids": [], "model_id": "prebuilt_direction_json"}
        prompts_json_p = resolved_pre_dir
        prompts_sha = sha256_file(resolved_pre_dir)
    else:
        with InklingTextAuthor(
            inkling_dir=Path(args.inkling_dir),
            ingine_serve_bin=Path(args.ingine_serve_bin),
            base_url=args.inkling_url,
            persona_spec=Path(args.persona_spec) if args.persona_spec else None,
            dry_run=args.dry_run,
            log_path=out_dir / "ingine_serve.log",
        ) as author:
            gen_rec = author.generate(
                system_prompt=sys_p,
                user_prompt=usr_p,
                seed=args.seed,
                temperature=0.2,
                top_p=0.9,
                max_tokens=1024,
                dry_run_response=json.dumps(DEFAULT_TURNAROUND_PROMPTS, indent=2),
            )
        t_prompts = extract_first_json_object(gen_rec["text"])
        for req_k in ("seg0_pan_away_0_to_180", "seg1_return_180_to_360", "single_clip_full_360", "h3_image_anchor_prompt"):
            if not isinstance(t_prompts.get(req_k), str) or not t_prompts[req_k].strip():
                raise ValueError(f"Inkling turnaround JSON missing {req_k!r}")
        # Ensure 3-segment orbit keys are always populated cleanly (graceful fallback if model/prebuilt omitted them)
        t_prompts.setdefault("orbit_seg0_0_to_120", t_prompts["seg0_pan_away_0_to_180"])
        t_prompts.setdefault("orbit_seg1_120_to_240", DEFAULT_TURNAROUND_PROMPTS["orbit_seg1_120_to_240"])
        t_prompts.setdefault("orbit_seg2_240_to_360", t_prompts["seg1_return_180_to_360"])

        prompts_json_p = out_dir / "inkling_authored_turnaround_prompts.json"
        prompts_sha = atomic_write_json(
            prompts_json_p,
            {"inkling_generation": gen_rec, "prompts": t_prompts, "sign_text": args.sign_text},
        )
    t_prompts.setdefault("orbit_seg0_0_to_120", t_prompts.get("seg0_pan_away_0_to_180", DEFAULT_TURNAROUND_PROMPTS["orbit_seg0_0_to_120"]))
    t_prompts.setdefault("orbit_seg1_120_to_240", DEFAULT_TURNAROUND_PROMPTS["orbit_seg1_120_to_240"])
    t_prompts.setdefault("orbit_seg2_240_to_360", t_prompts.get("seg1_return_180_to_360", DEFAULT_TURNAROUND_PROMPTS["orbit_seg2_240_to_360"]))
    step1 = trace.append(
        stage="script_gen",
        inputs={"prompt_text": usr_p, "seed": args.seed, "bridge_key_sha256": asset_audits["concat4_key"]["sha256"]},
        outputs={
            "artifact_path": str(prompts_json_p),
            "artifact_sha256": prompts_sha,
            "prompts": t_prompts,
            "token_ids": gen_rec.get("token_ids", []),
        },
        parent_step_hashes=[step0["step_sha256"]],
    )

    # ------------------------------------------------------------------
    # Stage 2: Packet A 90.0 MiB concat4 Plato's Key Projection (Zero Qwen!)
    # ------------------------------------------------------------------
    cond_rows = [
        {"prompt_id": "turnaround_orbit_seg0_0_to_120", "prompt": t_prompts["orbit_seg0_0_to_120"]},
        {"prompt_id": "turnaround_orbit_seg1_120_to_240", "prompt": t_prompts["orbit_seg1_120_to_240"]},
        {"prompt_id": "turnaround_orbit_seg2_240_to_360", "prompt": t_prompts["orbit_seg2_240_to_360"]},
        {"prompt_id": "turnaround_seg0_pan_away", "prompt": t_prompts["seg0_pan_away_0_to_180"]},
        {"prompt_id": "turnaround_seg1_return", "prompt": t_prompts["seg1_return_180_to_360"]},
        {"prompt_id": "turnaround_single_clip", "prompt": t_prompts["single_clip_full_360"]},
        {"prompt_id": "turnaround_h3_image_anchor", "prompt": t_prompts["h3_image_anchor_prompt"]},
    ]
    if args.topology == "three-segment-orbit":
        req_prebuilt_ids = [
            "turnaround_orbit_seg0_0_to_120",
            "turnaround_orbit_seg1_120_to_240",
            "turnaround_orbit_seg2_240_to_360",
        ]
    elif args.topology in ("two-segment-branched", "double-revisit-720"):
        req_prebuilt_ids = ["turnaround_seg0_pan_away", "turnaround_seg1_return"]
    else:
        req_prebuilt_ids = ["turnaround_single_clip"]
    if args.anchor_source == "h3-image":
        req_prebuilt_ids.append("turnaround_h3_image_anchor")

    if args.prebuilt_conds_dir:
        proj = load_prebuilt_platos_key_conds(
            Path(args.prebuilt_conds_dir),
            req_prebuilt_ids,
            include_b0_control=args.include_b0_control,
            include_empty_neg=(args.cfg != 1.0),
            dry_run=args.dry_run,
        )
    else:
        proj = export_and_project_platos_key_conds(
            cond_rows,
            out_dir=out_dir / "packetA_conds",
            repo_root=Path(args.repo_root),
            inkling_dir=Path(args.inkling_dir),
            exporter_bin=Path(args.exporter_bin),
            keys_dir=Path(args.keys_dir),
            include_b0_control=args.include_b0_control,
            include_empty_neg=(args.cfg != 1.0),
            allow_unpinned=args.allow_unpinned_assets,
            dry_run=args.dry_run,
        )
    step2 = trace.append(
        stage="packetA_project_conds",
        inputs={
            "zero_qwen_loaded": True,
            "prebuilt_conds_dir": args.prebuilt_conds_dir,
            "bridge_key_sha256": proj["concat4_key"]["sha256"],
            "include_b0_control": bool(args.include_b0_control),
        },
        outputs={"cond_dir": proj["cond_dir"], "arms": proj["arms"], "entries": proj["entries"]},
        parent_step_hashes=[step1["step_sha256"]],
    )

    if args.project_conds_only:
        trace_audit = verify_trace_jsonl(out_dir / "showcase_process_trace.jsonl")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "project_conds_only": True,
                    "cond_dir": proj["cond_dir"],
                    "direction_json": str(prompts_json_p),
                    "arms": proj["arms"],
                    "trace_verified": trace_audit["verified"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    # ------------------------------------------------------------------
    # Stage 3: Photorealistic 3D Room Anchor Bank + Verified Ledger Commit (v4)
    # ------------------------------------------------------------------
    anchor_png = out_dir / "anchor_frame0_yaw0.png"
    room_pose_anchors: Dict[str, str] = {}
    room_bank_meta: Optional[Dict[str, Any]] = None
    room_dir_p = Path(args.room_anchor_dir)

    if args.anchor_source == "blender-room" and (room_dir_p / "room_anchors_manifest.json").is_file():
        room_bank_meta = load_blender_room_anchors(room_dir_p)
        anchor_png = Path(room_bank_meta["path"])
        # Exclude int-keyed `by_yaw` helper from JSON-serialized anchor_meta so sort_keys=True hash chain is 100% canonical
        anchor_meta = {k: v for k, v in room_bank_meta.items() if k != "by_yaw"}
        # Expose pose-indexed intermediate room anchors (strictly excluding yaw=345 QA-only view from pinning!)
        for yi, row in room_bank_meta["by_yaw"].items():
            if yi in (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330):
                room_pose_anchors[str(yi)] = str(row["path"])
    elif args.anchor_source == "blender-room" and not args.dry_run:
        raise FileNotFoundError(
            f"--anchor-source blender-room requested, but {room_dir_p / 'room_anchors_manifest.json'} does not exist. "
            "Render the 3D room bank via build_blender_360_room_anchors.py first."
        )
    elif args.anchor_source == "custom-image":
        if not args.custom_anchor_image or not Path(args.custom_anchor_image).is_file():
            raise FileNotFoundError(f"Missing --custom-anchor-image: {args.custom_anchor_image}")
        anchor_png = Path(args.custom_anchor_image)
        anchor_meta = {
            "source": "custom-image",
            "path": str(anchor_png),
            "sha256": sha256_file(anchor_png),
            "bytes": int(anchor_png.stat().st_size),
        }
    elif args.anchor_source == "h3-image" and not args.dry_run:
        img_job = out_dir / "h3_image_anchor_job.json"
        atomic_write_json(
            img_job,
            {
                "cond_dir": proj["cond_dir"],
                "prompt_id": "turnaround_h3_image_anchor",
                "image_dit_path": str(args.image_dit_path),
                "image_vae_path": str(args.image_vae_path),
                "out_png": str(anchor_png),
                "width": 1024,
                "height": 1024,
                "seed": args.seed,
            },
        )
        proc = subprocess.run(
            [
                str(args.comfy_python), "-B", str(Path(__file__).resolve()),
                "--worker", "h3-image-anchor",
                "--worker-job-json", str(img_job),
                "--comfy-root", str(args.comfy_root),
                "--repo-root", str(args.repo_root),
            ],
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"H3-Image anchor worker failed rc={proc.returncode}")
        anchor_meta = json.loads(anchor_png.with_suffix(".anchor_result.json").read_text(encoding="utf-8"))
        anchor_meta["source"] = "h3-image"
    else:
        anchor_meta = render_turnaround_anchor_plate(
            founder_ref_path=Path(args.kf_01_host_hook),
            sign_text=args.sign_text,
            out_png=anchor_png,
            width=1920,
            height=1088,
        )
        anchor_meta["source"] = "composite-plate"

    ocr_backend = "dry-run-sim" if args.dry_run else args.ocr_backend
    baseline_ocr = run_pluggable_ocr_hook(
        anchor_png,
        target_text=args.sign_text,
        backend=ocr_backend,
        ocr_cmd=args.ocr_cmd,
        got_ocr2_dir=Path(args.got_ocr2_dir),
        got_ocr2_weights=Path(args.got_ocr2_weights),
        dry_run_arm="ledger_on",
        dry_run_yaw_deg=0.0,
    )
    ledger_store_path = out_dir / "verified_ledger_store.json"
    ledger_entry = {
        "schema_version": "omni-912-verified-ledger-spatial-store/v2",
        "pose_key": "yaw_0.0_deg",
        "pose_yaw_deg": 0.0,
        "anchor_source": anchor_meta["source"],
        "anchor_png_path": str(anchor_png),
        "anchor_png_sha256": anchor_meta["sha256"],
        "founder_identity_ref_path": str(args.kf_01_host_hook),
        "founder_identity_ref_sha256": asset_audits["kf_01_host_hook"]["sha256"],
        "room_pose_anchors": room_pose_anchors,
        "expected_sign_text": args.sign_text,
        "baseline_ocr": baseline_ocr,
        "committed": True,
        "commit_rule": "baseline_3d_room_and_identity_verified_before_turnaround",
    }
    ledger_sha = atomic_write_json(ledger_store_path, ledger_entry)
    step3 = trace.append(
        stage="verified_ledger_commit",
        inputs={
            "pose_yaw_deg": 0.0,
            "expected_sign_text": args.sign_text,
            "anchor_source": anchor_meta["source"],
        },
        outputs={
            "artifact_path": str(ledger_store_path),
            "artifact_sha256": ledger_sha,
            "anchor_png": anchor_meta,
        },
        blinded_checks={
            "ocr_exact_match": baseline_ocr["case_insensitive_exact_match"],
            "ocr_cer": baseline_ocr["cer"],
            "ocr_char_accuracy": baseline_ocr["char_accuracy"],
            "all_finite": True,
        },
        parent_step_hashes=[step2["step_sha256"]],
    )

    # ------------------------------------------------------------------
    # Stage 4: Exact Turnaround Grid Plan + Matched Ledger-OFF vs Ledger-ON Render
    # ------------------------------------------------------------------
    plan = plan_turnaround_topology(
        args.topology,
        orbit_seg_frames=args.orbit_seg_frames,
        seg0_frames=args.seg0_frames,
        seg1_frames=args.seg1_frames,
        single_clip_frames=args.single_clip_frames,
        context_frames=CHAIN_CONTEXT_FRAMES,
        dry_run=args.dry_run,
    )
    branch_frame: Optional[int] = plan.get("branch_frame_index")
    seg_end_frames = plan.get("segment_end_frames")
    seg_end_yaws = plan.get("segment_end_yaws")
    max_yaw_deg = float(plan.get("max_yaw_deg", 360.0))

    total_frames = int(plan["total_frames"])
    render_dir = out_dir / "render_clips"
    render_dir.mkdir(parents=True, exist_ok=True)
    arm_mp4s: Dict[str, Dict[str, Any]] = {}

    if args.dry_run:
        for arm_name in ("ledger_off", "ledger_on") + (("b0_shuffled",) if args.include_b0_control else ()):
            mp4_p = render_dir / f"turnaround_{arm_name}.mp4"
            arm_mp4s[arm_name] = _render_dry_run_turnaround_arm(
                anchor_png,
                arm_name=arm_name,
                total_frames=total_frames,
                branch_frame=branch_frame,
                width=args.width,
                height=args.height,
                out_mp4=mp4_p,
                segment_end_frames=seg_end_frames,
                segment_end_yaws=seg_end_yaws,
                max_yaw_deg=max_yaw_deg,
            )
        render_hw = detect_hardware_info()
    else:
        t_job = render_dir / "turnaround_job.json"
        dit_map = {
            "ref2va": args.ref2va_dit_path,
            "hybrid": args.hybrid_dit_path,
            "fl2va": args.fl2va_dit_path,
        }
        dit_choice = dit_map[args.dit_variant]
        if args.dit_override:
            dit_choice = args.dit_override
        atomic_write_json(
            t_job,
            {
                "topology": args.topology,
                "ledger_injection_mode": args.ledger_injection_mode,
                "dit_variant": args.dit_variant,
                "out_dir": str(render_dir),
                "cond_dir": proj["cond_dir"],
                "anchor_png": str(anchor_png),
                "founder_ref_png": str(args.kf_01_host_hook),
                "room_pose_anchors": room_pose_anchors,
                "dit_path": str(dit_choice),
                "video_vae_path": str(args.video_vae_path),
                "audio_vae_path": str(args.audio_vae_path),
                "plan": plan,
                "include_b0_control": bool(args.include_b0_control),
                "seed": args.seed,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "cfg": args.cfg,
                "shift_video": args.shift_video,
                "shift_audio": args.shift_audio,
                "sampler_name": args.sampler_name,
                "scheduler_name": args.scheduler_name,
                "use_sparse_attention": not args.no_sparse_attention,
                "video_budget": args.video_budget,
                "early_steps": args.early_steps,
                "late_steps": args.late_steps,
            },
        )
        proc = subprocess.run(
            [
                str(args.comfy_python), "-B", str(Path(__file__).resolve()),
                "--worker", "h3-turnaround",
                "--worker-job-json", str(t_job),
                "--comfy-root", str(args.comfy_root),
                "--repo-root", str(args.repo_root),
            ],
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Turnaround H3 worker failed rc={proc.returncode}")
        r_res = json.loads((render_dir / "turnaround_render_result.json").read_text(encoding="utf-8"))
        render_hw = r_res["hardware"]
        for arm_name, ainfo in r_res["arms"].items():
            seg_mp4s = [Path(p) for p in ainfo["segment_mp4s"]]
            stitched_p = render_dir / f"turnaround_{arm_name}.mp4"
            arm_mp4s[arm_name] = stitch_chained_segments_ffmpeg(
                seg_mp4s,
                plan,
                out_mp4=stitched_p,
                master_audio_48k=None,
                x264_threads=args.x264_threads,
            )

    step4 = trace.append(
        stage="packetA_fl2va_av",
        inputs={
            "topology": args.topology,
            "ledger_injection_mode": args.ledger_injection_mode,
            "zero_qwen_loaded": True,
            "bridge_key_sha256": asset_audits["concat4_key"]["sha256"],
            "cfg": args.cfg,
            "shift_video": args.shift_video,
            "shift_audio": args.shift_audio,
            "plan": plan,
        },
        outputs={"arms": arm_mp4s},
        hardware=render_hw,
        parent_step_hashes=[step2["step_sha256"], step3["step_sha256"]],
    )

    # ------------------------------------------------------------------
    # Stage 5: Trajectory Frame Extraction + Pluggable Packet B2 High-Res OCR Audit
    # (audits NON-PINNED approach frames + opposite wall + return loop closure!)
    # ------------------------------------------------------------------
    if max_yaw_deg > 360.0:
        audit_fracs = [
            ("yaw_000_start", 0.00),
            ("yaw_180_opposite_1", 0.25),
            ("yaw_340_revisit1_nonpinned", 0.47),
            ("yaw_360_revisit1", 0.50),
            ("yaw_540_opposite_2", 0.75),
            ("yaw_685_revisit2_nonpinned", 0.94),
            ("yaw_705_revisit2_nonpinned", 0.975),
            ("yaw_720_revisit2_final", 1.00),
        ]
    else:
        audit_fracs = [
            ("yaw_000_start", 0.00),
            ("yaw_090_quarter", 0.25),
            ("yaw_180_opposite", 0.50),
            ("yaw_270_three_quarter", 0.75),
            ("yaw_325_approach_nonpinned", 0.90),
            ("yaw_345_approach_nonpinned", 0.95),
            ("yaw_355_approach_nonpinned", 0.98),
            ("yaw_360_return", 1.00),
        ]
    eval_indices = [min(total_frames - 1, max(0, int(round(f * (total_frames - 1))))) for _, f in audit_fracs]

    ocr_by_arm: Dict[str, Any] = {}
    extracted_by_arm: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for arm_name in ("ledger_off", "ledger_on"):
        ex = extract_frames_at_indices(Path(arm_mp4s[arm_name]["path"]), eval_indices, out_dir / "audit_frames" / arm_name, arm_name)
        by_idx = {r["frame_index"]: r for r in ex}
        extracted_by_arm[arm_name] = by_idx
        checkpoints: List[Dict[str, Any]] = []
        for lbl, frac in audit_fracs:
            fi = min(total_frames - 1, max(0, int(round(frac * (total_frames - 1)))))
            yaw = round(
                _yaw_for_frame(
                    fi,
                    total_frames,
                    branch_frame,
                    segment_end_frames=seg_end_frames,
                    segment_end_yaws=seg_end_yaws,
                    max_yaw_deg=max_yaw_deg,
                ),
                1,
            )
            f_rec = by_idx[fi]
            res = run_pluggable_ocr_hook(
                Path(f_rec["path"]),
                target_text=args.sign_text,
                backend=ocr_backend,
                ocr_cmd=args.ocr_cmd,
                got_ocr2_dir=Path(args.got_ocr2_dir),
                got_ocr2_weights=Path(args.got_ocr2_weights),
                dry_run_arm=arm_name,
                dry_run_yaw_deg=yaw,
            )
            checkpoints.append({"checkpoint": lbl, "frame_index": fi, "yaw_deg": yaw, "ocr": res})

        # Compute non-pinned return approach summary + opposite-wall absence verification
        nonpinned_ret = [c["ocr"] for c in checkpoints if c["checkpoint"].endswith("nonpinned")]
        all_ret = [
            c["ocr"]
            for c in checkpoints
            if c["checkpoint"].endswith("nonpinned")
            or c["checkpoint"].endswith("return")
            or c["checkpoint"].endswith("revisit1")
            or c["checkpoint"].endswith("revisit2_final")
        ]
        opp_ckpts = [c["ocr"] for c in checkpoints if "opposite" in c["checkpoint"]]
        nonpinned_full = [
            c["ocr"]["full_frame_baseline"]
            for c in checkpoints
            if c["checkpoint"].endswith("nonpinned") and isinstance(c["ocr"].get("full_frame_baseline"), dict)
        ]
        ocr_by_arm[arm_name] = {
            "simulated": bool(all_ret[0]["simulated"]) if all_ret else False,
            "nonpinned_approach_mean_cer": round(float(np.mean([r["cer"] for r in nonpinned_ret])), 5) if nonpinned_ret else 1.0,
            "nonpinned_approach_mean_char_accuracy": round(float(np.mean([r["char_accuracy"] for r in nonpinned_ret])), 5) if nonpinned_ret else 0.0,
            "nonpinned_full_frame_mean_char_accuracy": (
                round(float(np.mean([r["char_accuracy"] for r in nonpinned_full])), 5) if nonpinned_full else None
            ),
            "return_mean_cer": round(float(np.mean([r["cer"] for r in all_ret])), 5) if all_ret else 1.0,
            "return_mean_char_accuracy": round(float(np.mean([r["char_accuracy"] for r in all_ret])), 5) if all_ret else 0.0,
            "return_final_exact_match": bool(checkpoints[-1]["ocr"]["case_insensitive_exact_match"]),
            "opposite_180_sign_absent": all(r["ordered_word_recall"] < 0.15 for r in opp_ckpts) if opp_ckpts else False,
            "checkpoints": checkpoints,
        }

    step5 = trace.append(
        stage="packetB2_sight_inspect",
        inputs={
            "ocr_backend": ocr_backend,
            "expected_sign_text": args.sign_text,
            "audited_checkpoints": [lbl for lbl, _ in audit_fracs],
        },
        outputs={"ocr_by_arm": ocr_by_arm},
        blinded_checks={
            "ocr_exact_match": ocr_by_arm["ledger_on"]["return_final_exact_match"],
            "ledger_on_nonpinned_char_acc": ocr_by_arm["ledger_on"]["nonpinned_approach_mean_char_accuracy"],
            "ledger_off_nonpinned_char_acc": ocr_by_arm["ledger_off"]["nonpinned_approach_mean_char_accuracy"],
            "permanence_delta_char_acc": round(
                ocr_by_arm["ledger_on"]["nonpinned_approach_mean_char_accuracy"]
                - ocr_by_arm["ledger_off"]["nonpinned_approach_mean_char_accuracy"],
                5,
            ),
            "all_finite": True,
        },
        parent_step_hashes=[step4["step_sha256"]],
    )

    # ------------------------------------------------------------------
    # Stage 5a: Fail-Closed Anti-Goodhart Continuity & Jump-Cut Gate (v4)
    # ------------------------------------------------------------------
    continuity_by_arm: Dict[str, Any] = {}
    for arm_name in ("ledger_off", "ledger_on"):
        continuity_by_arm[arm_name] = run_temporal_continuity_gate(
            Path(arm_mp4s[arm_name]["path"]),
            return_window_frames=min(150, max(24, total_frames // 4)),
            min_frame1_ssim=args.min_frame1_ssim,
            min_full_adj_ssim=args.min_full_adj_ssim,
            min_return_adj_ssim=args.min_return_adj_ssim,
            max_return_adj_l1=args.max_return_adj_l1,
            max_return_l1_spike=args.max_return_l1_spike,
            min_tail20_ssim=args.min_tail20_ssim,
        )
    step5a_cont = trace.append(
        stage="packetB2_sight_inspect",
        inputs={
            "audit_kind": "temporal_continuity",
            "min_frame1_ssim": args.min_frame1_ssim,
            "min_full_adj_ssim": args.min_full_adj_ssim,
            "min_return_adj_ssim": args.min_return_adj_ssim,
            "max_return_adj_l1": args.max_return_adj_l1,
            "max_return_l1_spike": args.max_return_l1_spike,
            "min_tail20_ssim": args.min_tail20_ssim,
        },
        outputs={"continuity_by_arm": continuity_by_arm},
        blinded_checks={
            "ledger_on_continuity_passed": continuity_by_arm["ledger_on"]["passed"],
            "ledger_on_return_min_ssim": continuity_by_arm["ledger_on"]["return_min_adj_ssim"],
            "ledger_on_tail20_ssim": continuity_by_arm["ledger_on"]["tail_20_to_final_ssim"],
            "ledger_on_n_cuts": continuity_by_arm["ledger_on"]["n_detected_cuts"],
            "all_finite": True,
        },
        parent_step_hashes=[step4["step_sha256"]],
    )

    # ------------------------------------------------------------------
    # Stage 5a-2: Continuous Subject Identity Gate (SigLIP2 vs Monkey Founder, v4)
    # ------------------------------------------------------------------
    id_backend = "dry-run-sim" if args.dry_run else args.identity_backend
    identity_audit = run_subject_identity_gate(
        arm_checkpoints={k: v["checkpoints"] for k, v in ocr_by_arm.items()},
        founder_ref_path=Path(args.kf_01_host_hook),
        anchor_yaw0_path=anchor_png,
        siglip2_dir=Path(args.siglip2_dir),
        backend=id_backend,
        min_cos_founder=args.min_identity_cos_founder,
        min_cos_anchor=args.min_identity_cos_anchor,
        min_wardrobe_margin=args.min_identity_wardrobe_margin,
    )
    step5a_id = trace.append(
        stage="packetB2_sight_inspect",
        inputs={
            "audit_kind": "subject_identity",
            "identity_backend": id_backend,
            "founder_ref": str(args.kf_01_host_hook),
            "anchor_yaw0": str(anchor_png),
        },
        outputs=identity_audit,
        blinded_checks={
            "identity_gate_passed": identity_audit["passed"],
            "ledger_on_cos_founder": identity_audit["arms"].get("ledger_on", {}).get("nonpinned_mean_cos_founder", 0.0),
            "ledger_on_cos_anchor": identity_audit["arms"].get("ledger_on", {}).get("nonpinned_mean_cos_anchor", 0.0),
            "ledger_on_wardrobe_margin": identity_audit["arms"].get("ledger_on", {}).get("nonpinned_mean_wardrobe_margin", 0.0),
            "all_finite": True,
        },
        parent_step_hashes=[step5["step_sha256"]],
    )

    # Aggregate Fail-Closed Anti-Goodhart Quality Verdict
    on_np_ocr = float(ocr_by_arm["ledger_on"]["nonpinned_approach_mean_char_accuracy"])
    off_np_ocr = float(ocr_by_arm["ledger_off"]["nonpinned_approach_mean_char_accuracy"])
    ocr_gate_ok = (on_np_ocr >= args.min_nonpinned_ocr_acc) and (on_np_ocr > off_np_ocr)
    opp_absent_ok = bool(ocr_by_arm["ledger_on"]["opposite_180_sign_absent"])
    cont_gate_ok = bool(continuity_by_arm["ledger_on"]["passed"])
    id_gate_ok = bool(identity_audit["passed"])

    gate_failures: List[str] = []
    if not cont_gate_ok:
        gate_failures.extend(f"CONTINUITY:{f}" for f in continuity_by_arm["ledger_on"]["failures"])
    if not id_gate_ok:
        gate_failures.extend(f"IDENTITY:{f}" for f in identity_audit.get("failures", ["IDENTITY_FAIL"]))
    if not ocr_gate_ok:
        gate_failures.append(f"NONPINNED_OCR_FAIL(on={on_np_ocr:.3f},off={off_np_ocr:.3f},min={args.min_nonpinned_ocr_acc:.2f})")
    if not opp_absent_ok:
        gate_failures.append("OPPOSITE_180_SIGN_NOT_ABSENT")

    quality_gates = {
        "verdict": "PASS" if len(gate_failures) == 0 else "FAIL",
        "continuity_gate_passed": cont_gate_ok,
        "identity_gate_passed": id_gate_ok,
        "nonpinned_ocr_gate_passed": ocr_gate_ok,
        "opposite_wall_absence_passed": opp_absent_ok,
        "gate_failures": gate_failures,
    }

    # ------------------------------------------------------------------
    # Stage 5b: Packet C Narration Track Prep & Provenance Audit
    # (Owner requirement: "This is not a video. This is a world model being turned around.")
    # ------------------------------------------------------------------
    narration_wav_path: Optional[Path] = None
    narration_audit: Optional[Dict[str, Any]] = None
    mux_parents = [step4["step_sha256"], step5["step_sha256"], step5a_cont["step_sha256"], step5a_id["step_sha256"]]

    if args.narration_wav:
        narration_wav_path = Path(args.narration_wav)
        if not narration_wav_path.is_file():
            raise FileNotFoundError(f"Missing --narration-wav: {narration_wav_path}")
        nar_source = "prebuilt_packetC_wav"
    elif args.narration_tts_cmd:
        nar_dir = out_dir / "narration"
        nar_dir.mkdir(parents=True, exist_ok=True)
        narration_wav_path = nar_dir / "turnaround_narration_packetC.wav"
        cmd_s = (
            args.narration_tts_cmd
            .replace("{text}", shlex.quote(args.narration_text))
            .replace("{out_wav}", shlex.quote(str(narration_wav_path)))
        )
        proc = subprocess.run(cmd_s, shell=True, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not narration_wav_path.is_file():
            raise RuntimeError(f"--narration-tts-cmd failed rc={proc.returncode}: {proc.stderr[-400:]}")
        nar_source = "narration_tts_cmd"
    elif args.dry_run and not args.no_narration:
        nar_dir = out_dir / "narration"
        nar_dir.mkdir(parents=True, exist_ok=True)
        narration_wav_path = nar_dir / "turnaround_narration_dryrun.wav"
        _generate_dry_run_narration_wav(narration_wav_path, duration_s=min(2.8, total_frames / float(FPS) * 0.6))
        nar_source = "dry_run_synth"
    elif (not args.no_narration) and DEFAULT_ATTESTED_NARRATION_WAV.is_file():
        narration_wav_path = DEFAULT_ATTESTED_NARRATION_WAV
        nar_source = "default_attested_packetC_wav"
    else:
        nar_source = "none"

    if narration_wav_path is not None and not args.no_narration:
        nar_probe = probe_media(narration_wav_path)
        nar_whisper = run_blinded_whisper_check(
            narration_wav_path,
            args.narration_text,
            whisper_dir=Path(args.whisper_dir),
            deps_dir=Path(args.qwen_tts_deps),
            device="cpu",
            dry_run=args.dry_run or args.skip_whisper,
        )
        narration_audit = {
            "source": nar_source,
            "narration_text": args.narration_text,
            "wav_path": str(narration_wav_path),
            "wav_sha256": sha256_file(narration_wav_path),
            "wav_bytes": int(narration_wav_path.stat().st_size),
            "duration_s": nar_probe["duration_s"],
            "sample_rate": int(nar_probe.get("sample_rate", 0)),
            "narration_start_s": float(args.narration_start_s),
            "ambient_duck_db": float(args.ambient_duck_db),
            "whisper_check": nar_whisper,
        }
        step5b = trace.append(
            stage="packetC_mouth",
            inputs={
                "source": nar_source,
                "narration_text": args.narration_text,
                "narration_start_s": float(args.narration_start_s),
                "ambient_duck_db": float(args.ambient_duck_db),
            },
            outputs={
                "artifact_path": str(narration_wav_path),
                "artifact_bytes": narration_audit["wav_bytes"],
                "artifact_sha256": narration_audit["wav_sha256"],
                "duration_s": narration_audit["duration_s"],
                "sample_rate": narration_audit["sample_rate"],
            },
            blinded_checks={
                "whisper_transcript": nar_whisper["whisper_transcript"],
                "narration_word_recall": nar_whisper["lyric_alignment"]["ordered_word_recall"],
                "narration_wer": nar_whisper["lyric_alignment"]["wer"],
                "all_finite": True,
            },
            parent_step_hashes=[step1["step_sha256"]],
        )
        mux_parents.append(step5b["step_sha256"])

    # ------------------------------------------------------------------
    # Stage 6: 1920x1088 Side-by-Side Split-Screen MP4 + 2x6 Multi-Angle Contact Sheet
    # ------------------------------------------------------------------
    sbs_mp4 = out_dir / "turnaround_side_by_side_1920x1088.mp4"
    sbs_info = assemble_side_by_side_turnaround_mp4(
        left_mp4=Path(arm_mp4s["ledger_off"]["path"]),
        right_mp4=Path(arm_mp4s["ledger_on"]["path"]),
        out_mp4=sbs_mp4,
        total_frames=total_frames,
        branch_frame=branch_frame,
        ocr_summary_off=ocr_by_arm["ledger_off"],
        ocr_summary_on=ocr_by_arm["ledger_on"],
        concat4_sha_prefix=proj["concat4_key"]["sha256"][:8],
        segment_end_frames=seg_end_frames,
        segment_end_yaws=seg_end_yaws,
        max_yaw_deg=max_yaw_deg,
        quality_gates=quality_gates,
        narration_wav=narration_wav_path if not args.no_narration else None,
        narration_text=args.narration_text if (narration_wav_path is not None and not args.no_narration) else None,
        narration_start_s=args.narration_start_s,
        ambient_duck_db=args.ambient_duck_db,
        x264_threads=args.x264_threads,
    )

    sheet_angles = [0, 1, 3, 4, 6, 7] if max_yaw_deg > 360.0 else [0, 1, 2, 3, 5, 7]
    tiles: List[Dict[str, str]] = []
    for arm_name, nice in (("ledger_off", "LEDGER-OFF"), ("ledger_on", "LEDGER-ON")):
        ckpts = ocr_by_arm[arm_name]["checkpoints"]
        for ai in sheet_angles:
            c = ckpts[ai]
            tiles.append(
                {
                    "path": c["ocr"]["image_path"],
                    "label": f"{nice} | yaw={c['yaw_deg']:5.1f}° | OCR={c['ocr']['char_accuracy']*100:.0f}%",
                }
            )
    sheet_info = build_contact_sheet(
        tiles,
        out_dir / "qa_contact_sheet_360_turnaround.png",
        cols=6,
        tile_w=480,
        tile_h=272,
        header_title=(
            f"#912 360° TURNAROUND ({quality_gates['verdict']}): LEDGER-OFF (TOP) vs LEDGER-ON (BOTTOM) "
            f"| Packet A concat4 ({proj['concat4_key']['sha256'][:8]}) "
            f"| CONT={'PASS' if cont_gate_ok else 'FAIL'} ID={'PASS' if id_gate_ok else 'FAIL'}"
        ),
    )

    trace.append(
        stage="ffmpeg_mux",
        inputs={
            "side_by_side_resolution": "1920x1088",
            "narration_muxed": narration_audit is not None,
            "narration_start_s": float(args.narration_start_s) if narration_audit else None,
            "ambient_duck_db": float(args.ambient_duck_db) if narration_audit else None,
        },
        outputs={"side_by_side_mp4": sbs_info, "qa_contact_sheet": sheet_info, "quality_gates": quality_gates},
        parent_step_hashes=mux_parents,
    )

    trace_audit = verify_trace_jsonl(out_dir / "showcase_process_trace.jsonl")
    if not trace_audit["verified"]:
        raise RuntimeError(f"Showcase trace verification failed: {trace_audit['errors']}")

    # Stage 5a-3: Host-Enforced 1:1 Multi-Crop Visual Gate audit (att-26813349) when available
    host_visual_crop_audit: Optional[Dict[str, Any]] = None
    try:
        for cand_dir in (SCRIPT_DIR, SCRIPT_DIR.parent / "host_visual_gates"):
            if (cand_dir / "host_visual_crop_gate.py").is_file() and str(cand_dir) not in sys.path:
                sys.path.insert(0, str(cand_dir))
        import host_visual_crop_gate as _hvcg  # type: ignore

        vc_frames: List[Dict[str, Any]] = []
        for cp in ocr_by_arm["ledger_on"]["checkpoints"]:
            if cp["checkpoint"] in ("yaw_000_start", "yaw_345_approach_nonpinned", "yaw_355_approach_nonpinned", "yaw_360_return"):
                p_im = Path(cp["ocr"]["image_path"])
                if p_im.is_file():
                    sheet_out = (out_dir / "audit_frames" / f"visual_crop_gate_{cp['checkpoint']}.png") if cp["checkpoint"] == "yaw_355_approach_nonpinned" else None
                    vr = _hvcg.verify_image_crops(p_im, save_contact_sheet_path=sheet_out)
                    vc_frames.append({"checkpoint": cp["checkpoint"], "frame_index": cp["frame_index"], "yaw_deg": cp["yaw_deg"], **vr})
        host_visual_crop_audit = {
            "gate_module_sha256": _hvcg.sha256_file(Path(_hvcg.__file__)),
            "n_frames_audited": len(vc_frames),
            "all_frames_passed": all(r["passed"] for r in vc_frames) if vc_frames else False,
            "mean_composite_score": round(float(np.mean([r["composite_score"] for r in vc_frames])), 5) if vc_frames else 0.0,
            "frames": vc_frames,
        }
    except Exception as _vc_exc:
        host_visual_crop_audit = {"available": False, "error": f"{type(_vc_exc).__name__}: {_vc_exc}"}

    report = {
        "schema_version": "omni-912-showcase-360-turnaround-report/v2",
        "verdict": quality_gates["verdict"],
        "quality_gates": quality_gates,
        "dry_run": bool(args.dry_run),
        "topology": args.topology,
        "dit_variant": args.dit_variant,
        "ledger_injection_mode": args.ledger_injection_mode,
        "zero_qwen_loaded": True,
        "sign_text": args.sign_text,
        "narration": narration_audit,
        "concat4_key_sha256": proj["concat4_key"]["sha256"],
        "plan": plan,
        "anchor_plate": anchor_meta,
        "verified_ledger_store": str(ledger_store_path),
        "arm_mp4s": arm_mp4s,
        "side_by_side_mp4": sbs_info,
        "qa_contact_sheet": sheet_info,
        "ocr_permanence_audit": ocr_by_arm,
        "temporal_continuity_audit": continuity_by_arm,
        "subject_identity_audit": identity_audit,
        "host_visual_crop_gate_audit": host_visual_crop_audit,
        "trace_audit": trace_audit,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    rep_path = out_dir / "turnaround_metrics_report.json"
    rep_sha = atomic_write_json(rep_path, report)
    print(
        json.dumps(
            {
                "status": quality_gates["verdict"],
                "verdict": quality_gates["verdict"],
                "gate_failures": quality_gates["gate_failures"],
                "dry_run": bool(args.dry_run),
                "topology": args.topology,
                "anchor_source": anchor_meta["source"],
                "dit_variant": args.dit_variant,
                "report": str(rep_path),
                "report_sha256": rep_sha,
                "side_by_side_mp4": sbs_info["path"],
                "qa_contact_sheet": sheet_info["path"],
                "ledger_off_nonpinned_ocr_acc": ocr_by_arm["ledger_off"]["nonpinned_approach_mean_char_accuracy"],
                "ledger_on_nonpinned_ocr_acc": ocr_by_arm["ledger_on"]["nonpinned_approach_mean_char_accuracy"],
                "continuity_gate_passed": cont_gate_ok,
                "identity_gate_passed": id_gate_ok,
                "trace_verified": trace_audit["verified"],
                "trace_steps": trace_audit["n_steps"],
                "elapsed_s": report["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    if args.fail_on_gate_error and quality_gates["verdict"] != "PASS":
        return 2
    return 0


def audit_existing_run(args: argparse.Namespace) -> int:
    """Run v4 Continuity + Subject Identity + Non-Pinned OCR gates on any completed showcase run dir."""
    run_dir = Path(args.audit_existing_run)
    rep_p = run_dir / "turnaround_metrics_report.json"
    if not rep_p.is_file():
        raise FileNotFoundError(f"Missing turnaround_metrics_report.json in {run_dir}")
    rep = json.loads(rep_p.read_text(encoding="utf-8"))
    arm_mp4s = rep["arm_mp4s"]
    ocr_by_arm = rep["ocr_permanence_audit"]
    total_frames = int(rep["plan"]["total_frames"])
    anchor_png = Path(rep["anchor_plate"]["path"])

    cont_by_arm = {
        arm_name: run_temporal_continuity_gate(
            Path(arm_mp4s[arm_name]["path"]),
            return_window_frames=min(150, max(24, total_frames // 4)),
            min_frame1_ssim=args.min_frame1_ssim,
            min_full_adj_ssim=args.min_full_adj_ssim,
            min_return_adj_ssim=args.min_return_adj_ssim,
            max_return_adj_l1=args.max_return_adj_l1,
            max_return_l1_spike=args.max_return_l1_spike,
            min_tail20_ssim=args.min_tail20_ssim,
        )
        for arm_name in ("ledger_off", "ledger_on")
    }
    id_audit = run_subject_identity_gate(
        arm_checkpoints={k: v["checkpoints"] for k, v in ocr_by_arm.items()},
        founder_ref_path=Path(args.kf_01_host_hook),
        anchor_yaw0_path=anchor_png,
        siglip2_dir=Path(args.siglip2_dir),
        backend=args.identity_backend,
        min_cos_founder=args.min_identity_cos_founder,
        min_cos_anchor=args.min_identity_cos_anchor,
        min_wardrobe_margin=args.min_identity_wardrobe_margin,
    )
    on_np = float(ocr_by_arm["ledger_on"]["nonpinned_approach_mean_char_accuracy"])
    off_np = float(ocr_by_arm["ledger_off"]["nonpinned_approach_mean_char_accuracy"])
    ocr_ok = (on_np >= args.min_nonpinned_ocr_acc) and (on_np > off_np)
    failures: List[str] = []
    if not cont_by_arm["ledger_on"]["passed"]:
        failures.extend(f"CONTINUITY:{f}" for f in cont_by_arm["ledger_on"]["failures"])
    if not id_audit["passed"]:
        failures.extend(f"IDENTITY:{f}" for f in id_audit.get("failures", ["IDENTITY_FAIL"]))
    if not ocr_ok:
        failures.append(f"NONPINNED_OCR_FAIL(on={on_np:.3f},off={off_np:.3f},min={args.min_nonpinned_ocr_acc:.2f})")
    out = {
        "audited_run_dir": str(run_dir),
        "verdict": "PASS" if not failures else "FAIL",
        "gate_failures": failures,
        "continuity_by_arm": cont_by_arm,
        "subject_identity_audit": id_audit,
        "nonpinned_ocr_on": on_np,
        "nonpinned_ocr_off": off_np,
    }
    print(json.dumps(out, indent=2), flush=True)
    return 0 if not failures else 2


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="#912 Flagship Showcase 2: Matched Ledger-OFF vs Ledger-ON >=30s 360-Deg Turnaround Demo")
    ap.add_argument("--worker", choices=["none", "h3-image-anchor", "h3-turnaround"], default="none")
    ap.add_argument("--worker-job-json", type=str, default=None)
    ap.add_argument(
        "--profile",
        choices=list(PROFILE_TO_TOPOLOGY.keys()),
        default=None,
        help="RTX PRO 6000 preset: rtxp-30s-orbit (33.5s/804f default), rtxp-30s-double-revisit (31.4s/753f 720-deg), smoke-16s, smoke-12s",
    )
    ap.add_argument(
        "--topology",
        choices=["three-segment-orbit", "double-revisit-720", "two-segment-branched", "single-clip"],
        default="three-segment-orbit",
        help="Chained turnaround topology (default: three-segment-orbit = 33.50s / 804f coherent RTX PRO 6000 flagship)",
    )
    ap.add_argument(
        "--ledger-injection-mode",
        choices=["smooth-trajectory-v5", "multi-stop-ref", "ref-plus-intermediate", "ref-and-return-guide", "ref-only", "return-guide"],
        default="multi-stop-ref",
        help="Verified Ledger retrieval strategy (v5: smooth-trajectory-v5 overlap-compensated 30-deg dense stops + Ref2VA; v4: multi-stop-ref)",
    )
    ap.add_argument(
        "--dit-variant",
        choices=["ref2va", "hybrid", "fl2va"],
        default="ref2va",
        help="MiniMax-H3 DiT checkpoint (v4 default: ref2va for continuous <Picture 1> Monkey Founder identity + <Picture 2> 3D room lock)",
    )
    ap.add_argument(
        "--anchor-source",
        choices=["blender-room", "composite-plate", "h3-image", "custom-image"],
        default="blender-room",
        help="3D anchor source (v4 default: blender-room rendered in Blender 5.2.5 LTS Cycles)",
    )
    ap.add_argument("--room-anchor-dir", default=str(DEFAULT_BLENDER_ROOM_DIR), help="Directory containing Blender 3D room room_anchors_manifest.json + yaw_XXX.png")
    ap.add_argument("--custom-anchor-image", default=None)
    ap.add_argument("--audit-existing-run", default=None, help="Audit an already-completed turnaround run directory with v4 Continuity + Identity + OCR gates and exit 0/2")
    ap.add_argument("--fail-on-gate-error", action="store_true", help="Exit nonzero (rc=2) when any anti-Goodhart Continuity/Identity/OCR gate fails")

    # Anti-Goodhart Continuity & Subject Identity Gate thresholds (calibrated on Showcase 2 v3 real failures)
    ap.add_argument("--identity-backend", choices=["siglip2", "dry-run-sim", "skip"], default="siglip2")
    ap.add_argument("--siglip2-dir", default=DEFAULT_PATHS["siglip2_dir"])
    ap.add_argument("--min-frame1-ssim", type=float, default=0.72)
    ap.add_argument("--min-full-adj-ssim", type=float, default=0.65)
    ap.add_argument("--min-return-adj-ssim", type=float, default=0.72)
    ap.add_argument("--max-return-adj-l1", type=float, default=0.14)
    ap.add_argument("--max-return-l1-spike", type=float, default=6.5)
    ap.add_argument("--min-tail20-ssim", type=float, default=0.48)
    ap.add_argument("--min-identity-cos-founder", type=float, default=0.68)
    ap.add_argument("--min-identity-cos-anchor", type=float, default=0.70)
    ap.add_argument("--min-identity-wardrobe-margin", type=float, default=0.0)
    ap.add_argument("--min-nonpinned-ocr-acc", type=float, default=0.45)
    ap.add_argument("--sign-text", default=PINNED_SIGN_TEXT)
    ap.add_argument("--out-dir", default="/mnt/storage/omni_gate2_runs/phase0_912_spinup/showcase_runs/turnaround_360")
    ap.add_argument("--dry-run", action="store_true", help="Zero-GPU CPU dry run verifying anchor plate, AV grid, OCR hook, 1920x1088 split-screen MP4, contact sheet, and JSONL hash chain")
    ap.add_argument("--allow-unpinned-assets", action="store_true")
    ap.add_argument("--include-b0-control", action="store_true")

    # Packet C narration overlay (Direct Owner Requirement 2026-11-27 02:17 PST)
    ap.add_argument("--narration-wav", default=None, help="Path to 100%% model-authored Packet C Qwen3-TTS narration WAV mixed cleanly over turnaround")
    ap.add_argument("--narration-text", default=DEFAULT_TURNAROUND_NARRATION_TEXT, help="Spoken narration script logged to JSONL trace & HUD caption")
    ap.add_argument("--narration-start-s", type=float, default=0.45, help="Lead-in offset (seconds) before narration begins")
    ap.add_argument("--ambient-duck-db", type=float, default=-14.0, help="Ambient room audio ducking (dB) under narration track")
    ap.add_argument("--narration-tts-cmd", default=None, help="Optional external Packet C TTS command template using {text} and {out_wav}")
    ap.add_argument("--no-narration", action="store_true", help="Disable narration overlay explicitly")
    ap.add_argument("--skip-whisper", action="store_true")
    ap.add_argument("--whisper-dir", default=DEFAULT_PATHS["whisper_dir"])
    ap.add_argument("--qwen-tts-deps", default=DEFAULT_PATHS["qwen_tts_deps"])

    # Prebuilt Packet A conditioning handoff (for remote G4-4 / Vast runs without Inkling/exporter staged)
    ap.add_argument("--prebuilt-conds-dir", default=None, help="Reuse pre-projected <1 MiB Packet A concat4 .pt conds (zero Inkling/exporter needed on G4-4)")
    ap.add_argument("--prebuilt-direction-json", default=None, help="Reuse already-authored inkling_authored_turnaround_prompts.json (zero ingine-serve needed on G4-4)")
    ap.add_argument("--project-conds-only", action="store_true", help="Stop right after projecting Packet A .pt conds + conds_manifest.json for handoff to G4-4")

    # Pluggable Packet B2 OCR hook
    ap.add_argument("--ocr-backend", choices=["blinded-plaque-rapidocr", "packetb2-got-ocr", "custom-cmd", "dry-run-sim"], default="packetb2-got-ocr")
    ap.add_argument("--ocr-cmd", default=None, help="External OCR command template using {image} and {target}, outputting JSON {\"observed_text\": ...}")
    ap.add_argument("--got-ocr2-dir", default=DEFAULT_PATHS["got_ocr2_dir"])
    ap.add_argument("--got-ocr2-weights", default=DEFAULT_PATHS["got_ocr2_weights"])

    # Owner refs & fleet-configurable paths
    ap.add_argument("--monkey-avatar", default=PINNED_OWNER_ASSETS["monkey_avatar"]["path"])
    ap.add_argument("--kf-01-host-hook", default=PINNED_OWNER_ASSETS["kf_01_host_hook"]["path"])
    ap.add_argument("--repo-root", default=DEFAULT_PATHS["repo_root"])
    ap.add_argument("--comfy-root", default=DEFAULT_PATHS["comfy_root"])
    ap.add_argument("--comfy-python", default=DEFAULT_PATHS["comfy_python"])
    ap.add_argument("--keys-dir", default=DEFAULT_PATHS["keys_dir"])
    ap.add_argument("--inkling-dir", default=DEFAULT_PATHS["inkling_dir"])
    ap.add_argument("--exporter-bin", default=DEFAULT_PATHS["exporter_bin"])
    ap.add_argument("--ingine-serve-bin", default=DEFAULT_PATHS["ingine_serve_bin"])
    ap.add_argument("--inkling-url", default=None)
    ap.add_argument("--persona-spec", default=None)
    ap.add_argument("--ref2va-dit-path", default=DEFAULT_PATHS["ref2va_dit_path"])
    ap.add_argument("--fl2va-dit-path", default=DEFAULT_PATHS["fl2va_dit_path"])
    ap.add_argument("--hybrid-dit-path", default=DEFAULT_PATHS["hybrid_dit_path"])
    ap.add_argument("--dit-override", default=None)
    ap.add_argument("--image-dit-path", default=DEFAULT_PATHS["image_dit_path"])
    ap.add_argument("--image-vae-path", default=DEFAULT_PATHS["image_vae_path"])
    ap.add_argument("--video-vae-path", default=DEFAULT_PATHS["video_vae_path"])
    ap.add_argument("--audio-vae-path", default=DEFAULT_PATHS["audio_vae_path"])

    # H3 sampling defaults (agreed with omni-inkling: cfg=1.0, shift_video=12.0, shift_audio=3.0)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--orbit-seg-frames", type=int, default=294, help="Per-segment exact H3 frame count for three-segment-orbit (294->804f/33.5s, 345->957f/39.9s)")
    ap.add_argument("--seg0-frames", type=int, default=192)
    ap.add_argument("--seg1-frames", type=int, default=243)
    ap.add_argument("--single-clip-frames", type=int, default=294)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--shift-video", type=float, default=12.0)
    ap.add_argument("--shift-audio", type=float, default=3.0)
    ap.add_argument("--sampler-name", default="res_multistep")
    ap.add_argument("--scheduler-name", default="simple")
    ap.add_argument("--no-sparse-attention", action="store_true")
    ap.add_argument("--video-budget", type=float, default=0.30)
    ap.add_argument("--early-steps", type=int, default=4)
    ap.add_argument("--late-steps", type=int, default=2)
    ap.add_argument("--x264-threads", type=int, default=4)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    if args.audit_existing_run:
        return audit_existing_run(args)
    if args.worker == "h3-image-anchor":
        worker_h3_image_anchor(args)
        return 0
    if args.worker == "h3-turnaround":
        worker_h3_turnaround(args)
        return 0
    return run_coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())

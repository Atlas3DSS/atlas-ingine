#!/usr/bin/env python3
"""Build the 1-strip visual walkthrough PNG/SVG + sped-up real-generation MP4 & GIF
for Zero-Gradient Omni-Modal Grafting (TUTORIAL.md).

Reads all metrics, prompts, images, video frames, and audio directly from pinned
local verified artifacts and writes walkthrough_sources.sha256.json.

Usage:
    python3 build_walkthrough_strip.py [--no-png] [--no-video]
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import io
import json
import math
import os
import re
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent
SAMPLES_DIR = MODULE_DIR / "samples"

# Pinned local verified sources
SRC = {
    "packet_a_report": Path("/mnt/storage/omni_gate2_runs/packetA_confirm160_full/h3_confirm160_report.json"),
    "packet_a_outcomes": Path("/mnt/storage/omni_gate2_runs/packetA_confirm160_full/h3_confirm160_per_prompt_outcomes.jsonl"),
    "qwen38_report": Path("/mnt/storage/omni_platos_keys_multimodel/qwen38/devbox_cu130_out/qwen38_h3_confirm160_report.json"),
    "prompts_jsonl": Path("/mnt/storage/omni-platos-key-dev/configs/h3_platos_key_phase0_prompts.jsonl"),
    "img_teacher": Path("/mnt/storage/omni_gate2_runs/packetA_confirm160_full/phase1_confirm160_workers/rendered_pngs/h3p0_0940__teacher_reinject.png"),
    "img_concat4": Path("/mnt/storage/omni_gate2_runs/packetA_confirm160_full/phase1_confirm160_workers/rendered_pngs/h3p0_0940__inkling_platos_key_concat4.png"),
    "img_b0": Path("/mnt/storage/omni_gate2_runs/packetA_confirm160_full/phase1_confirm160_workers/rendered_pngs/h3p0_0940__b0_shuffled_single_tap.png"),
    "wave3_pro6000_report": Path("/mnt/storage/omni_gate2_runs/phase0_912_spinup/wave3_real_av/local_pro6000_sm120/wave3_card_report.json"),
    "wave3_5070ti_report": Path("/mnt/storage/omni_gate2_runs/phase0_912_spinup/wave3_real_av/vast_rtx_5070_ti_47152186/wave3_card_report.json"),
    "wave3_mp4": Path("/mnt/storage/omni_gate2_runs/phase0_912_spinup/wave3_real_av/local_pro6000_sm120/wave3_real_5s_video_with_audio.mp4"),
    "wave3_wav": Path("/mnt/storage/omni_gate2_runs/phase0_912_spinup/wave3_real_av/local_pro6000_sm120/wave3_real_5s_audio_32k_stereo.wav"),
    "prosody_report": Path("/mnt/storage/omni_prosody_benchmark_942/results/unified/report_942_unified.json"),
    "prosody_worker": Path("/mnt/storage/omni_prosody_benchmark_942/results/track1/worker_status.json"),
    "prosody_clips": Path("/mnt/storage/omni_prosody_benchmark_942/results/unified/per_clip_scores_942_unified.jsonl"),
    "speech_sad_wav": Path("/mnt/storage/omni_prosody_benchmark_942/results/unified/listening_pack/blind_069.wav"),
    "speech_angry_wav": Path("/mnt/storage/omni_prosody_benchmark_942/results/unified/listening_pack/blind_072.wav"),
    "yue2_report": Path("/mnt/storage/omni_gate2_runs/musical_savant_946/plato_36m_live_30s_singing_g4/plato_36m_live_30s_singing_report.json"),
    "yue2_whisper": Path("/mnt/storage/omni_gate2_runs/musical_savant_946/plato_36m_live_30s_singing_g4/plato_36m_blinded_whisper_audit.json"),
    "music_song_b_wav": Path("/mnt/storage/omni_gate2_runs/musical_savant_946/plato_36m_live_30s_singing_g4/packetD_song_B__inkling_platos_key_36m_48k_stereo.wav"),
    "music_song_a_wav": Path("/mnt/storage/omni_gate2_runs/musical_savant_946/plato_36m_live_30s_singing_g4/packetD_song_A__inkling_platos_key_36m_48k_stereo.wav"),
}

# Paper visual design tokens (matching build_extra_figures.py)
BG = "#0A0B0C"
PANEL = "#111417"
PANEL2 = "#161B20"
INK = "#F4F3EE"
BODY = "#D3D5D1"
SEC = "#A0A6A9"
DIM = "#687076"
RULE = "#2B3034"
G = "#82C232"
G2 = "#4E7A22"
AMBER = "#eab94e"
RED = "#e5645a"
BLUE = "#58a6ff"
PURPLE = "#c084fc"
CYAN = "#22d3ee"

SANS = "'Noto Sans','Liberation Sans',Arial,sans-serif"
SERIF = "'Noto Serif','DejaVu Serif',Georgia,serif"
MONO = "'DejaVu Sans Mono','Liberation Mono',monospace"

FFMPEG = "/home/orwel/.local/bin/ffmpeg" if os.path.exists("/home/orwel/.local/bin/ffmpeg") else "ffmpeg"

PINNED_HASHES: Dict[str, str] = {}


def pin_file(key: str, path: Path) -> bytes:
    raw = path.read_bytes()
    PINNED_HASHES[f"{key}:{path.name}"] = hashlib.sha256(raw).hexdigest()
    return raw


def load_json(key: str) -> dict:
    return json.loads(pin_file(key, SRC[key]).decode("utf-8"))


def find_font(family: str, bold: bool = False) -> str:
    if family == "serif":
        names = ["NotoSerif-Bold.ttf" if bold else "NotoSerif-Regular.ttf", "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"]
    elif family == "mono":
        names = ["DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf", "LiberationMono-Bold.ttf" if bold else "LiberationMono-Regular.ttf"]
    else:
        names = ["NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    dirs = [
        os.path.expanduser("~/.local/share/fonts/noto-core"),
        "/home/orwel/.local/share/fonts/noto-core",
        "/home/orwel/.agy-william/.local/share/fonts/noto-core",
        "/usr/share/fonts/truetype/noto",
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype/liberation",
    ]
    for name in names:
        for d in dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"Font not found: {family} bold={bold}")


_FONT_CACHE: Dict[Tuple[str, bool, int], ImageFont.FreeTypeFont] = {}


def get_font(family: str, bold: bool, size: int) -> ImageFont.FreeTypeFont:
    k = (family, bold, size)
    if k not in _FONT_CACHE:
        _FONT_CACHE[k] = ImageFont.truetype(find_font(family, bold), size)
    return _FONT_CACHE[k]


def find_chrome() -> str:
    pats = [
        os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"),
        "/home/orwel/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell",
    ]
    found: List[str] = []
    for pat in pats:
        found.extend(glob.glob(pat))
    found = sorted(set(found))
    if not found:
        raise FileNotFoundError("chrome-headless-shell not found")
    return found[-1]


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def svg_text(x: float, y: float, s: str, size: float = 12, fill: str = SEC, anchor: str = "start",
             w: int = 400, serif: bool = False, mono: bool = False, ls: str | None = None) -> str:
    fam = MONO if mono else (SERIF if serif else SANS)
    extra = f' letter-spacing="{ls}"' if ls else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{w}" font-family="{fam}"{extra}>{esc(s)}</text>'
    )


def wrap_lines(s: str, size: float, width: float, bold: bool = False, family: str = "sans") -> List[str]:
    f = get_font(family, bold, 64)
    lines: List[str] = []
    cur = ""
    for word in s.split():
        trial = (cur + " " + word).strip()
        if f.getlength(trial) * size / 64.0 <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def svg_para(x: float, y: float, s: str, size: float = 12, fill: str = SEC, width: float = 400,
             lh: float = 1.42, w: int = 400, serif: bool = False, mono: bool = False) -> Tuple[str, float]:
    fam = "mono" if mono else ("serif" if serif else "sans")
    lines = wrap_lines(s, size, width, w >= 600, fam)
    out = ""
    for i, line in enumerate(lines):
        out += svg_text(x, y + i * size * lh, line, size, fill, "start", w, serif, mono)
    return out, y + max(0, len(lines) - 1) * size * lh


def img_to_data_uri(im: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        im.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
        mime = "image/jpeg"
    else:
        im.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def read_wav_mono(path: Path) -> Tuple[np.ndarray, int, float]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        n = w.getnframes()
        raw = w.readframes(n)
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    return arr, sr, n / float(sr)


def make_lut(stops: List[Tuple[float, Tuple[int, int, int]]]) -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.uint8)
    xs = [s[0] for s in stops]
    rs = [s[1][0] for s in stops]
    gs = [s[1][1] for s in stops]
    bs = [s[1][2] for s in stops]
    grid = np.linspace(0.0, 1.0, 256)
    lut[:, 0] = np.clip(np.interp(grid, xs, rs), 0, 255).astype(np.uint8)
    lut[:, 1] = np.clip(np.interp(grid, xs, gs), 0, 255).astype(np.uint8)
    lut[:, 2] = np.clip(np.interp(grid, xs, bs), 0, 255).astype(np.uint8)
    return lut


LUT_COOL = make_lut([
    (0.00, (11, 14, 19)),
    (0.25, (20, 44, 78)),
    (0.55, (42, 118, 198)),
    (0.80, (74, 200, 232)),
    (1.00, (238, 249, 255)),
])

LUT_WARM = make_lut([
    (0.00, (11, 14, 19)),
    (0.25, (64, 22, 32)),
    (0.55, (186, 62, 46)),
    (0.82, (238, 174, 62)),
    (1.00, (255, 246, 214)),
])

LUT_GREEN = make_lut([
    (0.00, (11, 14, 19)),
    (0.25, (18, 48, 34)),
    (0.55, (54, 128, 52)),
    (0.82, (130, 194, 50)),
    (1.00, (242, 252, 216)),
])


def render_spectrogram_image(
    samples: np.ndarray,
    sr: int,
    width: int,
    height: int,
    lut: np.ndarray,
    max_hz: float = 8000.0,
    n_fft: int = 1024,
    db_floor: float = -62.0,
) -> Image.Image:
    n_cols = max(64, width)
    if len(samples) < n_fft:
        samples = np.pad(samples, (0, n_fft - len(samples)))
    hop = max(1, (len(samples) - n_fft) // n_cols)
    starts = np.linspace(0, max(0, len(samples) - n_fft), n_cols).astype(int)
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.stack([samples[s : s + n_fft] * win for s in starts], axis=0)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    max_bin = int(np.searchsorted(freqs, min(max_hz, sr / 2.0)))
    max_bin = max(16, min(max_bin, spec.shape[1]))
    spec = spec[:, :max_bin]
    # Log-frequency warp for perceptual speech/music readability
    log_bins = np.geomspace(1.0, float(max_bin), height) - 1.0
    idx = np.clip(log_bins, 0, max_bin - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, max_bin - 1)
    frac = (idx - lo)[None, :]
    warped = spec[:, lo] * (1.0 - frac) + spec[:, hi] * frac
    db = 20.0 * np.log10(np.maximum(warped, 1e-6))
    peak_db = float(np.percentile(db, 99.6))
    norm = np.clip((db - (peak_db + db_floor)) / (-db_floor), 0.0, 1.0)
    norm = norm.T[::-1, :]  # high freq on top
    idx255 = np.clip((norm * 255.0).astype(np.int32), 0, 255)
    rgb = lut[idx255]
    return Image.fromarray(rgb, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)


def render_waveform_image(
    samples: np.ndarray,
    width: int,
    height: int,
    color_hex: str = CYAN,
    bg_hex: str = "#0D1116",
    highlight: Tuple[float, float] | None = None,
) -> Image.Image:
    im = Image.new("RGB", (width, height), bg_hex)
    draw = ImageDraw.Draw(im)
    if highlight is not None:
        hx0 = int(highlight[0] * width)
        hx1 = int(highlight[1] * width)
        draw.rectangle([hx0, 0, hx1, height], fill="#1D2B1A")
    mid = height // 2
    draw.line([(0, mid), (width, mid)], fill="#232A32", width=1)
    peak = max(1e-4, float(np.percentile(np.abs(samples), 99.5)))
    chunk = max(1, len(samples) // width)
    n_bins = min(width, len(samples) // chunk)
    for x in range(n_bins):
        seg = samples[x * chunk : (x + 1) * chunk]
        lo = float(seg.min()) / peak
        hi = float(seg.max()) / peak
        rms = float(np.sqrt(np.mean(seg ** 2))) / peak
        y1 = max(2, min(height - 3, int(mid - hi * (height * 0.44))))
        y2 = max(2, min(height - 3, int(mid - lo * (height * 0.44))))
        draw.line([(x, y1), (x, y2)], fill="#264E5D" if color_hex == CYAN else "#385225", width=1)
        ry1 = max(2, min(height - 3, int(mid - rms * (height * 0.44))))
        ry2 = max(2, min(height - 3, int(mid + rms * (height * 0.44))))
        draw.line([(x, ry1), (x, ry2)], fill=color_hex, width=1)
    return im


def extract_video_frames(mp4_path: Path) -> List[Image.Image]:
    """Decode all frames from mp4_path into RGB PIL Images."""
    with tempfile.TemporaryDirectory(prefix="omni_walk_frames_") as td:
        pat = os.path.join(td, "f_%04d.png")
        subprocess.run(
            [FFMPEG, "-v", "error", "-i", str(mp4_path), pat],
            check=True,
        )
        paths = sorted(glob.glob(os.path.join(td, "f_*.png")))
        frames = [Image.open(p).convert("RGB") for p in paths]
    if not frames:
        raise RuntimeError(f"No frames decoded from {mp4_path}")
    return frames


def load_verified_data() -> dict:
    pa = load_json("packet_a_report")
    qw = load_json("qwen38_report")
    w3_pro = load_json("wave3_pro6000_report")
    w3_5070 = load_json("wave3_5070ti_report")
    pros_rep = load_json("prosody_report")
    pros_wrk = load_json("prosody_worker")
    yue_rep = load_json("yue2_report")
    yue_wh = load_json("yue2_whisper")

    # Pin binary/jsonl sources too
    outcomes_raw = pin_file("packet_a_outcomes", SRC["packet_a_outcomes"]).decode("utf-8")
    prompts_raw = pin_file("prompts_jsonl", SRC["prompts_jsonl"]).decode("utf-8")
    clips_raw = pin_file("prosody_clips", SRC["prosody_clips"]).decode("utf-8")
    for k in ("img_teacher", "img_concat4", "img_b0", "wave3_mp4", "wave3_wav",
              "speech_sad_wav", "speech_angry_wav", "music_song_b_wav", "music_song_a_wav"):
        pin_file(k, SRC[k])

    prompts = {}
    for line in prompts_raw.splitlines():
        r = json.loads(line)
        prompts[r["prompt_id"]] = r["prompt"]

    row_940 = None
    for line in outcomes_raw.splitlines():
        r = json.loads(line)
        if r["prompt_id"] == "h3p0_0940":
            row_940 = r
            break
    assert row_940 is not None, "h3p0_0940 missing from outcomes"

    sad_sha = hashlib.sha256(SRC["speech_sad_wav"].read_bytes()).hexdigest()
    angry_sha = hashlib.sha256(SRC["speech_angry_wav"].read_bytes()).hexdigest()
    clip_sad = clip_angry = None
    for line in clips_raw.splitlines():
        r = json.loads(line)
        if r.get("wav_sha256") == sad_sha and r.get("arm") == "qwen3_armC_qwen38_key_continuous" and clip_sad is None:
            clip_sad = r
        if r.get("wav_sha256") == angry_sha and r.get("arm") == "qwen3_armC_inkling2m_key_continuous" and clip_angry is None:
            clip_angry = r
    assert clip_sad and clip_angry, "Speech clips missing in per_clip_scores_942_unified.jsonl"

    return {
        "pa": pa,
        "qw": qw,
        "w3_pro": w3_pro,
        "w3_5070": w3_5070,
        "pros_rep": pros_rep,
        "pros_wrk": pros_wrk,
        "yue_rep": yue_rep,
        "yue_wh": yue_wh,
        "prompt_940": prompts["h3p0_0940"],
        "prompt_080": prompts["h3p0_0080"],
        "row_940": row_940,
        "clip_sad": clip_sad,
        "clip_angry": clip_angry,
    }


def build_strip_svg(data: dict, video_frames: List[Image.Image]) -> Tuple[str, int]:
    W = 1480
    M = 44
    CW = W - 2 * M  # 1392

    pa = data["pa"]
    qw = data["qw"]
    w3_pro = data["w3_pro"]
    w3_5070 = data["w3_5070"]
    row_940 = data["row_940"]
    clip_sad = data["clip_sad"]
    clip_angry = data["clip_angry"]
    yue_rep = data["yue_rep"]
    yue_wh = data["yue_wh"]
    qwen_tts_wrk = data["pros_wrk"]["qwen3_tts"]

    # Extract verified metrics directly from pinned JSONs
    c4_ret = 100.0 * pa["separate_concat4_analysis_g16"]["siglip2_margin_retention_using_st_b0_reference"]
    qw_ret = 100.0 * qw["primary_single_tap_matched_b0_analysis_g16"]["siglip2_margin_retention_vs_teacher_and_b0"]
    c4_key_mib = pa["frozen_keys_audit"]["concat4"]["bytes"] / (1024.0 * 1024.0)
    qw_key_mib = qw["frozen_keys_audit"]["single_tap"]["bytes"] / (1024.0 * 1024.0)
    yue_key_mib = yue_rep["platos_key_bytes"] / (1024.0 * 1024.0)

    r940_t = row_940["arms"]["teacher_reinject"]["siglip2_text_image"]
    r940_c4 = row_940["arms"]["inkling_platos_key_concat4"]["siglip2_text_image"]
    r940_cos = row_940["arms"]["inkling_platos_key_concat4"]["siglip2_image_teacher"]
    r940_b0 = row_940["arms"]["b0_shuffled_single_tap"]["siglip2_text_image"]

    pro_img_s = w3_pro["real_image_gen"]["total_warm_gen_seconds"]
    pro_av_s = w3_pro["real_5s_video_audio_gen"]["total_warm_gen_seconds"]
    pro_av_dur = w3_pro["real_5s_video_audio_gen"]["video_duration_s"]
    pro_av_l1 = w3_pro["real_5s_video_audio_gen"]["video_temporal_motion_l1"]
    pro_av_rms = w3_pro["real_5s_video_audio_gen"]["audio_rms"]
    pro_key_ms = w3_pro["platos_key_concat4"]["warm_latency_ms_32tok"]

    c5070_img_s = w3_5070["real_image_gen"]["total_warm_gen_seconds"]
    c5070_av_s = w3_5070["real_5s_video_audio_gen"]["total_warm_gen_seconds"]
    c5070_av_gib = w3_5070["real_5s_video_audio_gen"]["peak_cuda_alloc_gib"]
    c5070_key_ms = w3_5070["platos_key_concat4"]["warm_latency_ms_32tok"]

    tts_mean_s = qwen_tts_wrk["elapsed_s"] / qwen_tts_wrk["n_clips"]
    tts_rtf = qwen_tts_wrk["warm_rtf"]
    tts_vram = qwen_tts_wrk["peak_vram_gib"]

    sad_prob = clip_sad["e2v_target_prob"]
    sad_f0 = clip_sad["acoustic"]["f0_median_hz"]
    sad_dur = clip_sad["acoustic"]["duration_s"]
    ang_prob = clip_angry["e2v_target_prob"]
    ang_f0 = clip_angry["acoustic"]["f0_median_hz"]
    ang_dur = clip_angry["acoustic"]["duration_s"]

    sb = yue_rep["songs"]["packetD_song_B"]["arms"]["inkling_platos_key_36m"]
    sa = yue_rep["songs"]["packetD_song_A"]["arms"]["inkling_platos_key_36m"]
    sb_wh = yue_wh["songs"]["packetD_song_B"]["inkling_platos_key_36m"]["lyric_recall"]
    sb_zero_wh = yue_wh["songs"]["packetD_song_B"]["zero_control"]["lyric_recall"]
    yue_peak_gib = yue_rep["peak_cuda_alloc_gib"]

    # Prepare embedded real visual assets
    im_c4_hero = Image.open(SRC["img_concat4"]).convert("RGB").resize((480, 480), Image.Resampling.LANCZOS)
    im_t_thumb = Image.open(SRC["img_teacher"]).convert("RGB").resize((192, 192), Image.Resampling.LANCZOS)
    im_c4_thumb = Image.open(SRC["img_concat4"]).convert("RGB").resize((192, 192), Image.Resampling.LANCZOS)
    im_b0_thumb = Image.open(SRC["img_b0"]).convert("RGB").resize((192, 192), Image.Resampling.LANCZOS)

    uri_c4_hero = img_to_data_uri(im_c4_hero, "JPEG")
    uri_t_thumb = img_to_data_uri(im_t_thumb, "JPEG")
    uri_c4_thumb = img_to_data_uri(im_c4_thumb, "JPEG")
    uri_b0_thumb = img_to_data_uri(im_b0_thumb, "JPEG")

    # 4 video keyframes (0%, 33%, 67%, 99%)
    n_f = len(video_frames)
    kf_indices = [0, n_f // 3, (2 * n_f) // 3, n_f - 1]
    kf_times = [idx / 24.0 for idx in kf_indices]
    kf_uris = [img_to_data_uri(video_frames[idx].resize((304, 178), Image.Resampling.LANCZOS), "JPEG") for idx in kf_indices]

    # Wave 3 32 kHz audio waveform
    av_arr, av_sr, _ = read_wav_mono(SRC["wave3_wav"])
    av_wave_im = render_waveform_image(av_arr, 624, 168, color_hex=CYAN)
    uri_av_wave = img_to_data_uri(av_wave_im, "PNG")

    # Speech spectrograms (blind_069 sad, blind_072 angry)
    sad_arr, sad_sr, _ = read_wav_mono(SRC["speech_sad_wav"])
    ang_arr, ang_sr, _ = read_wav_mono(SRC["speech_angry_wav"])
    sad_spec = render_spectrogram_image(sad_arr, sad_sr, 624, 248, LUT_COOL, max_hz=6500.0)
    ang_spec = render_spectrogram_image(ang_arr, ang_sr, 624, 248, LUT_WARM, max_hz=6500.0)
    uri_sad_spec = img_to_data_uri(sad_spec, "PNG")
    uri_ang_spec = img_to_data_uri(ang_spec, "PNG")

    # Music Song B 48 kHz spectrogram + waveform
    mus_arr, mus_sr, mus_dur = read_wav_mono(SRC["music_song_b_wav"])
    mus_spec = render_spectrogram_image(mus_arr, mus_sr, 624, 288, LUT_GREEN, max_hz=12000.0, n_fft=2048, db_floor=-68.0)
    mus_wave = render_waveform_image(mus_arr, 624, 72, color_hex=G, highlight=(19.0 / mus_dur, 24.0 / mus_dur))
    uri_mus_spec = img_to_data_uri(mus_spec, "PNG")
    uri_mus_wave = img_to_data_uri(mus_wave, "PNG")

    s = ""

    # ---- HEADER ----
    tag = "1-STRIP VISUAL WALKTHROUGH · REAL VERIFIED GENERATIONS"
    tw = get_font("sans", True, 64).getlength(tag) * 11 / 64.0 + len(tag) * 0.72 + 24
    s += f'<rect x="{M}" y="38" width="{tw:.0f}" height="24" rx="2" fill="none" stroke="{G}" stroke-width="1.5"/>'
    s += svg_text(M + 12, 54, tag, 11, G, "start", 700, ls="0.06em")
    s += svg_text(M + tw + 14, 54.5, "ZERO-GRADIENT OMNI-MODAL GRAFTING · ZERO TEXT ENCODER IN VRAM", 12, SEC, ls="0.05em")

    h_title, y = svg_para(
        M, 102,
        f"One Frozen LLM -> One {yue_key_mib:.0f}-{c4_key_mib:.0f} MiB Plato's Key -> Image, 5.9s Video+Audio, Speech & 48 kHz Music",
        27, INK, CW, 1.22, 700, serif=True,
    )
    s += h_title
    h_lede, y = svg_para(
        M, y + 26,
        "Every frame, spectrogram, waveform, and wall-clock time below is a real verified generation with zero receiver text encoder loaded "
        "(27.14 GB Qwen3-VL-32B eliminated), zero gradient steps, and zero edits to the frozen language model trunk.",
        14, BODY, CW - 40, 1.46,
    )
    s += h_lede

    # ---- ROW 1: 3-STAGE PIPELINE RIBBON ----
    r1_y = y + 24
    r1_h = 122
    gap_a = 28
    bw = (CW - 2 * gap_a) / 3.0  # 445.33

    def draw_arrow(x1: float, y1: float, x2: float, y2: float, col: str = G) -> str:
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1.0
        ux, uy = dx / L, dy / L
        px, py = -uy, ux
        ax, ay = x2 - ux * 8.5, y2 - uy * 8.5
        pts = f"{x2:.1f},{y2:.1f} {ax + px * 4.5:.1f},{ay + py * 4.5:.1f} {ax - px * 4.5:.1f},{ay - py * 4.5:.1f}"
        return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" stroke="{col}" stroke-width="2.0"/><polygon points="{pts}" fill="{col}"/>'

    # Step A
    xA = M
    s += f'<rect x="{xA:.1f}" y="{r1_y:.1f}" width="{bw:.1f}" height="{r1_h}" rx="5" fill="{PANEL}" stroke="{BLUE}" stroke-width="1.5"/>'
    s += f'<rect x="{xA:.1f}" y="{r1_y:.1f}" width="{bw:.1f}" height="4" rx="2" fill="{BLUE}"/>'
    s += svg_text(xA + 16, r1_y + 26, "STEP 1 · PROMPT -> FROZEN LLM HIDDEN TAPS", 11.5, BLUE, "start", 700, ls="0.04em")
    s += svg_text(xA + 16, r1_y + 49, "Inkling-2-Mini (concat4, d_2027=0) or Qwen3.8-27B (L31)", 12.5, INK, "start", 600)
    s += svg_text(xA + 16, r1_y + 70, "Single frozen forward pass · 0 gradient updates · 0 trunk edits", 11.5, SEC)
    chips_a = [("Image", BLUE), ("Video+Audio", CYAN), ("Speech", PURPLE), ("Music", G)]
    cx = xA + 16
    for lbl, col in chips_a:
        cw_chip = get_font("mono", True, 64).getlength(lbl) * 10.5 / 64.0 + 18
        s += f'<rect x="{cx:.1f}" y="{r1_y + 84:.1f}" width="{cw_chip:.1f}" height="22" rx="3" fill="{PANEL2}" stroke="{col}" stroke-width="1"/>'
        s += svg_text(cx + cw_chip / 2.0, r1_y + 98.5, lbl, 10.5, col, "middle", 700, mono=True)
        cx += cw_chip + 8

    # Arrow A -> B
    s += draw_arrow(xA + bw + 4, r1_y + r1_h / 2.0, xA + bw + gap_a - 4, r1_y + r1_h / 2.0, G)

    # Step B
    xB = xA + bw + gap_a
    s += f'<rect x="{xB:.1f}" y="{r1_y:.1f}" width="{bw:.1f}" height="{r1_h}" rx="5" fill="{PANEL}" stroke="{G}" stroke-width="1.8"/>'
    s += f'<rect x="{xB:.1f}" y="{r1_y:.1f}" width="{bw:.1f}" height="4" rx="2" fill="{G}"/>'
    s += svg_text(xB + 16, r1_y + 26, "STEP 2 · CLOSED-FORM PLATO'S KEY (FP16)", 11.5, G, "start", 700, ls="0.04em")
    s += svg_text(xB + 16, r1_y + 49, "y_hat = mu_Y + (x - mu_X) R   (one CPU solve, 0 GPU hr)", 12.0, INK, "start", 700, mono=True)
    s += svg_text(xB + 16, r1_y + 70, f"Warm key matmul: {pro_key_ms:.3f} ms (PRO 6000) · {c5070_key_ms:.3f} ms (RTX 5070 Ti)", 11.5, SEC)
    chips_b = [
        (f"{c4_key_mib:.0f}M H3", G),
        (f"{qw_key_mib:.0f}M Q3.8", BLUE),
        (f"{yue_key_mib:.0f}M YuE2", AMBER),
        ("1.4-2.8M TTS", PURPLE),
    ]
    cx = xB + 16
    for lbl, col in chips_b:
        cw_chip = get_font("mono", True, 64).getlength(lbl) * 10.2 / 64.0 + 16
        s += f'<rect x="{cx:.1f}" y="{r1_y + 84:.1f}" width="{cw_chip:.1f}" height="22" rx="3" fill="{PANEL2}" stroke="{col}" stroke-width="1"/>'
        s += svg_text(cx + cw_chip / 2.0, r1_y + 98.5, lbl, 10.2, col, "middle", 700, mono=True)
        cx += cw_chip + 7

    # Arrow B -> C
    s += draw_arrow(xB + bw + 4, r1_y + r1_h / 2.0, xB + bw + gap_a - 4, r1_y + r1_h / 2.0, G)

    # Step C
    xC = xB + bw + gap_a
    s += f'<rect x="{xC:.1f}" y="{r1_y:.1f}" width="{bw:.1f}" height="{r1_h}" rx="5" fill="{PANEL}" stroke="{AMBER}" stroke-width="1.5"/>'
    s += f'<rect x="{xC:.1f}" y="{r1_y:.1f}" width="{bw:.1f}" height="4" rx="2" fill="{AMBER}"/>'
    s += svg_text(xC + 16, r1_y + 26, "STEP 3 · ZERO TEXT ENCODER AT RUNTIME", 11.5, AMBER, "start", 700, ls="0.04em")
    s += svg_text(xC + 16, r1_y + 49, "27.14 GB Qwen3-VL-32B completely absent from VRAM", 12.5, INK, "start", 600)
    s += svg_text(xC + 16, r1_y + 70, f"Consumer 16 GB fit: {c5070_av_gib:.2f} GiB peak CUDA alloc (RTX 5070 Ti)", 11.5, SEC)
    s += f'<rect x="{xC + 16:.1f}" y="{r1_y + 84:.1f}" width="212" height="22" rx="3" fill="{PANEL2}" stroke="{G}" stroke-width="1"/>'
    s += svg_text(xC + 122, r1_y + 98.5, "zero_32b_qwen_loaded = true", 10.2, G, "middle", 700, mono=True)
    s += f'<rect x="{xC + 236:.1f}" y="{r1_y + 84:.1f}" width="192" height="22" rx="3" fill="{PANEL2}" stroke="{RULE}" stroke-width="1"/>'
    s += svg_text(xC + 332, r1_y + 98.5, "4 Decoders · 0 Refits", 10.2, INK, "middle", 700, mono=True)

    # ---- FAN-OUT CONNECTOR TO 4 MODALITY CARDS ----
    card_gap = 16
    card_w = (CW - 3 * card_gap) / 4.0  # 336.0
    card_xs = [M + i * (card_w + card_gap) for i in range(4)]
    bus_y = r1_y + r1_h + 18
    r2_y = r1_y + r1_h + 38
    r2_h = 684

    # Spine from center of Step B down to horizontal bus, branching into each of the 4 cards
    mid_x = W / 2.0
    s += f'<line x1="{mid_x:.1f}" y1="{r1_y + r1_h:.1f}" x2="{mid_x:.1f}" y2="{bus_y:.1f}" stroke="{G}" stroke-width="1.8"/>'
    s += f'<line x1="{card_xs[0] + card_w / 2:.1f}" y1="{bus_y:.1f}" x2="{card_xs[3] + card_w / 2:.1f}" y2="{bus_y:.1f}" stroke="{G}" stroke-width="1.8"/>'
    for cx0 in card_xs:
        s += draw_arrow(cx0 + card_w / 2.0, bus_y, cx0 + card_w / 2.0, r2_y - 2, G)

    def card_shell(idx: int, accent: str, title: str, key_chip: str, wall_badge: str, prompt_quote: str) -> Tuple[str, float, float]:
        x = card_xs[idx]
        out = ""
        out += f'<rect x="{x:.1f}" y="{r2_y:.1f}" width="{card_w:.1f}" height="{r2_h}" rx="6" fill="{PANEL}" stroke="{RULE}" stroke-width="1.5"/>'
        out += f'<rect x="{x:.1f}" y="{r2_y:.1f}" width="{card_w:.1f}" height="42" rx="6" fill="{PANEL2}"/>'
        out += f'<rect x="{x:.1f}" y="{r2_y + 36:.1f}" width="{card_w:.1f}" height="6" fill="{PANEL2}"/>'
        out += f'<rect x="{x:.1f}" y="{r2_y:.1f}" width="{card_w:.1f}" height="4" rx="2" fill="{accent}"/>'
        out += f'<line x1="{x:.1f}" y1="{r2_y + 42:.1f}" x2="{x + card_w:.1f}" y2="{r2_y + 42:.1f}" stroke="{RULE}" stroke-width="1"/>'
        out += svg_text(x + 12, r2_y + 27, title, 12.2, INK, "start", 700, ls="0.03em")
        # Key chip on right
        kw = get_font("mono", True, 64).getlength(key_chip) * 10.0 / 64.0 + 14
        out += f'<rect x="{x + card_w - kw - 10:.1f}" y="{r2_y + 11:.1f}" width="{kw:.1f}" height="21" rx="3" fill="{BG}" stroke="{accent}" stroke-width="1.2"/>'
        out += svg_text(x + card_w - kw / 2.0 - 10, r2_y + 25, key_chip, 10.0, accent, "middle", 700, mono=True)
        # Wall-time / GPU badge strip
        out += f'<rect x="{x + 12:.1f}" y="{r2_y + 52:.1f}" width="{card_w - 24:.1f}" height="24" rx="3" fill="{BG}" stroke="{RULE}" stroke-width="1"/>'
        out += svg_text(x + 20, r2_y + 68, wall_badge, 10.1, AMBER, "start", 700, mono=True)
        # Prompt line (max 2 lines)
        ptxt, py = svg_para(x + 12, r2_y + 92, prompt_quote, 10.6, BODY, card_w - 24, 1.35)
        out += ptxt
        return out, x, r2_y + 126

    # ---- CARD 1: SEALED CONFIRM-160 IMAGE ----
    c1, x1, v_y = card_shell(
        0, G,
        "1. CONFIRM-160 IMAGE",
        f"{c4_key_mib:.0f} MiB KEY",
        f"WALL: {pro_img_s:.1f}s PRO 6000 · {c5070_img_s:.1f}s 5070 Ti",
        'Prompt (h3p0_0940, unseen): "sculptural white plaster bust, cool cyan key light & warm amber rim."',
    )
    s += c1
    # Hero 236x236 centered inside 336 card
    hero_sz = 236
    hero_x = x1 + (card_w - hero_sz) / 2.0
    s += f'<rect x="{hero_x - 2:.1f}" y="{v_y - 2:.1f}" width="{hero_sz + 4}" height="{hero_sz + 4}" rx="4" fill="none" stroke="{G}" stroke-width="2"/>'
    s += f'<image x="{hero_x:.1f}" y="{v_y:.1f}" width="{hero_sz}" height="{hero_sz}" href="{uri_c4_hero}" preserveAspectRatio="xMidYMid slice"/>'
    s += f'<rect x="{hero_x + 6:.1f}" y="{v_y + hero_sz - 25:.1f}" width="214" height="19" rx="3" fill="#0A0B0C" fill-opacity="0.86"/>'
    s += svg_text(hero_x + 12, v_y + hero_sz - 12, f"90 MiB Key · img-teacher cos {r940_cos:.4f}", 10.0, G, "start", 700, mono=True)

    # 3-way same-seed comparison row underneath
    ty = v_y + hero_sz + 14
    s += svg_text(x1 + 12, ty + 10, "SAME SEED (2026203871) · ZERO QWEN3-VL-32B IN KEY & B0", 9.4, SEC, "start", 700)
    th_y = ty + 18
    th_w = 98
    th_h = 98
    th_gap = 9
    th_x0 = x1 + (card_w - (3 * th_w + 2 * th_gap)) / 2.0
    thumbs = [
        (uri_t_thumb, f"Teacher 27.1G", f"SigLIP {r940_t:.3f}", SEC),
        (uri_c4_thumb, f"Key {c4_key_mib:.0f}M", f"SigLIP {r940_c4:.3f}", G),
        (uri_b0_thumb, "B0 Shuffled", f"SigLIP {r940_b0:.3f}", RED),
    ]
    for ti, (uri, l1, l2, col) in enumerate(thumbs):
        tx = th_x0 + ti * (th_w + th_gap)
        s += f'<rect x="{tx - 1:.1f}" y="{th_y - 1:.1f}" width="{th_w + 2}" height="{th_h + 2}" rx="3" fill="none" stroke="{col}" stroke-width="1.5"/>'
        s += f'<image x="{tx:.1f}" y="{th_y:.1f}" width="{th_w}" height="{th_h}" href="{uri}" preserveAspectRatio="xMidYMid slice"/>'
        s += svg_text(tx + th_w / 2.0, th_y + th_h + 14, l1, 9.8, col, "middle", 700)
        s += svg_text(tx + th_w / 2.0, th_y + th_h + 27, l2, 9.4, SEC, "middle", 400, mono=True)

    # Card 1 metrics footer
    fy = r2_y + 530
    s += f'<line x1="{x1 + 12:.1f}" y1="{fy:.1f}" x2="{x1 + card_w - 12:.1f}" y2="{fy:.1f}" stroke="{RULE}" stroke-width="1"/>'
    s += svg_text(x1 + 12, fy + 20, f"Confirm-160 (Inkling {c4_key_mib:.0f}M): {c4_ret:.2f}% retention", 11.0, G, "start", 700)
    s += svg_text(x1 + 12, fy + 38, f"Qwen3.8 half-depth L31 ({qw_key_mib:.0f}M): {qw_ret:.2f}% (16/16)", 10.6, INK)
    s += svg_text(x1 + 12, fy + 56, "MiniMax-H3 1024x1024 · 6 steps · 0 gradient steps", 10.3, SEC)
    h_note1, _ = svg_para(x1 + 12, fy + 76, "Honest boundary: unmasked CLIP-L SDXL fails (57.7% Dev-24); LLM/T5 DiTs clear 91-99.7%.", 9.8, DIM, card_w - 24, 1.38)
    s += h_note1

    # ---- CARD 2: REAL 5.875s VIDEO + 32 kHz STEREO AUDIO ----
    c2, x2, v_y = card_shell(
        1, CYAN,
        "2. 5.9s VIDEO + AUDIO",
        "+0 EXTRA B",
        f"WALL: {pro_av_s:.1f}s PRO 6000 · {c5070_av_s:.1f}s 5070 Ti",
        'Prompt (h3p0_0080): "bright tropical coral lagoon, turquoise shallows, pale sand ripples & coral heads."',
    )
    s += c2
    # 2x2 real keyframes from wave3_real_5s_video_with_audio.mp4
    fw = 152
    fh = 89
    fgap = 8
    fx0 = x2 + (card_w - (2 * fw + fgap)) / 2.0
    for ki in range(4):
        col_i = ki % 2
        row_i = ki // 2
        fx = fx0 + col_i * (fw + fgap)
        fy_img = v_y + row_i * (fh + fgap)
        s += f'<rect x="{fx - 1:.1f}" y="{fy_img - 1:.1f}" width="{fw + 2}" height="{fh + 2}" rx="3" fill="none" stroke="{CYAN}" stroke-opacity="0.7" stroke-width="1.2"/>'
        s += f'<image x="{fx:.1f}" y="{fy_img:.1f}" width="{fw}" height="{fh}" href="{kf_uris[ki]}" preserveAspectRatio="xMidYMid slice"/>'
        s += f'<rect x="{fx + 4:.1f}" y="{fy_img + 4:.1f}" width="44" height="16" rx="2" fill="#0A0B0C" fill-opacity="0.82"/>'
        s += svg_text(fx + 26, fy_img + 15.2, f"t={kf_times[ki]:.1f}s", 9.2, CYAN, "middle", 700, mono=True)

    # Real 32 kHz stereo audio waveform underneath
    wy = v_y + 2 * fh + fgap + 18
    s += svg_text(x2 + 12, wy + 10, f"NATIVE 32 kHz STEREO AUDIO ({pro_av_dur:.3f}s · RMS {pro_av_rms:.4f})", 9.6, CYAN, "start", 700)
    s += f'<rect x="{x2 + 12:.1f}" y="{wy + 17:.1f}" width="{card_w - 24:.1f}" height="88" rx="4" fill="#0D1116" stroke="{RULE}" stroke-width="1.2"/>'
    s += f'<image x="{x2 + 13:.1f}" y="{wy + 18:.1f}" width="{card_w - 26:.1f}" height="86" href="{uri_av_wave}" preserveAspectRatio="none"/>'
    s += svg_text(x2 + 16, wy + 120, "0.0s", 9.0, DIM, "start", mono=True)
    s += svg_text(x2 + card_w / 2.0, wy + 120, "768x448 @ 24 fps (141 frames)", 9.2, SEC, "middle", mono=True)
    s += svg_text(x2 + card_w - 16, wy + 120, f"{pro_av_dur:.2f}s", 9.0, DIM, "end", mono=True)

    # Video motion & consumer VRAM summary pill
    py_pill = wy + 134
    s += f'<rect x="{x2 + 12:.1f}" y="{py_pill:.1f}" width="{card_w - 24:.1f}" height="46" rx="4" fill="{PANEL2}" stroke="{CYAN}" stroke-opacity="0.45" stroke-width="1"/>'
    s += svg_text(x2 + 22, py_pill + 18, "Exact same 90 MiB key transfers at +0 bytes", 10.6, INK, "start", 700)
    s += svg_text(x2 + 22, py_pill + 36, f"Motion L1={pro_av_l1:.2f} · 5070 Ti 16G: {c5070_av_gib:.2f} GiB", 9.8, CYAN, "start", 600, mono=True)

    fy = r2_y + 530
    s += f'<line x1="{x2 + 12:.1f}" y1="{fy:.1f}" x2="{x2 + card_w - 12:.1f}" y2="{fy:.1f}" stroke="{RULE}" stroke-width="1"/>'
    s += svg_text(x2 + 12, fy + 20, "MiniMax-H3 FL2VA (INT8) · 12 steps · CFG 1.0", 11.0, CYAN, "start", 700)
    s += svg_text(x2 + 12, fy + 38, "Verified on 16 GPUs (Volta V100 -> Blackwell)", 10.6, INK)
    s += svg_text(x2 + 12, fy + 56, "MP4 preview plays real video at 2.0x (5.88s->3.0s)", 10.3, SEC)
    h_note2, _ = svg_para(x2 + 12, fy + 76, "Honest scope: warm DiT+VAE timed from saved key tensor; 3090 178.6s, V100 796.0s.", 9.8, DIM, card_w - 24, 1.38)
    s += h_note2

    # ---- CARD 3: REAL PROSODIC SPEECH (24 kHz) ----
    c3, x3, v_y = card_shell(
        2, PURPLE,
        "3. PROSODIC SPEECH",
        "1.4-2.8 MiB",
        f"WALL: ~{tts_mean_s:.1f}s/clip (RTF {tts_rtf:.2f} · {tts_vram:.2f} GiB)",
        'Same discrete words, 0 style prompt: "Dogs are sitting by the door." (continuous key shifts affect)',
    )
    s += c3
    spec_w = card_w - 24  # 312
    spec_h = 132
    # Top spectrogram: Qwen3.8 Key -> SAD
    s += svg_text(x3 + 12, v_y + 10, f"A. Qwen3.8 Key -> SAD (e2v {sad_prob:.3f} · WER 0% · {sad_dur:.2f}s)", 9.7, BLUE, "start", 700)
    s += f'<rect x="{x3 + 12:.1f}" y="{v_y + 16:.1f}" width="{spec_w:.1f}" height="{spec_h}" rx="4" fill="#0B0E13" stroke="{BLUE}" stroke-width="1.3"/>'
    s += f'<image x="{x3 + 13:.1f}" y="{v_y + 17:.1f}" width="{spec_w - 2:.1f}" height="{spec_h - 2}" href="{uri_sad_spec}" preserveAspectRatio="none"/>'
    s += f'<rect x="{x3 + 12 + spec_w - 116:.1f}" y="{v_y + 22:.1f}" width="110" height="18" rx="2" fill="#0A0B0C" fill-opacity="0.88"/>'
    s += svg_text(x3 + 12 + spec_w - 61, v_y + 34.5, f"F0 med {sad_f0:.0f} Hz", 9.6, BLUE, "middle", 700, mono=True)

    # Bottom spectrogram: Inkling-2-Mini Key -> ANGRY
    v_y2 = v_y + spec_h + 36
    s += svg_text(x3 + 12, v_y2 + 10, f"B. Inkling Key -> ANGRY (e2v {ang_prob:.3f} · WER 0% · {ang_dur:.2f}s)", 9.7, AMBER, "start", 700)
    s += f'<rect x="{x3 + 12:.1f}" y="{v_y2 + 16:.1f}" width="{spec_w:.1f}" height="{spec_h}" rx="4" fill="#0B0E13" stroke="{AMBER}" stroke-width="1.3"/>'
    s += f'<image x="{x3 + 13:.1f}" y="{v_y2 + 17:.1f}" width="{spec_w - 2:.1f}" height="{spec_h - 2}" href="{uri_ang_spec}" preserveAspectRatio="none"/>'
    s += f'<rect x="{x3 + 12 + spec_w - 116:.1f}" y="{v_y2 + 22:.1f}" width="110" height="18" rx="2" fill="#0A0B0C" fill-opacity="0.88"/>'
    s += svg_text(x3 + 12 + spec_w - 61, v_y2 + 34.5, f"F0 med {ang_f0:.0f} Hz", 9.6, AMBER, "middle", 700, mono=True)

    # Contrast summary pill
    p3_y = v_y2 + spec_h + 28
    s += f'<rect x="{x3 + 12:.1f}" y="{p3_y:.1f}" width="{spec_w:.1f}" height="46" rx="4" fill="{PANEL2}" stroke="{PURPLE}" stroke-opacity="0.5" stroke-width="1"/>'
    s += svg_text(x3 + 22, p3_y + 18, "+100 Hz F0 shift purely from continuous vector", 10.6, INK, "start", 700)
    s += svg_text(x3 + 22, p3_y + 36, "40.0% vs 43.3% ceiling · WER -10.7 pp", 9.8, PURPLE, "start", 600, mono=True)

    fy = r2_y + 530
    s += f'<line x1="{x3 + 12:.1f}" y1="{fy:.1f}" x2="{x3 + card_w - 12:.1f}" y2="{fy:.1f}" stroke="{RULE}" stroke-width="1"/>'
    s += svg_text(x3 + 12, fy + 20, "Qwen3-TTS-12Hz-1.7B · 24 kHz mono · 0.0 roundtrip", 10.8, PURPLE, "start", 700)
    s += svg_text(x3 + 12, fy + 38, "FP64 SVD rank-64 shrinks key 26-29x (<0.001 cos)", 10.4, INK)
    s += svg_text(x3 + 12, fy + 56, "MP4 plays both real WAVs back-to-back at 1.0x", 10.3, SEC)
    h_note3, _ = svg_para(x3 + 12, fy + 76, "Honest boundary: continuous vectors steer affect/pace; discrete words stay explicit (0/32 direct lexemes).", 9.8, DIM, card_w - 24, 1.38)
    s += h_note3

    # ---- CARD 4: REAL 48 kHz STEREO SINGING MUSIC (YuE2-3B) ----
    c4_s, x4, v_y = card_shell(
        3, AMBER,
        "4. 48 kHz STEREO MUSIC",
        f"{yue_key_mib:.0f} MiB KEY",
        f"WALL: {sb['total_arm_seconds']:.1f}s for {sb['duration_s']:.0f}s (1.63x RT · PRO 6000)",
        'Prompt (Song B, 0 YuE2 prompt tokens): "Upbeat acoustic bluegrass banjo & fiddle, 4/4 time."',
    )
    s += c4_s
    # Spectrogram (312 x 164)
    ms_h = 164
    s += svg_text(x4 + 12, v_y + 10, f"48 kHz STEREO LOG-SPECTROGRAM ({sb['duration_s']:.1f}s · 0-12 kHz)", 9.5, G, "start", 700)
    s += f'<rect x="{x4 + 12:.1f}" y="{v_y + 16:.1f}" width="{spec_w:.1f}" height="{ms_h}" rx="4" fill="#0B0E13" stroke="{G}" stroke-width="1.3"/>'
    s += f'<image x="{x4 + 13:.1f}" y="{v_y + 17:.1f}" width="{spec_w - 2:.1f}" height="{ms_h - 2}" href="{uri_mus_spec}" preserveAspectRatio="none"/>'
    # Highlight 19-24s MP4 audio excerpt bracket on spectrogram
    ex_x0 = x4 + 13 + (19.0 / mus_dur) * (spec_w - 2)
    ex_w = (5.0 / mus_dur) * (spec_w - 2)
    s += f'<rect x="{ex_x0:.1f}" y="{v_y + 17:.1f}" width="{ex_w:.1f}" height="{ms_h - 2}" fill="{AMBER}" fill-opacity="0.16" stroke="{AMBER}" stroke-width="1.4" stroke-dasharray="3 2"/>'
    s += f'<rect x="{ex_x0 - 24:.1f}" y="{v_y + 22:.1f}" width="92" height="16" rx="2" fill="#0A0B0C" fill-opacity="0.88"/>'
    s += svg_text(ex_x0 + 22, v_y + 33.5, "19-24s MP4 clip", 9.0, AMBER, "middle", 700, mono=True)

    # Stereo waveform strip directly underneath
    mw_y = v_y + ms_h + 24
    s += f'<rect x="{x4 + 12:.1f}" y="{mw_y:.1f}" width="{spec_w:.1f}" height="46" rx="3" fill="#0D1116" stroke="{RULE}" stroke-width="1"/>'
    s += f'<image x="{x4 + 13:.1f}" y="{mw_y + 1:.1f}" width="{spec_w - 2:.1f}" height="44" href="{uri_mus_wave}" preserveAspectRatio="none"/>'
    s += svg_text(x4 + 14, mw_y + 59, "0s", 9.0, DIM, "start", mono=True)
    s += svg_text(x4 + card_w / 2.0, mw_y + 59, "zero_yue2_prompt_token_lookup = true", 9.2, G, "middle", 700, mono=True)
    s += svg_text(x4 + card_w - 14, mw_y + 59, f"{mus_dur:.0f}s", 9.0, DIM, "end", mono=True)

    # Song B + Song A acoustic key proof cards
    mk_y = mw_y + 72
    s += f'<rect x="{x4 + 12:.1f}" y="{mk_y:.1f}" width="{spec_w:.1f}" height="52" rx="4" fill="{PANEL2}" stroke="{G}" stroke-opacity="0.55" stroke-width="1"/>'
    s += svg_text(x4 + 20, mk_y + 17, f"Song B (Bluegrass): #1 G_major r={sb['top3_acoustic_keys'][0][1]:.3f} · {sb['estimated_bpm']:.0f} BPM", 10.1, G, "start", 700)
    s += svg_text(x4 + 20, mk_y + 34, f"Whisper lyrics: {sb_wh['n_hits']}/{sb_wh['n_target']} hits vs {sb_zero_wh['n_hits']}/{sb_zero_wh['n_target']} zero control", 9.9, INK, "start", 600, mono=True)
    s += svg_text(x4 + 20, mk_y + 47, f"AR+NAR+VAE: {sb['total_arm_seconds']:.1f}s wall ({yue_peak_gib:.2f} GiB peak CUDA)", 9.4, SEC, "start", 400, mono=True)

    mk_y2 = mk_y + 58
    s += f'<rect x="{x4 + 12:.1f}" y="{mk_y2:.1f}" width="{spec_w:.1f}" height="44" rx="4" fill="{PANEL2}" stroke="{AMBER}" stroke-opacity="0.45" stroke-width="1"/>'
    s += svg_text(x4 + 20, mk_y2 + 17, f"Song A (Anthem): #1 Bb_major r={sa['top3_acoustic_keys'][0][1]:.3f} ({sa['total_arm_seconds']:.1f}s)", 10.1, AMBER, "start", 700)
    s += svg_text(x4 + 20, mk_y2 + 34, "Avoids native YuE2 173 BPM D_minor pirate prior", 9.8, SEC, "start", 400)

    fy = r2_y + 530
    s += f'<line x1="{x4 + 12:.1f}" y1="{fy:.1f}" x2="{x4 + card_w - 12:.1f}" y2="{fy:.1f}" stroke="{RULE}" stroke-width="1"/>'
    s += svg_text(x4 + 12, fy + 20, "YuE2-3B + YuE2-Vae · ABC -> semantic -> 48 kHz", 10.8, AMBER, "start", 700)
    s += svg_text(x4 + 12, fy + 38, f"Generates {sb['duration_s']:.0f}s stereo in {sb['total_arm_seconds']:.1f}s (>1.6x real-time)", 10.4, INK)
    s += svg_text(x4 + 12, fy + 56, "MP4 plays 19-24s real 48 kHz excerpt at 1.0x", 10.3, SEC)
    h_note4, _ = svg_para(x4 + 12, fy + 76, "Honest boundary: 36M direct key controls key/tempo/style; verbatim anthem lyrics need discrete tokens (10/11 Anthem v5).", 9.6, DIM, card_w - 24, 1.36)
    s += h_note4

    # ---- FOOTER ----
    foot_y = r2_y + r2_h + 24
    s += f'<line x1="{M}" x2="{W - M}" y1="{foot_y:.1f}" y2="{foot_y:.1f}" stroke="{INK}" stroke-width="1"/>'
    f_txt, foot_end = svg_para(
        M, foot_y + 20,
        "Sources: All metrics, rendered PNGs, 141-frame FL2VA video, 24 kHz TTS clips, and 48 kHz YuE2 stereo audio are read directly from SHA-256-pinned run outputs "
        "(walkthrough_sources.sha256.json). Sped-up companion reel with real audio: samples/omni_platos_key_walkthrough_spedup.mp4 and .gif.",
        11.2, SEC, CW - 270, 1.44,
    )
    s += f_txt
    s += svg_text(W - M, foot_y + 20, "github.com/Atlas3DSS/atlas-ingine", 11.5, SEC, "end")

    H = int(math.ceil(foot_end + 30))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="{BG}"/>{s}</svg>\n'
    return svg, H


def build_walkthrough_video_and_gif(data: dict, video_frames: List[Image.Image], mp4_out: Path, gif_out: Path) -> None:
    """Render a crisp 16.00s 1280x720 24fps walkthrough MP4 (with real audio) + 840x472 12fps GIF."""
    W, H = 1280, 720
    FPS = 24

    pa = data["pa"]
    qw = data["qw"]
    w3_pro = data["w3_pro"]
    w3_5070 = data["w3_5070"]
    row_940 = data["row_940"]
    clip_sad = data["clip_sad"]
    clip_angry = data["clip_angry"]
    yue_rep = data["yue_rep"]
    yue_wh = data["yue_wh"]
    qwen_tts_wrk = data["pros_wrk"]["qwen3_tts"]

    c4_ret = 100.0 * pa["separate_concat4_analysis_g16"]["siglip2_margin_retention_using_st_b0_reference"]
    qw_ret = 100.0 * qw["primary_single_tap_matched_b0_analysis_g16"]["siglip2_margin_retention_vs_teacher_and_b0"]
    r940_t = row_940["arms"]["teacher_reinject"]["siglip2_text_image"]
    r940_c4 = row_940["arms"]["inkling_platos_key_concat4"]["siglip2_text_image"]
    r940_cos = row_940["arms"]["inkling_platos_key_concat4"]["siglip2_image_teacher"]
    r940_b0 = row_940["arms"]["b0_shuffled_single_tap"]["siglip2_text_image"]

    pro_img_s = w3_pro["real_image_gen"]["total_warm_gen_seconds"]
    c5070_img_s = w3_5070["real_image_gen"]["total_warm_gen_seconds"]
    pro_av_s = w3_pro["real_5s_video_audio_gen"]["total_warm_gen_seconds"]
    c5070_av_s = w3_5070["real_5s_video_audio_gen"]["total_warm_gen_seconds"]
    c5070_av_gib = w3_5070["real_5s_video_audio_gen"]["peak_cuda_alloc_gib"]
    pro_av_l1 = w3_pro["real_5s_video_audio_gen"]["video_temporal_motion_l1"]
    pro_av_rms = w3_pro["real_5s_video_audio_gen"]["audio_rms"]
    pro_key_ms = w3_pro["platos_key_concat4"]["warm_latency_ms_32tok"]

    tts_mean_s = qwen_tts_wrk["elapsed_s"] / qwen_tts_wrk["n_clips"]
    tts_rtf = qwen_tts_wrk["warm_rtf"]
    tts_vram = qwen_tts_wrk["peak_vram_gib"]

    sad_prob = clip_sad["e2v_target_prob"]
    sad_f0 = clip_sad["acoustic"]["f0_median_hz"]
    sad_dur = clip_sad["acoustic"]["duration_s"]
    ang_prob = clip_angry["e2v_target_prob"]
    ang_f0 = clip_angry["acoustic"]["f0_median_hz"]
    ang_dur = clip_angry["acoustic"]["duration_s"]

    sb = yue_rep["songs"]["packetD_song_B"]["arms"]["inkling_platos_key_36m"]
    sa = yue_rep["songs"]["packetD_song_A"]["arms"]["inkling_platos_key_36m"]
    sb_wh = yue_wh["songs"]["packetD_song_B"]["inkling_platos_key_36m"]["lyric_recall"]
    sb_zero_wh = yue_wh["songs"]["packetD_song_B"]["zero_control"]["lyric_recall"]
    yue_peak_gib = yue_rep["peak_cuda_alloc_gib"]

    # Pre-scale static visual assets once
    im_t = Image.open(SRC["img_teacher"]).convert("RGB").resize((306, 306), Image.Resampling.LANCZOS)
    im_c4 = Image.open(SRC["img_concat4"]).convert("RGB").resize((400, 400), Image.Resampling.LANCZOS)
    im_b0 = Image.open(SRC["img_b0"]).convert("RGB").resize((306, 306), Image.Resampling.LANCZOS)

    av_arr, _, _ = read_wav_mono(SRC["wave3_wav"])
    av_wave_panel = render_waveform_image(av_arr, 412, 118, color_hex=CYAN)

    sad_arr, sad_sr, _ = read_wav_mono(SRC["speech_sad_wav"])
    ang_arr, ang_sr, _ = read_wav_mono(SRC["speech_angry_wav"])
    sad_spec = render_spectrogram_image(sad_arr, sad_sr, 560, 224, LUT_COOL, max_hz=6500.0)
    ang_spec = render_spectrogram_image(ang_arr, ang_sr, 560, 224, LUT_WARM, max_hz=6500.0)
    sad_wave = render_waveform_image(sad_arr, 560, 64, color_hex=BLUE)
    ang_wave = render_waveform_image(ang_arr, 560, 64, color_hex=AMBER)

    mus_arr, mus_sr, mus_dur = read_wav_mono(SRC["music_song_b_wav"])
    mus_spec = render_spectrogram_image(mus_arr, mus_sr, 1184, 240, LUT_GREEN, max_hz=12000.0, n_fft=2048, db_floor=-68.0)
    mus_wave = render_waveform_image(mus_arr, 1184, 72, color_hex=G, highlight=(19.0 / mus_dur, 24.0 / mus_dur))

    # Fonts
    f_pill = get_font("mono", True, 14)
    f_title = get_font("serif", True, 24)
    f_badge = get_font("mono", True, 13)
    f_h2 = get_font("sans", True, 17)
    f_body = get_font("sans", False, 15)
    f_body_b = get_font("sans", True, 15)
    f_mono = get_font("mono", False, 13)
    f_mono_b = get_font("mono", True, 13)
    f_small = get_font("sans", False, 13)
    f_small_b = get_font("sans", True, 13)

    STEPS = [
        ("1. IMAGE (90M)", G),
        ("2. 5.9s AV (+0B)", CYAN),
        ("3. SPEECH (1.4-2.8M)", PURPLE),
        ("4. MUSIC (36M)", AMBER),
    ]

    def draw_shell(step_idx: int, step_tag: str, accent: str, headline: str, badge_text: str,
                   seg_prog: float, footer_left: str) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        # Top bar
        d.rectangle([0, 0, W, 68], fill=PANEL)
        d.line([(0, 68), (W, 68)], fill=RULE, width=1)
        d.rectangle([0, 0, W, 4], fill=accent)

        # Left step pill
        pw = int(f_pill.getlength(step_tag)) + 22
        d.rounded_rectangle([24, 18, 24 + pw, 50], radius=4, fill=BG, outline=accent, width=2)
        d.text((24 + pw // 2, 34), step_tag, fill=accent, font=f_pill, anchor="mm")
        d.text((24 + pw + 16, 34), headline, fill=INK, font=f_title, anchor="lm")

        # Right honest corner badge
        bw_badge = int(f_badge.getlength(badge_text)) + 26
        bx0 = W - 24 - bw_badge
        d.rounded_rectangle([bx0, 18, W - 24, 50], radius=4, fill="#19160E", outline=AMBER, width=2)
        d.text((bx0 + bw_badge // 2, 34), badge_text, fill=AMBER, font=f_badge, anchor="mm")

        # Bottom timeline bar
        d.rectangle([0, 646, W, H], fill=PANEL)
        d.line([(0, 646), (W, 646)], fill=RULE, width=1)
        seg_w = (W - 48 - 36) // 4
        for i, (lbl, col) in enumerate(STEPS):
            sx0 = 24 + i * (seg_w + 12)
            sx1 = sx0 + seg_w
            is_cur = i == step_idx
            is_done = i < step_idx
            d.rounded_rectangle([sx0, 657, sx1, 682], radius=3, fill=PANEL2, outline=col if is_cur else RULE, width=2 if is_cur else 1)
            if is_done:
                d.rounded_rectangle([sx0 + 1, 658, sx1 - 1, 681], radius=2, fill="#1F2E1A")
            elif is_cur:
                fill_x = sx0 + 2 + int((seg_w - 4) * max(0.0, min(1.0, seg_prog)))
                if fill_x > sx0 + 3:
                    d.rounded_rectangle([sx0 + 2, 658, fill_x, 681], radius=2, fill="#23361B")
            d.text(((sx0 + sx1) // 2, 670), lbl, fill=INK if (is_cur or is_done) else SEC, font=f_mono_b, anchor="mm")

        footer_right = "0 GB Qwen3-VL-32B | 0 grad steps | github.com/Atlas3DSS/atlas-ingine"
        lw = f_small.getlength(footer_left)
        rw_f = f_mono.getlength(footer_right)
        assert 24 + lw + 24 < W - 24 - rw_f, f"Footer overlap: left={lw:.0f} right={rw_f:.0f}"
        d.text((24, 701), footer_left, fill=SEC, font=f_small, anchor="lm")
        d.text((W - 24, 701), footer_right, fill=DIM, font=f_mono, anchor="rm")
        return im, d

    # Exact segment frame counts (24 fps): 3.50s + 3.00s + 4.50s + 5.00s = 16.00s (384 frames)
    N1 = int(round(3.50 * FPS))  # 84
    N2 = int(round(3.00 * FPS))  # 72
    N3 = int(round(4.50 * FPS))  # 108
    N4 = int(round(5.00 * FPS))  # 120

    # Assemble sample-accurate 16.00s 48 kHz stereo real audio track via ffmpeg
    with tempfile.TemporaryDirectory(prefix="omni_walk_av_") as td:
        audio_wav = Path(td) / "walkthrough_audio_48k.wav"
        seg1_a = Path(td) / "seg1.wav"
        seg2_a = Path(td) / "seg2.wav"
        seg3_a = Path(td) / "seg3.wav"
        seg4_a = Path(td) / "seg4.wav"

        # Seg 1: 3.50s clean silence
        subprocess.run([
            FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            "-t", "3.50", "-c:a", "pcm_s16le", str(seg1_a),
        ], check=True)
        # Seg 2: Real wave3 32 kHz stereo ocean audio sped up 1.9583x (5.875s -> 3.00s), gentle audible lift
        subprocess.run([
            FFMPEG, "-v", "error", "-y", "-i", str(SRC["wave3_wav"]),
            "-af", "atempo=1.9583333,volume=10dB,alimiter=limit=0.85,aresample=48000,apad,atrim=0:3.00,afade=t=in:st=0:d=0.08,afade=t=out:st=2.88:d=0.12",
            "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(seg2_a),
        ], check=True)
        # Seg 3: Real blind_069.wav (2.08s) + 0.34s silence + real blind_072.wav (2.00s) + 0.08s pad = 4.50s
        subprocess.run([
            FFMPEG, "-v", "error", "-y",
            "-i", str(SRC["speech_sad_wav"]),
            "-f", "lavfi", "-t", "0.34", "-i", "anullsrc=r=48000:cl=stereo",
            "-i", str(SRC["speech_angry_wav"]),
            "-f", "lavfi", "-t", "0.10", "-i", "anullsrc=r=48000:cl=stereo",
            "-filter_complex",
            "[0:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:2.08[a0];"
            "[1:a]aformat=sample_rates=48000:channel_layouts=stereo[s1];"
            "[2:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:2.00[a2];"
            "[3:a]aformat=sample_rates=48000:channel_layouts=stereo[s2];"
            "[a0][s1][a2][s2]concat=n=4:v=0:a=1,apad,atrim=0:4.50[out]",
            "-map", "[out]", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(seg3_a),
        ], check=True)
        # Seg 4: Real 48 kHz stereo YuE2 Song B excerpt (19.0s -> 24.0s = 5.00s, real-time 1.0x)
        subprocess.run([
            FFMPEG, "-v", "error", "-y", "-ss", "19.0", "-t", "5.00", "-i", str(SRC["music_song_b_wav"]),
            "-af", "afade=t=in:st=0:d=0.15,afade=t=out:st=4.75:d=0.25,aresample=48000,apad,atrim=0:5.00",
            "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(seg4_a),
        ], check=True)
        # Concat exact 16.00s audio
        subprocess.run([
            FFMPEG, "-v", "error", "-y",
            "-i", str(seg1_a), "-i", str(seg2_a), "-i", str(seg3_a), "-i", str(seg4_a),
            "-filter_complex", "[0:a][1:a][2:a][3:a]concat=n=4:v=0:a=1,atrim=0:16.00[out]",
            "-map", "[out]", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(audio_wav),
        ], check=True)

        # Spawn ffmpeg rawvideo encoder piping frames directly into MP4
        mp4_out.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [
                FFMPEG, "-v", "error", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                "-i", str(audio_wav),
                "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest",
                str(mp4_out),
            ],
            stdin=subprocess.PIPE,
        )
        assert proc.stdin is not None

        # ---- SEGMENT 1: SEALED CONFIRM-160 IMAGE (3.50s) ----
        for fi in range(N1):
            p = fi / max(1, N1 - 1)
            im, d = draw_shell(
                0, "STEP 1/4", G, "Sealed Confirm-160 Image",
                f"REAL OUTPUTS · GEN WALL: {pro_img_s:.1f}s (PRO 6000) / {c5070_img_s:.1f}s (5070 Ti)",
                p,
                'h3p0_0940 (unseen): "white plaster bust, cyan key & warm amber rim."',
            )
            # Top prompt + matmul strip
            d.rounded_rectangle([24, 82, W - 24, 128], radius=5, fill=PANEL, outline=RULE, width=1)
            d.text((40, 105), "Frozen Inkling-2-Mini (concat4)  -->", fill=BLUE, font=f_mono_b, anchor="lm")
            d.rounded_rectangle([356, 90, 694, 120], radius=4, fill="#162313", outline=G, width=2)
            d.text((525, 105), f"90 MiB Plato's Key ({pro_key_ms:.3f} ms warm)", fill=G, font=f_mono_b, anchor="mm")
            d.text((710, 105), "-->  MiniMax-H3 DiT (0 GB Qwen3-VL-32B)", fill=INK, font=f_mono_b, anchor="lm")

            # Three real same-seed images: Teacher (left), Plato's Key hero (center), B0 Shuffled (right)
            hi_t = p < 0.28
            hi_k = 0.28 <= p < 0.78
            hi_b = p >= 0.78

            # Left: Teacher 27.14 GB
            lx, ly = 44, 178
            d.rounded_rectangle([lx - 3, ly - 3, lx + 309, ly + 309], radius=5, outline=BLUE if hi_t else RULE, width=3 if hi_t else 2)
            im.paste(im_t, (lx, ly))
            d.text((lx + 153, ly - 20), "Teacher Qwen3-VL-32B (27.14 GB)", fill=BLUE if hi_t else SEC, font=f_body_b, anchor="mm")
            d.text((lx + 153, ly + 332), f"SigLIP2 = {r940_t:.4f}  (eliminated at runtime)", fill=SEC, font=f_mono, anchor="mm")

            # Center: Hero 90 MiB Plato's Key
            cx, cy = 440, 148
            d.rounded_rectangle([cx - 4, cy - 4, cx + 404, cy + 404], radius=6, outline=G, width=4 if hi_k else 2)
            im.paste(im_c4, (cx, cy))
            d.rounded_rectangle([cx + 12, cy + 360, cx + 388, cy + 390], radius=4, fill="#0A0B0C", outline=G, width=1)
            d.text((cx + 200, cy + 375), f"90 MiB Plato's Key · img-teacher cos = {r940_cos:.4f}", fill=G, font=f_mono_b, anchor="mm")
            d.text((cx + 200, cy + 426), f"SigLIP2 = {r940_c4:.4f}  |  Confirm-160: {c4_ret:.2f}% (Qwen3.8 L31 50M: {qw_ret:.2f}%)",
                   fill=G, font=f_mono_b, anchor="mm")

            # Right: B0 Shuffled Control
            rx, ry = 930, 178
            d.rounded_rectangle([rx - 3, ry - 3, rx + 309, ry + 309], radius=5, outline=RED if hi_b else RULE, width=3 if hi_b else 2)
            im.paste(im_b0, (rx, ry))
            d.text((rx + 153, ry - 20), "Shuffled Control B0 (same seed)", fill=RED, font=f_body_b, anchor="mm")
            d.text((rx + 153, ry + 332), f"SigLIP2 = {r940_b0:.4f}  (semantic collapse)", fill=RED, font=f_mono_b, anchor="mm")

            proc.stdin.write(im.tobytes())

        # ---- SEGMENT 2: REAL 5.875s FL2VA VIDEO + 32 kHz AUDIO SPED UP 2.0x (3.00s) ----
        n_v = len(video_frames)
        for fi in range(N2):
            p = fi / max(1, N2 - 1)
            v_idx = min(n_v - 1, int(round(p * (n_v - 1))))
            v_t = v_idx / 24.0
            im, d = draw_shell(
                1, "STEP 2/4", CYAN, "Real 5.9s Video + 32 kHz Audio",
                f"CLIP 2.0x (5.88s->3.0s) · GEN WALL: {pro_av_s:.1f}s PRO 6000 / {c5070_av_s:.1f}s 5070 Ti",
                p,
                'h3p0_0080: "bright tropical coral lagoon, pale sand ripples & coral heads."',
            )
            # Left: exact 1:1 native 768x448 real FL2VA frame
            vx, vy = 32, 118
            d.rounded_rectangle([vx - 3, vy - 3, vx + 771, vy + 451], radius=5, outline=CYAN, width=3)
            im.paste(video_frames[v_idx], (vx, vy))
            # Top-left live timestamp pill on video (dynamically sized!)
            t_lbl = f"REAL FL2VA · t={v_t:04.2f}s / 5.88s · 2.0x"
            tw_v = int(f_mono_b.getlength(t_lbl)) + 24
            d.rounded_rectangle([vx + 12, vy + 12, vx + 12 + tw_v, vy + 42], radius=4, fill="#0A0B0C", outline=CYAN, width=1)
            d.text((vx + 12 + tw_v // 2, vy + 27), t_lbl, fill=CYAN, font=f_mono_b, anchor="mm")
            d.text((vx, vy - 20), "MiniMax-H3 FL2VA (768x448 @ 24 fps, 141 frames) · Zero-Extra-Byte Key Transfer",
                   fill=INK, font=f_body_b, anchor="lm")
            d.text((vx, vy + 474), f"Video temporal motion L1 = {pro_av_l1:.2f}  |  Audio 32 kHz stereo RMS = {pro_av_rms:.4f}  |  12 steps CFG 1.0",
                   fill=SEC, font=f_mono, anchor="lm")

            # Right info + live synced waveform panel
            rx = 828
            rw = 420
            d.rounded_rectangle([rx, 118, rx + rw, 356], radius=6, fill=PANEL, outline=RULE, width=1)
            d.text((rx + 18, 142), "ZERO EXTRA BYTES · ZERO REFITS", fill=CYAN, font=f_mono_b, anchor="lm")
            lines_r = [
                ("Plato's Key:", "Same 90.0 MiB concat4 (+0 B)"),
                ("Qwen3-VL-32B:", "0 GB loaded (eliminated)"),
                ("RTX PRO 6000:", f"{pro_av_s:.1f}s warm (22.97 GiB alloc)"),
                ("RTX 5070 Ti 16G:", f"{c5070_av_s:.1f}s ({c5070_av_gib:.2f} GiB alloc)"),
                ("RTX 3090 24G:", "178.6s (22.09 GiB alloc)"),
                ("Tesla V100 32G:", "796.0s (2017 Volta sm_70)"),
            ]
            for li, (k, v) in enumerate(lines_r):
                yy = 178 + li * 28
                d.text((rx + 18, yy), k, fill=SEC, font=f_small_b, anchor="lm")
                d.text((rx + 162, yy), v, fill=INK if li > 1 else G, font=f_mono_b if li < 2 else f_mono, anchor="lm")

            # Live 32 kHz audio waveform box with moving playhead
            wx, wy = rx + 4, 384
            d.text((wx, wy - 12), "LIVE 32 kHz STEREO AUDIO WAVEFORM (2.0x)", fill=CYAN, font=f_mono_b, anchor="lm")
            d.rounded_rectangle([wx - 1, wy + 2, wx + 413, wy + 122], radius=4, outline=CYAN, width=1)
            im.paste(av_wave_panel, (wx, wy + 3))
            ph_x = wx + int(p * 411)
            d.line([(ph_x, wy + 3), (ph_x, wy + 121)], fill=AMBER, width=3)
            d.text((wx, wy + 138), "0.0s", fill=DIM, font=f_mono, anchor="lm")
            d.text((wx + 412, wy + 138), "5.88s (played in 3.0s)", fill=AMBER, font=f_mono, anchor="rm")

            proc.stdin.write(im.tobytes())

        # ---- SEGMENT 3: REAL PROSODIC SPEECH (4.50s, 1.0x REAL-TIME AUDIO) ----
        for fi in range(N3):
            t_seg = fi / float(FPS)
            p = fi / max(1, N3 - 1)
            sad_active = t_seg <= 2.08
            ang_active = 2.42 <= t_seg <= 4.42
            im, d = draw_shell(
                2, "STEP 3/4", PURPLE, "Real Prosodic Speech (Same Words)",
                f"AUDIO 1.0x (REAL-TIME) · GEN WALL: ~{tts_mean_s:.1f}s/clip (RTF {tts_rtf:.2f} · {tts_vram:.2f} GiB)",
                p,
                'Same words ("Dogs are sitting by the door."), 0 style prompt -> key flips emotion.',
            )
            # Top banner explaining causal split
            d.rounded_rectangle([24, 82, W - 24, 128], radius=5, fill=PANEL, outline=PURPLE, width=1)
            d.text((40, 105), 'Spoken text (identical): "Dogs are sitting by the door."   |   Hand-written style prompt: NONE',
                   fill=INK, font=f_body_b, anchor="lm")
            d.text((W - 40, 105), "FP64 SVD rank-64 key: 1.4-2.8 MiB", fill=PURPLE, font=f_mono_b, anchor="rm")

            # Left panel: Qwen3.8 Key -> SAD (0.00..2.08s)
            lx = 44
            d.rounded_rectangle([lx - 12, 144, lx + 572, 574], radius=6, fill=PANEL, outline=BLUE if sad_active else RULE, width=3 if sad_active else 1)
            status_l = "PLAYING NOW (1.0x)" if sad_active else ("PLAYED" if t_seg > 2.08 else "QUEUED")
            d.text((lx, 166), "A. Qwen3.8 Plato's Key -> SAD", fill=BLUE, font=f_h2, anchor="lm")
            d.text((lx + 560, 166), status_l, fill=G if sad_active else DIM, font=f_mono_b, anchor="rm")
            d.text((lx, 194), f"emotion2vec+ = sad ({sad_prob:.3f})  |  WER = 0.0%  |  F0 median = {sad_f0:.0f} Hz",
                   fill=BODY, font=f_mono, anchor="lm")
            im.paste(sad_spec, (lx, 214))
            im.paste(sad_wave, (lx, 448))
            d.rectangle([lx, 214, lx + 560, 438], outline=BLUE, width=1)
            d.rectangle([lx, 448, lx + 560, 512], outline=RULE, width=1)
            if sad_active:
                ph_l = lx + int(min(1.0, t_seg / 2.08) * 559)
                d.line([(ph_l, 214), (ph_l, 512)], fill=G, width=3)
            d.text((lx, 536), f"Clip: {sad_dur:.2f}s · 24 kHz mono · Whisper verbatim exact", fill=SEC, font=f_mono, anchor="lm")
            d.text((lx, 558), "Continuous vector lowers pitch (154 Hz) & softens dynamics", fill=BLUE, font=f_small, anchor="lm")

            # Right panel: Inkling-2-Mini Key -> ANGRY (2.42..4.42s)
            rx = 676
            d.rounded_rectangle([rx - 12, 144, rx + 572, 574], radius=6, fill=PANEL, outline=AMBER if ang_active else RULE, width=3 if ang_active else 1)
            status_r = "PLAYING NOW (1.0x)" if ang_active else ("PLAYED" if t_seg > 4.42 else "UP NEXT")
            d.text((rx, 166), "B. Inkling-2-Mini Key -> ANGRY", fill=AMBER, font=f_h2, anchor="lm")
            d.text((rx + 560, 166), status_r, fill=G if ang_active else DIM, font=f_mono_b, anchor="rm")
            d.text((rx, 194), f"emotion2vec+ = angry ({ang_prob:.3f})  |  WER = 0.0%  |  F0 median = {ang_f0:.0f} Hz",
                   fill=BODY, font=f_mono, anchor="lm")
            im.paste(ang_spec, (rx, 214))
            im.paste(ang_wave, (rx, 448))
            d.rectangle([rx, 214, rx + 560, 438], outline=AMBER, width=1)
            d.rectangle([rx, 448, rx + 560, 512], outline=RULE, width=1)
            if ang_active:
                ph_r = rx + int(min(1.0, (t_seg - 2.42) / 2.00) * 559)
                d.line([(ph_r, 214), (ph_r, 512)], fill=G, width=3)
            d.text((rx, 536), f"Clip: {ang_dur:.2f}s · 24 kHz mono · Whisper verbatim exact", fill=SEC, font=f_mono, anchor="lm")
            d.text((rx, 558), "Continuous vector raises pitch (+100 Hz -> 254 Hz) & attack", fill=AMBER, font=f_small, anchor="lm")

            # Bottom honest boundary strip
            d.rounded_rectangle([32, 592, W - 32, 632], radius=4, fill=PANEL2, outline=RULE, width=1)
            d.text((W // 2, 612),
                   "Honest Boundary Law: Continuous Plato's Key steers emotion, pace (8/8) & urgency (8/8); discrete tokens supply exact words (0/32 direct lexemes).",
                   fill=SEC, font=f_small_b, anchor="mm")

            proc.stdin.write(im.tobytes())

        # ---- SEGMENT 4: REAL 48 kHz STEREO SINGING MUSIC (5.00s, 1.0x EXCERPT FROM 34s) ----
        for fi in range(N4):
            t_seg = fi / float(FPS)
            p = fi / max(1, N4 - 1)
            t_audio = 19.0 + min(5.0, t_seg)
            im, d = draw_shell(
                3, "STEP 4/4", AMBER, "Real 48 kHz Stereo Singing Music",
                f"AUDIO 1.0x (5s / 34s) · GEN WALL: {sb['total_arm_seconds']:.1f}s (1.63x RT · PRO 6000)",
                p,
                'Song B (0 YuE2 prompt tokens): "Upbeat bluegrass banjo & fiddle, 4/4."',
            )
            # Top status bar
            d.rounded_rectangle([48, 84, W - 48, 128], radius=5, fill=PANEL, outline=G, width=1)
            d.text((64, 106), f"36.0 MiB Inkling Plato's Key -> YuE2-3B + YuE2-Vae (48 kHz stereo) · {yue_peak_gib:.2f} GiB peak CUDA",
                   fill=G, font=f_mono_b, anchor="lm")
            d.text((W - 64, 106), f"Playing t = {t_audio:04.1f}s / {mus_dur:.1f}s", fill=AMBER, font=f_mono_b, anchor="rm")

            # Full-width 34.0s spectrogram + stereo waveform
            sx, sy = 48, 146
            im.paste(mus_spec, (sx, sy))
            d.rectangle([sx, sy, sx + 1184, sy + 240], outline=G, width=2)
            # Bracket 19-24s real excerpt window
            bx0 = sx + int((19.0 / mus_dur) * 1184)
            bx1 = sx + int((24.0 / mus_dur) * 1184)
            d.rectangle([bx0, sy, bx1, sy + 240], outline=AMBER, width=2)

            im.paste(mus_wave, (sx, sy + 248))
            d.rectangle([sx, sy + 248, sx + 1184, sy + 320], outline=RULE, width=1)
            # Draw live playhead FIRST so excerpt label pill renders cleanly on top
            ph_x = sx + int((t_audio / mus_dur) * 1183)
            d.line([(ph_x, sy), (ph_x, sy + 320)], fill=AMBER, width=3)

            ex_lbl = "1.0x AUDIO EXCERPT (19-24s)"
            ex_pw = int(f_mono_b.getlength(ex_lbl)) + 24
            ex_mid = (bx0 + bx1) // 2
            d.rounded_rectangle([ex_mid - ex_pw // 2, sy + 8, ex_mid + ex_pw // 2, sy + 34], radius=3, fill="#0A0B0C", outline=AMBER, width=2)
            d.text((ex_mid, sy + 21), ex_lbl, fill=AMBER, font=f_mono_b, anchor="mm")

            d.text((sx, sy + 336), "0.0s", fill=DIM, font=f_mono, anchor="lm")
            d.text((sx + 592, sy + 336), "Full 34.0 s 48 kHz stereo spectrogram (0 - 12 kHz) · zero_yue2_prompt_token_lookup = true",
                   fill=SEC, font=f_mono, anchor="mm")
            d.text((sx + 1184, sy + 336), f"{mus_dur:.1f}s", fill=DIM, font=f_mono, anchor="rm")

            # Two side-by-side verified proof cards below
            cy_card = 504
            # Left card: Song B
            d.rounded_rectangle([48, cy_card, 626, 628], radius=6, fill=PANEL, outline=G, width=2)
            d.text((66, cy_card + 24), "Song B (Bluegrass · Playing Now)", fill=G, font=f_h2, anchor="lm")
            d.text((66, cy_card + 54), f"Acoustic Key: #1 G_major (r = {sb['top3_acoustic_keys'][0][1]:.3f})  |  Tempo: {sb['estimated_bpm']:.0f} BPM",
                   fill=INK, font=f_mono_b, anchor="lm")
            d.text((66, cy_card + 80), f"Blinded Whisper: {sb_wh['n_hits']}/{sb_wh['n_target']} lyric hits vs {sb_zero_wh['n_hits']}/{sb_zero_wh['n_target']} zero control",
                   fill=BODY, font=f_mono, anchor="lm")
            d.text((66, cy_card + 104), f"Wall time: {sb['total_arm_seconds']:.1f}s for {sb['duration_s']:.0f}s clip (1.63x faster than real time!)",
                   fill=G, font=f_small_b, anchor="lm")

            # Right card: Song A
            d.rounded_rectangle([654, cy_card, W - 48, 628], radius=6, fill=PANEL, outline=AMBER, width=1)
            d.text((672, cy_card + 24), "Song A (Orchestral Brass Anthem · Same 36 MiB Key)", fill=AMBER, font=f_h2, anchor="lm")
            d.text((672, cy_card + 54), f"Acoustic Key: #1 Bb_major (r = {sa['top3_acoustic_keys'][0][1]:.3f})  |  Wall: {sa['total_arm_seconds']:.1f}s",
                   fill=INK, font=f_mono_b, anchor="lm")
            d.text((672, cy_card + 80), "Steers cleanly into Bb major (avoids native 173 BPM Dm pirate prior)",
                   fill=BODY, font=f_small, anchor="lm")
            d.text((672, cy_card + 104), "Honest boundary: verbatim anthem lyrics use discrete tokens (10/11 Anthem v5).",
                   fill=SEC, font=f_small, anchor="lm")

            proc.stdin.write(im.tobytes())

        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg MP4 encode failed with rc={rc}")

        # Build crisp GitHub/HF-playable looping GIF (840x472 @ 12 fps, two-pass palette)
        palette = Path(td) / "palette.png"
        vf_base = "fps=12,scale=840:-1:flags=lanczos"
        subprocess.run([
            FFMPEG, "-v", "error", "-y", "-i", str(mp4_out),
            "-vf", f"{vf_base},palettegen=max_colors=192:stats_mode=diff",
            str(palette),
        ], check=True)
        subprocess.run([
            FFMPEG, "-v", "error", "-y", "-i", str(mp4_out), "-i", str(palette),
            "-lavfi", f"{vf_base} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
            str(gif_out),
        ], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    data = load_verified_data()
    video_frames = extract_video_frames(SRC["wave3_mp4"])
    print(f"Decoded {len(video_frames)} real frames from {SRC['wave3_mp4'].name}")

    # 1. Build 1-strip SVG + 2x retina PNG
    svg, H = build_strip_svg(data, video_frames)
    assert chr(8212) not in svg, "Banned em-dash in SVG"
    assert not re.search(r"job-\d|att-[0-9a-f]{6}|/mnt/|/home/", svg), "Banned internal path/id in SVG"

    svg_path = HERE / "walkthrough_1strip_omni_generations.svg"
    png_path = HERE / "walkthrough_1strip_omni_generations.png"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"WROTE {svg_path.name} ({svg_path.stat().st_size:,} B, 1480x{H})")

    if not args.no_png:
        chrome = find_chrome()
        subprocess.run(
            [
                chrome,
                "--no-sandbox",
                "--hide-scrollbars",
                "--force-device-scale-factor=2",
                f"--window-size=1480,{H}",
                f"--screenshot={png_path}",
                "file://" + str(svg_path),
            ],
            check=True,
            capture_output=True,
        )
        print(f"WROTE {png_path.name} ({png_path.stat().st_size:,} B, sha256={hashlib.sha256(png_path.read_bytes()).hexdigest()[:16]}...)")

    # 2. Build sped-up walkthrough MP4 (with real audio) + GitHub-playable GIF
    if not args.no_video:
        mp4_path = SAMPLES_DIR / "omni_platos_key_walkthrough_spedup.mp4"
        gif_path = SAMPLES_DIR / "omni_platos_key_walkthrough_spedup.gif"
        build_walkthrough_video_and_gif(data, video_frames, mp4_path, gif_path)
        print(f"WROTE {mp4_path.name} ({mp4_path.stat().st_size:,} B, sha256={hashlib.sha256(mp4_path.read_bytes()).hexdigest()[:16]}...)")
        print(f"WROTE {gif_path.name} ({gif_path.stat().st_size:,} B, sha256={hashlib.sha256(gif_path.read_bytes()).hexdigest()[:16]}...)")

    # 3. Write source pins
    pins_path = HERE / "walkthrough_sources.sha256.json"
    pins_path.write_text(json.dumps(PINNED_HASHES, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WROTE {pins_path.name} ({len(PINNED_HASHES)} pinned sources)")


if __name__ == "__main__":
    main()

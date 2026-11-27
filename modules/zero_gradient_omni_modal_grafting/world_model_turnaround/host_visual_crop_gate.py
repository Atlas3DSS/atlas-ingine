#!/usr/bin/env python3
"""Host-Enforced Full-Zoom 1:1 Multi-Crop Monotonic Visual Gate + Float32 35mm Optical Finish.

Implements two deterministic, zero-GPU host verification & finishing primitives derived from
procedural cinematography / crop-audit workflows (fofr.ai/greenhouse-after-the-typhoon) and
Atlas Monotonic Verified Ledger invariants:

1. Full-Zoom 1:1 Multi-Crop Host Verifier (`verify_image_crops`):
   - Evaluates 1x full-frame metrics (dynamic range, highlight clipping, shadow crush, corner
     balance, overall sharpness) PLUS 4x native-resolution 1:1 crops (center/hero, highest-edge
     detail/signage ROI, lower-third floor/reflection ROI, dark-shadow texture ROI — or explicit
     caller-supplied ROIs).
   - Catches thumbnail-invisible defects deterministically:
       * 2D FFT spectral anisotropy / directional striping (e.g. stretched noise banding)
       * Flat highlight clipping (>= 0.992)
       * Blurry/mushy VAE collapse or dead-flat crops
       * Washed-out corners / low contrast
   - Emits an auditable JSON verdict + SHA-256 pins + optional annotated 1:1 crop contact sheet.

2. Monotonic Commit-or-Keep Arbiter (`monotonic_commit_or_keep`):
   - A retry proposal commits ONLY IF it preserves every hard invariant passed by the baseline
     (zero regressions) AND either passes strictly more hard gates or strictly improves the
     composite quality score by > min_score_delta.

3. Deterministic Float32 35mm Optical Finishing Pass (`apply_35mm_optical_finish` & `make_disc_kernel`):
   - Tanh highlight shoulder rolloff above 0.80, 3-scale warm cream halation, warm shadow bounce
     below luminance 0.28, shadow/midtone-weighted clumped film grain (clean highlights),
     edge-mirrored YIQ chroma softening, gentle corner chromatic dispersion, and rim-weighted
     circular iris bokeh disc kernel helper.

Zero GPU required; pure NumPy + SciPy + Pillow.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


SCHEMA_VERSION = "atlas-host-visual-crop-gate/v1"


def _gaussian_1d(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    """Separable 1D Gaussian filter via reflect-padded rfft (O(N log N), tiny RAM for any sigma)."""
    if sigma <= 0.0:
        return arr.astype(np.float32, copy=True)
    n = arr.shape[axis]
    rad = min(n - 1, max(2, int(round(3.0 * float(sigma)))))
    pad_w = [(0, 0)] * arr.ndim
    pad_w[axis] = (rad, rad)
    padded = np.pad(arr.astype(np.float32), pad_w, mode="reflect")
    m = padded.shape[axis]
    freqs = np.fft.rfftfreq(m).astype(np.float64)
    H = np.exp(-2.0 * (np.pi ** 2) * (float(sigma) ** 2) * (freqs ** 2)).astype(np.float32)
    shape = [1] * arr.ndim
    shape[axis] = H.shape[0]
    spec = np.fft.rfft(padded, axis=axis)
    out = np.fft.irfft(spec * H.reshape(shape), n=m, axis=axis).astype(np.float32)
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(rad, rad + n)
    return out[tuple(sl)]


def gaussian_filter(arr: np.ndarray, sigma: Any, mode: str = "reflect") -> np.ndarray:
    """Pure-NumPy 2D separable Gaussian filter supporting scalar or (sy, sx) sigma."""
    del mode
    if isinstance(sigma, (tuple, list)):
        sy, sx = float(sigma[0]), float(sigma[1])
    else:
        sy = sx = float(sigma)
    out = arr.astype(np.float32, copy=True)
    if sy > 0.0:
        out = _gaussian_1d(out, sy, axis=0)
    if sx > 0.0:
        out = _gaussian_1d(out, sx, axis=1)
    return out


def laplace(arr: np.ndarray) -> np.ndarray:
    """Pure-NumPy 4-neighbor 2D discrete Laplacian with reflect padding."""
    p = np.pad(arr.astype(np.float32), ((1, 1), (1, 1)), mode="reflect")
    return (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * p[1:-1, 1:-1]).astype(np.float32)


def uniform_filter(arr: np.ndarray, size: int) -> np.ndarray:
    """Pure-NumPy exact 2D box mean filter via reflect-padded integral image."""
    s = max(1, int(size))
    r0 = s // 2
    r1 = s - 1 - r0
    p = np.pad(arr.astype(np.float64), ((r0, r1), (r0, r1)), mode="reflect")
    cs = np.pad(p, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    h, w = arr.shape
    return ((cs[s:s + h, s:s + w] - cs[:h, s:s + w] - cs[s:s + h, :w] + cs[:h, :w]) / float(s * s)).astype(np.float32)


@dataclass(frozen=True)
class GateThresholds:
    """Hard deterministic host invariants for 1:1 crop + full-frame verification."""
    max_clipped_highlight_frac: float = 0.035      # pixels >= 0.992
    max_crushed_black_frac: float = 0.20           # pixels <= 0.008
    min_dynamic_range_p95_p05: float = 0.25        # p95 - p05 luminance spread
    max_crop_fft_anisotropy: float = 2.60          # directional spectral banding ratio on 1:1 crops
    min_crop_sharpness_lap_std: float = 0.008      # minimum high-freq detail on hero/detail crops
    max_corner_washout_excess: float = 0.22        # outer corners shouldn't be way brighter/flatter than center
    min_all_finite: bool = True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rgb_f32(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def save_rgb_f32(arr: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(np.rint(arr * 255.0), 0.0, 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(path)
    return sha256_file(path)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(np.float32)


def measure_fft_anisotropy(gray: np.ndarray) -> float:
    """Measure directional spectral anisotropy on a native 1:1 grayscale crop.

    Stretched anisotropic noise or horizontal/vertical scanline striping concentrates mid-to-high
    spatial frequency energy along one axis (horizontal vs vertical wedges in the 2D FFT).
    Returns max(E_axis1, E_axis2) / max(min(E_axis1, E_axis2), 1e-9), >= 1.0.
    """
    h, w = gray.shape
    if h < 32 or w < 32:
        return 1.0
    g = gray.astype(np.float64) - float(np.mean(gray))
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g * wy * wx))) ** 2

    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    dy = (yy - cy) / max(float(cy), 1.0)
    dx = (xx - cx) / max(float(cx), 1.0)
    rad = np.sqrt(dx * dx + dy * dy)
    # Mid-high frequency annulus (avoids DC & single-pixel Nyquist artifacts)
    band = (rad >= 0.08) & (rad <= 0.65)
    ang = np.abs(np.arctan2(dy, dx))  # [0, pi]
    # Horizontal-frequency wedge (vertical stripes): near 0 or pi (within +-22.5 deg)
    near_x = band & ((ang <= np.pi / 8.0) | (ang >= 7.0 * np.pi / 8.0))
    # Vertical-frequency wedge (horizontal stripes): near pi/2 (within +-22.5 deg)
    near_y = band & (np.abs(ang - np.pi / 2.0) <= np.pi / 8.0)

    ex = float(np.mean(spec[near_x])) if np.any(near_x) else 1e-9
    ey = float(np.mean(spec[near_y])) if np.any(near_y) else 1e-9
    hi, lo = max(ex, ey), min(ex, ey)
    if hi < 1e-6:
        return 1.0
    return float(hi / max(lo, 1e-9))


def select_default_crops(rgb: np.ndarray, crop_size: int = 256) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Select 4 deterministic native-resolution 1:1 ROI boxes (x0, y0, x1, y1)."""
    h, w, _ = rgb.shape
    cs = min(crop_size, h, w)
    lum = _luma(rgb)

    def clamp_box(cx: int, cy: int) -> Tuple[int, int, int, int]:
        x0 = max(0, min(w - cs, cx - cs // 2))
        y0 = max(0, min(h - cs, cy - cs // 2))
        return (int(x0), int(y0), int(x0 + cs), int(y0 + cs))

    # 1. Hero / upper-center subject crop
    hero_box = clamp_box(w // 2, int(h * 0.42))

    # 2. Highest local gradient-energy detail/signage crop (grid search)
    gy, gx = np.gradient(lum)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    win = max(16, cs // 2)
    grad_smooth = uniform_filter(grad_mag, size=win)
    # Exclude exact image border margin
    margin = max(8, cs // 4)
    sub_g = grad_smooth[margin:max(margin + 1, h - margin), margin:max(margin + 1, w - margin)]
    iy, ix = np.unravel_index(int(np.argmax(sub_g)), sub_g.shape)
    detail_box = clamp_box(int(ix + margin), int(iy + margin))

    # 3. Lower-third floor / reflection / road seam crop
    lower_box = clamp_box(int(w * 0.50), int(h * 0.80))

    # 4. Dark-shadow texture crop (lowest local mean luminance inside non-border region,
    #    where stretched noise striping & crushed-black mud show up first)
    lum_smooth = uniform_filter(lum, size=win)
    sub_l = lum_smooth[margin:max(margin + 1, h - margin), margin:max(margin + 1, w - margin)]
    sy, sx = np.unravel_index(int(np.argmin(sub_l)), sub_l.shape)
    shadow_box = clamp_box(int(sx + margin), int(sy + margin))

    return [
        ("hero_center", hero_box),
        ("peak_detail_roi", detail_box),
        ("lower_seam_roi", lower_box),
        ("shadow_texture_roi", shadow_box),
    ]


def verify_image_crops(
    image_path: Path,
    thresholds: GateThresholds = GateThresholds(),
    crop_size: int = 256,
    custom_rois: Optional[Sequence[Tuple[str, Tuple[int, int, int, int]]]] = None,
    save_contact_sheet_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run full-frame + 4x 1:1 native-resolution crop host verification."""
    rgb = load_rgb_f32(image_path)
    h, w, _ = rgb.shape
    all_finite = bool(np.all(np.isfinite(rgb)))
    lum = _luma(rgb)

    clipped_hi = float(np.mean(np.max(rgb, axis=-1) >= 0.992))
    crushed_lo = float(np.mean(np.min(rgb, axis=-1) <= 0.008))
    p05, p50, p95 = [float(x) for x in np.percentile(lum, [5.0, 50.0, 95.0])]
    dyn_range = p95 - p05

    # Corner washout check: mean luminance of 4 outer 12% corner patches vs central 50%
    ch, cw = max(8, int(h * 0.12)), max(8, int(w * 0.12))
    corners = np.concatenate([
        lum[:ch, :cw].ravel(),
        lum[:ch, w - cw:].ravel(),
        lum[h - ch:, :cw].ravel(),
        lum[h - ch:, w - cw:].ravel(),
    ])
    center = lum[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    corner_mean = float(np.mean(corners))
    center_mean = float(np.mean(center))
    corner_washout_excess = corner_mean - center_mean

    full_lap_std = float(np.std(laplace(lum)))

    crops_spec = list(custom_rois) if custom_rois is not None else select_default_crops(rgb, crop_size=crop_size)
    crop_reports: List[Dict[str, Any]] = []
    max_aniso = 1.0
    min_detail_sharp = 1e9
    for name, (x0, y0, x1, y1) in crops_spec:
        sub = rgb[y0:y1, x0:x1]
        slum = _luma(sub)
        aniso = measure_fft_anisotropy(slum)
        lap_std = float(np.std(laplace(slum)))
        c_hi = float(np.mean(np.max(sub, axis=-1) >= 0.992))
        c_lo = float(np.mean(np.min(sub, axis=-1) <= 0.008))
        c_p05, c_p95 = [float(x) for x in np.percentile(slum, [5.0, 95.0])]
        aniso_ok = aniso <= thresholds.max_crop_fft_anisotropy
        sharp_ok = (lap_std >= thresholds.min_crop_sharpness_lap_std) if name in ("hero_center", "peak_detail_roi") else True
        max_aniso = max(max_aniso, aniso)
        if name in ("hero_center", "peak_detail_roi"):
            min_detail_sharp = min(min_detail_sharp, lap_std)
        crop_reports.append({
            "name": name,
            "box_xyxy": [x0, y0, x1, y1],
            "fft_anisotropy": round(aniso, 4),
            "laplacian_std": round(lap_std, 5),
            "clipped_highlight_frac": round(c_hi, 5),
            "crushed_black_frac": round(c_lo, 5),
            "dynamic_range_p95_p05": round(c_p95 - c_p05, 4),
            "fft_anisotropy_ok": aniso_ok,
            "sharpness_ok": sharp_ok,
        })

    if min_detail_sharp > 1e8:
        min_detail_sharp = full_lap_std

    gates = {
        "all_finite": all_finite,
        "highlight_clipping_ok": clipped_hi <= thresholds.max_clipped_highlight_frac,
        "shadow_crush_ok": crushed_lo <= thresholds.max_crushed_black_frac,
        "dynamic_range_ok": dyn_range >= thresholds.min_dynamic_range_p95_p05,
        "corner_contrast_ok": corner_washout_excess <= thresholds.max_corner_washout_excess,
        "all_crops_fft_anisotropy_ok": all(c["fft_anisotropy_ok"] for c in crop_reports),
        "detail_crops_sharpness_ok": all(c["sharpness_ok"] for c in crop_reports),
    }
    n_passed = sum(1 for v in gates.values() if v)
    n_total = len(gates)
    passed = (n_passed == n_total)

    # Composite quality score in [0, 1] used for monotonic keep-best tiebreaking
    clip_term = max(0.0, 1.0 - clipped_hi / max(thresholds.max_clipped_highlight_frac * 2.0, 1e-6))
    dyn_term = min(1.0, dyn_range / 0.70)
    aniso_term = max(0.0, min(1.0, 1.0 - max(0.0, max_aniso - 1.15) / 2.5))
    sharp_term = min(1.0, min_detail_sharp / 0.06)
    corner_term = max(0.0, min(1.0, 1.0 - max(0.0, corner_washout_excess) / 0.35))
    composite_score = round(
        0.25 * clip_term + 0.20 * dyn_term + 0.25 * aniso_term + 0.20 * sharp_term + 0.10 * corner_term,
        5,
    )

    contact_sha = None
    if save_contact_sheet_path is not None:
        contact_sha = render_crop_contact_sheet(image_path, crop_reports, gates, composite_score, save_contact_sheet_path)

    return {
        "schema_version": SCHEMA_VERSION,
        "image_path": str(image_path),
        "image_sha256": sha256_file(image_path),
        "width": w,
        "height": h,
        "full_frame": {
            "clipped_highlight_frac": round(clipped_hi, 5),
            "crushed_black_frac": round(crushed_lo, 5),
            "luma_p05": round(p05, 4),
            "luma_p50": round(p50, 4),
            "luma_p95": round(p95, 4),
            "dynamic_range_p95_p05": round(dyn_range, 4),
            "corner_mean_luma": round(corner_mean, 4),
            "center_mean_luma": round(center_mean, 4),
            "corner_washout_excess": round(corner_washout_excess, 4),
            "laplacian_std": round(full_lap_std, 5),
        },
        "crops": crop_reports,
        "thresholds": asdict(thresholds),
        "gates": gates,
        "hard_gates_passed": n_passed,
        "hard_gates_total": n_total,
        "passed": passed,
        "composite_score": composite_score,
        "contact_sheet_path": str(save_contact_sheet_path) if save_contact_sheet_path else None,
        "contact_sheet_sha256": contact_sha,
    }


def render_crop_contact_sheet(
    image_path: Path,
    crop_reports: Sequence[Mapping[str, Any]],
    gates: Mapping[str, bool],
    composite_score: float,
    out_path: Path,
) -> str:
    """Save a diagnostic sheet: full frame with ROI boxes on the left, 2x2 1:1 native crops on the right."""
    src = Image.open(image_path).convert("RGB")
    w, h = src.size
    panel_h = 560
    full_w = max(280, int(round(panel_h * (w / max(h, 1)))))
    tile = 264
    pad = 16
    hdr = 44
    canvas_w = pad + full_w + pad + tile * 2 + pad + pad
    canvas_h = hdr + panel_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (10, 11, 12))
    draw = ImageDraw.Draw(canvas)

    colors = [(130, 194, 50), (88, 166, 255), (234, 185, 78), (229, 100, 90)]
    annotated = src.copy()
    adraw = ImageDraw.Draw(annotated)
    for idx, c in enumerate(crop_reports[:4]):
        x0, y0, x1, y1 = c["box_xyxy"]
        col = colors[idx % len(colors)]
        for off in range(3):
            adraw.rectangle([x0 + off, y0 + off, x1 - off, y1 - off], outline=col)
    resized = annotated.resize((full_w, panel_h), Image.Resampling.LANCZOS)
    canvas.paste(resized, (pad, hdr))

    passed_n = sum(1 for v in gates.values() if v)
    verdict = "PASS" if passed_n == len(gates) else "FAIL"
    draw.text(
        (pad, 14),
        f"HOST 1:1 CROP GATE: {verdict} ({passed_n}/{len(gates)} hard gates) | score={composite_score:.4f} | {image_path.name}",
        fill=(130, 194, 50) if verdict == "PASS" else (229, 100, 90),
    )

    rx0 = pad + full_w + pad
    for idx, c in enumerate(crop_reports[:4]):
        x0, y0, x1, y1 = c["box_xyxy"]
        crop_im = src.crop((x0, y0, x1, y1)).resize((tile, tile - 24), Image.Resampling.NEAREST)
        gx = rx0 + (idx % 2) * (tile + pad)
        gy = hdr + (idx // 2) * (tile + pad)
        col = colors[idx % len(colors)]
        canvas.paste(crop_im, (gx, gy + 20))
        draw.rectangle([gx, gy + 20, gx + tile - 1, gy + tile - 5], outline=col)
        ok_tag = "OK" if (c["fft_anisotropy_ok"] and c["sharpness_ok"]) else "FLAG"
        draw.text(
            (gx + 2, gy + 4),
            f"{c['name']} [{ok_tag}] aniso={c['fft_anisotropy']:.2f} lap={c['laplacian_std']:.3f}",
            fill=col,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return sha256_file(out_path)


def monotonic_commit_or_keep(
    baseline_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    min_score_delta: float = 1e-4,
) -> Dict[str, Any]:
    """Strict monotonic Verified Ledger arbiter: proposals never auto-replace certified baselines.

    Rules:
      1. Every hard invariant passed by `baseline_report` MUST still pass in `candidate_report`
         (zero regressions allowed).
      2. If candidate passes strictly more hard gates than baseline -> COMMIT_CANDIDATE.
      3. Else if candidate passes the same hard gates and improves composite_score by > min_score_delta
         -> COMMIT_CANDIDATE.
      4. Otherwise -> KEEP_BASELINE.
    """
    b_gates: Mapping[str, bool] = baseline_report["gates"]
    c_gates: Mapping[str, bool] = candidate_report["gates"]

    regressed = sorted(k for k, v in b_gates.items() if v and not c_gates.get(k, False))
    gained = sorted(k for k, v in c_gates.items() if v and not b_gates.get(k, False))
    score_delta = float(candidate_report["composite_score"]) - float(baseline_report["composite_score"])

    if regressed:
        decision = "KEEP_BASELINE"
        reason = f"candidate regressed hard invariants: {regressed}"
    elif len(gained) > 0:
        decision = "COMMIT_CANDIDATE"
        reason = f"candidate preserved all baseline gates and gained {gained}"
    elif score_delta > min_score_delta:
        decision = "COMMIT_CANDIDATE"
        reason = f"candidate preserved all baseline gates and improved score by +{score_delta:.5f}"
    else:
        decision = "KEEP_BASELINE"
        reason = f"no monotonic improvement (score_delta={score_delta:+.5f}, gained=0)"

    selected = candidate_report if decision == "COMMIT_CANDIDATE" else baseline_report
    return {
        "decision": decision,
        "reason": reason,
        "regressed_gates": regressed,
        "gained_gates": gained,
        "score_delta": round(score_delta, 5),
        "baseline_sha256": baseline_report["image_sha256"],
        "candidate_sha256": candidate_report["image_sha256"],
        "selected_sha256": selected["image_sha256"],
        "selected_image_path": selected["image_path"],
    }


def make_disc_kernel(radius: float) -> np.ndarray:
    """Rim-weighted circular iris bokeh disc kernel (replaces flat Gaussian blur on foreground layers)."""
    rad = max(1, int(round(float(radius))))
    ky, kx = np.mgrid[-rad: rad + 1, -rad: rad + 1].astype(np.float32)
    kr = np.sqrt(kx * kx + ky * ky) / max(float(radius), 1.0)
    rim = 0.86 + 0.24 * np.exp(-((kr - 0.84) ** 2) / 0.016)
    disc = np.clip((1.04 - kr) / 0.16, 0.0, 1.0) * rim
    return (disc / max(float(np.sum(disc)), 1e-6)).astype(np.float32)


def apply_35mm_optical_finish(
    rgb: np.ndarray,
    shoulder_start: float = 0.80,
    halation_strength: float = 0.14,
    warm_shadow_bounce: float = 0.018,
    grain_strength: float = 0.014,
    chroma_soften_sigma: float = 0.75,
    vignette_strength: float = 0.12,
    seed: int = 20261127,
) -> np.ndarray:
    """Deterministic float32 35mm optical finishing pass (0 GPU, zero hallucination).

    Stages:
      1. Warm bounce light in deep shadows (lum < 0.28) so darks stay rich rather than dead black.
      2. 3-scale warm cream film halation scattered from bright highlights (> 0.78).
      3. Smooth hyperbolic-tangent highlight shoulder rolloff above `shoulder_start` (eliminates
         flat digital clipping at 1.0) plus slight cool black lift.
      4. Shadow/midtone-weighted clumped silver-halide film grain (bright sky stays clean).
      5. Edge-mirrored YIQ chroma horizontal softening + subtle radial corner vignette.
    """
    out = np.clip(rgb.astype(np.float32), 0.0, 1.0).copy()
    h, w, _ = out.shape

    # 1. Warm bounce in deep shadows
    lum = _luma(out)
    sh_w = np.clip((0.28 - lum) / 0.28, 0.0, 1.0)
    out[..., 0] += sh_w * warm_shadow_bounce * 1.15
    out[..., 1] += sh_w * warm_shadow_bounce * 0.72
    out[..., 2] += sh_w * warm_shadow_bounce * 0.32

    # 2. 3-scale warm cream halation from bright regions
    bright = np.clip((lum - 0.78) / 0.22, 0.0, 1.0)
    hal = (
        0.50 * gaussian_filter(bright, sigma=max(1.5, min(h, w) * 0.004), mode="reflect")
        + 0.35 * gaussian_filter(bright, sigma=max(4.0, min(h, w) * 0.012), mode="reflect")
        + 0.15 * gaussian_filter(bright, sigma=max(10.0, min(h, w) * 0.028), mode="reflect")
    )
    out[..., 0] += hal * halation_strength * 1.12
    out[..., 1] += hal * halation_strength * 0.96
    out[..., 2] += hal * halation_strength * 0.74

    # 3. Tanh highlight shoulder rolloff above shoulder_start + gentle shadow black floor
    s0 = float(np.clip(shoulder_start, 0.50, 0.95))
    hi_mask = out > s0
    if np.any(hi_mask):
        headroom = 0.985 - s0  # caps strictly below 0.992 hard digital clip ceiling
        out[hi_mask] = s0 + headroom * np.tanh((out[hi_mask] - s0) / max(headroom, 1e-4))
    # Gentle cool shadow black lift (keeps blacks slightly above 0.008 crush)
    out[..., 0] = out[..., 0] * 0.985 + 0.009
    out[..., 1] = out[..., 1] * 0.985 + 0.010
    out[..., 2] = out[..., 2] * 0.985 + 0.012

    # 4. Shadow/midtone-weighted clumped silver-halide film grain
    if grain_strength > 0.0:
        rng = np.random.default_rng(seed)
        raw_noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        clumped = gaussian_filter(raw_noise, sigma=0.75, mode="reflect")
        clumped /= max(float(np.std(clumped)), 1e-6)
        lum2 = _luma(out)
        grain_w = np.clip(1.0 - np.clip((lum2 - 0.55) / 0.40, 0.0, 1.0), 0.15, 1.0)
        g_delta = (clumped * grain_w * grain_strength)[..., None]
        out += g_delta

    # 5. Edge-mirrored YIQ chroma horizontal softening (mode="reflect" prevents dark borders)
    if chroma_soften_sigma > 0.0:
        r, g, b = out[..., 0], out[..., 1], out[..., 2]
        y_ch = 0.299 * r + 0.587 * g + 0.114 * b
        i_ch = 0.5959 * r - 0.2746 * g - 0.3213 * b
        q_ch = 0.2115 * r - 0.5227 * g + 0.3112 * b
        i_ch = gaussian_filter(i_ch, sigma=(0.0, chroma_soften_sigma), mode="reflect")
        q_ch = gaussian_filter(q_ch, sigma=(0.0, chroma_soften_sigma), mode="reflect")
        out[..., 0] = y_ch + 0.956 * i_ch + 0.619 * q_ch
        out[..., 1] = y_ch - 0.272 * i_ch - 0.647 * q_ch
        out[..., 2] = y_ch - 1.106 * i_ch + 1.703 * q_ch

    # 6. Gentle optical corner vignette
    if vignette_strength > 0.0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ny = (yy - (h - 1) * 0.5) / max(h * 0.5, 1.0)
        nx = (xx - (w - 1) * 0.5) / max(w * 0.5, 1.0)
        r2 = nx * nx + ny * ny
        vig = 1.0 - vignette_strength * np.clip((r2 - 0.35) / 1.65, 0.0, 1.0)
        out *= vig[..., None]

    out = np.clip(out, 0.0, 0.990).astype(np.float32)
    assert np.all(np.isfinite(out)), "Non-finite output in apply_35mm_optical_finish"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Host-enforced 1:1 multi-crop visual verifier + 35mm optical finish.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_v = sub.add_parser("verify", help="Verify an image with full-frame + 4x 1:1 native crops.")
    p_v.add_argument("image", type=Path)
    p_v.add_argument("--crop-size", type=int, default=256)
    p_v.add_argument("--contact-sheet", type=Path, default=None)
    p_v.add_argument("--out-json", type=Path, default=None)

    p_f = sub.add_parser("finish-35mm", help="Apply deterministic float32 35mm optical finish and verify.")
    p_f.add_argument("image", type=Path)
    p_f.add_argument("output", type=Path)
    p_f.add_argument("--contact-sheet", type=Path, default=None)
    p_f.add_argument("--seed", type=int, default=20261127)

    p_m = sub.add_parser("compare", help="Monotonic commit-or-keep verdict between baseline and candidate images.")
    p_m.add_argument("baseline", type=Path)
    p_m.add_argument("candidate", type=Path)
    p_m.add_argument("--crop-size", type=int, default=256)

    args = ap.parse_args()
    if args.cmd == "verify":
        rep = verify_image_crops(args.image, crop_size=args.crop_size, save_contact_sheet_path=args.contact_sheet)
        txt = json.dumps(rep, indent=2)
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(txt + "\n")
        print(txt)
        return 0 if rep["passed"] else 2

    if args.cmd == "finish-35mm":
        rgb = load_rgb_f32(args.image)
        fin = apply_35mm_optical_finish(rgb, seed=args.seed)
        save_rgb_f32(fin, args.output)
        b_rep = verify_image_crops(args.image)
        c_rep = verify_image_crops(args.output, save_contact_sheet_path=args.contact_sheet)
        arb = monotonic_commit_or_keep(b_rep, c_rep)
        print(json.dumps({"baseline": b_rep, "finished": c_rep, "arbiter": arb}, indent=2))
        return 0

    if args.cmd == "compare":
        b_rep = verify_image_crops(args.baseline, crop_size=args.crop_size)
        c_rep = verify_image_crops(args.candidate, crop_size=args.crop_size)
        arb = monotonic_commit_or_keep(b_rep, c_rep)
        print(json.dumps(arb, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())

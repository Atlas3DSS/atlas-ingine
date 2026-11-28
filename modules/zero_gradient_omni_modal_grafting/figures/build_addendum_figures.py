#!/usr/bin/env python3
"""Build plain-English, human-readable editorial figures for the two Omni Paper Addendums:
  1. fig_addendum_cross_family_scaling.{png,svg}
  2. fig_addendum_bio_robotics_manifolds.{png,svg}

Reads verified JSON artifacts directly for biological/robotics/Atlas numbers; fails loudly if any source is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

FIG_DIR = Path("/mnt/storage/atlas-ingine-omni-paper-912/modules/zero_gradient_omni_modal_grafting/figures")
FLY_PROBE_JSON = Path("/mnt/storage/fly_connectome_49k_probe/out/fly_connectome_49k_platos_probe_report.json")
FLY_CMP_JSON = Path("/mnt/storage/fly_connectome_49k_probe/out/fly_vs_atlas_omni_comparison.json")
H2_SPEC_JSON = Path("/mnt/storage/fly_connectome_49k_probe/out/unitree_h2_wbc_spectral_report.json")

BG = "#0d1117"
CARD = "#161b22"
CARD_ALT = "#1c2330"
GRID = "#30363d"
TEXT = "#e6edf3"
MUTED = "#9ba7b4"
GREEN = "#3fb950"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"
ORANGE = "#f0883e"
RED = "#f85149"
GOLD = "#e3b341"
CYAN = "#39c5cf"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID,
    "axes.linewidth": 1.0,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
})


def build_cross_family_figure() -> None:
    fig = plt.figure(figsize=(15.5, 8.4), dpi=180, facecolor=BG)

    # Header
    fig.text(
        0.04, 0.945,
        "Combining Different AI Brains Keeps Cutting Error — If You Weight Them Properly",
        fontsize=19, fontweight="bold", color=TEXT, ha="left", va="top",
    )
    fig.text(
        0.04, 0.903,
        "Zero gradient training, zero trunk weight edits: closed-form Ridge bridges from N = 1 to 5 distinct LLM families into MiniMax-H3 (160 held-out val_dev prompts)",
        fontsize=11.5, color=MUTED, ha="left", va="top",
    )

    # Layout: Left main chart + Right split (weights top, skepticism checklist bottom)
    ax_left = fig.add_axes([0.065, 0.14, 0.50, 0.69], facecolor=CARD)
    ax_wt = fig.add_axes([0.625, 0.47, 0.34, 0.36], facecolor=CARD)
    ax_box = fig.add_axes([0.625, 0.14, 0.34, 0.27], facecolor=CARD)

    # ---- LEFT PANEL: N=1..5 curve vs naive concat failures ----
    ns = np.array([1, 2, 3, 4, 5])
    mk_cos = np.array([0.95917, 0.96159, 0.96247, 0.96419, 0.96854])
    single_cos = np.array([0.95917, 0.91455, 0.84470, 0.93426, np.nan])
    # Verified naive failure points from ADDENDUM_CROSS_FAMILY_SCALING.md Section 1
    naive_ns = np.array([1, 2])
    naive_raw = np.array([0.95919, 0.95621])
    rms_ns = np.array([1, 2, 3])
    rms_cos = np.array([0.95919, 0.96033, 0.95739])
    eq_ns = np.array([1, 2, 3])
    eq_cos = np.array([0.95919, 0.9535, 0.94846])  # Note: only plot verified endpoints 1 and 3!

    # Plot verified lines only (zero interpolated fake intermediate points!)
    ax_left.plot(ns, mk_cos, color=GREEN, lw=3.2, marker="o", markersize=9,
                 markerfacecolor=GREEN, markeredgecolor=BG, markeredgewidth=1.5,
                 label="Weighted Multi-Kernel Ridge (monotonic win)", zorder=5)
    ax_left.fill_between(ns, 0.944, mk_cos, color=GREEN, alpha=0.08, zorder=1)

    ax_left.plot(rms_ns, rms_cos, color=ORANGE, lw=2.0, ls="--", marker="s", markersize=6.5,
                 label="Naive RMS-norm concat (regresses at N=3)", zorder=4)
    ax_left.plot(naive_ns, naive_raw, color=RED, lw=2.0, ls=":", marker="^", markersize=7,
                 label="Naive raw concat (drops immediately at N=2)", zorder=4)
    ax_left.plot([1, 3], [0.95919, 0.94846], color="#ff7b72", lw=2.0, ls="-.", marker="x", markersize=8,
                 label="Equal-weight average (collapses to 0.9485)", zorder=4)

    # Annotate each N=1..5 step clearly
    step_labels = [
        ("1. Qwen3.8\n(0.9592)", (0.08, 0.0018), "left"),
        ("+2. Inkling\n(0.9616)", (0.0, 0.0018), "center"),
        ("+3. DeepSeek-V4.1\n(0.9625)", (0.0, 0.0019), "center"),
        ("+4. GLM-5.3\n(0.9642)", (-0.05, 0.0019), "center"),
        ("+5. Ling-3.2\n0.9685 (-14.1% err!)", (-0.18, 0.0016), "right"),
    ]
    for x, y, (lbl, (dx, dy), ha) in zip(ns, mk_cos, step_labels):
        fw = "bold" if x == 5 else "normal"
        col = GREEN if x == 5 else TEXT
        ax_left.text(x + dx, y + dy, lbl, fontsize=9.5, fontweight=fw, color=col, ha=ha, va="bottom")

    # Annotate confirmed N=2,3 sealed points
    ax_left.annotate(
        "Sealed Confirm-160 verified:\nN=2: 0.9626 | N=3: 0.9632\n(N=4,5 sealed 0/160 untouched)",
        xy=(3, 0.96247), xytext=(3.15, 0.9522),
        fontsize=8.8, color=CYAN, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.45", facecolor=CARD_ALT, edgecolor=CYAN, lw=1.0),
        arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.2, connectionstyle="arc3,rad=-0.15"),
    )

    # Plain-English takeaway callout inside left plot (clear of equal-weight line!)
    ax_left.text(
        0.84, 0.9456,
        "Why naive merging breaks:\n"
        "Dumping a noisier brain (DeepSeek\n"
        "solo = 0.845) straight into the matrix\n"
        "floods stronger brains. Per-family\n"
        "weights turn it into extra signal.",
        fontsize=8.6, color=TEXT, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#201a16", edgecolor=ORANGE, lw=1.0),
    )

    ax_left.set_title("A. Accuracy vs. Number of Independent AI Families", fontsize=12.5, fontweight="bold", color=TEXT, pad=12, loc="left")
    ax_left.set_xlabel("Number of Distinct Frozen LLM Families Combined (N)", fontsize=10.5, color=TEXT, labelpad=8)
    ax_left.set_ylabel("Held-Out Alignment to MiniMax-H3 (val_dev Cosine)", fontsize=10.5, color=TEXT, labelpad=8)
    ax_left.set_xticks(ns)
    ax_left.set_xticklabels(["N = 1\nQwen3.8", "N = 2\n+Inkling", "N = 3\n+DS-V4.1", "N = 4\n+GLM-5.3", "N = 5\n+Ling-3.2"], fontsize=9.3)
    ax_left.set_xlim(0.75, 5.35)
    ax_left.set_ylim(0.9445, 0.9730)
    ax_left.grid(True, color=GRID, alpha=0.45, linestyle="--", linewidth=0.7)
    ax_left.legend(loc="upper left", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.6, labelcolor=TEXT)

    # ---- TOP RIGHT PANEL: Optimal Family Weights at N=5 ----
    fams = ["Qwen3.8-27B\n(solo 0.959)", "Ling-3.2-air\n(new family)", "GLM-5.3-Flash\n(solo 0.934)", "Inkling-2-Mini\n(solo 0.915)", "DeepSeek-V4.1\n(solo 0.845)"]
    wts = [47.5, 21.6, 13.9, 11.8, 5.1]
    cols = [BLUE, GREEN, CYAN, PURPLE, ORANGE]
    y_pos = np.arange(len(fams))[::-1]

    bars = ax_wt.barh(y_pos, wts, height=0.62, color=cols, edgecolor=BG, linewidth=1.0)
    for bar, w in zip(bars, wts):
        ax_wt.text(w + 1.2, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                   va="center", ha="left", fontsize=10, fontweight="bold", color=TEXT)

    ax_wt.set_title("B. Optimal Brain Mix at N = 5 (Every Family Contributes)", fontsize=11.5, fontweight="bold", color=TEXT, pad=10, loc="left")
    ax_wt.set_yticks(y_pos)
    ax_wt.set_yticklabels(fams, fontsize=8.8)
    ax_wt.set_xlim(0, 60)
    ax_wt.set_xlabel("Learned Multi-Kernel Weight (%)", fontsize=9.5, color=MUTED)
    ax_wt.grid(True, axis="x", color=GRID, alpha=0.45, linestyle="--", linewidth=0.7)
    ax_wt.text(
        26.5, 0.15,
        "Even the weakest solo\nmodel keeps 5.1% weight\nbecause it sees angles\nthe others miss.",
        fontsize=8.3, color=GOLD, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.38", facecolor=CARD_ALT, edgecolor=GOLD, lw=0.9),
    )

    # ---- BOTTOM RIGHT PANEL: 6 Skepticism Kill Tests ----
    ax_box.set_xticks([])
    ax_box.set_yticks([])
    for spine in ax_box.spines.values():
        spine.set_edgecolor(PURPLE)
        spine.set_linewidth(1.2)

    ax_box.text(0.04, 0.92, "C. 6 Preregistered Kill Tests (Trying to Disprove Ourselves)",
                fontsize=10.5, fontweight="bold", color=PURPLE, ha="left", va="top", transform=ax_box.transAxes)

    kills = [
        ("F1  Dimension Bloat:", "Does 16-layer Qwen alone match 5 families?"),
        ("F2  Lucky Order:", "Test all 31 subsets + ladder without Qwen"),
        ("F3  Bagging Artifact:", "Real families vs. perturbed clones & shuffles"),
        ("F4  One-Target Fluke:", "Replicate on FLUX.2, Speech (TTS), & Music"),
        ("F5  Pixel Saturation:", "8-domain OOD + sealed test + real renders"),
        ("F6  Early Plateau:", "Extend to N = 6–8 (GigaChat4 next!)"),
    ]
    y_k = 0.76
    for tag, desc in kills:
        ax_box.text(0.04, y_k, tag, fontsize=8.7, fontweight="bold", color=TEXT, ha="left", va="top", transform=ax_box.transAxes)
        ax_box.text(0.38, y_k, desc, fontsize=8.5, color=MUTED, ha="left", va="top", transform=ax_box.transAxes)
        y_k -= 0.123

    # Footer
    fig.text(
        0.04, 0.032,
        "Source: ADDENDUM_CROSS_FAMILY_SCALING.md (Atlas 3DSS, 2026-11-27) · N=5 multi-kernel val_dev cos=0.96854, R²=0.999423, rel_l2=0.02316 (p=3.05e-5 vs Qwen3.8 across 16/16 groups). Sealed confirm_reserve untouched (0/160) for N=4,5.",
        fontsize=8.3, color=MUTED, ha="left", va="bottom",
    )

    png_p = FIG_DIR / "fig_addendum_cross_family_scaling.png"
    svg_p = FIG_DIR / "fig_addendum_cross_family_scaling.svg"
    fig.savefig(png_p, dpi=180, facecolor=BG)
    fig.savefig(svg_p, facecolor=BG)
    plt.close(fig)
    print(f"WROTE {png_p} ({png_p.stat().st_size} B) and {svg_p}")


def build_bio_robotics_figure() -> None:
    fly_probe = json.loads(FLY_PROBE_JSON.read_text(encoding="utf-8"))
    fly_cmp = json.loads(FLY_CMP_JSON.read_text(encoding="utf-8"))
    h2_spec = json.loads(H2_SPEC_JSON.read_text(encoding="utf-8"))

    # Read exact verified values directly from JSON (zero hardcoding drift!)
    fly_r_r2 = fly_probe["test1_bilateral_left_to_right_platos_key"]["plato_ridge_fp64"]["regression"]["r2"]
    fly_o_r2 = fly_probe["test1_bilateral_left_to_right_platos_key"]["orthogonal_procrustes"]["regression"]["r2"]
    fly_ret = 100.0 * fly_o_r2 / fly_r_r2

    rig = fly_cmp["procrustes_vs_ridge_rigidity"]
    llm_ret = 100.0 * rig["atlas_inkling_to_qwen38_cross_brain"]["procrustes_rigid_r2"] / rig["atlas_inkling_to_qwen38_cross_brain"]["ridge_r2"]
    ink_h3_ret = 100.0 * rig["atlas_inkling_brain_to_h3_eyes"]["procrustes_rigid_r2"] / rig["atlas_inkling_brain_to_h3_eyes"]["ridge_r2"]
    qw_h3_ret = 100.0 * rig["atlas_qwen38_brain_to_h3_eyes"]["procrustes_rigid_r2"] / rig["atlas_qwen38_brain_to_h3_eyes"]["ridge_r2"]

    h2_by_name = {l["name"]: l for l in h2_spec["linear_layers"]}
    ks = fly_cmp["platos_key_spectra_comparison"]
    fm = fly_cmp["fly_bilateral_by_modality"]

    fig = plt.figure(figsize=(16.0, 9.0), dpi=180, facecolor=BG)

    fig.text(
        0.035, 0.952,
        "Can You Just Rotate One Brain Into Another?  Biology vs. Robots vs. Atlas AI",
        fontsize=19, fontweight="bold", color=TEXT, ha="left", va="top",
    )
    fig.text(
        0.035, 0.912,
        "Measured on a 49,393-neuron Fruit Fly connectome, a Unitree H2 Humanoid Robot RL controller, and Atlas Omni's frozen Plato's Keys ($0 GPU)",
        fontsize=11.3, color=MUTED, ha="left", va="top",
    )

    ax_l = fig.add_axes([0.145, 0.345, 0.345, 0.495], facecolor=CARD)
    ax_r = fig.add_axes([0.655, 0.145, 0.320, 0.695], facecolor=CARD)
    ax_note = fig.add_axes([0.035, 0.115, 0.455, 0.145], facecolor="#201a16")

    # ---- LEFT PANEL: Pure Rotation R² Retention ----
    rot_labels = [
        "Fruit Fly\nLeft ↔ Right\n(1,580 types)",
        "AI Brain ↔ Brain\nInkling → Qwen3.8\n(160 prompts)",
        "Brain → Video\nQwen3.8 → H3\n(160 prompts)",
        "Brain → Video\nInkling → H3\n(160 prompts)",
    ]
    rot_vals = [fly_ret, llm_ret, qw_h3_ret, ink_h3_ret]
    rot_cols = [GREEN, BLUE, RED, RED]
    rot_tags = [
        f"{fly_ret:.1f}%  Pure Mirror Rotation\n(100% Top-1 on motor neurons)",
        f"{llm_ret:.1f}%  Mostly Rigid Rotation",
        f"{qw_h3_ret:.1f}%  Rotation Crashes!\nNeeds per-axis stretching",
        f"{ink_h3_ret:.1f}%  Rotation Crashes!\nNeeds per-axis stretching",
    ]
    yl = np.arange(len(rot_labels))[::-1]
    bars_l = ax_l.barh(yl, rot_vals, height=0.56, color=rot_cols, edgecolor=BG, linewidth=1.2)
    for bar, v, tag in zip(bars_l, rot_vals, rot_tags):
        if v > 65:
            ax_l.text(v - 2.5, bar.get_y() + bar.get_height() / 2, tag,
                      va="center", ha="right", fontsize=8.4, fontweight="bold", color=BG)
        else:
            ax_l.text(v + 2.2, bar.get_y() + bar.get_height() / 2, tag,
                      va="center", ha="left", fontsize=8.4, fontweight="bold", color=TEXT)

    ax_l.set_title("A. How Much Signal Survives Pure Rotation?\n(Orthogonal Procrustes R² ÷ Full Ridge R²)", fontsize=11.0, fontweight="bold", color=TEXT, pad=10, loc="left")
    ax_l.set_yticks(yl)
    ax_l.set_yticklabels(rot_labels, fontsize=8.6)
    ax_l.set_xlim(0, 108)
    ax_l.set_xlabel("Centered Variance Preserved Without Stretching (%)", fontsize=9.2, color=MUTED, labelpad=7)
    ax_l.axvline(80, color=MUTED, ls=":", lw=1.0, alpha=0.5)
    ax_l.grid(True, axis="x", color=GRID, alpha=0.45, linestyle="--", linewidth=0.7)

    # Bottom-left warning box: The Uncentered Cosine Trap + Honest Nulls
    ax_note.set_xticks([])
    ax_note.set_yticks([])
    for spine in ax_note.spines.values():
        spine.set_edgecolor(GOLD)
        spine.set_linewidth(1.1)
    ax_note.text(
        0.025, 0.87,
        "WHY COSINE ALONE CAN FOOL YOU  +  HONEST NULLS",
        fontsize=8.8, fontweight="bold", color=GOLD, ha="left", va="top", transform=ax_note.transAxes,
    )
    ax_note.text(
        0.025, 0.62,
        "• On H3 Video, rigid rotation still scores 0.995 uncentered cosine because all H3 vectors\n"
        "  share a huge baseline offset — even while 82% of real variation collapses (R²: 0.92 → 0.16)!\n"
        "• Clean nulls: static Fly graph gave +1e-6 (zero gain) on H3; soma 3D R² = +0.13.",
        fontsize=8.2, color=TEXT, ha="left", va="top", transform=ax_note.transAxes, linespacing=1.35,
    )

    # ---- RIGHT PANEL: Signal Bandwidth Hourglass (PR, log scale) ----
    items = [
        ("DeepSeek → H3 Video Eyes", ks["ds41_to_h3_single"]["participation_ratio"], PURPLE),
        ("Qwen3.8 → H3 Video Eyes", ks["qwen38_to_h3_single"]["participation_ratio"], PURPLE),
        ("Ling-3.2 → H3 Video Eyes", ks["ling32_to_h3_single"]["participation_ratio"], PURPLE),
        ("Fly Sensory Input", fm["cb_sensory"]["feature_space_PR"], GOLD),
        ("Humanoid H2: Sensor In (W0)", h2_by_name["_tensor_constant19"]["participation_ratio"], GOLD),
        ("Humanoid H2: 14-Joint Out (W3)", h2_by_name["_tensor_constant25"]["participation_ratio"], GOLD),
        ("Fly Visual Projection", fm["visual_projection"]["feature_space_PR"], ORANGE),
        ("Humanoid H2: Hidden (W1)", h2_by_name["_tensor_constant21"]["participation_ratio"], GREEN),
        ("Fly Motor Output (Desc.)", fm["descending_neuron"]["feature_space_PR"], GREEN),
        ("Humanoid H2: Pre-Motor (W2)", h2_by_name["_tensor_constant23"]["participation_ratio"], GREEN),
        ("DeepSeek → Qwen3-TTS Speech", ks["ds41_to_mouth_tts"]["participation_ratio"], GREEN),
        ("Qwen3.8 → Qwen3-TTS Speech", ks["qwen38_to_mouth_tts"]["participation_ratio"], GREEN),
        ("Fly Central Brain (Intrinsic)", fm["cb_intrinsic"]["feature_space_PR"], CYAN),
        ("Fly Left ↔ Right Key", ks["fly_left_to_right_bio"]["participation_ratio"], CYAN),
        ("Qwen3.8 → YuE2 Music", ks["qwen38_to_yue2_music"]["participation_ratio"], BLUE),
    ]
    yr = np.arange(len(items))[::-1]
    r_labels = [x[0] for x in items]
    r_vals = [x[1] for x in items]
    r_cols = [x[2] for x in items]

    # Highlight the shared Motor & Speech sweet spot band (PR 35..72)
    ax_r.axvspan(35, 72, color=GREEN, alpha=0.14, zorder=0)
    ax_r.text(50.5, len(items) - 0.10, "Motor & Speech\n(~38–66 ch)", fontsize=8.0,
              fontweight="bold", color=GREEN, ha="center", va="top",
              bbox=dict(boxstyle="round,pad=0.25", facecolor=BG, edgecolor=GREEN, lw=0.8))

    bars_r = ax_r.barh(yr, r_vals, height=0.64, color=r_cols, edgecolor=BG, linewidth=0.9, zorder=3)
    for bar, v in zip(bars_r, r_vals):
        ax_r.text(v * 1.13, bar.get_y() + bar.get_height() / 2, f"{v:.1f}",
                  va="center", ha="left", fontsize=8.5, fontweight="bold", color=TEXT)

    ax_r.set_xscale("log")
    ax_r.set_xlim(0.8, 1600)
    ax_r.set_ylim(-0.7, len(items) + 0.85)
    ax_r.set_yticks(yr)
    ax_r.set_yticklabels(r_labels, fontsize=8.4)
    ax_r.set_title("B. Active Channel Width (Participation Ratio, Log Scale)",
                   fontsize=10.8, fontweight="bold", color=TEXT, pad=10, loc="left")
    ax_r.set_xlabel("Effective Channel Width (PR — Bigger = Wider, Flatter Signal)", fontsize=9.0, color=MUTED, labelpad=7)
    ax_r.grid(True, axis="x", which="major", color=GRID, alpha=0.45, linestyle="--", linewidth=0.7)

    # Legend in empty upper-right quadrant (zero bar overlap!)
    legend_handles = [
        patches.Patch(color=PURPLE, label="AI Video Eyes (1–4)"),
        patches.Patch(color=GOLD, label="Sensor / Joint (6–9)"),
        patches.Patch(color=GREEN, label="Bio + Robot + TTS (38–66)"),
        patches.Patch(color=BLUE, label="Brain & Music (118–417)"),
    ]
    ax_r.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.985, 0.84), frameon=True,
                facecolor=BG, edgecolor=GRID, fontsize=7.8, labelcolor=TEXT)

    fig.text(
        0.035, 0.030,
        "Source: ADDENDUM_BIOLOGICAL_AND_ROBOTICS_MANIFOLDS.md (Atlas 3DSS, 2026-11-27; att-89ea692a) · fly-connectome-49k (CC-BY-4.0), nvidia/agile_wbc_unitree_h2, Atlas frozen .safetensors keys. Spectral regime match ≠ shared semantic neuron axes.",
        fontsize=8.2, color=MUTED, ha="left", va="bottom",
    )

    png_p = FIG_DIR / "fig_addendum_bio_robotics_manifolds.png"
    svg_p = FIG_DIR / "fig_addendum_bio_robotics_manifolds.svg"
    fig.savefig(png_p, dpi=180, facecolor=BG)
    fig.savefig(svg_p, facecolor=BG)
    plt.close(fig)
    print(f"WROTE {png_p} ({png_p.stat().st_size} B) and {svg_p}")


if __name__ == "__main__":
    build_cross_family_figure()
    build_bio_robotics_figure()

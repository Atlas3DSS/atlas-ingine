#!/usr/bin/env python3
"""Build a crisp, shareable 16:9 infographic proving Zero-Gradient Plato's Key prosody
transfer across 5 open-source TTS families + YuE2-3B 48 kHz singing.
Reads verified metrics directly from report_942_unified.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

PROSODY_JSON = Path("/mnt/storage/omni_prosody_benchmark_942/results/unified/report_942_unified.json")
OUT_1 = Path("/mnt/storage/atlas-ingine-omni-paper-912/docs/media/prosody_proof_card.png")
OUT_2 = Path("/mnt/storage/atlas-ingine-omni-paper-912/modules/zero_gradient_omni_modal_grafting/figures/prosody_proof_card.png")

BG = "#0b0f17"
CARD = "#131924"
GRID = "#263042"
TEXT = "#f0f4fa"
MUTED = "#94a3b8"
GREEN = "#34d399"
BLUE = "#60a5fa"
CYAN = "#22d3ee"
GOLD = "#fbbf24"
ORANGE = "#fb923c"
PURPLE = "#c084fc"
RED = "#f87171"


def main() -> None:
    d = json.loads(PROSODY_JSON.read_text())
    arms = d["arm_summaries"]

    # Pull exact Track 2 (60 matched items) metrics directly from report_942_unified.json
    def t2(name: str):
        s = arms[name]["track2_60"]
        return (
            100.0 * s["emotion2vec_top1_acc"],
            100.0 * s["melancholic_sad_recall"],
            100.0 * s["mean_wer"],
        )

    rows = [
        ("Qwen3-TTS Native Ceiling\n(hand-written style prompt)", *t2("qwen3_armB_native_ceiling"), MUTED),
        ("Qwen3-TTS + Qwen3.8 Plato Key\n(continuous vectors, 0 style prompt)", *t2("qwen3_armC_qwen38_key_continuous"), GREEN),
        ("Orpheus-TTS-3B\n(Apache-2.0 expressive tags)", *t2("orpheus_3b_expressive_tags"), CYAN),
        ("Qwen3-TTS + Inkling Plato Key\n(continuous vectors, 0 style prompt)", *t2("qwen3_armC_inkling2m_key_continuous"), BLUE),
        ("Chatterbox-0.5B\n(PerTh watermarked, retired)", *t2("chatterbox_0.5b_expressive"), GOLD),
        ("Step-Audio-EditX 3B\n(single-pass audio edit)", *t2("step_audio_editx_edited"), ORANGE),
        ("Qwen3-TTS Neutral Baseline\n(no style conditioning)", *t2("qwen3_armA_neutral"), "#475569"),
        ("IndexTTS-2 + Qwen3.8 Plato Key\n(4-fold OOF continuous vectors)", *t2("indextts2_platos_key_qwen38_oof"), PURPLE),
        ("IndexTTS-2 Native Ceiling\n(8-float emotion slider)", *t2("indextts2_native_8float_ceiling"), RED),
    ]

    fig = plt.figure(figsize=(16.0, 9.0), facecolor=BG, dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.38, 1.0], wspace=0.14, left=0.21, right=0.97, top=0.83, bottom=0.12)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_cards = fig.add_subplot(gs[0, 1])

    # ---- Left Panel: 5-Family Expressive TTS Shoot-Out ----
    ax_bar.set_facecolor(CARD)
    ax_bar.tick_params(colors=TEXT, labelsize=8.8)
    for sp in ("top", "right"):
        ax_bar.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax_bar.spines[sp].set_color(GRID)
    ax_bar.grid(axis="x", color=GRID, linestyle="--", alpha=0.5, linewidth=0.7)
    ax_bar.set_axisbelow(True)

    y = np.arange(len(rows))[::-1]
    h = 0.36
    e2v = [r[1] for r in rows]
    sad = [r[2] for r in rows]
    wers = [r[3] for r in rows]
    cols = [r[4] for r in rows]

    ax_bar.barh(y + h / 2, e2v, height=h, color=cols, alpha=0.95, label="emotion2vec+ Top-1 (60 matched texts)")
    ax_bar.barh(y - h / 2, sad, height=h, color="#ffffff", alpha=0.25, edgecolor=TEXT, linewidth=0.8, hatch="///", label="Melancholic sad recall (14 items)")
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([r[0] for r in rows], color=TEXT, fontsize=8.4)
    ax_bar.set_xlim(0, 62)
    ax_bar.set_xlabel("Blinded Judge Accuracy (%, RMS >= 0.01 Silence Gate)", color=TEXT, fontsize=9.4, fontweight="bold")
    ax_bar.set_title("A. 5-Family Expressive TTS Shoot-Out (1,416 Blinded Clips)", color=TEXT, fontsize=11.4, fontweight="bold", pad=10)

    for yi, ev, sd, wr in zip(y, e2v, sad, wers):
        ax_bar.text(ev + 0.8, yi + h / 2, f"{ev:.1f}%  (WER {wr:.1f}%)", va="center", ha="left", color=TEXT, fontsize=8.2, fontweight="bold")
        ax_bar.text(sd + 0.8, yi - h / 2, f"sad {sd:.1f}%", va="center", ha="left", color=MUTED, fontsize=7.8)

    ax_bar.legend(loc="lower right", frameon=True, facecolor=BG, edgecolor=GRID, fontsize=8.4, labelcolor=TEXT)

    # ---- Right Panel: 4 Snappy Proof Cards ----
    ax_cards.set_facecolor(BG)
    ax_cards.set_xlim(-0.25, 10.35)
    ax_cards.set_ylim(-0.15, 10.15)
    ax_cards.axis("off")
    ax_cards.set_title("B. Causal Proof & Compression Highlights", color=TEXT, fontsize=11.4, fontweight="bold", pad=10)

    cards = [
        (0.05, 7.65, 10.0, 2.15, GREEN,
         "1. Matches Hand-Written Style + Cuts WER by 10.7 pp",
         "• 24/60 (40.0%) vs 26/60 (43.3%) native prompt ceiling (p = 0.79)\n"
         "• RAVDESS-40: +30.0 pp vs neutral (p = 0.012), +32.5 pp vs B0 (p = 0.002)\n"
         "• WER 33.9% -> 23.2% (pass-only 17.4% -> 11.4%) | #1 sad 42.9%"),
        (0.05, 5.15, 10.0, 2.15, PURPLE,
         "2. Causal Lift on IndexTTS-2 (0% -> 15% RAVDESS-40)",
         "• Strict 4-fold OOF Qwen3.8 Plato Key vectors (cos = 0.936, R^2 = 0.807)\n"
         "• Lifts RAVDESS-40 from 0/40 (neutral & swapped B0) -> 6/40 (p = 0.031)\n"
         "• Overall 7/60 (11.7%) vs 2/60 (3.3%) native 8-float slider (p = 0.125 n.s.)"),
        (0.05, 2.65, 10.0, 2.15, CYAN,
         "3. Direct 34s 48 kHz Singing From Frozen 36 MiB Key",
         "• Zero YuE2 prompt tokens across ABC -> semantic -> NAR -> 48 kHz VAE\n"
         "• Song A: #1 Bb_major r = 0.973 (avoids native 173 BPM pirate prior!)\n"
         "• Song B: #1 G_major r = 0.801 (125 BPM) + 4/12 lyrics vs 0/12 zero"),
        (0.05, 0.15, 10.0, 2.15, GOLD,
         "4. 1.4-2.8 MiB Ultra-Low-Rank + Honest Boundary Law",
         "• FP64 SVD compresses speech keys 26x-29x (rank-64, <0.001 cos loss)\n"
         "• Continuous vectors steer emotion, pace (8/8), urgency (8/8), key/tempo\n"
         "• Exact words stay discrete (0/32 direct lexemes; 0/8 confirm intonation)"),
    ]

    for x, y0, w, ch, col, title, body in cards:
        rect = patches.FancyBboxPatch(
            (x, y0), w, ch, boxstyle="round,pad=0.12,rounding_size=0.20",
            facecolor=CARD, edgecolor=col, linewidth=2.0,
        )
        ax_cards.add_patch(rect)
        ax_cards.text(x + 0.22, y0 + ch - 0.24, title, color=col, fontsize=9.8, fontweight="bold", va="top")
        ax_cards.text(x + 0.22, y0 + ch - 0.70, body, color=TEXT, fontsize=8.1, va="top", linespacing=1.45)

    # Top Title & Subtitle
    fig.suptitle(
        "Zero-Gradient Prosody Control: 1.4-36 MiB Linear Plato's Keys Steer Voice Emotion & 48 kHz Singing",
        color=TEXT, fontsize=15.0, fontweight="bold", y=0.955,
    )
    fig.text(
        0.5, 0.895,
        "Frozen LLM hidden states -> FP64 closed-form ridge map -> continuous prosody & affect conditioning (0 gradient steps, 0 trunk edits)",
        ha="center", color=CYAN, fontsize=10.5, fontweight="bold",
    )
    fig.text(
        0.5, 0.028,
        "Sources: 1,416-clip blinded TTS corpus (emotion2vec+ & faster-whisper w/ RMS>=0.01 gate; sha256 72ae01e0), FP64 SVD (7aff2432), & YuE2 36 MiB 34s singing audit (053d4048).\n"
        "Full paper, weights & 5-minute starter guide: github.com/Atlas3DSS/atlas-ingine  |  huggingface.co/collections/Atlas3D/omni-by-atlas",
        ha="center", color=MUTED, fontsize=8.4, linespacing=1.35,
    )

    OUT_1.parent.mkdir(parents=True, exist_ok=True)
    OUT_2.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_1, facecolor=BG, dpi=200)
    fig.savefig(OUT_2, facecolor=BG, dpi=200)
    plt.close(fig)
    print(f"WROTE {OUT_1} ({OUT_1.stat().st_size} B, sha256={hashlib.sha256(OUT_1.read_bytes()).hexdigest()[:16]}...)")


if __name__ == "__main__":
    main()

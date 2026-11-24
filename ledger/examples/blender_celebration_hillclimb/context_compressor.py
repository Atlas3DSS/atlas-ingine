"""3-Tier Rolling Context Compressor for Multimodal Blender Video Hillclimbing.

Prevents context bloat across 15-30+ hillclimb iterations while preserving 100% of
the high-leverage visual, structural, and ledger signal:

  - TIER 1 (Pinned): Celebration creative brief + Atlas editorial palette/style guide
    + compact active `celebration_video_ledger` (facts, active winners, active repairs,
    and recent retractions).
  - TIER 2 (Compressed Rolling History): Every older iteration compressed deterministically
    into a ~120-180 token structured delta record (`iter`, `host_status`, `scores`,
    `delta_vs_best`, `visual_verdict`, `ledger_adds`/`subtracts`, `kept`/`dropped`).
    No raw 4K-token historical scripts or old duplicate tracebacks!
  - TIER 3 (Live Verbatim Window): Current Reigning Champion Blender script in full +
    exact latest host telemetry + verbatim multi-timestamp rendered Cycles keyframes
    fed directly into `atlas-ingine`'s native GPU vision tower (plus latest failure diff
    and targeted repair playbooks if the immediately preceding iteration failed).
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from typing import Any

from .ledger import CelebrationVideoLedger


CELEBRATION_CREATIVE_BRIEF = """\
# FLAGSHIP CELEBRATION VIDEO BRIEF
Theme: **Celebrating 10 Years of NVIDIA AI Architecture (2016 Pascal -> 2026 Blackwell), Thinking Machines Lab's Inkling-2-Mini, and Open-Source + Human–AI ("Monkey Centaur") Collaboration**

## Visual Narrative & Choreography (10.0s @ 24fps, frames 1..240, 16:9 widescreen)
Craft an awe-inspiring, museum-grade editorial 3D monument and sweeping camera journey celebrating a decade of AI silicon breakthroughs culminating in Blackwell, Inkling-2-Mini, and the open-source community building together:
1. **Six Architectural Silicon Monoliths (2016 -> 2026)** arranged along a curved ceremonial obsidian gallery / reflective dark-slate stage (`z = 0`, flat floor plane >= 14m x 14m):
   - **2016 Pascal** -> **2017 Volta** -> **2018 Turing** -> **2020 Ampere** -> **2022 Hopper** -> **2026 Blackwell**: progressively taller, denser, finely beveled precision-machined architectural slabs/dies with metallic heat-spreader chamfers, luminous emerald-green (`#82C232`) silicon bus traces, and subtle glass/dielectric optical layers — zero deep AABB interpenetration, every plinth grounded cleanly on the dark studio stage.
2. **Hero Centerpiece — Inkling-2-Mini Looped-Recurrence Core & Open-Source Constellation** at the focal climax:
   - A luminous sculptural blue-steel (`#556D79`), crystal glass, and emerald-green (`#82C232`) multi-ring looped compute core with subtle crisp white (`#FFFFFF`) inner luminance (representing 3-pass looped sparse MoE recurrence and human+AI collaboration) hovering cleanly above a precision beveled pedestal, connected via delicate luminous emerald light-bridges/data arcs from the Blackwell monolith, surrounded by an active **Geometry Nodes** (`NODES`) orbital constellation of precision micro-prisms/tokens symbolizing the open-source ecosystem building together.
3. **Editorial 3D Milestones & Quiet Credit Discipline:**
   - Crisp, tasteful 3D architectural year markers (`2016` .. `2026`) grounded along the monolith gallery using `LiberationSans-Bold.ttf` (`/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf`). Keep the 3D frame uncluttered and sculptural — full editorial ALL-CAPS chapter titles (`//` technical tags, thick `#82C232` accent bars) & open-source tribute credits (NVIDIA, Thinking Machines Lab, satellitedown/ingine + fngine/cinference/NInfer, ggml-org/llama.cpp, realugbun, mw00/project-maya, turboderp/exllamav3, professorpalmer/Kintsugi) are composited cleanly in Liberation Sans during final 1080p finishing.
   - **QUIET CREDIT RULE:** Zero monetization, pure celebration of open source & human–AI ("monkey centaur") collaboration. Never redraw logos; do NOT shout Atlas's name in the 3D scene (small quiet footer credit only in end card).
4. **Atlas Editorial Palette, Materials, Lighting & Camera (Brand Guide p.19 exact):**
   - Direction: *"Cooler feels, nice contrast, always a hint of green on blue steel."*
   - Palette: Dark charcoal ground `#101518` (subtle micro-roughness reflections), Blue Steel `#556D79` brushed metallic monolith bodies, Dark Grey `#686868` structural bevels/plinths, Primary Atlas/NVIDIA Green `#82C232` kick lights & silicon bus traces, Light Grey `#E0E0E0` & White `#FFFFFF` crisp architectural highlights.
   - Materials: Principled BSDF with linked procedural `ShaderNodeTexNoise` micro-roughness/anisotropy & transmissive crystal glass.
   - Lighting & Post: >= 4 architectural studio lights (cool-white overhead key `(.86, .93, 1.0)`, signature `#82C232` green kick/rim lights grazing blue-steel monolith edges, cool blue-steel fill, soft core backlight), `AgX` view transform, and `CompositorNodeGlare` (`FOG_GLOW`) actively LINKED along `Render Layers -> Glare -> Composite`.
   - Camera & Motion: Sweeping smooth **BEZIER** crane/dolly/orbit across frames `1..240` (24 fps, 10.0s) moving from the Pascal-to-Hopper gallery arc into the glowing Blackwell + Inkling-2-Mini centerpiece climax (`>= 2.0m` smooth camera motion + living orbital core animation), keeping `>= 75%` of hero vertices inside frame with balanced `15%–82%` screen coverage across all 5 sampled timestamps (`f12, f60, f120, f180, f228`).
   - **Blender 4.5.14 LTS Contract:** Always start with `bpy.ops.wm.read_factory_settings(use_empty=True)`. Do NOT call `bpy.ops.render.render()` or `sys.exit()` — the host verifier saves, inspects, and renders your scene cleanly.
"""


@dataclass
class IterationDelta:
    """Compact ~120-180 token summary of one completed hillclimb iteration for Tier 2."""
    iter_num: int
    host_ok: bool
    host_score: float
    vision_score: float
    composite_score: float
    delta_vs_best: float
    is_new_best: bool
    visual_verdict: str
    next_focus: str = ""
    kept_elements: list[str] = field(default_factory=list)
    dropped_elements: list[str] = field(default_factory=list)
    ledger_added: list[str] = field(default_factory=list)
    ledger_subtracted: list[str] = field(default_factory=list)
    failure_note: str = ""

    def format_compact(self) -> str:
        status = "NEW_BEST" if self.is_new_best else ("PASS" if self.host_ok else "FAIL")
        parts = [
            f"- Iter #{self.iter_num} [{status}] comp={self.composite_score:.1f}/100 "
            f"(host={self.host_score:.1f}/50, vision={self.vision_score:.1f}/50, vs_best={self.delta_vs_best:+.1f})"
        ]
        if self.visual_verdict:
            parts.append(f"verdict: {self.visual_verdict.strip()[:220]}")
        if self.failure_note:
            parts.append(f"fail: {self.failure_note.strip()[:180]}")
        if self.kept_elements:
            parts.append("kept: " + ", ".join(x.strip()[:48] for x in self.kept_elements[:4]))
        if self.dropped_elements:
            parts.append("dropped: " + ", ".join(x.strip()[:48] for x in self.dropped_elements[:4]))
        if self.ledger_added or self.ledger_subtracted:
            parts.append(
                f"ledger: +{len(self.ledger_added)}({', '.join(self.ledger_added[:3])}) "
                f"-{len(self.ledger_subtracted)}({', '.join(self.ledger_subtracted[:3])})"
            )
        return " | ".join(parts)


def encode_png_part(path: str) -> dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


class RollingContextCompressor:
    """Builds exact 3-tier multimodal OpenAI chat payloads for the hillclimb loop."""

    def __init__(
        self,
        *,
        creative_brief: str = CELEBRATION_CREATIVE_BRIEF,
        max_tier1_ledger_chars: int = 6500,
        max_tier2_deltas: int = 12,
        max_tier2_chars: int = 4800,
        max_tier3_script_chars: int = 18000,
    ) -> None:
        self.creative_brief = creative_brief.strip()
        self.max_tier1_ledger_chars = max_tier1_ledger_chars
        self.max_tier2_deltas = max_tier2_deltas
        self.max_tier2_chars = max_tier2_chars
        self.max_tier3_script_chars = max_tier3_script_chars

    def build_tier1_system(self, ledger: CelebrationVideoLedger) -> str:
        ledger_txt = ledger.render_tier1_ledger(max_chars=self.max_tier1_ledger_chars)
        return (
            "You are a world-class Blender 4.5.14 LTS Python (`bpy`) technical director, cinematographer, "
            "and procedural 3D artist.\n"
            "Output a single complete, self-contained executable Python script inside ONE ```python ... ``` fenced block.\n\n"
            + self.creative_brief
            + "\n\n"
            + ledger_txt
        )

    def build_tier2_history(self, deltas: list[IterationDelta]) -> str:
        if not deltas:
            return "[ROLLING_HISTORY (Tier 2): Iteration 1 — establish the full 6-monolith Pascal->Blackwell gallery, glowing Inkling-2-Mini core, Geometry Nodes field, 3D milestones, lighting, glare, and sweeping 10s Bezier camera.]"

        lines: list[str] = ["[ROLLING_HISTORY (Tier 2 — compressed structured deltas)]"]
        older = deltas[:-self.max_tier2_deltas] if len(deltas) > self.max_tier2_deltas else []
        recent = deltas[-self.max_tier2_deltas:]

        if older:
            best_old = max(older, key=lambda d: d.composite_score)
            pass_cnt = sum(1 for d in older if d.host_ok)
            lines.append(
                f"- Earlier epoch (iters 1..{older[-1].iter_num}): {pass_cnt}/{len(older)} host-pass, "
                f"peak composite={best_old.composite_score:.1f}/100 at iter #{best_old.iter_num} "
                f"({best_old.visual_verdict[:120]})."
            )

        for d in recent:
            lines.append(d.format_compact())

        out = "\n".join(lines)
        if len(out) > self.max_tier2_chars:
            # Keep epoch summary + newest deltas within char cap
            kept = [lines[0]]
            used = len(lines[0]) + 1
            for ln in reversed(lines[1:]):
                if used + len(ln) + 1 <= self.max_tier2_chars:
                    kept.insert(1, ln)
                    used += len(ln) + 1
            out = "\n".join(kept)
        return out

    def build_author_messages(
        self,
        *,
        iter_num: int,
        total_iters: int,
        ledger: CelebrationVideoLedger,
        deltas: list[IterationDelta],
        best_script: str | None,
        best_score: float,
        best_iter: int | None,
        best_telemetry: str,
        best_render_paths: list[str],
        last_iter_failed: bool = False,
        last_script: str = "",
        last_failure_telemetry: str = "",
        last_repair_block: str = "",
        last_render_paths: list[str] | None = None,
        next_focus: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Assemble the 3-tier multimodal prompt for authoring iteration `iter_num`."""
        tier1_sys = self.build_tier1_system(ledger)
        tier2_hist = self.build_tier2_history(deltas)

        # Tier 3: Live Verbatim Window (champion script + latest keyframes + exact telemetry)
        t3_text_blocks: list[str] = [
            f"## ITERATION {iter_num} OF {total_iters} — LIVE WORKING WINDOW (Tier 3)",
            tier2_hist,
        ]
        if next_focus:
            t3_text_blocks.append(f"### HIGHEST-LEVERAGE VISUAL TARGET FOR ITERATION #{iter_num}\n{next_focus.strip()}")

        if best_script:
            trimmed_script = best_script.strip()
            if len(trimmed_script) > self.max_tier3_script_chars:
                trimmed_script = trimmed_script[: self.max_tier3_script_chars] + "\n# ...(script truncated for window)"
            t3_text_blocks.append(
                f"### CURRENT REIGNING CHAMPION SCRIPT (Iter #{best_iter}, composite={best_score:.1f}/100)\n"
                f"Host telemetry of champion:\n{best_telemetry.strip()}\n\n"
                f"Build UPWARD from this working champion — keep every verified strength, elevate visual craft, "
                f"lighting, architectural detail, typography legibility, and camera choreography:\n"
                f"```python\n{trimmed_script}\n```"
            )

        if last_iter_failed and last_failure_telemetry:
            fail_sec = f"### PREVIOUS ATTEMPT FAILED — PATCH THE ROOT CAUSE DIRECTLY\n{last_failure_telemetry.strip()}"
            if last_repair_block:
                fail_sec += "\n\n" + last_repair_block.strip()
            if last_script:
                trimmed_fail = last_script.strip()
                if len(trimmed_fail) > self.max_tier3_script_chars:
                    trimmed_fail = trimmed_fail[: self.max_tier3_script_chars] + "\n# ...(script truncated)"
                fail_sec += (
                    "\n\n### FAILED SCRIPT TO PATCH (fix the exact error above directly without losing your scene):\n"
                    f"```python\n{trimmed_fail}\n```"
                )
            t3_text_blocks.append(fail_sec)

        # Attach verbatim multi-timestamp Cycles keyframes into native GPU vision
        vision_paths = (last_render_paths if (last_iter_failed and last_render_paths) else best_render_paths) or []
        image_parts: list[dict[str, Any]] = []
        for rp in vision_paths[:5]:
            part = encode_png_part(rp)
            if part is not None:
                image_parts.append(part)

        user_parts: list[dict[str, Any]] = []
        if image_parts:
            src_label = "most recent attempt (which needs fixing)" if (last_iter_failed and last_render_paths) else f"current champion (Iter #{best_iter})"
            user_parts.append(
                {
                    "type": "text",
                    "text": (
                        f"[HOST_VISION (Tier 3): Attached are {len(image_parts)} Cycles-rendered multi-timestamp "
                        f"keyframes (early -> mid -> climax -> end) from the {src_label}. Inspect silhouette clarity, "
                        "lighting drama, monolith progression, Inkling-2-Mini core glow, typography readability, "
                        "framing, reflections, and motion progression directly.]"
                    ),
                }
            )
            user_parts.extend(image_parts)

        t3_text_blocks.append(
            "Output the complete, self-contained upgraded Blender 4.5.14 LTS Python script in ONE ```python ... ``` block."
        )
        user_parts.append({"type": "text", "text": "\n\n".join(t3_text_blocks)})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": tier1_sys},
            {"role": "user", "content": user_parts if image_parts else user_parts[0]["text"]},
        ]
        metrics = {
            "tier1_chars": len(tier1_sys),
            "tier2_chars": len(tier2_hist),
            "tier3_text_chars": sum(len(p["text"]) for p in user_parts if p["type"] == "text"),
            "tier3_vision_images": len(image_parts),
        }
        return messages, metrics

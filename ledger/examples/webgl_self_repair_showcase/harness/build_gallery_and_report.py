#!/usr/bin/env python3
"""Generate before/after comparison cards, Verified Ledger JSON dump, interactive index.html gallery, and SHOWCASE_REPORT.md."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_media import make_before_after_card  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def dump_ledger(db_path: Path, out_json: Path) -> dict:
    conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    data = {
        "meta": [dict(r) for r in conn.execute("SELECT * FROM meta")],
        "facts": [dict(r) for r in conn.execute("SELECT fact_key, category, value, env_version, status FROM facts")],
        "winners": [
            {k: r[k] for k in ("task_id", "task_text", "score", "summary", "tags_json", "env_version", "status")}
            for r in conn.execute("SELECT * FROM winners")
        ],
        "repairs": [
            {k: r[k] for k in ("error_signature", "rule", "hits", "env_version", "status")}
            for r in conn.execute("SELECT * FROM repairs")
        ],
        "failure_log_count": conn.execute("SELECT COUNT(*) FROM failure_log").fetchone()[0],
        "failures_by_sig": [
            dict(r)
            for r in conn.execute(
                "SELECT task_id, error_signature, COUNT(*) AS n FROM failure_log GROUP BY task_id, error_signature ORDER BY task_id, n DESC"
            )
        ],
    }
    conn.close()
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def make_text_placeholder_png(out_png: Path, heading: str, body_lines: list[str]) -> None:
    img = Image.new("RGB", (1280, 720), (16, 20, 28))
    draw = ImageDraw.Draw(img)
    try:
        f_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        f_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 20)
    except Exception:
        f_big = ImageFont.load_default()
        f_med = ImageFont.load_default()
    draw.rectangle([40, 40, 1240, 680], outline=(248, 81, 73), width=4)
    draw.text((80, 90), heading, fill=(248, 81, 73), font=f_big)
    y = 170
    for ln in body_lines:
        draw.text((80, y), ln, fill=(210, 220, 235), font=f_med)
        y += 42
    img.save(out_png, format="PNG")


def main() -> int:
    root = Path("/home/orwel/dev_genius/experiments/kernals/results/loktar_showcase_848")
    s1 = json.loads((root / "01_gaussian_splats/run_summary.json").read_text())
    s2 = json.loads((root / "02_articulated_creature/run_summary.json").read_text())
    s3 = json.loads((root / "03_atari_arcade/run_summary.json").read_text())

    # Dump official SQLite Verified Ledger
    ledger_dump = dump_ledger(root / "verified_ledger.sqlite", root / "verified_ledger_dump.json")

    # Always regenerate clean Attempt-0 text receipt for Creature (Attempt 0 hit TOKEN_TRUNCATION before HTML)
    c_att0 = root / "02_articulated_creature/attempt_00_stock.png"
    make_text_placeholder_png(
        c_att0,
        "ATTEMPT 0 (STOCK ONE-SHOT): FAILED — TOKEN_TRUNCATION",
        [
            "Model: Inkling-2-Mini (stock bare prompt, no Verified Ledger)",
            "Outcome: Exhausted max_tokens (16,384) inside reasoning before emitting HTML",
            "Rendered Output: NONE (0 bytes HTML, 0 frames)",
            "Next: Verified Ledger Pivot Controller stepped reasoning down -> JS_PAGE_ERROR",
            "      -> Pivot on repeated JS_PAGE_ERROR -> PASS (31 joints, 0 errors)!",
        ],
    )

    # Create Before/After comparison cards for all 3 targets
    make_before_after_card(
        before_png=root / "01_gaussian_splats/attempt_00_stock.png",
        after_png=root / "01_gaussian_splats/final_frame_orbit_90.png",
        out_png=root / "before_after_01_gaussian_splats.png",
        before_label="ATTEMPT 0 (Stock): Blank screen — GLSL mat2*mat3 compile error (5.0/100)",
        after_label="FINAL (Iter 6, Verified Ledger + Vision): 7,221 smooth 3DGS splats (95.4/100)",
        title="HERO TARGET 1: Real-Time WebGL2 3D Gaussian Splatting (Inkling-2-Mini Self-Repair)",
    )
    make_before_after_card(
        before_png=c_att0,
        after_png=root / "02_articulated_creature/final_frame_initial.png",
        out_png=root / "before_after_02_articulated_creature.png",
        before_label="ATTEMPT 0 (Stock): Truncation -> JS_PAGE_ERROR (0.0 -> 5.0/100)",
        after_label="FINAL (Iter 8, Verified Ledger + Vision): 31-joint animated dragon (85.2/100)",
        title="TARGET 2: Articulated 3D Celestial Dragon with Hierarchical Kinematics",
    )
    make_before_after_card(
        before_png=root / "03_atari_arcade/attempt_00_stock.png",
        after_png=root / "03_atari_arcade/final_frame_action_scored.png",
        out_png=root / "before_after_03_atari_arcade.png",
        before_label="ATTEMPT 0 (Stock): NO_SCORING_OR_COLLISIONS — 0 hits, 0 score (46.0/100)",
        after_label="FINAL (Iter 1, Playtest Verified): 12 hits, 1000 pts, full restart (94.2/100)",
        title="TARGET 3: Playable Atari Arcade Game (NEON GALACTICA) — Automated Playtest",
    )

    # Build interactive HTML gallery
    gallery_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Switchboard #848 — Inkling-2-Mini + Verified Ledger Showcase</title>
<style>
  :root {{ --bg:#090d16; --card:#111827; --border:#24324a; --accent:#58c4ff; --ok:#3fb950; --bad:#f85149; --text:#e6edf3; --muted:#9fb0c8; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; line-height:1.55; padding:32px; max-width:1440px; margin:0 auto; }}
  h1 {{ font-size:28px; letter-spacing:0.4px; margin-bottom:8px; color:#fff; }}
  h2 {{ font-size:20px; margin:28px 0 12px; color:var(--accent); border-bottom:1px solid var(--border); padding-bottom:6px; }}
  p.lead {{ color:var(--muted); margin-bottom:20px; max-width:960px; }}
  .pills {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:24px; }}
  .pill {{ background:#162033; border:1px solid var(--border); border-radius:999px; padding:5px 14px; font-size:13px; color:#cfe3ff; }}
  .pill.ok {{ border-color:rgba(63,185,80,0.45); color:#8be99c; }}
  .grid {{ display:grid; grid-template-columns:1fr; gap:26px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:22px; }}
  .card-head {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:14px; }}
  .card-head h3 {{ font-size:19px; color:#fff; }}
  .links {{ display:flex; flex-wrap:wrap; gap:10px; }}
  .btn {{ display:inline-block; padding:6px 12px; border-radius:8px; background:#1c2a42; border:1px solid #35507a; color:#dff0ff; text-decoration:none; font-size:13px; font-weight:600; }}
  .btn:hover {{ background:#263b5e; }}
  .btn.primary {{ background:#1d4ed8; border-color:#3b82f6; color:#fff; }}
  .media-row {{ display:grid; grid-template-columns:1.15fr 1fr; gap:16px; margin-top:12px; align-items:start; }}
  @media (max-width:980px) {{ .media-row {{ grid-template-columns:1fr; }} }}
  img, video {{ width:100%; border-radius:10px; border:1px solid var(--border); background:#000; display:block; }}
  table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:13px; }}
  th, td {{ border:1px solid var(--border); padding:8px 10px; text-align:left; vertical-align:top; }}
  th {{ background:#151f30; color:#b8d4f4; }}
  .ok-tag {{ color:var(--ok); font-weight:700; }}
  .bad-tag {{ color:var(--bad); font-weight:700; }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
</style>
</head>
<body>
  <h1>Inkling-2-Mini + Verified Ledger Self-Improvement Showcase (#848)</h1>
  <p class="lead">
    Every single line of HTML, WebGL2 GLSL ES 3.00, 3D kinematics, and Atari arcade game logic below was authored and self-repaired solely by
    <strong>Inkling-2-Mini</strong> served via <strong>atlas-ingine</strong> (<code>build-sm70-v3/ingine-serve --spec</code> + native vision tower on Quincy V100, $0.00 cloud spend),
    paired with the official SQLite <strong>VerifiedLedger</strong> (<code>atlas-ingine/ledger/verified_ledger.py</code>), an automated headless Chromium-1248 WebGL2 state-machine inspector &amp; Playwright playtester, and an explicit pivot / bounded-stopping controller.
  </p>
  <div class="pills">
    <span class="pill ok">3D Gaussian Splats: Attempt-0 FAIL (5.0) &rarr; Final PASS (95.4/100)</span>
    <span class="pill ok">Articulated 31-Joint Dragon: Attempt-0 FAIL (0.0) &rarr; Final PASS (85.2/100)</span>
    <span class="pill ok">Atari Arcade Playtest: Attempt-0 FAIL (46.0) &rarr; Final PASS (94.2/100)</span>
    <span class="pill">VerifiedLedger Active Winners: {len(ledger_dump['winners'])} | Linked Repair Rules: {len(ledger_dump['repairs'])} | Logged Failures: {ledger_dump['failure_log_count']}</span>
    <span class="pill">Compute Cost: $0.00 Vast.ai (Local Quincy V100)</span>
  </div>

  <div class="grid">
    <div class="card">
      <div class="card-head">
        <h3>1. HERO: Real-Time WebGL2 3D Gaussian Splatting (7,221 Instanced Splats)</h3>
        <div class="links">
          <a class="btn primary" href="01_gaussian_splats/final_verified.html" target="_blank">Launch Final Interactive 3DGS</a>
          <a class="btn" href="01_gaussian_splats/attempt_00_stock.html" target="_blank">View Broken Attempt-0</a>
          <a class="btn" href="01_gaussian_splats/run_summary.json" target="_blank">JSON Receipt</a>
        </div>
      </div>
      <p class="lead">
        Attempt-0 failed with a pitch-black blank screen caused by a GLSL ES 3.00 <code>mat2 * mat3</code> Jacobian compile error (exact <code>@loktar00</code> failure mode).
        Across 6 self-repair iterations using host WebGL2 telemetry (<code>uniformMatrix4fv</code> column-major layout check, <code>vertexAttribPointer</code> per-vertex vs instanced buffer check, pixel blowup check) + Inkling-2-Mini inspecting its own rendered frames via native vision, Inkling-2-Mini converged on <strong>7,221 smooth anisotropic 3D Gaussians</strong>, premultiplied alpha <code>blendFunc(ONE, ONE_MINUS_SRC_ALPHA)</code>, <code>depthMask(false)</code>, and real <code>~3.1 ms</code> back-to-front depth sorting.
      </p>
      <div class="media-row">
        <div>
          <img src="before_after_01_gaussian_splats.png" alt="3DGS Before vs After">
        </div>
        <div>
          <video controls loop muted playsinline src="01_gaussian_splats/shrine_diorama_3dgs_turntable.mp4"></video>
        </div>
      </div>
      <table>
        <thead><tr><th>Iter</th><th>Mode</th><th>Outcome</th><th>Score</th><th>Host Verifier / Ledger Signal</th></tr></thead>
        <tbody>
          {"".join(f"<tr><td class='mono'>{r['iter']}</td><td class='mono'>{r['mode']}</td><td class='{'ok-tag' if r['ok'] else 'bad-tag'}'>{'PASS' if r['ok'] else 'FAIL'}</td><td class='mono'>{r['score']}</td><td class='mono'>{(r['summary'] if r['ok'] else r['error_signature'] + ': ' + r['error_detail'])[:190]}</td></tr>" for r in s1['records'])}
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-head">
        <h3>2. Articulated 3D Celestial Dragon (31 Hierarchical Joints + Interactive Kinematics)</h3>
        <div class="links">
          <a class="btn primary" href="02_articulated_creature/final_verified.html" target="_blank">Launch Interactive Dragon</a>
          <a class="btn" href="02_articulated_creature/run_summary.json" target="_blank">JSON Receipt</a>
        </div>
      </div>
      <p class="lead">
        Demonstrates both articulated multi-limb 3D kinematics AND explicit <strong>Pivot Controller</strong> recovery (Loktar critique #4): after stock Attempt-0 truncated and two repairs repeated <code>JS_PAGE_ERROR</code>, the Verified Ledger controller triggered an explicit pivot in <code>iter_04</code>, producing a 31-joint hierarchical dragon (spine, neck, head, jaw, 4 multi-segment legs, 4-segment tail with crystal orb, translucent flapping wings) passing at <code>92.8 &rarr; 93.0</code>, followed by a vision-driven hero-framing repair (<code>top_edge_fg = 0.0</code>, full head-to-tail view).
      </p>
      <div class="media-row">
        <div>
          <img src="before_after_02_articulated_creature.png" alt="Creature Before vs After">
        </div>
        <div>
          <video controls loop muted playsinline src="02_articulated_creature/celestial_dragon_showcase.mp4"></video>
        </div>
      </div>
      <table>
        <thead><tr><th>Iter</th><th>Mode</th><th>Outcome</th><th>Score</th><th>Host Verifier / Ledger Signal</th></tr></thead>
        <tbody>
          {"".join(f"<tr><td class='mono'>{r['iter']}</td><td class='mono'>{r['mode']}</td><td class='{'ok-tag' if r['ok'] else 'bad-tag'}'>{'PASS' if r['ok'] else 'FAIL'}</td><td class='mono'>{r['score']}</td><td class='mono'>{(r['summary'] if r['ok'] else r['error_signature'] + ': ' + r['error_detail'])[:190]}</td></tr>" for r in s2['records'])}
        </tbody>
      </table>
    </div>

    <div class="card">
      <div class="card-head">
        <h3>3. Playable Atari-Style Arcade Game (NEON GALACTICA) — Automated Playwright Playtest</h3>
        <div class="links">
          <a class="btn primary" href="03_atari_arcade/final_verified.html" target="_blank">Launch Playable Game</a>
          <a class="btn" href="03_atari_arcade/attempt_00_stock.html" target="_blank">View Broken Attempt-0</a>
          <a class="btn" href="03_atari_arcade/run_summary.json" target="_blank">JSON Receipt</a>
        </div>
      </div>
      <p class="lead">
        Directly answers Loktar's critique of stock Inkling failing to test/tune its own Atari game: Attempt-0 rendered a nice title/starfield and moved the ship, but automated Playwright playtesting proved <code>NO_SCORING_OR_COLLISIONS</code> (0 hits, 0 score after 5.5s of aiming &amp; firing). In <strong>1 single Verified Ledger repair iteration</strong>, Inkling-2-Mini fixed hitboxes &amp; firing and passed every live playtest assertion (<code>ArrowRight=+190.6px</code>, <code>ArrowLeft=-256.6px</code>, <code>12</code> verified kills, score <code>1000</code>, lives <code>3&rarr;0&rarr;GAME OVER&rarr;clean restart</code>, <code>94.2/100</code>), triggering bounded early stop.
      </p>
      <div class="media-row">
        <div>
          <img src="before_after_03_atari_arcade.png" alt="Atari Before vs After">
        </div>
        <div>
          <video controls loop muted playsinline src="03_atari_arcade/neon_galactica_playtest.mp4"></video>
        </div>
      </div>
      <table>
        <thead><tr><th>Iter</th><th>Mode</th><th>Outcome</th><th>Score</th><th>Host Verifier / Ledger Signal</th></tr></thead>
        <tbody>
          {"".join(f"<tr><td class='mono'>{r['iter']}</td><td class='mono'>{r['mode']}</td><td class='{'ok-tag' if r['ok'] else 'bad-tag'}'>{'PASS' if r['ok'] else 'FAIL'}</td><td class='mono'>{r['score']}</td><td class='mono'>{(r['summary'] if r['ok'] else r['error_signature'] + ': ' + r['error_detail'])[:190]}</td></tr>" for r in s3['records'])}
        </tbody>
      </table>
    </div>
  </div>

  <h2>Official Verified Ledger State (<code>verified_ledger.sqlite</code>)</h2>
  <p class="lead">
    Every error signature -&gt; repair rule below was auto-linked only after the host Playwright WebGL2/playtest oracle confirmed a transition from FAIL to PASS. Full SQLite export: <a class="btn" href="verified_ledger_dump.json">verified_ledger_dump.json</a>
  </p>
  <table>
    <thead><tr><th>Error Signature</th><th>Verified Repair Rule Linked in Bank 3</th><th>Hits</th></tr></thead>
    <tbody>
      {"".join(f"<tr><td class='mono'>{r['error_signature']}</td><td class='mono'>{r['rule']}</td><td class='mono'>{r['hits']}</td></tr>" for r in ledger_dump['repairs'])}
    </tbody>
  </table>
</body>
</html>
"""
    (root / "index.html").write_text(gallery_html, encoding="utf-8")

    # Build concise factual SHOWCASE_REPORT.md
    report_md = f"""# Switchboard #848 — Inkling-2-Mini + Verified Ledger Showcase Report

## Summary

All three targets addressing `@loktar00`'s four critiques of stock Inkling-2-Mini were **100% authored and self-repaired by Inkling-2-Mini itself** (zero hand-written target HTML/JS/GLSL) via `atlas-ingine` (`/mnt/nvme/v100-knapcio-bench/build-sm70-v3/apps/serve/ingine-serve --spec`, native vision tower enabled) on local Quincy V100 GPUs (**$0.00** Vast.ai spend, GPUs 0–3 kept clear for `v100-port-agent`, zero Devbox PRO 6000 impact).

The self-improvement loop paired the official 4-bank SQLite **Verified Ledger** (`/home/orwel/dev_genius/engines/atlas-ingine/ledger/verified_ledger.py`) with a headless Chromium-1248 Playwright WebGL2 state-machine inspector, pixel/motion/playtest oracle, native Inkling-2-Mini vision self-review, and an explicit pivot + bounded-stopping controller.

## Before / After Results

| Target (Loktar Critique Addressed) | Attempt 0 (Stock One-Shot) | Verified Ledger + Vision Final | Iterations / Stop Reason | Verified Artifacts |
|---|---|---|---|---|
| **1. HERO: WebGL2 3D Gaussian Splatting** (Low-level GPU plumbing: instanced quads, 3D&rarr;2D covariance/conic projection, premultiplied alpha, back-to-front sorting) | **FAIL (`5.0/100`)** — Pitch-black blank screen (`JS_PAGE_ERROR`: GLSL ES 3.00 `mat2 * mat3` Jacobian compile error) | **PASS (`95.4/100`)** — **7,221** smooth anisotropic instanced splats, `blendFunc(ONE, ONE_MINUS_SRC_ALPHA)`, `depthMask(false)`, `~3.1 ms` dynamic back-to-front sorting, 0 console/GL/VAO/matrix errors, `orbit_mad=0.087`, `899–972` quantized colors | 7 iters (`iter_00`..`iter_06`), `converged_verified_pass` | [final_verified.html](01_gaussian_splats/final_verified.html), [attempt_00_stock.html](01_gaussian_splats/attempt_00_stock.html), [shrine_diorama_3dgs_turntable.mp4](01_gaussian_splats/shrine_diorama_3dgs_turntable.mp4), [before_after_01_gaussian_splats.png](before_after_01_gaussian_splats.png) |
| **2. Articulated 3D Celestial Dragon** (Detailed movable/jointed creature + explicit pivot recovery) | **FAIL (`0.0/100`)** — `TOKEN_TRUNCATION` (exhausted 16,384 tokens before emitting HTML) | **PASS (`85.2/100` framed; `93.0/100` prior close-up)** — **31 hierarchical joints** (spine, neck, head, jaw, 4 multi-segment legs, 4-segment tail, translucent flapping wings), continuous kinematics (`motion_mad=0.046`), interactive modes (`Walk`/`Flap`/`Roar`/`Idle`), full-body framing (`top_edge_fg=0.0`), 0 errors | 9 iters (`iter_00`..`iter_08`; explicit pivot on repeated `JS_PAGE_ERROR` &rarr; pass on `iter_04`, vision framing pass on `iter_08`) | [final_verified.html](02_articulated_creature/final_verified.html), [celestial_dragon_showcase.mp4](02_articulated_creature/celestial_dragon_showcase.mp4), [before_after_02_articulated_creature.png](before_after_02_articulated_creature.png) |
| **3. Playable Atari Arcade (`NEON GALACTICA`)** (Testing/tuning own code + bounded early stop) | **FAIL (`46.0/100`)** — `NO_SCORING_OR_COLLISIONS` (ship moved & fired, 0 hits / 0 score in 5.5s live playtest) | **PASS (`94.2/100`)** — Verified live Playwright inputs (`ArrowRight=+190.6px`, `ArrowLeft=-256.6px`), `12` verified enemy collisions, score `1000`, real enemy-fire damage `lives 3->0 -> GAME OVER -> clean restart (score=0, lives=3)`, Web Audio SFX, 0 errors | 2 iters (`iter_00`..`iter_01`), `converged_verified_pass` (bounded early stop) | [final_verified.html](03_atari_arcade/final_verified.html), [attempt_00_stock.html](03_atari_arcade/attempt_00_stock.html), [neon_galactica_playtest.mp4](03_atari_arcade/neon_galactica_playtest.mp4), [before_after_03_atari_arcade.png](before_after_03_atari_arcade.png) |

## Exact Failure -> Self-Repair Trajectories

### 1. HERO 3D Gaussian Splatting (`01_gaussian_splats/ledger_trace.jsonl`)
- **`iter_00` (Stock One-Shot)**: `FAIL (5.0)` — `JS_PAGE_ERROR` (`ERROR: 0:42: '*' : wrong operand types - mat2 * mat3` in GLSL ES 3.00 Jacobian covariance projection) &rarr; blank screen.
- **`iter_01` (Repair 1)**: Fixed shader compile error (`6,216` splats sorting in `2.00 ms`, 0 GL errors); caught `SPLAT_SCALE_OR_PROJECTION_BLOWUP` (giant translucent streaks filling viewport).
- **`iter_02` (Repair 2)**: Runtime `uniformMatrix4fv` state inspector caught `TRANSPOSED_VIEW_MATRIX_LAYOUT` (row-major translation at `m[3], m[7], m[11]` passed to column-major WebGL).
- **`iter_03` (Repair 3)**: Fixed column-major view matrix &rarr; **PASS (`91.34/100`)** — Shrine Diorama visible & orbiting cleanly.
- **`iter_04` (Visual Polish)**: Upgraded scene to `7,221` splats (`92.71/100`); WebGL2 per-vertex state inspector + vision audit caught `WEBGL_PER_VERTEX_BUFFER_MISMATCH` (`aCorner` sharing `instBuf` instead of `cornerBuf`, rendering sharp chevron triangles instead of soft Gaussian ellipses).
- **`iter_05`–`iter_06` (Quad-Buffer Self-Repair)**: Fixed orthonormal `lookAt()`, `cornerBuf` binding, and `TRIANGLE_STRIP` winding &rarr; **PASS (`95.42/100`)** — buttery-smooth translucent 3D Gaussian splats across all orbit angles.

### 2. Articulated 3D Creature (`02_articulated_creature/ledger_trace.jsonl`)
- **`iter_00`–`iter_01`**: `TOKEN_TRUNCATION` &rarr; Pivot Controller reduced `reasoning_effort` so full HTML completed cleanly.
- **`iter_02`–`iter_03`**: Repeated `JS_PAGE_ERROR` (duplicate `const hw` / `const bw` declarations) &rarr; triggered explicit `[VERIFIED_LEDGER PIVOT TRIGGERED]`.
- **`iter_04`–`iter_05`**: Pivot resolved all JS errors &rarr; **PASS (`92.76` &rarr; `92.95/100`)** with 31 animated hierarchical joints.
- **`iter_06`–`iter_08`**: Framing audit caught `CAMERA_TOO_CLOSE_CROPPED`; native vision inspection guided Inkling-2-Mini to rotate to a 3/4 view and pull back the camera (`top_edge_fg=0.0`) &rarr; **PASS (`85.2/100`)**.

### 3. Playable Atari Arcade (`03_atari_arcade/ledger_trace.jsonl`)
- **`iter_00` (Stock One-Shot)**: `FAIL (46.0)` — `NO_SCORING_OR_COLLISIONS` during live Playwright aim-and-shoot playtest.
- **`iter_01` (Verified Ledger Repair)**: Fixed projectile/hitbox collision loop &rarr; **PASS (`94.2/100`)** on full live movement + scoring + damage + gameover + restart cycle; bounded early stop triggered immediately.

## Verified Ledger Database (`verified_ledger.sqlite`)
- **Active Host Facts (Bank 1)**: {len(ledger_dump['facts'])}
- **Admitted Verified Winners (Bank 2)**: {len(ledger_dump['winners'])}
- **Auto-Linked Error &rarr; Repair Rules (Bank 3)**: {len(ledger_dump['repairs'])} (`JS_PAGE_ERROR`, `NO_SCORING_OR_COLLISIONS`, `TRANSPOSED_VIEW_MATRIX_LAYOUT`, `CAMERA_TOO_CLOSE_CROPPED`, `WEBGL_PER_VERTEX_BUFFER_MISMATCH`)
- **Total Logged Failures**: {ledger_dump['failure_log_count']}
- Interactive Gallery: [index.html](index.html) | Full Ledger Export: [verified_ledger_dump.json](verified_ledger_dump.json)
"""
    (root / "SHOWCASE_REPORT.md").write_text(report_md, encoding="utf-8")
    print("Generated gallery, comparison cards, ledger dump, and SHOWCASE_REPORT.md cleanly!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

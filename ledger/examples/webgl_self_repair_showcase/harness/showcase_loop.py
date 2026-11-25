#!/usr/bin/env python3
"""Inkling-2-Mini + Official VerifiedLedger + Multimodal Headless WebGL2 Self-Improvement Loop.

Honest scientific protocol for Switchboard #848:
  - Attempt 0 (Stock One-Shot): Bare prompt to Inkling-2-Mini via atlas-ingine ingine-serve,
    NO VerifiedLedger context, NO verifier feedback, NO visual critique.
  - Iterations 1..K (Verified Ledger + Multimodal Visual/Runtime Self-Repair):
    * Uses official /home/orwel/dev_genius/engines/atlas-ingine/ledger/verified_ledger.py
    * Feeds verbatim host verifier telemetry (WebGL2 shader/program logs, gl.getError(),
      instancing/blend/sort counters, pixel coverage/entropy, live Playwright playtest state)
    * Feeds Inkling-2-Mini its OWN rendered screenshot via ingine-serve's native vision tower!
    * Explicit Pivot & Bounded-Stop Controller (Loktar critique #4):
        - Detects repeated error_signature (>=2) or token truncation and emits a structured PIVOT
        - Keeps best verified candidate (never regresses on polish passes)
        - Stops cleanly upon verified convergence or budget exhaustion (never loops blindly)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

ATLAS_INGINE_ROOT = Path("/home/orwel/dev_genius/engines/atlas-ingine")
if str(ATLAS_INGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ATLAS_INGINE_ROOT))

from ledger.verified_ledger import VerifiedLedger, VerifyResult  # noqa: E402
from PIL import Image  # noqa: E402
from verifier import verify_html  # noqa: E402


TASK_SPECS: dict[str, dict[str, Any]] = {
    "gaussian_splats": {
        "task_id": "loktar_01_gaussian_splats",
        "title": "Real-Time WebGL2 3D Gaussian Splatting Viewer & Multi-Object Scene",
        "prompt": """Build a single self-contained HTML file (100% offline, zero external CDN/script links, pure HTML + CSS + JavaScript + WebGL2 GLSL ES 3.00) implementing a real-time interactive 3D Gaussian Splatting viewer and rich procedural 3D scene.

Technical requirements (strict WebGL2 3DGS plumbing):
1. Pure WebGL2 (`canvas.getContext('webgl2')`) with `#version 300 es` vertex and fragment shaders.
2. Procedural 3D Gaussian Splat scene containing at least 6,000 anisotropic 3D Gaussians (positions, 3D scales, quaternion rotations, RGB colors, opacities) forming a rich, colorful, asymmetric multi-object 3D diorama (for example: a luminous floating coral/crystal island or Japanese shrine bonsai garden with a distinct red/vermilion torii gate or pagoda in front, emerald/gold foliage canopy, glowing turquoise crystal spire/waterfall on the side, stone pedestal, and floating warm lanterns) so front vs. back orbit views are visibly distinct and visually striking.
3. True 3D Gaussian Splatting math in the vertex & fragment shaders:
   - Compute 3D covariance matrix Sigma = R * S * S^T * R^T (or pass precomputed symmetric 3D covariance).
   - Project to 2D screen-space covariance via perspective Jacobian J and view matrix W: cov2D = J * W * Sigma * W^T * J^T + low-pass regularization (+0.3 on diagonal).
   - Compute eigenvalues/major-minor axes of cov2D to orient and size each instanced quad (`gl.drawArraysInstanced` or `gl.drawElementsInstanced`, 1 quad per splat, `instanceCount >= 6000`).
   - Evaluate exact 2D Gaussian conic falloff `exp(-0.5 * power)` in the fragment shader, discarding fragments with alpha < 1/255.
4. Correct WebGL2 blend & depth state:
   - `gl.enable(gl.BLEND)`
   - Premultiplied alpha blending: `gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA)` (output `vec4(color * alpha, alpha)` from fragment shader)
   - `gl.depthMask(false)` during splat rendering so translucent Gaussians composite cleanly without hard quad z-fighting.
5. Real dynamic back-to-front depth sorting:
   - Whenever the camera orbits or moves (or every frame), compute view-space depth for all splats, sort back-to-front (fast counting/radix sort or typed-array sort), and upload the sorted splat index/attribute buffer via `gl.bufferSubData` / `gl.bufferData` so nearer splats always properly occlude farther splats from every angle!
6. Smooth interactive orbit camera (mouse drag to orbit yaw/pitch, wheel to zoom) plus clean unobtrusive HUD overlay showing splat count, FPS, sort time (ms), auto-orbit toggle, and splat scale slider. Frame the scene cleanly centered in the 1280x720 viewport with a deep atmospheric dark backdrop.
7. Optional read-only `window.__SPLAT_DEBUG__` reporting `{splatCount, sortMs, yaw, pitch, distance}`.

Output ONLY the complete self-contained HTML file inside a single ```html ... ``` fenced code block.""",
    },
    "articulated_creature": {
        "task_id": "loktar_02_articulated_creature",
        "title": "Articulated 3D Winged Mechanical-Organic Dragon/Griffin with Hierarchical Kinematics",
        "prompt": """Build a single self-contained HTML file (100% offline, zero external CDN/script links, pure HTML + CSS + JavaScript + WebGL2 GLSL ES 3.00) rendering a detailed, movable, articulated 3D creature (a richly shaded biomechanical celestial dragon / winged fox-griffin standing and moving on a subtle lit stone/grid stage).

Technical requirements (strict articulated kinematics & WebGL2 craft):
1. Pure WebGL2 (`#version 300 es`) with proper 3D perspective camera, Phong/PBR-style directional + rim + hemisphere lighting, depth testing (`gl.enable(gl.DEPTH_TEST)`), and distinct multi-tone materials (e.g., burnished crimson/obsidian armor plates, glowing amber/cyan energy joints, warm gold trim, translucent/feathered wing membranes).
2. Genuine hierarchical articulated skeleton (`>= 14` named joints using parent-to-child 4x4 matrix multiplication chain — child transforms inherit parent joint transforms cleanly!):
   - Articulated multi-segment spine (pelvis -> midSpine -> chest)
   - Articulated neck (2 segments) -> sculpted head -> moving lower jaw + ear/horn crests
   - 4 multi-segment articulated legs (frontLeft, frontRight, backLeft, backRight), each with Hip/Shoulder -> Upper Limb -> Knee/Elbow -> Lower Limb -> Paw/Claw (all moving in coordinated phase-shifted gait kinematics!)
   - Multi-segment sweeping articulated tail (at least 4 chained tail joints with traveling sinusoidal wave!)
   - Pair of articulated multi-joint wings (wingL_shoulder -> wingL_elbow -> wingL_tip + wingR_shoulder -> wingR_elbow -> wingR_tip) with visible flapping kinematics!
3. Smooth continuous animation loop (`requestAnimationFrame`) that starts animating immediately on load with clearly visible limb stride, wing flap, spine breathing, head/jaw motion, and sweeping tail wave.
4. Interactive controls:
   - Mouse drag to orbit 3D camera around the creature, wheel to zoom.
   - Clean compact UI panel with animation mode buttons (`Walk / Trot`, `Wing Flap / Hover`, `Roar / Alert`, `Idle`) and speed/wingspan sliders that immediately change the live joint kinematics.
5. Expose read-only `window.__CREATURE_DEBUG__` updated every frame:
   `{ jointCount: <int >= 14>, gaitMode: <string>, time: <float>, joints: { spine: <rad>, neck: <rad>, jaw: <rad>, wingL1: <rad>, ... } }`.
6. Frame the full creature generously in the center of the 1280x720 viewport with rich studio lighting so every articulated limb, wing, and tail segment is unmistakable.

Output ONLY the complete self-contained HTML file inside a single ```html ... ``` fenced code block.""",
    },
    "atari_arcade": {
        "task_id": "loktar_03_atari_arcade",
        "title": "Playable Retro Atari-Style Space Arcade Game with Full Collision/Score/Lives/Restart Loop",
        "prompt": """Build a single self-contained HTML file (100% offline, zero external CDN/script links, pure HTML + CSS + JavaScript + HTML5 Canvas + Web Audio API) implementing a crisp, responsive, genuinely playable Atari-style retro arcade shooter ("NEON GALACTICA" / "ASTRO VANGUARD").

Technical requirements (strict playability & state machine):
1. Crisp retro Atari/arcade visual polish centered in 1280x720: deep starfield parallax backdrop, subtle optional CRT scanline aesthetic, crisp pixel/vector player starfighter, colorful multi-row animated alien/enemy formations (multiple enemy tiers with distinct shapes and point values), laser bolts, particle explosions, screen shake, and synthesized Web Audio API retro SFX (laser pew, explosion, hit, wave clear, game over — guarded safely against autoplay restrictions).
2. Clean, responsive live controls:
   - `ArrowLeft` / `ArrowRight` (and `A` / `D`) move player smoothly across the playfield at responsive speed (~380-480 px/s) clamped to playfield bounds.
   - `Space` fires player lasers rapidly (cooldown <= 180ms) upward toward enemies.
   - `Enter` or `Space` (or clicking the Start/Restart button) starts the game from the title screen and cleanly restarts from the `gameover` screen. `R` also restarts from `gameover`.
3. Authentic arcade gameplay mechanics:
   - Live HUD showing `SCORE`, `HIGH SCORE`, `LIVES` (start with 3 lives), and `WAVE`.
   - Real AABB/circle collision detection between player bullets and enemies (destroys enemy, spawns particles, increments `score` by 50–200 points, increments `collisions` counter).
   - Active enemy threat: enemies march/swoop and fire visible glowing enemy projectiles frequently downward toward the player! When an enemy projectile or enemy collides with the player, the player loses 1 life, triggers explosion/brief invulnerability (<= 0.6s so consecutive hits are fair and testable), and when `lives <= 0` transitions cleanly to `state = 'gameover'` showing final score and restart prompt.
   - Restarting resets `state = 'playing'`, `score = 0`, `lives = 3`, `wave = 1`, `collisions = 0`, clears old bullets, and respawns a fresh enemy formation and centered player.
4. Expose read-only `window.__GAME_STATE__` updated EVERY animation frame:
   `{ state: 'title'|'playing'|'gameover', score: <int>, lives: <int>, wave: <int>, playerX: <number>, playerY: <number>, collisions: <int>, enemies: [{x, y, hp}], enemyBullets: [{x, y}], playerBullets: [{x, y}] }`
   (Use exact canvas/playfield pixel coordinates for playerX, enemies[i].x, enemyBullets[i].x so an automated player can read exact alignment).

Output ONLY the complete self-contained HTML file inside a single ```html ... ``` fenced code block.""",
    },
}


class _SshTunnel:
    """Managed SSH local port forward to Quincy ingine-serve, torn down cleanly on exit."""

    def __init__(self, remote_host: str, remote_port: int, local_port: int) -> None:
        self.remote_host = remote_host
        self.remote_port = remote_port
        self.local_port = local_port
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> str:
        # Check if already reachable locally first
        base = f"http://127.0.0.1:{self.local_port}/v1"
        if self._is_up(base):
            return base
        cmd = [
            "ssh",
            "-N",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-L",
            f"127.0.0.1:{self.local_port}:127.0.0.1:{self.remote_port}",
            self.remote_host,
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(40):
            if self.proc.poll() is not None:
                err = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
                raise RuntimeError(f"SSH tunnel exited early: {err}")
            if self._is_up(base):
                return base
            time.sleep(0.25)
        raise RuntimeError(f"Timed out waiting for SSH tunnel to {self.remote_host}:{self.remote_port}")

    @staticmethod
    def _is_up(base_url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=2.0) as r:
                return r.status == 200
        except Exception:
            return False

    def __exit__(self, *args: Any) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def encode_image_b64(img_path: Path, max_w: int = 640, max_h: int = 360) -> str:
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_contact_sheet(image_paths: list[tuple[str, Path]], out_path: Path) -> Path:
    """Create a labeled horizontal strip of screenshots for Inkling-2-Mini visual self-review."""
    imgs = []
    for label, p in image_paths:
        if p.exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail((426, 240), Image.Resampling.LANCZOS)
            imgs.append((label, im))
    if not imgs:
        return out_path
    w = sum(im.width for _, im in imgs) + 8 * (len(imgs) - 1)
    h = max(im.height for _, im in imgs)
    canvas = Image.new("RGB", (w, h), (12, 14, 20))
    x = 0
    for _, im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + 8
    canvas.save(out_path, format="PNG")
    return out_path


def extract_html(text: str) -> Optional[str]:
    if not text:
        return None
    # Prefer fenced ```html ... ``` block
    m = re.search(r"```html\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        if "<html" in cand.lower() or "<!doctype" in cand.lower() or "<canvas" in cand.lower():
            return cand
    # Any fenced block containing HTML
    for m in re.finditer(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", text, re.DOTALL):
        cand = m.group(1).strip()
        if "<!doctype html" in cand.lower() or "<html" in cand.lower():
            return cand
    # Raw HTML fallback
    idx = text.lower().find("<!doctype html")
    if idx == -1:
        idx = text.lower().find("<html")
    if idx != -1:
        end = text.lower().rfind("</html>")
        if end != -1:
            return text[idx : end + 7].strip()
        return text[idx:].strip()
    return None


def call_inkling(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 16384,
    temperature: float = 0.6,
    top_p: float = 0.95,
    reasoning_effort: Any = 0.35,
    timeout_s: float = 480.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "reasoning_effort": reasoning_effort,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    choice = data["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    finish_reason = str(choice.get("finish_reason") or "")
    usage = data.get("usage") or {}
    return {
        "content": content,
        "reasoning_content": reasoning,
        "finish_reason": finish_reason,
        "usage": usage,
        "elapsed_s": round(elapsed, 2),
        "id": data.get("id", ""),
    }


def compact_verifier_feedback(report: dict[str, Any], target_kind: str) -> str:
    m = report.get("metrics", {})
    t = m.get("webgl_telemetry", {})
    img0 = m.get("initial_image", {})
    lines = [
        f"VERIFIER STATUS: {'PASS' if report['ok'] else 'FAIL'} | score={report['score']}/100",
    ]
    if not report["ok"]:
        lines.append(f"ERROR_SIGNATURE: {report['error_signature']}")
        lines.append(f"ROOT_CAUSE: {report['error_detail']}")
    else:
        lines.append(f"SUMMARY: {report['summary']}")

    if m.get("page_errors"):
        lines.append(f"PAGE_ERRORS: {m['page_errors'][:4]}")
    if m.get("console_errors"):
        lines.append(f"CONSOLE_ERRORS: {m['console_errors'][:4]}")
    if t.get("shaderCompileErrors"):
        lines.append(f"SHADER_COMPILE_ERRORS: {t['shaderCompileErrors'][:3]}")
    if t.get("programLinkErrors"):
        lines.append(f"PROGRAM_LINK_ERRORS: {t['programLinkErrors'][:3]}")
    if t.get("glErrors"):
        lines.append(f"GL_ERRORS: {t['glErrors'][:6]}")

    if t:
        lines.append(
            f"WEBGL_TELEMETRY: webgl2={t.get('webgl2')}, instancedCalls={t.get('instancedCalls')}, "
            f"maxInstanceCount={t.get('maxInstanceCount')}, drawCalls={t.get('drawCalls')}, "
            f"blend={t.get('blendEnabled')} {(t.get('blendFuncs') or [])[:2]}, depthMaskFalse={t.get('depthMaskFalseCalls')}, "
            f"rafCount={t.get('rafCount')}"
        )
    if img0:
        lines.append(
            f"PIXEL_STATS: non_bg={img0.get('non_bg_ratio')}, center_non_bg={img0.get('center_non_bg_ratio')}, "
            f"bbox_area={img0.get('bbox_area_ratio')}, distinct_colors={img0.get('distinct_colors_q16')}, "
            f"entropy={img0.get('shannon_entropy')}, near_white={img0.get('near_white_ratio')}"
        )
    if target_kind == "gaussian_splats" and "splat_checks" in m:
        sc = m["splat_checks"]
        lines.append(
            f"SPLAT_CHECKS: orbit_0_90_mad={sc['orbit_0_90']['mad']}, orbit_0_180_mad={sc['orbit_0_180']['mad']}, "
            f"orbit_buffer_uploads={sc['orbit_buffer_uploads']}, glsl300={sc['has_glsl300']}, "
            f"exp_falloff={sc['has_exp_falloff']}, covariance_math={sc['has_covariance_math']}"
        )
    elif target_kind == "articulated_creature" and "creature_checks" in m:
        cc = m["creature_checks"]
        lines.append(
            f"CREATURE_CHECKS: joint_count={cc['joint_count']}, joint_angles_changed={cc['joint_angles_changed']}, "
            f"max_motion_mad={cc['max_motion_mad']}, max_motion_changed={cc['max_motion_changed']}, "
            f"orbit_mad={cc['orbit_cmp']['mad']}, buttons_clicked={cc['buttons_clicked']}"
        )
    elif target_kind == "atari_arcade" and "arcade_checks" in m:
        ac = m["arcade_checks"]
        lines.append(
            f"ATARI_PLAYTEST: hook={ac['has_game_state_hook']}, state_playing={(ac.get('state_playing') or {}).get('state')}, "
            f"ArrowR={ac['delta_right']}px, ArrowL={ac['delta_left']}px, best_score={ac['best_score']} "
            f"(incremented={ac['score_incremented']}), collisions={ac['max_collisions']}, "
            f"lives={ac['initial_lives']}->{ac['min_lives_seen']}, gameover={ac['gameover_reached']}, restart_ok={ac['restart_ok']}"
        )
    return "\n".join(lines)


def seed_host_facts(ledger: VerifiedLedger) -> None:
    """Stamp genuine host-inspected runtime environment facts into Bank 1."""
    ledger.set_fact(
        "runtime_webgl2",
        "Headless Chromium-1248 WebGL2 (GLSL ES 3.00, ANGLE SwiftShader Vulkan 1.3.0, MAX_TEXTURE_SIZE=8192, EXT_color_buffer_float=true).",
        category="hardware",
    )
    ledger.set_fact(
        "offline_single_file",
        "Zero external network/CDN fetches allowed; all JS/CSS/WebGL2/WebAudio code must be 100% self-contained inside one HTML file.",
        category="runtime",
    )
    ledger.set_fact(
        "viewport_framing",
        "Target viewport is 1280x720; ensure canvas fills viewport cleanly and subject occupies central 35%-75% of frame with strong lighting contrast.",
        category="runtime",
    )


def run_target(
    target_kind: str,
    base_url: str,
    model: str,
    ledger_path: Path,
    work_dir: Path,
    max_iters: int = 5,
    polish_pass: bool = True,
) -> dict[str, Any]:
    spec = TASK_SPECS[target_kind]
    task_id = spec["task_id"]
    task_prompt = spec["prompt"]
    work_dir.mkdir(parents=True, exist_ok=True)
    trace_path = work_dir / "ledger_trace.jsonl"

    with VerifiedLedger(ledger_path, env_version="v1_chromium1248_webgl2", max_context_chars=3800) as ledger:
        ledger.conn.execute("PRAGMA busy_timeout = 30000;")
        seed_host_facts(ledger)

        records: list[dict[str, Any]] = []
        best_score = -1.0
        best_iter: Optional[int] = None
        best_html_path: Optional[Path] = None
        best_report: Optional[dict[str, Any]] = None
        error_sig_history: list[str] = []
        last_error_sig = ""
        last_html = ""
        last_report: Optional[dict[str, Any]] = None
        last_iter_dir: Optional[Path] = None
        stop_reason = "budget_exhausted"
        polish_done = False

        for it in range(0, max_iters + 1):
            iter_dir = work_dir / f"iter_{it:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            is_stock_attempt0 = (it == 0)
            pivot_directive = ""

            if is_stock_attempt0:
                # HONEST ATTEMPT 0: stock one-shot, ZERO Verified Ledger context, ZERO verifier hints!
                messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": "You are an expert graphics and interactive web programmer. Produce complete, runnable single-file HTML applications.",
                    },
                    {"role": "user", "content": task_prompt},
                ]
                mode_label = "attempt0_stock_oneshot"
                temp = 0.6
                effort: Any = 0.12
            else:
                ledger_ctx = ledger.render_context(task_prompt, max_chars=3600, max_winners=1, max_repairs=5, winner_excerpt_chars=500)
                sys_msg = (
                    "You are Inkling-2-Mini operating inside an automated Verified Ledger + Headless Playwright WebGL2 "
                    "self-improvement loop. Every claim is verified by the real browser runtime and pixel/playtest oracle. "
                    "Inspect the host verifier telemetry AND your own rendered screenshot carefully, diagnose the exact root cause, "
                    "and output a complete, fixed, self-contained HTML file inside a single ```html ... ``` block.\n\n"
                    f"{ledger_ctx}"
                )

                # Explicit Pivot & Bounded-Stop Controller (Loktar critique #4)
                if last_report and not last_report["ok"]:
                    sig = last_report["error_signature"]
                    repeat_cnt = sum(1 for s in error_sig_history[-2:] if s == sig)
                    if repeat_cnt >= 2 or sig == "TOKEN_TRUNCATION":
                        pivot_directive = (
                            f"\n[VERIFIED_LEDGER PIVOT TRIGGERED: error_signature='{sig}' (repeat_cnt={repeat_cnt})]\n"
                            "STOP patching the same broken structure. Pivot cleanly: keep reasoning crisp, simplify the "
                            "failing subsystem, verify every matrix/shader/buffer/state invariant from first principles, "
                            "and guarantee the full HTML closes cleanly with </html>."
                        )
                    mode_label = "verified_ledger_pivot_repair" if pivot_directive else "verified_ledger_repair"
                    temp = 0.6
                    effort = 0.10 if sig == "TOKEN_TRUNCATION" else (0.16 if pivot_directive else 0.14)
                    feedback_txt = compact_verifier_feedback(last_report, target_kind)
                    prev_block = f"\n\nPREVIOUS HTML CODE (to diagnose and fix):\n```html\n{last_html[:26000]}\n```" if last_html else ""
                    user_text = (
                        f"ORIGINAL TASK:\n{task_prompt}\n\n"
                        f"=== YOUR PREVIOUS CANDIDATE (iter_{it-1:02d}) FAILED HOST VERIFICATION ===\n"
                        f"{feedback_txt}\n"
                        f"{pivot_directive}\n\n"
                        "Inspect BOTH the telemetry above AND any attached screenshot(s) of what Chromium actually rendered.\n"
                        "First briefly state the exact root cause in 2-3 bullet points, then output the COMPLETE fixed "
                        f"self-contained HTML file inside a single ```html ... ``` block.{prev_block}"
                    )
                else:
                    # Visual Self-Review & Polish pass after functional pass!
                    mode_label = "verified_ledger_visual_polish"
                    polish_done = True
                    temp = 0.6
                    effort = 0.16
                    feedback_txt = compact_verifier_feedback(last_report, target_kind) if last_report else ""
                    user_text = (
                        f"ORIGINAL TASK:\n{task_prompt}\n\n"
                        f"=== YOUR PREVIOUS CANDIDATE (iter_{it-1:02d}) PASSED ALL HARD GATES ===\n"
                        f"{feedback_txt}\n\n"
                        "VISUAL SELF-REVIEW & POLISH PASS:\n"
                        "Look closely at the attached rendered screenshot(s) of your own live scene/game. "
                        "Critique framing, visual richness, silhouette readability, lighting/color harmony, detail density, "
                        "and HUD elegance — then upgrade the visuals & polish while strictly preserving 100% of passing "
                        "technical invariants (0 console/WebGL errors, all interaction/sort/kinematics/playtest gates). "
                        "Do NOT break any working hook or feature! Output the complete upgraded HTML in ```html ... ```.\n\n"
                        f"CURRENT PASSING HTML CODE:\n```html\n{last_html[:26000]}\n```"
                    )

                # Attach actual rendered screenshot(s) using Inkling-2-Mini's native vision tower!
                user_parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
                if last_iter_dir is not None:
                    shot_candidates = []
                    for name in ["frame_initial.png", "frame_orbit_90.png", "frame_orbit_180.png", "frame_anim_t2.png", "frame_orbit.png", "frame_playing_start.png", "frame_action_scored.png"]:
                        sp = last_iter_dir / name
                        if sp.exists():
                            shot_candidates.append((name, sp))
                    if len(shot_candidates) >= 2:
                        sheet_path = iter_dir / "input_vision_sheet.png"
                        make_contact_sheet(shot_candidates[:3], sheet_path)
                        b64 = encode_image_b64(sheet_path, max_w=960, max_h=320)
                        user_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
                    elif len(shot_candidates) == 1:
                        b64 = encode_image_b64(shot_candidates[0][1], max_w=640, max_h=360)
                        user_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": user_parts},
                ]

            print(f"[{target_kind}] iter={it} ({mode_label}) calling Inkling-2-Mini...", flush=True)
            llm = call_inkling(
                base_url=base_url,
                model=model,
                messages=messages,
                max_tokens=16384,
                temperature=temp,
                reasoning_effort=effort,
            )
            (iter_dir / "llm_response.json").write_text(
                json.dumps(
                    {
                        "iter": it,
                        "mode": mode_label,
                        "id": llm["id"],
                        "elapsed_s": llm["elapsed_s"],
                        "finish_reason": llm["finish_reason"],
                        "usage": llm["usage"],
                        "pivot_directive": pivot_directive,
                        "reasoning_excerpt": llm["reasoning_content"][:2000],
                        "content_chars": len(llm["content"]),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (iter_dir / "raw_output.txt").write_text(llm["content"], encoding="utf-8")

            html_code = extract_html(llm["content"])
            html_path = iter_dir / "app.html"
            if not html_code:
                report = {
                    "ok": False,
                    "score": 0.0,
                    "error_signature": "TOKEN_TRUNCATION" if llm["finish_reason"] == "length" else "HTML_EXTRACTION_FAILED",
                    "error_detail": f"Could not extract valid complete HTML (finish_reason={llm['finish_reason']}, chars={len(llm['content'])}). Keep code concise and close </html> cleanly.",
                    "summary": "",
                    "metrics": {"target_kind": target_kind},
                }
                (iter_dir / "verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            else:
                html_path.write_text(html_code, encoding="utf-8")
                if llm["finish_reason"] == "length" and "</html>" not in html_code.lower():
                    report = {
                        "ok": False,
                        "score": 2.0,
                        "error_signature": "TOKEN_TRUNCATION",
                        "error_detail": "Generation hit max_tokens before closing </html>! Make code cleaner and more compact so the entire file completes.",
                        "summary": "",
                        "metrics": {"target_kind": target_kind},
                    }
                    (iter_dir / "verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                else:
                    report = verify_html(target_kind, html_path, iter_dir)

            vres = VerifyResult(
                ok=bool(report["ok"]),
                score=float(report["score"]),
                error_signature=str(report.get("error_signature") or ""),
                error_detail=str(report.get("error_detail") or ""),
                summary=str(report.get("summary") or ""),
                tags=[target_kind, mode_label],
            )
            ledger_upd = ledger.record_outcome(
                task_id=task_id,
                task_text=spec["title"],
                candidate=html_code or "",
                result=vres,
                recent_error_sig=last_error_sig if vres.ok else "",
            )

            rec = {
                "iter": it,
                "mode": mode_label,
                "ok": vres.ok,
                "score": vres.score,
                "error_signature": vres.error_signature,
                "error_detail": vres.error_detail[:350],
                "summary": vres.summary,
                "finish_reason": llm["finish_reason"],
                "elapsed_s": llm["elapsed_s"],
                "usage": llm["usage"],
                "pivot_triggered": bool(pivot_directive),
                "ledger_update": ledger_upd,
                "html_path": str(html_path) if html_path.exists() else "",
            }
            records.append(rec)
            with trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

            print(
                f"[{target_kind}] iter={it} ok={vres.ok} score={vres.score:.1f} "
                f"err={vres.error_signature or 'NONE'} elapsed={llm['elapsed_s']}s "
                f"tokens={llm['usage'].get('completion_tokens', '?')} finish={llm['finish_reason']}",
                flush=True,
            )

            if vres.ok:
                if vres.score > best_score:
                    best_score = vres.score
                    best_iter = it
                    best_html_path = html_path
                    best_report = report
                    last_html = html_code or ""
                    last_report = report
                    last_iter_dir = iter_dir
                else:
                    # Keep best candidate (rollback regression on polish pass!)
                    rec["kept_previous_best"] = True
                last_error_sig = ""
                # Bounded Early Stop logic (Loktar critique #4)
                if is_stock_attempt0 and not polish_pass:
                    stop_reason = "passed_attempt0_no_polish"
                    break
                if polish_done or (not polish_pass) or (best_score >= 93.0 and not is_stock_attempt0):
                    stop_reason = "converged_verified_pass"
                    break
            else:
                sig = vres.error_signature or "UNKNOWN_ERROR"
                error_sig_history.append(sig)
                last_error_sig = sig
                if html_code:
                    last_html = html_code
                last_report = report
                last_iter_dir = iter_dir
                # Bounded Pivot-Exhaustion Stop: if exact same error repeats 3 times even after pivot, halt cleanly!
                if len(error_sig_history) >= 3 and error_sig_history[-1] == error_sig_history[-2] == error_sig_history[-3]:
                    stop_reason = f"bounded_stop_repeated_error_after_pivot:{sig}"
                    break

        # Copy canonical attempt_00 and final_best files to work_dir root
        attempt0_html = work_dir / "iter_00" / "app.html"
        if attempt0_html.exists():
            shutil.copy2(attempt0_html, work_dir / "attempt_00_stock.html")
        attempt0_png = work_dir / "iter_00" / "frame_initial.png"
        if attempt0_png.exists():
            shutil.copy2(attempt0_png, work_dir / "attempt_00_stock.png")

        if best_html_path and best_html_path.exists():
            shutil.copy2(best_html_path, work_dir / "final_verified.html")
            best_dir = best_html_path.parent
            for fn in ["frame_initial.png", "frame_orbit_90.png", "frame_orbit_180.png", "frame_anim_t1.png", "frame_anim_t2.png", "frame_orbit.png", "frame_playing_start.png", "frame_action_scored.png", "frame_gameover.png", "frame_restarted.png"]:
                src = best_dir / fn
                if src.exists():
                    shutil.copy2(src, work_dir / f"final_{fn}")

        summary = {
            "target_kind": target_kind,
            "task_id": task_id,
            "title": spec["title"],
            "model": model,
            "solved": best_iter is not None,
            "attempt0_passed": bool(records[0]["ok"]) if records else False,
            "attempt0_score": records[0]["score"] if records else 0.0,
            "attempt0_error": records[0]["error_signature"] if records else "",
            "best_iter": best_iter,
            "best_score": round(best_score, 2) if best_iter is not None else 0.0,
            "stop_reason": stop_reason,
            "total_iters": len(records),
            "ledger_stats": ledger.stats(),
            "records": records,
        }
        (work_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["gaussian_splats", "articulated_creature", "atari_arcade"])
    ap.add_argument("--ssh-host", default="v100")
    ap.add_argument("--remote-port", type=int, default=18601)
    ap.add_argument("--local-port", type=int, default=18601)
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--out-root", default="/home/orwel/dev_genius/experiments/kernals/results/loktar_showcase_848")
    ap.add_argument("--max-iters", type=int, default=5)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ledger_path = out_root / "verified_ledger.sqlite"
    all_summaries = {}

    with _SshTunnel(args.ssh_host, args.remote_port, args.local_port) as base_url:
        print(f"Connected to Inkling-2-Mini via {base_url}", flush=True)
        for idx, kind in enumerate(args.targets, start=1):
            sub_map = {
                "gaussian_splats": "01_gaussian_splats",
                "articulated_creature": "02_articulated_creature",
                "atari_arcade": "03_atari_arcade",
            }
            wdir = out_root / sub_map[kind]
            all_summaries[kind] = run_target(
                target_kind=kind,
                base_url=base_url,
                model=args.model,
                ledger_path=ledger_path,
                work_dir=wdir,
                max_iters=args.max_iters,
                polish_pass=True,
            )

    (out_root / "all_runs_summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(json.dumps({k: {"solved": v["solved"], "attempt0_ok": v["attempt0_passed"], "attempt0_err": v["attempt0_error"], "best_iter": v["best_iter"], "best_score": v["best_score"], "stop_reason": v["stop_reason"]} for k, v in all_summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

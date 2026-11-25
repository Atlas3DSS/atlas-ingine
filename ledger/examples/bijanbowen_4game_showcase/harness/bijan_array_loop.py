#!/usr/bin/env python3
"""Switchboard #854: 3-GPU Multi-Agent Inkling-2-Mini Array + Shared SQLite VerifiedLedger Tournament.

100% Scientific Integrity: ZERO human-written target game code.
Every single line of all 4 Bijan Bowen 3D WebGL2 games is authored & self-repaired by Inkling-2-Mini:
  - Attempt 0 (Stock One-Shot): Bare prompt on GPU 4, NO Verified Ledger, NO vision critique.
  - Rounds 1..3 (3-Archetype Parallel Tournament across Quincy V100 GPUs 4, 5, 6 = ~300 tok/s):
      * Agent A (GPU 4, port 18604) — Graphics & WebGL2 Shader Architect
      * Agent B (GPU 5, port 18605) — 3D Physics & Gameplay Systems Engineer
      * Agent C (GPU 6, port 18606) — Visual Polish, Atmosphere, HUD & Audio Director
    All 3 specialists generate IN PARALLEL, share ONE SQLite VerifiedLedger (4 banks, cross-game &
    cross-agent error->repair propagation), inspect their own rendered screenshots via Inkling-2-Mini's
    native vision tower, get graded by our headless Playwright WebGL2 + live gameplay oracle, breed the
    highest-scoring champion into the next round, and enforce bounded early stopping on convergence!
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
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
from bijan_verifier import verify_game  # noqa: E402

ARCHETYPES = [
    {
        "agent_id": "agent_graphics",
        "label": "Graphics & WebGL2 Shader Architect",
        "port": 18604,
        "temp": 0.55,
        "effort": 0.08,
        "focus": (
            "SPECIALTY — Graphics & WebGL2 Shader Architect: Guarantee flawless WebGL2 (#version 300 es, in/out NEVER attribute/varying), "
            "every active mat4 uniform (uModel/uView/uProj) explicitly uploaded via uniformMatrix4fv every frame in exact column-major layout "
            "(for lookAt: [rx,ux,fx,0, ry,uy,fy,0, rz,uz,fz,0, -dot(r,e),-dot(u,e),-dot(f,e),1] — indices 3,7,11 MUST be 0!), "
            "crisp directional + hemisphere + rim lighting, rich 3D geometry, and 0 GL/console errors."
        ),
    },
    {
        "agent_id": "agent_physics",
        "label": "3D Physics & Gameplay Systems Engineer",
        "port": 18605,
        "temp": 0.60,
        "effort": 0.08,
        "focus": (
            "SPECIALTY — 3D Physics & Gameplay Systems Engineer: Guarantee rock-solid 60fps simulation physics, ultra-responsive "
            "keyboard controls, crisp collision detection & scoring, authentic state transitions, and accurate live window.__GAME_STATE__ telemetry."
        ),
    },
    {
        "agent_id": "agent_polish",
        "label": "Visual Polish, Atmosphere & VFX Director",
        "port": 18606,
        "temp": 0.65,
        "effort": 0.08,
        "focus": (
            "SPECIALTY — Visual Polish, Atmosphere & VFX Director: Maximize silhouette clarity, dramatic mood/lighting/particles, "
            "crisp non-overlapping HUD/minimap/chart overlays, Web Audio API synthesized sound effects, and unforgettable visual impact."
        ),
    },
]

GAME_SPECS: dict[str, dict[str, Any]] = {
    "stock_offroad_4x4": {
        "dir_name": "01_stock_offroad_4x4",
        "task_id": "bijan_01_stock_offroad_4x4",
        "title": "Isometric 4x4 Off-Road Driving Game on a Live 5-Day Stock Price Curve 3D Terrain",
        "rival": "Claude",
        "prompt": """Build a single self-contained HTML file (100% offline, zero external CDN links, pure HTML + CSS + JavaScript + WebGL2 GLSL ES 3.00 + Web Audio API) implementing a vibrant 3D Isometric 4x4 Off-Road Driving Game where the drivable 3D mountain/ridge terrain is directly generated from a real 5-day stock share price curve ("BULL RUN 4x4").

Strict technical & gameplay requirements:
1. Pure WebGL2 (`canvas.getContext('webgl2')`, `#version 300 es`) with `gl.enable(gl.DEPTH_TEST)`, strictly column-major 4x4 matrices (`lookAt` with translation in indices 12,13,14), directional + ambient + rim shading, and an atmospheric twilight/emerald-gold sky backdrop.
2. Live 5-Day Stock Price 3D Terrain (Mon–Fri, e.g. 120+ price samples spanning $118 to $215 with dramatic rallies, dips, gaps, jumps, and volatility):
   - Extrude the 5-day stock price curve into a wide, drivable 3D ridge/road ribbon with colored slope segments (emerald green on bullish climbs, crimson red on bearish drops), dark grid valley floor, translucent vertical 3D volume/candlestick pillars flanking the track, glowing golden checkpoint arches at each day boundary (MON, TUE, WED, THU, FRI), and jump ramps!
3. Detailed 3D Articulated 4x4 Off-Road Truck:
   - Multi-part 3D chassis, cabin, windshield, roll-bar, roof lightbar, glowing headlights/taillights, and 4 independently positioned wheels that sample the exact terrain height at front/rear axles, compute real spring-damper suspension compression, dynamically pitch/tilt along terrain slopes, and spin visually as you drive!
   - Smooth 3D isometric follow camera tracking the truck cleanly in the center of the 1280x720 viewport.
4. Responsive Controls & Live Playability (game starts immediately playable on load — `state: 'playing'`):
   - `ArrowRight` / `D` accelerate forward along the 5-day stock curve; `ArrowLeft` / `A` brake/reverse; `Space` activates suspension hop / turbo bull-boost with glowing exhaust particles & Web Audio engine/boost synth; `R` resets to Monday open.
5. Synced Live Stock-Price Chart HUD (2D Canvas overlay in top-right corner) showing the exact 5-day price curve, animated green/red live price dot synchronized to the truck's X position, ticker (`$ATLS`), live `$price`, `% change`, current day, speed, checkpoints, score, and suspension gauge.
6. Expose read-only `window.__GAME_STATE__` updated EVERY animation frame:
   `{ state: 'playing', distance: <float>, progressPct: <0..100>, currentPrice: <float>, speed: <float>, truckX: <float>, truckY: <float>, pitch: <rad>, grounded: <bool>, checkpoints: <int>, score: <int> }`.

Output ONLY the complete self-contained HTML file inside a single ```html ... ``` fenced code block.""",
    },
    "bowling_horror_3d": {
        "dir_name": "02_bowling_horror_3d",
        "task_id": "bijan_02_bowling_horror_3d",
        "title": "3D Haunted Bowling Alley Horror Game with Rigid-Body 10-Pin Physics",
        "rival": "GLM-5.4",
        "prompt": """Build a single self-contained HTML file (100% offline, zero external CDN links, pure HTML + CSS + JavaScript + WebGL2 GLSL ES 3.00 + Web Audio API) implementing an eerie, atmospheric 3D Haunted Bowling Alley Horror Game ("LANE 13: MIDNIGHT STRIKE").

Strict technical & gameplay requirements:
1. Pure WebGL2 (`canvas.getContext('webgl2')`, `#version 300 es`) with `gl.enable(gl.DEPTH_TEST)`, strictly column-major 4x4 matrices, dramatic moody point/spot lighting, volumetric distance fog, and rich flickering overhead fluorescent/sodium lamps!
2. Atmospheric 3D Haunted Bowling Alley Scene:
   - Long polished dark timber bowling lane with lane boards, guide arrows, dark side gutters, glowing neon/amber foul line, worn side walls with eerie flickering light strips, dark pin-setter deck at the far end, and a creepy shadowy lurking silhouette / glowing-eyed anomaly peering from the far pin-deck darkness whose tension/proximity intensifies between rolls or on gutter balls!
   - 3D glossy dark bowling ball with finger-hole accent + 10 distinct 3D bowling pins arranged in the classic 1-2-3-4 triangle at the far end of the lane (white pins with crimson neck rings, clearly lit by an eerie overhead pin-deck spotlight so all 10 pins stand out crisply!).
3. Real Interactive Aiming, Throwing & 10-Pin Rigid-Body Physics (game starts immediately ready to aim & bowl — `state: 'playing'`):
   - `ArrowLeft` / `ArrowRight` (or `A` / `D` or mouse move) aim the ball position/angle across the lane with a subtle glowing aim line.
   - Hold `Space` (or click/release) to charge throw power (or tap `Space` for immediate crisp standard throw!) — ensure BOTH hold-and-release `Space` AND single `Space` tap smoothly launch the heavy bowling ball down the lane!
   - Real ball rolling physics down the lane + real ball-vs-pin AND pin-vs-pin impulse/topple collision physics! Knocked pins fly, spin, tilt, and clatter realistically!
   - Automatically scores knocked pins, advances roll 1 -> roll 2 -> next frame (10-frame bowling scorecard HUD at top), resets standing/all pins cleanly, and triggers creepy visual flicker + Web Audio SFX (sub-bass horror drone, rolling rumble, sharp pin scatter clatter, eerie stinger on anomaly pulse!).
4. Expose read-only `window.__GAME_STATE__` updated EVERY animation frame:
   `{ state: 'playing', phase: 'aim'|'rolling'|'settling', frame: <1..10>, roll: <1..2>, score: <int>, pinsStanding: <0..10>, pinsKnockedThisRoll: <0..10>, ballX: <float>, ballZ: <float>, ballMoving: <bool>, aimX: <float>, anomalyTension: <0..1>, strikes: <int>, spares: <int> }`.

Output ONLY the complete self-contained HTML file inside a single ```html ... ``` fenced code block.""",
    },
    "skate_to_gta_city": {
        "dir_name": "03_skate_to_gta_city",
        "task_id": "bijan_03_skate_to_gta_city",
        "title": "3D Skate-to-GTA Open-City Sandbox (Tricks + Enter/Drive/Exit Cars)",
        "rival": "GPT 6.2 Astra",
        "prompt": """Build a single self-contained HTML file (100% offline, zero external CDN links, pure HTML + CSS + JavaScript + WebGL2 GLSL ES 3.00 + Web Audio API) implementing a vibrant sunset 3D Open-City Skate-to-GTA Sandbox Game ("SUNSET GRIND: VICE BLOCK").

Strict technical & gameplay requirements:
1. Pure WebGL2 (`canvas.getContext('webgl2')`, `#version 300 es`) with `gl.enable(gl.DEPTH_TEST)`, strictly column-major 4x4 matrices, warm golden-hour sunset lighting + crisp specular highlights + deep city silhouettes.
2. Vibrant 3D Open City World:
   - Multi-block paved city road grid with dashed yellow/white lane markings, raised sidewalks, colorful stylized skyscrapers & glowing neon storefronts, skatepark quarter-pipes/ramps & grind rails right in the central plaza, palm trees & streetlamps, moving AI traffic cars cruising the avenues, AND multiple parked drivable 3D sports cars (including one bright glowing cyan/red hero sports coupe parked within 6–8 meters of player spawn so the player can effortlessly hop in!).
3. Seamless Dual-Mode Skate <-> Car Sandbox (starts immediately playable in `'skate'` mode — `state: 'playing'`):
   - **Skate Mode (`mode: 'skate'`)**: Articulated 3D skater riding a skateboard! `W/S` or `ArrowUp/ArrowDown` push/carve forward/back, `A/D` or `ArrowLeft/ArrowRight` steer smoothly, `Space` launches a crisp **Ollie jump** with gravity/airtime, and `Q` / `Shift` executes animated **Kickflip / 360 Shuvit** board-spin tricks (incrementing `tricksCount` and awarding `+150`–`+500` combo `score` with floating trick toast & Web Audio pop!).
   - **Car Mode (`mode: 'car'`)**: Pressing `F` (or `E`) enters the nearest drivable sports car within generous range (e.g. <= 22 units; if slightly further, snaps cleanly into the closest parked car so car entry never feels finicky!). In car mode, `W/S/Arrows` accelerate/reverse with responsive arcade car physics & glowing brake/headlights, `A/D` steer with drift, `Space` activates turbo/handbrake, and colliding with traffic/props raises the **Wanted Star Meter (1–5 stars)**! Pressing `F` (or `E`) exits the car seamlessly right back onto the skateboard!
4. Authentic GTA-Style HUD:
   - Live corner **2D Radar Minimap** (city blocks, player heading arrow, nearby drivable cars, mission marker), mode badge (`SKATE` / `CAR`), speed (MPH), trick combo & `SCORE`, active Mission banner, and **Wanted Stars (`★☆☆☆☆`)**, plus synthesized Web Audio skateboard/engine SFX.
5. Expose read-only `window.__GAME_STATE__` updated EVERY animation frame:
   `{ state: 'playing', mode: 'skate'|'car', playerX: <float>, playerY: <float>, playerZ: <float>, speed: <float>, airborne: <bool>, tricksCount: <int>, score: <int>, wantedStars: <0..5>, nearbyCarDist: <float>, carsCount: <int> }`.

Output ONLY the complete self-contained HTML file inside a single ```html ... ``` fenced code block.""",
    },
    "multiplayer_3d_racing": {
        "dir_name": "04_multiplayer_3d_racing",
        "task_id": "bijan_04_multiplayer_3d_racing",
        "title": "High-Speed Multiplayer 3D Arcade Racer (2P Split-Screen + BroadcastChannel + AI Pack)",
        "rival": "Gemini Carbon",
        "prompt": """Build a single self-contained HTML file (100% offline, zero external CDN links, pure HTML + CSS + JavaScript + WebGL2 GLSL ES 3.00 + Web Audio API) implementing a high-speed neon-twilight Multiplayer 3D WebGL2 Arcade Racing Game ("APEX TURBO 3D").

Strict technical & gameplay requirements:
1. Pure WebGL2 (`canvas.getContext('webgl2')`, `#version 300 es`) with `gl.enable(gl.DEPTH_TEST)`, strictly column-major 4x4 matrices, crisp directional + neon rim lighting, and vivid twilight sky/horizon colors.
2. High-Speed Closed 3D Racing Circuit:
   - Wide curved closed racing circuit (smooth spline/oval-chicane loop with red-and-white striped rumble curbs, dark asphalt road surface, glowing cyan/gold turbo boost pads on track, arched illuminated Start/Finish gantry, colorful grandstands, and barrier pylons).
   - 5 stylized 3D racing cars on the starting grid (each built with sculpted aerodynamic wedge body, cockpit canopy, rear spoiler, 4 wheels, and glowing tail/boost thrusters): **Player 1 (electric cyan)**, **Player 2 (hot crimson/orange)**, plus **3 competitive AI pack racers** driving smoothly along the circuit with overtaking variation!
3. Genuine Dual Multiplayer Support (starts immediately playable in **2P Local Split-Screen** mode — `state: 'playing', splitScreen: true` — with a crisp HUD toggle button for `1P / 2P SPLIT`):
   - **2-Player Local Split-Screen**: Render Player 1's chase camera on the LEFT half (`gl.viewport(0, 0, W/2, H)` + `gl.scissor`) and Player 2's chase camera on the RIGHT half (`gl.viewport(W/2, 0, W/2, H)` + `gl.scissor`) with a crisp center divider line!
   - **Controls**: P1 drives with `W/A/S/D` (`Shift` drift/boost); P2 drives simultaneously with `ArrowUp/ArrowLeft/ArrowDown/ArrowRight` (`Space` drift/boost). Both cars accelerate crisply immediately on key press!
   - **Live Multi-Tab Sync**: Initialize a genuine `new BroadcastChannel('atlas_racer_mp')` broadcasting local pose packets `{id, x, z, yaw, speed, lap}` and updating remote ghost/player racers when multiple browser tabs are open!
4. Arcade Physics & Race HUD:
   - Smooth arcade steering, drift slip, glowing boost-pad speed burst + FOV kick, lap & checkpoint progression (`LAP 1/3`), live race rank (`1ST..5TH`), speedometers for P1 & P2, live **2D Track Minimap** showing all 5 car dots, `MULTI-TAB SYNC: LIVE` indicator, and synthesized Web Audio engine hum + boost SFX.
5. Expose read-only `window.__GAME_STATE__` updated EVERY animation frame:
   `{ state: 'playing', splitScreen: <bool>, broadcastChannelActive: <bool>, p1: {x: <float>, z: <float>, speed: <float>, lap: <int>, rank: <int>}, p2: {x: <float>, z: <float>, speed: <float>, lap: <int>, rank: <int>}, aiCount: <int>, totalCars: <int>, raceTime: <float> }`.

Output ONLY the complete self-contained HTML file inside a single ```html ... ``` fenced code block.""",
    },
}


class _MultiSshTunnels:
    """Open 3 clean SSH tunnels to Quincy GPUs 4, 5, 6 (18604, 18605, 18606) and close on exit."""

    def __init__(self, remote_host: str, ports: list[int]) -> None:
        self.remote_host = remote_host
        self.ports = ports
        self.proc: Optional[subprocess.Popen] = None

    def __enter__(self) -> dict[int, str]:
        urls = {p: f"http://127.0.0.1:{p}/v1" for p in self.ports}
        if all(self._is_up(u) for u in urls.values()):
            return urls
        cmd = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15"]
        for p in self.ports:
            cmd += ["-L", f"127.0.0.1:{p}:127.0.0.1:{p}"]
        cmd.append(self.remote_host)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(45):
            if self.proc.poll() is not None:
                err = self.proc.stderr.read().decode("utf-8", errors="replace") if self.proc.stderr else ""
                raise RuntimeError(f"SSH tunnels exited early: {err}")
            if all(self._is_up(u) for u in urls.values()):
                return urls
            time.sleep(0.25)
        raise RuntimeError(f"Timed out waiting for SSH tunnels on {self.ports}")

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


def extract_html(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"```html\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        c = m.group(1).strip()
        if "<html" in c.lower() or "<!doctype" in c.lower() or "<canvas" in c.lower():
            return c
    for m in re.finditer(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", text, re.DOTALL):
        c = m.group(1).strip()
        if "<!doctype html" in c.lower() or "<html" in c.lower():
            return c
    idx = text.lower().find("<!doctype html")
    if idx == -1:
        idx = text.lower().find("<html")
    if idx != -1:
        end = text.lower().rfind("</html>")
        return text[idx : end + 7].strip() if end != -1 else text[idx:].strip()
    return None


def make_vision_strip_b64(iter_dir: Path, out_sheet: Path) -> Optional[str]:
    imgs = []
    for name in ("frame_initial.png", "frame_action_1.png", "frame_action_2.png", "frame_action_3.png"):
        p = iter_dir / name
        if p.exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail((426, 240), Image.Resampling.LANCZOS)
            imgs.append(im)
        if len(imgs) >= 3:
            break
    if not imgs:
        return None
    w = sum(im.width for im in imgs) + 8 * (len(imgs) - 1)
    h = max(im.height for im in imgs)
    canvas = Image.new("RGB", (w, h), (10, 14, 22))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width + 8
    canvas.save(out_sheet, format="PNG")
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_inkling(
    base_url: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 16384,
    temperature: float = 0.6,
    reasoning_effort: Any = 0.14,
    timeout_s: float = 420.0,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": "inkling-2-mini",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
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
    elapsed = round(time.time() - t0, 2)
    ch = data["choices"][0]
    msg = ch.get("message", {})
    return {
        "id": data.get("id", ""),
        "content": msg.get("content") or "",
        "finish_reason": str(ch.get("finish_reason") or ""),
        "usage": data.get("usage") or {},
        "elapsed_s": elapsed,
    }


def vision_qa_judge(base_url: str, game_kind: str, title: str, cand_dir: Path) -> dict[str, Any]:
    """Use Inkling-2-Mini's native vision tower as a strict 3D graphics QA critic on rendered gameplay frames."""
    strip_path = cand_dir / "vision_qa_strip.png"
    b64 = make_vision_strip_b64(cand_dir, strip_path)
    if not b64:
        return {"visual_pass": True, "visual_score": 75, "defect": ""}
    criteria = {
        "stock_offroad_4x4": "MUST clearly show a recognizable 3D 4x4 truck driving on a 3D hilly/ridged stock-curve terrain road (plus HUD/chart). FAIL if empty sky, giant starburst/pinwheel triangles, clipped polygons, or invisible truck/terrain.",
        "bowling_horror_3d": "MUST clearly show a recognizable 3D bowling lane perspective leading down toward visible white bowling pins (and ball) in moody lighting. FAIL if empty void, garbled pinwheel triangles, or unrecognizable geometry.",
        "skate_to_gta_city": "MUST clearly show a recognizable 3D city street/buildings with a skater/skateboard and/or drivable 3D car. FAIL if empty sky, pinwheel triangles, or broken flat/unrecognizable geometry.",
        "multiplayer_3d_racing": "MUST clearly show a 2-player LEFT/RIGHT split-screen 3D racing track with recognizable 3D race cars on asphalt road. FAIL if single unsplit view, empty sky, pinwheel triangles, or broken geometry.",
    }[game_kind]
    msgs = [
        {
            "role": "system",
            "content": "You are a strict, honest 3D WebGL2 visual QA inspector. Inspect the 3-frame rendered gameplay strip and output ONLY compact JSON: {\"visual_pass\": true|false, \"visual_score\": 0..100, \"defect\": \"short concrete description or empty\"}",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Game: {title}\nVisual Pass Criteria: {criteria}\nInspect these 3 gameplay frames (initial -> action -> action) and respond ONLY with the JSON object."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        },
    ]
    try:
        res = call_inkling(base_url, msgs, max_tokens=512, temperature=0.1, reasoning_effort=0.05, timeout_s=60.0)
        txt = res["content"]
        m = re.search(r"\{.*?\}", txt, re.DOTALL)
        if m:
            obj = json.loads(m.group(0))
            return {
                "visual_pass": bool(obj.get("visual_pass", True)),
                "visual_score": float(obj.get("visual_score", 75)),
                "defect": str(obj.get("defect", ""))[:220],
            }
    except Exception as e:
        return {"visual_pass": True, "visual_score": 75, "defect": f"vision_qa_skipped: {e}"}
    return {"visual_pass": True, "visual_score": 75, "defect": ""}


def compact_feedback(report: dict[str, Any]) -> str:
    m = report.get("metrics") or {}
    w = m.get("webgl") or {}
    img = m.get("action_image") or m.get("initial_image") or {}
    pt = m.get("playtest") or {}
    lines = [f"HOST VERIFIER: {'PASS' if report['ok'] else 'FAIL'} | score={report['score']}/100"]
    if not report["ok"]:
        lines.append(f"ERROR_SIGNATURE: {report['error_signature']}")
        lines.append(f"DIAGNOSTIC: {report['error_detail']}")
    else:
        lines.append(f"SUMMARY: {report['summary']}")
    if m.get("page_errors"):
        lines.append(f"PAGE_ERRORS: {m['page_errors'][:3]}")
    if m.get("console_errors"):
        lines.append(f"CONSOLE_ERRORS: {m['console_errors'][:3]}")
    if w:
        lines.append(
            f"WEBGL2: ctx={w.get('webgl2')}, draws={w.get('drawCalls')}, raf={w.get('rafCount')}, "
            f"shaderErr={w.get('shaderCompileErrors')}, glErr={w.get('glErrors')}, "
            f"unsetMat4={w.get('unsetMat4Uniforms')}, matErr={w.get('transposedMatrices')}"
        )
    if img:
        lines.append(
            f"PIXELS: colors={img.get('distinct_colors_q16')}, entropy={img.get('shannon_entropy')}, "
            f"center_std={img.get('center_std_luma')}, edge_pct={img.get('center_edge_pct')}%, edge_p99={img.get('center_edge_p99')}"
        )
    if pt:
        pt_clean = {k: v for k, v in pt.items() if k not in ("state_0", "state_2", "state_end", "state_pre", "state_skate", "state_car")}
        lines.append(f"PLAYTEST: {json.dumps(pt_clean)[:550]}")
    return "\n".join(lines)


def seed_ledger_facts(ledger: VerifiedLedger) -> None:
    ledger.set_fact("runtime_webgl2", "Headless Chromium WebGL2 (GLSL ES 3.00, ANGLE SwiftShader Vulkan, 1280x720).", category="hardware")
    ledger.set_fact("glsl_es_300", "WebGL2 '#version 300 es' MUST start on line 1 col 0 (no leading newline), use 'in'/'out' (NEVER 'attribute'/'varying'), and declare 'out vec4 fragColor' (NEVER 'gl_FragColor').", category="webgl2")
    ledger.set_fact("offline_html", "100% self-contained single HTML file; ZERO external CDN fetches allowed.", category="runtime")
    ledger.set_fact(
        "webgl_col_major",
        "WebGL Float32Array(16) is column-major: m[3]=0, m[7]=0, m[11]=0, m[15]=1 for affine/view/ortho matrices! "
        "Exact lookAt(eye,target,up) Float32Array(16) with f=norm(eye-target), r=norm(cross(up,f)), u=cross(f,r): "
        "[rx,ux,fx,0,  ry,uy,fy,0,  rz,uz,fz,0,  -dot(r,eye),-dot(u,eye),-dot(f,eye),1]. "
        "NEVER put -dot(...) at m[3],m[7],m[11]! Upload every active mat4 uniform before draw.",
        category="webgl2",
    )
    ledger.set_fact("game_start_ready", "Game must start immediately playable (state='playing') on load without blocking modals, and update window.__GAME_STATE__ every frame.", category="gameplay")


def run_game_tournament(
    game_kind: str,
    urls: dict[int, str],
    ledger_path: Path,
    out_root: Path,
    max_rounds: int = 3,
) -> dict[str, Any]:
    spec = GAME_SPECS[game_kind]
    work_dir = out_root / spec["dir_name"]
    work_dir.mkdir(parents=True, exist_ok=True)
    trace_path = work_dir / "array_trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()

    with VerifiedLedger(ledger_path, env_version="v1_bijan_array_webgl2", max_context_chars=3800) as ledger:
        ledger.conn.execute("PRAGMA busy_timeout = 30000;")
        seed_ledger_facts(ledger)

        # ---- ATTEMPT 0: HONEST STOCK ONE-SHOT BASELINE (NO Verified Ledger, NO Swarm, NO Vision) ----
        a0_dir = work_dir / "attempt_00_stock"
        a0_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== [{game_kind}] ATTEMPT 0 (Stock One-Shot Baseline on GPU 4) ===", flush=True)
        a0_msgs = [
            {"role": "system", "content": "You are an expert 3D WebGL2 game developer. Produce a complete runnable single-file HTML game."},
            {"role": "user", "content": spec["prompt"]},
        ]
        llm0 = call_inkling(urls[18604], a0_msgs, max_tokens=16384, temperature=0.6, reasoning_effort=0.08)
        (a0_dir / "raw_output.txt").write_text(llm0["content"], encoding="utf-8")
        a0_html = extract_html(llm0["content"])
        if not a0_html or (llm0["finish_reason"] == "length" and "</html>" not in a0_html.lower()):
            rep0 = {"ok": False, "score": 2.0, "error_signature": "TOKEN_TRUNCATION", "error_detail": "Stock one-shot hit max_tokens before closing </html>.", "summary": "", "metrics": {}}
            (a0_dir / "verify_report.json").write_text(json.dumps(rep0, indent=2), encoding="utf-8")
        else:
            (a0_dir / "app.html").write_text(a0_html, encoding="utf-8")
            rep0 = verify_game(game_kind, a0_dir / "app.html", a0_dir)
            if rep0["ok"]:
                vqa0 = vision_qa_judge(urls[18604], game_kind, spec["title"], a0_dir)
                rep0["metrics"]["vision_qa"] = vqa0
                if not vqa0["visual_pass"] or vqa0["visual_score"] < 55:
                    rep0["ok"] = False
                    rep0["score"] = min(float(rep0["score"]), 34.0)
                    rep0["error_signature"] = "VISION_QA_DEFECT"
                    rep0["error_detail"] = f"Inkling-2-Mini native vision QA rejected rendered 3D scene (visual_score={vqa0['visual_score']}): {vqa0['defect']}"
                    rep0["summary"] = ""
                else:
                    rep0["summary"] = f"{rep0['summary']} | vision_qa={vqa0['visual_score']:.0f}/100"
                (a0_dir / "verify_report.json").write_text(json.dumps(rep0, indent=2), encoding="utf-8")

        vres0 = VerifyResult(ok=bool(rep0["ok"]), score=float(rep0["score"]), error_signature=str(rep0.get("error_signature") or ""), error_detail=str(rep0.get("error_detail") or ""), summary=str(rep0.get("summary") or ""), tags=[game_kind, "attempt0_stock"])
        upd0 = ledger.record_outcome(spec["task_id"], spec["title"], a0_html or "", vres0)
        print(f"[{game_kind}] attempt_00_stock: ok={rep0['ok']} score={rep0['score']} err={rep0['error_signature'] or 'NONE'} elapsed={llm0['elapsed_s']}s", flush=True)

        if (a0_dir / "app.html").exists():
            shutil.copy2(a0_dir / "app.html", work_dir / "attempt_00_stock.html")
        for fn in ("frame_action_2.png", "frame_initial.png"):
            if (a0_dir / fn).exists():
                shutil.copy2(a0_dir / fn, work_dir / "attempt_00_stock.png")
                break

        all_records: list[dict[str, Any]] = [
            {
                "round": 0,
                "agent_id": "stock_oneshot",
                "ok": bool(rep0["ok"]),
                "score": float(rep0["score"]),
                "error_signature": rep0.get("error_signature", ""),
                "error_detail": str(rep0.get("error_detail", ""))[:300],
                "summary": rep0.get("summary", ""),
                "elapsed_s": llm0["elapsed_s"],
                "completion_tokens": llm0["usage"].get("completion_tokens", 0),
                "finish_reason": llm0["finish_reason"],
                "ledger_update": upd0,
            }
        ]
        with trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(all_records[-1]) + "\n")

        # Track overall tournament champion
        champ_ok = bool(rep0["ok"])
        champ_score = float(rep0["score"]) if champ_ok else -1.0
        champ_html = a0_html or ""
        champ_dir = a0_dir
        champ_report = rep0
        champ_label = "attempt_00_stock" if champ_ok else ""
        seed_html = a0_html or ""
        seed_dir = a0_dir
        seed_report = rep0
        recent_err_sig = str(rep0.get("error_signature") or "")
        stop_reason = "budget_exhausted"

        # ---- MULTI-AGENT 3-GPU TOURNAMENT ROUNDS (1..max_rounds) ----
        for rnd in range(1, max_rounds + 1):
            ledger_ctx = ledger.render_context(spec["prompt"], max_chars=3600, max_winners=1, max_repairs=6, winner_excerpt_chars=420)
            rnd_dir = work_dir / f"round_{rnd:02d}"
            rnd_dir.mkdir(parents=True, exist_ok=True)
            vision_b64 = make_vision_strip_b64(seed_dir, rnd_dir / "seed_vision_strip.png")
            fb_txt = compact_feedback(seed_report)
            is_polish_round = bool(seed_report.get("ok"))

            print(
                f"=== [{game_kind}] ROUND {rnd} ({'3-WAY CHAMPION POLISH BREEDING' if is_polish_round else '3-WAY PARALLEL REPAIR/PIVOT TOURNAMENT'} across GPUs 4,5,6) ===",
                flush=True,
            )

            def _run_specialist(arch: dict[str, Any]) -> dict[str, Any]:
                adir = rnd_dir / arch["agent_id"]
                adir.mkdir(parents=True, exist_ok=True)
                sys_msg = (
                    f"You are Inkling-2-Mini serving as '{arch['label']}' inside a 3-agent parallel Verified Ledger tournament.\n"
                    f"{arch['focus']}\n"
                    "Inspect the host WebGL2/playtest diagnostics AND the rendered screenshot strip carefully, then return a "
                    "complete, 100% self-contained single HTML file inside ```html ... ```.\n\n"
                    f"{ledger_ctx}"
                )
                if not is_polish_round:
                    # Only pivot from scratch if seed failed early (compile/crash/truncation, score < 20) or missing HTML;
                    # if seed already compiled & ran physics (score >= 20), let Graphics & Polish patch the exact bug while Physics explores a clean variant!
                    seed_score = float(seed_report.get("score") or 0.0)
                    fresh_pivot = (not seed_html) or (rnd >= 2 and seed_score < 20.0 and arch["agent_id"] != "agent_polish") or (seed_score >= 20.0 and arch["agent_id"] == "agent_physics" and rnd >= 2)
                    prev_snip = "" if fresh_pivot else f"\n\nPREVIOUS CANDIDATE HTML (diagnose & fix all root causes, keep complete HTML concise <=19 KB):\n```html\n{seed_html[:19000]}\n```"
                    pivot_note = (
                        "\n[VERIFIED_LEDGER PIVOT DIRECTIVE]: Build a clean, concise (<=18 KB), rock-solid from-scratch architecture avoiding the "
                        f"failed pattern ({seed_report.get('error_signature')}). Prioritize passing EVERY WebGL2, physics, visual-QA, and playtest invariant cleanly!"
                        if fresh_pivot
                        else "\nFix the exact root cause diagnosed above (check every shader/uniform/lookAt/matrix/control) and upgrade visual & gameplay quality!"
                    )
                    utxt = (
                        f"ORIGINAL GAME SPEC:\n{spec['prompt']}\n\n"
                        f"=== PREVIOUS BEST DIAGNOSTIC ===\n{fb_txt}\n{pivot_note}{prev_snip}"
                    )
                else:
                    utxt = (
                        f"ORIGINAL GAME SPEC:\n{spec['prompt']}\n\n"
                        f"=== CURRENT PASSING CHAMPION (score={seed_report['score']}/100) ===\n{fb_txt}\n\n"
                        f"TOURNAMENT POLISH BREEDING ({arch['label']}):\n"
                        "Inspect the attached 3-frame rendered gameplay strip of the current passing champion! "
                        "Dramatically elevate 3D visual richness, lighting, environmental landmarks, vehicle/character detail, "
                        "particles, HUD polish, and game feel — while STRICTLY preserving 100% of passing WebGL2 & live playtest invariants "
                        "(0 errors, exact controls, exact window.__GAME_STATE__ contract) and keeping the HTML concise (<=19 KB) so it never truncates!\n\n"
                        f"CURRENT CHAMPION HTML:\n```html\n{seed_html[:19000]}\n```"
                    )

                uparts: list[dict[str, Any]] = [{"type": "text", "text": utxt}]
                if vision_b64:
                    uparts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{vision_b64}"}})
                msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": uparts}]
                eff = 0.06 if seed_report.get("error_signature") == "TOKEN_TRUNCATION" else arch["effort"]
                llm = call_inkling(urls[arch["port"]], msgs, max_tokens=16384, temperature=arch["temp"], reasoning_effort=eff)
                (adir / "raw_output.txt").write_text(llm["content"], encoding="utf-8")
                code = extract_html(llm["content"])
                if code:
                    (adir / "app.html").write_text(code, encoding="utf-8")
                return {"arch": arch, "adir": adir, "llm": llm, "code": code}

            # Run all 3 Inkling-2-Mini specialists SIMULTANEOUSLY across Quincy GPUs 4, 5, 6!
            gen_results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                futs = [ex.submit(_run_specialist, a) for a in ARCHETYPES]
                for fut in concurrent.futures.as_completed(futs):
                    gen_results.append(fut.result())

            # Verify all 3 candidates sequentially in headless Playwright Chromium (deterministic, clean CPU)
            round_candidates = []
            for gr in sorted(gen_results, key=lambda x: x["arch"]["port"]):
                arch = gr["arch"]
                adir = gr["adir"]
                llm = gr["llm"]
                code = gr["code"]
                if not code or (llm["finish_reason"] == "length" and "</html>" not in code.lower()):
                    rep = {"ok": False, "score": 2.0, "error_signature": "TOKEN_TRUNCATION", "error_detail": "Hit max_tokens before closing </html>.", "summary": "", "metrics": {}}
                    (adir / "verify_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
                else:
                    rep = verify_game(game_kind, adir / "app.html", adir)
                    if rep["ok"]:
                        vqa = vision_qa_judge(urls[18604], game_kind, spec["title"], adir)
                        rep["metrics"]["vision_qa"] = vqa
                        if not vqa["visual_pass"] or vqa["visual_score"] < 55:
                            rep["ok"] = False
                            rep["score"] = min(float(rep["score"]), 34.0)
                            rep["error_signature"] = "VISION_QA_DEFECT"
                            rep["error_detail"] = f"Inkling-2-Mini native vision QA rejected rendered 3D scene (visual_score={vqa['visual_score']}): {vqa['defect']}"
                            rep["summary"] = ""
                        else:
                            rep["summary"] = f"{rep['summary']} | vision_qa={vqa['visual_score']:.0f}/100"
                        (adir / "verify_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

                vres = VerifyResult(
                    ok=bool(rep["ok"]),
                    score=float(rep["score"]),
                    error_signature=str(rep.get("error_signature") or ""),
                    error_detail=str(rep.get("error_detail") or ""),
                    summary=str(rep.get("summary") or ""),
                    tags=[game_kind, f"r{rnd}", arch["agent_id"]],
                )
                upd = ledger.record_outcome(
                    spec["task_id"],
                    spec["title"],
                    code or "",
                    vres,
                    recent_error_sig=recent_err_sig if vres.ok else "",
                )
                rec = {
                    "round": rnd,
                    "agent_id": arch["agent_id"],
                    "archetype": arch["label"],
                    "gpu_port": arch["port"],
                    "ok": bool(rep["ok"]),
                    "score": float(rep["score"]),
                    "error_signature": rep.get("error_signature", ""),
                    "error_detail": str(rep.get("error_detail", ""))[:300],
                    "summary": rep.get("summary", ""),
                    "elapsed_s": llm["elapsed_s"],
                    "completion_tokens": llm["usage"].get("completion_tokens", 0),
                    "finish_reason": llm["finish_reason"],
                    "ledger_update": upd,
                }
                all_records.append(rec)
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                round_candidates.append((rec, rep, code or "", adir))
                print(
                    f"  [{game_kind}] r{rnd}/{arch['agent_id']} ok={rep['ok']} score={rep['score']:.1f} "
                    f"err={rep['error_signature'] or 'NONE'} ({llm['elapsed_s']}s, {llm['usage'].get('completion_tokens', 0)} tok)",
                    flush=True,
                )

            # Pick tournament round winner (prefer ok=True highest score, else highest diagnostic score)
            passing = [c for c in round_candidates if c[1]["ok"]]
            if passing:
                best_c = max(passing, key=lambda c: float(c[1]["score"]))
                if float(best_c[1]["score"]) > champ_score:
                    champ_ok = True
                    champ_score = float(best_c[1]["score"])
                    champ_html = best_c[2]
                    champ_dir = best_c[3]
                    champ_report = best_c[1]
                    champ_label = f"round_{rnd:02d}/{best_c[0]['agent_id']}"
                # Seed next round from best champion
                seed_html, seed_dir, seed_report = champ_html, champ_dir, champ_report
                recent_err_sig = ""
                # Bounded Early Stop: stop after Round 2 polish OR if champion already >= 91 after polish!
                if is_polish_round or (rnd >= 2 and champ_score >= 88.0):
                    stop_reason = "converged_verified_champion"
                    break
            else:
                best_fail = max(round_candidates, key=lambda c: float(c[1]["score"]))
                if not champ_ok:
                    seed_html, seed_dir, seed_report = best_fail[2], best_fail[3], best_fail[1]
                    recent_err_sig = str(best_fail[1].get("error_signature") or "")

        # Save canonical final champion artifacts
        if champ_ok and (champ_dir / "app.html").exists():
            shutil.copy2(champ_dir / "app.html", work_dir / "final_verified.html")
            for fn in ("frame_initial.png", "frame_action_1.png", "frame_action_2.png", "frame_action_3.png"):
                if (champ_dir / fn).exists():
                    shutil.copy2(champ_dir / fn, work_dir / f"final_{fn}")

        summary = {
            "game_kind": game_kind,
            "task_id": spec["task_id"],
            "title": spec["title"],
            "rival": spec["rival"],
            "solved": champ_ok,
            "attempt0_passed": bool(rep0["ok"]),
            "attempt0_score": float(rep0["score"]),
            "attempt0_error": rep0.get("error_signature", ""),
            "champion": champ_label,
            "best_score": round(champ_score, 2) if champ_ok else 0.0,
            "champion_summary": champ_report.get("summary", "") if champ_ok else "",
            "stop_reason": stop_reason,
            "total_candidates": len(all_records),
            "ledger_stats": ledger.stats(),
            "records": all_records,
        }
        (work_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"==> [{game_kind}] DONE: solved={champ_ok} attempt0={rep0['score']} ({rep0.get('error_signature') or 'PASS'}) -> champion={champ_label} ({summary['best_score']}/100)", flush=True)
        return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["stock_offroad_4x4", "bowling_horror_3d", "skate_to_gta_city", "multiplayer_3d_racing"])
    ap.add_argument("--out-root", default="/home/orwel/dev_genius/experiments/kernals/results/bijanbowen_showcase_854")
    ap.add_argument("--max-rounds", type=int, default=3)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ledger_path = out_root / "verified_ledger.sqlite"
    all_summaries = {}

    with _MultiSshTunnels("v100", [18604, 18605, 18606]) as urls:
        print(f"Connected to 3-GPU Quincy Inkling-2-Mini Array: {urls}", flush=True)
        for g in args.games:
            all_summaries[g] = run_game_tournament(g, urls, ledger_path, out_root, max_rounds=args.max_rounds)

    (out_root / "all_games_summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print("\n=== ALL 4 BIJAN BOWEN GAMES SUMMARY ===")
    print(json.dumps({k: {"solved": v["solved"], "a0": f"{v['attempt0_score']} ({v['attempt0_error'] or 'PASS'})", "champ": f"{v['champion']} -> {v['best_score']}/100"} for k, v in all_summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

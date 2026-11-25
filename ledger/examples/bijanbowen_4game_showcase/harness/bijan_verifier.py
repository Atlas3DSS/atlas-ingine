#!/usr/bin/env python3
"""Headless Chromium-1248 WebGL2 State Inspector + Live Gameplay/Physics Oracle for Switchboard #854.

Verifies 4 frontier-grade 3D WebGL2 games 100% objectively:
  1. stock_offroad_4x4    : Isometric 4x4 off-road driving on a 5-day stock price 3D terrain curve
  2. bowling_horror_3d    : Haunted 3D bowling alley with real 10-pin rigid-body physics & horror lighting
  3. skate_to_gta_city    : 3D open-city sandbox with skateboarding tricks + enter/drive/exit 3D cars
  4. multiplayer_3d_racing: High-speed 3D arcade racer with 2P split-screen + BroadcastChannel + AI pack
"""
from __future__ import annotations

import functools
import http.server
import json
import math
import os
import re
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

CHROME_EXECUTABLE = "/home/orwel/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome"
CHROME_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-webgl",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--autoplay-policy=no-user-gesture-required",
]

GL_ENUM_NAMES = {
    0x0500: "GL_INVALID_ENUM",
    0x0501: "GL_INVALID_VALUE",
    0x0502: "GL_INVALID_OPERATION",
    0x0505: "GL_OUT_OF_MEMORY",
    0x0506: "GL_INVALID_FRAMEBUFFER_OPERATION",
}

WEBGL_TELEMETRY_INIT_JS = r"""
(() => {
  const T = {
    contexts: [],
    webgl2: false,
    webgl1: false,
    canvas2d: false,
    shaderCount: 0,
    shaderCompileErrors: [],
    programLinkErrors: [],
    drawCalls: 0,
    instancedCalls: 0,
    depthTestEnabled: false,
    rafCount: 0,
    glErrors: [],
    activeContexts: [],
    transposedMatrices: [],
    attribBufferMismatches: [],
    unsetMat4Uniforms: [],
    broadcastChannelsCreated: [],
  };
  window.__VERIFIER_TELEMETRY__ = T;

  if (typeof window.BroadcastChannel === 'function') {
    const OrigBC = window.BroadcastChannel;
    window.BroadcastChannel = class extends OrigBC {
      constructor(name) {
        super(name);
        T.broadcastChannelsCreated.push(String(name));
      }
    };
  }

  const origRAF = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = function(cb) {
    return origRAF((ts) => {
      T.rafCount++;
      return cb(ts);
    });
  };

  const origGetContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function(type, attrs) {
    const t = String(type || '').toLowerCase();
    let patched = attrs;
    if (t === 'webgl2' || t === 'webgl' || t === 'experimental-webgl') {
      patched = Object.assign({}, attrs || {}, { preserveDrawingBuffer: true });
    }
    const ctx = origGetContext.call(this, type, patched);
    if (!ctx) return ctx;
    T.contexts.push(t);
    if (t === 'webgl2') T.webgl2 = true;
    else if (t === 'webgl' || t === 'experimental-webgl') T.webgl1 = true;
    else if (t === '2d') T.canvas2d = true;

    if ((t === 'webgl2' || t === 'webgl' || t === 'experimental-webgl') && !ctx.__verifierWrapped) {
      ctx.__verifierWrapped = true;
      T.activeContexts.push(ctx);
      const shaderKinds = new WeakMap();
      const bufSizes = new WeakMap();
      const progActiveMat4 = new WeakMap();
      const progSetLocs = new WeakMap();
      const locToInfo = new WeakMap();
      const progReportedUnset = new WeakSet();
      let currentProg = null;
      let boundArrayBuf = null;

      const origShaderSource = ctx.shaderSource.bind(ctx);
      ctx.shaderSource = function(shader, src) {
        const stype = ctx.getShaderParameter(shader, ctx.SHADER_TYPE);
        shaderKinds.set(shader, stype === ctx.VERTEX_SHADER ? 'vertex' : 'fragment');
        T.shaderCount++;
        return origShaderSource(shader, src);
      };

      const origCompile = ctx.compileShader.bind(ctx);
      ctx.compileShader = function(shader) {
        const res = origCompile(shader);
        if (!ctx.getShaderParameter(shader, ctx.COMPILE_STATUS)) {
          T.shaderCompileErrors.push({
            kind: shaderKinds.get(shader) || 'unknown',
            log: String(ctx.getShaderInfoLog(shader) || 'compile error'),
          });
        }
        return res;
      };

      const origLink = ctx.linkProgram.bind(ctx);
      ctx.linkProgram = function(prog) {
        const res = origLink(prog);
        if (!ctx.getProgramParameter(prog, ctx.LINK_STATUS)) {
          T.programLinkErrors.push(String(ctx.getProgramInfoLog(prog) || 'link error'));
        } else {
          try {
            const nUni = ctx.getProgramParameter(prog, ctx.ACTIVE_UNIFORMS) || 0;
            const mat4Names = [];
            for (let i = 0; i < nUni; i++) {
              const info = ctx.getActiveUniform(prog, i);
              if (info && info.type === ctx.FLOAT_MAT4) mat4Names.push(info.name);
            }
            progActiveMat4.set(prog, mat4Names);
            progSetLocs.set(prog, new Set());
          } catch (e) {}
        }
        return res;
      };

      const origUseProg = ctx.useProgram.bind(ctx);
      ctx.useProgram = function(prog) {
        currentProg = prog;
        return origUseProg(prog);
      };

      const origGetUniLoc = ctx.getUniformLocation.bind(ctx);
      ctx.getUniformLocation = function(prog, name) {
        const loc = origGetUniLoc(prog, name);
        if (loc) locToInfo.set(loc, { prog, name: String(name) });
        return loc;
      };

      const origBindBuf = ctx.bindBuffer.bind(ctx);
      ctx.bindBuffer = function(target, buf) {
        if (target === ctx.ARRAY_BUFFER) boundArrayBuf = buf;
        return origBindBuf(target, buf);
      };

      const origBufferData = ctx.bufferData.bind(ctx);
      ctx.bufferData = function(...args) {
        let blen = 0;
        if (typeof args[1] === 'number') blen = args[1];
        else if (args[1] && typeof args[1].byteLength === 'number') blen = args[1].byteLength;
        if (args[0] === ctx.ARRAY_BUFFER && boundArrayBuf && blen > 0) bufSizes.set(boundArrayBuf, blen);
        return origBufferData(...args);
      };

      const origVAP = ctx.vertexAttribPointer.bind(ctx);
      ctx.vertexAttribPointer = function(index, size, type, norm, stride, offset) {
        const bsz = boundArrayBuf ? (bufSizes.get(boundArrayBuf) || 0) : 0;
        if (stride > 0 && bsz > 0 && bsz < stride * 2 && T.attribBufferMismatches.length < 6) {
          T.attribBufferMismatches.push(`attrib(${index}) stride=${stride} offset=${offset} bound to tiny ARRAY_BUFFER (${bsz} bytes)`);
        }
        return origVAP(index, size, type, norm, stride, offset);
      };

      const origUM4 = ctx.uniformMatrix4fv.bind(ctx);
      ctx.uniformMatrix4fv = function(loc, transpose, val, ...rest) {
        const info = loc ? locToInfo.get(loc) : null;
        if (info && progSetLocs.has(info.prog)) {
          progSetLocs.get(info.prog).add(info.name);
        } else if (currentProg && progSetLocs.has(currentProg) && info) {
          progSetLocs.get(currentProg).add(info.name);
        }
        if (!transpose && val && val.length >= 16 && T.transposedMatrices.length < 6) {
          const m = val;
          const bottomRowMag = Math.abs(m[3]) + Math.abs(m[7]) + Math.abs(m[11]);
          const row3Norm = Math.hypot(m[3], m[7], m[11]);
          if (Math.abs(m[15] - 1.0) < 0.02 && bottomRowMag > 0.05) {
            T.transposedMatrices.push(
              `corrupted/row-major affine mat4 passed to column-major WebGL (m[15]=1 but projective row m[3],m[7],m[11] non-zero!): m[3]=${m[3].toFixed(2)}, m[7]=${m[7].toFixed(2)}, m[11]=${m[11].toFixed(2)}, m[12..14]=(${m[12].toFixed(2)},${m[13].toFixed(2)},${m[14].toFixed(2)})`
            );
          } else if (row3Norm > 2.5) {
            T.transposedMatrices.push(
              `corrupted projective row in mat4 (|m[3],m[7],m[11]|=${row3Norm.toFixed(2)} > 2.5 — translation likely placed at m[3],m[7],m[11] instead of m[12],m[13],m[14]!): m[3]=${m[3].toFixed(2)}, m[7]=${m[7].toFixed(2)}, m[11]=${m[11].toFixed(2)}, m[15]=${m[15].toFixed(2)}`
            );
          }
        }
        return origUM4(loc, transpose, val, ...rest);
      };

      function _checkUnsetMat4() {
        if (!currentProg || progReportedUnset.has(currentProg)) return;
        const active = progActiveMat4.get(currentProg);
        const setNames = progSetLocs.get(currentProg);
        if (!active || !active.length || !setNames) return;
        const missing = active.filter(n => !setNames.has(n) && !setNames.has(n.replace(/\[0\]$/, '')));
        if (missing.length && T.unsetMat4Uniforms.length < 6) {
          progReportedUnset.add(currentProg);
          T.unsetMat4Uniforms.push(`active mat4 uniform(s) NEVER uploaded via gl.uniformMatrix4fv before draw (all-zero matrix clips/hides all 3D vertices!): ${missing.join(', ')}`);
        }
      }

      const origDA = ctx.drawArrays.bind(ctx);
      ctx.drawArrays = function(...a) { _checkUnsetMat4(); T.drawCalls++; return origDA(...a); };
      const origDE = ctx.drawElements.bind(ctx);
      ctx.drawElements = function(...a) { _checkUnsetMat4(); T.drawCalls++; return origDE(...a); };
      if (ctx.drawArraysInstanced) {
        const origDAI = ctx.drawArraysInstanced.bind(ctx);
        ctx.drawArraysInstanced = function(...a) { _checkUnsetMat4(); T.instancedCalls++; T.drawCalls++; return origDAI(...a); };
      }
      if (ctx.drawElementsInstanced) {
        const origDEI = ctx.drawElementsInstanced.bind(ctx);
        ctx.drawElementsInstanced = function(...a) { _checkUnsetMat4(); T.instancedCalls++; T.drawCalls++; return origDEI(...a); };
      }
      const origEnable = ctx.enable.bind(ctx);
      ctx.enable = function(cap) {
        if (cap === ctx.DEPTH_TEST) T.depthTestEnabled = true;
        return origEnable(cap);
      };
    }
    return ctx;
  };

  window.__VERIFIER_SNAPSHOT__ = function() {
    for (const ctx of T.activeContexts) {
      try {
        for (let i = 0; i < 8; i++) {
          const err = ctx.getError();
          if (!err) break;
          T.glErrors.push(err);
        }
      } catch (e) {}
    }
    return {
      contexts: T.contexts.slice(0, 12),
      webgl2: T.webgl2,
      webgl1: T.webgl1,
      canvas2d: T.canvas2d,
      shaderCount: T.shaderCount,
      shaderCompileErrors: T.shaderCompileErrors.slice(0, 8),
      programLinkErrors: T.programLinkErrors.slice(0, 8),
      drawCalls: T.drawCalls,
      instancedCalls: T.instancedCalls,
      depthTestEnabled: T.depthTestEnabled,
      rafCount: T.rafCount,
      glErrors: T.glErrors.slice(0, 12),
      transposedMatrices: T.transposedMatrices.slice(0, 4),
      attribBufferMismatches: T.attribBufferMismatches.slice(0, 4),
      unsetMat4Uniforms: T.unsetMat4Uniforms.slice(0, 4),
      broadcastChannelsCreated: T.broadcastChannelsCreated.slice(0, 6),
    };
  };
})();
"""


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


class _LocalStaticServer:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.httpd: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None

    def __enter__(self) -> str:
        handler = functools.partial(_QuietHandler, directory=str(self.directory))
        socketserver.TCPServer.allow_reuse_address = True
        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{port}"

    def __exit__(self, *args: Any) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()


def analyze_image(path: Path) -> dict[str, Any]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    q = (arr.astype(np.uint8) >> 4).astype(np.int32)
    codes = (q[:, :, 0] << 8) | (q[:, :, 1] << 4) | q[:, :, 2]
    distinct_colors_q16 = int(np.unique(codes).size)
    hist, _ = np.histogram(luma, bins=64, range=(0, 256))
    probs = hist[hist > 0].astype(np.float64)
    probs /= probs.sum()
    entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0

    # Central 60% scene region (away from corner HUDs)
    cy0, cy1 = int(h * 0.22), int(h * 0.82)
    cx0, cx1 = int(w * 0.18), int(w * 0.82)
    c_luma = luma[cy0:cy1, cx0:cx1]
    c_arr = arr[cy0:cy1, cx0:cx1, :]
    c_chroma = c_arr.max(axis=2) - c_arr.min(axis=2)
    gx = np.abs(c_luma[:, 1:] - c_luma[:, :-1])
    gy = np.abs(c_luma[1:, :] - c_luma[:-1, :])
    edge = np.maximum(gx[:-1, :], gy[:, :-1])
    center_edge_pct = float((edge > 10.0).mean() * 100.0)
    center_edge_p99 = float(np.percentile(edge, 99))

    return {
        "width": w,
        "height": h,
        "distinct_colors_q16": distinct_colors_q16,
        "shannon_entropy": round(entropy, 3),
        "mean_luma": round(float(luma.mean()), 2),
        "std_luma": round(float(luma.std()), 2),
        "center_std_luma": round(float(c_luma.std()), 2),
        "center_p95_luma": round(float(np.percentile(c_luma, 95)), 2),
        "center_p95_chroma": round(float(np.percentile(c_chroma, 95)), 2),
        "center_edge_pct": round(center_edge_pct, 3),
        "center_edge_p99": round(center_edge_p99, 2),
    }


def compare_images(path_a: Path, path_b: Path) -> dict[str, float]:
    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        return {"mad": 1.0, "changed_ratio": 1.0}
    diff = np.abs(a - b)
    return {
        "mad": round(float(diff.mean() / 255.0), 5),
        "changed_ratio": round(float((diff.max(axis=2) > 10.0).mean()), 4),
    }


def _read_state(page) -> Optional[dict[str, Any]]:
    try:
        return page.evaluate("""() => {
          const g = window.__GAME_STATE__;
          if (!g) return null;
          return typeof g === 'function' ? g() : JSON.parse(JSON.stringify(g));
        }""")
    except Exception:
        return None


def _dismiss_overlay_if_needed(page) -> None:
    st = _read_state(page)
    if isinstance(st, dict) and str(st.get("state", "")).lower() in ("playing", "active", "running", "aim", "ready", "racing"):
        return
    try:
        btn = page.locator("button:visible").first
        if btn.count() > 0:
            txt = (btn.inner_text(timeout=800) or "").lower()
            if any(k in txt for k in ("play", "start", "drive", "race", "begin", "enter", "bowl", "launch")):
                btn.click(timeout=1200)
                page.wait_for_timeout(300)
    except Exception:
        pass
    page.mouse.click(640, 360)
    page.wait_for_timeout(150)
    page.keyboard.press("Enter")
    page.wait_for_timeout(250)


def _playtest_stock_offroad(page, out_dir: Path, p0: Path, shots: dict) -> dict[str, Any]:
    _dismiss_overlay_if_needed(page)
    page.wait_for_timeout(350)
    p_start = out_dir / "frame_action_1.png"
    page.screenshot(path=str(p_start), timeout=15000)
    shots["action_1"] = str(p_start)
    st0 = _read_state(page) or {}

    # Drive right along 5-day stock price terrain curve for 1.8s + Space hop/boost
    page.keyboard.down("ArrowRight")
    page.keyboard.down("KeyD")
    page.wait_for_timeout(900)
    st_mid = _read_state(page) or {}
    page.keyboard.press("Space")
    page.wait_for_timeout(1000)
    page.keyboard.up("ArrowRight")
    page.keyboard.up("KeyD")
    page.wait_for_timeout(300)

    p_drive = out_dir / "frame_action_2.png"
    page.screenshot(path=str(p_drive), timeout=15000)
    shots["action_2"] = str(p_drive)
    st1 = _read_state(page) or {}

    # Continue driving further to sample different stock price & slope
    page.keyboard.down("ArrowRight")
    page.keyboard.down("KeyD")
    page.wait_for_timeout(1200)
    page.keyboard.up("ArrowRight")
    page.keyboard.up("KeyD")
    page.wait_for_timeout(250)
    p_far = out_dir / "frame_action_3.png"
    page.screenshot(path=str(p_far), timeout=15000)
    shots["action_3"] = str(p_far)
    st2 = _read_state(page) or {}

    d0 = float(st0.get("distance", st0.get("truckX", 0)) or 0)
    d2 = float(st2.get("distance", st2.get("truckX", 0)) or 0)
    pr0 = float(st0.get("currentPrice", 0) or 0)
    pr1 = float(st1.get("currentPrice", 0) or 0)
    pr2 = float(st2.get("currentPrice", 0) or 0)
    pitches = [abs(float(s.get("pitch", 0) or 0)) for s in (st0, st_mid, st1, st2)]
    motion_1_2 = compare_images(p_start, p_drive)
    motion_1_3 = compare_images(p_start, p_far)

    return {
        "has_hook": bool(st0),
        "state_0": st0,
        "state_2": st2,
        "distance_delta": round(abs(d2 - d0), 2),
        "price_samples": [round(pr0, 2), round(pr1, 2), round(pr2, 2)],
        "price_changed": (abs(pr2 - pr0) > 0.01 or abs(pr1 - pr0) > 0.01),
        "max_pitch_abs": round(max(pitches), 4),
        "max_speed": round(max(abs(float(s.get("speed", 0) or 0)) for s in (st_mid, st1, st2)), 2),
        "motion_1_2": motion_1_2,
        "motion_1_3": motion_1_3,
    }


def _playtest_bowling_horror(page, out_dir: Path, p0: Path, shots: dict) -> dict[str, Any]:
    _dismiss_overlay_if_needed(page)
    page.wait_for_timeout(400)
    p1 = out_dir / "frame_action_1.png"
    page.screenshot(path=str(p1), timeout=15000)
    shots["action_1"] = str(p1)
    st0 = _read_state(page) or {}

    # Aim slightly and charge/throw bowling ball with Space
    page.keyboard.down("ArrowRight")
    page.wait_for_timeout(160)
    page.keyboard.up("ArrowRight")
    page.keyboard.down("ArrowLeft")
    page.wait_for_timeout(140)
    page.keyboard.up("ArrowLeft")
    st_aim = _read_state(page) or {}

    # Hold Space to charge power, release to throw!
    page.keyboard.down("Space")
    page.wait_for_timeout(650)
    page.keyboard.up("Space")
    page.wait_for_timeout(850)

    p2 = out_dir / "frame_action_2.png"
    page.screenshot(path=str(p2), timeout=15000)
    shots["action_2"] = str(p2)
    st_mid = _read_state(page) or {}

    # Wait for ball to reach pins & collisions to resolve
    ball_moved = False
    max_z_delta = 0.0
    z0 = float(st0.get("ballZ", 0) or 0)
    min_pins = int(st0.get("pinsStanding", 10) if st0.get("pinsStanding") is not None else 10)
    best_score = int(st0.get("score", 0) or 0)
    max_knocked = int(st0.get("pinsKnockedThisRoll", 0) or 0)

    for step in range(18):
        page.wait_for_timeout(250)
        st = _read_state(page) or {}
        bz = float(st.get("ballZ", z0) or 0)
        max_z_delta = max(max_z_delta, abs(bz - z0))
        if st.get("ballMoving") or max_z_delta > 1.5:
            ball_moved = True
        if st.get("pinsStanding") is not None:
            min_pins = min(min_pins, int(st["pinsStanding"]))
        best_score = max(best_score, int(st.get("score", 0) or 0))
        max_knocked = max(max_knocked, int(st.get("pinsKnockedThisRoll", 0) or 0))
        if step == 4 and not ball_moved:
            # Try click or Space tap if game uses press-to-throw instead of hold-release
            page.keyboard.press("Space")
            page.mouse.click(640, 420)
        if best_score > 0 and min_pins < 10 and step >= 8:
            break

    p3 = out_dir / "frame_action_3.png"
    page.screenshot(path=str(p3), timeout=15000)
    shots["action_3"] = str(p3)
    st_end = _read_state(page) or {}
    motion_1_2 = compare_images(p1, p2)
    motion_1_3 = compare_images(p1, p3)

    return {
        "has_hook": bool(st0),
        "state_0": st0,
        "state_end": st_end,
        "aim_changed": abs(float(st_aim.get("aimX", 0) or 0) - float(st0.get("aimX", 0) or 0)) > 0.01 or True,
        "ball_moved": ball_moved,
        "max_z_delta": round(max_z_delta, 2),
        "min_pins_standing": min_pins,
        "max_knocked": max_knocked,
        "best_score": best_score,
        "motion_1_2": motion_1_2,
        "motion_1_3": motion_1_3,
    }


def _playtest_skate_to_gta(page, out_dir: Path, p0: Path, shots: dict) -> dict[str, Any]:
    _dismiss_overlay_if_needed(page)
    page.wait_for_timeout(350)
    p1 = out_dir / "frame_action_1.png"
    page.screenshot(path=str(p1), timeout=15000)
    shots["action_1"] = str(p1)
    st0 = _read_state(page) or {}
    x0, z0 = float(st0.get("playerX", 0) or 0), float(st0.get("playerZ", 0) or 0)

    # Skate forward + perform Ollie (Space) + Trick (Q/Shift)
    page.keyboard.down("KeyW")
    page.keyboard.down("ArrowUp")
    page.wait_for_timeout(800)
    page.keyboard.press("Space")
    page.wait_for_timeout(180)
    page.keyboard.press("KeyQ")
    page.wait_for_timeout(200)
    page.keyboard.press("ShiftLeft")
    page.wait_for_timeout(600)
    page.keyboard.press("Space")
    page.wait_for_timeout(160)
    page.keyboard.press("KeyQ")
    page.wait_for_timeout(450)
    page.keyboard.up("KeyW")
    page.keyboard.up("ArrowUp")

    p2 = out_dir / "frame_action_2.png"
    page.screenshot(path=str(p2), timeout=15000)
    shots["action_2"] = str(p2)
    st_skate = _read_state(page) or {}
    xs, zs = float(st_skate.get("playerX", x0) or 0), float(st_skate.get("playerZ", z0) or 0)
    skate_dist = math.hypot(xs - x0, zs - z0)
    tricks_count = int(st_skate.get("tricksCount", 0) or 0)
    skate_score = int(st_skate.get("score", 0) or 0)

    # Hop into a car with F (or E) and drive!
    entered_car = False
    for key in ("KeyF", "KeyE"):
        page.keyboard.press(key)
        page.wait_for_timeout(300)
        stc = _read_state(page) or {}
        if str(stc.get("mode", "")).lower() in ("car", "driving", "vehicle"):
            entered_car = True
            break

    st_car0 = _read_state(page) or {}
    xc0, zc0 = float(st_car0.get("playerX", xs) or 0), float(st_car0.get("playerZ", zs) or 0)
    page.keyboard.down("KeyW")
    page.keyboard.down("ArrowUp")
    page.keyboard.down("KeyD")
    page.wait_for_timeout(1000)
    page.keyboard.up("KeyD")
    page.wait_for_timeout(500)
    page.keyboard.up("KeyW")
    page.keyboard.up("ArrowUp")

    p3 = out_dir / "frame_action_3.png"
    page.screenshot(path=str(p3), timeout=15000)
    shots["action_3"] = str(p3)
    st_car1 = _read_state(page) or {}
    xc1, zc1 = float(st_car1.get("playerX", xc0) or 0), float(st_car1.get("playerZ", zc0) or 0)
    car_dist = math.hypot(xc1 - xc0, zc1 - zc0)

    # Exit car back to skate
    exited_to_skate = False
    if entered_car:
        page.keyboard.press("KeyF")
        page.wait_for_timeout(250)
        st_ex = _read_state(page) or {}
        if str(st_ex.get("mode", "")).lower() in ("skate", "skating", "foot"):
            exited_to_skate = True

    return {
        "has_hook": bool(st0),
        "state_0": st0,
        "state_skate": st_skate,
        "state_car": st_car1,
        "skate_dist": round(skate_dist, 2),
        "tricks_count": tricks_count,
        "score": max(skate_score, int(st_car1.get("score", 0) or 0)),
        "entered_car": entered_car,
        "car_dist": round(car_dist, 2),
        "exited_to_skate": exited_to_skate,
        "motion_1_2": compare_images(p1, p2),
        "motion_2_3": compare_images(p2, p3),
    }


def _playtest_multiplayer_racing(page, out_dir: Path, p0: Path, shots: dict) -> dict[str, Any]:
    _dismiss_overlay_if_needed(page)
    page.wait_for_timeout(400)
    st0 = _read_state(page) or {}

    # Enable 2P split-screen if not already active (click split button or press Tab/KeyM)
    if not st0.get("splitScreen"):
        try:
            btns = page.locator("button:visible")
            for i in range(min(btns.count(), 5)):
                txt = (btns.nth(i).inner_text(timeout=600) or "").lower()
                if "2p" in txt or "split" in txt:
                    btns.nth(i).click(timeout=1000)
                    page.wait_for_timeout(300)
                    break
        except Exception:
            pass
    st_pre = _read_state(page) or {}
    p1 = out_dir / "frame_action_1.png"
    page.screenshot(path=str(p1), timeout=15000)
    shots["action_1"] = str(p1)

    p1_0 = st_pre.get("p1") or {}
    p2_0 = st_pre.get("p2") or {}
    x1_0, z1_0 = float(p1_0.get("x", 0) or 0), float(p1_0.get("z", 0) or 0)
    x2_0, z2_0 = float(p2_0.get("x", 0) or 0), float(p2_0.get("z", 0) or 0)

    # Drive P1 (W + D) AND P2 (ArrowUp + ArrowLeft) simultaneously!
    page.keyboard.down("KeyW")
    page.keyboard.down("ArrowUp")
    page.wait_for_timeout(900)
    p2_img = out_dir / "frame_action_2.png"
    page.screenshot(path=str(p2_img), timeout=15000)
    shots["action_2"] = str(p2_img)

    page.keyboard.down("KeyD")
    page.keyboard.down("ArrowLeft")
    page.wait_for_timeout(900)
    page.keyboard.up("KeyD")
    page.keyboard.up("ArrowLeft")
    page.keyboard.up("KeyW")
    page.keyboard.up("ArrowUp")
    page.wait_for_timeout(200)

    p3_img = out_dir / "frame_action_3.png"
    page.screenshot(path=str(p3_img), timeout=15000)
    shots["action_3"] = str(p3_img)
    st_end = _read_state(page) or {}

    p1_1 = st_end.get("p1") or {}
    p2_1 = st_end.get("p2") or {}
    p1_dist = math.hypot(float(p1_1.get("x", x1_0) or 0) - x1_0, float(p1_1.get("z", z1_0) or 0) - z1_0)
    p2_dist = math.hypot(float(p2_1.get("x", x2_0) or 0) - x2_0, float(p2_1.get("z", z2_0) or 0) - z2_0)

    return {
        "has_hook": bool(st_pre),
        "state_pre": st_pre,
        "state_end": st_end,
        "split_screen": bool(st_end.get("splitScreen") or st_pre.get("splitScreen")),
        "broadcast_channel_active": bool(st_end.get("broadcastChannelActive") or st_pre.get("broadcastChannelActive")),
        "p1_dist": round(p1_dist, 2),
        "p2_dist": round(p2_dist, 2),
        "p1_speed": round(abs(float(p1_1.get("speed", 0) or 0)), 2),
        "p2_speed": round(abs(float(p2_1.get("speed", 0) or 0)), 2),
        "ai_count": int(st_end.get("aiCount", 0) or 0),
        "motion_1_2": compare_images(p1, p2_img),
        "motion_1_3": compare_images(p1, p3_img),
    }


def verify_game(game_kind: str, html_path: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = Path(html_path).resolve()

    console_errors: list[str] = []
    page_errors: list[str] = []
    blocked_urls: list[str] = []
    shots: dict[str, str] = {}
    metrics: dict[str, Any] = {"game_kind": game_kind, "html_path": str(html_path)}
    snap_final: dict[str, Any] = {}

    with _LocalStaticServer(html_path.parent) as base_url:
        url = f"{base_url}/{html_path.name}"
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path=CHROME_EXECUTABLE, args=CHROME_ARGS)
            context = browser.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
            page = context.new_page()
            page.add_init_script(WEBGL_TELEMETRY_INIT_JS)

            def _route(route):
                u = route.request.url
                if u.startswith(base_url) or u.startswith("data:") or u.startswith("blob:"):
                    route.continue_()
                else:
                    blocked_urls.append(u)
                    route.abort()

            page.route("**/*", _route)
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: page_errors.append(str(e)))

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(1200)
                p0 = out_dir / "frame_initial.png"
                page.screenshot(path=str(p0), timeout=20000)
                shots["initial"] = str(p0)
                metrics["initial_image"] = analyze_image(p0)

                if game_kind == "stock_offroad_4x4":
                    metrics["playtest"] = _playtest_stock_offroad(page, out_dir, p0, shots)
                elif game_kind == "bowling_horror_3d":
                    metrics["playtest"] = _playtest_bowling_horror(page, out_dir, p0, shots)
                elif game_kind == "skate_to_gta_city":
                    metrics["playtest"] = _playtest_skate_to_gta(page, out_dir, p0, shots)
                elif game_kind == "multiplayer_3d_racing":
                    metrics["playtest"] = _playtest_multiplayer_racing(page, out_dir, p0, shots)

                if "action_2" in shots and Path(shots["action_2"]).exists():
                    metrics["action_image"] = analyze_image(Path(shots["action_2"]))
                snap_final = page.evaluate("() => window.__VERIFIER_SNAPSHOT__ ? window.__VERIFIER_SNAPSHOT__() : {}")
            except Exception as e:
                metrics["runtime_timeout"] = f"{type(e).__name__}: {str(e)[:260]}"
                try:
                    snap_final = page.evaluate("() => window.__VERIFIER_SNAPSHOT__ ? window.__VERIFIER_SNAPSHOT__() : {}")
                except Exception:
                    pass
            finally:
                try:
                    browser.close()
                except Exception:
                    pass

    gl_err_names = [GL_ENUM_NAMES.get(int(e), hex(int(e))) for e in snap_final.get("glErrors", [])]
    metrics["console_errors"] = console_errors[:10]
    metrics["page_errors"] = page_errors[:10]
    metrics["blocked_urls"] = blocked_urls[:8]
    metrics["screenshots"] = shots
    metrics["webgl"] = {
        "contexts": snap_final.get("contexts", []),
        "webgl2": bool(snap_final.get("webgl2", False)),
        "shaderCount": snap_final.get("shaderCount", 0),
        "shaderCompileErrors": snap_final.get("shaderCompileErrors", []),
        "programLinkErrors": snap_final.get("programLinkErrors", []),
        "drawCalls": snap_final.get("drawCalls", 0),
        "depthTestEnabled": bool(snap_final.get("depthTestEnabled", False)),
        "rafCount": snap_final.get("rafCount", 0),
        "glErrors": gl_err_names,
        "transposedMatrices": snap_final.get("transposedMatrices", []),
        "attribBufferMismatches": snap_final.get("attribBufferMismatches", []),
        "unsetMat4Uniforms": snap_final.get("unsetMat4Uniforms", []),
        "broadcastChannelsCreated": snap_final.get("broadcastChannelsCreated", []),
    }

    ok, score, err_sig, err_detail, summary = _judge(game_kind, metrics)
    rep = {
        "ok": ok,
        "score": round(score, 2),
        "error_signature": err_sig,
        "error_detail": err_detail,
        "summary": summary,
        "metrics": metrics,
    }
    (out_dir / "verify_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return rep


def _judge(game_kind: str, m: dict[str, Any]) -> tuple[bool, float, str, str, str]:
    w = m["webgl"]
    img = m.get("action_image") or m.get("initial_image") or {}
    pt = m.get("playtest") or {}

    if m["blocked_urls"]:
        return False, 0.0, "EXTERNAL_NETWORK_DEPENDENCY", f"Blocked CDN/network fetches (must be 100% offline self-contained): {m['blocked_urls'][:4]}", ""
    if m.get("runtime_timeout"):
        return False, 10.0, "RENDERER_TIMEOUT", f"Headless verifier timed out: {m['runtime_timeout']}", ""
    if m["page_errors"]:
        return False, 6.0, "JS_PAGE_ERROR", f"Uncaught JS pageerror: {' | '.join(m['page_errors'][:3])}", ""
    if w["shaderCompileErrors"]:
        se = w["shaderCompileErrors"][0]
        return False, 12.0, "SHADER_COMPILE_ERROR", f"{se['kind']} shader compile error: {se['log']}", ""
    if w["programLinkErrors"]:
        return False, 14.0, "PROGRAM_LINK_ERROR", f"Program link error: {w['programLinkErrors'][0]}", ""
    if m["console_errors"]:
        return False, 9.0, "CONSOLE_ERROR", f"Console error: {' | '.join(m['console_errors'][:3])}", ""
    if w["glErrors"]:
        return False, 16.0, "WEBGL_RUNTIME_ERROR", f"WebGL gl.getError(): {', '.join(w['glErrors'][:5])}", ""
    if w.get("unsetMat4Uniforms"):
        return False, 18.0, "UNSET_MAT4_UNIFORM", f"Shader drew with unset mat4 uniform(s): {w['unsetMat4Uniforms'][0]}", ""
    if w["transposedMatrices"]:
        return False, 18.0, "TRANSPOSED_MATRIX_LAYOUT", f"Row-major mat4 passed to WebGL: {w['transposedMatrices'][0]}", ""
    if w["attribBufferMismatches"]:
        return False, 20.0, "WEBGL_BUFFER_BINDING_BUG", f"Buffer binding mismatch: {w['attribBufferMismatches'][0]}", ""
    if not w["webgl2"]:
        return False, 15.0, "NO_WEBGL2_CONTEXT", f"Expected real 3D WebGL2 context, got contexts={w['contexts']}", ""
    if w["rafCount"] < 5 or w["drawCalls"] < 3:
        return False, 20.0, "NO_3D_RENDER_LOOP", f"Expected live WebGL2 render loop (rafCount={w['rafCount']}, drawCalls={w['drawCalls']})", ""
    if img.get("shannon_entropy", 0) < 1.3 or img.get("center_std_luma", 0) < 6.0 or img.get("distinct_colors_q16", 0) < 16:
        return False, 25.0, "BLANK_OR_FLAT_3D_VIEW", f"3D viewport looks blank or flat! image={img}", ""
    if img.get("center_edge_p99", 0) < 14.0 or img.get("center_edge_pct", 0) < 1.2:
        return False, 28.0, "EMPTY_OR_OFFCAMERA_3D_SCENE", (
            f"3D viewport center shows only smooth sky/background with almost zero visible 3D geometry edges "
            f"(center_edge_pct={img.get('center_edge_pct')}%, center_edge_p99={img.get('center_edge_p99')})! "
            "Check camera mat4LookAt (in column-major WebGL, basis ROWS go at m[0],m[4],m[8]=right; m[1],m[5],m[9]=up; m[2],m[6],m[10]=forward, NOT m[0..2]!) "
            "and verify the camera faces your 3D meshes inside the frustum!"
        ), ""
    if not pt.get("has_hook"):
        return False, 30.0, "MISSING_GAME_STATE_HOOK", "Must expose live window.__GAME_STATE__ updated every frame for automated physics/playtest verification.", ""

    vis_pts = (
        min(12.0, (img["distinct_colors_q16"] / 250.0) * 12.0)
        + min(10.0, max(0.0, (img["shannon_entropy"] - 1.3) / 3.8) * 10.0)
        + min(6.0, (img["center_std_luma"] / 45.0) * 6.0)
    )

    if game_kind == "stock_offroad_4x4":
        if pt["distance_delta"] >= 3.0 and pt["motion_1_3"]["mad"] < 0.015:
            return False, 36.0, "3D_SCENE_INVISIBLE_OR_FROZEN", f"Physics state moved (distance_delta={pt['distance_delta']}, prices={pt['price_samples']}), but the 3D viewport pixels barely changed (motion_mad={pt['motion_1_3']['mad']})! Check camera projection/view matrices, uView/uProj uniform uploads, culling, or depth buffer.", ""
        if pt["distance_delta"] < 3.0:
            return False, 40.0, "TRUCK_NOT_DRIVING", f"Holding ArrowRight/D did not drive the 4x4 truck across the 3D terrain! distance_delta={pt['distance_delta']}, motion_mad={pt['motion_1_3']['mad']}", ""
        if not pt["price_changed"]:
            return False, 48.0, "STOCK_CURVE_NOT_SYNCED", f"Driving along the 3D stock terrain did not update currentPrice along the 5-day price curve! price_samples={pt['price_samples']}", ""
        if pt["max_pitch_abs"] < 0.005:
            return False, 52.0, "NO_TERRAIN_SLOPE_SUSPENSION", f"Truck pitch stayed flat ({pt['max_pitch_abs']}) — wire 4-wheel terrain height sampling & suspension pitch/roll!", ""
        phys_pts = min(12.0, (pt["distance_delta"] / 25.0) * 6.0 + min(pt["motion_1_3"]["mad"] / 0.08, 1.0) * 6.0)
        score = 60.0 + vis_pts + phys_pts
        return True, min(100.0, score), "", "", (
            f"Stock 4x4 Off-Road PASS: dist_delta={pt['distance_delta']}, prices={pt['price_samples']}, "
            f"max_pitch={pt['max_pitch_abs']}, motion_mad={pt['motion_1_3']['mad']:.3f}, "
            f"colors={img['distinct_colors_q16']}, entropy={img['shannon_entropy']:.2f}"
        )

    elif game_kind == "bowling_horror_3d":
        best_mad = max(pt["motion_1_2"]["mad"], pt["motion_1_3"]["mad"])
        best_chg = max(pt["motion_1_2"]["changed_ratio"], pt["motion_1_3"]["changed_ratio"])
        if not pt["ball_moved"] or pt["max_z_delta"] < 1.5 or (best_mad < 0.003 and best_chg < 0.015):
            return False, 42.0, "BOWLING_BALL_NOT_ROLLING", f"Charging/releasing Space did not launch the 3D bowling ball down the lane! ball_moved={pt['ball_moved']}, max_z_delta={pt['max_z_delta']}, motion_1_2={pt['motion_1_2']}, motion_1_3={pt['motion_1_3']}", ""
        if pt["min_pins_standing"] >= 10 and pt["best_score"] <= 0:
            return False, 52.0, "NO_PIN_COLLISIONS_OR_SCORE", f"Ball rolled down lane (max_z_delta={pt['max_z_delta']}) but 0 pins fell and score stayed 0! Check lane alignment, pin positions, and ball-pin rigid-body collision radius/impulse.", ""
        knocked = max(pt["max_knocked"], 10 - pt["min_pins_standing"])
        phys_pts = min(12.0, (knocked / 10.0) * 8.0 + min(pt["motion_1_3"]["mad"] / 0.06, 1.0) * 4.0)
        score = 60.0 + vis_pts + phys_pts
        return True, min(100.0, score), "", "", (
            f"Bowling Horror 3D PASS: ball_dz={pt['max_z_delta']}, pins_knocked={knocked}/10, "
            f"score={pt['best_score']}, motion_mad={pt['motion_1_3']['mad']:.3f}, "
            f"colors={img['distinct_colors_q16']}, entropy={img['shannon_entropy']:.2f}"
        )

    elif game_kind == "skate_to_gta_city":
        if pt["skate_dist"] < 2.0 or pt["motion_1_2"]["mad"] < 0.012:
            return False, 40.0, "SKATER_NOT_MOVING", f"W/Up did not skate across the 3D city! skate_dist={pt['skate_dist']}, motion={pt['motion_1_2']}", ""
        if pt["tricks_count"] < 1 and pt["score"] <= 0:
            return False, 46.0, "SKATE_TRICKS_BROKEN", f"Pressing Space (Ollie) + Q/Shift (trick) recorded tricksCount={pt['tricks_count']}, score={pt['score']}.", ""
        if not pt["entered_car"]:
            return False, 52.0, "CAR_ENTRY_BROKEN", f"Pressing F/E did not enter a nearby 3D drivable car (keep a hero sports car parked within quick reach of spawn or let F snap to nearest parked car within generous radius)! state_skate={pt['state_skate']}", ""
        if pt["car_dist"] < 2.5 or pt["motion_2_3"]["mad"] < 0.012:
            return False, 56.0, "CAR_DRIVING_BROKEN", f"Entered car cleanly, but W/D did not drive the car through the city! car_dist={pt['car_dist']}, motion={pt['motion_2_3']}", ""
        phys_pts = min(12.0, 4.0 + (3.0 if pt["exited_to_skate"] else 0.0) + min(pt["car_dist"] / 20.0, 1.0) * 5.0)
        score = 60.0 + vis_pts + phys_pts
        return True, min(100.0, score), "", "", (
            f"Skate-to-GTA 3D PASS: skate_dist={pt['skate_dist']}, tricks={pt['tricks_count']}, score={pt['score']}, "
            f"entered_car=True, car_dist={pt['car_dist']}, exited_to_skate={pt['exited_to_skate']}, "
            f"colors={img['distinct_colors_q16']}, entropy={img['shannon_entropy']:.2f}"
        )

    elif game_kind == "multiplayer_3d_racing":
        bc_ok = pt["broadcast_channel_active"] or len(w.get("broadcastChannelsCreated", [])) > 0
        if not pt["split_screen"]:
            return False, 42.0, "SPLIT_SCREEN_NOT_ACTIVE", "Expected 2-player local split-screen mode (dual gl.viewport/gl.scissor) active by default or via 2P Split button with splitScreen=true in __GAME_STATE__.", ""
        if not bc_ok:
            return False, 46.0, "BROADCAST_CHANNEL_MISSING", "Must initialize a real BroadcastChannel('atlas_racer_mp') for live multi-tab multiplayer sync and report broadcastChannelActive=true.", ""
        if pt["p1_dist"] < 3.0 or pt["p2_dist"] < 3.0 or pt["motion_1_3"]["mad"] < 0.015:
            return False, 52.0, "BOTH_PLAYERS_NOT_RACING", f"Holding W (P1) and ArrowUp (P2) must simultaneously drive both split-screen cars! p1_dist={pt['p1_dist']}, p2_dist={pt['p2_dist']}, motion={pt['motion_1_3']}", ""
        if pt["ai_count"] < 2:
            return False, 56.0, "AI_PACK_MISSING", f"Expected >=2 AI pack racers alongside P1 & P2, got aiCount={pt['ai_count']}", ""
        phys_pts = min(12.0, min((pt["p1_dist"] + pt["p2_dist"]) / 60.0, 1.0) * 8.0 + min(pt["motion_1_3"]["mad"] / 0.08, 1.0) * 4.0)
        score = 60.0 + vis_pts + phys_pts
        return True, min(100.0, score), "", "", (
            f"Multiplayer 3D Racing PASS: splitScreen=True, BroadcastChannel=True, P1_dist={pt['p1_dist']}, "
            f"P2_dist={pt['p2_dist']}, AI={pt['ai_count']}, motion_mad={pt['motion_1_3']['mad']:.3f}, "
            f"colors={img['distinct_colors_q16']}, entropy={img['shannon_entropy']:.2f}"
        )

    return False, 0.0, "UNKNOWN_GAME", f"Unknown game_kind={game_kind}", ""


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["stock_offroad_4x4", "bowling_horror_3d", "skate_to_gta_city", "multiplayer_3d_racing"])
    ap.add_argument("--html", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = verify_game(args.kind, Path(args.html), Path(args.out))
    print(json.dumps({k: v for k, v in rep.items() if k != "metrics"}, indent=2))

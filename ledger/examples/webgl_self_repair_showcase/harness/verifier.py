#!/usr/bin/env python3
"""Headless Playwright Chromium-1248 WebGL2 + Canvas + Playtest Verifier for Switchboard #848.

Provides non-invasive runtime telemetry for:
  1. gaussian_splats       : WebGL2 3D Gaussian Splatting (instanced quads, covariance/conic math,
                             blend/depthMask state, dynamic depth-sort uploads, orbit camera,
                             front/back occlusion sanity, pixel richness)
  2. articulated_creature  : 3D hierarchical articulated creature (WebGL2, joint hierarchy,
                             continuous kinematics motion, interactive orbit/controls, shading)
  3. atari_arcade          : Playable retro Atari arcade game (live keyboard input verification,
                             aiming + shooting + scoring, real collision -> life loss -> gameover
                             -> clean restart state machine, visual polish)
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
from dataclasses import dataclass, field, asdict
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
    0x0000: "ZERO",
    0x0001: "ONE",
    0x0300: "SRC_COLOR",
    0x0301: "ONE_MINUS_SRC_COLOR",
    0x0302: "SRC_ALPHA",
    0x0303: "ONE_MINUS_SRC_ALPHA",
    0x0304: "DST_ALPHA",
    0x0305: "ONE_MINUS_DST_ALPHA",
    0x0306: "DST_COLOR",
    0x0307: "ONE_MINUS_DST_COLOR",
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
    shaderSources: [],
    shaderCompileErrors: [],
    programLinkErrors: [],
    instancedCalls: 0,
    maxInstanceCount: 0,
    totalInstancesDrawn: 0,
    drawCalls: 0,
    blendEnabled: false,
    depthTestEnabled: false,
    depthMaskFalseCalls: 0,
    depthMaskTrueCalls: 0,
    blendFuncs: [],
    bufferUploads: 0,
    bufferSubUploads: 0,
    bufferUploadBytes: 0,
    rafCount: 0,
    glErrors: [],
    activeContexts: [],
    vaosCreated: 0,
    vaoBinds: 0,
    attribBufferMismatches: [],
    transposedMatrices: [],
    perVertexInstancedConflicts: [],
  };
  window.__VERIFIER_TELEMETRY__ = T;

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
    let patchedAttrs = attrs;
    if (t === 'webgl2' || t === 'webgl' || t === 'experimental-webgl') {
      patchedAttrs = Object.assign({}, attrs || {}, { preserveDrawingBuffer: true });
    }
    const ctx = origGetContext.call(this, type, patchedAttrs);
    if (!ctx) return ctx;
    T.contexts.push(t);
    if (t === 'webgl2') T.webgl2 = true;
    else if (t === 'webgl' || t === 'experimental-webgl') T.webgl1 = true;
    else if (t === '2d') T.canvas2d = true;

    if ((t === 'webgl2' || t === 'webgl' || t === 'experimental-webgl') && !ctx.__verifierWrapped) {
      ctx.__verifierWrapped = true;
      T.activeContexts.push(ctx);
      const shaderMap = new WeakMap();

      const origShaderSource = ctx.shaderSource.bind(ctx);
      ctx.shaderSource = function(shader, src) {
        const stype = ctx.getShaderParameter(shader, ctx.SHADER_TYPE);
        const kind = stype === ctx.VERTEX_SHADER ? 'vertex' : (stype === ctx.FRAGMENT_SHADER ? 'fragment' : 'unknown');
        shaderMap.set(shader, { kind, src: String(src || '') });
        T.shaderSources.push({ kind, src: String(src || '') });
        return origShaderSource(shader, src);
      };

      const origCompile = ctx.compileShader.bind(ctx);
      ctx.compileShader = function(shader) {
        const res = origCompile(shader);
        if (!ctx.getShaderParameter(shader, ctx.COMPILE_STATUS)) {
          const info = shaderMap.get(shader) || { kind: 'unknown', src: '' };
          T.shaderCompileErrors.push({
            kind: info.kind,
            log: String(ctx.getShaderInfoLog(shader) || 'unknown compile error'),
          });
        }
        return res;
      };

      const origLink = ctx.linkProgram.bind(ctx);
      ctx.linkProgram = function(prog) {
        const res = origLink(prog);
        if (!ctx.getProgramParameter(prog, ctx.LINK_STATUS)) {
          T.programLinkErrors.push(String(ctx.getProgramInfoLog(prog) || 'unknown link error'));
        }
        return res;
      };

      if (ctx.drawArraysInstanced) {
        const origDAI = ctx.drawArraysInstanced.bind(ctx);
        ctx.drawArraysInstanced = function(mode, first, count, instanceCount) {
          T.instancedCalls++;
          T.drawCalls++;
          const ic = Number(instanceCount) || 0;
          if (ic > T.maxInstanceCount) T.maxInstanceCount = ic;
          T.totalInstancesDrawn += ic;
          if (T.perVertexInstancedConflicts.length === 0) {
            const instBufs = new Set();
            for (const [idx, div] of attribDivisors.entries()) {
              if (div > 0 && attribBufs.get(idx)) instBufs.add(attribBufs.get(idx));
            }
            for (const [idx, buf] of attribBufs.entries()) {
              const div = attribDivisors.get(idx) || 0;
              if (div === 0 && buf && instBufs.has(buf)) {
                T.perVertexInstancedConflicts.push(
                  `per-vertex attrib(${idx}) (divisor=0) shares the exact same ARRAY_BUFFER (${bufSizes.get(buf) || 0} bytes) as instanced (divisor=1) attributes! Bind the static quad corner buffer BEFORE calling vertexAttribPointer(${idx}, ...).`
                );
              }
            }
          }
          return origDAI(mode, first, count, instanceCount);
        };
      }

      if (ctx.drawElementsInstanced) {
        const origDEI = ctx.drawElementsInstanced.bind(ctx);
        ctx.drawElementsInstanced = function(mode, count, type, offset, instanceCount) {
          T.instancedCalls++;
          T.drawCalls++;
          const ic = Number(instanceCount) || 0;
          if (ic > T.maxInstanceCount) T.maxInstanceCount = ic;
          T.totalInstancesDrawn += ic;
          return origDEI(mode, count, type, offset, instanceCount);
        };
      }

      const origDA = ctx.drawArrays.bind(ctx);
      ctx.drawArrays = function(mode, first, count) {
        T.drawCalls++;
        return origDA(mode, first, count);
      };

      const origDE = ctx.drawElements.bind(ctx);
      ctx.drawElements = function(mode, count, type, offset) {
        T.drawCalls++;
        return origDE(mode, count, type, offset);
      };

      const origEnable = ctx.enable.bind(ctx);
      ctx.enable = function(cap) {
        if (cap === ctx.BLEND) T.blendEnabled = true;
        if (cap === ctx.DEPTH_TEST) T.depthTestEnabled = true;
        return origEnable(cap);
      };

      const origDepthMask = ctx.depthMask.bind(ctx);
      ctx.depthMask = function(flag) {
        if (flag) T.depthMaskTrueCalls++;
        else T.depthMaskFalseCalls++;
        return origDepthMask(flag);
      };

      const origBlendFunc = ctx.blendFunc.bind(ctx);
      ctx.blendFunc = function(s, d) {
        if (T.blendFuncs.length < 24) T.blendFuncs.push(['blendFunc', s, d]);
        return origBlendFunc(s, d);
      };

      const origBlendFuncSep = ctx.blendFuncSeparate.bind(ctx);
      ctx.blendFuncSeparate = function(sr, dr, sa, da) {
        if (T.blendFuncs.length < 24) T.blendFuncs.push(['blendFuncSeparate', sr, dr, sa, da]);
        return origBlendFuncSep(sr, dr, sa, da);
      };

      const bufSizes = new WeakMap();
      const attribBufs = new Map();
      const attribDivisors = new Map();
      let boundArrayBuf = null;

      if (ctx.vertexAttribDivisor) {
        const origVAD = ctx.vertexAttribDivisor.bind(ctx);
        ctx.vertexAttribDivisor = function(index, divisor) {
          attribDivisors.set(index, divisor);
          return origVAD(index, divisor);
        };
      }

      if (ctx.createVertexArray) {
        const origCVA = ctx.createVertexArray.bind(ctx);
        ctx.createVertexArray = function() {
          T.vaosCreated++;
          return origCVA();
        };
      }
      if (ctx.bindVertexArray) {
        const origBVA = ctx.bindVertexArray.bind(ctx);
        ctx.bindVertexArray = function(v) {
          if (v) T.vaoBinds++;
          return origBVA(v);
        };
      }

      const origBindBuf = ctx.bindBuffer.bind(ctx);
      ctx.bindBuffer = function(target, buf) {
        if (target === ctx.ARRAY_BUFFER) boundArrayBuf = buf;
        return origBindBuf(target, buf);
      };

      const origVAP = ctx.vertexAttribPointer.bind(ctx);
      ctx.vertexAttribPointer = function(index, size, type, norm, stride, offset) {
        attribBufs.set(index, boundArrayBuf);
        const bsz = boundArrayBuf ? (bufSizes.get(boundArrayBuf) || 0) : 0;
        if (stride > 0 && bsz > 0 && bsz < stride * 2 && T.attribBufferMismatches.length < 8) {
          T.attribBufferMismatches.push(`attrib(${index}) stride=${stride} offset=${offset} bound to tiny ARRAY_BUFFER (${bsz} bytes)`);
        }
        return origVAP(index, size, type, norm, stride, offset);
      };

      const origUM4 = ctx.uniformMatrix4fv.bind(ctx);
      ctx.uniformMatrix4fv = function(loc, transpose, val, ...rest) {
        if (!transpose && val && val.length >= 16 && T.transposedMatrices.length < 6) {
          const m = val;
          // Standard column-major affine/view matrix has m[15]~=1, bottom row [3,7,11]~=0, translation in [12,13,14].
          const bottomRowMag = Math.abs(m[3]) + Math.abs(m[7]) + Math.abs(m[11]);
          const col3Mag = Math.abs(m[12]) + Math.abs(m[13]) + Math.abs(m[14]);
          if (Math.abs(m[15] - 1.0) < 1e-3 && bottomRowMag > 0.05 && col3Mag < 1e-3) {
            T.transposedMatrices.push(
              `mat4 has m[15]=1 with translation in row3 [m[3]=${m[3].toFixed(2)}, m[7]=${m[7].toFixed(2)}, m[11]=${m[11].toFixed(2)}] and zero col3 [m[12..14]=0] (row-major layout passed to column-major WebGL!)`
            );
          }
        }
        return origUM4(loc, transpose, val, ...rest);
      };

      const origBufferData = ctx.bufferData.bind(ctx);
      ctx.bufferData = function(...args) {
        T.bufferUploads++;
        let blen = 0;
        if (typeof args[1] === 'number') blen = args[1];
        else if (args[1] && typeof args[1].byteLength === 'number') blen = args[1].byteLength;
        if (blen > 0) {
          T.bufferUploadBytes += blen;
          if (args[0] === ctx.ARRAY_BUFFER && boundArrayBuf) bufSizes.set(boundArrayBuf, blen);
        }
        return origBufferData(...args);
      };

      const origBufferSubData = ctx.bufferSubData.bind(ctx);
      ctx.bufferSubData = function(...args) {
        T.bufferSubUploads++;
        if (args[2] && typeof args[2].byteLength === 'number') T.bufferUploadBytes += args[2].byteLength;
        return origBufferSubData(...args);
      };
    }
    return ctx;
  };

  window.__VERIFIER_SNAPSHOT__ = function() {
    for (const ctx of T.activeContexts) {
      try {
        let guard = 0;
        while (guard < 8) {
          const err = ctx.getError();
          if (!err) break;
          T.glErrors.push(err);
          guard++;
        }
      } catch (e) {}
    }
    return {
      contexts: T.contexts.slice(0, 16),
      webgl2: T.webgl2,
      webgl1: T.webgl1,
      canvas2d: T.canvas2d,
      shaderCount: T.shaderSources.length,
      shaderSources: T.shaderSources.slice(0, 12),
      shaderCompileErrors: T.shaderCompileErrors.slice(0, 12),
      programLinkErrors: T.programLinkErrors.slice(0, 12),
      instancedCalls: T.instancedCalls,
      maxInstanceCount: T.maxInstanceCount,
      totalInstancesDrawn: T.totalInstancesDrawn,
      drawCalls: T.drawCalls,
      blendEnabled: T.blendEnabled,
      depthTestEnabled: T.depthTestEnabled,
      depthMaskFalseCalls: T.depthMaskFalseCalls,
      depthMaskTrueCalls: T.depthMaskTrueCalls,
      blendFuncs: T.blendFuncs.slice(0, 12),
      bufferUploads: T.bufferUploads,
      bufferSubUploads: T.bufferSubUploads,
      bufferUploadBytes: T.bufferUploadBytes,
      rafCount: T.rafCount,
      glErrors: T.glErrors.slice(0, 16),
      vaosCreated: T.vaosCreated,
      vaoBinds: T.vaoBinds,
      attribBufferMismatches: T.attribBufferMismatches.slice(0, 8),
      transposedMatrices: T.transposedMatrices.slice(0, 6),
      perVertexInstancedConflicts: T.perVertexInstancedConflicts.slice(0, 6),
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
        self.port: int = 0

    def __enter__(self) -> str:
        handler = functools.partial(_QuietHandler, directory=str(self.directory))
        socketserver.TCPServer.allow_reuse_address = True
        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.port}"

    def __exit__(self, *args: Any) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()


def analyze_image(path: Path) -> dict[str, Any]:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    h, w, _ = arr.shape
    border = np.concatenate(
        [
            arr[:10, :, :].reshape(-1, 3),
            arr[-10:, :, :].reshape(-1, 3),
            arr[:, :10, :].reshape(-1, 3),
            arr[:, -10:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border, axis=0)
    dist = np.abs(arr - bg[None, None, :]).sum(axis=2)
    fg_mask = dist > 24.0
    non_bg_ratio = float(fg_mask.mean())

    cy0, cy1 = int(h * 0.2), int(h * 0.8)
    cx0, cx1 = int(w * 0.2), int(w * 0.8)
    center_non_bg_ratio = float(fg_mask[cy0:cy1, cx0:cx1].mean())
    center_rgb = arr[cy0:cy1, cx0:cx1, :].mean(axis=(0, 1))

    if fg_mask.sum() > 20:
        ys, xs = np.nonzero(fg_mask)
        y0, y1 = np.percentile(ys, [2, 98])
        x0, x1 = np.percentile(xs, [2, 98])
        bbox_area_ratio = float(max(0.0, (y1 - y0) * (x1 - x0)) / (h * w))
    else:
        bbox_area_ratio = 0.0

    # Quantized distinct colors (16 levels per channel)
    q = (arr.astype(np.uint8) >> 4).astype(np.int32)
    codes = (q[:, :, 0] << 8) | (q[:, :, 1] << 4) | q[:, :, 2]
    distinct_colors_q16 = int(np.unique(codes).size)

    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    hist, _ = np.histogram(luma, bins=64, range=(0, 256))
    probs = hist[hist > 0].astype(np.float64)
    probs /= probs.sum()
    shannon_entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0

    # Saturated white blowout check & corner overdraw check (catches blown-up full-screen splats)
    near_white_ratio = float(((arr[:, :, 0] > 248) & (arr[:, :, 1] > 248) & (arr[:, :, 2] > 248)).mean())
    corners = np.concatenate(
        [
            luma[:40, :40].ravel(),
            luma[:40, -40:].ravel(),
            luma[-40:, :40].ravel(),
            luma[-40:, -40:].ravel(),
        ]
    )
    corner_mean_luma = float(corners.mean())
    corner_std_luma = float(corners.std())

    # Central scene region excluding top-left HUD & bottom hint strip
    scene_arr = arr[80 : h - 80, 320 : w - 80, :]
    scene_luma = 0.299 * scene_arr[:, :, 0] + 0.587 * scene_arr[:, :, 1] + 0.114 * scene_arr[:, :, 2]
    scene_chroma = scene_arr.max(axis=2) - scene_arr.min(axis=2)
    scene_std_luma = float(scene_luma.std())
    scene_p95_luma = float(np.percentile(scene_luma, 95))
    scene_p99_luma = float(np.percentile(scene_luma, 99))
    scene_p95_chroma = float(np.percentile(scene_chroma, 95))
    # Compare top edge against top-right sky corner (not bottom floor median!)
    sky_bg = np.median(arr[:16, w - 60 : w - 8, :].reshape(-1, 3), axis=0)
    top_strip_dist = np.abs(arr[:16, 360 : w - 40, :] - sky_bg[None, None, :]).sum(axis=2)
    top_edge_fg = float((top_strip_dist > 28.0).mean())

    return {
        "width": w,
        "height": h,
        "bg_rgb": [round(float(x), 1) for x in bg],
        "center_rgb": [round(float(x), 1) for x in center_rgb],
        "non_bg_ratio": round(non_bg_ratio, 4),
        "center_non_bg_ratio": round(center_non_bg_ratio, 4),
        "bbox_area_ratio": round(bbox_area_ratio, 4),
        "distinct_colors_q16": distinct_colors_q16,
        "shannon_entropy": round(shannon_entropy, 3),
        "mean_luma": round(float(luma.mean()), 2),
        "std_luma": round(float(luma.std()), 2),
        "near_white_ratio": round(near_white_ratio, 4),
        "corner_mean_luma": round(corner_mean_luma, 2),
        "corner_std_luma": round(corner_std_luma, 2),
        "scene_std_luma": round(scene_std_luma, 2),
        "scene_p95_luma": round(scene_p95_luma, 2),
        "scene_p99_luma": round(scene_p99_luma, 2),
        "scene_p95_chroma": round(scene_p95_chroma, 2),
        "top_edge_fg": round(top_edge_fg, 4),
    }


def compare_images(path_a: Path, path_b: Path) -> dict[str, float]:
    a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32)
    b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        return {"mad": 1.0, "changed_ratio": 1.0}
    diff = np.abs(a - b)
    mad = float(diff.mean() / 255.0)
    changed_ratio = float((diff.max(axis=2) > 12.0).mean())
    return {"mad": round(mad, 5), "changed_ratio": round(changed_ratio, 4)}


def format_blend_funcs(raw: list[Any]) -> list[str]:
    out = []
    for entry in raw:
        name = entry[0]
        args = [GL_ENUM_NAMES.get(int(v), hex(int(v))) for v in entry[1:]]
        out.append(f"{name}({', '.join(args)})")
    return out


def verify_html(
    target_kind: str,
    html_path: Path,
    out_dir: Path,
    viewport: tuple[int, int] = (1280, 720),
) -> dict[str, Any]:
    """Run full headless Chromium-1248 verification for target_kind and return structured report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = Path(html_path).resolve()
    html_text = html_path.read_text(encoding="utf-8", errors="replace")

    console_errors: list[str] = []
    console_warnings: list[str] = []
    page_errors: list[str] = []
    blocked_external_urls: list[str] = []
    screenshots: dict[str, str] = {}
    metrics: dict[str, Any] = {"target_kind": target_kind, "html_path": str(html_path)}

    with _LocalStaticServer(html_path.parent) as base_url:
        url = f"{base_url}/{html_path.name}"
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                executable_path=CHROME_EXECUTABLE,
                args=CHROME_ARGS,
            )
            context = browser.new_context(
                viewport={"width": viewport[0], "height": viewport[1]},
                device_scale_factor=1,
            )
            page = context.new_page()
            page.add_init_script(WEBGL_TELEMETRY_INIT_JS)

            def _route(route):
                req_url = route.request.url
                if req_url.startswith(base_url) or req_url.startswith("data:") or req_url.startswith("blob:"):
                    route.continue_()
                else:
                    blocked_external_urls.append(req_url)
                    route.abort()

            page.route("**/*", _route)
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text)
                if msg.type == "error"
                else (console_warnings.append(msg.text) if msg.type == "warning" else None),
            )
            page.on("pageerror", lambda err: page_errors.append(str(err)))

            timed_out_stage = ""
            snap_initial: dict[str, Any] = {}
            snap_final: dict[str, Any] = {}
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(1500)
                p0 = out_dir / "frame_initial.png"
                page.screenshot(path=str(p0), timeout=25000)
                screenshots["initial"] = str(p0)
                img0_stats = analyze_image(p0)
                metrics["initial_image"] = img0_stats

                snap_initial = page.evaluate("() => window.__VERIFIER_SNAPSHOT__ ? window.__VERIFIER_SNAPSHOT__() : {}")

                # Fast-fail huge splat overdraw blowout before dragging so SwiftShader doesn't hang!
                splat_blowup = (
                    target_kind == "gaussian_splats"
                    and (img0_stats["corner_mean_luma"] > 80.0 or (img0_stats["corner_std_luma"] > 22.0 and img0_stats["mean_luma"] > 95.0))
                )
                if not splat_blowup:
                    if target_kind == "gaussian_splats":
                        metrics["splat_checks"] = _verify_gaussian_splats_live(page, out_dir, p0, snap_initial, html_text, screenshots)
                    elif target_kind == "articulated_creature":
                        metrics["creature_checks"] = _verify_creature_live(page, out_dir, p0, html_text, screenshots)
                    elif target_kind == "atari_arcade":
                        metrics["arcade_checks"] = _verify_atari_live(page, out_dir, p0, html_text, screenshots)
                else:
                    metrics["splat_blowup_detected"] = True

                snap_final = page.evaluate("() => window.__VERIFIER_SNAPSHOT__ ? window.__VERIFIER_SNAPSHOT__() : {}")
            except Exception as e:
                timed_out_stage = f"{type(e).__name__}: {str(e)[:280]}"
                metrics["runtime_timeout"] = timed_out_stage
                try:
                    snap_final = page.evaluate("() => window.__VERIFIER_SNAPSHOT__ ? window.__VERIFIER_SNAPSHOT__() : {}")
                except Exception:
                    snap_final = snap_initial
            finally:
                try:
                    browser.close()
                except Exception:
                    pass

    gl_error_names = [GL_ENUM_NAMES.get(int(e), hex(int(e))) for e in snap_final.get("glErrors", [])]
    blend_names = format_blend_funcs(snap_final.get("blendFuncs", []))

    metrics["console_errors"] = console_errors[:15]
    metrics["console_warnings"] = console_warnings[:15]
    metrics["page_errors"] = page_errors[:15]
    metrics["blocked_external_urls"] = blocked_external_urls[:10]
    metrics["screenshots"] = screenshots
    metrics["webgl_telemetry"] = {
        "contexts": snap_final.get("contexts", []),
        "webgl2": bool(snap_final.get("webgl2", False)),
        "shaderCount": snap_final.get("shaderCount", 0),
        "shaderCompileErrors": snap_final.get("shaderCompileErrors", []),
        "programLinkErrors": snap_final.get("programLinkErrors", []),
        "instancedCalls": snap_final.get("instancedCalls", 0),
        "maxInstanceCount": snap_final.get("maxInstanceCount", 0),
        "totalInstancesDrawn": snap_final.get("totalInstancesDrawn", 0),
        "drawCalls": snap_final.get("drawCalls", 0),
        "blendEnabled": bool(snap_final.get("blendEnabled", False)),
        "depthTestEnabled": bool(snap_final.get("depthTestEnabled", False)),
        "depthMaskFalseCalls": snap_final.get("depthMaskFalseCalls", 0),
        "depthMaskTrueCalls": snap_final.get("depthMaskTrueCalls", 0),
        "blendFuncs": blend_names,
        "bufferUploads": snap_final.get("bufferUploads", 0),
        "bufferSubUploads": snap_final.get("bufferSubUploads", 0),
        "rafCount": snap_final.get("rafCount", 0),
        "glErrors": gl_error_names,
        "vaosCreated": snap_final.get("vaosCreated", 0),
        "vaoBinds": snap_final.get("vaoBinds", 0),
        "attribBufferMismatches": snap_final.get("attribBufferMismatches", []),
        "transposedMatrices": snap_final.get("transposedMatrices", []),
        "perVertexInstancedConflicts": snap_final.get("perVertexInstancedConflicts", []),
    }

    ok, score, err_sig, err_detail, summary = _judge_report(target_kind, metrics, snap_final, html_text)
    report = {
        "ok": ok,
        "score": round(score, 2),
        "error_signature": err_sig,
        "error_detail": err_detail,
        "summary": summary,
        "metrics": metrics,
    }
    (out_dir / "verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _drag_canvas(page, dx: float, dy: float, steps: int = 5) -> None:
    box = page.evaluate("""() => {
      const c = document.querySelector('canvas');
      if (!c) return {x: 640, y: 360, w: 800, h: 500};
      const r = c.getBoundingClientRect();
      return {x: r.left + r.width * 0.5, y: r.top + r.height * 0.5, w: r.width, h: r.height};
    }""")
    cx, cy = box["x"], box["y"]
    page.mouse.move(cx, cy)
    page.mouse.down()
    for i in range(1, steps + 1):
        t = i / steps
        page.mouse.move(cx + dx * t, cy + dy * t)
        page.wait_for_timeout(30)
    page.mouse.up()
    page.wait_for_timeout(450)


def _verify_gaussian_splats_live(page, out_dir: Path, p0: Path, snap_initial: dict, html_text: str, screenshots: dict) -> dict[str, Any]:
    uploads_before = int(snap_initial.get("bufferUploads", 0)) + int(snap_initial.get("bufferSubUploads", 0))

    # Orbit ~90 deg horizontally + slight tilt
    _drag_canvas(page, 220, -45, steps=5)
    p90 = out_dir / "frame_orbit_90.png"
    page.screenshot(path=str(p90), timeout=15000)
    screenshots["orbit_90"] = str(p90)
    cmp_0_90 = compare_images(p0, p90)

    # Orbit another ~90 deg (total ~180 deg) to test back-side view & depth-sort occlusion flip
    _drag_canvas(page, 220, 20, steps=5)
    p180 = out_dir / "frame_orbit_180.png"
    page.screenshot(path=str(p180), timeout=15000)
    screenshots["orbit_180"] = str(p180)
    cmp_0_180 = compare_images(p0, p180)
    cmp_90_180 = compare_images(p90, p180)

    snap_after = page.evaluate("() => window.__VERIFIER_SNAPSHOT__ ? window.__VERIFIER_SNAPSHOT__() : {}")
    uploads_after = int(snap_after.get("bufferUploads", 0)) + int(snap_after.get("bufferSubUploads", 0))
    orbit_buffer_uploads = max(0, uploads_after - uploads_before)

    # Inspect captured shader sources for authentic 3D->2D Gaussian splatting math
    shaders = snap_after.get("shaderSources", [])
    all_glsl = "\n".join(s.get("src", "") for s in shaders)
    has_exp_falloff = bool(re.search(r"\bexp\s*\(", all_glsl))
    has_covariance_math = bool(
        re.search(r"cov|conic|jacobian|eigen|sigma|det|invCov|cov2d|cov3d|vrk", all_glsl, re.IGNORECASE)
    )
    has_glsl300 = "#version 300 es" in all_glsl
    has_sort_code = bool(re.search(r"sort|radix|bucket|count|depth|order|zBuf", html_text, re.IGNORECASE))

    img180_stats = analyze_image(p180)
    img90_stats = analyze_image(p90)

    # Check if optional debug hook exists
    splat_debug = page.evaluate("() => window.__SPLAT_DEBUG__ || null")

    return {
        "orbit_0_90": cmp_0_90,
        "orbit_0_180": cmp_0_180,
        "orbit_90_180": cmp_90_180,
        "orbit_90_image": img90_stats,
        "orbit_180_image": img180_stats,
        "orbit_buffer_uploads": orbit_buffer_uploads,
        "has_glsl300": has_glsl300,
        "has_exp_falloff": has_exp_falloff,
        "has_covariance_math": has_covariance_math,
        "has_sort_code": has_sort_code,
        "splat_debug": splat_debug,
    }


def _verify_creature_live(page, out_dir: Path, p0: Path, html_text: str, screenshots: dict) -> dict[str, Any]:
    # Capture temporal motion frames without touching camera
    debug_samples = []
    d0 = page.evaluate("() => window.__CREATURE_DEBUG__ ? JSON.parse(JSON.stringify(window.__CREATURE_DEBUG__)) : null")
    if d0:
        debug_samples.append(d0)

    page.wait_for_timeout(650)
    pt1 = out_dir / "frame_anim_t1.png"
    page.screenshot(path=str(pt1))
    screenshots["anim_t1"] = str(pt1)
    d1 = page.evaluate("() => window.__CREATURE_DEBUG__ ? JSON.parse(JSON.stringify(window.__CREATURE_DEBUG__)) : null")
    if d1:
        debug_samples.append(d1)

    page.wait_for_timeout(650)
    pt2 = out_dir / "frame_anim_t2.png"
    page.screenshot(path=str(pt2))
    screenshots["anim_t2"] = str(pt2)
    d2 = page.evaluate("() => window.__CREATURE_DEBUG__ ? JSON.parse(JSON.stringify(window.__CREATURE_DEBUG__)) : null")
    if d2:
        debug_samples.append(d2)

    motion_0_1 = compare_images(p0, pt1)
    motion_1_2 = compare_images(pt1, pt2)
    motion_0_2 = compare_images(p0, pt2)
    max_motion_mad = max(motion_0_1["mad"], motion_1_2["mad"], motion_0_2["mad"])
    max_motion_changed = max(motion_0_1["changed_ratio"], motion_1_2["changed_ratio"], motion_0_2["changed_ratio"])

    # Test interactive UI buttons if present (click up to 2 pose/gait buttons)
    buttons_clicked = []
    try:
        btns = page.locator("button:visible")
        n_btns = min(btns.count(), 3)
        for i in range(n_btns):
            txt = (btns.nth(i).inner_text(timeout=1000) or "").strip()
            if txt and "reset" not in txt.lower():
                btns.nth(i).click(timeout=1500)
                page.wait_for_timeout(350)
                buttons_clicked.append(txt)
    except Exception:
        pass

    # Test camera orbit drag
    _drag_canvas(page, 210, -40, steps=12)
    porbit = out_dir / "frame_orbit.png"
    page.screenshot(path=str(porbit))
    screenshots["orbit"] = str(porbit)
    orbit_cmp = compare_images(pt2, porbit)
    orbit_img = analyze_image(porbit)

    # Check hierarchical kinematics in debug hook or source
    joint_count = 0
    joint_angles_changed = False
    if debug_samples:
        j0 = debug_samples[0].get("joints") or debug_samples[0].get("jointAngles") or {}
        if isinstance(j0, dict):
            joint_count = len(j0)
        elif isinstance(j0, list):
            joint_count = len(j0)
        elif isinstance(debug_samples[0].get("jointCount"), (int, float)):
            joint_count = int(debug_samples[0]["jointCount"])
        if len(debug_samples) >= 2 and json.dumps(debug_samples[0], sort_keys=True) != json.dumps(debug_samples[-1], sort_keys=True):
            joint_angles_changed = True

    has_hierarchy_code = bool(
        re.search(r"parent|child|joint|bone|hierarchy|mat4|multiply|pushMatrix|kinemat", html_text, re.IGNORECASE)
    )

    return {
        "motion_0_1": motion_0_1,
        "motion_1_2": motion_1_2,
        "motion_0_2": motion_0_2,
        "max_motion_mad": round(max_motion_mad, 5),
        "max_motion_changed": round(max_motion_changed, 4),
        "orbit_cmp": orbit_cmp,
        "orbit_image": orbit_img,
        "buttons_clicked": buttons_clicked,
        "joint_count": joint_count,
        "joint_angles_changed": joint_angles_changed,
        "has_hierarchy_code": has_hierarchy_code,
        "debug_sample_0": debug_samples[0] if debug_samples else None,
        "debug_sample_last": debug_samples[-1] if debug_samples else None,
    }


def _read_game_state(page) -> Optional[dict[str, Any]]:
    try:
        return page.evaluate("""() => {
          const g = window.__GAME_STATE__;
          if (!g) return null;
          if (typeof g === 'function') return g();
          return JSON.parse(JSON.stringify(g));
        }""")
    except Exception:
        return None


def _verify_atari_live(page, out_dir: Path, p0: Path, html_text: str, screenshots: dict) -> dict[str, Any]:
    state_initial = _read_game_state(page)

    # Focus canvas/page and trigger Start if on title screen
    page.mouse.click(640, 360)
    page.wait_for_timeout(150)
    if not state_initial or str(state_initial.get("state", "")).lower() not in ("playing", "active", "running"):
        # Try Start button or Enter/Space
        try:
            btn = page.locator("button:visible").first
            if btn.count() > 0:
                btn.click(timeout=1000)
                page.wait_for_timeout(200)
        except Exception:
            pass
        page.keyboard.press("Enter")
        page.wait_for_timeout(250)
        st_check = _read_game_state(page)
        if not st_check or str(st_check.get("state", "")).lower() not in ("playing", "active", "running"):
            page.keyboard.press("Space")
            page.wait_for_timeout(300)

    state_playing = _read_game_state(page)
    p_play = out_dir / "frame_playing_start.png"
    page.screenshot(path=str(p_play))
    screenshots["playing_start"] = str(p_play)
    play_img = analyze_image(p_play)

    # Test live player movement: ArrowRight then ArrowLeft
    x_before = float(state_playing.get("playerX", 0)) if isinstance(state_playing, dict) else 0.0
    page.keyboard.down("ArrowRight")
    page.wait_for_timeout(420)
    page.keyboard.up("ArrowRight")
    state_after_right = _read_game_state(page)
    x_right = float(state_after_right.get("playerX", 0)) if isinstance(state_after_right, dict) else 0.0
    p_right = out_dir / "frame_after_right.png"
    page.screenshot(path=str(p_right))
    screenshots["after_right"] = str(p_right)
    move_right_img = compare_images(p_play, p_right)

    page.keyboard.down("ArrowLeft")
    page.wait_for_timeout(650)
    page.keyboard.up("ArrowLeft")
    state_after_left = _read_game_state(page)
    x_left = float(state_after_left.get("playerX", 0)) if isinstance(state_after_left, dict) else 0.0

    # Active playtest loop: steer under enemies and fire Space to score real points!
    initial_score = int(state_playing.get("score", 0)) if isinstance(state_playing, dict) else 0
    initial_lives = int(state_playing.get("lives", 3)) if isinstance(state_playing, dict) else 3
    best_score = initial_score
    max_collisions = 0
    t_start = time.time()
    step_i = 0
    while time.time() - t_start < 5.5:
        st = _read_game_state(page)
        if isinstance(st, dict):
            best_score = max(best_score, int(st.get("score", 0) or 0))
            max_collisions = max(max_collisions, int(st.get("collisions", 0) or 0))
            if str(st.get("state", "")).lower() != "playing":
                break
            enemies = st.get("enemies") or []
            px = float(st.get("playerX", 0))
            if isinstance(enemies, list) and enemies:
                # Target nearest alive enemy by |dx|
                target = min(
                    enemies,
                    key=lambda e: abs(float(e.get("x", px)) - px) if isinstance(e, dict) else 9999.0,
                )
                tx = float(target.get("x", px)) if isinstance(target, dict) else px
                dx = tx - px
                if dx > 14:
                    page.keyboard.down("ArrowRight")
                    page.wait_for_timeout(90)
                    page.keyboard.up("ArrowRight")
                elif dx < -14:
                    page.keyboard.down("ArrowLeft")
                    page.wait_for_timeout(90)
                    page.keyboard.up("ArrowLeft")
            else:
                key = "ArrowRight" if (step_i % 6) < 3 else "ArrowLeft"
                page.keyboard.down(key)
                page.wait_for_timeout(80)
                page.keyboard.up(key)
        page.keyboard.press("Space")
        page.wait_for_timeout(95)
        step_i += 1
        if best_score >= initial_score + 150 and step_i >= 18:
            break

    p_action = out_dir / "frame_action_scored.png"
    page.screenshot(path=str(p_action))
    screenshots["action_scored"] = str(p_action)
    state_after_action = _read_game_state(page)
    if isinstance(state_after_action, dict):
        best_score = max(best_score, int(state_after_action.get("score", 0) or 0))
        max_collisions = max(max_collisions, int(state_after_action.get("collisions", 0) or 0))

    # Now test real damage -> lives decrement -> gameover -> restart!
    # Steer into incoming enemy projectiles or nearest descending enemy without shooting.
    lives_dropped = False
    gameover_reached = False
    min_lives_seen = initial_lives
    t_dmg = time.time()
    while time.time() - t_dmg < 14.0:
        st = _read_game_state(page)
        if not isinstance(st, dict):
            break
        cur_state = str(st.get("state", "")).lower()
        cur_lives = int(st.get("lives", initial_lives) or 0)
        if cur_lives < min_lives_seen:
            min_lives_seen = cur_lives
            lives_dropped = True
        if cur_state in ("gameover", "game_over", "over", "lost", "dead"):
            gameover_reached = True
            break
        # Steer toward closest enemy bullet or lowest enemy to force genuine collision
        px = float(st.get("playerX", 0))
        hazards = st.get("enemyBullets") or st.get("hazards") or st.get("enemies") or []
        if isinstance(hazards, list) and hazards:
            # Prefer enemy bullet closest to player Y (largest y)
            hz = max(hazards, key=lambda h: float(h.get("y", 0)) if isinstance(h, dict) else 0.0)
            tx = float(hz.get("x", px)) if isinstance(hz, dict) else px
            dx = tx - px
            if dx > 8:
                page.keyboard.down("ArrowRight")
                page.wait_for_timeout(85)
                page.keyboard.up("ArrowRight")
            elif dx < -8:
                page.keyboard.down("ArrowLeft")
                page.wait_for_timeout(85)
                page.keyboard.up("ArrowLeft")
            else:
                page.wait_for_timeout(90)
        else:
            page.wait_for_timeout(120)

    p_gameover = out_dir / "frame_gameover.png"
    page.screenshot(path=str(p_gameover))
    screenshots["gameover"] = str(p_gameover)
    state_gameover = _read_game_state(page)

    # Test Restart from gameover
    restart_ok = False
    state_restarted = None
    if gameover_reached:
        page.keyboard.press("Enter")
        page.wait_for_timeout(300)
        state_restarted = _read_game_state(page)
        if not state_restarted or str(state_restarted.get("state", "")).lower() not in ("playing", "active", "running"):
            page.keyboard.press("KeyR")
            page.wait_for_timeout(300)
            state_restarted = _read_game_state(page)
        if not state_restarted or str(state_restarted.get("state", "")).lower() not in ("playing", "active", "running"):
            page.keyboard.press("Space")
            page.wait_for_timeout(300)
            state_restarted = _read_game_state(page)
        if isinstance(state_restarted, dict):
            st_r = str(state_restarted.get("state", "")).lower()
            sc_r = int(state_restarted.get("score", -1) or 0)
            lv_r = int(state_restarted.get("lives", 0) or 0)
            if st_r in ("playing", "active", "running") and sc_r == 0 and lv_r >= initial_lives:
                restart_ok = True

    p_restart = out_dir / "frame_restarted.png"
    page.screenshot(path=str(p_restart))
    screenshots["restarted"] = str(p_restart)

    return {
        "has_game_state_hook": state_playing is not None,
        "state_initial": state_initial,
        "state_playing": state_playing,
        "playing_image": play_img,
        "delta_right": round(x_right - x_before, 2),
        "delta_left": round(x_left - x_right, 2),
        "move_right_img": move_right_img,
        "initial_score": initial_score,
        "best_score": best_score,
        "score_incremented": best_score > initial_score,
        "max_collisions": max_collisions,
        "initial_lives": initial_lives,
        "min_lives_seen": min_lives_seen,
        "lives_dropped": lives_dropped,
        "gameover_reached": gameover_reached,
        "restart_ok": restart_ok,
        "state_gameover": state_gameover,
        "state_restarted": state_restarted,
    }


def _judge_report(
    target_kind: str,
    metrics: dict[str, Any],
    snap: dict[str, Any],
    html_text: str,
) -> tuple[bool, float, str, str, str]:
    t = metrics["webgl_telemetry"]
    img0 = metrics.get("initial_image") or {}

    if metrics.get("splat_blowup_detected"):
        return (
            False,
            24.0,
            "SPLAT_SCALE_OR_PROJECTION_BLOWUP",
            f"Rendered splats are massively oversized / blown up across the entire 1280x720 viewport and corners (corner_mean_luma={img0.get('corner_mean_luma')}, corner_std_luma={img0.get('corner_std_luma')}, mean_luma={img0.get('mean_luma')}), causing severe quad overdraw. Inspect the attached screenshot and check view-space Z sign/lookAt matrix row vs column convention, focal/viewport NDC conversion, and clamp maximum splat screen radius!",
            "",
        )
    if metrics.get("runtime_timeout"):
        return (
            False,
            18.0,
            "RENDERER_OVERDRAW_OR_TIMEOUT",
            f"Headless WebGL2 verifier timed out during live render/interaction ({metrics['runtime_timeout']}). Likely massive overdraw (oversized splat quads filling the whole screen due to view-space z sign/clamping or projection scale error) or a blocking loop. Check frame_initial.png, view matrix Z sign, and clamp splat quad radius in pixels!",
            "",
        )

    if metrics["blocked_external_urls"]:
        return (
            False,
            0.0,
            "EXTERNAL_NETWORK_DEPENDENCY",
            f"Blocked external CDN/network URLs (HTML must be 100% self-contained offline): {metrics['blocked_external_urls'][:5]}",
            "",
        )
    if metrics["page_errors"]:
        return (
            False,
            5.0,
            "JS_PAGE_ERROR",
            f"Uncaught pageerror(s): {' | '.join(metrics['page_errors'][:4])}",
            "",
        )
    if metrics["console_errors"]:
        return (
            False,
            8.0,
            "CONSOLE_ERROR",
            f"Console error(s): {' | '.join(metrics['console_errors'][:4])}",
            "",
        )
    if t["shaderCompileErrors"]:
        se = t["shaderCompileErrors"][0]
        return (
            False,
            10.0,
            "SHADER_COMPILE_ERROR",
            f"{se.get('kind')} shader compile failed: {se.get('log')}",
            "",
        )
    if t["programLinkErrors"]:
        return (
            False,
            12.0,
            "PROGRAM_LINK_ERROR",
            f"WebGL program link failed: {t['programLinkErrors'][0]}",
            "",
        )
    if t["glErrors"]:
        return (
            False,
            15.0,
            "WEBGL_RUNTIME_ERROR",
            f"WebGL gl.getError() returned: {', '.join(t['glErrors'][:6])}",
            "",
        )

    if target_kind == "gaussian_splats":
        sc = metrics["splat_checks"]
        if not t["webgl2"]:
            return False, 12.0, "NO_WEBGL2_CONTEXT", f"Expected WebGL2 context, got contexts={t['contexts']}", ""
        if t["maxInstanceCount"] < 1500 or t["instancedCalls"] < 1:
            return (
                False,
                20.0,
                "NO_INSTANCED_SPLATS",
                f"Expected real WebGL2 instanced drawing (drawArraysInstanced/drawElementsInstanced with >=1500 splats), got instancedCalls={t['instancedCalls']}, maxInstanceCount={t['maxInstanceCount']}, drawCalls={t['drawCalls']}",
                "",
            )
        if not sc["has_glsl300"] or not sc["has_exp_falloff"] or not sc["has_covariance_math"]:
            return (
                False,
                25.0,
                "MISSING_GAUSSIAN_COVARIANCE_MATH",
                f"Shaders must use #version 300 es, 3D->2D covariance/conic projection, and exp() Gaussian falloff. Observed: glsl300={sc['has_glsl300']}, exp={sc['has_exp_falloff']}, covariance_math={sc['has_covariance_math']}",
                "",
            )
        if not t["blendEnabled"] or t["depthMaskFalseCalls"] < 1:
            return (
                False,
                30.0,
                "MISSING_BLEND_OR_DEPTH_MASK",
                f"3D Gaussian Splatting requires gl.enable(gl.BLEND) and gl.depthMask(false) during splat pass. Observed: blendEnabled={t['blendEnabled']}, depthMaskFalseCalls={t['depthMaskFalseCalls']}, blendFuncs={t['blendFuncs']}",
                "",
            )
        # Reject broken pure additive blend (ONE, ONE) which blows out into white soup
        blend_str = " ".join(t["blendFuncs"])
        if ("(ONE, ONE)" in blend_str or "(SRC_ALPHA, ONE)" in blend_str) and "ONE_MINUS_SRC_ALPHA" not in blend_str:
            return (
                False,
                32.0,
                "WRONG_BLEND_STATE_ADDITIVE",
                f"Detected additive-only blend state ({t['blendFuncs']}) without ONE_MINUS_SRC_ALPHA. Use premultiplied alpha gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA) or blendFuncSeparate with ONE_MINUS_SRC_ALPHA.",
                "",
            )
        if t.get("transposedMatrices"):
            return (
                False,
                21.0,
                "TRANSPOSED_VIEW_MATRIX_LAYOUT",
                f"WebGL2 uniformMatrix4fv inspector caught a ROW-MAJOR view/affine matrix passed to column-major WebGL! {t['transposedMatrices'][0]}. In WebGL column-major Float32Array(16), columns are contiguous: [s0,u0,-f0,0,  s1,u1,-f1,0,  s2,u2,-f2,0,  -dot(s,eye),-dot(u,eye),dot(f,eye),1] (translation at indices 12,13,14; indices 3,7,11 MUST be 0).",
                "",
            )
        if t.get("perVertexInstancedConflicts"):
            return (
                False,
                56.0,
                "WEBGL_PER_VERTEX_BUFFER_MISMATCH",
                f"WebGL2 state inspector caught per-vertex quad attribute bound to instanced splat buffer! {t['perVertexInstancedConflicts'][0]} Also ensure quad corners use proper TRIANGLE_STRIP order [-1,-1, 1,-1, -1,1, 1,1] (or 6-vertex TRIANGLES) so every splat renders as a smooth elliptical Gaussian instead of sharp triangles/chevrons!",
                "",
            )
        if t.get("attribBufferMismatches"):
            return (
                False,
                22.0,
                "WEBGL_VAO_BUFFER_BINDING_BUG",
                f"WebGL2 state inspector caught vertexAttribPointer bound to wrong/tiny ARRAY_BUFFER! {t['attribBufferMismatches'][:4]}. Always isolate each shader program in its own WebGL2 VAO (gl.createVertexArray / gl.bindVertexArray) AND explicitly bind the splat ARRAY_BUFFER before configuring per-instance attributes!",
                "",
            )
        if img0["non_bg_ratio"] < 0.06 or img0["shannon_entropy"] < 1.5 or img0["std_luma"] < 6.0 or (img0.get("scene_std_luma", 99) < 8.0 and img0.get("scene_p99_luma", 255) < 60.0):
            return (
                False,
                24.0,
                "BLANK_3D_SCENE_ZERO_VISIBLE_SPLATS",
                f"Rendered WebGL2 canvas shows ONLY a dark blank background gradient with ZERO visible 3D splats (excluding HUD: scene_std_luma={img0.get('scene_std_luma')}, scene_p95_luma={img0.get('scene_p95_luma')}, scene_p99_luma={img0.get('scene_p99_luma')}, orbit_0_90_mad={sc['orbit_0_90']['mad']}, vaosCreated={t.get('vaosCreated')}). Check: (1) use separate VAOs for background vs splats, (2) bind splat ARRAY_BUFFER before vertexAttribPointer, (3) verify lookAt view-space Z sign (-viewPos.z > 0) and clip-space coordinates, (4) verify quad corner winding / TRIANGLE_STRIP order ((-1,-1),(1,-1),(-1,1),(1,1))!",
                "",
            )
        if img0["near_white_ratio"] > 0.65:
            return (
                False,
                34.0,
                "OVEREXPOSED_WHITE_BLOWOUT",
                f"Canvas is blown out near-white ({img0['near_white_ratio']*100:.1f}% white pixels), likely wrong premultiplied alpha or scale/opacity explosion.",
                "",
            )
        if img0["center_non_bg_ratio"] < 0.08 or img0["bbox_area_ratio"] < 0.10 or img0["distinct_colors_q16"] < 20:
            return (
                False,
                42.0,
                "POOR_FRAMING_OR_FLAT_RENDER",
                f"Scene is too small, off-center, or flat. center_non_bg={img0['center_non_bg_ratio']}, bbox_area={img0['bbox_area_ratio']}, distinct_colors={img0['distinct_colors_q16']}, entropy={img0['shannon_entropy']}",
                "",
            )
        if sc["orbit_0_90"]["mad"] < 0.02 or sc["orbit_0_180"]["mad"] < 0.025:
            return (
                False,
                48.0,
                "ORBIT_CAMERA_BROKEN",
                f"Mouse drag orbit did not noticeably rotate the 3D splat scene! orbit_0_90_mad={sc['orbit_0_90']['mad']}, orbit_0_180_mad={sc['orbit_0_180']['mad']}",
                "",
            )
        if sc["orbit_buffer_uploads"] < 1 or not sc["has_sort_code"]:
            return (
                False,
                54.0,
                "DEPTH_SORTING_MISSING",
                f"Expected dynamic back-to-front depth sorting uploading sorted order/buffers when camera orbits! orbit_buffer_uploads={sc['orbit_buffer_uploads']}, has_sort_code={sc['has_sort_code']}",
                "",
            )

        # All hard gates passed! Compute rich quality score 60..100
        premult_bonus = 6.0 if ("(ONE, ONE_MINUS_SRC_ALPHA)" in blend_str or "ONE, ONE_MINUS_SRC_ALPHA" in blend_str) else 2.0
        splat_score = min(10.0, (t["maxInstanceCount"] / 8000.0) * 10.0)
        color_score = min(8.0, (img0["distinct_colors_q16"] / 180.0) * 8.0)
        entropy_score = min(8.0, max(0.0, (img0["shannon_entropy"] - 1.5) / 3.5) * 8.0)
        framing_score = min(8.0, min(img0["bbox_area_ratio"] / 0.45, 1.0) * 4.0 + min(img0["center_non_bg_ratio"] / 0.45, 1.0) * 4.0)
        score = 60.0 + premult_bonus + splat_score + color_score + entropy_score + framing_score
        summary = (
            f"WebGL2 3DGS PASS: {t['maxInstanceCount']} instanced splats, 0 GL/console errors, "
            f"blend={t['blendFuncs'][:1]}, depthMask(false)={t['depthMaskFalseCalls']}, "
            f"orbit_sort_uploads={sc['orbit_buffer_uploads']}, orbit_mad={sc['orbit_0_180']['mad']:.3f}, "
            f"entropy={img0['shannon_entropy']:.2f}, colors={img0['distinct_colors_q16']}"
        )
        return True, min(100.0, score), "", "", summary

    elif target_kind == "articulated_creature":
        cc = metrics["creature_checks"]
        if not t["webgl2"]:
            return False, 12.0, "NO_WEBGL2_CONTEXT", f"Expected WebGL2 3D context, got contexts={t['contexts']}", ""
        if t["rafCount"] < 5 or t["drawCalls"] < 3:
            return False, 18.0, "NO_ANIMATION_LOOP", f"Expected active WebGL2 animation loop, got rafCount={t['rafCount']}, drawCalls={t['drawCalls']}", ""
        if img0["non_bg_ratio"] < 0.06 or img0["shannon_entropy"] < 1.5 or img0["std_luma"] < 6.0:
            return False, 22.0, "BLANK_OR_EMPTY_CANVAS", f"Rendered canvas appears blank/empty! initial_image={img0}", ""
        if img0["center_non_bg_ratio"] < 0.07 or img0["bbox_area_ratio"] < 0.10 or img0["distinct_colors_q16"] < 18:
            return (
                False,
                38.0,
                "POOR_FRAMING_OR_FLAT_RENDER",
                f"Creature too small, off-center, or flat. center_non_bg={img0['center_non_bg_ratio']}, bbox_area={img0['bbox_area_ratio']}, colors={img0['distinct_colors_q16']}",
                "",
            )
        if img0.get("top_edge_fg", 0.0) > 0.14:
            return (
                False,
                52.0,
                "CAMERA_TOO_CLOSE_CROPPED",
                f"Creature is cropped at the top edge of the viewport (top_edge_fg={img0.get('top_edge_fg')})! Pull the camera back and frame a clean 3/4 side-front full-body view so the entire head, wings, legs, and tail stay well inside the 1280x720 frame.",
                "",
            )
        if not cc["has_hierarchy_code"] or (cc["debug_sample_0"] is not None and cc["joint_count"] < 8):
            return (
                False,
                42.0,
                "MISSING_ARTICULATED_HIERARCHY",
                f"Expected >=8 hierarchical articulated joints (spine, neck, head, jaw, multi-segment legs, wings, tail) exposed in window.__CREATURE_DEBUG__. Observed joint_count={cc['joint_count']}, has_hierarchy_code={cc['has_hierarchy_code']}",
                "",
            )
        if cc["debug_sample_0"] is None:
            return (
                False,
                44.0,
                "MISSING_CREATURE_DEBUG_HOOK",
                "Expose read-only window.__CREATURE_DEBUG__ with {jointCount, joints: {name: angle}, gaitMode, time} updated each frame so joint kinematics can be audited.",
                "",
            )
        if cc["max_motion_mad"] < 0.006 or cc["max_motion_changed"] < 0.015 or not cc["joint_angles_changed"]:
            return (
                False,
                48.0,
                "STATIC_OR_FROZEN_CREATURE",
                f"Creature is not visibly animating! max_motion_mad={cc['max_motion_mad']}, max_motion_changed={cc['max_motion_changed']}, joint_angles_changed={cc['joint_angles_changed']}",
                "",
            )
        if cc["orbit_cmp"]["mad"] < 0.02:
            return (
                False,
                54.0,
                "ORBIT_CAMERA_BROKEN",
                f"Mouse drag orbit did not rotate 3D creature view! orbit_mad={cc['orbit_cmp']['mad']}",
                "",
            )

        joint_score = min(10.0, (cc["joint_count"] / 18.0) * 10.0)
        motion_score = min(8.0, (cc["max_motion_mad"] / 0.04) * 8.0)
        color_score = min(8.0, (img0["distinct_colors_q16"] / 150.0) * 8.0)
        entropy_score = min(7.0, max(0.0, (img0["shannon_entropy"] - 1.5) / 3.5) * 7.0)
        framing_score = min(7.0, min(img0["bbox_area_ratio"] / 0.40, 1.0) * 3.5 + min(img0["center_non_bg_ratio"] / 0.35, 1.0) * 3.5)
        score = 60.0 + joint_score + motion_score + color_score + entropy_score + framing_score
        summary = (
            f"WebGL2 Articulated Creature PASS: {cc['joint_count']} hierarchical joints animating, "
            f"motion_mad={cc['max_motion_mad']:.4f}, orbit_mad={cc['orbit_cmp']['mad']:.3f}, "
            f"buttons={cc['buttons_clicked']}, entropy={img0['shannon_entropy']:.2f}, colors={img0['distinct_colors_q16']}"
        )
        return True, min(100.0, score), "", "", summary

    elif target_kind == "atari_arcade":
        ac = metrics["arcade_checks"]
        pimg = ac["playing_image"]
        if t["rafCount"] < 5:
            return False, 15.0, "NO_ANIMATION_LOOP", f"Expected live animation loop, rafCount={t['rafCount']}", ""
        if pimg["non_bg_ratio"] < 0.04 or pimg["shannon_entropy"] < 0.8:
            return False, 20.0, "BLANK_OR_EMPTY_CANVAS", f"Game viewport appears blank/empty! playing_image={pimg}", ""
        if not ac["has_game_state_hook"]:
            return (
                False,
                25.0,
                "MISSING_GAME_STATE_HOOK",
                "Expose read-only window.__GAME_STATE__ returning {state, score, lives, wave, playerX, playerY, enemies:[{x,y}], enemyBullets:[{x,y}], collisions} updated every frame.",
                "",
            )
        st_play = str((ac["state_playing"] or {}).get("state", "")).lower()
        if st_play not in ("playing", "active", "running"):
            return (
                False,
                30.0,
                "GAME_START_FAILED",
                f"Pressing Enter/Space/Start did not enter playing state! state_playing={ac['state_playing']}",
                "",
            )
        if ac["delta_right"] < 12.0 or ac["delta_left"] > -12.0 or ac["move_right_img"]["changed_ratio"] < 0.002:
            return (
                False,
                38.0,
                "PLAYER_INPUT_BROKEN",
                f"ArrowRight/ArrowLeft keys did not move player smoothly! delta_right={ac['delta_right']}, delta_left={ac['delta_left']}, canvas_changed={ac['move_right_img']['changed_ratio']}",
                "",
            )
        if not ac["score_incremented"] or ac["best_score"] <= ac["initial_score"]:
            return (
                False,
                46.0,
                "NO_SCORING_OR_COLLISIONS",
                f"Live playtest steered under enemies and fired Space for 5.5s, but score stayed {ac['best_score']} (collisions={ac['max_collisions']}). Check Space firing rate, bullet speed, hitbox collision, and enemy coordinates in __GAME_STATE__.",
                "",
            )
        if not ac["lives_dropped"] or not ac["gameover_reached"]:
            return (
                False,
                54.0,
                "NO_DAMAGE_OR_GAMEOVER",
                f"Live playtest steered into enemy bullets/enemies for 14s without reaching gameover! initial_lives={ac['initial_lives']}, min_lives_seen={ac['min_lives_seen']}, gameover_reached={ac['gameover_reached']}. Ensure enemy fire/hazards are active, exposed in enemyBullets/enemies, have fair brief invulnerability (<=0.7s), and trigger gameover when lives hit 0.",
                "",
            )
        if not ac["restart_ok"]:
            return (
                False,
                58.0,
                "RESTART_STATE_BROKEN",
                f"Game reached gameover cleanly, but pressing Enter/R/Space did not cleanly reset state='playing', score=0, lives={ac['initial_lives']}. Observed state_restarted={ac['state_restarted']}",
                "",
            )

        score_pts = min(12.0, (ac["best_score"] / 500.0) * 12.0)
        color_pts = min(10.0, (pimg["distinct_colors_q16"] / 60.0) * 10.0)
        entropy_pts = min(10.0, max(0.0, (pimg["shannon_entropy"] - 0.8) / 3.0) * 10.0)
        move_pts = min(8.0, min(abs(ac["delta_right"]) / 80.0, 1.0) * 4.0 + min(abs(ac["delta_left"]) / 80.0, 1.0) * 4.0)
        score = 60.0 + score_pts + color_pts + entropy_pts + move_pts
        summary = (
            f"Atari Arcade PLAYTEST PASS: ArrowR={ac['delta_right']:+.1f}px ArrowL={ac['delta_left']:+.1f}px, "
            f"live_score={ac['best_score']}, collisions={ac['max_collisions']}, "
            f"lives {ac['initial_lives']}->0->gameover->clean restart, entropy={pimg['shannon_entropy']:.2f}"
        )
        return True, min(100.0, score), "", "", summary

    return False, 0.0, "UNKNOWN_TARGET", f"Unknown target_kind {target_kind}", ""


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["gaussian_splats", "articulated_creature", "atari_arcade"])
    ap.add_argument("--html", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = verify_html(args.kind, Path(args.html), Path(args.out))
    print(json.dumps({k: v for k, v in rep.items() if k != "metrics"}, indent=2))

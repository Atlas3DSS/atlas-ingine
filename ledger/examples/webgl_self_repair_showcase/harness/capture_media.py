#!/usr/bin/env python3
"""Capture high-res 30fps MP4 clips and Attempt-0 vs Final comparison images for Switchboard #848."""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")

from PIL import Image, ImageDraw, ImageFont
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


def _encode_mp4(frames_dir: Path, out_mp4: Path, fps: int = 24) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _drag_step(page, cx: float, cy: float, dx: float, dy: float) -> None:
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(cx + dx, cy + dy, steps=3)
    page.mouse.up()


def record_gaussian_turntable(html_path: Path, out_mp4: Path, n_frames: int = 120, fps: int = 24) -> None:
    """Capture a smooth 360-degree orbit turntable clip of the verified 3D Gaussian Splat scene."""
    html_path = html_path.resolve()
    with tempfile.TemporaryDirectory(prefix="splat_frames_") as tmp:
        fdir = Path(tmp)
        with _LocalStaticServer(html_path.parent) as base_url:
            with sync_playwright() as p:
                b = p.chromium.launch(headless=True, executable_path=CHROME_EXECUTABLE, args=CHROME_ARGS)
                page = b.new_page(viewport={"width": 1280, "height": 720})
                page.goto(f"{base_url}/{html_path.name}", wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(1400)
                box = page.evaluate("""() => {
                  const c = document.querySelector('canvas');
                  if (!c) return {x: 640, y: 360};
                  const r = c.getBoundingClientRect();
                  return {x: r.left + r.width * 0.5, y: r.top + r.height * 0.5};
                }""")
                cx, cy = box["x"], box["y"]
                for i in range(n_frames):
                    page.screenshot(path=str(fdir / f"frame_{i:04d}.png"))
                    # Smooth horizontal orbit + gentle elevation wave
                    dy = 2.0 if (i % 60) < 30 else -2.0
                    _drag_step(page, cx, cy, 8.0, dy * 0.35)
                    page.wait_for_timeout(35)
                b.close()
        _encode_mp4(fdir, out_mp4, fps=fps)


def record_creature_clip(html_path: Path, out_mp4: Path, n_frames: int = 120, fps: int = 24) -> None:
    """Capture animated creature clip showcasing continuous limb/wing/tail kinematics + orbit + mode switch."""
    html_path = html_path.resolve()
    with tempfile.TemporaryDirectory(prefix="creature_frames_") as tmp:
        fdir = Path(tmp)
        with _LocalStaticServer(html_path.parent) as base_url:
            with sync_playwright() as p:
                b = p.chromium.launch(headless=True, executable_path=CHROME_EXECUTABLE, args=CHROME_ARGS)
                page = b.new_page(viewport={"width": 1280, "height": 720})
                page.goto(f"{base_url}/{html_path.name}", wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(1200)
                box = page.evaluate("""() => {
                  const c = document.querySelector('canvas');
                  if (!c) return {x: 640, y: 360};
                  const r = c.getBoundingClientRect();
                  return {x: r.left + r.width * 0.5, y: r.top + r.height * 0.5};
                }""")
                cx, cy = box["x"], box["y"]
                btns = page.locator("button:visible")
                btn_count = btns.count()
                for i in range(n_frames):
                    if i == 45 and btn_count >= 2:
                        try:
                            btns.nth(1).click(timeout=800)
                        except Exception:
                            pass
                    elif i == 85 and btn_count >= 3:
                        try:
                            btns.nth(2).click(timeout=800)
                        except Exception:
                            pass
                    page.screenshot(path=str(fdir / f"frame_{i:04d}.png"))
                    if 25 <= i <= 110:
                        _drag_step(page, cx, cy, 6.5, 0.0)
                    page.wait_for_timeout(45)
                b.close()
        _encode_mp4(fdir, out_mp4, fps=fps)


def record_atari_clip(html_path: Path, out_mp4: Path, n_frames: int = 140, fps: int = 20) -> None:
    """Capture live automated Atari gameplay clip: start -> move & shoot -> score -> damage -> gameover -> restart."""
    html_path = html_path.resolve()
    with tempfile.TemporaryDirectory(prefix="atari_frames_") as tmp:
        fdir = Path(tmp)
        with _LocalStaticServer(html_path.parent) as base_url:
            with sync_playwright() as p:
                b = p.chromium.launch(headless=True, executable_path=CHROME_EXECUTABLE, args=CHROME_ARGS)
                page = b.new_page(viewport={"width": 1280, "height": 720})
                page.goto(f"{base_url}/{html_path.name}", wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(800)
                # Show title for 8 frames
                for i in range(8):
                    page.screenshot(path=str(fdir / f"frame_{i:04d}.png"))
                    page.wait_for_timeout(50)
                # Start game
                page.mouse.click(640, 360)
                page.keyboard.press("Enter")
                page.wait_for_timeout(150)
                page.keyboard.press("Space")
                page.wait_for_timeout(150)

                for i in range(8, n_frames):
                    st = page.evaluate("""() => {
                      const g = window.__GAME_STATE__;
                      if (!g) return null;
                      return typeof g === 'function' ? g() : g;
                    }""")
                    page.screenshot(path=str(fdir / f"frame_{i:04d}.png"))
                    if isinstance(st, dict):
                        cur = str(st.get("state", "")).lower()
                        if cur in ("gameover", "game_over", "over"):
                            if i > n_frames - 20:
                                page.keyboard.press("Enter")
                            page.wait_for_timeout(60)
                            continue
                        px = float(st.get("playerX", 640))
                        # Play actively for frames 8..95, then allow enemy fire hits for gameover demo
                        if i < 95:
                            enemies = st.get("enemies") or []
                            if enemies:
                                tgt = min(enemies, key=lambda e: abs(float(e.get("x", px)) - px))
                                dx = float(tgt.get("x", px)) - px
                                if dx > 12:
                                    page.keyboard.down("ArrowRight")
                                    page.wait_for_timeout(55)
                                    page.keyboard.up("ArrowRight")
                                elif dx < -12:
                                    page.keyboard.down("ArrowLeft")
                                    page.wait_for_timeout(55)
                                    page.keyboard.up("ArrowLeft")
                            if i % 2 == 0:
                                page.keyboard.press("Space")
                        else:
                            hazards = st.get("enemyBullets") or st.get("enemies") or []
                            if hazards:
                                hz = max(hazards, key=lambda h: float(h.get("y", 0)))
                                dx = float(hz.get("x", px)) - px
                                if dx > 8:
                                    page.keyboard.down("ArrowRight")
                                    page.wait_for_timeout(55)
                                    page.keyboard.up("ArrowRight")
                                elif dx < -8:
                                    page.keyboard.down("ArrowLeft")
                                    page.wait_for_timeout(55)
                                    page.keyboard.up("ArrowLeft")
                    page.wait_for_timeout(45)
                b.close()
        _encode_mp4(fdir, out_mp4, fps=fps)


def make_before_after_card(
    before_png: Path,
    after_png: Path,
    out_png: Path,
    before_label: str,
    after_label: str,
    title: str,
) -> None:
    w, h = 640, 360
    pad = 16
    header_h = 54
    footer_h = 44
    canvas = Image.new("RGB", (w * 2 + pad * 3, h + header_h + footer_h + pad), (13, 17, 23))
    draw = ImageDraw.Draw(canvas)

    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    draw.text((pad, 14), title, fill=(240, 246, 252), font=font_title)

    def _fit(p: Path) -> Image.Image:
        if p.exists():
            im = Image.open(p).convert("RGB")
            return im.resize((w, h), Image.Resampling.LANCZOS)
        blank = Image.new("RGB", (w, h), (22, 27, 34))
        d = ImageDraw.Draw(blank)
        d.text((220, 170), "NO RENDERED OUTPUT", fill=(248, 81, 73), font=font_sub)
        return blank

    im_b = _fit(before_png)
    im_a = _fit(after_png)
    canvas.paste(im_b, (pad, header_h))
    canvas.paste(im_a, (pad * 2 + w, header_h))

    # Colored status borders
    draw.rectangle([pad - 2, header_h - 2, pad + w + 1, header_h + h + 1], outline=(248, 81, 73), width=3)
    draw.rectangle([pad * 2 + w - 2, header_h - 2, pad * 2 + w * 2 + 1, header_h + h + 1], outline=(63, 185, 80), width=3)

    draw.text((pad, header_h + h + 12), before_label, fill=(255, 161, 152), font=font_sub)
    draw.text((pad * 2 + w, header_h + h + 12), after_label, fill=(126, 231, 135), font=font_sub)
    canvas.save(out_png, format="PNG")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["gaussian_splats", "articulated_creature", "atari_arcade"])
    ap.add_argument("--html", required=True)
    ap.add_argument("--mp4", required=True)
    args = ap.parse_args()
    if args.kind == "gaussian_splats":
        record_gaussian_turntable(Path(args.html), Path(args.mp4))
    elif args.kind == "articulated_creature":
        record_creature_clip(Path(args.html), Path(args.mp4))
    else:
        record_atari_clip(Path(args.html), Path(args.mp4))
    print(f"Recorded {args.mp4}")

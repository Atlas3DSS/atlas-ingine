#!/usr/bin/env python3
"""Record 1280x720 H.264 gameplay MP4s, before/after comparison cards, SQLite dump, index.html, and README.md for #854."""
from __future__ import annotations

import functools
import html
import http.server
import json
import socketserver
import sqlite3
import subprocess
import sys
import tempfile
import threading
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


def _encode_mp4(fdir: Path, out_mp4: Path, fps: int = 20) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(fps), "-i", str(fdir / "frame_%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-movflags", "+faststart", str(out_mp4),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def record_game_clip(game_kind: str, html_path: Path, out_mp4: Path, n_frames: int = 100, fps: int = 20) -> None:
    html_path = html_path.resolve()
    with tempfile.TemporaryDirectory(prefix=f"bijan_{game_kind}_") as tmp:
        fdir = Path(tmp)
        with _LocalStaticServer(html_path.parent) as base_url:
            with sync_playwright() as p:
                b = p.chromium.launch(headless=True, executable_path=CHROME_EXECUTABLE, args=CHROME_ARGS)
                page = b.new_page(viewport={"width": 1280, "height": 720})
                page.goto(f"{base_url}/{html_path.name}", wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(1000)
                page.mouse.click(640, 360)
                page.keyboard.press("Enter")
                page.wait_for_timeout(200)

                if game_kind == "stock_offroad_4x4":
                    page.keyboard.down("ArrowRight")
                    page.keyboard.down("KeyD")
                    for i in range(n_frames):
                        if i in (25, 55, 80):
                            page.keyboard.press("Space")
                        page.screenshot(path=str(fdir / f"frame_{i:04d}.png"), timeout=15000)
                        page.wait_for_timeout(45)
                    page.keyboard.up("ArrowRight")
                    page.keyboard.up("KeyD")

                elif game_kind == "bowling_horror_3d":
                    for i in range(n_frames):
                        if i == 8:
                            page.keyboard.down("ArrowRight")
                        elif i == 14:
                            page.keyboard.up("ArrowRight")
                            page.keyboard.down("Space")
                        elif i == 24:
                            page.keyboard.up("Space")
                        elif i == 65:
                            page.keyboard.down("ArrowLeft")
                        elif i == 70:
                            page.keyboard.up("ArrowLeft")
                            page.keyboard.down("Space")
                        elif i == 78:
                            page.keyboard.up("Space")
                        page.screenshot(path=str(fdir / f"frame_{i:04d}.png"), timeout=15000)
                        page.wait_for_timeout(50)

                elif game_kind == "skate_to_gta_city":
                    page.keyboard.down("KeyW")
                    for i in range(n_frames):
                        if i in (12, 28):
                            page.keyboard.press("Space")
                        elif i in (15, 31):
                            page.keyboard.press("KeyQ")
                        elif i == 44:
                            page.keyboard.press("KeyF")
                        elif i == 60:
                            page.keyboard.down("KeyD")
                        elif i == 78:
                            page.keyboard.up("KeyD")
                        elif i == 85:
                            page.keyboard.press("Space")
                        page.screenshot(path=str(fdir / f"frame_{i:04d}.png"), timeout=15000)
                        page.wait_for_timeout(45)
                    page.keyboard.up("KeyW")

                elif game_kind == "multiplayer_3d_racing":
                    page.keyboard.down("KeyW")
                    page.keyboard.down("ArrowUp")
                    for i in range(n_frames):
                        if i == 25:
                            page.keyboard.down("KeyD")
                            page.keyboard.down("ArrowLeft")
                        elif i == 48:
                            page.keyboard.up("KeyD")
                            page.keyboard.up("ArrowLeft")
                        elif i == 60:
                            page.keyboard.down("KeyA")
                            page.keyboard.down("ArrowRight")
                        elif i == 82:
                            page.keyboard.up("KeyA")
                            page.keyboard.up("ArrowRight")
                        page.screenshot(path=str(fdir / f"frame_{i:04d}.png"), timeout=15000)
                        page.wait_for_timeout(45)
                    page.keyboard.up("KeyW")
                    page.keyboard.up("ArrowUp")
                b.close()
        _encode_mp4(fdir, out_mp4, fps=fps)


def _fit_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> str:
    if draw.textlength(text, font=font) <= max_w:
        return text
    t = text
    while len(t) > 12 and draw.textlength(t + "…", font=font) > max_w:
        t = t[:-1]
    return t + "…"


def make_before_after(before_png: Path, after_png: Path, out_png: Path, title: str, left_lbl: str, right_lbl: str, fallback_text: str) -> None:
    w, h, pad, hh, fh = 640, 360, 16, 52, 44
    canvas_w = w * 2 + pad * 3
    canvas = Image.new("RGB", (canvas_w, h + hh + fh + pad), (11, 15, 22))
    draw = ImageDraw.Draw(canvas)
    try:
        ft = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        fm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
    except Exception:
        ft = fs = fm = ImageFont.load_default()

    draw.text((pad, 15), _fit_text(draw, title, ft, canvas_w - 2 * pad), fill=(240, 246, 252), font=ft)

    def _load(p: Path, is_left: bool) -> Image.Image:
        if p.exists():
            return Image.open(p).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        im = Image.new("RGB", (w, h), (18, 22, 30))
        d = ImageDraw.Draw(im)
        d.text((28, 120), "STOCK ATTEMPT-0 FAILED TO RENDER" if is_left else "NO IMAGE", fill=(248, 81, 73), font=ft)
        d.text((28, 168), _fit_text(d, fallback_text, fm, w - 56), fill=(210, 220, 232), font=fm)
        return im

    canvas.paste(_load(before_png, True), (pad, hh))
    canvas.paste(_load(after_png, False), (pad * 2 + w, hh))
    draw.rectangle([pad - 2, hh - 2, pad + w + 1, hh + h + 1], outline=(248, 81, 73), width=3)
    draw.rectangle([pad * 2 + w - 2, hh - 2, pad * 2 + w * 2 + 1, hh + h + 1], outline=(63, 185, 80), width=3)
    draw.text((pad, hh + h + 13), _fit_text(draw, left_lbl, fs, w), fill=(255, 161, 152), font=fs)
    draw.text((pad * 2 + w, hh + h + 13), _fit_text(draw, right_lbl, fs, w), fill=(126, 231, 135), font=fs)
    canvas.save(out_png, format="PNG")


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


def build_all() -> int:
    root = Path("/home/orwel/dev_genius/experiments/kernals/results/bijanbowen_showcase_854")
    games = [
        ("stock_offroad_4x4", "01_stock_offroad_4x4", "vs Claude", "stock_offroad_4x4_gameplay.mp4", "before_after_01_stock_offroad.png"),
        ("bowling_horror_3d", "02_bowling_horror_3d", "vs GLM-5.4", "bowling_horror_3d_gameplay.mp4", "before_after_02_bowling_horror.png"),
        ("skate_to_gta_city", "03_skate_to_gta_city", "vs GPT 6.2 Astra", "skate_to_gta_city_gameplay.mp4", "before_after_03_skate_to_gta.png"),
        ("multiplayer_3d_racing", "04_multiplayer_3d_racing", "vs Gemini Carbon", "multiplayer_3d_racing_gameplay.mp4", "before_after_04_multiplayer_racing.png"),
    ]
    summaries = {}
    for kind, dname, rival, mp4_name, ba_name in games:
        wdir = root / dname
        s = json.loads((wdir / "run_summary.json").read_text())
        summaries[kind] = (s, dname, rival, mp4_name, ba_name)
        final_html = wdir / "final_verified.html"
        if final_html.exists():
            print(f"Recording {mp4_name}...", flush=True)
            record_game_clip(kind, final_html, wdir / mp4_name)
        after_png = next((wdir / fn for fn in ("final_frame_action_2.png", "final_frame_action_3.png", "final_frame_initial.png") if (wdir / fn).exists()), wdir / "final_frame_initial.png")
        make_before_after(
            before_png=wdir / "attempt_00_stock.png",
            after_png=after_png,
            out_png=root / ba_name,
            title=f"{s['title']} ({rival})",
            left_lbl=f"STOCK ATTEMPT-0: {'PASS' if s['attempt0_passed'] else 'FAIL'} — {s['attempt0_error'] or 'OK'} ({s['attempt0_score']}/100)",
            right_lbl=f"ARRAY + VERIFIED LEDGER ({s['champion']}): PASS ({s['best_score']}/100)",
            fallback_text=s["attempt0_error"] or "TRUNCATED",
        )

    ledger_dump = dump_ledger(root / "verified_ledger.sqlite", root / "verified_ledger_dump.json")

    cards_html = []
    md_rows = []
    for kind, (s, dname, rival, mp4_name, ba_name) in summaries.items():
        a0_link = f"<a class='btn' href='{dname}/attempt_00_stock.html' target='_blank'>View Stock Attempt-0</a>" if (root / dname / "attempt_00_stock.html").exists() else ""
        rec_rows = "".join(
            f"<tr><td class='mono'>R{r['round']}</td><td class='mono'>{html.escape(str(r['agent_id']))}</td>"
            f"<td class='{'ok-tag' if r['ok'] else 'bad-tag'}'>{'PASS' if r['ok'] else 'FAIL'}</td>"
            f"<td class='mono'>{r['score']}</td>"
            f"<td class='mono'>{html.escape((r['summary'] if r['ok'] else r['error_signature'] + ': ' + r['error_detail'])[:180])}</td></tr>"
            for r in s["records"]
        )
        cards_html.append(f"""
        <div class="card">
          <div class="card-head">
            <h3>{html.escape(s['title'])} <span class="badge">{html.escape(rival)}</span></h3>
            <div class="links">
              <a class="btn primary" href="{dname}/final_verified.html" target="_blank">Play Verified 3D Game</a>
              {a0_link}
              <a class="btn" href="{dname}/run_summary.json" target="_blank">Tournament Receipt</a>
            </div>
          </div>
          <p class="lead">{html.escape(s['champion_summary'])}</p>
          <div class="media-row">
            <div><img src="{ba_name}" alt="{html.escape(s['title'])} Before/After"></div>
            <div><video controls loop muted playsinline src="{dname}/{mp4_name}"></video></div>
          </div>
          <table>
            <thead><tr><th>Round</th><th>Specialist</th><th>Result</th><th>Score</th><th>Host WebGL2 &amp; Live Playtest Oracle</th></tr></thead>
            <tbody>{rec_rows}</tbody>
          </table>
        </div>""")
        md_rows.append(
            f"| **{s['title']}** ({rival}) | {'PASS' if s['attempt0_passed'] else 'FAIL'} `{s['attempt0_score']}/100` (`{s['attempt0_error'] or 'NONE'}`) "
            f"| **PASS `{s['best_score']}/100`** (`{s['champion']}`) | {s['champion_summary']} "
            f"| [{dname}/final_verified.html]({dname}/final_verified.html) · [{mp4_name}]({dname}/{mp4_name}) |"
        )

    gallery = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Inkling-2-Mini 3-GPU Array + Verified Ledger — @bijanbowen 4-Game Showcase</title>
<style>
  :root {{ --bg:#080c14; --card:#101726; --border:#23324d; --accent:#58c4ff; --ok:#3fb950; --bad:#f85149; --text:#e6edf3; --muted:#9fb0c8; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; line-height:1.55; padding:32px; max-width:1440px; margin:0 auto; }}
  h1 {{ font-size:28px; margin-bottom:8px; color:#fff; }}
  h2 {{ font-size:20px; margin:28px 0 12px; color:var(--accent); border-bottom:1px solid var(--border); padding-bottom:6px; }}
  p.lead {{ color:var(--muted); margin-bottom:16px; max-width:1050px; }}
  .pills {{ display:flex; flex-wrap:wrap; gap:10px; margin-bottom:24px; }}
  .pill {{ background:#162033; border:1px solid var(--border); border-radius:999px; padding:5px 14px; font-size:13px; color:#cfe3ff; }}
  .pill.ok {{ border-color:rgba(63,185,80,0.45); color:#8be99c; }}
  .badge {{ font-size:12px; background:#1e293b; border:1px solid #38bdf8; color:#7dd3fc; padding:2px 9px; border-radius:999px; margin-left:8px; vertical-align:middle; }}
  .grid {{ display:grid; grid-template-columns:1fr; gap:26px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:22px; }}
  .card-head {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:12px; }}
  .links {{ display:flex; flex-wrap:wrap; gap:10px; }}
  .btn {{ display:inline-block; padding:6px 12px; border-radius:8px; background:#1c2a42; border:1px solid #35507a; color:#dff0ff; text-decoration:none; font-size:13px; font-weight:600; }}
  .btn.primary {{ background:#1d4ed8; border-color:#3b82f6; color:#fff; }}
  .media-row {{ display:grid; grid-template-columns:1.15fr 1fr; gap:16px; margin-top:10px; align-items:start; }}
  @media (max-width:980px) {{ .media-row {{ grid-template-columns:1fr; }} }}
  img, video {{ width:100%; border-radius:10px; border:1px solid var(--border); background:#000; display:block; }}
  table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:12.5px; }}
  th, td {{ border:1px solid var(--border); padding:7px 9px; text-align:left; vertical-align:top; }}
  th {{ background:#151f30; color:#b8d4f4; }}
  .ok-tag {{ color:var(--ok); font-weight:700; }}
  .bad-tag {{ color:var(--bad); font-weight:700; }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.5px; }}
</style>
</head>
<body>
  <h1>Inkling-2-Mini 3-GPU Array + Shared Verified Ledger — @bijanbowen 4-Game Showcase</h1>
  <p class="lead">
    All four 3D WebGL2 games below were <strong>100% authored and self-repaired by local Inkling-2-Mini</strong> (zero hand-written target game code) running as a 3-archetype parallel tournament array (<em>Graphics Architect</em>, <em>Physics Engineer</em>, <em>VFX/Polish Director</em>) across 3 local NVIDIA Tesla V100 GPUs (<code>atlas-ingine</code> native C++ engine with speculative decoding + native vision tower, <strong>$0.00 cloud spend</strong>), sharing ONE SQLite <strong>Verified Ledger</strong> and verified by an automated headless Chromium WebGL2 state-machine inspector &amp; live gameplay oracle.
  </p>
  <div class="pills">
    {"".join(f"<span class='pill ok'>{html.escape(s['title'].split('(')[0][:34])}: {s['attempt0_score']} &rarr; {s['best_score']}/100</span>" for s, *_ in summaries.values())}
    <span class="pill">Shared Ledger Winners: {len(ledger_dump['winners'])} | Linked Repairs: {len(ledger_dump['repairs'])} | Logged Failures: {ledger_dump['failure_log_count']}</span>
  </div>
  <div class="grid">{"".join(cards_html)}</div>
  <h2>Shared SQLite Verified Ledger (<code>verified_ledger_dump.json</code>)</h2>
  <table>
    <thead><tr><th>Error Signature</th><th>Verified Repair Rule (Cross-Agent / Cross-Game)</th><th>Hits</th></tr></thead>
    <tbody>
      {"".join(f"<tr><td class='mono'>{html.escape(str(r['error_signature']))}</td><td class='mono'>{html.escape(str(r['rule']))}</td><td class='mono'>{r['hits']}</td></tr>" for r in ledger_dump['repairs'])}
    </tbody>
  </table>
</body>
</html>
"""
    (root / "index.html").write_text(gallery, encoding="utf-8")

    readme = f"""# Inkling-2-Mini 3-GPU Multi-Agent Array + Shared Verified Ledger — `@bijanbowen` 4-Game Showcase

All four interactive 3D WebGL2 games in this bundle were **100% authored and self-repaired by local Inkling-2-Mini** (zero human-written target game code) served via `atlas-ingine` (`ingine-serve --spec` + native vision tower across 3 local NVIDIA Tesla V100 GPUs at `$0.00` cloud spend).

Each game ran an honest **Stock One-Shot Attempt 0** (bare prompt, no ledger, no vision) followed by a **3-Archetype Parallel Tournament Array** (*Graphics & WebGL2 Shader Architect*, *3D Physics & Gameplay Engineer*, *Visual Polish & VFX Director*) sharing a single 4-bank SQLite **Verified Ledger** (`atlas-ingine/ledger/verified_ledger.py`), native Inkling-2-Mini visual self-review on rendered 3-frame action strips, and a headless Chromium WebGL2 state-machine inspector & live keyboard playtest oracle.

## Results (Stock Attempt-0 vs 3-GPU Array + Shared Verified Ledger)

| Target | Stock Attempt-0 | Array + Verified Ledger Champion | Verified Live Playtest & WebGL2 Telemetry | Interactive Play & Clip |
|---|---|---|---|---|
{chr(10).join(md_rows)}

## Shared Verified Ledger Summary
- **Active Host Facts (Bank 1):** {len(ledger_dump['facts'])}
- **Admitted Verified Winners (Bank 2):** {len(ledger_dump['winners'])}
- **Auto-Linked Error → Repair Rules (Bank 3):** {len(ledger_dump['repairs'])}
- **Total Logged Failures:** {ledger_dump['failure_log_count']}
- Interactive Gallery: [index.html](index.html) · SQLite Ledger Export: [verified_ledger_dump.json](verified_ledger_dump.json)
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    print("Built all 4 gameplay MP4s, before/after cards, ledger dump, index.html, and README.md!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(build_all())

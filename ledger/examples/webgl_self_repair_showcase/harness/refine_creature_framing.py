#!/usr/bin/env python3
"""Continue 02_articulated_creature with the upgraded CAMERA_TOO_CLOSE_CROPPED framing verifier on GPU 5."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from showcase_loop import (  # noqa: E402
    TASK_SPECS,
    VerifiedLedger,
    VerifyResult,
    _SshTunnel,
    call_inkling,
    compact_verifier_feedback,
    encode_image_b64,
    extract_html,
    make_contact_sheet,
)
from verifier import verify_html  # noqa: E402


def main() -> int:
    out_root = Path("/home/orwel/dev_genius/experiments/kernals/results/loktar_showcase_848")
    work_dir = out_root / "02_articulated_creature"
    ledger_path = out_root / "verified_ledger.sqlite"
    summary_path = work_dir / "run_summary.json"
    trace_path = work_dir / "ledger_trace.jsonl"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = summary["records"]
    spec = TASK_SPECS["articulated_creature"]
    cur_html = (work_dir / "final_verified.html").read_text(encoding="utf-8")
    prev_dir = work_dir / f"iter_{summary['best_iter']:02d}"

    # Re-evaluate current best under upgraded framing check
    recheck_dir = work_dir / "framing_audit"
    audit_report = verify_html("articulated_creature", work_dir / "final_verified.html", recheck_dir)
    print(f"[creature framing audit] ok={audit_report['ok']} score={audit_report['score']} err={audit_report['error_signature']}", flush=True)
    if audit_report["ok"] and audit_report["metrics"]["initial_image"].get("top_edge_fg", 1.0) <= 0.10:
        print("Already cleanly framed!", flush=True)
        return 0

    with _SshTunnel("v100", 18602, 18602) as base_url:
        with VerifiedLedger(ledger_path, env_version="v1_chromium1248_webgl2", max_context_chars=3800) as ledger:
            ledger.conn.execute("PRAGMA busy_timeout = 30000;")
            ledger_ctx = ledger.render_context(spec["prompt"], max_chars=3400, max_winners=1, max_repairs=5, winner_excerpt_chars=400)

            last_html = cur_html
            last_report = audit_report
            last_dir = recheck_dir
            start_iter = len(records)

            for offset in range(3):
                it = start_iter + offset
                iter_dir = work_dir / f"iter_{it:02d}"
                iter_dir.mkdir(parents=True, exist_ok=True)
                feedback = compact_verifier_feedback(last_report, "articulated_creature")
                sys_msg = (
                    "You are Inkling-2-Mini inside an automated Verified Ledger + Headless Playwright WebGL2 self-improvement loop. "
                    "Inspect the host verifier telemetry AND the attached rendered screenshot(s) carefully, then return a complete "
                    "upgraded single-file HTML inside a single ```html ... ``` block.\n\n" + ledger_ctx
                )
                user_text = (
                    f"ORIGINAL TASK:\n{spec['prompt']}\n\n"
                    f"=== VISUAL FRAMING & QUALITY SELF-REPAIR (iter_{it:02d}) ===\n"
                    f"{feedback}\n\n"
                    "Look closely at the attached screenshot of your current 31-joint Celestial Dragon:\n"
                    "- The camera is currently too close and viewing from behind/above, clipping the head and wings off the top edge of the 1280x720 viewport.\n"
                    "- Fix the default camera distance, elevation, target center, and yaw to a dramatic 3/4 side-front hero angle so the ENTIRE articulated dragon "
                    "(sculpted horned head & moving jaw, arched neck, flapping wings, armored torso, all 4 multi-jointed legs/paws on the stage, and sweeping articulated tail) "
                    "fits comfortably inside the center of the 1280x720 frame with generous margin around all 4 edges (top_edge_fg < 0.05)!\n"
                    "- Keep ALL 31+ hierarchical joints, continuous animation, gait buttons, sliders, orbit drag, and window.__CREATURE_DEBUG__ working with 0 errors.\n\n"
                    f"CURRENT HTML CODE:\n```html\n{last_html[:26000]}\n```"
                )
                sheet = iter_dir / "input_vision_sheet.png"
                shots = [(n, last_dir / n) for n in ["frame_initial.png", "frame_anim_t2.png", "frame_orbit.png"] if (last_dir / n).exists()]
                make_contact_sheet(shots[:3], sheet)
                b64 = encode_image_b64(sheet, max_w=960, max_h=320)
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]},
                ]

                print(f"[articulated_creature] iter={it} (verified_ledger_framing_repair) calling Inkling-2-Mini...", flush=True)
                llm = call_inkling(base_url, "inkling-2-mini", messages, max_tokens=16384, temperature=0.6, reasoning_effort=0.15)
                (iter_dir / "llm_response.json").write_text(json.dumps({"iter": it, "mode": "verified_ledger_framing_repair", "id": llm["id"], "elapsed_s": llm["elapsed_s"], "finish_reason": llm["finish_reason"], "usage": llm["usage"]}, indent=2), encoding="utf-8")
                (iter_dir / "raw_output.txt").write_text(llm["content"], encoding="utf-8")

                html_code = extract_html(llm["content"])
                html_path = iter_dir / "app.html"
                if not html_code or (llm["finish_reason"] == "length" and "</html>" not in html_code.lower()):
                    rep = {"ok": False, "score": 2.0, "error_signature": "TOKEN_TRUNCATION", "error_detail": "Truncated before </html>", "summary": "", "metrics": {}}
                else:
                    html_path.write_text(html_code, encoding="utf-8")
                    rep = verify_html("articulated_creature", html_path, iter_dir)

                vres = VerifyResult(ok=bool(rep["ok"]), score=float(rep["score"]), error_signature=str(rep.get("error_signature") or ""), error_detail=str(rep.get("error_detail") or ""), summary=str(rep.get("summary") or ""), tags=["articulated_creature", "framing_repair"])
                upd = ledger.record_outcome(spec["task_id"], spec["title"], html_code or "", vres, recent_error_sig="CAMERA_TOO_CLOSE_CROPPED" if vres.ok else "")
                rec = {
                    "iter": it,
                    "mode": "verified_ledger_framing_repair",
                    "ok": vres.ok,
                    "score": vres.score,
                    "error_signature": vres.error_signature,
                    "error_detail": vres.error_detail[:350],
                    "summary": vres.summary,
                    "finish_reason": llm["finish_reason"],
                    "elapsed_s": llm["elapsed_s"],
                    "usage": llm["usage"],
                    "pivot_triggered": False,
                    "ledger_update": upd,
                    "html_path": str(html_path) if html_path.exists() else "",
                }
                records.append(rec)
                with trace_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[articulated_creature] iter={it} ok={vres.ok} score={vres.score:.1f} err={vres.error_signature or 'NONE'} top_edge_fg={rep.get('metrics', {}).get('initial_image', {}).get('top_edge_fg')}", flush=True)

                if vres.ok:
                    shutil.copy2(html_path, work_dir / "final_verified.html")
                    for fn in ["frame_initial.png", "frame_anim_t1.png", "frame_anim_t2.png", "frame_orbit.png"]:
                        src = iter_dir / fn
                        if src.exists():
                            shutil.copy2(src, work_dir / f"final_{fn}")
                    summary["best_iter"] = it
                    summary["best_score"] = round(vres.score, 2)
                    summary["total_iters"] = len(records)
                    summary["ledger_stats"] = ledger.stats()
                    summary["records"] = records
                    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                    break
                else:
                    if html_code:
                        last_html = html_code
                    last_report = rep
                    last_dir = iter_dir
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

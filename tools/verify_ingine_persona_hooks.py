#!/usr/bin/env python3
"""End-to-end GPU verification suite for Track 5 (feat/mech-interp-persona) in atlas-ingine.

Gates verified on live Inkling-2-Mini-ingine:
  Gate 1: Exact-0.0 bit-identical no-op table across ALL SIX required visits
          {3, 6, 10, 18, 29, 44} (prelude 3, loop entry 6, mid pass-0 10,
          first re-entry / loop_msg boundary 18, late pass-1 29, coda 44)
          through the exact same fused path for every hook type, plus disabled
          fast-path and hooked CUDA-graph vs hooked-eager bitlock.
  Gate 2: Active causal interventions (ablate, force, slot-permute noise floor,
          cross-prompt route-replay, cross-prompt loop_msg depth-replace,
          3b neuron affine + simultaneous 3a readback & control-neuron check,
          residual steer with d_2027 guard).
  Gate 3: Cross-request cache-hit & parked-conversation persona isolation in
          EngineGenerator (neutral B alone vs B after A/C, short checkpoint &
          long >=256-token parked paths, same-persona reuse intact).
  Gate 4: Live ingine-serve HTTP control plane (--persona-spec, --refusal-map,
          strict provenance/6-visit/d_2027/trained/refusal rejection,
          GET /v1/personas, per-request persona, persona_metadata, HTTP-level
          cross-request cache isolation).
  Gate 5: Optional matched-prompt parity comparison against atlas-eval-dump
          (explicitly labelled as separate pools: ingine NVFP4/BF16 vs GGUF Q8_0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REQUIRED_NOOP_VISITS = [3, 6, 10, 18, 29, 44]
HEX64 = "0123456789abcdef" * 4


def run_cmd(cmd: list[str], check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    res = subprocess.run(cmd, text=True, capture_output=True, env=env)
    if res.stdout:
        for line in res.stdout.strip().splitlines()[-15:]:
            print("  ", line, flush=True)
    if res.stderr:
        for line in res.stderr.strip().splitlines()[-15:]:
            print("  [stderr]", line, flush=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(cmd)}")
    return res


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body


def http_sse(url: str, payload: dict[str, Any], timeout: float = 60.0) -> tuple[int, list[Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    events: list[Any] = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            item = line[6:]
            if item == "[DONE]":
                events.append("[DONE]")
                break
            events.append(json.loads(item))
        return resp.status, events


def make_valid_spec(visit_a: int, expert_a: int, visit_c: int, expert_c: int, hardware: str) -> dict[str, Any]:
    def prov(purpose: str, visit: int, expert: int) -> dict[str, Any]:
        return {
            "status": "validated",
            "purpose": purpose,
            "experiment": "verify_ingine_persona_hooks",
            "evidence_file": "tools/verify_ingine_persona_hooks.py",
            "evidence_split": "confirmation",
            "reaches": "tone",
            "origin": {"character": "verify_test", "source_text": "train_prompt_01"},
            "noop_check": {
                "flag": f"--force-unit {visit}:{expert}:0.0:1.0",
                "kl": 0,
                "delta_nll": 0,
                "fused_path": "dense::router_hooked+prefill::moe_topk_hooked",
                "hardware": hardware,
                "visits": list(REQUIRED_NOOP_VISITS),
            },
            "measured_effect": {
                "summary": "causal shift above slot-permute floor",
                "metric": "logit_kl",
                "engine": "atlas-ingine",
                "weights_sha256": HEX64,
                "value": 0.15,
                "floor": 0.005,
                "floor_kind": "slot_permute_null",
                "floor_n": 4,
                "sample_count": 16,
            },
            "refusal_checked": {
                "passed": True,
                "evidence_file": "results/refusal_check.json",
                "rate_before": 0.95,
                "rate_after": 0.95,
            },
        }

    def profile(desc: str, visit: int, expert: int, gain: float) -> dict[str, Any]:
        return {
            "description": desc,
            "category": "validated",
            "model_binding": {
                "weights_sha256": HEX64,
                "quantisation": "NVFP4",
                "engine_commit": "f3848d8",
            },
            "leakage_check": {"passed": True, "evidence_file": "results/leakage_check.json"},
            "interventions": [
                {
                    "id": f"force_v{visit}_e{expert}",
                    "type": "router_force",
                    "role": "probe",
                    "visit": visit,
                    "expert": expert,
                    "bias": 100.0,
                    "gain": gain,
                    "provenance": prov(desc, visit, expert),
                }
            ],
        }

    return {
        "schema_version": "atlas-persona-spec/v1",
        "spec_metadata": {"created": "2026-11-24T01:30:00Z", "author": "verify_ingine_persona_hooks"},
        "default_profile": "none",
        "profiles": {
            "calm_a": profile("test persona A", visit_a, expert_a, 20.0),
            "calm_c": profile("test persona C", visit_c, expert_c, 20.0),
        },
    }


def verify_serve_control_plane(
    serve_bin: Path,
    model_dir: Path,
    workdir: Path,
    visit_a: int,
    expert_a: int,
    visit_c: int,
    expert_c: int,
    hardware: str,
) -> dict[str, Any]:
    spec_ok = make_valid_spec(visit_a, expert_a, visit_c, expert_c, hardware)
    spec_ok_path = workdir / "spec_valid.json"
    spec_ok_path.write_text(json.dumps(spec_ok, indent=2))
    spec_ok_sha = hashlib.sha256(spec_ok_path.read_bytes()).hexdigest()

    # 1. Negative startup checks (must exit non-zero BEFORE loading model weights; use --fake for instant startup check)
    reject_checks = {}

    # 1a. Probe category rejected in strict mode
    s_probe = json.loads(json.dumps(spec_ok))
    s_probe["profiles"]["calm_a"]["category"] = "probe"
    p_probe = workdir / "spec_probe.json"
    p_probe.write_text(json.dumps(s_probe))
    res = run_cmd([str(serve_bin), "--fake", "--model-dir", str(model_dir), "--persona-spec", str(p_probe)], check=False)
    assert res.returncode != 0, "strict mode failed to reject probe profile"
    reject_checks["strict_rejects_probe"] = True

    # 1b. Missing visit 18 from noop_check.visits rejected in strict mode
    s_miss_v = json.loads(json.dumps(spec_ok))
    s_miss_v["profiles"]["calm_a"]["interventions"][0]["provenance"]["noop_check"]["visits"] = [3, 6, 10, 29, 44]
    p_miss_v = workdir / "spec_miss_v18.json"
    p_miss_v.write_text(json.dumps(s_miss_v))
    res = run_cmd([str(serve_bin), "--fake", "--model-dir", str(model_dir), "--persona-spec", str(p_miss_v)], check=False)
    assert res.returncode != 0, "strict mode failed to reject missing noop_check visit 18"
    reject_checks["strict_rejects_missing_visit18"] = True

    # 1c. fit_method="trained" rejected even in --persona-probe-mode
    s_trained = json.loads(json.dumps(spec_ok))
    s_trained["profiles"]["calm_a"]["interventions"][0]["fit_method"] = "trained"
    p_trained = workdir / "spec_trained.json"
    p_trained.write_text(json.dumps(s_trained))
    res = run_cmd(
        [str(serve_bin), "--fake", "--model-dir", str(model_dir), "--persona-spec", str(p_trained), "--persona-probe-mode"],
        check=False,
    )
    assert res.returncode != 0, "failed to reject fit_method=trained even in probe mode"
    reject_checks["rejects_fit_method_trained"] = True

    # 1d. d_2027 residual_steer rejected even in --persona-probe-mode
    vec = [0.0] * 2304
    vec[2027] = 0.01
    s_2027 = {
        "schema_version": "atlas-persona-spec/v1",
        "spec_metadata": {"created": "2026-11-24T01:30:00Z", "author": "test"},
        "profiles": {
            "bad2027": {
                "category": "probe",
                "interventions": [
                    {
                        "id": "s2027",
                        "type": "residual_steer",
                        "role": "probe",
                        "visit": 10,
                        "site": "mlp_write",
                        "scale": 1.0,
                        "vector": vec,
                    }
                ],
            }
        },
    }
    p_2027 = workdir / "spec_2027.json"
    p_2027.write_text(json.dumps(s_2027))
    res = run_cmd(
        [str(serve_bin), "--fake", "--model-dir", str(model_dir), "--persona-spec", str(p_2027), "--persona-probe-mode"],
        check=False,
    )
    assert res.returncode != 0, "failed to reject non-zero d_2027 even in probe mode"
    reject_checks["rejects_d2027"] = True

    # 1e. Refusal-map protected unit rejected even in --persona-probe-mode
    rmap_path = workdir / "refusal_map_block.json"
    rmap_path.write_text(json.dumps({"protected_units": [{"visit": visit_a, "expert": expert_a}], "protected_coords": [2027]}))
    res = run_cmd(
        [
            str(serve_bin),
            "--fake",
            "--model-dir",
            str(model_dir),
            "--persona-spec",
            str(spec_ok_path),
            "--refusal-map",
            str(rmap_path),
            "--persona-probe-mode",
        ],
        check=False,
    )
    assert res.returncode != 0, "failed to reject refusal-map protected unit"
    reject_checks["rejects_refusal_map_unit"] = True

    # 2. Live GPU ingine-serve test with valid strict spec
    port = pick_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = workdir / "ingine_serve.log"
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            [
                str(serve_bin),
                "--engine",
                "--no-vision",
                "--model-dir",
                str(model_dir),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--max-ctx",
                "4096",
                "--persona-spec",
                str(spec_ok_path),
            ],
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            deadline = time.time() + 120.0
            ready = False
            while time.time() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"ingine-serve exited early ({proc.returncode}):\n{log_path.read_text()}")
                try:
                    st, body = http_json("GET", f"{base_url}/health", timeout=2.0)
                    if st == 200 and body.get("status") == "ok":
                        ready = True
                        break
                except Exception:
                    time.sleep(0.5)
            assert ready, f"ingine-serve did not become ready:\n{log_path.read_text()}"

            st, plist = http_json("GET", f"{base_url}/v1/personas")
            assert st == 200 and plist["object"] == "list" and plist["default"] == "none"
            assert plist["probe_mode"] is False and plist["spec_sha256"] == spec_ok_sha
            ids = {d["id"]: d for d in plist["data"]}
            assert "calm_a" in ids and "calm_c" in ids
            assert ids["calm_a"]["status"] == "validated" and ids["calm_c"]["status"] == "validated"

            common_req = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain in one short sentence why prime numbers matter in cryptography.",
                    }
                ],
                "enable_thinking": False,
                "temperature": 0.0,
                "max_tokens": 24,
            }

            st, b0 = http_json("POST", f"{base_url}/v1/chat/completions", {**common_req, "persona": "none"})
            assert st == 200 and "persona_metadata" not in b0
            text_b0 = b0["choices"][0]["message"]["content"]

            st, a0 = http_json("POST", f"{base_url}/v1/chat/completions", {**common_req, "persona": "calm_a"})
            assert st == 200 and "persona_metadata" in a0
            assert a0["persona_metadata"]["profile"] == "calm_a"
            assert a0["persona_metadata"]["status"] == "validated"
            assert a0["persona_metadata"]["spec_sha256"] == spec_ok_sha
            assert a0["persona_metadata"]["persona_fingerprint"] == ids["calm_a"]["persona_fingerprint"]
            text_a0 = a0["choices"][0]["message"]["content"]

            # Immediately request neutral "none" on the EXACT same prompt -> must match B0 bit-for-bit!
            st, b1 = http_json("POST", f"{base_url}/v1/chat/completions", {**common_req, "persona": "none"})
            assert st == 200 and "persona_metadata" not in b1
            text_b1 = b1["choices"][0]["message"]["content"]
            assert text_b1 == text_b0, f"HTTP cache leak! B_after_A != B0:\nB0={text_b0!r}\nB1={text_b1!r}"

            st, c0 = http_json("POST", f"{base_url}/v1/chat/completions", {**common_req, "persona": "calm_c"})
            assert st == 200 and c0["persona_metadata"]["profile"] == "calm_c"

            st, b2 = http_json("POST", f"{base_url}/v1/chat/completions", {**common_req, "persona": "none"})
            assert st == 200 and b2["choices"][0]["message"]["content"] == text_b0

            # Streaming SSE persona_metadata check
            st, events = http_sse(f"{base_url}/v1/chat/completions", {**common_req, "persona": "calm_a", "stream": True})
            assert st == 200 and events[-1] == "[DONE]"
            meta_chunks = [e["persona_metadata"] for e in events if isinstance(e, dict) and "persona_metadata" in e]
            assert len(meta_chunks) == 1 and meta_chunks[0]["profile"] == "calm_a"

            # Unknown persona -> 400 with param="persona"
            st, err = http_json("POST", f"{base_url}/v1/chat/completions", {**common_req, "persona": "bogus"})
            assert st == 400 and err["error"]["param"] == "persona"

            return {
                "reject_checks": reject_checks,
                "spec_sha256": spec_ok_sha,
                "personas_endpoint": plist,
                "text_b0": text_b0,
                "text_a0": text_a0,
                "b_after_a_identical": text_b1 == text_b0,
                "b_after_c_identical": b2["choices"][0]["message"]["content"] == text_b0,
                "a_differs_from_b": text_a0 != text_b0,
                "sse_persona_metadata_ok": True,
                "unknown_persona_400_ok": True,
            }
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def read_f32_bin(path: Path) -> list[float]:
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}f", raw))


def read_f16_bin(path: Path) -> list[float]:
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw)//2}e", raw))


def read_i32_bin(path: Path) -> list[int]:
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}i", raw))


def cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def run_eval_dump_parity(
    eval_dump_bin: Path,
    gguf_path: Path,
    prompt_text: str,
    prompt_ids_ingine: list[int],
    ingine_dump_dir: Path,
    workdir: Path,
) -> dict[str, Any]:
    manifest = workdir / "parity_manifest.jsonl"
    manifest.write_text(json.dumps({"id": "p0", "text": prompt_text}) + "\n")
    out_dir = workdir / "eval_dump_out"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{eval_dump_bin.parent}:{env.get('LD_LIBRARY_PATH', '')}"
    run_cmd(
        [
            str(eval_dump_bin),
            "--model",
            str(gguf_path),
            "--manifest",
            str(manifest),
            "--out",
            str(out_dir),
            "--limit",
            "1",
            "--no-moment2",
            "--preserve-topk-order",
            "--ngl",
            "99",
        ],
        env=env,
    )
    summary: dict[str, Any] = {
        "pool_label_ingine": "atlas-ingine (NVFP4/BF16 native checkpoint)",
        "pool_label_eval_dump": "atlas-eval-dump (GGUF Q8_0 llama.cpp reference)",
        "note": "Separate quantisation pools (NVFP4/BF16 vs Q8_0); compares routing agreement and ffn_moe_inp activation cosine similarity on matched chat-rendered prompt.",
        "eval_dump_files": sorted(str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.is_file()),
        "ingine_dump_files": sorted(p.name for p in ingine_dump_dir.glob("*")),
        "ingine_prompt_tokens": len(prompt_ids_ingine),
    }
    rows_path = out_dir / "rows.jsonl"
    n_ed = len(prompt_ids_ingine)
    if rows_path.exists():
        row = json.loads(rows_path.read_text().splitlines()[0])
        n_ed = int(row.get("n_tokens", n_ed))
        summary["eval_dump_prompt_tokens"] = n_ed
        summary["prompt_token_count_match"] = (n_ed == len(prompt_ids_ingine))

    n_ing = len(prompt_ids_ingine)
    route_u8 = out_dir / "topk_idx.u8"
    if route_u8.exists():
        ed_idx = list(route_u8.read_bytes())
        if len(ed_idx) >= n_ed * 48 * 8 and n_ed > 0 and n_ing > 0:
            n_cmp = min(n_ing, n_ed)
            ing_off = n_ing - n_cmp
            ed_off = n_ed - n_cmp
            overlaps: dict[str, float] = {}
            slot_exact: dict[str, float] = {}
            for v in REQUIRED_NOOP_VISITS:
                ing_idx = read_i32_bin(ingine_dump_dir / f"parity_v{v}_idx.i32")
                ov_sum = 0.0
                sl_sum = 0.0
                for t in range(n_cmp):
                    ti = ing_off + t
                    te = ed_off + t
                    row_ing = ing_idx[ti * 8 : (ti + 1) * 8]
                    row_ed = ed_idx[(te * 48 + v) * 8 : (te * 48 + v) * 8 + 8]
                    ov_sum += len(set(row_ing) & set(row_ed)) / 8.0
                    sl_sum += sum(1 for a, b in zip(row_ing, row_ed) if a == b) / 8.0
                overlaps[str(v)] = round(ov_sum / n_cmp, 4)
                slot_exact[str(v)] = round(sl_sum / n_cmp, 4)
            summary["visit_top8_overlap"] = overlaps
            summary["visit_slot_order_match"] = slot_exact

    ffn_mean_path = out_dir / "ffn_mean.f16"
    ffn_gen_path = out_dir / "ffn_gen.f16"
    H = 2304
    if ffn_mean_path.exists() and ffn_gen_path.exists():
        ed_mean = read_f16_bin(ffn_mean_path)
        ed_gen = read_f16_bin(ffn_gen_path)
        if len(ed_mean) >= 48 * H and len(ed_gen) >= 48 * H and n_ing > 0:
            cos_mean: dict[str, float] = {}
            cos_last: dict[str, float] = {}
            for v in REQUIRED_NOOP_VISITS:
                f_ing = ingine_dump_dir / f"parity_v{v}_ffn_inp.f32"
                if not f_ing.exists():
                    continue
                xm = read_f32_bin(f_ing)
                if len(xm) < n_ing * H:
                    continue
                ing_mean = [0.0] * H
                for t in range(n_ing):
                    base = t * H
                    for j in range(H):
                        ing_mean[j] += xm[base + j]
                inv = 1.0 / float(n_ing)
                ing_mean = [x * inv for x in ing_mean]
                ing_last = xm[(n_ing - 1) * H : n_ing * H]
                cos_mean[str(v)] = round(cosine_sim(ing_mean, ed_mean[v * H : (v + 1) * H]), 6)
                cos_last[str(v)] = round(cosine_sim(ing_last, ed_gen[v * H : (v + 1) * H]), 6)
            summary["ffn_moe_inp_mean_cosine"] = cos_mean
            summary["ffn_moe_inp_last_prompt_cosine"] = cos_last
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", required=True, type=Path, help="native/build-multiarch directory")
    ap.add_argument("--model-dir", required=True, type=Path, help="Inkling-2-Mini-ingine directory")
    ap.add_argument("--report", type=Path, default=None, help="optional output JSON report path")
    ap.add_argument("--eval-dump-bin", type=Path, default=None, help="optional atlas-eval-dump binary for Gate 5 parity")
    ap.add_argument("--gguf", type=Path, default=None, help="optional GGUF path for Gate 5 parity")
    ap.add_argument("--skip-serve", action="store_true", help="skip Gate 4 HTTP server test")
    args = ap.parse_args()

    probe_bin = args.build_dir / "apps" / "persona_probe" / "ingine-persona-probe"
    serve_bin = args.build_dir / "apps" / "serve" / "ingine-serve"
    assert probe_bin.exists(), f"missing {probe_bin}"
    assert serve_bin.exists(), f"missing {serve_bin}"

    with tempfile.TemporaryDirectory(prefix="ingine_persona_verify_") as td:
        workdir = Path(td)
        dump_dir = workdir / "parity_dumps"
        dump_dir.mkdir()

        prompt_a = (
            "Summarize the exact difference between adiabatic and isothermal expansion of an ideal gas "
            "in two clear sentences."
        )
        prompt_b = (
            "Write a short vivid limerick about an elderly maritime astronomer tracking an icy green comet "
            "through an antique brass refracting telescope on a windy cliffside at midnight in winter."
        )

        # ---- Discovery pass: identify active_e[v], never_e[v] for all 6 visits, plus top active neurons at v=10
        disc_batch = {
            "prompts": [{"id": "p0", "text": prompt_a, "chat": True, "enable_thinking": False, "gen_tokens": 8}],
            "cases": [
                {
                    "name": "discover_routing",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "taps": {"routing_visits": REQUIRED_NOOP_VISITS, "ffn_inp_visits": REQUIRED_NOOP_VISITS},
                    "dump_prefix": "parity",
                }
            ],
        }
        disc_in = workdir / "disc_in.json"
        disc_out = workdir / "disc_out.json"
        disc_in.write_text(json.dumps(disc_batch, indent=2))
        run_cmd(
            [
                str(probe_bin),
                "--model-dir",
                str(args.model_dir),
                "--batch",
                str(disc_in),
                "--out",
                str(disc_out),
                "--dump-dir",
                str(dump_dir),
            ]
        )
        disc = json.loads(disc_out.read_text())
        gpu_name = disc["gpu_name"]
        gpu_sm = disc["gpu_sm"]
        hardware = f"{gpu_name} (sm_{gpu_sm.replace('.', '')})"
        rtaps = disc["cases"][0]["routing_taps"]

        active_e: dict[int, int] = {}
        never_e: dict[int, int] = {}
        for v in REQUIRED_NOOP_VISITS:
            info = rtaps[str(v)]
            active_e[v] = int(info["top_expert"])
            ne = int(info["never_expert"])
            if ne < 0:
                # fallback: least-selected expert
                counts = info["counts"]
                ne = min(range(len(counts)), key=lambda i: counts[i])
            never_e[v] = ne
        print(
            f"[DISCOVER] hardware={hardware} active_e={active_e} never_e={never_e}",
            flush=True,
        )

        # ---- Build full Gate 1 + Gate 2 + Gate 3 batch (runs in ONE model load for probe + ONE for cache isolation!)
        # First discover top neurons of (v=10, active_e[10]) via a quick 3a tap in a small batch, or pick from a
        # 2-case pre-pass? Actually let's run the 3a baseline tap right inside a 2nd mini-pass? Even simpler:
        # put a 3a baseline tap in the main batch and use neurons 0, 17, 511? No, we want guaranteed non-zero mean
        # neurons like verify_persona_batch4.py! Running a 1-case 3a discovery takes ~4s. Let's check!
        disc3a_in = workdir / "disc3a_in.json"
        disc3a_out = workdir / "disc3a_out.json"
        disc3a_in.write_text(
            json.dumps(
                {
                    "prompts": [{"id": "p0", "text": prompt_a, "chat": True, "enable_thinking": False, "gen_tokens": 8}],
                    "cases": [
                        {
                            "name": "base_3a_v10",
                            "prompt": "p0",
                            "gen_tokens": 8,
                            "taps": {"expert_taps": [{"visit": 10, "expert": active_e[10], "per_token": False}]},
                        }
                    ],
                },
                indent=2,
            )
        )
        run_cmd([str(probe_bin), "--model-dir", str(args.model_dir), "--batch", str(disc3a_in), "--out", str(disc3a_out)])
        d3a = json.loads(disc3a_out.read_text())
        top_n = d3a["cases"][0]["expert_3a"][0]["top_neurons"]
        n0 = int(top_n[0]["neuron"])
        n1 = int(top_n[1]["neuron"])
        n_ctrl = next(k for k in range(1152) if k not in (n0, n1))
        print(f"[DISCOVER 3a] v=10 e={active_e[10]} n0={n0} (mean={top_n[0]['mean']:.4f}) n1={n1} (mean={top_n[1]['mean']:.4f}) ctrl={n_ctrl}", flush=True)

        cases: list[dict[str, Any]] = []

        # Disabled fast-path control
        cases.append({"name": "noop_disabled_fast_path", "prompt": "p0", "gen_tokens": 8, "use_hooks": False})
        # Read-only taps control across all 6 visits
        cases.append(
            {
                "name": "noop_readonly_taps_all6",
                "prompt": "p0",
                "gen_tokens": 8,
                "taps": {
                    "routing_visits": REQUIRED_NOOP_VISITS,
                    "ffn_inp_visits": REQUIRED_NOOP_VISITS,
                    "expert_taps": [{"visit": 10, "expert": active_e[10], "per_token": True}],
                },
                "sample_neurons": [n0, n1, n_ctrl],
            }
        )

        # Gate 1: Full 6-visit exact-0.0 no-op matrix
        for v in REQUIRED_NOOP_VISITS:
            ae = active_e[v]
            ne = never_e[v]
            cases.extend(
                [
                    {
                        "name": f"noop_ablate_never_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [{"type": "router_ablate", "visit": v, "expert": ne}],
                    },
                    {
                        "name": f"noop_force_b0_active_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [{"type": "router_force", "visit": v, "expert": ae, "bias": 0.0, "gain": 1.0}],
                    },
                    {
                        "name": f"noop_force_b0_never_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [{"type": "router_force", "visit": v, "expert": ne, "bias": 0.0, "gain": 1.0}],
                    },
                    {
                        "name": f"noop_slot_permute_id_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [{"type": "router_slot_permute", "visit": v, "perm": [0, 1, 2, 3, 4, 5, 6, 7]}],
                    },
                    {
                        "name": f"noop_self_replay_recompute_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [{"type": "route_replay", "visit": v, "copy_weights": False}],
                    },
                    {
                        "name": f"noop_self_replay_copy_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [{"type": "route_replay", "visit": v, "copy_weights": True}],
                    },
                    {
                        "name": f"noop_depth_self_ffn_read_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [{"type": "depth_replace", "visit": v, "site": "ffn_res", "mode": "read"}],
                    },
                    {
                        "name": f"noop_depth_self_ffn_value_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [{"type": "depth_replace", "visit": v, "site": "ffn_res", "mode": "value"}],
                    },
                    {
                        "name": f"noop_depth_self_ffn_both_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [{"type": "depth_replace", "visit": v, "site": "ffn_res", "mode": "both"}],
                    },
                    {
                        "name": f"noop_3b_identity_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [
                            {"type": "neuron_affine", "visit": v, "expert": ae, "neuron": n0, "gain": 1.0, "offset": 0.0}
                        ],
                    },
                    {
                        "name": f"noop_steer_zero_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [
                            {"type": "residual_steer", "visit": v, "site": "mlp_write", "coord": 42, "value": 1.0, "scale": 0.0}
                        ],
                    },
                ]
            )
            if v in (18, 29):
                cases.append(
                    {
                        "name": f"noop_depth_self_loop_msg_both_v{v}",
                        "prompt": "p0",
                        "gen_tokens": 8,
                        "interventions": [{"type": "depth_replace", "visit": v, "site": "loop_msg", "mode": "both"}],
                    }
                )

        # Output residual self-replace no-op
        cases.append(
            {
                "name": "noop_depth_self_output_both",
                "prompt": "p0",
                "gen_tokens": 8,
                "interventions": [{"type": "depth_replace", "site": "output_res", "mode": "both"}],
            }
        )

        # Gate 2: Active causal interventions + simultaneous 3b/3a + eager-vs-graph check
        cases.extend(
            [
                {
                    "name": "active_ablate_v10",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "check_eager_graph": True,
                    "interventions": [{"type": "router_ablate", "visit": 10, "expert": active_e[10]}],
                },
                {
                    "name": "active_ablate_v29",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "interventions": [{"type": "router_ablate", "visit": 29, "expert": active_e[29]}],
                },
                {
                    "name": "active_force_never_v10",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "check_eager_graph": True,
                    "interventions": [
                        {"type": "router_force", "visit": 10, "expert": never_e[10], "bias": 100.0, "gain": 1.0}
                    ],
                },
                {
                    "name": "active_force_gain2_v29",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "interventions": [
                        {"type": "router_force", "visit": 29, "expert": active_e[29], "bias": 0.0, "gain": 2.0}
                    ],
                },
                {
                    "name": "active_slot_permute_rev_v10",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "interventions": [{"type": "router_slot_permute", "visit": 10, "perm": [7, 6, 5, 4, 3, 2, 1, 0]}],
                },
                {
                    "name": "active_cross_replay_v10",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "interventions": [
                        {"type": "route_replay", "visit": 10, "copy_weights": False, "donor_prompt": "p1"}
                    ],
                },
                {
                    "name": "active_cross_depth_loop_msg_v18",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "interventions": [
                        {"type": "depth_replace", "visit": 18, "site": "loop_msg", "mode": "both", "donor_prompt": "p1"}
                    ],
                },
                {
                    "name": "active_3b_gain3_v10",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "check_eager_graph": True,
                    "interventions": [
                        {"type": "neuron_affine", "visit": 10, "expert": active_e[10], "neuron": n0, "gain": 3.0, "offset": 0.0}
                    ],
                    "taps": {"expert_taps": [{"visit": 10, "expert": active_e[10], "per_token": True}]},
                    "sample_neurons": [n0, n1, n_ctrl],
                },
                {
                    "name": "active_3b_off2p5_v10",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "interventions": [
                        {"type": "neuron_affine", "visit": 10, "expert": active_e[10], "neuron": n0, "gain": 1.0, "offset": 2.5}
                    ],
                    "taps": {"expert_taps": [{"visit": 10, "expert": active_e[10], "per_token": False}]},
                    "sample_neurons": [n0, n1, n_ctrl],
                },
                {
                    "name": "active_3b_zero_v10",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "interventions": [
                        {"type": "neuron_affine", "visit": 10, "expert": active_e[10], "neuron": n0, "gain": 0.0, "offset": 0.0}
                    ],
                    "taps": {"expert_taps": [{"visit": 10, "expert": active_e[10], "per_token": False}]},
                    "sample_neurons": [n0, n1, n_ctrl],
                },
                {
                    "name": "active_3b_multi_v10",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "interventions": [
                        {"type": "neuron_affine", "visit": 10, "expert": active_e[10], "neuron": n0, "gain": 2.0, "offset": 1.0},
                        {"type": "neuron_affine", "visit": 10, "expert": active_e[10], "neuron": n1, "gain": 0.5, "offset": -1.5},
                    ],
                    "taps": {"expert_taps": [{"visit": 10, "expert": active_e[10], "per_token": False}]},
                    "sample_neurons": [n0, n1, n_ctrl],
                },
                {
                    "name": "active_residual_steer_v10",
                    "prompt": "p0",
                    "gen_tokens": 8,
                    "check_eager_graph": True,
                    "interventions": [
                        {"type": "residual_steer", "visit": 10, "site": "mlp_write", "coord": 42, "value": 1.0, "scale": 2.0}
                    ],
                },
            ]
        )

        main_batch = {
            "prompts": [
                {"id": "p0", "text": prompt_a, "chat": True, "enable_thinking": False, "gen_tokens": 8},
                {"id": "p1", "text": prompt_b, "chat": True, "enable_thinking": False, "gen_tokens": 8},
            ],
            "cases": cases,
            "cache_isolation": {
                "visit_a": 10,
                "expert_a": never_e[10],
                "visit_c": 29,
                "expert_c": never_e[29],
                "gen_tokens": 12,
            },
        }
        main_in = workdir / "main_in.json"
        main_out = workdir / "main_out.json"
        main_in.write_text(json.dumps(main_batch, indent=2))
        run_cmd([str(probe_bin), "--model-dir", str(args.model_dir), "--batch", str(main_in), "--out", str(main_out)])
        main_res = json.loads(main_out.read_text())

        by_name = {c["name"]: c for c in main_res["cases"]}

        # ---- Assert Gate 1: EVERY noop_* case must be bit-identical (prefill AND decode), kl==0.0, delta_nll==0.0
        noop_failures = []
        noop_count = 0
        for name, c in by_name.items():
            if not name.startswith("noop_"):
                continue
            noop_count += 1
            ok = (
                c["prefill_bit_identical"] is True
                and c["decode_bit_identical"] is True
                and c["prefill_max_abs_diff"] == 0.0
                and c["decode_max_abs_diff"] == 0.0
                and c["prefill_kl"] == 0.0
                and c["decode_kl_mean"] == 0.0
                and c["prefill_delta_nll"] == 0.0
                and c["decode_delta_nll_mean"] == 0.0
            )
            if not ok:
                noop_failures.append((name, c))
        assert not noop_failures, (
            f"Gate 1 FAILED on {len(noop_failures)} no-op case(s): "
            f"names={[x[0] for x in noop_failures]} sample={noop_failures[:3]}"
        )
        print(
            f"[PASS] Gate 1 (6-visit exact-0.0 bit-identical no-op matrix): {noop_count}/{noop_count} cases "
            f"bit_identical=True kl=0.0 delta_nll=0.0 across visits {REQUIRED_NOOP_VISITS} on {hardware}",
            flush=True,
        )

        # ---- Assert Gate 2: Active causal interventions + 3b/3a consistency + eager==graph bitlock
        base_ctrl_mean = by_name["noop_readonly_taps_all6"]["expert_3a"][0]["sample_neurons"][str(n_ctrl)]["mean"]
        for cname in [
            "active_ablate_v10",
            "active_ablate_v29",
            "active_force_never_v10",
            "active_force_gain2_v29",
            "active_cross_replay_v10",
            "active_cross_depth_loop_msg_v18",
            "active_residual_steer_v10",
        ]:
            c = by_name[cname]
            kl_any = max(c["prefill_kl"], c["decode_kl_mean"])
            assert kl_any > 1e-5, f"Gate 2 FAILED: {cname} had negligible KL={kl_any}"
            if "decode_graph_matches_eager_bits" in c:
                assert c["decode_graph_matches_eager_bits"] is True, f"Gate 2 FAILED: {cname} graph != eager"

        # Slot-permute noise floor: tiny non-catastrophic difference
        sp_c = by_name["active_slot_permute_rev_v10"]
        assert math.isfinite(sp_c["prefill_kl"]) and sp_c["prefill_kl"] < 0.05, f"unexpected slot_permute KL: {sp_c}"

        # 3b + simultaneous 3a checks
        for cname in ["active_3b_gain3_v10", "active_3b_off2p5_v10", "active_3b_zero_v10", "active_3b_multi_v10"]:
            c = by_name[cname]
            kl_any = max(c["prefill_kl"], c["decode_kl_mean"])
            assert kl_any > 1e-6, f"Gate 2 3b FAILED: {cname} had KL={kl_any}"
            ns0 = c["neuron_stats"][0]
            e3a = c["expert_3a"][0]
            assert ns0["hits"] > 0 and ns0["hits"] == e3a["hits"], f"3b/3a hit mismatch in {cname}: {ns0} vs {e3a}"
            mean_3a = e3a["sample_neurons"][str(ns0["neuron"])]["mean"]
            assert abs(ns0["post_mean"] - mean_3a) < 0.05, f"3b post_mean vs 3a mismatch in {cname}: {ns0['post_mean']} vs {mean_3a}"
            ctrl_mean = e3a["sample_neurons"][str(n_ctrl)]["mean"]
            assert abs(ctrl_mean - base_ctrl_mean) < 1e-4, f"untouched control neuron drifted in {cname}: {ctrl_mean} vs {base_ctrl_mean}"
            if "decode_graph_matches_eager_bits" in c:
                assert c["decode_graph_matches_eager_bits"] is True, f"3b graph != eager in {cname}"
        print(
            f"[PASS] Gate 2 (active causal + 3b/3a simultaneous + graph==eager bitlock): "
            f"ablate_v10_kl={by_name['active_ablate_v10']['prefill_kl']:.3e} "
            f"force_never_v10_kl={by_name['active_force_never_v10']['prefill_kl']:.3e} "
            f"3b_gain3_kl={by_name['active_3b_gain3_v10']['prefill_kl']:.3e} "
            f"slot_perm_floor_kl={sp_c['prefill_kl']:.3e}",
            flush=True,
        )

        # ---- Assert Gate 3: Cross-request cache-hit & parked-conversation isolation
        ci = main_res["cache_isolation"]
        assert ci["short_a_differs_from_b"] is True, f"Gate 3 FAILED: persona A did not differ from B: {ci}"
        assert ci["short_b_after_a_identical"] is True and ci["short_b_after_a_reuse"] == "none", f"Gate 3 leak A->B: {ci}"
        assert ci["short_b_after_c_identical"] is True and ci["short_b_after_c_reuse"] == "none", f"Gate 3 leak C->B: {ci}"
        assert ci["short_a_after_b_identical"] is True and ci["short_a_after_b_reuse"] == "none", f"Gate 3 leak B->A: {ci}"
        assert ci["short_b_repeat_identical"] is True and ci["short_b_repeat_reuse"] == "checkpoint", f"Gate 3 repeat B: {ci}"
        assert ci["long_a_differs_from_b"] is True, f"Gate 3 long A == B: {ci}"
        assert ci["long_b_after_ac_identical_to_cold"] is True and ci["long_b_after_ac_reuse"] == "parked", f"Gate 3 parked B: {ci}"
        assert ci["long_a_reuse"] == "none" and ci["long_c_reuse"] == "none" and ci["long_a_after_b_reuse"] == "parked", f"Gate 3 parked A/C: {ci}"
        print(
            f"[PASS] Gate 3 (EngineGenerator cache-hit & parked-conversation isolation): "
            f"short_b_after_a=exact(reuse={ci['short_b_after_a_reuse']}) "
            f"long_b_after_ac=exact(reuse={ci['long_b_after_ac_reuse']}, tokens={ci['long_prompt_tokens']})",
            flush=True,
        )

        # ---- Gate 4: Live ingine-serve HTTP control plane
        serve_res = None
        if not args.skip_serve:
            serve_res = verify_serve_control_plane(
                serve_bin, args.model_dir, workdir, 10, never_e[10], 29, never_e[29], hardware
            )
            print(
                f"[PASS] Gate 4 (ingine-serve HTTP control plane & strict/safety rejection): "
                f"rejects={serve_res['reject_checks']} b_after_a={serve_res['b_after_a_identical']} "
                f"b_after_c={serve_res['b_after_c_identical']} a_differs={serve_res['a_differs_from_b']}",
                flush=True,
            )

        # ---- Gate 5: Optional matched-prompt parity vs atlas-eval-dump
        parity_res = None
        if args.eval_dump_bin and args.gguf:
            parity_res = run_eval_dump_parity(
                args.eval_dump_bin,
                args.gguf,
                prompt_a,
                disc["prompts"]["p0"]["prompt_ids"],
                dump_dir,
                workdir,
            )
            print(f"[PASS] Gate 5 (matched-prompt parity vs atlas-eval-dump): {parity_res.get('visit_top8_overlap', 'dumped')}", flush=True)

        full_report = {
            "status": "PASS",
            "hardware": hardware,
            "required_noop_visits": REQUIRED_NOOP_VISITS,
            "active_experts": active_e,
            "never_experts": never_e,
            "gate1_noop_cases": noop_count,
            "gate1_all_bit_identical": True,
            "gate2_summary": {
                k: {
                    "prefill_kl": by_name[k]["prefill_kl"],
                    "decode_kl_mean": by_name[k]["decode_kl_mean"],
                    "graph_eq_eager": by_name[k].get("decode_graph_matches_eager_bits"),
                }
                for k in [
                    "active_ablate_v10",
                    "active_ablate_v29",
                    "active_force_never_v10",
                    "active_force_gain2_v29",
                    "active_slot_permute_rev_v10",
                    "active_cross_replay_v10",
                    "active_cross_depth_loop_msg_v18",
                    "active_3b_gain3_v10",
                    "active_3b_off2p5_v10",
                    "active_3b_zero_v10",
                    "active_3b_multi_v10",
                    "active_residual_steer_v10",
                ]
            },
            "gate3_cache_isolation": ci,
            "gate4_serve": serve_res,
            "gate5_parity": parity_res,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(full_report, indent=2) + "\n")
            print(f"Wrote verification report to {args.report}", flush=True)

    print("VERIFY_INGINE_PERSONA_HOOKS=PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

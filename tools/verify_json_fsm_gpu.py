#!/usr/bin/env python3
"""Live GPU verification for #830 token-level JSON / JSON-Schema FSM constrained decoding."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def post_json(url: str, body: dict, timeout: float = 180.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_health(port: int, timeout: float = 180.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.0) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError(f"ingine-serve on port {port} did not become healthy in {timeout}s")


def run_suite(server_bin: str, model_dir: str, port: int) -> dict:
    results: dict = {}

    # ---- Arm 1: B=1 with DSpark --spec enabled ----
    proc1 = subprocess.Popen(
        [
            server_bin,
            "--engine",
            "--model-dir",
            model_dir,
            "--max-ctx",
            "8192",
            "--dense-fp8",
            "--kv-fp8",
            "--spec",
            "--spec-threshold",
            "0.8",
            "--no-vision",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_health(port)
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        # 1a. Plain text with --spec
        r_text1 = post_json(
            url,
            {
                "messages": [{"role": "user", "content": "Say hello in five words."}],
                "temperature": 0.0,
                "max_tokens": 64,
                "enable_thinking": False,
                "response_format": {"type": "text"},
            },
        )
        results["spec_text_before"] = r_text1["choices"][0]["message"]["content"]
        assert results["spec_text_before"].strip(), "empty text before"

        # 1b. json_object with thinking OFF (immediate constrained token 0)
        r_obj_nt = post_json(
            url,
            {
                "messages": [{"role": "user", "content": "Give me a JSON object with keys color and count for 3 green apples."}],
                "temperature": 0.0,
                "max_tokens": 128,
                "enable_thinking": False,
                "response_format": {"type": "json_object"},
            },
        )
        c_obj_nt = r_obj_nt["choices"][0]["message"]["content"]
        parsed_obj_nt = json.loads(c_obj_nt)
        assert isinstance(parsed_obj_nt, dict), f"not dict: {c_obj_nt!r}"
        results["spec_json_object_nothink"] = parsed_obj_nt

        # 1c. json_object with thinking ON (unconstrained thinking -> constrained content)
        r_obj_th = post_json(
            url,
            {
                "messages": [{"role": "user", "content": "What is 6 * 7? Return JSON {\"product\": 42}."}],
                "temperature": 0.0,
                "max_tokens": 256,
                "enable_thinking": True,
                "reasoning_effort": "low",
                "response_format": {"type": "json_object"},
            },
        )
        c_obj_th = r_obj_th["choices"][0]["message"]["content"]
        parsed_obj_th = json.loads(c_obj_th)
        assert isinstance(parsed_obj_th, dict), f"not dict: {c_obj_th!r}"
        results["spec_json_object_think"] = {
            "has_reasoning": bool(r_obj_th["choices"][0]["message"].get("reasoning_content")),
            "parsed": parsed_obj_th,
        }

        # 1d. Strict json_schema (enum + integer + array + additionalProperties: false)
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "fail"]},
                "score": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["verdict", "score", "tags"],
        }
        r_schema = post_json(
            url,
            {
                "messages": [{"role": "user", "content": "Grade this passing test with score 98 and two short tags."}],
                "temperature": 0.0,
                "max_tokens": 192,
                "enable_thinking": False,
                "response_format": {"type": "json_schema", "json_schema": {"name": "grade", "schema": schema}},
            },
        )
        c_schema = r_schema["choices"][0]["message"]["content"]
        parsed_schema = json.loads(c_schema)
        assert set(parsed_schema.keys()) == {"verdict", "score", "tags"}, f"bad keys: {parsed_schema}"
        assert parsed_schema["verdict"] in ("pass", "fail"), f"bad enum: {parsed_schema}"
        assert isinstance(parsed_schema["score"], int) and not isinstance(parsed_schema["score"], bool), f"bad int: {parsed_schema}"
        assert isinstance(parsed_schema["tags"], list) and all(isinstance(x, str) for x in parsed_schema["tags"])
        results["spec_json_schema"] = parsed_schema

        # 1e. Plain text AFTER constrained requests on the same --spec server (proves drafter resync!)
        r_text2 = post_json(
            url,
            {
                "messages": [{"role": "user", "content": "Say goodbye in four words."}],
                "temperature": 0.0,
                "max_tokens": 64,
                "enable_thinking": False,
                "response_format": {"type": "text"},
            },
        )
        results["spec_text_after"] = r_text2["choices"][0]["message"]["content"]
        assert results["spec_text_after"].strip(), "empty text after"
    finally:
        proc1.terminate()
        try:
            proc1.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc1.kill()

    # ---- Arm 2: Continuous batching --max-batch 2 (concurrent constrained + unconstrained) ----
    port2 = port + 1
    proc2 = subprocess.Popen(
        [
            server_bin,
            "--engine",
            "--model-dir",
            model_dir,
            "--max-ctx",
            "8192",
            "--dense-fp8",
            "--kv-fp8",
            "--max-batch",
            "2",
            "--no-vision",
            "--host",
            "127.0.0.1",
            "--port",
            str(port2),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_health(port2)
        url2 = f"http://127.0.0.1:{port2}/v1/chat/completions"
        batch_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "ok": {"type": "boolean"},
                "count": {"type": "integer"},
            },
            "required": ["ok", "count"],
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_json = pool.submit(
                post_json,
                url2,
                {
                    "messages": [{"role": "user", "content": "Return ok=true and count=7."}],
                    "temperature": 0.0,
                    "max_tokens": 96,
                    "enable_thinking": False,
                    "response_format": {"type": "json_schema", "json_schema": {"name": "b", "schema": batch_schema}},
                },
            )
            f_txt = pool.submit(
                post_json,
                url2,
                {
                    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "enable_thinking": False,
                    "response_format": {"type": "text"},
                },
            )
            r_bj = f_json.result()
            r_bt = f_txt.result()
        parsed_bj = json.loads(r_bj["choices"][0]["message"]["content"])
        assert set(parsed_bj.keys()) == {"ok", "count"}, f"bad batch keys: {parsed_bj}"
        assert isinstance(parsed_bj["ok"], bool) and isinstance(parsed_bj["count"], int) and not isinstance(parsed_bj["count"], bool)
        results["batch2_mixed_json_schema"] = parsed_bj
        results["batch2_mixed_text"] = r_bt["choices"][0]["message"]["content"]
        assert results["batch2_mixed_text"].strip(), "empty batch text"
    finally:
        proc2.terminate()
        try:
            proc2.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc2.kill()

    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--port", type=int, default=18830)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    res = run_suite(args.server, args.model_dir, args.port)
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print("VERIFY_JSON_FSM_GPU_ALL_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

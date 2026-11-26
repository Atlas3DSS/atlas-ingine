#!/usr/bin/env python3
"""Paired #829 server-level task & long-context evaluator (per_pass vs opt-in L11/L17 last_full).

Runs two matched B=1 cold-prefix arms (per_pass default vs last_full) on the same physical GPU,
verifying GET /health reports the expected kv_cache_mode and cold cached_tokens == 0 per task,
and grades:
  1. Multi-chunk long-context retrieval (needles placed beyond the 1024-token SWA window across
     multiple prefill chunks).
  2. Deterministic executable Python coding/debugging tasks (sandboxed subprocess unit tests).
  3. Deterministic tool-call / structured JSON contract tasks.

Reports per-arm accuracy, exact regressions & wins (per_pass -> last_full), latency, peak VRAM
for the pinned CUDA logical device 0 UUID, GPU-seconds, and estimated dollar cost.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def resolve_cuda_device0_uuid() -> tuple[str, str]:
    """Resolve logical CUDA device 0 UUID + name via libcuda.so.1 (independent of nvidia-smi PCI order)."""
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        if libcuda.cuInit(0) != 0:
            return ("unknown", "unknown")
        dev = ctypes.c_int(0)
        if libcuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            return ("unknown", "unknown")
        uuid_buf = (ctypes.c_ubyte * 16)()
        if libcuda.cuDeviceGetUuid(uuid_buf, dev) != 0:
            return ("unknown", "unknown")
        b = bytes(uuid_buf)
        uuid = (
            f"GPU-{b[0:4].hex()}-{b[4:6].hex()}-{b[6:8].hex()}-"
            f"{b[8:10].hex()}-{b[10:16].hex()}"
        )
        name_buf = ctypes.create_string_buffer(256)
        name = "unknown"
        if libcuda.cuDeviceGetName(name_buf, 256, dev) == 0:
            name = name_buf.value.decode("utf-8", errors="replace").strip()
        return (uuid, name)
    except OSError:
        return ("unknown", "unknown")


def query_gpu_mem_mib(uuid: str) -> int | None:
    if not uuid or uuid == "unknown":
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-i", uuid, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def http_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 300.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_long_context_tasks() -> list[dict[str, Any]]:
    """Build multi-chunk retrieval tasks whose target facts lie well past the 1024-token SWA window."""
    tasks: list[dict[str, Any]] = []
    specs = [
        ("lc_needle_early_4k", 180, 0.10, "QUARTZ-4192-OMEGA", "telemetry_vault_key"),
        ("lc_needle_mid_8k", 360, 0.45, "COBALT-7731-DELTA", "reactor_seal_code"),
        ("lc_needle_deep_12k", 540, 0.20, "AMBER-9084-SIGMA", "archive_master_token"),
        ("lc_multi_fact_8k", 340, 0.15, "VERMILION-6120-KAPPA", "secondary_failover_key"),
    ]
    for tid, n_blocks, frac, secret, key_name in specs:
        insert_at = max(1, int(n_blocks * frac))
        parts = [f"You are auditing a multi-section operations log ({tid}). Read carefully and answer at the end.\n"]
        for i in range(n_blocks):
            if i == insert_at:
                parts.append(
                    f"[LOG-ENTRY {i:04d} PRIORITY=CRITICAL] Verified parameter {key_name} = {secret}. "
                    f"Record this exact token verbatim without alteration.\n"
                )
            parts.append(
                f"[LOG-ENTRY {i:04d}] subsystem=mesh_{i % 19} shard={i * 13 % 97} "
                f"digest={hashlib.sha1(f'{tid}:{i}'.encode()).hexdigest()[:12]} "
                f"status=nominal latency_ms={12 + (i * 7) % 83} queue_depth={(i * 3) % 17}.\n"
            )
        prompt = (
            "".join(parts)
            + f"\nQuestion: What is the exact value of `{key_name}` recorded in the operations log above? "
            + "Reply with ONLY the exact value string."
        )
        tasks.append(
            {
                "id": tid,
                "category": "long_context_retrieval",
                "messages": [{"role": "user", "content": prompt}],
                "expected_substring": secret,
                "max_tokens": 256,
            }
        )
    return tasks


def build_coding_tasks() -> list[dict[str, Any]]:
    """Deterministic Python coding tasks verified by executing unit tests in an isolated subprocess."""
    return [
        {
            "id": "code_lru_cache",
            "category": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a Python class `LRUCache` with `__init__(self, capacity: int)`, "
                        "`get(self, key: int) -> int` (returns -1 if missing), and "
                        "`put(self, key: int, value: int) -> None` in O(1) time. "
                        "Return ONLY a single ```python code block with no extra commentary."
                    ),
                }
            ],
            "test_code": (
                "c = LRUCache(2)\n"
                "c.put(1, 10); c.put(2, 20)\n"
                "assert c.get(1) == 10\n"
                "c.put(3, 30)\n"
                "assert c.get(2) == -1 and c.get(3) == 30\n"
                "c.put(1, 15); c.put(4, 40)\n"
                "assert c.get(3) == -1 and c.get(1) == 15 and c.get(4) == 40\n"
            ),
            "max_tokens": 1024,
        },
        {
            "id": "code_interval_merge",
            "category": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` "
                        "that merges all overlapping or touching closed intervals `[start, end]` and returns them sorted "
                        "by start. Return ONLY a single ```python code block."
                    ),
                }
            ],
            "test_code": (
                "assert merge_intervals([]) == []\n"
                "assert merge_intervals([[5, 7], [1, 3], [2, 4], [4, 5], [10, 12]]) == [[1, 7], [10, 12]]\n"
                "assert merge_intervals([[1, 10], [2, 3], [4, 8]]) == [[1, 10]]\n"
            ),
            "max_tokens": 768,
        },
        {
            "id": "code_topo_sort",
            "category": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a Python function `lex_topo_sort(n: int, edges: list[tuple[int, int]]) -> list[int]` "
                        "that returns the lexicographically smallest topological ordering of vertices `0..n-1` for "
                        "directed edges `(u, v)` meaning `u` precedes `v`, or `[]` if a cycle exists. "
                        "Return ONLY a single ```python code block."
                    ),
                }
            ],
            "test_code": (
                "assert lex_topo_sort(4, [(2, 0), (2, 1), (1, 3)]) == [2, 0, 1, 3]\n"
                "assert lex_topo_sort(3, [(0, 1), (1, 2), (2, 0)]) == []\n"
                "assert lex_topo_sort(3, []) == [0, 1, 2]\n"
            ),
            "max_tokens": 1024,
        },
        {
            "id": "code_rpn_eval",
            "category": "coding",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a Python function `eval_rpn(tokens: list[str]) -> int` that evaluates Reverse Polish "
                        "Notation with `+`, `-`, `*`, `/` where integer division truncates toward zero (like C/Python `int(a/b)`). "
                        "Return ONLY a single ```python code block."
                    ),
                }
            ],
            "test_code": (
                "assert eval_rpn(['2', '1', '+', '3', '*']) == 9\n"
                "assert eval_rpn(['4', '13', '5', '/', '+']) == 6\n"
                "assert eval_rpn(['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']) == 22\n"
                "assert eval_rpn(['-7', '2', '/']) == -3\n"
            ),
            "max_tokens": 768,
        },
    ]


def build_tool_tasks() -> list[dict[str, Any]]:
    """Deterministic OpenAI tool-call tasks verifying exact function selection and arguments."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_ledger",
                "description": "Look up an exact ledger record by namespace and record_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "namespace": {"type": "string"},
                        "record_id": {"type": "string"},
                    },
                    "required": ["namespace", "record_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "schedule_worker",
                "description": "Admit a worker slot with a selected kv_cache_mode.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gpu_id": {"type": "integer"},
                        "kv_cache_mode": {"type": "string", "enum": ["per_pass", "last_full"]},
                    },
                    "required": ["gpu_id", "kv_cache_mode"],
                },
            },
        },
    ]
    return [
        {
            "id": "tool_query_ledger",
            "category": "tool_use",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Look up record `REC-829-ALPHA` in namespace `chemistry_kv` using the available tool."
                    ),
                }
            ],
            "tools": tools,
            "expected_tool": "query_ledger",
            "expected_args": {"namespace": "chemistry_kv", "record_id": "REC-829-ALPHA"},
            "max_tokens": 512,
        },
        {
            "id": "tool_schedule_worker",
            "category": "tool_use",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Admit a worker on `gpu_id` 2 using `kv_cache_mode` set to `last_full`."
                    ),
                }
            ],
            "tools": tools,
            "expected_tool": "schedule_worker",
            "expected_args": {"gpu_id": 2, "kv_cache_mode": "last_full"},
            "max_tokens": 512,
        },
    ]


def build_task_suite() -> list[dict[str, Any]]:
    return build_long_context_tasks() + build_coding_tasks() + build_tool_tasks()


def extract_python_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


def grade_task(task: dict[str, Any], response: dict[str, Any]) -> tuple[bool, str]:
    choice = response["choices"][0]
    msg = choice["message"]
    cat = task["category"]
    if cat == "long_context_retrieval":
        content = (msg.get("content") or "").strip()
        exp = task["expected_substring"]
        ok = exp in content
        return ok, (f"found {exp}" if ok else f"missing {exp} in {content[:120]!r}")
    if cat == "coding":
        code = extract_python_code(msg.get("content") or "")
        script = code + "\n\n" + task["test_code"]
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp_path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, "-I", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                return True, "unit_tests_passed"
            err = (proc.stderr or proc.stdout).strip().splitlines()
            return False, (err[-1][:200] if err else f"rc={proc.returncode}")
        except subprocess.TimeoutExpired:
            return False, "timeout"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    if cat == "tool_use":
        tcs = msg.get("tool_calls") or []
        if not tcs:
            return False, "no_tool_calls"
        fn = tcs[0].get("function") or {}
        name = fn.get("name")
        if name != task["expected_tool"]:
            return False, f"wrong_tool={name}"
        raw_args = fn.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception as exc:
            return False, f"invalid_args_json:{exc}"
        for k, v in task["expected_args"].items():
            if args.get(k) != v:
                return False, f"arg_mismatch:{k}={args.get(k)!r}!=expected {v!r}"
        return True, "tool_call_exact"
    raise ValueError(f"unknown category: {cat}")


def run_arm_against_url(
    base_url: str,
    expected_mode: str,
    tasks: list[dict[str, Any]],
    gpu_uuid: str,
    require_cold: bool = True,
    timeout: float = 300.0,
) -> dict[str, Any]:
    health = http_json("GET", f"{base_url}/health", timeout=30.0)
    actual_mode = health.get("kv_cache_mode")
    if actual_mode != expected_mode:
        raise RuntimeError(
            f"/health kv_cache_mode mismatch: expected={expected_mode!r}, actual={actual_mode!r}"
        )
    models = http_json("GET", f"{base_url}/v1/models", timeout=30.0)
    model_id = models["data"][0]["id"]

    results: list[dict[str, Any]] = []
    peak_vram_mib = query_gpu_mem_mib(gpu_uuid)
    t_arm0 = time.monotonic()

    for task in tasks:
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": task["messages"],
            "temperature": 0.0,
            "max_tokens": task["max_tokens"],
            "stream": False,
        }
        if "tools" in task:
            payload["tools"] = task["tools"]
        t0 = time.monotonic()
        resp = http_json("POST", f"{base_url}/v1/chat/completions", payload=payload, timeout=timeout)
        elapsed = time.monotonic() - t0
        mem_now = query_gpu_mem_mib(gpu_uuid)
        if mem_now is not None:
            peak_vram_mib = mem_now if peak_vram_mib is None else max(peak_vram_mib, mem_now)

        usage = resp.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if require_cold and cached not in (None, 0):
            raise RuntimeError(
                f"cold-prefix violation on task {task['id']}: cached_tokens={cached} (expected 0)"
            )
        passed, reason = grade_task(task, resp)
        results.append(
            {
                "id": task["id"],
                "category": task["category"],
                "passed": passed,
                "reason": reason,
                "finish_reason": resp["choices"][0].get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": cached,
                "completion_tokens": usage.get("completion_tokens"),
                "elapsed_seconds": round(elapsed, 4),
            }
        )

    gpu_seconds = time.monotonic() - t_arm0
    passed_count = sum(1 for r in results if r["passed"])
    return {
        "kv_cache_mode": expected_mode,
        "health": health,
        "model_id": model_id,
        "tasks_passed": passed_count,
        "tasks_total": len(results),
        "pass_rate": passed_count / max(1, len(results)),
        "gpu_seconds": round(gpu_seconds, 4),
        "peak_vram_mib": peak_vram_mib,
        "results": results,
    }


def compare_arms(per_pass: dict[str, Any], last_full: dict[str, Any], hourly_usd: float) -> dict[str, Any]:
    pp_map = {r["id"]: r for r in per_pass["results"]}
    lf_map = {r["id"]: r for r in last_full["results"]}
    regressions = [tid for tid, r in pp_map.items() if r["passed"] and not lf_map[tid]["passed"]]
    wins = [tid for tid, r in lf_map.items() if r["passed"] and not pp_map[tid]["passed"]]
    both_pass = [tid for tid, r in pp_map.items() if r["passed"] and lf_map[tid]["passed"]]
    both_fail = [tid for tid, r in pp_map.items() if not r["passed"] and not lf_map[tid]["passed"]]
    return {
        "per_pass_passed": per_pass["tasks_passed"],
        "last_full_passed": last_full["tasks_passed"],
        "total_tasks": per_pass["tasks_total"],
        "regressions": regressions,
        "wins": wins,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "per_pass_gpu_seconds": per_pass["gpu_seconds"],
        "last_full_gpu_seconds": last_full["gpu_seconds"],
        "per_pass_est_usd": round(per_pass["gpu_seconds"] * hourly_usd / 3600.0, 6),
        "last_full_est_usd": round(last_full["gpu_seconds"] * hourly_usd / 3600.0, 6),
    }


def wait_for_health(base_url: str, proc: subprocess.Popen[bytes], startup_timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ingine-serve exited early with code {proc.returncode}")
        try:
            return http_json("GET", f"{base_url}/health", timeout=5.0)
        except Exception:
            time.sleep(1.0)
    raise TimeoutError(f"ingine-serve did not become healthy within {startup_timeout}s")


def launch_and_eval_arm(
    serve_bin: str,
    model_dir: str,
    mode: str,
    max_ctx: int,
    prefill_chunk: int,
    kv_format: str,
    startup_timeout: float,
    tasks: list[dict[str, Any]],
    gpu_uuid: str,
    log_path: Path,
) -> dict[str, Any]:
    port = pick_free_port()
    cmd = [
        serve_bin,
        "--engine",
        "--model-dir",
        model_dir,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dense-fp8",
        "--no-vision",
        "--max-batch",
        "1",
        "--park-mb",
        "0",
        "--max-ctx",
        str(max_ctx),
        "--prefill-chunk",
        str(prefill_chunk),
        "--kv-cache-mode",
        mode,
        "--temperature",
        "0",
    ]
    if kv_format == "q4_0":
        cmd.append("--kv-q4")
    elif kv_format == "fp8":
        cmd.append("--kv-fp8")
    elif kv_format != "bf16":
        raise ValueError(f"unsupported kv_format: {kv_format}")

    with log_path.open("wb") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        try:
            base_url = f"http://127.0.0.1:{port}"
            wait_for_health(base_url, proc, startup_timeout)
            return run_arm_against_url(base_url, mode, tasks, gpu_uuid=gpu_uuid, require_cold=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired #829 per_pass vs last_full server task evaluator")
    ap.add_argument("--serve-bin", required=True, help="Path to ingine-serve binary")
    ap.add_argument("--model-dir", required=True, help="Path to native model pack directory")
    ap.add_argument("--kv-format", default="q4_0", choices=["q4_0", "fp8", "bf16"])
    ap.add_argument("--max-ctx", type=int, default=32768)
    ap.add_argument("--prefill-chunk", type=int, default=512, help="<=512 ensures >1 prefill chunks on long tasks")
    ap.add_argument("--startup-timeout", type=float, default=360.0)
    ap.add_argument("--hourly-usd", type=float, default=0.0, help="Optional hourly GPU rate for cost logging")
    ap.add_argument("--log-dir", default="/tmp", help="Directory for per-arm server logs")
    ap.add_argument("--out-json", required=True, help="Output JSON report path")
    args = ap.parse_args()

    gpu_uuid, gpu_name = resolve_cuda_device0_uuid()
    tasks = build_task_suite()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    arm_pp = launch_and_eval_arm(
        args.serve_bin,
        args.model_dir,
        "per_pass",
        args.max_ctx,
        args.prefill_chunk,
        args.kv_format,
        args.startup_timeout,
        tasks,
        gpu_uuid,
        log_dir / "eval_829_per_pass.log",
    )
    arm_lf = launch_and_eval_arm(
        args.serve_bin,
        args.model_dir,
        "last_full",
        args.max_ctx,
        args.prefill_chunk,
        args.kv_format,
        args.startup_timeout,
        tasks,
        gpu_uuid,
        log_dir / "eval_829_last_full.log",
    )

    report = {
        "experiment": "829_native_last_full_server_tasks",
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "config": {
            "model_dir": args.model_dir,
            "kv_format": args.kv_format,
            "max_ctx": args.max_ctx,
            "prefill_chunk": args.prefill_chunk,
            "max_batch": 1,
            "spec": False,
            "park_mb": 0,
            "temperature": 0.0,
        },
        "per_pass": arm_pp,
        "last_full": arm_lf,
        "comparison": compare_arms(arm_pp, arm_lf, args.hourly_usd),
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cmp = report["comparison"]
    print(
        f"[#829 tasks] per_pass={cmp['per_pass_passed']}/{cmp['total_tasks']} "
        f"last_full={cmp['last_full_passed']}/{cmp['total_tasks']} "
        f"regressions={cmp['regressions']} wins={cmp['wins']} -> {args.out_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

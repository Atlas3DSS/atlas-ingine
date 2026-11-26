#!/usr/bin/env python3
"""Self-contained AMD Strix Halo (gfx1151) build/test and benchmark verification harness (#835 / #859).

Uses Python 3 standard library only (zero third-party packages) so an external tester on physical
AMD Strix Halo (Ryzen AI Max+ 395 128 GB, gfx1151) hardware can reproduce:
  1. Hardware/ROCm preflight & model provenance fingerprinting (`preflight`)
  2. Clean HIP (or CPU dry-run) CMake build, CTest execution, and zero-CUDA `ldd` audit (`build-test`)
  3. Clean-room OpenAI SSE decode, cold prefill (8K/32K/131K), warm agent turn (30K-96K), and
     cold 34K startup+run benchmarks against `ingine-serve` (`bench`)
  4. Strict scorecard evaluation against the owner-pinned target floor (`score`)

Clean-room notice:
  Target floor numbers correspond to the owner-pinned minimum bar (peonist-ai/halogen-flash-server
  0.20.7 benchmark numbers ONLY; proprietary Peonist EULA — strict zero code/binary reuse).
  All prompts, timing logic, and scoring in this script are clean-room native atlas-ingine code.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricRule:
    key: str
    label: str
    unit: str
    op: str  # ">" or "<"
    threshold: float


# Must stay bit-for-bit aligned with ingine::vendor::hip::StrixHaloTargetFloor in
# native/src/core/vendor_policy.h (enforced by tools/test_strix_halo_verify.py).
STRIX_HALO_TARGET_FLOOR: tuple[MetricRule, ...] = (
    MetricRule("decode_code_tok_s", "Decode — Code", "tok/s", ">", 100.0),
    MetricRule("decode_prose_tok_s", "Decode — Prose", "tok/s", ">", 70.0),
    MetricRule("decode_edit_tok_s", "Decode — Edit", "tok/s", ">", 121.0),
    MetricRule("decode_tool_tok_s", "Decode — Tool", "tok/s", ">", 88.0),
    MetricRule("decode_multilang_tok_s", "Decode — 6 Languages", "tok/s", ">", 67.0),
    MetricRule("prefill_8k_tok_s", "Prefill — 8K", "tok/s", ">", 1983.0),
    MetricRule("prefill_32k_tok_s", "Prefill — 32K", "tok/s", ">", 1876.0),
    MetricRule("prefill_131k_tok_s", "Prefill — 131K", "tok/s", ">", 1796.0),
    MetricRule("agent_turn_30k_96k_s", "Agent Turn (30K-96K ctx, max)", "s", "<", 2.1),
    MetricRule("agent_ttft_30k_96k_s", "Agent TTFT (30K-96K ctx, max)", "s", "<", 1.2),
    MetricRule("cold_34k_s", "Cold Load + 34K Run", "s", "<", 19.0),
)

DECODE_PROMPTS: dict[str, list[dict[str, str]]] = {
    "code": [
        {
            "role": "system",
            "content": "You are a systems engineer. Write concise, complete, bug-free production code.",
        },
        {
            "role": "user",
            "content": (
                "Implement a thread-safe LRU cache with TTL expiration in Python using only the standard "
                "library. Include O(1) get/put, size eviction, monotonic-clock TTL checks, type annotations, "
                "and a unittest test suite covering expiry, promotion, and eviction."
            ),
        },
    ],
    "prose": [
        {
            "role": "system",
            "content": "You are a computer architecture writer. Explain mechanisms clearly in structured prose.",
        },
        {
            "role": "user",
            "content": (
                "Write a detailed technical explanation of how unified-memory APU architectures change the "
                "tradeoffs between memory bandwidth, KV cache quantization, page-locked weight residency, "
                "and speculative drafting during long-context autoregressive decoding."
            ),
        },
    ],
    "edit": [
        {
            "role": "system",
            "content": "You are a precise code editor. Return clean unified diffs and updated functions.",
        },
        {
            "role": "user",
            "content": (
                "Refactor the following Python ring buffer so push/pop are O(1) without list slicing, add "
                "wrap-around bounds checks, preserve FIFO ordering, and output a unified diff followed by "
                "the complete revised module:\n\n"
                "class Ring:\n"
                "    def __init__(self, cap):\n"
                "        self.cap = cap\n"
                "        self.items = []\n"
                "    def push(self, x):\n"
                "        if len(self.items) == self.cap:\n"
                "            self.items = self.items[1:]\n"
                "        self.items.append(x)\n"
                "    def pop(self):\n"
                "        v = self.items[0]\n"
                "        self.items = self.items[1:]\n"
                "        return v\n"
            ),
        },
    ],
    "tool": [
        {
            "role": "system",
            "content": (
                "You are an autonomous coding agent. Respond with a sequence of structured JSON tool call "
                "objects (one per step) using tools: read_file, grep_search, edit_file, run_tests."
            ),
        },
        {
            "role": "user",
            "content": (
                "A unit test `test_weight_pack` fails on a 36 MiB shard replacement with preserved mtime. "
                "Emit a 6-step JSON tool-call plan inspecting `weight_pack.cpp`, `safetensors_io.cpp`, "
                "verifying inode/ctime guards, applying a minimal fix, and re-running the test suite."
            ),
        },
    ],
    "multilang": [
        {
            "role": "system",
            "content": "You are a polyglot software engineer. Be concise and syntactically valid in every language.",
        },
        {
            "role": "user",
            "content": (
                "Implement a deterministic FNV-1a 64-bit hash function for UTF-8 byte slices in six "
                "languages: Python, Rust, C++20, Go, TypeScript, and PostgreSQL SQL. Include a short "
                "verification example for the input \"atlas-ingine\" in each language."
            ),
        },
    ],
}

# Deterministic technical vocabulary for clean-room synthetic long-context prefill/agent benchmarks.
_CORPUS_VOCAB: tuple[str, ...] = (
    "unified", "memory", "bandwidth", "wavefront", "tensor", "stride", "quantized", "block",
    "router", "expert", "prefix", "cache", "checkpoint", "stream", "kernel", "scheduler",
    "allocator", "arena", "hugepage", "aligned", "verify", "digest", "provenance", "header",
    "offset", "payload", "context", "window", "recurrent", "residual", "norm", "projection",
    "attention", "softmax", "logit", "sampler", "token", "batch", "slot", "pipeline",
    "latency", "throughput", "dispatch", "barrier", "register", "shared", "vector", "matrix",
    "tile", "chunk", "segment", "record", "journal", "snapshot", "atomic", "rename",
    "manifest", "shard", "index", "loader", "runtime", "engine", "server", "metric",
)


def repo_root_default() -> Path:
    return Path(__file__).resolve().parent.parent


def run_capture(cmd: list[str], cwd: Path | None = None, timeout: float = 30.0) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except FileNotFoundError:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": "not found"}
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "returncode": None, "stdout": "", "stderr": f"timeout after {timeout}s"}


def parse_gfx_targets_from_rocminfo(text: str) -> list[str]:
    targets: list[str] = []
    for match in re.finditer(r"\bgfx[0-9a-fA-F]{3,4}\b", text):
        arch = match.group(0).lower()
        if arch not in targets:
            targets.append(arch)
    return targets


def parse_meminfo_bytes(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].endswith(":"):
            key = parts[0][:-1]
            if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                try:
                    out[key] = int(parts[1]) * 1024
                except ValueError:
                    pass
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_ingpack_header_identities(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            hdr = f.read(256)
    except OSError:
        return None
    if len(hdr) < 160 or hdr[:8] != b"INGPACK1":
        return None
    flags = int.from_bytes(hdr[12:16], "little")
    header_sha = hdr[96:128].hex()
    content_sha = hdr[128:160].hex() if (flags & 1) else None
    return {
        "pack_header_sha256": header_sha,
        "pack_content_sha256": content_sha,
        "has_content_sha256": bool(flags & 1),
    }


def collect_model_provenance(model_dir: Path | None, hash_packs: bool = False) -> dict[str, Any]:
    if model_dir is None:
        return {"present": False, "path": None}
    if not model_dir.exists():
        return {"present": False, "path": str(model_dir), "error": "directory does not exist"}

    entries: list[dict[str, Any]] = []
    pack_content_sha256: str | None = None
    pack_header_sha256: str | None = None
    small_meta_names = {
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "chat_template.jinja",
    }
    for p in sorted(model_dir.iterdir()):
        if not p.is_file():
            continue
        st = p.stat()
        row: dict[str, Any] = {"name": p.name, "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns}
        if p.suffix == ".ingpack":
            ident = read_ingpack_header_identities(p)
            if ident:
                row.update(ident)
                if pack_content_sha256 is None and ident.get("pack_content_sha256"):
                    pack_content_sha256 = ident["pack_content_sha256"]
                if pack_header_sha256 is None and ident.get("pack_header_sha256"):
                    pack_header_sha256 = ident["pack_header_sha256"]
        if p.name in small_meta_names or (hash_packs and p.suffix == ".ingpack"):
            row["sha256"] = sha256_file(p)
        entries.append(row)
    return {
        "present": True,
        "path": str(model_dir.resolve()),
        "file_count": len(entries),
        "has_ingpack": any(e["name"].endswith(".ingpack") for e in entries),
        "pack_content_sha256": pack_content_sha256,
        "pack_header_sha256": pack_header_sha256,
        "files": entries,
    }


def run_preflight(repo_root: Path, model_dir: Path | None, require_gfx1151: bool, hash_packs: bool) -> dict[str, Any]:
    meminfo_path = Path("/proc/meminfo")
    meminfo = parse_meminfo_bytes(meminfo_path.read_text(encoding="utf-8", errors="replace")) if meminfo_path.exists() else {}

    cpu_model = platform.processor()
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        for line in cpuinfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                _, _, val = line.partition(":")
                cpu_model = val.strip()
                break

    rocminfo = run_capture(["rocminfo"]) if shutil.which("rocminfo") else {"returncode": None, "stdout": "", "stderr": "not found"}
    rocm_smi = run_capture(["rocm-smi", "--showproductname", "--showmeminfo", "vram"]) if shutil.which("rocm-smi") else {"returncode": None, "stdout": "", "stderr": "not found"}
    hipcc = run_capture(["hipcc", "--version"]) if shutil.which("hipcc") else {"returncode": None, "stdout": "", "stderr": "not found"}
    cmake_v = run_capture(["cmake", "--version"]) if shutil.which("cmake") else {"returncode": None, "stdout": "", "stderr": "not found"}
    ninja_v = run_capture(["ninja", "--version"]) if shutil.which("ninja") else {"returncode": None, "stdout": "", "stderr": "not found"}
    git_head = run_capture(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    git_dirty = run_capture(["git", "-C", str(repo_root), "status", "--porcelain"])

    gfx_targets = parse_gfx_targets_from_rocminfo(rocminfo.get("stdout", ""))
    gfx1151_detected = "gfx1151" in gfx_targets

    result: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "memory_bytes": meminfo,
        "rocm": {
            "rocminfo_rc": rocminfo.get("returncode"),
            "gfx_targets": gfx_targets,
            "gfx1151_detected": gfx1151_detected,
            "hipcc_version": hipcc.get("stdout", "").splitlines()[0] if hipcc.get("stdout") else None,
            "rocm_smi_rc": rocm_smi.get("returncode"),
        },
        "tools": {
            "cmake": cmake_v.get("stdout", "").splitlines()[0] if cmake_v.get("stdout") else None,
            "ninja": ninja_v.get("stdout", "").strip() or None,
        },
        "git": {
            "repo_root": str(repo_root),
            "head": git_head.get("stdout", "").strip() if git_head.get("returncode") == 0 else None,
            "dirty": bool(git_dirty.get("stdout", "").strip()) if git_dirty.get("returncode") == 0 else None,
        },
        "model": collect_model_provenance(model_dir, hash_packs=hash_packs),
        "ok": True,
    }
    if require_gfx1151 and not gfx1151_detected:
        result["ok"] = False
        result["error"] = f"gfx1151 not detected in rocminfo output (found: {gfx_targets})"
    return result


def resolve_preset_build_dir(native_dir: Path, preset: str) -> Path:
    presets_path = native_dir / "CMakePresets.json"
    if presets_path.exists():
        data = json.loads(presets_path.read_text(encoding="utf-8"))
        by_name = {p["name"]: p for p in data.get("configurePresets", [])}
        cur = by_name.get(preset)
        while cur is not None:
            bdir = cur.get("binaryDir")
            if bdir:
                return Path(bdir.replace("${sourceDir}", str(native_dir)))
            parent = cur.get("inherits")
            cur = by_name.get(parent) if isinstance(parent, str) else None
    return native_dir / f"build-{preset}"


def parse_ctest_summary(output: str) -> dict[str, Any]:
    passed = 0
    failed = 0
    skipped = 0
    tests: list[dict[str, str]] = []
    for line in output.splitlines():
        m = re.search(
            r"Test\s+#\d+:\s+(\S+)\s+\.+\s*(\*\*\*Skipped|Passed|\*\*\*Failed|\*\*\*Not Run|\*\*\*Timeout|\*\*\*Exception)",
            line,
        )
        if m:
            name, status = m.group(1), m.group(2)
            tests.append({"name": name, "status": status})
            if status == "Passed":
                passed += 1
            elif status == "***Skipped":
                skipped += 1
            else:
                failed += 1
    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total_observed": len(tests),
        "tests": tests,
    }


def check_zero_cuda_linkage(binaries: list[Path]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    clean = True
    ldd_bin = shutil.which("ldd")
    for b in binaries:
        if not b.exists():
            continue
        if not ldd_bin:
            checked.append({"binary": str(b), "status": "ldd_unavailable", "cuda_libs": []})
            continue
        res = run_capture([ldd_bin, str(b)])
        cuda_hits = [ln.strip() for ln in res.get("stdout", "").splitlines() if "cuda" in ln.lower() or "cublas" in ln.lower()]
        if cuda_hits:
            clean = False
        checked.append({"binary": str(b), "returncode": res.get("returncode"), "cuda_libs": cuda_hits})
    return {"zero_cuda_clean": clean, "checked": checked}


def run_build_and_test(
    repo_root: Path,
    preset: str,
    jobs: int,
    ctest_regex: str | None,
    model_dir: Path | None,
    extra_cmake_args: list[str],
) -> dict[str, Any]:
    native_dir = repo_root / "native"
    build_dir = resolve_preset_build_dir(native_dir, preset)

    cfg_cmd = ["cmake", "--preset", preset]
    if model_dir is not None:
        cfg_cmd.append(f"-DINGINE_TEST_MODEL_DIR={model_dir}")
    cfg_cmd.extend(extra_cmake_args)

    t0 = time.monotonic()
    cfg = run_capture(cfg_cmd, cwd=native_dir, timeout=600.0)
    if cfg["returncode"] != 0:
        return {
            "ok": False,
            "stage": "configure",
            "preset": preset,
            "build_dir": str(build_dir),
            "elapsed_s": round(time.monotonic() - t0, 3),
            "configure": cfg,
        }

    bld_cmd = ["cmake", "--build", "--preset", preset, "-j", str(jobs)]
    bld = run_capture(bld_cmd, cwd=native_dir, timeout=3600.0)
    if bld["returncode"] != 0:
        return {
            "ok": False,
            "stage": "build",
            "preset": preset,
            "build_dir": str(build_dir),
            "elapsed_s": round(time.monotonic() - t0, 3),
            "build": {"returncode": bld["returncode"], "tail": (bld["stdout"] + "\n" + bld["stderr"])[-4000:]},
        }

    ctest_cmd = ["ctest", "--test-dir", str(build_dir), "--output-on-failure"]
    if ctest_regex:
        ctest_cmd.extend(["-R", ctest_regex])
    env = os.environ.copy()
    if model_dir is not None:
        env["INGINE_MODEL_DIR"] = str(model_dir)
    proc = subprocess.run(ctest_cmd, cwd=str(native_dir), env=env, capture_output=True, text=True, timeout=1800.0, check=False)
    ctest_out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    summary = parse_ctest_summary(ctest_out)

    serve_bin = build_dir / "apps" / "serve" / "ingine-serve"
    pack_bin = build_dir / "apps" / "pack" / "ingine-pack"
    cuda_audit = check_zero_cuda_linkage([serve_bin, pack_bin])
    if preset.startswith(("cpu", "hip", "sycl")):
        cuda_ok = cuda_audit["zero_cuda_clean"]
    else:
        cuda_ok = True

    return {
        "ok": proc.returncode == 0 and cuda_ok,
        "stage": "complete",
        "preset": preset,
        "build_dir": str(build_dir),
        "serve_bin": str(serve_bin) if serve_bin.exists() else None,
        "pack_bin": str(pack_bin) if pack_bin.exists() else None,
        "elapsed_s": round(time.monotonic() - t0, 3),
        "ctest_returncode": proc.returncode,
        "ctest_summary": summary,
        "ctest_tail": ctest_out[-4000:],
        "zero_cuda_audit": cuda_audit,
    }


def make_deterministic_text(word_count: int, nonce: str, header: str) -> str:
    n_vocab = len(_CORPUS_VOCAB)
    words: list[str] = [f"[{header} nonce={nonce}]"]
    # LCG keeps deterministic non-repeating sequence across long contexts (up to 131K+ tokens).
    state = int(hashlib.sha256(f"{header}:{nonce}".encode("utf-8")).hexdigest()[:8], 16)
    for i in range(max(1, word_count)):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        w = _CORPUS_VOCAB[(state >> 16) % n_vocab]
        if i > 0 and i % 24 == 0:
            words.append(f"\nline_{i // 24:05d}: {w}")
        else:
            words.append(w)
    return " ".join(words)


def sse_chat_completion(
    base_url: str,
    model_id: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    temperature: float = 0.0,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(base_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    prefix = parsed.path.rstrip("/")
    path = f"{prefix}/v1/chat/completions"

    payload = json.dumps(
        {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")

    conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=timeout_s)
    t_start = time.monotonic()
    first_token_t: float | None = None
    first_chunk_t: float | None = None
    chunks_seen = 0
    text_parts: list[str] = []
    usage: dict[str, Any] = {}

    try:
        conn.request(
            "POST",
            path,
            body=payload,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            body = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"POST {path} failed with HTTP {resp.status}: {body[:500]}")

        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            now = time.monotonic()
            if first_chunk_t is None:
                first_chunk_t = now
            obj = json.loads(data_str)
            chunks_seen += 1
            if isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                has_tool = bool(delta.get("tool_calls"))
                if (piece or has_tool) and first_token_t is None:
                    first_token_t = now
                if isinstance(piece, str) and piece:
                    text_parts.append(piece)
        t_end = time.monotonic()
    finally:
        conn.close()

    if first_token_t is None:
        first_token_t = first_chunk_t if first_chunk_t is not None else t_end

    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached_raw = details.get("cached_tokens") if isinstance(details, dict) else None
    cached_tokens = int(cached_raw) if isinstance(cached_raw, int) else 0

    ttft_s = max(0.0, first_token_t - t_start)
    total_s = max(0.0, t_end - t_start)
    if completion_tokens > 1 and (t_end - first_token_t) > 0.0:
        decode_s = t_end - first_token_t
        decode_tok_s = (completion_tokens - 1) / decode_s
    elif completion_tokens >= 1 and total_s > 0.0:
        decode_s = total_s
        decode_tok_s = completion_tokens / decode_s
    else:
        decode_s = 0.0
        decode_tok_s = 0.0

    uncached_prompt = max(0, prompt_tokens - cached_tokens)
    prefill_tok_s = (uncached_prompt / ttft_s) if (uncached_prompt > 0 and ttft_s > 0.0) else 0.0

    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "uncached_prompt_tokens": uncached_prompt,
        "completion_tokens": completion_tokens,
        "chunks_seen": chunks_seen,
        "ttft_s": ttft_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "decode_tok_s": decode_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "output_preview": "".join(text_parts)[:240],
        "output_text": "".join(text_parts),
    }


def http_get_json(base_url: str, path: str, timeout_s: float = 10.0) -> tuple[int, Any]:
    parsed = urllib.parse.urlparse(base_url)
    scheme = parsed.scheme or "http"
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if scheme == "https" else 80)
    prefix = parsed.path.rstrip("/")
    conn_cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=timeout_s)
    try:
        conn.request("GET", f"{prefix}{path}")
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body) if body else None
        except json.JSONDecodeError:
            data = body
        return resp.status, data
    finally:
        conn.close()


def wait_for_health(base_url: str, proc: subprocess.Popen[str] | None, timeout_s: float) -> float:
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            err = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"ingine-serve exited prematurely (rc={proc.returncode}): {err[-1000:] or out[-1000:]}")
        try:
            status, _ = http_get_json(base_url, "/health", timeout_s=2.0)
            if status == 200:
                return time.monotonic() - t0
        except OSError:
            pass
        time.sleep(0.1)
    raise TimeoutError(f"Server at {base_url}/health did not become ready within {timeout_s}s")


def detect_server_runtime_info(base_url: str, model_id_override: str | None) -> dict[str, Any]:
    model_id = model_id_override or "ingine"
    backend: str | None = None
    selected_device_arch: str | None = None
    model_sha256: str | None = None

    def extract_meta(obj: Any) -> None:
        nonlocal backend, selected_device_arch, model_sha256
        if not isinstance(obj, dict):
            return
        if backend is None and isinstance(obj.get("backend"), str):
            backend = obj["backend"].strip().lower()
        for k in ("selected_device_arch", "device_arch", "arch_name", "gpu_arch"):
            if selected_device_arch is None and isinstance(obj.get(k), str) and obj[k].strip():
                selected_device_arch = obj[k].strip().lower().split(":")[0]
        for k in ("content_sha256", "model_sha256", "pack_content_sha256"):
            if model_sha256 is None and isinstance(obj.get(k), str) and obj[k].strip():
                model_sha256 = obj[k].strip().lower()

    status_h, data_h = http_get_json(base_url, "/health", timeout_s=10.0)
    if status_h == 200:
        extract_meta(data_h)
    status_m, data_m = http_get_json(base_url, "/v1/models", timeout_s=10.0)
    if status_m == 200 and isinstance(data_m, dict):
        extract_meta(data_m)
        items = data_m.get("data")
        if isinstance(items, list) and items and isinstance(items[0], dict):
            if not model_id_override and items[0].get("id"):
                model_id = str(items[0]["id"])
            extract_meta(items[0])

    return {
        "model_id": model_id,
        "backend": backend,
        "selected_device_arch": selected_device_arch,
        "content_sha256": model_sha256,
    }


def detect_model_id(base_url: str, override: str | None) -> str:
    return str(detect_server_runtime_info(base_url, override)["model_id"])


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def launch_serve_process(
    serve_bin: Path,
    model_dir: Path,
    port: int,
    fake_smoke: bool,
    extra_args: list[str],
) -> subprocess.Popen[str]:
    cmd = [
        str(serve_bin),
        "--model-dir",
        str(model_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--fake" if fake_smoke else "--engine",
        *extra_args,
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def stop_serve_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
    if proc.stdout:
        proc.stdout.close()
    if proc.stderr:
        proc.stderr.close()


# Strict eligibility bounds for the named Strix Halo target workloads.
# Short custom diagnostic runs stay available in bench["prefill_runs"] / bench["agent_runs"] /
# bench["cold_run"] / bench["diagnostic_metrics"], but cannot satisfy the named target floors.
MIN_DECODE_COMPLETION_TOKENS = 2
PREFILL_BANDS: dict[str, tuple[int, int | None]] = {
    "prefill_8k_tok_s": (8000, 12288),
    "prefill_32k_tok_s": (32000, 40960),
    "prefill_131k_tok_s": (130000, None),
}
AGENT_LOW_BAND_MIN_TOKENS = 30000
AGENT_LOW_BAND_MAX_TOKENS = 49152
AGENT_HIGH_BAND_MIN_TOKENS = 95000
AGENT_MIN_CACHE_HIT_FRACTION_OF_TURN1 = 0.80
COLD_34K_MIN_UNCACHED_TOKENS = 34000


def calibrate_words_per_token(base_url: str, model_id: str) -> float:
    # Two-point probe cancels constant chat-template/system overhead across tokenizers.
    small_words = 64
    large_words = 320
    t_small = make_deterministic_text(small_words, f"cal-s-{uuid.uuid4().hex[:8]}", "calibration")
    t_large = make_deterministic_text(large_words, f"cal-l-{uuid.uuid4().hex[:8]}", "calibration")
    r_small = sse_chat_completion(
        base_url,
        model_id,
        [{"role": "user", "content": t_small}],
        max_completion_tokens=1,
        temperature=0.0,
    )
    r_large = sse_chat_completion(
        base_url,
        model_id,
        [{"role": "user", "content": t_large}],
        max_completion_tokens=1,
        temperature=0.0,
    )
    delta_tokens = r_large["prompt_tokens"] - r_small["prompt_tokens"]
    if delta_tokens > 32:
        return (large_words - small_words) / float(delta_tokens)
    pt = r_large["prompt_tokens"]
    if pt > 32:
        return large_words / float(pt)
    return 0.75


def _observed_uncached_prompt_tokens(run: dict[str, Any]) -> tuple[int, int, int]:
    prompt_tokens = int(run.get("prompt_tokens") or 0)
    cached_tokens = int(run.get("cached_tokens") or 0)
    uncached = max(0, prompt_tokens - cached_tokens)
    return prompt_tokens, cached_tokens, uncached


def _compute_prefill_tok_s(run: dict[str, Any]) -> float | None:
    _, cached, uncached = _observed_uncached_prompt_tokens(run)
    ttft_s = run.get("ttft_s")
    if cached != 0 or uncached <= 0 or not isinstance(ttft_s, (int, float)) or ttft_s <= 0.0:
        return None
    return round(uncached / float(ttft_s), 3)


def extract_verified_metrics(bench: dict[str, Any]) -> tuple[dict[str, float | None], dict[str, str]]:
    """Derive floor-eligible metrics strictly from observed run evidence, never trusting bare labels."""
    verified: dict[str, float | None] = {rule.key: None for rule in STRIX_HALO_TARGET_FLOOR}
    notes: dict[str, str] = {}

    # 1. Named decode suites (all 5 required, >= 2 observed completion tokens and positive decode_s).
    decode_runs = bench.get("decode_runs") if isinstance(bench.get("decode_runs"), dict) else {}
    for suite in ("code", "prose", "edit", "tool", "multilang"):
        key = f"decode_{suite}_tok_s"
        run = decode_runs.get(suite)
        if not isinstance(run, dict):
            notes[key] = f"missing decode suite '{suite}'"
            continue
        prompt_toks = int(run.get("prompt_tokens") or 0)
        comp_toks = int(run.get("completion_tokens") or 0)
        decode_s = run.get("decode_s")
        tok_s = run.get("decode_tok_s")
        if prompt_toks <= 0 or comp_toks < MIN_DECODE_COMPLETION_TOKENS:
            notes[key] = f"insufficient observed decode tokens (prompt={prompt_toks}, completion={comp_toks})"
            continue
        if not isinstance(decode_s, (int, float)) or decode_s <= 0.0:
            notes[key] = "non-positive decode_s"
            continue
        if not isinstance(tok_s, (int, float)) or tok_s <= 0.0:
            notes[key] = "non-positive decode_tok_s"
            continue
        verified[key] = round(float(tok_s), 3)

    # 2. Cold prefill suites bound to distinct, non-overlapping observed uncached prompt-token bands.
    raw_prefill = bench.get("prefill_runs")
    prefill_list: list[dict[str, Any]] = []
    if isinstance(raw_prefill, dict):
        prefill_list = [v for v in raw_prefill.values() if isinstance(v, dict)]
    elif isinstance(raw_prefill, list):
        prefill_list = [v for v in raw_prefill if isinstance(v, dict)]

    for key, (lo, hi) in PREFILL_BANDS.items():
        candidates: list[tuple[int, float]] = []
        observed_in_band_cached: list[int] = []
        for run in prefill_list:
            prompt_toks, cached_toks, uncached_toks = _observed_uncached_prompt_tokens(run)
            comp_toks = int(run.get("completion_tokens") or 0)
            in_prompt_band = prompt_toks >= lo and (hi is None or prompt_toks <= hi)
            if in_prompt_band and cached_toks > 0:
                observed_in_band_cached.append(cached_toks)
                continue
            if cached_toks != 0 or comp_toks < 1:
                continue
            if uncached_toks >= lo and (hi is None or uncached_toks <= hi):
                rate = _compute_prefill_tok_s(run)
                if rate is not None:
                    candidates.append((uncached_toks, rate))
        if candidates:
            # Conservative worst-case (minimum) prefill throughput among qualifying runs in the band.
            verified[key] = min(rate for _, rate in candidates)
        elif observed_in_band_cached:
            notes[key] = f"prefill run in band was cached (cached_tokens={max(observed_in_band_cached)})"
        else:
            max_obs = max((_observed_uncached_prompt_tokens(r)[2] for r in prefill_list), default=0)
            band_desc = f"[{lo}, {hi}]" if hi is not None else f">= {lo}"
            notes[key] = f"no uncached prefill run observed in {band_desc} tokens (max uncached={max_obs})"

    # 3. Warm agent turns across the 30K-96K context span with observed Turn-2 prefix cache hits.
    raw_agent = bench.get("agent_runs")
    agent_list = [r for r in raw_agent if isinstance(r, dict)] if isinstance(raw_agent, list) else []
    long_runs: list[dict[str, Any]] = []
    agent_issue: str | None = None
    for run in agent_list:
        t1 = run.get("turn1_prime") if isinstance(run.get("turn1_prime"), dict) else None
        t2 = run.get("turn2_warm") if isinstance(run.get("turn2_warm"), dict) else None
        if t1 is None or t2 is None:
            continue
        t1_pt = int(t1.get("prompt_tokens") or 0)
        t2_pt = int(t2.get("prompt_tokens") or 0)
        if t1_pt >= AGENT_LOW_BAND_MIN_TOKENS or t2_pt >= AGENT_LOW_BAND_MIN_TOKENS:
            long_runs.append(run)

    if not long_runs:
        max_t2 = max(
            (int((r.get("turn2_warm") or {}).get("prompt_tokens") or 0) for r in agent_list),
            default=0,
        )
        agent_issue = f"no warm agent runs >= {AGENT_LOW_BAND_MIN_TOKENS} observed prompt tokens (max={max_t2})"
    else:
        has_low_band = False
        has_high_band = False
        turn_latencies: list[float] = []
        ttft_latencies: list[float] = []
        for run in long_runs:
            t1 = run["turn1_prime"]
            t2 = run["turn2_warm"]
            t1_pt = int(t1.get("prompt_tokens") or 0)
            t2_pt = int(t2.get("prompt_tokens") or 0)
            t2_cached = int(t2.get("cached_tokens") or 0)
            t2_comp = int(t2.get("completion_tokens") or 0)
            t2_ttft = t2.get("ttft_s")
            t2_total = t2.get("total_s")
            min_required_cache = int(AGENT_MIN_CACHE_HIT_FRACTION_OF_TURN1 * t1_pt)
            if t1_pt < AGENT_LOW_BAND_MIN_TOKENS or t2_pt < t1_pt:
                agent_issue = f"agent context invalid (turn1={t1_pt}, turn2={t2_pt})"
                break
            if t2_cached <= 0 or t2_cached < min_required_cache:
                agent_issue = (
                    f"warm prefix cache not observed at ctx={t2_pt} "
                    f"(cached={t2_cached}, required>={min_required_cache})"
                )
                break
            if (
                t2_comp < 1
                or not isinstance(t2_ttft, (int, float))
                or t2_ttft <= 0.0
                or not isinstance(t2_total, (int, float))
                or t2_total <= 0.0
            ):
                agent_issue = f"invalid warm turn timing/completion at ctx={t2_pt}"
                break
            if AGENT_LOW_BAND_MIN_TOKENS <= t2_pt <= AGENT_LOW_BAND_MAX_TOKENS:
                has_low_band = True
            if t2_pt >= AGENT_HIGH_BAND_MIN_TOKENS:
                has_high_band = True
            turn_latencies.append(float(t2_total))
            ttft_latencies.append(float(t2_ttft))

        if agent_issue is None and (not has_low_band or not has_high_band):
            obs_pts = sorted(int(r["turn2_warm"].get("prompt_tokens") or 0) for r in long_runs)
            agent_issue = (
                f"agent 30K-96K span incomplete: need observed ctx in "
                f"[{AGENT_LOW_BAND_MIN_TOKENS}, {AGENT_LOW_BAND_MAX_TOKENS}] and >= {AGENT_HIGH_BAND_MIN_TOKENS} "
                f"(observed={obs_pts})"
            )

        if agent_issue is None:
            verified["agent_turn_30k_96k_s"] = round(max(turn_latencies), 4)
            verified["agent_ttft_30k_96k_s"] = round(max(ttft_latencies), 4)

    if agent_issue is not None:
        notes["agent_turn_30k_96k_s"] = agent_issue
        notes["agent_ttft_30k_96k_s"] = agent_issue

    # 4. Cold process-spawn + 34K uncached run.
    cold_run = bench.get("cold_run") if isinstance(bench.get("cold_run"), dict) else {}
    if not cold_run.get("measured"):
        notes["cold_34k_s"] = str(cold_run.get("reason") or "cold run not measured")
    else:
        cold_req = cold_run.get("request") if isinstance(cold_run.get("request"), dict) else {}
        cold_total_s = cold_run.get("cold_total_s")
        _, cold_cached, cold_uncached = _observed_uncached_prompt_tokens(cold_req)
        cold_comp = int(cold_req.get("completion_tokens") or 0)
        if cold_cached != 0:
            notes["cold_34k_s"] = f"cold 34K run reported cached_tokens={cold_cached} (must be 0)"
        elif cold_uncached < COLD_34K_MIN_UNCACHED_TOKENS:
            notes["cold_34k_s"] = (
                f"observed cold uncached prompt tokens {cold_uncached} < {COLD_34K_MIN_UNCACHED_TOKENS}"
            )
        elif cold_comp < 1 or not isinstance(cold_total_s, (int, float)) or cold_total_s <= 0.0:
            notes["cold_34k_s"] = "invalid cold 34K completion or timing"
        else:
            verified["cold_34k_s"] = round(float(cold_total_s), 4)

    return verified, notes


def _is_sha256_hex(value: str | None) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", value.strip()))


def run_benchmark_suite(
    base_url: str | None,
    serve_bin: Path | None,
    model_dir: Path | None,
    model_id_override: str | None,
    fake_smoke: bool,
    extra_serve_args: list[str],
    decode_max_tokens: int,
    prefill_lengths: list[int],
    agent_lengths: list[int],
    cold_tokens: int,
    startup_timeout_s: float,
    expected_model_sha256: str | None = None,
    selected_device_arch: str | None = None,
    selected_backend: str | None = None,
) -> dict[str, Any]:
    managed_proc: subprocess.Popen[str] | None = None
    active_url = base_url
    startup_ready_s: float | None = None

    if active_url is None:
        if serve_bin is None or model_dir is None:
            raise ValueError("Provide either --base-url or both --serve-bin and --model-dir")
        port = find_free_port()
        active_url = f"http://127.0.0.1:{port}"
        managed_proc = launch_serve_process(serve_bin, model_dir, port, fake_smoke, extra_serve_args)
        startup_ready_s = wait_for_health(active_url, managed_proc, startup_timeout_s)
    else:
        wait_for_health(active_url, None, min(startup_timeout_s, 30.0))

    try:
        runtime_info = detect_server_runtime_info(active_url, model_id_override)
        model_id = str(runtime_info["model_id"])
        words_per_tok = calibrate_words_per_token(active_url, model_id)
        margin = lambda target: 1.02 if target >= 1024 else 1.0  # noqa: E731

        decode_runs: dict[str, dict[str, Any]] = {}
        for name, msgs in DECODE_PROMPTS.items():
            # Unique nonce in system message prevents cross-run caching across suites.
            run_msgs = [dict(m) for m in msgs]
            run_msgs[0]["content"] = f"[decode-suite={name} nonce={uuid.uuid4().hex[:10]}] " + run_msgs[0]["content"]
            res = sse_chat_completion(
                active_url,
                model_id,
                run_msgs,
                max_completion_tokens=decode_max_tokens,
                temperature=0.0,
            )
            res.pop("output_text", None)
            decode_runs[name] = res

        prefill_runs: dict[str, dict[str, Any]] = {}
        for length in prefill_lengths:
            nonce = uuid.uuid4().hex[:12]
            words = max(16, int(round(length * words_per_tok * margin(length))))
            body = make_deterministic_text(words, nonce, f"prefill-{length}")
            prompt_msgs = [
                {"role": "system", "content": "Acknowledge receipt with one word: OK."},
                {"role": "user", "content": body + "\nReply OK."},
            ]
            res = sse_chat_completion(active_url, model_id, prompt_msgs, max_completion_tokens=1, temperature=0.0)
            res.pop("output_text", None)
            res["target_tokens"] = length
            res["nonce"] = nonce
            prefill_runs[str(length)] = res

        agent_runs: list[dict[str, Any]] = []
        for length in agent_lengths:
            session_nonce = uuid.uuid4().hex[:12]
            words = max(16, int(round(length * words_per_tok * margin(length))))
            repo_ctx = make_deterministic_text(words, session_nonce, f"agent-ctx-{length}")
            t1_msgs = [
                {"role": "system", "content": "You are a concise coding agent."},
                {"role": "user", "content": repo_ctx + "\nIndex this workspace and state ready."},
            ]
            t1 = sse_chat_completion(active_url, model_id, t1_msgs, max_completion_tokens=8, temperature=0.0)
            t1_reply = t1.pop("output_text", "") or "ready"
            t2_msgs = [
                *t1_msgs,
                {"role": "assistant", "content": t1_reply},
                {
                    "role": "user",
                    "content": (
                        "Tool result: `ctest -R core_weight_pack` exited 0 (4/4 assertions passed). "
                        "Summarize the next action in two short bullet points."
                    ),
                },
            ]
            t2 = sse_chat_completion(
                active_url,
                model_id,
                t2_msgs,
                max_completion_tokens=min(64, decode_max_tokens),
                temperature=0.0,
            )
            t2.pop("output_text", None)
            hit_ratio = (t2["cached_tokens"] / float(t2["prompt_tokens"])) if t2["prompt_tokens"] > 0 else 0.0
            agent_runs.append(
                {
                    "target_ctx_tokens": length,
                    "session_nonce": session_nonce,
                    "turn1_prime": t1,
                    "turn2_warm": {**t2, "cache_hit_ratio": round(hit_ratio, 4)},
                }
            )
    finally:
        stop_serve_process(managed_proc)
        managed_proc = None

    cold_run: dict[str, Any]
    if base_url is None and serve_bin is not None and model_dir is not None:
        cold_port = find_free_port()
        cold_url = f"http://127.0.0.1:{cold_port}"
        cold_nonce = uuid.uuid4().hex[:12]
        cold_words = max(16, int(round(cold_tokens * words_per_tok * margin(cold_tokens))))
        cold_text = make_deterministic_text(cold_words, cold_nonce, f"cold-{cold_tokens}")
        cold_msgs = [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": cold_text + "\nSummarize in one sentence."},
        ]
        cold_t0 = time.monotonic()
        cold_proc = launch_serve_process(serve_bin, model_dir, cold_port, fake_smoke, extra_serve_args)
        try:
            cold_ready_s = wait_for_health(cold_url, cold_proc, startup_timeout_s)
            cold_model_id = detect_model_id(cold_url, model_id_override)
            cold_req = sse_chat_completion(
                cold_url,
                cold_model_id,
                cold_msgs,
                max_completion_tokens=min(16, decode_max_tokens),
                temperature=0.0,
            )
            cold_total_s = time.monotonic() - cold_t0
            cold_req.pop("output_text", None)
            cold_run = {
                "measured": True,
                "target_tokens": cold_tokens,
                "server_ready_s": round(cold_ready_s, 4),
                "cold_total_s": round(cold_total_s, 4),
                "request": cold_req,
            }
        finally:
            stop_serve_process(cold_proc)
    else:
        cold_run = {
            "measured": False,
            "reason": "UNMEASURED_EXTERNAL_SERVER",
            "cold_total_s": None,
        }

    # Raw diagnostics preserve short/custom runs without mislabeling them as qualifying Strix floors.
    diagnostic_metrics: dict[str, Any] = {
        "decode_tok_s": {k: round(v["decode_tok_s"], 3) for k, v in decode_runs.items()},
        "prefill_tok_s_by_requested_len": {
            k: {
                "observed_prompt_tokens": v["prompt_tokens"],
                "cached_tokens": v["cached_tokens"],
                "prefill_tok_s": round(v["prefill_tok_s"], 3),
            }
            for k, v in prefill_runs.items()
        },
        "agent_warm_turns": [
            {
                "requested_ctx_tokens": r["target_ctx_tokens"],
                "observed_turn1_prompt_tokens": r["turn1_prime"]["prompt_tokens"],
                "observed_turn2_prompt_tokens": r["turn2_warm"]["prompt_tokens"],
                "cached_tokens": r["turn2_warm"]["cached_tokens"],
                "cache_hit_ratio": r["turn2_warm"]["cache_hit_ratio"],
                "ttft_s": round(r["turn2_warm"]["ttft_s"], 4),
                "total_s": round(r["turn2_warm"]["total_s"], 4),
            }
            for r in agent_runs
        ],
        "cold_run": {
            "measured": bool(cold_run.get("measured")),
            "requested_tokens": cold_run.get("target_tokens"),
            "observed_prompt_tokens": (cold_run.get("request") or {}).get("prompt_tokens"),
            "cached_tokens": (cold_run.get("request") or {}).get("cached_tokens"),
            "cold_total_s": cold_run.get("cold_total_s"),
        },
    }

    pack_prov = collect_model_provenance(model_dir, hash_packs=False) if model_dir is not None else {}
    observed_sha = runtime_info.get("content_sha256") or pack_prov.get("pack_content_sha256")
    resolved_arch = (selected_device_arch or runtime_info.get("selected_device_arch") or "").strip().lower().split(":")[0] or None
    resolved_backend = (selected_backend or runtime_info.get("backend") or "").strip().lower() or None

    bench_payload: dict[str, Any] = {
        "mode": "smoke_fake_only" if fake_smoke else "live_engine",
        "fake_smoke": fake_smoke,
        "model_id": model_id,
        "accelerator": {
            "backend": resolved_backend,
            "selected_device_arch": resolved_arch,
        },
        "model_identity": {
            "model_id": model_id,
            "content_sha256": observed_sha.lower() if isinstance(observed_sha, str) else None,
            "pack_header_sha256": pack_prov.get("pack_header_sha256"),
            "expected_model_sha256": expected_model_sha256.strip().lower() if expected_model_sha256 else None,
        },
        "startup_ready_s": round(startup_ready_s, 4) if startup_ready_s is not None else None,
        "words_per_token_calibrated": round(words_per_tok, 4),
        "decode_runs": decode_runs,
        "prefill_runs": prefill_runs,
        "agent_runs": agent_runs,
        "cold_run": cold_run,
        "diagnostic_metrics": diagnostic_metrics,
    }
    verified_metrics, eligibility_notes = extract_verified_metrics(bench_payload)
    bench_payload["summary_metrics"] = verified_metrics
    bench_payload["eligibility_notes"] = eligibility_notes
    return bench_payload


def evaluate_metric(rule: MetricRule, value: float | None) -> str:
    if value is None:
        return "UNMEASURED"
    if rule.op == ">":
        return "PASS" if value > rule.threshold else "FAIL"
    return "PASS" if (value > 0.0 and value < rule.threshold) else "FAIL"


def score_report(report: dict[str, Any], expected_model_sha256: str | None = None) -> dict[str, Any]:
    preflight = report.get("preflight") or {}
    bench = report.get("bench") or {}
    mode = bench.get("mode", "unmeasured")
    fake_smoke = bool(bench.get("fake_smoke", mode == "smoke_fake_only"))
    gfx1151 = bool((preflight.get("rocm") or {}).get("gfx1151_detected", False))

    accel = bench.get("accelerator") if isinstance(bench.get("accelerator"), dict) else {}
    selected_backend = str(accel.get("backend") or "").strip().lower() or None
    selected_arch_raw = str(accel.get("selected_device_arch") or "").strip().lower()
    selected_arch = selected_arch_raw.split(":")[0] if selected_arch_raw else None
    selected_accelerator_verified = (selected_backend == "hip" and selected_arch == "gfx1151")

    model_ident = bench.get("model_identity") if isinstance(bench.get("model_identity"), dict) else {}
    preflight_model = preflight.get("model") if isinstance(preflight.get("model"), dict) else {}
    observed_sha_raw = model_ident.get("content_sha256") or preflight_model.get("pack_content_sha256")
    observed_model_sha = str(observed_sha_raw).strip().lower() if _is_sha256_hex(observed_sha_raw) else None
    expected_sha_raw = expected_model_sha256 or model_ident.get("expected_model_sha256")
    expected_model_sha = str(expected_sha_raw).strip().lower() if _is_sha256_hex(expected_sha_raw) else None
    model_identity_mismatch = (
        expected_model_sha is not None and observed_model_sha is not None and observed_model_sha != expected_model_sha
    )
    model_identity_verified = (
        expected_model_sha is not None and observed_model_sha is not None and observed_model_sha == expected_model_sha
    )

    # Always recompute eligible metrics from raw run evidence so bare/short labels cannot spoof floors.
    verified_metrics, eligibility_notes = extract_verified_metrics(bench)

    rows: list[dict[str, Any]] = []
    passed_count = 0
    failed_count = 0
    unmeasured_count = 0

    for rule in STRIX_HALO_TARGET_FLOOR:
        val = verified_metrics.get(rule.key)
        note = eligibility_notes.get(rule.key, "")
        if fake_smoke:
            status = "UNMEASURED_FAKE"
            note = "fake smoke mode" + (f"; {note}" if note else "")
        else:
            status = evaluate_metric(rule, val)
        if status == "PASS":
            passed_count += 1
        elif status == "FAIL":
            failed_count += 1
        else:
            unmeasured_count += 1
        rows.append(
            {
                "key": rule.key,
                "label": rule.label,
                "unit": rule.unit,
                "target": f"{rule.op} {rule.threshold}",
                "measured": val,
                "status": status,
                "note": note,
            }
        )

    all_metrics_pass = (not fake_smoke) and (passed_count == len(STRIX_HALO_TARGET_FLOOR))
    hardware_verified = (
        all_metrics_pass
        and mode == "live_engine"
        and gfx1151
        and selected_accelerator_verified
        and model_identity_verified
    )

    if fake_smoke:
        verdict = "UNMEASURED_FAKE_MODE"
    elif unmeasured_count > 0:
        verdict = "INCOMPLETE_UNMEASURED"
    elif not all_metrics_pass:
        verdict = "FAIL_STRIX_HALO_FLOOR"
    elif not gfx1151 or (selected_arch is not None and selected_arch != "gfx1151") or (selected_backend is not None and selected_backend != "hip"):
        verdict = "PASS_FLOOR_NON_GFX1151"
    elif model_identity_mismatch:
        verdict = "PASS_FLOOR_MODEL_IDENTITY_MISMATCH"
    elif not selected_accelerator_verified or not model_identity_verified:
        verdict = "PASS_FLOOR_UNBOUND_IDENTITY"
    else:
        verdict = "PASS_STRIX_HALO_FLOOR"

    return {
        "verdict": verdict,
        "hardware_verified": hardware_verified,
        "all_metrics_pass": all_metrics_pass,
        "gfx1151_detected": gfx1151,
        "selected_backend": selected_backend,
        "selected_device_arch": selected_arch,
        "selected_accelerator_verified": selected_accelerator_verified,
        "observed_model_sha256": observed_model_sha,
        "expected_model_sha256": expected_model_sha,
        "model_identity_verified": model_identity_verified,
        "mode": mode,
        "passed": passed_count,
        "failed": failed_count,
        "unmeasured": unmeasured_count,
        "total": len(STRIX_HALO_TARGET_FLOOR),
        "rows": rows,
    }


def render_scorecard_markdown(report: dict[str, Any], score: dict[str, Any]) -> str:
    preflight = report.get("preflight") or {}
    rocm = preflight.get("rocm") or {}
    bt = report.get("build_test") or {}
    lines = [
        "# AMD Strix Halo (`gfx1151`) Verification Scorecard (`#835` / `#859`)",
        "",
        f"- **Verdict:** `{score['verdict']}` (`hardware_verified={score['hardware_verified']}`, `mode={score['mode']}`)",
        f"- **Host / Kernel:** `{preflight.get('hostname')}` / `{preflight.get('kernel')}`",
        f"- **ROCm Targets:** `{rocm.get('gfx_targets')}` (`gfx1151_detected={rocm.get('gfx1151_detected')}`)",
        f"- **Selected Accelerator:** `backend={score.get('selected_backend')}`, `arch={score.get('selected_device_arch')}` (`verified={score.get('selected_accelerator_verified')}`)",
        f"- **Model Pack SHA-256:** `observed={score.get('observed_model_sha256')}`, `expected={score.get('expected_model_sha256')}` (`verified={score.get('model_identity_verified')}`)",
        f"- **Git HEAD:** `{((preflight.get('git') or {}).get('head'))}` (dirty=`{((preflight.get('git') or {}).get('dirty'))}`)",
    ]
    if bt:
        cs = bt.get("ctest_summary") or {}
        lines.append(
            f"- **Build & CTest (`{bt.get('preset')}`):** ok=`{bt.get('ok')}` "
            f"(passed={cs.get('passed', 0)}, failed={cs.get('failed', 0)}, skipped={cs.get('skipped', 0)}, "
            f"zero_cuda_clean=`{((bt.get('zero_cuda_audit') or {}).get('zero_cuda_clean'))}`)"
        )
    lines.extend(
        [
            "",
            "> Target floor numbers are owner-pinned benchmarks to beat (`peonist-ai/halogen-flash-server 0.20.7`",
            "> numbers only; proprietary Peonist EULA — strict zero code reuse).",
            "",
            "| Metric | Owner Floor | Measured | Unit | Status | Notes |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in score["rows"]:
        val_str = "n/a" if row["measured"] is None else f"{row['measured']}"
        note_str = row.get("note") or ""
        lines.append(
            f"| {row['label']} | `{row['target']}` | `{val_str}` | {row['unit']} | **{row['status']}** | {note_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="strix_halo_verify.py",
        description="Self-contained AMD Strix Halo (gfx1151) build/test + benchmark verifier (#835/#859).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("preflight", help="Collect hardware, ROCm, tool, git, and model provenance.")
    pf.add_argument("--repo-root", type=Path, default=repo_root_default())
    pf.add_argument("--model-dir", type=Path, default=None)
    pf.add_argument("--require-gfx1151", action="store_true")
    pf.add_argument("--hash-packs", action="store_true")
    pf.add_argument("--out", type=Path, default=None)

    bt = sub.add_parser("build-test", help="Configure, build, run CTest, and check zero-CUDA linkage.")
    bt.add_argument("--repo-root", type=Path, default=repo_root_default())
    bt.add_argument("--preset", default="hip-strix-dev")
    bt.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4)))
    bt.add_argument("--ctest-regex", default=None)
    bt.add_argument("--model-dir", type=Path, default=None)
    bt.add_argument("--extra-cmake-args", default="")
    bt.add_argument("--out", type=Path, default=None)

    bn = sub.add_parser("bench", help="Run decode, prefill, warm agent turn, and cold 34K benchmarks.")
    bn.add_argument("--base-url", default=None)
    bn.add_argument("--serve-bin", type=Path, default=None)
    bn.add_argument("--model-dir", type=Path, default=None)
    bn.add_argument("--model-id", default=None)
    bn.add_argument("--fake-smoke", action="store_true", help="Launch ingine-serve --fake for harness smoke testing only.")
    bn.add_argument("--extra-serve-args", default="")
    bn.add_argument("--decode-max-tokens", type=int, default=256)
    bn.add_argument("--prefill-lengths", default="8192,32768,131072")
    bn.add_argument("--agent-lengths", default="32768,65536,96000")
    bn.add_argument("--cold-tokens", type=int, default=34816)
    bn.add_argument("--startup-timeout-s", type=float, default=180.0)
    bn.add_argument("--expected-model-sha256", default=None)
    bn.add_argument("--selected-device-arch", default=None)
    bn.add_argument("--selected-backend", default=None)
    bn.add_argument("--out", type=Path, default=None)

    sc = sub.add_parser("score", help="Score a saved verification report JSON against the Strix Halo floor.")
    sc.add_argument("report_json", type=Path)
    sc.add_argument("--expected-model-sha256", default=None)
    sc.add_argument("--markdown-out", type=Path, default=None)
    sc.add_argument("--allow-unmeasured", action="store_true")

    al = sub.add_parser("all", help="Run preflight -> build-test -> bench -> score end to end.")
    al.add_argument("--repo-root", type=Path, default=repo_root_default())
    al.add_argument("--preset", default="hip-strix-dev")
    al.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4)))
    al.add_argument("--ctest-regex", default=None)
    al.add_argument("--extra-cmake-args", default="")
    al.add_argument("--model-dir", type=Path, required=True)
    al.add_argument("--model-id", default=None)
    al.add_argument("--expected-model-sha256", default=None)
    al.add_argument("--selected-device-arch", default=None)
    al.add_argument("--selected-backend", default=None)
    al.add_argument("--require-gfx1151", action="store_true")
    al.add_argument("--hash-packs", action="store_true")
    al.add_argument("--fake-smoke", action="store_true")
    al.add_argument("--extra-serve-args", default="")
    al.add_argument("--decode-max-tokens", type=int, default=256)
    al.add_argument("--prefill-lengths", default="8192,32768,131072")
    al.add_argument("--agent-lengths", default="32768,65536,96000")
    al.add_argument("--cold-tokens", type=int, default=34816)
    al.add_argument("--startup-timeout-s", type=float, default=180.0)
    al.add_argument("--out-dir", type=Path, required=True)
    al.add_argument("--allow-unmeasured", action="store_true")
    return p


def write_json(path: Path | None, obj: Any) -> None:
    text = json.dumps(obj, indent=2, sort_keys=False) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "preflight":
        res = run_preflight(args.repo_root, args.model_dir, args.require_gfx1151, args.hash_packs)
        write_json(args.out, res)
        return 0 if res["ok"] else 2

    if args.command == "build-test":
        res = run_build_and_test(
            args.repo_root,
            args.preset,
            args.jobs,
            args.ctest_regex,
            args.model_dir,
            shlex.split(args.extra_cmake_args),
        )
        write_json(args.out, res)
        return 0 if res["ok"] else 2

    if args.command == "bench":
        res = run_benchmark_suite(
            base_url=args.base_url,
            serve_bin=args.serve_bin,
            model_dir=args.model_dir,
            model_id_override=args.model_id,
            fake_smoke=args.fake_smoke,
            extra_serve_args=shlex.split(args.extra_serve_args),
            decode_max_tokens=args.decode_max_tokens,
            prefill_lengths=parse_csv_ints(args.prefill_lengths),
            agent_lengths=parse_csv_ints(args.agent_lengths),
            cold_tokens=args.cold_tokens,
            startup_timeout_s=args.startup_timeout_s,
            expected_model_sha256=args.expected_model_sha256,
            selected_device_arch=args.selected_device_arch,
            selected_backend=args.selected_backend,
        )
        write_json(args.out, res)
        return 0

    if args.command == "score":
        report = json.loads(args.report_json.read_text(encoding="utf-8"))
        score = score_report(report, expected_model_sha256=args.expected_model_sha256)
        md = render_scorecard_markdown(report, score)
        if args.markdown_out is not None:
            args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_out.write_text(md, encoding="utf-8")
        else:
            sys.stdout.write(md)
        if score["hardware_verified"]:
            return 0
        if args.allow_unmeasured and score["verdict"] in (
            "UNMEASURED_FAKE_MODE",
            "INCOMPLETE_UNMEASURED",
            "PASS_FLOOR_NON_GFX1151",
            "PASS_FLOOR_UNBOUND_IDENTITY",
        ):
            return 0
        return 2

    if args.command == "all":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        preflight = run_preflight(args.repo_root, args.model_dir, args.require_gfx1151, args.hash_packs)
        report: dict[str, Any] = {"preflight": preflight}
        if not preflight["ok"]:
            write_json(args.out_dir / "strix_halo_report.json", report)
            return 2

        bt = run_build_and_test(
            args.repo_root,
            args.preset,
            args.jobs,
            args.ctest_regex,
            args.model_dir,
            shlex.split(args.extra_cmake_args),
        )
        report["build_test"] = bt
        if not bt["ok"] or not bt.get("serve_bin"):
            write_json(args.out_dir / "strix_halo_report.json", report)
            return 2

        bench = run_benchmark_suite(
            base_url=None,
            serve_bin=Path(bt["serve_bin"]),
            model_dir=args.model_dir,
            model_id_override=args.model_id,
            fake_smoke=args.fake_smoke,
            extra_serve_args=shlex.split(args.extra_serve_args),
            decode_max_tokens=args.decode_max_tokens,
            prefill_lengths=parse_csv_ints(args.prefill_lengths),
            agent_lengths=parse_csv_ints(args.agent_lengths),
            cold_tokens=args.cold_tokens,
            startup_timeout_s=args.startup_timeout_s,
            expected_model_sha256=args.expected_model_sha256,
            selected_device_arch=args.selected_device_arch,
            selected_backend=args.selected_backend,
        )
        report["bench"] = bench
        score = score_report(report, expected_model_sha256=args.expected_model_sha256)
        report["score"] = score

        write_json(args.out_dir / "strix_halo_report.json", report)
        md = render_scorecard_markdown(report, score)
        (args.out_dir / "strix_halo_scorecard.md").write_text(md, encoding="utf-8")

        if score["hardware_verified"]:
            return 0
        if args.allow_unmeasured and score["verdict"] in (
            "UNMEASURED_FAKE_MODE",
            "INCOMPLETE_UNMEASURED",
            "PASS_FLOOR_NON_GFX1151",
            "PASS_FLOOR_UNBOUND_IDENTITY",
        ):
            return 0
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

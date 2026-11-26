#!/usr/bin/env python3
"""Static HIP portability audit for atlas-ingine native/ (#835).

Scans every source/header/CMake file compiled when INGINE_BACKEND=HIP to verify:
  1. Zero unguarded raw CUDA-only headers (<cuda_runtime.h>, <cuda_fp8.h>, <cuda_fp16.h>,
     <cuda_bf16.h>, <cublasLt.h>, <cooperative_groups.h>, <nccl.h>, <nvtx3/...>).
  2. Zero unguarded NVIDIA-only device intrinsics / PTX / CUDA-12.8 conditional-graph APIs
     (__nv_cvt_*, __nv_fp8*, __cvta_generic_to_shared, cudaLaunchKernelEx,
     cudaOccupancyMaxActiveClusters, cudaGraphConditionalHandleCreate, cudaGraphSetConditional,
     inline `asm` / `asm volatile` PTX) outside CUDA-only preprocessor guards.
  3. Zero hardcoded `CUDA::cudart` / `CUDA::cublasLt` links in shared CMakeLists paths
     that execute when INGINE_BACKEND=HIP.
  4. Zero definitions of `__CUDA_ARCH__` under HIP headers/build scripts.

Uses Python 3 standard library only; exits 0 iff 0 violations.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CUDA_ONLY_FILES: set[str] = {
    # Compiled only when INGINE_BACKEND=CUDA (excluded from HIP CMakeLists).
    "native/src/core/arena.cu",
    "native/src/core/device.cu",
    "native/src/core/hal_cuda.cpp",
    "native/src/core/host_memory.cpp",
    "native/src/core/safetensors_upload_cuda.cpp",
    "native/src/core/ops_cuda.cpp",
    "native/src/core/cuda_compat.h",
    "native/src/core/nvtx.h",
    "native/src/core/decode_graph.h",
    "native/src/core/gpu_compat.h",
    "native/src/core/hal.h",
    "native/src/core/device.h",
    "native/src/core/act.h",
    "native/src/core/pdl.cuh",
    "native/src/kernels/common/arch_ops.cuh",
    "native/src/kernels/common/warp.cuh",
}

NON_HIP_BACKEND_FILES: set[str] = {
    "native/src/core/hal_cpu.cpp",
    "native/src/core/hal_mlx.cpp",
    "native/src/core/hal_sycl.cpp",
    "native/src/core/ops_cpu.cpp",
    "native/src/core/ops_mlx.cpp",
    "native/src/core/ops_sycl.cpp",
    "native/src/core/safetensors_upload_cpu.cpp",
    "native/src/core/safetensors_upload_mlx.cpp",
    "native/src/core/safetensors_upload_sycl.cpp",
}

RAW_CUDA_HEADER_RE = re.compile(
    r"^\s*#\s*include\s*<(cuda_runtime(?:_api)?\.h|cuda\.h|cuda_fp8\.h|cuda_fp16\.h|"
    r"cuda_bf16\.h|cublasLt\.h|cublas_v2\.h|cooperative_groups\.h|nccl\.h|nvtx3/[^>]+)>",
)

CUDA_ONLY_SYMBOL_RES: list[tuple[str, re.Pattern[str]]] = [
    ("nv_fp8_intrinsic", re.compile(r"\b__nv_(?:cvt_float|cvt_fp8|fp8)\w*")),
    ("cvta_generic_to_shared", re.compile(r"\b__cvta_generic_to_shared\b")),
    ("cudaLaunchKernelEx", re.compile(r"\bcudaLaunchKernelEx\b")),
    ("cudaOccupancyMaxActiveClusters", re.compile(r"\bcudaOccupancyMaxActiveClusters\b")),
    ("cudaGraphConditionalHandleCreate", re.compile(r"\bcudaGraphConditionalHandleCreate\b")),
    ("cudaGraphSetConditional", re.compile(r"\bcudaGraphSetConditional\b")),
    ("inline_ptx_asm", re.compile(r"\basm\s+(?:volatile\s*)?\(")),
]

CUDA_ONLY_GUARD_RE = re.compile(
    r"^\s*#\s*(?:if|elif)\b.*("
    r"INGINE_BACKEND_CUDA"
    r"|__CUDA_ARCH__"
    r"|INGINE_TP_SIZE\s*>\s*1"
    r"|CUDART_VERSION\s*>=\s*12080"
    r")",
)

NEGATED_CUDA_GUARD_RE = re.compile(
    r"!\s*(?:defined\s*\(\s*INGINE_BACKEND_CUDA\s*\)|INGINE_BACKEND_CUDA)"
    r"|INGINE_BACKEND_CUDA\s*==\s*0"
    r"|defined\s*\(\s*INGINE_BACKEND_HIP\s*\)"
    r"|INGINE_BACKEND_HIP",
)

DEFINE_CUDA_ARCH_RE = re.compile(r"^\s*#\s*define\s+__CUDA_ARCH__\b")


def _strip_comments_line(line: str, in_block_comment: bool) -> tuple[str, bool]:
    out: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        if in_block_comment:
            end = line.find("*/", i)
            if end == -1:
                return "".join(out), True
            i = end + 2
            in_block_comment = False
        else:
            if line[i : i + 2] == "//":
                break
            if line[i : i + 2] == "/*":
                in_block_comment = True
                i += 2
            else:
                out.append(line[i])
                i += 1
    return "".join(out), in_block_comment


def _is_cuda_only_guard(cond_line: str) -> bool:
    if not CUDA_ONLY_GUARD_RE.search(cond_line):
        return False
    # Reject negated CUDA guards like `#if !INGINE_BACKEND_CUDA` or `#if INGINE_BACKEND_CUDA || INGINE_BACKEND_HIP`.
    if "||" in cond_line and ("INGINE_BACKEND_HIP" in cond_line or "__HIP" in cond_line):
        return False
    if re.search(r"!\s*\(?\s*(?:defined\s*\(\s*INGINE_BACKEND_CUDA\s*\)|INGINE_BACKEND_CUDA)", cond_line):
        return False
    if "INGINE_BACKEND_CUDA == 0" in cond_line or "INGINE_BACKEND_CUDA==0" in cond_line:
        return False
    if "INGINE_BACKEND_HIP" in cond_line and "!" not in cond_line:
        return False
    return True


def audit_cpp_file(path: Path, rel: str) -> list[str]:
    violations: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # Check raw __CUDA_ARCH__ definition everywhere (even in compat headers).
    in_bc = False
    for lineno, raw in enumerate(lines, 1):
        code, in_bc = _strip_comments_line(raw, in_bc)
        if DEFINE_CUDA_ARCH_RE.search(code):
            violations.append(f"{rel}:{lineno}: forbidden #define __CUDA_ARCH__")

    if rel in CUDA_ONLY_FILES or rel in NON_HIP_BACKEND_FILES:
        return violations

    # Track nested #if stack: each frame is True iff this active branch is CUDA-only (skipped on HIP).
    guard_stack: list[bool] = []
    in_bc = False
    for lineno, raw in enumerate(lines, 1):
        code, in_bc = _strip_comments_line(raw, in_bc)
        stripped = code.strip()

        if re.match(r"^#\s*if(?:def|ndef)?\b", stripped):
            is_cuda_branch = _is_cuda_only_guard(stripped)
            guard_stack.append(is_cuda_branch)
            continue
        if re.match(r"^#\s*elif\b", stripped):
            if guard_stack:
                guard_stack[-1] = _is_cuda_only_guard(stripped)
            continue
        if re.match(r"^#\s*else\b", stripped):
            if guard_stack:
                guard_stack[-1] = False
            continue
        if re.match(r"^#\s*endif\b", stripped):
            if guard_stack:
                guard_stack.pop()
            continue

        if any(guard_stack):
            continue

        if RAW_CUDA_HEADER_RE.search(code):
            violations.append(f"{rel}:{lineno}: unguarded CUDA-only header include: {stripped}")
        for label, pat in CUDA_ONLY_SYMBOL_RES:
            if pat.search(code):
                violations.append(f"{rel}:{lineno}: unguarded {label} in HIP-compiled path: {stripped}")

    return violations


def audit_cmake_files(repo_root: Path) -> list[str]:
    violations: list[str] = []
    cmake_dirs = [
        repo_root / "native" / "src" / "kernels",
        repo_root / "native" / "src" / "model",
        repo_root / "native" / "apps",
    ]
    for base in cmake_dirs:
        if not base.exists():
            continue
        for cm in sorted(base.rglob("CMakeLists.txt")):
            rel = cm.relative_to(repo_root).as_posix()
            in_cuda_only_if = 0
            for lineno, raw in enumerate(cm.read_text(encoding="utf-8").splitlines(), 1):
                line = raw.split("#", 1)[0].strip()
                if re.match(r"^if\s*\(\s*INGINE_BACKEND\s+STREQUAL\s+\"CUDA\"\s*\)", line):
                    in_cuda_only_if += 1
                    continue
                if in_cuda_only_if > 0 and re.match(r"^(?:else|elseif|endif)\s*\(", line):
                    in_cuda_only_if -= 1
                    continue
                if in_cuda_only_if == 0 and ("CUDA::cudart" in line or "CUDA::cublasLt" in line):
                    violations.append(f"{rel}:{lineno}: hardcoded CUDA target in HIP-shared CMakeLists: {line}")
    return violations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Static HIP portability audit for atlas-ingine.")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--include-tests", action="store_true", help="Also scan native/tests/ for unguarded CUDA headers.")
    args = ap.parse_args(argv)
    repo_root = args.repo_root.resolve()

    scan_roots = [
        repo_root / "native" / "src" / "core",
        repo_root / "native" / "src" / "kernels",
        repo_root / "native" / "src" / "model",
        repo_root / "native" / "apps",
    ]
    if args.include_tests:
        scan_roots.append(repo_root / "native" / "tests")

    exts = {".h", ".hpp", ".cuh", ".cpp", ".cu", ".hip"}
    checked_files = 0
    violations: list[str] = []

    for root in scan_roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix not in exts:
                continue
            rel = p.relative_to(repo_root).as_posix()
            checked_files += 1
            violations.extend(audit_cpp_file(p, rel))

    violations.extend(audit_cmake_files(repo_root))

    if violations:
        print(f"FAIL audit_hip_portability: {len(violations)} violation(s) across {checked_files} files:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print(f"PASS audit_hip_portability: 0 violations across {checked_files} C++/CUDA/HIP files + shared CMakeLists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

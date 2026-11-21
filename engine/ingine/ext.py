"""JIT build of ingine's CUDA extensions (sm_120a, RTX 5090).

Each extension lives in ``engine/csrc/<name>/`` (``*.cu`` and ``*.cpp``, one ``PYBIND11_MODULE``) and is
built on first import into ``engine/build/<name>``. Kernels launch on the current PyTorch stream and take
preallocated outputs, so calls are CUDA-graph capturable.
"""
from __future__ import annotations

import os
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSRC = ROOT / "csrc"
BUILD = ROOT / "build"

os.environ.setdefault("CUDA_HOME", "/opt/cuda")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")


@cache
def load(name: str):
    from torch.utils.cpp_extension import load as _load
    src = CSRC / name
    sources = sorted(str(p) for p in src.iterdir() if p.suffix in (".cu", ".cpp"))
    out = BUILD / name
    out.mkdir(parents=True, exist_ok=True)
    return _load(
        name=f"ingine_{name}",
        sources=sources,
        extra_include_paths=[str(CSRC / "common")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=["-O3", "-std=c++20", "-gencode=arch=compute_120a,code=sm_120a", "-lineinfo",
                           "--expt-relaxed-constexpr"] + (["-Xptxas=-v"] if os.environ.get("INGINE_PTXAS_V") else []),
        build_directory=str(out),
        verbose=bool(os.environ.get("INGINE_BUILD_VERBOSE")),
    )

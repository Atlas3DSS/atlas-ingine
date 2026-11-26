#!/usr/bin/env python3
"""Hardware-aware self-setup, GPU-family auto-tuner, and localhost launcher for atlas-ingine.

Zero external Python dependencies (stdlib only). Provides:
  1. Hardware & checkpoint auto-detection (NVIDIA sm_60..sm_120a, VRAM, driver, RAM, disk,
     iommu=pt P2P status, local binaries & checkpoints, plus multi-vendor detection hooks).
  2. Deterministic GPU-family auto-tuner emitting verified ingine-serve flags and CMake presets.
  3. Atlas-branded (obsidian/emerald) localhost Web UI with clickable-circle Model Tier &
     Array Count selectors, Multimodal Chat UI, OpenAI API snippets, and explicit permission
     gating before downloading weights or launching ingine-serve.
  4. Full CLI fallback (--cli, --plan, --json, --dry-run, --yes).
"""
from __future__ import annotations

import argparse
import atexit
import dataclasses
import html
import hashlib
import http.client
import http.server
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "runtime-manifest.json"
DEFAULT_UI_HOST = "127.0.0.1"
DEFAULT_UI_PORT = 8088
DEFAULT_SERVE_HOST = "127.0.0.1"
DEFAULT_SERVE_PORT = 8003
MIN_CUDA13_DRIVER = 580

# Exact HF checkpoints documented in README.md and runtime-manifest.json.
MODEL_TIERS: dict[str, dict[str, Any]] = {
    "small": {
        "id": "small",
        "circle": "9G",
        "label": "Small",
        "subtitle": "EXL3 9 GB (FP8)",
        "repo_id": "Atlas3D/Inkling-2-Mini-atlas-exl3-9g-fp8-ingine",
        "local_dir_name": "Inkling-2-Mini-exl3-9g",
        "model_id": "inkling-2-mini-exl3-9g",
        "disk_gb": 9.0,
        "base_vram_gb": 9.6,
        "min_vram_gb": 5.5,
        "supports_atlas_6g": True,
        "supports_vision": True,
        "supports_spec": True,
        "dense_prebaked_fp8": True,
        "ready": True,
        "family": "inkling",
        "notes": "Fits 6 GB-24 GB GPUs including 1M context (--kv-q4).",
    },
    "small_10g": {
        "id": "small_10g",
        "circle": "10G",
        "label": "Compact",
        "subtitle": "Elastic atlas-10g",
        "repo_id": "Atlas3D/Inkling-2-Mini-atlas-10g-ingine",
        "local_dir_name": "Inkling-2-Mini-atlas-10g",
        "model_id": "inkling-2-mini-atlas-10g",
        "disk_gb": 12.01,
        "base_vram_gb": 12.6,
        "min_vram_gb": 12.0,
        "supports_atlas_6g": False,
        "supports_vision": True,
        "supports_spec": True,
        "dense_prebaked_fp8": False,
        "ready": True,
        "family": "inkling",
        "notes": "Native ATLAS2/ATLAS3 MoE for 12 GB-16 GB cards (P100/T4/3090).",
    },
    "medium": {
        "id": "medium",
        "circle": "12G",
        "label": "Medium",
        "subtitle": "Elastic atlas-12g",
        "repo_id": "Atlas3D/Inkling-2-Mini-atlas-12g-ingine",
        "local_dir_name": "Inkling-2-Mini-atlas-12g",
        "model_id": "inkling-2-mini-atlas-12g",
        "disk_gb": 14.05,
        "base_vram_gb": 14.7,
        "min_vram_gb": 15.5,
        "supports_atlas_6g": False,
        "supports_vision": True,
        "supports_spec": True,
        "dense_prebaked_fp8": False,
        "ready": True,
        "family": "inkling",
        "notes": "Higher-bit Elastic MoE for 16 GB-24 GB cards (64K-128K ctx).",
    },
    "large": {
        "id": "large",
        "circle": "20G",
        "label": "Large",
        "subtitle": "Native NVFP4+MXFP8",
        "repo_id": "thinkingmachines/Inkling-2-Mini",
        "local_dir_name": "Inkling-2-Mini",
        "model_id": "inkling-2-mini",
        "disk_gb": 20.2,
        "base_vram_gb": 20.8,
        "min_vram_gb": 22.0,
        "supports_atlas_6g": False,
        "supports_vision": True,
        "supports_spec": True,
        "dense_prebaked_fp8": False,
        "ready": True,
        "family": "inkling",
        "notes": "Full base weights for 24 GB-96 GB cards (fastest on 3090/4090/V100/H100/PRO 6000).",
    },
}

UPCOMING_FAMILIES: list[dict[str, str]] = [
    {"id": "qwen38", "label": "Qwen3.8-27B-NVFP4", "status": "branch-staged (#821)"},
    {"id": "deepseek_v41", "label": "DeepSeek-V4.1-Flash", "status": "branch-staged (#833)"},
    {"id": "glm53", "label": "GLM-5.3-Flash", "status": "branch-staged (#834)"},
]

CONTEXT_OPTIONS = (16384, 65536, 262144, 1048576)

# KV bytes/token for Inkling-2-Mini full-attention layers (per_pass, 8 visit slots):
#   FP8 E4M3 ~= 16 KiB/tok (~16.0 GiB @ 1M); Q4_0 ~= 9 KiB/tok (~9.0 GiB @ 1M);
#   last_full cuts full-attention KV in half (~4.5 GiB @ 1M in Q4_0).
KV_BYTES_PER_TOK = {
    ("fp8", "per_pass"): 16384,
    ("fp8", "last_full"): 8192,
    ("q4", "per_pass"): 9216,
    ("q4", "last_full"): 4608,
}


def sm_to_int(compute_cap: str) -> int:
    m = re.match(r"^\s*(\d+)\.(\d+)", str(compute_cap or ""))
    if not m:
        return 0
    return int(m.group(1)) * 10 + int(m.group(2))


def sm_arch_tag(sm_int: int) -> str:
    if sm_int >= 120:
        return "sm_120a"
    if sm_int <= 0:
        return "unknown"
    return f"sm_{sm_int}"


def sm_generation_name(sm_int: int) -> str:
    if sm_int >= 120:
        return "Blackwell"
    if sm_int >= 90:
        return "Hopper"
    if sm_int >= 89:
        return "Ada Lovelace"
    if sm_int >= 80:
        return "Ampere"
    if sm_int >= 75:
        return "Turing"
    if sm_int >= 70:
        return "Volta"
    if sm_int >= 60:
        return "Pascal"
    return "Unknown"


def _run_cmd(cmd: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, "", str(exc)


def parse_nvidia_smi_csv(csv_text: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for raw in csv_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
            name = parts[1]
            total_mib = float(parts[2])
            free_mib = float(parts[3])
            cc = parts[4]
            driver = parts[5]
            bus_id = parts[6] if len(parts) > 6 else ""
        except ValueError:
            continue
        sm_int = sm_to_int(cc)
        drv_major = 0
        dm = re.match(r"^(\d+)", driver)
        if dm:
            drv_major = int(dm.group(1))
        gpus.append(
            {
                "index": idx,
                "vendor": "nvidia",
                "name": name,
                "vram_total_mib": int(round(total_mib)),
                "vram_free_mib": int(round(free_mib)),
                "vram_total_gb": round(total_mib / 1024.0, 2),
                "vram_free_gb": round(free_mib / 1024.0, 2),
                "compute_cap": cc,
                "sm_int": sm_int,
                "arch": sm_arch_tag(sm_int),
                "generation": sm_generation_name(sm_int),
                "driver": driver,
                "driver_major": drv_major,
                "bus_id": bus_id,
                "act_fp16_required": 0 < sm_int < 80,
                "supported_native_cuda": sm_int >= 60,
            }
        )
    return gpus


def detect_iommu_pt(cmdline_text: str | None = None) -> dict[str, Any]:
    if cmdline_text is None:
        p = Path("/proc/cmdline")
        cmdline_text = p.read_text(encoding="utf-8", errors="ignore") if p.is_file() else ""
    tokens = cmdline_text.split()
    iommu_pt = "iommu=pt" in tokens
    iommu_off = any(t in ("intel_iommu=off", "amd_iommu=off", "iommu=off") for t in tokens)
    return {
        "iommu_pt": iommu_pt,
        "iommu_off": iommu_off,
        "p2p_friendly": iommu_pt or iommu_off,
    }


def detect_host_ram_gb(meminfo_text: str | None = None) -> tuple[float, float]:
    if meminfo_text is None:
        p = Path("/proc/meminfo")
        meminfo_text = p.read_text(encoding="utf-8", errors="ignore") if p.is_file() else ""
    total_kb = 0
    avail_kb = 0
    for line in meminfo_text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                total_kb = int(parts[1])
        elif line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                avail_kb = int(parts[1])
    if total_kb == 0:
        return 0.0, 0.0
    return round(total_kb / (1024.0 * 1024.0), 2), round(avail_kb / (1024.0 * 1024.0), 2)


FAKE_REQUIRED_FILES = ("config.json", "tokenizer.json", "chat_template.jinja")
FAKE_DOWNLOAD_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def _effective_vram_gb(gpu: dict[str, Any]) -> float:
    total_gb = float(gpu.get("vram_total_gb") or 0.0)
    free = gpu.get("vram_free_gb")
    return total_gb if free is None else min(total_gb, max(0.0, float(free)))


def _shard_complete(path: Path, tensors: tuple[str, ...] | list[str] = ()) -> bool:
    try:
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            header_len = int.from_bytes(stream.read(8), "little")
            if not 0 < header_len <= min(size - 8, 100_000_000):
                return False
            header = json.loads(stream.read(header_len))
        if not isinstance(header, dict) or any(name not in header for name in tensors):
            return False
        ends = [entry["data_offsets"][1] for name, entry in header.items() if name != "__metadata__"]
        return bool(ends) and max(ends) == size - 8 - header_len
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        return False


def inspect_checkpoint_dir(path: Path) -> dict[str, Any] | None:
    if not path.is_dir():
        return None
    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        return None
    missing_files: list[str] = []
    try:
        cfg_obj = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(cfg_obj, dict):
            missing_files.append("config.json (invalid JSON object)")
    except Exception:
        missing_files.append("config.json (unreadable JSON)")

    tok_path = path / "tokenizer.json"
    tmpl_path = path / "chat_template.jinja"
    has_tokenizer = tok_path.is_file() and tok_path.stat().st_size > 0
    has_chat_template = tmpl_path.is_file() and tmpl_path.stat().st_size > 0
    if not has_tokenizer:
        missing_files.append("tokenizer.json")
    if not has_chat_template:
        missing_files.append("chat_template.jinja")

    safetensors = sorted(path.glob("*.safetensors"))
    index_path = path / "model.safetensors.index.json"
    weight_errors = []
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("empty weight_map")
            shards: dict[str, list[str]] = {}
            for tensor, shard in weight_map.items():
                shards.setdefault(shard, []).append(tensor)
            for shard, tensors in shards.items():
                if not _shard_complete(path / shard, tensors):
                    weight_errors.append(shard)
        except (OSError, ValueError, TypeError, KeyError):
            weight_errors.append("model.safetensors.index.json (invalid weight map)")
    elif not _shard_complete(path / "model.safetensors"):
        weight_errors.append("model.safetensors or model.safetensors.index.json")
    complete_weights = not weight_errors
    missing_files.extend(weight_errors)

    complete_fake_assets = has_tokenizer and has_chat_template and not any(
        m.startswith("config.json") for m in missing_files
    )

    if not safetensors and not complete_fake_assets:
        return None

    total_bytes = sum(f.stat().st_size for f in safetensors if f.is_file())
    size_gb = round(total_bytes / 1e9, 2)
    has_mtp = (path / "mtp.safetensors").is_file() and (path / "mtp.safetensors").stat().st_size > 0
    has_ablation = (
        (path / "inkling-2-mini-abliterated.safetensors").is_file()
        and (path / "inkling-2-mini-abliterated.safetensors").stat().st_size > 0
    )

    name_l = path.name.lower()
    inferred_tier = "large"
    if "exl3" in name_l or "9g" in name_l or (complete_weights and size_gb <= 10.8):
        inferred_tier = "small"
    elif "10g" in name_l or (complete_weights and size_gb <= 13.0):
        inferred_tier = "small_10g"
    elif "12g" in name_l or (complete_weights and size_gb <= 16.5):
        inferred_tier = "medium"

    return {
        "path": str(path),
        "name": path.name,
        "size_gb": size_gb,
        "has_mtp": has_mtp,
        "has_ablation": has_ablation,
        "has_tokenizer": has_tokenizer,
        "has_chat_template": has_chat_template,
        "complete_weights": complete_weights,
        "complete_fake_assets": complete_fake_assets,
        "usable_engine": complete_weights and complete_fake_assets,
        "missing_files": missing_files,
        "inferred_tier": inferred_tier,
        "safetensor_count": len(safetensors),
    }


def verify_checkpoint_dir(path: Path, *, require_weights: bool = True) -> tuple[bool, list[str]]:
    info = inspect_checkpoint_dir(path)
    if info is None:
        return False, list(FAKE_REQUIRED_FILES) + (["model.safetensors or model.safetensors.index.json"] if require_weights else [])
    if require_weights:
        ok = bool(info["usable_engine"])
    else:
        ok = bool(info["complete_fake_assets"])
    return ok, list(info["missing_files"])


def scan_local_checkpoints(root: Path = ROOT, extra_dirs: list[Path] | None = None) -> list[dict[str, Any]]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_dir(p: Path) -> None:
        try:
            rp = str(p.expanduser().resolve())
        except OSError:
            return
        if rp not in seen:
            seen.add(rp)
            candidates.append(Path(rp))

    env_dir = os.environ.get("INGINE_MODEL_DIR", "").strip()
    if env_dir:
        add_dir(Path(env_dir))

    search_roots = [root / "models", Path.home() / "models", Path("/mnt/nvme/models"), Path("/mnt/storage/models")]
    if extra_dirs:
        search_roots.extend(extra_dirs)

    for parent in search_roots:
        if not parent.is_dir():
            continue
        if (parent / "config.json").is_file():
            add_dir(parent)
        try:
            for child in sorted(parent.iterdir()):
                if child.is_dir() and not child.name.startswith("."):
                    if "inkling" in child.name.lower() or "atlas" in child.name.lower() or (child / "hf_quant_config.json").is_file():
                        add_dir(child)
        except OSError:
            continue

    found: list[dict[str, Any]] = []
    for cand in candidates:
        info = inspect_checkpoint_dir(cand)
        if info is not None:
            found.append(info)
    return found


def find_local_binaries(root: Path = ROOT) -> dict[str, str]:
    probes = {
        "install": root / "native" / "build-install" / "apps" / "serve" / "ingine-serve",
        "multiarch": root / "native" / "build-multiarch" / "apps" / "serve" / "ingine-serve",
        "release": root / "native" / "build" / "apps" / "serve" / "ingine-serve",
        "sm70": root / "native" / "build-sm70" / "apps" / "serve" / "ingine-serve",
        "sm80": root / "native" / "build-sm80" / "apps" / "serve" / "ingine-serve",
        "sm90": root / "native" / "build-sm90" / "apps" / "serve" / "ingine-serve",
        "legacy_fp16": root / "native" / "build-legacy-fp16" / "apps" / "serve" / "ingine-serve",
        "cpu": root / "native" / "build-cpu" / "apps" / "serve" / "ingine-serve",
    }
    found: dict[str, str] = {}
    for key, path in probes.items():
        if path.is_file() and os.access(path, os.X_OK):
            found[key] = str(path)
    return found


def detect_other_vendors() -> list[dict[str, Any]]:
    vendors: list[dict[str, Any]] = []
    if shutil.which("rocm-smi"):
        rc, out, _ = _run_cmd(["rocm-smi", "--showproductname"])
        if rc == 0 and out.strip():
            vendors.append({"vendor": "amd_rocm", "detail": out.splitlines()[0].strip(), "status": "staged (#835)"})
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        vendors.append({"vendor": "apple_metal", "detail": "Apple Silicon (Metal/MLX)", "status": "staged (#835)"})
    if shutil.which("sycl-ls"):
        rc, out, _ = _run_cmd(["sycl-ls"])
        if rc == 0 and "level_zero" in out.lower():
            vendors.append({"vendor": "intel_sycl", "detail": "Intel Level-Zero SYCL GPU", "status": "staged (#835)"})
    return vendors


def detect_hardware(root: Path = ROOT) -> dict[str, Any]:
    gpus: list[dict[str, Any]] = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        rc, out, _ = _run_cmd(
            [
                nvidia_smi,
                "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version,pci.bus_id",
                "--format=csv,noheader,nounits",
            ]
        )
        if rc == 0:
            gpus = parse_nvidia_smi_csv(out)

    other_vendors = detect_other_vendors()
    ram_total_gb, ram_free_gb = detect_host_ram_gb()
    try:
        disk = shutil.disk_usage(root)
        disk_free_gb = round(disk.free / 1e9, 2)
        disk_total_gb = round(disk.total / 1e9, 2)
    except OSError:
        disk_free_gb, disk_total_gb = 0.0, 0.0

    binaries = find_local_binaries(root)
    checkpoints = scan_local_checkpoints(root)
    iommu = detect_iommu_pt()

    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "cpu_threads": os.cpu_count() or 1,
        "ram_total_gb": ram_total_gb,
        "ram_free_gb": ram_free_gb,
        "disk_free_gb": disk_free_gb,
        "disk_total_gb": disk_total_gb,
        "gpus": gpus,
        "other_vendors": other_vendors,
        "iommu": iommu,
        "binaries": binaries,
        "checkpoints": checkpoints,
        "has_private_cuda": (root / ".cuda-toolkit" / "nvidia" / "cu13" / "bin" / "nvcc").is_file(),
        "has_hf_cli": shutil.which("hf") is not None or shutil.which("huggingface-cli") is not None,
    }


def _select_best_gpu(gpus: list[dict[str, Any]], gpu_indices: list[int] | None = None) -> list[dict[str, Any]]:
    if not gpus:
        return []
    if gpu_indices:
        by_idx = {g["index"]: g for g in gpus}
        picked = [by_idx[i] for i in gpu_indices if i in by_idx]
        if picked:
            return picked
    # Prefer highest detected free VRAM, then total VRAM, then newest SM architecture.
    return sorted(
        gpus,
        key=lambda g: (_effective_vram_gb(g), g["vram_total_mib"], g["sm_int"], -g["index"]),
        reverse=True,
    )


def recommend_defaults(hw: dict[str, Any]) -> dict[str, Any]:
    gpus = _select_best_gpu(hw.get("gpus", []))
    if not gpus:
        return {
            "tier": "small",
            "ctx": 16384,
            "array_count": 1,
            "layout": "single",
            "mode": "chat",
        }
    primary = gpus[0]
    vram_gb = _effective_vram_gb(primary)
    if vram_gb >= 22.0:
        tier = "large"
    elif vram_gb >= 15.5:
        tier = "small_10g" if primary["sm_int"] < 80 else "medium"
    elif vram_gb >= 12.0:
        tier = "small_10g"
    else:
        tier = "small"

    if vram_gb >= 72.0:
        ctx = 262144
    elif vram_gb >= 22.0:
        ctx = 65536
    elif vram_gb >= 11.5:
        ctx = 65536 if tier == "small" else 16384
    else:
        ctx = 65536

    return {
        "tier": tier,
        "ctx": ctx,
        "array_count": 1,
        "layout": "single",
        "mode": "chat",
    }


def _gpu_dp_slots(gpu: dict[str, Any], tier: dict[str, Any], ctx: int) -> int:
    eff = _effective_vram_gb(gpu)
    if eff < 5.2:
        return 0
    if eff < tier["min_vram_gb"]:
        return 1 if (tier["supports_atlas_6g"] and eff >= 5.2) else 0
    # Estimate minimum VRAM per independent worker (weights + smallest viable KV + runtime headroom).
    kv_gb = (ctx * KV_BYTES_PER_TOK[("q4", "per_pass")]) / (1024.0 ** 3)
    if eff < 22.0 and ctx >= 262144:
        kv_gb = 2.0  # paged Q4 VRAM cache cap
    elif eff < 31.0 and ctx >= 524288 and tier["id"] == "large":
        kv_gb = 1.0
    per_worker_gb = max(tier["min_vram_gb"], tier["base_vram_gb"] + min(kv_gb, 16.0) + 1.0)
    if eff < per_worker_gb:
        return 1
    return max(1, min(4, int((eff - 0.5) // per_worker_gb)))


def _choose_tp_group(gpus: list[dict[str, Any]], tier: dict[str, Any], tp_size: int) -> list[dict[str, Any]]:
    if tp_size <= 1:
        return []
    min_per_gpu = (tier["base_vram_gb"] / tp_size) + 2.0
    sm_groups: dict[int, list[dict[str, Any]]] = {}
    for g in gpus:
        if _effective_vram_gb(g) >= min_per_gpu:
            sm_groups.setdefault(g["sm_int"], []).append(g)
    candidates: list[list[dict[str, Any]]] = []
    for grp in sm_groups.values():
        if len(grp) >= tp_size:
            ordered = sorted(grp, key=lambda g: (_effective_vram_gb(g), g["vram_total_mib"], -g["index"]), reverse=True)
            candidates.append(ordered[:tp_size])
    if not candidates:
        return []
    return max(candidates, key=lambda grp: (min(_effective_vram_gb(g) for g in grp), grp[0]["sm_int"]))


def compute_array_bounds(hw: dict[str, Any], tier_id: str, ctx: int) -> dict[str, Any]:
    tier = MODEL_TIERS.get(tier_id, MODEL_TIERS["small"])
    gpus = _select_best_gpu(hw.get("gpus", []))
    if not gpus:
        return {
            "max_array": 1,
            "circles": [1],
            "supports_tp": [],
            "supports_dp": False,
            "dp_capacity": 0,
            "eligible_dp_gpu_count": 0,
            "per_gpu_dp_slots": [],
            "max_batch_single": 1,
        }

    primary = gpus[0]
    vram_gb = _effective_vram_gb(primary)
    headroom_gb = max(0.0, vram_gb - tier["base_vram_gb"] - 2.0)

    # Single-card continuous batching capacity estimate (~1.5 GiB per concurrent active slot at 64K Q4/FP8).
    per_slot_gb = max(0.6, (ctx / 65536.0) * (0.58 if vram_gb < 48.0 else 1.0))
    if vram_gb < 11.5:
        max_batch_single = 1
    else:
        max_batch_single = max(1, min(16, 1 + int(headroom_gb // max(0.75, per_slot_gb))))
        if vram_gb >= 72.0 and ctx <= 65536:
            max_batch_single = min(32, max(max_batch_single, 8))

    per_gpu_dp_slots: list[dict[str, Any]] = []
    dp_capacity = 0
    eligible_dp_gpu_count = 0
    for g in gpus:
        slots = _gpu_dp_slots(g, tier, ctx)
        if slots > 0:
            eligible_dp_gpu_count += 1
            dp_capacity += slots
        per_gpu_dp_slots.append(
            {
                "index": g["index"],
                "name": g["name"],
                "arch": g["arch"],
                "free_vram_gb": _effective_vram_gb(g),
                "total_vram_gb": g["vram_total_gb"],
                "slots": slots,
            }
        )

    supports_tp = [n for n in (2, 4) if _choose_tp_group(gpus, tier, n)]

    max_array = max(1, max_batch_single, dp_capacity, max(supports_tp) if supports_tp else 1)
    circles = [n for n in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32) if n <= max_array]
    if max_array not in circles and max_array <= 32:
        circles.append(max_array)
        circles.sort()
    if not circles:
        circles = [1]

    return {
        "max_array": max_array,
        "circles": circles,
        "supports_tp": supports_tp,
        "supports_dp": dp_capacity >= 2,
        "dp_capacity": dp_capacity,
        "eligible_dp_gpu_count": eligible_dp_gpu_count,
        "per_gpu_dp_slots": per_gpu_dp_slots,
        "max_batch_single": max_batch_single,
    }


def _choose_binary(hw: dict[str, Any], target_gpu: dict[str, Any] | None, tp_size: int) -> tuple[str, str, bool]:
    """Return (binary_path, build_command_from_repo_root, binary_ready) matching target_gpu's SM arch."""
    binaries = hw.get("binaries", {})
    if target_gpu is None:
        if "cpu" in binaries:
            return binaries["cpu"], "", True
        return (
            str(ROOT / "native" / "build-cpu" / "apps" / "serve" / "ingine-serve"),
            "(cd native && cmake --preset cpu-dev && cmake --build --preset cpu-dev --target ingine-serve)",
            False,
        )

    sm = target_gpu["sm_int"]
    arch_val = "120a" if sm >= 120 else str(sm)
    has_private_cuda = bool(hw.get("has_private_cuda"))

    if tp_size > 1:
        build_dir = f"native/build-tp{tp_size}-sm{arch_val}"
        bin_path = str(ROOT / build_dir / "apps" / "serve" / "ingine-serve")
        fp16_flag = " -DINGINE_ACT_FP16=ON" if sm < 80 else ""
        nvcc_hint = (
            " -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc"
            if sm < 80
            else ' -DCMAKE_CUDA_COMPILER="$(pwd)/.cuda-toolkit/nvidia/cu13/bin/nvcc"'
        )
        prefix_setup = "" if (sm < 80 or has_private_cuda) else "bash scripts/install.sh --build-only && "
        cmd = (
            f"{prefix_setup}cmake -G Ninja -S native -B {build_dir} -DCMAKE_BUILD_TYPE=Release "
            f"-DCMAKE_CUDA_ARCHITECTURES={arch_val} -DINGINE_TP_SIZE={tp_size}{fp16_flag}{nvcc_hint} "
            f"&& cmake --build {build_dir} --target ingine-serve"
        )
        tp_key = f"tp{tp_size}_sm{arch_val}"
        if tp_key in binaries:
            return binaries[tp_key], "", True
        ready = Path(bin_path).is_file() and os.access(bin_path, os.X_OK)
        return bin_path, ("" if ready else cmd), ready

    if sm == 70:
        for key in ("sm70", "legacy_fp16"):
            if key in binaries:
                return binaries[key], "", True
        return (
            str(ROOT / "native" / "build-sm70" / "apps" / "serve" / "ingine-serve"),
            "(cd native && cmake --preset sm70 && cmake --build --preset sm70 --target ingine-serve)",
            False,
        )

    if sm < 80:
        # Pascal sm_60/61 and Turing sm_75 require legacy-fp16 (NEVER sm70-only binary!).
        if "legacy_fp16" in binaries:
            return binaries["legacy_fp16"], "", True
        return (
            str(ROOT / "native" / "build-legacy-fp16" / "apps" / "serve" / "ingine-serve"),
            "(cd native && cmake --preset legacy-fp16 && cmake --build --preset legacy-fp16 --target ingine-serve)",
            False,
        )

    if sm in (80, 90):
        # A100 (sm_80) and H100 (sm_90) are not in the install preset's 86;89;120a list.
        key = f"sm{sm}"
        if key in binaries:
            return binaries[key], "", True
        build_dir = f"native/build-sm{sm}"
        prefix_setup = "" if has_private_cuda else "bash scripts/install.sh --build-only && "
        return (
            str(ROOT / build_dir / "apps" / "serve" / "ingine-serve"),
            (
                f"{prefix_setup}cmake -G Ninja -S native -B {build_dir} -DCMAKE_BUILD_TYPE=Release "
                f'-DCMAKE_CUDA_ARCHITECTURES={sm} -DCMAKE_CUDA_COMPILER="$(pwd)/.cuda-toolkit/nvidia/cu13/bin/nvcc" '
                f"&& cmake --build {build_dir} --target ingine-serve"
            ),
            False,
        )

    # sm_86 / sm_89 / sm_120a: install & multiarch include all three; release is sm_120a-only.
    accepted = ("install", "multiarch", "release") if sm >= 120 else ("install", "multiarch")
    for key in accepted:
        if key in binaries:
            return binaries[key], "", True
    cmd = (
        "(cd native && cmake --preset install && cmake --build --preset install --target ingine-serve)"
        if has_private_cuda
        else "bash scripts/install.sh --build-only"
    )
    return str(ROOT / "native" / "build-install" / "apps" / "serve" / "ingine-serve"), cmd, False


def _match_local_checkpoint(
    hw: dict[str, Any],
    tier_id: str,
    override_dir: str | None = None,
    *,
    require_weights: bool = True,
) -> tuple[str, dict[str, Any] | None]:
    tier = MODEL_TIERS[tier_id]
    if override_dir:
        p = Path(override_dir).expanduser().resolve()
        info = inspect_checkpoint_dir(p)
        return str(p), info
    for ckpt in hw.get("checkpoints", []):
        if require_weights and ckpt.get("inferred_tier") == tier_id and ckpt.get("usable_engine"):
            return ckpt["path"], ckpt
    if not require_weights:
        for ckpt in hw.get("checkpoints", []):
            if ckpt.get("complete_fake_assets"):
                return ckpt["path"], ckpt
    default_dir = ROOT / "models" / tier["local_dir_name"]
    info = inspect_checkpoint_dir(default_dir)
    return str(default_dir), info


def _tune_worker_for_gpu(
    hw: dict[str, Any],
    gpu: dict[str, Any],
    tier: dict[str, Any],
    ctx: int,
    ckpt_path: str,
    ckpt_info: dict[str, Any] | None,
    *,
    kv_cache_mode: str,
    max_batch: int,
    tp_size: int,
    vision: bool | None,
    spec: bool | None,
    vram_budget_gb: float | None = None,
) -> dict[str, Any]:
    """Compute matching binary, flags, footprint, and warnings for one GPU (or TP bottleneck GPU)."""
    warnings: list[str] = []
    tier_id = tier["id"]
    total_gb = float(gpu["vram_total_gb"])
    free_gb = _effective_vram_gb(gpu)
    vram_gb = min(free_gb, vram_budget_gb) if vram_budget_gb is not None else free_gb
    sm = gpu["sm_int"]

    if total_gb - free_gb >= 2.0:
        warnings.append(
            f"GPU {gpu['index']} ({gpu['name']}) currently has {free_gb:.1f}/{total_gb:.1f} GB free VRAM; sizing worker to free VRAM."
        )
    if sm >= 80 and gpu["driver_major"] and gpu["driver_major"] < MIN_CUDA13_DRIVER:
        warnings.append(
            f"GPU {gpu['index']}: NVIDIA driver {gpu['driver']} is older than {MIN_CUDA13_DRIVER}; CUDA 13.4 builds require driver >= 580."
        )

    use_atlas_6g = False
    kv_format = "fp8"
    kv_paged = False
    kv_vram_mb = 0
    kv_ram_mb = 12288
    prefill_chunk = 3072
    use_dense_fp8 = True
    use_vision = True if vision is None else bool(vision)
    use_spec = True if spec is None else bool(spec)
    eff_kv_mode = kv_cache_mode

    if vram_gb < 11.0 and tp_size == 1:
        use_atlas_6g = True
        if tier_id != "small":
            warnings.append(
                f"GPU {gpu['index']}: {tier['label']} ({tier['disk_gb']} GB) exceeds {vram_gb:.1f} GB free VRAM; select Small (EXL3 9 GB) for <=10 GB cards."
            )
        kv_format = "q4" if ctx >= 131072 else "fp8"
        if ctx >= 131072:
            kv_paged = True
            kv_vram_mb = 512
            kv_ram_mb = min(12288, max(4096, int((hw.get("ram_free_gb", 16.0) - 2.0) * 1024)))
            prefill_chunk = 512
        else:
            prefill_chunk = 1024
        use_vision = False
        use_spec = False
    elif vram_gb < 22.0 and tp_size == 1:
        if ctx >= 262144:
            kv_format = "q4"
            kv_paged = True
            kv_vram_mb = 2048 if vram_gb >= 15.0 else 1024
            kv_ram_mb = min(12288, max(4096, int((hw.get("ram_free_gb", 16.0) - 2.0) * 1024)))
            prefill_chunk = 512
        else:
            kv_format = "q4" if vram_gb < 15.0 else "fp8"
            prefill_chunk = 1024 if ctx >= 65536 else 2048
        if tier_id == "large":
            warnings.append(
                f"GPU {gpu['index']}: Large (20.2 GB) exceeds {vram_gb:.1f} GB free VRAM; switch to Small (EXL3 9 GB) or Compact (atlas-10g)."
            )
    elif vram_gb < 31.0:
        if ctx >= 524288:
            kv_format = "q4"
            prefill_chunk = 2048
            if tier_id == "large" and tp_size == 1:
                kv_paged = True
                kv_vram_mb = 1024
                warnings.append(
                    f"GPU {gpu['index']}: 1M context with Large (20.2 GB) on {vram_gb:.1f} GB uses paged Q4 KV; Small (EXL3 9 GB) runs 1M fully resident."
                )
            if vision is None:
                use_vision = False
        elif ctx >= 262144:
            kv_format = "q4"
            prefill_chunk = 2048
        else:
            kv_format = "fp8"
            prefill_chunk = 3072
    elif vram_gb < 64.0:
        if ctx >= 524288:
            kv_format = "q4"
            prefill_chunk = 2048
        else:
            kv_format = "fp8"
            prefill_chunk = 3072
    else:
        kv_format = "fp8"
        prefill_chunk = 3072

    if max_batch > 1:
        if eff_kv_mode == "last_full":
            eff_kv_mode = "per_pass"
            warnings.append("--kv-cache-mode last_full is B=1 non-spec only; using per_pass for batched mode.")
        if use_spec:
            use_spec = False
            warnings.append("DSpark --spec is disabled for --max-batch > 1 on main (batched MTP staged under #851).")

    if eff_kv_mode == "last_full" and use_spec:
        use_spec = False
        warnings.append("--kv-cache-mode last_full is non-spec only; disabled --spec.")

    if ckpt_info is not None and not ckpt_info.get("has_mtp") and use_spec:
        use_spec = False
        warnings.append(f"mtp.safetensors not found in {ckpt_path}; disabled --spec.")

    binary_path, build_command, binary_ready = _choose_binary(hw, gpu, tp_size)

    flags: list[str] = ["--engine", "--model-dir", ckpt_path, "--max-ctx", str(ctx)]
    if use_atlas_6g:
        flags.append("--atlas-6g")
    elif use_dense_fp8:
        flags.append("--dense-fp8")

    if kv_paged:
        flags.extend(["--kv-q4", "--kv-paged", "--kv-vram-mb", str(kv_vram_mb), "--kv-ram-mb", str(kv_ram_mb)])
    elif kv_format == "q4":
        flags.append("--kv-q4")
    else:
        flags.append("--kv-fp8")

    if eff_kv_mode == "last_full":
        flags.extend(["--kv-cache-mode", "last_full"])

    if prefill_chunk != 3072:
        flags.extend(["--prefill-chunk", str(prefill_chunk)])

    if use_spec:
        flags.extend(["--spec", "--spec-threshold", "0.8"])

    if not use_vision and not use_atlas_6g:
        flags.append("--no-vision")

    if max_batch > 1:
        flags.extend(["--max-batch", str(max_batch)])

    if ckpt_info and ckpt_info.get("has_ablation"):
        flags.extend(["--ablation", str(Path(ckpt_path) / "inkling-2-mini-abliterated.safetensors")])

    flags.extend(["--model-id", tier["model_id"]])

    kv_bpt = KV_BYTES_PER_TOK[(kv_format, eff_kv_mode)]
    kv_total_gb = (ctx * kv_bpt) / (1024.0 ** 3)
    if use_atlas_6g:
        est_vram_gb = 5.4
        est_ram_gb = round(min(16.0, 4.0 + kv_ram_mb / 1024.0), 1)
    else:
        vram_kv_gb = (kv_vram_mb / 1024.0) if kv_paged else kv_total_gb
        batch_extra_gb = 0.0 if max_batch <= 1 else max(0.0, (max_batch - 1) * 0.9)
        vision_gb = 0.6 if use_vision else 0.0
        spec_gb = 0.7 if use_spec else 0.0
        per_gpu_weights = tier["base_vram_gb"] / tp_size
        est_vram_gb = round(per_gpu_weights + vram_kv_gb + batch_extra_gb + vision_gb + spec_gb, 2)
        est_ram_gb = round((kv_ram_mb / 1024.0 if kv_paged else 1.2) + 0.5, 1)

    return {
        "binary_path": binary_path,
        "build_command": build_command,
        "binary_ready": binary_ready,
        "flags": flags,
        "kv_format": kv_format,
        "kv_cache_mode": eff_kv_mode,
        "kv_paged": kv_paged,
        "vision": use_vision,
        "spec": use_spec,
        "estimated_vram_gb": est_vram_gb,
        "estimated_ram_gb": est_ram_gb,
        "warnings": warnings,
    }


def build_plan(
    hw: dict[str, Any],
    *,
    tier_id: str | None = None,
    ctx: int | None = None,
    array_count: int | None = None,
    layout: str = "auto",
    mode: str = "chat",
    kv_cache_mode: str = "per_pass",
    host: str = DEFAULT_SERVE_HOST,
    port: int = DEFAULT_SERVE_PORT,
    model_dir: str | None = None,
    vision: bool | None = None,
    spec: bool | None = None,
    gpu_indices: list[int] | None = None,
) -> dict[str, Any]:
    defaults = recommend_defaults(hw)
    tier_id = tier_id if tier_id in MODEL_TIERS else defaults["tier"]
    tier = MODEL_TIERS[tier_id]
    ctx = int(ctx) if ctx else defaults["ctx"]
    if ctx < 4096:
        ctx = 4096
    if ctx > 1048576:
        ctx = 1048576

    mode = "api" if mode == "api" else "chat"
    kv_cache_mode = "last_full" if kv_cache_mode == "last_full" else "per_pass"

    gpus = _select_best_gpu(hw.get("gpus", []), gpu_indices)
    primary = gpus[0] if gpus else None
    bounds = compute_array_bounds(hw, tier_id, ctx)

    requested_array = max(1, int(array_count or defaults["array_count"]))
    clamped_array = min(requested_array, bounds["max_array"])
    warnings: list[str] = []
    if requested_array > clamped_array:
        warnings.append(
            f"Requested array count {requested_array} exceeds detected hardware capacity ({bounds['max_array']}); clamped to {clamped_array}."
        )

    # Resolve layout: single | batch | dp | tp | cpu_fake
    resolved_layout = layout
    tp_size = 1
    tp_group: list[dict[str, Any]] = []

    if primary is None:
        resolved_layout = "cpu_fake"
        clamped_array = 1
        warnings.append("No NVIDIA CUDA GPU detected; falling back to CPU --fake smoke mode until a supported GPU is present.")
        if hw.get("other_vendors"):
            names = ", ".join(v["vendor"] for v in hw["other_vendors"])
            warnings.append(f"Detected non-NVIDIA hardware ({names}); multi-vendor backends are staged under #835.")
    elif clamped_array == 1:
        resolved_layout = "single"
    else:
        if resolved_layout == "tp":
            tp_group = _choose_tp_group(gpus, tier, clamped_array)
            if tp_group:
                tp_size = clamped_array
            else:
                warnings.append(
                    f"TP={clamped_array} requires {clamped_array} same-architecture GPUs with sufficient free VRAM; using DP/batch instead."
                )
                resolved_layout = "auto"
        if resolved_layout == "dp" and bounds["dp_capacity"] < clamped_array:
            if bounds["dp_capacity"] >= 1:
                warnings.append(
                    f"DP capacity across free GPU VRAM is {bounds['dp_capacity']} worker(s); clamped from {clamped_array}."
                )
                clamped_array = bounds["dp_capacity"]
                if clamped_array == 1:
                    resolved_layout = "single"
            else:
                warnings.append("No GPU has enough free VRAM for DP replicas; falling back to single worker.")
                clamped_array = 1
                resolved_layout = "single"
        if resolved_layout == "auto":
            if bounds["eligible_dp_gpu_count"] >= clamped_array:
                resolved_layout = "dp"
            elif clamped_array <= bounds["max_batch_single"]:
                resolved_layout = "batch"
            elif bounds["dp_capacity"] >= clamped_array:
                resolved_layout = "dp"
            else:
                tp_group = _choose_tp_group(gpus, tier, clamped_array)
                if tp_group:
                    resolved_layout = "tp"
                    tp_size = clamped_array
                else:
                    resolved_layout = "single"
                    clamped_array = 1

    if resolved_layout == "tp":
        if not tp_group:
            tp_group = _choose_tp_group(gpus, tier, tp_size)
        if not hw.get("iommu", {}).get("p2p_friendly", False):
            warnings.append(
                "Multi-GPU TP on PCIe performs best with kernel boot parameter iommu=pt (P2P DMA fallback works automatically, slightly slower)."
            )
        if kv_cache_mode == "last_full":
            kv_cache_mode = "per_pass"
            warnings.append("TP mode requires --kv-cache-mode per_pass; switched automatically.")

    require_weights = resolved_layout != "cpu_fake"
    ckpt_path, ckpt_info = _match_local_checkpoint(hw, tier_id, model_dir, require_weights=require_weights)
    if require_weights:
        checkpoint_present = bool(ckpt_info and ckpt_info.get("usable_engine"))
    else:
        checkpoint_present = bool(ckpt_info and ckpt_info.get("complete_fake_assets"))

    needs_download = not checkpoint_present
    if ckpt_info is not None and needs_download and ckpt_info.get("missing_files"):
        warnings.append(
            f"Incomplete checkpoint at {ckpt_path}; missing: {', '.join(ckpt_info['missing_files'][:6])}."
        )

    download_files: list[str] = []
    if needs_download:
        if require_weights:
            download_gb = tier["disk_gb"]
            download_cmd = f"hf download {tier['repo_id']} --local-dir {shlex.quote(ckpt_path)}"
        else:
            download_gb = 0.03
            download_files = list(FAKE_DOWNLOAD_FILES)
            files_arg = " ".join(shlex.quote(f) for f in download_files)
            download_cmd = f"hf download {tier['repo_id']} {files_arg} --local-dir {shlex.quote(ckpt_path)}"
    else:
        download_gb = 0.0
        download_cmd = ""

    if needs_download and hw.get("disk_free_gb", 0.0) > 0 and hw["disk_free_gb"] < download_gb + 2.0:
        warnings.append(
            f"Low free disk space ({hw['disk_free_gb']:.1f} GB free, need ~{download_gb + 2.0:.1f} GB for {tier['repo_id']})."
        )

    # Construct per-worker launch commands with per-GPU capacity, binary, and flags.
    workers: list[dict[str, Any]] = []
    build_commands: list[str] = []

    def _add_build_cmd(cmd: str) -> None:
        if cmd and cmd not in build_commands:
            build_commands.append(cmd)

    if resolved_layout == "cpu_fake":
        binary_path, build_cmd, binary_ready = _choose_binary(hw, None, 1)
        _add_build_cmd(build_cmd)
        w_argv = [
            binary_path,
            "--fake",
            "--model-dir",
            ckpt_path,
            "--model-id",
            f"{tier['model_id']}-fake",
            "--host",
            host,
            "--port",
            str(port),
        ]
        workers.append(
            {
                "worker_index": 0,
                "cuda_visible_devices": "",
                "gpu_name": "CPU (--fake)",
                "arch": "cpu",
                "binary_path": binary_path,
                "binary_ready": binary_ready,
                "build_command": build_cmd,
                "estimated_vram_gb": 0.0,
                "host": host,
                "port": port,
                "base_url": f"http://{host}:{port}/v1",
                "argv": w_argv,
                "shell": " ".join(shlex.quote(a) for a in w_argv),
            }
        )
        kv_format = "fp8"
        kv_paged = False
        use_vision = False
        use_spec = False
        est_vram_gb = 0.0
        est_ram_gb = 0.5
    elif resolved_layout == "tp":
        # Derive architecture, binary, and flags strictly from the actual TP group's bottleneck GPU.
        tp_bottleneck = min(tp_group, key=_effective_vram_gb)
        tuned = _tune_worker_for_gpu(
            hw,
            tp_bottleneck,
            tier,
            ctx,
            ckpt_path,
            ckpt_info,
            kv_cache_mode=kv_cache_mode,
            max_batch=1,
            tp_size=tp_size,
            vision=vision,
            spec=spec,
        )
        warnings.extend(w for w in tuned["warnings"] if w not in warnings)
        _add_build_cmd(tuned["build_command"])
        vis = ",".join(str(g["index"]) for g in tp_group)
        w_argv = [tuned["binary_path"], *tuned["flags"], "--host", host, "--port", str(port)]
        workers.append(
            {
                "worker_index": 0,
                "cuda_visible_devices": vis,
                "gpu_name": f"{tp_size}x {tp_group[0]['name']} (TP={tp_size})",
                "arch": tp_bottleneck["arch"],
                "binary_path": tuned["binary_path"],
                "binary_ready": tuned["binary_ready"],
                "build_command": tuned["build_command"],
                "estimated_vram_gb": tuned["estimated_vram_gb"],
                "host": host,
                "port": port,
                "base_url": f"http://{host}:{port}/v1",
                "argv": w_argv,
                "shell": f"CUDA_VISIBLE_DEVICES={vis} " + " ".join(shlex.quote(a) for a in w_argv),
            }
        )
        kv_format = tuned["kv_format"]
        kv_cache_mode = tuned["kv_cache_mode"]
        kv_paged = tuned["kv_paged"]
        use_vision = tuned["vision"]
        use_spec = tuned["spec"]
        est_vram_gb = tuned["estimated_vram_gb"]
        est_ram_gb = tuned["estimated_ram_gb"]
    elif resolved_layout == "dp":
        # Spread 1 worker per eligible GPU first, then fill remaining per-GPU capacity without oversubscribing.
        gpu_slots = {g["index"]: _gpu_dp_slots(g, tier, ctx) for g in gpus}
        assigned_gpus: list[dict[str, Any]] = []
        used_on_gpu: dict[int, int] = {g["index"]: 0 for g in gpus}
        while len(assigned_gpus) < clamped_array:
            progressed = False
            for g in gpus:
                if used_on_gpu[g["index"]] < gpu_slots[g["index"]]:
                    assigned_gpus.append(g)
                    used_on_gpu[g["index"]] += 1
                    progressed = True
                    if len(assigned_gpus) >= clamped_array:
                        break
            if not progressed:
                break

        tuned_list: list[dict[str, Any]] = []
        for i, g in enumerate(assigned_gpus):
            copies_on_g = used_on_gpu[g["index"]]
            per_worker_budget = _effective_vram_gb(g) / max(1, copies_on_g)
            tuned = _tune_worker_for_gpu(
                hw,
                g,
                tier,
                ctx,
                ckpt_path,
                ckpt_info,
                kv_cache_mode=kv_cache_mode,
                max_batch=1,
                tp_size=1,
                vision=vision,
                spec=spec,
                vram_budget_gb=per_worker_budget,
            )
            tuned_list.append(tuned)
            warnings.extend(w for w in tuned["warnings"] if w not in warnings)
            _add_build_cmd(tuned["build_command"])
            w_port = port + i
            w_argv = [tuned["binary_path"], *tuned["flags"], "--host", host, "--port", str(w_port)]
            workers.append(
                {
                    "worker_index": i,
                    "cuda_visible_devices": str(g["index"]),
                    "gpu_name": g["name"],
                    "arch": g["arch"],
                    "binary_path": tuned["binary_path"],
                    "binary_ready": tuned["binary_ready"],
                    "build_command": tuned["build_command"],
                    "estimated_vram_gb": tuned["estimated_vram_gb"],
                    "host": host,
                    "port": w_port,
                    "base_url": f"http://{host}:{w_port}/v1",
                    "argv": w_argv,
                    "shell": f"CUDA_VISIBLE_DEVICES={g['index']} " + " ".join(shlex.quote(a) for a in w_argv),
                }
            )
        clamped_array = len(workers)
        lead = tuned_list[0]
        kv_format = lead["kv_format"]
        kv_cache_mode = lead["kv_cache_mode"]
        kv_paged = any(t["kv_paged"] for t in tuned_list)
        use_vision = all(t["vision"] for t in tuned_list)
        use_spec = all(t["spec"] for t in tuned_list)
        est_vram_gb = max(t["estimated_vram_gb"] for t in tuned_list)
        est_ram_gb = round(sum(t["estimated_ram_gb"] for t in tuned_list), 1)
    else:
        assert primary is not None
        max_batch = clamped_array if resolved_layout == "batch" else 1
        tuned = _tune_worker_for_gpu(
            hw,
            primary,
            tier,
            ctx,
            ckpt_path,
            ckpt_info,
            kv_cache_mode=kv_cache_mode,
            max_batch=max_batch,
            tp_size=1,
            vision=vision,
            spec=spec,
        )
        warnings.extend(w for w in tuned["warnings"] if w not in warnings)
        _add_build_cmd(tuned["build_command"])
        vis = str(primary["index"])
        w_argv = [tuned["binary_path"], *tuned["flags"], "--host", host, "--port", str(port)]
        workers.append(
            {
                "worker_index": 0,
                "cuda_visible_devices": vis,
                "gpu_name": primary["name"],
                "arch": primary["arch"],
                "binary_path": tuned["binary_path"],
                "binary_ready": tuned["binary_ready"],
                "build_command": tuned["build_command"],
                "estimated_vram_gb": tuned["estimated_vram_gb"],
                "host": host,
                "port": port,
                "base_url": f"http://{host}:{port}/v1",
                "argv": w_argv,
                "shell": f"CUDA_VISIBLE_DEVICES={vis} " + " ".join(shlex.quote(a) for a in w_argv),
            }
        )
        kv_format = tuned["kv_format"]
        kv_cache_mode = tuned["kv_cache_mode"]
        kv_paged = tuned["kv_paged"]
        use_vision = tuned["vision"]
        use_spec = tuned["spec"]
        est_vram_gb = tuned["estimated_vram_gb"]
        est_ram_gb = tuned["estimated_ram_gb"]

    hardware_ready = resolved_layout not in ("single", "batch") or _gpu_dp_slots(primary, tier, ctx) > 0
    if not hardware_ready:
        warnings.append(f"{primary['name']} has insufficient free GPU memory for {tier['label']}.")
    binary_path = workers[0]["binary_path"]
    binary_ready = all(w["binary_ready"] for w in workers)
    build_command = " && ".join(build_commands)

    return {
        "tier": tier,
        "ctx": ctx,
        "array_count": clamped_array,
        "requested_array": requested_array,
        "hardware_ready": hardware_ready,
        "layout": resolved_layout,
        "tp_size": tp_size,
        "mode": mode,
        "kv_format": kv_format,
        "kv_cache_mode": kv_cache_mode,
        "kv_paged": kv_paged,
        "vision": use_vision,
        "spec": use_spec,
        "model_dir": ckpt_path,
        "checkpoint_present": checkpoint_present,
        "needs_download": needs_download,
        "download_files": download_files,
        "download_gb": download_gb,
        "download_command": download_cmd,
        "binary_path": binary_path,
        "binary_ready": binary_ready,
        "build_command": build_command,
        "build_commands": build_commands,
        "estimated_vram_per_gpu_gb": est_vram_gb,
        "estimated_host_ram_gb": est_ram_gb,
        "array_bounds": bounds,
        "workers": workers,
        "warnings": warnings,
        "endpoint_url": workers[0]["base_url"],
        "model_id": tier["model_id"] if require_weights else f"{tier['model_id']}-fake",
    }


def _http_json(url: str, timeout: float = 2.0) -> tuple[bool, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return True, json.loads(raw)
    except Exception as exc:
        return False, str(exc)


class AtlasSetupController:
    """Permission-gated download and ingine-serve lifecycle controller."""

    def __init__(self, root: Path = ROOT, hw: dict[str, Any] | None = None) -> None:
        self.root = root
        self._lock = threading.RLock()
        self._log_lock = threading.Lock()
        self._hw_override = hw
        self._download_proc: subprocess.Popen[str] | None = None
        self._worker_procs: list[tuple[dict[str, Any], subprocess.Popen[str]]] = []
        self._logs: deque[str] = deque(maxlen=250)
        self._active_plan: dict[str, Any] | None = None
        atexit.register(self.stop_all)

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with self._log_lock:
            self._logs.append(f"[{ts}] {msg}")

    def get_hardware(self) -> dict[str, Any]:
        if self._hw_override is not None:
            return self._hw_override
        return detect_hardware(self.root)

    def _pipe_reader(self, prefix: str, stream: Any) -> None:
        try:
            for line in iter(stream.readline, ""):
                s = line.rstrip()
                if s:
                    self.log(f"{prefix}: {s}")
        finally:
            stream.close()

    def start_download(self, plan: dict[str, Any], *, approved: bool) -> dict[str, Any]:
        if approved is not True:
            raise PermissionError("Explicit user approval (approved=true) is required before downloading from Hugging Face.")
        if not plan.get("needs_download"):
            return {"ok": True, "message": "Checkpoint already present locally.", "model_dir": plan["model_dir"]}

        with self._lock:
            if self._download_proc is not None and self._download_proc.poll() is None:
                return {"ok": True, "message": "Download already in progress."}

            tier = plan["tier"]
            dest = Path(plan["model_dir"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            files = plan.get("download_files", [])
            hf_bin = shutil.which("hf") or shutil.which("huggingface-cli")
            if hf_bin:
                cmd = [hf_bin, "download", tier["repo_id"], *files, "--local-dir", str(dest)]
            else:
                cmd = [
                    sys.executable,
                    "-c",
                    (
                        "from huggingface_hub import snapshot_download; "
                        f"snapshot_download(repo_id={tier['repo_id']!r}, local_dir={str(dest)!r}, allow_patterns={files or None!r})"
                    ),
                ]
            self.log(f"Starting approved download: {' '.join(cmd)}")
            self._download_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            threading.Thread(
                target=self._pipe_reader,
                args=("download", self._download_proc.stdout),
                daemon=True,
            ).start()
            return {"ok": True, "message": f"Downloading {tier['repo_id']} into {dest}", "pid": self._download_proc.pid}

    def launch_workers(self, plan: dict[str, Any], *, approved: bool) -> dict[str, Any]:
        if approved is not True:
            raise PermissionError("Explicit user approval (approved=true) is required before launching ingine-serve.")
        if not plan.get("hardware_ready", True):
            raise RuntimeError("Insufficient free GPU memory for the selected plan.")
        if not plan.get("binary_ready"):
            raise RuntimeError(f"ingine-serve binary not built yet. Run: {plan['build_command']}")
        if plan.get("needs_download"):
            raise RuntimeError(f"Model files missing at {plan['model_dir']}. Approve download first.")

        with self._lock:
            self.stop_workers()
            started: list[dict[str, Any]] = []
            try:
                for w in plan["workers"]:
                    env = os.environ.copy()
                    if w["cuda_visible_devices"]:
                        env["CUDA_VISIBLE_DEVICES"] = w["cuda_visible_devices"]
                    self.log(f"Launching worker #{w['worker_index']} on {w['host']}:{w['port']}: {w['shell']}")
                    proc = subprocess.Popen(
                        w["argv"],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    self._worker_procs.append((w, proc))
                    threading.Thread(
                        target=self._pipe_reader,
                        args=(f"w{w['worker_index']}", proc.stdout),
                        daemon=True,
                    ).start()
                    started.append({"worker_index": w["worker_index"], "pid": proc.pid, "base_url": w["base_url"]})
            except BaseException:
                self.stop_workers()
                raise
            self._active_plan = plan
            return {"ok": True, "workers": started}

    @staticmethod
    def _stop_process(proc: subprocess.Popen[str]) -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def stop_workers(self) -> None:
        with self._lock:
            for w, proc in self._worker_procs:
                self.log(f"Stopping worker #{w['worker_index']} (pid {proc.pid})")
                self._stop_process(proc)
            self._worker_procs.clear()
            self._active_plan = None

    def stop_all(self) -> None:
        with self._lock:
            self.stop_workers()
            if self._download_proc is not None:
                self._stop_process(self._download_proc)
                self._download_proc = None

    def status(self, probe_host: str = DEFAULT_SERVE_HOST, probe_port: int = DEFAULT_SERVE_PORT) -> dict[str, Any]:
        with self._lock:
            dl_running = self._download_proc is not None and self._download_proc.poll() is None
            dl_rc = None if self._download_proc is None else self._download_proc.poll()
            worker_items = list(self._worker_procs)
            active_plan = self._active_plan
        with self._log_lock:
            logs = list(self._logs)[-80:]

        worker_states: list[dict[str, Any]] = []
        any_healthy = False
        if worker_items:
            for w, proc in worker_items:
                rc = proc.poll()
                ok_h, health = _http_json(f"http://{w['host']}:{w['port']}/health", timeout=1.5) if rc is None else (False, None)
                ok_m, models = _http_json(f"http://{w['host']}:{w['port']}/v1/models", timeout=1.5) if ok_h else (False, None)
                if ok_h:
                    any_healthy = True
                worker_states.append(
                    {
                        "worker_index": w["worker_index"],
                        "pid": proc.pid,
                        "alive": rc is None,
                        "exit_code": rc,
                        "base_url": w["base_url"],
                        "healthy": ok_h,
                        "health": health if ok_h else None,
                        "models": models if ok_m else None,
                    }
                )
        else:
            ok_h, health = _http_json(f"http://{probe_host}:{probe_port}/health", timeout=1.0)
            if ok_h:
                any_healthy = True
                ok_m, models = _http_json(f"http://{probe_host}:{probe_port}/v1/models", timeout=1.0)
                worker_states.append(
                    {
                        "worker_index": 0,
                        "pid": None,
                        "alive": True,
                        "exit_code": None,
                        "base_url": f"http://{probe_host}:{probe_port}/v1",
                        "healthy": True,
                        "health": health,
                        "models": models if ok_m else None,
                        "external": True,
                    }
                )

        state = "idle"
        if dl_running:
            state = "downloading"
        elif any(worker["exit_code"] is not None for worker in worker_states):
            state = "exited"
        elif any_healthy:
            state = "ready"
        elif worker_items:
            state = "starting"
        return {
            "state": state,
            "download_running": dl_running,
            "download_exit_code": dl_rc,
            "workers": worker_states,
            "endpoint_url": worker_states[0]["base_url"] if worker_states else (active_plan["endpoint_url"] if active_plan else f"http://{probe_host}:{probe_port}/v1"),
            "logs": logs,
        }


ATLAS_UI_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Atlas Ingine — Hardware-Aware Setup & Launcher</title>
<style>
  :root {
    --bg: #070a0d;
    --surface: #0d131a;
    --surface-2: #121b24;
    --border: #1e2d3b;
    --border-hi: #10b981;
    --text: #e6eef5;
    --muted: #8ea3b8;
    --emerald: #10b981;
    --emerald-bright: #34d399;
    --emerald-dim: rgba(16, 185, 129, 0.14);
    --amber: #f59e0b;
    --rose: #f43f5e;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: radial-gradient(1200px 600px at 15% -10%, rgba(16,185,129,0.12), transparent 60%), var(--bg);
    color: var(--text);
    font-family: var(--sans);
    line-height: 1.45;
  }
  header {
    display: flex; align-items: center; justify-content: space-between; gap: 16px;
    padding: 16px 24px; border-bottom: 1px solid var(--border);
    background: rgba(13,19,26,0.9); position: sticky; top: 0; z-index: 10; backdrop-filter: blur(8px);
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand h1 { margin: 0; font-size: 18px; letter-spacing: 0.03em; font-weight: 700; }
  .brand span { color: var(--emerald-bright); font-size: 12px; display: block; font-family: var(--mono); }
  .pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 5px 11px; border-radius: 999px;
    font-size: 12px; font-family: var(--mono); border: 1px solid var(--border); background: var(--surface-2);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .dot.ready { background: var(--emerald-bright); box-shadow: 0 0 8px var(--emerald); }
  .dot.busy { background: var(--amber); }
  main { max-width: 1220px; margin: 0 auto; padding: 20px 24px 48px; display: grid; gap: 18px; }
  .grid-2 { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; }
  @media (max-width: 920px) { .grid-2 { grid-template-columns: 1fr; } }
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 18px;
  }
  .card h2 {
    margin: 0 0 12px; font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--emerald-bright); font-family: var(--mono);
  }
  .hw-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
  .stat {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px;
  }
  .stat .k { font-size: 11px; color: var(--muted); text-transform: uppercase; font-family: var(--mono); }
  .stat .v { font-size: 15px; font-weight: 600; margin-top: 2px; }
  .stat .s { font-size: 11px; color: var(--muted); margin-top: 2px; font-family: var(--mono); }
  .section-label { font-size: 12px; color: var(--muted); margin: 14px 0 8px; font-weight: 600; }
  .circle-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
  .circle-btn {
    width: 78px; height: 78px; border-radius: 50%; border: 2px solid var(--border);
    background: var(--surface-2); color: var(--text); cursor: pointer;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    transition: all 0.15s ease; padding: 6px; text-align: center;
  }
  .circle-btn.sm { width: 52px; height: 52px; }
  .circle-btn:hover { border-color: var(--emerald); transform: translateY(-1px); }
  .circle-btn.active {
    border-color: var(--emerald-bright); background: var(--emerald-dim);
    box-shadow: 0 0 0 3px rgba(16,185,129,0.22);
  }
  .circle-btn .big { font-size: 16px; font-weight: 700; font-family: var(--mono); color: var(--emerald-bright); }
  .circle-btn .sub { font-size: 10px; color: var(--muted); margin-top: 2px; line-height: 1.1; }
  .seg-row { display: flex; flex-wrap: wrap; gap: 8px; }
  .seg-btn {
    padding: 7px 13px; border-radius: 999px; border: 1px solid var(--border);
    background: var(--surface-2); color: var(--text); cursor: pointer; font-size: 12px; font-family: var(--mono);
  }
  .seg-btn.active { border-color: var(--emerald-bright); background: var(--emerald-dim); color: var(--emerald-bright); }
  .seg-btn:disabled { opacity: 0.4; cursor: not-allowed; }
  pre.cmd {
    background: #05080b; border: 1px solid var(--border); border-radius: 10px; padding: 12px;
    overflow-x: auto; font-size: 12px; font-family: var(--mono); color: #b7f5d8; margin: 8px 0;
    white-space: pre-wrap; word-break: break-all;
  }
  .warn {
    background: rgba(245, 158, 11, 0.11); border: 1px solid rgba(245, 158, 11, 0.35);
    color: #fcd34d; border-radius: 8px; padding: 8px 11px; font-size: 12px; margin-top: 8px;
  }
  .consent-box {
    display: flex; align-items: flex-start; gap: 10px; margin: 12px 0; padding: 10px 12px;
    border-radius: 10px; border: 1px solid var(--border); background: var(--surface-2); font-size: 13px;
  }
  .actions { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
  button.primary {
    background: var(--emerald); color: #04110b; border: 0; border-radius: 10px; padding: 10px 16px;
    font-weight: 700; cursor: pointer; font-size: 13px;
  }
  button.primary:disabled { opacity: 0.45; cursor: not-allowed; }
  button.secondary {
    background: var(--surface-2); color: var(--text); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 14px; font-weight: 600; cursor: pointer; font-size: 13px;
  }
  button.danger {
    background: rgba(244,63,94,0.16); color: #fda4af; border: 1px solid rgba(244,63,94,0.4);
    border-radius: 10px; padding: 10px 14px; font-weight: 600; cursor: pointer; font-size: 13px;
  }
  .chat-box {
    height: 300px; overflow-y: auto; background: #05080b; border: 1px solid var(--border);
    border-radius: 10px; padding: 12px; display: flex; flex-direction: column; gap: 10px;
  }
  .msg { padding: 9px 12px; border-radius: 10px; max-width: 90%; font-size: 13px; white-space: pre-wrap; }
  .msg.user { align-self: flex-end; background: var(--emerald-dim); border: 1px solid rgba(16,185,129,0.4); }
  .msg.assistant { align-self: flex-start; background: var(--surface-2); border: 1px solid var(--border); }
  .msg img { max-width: 180px; max-height: 140px; border-radius: 6px; display: block; margin-bottom: 6px; }
  .chat-form { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
  .chat-form input[type="text"] {
    flex: 1; min-width: 220px; background: var(--surface-2); border: 1px solid var(--border);
    color: var(--text); border-radius: 10px; padding: 10px 12px; font-size: 13px;
  }
  .logs {
    height: 140px; overflow-y: auto; background: #05080b; border: 1px solid var(--border);
    border-radius: 10px; padding: 10px; font-family: var(--mono); font-size: 11px; color: #9fb3c8;
  }
  .upcoming { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
  .tag { font-size: 11px; font-family: var(--mono); padding: 3px 8px; border-radius: 6px; background: var(--surface-2); border: 1px solid var(--border); color: var(--muted); }
</style>
</head>
<body>
<header>
  <div class="brand">
    <svg width="36" height="36" viewBox="0 0 64 64" fill="none" aria-label="Atlas Logo">
      <rect width="64" height="64" rx="14" fill="#0d131a" stroke="#10b981" stroke-width="2.5"/>
      <path d="M32 11L51 49H42.5L32 27.5L21.5 49H13L32 11Z" fill="#10b981"/>
      <circle cx="32" cy="43" r="5" fill="#34d399"/>
    </svg>
    <div>
      <h1>ATLAS INGINE</h1>
      <span>Hardware-Aware Setup &amp; GPU Auto-Tuner</span>
    </div>
  </div>
  <div class="pill"><span id="statusDot" class="dot"></span><span id="statusText">Detecting...</span></div>
</header>

<main>
  <section class="card">
    <h2>1. Detected Hardware &amp; Local Assets</h2>
    <div id="hwGrid" class="hw-grid"></div>
  </section>

  <div class="grid-2">
    <section class="card">
      <h2>2. Select Model Tier, Array Count &amp; Mode</h2>

      <div class="section-label">Model Tier (clickable circles)</div>
      <div id="tierCircles" class="circle-row"></div>
      <div id="tierNotes" style="font-size:12px;color:var(--muted);margin-top:8px;"></div>
      <div id="upcomingRow" class="upcoming"></div>

      <div class="section-label">Array Count (dynamically bounded by detected VRAM / GPUs)</div>
      <div id="arrayCircles" class="circle-row"></div>

      <div class="section-label">Layout &amp; Context Window</div>
      <div class="seg-row" id="layoutRow" style="margin-bottom:8px;"></div>
      <div class="seg-row" id="ctxRow"></div>

      <div class="section-label">Serve Mode</div>
      <div class="seg-row" id="modeRow"></div>
    </section>

    <section class="card">
      <h2>3. Computed Plan &amp; Explicit Permission</h2>
      <div id="footprintGrid" class="hw-grid"></div>
      <div class="section-label">Exact Launch Command(s)</div>
      <pre id="cmdBox" class="cmd"></pre>
      <div id="buildBoxWrap" style="display:none;">
        <div class="section-label">Build Command (ingine-serve binary not yet built for this target)</div>
        <pre id="buildBox" class="cmd"></pre>
      </div>
      <div id="downloadBoxWrap" style="display:none;">
        <div class="section-label">Hugging Face Download (requires your approval)</div>
        <pre id="downloadBox" class="cmd"></pre>
      </div>
      <div id="warningsBox"></div>

      <label class="consent-box">
        <input id="approveCheck" type="checkbox" style="margin-top:3px;" />
        <span>I have reviewed the flags, VRAM/RAM footprint, and any Hugging Face download above, and authorize Atlas Ingine to proceed.</span>
      </label>

      <div class="actions">
        <button id="downloadBtn" class="secondary" disabled>Download Weights (HF)</button>
        <button id="launchBtn" class="primary" disabled>Launch ingine-serve</button>
        <button id="stopBtn" class="danger">Stop Server</button>
      </div>
    </section>
  </div>

  <section class="card" id="modePanel">
    <h2 id="modePanelTitle">4. Localhost Multimodal Chat UI</h2>
    <div id="chatView">
      <div id="chatBox" class="chat-box"></div>
      <form id="chatForm" class="chat-form">
        <input id="chatInput" type="text" placeholder="Ask Inkling-2-Mini anything (or attach an image)..." />
        <input id="imgInput" type="file" accept="image/*" style="max-width:200px;color:var(--muted);font-size:12px;" />
        <button class="primary" type="submit">Send</button>
      </form>
      <div id="chatStats" style="font-size:11px;font-family:var(--mono);color:var(--muted);margin-top:6px;"></div>
    </div>
    <div id="apiView" style="display:none;">
      <div class="section-label">OpenAI-Compatible Endpoint</div>
      <pre id="apiSnippets" class="cmd"></pre>
    </div>
  </section>

  <section class="card">
    <h2>Live Status &amp; Launcher Logs</h2>
    <div id="logsBox" class="logs"></div>
  </section>
</main>

<script>
const state = { hw: null, plan: null, tier: null, array_count: 1, layout: 'auto', ctx: 65536, mode: 'chat', chatHistory: [] };

function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function api(path, body) {
  const opts = body !== undefined ? { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) } : {};
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
  return data;
}

function renderHw() {
  const hw = state.hw;
  const g = document.getElementById('hwGrid');
  g.innerHTML = '';
  const cards = [];
  if (hw.gpus.length) {
    hw.gpus.forEach(gpu => {
      cards.push({ k: `GPU ${gpu.index} (${gpu.arch} / ${gpu.generation})`, v: gpu.name, s: `${gpu.vram_free_gb}/${gpu.vram_total_gb} GB free | drv ${gpu.driver}` });
    });
  } else {
    cards.push({ k: 'GPU', v: 'None (CPU fallback)', s: hw.other_vendors.map(v => `${v.vendor}: ${v.status}`).join('; ') || 'No CUDA GPU found' });
  }
  cards.push({ k: 'Host RAM / CPU', v: `${hw.ram_free_gb} / ${hw.ram_total_gb} GB`, s: `${hw.cpu_threads} threads | disk free ${hw.disk_free_gb} GB` });
  cards.push({ k: 'Multi-GPU P2P', v: hw.iommu.p2p_friendly ? 'iommu=pt active' : 'Default IOMMU', s: hw.iommu.p2p_friendly ? 'PCIe P2P optimal' : 'Add iommu=pt for PCIe TP' });
  cards.push({ k: 'Local Checkpoints', v: `${hw.checkpoints.length} found`, s: hw.checkpoints.slice(0,2).map(c => `${c.name}${c.has_mtp ? ' +MTP' : ''}`).join(', ') || 'None yet' });
  cards.forEach(c => {
    const el = document.createElement('div');
    el.className = 'stat';
    el.innerHTML = `<div class="k">${esc(c.k)}</div><div class="v">${esc(c.v)}</div><div class="s">${esc(c.s)}</div>`;
    g.appendChild(el);
  });
}

function renderSelectors() {
  const plan = state.plan;
  const tiers = state.tiers;
  const tc = document.getElementById('tierCircles');
  tc.innerHTML = '';
  Object.values(tiers).forEach(t => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'circle-btn' + (state.tier === t.id ? ' active' : '');
    b.innerHTML = `<span class="big">${t.circle}</span><span class="sub">${t.label}</span>`;
    b.title = `${t.subtitle} (${t.repo_id})`;
    b.onclick = () => { state.tier = t.id; refreshPlan(); };
    tc.appendChild(b);
  });
  document.getElementById('tierNotes').textContent = `${plan.tier.subtitle} — ${plan.tier.notes}`;

  const ur = document.getElementById('upcomingRow');
  ur.innerHTML = state.upcoming.map(u => `<span class="tag">${u.label}: ${u.status}</span>`).join('');

  const ac = document.getElementById('arrayCircles');
  ac.innerHTML = '';
  plan.array_bounds.circles.forEach(n => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'circle-btn sm' + (plan.array_count === n ? ' active' : '');
    b.innerHTML = `<span class="big">${n}</span>`;
    b.onclick = () => { state.array_count = n; refreshPlan(); };
    ac.appendChild(b);
  });

  const lr = document.getElementById('layoutRow');
  lr.innerHTML = '';
  [['auto','Auto Layout'], ['batch','Continuous Batch'], ['dp','DP Worker Array'], ['tp','Tensor Parallel (TP)']].forEach(([id, label]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'seg-btn' + (state.layout === id ? ' active' : '');
    b.textContent = label;
    if (id === 'tp' && !plan.array_bounds.supports_tp.length) b.disabled = true;
    b.onclick = () => { state.layout = id; refreshPlan(); };
    lr.appendChild(b);
  });

  const cr = document.getElementById('ctxRow');
  cr.innerHTML = '';
  [[16384,'16K'], [65536,'64K'], [262144,'262K'], [1048576,'1M']].forEach(([val, label]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'seg-btn' + (plan.ctx === val ? ' active' : '');
    b.textContent = label + ' ctx';
    b.onclick = () => { state.ctx = val; refreshPlan(); };
    cr.appendChild(b);
  });

  const mr = document.getElementById('modeRow');
  mr.innerHTML = '';
  [['chat','Localhost Multimodal Chat UI'], ['api','OpenAI-Compliant API Endpoint']].forEach(([id, label]) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'seg-btn' + (state.mode === id ? ' active' : '');
    b.textContent = label;
    b.onclick = () => { state.mode = id; refreshPlan(); };
    mr.appendChild(b);
  });
}

function renderPlan() {
  const p = state.plan;
  const fg = document.getElementById('footprintGrid');
  fg.innerHTML = `
    <div class="stat"><div class="k">Est. VRAM / GPU</div><div class="v">${p.estimated_vram_per_gpu_gb} GB</div><div class="s">KV: ${p.kv_format.toUpperCase()}${p.kv_paged ? ' paged' : ''} (${p.kv_cache_mode})</div></div>
    <div class="stat"><div class="k">Host RAM / Download</div><div class="v">${p.estimated_host_ram_gb} GB RAM</div><div class="s">${p.needs_download ? p.download_gb + ' GB HF download' : 'Local checkpoint ready'}</div></div>
    <div class="stat"><div class="k">Layout / Features</div><div class="v">${p.layout} x${p.array_count}</div><div class="s">Vision: ${p.vision ? 'on' : 'off'} | DSpark: ${p.spec ? 'on' : 'off'}</div></div>
  `;
  document.getElementById('cmdBox').textContent = p.workers.map(w => w.shell).join('\n');
  document.getElementById('buildBoxWrap').style.display = p.binary_ready ? 'none' : 'block';
  document.getElementById('buildBox').textContent = p.build_command || '';
  document.getElementById('downloadBoxWrap').style.display = p.needs_download ? 'block' : 'none';
  document.getElementById('downloadBox').textContent = p.download_command || '';

  const wb = document.getElementById('warningsBox');
  wb.innerHTML = p.warnings.map(w => `<div class="warn">${esc(w)}</div>`).join('');

  syncButtons();
  renderMode();
}

function syncButtons() {
  const approved = document.getElementById('approveCheck').checked;
  const p = state.plan;
  document.getElementById('downloadBtn').disabled = !(approved && p && p.needs_download);
  document.getElementById('launchBtn').disabled = !(approved && p && p.hardware_ready && p.binary_ready && !p.needs_download);
}

function renderMode() {
  const p = state.plan;
  const isChat = state.mode === 'chat';
  document.getElementById('modePanelTitle').textContent = isChat ? '4. Localhost Multimodal Chat UI' : '4. OpenAI-Compliant API Endpoint';
  document.getElementById('chatView').style.display = isChat ? 'block' : 'none';
  document.getElementById('apiView').style.display = isChat ? 'none' : 'block';
  if (!isChat && p) {
    document.getElementById('apiSnippets').textContent =
`# Endpoint: ${p.endpoint_url}   Model: ${p.model_id}

# curl
curl ${p.endpoint_url}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"${p.model_id}","messages":[{"role":"user","content":"Hello from atlas-ingine!"}],"max_tokens":256}'

# Python (openai SDK)
from openai import OpenAI
client = OpenAI(base_url="${p.endpoint_url}", api_key="local")
resp = client.chat.completions.create(
    model="${p.model_id}",
    messages=[{"role": "user", "content": "Summarize our GPU setup."}],
)
print(resp.choices[0].message.content)`;
  }
}

let planRequest = 0;
let lastDownloadExit = null;
async function refreshPlan() {
  const request = ++planRequest;
  document.getElementById('approveCheck').checked = false;
  state.plan = null;
  syncButtons();
  const plan = await api('/api/plan', {
    tier: state.tier,
    ctx: state.ctx,
    array_count: state.array_count,
    layout: state.layout,
    mode: state.mode,
  });
  if (request !== planRequest) return;
  state.plan = plan;
  state.tier = state.plan.tier.id;
  state.ctx = state.plan.ctx;
  state.array_count = state.plan.array_count;
  renderSelectors();
  renderPlan();
}

async function pollStatus() {
  try {
    const s = await api('/api/status');
    if (s.download_exit_code !== lastDownloadExit) {
      lastDownloadExit = s.download_exit_code;
      if (lastDownloadExit === 0) await refreshPlan();
    }
    const dot = document.getElementById('statusDot');
    dot.className = 'dot ' + (s.state === 'ready' ? 'ready' : (s.state === 'idle' ? '' : 'busy'));
    document.getElementById('statusText').textContent = `${s.state.toUpperCase()} | ${s.endpoint_url}`;
    const lb = document.getElementById('logsBox');
    lb.textContent = (s.logs || []).join('\n') || 'Waiting for action...';
    lb.scrollTop = lb.scrollHeight;
  } catch (e) {}
}

document.getElementById('approveCheck').addEventListener('change', syncButtons);
document.getElementById('downloadBtn').addEventListener('click', async () => {
  try {
    await api('/api/download', {
      plan_id: state.plan.plan_id, approved: true, tier: state.tier, ctx: state.ctx, array_count: state.array_count, layout: state.layout, mode: state.mode,
    });
    pollStatus();
  } catch (e) { alert(e.message); await refreshPlan(); }
});
document.getElementById('launchBtn').addEventListener('click', async () => {
  try {
    await api('/api/launch', {
      plan_id: state.plan.plan_id, approved: true, tier: state.tier, ctx: state.ctx, array_count: state.array_count, layout: state.layout, mode: state.mode,
    });
    pollStatus();
  } catch (e) { alert(e.message); await refreshPlan(); }
});
document.getElementById('stopBtn').addEventListener('click', async () => {
  try { await api('/api/stop', {}); pollStatus(); }
  catch (e) { alert(e.message); }
});

document.getElementById('chatForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const input = document.getElementById('chatInput');
  const fileEl = document.getElementById('imgInput');
  const text = input.value.trim();
  if (!text && !fileEl.files.length) return;
  let dataUri = null;
  if (fileEl.files.length) {
    dataUri = await new Promise((res, rej) => {
      const r = new FileReader(); r.onload = () => res(r.result); r.onerror = rej; r.readAsDataURL(fileEl.files[0]);
    });
  }
  const content = dataUri ? [{type:'image_url', image_url:{url:dataUri}}, {type:'text', text: text || 'Describe this image.'}] : text;
  state.chatHistory.push({ role: 'user', content });
  const box = document.getElementById('chatBox');
  const udiv = document.createElement('div'); udiv.className = 'msg user';
  if (dataUri) { const im = document.createElement('img'); im.src = dataUri; udiv.appendChild(im); }
  udiv.appendChild(document.createTextNode(text || '[Image]'));
  box.appendChild(udiv);
  input.value = ''; fileEl.value = '';
  const adiv = document.createElement('div'); adiv.className = 'msg assistant'; adiv.textContent = '...';
  box.appendChild(adiv); box.scrollTop = box.scrollHeight;
  const t0 = performance.now();
  try {
    const resp = await api('/api/chat', {
      endpoint_url: state.plan.endpoint_url,
      model: state.plan.model_id,
      messages: state.chatHistory,
      max_tokens: 512,
    });
    const reply = resp.choices?.[0]?.message?.content || JSON.stringify(resp);
    adiv.textContent = reply;
    state.chatHistory.push({ role: 'assistant', content: reply });
    const dt = (performance.now() - t0) / 1000;
    const toks = resp.usage?.completion_tokens || 0;
    document.getElementById('chatStats').textContent = toks ? `${toks} tokens in ${dt.toFixed(2)}s (${(toks/Math.max(0.01,dt)).toFixed(1)} tok/s)` : `Completed in ${dt.toFixed(2)}s`;
  } catch (e) {
    adiv.textContent = 'Error: ' + e.message;
  }
  box.scrollTop = box.scrollHeight;
});

(async function init() {
  const initData = await api('/api/detect');
  state.hw = initData.hardware;
  state.tiers = initData.tiers;
  state.upcoming = initData.upcoming;
  state.tier = initData.defaults.tier;
  state.ctx = initData.defaults.ctx;
  state.array_count = initData.defaults.array_count;
  state.layout = initData.defaults.layout === 'single' ? 'auto' : initData.defaults.layout;
  state.mode = initData.defaults.mode;
  renderHw();
  await refreshPlan();
  await pollStatus();
  setInterval(pollStatus, 2500);
})();
</script>
</body>
</html>
"""


def make_http_handler(
    controller: AtlasSetupController, serve_host: str, serve_port: int, *,
    defaults: dict[str, Any] | None = None, model_dir: str | None = None, kv_cache_mode: str = "per_pass",
) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send_json(self, code: int, obj: Any) -> None:
            payload = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _local_request(self) -> bool:
            port = self.server.server_port
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if port == 80:
                hosts.update(("127.0.0.1", "localhost"))
            host = self.headers.get("Host", "").lower()
            origin = self.headers.get("Origin")
            if host not in hosts or (origin is not None and origin != f"http://{host}"):
                self._send_json(403, {"error": "Open the local setup page to use this endpoint."})
                return False
            return True

        def _read_json(self) -> dict[str, Any]:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected application/json")
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                raise ValueError("Expected a JSON object")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            return body

        def do_GET(self) -> None:
            if not self._local_request():
                return
            if self.path in ("/", "/index.html"):
                body = ATLAS_UI_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/detect":
                hw = controller.get_hardware()
                self._send_json(
                    200,
                    {
                        "hardware": hw,
                        "tiers": MODEL_TIERS,
                        "upcoming": UPCOMING_FAMILIES,
                        "defaults": {**recommend_defaults(hw), **(defaults or {})},
                    },
                )
                return
            if self.path == "/api/status":
                self._send_json(200, controller.status(serve_host, serve_port))
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if not self._local_request():
                return
            try:
                body = self._read_json()
            except Exception as exc:
                self._send_json(400, {"error": f"Invalid JSON: {exc}"})
                return

            def _server_plan(req_body: dict[str, Any]) -> dict[str, Any]:
                # Never trust client-supplied argv, repo_id, or arbitrary filesystem paths;
                # recompute from validated selectors and server-discovered checkpoints only.
                hw = controller.get_hardware()
                plan = build_plan(
                    hw,
                    tier_id=req_body.get("tier"),
                    ctx=req_body.get("ctx"),
                    array_count=req_body.get("array_count"),
                    layout=req_body.get("layout", "auto"),
                    mode=req_body.get("mode", "chat"),
                    kv_cache_mode=req_body.get("kv_cache_mode", kv_cache_mode),
                    model_dir=model_dir,
                    host=serve_host,
                    port=serve_port,
                    vision=req_body.get("vision"),
                    spec=req_body.get("spec"),
                )
                actions = (plan["tier"]["repo_id"], plan["model_dir"], plan["needs_download"], plan["download_files"], plan["hardware_ready"],
                           [(w["cuda_visible_devices"], w["argv"]) for w in plan["workers"]])
                plan["plan_id"] = hashlib.sha256(json.dumps(actions).encode()).hexdigest()
                return plan

            if self.path == "/api/plan":
                try:
                    plan = _server_plan(body)
                except (KeyError, TypeError, ValueError) as exc:
                    self._send_json(400, {"error": f"Invalid plan selection: {exc}"})
                    return
                self._send_json(200, plan)
                return

            if self.path in ("/api/download", "/api/launch"):
                if body.get("approved") is not True:
                    self._send_json(403, {"error": "Explicit approval is required."})
                    return
                try:
                    plan = _server_plan(body)
                    if "plan_id" in body and body["plan_id"] != plan["plan_id"]:
                        self._send_json(409, {"error": "The plan changed. Review it again before continuing."})
                        return
                    action = controller.start_download if self.path == "/api/download" else controller.launch_workers
                    self._send_json(200, action(plan, approved=True))
                except (OSError, RuntimeError, TypeError, ValueError) as exc:
                    self._send_json(400, {"error": str(exc)})
                return

            if self.path == "/api/stop":
                controller.stop_workers()
                self._send_json(200, {"ok": True})
                return

            if self.path == "/api/chat":
                endpoint = f"http://{serve_host}:{serve_port}/v1"
                if body.get("endpoint_url", endpoint) != endpoint:
                    self._send_json(400, {"error": "Choose the inference endpoint configured for this launcher."})
                    return
                conn = http.client.HTTPConnection(serve_host, serve_port, timeout=180)
                try:
                    data = json.dumps({
                        "model": body.get("model", "inkling-2-mini"),
                        "messages": body.get("messages", []),
                        "max_tokens": int(body.get("max_tokens", 512)),
                        "stream": False,
                    }).encode("utf-8")
                    conn.request("POST", "/v1/chat/completions", body=data, headers={"Content-Type": "application/json"})
                    response = conn.getresponse()
                    raw = response.read()
                    if 300 <= response.status < 400:
                        self._send_json(502, {"error": "The configured inference endpoint returned a redirect."})
                    else:
                        try:
                            result = json.loads(raw)
                        except (UnicodeError, ValueError):
                            result = {"error": raw.decode("utf-8", errors="replace")}
                        self._send_json(response.status, result)
                except (TypeError, ValueError) as exc:
                    self._send_json(400, {"error": str(exc)})
                except (OSError, http.client.HTTPException) as exc:
                    self._send_json(502, {"error": f"Server unreachable at {endpoint}: {exc}"})
                finally:
                    conn.close()
                return

            self._send_json(404, {"error": "Not found"})

    return Handler


def print_cli_plan(hw: dict[str, Any], plan: dict[str, Any]) -> None:
    print("Atlas Ingine — Hardware-Aware Auto-Tuner")
    print("=" * 60)
    if hw["gpus"]:
        for g in hw["gpus"]:
            print(
                f"  GPU {g['index']}: {g['name']} ({g['arch']} / {g['generation']}) "
                f"| {g['vram_free_gb']:.1f}/{g['vram_total_gb']:.1f} GB free | driver {g['driver']}"
            )
    else:
        print("  GPU: None detected (CPU --fake fallback)")
    print(
        f"  Host: {hw['cpu_threads']} threads | RAM {hw['ram_free_gb']:.1f}/{hw['ram_total_gb']:.1f} GB free "
        f"| Disk {hw['disk_free_gb']:.1f} GB free | iommu=pt={hw['iommu']['p2p_friendly']}"
    )
    print("-" * 60)
    print(f"  Tier:        {plan['tier']['label']} ({plan['tier']['subtitle']}) — {plan['tier']['repo_id']}")
    print(f"  Layout:      {plan['layout']} (array={plan['array_count']}, max={plan['array_bounds']['max_array']})")
    print(f"  Context:     {plan['ctx']:,} tokens | KV: {plan['kv_format'].upper()}{' paged' if plan['kv_paged'] else ''} ({plan['kv_cache_mode']})")
    print(f"  Features:    vision={'on' if plan['vision'] else 'off'} | DSpark={'on' if plan['spec'] else 'off'}")
    print(f"  Footprint:   ~{plan['estimated_vram_per_gpu_gb']} GB VRAM/GPU | ~{plan['estimated_host_ram_gb']} GB host RAM | download {plan['download_gb']} GB")
    print(f"  Model dir:   {plan['model_dir']} ({'present' if plan['checkpoint_present'] else 'needs download'})")
    print(f"  Binary:      {plan['binary_path']} ({'ready' if plan['binary_ready'] else 'needs build'})")
    if plan["build_command"]:
        print(f"  Build cmd:   {plan['build_command']}")
    if plan["download_command"]:
        print(f"  Download:    {plan['download_command']}")
    for w in plan["workers"]:
        print(f"  Worker {w['worker_index']}:    {w['shell']}")
    for warn in plan["warnings"]:
        print(f"  WARNING:     {warn}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Atlas Ingine hardware-aware self-setup & GPU auto-tuner.")
    ap.add_argument("--cli", action="store_true", help="Run in terminal CLI mode instead of localhost Web UI.")
    ap.add_argument("--plan", action="store_true", help="Print the computed plan and exit without launching.")
    ap.add_argument("--json", action="store_true", help="Emit hardware + plan as JSON (implies --plan).")
    ap.add_argument("--dry-run", action="store_true", help="Alias for --plan.")
    ap.add_argument("--yes", action="store_true", help="Explicitly approve download/launch in CLI mode without prompting.")
    ap.add_argument("--tier", choices=tuple(MODEL_TIERS), help="Model tier override.")
    ap.add_argument("--ctx", type=int, help="Context length override (e.g. 65536, 262144, 1048576).")
    ap.add_argument("--array", type=int, dest="array_count", help="Array count override (1..N).")
    ap.add_argument("--layout", choices=("auto", "single", "batch", "dp", "tp"), default="auto")
    ap.add_argument("--mode", choices=("chat", "api"), default="chat")
    ap.add_argument("--kv-cache-mode", choices=("per_pass", "last_full"), default="per_pass")
    ap.add_argument("--model-dir", help="Use an existing local checkpoint directory.")
    ap.add_argument("--host", default=DEFAULT_SERVE_HOST, help="ingine-serve bind host.")
    ap.add_argument("--port", type=int, default=DEFAULT_SERVE_PORT, help="ingine-serve base port.")
    ap.add_argument("--ui-host", choices=("127.0.0.1", "localhost"), default=DEFAULT_UI_HOST, help="Setup web UI bind host.")
    ap.add_argument("--ui-port", type=int, default=DEFAULT_UI_PORT, help="Setup web UI port.")
    ap.add_argument("--no-browser", action="store_true", help="Do not auto-open the system browser.")
    args = ap.parse_args(argv)

    hw = detect_hardware(ROOT)
    plan = build_plan(
        hw,
        tier_id=args.tier,
        ctx=args.ctx,
        array_count=args.array_count,
        layout=args.layout,
        mode=args.mode,
        kv_cache_mode=args.kv_cache_mode,
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
    )

    if args.json:
        print(json.dumps({"hardware": hw, "plan": plan}, indent=2))
        return 0
    if args.plan or args.dry_run:
        print_cli_plan(hw, plan)
        return 0

    controller = AtlasSetupController(ROOT)
    if args.cli:
        print_cli_plan(hw, plan)
        if not plan["binary_ready"]:
            print(f"\nBuild the engine first:\n  {plan['build_command']}", file=sys.stderr)
            return 1
        approved = args.yes
        if not approved:
            if not sys.stdin.isatty():
                print("\nNon-interactive terminal: pass --yes to approve download/launch, or --plan to inspect.", file=sys.stderr)
                return 2
            ans = input("\nApprove this plan and proceed? [y/N] ").strip().lower()
            approved = ans in ("y", "yes")
        if not approved:
            print("Cancelled; nothing downloaded or started.")
            return 0
        if plan["needs_download"]:
            controller.start_download(plan, approved=True)
            assert controller._download_proc is not None
            rc = controller._download_proc.wait()
            if rc != 0:
                print(f"Download exited with status {rc}", file=sys.stderr)
                return rc
            ok_ckpt, missing = verify_checkpoint_dir(
                Path(plan["model_dir"]),
                require_weights=(plan["layout"] != "cpu_fake"),
            )
            if not ok_ckpt:
                print(
                    f"Downloaded checkpoint at {plan['model_dir']} is incomplete; missing: {', '.join(missing)}",
                    file=sys.stderr,
                )
                return 1
            hw = detect_hardware(ROOT)
            plan = build_plan(
                hw,
                tier_id=args.tier or plan["tier"]["id"],
                ctx=args.ctx or plan["ctx"],
                array_count=args.array_count or plan["array_count"],
                layout=args.layout,
                mode=args.mode,
                kv_cache_mode=args.kv_cache_mode,
                host=args.host,
                port=args.port,
                model_dir=plan["model_dir"],
            )
        controller.launch_workers(plan, approved=True)
        print(f"\nServing on {plan['endpoint_url']} (Ctrl-C to stop)...")
        try:
            while True:
                time.sleep(1.0)
                st = controller.status(args.host, args.port)
                if not any(w["alive"] for w in st["workers"]):
                    break
        except KeyboardInterrupt:
            print("\nStopping workers...")
        finally:
            controller.stop_all()
        return 0

    defaults = {key: getattr(args, key) for key in ("tier", "ctx", "array_count", "layout", "mode")
                if getattr(args, key) is not None}
    handler = make_http_handler(controller, args.host, args.port, defaults=defaults,
                                model_dir=args.model_dir, kv_cache_mode=args.kv_cache_mode)
    httpd = http.server.ThreadingHTTPServer((args.ui_host, args.ui_port), handler)
    url = f"http://{args.ui_host}:{httpd.server_address[1]}/"
    print(f"Atlas Ingine Setup UI running at {url}  (Ctrl-C to stop)", flush=True)
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Atlas Ingine Setup UI...")
    finally:
        httpd.server_close()
        controller.stop_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

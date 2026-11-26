#!/usr/bin/env python3
"""Host/Array Per-Worker KV Cache-Mode Policy, Admission & Mode-Switch Gate (#829).

Implements the host/array control plane specified in #829 Requirement 3 and
`elastic_quant_eval/host_managed_rehearsal.md`:
  1. Worker capability discovery: reads `kv_cache_mode`, `supported_kv_cache_modes`,
     `kv_cache_mode_mutable`, `max_model_len`, and `max_batch` from `/health` and
     `/v1/models`, verifying consistency with `X-Ingine-Kv-Cache-Mode`.
  2. Per-worker admission & mode selection under explicit quality, context, memory,
     and deadline budgets. Default mode is ALWAYS `per_pass`. Opt-in `last_full` is
     selected only when `per_pass` cannot satisfy the memory/context budget (or is
     explicitly requested under budget pressure), `last_full` satisfies quality,
     context, memory, and deadline budgets, and the worker advertises `last_full`
     support (`B=1`, non-spec, `TP=1`).
  3. Fail-closed mode-switch contract (HTTP `/v1/chat/completions`): switching a
     session/task between `per_pass` and `last_full` NEVER reuses raw KV state across
     modes. Because HTTP `/v1/chat/completions` exposes no runtime checkpoint-restore
     operation or restore receipt (and `usage.cached_tokens` can reflect ordinary
     in-memory prefix hits), HTTP mode transitions require explicit `authorized_replay`
     with `bypass_prefix_cache=True`, a mode-scoped `cache_namespace`, replay latency
     charged to the deadline budget, and an observed integer `cached_tokens == 0` on
     the server response. Mode transitions are always bound to the registry-confirmed
     prior session mode even if a caller supplies stale mode-switch context.
  4. Persistent SQLite (WAL) dispatch/result audit log recording `kv_cache_mode`,
     worker capability view, budgets, mode-switch action, checkpoint/prefix identity,
     and server response mode verification.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

VALID_KV_CACHE_MODES = ("per_pass", "last_full")
DEFAULT_KV_CACHE_MODE = "per_pass"
DEFAULT_LOOP_PASSES = 3
MIN_LOOP_PASSES = 1
MAX_LOOP_PASSES = 8
VALID_MODEL_FAMILIES = ("inkling2", "qwen3_5")
DEFAULT_MODEL_FAMILY = "inkling2"

# Exact Qwen3.8-27B dense GDN + BF16 full-KV state geometry (mirrors native/src/backends/qwen38/qwen38_dense_gpu.cu):
#   - 48 linear (GDN) layers: conv_state [10240, 4] FP32 (163,840 B) + rec_state [48, 128, 128] FP32 (3,145,728 B)
#     = 3,309,568 B/layer -> 48 * 3,309,568 = 158,859,264 B (151.50 MiB fixed per session; zero sliding rings).
#   - 16 full-attention layers: K+V BF16 (num_kv_heads=4, head_dim=256 -> kv_stride=1024)
#     = 16 * 2 * 1024 * 2 = 65,536 B/tok (64 KiB/tok).
#   - Non-looped single-pass dense architecture: loop_passes == 1, kv_cache_mode == "per_pass" only.
QWEN38_NUM_LINEAR_LAYERS = 48
QWEN38_NUM_FULL_LAYERS = 16
QWEN38_GDN_FIXED_BYTES = QWEN38_NUM_LINEAR_LAYERS * (10240 * 4 + 48 * 128 * 128) * 4  # 158,859,264 B (151.50 MiB)
QWEN38_GDN_FIXED_STATE_MIB = QWEN38_GDN_FIXED_BYTES / (1024.0 * 1024.0)
QWEN38_FULL_KV_BYTES_PER_TOKEN: dict[str, int] = {
    "bf16": QWEN38_NUM_FULL_LAYERS * 2 * 1024 * 2,                      # 65,536 B/tok (64 KiB/tok)
    "gdn_f32_conv_bf16_kv_bf16": QWEN38_NUM_FULL_LAYERS * 2 * 1024 * 2, # 65,536 B/tok (64 KiB/tok)
}

# Exact Inkling-2-Mini KV/state geometry (mirrors native/src/model/model_data.h & state.cpp):
#   - VisitPlan(L): visits = 12 + 12*L, sliding = 10 + 10*L, full(per_pass) = 2 + 2*L, full(last_full, L=3) = 4.
#   - Full-attention KV bytes per (slot, token) across K+V (KV_DIM = 8*128 = 1024; Q4_0 row = 576 B):
#       q4_0: 2 * 576 = 1152 B/slot/tok (L=3 per_pass 9216 B/tok -> 9.00 GiB @ 1M; last_full 4608 -> 4.50 GiB)
#       fp8:  2 * 1024 * 1 = 2048 B/slot/tok
#       bf16: 2 * 1024 * 2 = 4096 B/slot/tok
#   - Pass-local state bytes for depth L:
#       sliding rings (k_ring + v_ring, 2 B/elem, WINDOW=1024, KV_DIM=1024): ns * 4,194,304 B (4 MiB / sliding visit)
#       ring_pos (int64, WINDOW=1024):                                        ns * 8,192 B
#       conv states (conv_k, conv_v [1024], conv_attn, conv_mlp [2304], 3*FP32): nv * 79,872 B
FULL_KV_BYTES_PER_SLOT_TOKEN: dict[str, int] = {
    "q4_0": 1152,
    "fp8": 2048,
    "bf16": 4096,
}
FULL_KV_BYTES_PER_TOKEN: dict[tuple[str, str], int] = {
    ("q4_0", "per_pass"): 9216,     # 9 KiB/tok -> 9.00 GiB at 1,048,576 tokens (L=3)
    ("q4_0", "last_full"): 4608,    # 4.5 KiB/tok -> 4.50 GiB at 1,048,576 tokens (L=3)
    ("fp8", "per_pass"): 16384,     # 16 KiB/tok -> 16.00 GiB at 1,048,576 tokens (L=3)
    ("fp8", "last_full"): 8192,     # 8 KiB/tok -> 8.00 GiB at 1,048,576 tokens (L=3)
    ("bf16", "per_pass"): 32768,    # 32 KiB/tok -> 32.00 GiB at 1,048,576 tokens (L=3)
    ("bf16", "last_full"): 16384,   # 16 KiB/tok -> 16.00 GiB at 1,048,576 tokens (L=3)
}
SLIDING_RING_BYTES_PER_VISIT = 2 * 1024 * 1024 * 2  # 4,194,304 B (4 MiB)
RING_POS_BYTES_PER_VISIT = 1024 * 8                 # 8,192 B
CONV_BYTES_PER_VISIT = 3 * (2 * 1024 + 2 * 2304) * 4  # 79,872 B


def validate_kv_cache_mode(mode: str) -> str:
    if mode not in VALID_KV_CACHE_MODES:
        raise ValueError(f"invalid kv_cache_mode {mode!r} (expected one of {VALID_KV_CACHE_MODES})")
    return mode


def validate_loop_passes(loop_passes: int) -> int:
    if type(loop_passes) is not int or loop_passes < MIN_LOOP_PASSES or loop_passes > MAX_LOOP_PASSES:
        raise ValueError(
            f"invalid loop_passes {loop_passes!r} (expected integer in {MIN_LOOP_PASSES}..{MAX_LOOP_PASSES})"
        )
    return loop_passes


def num_visits_for_passes(loop_passes: int = DEFAULT_LOOP_PASSES) -> int:
    lp = validate_loop_passes(loop_passes)
    return 12 + 12 * lp


def num_sliding_visits_for_passes(loop_passes: int = DEFAULT_LOOP_PASSES) -> int:
    lp = validate_loop_passes(loop_passes)
    return 10 + 10 * lp


def num_full_slots_for_passes(loop_passes: int, kv_cache_mode: str) -> int:
    lp = validate_loop_passes(loop_passes)
    validate_kv_cache_mode(kv_cache_mode)
    if kv_cache_mode == "last_full":
        if lp != DEFAULT_LOOP_PASSES:
            raise ValueError(f"last_full kv_cache_mode requires loop_passes={DEFAULT_LOOP_PASSES} (got {lp})")
        return 4
    return 2 + 2 * lp


def estimate_pass_local_state_mib(loop_passes: int = DEFAULT_LOOP_PASSES) -> float:
    ns = num_sliding_visits_for_passes(loop_passes)
    nv = num_visits_for_passes(loop_passes)
    total_bytes = ns * (SLIDING_RING_BYTES_PER_VISIT + RING_POS_BYTES_PER_VISIT) + nv * CONV_BYTES_PER_VISIT
    return total_bytes / (1024.0 * 1024.0)


FIXED_PASS_LOCAL_STATE_MIB = estimate_pass_local_state_mib(DEFAULT_LOOP_PASSES)


def estimate_full_kv_mib(
    context_tokens: int,
    kv_format: str,
    kv_cache_mode: str,
    loop_passes: int = DEFAULT_LOOP_PASSES,
) -> float:
    validate_kv_cache_mode(kv_cache_mode)
    lp = validate_loop_passes(loop_passes)
    fmt = kv_format.lower()
    if fmt not in FULL_KV_BYTES_PER_SLOT_TOKEN:
        raise ValueError(f"unsupported kv_format {kv_format!r} (expected q4_0, fp8, or bf16)")
    if context_tokens < 0:
        raise ValueError("context_tokens must be non-negative")
    slots = num_full_slots_for_passes(lp, kv_cache_mode)
    return (context_tokens * slots * FULL_KV_BYTES_PER_SLOT_TOKEN[fmt]) / (1024.0 * 1024.0)


def estimate_total_state_mib(
    context_tokens: int,
    kv_format: str,
    kv_cache_mode: str,
    loop_passes: int = DEFAULT_LOOP_PASSES,
) -> float:
    lp = validate_loop_passes(loop_passes)
    return estimate_pass_local_state_mib(lp) + estimate_full_kv_mib(context_tokens, kv_format, kv_cache_mode, lp)


def estimate_qwen38_state_mib(
    context_tokens: int,
    kv_format: str = "bf16",
    kv_cache_mode: str = "per_pass",
    loop_passes: int = 1,
) -> float:
    validate_kv_cache_mode(kv_cache_mode)
    if kv_cache_mode != "per_pass":
        raise ValueError(f"qwen3_5 does not support Inkling kv_cache_mode {kv_cache_mode!r} (only 'per_pass')")
    lp = validate_loop_passes(loop_passes)
    if lp != 1:
        raise ValueError(f"qwen3_5 is a non-looped architecture and requires loop_passes=1 (got {lp})")
    fmt = kv_format.lower()
    if fmt not in QWEN38_FULL_KV_BYTES_PER_TOKEN:
        raise ValueError(
            f"unsupported qwen3_5 kv_format {kv_format!r} (expected one of {tuple(QWEN38_FULL_KV_BYTES_PER_TOKEN)})"
        )
    if context_tokens < 0:
        raise ValueError("context_tokens must be non-negative")
    return QWEN38_GDN_FIXED_STATE_MIB + (context_tokens * QWEN38_FULL_KV_BYTES_PER_TOKEN[fmt]) / (1024.0 * 1024.0)


def infer_model_family(model_id: str, declared_family: Optional[str] = None) -> str:
    if declared_family is not None:
        if not isinstance(declared_family, str) or not declared_family.strip():
            raise ValueError(f"invalid model_family {declared_family!r}")
        return declared_family.strip().lower()
    if "qwen" in (model_id or "").lower():
        return "qwen3_5"
    return DEFAULT_MODEL_FAMILY


def estimate_worker_state_mib(worker: "WorkerCapability", context_tokens: int, kv_cache_mode: str) -> float:
    fam = worker.model_family
    if fam == "inkling2":
        if "qwen" in worker.model_id.lower():
            raise ValueError(
                f"worker {worker.worker_id} ({worker.model_id!r}) must not use Inkling-2 state geometry"
            )
        return estimate_total_state_mib(context_tokens, worker.kv_format, kv_cache_mode, loop_passes=worker.loop_passes)
    if fam == "qwen3_5":
        return estimate_qwen38_state_mib(
            context_tokens, worker.kv_format, kv_cache_mode, loop_passes=worker.loop_passes
        )
    raise ValueError(
        f"unsupported model_family {fam!r} for worker {worker.worker_id}: refusing to apply Inkling-2 state geometry"
    )


def mode_scoped_namespace(
    base_namespace: str,
    kv_cache_mode: str,
    loop_passes: int = DEFAULT_LOOP_PASSES,
) -> str:
    validate_kv_cache_mode(kv_cache_mode)
    lp = validate_loop_passes(loop_passes)
    base = (base_namespace or "default").strip()
    # Preserve opaque caller scopes (including literal `:loop_passes=<n>` suffixes) verbatim:
    # never strip a depth suffix from the caller input, and only match the legacy #829
    # `:kv_mode=<m>` suffix when applying the default L=3 scope.
    if lp == DEFAULT_LOOP_PASSES:
        marker = f":kv_mode={kv_cache_mode}"
        if base.endswith(marker):
            return base
        for m in VALID_KV_CACHE_MODES:
            suffix = f":kv_mode={m}"
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return f"{base}{marker}"
    return f"{base}:kv_mode={kv_cache_mode}:loop_passes={lp}"


@dataclass(frozen=True)
class CheckpointIdentity:
    """Explicit checkpoint / prefix identity binding mode, depth, model, format, namespace, and prefix hash."""

    model_id: str
    kv_format: str
    kv_cache_mode: str
    cache_namespace: str
    prefix_tokens: int
    prefix_sha256: str
    checkpoint_path: Optional[str] = None
    loop_passes: int = DEFAULT_LOOP_PASSES

    def __post_init__(self) -> None:
        validate_kv_cache_mode(self.kv_cache_mode)
        lp = validate_loop_passes(self.loop_passes)
        if self.kv_cache_mode == "last_full" and lp != DEFAULT_LOOP_PASSES:
            raise ValueError(f"last_full checkpoint identity requires loop_passes={DEFAULT_LOOP_PASSES} (got {lp})")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def identity_digest(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class WorkerCapability:
    """Advertised capability & active configuration of one inference worker."""

    worker_id: str
    base_url: str
    model_id: str
    active_kv_cache_mode: str = DEFAULT_KV_CACHE_MODE
    supported_kv_cache_modes: tuple[str, ...] = ("per_pass",)
    kv_cache_mode_mutable: bool = False
    max_model_len: int = 65536
    max_batch: int = 1
    spec_enabled: bool = False
    tp_size: int = 1
    kv_format: str = "q4_0"
    available_kv_vram_mib: float = 16384.0
    prefill_tok_per_s: float = 1200.0
    decode_tok_per_s: float = 40.0
    measured_last_full_nll_delta: float = 0.01056  # measured #829 reference delta unless overridden
    cache_namespace_header: Optional[str] = None
    loop_passes: int = DEFAULT_LOOP_PASSES
    model_family: str = DEFAULT_MODEL_FAMILY

    def __post_init__(self) -> None:
        validate_kv_cache_mode(self.active_kv_cache_mode)
        lp = validate_loop_passes(self.loop_passes)
        for m in self.supported_kv_cache_modes:
            validate_kv_cache_mode(m)
        if not isinstance(self.model_family, str) or not self.model_family.strip():
            raise ValueError(f"worker {self.worker_id} has invalid model_family {self.model_family!r}")
        if self.model_family == "inkling2" and "qwen" in self.model_id.lower():
            raise ValueError(
                f"worker {self.worker_id} ({self.model_id!r}) cannot use model_family='inkling2' "
                f"(refusing Inkling-only state geometry for Qwen; use model_family='qwen3_5')"
            )
        if self.model_family == "qwen3_5":
            if self.active_kv_cache_mode != "per_pass" or "last_full" in self.supported_kv_cache_modes:
                raise ValueError(
                    f"worker {self.worker_id} (qwen3_5) does not support Inkling last_full kv_cache_mode"
                )
            if lp != 1:
                raise ValueError(
                    f"worker {self.worker_id} (qwen3_5) is non-looped and requires loop_passes=1 (got {lp})"
                )
            if self.kv_format.lower() not in QWEN38_FULL_KV_BYTES_PER_TOKEN:
                raise ValueError(
                    f"worker {self.worker_id} (qwen3_5) does not support Inkling kv_format {self.kv_format!r} "
                    f"(expected one of {tuple(QWEN38_FULL_KV_BYTES_PER_TOKEN)})"
                )
        if self.active_kv_cache_mode == "last_full" and lp != DEFAULT_LOOP_PASSES:
            raise ValueError(f"worker {self.worker_id} cannot use active last_full with loop_passes={lp}")

    def supports_mode(self, mode: str) -> bool:
        validate_kv_cache_mode(mode)
        if mode == "last_full":
            if (
                self.model_family != "inkling2"
                or self.max_batch > 1
                or self.spec_enabled
                or self.tp_size > 1
                or self.loop_passes != DEFAULT_LOOP_PASSES
            ):
                return False
        return mode in self.supported_kv_cache_modes

    def can_serve_immediately(self, mode: str) -> bool:
        return self.supports_mode(mode) and (self.active_kv_cache_mode == mode or self.kv_cache_mode_mutable)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["supported_kv_cache_modes"] = list(self.supported_kv_cache_modes)
        return d


@dataclass(frozen=True)
class AdmissionBudgets:
    """Measured quality, context, memory, and deadline budgets governing mode selection."""

    max_nll_delta: float = 0.02
    require_exact_per_pass: bool = False
    memory_budget_mib: Optional[float] = None
    deadline_seconds: Optional[float] = None
    vram_headroom_mib: float = 512.0


@dataclass(frozen=True)
class ModeSwitchContext:
    """Prior session state & authorized transition proof when changing cache modes or loop depth."""

    prior_kv_cache_mode: str
    compatible_checkpoint: Optional[CheckpointIdentity] = None
    replay_authorized: bool = False
    authorized_replay_tokens: int = 0
    prior_loop_passes: int = DEFAULT_LOOP_PASSES

    def __post_init__(self) -> None:
        validate_kv_cache_mode(self.prior_kv_cache_mode)
        validate_loop_passes(self.prior_loop_passes)


@dataclass(frozen=True)
class AdmissionRequest:
    """Host admission request for a worker turn or task."""

    task_id: str
    session_id: str
    cache_namespace: str
    context_tokens: int
    expected_new_tokens: int
    prefix_sha256: str
    budgets: AdmissionBudgets = field(default_factory=AdmissionBudgets)
    preferred_mode: Optional[str] = None
    mode_switch: Optional[ModeSwitchContext] = None

    def __post_init__(self) -> None:
        if self.context_tokens < 0:
            raise ValueError(f"context_tokens must be non-negative (got {self.context_tokens})")
        if self.expected_new_tokens <= 0:
            raise ValueError(f"expected_new_tokens must be positive (got {self.expected_new_tokens})")
        if self.preferred_mode is not None:
            validate_kv_cache_mode(self.preferred_mode)


@dataclass(frozen=True)
class AdmissionDecision:
    """Structured admission & mode-selection outcome."""

    admitted: bool
    worker_id: Optional[str]
    selected_mode: Optional[str]
    reason: str
    needs_worker_relaunch: bool = False
    mode_switch_action: str = "none"  # none | compatible_checkpoint | authorized_replay
    bypass_prefix_cache: bool = False
    effective_cache_namespace: Optional[str] = None
    checkpoint_identity: Optional[CheckpointIdentity] = None
    estimated_state_mib: Optional[float] = None
    estimated_latency_seconds: Optional[float] = None
    worker_supported_modes: tuple[str, ...] = ()
    worker_loop_passes: int = DEFAULT_LOOP_PASSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "worker_id": self.worker_id,
            "selected_mode": self.selected_mode,
            "reason": self.reason,
            "needs_worker_relaunch": self.needs_worker_relaunch,
            "mode_switch_action": self.mode_switch_action,
            "bypass_prefix_cache": self.bypass_prefix_cache,
            "effective_cache_namespace": self.effective_cache_namespace,
            "checkpoint_identity": self.checkpoint_identity.to_dict() if self.checkpoint_identity else None,
            "checkpoint_identity_digest": (
                self.checkpoint_identity.identity_digest() if self.checkpoint_identity else None
            ),
            "estimated_state_mib": (
                round(self.estimated_state_mib, 6) if self.estimated_state_mib is not None else None
            ),
            "estimated_latency_seconds": (
                round(self.estimated_latency_seconds, 4) if self.estimated_latency_seconds is not None else None
            ),
            "worker_supported_modes": list(self.worker_supported_modes),
            "worker_loop_passes": self.worker_loop_passes,
        }


def _http_json_with_headers(
    method: str,
    url: str,
    payload: Optional[dict[str, Any]] = None,
    extra_headers: Optional[dict[str, str]] = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        return int(resp.status), resp_headers, json.loads(raw)


def discover_worker_capability(
    worker_id: str,
    base_url: str,
    kv_format: Optional[str] = None,
    available_kv_vram_mib: float = 16384.0,
    spec_enabled: bool = False,
    tp_size: int = 1,
    prefill_tok_per_s: float = 1200.0,
    decode_tok_per_s: float = 40.0,
    measured_last_full_nll_delta: float = 0.01056,
    cache_namespace_header: Optional[str] = None,
    timeout: float = 15.0,
) -> WorkerCapability:
    """Query GET /health and GET /v1/models and build a verified WorkerCapability."""
    url = base_url.rstrip("/")
    _, h_headers, health = _http_json_with_headers("GET", f"{url}/health", timeout=timeout)
    _, m_headers, models = _http_json_with_headers("GET", f"{url}/v1/models", timeout=timeout)

    if health.get("status") != "ok":
        raise RuntimeError(f"worker {worker_id} unhealthy: {health!r}")
    data = models.get("data") or []
    if not data:
        raise RuntimeError(f"worker {worker_id} /v1/models returned empty model list")
    card = data[0]

    active_health = validate_kv_cache_mode(health.get("kv_cache_mode", DEFAULT_KV_CACHE_MODE))
    active_model = validate_kv_cache_mode(card.get("kv_cache_mode", active_health))
    if active_health != active_model:
        raise RuntimeError(
            f"worker {worker_id} inconsistent kv_cache_mode: /health={active_health!r}, /v1/models={active_model!r}"
        )
    hdr_mode = h_headers.get("x-ingine-kv-cache-mode") or m_headers.get("x-ingine-kv-cache-mode")
    if hdr_mode is not None and hdr_mode != active_health:
        raise RuntimeError(
            f"worker {worker_id} X-Ingine-Kv-Cache-Mode header={hdr_mode!r} != body={active_health!r}"
        )

    def _parse_supported_list(src_name: str, raw_val: Any) -> tuple[str, ...]:
        if not isinstance(raw_val, list) or not raw_val:
            raise RuntimeError(
                f"worker {worker_id} invalid {src_name} supported_kv_cache_modes: {raw_val!r}"
            )
        return tuple(validate_kv_cache_mode(m) for m in raw_val)

    has_health_sup = "supported_kv_cache_modes" in health
    has_card_sup = "supported_kv_cache_modes" in card
    if has_health_sup and has_card_sup:
        sup_health = _parse_supported_list("/health", health["supported_kv_cache_modes"])
        sup_card = _parse_supported_list("/v1/models", card["supported_kv_cache_modes"])
        if set(sup_health) != set(sup_card):
            raise RuntimeError(
                f"worker {worker_id} inconsistent supported_kv_cache_modes: "
                f"/health={sup_health!r}, /v1/models={sup_card!r}"
            )
        supported = sup_health
    elif has_health_sup:
        supported = _parse_supported_list("/health", health["supported_kv_cache_modes"])
    elif has_card_sup:
        supported = _parse_supported_list("/v1/models", card["supported_kv_cache_modes"])
    else:
        supported = (active_health,)

    if active_health not in supported:
        raise RuntimeError(
            f"worker {worker_id} active kv_cache_mode {active_health!r} absent from supported {supported!r}"
        )
    mutable = bool(health.get("kv_cache_mode_mutable", card.get("kv_cache_mode_mutable", False)))
    max_batch = int(health.get("max_batch", 1))
    max_model_len = int(card.get("max_model_len", 65536))
    model_id = str(card.get("id") or health.get("model") or "unknown")

    has_health_fam = "model_family" in health
    has_card_fam = "model_family" in card
    if has_health_fam and has_card_fam:
        fam_health = infer_model_family(model_id, health["model_family"])
        fam_card = infer_model_family(model_id, card["model_family"])
        if fam_health != fam_card:
            raise RuntimeError(
                f"worker {worker_id} inconsistent model_family: /health={fam_health!r}, /v1/models={fam_card!r}"
            )
        model_family = fam_health
    elif has_health_fam:
        model_family = infer_model_family(model_id, health["model_family"])
    elif has_card_fam:
        model_family = infer_model_family(model_id, card["model_family"])
    else:
        model_family = infer_model_family(model_id, None)

    if model_family not in VALID_MODEL_FAMILIES:
        raise RuntimeError(
            f"worker {worker_id} advertises unsupported model_family {model_family!r}: "
            f"refusing to apply Inkling-2 state geometry"
        )
    if model_family == "inkling2" and "qwen" in model_id.lower():
        raise RuntimeError(
            f"worker {worker_id} ({model_id!r}) advertises Inkling-2 model_family for a Qwen model"
        )

    # Reserve L=3 default strictly for genuinely absent legacy Inkling fields; non-looped qwen3_5 defaults to 1.
    has_health_lp = "loop_passes" in health
    has_card_lp = "loop_passes" in card
    if has_health_lp and has_card_lp:
        lp_health = validate_loop_passes(health["loop_passes"])
        lp_card = validate_loop_passes(card["loop_passes"])
        if lp_health != lp_card:
            raise RuntimeError(
                f"worker {worker_id} inconsistent loop_passes: /health={lp_health!r}, /v1/models={lp_card!r}"
            )
        loop_passes = lp_health
    elif has_health_lp:
        loop_passes = validate_loop_passes(health["loop_passes"])
    elif has_card_lp:
        loop_passes = validate_loop_passes(card["loop_passes"])
    else:
        loop_passes = 1 if model_family == "qwen3_5" else DEFAULT_LOOP_PASSES

    for hdr_src, hdrs in (("/health", h_headers), ("/v1/models", m_headers)):
        hdr_lp_raw = hdrs.get("x-ingine-loop-passes")
        if hdr_lp_raw is not None:
            try:
                hdr_lp = validate_loop_passes(int(hdr_lp_raw))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"worker {worker_id} invalid {hdr_src} X-Ingine-Loop-Passes header={hdr_lp_raw!r}"
                ) from exc
            if hdr_lp != loop_passes:
                raise RuntimeError(
                    f"worker {worker_id} {hdr_src} X-Ingine-Loop-Passes header={hdr_lp!r} != body={loop_passes!r}"
                )

    if model_family == "qwen3_5":
        if loop_passes != 1:
            raise RuntimeError(
                f"worker {worker_id} (qwen3_5) advertises unsupported Inkling loop_passes={loop_passes} (expected 1)"
            )
        if active_health != "per_pass" or "last_full" in supported:
            raise RuntimeError(
                f"worker {worker_id} (qwen3_5) advertises unsupported last_full kv_cache_mode: "
                f"active={active_health!r}, supported={supported!r}"
            )
        resolved_kv_format = kv_format if kv_format is not None else "bf16"
    else:
        if loop_passes != DEFAULT_LOOP_PASSES and (active_health == "last_full" or "last_full" in supported):
            raise RuntimeError(
                f"worker {worker_id} advertises incompatible last_full with loop_passes={loop_passes}: "
                f"active={active_health!r}, supported={supported!r}"
            )
        resolved_kv_format = kv_format if kv_format is not None else "q4_0"

    return WorkerCapability(
        worker_id=worker_id,
        base_url=url,
        model_id=model_id,
        active_kv_cache_mode=active_health,
        supported_kv_cache_modes=supported,
        kv_cache_mode_mutable=mutable,
        max_model_len=max_model_len,
        max_batch=max_batch,
        loop_passes=loop_passes,
        spec_enabled=spec_enabled,
        tp_size=tp_size,
        kv_format=resolved_kv_format,
        available_kv_vram_mib=available_kv_vram_mib,
        prefill_tok_per_s=prefill_tok_per_s,
        decode_tok_per_s=decode_tok_per_s,
        measured_last_full_nll_delta=measured_last_full_nll_delta,
        cache_namespace_header=cache_namespace_header,
        model_family=model_family,
    )


def _validate_mode_switch(
    worker: WorkerCapability,
    req: AdmissionRequest,
    target_mode: str,
) -> tuple[bool, str, str, bool, int, Optional[str]]:
    """Enforce fail-closed mode-switch and loop-depth transition rules.

    Returns: (ok, reason, action, bypass_prefix_cache, extra_replay_tokens, ckpt_path)
    """
    ms = req.mode_switch
    if ms is None or (ms.prior_kv_cache_mode == target_mode and ms.prior_loop_passes == worker.loop_passes):
        return True, "same_or_fresh_mode", "none", False, 0, None

    # Mode or loop_passes change requested: raw cross-mode / cross-depth KV reuse is strictly forbidden.
    # HTTP /v1/chat/completions has no runtime checkpoint-restore operation or restore receipt
    # (and `usage.cached_tokens` can come from ordinary in-memory prefix hits), so HTTP transitions
    # strictly require explicit authorized replay until a verified runtime restore receipt exists.
    if ms.replay_authorized and ms.authorized_replay_tokens > 0:
        return (
            True,
            "mode_switch_via_authorized_replay",
            "authorized_replay",
            True,
            ms.authorized_replay_tokens,
            None,
        )

    if ms.compatible_checkpoint is not None:
        return (
            False,
            "mode_switch_checkpoint_restore_unavailable_over_http_require_authorized_replay",
            "none",
            True,
            0,
            None,
        )

    return (
        False,
        (
            f"mode_switch_blocked_no_authorized_replay:"
            f"{ms.prior_kv_cache_mode}(L={ms.prior_loop_passes})->{target_mode}(L={worker.loop_passes})"
        ),
        "none",
        True,
        0,
        None,
    )


def evaluate_worker_for_mode(
    worker: WorkerCapability,
    req: AdmissionRequest,
    mode: str,
) -> AdmissionDecision:
    """Evaluate whether `worker` can admit `req` under `mode` and all budgets."""
    validate_kv_cache_mode(mode)
    total_tokens = req.context_tokens + req.expected_new_tokens
    if total_tokens > worker.max_model_len:
        return AdmissionDecision(
            admitted=False,
            worker_id=worker.worker_id,
            selected_mode=None,
            reason=f"context_exceeds_max_model_len:{total_tokens}>{worker.max_model_len}",
            worker_supported_modes=worker.supported_kv_cache_modes,
        )

    if not worker.supports_mode(mode):
        return AdmissionDecision(
            admitted=False,
            worker_id=worker.worker_id,
            selected_mode=None,
            reason=f"worker_does_not_support_mode:{mode}",
            worker_supported_modes=worker.supported_kv_cache_modes,
        )

    if mode == "last_full":
        if req.budgets.require_exact_per_pass:
            return AdmissionDecision(
                admitted=False,
                worker_id=worker.worker_id,
                selected_mode=None,
                reason="quality_budget_requires_exact_per_pass",
                worker_supported_modes=worker.supported_kv_cache_modes,
            )
        if worker.measured_last_full_nll_delta > req.budgets.max_nll_delta:
            return AdmissionDecision(
                admitted=False,
                worker_id=worker.worker_id,
                selected_mode=None,
                reason=(
                    f"quality_budget_exceeded:measured_nll_delta={worker.measured_last_full_nll_delta}"
                    f">budget={req.budgets.max_nll_delta}"
                ),
                worker_supported_modes=worker.supported_kv_cache_modes,
            )

    if worker.model_family not in VALID_MODEL_FAMILIES:
        return AdmissionDecision(
            admitted=False,
            worker_id=worker.worker_id,
            selected_mode=None,
            reason=f"unsupported_model_family:{worker.model_family}",
            worker_supported_modes=worker.supported_kv_cache_modes,
            worker_loop_passes=worker.loop_passes,
        )

    state_mib = estimate_worker_state_mib(worker, total_tokens, mode)
    eff_mem_limit = worker.available_kv_vram_mib - req.budgets.vram_headroom_mib
    if req.budgets.memory_budget_mib is not None:
        eff_mem_limit = min(eff_mem_limit, req.budgets.memory_budget_mib)
    if state_mib > eff_mem_limit:
        return AdmissionDecision(
            admitted=False,
            worker_id=worker.worker_id,
            selected_mode=None,
            reason=f"memory_budget_exceeded:{state_mib:.3f}MiB>{eff_mem_limit:.3f}MiB",
            estimated_state_mib=state_mib,
            worker_supported_modes=worker.supported_kv_cache_modes,
            worker_loop_passes=worker.loop_passes,
        )

    sw_ok, sw_reason, sw_action, bypass_cache, replay_tokens, ckpt_path = _validate_mode_switch(worker, req, mode)
    if not sw_ok:
        return AdmissionDecision(
            admitted=False,
            worker_id=worker.worker_id,
            selected_mode=None,
            reason=sw_reason,
            estimated_state_mib=state_mib,
            worker_supported_modes=worker.supported_kv_cache_modes,
            worker_loop_passes=worker.loop_passes,
        )

    prefill_tokens = max(req.context_tokens, replay_tokens) if sw_action == "authorized_replay" else req.context_tokens
    est_latency = (
        prefill_tokens / max(worker.prefill_tok_per_s, 1e-6)
        + req.expected_new_tokens / max(worker.decode_tok_per_s, 1e-6)
    )
    if req.budgets.deadline_seconds is not None and est_latency > req.budgets.deadline_seconds:
        return AdmissionDecision(
            admitted=False,
            worker_id=worker.worker_id,
            selected_mode=None,
            reason=f"deadline_budget_exceeded:{est_latency:.3f}s>{req.budgets.deadline_seconds:.3f}s",
            mode_switch_action=sw_action,
            estimated_state_mib=state_mib,
            estimated_latency_seconds=est_latency,
            worker_supported_modes=worker.supported_kv_cache_modes,
            worker_loop_passes=worker.loop_passes,
        )

    eff_ns = mode_scoped_namespace(req.cache_namespace, mode, loop_passes=worker.loop_passes)
    ckpt_id = CheckpointIdentity(
        model_id=worker.model_id,
        kv_format=worker.kv_format,
        kv_cache_mode=mode,
        cache_namespace=eff_ns,
        prefix_tokens=req.context_tokens,
        prefix_sha256=req.prefix_sha256,
        checkpoint_path=ckpt_path,
        loop_passes=worker.loop_passes,
    )

    if not worker.can_serve_immediately(mode):
        return AdmissionDecision(
            admitted=False,
            worker_id=worker.worker_id,
            selected_mode=mode,
            reason=f"worker_active_mode_mismatch_relaunch_required:active={worker.active_kv_cache_mode},want={mode}",
            needs_worker_relaunch=True,
            mode_switch_action=sw_action,
            bypass_prefix_cache=bypass_cache,
            effective_cache_namespace=eff_ns,
            checkpoint_identity=ckpt_id,
            estimated_state_mib=state_mib,
            estimated_latency_seconds=est_latency,
            worker_supported_modes=worker.supported_kv_cache_modes,
            worker_loop_passes=worker.loop_passes,
        )

    reason = (
        f"admitted:{mode}:{sw_reason}"
        if sw_action != "none"
        else f"admitted:{mode}:within_budgets"
    )
    return AdmissionDecision(
        admitted=True,
        worker_id=worker.worker_id,
        selected_mode=mode,
        reason=reason,
        needs_worker_relaunch=False,
        mode_switch_action=sw_action,
        bypass_prefix_cache=bypass_cache,
        effective_cache_namespace=eff_ns,
        checkpoint_identity=ckpt_id,
        estimated_state_mib=state_mib,
        estimated_latency_seconds=est_latency,
        worker_supported_modes=worker.supported_kv_cache_modes,
        worker_loop_passes=worker.loop_passes,
    )


def select_worker_and_mode(
    workers: Iterable[WorkerCapability],
    req: AdmissionRequest,
) -> AdmissionDecision:
    """Select (worker, kv_cache_mode) at admission across an array of workers.

    Policy order:
      - Default ALWAYS tries `per_pass` first (unless `preferred_mode == "last_full"`
        is explicitly requested, in which case `last_full` is tried first with
        fallback to `per_pass` if `last_full` violates quality/support constraints).
      - Within a candidate mode, immediately-servable workers (active mode matches)
        are preferred over workers requiring relaunch.
    """
    worker_list = list(workers)
    if not worker_list:
        return AdmissionDecision(admitted=False, worker_id=None, selected_mode=None, reason="no_workers_available")

    mode_order = ("last_full", "per_pass") if req.preferred_mode == "last_full" else ("per_pass", "last_full")
    relaunch_candidates: list[AdmissionDecision] = []
    rejection_reasons: list[str] = []

    for mode in mode_order:
        immediate_hits: list[AdmissionDecision] = []
        for w in worker_list:
            dec = evaluate_worker_for_mode(w, req, mode)
            if dec.admitted:
                immediate_hits.append(dec)
            elif dec.needs_worker_relaunch:
                relaunch_candidates.append(dec)
            else:
                rejection_reasons.append(f"{w.worker_id}[{mode}]:{dec.reason}")
        if immediate_hits:
            # Prefer lowest estimated latency, then deterministic worker_id tie-break.
            immediate_hits.sort(key=lambda d: (d.estimated_latency_seconds or 0.0, d.worker_id or ""))
            return immediate_hits[0]

    if relaunch_candidates:
        return relaunch_candidates[0]

    return AdmissionDecision(
        admitted=False,
        worker_id=None,
        selected_mode=None,
        reason="all_workers_rejected:" + ";".join(rejection_reasons[:6]),
    )


class HostCacheModeRegistry:
    """SQLite (WAL) host authority recording dispatches, results, and session mode state."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS dispatches (
                dispatch_id TEXT PRIMARY KEY,
                created_at REAL NOT NULL,
                worker_id TEXT,
                session_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                admitted INTEGER NOT NULL,
                kv_cache_mode TEXT,
                worker_supported_modes_json TEXT NOT NULL,
                reason TEXT NOT NULL,
                mode_switch_action TEXT NOT NULL,
                bypass_prefix_cache INTEGER NOT NULL,
                effective_cache_namespace TEXT,
                checkpoint_identity_json TEXT,
                checkpoint_identity_digest TEXT,
                max_tokens INTEGER NOT NULL,
                estimated_state_mib REAL,
                estimated_latency_seconds REAL
            );
            CREATE TABLE IF NOT EXISTS results (
                dispatch_id TEXT PRIMARY KEY,
                completed_at REAL NOT NULL,
                status TEXT NOT NULL,
                admitted_kv_cache_mode TEXT NOT NULL,
                server_kv_cache_mode TEXT,
                verified_mode_match INTEGER NOT NULL,
                prompt_tokens INTEGER,
                cached_tokens INTEGER,
                completion_tokens INTEGER,
                elapsed_seconds REAL,
                error_detail TEXT,
                FOREIGN KEY (dispatch_id) REFERENCES dispatches(dispatch_id)
            );
            CREATE TABLE IF NOT EXISTS session_modes (
                session_id TEXT PRIMARY KEY,
                kv_cache_mode TEXT NOT NULL,
                loop_passes INTEGER NOT NULL DEFAULT 3,
                checkpoint_identity_digest TEXT,
                updated_at REAL NOT NULL
            );
            """
        )
        # Backwards-compatible migration for pre-existing #829 SQLite registries lacking loop_passes.
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(session_modes)").fetchall()}
        if "loop_passes" not in cols:
            self.conn.execute("ALTER TABLE session_modes ADD COLUMN loop_passes INTEGER NOT NULL DEFAULT 3")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "HostCacheModeRegistry":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def get_session_mode(self, session_id: str) -> Optional[str]:
        row = self.conn.execute(
            "SELECT kv_cache_mode FROM session_modes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return str(row["kv_cache_mode"]) if row else None

    def get_session_state(self, session_id: str) -> Optional[tuple[str, int]]:
        row = self.conn.execute(
            "SELECT kv_cache_mode, loop_passes FROM session_modes WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return (str(row["kv_cache_mode"]), int(row["loop_passes"])) if row else None

    def record_dispatch(self, req: AdmissionRequest, decision: AdmissionDecision) -> str:
        dispatch_id = f"disp-{uuid.uuid4().hex[:12]}"
        ckpt_json = (
            json.dumps(decision.checkpoint_identity.to_dict(), sort_keys=True)
            if decision.checkpoint_identity
            else None
        )
        ckpt_digest = decision.checkpoint_identity.identity_digest() if decision.checkpoint_identity else None
        self.conn.execute(
            """
            INSERT INTO dispatches (
                dispatch_id, created_at, worker_id, session_id, task_id, admitted,
                kv_cache_mode, worker_supported_modes_json, reason, mode_switch_action,
                bypass_prefix_cache, effective_cache_namespace, checkpoint_identity_json,
                checkpoint_identity_digest, max_tokens, estimated_state_mib, estimated_latency_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dispatch_id,
                time.time(),
                decision.worker_id,
                req.session_id,
                req.task_id,
                1 if decision.admitted else 0,
                decision.selected_mode,
                json.dumps(list(decision.worker_supported_modes)),
                decision.reason,
                decision.mode_switch_action,
                1 if decision.bypass_prefix_cache else 0,
                decision.effective_cache_namespace,
                ckpt_json,
                ckpt_digest,
                req.expected_new_tokens,
                decision.estimated_state_mib,
                decision.estimated_latency_seconds,
            ),
        )
        self.conn.commit()
        return dispatch_id

    def record_result(
        self,
        dispatch_id: str,
        session_id: str,
        admitted_mode: str,
        server_mode: Optional[str],
        status: str,
        prompt_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        elapsed_seconds: Optional[float] = None,
        checkpoint_identity_digest: Optional[str] = None,
        error_detail: Optional[str] = None,
        loop_passes: int = DEFAULT_LOOP_PASSES,
    ) -> None:
        verified = 1 if (server_mode == admitted_mode and status == "ok") else 0
        now = time.time()
        self.conn.execute(
            """
            INSERT INTO results (
                dispatch_id, completed_at, status, admitted_kv_cache_mode,
                server_kv_cache_mode, verified_mode_match, prompt_tokens,
                cached_tokens, completion_tokens, elapsed_seconds, error_detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dispatch_id,
                now,
                status,
                admitted_mode,
                server_mode,
                verified,
                prompt_tokens,
                cached_tokens,
                completion_tokens,
                elapsed_seconds,
                error_detail,
            ),
        )
        if verified:
            self.conn.execute(
                """
                INSERT INTO session_modes (session_id, kv_cache_mode, loop_passes, checkpoint_identity_digest, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    kv_cache_mode = excluded.kv_cache_mode,
                    loop_passes = excluded.loop_passes,
                    checkpoint_identity_digest = excluded.checkpoint_identity_digest,
                    updated_at = excluded.updated_at
                """,
                (session_id, admitted_mode, validate_loop_passes(loop_passes), checkpoint_identity_digest, now),
            )
        self.conn.commit()

    def dispatch_chat_completion(
        self,
        workers: Iterable[WorkerCapability],
        req: AdmissionRequest,
        messages: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Admit `req`, enforce session mode-switch contract, dispatch HTTP request, and audit.

        Generation limit is bound end-to-end to `req.expected_new_tokens` (the exact budget
        validated by admission). Passing a conflicting `max_tokens` fails closed with ValueError.
        """
        if max_tokens is not None and max_tokens != req.expected_new_tokens:
            raise ValueError(
                f"max_tokens mismatch: explicit max_tokens={max_tokens} != "
                f"admitted req.expected_new_tokens={req.expected_new_tokens}"
            )
        bound_max_tokens = req.expected_new_tokens

        worker_map = {w.worker_id: w for w in workers}
        prior_state = self.get_session_state(req.session_id)
        eff_req = req
        if prior_state is not None:
            prior_mode, prior_lp = prior_state
            # Always bind mode/depth transitions to the registry-confirmed prior session state,
            # even if the caller supplied stale ModeSwitchContext.
            if req.mode_switch is None:
                eff_req = dataclasses.replace(
                    req,
                    mode_switch=ModeSwitchContext(prior_kv_cache_mode=prior_mode, prior_loop_passes=prior_lp),
                )
            elif (
                req.mode_switch.prior_kv_cache_mode != prior_mode
                or req.mode_switch.prior_loop_passes != prior_lp
            ):
                stale_dec = AdmissionDecision(
                    admitted=False,
                    worker_id=None,
                    selected_mode=None,
                    reason=(
                        f"stale_session_mode_context:caller_prior={req.mode_switch.prior_kv_cache_mode}"
                        f"(L={req.mode_switch.prior_loop_passes})!=registry_prior={prior_mode}(L={prior_lp})"
                    ),
                )
                dispatch_id = self.record_dispatch(eff_req, stale_dec)
                return {
                    "dispatch_id": dispatch_id,
                    "admitted": False,
                    "decision": stale_dec.to_dict(),
                    "response": None,
                }

        decision = select_worker_and_mode(worker_map.values(), eff_req)
        dispatch_id = self.record_dispatch(eff_req, decision)
        if not decision.admitted or decision.worker_id is None or decision.selected_mode is None:
            return {
                "dispatch_id": dispatch_id,
                "admitted": False,
                "decision": decision.to_dict(),
                "response": None,
            }

        worker = worker_map[decision.worker_id]
        extra_headers: dict[str, str] = {
            # Pre-generation depth precondition: #838 servers reject mismatched depths with HTTP 409
            # before generating a token; legacy L=3 servers ignore unknown request headers.
            "X-Ingine-Expected-Loop-Passes": str(worker.loop_passes),
        }
        if worker.cache_namespace_header and decision.effective_cache_namespace:
            if not decision.bypass_prefix_cache:
                extra_headers[worker.cache_namespace_header] = decision.effective_cache_namespace
            # When bypass_prefix_cache=True with cache_namespace_header configured, omitting the
            # header triggers ChatServer's #827 fail-closed bypass_prefix_cache=true.

        payload: dict[str, Any] = {
            "model": worker.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": bound_max_tokens,
            "stream": False,
        }
        if decision.bypass_prefix_cache and not worker.cache_namespace_header:
            # In single-trusted-scope mode, #827 fails closed to bypass_prefix_cache=true when
            # prompt_cache_key is present without cache_namespace_header.
            payload["prompt_cache_key"] = f"mode-switch-replay:{dispatch_id}"

        t0 = time.monotonic()
        try:
            _, resp_headers, body = _http_json_with_headers(
                "POST",
                f"{worker.base_url}/v1/chat/completions",
                payload=payload,
                extra_headers=extra_headers,
                timeout=timeout,
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            self.record_result(
                dispatch_id=dispatch_id,
                session_id=eff_req.session_id,
                admitted_mode=decision.selected_mode,
                server_mode=None,
                status="http_error",
                elapsed_seconds=elapsed,
                error_detail=str(exc),
                loop_passes=worker.loop_passes,
            )
            raise

        elapsed = time.monotonic() - t0
        hdr_mode = resp_headers.get("x-ingine-kv-cache-mode")
        body_mode = body.get("kv_cache_mode")
        if hdr_mode is not None and body_mode is not None and hdr_mode != body_mode:
            self.record_result(
                dispatch_id=dispatch_id,
                session_id=eff_req.session_id,
                admitted_mode=decision.selected_mode,
                server_mode=f"conflict:{hdr_mode}!={body_mode}",
                status="mode_mismatch",
                elapsed_seconds=elapsed,
                error_detail="response header and body kv_cache_mode disagree",
                loop_passes=worker.loop_passes,
            )
            raise RuntimeError(
                f"server kv_cache_mode conflict on {dispatch_id}: header={hdr_mode!r} body={body_mode!r}"
            )
        server_mode = hdr_mode or body_mode
        if server_mode != decision.selected_mode:
            self.record_result(
                dispatch_id=dispatch_id,
                session_id=eff_req.session_id,
                admitted_mode=decision.selected_mode,
                server_mode=server_mode,
                status="mode_mismatch",
                elapsed_seconds=elapsed,
                error_detail=f"expected={decision.selected_mode!r}, got={server_mode!r}",
                loop_passes=worker.loop_passes,
            )
            raise RuntimeError(
                f"server kv_cache_mode mismatch on {dispatch_id}: admitted={decision.selected_mode!r}, "
                f"server={server_mode!r}"
            )

        # Post-response loop_passes verification (preserves legacy L=3 servers that omit depth fields;
        # fails closed on any explicit mismatch, malformed depth, or omitted depth for L != 3).
        hdr_lp_raw = resp_headers.get("x-ingine-loop-passes")
        has_body_lp = "loop_passes" in body
        parsed_hdr_lp: Optional[int] = None
        parsed_body_lp: Optional[int] = None
        if hdr_lp_raw is not None:
            try:
                if not hdr_lp_raw.isdigit():
                    raise ValueError(hdr_lp_raw)
                parsed_hdr_lp = validate_loop_passes(int(hdr_lp_raw))
            except (TypeError, ValueError):
                self.record_result(
                    dispatch_id=dispatch_id,
                    session_id=eff_req.session_id,
                    admitted_mode=decision.selected_mode,
                    server_mode=server_mode,
                    status="depth_mismatch",
                    elapsed_seconds=elapsed,
                    error_detail=f"invalid response X-Ingine-Loop-Passes header={hdr_lp_raw!r}",
                    loop_passes=worker.loop_passes,
                )
                raise RuntimeError(
                    f"invalid server loop_passes header on {dispatch_id}: {hdr_lp_raw!r}"
                )
        if has_body_lp:
            try:
                parsed_body_lp = validate_loop_passes(body["loop_passes"])
            except (TypeError, ValueError):
                self.record_result(
                    dispatch_id=dispatch_id,
                    session_id=eff_req.session_id,
                    admitted_mode=decision.selected_mode,
                    server_mode=server_mode,
                    status="depth_mismatch",
                    elapsed_seconds=elapsed,
                    error_detail=f"invalid response body loop_passes={body['loop_passes']!r}",
                    loop_passes=worker.loop_passes,
                )
                raise RuntimeError(
                    f"invalid server body loop_passes on {dispatch_id}: {body['loop_passes']!r}"
                )
        if parsed_hdr_lp is not None and parsed_body_lp is not None and parsed_hdr_lp != parsed_body_lp:
            self.record_result(
                dispatch_id=dispatch_id,
                session_id=eff_req.session_id,
                admitted_mode=decision.selected_mode,
                server_mode=server_mode,
                status="depth_mismatch",
                elapsed_seconds=elapsed,
                error_detail=f"response header loop_passes={parsed_hdr_lp} != body={parsed_body_lp}",
                loop_passes=worker.loop_passes,
            )
            raise RuntimeError(
                f"server loop_passes conflict on {dispatch_id}: header={parsed_hdr_lp} body={parsed_body_lp}"
            )
        server_lp = parsed_hdr_lp if parsed_hdr_lp is not None else parsed_body_lp
        if server_lp is None:
            if worker.loop_passes != DEFAULT_LOOP_PASSES:
                self.record_result(
                    dispatch_id=dispatch_id,
                    session_id=eff_req.session_id,
                    admitted_mode=decision.selected_mode,
                    server_mode=server_mode,
                    status="depth_mismatch",
                    elapsed_seconds=elapsed,
                    error_detail=(
                        f"non-default loop_passes={worker.loop_passes} requires explicit completion depth confirmation"
                    ),
                    loop_passes=worker.loop_passes,
                )
                raise RuntimeError(
                    f"server loop_passes omitted on {dispatch_id} for non-default admitted depth {worker.loop_passes}"
                )
            server_lp = DEFAULT_LOOP_PASSES
        elif server_lp != worker.loop_passes:
            self.record_result(
                dispatch_id=dispatch_id,
                session_id=eff_req.session_id,
                admitted_mode=decision.selected_mode,
                server_mode=server_mode,
                status="depth_mismatch",
                elapsed_seconds=elapsed,
                error_detail=f"expected loop_passes={worker.loop_passes}, got={server_lp}",
                loop_passes=worker.loop_passes,
            )
            raise RuntimeError(
                f"server loop_passes mismatch on {dispatch_id}: admitted={worker.loop_passes}, server={server_lp}"
            )

        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = details.get("cached_tokens")

        # Post-response mode-switch verification:
        #   - authorized_replay: server MUST report an explicit integer cached_tokens == 0
        #     (missing, null, boolean, or positive counts fail closed).
        if decision.mode_switch_action == "authorized_replay":
            if type(cached_tokens) is not int or cached_tokens != 0:
                self.record_result(
                    dispatch_id=dispatch_id,
                    session_id=eff_req.session_id,
                    admitted_mode=decision.selected_mode,
                    server_mode=server_mode,
                    status="replay_prefix_cache_violation",
                    prompt_tokens=usage.get("prompt_tokens"),
                    cached_tokens=cached_tokens if type(cached_tokens) is int else None,
                    completion_tokens=usage.get("completion_tokens"),
                    elapsed_seconds=elapsed,
                    error_detail=(
                        f"authorized_replay requires observed integer cached_tokens==0; got {cached_tokens!r}"
                    ),
                )
                raise RuntimeError(
                    f"replay prefix-cache violation on {dispatch_id}: "
                    f"observed cached_tokens={cached_tokens!r} (expected integer 0)"
                )

        ckpt_digest = decision.checkpoint_identity.identity_digest() if decision.checkpoint_identity else None
        self.record_result(
            dispatch_id=dispatch_id,
            session_id=eff_req.session_id,
            admitted_mode=decision.selected_mode,
            server_mode=server_mode,
            status="ok",
            prompt_tokens=usage.get("prompt_tokens"),
            cached_tokens=cached_tokens,
            completion_tokens=usage.get("completion_tokens"),
            elapsed_seconds=elapsed,
            checkpoint_identity_digest=ckpt_digest,
            loop_passes=worker.loop_passes,
        )
        return {
            "dispatch_id": dispatch_id,
            "admitted": True,
            "decision": decision.to_dict(),
            "server_kv_cache_mode": server_mode,
            "server_loop_passes": server_lp,
            "bound_max_tokens": bound_max_tokens,
            "elapsed_seconds": round(elapsed, 4),
            "response": body,
        }

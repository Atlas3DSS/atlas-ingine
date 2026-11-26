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
  3. Fail-closed mode-switch contract: switching a session/task between `per_pass`
     and `last_full` NEVER reuses raw KV state across modes. A mode change requires
     either (a) a verified checkpoint whose identity binds the target `kv_cache_mode`,
     `model_id`, `kv_format`, and `prefix_sha256`, or (b) explicit replay of
     authorized input with `bypass_prefix_cache=True` and a mode-scoped
     `cache_namespace`, with replay latency charged to the deadline budget.
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

# Exact Inkling-2-Mini full-attention KV bytes per token across full slots:
# 4 KV heads * 256 head_dim = 1024 KV_DIM; 8 full slots in per_pass, 4 in last_full.
# Sliding rings (40 visits * 1024 * 1024 * 2 * 2 B = 160 MiB) and conv states (~3.7 MiB)
# remain strictly pass-local and invariant across modes.
FULL_KV_BYTES_PER_TOKEN: dict[tuple[str, str], int] = {
    ("q4_0", "per_pass"): 9216,     # 9 KiB/tok -> 9.00 GiB at 1,048,576 tokens
    ("q4_0", "last_full"): 4608,    # 4.5 KiB/tok -> 4.50 GiB at 1,048,576 tokens
    ("fp8", "per_pass"): 16384,     # 16 KiB/tok -> 16.00 GiB at 1,048,576 tokens
    ("fp8", "last_full"): 8192,     # 8 KiB/tok -> 8.00 GiB at 1,048,576 tokens
    ("bf16", "per_pass"): 32768,    # 32 KiB/tok -> 32.00 GiB at 1,048,576 tokens
    ("bf16", "last_full"): 16384,   # 16 KiB/tok -> 16.00 GiB at 1,048,576 tokens
}
FIXED_PASS_LOCAL_STATE_MIB = 164.0  # 160 MiB sliding rings + ~3.7 MiB conv states


def validate_kv_cache_mode(mode: str) -> str:
    if mode not in VALID_KV_CACHE_MODES:
        raise ValueError(f"invalid kv_cache_mode {mode!r} (expected one of {VALID_KV_CACHE_MODES})")
    return mode


def estimate_full_kv_mib(context_tokens: int, kv_format: str, kv_cache_mode: str) -> float:
    validate_kv_cache_mode(kv_cache_mode)
    fmt = kv_format.lower()
    key = (fmt, kv_cache_mode)
    if key not in FULL_KV_BYTES_PER_TOKEN:
        raise ValueError(f"unsupported kv_format {kv_format!r} (expected q4_0, fp8, or bf16)")
    if context_tokens < 0:
        raise ValueError("context_tokens must be non-negative")
    return (context_tokens * FULL_KV_BYTES_PER_TOKEN[key]) / (1024.0 * 1024.0)


def estimate_total_state_mib(context_tokens: int, kv_format: str, kv_cache_mode: str) -> float:
    return FIXED_PASS_LOCAL_STATE_MIB + estimate_full_kv_mib(context_tokens, kv_format, kv_cache_mode)


def mode_scoped_namespace(base_namespace: str, kv_cache_mode: str) -> str:
    validate_kv_cache_mode(kv_cache_mode)
    base = (base_namespace or "default").strip()
    marker = f":kv_mode={kv_cache_mode}"
    if base.endswith(marker):
        return base
    for m in VALID_KV_CACHE_MODES:
        suffix = f":kv_mode={m}"
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f"{base}{marker}"


@dataclass(frozen=True)
class CheckpointIdentity:
    """Explicit checkpoint / prefix identity binding mode, model, format, namespace, and prefix hash."""

    model_id: str
    kv_format: str
    kv_cache_mode: str
    cache_namespace: str
    prefix_tokens: int
    prefix_sha256: str
    checkpoint_path: Optional[str] = None

    def __post_init__(self) -> None:
        validate_kv_cache_mode(self.kv_cache_mode)

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
    # Digest of a checkpoint already loaded & verified inside this worker runtime (e.g. via
    # `--kv-ckpt-resume <path>`). None means this HTTP worker has no preloaded checkpoint to
    # satisfy a `compatible_checkpoint` mode switch (HTTP /v1/chat/completions has no on-demand
    # disk-checkpoint restore operation, so mode switches must use authorized_replay instead).
    loaded_checkpoint_digest: Optional[str] = None

    def __post_init__(self) -> None:
        validate_kv_cache_mode(self.active_kv_cache_mode)
        for m in self.supported_kv_cache_modes:
            validate_kv_cache_mode(m)

    def supports_mode(self, mode: str) -> bool:
        validate_kv_cache_mode(mode)
        if mode == "last_full":
            if self.max_batch > 1 or self.spec_enabled or self.tp_size > 1:
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
    """Prior session state & authorized transition proof when changing cache modes."""

    prior_kv_cache_mode: str
    compatible_checkpoint: Optional[CheckpointIdentity] = None
    replay_authorized: bool = False
    authorized_replay_tokens: int = 0

    def __post_init__(self) -> None:
        validate_kv_cache_mode(self.prior_kv_cache_mode)


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
                round(self.estimated_state_mib, 3) if self.estimated_state_mib is not None else None
            ),
            "estimated_latency_seconds": (
                round(self.estimated_latency_seconds, 4) if self.estimated_latency_seconds is not None else None
            ),
            "worker_supported_modes": list(self.worker_supported_modes),
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
    kv_format: str = "q4_0",
    available_kv_vram_mib: float = 16384.0,
    spec_enabled: bool = False,
    tp_size: int = 1,
    prefill_tok_per_s: float = 1200.0,
    decode_tok_per_s: float = 40.0,
    measured_last_full_nll_delta: float = 0.01056,
    cache_namespace_header: Optional[str] = None,
    loaded_checkpoint_digest: Optional[str] = None,
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

    raw_supported = health.get("supported_kv_cache_modes") or card.get("supported_kv_cache_modes") or [active_health]
    supported = tuple(validate_kv_cache_mode(m) for m in raw_supported)
    if active_health not in supported:
        raise RuntimeError(
            f"worker {worker_id} active kv_cache_mode {active_health!r} absent from supported {supported!r}"
        )
    mutable = bool(health.get("kv_cache_mode_mutable", card.get("kv_cache_mode_mutable", False)))
    max_batch = int(health.get("max_batch", 1))
    max_model_len = int(card.get("max_model_len", 65536))
    model_id = str(card.get("id") or health.get("model") or "unknown")

    return WorkerCapability(
        worker_id=worker_id,
        base_url=url,
        model_id=model_id,
        active_kv_cache_mode=active_health,
        supported_kv_cache_modes=supported,
        kv_cache_mode_mutable=mutable,
        max_model_len=max_model_len,
        max_batch=max_batch,
        spec_enabled=spec_enabled,
        tp_size=tp_size,
        kv_format=kv_format,
        available_kv_vram_mib=available_kv_vram_mib,
        prefill_tok_per_s=prefill_tok_per_s,
        decode_tok_per_s=decode_tok_per_s,
        measured_last_full_nll_delta=measured_last_full_nll_delta,
        cache_namespace_header=cache_namespace_header,
        loaded_checkpoint_digest=loaded_checkpoint_digest,
    )


def _validate_mode_switch(
    worker: WorkerCapability,
    req: AdmissionRequest,
    target_mode: str,
) -> tuple[bool, str, str, bool, int, Optional[str]]:
    """Enforce fail-closed mode-switch rules.

    Returns: (ok, reason, action, bypass_prefix_cache, extra_replay_tokens, ckpt_path)
    """
    ms = req.mode_switch
    if ms is None or ms.prior_kv_cache_mode == target_mode:
        return True, "same_or_fresh_mode", "none", False, 0, None

    # Mode change requested: raw cross-mode KV reuse is strictly forbidden.
    ckpt = ms.compatible_checkpoint
    if ckpt is not None:
        if ckpt.kv_cache_mode != target_mode:
            return (
                False,
                f"mode_switch_checkpoint_mode_mismatch:ckpt={ckpt.kv_cache_mode}!=target={target_mode}",
                "none",
                True,
                0,
                None,
            )
        if ckpt.model_id != worker.model_id or ckpt.kv_format.lower() != worker.kv_format.lower():
            return (
                False,
                "mode_switch_checkpoint_runtime_mismatch",
                "none",
                True,
                0,
                None,
            )
        if ckpt.prefix_sha256 != req.prefix_sha256:
            return (
                False,
                "mode_switch_checkpoint_prefix_mismatch",
                "none",
                True,
                0,
                None,
            )
        if ckpt.prefix_tokens <= 0:
            return (
                False,
                "mode_switch_checkpoint_empty_prefix",
                "none",
                True,
                0,
                None,
            )
        # HTTP /v1/chat/completions exposes no on-demand disk checkpoint restore operation.
        # Admit `compatible_checkpoint` ONLY when the target worker already holds that exact
        # verified checkpoint loaded (e.g. launched with `--kv-ckpt-resume`) and verify
        # `cached_tokens >= ckpt.prefix_tokens` on the response. Otherwise require authorized_replay.
        if worker.loaded_checkpoint_digest == ckpt.identity_digest():
            return (
                True,
                "mode_switch_via_verified_loaded_checkpoint",
                "compatible_checkpoint",
                False,
                0,
                ckpt.checkpoint_path,
            )
        if ms.replay_authorized and ms.authorized_replay_tokens > 0:
            return (
                True,
                "mode_switch_via_authorized_replay",
                "authorized_replay",
                True,
                ms.authorized_replay_tokens,
                None,
            )
        return (
            False,
            "mode_switch_checkpoint_restore_unavailable_require_authorized_replay",
            "none",
            True,
            0,
            None,
        )

    if ms.replay_authorized and ms.authorized_replay_tokens > 0:
        return (
            True,
            "mode_switch_via_authorized_replay",
            "authorized_replay",
            True,
            ms.authorized_replay_tokens,
            None,
        )

    return (
        False,
        f"mode_switch_blocked_no_checkpoint_or_replay:{ms.prior_kv_cache_mode}->{target_mode}",
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

    state_mib = estimate_total_state_mib(total_tokens, worker.kv_format, mode)
    eff_mem_limit = worker.available_kv_vram_mib - req.budgets.vram_headroom_mib
    if req.budgets.memory_budget_mib is not None:
        eff_mem_limit = min(eff_mem_limit, req.budgets.memory_budget_mib)
    if state_mib > eff_mem_limit:
        return AdmissionDecision(
            admitted=False,
            worker_id=worker.worker_id,
            selected_mode=None,
            reason=f"memory_budget_exceeded:{state_mib:.1f}MiB>{eff_mem_limit:.1f}MiB",
            estimated_state_mib=state_mib,
            worker_supported_modes=worker.supported_kv_cache_modes,
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
        )

    eff_ns = mode_scoped_namespace(req.cache_namespace, mode)
    ckpt_id = CheckpointIdentity(
        model_id=worker.model_id,
        kv_format=worker.kv_format,
        kv_cache_mode=mode,
        cache_namespace=eff_ns,
        prefix_tokens=req.context_tokens,
        prefix_sha256=req.prefix_sha256,
        checkpoint_path=ckpt_path,
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
                checkpoint_identity_digest TEXT,
                updated_at REAL NOT NULL
            );
            """
        )
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
                INSERT INTO session_modes (session_id, kv_cache_mode, checkpoint_identity_digest, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    kv_cache_mode = excluded.kv_cache_mode,
                    checkpoint_identity_digest = excluded.checkpoint_identity_digest,
                    updated_at = excluded.updated_at
                """,
                (session_id, admitted_mode, checkpoint_identity_digest, now),
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
        prior_mode = self.get_session_mode(req.session_id)
        eff_req = req
        if prior_mode is not None and req.mode_switch is None:
            # Bind recorded session mode automatically so silent mode switches are blocked.
            eff_req = dataclasses.replace(req, mode_switch=ModeSwitchContext(prior_kv_cache_mode=prior_mode))

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
        extra_headers: dict[str, str] = {}
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
            )
            raise RuntimeError(
                f"server kv_cache_mode mismatch on {dispatch_id}: admitted={decision.selected_mode!r}, "
                f"server={server_mode!r}"
            )

        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = details.get("cached_tokens")

        # Post-response mode-switch verification:
        #   - compatible_checkpoint: server MUST prove cached_tokens >= checkpoint.prefix_tokens.
        #   - authorized_replay: server MUST NOT reuse cached tokens (cached_tokens must be 0 or None).
        if decision.mode_switch_action == "compatible_checkpoint":
            want_prefix = (
                eff_req.mode_switch.compatible_checkpoint.prefix_tokens
                if (eff_req.mode_switch and eff_req.mode_switch.compatible_checkpoint)
                else 1
            )
            if not isinstance(cached_tokens, int) or cached_tokens < want_prefix:
                self.record_result(
                    dispatch_id=dispatch_id,
                    session_id=eff_req.session_id,
                    admitted_mode=decision.selected_mode,
                    server_mode=server_mode,
                    status="checkpoint_restore_unverified",
                    prompt_tokens=usage.get("prompt_tokens"),
                    cached_tokens=cached_tokens if isinstance(cached_tokens, int) else None,
                    completion_tokens=usage.get("completion_tokens"),
                    elapsed_seconds=elapsed,
                    error_detail=f"cached_tokens={cached_tokens!r} < required checkpoint prefix_tokens={want_prefix}",
                )
                raise RuntimeError(
                    f"checkpoint restore unverified on {dispatch_id}: "
                    f"cached_tokens={cached_tokens!r} < required prefix_tokens={want_prefix}"
                )
        elif decision.mode_switch_action == "authorized_replay":
            if cached_tokens not in (None, 0):
                self.record_result(
                    dispatch_id=dispatch_id,
                    session_id=eff_req.session_id,
                    admitted_mode=decision.selected_mode,
                    server_mode=server_mode,
                    status="replay_prefix_cache_violation",
                    prompt_tokens=usage.get("prompt_tokens"),
                    cached_tokens=cached_tokens if isinstance(cached_tokens, int) else None,
                    completion_tokens=usage.get("completion_tokens"),
                    elapsed_seconds=elapsed,
                    error_detail=f"authorized_replay observed non-zero cached_tokens={cached_tokens!r}",
                )
                raise RuntimeError(
                    f"replay prefix-cache violation on {dispatch_id}: cached_tokens={cached_tokens!r} (expected 0)"
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
        )
        return {
            "dispatch_id": dispatch_id,
            "admitted": True,
            "decision": decision.to_dict(),
            "server_kv_cache_mode": server_mode,
            "bound_max_tokens": bound_max_tokens,
            "elapsed_seconds": round(elapsed, 4),
            "response": body,
        }

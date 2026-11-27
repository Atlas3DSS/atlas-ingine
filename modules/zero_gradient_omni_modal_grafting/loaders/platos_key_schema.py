"""Universal closed-form Plato's Key v2 schema, multi-layer FP64 Cholesky Ridge solver,
multi-encoder / pooled-embedding container, and Voltron Android LAN (<2 MiB) wire packet.

Shared across #943 Wave A (mecha-monkey) and Wave B (voltron-hub) in
`/mnt/storage/omni_platos_keys_images/`.

Backward-compatible with `platos_key_v1` (`R`, `scale`, `donor_mean`, `target_mean`),
while extending cleanly to:
  1. Multi-layer stacked conditioning targets (Krea-2 12x2560=30720, FLUX-3-klein 8x2560=20480,
     FLUX.2-klein-4B 3x2560=7680, FLUX.2-klein-9B 3x4096=12288, FLUX.2-dev 3x5120=15360) via
     exact column-separable FP64 Cholesky Ridge + optional per-layer variance matching.
  2. Dual-encoder / pooled-embedding targets (FLUX.1-schnell/dev T5-XXL + CLIP-L pooled;
     SDXL-base/turbo CLIP-L + OpenCLIP-bigG seq + OpenCLIP-bigG pooled) via `pooled.*` and
     optional `branch.<name>.*` tensors inside a single `.safetensors` artifact.
  3. Voltron Android #914 LAN conditioning wire packets (`VPKE` v1, FP16 `[seq_len, 2560]`,
     <= 2 MiB hard cap + SHA-256 verification) for `Tongyi-MAI/Z-Image-Turbo` pre-`cap_embedder`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION_V2 = "omni-platos-key-image/v2"
FORMAT_V2 = "platos_key_v2"
FORMAT_V1 = "platos_key_v1"

INKLING_HIDDEN_SIZE = 2304
INKLING_FORBIDDEN_COORD = 2027
INKLING_TAP_SLOTS: Tuple[str, ...] = ("visit_5", "visit_23", "visit_41", "final_xf_normed")
INKLING_TAP_SPECS: Tuple[str, ...] = ("visit_5", "visit_23", "visit_41", "final_xf_normed", "concat_all4")
PLATOS_KEY_SOLVERS: Tuple[str, ...] = ("ridge", "ridge_norm", "ridge_per_layer_norm", "procrustes_var_match")
DEFAULT_RIDGE_LAMBDAS: Tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)

# Frozen Inkling-2-Mini donor corpus & binaries reused across Wave A & Wave B (zero Inkling forward passes)
FROZEN_DONOR_ARTIFACTS: Dict[str, Any] = {
    "donor_model_repo": "thinkingmachines/Inkling-2-Mini",
    "donor_hidden_dim": INKLING_HIDDEN_SIZE,
    "donor_tap_slots": list(INKLING_TAP_SLOTS),
    "donor_forbidden_coord": INKLING_FORBIDDEN_COORD,
    "prompt_manifest_path": "/mnt/storage/omni-platos-key-dev/configs/h3_platos_key_phase0_prompts.jsonl",
    "prompt_manifest_sha256": "9a577ef24ecf6234db1374bbd86fcf8088f46684cc2d3c8d65b0f8827ade8c52",
    "donor_tokenizer_json_quincy": "/mnt/nvme2/models/Inkling-2-Mini/tokenizer.json",
    "donor_tokenizer_json_sha256": "7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6",
    "fit_valdev_800": {
        "dir": "/mnt/storage/omni_gate2_runs/h3_platos_key_phase0_24p_a8146fa7",
        "bin_path": "/mnt/storage/omni_gate2_runs/h3_platos_key_phase0_24p_a8146fa7/donor_inkling_taps_bf16.bin",
        "bin_bytes": 513054720,
        "bin_sha256": "53791b1e7a50e12f8f200af9dfed1cb6a8b2b324b0dc57c0b061ab89df407828",
        "stdout_path": "/mnt/storage/omni_gate2_runs/h3_platos_key_phase0_24p_a8146fa7/phase1_donor_export.stdout.txt",
        "stdout_sha256": "8a2561e7e13c38b203c4af64b6f5557a3807511eceffa14e4d614ca324b33fa0",
        "offsets_path": "/mnt/storage/omni_gate2_runs/h3_platos_key_phase0_24p_a8146fa7/donor_offsets.json",
        "offsets_sha256": "c1c5dc76361643bd4f3d6479033dfb02d27ec29f13d734002fc9459c768cde57",
        "prompts_jsonl_path": "/mnt/storage/omni_gate2_runs/h3_platos_key_phase0_24p_a8146fa7/donor_text_prompts.jsonl",
        "prompts_jsonl_sha256": "07f8c72006c41754394502b34d1ef49941e9d865b4ea3c62d6b729f6903046fe",
        "n_prompts": 800,
        "n_fit": 640,
        "n_val_dev": 160,
    },
    "confirm160_sealed": {
        "dir": "/mnt/storage/omni_gate2_runs/packetA_confirm160_full/phase1_confirm160_workers",
        "bin_path": "/mnt/storage/omni_gate2_runs/packetA_confirm160_full/phase1_confirm160_workers/confirm160_donor_inkling_taps_bf16.bin",
        "bin_bytes": 102998016,
        "bin_sha256": "21811907000f84a8a31995c254fa1e929b298f38d48bb3c5ec243a680b62f7af",
        "stdout_path": "/mnt/storage/omni_gate2_runs/packetA_confirm160_full/phase1_confirm160_workers/donor_160.stdout.txt",
        "stdout_sha256": "2eecf864ced2ded1c8df8d2a43b15eefdbf3f7acecfff703f805540ae66c7c49",
        "prompts_jsonl_path": "/mnt/storage/omni_gate2_runs/packetA_confirm160_full/phase1_confirm160_workers/confirm_only_160.jsonl",
        "prompts_jsonl_sha256": "c001f3cd2b2135ee9910798b191520615a0e05c6307ddfac4abbd19952a643ec",
        "donor_req_jsonl_path": "/mnt/storage/omni_gate2_runs/packetA_confirm160_full/phase1_confirm160_workers/confirm160_donor_req.jsonl",
        "donor_req_jsonl_sha256": "4f9a8c5ccb1978421677cbd2e2e864c666a4e2a3d1a65c83b5297af458e2a768",
        "n_prompts": 160,
    },
}

WORD_SPAN_RE = re.compile(r"\S+")

# Voltron Android #914 LAN packet constants (2 MiB strict cap)
VOLTRON_PACKET_MAGIC = b"VPKE"
VOLTRON_PACKET_VERSION = 1
VOLTRON_PACKET_SCHEMA = "voltron-plato-cond-packet/v1"
VOLTRON_PACKET_MAX_BYTES = 2 * 1024 * 1024  # 2,097,152 bytes


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> Tuple[int, str]:
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_bytes)
            if not chunk:
                break
            total += len(chunk)
            h.update(chunk)
    return total, h.hexdigest()


def bf16_u16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    """Exact bit-shift BF16 uint16 -> FP32 conversion without torch dependency."""
    u32 = np.asarray(raw_u16, dtype=np.uint16).astype(np.uint32) << 16
    return u32.view(np.float32)


def verify_donor_d2027_zero(
    taps_l4h: np.ndarray,
    *,
    forbidden_coord: int = INKLING_FORBIDDEN_COORD,
) -> float:
    """Fail-closed check that Inkling forbidden coordinate d_2027 is exactly 0.0."""
    arr = np.asarray(taps_l4h)
    if arr.ndim != 3 or arr.shape[1] != 4:
        raise ValueError(f"Expected [L, 4, H] donor taps, got {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError("Donor taps sequence length must be >= 1")
    if arr.shape[2] <= forbidden_coord:
        raise ValueError(f"Donor dim {arr.shape[2]} <= forbidden_coord {forbidden_coord}")
    max_abs = float(np.max(np.abs(arr[:, :, forbidden_coord].astype(np.float64))))
    if not math.isfinite(max_abs) or max_abs > 0.0:
        raise ValueError(f"Forbidden Inkling coordinate d_{forbidden_coord} non-zero/non-finite: {max_abs}")
    return max_abs


def extract_tap_matrix(taps_l4h: np.ndarray, tap_spec: str, *, check_d2027: bool = True) -> np.ndarray:
    """Extract `[L, 2304]` single tap or `[L, 9216]` concat_all4 from `[L, 4, 2304]`."""
    arr = np.asarray(taps_l4h)
    if check_d2027:
        verify_donor_d2027_zero(arr)
    if tap_spec == "concat_all4":
        return arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])
    if tap_spec not in INKLING_TAP_SLOTS:
        raise ValueError(f"Unknown tap_spec {tap_spec!r}; expected one of {INKLING_TAP_SPECS}")
    return arr[:, INKLING_TAP_SLOTS.index(tap_spec), :]


def pool_donor_sequence(feat_ld: np.ndarray, mode: str) -> np.ndarray:
    """Pool a sequence `[L, D]` into a single prompt vector `[D]` for pooled conditioning branches."""
    arr = np.asarray(feat_ld, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 1:
        raise ValueError(f"Expected non-empty 2D sequence [L, D], got {arr.shape}")
    if mode == "mean":
        return arr.mean(axis=0, dtype=np.float64)
    if mode == "last":
        return arr[-1].astype(np.float64, copy=True)
    raise ValueError(f"Unsupported pooled_donor_pooling mode {mode!r}; expected 'mean' or 'last'")


def whitespace_word_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_SPAN_RE.finditer(text)]


def pool_tokens_to_word_spans(
    token_hidden: np.ndarray,
    token_offsets: Sequence[Tuple[int, int]],
    word_spans: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, List[int]]:
    """Mean-pool subword hidden states `[L, D]` into whitespace-delimited word spans in FP64."""
    h = np.asarray(token_hidden, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != len(token_offsets):
        raise ValueError(f"token_hidden shape {h.shape} incompatible with {len(token_offsets)} offsets")
    vecs: List[np.ndarray] = []
    valid_ids: List[int] = []
    for w_idx, (w0, w1) in enumerate(word_spans):
        tok_idx = [
            ti for ti, (t0, t1) in enumerate(token_offsets)
            if int(t1) > int(t0) and int(t0) < w1 and int(t1) > w0
        ]
        if tok_idx:
            vecs.append(h[tok_idx].mean(axis=0, dtype=np.float64))
            valid_ids.append(w_idx)
    if not vecs:
        return np.zeros((0, h.shape[1]), dtype=np.float64), []
    return np.stack(vecs, axis=0), valid_ids


def align_cross_tokenizer_word_spans(
    prompt: str,
    *,
    donor_hidden: np.ndarray,
    donor_offsets: Sequence[Tuple[int, int]],
    target_hidden: np.ndarray,
    target_offsets: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Align donor and target per-token hidden states across disjoint tokenizers by common word spans."""
    spans = whitespace_word_spans(prompt)
    d_vecs, d_ids = pool_tokens_to_word_spans(donor_hidden, donor_offsets, spans)
    t_vecs, t_ids = pool_tokens_to_word_spans(target_hidden, target_offsets, spans)
    d_map = {wid: i for i, wid in enumerate(d_ids)}
    t_map = {wid: i for i, wid in enumerate(t_ids)}
    common = sorted(set(d_map.keys()) & set(t_map.keys()))
    if not common:
        raise ValueError(f"Zero aligned word spans for prompt: {prompt!r}")
    xd = np.stack([d_vecs[d_map[w]] for w in common], axis=0)
    xt = np.stack([t_vecs[t_map[w]] for w in common], axis=0)
    return xd, xt, common


@dataclass(frozen=True)
class PlatosKeyBranch:
    """Single affine projection branch: y_hat = target_mean + scale_vec * ((x - donor_mean) @ R)."""

    R: np.ndarray  # [d_donor, d_target] float64
    scale: float
    donor_mean: np.ndarray  # [d_donor] float64
    target_mean: np.ndarray  # [d_target] float64
    method: str
    ridge_lambda: Optional[float] = None
    per_layer_scale: Optional[np.ndarray] = None  # [n_layers] float64 when multi-layer scaled
    layer_dims: Optional[Tuple[int, ...]] = None  # e.g. (2560,)*12 for Krea-2

    @property
    def d_donor(self) -> int:
        return int(self.R.shape[0])

    @property
    def d_target(self) -> int:
        return int(self.R.shape[1])

    def effective_col_scale(self) -> Optional[np.ndarray]:
        if self.per_layer_scale is None:
            return None
        if self.layer_dims is None:
            n_layers = int(self.per_layer_scale.shape[0])
            if self.d_target % n_layers != 0:
                raise ValueError(f"d_target={self.d_target} not divisible by n_layers={n_layers}")
            ldims = (self.d_target // n_layers,) * n_layers
        else:
            ldims = tuple(int(d) for d in self.layer_dims)
        if sum(ldims) != self.d_target or len(ldims) != int(self.per_layer_scale.shape[0]):
            raise ValueError(f"layer_dims {ldims} incompatible with d_target={self.d_target} and per_layer_scale")
        parts = [np.full((d,), float(s), dtype=np.float64) for d, s in zip(ldims, self.per_layer_scale)]
        return np.concatenate(parts, axis=0)

    def apply(self, x: np.ndarray, *, out_dtype: Any = np.float32) -> np.ndarray:
        """Apply affine map `[..., d_donor] -> [..., d_target]` in FP64/FP32 -> out_dtype."""
        x64 = np.asarray(x, dtype=np.float64)
        if x64.shape[-1] != self.d_donor:
            raise ValueError(f"Donor dim mismatch: got {x64.shape[-1]}, expected {self.d_donor}")
        centered = x64 - self.donor_mean
        proj = centered @ self.R
        col_scale = self.effective_col_scale()
        if col_scale is not None:
            mapped = self.target_mean + (float(self.scale) * proj) * col_scale
        else:
            mapped = self.target_mean + float(self.scale) * proj
        return mapped.astype(out_dtype)

    def digest_sha256(self) -> str:
        h = hashlib.sha256()
        items = [self.R, np.array([self.scale], dtype=np.float64), self.donor_mean, self.target_mean]
        if self.per_layer_scale is not None:
            items.append(self.per_layer_scale)
        for arr in items:
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class PlatosKeyArtifact:
    """Universal `platos_key_v2` container supporting single-encoder, multi-layer, dual-encoder+pooled,
    template bookends (`prefix_embeds` / `suffix_embeds`), and token-0 `sink_embedding` (P-Image-OSS)."""

    primary: PlatosKeyBranch
    pooled: Optional[PlatosKeyBranch] = None
    extra_branches: Dict[str, PlatosKeyBranch] = field(default_factory=dict)
    prefix_embeds: Optional[np.ndarray] = None  # [N_pre, d_target] float64 optional chat-template prefix
    suffix_embeds: Optional[np.ndarray] = None  # [N_suf, d_target] float64 optional chat-template suffix
    sink_embedding: Optional[np.ndarray] = None  # [d_target] float64 optional token-0 sink (P-Image-OSS)
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def d_donor(self) -> int:
        return self.primary.d_donor

    @property
    def d_target(self) -> int:
        return self.primary.d_target

    @property
    def target_id(self) -> str:
        return str(self.metadata.get("target_id", "unknown"))

    @property
    def tap_spec(self) -> str:
        return str(self.metadata.get("tap_spec", "concat_all4"))

    @property
    def pooled_donor_pooling(self) -> str:
        return str(self.metadata.get("pooled_donor_pooling", "mean"))

    def apply_primary(
        self,
        x: np.ndarray,
        *,
        out_dtype: Any = np.float32,
        include_bookends: bool = False,
        apply_sink: bool = True,
    ) -> np.ndarray:
        mapped = self.primary.apply(x, out_dtype=np.float64)
        if apply_sink and self.sink_embedding is not None:
            sink = np.asarray(self.sink_embedding, dtype=np.float64).reshape(1, self.d_target)
            mapped = np.concatenate([sink, mapped], axis=0) if bool(
                self.metadata.get("sink_mode", "replace_token0") == "prepend"
            ) else np.concatenate([sink, mapped[1:]], axis=0)
        if include_bookends:
            parts: List[np.ndarray] = []
            if self.prefix_embeds is not None and self.prefix_embeds.shape[0] > 0:
                parts.append(np.asarray(self.prefix_embeds, dtype=np.float64))
            parts.append(mapped)
            if self.suffix_embeds is not None and self.suffix_embeds.shape[0] > 0:
                parts.append(np.asarray(self.suffix_embeds, dtype=np.float64))
            mapped = np.concatenate(parts, axis=0)
        return mapped.astype(out_dtype)

    def apply_pooled(self, x_seq_or_vec: np.ndarray, *, out_dtype: Any = np.float32) -> np.ndarray:
        if self.pooled is None:
            raise ValueError(f"Key for {self.target_id!r} has no pooled branch")
        arr = np.asarray(x_seq_or_vec, dtype=np.float64)
        if arr.ndim == 2:
            vec = pool_donor_sequence(arr, self.pooled_donor_pooling)
        elif arr.ndim == 1:
            vec = arr
        else:
            raise ValueError(f"Expected [L, D] or [D] for pooled branch, got {arr.shape}")
        return self.pooled.apply(vec, out_dtype=out_dtype)

    def digest_sha256(self) -> str:
        h = hashlib.sha256()
        h.update(self.primary.digest_sha256().encode("ascii"))
        if self.pooled is not None:
            h.update(b"|pooled|")
            h.update(self.pooled.digest_sha256().encode("ascii"))
        for name in sorted(self.extra_branches.keys()):
            h.update(f"|branch:{name}|".encode("ascii"))
            h.update(self.extra_branches[name].digest_sha256().encode("ascii"))
        for label, arr in (
            ("prefix_embeds", self.prefix_embeds),
            ("suffix_embeds", self.suffix_embeds),
            ("sink_embedding", self.sink_embedding),
        ):
            if arr is not None:
                h.update(f"|{label}|".encode("ascii"))
                h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()


class StreamingMultiLayerRidgeFP64:
    """Exact FP64 streaming normal-equations accumulator + column-separable Cholesky Ridge solver.

    Accumulates exact FP64 sufficient statistics `(n, sum_x, sum_y, X^T X, X^T Y, diag_y2_per_layer)`
    across arbitrary prompt batches without materializing giant `[N, 30720]` FP64 matrices in RAM.
    Solving ONE Cholesky factorization of `C_xx + lambda * I` (`d_donor x d_donor`) and solving
    each target layer block `k` is bit-identical to full-matrix Ridge up to FP64 rounding (`<1e-12`).
    """

    def __init__(self, d_donor: int, layer_dims: Sequence[int]) -> None:
        self.d_donor = int(d_donor)
        self.layer_dims: Tuple[int, ...] = tuple(int(d) for d in layer_dims)
        if not self.layer_dims or any(d <= 0 for d in self.layer_dims):
            raise ValueError(f"Invalid layer_dims: {layer_dims}")
        self.d_target = int(sum(self.layer_dims))
        self.n_samples: int = 0
        self.sum_x = np.zeros((self.d_donor,), dtype=np.float64)
        self.sum_y = np.zeros((self.d_target,), dtype=np.float64)
        self.xtx = np.zeros((self.d_donor, self.d_donor), dtype=np.float64)
        self.xty = np.zeros((self.d_donor, self.d_target), dtype=np.float64)
        self.sum_y2_per_layer = np.zeros((len(self.layer_dims),), dtype=np.float64)
        self._layer_slices: List[Tuple[int, int]] = []
        off = 0
        for d in self.layer_dims:
            self._layer_slices.append((off, off + d))
            off += d

    def update(self, x_batch: np.ndarray, y_batch: np.ndarray) -> None:
        x = np.asarray(x_batch, dtype=np.float64)
        y = np.asarray(y_batch, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError(f"Incompatible batch shapes: x={x.shape}, y={y.shape}")
        if x.shape[1] != self.d_donor or y.shape[1] != self.d_target:
            raise ValueError(
                f"Dim mismatch: got x={x.shape[1]}, y={y.shape[1]}; expected ({self.d_donor}, {self.d_target})"
            )
        if x.shape[0] == 0:
            return
        if not (bool(np.all(np.isfinite(x))) and bool(np.all(np.isfinite(y)))):
            raise ValueError("Non-finite value in Ridge accumulator batch")
        self.n_samples += int(x.shape[0])
        self.sum_x += x.sum(axis=0, dtype=np.float64)
        self.sum_y += y.sum(axis=0, dtype=np.float64)
        self.xtx += x.T @ x
        self.xty += x.T @ y
        for li, (s0, s1) in enumerate(self._layer_slices):
            yl = y[:, s0:s1]
            self.sum_y2_per_layer[li] += float(np.sum(yl * yl, dtype=np.float64))

    def _centered_stats(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.n_samples < 2:
            raise ValueError(f"Need >= 2 samples to fit Ridge, got {self.n_samples}")
        n = float(self.n_samples)
        mu_x = self.sum_x / n
        mu_y = self.sum_y / n
        c_xx = self.xtx - n * np.outer(mu_x, mu_x)
        c_xx = 0.5 * (c_xx + c_xx.T)
        c_xy = self.xty - n * np.outer(mu_x, mu_y)
        y_c_frob2_per_layer = np.zeros((len(self.layer_dims),), dtype=np.float64)
        for li, (s0, s1) in enumerate(self._layer_slices):
            mu_l = mu_y[s0:s1]
            f2 = self.sum_y2_per_layer[li] - n * float(np.dot(mu_l, mu_l))
            y_c_frob2_per_layer[li] = max(f2, 0.0)
        return mu_x, mu_y, c_xx, c_xy, y_c_frob2_per_layer

    def solve(
        self,
        ridge_lambda: float,
        *,
        norm_mode: str = "none",
    ) -> Tuple[PlatosKeyBranch, Dict[str, Any]]:
        """Solve closed-form FP64 Ridge via Cholesky factorization.

        `norm_mode`:
          - `"none"`: standard Ridge (`scale = 1.0`, `per_layer_scale = None`)
          - `"global"`: fit-only global Frobenius variance matching (`ridge_norm`)
          - `"per_layer"`: fit-only per-layer Frobenius variance matching (`ridge_per_layer_norm`)
        """
        lam = float(ridge_lambda)
        if not (math.isfinite(lam) and lam > 0.0):
            raise ValueError(f"ridge_lambda must be finite > 0, got {ridge_lambda}")
        if norm_mode not in ("none", "global", "per_layer"):
            raise ValueError(f"Unknown norm_mode {norm_mode!r}")

        mu_x, mu_y, c_xx, c_xy, y_c_frob2_layers = self._centered_stats()
        gram = c_xx.copy()
        gram[np.diag_indices(self.d_donor)] += lam

        # Exact SPD Cholesky solve L L^T W = C_xy, solved layer-by-layer to bound peak RAM
        try:
            chol = np.linalg.cholesky(gram)
            use_chol = True
        except np.linalg.LinAlgError:
            chol = None
            use_chol = False

        w_mat = np.empty((self.d_donor, self.d_target), dtype=np.float64)
        pred_frob2_layers = np.zeros((len(self.layer_dims),), dtype=np.float64)
        for li, (s0, s1) in enumerate(self._layer_slices):
            rhs_l = c_xy[:, s0:s1]
            if use_chol and chol is not None:
                z_l = np.linalg.solve(chol, rhs_l)
                w_l = np.linalg.solve(chol.T, z_l)
            else:
                w_l = np.linalg.solve(gram, rhs_l)
            w_mat[:, s0:s1] = w_l
            # Exact centered prediction Frobenius^2: tr(W_l^T C_xx W_l) = sum(W_l * (C_xx @ W_l))
            cw_l = c_xx @ w_l
            pred_frob2_layers[li] = max(float(np.sum(w_l * cw_l, dtype=np.float64)), 0.0)

        frob_y_global = math.sqrt(max(float(np.sum(y_c_frob2_layers)), 0.0))
        frob_pred_global = math.sqrt(max(float(np.sum(pred_frob2_layers)), 0.0))
        if not (math.isfinite(frob_y_global) and math.isfinite(frob_pred_global) and frob_pred_global > 1e-12):
            raise ValueError(f"Degenerate Ridge norms: frob_y={frob_y_global}, frob_pred={frob_pred_global}")

        per_layer_scale: Optional[np.ndarray] = None
        if norm_mode == "none":
            scale = 1.0
            method = "ridge"
        elif norm_mode == "global":
            scale = float(frob_y_global / frob_pred_global)
            method = "ridge_norm"
        else:
            scale = 1.0
            method = "ridge_per_layer_norm"
            pls = np.ones((len(self.layer_dims),), dtype=np.float64)
            for li in range(len(self.layer_dims)):
                fy = math.sqrt(max(float(y_c_frob2_layers[li]), 0.0))
                fp = math.sqrt(max(float(pred_frob2_layers[li]), 0.0))
                if fp <= 1e-12:
                    raise ValueError(f"Degenerate layer-{li} Ridge pred norm: {fp}")
                pls[li] = float(fy / fp)
            per_layer_scale = pls

        branch = PlatosKeyBranch(
            R=w_mat,
            scale=scale,
            donor_mean=mu_x,
            target_mean=mu_y,
            method=method,
            ridge_lambda=lam,
            per_layer_scale=per_layer_scale,
            layer_dims=self.layer_dims,
        )
        audit = {
            "method": method,
            "solver_backend": "fp64_cholesky" if use_chol else "fp64_lu_fallback",
            "ridge_lambda": lam,
            "n_samples": self.n_samples,
            "d_donor": self.d_donor,
            "d_target": self.d_target,
            "layer_dims": list(self.layer_dims),
            "fit_target_centered_frob": frob_y_global,
            "fit_raw_pred_centered_frob": frob_pred_global,
            "fit_raw_pred_to_target_rms_ratio": float(frob_pred_global / max(frob_y_global, 1e-12)),
            "scale": scale,
            "per_layer_scale": [float(v) for v in per_layer_scale] if per_layer_scale is not None else None,
        }
        return branch, audit


def fit_procrustes_var_match(
    x_donor: np.ndarray,
    y_target: np.ndarray,
    *,
    layer_dims: Optional[Sequence[int]] = None,
) -> Tuple[PlatosKeyBranch, Dict[str, Any]]:
    """Closed-form Orthogonal Procrustes + variance-matching global scale (Stanford Plato's Key)."""
    xd = np.asarray(x_donor, dtype=np.float64)
    yt = np.asarray(y_target, dtype=np.float64)
    if xd.ndim != 2 or yt.ndim != 2 or xd.shape[0] != yt.shape[0] or xd.shape[0] < 2:
        raise ValueError(f"Invalid shapes for Procrustes: x={xd.shape}, y={yt.shape}")
    mu_x = xd.mean(axis=0)
    mu_y = yt.mean(axis=0)
    xc = xd - mu_x
    yc = yt - mu_y
    u, s_vals, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
    r_mat = u @ vt
    proj = xc @ r_mat
    frob_y = float(np.linalg.norm(yc))
    frob_proj = float(np.linalg.norm(proj))
    if not (math.isfinite(frob_y) and math.isfinite(frob_proj) and frob_proj > 1e-12):
        raise ValueError(f"Degenerate Procrustes norms: frob_y={frob_y}, frob_proj={frob_proj}")
    scale_vm = float(frob_y / frob_proj)
    ldims = tuple(int(d) for d in layer_dims) if layer_dims is not None else (int(yt.shape[1]),)
    branch = PlatosKeyBranch(
        R=r_mat,
        scale=scale_vm,
        donor_mean=mu_x,
        target_mean=mu_y,
        method="procrustes_var_match",
        ridge_lambda=None,
        per_layer_scale=None,
        layer_dims=ldims,
    )
    audit = {
        "method": "procrustes_var_match",
        "n_samples": int(xd.shape[0]),
        "d_donor": int(xd.shape[1]),
        "d_target": int(yt.shape[1]),
        "fit_target_centered_frob": frob_y,
        "fit_rot_centered_frob": frob_proj,
        "scale": scale_vm,
        "scale_least_squares_ref": float(np.sum(s_vals) / (frob_proj * frob_proj)),
    }
    return branch, audit


def evaluate_branch_metrics(
    x_donor: np.ndarray,
    y_target: np.ndarray,
    branch: PlatosKeyBranch,
    *,
    layer_dims: Optional[Sequence[int]] = None,
    layer_labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compute exact FP64 overall and per-layer aligned word-span metrics."""
    xd = np.asarray(x_donor, dtype=np.float64)
    yt = np.asarray(y_target, dtype=np.float64)
    pred = branch.apply(xd, out_dtype=np.float64)

    def _metrics_for_slice(p: np.ndarray, t: np.ndarray) -> Dict[str, float]:
        dot = np.sum(p * t, axis=-1)
        cos = dot / np.maximum(np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1), 1e-12)
        res = t - p
        ss_res = float(np.sum(res * res, dtype=np.float64))
        tc = t - t.mean(axis=0, keepdims=True)
        ss_tot = float(np.sum(tc * tc, dtype=np.float64))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        p_rms = float(math.sqrt(np.mean(p * p, dtype=np.float64)))
        t_rms = float(math.sqrt(np.mean(t * t, dtype=np.float64)))
        return {
            "n_rows": int(t.shape[0]),
            "dim": int(t.shape[1]),
            "cosine_mean": float(np.mean(cos)),
            "cosine_p10": float(np.quantile(cos, 0.10)),
            "cosine_min": float(np.min(cos)),
            "r2": float(r2),
            "mse": float(np.mean(res * res, dtype=np.float64)),
            "rel_l2": float(math.sqrt(ss_res) / (np.linalg.norm(t) + 1e-12)),
            "pred_rms": p_rms,
            "target_rms": t_rms,
            "pred_to_target_rms_ratio": float(p_rms / max(t_rms, 1e-12)),
        }

    overall = _metrics_for_slice(pred, yt)
    ldims = tuple(int(d) for d in layer_dims) if layer_dims is not None else (branch.layer_dims or (int(yt.shape[1]),))
    per_layer: List[Dict[str, Any]] = []
    if len(ldims) > 1:
        off = 0
        for idx, d in enumerate(ldims):
            lbl = str(layer_labels[idx]) if (layer_labels and idx < len(layer_labels)) else f"layer_{idx}"
            m = _metrics_for_slice(pred[:, off : off + d], yt[:, off : off + d])
            m["layer_label"] = lbl
            per_layer.append(m)
            off += d
    out: Dict[str, Any] = dict(overall)
    if per_layer:
        out["per_layer"] = per_layer
    return out


def assign_source_group_folds(
    records: Sequence[Mapping[str, Any]],
    *,
    n_folds: int = 5,
    seed: int = 20261126,
) -> Dict[str, int]:
    """Deterministic balanced folds strictly by `source_group_id` (zero group leakage across folds)."""
    groups = sorted({str(r["source_group_id"]) for r in records})
    if len(groups) < n_folds:
        raise ValueError(f"Need >= {n_folds} distinct source groups, got {len(groups)}")
    rng = random.Random(seed)
    rng.shuffle(groups)
    grp_fold = {g: i % n_folds for i, g in enumerate(groups)}
    return {str(r["prompt_id"]): grp_fold[str(r["source_group_id"])] for r in records}


def build_group_disjoint_prompt_derangement(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 20261127,
    fold_by_pid: Optional[Mapping[str, int]] = None,
) -> Dict[str, str]:
    """Deterministic bijection `pid -> partner_pid` with different `source_group_id` (and same fold if provided)."""
    buckets: Dict[int, List[Mapping[str, Any]]] = {}
    for r in records:
        pid = str(r["prompt_id"])
        bk = int(fold_by_pid[pid]) if fold_by_pid is not None else 0
        buckets.setdefault(bk, []).append(r)
    mapping: Dict[str, str] = {}
    for bk, recs in sorted(buckets.items()):
        ordered = sorted(recs, key=lambda r: (str(r["source_group_id"]), str(r["prompt_id"])))
        grp_counts: Dict[str, int] = {}
        for r in ordered:
            g = str(r["source_group_id"])
            grp_counts[g] = grp_counts.get(g, 0) + 1
        max_g = max(grp_counts.values())
        n = len(ordered)
        if n <= max_g or len(grp_counts) < 2:
            raise ValueError(f"Cannot build group-disjoint derangement in bucket {bk}: n={n}, max_g={max_g}")
        rng = random.Random(seed + 1009 * bk)
        shift = rng.randint(max_g, n - max_g)
        for i, r in enumerate(ordered):
            partner = ordered[(i + shift) % n]
            if str(partner["prompt_id"]) == str(r["prompt_id"]) or str(partner["source_group_id"]) == str(r["source_group_id"]):
                raise RuntimeError("Derangement invariant violated")
            mapping[str(r["prompt_id"])] = str(partner["prompt_id"])
    return mapping


def verify_prompt_derangement(
    records: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, str],
    *,
    fold_by_pid: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    by_pid = {str(r["prompt_id"]): r for r in records}
    errors: List[str] = []
    if set(mapping.keys()) != set(by_pid.keys()):
        errors.append("mapping keys do not match records")
    if set(mapping.values()) != set(by_pid.keys()):
        errors.append("mapping is not a bijection")
    for pid, partner in mapping.items():
        if pid == partner:
            errors.append(f"fixed point at {pid}")
        if pid in by_pid and partner in by_pid:
            if str(by_pid[pid]["source_group_id"]) == str(by_pid[partner]["source_group_id"]):
                errors.append(f"same source_group_id at {pid}->{partner}")
            if fold_by_pid is not None and int(fold_by_pid[pid]) != int(fold_by_pid[partner]):
                errors.append(f"cross-fold leak at {pid}->{partner}")
    return {"verified": len(errors) == 0, "n_prompts": len(records), "errors": errors[:10]}


def run_inner_group_cv_selection(
    *,
    fit_records: Sequence[Mapping[str, Any]],
    fold_by_pid: Mapping[str, int],
    aligned_x_by_tap_and_pid: Mapping[str, Mapping[str, np.ndarray]],
    aligned_y_by_pid: Mapping[str, np.ndarray],
    layer_dims: Sequence[int],
    tap_specs: Sequence[str] = INKLING_TAP_SPECS,
    solvers: Sequence[str] = ("ridge", "ridge_norm", "ridge_per_layer_norm"),
    ridge_lambdas: Sequence[float] = DEFAULT_RIDGE_LAMBDAS,
    inner_val_fold: int = 0,
    selection_metric: str = "r2",
) -> Dict[str, Any]:
    """Select `(tap_spec, method, ridge_lambda)` strictly on fold `inner_val_fold` of a 5-way
    source-group-disjoint split inside `fit` (zero `val_dev` or `confirm_reserve` touch).
    """
    train_pids = [str(r["prompt_id"]) for r in fit_records if int(fold_by_pid[str(r["prompt_id"])]) != inner_val_fold]
    val_pids = [str(r["prompt_id"]) for r in fit_records if int(fold_by_pid[str(r["prompt_id"])]) == inner_val_fold]
    if not train_pids or not val_pids:
        raise ValueError(f"Empty inner split: train={len(train_pids)}, val={len(val_pids)}")
    train_groups = {str(r["source_group_id"]) for r in fit_records if str(r["prompt_id"]) in set(train_pids)}
    val_groups = {str(r["source_group_id"]) for r in fit_records if str(r["prompt_id"]) in set(val_pids)}
    if train_groups & val_groups:
        raise RuntimeError("Source-group leakage between inner train and inner val folds")

    ldims = tuple(int(d) for d in layer_dims)
    y_va = np.concatenate([aligned_y_by_pid[p] for p in val_pids], axis=0)
    candidates: List[Dict[str, Any]] = []

    for tap in tap_specs:
        sample_x = aligned_x_by_tap_and_pid[tap][train_pids[0]]
        d_donor = int(sample_x.shape[1])
        x_va = np.concatenate([aligned_x_by_tap_and_pid[tap][p] for p in val_pids], axis=0)

        need_ridge = any(s in ("ridge", "ridge_norm", "ridge_per_layer_norm") for s in solvers)
        acc: Optional[StreamingMultiLayerRidgeFP64] = None
        if need_ridge:
            acc = StreamingMultiLayerRidgeFP64(d_donor, ldims)
            for p in train_pids:
                acc.update(aligned_x_by_tap_and_pid[tap][p], aligned_y_by_pid[p])

        for solver in solvers:
            if solver == "procrustes_var_match":
                x_tr = np.concatenate([aligned_x_by_tap_and_pid[tap][p] for p in train_pids], axis=0)
                y_tr = np.concatenate([aligned_y_by_pid[p] for p in train_pids], axis=0)
                br, aud = fit_procrustes_var_match(x_tr, y_tr, layer_dims=ldims)
                vm = evaluate_branch_metrics(x_va, y_va, br, layer_dims=ldims)
                candidates.append(
                    {
                        "tap_spec": tap,
                        "method": solver,
                        "ridge_lambda": None,
                        "inner_val_metrics": vm,
                        "inner_fit_audit": aud,
                    }
                )
            elif solver in ("ridge", "ridge_norm", "ridge_per_layer_norm"):
                assert acc is not None
                if solver == "ridge_per_layer_norm" and len(ldims) <= 1:
                    continue
                norm_mode = {"ridge": "none", "ridge_norm": "global", "ridge_per_layer_norm": "per_layer"}[solver]
                for lam in ridge_lambdas:
                    br, aud = acc.solve(float(lam), norm_mode=norm_mode)
                    vm = evaluate_branch_metrics(x_va, y_va, br, layer_dims=ldims)
                    candidates.append(
                        {
                            "tap_spec": tap,
                            "method": solver,
                            "ridge_lambda": float(lam),
                            "inner_val_metrics": vm,
                            "inner_fit_audit": aud,
                        }
                    )
            else:
                raise ValueError(f"Unknown solver: {solver}")

    def _score(c: Mapping[str, Any]) -> Tuple[float, float]:
        m = c["inner_val_metrics"]
        if selection_metric == "cosine_mean":
            return (float(m["cosine_mean"]), float(m["r2"]))
        return (float(m["r2"]), float(m["cosine_mean"]))

    best = max(candidates, key=_score)
    n_folds = len({int(fold_by_pid[str(r["prompt_id"])]) for r in fit_records})
    return {
        "inner_selection_design": f"single_inner_holdout_fold{int(inner_val_fold)}_of_{n_folds}way_grouped_split",
        "inner_split_ways": int(n_folds),
        "inner_val_fold": int(inner_val_fold),
        "inner_train_prompts": len(train_pids),
        "inner_val_prompts": len(val_pids),
        "inner_train_words": int(sum(aligned_y_by_pid[p].shape[0] for p in train_pids)),
        "inner_val_words": int(y_va.shape[0]),
        "inner_source_groups_disjoint": True,
        "selection_metric": selection_metric,
        "selected": {
            "tap_spec": best["tap_spec"],
            "method": best["method"],
            "ridge_lambda": best["ridge_lambda"],
            "inner_val_metrics": best["inner_val_metrics"],
        },
        "all_candidates": candidates,
    }


def save_platos_key_v2(
    artifact: PlatosKeyArtifact,
    out_path: Path,
    *,
    save_dtype: str = "float16",
) -> Dict[str, Any]:
    """Serialize `PlatosKeyArtifact` to universal `platos_key_v2` `.safetensors` (v1-compatible primary tensors)."""
    from safetensors.numpy import save_file

    dtype_map = {"float16": np.float16, "float32": np.float32}
    if save_dtype not in dtype_map:
        raise ValueError(f"Unsupported save_dtype {save_dtype!r}")
    np_dt = dtype_map[save_dtype]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tensors: Dict[str, np.ndarray] = {}

    def _pack_branch(prefix: str, br: PlatosKeyBranch) -> None:
        tensors[f"{prefix}R"] = np.ascontiguousarray(br.R.astype(np_dt))
        tensors[f"{prefix}scale"] = np.ascontiguousarray(np.array([br.scale], dtype=np_dt))
        tensors[f"{prefix}donor_mean"] = np.ascontiguousarray(br.donor_mean.astype(np_dt))
        tensors[f"{prefix}target_mean"] = np.ascontiguousarray(br.target_mean.astype(np_dt))
        if br.per_layer_scale is not None:
            tensors[f"{prefix}per_layer_scale"] = np.ascontiguousarray(br.per_layer_scale.astype(np.float32))

    _pack_branch("", artifact.primary)
    if artifact.pooled is not None:
        _pack_branch("pooled.", artifact.pooled)
    for bname, bobj in sorted(artifact.extra_branches.items()):
        _pack_branch(f"branch.{bname}.", bobj)
    if artifact.prefix_embeds is not None:
        arr_p = np.asarray(artifact.prefix_embeds, dtype=np_dt)
        if arr_p.ndim != 2 or arr_p.shape[1] != artifact.d_target:
            raise ValueError(f"prefix_embeds shape {arr_p.shape} incompatible with d_target={artifact.d_target}")
        tensors["prefix_embeds"] = np.ascontiguousarray(arr_p)
    if artifact.suffix_embeds is not None:
        arr_s = np.asarray(artifact.suffix_embeds, dtype=np_dt)
        if arr_s.ndim != 2 or arr_s.shape[1] != artifact.d_target:
            raise ValueError(f"suffix_embeds shape {arr_s.shape} incompatible with d_target={artifact.d_target}")
        tensors["suffix_embeds"] = np.ascontiguousarray(arr_s)
    if artifact.sink_embedding is not None:
        arr_k = np.asarray(artifact.sink_embedding, dtype=np_dt).reshape(artifact.d_target)
        tensors["sink_embedding"] = np.ascontiguousarray(arr_k)

    meta: Dict[str, str] = {str(k): (v if isinstance(v, str) else json.dumps(v)) for k, v in artifact.metadata.items()}
    meta.setdefault("format", FORMAT_V2)
    meta.setdefault("schema_version", SCHEMA_VERSION_V2)
    meta["save_dtype"] = save_dtype
    meta["method"] = artifact.primary.method
    meta["ridge_lambda"] = "null" if artifact.primary.ridge_lambda is None else str(float(artifact.primary.ridge_lambda))
    meta["d_donor"] = str(artifact.primary.d_donor)
    meta["d_target"] = str(artifact.primary.d_target)
    meta["scale"] = f"{artifact.primary.scale:.8f}"
    meta["layer_dims_json"] = json.dumps(list(artifact.primary.layer_dims or (artifact.primary.d_target,)))
    meta["has_per_layer_scale"] = "true" if artifact.primary.per_layer_scale is not None else "false"
    meta["has_pooled_branch"] = "true" if artifact.pooled is not None else "false"
    meta["pooled_dim"] = str(artifact.pooled.d_target if artifact.pooled is not None else 0)
    meta["has_prefix_embeds"] = "true" if artifact.prefix_embeds is not None else "false"
    meta["has_suffix_embeds"] = "true" if artifact.suffix_embeds is not None else "false"
    meta["has_sink_embedding"] = "true" if artifact.sink_embedding is not None else "false"
    meta["extra_branch_names_json"] = json.dumps(sorted(artifact.extra_branches.keys()))
    meta.setdefault("forbidden_coord_d2027_zero_masked", "true")
    meta.setdefault("donor_corpus_sha256", FROZEN_DONOR_ARTIFACTS["prompt_manifest_sha256"])
    meta.setdefault("donor_fit_valdev_bin_sha256", FROZEN_DONOR_ARTIFACTS["fit_valdev_800"]["bin_sha256"])
    meta.setdefault("donor_confirm160_bin_sha256", FROZEN_DONOR_ARTIFACTS["confirm160_sealed"]["bin_sha256"])

    save_file(tensors, str(out_path), metadata=meta)
    nbytes, digest = sha256_file(out_path)
    return {
        "path": str(out_path),
        "bytes": int(nbytes),
        "sha256": digest,
        "format": meta["format"],
        "schema_version": meta["schema_version"],
        "save_dtype": save_dtype,
        "d_donor": artifact.primary.d_donor,
        "d_target": artifact.primary.d_target,
        "has_pooled_branch": artifact.pooled is not None,
        "pooled_dim": artifact.pooled.d_target if artifact.pooled is not None else 0,
        "has_prefix_embeds": artifact.prefix_embeds is not None,
        "has_suffix_embeds": artifact.suffix_embeds is not None,
        "has_sink_embedding": artifact.sink_embedding is not None,
        "extra_branches": sorted(artifact.extra_branches.keys()),
    }


def load_platos_key_v2(in_path: Path) -> PlatosKeyArtifact:
    """Load `platos_key_v2` OR legacy `platos_key_v1` `.safetensors` into `PlatosKeyArtifact`."""
    from safetensors import safe_open
    from safetensors.numpy import load_file

    tensors = load_file(str(in_path))
    with safe_open(str(in_path), framework="numpy") as f:
        meta = dict(f.metadata() or {})

    def _unpack_branch(prefix: str, default_method: str, default_lam: Optional[float], ldims: Optional[Tuple[int, ...]]) -> PlatosKeyBranch:
        for req in ("R", "scale", "donor_mean", "target_mean"):
            k = f"{prefix}{req}"
            if k not in tensors:
                raise KeyError(f"Missing required tensor {k!r} in {in_path}")
        pls_key = f"{prefix}per_layer_scale"
        pls = np.asarray(tensors[pls_key], dtype=np.float64) if pls_key in tensors else None
        r_mat = np.asarray(tensors[f"{prefix}R"], dtype=np.float64)
        eff_ldims = ldims if ldims is not None else (int(r_mat.shape[1]),)
        return PlatosKeyBranch(
            R=r_mat,
            scale=float(np.asarray(tensors[f"{prefix}scale"], dtype=np.float64).reshape(-1)[0]),
            donor_mean=np.asarray(tensors[f"{prefix}donor_mean"], dtype=np.float64),
            target_mean=np.asarray(tensors[f"{prefix}target_mean"], dtype=np.float64),
            method=default_method,
            ridge_lambda=default_lam,
            per_layer_scale=pls,
            layer_dims=eff_ldims,
        )

    lam_raw = meta.get("ridge_lambda")
    lam = float(lam_raw) if lam_raw not in (None, "", "None", "null") else None
    ldims_raw = meta.get("layer_dims_json")
    ldims = tuple(int(x) for x in json.loads(ldims_raw)) if ldims_raw else None
    primary = _unpack_branch("", str(meta.get("method", "unknown")), lam, ldims)

    pooled: Optional[PlatosKeyBranch] = None
    if "pooled.R" in tensors:
        pooled = _unpack_branch("pooled.", str(meta.get("pooled_method", primary.method)), lam, None)
    elif "pooled_R" in tensors:
        # Alias support for underscore pooled_* layout
        pooled = _unpack_branch("pooled_", str(meta.get("pooled_method", primary.method)), lam, None)

    extra: Dict[str, PlatosKeyBranch] = {}
    extra_names_raw = meta.get("extra_branch_names_json")
    extra_names: List[str] = list(json.loads(extra_names_raw)) if extra_names_raw else []
    if not extra_names:
        for k in tensors:
            if k.startswith("branch.") and k.endswith(".R"):
                extra_names.append(k[len("branch.") : -len(".R")])
    for name in sorted(set(extra_names)):
        extra[name] = _unpack_branch(f"branch.{name}.", primary.method, lam, None)

    prefix_embeds = np.asarray(tensors["prefix_embeds"], dtype=np.float64) if "prefix_embeds" in tensors else None
    suffix_embeds = np.asarray(tensors["suffix_embeds"], dtype=np.float64) if "suffix_embeds" in tensors else None
    sink_embedding = np.asarray(tensors["sink_embedding"], dtype=np.float64) if "sink_embedding" in tensors else None

    return PlatosKeyArtifact(
        primary=primary,
        pooled=pooled,
        extra_branches=extra,
        prefix_embeds=prefix_embeds,
        suffix_embeds=suffix_embeds,
        sink_embedding=sink_embedding,
        metadata=meta,
    )


def pack_voltron_cond_packet(
    embeds_seq_dim: np.ndarray,
    *,
    target_id: str = "z_image_turbo",
    target_encoder_id: str = "Qwen/Qwen3-4B",
    source_layer: int = -2,
    key_version_sha256: str,
    donor_tap_spec: str,
    valid_len: Optional[int] = None,
    max_packet_bytes: int = VOLTRON_PACKET_MAX_BYTES,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Serialize `[seq_len, dim]` conditioning tensor into Voltron Android #914 LAN packet (`VPKE` v1).

    Wire layout (little-endian):
      [0:4]   magic b"VPKE"
      [4:6]   uint16 version = 1
      [6:10]  uint32 header_json_len
      [10:10+H] UTF-8 canonical JSON header
      [10+H:] contiguous row-major FP16 `<f2` payload bytes
    Enforces fail-closed `total_bytes <= max_packet_bytes` (default 2 MiB = 2,097,152 B).
    """
    arr = np.asarray(embeds_seq_dim)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] < 1:
        raise ValueError(f"Expected [seq_len, dim] tensor, got {arr.shape}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("Non-finite values in Voltron conditioning tensor")
    arr_f16 = np.ascontiguousarray(arr.astype("<f2"))
    if not bool(np.all(np.isfinite(arr_f16))):
        raise ValueError("FP16 overflow/non-finite values in Voltron conditioning tensor")
    seq_len, dim = int(arr_f16.shape[0]), int(arr_f16.shape[1])
    vlen = seq_len if valid_len is None else int(valid_len)
    if vlen < 1 or vlen > seq_len:
        raise ValueError(f"Invalid valid_len={vlen} for seq_len={seq_len}")
    payload = arr_f16.tobytes(order="C")
    payload_sha = hashlib.sha256(payload).hexdigest()

    header: Dict[str, Any] = {
        "schema_version": VOLTRON_PACKET_SCHEMA,
        "target_id": str(target_id),
        "target_encoder_id": str(target_encoder_id),
        "source_layer": int(source_layer),
        "pre_cap_embedder": True,
        "dtype": "fp16",
        "layout": "row_major",
        "shape": [seq_len, dim],
        "seq_len": seq_len,
        "valid_len": vlen,
        "dim": dim,
        "donor_tap_spec": str(donor_tap_spec),
        "donor_forbidden_coord_d2027_zero_masked": True,
        "key_version_sha256": str(key_version_sha256),
        "payload_nbytes": len(payload),
        "payload_sha256": payload_sha,
    }
    if extra_meta:
        for k, v in extra_meta.items():
            if k not in header:
                header[str(k)] = v
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    packet = VOLTRON_PACKET_MAGIC + struct.pack("<HI", VOLTRON_PACKET_VERSION, len(header_bytes)) + header_bytes + payload
    if len(packet) > int(max_packet_bytes):
        raise ValueError(
            f"Voltron packet size {len(packet)} B exceeds {int(max_packet_bytes)} B cap "
            f"(seq_len={seq_len}, dim={dim}, max_seq_len≈{(int(max_packet_bytes) - 10 - len(header_bytes)) // (dim * 2)})"
        )
    packet_sha = hashlib.sha256(packet).hexdigest()
    audit = {
        "total_bytes": len(packet),
        "header_bytes": len(header_bytes),
        "payload_bytes": len(payload),
        "packet_sha256": packet_sha,
        "payload_sha256": payload_sha,
        "seq_len": seq_len,
        "valid_len": vlen,
        "dim": dim,
        "within_2mib_cap": len(packet) <= VOLTRON_PACKET_MAX_BYTES,
    }
    return packet, audit


def unpack_and_verify_voltron_cond_packet(
    packet: bytes,
    *,
    max_packet_bytes: int = VOLTRON_PACKET_MAX_BYTES,
    expected_dim: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fail-closed decoder + SHA-256 verifier for `VPKE` v1 LAN conditioning packets."""
    if len(packet) > int(max_packet_bytes):
        raise ValueError(f"Packet length {len(packet)} exceeds cap {int(max_packet_bytes)}")
    if len(packet) < 10 or packet[:4] != VOLTRON_PACKET_MAGIC:
        raise ValueError("Invalid VPKE packet magic")
    ver, hlen = struct.unpack("<HI", packet[4:10])
    if ver != VOLTRON_PACKET_VERSION:
        raise ValueError(f"Unsupported VPKE version {ver}")
    if 10 + hlen > len(packet):
        raise ValueError("Truncated VPKE header")
    header = json.loads(packet[10 : 10 + hlen].decode("utf-8"))
    payload = packet[10 + hlen :]
    if len(payload) != int(header["payload_nbytes"]):
        raise ValueError(f"Payload length mismatch: {len(payload)} != {header['payload_nbytes']}")
    obs_sha = hashlib.sha256(payload).hexdigest()
    if obs_sha != str(header["payload_sha256"]):
        raise ValueError(f"Payload SHA-256 mismatch: {obs_sha} != {header['payload_sha256']}")
    seq_len, dim = int(header["seq_len"]), int(header["dim"])
    if expected_dim is not None and dim != int(expected_dim):
        raise ValueError(f"Expected dim={expected_dim}, got {dim}")
    if len(payload) != seq_len * dim * 2:
        raise ValueError(f"Payload byte count {len(payload)} != {seq_len}*{dim}*2")
    arr = np.frombuffer(payload, dtype="<f2").reshape(seq_len, dim).copy()
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("Decoded VPKE payload contains non-finite values")
    return arr, header


AUTHORITATIVE_CONFIRM160_LOCK_ROOT = Path("/mnt/storage16tb/omni943_wave_a/confirm160_canonical_locks")


def default_canonical_confirm160_lock_root() -> Path:
    """Single authoritative host/store lock root (Quincy /mnt/storage16tb/omni943_wave_a/confirm160_canonical_locks).

    Fail-closed on any host where the single authoritative store is not mounted (never splits across two hosts).
    """
    if not AUTHORITATIVE_CONFIRM160_LOCK_ROOT.parent.is_dir():
        raise RuntimeError(
            f"Authoritative confirm160 lock store {AUTHORITATIVE_CONFIRM160_LOCK_ROOT.parent} is not mounted on this host; "
            f"confirm160 extraction and evaluation MUST run strictly against the single authoritative Quincy store."
        )
    return AUTHORITATIVE_CONFIRM160_LOCK_ROOT


def acquire_canonical_confirm160_lock(
    *,
    target_id: str,
    stage: str,  # "extract_confirm160" | "eval_confirm160"
    primary_key_path: Path,
    expected_primary_key_sha256: str,
    b0_key_path: Path,
    expected_b0_key_sha256: str,
    _test_lock_root: Optional[Path] = None,
) -> Path:
    """Bind confirm160 teacher extraction AND eval atomically to ONE key pair on ONE authoritative host/store.

    Race-free protocol:
      1. Verify approved primary & B0 key file SHA-256s BEFORE touching any confirm160 bytes.
      2. Atomically create `key_pair_binding.json` via `O_CREAT | O_EXCL` FIRST (binding the single
         approved key pair before ANY stage receipt exists). If it already exists (e.g. concurrent
         stage or second stage), read it and fail closed on any key digest mismatch.
      3. Atomically create `<stage>_attempt_receipt.json` via `O_CREAT | O_EXCL` (blocks reruns/overwrites).
    """
    import os
    import time as _time

    if stage not in ("extract_confirm160", "eval_confirm160"):
        raise ValueError(f"Unknown confirm160 lock stage: {stage!r}")
    if not expected_primary_key_sha256 or len(str(expected_primary_key_sha256)) != 64:
        raise ValueError("64-char --expected-primary-key-sha256 required before confirm160 access")
    if not expected_b0_key_sha256 or len(str(expected_b0_key_sha256)) != 64:
        raise ValueError("64-char --expected-b0-key-sha256 required before confirm160 access")
    if not primary_key_path.is_file() or not b0_key_path.is_file():
        raise FileNotFoundError(f"Approved frozen keys missing: {primary_key_path} / {b0_key_path}")

    _, obs_p = sha256_file(primary_key_path)
    _, obs_b = sha256_file(b0_key_path)
    if obs_p != str(expected_primary_key_sha256):
        raise ValueError(f"Primary key SHA mismatch before confirm160: {obs_p} != {expected_primary_key_sha256}")
    if obs_b != str(expected_b0_key_sha256):
        raise ValueError(f"B0 key SHA mismatch before confirm160: {obs_b} != {expected_b0_key_sha256}")

    lock_dir = (_test_lock_root or default_canonical_confirm160_lock_root()) / str(target_id)
    lock_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 (atomic, race-free): bind the single approved key pair BEFORE creating any stage receipt!
    binding_path = lock_dir / "key_pair_binding.json"
    binding_tmp = lock_dir / f".key_pair_binding.{os.getpid()}.{_time.time_ns()}.tmp"
    binding_payload = (
        json.dumps(
            {
                "schema_version": "omni-943-confirm160-key-binding/v1",
                "target_id": str(target_id),
                "bound_utc_epoch": _time.time(),
                "bound_by_stage": stage,
                "approved_primary_key_sha256": obs_p,
                "approved_b0_key_sha256": obs_b,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    # Write complete payload to private temp file, fsync, then atomically hard-link to binding_path (POSIX atomic O_EXCL + never half-written!)
    tfd = os.open(str(binding_tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(tfd, binding_payload)
        os.fsync(tfd)
    finally:
        os.close(tfd)
    try:
        os.link(str(binding_tmp), str(binding_path))
    except FileExistsError:
        pass
    finally:
        try:
            os.unlink(str(binding_tmp))
        except FileNotFoundError:
            pass

    # Verify binding_path matches our approved key pair (whether we created it or a concurrent/prior stage did!)
    bound_rec = json.loads(binding_path.read_text(encoding="utf-8"))
    if (
        str(bound_rec.get("approved_primary_key_sha256")) != obs_p
        or str(bound_rec.get("approved_b0_key_sha256")) != obs_b
    ):
        raise RuntimeError(
            f"CONFIRM160 ATOMIC KEY-PAIR BINDING MISMATCH for {target_id}: "
            f"bound=({bound_rec.get('approved_primary_key_sha256')}, {bound_rec.get('approved_b0_key_sha256')}) "
            f"!= requested=({obs_p}, {obs_b})!"
        )

    # Step 2 (atomic one-shot stage receipt): O_CREAT | O_EXCL blocks reruns/overwrites
    receipt_path = lock_dir / f"{stage}_attempt_receipt.json"
    receipt_tmp = lock_dir / f".{stage}.{os.getpid()}.{_time.time_ns()}.tmp"
    payload = (
        json.dumps(
            {
                "schema_version": "omni-943-confirm160-canonical-receipt/v3",
                "target_id": str(target_id),
                "stage": stage,
                "created_utc_epoch": _time.time(),
                "key_pair_binding_path": str(binding_path),
                "approved_primary_key_path": str(primary_key_path),
                "approved_primary_key_sha256": obs_p,
                "approved_b0_key_path": str(b0_key_path),
                "approved_b0_key_sha256": obs_b,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    rfd = os.open(str(receipt_tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(rfd, payload)
        os.fsync(rfd)
    finally:
        os.close(rfd)
    try:
        os.link(str(receipt_tmp), str(receipt_path))
    except FileExistsError as exc:
        raise RuntimeError(
            f"SEALED CONFIRM160 ONE-SHOT VIOLATION: canonical receipt already exists at {receipt_path}; "
            f"rerunning or bypassing via --out-dir is forbidden!"
        ) from exc
    finally:
        try:
            os.unlink(str(receipt_tmp))
        except FileNotFoundError:
            pass
    return receipt_path

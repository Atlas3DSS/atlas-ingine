"""Drop-in `diffusers` PlatosKeyConditioning loader (zero upstream text encoder loaded).

Projects native Inkling-2-Mini hidden states `[L, 4, 2304]` directly through a frozen
`platos_key_v2` `.safetensors` key into ready-to-pass `diffusers` pipeline kwargs for
every Wave A and Wave B target (`QwenImage21Pipeline`, `Flux3DistilledModularPipeline`,
`ZImagePipeline`, `Flux2KleinPipeline`, `Flux2Pipeline`, `MOSSImagePipeline`,
`FluxPipeline`, `StableDiffusionXLPipeline`), PLUS Voltron Android #914 `<2 MiB` LAN
wire packet export for `Tongyi-MAI/Z-Image-Turbo`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from platos_key_schema import (
    INKLING_FORBIDDEN_COORD,
    PlatosKeyArtifact,
    VOLTRON_PACKET_MAX_BYTES,
    extract_tap_matrix,
    load_platos_key_v2,
    pack_voltron_cond_packet,
    sha256_file,
    verify_donor_d2027_zero,
)
from target_registry import TARGET_REGISTRY, TargetEncoderSpec, get_target_spec


def _apply_padding_policy(
    mapped_ld: np.ndarray,
    spec: TargetEncoderSpec,
    *,
    pad_value: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply target DiT padding policy to `[L, D]` embeddings; returns `(padded [L_out, D], valid_mask [L_out] bool)`."""
    seq_len, dim = int(mapped_ld.shape[0]), int(mapped_ld.shape[1])
    if spec.max_seq_len > 0 and seq_len > spec.max_seq_len:
        mapped_ld = mapped_ld[: spec.max_seq_len]
        seq_len = spec.max_seq_len

    target_len = seq_len
    if spec.padding_policy == "multiple_of_80":
        mult = max(1, int(spec.pad_multiple))
        target_len = max(spec.min_seq_len, ((seq_len + mult - 1) // mult) * mult)
    elif spec.padding_policy == "min_length_512":
        target_len = max(int(spec.min_seq_len), seq_len)
    elif spec.padding_policy == "fixed_77":
        target_len = 77
    elif spec.padding_policy == "unpadded":
        target_len = seq_len
    else:
        raise ValueError(f"Unknown padding_policy: {spec.padding_policy!r}")

    mask = np.zeros((target_len,), dtype=bool)
    mask[:seq_len] = True
    if target_len == seq_len:
        return np.ascontiguousarray(mapped_ld), mask

    out = np.full((target_len, dim), float(pad_value), dtype=mapped_ld.dtype)
    out[:seq_len] = mapped_ld
    return out, mask


class PlatosKeyConditioning:
    """Universal drop-in conditioning projector replacing upstream text encoders in diffusers & ComfyUI."""

    def __init__(
        self,
        key_path: Path | str,
        *,
        target_id: Optional[str] = None,
        forbidden_coord: int = INKLING_FORBIDDEN_COORD,
    ) -> None:
        self.key_path = Path(key_path)
        self.artifact: PlatosKeyArtifact = load_platos_key_v2(self.key_path)
        _, self.key_sha256 = sha256_file(self.key_path)
        resolved_tid = target_id or self.artifact.target_id
        self.spec: Optional[TargetEncoderSpec] = (
            get_target_spec(resolved_tid) if resolved_tid in TARGET_REGISTRY else None
        )
        self.target_id = resolved_tid
        self.forbidden_coord = int(forbidden_coord)
        if self.spec is not None and self.artifact.d_target != self.spec.dit_conditioning_dim:
            raise ValueError(
                f"Key d_target={self.artifact.d_target} != registry dit_conditioning_dim="
                f"{self.spec.dit_conditioning_dim} for {self.target_id!r}"
            )

    def project_numpy(
        self,
        donor_taps_l4h: np.ndarray,
        *,
        apply_padding: bool = True,
        include_bookends: bool = False,
        apply_sink: bool = True,
        out_dtype: Any = np.float32,
    ) -> Dict[str, Any]:
        """Project Inkling-2-Mini taps `[L, 4, 2304]` -> numpy conditioning arrays (fail-closed on d_2027)."""
        verify_donor_d2027_zero(donor_taps_l4h, forbidden_coord=self.forbidden_coord)
        feat = extract_tap_matrix(donor_taps_l4h, self.artifact.tap_spec, check_d2027=False)
        mapped = self.artifact.apply_primary(
            feat,
            out_dtype=out_dtype,
            include_bookends=include_bookends,
            apply_sink=apply_sink,
        )  # [L(+bookends), d_target]
        if not bool(np.all(np.isfinite(mapped))):
            raise ValueError(f"Non-finite primary projection produced for {self.target_id!r}")

        valid_len = int(mapped.shape[0])
        if apply_padding and self.spec is not None:
            padded, valid_mask = _apply_padding_policy(mapped, self.spec)
        else:
            padded = np.ascontiguousarray(mapped)
            valid_mask = np.ones((valid_len,), dtype=bool)

        pooled_vec: Optional[np.ndarray] = None
        if self.artifact.pooled is not None:
            pooled_vec = self.artifact.apply_pooled(feat, out_dtype=out_dtype)  # [d_pooled]
            if not bool(np.all(np.isfinite(pooled_vec))):
                raise ValueError(f"Non-finite pooled projection produced for {self.target_id!r}")

        return {
            "target_id": self.target_id,
            "prompt_embeds": padded[None, :, :],  # [1, L_out, d_target]
            "prompt_embeds_unbatched": padded,  # [L_out, d_target]
            "valid_mask": valid_mask[None, :],  # [1, L_out] bool
            "valid_len": valid_len,
            "seq_len": int(padded.shape[0]),
            "dim": int(padded.shape[1]),
            "pooled_prompt_embeds": pooled_vec[None, :] if pooled_vec is not None else None,
            "key_sha256": self.key_sha256,
            "tap_spec": self.artifact.tap_spec,
        }

    def build_diffusers_kwargs(
        self,
        donor_taps_l4h: np.ndarray,
        *,
        device: Any = "cpu",
        dtype: Any = None,
        apply_padding: bool = True,
    ) -> Dict[str, Any]:
        """Return exact keyword arguments to pass directly into `pipe(**kwargs)` with `prompt=None`."""
        import torch

        torch_dtype = dtype if dtype is not None else torch.bfloat16
        proj = self.project_numpy(donor_taps_l4h, apply_padding=apply_padding, out_dtype=np.float32)
        embeds_t = torch.from_numpy(proj["prompt_embeds"]).to(device=device, dtype=torch_dtype)
        mask_t = torch.from_numpy(proj["valid_mask"]).to(device=device, dtype=torch.bool)

        if self.spec is None:
            out: Dict[str, Any] = {"prompt_embeds": embeds_t}
            if proj["pooled_prompt_embeds"] is not None:
                out["pooled_prompt_embeds"] = torch.from_numpy(proj["pooled_prompt_embeds"]).to(
                    device=device, dtype=torch_dtype
                )
            return out

        out = {}
        if self.spec.diffusers_expects_list:
            # ZImagePipeline expects `prompt_embeds: List[Tensor[L_i, 2560]]`
            out[self.spec.diffusers_cond_kwarg] = [embeds_t[0]]
        else:
            out[self.spec.diffusers_cond_kwarg] = embeds_t

        if self.spec.diffusers_mask_kwarg:
            if self.spec.diffusers_mask_kwarg == "prompt_embeds_mask":
                out[self.spec.diffusers_mask_kwarg] = mask_t.to(dtype=torch.long)
            else:
                out[self.spec.diffusers_mask_kwarg] = mask_t

        if proj["pooled_prompt_embeds"] is not None:
            out["pooled_prompt_embeds"] = torch.from_numpy(proj["pooled_prompt_embeds"]).to(
                device=device, dtype=torch_dtype
            )
        return out

    def build_voltron_mobile_packet(
        self,
        donor_taps_l4h: np.ndarray,
        *,
        max_packet_bytes: int = VOLTRON_PACKET_MAX_BYTES,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Export a verified `VPKE` v1 FP16 packet (`<= 2 MiB`) for Voltron Android #914 `/v1/voltron/media`."""
        proj = self.project_numpy(donor_taps_l4h, apply_padding=False, out_dtype=np.float32)
        src_layer = int(self.spec.tap_layers[0]) if (self.spec and self.spec.tap_layers) else -2
        if self.target_id == "z_image_turbo":
            src_layer = -2
        enc_id = self.spec.encoder_repo if self.spec else str(self.artifact.metadata.get("target_encoder_id", "unknown"))
        return pack_voltron_cond_packet(
            proj["prompt_embeds_unbatched"],
            target_id=self.target_id,
            target_encoder_id=enc_id,
            source_layer=src_layer,
            key_version_sha256=self.key_sha256,
            donor_tap_spec=self.artifact.tap_spec,
            valid_len=proj["valid_len"],
            max_packet_bytes=max_packet_bytes,
        )

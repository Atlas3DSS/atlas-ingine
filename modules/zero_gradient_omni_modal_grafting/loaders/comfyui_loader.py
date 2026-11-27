"""Drop-in ComfyUI `PlatosKeyConditioning` loader & custom node (zero upstream text encoder loaded).

Produces exact ComfyUI `CONDITIONING` (`[[cond_tensor, extra_dict]]`) compatible with:
  - Krea-2 (`[1, L, 30720]` -> `txtfusion` unpack)
  - Qwen-Image-2.1 (`[1, L, 4096]`)
  - FLUX.3-image-klein (`[1, L_pad80, 20480]`, with `attention_mask`)
  - Z-Image / Z-Image-Turbo (`[1, L, 2560]`, pre-`cap_embedder`)
  - FLUX.2-klein-4B / 9B / FLUX.2-dev (`[1, L_pad512, K*H]`)
  - FLUX.1-schnell / dev & SDXL-base / turbo (`[1, L, D_seq]` + `{"pooled_output": [1, D_pool]}`)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from diffusers_loader import PlatosKeyConditioning, _apply_padding_policy
from target_registry import TARGET_REGISTRY, TargetEncoderSpec


def select_target_bookends(
    loader: PlatosKeyConditioning,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return exact `(active_prefix, active_suffix)` matching each target's post-drop native DiT contract.

    - `krea2` (`drop_system_and_user_header_krea2`): native DiT drops everything before `{prompt}` -> `prefix=None`, keep `suffix`.
    - `qwen_image_21` (`drop_system_turn_qwen_image21`): native DiT drops system turn and keeps `<|im_start|>user\\n`
      (last 3 tokens of fit-mean prefix) + `{prompt}` + `suffix`.
    - All other targets: keep full `prefix_embeds` and `suffix_embeds`.
    """
    pre = loader.artifact.prefix_embeds
    suf = loader.artifact.suffix_embeds
    rule = loader.spec.prompt_prefix_drop_rule if loader.spec is not None else ""
    if rule == "drop_system_and_user_header_krea2":
        pre = None
    elif rule == "drop_system_turn_qwen_image21" and pre is not None and pre.shape[0] > 3:
        pre = pre[-3:]
    return pre, suf


def build_comfyui_conditioning(
    loader: PlatosKeyConditioning,
    donor_taps_l4h: np.ndarray,
    *,
    device: Any = "cpu",
    dtype: Any = None,
    apply_padding: bool = True,
    include_bookends: bool = False,
) -> List[List[Any]]:
    """Build standard ComfyUI `CONDITIONING` list `[[cond_tensor, extra_dict]]` without loading any text encoder."""
    import torch

    torch_dtype = dtype if dtype is not None else torch.bfloat16
    if include_bookends:
        proj = loader.project_numpy(
            donor_taps_l4h,
            apply_padding=False,
            include_bookends=False,
            out_dtype=np.float32,
        )
        parts: List[np.ndarray] = []
        pre, suf = select_target_bookends(loader)
        if pre is not None and pre.shape[0] > 0:
            parts.append(np.asarray(pre, dtype=np.float32))
        parts.append(np.asarray(proj["prompt_embeds_unbatched"], dtype=np.float32))
        if suf is not None and suf.shape[0] > 0:
            parts.append(np.asarray(suf, dtype=np.float32))
        seq = np.concatenate(parts, axis=0)
        valid_len = int(seq.shape[0])
        if apply_padding and loader.spec is not None:
            padded, valid_mask = _apply_padding_policy(seq, loader.spec)
        else:
            padded = np.ascontiguousarray(seq)
            valid_mask = np.ones((valid_len,), dtype=bool)
        proj["prompt_embeds"] = padded[None, :, :]
        proj["prompt_embeds_unbatched"] = padded
        proj["valid_mask"] = valid_mask[None, :]
        proj["valid_len"] = valid_len
        proj["seq_len"] = int(padded.shape[0])
    else:
        proj = loader.project_numpy(donor_taps_l4h, apply_padding=apply_padding, out_dtype=np.float32)

    cond_t = torch.from_numpy(proj["prompt_embeds"]).to(device=device, dtype=torch_dtype).contiguous()

    extra: Dict[str, Any] = {}
    if proj["pooled_prompt_embeds"] is not None:
        extra["pooled_output"] = (
            torch.from_numpy(proj["pooled_prompt_embeds"]).to(device=device, dtype=torch_dtype).contiguous()
        )
    else:
        extra["pooled_output"] = None

    if loader.spec is not None and "attention_mask" in loader.spec.comfy_extra_dict_keys:
        extra["attention_mask"] = torch.from_numpy(proj["valid_mask"]).to(device=device, dtype=torch.long).contiguous()

    extra["platos_key_meta"] = {
        "target_id": proj["target_id"],
        "key_sha256": proj["key_sha256"],
        "tap_spec": proj["tap_spec"],
        "valid_len": proj["valid_len"],
        "seq_len": proj["seq_len"],
        "dim": proj["dim"],
        "include_bookends": bool(include_bookends),
        "upstream_text_encoder_loaded": False,
    }
    return [[cond_t, extra]]


def build_comfyui_empty_negative_conditioning(
    loader: PlatosKeyConditioning,
    *,
    device: Any = "cpu",
    dtype: Any = None,
    apply_padding: bool = True,
) -> List[List[Any]]:
    """Build zero-encoder empty-prompt (`""`) ComfyUI negative `CONDITIONING` for Base CFG from fit-only bookends."""
    import torch

    torch_dtype = dtype if dtype is not None else torch.bfloat16
    pre, suf = select_target_bookends(loader)
    parts: List[np.ndarray] = []
    if pre is not None and pre.shape[0] > 0:
        parts.append(np.asarray(pre, dtype=np.float32))
    if suf is not None and suf.shape[0] > 0:
        parts.append(np.asarray(suf, dtype=np.float32))
    if parts:
        seq = np.concatenate(parts, axis=0)
    else:
        seq = np.zeros((1, loader.artifact.d_target), dtype=np.float32)

    if apply_padding and loader.spec is not None:
        padded, valid_mask = _apply_padding_policy(seq, loader.spec)
    else:
        padded = np.ascontiguousarray(seq)
        valid_mask = np.ones((seq.shape[0],), dtype=bool)

    cond_t = torch.from_numpy(padded[None, :, :]).to(device=device, dtype=torch_dtype).contiguous()
    extra: Dict[str, Any] = {"pooled_output": None}
    if loader.spec is not None and "attention_mask" in loader.spec.comfy_extra_dict_keys:
        extra["attention_mask"] = torch.from_numpy(valid_mask[None, :]).to(device=device, dtype=torch.long).contiguous()
    return [[cond_t, extra]]


class PlatosKeyConditioningNode:
    """ComfyUI custom node: projects Inkling-2-Mini taps `[L, 4, 2304]` through a Plato's Key `.safetensors`."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "key_path": ("STRING", {"default": "", "multiline": False}),
                "inkling_taps_npy_path": ("STRING", {"default": "", "multiline": False}),
                "target_id": (["auto"] + sorted(TARGET_REGISTRY.keys()), {"default": "auto"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "project"
    CATEGORY = "Atlas/PlatosKey"

    def project(
        self,
        key_path: str,
        inkling_taps_npy_path: str,
        target_id: str = "auto",
        dtype: str = "bfloat16",
    ) -> Tuple[List[List[Any]]]:
        import torch

        dt_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        tid = None if target_id == "auto" else target_id
        loader = PlatosKeyConditioning(Path(key_path), target_id=tid)
        taps = np.load(inkling_taps_npy_path)
        cond = build_comfyui_conditioning(loader, taps, device="cpu", dtype=dt_map[dtype])
        return (cond,)


NODE_CLASS_MAPPINGS = {
    "PlatosKeyConditioning": PlatosKeyConditioningNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PlatosKeyConditioning": "Plato's Key Conditioning (Inkling-2-Mini)",
}

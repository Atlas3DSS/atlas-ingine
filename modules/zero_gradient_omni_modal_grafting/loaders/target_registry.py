"""Authoritative target registry for #943 Universal Nov-2026 Base/Turbo Image Plato's Key Sweep.

Strict scope: canonical upstream Base + Turbo checkpoints ONLY (zero community finetunes;
no HiDream / Sana-Sprint; users bring their own finetunes & quants on top).
Every HF repo commit SHA is pinned to its exact verified full 40-character git revision.

Wave A (owned by mecha-monkey):
  1. `z_image_turbo`      — Tongyi-MAI/Z-Image & Z-Image-Turbo (6B S3-DiT, Voltron mobile target), Qwen3-4B
  2. `flux2_klein_4b`     — black-forest-labs/FLUX.2-klein-base-4B & FLUX.2-klein-4B, Qwen3-4B
  3. `krea2`              — krea/Krea-2-Raw & Turbo via official Comfy-Org/Krea-2 mirror, Qwen3-VL-4B-Instruct
  4. `flux3_image_klein`  — black-forest-labs/flux-3-image-klein-base & flux-3-image-klein, Qwen3-VL-4B-Instruct
  5. `qwen_image_21`      — Qwen/Qwen-Image-2.1 (Base) + official Comfy-Org/Qwen-Image-2.1 mirror (Turbo), Qwen3-VL-8B-Instruct

Wave B (owned by voltron-hub, registered here so loaders & schema stay unified):
  6. `flux2_klein_9b`     — black-forest-labs/FLUX.2-klein-base-9B & FLUX.2-klein-9B (Qwen3-8B)
  7. `flux2_dev`          — black-forest-labs/FLUX.2-dev (official singleton, Mistral-Small-3.1-24B)
  8. `p_image_oss`        — PrunaAI/P-Image-OSS (official singleton, PrunaAI Qwen3-VL-4B + PImageOSSTextConnector 5120)
  9. `moss_image`         — OpenMOSS-Team/MOSS-Image-Base & MOSS-Image-SFT (Qwen3_5Model)
 10. `flux1_schnell_dev`  — black-forest-labs/FLUX.1-dev & FLUX.1-schnell (T5-XXL + CLIP-L pooled)
 11. `sdxl_base_turbo`    — stabilityai/stable-diffusion-xl-base-1.0 & sdxl-turbo (CLIP-L + OpenCLIP-bigG + pooled)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TargetEncoderSpec:
    target_id: str
    wave: str  # "wave_a" | "wave_b"
    display_name: str
    repo_kind: str  # "base_and_turbo" | "official_comfy_mirror_base_and_turbo" | "base_and_sft" | "official_singleton"
    canonical_base_repo: str
    canonical_base_sha: str  # exact 40-char git commit SHA
    canonical_turbo_repo: Optional[str]  # None for official singletons
    canonical_turbo_sha: Optional[str]  # exact 40-char git commit SHA or None
    comfy_mirror_repo: Optional[str]
    comfy_mirror_sha: Optional[str]  # exact 40-char git commit SHA or None
    encoder_group: str
    encoder_repo: str
    encoder_sha: str  # exact 40-char git commit SHA (passed as `revision=` on every load!)
    encoder_arch: str
    hidden_size: int
    num_hidden_layers: int
    tap_layers: Tuple[int, ...]
    layer_norm_hidden_state: bool
    dit_conditioning_dim: int
    chat_template: str
    prompt_prefix_drop_rule: str
    padding_policy: str  # "unpadded" | "multiple_of_80" | "min_length_512" | "fixed_77"
    pad_multiple: int = 1
    min_seq_len: int = 1
    max_seq_len: int = 512
    native_pad_to_max_length: bool = False
    native_pad_token_id: int = 151643
    native_dit_returns_valid_rows_only: bool = True
    native_enable_thinking: Optional[bool] = None
    diffusers_pipeline_class: Optional[str] = None
    diffusers_cond_kwarg: str = "prompt_embeds"
    diffusers_mask_kwarg: Optional[str] = None
    diffusers_expects_list: bool = False
    comfy_extra_dict_keys: Tuple[str, ...] = ("pooled_output",)
    has_pooled_branch: bool = False
    pooled_dim: int = 0
    pooled_encoder_repo: Optional[str] = None
    pooled_donor_pooling: str = "none"
    voltron_mobile_target: bool = False
    byo_quant_variants: Tuple[str, ...] = ()
    explicit_layer_dims: Optional[Tuple[int, ...]] = None
    explicit_layer_labels: Optional[Tuple[str, ...]] = None
    encoder_subfolder: Optional[str] = None
    tokenizer_subfolder: Optional[str] = None
    notes: str = ""

    @property
    def layer_dims(self) -> Tuple[int, ...]:
        if self.explicit_layer_dims is not None:
            return tuple(int(d) for d in self.explicit_layer_dims)
        if int(self.hidden_size) * len(self.tap_layers) == int(self.dit_conditioning_dim):
            return (int(self.hidden_size),) * len(self.tap_layers)
        return (int(self.dit_conditioning_dim),)

    @property
    def layer_labels(self) -> Tuple[str, ...]:
        if self.explicit_layer_labels is not None:
            return tuple(str(x) for x in self.explicit_layer_labels)
        if len(self.layer_dims) == len(self.tap_layers):
            return tuple(f"h_{idx}" for idx in self.tap_layers)
        return ("connector_out",)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


TARGET_REGISTRY: Dict[str, TargetEncoderSpec] = {
    # =========================================================================
    # WAVE A (mecha-monkey) — 5 targets, every revision pinned to full 40-char SHA
    # =========================================================================
    "z_image_turbo": TargetEncoderSpec(
        target_id="z_image_turbo",
        wave="wave_a",
        display_name="Tongyi-MAI Z-Image / Z-Image-Turbo (6B S3-DiT, Voltron Mobile Target)",
        repo_kind="base_and_turbo",
        canonical_base_repo="Tongyi-MAI/Z-Image",
        canonical_base_sha="04cc4abb7c5069926f75c9bfde9ef43d49423021",
        canonical_turbo_repo="Tongyi-MAI/Z-Image-Turbo",
        canonical_turbo_sha="f332072aa78be7aecdf3ee76d5c247082da564a6",
        comfy_mirror_repo=None,
        comfy_mirror_sha=None,
        encoder_group="qwen3_4b",
        encoder_repo="Tongyi-MAI/Z-Image-Turbo",
        encoder_sha="f332072aa78be7aecdf3ee76d5c247082da564a6",
        encoder_subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
        encoder_arch="Qwen3ForCausalLM",
        hidden_size=2560,
        num_hidden_layers=36,
        tap_layers=(35,),  # hidden_states[-2] = block 34 output (35 blocks), unnormed (pre-cap_embedder!)
        layer_norm_hidden_state=False,
        dit_conditioning_dim=2560,
        chat_template="<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        prompt_prefix_drop_rule="prompt_span_only_for_word_align",
        padding_policy="unpadded",
        pad_multiple=1,
        min_seq_len=1,
        max_seq_len=512,
        native_pad_to_max_length=True,  # ZImagePipeline pads batch to max_length=512 during encode, then slices valid rows
        native_pad_token_id=151643,
        native_dit_returns_valid_rows_only=True,  # returns ONLY valid rows [L_valid, 2560]
        native_enable_thinking=True,
        diffusers_pipeline_class="ZImagePipeline",
        diffusers_cond_kwarg="prompt_embeds",
        diffusers_mask_kwarg=None,
        diffusers_expects_list=True,
        comfy_extra_dict_keys=("pooled_output",),
        has_pooled_branch=False,
        pooled_dim=0,
        voltron_mobile_target=True,
        byo_quant_variants=("bf16", "fp8", "int8", "gguf_q8_0", "gguf_q4_k_m"),
        notes=(
            "3/3 text_encoder shards byte-identical between Tongyi-MAI/Z-Image and Z-Image-Turbo "
            "(328a91d3, 6cd087b3, 7ca841ee). Targets pre-cap_embedder layer -2 (dim=2560); returns valid rows only."
        ),
    ),
    "flux2_klein_4b": TargetEncoderSpec(
        target_id="flux2_klein_4b",
        wave="wave_a",
        display_name="Black Forest Labs FLUX.2-klein-4B (Base + Turbo, Qwen3-4B)",
        repo_kind="base_and_turbo",
        canonical_base_repo="black-forest-labs/FLUX.2-klein-base-4B",
        canonical_base_sha="a3b4f4849157f664bdbc776fd7453c2783562f4d",
        canonical_turbo_repo="black-forest-labs/FLUX.2-klein-4B",
        canonical_turbo_sha="e7b7dc27f91deacad38e78976d1f2b499d76a294",
        comfy_mirror_repo=None,
        comfy_mirror_sha=None,
        encoder_group="qwen3_4b",
        encoder_repo="black-forest-labs/FLUX.2-klein-4B",
        encoder_sha="e7b7dc27f91deacad38e78976d1f2b499d76a294",
        encoder_subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
        encoder_arch="Qwen3ForCausalLM",
        hidden_size=2560,
        num_hidden_layers=36,
        tap_layers=(9, 18, 27),  # unnormed hidden_states[9, 18, 27]
        layer_norm_hidden_state=False,
        dit_conditioning_dim=7680,
        chat_template="<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        prompt_prefix_drop_rule="prompt_span_only_for_word_align",
        padding_policy="min_length_512",
        pad_multiple=1,
        min_seq_len=512,
        max_seq_len=512,
        native_pad_to_max_length=True,  # Flux2KleinPipeline pads to max_length=512 with attention_mask
        native_pad_token_id=151643,
        native_dit_returns_valid_rows_only=False,  # Klein native DiT conditioning includes ALL 512 rows!
        native_enable_thinking=False,
        diffusers_pipeline_class="Flux2KleinPipeline",
        diffusers_cond_kwarg="prompt_embeds",
        diffusers_mask_kwarg=None,
        diffusers_expects_list=False,
        comfy_extra_dict_keys=("pooled_output",),
        has_pooled_branch=False,
        pooled_dim=0,
        voltron_mobile_target=False,
        byo_quant_variants=("bf16", "fp8", "nvfp4"),
        notes=(
            "2/2 text_encoder shards byte-identical between FLUX.2-klein-base-4B and FLUX.2-klein-4B "
            "(8c0506e7, 82f2bd83). Native conditioning pads to max_length=512 with attention_mask and passes all 512 rows."
        ),
    ),
    "krea2": TargetEncoderSpec(
        target_id="krea2",
        wave="wave_a",
        display_name="Krea-2 Raw & Turbo (via official Comfy-Org/Krea-2 mirror)",
        repo_kind="official_comfy_mirror_base_and_turbo",
        canonical_base_repo="krea/Krea-2-Raw",
        canonical_base_sha="588feaf955b08b9ac4723eeb751752ab045dba71",  # pinned via official Comfy-Org/Krea-2 mirror
        canonical_turbo_repo="krea/Krea-2-Turbo",
        canonical_turbo_sha="588feaf955b08b9ac4723eeb751752ab045dba71",
        comfy_mirror_repo="Comfy-Org/Krea-2",
        comfy_mirror_sha="588feaf955b08b9ac4723eeb751752ab045dba71",
        encoder_group="qwen3_vl_4b",
        encoder_repo="Qwen/Qwen3-VL-4B-Instruct",
        encoder_sha="ebb281ec70b05090aa6165b016eac8ec08e71b17",
        encoder_subfolder=None,
        tokenizer_subfolder=None,
        encoder_arch="Qwen3VLForConditionalGeneration",
        hidden_size=2560,
        num_hidden_layers=36,
        tap_layers=(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
        layer_norm_hidden_state=False,
        dit_conditioning_dim=30720,
        chat_template=(
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
            "text, spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ),
        prompt_prefix_drop_rule="drop_system_and_user_header_krea2",
        padding_policy="unpadded",
        pad_multiple=1,
        min_seq_len=1,
        max_seq_len=512,
        native_pad_to_max_length=False,
        native_pad_token_id=151643,
        native_dit_returns_valid_rows_only=True,
        diffusers_pipeline_class=None,
        diffusers_cond_kwarg="prompt_embeds",
        diffusers_mask_kwarg=None,
        diffusers_expects_list=False,
        comfy_extra_dict_keys=("pooled_output",),
        has_pooled_branch=False,
        pooled_dim=0,
        voltron_mobile_target=False,
        byo_quant_variants=("bf16", "fp8_scaled", "int8_convrot", "mxfp8", "nvfp4"),
        notes=(
            "Official ungated Comfy-Org/Krea-2 mirror of krea/Krea-2-Raw & Krea-2-Turbo. "
            "12-layer Qwen3-VL-4B stack [2,5,8,...,35] -> 30,720 dims; native parity verifies against Comfy-Org qwen3vl_4b_bf16."
        ),
    ),
    "flux3_image_klein": TargetEncoderSpec(
        target_id="flux3_image_klein",
        wave="wave_a",
        display_name="Black Forest Labs FLUX.3-image-klein (Base + Turbo, Qwen3-VL-4B)",
        repo_kind="base_and_turbo",
        canonical_base_repo="black-forest-labs/flux-3-image-klein-base",
        canonical_base_sha="617a460237d514f6affc64684ca4e7fb2edecd5c",
        canonical_turbo_repo="black-forest-labs/flux-3-image-klein",
        canonical_turbo_sha="3b767ff9b5261d71645e2ec49d87d298d955238d",
        comfy_mirror_repo=None,
        comfy_mirror_sha=None,
        encoder_group="qwen3_vl_4b",
        encoder_repo="black-forest-labs/flux-3-image-klein",
        encoder_sha="3b767ff9b5261d71645e2ec49d87d298d955238d",
        encoder_subfolder="text_encoder",
        tokenizer_subfolder="text_encoder",
        encoder_arch="Qwen3VLForConditionalGeneration",
        hidden_size=2560,
        num_hidden_layers=36,
        tap_layers=(4, 8, 12, 16, 20, 24, 28, 32),
        layer_norm_hidden_state=False,
        dit_conditioning_dim=20480,
        chat_template="<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
        prompt_prefix_drop_rule="prompt_span_only_for_word_align",
        padding_policy="multiple_of_80",
        pad_multiple=80,
        min_seq_len=80,
        max_seq_len=560,
        native_pad_to_max_length=False,
        native_pad_token_id=151643,
        native_dit_returns_valid_rows_only=False,  # right-padded to multiple of 80 with RoPE pos=1 on pad
        diffusers_pipeline_class="Flux3DistilledModularPipeline",
        diffusers_cond_kwarg="prompt_embeds",
        diffusers_mask_kwarg=None,
        diffusers_expects_list=False,
        comfy_extra_dict_keys=("pooled_output", "attention_mask"),
        has_pooled_branch=False,
        pooled_dim=0,
        voltron_mobile_target=False,
        byo_quant_variants=("bf16", "fp8", "nvfp4"),
        notes=(
            "2/2 shards (30a01a05, 046296a2) + tokenizer (c6cc1014) 100% byte-identical across "
            "Qwen/Qwen3-VL-4B-Instruct, flux-3-image-klein, and flux-3-image-klein-base. "
            "8-layer stack [4..32] -> 20,480 dims, right-padded to multiple of 80 tokens with pad RoPE pos=1."
        ),
    ),
    "qwen_image_21": TargetEncoderSpec(
        target_id="qwen_image_21",
        wave="wave_a",
        display_name="Qwen-Image-2.1 (Base + Turbo via official Comfy-Org/Qwen-Image-2.1 mirror)",
        repo_kind="official_comfy_mirror_base_and_turbo",
        canonical_base_repo="Qwen/Qwen-Image-2.1",
        canonical_base_sha="d26bb61231c349cf6b7896fa83353113880e1ba3",
        canonical_turbo_repo="Comfy-Org/Qwen-Image-2.1",
        canonical_turbo_sha="aacf31a6d48e707f352ecba714cee0680da3479b",
        comfy_mirror_repo="Comfy-Org/Qwen-Image-2.1",
        comfy_mirror_sha="aacf31a6d48e707f352ecba714cee0680da3479b",
        encoder_group="qwen3_vl_8b",
        encoder_repo="Qwen/Qwen-Image-2.1",
        encoder_sha="d26bb61231c349cf6b7896fa83353113880e1ba3",
        encoder_subfolder="text_encoder",
        tokenizer_subfolder="processor",
        encoder_arch="Qwen3VLForConditionalGeneration",
        hidden_size=4096,
        num_hidden_layers=36,
        tap_layers=(36,),  # block 35 output BEFORE final RMSNorm (matching transformers 4.57 & ComfyUI qwen_image21.py)
        layer_norm_hidden_state=False,
        dit_conditioning_dim=4096,
        chat_template=(
            "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        ),
        prompt_prefix_drop_rule="drop_system_turn_qwen_image21",
        padding_policy="unpadded",
        pad_multiple=1,
        min_seq_len=1,
        max_seq_len=512,
        native_pad_to_max_length=False,
        native_pad_token_id=151643,
        native_dit_returns_valid_rows_only=True,
        diffusers_pipeline_class="QwenImage21Pipeline",
        diffusers_cond_kwarg="prompt_embeds",
        diffusers_mask_kwarg="prompt_embeds_mask",
        diffusers_expects_list=False,
        comfy_extra_dict_keys=("pooled_output",),
        has_pooled_branch=False,
        pooled_dim=0,
        voltron_mobile_target=False,
        byo_quant_variants=("bf16", "int8_convrot", "turbo_bf16", "turbo_int8_convrot"),
        notes=(
            "Pinned directly to Qwen/Qwen-Image-2.1 text_encoder + processor (4 shards dde00291..5311532a); "
            "Turbo checkpoint sourced from official Comfy-Org/Qwen-Image-2.1 mirror (qwen_image_2.1_turbo_bf16.safetensors)."
        ),
    ),
    # =========================================================================
    # WAVE B (voltron-hub) — full 40-char SHAs, zero placeholders
    # =========================================================================
    "flux2_klein_9b": TargetEncoderSpec(
        target_id="flux2_klein_9b",
        wave="wave_b",
        display_name="Black Forest Labs FLUX.2-klein-9B (Base + Distilled, Qwen3-8B)",
        repo_kind="base_and_turbo",
        canonical_base_repo="black-forest-labs/FLUX.2-klein-base-9B",
        canonical_base_sha="8e4154e4c94c6eb08644d0761d404e1c28d325c2",
        canonical_turbo_repo="black-forest-labs/FLUX.2-klein-9B",
        canonical_turbo_sha="1498bcb8aaebb230bae82c34b02ecc4087beebac",
        comfy_mirror_repo=None,
        comfy_mirror_sha=None,
        encoder_group="qwen3_8b",
        encoder_repo="black-forest-labs/FLUX.2-klein-base-9B",
        encoder_sha="8e4154e4c94c6eb08644d0761d404e1c28d325c2",
        encoder_subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
        encoder_arch="Qwen3ForCausalLM",
        hidden_size=4096,
        num_hidden_layers=36,
        tap_layers=(9, 18, 27),
        layer_norm_hidden_state=False,
        dit_conditioning_dim=12288,
        chat_template="<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        prompt_prefix_drop_rule="prompt_span_only_for_word_align",
        padding_policy="min_length_512",
        min_seq_len=512,
        max_seq_len=512,
        native_pad_to_max_length=True,
        native_dit_returns_valid_rows_only=False,
        native_enable_thinking=False,
        diffusers_pipeline_class="Flux2KleinPipeline",
        notes="Wave B (voltron-hub); 4/4 Qwen3-8B shards byte-identical between base-9B (200 OK) and 9B.",
    ),
    "flux2_dev": TargetEncoderSpec(
        target_id="flux2_dev",
        wave="wave_b",
        display_name="Black Forest Labs FLUX.2-dev (Official Singleton, Mistral-Small-3.1-24B)",
        repo_kind="official_singleton",
        canonical_base_repo="black-forest-labs/FLUX.2-dev",
        canonical_base_sha="e8af80c9dd820ab07ee4171d024f03e5c4b3024b",
        canonical_turbo_repo=None,
        canonical_turbo_sha=None,
        comfy_mirror_repo=None,
        comfy_mirror_sha=None,
        encoder_group="mistral3_24b",
        encoder_repo="black-forest-labs/FLUX.2-dev",
        encoder_sha="e8af80c9dd820ab07ee4171d024f03e5c4b3024b",
        encoder_subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
        encoder_arch="Mistral3ForConditionalGeneration",
        hidden_size=5120,
        num_hidden_layers=40,
        tap_layers=(10, 20, 30),
        layer_norm_hidden_state=False,
        dit_conditioning_dim=15360,
        chat_template="PixtralProcessor.apply_chat_template(SYSTEM_MESSAGE, user={prompt}, add_generation_prompt=False)",
        prompt_prefix_drop_rule="prompt_span_only_for_word_align",
        padding_policy="min_length_512",
        min_seq_len=512,
        max_seq_len=512,
        native_pad_to_max_length=True,
        native_dit_returns_valid_rows_only=False,
        diffusers_pipeline_class="Flux2Pipeline",
        notes="Wave B (voltron-hub); official singleton (no canonical Turbo checkpoint).",
    ),
    "p_image_oss": TargetEncoderSpec(
        target_id="p_image_oss",
        wave="wave_b",
        display_name="PrunaAI P-Image-OSS (Official Singleton, PrunaAI Qwen3-VL-4B + PImageOSSTextConnector 5120)",
        repo_kind="official_singleton",
        canonical_base_repo="PrunaAI/P-Image-OSS",
        canonical_base_sha="e6030847022639fea1ca0a52cf7f945ba5ae8d07",
        canonical_turbo_repo=None,
        canonical_turbo_sha=None,
        comfy_mirror_repo="PrunaAI/P-Image-OSS-Comfy",
        comfy_mirror_sha="177745a635feb2f0adf3db3693297cb63fea3dda",
        encoder_group="p_image_oss_qwen3vl4b",
        encoder_repo="PrunaAI/P-Image-OSS",
        encoder_sha="e6030847022639fea1ca0a52cf7f945ba5ae8d07",
        encoder_subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
        encoder_arch="Qwen3VLForConditionalGeneration+PImageOSSTextConnector",
        hidden_size=2560,
        num_hidden_layers=36,
        tap_layers=(13, 19, 25, 31, 36),
        layer_norm_hidden_state=False,
        dit_conditioning_dim=5120,
        chat_template="{prompt}",
        prompt_prefix_drop_rule="none",
        padding_policy="unpadded",
        diffusers_pipeline_class="PImagePipeline",
        explicit_layer_dims=(5120,),
        explicit_layer_labels=("p_image_oss_connector_5120",),
        notes=(
            "Wave B (voltron-hub); official singleton (Comfy mirror PrunaAI/P-Image-OSS-Comfy). "
            "PrunaAI text_encoder shards differ from vanilla Qwen3-VL-4B-Instruct; requires token-0 sink_embedding [5120]."
        ),
    ),
    "moss_image": TargetEncoderSpec(
        target_id="moss_image",
        wave="wave_b",
        display_name="OpenMOSS MOSS-Image (Base + SFT, Qwen3_5Model)",
        repo_kind="base_and_sft",
        canonical_base_repo="OpenMOSS-Team/MOSS-Image-Base",
        canonical_base_sha="85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
        canonical_turbo_repo="OpenMOSS-Team/MOSS-Image-SFT",
        canonical_turbo_sha="f972bd4b4b9e3852133cc5da954d87bc32cd8fb9",
        comfy_mirror_repo=None,
        comfy_mirror_sha=None,
        encoder_group="moss_qwen35",
        encoder_repo="OpenMOSS-Team/MOSS-Image-Base",
        encoder_sha="85975793d8b3d432f1b6e27f352abaa4c2a4c0d3",
        encoder_subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
        encoder_arch="Qwen3_5Model",
        hidden_size=4096,
        num_hidden_layers=32,
        tap_layers=(32,),
        layer_norm_hidden_state=True,
        dit_conditioning_dim=4096,
        chat_template="tokenizer.apply_chat_template([user={prompt}], add_generation_prompt=True, enable_thinking=True)",
        prompt_prefix_drop_rule="prompt_span_only_for_word_align",
        padding_policy="unpadded",
        native_enable_thinking=True,
        diffusers_pipeline_class="MOSSImagePipeline",
        diffusers_cond_kwarg="prompt_embeds",
        diffusers_mask_kwarg="prompt_attention_mask",
        notes="Wave B (voltron-hub); 4/4 Qwen3_5Model shards byte-identical between MOSS-Image-Base and SFT.",
    ),
    "flux1_schnell_dev": TargetEncoderSpec(
        target_id="flux1_schnell_dev",
        wave="wave_b",
        display_name="Black Forest Labs FLUX.1-dev & FLUX.1-schnell (T5-XXL + CLIP-L pooled)",
        repo_kind="base_and_turbo",
        canonical_base_repo="black-forest-labs/FLUX.1-dev",
        canonical_base_sha="45d95c497444bbaebe766e1528792fc9158c0c3a",
        canonical_turbo_repo="black-forest-labs/FLUX.1-schnell",
        canonical_turbo_sha="169573f54945f633efbe126a40a5676b4228363c",
        comfy_mirror_repo=None,
        comfy_mirror_sha=None,
        encoder_group="t5xxl_clipl",
        encoder_repo="black-forest-labs/FLUX.1-schnell",
        encoder_sha="169573f54945f633efbe126a40a5676b4228363c",
        encoder_subfolder="text_encoder_2",
        tokenizer_subfolder="tokenizer_2",
        encoder_arch="T5EncoderModel+CLIPTextModel",
        hidden_size=4096,
        num_hidden_layers=24,
        tap_layers=(24,),
        layer_norm_hidden_state=True,
        dit_conditioning_dim=4096,
        chat_template="{prompt}",
        prompt_prefix_drop_rule="none",
        padding_policy="unpadded",
        max_seq_len=512,
        diffusers_pipeline_class="FluxPipeline",
        diffusers_cond_kwarg="prompt_embeds",
        has_pooled_branch=True,
        pooled_dim=768,
        pooled_encoder_repo="black-forest-labs/FLUX.1-schnell",
        pooled_donor_pooling="mean",
        notes="Wave B (voltron-hub); 3/3 shards byte-identical between FLUX.1-dev and FLUX.1-schnell.",
    ),
    "sdxl_base_turbo": TargetEncoderSpec(
        target_id="sdxl_base_turbo",
        wave="wave_b",
        display_name="Stability AI SDXL-base-1.0 & SDXL-Turbo (CLIP-L + OpenCLIP-bigG + pooled)",
        repo_kind="base_and_turbo",
        canonical_base_repo="stabilityai/stable-diffusion-xl-base-1.0",
        canonical_base_sha="462165984030d82259a11f4367a4eed129e94a7b",
        canonical_turbo_repo="stabilityai/sdxl-turbo",
        canonical_turbo_sha="71153311d3dbb46851df1931d3ca6e939de83304",
        comfy_mirror_repo=None,
        comfy_mirror_sha=None,
        encoder_group="clipl_clipg",
        encoder_repo="stabilityai/stable-diffusion-xl-base-1.0",
        encoder_sha="462165984030d82259a11f4367a4eed129e94a7b",
        encoder_subfolder="text_encoder",
        tokenizer_subfolder="tokenizer",
        encoder_arch="CLIPTextModel+CLIPTextModelWithProjection",
        hidden_size=2048,
        num_hidden_layers=32,
        tap_layers=(-2,),
        layer_norm_hidden_state=False,
        dit_conditioning_dim=2048,
        chat_template="{prompt}",
        prompt_prefix_drop_rule="none",
        padding_policy="fixed_77",
        min_seq_len=77,
        max_seq_len=77,
        native_pad_to_max_length=True,
        native_dit_returns_valid_rows_only=False,
        diffusers_pipeline_class="StableDiffusionXLPipeline",
        diffusers_cond_kwarg="prompt_embeds",
        has_pooled_branch=True,
        pooled_dim=1280,
        pooled_encoder_repo="stabilityai/stable-diffusion-xl-base-1.0",
        pooled_donor_pooling="mean",
        explicit_layer_dims=(768, 1280),
        explicit_layer_labels=("clip_l_h_m2", "openclip_bigg_h_m2"),
        notes="Wave B (voltron-hub); 2/2 fp16 shards byte-identical between base-1.0 and sdxl-turbo.",
    ),
}


def get_target_spec(target_id: str) -> TargetEncoderSpec:
    if target_id not in TARGET_REGISTRY:
        raise KeyError(f"Unknown target_id {target_id!r}; known: {sorted(TARGET_REGISTRY.keys())}")
    return TARGET_REGISTRY[target_id]


def list_targets(wave: Optional[str] = None) -> List[TargetEncoderSpec]:
    items = list(TARGET_REGISTRY.values())
    if wave is not None:
        items = [t for t in items if t.wave == wave]
    return items


def wave_a_encoder_groups() -> Dict[str, List[TargetEncoderSpec]]:
    groups: Dict[str, List[TargetEncoderSpec]] = {}
    for spec in list_targets("wave_a"):
        groups.setdefault(spec.encoder_group, []).append(spec)
    return groups

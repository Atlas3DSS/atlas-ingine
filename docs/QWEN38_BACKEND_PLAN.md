# #821/#822 Frozen Phase 0 Plan: Qwen3.8-27B-NVFP4 Backend on atlas-ingine

- **Work items**: `#821` (Port dense `Qwen3.8-27B-NVFP4-482ca0f` text worker onto `atlas-ingine`), `#822` (Mixed Qwen + Inkling workers & `kuraneko1` QSA v2 sparse attention)
- **Worktree / Branch**: `/home/orwel/dev_genius/engines/atlas-ingine-qwen` on `feat/backend-qwen38` (branched from `main` @ `5e9b41d59f860b5e020b9aa64bc489304218b924`)
- **Host / Family Separation**: Zero edits to `main`; zero edits to `native/include/ingine/generator.h` (`0-byte diff`). Shared interface lives in `native/include/ingine/model_backend.h`; Qwen-specific Phase 0 CPU code lives strictly under `native/src/backends/qwen38/`.
- **Phase 0 Execution Boundary**: Strict CPU-only build and test (`CUDA_VISIBLE_DEVICES=""`). Zero Devbox PRO 6000 or cloud GPU runs in Phase 0.

---

## 1. Pinned Checkpoint & Reference Inventory (SHA-256 Verified)

### 1.1 Primary Target: `Qwen3.8-27B-NVFP4-482ca0f`
- **Devbox path**: `/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f`
- **v100 path**: `/mnt/nvme/models/Qwen3.8-27B-NVFP4` (`config.json`, `tokenizer.json`, `model.safetensors.index.json` verified bit-identical)
- **HF revision**: `482ca0f3832238542f8f5295dde86b5f22711d80`

| Artifact | Size (bytes) | SHA-256 |
|---|---:|---|
| `model-00001-of-00003.safetensors` | 9,965,652,544 | `7d0fd155118901373eb0fd13ed3aae68f95be747bc205be72ebc41739c25ee80` |
| `model-00002-of-00003.safetensors` | 9,985,757,064 | `98a7e9486baa860c792c9463a770cb9d017696bdd17606300a5b9334149f4c27` |
| `model-00003-of-00003.safetensors` | 1,970,287,672 | `0506ad35dc21469708e7813bd76c592cef08bd0317b4a1fe899745ebe2435271` |
| `model.safetensors.index.json` | — | `7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2` |
| `config.json` | — | `71fa01bb64de20971045cd82a82c0ebb25a54e1b0a3f09ef23cf5479e9c54df8` |
| `tokenizer.json` | — | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `tokenizer_config.json` | — | `e5d078b00e6c1223b32444db8c1001dc71d86ceef8ee706b5bf084c3a43a1f9c` |
| `chat_template.jinja` | — | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| `generation_config.json` | — | `8a3a817a9d330f6f9e62f048fe62a48594946c4a5c4cb314a8011316d5094986` |
| `hf_quant_config.json` | — | `5f4aaa9e462ddb26a9790ab459eee69d100b9ade7541f0c86038e925c013f80b` |

Active text worker loads **1,846** tensors (`20,150,569,096 B` / `18.767 GiB`: `model.language_model.*` + `lm_head.*`) and excludes **333** `model.visual.*` tensors (`921,460,192 B` / `878.8 MiB`) and **15** `mtp.*` tensors (`849,398,784 B` / `810.0 MiB`).

### 1.2 Secondary Same-Architecture Brings-Up Model: `Qwen3.5-0.8B`
- **Path**: `/mnt/storage/models/Qwen3.5-0.8B`
- **Layout**: 24 layers (18 `linear_attention` + 6 `full_attention`), pure `BF16`, `tie_word_embeddings=true`, 320 active text tensors (`1,504,791,232 B` / `1.401 GiB`), 153 excluded visual + 15 excluded MTP tensors.

### 1.3 Trusted Reference Truth Artifacts (`/mnt/storage/flare/release-references/`)

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `reference.safetensors` | HF canonical eager FP32 + raw NVFP4 weight-only + calibrated FP8 activations (3x 32-tok prompts) | `30dd669b1baf6a1ec517b635c083b275dd72417ca07521bf33cecdd954f02a3f` |
| `long-code-reference.safetensors` | 6x ~4,096-tok code prompts (`code-00..code-05`, `enable_thinking=false`) | `d4479957d5ec101c8274d7a7fe59032d0171423ebacd8607a6aeb61e00f1afc9` |
| `prompts.jsonl` | Source chat messages for `code-00..code-05` | `b200be2010c1c4eb24b7cd787141e7db23cefc7df7cc41e2a232764a92aad134` |
| `reference.json` | Short prompt text/IDs & generated tokens | `277f47825bac44feba200dfab6ab555d637b66a30d31c2bfdfa2f0f09afcc5a3` |
| `long-code-reference.json` | Rendered token IDs, continuation tokens & teacher-forced self-checks | `663ab9262b33ae26a4f07dbbd08ac02ea188990dbc1cc7c8df1effa474e0f15a` |

Verified Flare SM120 baselines against these references (`sm120.json` and `long-code-reference-noise-floor.json`):
- Short prompts (`reference.safetensors`): `prompt_0` KLD mean `0.00116` / max `0.00582` / `100%` top1; `prompt_1` mean `0.00224` / max `0.01961` / `91.5%` top1; `prompt_2` mean `0.00158` / max `0.01057` / `97.9%` top1.
- Long-code noise floor (`B=2` vs `B=1`): `kl_mean = 0.001936`, `kl_p99 = 0.01704`, `kl_max = 0.11033`, `top1_agree = 288/288`.

---

## 2. Shared Multi-Family `ModelBackend` Contract (`model_backend.h`)

`native/include/ingine/model_backend.h` defines the single shared host/model-family boundary extending `ingine::model::Generator` without changing Inkling callers or `generator.h`, incorporating `astartes-maxima`'s `#834` architectural review (`att-72ce84e5`) and submitted for `#833` DeepSeek and `#834` GLM review prior to adoption:
- **Checkpoint Format Dispatch & Immutable Provenance**: `detect_checkpoint_format()` dispatches `CheckpointFormat` (`Safetensors`, `Gguf`, fail-closed on empty/ambiguous directories) alongside `detect_model_family()`. `CheckpointIdentity` carries immutable `shard_set_sha256` (exact sorted shard set digest), `shard_count`, `schema_variant`, `config_sha256`, `tokenizer_sha256`, `chat_template_sha256`, `weight_manifest_sha256`, `weight_manifest_verified`, and `shard_digests_verified`.
- **Fail-Closed Numeric/State ABI & Exact Prefix Check**: `ActivationDtype` (`Bf16`, `Fp16`, `Fp32`, `Unknown`) keeps `F16` and `BF16` strictly distinct. `BackendStateIdentity::is_reusable_with()` and `hash_backend_state_identity()` cover `#827` compatibility, `#829` `kv_cache_mode`, `#838` `gear`, `mtp_enabled`, `rendering_fingerprint`, and opaque `family_state_fingerprint` while DeepSeek (`Engram`/`CSA`/`MLA`) and GLM (`KDA`/`MLA`) retain internal state representations privately. Cache lookups pair `is_reusable_with()` with exact `prefix_matches_exact()` / `prefix_matches_exact_with_mask()`.
- **Capabilities & Admission**: `BackendCapabilities`, `WorkerProfile`, `BackendSession::truncate()`, `validate_runtime_options_against_capabilities()`, and `validate_request_against_capabilities()`.
- **Tiered Residency (`#836`) & HAL (`#835`)**: Common `ResidencyReport` (`device_bytes`, `host_ram_bytes`, `mmap_bytes`, `nvme_swap_bytes`, `process_rss_bytes`) and `AdmissionEstimate` (`required_device_bytes`, `required_host_ram_bytes`, `required_mmap_bytes`, `required_nvme_swap_bytes`) plus `park_weights()` / `evict_caches()`. Family-specific state packing and kernels stay private behind `#835` HAL.

---

## 3. Tokenizer, Chat Template & Architecture Spec

- **Text Stack (`ingine_text`, `openai.cpp`)**:
  - `SplitMode::Qwen2` (Qwen2 regex pre-tokenizer), `NormalizerKind::Nfc` via `utf8proc` (`UTF8PROC_STABLE | UTF8PROC_COMPOSE`), `ignore_merges=true`, vocab size `248,320`, special tokens `<|endoftext|>` (`248044`), `<|im_start|>` (`248045`), `<|im_end|>` (`248046`), `<think>` (`248068`), `</think>` (`248069`), `<tool_call>` (`248058`), `</tool_call>` (`248059`).
  - `ChatFormatKind::Qwen3_5`: ChatML + XML `<tool_call><function=...><parameter=...>` rendering (`enable_thinking`, `reasoning_effort` in `{xhigh, medium, low}`), `StreamParser` `Kind::ToolXml`, `openai.cpp` guard on `-1` Inkling control IDs.
- **Backbone (`Qwen38Config` — 64 layers `[linear, linear, linear, full] * 16`)**:
  - `hidden_size = 5120`, `intermediate_size = 17408`, `rms_norm_eps = 1e-6`, `tie_word_embeddings = false`.
  - **Gated DeltaNet (`48` layers)**: `conv_kernel = 4`, `conv_dim = 10240`, `num_k_heads = 16`, `num_v_heads = 48`, `k_dim = 128`, `v_dim = 128`, `mamba_ssm_dtype = float32`; `in_proj_qkv`/`in_proj_z`/`out_proj` in calibrated FP8 E4M3, `in_proj_{a,b}`/`A_log`/`dt_bias`/`conv1d`/`norm` in BF16.
  - **Full Attention (`16` layers)**: `num_heads = 24`, `num_kv_heads = 4`, `head_dim = 256`, `attn_output_gate = true` (`q_proj` interleaved `[q, gate]` with `sigmoid(gate)` before `o_proj`), `q_norm`/`k_norm` zero-centered RMSNorm, NeoX partial RoPE `rotary_dim = 64` (`0.25`, `theta = 1e7`), FP8 E4M3 projections.
  - **MLP & `lm_head`**: `SwiGLU` + `lm_head` in ModelOpt `NVFP4` (`E2M1` packed U8 + `F8_E4M3` group-16 block scales + `F32 weight_scale_2` + `input_scale`).
  - **Norm Polarity**: `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, and final `norm` are **zero-centered** `(1.0 + w) * norm(x)`; GDN `linear_attn.norm` is **standard** `w * norm(x) * silu(z)`.

### VRAM & State Footprint (`Qwen3.8-27B-NVFP4`)

| Component | Exact Bytes | Human Size | Notes |
|---|---:|---:|---|
| Active text weights (`1,846` tensors) | 20,150,569,096 | 18.767 GiB | Excludes 878.8 MiB visual & 810.0 MiB MTP |
| GDN recurrent state (`48 * 48 * 128 * 128 * F32`) | 150,994,944 | 144.0 MiB | Fixed per slot |
| GDN conv1d state (`48 * 10240 * 4 * BF16`) | 3,932,160 | 3.75 MiB | Fixed per slot |
| Full-attention KV cache (`16 * 2 * 4 * 256 * BF16`) | 65,536 / tok | 64 KiB / tok | `2.0 GiB` @ 32k ctx, `8.0 GiB` @ 128k ctx |

Fits 1x RTX PRO 6000 (96 GB) through 128k–256k context, or 1x 24–48 GB GPU for Phase 1 kernel/parity tests (`<= 8k–32k` ctx).

---

## 4. Frozen Qualification Gates (Gates 0–4)

| Gate | Scope | Frozen Pass Criteria |
|---|---|---|
| **Gate 0 (Phase 0, Current)** | CPU-only (`CUDA_VISIBLE_DEVICES=""`) | 100% pass across all `-LE gpu` CTests (`backend_model_contract`, `qwen38_manifest`, `qwen38_text`, `qwen38_cpu_ops` + all existing Inkling CPU tests); zero CUDA headers/linkage in `model_backend.h` and `tests/backends/*`; `generator.h` 0-byte diff from `main`. |
| **Gate 1 (Phase 1A)** | GPU unit kernels | CUDA kernels (`nvfp4_linear`, `fp8_linear`, `rmsnorm_zero_centered`, `rmsnorm_gated_silu`, `partial_rope_neox`, `causal_conv1d_silu`, `gdn_recurrent_step`, `gdn_chunk_prefill`, `full_attn_gated`) match CPU/FP32 oracle within `max_abs <= 1e-3` (BF16) / `<= 2e-4` (F32 GDN state); 1-layer and 4-layer block parity on `0.8B` and `27B`. |
| **Gate 2 (Phase 1B)** | Full 27B dense reference parity | **Short prompts (`reference.safetensors`)**: `prompt_0` 16/16 exact greedy match; aggregate `kld_mean <= 0.005`, `kld_max <= 0.05`, `top1_agreement >= 0.90`. **Long code (`long-code-reference.safetensors`, `code-00..code-05`)**: aggregate `kld_mean <= 0.01`, `kld_p99 <= 0.08`, `top1_agreement >= 0.95`. **Prefill-vs-decode**: chunked prefill final logits match stepwise prefill (`kld <= 1e-4`, 100% short greedy match). |
| **Gate 3 (Phase 1C)** | State reuse & isolation | **Repeated prefix**: `BackendStateIdentity::is_reusable_with()` + `prefix_matches_exact()` hit produces exact greedy continuation and `max_abs_logit_diff <= 1e-4` vs cold uncached prefill. Any identity or prefix mismatch fails closed to miss. **Fresh session isolation**: interleaved requests + `reset()` yield zero conv/recurrent/KV leakage (`<= 1e-5` logit parity vs cold run). |
| **Gate 4 (Zero Regression)** | Inkling regression | Full existing Inkling CPU and GPU parity suites pass 100%. |

---

## 5. Hardware Queue, Cloud Policy, Telemetry & `#822` Plan

- **Local PRO 6000 Queue Lock**: Devbox PRO 6000 is locked behind `#826 -> #827 -> #829 -> #838 -> #835`. No local PRO 6000 jobs until the queue clears and `astartes-maxima` approves Phase 1 launch.
- **Cloud GPU Policy (Owner Order & Budget Rule)**:
  1. Check Google Cloud (`/home/orwel/.local/bin/gcloud`, project `gen-lang-client-0170916850`) **FIRST** to consume credits before cash (currently `GPUS_ALL_REGIONS limit 0`; re-verify at launch time).
  2. Only fall back to Vast.ai (`USD 22.32` shared baseline credit across `omni-inkling`, `qwen-integration`, `ds-integration`, `glm-integration`) when GCP quota is unavailable, strictly observing the **2026-11-25 08:12 PST Owner Vast.ai Budget Rule**: `USD 5.00` per auto-refill, **hard cap `<= 2x` auto-refills (`USD 10.00` max auto-refill, `USD 32.32` cumulative team ceiling, currently `0/2`)** before explicit owner `GO/NO-GO`; obtain `astartes-maxima` plan + per-run hard all-in dollar cap sign-off and coordinate with `inkling-agent` (lead) when approaching ceiling; sync artifacts and destroy instances immediately upon completion.
- **Required Phase 1–2 Telemetry**: Every GPU run records `cold_manifest_sha_ms`, `cold_safetensors_load_ms`, `warm_reset_ms`, `warm_prefix_cache_lookup_ms`, `prefill_ms`, `prefill_tok_s`, `decode_ms`, `decode_tok_s`, `peak_vram_bytes`, and `host_rss_bytes`.
- **`#822` Scope After `#821` Passes**:
  1. Multi-worker dispatch using `WorkerProfile` (separate `Qwen3_5` and `Inkling2` workers, zero shared mutable weights).
  2. `kuraneko1` QSA v2 sparse attention applied strictly on the 16 `full_attention` layers (`4-token` blocks, `2,048-token` block selection budget + current block, full BF16 KV cache kept allocated in slot); 48 Gated DeltaNet layers run dense unmodified.

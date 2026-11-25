#pragma once

#include "backends/qwen38/qwen38_config.h"
#include "backends/qwen38/qwen38_manifest.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

namespace ingine::backends::qwen38 {

// Fast IEEE-754 F32 -> E4M3FN round-to-nearest-even encoder, bit-exact with quantize_e4m3fn().
std::uint8_t quantize_e4m3fn_fast(float x) noexcept;

struct Qwen38ProjWeightsHost {
  int out_features = 0;
  int in_features = 0;
  Qwen38LinearQuantKind quant = Qwen38LinearQuantKind::Bf16;

  std::vector<std::uint16_t> bf16_w;       // [out_features, in_features] when quant == Bf16
  std::vector<std::uint8_t> fp8_w;         // [out_features, in_features] when quant == Fp8E4M3
  float weight_scale = 1.0f;
  float input_scale = 0.0f;

  std::vector<std::uint8_t> nvfp4_packed;  // [out_features, in_features / 2] when quant == Nvfp4
  std::vector<std::uint8_t> nvfp4_scales;  // [out_features, in_features / 16] E4M3
  float weight_scale_2 = 1.0f;
};

struct Qwen38BlockLayerHost {
  int layer_index = 0;
  Qwen38LayerKind kind = Qwen38LayerKind::LinearAttention;

  std::vector<std::uint16_t> input_layernorm;           // [hidden_size] BF16
  std::vector<std::uint16_t> post_attention_layernorm;  // [hidden_size] BF16

  // LinearAttention (Gated DeltaNet) weights.
  Qwen38ProjWeightsHost in_proj_qkv;
  Qwen38ProjWeightsHost in_proj_z;
  Qwen38ProjWeightsHost in_proj_a;
  Qwen38ProjWeightsHost in_proj_b;
  Qwen38ProjWeightsHost out_proj;
  std::vector<float> A_log;               // [linear_num_value_heads]
  std::vector<float> dt_bias;             // [linear_num_value_heads]
  std::vector<float> conv1d_weight;       // [linear_conv_dim, linear_conv_kernel_dim]
  std::vector<float> linear_norm_weight;  // [linear_value_head_dim]

  // FullAttention (gated GQA + partial NeoX RoPE) weights.
  Qwen38ProjWeightsHost q_proj;
  Qwen38ProjWeightsHost k_proj;
  Qwen38ProjWeightsHost v_proj;
  Qwen38ProjWeightsHost o_proj;
  std::vector<std::uint16_t> q_norm;  // [head_dim] BF16
  std::vector<std::uint16_t> k_norm;  // [head_dim] BF16

  // SwiGLU MLP weights.
  Qwen38ProjWeightsHost gate_proj;
  Qwen38ProjWeightsHost up_proj;
  Qwen38ProjWeightsHost down_proj;
};

struct Qwen38BlockBundleHost {
  Qwen38Config cfg;
  std::vector<std::int32_t> token_ids;
  std::vector<std::uint16_t> prompt_embeddings_bf16;  // [token_ids.size(), cfg.hidden_size]
  std::vector<Qwen38BlockLayerHost> layers;
};

struct Qwen38BlockRunOutput {
  std::vector<float> hidden_seq;  // [token_ids.size(), cfg.hidden_size]
  std::vector<std::vector<float>> conv_states;
  std::vector<std::vector<float>> recurrent_states;
  std::vector<std::vector<float>> k_caches;
  std::vector<std::vector<float>> v_caches;
  double elapsed_ms = 0.0;
  std::uint64_t peak_vram_bytes = 0;
};

// Deterministic synthetic 1-layer or 4-layer block bundle (Bf16 or MixedFp8Nvfp4) for fast parity tests.
Qwen38BlockBundleHost make_synthetic_qwen38_block_bundle(Qwen38WeightPrecision prec,
                                                         std::span<const int> layer_indices,
                                                         std::span<const std::int32_t> token_ids,
                                                         std::uint32_t seed = 20261125u);

// Loads only `embed_tokens` rows for `token_ids` and requested `layer_indices` from a safetensors
// checkpoint. Uses stream_verified_file() when manifest is PinnedVerified, or open-FD pread verified
// against manifest.verified_files when unpinned (HeadersOnly/SelfHashed).
Qwen38BlockBundleHost load_qwen38_block_bundle(const std::filesystem::path& model_dir,
                                               const Qwen38Manifest& manifest,
                                               std::span<const int> layer_indices,
                                               std::span<const std::int32_t> token_ids);

// Executes the block bundle on CPU with deterministic row-partitioned multi-threading and FP64
// inner products (bit-identical output across thread counts).
Qwen38BlockRunOutput run_qwen38_blocks_cpu(const Qwen38BlockBundleHost& bundle, bool bf16_boundaries,
                                           bool use_chunk_prefill, int chunk_size = 64,
                                           bool use_calibrated_fp8_input_scale = true, int num_threads = 0);

}  // namespace ingine::backends::qwen38

#pragma once

#include "backends/qwen38/qwen38_config.h"
#include "backends/qwen38/qwen38_manifest.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
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
// inner products (or optional 32-lane warp FP32 simulation when fp32_accum_sim=true).
Qwen38BlockRunOutput run_qwen38_blocks_cpu(const Qwen38BlockBundleHost& bundle, bool bf16_boundaries,
                                           bool use_chunk_prefill, int chunk_size = 64,
                                           bool use_calibrated_fp8_input_scale = true, int num_threads = 0,
                                           bool fp32_accum_sim = false);

// Extracts a single layer bundle (keeping config, tokens, and prompt embeddings) to avoid re-reading shards.
Qwen38BlockBundleHost slice_qwen38_block_layer(const Qwen38BlockBundleHost& bundle, std::size_t layer_pos);

// Computes finite BF16 ULP distance (+0.0 == -0.0); returns UINT32_MAX if either operand is Inf or NaN.
std::uint32_t bf16_ulp_distance(std::uint16_t a, std::uint16_t b) noexcept;

struct Qwen38Bf16CompareStats {
  float max_abs_vs_bf16_oracle = 0.0f;
  std::uint32_t max_bf16_ulp = 0;
  std::size_t exact_bf16_matches = 0;
  std::size_t non_exact_bf16_count = 0;
  std::size_t criterion_violations = 0;
  bool passed = false;
};

// Compares BF16 outputs against explicit round-to-nearest-even BF16 conversion of ref_f32.
// Each element fails iff ULP distance > max_ulp AND abs diff vs rounded oracle > abs_floor (or non-finite).
Qwen38Bf16CompareStats compare_bf16_against_rounded_oracle(std::span<const std::uint16_t> got_bf16_bits,
                                                           std::span<const float> ref_f32,
                                                           std::uint32_t max_ulp = 1,
                                                           float abs_floor = 1e-3f);

struct Qwen38Gate1LinearFixtures {
  int m = 4;
  int n = 256;
  int k = 5120;

  std::vector<std::uint16_t> nvfp4_x_bf16;
  std::vector<float> nvfp4_x_q;
  std::vector<std::uint8_t> nvfp4_packed;
  std::vector<std::uint8_t> nvfp4_scales;
  float nvfp4_ws2 = 0.015f;
  std::vector<float> nvfp4_ref_f32;
  std::vector<std::uint16_t> nvfp4_ref_bf16;

  std::vector<std::uint16_t> fp8_x_bf16;
  std::vector<float> fp8_x_q;
  std::vector<std::uint8_t> fp8_w;
  float fp8_ws = 0.025f;
  float fp8_in_sc = 0.125f;
  std::vector<float> fp8_wo_ref_f32;
  std::vector<std::uint16_t> fp8_wo_ref_bf16;
  std::vector<float> fp8_cal_ref_f32;
  std::vector<std::uint16_t> fp8_cal_ref_bf16;
};

// Generates the deterministic Gate 1 NVFP4 + FP8 unit fixtures (seed=20261125u) and CPU FP64-accum references.
Qwen38Gate1LinearFixtures make_gate1_linear_fixtures(std::uint32_t seed = 20261125u);

// Simulates the 32-lane warp strided FP32 accumulation + butterfly reduction order of qwen38_cuda_ops.cu.
void simulate_warp_f32_nvfp4_linear(std::span<const float> x_in, std::span<const std::uint8_t> packed_u8,
                                    std::span<const std::uint8_t> block_scales_e4m3, float weight_scale_2, int m,
                                    int out_features, int in_features, std::span<float> out);
void simulate_warp_f32_fp8_linear(std::span<const float> x_in, std::span<const std::uint8_t> w_e4m3,
                                  float weight_scale, float input_scale, int m, int out_features, int in_features,
                                  std::span<float> out);

struct Qwen38Gate1RowStatus {
  std::string name;
  std::string status;  // "pass", "fail", "skipped_diagnostic", or "missing_checkpoint"
  bool passed = false;
};

struct Qwen38Gate1CoverageSummary {
  std::string mode;             // "qualification" or "diagnostic"
  std::string coverage_status;  // "complete_qualified", "incomplete_diagnostic", or "failed"
  bool complete_coverage = false;
  bool executed_checks_passed = false;
  bool qwen27b_pins_verified = false;
  bool qualification_passed = false;
  bool all_passed = false;
  std::size_t required_row_count = 0;
  std::size_t passed_required_row_count = 0;
  std::vector<std::string> missing_required_rows;
  std::vector<std::string> failed_rows;
};

const std::vector<std::string>& qwen38_gate1_required_row_names();
Qwen38Gate1CoverageSummary evaluate_gate1_coverage(std::span<const Qwen38Gate1RowStatus> rows, bool diagnostic_mode,
                                                   bool qwen27b_pins_verified);

struct Qwen38VectorCompareStats {
  float max_abs = 0.0f;
  float ref_max_abs = 0.0f;
  float rel_max_abs = 0.0f;
  float rel_l2 = 0.0f;
  double cosine_sim = 1.0;
  bool finite = true;
};

Qwen38VectorCompareStats compare_f32_vectors(std::span<const float> got, std::span<const float> ref);

enum class Qwen38BlockParityTier { SyntheticStrict, RealSingleLayer, RealStacked4Layer };

struct Qwen38BlockParityResult {
  Qwen38VectorCompareStats hidden;
  Qwen38Bf16CompareStats hidden_bf16;
  float layer0_gdn_max_abs = 0.0f;
  float stacked_gdn_max_abs = 0.0f;
  float stacked_gdn_rel_l2 = 0.0f;
  bool gdn_finite = true;
  std::string criterion;
  bool passed = false;
};

Qwen38BlockParityResult evaluate_block_parity(const Qwen38BlockBundleHost& bundle, const Qwen38BlockRunOutput& got,
                                              const Qwen38BlockRunOutput& ref, Qwen38BlockParityTier tier);

}  // namespace ingine::backends::qwen38

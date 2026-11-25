#pragma once

#include "backends/qwen38/qwen38_config.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ingine::backends::qwen38 {

float bf16_to_f32(std::uint16_t bits) noexcept;
std::uint16_t f32_to_bf16(float x) noexcept;
float decode_e4m3fn(std::uint8_t bits) noexcept;
std::uint8_t quantize_e4m3fn(float x) noexcept;
float decode_e2m1(std::uint8_t nibble) noexcept;

void decode_nvfp4_row(std::span<const std::uint8_t> packed_u8, std::span<const std::uint8_t> block_scale_e4m3,
                      float weight_scale_2, std::span<float> out_f32);

// Unquantized activation * FP8 weight (W8A32 weight-only reference).
void fp8_weight_only_linear(std::span<const float> x_in, std::span<const std::uint8_t> w_e4m3, float weight_scale,
                            int out_features, int in_features, std::span<float> out);

// Calibrated FP8 activation * FP8 weight linear matching qwen38_quant.py / Flare reference.
void fp8_calibrated_linear(std::span<const float> x_in, float input_scale, std::span<const std::uint8_t> w_e4m3,
                           float weight_scale, int out_features, int in_features, std::span<float> out);

// Raw NVFP4 weight-only linear (W4A32/W4A16 reference): out[r] = dot(x_in, decode_nvfp4_row(r)).
void nvfp4_weight_only_linear(std::span<const float> x_in, std::span<const std::uint8_t> packed_u8,
                              std::span<const std::uint8_t> block_scales_e4m3, float weight_scale_2, int out_features,
                              int in_features, std::span<float> out);

// Qwen3_5RMSNorm: (1.0f + weight[i]) * x[i] * rsqrt(mean(x^2) + eps).
void rmsnorm_zero_centered(std::span<const float> x, std::span<const float> weight, float eps, std::span<float> out);

// Qwen3_5RMSNormGated: weight[i] * (x[i] * rsqrt(mean(x^2) + eps)) * silu(gate_z[i]).
void rmsnorm_gated_silu(std::span<const float> x, std::span<const float> gate_z, std::span<const float> weight,
                        float eps, std::span<float> out);

// NeoX half-split partial RoPE over [0, rotary_dim), leaving [rotary_dim, head_dim) untouched.
void apply_partial_rope_neox(std::span<float> qk_heads, int num_heads, int head_dim, int rotary_dim, int pos,
                             float rope_theta);

// Causal depthwise conv1d (kernel_size=4) + SiLU for single decode step and contiguous prefill.
void causal_conv1d_silu_step(std::span<const float> x_conv_in, std::span<const float> conv_weight, int conv_dim,
                             int kernel_size, std::vector<float>& conv_state, std::span<float> out);

void causal_conv1d_silu_prefill(std::span<const float> x_seq, std::span<const float> conv_weight, int seq_len,
                                int conv_dim, int kernel_size, std::vector<float>& conv_state,
                                std::span<float> out_seq);

// Exact HF Qwen3_5 gated delta rule recurrent step (single token) and prefill (stepwise & chunked).
void gdn_recurrent_step(std::span<const float> q_kheads, std::span<const float> k_kheads,
                        std::span<const float> v_vheads, std::span<const float> z_vheads,
                        std::span<const float> a_vheads, std::span<const float> b_vheads, std::span<const float> a_log,
                        std::span<const float> dt_bias, std::span<const float> norm_weight, float rms_eps,
                        int num_k_heads, int num_v_heads, int k_dim, int v_dim, std::vector<float>& recurrent_state,
                        std::span<float> out_vdim);

void gdn_recurrent_prefill(std::span<const float> q_seq, std::span<const float> k_seq, std::span<const float> v_seq,
                           std::span<const float> z_seq, std::span<const float> a_seq, std::span<const float> b_seq,
                           std::span<const float> a_log, std::span<const float> dt_bias,
                           std::span<const float> norm_weight, float rms_eps, int seq_len, int num_k_heads,
                           int num_v_heads, int k_dim, int v_dim, std::vector<float>& recurrent_state,
                           std::span<float> out_seq);

void gdn_chunk_prefill(std::span<const float> q_seq, std::span<const float> k_seq, std::span<const float> v_seq,
                       std::span<const float> z_seq, std::span<const float> a_seq, std::span<const float> b_seq,
                       std::span<const float> a_log, std::span<const float> dt_bias,
                       std::span<const float> norm_weight, float rms_eps, int seq_len, int chunk_size,
                       int num_k_heads, int num_v_heads, int k_dim, int v_dim, std::vector<float>& recurrent_state,
                       std::span<float> out_seq);

// Full gated attention step (updates KV cache and attends causally over [0 .. pos]).
void full_attn_gated_step(std::span<const float> q_with_gate, std::span<const float> k_raw, std::span<const float> v,
                          std::span<const float> q_norm_w, std::span<const float> k_norm_w, float rms_eps,
                          float rope_theta, int num_heads, int num_kv_heads, int head_dim, int rotary_dim, int pos,
                          std::vector<float>& k_cache, std::vector<float>& v_cache, std::span<float> out_attn);

// Minimal deterministic CPU Qwen3.8 mini-model + opaque session state for Step-0 parity/reuse tests.
struct Qwen38MiniLayerWeights {
  Qwen38LayerKind kind = Qwen38LayerKind::LinearAttention;
  std::vector<float> input_layernorm;
  std::vector<float> post_attention_layernorm;

  // LinearAttention weights (FP8 projections for conv/z/out).
  std::vector<std::uint8_t> in_proj_qkv_fp8;
  float in_proj_qkv_wscale = 1.0f;
  std::vector<std::uint8_t> in_proj_z_fp8;
  float in_proj_z_wscale = 1.0f;
  std::vector<float> in_proj_a;
  std::vector<float> in_proj_b;
  std::vector<float> A_log;
  std::vector<float> dt_bias;
  std::vector<float> conv1d_weight;
  std::vector<float> linear_norm_weight;
  std::vector<std::uint8_t> out_proj_fp8;
  float out_proj_wscale = 1.0f;

  // FullAttention weights (FP8 projections).
  std::vector<std::uint8_t> q_proj_fp8;
  float q_proj_wscale = 1.0f;
  std::vector<std::uint8_t> k_proj_fp8;
  float k_proj_wscale = 1.0f;
  std::vector<std::uint8_t> v_proj_fp8;
  float v_proj_wscale = 1.0f;
  std::vector<float> q_norm;
  std::vector<float> k_norm;
  std::vector<std::uint8_t> o_proj_fp8;
  float o_proj_wscale = 1.0f;

  // SwiGLU MLP weights (NVFP4).
  std::vector<std::uint8_t> gate_nvfp4;
  std::vector<std::uint8_t> gate_scales;
  float gate_scale2 = 1.0f;
  std::vector<std::uint8_t> up_nvfp4;
  std::vector<std::uint8_t> up_scales;
  float up_scale2 = 1.0f;
  std::vector<std::uint8_t> down_nvfp4;
  std::vector<std::uint8_t> down_scales;
  float down_scale2 = 1.0f;
};

struct Qwen38MiniSessionSnapshot {
  std::vector<std::int32_t> committed_tokens;
  std::vector<std::vector<float>> conv_states;
  std::vector<std::vector<float>> recurrent_states;
  std::vector<std::vector<float>> k_caches;
  std::vector<std::vector<float>> v_caches;
};

class Qwen38MiniModel;

class Qwen38MiniSession {
 public:
  explicit Qwen38MiniSession(const Qwen38Config& cfg);

  void reset();
  [[nodiscard]] std::size_t state_bytes() const noexcept;
  [[nodiscard]] int position() const noexcept { return static_cast<int>(committed_tokens_.size()); }
  [[nodiscard]] const std::vector<std::int32_t>& committed_tokens() const noexcept { return committed_tokens_; }

  [[nodiscard]] Qwen38MiniSessionSnapshot snapshot() const;
  void restore(const Qwen38MiniSessionSnapshot& snap);

  std::vector<float> step(const Qwen38MiniModel& model, std::int32_t token_id);
  std::vector<float> prefill_last_logits(const Qwen38MiniModel& model, std::span<const std::int32_t> token_ids);

 private:
  friend class Qwen38MiniModel;
  Qwen38Config cfg_;
  std::vector<std::int32_t> committed_tokens_;
  std::vector<std::vector<float>> conv_states_;
  std::vector<std::vector<float>> recurrent_states_;
  std::vector<std::vector<float>> k_caches_;
  std::vector<std::vector<float>> v_caches_;
};

class Qwen38MiniModel {
 public:
  static Qwen38MiniModel make_synthetic(std::uint32_t seed = 20261125u);

  [[nodiscard]] const Qwen38Config& config() const noexcept { return cfg_; }
  std::vector<float> forward_token(Qwen38MiniSession& session, std::int32_t token_id) const;

 private:
  Qwen38Config cfg_;
  std::vector<float> embed_tokens_;
  std::vector<Qwen38MiniLayerWeights> layers_;
  std::vector<float> final_norm_;
  std::vector<std::uint8_t> lm_head_nvfp4_;
  std::vector<std::uint8_t> lm_head_scales_;
  float lm_head_scale2_ = 1.0f;
};

}  // namespace ingine::backends::qwen38

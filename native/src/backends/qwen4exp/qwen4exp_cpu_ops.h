// CUDA-free FP32 reference operators & stateful CPU oracle for Qwen3.8-Flash-Next (qwen4exp, #849).
// Matches llama.cpp-qwen38-flash-next-035e2273/src/models/qwen4exp.cpp op-for-op.
#pragma once

#include "backends/qwen4exp/qwen4exp_config.h"
#include "backends/qwen4exp/qwen4exp_gguf.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ingine::backends::qwen4exp {

// Standard RMSNorm: out[i] = w[i] * x[i] * rsqrt(mean(x^2) + eps).
// Note: GGUF qwen4exp norm weights already have (1 + w) folded by the converter.
void rmsnorm_f32(std::span<const float> x, std::span<const float> weight, float eps, std::span<float> out);

// Grouped RMSNorm over `groups` contiguous vectors of width `group_dim` with flat weight [groups * group_dim].
void grouped_rmsnorm_f32(std::span<const float> x, std::span<const float> weight, int groups, int group_dim, float eps,
                         std::span<float> out);

// Row-major matrix-vector multiply: y[r] = dot(W[r, :], x), W is [out_features, in_features].
void matvec_f32(std::span<const float> w, std::span<const float> x, int out_features, int in_features,
                std::span<float> y);

// Hyper-Connections mix (qwen4exp.cpp build_hc_mix):
//   x: [hc, H], w_norm: [hc*H], w_down: [hc_lr, hc*H], w_up: [hc*H, hc_lr], w_inject (optional): [hc, hc*H].
//   out_mixed: [H], out_inject (if w_inject non-empty): [hc].
void hc_mix_f32(std::span<const float> x, std::span<const float> w_norm, std::span<const float> w_down,
                std::span<const float> w_up, std::span<const float> w_inject, int hc, int hidden, int hc_lr, float eps,
                std::span<float> out_mixed, std::span<float> out_inject);

// Hyper-Connections combine (qwen4exp.cpp build_hc_combine):
//   out[c, i] = residual[c, i] + block_out[i] * (2 * sigmoid(inject[c] / hc)).
void hc_combine_f32(std::span<const float> residual, std::span<const float> block_out, std::span<const float> inject,
                    int hc, int hidden, std::span<float> out);

// Exact 64-bit XOR-multiply PLE n-gram hash matching llama.cpp llm_graph_input_ple::set_input.
// Updates `token_history` (length ngram_size - 1, initialized to ple_eos_token_id) and writes `ple_n_heads` row ids.
void ple_hash_step(const Qwen4ExpConfig& cfg, std::int32_t token_id, std::vector<std::int32_t>& token_history,
                   std::span<std::int32_t> out_rows);

// Stateful PLE module step (qwen4exp.cpp build_ple):
//   hidden_in: [hc, H], ple_emb: [H] (gathered from ple_n_heads rows of width ple_head_dim),
//   conv_state: [(ple_conv_kernel - 1) * ple_ngram_size, hc * H], updated in place.
void ple_step_f32(const Qwen4ExpConfig& cfg, std::span<const float> hidden_in, std::span<const float> ple_emb,
                  std::span<const float> w_key, std::span<const float> w_value, std::span<const float> w_norm_key,
                  std::span<const float> w_norm_query, std::span<const float> w_norm_conv,
                  std::span<const float> w_conv1d, std::vector<float>& ple_conv_state, std::span<float> hidden_out);

// Gated DeltaNet single-token step for qwen4exp (uses pre-folded negative `ssm_a` and SIGMOID output gate).
void gdn_step_sigmoid_f32(std::span<const float> qkv_mixed, std::span<const float> z_vdim,
                          std::span<const float> alpha_vheads, std::span<const float> beta_raw_vheads,
                          std::span<const float> ssm_conv_weight, std::span<const float> ssm_dt_bias,
                          std::span<const float> ssm_a_neg, std::span<const float> ssm_norm_weight, float rms_eps,
                          int conv_kernel, int num_k_heads, int num_v_heads, int head_dim,
                          std::vector<float>& conv_state, std::vector<float>& recurrent_state,
                          std::span<float> out_vdim);

// QSA indexer scoring & top-k selection for position `pos` (qwen4exp.cpp build_qsa_top_k + set_input_qsa).
// Appends `k_idx_raw` ([idx_dim]) into `idx_k_cache`, returns sorted selected token positions in [0 .. pos].
std::vector<int> qsa_select_positions_f32(std::span<const float> q_idx_raw, std::span<const float> k_idx_raw,
                                          std::span<const float> idx_q_norm, std::span<const float> idx_k_norm,
                                          float rms_eps, float rope_theta, int rotary_dim, int idx_heads, int idx_dim,
                                          int compress_ratio, int top_k_budget, int pos,
                                          std::vector<float>& idx_k_cache);

// Full gated GQA step restricted to `selected_positions` (or dense [0..pos] if `selected_positions` is empty).
// Appends K/V to caches and outputs pre-o_proj gated attention vector [num_heads * head_dim].
void full_attn_qsa_step_f32(std::span<const float> q_with_gate, std::span<const float> k_raw, std::span<const float> v,
                            std::span<const float> q_norm_w, std::span<const float> k_norm_w, float rms_eps,
                            float rope_theta, int num_heads, int num_kv_heads, int head_dim, int rotary_dim, int pos,
                            std::span<const int> selected_positions, std::vector<float>& k_cache,
                            std::vector<float>& v_cache, std::span<float> out_attn);

// Top-K Softmax router (softmax -> top-k -> renormalize * scale).
struct RouterChoice {
  int expert = 0;
  float weight = 0.0f;
};
std::vector<RouterChoice> route_top_k_softmax(std::span<const float> router_logits, int top_k, float weights_scale);

// SwiGLU single-expert FFN: down @ (silu(gate @ x) * (up @ x)).
void swiglu_ffn_f32(std::span<const float> x, std::span<const float> w_gate, std::span<const float> w_up,
                    std::span<const float> w_down, int hidden, int ffn_dim, std::span<float> out);

// Complete MoE + optional sigmoid-gated shared expert step (in-memory expert weights [E, ...]).
void moe_ffn_step_f32(std::span<const float> x, std::span<const float> w_router, std::span<const float> w_gate_exps,
                      std::span<const float> w_up_exps, std::span<const float> w_down_exps,
                      std::span<const float> w_sh_gate_inp, std::span<const float> w_sh_gate,
                      std::span<const float> w_sh_up, std::span<const float> w_sh_down, int hidden, int expert_count,
                      int expert_used, int expert_ffn, int shared_ffn, float weights_scale, std::span<float> out);

// Synthetic small Qwen4Exp CPU model + stateful session for end-to-end Phase 0 step-vs-chunk & QSA/PLE tests.
struct Qwen4ExpLayerWeightsF32 {
  Qwen4ExpLayerKind kind = Qwen4ExpLayerKind::LinearAttention;
  bool has_ple = false;

  std::vector<float> hc_attn_norm, hc_attn_down, hc_attn_up, hc_attn_inject;
  std::vector<float> hc_ffn_norm, hc_ffn_down, hc_ffn_up, hc_ffn_inject;

  // PLE (only when has_ple == true)
  std::vector<float> ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv1d;

  // LinearAttention (GDN)
  std::vector<float> w_qkv, w_gate, ssm_conv1d, ssm_dt_bias, ssm_a, ssm_beta, ssm_alpha, ssm_norm, ssm_out;

  // FullAttention + QSA indexer
  std::vector<float> w_q, w_k, w_v, w_o, attn_q_norm, attn_k_norm;
  std::vector<float> idx_q_proj, idx_k_proj, idx_q_norm, idx_k_norm;

  // MoE + shared expert
  std::vector<float> ffn_gate_inp;
  std::vector<float> ffn_gate_exps, ffn_up_exps, ffn_down_exps;
  std::vector<float> ffn_gate_inp_shexp, ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
};

struct Qwen4ExpSessionCpu {
  Qwen4ExpConfig cfg;
  std::vector<std::int32_t> committed_tokens;
  std::vector<std::int32_t> ple_token_history;
  std::vector<std::vector<float>> ple_conv_states;
  std::vector<std::vector<float>> gdn_conv_states;
  std::vector<std::vector<float>> gdn_recurrent_states;
  std::vector<std::vector<float>> k_caches;
  std::vector<std::vector<float>> v_caches;
  std::vector<std::vector<float>> idx_k_caches;

  explicit Qwen4ExpSessionCpu(const Qwen4ExpConfig& c);
  void reset();
  [[nodiscard]] int position() const noexcept { return static_cast<int>(committed_tokens.size()); }
};

class Qwen4ExpMiniModelCpu {
 public:
  static Qwen4ExpMiniModelCpu make_synthetic(std::uint32_t seed = 20261126u, int top_k_budget = 8);

  [[nodiscard]] const Qwen4ExpConfig& config() const noexcept { return cfg_; }
  std::vector<float> forward_token(Qwen4ExpSessionCpu& session, std::int32_t token_id) const;
  std::vector<float> forward_sequence_last_logits(Qwen4ExpSessionCpu& session,
                                                  std::span<const std::int32_t> token_ids) const;

 private:
  Qwen4ExpConfig cfg_;
  std::vector<float> token_embd_;
  std::vector<float> ple_table_;
  std::vector<Qwen4ExpLayerWeightsF32> layers_;
  std::vector<float> output_hc_norm_, output_hc_down_, output_hc_up_;
  std::vector<float> output_weight_;
};

// Executes a single real GGUF layer `il` on CPU using streaming POSIX pread for only the routed top-K experts
// and gathered PLE rows (works directly on 169 GB UD-Q6_K_XL shards in < 1 GiB RAM!).
void forward_gguf_layer_cpu(const Qwen4ExpGgufSet& gguf, Qwen4ExpSessionCpu& session, int layer_idx,
                            std::int32_t token_id, int pos, std::span<const float> res_hc_in,
                            std::span<float> res_hc_out);

// Streaming multi-threaded row-slab matvec directly against a quantized GGUF 2D tensor [in_features, out_features].
void matvec_gguf_tensor_f32(const Qwen4ExpGgufSet& gguf, const GgufTensorEntry& entry, std::span<const float> x,
                            std::span<float> y, int num_threads = 0);

// Full 48-layer streaming CPU forward for one token on a real Qwen4Exp GGUF set (updates `session` in place).
// When `compute_logits` is true, returns [vocab_size] FP32 logits; otherwise returns empty vector (faster for
// intermediate prompt tokens). Optionally records per-layer output L2 norms and final `result_norm` [hidden_size].
std::vector<float> forward_gguf_token_logits_cpu(const Qwen4ExpGgufSet& gguf, Qwen4ExpSessionCpu& session,
                                                 std::int32_t token_id, bool compute_logits = true,
                                                 std::vector<double>* layer_l2_out = nullptr,
                                                 std::vector<float>* result_norm_out = nullptr,
                                                 int num_threads = 0);

}  // namespace ingine::backends::qwen4exp

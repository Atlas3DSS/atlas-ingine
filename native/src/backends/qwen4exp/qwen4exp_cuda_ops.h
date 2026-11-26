// CUDA kernels for Qwen3.8-Flash-Next (qwen4exp, #849):
//   - Direct GGML quantized GEMV/GEMM (F32, F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K)
//   - Native Elastic Quant GEMV/GEMM (NVFP4 4.50 b/w, ATLAS3 3.50 b/w, ATLAS2 2.50 b/w)
//   - Hyper-Connections (grouped RMSNorm, low-rank gate collapse, 2*sigmoid combine)
//   - PLE Layer-1 step (signed-sqrt sigmoid gate + dilated causal depthwise conv1d + SiLU)
//   - Gated DeltaNet with sigmoid output gate (causal conv1d+SiLU, L2 norm, exact FP32 delta rule)
//   - Full Attention + QSA block-4 sparse indexer (partial NeoX RoPE, Q/K RMSNorm, gated GQA)
//   - Top-K Softmax MoE router + SwiGLU expert accumulation + sigmoid-gated shared expert
//
// Strictly portable across SM70 (Volta V100) through SM120a (Blackwell PRO 6000) using FP32 activations
// at operator boundaries for deterministic exact parity against qwen4exp_cpu_ops and llama.cpp.
#pragma once

#include "backends/qwen4exp/qwen4exp_config.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::backends::qwen4exp {

enum class ElasticWeightFormat : std::uint8_t {
  Nvfp4 = 0,   // E2M1 + E4M3 scale/16 + FP32 outer factor
  Atlas3 = 1,  // 8-centroid Lloyd-Max (12B/32) + E4M3 scale/16 + FP32 outer factor
  Atlas2 = 2,  // 4-centroid Lloyd-Max (4B/16) + E4M3 scale/16 + FP32 outer factor
};

// Direct GGML quantized row-major matrix-vector/matrix multiply:
//   y[m, r] = dot(dequant(W[r, :]), x[m, :])
// `w_raw` points to row-major GGML quantized weights of shape [out_features, in_features]
// (row stride = (in_features / block_elems) * block_bytes). Supports type_id: F32(0), F16(1),
// Q4_0(2), Q5_1(7), Q8_0(8), Q4_K(12), Q5_K(13), Q6_K(14), BF16(30).
void ggml_linear_f32_cuda(std::uint32_t type_id, const void* w_raw, const float* x, float* y, int m, int out_features,
                          int in_features, cudaStream_t stream);

// Native Elastic Quant linear multiply (matching tools/quantize_elastic_ingine.py & kernels/moe/dequant.cu):
//   y[m, r] = dot(dequant_elastic(codes[r, :], scales[r, :], factor), x[m, :])
// `factor_ptr` points to 1 FP32 scalar on device (or `factor_scalar` when `factor_ptr == nullptr`).
void elastic_linear_f32_cuda(ElasticWeightFormat fmt, const std::uint8_t* codes, const std::uint8_t* scales_e4m3,
                             const float* factor_ptr, float factor_scalar, const float* x, float* y, int m,
                             int out_features, int in_features, cudaStream_t stream);

// Weight-only FP8 E4M3FN linear multiply (used by dense projections in Elastic Qwen4Exp packs):
void fp8_linear_f32_cuda(const std::uint8_t* w_e4m3, float weight_scale, const float* x, float* y, int m,
                         int out_features, int in_features, cudaStream_t stream);

// Replicate token embedding [H] into wide Hyper-Connections residual [hc, H].
void hc_init_repeat_f32_cuda(const float* emb, float* res_hc, int hc, int hidden, cudaStream_t stream);

// Grouped RMSNorm: normalizes each group of `group_dim` elements independently and multiplies by flat `weight`.
void grouped_rmsnorm_f32_cuda(const float* x, const float* weight, float* out, int groups, int group_dim, float eps,
                              cudaStream_t stream);

// Hyper-Connections low-rank activation: lo[i] = silu(lo[i] / hc).
void hc_silu_scale_inplace_f32_cuda(float* lo, int n, int hc, cudaStream_t stream);

// Hyper-Connections gated mean collapse: mixed[i] = (1/hc) * sum_c xn[c, i] * sigmoid(gate[c, i]).
void hc_gate_collapse_f32_cuda(const float* xn, const float* gate, float* mixed, int hc, int hidden,
                               cudaStream_t stream);

// Hyper-Connections combine: out[c, i] = residual[c, i] + block_out[i] * (2 * sigmoid(inject[c] / hc)).
void hc_combine_f32_cuda(const float* residual, const float* block_out, const float* inject, float* out, int hc,
                         int hidden, cudaStream_t stream);

// PLE gate + broadcast value:
//   per stream c in [0..hc): s = dot(key[c], query[c]) / sqrt(H),
//   g = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6))), gated[c, i] = value[i] * g.
void ple_gate_value_f32_cuda(const float* key, const float* query, const float* value, float* gated, int hc, int hidden,
                             cudaStream_t stream);

// PLE dilated causal depthwise conv1d + SiLU + residual update:
//   updates `ple_conv_state [hist, hc_dim]` in place (shift left by 1, store `normed` at tail),
//   computes `hidden_out = hidden_in + gated + silu(conv_out)`.
void ple_conv_residual_step_f32_cuda(const float* hidden_in, const float* gated, const float* normed,
                                     const float* w_conv1d, float* ple_conv_state, float* hidden_out, int hc_dim,
                                     int kern, int dil, cudaStream_t stream);

// Gated DeltaNet single-token step with SIGMOID output gate:
//   1. Causal conv1d + SiLU over `qkv_mixed [conv_dim]` updating `conv_state [conv_kernel, conv_dim]`.
//   2. L2-norm on q/k, exact FP32 delta-rule update of `recurrent_state [num_v_heads, head_dim, head_dim]`,
//      RMSNorm(raw_out, ssm_norm_weight) * sigmoid(z_vdim).
void gdn_step_sigmoid_f32_cuda(const float* qkv_mixed, const float* z_vdim, const float* alpha_vheads,
                               const float* beta_raw_vheads, const float* ssm_conv_weight, const float* ssm_dt_bias,
                               const float* ssm_a_neg, const float* ssm_norm_weight, float rms_eps, int conv_kernel,
                               int num_k_heads, int num_v_heads, int head_dim, float* conv_state,
                               float* recurrent_state, float* conv_tmp, float* out_vdim, cudaStream_t stream);

// QSA indexer scoring kernel (appends raw k_idx at `pos`, computes block-4 pooled scores on GPU when sparse).
// Writes `n_blocks` block scores to `block_scores_out` (caller selects top blocks + tail, or dense when n_kv <= width).
void qsa_indexer_append_and_score_f32_cuda(const float* q_idx_raw, const float* k_idx_raw, const float* idx_q_norm,
                                           const float* idx_k_norm, float rms_eps, float rope_theta, int rotary_dim,
                                           int idx_heads, int idx_dim, int compress_ratio, int pos,
                                           float* idx_k_cache, float* block_scores_out, cudaStream_t stream);

// Full gated GQA single-token step restricted to `selected_positions` (`num_selected` ints on device,
// or dense `[0..pos]` when `selected_positions == nullptr`). Appends K/V into FP32 caches.
void full_attn_qsa_step_f32_cuda(const float* q_with_gate, const float* k_raw, const float* v_raw,
                                 const float* q_norm_w, const float* k_norm_w, float rms_eps, float rope_theta,
                                 int num_heads, int num_kv_heads, int head_dim, int rotary_dim, int pos,
                                 const int32_t* selected_positions, int num_selected, float* k_cache, float* v_cache,
                                 float* q_scratch, float* gate_scratch, float* out_attn, cudaStream_t stream);

// Top-K Softmax router on GPU (single token, `expert_count <= 1024`, `top_k <= 16`).
void route_top_k_softmax_f32_cuda(const float* router_logits, int expert_count, int top_k, float weights_scale,
                                  int32_t* out_experts, float* out_weights, cudaStream_t stream);

// Elementwise SwiGLU activation: `out[i] = silu(gate[i]) * up[i]`.
void swiglu_mul_f32_cuda(const float* gate, const float* up, float* out, int n, cudaStream_t stream);

// Weighted residual accumulate: `dst[i] += weight * src[i]`.
void axpy_f32_cuda(float* dst, const float* src, float weight, int n, cudaStream_t stream);

// Sigmoid-gated shared expert accumulate: `dst[i] += sigmoid(dot(gate_w, x)) * sh_out[i]`.
void shared_expert_gate_add_f32_cuda(float* dst, const float* sh_out, const float* gate_w, const float* x, int hidden,
                                     cudaStream_t stream);

}  // namespace ingine::backends::qwen4exp

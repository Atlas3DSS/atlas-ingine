#pragma once

#include "backends/qwen38/qwen38_block_cpu.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::backends::qwen38 {

// Gate 1 CUDA unit kernels (internal accumulation strictly FP32; F32 and BF16 boundary overloads).

void qwen38_cuda_f32_to_bf16(const float* src, __nv_bfloat16* dst, std::size_t n, cudaStream_t stream = nullptr);
void qwen38_cuda_bf16_to_f32(const __nv_bfloat16* src, float* dst, std::size_t n, cudaStream_t stream = nullptr);

void qwen38_cuda_nvfp4_linear_f32(const float* x, const std::uint8_t* w_packed, const std::uint8_t* block_scales_e4m3,
                                  float weight_scale_2, int m, int out_features, int in_features, float* out,
                                  cudaStream_t stream = nullptr);
void qwen38_cuda_nvfp4_linear_bf16(const __nv_bfloat16* x, const std::uint8_t* w_packed,
                                   const std::uint8_t* block_scales_e4m3, float weight_scale_2, int m, int out_features,
                                   int in_features, __nv_bfloat16* out, cudaStream_t stream = nullptr);

void qwen38_cuda_fp8_linear_f32(const float* x, const std::uint8_t* w_e4m3, float weight_scale, float input_scale,
                                int m, int out_features, int in_features, float* out, cudaStream_t stream = nullptr);
void qwen38_cuda_fp8_linear_bf16(const __nv_bfloat16* x, const std::uint8_t* w_e4m3, float weight_scale,
                                 float input_scale, int m, int out_features, int in_features, __nv_bfloat16* out,
                                 cudaStream_t stream = nullptr);

void qwen38_cuda_bf16_linear_f32(const float* x, const __nv_bfloat16* w_bf16, int m, int out_features, int in_features,
                                 float* out, cudaStream_t stream = nullptr);
void qwen38_cuda_bf16_linear_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w_bf16, int m, int out_features,
                                  int in_features, __nv_bfloat16* out, cudaStream_t stream = nullptr);

void qwen38_cuda_rmsnorm_zero_centered_f32(const float* x, const __nv_bfloat16* weight_bf16, float eps, int m, int dim,
                                           float* out, cudaStream_t stream = nullptr);
void qwen38_cuda_rmsnorm_zero_centered_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight_bf16, float eps, int m,
                                            int dim, __nv_bfloat16* out, cudaStream_t stream = nullptr);

void qwen38_cuda_rmsnorm_gated_silu_f32(const float* x, const float* gate_z, const float* weight, float eps, int m,
                                        int dim, float* out, cudaStream_t stream = nullptr);
void qwen38_cuda_rmsnorm_gated_silu_bf16(const __nv_bfloat16* x, const __nv_bfloat16* gate_z, const float* weight,
                                         float eps, int m, int dim, __nv_bfloat16* out, cudaStream_t stream = nullptr);

void qwen38_cuda_partial_rope_neox_f32(float* qk, int m, int num_heads, int head_dim, int rotary_dim, int pos_offset,
                                       float rope_theta, cudaStream_t stream = nullptr);
void qwen38_cuda_partial_rope_neox_bf16(__nv_bfloat16* qk, int m, int num_heads, int head_dim, int rotary_dim,
                                        int pos_offset, float rope_theta, cudaStream_t stream = nullptr);

void qwen38_cuda_causal_conv1d_silu_step_f32(const float* x_in, const float* conv_weight, int conv_dim, int kernel_size,
                                             float* conv_state, float* out, cudaStream_t stream = nullptr);
void qwen38_cuda_causal_conv1d_silu_step_bf16(const __nv_bfloat16* x_in, const float* conv_weight, int conv_dim,
                                              int kernel_size, float* conv_state, __nv_bfloat16* out,
                                              cudaStream_t stream = nullptr);
void qwen38_cuda_causal_conv1d_silu_prefill_f32(const float* x_seq, const float* conv_weight, int seq_len, int conv_dim,
                                                int kernel_size, float* conv_state, float* out_seq,
                                                cudaStream_t stream = nullptr);
void qwen38_cuda_causal_conv1d_silu_prefill_bf16(const __nv_bfloat16* x_seq, const float* conv_weight, int seq_len,
                                                 int conv_dim, int kernel_size, float* conv_state,
                                                 __nv_bfloat16* out_seq, cudaStream_t stream = nullptr);

void qwen38_cuda_gdn_recurrent_step_f32(const float* q_kheads, const float* k_kheads, const float* v_vheads,
                                        const float* z_vheads, const float* a_vheads, const float* b_vheads,
                                        const float* a_log, const float* dt_bias, const float* norm_weight,
                                        float rms_eps, int num_k_heads, int num_v_heads, int k_dim, int v_dim,
                                        float* recurrent_state, float* qk_norm_scratch, float* out_vdim,
                                        cudaStream_t stream = nullptr);
void qwen38_cuda_gdn_recurrent_step_bf16(const __nv_bfloat16* q_kheads, const __nv_bfloat16* k_kheads,
                                         const __nv_bfloat16* v_vheads, const __nv_bfloat16* z_vheads,
                                         const __nv_bfloat16* a_vheads, const __nv_bfloat16* b_vheads,
                                         const float* a_log, const float* dt_bias, const float* norm_weight,
                                         float rms_eps, int num_k_heads, int num_v_heads, int k_dim, int v_dim,
                                         float* recurrent_state, float* qk_norm_scratch, __nv_bfloat16* out_vdim,
                                         cudaStream_t stream = nullptr);

// Required scratch floats for qwen38_cuda_gdn_chunk_prefill*:
//   qn [seq_len * num_k_heads * k_dim] + kn [seq_len * num_k_heads * k_dim] +
//   w_mat [num_v_heads * chunk_size * k_dim] + v_new [num_v_heads * chunk_size * v_dim].
std::size_t qwen38_cuda_gdn_chunk_scratch_floats(int seq_len, int chunk_size, int num_k_heads, int num_v_heads,
                                                 int k_dim, int v_dim) noexcept;

void qwen38_cuda_gdn_chunk_prefill_f32(const float* q_seq, const float* k_seq, const float* v_seq, const float* z_seq,
                                       const float* a_seq, const float* b_seq, const float* a_log, const float* dt_bias,
                                       const float* norm_weight, float rms_eps, int seq_len, int chunk_size,
                                       int num_k_heads, int num_v_heads, int k_dim, int v_dim, float* recurrent_state,
                                       float* scratch, float* out_seq, cudaStream_t stream = nullptr);
void qwen38_cuda_gdn_chunk_prefill_bf16(const __nv_bfloat16* q_seq, const __nv_bfloat16* k_seq,
                                        const __nv_bfloat16* v_seq, const __nv_bfloat16* z_seq,
                                        const __nv_bfloat16* a_seq, const __nv_bfloat16* b_seq, const float* a_log,
                                        const float* dt_bias, const float* norm_weight, float rms_eps, int seq_len,
                                        int chunk_size, int num_k_heads, int num_v_heads, int k_dim, int v_dim,
                                        float* recurrent_state, float* scratch, __nv_bfloat16* out_seq,
                                        cudaStream_t stream = nullptr);

void qwen38_cuda_full_attn_gated_f32(const float* q_with_gate, const float* k_raw, const float* v_in,
                                     const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w, float rms_eps,
                                     float rope_theta, int seq_len, int pos_offset, int num_heads, int num_kv_heads,
                                     int head_dim, int rotary_dim, float* k_cache, float* v_cache, int max_cache_len,
                                     float* q_scratch, float* gate_scratch, float* out_attn,
                                     cudaStream_t stream = nullptr);
void qwen38_cuda_full_attn_gated_bf16(const __nv_bfloat16* q_with_gate, const __nv_bfloat16* k_raw,
                                      const __nv_bfloat16* v_in, const __nv_bfloat16* q_norm_w,
                                      const __nv_bfloat16* k_norm_w, float rms_eps, float rope_theta, int seq_len,
                                      int pos_offset, int num_heads, int num_kv_heads, int head_dim, int rotary_dim,
                                      __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int max_cache_len,
                                      __nv_bfloat16* q_scratch, __nv_bfloat16* gate_scratch, __nv_bfloat16* out_attn,
                                      cudaStream_t stream = nullptr);

void qwen38_cuda_swiglu_mul_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                                 std::size_t n, cudaStream_t stream = nullptr);
void qwen38_cuda_add_residual_bf16(__nv_bfloat16* hidden, const __nv_bfloat16* residual, std::size_t n,
                                   cudaStream_t stream = nullptr);
void qwen38_cuda_split_qkv_sep_bf16(const __nv_bfloat16* mixed, __nv_bfloat16* q_sep, __nv_bfloat16* k_sep,
                                    __nv_bfloat16* v_sep, int seq_len, int kd, int vd, cudaStream_t stream = nullptr);

// Phase 1B prefill & logit helpers: bit-exact BF16 weight/activation converters, gather, and M=1 F32 linear out.
void qwen38_cuda_dequant_nvfp4_to_bf16(const std::uint8_t* w_packed, const std::uint8_t* block_scales_e4m3, int n_rows,
                                       int in_features, __nv_bfloat16* out_bf16, cudaStream_t stream = nullptr);
void qwen38_cuda_dequant_fp8_to_bf16(const std::uint8_t* w_e4m3, int n_rows, int in_features, __nv_bfloat16* out_bf16,
                                     cudaStream_t stream = nullptr);
void qwen38_cuda_quant_fp8_act_to_bf16(const __nv_bfloat16* x_in, float input_scale, std::size_t n,
                                       __nv_bfloat16* x_out, cudaStream_t stream = nullptr);
void qwen38_cuda_embed_lookup_bf16(const __nv_bfloat16* embed_table, const std::int32_t* token_ids, int seq_len,
                                   int hidden_size, int vocab_size, __nv_bfloat16* out_hidden,
                                   cudaStream_t stream = nullptr);
void qwen38_cuda_gather_rows_bf16(const __nv_bfloat16* src, const int* row_indices, int num_rows, int hidden_size,
                                  __nv_bfloat16* dst, cudaStream_t stream = nullptr);
void qwen38_cuda_nvfp4_linear_bf16_to_f32(const __nv_bfloat16* x, const std::uint8_t* w_packed,
                                          const std::uint8_t* block_scales_e4m3, float weight_scale_2, int m,
                                          int out_features, int in_features, float* out_f32,
                                          cudaStream_t stream = nullptr);
void qwen38_cuda_bf16_linear_bf16_to_f32(const __nv_bfloat16* x, const __nv_bfloat16* w_bf16, int m, int out_features,
                                         int in_features, float* out_f32, cudaStream_t stream = nullptr);

// Runs a 1-layer or 4-layer block bundle on the active CUDA device with BF16 activation boundaries
// and F32 GDN recurrent state. Caller must ensure a valid CUDA device is active.
Qwen38BlockRunOutput run_qwen38_blocks_gpu(const Qwen38BlockBundleHost& bundle, bool use_chunk_prefill,
                                           int chunk_size = 64, bool use_calibrated_fp8_input_scale = true);

}  // namespace ingine::backends::qwen38

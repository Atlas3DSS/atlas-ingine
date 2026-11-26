// Elastic Quant (ATLAS3 / ATLAS2) linear & quantized embedding CUDA kernels for Qwen3.8-27B (#849).
// Uses the exact same Lloyd-Max codebooks and bit-packing as tools/quantize_elastic_qwen38.py
// and native/src/kernels/moe/dequant.cu. Orthogonal to qwen38_cuda_ops.{h,cu}.
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ingine::backends::qwen38 {

// Elastic linear forward: Y[M, N] = X[M, K] @ W_elastic[N, K]^T (FP32 in/out, SM70..SM120a).
//   ATLAS3: codes [N, 3*K/8] uint8, scales [N, K/16] fp8_e4m3fn, weight_scale_2 f32
//   ATLAS2: codes [N, K/4]   uint8, scales [N, K/16] fp8_e4m3fn, weight_scale_2 f32
void qwen38_cuda_atlas3_linear_f32_batched(const float* x, const std::uint8_t* codes, const std::uint8_t* block_scales,
                                           float weight_scale_2, int m, int n, int k, float* out, cudaStream_t stream);

void qwen38_cuda_atlas2_linear_f32_batched(const float* x, const std::uint8_t* codes, const std::uint8_t* block_scales,
                                           float weight_scale_2, int m, int n, int k, float* out, cudaStream_t stream);

// Quantized embedding lookup kernels (saves 1.27 GB VRAM in atlas-14g FP8, 1.83 GB in atlas-11g NVFP4):
void qwen38_cuda_embed_lookup_fp8_f32(const std::uint8_t* table_fp8, float weight_scale, const std::int32_t* token_ids,
                                      int m, int hidden_size, int vocab_size, float* out_f32, cudaStream_t stream);

void qwen38_cuda_embed_lookup_nvfp4_f32(const std::uint8_t* codes, const std::uint8_t* block_scales,
                                        float weight_scale_2, const std::int32_t* token_ids, int m, int hidden_size,
                                        int vocab_size, float* out_f32, cudaStream_t stream);

}  // namespace ingine::backends::qwen38

// Tight AVX2 / AVX-512 / NEON Q4_0, Q8_0, and EXL3 CPU microkernels with monotonic next-row prefetch.
// Adapted from alesha-pro/llm.asm 02-baremetal-os (Q8_0/prefetch/spin-pool) + ExLlamaV3/atlas-ingine EXL3.
// 100% CUDA-free; available in both INGINE_BACKEND=CPU and INGINE_BACKEND=CUDA (for hybrid host execution).
#pragma once

#include "core/cpu_pool.h"
#include "core/ops.h"

#include <cstddef>
#include <cstdint>

namespace ingine::ops::cpu {

enum class CpuIsa : std::uint8_t {
  Scalar = 0,
  Avx2 = 1,
  Avx512 = 2,
  Neon = 3,
};

CpuIsa detect_cpu_isa() noexcept;
const char* cpu_isa_name(CpuIsa isa) noexcept;

// Canonical Q8_0 block (32 values per block, 34 bytes/block: FP16 scale + 32 int8 quants).
struct BlockQ8_0 {
  std::uint16_t d_fp16;
  std::int8_t qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 packed bytes");

// --------------------------------------------------------------------------------------------- Q4_0 SoA kernels
// Bit-exact Q4_0 SoA row decoder matching ops::decode_q4_0_row canonical NaN/Inf/+0.0f semantics.
void decode_q4_0_row_simd(const std::uint8_t* row_bytes, int hidden_size, float* out_f32,
                          CpuIsa isa = detect_cpu_isa());

void decode_q4_0_rows(const std::uint8_t* table_q4, int rows, int hidden_size, float* out_f32,
                      CpuThreadPool* pool = nullptr, CpuIsa isa = detect_cpu_isa());

void embed_q4_rmsnorm_cpu(const EmbedQ4RmsNormDesc& desc, const EmbedQ4RmsNormArgs& args,
                          CpuThreadPool* pool = nullptr, CpuIsa isa = detect_cpu_isa());

// y[r] = dot(decode_q4_0(w_q4[r]), x) for r in [0, N), K % 32 == 0.
void gemv_q4_0_f32(const std::uint8_t* w_q4, const float* x, float* y, int N, int K,
                   CpuThreadPool* pool = nullptr, CpuIsa isa = detect_cpu_isa());

// ------------------------------------------------------------------------------------------------ Q8_0 kernels
void quantize_f32_to_q8_0(const float* x, BlockQ8_0* out_q8, int K);

// y[r] = dot(dequant_q8_0(w_q8[r]), x) for r in [0, N), K % 32 == 0.
void gemv_q8_0_f32(const BlockQ8_0* w_q8, const float* x, float* y, int N, int K,
                   CpuThreadPool* pool = nullptr, CpuIsa isa = detect_cpu_isa());

// llm.asm-style pure integer-dot Q8_0 x Q8_0 GEMV (AVX2 vpmaddubsw / AVX-512 / NEON).
void gemv_q8_0_q8_0(const BlockQ8_0* w_q8, const BlockQ8_0* x_q8, float* y, int N, int K,
                    CpuThreadPool* pool = nullptr, CpuIsa isa = detect_cpu_isa());

// --------------------------------------------------------------------------------------- EXL3 (2-bit & 3-bit)
// 128-point normalized Sylvester Fast Walsh-Hadamard Transform (involution: H_128(H_128(v)) == v).
void fwht_128_cpu(float* v, CpuIsa isa = detect_cpu_isa()) noexcept;

// Exact CPU mirror of kernels::moe::dequant_experts for EXL3_2 / EXL3_3:
// W = diag(rout) * H_128 * W_hat^T * H_128 * diag(rin), out_f32 [rows, cols].
void dequant_exl3_matrix_cpu(const std::uint8_t* codes, const std::uint16_t* rin_fp16,
                             const std::uint16_t* rout_fp16, int bits, int rows, int cols,
                             float* out_f32, CpuThreadPool* pool = nullptr,
                             CpuIsa isa = detect_cpu_isa());

// Zero-materialization EXL3 GEMV: y = diag(rout) * H_128 * (W_hat^T * (H_128 * (diag(rin) * x))).
void gemv_exl3_f32(const std::uint8_t* codes, const std::uint16_t* rin_fp16,
                   const std::uint16_t* rout_fp16, int bits, const float* x, float* y,
                   int rows, int cols, CpuThreadPool* pool = nullptr,
                   CpuIsa isa = detect_cpu_isa());

// Hybrid host-side SwiGLU MoE expert MLP in EXL3 encoding (single expert, W13 stacked [gate; up]):
//   h = silu(W_gate * x) * (W_up * x), out = W_down * h.
void moe_expert_mlp_exl3_cpu(const std::uint8_t* w13_codes, const std::uint16_t* w13_rin_fp16,
                             const std::uint16_t* w13_rout_fp16, int w13_bits,
                             const std::uint8_t* w2_codes, const std::uint16_t* w2_rin_fp16,
                             const std::uint16_t* w2_rout_fp16, int w2_bits, const float* x,
                             float* out, int hidden_size, int inter_size,
                             CpuThreadPool* pool = nullptr, CpuIsa isa = detect_cpu_isa());

}  // namespace ingine::ops::cpu

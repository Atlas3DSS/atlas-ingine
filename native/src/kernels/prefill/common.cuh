// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Device helpers of the prefill kernels: ldmatrix / mma.sync wrappers and cp.async (from fngine
// src/kernels/common/mma.cuh and memory.cuh), BF16/FP16/FP8 packing, swizzled shared-memory tiles.
#pragma once

#include "core/act.h"
#include "kernels/common/arch_ops.cuh"
#include "core/gpu_compat.h"
#include "kernels/common/copy.cuh"
#include "kernels/common/mma_sm70.cuh"

#include <cstdint>

namespace ingine::kernels::prefill {

__device__ __forceinline__ ingine::gpu::SmemAddr smem_addr(const void* ptr) {
  return ingine::gpu::make_smem_addr(ptr);
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            ingine::gpu::SmemAddr addr) {
  mma::ldmatrix_x4(r0, r1, r2, r3, addr);
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                              ingine::gpu::SmemAddr addr) {
  mma::ldmatrix_x4_trans(r0, r1, r2, r3, addr);
}

// D += A * B, m16n8k16, ActT inputs (BF16 on sm_80+, FP16 on INGINE_ACT_FP16 / sm_70), FP32 accumulation.
__device__ __forceinline__ void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  mma::mma_act(c, a[0], a[1], a[2], a[3], b0, b1);
}

// D += A * B, m16n8k16, FP16 inputs, FP32 accumulation.
__device__ __forceinline__ void mma_f16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
  mma::mma_f16(c, a[0], a[1], a[2], a[3], b0, b1);
}

// 16-byte cp.async (L2 only); src_bytes < 16 zero-fills the rest (0: nothing is read).
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gmem_src, int src_bytes = 16) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :
               : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
#else
  copy::sync<16>(smem_dst, gmem_src, src_bytes);
#endif
}

__device__ __forceinline__ void cp_commit() { copy::commit(); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
  copy::wait<Groups>();
}

__device__ __forceinline__ float bf_lo(uint32_t u) { return act::lo_f(u); }
__device__ __forceinline__ float bf_hi(uint32_t u) { return act::hi_f(u); }
__device__ __forceinline__ float bf2f(uint16_t u) { return act::bits_to_float(u); }
__device__ __forceinline__ float round_bf16(float x) { return act::round_act(x); }

__device__ __forceinline__ uint16_t f2bf_bits(float x) {
  return act::float_to_bits(x);
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  return act::pack2(lo, hi);
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// 2 E4M3 codes (low byte first) -> 2 floats (exact via FP16).
__device__ __forceinline__ float2 e4m3x2_to_float2(uint16_t u) {
  const uint32_t h2 = act::e4m3x2_to_half2_bits(u);
  return __half22float2(*reinterpret_cast<const __half2*>(&h2));
}

// 2 E4M3 codes -> packed FP16 pair (exact).
__device__ __forceinline__ uint32_t e4m3x2_to_half2(uint16_t u) {
  return act::e4m3x2_to_half2_bits(u);
}

// float -> E4M3 code, round to nearest even after saturating to +-448 (torch clamp + .to(float8_e4m3fn)).
__device__ __forceinline__ uint8_t f2e4m3(float x) {
  return ingine::gpu::float_to_e4m3fn_sat(fminf(fmaxf(x, -448.f), 448.f));
}

// Swizzled [rows][Chunks x 16 B] tile: element offset (in 16-bit units) of 16-byte chunk `chunk` of `row`.
// XOR with row & 7 spreads the 8 rows of an ldmatrix 8x8 matrix over distinct banks.
template <int Chunks>
__device__ __forceinline__ int swz(int row, int chunk) {
  return row * (Chunks * 8) + ((chunk ^ (row & 7)) << 3);
}

}  // namespace ingine::kernels::prefill

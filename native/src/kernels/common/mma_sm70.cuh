// FP16 tensor-core and FP32 SIMT implementations of the shared MMA fragment layout.
//
// Fragment conventions match PTX mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}.{bf16,f16}.f32:
//   lane = 4 * g + t   (g = lane >> 2 in 0..7, t = lane & 3 in 0..3)
//   a0 = A[g,     2t..2t+1],  a1 = A[g + 8, 2t..2t+1]
//   a2 = A[g,   8+2t..9+2t],  a3 = A[g + 8, 8+2t..9+2t]
//   b0 = B[g,     2t..2t+1],  b1 = B[g,   8+2t..9+2t]   (B col-major: col = g, k = 0..15)
//   d[0] = D[g,     2t],      d[1] = D[g,     2t + 1]
//   d[2] = D[g + 8, 2t],      d[3] = D[g + 8, 2t + 1]
//
// On Volta sm_70, mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 executes 4 independent 8x8x4 tiles
// across the warp's 4 quadpairs (QP0..QP3). Mapping:
//   QP0: Top 8x8 (rows 0..7),  k in {0..3,  8..11}
//   QP1: Top 8x8 (rows 0..7),  k in {4..7, 12..15}
//   QP2: Bot 8x8 (rows 8..15), k in {0..3,  8..11}
//   QP3: Bot 8x8 (rows 8..15), k in {4..7, 12..15}
// Two m8n8k4 instructions cover the 16x8x16 tile.
#pragma once

#include "core/act.h"
#include "core/gpu_compat.h"

#include <cstdint>

namespace ingine::kernels::mma {

__device__ __forceinline__ std::uint32_t ld_smem_u32(ingine::gpu::SmemAddr saddr) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  std::uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(saddr));
  return v;
#else
  return *reinterpret_cast<const std::uint32_t*>(saddr);
#endif
}

__device__ __forceinline__ uint2 ld_smem_u64(ingine::gpu::SmemAddr saddr) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(saddr));
  return v;
#else
  return *reinterpret_cast<const uint2*>(saddr);
#endif
}

__device__ __forceinline__ std::uint16_t ld_smem_u16(ingine::gpu::SmemAddr saddr) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  std::uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(saddr));
  return v;
#else
  return *reinterpret_cast<const std::uint16_t*>(saddr);
#endif
}

__device__ __forceinline__ ingine::gpu::SmemAddr shfl_smem_addr(ingine::gpu::SmemAddr saddr, int src_lane) {
#if defined(INGINE_BACKEND_HIP) && INGINE_BACKEND_HIP
  const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(saddr);
  const std::uint32_t lo = __shfl_sync(0xffffffffu, static_cast<std::uint32_t>(raw), src_lane);
  const std::uint32_t hi = __shfl_sync(0xffffffffu, static_cast<std::uint32_t>(raw >> 32), src_lane);
  return reinterpret_cast<ingine::gpu::SmemAddr>((static_cast<std::uintptr_t>(hi) << 32) | lo);
#else
  return __shfl_sync(0xffffffffu, saddr, src_lane);
#endif
}

#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
// Native Volta sm_70 HMMA.884 primitives: keep acc[8] across entire K-loops and eliminate
// per-step accumulator reductions (and per-step input shuffles when A/B are loaded directly).
__device__ __forceinline__ void sm70_mma_f16_direct(float (&acc)[8], unsigned ma0, unsigned ma1, unsigned na0,
                                                    unsigned na1, unsigned mb0, unsigned mb1, unsigned nb0,
                                                    unsigned nb1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
               : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
                 "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
               : "r"(ma0), "r"(ma1), "r"(mb0), "r"(mb1));
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
               : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
                 "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
               : "r"(na0), "r"(na1), "r"(nb0), "r"(nb1));
}

__device__ __forceinline__ void sm70_permute_a(unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned& ma0, unsigned& ma1, unsigned& na0, unsigned& na1) {
  const int lane = threadIdx.x & 31;
  const int qp = (lane >> 2) & 3;
  const int idx = (lane & 3) | ((lane & 16) >> 2);
  const int src0 = (idx << 2) | ((qp & 1) << 1);
  const int src1 = src0 | 1;
  const unsigned a0_s0 = __shfl_sync(0xffffffffu, a0, src0);
  const unsigned a0_s1 = __shfl_sync(0xffffffffu, a0, src1);
  const unsigned a1_s0 = __shfl_sync(0xffffffffu, a1, src0);
  const unsigned a1_s1 = __shfl_sync(0xffffffffu, a1, src1);
  ma0 = qp < 2 ? a0_s0 : a1_s0;
  ma1 = qp < 2 ? a0_s1 : a1_s1;
  const unsigned a2_s0 = __shfl_sync(0xffffffffu, a2, src0);
  const unsigned a2_s1 = __shfl_sync(0xffffffffu, a2, src1);
  const unsigned a3_s0 = __shfl_sync(0xffffffffu, a3, src0);
  const unsigned a3_s1 = __shfl_sync(0xffffffffu, a3, src1);
  na0 = qp < 2 ? a2_s0 : a3_s0;
  na1 = qp < 2 ? a2_s1 : a3_s1;
}

__device__ __forceinline__ void sm70_mma_f16_acc(float (&acc)[8], unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                                 unsigned b0, unsigned b1) {
  const int lane = threadIdx.x & 31;
  const int qp = (lane >> 2) & 3;
  const int idx = (lane & 3) | ((lane & 16) >> 2);
  const int src0 = (idx << 2) | ((qp & 1) << 1);
  const int src1 = src0 | 1;
  unsigned ma0, ma1, na0, na1;
  sm70_permute_a(a0, a1, a2, a3, ma0, ma1, na0, na1);
  const unsigned mb0 = __shfl_sync(0xffffffffu, b0, src0);
  const unsigned mb1 = __shfl_sync(0xffffffffu, b0, src1);
  const unsigned nb0 = __shfl_sync(0xffffffffu, b1, src0);
  const unsigned nb1 = __shfl_sync(0xffffffffu, b1, src1);
  sm70_mma_f16_direct(acc, ma0, ma1, na0, na1, mb0, mb1, nb0, nb1);
}

__device__ __forceinline__ void sm70_reduce_acc8_to_d4(const float (&acc)[8], float (&d)[4]) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int qp = g & 3;
  const float p0 = acc[0] + __shfl_xor_sync(0xffffffffu, acc[0], 4);
  const float p1 = acc[1] + __shfl_xor_sync(0xffffffffu, acc[1], 4);
  const float p2 = acc[2] + __shfl_xor_sync(0xffffffffu, acc[2], 4);
  const float p3 = acc[3] + __shfl_xor_sync(0xffffffffu, acc[3], 4);
  const float p4 = acc[4] + __shfl_xor_sync(0xffffffffu, acc[4], 4);
  const float p5 = acc[5] + __shfl_xor_sync(0xffffffffu, acc[5], 4);
  const float p6 = acc[6] + __shfl_xor_sync(0xffffffffu, acc[6], 4);
  const float p7 = acc[7] + __shfl_xor_sync(0xffffffffu, acc[7], 4);
  const bool odd_qp = (qp & 1) != 0;
  const float s0 = odd_qp ? p4 : p0;
  const float s1 = odd_qp ? p5 : p1;
  const float s2 = odd_qp ? p6 : p2;
  const float s3 = odd_qp ? p7 : p3;
  const int owner_top = ((g & 4) << 2) | (t << 1) | (g & 1);
  const int owner_bot = owner_top | 8;
  const bool hi_row = (g & 2) != 0;

  const float t0 = __shfl_sync(0xffffffffu, s0, owner_top);
  const float t1 = __shfl_sync(0xffffffffu, s1, owner_top);
  const float t2 = __shfl_sync(0xffffffffu, s2, owner_top);
  const float t3 = __shfl_sync(0xffffffffu, s3, owner_top);
  d[0] += hi_row ? t2 : t0;
  d[1] += hi_row ? t3 : t1;

  const float b0_f = __shfl_sync(0xffffffffu, s0, owner_bot);
  const float b1_f = __shfl_sync(0xffffffffu, s1, owner_bot);
  const float b2_f = __shfl_sync(0xffffffffu, s2, owner_bot);
  const float b3_f = __shfl_sync(0xffffffffu, s3, owner_bot);
  d[2] += hi_row ? b2_f : b0_f;
  d[3] += hi_row ? b3_f : b1_f;
}
#endif

#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__) && !defined(INGINE_FORCE_SIMT_MMA)
// Native AMD RDNA 3 / RDNA 3.5 (gfx1100/gfx1101/gfx1151) Wave32 16x16x16 WMMA primitives.
// Exact operand layout (AMD RDNA 3 ISA / Matrix Instruction Calculator):
//   A [16x16]: lane l holds row m = l & 15, k = 0..15 (lanes 0..15 MUST be replicated in lanes 16..31)
//   B [16x16]: lane l holds col n = l & 15, k = 0..15 (lanes 0..15 MUST be replicated in lanes 16..31)
//   D [16x16]: D[2*i + (l >> 4)][l & 15] = acc[i] for i = 0..7  (D = A * B + C)
using Gfx11Bf16Frag = short __attribute__((ext_vector_type(16)));
using Gfx11F16Frag  = _Float16 __attribute__((ext_vector_type(16)));
using Gfx11Acc8     = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ Gfx11Acc8 gfx11_acc_zero() {
  return Gfx11Acc8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
}

__device__ __forceinline__ void gfx11_wmma_bf16(Gfx11Acc8& acc, const std::uint32_t (&a)[8],
                                                const std::uint32_t (&b)[8]) {
  Gfx11Bf16Frag af, bf;
  __builtin_memcpy(&af, a, sizeof(af));
  __builtin_memcpy(&bf, b, sizeof(bf));
  acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, acc);
}

__device__ __forceinline__ void gfx11_wmma_f16(Gfx11Acc8& acc, const std::uint32_t (&a)[8],
                                               const std::uint32_t (&b)[8]) {
  Gfx11F16Frag af, bf;
  __builtin_memcpy(&af, a, sizeof(af));
  __builtin_memcpy(&bf, b, sizeof(bf));
  acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc);
}

__device__ __forceinline__ void gfx11_wmma_act(Gfx11Acc8& acc, const std::uint32_t (&a)[8],
                                               const std::uint32_t (&b)[8]) {
#if INGINE_ACT_FP16
  gfx11_wmma_f16(acc, a, b);
#else
  gfx11_wmma_bf16(acc, a, b);
#endif
}

// Permute PTX m16n8k16 A/B fragments into RDNA 3 16x16x16 WMMA fragments (replicated across wave halves)
// and accumulate into Gfx11Acc8. Columns 8..15 of B are duplicated (ignored on extraction).
template <bool IS_BF16>
__device__ __forceinline__ void gfx11_wmma_m16n8k16_acc(Gfx11Acc8& acc, unsigned a0, unsigned a1, unsigned a2,
                                                        unsigned a3, unsigned b0, unsigned b1) {
  const int lane = threadIdx.x & 31;
  const int row_g = lane & 7;
  const bool hi_m = (lane & 8) != 0;
  std::uint32_t af[8], bf[8];
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const int src = (row_g << 2) | p;
    const unsigned a0_s = __shfl_sync(0xffffffffu, a0, src);
    const unsigned a1_s = __shfl_sync(0xffffffffu, a1, src);
    const unsigned a2_s = __shfl_sync(0xffffffffu, a2, src);
    const unsigned a3_s = __shfl_sync(0xffffffffu, a3, src);
    af[p] = hi_m ? a1_s : a0_s;
    af[4 + p] = hi_m ? a3_s : a2_s;
    bf[p] = __shfl_sync(0xffffffffu, b0, src);
    bf[4 + p] = __shfl_sync(0xffffffffu, b1, src);
  }
  if constexpr (IS_BF16) {
    gfx11_wmma_bf16(acc, af, bf);
  } else {
    gfx11_wmma_f16(acc, af, bf);
  }
}

// Extract PTX m16n8k16 output d[4] (D[g, 2t], D[g, 2t+1], D[g+8, 2t], D[g+8, 2t+1]) from Gfx11Acc8.
__device__ __forceinline__ void gfx11_reduce_acc8_to_d4(const Gfx11Acc8& acc, float (&d)[4]) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int reg_lo = g >> 1;  // 0..3 -> rows g = 2*reg_lo + (g&1), g+8 = 2*(reg_lo+4) + (g&1)
  const int src0 = ((g & 1) << 4) | (t << 1);
  const int src1 = src0 | 1;
  float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float lo0 = __shfl_sync(0xffffffffu, acc[r], src0);
    const float lo1 = __shfl_sync(0xffffffffu, acc[r], src1);
    const float hi0 = __shfl_sync(0xffffffffu, acc[r + 4], src0);
    const float hi1 = __shfl_sync(0xffffffffu, acc[r + 4], src1);
    if (reg_lo == r) {
      v0 = lo0;
      v1 = lo1;
      v2 = hi0;
      v3 = hi1;
    }
  }
  d[0] += v0;
  d[1] += v1;
  d[2] += v2;
  d[3] += v3;
}
#endif

// Drop-in replacement for ldmatrix.sync.aligned.m8n8.x4.shared.b16
__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            ingine::gpu::SmemAddr saddr) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && __CUDA_ARCH__ >= 750
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(saddr)
               : "memory");
#else
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const unsigned off = static_cast<unsigned>(lane & 3) * 4u;
  r0 = ld_smem_u32(shfl_smem_addr(saddr, g) + off);
  r1 = ld_smem_u32(shfl_smem_addr(saddr, 8 + g) + off);
  r2 = ld_smem_u32(shfl_smem_addr(saddr, 16 + g) + off);
  r3 = ld_smem_u32(shfl_smem_addr(saddr, 24 + g) + off);
#endif
}

// Drop-in replacement for ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                                  ingine::gpu::SmemAddr saddr) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && __CUDA_ARCH__ >= 750
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(saddr)
               : "memory");
#else
  const int lane = threadIdx.x & 31;
  const unsigned off = static_cast<unsigned>(lane >> 2) * 2u;  // element g
  const int row0 = (lane & 3) * 2;
  const int row1 = row0 + 1;
  auto load_t = [&](int base) -> unsigned {
    const ingine::gpu::SmemAddr a0 = shfl_smem_addr(saddr, base + row0);
    const ingine::gpu::SmemAddr a1 = shfl_smem_addr(saddr, base + row1);
    return static_cast<unsigned>(ld_smem_u16(a0 + off)) | (static_cast<unsigned>(ld_smem_u16(a1 + off)) << 16);
  };
  r0 = load_t(0);
  r1 = load_t(8);
  r2 = load_t(16);
  r3 = load_t(24);
#endif
}

// Drop-in replacement for ldmatrix.sync.aligned.m8n8.x2.shared.b16
__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, ingine::gpu::SmemAddr saddr) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && __CUDA_ARCH__ >= 750
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(saddr)
               : "memory");
#else
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const unsigned off = static_cast<unsigned>(lane & 3) * 4u;
  r0 = ld_smem_u32(shfl_smem_addr(saddr, g) + off);
  r1 = ld_smem_u32(shfl_smem_addr(saddr, 8 + g) + off);
#endif
}

// D += A * B, m16n8k16, FP16 inputs, FP32 accumulation (native m16n8k16 on sm_80+, native HMMA.884 on sm_70,
// native RDNA 3 WMMA on gfx11, portable wave32 SIMT on other HIP targets and pre-sm70).
__device__ __forceinline__ void mma_f16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__) && !defined(INGINE_FORCE_SIMT_MMA)
  Gfx11Acc8 acc = gfx11_acc_zero();
  gfx11_wmma_m16n8k16_acc<false>(acc, a0, a1, a2, a3, b0, b1);
  gfx11_reduce_acc8_to_d4(acc, d);
#elif defined(__HIP_DEVICE_COMPILE__) || !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA)
  // SIMT seam for targets without tensor cores. Each lane owns two columns
  // in two rows of the m16n8k16 tile; exchange the packed K pairs within the warp.
  const int row = (threadIdx.x & 31) / 4;
  const int col = (threadIdx.x & 3) * 2;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const int a_lane = 4 * row + (pair & 3);
    const int b_lane = 4 * col + (pair & 3);
    const unsigned top = __shfl_sync(0xffffffffu, pair < 4 ? a0 : a2, a_lane);
    const unsigned bottom = __shfl_sync(0xffffffffu, pair < 4 ? a1 : a3, a_lane);
    const unsigned left = __shfl_sync(0xffffffffu, pair < 4 ? b0 : b1, b_lane);
    const unsigned right = __shfl_sync(0xffffffffu, pair < 4 ? b0 : b1, b_lane + 4);
#pragma unroll
    for (int element = 0; element < 2; ++element) {
      const float x0 = __half2float(__ushort_as_half(top >> (16 * element)));
      const float x1 = __half2float(__ushort_as_half(bottom >> (16 * element)));
      const float y0 = __half2float(__ushort_as_half(left >> (16 * element)));
      const float y1 = __half2float(__ushort_as_half(right >> (16 * element)));
      d[0] = fmaf(x0, y0, d[0]);
      d[1] = fmaf(x0, y1, d[1]);
      d[2] = fmaf(x1, y0, d[2]);
      d[3] = fmaf(x1, y1, d[3]);
    }
  }
#elif __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif __CUDA_ARCH__ >= 750
  // Native Turing sm_75 HMMA.1688: two m16n8k8 instructions match m16n8k16 fragment layout with zero shuffles.
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(b0));
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a2), "r"(a3), "r"(b1));
#elif defined(INGINE_BACKEND_CUDA) && INGINE_BACKEND_CUDA
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int qp = g & 3;                          // quadpair 0..3
  const int idx = t | ((lane & 16) >> 2);        // 0..7 within quadpair
  const int src0 = (idx << 2) | ((qp & 1) << 1); // m16n8k16 lane holding k_pair 0 (even qp) or 2 (odd qp)
  const int src1 = src0 | 1;                     // m16n8k16 lane holding k_pair 1 (even qp) or 3 (odd qp)

  const unsigned a0_s0 = __shfl_sync(0xffffffffu, a0, src0);
  const unsigned a0_s1 = __shfl_sync(0xffffffffu, a0, src1);
  const unsigned a1_s0 = __shfl_sync(0xffffffffu, a1, src0);
  const unsigned a1_s1 = __shfl_sync(0xffffffffu, a1, src1);
  const unsigned ma0 = qp < 2 ? a0_s0 : a1_s0;
  const unsigned ma1 = qp < 2 ? a0_s1 : a1_s1;
  const unsigned mb0 = __shfl_sync(0xffffffffu, b0, src0);
  const unsigned mb1 = __shfl_sync(0xffffffffu, b0, src1);

  const unsigned a2_s0 = __shfl_sync(0xffffffffu, a2, src0);
  const unsigned a2_s1 = __shfl_sync(0xffffffffu, a2, src1);
  const unsigned a3_s0 = __shfl_sync(0xffffffffu, a3, src0);
  const unsigned a3_s1 = __shfl_sync(0xffffffffu, a3, src1);
  const unsigned na0 = qp < 2 ? a2_s0 : a3_s0;
  const unsigned na1 = qp < 2 ? a2_s1 : a3_s1;
  const unsigned nb0 = __shfl_sync(0xffffffffu, b1, src0);
  const unsigned nb1 = __shfl_sync(0xffffffffu, b1, src1);

  float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
               : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
                 "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
               : "r"(ma0), "r"(ma1), "r"(mb0), "r"(mb1));
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
               : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
                 "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
               : "r"(na0), "r"(na1), "r"(nb0), "r"(nb1));

  // Reduce QP0+QP1 (Top 8x8) and QP2+QP3 (Bot 8x8) warp-convergently. Even QPs retain cols 0..3 (p0..p3),
  // odd QPs retain cols 4..7 (p4..p7) so each target lane needs only 4 shuffles per half.
  const float p0 = acc[0] + __shfl_xor_sync(0xffffffffu, acc[0], 4);
  const float p1 = acc[1] + __shfl_xor_sync(0xffffffffu, acc[1], 4);
  const float p2 = acc[2] + __shfl_xor_sync(0xffffffffu, acc[2], 4);
  const float p3 = acc[3] + __shfl_xor_sync(0xffffffffu, acc[3], 4);
  const float p4 = acc[4] + __shfl_xor_sync(0xffffffffu, acc[4], 4);
  const float p5 = acc[5] + __shfl_xor_sync(0xffffffffu, acc[5], 4);
  const float p6 = acc[6] + __shfl_xor_sync(0xffffffffu, acc[6], 4);
  const float p7 = acc[7] + __shfl_xor_sync(0xffffffffu, acc[7], 4);
  const bool odd_qp = (qp & 1) != 0;
  const float s0 = odd_qp ? p4 : p0;
  const float s1 = odd_qp ? p5 : p1;
  const float s2 = odd_qp ? p6 : p2;
  const float s3 = odd_qp ? p7 : p3;
  const int owner_top = ((g & 4) << 2) | (t << 1) | (g & 1);
  const int owner_bot = owner_top | 8;
  const bool hi_row = (g & 2) != 0;

  const float t0 = __shfl_sync(0xffffffffu, s0, owner_top);
  const float t1 = __shfl_sync(0xffffffffu, s1, owner_top);
  const float t2 = __shfl_sync(0xffffffffu, s2, owner_top);
  const float t3 = __shfl_sync(0xffffffffu, s3, owner_top);
  d[0] += hi_row ? t2 : t0;
  d[1] += hi_row ? t3 : t1;

  const float b0_f = __shfl_sync(0xffffffffu, s0, owner_bot);
  const float b1_f = __shfl_sync(0xffffffffu, s1, owner_bot);
  const float b2_f = __shfl_sync(0xffffffffu, s2, owner_bot);
  const float b3_f = __shfl_sync(0xffffffffu, s3, owner_bot);
  d[2] += hi_row ? b2_f : b0_f;
  d[3] += hi_row ? b3_f : b1_f;
#endif
}

// D += A * B, m16n8k16, ActT inputs (FP16 on INGINE_ACT_FP16 / sm_70, BF16 on sm_80+ and HIP BF16), FP32 accumulation.
__device__ __forceinline__ void mma_act(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if INGINE_ACT_FP16 || (defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && __CUDA_ARCH__ < 800)
  mma_f16(d, a0, a1, a2, a3, b0, b1);
#elif defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#elif defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__) && !defined(INGINE_FORCE_SIMT_MMA)
  Gfx11Acc8 acc = gfx11_acc_zero();
  gfx11_wmma_m16n8k16_acc<true>(acc, a0, a1, a2, a3, b0, b1);
  gfx11_reduce_acc8_to_d4(acc, d);
#else
  // Portable BF16 m16n8k16 with exact FP32 accumulation for HIP (matches CUDA fragment layout).
  const int row = (threadIdx.x & 31) / 4;
  const int col = (threadIdx.x & 3) * 2;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    const int a_lane = 4 * row + (pair & 3);
    const int b_lane = 4 * col + (pair & 3);
    const unsigned top = __shfl_sync(0xffffffffu, pair < 4 ? a0 : a2, a_lane);
    const unsigned bottom = __shfl_sync(0xffffffffu, pair < 4 ? a1 : a3, a_lane);
    const unsigned left = __shfl_sync(0xffffffffu, pair < 4 ? b0 : b1, b_lane);
    const unsigned right = __shfl_sync(0xffffffffu, pair < 4 ? b0 : b1, b_lane + 4);
#pragma unroll
    for (int element = 0; element < 2; ++element) {
      const float x0 = act::bits_to_float(static_cast<std::uint16_t>(top >> (16 * element)));
      const float x1 = act::bits_to_float(static_cast<std::uint16_t>(bottom >> (16 * element)));
      const float y0 = act::bits_to_float(static_cast<std::uint16_t>(left >> (16 * element)));
      const float y1 = act::bits_to_float(static_cast<std::uint16_t>(right >> (16 * element)));
      d[0] = fmaf(x0, y0, d[0]);
      d[1] = fmaf(x0, y1, d[1]);
      d[2] = fmaf(x1, y0, d[2]);
      d[3] = fmaf(x1, y1, d[3]);
    }
  }
#endif
}

}  // namespace ingine::kernels::mma

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

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::mma {

__device__ __forceinline__ std::uint32_t ld_smem_u32(unsigned saddr) {
  std::uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(saddr));
  return v;
}

__device__ __forceinline__ uint2 ld_smem_u64(unsigned saddr) {
  uint2 v;
  asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(saddr));
  return v;
}

__device__ __forceinline__ std::uint16_t ld_smem_u16(unsigned saddr) {
  std::uint16_t v;
  asm volatile("ld.shared.u16 %0, [%1];" : "=h"(v) : "r"(saddr));
  return v;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
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

// Drop-in replacement for ldmatrix.sync.aligned.m8n8.x4.shared.b16
__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3, unsigned saddr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(saddr)
               : "memory");
#else
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const unsigned off = static_cast<unsigned>(lane & 3) * 4u;
  r0 = ld_smem_u32(__shfl_sync(0xffffffffu, saddr, g) + off);
  r1 = ld_smem_u32(__shfl_sync(0xffffffffu, saddr, 8 + g) + off);
  r2 = ld_smem_u32(__shfl_sync(0xffffffffu, saddr, 16 + g) + off);
  r3 = ld_smem_u32(__shfl_sync(0xffffffffu, saddr, 24 + g) + off);
#endif
}

// Drop-in replacement for ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                                  unsigned saddr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
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
    const unsigned a0 = __shfl_sync(0xffffffffu, saddr, base + row0);
    const unsigned a1 = __shfl_sync(0xffffffffu, saddr, base + row1);
    return static_cast<unsigned>(ld_smem_u16(a0 + off)) | (static_cast<unsigned>(ld_smem_u16(a1 + off)) << 16);
  };
  r0 = load_t(0);
  r1 = load_t(8);
  r2 = load_t(16);
  r3 = load_t(24);
#endif
}

// Drop-in replacement for ldmatrix.sync.aligned.m8n8.x2.shared.b16
__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned saddr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(saddr)
               : "memory");
#else
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const unsigned off = static_cast<unsigned>(lane & 3) * 4u;
  r0 = ld_smem_u32(__shfl_sync(0xffffffffu, saddr, g) + off);
  r1 = ld_smem_u32(__shfl_sync(0xffffffffu, saddr, 8 + g) + off);
#endif
}

// D += A * B, m16n8k16, FP16 inputs, FP32 accumulation (native m16n8k16 on sm_80+, native HMMA.884 on sm_70).
__device__ __forceinline__ void mma_f16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if __CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA)
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
#else
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

// D += A * B, m16n8k16, ActT inputs (FP16 on INGINE_ACT_FP16 / sm_70, BF16 on sm_80+), FP32 accumulation.
__device__ __forceinline__ void mma_act(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if INGINE_ACT_FP16 || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)
  mma_f16(d, a0, a1, a2, a3, b0, b1);
#else
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

}  // namespace ingine::kernels::mma

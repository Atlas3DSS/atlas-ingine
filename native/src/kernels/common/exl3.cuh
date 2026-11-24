// EXL3 MCG trellis codebook decode and 128-point normalized Sylvester Hadamard transform.
//
// Attribution:
//   EXL3 MCG codebook (multiplier 0xCBAC1FED, lop3 mask/xor 0x8fff8fff/0x3b603b60, hadd2),
//   16x16 tail-biting ring bitstream, and tile_perm() m16n8k16 fragment layout are derived from
//   ExLlamaV3 (https://github.com/turboderp-org/exllamav3, MIT License, Copyright (c) turboderp).
//   Escha W2 / Kintsugi-Inkling-2-Mini Map M2 128-pt Hadamard rotation convention (Apache-2.0).
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace ingine::kernels::exl3 {

// Decode two 16-bit MCG codebook states into packed FP16 pair (lo = s0, hi = s1).
__device__ __forceinline__ uint32_t decode_mcg_pair(uint32_t s0, uint32_t s1) {
  uint32_t x0 = s0 * 0xCBAC1FEDu;
  uint32_t x1 = s1 * 0xCBAC1FEDu;
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x0));
  asm("lop3.b32 %0, %0, 0x8fff8fff, 0x3b603b60, 0x6a;" : "+r"(x1));
  const __half2 h0 = *reinterpret_cast<const __half2*>(&x0);
  const __half2 h1 = *reinterpret_cast<const __half2*>(&x1);
  const __half2 res = __hadd2(__lows2half2(h0, h1), __highs2half2(h0, h1));
  return *reinterpret_cast<const uint32_t*>(&res);
}

// Decode this lane's 8 elements of a 16x16 EXL3 tile (BITS in {2, 3}) directly into 4 packed FP16 pairs:
//   h01 = (j0, j1), h23 = (j2, j3), h45 = (j4, j5), h67 = (j6, j7)
// where tile_perm() maps (lane, j) directly to hardware mma.m16n8k16 FP16 fragments:
//   - Decode A-fragment (W_hat^T, 16_n x 16_k): a0 = h01, a1 = h45, a2 = h23, a3 = h67
//   - Prefill B-fragments (W_hat, two 16_k x 8_n): b[0] = {h01, h23}, b[1] = {h45, h67}
// Splitting into two 4-state windows guarantees bit span = 16 + 3*BITS <= 25 <= 32, so each
// 4-state group fits in a single __funnelshift_l over 2 consecutive 32-bit words.
template <int BITS, class LoadWord>
__device__ __forceinline__ void dq8_mcg(LoadWord&& load_word, int lane, uint32_t& h01, uint32_t& h23,
                                        uint32_t& h45, uint32_t& h67) {
  static_assert(BITS == 2 || BITS == 3, "exl3::dq8_mcg supports 2-bit and 3-bit tiles");
  constexpr int NW = 8 * BITS;  // 16 uint32 words (64B) for 2-bit; 24 uint32 words (96B) for 3-bit
  const int b0a = (lane * 8 + 257) * BITS - 16;
  const int w0a = (b0a >> 5) % NW;
  const int w1a = (w0a + 1) % NW;

  const int b0b = (lane * 8 + 261) * BITS - 16;
  const int w0b = (b0b >> 5) % NW;
  const int w1b = (w0b + 1) % NW;

  const uint32_t a0 = load_word(w0a);
  const uint32_t a1 = load_word(w1a);
  const uint32_t b0 = (w0b == w0a) ? a0 : ((w0b == w1a) ? a1 : load_word(w0b));
  const uint32_t b1 = (w1b == w1a) ? a1 : ((w1b == w0a) ? a0 : load_word(w1b));

  const uint32_t wa = __funnelshift_l(a1, a0, b0a & 31);
  const uint32_t s0 = wa >> 16;
  const uint32_t s1 = (wa >> (16 - BITS)) & 0xffffu;
  const uint32_t s2 = (wa >> (16 - 2 * BITS)) & 0xffffu;
  const uint32_t s3 = (wa >> (16 - 3 * BITS)) & 0xffffu;

  const uint32_t wb = __funnelshift_l(b1, b0, b0b & 31);
  const uint32_t s4 = wb >> 16;
  const uint32_t s5 = (wb >> (16 - BITS)) & 0xffffu;
  const uint32_t s6 = (wb >> (16 - 2 * BITS)) & 0xffffu;
  const uint32_t s7 = (wb >> (16 - 3 * BITS)) & 0xffffu;

  h01 = decode_mcg_pair(s0, s1);
  h23 = decode_mcg_pair(s2, s3);
  h45 = decode_mcg_pair(s4, s5);
  h67 = decode_mcg_pair(s6, s7);
}

// 128-point normalized Sylvester Fast Walsh-Hadamard Transform across a 32-lane warp.
// Lane `lane` holds elements [4*lane + 0 .. 4*lane + 3] in v[0..3].
__device__ __forceinline__ void fwht_128_warp(float (&v)[4], int lane) {
  {
    const float a0 = v[0], b0 = v[1], a1 = v[2], b1 = v[3];
    v[0] = a0 + b0;
    v[1] = a0 - b0;
    v[2] = a1 + b1;
    v[3] = a1 - b1;
  }
  {
    const float a0 = v[0], b0 = v[2], a1 = v[1], b1 = v[3];
    v[0] = a0 + b0;
    v[2] = a0 - b0;
    v[1] = a1 + b1;
    v[3] = a1 - b1;
  }
#pragma unroll
  for (int mask = 1; mask <= 16; mask <<= 1) {
    const bool neg = (lane & mask) != 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float o = __shfl_xor_sync(0xffffffffu, v[i], mask);
      v[i] = neg ? (o - v[i]) : (v[i] + o);
    }
  }
  constexpr float kInvSqrt128 = 0.08838834764831844296f;
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] *= kInvSqrt128;
}

}  // namespace ingine::kernels::exl3

// Compile-time 16-bit activation/weight storage type and conversion helpers.
//
// SM80+ (Ampere/Ada/Hopper/Blackwell): ActT = __nv_bfloat16, CUDA_R_16BF, DType::BF16 (bit-identical to upstream).
// SM70 (Volta V100, built with -DINGINE_ACT_FP16=1): ActT = __half, CUDA_R_16F, DType::FP16, because V100 has no
// BF16 tensor cores and no CUDA_R_16BF support in cuBLASLt. Reductions, RMSNorm, softmax, router, and MMA
// accumulators remain FP32. Finite FP16 conversions saturate to +-65504 to prevent Inf/NaN cascades.
#pragma once

#include "core/dtype.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <library_types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#ifndef INGINE_ACT_FP16
#define INGINE_ACT_FP16 0
#endif

namespace ingine {

#if INGINE_ACT_FP16
using ActT = __half;
using Act2T = __half2;
inline constexpr DType kActDType = DType::FP16;
inline constexpr cudaDataType_t kCublasActType = CUDA_R_16F;
inline constexpr bool kActIsFP16 = true;
#else
using ActT = __nv_bfloat16;
using Act2T = __nv_bfloat162;
inline constexpr DType kActDType = DType::BF16;
inline constexpr cudaDataType_t kCublasActType = CUDA_R_16BF;
inline constexpr bool kActIsFP16 = false;
#endif

namespace act {

inline constexpr float kFp16Max = 65504.0f;

// --------------------------------------------------------------------------------------------- host helpers

inline float host_bf16_bits_to_float(std::uint16_t b) {
  const std::uint32_t w = std::uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

inline std::uint16_t host_float_to_bf16_bits(float f) {
  std::uint32_t w;
  std::memcpy(&w, &f, 4);
  if ((w & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((w >> 16) | 0x40);
  return std::uint16_t((w + 0x7fffu + ((w >> 16) & 1u)) >> 16);
}

// IEEE-754 binary16 -> binary32 on host (exact).
inline float host_fp16_bits_to_float(std::uint16_t h) {
  const std::uint32_t s = std::uint32_t(h & 0x8000u) << 16;
  const std::uint32_t e = (h >> 10) & 0x1fu;
  const std::uint32_t m = h & 0x03ffu;
  std::uint32_t w;
  if (e == 0) {
    if (m == 0) {
      w = s;
    } else {
      // Subnormal: normalize
      std::uint32_t shift = 0;
      std::uint32_t mm = m;
      while ((mm & 0x0400u) == 0) {
        mm <<= 1;
        ++shift;
      }
      w = s | ((127u - 14u - shift) << 23) | ((mm & 0x03ffu) << 13);
    }
  } else if (e == 31) {
    w = s | 0x7f800000u | (m << 13);
  } else {
    w = s | ((e + 112u) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

// IEEE-754 binary32 -> binary16 on host, round-to-nearest-even, finite values clamped to +-65504.
inline std::uint16_t host_float_to_fp16_bits(float f) {
  if (std::isfinite(f)) {
    f = std::clamp(f, -kFp16Max, kFp16Max);
  }
  std::uint32_t w;
  std::memcpy(&w, &f, 4);
  const std::uint16_t sign = static_cast<std::uint16_t>((w >> 16) & 0x8000u);
  const std::uint32_t abs_w = w & 0x7fffffffu;
  if (abs_w > 0x7f800000u) return static_cast<std::uint16_t>(sign | 0x7e00u);  // NaN
  if (abs_w == 0x7f800000u) return static_cast<std::uint16_t>(sign | 0x7c00u); // Inf
  if (abs_w >= 0x477fe000u) return static_cast<std::uint16_t>(sign | 0x7bffu); // saturate max finite 65504
  if (abs_w < 0x38800000u) {
    // Subnormal or underflow to zero (23 FP32 mantissa bits + 1 implicit bit, bias diff 127 - 14 = 113, + 13 = 126)
    const int shift = 126 - int(abs_w >> 23);
    if (shift > 24) return sign;
    const std::uint32_t mant = (abs_w & 0x007fffffu) | 0x00800000u;
    const std::uint32_t half_bit = 1u << (shift - 1);
    const std::uint32_t rem = mant & ((1u << shift) - 1u);
    std::uint32_t rounded = mant >> shift;
    if (rem > half_bit || (rem == half_bit && (rounded & 1u))) ++rounded;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  // Normal FP16
  const std::uint32_t exp = (abs_w >> 23) - 112u;
  const std::uint32_t mant = abs_w & 0x007fffffu;
  const std::uint32_t rem = mant & 0x1fffu;
  std::uint32_t h = (exp << 10) | (mant >> 13);
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return static_cast<std::uint16_t>(sign | h);
}

inline std::uint16_t host_float_to_bits(float f) {
#if INGINE_ACT_FP16
  return host_float_to_fp16_bits(f);
#else
  return host_float_to_bf16_bits(f);
#endif
}

inline float host_bits_to_float(std::uint16_t b) {
#if INGINE_ACT_FP16
  return host_fp16_bits_to_float(b);
#else
  return host_bf16_bits_to_float(b);
#endif
}

inline ActT host_from_float(float f) {
  const std::uint16_t b = host_float_to_bits(f);
  ActT v;
  std::memcpy(&v, &b, 2);
  return v;
}

inline float host_to_float(ActT v) {
  std::uint16_t b;
  std::memcpy(&b, &v, 2);
  return host_bits_to_float(b);
}

inline float host_round(float f) { return host_bits_to_float(host_float_to_bits(f)); }

// ------------------------------------------------------------------------------------------- device helpers

#if defined(__CUDACC__)

__device__ __forceinline__ float clamp_fp16_finite(float x) {
  return isfinite(x) ? fminf(fmaxf(x, -kFp16Max), kFp16Max) : x;
}

__device__ __forceinline__ ActT from_float(float x) {
#if INGINE_ACT_FP16
  return __float2half_rn(clamp_fp16_finite(x));
#else
  return __float2bfloat16_rn(x);
#endif
}

__device__ __forceinline__ float to_float(ActT x) {
#if INGINE_ACT_FP16
  return __half2float(x);
#else
  return __bfloat162float(x);
#endif
}

__device__ __forceinline__ float round_act(float x) {
#if INGINE_ACT_FP16
  return __half2float(__float2half_rn(clamp_fp16_finite(x)));
#else
  return __bfloat162float(__float2bfloat16_rn(x));
#endif
}

__device__ __forceinline__ std::uint16_t float_to_bits(float x) {
#if INGINE_ACT_FP16
  return __half_as_ushort(__float2half_rn(clamp_fp16_finite(x)));
#else
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
#endif
}

__device__ __forceinline__ float bits_to_float(std::uint16_t u) {
#if INGINE_ACT_FP16
  return __half2float(__ushort_as_half(u));
#else
  return __uint_as_float(std::uint32_t(u) << 16);
#endif
}

__device__ __forceinline__ std::uint32_t pack2(float lo, float hi) {
#if INGINE_ACT_FP16
  const __half2 h = __floats2half2_rn(clamp_fp16_finite(lo), clamp_fp16_finite(hi));
  return *reinterpret_cast<const std::uint32_t*>(&h);
#else
  const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const std::uint32_t*>(&b);
#endif
}

__device__ __forceinline__ float2 unpack2(std::uint32_t u) {
#if INGINE_ACT_FP16
  return __half22float2(*reinterpret_cast<const __half2*>(&u));
#else
  return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
#endif
}

__device__ __forceinline__ float lo_f(std::uint32_t u) {
#if INGINE_ACT_FP16
  return __half2float(__ushort_as_half(static_cast<unsigned short>(u & 0xffffu)));
#else
  return __uint_as_float(u << 16);
#endif
}

__device__ __forceinline__ float hi_f(std::uint32_t u) {
#if INGINE_ACT_FP16
  return __half2float(__ushort_as_half(static_cast<unsigned short>(u >> 16)));
#else
  return __uint_as_float(u & 0xffff0000u);
#endif
}

__device__ __forceinline__ void unpack8(const uint4& v, float* f) {
  const float2 p0 = unpack2(v.x), p1 = unpack2(v.y), p2 = unpack2(v.z), p3 = unpack2(v.w);
  f[0] = p0.x; f[1] = p0.y;
  f[2] = p1.x; f[3] = p1.y;
  f[4] = p2.x; f[5] = p2.y;
  f[6] = p3.x; f[7] = p3.y;
}

__device__ __forceinline__ uint4 pack8(const float* f) {
  return make_uint4(pack2(f[0], f[1]), pack2(f[2], f[3]), pack2(f[4], f[5]), pack2(f[6], f[7]));
}

__device__ __forceinline__ std::uint16_t e4m3_to_f16_bits_portable(std::uint32_t b) {
  const std::uint32_t s = (b & 0x80u) << 8;
  const std::uint32_t a = b & 0x7fu;
  const std::uint32_t norm = (a + 64u) << 7;
  const std::uint32_t sub = a ? (0x1800u + (((0xBA986400u >> (a * 4u)) & 0xfu) << 8)) : 0u;
  return static_cast<std::uint16_t>(s | (a >= 8u ? norm : sub));
}

// 2 E4M3 codes (low 16 bits of u) -> packed FP16 pair (exact on all SM70+).
__device__ __forceinline__ std::uint32_t e4m3x2_to_half2_bits(std::uint32_t u) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  std::uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<unsigned short>(u)));
  return r;
#else
  return static_cast<std::uint32_t>(e4m3_to_f16_bits_portable(u & 0xffu)) |
         (static_cast<std::uint32_t>(e4m3_to_f16_bits_portable((u >> 8) & 0xffu)) << 16);
#endif
}

// 2 E4M3 codes (low 16 bits of u) -> packed Act2T bits (exact in FP16 on SM70, exact in BF16 on SM80+).
__device__ __forceinline__ std::uint32_t e4m3x2_to_act2_bits(std::uint32_t u) {
#if INGINE_ACT_FP16
  return e4m3x2_to_half2_bits(u);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  std::uint32_t r;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<unsigned short>(u)));
  return r;
#else
  const std::uint32_t h2 = e4m3x2_to_half2_bits(u);
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
  const __nv_bfloat162 b = __float22bfloat162_rn(f);
  return *reinterpret_cast<const std::uint32_t*>(&b);
#endif
}

#endif  // __CUDACC__

}  // namespace act
}  // namespace ingine

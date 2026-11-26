// Portable device/host bit manipulation and integer dot-product primitives across CUDA, AMD ROCm/HIP,
// Intel SYCL, and standard C++20 hosts.
//
// CRITICAL HIP SAFETY INVARIANT: Never define __CUDA_ARCH__ under HIP. Inline PTX is strictly gated by
// defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__). HIP paths follow Strata / Project Maya
// hip_compat (MIT) without private-memory spills or missing partial-mask semantics.
#pragma once

#include "core/act.h"

#include <cmath>
#include <cstdint>

namespace ingine::kernels::arch {

#if (defined(__CUDACC__) || defined(__HIPCC__))
#define INGINE_ARCH_HD __host__ __device__ __forceinline__
#else
#define INGINE_ARCH_HD inline
#endif

// 3-input bitwise LUT matching PTX `lop3.b32 d, a, b, c, IMM` bit-for-bit across all backends.
template <std::uint32_t LUT>
INGINE_ARCH_HD std::uint32_t lop3_lut(std::uint32_t a, std::uint32_t b, std::uint32_t c) noexcept {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  std::uint32_t d = 0;
  asm("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(d) : "r"(a), "r"(b), "r"(c), "n"(LUT));
  return d;
#else
  if constexpr (LUT == 0x6au) {
    // Specialized fast path for EXL3 MCG decoding: (a & b) ^ c.
    return (a & b) ^ c;
  } else {
    std::uint32_t d = 0;
    if constexpr ((LUT & 0x01u) != 0u) d |= (~a & ~b & ~c);
    if constexpr ((LUT & 0x02u) != 0u) d |= (~a & ~b & c);
    if constexpr ((LUT & 0x04u) != 0u) d |= (~a & b & ~c);
    if constexpr ((LUT & 0x08u) != 0u) d |= (~a & b & c);
    if constexpr ((LUT & 0x10u) != 0u) d |= (a & ~b & ~c);
    if constexpr ((LUT & 0x20u) != 0u) d |= (a & ~b & c);
    if constexpr ((LUT & 0x40u) != 0u) d |= (a & b & ~c);
    if constexpr ((LUT & 0x80u) != 0u) d |= (a & b & c);
    return d;
  }
#endif
}

// EXL3 MCG bit-permutation constant transform: lop3<0x6a>(x, 0x8fff8fff, 0x3b603b60).
INGINE_ARCH_HD std::uint32_t exl3_mcg_mask_xor(std::uint32_t x) noexcept {
  return lop3_lut<0x6au>(x, 0x8fff8fffu, 0x3b603b60u);
}

// Left funnel shift matching CUDA `__funnelshift_l(lo, hi, shift)`: upper 32 bits of ({hi, lo} << (shift & 31)).
INGINE_ARCH_HD std::uint32_t funnelshift_l(std::uint32_t lo, std::uint32_t hi, int shift) noexcept {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  return __funnelshift_l(lo, hi, shift);
#else
  const unsigned s = static_cast<unsigned>(shift) & 31u;
  return (s == 0u) ? hi : ((hi << s) | (lo >> (32u - s)));
#endif
}

// Signed 4x int8 dot-product accumulated modulo 2^32, matching CUDA `__dp4a(a, b, c)`.
INGINE_ARCH_HD std::int32_t dp4a_s8(std::int32_t a, std::int32_t b, std::int32_t c) noexcept {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  return __dp4a(a, b, c);
#elif defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1150__) || \
     defined(__gfx1151__) || defined(__gfx1200__) || defined(__gfx1201__)) && \
    __has_builtin(__builtin_amdgcn_sudot4)
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
     defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || \
     defined(__gfx942__) || defined(__gfx950__)) && \
    __has_builtin(__builtin_amdgcn_sdot4)
  return __builtin_amdgcn_sdot4(a, b, c, false);
#else
  const auto ua = static_cast<std::uint32_t>(a);
  const auto ub = static_cast<std::uint32_t>(b);
  std::uint32_t sum = static_cast<std::uint32_t>(c);
  for (int lane = 0; lane < 4; ++lane) {
    const unsigned ba = (ua >> (lane * 8)) & 0xffu;
    const unsigned bb = (ub >> (lane * 8)) & 0xffu;
    const int sa = ba < 0x80u ? static_cast<int>(ba) : static_cast<int>(ba) - 0x100;
    const int sb = bb < 0x80u ? static_cast<int>(bb) : static_cast<int>(bb) - 0x100;
    sum += static_cast<std::uint32_t>(sa * sb);
  }
  return static_cast<std::int32_t>(sum);
#endif
}

// Default-mode 4-byte permutation over {y:x} (x = bytes 0..3, y = bytes 4..7), matching CUDA `__byte_perm`.
INGINE_ARCH_HD std::uint32_t byte_perm(std::uint32_t x, std::uint32_t y, std::uint32_t sel) noexcept {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  return __byte_perm(x, y, sel);
#elif defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_perm)
  const std::uint32_t mapped = (sel & 0x7u) | ((sel & 0x70u) << 4) | ((sel & 0x700u) << 8) | ((sel & 0x7000u) << 12);
  return __builtin_amdgcn_perm(y, x, mapped);
#else
  std::uint32_t out = 0;
  for (int i = 0; i < 4; ++i) {
    const unsigned idx = (sel >> (i * 4)) & 0x7u;
    const std::uint32_t src = (idx < 4u) ? (x >> (idx * 8u)) : (y >> ((idx - 4u) * 8u));
    out |= (src & 0xffu) << (i * 8u);
  }
  return out;
#endif
}

// SWAR 4-lane unsigned byte wrap subtraction matching CUDA `__vsub4` (Hacker's Delight 2-18).
INGINE_ARCH_HD std::uint32_t vsub4_u8(std::uint32_t ua, std::uint32_t ub) noexcept {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  return static_cast<std::uint32_t>(__vsub4(static_cast<int>(ua), static_cast<int>(ub)));
#else
  constexpr std::uint32_t kHigh = 0x80808080u;
  return ((ua | kHigh) - (ub & ~kHigh)) ^ ((ua ^ ~ub) & kHigh);
#endif
}

// Canonical Q4_0 SoA single-nibble decode rule matching CUDA/CPU bit-for-bit (including negative scale * nibble=8 -> +0.0f).
// Fully host- and device-callable without host-only calls.
INGINE_ARCH_HD float decode_q4_0_nibble_canonical(std::uint8_t nibble, std::uint16_t fp16_scale_bits) noexcept {
  if ((fp16_scale_bits & 0x7c00u) == 0x7c00u) return 0.0f;
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && INGINE_BACKEND_CUDA
  const float d = __half2float(__ushort_as_half(fp16_scale_bits));
  return fmaf(static_cast<float>(nibble & 0x0fu), d, -8.0f * d);
#elif defined(__HIP_DEVICE_COMPILE__) && INGINE_BACKEND_HIP
  const float d = __half2float(*reinterpret_cast<const __half*>(&fp16_scale_bits));
  return fmaf(static_cast<float>(nibble & 0x0fu), d, -8.0f * d);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const float d = ingine::act::fp16_bits_to_float_portable(fp16_scale_bits);
  return fmaf(static_cast<float>(nibble & 0x0fu), d, -8.0f * d);
#else
  const float d = ingine::act::fp16_bits_to_float_portable(fp16_scale_bits);
  return std::fmaf(static_cast<float>(nibble & 0x0fu), d, -8.0f * d);
#endif
}

#undef INGINE_ARCH_HD

}  // namespace ingine::kernels::arch
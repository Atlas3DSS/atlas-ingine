// Pinned GGML Q5_K super-block format (176 bytes / 256 elements = 5.50 bits/weight) and pure-CPU reference oracle.
//
// Byte layout matches ggml's `block_q5_K` exactly:
//   [0..1]     uint16_t d          (IEEE-754 binary16 super-block scale for quantized sub-block scales)
//   [2..3]     uint16_t dmin       (IEEE-754 binary16 super-block scale for quantized sub-block mins)
//   [4..15]    uint8_t  scales[12] (8 x 6-bit sub-block scales + 8 x 6-bit sub-block mins via get_scale_min_k4)
//   [16..47]   uint8_t  qh[32]     (high bit 4 of each of the 256 5-bit quants)
//   [48..175]  uint8_t  qs[128]    (low 4 bits of each of the 256 5-bit quants; 4 chunks of 64 elements)
//
// Exact decode formula for chunk c in 0..3, lane l in 0..31:
//   elem (64*c + l)      in sub-block 2*c:   q = (qs[32*c + l] & 0x0F) + ((qh[l] & (1 << (2*c)))     ? 16 : 0)
//   elem (64*c + 32 + l) in sub-block 2*c+1: q = (qs[32*c + l] >> 4)   + ((qh[l] & (1 << (2*c + 1))) ? 16 : 0)
//   value = f16(d) * sc[j] * q - f16(dmin) * m[j]
//
// CUDA-free header usable from pure-CPU unit tests, CUDA kernels (gemv.cu), and model/weight code.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace ingine::q5k {

inline constexpr int kBlockElems = 256;
inline constexpr int kSubBlocks = 8;
inline constexpr int kSubBlockElems = 32;
inline constexpr int kScaleBytes = 12;
inline constexpr int kBlockBytes = 176;

struct BlockQ5K {
  std::uint16_t d = 0;
  std::uint16_t dmin = 0;
  std::uint8_t scales[kScaleBytes]{};
  std::uint8_t qh[32]{};
  std::uint8_t qs[128]{};
};
static_assert(sizeof(BlockQ5K) == kBlockBytes, "BlockQ5K must match GGML 176-byte layout exactly");

inline constexpr std::size_t row_bytes(int cols) noexcept {
  return std::size_t(cols / kBlockElems) * kBlockBytes;
}

// Host IEEE-754 binary16 <-> float32 conversion (round-to-nearest-even, subnormal-safe, zero CUDA dependency).
inline std::uint16_t f32_to_f16_bits(float v) noexcept {
  std::uint32_t u = 0;
  std::memcpy(&u, &v, 4);
  const std::uint32_t sign = (u >> 16) & 0x8000u;
  const std::uint32_t abs_u = u & 0x7fffffffu;
  if (abs_u >= 0x7f800000u) {
    // Inf or NaN
    if (abs_u > 0x7f800000u) return static_cast<std::uint16_t>(sign | 0x7e00u | ((abs_u >> 13) & 0x03ffu));
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  if (abs_u > 0x477fefffu) {
    // Clamp overflow to max finite FP16 (65504.0)
    return static_cast<std::uint16_t>(sign | 0x7bffu);
  }
  const int exp = int((abs_u >> 23) & 0xffu) - 127;
  const std::uint32_t mant = abs_u & 0x007fffffu;
  if (exp < -14) {
    // Subnormal or underflow to zero
    const int shift = -14 - exp;
    if (shift > 24) return static_cast<std::uint16_t>(sign);
    const std::uint32_t sig = 0x00800000u | mant;
    const int rshift = shift + 13;
    std::uint32_t half_m = sig >> rshift;
    const std::uint32_t rem = sig & ((1u << rshift) - 1u);
    const std::uint32_t mid = 1u << (rshift - 1);
    if (rem > mid || (rem == mid && (half_m & 1u))) ++half_m;
    return static_cast<std::uint16_t>(sign | half_m);
  }
  std::uint32_t h_exp = std::uint32_t(exp + 15);
  std::uint32_t h_mant = mant >> 13;
  const std::uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h_mant & 1u))) {
    ++h_mant;
    if (h_mant == 0x0400u) {
      h_mant = 0;
      ++h_exp;
      if (h_exp >= 31u) return static_cast<std::uint16_t>(sign | 0x7bffu);
    }
  }
  return static_cast<std::uint16_t>(sign | (h_exp << 10) | h_mant);
}

inline float f16_bits_to_f32(std::uint16_t h) noexcept {
  const std::uint32_t sign = (std::uint32_t(h) & 0x8000u) << 16;
  const std::uint32_t exp = (std::uint32_t(h) >> 10) & 0x1fu;
  const std::uint32_t mant = std::uint32_t(h) & 0x03ffu;
  std::uint32_t u = 0;
  if (exp == 0u) {
    if (mant == 0u) {
      u = sign;
    } else {
      // Subnormal: mant * 2^-24
      float f = std::ldexp(float(mant), -24);
      if (sign) f = -f;
      return f;
    }
  } else if (exp == 31u) {
    u = sign | 0x7f800000u | (mant << 13);
  } else {
    u = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float out = 0.f;
  std::memcpy(&out, &u, 4);
  return out;
}

// Exact GGML get_scale_min_k4 unpacking for sub-block j in 0..7.
inline void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& sc, std::uint8_t& m) noexcept {
  if (j < 4) {
    sc = q[j] & 63u;
    m = q[j + 4] & 63u;
  } else {
    sc = static_cast<std::uint8_t>((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
    m = static_cast<std::uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

// Exact inverse of get_scale_min_k4 for 8 6-bit scales sc[0..7] and 8 6-bit mins m[0..7].
inline void pack_scale_min_k4(const std::uint8_t sc[kSubBlocks], const std::uint8_t m[kSubBlocks],
                              std::uint8_t out[kScaleBytes]) noexcept {
  for (int j = 0; j < 4; ++j) {
    out[j] = static_cast<std::uint8_t>((sc[j] & 63u) | ((sc[j + 4] >> 4) << 6));
    out[j + 4] = static_cast<std::uint8_t>((m[j] & 63u) | ((m[j + 4] >> 4) << 6));
    out[j + 8] = static_cast<std::uint8_t>((sc[j + 4] & 0x0Fu) | ((m[j + 4] & 0x0Fu) << 4));
  }
}

// Quantize 256 FP32 weights into one exact GGML BlockQ5K (matches GPU quantize_q5_k_rows_kernel).
// Two-pass no-clip sub-block quantization:
//   1. Quantize dmin = f16(max_j(-mn_j) / 63), then round each sub-block m_q[j] = ceil((-mn_j) / dmin)
//      so -dmin * m_q[j] <= mn_j (no lower-tail clipping).
//   2. Compute required per-sub-block scale need_sc[j] = (mx_j + dmin * m_q[j]) / 31, quantize
//      d = f16(max_j(need_sc[j]) / 63), and round sc_q[j] = ceil(need_sc[j] / d) so
//      31 * d * sc_q[j] - dmin * m_q[j] >= mx_j (no upper-tail clipping).
// This eliminates sub-block scale/min clipping when an outlier inflates another sub-block in the same
// 256-element super-block, enforcing <= 2.5% per-32-value source-range reconstruction error.
inline void quantize_block_cpu(const float* src, BlockQ5K& dst) noexcept {
  float sb_mn[kSubBlocks];
  float sb_mx[kSubBlocks];
  float max_min = 0.f;
  for (int j = 0; j < kSubBlocks; ++j) {
    float mn = 0.f;
    float mx = 0.f;
    const float* p = src + j * kSubBlockElems;
    for (int i = 0; i < kSubBlockElems; ++i) {
      const float v = p[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    sb_mn[j] = mn;
    sb_mx[j] = mx;
    const float neg_mn = -mn;
    if (neg_mn > max_min) max_min = neg_mn;
  }

  const float dmin_f = max_min / 63.0f;
  dst.dmin = f32_to_f16_bits(dmin_f);
  const float dmin_val = f16_bits_to_f32(dst.dmin);

  std::uint8_t m_q[kSubBlocks];
  float need_sc[kSubBlocks];
  float max_scale = 0.f;
  for (int j = 0; j < kSubBlocks; ++j) {
    const int mq = (dmin_val > 0.f) ? std::clamp(int(std::ceil((-sb_mn[j]) / dmin_val)), 0, 63) : 0;
    m_q[j] = static_cast<std::uint8_t>(mq);
    const float eff_m = dmin_val * float(mq);
    const float sc = std::max(0.f, (sb_mx[j] + eff_m) / 31.0f);
    need_sc[j] = sc;
    if (sc > max_scale) max_scale = sc;
  }

  const float d_f = max_scale / 63.0f;
  dst.d = f32_to_f16_bits(d_f);
  const float d_val = f16_bits_to_f32(dst.d);

  std::uint8_t sc_q[kSubBlocks];
  for (int j = 0; j < kSubBlocks; ++j) {
    const int sq = (d_val > 0.f) ? std::clamp(int(std::ceil(need_sc[j] / d_val)), 0, 63) : 0;
    sc_q[j] = static_cast<std::uint8_t>(sq);
  }
  pack_scale_min_k4(sc_q, m_q, dst.scales);

  for (int l = 0; l < 32; ++l) dst.qh[l] = 0;
  for (int c = 0; c < 4; ++c) {
    const int j0 = 2 * c;
    const int j1 = 2 * c + 1;
    const float eff_s0 = d_val * float(sc_q[j0]);
    const float eff_m0 = dmin_val * float(m_q[j0]);
    const float eff_s1 = d_val * float(sc_q[j1]);
    const float eff_m1 = dmin_val * float(m_q[j1]);
    const float inv_s0 = (eff_s0 > 0.f) ? (1.0f / eff_s0) : 0.f;
    const float inv_s1 = (eff_s1 > 0.f) ? (1.0f / eff_s1) : 0.f;
    const std::uint8_t u1 = static_cast<std::uint8_t>(1u << (2 * c));
    const std::uint8_t u2 = static_cast<std::uint8_t>(1u << (2 * c + 1));
    for (int l = 0; l < 32; ++l) {
      const float v0 = src[64 * c + l];
      const float v1 = src[64 * c + 32 + l];
      const int q0 = (inv_s0 > 0.f) ? std::clamp(int(std::nearbyint((v0 + eff_m0) * inv_s0)), 0, 31) : 0;
      const int q1 = (inv_s1 > 0.f) ? std::clamp(int(std::nearbyint((v1 + eff_m1) * inv_s1)), 0, 31) : 0;
      dst.qs[32 * c + l] = static_cast<std::uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
      if (q0 & 0x10) dst.qh[l] |= u1;
      if (q1 & 0x10) dst.qh[l] |= u2;
    }
  }
}

// Exact GGML BlockQ5K dequantization into 256 FP32 values.
inline void dequant_block_cpu(const BlockQ5K& src, float* dst) noexcept {
  const float d = f16_bits_to_f32(src.d);
  const float dmin = f16_bits_to_f32(src.dmin);
  for (int c = 0; c < 4; ++c) {
    std::uint8_t sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
    get_scale_min_k4(2 * c, src.scales, sc0, m0);
    get_scale_min_k4(2 * c + 1, src.scales, sc1, m1);
    const float d0 = d * float(sc0);
    const float dm0 = dmin * float(m0);
    const float d1 = d * float(sc1);
    const float dm1 = dmin * float(m1);
    const std::uint8_t u1 = static_cast<std::uint8_t>(1u << (2 * c));
    const std::uint8_t u2 = static_cast<std::uint8_t>(1u << (2 * c + 1));
    for (int l = 0; l < 32; ++l) {
      const std::uint8_t ql = src.qs[32 * c + l];
      const std::uint8_t qh = src.qh[l];
      const int q0 = int(ql & 0x0Fu) + ((qh & u1) ? 16 : 0);
      const int q1 = int(ql >> 4) + ((qh & u2) ? 16 : 0);
      dst[64 * c + l] = d0 * float(q0) - dm0;
      dst[64 * c + 32 + l] = d1 * float(q1) - dm1;
    }
  }
}

inline void quantize_rows_cpu(const float* src, int rows, int cols, void* dst_bytes) {
  if (rows < 0 || cols <= 0 || (cols % kBlockElems) != 0) {
    throw std::invalid_argument("q5k::quantize_rows_cpu: cols must be a positive multiple of 256");
  }
  const int nb = cols / kBlockElems;
  auto* blocks = static_cast<BlockQ5K*>(dst_bytes);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < nb; ++b) {
      quantize_block_cpu(src + std::size_t(r) * cols + std::size_t(b) * kBlockElems, blocks[std::size_t(r) * nb + b]);
    }
  }
}

inline void dequant_rows_cpu(const void* src_bytes, int rows, int cols, float* dst) {
  if (rows < 0 || cols <= 0 || (cols % kBlockElems) != 0) {
    throw std::invalid_argument("q5k::dequant_rows_cpu: cols must be a positive multiple of 256");
  }
  const int nb = cols / kBlockElems;
  const auto* blocks = static_cast<const BlockQ5K*>(src_bytes);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < nb; ++b) {
      dequant_block_cpu(blocks[std::size_t(r) * nb + b], dst + std::size_t(r) * cols + std::size_t(b) * kBlockElems);
    }
  }
}

// Independent FP64 reference dot product over packed GGML BlockQ5K rows:
//   out[m, r] = sum_k double(dequant(w_q5k[r, k])) * double(x[m, k])
inline void gemv_fp64_ref(const void* w_q5k, const float* x, int M, int rows, int cols, double* out) {
  if (M < 1 || rows < 0 || cols <= 0 || (cols % kBlockElems) != 0) {
    throw std::invalid_argument("q5k::gemv_fp64_ref: bad shape");
  }
  const int nb = cols / kBlockElems;
  const auto* blocks = static_cast<const BlockQ5K*>(w_q5k);
  float w_blk[kBlockElems];
  for (int m = 0; m < M; ++m) {
    const float* xm = x + std::size_t(m) * cols;
    for (int r = 0; r < rows; ++r) {
      double acc = 0.0;
      for (int b = 0; b < nb; ++b) {
        dequant_block_cpu(blocks[std::size_t(r) * nb + b], w_blk);
        const int base = b * kBlockElems;
        for (int i = 0; i < kBlockElems; ++i) {
          acc += double(w_blk[i]) * double(xm[base + i]);
        }
      }
      out[std::size_t(m) * rows + r] = acc;
    }
  }
}

}  // namespace ingine::q5k

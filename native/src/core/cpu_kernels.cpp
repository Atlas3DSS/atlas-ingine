#include "core/cpu_kernels.h"

#include "core/act.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace ingine::ops::cpu {
namespace {

constexpr float kInvSqrt128 = 0.08838834764831844296f;

inline void prefetch_read(const void* p) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch(p, 0, 1);
#else
  (void)p;
#endif
}

inline float round_cpu_act(float x, ActRounding rounding) noexcept {
  switch (rounding) {
    case ActRounding::None: return x;
    case ActRounding::Bf16Rn: return act::host_bf16_bits_to_float(act::host_float_to_bf16_bits(x));
    case ActRounding::Fp16RnSat: return act::host_fp16_bits_to_float(act::host_float_to_fp16_bits(x));
  }
  return x;
}

inline float fp16_to_f32_finite_scale(std::uint16_t bits) noexcept {
  const float raw = act::host_fp16_bits_to_float(bits);
  return std::isfinite(raw) ? raw : 0.0f;
}

inline float mcg_decode_state_scalar(std::uint32_t st16, bool fp16_pair_round) noexcept {
  const std::uint32_t p = (st16 & 0xffffu) * 0xCBAC1FEDu;
  const std::uint32_t u = (p & 0x8fff8fffu) ^ 0x3b603b60u;
  const float lo = act::host_fp16_bits_to_float(static_cast<std::uint16_t>(u & 0xffffu));
  const float hi = act::host_fp16_bits_to_float(static_cast<std::uint16_t>(u >> 16));
  const float sum = lo + hi;
  return fp16_pair_round ? act::host_fp16_bits_to_float(act::host_float_to_fp16_bits(sum)) : sum;
}

void fwht_128_scalar(float* v) noexcept {
  for (int h = 1; h < 128; h <<= 1) {
    for (int i = 0; i < 128; i += 2 * h) {
      for (int j = i; j < i + h; ++j) {
        const float a = v[j];
        const float b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
    }
  }
  for (int i = 0; i < 128; ++i) v[i] *= kInvSqrt128;
}

void decode_q4_0_row_scalar_impl(const std::uint8_t* row_bytes, int hidden_size, float* out_f32) noexcept {
  const int nib_bytes = hidden_size / 2;
  const int blocks = hidden_size / 32;
  const std::uint8_t* scales = row_bytes + nib_bytes;
  for (int b = 0; b < blocks; ++b) {
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, scales + std::size_t(b) * 2, 2);
    const float d = fp16_to_f32_finite_scale(d_bits);
    const float m8d = -8.0f * d;
    const std::uint8_t* q = row_bytes + std::size_t(b) * 16;
    float* dst = out_f32 + std::size_t(b) * 32;
    for (int i = 0; i < 16; ++i) {
      const std::uint8_t byte = q[i];
      dst[2 * i + 0] = std::fmaf(static_cast<float>(byte & 0x0fu), d, m8d);
      dst[2 * i + 1] = std::fmaf(static_cast<float>(byte >> 4), d, m8d);
    }
  }
}

void gemv_q4_0_f32_scalar_range(const std::uint8_t* w_q4, const float* x, float* y, int r0, int r1, int K) noexcept {
  const int nib_bytes = K / 2;
  const int blocks = K / 32;
  const std::size_t row_bytes = static_cast<std::size_t>(nib_bytes) + static_cast<std::size_t>(blocks) * 2;
  for (int r = r0; r < r1; ++r) {
    const std::uint8_t* row = w_q4 + static_cast<std::size_t>(r) * row_bytes;
    if (r + 2 < r1) {
      prefetch_read(w_q4 + static_cast<std::size_t>(r + 2) * row_bytes);
    }
    const std::uint8_t* scales = row + nib_bytes;
    float acc = 0.0f;
    for (int b = 0; b < blocks; ++b) {
      std::uint16_t d_bits = 0;
      std::memcpy(&d_bits, scales + std::size_t(b) * 2, 2);
      const float d = fp16_to_f32_finite_scale(d_bits);
      const float m8d = -8.0f * d;
      const std::uint8_t* q = row + std::size_t(b) * 16;
      const float* xp = x + std::size_t(b) * 32;
      float bacc = 0.0f;
      for (int i = 0; i < 16; ++i) {
        const std::uint8_t byte = q[i];
        const float w0 = std::fmaf(static_cast<float>(byte & 0x0fu), d, m8d);
        const float w1 = std::fmaf(static_cast<float>(byte >> 4), d, m8d);
        bacc += w0 * xp[2 * i + 0] + w1 * xp[2 * i + 1];
      }
      acc += bacc;
    }
    y[r] = acc;
  }
}

void gemv_q8_0_f32_scalar_range(const BlockQ8_0* w_q8, const float* x, float* y, int r0, int r1, int K) noexcept {
  const int blocks = K / 32;
  for (int r = r0; r < r1; ++r) {
    const BlockQ8_0* row = w_q8 + static_cast<std::size_t>(r) * blocks;
    if (r + 2 < r1) prefetch_read(w_q8 + static_cast<std::size_t>(r + 2) * blocks);
    float acc = 0.0f;
    for (int b = 0; b < blocks; ++b) {
      const float d = fp16_to_f32_finite_scale(row[b].d_fp16);
      const float* xp = x + std::size_t(b) * 32;
      float s = 0.0f;
      for (int j = 0; j < 32; ++j) s += static_cast<float>(row[b].qs[j]) * xp[j];
      acc += d * s;
    }
    y[r] = acc;
  }
}

void gemv_q8_0_q8_0_scalar_range(const BlockQ8_0* w_q8, const BlockQ8_0* x_q8, float* y,
                                 int r0, int r1, int K) noexcept {
  const int blocks = K / 32;
  for (int r = r0; r < r1; ++r) {
    const BlockQ8_0* row = w_q8 + static_cast<std::size_t>(r) * blocks;
    if (r + 2 < r1) prefetch_read(w_q8 + static_cast<std::size_t>(r + 2) * blocks);
    float acc = 0.0f;
    for (int b = 0; b < blocks; ++b) {
      const float dw = fp16_to_f32_finite_scale(row[b].d_fp16);
      const float dx = fp16_to_f32_finite_scale(x_q8[b].d_fp16);
      std::int32_t isum = 0;
      for (int j = 0; j < 32; ++j) {
        isum += static_cast<std::int32_t>(row[b].qs[j]) * static_cast<std::int32_t>(x_q8[b].qs[j]);
      }
      acc += (dw * dx) * static_cast<float>(isum);
    }
    y[r] = acc;
  }
}

inline void extract_exl3_tile_words(const std::uint8_t* tp, int bits, std::uint32_t (&words)[24]) noexcept {
  std::memcpy(words, tp, static_cast<std::size_t>(32 * bits));
}

template <int BITS>
inline void unpack_exl3_half_states(const std::uint32_t (&words)[24], int lane, int half,
                                    std::uint32_t (&st)[4]) noexcept {
  constexpr int NW = 8 * BITS;
  const int t_off = lane * 8 + half * 4;
  const int b0 = (t_off + 257) * BITS - 16;
  const int w0 = (b0 >> 5) % NW;
  const int w1 = (w0 + 1) % NW;
  const int sh = b0 & 31;
  const std::uint64_t pair = (static_cast<std::uint64_t>(words[w0]) << 32) | static_cast<std::uint64_t>(words[w1]);
  const std::uint32_t win = static_cast<std::uint32_t>((pair << sh) >> 32);
  st[0] = (win >> 16) & 0xffffu;
  st[1] = (win >> (16 - BITS)) & 0xffffu;
  st[2] = (win >> (16 - 2 * BITS)) & 0xffffu;
  st[3] = (win >> (16 - 3 * BITS)) & 0xffffu;
}

template <int BITS>
void dequant_exl3_superblock_scalar(const std::uint8_t* codes, const std::uint16_t* rin_fp16,
                                    const std::uint16_t* rout_fp16, int rb, int rows, int cols,
                                    float* out_f32) noexcept {
  const int nt = rows / 16;
  const int kt = cols / 16;
  const std::size_t tb = static_cast<std::size_t>(32 * BITS);
  std::vector<float> mat(128u * static_cast<std::size_t>(cols), 0.0f);

  const int nt0 = (rb / 16);
  for (int nr = 0; nr < 8; ++nr) {
    const int n_t = nt0 + nr;
    for (int k_t = 0; k_t < kt; ++k_t) {
      const std::uint8_t* tp = codes + (static_cast<std::size_t>(n_t) * kt + k_t) * tb;
      if (k_t + 2 < kt) prefetch_read(tp + 2 * tb);
      std::uint32_t words[24]{};
      extract_exl3_tile_words(tp, BITS, words);
      for (int lane = 0; lane < 32; ++lane) {
        const int g = lane >> 2;
        const int ti = lane & 3;
        for (int half = 0; half < 2; ++half) {
          std::uint32_t st[4];
          unpack_exl3_half_states<BITS>(words, lane, half, st);
          const int r = nr * 16 + g + (half ? 8 : 0);
          const int c0 = k_t * 16 + 2 * ti;
          const int c1 = c0 + 8;
          mat[static_cast<std::size_t>(r) * cols + c0 + 0] = mcg_decode_state_scalar(st[0], true);
          mat[static_cast<std::size_t>(r) * cols + c0 + 1] = mcg_decode_state_scalar(st[1], true);
          mat[static_cast<std::size_t>(r) * cols + c1 + 0] = mcg_decode_state_scalar(st[2], true);
          mat[static_cast<std::size_t>(r) * cols + c1 + 1] = mcg_decode_state_scalar(st[3], true);
        }
      }
    }
  }

  for (int r = 0; r < 128; ++r) {
    float* mrow = mat.data() + static_cast<std::size_t>(r) * cols;
    for (int cb = 0; cb < cols; cb += 128) {
      fwht_128_scalar(mrow + cb);
      for (int i = 0; i < 128; ++i) {
        mrow[cb + i] *= act::host_fp16_bits_to_float(rin_fp16[cb + i]);
      }
    }
  }
  for (int c = 0; c < cols; ++c) {
    float col[128];
    for (int r = 0; r < 128; ++r) col[r] = mat[static_cast<std::size_t>(r) * cols + c];
    fwht_128_scalar(col);
    for (int r = 0; r < 128; ++r) {
      out_f32[static_cast<std::size_t>(rb + r) * cols + c] =
          col[r] * act::host_fp16_bits_to_float(rout_fp16[rb + r]);
    }
  }
}

template <int BITS>
void gemv_exl3_superblock_scalar(const std::uint8_t* codes, const std::uint16_t* rout_fp16,
                                 const float* x_packed, float* y, int rb, int cols) noexcept {
  const int kt = cols / 16;
  const std::size_t tb = static_cast<std::size_t>(32 * BITS);
  const int nt0 = rb / 16;
  float y128[128]{};

  for (int nr = 0; nr < 8; ++nr) {
    const int n_t = nt0 + nr;
    float acc16[16]{};
    for (int k_t = 0; k_t < kt; ++k_t) {
      const std::uint8_t* tp = codes + (static_cast<std::size_t>(n_t) * kt + k_t) * tb;
      if (k_t + 2 < kt) prefetch_read(tp + 2 * tb);
      std::uint32_t words[24]{};
      extract_exl3_tile_words(tp, BITS, words);
      const float* xp = x_packed + static_cast<std::size_t>(k_t) * 16;
      for (int g = 0; g < 8; ++g) {
        float s_lo = 0.0f;
        float s_hi = 0.0f;
        for (int ti = 0; ti < 4; ++ti) {
          const int lane = (g << 2) | ti;
          std::uint32_t st0[4], st1[4];
          unpack_exl3_half_states<BITS>(words, lane, 0, st0);
          unpack_exl3_half_states<BITS>(words, lane, 1, st1);
          const float x0 = xp[ti * 4 + 0];
          const float x1 = xp[ti * 4 + 1];
          const float x2 = xp[ti * 4 + 2];
          const float x3 = xp[ti * 4 + 3];
          s_lo += mcg_decode_state_scalar(st0[0], true) * x0 + mcg_decode_state_scalar(st0[1], true) * x1 +
                  mcg_decode_state_scalar(st0[2], true) * x2 + mcg_decode_state_scalar(st0[3], true) * x3;
          s_hi += mcg_decode_state_scalar(st1[0], true) * x0 + mcg_decode_state_scalar(st1[1], true) * x1 +
                  mcg_decode_state_scalar(st1[2], true) * x2 + mcg_decode_state_scalar(st1[3], true) * x3;
        }
        acc16[g] += s_lo;
        acc16[g + 8] += s_hi;
      }
    }
    for (int i = 0; i < 16; ++i) y128[nr * 16 + i] = acc16[i];
  }
  fwht_128_scalar(y128);
  for (int i = 0; i < 128; ++i) {
    y[rb + i] = y128[i] * act::host_fp16_bits_to_float(rout_fp16[rb + i]);
  }
}

#if defined(__x86_64__) || defined(_M_X64)

__attribute__((target("avx2,fma,f16c")))
inline float hsum256_ps(__m256 v) noexcept {
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  const __m128 lo = _mm256_castps256_ps128(v);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
  return _mm_cvtss_f32(s);
}

__attribute__((target("avx2,fma,f16c")))
inline std::int32_t hsum256_epi32(__m256i v) noexcept {
  const __m128i hi = _mm256_extracti128_si256(v, 1);
  const __m128i lo = _mm256_castsi256_si128(v);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return _mm_cvtsi128_si32(s);
}

__attribute__((target("avx2,fma,f16c")))
void fwht_128_avx2(float* v) noexcept {
  const __m256 sign1 = _mm256_set_ps(-1.f, 1.f, -1.f, 1.f, -1.f, 1.f, -1.f, 1.f);
  const __m256 sign2 = _mm256_set_ps(-1.f, -1.f, 1.f, 1.f, -1.f, -1.f, 1.f, 1.f);
  const __m256 sign4 = _mm256_set_ps(-1.f, -1.f, -1.f, -1.f, 1.f, 1.f, 1.f, 1.f);
  for (int i = 0; i < 128; i += 8) {
    __m256 x = _mm256_loadu_ps(v + i);
    // h = 1: pairs (0,1),(2,3),(4,5),(6,7)
    x = _mm256_add_ps(_mm256_mul_ps(x, sign1), _mm256_permute_ps(x, 0xB1));
    // h = 2: pairs (0,2),(1,3),(4,6),(5,7)
    x = _mm256_add_ps(_mm256_mul_ps(x, sign2), _mm256_permute_ps(x, 0x4E));
    // h = 4: halves of 128-bit lane
    const __m256 lo_hi_sw = _mm256_permute2f128_ps(x, x, 0x01);  // swaps 128-bit halves (distance 4)!
    x = _mm256_add_ps(_mm256_mul_ps(x, sign4), lo_hi_sw);
    _mm256_storeu_ps(v + i, x);
  }
  for (int h = 8; h < 128; h <<= 1) {
    for (int i = 0; i < 128; i += 2 * h) {
      for (int j = i; j < i + h; j += 8) {
        const __m256 a = _mm256_loadu_ps(v + j);
        const __m256 b = _mm256_loadu_ps(v + j + h);
        _mm256_storeu_ps(v + j, _mm256_add_ps(a, b));
        _mm256_storeu_ps(v + j + h, _mm256_sub_ps(a, b));
      }
    }
  }
  const __m256 vinv = _mm256_set1_ps(kInvSqrt128);
  for (int i = 0; i < 128; i += 8) {
    _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_loadu_ps(v + i), vinv));
  }
}

__attribute__((target("avx2,fma,f16c")))
void decode_q4_0_row_avx2(const std::uint8_t* row_bytes, int hidden_size, float* out_f32) noexcept {
  const int nib_bytes = hidden_size / 2;
  const int blocks = hidden_size / 32;
  const std::uint8_t* scales = row_bytes + nib_bytes;
  const __m128i mask0f = _mm_set1_epi8(0x0f);
  for (int b = 0; b < blocks; ++b) {
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, scales + std::size_t(b) * 2, 2);
    const float d = fp16_to_f32_finite_scale(d_bits);
    const __m256 vd = _mm256_set1_ps(d);
    const __m256 vm8d = _mm256_set1_ps(-8.0f * d);
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_bytes + std::size_t(b) * 16));
    const __m128i lo = _mm_and_si128(raw, mask0f);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask0f);
    const __m128i u0_15 = _mm_unpacklo_epi8(lo, hi);
    const __m128i u16_31 = _mm_unpackhi_epi8(lo, hi);

    float* dst = out_f32 + std::size_t(b) * 32;
    const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u0_15));
    const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(u0_15, 8)));
    const __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u16_31));
    const __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(u16_31, 8)));
    _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(f0, vd, vm8d));
    _mm256_storeu_ps(dst + 8, _mm256_fmadd_ps(f1, vd, vm8d));
    _mm256_storeu_ps(dst + 16, _mm256_fmadd_ps(f2, vd, vm8d));
    _mm256_storeu_ps(dst + 24, _mm256_fmadd_ps(f3, vd, vm8d));
  }
}

__attribute__((target("avx2,fma,f16c")))
void gemv_q4_0_f32_avx2_range(const std::uint8_t* w_q4, const float* x, float* y, int r0, int r1, int K) noexcept {
  const int nib_bytes = K / 2;
  const int blocks = K / 32;
  const std::size_t row_bytes = static_cast<std::size_t>(nib_bytes) + static_cast<std::size_t>(blocks) * 2;
  const __m128i mask0f = _mm_set1_epi8(0x0f);
  const __m256 v_neg8 = _mm256_set1_ps(-8.0f);

  for (int r = r0; r < r1; ++r) {
    const std::uint8_t* row = w_q4 + static_cast<std::size_t>(r) * row_bytes;
    if (r + 2 < r1) {
      const std::uint8_t* pf = w_q4 + static_cast<std::size_t>(r + 2) * row_bytes;
      prefetch_read(pf);
      prefetch_read(pf + nib_bytes);
    }
    const std::uint8_t* scales = row + nib_bytes;
    __m256 vacc0 = _mm256_setzero_ps();
    __m256 vacc1 = _mm256_setzero_ps();

    for (int b = 0; b < blocks; ++b) {
      std::uint16_t d_bits = 0;
      std::memcpy(&d_bits, scales + std::size_t(b) * 2, 2);
      const float d = fp16_to_f32_finite_scale(d_bits);
      const __m256 vd = _mm256_set1_ps(d);
      const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + std::size_t(b) * 16));
      const __m128i lo = _mm_and_si128(raw, mask0f);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask0f);
      const __m128i u0_15 = _mm_unpacklo_epi8(lo, hi);
      const __m128i u16_31 = _mm_unpackhi_epi8(lo, hi);

      const float* xp = x + std::size_t(b) * 32;
      const __m256 x0 = _mm256_loadu_ps(xp + 0);
      const __m256 x1 = _mm256_loadu_ps(xp + 8);
      const __m256 x2 = _mm256_loadu_ps(xp + 16);
      const __m256 x3 = _mm256_loadu_ps(xp + 24);

      const __m256 q0 = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u0_15)), v_neg8);
      const __m256 q1 = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(u0_15, 8))), v_neg8);
      const __m256 q2 = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u16_31)), v_neg8);
      const __m256 q3 = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(u16_31, 8))), v_neg8);

      __m256 bsum0 = _mm256_mul_ps(q0, x0);
      bsum0 = _mm256_fmadd_ps(q1, x1, bsum0);
      __m256 bsum1 = _mm256_mul_ps(q2, x2);
      bsum1 = _mm256_fmadd_ps(q3, x3, bsum1);
      vacc0 = _mm256_fmadd_ps(vd, bsum0, vacc0);
      vacc1 = _mm256_fmadd_ps(vd, bsum1, vacc1);
    }
    y[r] = hsum256_ps(_mm256_add_ps(vacc0, vacc1));
  }
}

__attribute__((target("avx2,fma,f16c")))
void gemv_q8_0_f32_avx2_range(const BlockQ8_0* w_q8, const float* x, float* y, int r0, int r1, int K) noexcept {
  const int blocks = K / 32;
  for (int r = r0; r < r1; ++r) {
    const BlockQ8_0* row = w_q8 + static_cast<std::size_t>(r) * blocks;
    if (r + 2 < r1) prefetch_read(w_q8 + static_cast<std::size_t>(r + 2) * blocks);
    __m256 vacc0 = _mm256_setzero_ps();
    __m256 vacc1 = _mm256_setzero_ps();
    for (int b = 0; b < blocks; ++b) {
      const float d = fp16_to_f32_finite_scale(row[b].d_fp16);
      const __m256 vd = _mm256_set1_ps(d);
      const __m128i q_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[b].qs));
      const __m128i q_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[b].qs + 16));
      const float* xp = x + std::size_t(b) * 32;
      const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_lo));
      const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8)));
      const __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q_hi));
      const __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8)));

      __m256 bs0 = _mm256_mul_ps(f0, _mm256_loadu_ps(xp + 0));
      bs0 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xp + 8), bs0);
      __m256 bs1 = _mm256_mul_ps(f2, _mm256_loadu_ps(xp + 16));
      bs1 = _mm256_fmadd_ps(f3, _mm256_loadu_ps(xp + 24), bs1);
      vacc0 = _mm256_fmadd_ps(vd, bs0, vacc0);
      vacc1 = _mm256_fmadd_ps(vd, bs1, vacc1);
    }
    y[r] = hsum256_ps(_mm256_add_ps(vacc0, vacc1));
  }
}

__attribute__((target("avx2,fma,f16c")))
void gemv_q8_0_q8_0_avx2_range(const BlockQ8_0* w_q8, const BlockQ8_0* x_q8, float* y,
                               int r0, int r1, int K) noexcept {
  const int blocks = K / 32;
  const __m256i ones16 = _mm256_set1_epi16(1);
  int r = r0;
  for (; r + 1 < r1; r += 2) {
    const BlockQ8_0* row0 = w_q8 + static_cast<std::size_t>(r) * blocks;
    const BlockQ8_0* row1 = w_q8 + static_cast<std::size_t>(r + 1) * blocks;
    if (r + 3 < r1) {
      prefetch_read(w_q8 + static_cast<std::size_t>(r + 2) * blocks);
      prefetch_read(w_q8 + static_cast<std::size_t>(r + 3) * blocks);
    }
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int b = 0; b < blocks; ++b) {
      const float dx = fp16_to_f32_finite_scale(x_q8[b].d_fp16);
      const float dw0 = fp16_to_f32_finite_scale(row0[b].d_fp16);
      const float dw1 = fp16_to_f32_finite_scale(row1[b].d_fp16);
      const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8[b].qs));

      const __m256i vw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row0[b].qs));
      const __m256i aw0 = _mm256_sign_epi8(vw0, vw0);
      const __m256i sx0 = _mm256_sign_epi8(vx, vw0);
      const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw0, sx0), ones16);
      acc0 += (dw0 * dx) * static_cast<float>(hsum256_epi32(p0));

      const __m256i vw1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row1[b].qs));
      const __m256i aw1 = _mm256_sign_epi8(vw1, vw1);
      const __m256i sx1 = _mm256_sign_epi8(vx, vw1);
      const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw1, sx1), ones16);
      acc1 += (dw1 * dx) * static_cast<float>(hsum256_epi32(p1));
    }
    y[r] = acc0;
    y[r + 1] = acc1;
  }
  if (r < r1) {
    const BlockQ8_0* row0 = w_q8 + static_cast<std::size_t>(r) * blocks;
    float acc0 = 0.0f;
    for (int b = 0; b < blocks; ++b) {
      const float dx = fp16_to_f32_finite_scale(x_q8[b].d_fp16);
      const float dw0 = fp16_to_f32_finite_scale(row0[b].d_fp16);
      const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8[b].qs));
      const __m256i vw0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row0[b].qs));
      const __m256i p0 = _mm256_madd_epi16(
          _mm256_maddubs_epi16(_mm256_sign_epi8(vw0, vw0), _mm256_sign_epi8(vx, vw0)), ones16);
      acc0 += (dw0 * dx) * static_cast<float>(hsum256_epi32(p0));
    }
    y[r] = acc0;
  }
}

__attribute__((target("avx2,fma,f16c")))
inline __m256 mcg_decode_8_avx2(__m256i vs, __m256i mul_c, __m256i mask_c, __m256i xor_c,
                                __m256i pack_shuf) noexcept {
  const __m256i p = _mm256_mullo_epi32(vs, mul_c);
  const __m256i u = _mm256_xor_si256(_mm256_and_si256(p, mask_c), xor_c);
  const __m256i sh = _mm256_shuffle_epi8(u, pack_shuf);
  const __m128i u0 = _mm256_castsi256_si128(sh);
  const __m128i u1 = _mm256_extracti128_si256(sh, 1);
  const __m128i lo_h = _mm_unpacklo_epi64(u0, u1);
  const __m128i hi_h = _mm_unpackhi_epi64(u0, u1);
  const __m256 sum = _mm256_add_ps(_mm256_cvtph_ps(lo_h), _mm256_cvtph_ps(hi_h));
  return _mm256_cvtph_ps(_mm256_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

template <int BITS>
__attribute__((target("avx2,fma,f16c")))
void gemv_exl3_superblock_avx2(const std::uint8_t* codes, const std::uint16_t* rout_fp16,
                               const float* x_packed, float* y, int rb, int cols) noexcept {
  const int kt = cols / 16;
  const std::size_t tb = static_cast<std::size_t>(32 * BITS);
  const int nt0 = rb / 16;
  const __m256i mul_c = _mm256_set1_epi32(static_cast<int>(0xCBAC1FEDu));
  const __m256i mask_c = _mm256_set1_epi32(static_cast<int>(0x8fff8fffu));
  const __m256i xor_c = _mm256_set1_epi32(static_cast<int>(0x3b603b60u));
  // Pack 4 low uint16 (bytes 0,1,4,5,8,9,12,13) to bytes 0..7 and 4 high uint16 (2,3,6,7,10,11,14,15) to 8..15.
  const __m256i pack_shuf = _mm256_set_epi8(
      15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0,
      15, 14, 11, 10, 7, 6, 3, 2, 13, 12, 9, 8, 5, 4, 1, 0);

  alignas(32) float y128[128]{};
  for (int nr = 0; nr < 8; ++nr) {
    const int n_t = nt0 + nr;
    // Maintain 16 row accumulators in 8 pairs (g = 0..7 -> row g & row g+8, summed over 8-wide vectors).
    __m256 acc_lo[8];
    __m256 acc_hi[8];
    for (int g = 0; g < 8; ++g) {
      acc_lo[g] = _mm256_setzero_ps();
      acc_hi[g] = _mm256_setzero_ps();
    }

    for (int k_t = 0; k_t < kt; ++k_t) {
      const std::uint8_t* tp = codes + (static_cast<std::size_t>(n_t) * kt + k_t) * tb;
      if (k_t + 2 < kt) {
        prefetch_read(tp + 2 * tb);
        prefetch_read(tp + 2 * tb + 64);
      }
      std::uint32_t words[24]{};
      extract_exl3_tile_words(tp, BITS, words);
      const float* xp = x_packed + static_cast<std::size_t>(k_t) * 16;
      const __m256 vx0 = _mm256_loadu_ps(xp + 0);
      const __m256 vx1 = _mm256_loadu_ps(xp + 8);

      for (int g = 0; g < 8; ++g) {
        alignas(32) std::uint32_t s_lo0[8], s_lo1[8], s_hi0[8], s_hi1[8];
        for (int ti = 0; ti < 2; ++ti) {
          unpack_exl3_half_states<BITS>(words, (g << 2) | ti, 0,
                                        *reinterpret_cast<std::uint32_t(*)[4]>(&s_lo0[ti * 4]));
          unpack_exl3_half_states<BITS>(words, (g << 2) | (ti + 2), 0,
                                        *reinterpret_cast<std::uint32_t(*)[4]>(&s_lo1[ti * 4]));
          unpack_exl3_half_states<BITS>(words, (g << 2) | ti, 1,
                                        *reinterpret_cast<std::uint32_t(*)[4]>(&s_hi0[ti * 4]));
          unpack_exl3_half_states<BITS>(words, (g << 2) | (ti + 2), 1,
                                        *reinterpret_cast<std::uint32_t(*)[4]>(&s_hi1[ti * 4]));
        }
        const __m256 wlo0 = mcg_decode_8_avx2(_mm256_load_si256(reinterpret_cast<const __m256i*>(s_lo0)),
                                              mul_c, mask_c, xor_c, pack_shuf);
        const __m256 wlo1 = mcg_decode_8_avx2(_mm256_load_si256(reinterpret_cast<const __m256i*>(s_lo1)),
                                              mul_c, mask_c, xor_c, pack_shuf);
        const __m256 whi0 = mcg_decode_8_avx2(_mm256_load_si256(reinterpret_cast<const __m256i*>(s_hi0)),
                                              mul_c, mask_c, xor_c, pack_shuf);
        const __m256 whi1 = mcg_decode_8_avx2(_mm256_load_si256(reinterpret_cast<const __m256i*>(s_hi1)),
                                              mul_c, mask_c, xor_c, pack_shuf);
        acc_lo[g] = _mm256_fmadd_ps(wlo0, vx0, acc_lo[g]);
        acc_lo[g] = _mm256_fmadd_ps(wlo1, vx1, acc_lo[g]);
        acc_hi[g] = _mm256_fmadd_ps(whi0, vx0, acc_hi[g]);
        acc_hi[g] = _mm256_fmadd_ps(whi1, vx1, acc_hi[g]);
      }
    }
    for (int g = 0; g < 8; ++g) {
      y128[nr * 16 + g] = hsum256_ps(acc_lo[g]);
      y128[nr * 16 + g + 8] = hsum256_ps(acc_hi[g]);
    }
  }
  fwht_128_avx2(y128);
  for (int i = 0; i < 128; ++i) {
    y[rb + i] = y128[i] * act::host_fp16_bits_to_float(rout_fp16[rb + i]);
  }
}

// ----------------------------------------------------------------------------------------------- AVX-512 path
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c")))
inline float hsum512_ps(__m512 v) noexcept {
  return _mm512_reduce_add_ps(v);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c")))
void fwht_128_avx512(float* v) noexcept {
  fwht_128_avx2(v);
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c")))
void decode_q4_0_row_avx512(const std::uint8_t* row_bytes, int hidden_size, float* out_f32) noexcept {
  const int nib_bytes = hidden_size / 2;
  const int blocks = hidden_size / 32;
  const std::uint8_t* scales = row_bytes + nib_bytes;
  const __m128i mask0f = _mm_set1_epi8(0x0f);
  for (int b = 0; b < blocks; ++b) {
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, scales + std::size_t(b) * 2, 2);
    const float d = fp16_to_f32_finite_scale(d_bits);
    const __m512 vd = _mm512_set1_ps(d);
    const __m512 vm8d = _mm512_set1_ps(-8.0f * d);
    const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_bytes + std::size_t(b) * 16));
    const __m128i lo = _mm_and_si128(raw, mask0f);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask0f);
    const __m128i u0_15 = _mm_unpacklo_epi8(lo, hi);
    const __m128i u16_31 = _mm_unpackhi_epi8(lo, hi);
    float* dst = out_f32 + std::size_t(b) * 32;
    _mm512_storeu_ps(dst + 0, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u0_15)), vd, vm8d));
    _mm512_storeu_ps(dst + 16, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u16_31)), vd, vm8d));
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c")))
void gemv_q4_0_f32_avx512_range(const std::uint8_t* w_q4, const float* x, float* y, int r0, int r1, int K) noexcept {
  const int nib_bytes = K / 2;
  const int blocks = K / 32;
  const std::size_t row_bytes = static_cast<std::size_t>(nib_bytes) + static_cast<std::size_t>(blocks) * 2;
  const __m128i mask0f = _mm_set1_epi8(0x0f);
  const __m512 v_neg8 = _mm512_set1_ps(-8.0f);
  for (int r = r0; r < r1; ++r) {
    const std::uint8_t* row = w_q4 + static_cast<std::size_t>(r) * row_bytes;
    if (r + 2 < r1) {
      const std::uint8_t* pf = w_q4 + static_cast<std::size_t>(r + 2) * row_bytes;
      prefetch_read(pf);
      prefetch_read(pf + nib_bytes);
    }
    const std::uint8_t* scales = row + nib_bytes;
    __m512 vacc = _mm512_setzero_ps();
    for (int b = 0; b < blocks; ++b) {
      std::uint16_t d_bits = 0;
      std::memcpy(&d_bits, scales + std::size_t(b) * 2, 2);
      const float d = fp16_to_f32_finite_scale(d_bits);
      const __m512 vd = _mm512_set1_ps(d);
      const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + std::size_t(b) * 16));
      const __m128i lo = _mm_and_si128(raw, mask0f);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(raw, 4), mask0f);
      const __m128i u0_15 = _mm_unpacklo_epi8(lo, hi);
      const __m128i u16_31 = _mm_unpackhi_epi8(lo, hi);
      const float* xp = x + std::size_t(b) * 32;
      const __m512 q0 = _mm512_add_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u0_15)), v_neg8);
      const __m512 q1 = _mm512_add_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u16_31)), v_neg8);
      __m512 bs = _mm512_mul_ps(q0, _mm512_loadu_ps(xp + 0));
      bs = _mm512_fmadd_ps(q1, _mm512_loadu_ps(xp + 16), bs);
      vacc = _mm512_fmadd_ps(vd, bs, vacc);
    }
    y[r] = hsum512_ps(vacc);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c")))
void gemv_q8_0_f32_avx512_range(const BlockQ8_0* w_q8, const float* x, float* y, int r0, int r1, int K) noexcept {
  const int blocks = K / 32;
  for (int r = r0; r < r1; ++r) {
    const BlockQ8_0* row = w_q8 + static_cast<std::size_t>(r) * blocks;
    if (r + 2 < r1) prefetch_read(w_q8 + static_cast<std::size_t>(r + 2) * blocks);
    __m512 vacc = _mm512_setzero_ps();
    for (int b = 0; b < blocks; ++b) {
      const float d = fp16_to_f32_finite_scale(row[b].d_fp16);
      const __m512 vd = _mm512_set1_ps(d);
      const __m128i q_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[b].qs));
      const __m128i q_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row[b].qs + 16));
      const float* xp = x + std::size_t(b) * 32;
      const __m512 f0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q_lo));
      const __m512 f1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q_hi));
      __m512 bs = _mm512_mul_ps(f0, _mm512_loadu_ps(xp + 0));
      bs = _mm512_fmadd_ps(f1, _mm512_loadu_ps(xp + 16), bs);
      vacc = _mm512_fmadd_ps(vd, bs, vacc);
    }
    y[r] = hsum512_ps(vacc);
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c")))
void gemv_q8_0_q8_0_avx512_range(const BlockQ8_0* w_q8, const BlockQ8_0* x_q8, float* y,
                                 int r0, int r1, int K) noexcept {
  // Reuse AVX2 vpmaddubsw across block-32 (or pair 2 blocks into 512-bit when blocks is even).
  const int blocks = K / 32;
  const __m256i ones16 = _mm256_set1_epi16(1);
  for (int r = r0; r < r1; ++r) {
    const BlockQ8_0* row = w_q8 + static_cast<std::size_t>(r) * blocks;
    if (r + 2 < r1) prefetch_read(w_q8 + static_cast<std::size_t>(r + 2) * blocks);
    float acc = 0.0f;
    for (int b = 0; b < blocks; ++b) {
      const float dw = fp16_to_f32_finite_scale(row[b].d_fp16);
      const float dx = fp16_to_f32_finite_scale(x_q8[b].d_fp16);
      const __m256i vw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row[b].qs));
      const __m256i vx = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x_q8[b].qs));
      const __m256i p = _mm256_madd_epi16(
          _mm256_maddubs_epi16(_mm256_sign_epi8(vw, vw), _mm256_sign_epi8(vx, vw)), ones16);
      acc += (dw * dx) * static_cast<float>(hsum256_epi32(p));
    }
    y[r] = acc;
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c")))
inline __m512 mcg_decode_16_avx512(__m512i vs, __m512i mul_c, __m512i mask_c, __m512i xor_c,
                                   __m512i lo_mask) noexcept {
  const __m512i p = _mm512_mullo_epi32(vs, mul_c);
  const __m512i u = _mm512_xor_si512(_mm512_and_si512(p, mask_c), xor_c);
  const __m256i lo16 = _mm512_cvtepi32_epi16(_mm512_and_si512(u, lo_mask));
  const __m256i hi16 = _mm512_cvtepi32_epi16(_mm512_srli_epi32(u, 16));
  const __m512 sum = _mm512_add_ps(_mm512_cvtph_ps(lo16), _mm512_cvtph_ps(hi16));
  return _mm512_cvtph_ps(_mm512_cvtps_ph(sum, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

template <int BITS>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,f16c")))
void gemv_exl3_superblock_avx512(const std::uint8_t* codes, const std::uint16_t* rout_fp16,
                                 const float* x_packed, float* y, int rb, int cols) noexcept {
  const int kt = cols / 16;
  const std::size_t tb = static_cast<std::size_t>(32 * BITS);
  const int nt0 = rb / 16;
  const __m512i mul_c = _mm512_set1_epi32(static_cast<int>(0xCBAC1FEDu));
  const __m512i mask_c = _mm512_set1_epi32(static_cast<int>(0x8fff8fffu));
  const __m512i xor_c = _mm512_set1_epi32(static_cast<int>(0x3b603b60u));
  const __m512i lo_mask = _mm512_set1_epi32(0x0000ffff);

  alignas(64) float y128[128]{};
  for (int nr = 0; nr < 8; ++nr) {
    const int n_t = nt0 + nr;
    __m512 acc_lo[8], acc_hi[8];
    for (int g = 0; g < 8; ++g) {
      acc_lo[g] = _mm512_setzero_ps();
      acc_hi[g] = _mm512_setzero_ps();
    }
    for (int k_t = 0; k_t < kt; ++k_t) {
      const std::uint8_t* tp = codes + (static_cast<std::size_t>(n_t) * kt + k_t) * tb;
      if (k_t + 2 < kt) prefetch_read(tp + 2 * tb);
      std::uint32_t words[24]{};
      extract_exl3_tile_words(tp, BITS, words);
      const __m512 vx = _mm512_loadu_ps(x_packed + static_cast<std::size_t>(k_t) * 16);
      for (int g = 0; g < 8; ++g) {
        alignas(64) std::uint32_t slo[16], shi[16];
        for (int ti = 0; ti < 4; ++ti) {
          const int lane = (g << 2) | ti;
          unpack_exl3_half_states<BITS>(words, lane, 0,
                                        *reinterpret_cast<std::uint32_t(*)[4]>(&slo[ti * 4]));
          unpack_exl3_half_states<BITS>(words, lane, 1,
                                        *reinterpret_cast<std::uint32_t(*)[4]>(&shi[ti * 4]));
        }
        acc_lo[g] = _mm512_fmadd_ps(
            mcg_decode_16_avx512(_mm512_load_si512(slo), mul_c, mask_c, xor_c, lo_mask), vx, acc_lo[g]);
        acc_hi[g] = _mm512_fmadd_ps(
            mcg_decode_16_avx512(_mm512_load_si512(shi), mul_c, mask_c, xor_c, lo_mask), vx, acc_hi[g]);
      }
    }
    for (int g = 0; g < 8; ++g) {
      y128[nr * 16 + g] = hsum512_ps(acc_lo[g]);
      y128[nr * 16 + g + 8] = hsum512_ps(acc_hi[g]);
    }
  }
  fwht_128_avx512(y128);
  for (int i = 0; i < 128; ++i) {
    y[rb + i] = y128[i] * act::host_fp16_bits_to_float(rout_fp16[rb + i]);
  }
}

#endif  // x86_64

#if defined(__aarch64__) || defined(__ARM_NEON)

void fwht_128_neon(float* v) noexcept {
  for (int h = 1; h < 128; h <<= 1) {
    if (h < 4) {
      for (int i = 0; i < 128; i += 2 * h) {
        for (int j = i; j < i + h; ++j) {
          const float a = v[j], b = v[j + h];
          v[j] = a + b;
          v[j + h] = a - b;
        }
      }
    } else {
      for (int i = 0; i < 128; i += 2 * h) {
        for (int j = i; j < i + h; j += 4) {
          const float32x4_t a = vld1q_f32(v + j);
          const float32x4_t b = vld1q_f32(v + j + h);
          vst1q_f32(v + j, vaddq_f32(a, b));
          vst1q_f32(v + j + h, vsubq_f32(a, b));
        }
      }
    }
  }
  const float32x4_t vinv = vdupq_n_f32(kInvSqrt128);
  for (int i = 0; i < 128; i += 4) {
    vst1q_f32(v + i, vmulq_f32(vld1q_f32(v + i), vinv));
  }
}

void decode_q4_0_row_neon(const std::uint8_t* row_bytes, int hidden_size, float* out_f32) noexcept {
  const int nib_bytes = hidden_size / 2;
  const int blocks = hidden_size / 32;
  const std::uint8_t* scales = row_bytes + nib_bytes;
  const uint8x16_t mask0f = vdupq_n_u8(0x0f);
  for (int b = 0; b < blocks; ++b) {
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, scales + std::size_t(b) * 2, 2);
    const float d = fp16_to_f32_finite_scale(d_bits);
    const float32x4_t vd = vdupq_n_f32(d);
    const float32x4_t vm8d = vdupq_n_f32(-8.0f * d);
    const uint8x16_t raw = vld1q_u8(row_bytes + std::size_t(b) * 16);
    const uint8x16_t lo = vandq_u8(raw, mask0f);
    const uint8x16_t hi = vshrq_n_u8(raw, 4);
    const uint8x16x2_t interleaved = vzipq_u8(lo, hi);
    const uint8x16_t chunks[2] = {interleaved.val[0], interleaved.val[1]};
    float* dst = out_f32 + std::size_t(b) * 32;
    for (int c = 0; c < 2; ++c) {
      const uint16x8_t u16_lo = vmovl_u8(vget_low_u8(chunks[c]));
      const uint16x8_t u16_hi = vmovl_u8(vget_high_u8(chunks[c]));
      const float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_lo)));
      const float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_lo)));
      const float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16_hi)));
      const float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16_hi)));
      vst1q_f32(dst + c * 16 + 0, vfmaq_f32(vm8d, f0, vd));
      vst1q_f32(dst + c * 16 + 4, vfmaq_f32(vm8d, f1, vd));
      vst1q_f32(dst + c * 16 + 8, vfmaq_f32(vm8d, f2, vd));
      vst1q_f32(dst + c * 16 + 12, vfmaq_f32(vm8d, f3, vd));
    }
  }
}

void gemv_q4_0_f32_neon_range(const std::uint8_t* w_q4, const float* x, float* y, int r0, int r1, int K) noexcept {
  const int nib_bytes = K / 2;
  const int blocks = K / 32;
  const std::size_t row_bytes = static_cast<std::size_t>(nib_bytes) + static_cast<std::size_t>(blocks) * 2;
  const uint8x16_t mask0f = vdupq_n_u8(0x0f);
  const float32x4_t v_neg8 = vdupq_n_f32(-8.0f);
  for (int r = r0; r < r1; ++r) {
    const std::uint8_t* row = w_q4 + static_cast<std::size_t>(r) * row_bytes;
    if (r + 2 < r1) prefetch_read(w_q4 + static_cast<std::size_t>(r + 2) * row_bytes);
    const std::uint8_t* scales = row + nib_bytes;
    float32x4_t vacc0 = vdupq_n_f32(0.0f);
    float32x4_t vacc1 = vdupq_n_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
      std::uint16_t d_bits = 0;
      std::memcpy(&d_bits, scales + std::size_t(b) * 2, 2);
      const float32x4_t vd = vdupq_n_f32(fp16_to_f32_finite_scale(d_bits));
      const uint8x16_t raw = vld1q_u8(row + std::size_t(b) * 16);
      const uint8x16x2_t interleaved = vzipq_u8(vandq_u8(raw, mask0f), vshrq_n_u8(raw, 4));
      const float* xp = x + std::size_t(b) * 32;
      float32x4_t bs0 = vdupq_n_f32(0.0f), bs1 = vdupq_n_f32(0.0f);
      for (int c = 0; c < 2; ++c) {
        const uint16x8_t u_lo = vmovl_u8(vget_low_u8(interleaved.val[c]));
        const uint16x8_t u_hi = vmovl_u8(vget_high_u8(interleaved.val[c]));
        const float32x4_t f0 = vaddq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u_lo))), v_neg8);
        const float32x4_t f1 = vaddq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u_lo))), v_neg8);
        const float32x4_t f2 = vaddq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(u_hi))), v_neg8);
        const float32x4_t f3 = vaddq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(u_hi))), v_neg8);
        bs0 = vfmaq_f32(bs0, f0, vld1q_f32(xp + c * 16 + 0));
        bs1 = vfmaq_f32(bs1, f1, vld1q_f32(xp + c * 16 + 4));
        bs0 = vfmaq_f32(bs0, f2, vld1q_f32(xp + c * 16 + 8));
        bs1 = vfmaq_f32(bs1, f3, vld1q_f32(xp + c * 16 + 12));
      }
      vacc0 = vfmaq_f32(vacc0, vd, bs0);
      vacc1 = vfmaq_f32(vacc1, vd, bs1);
    }
    y[r] = vaddvq_f32(vaddq_f32(vacc0, vacc1));
  }
}

void gemv_q8_0_f32_neon_range(const BlockQ8_0* w_q8, const float* x, float* y, int r0, int r1, int K) noexcept {
  const int blocks = K / 32;
  for (int r = r0; r < r1; ++r) {
    const BlockQ8_0* row = w_q8 + static_cast<std::size_t>(r) * blocks;
    if (r + 2 < r1) prefetch_read(w_q8 + static_cast<std::size_t>(r + 2) * blocks);
    float32x4_t vacc0 = vdupq_n_f32(0.0f);
    float32x4_t vacc1 = vdupq_n_f32(0.0f);
    for (int b = 0; b < blocks; ++b) {
      const float32x4_t vd = vdupq_n_f32(fp16_to_f32_finite_scale(row[b].d_fp16));
      const float* xp = x + std::size_t(b) * 32;
      float32x4_t bs0 = vdupq_n_f32(0.0f), bs1 = vdupq_n_f32(0.0f);
      for (int c = 0; c < 2; ++c) {
        const int8x16_t q = vld1q_s8(row[b].qs + c * 16);
        const int16x8_t q_lo = vmovl_s8(vget_low_s8(q));
        const int16x8_t q_hi = vmovl_s8(vget_high_s8(q));
        bs0 = vfmaq_f32(bs0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_lo))), vld1q_f32(xp + c * 16 + 0));
        bs1 = vfmaq_f32(bs1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_lo))), vld1q_f32(xp + c * 16 + 4));
        bs0 = vfmaq_f32(bs0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_hi))), vld1q_f32(xp + c * 16 + 8));
        bs1 = vfmaq_f32(bs1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_hi))), vld1q_f32(xp + c * 16 + 12));
      }
      vacc0 = vfmaq_f32(vacc0, vd, bs0);
      vacc1 = vfmaq_f32(vacc1, vd, bs1);
    }
    y[r] = vaddvq_f32(vaddq_f32(vacc0, vacc1));
  }
}

void gemv_q8_0_q8_0_neon_range(const BlockQ8_0* w_q8, const BlockQ8_0* x_q8, float* y,
                               int r0, int r1, int K) noexcept {
  const int blocks = K / 32;
  for (int r = r0; r < r1; ++r) {
    const BlockQ8_0* row = w_q8 + static_cast<std::size_t>(r) * blocks;
    if (r + 2 < r1) prefetch_read(w_q8 + static_cast<std::size_t>(r + 2) * blocks);
    float acc = 0.0f;
    for (int b = 0; b < blocks; ++b) {
      const float dw = fp16_to_f32_finite_scale(row[b].d_fp16);
      const float dx = fp16_to_f32_finite_scale(x_q8[b].d_fp16);
      const int8x16_t w0 = vld1q_s8(row[b].qs);
      const int8x16_t w1 = vld1q_s8(row[b].qs + 16);
      const int8x16_t x0 = vld1q_s8(x_q8[b].qs);
      const int8x16_t x1 = vld1q_s8(x_q8[b].qs + 16);
#if defined(__ARM_FEATURE_DOTPROD)
      int32x4_t sum = vdotq_s32(vdupq_n_s32(0), w0, x0);
      sum = vdotq_s32(sum, w1, x1);
      const std::int32_t isum = vaddvq_s32(sum);
#else
      const int16x8_t p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
      const int16x8_t p1 = vmull_s8(vget_high_s8(w0), vget_high_s8(x0));
      const int16x8_t p2 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
      const int16x8_t p3 = vmull_s8(vget_high_s8(w1), vget_high_s8(x1));
      const int32x4_t sum = vaddq_s32(vaddl_s16(vget_low_s16(p0), vget_high_s16(p0)),
                                      vaddl_s16(vget_low_s16(p1), vget_high_s16(p1)));
      const int32x4_t sum2 = vaddq_s32(vaddl_s16(vget_low_s16(p2), vget_high_s16(p2)),
                                       vaddl_s16(vget_low_s16(p3), vget_high_s16(p3)));
      const std::int32_t isum = vaddvq_s32(vaddq_s32(sum, sum2));
#endif
      acc += (dw * dx) * static_cast<float>(isum);
    }
    y[r] = acc;
  }
}

#endif  // aarch64

void validate_gemv_dims(const void* w, const void* x, void* y, int N, int K) {
  if (!w || !x || !y) throw std::invalid_argument("cpu gemv: null pointer");
  if (N < 1 || K <= 0 || (K % 32) != 0) {
    throw std::invalid_argument("cpu gemv: N must be >= 1 and K must be a positive multiple of 32");
  }
}

void validate_exl3_dims(const void* codes, const void* rin, const void* rout, int bits, int rows, int cols) {
  if (!codes || !rin || !rout) throw std::invalid_argument("exl3 cpu: null pointer");
  if (bits != 2 && bits != 3) throw std::invalid_argument("exl3 cpu: bits must be 2 or 3");
  if (rows <= 0 || cols <= 0 || (rows % 128) != 0 || (cols % 128) != 0) {
    throw std::invalid_argument("exl3 cpu: rows and cols must be positive multiples of 128");
  }
}

void prepare_exl3_input_packed(const std::uint16_t* rin_fp16, const float* x, int cols, CpuIsa isa,
                               std::vector<float>& out_packed) {
  std::vector<float> tmp(static_cast<std::size_t>(cols));
  for (int c = 0; c < cols; ++c) {
    tmp[static_cast<std::size_t>(c)] = x[c] * act::host_fp16_bits_to_float(rin_fp16[c]);
  }
  for (int cb = 0; cb < cols; cb += 128) {
    fwht_128_cpu(tmp.data() + cb, isa);
  }
  out_packed.resize(static_cast<std::size_t>(cols));
  const int kt = cols / 16;
  for (int k_t = 0; k_t < kt; ++k_t) {
    const float* src = tmp.data() + static_cast<std::size_t>(k_t) * 16;
    float* dst = out_packed.data() + static_cast<std::size_t>(k_t) * 16;
    for (int ti = 0; ti < 4; ++ti) {
      dst[ti * 4 + 0] = src[2 * ti + 0];
      dst[ti * 4 + 1] = src[2 * ti + 1];
      dst[ti * 4 + 2] = src[2 * ti + 8];
      dst[ti * 4 + 3] = src[2 * ti + 9];
    }
  }
}

}  // namespace

CpuIsa detect_cpu_isa() noexcept {
  static const CpuIsa kDetected = []() noexcept -> CpuIsa {
#if defined(__x86_64__) || defined(_M_X64)
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
        __builtin_cpu_supports("f16c")) {
      return CpuIsa::Avx512;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
        __builtin_cpu_supports("f16c")) {
      return CpuIsa::Avx2;
    }
    return CpuIsa::Scalar;
#elif defined(__aarch64__) || defined(__ARM_NEON)
    return CpuIsa::Neon;
#else
    return CpuIsa::Scalar;
#endif
  }();
  return kDetected;
}

const char* cpu_isa_name(CpuIsa isa) noexcept {
  switch (isa) {
    case CpuIsa::Scalar: return "scalar";
    case CpuIsa::Avx2: return "avx2+fma+f16c";
    case CpuIsa::Avx512: return "avx512+f16c";
    case CpuIsa::Neon: return "arm64-neon";
  }
  return "unknown";
}

void decode_q4_0_row_simd(const std::uint8_t* row_bytes, int hidden_size, float* out_f32, CpuIsa isa) {
  if (!row_bytes || !out_f32) throw std::invalid_argument("decode_q4_0_row_simd: null pointer");
  if (hidden_size <= 0 || (hidden_size % 32) != 0) {
    throw std::invalid_argument("decode_q4_0_row_simd: hidden_size must be a positive multiple of 32");
  }
#if defined(__x86_64__) || defined(_M_X64)
  if (isa == CpuIsa::Avx512) {
    decode_q4_0_row_avx512(row_bytes, hidden_size, out_f32);
    return;
  }
  if (isa == CpuIsa::Avx2) {
    decode_q4_0_row_avx2(row_bytes, hidden_size, out_f32);
    return;
  }
#elif defined(__aarch64__) || defined(__ARM_NEON)
  if (isa == CpuIsa::Neon) {
    decode_q4_0_row_neon(row_bytes, hidden_size, out_f32);
    return;
  }
#else
  (void)isa;
#endif
  decode_q4_0_row_scalar_impl(row_bytes, hidden_size, out_f32);
}

void decode_q4_0_rows(const std::uint8_t* table_q4, int rows, int hidden_size, float* out_f32,
                      CpuThreadPool* pool, CpuIsa isa) {
  if (!table_q4 || !out_f32) throw std::invalid_argument("decode_q4_0_rows: null pointer");
  if (rows < 1 || hidden_size <= 0 || (hidden_size % 32) != 0) {
    throw std::invalid_argument("decode_q4_0_rows: invalid rows or hidden_size");
  }
  const std::size_t row_bytes =
      static_cast<std::size_t>(hidden_size / 2) + static_cast<std::size_t>(hidden_size / 32) * 2;
  auto work = [&](int r0, int r1) {
    for (int r = r0; r < r1; ++r) {
      if (r + 2 < r1) prefetch_read(table_q4 + static_cast<std::size_t>(r + 2) * row_bytes);
      decode_q4_0_row_simd(table_q4 + static_cast<std::size_t>(r) * row_bytes, hidden_size,
                           out_f32 + static_cast<std::size_t>(r) * static_cast<std::size_t>(hidden_size), isa);
    }
  };
  if (pool != nullptr && rows >= 8) {
    pool->parallel_for(rows, 8, work);
  } else {
    work(0, rows);
  }
}

void embed_q4_rmsnorm_cpu(const EmbedQ4RmsNormDesc& desc, const EmbedQ4RmsNormArgs& args,
                          CpuThreadPool* pool, CpuIsa isa) {
  const int M = desc.tokens;
  const int H = desc.hidden_size;
  const auto* tbl = static_cast<const std::uint8_t*>(args.table_q4);
  auto work = [&](int m0, int m1) {
    std::vector<float> decoded(static_cast<std::size_t>(H));
    for (int m = m0; m < m1; ++m) {
      if (m + 1 < m1) {
        const std::int64_t next_tok = args.token_ids[m + 1];
        if (next_tok >= 0 && next_tok < desc.vocab_size) {
          prefetch_read(tbl + static_cast<std::size_t>(next_tok) * desc.weight_row_bytes);
        }
      }
      const std::int64_t tok = args.token_ids[m];
      const std::uint8_t* row = tbl + static_cast<std::size_t>(tok) * desc.weight_row_bytes;
      decode_q4_0_row_simd(row, H, decoded.data(), isa);
      float ss = 0.0f;
      for (int j = 0; j < H; ++j) ss += decoded[j] * decoded[j];
      const float rinv = 1.0f / std::sqrt(ss / static_cast<float>(H) + desc.eps);
      float* out_row = args.out + static_cast<std::size_t>(m) * static_cast<std::size_t>(desc.out_row_stride);
      for (int j = 0; j < H; ++j) {
        out_row[j] = round_cpu_act(decoded[j] * rinv * args.norm_weight[j], desc.output_rounding);
      }
    }
  };
  if (pool != nullptr && M >= 4) {
    pool->parallel_for(M, 2, work);
  } else {
    work(0, M);
  }
}

void gemv_q4_0_f32(const std::uint8_t* w_q4, const float* x, float* y, int N, int K,
                   CpuThreadPool* pool, CpuIsa isa) {
  validate_gemv_dims(w_q4, x, y, N, K);
  auto work = [&](int r0, int r1) {
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == CpuIsa::Avx512) {
      gemv_q4_0_f32_avx512_range(w_q4, x, y, r0, r1, K);
      return;
    }
    if (isa == CpuIsa::Avx2) {
      gemv_q4_0_f32_avx2_range(w_q4, x, y, r0, r1, K);
      return;
    }
#elif defined(__aarch64__) || defined(__ARM_NEON)
    if (isa == CpuIsa::Neon) {
      gemv_q4_0_f32_neon_range(w_q4, x, y, r0, r1, K);
      return;
    }
#else
    (void)isa;
#endif
    gemv_q4_0_f32_scalar_range(w_q4, x, y, r0, r1, K);
  };
  if (pool != nullptr && N >= 32) {
    pool->parallel_for(N, 16, work);
  } else {
    work(0, N);
  }
}

void quantize_f32_to_q8_0(const float* x, BlockQ8_0* out_q8, int K) {
  if (!x || !out_q8) throw std::invalid_argument("quantize_f32_to_q8_0: null pointer");
  if (K <= 0 || (K % 32) != 0) throw std::invalid_argument("quantize_f32_to_q8_0: invalid K");
  const int blocks = K / 32;
  for (int b = 0; b < blocks; ++b) {
    const float* xp = x + std::size_t(b) * 32;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xp[j]));
    const float d_ideal = amax / 127.0f;
    const std::uint16_t d_bits = act::host_float_to_fp16_bits(d_ideal);
    const float d = fp16_to_f32_finite_scale(d_bits);
    const float id = (d > 0.0f) ? (1.0f / d) : 0.0f;
    out_q8[b].d_fp16 = d_bits;
    for (int j = 0; j < 32; ++j) {
      const long q = std::lround(xp[j] * id);
      out_q8[b].qs[j] = static_cast<std::int8_t>(std::clamp(q, -127L, 127L));
    }
  }
}

void gemv_q8_0_f32(const BlockQ8_0* w_q8, const float* x, float* y, int N, int K,
                   CpuThreadPool* pool, CpuIsa isa) {
  validate_gemv_dims(w_q8, x, y, N, K);
  auto work = [&](int r0, int r1) {
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == CpuIsa::Avx512) {
      gemv_q8_0_f32_avx512_range(w_q8, x, y, r0, r1, K);
      return;
    }
    if (isa == CpuIsa::Avx2) {
      gemv_q8_0_f32_avx2_range(w_q8, x, y, r0, r1, K);
      return;
    }
#elif defined(__aarch64__) || defined(__ARM_NEON)
    if (isa == CpuIsa::Neon) {
      gemv_q8_0_f32_neon_range(w_q8, x, y, r0, r1, K);
      return;
    }
#else
    (void)isa;
#endif
    gemv_q8_0_f32_scalar_range(w_q8, x, y, r0, r1, K);
  };
  if (pool != nullptr && N >= 32) {
    pool->parallel_for(N, 16, work);
  } else {
    work(0, N);
  }
}

void gemv_q8_0_q8_0(const BlockQ8_0* w_q8, const BlockQ8_0* x_q8, float* y, int N, int K,
                    CpuThreadPool* pool, CpuIsa isa) {
  validate_gemv_dims(w_q8, x_q8, y, N, K);
  auto work = [&](int r0, int r1) {
#if defined(__x86_64__) || defined(_M_X64)
    if (isa == CpuIsa::Avx512) {
      gemv_q8_0_q8_0_avx512_range(w_q8, x_q8, y, r0, r1, K);
      return;
    }
    if (isa == CpuIsa::Avx2) {
      gemv_q8_0_q8_0_avx2_range(w_q8, x_q8, y, r0, r1, K);
      return;
    }
#elif defined(__aarch64__) || defined(__ARM_NEON)
    if (isa == CpuIsa::Neon) {
      gemv_q8_0_q8_0_neon_range(w_q8, x_q8, y, r0, r1, K);
      return;
    }
#else
    (void)isa;
#endif
    gemv_q8_0_q8_0_scalar_range(w_q8, x_q8, y, r0, r1, K);
  };
  if (pool != nullptr && N >= 32) {
    pool->parallel_for(N, 16, work);
  } else {
    work(0, N);
  }
}

void fwht_128_cpu(float* v, CpuIsa isa) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
  if (isa == CpuIsa::Avx512) {
    fwht_128_avx512(v);
    return;
  }
  if (isa == CpuIsa::Avx2) {
    fwht_128_avx2(v);
    return;
  }
#elif defined(__aarch64__) || defined(__ARM_NEON)
  if (isa == CpuIsa::Neon) {
    fwht_128_neon(v);
    return;
  }
#else
  (void)isa;
#endif
  fwht_128_scalar(v);
}

void dequant_exl3_matrix_cpu(const std::uint8_t* codes, const std::uint16_t* rin_fp16,
                             const std::uint16_t* rout_fp16, int bits, int rows, int cols,
                             float* out_f32, CpuThreadPool* pool, CpuIsa /*isa*/) {
  validate_exl3_dims(codes, rin_fp16, rout_fp16, bits, rows, cols);
  if (!out_f32) throw std::invalid_argument("dequant_exl3_matrix_cpu: null out_f32");
  const int n_sb = rows / 128;
  auto work = [&](int sb0, int sb1) {
    for (int sb = sb0; sb < sb1; ++sb) {
      const int rb = sb * 128;
      if (bits == 2) {
        dequant_exl3_superblock_scalar<2>(codes, rin_fp16, rout_fp16, rb, rows, cols, out_f32);
      } else {
        dequant_exl3_superblock_scalar<3>(codes, rin_fp16, rout_fp16, rb, rows, cols, out_f32);
      }
    }
  };
  if (pool != nullptr && n_sb >= 2) {
    pool->parallel_for(n_sb, 1, work);
  } else {
    work(0, n_sb);
  }
}

void gemv_exl3_f32(const std::uint8_t* codes, const std::uint16_t* rin_fp16,
                   const std::uint16_t* rout_fp16, int bits, const float* x, float* y,
                   int rows, int cols, CpuThreadPool* pool, CpuIsa isa) {
  validate_exl3_dims(codes, rin_fp16, rout_fp16, bits, rows, cols);
  if (!x || !y) throw std::invalid_argument("gemv_exl3_f32: null x or y");
  std::vector<float> x_packed;
  prepare_exl3_input_packed(rin_fp16, x, cols, isa, x_packed);

  const int n_sb = rows / 128;
  auto work = [&](int sb0, int sb1) {
    for (int sb = sb0; sb < sb1; ++sb) {
      const int rb = sb * 128;
#if defined(__x86_64__) || defined(_M_X64)
      if (isa == CpuIsa::Avx512) {
        if (bits == 2) gemv_exl3_superblock_avx512<2>(codes, rout_fp16, x_packed.data(), y, rb, cols);
        else gemv_exl3_superblock_avx512<3>(codes, rout_fp16, x_packed.data(), y, rb, cols);
        continue;
      }
      if (isa == CpuIsa::Avx2) {
        if (bits == 2) gemv_exl3_superblock_avx2<2>(codes, rout_fp16, x_packed.data(), y, rb, cols);
        else gemv_exl3_superblock_avx2<3>(codes, rout_fp16, x_packed.data(), y, rb, cols);
        continue;
      }
#endif
      if (bits == 2) gemv_exl3_superblock_scalar<2>(codes, rout_fp16, x_packed.data(), y, rb, cols);
      else gemv_exl3_superblock_scalar<3>(codes, rout_fp16, x_packed.data(), y, rb, cols);
    }
  };
  if (pool != nullptr && n_sb >= 2) {
    pool->parallel_for(n_sb, 1, work);
  } else {
    work(0, n_sb);
  }
}

void moe_expert_mlp_exl3_cpu(const std::uint8_t* w13_codes, const std::uint16_t* w13_rin_fp16,
                             const std::uint16_t* w13_rout_fp16, int w13_bits,
                             const std::uint8_t* w2_codes, const std::uint16_t* w2_rin_fp16,
                             const std::uint16_t* w2_rout_fp16, int w2_bits, const float* x,
                             float* out, int hidden_size, int inter_size, CpuThreadPool* pool,
                             CpuIsa isa) {
  if (!x || !out) throw std::invalid_argument("moe_expert_mlp_exl3_cpu: null x or out");
  if (inter_size <= 0 || (inter_size % 128) != 0) {
    throw std::invalid_argument("moe_expert_mlp_exl3_cpu: inter_size must be positive multiple of 128");
  }
  std::vector<float> gate_up(static_cast<std::size_t>(2 * inter_size));
  gemv_exl3_f32(w13_codes, w13_rin_fp16, w13_rout_fp16, w13_bits, x, gate_up.data(),
                2 * inter_size, hidden_size, pool, isa);
  std::vector<float> act_h(static_cast<std::size_t>(inter_size));
  for (int i = 0; i < inter_size; ++i) {
    const float g = gate_up[static_cast<std::size_t>(i)];
    const float u = gate_up[static_cast<std::size_t>(inter_size + i)];
    const float silu_g = g / (1.0f + std::exp(-g));
    act_h[static_cast<std::size_t>(i)] = silu_g * u;
  }
  gemv_exl3_f32(w2_codes, w2_rin_fp16, w2_rout_fp16, w2_bits, act_h.data(), out,
                hidden_size, inter_size, pool, isa);
}

}  // namespace ingine::ops::cpu

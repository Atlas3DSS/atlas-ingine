// Pure C++20 scalar oracles, IEEE-754 half/BF16/FP8 bit conversions, Q4_0 packing/unpacking,
// and FP64 comparison helpers for ingine tests. Zero CUDA headers or runtime dependencies.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ingine::test {

inline constexpr int kSkip = 77;

#define INGINE_CHECK(cond)                                                                \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);       \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

// Returns the value of an environment variable or an empty string (and prints a skip line).
inline std::string env_or_skip(const char* name) {
  const char* v = std::getenv(name);
  if (!v || !*v) {
    std::printf("skip: %s is not set\n", name);
    return {};
  }
  return v;
}

// Deterministic splitmix64-based generator.
struct Rng {
  std::uint64_t state;
  explicit Rng(std::uint64_t seed) : state(seed) {}
  std::uint64_t next() {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  // Uniform in [0, 1).
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  double uniform(double lo, double hi) { return lo + (hi - lo) * uniform(); }
  double normal() {
    double u1 = uniform(), u2 = uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
  }
};

// ----------------------------------------------------------------------------- Scalar BF16 / FP16 / FP8 / E2M1

inline float bf16_bits_to_float(std::uint16_t bits) {
  const std::uint32_t w = std::uint32_t(bits) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

inline std::uint16_t float_to_bf16_bits(float f) {
  std::uint32_t w;
  std::memcpy(&w, &f, 4);
  if ((w & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((w >> 16) | 0x40u);
  return static_cast<std::uint16_t>((w + 0x7fffu + ((w >> 16) & 1u)) >> 16);
}

inline float round_bf16_scalar(float f) { return bf16_bits_to_float(float_to_bf16_bits(f)); }

inline float fp16_bits_to_float(std::uint16_t h) {
  const std::uint32_t s = std::uint32_t(h & 0x8000u) << 16;
  const std::uint32_t e = (h >> 10) & 0x1fu;
  const std::uint32_t m = h & 0x03ffu;
  std::uint32_t w;
  if (e == 0) {
    if (m == 0) {
      w = s;
    } else {
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

inline std::uint16_t float_to_fp16_bits(float f, bool clamp_finite = false) {
  constexpr float kFp16Max = 65504.0f;
  if (clamp_finite && std::isfinite(f)) {
    f = std::clamp(f, -kFp16Max, kFp16Max);
  }
  std::uint32_t w;
  std::memcpy(&w, &f, 4);
  const std::uint16_t sign = static_cast<std::uint16_t>((w >> 16) & 0x8000u);
  const std::uint32_t abs_w = w & 0x7fffffffu;
  if (abs_w > 0x7f800000u) return static_cast<std::uint16_t>(sign | 0x7e00u);  // NaN
  if (abs_w == 0x7f800000u) return static_cast<std::uint16_t>(sign | 0x7c00u); // Inf
  if (abs_w >= 0x477ff000u) return static_cast<std::uint16_t>(sign | (clamp_finite ? 0x7bffu : 0x7c00u));
  if (abs_w >= 0x477fe000u && clamp_finite) return static_cast<std::uint16_t>(sign | 0x7bffu);
  if (abs_w < 0x38800000u) {
    const int shift = 126 - int(abs_w >> 23);
    if (shift > 24) return sign;
    const std::uint32_t mant = (abs_w & 0x007fffffu) | 0x00800000u;
    const std::uint32_t half_bit = 1u << (shift - 1);
    const std::uint32_t rem = mant & ((1u << shift) - 1u);
    std::uint32_t rounded = mant >> shift;
    if (rem > half_bit || (rem == half_bit && (rounded & 1u))) ++rounded;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  const std::uint32_t exp = (abs_w >> 23) - 112u;
  const std::uint32_t mant = abs_w & 0x007fffffu;
  const std::uint32_t rem = mant & 0x1fffu;
  std::uint32_t h = (exp << 10) | (mant >> 13);
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
  return static_cast<std::uint16_t>(sign | h);
}

inline float round_fp16_sat_scalar(float f) {
  return fp16_bits_to_float(float_to_fp16_bits(f, /*clamp_finite=*/true));
}

inline std::uint16_t float_to_act_bits(float f, bool fp16_act) {
  return fp16_act ? float_to_fp16_bits(f, /*clamp_finite=*/true) : float_to_bf16_bits(f);
}

inline float act_bits_to_float(std::uint16_t bits, bool fp16_act) {
  return fp16_act ? fp16_bits_to_float(bits) : bf16_bits_to_float(bits);
}

inline float round_act_scalar(float f, bool fp16_act) {
  return fp16_act ? round_fp16_sat_scalar(f) : round_bf16_scalar(f);
}

inline constexpr float kE2M1[16] = {
    0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// E4M3FN code -> float.
inline float e4m3_to_float(std::uint8_t code) {
  const int sign = code >> 7, exp = (code >> 3) & 0xF, man = code & 7;
  float v;
  if (exp == 0xF && man == 7) return NAN;
  if (exp == 0) v = std::ldexp(float(man) / 8.0f, -6);
  else v = std::ldexp(1.0f + float(man) / 8.0f, exp - 7);
  return sign ? -v : v;
}

// float -> E4M3FN, nearest even, saturated to +-448.
inline std::uint8_t float_to_e4m3(float x) {
  static const std::vector<float> table = [] {
    std::vector<float> t(127);
    for (int c = 0; c < 127; ++c) t[c] = e4m3_to_float(static_cast<std::uint8_t>(c));
    return t;
  }();
  const std::uint8_t sign = std::signbit(x) ? 0x80 : 0;
  const float a = std::min(std::fabs(x), 448.f);
  const int hi = int(std::lower_bound(table.begin(), table.end(), a) - table.begin());
  int best = std::min(hi, 126);
  if (hi > 0 && hi <= 126) {
    const float dl = a - table[hi - 1], dh = table[hi] - a;
    if (dl < dh || (dl == dh && ((hi - 1) & 1) == 0)) best = hi - 1;
  }
  return static_cast<std::uint8_t>(sign | best);
}

inline std::vector<float> randn(Rng& r, std::size_t n, double scale) {
  std::vector<float> v(n);
  for (auto& x : v) x = float(r.normal() * scale);
  return v;
}

inline std::vector<float> randn_bf16(Rng& r, std::size_t n, double scale, bool fp16_act = false) {
  std::vector<float> v = randn(r, n, scale);
  for (auto& x : v) x = round_act_scalar(x, fp16_act);
  return v;
}

// ----------------------------------------------------------------------------- Q4_0 packing, unpacking & RMSNorm oracle

// Q4_0 per-row KV cache layout: [8 heads x 64B nibbles | 8 heads x 4 FP16 scales] = 576B/row.
inline constexpr int kTestKvQ4NibBytes = 512;
inline constexpr int kTestKvQ4RowBytes = 576;

inline float q4_pick_max(float a, float b) {
  const float aa = std::fabs(a), ab = std::fabs(b);
  return (ab > aa || (ab == aa && b < a)) ? b : a;
}

// General Q4_0 pack for any [num_groups, group_size] row (group_size % 32 == 0):
// layout = [all nibble bytes (num_groups * group_size / 2) | all FP16 scales (num_groups * group_size / 32 * 2)].
inline void scalar_q4_0_pack_row(const float* in, int num_groups, int group_size, std::uint8_t* row_out,
                                 float* deq_out = nullptr) {
  const int blocks_per_group = group_size / 32;
  const int nib_bytes_per_group = group_size / 2;
  const int total_nib_bytes = num_groups * nib_bytes_per_group;
  const int row_bytes = total_nib_bytes + num_groups * blocks_per_group * 2;
  std::memset(row_out, 0, static_cast<std::size_t>(row_bytes));
  for (int g = 0; g < num_groups; ++g) {
    for (int blk = 0; blk < blocks_per_group; ++blk) {
      const float* x = in + g * group_size + blk * 32;
      float mx = x[0];
      for (int j = 1; j < 32; ++j) mx = q4_pick_max(mx, x[j]);
      const std::uint16_t dh_bits = float_to_fp16_bits(mx * (-0.125f), /*clamp_finite=*/false);
      const float d_raw = fp16_bits_to_float(dh_bits);
      const float d = std::isfinite(d_raw) ? d_raw : 0.f;
      const float id = d != 0.f ? (1.f / d) : 0.f;
      std::memcpy(row_out + total_nib_bytes + g * blocks_per_group * 2 + blk * 2, &dh_bits, 2);
      for (int j = 0; j < 32; ++j) {
        int qi = int(std::nearbyint(x[j] * id));
        qi = std::max(-8, std::min(7, qi));
        const std::uint8_t q = static_cast<std::uint8_t>(qi + 8);
        std::uint8_t& byte = row_out[g * nib_bytes_per_group + blk * 16 + (j >> 1)];
        if ((j & 1) == 0) byte = q;
        else byte |= static_cast<std::uint8_t>(q << 4);
        if (deq_out) deq_out[g * group_size + blk * 32 + j] = float(qi) * d;
      }
    }
  }
}

inline void scalar_q4_0_unpack_row(const std::uint8_t* row_in, int num_groups, int group_size, float* out) {
  const int blocks_per_group = group_size / 32;
  const int nib_bytes_per_group = group_size / 2;
  const int total_nib_bytes = num_groups * nib_bytes_per_group;
  for (int g = 0; g < num_groups; ++g) {
    for (int blk = 0; blk < blocks_per_group; ++blk) {
      std::uint16_t dh_bits = 0;
      std::memcpy(&dh_bits, row_in + total_nib_bytes + g * blocks_per_group * 2 + blk * 2, 2);
      const float d_raw = fp16_bits_to_float(dh_bits);
      const float d = std::isfinite(d_raw) ? d_raw : 0.f;
      for (int j = 0; j < 32; ++j) {
        const std::uint8_t byte = row_in[g * nib_bytes_per_group + blk * 16 + (j >> 1)];
        const std::uint8_t q = (j & 1) ? (byte >> 4) : (byte & 0x0f);
        out[g * group_size + blk * 32 + j] = float(int(q) - 8) * d;
      }
    }
  }
}

// Quantize one row [8, 128] of BF16/FP16-valued floats into 576 Q4_0 bytes.
inline void q4_0_pack_row(const float* in, std::uint8_t* row_out, float* deq_out = nullptr) {
  scalar_q4_0_pack_row(in, 8, 128, row_out, deq_out);
}

// Unpack one 576-byte Q4_0 row into 1024 floats [8, 128].
inline void q4_0_unpack_row(const std::uint8_t* row_in, float* out) {
  scalar_q4_0_unpack_row(row_in, 8, 128, out);
}

// Independent FP64 scalar reference for Q4_0 embedding lookup + RMSNorm:
// table is [vocab_size, row_bytes] packed with row_bytes = (hidden_size / 32) * 18.
inline void scalar_embed_q4_rmsnorm_fp64(const std::int64_t* token_ids, int tokens,
                                         const std::uint8_t* table_bytes, int vocab_size,
                                         const float* norm_weights, int hidden_size, double eps,
                                         bool fp16_act, std::vector<float>* decoded_lanes_out,
                                         std::vector<double>* unrounded_out,
                                         std::vector<double>* rounded_out) {
  const int row_bytes = (hidden_size / 32) * 18;
  const std::size_t total = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(hidden_size);
  if (decoded_lanes_out) decoded_lanes_out->assign(total, 0.f);
  if (unrounded_out) unrounded_out->assign(total, 0.0);
  if (rounded_out) rounded_out->assign(total, 0.0);

  std::vector<float> decoded(static_cast<std::size_t>(hidden_size));
  for (int m = 0; m < tokens; ++m) {
    const std::int64_t tok = token_ids[m];
    const bool valid = tok >= 0 && tok < static_cast<std::int64_t>(vocab_size);
    if (valid) {
      const std::uint8_t* row = table_bytes + static_cast<std::size_t>(tok) * static_cast<std::size_t>(row_bytes);
      scalar_q4_0_unpack_row(row, 1, hidden_size, decoded.data());
    } else {
      std::fill(decoded.begin(), decoded.end(), 0.f);
    }
    double ss = 0.0;
    for (int j = 0; j < hidden_size; ++j) {
      const double x = static_cast<double>(decoded[j]);
      ss += x * x;
    }
    const double rinv = 1.0 / std::sqrt(ss / static_cast<double>(hidden_size) + eps);
    const std::size_t base = static_cast<std::size_t>(m) * static_cast<std::size_t>(hidden_size);
    for (int j = 0; j < hidden_size; ++j) {
      if (decoded_lanes_out) (*decoded_lanes_out)[base + static_cast<std::size_t>(j)] = decoded[j];
      const double y = static_cast<double>(decoded[j]) * rinv * static_cast<double>(norm_weights[j]);
      if (unrounded_out) (*unrounded_out)[base + static_cast<std::size_t>(j)] = y;
      if (rounded_out) (*rounded_out)[base + static_cast<std::size_t>(j)] =
          static_cast<double>(round_act_scalar(static_cast<float>(y), fp16_act));
    }
  }
}

// ----------------------------------------------------------------------------- Comparison & oracle check

struct ErrorStats {
  double max_abs = 0, ref_rms = 0, max_rel_to_rms = 0;
  std::size_t worst = 0;
};

// max |got - ref| and rms(ref); passes when max_abs <= tol * rms(ref).
inline ErrorStats compare(const float* got, const double* ref, std::size_t n) {
  ErrorStats s;
  double sq = 0;
  for (std::size_t i = 0; i < n; ++i) sq += ref[i] * ref[i];
  s.ref_rms = std::sqrt(sq / double(n ? n : 1));
  for (std::size_t i = 0; i < n; ++i) {
    double d = std::fabs(double(got[i]) - ref[i]);
    if (!(d <= s.max_abs)) {
      s.max_abs = d;
      s.worst = i;
    }
  }
  s.max_rel_to_rms = s.ref_rms > 0 ? s.max_abs / s.ref_rms : s.max_abs;
  return s;
}

inline bool report(const char* what, const ErrorStats& s, double tol_rel_rms) {
  const bool ok = s.max_rel_to_rms <= tol_rel_rms;
  std::printf("%s %s: max_abs=%.3e rms(ref)=%.3e ratio=%.3e (tol %.1e) worst=%zu\n",
              ok ? "PASS" : "FAIL", what, s.max_abs, s.ref_rms, s.max_rel_to_rms, tol_rel_rms, s.worst);
  return ok;
}

// Values close to an oracle: |got - ref| <= rel * |ref| + abs for all, and at most `frac` of values off
// the oracle by more than its last-bit rounding (got != rounded ref).
struct OracleCheck {
  const char* what;
  double rel, abs, frac;
  bool operator()(const std::vector<float>& got, const std::vector<double>& ref,
                  const std::vector<double>* rounded = nullptr) const {
    double worst = 0;
    std::size_t bad = 0, off = 0, wi = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (!std::isfinite(got[i])) ++bad;
      const double d = std::fabs(double(got[i]) - ref[i]);
      const double tol = rel * std::fabs(ref[i]) + abs;
      if (!(d <= tol)) ++bad;
      if (d / (tol > 0 ? tol : 1) > worst) worst = d / (tol > 0 ? tol : 1), wi = i;
      if (rounded && double(got[i]) != (*rounded)[i]) ++off;
    }
    const double f = got.empty() ? 0 : double(off) / double(got.size());
    const bool ok = bad == 0 && f <= frac;
    std::printf("%s %s: n=%zu worst err/tol=%.3f at %zu (got %.6g ref %.6g) out-of-tol=%zu off-rounding=%.4f\n",
                ok ? "PASS" : "FAIL", what, got.size(), worst, wi, got.empty() ? 0.0 : double(got[wi]),
                got.empty() ? 0.0 : ref[wi], bad, f);
    return ok;
  }
};

}  // namespace ingine::test

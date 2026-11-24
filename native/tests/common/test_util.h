// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
#pragma once

// Shared helpers for ingine tests: skip codes, deterministic data, ActT/FP8 conversion,
// and tolerance comparison against FP64 oracles.
#include "core/act.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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

#define INGINE_TEST_CUDA(expr)                                                            \
  do {                                                                                    \
    cudaError_t e_ = (expr);                                                              \
    if (e_ != cudaSuccess) {                                                              \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr,           \
                   cudaGetErrorString(e_));                                               \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

#define INGINE_CHECK(cond)                                                                \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);       \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

// Returns true (after printing a skip line) when no CUDA device is usable.
inline bool skip_without_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("skip: no CUDA device\n");
    return true;
  }
  return false;
}

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
  uint64_t state;
  explicit Rng(uint64_t seed) : state(seed) {}
  uint64_t next() {
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
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

inline float bf16_to_float(ingine::ActT v) { return ingine::act::host_to_float(v); }

// Round-to-nearest-even FP32 -> ActT (BF16 on SM80+, FP16 clamped on INGINE_ACT_FP16).
inline ingine::ActT float_to_bf16(float f) { return ingine::act::host_from_float(f); }

inline float round_bf16(float f) { return ingine::act::host_round(f); }

// E4M3FN code -> float.
inline float e4m3_to_float(uint8_t code) {
  const int sign = code >> 7, exp = (code >> 3) & 0xF, man = code & 7;
  float v;
  if (exp == 0xF && man == 7) return NAN;
  if (exp == 0) v = std::ldexp(float(man) / 8.0f, -6);
  else v = std::ldexp(1.0f + float(man) / 8.0f, exp - 7);
  return sign ? -v : v;
}

// Q4_0 per-row KV cache layout: [8 heads x 64B nibbles | 8 heads x 4 FP16 scales] = 576B/row.
inline constexpr int kTestKvQ4NibBytes = 512;
inline constexpr int kTestKvQ4RowBytes = 576;

inline float q4_pick_max(float a, float b) {
  const float aa = std::fabs(a), ab = std::fabs(b);
  return (ab > aa || (ab == aa && b < a)) ? b : a;
}

// Quantize one row [8, 128] of BF16-valued floats into 576 Q4_0 bytes; optionally also write dequantized floats.
inline void q4_0_pack_row(const float* in, uint8_t* row_out, float* deq_out = nullptr) {
  std::memset(row_out, 0, kTestKvQ4RowBytes);
  for (int g = 0; g < 8; ++g) {
    for (int blk = 0; blk < 4; ++blk) {
      const float* x = in + g * 128 + blk * 32;
      float mx = x[0];
      for (int j = 1; j < 32; ++j) mx = q4_pick_max(mx, x[j]);
      const __half dh = __float2half_rn(mx * (-0.125f));
      const float d = __half2float(dh);
      const float id = d != 0.f ? (1.f / d) : 0.f;
      uint16_t dh_bits = 0;
      std::memcpy(&dh_bits, &dh, 2);
      std::memcpy(row_out + kTestKvQ4NibBytes + g * 8 + blk * 2, &dh_bits, 2);
      for (int j = 0; j < 32; ++j) {
        int qi = int(std::nearbyint(x[j] * id));
        qi = std::max(-8, std::min(7, qi));
        const uint8_t q = uint8_t(qi + 8);
        uint8_t& byte = row_out[g * 64 + blk * 16 + (j >> 1)];
        if ((j & 1) == 0) byte = q;
        else byte |= uint8_t(q << 4);
        if (deq_out) deq_out[g * 128 + blk * 32 + j] = float(qi) * d;
      }
    }
  }
}

// Unpack one 576-byte Q4_0 row into 1024 floats [8, 128].
inline void q4_0_unpack_row(const uint8_t* row_in, float* out) {
  for (int g = 0; g < 8; ++g) {
    for (int blk = 0; blk < 4; ++blk) {
      uint16_t dh_bits = 0;
      std::memcpy(&dh_bits, row_in + kTestKvQ4NibBytes + g * 8 + blk * 2, 2);
      __half dh;
      std::memcpy(&dh, &dh_bits, 2);
      const float d_raw = __half2float(dh);
      const float d = std::isfinite(d_raw) ? d_raw : 0.f;
      for (int j = 0; j < 32; ++j) {
        const uint8_t byte = row_in[g * 64 + blk * 16 + (j >> 1)];
        const uint8_t q = (j & 1) ? (byte >> 4) : (byte & 0x0f);
        out[g * 128 + blk * 32 + j] = float(int(q) - 8) * d;
      }
    }
  }
}

struct ErrorStats {
  double max_abs = 0, ref_rms = 0, max_rel_to_rms = 0;
  size_t worst = 0;
};

// max |got - ref| and rms(ref); passes when max_abs <= tol * rms(ref).
inline ErrorStats compare(const float* got, const double* ref, size_t n) {
  ErrorStats s;
  double sq = 0;
  for (size_t i = 0; i < n; ++i) sq += ref[i] * ref[i];
  s.ref_rms = std::sqrt(sq / double(n ? n : 1));
  for (size_t i = 0; i < n; ++i) {
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
              ok ? "PASS" : "FAIL", what, s.max_abs, s.ref_rms, s.max_rel_to_rms, tol_rel_rms,
              s.worst);
  return ok;
}

// RAII device buffer for tests. Construction and upload complete their default-stream writes
// before returning, so consumers on nonblocking streams never race initialization.
template <class T>
struct DeviceArray {
  T* ptr = nullptr;
  size_t count = 0;
  DeviceArray() = default;
  explicit DeviceArray(size_t n) : count(n) {
    INGINE_TEST_CUDA(cudaMalloc(&ptr, n * sizeof(T) + 16));
    INGINE_TEST_CUDA(cudaMemset(ptr, 0, n * sizeof(T) + 16));
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  }
  explicit DeviceArray(const std::vector<T>& h) : DeviceArray(h.size()) { upload(h); }
  ~DeviceArray() {
    if (ptr) cudaFree(ptr);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& o) noexcept : ptr(o.ptr), count(o.count) { o.ptr = nullptr; }
  void upload(const std::vector<T>& h) {
    INGINE_TEST_CUDA(cudaMemcpy(ptr, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  }
  std::vector<T> download() const {
    std::vector<T> h(count);
    INGINE_TEST_CUDA(cudaMemcpy(h.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  T* get() const { return ptr; }
};

}  // namespace ingine::test

#include "backends/qwen4exp/qwen4exp_cuda_ops.h"
#include "core/device.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ingine::backends::qwen4exp {
namespace {

__device__ __forceinline__ float dev_sigmoid(float x) {
  if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
  const float z = expf(x);
  return z / (1.0f + z);
}

__device__ __forceinline__ float dev_silu(float x) { return x * dev_sigmoid(x); }

__device__ __forceinline__ float dev_softplus(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return expf(x);
  return log1pf(expf(x));
}

__device__ __forceinline__ float dev_fp16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1fu;
  const std::uint32_t mant = h & 0x03ffu;
  if (exp == 0) {
    if (mant == 0) return __uint_as_float(sign);
    const float v = ldexpf(static_cast<float>(mant), -24);
    return (h & 0x8000u) ? -v : v;
  }
  if (exp == 31) return __uint_as_float(sign | 0x7f800000u | (mant << 13));
  return __uint_as_float(sign | ((exp + 112u) << 23) | (mant << 13));
}

__device__ __forceinline__ float dev_bf16_to_f32(std::uint16_t b) {
  return __uint_as_float(static_cast<std::uint32_t>(b) << 16);
}

__device__ __forceinline__ float dev_decode_e4m3fn(std::uint8_t bits) {
  const std::uint8_t sign = (bits >> 7) & 1u;
  const std::uint8_t exp = (bits >> 3) & 0x0fu;
  const std::uint8_t mant = bits & 0x07u;
  if (exp == 0x0fu && mant == 0x07u) return __uint_as_float(0x7fc00000u);
  const float val = (exp == 0) ? ldexpf(static_cast<float>(mant) * 0.125f, -6)
                               : ldexpf(1.0f + static_cast<float>(mant) * 0.125f, static_cast<int>(exp) - 7);
  return sign ? -val : val;
}

__device__ __forceinline__ std::uint16_t dev_le16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}

__device__ __forceinline__ std::uint32_t dev_le32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ void dev_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63u;
    m = q[j + 4] & 63u;
  } else {
    d = (q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float block_sum_256(float v) {
  __shared__ float smem[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  float total = 0.0f;
  if (warp == 0) {
    total = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
    total = warp_sum(total);
    if (lane == 0) smem[0] = total;
  }
  __syncthreads();
  return smem[0];
}

__constant__ float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
__constant__ float kAtlas2[4] = {-1.5f, -0.453125f, 0.453125f, 1.5f};
__constant__ float kAtlas3[8] = {-2.0f, -1.25f, -0.703125f, -0.2265625f, 0.2265625f, 0.703125f, 1.25f, 2.0f};

// ============================================================================
// Direct GGML quantized row-major linear kernels (4 warps / block, 1 warp / row)
// ============================================================================

__global__ void ggml_f32_linear_kernel(const float* __restrict__ w, const float* __restrict__ x, float* __restrict__ y,
                                       int m, int out_f, int in_f) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp;
  if (row >= m || n >= out_f) return;
  const float* xp = x + static_cast<std::size_t>(row) * in_f;
  const float* wp = w + static_cast<std::size_t>(n) * in_f;
  float acc = 0.0f;
  for (int k = lane; k < in_f; k += 32) acc += wp[k] * xp[k];
  acc = warp_sum(acc);
  if (lane == 0) y[static_cast<std::size_t>(row) * out_f + n] = acc;
}

__global__ void ggml_f16_bf16_linear_kernel(const std::uint16_t* __restrict__ w, const float* __restrict__ x,
                                            float* __restrict__ y, int m, int out_f, int in_f, bool is_bf16) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp;
  if (row >= m || n >= out_f) return;
  const float* xp = x + static_cast<std::size_t>(row) * in_f;
  const std::uint16_t* wp = w + static_cast<std::size_t>(n) * in_f;
  float acc = 0.0f;
  for (int k = lane; k < in_f; k += 32) {
    const float wv = is_bf16 ? dev_bf16_to_f32(wp[k]) : dev_fp16_to_f32(wp[k]);
    acc += wv * xp[k];
  }
  acc = warp_sum(acc);
  if (lane == 0) y[static_cast<std::size_t>(row) * out_f + n] = acc;
}

__global__ void ggml_q8_0_linear_kernel(const std::uint8_t* __restrict__ w, const float* __restrict__ x,
                                        float* __restrict__ y, int m, int out_f, int in_f) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp;
  if (row >= m || n >= out_f) return;
  const int nb = in_f >> 5;
  const float* xp = x + static_cast<std::size_t>(row) * in_f;
  const std::uint8_t* wp = w + static_cast<std::size_t>(n) * static_cast<std::size_t>(nb) * 34ull;
  float acc = 0.0f;
  for (int b = lane; b < nb; b += 32) {
    const std::uint8_t* blk = wp + static_cast<std::size_t>(b) * 34ull;
    const float d = dev_fp16_to_f32(dev_le16(blk));
    const auto* qs = reinterpret_cast<const std::int8_t*>(blk + 2);
    const float* xb = xp + (b << 5);
    float bsum = 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) bsum += static_cast<float>(qs[j]) * xb[j];
    acc += d * bsum;
  }
  acc = warp_sum(acc);
  if (lane == 0) y[static_cast<std::size_t>(row) * out_f + n] = acc;
}

__global__ void ggml_q4_0_linear_kernel(const std::uint8_t* __restrict__ w, const float* __restrict__ x,
                                        float* __restrict__ y, int m, int out_f, int in_f) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp;
  if (row >= m || n >= out_f) return;
  const int nb = in_f >> 5;
  const float* xp = x + static_cast<std::size_t>(row) * in_f;
  const std::uint8_t* wp = w + static_cast<std::size_t>(n) * static_cast<std::size_t>(nb) * 18ull;
  float acc = 0.0f;
  for (int b = lane; b < nb; b += 32) {
    const std::uint8_t* blk = wp + static_cast<std::size_t>(b) * 18ull;
    const float d = dev_fp16_to_f32(dev_le16(blk));
    const std::uint8_t* qs = blk + 2;
    const float* xb = xp + (b << 5);
    float bsum = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      bsum += static_cast<float>(static_cast<int>(qs[j] & 0x0fu) - 8) * xb[j];
      bsum += static_cast<float>(static_cast<int>(qs[j] >> 4) - 8) * xb[j + 16];
    }
    acc += d * bsum;
  }
  acc = warp_sum(acc);
  if (lane == 0) y[static_cast<std::size_t>(row) * out_f + n] = acc;
}

__global__ void ggml_q5_1_linear_kernel(const std::uint8_t* __restrict__ w, const float* __restrict__ x,
                                        float* __restrict__ y, int m, int out_f, int in_f) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp;
  if (row >= m || n >= out_f) return;
  const int nb = in_f >> 5;
  const float* xp = x + static_cast<std::size_t>(row) * in_f;
  const std::uint8_t* wp = w + static_cast<std::size_t>(n) * static_cast<std::size_t>(nb) * 24ull;
  float acc = 0.0f;
  for (int b = lane; b < nb; b += 32) {
    const std::uint8_t* blk = wp + static_cast<std::size_t>(b) * 24ull;
    const float d = dev_fp16_to_f32(dev_le16(blk));
    const float mv = dev_fp16_to_f32(dev_le16(blk + 2));
    const std::uint32_t qh = dev_le32(blk + 4);
    const std::uint8_t* qs = blk + 8;
    const float* xb = xp + (b << 5);
    float bsum = 0.0f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int xh0 = static_cast<int>((qh >> j) & 1u) << 4;
      const int xh1 = static_cast<int>((qh >> (j + 16)) & 1u) << 4;
      bsum += (static_cast<float>((qs[j] & 0x0fu) | xh0) * d + mv) * xb[j];
      bsum += (static_cast<float>((qs[j] >> 4) | xh1) * d + mv) * xb[j + 16];
    }
    acc += bsum;
  }
  acc = warp_sum(acc);
  if (lane == 0) y[static_cast<std::size_t>(row) * out_f + n] = acc;
}

__global__ void ggml_q4_k_linear_kernel(const std::uint8_t* __restrict__ w, const float* __restrict__ x,
                                        float* __restrict__ y, int m, int out_f, int in_f) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp;
  if (row >= m || n >= out_f) return;
  const int nb = in_f >> 8;
  const float* xp = x + static_cast<std::size_t>(row) * in_f;
  const std::uint8_t* wp = w + static_cast<std::size_t>(n) * static_cast<std::size_t>(nb) * 144ull;
  float acc = 0.0f;
  for (int b = lane; b < nb; b += 32) {
    const std::uint8_t* blk = wp + static_cast<std::size_t>(b) * 144ull;
    const float d = dev_fp16_to_f32(dev_le16(blk));
    const float min_v = dev_fp16_to_f32(dev_le16(blk + 2));
    const std::uint8_t* scales = blk + 4;
    const std::uint8_t* q = blk + 16;
    const float* xb = xp + (b << 8);
    int is = 0;
    for (int j = 0; j < 256; j += 64) {
      std::uint8_t sc = 0, mv = 0;
      dev_scale_min_k4(is + 0, scales, sc, mv);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = min_v * static_cast<float>(mv);
      dev_scale_min_k4(is + 1, scales, sc, mv);
      const float d2 = d * static_cast<float>(sc);
      const float m2 = min_v * static_cast<float>(mv);
      float s1 = 0.0f, sx1 = 0.0f, s2 = 0.0f, sx2 = 0.0f;
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        const float x1 = xb[j + l];
        const float x2 = xb[j + 32 + l];
        s1 += static_cast<float>(q[l] & 0x0fu) * x1;
        sx1 += x1;
        s2 += static_cast<float>(q[l] >> 4) * x2;
        sx2 += x2;
      }
      acc += (d1 * s1 - m1 * sx1) + (d2 * s2 - m2 * sx2);
      q += 32;
      is += 2;
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[static_cast<std::size_t>(row) * out_f + n] = acc;
}

__global__ void ggml_q5_k_linear_kernel(const std::uint8_t* __restrict__ w, const float* __restrict__ x,
                                        float* __restrict__ y, int m, int out_f, int in_f) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp;
  if (row >= m || n >= out_f) return;
  const int nb = in_f >> 8;
  const float* xp = x + static_cast<std::size_t>(row) * in_f;
  const std::uint8_t* wp = w + static_cast<std::size_t>(n) * static_cast<std::size_t>(nb) * 176ull;
  float acc = 0.0f;
  for (int b = lane; b < nb; b += 32) {
    const std::uint8_t* blk = wp + static_cast<std::size_t>(b) * 176ull;
    const float d = dev_fp16_to_f32(dev_le16(blk));
    const float min_v = dev_fp16_to_f32(dev_le16(blk + 2));
    const std::uint8_t* scales = blk + 4;
    const std::uint8_t* qh = blk + 16;
    const std::uint8_t* ql = blk + 48;
    const float* xb = xp + (b << 8);
    int is = 0;
    std::uint8_t u1 = 1u, u2 = 2u;
    for (int j = 0; j < 256; j += 64) {
      std::uint8_t sc = 0, mv = 0;
      dev_scale_min_k4(is + 0, scales, sc, mv);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = min_v * static_cast<float>(mv);
      dev_scale_min_k4(is + 1, scales, sc, mv);
      const float d2 = d * static_cast<float>(sc);
      const float m2 = min_v * static_cast<float>(mv);
      float s1 = 0.0f, sx1 = 0.0f, s2 = 0.0f, sx2 = 0.0f;
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        const float x1 = xb[j + l];
        const float x2 = xb[j + 32 + l];
        s1 += static_cast<float>((ql[l] & 0x0fu) + ((qh[l] & u1) ? 16 : 0)) * x1;
        sx1 += x1;
        s2 += static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) * x2;
        sx2 += x2;
      }
      acc += (d1 * s1 - m1 * sx1) + (d2 * s2 - m2 * sx2);
      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[static_cast<std::size_t>(row) * out_f + n] = acc;
}

__global__ void ggml_q6_k_linear_kernel(const std::uint8_t* __restrict__ w, const float* __restrict__ x,
                                        float* __restrict__ y, int m, int out_f, int in_f) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp;
  if (row >= m || n >= out_f) return;
  const int nb = in_f >> 8;
  const float* xp = x + static_cast<std::size_t>(row) * in_f;
  const std::uint8_t* wp = w + static_cast<std::size_t>(n) * static_cast<std::size_t>(nb) * 210ull;
  float acc = 0.0f;
  for (int b = lane; b < nb; b += 32) {
    const std::uint8_t* blk = wp + static_cast<std::size_t>(b) * 210ull;
    const std::uint8_t* ql = blk;
    const std::uint8_t* qh = blk + 128;
    const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
    const float d = dev_fp16_to_f32(dev_le16(blk + 208));
    const float* xb = xp + (b << 8);
    float bsum = 0.0f;
    for (int n128 = 0; n128 < 256; n128 += 128) {
#pragma unroll
      for (int l = 0; l < 32; ++l) {
        const int is = l >> 4;
        const int q1 = static_cast<int>((ql[l + 0] & 0x0fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
        const int q2 = static_cast<int>((ql[l + 32] & 0x0fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
        const int q3 = static_cast<int>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
        const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
        bsum += static_cast<float>(sc[is + 0]) * static_cast<float>(q1) * xb[l + 0];
        bsum += static_cast<float>(sc[is + 2]) * static_cast<float>(q2) * xb[l + 32];
        bsum += static_cast<float>(sc[is + 4]) * static_cast<float>(q3) * xb[l + 64];
        bsum += static_cast<float>(sc[is + 6]) * static_cast<float>(q4) * xb[l + 96];
      }
      xb += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
    acc += d * bsum;
  }
  acc = warp_sum(acc);
  if (lane == 0) y[static_cast<std::size_t>(row) * out_f + n] = acc;
}

// ============================================================================
// Native Elastic Quant linear kernels (NVFP4, ATLAS3, ATLAS2)
// ============================================================================

template <int FMT>
__global__ void elastic_linear_kernel(const std::uint8_t* __restrict__ codes,
                                      const std::uint8_t* __restrict__ scales_e4m3, const float* __restrict__ factor_ptr,
                                      float factor_scalar, const float* __restrict__ x, float* __restrict__ y, int m,
                                      int out_f, int in_f) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp;
  if (row >= m || n >= out_f) return;

  const float factor = factor_ptr ? *factor_ptr : factor_scalar;
  const int nb = in_f >> 4;
  const float* xp = x + static_cast<std::size_t>(row) * in_f;
  const std::uint8_t* sp = scales_e4m3 + static_cast<std::size_t>(n) * nb;

  float acc = 0.0f;
  if constexpr (FMT == 0) {  // NVFP4 (8 bytes / 16 elems)
    const std::uint8_t* cp = codes + static_cast<std::size_t>(n) * (in_f >> 1);
    for (int b = lane; b < nb; b += 32) {
      const float sc = dev_decode_e4m3fn(sp[b]) * factor;
      const std::uint8_t* cb = cp + (b << 3);
      const float* xb = xp + (b << 4);
      float bsum = 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const std::uint8_t pb = cb[i];
        bsum += kE2M1[pb & 0x0fu] * xb[2 * i] + kE2M1[pb >> 4] * xb[2 * i + 1];
      }
      acc += bsum * sc;
    }
  } else if constexpr (FMT == 1) {  // ATLAS3 (12 bytes / 32 elems: lo0(4B), lo1(4B), hi01(4B))
    const std::uint8_t* cp = codes + static_cast<std::size_t>(n) * ((static_cast<std::size_t>(in_f) * 3ull) >> 3);
    for (int b = lane; b < nb; b += 32) {
      const float sc = dev_decode_e4m3fn(sp[b]) * factor;
      const int pair = b >> 1;
      const int half = b & 1;
      const std::uint8_t* blk = cp + static_cast<std::size_t>(pair) * 12ull;
      const std::uint32_t lo = dev_le32(blk + half * 4);
      const std::uint16_t hi = dev_le16(blk + 8 + half * 2);
      const float* xb = xp + (b << 4);
      float bsum = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const std::uint32_t code = ((lo >> (2 * i)) & 3u) | (((hi >> i) & 1u) << 2);
        bsum += kAtlas3[code] * xb[i];
      }
      acc += bsum * sc;
    }
  } else {  // ATLAS2 (4 bytes / 16 elems)
    const std::uint8_t* cp = codes + static_cast<std::size_t>(n) * (in_f >> 2);
    for (int b = lane; b < nb; b += 32) {
      const float sc = dev_decode_e4m3fn(sp[b]) * factor;
      const std::uint32_t w = dev_le32(cp + (b << 2));
      const float* xb = xp + (b << 4);
      float bsum = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        bsum += kAtlas2[(w >> (2 * i)) & 3u] * xb[i];
      }
      acc += bsum * sc;
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) y[static_cast<std::size_t>(row) * out_f + n] = acc;
}

__global__ void fp8_e4m3_linear_kernel(const std::uint8_t* __restrict__ w, float weight_scale,
                                       const float* __restrict__ x, float* __restrict__ y, int m, int out_f, int in_f) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp;
  if (row >= m || n >= out_f) return;
  const std::uint8_t* wp = w + static_cast<std::size_t>(n) * in_f;
  const float* xp = x + static_cast<std::size_t>(row) * in_f;
  float acc = 0.0f;
  for (int i = lane; i < in_f; i += 32) {
    acc += dev_decode_e4m3fn(wp[i]) * xp[i];
  }
  acc = warp_sum(acc);
  if (lane == 0) y[static_cast<std::size_t>(row) * out_f + n] = acc * weight_scale;
}

// ============================================================================
// Hyper-Connections & PLE kernels
// ============================================================================

__global__ void hc_init_repeat_kernel(const float* __restrict__ emb, float* __restrict__ res_hc, int hc, int hidden) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < hc * hidden) res_hc[idx] = emb[idx % hidden];
}

__global__ void grouped_rmsnorm_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out,
                                       int group_dim, float eps) {
  const int g = blockIdx.x;
  const float* xg = x + static_cast<std::size_t>(g) * group_dim;
  const float* wg = w + static_cast<std::size_t>(g) * group_dim;
  float* og = out + static_cast<std::size_t>(g) * group_dim;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < group_dim; i += blockDim.x) ss += xg[i] * xg[i];
  const float total = block_sum_256(ss);
  const float inv = rsqrtf(total / static_cast<float>(group_dim) + eps);
  for (int i = threadIdx.x; i < group_dim; i += blockDim.x) og[i] = xg[i] * inv * wg[i];
}

__global__ void hc_silu_scale_kernel(float* __restrict__ lo, int n, float inv_hc) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) lo[i] = dev_silu(lo[i] * inv_hc);
}

__global__ void hc_gate_collapse_kernel(const float* __restrict__ xn, const float* __restrict__ gate,
                                        float* __restrict__ mixed, int hc, int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= hidden) return;
  float acc = 0.0f;
  for (int c = 0; c < hc; ++c) {
    const int idx = c * hidden + i;
    acc += xn[idx] * dev_sigmoid(gate[idx]);
  }
  mixed[i] = acc / static_cast<float>(hc);
}

__global__ void hc_combine_kernel(const float* __restrict__ res, const float* __restrict__ blk,
                                  const float* __restrict__ inject, float* __restrict__ out, int hc, int hidden) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= hc * hidden) return;
  const int c = idx / hidden;
  const int i = idx - c * hidden;
  const float w = 2.0f * dev_sigmoid(inject[c] / static_cast<float>(hc));
  out[idx] = res[idx] + blk[i] * w;
}

__global__ void ple_gate_value_kernel(const float* __restrict__ key, const float* __restrict__ query,
                                      const float* __restrict__ value, float* __restrict__ gated, int hidden) {
  const int c = blockIdx.x;
  const float* kc = key + static_cast<std::size_t>(c) * hidden;
  const float* qc = query + static_cast<std::size_t>(c) * hidden;
  float* gc = gated + static_cast<std::size_t>(c) * hidden;
  float dot = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dot += kc[i] * qc[i];
  const float s = block_sum_256(dot) * rsqrtf(static_cast<float>(hidden));
  const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
  const float sgn = (s > 0.0f) ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
  const float g = dev_sigmoid(sgn * mag);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) gc[i] = value[i] * g;
}

__global__ void ple_conv_residual_kernel(const float* __restrict__ hidden_in, const float* __restrict__ gated,
                                         const float* __restrict__ normed, const float* __restrict__ w_conv1d,
                                         float* __restrict__ conv_state, float* __restrict__ hidden_out, int hc_dim,
                                         int kern, int dil, int hist) {
  const int ch = blockIdx.x * blockDim.x + threadIdx.x;
  if (ch >= hc_dim) return;
  const float* wc = w_conv1d + static_cast<std::size_t>(ch) * kern;
  float acc = 0.0f;
  for (int k = 0; k < kern; ++k) {
    const int idx = hist - (kern - 1 - k) * dil;
    const float xval = (idx == hist) ? normed[ch] : conv_state[static_cast<std::size_t>(idx) * hc_dim + ch];
    acc += wc[k] * xval;
  }
  const float conv_val = dev_silu(acc);
  if (hist > 0) {
    for (int col = 0; col + 1 < hist; ++col) {
      conv_state[static_cast<std::size_t>(col) * hc_dim + ch] =
          conv_state[static_cast<std::size_t>(col + 1) * hc_dim + ch];
    }
    conv_state[static_cast<std::size_t>(hist - 1) * hc_dim + ch] = normed[ch];
  }
  hidden_out[ch] = hidden_in[ch] + gated[ch] + conv_val;
}

// ============================================================================
// Gated DeltaNet (sigmoid output gate) kernels
// ============================================================================

__global__ void gdn_conv1d_silu_kernel(const float* __restrict__ x_in, const float* __restrict__ w_conv,
                                       float* __restrict__ conv_state, float* __restrict__ out, int conv_dim,
                                       int kern) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;
  float* st = conv_state + static_cast<std::size_t>(c) * kern;
  const float* w = w_conv + static_cast<std::size_t>(c) * kern;
  for (int k = 0; k + 1 < kern; ++k) st[k] = st[k + 1];
  st[kern - 1] = x_in[c];
  float acc = 0.0f;
  for (int k = 0; k < kern; ++k) acc += st[k] * w[k];
  out[c] = dev_silu(acc);
}

__global__ void gdn_l2_norm_heads_kernel(float* __restrict__ x, int head_dim, float eps) {
  const int h = blockIdx.x;
  float* hp = x + static_cast<std::size_t>(h) * head_dim;
  float ss = 0.0f;
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) ss += hp[d] * hp[d];
  const float inv = 1.0f / fmaxf(sqrtf(block_sum_256(ss)), eps);
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) hp[d] *= inv;
}

// One block per v_head (128 threads for head_dim=128). Thread j owns column j of S[head_dim, head_dim].
__global__ void gdn_delta_sigmoid_kernel(const float* __restrict__ conv_out, const float* __restrict__ z_vdim,
                                         const float* __restrict__ alpha, const float* __restrict__ beta_raw,
                                         const float* __restrict__ dt_bias, const float* __restrict__ ssm_a_neg,
                                         const float* __restrict__ ssm_norm_w, float rms_eps, int num_k_heads,
                                         int num_v_heads, int head_dim, float* __restrict__ recurrent_state,
                                         float* __restrict__ out_vdim) {
  const int vh = blockIdx.x;
  const int j = threadIdx.x;
  const int k_dim_total = num_k_heads * head_dim;
  // GGUF qwen4exp tiles Q/K heads across V heads via ggml_repeat_4d (iq1 = iv1 % num_k_heads).
  const int kh = vh % num_k_heads;

  const float* qh = conv_out + static_cast<std::size_t>(kh) * head_dim;
  const float* khp = conv_out + static_cast<std::size_t>(k_dim_total) + static_cast<std::size_t>(kh) * head_dim;
  const float* vhp = conv_out + static_cast<std::size_t>(2 * k_dim_total) + static_cast<std::size_t>(vh) * head_dim;
  const float* zhp = z_vdim + static_cast<std::size_t>(vh) * head_dim;
  float* state = recurrent_state + static_cast<std::size_t>(vh) * head_dim * head_dim;

  const float decay = expf(dev_softplus(alpha[vh] + dt_bias[vh]) * ssm_a_neg[vh]);
  const float beta = dev_sigmoid(beta_raw[vh]);
  const float q_scale = rsqrtf(static_cast<float>(head_dim));

  extern __shared__ float smem[];
  float* sq = smem;                  // [head_dim]
  float* sk = smem + head_dim;       // [head_dim]
  float* sraw = smem + 2 * head_dim; // [head_dim]
  if (j < head_dim) {
    sq[j] = qh[j] * q_scale;
    sk[j] = khp[j];
  }
  __syncthreads();

  float raw_j = 0.0f;
  if (j < head_dim) {
    float kv_mem = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      const float s_dec = state[i * head_dim + j] * decay;
      state[i * head_dim + j] = s_dec;
      kv_mem += s_dec * sk[i];
    }
    const float delta = (vhp[j] - kv_mem) * beta;
    float o = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
      const float ns = state[i * head_dim + j] + sk[i] * delta;
      state[i * head_dim + j] = ns;
      o += ns * sq[i];
    }
    raw_j = o;
    sraw[j] = raw_j;
  }
  __syncthreads();

  // RMSNorm over head_dim + sigmoid(z) output gate
  float ss = (j < head_dim) ? (raw_j * raw_j) : 0.0f;
  const float total = block_sum_256(ss);
  const float inv_rms = rsqrtf(total / static_cast<float>(head_dim) + rms_eps);
  if (j < head_dim) {
    out_vdim[static_cast<std::size_t>(vh) * head_dim + j] =
        (sraw[j] * inv_rms * ssm_norm_w[j]) * dev_sigmoid(zhp[j]);
  }
}

// ============================================================================
// Full Attention + QSA Indexer kernels
// ============================================================================

__device__ __forceinline__ void apply_rope_neox_elem(float* vec, int rotary_dim, int pos, float rope_theta, int j) {
  const int half = rotary_dim >> 1;
  if (j >= half) return;
  const double exponent = -2.0 * static_cast<double>(j) / static_cast<double>(rotary_dim);
  const double angle = static_cast<double>(pos) * pow(static_cast<double>(rope_theta), exponent);
  const float c = static_cast<float>(cos(angle));
  const float s = static_cast<float>(sin(angle));
  const float x0 = vec[j];
  const float x1 = vec[j + half];
  vec[j] = x0 * c - x1 * s;
  vec[j + half] = x1 * c + x0 * s;
}

__global__ void qsa_append_k_kernel(const float* __restrict__ k_raw, float* __restrict__ idx_k_cache, int idx_dim,
                                    int pos) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d < idx_dim) idx_k_cache[static_cast<std::size_t>(pos) * idx_dim + d] = k_raw[d];
}

__global__ void qsa_score_blocks_kernel(const float* __restrict__ q_idx_raw, const float* __restrict__ idx_q_norm,
                                        const float* __restrict__ idx_k_norm, const float* __restrict__ idx_k_cache,
                                        float rms_eps, float rope_theta, int rotary_dim, int idx_heads, int idx_dim,
                                        int compress_ratio, int pos, int n_blocks, float* __restrict__ block_scores) {
  const int b = blockIdx.x;
  if (b >= n_blocks) return;
  extern __shared__ float smem[];
  float* k_rot = smem;  // [idx_dim]
  float* q_tmp = smem + idx_dim;  // [idx_dim]

  const int d = threadIdx.x;
  float kp = 0.0f;
  if (d < idx_dim) {
    for (int off = 0; off < compress_ratio; ++off) {
      kp += idx_k_cache[static_cast<std::size_t>(b * compress_ratio + off) * idx_dim + d];
    }
    kp /= static_cast<float>(compress_ratio);
  }
  const float kss = block_sum_256((d < idx_dim) ? kp * kp : 0.0f);
  const float kinv = rsqrtf(kss / static_cast<float>(idx_dim) + rms_eps);
  if (d < idx_dim) k_rot[d] = kp * kinv * idx_k_norm[d];
  __syncthreads();
  apply_rope_neox_elem(k_rot, rotary_dim, b * compress_ratio, rope_theta, d);
  __syncthreads();

  float bsum = 0.0f;
  for (int h = 0; h < idx_heads; ++h) {
    const float* qh = q_idx_raw + static_cast<std::size_t>(h) * idx_dim;
    const float qv = (d < idx_dim) ? qh[d] : 0.0f;
    const float qss = block_sum_256((d < idx_dim) ? qv * qv : 0.0f);
    const float qinv = rsqrtf(qss / static_cast<float>(idx_dim) + rms_eps);
    if (d < idx_dim) q_tmp[d] = qv * qinv * idx_q_norm[d];
    __syncthreads();
    apply_rope_neox_elem(q_tmp, rotary_dim, pos, rope_theta, d);
    __syncthreads();
    const float dot = block_sum_256((d < idx_dim) ? q_tmp[d] * k_rot[d] : 0.0f);
    if (dot > 0.0f) bsum += dot;
    __syncthreads();
  }
  if (d == 0) block_scores[b] = bsum;
}

__global__ void full_attn_prep_q_kernel(const float* __restrict__ q_with_gate, const float* __restrict__ q_norm_w,
                                        float rms_eps, float rope_theta, int head_dim, int rotary_dim, int pos,
                                        float* __restrict__ q_out, float* __restrict__ gate_out) {
  const int h = blockIdx.x;
  const int d = threadIdx.x;
  const float* src = q_with_gate + static_cast<std::size_t>(h) * (2 * head_dim);
  float* qh = q_out + static_cast<std::size_t>(h) * head_dim;
  float* gh = gate_out + static_cast<std::size_t>(h) * head_dim;
  const float qv = (d < head_dim) ? src[d] : 0.0f;
  if (d < head_dim) gh[d] = src[head_dim + d];
  const float ss = block_sum_256((d < head_dim) ? qv * qv : 0.0f);
  const float inv = rsqrtf(ss / static_cast<float>(head_dim) + rms_eps);
  if (d < head_dim) qh[d] = qv * inv * q_norm_w[d];
  __syncthreads();
  apply_rope_neox_elem(qh, rotary_dim, pos, rope_theta, d);
}

__global__ void full_attn_prep_kv_kernel(const float* __restrict__ k_raw, const float* __restrict__ v_raw,
                                         const float* __restrict__ k_norm_w, float rms_eps, float rope_theta,
                                         int num_kv_heads, int head_dim, int rotary_dim, int pos,
                                         float* __restrict__ k_cache, float* __restrict__ v_cache) {
  const int kvh = blockIdx.x;
  const int d = threadIdx.x;
  const std::size_t kv_stride = static_cast<std::size_t>(num_kv_heads) * head_dim;
  const float* ksrc = k_raw + static_cast<std::size_t>(kvh) * head_dim;
  const float* vsrc = v_raw + static_cast<std::size_t>(kvh) * head_dim;
  float* kdst = k_cache + static_cast<std::size_t>(pos) * kv_stride + static_cast<std::size_t>(kvh) * head_dim;
  float* vdst = v_cache + static_cast<std::size_t>(pos) * kv_stride + static_cast<std::size_t>(kvh) * head_dim;
  const float kv = (d < head_dim) ? ksrc[d] : 0.0f;
  if (d < head_dim) vdst[d] = vsrc[d];
  const float ss = block_sum_256((d < head_dim) ? kv * kv : 0.0f);
  const float inv = rsqrtf(ss / static_cast<float>(head_dim) + rms_eps);
  if (d < head_dim) kdst[d] = kv * inv * k_norm_w[d];
  __syncthreads();
  apply_rope_neox_elem(kdst, rotary_dim, pos, rope_theta, d);
}

// One block per Q head (256 threads for head_dim=256). Online-softmax gated GQA over selected positions!
__global__ void full_attn_gqa_decode_kernel(const float* __restrict__ q_all, const float* __restrict__ gate_all,
                                            const float* __restrict__ k_cache, const float* __restrict__ v_cache,
                                            const int32_t* __restrict__ sel_pos, int n_sel, int num_heads,
                                            int num_kv_heads, int head_dim, float* __restrict__ out_attn) {
  const int h = blockIdx.x;
  const int d = threadIdx.x;
  const int kvh = h / (num_heads / num_kv_heads);
  const std::size_t kv_stride = static_cast<std::size_t>(num_kv_heads) * head_dim;
  const float* qh = q_all + static_cast<std::size_t>(h) * head_dim;
  const float* gh = gate_all + static_cast<std::size_t>(h) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const float qd = (d < head_dim) ? qh[d] : 0.0f;

  float m_max = -1e30f;
  float l_sum = 0.0f;
  float o_acc = 0.0f;

  for (int idx = 0; idx < n_sel; ++idx) {
    const int p = sel_pos ? sel_pos[idx] : idx;
    const float* kp = k_cache + static_cast<std::size_t>(p) * kv_stride + static_cast<std::size_t>(kvh) * head_dim;
    const float* vp = v_cache + static_cast<std::size_t>(p) * kv_stride + static_cast<std::size_t>(kvh) * head_dim;
    const float dot = block_sum_256((d < head_dim) ? qd * kp[d] : 0.0f) * scale;
    const float new_m = fmaxf(m_max, dot);
    const float exp_old = expf(m_max - new_m);
    const float exp_cur = expf(dot - new_m);
    l_sum = l_sum * exp_old + exp_cur;
    if (d < head_dim) o_acc = o_acc * exp_old + exp_cur * vp[d];
    m_max = new_m;
  }
  if (d < head_dim) {
    out_attn[static_cast<std::size_t>(h) * head_dim + d] = (o_acc / fmaxf(l_sum, 1e-12f)) * dev_sigmoid(gh[d]);
  }
}

// ============================================================================
// MoE Router + SwiGLU + Shared Expert kernels
// ============================================================================

__global__ void route_top_k_softmax_kernel(const float* __restrict__ logits, int E, int top_k, float scale,
                                           int32_t* __restrict__ out_exp, float* __restrict__ out_w) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  float max_v = -1e30f;
  for (int i = 0; i < E; ++i) max_v = fmaxf(max_v, logits[i]);
  double sum_e = 0.0;
  for (int i = 0; i < E; ++i) sum_e += exp(static_cast<double>(logits[i] - max_v));

  int best_id[16];
  float best_val[16];
  for (int k = 0; k < top_k; ++k) {
    best_id[k] = -1;
    best_val[k] = -1e30f;
  }
  for (int i = 0; i < E; ++i) {
    const float v = logits[i];
    if (v > best_val[top_k - 1]) {
      int pos = top_k - 1;
      while (pos > 0 && v > best_val[pos - 1]) {
        best_val[pos] = best_val[pos - 1];
        best_id[pos] = best_id[pos - 1];
        --pos;
      }
      best_val[pos] = v;
      best_id[pos] = i;
    }
  }
  double top_sum = 0.0;
  double probs[16];
  for (int k = 0; k < top_k; ++k) {
    probs[k] = exp(static_cast<double>(best_val[k] - max_v)) / sum_e;
    top_sum += probs[k];
  }
  const double denom = top_sum > 1e-12 ? top_sum : 1e-12;
  for (int k = 0; k < top_k; ++k) {
    out_exp[k] = best_id[k];
    out_w[k] = static_cast<float>((probs[k] / denom) * static_cast<double>(scale));
  }
}

__global__ void swiglu_mul_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ out,
                                  int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = dev_silu(g[i]) * u[i];
}

__global__ void axpy_kernel(float* __restrict__ dst, const float* __restrict__ src, float w, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += w * src[i];
}

__global__ void shared_expert_gate_add_kernel(float* __restrict__ dst, const float* __restrict__ sh_out,
                                              const float* __restrict__ gate_w, const float* __restrict__ x,
                                              int hidden) {
  float dot = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dot += gate_w[i] * x[i];
  const float sg = dev_sigmoid(block_sum_256(dot));
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] += sg * sh_out[i];
}

}  // namespace

void ggml_linear_f32_cuda(std::uint32_t type_id, const void* w_raw, const float* x, float* y, int m, int out_features,
                          int in_features, cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features <= 0) return;
  const dim3 grid((out_features + 3) / 4, m);
  constexpr int kThreads = 128;
  switch (type_id) {
    case 0:
      ggml_f32_linear_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const float*>(w_raw), x, y, m, out_features,
                                                            in_features);
      break;
    case 1:
    case 30:
      ggml_f16_bf16_linear_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const std::uint16_t*>(w_raw), x, y, m,
                                                                 out_features, in_features, type_id == 30);
      break;
    case 2:
      ggml_q4_0_linear_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const std::uint8_t*>(w_raw), x, y, m,
                                                             out_features, in_features);
      break;
    case 7:
      ggml_q5_1_linear_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const std::uint8_t*>(w_raw), x, y, m,
                                                             out_features, in_features);
      break;
    case 8:
      ggml_q8_0_linear_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const std::uint8_t*>(w_raw), x, y, m,
                                                             out_features, in_features);
      break;
    case 12:
      ggml_q4_k_linear_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const std::uint8_t*>(w_raw), x, y, m,
                                                             out_features, in_features);
      break;
    case 13:
      ggml_q5_k_linear_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const std::uint8_t*>(w_raw), x, y, m,
                                                             out_features, in_features);
      break;
    case 14:
      ggml_q6_k_linear_kernel<<<grid, kThreads, 0, stream>>>(static_cast<const std::uint8_t*>(w_raw), x, y, m,
                                                             out_features, in_features);
      break;
    default:
      throw std::invalid_argument("ggml_linear_f32_cuda: unsupported GGML type_id " + std::to_string(type_id));
  }
  CUDA_CHECK(cudaGetLastError());
}

void elastic_linear_f32_cuda(ElasticWeightFormat fmt, const std::uint8_t* codes, const std::uint8_t* scales_e4m3,
                             const float* factor_ptr, float factor_scalar, const float* x, float* y, int m,
                             int out_features, int in_features, cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features <= 0) return;
  const dim3 grid((out_features + 3) / 4, m);
  constexpr int kThreads = 128;
  switch (fmt) {
    case ElasticWeightFormat::Nvfp4:
      elastic_linear_kernel<0><<<grid, kThreads, 0, stream>>>(codes, scales_e4m3, factor_ptr, factor_scalar, x, y, m,
                                                              out_features, in_features);
      break;
    case ElasticWeightFormat::Atlas3:
      elastic_linear_kernel<1><<<grid, kThreads, 0, stream>>>(codes, scales_e4m3, factor_ptr, factor_scalar, x, y, m,
                                                              out_features, in_features);
      break;
    case ElasticWeightFormat::Atlas2:
      elastic_linear_kernel<2><<<grid, kThreads, 0, stream>>>(codes, scales_e4m3, factor_ptr, factor_scalar, x, y, m,
                                                              out_features, in_features);
      break;
  }
  CUDA_CHECK(cudaGetLastError());
}

void fp8_linear_f32_cuda(const std::uint8_t* w_e4m3, float weight_scale, const float* x, float* y, int m,
                         int out_features, int in_features, cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features <= 0) return;
  const dim3 grid((out_features + 3) / 4, m);
  fp8_e4m3_linear_kernel<<<grid, 128, 0, stream>>>(w_e4m3, weight_scale, x, y, m, out_features, in_features);
  CUDA_CHECK(cudaGetLastError());
}

void hc_init_repeat_f32_cuda(const float* emb, float* res_hc, int hc, int hidden, cudaStream_t stream) {
  const int n = hc * hidden;
  hc_init_repeat_kernel<<<(n + 255) / 256, 256, 0, stream>>>(emb, res_hc, hc, hidden);
  CUDA_CHECK(cudaGetLastError());
}

void grouped_rmsnorm_f32_cuda(const float* x, const float* weight, float* out, int groups, int group_dim, float eps,
                              cudaStream_t stream) {
  grouped_rmsnorm_kernel<<<groups, 256, 0, stream>>>(x, weight, out, group_dim, eps);
  CUDA_CHECK(cudaGetLastError());
}

void hc_silu_scale_inplace_f32_cuda(float* lo, int n, int hc, cudaStream_t stream) {
  hc_silu_scale_kernel<<<(n + 255) / 256, 256, 0, stream>>>(lo, n, 1.0f / static_cast<float>(hc));
  CUDA_CHECK(cudaGetLastError());
}

void hc_gate_collapse_f32_cuda(const float* xn, const float* gate, float* mixed, int hc, int hidden,
                               cudaStream_t stream) {
  hc_gate_collapse_kernel<<<(hidden + 255) / 256, 256, 0, stream>>>(xn, gate, mixed, hc, hidden);
  CUDA_CHECK(cudaGetLastError());
}

void hc_combine_f32_cuda(const float* residual, const float* block_out, const float* inject, float* out, int hc,
                         int hidden, cudaStream_t stream) {
  const int n = hc * hidden;
  hc_combine_kernel<<<(n + 255) / 256, 256, 0, stream>>>(residual, block_out, inject, out, hc, hidden);
  CUDA_CHECK(cudaGetLastError());
}

void ple_gate_value_f32_cuda(const float* key, const float* query, const float* value, float* gated, int hc, int hidden,
                             cudaStream_t stream) {
  ple_gate_value_kernel<<<hc, 256, 0, stream>>>(key, query, value, gated, hidden);
  CUDA_CHECK(cudaGetLastError());
}

void ple_conv_residual_step_f32_cuda(const float* hidden_in, const float* gated, const float* normed,
                                     const float* w_conv1d, float* ple_conv_state, float* hidden_out, int hc_dim,
                                     int kern, int dil, cudaStream_t stream) {
  const int hist = (kern > 1 && dil > 0) ? (kern - 1) * dil : 0;
  ple_conv_residual_kernel<<<(hc_dim + 255) / 256, 256, 0, stream>>>(hidden_in, gated, normed, w_conv1d, ple_conv_state,
                                                                     hidden_out, hc_dim, kern, dil, hist);
  CUDA_CHECK(cudaGetLastError());
}

void gdn_step_sigmoid_f32_cuda(const float* qkv_mixed, const float* z_vdim, const float* alpha_vheads,
                               const float* beta_raw_vheads, const float* ssm_conv_weight, const float* ssm_dt_bias,
                               const float* ssm_a_neg, const float* ssm_norm_weight, float rms_eps, int conv_kernel,
                               int num_k_heads, int num_v_heads, int head_dim, float* conv_state,
                               float* recurrent_state, float* conv_tmp, float* out_vdim, cudaStream_t stream) {
  const int k_dim_total = num_k_heads * head_dim;
  const int v_dim_total = num_v_heads * head_dim;
  const int conv_dim = 2 * k_dim_total + v_dim_total;

  gdn_conv1d_silu_kernel<<<(conv_dim + 255) / 256, 256, 0, stream>>>(qkv_mixed, ssm_conv_weight, conv_state, conv_tmp,
                                                                     conv_dim, conv_kernel);
  gdn_l2_norm_heads_kernel<<<num_k_heads, 256, 0, stream>>>(conv_tmp, head_dim, rms_eps);
  gdn_l2_norm_heads_kernel<<<num_k_heads, 256, 0, stream>>>(conv_tmp + k_dim_total, head_dim, rms_eps);

  const int threads = head_dim <= 256 ? 256 : head_dim;
  const std::size_t smem = static_cast<std::size_t>(3 * head_dim) * sizeof(float);
  gdn_delta_sigmoid_kernel<<<num_v_heads, threads, smem, stream>>>(
      conv_tmp, z_vdim, alpha_vheads, beta_raw_vheads, ssm_dt_bias, ssm_a_neg, ssm_norm_weight, rms_eps, num_k_heads,
      num_v_heads, head_dim, recurrent_state, out_vdim);
  CUDA_CHECK(cudaGetLastError());
}

void qsa_indexer_append_and_score_f32_cuda(const float* q_idx_raw, const float* k_idx_raw, const float* idx_q_norm,
                                           const float* idx_k_norm, float rms_eps, float rope_theta, int rotary_dim,
                                           int idx_heads, int idx_dim, int compress_ratio, int pos, float* idx_k_cache,
                                           float* block_scores_out, cudaStream_t stream) {
  qsa_append_k_kernel<<<(idx_dim + 255) / 256, 256, 0, stream>>>(k_idx_raw, idx_k_cache, idx_dim, pos);
  const int n_blocks = (pos + 1) / compress_ratio;
  if (n_blocks > 0 && block_scores_out != nullptr) {
    const std::size_t smem = static_cast<std::size_t>(2 * idx_dim) * sizeof(float);
    qsa_score_blocks_kernel<<<n_blocks, 256, smem, stream>>>(q_idx_raw, idx_q_norm, idx_k_norm, idx_k_cache, rms_eps,
                                                             rope_theta, rotary_dim, idx_heads, idx_dim, compress_ratio,
                                                             pos, n_blocks, block_scores_out);
  }
  CUDA_CHECK(cudaGetLastError());
}

void full_attn_qsa_step_f32_cuda(const float* q_with_gate, const float* k_raw, const float* v_raw,
                                 const float* q_norm_w, const float* k_norm_w, float rms_eps, float rope_theta,
                                 int num_heads, int num_kv_heads, int head_dim, int rotary_dim, int pos,
                                 const int32_t* selected_positions, int num_selected, float* k_cache, float* v_cache,
                                 float* q_scratch, float* gate_scratch, float* out_attn, cudaStream_t stream) {
  full_attn_prep_q_kernel<<<num_heads, 256, 0, stream>>>(q_with_gate, q_norm_w, rms_eps, rope_theta, head_dim,
                                                         rotary_dim, pos, q_scratch, gate_scratch);
  full_attn_prep_kv_kernel<<<num_kv_heads, 256, 0, stream>>>(k_raw, v_raw, k_norm_w, rms_eps, rope_theta, num_kv_heads,
                                                             head_dim, rotary_dim, pos, k_cache, v_cache);
  const int n_sel = (selected_positions != nullptr && num_selected > 0) ? num_selected : (pos + 1);
  full_attn_gqa_decode_kernel<<<num_heads, 256, 0, stream>>>(q_scratch, gate_scratch, k_cache, v_cache,
                                                             selected_positions, n_sel, num_heads, num_kv_heads,
                                                             head_dim, out_attn);
  CUDA_CHECK(cudaGetLastError());
}

void route_top_k_softmax_f32_cuda(const float* router_logits, int expert_count, int top_k, float weights_scale,
                                  int32_t* out_experts, float* out_weights, cudaStream_t stream) {
  if (top_k <= 0 || top_k > 16 || expert_count < top_k) {
    throw std::invalid_argument("route_top_k_softmax_f32_cuda: invalid top_k or expert_count");
  }
  route_top_k_softmax_kernel<<<1, 1, 0, stream>>>(router_logits, expert_count, top_k, weights_scale, out_experts,
                                                  out_weights);
  CUDA_CHECK(cudaGetLastError());
}

void swiglu_mul_f32_cuda(const float* gate, const float* up, float* out, int n, cudaStream_t stream) {
  swiglu_mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
  CUDA_CHECK(cudaGetLastError());
}

void axpy_f32_cuda(float* dst, const float* src, float weight, int n, cudaStream_t stream) {
  axpy_kernel<<<(n + 255) / 256, 256, 0, stream>>>(dst, src, weight, n);
  CUDA_CHECK(cudaGetLastError());
}

void shared_expert_gate_add_f32_cuda(float* dst, const float* sh_out, const float* gate_w, const float* x, int hidden,
                                     cudaStream_t stream) {
  shared_expert_gate_add_kernel<<<1, 256, 0, stream>>>(dst, sh_out, gate_w, x, hidden);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::backends::qwen4exp

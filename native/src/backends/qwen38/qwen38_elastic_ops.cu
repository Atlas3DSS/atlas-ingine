#include "backends/qwen38/qwen38_elastic_ops.h"

#include "core/device.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ingine::backends::qwen38 {
namespace {

__device__ __constant__ float kE2M1Lut[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};
__device__ __constant__ float kAtlas2Lut[4] = {-1.5f, -0.453125f, 0.453125f, 1.5f};
__device__ __constant__ float kAtlas3Lut[8] = {
    -2.0f, -1.25f, -0.703125f, -0.2265625f, 0.2265625f, 0.703125f, 1.25f, 2.0f,
};

__device__ __forceinline__ float decode_e4m3fn_dev(std::uint8_t b) {
  const std::uint32_t s = (b >> 7) & 1u;
  const std::uint32_t e = (b >> 3) & 0xfu;
  const std::uint32_t m = b & 0x7u;
  float v = 0.0f;
  if (e == 0) {
    v = ldexpf(static_cast<float>(m), -9);
  } else if (e == 15 && m == 7) {
    return 0.0f;
  } else {
    v = ldexpf(static_cast<float>(8u + m), static_cast<int>(e) - 10);
  }
  return s ? -v : v;
}

__device__ __forceinline__ std::uint16_t load_le16_dev(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}

__device__ __forceinline__ std::uint32_t load_le32_dev(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ float warp_sum_dev(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

// 4 warps/block -> 4 output rows per block, row-invariant across M=1..256 (exact batch invariance!).
template <int BITS>
__global__ void qwen38_elastic_linear_kernel(const float* __restrict__ x, const std::uint8_t* __restrict__ codes,
                                             const std::uint8_t* __restrict__ scales_e4m3, float weight_scale_2, int m,
                                             int n_out, int k_in, float* __restrict__ out) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp;
  if (row >= m || n >= n_out) return;

  const int nb = k_in >> 4;
  const float* xp = x + static_cast<std::size_t>(row) * k_in;
  const std::uint8_t* sp = scales_e4m3 + static_cast<std::size_t>(n) * nb;

  float acc = 0.0f;
  if constexpr (BITS == 3) {
    const std::uint8_t* cp = codes + static_cast<std::size_t>(n) * ((static_cast<std::size_t>(k_in) * 3ull) >> 3);
    for (int b = lane; b < nb; b += 32) {
      const float sc = decode_e4m3fn_dev(sp[b]) * weight_scale_2;
      const int pair = b >> 1;
      const int half = b & 1;
      const std::uint8_t* blk = cp + static_cast<std::size_t>(pair) * 12ull;
      const std::uint32_t lo = load_le32_dev(blk + half * 4);
      const std::uint16_t hi = load_le16_dev(blk + 8 + half * 2);
      const float* xb = xp + (b << 4);
      float bsum = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const std::uint32_t code = ((lo >> (2 * i)) & 3u) | (((hi >> i) & 1u) << 2);
        bsum += kAtlas3Lut[code] * xb[i];
      }
      acc += bsum * sc;
    }
  } else {  // BITS == 2
    const std::uint8_t* cp = codes + static_cast<std::size_t>(n) * (k_in >> 2);
    for (int b = lane; b < nb; b += 32) {
      const float sc = decode_e4m3fn_dev(sp[b]) * weight_scale_2;
      const std::uint32_t w = load_le32_dev(cp + (b << 2));
      const float* xb = xp + (b << 4);
      float bsum = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        bsum += kAtlas2Lut[(w >> (2 * i)) & 3u] * xb[i];
      }
      acc += bsum * sc;
    }
  }
  acc = warp_sum_dev(acc);
  if (lane == 0) out[static_cast<std::size_t>(row) * n_out + n] = acc;
}

__global__ void embed_lookup_fp8_kernel(const std::uint8_t* __restrict__ table, float weight_scale,
                                        const std::int32_t* __restrict__ tids, int m, int hidden, int vocab,
                                        float* __restrict__ out) {
  const int row = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || d >= hidden) return;
  const int tid = tids[row];
  if (tid < 0 || tid >= vocab) {
    out[static_cast<std::size_t>(row) * hidden + d] = 0.0f;
    return;
  }
  out[static_cast<std::size_t>(row) * hidden + d] =
      decode_e4m3fn_dev(table[static_cast<std::size_t>(tid) * hidden + d]) * weight_scale;
}

__global__ void embed_lookup_nvfp4_kernel(const std::uint8_t* __restrict__ codes,
                                          const std::uint8_t* __restrict__ scales, float weight_scale_2,
                                          const std::int32_t* __restrict__ tids, int m, int hidden, int vocab,
                                          float* __restrict__ out) {
  const int row = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || d >= hidden) return;
  const int tid = tids[row];
  if (tid < 0 || tid >= vocab) {
    out[static_cast<std::size_t>(row) * hidden + d] = 0.0f;
    return;
  }
  const int b = d >> 4;
  const float sc = decode_e4m3fn_dev(scales[static_cast<std::size_t>(tid) * (hidden >> 4) + b]) * weight_scale_2;
  const std::uint8_t pb = codes[static_cast<std::size_t>(tid) * (hidden >> 1) + (d >> 1)];
  const std::uint8_t code = (d & 1) ? (pb >> 4) : (pb & 0x0fu);
  out[static_cast<std::size_t>(row) * hidden + d] = kE2M1Lut[code] * sc;
}

}  // namespace

void qwen38_cuda_atlas3_linear_f32_batched(const float* x, const std::uint8_t* codes, const std::uint8_t* block_scales,
                                           float weight_scale_2, int m, int n, int k, float* out, cudaStream_t stream) {
  if (k % 32 != 0) throw std::invalid_argument("qwen38_cuda_atlas3_linear_f32_batched: k must be multiple of 32");
  if (m <= 0 || n <= 0) return;
  const dim3 grid(static_cast<unsigned>((n + 3) / 4), static_cast<unsigned>(m));
  qwen38_elastic_linear_kernel<3><<<grid, 128, 0, stream>>>(x, codes, block_scales, weight_scale_2, m, n, k, out);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_atlas2_linear_f32_batched(const float* x, const std::uint8_t* codes, const std::uint8_t* block_scales,
                                           float weight_scale_2, int m, int n, int k, float* out, cudaStream_t stream) {
  if (k % 16 != 0) throw std::invalid_argument("qwen38_cuda_atlas2_linear_f32_batched: k must be multiple of 16");
  if (m <= 0 || n <= 0) return;
  const dim3 grid(static_cast<unsigned>((n + 3) / 4), static_cast<unsigned>(m));
  qwen38_elastic_linear_kernel<2><<<grid, 128, 0, stream>>>(x, codes, block_scales, weight_scale_2, m, n, k, out);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_embed_lookup_fp8_f32(const std::uint8_t* table_fp8, float weight_scale, const std::int32_t* token_ids,
                                      int m, int hidden_size, int vocab_size, float* out_f32, cudaStream_t stream) {
  if (m <= 0 || hidden_size <= 0) return;
  const dim3 grid(static_cast<unsigned>((hidden_size + 255) / 256), static_cast<unsigned>(m));
  embed_lookup_fp8_kernel<<<grid, 256, 0, stream>>>(table_fp8, weight_scale, token_ids, m, hidden_size, vocab_size,
                                                    out_f32);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_embed_lookup_nvfp4_f32(const std::uint8_t* codes, const std::uint8_t* block_scales,
                                        float weight_scale_2, const std::int32_t* token_ids, int m, int hidden_size,
                                        int vocab_size, float* out_f32, cudaStream_t stream) {
  if (hidden_size % 16 != 0) {
    throw std::invalid_argument("qwen38_cuda_embed_lookup_nvfp4_f32: hidden_size must be multiple of 16");
  }
  if (m <= 0 || hidden_size <= 0) return;
  const dim3 grid(static_cast<unsigned>((hidden_size + 255) / 256), static_cast<unsigned>(m));
  embed_lookup_nvfp4_kernel<<<grid, 256, 0, stream>>>(codes, block_scales, weight_scale_2, token_ids, m, hidden_size,
                                                      vocab_size, out_f32);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::backends::qwen38

// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Token selection kernels (see head.h). fngine's sampler keeps a top-20 candidate list; engine.Engine
// samples from the full top-k/top-p distribution, so this file keeps fngine's ordering key, argmax tie rule
// and splitmix counter RNG and replaces the candidate pipeline with exact radix selections over the row:
//   * top-k: the k-th largest z by a 4-pass MSB radix select (8-bit digits) over order-preserving keys;
//   * top-p: the boundary token by the same descent over the probability mass in 2^-40 fixed point
//     (integer sums, so the result does not depend on the summation order);
//   * the draw: Gumbel-max (argmax of z + G, G = -log(-log u)) over the kept tokens, an exact sample of the
//     renormalised distribution without a prefix scan.
// One CTA of 1024 threads per row; every pass streams the row from L2.
#include "kernels/head/head.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <math_constants.h>
#include <stdexcept>
#include <string>

namespace ingine::kernels::head {
namespace {

constexpr int kBlock = 1024;
constexpr int kWarps = kBlock / 32;
constexpr double kMassOne = 1099511627776.0;  // 2^40: fixed-point probability scale

// Order-preserving key of a finite float: larger float, larger key.
__device__ __forceinline__ std::uint32_t fkey(float x) {
  const std::uint32_t u = __float_as_uint(x);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__host__ __device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// 24 random bits for (seed, counter, row, index): a pure function of its inputs.
__device__ __forceinline__ std::uint32_t random_bits(unsigned long long seed, unsigned long long counter,
                                                     std::uint32_t row, std::uint32_t index) {
  unsigned long long k = splitmix64(seed ^ (counter * 0xD1B54A32D192ED03ull));
  k = splitmix64(k ^ ((static_cast<unsigned long long>(row) << 32) | index));
  return static_cast<std::uint32_t>(k >> 40);
}

// Gumbel noise from a uniform strictly inside (0, 1).
__device__ __forceinline__ float gumbel(unsigned long long seed, unsigned long long counter, std::uint32_t row,
                                        std::uint32_t index) {
  const float u = (static_cast<float>(random_bits(seed, counter, row, index)) + 0.5f) * (1.0f / 16777216.0f);
  return -logf(-logf(u));
}

__device__ __forceinline__ bool better(float value, int index, float best_value, int best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

struct Reduce {
  float f[kWarps];
  int i[kWarps];
  float bf;
  int bi;
};

// Block sum in a fixed order (deterministic). Contains two barriers.
__device__ float block_sum(float v, Reduce& r) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) r.f[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int w = 0; w < kWarps; ++w) s += r.f[w];
    r.bf = s;
  }
  __syncthreads();
  return r.bf;
}

__device__ float block_max(float v, Reduce& r) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  if ((threadIdx.x & 31) == 0) r.f[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x == 0) {
    float s = r.f[0];
    for (int w = 1; w < kWarps; ++w) s = fmaxf(s, r.f[w]);
    r.bf = s;
  }
  __syncthreads();
  return r.bf;
}

// (value, index) argmax, ties to the lower index; returns the index (INT_MAX if every value was -inf/NaN).
__device__ int block_argmax(float v, int i, Reduce& r) {
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, i, o);
    if (better(ov, oi, v, i)) v = ov, i = oi;
  }
  if ((threadIdx.x & 31) == 0) r.f[threadIdx.x >> 5] = v, r.i[threadIdx.x >> 5] = i;
  __syncthreads();
  if (threadIdx.x == 0) {
    float bv = r.f[0];
    int bi = r.i[0];
    for (int w = 1; w < kWarps; ++w)
      if (better(r.f[w], r.i[w], bv, bi)) bv = r.f[w], bi = r.i[w];
    r.bi = bi;
  }
  __syncthreads();
  return r.bi;
}

__global__ void __launch_bounds__(kBlock) argmax_kernel(const float* __restrict__ logits, int V, long long stride,
                                                       std::int32_t* __restrict__ out) {
  __shared__ Reduce red;
  const float* row = logits + blockIdx.x * stride;
  float bv = -CUDART_INF_F;
  int bi = INT_MAX;
  for (int v = threadIdx.x; v < V; v += kBlock) {
    const float x = row[v];
    if (better(x, v, bv, bi)) bv = x, bi = v;
  }
  const int best = block_argmax(bv, bi, red);
  if (threadIdx.x == 0) out[blockIdx.x] = best == INT_MAX ? 0 : best;
}

struct Select {
  std::uint32_t hist[256];
  unsigned long long mass[256];
  std::uint32_t prefix;
  int nocut;
};

// Key of the k-th largest (1-based) z = row * inv_t over the row.
__device__ std::uint32_t kth_key(const float* row, int V, float inv_t, int k, Select& s) {
  std::uint32_t prefix = 0, mask = 0;
  int remaining = k;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = threadIdx.x; b < 256; b += kBlock) s.hist[b] = 0;
    __syncthreads();
    for (int v = threadIdx.x; v < V; v += kBlock) {
      const std::uint32_t key = fkey(row[v] * inv_t);
      if ((key & mask) == prefix) atomicAdd(&s.hist[(key >> shift) & 255u], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int b = 255;
      for (; b > 0; --b) {
        if (static_cast<int>(s.hist[b]) >= remaining) break;
        remaining -= static_cast<int>(s.hist[b]);
      }
      s.prefix = prefix | (static_cast<std::uint32_t>(b) << shift);
    }
    __syncthreads();
    prefix = s.prefix;
    mask |= 0xffu << shift;
    // every thread has read s.prefix before thread 0 rewrites it (after the next two barriers)
  }
  return prefix;
}

// Fixed-point probability of a token (0 outside the top-k set).
__device__ __forceinline__ unsigned long long fixed_mass(float p) {
  return static_cast<unsigned long long>(static_cast<double>(p) * kMassOne);
}

// Key of the last token top-p keeps: the first one (descending) whose inclusive mass exceeds `limit`.
// Sets s.nocut when no token does (everything is kept).
__device__ std::uint32_t top_p_key(const float* row, int V, float inv_t, std::uint32_t kth, float zmax, float inv_sum,
                                   unsigned long long limit, Select& s) {
  std::uint32_t prefix = 0, mask = 0;
  unsigned long long above = 0;
  if (threadIdx.x == 0) s.nocut = 0;
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int b = threadIdx.x; b < 256; b += kBlock) s.mass[b] = 0;
    __syncthreads();
    for (int v = threadIdx.x; v < V; v += kBlock) {
      const float z = row[v] * inv_t;
      const std::uint32_t key = fkey(z);
      if (key >= kth && (key & mask) == prefix) {
        const unsigned long long q = fixed_mass(expf(z - zmax) * inv_sum);
        if (q) atomicAdd(&s.mass[(key >> shift) & 255u], q);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int b = 255;
      bool hit = false;
      for (; b >= 0; --b) {
        if (above + s.mass[b] > limit) {
          hit = true;
          break;
        }
        above += s.mass[b];
      }
      if (!hit) {
        s.nocut = 1;
        b = 0;
      }
      s.prefix = prefix | (static_cast<std::uint32_t>(b) << shift);
    }
    __syncthreads();
    if (s.nocut) return kth;  // uniform across the block
    prefix = s.prefix;
    mask |= 0xffu << shift;
  }
  return prefix;
}

// MODE 0: one Gumbel-max draw per row into out; MODE 1: the distribution into probs [rows, V].
template <int MODE>
__global__ void __launch_bounds__(kBlock) dist_kernel(const float* __restrict__ logits, int V, long long stride,
                                                     float inv_t, int top_k, float top_p, unsigned long long seed,
                                                     unsigned long long counter, std::int32_t* __restrict__ out,
                                                     float* __restrict__ probs) {
  __shared__ Reduce red;
  __shared__ Select sel;
  const int r = blockIdx.x;
  const float* row = logits + r * stride;

  const std::uint32_t kth = (top_k > 0 && top_k < V) ? kth_key(row, V, inv_t, top_k, sel) : 0u;

  float mx = -CUDART_INF_F;
  for (int v = threadIdx.x; v < V; v += kBlock) mx = fmaxf(mx, row[v] * inv_t);
  const float zmax = block_max(mx, red);

  float s = 0.f;
  for (int v = threadIdx.x; v < V; v += kBlock) {
    const float z = row[v] * inv_t;
    if (fkey(z) >= kth) s += expf(z - zmax);
  }
  const float sum = block_sum(s, red);
  const float inv_sum = 1.f / sum;

  std::uint32_t thr = kth;
  bool cut = false;
  if (top_p < 1.f) {
    const unsigned long long limit = static_cast<unsigned long long>(static_cast<double>(top_p) * kMassOne);
    thr = top_p_key(row, V, inv_t, kth, zmax, inv_sum, limit, sel);
    cut = !sel.nocut;
  }

  if constexpr (MODE == 0) {
    float bv = -CUDART_INF_F;
    int bi = INT_MAX;
    for (int v = threadIdx.x; v < V; v += kBlock) {
      const float z = row[v] * inv_t;
      if (fkey(z) >= thr) {
        const float score = z + gumbel(seed, counter, static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(v));
        if (better(score, v, bv, bi)) bv = score, bi = v;
      }
    }
    const int best = block_argmax(bv, bi, red);
    if (threadIdx.x == 0) out[r] = best == INT_MAX ? 0 : best;
  } else {
    float* pr = probs + static_cast<long long>(r) * V;
    if (cut) {
      float k = 0.f;
      for (int v = threadIdx.x; v < V; v += kBlock) {
        const float z = row[v] * inv_t;
        if (fkey(z) >= thr) k += expf(z - zmax) / sum;
      }
      const float kept = block_sum(k, red);
      for (int v = threadIdx.x; v < V; v += kBlock) {
        const float z = row[v] * inv_t;
        pr[v] = fkey(z) >= thr ? (expf(z - zmax) / sum) / kept : 0.f;
      }
    } else {
      for (int v = threadIdx.x; v < V; v += kBlock) {
        const float z = row[v] * inv_t;
        pr[v] = fkey(z) >= kth ? expf(z - zmax) / sum : 0.f;
      }
    }
  }
}

__global__ void __launch_bounds__(kBlock) multinomial_kernel(const float* __restrict__ probs, int V, long long stride,
                                                            unsigned long long seed, unsigned long long counter,
                                                            std::int32_t* __restrict__ out) {
  __shared__ Reduce red;
  const int r = blockIdx.x;
  const float* row = probs + r * stride;
  float bv = -CUDART_INF_F;
  int bi = INT_MAX;
  for (int v = threadIdx.x; v < V; v += kBlock) {
    const float p = row[v];
    if (p > 0.f) {
      const float score = logf(p) + gumbel(seed, counter, static_cast<std::uint32_t>(r), static_cast<std::uint32_t>(v));
      if (better(score, v, bv, bi)) bv = score, bi = v;
    }
  }
  const int best = block_argmax(bv, bi, red);
  if (threadIdx.x == 0) out[r] = best == INT_MAX ? 0 : best;
}

__global__ void uniform_kernel(int n, unsigned long long seed, unsigned long long counter, float* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = static_cast<float>(random_bits(seed, counter, 0xffffffffu, static_cast<std::uint32_t>(i))) *
                      (1.0f / 16777216.0f);
}

void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("head: ") + what);
}

void check_rows(const void* p, int rows, int V, std::int64_t stride) {
  require(p != nullptr && rows >= 0 && V > 0 && stride >= V, "bad logits/probs arguments");
}

}  // namespace

void argmax(const float* logits, int rows, int V, std::int64_t stride, std::int32_t* out, cudaStream_t stream) {
  check_rows(logits, rows, V, stride);
  if (rows == 0) return;
  argmax_kernel<<<rows, kBlock, 0, stream>>>(logits, V, stride, out);
  CUDA_CHECK(cudaGetLastError());
}

void sampling_probs(const float* logits, int rows, int V, std::int64_t stride, const SamplingParams& params,
                    float* probs, cudaStream_t stream) {
  check_rows(logits, rows, V, stride);
  require(params.temperature > 0.f, "sampling_probs needs temperature > 0");
  if (rows == 0) return;
  const float inv_t = 1.0f / params.temperature;
  dist_kernel<1><<<rows, kBlock, 0, stream>>>(logits, V, stride, inv_t, params.top_k, params.top_p, 0ull, 0ull,
                                              nullptr, probs);
  CUDA_CHECK(cudaGetLastError());
}

void sample(const float* logits, int rows, int V, std::int64_t stride, const SamplingParams& params, SamplerRng& rng,
            std::int32_t* out, cudaStream_t stream) {
  check_rows(logits, rows, V, stride);
  if (rows == 0) return;
  if (params.temperature <= 0.f) {
    argmax(logits, rows, V, stride, out, stream);
    return;
  }
  const float inv_t = 1.0f / params.temperature;
  dist_kernel<0><<<rows, kBlock, 0, stream>>>(logits, V, stride, inv_t, params.top_k, params.top_p, rng.seed,
                                              rng.counter++, out, nullptr);
  CUDA_CHECK(cudaGetLastError());
}

void sample_from_probs(const float* probs, int rows, int V, std::int64_t stride, SamplerRng& rng, std::int32_t* out,
                       cudaStream_t stream) {
  check_rows(probs, rows, V, stride);
  if (rows == 0) return;
  multinomial_kernel<<<rows, kBlock, 0, stream>>>(probs, V, stride, rng.seed, rng.counter++, out);
  CUDA_CHECK(cudaGetLastError());
}

void uniform(int n, SamplerRng& rng, float* out, cudaStream_t stream) {
  require(n >= 0 && (n == 0 || out != nullptr), "bad uniform arguments");
  if (n == 0) return;
  uniform_kernel<<<(n + 255) / 256, 256, 0, stream>>>(n, rng.seed, rng.counter++, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::head

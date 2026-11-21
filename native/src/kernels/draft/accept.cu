// Speculative acceptance (engine.Engine._accept): greedy prefix match and speculative rejection sampling.
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include "core/device.h"

#include <stdexcept>

namespace ingine::kernels::draft {
namespace {

__global__ void greedy_accept_kernel(const std::int32_t* __restrict__ target, const std::int32_t* __restrict__ drafts,
                                     int n, AcceptResult* __restrict__ out) {
  int acc = 0;
  while (acc < n && drafts[acc] == target[acc]) ++acc;
  out->accepted = acc;
  out->next = target[acc];
}

constexpr int Q_THREADS = 1024;
constexpr int R_THREADS = 256;
constexpr int R_PER = 4;  // elements per thread in the residual kernels

// q_r = softmax(draft_logits[r] / T): row max and sum (ws[2 r], ws[2 r + 1]); ws[2 GAMMA] = 0.
__global__ void __launch_bounds__(Q_THREADS) q_stats_kernel(const float* __restrict__ draft_logits, int V,
                                                            float temperature, float* __restrict__ ws) {
  __shared__ float scratch[33];
  const float* row = draft_logits + size_t(blockIdx.x) * V;
  float m = -INFINITY;
  for (int x = threadIdx.x; x < V; x += Q_THREADS) m = fmaxf(m, __fdiv_rn(row[x], temperature));
  m = block_max(m, scratch);
  float s = 0.f;
  for (int x = threadIdx.x; x < V; x += Q_THREADS) s += expf(__fsub_rn(__fdiv_rn(row[x], temperature), m));
  s = block_sum(s, scratch);
  if (threadIdx.x == 0) {
    ws[2 * blockIdx.x] = m;
    ws[2 * blockIdx.x + 1] = s;
    if (blockIdx.x == 0) ws[2 * GAMMA] = 0.f;
  }
}

__device__ __forceinline__ float q_of(const float* draft_logits, int V, int r, int x, float temperature,
                                      const float* ws) {
  const float s = __fdiv_rn(draft_logits[size_t(r) * V + x], temperature);
  return __fdiv_rn(expf(__fsub_rn(s, ws[2 * r])), ws[2 * r + 1]);
}

// accepted count (every CTA computes it), then the residual row and its sum
__global__ void __launch_bounds__(R_THREADS) residual_kernel(const float* __restrict__ p,
                                                             const float* __restrict__ draft_logits, int V,
                                                             const std::int32_t* __restrict__ drafts, int n,
                                                             float temperature, const float* __restrict__ u,
                                                             float* __restrict__ ws, float* __restrict__ resid,
                                                             AcceptResult* __restrict__ out) {
  __shared__ int sh_acc;
  __shared__ float scratch[33];
  if (threadIdx.x == 0) {
    int acc = 0;
    for (; acc < n; ++acc) {
      const int d = drafts[acc];
      const float qd = q_of(draft_logits, V, acc, d, temperature, ws);
      const float pd = p[size_t(acc) * V + d];
      if (!(__fmul_rn(u[acc], qd) < pd)) break;
    }
    sh_acc = acc;
    if (blockIdx.x == 0) out->accepted = acc;
  }
  __syncthreads();
  const int a = sh_acc;
  const float* pa = p + size_t(a) * V;
  float s = 0.f;
  const int x0 = blockIdx.x * R_THREADS * R_PER;
#pragma unroll
  for (int j = 0; j < R_PER; ++j) {
    const int x = x0 + j * R_THREADS + threadIdx.x;
    if (x >= V) break;
    const float w = a < n ? fmaxf(__fsub_rn(pa[x], q_of(draft_logits, V, a, x, temperature, ws)), 0.f) : pa[x];
    resid[x] = w;
    s += w;
  }
  s = block_sum(s, scratch);
  if (threadIdx.x == 0) atomicAdd(&ws[2 * GAMMA], s);
}

// all-zero residual: draw from p_a instead
__global__ void __launch_bounds__(R_THREADS) residual_fallback_kernel(const float* __restrict__ p, int V,
                                                                      const float* __restrict__ ws,
                                                                      float* __restrict__ resid,
                                                                      const AcceptResult* __restrict__ out) {
  if (ws[2 * GAMMA] > 0.f) return;
  const float* pa = p + size_t(out->accepted) * V;
  const int x0 = blockIdx.x * R_THREADS * R_PER;
#pragma unroll
  for (int j = 0; j < R_PER; ++j) {
    const int x = x0 + j * R_THREADS + threadIdx.x;
    if (x < V) resid[x] = pa[x];
  }
}

}  // namespace

void greedy_accept(const std::int32_t* target, const std::int32_t* drafts, int n, AcceptResult* out,
                   cudaStream_t stream) {
  if (n < 0 || n > GAMMA) throw std::invalid_argument("greedy_accept: n must be in 0..7");
  greedy_accept_kernel<<<1, 1, 0, stream>>>(target, drafts, n, out);
  CUDA_CHECK(cudaGetLastError());
}

void sampled_accept(const float* p, const float* draft_logits, int V, const std::int32_t* drafts, int n,
                    float temperature, const float* u, float* ws, float* resid, AcceptResult* out,
                    cudaStream_t stream) {
  if (n < 1 || n > GAMMA) throw std::invalid_argument("sampled_accept: n must be in 1..7");
  if (!(temperature > 0.f)) throw std::invalid_argument("sampled_accept: temperature must be positive");
  q_stats_kernel<<<n, Q_THREADS, 0, stream>>>(draft_logits, V, temperature, ws);
  CUDA_CHECK(cudaGetLastError());
  const int blocks = (V + R_THREADS * R_PER - 1) / (R_THREADS * R_PER);
  residual_kernel<<<blocks, R_THREADS, 0, stream>>>(p, draft_logits, V, drafts, n, temperature, u, ws, resid, out);
  CUDA_CHECK(cudaGetLastError());
  residual_fallback_kernel<<<blocks, R_THREADS, 0, stream>>>(p, V, ws, resid, out);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::draft

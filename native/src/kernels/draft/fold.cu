// Load-time fold of the Markov head's hidden_out into the unembedding (dspark.Drafter.__init__):
// fold_w = bf16(U @ hw), fold_b = bf16(U @ hb), computed in FP64 from the dequantized unembedding.
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include "core/device.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace ingine::kernels::draft {
namespace {

constexpr int FOLD_ROWS = 32;  // unembedding rows per CTA
constexpr int FOLD_K = 16;     // k tile
constexpr int FOLD_THREADS = RANK;
static_assert(H % FOLD_K == 0);

// weights.dequant: BF16 rows as is, FP8 rows -> bf16(float(q) * scale)
__device__ __forceinline__ double u_at(const void* u, const float* scale, int row, int k) {
  if (scale == nullptr) return double(bf(static_cast<const ActT*>(u)[size_t(row) * H + k]));
  const __nv_fp8_storage_t q = static_cast<const __nv_fp8_storage_t*>(u)[size_t(row) * H + k];
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
  const float f = __half2float(__half(hr));
  return double(bf_round(__fmul_rn(f, scale[row])));
}

// torch double -> bfloat16: through float (round to nearest), then to BF16 (round to nearest even)
__device__ __forceinline__ ActT to_bf16(double x) { return act::from_float(__double2float_rn(x)); }

__global__ void __launch_bounds__(FOLD_THREADS) fold_w_kernel(const void* __restrict__ u,
                                                              const float* __restrict__ scale, int rows,
                                                              const ActT* __restrict__ hw,
                                                              ActT* __restrict__ fold_w) {
  __shared__ double us[FOLD_ROWS][FOLD_K];
  __shared__ double hs[FOLD_K][RANK];
  const int r0 = blockIdx.x * FOLD_ROWS, t = threadIdx.x;
  double acc[FOLD_ROWS];
#pragma unroll
  for (int r = 0; r < FOLD_ROWS; ++r) acc[r] = 0.0;
  for (int k0 = 0; k0 < H; k0 += FOLD_K) {
    for (int idx = t; idx < FOLD_ROWS * FOLD_K; idx += FOLD_THREADS) {
      const int r = idx / FOLD_K, k = idx % FOLD_K;
      us[r][k] = r0 + r < rows ? u_at(u, scale, r0 + r, k0 + k) : 0.0;
    }
#pragma unroll
    for (int k = 0; k < FOLD_K; ++k) hs[k][t] = double(bf(hw[size_t(k0 + k) * RANK + t]));
    __syncthreads();
#pragma unroll
    for (int k = 0; k < FOLD_K; ++k) {
      const double h = hs[k][t];
#pragma unroll
      for (int r = 0; r < FOLD_ROWS; ++r) acc[r] = fma(us[r][k], h, acc[r]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < FOLD_ROWS; ++r)
    if (r0 + r < rows) fold_w[size_t(r0 + r) * RANK + t] = to_bf16(acc[r]);
}

// warp per row
__global__ void __launch_bounds__(256) fold_b_kernel(const void* __restrict__ u, const float* __restrict__ scale,
                                                     int rows, const ActT* __restrict__ hb,
                                                     ActT* __restrict__ fold_b) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (row >= rows) return;
  double acc = 0.0;
  for (int k = lane; k < H; k += 32) acc = fma(u_at(u, scale, row, k), double(bf(hb[k])), acc);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) fold_b[row] = to_bf16(acc);
}

__device__ __forceinline__ double u_at_shard(const HeadShards& hs, int row, int k) {
  const int s = (hs.n_shards > 1) ? (row / hs.rows_per_shard) : 0;
  const int lr = row - s * hs.rows_per_shard;
  const auto& m = hs.shard[s];
  if (m.blk_off != nullptr) {
    const int b = lr >> 4, rl = lr & 15;
    const std::int32_t so = m.blk_soff[b];
    const uint8_t* base = static_cast<const uint8_t*>(m.w) + m.blk_off[b];
    if (so <= -2) {
      const float d = reinterpret_cast<const float*>(m.q8_scale + (-2 - so))[rl * (H / 16) + (k >> 4)];
      return double(bf_round(float(int(static_cast<int8_t>(base[size_t(rl) * H + k]))) * d));
    } else if (so >= 0) {
      const __half d = __ushort_as_half(m.q8_scale[so + rl * (H / 32) + (k >> 5)]);
      const __half q = __int2half_rn(int(static_cast<int8_t>(base[size_t(rl) * H + k])));
      return double(bf_round(__half2float(__hmul(q, d))));
    } else {
      return double(bf(reinterpret_cast<const ActT*>(base)[size_t(rl) * H + k]));
    }
  }
  return u_at(m.w, m.scale, lr, k);
}

__global__ void __launch_bounds__(FOLD_THREADS) fold_w_shards_kernel(HeadShards hs, const ActT* __restrict__ hw,
                                                                     ActT* __restrict__ fold_w) {
  __shared__ double us[FOLD_ROWS][FOLD_K];
  __shared__ double hs_sm[FOLD_K][RANK];
  const int r0 = blockIdx.x * FOLD_ROWS, t = threadIdx.x, rows = hs.total_rows;
  double acc[FOLD_ROWS];
#pragma unroll
  for (int r = 0; r < FOLD_ROWS; ++r) acc[r] = 0.0;
  for (int k0 = 0; k0 < H; k0 += FOLD_K) {
    for (int idx = t; idx < FOLD_ROWS * FOLD_K; idx += FOLD_THREADS) {
      const int r = idx / FOLD_K, k = idx % FOLD_K;
      us[r][k] = r0 + r < rows ? u_at_shard(hs, r0 + r, k0 + k) : 0.0;
    }
#pragma unroll
    for (int k = 0; k < FOLD_K; ++k) hs_sm[k][t] = double(bf(hw[size_t(k0 + k) * RANK + t]));
    __syncthreads();
#pragma unroll
    for (int k = 0; k < FOLD_K; ++k) {
      const double h = hs_sm[k][t];
#pragma unroll
      for (int r = 0; r < FOLD_ROWS; ++r) acc[r] = fma(us[r][k], h, acc[r]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < FOLD_ROWS; ++r)
    if (r0 + r < rows) fold_w[size_t(r0 + r) * RANK + t] = to_bf16(acc[r]);
}

__global__ void __launch_bounds__(256) fold_b_shards_kernel(HeadShards hs, const ActT* __restrict__ hb,
                                                            ActT* __restrict__ fold_b) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (row >= hs.total_rows) return;
  double acc = 0.0;
  for (int k = lane; k < H; k += 32) acc = fma(u_at_shard(hs, row, k), double(bf(hb[k])), acc);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) fold_b[row] = to_bf16(acc);
}

}  // namespace

void fold_head(const void* u, const float* u_scale, int rows, const ActT* hw, const ActT* hb,
               ActT* fold_w, ActT* fold_b, cudaStream_t stream) {
  if (rows <= 0) return;
  fold_w_kernel<<<(rows + FOLD_ROWS - 1) / FOLD_ROWS, FOLD_THREADS, 0, stream>>>(u, u_scale, rows, hw, fold_w);
  CUDA_CHECK(cudaGetLastError());
  fold_b_kernel<<<(rows * 32 + 255) / 256, 256, 0, stream>>>(u, u_scale, rows, hb, fold_b);
  CUDA_CHECK(cudaGetLastError());
}

void fold_head_shards(const HeadShards& hs, const ActT* hw, const ActT* hb,
                      ActT* fold_w, ActT* fold_b, cudaStream_t stream) {
  const int rows = hs.total_rows;
  if (rows <= 0) return;
  if (hs.n_shards == 1 && !hs.blocked()) {
    fold_head(hs.shard[0].w, hs.shard[0].scale, rows, hw, hb, fold_w, fold_b, stream);
    return;
  }
  fold_w_shards_kernel<<<(rows + FOLD_ROWS - 1) / FOLD_ROWS, FOLD_THREADS, 0, stream>>>(hs, hw, fold_w);
  CUDA_CHECK(cudaGetLastError());
  fold_b_shards_kernel<<<(rows * 32 + 255) / 256, 256, 0, stream>>>(hs, hb, fold_b);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::draft

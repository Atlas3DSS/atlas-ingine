// DSpark Markov head (dspark.Drafter._draft after the base logits): per-step top-1024 candidates, the
// GRU-like cell, candidate logits with the folded hidden_out correction, greedy or Gumbel-max picks, and
// the draft outputs.
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include "core/device.h"

#include <stdexcept>

namespace ingine::kernels::draft {
namespace {

// ---- top-k candidates ----------------------------------------------------------------------------------
// One CTA per row. The values are BF16-representable, so the upper 16 bits of their order key decide the
// order: a two-level 8-bit radix select finds the 1024th key T16; every element above it is a candidate,
// and of the elements equal to it the lowest indices fill the rest. The candidates are then sorted
// (value descending, index ascending) with a bitonic sort.

constexpr int TOPK_THREADS = 1024;
static_assert(CANDIDATES == TOPK_THREADS);

__device__ __forceinline__ uint32_t key16(float v) { return float_key(v) >> 16; }

// Thread 0: the bin holding the k-th largest element, scanning `hist` from the top; returns the bin and
// sets *above to the count of elements in higher bins (plus `above0`).
__device__ int select_bin(const uint32_t* hist, int k, int above0, int* above) {
  int cum = above0;
  for (int b = 255; b >= 0; --b) {
    if (cum + int(hist[b]) >= k) {
      *above = cum;
      return b;
    }
    cum += hist[b];
  }
  *above = cum;
  return 0;
}

__global__ void __launch_bounds__(TOPK_THREADS) topk_kernel(const float* __restrict__ base, int V,
                                                            float* __restrict__ cand_val,
                                                            std::int32_t* __restrict__ cand_idx) {
  __shared__ uint32_t hist[256];
  __shared__ unsigned long long items[CANDIDATES];
  __shared__ int sh_bin, sh_above, sh_gt, sh_taken;
  __shared__ int warp_ties[TOPK_THREADS / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float* x = base + size_t(blockIdx.x) * V;

  if (tid < 256) hist[tid] = 0;
  __syncthreads();
  for (int i = tid; i < V; i += TOPK_THREADS) atomicAdd(&hist[key16(x[i]) >> 8], 1u);
  __syncthreads();
  if (tid == 0) sh_bin = select_bin(hist, CANDIDATES, 0, &sh_above);
  __syncthreads();
  const uint32_t b1 = sh_bin;
  const int above1 = sh_above;
  __syncthreads();
  if (tid < 256) hist[tid] = 0;
  __syncthreads();
  for (int i = tid; i < V; i += TOPK_THREADS) {
    const uint32_t k = key16(x[i]);
    if ((k >> 8) == b1) atomicAdd(&hist[k & 255], 1u);
  }
  __syncthreads();
  if (tid == 0) {
    sh_bin = select_bin(hist, CANDIDATES, above1, &sh_above);
    sh_gt = 0;
    sh_taken = 0;
  }
  __syncthreads();
  const uint32_t T = (b1 << 8) | uint32_t(sh_bin);
  const int n_gt = sh_above, need = CANDIDATES - n_gt;

  for (int i0 = 0; i0 < V; i0 += TOPK_THREADS) {
    const int i = i0 + tid;
    uint32_t k = 0;
    float v = 0.f;
    if (i < V) {
      v = x[i];
      k = key16(v);
    }
    const bool gt = i < V && k > T, tie = i < V && k == T;
    if (gt) items[atomicAdd(&sh_gt, 1)] = pack_argmax(v, uint32_t(i));
    const unsigned ballot = __ballot_sync(0xffffffffu, tie);
    if (lane == 0) warp_ties[warp] = __popc(ballot);
    __syncthreads();
    const int taken = sh_taken;
    if (tie) {
      int rank = taken + __popc(ballot & ((1u << lane) - 1u));
      for (int w = 0; w < warp; ++w) rank += warp_ties[w];
      if (rank < need) items[n_gt + rank] = pack_argmax(v, uint32_t(i));
    }
    __syncthreads();
    if (tid == 0) {
      int t = 0;
      for (int w = 0; w < TOPK_THREADS / 32; ++w) t += warp_ties[w];
      sh_taken = taken + t;
    }
    __syncthreads();
    if (sh_taken >= need && i0 + TOPK_THREADS < V) {
      // all ties taken: only elements above T remain to be found
      for (int j = i0 + TOPK_THREADS + tid; j < V; j += TOPK_THREADS) {
        const float vj = x[j];
        if (key16(vj) > T) items[atomicAdd(&sh_gt, 1)] = pack_argmax(vj, uint32_t(j));
      }
      break;
    }
  }
  __syncthreads();

  // bitonic sort, descending
  for (int k = 2; k <= CANDIDATES; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      const int ixj = tid ^ j;
      if (ixj > tid) {
        const unsigned long long a = items[tid], b = items[ixj];
        const bool desc = (tid & k) == 0;
        if (desc ? a < b : a > b) {
          items[tid] = b;
          items[ixj] = a;
        }
      }
      __syncthreads();
    }
  }
  const unsigned long long it = items[tid];
  const size_t at = size_t(blockIdx.x) * CANDIDATES + tid;
  cand_val[at] = key_float(static_cast<uint32_t>(it >> 32));
  cand_idx[at] = std::int32_t(unpack_argmax_index(it));
}

// ---- rnn projection ------------------------------------------------------------------------------------

constexpr int PRE_COLS = RNN_IN - 2 * RANK;  // 4608 = h | f_rnn
constexpr int PRE_CHUNKS = PRE_COLS / 8;     // 576 uint4 chunks
static_assert(PRE_CHUNKS % 32 == 0);

// warp per output row, 7 vectors
__global__ void __launch_bounds__(256) markov_pre_kernel(const __nv_bfloat16* __restrict__ rnn_w,
                                                         const __nv_bfloat16* __restrict__ h,
                                                         const __nv_bfloat16* __restrict__ f_rnn,
                                                         float* __restrict__ part) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (row >= RNN_OUT) return;
  const __nv_bfloat16* wr = rnn_w + size_t(row) * RNN_IN + 2 * RANK;
  float acc[GAMMA];
#pragma unroll
  for (int i = 0; i < GAMMA; ++i) acc[i] = 0.f;
  for (int c = lane; c < PRE_CHUNKS; c += 32) {
    float w[8];
    unpack8(*reinterpret_cast<const uint4*>(wr + 8 * c), w);
    const int col = 8 * c;
    if (col < H) {
#pragma unroll
      for (int i = 0; i < GAMMA; ++i) {
        float z[8];
        unpack8(*reinterpret_cast<const uint4*>(h + size_t(i) * H + col), z);
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i] = fmaf(w[e], z[e], acc[i]);
      }
    } else {
      float z[8];
      unpack8(*reinterpret_cast<const uint4*>(f_rnn + (col - H)), z);
      float s = 0.f;
#pragma unroll
      for (int e = 0; e < 8; ++e) s = fmaf(w[e], z[e], s);
#pragma unroll
      for (int i = 0; i < GAMMA; ++i) acc[i] += s;
    }
  }
#pragma unroll
  for (int i = 0; i < GAMMA; ++i) {
    const float t = warp_sum(acc[i]);
    if (lane == 0) part[i * RNN_OUT + row] = t;
  }
}

__device__ __forceinline__ int step_prev(int i, const unsigned long long* picks, const std::int32_t* cand_idx,
                                         const DraftParams* p) {
  if (i == 0) return p->anchor;
  return cand_idx[(i - 1) * CANDIDATES + int(unpack_argmax_index(picks[i - 1]))];
}

// warp per state element j (rows j, 256 + j, 512 + j of the projection)
__global__ void __launch_bounds__(256) markov_step_kernel(int i, const __nv_bfloat16* __restrict__ rnn_w,
                                                          const __nv_bfloat16* __restrict__ rnn_b,
                                                          const __nv_bfloat16* __restrict__ w1,
                                                          const float* __restrict__ part,
                                                          const unsigned long long* __restrict__ picks,
                                                          const std::int32_t* __restrict__ cand_idx,
                                                          const DraftParams* __restrict__ p,
                                                          __nv_bfloat16* __restrict__ state,
                                                          __nv_bfloat16* __restrict__ latent) {
  const int j = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (j >= RANK) return;
  const int prev = step_prev(i, picks, cand_idx, p);
  float z0[8], z1[8];
  unpack8(*reinterpret_cast<const uint4*>(state + size_t(i) * RANK + 8 * lane), z0);
  unpack8(*reinterpret_cast<const uint4*>(w1 + size_t(prev) * RANK + 8 * lane), z1);
  float raw[3];
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    const int row = g * RANK + j;
    const __nv_bfloat16* wr = rnn_w + size_t(row) * RNN_IN;
    float w0[8], wv1[8];
    unpack8(*reinterpret_cast<const uint4*>(wr + 8 * lane), w0);
    unpack8(*reinterpret_cast<const uint4*>(wr + RANK + 8 * lane), wv1);
    float acc = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc = fmaf(w0[e], z0[e], fmaf(wv1[e], z1[e], acc));
    acc = warp_sum(acc);
    // raw = bf16(bf16(z . w) + b)
    raw[g] = bf_round(__fadd_rn(bf_round(acc + part[i * RNN_OUT + row]), bf(rnn_b[row])));
  }
  if (lane == 0) {
    const float gate = bf_round(__fdiv_rn(1.f, __fadd_rn(1.f, expf(-raw[0]))));
    const float cell = bf_round(tanhf(raw[1]));
    const float s = bf(state[size_t(i) * RANK + j]);
    const float gs = bf_round(__fmul_rn(gate, s));
    const float om = bf_round(__fsub_rn(1.f, gate));
    const float oc = bf_round(__fmul_rn(om, cell));
    state[size_t(i + 1) * RANK + j] = __float2bfloat16_rn(__fadd_rn(gs, oc));
    latent[j] = __float2bfloat16_rn(tanhf(raw[2]));
  }
}

// warp per candidate
__global__ void __launch_bounds__(256) markov_logits_kernel(int i, const __nv_bfloat16* __restrict__ fold_w,
                                                            const __nv_bfloat16* __restrict__ fold_b,
                                                            const float* __restrict__ cand_val,
                                                            const std::int32_t* __restrict__ cand_idx,
                                                            const __nv_bfloat16* __restrict__ latent,
                                                            const DraftParams* __restrict__ p,
                                                            float* __restrict__ cand_logits,
                                                            unsigned long long* __restrict__ picks) {
  const int k = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (k >= CANDIDATES) return;
  const int at = i * CANDIDATES + k;
  const int c = cand_idx[at];
  float w[8], l[8];
  unpack8(*reinterpret_cast<const uint4*>(fold_w + size_t(c) * RANK + 8 * lane), w);
  unpack8(*reinterpret_cast<const uint4*>(latent + 8 * lane), l);
  float acc = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) acc = fmaf(w[e], l[e], acc);
  acc = warp_sum(acc);
  if (lane == 0) {
    const float bias = bf_round(__fadd_rn(bf(fold_b[c]), bf_round(acc)));
    const float t = bf_round(__fadd_rn(cand_val[at], bias));
    const float logit = bf_round(__fmul_rn(30.f, tanhf(__fdiv_rn(t, 30.f))));
    cand_logits[at] = logit;
    const float T = p->temperature;
    float score = logit;
    if (T > 0.f) {
      const float u = fmaxf(uniform01(p->seed, p->draws, uint32_t(i), uint32_t(k)), 1e-20f);
      score = __fsub_rn(__fdiv_rn(logit, fmaxf(T, 1e-6f)), logf(-logf(u)));
    }
    atomicMax(&picks[i], pack_argmax(score, uint32_t(k)));
  }
}

// CTA per draft row
__global__ void __launch_bounds__(CANDIDATES) markov_finish_kernel(const std::int32_t* __restrict__ cand_idx,
                                                                   const float* __restrict__ cand_logits,
                                                                   const unsigned long long* __restrict__ picks,
                                                                   std::int32_t* __restrict__ prev_cand,
                                                                   float* __restrict__ draft_logits, int V,
                                                                   std::int32_t* __restrict__ tokens,
                                                                   float* __restrict__ conf,
                                                                   DraftParams* __restrict__ p) {
  __shared__ float scratch[33];
  const int i = blockIdx.x, t = threadIdx.x;
  const int at = i * CANDIDATES + t;
  float* row = draft_logits + size_t(i) * V;
  row[prev_cand[at]] = -INFINITY;
  __syncthreads();
  const int c = cand_idx[at];
  const float l = cand_logits[at];
  row[c] = l;
  prev_cand[at] = c;
  const float m = block_max(l, scratch);
  const float s = block_sum(expf(l - m), scratch);
  if (t == 0) {
    conf[i] = 1.f / s;
    tokens[i] = cand_idx[i * CANDIDATES + int(unpack_argmax_index(picks[i]))];
    if (i == 0) p->draws += 1;
  }
}

void check_launch() { CUDA_CHECK(cudaGetLastError()); }

}  // namespace

void topk_candidates(const float* base, int rows, int V, float* cand_val, std::int32_t* cand_idx,
                     cudaStream_t stream) {
  if (V < CANDIDATES) throw std::invalid_argument("topk_candidates: V must be at least 1024");
  if (rows <= 0) return;
  topk_kernel<<<rows, TOPK_THREADS, 0, stream>>>(base, V, cand_val, cand_idx);
  check_launch();
}

void markov_pre(const __nv_bfloat16* rnn_w, const __nv_bfloat16* h, const __nv_bfloat16* f_rnn, float* part,
                cudaStream_t stream) {
  markov_pre_kernel<<<RNN_OUT * 32 / 256, 256, 0, stream>>>(rnn_w, h, f_rnn, part);
  check_launch();
}

void markov_step(int i, const __nv_bfloat16* rnn_w, const __nv_bfloat16* rnn_b, const __nv_bfloat16* w1,
                 const float* part, const unsigned long long* picks, const std::int32_t* cand_idx,
                 const DraftParams* p, __nv_bfloat16* state, __nv_bfloat16* latent, cudaStream_t stream) {
  if (i < 0 || i >= GAMMA) throw std::invalid_argument("markov_step: step out of range");
  markov_step_kernel<<<RANK * 32 / 256, 256, 0, stream>>>(i, rnn_w, rnn_b, w1, part, picks, cand_idx, p, state,
                                                          latent);
  check_launch();
}

void markov_logits(int i, const __nv_bfloat16* fold_w, const __nv_bfloat16* fold_b, const float* cand_val,
                   const std::int32_t* cand_idx, const __nv_bfloat16* latent, const DraftParams* p,
                   float* cand_logits, unsigned long long* picks, cudaStream_t stream) {
  if (i < 0 || i >= GAMMA) throw std::invalid_argument("markov_logits: step out of range");
  markov_logits_kernel<<<CANDIDATES * 32 / 256, 256, 0, stream>>>(i, fold_w, fold_b, cand_val, cand_idx, latent, p,
                                                                  cand_logits, picks);
  check_launch();
}

void markov_finish(const std::int32_t* cand_idx, const float* cand_logits, const unsigned long long* picks,
                   std::int32_t* prev_cand, float* draft_logits, int V, std::int32_t* tokens, float* conf,
                   DraftParams* p, cudaStream_t stream) {
  markov_finish_kernel<<<GAMMA, CANDIDATES, 0, stream>>>(cand_idx, cand_logits, picks, prev_cand, draft_logits, V,
                                                         tokens, conf, p);
  check_launch();
}

}  // namespace ingine::kernels::draft

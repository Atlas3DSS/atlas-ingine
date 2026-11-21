// DSpark Markov head (dspark.Drafter._draft after the base logits): per-step top-1024 candidates, the
// GRU-like cell, candidate logits with the folded hidden_out correction, greedy or Gumbel-max picks, and
// the draft outputs. PDL launches (see block.cu).
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include <stdexcept>

namespace ingine::kernels::draft {
namespace {

// ---- top-k candidates ----------------------------------------------------------------------------------
// The values are BF16-representable, so the upper 16 bits of their order key decide the order. A two-level
// 8-bit radix select over TOPK_SLICES CTAs per row finds the 1024th key T16; every element above it is a
// candidate, and of the elements equal to it the lowest indices fill the rest (ties in index order across
// slices: each slice knows how many ties the slices before it hold). The 1024 candidates are then sorted by
// (value descending, index ascending), which makes the result independent of the order they were gathered.
//   hist_hi: per-slice histograms of the high byte
//   hist_lo: the high byte b1 holding the 1024th key (every CTA sums the slice histograms), then per-slice
//            histograms of the low byte of the keys in bin b1
//   count:   T16, the count above it and the ties needed; per-slice counts of keys above T16 and equal to it
//   scatter: each slice writes its keys above T16 and its needed ties at their offsets
//   sort:    bitonic sort of the 1024 candidates per row

constexpr int TOPK_SLICES = 32;
constexpr int TOPK_THREADS = 256;
constexpr int SORT_THREADS = 1024;
static_assert(CANDIDATES == SORT_THREADS);

// Per-row workspace. The total histograms are accumulated with integer atomics (exact, order independent)
// and cleared again by the sort kernel, so the workspace must start zeroed and stays so between calls.
struct TopkRowWs {
  uint32_t total_hi[256];
  uint32_t total_lo[256];
  int count_gt[TOPK_SLICES];
  int count_tie[TOPK_SLICES];
  unsigned long long items[CANDIDATES];
};

__device__ __forceinline__ uint32_t key16(float v) { return float_key(v) >> 16; }

__device__ __forceinline__ void slice_range(int V, int s, int& b, int& e) {
  const int per = (V + TOPK_SLICES - 1) / TOPK_SLICES;
  b = min(V, s * per);
  e = min(V, b + per);
}

// Block-wide (TOPK_THREADS = 256 threads, one bin each) count of the elements in bins above this thread's
// bin: bins are scanned from the top (bin 255 first).
__device__ __forceinline__ int count_above(int c, int* warp_tot) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = TOPK_THREADS / 32;
  // reversed order: thread t's bin is 255 - (position in scan); scan over bins descending
  // inclusive suffix sum within the warp (lanes with higher bin index come first)
  int v = c;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_down_sync(0xffffffffu, v, o);
    if (lane + o < 32) v += u;
  }
  if (lane == 0) warp_tot[warp] = v;  // total of this warp's 32 bins
  __syncthreads();
  int above = v - c;  // bins above in the same warp
  for (int w = warp + 1; w < nw; ++w) above += warp_tot[w];
  __syncthreads();
  return above;
}

// The bin (= threadIdx.x) holding the k-th largest key of the histogram, with the count of keys in the bins
// above it (plus `above0`).
__device__ __forceinline__ void select_bin(const uint32_t* hist, int k, int above0, int* sh_bin, int* sh_above,
                                           int* warp_tot) {
  const int c = int(hist[threadIdx.x]);
  const int above = above0 + count_above(c, warp_tot);
  if (above < k && above + c >= k) {
    *sh_bin = threadIdx.x;
    *sh_above = above;
  }
  __syncthreads();
}

__global__ void __launch_bounds__(TOPK_THREADS) topk_hist_hi_kernel(const float* base, int V, TopkRowWs* ws) {
  __shared__ uint32_t hist[256];
  pdl_trigger();
  hist[threadIdx.x] = 0;
  pdl_wait();
  __syncthreads();
  const int s = blockIdx.x, row = blockIdx.y;
  const float* x = base + size_t(row) * V;
  int b, e;
  slice_range(V, s, b, e);
  for (int i = b + threadIdx.x; i < e; i += TOPK_THREADS) atomicAdd(&hist[key16(x[i]) >> 8], 1u);
  __syncthreads();
  if (hist[threadIdx.x] != 0) atomicAdd(&ws[row].total_hi[threadIdx.x], hist[threadIdx.x]);
}

__global__ void __launch_bounds__(TOPK_THREADS) topk_hist_lo_kernel(const float* base, int V, TopkRowWs* ws) {
  __shared__ uint32_t hist[256];
  __shared__ int sh_bin, sh_above, warp_tot[TOPK_THREADS / 32];
  pdl_trigger();
  hist[threadIdx.x] = 0;
  pdl_wait();
  const int s = blockIdx.x, row = blockIdx.y;
  TopkRowWs& w = ws[row];
  select_bin(w.total_hi, CANDIDATES, 0, &sh_bin, &sh_above, warp_tot);
  const uint32_t b1 = sh_bin;
  const float* x = base + size_t(row) * V;
  int b, e;
  slice_range(V, s, b, e);
  for (int i = b + threadIdx.x; i < e; i += TOPK_THREADS) {
    const uint32_t k = key16(x[i]);
    if ((k >> 8) == b1) atomicAdd(&hist[k & 255], 1u);
  }
  __syncthreads();
  if (hist[threadIdx.x] != 0) atomicAdd(&w.total_lo[threadIdx.x], hist[threadIdx.x]);
}

// T16, the count of keys above it and the ties needed (recomputed by every CTA from the histograms)
__device__ __forceinline__ void select_threshold(const TopkRowWs& w, uint32_t& T, int& n_gt, int* sh_bin,
                                                 int* sh_above, int* warp_tot) {
  select_bin(w.total_hi, CANDIDATES, 0, sh_bin, sh_above, warp_tot);
  const uint32_t b1 = *sh_bin;
  const int above1 = *sh_above;
  __syncthreads();
  select_bin(w.total_lo, CANDIDATES, above1, sh_bin, sh_above, warp_tot);
  T = (b1 << 8) | uint32_t(*sh_bin);
  n_gt = *sh_above;
}

__global__ void __launch_bounds__(TOPK_THREADS) topk_count_kernel(const float* base, int V, TopkRowWs* ws) {
  __shared__ int sh_bin, sh_above, warp_tot[TOPK_THREADS / 32];
  __shared__ int n_above, n_tie;
  pdl_trigger();
  if (threadIdx.x == 0) n_above = n_tie = 0;
  pdl_wait();
  const int s = blockIdx.x, row = blockIdx.y;
  TopkRowWs& w = ws[row];
  uint32_t T;
  int n_gt;
  select_threshold(w, T, n_gt, &sh_bin, &sh_above, warp_tot);
  const float* x = base + size_t(row) * V;
  int b, e;
  slice_range(V, s, b, e);
  int gt = 0, tie = 0;
  for (int i = b + threadIdx.x; i < e; i += TOPK_THREADS) {
    const uint32_t k = key16(x[i]);
    gt += k > T;
    tie += k == T;
  }
  atomicAdd(&n_above, gt);
  atomicAdd(&n_tie, tie);
  __syncthreads();
  if (threadIdx.x == 0) {
    w.count_gt[s] = n_above;
    w.count_tie[s] = n_tie;
  }
}

__global__ void __launch_bounds__(TOPK_THREADS) topk_scatter_kernel(const float* base, int V, TopkRowWs* ws) {
  __shared__ int sh_bin, sh_above, warp_tot[TOPK_THREADS / 32];
  __shared__ int sh_gt, warp_ties[TOPK_THREADS / 32];
  pdl_trigger();
  if (threadIdx.x == 0) sh_gt = 0;
  pdl_wait();
  const int s = blockIdx.x, row = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  TopkRowWs& w = ws[row];
  uint32_t T;
  int n_gt;
  select_threshold(w, T, n_gt, &sh_bin, &sh_above, warp_tot);
  const int need = CANDIDATES - n_gt;
  int gt_off = 0, tie_off = 0;
  for (int t = 0; t < s; ++t) {
    gt_off += w.count_gt[t];
    tie_off += w.count_tie[t];
  }
  const float* x = base + size_t(row) * V;
  int b, e;
  slice_range(V, s, b, e);
  const bool want_ties = tie_off < need && w.count_tie[s] > 0;
  for (int i0 = b; i0 < e; i0 += TOPK_THREADS) {
    const int i = i0 + threadIdx.x;
    float v = 0.f;
    uint32_t k = 0;
    if (i < e) {
      v = x[i];
      k = key16(v);
    }
    if (i < e && k > T) w.items[gt_off + atomicAdd(&sh_gt, 1)] = pack_argmax(v, uint32_t(i));
    if (want_ties) {  // block-uniform
      const bool tie = i < e && k == T;
      const unsigned ballot = __ballot_sync(0xffffffffu, tie);
      if (lane == 0) warp_ties[warp] = __popc(ballot);
      __syncthreads();
      int rank = tie_off + __popc(ballot & ((1u << lane) - 1u));
      int total = 0;
      for (int t = 0; t < TOPK_THREADS / 32; ++t) {
        if (t < warp) rank += warp_ties[t];
        total += warp_ties[t];
      }
      if (tie && rank < need) w.items[n_gt + rank] = pack_argmax(v, uint32_t(i));
      tie_off += total;
      __syncthreads();
    }
  }
}

__global__ void __launch_bounds__(SORT_THREADS) topk_sort_kernel(TopkRowWs* ws, float* cand_val,
                                                                 std::int32_t* cand_idx) {
  __shared__ unsigned long long items[CANDIDATES];
  pdl_trigger();
  pdl_wait();
  const int tid = threadIdx.x;
  items[tid] = ws[blockIdx.x].items[tid];
  if (tid < 256) {  // every reader of the totals completed (pdl_wait): leave them zeroed for the next call
    ws[blockIdx.x].total_hi[tid] = 0;
    ws[blockIdx.x].total_lo[tid] = 0;
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
constexpr int PRE_PER_LANE = PRE_CHUNKS / 32;  // 18
constexpr int PRE_THREADS = 128;
static_assert(PRE_CHUNKS % 32 == 0);

// warp per output row, 7 vectors; the row's weights are loaded before the PDL wait
__global__ void __launch_bounds__(PRE_THREADS) markov_pre_kernel(const __nv_bfloat16* __restrict__ rnn_w,
                                                                 const __nv_bfloat16* h, const __nv_bfloat16* f_rnn,
                                                                 float* part) {
  pdl_trigger();
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  const __nv_bfloat16* wr = rnn_w + size_t(row) * RNN_IN + 2 * RANK;
  uint4 wv[PRE_PER_LANE];
#pragma unroll
  for (int j = 0; j < PRE_PER_LANE; ++j) wv[j] = ld_weight(wr + 8 * (lane + 32 * j));
  pdl_wait();
  float acc[GAMMA];
#pragma unroll
  for (int i = 0; i < GAMMA; ++i) acc[i] = 0.f;
#pragma unroll
  for (int j = 0; j < PRE_PER_LANE; ++j) {
    const int col = 8 * (lane + 32 * j);
    float w[8];
    unpack8(wv[j], w);
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

// warp per state element j (rows j, 256 + j, 512 + j of the projection); weights loaded before the wait
__global__ void __launch_bounds__(256) markov_step_kernel(int i, const __nv_bfloat16* __restrict__ rnn_w,
                                                          const __nv_bfloat16* __restrict__ rnn_b,
                                                          const __nv_bfloat16* __restrict__ w1, const float* part,
                                                          const unsigned long long* picks,
                                                          const std::int32_t* cand_idx, const DraftParams* p,
                                                          __nv_bfloat16* state, __nv_bfloat16* latent) {
  pdl_trigger();
  const int j = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  uint4 wa[3], wb[3];
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    const __nv_bfloat16* wr = rnn_w + size_t(g * RANK + j) * RNN_IN;
    wa[g] = ld_weight(wr + 8 * lane);
    wb[g] = ld_weight(wr + RANK + 8 * lane);
  }
  pdl_wait();
  const int prev = step_prev(i, picks, cand_idx, p);
  float z0[8], z1[8];
  unpack8(*reinterpret_cast<const uint4*>(state + size_t(i) * RANK + 8 * lane), z0);
  unpack8(*reinterpret_cast<const uint4*>(w1 + size_t(prev) * RANK + 8 * lane), z1);
  float raw[3];
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    const int row = g * RANK + j;
    float w0[8], wv1[8];
    unpack8(wa[g], w0);
    unpack8(wb[g], wv1);
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
                                                            const float* cand_val, const std::int32_t* cand_idx,
                                                            const __nv_bfloat16* latent, const DraftParams* p,
                                                            float* cand_logits, unsigned long long* picks) {
  pdl_trigger();
  pdl_wait();
  const int k = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
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
__global__ void __launch_bounds__(CANDIDATES) markov_finish_kernel(const std::int32_t* cand_idx,
                                                                   const float* cand_logits,
                                                                   const unsigned long long* picks,
                                                                   std::int32_t* prev_cand, float* draft_logits,
                                                                   int V, std::int32_t* tokens, float* conf,
                                                                   float* qstats, DraftParams* p) {
  __shared__ float scratch[33];
  pdl_trigger();
  pdl_wait();
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
  const float T = p->temperature;
  float qm = 0.f, qs = 0.f;
  if (T > 0.f) {  // q = softmax(logits / T) over the candidates (exp(-inf) = 0 elsewhere)
    const float x = __fdiv_rn(l, T);
    qm = block_max(x, scratch);
    qs = block_sum(expf(__fsub_rn(x, qm)), scratch);
  }
  if (t == 0) {
    conf[i] = 1.f / s;
    qstats[2 * i] = qm;
    qstats[2 * i + 1] = qs;
    tokens[i] = cand_idx[i * CANDIDATES + int(unpack_argmax_index(picks[i]))];
    if (i == 0) p->draws += 1;
  }
}

}  // namespace

std::size_t topk_workspace_bytes(int rows, int V) {
  (void)V;
  return std::size_t(rows) * sizeof(TopkRowWs);
}

void topk_candidates(const float* base, int rows, int V, float* cand_val, std::int32_t* cand_idx, void* ws,
                     cudaStream_t stream) {
  if (V < CANDIDATES) throw std::invalid_argument("topk_candidates: V must be at least 1024");
  if (rows <= 0) return;
  auto* w = static_cast<TopkRowWs*>(ws);
  const dim3 grid(TOPK_SLICES, rows);
  launch_pdl(topk_hist_hi_kernel, grid, dim3(TOPK_THREADS), stream, base, V, w);
  launch_pdl(topk_hist_lo_kernel, grid, dim3(TOPK_THREADS), stream, base, V, w);
  launch_pdl(topk_count_kernel, grid, dim3(TOPK_THREADS), stream, base, V, w);
  launch_pdl(topk_scatter_kernel, grid, dim3(TOPK_THREADS), stream, base, V, w);
  launch_pdl(topk_sort_kernel, dim3(rows), dim3(SORT_THREADS), stream, w, cand_val,
             cand_idx);
}

void markov_pre(const __nv_bfloat16* rnn_w, const __nv_bfloat16* h, const __nv_bfloat16* f_rnn, float* part,
                cudaStream_t stream) {
  launch_pdl(markov_pre_kernel, dim3(RNN_OUT * 32 / PRE_THREADS), dim3(PRE_THREADS), stream, rnn_w, h, f_rnn, part);
}

void markov_step(int i, const __nv_bfloat16* rnn_w, const __nv_bfloat16* rnn_b, const __nv_bfloat16* w1,
                 const float* part, const unsigned long long* picks, const std::int32_t* cand_idx,
                 const DraftParams* p, __nv_bfloat16* state, __nv_bfloat16* latent, cudaStream_t stream) {
  if (i < 0 || i >= GAMMA) throw std::invalid_argument("markov_step: step out of range");
  launch_pdl(markov_step_kernel, dim3(RANK * 32 / 256), dim3(256), stream, i, rnn_w, rnn_b, w1, part, picks,
             cand_idx, p, state, latent);
}

void markov_logits(int i, const __nv_bfloat16* fold_w, const __nv_bfloat16* fold_b, const float* cand_val,
                   const std::int32_t* cand_idx, const __nv_bfloat16* latent, const DraftParams* p,
                   float* cand_logits, unsigned long long* picks, cudaStream_t stream) {
  if (i < 0 || i >= GAMMA) throw std::invalid_argument("markov_logits: step out of range");
  launch_pdl(markov_logits_kernel, dim3(CANDIDATES * 32 / 256), dim3(256), stream, i, fold_w, fold_b, cand_val,
             cand_idx, latent, p, cand_logits, picks);
}

void markov_finish(const std::int32_t* cand_idx, const float* cand_logits, const unsigned long long* picks,
                   std::int32_t* prev_cand, float* draft_logits, int V, std::int32_t* tokens, float* conf,
                   float* qstats, DraftParams* p, cudaStream_t stream) {
  launch_pdl(markov_finish_kernel, dim3(GAMMA), dim3(CANDIDATES), stream, cand_idx, cand_logits, picks, prev_cand,
             draft_logits, V, tokens, conf, qstats, p);
}

}  // namespace ingine::kernels::draft

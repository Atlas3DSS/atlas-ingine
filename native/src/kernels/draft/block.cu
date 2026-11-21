// DSpark draft block: parameters, norms, RoPE, context commit and block attention (dspark.py
// mix_rmsnorm / rms / rope / Drafter._commit / Drafter._attention / the residual stream of _draft).
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include "core/device.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ingine::kernels::draft {
namespace {

constexpr int ROW_THREADS = 256;
constexpr int ROW_PER = H / ROW_THREADS;  // 9
static_assert(H % ROW_THREADS == 0);

// rms(x) as dspark.rms: x * rsqrt(mean(x^2) + eps), then * w (separately rounded FP32 products).
__device__ __forceinline__ float rms_scale(const float (&x)[ROW_PER], float* scratch) {
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < ROW_PER; ++j) ss += x[j] * x[j];
  const float total = block_sum(ss, scratch);
  return rsqrtf(total / float(H) + EPS);
}

// ---- parameters ---------------------------------------------------------------------------------------

__global__ void set_draft_inputs_kernel(DraftParams* p, int anchor, int P, float temperature) {
  p->anchor = anchor;
  p->P = P;
  p->temperature = temperature;
}
__global__ void set_commit_start_kernel(DraftParams* p, int start) { p->start = start; }
__global__ void set_seed_kernel(DraftParams* p, std::uint64_t seed) {
  p->seed = seed;
  p->draws = 0;
}

// ---- context -------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(ROW_THREADS) mix_rmsnorm_kernel(const __nv_bfloat16* __restrict__ feats,
                                                                  const __nv_bfloat16* __restrict__ alpha,
                                                                  const __nv_bfloat16* __restrict__ w,
                                                                  __nv_bfloat16* __restrict__ out) {
  __shared__ float scratch[33];
  const int r = blockIdx.x;
  const __nv_bfloat16* f = feats + size_t(r) * 4 * H;
  const float a0 = bf(alpha[0]), a1 = bf(alpha[1]), a2 = bf(alpha[2]), a3 = bf(alpha[3]);
  float m[ROW_PER];
#pragma unroll
  for (int j = 0; j < ROW_PER; ++j) {
    const int d = threadIdx.x + j * ROW_THREADS;
    float s = __fmul_rn(bf(f[d]), a0);
    s = __fadd_rn(s, __fmul_rn(bf(f[H + d]), a1));
    s = __fadd_rn(s, __fmul_rn(bf(f[2 * H + d]), a2));
    m[j] = __fadd_rn(s, __fmul_rn(bf(f[3 * H + d]), a3));
  }
  const float rs = rms_scale(m, scratch);
#pragma unroll
  for (int j = 0; j < ROW_PER; ++j) {
    const int d = threadIdx.x + j * ROW_THREADS;
    out[size_t(r) * H + d] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(m[j], rs), bf(w[d])));
  }
}

// One head row of 128 values, one per thread (blockDim 128): bf16(rms(x) * w), then neox RoPE at `pos`
// over the full head dim, rounded to BF16. `x` is this thread's value; returns the thread's output.
__device__ __forceinline__ float norm_rope_head(float x, const __nv_bfloat16* __restrict__ w, int pos,
                                                float* scratch, float* pair) {
  const int d = threadIdx.x;
  const float ss = block_sum(x * x, scratch);
  const float rs = rsqrtf(ss / float(HEAD_DIM) + EPS);
  const float y = bf_round(__fmul_rn(__fmul_rn(x, rs), bf(w[d])));
  pair[d] = y;
  __syncthreads();
  const int half = HEAD_DIM / 2, i = d & (half - 1);
  // inv = theta ** (-(2 i) / 128) and the angle in FP32, as dspark.rope
  const float inv = powf(ROPE_THETA, (-float(i) * 2.f) / float(HEAD_DIM));
  const float ang = __fmul_rn(float(pos), inv);
  const float c = cosf(ang), s = sinf(ang);
  const float x1 = pair[i], x2 = pair[i + half];
  const float o = d < half ? __fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s))
                           : __fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s));
  __syncthreads();  // pair is reused by the caller's next head
  return bf_round(o);
}

__global__ void __launch_bounds__(HEAD_DIM) ctx_kv_store_kernel(const __nv_bfloat16* __restrict__ kv,
                                                               const DraftParams* __restrict__ p, int row0,
                                                               int keep_from,
                                                               const __nv_bfloat16* __restrict__ k_norm,
                                                               __nv_bfloat16* __restrict__ k_cache,
                                                               __nv_bfloat16* __restrict__ v_cache, int size,
                                                               std::int32_t* __restrict__ ctx_pos) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  const int r = keep_from + blockIdx.x, g = blockIdx.y, d = threadIdx.x;
  const int pos = p->start + row0 + r;
  const int slot = pos % size;
  const __nv_bfloat16* row = kv + size_t(r) * 2 * KV_DIM;
  const float k = norm_rope_head(bf(row[g * HEAD_DIM + d]), k_norm, pos, scratch, pair);
  const size_t at = (size_t(slot) * KV_HEADS + g) * HEAD_DIM + d;
  k_cache[at] = __float2bfloat16_rn(k);
  v_cache[at] = row[KV_DIM + g * HEAD_DIM + d];
  if (ctx_pos != nullptr && g == 0 && d == 0) ctx_pos[slot] = pos;
}

// ---- draft block ---------------------------------------------------------------------------------------

__global__ void __launch_bounds__(ROW_THREADS) embed_block_kernel(const __nv_bfloat16* __restrict__ embed,
                                                                  const float* __restrict__ embed_norm,
                                                                  const __nv_bfloat16* __restrict__ mask_embed,
                                                                  const __nv_bfloat16* __restrict__ in_norm,
                                                                  const DraftParams* __restrict__ p,
                                                                  __nv_bfloat16* __restrict__ res,
                                                                  __nv_bfloat16* __restrict__ xn,
                                                                  unsigned long long* __restrict__ picks) {
  __shared__ float scratch[33];
  const int r = blockIdx.x;
  float x[ROW_PER];
  if (r == 0) {
    const __nv_bfloat16* e = embed + size_t(p->anchor) * H;
    float v[ROW_PER];
#pragma unroll
    for (int j = 0; j < ROW_PER; ++j) v[j] = bf(e[threadIdx.x + j * ROW_THREADS]);
    const float rs = rms_scale(v, scratch);
#pragma unroll
    for (int j = 0; j < ROW_PER; ++j)
      x[j] = bf_round(__fmul_rn(__fmul_rn(v[j], rs), embed_norm[threadIdx.x + j * ROW_THREADS]));
  } else {
#pragma unroll
    for (int j = 0; j < ROW_PER; ++j) x[j] = bf(mask_embed[threadIdx.x + j * ROW_THREADS]);
  }
  const float rs = rms_scale(x, scratch);
#pragma unroll
  for (int j = 0; j < ROW_PER; ++j) {
    const int d = threadIdx.x + j * ROW_THREADS;
    res[size_t(r) * H + d] = __float2bfloat16_rn(x[j]);
    xn[size_t(r) * H + d] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[j], rs), bf(in_norm[d])));
  }
  if (r == 0 && threadIdx.x < GAMMA) picks[threadIdx.x] = 0ull;
}

// grid (8 rows, 24 q + 8 k + 8 v heads), block 128
__global__ void __launch_bounds__(HEAD_DIM) qkv_rope_kernel(const float* __restrict__ qkv,
                                                           const __nv_bfloat16* __restrict__ q_norm,
                                                           const __nv_bfloat16* __restrict__ k_norm,
                                                           const DraftParams* __restrict__ p,
                                                           __nv_bfloat16* __restrict__ q_out,
                                                           __nv_bfloat16* __restrict__ k_cache,
                                                           __nv_bfloat16* __restrict__ v_cache, bool local,
                                                           int ctx) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int pos = p->P + r;
  const float* row = qkv + size_t(r) * QKV_ROWS;
  const int slot = local ? WINDOW + r : pos % ctx;
  if (h < HEADS) {
    const float q = norm_rope_head(row[h * HEAD_DIM + d], q_norm, pos, scratch, pair);
    q_out[(size_t(r) * HEADS + h) * HEAD_DIM + d] = __float2bfloat16_rn(q);
  } else if (h < HEADS + KV_HEADS) {
    const int g = h - HEADS;
    const float k = norm_rope_head(row[Q_DIM + g * HEAD_DIM + d], k_norm, pos, scratch, pair);
    k_cache[(size_t(slot) * KV_HEADS + g) * HEAD_DIM + d] = __float2bfloat16_rn(k);
  } else {
    const int g = h - HEADS - KV_HEADS;
    v_cache[(size_t(slot) * KV_HEADS + g) * HEAD_DIM + d] =
        __float2bfloat16_rn(row[Q_DIM + KV_DIM + g * HEAD_DIM + d]);
  }
}

// ---- block attention (split over key chunks, then combined) ----------------------------------------------

constexpr int ATT_THREADS = 128;           // one per head dim in the PV phase
constexpr int ATT_TILE = 64;               // keys per shared-memory tile
constexpr int ATT_CHUNK = 512;             // keys per CTA
constexpr int NQ = (HEADS / KV_HEADS) * BLOCK;  // 24 queries per kv head: 3 heads x 8 rows
constexpr int KW = HEAD_DIM / 2 + 1;       // padded tile row in 32-bit words (conflict-free columns)

int att_chunks(int nkeys) { return (nkeys + ATT_CHUNK - 1) / ATT_CHUNK; }

// Query qi = hh * 8 + r is row r of head 3 g + hh.
__global__ void __launch_bounds__(ATT_THREADS) attn_partial_kernel(const __nv_bfloat16* __restrict__ q,
                                                                  const __nv_bfloat16* __restrict__ k_cache,
                                                                  const __nv_bfloat16* __restrict__ v_cache,
                                                                  int nkeys, const std::int32_t* __restrict__ ctx_pos,
                                                                  const DraftParams* __restrict__ p,
                                                                  float* __restrict__ ws_o,
                                                                  float* __restrict__ ws_ml) {
  __shared__ float qs[NQ][HEAD_DIM];
  __shared__ uint32_t tile[ATT_TILE][KW];
  __shared__ float sc[NQ][ATT_TILE];
  __shared__ float m_run[NQ], l_run[NQ], alpha[NQ];
  const int chunk = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
  const int k0 = chunk * ATT_CHUNK, k1 = min(k0 + ATT_CHUNK, nkeys);
  const int P = p->P;
  const bool local = ctx_pos != nullptr;
  constexpr float scale = 0.08838834764831845f;  // 1 / sqrt(128)

  for (int idx = tid; idx < NQ * HEAD_DIM; idx += ATT_THREADS) {
    const int qi = idx / HEAD_DIM, d = idx % HEAD_DIM, hh = qi / BLOCK, r = qi % BLOCK;
    qs[qi][d] = bf(q[(size_t(r) * HEADS + g * (HEADS / KV_HEADS) + hh) * HEAD_DIM + d]);
  }
  if (tid < NQ) {
    m_run[tid] = -INFINITY;
    l_run[tid] = 0.f;
  }
  float o[NQ];
#pragma unroll
  for (int i = 0; i < NQ; ++i) o[i] = 0.f;
  __syncthreads();

  // global layer: no valid key at or beyond P + 8
  const int kend = local ? k1 : min(k1, P + BLOCK);
  for (int t0 = k0; t0 < kend; t0 += ATT_TILE) {
    const int nt = min(ATT_TILE, k1 - t0);
    for (int idx = tid; idx < ATT_TILE * (HEAD_DIM / 2); idx += ATT_THREADS) {
      const int row = idx / (HEAD_DIM / 2), w = idx % (HEAD_DIM / 2);
      tile[row][w] = row < nt ? reinterpret_cast<const uint32_t*>(
                                    k_cache + (size_t(t0 + row) * KV_HEADS + g) * HEAD_DIM)[w]
                              : 0u;
    }
    __syncthreads();
    {
      const int kk = tid & (ATT_TILE - 1), q0 = (tid / ATT_TILE) * (NQ / 2);
      const int j = t0 + kk;
      bool valid = kk < nt;
      if (valid) {
        if (local) {
          if (j < WINDOW) {
            const int cp = ctx_pos[j];
            valid = cp >= 0 && cp < P && cp >= P - WINDOW;
          }
        } else {
          valid = j < P + BLOCK;
        }
      }
      float acc[NQ / 2];
#pragma unroll
      for (int i = 0; i < NQ / 2; ++i) acc[i] = 0.f;
#pragma unroll 4
      for (int w = 0; w < HEAD_DIM / 2; ++w) {
        const uint32_t kw = tile[kk][w];
        const float klo = __uint_as_float(kw << 16), khi = __uint_as_float(kw & 0xffff0000u);
#pragma unroll
        for (int i = 0; i < NQ / 2; ++i) acc[i] = fmaf(qs[q0 + i][2 * w + 1], khi, fmaf(qs[q0 + i][2 * w], klo, acc[i]));
      }
#pragma unroll
      for (int i = 0; i < NQ / 2; ++i) sc[q0 + i][kk] = valid ? acc[i] * scale : -INFINITY;
    }
    __syncthreads();
    {
      const int warp = tid >> 5, lane = tid & 31;
      for (int qi = warp; qi < NQ; qi += ATT_THREADS / 32) {
        const float v0 = sc[qi][lane], v1 = sc[qi][lane + 32];
        const float m_old = m_run[qi];
        const float m_new = fmaxf(m_old, warp_max(fmaxf(v0, v1)));
        float a = 1.f, p0 = 0.f, p1 = 0.f;
        if (m_new != -INFINITY) {
          a = m_old == -INFINITY ? 0.f : expf(m_old - m_new);
          p0 = expf(v0 - m_new);
          p1 = expf(v1 - m_new);
        }
        const float s = warp_sum(p0 + p1);
        sc[qi][lane] = p0;
        sc[qi][lane + 32] = p1;
        if (lane == 0) {
          l_run[qi] = l_run[qi] * a + s;
          m_run[qi] = m_new;
          alpha[qi] = a;
        }
      }
    }
    for (int idx = tid; idx < ATT_TILE * (HEAD_DIM / 2); idx += ATT_THREADS) {
      const int row = idx / (HEAD_DIM / 2), w = idx % (HEAD_DIM / 2);
      tile[row][w] = row < nt ? reinterpret_cast<const uint32_t*>(
                                    v_cache + (size_t(t0 + row) * KV_HEADS + g) * HEAD_DIM)[w]
                              : 0u;
    }
    __syncthreads();
    {
      const int d = tid;
      const __nv_bfloat16* vt = reinterpret_cast<const __nv_bfloat16*>(&tile[0][0]);
#pragma unroll
      for (int i = 0; i < NQ; ++i) o[i] *= alpha[i];
      for (int kk = 0; kk < nt; ++kk) {
        const float v = bf(vt[kk * (2 * KW) + d]);
#pragma unroll
        for (int i = 0; i < NQ; ++i) o[i] = fmaf(sc[i][kk], v, o[i]);
      }
    }
    __syncthreads();
  }
  const size_t base = (size_t(chunk) * KV_HEADS + g) * NQ;
#pragma unroll
  for (int i = 0; i < NQ; ++i) ws_o[(base + i) * HEAD_DIM + tid] = o[i];
  if (tid < NQ) {
    ws_ml[(base + tid) * 2] = m_run[tid];
    ws_ml[(base + tid) * 2 + 1] = l_run[tid];
  }
}

// grid KV_HEADS * NQ, block 128
__global__ void __launch_bounds__(HEAD_DIM) attn_combine_kernel(const float* __restrict__ ws_o,
                                                               const float* __restrict__ ws_ml, int chunks,
                                                               __nv_bfloat16* __restrict__ out) {
  const int g = blockIdx.x / NQ, qi = blockIdx.x % NQ, d = threadIdx.x;
  float m = -INFINITY;
  for (int c = 0; c < chunks; ++c) m = fmaxf(m, ws_ml[((size_t(c) * KV_HEADS + g) * NQ + qi) * 2]);
  float l = 0.f, o = 0.f;
  if (m != -INFINITY) {
    for (int c = 0; c < chunks; ++c) {
      const size_t at = (size_t(c) * KV_HEADS + g) * NQ + qi;
      const float mc = ws_ml[at * 2];
      if (mc == -INFINITY) continue;
      const float f = expf(mc - m);
      l += ws_ml[at * 2 + 1] * f;
      o += ws_o[at * HEAD_DIM + d] * f;
    }
  }
  const int hh = qi / BLOCK, r = qi % BLOCK, h = g * (HEADS / KV_HEADS) + hh;
  out[size_t(r) * Q_DIM + h * HEAD_DIM + d] = __float2bfloat16_rn(l > 0.f ? o / l : 0.f);
}

// ---- residual stream -----------------------------------------------------------------------------------

__global__ void __launch_bounds__(ROW_THREADS) add_rmsnorm_kernel(__nv_bfloat16* __restrict__ res,
                                                                  const __nv_bfloat16* __restrict__ delta,
                                                                  const __nv_bfloat16* __restrict__ w,
                                                                  __nv_bfloat16* __restrict__ xn) {
  __shared__ float scratch[33];
  const size_t r = blockIdx.x;
  float x[ROW_PER];
#pragma unroll
  for (int j = 0; j < ROW_PER; ++j) {
    const size_t at = r * H + threadIdx.x + j * ROW_THREADS;
    x[j] = bf_round(__fadd_rn(bf(res[at]), bf(delta[at])));
    res[at] = __float2bfloat16_rn(x[j]);
  }
  const float rs = rms_scale(x, scratch);
#pragma unroll
  for (int j = 0; j < ROW_PER; ++j) {
    const int d = threadIdx.x + j * ROW_THREADS;
    xn[r * H + d] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[j], rs), bf(w[d])));
  }
}

__global__ void silu_mul_kernel(const float* __restrict__ gu, int rows, int inter, __nv_bfloat16* __restrict__ out) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(rows) * inter) return;
  const size_t r = i / inter, j = i % inter;
  const float g = gu[r * 2 * inter + j], u = gu[r * 2 * inter + inter + j];
  const float s = __fdiv_rn(g, __fadd_rn(1.f, expf(-g)));  // torch silu: x / (1 + exp(-x))
  out[i] = __float2bfloat16_rn(__fmul_rn(s, u));
}

__global__ void fill_kernel(float* x, size_t n, float value) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
    x[i] = value;
}

void check_launch() { CUDA_CHECK(cudaGetLastError()); }

}  // namespace

void set_draft_inputs(DraftParams* p, int anchor, int P, float temperature, cudaStream_t stream) {
  set_draft_inputs_kernel<<<1, 1, 0, stream>>>(p, anchor, P, temperature);
  check_launch();
}
void set_commit_start(DraftParams* p, int start, cudaStream_t stream) {
  set_commit_start_kernel<<<1, 1, 0, stream>>>(p, start);
  check_launch();
}
void set_seed(DraftParams* p, std::uint64_t seed, cudaStream_t stream) {
  set_seed_kernel<<<1, 1, 0, stream>>>(p, seed);
  check_launch();
}

void mix_rmsnorm(const __nv_bfloat16* feats, int n, const __nv_bfloat16* alpha, const __nv_bfloat16* w,
                 __nv_bfloat16* out, cudaStream_t stream) {
  if (n <= 0) return;
  mix_rmsnorm_kernel<<<n, ROW_THREADS, 0, stream>>>(feats, alpha, w, out);
  check_launch();
}

void ctx_kv_store(const __nv_bfloat16* kv, int n, const DraftParams* p, int row0, int keep_from,
                  const __nv_bfloat16* k_norm, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int size,
                  std::int32_t* ctx_pos, cudaStream_t stream) {
  if (keep_from < 0 || size <= 0) throw std::invalid_argument("ctx_kv_store: bad keep_from/size");
  if (n - keep_from <= 0) return;
  ctx_kv_store_kernel<<<dim3(n - keep_from, KV_HEADS), HEAD_DIM, 0, stream>>>(kv, p, row0, keep_from, k_norm,
                                                                               k_cache, v_cache, size, ctx_pos);
  check_launch();
}

void embed_block(const __nv_bfloat16* embed, const float* embed_norm, const __nv_bfloat16* mask_embed,
                 const __nv_bfloat16* in_norm, const DraftParams* p, __nv_bfloat16* res, __nv_bfloat16* xn,
                 unsigned long long* picks, cudaStream_t stream) {
  embed_block_kernel<<<BLOCK, ROW_THREADS, 0, stream>>>(embed, embed_norm, mask_embed, in_norm, p, res, xn, picks);
  check_launch();
}

void qkv_rope(const float* qkv, const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm, const DraftParams* p,
              __nv_bfloat16* q_out, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, bool local, int ctx,
              cudaStream_t stream) {
  qkv_rope_kernel<<<dim3(BLOCK, HEADS + 2 * KV_HEADS), HEAD_DIM, 0, stream>>>(qkv, q_norm, k_norm, p, q_out, k_cache,
                                                                              v_cache, local, ctx);
  check_launch();
}

std::size_t attention_workspace_floats(int nkeys) {
  return std::size_t(att_chunks(nkeys)) * KV_HEADS * NQ * (HEAD_DIM + 2);
}

void block_attention(const __nv_bfloat16* q, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                     int nkeys, const std::int32_t* ctx_pos, const DraftParams* p, float* ws,
                     __nv_bfloat16* out, cudaStream_t stream) {
  if (nkeys <= 0) throw std::invalid_argument("block_attention: nkeys must be positive");
  if (ctx_pos != nullptr && nkeys != WINDOW + BLOCK)
    throw std::invalid_argument("block_attention: local layers attend to WINDOW + 8 keys");
  const int chunks = att_chunks(nkeys);
  float* ws_o = ws;
  float* ws_ml = ws + std::size_t(chunks) * KV_HEADS * NQ * HEAD_DIM;
  attn_partial_kernel<<<dim3(chunks, KV_HEADS), ATT_THREADS, 0, stream>>>(q, k_cache, v_cache, nkeys, ctx_pos, p,
                                                                          ws_o, ws_ml);
  check_launch();
  attn_combine_kernel<<<KV_HEADS * NQ, HEAD_DIM, 0, stream>>>(ws_o, ws_ml, chunks, out);
  check_launch();
}

void add_rmsnorm(__nv_bfloat16* res, const __nv_bfloat16* delta, const __nv_bfloat16* w, __nv_bfloat16* xn,
                 int rows, cudaStream_t stream) {
  if (rows <= 0) return;
  add_rmsnorm_kernel<<<rows, ROW_THREADS, 0, stream>>>(res, delta, w, xn);
  check_launch();
}

void silu_mul(const float* gu, int rows, int inter, __nv_bfloat16* out, cudaStream_t stream) {
  const size_t n = size_t(rows) * inter;
  if (n == 0) return;
  silu_mul_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(gu, rows, inter, out);
  check_launch();
}

void fill(float* x, std::size_t n, float value, cudaStream_t stream) {
  if (n == 0) return;
  const unsigned blocks = unsigned(std::min<std::size_t>((n + 255) / 256, 4096));
  fill_kernel<<<blocks, 256, 0, stream>>>(x, n, value);
  check_launch();
}

}  // namespace ingine::kernels::draft

// DSpark draft block: parameters, norms, RoPE, context commit and block attention (dspark.py
// mix_rmsnorm / rms / rope / Drafter._commit / Drafter._attention / the residual stream of _draft).
//
// Kernels of the commit/draft chains are PDL launches (launch_pdl): pdl_trigger() first, pdl_wait() before
// any read of data another kernel writes. Pointers to such data are not __restrict__ (a const __restrict__
// load may be scheduled above the wait); weights keep it.
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

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

// mix_rmsnorm of one feats row [4, H] into this thread's ROW_PER values (normed, BF16-rounded)
__device__ __forceinline__ void mix_row(const __nv_bfloat16* f, const __nv_bfloat16* __restrict__ alpha,
                                        const __nv_bfloat16* __restrict__ w, float* scratch, float (&y)[ROW_PER]) {
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
  for (int j = 0; j < ROW_PER; ++j)
    y[j] = bf_round(__fmul_rn(__fmul_rn(m[j], rs), bf(w[threadIdx.x + j * ROW_THREADS])));
}

// ---- parameters (ordinary launches) ------------------------------------------------------------------

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
__global__ void set_commit_kernel(DraftParams* p, const __nv_bfloat16* feats, int n, int start) {
  p->commit_feats = feats;
  p->ncommit = n;
  p->start = start;
}
__global__ void set_round_kernel(DraftParams* p, RoundInputs in) {
  const int m = *in.m, n = *in.accepted + 1, start = p->P;
  const __nv_bfloat16* feats = in.taps[0];
  for (int i = 0; i < in.nwidths; ++i)
    if (in.widths[i] == m) feats = in.taps[i];
  p->commit_feats = feats;
  p->ncommit = n;
  p->start = start;
  p->anchor = p->temperature > 0.f ? *in.next_sampled : *in.next_greedy;
  p->P = start + n;
}

// ---- context -------------------------------------------------------------------------------------------

__global__ void __launch_bounds__(ROW_THREADS) mix_rmsnorm_kernel(const __nv_bfloat16* feats,
                                                                  const __nv_bfloat16* __restrict__ alpha,
                                                                  const __nv_bfloat16* __restrict__ w,
                                                                  __nv_bfloat16* out) {
  __shared__ float scratch[33];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x;
  float y[ROW_PER];
  mix_row(feats + size_t(r) * 4 * H, alpha, w, scratch, y);
#pragma unroll
  for (int j = 0; j < ROW_PER; ++j) out[size_t(r) * H + threadIdx.x + j * ROW_THREADS] = __float2bfloat16_rn(y[j]);
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

__global__ void __launch_bounds__(HEAD_DIM) ctx_kv_store_kernel(const __nv_bfloat16* kv, const DraftParams* p,
                                                               int row0, int keep_from,
                                                               const __nv_bfloat16* __restrict__ k_norm,
                                                               __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
                                                               int size, std::int32_t* ctx_pos) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
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

// grid (BLOCK rows, 4 parts): parts 0-2 the layers' context inputs, part 3 f_rnn (row n - 1)
__global__ void __launch_bounds__(ROW_THREADS) commit_mix_kernel(const DraftParams* p,
                                                                 const __nv_bfloat16* __restrict__ alpha,
                                                                 CtxLayers L, const __nv_bfloat16* __restrict__ f_norm,
                                                                 __nv_bfloat16* xin, __nv_bfloat16* f_rnn) {
  __shared__ float scratch[33];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, part = blockIdx.y;
  const int n = p->ncommit;
  const __nv_bfloat16* f = p->commit_feats + size_t(r) * 4 * H;
  if (part == 3) {
    if (r != n - 1) return;
    float y[ROW_PER];
    mix_row(f, alpha + 12, f_norm, scratch, y);
#pragma unroll
    for (int j = 0; j < ROW_PER; ++j) f_rnn[threadIdx.x + j * ROW_THREADS] = __float2bfloat16_rn(y[j]);
    return;
  }
  __nv_bfloat16* out = xin + (size_t(part) * BLOCK + r) * H;
  if (r >= n) {
#pragma unroll
    for (int j = 0; j < ROW_PER; ++j) out[threadIdx.x + j * ROW_THREADS] = __float2bfloat16_rn(0.f);
    return;
  }
  float y[ROW_PER];
  mix_row(f, alpha + 4 * part, L.ctx_norm[part], scratch, y);
#pragma unroll
  for (int j = 0; j < ROW_PER; ++j) out[threadIdx.x + j * ROW_THREADS] = __float2bfloat16_rn(y[j]);
}

// grid (BLOCK rows, KV_HEADS, 3 layers), block 128
__global__ void __launch_bounds__(HEAD_DIM) commit_store_kernel(const float* kv, const DraftParams* p, CtxLayers L) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, g = blockIdx.y, l = blockIdx.z, d = threadIdx.x;
  if (r >= p->ncommit) return;
  const int pos = p->start + r;
  const int slot = pos % L.size[l];
  const float* row = kv + size_t(r) * 3 * 2 * KV_DIM + size_t(l) * 2 * KV_DIM;
  const float k = norm_rope_head(row[g * HEAD_DIM + d], L.k_norm_ctx[l], pos, scratch, pair);
  const size_t at = (size_t(slot) * KV_HEADS + g) * HEAD_DIM + d;
  L.k_cache[l][at] = __float2bfloat16_rn(k);
  L.v_cache[l][at] = __float2bfloat16_rn(row[KV_DIM + g * HEAD_DIM + d]);
  if (L.ctx_pos[l] != nullptr && g == 0 && d == 0) L.ctx_pos[l][slot] = pos;
}

// ---- draft block ---------------------------------------------------------------------------------------

__global__ void __launch_bounds__(ROW_THREADS) embed_block_kernel(const __nv_bfloat16* __restrict__ embed,
                                                                  const float* __restrict__ embed_norm,
                                                                  const __nv_bfloat16* __restrict__ mask_embed,
                                                                  const __nv_bfloat16* __restrict__ in_norm,
                                                                  const DraftParams* p, __nv_bfloat16* res,
                                                                  __nv_bfloat16* xn, unsigned long long* picks) {
  __shared__ float scratch[33];
  pdl_trigger();
  pdl_wait();
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
__global__ void __launch_bounds__(HEAD_DIM) qkv_rope_kernel(const float* qkv, const __nv_bfloat16* __restrict__ q_norm,
                                                           const __nv_bfloat16* __restrict__ k_norm,
                                                           const DraftParams* p, __nv_bfloat16* q_out,
                                                           __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, bool local,
                                                           int ctx) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
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
// The keys split into chunks of ATT_TILE * tiles keys, tiles = ceil(nkeys / (ATT_TILE * ATT_MAX_CHUNKS)):
// short contexts get one 64-key tile per CTA (many CTAs, little serial work each), long ones at most 128
// chunks per kv head (the combine stays small). A CTA handles one kv head's 24 queries (3 heads x 8 rows),
// so each key and value row is read once; tiles without a valid key are skipped without loading them.

constexpr int ATT_THREADS = 256;
constexpr int ATT_TILE = 64;                    // keys per shared-memory tile
constexpr int ATT_MAX_CHUNKS = 128;             // chunks per kv head (at most)
constexpr int NQ = (HEADS / KV_HEADS) * BLOCK;  // 24 queries per kv head: 3 heads x 8 rows
constexpr int KW = HEAD_DIM / 2 + 1;            // padded tile row in 32-bit words (conflict-free rows)

int att_tiles_per_chunk(int nkeys) {
  return std::max(1, (nkeys + ATT_TILE * ATT_MAX_CHUNKS - 1) / (ATT_TILE * ATT_MAX_CHUNKS));
}
int att_chunks(int nkeys) {
  const int ck = ATT_TILE * att_tiles_per_chunk(nkeys);
  return (nkeys + ck - 1) / ck;
}

// Query qi = hh * 8 + r is row r of head 3 g + hh. (Splitting the 24 queries over 3 CTAs was measured: no
// faster at short contexts, the CTA's time is load latency, not arithmetic.)
__global__ void __launch_bounds__(ATT_THREADS) attn_partial_kernel(const __nv_bfloat16* q,
                                                                  const __nv_bfloat16* k_cache,
                                                                  const __nv_bfloat16* v_cache, int nkeys,
                                                                  int chunk_keys, const std::int32_t* ctx_pos,
                                                                  const DraftParams* p, float* ws_o, float* ws_ml) {
  constexpr int QG = NQ / (ATT_THREADS / ATT_TILE);  // queries per thread in the score phase: 6
  constexpr int QV = NQ / (ATT_THREADS / HEAD_DIM);  // queries per thread in the PV phase: 12
  __shared__ __align__(16) float qs[NQ][HEAD_DIM];
  __shared__ uint32_t tile[ATT_TILE][KW];
  __shared__ __align__(16) float sc[NQ][ATT_TILE];
  __shared__ float m_run[NQ], l_run[NQ], alpha[NQ];
  __shared__ int valid[ATT_TILE];
  pdl_trigger();
  pdl_wait();
  const int chunk = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
  const int k0 = chunk * chunk_keys, k1 = min(k0 + chunk_keys, nkeys);
  const int P = p->P;
  const bool local = ctx_pos != nullptr;
  constexpr float scale = 0.08838834764831845f;  // 1 / sqrt(128)

  // q rows: 24 x 16 chunks of 8 BF16, all loads in flight together
  {
    constexpr int QCH = NQ * (HEAD_DIM / 8);
    constexpr int QPT = (QCH + ATT_THREADS - 1) / ATT_THREADS;
    uint4 qv[QPT];
#pragma unroll
    for (int u = 0; u < QPT; ++u) {
      const int idx = tid + u * ATT_THREADS, qi = idx >> 4, c = idx & 15, hh = qi / BLOCK, r = qi % BLOCK;
      if (idx < QCH)
        qv[u] = *reinterpret_cast<const uint4*>(
            q + (size_t(r) * HEADS + g * (HEADS / KV_HEADS) + hh) * HEAD_DIM + 8 * c);
    }
#pragma unroll
    for (int u = 0; u < QPT; ++u) {
      const int idx = tid + u * ATT_THREADS, c = idx & 15;
      if (idx < QCH) unpack8(qv[u], &qs[idx >> 4][8 * c]);
    }
  }
  if (tid < NQ) {
    m_run[tid] = -INFINITY;
    l_run[tid] = 0.f;
  }
  float o[QV];
#pragma unroll
  for (int i = 0; i < QV; ++i) o[i] = 0.f;

  // global layer: no valid key at or beyond P + 8
  const int kend = local ? k1 : min(k1, P + BLOCK);
  constexpr int TCH = ATT_TILE * (HEAD_DIM / 8) / ATT_THREADS;  // 4 chunks of K and of V per thread
  // a tile's keys and values (16-byte loads, all in flight together); `live` says which rows to read
  auto load = [&](int t, auto live, uint4 (&kr)[TCH], uint4 (&vr)[TCH]) {
#pragma unroll
    for (int u = 0; u < TCH; ++u) {
      const int idx = tid + u * ATT_THREADS, row = idx >> 4, c = idx & 15;
      const size_t at = (size_t(t + row) * KV_HEADS + g) * HEAD_DIM + 8 * c;
      const bool l = live(row);
      kr[u] = l ? *reinterpret_cast<const uint4*>(k_cache + at) : make_uint4(0, 0, 0, 0);
      vr[u] = l ? *reinterpret_cast<const uint4*>(v_cache + at) : make_uint4(0, 0, 0, 0);
    }
  };
  uint4 kreg[TCH], vreg[TCH], knext[TCH], vnext[TCH];
  bool prefetched = false;  // kreg/vreg already hold this tile (global layer, multi-tile chunks)
  for (int t0 = k0; t0 < kend; t0 += ATT_TILE) {
    const int nt = min(ATT_TILE, k1 - t0);
    int any = 0;
    if (tid < ATT_TILE) {
      const int j = t0 + tid;
      bool v = tid < nt;
      if (v) {
        if (local) {
          if (j < WINDOW) {
            const int cp = ctx_pos[j];
            v = cp >= 0 && cp < P && cp >= P - WINDOW;
          }
        } else {
          v = j < P + BLOCK;
        }
      }
      valid[tid] = v;
      any = v;
    }
    if (!__syncthreads_or(any)) continue;  // also orders qs/m_run and the previous tile's readers
    // the values wait in registers until the scores are done with the shared tile
    if (!prefetched) load(t0, [&](int row) { return row < nt && valid[row]; }, kreg, vreg);
    auto stage = [&](const uint4 (&reg)[TCH]) {
#pragma unroll
      for (int u = 0; u < TCH; ++u) {
        const int idx = tid + u * ATT_THREADS, row = idx >> 4, c = idx & 15;
        tile[row][4 * c] = reg[u].x;
        tile[row][4 * c + 1] = reg[u].y;
        tile[row][4 * c + 2] = reg[u].z;
        tile[row][4 * c + 3] = reg[u].w;
      }
    };
    stage(kreg);
    __syncthreads();
    // global layer: the next tile's loads overlap this tile's arithmetic (its valid keys are those below kend)
    prefetched = !local && t0 + ATT_TILE < kend;
    if (prefetched) load(t0 + ATT_TILE, [&](int row) { return t0 + ATT_TILE + row < kend; }, knext, vnext);
    {
      const int kk = tid & (ATT_TILE - 1), q0 = (tid / ATT_TILE) * QG;
      float acc[QG];
#pragma unroll
      for (int i = 0; i < QG; ++i) acc[i] = 0.f;
#pragma unroll 4
      for (int w2 = 0; w2 < HEAD_DIM / 4; ++w2) {
        const uint32_t ka = tile[kk][2 * w2], kb = tile[kk][2 * w2 + 1];
        const float k0f = __uint_as_float(ka << 16), k1f = __uint_as_float(ka & 0xffff0000u);
        const float k2f = __uint_as_float(kb << 16), k3f = __uint_as_float(kb & 0xffff0000u);
#pragma unroll
        for (int i = 0; i < QG; ++i) {
          const float4 qv = *reinterpret_cast<const float4*>(&qs[q0 + i][4 * w2]);
          acc[i] = fmaf(qv.w, k3f, fmaf(qv.z, k2f, fmaf(qv.y, k1f, fmaf(qv.x, k0f, acc[i]))));
        }
      }
      const bool v = valid[kk];
#pragma unroll
      for (int i = 0; i < QG; ++i) sc[q0 + i][kk] = v ? acc[i] * scale : -INFINITY;
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
    stage(vreg);  // the score phase is done with the keys (barrier above)
    __syncthreads();
    {
      const int d = tid & (HEAD_DIM - 1), q0 = (tid / HEAD_DIM) * QV;
      const __nv_bfloat16* vt = reinterpret_cast<const __nv_bfloat16*>(&tile[0][0]);
#pragma unroll
      for (int i = 0; i < QV; ++i) o[i] *= alpha[q0 + i];
      const int nt4 = (nt + 3) & ~3;  // padded keys have p = 0 and zero values
      for (int kk = 0; kk < nt4; kk += 4) {
        const float v0 = bf(vt[(kk + 0) * (2 * KW) + d]), v1 = bf(vt[(kk + 1) * (2 * KW) + d]);
        const float v2 = bf(vt[(kk + 2) * (2 * KW) + d]), v3 = bf(vt[(kk + 3) * (2 * KW) + d]);
#pragma unroll
        for (int i = 0; i < QV; ++i) {
          const float4 pv = *reinterpret_cast<const float4*>(&sc[q0 + i][kk]);
          o[i] = fmaf(pv.w, v3, fmaf(pv.z, v2, fmaf(pv.y, v1, fmaf(pv.x, v0, o[i]))));
        }
      }
    }
    if (prefetched) {
#pragma unroll
      for (int u = 0; u < TCH; ++u) {
        kreg[u] = knext[u];
        vreg[u] = vnext[u];
      }
    }
  }
  __syncthreads();
  const size_t base = (size_t(chunk) * KV_HEADS + g) * NQ;
  {
    const int d = tid & (HEAD_DIM - 1), q0 = (tid / HEAD_DIM) * QV;
#pragma unroll
    for (int i = 0; i < QV; ++i) ws_o[(base + q0 + i) * HEAD_DIM + d] = o[i];
  }
  if (tid < NQ) {
    ws_ml[(base + tid) * 2] = m_run[tid];
    ws_ml[(base + tid) * 2 + 1] = l_run[tid];
  }
}

// grid KV_HEADS * NQ, block 128
__global__ void __launch_bounds__(HEAD_DIM) attn_combine_kernel(const float* ws_o, const float* ws_ml, int chunks,
                                                               __nv_bfloat16* out) {
  pdl_trigger();
  pdl_wait();
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

__global__ void __launch_bounds__(ROW_THREADS) add_rmsnorm_kernel(__nv_bfloat16* res, const float* delta,
                                                                  const __nv_bfloat16* __restrict__ w,
                                                                  __nv_bfloat16* xn) {
  __shared__ float scratch[33];
  pdl_trigger();
  pdl_wait();
  const size_t r = blockIdx.x;
  float x[ROW_PER];
#pragma unroll
  for (int j = 0; j < ROW_PER; ++j) {
    const size_t at = r * H + threadIdx.x + j * ROW_THREADS;
    x[j] = bf_round(__fadd_rn(bf(res[at]), delta[at]));
    res[at] = __float2bfloat16_rn(x[j]);
  }
  const float rs = rms_scale(x, scratch);
#pragma unroll
  for (int j = 0; j < ROW_PER; ++j) {
    const int d = threadIdx.x + j * ROW_THREADS;
    xn[r * H + d] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[j], rs), bf(w[d])));
  }
}

__global__ void silu_mul_kernel(const float* gu, int rows, int inter, __nv_bfloat16* out) {
  pdl_trigger();
  pdl_wait();
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
void set_commit(DraftParams* p, const __nv_bfloat16* feats, int n, int start, cudaStream_t stream) {
  if (n < 1 || n > BLOCK) throw std::invalid_argument("set_commit: n must be in 1..8");
  set_commit_kernel<<<1, 1, 0, stream>>>(p, feats, n, start);
  check_launch();
}
void set_round(DraftParams* p, const RoundInputs& in, cudaStream_t stream) {
  if (in.nwidths < 1 || in.nwidths > 3 || !in.m || !in.accepted || !in.next_greedy || !in.next_sampled)
    throw std::invalid_argument("set_round: incomplete round inputs");
  set_round_kernel<<<1, 1, 0, stream>>>(p, in);
  check_launch();
}

void mix_rmsnorm(const __nv_bfloat16* feats, int n, const __nv_bfloat16* alpha, const __nv_bfloat16* w,
                 __nv_bfloat16* out, cudaStream_t stream) {
  if (n <= 0) return;
  launch_pdl(mix_rmsnorm_kernel, dim3(n), dim3(ROW_THREADS), stream, feats, alpha, w, out);
}

void ctx_kv_store(const __nv_bfloat16* kv, int n, const DraftParams* p, int row0, int keep_from,
                  const __nv_bfloat16* k_norm, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int size,
                  std::int32_t* ctx_pos, cudaStream_t stream) {
  if (keep_from < 0 || size <= 0) throw std::invalid_argument("ctx_kv_store: bad keep_from/size");
  if (n - keep_from <= 0) return;
  launch_pdl(ctx_kv_store_kernel, dim3(n - keep_from, KV_HEADS), dim3(HEAD_DIM), stream, kv, p, row0, keep_from,
             k_norm, k_cache, v_cache, size, ctx_pos);
}

void commit_mix(const DraftParams* p, const __nv_bfloat16* alpha, const CtxLayers& L, const __nv_bfloat16* f_norm,
                __nv_bfloat16* xin, __nv_bfloat16* f_rnn, cudaStream_t stream) {
  launch_pdl(commit_mix_kernel, dim3(BLOCK, 4), dim3(ROW_THREADS), stream, p, alpha, L, f_norm, xin, f_rnn);
}

void commit_store(const float* kv, const DraftParams* p, const CtxLayers& L, cudaStream_t stream) {
  for (int l = 0; l < 3; ++l)
    if (L.size[l] < BLOCK) throw std::invalid_argument("commit_store: rings must hold a block");
  launch_pdl(commit_store_kernel, dim3(BLOCK, KV_HEADS, 3), dim3(HEAD_DIM), stream, kv, p, L);
}

void embed_block(const __nv_bfloat16* embed, const float* embed_norm, const __nv_bfloat16* mask_embed,
                 const __nv_bfloat16* in_norm, const DraftParams* p, __nv_bfloat16* res, __nv_bfloat16* xn,
                 unsigned long long* picks, cudaStream_t stream) {
  launch_pdl(embed_block_kernel, dim3(BLOCK), dim3(ROW_THREADS), stream, embed, embed_norm, mask_embed, in_norm, p,
             res, xn, picks);
}

void qkv_rope(const float* qkv, const __nv_bfloat16* q_norm, const __nv_bfloat16* k_norm, const DraftParams* p,
              __nv_bfloat16* q_out, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, bool local, int ctx,
              cudaStream_t stream) {
  launch_pdl(qkv_rope_kernel, dim3(BLOCK, HEADS + 2 * KV_HEADS), dim3(HEAD_DIM), stream, qkv, q_norm, k_norm, p, q_out,
             k_cache, v_cache, local, ctx);
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
  const int chunks = att_chunks(nkeys), chunk_keys = ATT_TILE * att_tiles_per_chunk(nkeys);
  float* ws_o = ws;
  float* ws_ml = ws + std::size_t(chunks) * KV_HEADS * NQ * HEAD_DIM;
  launch_pdl(attn_partial_kernel, dim3(chunks, KV_HEADS), dim3(ATT_THREADS), stream, q, k_cache, v_cache, nkeys,
             chunk_keys, ctx_pos, p, ws_o, ws_ml);
  launch_pdl(attn_combine_kernel, dim3(KV_HEADS * NQ), dim3(HEAD_DIM), stream, static_cast<const float*>(ws_o),
             static_cast<const float*>(ws_ml), chunks, out);
}

void add_rmsnorm(__nv_bfloat16* res, const float* delta, const __nv_bfloat16* w, __nv_bfloat16* xn, int rows,
                 cudaStream_t stream) {
  if (rows <= 0) return;
  launch_pdl(add_rmsnorm_kernel, dim3(rows), dim3(ROW_THREADS), stream, res, delta, w, xn);
}

void silu_mul(const float* gu, int rows, int inter, __nv_bfloat16* out, cudaStream_t stream) {
  const size_t n = size_t(rows) * inter;
  if (n == 0) return;
  launch_pdl(silu_mul_kernel, dim3(unsigned((n + 255) / 256)), dim3(256), stream, gu, rows, inter, out);
}

void fill(float* x, std::size_t n, float value, cudaStream_t stream) {
  if (n == 0) return;
  const unsigned blocks = unsigned(std::min<std::size_t>((n + 255) / 256, 4096));
  fill_kernel<<<blocks, 256, 0, stream>>>(x, n, value);
  check_launch();
}

}  // namespace ingine::kernels::draft

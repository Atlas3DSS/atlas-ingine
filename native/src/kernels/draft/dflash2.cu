// Native Apache-2.0 DFlash2 block-diffusion drafter kernels (incoai/Inkling-2-Mini-DFlash2, rev 4f9088ea).
// All hot-path kernels are PDL-launched and CUDA-graph capturable (zero host syncs, zero allocations).
// Architecture & math match DFlash2DraftModel: 5 sliding-attention layers (H=2304, 32Q/8KV, head_dim=128,
// window=2048, RoPE theta=10000), 2-tap grouped dynamic causal conv (group_size=16, 144 groups) around
// attention and MLP, and rank-256 top-16 left-to-right conditional candidate selector with post-correction
// tanh softcapping (30.0).
#include "kernels/common/l2.cuh"
#include "kernels/common/mma_sm70.cuh"
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ingine::kernels::draft {
namespace {

constexpr int ROW_THREADS = H / 8;  // 288
static_assert(H % 8 == 0 && ROW_THREADS % 32 == 0);
static_assert(DF2_HEADS / KV_HEADS == 4 && BLOCK == 8 && HEAD_DIM == 128);
static_assert(DF2_RANK == 256 && DF2_TOPK == 16 && DF2_CONV_GROUP == 16);

__device__ __forceinline__ uint4 pack8(const float* f) { return act::pack8(f); }

__device__ __forceinline__ float rms_scale(const float (&x)[8], float* scratch) {
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) ss += x[j] * x[j];
  const float total = block_sum(ss, scratch);
  return rsqrtf(total / float(H) + EPS);
}

__device__ __forceinline__ uint4 rms_norm(const float (&x)[8], const uint4& wv, float* scratch) {
  const float rs = rms_scale(x, scratch);
  float w[8], y[8];
  unpack8(wv, w);
#pragma unroll
  for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(__fmul_rn(x[j], rs), w[j]);
  return pack8(y);
}

__device__ __forceinline__ void load_embed_8(const ActT* embed, bool embed_q4, int token_id, int c, float (&v)[8]) {
  if (embed_q4) {
    const uint8_t* row = reinterpret_cast<const uint8_t*>(embed) + size_t(token_id) * 1296;
    const uint32_t nib4 = *reinterpret_cast<const uint32_t*>(row + (c >> 1));
    const float d = __half2float(reinterpret_cast<const __half*>(row + 1152)[c >> 5]);
    const float m8d = -8.f * d;
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = fmaf(float((nib4 >> (4 * j)) & 0xfu), d, m8d);
  } else {
    unpack8(*reinterpret_cast<const uint4*>(embed + size_t(token_id) * H + c), v);
  }
}

__device__ __forceinline__ float norm_rope_head_theta(float x, const ActT* __restrict__ w, int pos, float rope_theta,
                                                      float* scratch, float* pair) {
  const int d = threadIdx.x;
  const float ss = block_sum(x * x, scratch);
  const float rs = rsqrtf(ss / float(HEAD_DIM) + EPS);
  const float y = bf_round(__fmul_rn(__fmul_rn(x, rs), bf(w[d])));
  pair[d] = y;
  __syncthreads();
  const int half = HEAD_DIM / 2, i = d & (half - 1);
  const float inv = powf(rope_theta, (-float(i) * 2.f) / float(HEAD_DIM));
  const float ang = __fmul_rn(float(pos), inv);
  const float c = cosf(ang), s = sinf(ang);
  const float x1 = pair[i], x2 = pair[i + half];
  const float o = d < half ? __fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s))
                           : __fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s));
  __syncthreads();
  return bf_round(o);
}

// ---- embedding / rmsnorm / commit prep -----------------------------------------------------------------

__global__ void __launch_bounds__(ROW_THREADS) df2_embed_token_kernel(const ActT* embed, const float* embed_norm,
                                                                      int token_id, bool embed_q4, ActT* out) {
  __shared__ float scratch[33];
  const int c = 8 * threadIdx.x;
  float v[8];
  load_embed_8(embed, embed_q4, token_id, c, v);
  const float rs = rms_scale(v, scratch);
  const float4 e0 = *reinterpret_cast<const float4*>(embed_norm + c);
  const float4 e1 = *reinterpret_cast<const float4*>(embed_norm + c + 4);
  const float en[8] = {e0.x, e0.y, e0.z, e0.w, e1.x, e1.y, e1.z, e1.w};
  float y[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) y[j] = bf_round(__fmul_rn(__fmul_rn(v[j], rs), en[j]));
  *reinterpret_cast<uint4*>(out + c) = pack8(y);
}

__global__ void __launch_bounds__(ROW_THREADS) df2_rmsnorm_act_kernel(const ActT* in, const ActT* __restrict__ w,
                                                                      ActT* out) {
  __shared__ float scratch[33];
  pdl_trigger();
  const int r = blockIdx.x, c = 8 * threadIdx.x;
  const uint4 wv = ld_weight(w + c);
  pdl_wait();
  float x[8];
  unpack8(*reinterpret_cast<const uint4*>(in + size_t(r) * H + c), x);
  *reinterpret_cast<uint4*>(out + size_t(r) * H + c) = rms_norm(x, wv, scratch);
}

__global__ void __launch_bounds__(ROW_THREADS) df2_rmsnorm_f32_kernel(const float* in, const ActT* __restrict__ w,
                                                                      ActT* out) {
  __shared__ float scratch[33];
  pdl_trigger();
  const int r = blockIdx.x, c = 8 * threadIdx.x;
  const uint4 wv = ld_weight(w + c);
  pdl_wait();
  const float4 d0 = *reinterpret_cast<const float4*>(in + size_t(r) * H + c);
  const float4 d1 = *reinterpret_cast<const float4*>(in + size_t(r) * H + c + 4);
  const float x[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
  *reinterpret_cast<uint4*>(out + size_t(r) * H + c) = rms_norm(x, wv, scratch);
}

// grid (BLOCK, 4): copies tap j (0..3) of row r < ncommit into taps_out[r, j*H..], zeroing rows >= ncommit.
__global__ void __launch_bounds__(ROW_THREADS) df2_commit_prep_kernel(const DraftParams* p, ActT* taps_out) {
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, tap = blockIdx.y, c = 8 * threadIdx.x;
  uint4* dst = reinterpret_cast<uint4*>(taps_out + (size_t(r) * 4 + tap) * H + c);
  if (r < p->ncommit) {
    *dst = *reinterpret_cast<const uint4*>(p->commit_feats + (size_t(r) * 4 + tap) * H + c);
  } else {
    *dst = make_uint4(0, 0, 0, 0);
  }
}

// grid (BLOCK, 5): y = 0..3 copies tap y of commit_feats[0..n-1] into taps_out; y = 4 embeds block rows & in_norm.
__global__ void __launch_bounds__(ROW_THREADS) df2_round_begin_kernel(DraftParams* p, RoundInputs in,
                                                                      const ActT* embed, const float* embed_norm,
                                                                      const ActT* mask_embed, const ActT* in_norm,
                                                                      bool embed_q4, ActT* taps_out, ActT* res,
                                                                      ActT* xn) {
  __shared__ float scratch[33];
  pdl_trigger();
  const int r = blockIdx.x, part = blockIdx.y, c = 8 * threadIdx.x;
  uint4 wv = make_uint4(0, 0, 0, 0);
  uint4 aux[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
  if (part == 4) {
    wv = ld_weight(in_norm + c);
    if (r == 0) {
      aux[0] = ld_weight(embed_norm + c);
      aux[1] = ld_weight(embed_norm + c + 4);
    } else {
      aux[0] = ld_weight(mask_embed + c);
    }
  }
  pdl_wait();
  const int m = *in.m, start = p->round_start;
  const int n = *in.accepted + 1;
  const ActT* feats = in.taps[0];
  for (int i = 0; i < in.nwidths; ++i)
    if (in.widths[i] == m) feats = in.taps[i];
  const int anchor = p->temperature > 0.f ? *in.next_sampled : *in.next_greedy;
  if (r == 0 && part == 0 && threadIdx.x == 0) {
    p->commit_feats = feats;
    p->ncommit = n;
    p->start = start;
    p->anchor = anchor;
    p->P = start + n;
  }
  if (part < 4) {
    uint4* dst = reinterpret_cast<uint4*>(taps_out + (size_t(r) * 4 + part) * H + c);
    if (r < n) *dst = *reinterpret_cast<const uint4*>(feats + (size_t(r) * 4 + part) * H + c);
    else *dst = make_uint4(0, 0, 0, 0);
    return;
  }
  float x[8];
  if (r == 0) {
    float v[8];
    load_embed_8(embed, embed_q4, anchor, c, v);
    const float rs = rms_scale(v, scratch);
    const float en[8] = {__uint_as_float(aux[0].x), __uint_as_float(aux[0].y), __uint_as_float(aux[0].z),
                         __uint_as_float(aux[0].w), __uint_as_float(aux[1].x), __uint_as_float(aux[1].y),
                         __uint_as_float(aux[1].z), __uint_as_float(aux[1].w)};
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = bf_round(__fmul_rn(__fmul_rn(v[j], rs), en[j]));
  } else {
    unpack8(aux[0], x);
  }
  *reinterpret_cast<uint4*>(res + size_t(r) * H + c) = pack8(x);
  *reinterpret_cast<uint4*>(xn + size_t(r) * H + c) = rms_norm(x, wv, scratch);
}

// ---- context K/V stores --------------------------------------------------------------------------------

__global__ void __launch_bounds__(HEAD_DIM) df2_ctx_kv_store_kernel(const ActT* kv, const DraftParams* p, int row0,
                                                                    int keep_from, const ActT* __restrict__ k_norm,
                                                                    ActT* k_cache, ActT* v_cache, int size,
                                                                    std::int32_t* ctx_pos, float rope_theta) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
  const int r = keep_from + blockIdx.x, g = blockIdx.y, d = threadIdx.x;
  const int pos = p->start + row0 + r;
  const int slot = pos % size;
  const ActT* row = kv + size_t(r) * 2 * KV_DIM;
  const float k = norm_rope_head_theta(bf(row[g * HEAD_DIM + d]), k_norm, pos, rope_theta, scratch, pair);
  const size_t at = (size_t(slot) * KV_HEADS + g) * HEAD_DIM + d;
  k_cache[at] = act::from_float(k);
  v_cache[at] = row[KV_DIM + g * HEAD_DIM + d];
  if (ctx_pos != nullptr && g == 0 && d == 0) ctx_pos[slot] = pos;
}

// grid (BLOCK, KV_HEADS, DF2_LAYERS), block 128
__global__ void __launch_bounds__(HEAD_DIM) df2_commit_store_kernel(const float* kv, const DraftParams* p,
                                                                    DFlash2CtxLayers L, float rope_theta) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, g = blockIdx.y, l = blockIdx.z, d = threadIdx.x;
  if (r >= p->ncommit) return;
  const int pos = p->start + r;
  const int slot = pos % DF2_WINDOW;
  const float* row = kv + size_t(r) * DF2_LAYERS * 2 * KV_DIM + size_t(l) * 2 * KV_DIM;
  const float k = norm_rope_head_theta(row[g * HEAD_DIM + d], L.k_norm[l], pos, rope_theta, scratch, pair);
  const size_t at = (size_t(slot) * KV_HEADS + g) * HEAD_DIM + d;
  L.k_cache[l][at] = act::from_float(k);
  L.v_cache[l][at] = act::from_float(row[KV_DIM + g * HEAD_DIM + d]);
  if (L.ctx_pos[l] != nullptr && g == 0 && d == 0) L.ctx_pos[l][slot] = pos;
}

// ---- dynamic grouped causal 2-tap conv -----------------------------------------------------------------
// base_kernel: [2 sides, 2 taps, H]. dyn: [rows, 2 sides * 2 taps * 144 groups].
// For h = 16*g + j: each thread owns 8 elements belonging to ONE group g = c / 16 = threadIdx.x >> 1!

__global__ void __launch_bounds__(ROW_THREADS) df2_conv_prepare_kernel(const ActT* x, const ActT* __restrict__ base_k,
                                                                       const float* dyn, ActT* out) {
  pdl_trigger();
  const int r = blockIdx.x, c = 8 * threadIdx.x, g = threadIdx.x >> 1;
  const uint4 b0v = ld_weight(base_k + c);
  const uint4 b1v = ld_weight(base_k + H + c);
  pdl_wait();
  float b0[8], b1[8], cur[8], y[8];
  unpack8(b0v, b0);
  unpack8(*reinterpret_cast<const uint4*>(x + size_t(r) * H + c), cur);
  const float d0 = dyn[size_t(r) * DF2_CONV_PROJ_ROWS + g];
#pragma unroll
  for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(__fadd_rn(b0[j], d0), cur[j]);
  if (r > 0) {
    float prev[8];
    unpack8(b1v, b1);
    unpack8(*reinterpret_cast<const uint4*>(x + size_t(r - 1) * H + c), prev);
    const float d1 = dyn[size_t(r) * DF2_CONV_PROJ_ROWS + DF2_CONV_GROUPS + g];
#pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = fmaf(__fadd_rn(b1[j], d1), prev[j], y[j]);
  }
  *reinterpret_cast<uint4*>(out + size_t(r) * H + c) = pack8(y);
}

__global__ void __launch_bounds__(ROW_THREADS) df2_conv_finish_add_rmsnorm_kernel(
    ActT* res, const float* delta, const ActT* __restrict__ base_k, const float* dyn, const ActT* __restrict__ norm_w,
    ActT* xn) {
  __shared__ float scratch[33];
  pdl_trigger();
  const int r = blockIdx.x, c = 8 * threadIdx.x, g = threadIdx.x >> 1;
  const uint4 b0v = ld_weight(base_k + 2 * H + c);
  const uint4 b1v = ld_weight(base_k + 3 * H + c);
  const uint4 wv = ld_weight(norm_w + c);
  pdl_wait();
  const size_t at = size_t(r) * H + c;
  float b0[8], rv[8];
  unpack8(b0v, b0);
  unpack8(*reinterpret_cast<const uint4*>(res + at), rv);
  const float4 dc0 = *reinterpret_cast<const float4*>(delta + at);
  const float4 dc1 = *reinterpret_cast<const float4*>(delta + at + 4);
  const float cur[8] = {dc0.x, dc0.y, dc0.z, dc0.w, dc1.x, dc1.y, dc1.z, dc1.w};
  const float d0 = dyn[size_t(r) * DF2_CONV_PROJ_ROWS + 2 * DF2_CONV_GROUPS + g];
  float conv[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) conv[j] = __fmul_rn(__fadd_rn(b0[j], d0), cur[j]);
  if (r > 0) {
    float b1[8];
    unpack8(b1v, b1);
    const size_t pat = size_t(r - 1) * H + c;
    const float4 dp0 = *reinterpret_cast<const float4*>(delta + pat);
    const float4 dp1 = *reinterpret_cast<const float4*>(delta + pat + 4);
    const float prev[8] = {dp0.x, dp0.y, dp0.z, dp0.w, dp1.x, dp1.y, dp1.z, dp1.w};
    const float d1 = dyn[size_t(r) * DF2_CONV_PROJ_ROWS + 3 * DF2_CONV_GROUPS + g];
#pragma unroll
    for (int j = 0; j < 8; ++j) conv[j] = fmaf(__fadd_rn(b1[j], d1), prev[j], conv[j]);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) rv[j] = bf_round(__fadd_rn(rv[j], conv[j]));
  *reinterpret_cast<uint4*>(res + at) = pack8(rv);
  *reinterpret_cast<uint4*>(xn + at) = rms_norm(rv, wv, scratch);
}

// ---- QKV norm + RoPE (32 Q + 8 K + 8 V) ---------------------------------------------------------------

__global__ void __launch_bounds__(HEAD_DIM) df2_qkv_rope_kernel(const float* qkv, const ActT* __restrict__ q_norm,
                                                                const ActT* __restrict__ k_norm, const DraftParams* p,
                                                                ActT* q_out, ActT* k_cache, ActT* v_cache,
                                                                float rope_theta) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
  const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  const int pos = p->P + r;
  const float* row = qkv + size_t(r) * DF2_QKV_ROWS;
  const int slot = DF2_WINDOW + r;
  if (h < DF2_HEADS) {
    const float q = norm_rope_head_theta(row[h * HEAD_DIM + d], q_norm, pos, rope_theta, scratch, pair);
    q_out[(size_t(r) * DF2_HEADS + h) * HEAD_DIM + d] = act::from_float(q);
  } else if (h < DF2_HEADS + KV_HEADS) {
    const int g = h - DF2_HEADS;
    const float k = norm_rope_head_theta(row[DF2_Q_DIM + g * HEAD_DIM + d], k_norm, pos, rope_theta, scratch, pair);
    k_cache[(size_t(slot) * KV_HEADS + g) * HEAD_DIM + d] = act::from_float(k);
  } else {
    const int g = h - DF2_HEADS - KV_HEADS;
    v_cache[(size_t(slot) * KV_HEADS + g) * HEAD_DIM + d] = act::from_float(row[DF2_Q_DIM + KV_DIM + g * HEAD_DIM + d]);
  }
}

// ---- DFlash2 bidirectional local sliding-window block attention (QH = 4, S = 2048) ---------------------

constexpr int DF2_QH = DF2_HEADS / KV_HEADS;  // 4 query heads per KV head (all 4 warps active!)
constexpr int DF2_NQ = DF2_QH * BLOCK;        // 32 queries per KV head
constexpr int DF2_ATT_TILE = 16;
constexpr int DF2_ATT_THREADS = 128;          // 4 warps: warp w owns query head 4*g + w
constexpr int DF2_ATT_TILES_PER_CHUNK = 4;    // 64 keys per chunk -> 33 chunks for 2056 keys
constexpr int DF2_ATT_CHUNK_KEYS = DF2_ATT_TILE * DF2_ATT_TILES_PER_CHUNK;
constexpr int DF2_ATT_CHUNKS = (DF2_LOCAL_SLOTS + DF2_ATT_CHUNK_KEYS - 1) / DF2_ATT_CHUNK_KEYS;  // 33
static_assert(DF2_QH == 4 && DF2_ATT_CHUNKS <= HEAD_DIM);

__device__ __forceinline__ void df2_mma_act(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  mma::mma_act(c, a[0], a[1], a[2], a[3], b[0], b[1]);
}
__device__ __forceinline__ void df2_ldmatrix_x4(uint32_t (&r)[4], const void* row) {
  mma::ldmatrix_x4(r[0], r[1], r[2], r[3], static_cast<unsigned>(__cvta_generic_to_shared(row)));
}
__device__ __forceinline__ void df2_ldmatrix_x4_trans(uint32_t (&r)[4], const void* row) {
  mma::ldmatrix_x4_trans(r[0], r[1], r[2], r[3], static_cast<unsigned>(__cvta_generic_to_shared(row)));
}
__device__ __forceinline__ void df2_ldmatrix_x2(uint32_t (&r)[2], const void* row) {
  mma::ldmatrix_x2(r[0], r[1], static_cast<unsigned>(__cvta_generic_to_shared(row)));
}
__device__ __forceinline__ uint4 df2_ld_cache(const void* p) {
  uint4 r;
  asm volatile("ld.global" INGINE_L1_NO_ALLOCATE ".v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p)
               : "memory");
  return r;
}

struct Df2TileRegs {
  uint4 k[2], v[2];
  int cp;
};

__global__ void __launch_bounds__(DF2_ATT_THREADS) df2_attn_partial_kernel(
    const ActT* q, const ActT* k_cache, const ActT* v_cache, const std::int32_t* ctx_pos, const DraftParams* p,
    float* ws_o, float* ws_ml) {
  __shared__ __align__(16) ActT qs[DF2_QH][BLOCK][HEAD_DIM];
  __shared__ __align__(16) ActT kt[2][DF2_ATT_TILE][HEAD_DIM];
  __shared__ __align__(16) ActT vt[2][DF2_ATT_TILE][HEAD_DIM];
  __shared__ __align__(16) ActT pt[DF2_QH][BLOCK][DF2_ATT_TILE];
  __shared__ uint16_t vmask_row[2][BLOCK];  // exact per-query sliding mask for |p_key - p_query| < 2048
  pdl_trigger();
  pdl_wait();
  const int chunk = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int gl = lane >> 2, t = lane & 3, lm = lane >> 3, lr = lane & 7;
  const int k0 = chunk * DF2_ATT_CHUNK_KEYS, kend = min(k0 + DF2_ATT_CHUNK_KEYS, DF2_LOCAL_SLOTS);
  const int P = p->P;
  constexpr float scale = 0.08838834764831845f;  // 1 / sqrt(128)

  auto load = [&](Df2TileRegs& x, int t0) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int idx = tid + DF2_ATT_THREADS * j;
      const int key = t0 + (idx >> 4), piece = idx & 15;
      const size_t at = (size_t(key) * KV_HEADS + g) * HEAD_DIM + 8 * piece;
      const bool live = key < kend;
      x.k[j] = live ? df2_ld_cache(k_cache + at) : make_uint4(0, 0, 0, 0);
      x.v[j] = live ? df2_ld_cache(v_cache + at) : make_uint4(0, 0, 0, 0);
    }
    x.cp = -1;
    if (tid < DF2_ATT_TILE && t0 + tid < DF2_WINDOW) x.cp = ctx_pos[t0 + tid];
  };
  Df2TileRegs ra, rb;
  if (k0 < kend) load(ra, k0);
  if (k0 + DF2_ATT_TILE < kend) load(rb, k0 + DF2_ATT_TILE);

  // Load 4 query heads x 8 rows x 128 dims = 512 uint4s (4 per thread)
  {
    uint4 qv[DF2_QH];
#pragma unroll
    for (int j = 0; j < DF2_QH; ++j) {
      const int idx = tid + DF2_ATT_THREADS * j, hh = idx >> 7, r = (idx >> 4) & 7, piece = idx & 15;
      qv[j] = *reinterpret_cast<const uint4*>(q + (size_t(r) * DF2_HEADS + g * DF2_QH + hh) * HEAD_DIM + 8 * piece);
    }
#pragma unroll
    for (int j = 0; j < DF2_QH; ++j) {
      const int idx = tid + DF2_ATT_THREADS * j;
      *reinterpret_cast<uint4*>(&qs[idx >> 7][(idx >> 4) & 7][8 * (idx & 15)]) = qv[j];
    }
  }
  __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
  const int v_qp = (lane >> 2) & 3;
  const int v_idx = (lane & 3) | ((lane & 16) >> 2);
  const int v_row_a = (v_qp >= 2 ? 8 : 0) + v_idx;
  const int v_kh = (v_qp & 1) * 4;
  uint32_t qb_v[HEAD_DIM / 16][4];
#pragma unroll
  for (int s = 0; s < HEAD_DIM / 16; ++s) {
    const uint2 q0 = mma::ld_smem_u64(static_cast<unsigned>(__cvta_generic_to_shared(&qs[w][v_idx][16 * s + v_kh])));
    const uint2 q1 = mma::ld_smem_u64(static_cast<unsigned>(__cvta_generic_to_shared(&qs[w][v_idx][16 * s + 8 + v_kh])));
    qb_v[s][0] = q0.x, qb_v[s][1] = q0.y, qb_v[s][2] = q1.x, qb_v[s][3] = q1.y;
  }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
  // Pascal / SIMT path reads qs directly from shared memory.
#else
  uint32_t qb[HEAD_DIM / 16][2];
#pragma unroll
  for (int s = 0; s < HEAD_DIM / 16; ++s) df2_ldmatrix_x2(qb[s], &qs[w][lr][16 * s + 8 * (lm & 1)]);
#endif

  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
  float oacc[HEAD_DIM / 16][4];
#pragma unroll
  for (int d = 0; d < HEAD_DIM / 16; ++d)
#pragma unroll
    for (int i = 0; i < 4; ++i) oacc[d][i] = 0.f;

  auto tile = [&](Df2TileRegs& x, int t0, int buf) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int idx = tid + DF2_ATT_THREADS * j;
      const int key = idx >> 4, piece = idx & 15;
      *reinterpret_cast<uint4*>(&kt[buf][key][8 * piece]) = x.k[j];
      *reinterpret_cast<uint4*>(&vt[buf][key][8 * piece]) = x.v[j];
    }
    if (w == 0) {
      const int j = t0 + lane;
      const bool in_bounds = lane < DF2_ATT_TILE && j < kend;
      const bool is_block_key = j >= DF2_WINDOW;
#pragma unroll
      for (int r = 0; r < BLOCK; ++r) {
        // Exact DFlash2 local sliding mask: allowed(p_query = P + r, p_key) iff |p_key - p_query| < DF2_WINDOW.
        // Block keys (j >= DF2_WINDOW) have |p_key - p_query| <= 7 < 2048; ring keys have cp < P.
        const bool vr = in_bounds && (is_block_key || (x.cp >= 0 && x.cp < P && x.cp + DF2_WINDOW > P + r));
        const unsigned mask = __ballot_sync(0xffffffffu, vr);
        if (lane == 0) vmask_row[buf][r] = static_cast<uint16_t>(mask);
      }
    }
    __syncthreads();
    if (t0 + 2 * DF2_ATT_TILE < kend) load(x, t0 + 2 * DF2_ATT_TILE);

    float sc[4] = {0.f, 0.f, 0.f, 0.f};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
    float sc8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < HEAD_DIM / 16; ++s) {
      const uint2 k0s = mma::ld_smem_u64(static_cast<unsigned>(__cvta_generic_to_shared(&kt[buf][v_row_a][16 * s + v_kh])));
      const uint2 k1s = mma::ld_smem_u64(static_cast<unsigned>(__cvta_generic_to_shared(&kt[buf][v_row_a][16 * s + 8 + v_kh])));
      mma::sm70_mma_f16_direct(sc8, k0s.x, k0s.y, k1s.x, k1s.y, qb_v[s][0], qb_v[s][1], qb_v[s][2], qb_v[s][3]);
    }
    mma::sm70_reduce_acc8_to_d4(sc8, sc);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
#pragma unroll
    for (int s = 0; s < HEAD_DIM / 16; ++s) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const uint4 ka = *reinterpret_cast<const uint4*>(&kt[buf][gl][16 * s + 8 * c]);
        const uint4 kb = *reinterpret_cast<const uint4*>(&kt[buf][gl + 8][16 * s + 8 * c]);
        const uint4 qa = *reinterpret_cast<const uint4*>(&qs[w][2 * t][16 * s + 8 * c]);
        const uint4 qb4 = *reinterpret_cast<const uint4*>(&qs[w][2 * t + 1][16 * s + 8 * c]);
        const uint32_t kaw[4] = {ka.x, ka.y, ka.z, ka.w}, kbw[4] = {kb.x, kb.y, kb.z, kb.w};
        const uint32_t qaw[4] = {qa.x, qa.y, qa.z, qa.w}, qbw[4] = {qb4.x, qb4.y, qb4.z, qb4.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 fka = act::unpack2(kaw[i]);
          const float2 fkb = act::unpack2(kbw[i]);
          const float2 fqa = act::unpack2(qaw[i]);
          const float2 fqb = act::unpack2(qbw[i]);
          sc[0] = fmaf(fka.y, fqa.y, fmaf(fka.x, fqa.x, sc[0]));
          sc[1] = fmaf(fka.y, fqb.y, fmaf(fka.x, fqb.x, sc[1]));
          sc[2] = fmaf(fkb.y, fqa.y, fmaf(fkb.x, fqa.x, sc[2]));
          sc[3] = fmaf(fkb.y, fqb.y, fmaf(fkb.x, fqb.x, sc[3]));
        }
      }
    }
#else
#pragma unroll
    for (int s = 0; s < HEAD_DIM / 16; ++s) {
      uint32_t a[4];
      df2_ldmatrix_x4(a, &kt[buf][8 * (lm & 1) + lr][16 * s + 8 * (lm >> 1)]);
      df2_mma_act(sc, a, qb[s]);
    }
#endif
    float al[2];
    ActT pb16[2][2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const unsigned mask = vmask_row[buf][2 * t + e];
      const bool valid[2] = {((mask >> gl) & 1u) != 0, ((mask >> (gl + 8)) & 1u) != 0};
      float xs[2], mt = -INFINITY;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        xs[kk] = valid[kk] ? sc[2 * kk + e] * scale : -INFINITY;
        mt = fmaxf(mt, xs[kk]);
      }
#pragma unroll
      for (int sh = 4; sh < 32; sh <<= 1) mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, sh));
      const float m_new = fmaxf(m_run[e], mt);
      al[e] = m_new == m_run[e] ? 1.f : (m_run[e] == -INFINITY ? 0.f : __expf(m_run[e] - m_new));
      float ps = 0.f;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        pb16[e][kk] = act::from_float(valid[kk] ? __expf(xs[kk] - m_new) : 0.f);
        ps += act::to_float(pb16[e][kk]);
      }
#pragma unroll
      for (int sh = 4; sh < 32; sh <<= 1) ps += __shfl_xor_sync(0xffffffffu, ps, sh);
      l_run[e] = l_run[e] * al[e] + ps;
      m_run[e] = m_new;
    }
#pragma unroll
    for (int e = 0; e < 2; ++e)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) pt[w][2 * t + e][gl + 8 * kk] = pb16[e][kk];
    __syncwarp();
    const bool rescale = __any_sync(0xffffffffu, al[0] != 1.f || al[1] != 1.f);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
    const uint2 p0 = mma::ld_smem_u64(static_cast<unsigned>(__cvta_generic_to_shared(&pt[w][v_idx][v_kh])));
    const uint2 p1 = mma::ld_smem_u64(static_cast<unsigned>(__cvta_generic_to_shared(&pt[w][v_idx][8 + v_kh])));
#pragma unroll
    for (int d = 0; d < HEAD_DIM / 16; ++d) {
      if (rescale) {
        oacc[d][0] *= al[0];
        oacc[d][1] *= al[1];
        oacc[d][2] *= al[0];
        oacc[d][3] *= al[1];
      }
      const int dc = 16 * d + v_row_a;
      const unsigned v0 = mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][v_kh + 0][dc])));
      const unsigned v1 = mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][v_kh + 1][dc])));
      const unsigned v2 = mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][v_kh + 2][dc])));
      const unsigned v3 = mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][v_kh + 3][dc])));
      const unsigned v8 = mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][v_kh + 8][dc])));
      const unsigned v9 = mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][v_kh + 9][dc])));
      const unsigned v10 = mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][v_kh + 10][dc])));
      const unsigned v11 = mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][v_kh + 11][dc])));
      float ov[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      mma::sm70_mma_f16_direct(ov, v0 | (v1 << 16), v2 | (v3 << 16), v8 | (v9 << 16), v10 | (v11 << 16), p0.x, p0.y,
                               p1.x, p1.y);
      mma::sm70_reduce_acc8_to_d4(ov, oacc[d]);
    }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
    const uint4 pq0a = *reinterpret_cast<const uint4*>(&pt[w][2 * t][0]);
    const uint4 pq0b = *reinterpret_cast<const uint4*>(&pt[w][2 * t][8]);
    const uint4 pq1a = *reinterpret_cast<const uint4*>(&pt[w][2 * t + 1][0]);
    const uint4 pq1b = *reinterpret_cast<const uint4*>(&pt[w][2 * t + 1][8]);
    const uint32_t pw0[8] = {pq0a.x, pq0a.y, pq0a.z, pq0a.w, pq0b.x, pq0b.y, pq0b.z, pq0b.w};
    const uint32_t pw1[8] = {pq1a.x, pq1a.y, pq1a.z, pq1a.w, pq1b.x, pq1b.y, pq1b.z, pq1b.w};
    float2 fp0[8], fp1[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      fp0[k] = act::unpack2(pw0[k]);
      fp1[k] = act::unpack2(pw1[k]);
    }
#pragma unroll
    for (int d = 0; d < HEAD_DIM / 16; ++d) {
      if (rescale) {
        oacc[d][0] *= al[0];
        oacc[d][1] *= al[1];
        oacc[d][2] *= al[0];
        oacc[d][3] *= al[1];
      }
      const int da = 16 * d + gl, db = da + 8;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float va0 = act::bits_to_float(mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][2 * k][da]))));
        const float va1 = act::bits_to_float(mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][2 * k + 1][da]))));
        const float vb0 = act::bits_to_float(mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][2 * k][db]))));
        const float vb1 = act::bits_to_float(mma::ld_smem_u16(static_cast<unsigned>(__cvta_generic_to_shared(&vt[buf][2 * k + 1][db]))));
        oacc[d][0] = fmaf(va1, fp0[k].y, fmaf(va0, fp0[k].x, oacc[d][0]));
        oacc[d][1] = fmaf(va1, fp1[k].y, fmaf(va0, fp1[k].x, oacc[d][1]));
        oacc[d][2] = fmaf(vb1, fp0[k].y, fmaf(vb0, fp0[k].x, oacc[d][2]));
        oacc[d][3] = fmaf(vb1, fp1[k].y, fmaf(vb0, fp1[k].x, oacc[d][3]));
      }
    }
#else
    uint32_t pf[2];
    df2_ldmatrix_x2(pf, &pt[w][lr][8 * (lm & 1)]);
#pragma unroll
    for (int d = 0; d < HEAD_DIM / 16; ++d) {
      if (rescale) {
        oacc[d][0] *= al[0];
        oacc[d][1] *= al[1];
        oacc[d][2] *= al[0];
        oacc[d][3] *= al[1];
      }
      uint32_t a[4];
      df2_ldmatrix_x4_trans(a, &vt[buf][8 * (lm >> 1) + lr][16 * d + 8 * (lm & 1)]);
      df2_mma_act(oacc[d], a, pf);
    }
#endif
  };
  for (int t0 = k0; t0 < kend; t0 += 2 * DF2_ATT_TILE) {
    tile(ra, t0, 0);
    if (t0 + DF2_ATT_TILE < kend) tile(rb, t0 + DF2_ATT_TILE, 1);
  }
  const size_t base = size_t(chunk) * KV_HEADS + g;
  const int qi = BLOCK * w + 2 * t;
#pragma unroll
  for (int d = 0; d < HEAD_DIM / 16; ++d) {
    float* at = ws_o + (base * DF2_NQ + qi) * HEAD_DIM + 16 * d + gl;
    at[0] = oacc[d][0];
    at[HEAD_DIM] = oacc[d][1];
    at[8] = oacc[d][2];
    at[HEAD_DIM + 8] = oacc[d][3];
  }
  if (gl == 0) {
    float* at = ws_ml + (base * DF2_NQ + qi) * 2;
    *reinterpret_cast<float4*>(at) = make_float4(m_run[0], l_run[0], m_run[1], l_run[1]);
  }
}

__global__ void __launch_bounds__(HEAD_DIM) df2_attn_combine_kernel(const float* ws_o, const float* ws_ml, ActT* out) {
  __shared__ float scratch[33];
  __shared__ float f[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
  const int g = blockIdx.x / DF2_NQ, qi = blockIdx.x % DF2_NQ, d = threadIdx.x;
  float mc = -INFINITY, lc = 0.f;
  if (d < DF2_ATT_CHUNKS) {
    const float2 ml = *reinterpret_cast<const float2*>(ws_ml + ((size_t(d) * KV_HEADS + g) * DF2_NQ + qi) * 2);
    mc = ml.x;
    lc = ml.y;
  }
  const float m = block_max(mc, scratch);
  const float fc = mc == -INFINITY ? 0.f : __expf(mc - m);
  f[d] = fc;
  const float l = block_sum(lc * fc, scratch);
  float o = 0.f;
#pragma unroll 8
  for (int c = 0; c < DF2_ATT_CHUNKS; ++c) o += ws_o[((size_t(c) * KV_HEADS + g) * DF2_NQ + qi) * HEAD_DIM + d] * f[c];
  const int hh = qi / BLOCK, r = qi % BLOCK, h = g * DF2_QH + hh;
  out[size_t(r) * DF2_Q_DIM + h * HEAD_DIM + d] = act::from_float(l > 0.f ? o / l : 0.f);
}

// ---- exact Top-16 selection ----------------------------------------------------------------------------

constexpr int DF2_TOPK_SLICES = 32;
constexpr int DF2_TOPK_THREADS = 256;

struct Df2TopkPair {
  float val;
  std::int32_t idx;
};

__device__ __forceinline__ unsigned long long warp_max_u64(unsigned long long v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

__device__ __forceinline__ unsigned long long block_max_u64(unsigned long long v, unsigned long long* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_max_u64(v);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  unsigned long long m = 0ull;
  for (int w = 0; w < nw; ++w) m = max(m, scratch[w]);
  return m;
}

// Phase 1: grid (DF2_TOPK_SLICES, rows), block 256. Each slice extracts its exact top-16 (largest val, lowest idx).
__global__ void __launch_bounds__(DF2_TOPK_THREADS) df2_topk16_slice_kernel(const float* base, int V,
                                                                            Df2TopkPair* partial) {
  __shared__ unsigned long long scratch[32];
  __shared__ std::int32_t picked[DF2_TOPK];
  pdl_trigger();
  pdl_wait();
  const int s = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
  const int per = (V + DF2_TOPK_SLICES - 1) / DF2_TOPK_SLICES;
  const int b = min(V, s * per), e = min(V, b + per);
  const float* x = base + size_t(row) * V;
  for (int k = 0; k < DF2_TOPK; ++k) {
    unsigned long long best = 0ull;  // pack_argmax(-inf, UINT_MAX) > 0, so 0ull means empty
    for (int i = b + tid; i < e; i += DF2_TOPK_THREADS) {
      bool used = false;
      for (int p = 0; p < k; ++p)
        if (picked[p] == i) used = true;
      if (!used) best = max(best, pack_argmax(x[i], static_cast<uint32_t>(i)));
    }
    const unsigned long long win = block_max_u64(best, scratch);
    if (tid == 0) {
      if (win == 0ull) {
        picked[k] = -1;
        partial[(size_t(row) * DF2_TOPK_SLICES + s) * DF2_TOPK + k] = Df2TopkPair{-INFINITY, -1};
      } else {
        const std::int32_t idx = static_cast<std::int32_t>(unpack_argmax_index(win));
        picked[k] = idx;
        partial[(size_t(row) * DF2_TOPK_SLICES + s) * DF2_TOPK + k] = Df2TopkPair{x[idx], idx};
      }
    }
    __syncthreads();
  }
}

// Phase 2: grid (rows), block 256. Reduces DF2_TOPK_SLICES * 16 = 512 partial candidates per row to exact top-16.
__global__ void __launch_bounds__(DF2_TOPK_THREADS) df2_topk16_reduce_kernel(const Df2TopkPair* partial,
                                                                             const std::int32_t* map, float* cand_val,
                                                                             std::int32_t* cand_idx) {
  __shared__ unsigned long long scratch[32];
  __shared__ float vals[DF2_TOPK_SLICES * DF2_TOPK];
  __shared__ std::int32_t idxs[DF2_TOPK_SLICES * DF2_TOPK];
  pdl_trigger();
  pdl_wait();
  const int row = blockIdx.x, tid = threadIdx.x;
  constexpr int N = DF2_TOPK_SLICES * DF2_TOPK;  // 512
  const Df2TopkPair* src = partial + size_t(row) * N;
  for (int i = tid; i < N; i += DF2_TOPK_THREADS) {
    vals[i] = src[i].val;
    idxs[i] = src[i].idx;
  }
  __syncthreads();
  for (int k = 0; k < DF2_TOPK; ++k) {
    unsigned long long best = 0ull;
    for (int i = tid; i < N; i += DF2_TOPK_THREADS) {
      if (idxs[i] >= 0) best = max(best, pack_argmax(vals[i], static_cast<uint32_t>(idxs[i])));
    }
    const unsigned long long win = block_max_u64(best, scratch);
    const std::int32_t win_idx = win == 0ull ? -1 : static_cast<std::int32_t>(unpack_argmax_index(win));
    // Clear the winning slot in shared memory so the next iteration picks the next candidate.
    for (int i = tid; i < N; i += DF2_TOPK_THREADS) {
      if (idxs[i] == win_idx && win_idx >= 0) {
        cand_val[size_t(row) * DF2_TOPK + k] = vals[i];
        cand_idx[size_t(row) * DF2_TOPK + k] = map != nullptr ? map[win_idx] : win_idx;
        idxs[i] = -1;
      }
    }
    __syncthreads();
  }
}

// ---- DFlash2 left-to-right conditional candidate selector ----------------------------------------------
// Single CTA, 512 threads = 16 warps (one warp per candidate c = 0..15; lane l owns 8 elements of rank 256).

__global__ void __launch_bounds__(512) df2_select_path_kernel(
    const float* cand_val, const std::int32_t* cand_idx, const float* g_proj, const ActT* __restrict__ pred_cb,
    const ActT* __restrict__ succ_cb, std::int32_t* prev_cand, float* draft_logits, int V, float softcap,
    bool softcap_after_correction, std::int32_t* tokens, float* conf, float* qstats, DraftParams* p) {
  __shared__ float sh_pg[DF2_RANK];
  __shared__ float sh_logits[DF2_TOPK];
  __shared__ std::int32_t sh_prev;
  pdl_trigger();
  pdl_wait();

  const int tid = threadIdx.x, c = tid >> 5, lane = tid & 31;  // c in 0..15, lane in 0..31

  // Clear previous draft's sparse entries (GAMMA * 16 = 112 entries).
  if (tid < GAMMA * DF2_TOPK) {
    const int i = tid / DF2_TOPK;
    const std::int32_t old_tok = prev_cand[tid];
    if (old_tok >= 0 && old_tok < V) draft_logits[size_t(i) * V + old_tok] = -INFINITY;
  }
  if (tid == 0) sh_prev = p->anchor;
  __syncthreads();

  const float T = p->temperature;
  const uint64_t seed = p->seed, draws = p->draws;

  for (int i = 0; i < GAMMA; ++i) {
    const int prev = sh_prev;
    if (tid < DF2_RANK) {
      sh_pg[tid] = __fmul_rn(bf(pred_cb[size_t(prev) * DF2_RANK + tid]), g_proj[size_t(i) * DF2_RANK + tid]);
    }
    __syncthreads();

    // Warp c computes edge score for candidate c (dot over 256 dims = 8 dims/lane).
    const std::int32_t tok = cand_idx[i * DF2_TOPK + c];
    float sv[8];
    unpack8(*reinterpret_cast<const uint4*>(succ_cb + size_t(tok) * DF2_RANK + 8 * lane), sv);
    float acc = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc = fmaf(sh_pg[8 * lane + e], sv[e], acc);
    acc = warp_sum(acc);
    if (lane == 0) {
      float u = cand_val[i * DF2_TOPK + c];
      if (!softcap_after_correction && softcap > 0.f) u = __fmul_rn(softcap, tanhf(__fdiv_rn(u, softcap)));
      const float raw = __fadd_rn(u, acc);
      const float logit =
          (softcap_after_correction && softcap > 0.f) ? __fmul_rn(softcap, tanhf(__fdiv_rn(raw, softcap))) : raw;
      sh_logits[c] = logit;
    }
    __syncthreads();

    // Warp 0 (lanes 0..15) selects winner, writes sparse draft_logits/prev_cand, and computes conf + qstats.
    if (c == 0) {
      const bool active = lane < DF2_TOPK;
      const float l = active ? sh_logits[lane] : -INFINITY;
      float score = l;
      if (active && T > 0.f) {
        score = __fadd_rn(__fdiv_rn(l, fmaxf(T, 1e-6f)), gumbel(seed, draws, uint32_t(i), uint32_t(lane)));
      }
      const unsigned long long packed = active ? pack_argmax(score, uint32_t(lane)) : 0ull;
      const unsigned long long best = warp_max_u64(packed);
      const int best_c = int(unpack_argmax_index(best));

      // Confidence = max_c softmax(sh_logits)[c] = 1 / sum_c exp(l_c - max_l).
      const float m = warp_max(l);
      const float s = warp_sum(active ? expf(l - m) : 0.f);

      float qm = 0.f, qs = 0.f;
      if (T > 0.f) {
        const float x = active ? __fdiv_rn(l, T) : -INFINITY;
        qm = warp_max(x);
        qs = warp_sum(active ? expf(__fsub_rn(x, qm)) : 0.f);
      }

      if (active) {
        const std::int32_t ctok = cand_idx[i * DF2_TOPK + lane];
        draft_logits[size_t(i) * V + ctok] = l;
        prev_cand[i * DF2_TOPK + lane] = ctok;
      }
      if (lane == 0) {
        const std::int32_t picked_tok = cand_idx[i * DF2_TOPK + best_c];
        tokens[i] = picked_tok;
        sh_prev = picked_tok;
        conf[i] = 1.f / s;
        qstats[2 * i] = qm;
        qstats[2 * i + 1] = qs;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    p->draws = draws + 1;
    p->round_start = p->P;
  }
}

void require_aligned(const void* ptr, const char* what) {
  if (reinterpret_cast<std::uintptr_t>(ptr) % 16 != 0) throw std::invalid_argument(what);
}

}  // namespace

void dflash2_embed_token(const ActT* embed, const float* embed_norm, int token_id, ActT* out, cudaStream_t stream,
                         bool embed_q4) {
  require_aligned(embed, "dflash2_embed_token: embed must be 16-byte aligned");
  require_aligned(out, "dflash2_embed_token: out must be 16-byte aligned");
  df2_embed_token_kernel<<<1, ROW_THREADS, 0, stream>>>(embed, embed_norm, token_id, embed_q4, out);
  CUDA_CHECK(cudaGetLastError());
}

void dflash2_rmsnorm(const ActT* in, int rows, const ActT* w, ActT* out, cudaStream_t stream) {
  if (rows <= 0) return;
  require_aligned(in, "dflash2_rmsnorm: in must be 16-byte aligned");
  require_aligned(out, "dflash2_rmsnorm: out must be 16-byte aligned");
  launch_pdl(df2_rmsnorm_act_kernel, dim3(rows), dim3(ROW_THREADS), stream, in, w, out);
}

void dflash2_rmsnorm_f32(const float* in, int rows, const ActT* w, ActT* out, cudaStream_t stream) {
  if (rows <= 0) return;
  require_aligned(in, "dflash2_rmsnorm_f32: in must be 16-byte aligned");
  require_aligned(out, "dflash2_rmsnorm_f32: out must be 16-byte aligned");
  launch_pdl(df2_rmsnorm_f32_kernel, dim3(rows), dim3(ROW_THREADS), stream, in, w, out);
}

void dflash2_commit_prep(const DraftParams* p, ActT* taps_out, cudaStream_t stream) {
  require_aligned(taps_out, "dflash2_commit_prep: taps_out must be 16-byte aligned");
  launch_pdl(df2_commit_prep_kernel, dim3(BLOCK, 4), dim3(ROW_THREADS), stream, p, taps_out);
}

void dflash2_ctx_kv_store(const ActT* kv, int n, const DraftParams* p, int row0, int keep_from, const ActT* k_norm,
                          ActT* k_cache, ActT* v_cache, int size, std::int32_t* ctx_pos, float rope_theta,
                          cudaStream_t stream) {
  if (keep_from < 0 || size <= 0) throw std::invalid_argument("dflash2_ctx_kv_store: bad keep_from/size");
  if (n - keep_from <= 0) return;
  launch_pdl(df2_ctx_kv_store_kernel, dim3(n - keep_from, KV_HEADS), dim3(HEAD_DIM), stream, kv, p, row0, keep_from,
             k_norm, k_cache, v_cache, size, ctx_pos, rope_theta);
}

void dflash2_commit_store(const float* kv, const DraftParams* p, const DFlash2CtxLayers& L, float rope_theta,
                          cudaStream_t stream) {
  launch_pdl(df2_commit_store_kernel, dim3(BLOCK, KV_HEADS, DF2_LAYERS), dim3(HEAD_DIM), stream, kv, p, L, rope_theta);
}

void dflash2_round_begin(DraftParams* p, const RoundInputs& in, const ActT* embed, const float* embed_norm,
                         const ActT* mask_embed, const ActT* in_norm, ActT* taps_out, ActT* res, ActT* xn,
                         cudaStream_t stream, bool embed_q4) {
  if (in.nwidths < 1 || in.nwidths > 3 || !in.m || !in.accepted || !in.next_greedy || !in.next_sampled)
    throw std::invalid_argument("dflash2_round_begin: incomplete round inputs");
  for (int i = 0; i < in.nwidths; ++i) require_aligned(in.taps[i], "dflash2_round_begin: taps must be 16-byte aligned");
  require_aligned(embed, "dflash2_round_begin: embed must be 16-byte aligned");
  require_aligned(taps_out, "dflash2_round_begin: taps_out must be 16-byte aligned");
  launch_pdl(df2_round_begin_kernel, dim3(BLOCK, 5), dim3(ROW_THREADS), stream, p, in, embed, embed_norm, mask_embed,
             in_norm, embed_q4, taps_out, res, xn);
}

void dflash2_conv_prepare(const ActT* x, const ActT* base_kernel, const float* dyn, ActT* out, int rows,
                          cudaStream_t stream) {
  if (rows <= 0) return;
  require_aligned(x, "dflash2_conv_prepare: x must be 16-byte aligned");
  require_aligned(out, "dflash2_conv_prepare: out must be 16-byte aligned");
  launch_pdl(df2_conv_prepare_kernel, dim3(rows), dim3(ROW_THREADS), stream, x, base_kernel, dyn, out);
}

void dflash2_conv_finish_add_rmsnorm(ActT* res, const float* delta, const ActT* base_kernel, const float* dyn,
                                     const ActT* norm_w, ActT* xn, int rows, cudaStream_t stream) {
  if (rows <= 0) return;
  require_aligned(res, "dflash2_conv_finish_add_rmsnorm: res must be 16-byte aligned");
  require_aligned(delta, "dflash2_conv_finish_add_rmsnorm: delta must be 16-byte aligned");
  require_aligned(xn, "dflash2_conv_finish_add_rmsnorm: xn must be 16-byte aligned");
  launch_pdl(df2_conv_finish_add_rmsnorm_kernel, dim3(rows), dim3(ROW_THREADS), stream, res, delta, base_kernel, dyn,
             norm_w, xn);
}

void dflash2_qkv_rope(const float* qkv, const ActT* q_norm, const ActT* k_norm, const DraftParams* p, ActT* q_out,
                      ActT* k_cache, ActT* v_cache, float rope_theta, cudaStream_t stream) {
  launch_pdl(df2_qkv_rope_kernel, dim3(BLOCK, DF2_HEADS + 2 * KV_HEADS), dim3(HEAD_DIM), stream, qkv, q_norm, k_norm, p,
             q_out, k_cache, v_cache, rope_theta);
}

std::size_t dflash2_attention_workspace_floats() {
  return std::size_t(DF2_ATT_CHUNKS) * KV_HEADS * DF2_NQ * (HEAD_DIM + 2);
}

void dflash2_block_attention(const ActT* q, const ActT* k_cache, const ActT* v_cache, const std::int32_t* ctx_pos,
                             const DraftParams* p, float* ws, ActT* out, cudaStream_t stream) {
  float* ws_o = ws;
  float* ws_ml = ws + std::size_t(DF2_ATT_CHUNKS) * KV_HEADS * DF2_NQ * HEAD_DIM;
  launch_pdl(df2_attn_partial_kernel, dim3(DF2_ATT_CHUNKS, KV_HEADS), dim3(DF2_ATT_THREADS), stream, q, k_cache,
             v_cache, ctx_pos, p, ws_o, ws_ml);
  launch_pdl(df2_attn_combine_kernel, dim3(KV_HEADS * DF2_NQ), dim3(HEAD_DIM), stream,
             static_cast<const float*>(ws_o), static_cast<const float*>(ws_ml), out);
}

std::size_t dflash2_topk16_workspace_bytes(int rows) {
  return std::size_t(rows) * DF2_TOPK_SLICES * DF2_TOPK * sizeof(Df2TopkPair);
}

void dflash2_topk16(const float* base, int rows, int V, const std::int32_t* map, float* cand_val,
                    std::int32_t* cand_idx, void* ws, cudaStream_t stream) {
  if (V < DF2_TOPK) throw std::invalid_argument("dflash2_topk16: V must be at least 16");
  if (rows <= 0) return;
  auto* partial = static_cast<Df2TopkPair*>(ws);
  launch_pdl(df2_topk16_slice_kernel, dim3(DF2_TOPK_SLICES, rows), dim3(DF2_TOPK_THREADS), stream, base, V, partial);
  launch_pdl(df2_topk16_reduce_kernel, dim3(rows), dim3(DF2_TOPK_THREADS), stream,
             static_cast<const Df2TopkPair*>(partial), map, cand_val, cand_idx);
}

void dflash2_select_path(const float* cand_val, const std::int32_t* cand_idx, const float* g_proj, const ActT* pred_cb,
                         const ActT* succ_cb, std::int32_t* prev_cand, float* draft_logits, int V, float softcap,
                         bool softcap_after_correction, std::int32_t* tokens, float* conf, float* qstats,
                         DraftParams* p, cudaStream_t stream) {
  launch_pdl(df2_select_path_kernel, dim3(1), dim3(512), stream, cand_val, cand_idx, g_proj, pred_cb, succ_cb,
             prev_cand, draft_logits, V, softcap, softcap_after_correction, tokens, conf, qstats, p);
}

}  // namespace ingine::kernels::draft

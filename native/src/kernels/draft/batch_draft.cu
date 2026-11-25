// Batched multi-slot DSpark drafter kernels (B = 1..64+, SM60..SM120a, BF16 & INGINE_ACT_FP16).
// Exact arithmetic parity with kernels/draft/{block,attention,markov,accept}.cu, extended across B independent slots.
#include "kernels/draft/batch_draft.h"

#include "kernels/common/l2.cuh"
#include "kernels/common/mma_sm70.cuh"
#include "kernels/draft/draft_device.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ingine::kernels::draft {
namespace {

constexpr int ROW_THREADS = H / 8;  // 288
static_assert(H % 8 == 0 && ROW_THREADS % 32 == 0);

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

__device__ __forceinline__ uint4 mix_row(const ActT* f, int c, const float (&a)[4], const uint4& wv, float* scratch) {
  uint4 fv[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) fv[j] = *reinterpret_cast<const uint4*>(f + j * H + c);
  float m[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float x[8];
    unpack8(fv[j], x);
#pragma unroll
    for (int i = 0; i < 8; ++i) m[i] = j == 0 ? __fmul_rn(x[i], a[0]) : __fadd_rn(m[i], __fmul_rn(x[i], a[j]));
  }
  return rms_norm(m, wv, scratch);
}

__device__ __forceinline__ void load_alpha(const ActT* alpha, int part, float (&a)[4]) {
  const uint4 v = ld_weight(alpha + 8 * (part >> 1));
  float f[8];
  unpack8(v, f);
#pragma unroll
  for (int j = 0; j < 4; ++j) a[j] = f[4 * (part & 1) + j];
}

__device__ __forceinline__ float norm_rope_head(float x, const ActT* __restrict__ w, int pos, float* scratch,
                                                float* pair) {
  const int d = threadIdx.x;
  const float ss = block_sum(x * x, scratch);
  const float rs = rsqrtf(ss / float(HEAD_DIM) + EPS);
  const float y = bf_round(__fmul_rn(__fmul_rn(x, rs), bf(w[d])));
  pair[d] = y;
  __syncthreads();
  const int half = HEAD_DIM / 2, i = d & (half - 1);
  const float inv = powf(ROPE_THETA, (-float(i) * 2.f) / float(HEAD_DIM));
  const float ang = __fmul_rn(float(pos), inv);
  const float c = cosf(ang), s = sinf(ang);
  const float x1 = pair[i], x2 = pair[i + half];
  const float o = d < half ? __fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s))
                           : __fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s));
  __syncthreads();
  return bf_round(o);
}

template <bool FP32_IN>
__device__ __forceinline__ float load_val(const void* p, std::size_t idx) {
  if constexpr (FP32_IN) return static_cast<const float*>(p)[idx];
  else return act::to_float(static_cast<const ActT*>(p)[idx]);
}

// ---- Batched context commit --------------------------------------------------------------------------------

// grid (W, 4 parts, B), block ROW_THREADS (288)
__global__ void __launch_bounds__(ROW_THREADS) batch_commit_mix_kernel(const BatchSlotCommitMeta* meta, int B, int W,
                                                                       const ActT* taps, const ActT* alpha,
                                                                       BatchCtxLayers L, const ActT* f_norm, ActT* xin,
                                                                       ActT* f_rnn_pool) {
  __shared__ float scratch[33];
  const int r = blockIdx.x, part = blockIdx.y, b = blockIdx.z, c = 8 * threadIdx.x;
  const BatchSlotCommitMeta m = meta[b];
  if (m.slot < 0) {
    if (part < 3) {
      *reinterpret_cast<uint4*>(xin + (std::size_t(part) * std::size_t(B) * W + std::size_t(b) * W + r) * H + c) =
          make_uint4(0, 0, 0, 0);
    }
    return;
  }
  const ActT* norm = part < 3 ? L.ctx_norm[part] : f_norm;
  const uint4 wv = ld_weight(norm + c);
  float a[4];
  load_alpha(alpha, part, a);
  const ActT* f = taps + std::size_t(b * W + r) * 4 * H;
  if (part == 3) {
    if (r == m.ncommit - 1) {
      *reinterpret_cast<uint4*>(f_rnn_pool + std::size_t(m.slot) * H + c) = mix_row(f, c, a, wv, scratch);
    }
    return;
  }
  uint4* out =
      reinterpret_cast<uint4*>(xin + (std::size_t(part) * std::size_t(B) * W + std::size_t(b) * W + r) * H + c);
  if (r >= m.ncommit) {
    *out = make_uint4(0, 0, 0, 0);
    return;
  }
  *out = mix_row(f, c, a, wv, scratch);
}

// grid (W, KV_HEADS, B), block HEAD_DIM (128)
template <bool KV_FP32>
__global__ void __launch_bounds__(HEAD_DIM) batch_commit_store_kernel(const void* kv, int ld_kv, int col_off, int l,
                                                                      const BatchSlotCommitMeta* meta, int W,
                                                                      BatchCtxLayers L) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  const int r = blockIdx.x, g = blockIdx.y, b = blockIdx.z, d = threadIdx.x;
  const BatchSlotCommitMeta m = meta[b];
  if (m.slot < 0 || r >= m.ncommit) return;
  const int pos = m.start + r;
  const int ring_slot = pos % L.size[l];
  const std::size_t row_off = std::size_t(b * W + r) * ld_kv + col_off;
  const float k_in = load_val<KV_FP32>(kv, row_off + g * HEAD_DIM + d);
  const float v_in = load_val<KV_FP32>(kv, row_off + KV_DIM + g * HEAD_DIM + d);
  const float k = norm_rope_head(k_in, L.k_norm_ctx[l], pos, scratch, pair);
  const std::size_t at = (std::size_t(ring_slot) * KV_HEADS + g) * HEAD_DIM + d;
  void* dst_k = static_cast<std::uint8_t*>(L.k_cache[l]) + std::size_t(m.slot) * L.kv_slot_bytes[l];
  void* dst_v = static_cast<std::uint8_t*>(L.v_cache[l]) + std::size_t(m.slot) * L.kv_slot_bytes[l];
  store_cache(dst_k, at, k, L.fp8[l]);
  store_cache(dst_v, at, v_in, L.fp8[l]);
  if (L.ctx_pos[l] != nullptr && g == 0 && d == 0) {
    L.ctx_pos[l][std::size_t(m.slot) * WINDOW + ring_slot] = pos;
  }
}

// ---- Batched draft block -----------------------------------------------------------------------------------

// grid (BLOCK, B), block ROW_THREADS (288)
__global__ void __launch_bounds__(ROW_THREADS) batch_embed_block_kernel(const ActT* embed, const float* embed_norm,
                                                                        const ActT* mask_embed, const ActT* in_norm,
                                                                        const BatchSlotDraftMeta* meta, ActT* res,
                                                                        ActT* xn, unsigned long long* picks,
                                                                        bool embed_q4) {
  __shared__ float scratch[33];
  const int r = blockIdx.x, b = blockIdx.y, c = 8 * threadIdx.x;
  const BatchSlotDraftMeta m = meta[b];
  if (m.slot < 0) return;
  const uint4 wv = ld_weight(in_norm + c);
  float x[8];
  if (r == 0) {
    const uint4 a0 = ld_weight(embed_norm + c);
    const uint4 a1 = ld_weight(embed_norm + c + 4);
    float v[8];
    if (embed_q4) {
      const uint8_t* row = reinterpret_cast<const uint8_t*>(embed) + std::size_t(m.anchor) * 1296;
      const uint32_t nib4 = *reinterpret_cast<const uint32_t*>(row + (c >> 1));
      const float d = __half2float(reinterpret_cast<const __half*>(row + 1152)[c >> 5]);
      const float m8d = -8.f * d;
#pragma unroll
      for (int j = 0; j < 8; ++j) v[j] = fmaf(float((nib4 >> (4 * j)) & 0xfu), d, m8d);
    } else {
      unpack8(*reinterpret_cast<const uint4*>(embed + std::size_t(m.anchor) * H + c), v);
    }
    const float rs = rms_scale(v, scratch);
    const float en[8] = {__uint_as_float(a0.x), __uint_as_float(a0.y), __uint_as_float(a0.z), __uint_as_float(a0.w),
                         __uint_as_float(a1.x), __uint_as_float(a1.y), __uint_as_float(a1.z), __uint_as_float(a1.w)};
#pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = bf_round(__fmul_rn(__fmul_rn(v[j], rs), en[j]));
  } else {
    unpack8(ld_weight(mask_embed + c), x);
  }
  const std::size_t out_row = std::size_t(b * BLOCK + r) * H + c;
  *reinterpret_cast<uint4*>(res + out_row) = pack8(x);
  *reinterpret_cast<uint4*>(xn + out_row) = rms_norm(x, wv, scratch);
  if (r == 0 && threadIdx.x < GAMMA) picks[std::size_t(b) * GAMMA + threadIdx.x] = 0ull;
}

// grid (BLOCK, HEADS + 2 * KV_HEADS, B), block HEAD_DIM (128)
template <bool QKV_FP32>
__global__ void __launch_bounds__(HEAD_DIM) batch_qkv_rope_kernel(const void* qkv, const ActT* __restrict__ q_norm,
                                                                  const ActT* __restrict__ k_norm,
                                                                  const BatchSlotDraftMeta* meta, ActT* q_out,
                                                                  void* k_cache, void* v_cache,
                                                                  std::size_t kv_slot_bytes, bool fp8, bool local,
                                                                  int ctx) {
  __shared__ float scratch[33];
  __shared__ float pair[HEAD_DIM];
  const int r = blockIdx.x, h = blockIdx.y, b = blockIdx.z, d = threadIdx.x;
  const BatchSlotDraftMeta m = meta[b];
  if (m.slot < 0) return;
  const int pos = m.P + r;
  const std::size_t row_off = std::size_t(b * BLOCK + r) * QKV_ROWS;
  const int ring_slot = local ? (WINDOW + r) : (pos % ctx);
  if (h < HEADS) {
    const float q = norm_rope_head(load_val<QKV_FP32>(qkv, row_off + h * HEAD_DIM + d), q_norm, pos, scratch, pair);
    q_out[(std::size_t(b * BLOCK + r) * HEADS + h) * HEAD_DIM + d] = act::from_float(q);
  } else if (h < HEADS + KV_HEADS) {
    const int g = h - HEADS;
    const float k =
        norm_rope_head(load_val<QKV_FP32>(qkv, row_off + Q_DIM + g * HEAD_DIM + d), k_norm, pos, scratch, pair);
    void* dst_k = static_cast<std::uint8_t*>(k_cache) + std::size_t(m.slot) * kv_slot_bytes;
    store_cache(dst_k, (std::size_t(ring_slot) * KV_HEADS + g) * HEAD_DIM + d, k, fp8);
  } else {
    const int g = h - HEADS - KV_HEADS;
    const float v = load_val<QKV_FP32>(qkv, row_off + Q_DIM + KV_DIM + g * HEAD_DIM + d);
    void* dst_v = static_cast<std::uint8_t*>(v_cache) + std::size_t(m.slot) * kv_slot_bytes;
    store_cache(dst_v, (std::size_t(ring_slot) * KV_HEADS + g) * HEAD_DIM + d, v, fp8);
  }
}

// ---- Batched block attention (exact mirror of attention.cu with grid.z = b and bounded B*chunks <= max(128, B)) ----

constexpr int QH = HEADS / KV_HEADS;
constexpr int NQ = QH * BLOCK;
constexpr int ATT_TILE = 16;
constexpr int ATT_THREADS = 128;
constexpr int ATT_MAX_CHUNKS = 128;
constexpr int ATT_MIN_TILES = 2;

int batch_att_max_chunks_for(int B) { return std::max(1, std::min(ATT_MAX_CHUNKS, 128 / std::max(1, B))); }
int batch_att_tiles_per_chunk(int B, int nkeys) {
  const int mc = batch_att_max_chunks_for(B);
  return std::max(ATT_MIN_TILES, (nkeys + ATT_TILE * mc - 1) / (ATT_TILE * mc));
}
int batch_att_chunks(int B, int nkeys) {
  const int ck = ATT_TILE * batch_att_tiles_per_chunk(B, nkeys);
  return (nkeys + ck - 1) / ck;
}

__device__ __forceinline__ void mma_bf16_b(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  mma::mma_act(c, a[0], a[1], a[2], a[3], b[0], b[1]);
}
__device__ __forceinline__ void ldmatrix_x4_b(uint32_t (&r)[4], const void* row) {
  mma::ldmatrix_x4(r[0], r[1], r[2], r[3], static_cast<unsigned>(__cvta_generic_to_shared(row)));
}
__device__ __forceinline__ void ldmatrix_x4_trans_b(uint32_t (&r)[4], const void* row) {
  mma::ldmatrix_x4_trans(r[0], r[1], r[2], r[3], static_cast<unsigned>(__cvta_generic_to_shared(row)));
}
__device__ __forceinline__ void ldmatrix_x2_b(uint32_t (&r)[2], const void* row) {
  mma::ldmatrix_x2(r[0], r[1], static_cast<unsigned>(__cvta_generic_to_shared(row)));
}
__device__ __forceinline__ uint4 ld_cache_b(const void* p) {
  uint4 r;
  asm volatile("ld.global" INGINE_L1_NO_ALLOCATE ".v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p)
               : "memory");
  return r;
}

template <bool FP8>
struct BatchTileRegs {
  static constexpr int NL = FP8 ? 1 : 2;
  uint4 k[NL], v[NL];
  int cp;
};

template <bool LOCAL, bool FP8>
__global__ void __launch_bounds__(ATT_THREADS) batch_attn_partial_kernel(const ActT* q, const uint8_t* k_cache_base,
                                                                         const uint8_t* v_cache_base,
                                                                         std::size_t kv_slot_bytes, int nkeys,
                                                                         int chunk_keys, int chunks,
                                                                         const std::int32_t* ctx_pos_base,
                                                                         const BatchSlotDraftMeta* meta, float* ws_o,
                                                                         float* ws_ml) {
  constexpr int ROW_BYTES = FP8 ? HEAD_DIM : 2 * HEAD_DIM;
  constexpr int NL = BatchTileRegs<FP8>::NL;
  __shared__ __align__(16) ActT qs[QH][BLOCK][HEAD_DIM];
  __shared__ __align__(16) ActT kt[2][ATT_TILE][HEAD_DIM];
  __shared__ __align__(16) ActT vt[2][ATT_TILE][HEAD_DIM];
  __shared__ __align__(16) ActT pt[QH][BLOCK][ATT_TILE];
  __shared__ unsigned vmask[2];

  const int chunk = blockIdx.x, g = blockIdx.y, b = blockIdx.z;
  const BatchSlotDraftMeta m = meta[b];
  if (m.slot < 0) return;

  const uint8_t* k_cache = k_cache_base + std::size_t(m.slot) * kv_slot_bytes;
  const uint8_t* v_cache = v_cache_base + std::size_t(m.slot) * kv_slot_bytes;
  const std::int32_t* ctx_pos = LOCAL ? (ctx_pos_base + std::size_t(m.slot) * WINDOW) : nullptr;
  const ActT* q_slot = q + std::size_t(b) * BLOCK * Q_DIM;

  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int gl = lane >> 2, t = lane & 3, lm = lane >> 3, lr = lane & 7;
  const int k0 = chunk * chunk_keys, k1 = min(k0 + chunk_keys, nkeys);
  const int P = m.P;
  const int kend = LOCAL ? k1 : min(k1, P + BLOCK);
  constexpr float scale = 0.08838834764831845f;

  auto load = [&](BatchTileRegs<FP8>& x, int t0) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const int idx = tid + ATT_THREADS * j;
      const int key = t0 + idx / (ROW_BYTES / 16), piece = idx % (ROW_BYTES / 16);
      const std::size_t at = (std::size_t(key) * KV_HEADS + g) * ROW_BYTES + 16 * piece;
      const bool live = key < kend;
      x.k[j] = live ? ld_cache_b(k_cache + at) : make_uint4(0, 0, 0, 0);
      x.v[j] = live ? ld_cache_b(v_cache + at) : make_uint4(0, 0, 0, 0);
    }
    x.cp = 0;
    if constexpr (LOCAL) {
      if (tid < ATT_TILE && t0 + tid < WINDOW) x.cp = ctx_pos[t0 + tid];
    }
  };
  BatchTileRegs<FP8> ra, rb;
  if (k0 < kend) load(ra, k0);
  if (k0 + ATT_TILE < kend) load(rb, k0 + ATT_TILE);

  {
    uint4 qv[QH];
#pragma unroll
    for (int j = 0; j < QH; ++j) {
      const int idx = tid + ATT_THREADS * j, hh = idx >> 7, r = (idx >> 4) & 7, piece = idx & 15;
      qv[j] = *reinterpret_cast<const uint4*>(q_slot + (std::size_t(r) * HEADS + g * QH + hh) * HEAD_DIM + 8 * piece);
    }
#pragma unroll
    for (int j = 0; j < QH; ++j) {
      const int idx = tid + ATT_THREADS * j;
      *reinterpret_cast<uint4*>(&qs[idx >> 7][(idx >> 4) & 7][8 * (idx & 15)]) = qv[j];
    }
  }
  __syncthreads();
  const bool head = w < QH;
  const int hw = head ? w : 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
  const int v_qp = (lane >> 2) & 3;
  const int v_idx = (lane & 3) | ((lane & 16) >> 2);
  const int v_row_a = (v_qp >= 2 ? 8 : 0) + v_idx;
  const int v_kh = (v_qp & 1) * 4;
  uint32_t qb_v[HEAD_DIM / 16][4];
#pragma unroll
  for (int s = 0; s < HEAD_DIM / 16; ++s) {
    const uint2 q0 = mma::ld_smem_u64(static_cast<unsigned>(__cvta_generic_to_shared(&qs[hw][v_idx][16 * s + v_kh])));
    const uint2 q1 = mma::ld_smem_u64(static_cast<unsigned>(__cvta_generic_to_shared(&qs[hw][v_idx][16 * s + 8 + v_kh])));
    qb_v[s][0] = q0.x, qb_v[s][1] = q0.y, qb_v[s][2] = q1.x, qb_v[s][3] = q1.y;
  }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
#else
  uint32_t qb[HEAD_DIM / 16][2];
#pragma unroll
  for (int s = 0; s < HEAD_DIM / 16; ++s) ldmatrix_x2_b(qb[s], &qs[hw][lr][16 * s + 8 * (lm & 1)]);
#endif

  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
  float oacc[HEAD_DIM / 16][4];
#pragma unroll
  for (int d = 0; d < HEAD_DIM / 16; ++d)
#pragma unroll
    for (int i = 0; i < 4; ++i) oacc[d][i] = 0.f;

  auto tile = [&](BatchTileRegs<FP8>& x, int t0, int buf) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const int idx = tid + ATT_THREADS * j;
      const int key = idx / (ROW_BYTES / 16), piece = idx % (ROW_BYTES / 16);
      if constexpr (FP8) {
        const uint32_t kw[4] = {x.k[j].x, x.k[j].y, x.k[j].z, x.k[j].w};
        const uint32_t vw[4] = {x.v[j].x, x.v[j].y, x.v[j].z, x.v[j].w};
        uint32_t kb[8], vb[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          kb[2 * i] = e4m3x2_to_bf16x2(kw[i]);
          kb[2 * i + 1] = e4m3x2_to_bf16x2(kw[i] >> 16);
          vb[2 * i] = e4m3x2_to_bf16x2(vw[i]);
          vb[2 * i + 1] = e4m3x2_to_bf16x2(vw[i] >> 16);
        }
        uint4* kd = reinterpret_cast<uint4*>(&kt[buf][key][16 * piece]);
        uint4* vd = reinterpret_cast<uint4*>(&vt[buf][key][16 * piece]);
        kd[0] = make_uint4(kb[0], kb[1], kb[2], kb[3]);
        kd[1] = make_uint4(kb[4], kb[5], kb[6], kb[7]);
        vd[0] = make_uint4(vb[0], vb[1], vb[2], vb[3]);
        vd[1] = make_uint4(vb[4], vb[5], vb[6], vb[7]);
      } else {
        *reinterpret_cast<uint4*>(&kt[buf][key][8 * piece]) = x.k[j];
        *reinterpret_cast<uint4*>(&vt[buf][key][8 * piece]) = x.v[j];
      }
    }
    if (w == 0) {
      const int j = t0 + lane;
      bool v = lane < ATT_TILE && j < kend;
      if constexpr (LOCAL) {
        if (j < WINDOW) v = v && x.cp >= 0 && x.cp < P && x.cp >= P - WINDOW;
      }
      const unsigned mask = __ballot_sync(0xffffffffu, v);
      if (lane == 0) vmask[buf] = mask;
    }
    __syncthreads();
    if (t0 + 2 * ATT_TILE < kend) load(x, t0 + 2 * ATT_TILE);
    if (!head) return;

    float sc[4] = {0.f, 0.f, 0.f, 0.f};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
    float sc8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s = 0; s < HEAD_DIM / 16; ++s) {
      const uint2 k0 = mma::ld_smem_u64(static_cast<unsigned>(__cvta_generic_to_shared(&kt[buf][v_row_a][16 * s + v_kh])));
      const uint2 k1 = mma::ld_smem_u64(static_cast<unsigned>(__cvta_generic_to_shared(&kt[buf][v_row_a][16 * s + 8 + v_kh])));
      mma::sm70_mma_f16_direct(sc8, k0.x, k0.y, k1.x, k1.y, qb_v[s][0], qb_v[s][1], qb_v[s][2], qb_v[s][3]);
    }
    mma::sm70_reduce_acc8_to_d4(sc8, sc);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
#pragma unroll
    for (int s = 0; s < HEAD_DIM / 16; ++s) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const uint4 ka = *reinterpret_cast<const uint4*>(&kt[buf][gl][16 * s + 8 * c]);
        const uint4 kb = *reinterpret_cast<const uint4*>(&kt[buf][gl + 8][16 * s + 8 * c]);
        const uint4 qa = *reinterpret_cast<const uint4*>(&qs[hw][2 * t][16 * s + 8 * c]);
        const uint4 qb4 = *reinterpret_cast<const uint4*>(&qs[hw][2 * t + 1][16 * s + 8 * c]);
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
      ldmatrix_x4_b(a, &kt[buf][8 * (lm & 1) + lr][16 * s + 8 * (lm >> 1)]);
      mma_bf16_b(sc, a, qb[s]);
    }
#endif
    const unsigned mask = vmask[buf];
    const bool valid[2] = {((mask >> gl) & 1u) != 0, ((mask >> (gl + 8)) & 1u) != 0};
    float al[2];
    ActT pb16[2][2];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
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
      mma::sm70_mma_f16_direct(ov, v0 | (v1 << 16), v2 | (v3 << 16), v8 | (v9 << 16), v10 | (v11 << 16),
                               p0.x, p0.y, p1.x, p1.y);
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
    ldmatrix_x2_b(pf, &pt[w][lr][8 * (lm & 1)]);
#pragma unroll
    for (int d = 0; d < HEAD_DIM / 16; ++d) {
      if (rescale) {
        oacc[d][0] *= al[0];
        oacc[d][1] *= al[1];
        oacc[d][2] *= al[0];
        oacc[d][3] *= al[1];
      }
      uint32_t a[4];
      ldmatrix_x4_trans_b(a, &vt[buf][8 * (lm >> 1) + lr][16 * d + 8 * (lm & 1)]);
      mma_bf16_b(oacc[d], a, pf);
    }
#endif
  };
  for (int t0 = k0; t0 < kend; t0 += 2 * ATT_TILE) {
    tile(ra, t0, 0);
    if (t0 + ATT_TILE < kend) tile(rb, t0 + ATT_TILE, 1);
  }
  if (!head) return;

  const std::size_t base = (std::size_t(b) * chunks + chunk) * KV_HEADS + g;
  const int qi = BLOCK * w + 2 * t;
#pragma unroll
  for (int d = 0; d < HEAD_DIM / 16; ++d) {
    float* at = ws_o + (base * NQ + qi) * HEAD_DIM + 16 * d + gl;
    at[0] = oacc[d][0];
    at[HEAD_DIM] = oacc[d][1];
    at[8] = oacc[d][2];
    at[HEAD_DIM + 8] = oacc[d][3];
  }
  if (gl == 0) {
    float* at = ws_ml + (base * NQ + qi) * 2;
    *reinterpret_cast<float4*>(at) = make_float4(m_run[0], l_run[0], m_run[1], l_run[1]);
  }
}

__global__ void __launch_bounds__(HEAD_DIM) batch_attn_combine_kernel(const float* ws_o, const float* ws_ml, int chunks,
                                                                      const BatchSlotDraftMeta* meta, ActT* out) {
  __shared__ float scratch[33];
  __shared__ float f[HEAD_DIM];
  const int b = blockIdx.y;
  if (meta[b].slot < 0) return;
  const int g = blockIdx.x / NQ, qi = blockIdx.x % NQ, d = threadIdx.x;
  float mc = -INFINITY, lc = 0.f;
  if (d < chunks) {
    const std::size_t base = (std::size_t(b) * chunks + d) * KV_HEADS + g;
    const float2 ml = *reinterpret_cast<const float2*>(ws_ml + (base * NQ + qi) * 2);
    mc = ml.x;
    lc = ml.y;
  }
  const float m = block_max(mc, scratch);
  const float fc = mc == -INFINITY ? 0.f : __expf(mc - m);
  f[d] = fc;
  const float l = block_sum(lc * fc, scratch);
  float o = 0.f;
#pragma unroll 4
  for (int c = 0; c < chunks; ++c) {
    const std::size_t base = (std::size_t(b) * chunks + c) * KV_HEADS + g;
    o += ws_o[(base * NQ + qi) * HEAD_DIM + d] * f[c];
  }
  const int hh = qi / BLOCK, r = qi % BLOCK, h = g * QH + hh;
  out[std::size_t(b * BLOCK + r) * Q_DIM + h * HEAD_DIM + d] = act::from_float(l > 0.f ? o / l : 0.f);
}

// ---- Residual & activation helpers -------------------------------------------------------------------------

template <bool DELTA_FP32>
__global__ void __launch_bounds__(ROW_THREADS) batch_add_rmsnorm_kernel(ActT* res, const void* delta,
                                                                        const ActT* __restrict__ w, ActT* xn) {
  __shared__ float scratch[33];
  const std::size_t at = std::size_t(blockIdx.x) * H + 8 * threadIdx.x;
  const uint4 wv = ld_weight(w + 8 * threadIdx.x);
  float x[8];
  unpack8(*reinterpret_cast<const uint4*>(res + at), x);
  float d[8];
  if constexpr (DELTA_FP32) {
    const float4 d0 = *reinterpret_cast<const float4*>(static_cast<const float*>(delta) + at);
    const float4 d1 = *reinterpret_cast<const float4*>(static_cast<const float*>(delta) + at + 4);
    d[0] = d0.x, d[1] = d0.y, d[2] = d0.z, d[3] = d0.w;
    d[4] = d1.x, d[5] = d1.y, d[6] = d1.z, d[7] = d1.w;
  } else {
    unpack8(*reinterpret_cast<const uint4*>(static_cast<const ActT*>(delta) + at), d);
  }
#pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = bf_round(__fadd_rn(x[j], d[j]));
  *reinterpret_cast<uint4*>(res + at) = pack8(x);
  *reinterpret_cast<uint4*>(xn + at) = rms_norm(x, wv, scratch);
}

template <bool GU_FP32>
__global__ void batch_silu_mul_kernel(const void* gu, int rows, int inter, ActT* out) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(rows) * inter) return;
  const std::size_t r = i / inter, j = i % inter;
  const float g = load_val<GU_FP32>(gu, r * 2 * inter + j);
  const float u = load_val<GU_FP32>(gu, r * 2 * inter + inter + j);
  const float s = __fdiv_rn(g, __fadd_rn(1.f, expf(-g)));
  out[i] = act::from_float(__fmul_rn(s, u));
}

__global__ void batch_extract_gamma_rows_kernel(const ActT* __restrict__ xn, int B, ActT* __restrict__ h_gamma) {
  const int r = blockIdx.x;  // 0 .. B * GAMMA - 1
  const int b = r / GAMMA;
  const int g = r % GAMMA;
  const int c = 8 * threadIdx.x;
  *reinterpret_cast<uint4*>(h_gamma + std::size_t(r) * H + c) =
      *reinterpret_cast<const uint4*>(xn + std::size_t(b * BLOCK + 1 + g) * H + c);
}

__global__ void batch_cast_base_chunk_kernel(const ActT* __restrict__ src, int rows, int chunk_V, int v_offset,
                                             int dst_stride, float* __restrict__ dst) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (v >= chunk_V || r >= rows) return;
  dst[std::size_t(r) * dst_stride + v_offset + v] = act::to_float(src[std::size_t(r) * chunk_V + v]);
}

__global__ void batch_copy_base_rows_kernel(const float* __restrict__ src, int rows, int src_stride, int dst_stride,
                                            float* __restrict__ dst) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y;
  if (v >= src_stride || r >= rows) return;
  dst[std::size_t(r) * dst_stride + v] = src[std::size_t(r) * src_stride + v];
}

// Slot-local dynamic shortlist scoring: one warp per (b, extra_idx e) computes all GAMMA=7 dot products
// reading unembed[tok_id] ONCE directly from resident weights! Zero cross-slot reads or writes.
__device__ __forceinline__ float e4m3_to_f32_scalar(uint8_t b) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(b), __NV_E4M3)));
}

template <bool FP8_W>
__global__ void __launch_bounds__(128) batch_score_slot_extra_kernel(const ActT* __restrict__ h_gamma,
                                                                     const void* __restrict__ unembed_w,
                                                                     const float* __restrict__ unembed_scale,
                                                                     const std::int32_t* __restrict__ slot_extra_ids,
                                                                     int max_extra, const BatchSlotDraftMeta* meta,
                                                                     int short_K, int base_stride, float* base) {
  const int b = blockIdx.y;
  const BatchSlotDraftMeta m = meta[b];
  if (m.slot < 0) return;
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int e = blockIdx.x * 4 + warp_in_block;
  if (e >= m.extra_count) return;
  const int tok_id = slot_extra_ids[std::size_t(m.slot) * max_extra + e];
  const float scale = FP8_W ? unembed_scale[tok_id] : 1.f;
  float acc[GAMMA] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  // H = 2304 = 288 * 8 = 9 * 32 * 8 -> 9 chunks of 8 elements per lane!
  constexpr int CHUNKS_PER_LANE = H / (32 * 8);
  static_assert(CHUNKS_PER_LANE == 9);
#pragma unroll
  for (int ch = 0; ch < CHUNKS_PER_LANE; ++ch) {
    const int col = (ch * 32 + lane) * 8;
    float w[8];
    if constexpr (FP8_W) {
      const uint2 raw = *reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(unembed_w) +
                                                        std::size_t(tok_id) * H + col);
      const uint32_t words[2] = {raw.x, raw.y};
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        w[4 * p + 0] = bf_round(e4m3_to_f32_scalar(words[p] & 0xffu) * scale);
        w[4 * p + 1] = bf_round(e4m3_to_f32_scalar((words[p] >> 8) & 0xffu) * scale);
        w[4 * p + 2] = bf_round(e4m3_to_f32_scalar((words[p] >> 16) & 0xffu) * scale);
        w[4 * p + 3] = bf_round(e4m3_to_f32_scalar((words[p] >> 24) & 0xffu) * scale);
      }
    } else {
      unpack8(*reinterpret_cast<const uint4*>(static_cast<const ActT*>(unembed_w) + std::size_t(tok_id) * H + col), w);
    }
#pragma unroll
    for (int g = 0; g < GAMMA; ++g) {
      float hv[8];
      unpack8(*reinterpret_cast<const uint4*>(h_gamma + std::size_t(b * GAMMA + g) * H + col), hv);
#pragma unroll
      for (int k = 0; k < 8; ++k) acc[g] = fmaf(w[k], hv[k], acc[g]);
    }
  }
#pragma unroll
  for (int g = 0; g < GAMMA; ++g) {
    const float sum = warp_sum(acc[g]);
    if (lane == 0) base[std::size_t(b * GAMMA + g) * base_stride + short_K + e] = bf_round(sum);
  }
}

// ---- Batched top-1024 radix select with per-slot V_row = short_K + extra_count -----------------------------

constexpr int B_TOPK_SLICES = 8;
constexpr int B_TOPK_THREADS = 256;
constexpr int B_GATHER_THREADS = 1024;
static_assert(CANDIDATES == B_GATHER_THREADS);

struct BatchTopkItem {
  float val;
  std::int32_t idx;
};

struct BatchTopkRowWs {
  uint32_t total_hi[256];
  uint32_t total_lo[256];
  int count_gt[B_TOPK_SLICES];
  int count_tie[B_TOPK_SLICES];
  BatchTopkItem above[B_TOPK_SLICES][CANDIDATES];
  BatchTopkItem ties[B_TOPK_SLICES][CANDIDATES];
};

__device__ __forceinline__ uint32_t b_key16(float v) {
#if INGINE_ACT_FP16
  const unsigned bits = act::float_to_bits(v);
  return bits ^ ((bits & 0x8000u) ? 0xffffu : 0x8000u);
#else
  return float_key(v) >> 16;
#endif
}

__device__ __forceinline__ void b_slice_range(int V, int s, int& b, int& e) {
  const int per = (V + B_TOPK_SLICES - 1) / B_TOPK_SLICES;
  b = min(V, s * per);
  e = min(V, b + per);
}

__device__ __forceinline__ int b_count_above(int c, int* warp_tot) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = B_TOPK_THREADS / 32;
  int v = c;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int u = __shfl_down_sync(0xffffffffu, v, o);
    if (lane + o < 32) v += u;
  }
  if (lane == 0) warp_tot[warp] = v;
  __syncthreads();
  int above = v - c;
  for (int w = warp + 1; w < nw; ++w) above += warp_tot[w];
  __syncthreads();
  return above;
}

__device__ __forceinline__ void b_select_bin(const uint32_t* hist, int k, int above0, int* sh_bin, int* sh_above,
                                             int* warp_tot) {
  const int c = int(hist[threadIdx.x]);
  const int above = above0 + b_count_above(c, warp_tot);
  if (above < k && above + c >= k) {
    *sh_bin = threadIdx.x;
    *sh_above = above;
  }
  __syncthreads();
}

__device__ __forceinline__ void b_select_threshold(const BatchTopkRowWs& w, uint32_t& T, int& n_gt, int* sh_bin,
                                                   int* sh_above, int* warp_tot) {
  b_select_bin(w.total_hi, CANDIDATES, 0, sh_bin, sh_above, warp_tot);
  const uint32_t b1 = *sh_bin;
  const int above1 = *sh_above;
  __syncthreads();
  b_select_bin(w.total_lo, CANDIDATES, above1, sh_bin, sh_above, warp_tot);
  T = (b1 << 8) | uint32_t(*sh_bin);
  n_gt = *sh_above;
}

__global__ void __launch_bounds__(B_TOPK_THREADS) batch_topk_hist_hi_kernel(const float* base, int base_stride,
                                                                            int short_K, const BatchSlotDraftMeta* meta,
                                                                            BatchTopkRowWs* ws) {
  __shared__ uint32_t hist[256];
  hist[threadIdx.x] = 0;
  __syncthreads();
  const int s = blockIdx.x, row = blockIdx.y;
  const int b_idx = row / GAMMA;
  const BatchSlotDraftMeta m = meta[b_idx];
  if (m.slot < 0) return;
  const int V = short_K + m.extra_count;
  const float* x = base + std::size_t(row) * base_stride;
  int b0, e0;
  b_slice_range(V, s, b0, e0);
  for (int i = b0 + threadIdx.x; i < e0; i += B_TOPK_THREADS) atomicAdd(&hist[b_key16(x[i]) >> 8], 1u);
  __syncthreads();
  if (hist[threadIdx.x] != 0) atomicAdd(&ws[row].total_hi[threadIdx.x], hist[threadIdx.x]);
}

__global__ void __launch_bounds__(B_TOPK_THREADS) batch_topk_hist_lo_kernel(const float* base, int base_stride,
                                                                            int short_K, const BatchSlotDraftMeta* meta,
                                                                            BatchTopkRowWs* ws) {
  __shared__ uint32_t hist[256];
  __shared__ int sh_bin, sh_above, warp_tot[B_TOPK_THREADS / 32];
  hist[threadIdx.x] = 0;
  const int s = blockIdx.x, row = blockIdx.y;
  const int b_idx = row / GAMMA;
  const BatchSlotDraftMeta m = meta[b_idx];
  if (m.slot < 0) return;
  BatchTopkRowWs& w = ws[row];
  b_select_bin(w.total_hi, CANDIDATES, 0, &sh_bin, &sh_above, warp_tot);
  const uint32_t b1 = sh_bin;
  const int V = short_K + m.extra_count;
  const float* x = base + std::size_t(row) * base_stride;
  int b0, e0;
  b_slice_range(V, s, b0, e0);
  for (int i = b0 + threadIdx.x; i < e0; i += B_TOPK_THREADS) {
    const uint32_t k = b_key16(x[i]);
    if ((k >> 8) == b1) atomicAdd(&hist[k & 255], 1u);
  }
  __syncthreads();
  if (hist[threadIdx.x] != 0) atomicAdd(&w.total_lo[threadIdx.x], hist[threadIdx.x]);
}

__global__ void __launch_bounds__(B_TOPK_THREADS) batch_topk_collect_kernel(const float* base, int base_stride,
                                                                            int short_K, const BatchSlotDraftMeta* meta,
                                                                            BatchTopkRowWs* ws) {
  __shared__ int sh_bin, sh_above, warp_tot[B_TOPK_THREADS / 32];
  __shared__ uint32_t warp_cnt[B_TOPK_THREADS / 32];
  const int s = blockIdx.x, row = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int b_idx = row / GAMMA;
  const BatchSlotDraftMeta m = meta[b_idx];
  if (m.slot < 0) return;
  BatchTopkRowWs& w = ws[row];
  uint32_t T;
  int n_gt;
  b_select_threshold(w, T, n_gt, &sh_bin, &sh_above, warp_tot);
  const int V = short_K + m.extra_count;
  const float* x = base + std::size_t(row) * base_stride;
  int b0, e0;
  b_slice_range(V, s, b0, e0);
  int n_above = 0, n_tie = 0;
  for (int i0 = b0; i0 < e0; i0 += B_TOPK_THREADS) {
    const int i = i0 + threadIdx.x;
    float v = 0.f;
    uint32_t k = 0;
    if (i < e0) {
      v = x[i];
      k = b_key16(v);
    }
    const bool gt = i < e0 && k > T, tie = i < e0 && k == T;
    const unsigned bg = __ballot_sync(0xffffffffu, gt), bt = __ballot_sync(0xffffffffu, tie);
    if (lane == 0) warp_cnt[warp] = uint32_t(__popc(bg)) | (uint32_t(__popc(bt)) << 16);
    __syncthreads();
    int rg = n_above + __popc(bg & ((1u << lane) - 1u)), rt = n_tie + __popc(bt & ((1u << lane) - 1u));
#pragma unroll
    for (int u = 0; u < B_TOPK_THREADS / 32; ++u) {
      const uint32_t c = warp_cnt[u];
      if (u < warp) {
        rg += int(c & 0xffffu);
        rt += int(c >> 16);
      }
      n_above += int(c & 0xffffu);
      n_tie += int(c >> 16);
    }
    if (gt) w.above[s][rg] = BatchTopkItem{v, i};
    if (tie && rt < CANDIDATES) w.ties[s][rt] = BatchTopkItem{v, i};
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    w.count_gt[s] = n_above;
    w.count_tie[s] = min(n_tie, CANDIDATES);
  }
}

__global__ void __launch_bounds__(B_GATHER_THREADS) batch_topk_gather_kernel(
    BatchTopkRowWs* ws, int short_K, const std::int32_t* short_map, const std::int32_t* slot_extra_ids, int max_extra,
    const BatchSlotDraftMeta* meta, float* cand_val, std::int32_t* cand_idx) {
  __shared__ int pre_gt[B_TOPK_SLICES + 1], pre_tie[B_TOPK_SLICES + 1];
  const int row = blockIdx.x, tid = threadIdx.x;
  const int b_idx = row / GAMMA;
  const BatchSlotDraftMeta m = meta[b_idx];
  if (m.slot < 0) return;
  BatchTopkRowWs& w = ws[row];
  if (tid < B_TOPK_SLICES) {
    int cg = w.count_gt[tid], ct = w.count_tie[tid];
#pragma unroll
    for (int o = 1; o < B_TOPK_SLICES; o <<= 1) {
      const int ug = __shfl_up_sync(0xffu, cg, o), ut = __shfl_up_sync(0xffu, ct, o);
      if (tid >= o) {
        cg += ug;
        ct += ut;
      }
    }
    pre_gt[tid + 1] = cg;
    pre_tie[tid + 1] = ct;
    if (tid == 0) pre_gt[0] = pre_tie[0] = 0;
  }
  if (tid < 256) {
    w.total_hi[tid] = 0;
    w.total_lo[tid] = 0;
  }
  __syncthreads();
  const int n_gt = pre_gt[B_TOPK_SLICES];
  const bool above = tid < n_gt;
  const int* pre = above ? pre_gt : pre_tie;
  const int j = above ? tid : tid - n_gt;
  int s = 0;
#pragma unroll
  for (int step = B_TOPK_SLICES / 2; step > 0; step >>= 1)
    if (pre[s + step] <= j) s += step;
  const BatchTopkItem it = above ? w.above[s][j - pre[s]] : w.ties[s][j - pre[s]];
  const std::size_t at = std::size_t(row) * CANDIDATES + tid;
  cand_val[at] = it.val;
  cand_idx[at] = (it.idx < short_K) ? short_map[it.idx] : slot_extra_ids[std::size_t(m.slot) * max_extra + (it.idx - short_K)];
}

// ---- Batched Markov RNN + logits + finish ------------------------------------------------------------------

constexpr int PRE_COLS = RNN_IN - 2 * RANK;
constexpr int PRE_CHUNKS = PRE_COLS / 8;
constexpr int PRE_PER_LANE = PRE_CHUNKS / 32;
constexpr int PRE_THREADS = 128;

// grid (RNN_OUT * 32 / PRE_THREADS, B), block PRE_THREADS
__global__ void __launch_bounds__(PRE_THREADS) batch_markov_pre_kernel(const ActT* __restrict__ rnn_w,
                                                                       const ActT* h_gamma, const ActT* f_rnn_pool,
                                                                       const BatchSlotDraftMeta* meta, float* part) {
  const int b = blockIdx.y;
  const BatchSlotDraftMeta m = meta[b];
  if (m.slot < 0) return;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  const ActT* wr = rnn_w + std::size_t(row) * RNN_IN + 2 * RANK;
  const ActT* h = h_gamma + std::size_t(b) * GAMMA * H;
  const ActT* f_rnn = f_rnn_pool + std::size_t(m.slot) * H;
  float acc[GAMMA];
#pragma unroll
  for (int i = 0; i < GAMMA; ++i) acc[i] = 0.f;
#pragma unroll
  for (int j = 0; j < PRE_PER_LANE; ++j) {
    const int col = 8 * (lane + 32 * j);
    float w[8];
    unpack8(ld_weight(wr + col), w);
    if (col < H) {
#pragma unroll
      for (int i = 0; i < GAMMA; ++i) {
        float z[8];
        unpack8(*reinterpret_cast<const uint4*>(h + std::size_t(i) * H + col), z);
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
  float* part_b = part + std::size_t(b) * GAMMA * RNN_OUT;
#pragma unroll
  for (int i = 0; i < GAMMA; ++i) {
    const float t = warp_sum(acc[i]);
    if (lane == 0) part_b[i * RNN_OUT + row] = t;
  }
}

// grid (RANK * 32 / 256, B), block 256
__global__ void __launch_bounds__(256) batch_markov_step_kernel(int i, const ActT* __restrict__ rnn_w,
                                                                const ActT* __restrict__ rnn_b,
                                                                const ActT* __restrict__ w1, const float* part,
                                                                const unsigned long long* picks,
                                                                const std::int32_t* cand_idx,
                                                                const BatchSlotDraftMeta* meta, ActT* state,
                                                                ActT* latent) {
  const int b = blockIdx.y;
  const BatchSlotDraftMeta m = meta[b];
  if (m.slot < 0) return;
  const int j = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  const unsigned long long* picks_b = picks + std::size_t(b) * GAMMA;
  const std::int32_t* cidx_b = cand_idx + std::size_t(b) * GAMMA * CANDIDATES;
  const int prev = (i == 0) ? m.anchor : cidx_b[(i - 1) * CANDIDATES + int(unpack_argmax_index(picks_b[i - 1]))];
  ActT* state_b = state + std::size_t(b) * (GAMMA + 1) * RANK;
  ActT* latent_b = latent + std::size_t(b) * RANK;
  const float* part_b = part + std::size_t(b) * GAMMA * RNN_OUT;
  float z0[8], z1[8];
  unpack8(*reinterpret_cast<const uint4*>(state_b + std::size_t(i) * RANK + 8 * lane), z0);
  unpack8(*reinterpret_cast<const uint4*>(w1 + std::size_t(prev) * RANK + 8 * lane), z1);
  float raw[3];
#pragma unroll
  for (int g = 0; g < 3; ++g) {
    const int row = g * RANK + j;
    const ActT* wr = rnn_w + std::size_t(row) * RNN_IN;
    float w0[8], wv1[8];
    unpack8(ld_weight(wr + 8 * lane), w0);
    unpack8(ld_weight(wr + RANK + 8 * lane), wv1);
    float acc = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc = fmaf(w0[e], z0[e], fmaf(wv1[e], z1[e], acc));
    acc = warp_sum(acc);
    raw[g] = bf_round(__fadd_rn(bf_round(acc + part_b[i * RNN_OUT + row]), bf(rnn_b[row])));
  }
  if (lane == 0) {
    const float gate = bf_round(__fdiv_rn(1.f, __fadd_rn(1.f, expf(-raw[0]))));
    const float cell = bf_round(tanhf(raw[1]));
    const float s = bf(state_b[std::size_t(i) * RANK + j]);
    const float gs = bf_round(__fmul_rn(gate, s));
    const float om = bf_round(__fsub_rn(1.f, gate));
    const float oc = bf_round(__fmul_rn(om, cell));
    state_b[std::size_t(i + 1) * RANK + j] = act::from_float(__fadd_rn(gs, oc));
    latent_b[j] = act::from_float(tanhf(raw[2]));
  }
}

// grid (CANDIDATES * 32 / 256, B), block 256
__global__ void __launch_bounds__(256) batch_markov_logits_kernel(int i, const ActT* __restrict__ fold_w,
                                                                  const ActT* __restrict__ fold_b,
                                                                  const float* cand_val, const std::int32_t* cand_idx,
                                                                  const ActT* latent, const BatchSlotDraftMeta* meta,
                                                                  float* cand_logits, unsigned long long* picks) {
  const int b = blockIdx.y;
  const BatchSlotDraftMeta m = meta[b];
  if (m.slot < 0) return;
  const int k = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  const std::size_t at = (std::size_t(b) * GAMMA + i) * CANDIDATES + k;
  const int c = cand_idx[at];
  const ActT* latent_b = latent + std::size_t(b) * RANK;
  float w[8], l[8];
  unpack8(*reinterpret_cast<const uint4*>(fold_w + std::size_t(c) * RANK + 8 * lane), w);
  unpack8(*reinterpret_cast<const uint4*>(latent_b + 8 * lane), l);
  float acc = 0.f;
#pragma unroll
  for (int e = 0; e < 8; ++e) acc = fmaf(w[e], l[e], acc);
  acc = warp_sum(acc);
  if (lane == 0) {
    const float bias = bf_round(__fadd_rn(bf(fold_b[c]), bf_round(acc)));
    const float t = bf_round(__fadd_rn(cand_val[at], bias));
    const float logit = bf_round(__fmul_rn(30.f, tanhf(__fdiv_rn(t, 30.f))));
    cand_logits[at] = logit;
    const float T = m.temperature;
    float score = logit;
    if (T > 0.f) score = __fadd_rn(__fdiv_rn(logit, fmaxf(T, 1e-6f)), gumbel(m.seed, m.draws, uint32_t(i), uint32_t(k)));
    atomicMax(&picks[std::size_t(b) * GAMMA + i], pack_argmax(score, uint32_t(k)));
  }
}

// grid (GAMMA, B), block CANDIDATES (1024)
__global__ void __launch_bounds__(CANDIDATES) batch_markov_finish_kernel(const std::int32_t* cand_idx,
                                                                         const float* cand_logits,
                                                                         const unsigned long long* picks,
                                                                         const BatchSlotDraftMeta* meta,
                                                                         std::int32_t* tokens, float* conf,
                                                                         float* qstats) {
  __shared__ float scratch[33];
  const int i = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
  const BatchSlotDraftMeta m = meta[b];
  if (m.slot < 0) return;
  const std::size_t at = (std::size_t(b) * GAMMA + i) * CANDIDATES + t;
  const float l = cand_logits[at];
  const float mx = block_max(l, scratch);
  const float s = block_sum(expf(l - mx), scratch);
  const float T = m.temperature;
  float qm = 0.f, qs = 0.f;
  if (T > 0.f) {
    const float x = __fdiv_rn(l, T);
    qm = block_max(x, scratch);
    qs = block_sum(expf(__fsub_rn(x, qm)), scratch);
  }
  if (t == 0) {
    const std::size_t row = std::size_t(b) * GAMMA + i;
    conf[row] = 1.f / s;
    qstats[2 * row] = qm;
    qstats[2 * row + 1] = qs;
    tokens[row] = cand_idx[(std::size_t(b) * GAMMA + i) * CANDIDATES + int(unpack_argmax_index(picks[row]))];
  }
}

__global__ void scatter_slot_draft_logits_kernel(const std::int32_t* __restrict__ cand_idx,
                                                 const float* __restrict__ cand_logits, int V,
                                                 float* __restrict__ dst) {
  const int i = blockIdx.x, t = threadIdx.x;
  const int at = i * CANDIDATES + t;
  dst[std::size_t(i) * V + cand_idx[at]] = cand_logits[at];
}

__global__ void clear_slot_draft_logits_kernel(const std::int32_t* __restrict__ cand_idx, int V,
                                               float* __restrict__ dst) {
  const int i = blockIdx.x, t = threadIdx.x;
  const int at = i * CANDIDATES + t;
  dst[std::size_t(i) * V + cand_idx[at]] = -INFINITY;
}

__global__ void batch_greedy_accept_kernel(const std::int32_t* __restrict__ target_argmax,
                                           const std::int32_t* __restrict__ drafts,
                                           const std::int32_t* __restrict__ slot_w_limit, int B, int W,
                                           std::int32_t* __restrict__ accepted_out, std::int32_t* __restrict__ next_out,
                                           std::int32_t* __restrict__ commit_out) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const int w_lim = slot_w_limit ? min(W, max(1, int(slot_w_limit[b]))) : W;
  const int n_drafts = w_lim - 1;
  const std::int32_t* t_row = target_argmax + std::size_t(b) * W;
  const std::int32_t* d_row = drafts + std::size_t(b) * GAMMA;
  int acc = 0;
  while (acc < n_drafts && d_row[acc] == t_row[acc]) ++acc;
  accepted_out[b] = acc;
  next_out[b] = t_row[acc];
  commit_out[b] = acc + 1;
}

void check_b() { CUDA_CHECK(cudaGetLastError()); }

}  // namespace

void batch_commit_mix(const BatchSlotCommitMeta* meta, int B, int W, const ActT* taps, const ActT* alpha,
                      const BatchCtxLayers& L, const ActT* f_norm, ActT* xin, ActT* f_rnn_pool, cudaStream_t stream) {
  if (B <= 0 || W <= 0) return;
  batch_commit_mix_kernel<<<dim3(W, 4, B), dim3(ROW_THREADS), 0, stream>>>(meta, B, W, taps, alpha, L, f_norm, xin,
                                                                           f_rnn_pool);
  check_b();
}

void batch_commit_store(const void* kv, bool kv_fp32, int ld_kv, int col_off, int l, const BatchSlotCommitMeta* meta,
                        int B, int W, const BatchCtxLayers& L, cudaStream_t stream) {
  if (B <= 0 || W <= 0) return;
  if (kv_fp32) {
    batch_commit_store_kernel<true><<<dim3(W, KV_HEADS, B), dim3(HEAD_DIM), 0, stream>>>(kv, ld_kv, col_off, l, meta, W,
                                                                                         L);
  } else {
    batch_commit_store_kernel<false><<<dim3(W, KV_HEADS, B), dim3(HEAD_DIM), 0, stream>>>(kv, ld_kv, col_off, l, meta,
                                                                                          W, L);
  }
  check_b();
}

void batch_embed_block(const ActT* embed, const float* embed_norm, const ActT* mask_embed, const ActT* in_norm,
                       const BatchSlotDraftMeta* meta, int B, ActT* res, ActT* xn, unsigned long long* picks,
                       cudaStream_t stream, bool embed_q4) {
  if (B <= 0) return;
  batch_embed_block_kernel<<<dim3(BLOCK, B), dim3(ROW_THREADS), 0, stream>>>(embed, embed_norm, mask_embed, in_norm,
                                                                             meta, res, xn, picks, embed_q4);
  check_b();
}

void batch_qkv_rope(const void* qkv, bool qkv_fp32, const ActT* q_norm, const ActT* k_norm,
                    const BatchSlotDraftMeta* meta, int B, ActT* q_out, void* k_cache, void* v_cache,
                    std::size_t kv_slot_bytes, bool fp8, bool local, int ctx, cudaStream_t stream) {
  if (B <= 0) return;
  const dim3 grid(BLOCK, HEADS + 2 * KV_HEADS, B);
  if (qkv_fp32) {
    batch_qkv_rope_kernel<true><<<grid, dim3(HEAD_DIM), 0, stream>>>(qkv, q_norm, k_norm, meta, q_out, k_cache, v_cache,
                                                                     kv_slot_bytes, fp8, local, ctx);
  } else {
    batch_qkv_rope_kernel<false><<<grid, dim3(HEAD_DIM), 0, stream>>>(qkv, q_norm, k_norm, meta, q_out, k_cache,
                                                                      v_cache, kv_slot_bytes, fp8, local, ctx);
  }
  check_b();
}

std::size_t batch_attention_workspace_floats(int max_b, int /*max_nkeys*/) {
  const std::size_t max_total_chunks = std::size_t(std::max(ATT_MAX_CHUNKS, max_b));
  return max_total_chunks * KV_HEADS * NQ * (HEAD_DIM + 2);
}

void batch_block_attention(const ActT* q, const void* k_cache, const void* v_cache, std::size_t kv_slot_bytes,
                           int nkeys, const std::int32_t* ctx_pos_base, bool fp8, const BatchSlotDraftMeta* meta,
                           int B, float* ws, ActT* out, cudaStream_t stream) {
  if (B <= 0) return;
  if (nkeys <= 0) throw std::invalid_argument("batch_block_attention: nkeys must be positive");
  const int chunks = batch_att_chunks(B, nkeys);
  const int chunk_keys = ATT_TILE * batch_att_tiles_per_chunk(B, nkeys);
  float* ws_o = ws;
  float* ws_ml = ws + std::size_t(B) * chunks * KV_HEADS * NQ * HEAD_DIM;
  const dim3 grid(chunks, KV_HEADS, B);
  const auto* k = static_cast<const uint8_t*>(k_cache);
  const auto* v = static_cast<const uint8_t*>(v_cache);
  if (ctx_pos_base != nullptr) {
    batch_attn_partial_kernel<true, false><<<grid, dim3(ATT_THREADS), 0, stream>>>(
        q, k, v, kv_slot_bytes, nkeys, chunk_keys, chunks, ctx_pos_base, meta, ws_o, ws_ml);
  } else if (fp8) {
    batch_attn_partial_kernel<false, true><<<grid, dim3(ATT_THREADS), 0, stream>>>(
        q, k, v, kv_slot_bytes, nkeys, chunk_keys, chunks, ctx_pos_base, meta, ws_o, ws_ml);
  } else {
    batch_attn_partial_kernel<false, false><<<grid, dim3(ATT_THREADS), 0, stream>>>(
        q, k, v, kv_slot_bytes, nkeys, chunk_keys, chunks, ctx_pos_base, meta, ws_o, ws_ml);
  }
  batch_attn_combine_kernel<<<dim3(KV_HEADS * NQ, B), dim3(HEAD_DIM), 0, stream>>>(ws_o, ws_ml, chunks, meta, out);
  check_b();
}

void batch_add_rmsnorm(ActT* res, const void* delta, bool delta_fp32, const ActT* w, ActT* xn, int rows,
                       cudaStream_t stream) {
  if (rows <= 0) return;
  if (delta_fp32) {
    batch_add_rmsnorm_kernel<true><<<dim3(rows), dim3(ROW_THREADS), 0, stream>>>(res, delta, w, xn);
  } else {
    batch_add_rmsnorm_kernel<false><<<dim3(rows), dim3(ROW_THREADS), 0, stream>>>(res, delta, w, xn);
  }
  check_b();
}

void batch_silu_mul(const void* gu, bool gu_fp32, int rows, int inter, ActT* out, cudaStream_t stream) {
  const std::size_t n = std::size_t(rows) * inter;
  if (n == 0) return;
  const unsigned blocks = unsigned((n + 255) / 256);
  if (gu_fp32) {
    batch_silu_mul_kernel<true><<<blocks, 256, 0, stream>>>(gu, rows, inter, out);
  } else {
    batch_silu_mul_kernel<false><<<blocks, 256, 0, stream>>>(gu, rows, inter, out);
  }
  check_b();
}

void batch_extract_gamma_rows(const ActT* xn, int B, ActT* h_gamma, cudaStream_t stream) {
  if (B <= 0) return;
  batch_extract_gamma_rows_kernel<<<dim3(B * GAMMA), dim3(ROW_THREADS), 0, stream>>>(xn, B, h_gamma);
  check_b();
}

void batch_cast_base_chunk(const ActT* src, int rows, int chunk_V, int v_offset, int dst_stride, float* dst,
                           cudaStream_t stream) {
  if (rows <= 0 || chunk_V <= 0) return;
  batch_cast_base_chunk_kernel<<<dim3((chunk_V + 255) / 256, rows), dim3(256), 0, stream>>>(src, rows, chunk_V,
                                                                                            v_offset, dst_stride, dst);
  check_b();
}

void batch_copy_base_rows(const float* src, int rows, int src_stride, int dst_stride, float* dst, cudaStream_t stream) {
  if (rows <= 0 || src_stride <= 0) return;
  batch_copy_base_rows_kernel<<<dim3((src_stride + 255) / 256, rows), dim3(256), 0, stream>>>(src, rows, src_stride,
                                                                                              dst_stride, dst);
  check_b();
}

void batch_score_slot_extra(const ActT* h_gamma, int B, const void* unembed_w, const float* unembed_scale,
                            const std::int32_t* slot_extra_ids, int max_extra, const BatchSlotDraftMeta* meta,
                            int short_K, int base_stride, float* base, cudaStream_t stream) {
  if (B <= 0 || max_extra <= 0) return;
  const dim3 grid((max_extra + 3) / 4, B);
  if (unembed_scale != nullptr) {
    batch_score_slot_extra_kernel<true><<<grid, dim3(128), 0, stream>>>(h_gamma, unembed_w, unembed_scale,
                                                                        slot_extra_ids, max_extra, meta, short_K,
                                                                        base_stride, base);
  } else {
    batch_score_slot_extra_kernel<false><<<grid, dim3(128), 0, stream>>>(h_gamma, unembed_w, unembed_scale,
                                                                         slot_extra_ids, max_extra, meta, short_K,
                                                                         base_stride, base);
  }
  check_b();
}

std::size_t batch_topk_workspace_bytes(int max_rows) { return std::size_t(max_rows) * sizeof(BatchTopkRowWs); }

void batch_topk_candidates(const float* base, int B, int base_stride, int short_K, const std::int32_t* short_map,
                           const std::int32_t* slot_extra_ids, int max_extra, const BatchSlotDraftMeta* meta,
                           float* cand_val, std::int32_t* cand_idx, void* ws, cudaStream_t stream) {
  if (B <= 0) return;
  const int rows = B * GAMMA;
  auto* w = static_cast<BatchTopkRowWs*>(ws);
  const dim3 grid(B_TOPK_SLICES, rows);
  batch_topk_hist_hi_kernel<<<grid, dim3(B_TOPK_THREADS), 0, stream>>>(base, base_stride, short_K, meta, w);
  batch_topk_hist_lo_kernel<<<grid, dim3(B_TOPK_THREADS), 0, stream>>>(base, base_stride, short_K, meta, w);
  batch_topk_collect_kernel<<<grid, dim3(B_TOPK_THREADS), 0, stream>>>(base, base_stride, short_K, meta, w);
  batch_topk_gather_kernel<<<dim3(rows), dim3(B_GATHER_THREADS), 0, stream>>>(
      w, short_K, short_map, slot_extra_ids, max_extra, meta, cand_val, cand_idx);
  check_b();
}

void batch_markov_pre(const ActT* rnn_w, const ActT* h_gamma, const ActT* f_rnn_pool, const BatchSlotDraftMeta* meta,
                      int B, float* part, cudaStream_t stream) {
  if (B <= 0) return;
  batch_markov_pre_kernel<<<dim3(RNN_OUT * 32 / PRE_THREADS, B), dim3(PRE_THREADS), 0, stream>>>(
      rnn_w, h_gamma, f_rnn_pool, meta, part);
  check_b();
}

void batch_markov_step(int i, const ActT* rnn_w, const ActT* rnn_b, const ActT* w1, const float* part,
                       const unsigned long long* picks, const std::int32_t* cand_idx, const BatchSlotDraftMeta* meta,
                       int B, ActT* state, ActT* latent, cudaStream_t stream) {
  if (B <= 0) return;
  batch_markov_step_kernel<<<dim3(RANK * 32 / 256, B), dim3(256), 0, stream>>>(
      i, rnn_w, rnn_b, w1, part, picks, cand_idx, meta, state, latent);
  check_b();
}

void batch_markov_logits(int i, const ActT* fold_w, const ActT* fold_b, const float* cand_val,
                         const std::int32_t* cand_idx, const ActT* latent, const BatchSlotDraftMeta* meta, int B,
                         float* cand_logits, unsigned long long* picks, cudaStream_t stream) {
  if (B <= 0) return;
  batch_markov_logits_kernel<<<dim3(CANDIDATES * 32 / 256, B), dim3(256), 0, stream>>>(
      i, fold_w, fold_b, cand_val, cand_idx, latent, meta, cand_logits, picks);
  check_b();
}

void batch_markov_finish(const std::int32_t* cand_idx, const float* cand_logits, const unsigned long long* picks,
                         const BatchSlotDraftMeta* meta, int B, std::int32_t* tokens, float* conf, float* qstats,
                         cudaStream_t stream) {
  if (B <= 0) return;
  batch_markov_finish_kernel<<<dim3(GAMMA, B), dim3(CANDIDATES), 0, stream>>>(cand_idx, cand_logits, picks, meta,
                                                                              tokens, conf, qstats);
  check_b();
}

void scatter_slot_draft_logits(const std::int32_t* cand_idx, const float* cand_logits, int V, float* dst_logits,
                               cudaStream_t stream) {
  scatter_slot_draft_logits_kernel<<<dim3(GAMMA), dim3(CANDIDATES), 0, stream>>>(cand_idx, cand_logits, V, dst_logits);
  check_b();
}

void clear_slot_draft_logits(const std::int32_t* cand_idx, int V, float* dst_logits, cudaStream_t stream) {
  clear_slot_draft_logits_kernel<<<dim3(GAMMA), dim3(CANDIDATES), 0, stream>>>(cand_idx, V, dst_logits);
  check_b();
}

void batch_greedy_accept(const std::int32_t* target_argmax, const std::int32_t* drafts,
                         const std::int32_t* slot_w_limit, int B, int W, std::int32_t* accepted_out,
                         std::int32_t* next_out, std::int32_t* commit_out, cudaStream_t stream) {
  if (B <= 0) return;
  batch_greedy_accept_kernel<<<(B + 63) / 64, 64, 0, stream>>>(target_argmax, drafts, slot_w_limit, B, W, accepted_out,
                                                               next_out, commit_out);
  check_b();
}

}  // namespace ingine::kernels::draft

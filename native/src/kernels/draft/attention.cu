// DSpark draft block attention (dspark.Drafter._attention) on tensor cores: partial softmax-weighted sums
// over key chunks, then a combine pass. PDL launches (see block.cu).
//
// A CTA of the partial kernel handles one kv head's 24 queries (3 heads x 8 block rows) over one chunk of
// keys, a tile of ATT_TILE = 16 keys at a time. Warp h < 3 owns query head 3 g + h (8 queries = the n of
// mma.m16n8k16); all four warps load and stage the tile:
//   load    every thread loads 16 bytes of the tile's keys and of its values (BF16 rows: two loads each), so a
//           load instruction reads whole rows; they go to shared memory as BF16 [16 keys][128] (E4M3 rows are
//           converted, one instruction per pair). Two tiles are in flight: a tile's registers are reloaded as
//           soon as it is staged. One barrier per tile (the staged tiles alternate between two buffers).
//   scores  S^T = K Q^T: A = 16 keys x 16 dims (ldmatrix), B = 16 dims x 8 queries (registers), 8 mma.
//   softmax running max/sum per query over the lanes holding its 16 keys; probabilities rounded to BF16.
//   values  O^T += V^T P^T: A = 16 dims x 16 keys (ldmatrix.trans of the staged values), B = 16 keys x 8
//           queries (the probabilities, transposed through shared memory), 8 mma.
// About 350 warp instructions per tile (8 KB of BF16 rows), where the CUDA-core kernel this replaces took
// 0.5 per byte and was compute bound (64K keys: 305 us for 268 MB; local layers: 10 us for 4 MB).
#include "kernels/common/l2.cuh"
#include "kernels/common/mma_sm70.cuh"
#include "kernels/draft/draft.h"
#include "kernels/draft/draft_device.cuh"

#include <algorithm>
#include <stdexcept>

namespace ingine::kernels::draft {
namespace {

constexpr int QH = HEADS / KV_HEADS;  // 3 query heads per kv head
constexpr int NQ = QH * BLOCK;        // 24 queries per kv head: query qi = head * 8 + row
constexpr int ATT_TILE = 16;          // keys per tile (the m of the score mma, the k of the value mma)
constexpr int ATT_THREADS = 128;      // warps 0..2: a query head each; all four stage the tiles
constexpr int ATT_MAX_CHUNKS = 128;   // chunks per kv head (at most; one combine thread each)
#ifndef ATT_MIN_TILES
#define ATT_MIN_TILES 2               // tiles per chunk (at least)
#endif
static_assert(QH == 3 && BLOCK == 8 && HEAD_DIM == 128 && ATT_MAX_CHUNKS <= HEAD_DIM);

int att_tiles_per_chunk(int nkeys) {
  return std::max(ATT_MIN_TILES, (nkeys + ATT_TILE * ATT_MAX_CHUNKS - 1) / (ATT_TILE * ATT_MAX_CHUNKS));
}
int att_chunks(int nkeys) {
  const int ck = ATT_TILE * att_tiles_per_chunk(nkeys);
  return (nkeys + ck - 1) / ck;
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  mma::mma_act(c, a[0], a[1], a[2], a[3], b[0], b[1]);
}
// ldmatrix: lane l < 8 n gives the address of row l % 8 (8 BF16) of matrix l / 8; every lane (g, t) receives
// elements (g, 2 t), (g, 2 t + 1) of each matrix, or with `trans` elements (2 t, g), (2 t + 1, g).
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* row) {
  mma::ldmatrix_x4(r[0], r[1], r[2], r[3], static_cast<unsigned>(__cvta_generic_to_shared(row)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* row) {
  mma::ldmatrix_x4_trans(r[0], r[1], r[2], r[3], static_cast<unsigned>(__cvta_generic_to_shared(row)));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* row) {
  mma::ldmatrix_x2(r[0], r[1], static_cast<unsigned>(__cvta_generic_to_shared(row)));
}

// 16-byte coherent load issued where it is written (queries and cache rows are written by earlier kernels).
// The caches are read once per draft: no L1 allocation.
__device__ __forceinline__ uint4 ld_cache(const void* p) {
  uint4 r;
  asm volatile("ld.global" INGINE_L1_NO_ALLOCATE ".v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
               : "l"(p)
               : "memory");
  return r;
}

// A tile's raw keys and values in registers (16 bytes per thread and load) and, in threads 0..15 of local
// layers, the ring position of key t0 + thread.
template <bool FP8>
struct TileRegs {
  static constexpr int NL = FP8 ? 1 : 2;  // loads per thread: a tile's keys are 16 x 128 (x 2) bytes
  uint4 k[NL], v[NL];
  int cp;
};

// grid (chunks, KV_HEADS), block 128. LOCAL: keys = ring rows [0, WINDOW) valid when P - WINDOW <= ctx_pos < P
// plus the block rows; else keys [0, nkeys) valid below P + 8. FP8: E4M3 cache rows (else BF16).
// ws_o fp32 [chunks, 8, 24 queries, 128 dims], ws_ml fp32 [chunks, 8, 24, 2] (running max, sum).
template <bool LOCAL, bool FP8>
__global__ void __launch_bounds__(ATT_THREADS) attn_partial_kernel(const ActT* q, const uint8_t* k_cache,
                                                                  const uint8_t* v_cache, int nkeys, int chunk_keys,
                                                                  const std::int32_t* ctx_pos, const DraftParams* p,
                                                                  float* ws_o, float* ws_ml) {
  constexpr int ROW_BYTES = FP8 ? HEAD_DIM : 2 * HEAD_DIM;
  constexpr int NL = TileRegs<FP8>::NL;
  __shared__ __align__(16) ActT qs[QH][BLOCK][HEAD_DIM];    // the queries, by head
  __shared__ __align__(16) ActT kt[2][ATT_TILE][HEAD_DIM];  // keys of a tile, by tile parity
  __shared__ __align__(16) ActT vt[2][ATT_TILE][HEAD_DIM];  // values
  __shared__ __align__(16) ActT pt[QH][BLOCK][ATT_TILE];    // a head's probabilities [query][key]
  __shared__ unsigned vmask[2];                                      // valid keys of a tile
  pdl_trigger();
  pdl_wait();
  const int chunk = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int gl = lane >> 2, t = lane & 3, lm = lane >> 3, lr = lane & 7;  // ldmatrix: matrix lm, row lr
  const int k0 = chunk * chunk_keys, k1 = min(k0 + chunk_keys, nkeys);
  const int P = p->P;
  const int kend = LOCAL ? k1 : min(k1, P + BLOCK);  // global: no valid key at or beyond P + 8
  constexpr float scale = 0.08838834764831845f;      // 1 / sqrt(128)

  auto load = [&](TileRegs<FP8>& x, int t0) {
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const int idx = tid + ATT_THREADS * j;
      const int key = t0 + idx / (ROW_BYTES / 16), piece = idx % (ROW_BYTES / 16);
      const size_t at = (size_t(key) * KV_HEADS + g) * ROW_BYTES + 16 * piece;
      const bool live = key < kend;
      x.k[j] = live ? ld_cache(k_cache + at) : make_uint4(0, 0, 0, 0);
      x.v[j] = live ? ld_cache(v_cache + at) : make_uint4(0, 0, 0, 0);
    }
    x.cp = 0;
    if constexpr (LOCAL) {
      if (tid < ATT_TILE && t0 + tid < WINDOW) x.cp = ctx_pos[t0 + tid];
    }
  };
  TileRegs<FP8> ra, rb;
  if (k0 < kend) load(ra, k0);
  if (k0 + ATT_TILE < kend) load(rb, k0 + ATT_TILE);

  // the queries of heads 3 g + (0, 1, 2), rows 0..7
  {
    uint4 qv[QH];
#pragma unroll
    for (int j = 0; j < QH; ++j) {
      const int idx = tid + ATT_THREADS * j, hh = idx >> 7, r = (idx >> 4) & 7, piece = idx & 15;
      qv[j] = *reinterpret_cast<const uint4*>(q + (size_t(r) * HEADS + g * QH + hh) * HEAD_DIM + 8 * piece);
    }
#pragma unroll
    for (int j = 0; j < QH; ++j) {
      const int idx = tid + ATT_THREADS * j;
      *reinterpret_cast<uint4*>(&qs[idx >> 7][(idx >> 4) & 7][8 * (idx & 15)]) = qv[j];
    }
  }
  __syncthreads();
  const bool head = w < QH;       // warp-uniform
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
  // Pascal sm_60/sm_61: reads qs directly from SMEM via uint4 in tile() with exact FP32 fmaf (0 shuffles).
#else
  uint32_t qb[HEAD_DIM / 16][2];  // this head's queries as B fragments (16 dims x 8 queries), per 16 dims
#pragma unroll
  for (int s = 0; s < HEAD_DIM / 16; ++s) ldmatrix_x2(qb[s], &qs[hw][lr][16 * s + 8 * (lm & 1)]);
#endif

  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};  // queries 2 t, 2 t + 1
  float oacc[HEAD_DIM / 16][4];  // value sums: dims 16 d + gl (0, 1) and 16 d + 8 + gl (2, 3) of queries 2 t, 2 t + 1
#pragma unroll
  for (int d = 0; d < HEAD_DIM / 16; ++d)
#pragma unroll
    for (int i = 0; i < 4; ++i) oacc[d][i] = 0.f;

  auto tile = [&](TileRegs<FP8>& x, int t0, int buf) {
    // stage the tile as BF16 [16 keys][128]
#pragma unroll
    for (int j = 0; j < NL; ++j) {
      const int idx = tid + ATT_THREADS * j;
      const int key = idx / (ROW_BYTES / 16), piece = idx % (ROW_BYTES / 16);
      if constexpr (FP8) {
        const uint32_t kw[4] = {x.k[j].x, x.k[j].y, x.k[j].z, x.k[j].w}, vw[4] = {x.v[j].x, x.v[j].y, x.v[j].z, x.v[j].w};
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
    // these registers are free: the tile after the next one goes in flight (block-uniform condition)
    if (t0 + 2 * ATT_TILE < kend) load(x, t0 + 2 * ATT_TILE);
    if (!head) return;

    // scores: keys gl (0, 1) and gl + 8 (2, 3) against queries 2 t, 2 t + 1
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
      ldmatrix_x4(a, &kt[buf][8 * (lm & 1) + lr][16 * s + 8 * (lm >> 1)]);
      mma_bf16(sc, a, qb[s]);
    }
#endif
    const unsigned mask = vmask[buf];
    const bool valid[2] = {((mask >> gl) & 1u) != 0, ((mask >> (gl + 8)) & 1u) != 0};
    float al[2];
    ActT pb16[2][2];  // [query e][key gl + 8 kk]
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
        pb16[e][kk] = act::from_float(valid[kk] ? __expf(xs[kk] - m_new) : 0.f);  // the value mma takes BF16
        ps += act::to_float(pb16[e][kk]);
      }
#pragma unroll
      for (int sh = 4; sh < 32; sh <<= 1) ps += __shfl_xor_sync(0xffffffffu, ps, sh);
      l_run[e] = l_run[e] * al[e] + ps;
      m_run[e] = m_new;
    }
    // the probabilities as a B fragment (16 keys x 8 queries): transposed through shared memory
#pragma unroll
    for (int e = 0; e < 2; ++e)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) pt[w][2 * t + e][gl + 8 * kk] = pb16[e][kk];
    __syncwarp();
    // O^T = O^T alpha + V^T P^T: the tile's 16 keys are the k of the mma
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
    ldmatrix_x2(pf, &pt[w][lr][8 * (lm & 1)]);
#pragma unroll
    for (int d = 0; d < HEAD_DIM / 16; ++d) {
      if (rescale) {
        oacc[d][0] *= al[0];
        oacc[d][1] *= al[1];
        oacc[d][2] *= al[0];
        oacc[d][3] *= al[1];
      }
      uint32_t a[4];
      ldmatrix_x4_trans(a, &vt[buf][8 * (lm >> 1) + lr][16 * d + 8 * (lm & 1)]);
      mma_bf16(oacc[d], a, pf);
    }
#endif
  };
  for (int t0 = k0; t0 < kend; t0 += 2 * ATT_TILE) {
    tile(ra, t0, 0);
    if (t0 + ATT_TILE < kend) tile(rb, t0 + ATT_TILE, 1);
  }
  if (!head) return;

  // partial outputs of queries qi = 8 w + 2 t + (0, 1) at dims 16 d + gl and 16 d + 8 + gl
  const size_t base = size_t(chunk) * KV_HEADS + g;
  const int qi = BLOCK * w + 2 * t;
#pragma unroll
  for (int d = 0; d < HEAD_DIM / 16; ++d) {
    float* at = ws_o + (base * NQ + qi) * HEAD_DIM + 16 * d + gl;  // query qi; the next query is 128 on
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

// grid KV_HEADS * NQ, block 128: thread c weighs chunk c, then every thread sums its dim over the chunks
__global__ void __launch_bounds__(HEAD_DIM) attn_combine_kernel(const float* ws_o, const float* ws_ml, int chunks,
                                                               ActT* out) {
  __shared__ float scratch[33];
  __shared__ float f[HEAD_DIM];
  pdl_trigger();
  pdl_wait();
  const int g = blockIdx.x / NQ, qi = blockIdx.x % NQ, d = threadIdx.x;
  float mc = -INFINITY, lc = 0.f;
  if (d < chunks) {
    const float2 ml = *reinterpret_cast<const float2*>(ws_ml + ((size_t(d) * KV_HEADS + g) * NQ + qi) * 2);
    mc = ml.x;
    lc = ml.y;
  }
  const float m = block_max(mc, scratch);
  const float fc = mc == -INFINITY ? 0.f : __expf(mc - m);
  f[d] = fc;
  const float l = block_sum(lc * fc, scratch);  // its barriers publish f
  float o = 0.f;
#pragma unroll 8
  for (int c = 0; c < chunks; ++c) o += ws_o[((size_t(c) * KV_HEADS + g) * NQ + qi) * HEAD_DIM + d] * f[c];
  const int hh = qi / BLOCK, r = qi % BLOCK, h = g * QH + hh;
  out[size_t(r) * Q_DIM + h * HEAD_DIM + d] = act::from_float(l > 0.f ? o / l : 0.f);
}

}  // namespace

std::size_t attention_workspace_floats(int nkeys) {
  return std::size_t(att_chunks(nkeys)) * KV_HEADS * NQ * (HEAD_DIM + 2);
}

void block_attention(const ActT* q, const void* k_cache, const void* v_cache, int nkeys,
                     const std::int32_t* ctx_pos, bool fp8, const DraftParams* p, float* ws, ActT* out,
                     cudaStream_t stream) {
  if (nkeys <= 0) throw std::invalid_argument("block_attention: nkeys must be positive");
  if (ctx_pos != nullptr && (nkeys != WINDOW + BLOCK || fp8))
    throw std::invalid_argument("block_attention: local layers attend to WINDOW + 8 BF16 keys");
  const int chunks = att_chunks(nkeys), chunk_keys = ATT_TILE * att_tiles_per_chunk(nkeys);
  float* ws_o = ws;
  float* ws_ml = ws + std::size_t(chunks) * KV_HEADS * NQ * HEAD_DIM;
  const dim3 grid(chunks, KV_HEADS), block(ATT_THREADS);
  const auto* k = static_cast<const uint8_t*>(k_cache);
  const auto* v = static_cast<const uint8_t*>(v_cache);
  if (ctx_pos != nullptr)
    launch_pdl(attn_partial_kernel<true, false>, grid, block, stream, q, k, v, nkeys, chunk_keys, ctx_pos, p, ws_o, ws_ml);
  else if (fp8)
    launch_pdl(attn_partial_kernel<false, true>, grid, block, stream, q, k, v, nkeys, chunk_keys, ctx_pos, p, ws_o, ws_ml);
  else
    launch_pdl(attn_partial_kernel<false, false>, grid, block, stream, q, k, v, nkeys, chunk_keys, ctx_pos, p, ws_o,
               ws_ml);
  launch_pdl(attn_combine_kernel, dim3(KV_HEADS * NQ), dim3(HEAD_DIM), stream, static_cast<const float*>(ws_o),
             static_cast<const float*>(ws_ml), chunks, out);
}

}  // namespace ingine::kernels::draft

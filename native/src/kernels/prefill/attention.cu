#include <atomic>
// Prefill attention (flex.attend + model._visit's output gate) as a flash-attention kernel for 24 query /
// 8 KV heads, head dim 128.
//
// Design. A CTA owns one KV head and 128 query rows of the chunk, rows ordered (token, head of the GQA group):
// row R is token R / 3, head 3 g + R % 3, so all of them read the same keys and values and every key/value
// tile in shared memory serves 128 rows. 8 warps of 16 rows. Q (BF16) stays in registers as m16n8k16 A
// fragments. Per 64-key tile: S = Q K^T with BF16 MMA and FP32 accumulation (BF16 values are exact, as in
// flex's TF32 dots), scale 1/sqrt(128), relative bias rel_logits[t, h, d] for distances d < 1024, mask
// (causal, window, empty slots), online softmax in FP32 (base 2), P = FP16(p * 2^15) (an FP16 / TF32 10-bit
// mantissa without FP16's underflow below 2^-14), O += P V with FP16 MMA (values converted BF16/E4M3 -> FP16
// when staged: exact for the model's RMS-normed values), FP32 accumulation. The row sum uses the FP32 p.
// Epilogue: o = O 2^-15 / l, lse = ln(sum e^s), o * sigmoid(lse - sink) rounded to BF16, times
// sigmoid(gate), rounded to BF16.
// Keys/values are staged through registers into double-buffered swizzled shared memory: tile i + 1 is loaded
// from global memory at the start of tile i and stored to the other buffer right after tile i's Q.K^T (one
// barrier per tile); fragment loads of a k16 group are issued before its MMAs. Keys are either implicit (key j at
// position j: the full-attention cache prefix) or explicit (kpos, the sliding ring + chunk rows) with per-tile
// position bounds deciding which tiles are skipped, need no mask, or need the bias. CTAs run heaviest row tiles
// (latest tokens) first.
#include "core/device.h"
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

constexpr int D = kHeadDim;   // 128
constexpr int DCH = D / 8;    // 16-byte chunks per 16-bit row
constexpr int BK = kAttnKeyTile;
constexpr int NWARPS = 8, NT = NWARPS * 32, BR = NWARPS * 16;  // 128 query rows per CTA
constexpr int GQA = kHeads / kKvHeads;
constexpr int GATE_COL = kHeads * kHeadDim;                       // 3072
constexpr int SINK_COL = 2 * kHeads * kHeadDim + kHeads * kRel;   // 6528
constexpr float kScale = 0.08838834764831845f;                    // 128 ** -0.5
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
[[maybe_unused]] constexpr float kPScale = 32768.f, kPUnscale = 1.f / 32768.f;

template <int STAGES>
struct Smem {
  uint16_t k[STAGES][BK * D];  // BF16 keys, swizzled [64][16 chunks]
  uint16_t v[STAGES][BK * D];  // FP16 values
  int kpos[STAGES][BK];
  int4 bounds[kAttnMaxTiles];
  int list[kAttnMaxTiles];
  int nlist;
};
static_assert(sizeof(Smem<1>) <= 49152, "single-stage prefill attention must fit 48 KiB SMEM");
static_assert(offsetof(Smem<1>, v) == sizeof(Smem<1>::k), "Q staging spans k and v contiguously");
static_assert(sizeof(Smem<1>::k) + sizeof(Smem<1>::v) >= BR * D * sizeof(uint16_t), "Q staging must fit k+v");

// KV_FMT: 0 = BF16, 1 = FP8 E4M3, 2 = Q4_0 (per-32-element block, 576B/row), 3 = FP8 K + FP4 V (paged pool)
template <int KV_FMT>
struct KvRegs {
  static constexpr int N = (KV_FMT == 1 || KV_FMT == 3) ? 2 : (KV_FMT == 2 ? 1 : 4);  // 16-byte loads per thread
  uint4 k[N], v[N];
  uint16_t ks, vs;  // Q4_0 FP16 block scales (unused for BF16/FP8)
  int kp;
};

// element offset of key row j: contiguous rows, or the TP batch pool's 64-token pages
__device__ __forceinline__ size_t kv_row_off(const AttnParams& p, int j) {
  if (p.kv_page_row)
    return ((size_t(__ldg(p.kv_page_row + (j >> 6))) * p.kv_page_visits + p.kv_page_visit) * 64 + (j & 63)) * kKvDim;
  return size_t(j) * kKvDim;
}

// E2M1 code (sign bit 8) -> value / 1.5 (the FP4 V pool's code book; same FP32 value as the batch decode)
__device__ __forceinline__ float fp4v_f(uint32_t c) {
  const uint32_t m = c & 7u;
  const float a = m < 2u ? 0.5f * float(m) : float(2u + (m & 1u)) * __int_as_float(int(125u + (m >> 1)) << 23);
  const float v = a * (1.f / 1.5f);
  return (c & 8u) ? -v : v;
}
__device__ __forceinline__ uint4 fp4x8_to_half8(uint32_t w) {
  uint4 o;
  o.x = pack_half2(fp4v_f(w), fp4v_f(w >> 4));
  o.y = pack_half2(fp4v_f(w >> 8), fp4v_f(w >> 12));
  o.z = pack_half2(fp4v_f(w >> 16), fp4v_f(w >> 20));
  o.w = pack_half2(fp4v_f(w >> 24), fp4v_f(w >> 28));
  return o;
}

template <int KV_FMT, bool EXPLICIT>
__device__ __forceinline__ void load_tile(const AttnParams& p, int g, int j0, KvRegs<KV_FMT>& r) {
  const int tid = threadIdx.x;
  if constexpr (KV_FMT == 2) {
    // 64 keys x 4 blocks/head = 256 threads: exactly one Q4_0 block (32 elems) per thread
    const int row = tid >> 2, blk = tid & 3;
    const int j = j0 + row;
    if (j < p.L) {
      const uint8_t* Krow;
      const uint8_t* Vrow;
      if (p.k_pages) {
        const int mask = (1 << p.page_shift) - 1;
        Krow = static_cast<const uint8_t*>(p.k_pages[j >> p.page_shift]) + size_t(j & mask) * kKvQ4RowBytes;
        Vrow = static_cast<const uint8_t*>(p.v_pages[j >> p.page_shift]) + size_t(j & mask) * kKvQ4RowBytes;
      } else {
        Krow = static_cast<const uint8_t*>(p.k) + size_t(j) * kKvQ4RowBytes;
        Vrow = static_cast<const uint8_t*>(p.v) + size_t(j) * kKvQ4RowBytes;
      }
      r.k[0] = __ldg(reinterpret_cast<const uint4*>(Krow + g * 64 + blk * 16));
      r.v[0] = __ldg(reinterpret_cast<const uint4*>(Vrow + g * 64 + blk * 16));
      r.ks = __ldg(reinterpret_cast<const uint16_t*>(Krow + kKvQ4NibbleBytes + g * 8 + blk * 2));
      r.vs = __ldg(reinterpret_cast<const uint16_t*>(Vrow + kKvQ4NibbleBytes + g * 8 + blk * 2));
    } else {
      r.k[0] = r.v[0] = make_uint4(0, 0, 0, 0);
      r.ks = r.vs = 0;
    }
  } else if constexpr (KV_FMT == 1 || KV_FMT == 3) {
    const uint8_t* K = static_cast<const uint8_t*>(p.k);
    const uint8_t* V = static_cast<const uint8_t*>(p.v);
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 3, c = id & 7;
      const int j = j0 + row;
      if (j < p.L) {
        const size_t off = kv_row_off(p, j) + g * D + c * 16;
        r.k[i] = __ldg(reinterpret_cast<const uint4*>(K + off));
        if constexpr (KV_FMT == 3) {
          const uint2 v4 = __ldg(reinterpret_cast<const uint2*>(V + off / 2));
          r.v[i] = make_uint4(v4.x, v4.y, 0u, 0u);
        } else {
          r.v[i] = __ldg(reinterpret_cast<const uint4*>(V + off));
        }
      } else {
        r.k[i] = r.v[i] = make_uint4(0, 0, 0, 0);
      }
    }
  } else {
    const uint16_t* K = static_cast<const uint16_t*>(p.k);
    const uint16_t* V = static_cast<const uint16_t*>(p.v);
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 4, c = id & 15;
      const int j = j0 + row;
      if (j < p.L) {
        const size_t off = kv_row_off(p, j) + g * D + c * 8;
        r.k[i] = __ldg(reinterpret_cast<const uint4*>(K + off));
        r.v[i] = __ldg(reinterpret_cast<const uint4*>(V + off));
      } else {
        r.k[i] = r.v[i] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  if constexpr (EXPLICIT) {
    if (tid < BK) r.kp = __ldg(p.kpos + j0 + tid);
  }
}

__device__ __forceinline__ uint32_t bf16x2_to_half2(uint32_t u) {
#if INGINE_ACT_FP16
  return u;
#else
  return pack_half2(bf_lo(u), bf_hi(u));
#endif
}

__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint16_t u) {
#if INGINE_ACT_FP16
  return e4m3x2_to_half2(u);
#else
  const float2 f = e4m3x2_to_float2(u);
  return pack_bf16x2(f.x, f.y);
#endif
}

template <int KV_FMT, bool EXPLICIT, int STAGES>
__device__ __forceinline__ void store_tile(Smem<STAGES>& s, int buf, const KvRegs<KV_FMT>& r) {
  const int tid = threadIdx.x;
  if constexpr (KV_FMT == 2) {
    const int row = tid >> 2, blk = tid & 3;
    const float kd_raw = __half2float(__ushort_as_half(r.ks));
    const float vd_raw = __half2float(__ushort_as_half(r.vs));
    const float kd = isfinite(kd_raw) ? kd_raw : 0.f;
    const float vd = isfinite(vd_raw) ? vd_raw : 0.f;
    const float k_neg8d = -8.f * kd, v_neg8d = -8.f * vd;
    const uint32_t kw[4] = {r.k[0].x, r.k[0].y, r.k[0].z, r.k[0].w};
    const uint32_t vw[4] = {r.v[0].x, r.v[0].y, r.v[0].z, r.v[0].w};
    auto safe_h2 = [](float a, float b) {
      return pack_half2(fabsf(a) <= 65504.f ? a : 0.f, fabsf(b) <= 65504.f ? b : 0.f);
    };
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
      const uint32_t ku = kw[sub], vu = vw[sub];
      uint4 kout, vout;
      kout.x = pack_bf16x2(fmaf(float(ku & 0xfu), kd, k_neg8d), fmaf(float((ku >> 4) & 0xfu), kd, k_neg8d));
      kout.y = pack_bf16x2(fmaf(float((ku >> 8) & 0xfu), kd, k_neg8d), fmaf(float((ku >> 12) & 0xfu), kd, k_neg8d));
      kout.z = pack_bf16x2(fmaf(float((ku >> 16) & 0xfu), kd, k_neg8d), fmaf(float((ku >> 20) & 0xfu), kd, k_neg8d));
      kout.w = pack_bf16x2(fmaf(float((ku >> 24) & 0xfu), kd, k_neg8d), fmaf(float((ku >> 28) & 0xfu), kd, k_neg8d));
      vout.x = safe_h2(fmaf(float(vu & 0xfu), vd, v_neg8d), fmaf(float((vu >> 4) & 0xfu), vd, v_neg8d));
      vout.y = safe_h2(fmaf(float((vu >> 8) & 0xfu), vd, v_neg8d), fmaf(float((vu >> 12) & 0xfu), vd, v_neg8d));
      vout.z = safe_h2(fmaf(float((vu >> 16) & 0xfu), vd, v_neg8d), fmaf(float((vu >> 20) & 0xfu), vd, v_neg8d));
      vout.w = safe_h2(fmaf(float((vu >> 24) & 0xfu), vd, v_neg8d), fmaf(float((vu >> 28) & 0xfu), vd, v_neg8d));
      *reinterpret_cast<uint4*>(&s.k[buf][swz<DCH>(row, 4 * blk + sub)]) = kout;
      *reinterpret_cast<uint4*>(&s.v[buf][swz<DCH>(row, 4 * blk + sub)]) = vout;
    }
  } else if constexpr (KV_FMT == 3) {
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 3, c = id & 7;
      const uint32_t kw[4] = {r.k[i].x, r.k[i].y, r.k[i].z, r.k[i].w};
      uint4 k0, k1;
      k0.x = e4m3x2_to_bf16x2(kw[0] & 0xffff); k0.y = e4m3x2_to_bf16x2(kw[0] >> 16);
      k0.z = e4m3x2_to_bf16x2(kw[1] & 0xffff); k0.w = e4m3x2_to_bf16x2(kw[1] >> 16);
      k1.x = e4m3x2_to_bf16x2(kw[2] & 0xffff); k1.y = e4m3x2_to_bf16x2(kw[2] >> 16);
      k1.z = e4m3x2_to_bf16x2(kw[3] & 0xffff); k1.w = e4m3x2_to_bf16x2(kw[3] >> 16);
      *reinterpret_cast<uint4*>(&s.k[buf][swz<DCH>(row, 2 * c)]) = k0;
      *reinterpret_cast<uint4*>(&s.k[buf][swz<DCH>(row, 2 * c + 1)]) = k1;
      *reinterpret_cast<uint4*>(&s.v[buf][swz<DCH>(row, 2 * c)]) = fp4x8_to_half8(r.v[i].x);
      *reinterpret_cast<uint4*>(&s.v[buf][swz<DCH>(row, 2 * c + 1)]) = fp4x8_to_half8(r.v[i].y);
    }
  } else if constexpr (KV_FMT == 1) {
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 3, c = id & 7;
      const uint32_t kw[4] = {r.k[i].x, r.k[i].y, r.k[i].z, r.k[i].w};
      const uint32_t vw[4] = {r.v[i].x, r.v[i].y, r.v[i].z, r.v[i].w};
      uint4 k0, k1, v0, v1;
      k0.x = e4m3x2_to_bf16x2(kw[0] & 0xffff); k0.y = e4m3x2_to_bf16x2(kw[0] >> 16);
      k0.z = e4m3x2_to_bf16x2(kw[1] & 0xffff); k0.w = e4m3x2_to_bf16x2(kw[1] >> 16);
      k1.x = e4m3x2_to_bf16x2(kw[2] & 0xffff); k1.y = e4m3x2_to_bf16x2(kw[2] >> 16);
      k1.z = e4m3x2_to_bf16x2(kw[3] & 0xffff); k1.w = e4m3x2_to_bf16x2(kw[3] >> 16);
      v0.x = e4m3x2_to_half2(vw[0] & 0xffff); v0.y = e4m3x2_to_half2(vw[0] >> 16);
      v0.z = e4m3x2_to_half2(vw[1] & 0xffff); v0.w = e4m3x2_to_half2(vw[1] >> 16);
      v1.x = e4m3x2_to_half2(vw[2] & 0xffff); v1.y = e4m3x2_to_half2(vw[2] >> 16);
      v1.z = e4m3x2_to_half2(vw[3] & 0xffff); v1.w = e4m3x2_to_half2(vw[3] >> 16);
      *reinterpret_cast<uint4*>(&s.k[buf][swz<DCH>(row, 2 * c)]) = k0;
      *reinterpret_cast<uint4*>(&s.k[buf][swz<DCH>(row, 2 * c + 1)]) = k1;
      *reinterpret_cast<uint4*>(&s.v[buf][swz<DCH>(row, 2 * c)]) = v0;
      *reinterpret_cast<uint4*>(&s.v[buf][swz<DCH>(row, 2 * c + 1)]) = v1;
    }
  } else {
#pragma unroll
    for (int i = 0; i < r.N; ++i) {
      const int id = tid + i * NT, row = id >> 4, c = id & 15;
      *reinterpret_cast<uint4*>(&s.k[buf][swz<DCH>(row, c)]) = r.k[i];
      uint4 v;
      v.x = bf16x2_to_half2(r.v[i].x);
      v.y = bf16x2_to_half2(r.v[i].y);
      v.z = bf16x2_to_half2(r.v[i].z);
      v.w = bf16x2_to_half2(r.v[i].w);
      *reinterpret_cast<uint4*>(&s.v[buf][swz<DCH>(row, c)]) = v;
    }
  }
  if constexpr (EXPLICIT) {
    if (tid < BK) s.kpos[buf][tid] = r.kp;
  }
}

template <int KV_FMT, bool EXPLICIT, int STAGES>
__global__ void __launch_bounds__(NT, 1) attn_kernel(const AttnParams p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem<STAGES>& s = *reinterpret_cast<Smem<STAGES>*>(smem_raw);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = blockIdx.y;
  const int rows = GQA * p.T;
  const int R0 = (gridDim.x - 1 - blockIdx.x) * BR;  // heaviest (latest) rows first

  // ---- Q tile -> registers (staged through the key buffers)
  {
    uint16_t* stage = &s.k[0][0];
#pragma unroll
    for (int i = 0; i < (BR * DCH) / NT; ++i) {
      const int id = tid + i * NT, r = id >> 4, c = id & 15;
      const int R = min(R0 + r, rows - 1);
      const int t = R / GQA, h = g * GQA + R % GQA;
      cp_async16(stage + swz<DCH>(r, c), p.q + (size_t(t) * kHeads + h) * D + c * 8);
    }
    cp_commit();
  }
  // per-thread rows: lane / 4 and lane / 4 + 8 of the warp's 16
  int qpos[2];
  const uint16_t* relp[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int R = min(R0 + warp * 16 + (lane >> 2) + r * 8, rows - 1);
    const int t = R / GQA, h = g * GQA + R % GQA;
    qpos[r] = p.pos0 + t;
    relp[r] = reinterpret_cast<const uint16_t*>(p.rel) + (size_t(t) * kHeads + h) * kRelExtent;
  }
  const int qmin = p.pos0 + R0 / GQA;
  const int qmax = p.pos0 + min(R0 + BR - 1, rows - 1) / GQA;

  // ---- key tiles to visit
  int first = 0, ntiles = 0;
  if constexpr (EXPLICIT) {
    const int total = (p.L + BK - 1) / BK;
    if (tid < total) s.bounds[tid] = p.bounds[tid];
    __syncthreads();
    if (tid == 0) {
      int n = 0;
      for (int i = 0; i < total; ++i) {
        const int4 b = s.bounds[i];
        const bool reach = b.w && qmax - b.x >= 0 && (p.window == 0 || qmin - b.y < p.window);
        if (reach) s.list[n++] = i;
      }
      s.nlist = n;
    }
  } else {
    const int lo_abs = p.window ? max(0, qmin - p.window + 1) : 0;
    const int hi_abs = min(qmax, p.k_pos_base + p.L - 1);
    const int lo = max(0, lo_abs - p.k_pos_base);
    const int hi = hi_abs - p.k_pos_base;
    first = lo / BK;
    ntiles = hi >= lo ? hi / BK - first + 1 : 0;
  }
  cp_wait<0>();
  __syncthreads();
  if constexpr (EXPLICIT) ntiles = s.nlist;

  unsigned qf[DCH / 2][4];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
  const int v_qp = (lane >> 2) & 3;
  const int v_idx = (lane & 3) | ((lane & 16) >> 2);
  const int v_row_a = (v_qp >= 2 ? 8 : 0) + v_idx;
  const int v_kh = (v_qp & 1) * 4;
#pragma unroll
  for (int kk = 0; kk < DCH / 2; ++kk) {
    const uint2 q0 = mma::ld_smem_u64(smem_addr(&s.k[0][0] + swz<DCH>(warp * 16 + v_row_a, kk * 2) + v_kh));
    const uint2 q1 = mma::ld_smem_u64(smem_addr(&s.k[0][0] + swz<DCH>(warp * 16 + v_row_a, kk * 2 + 1) + v_kh));
    qf[kk][0] = q0.x;
    qf[kk][1] = q0.y;
    qf[kk][2] = q1.x;
    qf[kk][3] = q1.y;
  }
#else
#pragma unroll
  for (int kk = 0; kk < DCH / 2; ++kk)
    ldmatrix_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3],
                smem_addr(&s.k[0][0] + swz<DCH>(warp * 16 + (lane & 15), kk * 2 + (lane >> 4))));
#endif
  __syncthreads();  // Q staging is overwritten by the first key tile

  auto tile_at = [&](int i) { return EXPLICIT ? s.list[i] : first + i; };

  float o[D / 8][4];
#pragma unroll
  for (int i = 0; i < D / 8; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
  if (p.partial_o && !p.partial_init) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int R = R0 + warp * 16 + (lane >> 2) + r * 8;
      if (R >= rows) continue;
      const int t = R / GQA, h = g * GQA + R % GQA;
      const size_t rh = size_t(t) * kHeads + h;
      m[r] = p.partial_m[rh];
      // Total row sum lives in partial_l[rh]; seed lane&3 == 0 so the 4-lane xor reduction stays exact.
      l[r] = (lane & 3) == 0 ? p.partial_l[rh] : 0.f;
#pragma unroll
      for (int dt = 0; dt < D / 8; ++dt) {
        const int col = dt * 8 + (lane & 3) * 2;
        o[dt][2 * r] = p.partial_o[rh * D + col];
        o[dt][2 * r + 1] = p.partial_o[rh * D + col + 1];
      }
    }
  }

  KvRegs<KV_FMT> regs;
  if (ntiles > 0) {
    load_tile<KV_FMT, EXPLICIT>(p, g, tile_at(0) * BK, regs);
    store_tile<KV_FMT, EXPLICIT>(s, 0, regs);
  }
  __syncthreads();

#pragma unroll 1
  for (int i = 0; i < ntiles; ++i) {
    const int kt = tile_at(i), buf = STAGES == 2 ? (i & 1) : 0;
    if (i + 1 < ntiles) load_tile<KV_FMT, EXPLICIT>(p, g, tile_at(i + 1) * BK, regs);

    // tile class (uniform over the CTA)
    int kmin, kmax;
    bool allv;
    if constexpr (EXPLICIT) {
      const int4 b = s.bounds[kt];
      kmin = b.x, kmax = b.y, allv = b.z != 0;
    } else {
      kmin = p.k_pos_base + kt * BK;
      kmax = p.k_pos_base + min(kt * BK + BK - 1, p.L - 1);
      allv = kt * BK + BK <= p.L;
    }
    const int dmin = qmin - kmax, dmax = qmax - kmin;
    const bool need_mask = !(allv && dmin >= 0 && (p.window == 0 || dmax < p.window));
    const bool need_bias = dmin < kRelExtent;

    // ---- S = Q K^T
    float sc[BK / 8][4];
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt) sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
    const uint16_t* Ks = s.k[buf];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
    // Native Volta HMMA.884 Q*K^T: direct SMEM K-loads + hoisted acc8 -> ZERO input shuffles, 8x fewer reductions!
    float sc_v[BK / 8][8];
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt)
#pragma unroll
      for (int v = 0; v < 8; ++v) sc_v[nt][v] = 0.f;
#pragma unroll
    for (int kk = 0; kk < DCH / 2; ++kk) {
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
        const uint2 k0 = mma::ld_smem_u64(smem_addr(Ks + swz<DCH>(nt * 8 + v_idx, kk * 2) + v_kh));
        const uint2 k1 = mma::ld_smem_u64(smem_addr(Ks + swz<DCH>(nt * 8 + v_idx, kk * 2 + 1) + v_kh));
        mma::sm70_mma_f16_direct(sc_v[nt], qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], k0.x, k0.y, k1.x, k1.y);
      }
    }
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt) mma::sm70_reduce_acc8_to_d4(sc_v[nt], sc[nt]);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
    // Pascal sm_60/sm_61 Q*K^T: hoist Q gather + FP32 unpack once across all 8 nt tiles (17x fewer shuffles),
    // read K directly from swizzled SMEM via uint4 loads (0 K shuffles), accumulate in exact FP32 fmaf.
    const int g60 = lane >> 2, t60 = lane & 3;
#pragma unroll
    for (int kk = 0; kk < DCH / 2; ++kk) {
      float2 qr0[8], qr1[8];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int sl = 4 * g60 + s;
        qr0[s] = act::unpack2(__shfl_sync(0xffffffffu, qf[kk][0], sl));
        qr1[s] = act::unpack2(__shfl_sync(0xffffffffu, qf[kk][1], sl));
        qr0[4 + s] = act::unpack2(__shfl_sync(0xffffffffu, qf[kk][2], sl));
        qr1[4 + s] = act::unpack2(__shfl_sync(0xffffffffu, qf[kk][3], sl));
      }
#pragma unroll
      for (int nt = 0; nt < BK / 8; ++nt) {
        const int ke = nt * 8 + 2 * t60, ko = ke + 1;
        const uint4 ke0 = *reinterpret_cast<const uint4*>(Ks + swz<DCH>(ke, kk * 2));
        const uint4 ke1 = *reinterpret_cast<const uint4*>(Ks + swz<DCH>(ke, kk * 2 + 1));
        const uint4 ko0 = *reinterpret_cast<const uint4*>(Ks + swz<DCH>(ko, kk * 2));
        const uint4 ko1 = *reinterpret_cast<const uint4*>(Ks + swz<DCH>(ko, kk * 2 + 1));
        const uint32_t kew[8] = {ke0.x, ke0.y, ke0.z, ke0.w, ke1.x, ke1.y, ke1.z, ke1.w};
        const uint32_t kow[8] = {ko0.x, ko0.y, ko0.z, ko0.w, ko1.x, ko1.y, ko1.z, ko1.w};
#pragma unroll
        for (int s = 0; s < 8; ++s) {
          const float2 fke = act::unpack2(kew[s]);
          const float2 fko = act::unpack2(kow[s]);
          sc[nt][0] = fmaf(qr0[s].y, fke.y, fmaf(qr0[s].x, fke.x, sc[nt][0]));
          sc[nt][1] = fmaf(qr0[s].y, fko.y, fmaf(qr0[s].x, fko.x, sc[nt][1]));
          sc[nt][2] = fmaf(qr1[s].y, fke.y, fmaf(qr1[s].x, fke.x, sc[nt][2]));
          sc[nt][3] = fmaf(qr1[s].y, fko.y, fmaf(qr1[s].x, fko.x, sc[nt][3]));
        }
      }
    }
#else
#pragma unroll
    for (int kk = 0; kk < DCH / 2; ++kk) {
      // all of the k16 group's key fragments first, then its 8 MMAs (each MMA no longer waits on the ldmatrix
      // issued right before it)
      unsigned b[BK / 16][4];
#pragma unroll
      for (int np = 0; np < BK / 16; ++np)
        ldmatrix_x4(b[np][0], b[np][1], b[np][2], b[np][3],
                    smem_addr(Ks + swz<DCH>(np * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1))));
#pragma unroll
      for (int np = 0; np < BK / 16; ++np) {
        mma_bf16(sc[2 * np], qf[kk], b[np][0], b[np][1]);
        mma_bf16(sc[2 * np + 1], qf[kk], b[np][2], b[np][3]);
      }
    }
#endif
    // stage the next tile now (its buffer was last read before the previous barrier): the conversion and the
    // shared stores overlap the Q.K^T MMAs in flight instead of trailing the P.V MMAs
    if constexpr (STAGES == 2) {
      if (i + 1 < ntiles) store_tile<KV_FMT, EXPLICIT>(s, buf ^ 1, regs);
    }

    // ---- scores (log2 units), bias, mask
    float rmax[2] = {m[0], m[1]};
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int r = e >> 1, jl = nt * 8 + (lane & 3) * 2 + (e & 1);
        float x = sc[nt][e] * kScale;
        if (need_mask || need_bias) {
          const int j_local = kt * BK + jl;
          const int kp = EXPLICIT ? s.kpos[buf][jl] : (p.k_pos_base + j_local);
          const int d = qpos[r] - kp;
          const bool ok = kp >= 0 && d >= 0 && (p.window == 0 || d < p.window) && (EXPLICIT || j_local < p.L);
          if (need_bias && ok && d < kRelExtent) x += bf2f(__ldg(relp[r] + d));
          x = ok ? x * kLog2e : -INFINITY;
        } else {
          x *= kLog2e;
        }
        sc[nt][e] = x;
        rmax[r] = fmaxf(rmax[r], x);
      }
    }
    float mref[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 1));
      rmax[r] = fmaxf(rmax[r], __shfl_xor_sync(0xffffffffu, rmax[r], 2));
      mref[r] = rmax[r] == -INFINITY ? 0.f : rmax[r];
      const float alpha = exp2f(m[r] - mref[r]);
      m[r] = rmax[r];
      l[r] *= alpha;
#pragma unroll
      for (int dt = 0; dt < D / 8; ++dt) {
        o[dt][2 * r] *= alpha;
        o[dt][2 * r + 1] *= alpha;
      }
    }
#pragma unroll
    for (int nt = 0; nt < BK / 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float pv = exp2f(sc[nt][e] - mref[e >> 1]);
        l[e >> 1] += pv;
        sc[nt][e] = pv * kPScale;
      }
    }

    // ---- O += P V
    const uint16_t* Vs = s.v[buf];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
    // Native Volta HMMA.884 P*V: permute P once across all 16 d-tiles (16x fewer A shuffles),
    // direct SMEM V^T loads (0 B shuffles), single acc8 reduction per d-tile (4x fewer reductions).
    unsigned pa[BK / 16][4];
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      mma::sm70_permute_a(pack_half2(sc[2 * kk][0], sc[2 * kk][1]), pack_half2(sc[2 * kk][2], sc[2 * kk][3]),
                          pack_half2(sc[2 * kk + 1][0], sc[2 * kk + 1][1]),
                          pack_half2(sc[2 * kk + 1][2], sc[2 * kk + 1][3]), pa[kk][0], pa[kk][1], pa[kk][2],
                          pa[kk][3]);
    }
#pragma unroll
    for (int dt = 0; dt < D / 8; ++dt) {
      const int d_col = dt * 8 + v_idx;
      const int d_ch = d_col >> 3, d_in = d_col & 7;
      float ov[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) {
        const int k0 = kk * 16 + v_kh;
        const unsigned v0 = mma::ld_smem_u16(smem_addr(Vs + swz<DCH>(k0 + 0, d_ch) + d_in));
        const unsigned v1 = mma::ld_smem_u16(smem_addr(Vs + swz<DCH>(k0 + 1, d_ch) + d_in));
        const unsigned v2 = mma::ld_smem_u16(smem_addr(Vs + swz<DCH>(k0 + 2, d_ch) + d_in));
        const unsigned v3 = mma::ld_smem_u16(smem_addr(Vs + swz<DCH>(k0 + 3, d_ch) + d_in));
        const unsigned v8 = mma::ld_smem_u16(smem_addr(Vs + swz<DCH>(k0 + 8, d_ch) + d_in));
        const unsigned v9 = mma::ld_smem_u16(smem_addr(Vs + swz<DCH>(k0 + 9, d_ch) + d_in));
        const unsigned v10 = mma::ld_smem_u16(smem_addr(Vs + swz<DCH>(k0 + 10, d_ch) + d_in));
        const unsigned v11 = mma::ld_smem_u16(smem_addr(Vs + swz<DCH>(k0 + 11, d_ch) + d_in));
        mma::sm70_mma_f16_direct(ov, pa[kk][0], pa[kk][1], pa[kk][2], pa[kk][3], v0 | (v1 << 16), v2 | (v3 << 16),
                                 v8 | (v9 << 16), v10 | (v11 << 16));
      }
      mma::sm70_reduce_acc8_to_d4(ov, o[dt]);
    }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
    // Pascal sm_60/sm_61 P*V: hoist P gather + FP32 unpack once across all 16 d-tiles (36x fewer shuffles),
    // load FP16 V[k, d0..d1] via 32-bit SMEM reads (0 V shuffles), accumulate in exact FP32 fmaf.
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      const unsigned a0 = pack_half2(sc[2 * kk][0], sc[2 * kk][1]);
      const unsigned a1 = pack_half2(sc[2 * kk][2], sc[2 * kk][3]);
      const unsigned a2 = pack_half2(sc[2 * kk + 1][0], sc[2 * kk + 1][1]);
      const unsigned a3 = pack_half2(sc[2 * kk + 1][2], sc[2 * kk + 1][3]);
      float2 pr0[8], pr1[8];
#pragma unroll
      for (int s = 0; s < 4; ++s) {
        const int sl = 4 * g60 + s;
        const uint32_t w0 = __shfl_sync(0xffffffffu, a0, sl);
        const uint32_t w1 = __shfl_sync(0xffffffffu, a1, sl);
        const uint32_t w2 = __shfl_sync(0xffffffffu, a2, sl);
        const uint32_t w3 = __shfl_sync(0xffffffffu, a3, sl);
        pr0[s] = __half22float2(*reinterpret_cast<const __half2*>(&w0));
        pr1[s] = __half22float2(*reinterpret_cast<const __half2*>(&w1));
        pr0[4 + s] = __half22float2(*reinterpret_cast<const __half2*>(&w2));
        pr1[4 + s] = __half22float2(*reinterpret_cast<const __half2*>(&w3));
      }
#pragma unroll
      for (int dt = 0; dt < D / 8; ++dt) {
        const int d_off = 2 * t60;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int kb = kk * 16 + 2 * k;
          const uint32_t v0 = mma::ld_smem_u32(smem_addr(Vs + swz<DCH>(kb, dt) + d_off));
          const uint32_t v1 = mma::ld_smem_u32(smem_addr(Vs + swz<DCH>(kb + 1, dt) + d_off));
          const float2 fv0 = __half22float2(*reinterpret_cast<const __half2*>(&v0));
          const float2 fv1 = __half22float2(*reinterpret_cast<const __half2*>(&v1));
          o[dt][0] = fmaf(pr0[k].y, fv1.x, fmaf(pr0[k].x, fv0.x, o[dt][0]));
          o[dt][1] = fmaf(pr0[k].y, fv1.y, fmaf(pr0[k].x, fv0.y, o[dt][1]));
          o[dt][2] = fmaf(pr1[k].y, fv1.x, fmaf(pr1[k].x, fv0.x, o[dt][2]));
          o[dt][3] = fmaf(pr1[k].y, fv1.y, fmaf(pr1[k].x, fv0.y, o[dt][3]));
        }
      }
    }
#else
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      const unsigned a[4] = {pack_half2(sc[2 * kk][0], sc[2 * kk][1]), pack_half2(sc[2 * kk][2], sc[2 * kk][3]),
                             pack_half2(sc[2 * kk + 1][0], sc[2 * kk + 1][1]),
                             pack_half2(sc[2 * kk + 1][2], sc[2 * kk + 1][3])};
#pragma unroll
      for (int dq = 0; dq < D / 32; ++dq) {  // two value fragment loads, then their four MMAs
        unsigned b[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h)
          ldmatrix_x4_t(b[h][0], b[h][1], b[h][2], b[h][3],
                        smem_addr(Vs + swz<DCH>(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                                                (2 * dq + h) * 2 + (lane >> 4))));
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          mma_f16(o[2 * (2 * dq + h)], a, b[h][0], b[h][1]);
          mma_f16(o[2 * (2 * dq + h) + 1], a, b[h][2], b[h][3]);
        }
      }
    }
#endif

    __syncthreads();
    if constexpr (STAGES == 1) {
      if (i + 1 < ntiles) {
        store_tile<KV_FMT, EXPLICIT>(s, 0, regs);
        __syncthreads();
      }
    }
  }

  // ---- epilogue (or save online-softmax partials for next paged KV batch)
  if (p.partial_o && !p.partial_final) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float lr = l[r];
      lr += __shfl_xor_sync(0xffffffffu, lr, 1);
      lr += __shfl_xor_sync(0xffffffffu, lr, 2);
      const int R = R0 + warp * 16 + (lane >> 2) + r * 8;
      if (R >= rows) continue;
      const int t = R / GQA, h = g * GQA + R % GQA;
      const size_t rh = size_t(t) * kHeads + h;
      if ((lane & 3) == 0) {
        p.partial_m[rh] = m[r];
        p.partial_l[rh] = lr;
      }
#pragma unroll
      for (int dt = 0; dt < D / 8; ++dt) {
        const int col = dt * 8 + (lane & 3) * 2;
        p.partial_o[rh * D + col] = o[dt][2 * r];
        p.partial_o[rh * D + col + 1] = o[dt][2 * r + 1];
      }
    }
    return;
  }
  const uint16_t* proj = reinterpret_cast<const uint16_t*>(p.proj);
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float lr = l[r];
    lr += __shfl_xor_sync(0xffffffffu, lr, 1);
    lr += __shfl_xor_sync(0xffffffffu, lr, 2);
    const int R = R0 + warp * 16 + (lane >> 2) + r * 8;
    if (R >= rows) continue;
    const int t = R / GQA, h = g * GQA + R % GQA;
    const uint16_t* prow = proj + size_t(t) * p.ld;
    const float lse = (m[r] + log2f(lr)) * kLn2;
    const float factor = 1.f / (1.f + expf(-(lse - bf2f(prow[SINK_COL + h]))));
    uint16_t* out = reinterpret_cast<uint16_t*>(p.out) + (size_t(t) * kHeads + h) * D;
#pragma unroll
    for (int dt = 0; dt < D / 8; ++dt) {
      const int col = dt * 8 + (lane & 3) * 2;
      const uint32_t gw = *reinterpret_cast<const uint32_t*>(prow + GATE_COL + h * D + col);
      const float g0 = bf_lo(gw), g1 = bf_hi(gw);
      const float a0 = round_bf16(o[dt][2 * r] * kPUnscale / lr * factor);
      const float a1 = round_bf16(o[dt][2 * r + 1] * kPUnscale / lr * factor);
      const float s0 = 1.f / (1.f + expf(-g0)), s1 = 1.f / (1.f + expf(-g1));
      *reinterpret_cast<uint32_t*>(out + col) = pack_bf16x2(a0 * s0, a1 * s1);
    }
  }
}

template <int KV_FMT, bool EXPLICIT>
void launch(const AttnParams& p, cudaStream_t stream) {
  const dim3 grid((GQA * p.T + BR - 1) / BR, kKvHeads);
  if (current_max_smem_optin() >= int(sizeof(Smem<2>))) {
    static std::atomic<uint32_t> attr_done{0};
    int dev = 0;
    cudaGetDevice(&dev);
    const uint32_t bit = 1u << (dev & 31);
    if (!(attr_done.load(std::memory_order_acquire) & bit)) {
      const cudaError_t attr = cudaFuncSetAttribute(
          attn_kernel<KV_FMT, EXPLICIT, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem<2>)));
      if (attr != cudaSuccess) throw std::runtime_error("prefill attention: shared memory attribute failed");
      attr_done.fetch_or(bit, std::memory_order_release);
    }
    attn_kernel<KV_FMT, EXPLICIT, 2><<<grid, NT, sizeof(Smem<2>), stream>>>(p);
  } else {
    // Turing (64 KiB opt-in) and Pascal (48 KiB) use single-stage K/V SMEM (34,308 B <= 48 KiB).
    attn_kernel<KV_FMT, EXPLICIT, 1><<<grid, NT, sizeof(Smem<1>), stream>>>(p);
  }
}

// ---- 128-CTA split-K Q4_0 decode chunk kernel for streamed paged KV (T <= 8)
// Replaces the 8-CTA 128-row prefill kernel during M=1..8 streamed decode so every SM stays saturated
// with coalesced 64-byte Q4_0 reads and 16 KiB SMEM (works on Pascal..Blackwell, zero FP16 ALU traps).

constexpr int kDecodeWsStride = D + 2;  // [0..127]=O, [128]=m, [129]=l

template <int M_CAP>
__global__ void __launch_bounds__(NT, 1) decode_q4_split_kernel(const AttnParams p, int nsplits) {
  const int split = blockIdx.x;
  const int g = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int M = p.T;
  const int R = M * GQA;
  const int blk = lane >> 3;
  const int d0 = lane * 4;

  __shared__ float sq[M_CAP * GQA][D];
  __shared__ float sm_row_o[NWARPS][D];
  __shared__ float sm_row_m[NWARPS];
  __shared__ float sm_row_l[NWARPS];

  const int total_q = R * D;
  for (int idx = tid; idx < total_q; idx += NT) {
    const int r = idx / D, d = idx % D;
    const int t = r / GQA, h = g * GQA + (r % GQA);
    sq[r][d] = bf2f(reinterpret_cast<const uint16_t*>(p.q)[(size_t(t) * kHeads + h) * D + d]);
  }
  __syncthreads();

  int qpos[M_CAP];
  const uint16_t* relp[M_CAP * GQA];
#pragma unroll
  for (int t = 0; t < M_CAP; ++t) qpos[t] = p.pos0 + min(t, M - 1);
#pragma unroll
  for (int r = 0; r < M_CAP * GQA; ++r) {
    const int t = min(r / GQA, M - 1), h = g * GQA + (r % GQA);
    relp[r] = reinterpret_cast<const uint16_t*>(p.rel) + (size_t(t) * kHeads + h) * kRelExtent;
  }

  float ow[M_CAP * GQA][4];
  float mw[M_CAP * GQA], lw[M_CAP * GQA];
#pragma unroll
  for (int r = 0; r < M_CAP * GQA; ++r) {
    ow[r][0] = ow[r][1] = ow[r][2] = ow[r][3] = 0.f;
    mw[r] = -INFINITY;
    lw[r] = 0.f;
  }

  const int chunk = (p.L + nsplits - 1) / nsplits;
  const int j_begin = split * chunk;
  const int j_end = min(p.L, j_begin + chunk);
  const int max_qpos = p.pos0 + M - 1;
  const int page_mask = p.k_pages ? ((1 << p.page_shift) - 1) : 0;

  for (int j = j_begin + warp; j < j_end; j += NWARPS) {
    const int kp = p.k_pos_base + j;
    if (kp > max_qpos) break;
    const uint8_t* Krow;
    const uint8_t* Vrow;
    if (p.k_pages) {
      Krow = static_cast<const uint8_t*>(p.k_pages[j >> p.page_shift]) + size_t(j & page_mask) * kKvQ4RowBytes;
      Vrow = static_cast<const uint8_t*>(p.v_pages[j >> p.page_shift]) + size_t(j & page_mask) * kKvQ4RowBytes;
    } else {
      Krow = static_cast<const uint8_t*>(p.k) + size_t(j) * kKvQ4RowBytes;
      Vrow = static_cast<const uint8_t*>(p.v) + size_t(j) * kKvQ4RowBytes;
    }
    const uint16_t knib = __ldg(reinterpret_cast<const uint16_t*>(Krow + g * 64 + lane * 2));
    const uint16_t ks_u = __ldg(reinterpret_cast<const uint16_t*>(Krow + kKvQ4NibbleBytes + g * 8 + blk * 2));
    const uint16_t vnib = __ldg(reinterpret_cast<const uint16_t*>(Vrow + g * 64 + lane * 2));
    const uint16_t vs_u = __ldg(reinterpret_cast<const uint16_t*>(Vrow + kKvQ4NibbleBytes + g * 8 + blk * 2));
    const float kd_raw = __half2float(__ushort_as_half(ks_u));
    const float vd_raw = __half2float(__ushort_as_half(vs_u));
    const float kd = isfinite(kd_raw) ? kd_raw : 0.f;
    const float vd = isfinite(vd_raw) ? vd_raw : 0.f;
    const float k_neg8 = -8.f * kd, v_neg8 = -8.f * vd;
    const float k0 = fmaf(float(knib & 0xfu), kd, k_neg8);
    const float k1 = fmaf(float((knib >> 4) & 0xfu), kd, k_neg8);
    const float k2 = fmaf(float((knib >> 8) & 0xfu), kd, k_neg8);
    const float k3 = fmaf(float((knib >> 12) & 0xfu), kd, k_neg8);
    const float v0 = fmaf(float(vnib & 0xfu), vd, v_neg8);
    const float v1 = fmaf(float((vnib >> 4) & 0xfu), vd, v_neg8);
    const float v2 = fmaf(float((vnib >> 8) & 0xfu), vd, v_neg8);
    const float v3 = fmaf(float((vnib >> 12) & 0xfu), vd, v_neg8);

#pragma unroll
    for (int r = 0; r < M_CAP * GQA; ++r) {
      if (r >= R) break;
      const int t = r / GQA;
      const int dist = qpos[t] - kp;
      if (dist < 0) continue;
      const float4 qv = *reinterpret_cast<const float4*>(&sq[r][d0]);
      float dot = fmaf(qv.x, k0, fmaf(qv.y, k1, fmaf(qv.z, k2, qv.w * k3)));
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
      float score = dot * kScale;
      if (dist < kRelExtent) score += bf2f(__ldg(relp[r] + dist));
      score *= kLog2e;
      const float m_new = fmaxf(mw[r], score);
      const float alpha = (mw[r] == -INFINITY) ? 0.f : exp2f(mw[r] - m_new);
      const float pv = exp2f(score - m_new);
      mw[r] = m_new;
      lw[r] = fmaf(lw[r], alpha, pv);
      ow[r][0] = fmaf(ow[r][0], alpha, pv * v0);
      ow[r][1] = fmaf(ow[r][1], alpha, pv * v1);
      ow[r][2] = fmaf(ow[r][2], alpha, pv * v2);
      ow[r][3] = fmaf(ow[r][3], alpha, pv * v3);
    }
  }

  for (int r = 0; r < R; ++r) {
    if (lane == 0) {
      sm_row_m[warp] = mw[r];
      sm_row_l[warp] = lw[r];
    }
    *reinterpret_cast<float4*>(&sm_row_o[warp][d0]) = make_float4(ow[r][0], ow[r][1], ow[r][2], ow[r][3]);
    __syncthreads();
    if (tid < D) {
      float mmax = -INFINITY;
#pragma unroll
      for (int w = 0; w < NWARPS; ++w) mmax = fmaxf(mmax, sm_row_m[w]);
      float lsum = 0.f, osum = 0.f;
      if (mmax > -INFINITY) {
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
          if (sm_row_l[w] > 0.f) {
            const float fac = exp2f(sm_row_m[w] - mmax);
            lsum += sm_row_l[w] * fac;
            osum = fmaf(sm_row_o[w][tid], fac, osum);
          }
        }
      }
      const int t = r / GQA, h = g * GQA + (r % GQA);
      float* dst = p.decode_ws + (((size_t(split) * M + t) * kHeads + h) * kDecodeWsStride);
      dst[tid] = osum;
      if (tid == 0) {
        dst[D] = mmax;
        dst[D + 1] = lsum;
      }
    }
    __syncthreads();
  }
}

__global__ void reduce_q4_splits_kernel(const AttnParams p, int nsplits) {
  const int rh = blockIdx.x;
  const int d = threadIdx.x;
  const int M = p.T;
  const int t = rh / kHeads, h = rh % kHeads;
  const bool resume = (p.partial_o != nullptr && !p.partial_init);

  float mmax = resume ? p.partial_m[rh] : -INFINITY;
  for (int s = 0; s < nsplits; ++s) {
    const float* src = p.decode_ws + (((size_t(s) * M + t) * kHeads + h) * kDecodeWsStride);
    mmax = fmaxf(mmax, src[D]);
  }
  float lsum = 0.f, osum = 0.f;
  if (mmax > -INFINITY) {
    if (resume) {
      const float lp = p.partial_l[rh];
      if (lp > 0.f) {
        const float fac = exp2f(p.partial_m[rh] - mmax);
        lsum = lp * fac;
        osum = p.partial_o[size_t(rh) * D + d] * fac;
      }
    }
    for (int s = 0; s < nsplits; ++s) {
      const float* src = p.decode_ws + (((size_t(s) * M + t) * kHeads + h) * kDecodeWsStride);
      const float ls = src[D + 1];
      if (ls > 0.f) {
        const float fac = exp2f(src[D] - mmax);
        lsum += ls * fac;
        osum = fmaf(src[d], fac, osum);
      }
    }
  }

  if (p.partial_o && !p.partial_final) {
    p.partial_o[size_t(rh) * D + d] = osum;
    if (d == 0) {
      p.partial_m[rh] = mmax;
      p.partial_l[rh] = lsum;
    }
    return;
  }

  const uint16_t* prow = reinterpret_cast<const uint16_t*>(p.proj) + size_t(t) * p.ld;
  const float lse = (mmax + log2f(lsum)) * kLn2;
  const float factor = 1.f / (1.f + expf(-(lse - bf2f(prow[SINK_COL + h]))));
  const float a = round_bf16((osum / lsum) * factor);
  const float gate = bf2f(prow[GATE_COL + h * D + d]);
  const float sg = 1.f / (1.f + expf(-gate));
  reinterpret_cast<uint16_t*>(p.out)[size_t(rh) * D + d] = f2bf_bits(a * sg);
}

void launch_decode_q4_split(const AttnParams& p, cudaStream_t stream) {
  const int nsplits = std::max(1, std::min(p.decode_splits, (p.L + 127) / 128));
  const dim3 grid(nsplits, kKvHeads);
  if (p.T <= 1) decode_q4_split_kernel<1><<<grid, NT, 0, stream>>>(p, nsplits);
  else if (p.T <= 2) decode_q4_split_kernel<2><<<grid, NT, 0, stream>>>(p, nsplits);
  else if (p.T <= 4) decode_q4_split_kernel<4><<<grid, NT, 0, stream>>>(p, nsplits);
  else decode_q4_split_kernel<8><<<grid, NT, 0, stream>>>(p, nsplits);
  reduce_q4_splits_kernel<<<p.T * kHeads, D, 0, stream>>>(p, nsplits);
}

// ---- sliding-visit key rows

constexpr int kRowVecs = kKvDim * 2 / 16;  // 16-byte vectors per key row (128)

__global__ void ring_gather_kernel(const uint4* __restrict__ k_ring, const uint4* __restrict__ v_ring,
                                   const int64_t* __restrict__ ring_pos, int pos0, int T, uint4* __restrict__ k_rows,
                                   uint4* __restrict__ v_rows, int* __restrict__ kpos, int kpos_rows) {
  const int i = blockIdx.x, slot = (pos0 + i) & (kRelExtent - 1);
  k_rows[size_t(i) * kRowVecs + threadIdx.x] = k_ring[size_t(slot) * kRowVecs + threadIdx.x];
  v_rows[size_t(i) * kRowVecs + threadIdx.x] = v_ring[size_t(slot) * kRowVecs + threadIdx.x];
  if (threadIdx.x == 0) kpos[i] = int(ring_pos[slot]);
  for (int r = kRelExtent + i * kRowVecs + threadIdx.x; r < kpos_rows; r += gridDim.x * kRowVecs)
    kpos[r] = r - kRelExtent < T ? pos0 + r - kRelExtent : -1;
}

__global__ void ring_update_kernel(const uint4* __restrict__ k_rows, const uint4* __restrict__ v_rows, int pos0, int T,
                                   uint4* __restrict__ k_ring, uint4* __restrict__ v_ring, int64_t* __restrict__ ring_pos) {
  const int n = min(T, kRelExtent);
  const int c = T - n + blockIdx.x;  // chunk row
  const int pos = pos0 + c, slot = pos & (kRelExtent - 1);
  k_ring[size_t(slot) * kRowVecs + threadIdx.x] = k_rows[size_t(kRelExtent + c) * kRowVecs + threadIdx.x];
  v_ring[size_t(slot) * kRowVecs + threadIdx.x] = v_rows[size_t(kRelExtent + c) * kRowVecs + threadIdx.x];
  if (threadIdx.x == 0) ring_pos[slot] = pos;
}

__global__ void bounds_kernel(const int* __restrict__ kpos, int L, int4* __restrict__ bounds) {
  const int tile = blockIdx.x, lane = threadIdx.x;
  int lo = INT_MAX, hi = INT_MIN, valid = 0;
#pragma unroll
  for (int i = 0; i < BK / 32; ++i) {
    const int j = tile * BK + i * 32 + lane;
    const int kp = j < L ? kpos[j] : -1;
    if (kp >= 0) {
      lo = min(lo, kp);
      hi = max(hi, kp);
      ++valid;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, o));
    hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, o));
    valid += __shfl_xor_sync(0xffffffffu, valid, o);
  }
  if (lane == 0) bounds[tile] = make_int4(lo, hi, valid == BK ? 1 : 0, valid > 0 ? 1 : 0);
}

}  // namespace

void attention(const AttnParams& p, cudaStream_t stream) {
  const bool have_kv = (p.k && p.v) || (p.kv_q4 && p.k_pages && p.v_pages && p.page_shift > 0);
  const bool have_out = p.partial_final ? (p.out != nullptr && p.proj != nullptr) : (p.partial_o && p.partial_m && p.partial_l);
  if (p.T <= 0 || p.L <= 0 || !p.q || !have_kv || !p.rel || !have_out)
    throw std::invalid_argument("prefill attention: missing argument");
  const bool expl = p.kpos != nullptr;
  if (expl && (!p.bounds || (p.L + BK - 1) / BK > kAttnMaxTiles))
    throw std::invalid_argument("prefill attention: explicit keys need bounds and at most 64 tiles");
  if (p.kv_fp8 && p.kv_q4)
    throw std::invalid_argument("prefill attention: kv_fp8 and kv_q4 are mutually exclusive");
  if (p.kv_page_row && (expl || p.kv_q4 || p.k_pages || p.kv_page_visits <= p.kv_page_visit || p.kv_page_visit < 0))
    throw std::invalid_argument("prefill attention: pool paging is for full layers with BF16/FP8 keys");
  if (p.kv_v4 && !(p.kv_fp8 && p.kv_page_row))
    throw std::invalid_argument("prefill attention: FP4 values need FP8 keys in the paged pool");
  if (p.kv_v4) {
    launch<3, false>(p, stream);
  } else if (p.kv_q4) {
    if (expl) throw std::invalid_argument("prefill attention: kv_q4 is supported on full layers only");
    if (p.T <= 8 && p.decode_ws && p.decode_splits > 0) {
      launch_decode_q4_split(p, stream);
    } else {
      launch<2, false>(p, stream);
    }
  } else if (p.kv_fp8) {
    if (expl) launch<1, true>(p, stream);
    else launch<1, false>(p, stream);
  } else {
    if (expl) launch<0, true>(p, stream);
    else launch<0, false>(p, stream);
  }
}

void ring_gather(const ActT* k_ring, const ActT* v_ring, const int64_t* ring_pos, int pos0, int T,
                 ActT* k_rows, ActT* v_rows, int* kpos, int kpos_rows, cudaStream_t stream) {
  ring_gather_kernel<<<kRelExtent, kRowVecs, 0, stream>>>(
      reinterpret_cast<const uint4*>(k_ring), reinterpret_cast<const uint4*>(v_ring), ring_pos, pos0, T,
      reinterpret_cast<uint4*>(k_rows), reinterpret_cast<uint4*>(v_rows), kpos, kpos_rows);
}

void ring_update(const ActT* k_rows, const ActT* v_rows, int pos0, int T, ActT* k_ring,
                 ActT* v_ring, int64_t* ring_pos, cudaStream_t stream) {
  ring_update_kernel<<<std::min(T, kRelExtent), kRowVecs, 0, stream>>>(
      reinterpret_cast<const uint4*>(k_rows), reinterpret_cast<const uint4*>(v_rows), pos0, T,
      reinterpret_cast<uint4*>(k_ring), reinterpret_cast<uint4*>(v_ring), ring_pos);
}

void key_tile_bounds(const int* kpos, int L, int4* bounds, cudaStream_t stream) {
  bounds_kernel<<<(L + BK - 1) / BK, 32, 0, stream>>>(kpos, L, bounds);
}

}  // namespace ingine::kernels::prefill

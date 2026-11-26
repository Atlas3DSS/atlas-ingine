// Decode-width routed experts on Volta (sm_70): moe_gate_up / moe_down for T <= 8 tokens, where every expert of the
// step sees at most 8 pairs. The grouped prefill GEMM (wgemm.cu) tiles 128 weight rows x 32 pair rows per CTA and
// launches a CTA per possible tile; at decode width it streamed the expert bytes at 200-400 GB/s. This kernel is
// built like the decode experts of Project Maya's ik2 engine (MIT, credited in NOTICE): mma.m8n8k4 with the
// weights as A (8 rows x 4 k) and the tile's pairs as B (4 k x 8 pairs), the four quad pairs of a warp on the four
// 16-value sub-blocks of each 64-value block, SPLIT warps over K, no shared memory but the split-K partials.
//
// Values: NVFP4 code x E4M3 block scale is exact in FP16 (3 x 4 significant bits, |v| <= 6 * 448); the expert
// factor multiplies the FP32 sum (wgemm rounds code x scale x factor per element). GGUF_Q5L / GGUF_IQ4 values are
// formed in FP32 (scale * code + min, scale * iq4nl[code]) and rounded once to FP16, as wgemm does. Epilogues are
// wgemm's: up rounds gate and up to the activation type and stores the SwiGLU, down stores the pair's row.
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <cuda_fp16.h>

#include <cstdlib>

namespace ingine::kernels::prefill {
namespace {

constexpr int FMT_NVFP4 = 0, FMT_GQ5 = 6, FMT_GQI4 = 7, FMT_GQ4 = 8;

__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ float softcap_d(float x, float cap) { return tanhf(x * (1.f / cap)) * cap; }

// FP16x2 bits of an E2M1 byte's two codes (low nibble first), exact
__device__ __forceinline__ uint32_t e2m1x2_h2(uint32_t byte) {
  const uint32_t sel = ((byte & 7u) << 4) | (((byte >> 4) & 7u) << 12);
  return __byte_perm(0x3E3C3800u, 0x46444240u, sel) | ((byte & 0x08u) << 12) | ((byte & 0x80u) << 24);
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t r;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}
// FP16 of an E4M3 byte (positive block scales), exact
__device__ __forceinline__ uint32_t e4m3_h2(uint32_t b) {
  const uint32_t a = b & 0x7fu;
  const uint32_t norm = (a + 64u) << 7;
  const uint32_t sub = a ? (0x1800u + (((0xBA986400u >> (a * 4u)) & 0xfu) << 8)) : 0u;
  const uint32_t h = a >= 8u ? norm : sub;
  return h | (h << 16);
}
__device__ __forceinline__ uint32_t f2h2(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

struct DecArgs {
  const uint8_t* codes;
  const uint8_t* scales;
  const float* factors;
  const uint8_t* const* code_ptrs;
  const uint8_t* const* scale_ptrs;
  const uint8_t* active_mask;
  const uint16_t* x;  // up: [T, H] by token; down: [pairs, I] by sorted pair
  MoeWork w;
  uint16_t* out;      // up: act [pairs, I]; down: [T * 8, H] at pair_dst
};

// a row's step (one 64-value block) for the lane's sub-block qp: 8 code bytes and the scale data
struct RowStep {
  uint2 c;
  uint32_t s0, s1, s2;
};

template <int FMT, bool UP, int SPLIT, int MAX_TILE>
__global__ void __launch_bounds__(SPLIT * 32) moe_decode_v70_kernel(const DecArgs a) {
  constexpr int K = UP ? kHidden : kInter;
  constexpr int NE = UP ? 2 * kInter : kHidden;   // expert rows
  constexpr int CODE_ROW = K / 2;
  constexpr int SCALE_ROW =
      FMT == FMT_NVFP4 ? K / 16 : (FMT == FMT_GQ5 ? (K / 32) * 8 : (K / 32) * 4);
  constexpr int NK = K / 64;
  __shared__ float part[SPLIT][8][16];
  __shared__ float lut[16];
  if (blockIdx.y >= unsigned(*a.w.tiles_n)) return;
  const int4 tile = a.w.tiles[blockIdx.y];
  const int e = tile.x, pb = tile.y, n = min(tile.z, MAX_TILE);
  if (a.active_mask && !a.active_mask[e]) return;
  if constexpr (FMT == FMT_GQI4) {
    constexpr float kIq4[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                                1.f,    13.f,   25.f,  38.f,  53.f,  69.f,  89.f,  113.f};
    if (threadIdx.x < 16) lut[threadIdx.x] = kIq4[threadIdx.x];
    __syncthreads();
  }
  const int lane = threadIdx.x & 31, split = threadIdx.x >> 5;
  const int j = lane & 3, hi = lane >> 4, qp = (lane >> 2) & 3, rr = j + 4 * hi;
  // the lane's two rows: up -> gate (2 i) and up (2 i + 1) of i = 8 blockIdx.x + rr; down -> r and r + 8
  const int rowA = UP ? 2 * (8 * int(blockIdx.x) + rr) : 16 * int(blockIdx.x) + rr;
  const int rowB = UP ? rowA + 1 : rowA + 8;
  const uint8_t* cbase = a.code_ptrs ? a.code_ptrs[e] : a.codes + int64_t(e) * NE * CODE_ROW;
  const uint8_t* sbase = a.scale_ptrs ? a.scale_ptrs[e] : a.scales + int64_t(e) * NE * SCALE_ROW;
  const uint8_t* cA = cbase + int64_t(rowA) * CODE_ROW + 8 * qp;
  const uint8_t* cB = cbase + int64_t(rowB) * CODE_ROW + 8 * qp;
  const uint8_t* sA = sbase + int64_t(rowA) * SCALE_ROW;
  const uint8_t* sB = sbase + int64_t(rowB) * SCALE_ROW;
  const bool x_ok = rr < n;
  const int xrow = x_ok ? (UP ? a.w.pair_token[pb + rr] : pb + rr) : 0;
  const uint16_t* xr = a.x + int64_t(xrow) * K + 16 * qp;
  const bool has_hi = (MAX_TILE > 8) && (n > 8);
  const bool x_ok1 = has_hi && (rr + 8 < n);
  const int xrow1 = x_ok1 ? (UP ? a.w.pair_token[pb + rr + 8] : pb + rr + 8) : 0;
  const uint16_t* xr1 = a.x + int64_t(xrow1) * K + 16 * qp;

  auto load_row = [&](const uint8_t* c, const uint8_t* s, int k) {
    RowStep r;
    r.c = __ldg(reinterpret_cast<const uint2*>(c + 32 * k));
    if constexpr (FMT == FMT_NVFP4) {
      r.s0 = __ldg(reinterpret_cast<const uint32_t*>(s + 4 * k));  // the block's four sub-block scales
    } else if constexpr (FMT == FMT_GQ5) {
      const uint2 g = __ldg(reinterpret_cast<const uint2*>(s + 8 * (2 * k + (qp >> 1))));
      r.s0 = g.x, r.s1 = g.y;  // half2(scale, min) and fifth bits of the group
    } else {
      r.s0 = __ldg(reinterpret_cast<const uint32_t*>(s + 4 * (2 * k + (qp >> 1))));
    }
    return r;
  };
  // A operands of the step's four MMAs (values 4 m .. 4 m + 3 of the sub-block) for one row
  auto decode = [&](const RowStep& r, uint32_t (&ah)[8]) {
    if constexpr (FMT == FMT_NVFP4) {
      const uint32_t sc = e4m3_h2((r.s0 >> (8 * qp)) & 0xffu);
#pragma unroll
      for (int b = 0; b < 8; ++b) ah[b] = hmul2(e2m1x2_h2(((b < 4 ? r.c.x : r.c.y) >> (8 * (b & 3))) & 0xffu), sc);
    } else if constexpr (FMT == FMT_GQ5 || FMT == FMT_GQ4) {
      const float2 sm = __half22float2(*reinterpret_cast<const __half2*>(&r.s0));
      const float sc = sm.x, mn = sm.y;
      const uint32_t hb = FMT == FMT_GQ5 ? (r.s1 >> ((qp & 1) * 16)) : 0u;  // fifth bits of the sub-block
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const uint32_t byte = ((b < 4 ? r.c.x : r.c.y) >> (8 * (b & 3))) & 0xffu;
        uint32_t lo = byte & 15u, hi4 = byte >> 4;
        if constexpr (FMT == FMT_GQ5) {
          lo |= ((hb >> (2 * b)) & 1u) << 4;
          hi4 |= ((hb >> (2 * b + 1)) & 1u) << 4;
        }
        const float vl = __fadd_rn(__fmul_rn(sc, float(lo)), mn);
        const float vh = __fadd_rn(__fmul_rn(sc, float(hi4)), mn);
        ah[b] = f2h2(vl, vh);
      }
    } else {
      const float sc = __uint_as_float(r.s0);
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const uint32_t byte = ((b < 4 ? r.c.x : r.c.y) >> (8 * (b & 3))) & 0xffu;
        const float vl = __fmul_rn(sc, lut[byte & 15u]);
        const float vh = __fmul_rn(sc, lut[byte >> 4]);
        ah[b] = f2h2(vl, vh);
      }
    }
  };

  float v[16];
  float v1[16];
#pragma unroll
  for (int m = 0; m < 16; ++m) v[m] = 0.f;
  if constexpr (MAX_TILE > 8) {
#pragma unroll
    for (int m = 0; m < 16; ++m) v1[m] = 0.f;
  }
  struct Step {
    RowStep ra, rb;
    uint4 xa, xb;
    uint4 xa1, xb1;
  };
  auto load = [&](int k) {
    Step s;
    s.ra = load_row(cA, sA, k);
    s.rb = load_row(cB, sB, k);
    s.xa = s.xb = make_uint4(0u, 0u, 0u, 0u);
    if (x_ok) {
      s.xa = __ldg(reinterpret_cast<const uint4*>(xr + 64 * k));
      s.xb = __ldg(reinterpret_cast<const uint4*>(xr + 64 * k + 8));
    }
    if constexpr (MAX_TILE > 8) {
      s.xa1 = s.xb1 = make_uint4(0u, 0u, 0u, 0u);
      if (x_ok1) {
        s.xa1 = __ldg(reinterpret_cast<const uint4*>(xr1 + 64 * k));
        s.xb1 = __ldg(reinterpret_cast<const uint4*>(xr1 + 64 * k + 8));
      }
    }
    return s;
  };
  Step nx = load(min(split, NK - 1));
#pragma unroll 1
  for (int k = split; k < NK; k += SPLIT) {
    const Step cur = nx;
    if (k + SPLIT < NK) nx = load(k + SPLIT);
    uint32_t aa[8], ab[8];
    decode(cur.ra, aa);
    decode(cur.rb, ab);
    mma884(v, aa[0], aa[1], cur.xa.x, cur.xa.y);
    mma884(v, aa[2], aa[3], cur.xa.z, cur.xa.w);
    mma884(v, aa[4], aa[5], cur.xb.x, cur.xb.y);
    mma884(v, aa[6], aa[7], cur.xb.z, cur.xb.w);
    mma884(v + 8, ab[0], ab[1], cur.xa.x, cur.xa.y);
    mma884(v + 8, ab[2], ab[3], cur.xa.z, cur.xa.w);
    mma884(v + 8, ab[4], ab[5], cur.xb.x, cur.xb.y);
    mma884(v + 8, ab[6], ab[7], cur.xb.z, cur.xb.w);
    if constexpr (MAX_TILE > 8) {
      if (has_hi) {
        mma884(v1, aa[0], aa[1], cur.xa1.x, cur.xa1.y);
        mma884(v1, aa[2], aa[3], cur.xa1.z, cur.xa1.w);
        mma884(v1, aa[4], aa[5], cur.xb1.x, cur.xb1.y);
        mma884(v1, aa[6], aa[7], cur.xb1.z, cur.xb1.w);
        mma884(v1 + 8, ab[0], ab[1], cur.xa1.x, cur.xa1.y);
        mma884(v1 + 8, ab[2], ab[3], cur.xa1.z, cur.xa1.w);
        mma884(v1 + 8, ab[4], ab[5], cur.xb1.x, cur.xb1.y);
        mma884(v1 + 8, ab[6], ab[7], cur.xb1.z, cur.xb1.w);
      }
    }
  }
  const float f = FMT == FMT_NVFP4 ? __ldg(a.factors + e) : 1.f;
  auto write_group = [&](float (&vg)[16], int base_off, int gn) {
#pragma unroll
    for (int sh = 4; sh <= 8; sh <<= 1)
#pragma unroll
      for (int m = 0; m < 16; ++m) vg[m] += __shfl_xor_sync(0xffffffffu, vg[m], sh);
    if (qp == 0) {
#pragma unroll
      for (int m = 0; m < 16; ++m) part[split][rr][m] = vg[m];
    }
    __syncthreads();
    if (!split && !qp) {
      // c[i]: row (j & 1) + 2 ((i >> 1) & 1) + 4 hi of the 8 against pair member (i & 1) + 2 (j >> 1) + 4 (i >> 2)
#pragma unroll 1
      for (int i = 0; i < 8; ++i) {
        const int mem = (i & 1) + 2 * (j >> 1) + 4 * (i >> 2);
        if (mem >= gn) continue;
        float tA = part[0][rr][i], tB = part[0][rr][8 + i];
#pragma unroll
        for (int s = 1; s < SPLIT; ++s) tA += part[s][rr][i], tB += part[s][rr][8 + i];
        tA *= f, tB *= f;
        const int rin = (j & 1) + 2 * ((i >> 1) & 1) + 4 * hi;
        if constexpr (UP) {
          const float gt = round_bf16(tA), u = round_bf16(tB);
          const float sig = 1.f / (1.f + expf(-gt));
          const float act = __fmul_rn(__fmul_rn(softcap_d(gt, 4.f), sig), softcap_d(u, 25.f));
          a.out[int64_t(pb + base_off + mem) * kInter + 8 * int(blockIdx.x) + rin] = f2bf_bits(act);
        } else {
          const int dst = a.w.pair_dst[pb + base_off + mem];
          a.out[int64_t(dst) * kHidden + 16 * int(blockIdx.x) + rin] = f2bf_bits(tA);
          a.out[int64_t(dst) * kHidden + 16 * int(blockIdx.x) + 8 + rin] = f2bf_bits(tB);
        }
      }
    }
  };
  write_group(v, 0, min(n, 8));
  if constexpr (MAX_TILE > 8) {
    if (has_hi) {
      __syncthreads();
      write_group(v1, 8, n - 8);
    }
  }
}


// ---- decode routing in one launch (B <= 16): router logits (FP32 weights, FP32 sums), then the last CTA to finish
// selects (topk_kernel's rule: sigmoid scores, 8 arg-max rounds over s + bias, lowest expert on ties, wts = 8 s / (sum
// + 1e-20)) and builds the tiles and pair lists (route_kernel's). Replaces gemm + memset + topk + route.
constexpr int kRouteCtas = kExperts, kRouteWarps = 4;  // a CTA an expert: every load in flight at once
template <int TMAX>
__global__ void __launch_bounds__(kRouteWarps * 32) route_decode_kernel(const uint16_t* __restrict__ x, int T,
                                                                        const float* __restrict__ w,
                                                                        const float* __restrict__ bias, int TR,
                                                                        int* __restrict__ idx, float* __restrict__ wts,
                                                                        float* logits, unsigned* done, MoeWork mw) {
  __shared__ int counts[kExperts], offs[kExperts], tile0[kExperts], cursor[kExperts];
  __shared__ float red[kRouteWarps][TMAX];
  __shared__ bool last;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, e = blockIdx.x;
  {
    constexpr int N4 = kHidden / 4, NT = kRouteWarps * 32, IT = (N4 + NT - 1) / NT;  // 576 float4s, 5 a thread
    const float4* wr = reinterpret_cast<const float4*>(w + int64_t(e) * kHidden);
    float4 wv[IT];
#pragma unroll
    for (int j = 0; j < IT; ++j) {
      const int i = int(threadIdx.x) + NT * j;
      wv[j] = i < N4 ? __ldg(wr + i) : make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float acc[TMAX];
#pragma unroll
    for (int t = 0; t < TMAX; ++t) acc[t] = 0.f;
#pragma unroll
    for (int t = 0; t < TMAX; ++t) {
      if (t >= T) break;
      const uint2* xt = reinterpret_cast<const uint2*>(x + int64_t(t) * kHidden);
#pragma unroll
      for (int j = 0; j < IT; ++j) {
        const int i = int(threadIdx.x) + NT * j;
        if (i >= N4) break;
        const uint2 xv = __ldg(xt + i);
        const float2 a2 = __half22float2(*reinterpret_cast<const __half2*>(&xv.x));
        const float2 b2 = __half22float2(*reinterpret_cast<const __half2*>(&xv.y));
        acc[t] += a2.x * wv[j].x + a2.y * wv[j].y + b2.x * wv[j].z + b2.y * wv[j].w;
      }
    }
#pragma unroll
    for (int t = 0; t < TMAX; ++t) {
      if (t >= T) break;
      float v = acc[t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[warp][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < unsigned(T)) {
      float v = red[0][threadIdx.x];
#pragma unroll
      for (int k = 1; k < kRouteWarps; ++k) v += red[k][threadIdx.x];
      logits[threadIdx.x * kExperts + e] = v;
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned ticket = atomicAdd(done, 1u);
    last = ticket == gridDim.x - 1;
    if (last) *done = 0u;  // (the next replay)
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  for (int i = threadIdx.x; i < kExperts; i += blockDim.x) counts[i] = 0, cursor[i] = 0;
  __syncthreads();
  for (int t = warp; t < T; t += kRouteWarps) {
    float sc[4], key[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int ee = lane + 32 * i;
      sc[i] = 1.f / (1.f + expf(-__ldcg(logits + t * kExperts + ee)));
      key[i] = sc[i] + bias[ee];
    }
    int sel_e[kTopK];
    float sel_s[kTopK];
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      float best = -INFINITY;
      int be = kExperts;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (key[i] > best) best = key[i], be = lane + 32 * i;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oe = __shfl_xor_sync(0xffffffffu, be, o);
        if (ob > best || (ob == best && oe < be)) best = ob, be = oe;
      }
      sel_e[k] = be;
      float s_be = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (lane + 32 * i == be) s_be = sc[i], key[i] = -INFINITY;
      sel_s[k] = __shfl_sync(0xffffffffu, s_be, be & 31);
    }
    if (lane == 0) {
      float sum = sel_s[0];
#pragma unroll
      for (int k = 1; k < kTopK; ++k) sum += sel_s[k];
      const float den = sum + 1e-20f;
#pragma unroll
      for (int k = 0; k < kTopK; ++k) {
        idx[t * kTopK + k] = sel_e[k];
        wts[t * kTopK + k] = sel_s[k] / den * 8.f;
        atomicAdd(&counts[sel_e[k]], 1);
      }
    }
  }
  __syncthreads();
  {  // exclusive scans of the pair counts and tile counts, an expert a thread (kRouteWarps * 32 == kExperts)
    static_assert(kRouteWarps * 32 == kExperts, "an expert a thread");
    __shared__ int wsum_c[kRouteWarps], wsum_t[kRouteWarps];
    const int ee = threadIdx.x, c = counts[ee], nt = (c + TR - 1) / TR;
    int ic = c, it = nt;  // inclusive within the warp
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int pc = __shfl_up_sync(0xffffffffu, ic, o), pt = __shfl_up_sync(0xffffffffu, it, o);
      if (lane >= o) ic += pc, it += pt;
    }
    if (lane == 31) wsum_c[warp] = ic, wsum_t[warp] = it;
    __syncthreads();
    int bc = 0, bt = 0;
    for (int k = 0; k < warp; ++k) bc += wsum_c[k], bt += wsum_t[k];
    offs[ee] = bc + ic - c, tile0[ee] = bt + it - nt;
    if (ee == kExperts - 1) *mw.tiles_n = bt + it;
  }
  __syncthreads();
  for (int ee = threadIdx.x; ee < kExperts; ee += blockDim.x) {
    const int c = counts[ee];
    for (int i = 0; i * TR < c; ++i) mw.tiles[tile0[ee] + i] = make_int4(ee, offs[ee] + i * TR, min(TR, c - i * TR), 0);
  }
  __syncthreads();
  for (int p = threadIdx.x; p < T * kTopK; p += blockDim.x) {
    const int ee = idx[p];
    const int pos = offs[ee] + atomicAdd(&cursor[ee], 1);
    mw.pair_token[pos] = p / kTopK;
    mw.pair_dst[pos] = p;
  }
}

bool decode_enabled(int T, int fmt) {
  static const int mode = [] {
    const char* e = std::getenv("INGINE_MOE_DECODE");
    return e ? std::atoi(e) : 1;
  }();
  if (!mode || T < 1 || T > 16) return false;
  if (fmt != FMT_NVFP4 && fmt != FMT_GQ5 && fmt != FMT_GQI4 && fmt != FMT_GQ4) return false;
  static const bool volta = [] {
    int dev = 0, major = 0, minor = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    return major == 7 && minor == 0;
  }();
  return volta;
}

template <bool UP, int SPLIT>
void launch_dec(int fmt, const DecArgs& d, int T, cudaStream_t stream) {
  const int max_y = T * kTopK < kExperts ? T * kTopK : kExperts;
  const dim3 grid(UP ? (2 * kInter) / 16 : kHidden / 16, unsigned(max_y));
  if (T <= 8) {
    if (fmt == FMT_NVFP4) moe_decode_v70_kernel<FMT_NVFP4, UP, SPLIT, 8><<<grid, SPLIT * 32, 0, stream>>>(d);
    else if (fmt == FMT_GQ5) moe_decode_v70_kernel<FMT_GQ5, UP, SPLIT, 8><<<grid, SPLIT * 32, 0, stream>>>(d);
    else if (fmt == FMT_GQ4) moe_decode_v70_kernel<FMT_GQ4, UP, SPLIT, 8><<<grid, SPLIT * 32, 0, stream>>>(d);
    else moe_decode_v70_kernel<FMT_GQI4, UP, SPLIT, 8><<<grid, SPLIT * 32, 0, stream>>>(d);
  } else {
    if (fmt == FMT_NVFP4) moe_decode_v70_kernel<FMT_NVFP4, UP, SPLIT, 16><<<grid, SPLIT * 32, 0, stream>>>(d);
    else if (fmt == FMT_GQ5) moe_decode_v70_kernel<FMT_GQ5, UP, SPLIT, 16><<<grid, SPLIT * 32, 0, stream>>>(d);
    else if (fmt == FMT_GQ4) moe_decode_v70_kernel<FMT_GQ4, UP, SPLIT, 16><<<grid, SPLIT * 32, 0, stream>>>(d);
    else moe_decode_v70_kernel<FMT_GQI4, UP, SPLIT, 16><<<grid, SPLIT * 32, 0, stream>>>(d);
  }
}

}  // namespace

bool moe_route_decode(const ActT* x, int T, const float* router_f32, const float* bias, int* idx, float* wts,
                      float* logits_scratch, unsigned* done, const MoeWork& w, cudaStream_t stream) {
  static const bool on = [] {
    const char* e = std::getenv("INGINE_MOE_ROUTE_FUSED");
    return !(e && e[0] == '0');
  }();
  if (!on || T < 1 || T > 16 || !router_f32 || !done) return false;
  if (T <= 8) {
    route_decode_kernel<8><<<kRouteCtas, kRouteWarps * 32, 0, stream>>>(reinterpret_cast<const uint16_t*>(x), T,
                                                                        router_f32, bias, moe_tile_rows(T), idx, wts,
                                                                        logits_scratch, done, w);
  } else {
    route_decode_kernel<16><<<kRouteCtas, kRouteWarps * 32, 0, stream>>>(reinterpret_cast<const uint16_t*>(x), T,
                                                                         router_f32, bias, moe_tile_rows(T), idx, wts,
                                                                         logits_scratch, done, w);
  }
  return true;
}

bool moe_gate_up_decode(const MoeExperts& e, const ActT* x, int T, const MoeWork& w, ActT* act, cudaStream_t stream) {
  if (!decode_enabled(T, e.fmt)) return false;
  const DecArgs d{static_cast<const uint8_t*>(e.w13), static_cast<const uint8_t*>(e.w13_scale), e.w13_factor,
                  e.w13_ptrs, e.w13_scale_ptrs, e.active_mask, reinterpret_cast<const uint16_t*>(x), w,
                  reinterpret_cast<uint16_t*>(act)};
  launch_dec<true, 4>(e.fmt, d, T, stream);
  return true;
}

bool moe_down_decode(const MoeExperts& e, const ActT* act, int T, const MoeWork& w, ActT* down, cudaStream_t stream) {
  const int fmt = e.w2_fmt >= 0 ? e.w2_fmt : e.fmt;
  if (!decode_enabled(T, fmt)) return false;
  const DecArgs d{static_cast<const uint8_t*>(e.w2), static_cast<const uint8_t*>(e.w2_scale), e.w2_factor,
                  e.w2_ptrs, e.w2_scale_ptrs, e.active_mask, reinterpret_cast<const uint16_t*>(act), w,
                  reinterpret_cast<uint16_t*>(down)};
  launch_dec<false, 2>(fmt, d, T, stream);
  return true;
}

}  // namespace ingine::kernels::prefill

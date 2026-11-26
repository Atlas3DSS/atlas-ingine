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

constexpr int FMT_NVFP4 = 0, FMT_GQ5 = 6, FMT_GQI4 = 7;

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

template <int FMT, bool UP, int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32) moe_decode_v70_kernel(const DecArgs a) {
  constexpr int K = UP ? kHidden : kInter;
  constexpr int NE = UP ? 2 * kInter : kHidden;   // expert rows
  constexpr int CODE_ROW = K / 2;
  constexpr int SCALE_ROW = FMT == FMT_NVFP4 ? K / 16 : (FMT == FMT_GQ5 ? (K / 32) * 12 : (K / 32) * 4);
  constexpr int NK = K / 64;
  __shared__ float part[SPLIT][8][16];
  __shared__ float lut[16];
  if (blockIdx.y >= unsigned(*a.w.tiles_n)) return;
  const int4 tile = a.w.tiles[blockIdx.y];
  const int e = tile.x, pb = tile.y, n = min(tile.z, 8);
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

  auto load_row = [&](const uint8_t* c, const uint8_t* s, int k) {
    RowStep r;
    r.c = __ldg(reinterpret_cast<const uint2*>(c + 32 * k));
    if constexpr (FMT == FMT_NVFP4) {
      r.s0 = __ldg(reinterpret_cast<const uint32_t*>(s + 4 * k));  // the block's four sub-block scales
    } else if constexpr (FMT == FMT_GQ5) {
      const uint32_t* g = reinterpret_cast<const uint32_t*>(s + 12 * (2 * k + (qp >> 1)));
      r.s0 = __ldg(g), r.s1 = __ldg(g + 1), r.s2 = __ldg(g + 2);  // scale, min, fifth bits of the group
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
    } else {
      const float sc = __uint_as_float(r.s0);
      const float mn = FMT == FMT_GQ5 ? __uint_as_float(r.s1) : 0.f;
      const uint32_t hb = FMT == FMT_GQ5 ? (r.s2 >> ((qp & 1) * 16)) : 0u;  // fifth bits of the sub-block
#pragma unroll
      for (int b = 0; b < 8; ++b) {
        const uint32_t byte = ((b < 4 ? r.c.x : r.c.y) >> (8 * (b & 3))) & 0xffu;
        uint32_t lo = byte & 15u, hi4 = byte >> 4;
        float vl, vh;
        if constexpr (FMT == FMT_GQ5) {
          lo |= ((hb >> (2 * b)) & 1u) << 4;
          hi4 |= ((hb >> (2 * b + 1)) & 1u) << 4;
          vl = __fadd_rn(__fmul_rn(sc, float(lo)), mn);
          vh = __fadd_rn(__fmul_rn(sc, float(hi4)), mn);
        } else {
          vl = __fmul_rn(sc, lut[lo]);
          vh = __fmul_rn(sc, lut[hi4]);
        }
        ah[b] = f2h2(vl, vh);
      }
    }
  };

  float v[16];
#pragma unroll
  for (int m = 0; m < 16; ++m) v[m] = 0.f;
  struct Step {
    RowStep ra, rb;
    uint4 xa, xb;
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
  }
#pragma unroll
  for (int sh = 4; sh <= 8; sh <<= 1)
#pragma unroll
    for (int m = 0; m < 16; ++m) v[m] += __shfl_xor_sync(0xffffffffu, v[m], sh);
  if (qp == 0) {
#pragma unroll
    for (int m = 0; m < 16; ++m) part[split][rr][m] = v[m];
  }
  __syncthreads();
  if (split || qp) return;
  const float f = FMT == FMT_NVFP4 ? __ldg(a.factors + e) : 1.f;
  // c[i]: row (j & 1) + 2 ((i >> 1) & 1) + 4 hi of the 8 against pair member (i & 1) + 2 (j >> 1) + 4 (i >> 2)
#pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    const int mem = (i & 1) + 2 * (j >> 1) + 4 * (i >> 2);
    if (mem >= n) continue;
    float tA = part[0][rr][i], tB = part[0][rr][8 + i];
#pragma unroll
    for (int s = 1; s < SPLIT; ++s) tA += part[s][rr][i], tB += part[s][rr][8 + i];
    tA *= f, tB *= f;
    const int rin = (j & 1) + 2 * ((i >> 1) & 1) + 4 * hi;
    if constexpr (UP) {
      const float gt = round_bf16(tA), u = round_bf16(tB);
      const float sig = 1.f / (1.f + expf(-gt));
      const float act = __fmul_rn(__fmul_rn(softcap_d(gt, 4.f), sig), softcap_d(u, 25.f));
      a.out[int64_t(pb + mem) * kInter + 8 * int(blockIdx.x) + rin] = f2bf_bits(act);
    } else {
      const int dst = a.w.pair_dst[pb + mem];
      a.out[int64_t(dst) * kHidden + 16 * int(blockIdx.x) + rin] = f2bf_bits(tA);
      a.out[int64_t(dst) * kHidden + 16 * int(blockIdx.x) + 8 + rin] = f2bf_bits(tB);
    }
  }
}


bool decode_enabled(int T, int fmt) {
  static const int mode = [] {
    const char* e = std::getenv("INGINE_MOE_DECODE");
    return e ? std::atoi(e) : 1;
  }();
  if (!mode || T < 1 || T > 8) return false;
  if (fmt != FMT_NVFP4 && fmt != FMT_GQ5 && fmt != FMT_GQI4) return false;
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
  const dim3 grid(UP ? (2 * kInter) / 16 : kHidden / 16, unsigned(T * kTopK));
  if (fmt == FMT_NVFP4) moe_decode_v70_kernel<FMT_NVFP4, UP, SPLIT><<<grid, SPLIT * 32, 0, stream>>>(d);
  else if (fmt == FMT_GQ5) moe_decode_v70_kernel<FMT_GQ5, UP, SPLIT><<<grid, SPLIT * 32, 0, stream>>>(d);
  else moe_decode_v70_kernel<FMT_GQI4, UP, SPLIT><<<grid, SPLIT * 32, 0, stream>>>(d);
}

}  // namespace

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

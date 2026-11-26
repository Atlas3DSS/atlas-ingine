#include <atomic>
// Grouped expert GEMMs of the prefill (model._moe_grouped) over NVFP4 / MXFP8 experts decoded on the fly.
//
// Design. A CTA computes up to 128 activation rows x 128 weight rows, K in steps of 64, mma.m16n8k16 BF16 with
// FP32 accumulation. The 8 warps split the weight rows (16 each) and every warp covers all of the tile's
// activation rows, so each weight element is decoded exactly once per CTA, straight from the staged checkpoint
// codes into the MMA B fragments in registers (no BF16 weight tile in shared memory):
//   NVFP4: bf16((e2m1 * e4m3 block scale) * factor)      (model.decode_experts)
//   MXFP8: bf16(e4m3 * 2^(e8m0 - 127))                    (model.decode_experts)
// Activation rows (gathered by pair for the experts) and the weight codes stream in through 16-byte cp.async:
// activations in 3 stages of 64 elements, the codes in 2 slots of 128-byte (64-byte for the NVFP4 down
// projection) row pieces, so DRAM reads whole lines and small chunks, which are bound by the weight bytes, keep
// enough loads in flight. One barrier per step. Code rows are padded in shared memory so the 8 rows of a fragment
// load hit distinct banks. m16 row blocks past the tile's row count skip their MMAs (zero-filled copies).
// The per-element products and the k order of the FP32 sums are those of the BF16 MMA over decoded rows.
// Epilogues: gate/up (w13 rows interleaved, 2i gate and 2i + 1 up, so each thread holds gate and up of the same
// i): act = bf16(softcap(bf16 g, 4) * sigmoid(bf16 g) * softcap(bf16 u, 25)); down: bf16 per pair placed at
// t * 8 + slot (moe_combine sums the slots in order: deterministic).
#include "core/device.h"
#include "kernels/common/exl3.cuh"
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <cmath>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

enum class Mode { Up, Down };
constexpr int FMT_NVFP4 = 0, FMT_MXFP8 = 1, FMT_ATLAS2 = 2, FMT_ATLAS3 = 3, FMT_EXL3_2 = 4, FMT_EXL3_3 = 5;
// GGUF experts (model/weights_gguf.cpp): codes as NVFP4's nibbles; per 32-value group Q5L 12 B (FP32 scale, FP32
// min, the 32 fifth bits: value = scale * code + min), IQ4 4 B (FP32 scale: value = scale * iq4nl[code]).
constexpr int FMT_GQ5 = 6, FMT_GQI4 = 7;
constexpr bool is_gguf_fmt(int f) { return f == FMT_GQ5 || f == FMT_GQI4; }

constexpr int TN = 128, BK = 64;
constexpr int AS = INGINE_ACT_FP16 ? 2 : 3;  // activation stages (2 fits Volta 96 KiB shared memory)
constexpr int NT = 256;
constexpr int KCH = BK / 8;  // 16-byte chunks per activation tile row

__constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
__constant__ float kAtlas2[4] = {-1.5f, -0.453125f, 0.453125f, 1.5f};
__constant__ float kAtlas3[8] = {-2.0f, -1.25f, -0.703125f, -0.2265625f, 0.2265625f, 0.703125f, 1.25f, 2.0f};
__constant__ float kIq4nl[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                                 1.f,    13.f,   25.f,  38.f,  53.f,  69.f,  89.f,  113.f};

template <int FMT, int K, int TM, bool LOW_SMEM = false>
struct Cfg {
  static constexpr int CODE_ROW = (FMT == FMT_NVFP4 || is_gguf_fmt(FMT)) ? K / 2 : (FMT == FMT_MXFP8 ? K : (FMT == FMT_ATLAS2 ? K / 4 : (K * 3) / 8));
  static constexpr int SCALE_ROW = FMT == FMT_GQ5 ? (K / 32) * 12 : (FMT == FMT_GQI4 ? (K / 32) * 4 : (FMT == FMT_MXFP8 ? K / 32 : K / 16));
  // weight chunk: 128 elements for NVFP4/ATLAS2/ATLAS3 (64/32/48 code bytes per row, all multiples of 16), 64 for MXFP8
  static constexpr int WK = (FMT == FMT_MXFP8 || K % 128 != 0) ? 64 : 128;  // (K = 576: expert TP down)
  static constexpr int R = WK / BK;  // activation steps per weight chunk
  static constexpr int KC = K / WK;  // weight chunks
  static constexpr int AS_STG = LOW_SMEM ? 2 : AS;
  // On Volta sm_70 (INGINE_ACT_FP16 && !LOW_SMEM), each warp owns its 16 weight rows exclusively
  // and cp.async does not exist. Streaming NVFP4/MXFP8 weights + scales directly via __ldg into
  // registers eliminates w_smem (20 KiB) and sc_smem (18 KiB), dropping TM=32 SMEM from 88 KiB -> 8.2 KiB!
  static constexpr bool VOLTA_DIRECT = INGINE_ACT_FP16 && !LOW_SMEM && (FMT == FMT_NVFP4 || FMT == FMT_MXFP8 || is_gguf_fmt(FMT));
  static constexpr bool VOLTA_COMPACT = INGINE_ACT_FP16 && !LOW_SMEM;
  static constexpr bool STAGE_W = !VOLTA_DIRECT;
  static constexpr bool STAGE_SC = !VOLTA_COMPACT && !is_gguf_fmt(FMT);
  static constexpr bool W64 = WK == 64;  // (MXFP8, or a K that is not a multiple of 128: stage as MXFP8 does)
  static constexpr int WS = (LOW_SMEM || VOLTA_COMPACT) ? (W64 ? 3 : 2)
                                     : (TM <= 32 ? (W64 ? 7 : 6)
                                                 : (W64 ? 4 : 3));
  static_assert(R * (WS - 1) >= AS_STG);
  static constexpr int WCODE = (FMT == FMT_NVFP4 || is_gguf_fmt(FMT)) ? WK / 2 : (FMT == FMT_MXFP8 ? WK : (FMT == FMT_ATLAS2 ? WK / 4 : (WK * 3) / 8));
  static constexpr int WSTRIDE = WCODE + 16;                    // padded: fragment loads on distinct banks
  static constexpr int W_BYTES = STAGE_W ? (TN * WSTRIDE) : 0;
  static constexpr int A_BYTES = TM * BK * 2;
  static constexpr int SC_BYTES = STAGE_SC ? (TN * SCALE_ROW) : 0;
  static constexpr int OFF_W = AS_STG * A_BYTES, OFF_SC = OFF_W + (STAGE_W ? WS * W_BYTES : 0);
  static constexpr int OFF_LUT = (OFF_SC + SC_BYTES + 15) / 16 * 16;
  static constexpr int SMEM = OFF_LUT + 64;
  static_assert(K % WK == 0 && SMEM <= (LOW_SMEM ? 49152 : (INGINE_ACT_FP16 ? 98304 : 101376)));
};

struct GemmArgs {
  const uint8_t* codes = nullptr;   // NVFP4 / MXFP8 expert codes [128, N, ..]
  const uint8_t* scales = nullptr;  // block scales
  const float* factors = nullptr;   // NVFP4 per-expert factor
  const uint8_t* const* code_ptrs = nullptr;   // optional [128] per-expert code base pointers
  const uint8_t* const* scale_ptrs = nullptr;  // optional [128] per-expert scale base pointers
  const uint8_t* active_mask = nullptr;        // optional [128] mask (skip expert when 0)
  const uint16_t* a = nullptr;      // activations (bf16)
  MoeWork w;                        // tiles and pair lists
  uint16_t* out = nullptr;
};

__device__ __forceinline__ void cp_async4(void* smem_dst, const void* gmem_src) {
#if __CUDA_ARCH__ < 800
  copy::sync<4>(smem_dst, gmem_src);
#else
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr(smem_dst)), "l"(gmem_src));
#endif
}

__device__ __forceinline__ float softcap(float x, float cap) { return tanhf(x * (1.f / cap)) * cap; }

// One activation step of MMAs for the first MT 16-row blocks: per k16 group, the A fragments by ldmatrix, then the
// MMAs, then `after(kk)` (the next step's B decoding, which runs under the issued MMAs).
template <int MT, int TM, class After>
__device__ __forceinline__ void mma_step(float (&acc)[TM / 16][2][4], const uint16_t* A,
                                         const unsigned (&b)[BK / 16][2][2], int lane, After&& after) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
  // Pascal sm_60/sm_61 fast path: direct SMEM uint4 A reads (0 shuffles) + hoisted B shuffles and FP32
  // unpacks across MT + exact FP32 fmaf accumulation.
  const int g = lane >> 2, c = lane & 3;
#pragma unroll
  for (int kk = 0; kk < BK / 16; ++kk) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int bl0 = 4 * (2 * c), bl1 = 4 * (2 * c + 1);
      float2 b0[8], b1[8];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        b0[p] = act::unpack2(__shfl_sync(0xffffffffu, b[kk][nt][0], bl0 + p));
        b0[4 + p] = act::unpack2(__shfl_sync(0xffffffffu, b[kk][nt][1], bl0 + p));
        b1[p] = act::unpack2(__shfl_sync(0xffffffffu, b[kk][nt][0], bl1 + p));
        b1[4 + p] = act::unpack2(__shfl_sync(0xffffffffu, b[kk][nt][1], bl1 + p));
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const uint4 alo0 = *reinterpret_cast<const uint4*>(A + swz<KCH>(mt * 16 + g, kk * 2));
        const uint4 alo1 = *reinterpret_cast<const uint4*>(A + swz<KCH>(mt * 16 + g, kk * 2 + 1));
        const uint4 ahi0 = *reinterpret_cast<const uint4*>(A + swz<KCH>(mt * 16 + g + 8, kk * 2));
        const uint4 ahi1 = *reinterpret_cast<const uint4*>(A + swz<KCH>(mt * 16 + g + 8, kk * 2 + 1));
        const uint32_t al[8] = {alo0.x, alo0.y, alo0.z, alo0.w, alo1.x, alo1.y, alo1.z, alo1.w};
        const uint32_t ah[8] = {ahi0.x, ahi0.y, ahi0.z, ahi0.w, ahi1.x, ahi1.y, ahi1.z, ahi1.w};
#pragma unroll
        for (int p = 0; p < 8; ++p) {
          const float2 fal = act::unpack2(al[p]);
          const float2 fah = act::unpack2(ah[p]);
          acc[mt][nt][0] = fmaf(fal.y, b0[p].y, fmaf(fal.x, b0[p].x, acc[mt][nt][0]));
          acc[mt][nt][1] = fmaf(fal.y, b1[p].y, fmaf(fal.x, b1[p].x, acc[mt][nt][1]));
          acc[mt][nt][2] = fmaf(fah.y, b0[p].y, fmaf(fah.x, b0[p].x, acc[mt][nt][2]));
          acc[mt][nt][3] = fmaf(fah.y, b1[p].y, fmaf(fah.x, b1[p].x, acc[mt][nt][3]));
        }
      }
    }
    after(kk);
  }
#else
#pragma unroll
  for (int kk = 0; kk < BK / 16; ++kk) {
    unsigned a[MT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
      ldmatrix_x4(a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                  smem_addr(A + swz<KCH>(mt * 16 + (lane & 15), kk * 2 + (lane >> 4))));
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      mma_bf16(acc[mt][0], a[mt], b[kk][0][0], b[kk][0][1]);
      mma_bf16(acc[mt][1], a[mt], b[kk][1][0], b[kk][1][1]);
    }
    after(kk);
  }
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
// Native Volta sm_70 HMMA.884 step: direct SMEM A-loads + direct Volta B-frags -> ZERO warp shuffles in K-loop!
template <int MT, int TM, class DecodeB>
__device__ __forceinline__ void mma_step_sm70(float (&acc_v)[TM / 16][2][8], const uint16_t* A, int lane,
                                              DecodeB&& decode_b) {
  const int qp = (lane >> 2) & 3;
  const int idx = (lane & 3) | ((lane & 16) >> 2);
  const int row_a = (qp >= 2 ? 8 : 0) + idx;
  const int k_half = (qp & 1) * 4;
#pragma unroll
  for (int kk = 0; kk < BK / 16; ++kk) {
    unsigned b[2][4];
    decode_b(kk, b);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const uint2 a01 = mma::ld_smem_u64(smem_addr(A + swz<KCH>(mt * 16 + row_a, kk * 2) + k_half));
      const uint2 a23 = mma::ld_smem_u64(smem_addr(A + swz<KCH>(mt * 16 + row_a, kk * 2 + 1) + k_half));
      mma::sm70_mma_f16_direct(acc_v[mt][0], a01.x, a01.y, a23.x, a23.y, b[0][0], b[0][1], b[0][2], b[0][3]);
      mma::sm70_mma_f16_direct(acc_v[mt][1], a01.x, a01.y, a23.x, a23.y, b[1][0], b[1][1], b[1][2], b[1][3]);
    }
  }
}
#endif

// Up: A = x rows of the pairs' tokens (K = H), W = w13 (N = 2I), out act [pairs, I].
// Down: A = act rows (K = I), W = w2 (N = H), out down [T * 8, H] at pair_dst.
template <int FMT, Mode MODE, int K, int TM, bool LOW_SMEM = false>
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
__global__ void __launch_bounds__(NT, (TM <= 32 && !LOW_SMEM) ? 2 : 1) wgemm_kernel(const GemmArgs g) {
#else
__global__ void __launch_bounds__(NT, 1) wgemm_kernel(const GemmArgs g) {
#endif
  constexpr int KS = K / BK;
  constexpr int NE = MODE == Mode::Up ? 2 * kInter : kHidden;  // expert weight rows
  using C = Cfg<FMT, K, TM, LOW_SMEM>;
  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* a_smem = smem;
  unsigned char* w_smem = smem + C::OFF_W;
  uint8_t* sc_smem = smem + C::OFF_SC;
  float* lut = reinterpret_cast<float*>(smem + C::OFF_LUT);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // ---- tile: n pair rows (from pb) of expert e x weight rows n0..n0+127
  constexpr int NTILES_N = NE / TN;
  const int tile_idx = blockIdx.x / NTILES_N;
  if (tile_idx >= *g.w.tiles_n) return;
  const int4 tile = g.w.tiles[tile_idx];
  const int e = tile.x, pb = tile.y, n = tile.z;
  if (g.active_mask && !g.active_mask[e]) return;
  const int n0 = (blockIdx.x % NTILES_N) * TN;
  const uint8_t* code_tile =
      g.code_ptrs ? (g.code_ptrs[e] + int64_t(n0) * C::CODE_ROW) : (g.codes + (int64_t(e) * NE + n0) * C::CODE_ROW);
  const uint8_t* scale_tile =
      g.scale_ptrs ? (g.scale_ptrs[e] + int64_t(n0) * C::SCALE_ROW) : (g.scales + (int64_t(e) * NE + n0) * C::SCALE_ROW);
  if constexpr (FMT == FMT_NVFP4) {
    if (tid < 16) lut[tid] = kE2M1[tid];
  } else if constexpr (FMT == FMT_ATLAS2) {
    if (tid < 4) lut[tid] = kAtlas2[tid];
  } else if constexpr (FMT == FMT_ATLAS3) {
    if (tid < 8) lut[tid] = kAtlas3[tid];
  } else if constexpr (FMT == FMT_GQI4) {
    if (tid < 16) lut[tid] = kIq4nl[tid];
  }

  // activation rows of this thread's cp.async chunks: rows tid / 8 + 32 i, chunk tid % 8
  const uint16_t* arow[TM / 32];
#pragma unroll
  for (int i = 0; i < TM / 32; ++i) {
    const int r = (tid >> 3) + 32 * i;
    const int rr = r < n ? r : 0;
    if constexpr (MODE == Mode::Up) arow[i] = g.a + size_t(g.w.pair_token[pb + rr]) * K;
    else arow[i] = g.a + size_t(pb + rr) * K;
  }

  auto load_a = [&](int ks) {
    uint16_t* A = reinterpret_cast<uint16_t*>(a_smem + (ks % C::AS_STG) * C::A_BYTES);
#pragma unroll
    for (int i = 0; i < TM / 32; ++i) {
      const int r = (tid >> 3) + 32 * i, c = tid & 7;
      cp_async16(A + swz<KCH>(r, c), arow[i] + ks * BK + c * 8, r < n ? 16 : 0);
    }
  };
  auto load_w = [&](int chunk) {
    unsigned char* ws = w_smem + (chunk % C::WS) * C::W_BYTES;
    constexpr int CPR = C::WCODE / 16;  // 16-byte pieces per row
    constexpr int TOTAL = TN * CPR;
#pragma unroll
    for (int id = tid; id < TOTAL; id += NT) {
      const int row = id / CPR, ch = id % CPR;
      cp_async16(ws + row * C::WSTRIDE + ch * 16, code_tile + int64_t(row) * C::CODE_ROW + chunk * C::WCODE + ch * 16);
    }
  };
  // group j: activation step j (plus whatever weight chunk was issued since the previous commit)
  auto issue = [&](int j) {
    if (j < KS) load_a(j);
    cp_commit();
  };

  // the tile's block scales, with the first activation group
  if constexpr (C::STAGE_SC) {
    for (int id = tid; id < C::SC_BYTES / 4; id += NT)
      cp_async4(sc_smem + id * 4, scale_tile + (id * 4 / C::SCALE_ROW) * int64_t(C::SCALE_ROW) + (id * 4) % C::SCALE_ROW);
  }
  if constexpr (C::STAGE_W) {
#pragma unroll
    for (int j = 0; j < C::WS - 1; ++j)
      if (j < C::KC) load_w(j);
  }
#pragma unroll
  for (int j = 0; j < C::AS_STG - 1; ++j) issue(j);

  const int c = lane & 3;
  const float factor = FMT != FMT_MXFP8 ? __ldg(g.factors + e) : 1.f;
  const int mact = min((n + 15) / 16, TM / 16);  // active m16 row blocks (same for every warp)
  float acc[TM / 16][2][4];
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA))
#pragma unroll
  for (int a = 0; a < TM / 16; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;
#endif

  // B fragments of step ks (k16 group kk) decoded from the staged codes of its weight chunk
  auto scales_of = [&](int ks, float (&sc)[2][4]) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int row = warp * 16 + nt * 8 + (lane >> 2);
      if constexpr (FMT != FMT_MXFP8) {
        const uint32_t s4 = *reinterpret_cast<const uint32_t*>(sc_smem + row * C::SCALE_ROW + ks * 4);
        const float2 lo = e4m3x2_to_float2(uint16_t(s4 & 0xffff)), hi = e4m3x2_to_float2(uint16_t(s4 >> 16));
        sc[nt][0] = lo.x, sc[nt][1] = lo.y, sc[nt][2] = hi.x, sc[nt][3] = hi.y;
      } else {
        const uint8_t* ms = sc_smem + row * C::SCALE_ROW + ks * 2;
        sc[nt][0] = sc[nt][1] = exp2f(float(ms[0]) - 127.f);
        sc[nt][2] = sc[nt][3] = exp2f(float(ms[1]) - 127.f);
      }
    }
  };
  auto decode = [&](int ks, int kk, const float (&sc)[2][4], unsigned (&b)[BK / 16][2][2]) {
    const unsigned char* ws = w_smem + ((ks / C::R) % C::WS) * C::W_BYTES;
    const int kc = (ks % C::R) * BK + kk * 16;  // first element of the k16 group inside the chunk
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int row = warp * 16 + nt * 8 + (lane >> 2);
      const float s = sc[nt][kk];
      if constexpr (FMT == FMT_NVFP4) {
        // elements 2c, 2c+1 (byte c) and 8 + 2c, +1 (byte 4 + c) of the k16 group
        const uint2 wd = *reinterpret_cast<const uint2*>(ws + row * C::WSTRIDE + kc / 2);
        const uint32_t b0 = (wd.x >> (8 * c)) & 0xffu, b1 = (wd.y >> (8 * c)) & 0xffu;
        b[kk][nt][0] = pack_bf16x2((lut[b0 & 0xf] * s) * factor, (lut[b0 >> 4] * s) * factor);
        b[kk][nt][1] = pack_bf16x2((lut[b1 & 0xf] * s) * factor, (lut[b1 >> 4] * s) * factor);
      } else if constexpr (FMT == FMT_MXFP8) {
        const uint4 q = *reinterpret_cast<const uint4*>(ws + row * C::WSTRIDE + kc);
        const uint32_t w0 = (c >> 1) ? q.y : q.x, w1 = (c >> 1) ? q.w : q.z;
        const float2 f0 = e4m3x2_to_float2(uint16_t(w0 >> (16 * (c & 1))));
        const float2 f1 = e4m3x2_to_float2(uint16_t(w1 >> (16 * (c & 1))));
        b[kk][nt][0] = pack_bf16x2(f0.x * s, f0.y * s);
        b[kk][nt][1] = pack_bf16x2(f1.x * s, f1.y * s);
      } else if constexpr (FMT == FMT_ATLAS2) {
        const uint32_t wd = *reinterpret_cast<const uint32_t*>(ws + row * C::WSTRIDE + kc / 4);
        b[kk][nt][0] = pack_bf16x2((lut[(wd >> (4 * c)) & 3u] * s) * factor,
                                   (lut[(wd >> (4 * c + 2)) & 3u] * s) * factor);
        b[kk][nt][1] = pack_bf16x2((lut[(wd >> (16 + 4 * c)) & 3u] * s) * factor,
                                   (lut[(wd >> (16 + 4 * c + 2)) & 3u] * s) * factor);
      } else {
        const unsigned char* blk = ws + row * C::WSTRIDE + (kc / 32) * 12;
        const int half = (kc >> 4) & 1;
        const uint32_t lo = *reinterpret_cast<const uint32_t*>(blk + half * 4);
        const uint16_t hi16 = *reinterpret_cast<const uint16_t*>(blk + 8 + half * 2);
        const uint32_t c0 = ((lo >> (4 * c)) & 3u) | (((hi16 >> (2 * c)) & 1u) << 2);
        const uint32_t c1 = ((lo >> (4 * c + 2)) & 3u) | (((hi16 >> (2 * c + 1)) & 1u) << 2);
        const uint32_t c2 = ((lo >> (16 + 4 * c)) & 3u) | (((hi16 >> (8 + 2 * c)) & 1u) << 2);
        const uint32_t c3 = ((lo >> (16 + 4 * c + 2)) & 3u) | (((hi16 >> (8 + 2 * c + 1)) & 1u) << 2);
        b[kk][nt][0] = pack_bf16x2((lut[c0] * s) * factor, (lut[c1] * s) * factor);
        b[kk][nt][1] = pack_bf16x2((lut[c2] * s) * factor, (lut[c3] * s) * factor);
      }
      asm volatile("" : "+r"(b[kk][nt][0]), "+r"(b[kk][nt][1]));
    }
  };

  // Software pipeline: step ks's MMAs run on fragments decoded during step ks - 1; the decoding of step ks + 1
  // is interleaved with them (its chunk has landed: R (WS - 1) >= AS, and the barrier published it).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
  // Native Volta sm_70 path: keep acc_v[TM/16][2][8] across entire K-loop, decode B directly into
  // Volta HMMA.884 layout, load A directly from SMEM -> ZERO shuffles inside K-loop, 1 reduction at end!
  float acc_v[TM / 16][2][8];
#pragma unroll
  for (int a = 0; a < TM / 16; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b)
#pragma unroll
      for (int v = 0; v < 8; ++v) acc_v[a][b][v] = 0.f;
  const int v_idx = (lane & 3) | ((lane & 16) >> 2);
  const int v_kh = ((lane >> 2) & 1) * 4;
  if constexpr (C::VOLTA_DIRECT && FMT == FMT_NVFP4) {
    // Direct __ldg register-pipelined Volta sm_70 NVFP4 path: ZERO w_smem, ZERO sc_smem, ZERO spills!
    const uint8_t* wrow[2] = {
        code_tile + int64_t(warp * 16 + v_idx) * C::CODE_ROW,
        code_tile + int64_t(warp * 16 + 8 + v_idx) * C::CODE_ROW,
    };
    const uint8_t* srow[2] = {
        scale_tile + int64_t(warp * 16 + v_idx) * C::SCALE_ROW,
        scale_tile + int64_t(warp * 16 + 8 + v_idx) * C::SCALE_ROW,
    };
    uint4 w0_cur[2], w1_cur[2];
    uint32_t s4_cur[2];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      w0_cur[nt] = __ldg(reinterpret_cast<const uint4*>(wrow[nt]));
      w1_cur[nt] = __ldg(reinterpret_cast<const uint4*>(wrow[nt] + 16));
      s4_cur[nt] = __ldg(reinterpret_cast<const uint32_t*>(srow[nt]));
    }
#pragma unroll 1
    for (int ks = 0; ks < KS; ++ks) {
      __syncthreads();
      issue(ks + C::AS_STG - 1);
      const uint16_t* A = reinterpret_cast<const uint16_t*>(a_smem + (ks % C::AS_STG) * C::A_BYTES);
      uint4 w0_nxt[2], w1_nxt[2];
      uint32_t s4_nxt[2];
      if (ks + 1 < KS) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          w0_nxt[nt] = __ldg(reinterpret_cast<const uint4*>(wrow[nt] + (ks + 1) * 32));
          w1_nxt[nt] = __ldg(reinterpret_cast<const uint4*>(wrow[nt] + (ks + 1) * 32 + 16));
          s4_nxt[nt] = __ldg(reinterpret_cast<const uint32_t*>(srow[nt] + (ks + 1) * 4));
        }
      }
      __half2 sfh_v[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const float2 lo = e4m3x2_to_float2(uint16_t(s4_cur[nt] & 0xffff));
        const float2 hi = e4m3x2_to_float2(uint16_t(s4_cur[nt] >> 16));
        // (the E4M3 scale alone: code x scale is exact in FP16; scale x factor underflowed FP16 for small block scales
        //  - the expert factor multiplies the FP32 sums in the epilogue instead)
        sfh_v[nt][0] = __float2half2_rn(lo.x);
        sfh_v[nt][1] = __float2half2_rn(lo.y);
        sfh_v[nt][2] = __float2half2_rn(hi.x);
        sfh_v[nt][3] = __float2half2_rn(hi.y);
      }
      auto decode_v = [&](int kk, unsigned (&b)[2][4]) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const uint32_t wx = (kk == 0) ? w0_cur[nt].x : (kk == 1) ? w0_cur[nt].z : (kk == 2) ? w1_cur[nt].x : w1_cur[nt].z;
          const uint32_t wy = (kk == 0) ? w0_cur[nt].y : (kk == 1) ? w0_cur[nt].w : (kk == 2) ? w1_cur[nt].y : w1_cur[nt].w;
          const uint32_t w4 = ((wx >> (v_kh * 4)) & 0xffffu) | (((wy >> (v_kh * 4)) & 0xffffu) << 16);
          const __half2 sfh = sfh_v[nt][kk];
#pragma unroll
          for (int bi = 0; bi < 4; ++bi) {
            const uint32_t byte = (w4 >> (8 * bi)) & 0xffu;
            const uint32_t sel = ((byte & 7u) << 4) | (((byte >> 4) & 7u) << 12);
            const uint32_t hbits =
                __byte_perm(0x3E3C3800u, 0x46444240u, sel) | ((byte & 0x08u) << 12) | ((byte & 0x80u) << 24);
            const __half2 scaled = __hmul2(*reinterpret_cast<const __half2*>(&hbits), sfh);
            b[nt][bi] = *reinterpret_cast<const unsigned*>(&scaled);
          }
        }
      };
      switch (mact) {
        case 1: if constexpr (1 <= TM / 16) mma_step_sm70<1, TM>(acc_v, A, lane, decode_v); break;
        case 2: if constexpr (2 <= TM / 16) mma_step_sm70<2, TM>(acc_v, A, lane, decode_v); break;
        case 3: if constexpr (3 <= TM / 16) mma_step_sm70<3, TM>(acc_v, A, lane, decode_v); break;
        case 4: if constexpr (4 <= TM / 16) mma_step_sm70<4, TM>(acc_v, A, lane, decode_v); break;
        case 5: if constexpr (5 <= TM / 16) mma_step_sm70<5, TM>(acc_v, A, lane, decode_v); break;
        case 6: if constexpr (6 <= TM / 16) mma_step_sm70<6, TM>(acc_v, A, lane, decode_v); break;
        case 7: if constexpr (7 <= TM / 16) mma_step_sm70<7, TM>(acc_v, A, lane, decode_v); break;
        default: mma_step_sm70<TM / 16, TM>(acc_v, A, lane, decode_v); break;
      }
      if (ks + 1 < KS) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          w0_cur[nt] = w0_nxt[nt];
          w1_cur[nt] = w1_nxt[nt];
          s4_cur[nt] = s4_nxt[nt];
        }
      }
    }
  } else if constexpr (C::VOLTA_DIRECT && is_gguf_fmt(FMT)) {
    // Direct __ldg Volta path for the GGUF expert formats: the NVFP4 path's code stream (32 B a row a step), the
    // step's two 32-value groups' scales (Q5L: scale, min, fifth bits each; IQ4: scale), each value formed in FP32
    // (scale * code + min, or scale * iq4nl[code]) and rounded once to FP16.
    constexpr int SB = FMT == FMT_GQ5 ? 24 : 8;  // scale bytes a step (two groups)
    const uint8_t* wrow[2] = {
        code_tile + int64_t(warp * 16 + v_idx) * C::CODE_ROW,
        code_tile + int64_t(warp * 16 + 8 + v_idx) * C::CODE_ROW,
    };
    const uint8_t* srow[2] = {
        scale_tile + int64_t(warp * 16 + v_idx) * C::SCALE_ROW,
        scale_tile + int64_t(warp * 16 + 8 + v_idx) * C::SCALE_ROW,
    };
    uint4 w0_cur[2], w1_cur[2];
    uint2 s_cur[2][3];
    auto load_step = [&](int ks, uint4 (&w0)[2], uint4 (&w1)[2], uint2 (&sv)[2][3]) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        w0[nt] = __ldg(reinterpret_cast<const uint4*>(wrow[nt] + ks * 32));
        w1[nt] = __ldg(reinterpret_cast<const uint4*>(wrow[nt] + ks * 32 + 16));
        const uint2* sp = reinterpret_cast<const uint2*>(srow[nt] + ks * SB);
        sv[nt][0] = __ldg(sp);
        if constexpr (FMT == FMT_GQ5) sv[nt][1] = __ldg(sp + 1), sv[nt][2] = __ldg(sp + 2);
      }
    };
    load_step(0, w0_cur, w1_cur, s_cur);
#pragma unroll 1
    for (int ks = 0; ks < KS; ++ks) {
      __syncthreads();
      issue(ks + C::AS_STG - 1);
      const uint16_t* A = reinterpret_cast<const uint16_t*>(a_smem + (ks % C::AS_STG) * C::A_BYTES);
      uint4 w0_nxt[2], w1_nxt[2];
      uint2 s_nxt[2][3];
      if (ks + 1 < KS) load_step(ks + 1, w0_nxt, w1_nxt, s_nxt);
      auto decode_v = [&](int kk, unsigned (&b)[2][4]) {
        const int grp = kk >> 1, base = (kk & 1) * 16 + v_kh;  // the group and the lane's first value in it
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const uint32_t wx = (kk == 0) ? w0_cur[nt].x : (kk == 1) ? w0_cur[nt].z : (kk == 2) ? w1_cur[nt].x : w1_cur[nt].z;
          const uint32_t wy = (kk == 0) ? w0_cur[nt].y : (kk == 1) ? w0_cur[nt].w : (kk == 2) ? w1_cur[nt].y : w1_cur[nt].w;
          const uint32_t ca = wx >> (v_kh * 4), cb = wy >> (v_kh * 4);  // values base + j, base + 8 + j
          float sc, mn = 0.f;
          uint32_t hb = 0;
          if constexpr (FMT == FMT_GQ5) {
            // step layout: s0 m0 h0 | s1 m1 h1 as three 8-byte words
            sc = __uint_as_float(grp ? s_cur[nt][1].y : s_cur[nt][0].x);
            mn = __uint_as_float(grp ? s_cur[nt][2].x : s_cur[nt][0].y);
            hb = grp ? s_cur[nt][2].y : s_cur[nt][1].x;
          } else {
            sc = __uint_as_float(grp ? s_cur[nt][0].y : s_cur[nt][0].x);
          }
          float v[8];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t qa = (ca >> (4 * j)) & 15u, qb = (cb >> (4 * j)) & 15u;
            if constexpr (FMT == FMT_GQ5) {
              const uint32_t a5 = qa | (((hb >> (base + j)) & 1u) << 4), b5 = qb | (((hb >> (base + 8 + j)) & 1u) << 4);
              v[j] = __fadd_rn(__fmul_rn(sc, float(a5)), mn);
              v[4 + j] = __fadd_rn(__fmul_rn(sc, float(b5)), mn);
            } else {
              v[j] = __fmul_rn(sc, lut[qa]);
              v[4 + j] = __fmul_rn(sc, lut[qb]);
            }
          }
#pragma unroll
          for (int bi = 0; bi < 4; ++bi) {
            const __half2 h = __floats2half2_rn(v[2 * bi], v[2 * bi + 1]);
            b[nt][bi] = *reinterpret_cast<const unsigned*>(&h);
          }
        }
      };
      switch (mact) {
        case 1: if constexpr (1 <= TM / 16) mma_step_sm70<1, TM>(acc_v, A, lane, decode_v); break;
        case 2: if constexpr (2 <= TM / 16) mma_step_sm70<2, TM>(acc_v, A, lane, decode_v); break;
        case 3: if constexpr (3 <= TM / 16) mma_step_sm70<3, TM>(acc_v, A, lane, decode_v); break;
        case 4: if constexpr (4 <= TM / 16) mma_step_sm70<4, TM>(acc_v, A, lane, decode_v); break;
        case 5: if constexpr (5 <= TM / 16) mma_step_sm70<5, TM>(acc_v, A, lane, decode_v); break;
        case 6: if constexpr (6 <= TM / 16) mma_step_sm70<6, TM>(acc_v, A, lane, decode_v); break;
        case 7: if constexpr (7 <= TM / 16) mma_step_sm70<7, TM>(acc_v, A, lane, decode_v); break;
        default: mma_step_sm70<TM / 16, TM>(acc_v, A, lane, decode_v); break;
      }
      if (ks + 1 < KS) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          w0_cur[nt] = w0_nxt[nt], w1_cur[nt] = w1_nxt[nt];
          s_cur[nt][0] = s_nxt[nt][0], s_cur[nt][1] = s_nxt[nt][1], s_cur[nt][2] = s_nxt[nt][2];
        }
      }
    }
  } else if constexpr (C::VOLTA_DIRECT && FMT == FMT_MXFP8) {
    // Direct __ldg Volta sm_70 MXFP8 path: ZERO w_smem, ZERO sc_smem, native FP16x2 __hmul2!
    const uint8_t* wrow[2] = {
        code_tile + int64_t(warp * 16 + v_idx) * C::CODE_ROW,
        code_tile + int64_t(warp * 16 + 8 + v_idx) * C::CODE_ROW,
    };
    const uint8_t* srow[2] = {
        scale_tile + int64_t(warp * 16 + v_idx) * C::SCALE_ROW,
        scale_tile + int64_t(warp * 16 + 8 + v_idx) * C::SCALE_ROW,
    };
#pragma unroll 1
    for (int ks = 0; ks < KS; ++ks) {
      __syncthreads();
      issue(ks + C::AS_STG - 1);
      const uint16_t* A = reinterpret_cast<const uint16_t*>(a_smem + (ks % C::AS_STG) * C::A_BYTES);
      __half2 sh_v[2][2];
      uint4 qbuf[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const uint8_t s0 = __ldg(srow[nt] + ks * 2);
        const uint8_t s1 = __ldg(srow[nt] + ks * 2 + 1);
        sh_v[nt][0] = __float2half2_rn(s0 ? __uint_as_float(uint32_t(s0) << 23) : __uint_as_float(0x00400000u));
        sh_v[nt][1] = __float2half2_rn(s1 ? __uint_as_float(uint32_t(s1) << 23) : __uint_as_float(0x00400000u));
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          qbuf[nt][kk] = __ldg(reinterpret_cast<const uint4*>(wrow[nt] + ks * 64 + kk * 16));
        }
      }
      auto decode_v = [&](int kk, unsigned (&b)[2][4]) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const uint4 q = qbuf[nt][kk];
          const uint32_t wm = v_kh ? q.y : q.x;
          const uint32_t wn = v_kh ? q.w : q.z;
          const __half2 sh = sh_v[nt][kk >> 1];
          const uint32_t h0 = e4m3x2_to_half2(uint16_t(wm & 0xffff));
          const uint32_t h1 = e4m3x2_to_half2(uint16_t(wm >> 16));
          const uint32_t h2 = e4m3x2_to_half2(uint16_t(wn & 0xffff));
          const uint32_t h3 = e4m3x2_to_half2(uint16_t(wn >> 16));
          const __half2 r0 = __hmul2(*reinterpret_cast<const __half2*>(&h0), sh);
          const __half2 r1 = __hmul2(*reinterpret_cast<const __half2*>(&h1), sh);
          const __half2 r2 = __hmul2(*reinterpret_cast<const __half2*>(&h2), sh);
          const __half2 r3 = __hmul2(*reinterpret_cast<const __half2*>(&h3), sh);
          b[nt][0] = *reinterpret_cast<const unsigned*>(&r0);
          b[nt][1] = *reinterpret_cast<const unsigned*>(&r1);
          b[nt][2] = *reinterpret_cast<const unsigned*>(&r2);
          b[nt][3] = *reinterpret_cast<const unsigned*>(&r3);
        }
      };
      switch (mact) {
        case 1: if constexpr (1 <= TM / 16) mma_step_sm70<1, TM>(acc_v, A, lane, decode_v); break;
        case 2: if constexpr (2 <= TM / 16) mma_step_sm70<2, TM>(acc_v, A, lane, decode_v); break;
        case 3: if constexpr (3 <= TM / 16) mma_step_sm70<3, TM>(acc_v, A, lane, decode_v); break;
        case 4: if constexpr (4 <= TM / 16) mma_step_sm70<4, TM>(acc_v, A, lane, decode_v); break;
        case 5: if constexpr (5 <= TM / 16) mma_step_sm70<5, TM>(acc_v, A, lane, decode_v); break;
        case 6: if constexpr (6 <= TM / 16) mma_step_sm70<6, TM>(acc_v, A, lane, decode_v); break;
        case 7: if constexpr (7 <= TM / 16) mma_step_sm70<7, TM>(acc_v, A, lane, decode_v); break;
        default: mma_step_sm70<TM / 16, TM>(acc_v, A, lane, decode_v); break;
      }
    }
  } else {
    auto scales_v = [&](int ks, float (&sc)[2][4]) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int row = warp * 16 + nt * 8 + v_idx;
        if constexpr (FMT != FMT_MXFP8) {
          const uint32_t s4 = C::STAGE_SC
              ? *reinterpret_cast<const uint32_t*>(sc_smem + row * C::SCALE_ROW + ks * 4)
              : __ldg(reinterpret_cast<const uint32_t*>(scale_tile + int64_t(row) * C::SCALE_ROW + ks * 4));
          const float2 lo = e4m3x2_to_float2(uint16_t(s4 & 0xffff)), hi = e4m3x2_to_float2(uint16_t(s4 >> 16));
          sc[nt][0] = lo.x, sc[nt][1] = lo.y, sc[nt][2] = hi.x, sc[nt][3] = hi.y;
        } else {
          const uint8_t* ms = sc_smem + row * C::SCALE_ROW + ks * 2;
          sc[nt][0] = sc[nt][1] = exp2f(float(ms[0]) - 127.f);
          sc[nt][2] = sc[nt][3] = exp2f(float(ms[1]) - 127.f);
        }
      }
    };
#pragma unroll 1
    for (int ks = 0; ks < KS; ++ks) {
      cp_wait<C::AS_STG - 2>();
      __syncthreads();
      const int chunk = ks / C::R;
      if constexpr (C::STAGE_W) {
        if (ks % C::R == 0 && chunk + C::WS - 1 < C::KC) load_w(chunk + C::WS - 1);
      }
      issue(ks + C::AS_STG - 1);
      const uint16_t* A = reinterpret_cast<const uint16_t*>(a_smem + (ks % C::AS_STG) * C::A_BYTES);
      const unsigned char* ws = w_smem + ((ks / C::R) % C::WS) * C::W_BYTES;
      float sc[2][4];
      scales_v(ks, sc);
      auto decode_v = [&](int kk, unsigned (&b)[2][4]) {
        const int kc = (ks % C::R) * BK + kk * 16;
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int row = warp * 16 + nt * 8 + v_idx;
          const float s = sc[nt][kk];
          if constexpr (FMT == FMT_NVFP4) {
            const uint2 wd = *reinterpret_cast<const uint2*>(ws + row * C::WSTRIDE + kc / 2);
            const uint32_t m16 = wd.x >> (v_kh * 4);
            const uint32_t n16 = wd.y >> (v_kh * 4);
            const uint32_t b0 = m16 & 0xffu, b1 = (m16 >> 8) & 0xffu;
            const uint32_t b2 = n16 & 0xffu, b3 = (n16 >> 8) & 0xffu;
            b[nt][0] = pack_bf16x2((lut[b0 & 0xf] * s) * factor, (lut[b0 >> 4] * s) * factor);
            b[nt][1] = pack_bf16x2((lut[b1 & 0xf] * s) * factor, (lut[b1 >> 4] * s) * factor);
            b[nt][2] = pack_bf16x2((lut[b2 & 0xf] * s) * factor, (lut[b2 >> 4] * s) * factor);
            b[nt][3] = pack_bf16x2((lut[b3 & 0xf] * s) * factor, (lut[b3 >> 4] * s) * factor);
          } else if constexpr (FMT == FMT_MXFP8) {
            const uint4 q = *reinterpret_cast<const uint4*>(ws + row * C::WSTRIDE + kc);
            const uint32_t wm = v_kh ? q.y : q.x;
            const uint32_t wn = v_kh ? q.w : q.z;
            const float2 f0 = e4m3x2_to_float2(uint16_t(wm & 0xffff)), f1 = e4m3x2_to_float2(uint16_t(wm >> 16));
            const float2 f2 = e4m3x2_to_float2(uint16_t(wn & 0xffff)), f3 = e4m3x2_to_float2(uint16_t(wn >> 16));
            b[nt][0] = pack_bf16x2(f0.x * s, f0.y * s);
            b[nt][1] = pack_bf16x2(f1.x * s, f1.y * s);
            b[nt][2] = pack_bf16x2(f2.x * s, f2.y * s);
            b[nt][3] = pack_bf16x2(f3.x * s, f3.y * s);
          } else if constexpr (FMT == FMT_ATLAS2) {
            const uint32_t wd = *reinterpret_cast<const uint32_t*>(ws + row * C::WSTRIDE + kc / 4);
            const uint32_t m8 = (wd >> (v_kh * 2)) & 0xffu;
            const uint32_t n8 = (wd >> (16 + v_kh * 2)) & 0xffu;
            b[nt][0] = pack_bf16x2((lut[m8 & 3u] * s) * factor, (lut[(m8 >> 2) & 3u] * s) * factor);
            b[nt][1] = pack_bf16x2((lut[(m8 >> 4) & 3u] * s) * factor, (lut[(m8 >> 6) & 3u] * s) * factor);
            b[nt][2] = pack_bf16x2((lut[n8 & 3u] * s) * factor, (lut[(n8 >> 2) & 3u] * s) * factor);
            b[nt][3] = pack_bf16x2((lut[(n8 >> 4) & 3u] * s) * factor, (lut[(n8 >> 6) & 3u] * s) * factor);
          } else {
            const unsigned char* blk = ws + row * C::WSTRIDE + (kc / 32) * 12;
            const int half = (kc >> 4) & 1;
            const uint32_t lo = *reinterpret_cast<const uint32_t*>(blk + half * 4);
            const uint16_t hi16 = *reinterpret_cast<const uint16_t*>(blk + 8 + half * 2);
            const uint32_t mlo = (lo >> (v_kh * 2)) & 0xffu, mhi = (hi16 >> v_kh) & 0xfu;
            const uint32_t nlo = (lo >> (16 + v_kh * 2)) & 0xffu, nhi = (hi16 >> (8 + v_kh)) & 0xfu;
            b[nt][0] = pack_bf16x2((lut[(mlo & 3u) | ((mhi & 1u) << 2)] * s) * factor,
                                   (lut[((mlo >> 2) & 3u) | (((mhi >> 1) & 1u) << 2)] * s) * factor);
            b[nt][1] = pack_bf16x2((lut[((mlo >> 4) & 3u) | (((mhi >> 2) & 1u) << 2)] * s) * factor,
                                   (lut[((mlo >> 6) & 3u) | (((mhi >> 3) & 1u) << 2)] * s) * factor);
            b[nt][2] = pack_bf16x2((lut[(nlo & 3u) | ((nhi & 1u) << 2)] * s) * factor,
                                   (lut[((nlo >> 2) & 3u) | (((nhi >> 1) & 1u) << 2)] * s) * factor);
            b[nt][3] = pack_bf16x2((lut[((nlo >> 4) & 3u) | (((nhi >> 2) & 1u) << 2)] * s) * factor,
                                   (lut[((nlo >> 6) & 3u) | (((nhi >> 3) & 1u) << 2)] * s) * factor);
          }
        }
      };
      switch (mact) {
        case 1: if constexpr (1 <= TM / 16) mma_step_sm70<1, TM>(acc_v, A, lane, decode_v); break;
        case 2: if constexpr (2 <= TM / 16) mma_step_sm70<2, TM>(acc_v, A, lane, decode_v); break;
        case 3: if constexpr (3 <= TM / 16) mma_step_sm70<3, TM>(acc_v, A, lane, decode_v); break;
        case 4: if constexpr (4 <= TM / 16) mma_step_sm70<4, TM>(acc_v, A, lane, decode_v); break;
        case 5: if constexpr (5 <= TM / 16) mma_step_sm70<5, TM>(acc_v, A, lane, decode_v); break;
        case 6: if constexpr (6 <= TM / 16) mma_step_sm70<6, TM>(acc_v, A, lane, decode_v); break;
        case 7: if constexpr (7 <= TM / 16) mma_step_sm70<7, TM>(acc_v, A, lane, decode_v); break;
        default: mma_step_sm70<TM / 16, TM>(acc_v, A, lane, decode_v); break;
      }
    }
  }
  cp_wait<0>();
#pragma unroll
  for (int mt = 0; mt < TM / 16; ++mt) {
    if (mt >= mact) continue;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;
      mma::sm70_reduce_acc8_to_d4(acc_v[mt][nt], acc[mt][nt]);
    }
  }
#else
  unsigned bcur[BK / 16][2][2], bnext[BK / 16][2][2];
#pragma unroll 1
  for (int ks = 0; ks < KS; ++ks) {
    cp_wait<C::AS_STG - 2>();
    __syncthreads();
    const int chunk = ks / C::R;
    // first step of a chunk: the chunk WS - 1 ahead into the slot the previous chunk used (read before the barrier)
    if (ks % C::R == 0 && chunk + C::WS - 1 < C::KC) load_w(chunk + C::WS - 1);
    issue(ks + C::AS_STG - 1);
    if (ks == 0) {
      float sc0[2][4];
      scales_of(0, sc0);
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) decode(0, kk, sc0, bcur);
    }
    const uint16_t* A = reinterpret_cast<const uint16_t*>(a_smem + (ks % C::AS_STG) * C::A_BYTES);
    const bool more = ks + 1 < KS;
    float scn[2][4];
    if (more) scales_of(ks + 1, scn);
    auto after = [&](int kk) {
      if (more) decode(ks + 1, kk, scn, bnext);
    };
    // the row-block count is a template constant inside mma_step: a runtime `mt < mact` test would be if-converted
    // into predicated MMAs, which still occupy the tensor pipe (8x the work for a 16-row tile)
    switch (mact) {
      case 1: if constexpr (1 <= TM / 16) mma_step<1, TM>(acc, A, bcur, lane, after); break;
      case 2: if constexpr (2 <= TM / 16) mma_step<2, TM>(acc, A, bcur, lane, after); break;
      case 3: if constexpr (3 <= TM / 16) mma_step<3, TM>(acc, A, bcur, lane, after); break;
      case 4: if constexpr (4 <= TM / 16) mma_step<4, TM>(acc, A, bcur, lane, after); break;
      case 5: if constexpr (5 <= TM / 16) mma_step<5, TM>(acc, A, bcur, lane, after); break;
      case 6: if constexpr (6 <= TM / 16) mma_step<6, TM>(acc, A, bcur, lane, after); break;
      case 7: if constexpr (7 <= TM / 16) mma_step<7, TM>(acc, A, bcur, lane, after); break;
      default: mma_step<TM / 16, TM>(acc, A, bcur, lane, after); break;
    }
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) bcur[kk][nt][0] = bnext[kk][nt][0], bcur[kk][nt][1] = bnext[kk][nt][1];
  }
  cp_wait<0>();
#endif

#pragma unroll
  for (int mt = 0; mt < TM / 16; ++mt) {
    if (mt >= mact) continue;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = mt * 16 + (lane >> 2) + half * 8;
      if (row >= n) continue;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int cl = warp * 16 + nt * 8 + c * 2;  // column inside the tile
        float c0 = acc[mt][nt][2 * half], c1 = acc[mt][nt][2 * half + 1];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
        if constexpr (C::VOLTA_DIRECT && FMT == FMT_NVFP4) c0 *= factor, c1 *= factor;
#endif
        if constexpr (MODE == Mode::Up) {
          const float gt = round_bf16(c0), u = round_bf16(c1);
          const float sig = 1.f / (1.f + expf(-gt));
          const float v = __fmul_rn(__fmul_rn(softcap(gt, 4.f), sig), softcap(u, 25.f));
          g.out[size_t(pb + row) * kInter + (n0 + cl) / 2] = f2bf_bits(v);
        } else {
          const int dst = g.w.pair_dst[pb + row];
          *reinterpret_cast<uint32_t*>(g.out + size_t(dst) * kHidden + n0 + cl) = pack_bf16x2(c0, c1);
        }
      }
    }
  }
}

template <int FMT, Mode MODE, int K, int TM, bool LOW_SMEM = false>
void launch(const GemmArgs& g, int grid, cudaStream_t stream) {
  constexpr int SMEM = Cfg<FMT, K, TM, LOW_SMEM>::SMEM;
  if constexpr (SMEM > 49152) {
    static std::atomic<uint32_t> attr_done{0};
    int dev = 0;
    cudaGetDevice(&dev);
    const uint32_t bit = 1u << (dev & 31);
    if (!(attr_done.load(std::memory_order_acquire) & bit)) {
      const cudaError_t attr = cudaFuncSetAttribute(
          wgemm_kernel<FMT, MODE, K, TM, LOW_SMEM>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
      if (attr != cudaSuccess) throw std::runtime_error("prefill wgemm: shared memory attribute failed");
      attr_done.fetch_or(bit, std::memory_order_release);
    }
  }
  wgemm_kernel<FMT, MODE, K, TM, LOW_SMEM><<<grid, NT, SMEM, stream>>>(g);
}

template <int FMT, Mode MODE, int K>
void launch_rows(const GemmArgs& g, int tm, int grid_per_row_tile, int row_tiles, cudaStream_t stream) {
  if (current_max_smem_optin() < 90000) {
    if (tm != 32) throw std::invalid_argument("prefill wgemm: low-SMEM devices require moe_tile_rows == 32");
    launch<FMT, MODE, K, 32, true>(g, grid_per_row_tile * row_tiles, stream);
    return;
  }
  if (tm == 32) launch<FMT, MODE, K, 32, false>(g, grid_per_row_tile * row_tiles, stream);
  else launch<FMT, MODE, K, 128, false>(g, grid_per_row_tile * row_tiles, stream);
}

GemmArgs expert_args(const void* codes, const void* scales, const float* factors,
                     const uint8_t* const* code_ptrs, const uint8_t* const* scale_ptrs,
                     const uint8_t* active_mask, const ActT* a, const MoeWork& w, ActT* out) {
  GemmArgs g;
  g.codes = static_cast<const uint8_t*>(codes);
  g.scales = static_cast<const uint8_t*>(scales);
  g.factors = factors;
  g.code_ptrs = code_ptrs;
  g.scale_ptrs = scale_ptrs;
  g.active_mask = active_mask;
  g.a = reinterpret_cast<const uint16_t*>(a);
  g.w = w;
  g.out = reinterpret_cast<uint16_t*>(out);
  return g;
}

// ----------------------------------------------------------------------------- EXL3 / Escha W2 prefill

struct Exl3GemmArgs {
  const uint8_t* codes = nullptr;  // u8 [128, N/16, K/16, 32*BITS]
  const __half* rin = nullptr;     // f16 [128, K]
  const __half* rout = nullptr;    // f16 [128, N]
  const uint8_t* const* code_ptrs = nullptr;  // optional [128] per-expert codes base pointers
  const uint8_t* const* rin_ptrs = nullptr;   // optional [128] per-expert rin base pointers
  const uint8_t* const* rout_ptrs = nullptr;  // optional [128] per-expert rout base pointers
  const uint8_t* active_mask = nullptr;       // optional [128] u8 mask for sub-batched prefill
  const uint16_t* a = nullptr;     // bf16 activations
  MoeWork w;
  uint16_t* out = nullptr;         // bf16 [T*8, 2304] (Up: scratch [gate; up]; Down: down at pair_dst)
};

template <int MT, int TM, int BITS>
__device__ __forceinline__ void mma_step_exl3(float (&acc)[TM / 16][2][4], const uint16_t* A,
                                              const uint8_t* w_warp, int lane) {
  constexpr int TILE_BYTES = 32 * BITS;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
  // Pascal sm_60/sm_61 fast path: direct SMEM uint4 A reads (0 A shuffles) + hoisted B shuffles across MT
  // processed in two 8-element K-halves (h=0,1) so live B/A regs stay tiny (zero local-memory spills).
  const int g = lane >> 2, c = lane & 3;
  const int bl0 = 4 * (2 * c), bl1 = 4 * (2 * c + 1);
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_warp + kk * TILE_BYTES);
    uint32_t h01, h23, h45, h67;
    exl3::dq8_mcg<BITS>([&](int wi) { return tw[wi]; }, lane, h01, h23, h45, h67);
    const uint32_t bsrc[2][2] = {{h01, h23}, {h45, h67}};
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
      for (int kh = 0; kh < 2; ++kh) {
        float2 b0[4], b1[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          const uint32_t v0 = __shfl_sync(0xffffffffu, bsrc[nt][kh], bl0 + p);
          const uint32_t v1 = __shfl_sync(0xffffffffu, bsrc[nt][kh], bl1 + p);
          b0[p] = __half22float2(*reinterpret_cast<const __half2*>(&v0));
          b1[p] = __half22float2(*reinterpret_cast<const __half2*>(&v1));
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          const uint4 alo = *reinterpret_cast<const uint4*>(A + swz<16>(mt * 16 + g, kk * 2 + kh));
          const uint4 ahi = *reinterpret_cast<const uint4*>(A + swz<16>(mt * 16 + g + 8, kk * 2 + kh));
          const uint32_t al[4] = {alo.x, alo.y, alo.z, alo.w};
          const uint32_t ah[4] = {ahi.x, ahi.y, ahi.z, ahi.w};
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            const float2 fal = __half22float2(*reinterpret_cast<const __half2*>(&al[p]));
            const float2 fah = __half22float2(*reinterpret_cast<const __half2*>(&ah[p]));
            acc[mt][nt][0] = fmaf(fal.y, b0[p].y, fmaf(fal.x, b0[p].x, acc[mt][nt][0]));
            acc[mt][nt][1] = fmaf(fal.y, b1[p].y, fmaf(fal.x, b1[p].x, acc[mt][nt][1]));
            acc[mt][nt][2] = fmaf(fah.y, b0[p].y, fmaf(fah.x, b0[p].x, acc[mt][nt][2]));
            acc[mt][nt][3] = fmaf(fah.y, b1[p].y, fmaf(fah.x, b1[p].x, acc[mt][nt][3]));
          }
        }
      }
    }
  }
#else
  uint32_t h01, h23, h45, h67;
  {
    const uint32_t* tw0 = reinterpret_cast<const uint32_t*>(w_warp);
    exl3::dq8_mcg<BITS>([&](int wi) { return tw0[wi]; }, lane, h01, h23, h45, h67);
  }
#pragma unroll
  for (int kk = 0; kk < 8; ++kk) {
    uint32_t nh01, nh23, nh45, nh67;
    // Stream ldmatrix_x4 per 16-row tile (4 regs instead of 4*MT regs) to eliminate register spills at TM=128.
    constexpr int HALF_MT = (MT + 1) / 2;
#pragma unroll
    for (int mt = 0; mt < HALF_MT; ++mt) {
      unsigned a[4];
      ldmatrix_x4(a[0], a[1], a[2], a[3],
                  smem_addr(A + swz<16>(mt * 16 + (lane & 15), kk * 2 + (lane >> 4))));
      mma_f16(acc[mt][0], a, h01, h23);
      mma_f16(acc[mt][1], a, h45, h67);
    }
    if (kk + 1 < 8) {
      const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_warp + (kk + 1) * TILE_BYTES);
      exl3::dq8_mcg<BITS>([&](int wi) { return tw[wi]; }, lane, nh01, nh23, nh45, nh67);
    }
#pragma unroll
    for (int mt = HALF_MT; mt < MT; ++mt) {
      unsigned a[4];
      ldmatrix_x4(a[0], a[1], a[2], a[3],
                  smem_addr(A + swz<16>(mt * 16 + (lane & 15), kk * 2 + (lane >> 4))));
      mma_f16(acc[mt][0], a, h01, h23);
      mma_f16(acc[mt][1], a, h45, h67);
    }
    if (kk + 1 < 8) {
      h01 = nh01;
      h23 = nh23;
      h45 = nh45;
      h67 = nh67;
    }
  }
#endif
}

template <int TM>
inline constexpr int exl3_a_stages() {
  return (TM == 32 || !INGINE_ACT_FP16) ? 2 : 1;
}

template <int BITS, Mode MODE, int K, int TM>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
__global__ void __launch_bounds__(NT, TM == 32 ? 2 : 1) wgemm_exl3_kernel(const Exl3GemmArgs g) {
#else
__global__ void __launch_bounds__(NT, TM == 32 ? 3 : 1) wgemm_exl3_kernel(const Exl3GemmArgs g) {
#endif
  constexpr int NE = MODE == Mode::Up ? 2 * kInter : kHidden;  // 2304
  constexpr int NTILES_N = NE / TN;                            // 18
  constexpr int BK_EXL = 128;
  constexpr int KS = K / BK_EXL;                               // 18 (Up) or 9 (Down)
  constexpr int TILE_BYTES = 32 * BITS;                        // 64 or 96
  constexpr int KT = K / 16;                                   // 144 or 72
  constexpr int W_CHUNK_ROW = 8 * TILE_BYTES;                  // 512B (2-bit) or 768B (3-bit)
  constexpr int W_STAGE_BYTES = 8 * W_CHUNK_ROW;               // 4096B (2-bit) or 6144B (3-bit)
  constexpr int AS_EXL = exl3_a_stages<TM>();
  constexpr int A_BYTES = TM * BK_EXL * 2;                     // 8192B (TM=32) or 32768B (TM=128)
  static_assert(AS_EXL * A_BYTES >= 16 * 128 * 4);

  extern __shared__ __align__(128) unsigned char smem[];
  unsigned char* A_base = smem;
  unsigned char* W_smem = smem + AS_EXL * A_BYTES;

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int tile_idx = blockIdx.x / NTILES_N;
  if (tile_idx >= *g.w.tiles_n) return;
  const int4 tile = g.w.tiles[tile_idx];
  const int e = tile.x, pb = tile.y, n = tile.z;
  if (g.active_mask && !g.active_mask[e]) return;
  const int n_blk = blockIdx.x % NTILES_N;
  const int n0 = n_blk * TN;
  const int nt0 = n_blk * 8;
  const int mact = min((n + 15) / 16, TM / 16);

  const uint8_t* w_base =
      (g.code_ptrs ? g.code_ptrs[e] : (g.codes + int64_t(e) * (NE / 16) * KT * TILE_BYTES)) +
      int64_t(nt0) * KT * TILE_BYTES;
  auto load_w = [&](int ks) {
    unsigned char* dst = W_smem + (ks & 1) * W_STAGE_BYTES;
    constexpr int CPR = W_CHUNK_ROW / 16;  // 32 (2-bit) or 48 (3-bit)
    constexpr int TOTAL = 8 * CPR;         // 256 (2-bit) or 384 (3-bit)
#pragma unroll
    for (int id = tid; id < TOTAL; id += NT) {
      const int nr = id / CPR, ch = id % CPR;
      cp_async16(dst + nr * W_CHUNK_ROW + ch * 16,
                 w_base + int64_t(nr) * KT * TILE_BYTES + ks * W_CHUNK_ROW + ch * 16);
    }
    cp_commit();
  };

  load_w(0);

  float acc[TM / 16][2][4];
#pragma unroll
  for (int a = 0; a < TM / 16; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

  const __half* rin_base = g.rin_ptrs ? reinterpret_cast<const __half*>(g.rin_ptrs[e]) : (g.rin + size_t(e) * K);

#pragma unroll 1
  for (int ks = 0; ks < KS; ++ks) {
    // Rotate TM activation rows for this 128-element K-block into swizzled FP16 A_smem stage.
    uint16_t* A_smem = reinterpret_cast<uint16_t*>(A_base + (AS_EXL > 1 ? (ks & 1) * A_BYTES : 0));
    const uint2 rin_raw = __ldg(reinterpret_cast<const uint2*>(rin_base + ks * BK_EXL + lane * 4));
    const __half2 rin01 = *reinterpret_cast<const __half2*>(&rin_raw.x);
    const __half2 rin23 = *reinterpret_cast<const __half2*>(&rin_raw.y);
    const float r_in0 = __low2float(rin01), r_in1 = __high2float(rin01);
    const float r_in2 = __low2float(rin23), r_in3 = __high2float(rin23);

    const int r_lim = mact * 16;
    for (int r = warp; r < r_lim; r += 8) {
      uint2 h4 = {0u, 0u};
      if (r < n) {
        const size_t src_row = (MODE == Mode::Up) ? size_t(g.w.pair_token[pb + r]) : size_t(pb + r);
        const uint2 raw = *reinterpret_cast<const uint2*>(g.a + src_row * K + ks * BK_EXL + lane * 4);
        float v[4] = {bf_lo(raw.x) * r_in0, bf_hi(raw.x) * r_in1, bf_lo(raw.y) * r_in2, bf_hi(raw.y) * r_in3};
        exl3::fwht_128_warp(v, lane);
        h4.x = pack_half2(v[0], v[1]);
        h4.y = pack_half2(v[2], v[3]);
      }
      *reinterpret_cast<uint2*>(A_smem + swz<16>(r, lane >> 1) + (lane & 1) * 4) = h4;
    }

    cp_wait<0>();
    __syncthreads();
    if (ks + 1 < KS) load_w(ks + 1);

    const uint8_t* w_warp = W_smem + (ks & 1) * W_STAGE_BYTES + warp * W_CHUNK_ROW;
    switch (mact) {
      case 1: if constexpr (1 <= TM / 16) mma_step_exl3<1, TM, BITS>(acc, A_smem, w_warp, lane); break;
      case 2: if constexpr (2 <= TM / 16) mma_step_exl3<2, TM, BITS>(acc, A_smem, w_warp, lane); break;
      case 3: if constexpr (3 <= TM / 16) mma_step_exl3<3, TM, BITS>(acc, A_smem, w_warp, lane); break;
      case 4: if constexpr (4 <= TM / 16) mma_step_exl3<4, TM, BITS>(acc, A_smem, w_warp, lane); break;
      case 5: if constexpr (5 <= TM / 16) mma_step_exl3<5, TM, BITS>(acc, A_smem, w_warp, lane); break;
      case 6: if constexpr (6 <= TM / 16) mma_step_exl3<6, TM, BITS>(acc, A_smem, w_warp, lane); break;
      case 7: if constexpr (7 <= TM / 16) mma_step_exl3<7, TM, BITS>(acc, A_smem, w_warp, lane); break;
      default: mma_step_exl3<TM / 16, TM, BITS>(acc, A_smem, w_warp, lane); break;
    }
    if constexpr (AS_EXL == 1) __syncthreads();
  }
  if constexpr (AS_EXL > 1) __syncthreads();

  // Epilogue: 16 rows at a time through shared ep[16][128] (reuses A_smem), output H_128 + rout -> BF16
  float* ep = reinterpret_cast<float*>(smem);
  const __half* rout_p =
      (g.rout_ptrs ? reinterpret_cast<const __half*>(g.rout_ptrs[e]) : (g.rout + size_t(e) * NE)) + n0 + lane * 4;
  const uint2 rout_raw = __ldg(reinterpret_cast<const uint2*>(rout_p));
  const __half2 rout01 = *reinterpret_cast<const __half2*>(&rout_raw.x);
  const __half2 rout23 = *reinterpret_cast<const __half2*>(&rout_raw.y);
  const float r_out0 = __low2float(rout01), r_out1 = __high2float(rout01);
  const float r_out2 = __low2float(rout23), r_out3 = __high2float(rout23);

  const int g_row = lane >> 2, c_pair = lane & 3;
  for (int mt = 0; mt < mact; ++mt) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int cl = warp * 16 + nt * 8 + c_pair * 2;
      ep[g_row * 128 + cl] = acc[mt][nt][0];
      ep[g_row * 128 + cl + 1] = acc[mt][nt][1];
      ep[(g_row + 8) * 128 + cl] = acc[mt][nt][2];
      ep[(g_row + 8) * 128 + cl + 1] = acc[mt][nt][3];
    }
    __syncthreads();
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = warp + half * 8;
      const int row = mt * 16 + r;
      if (row < n) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = ep[r * 128 + lane * 4 + i];
        exl3::fwht_128_warp(v, lane);
        uint2 out4;
        out4.x = pack_bf16x2(v[0] * r_out0, v[1] * r_out1);
        out4.y = pack_bf16x2(v[2] * r_out2, v[3] * r_out3);
        const size_t dst_row = (MODE == Mode::Up) ? size_t(pb + row) : size_t(g.w.pair_dst[pb + row]);
        *reinterpret_cast<uint2*>(g.out + dst_row * NE + n0 + lane * 4) = out4;
      }
    }
    __syncthreads();
  }
}

__device__ __forceinline__ uint2 swiglu4(uint2 g4, uint2 u4) {
  auto elem = [](float g, float u) {
    const float sig = 1.f / (1.f + expf(-g));
    return __fmul_rn(__fmul_rn(softcap(g, 4.f), sig), softcap(u, 25.f));
  };
  uint2 out;
  out.x = pack_bf16x2(elem(bf_lo(g4.x), bf_lo(u4.x)), elem(bf_hi(g4.x), bf_hi(u4.x)));
  out.y = pack_bf16x2(elem(bf_lo(g4.y), bf_lo(u4.y)), elem(bf_hi(g4.y), bf_hi(u4.y)));
  return out;
}

// Stacked [gate(0..1151); up(1152..2303)] -> softcapped SwiGLU act [pairs, 1152].
__global__ void moe_exl3_swiglu_kernel(const uint16_t* __restrict__ gu, uint16_t* __restrict__ act, int pairs) {
  const int p = blockIdx.x;
  if (p >= pairs) return;
  const uint2* g_row = reinterpret_cast<const uint2*>(gu + size_t(p) * (2 * kInter));
  const uint2* u_row = reinterpret_cast<const uint2*>(gu + size_t(p) * (2 * kInter) + kInter);
  uint2* dst = reinterpret_cast<uint2*>(act + size_t(p) * kInter);
  constexpr int N4 = kInter / 4;  // 288
  for (int i = threadIdx.x; i < N4; i += blockDim.x) dst[i] = swiglu4(g_row[i], u_row[i]);
}

// Tile-masked variant for sub-batched tiered prefill: only writes `act` rows belonging to active experts.
__global__ void moe_exl3_swiglu_masked_kernel(const uint16_t* __restrict__ gu, uint16_t* __restrict__ act,
                                              const MoeWork w, const uint8_t* __restrict__ active_mask) {
  const int tile_idx = blockIdx.x;
  if (tile_idx >= *w.tiles_n) return;
  const int4 tile = w.tiles[tile_idx];
  if (!active_mask[tile.x]) return;
  const int pb = tile.y, n = tile.z;
  constexpr int N4 = kInter / 4;  // 288
  for (int r = 0; r < n; ++r) {
    const int p = pb + r;
    const uint2* g_row = reinterpret_cast<const uint2*>(gu + size_t(p) * (2 * kInter));
    const uint2* u_row = reinterpret_cast<const uint2*>(gu + size_t(p) * (2 * kInter) + kInter);
    uint2* dst = reinterpret_cast<uint2*>(act + size_t(p) * kInter);
    for (int i = threadIdx.x; i < N4; i += blockDim.x) dst[i] = swiglu4(g_row[i], u_row[i]);
  }
}

template <int BITS, Mode MODE, int K, int TM>
void launch_exl3(const Exl3GemmArgs& g, int grid, cudaStream_t stream) {
  constexpr int SMEM = exl3_a_stages<TM>() * TM * 128 * 2 + 2 * 8 * 8 * 32 * BITS;
  if constexpr (SMEM > 49152) {
    static std::atomic<uint32_t> attr_done{0};
    int dev = 0;
    cudaGetDevice(&dev);
    const uint32_t bit = 1u << (dev & 31);
    if (!(attr_done.load(std::memory_order_acquire) & bit)) {
      const cudaError_t attr =
          cudaFuncSetAttribute(wgemm_exl3_kernel<BITS, MODE, K, TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
      if (attr != cudaSuccess) throw std::runtime_error("prefill wgemm(EXL3): shared memory attribute failed");
      attr_done.fetch_or(bit, std::memory_order_release);
    }
  }
  wgemm_exl3_kernel<BITS, MODE, K, TM><<<grid, NT, SMEM, stream>>>(g);
}

template <int BITS, Mode MODE, int K>
void launch_exl3_rows(const Exl3GemmArgs& g, int tm, int grid_per_row_tile, int row_tiles, cudaStream_t stream) {
  if (tm == 32) launch_exl3<BITS, MODE, K, 32>(g, grid_per_row_tile * row_tiles, stream);
  else launch_exl3<BITS, MODE, K, 128>(g, grid_per_row_tile * row_tiles, stream);
}

}  // namespace

void moe_gate_up(const MoeExperts& e, const ActT* x, int T, const MoeWork& w, ActT* act,
                 cudaStream_t stream) {
  if (moe_gate_up_decode(e, x, T, w, act, stream)) return;
  if (e.fmt == FMT_EXL3_2 || e.fmt == FMT_EXL3_3) {
    if (!e.exl3_scratch || (!e.w13_rout && !e.w13_rout_ptrs))
      throw std::invalid_argument("prefill moe_gate_up(EXL3): exl3_scratch and w13_rout required");
    Exl3GemmArgs g;
    g.codes = static_cast<const uint8_t*>(e.w13);
    g.rin = static_cast<const __half*>(e.w13_scale);
    g.rout = static_cast<const __half*>(e.w13_rout);
    g.code_ptrs = e.w13_ptrs;
    g.rin_ptrs = e.w13_scale_ptrs;
    g.rout_ptrs = e.w13_rout_ptrs;
    g.active_mask = e.active_mask;
    g.a = reinterpret_cast<const uint16_t*>(x);
    g.w = w;
    g.out = reinterpret_cast<uint16_t*>(e.exl3_scratch);
    const int tr = e.tile_rows > 0 ? e.tile_rows : moe_tile_rows(T);
    const int mt = moe_max_tiles(T, tr);
    if (e.fmt == FMT_EXL3_2)
      launch_exl3_rows<2, Mode::Up, kHidden>(g, tr, 2 * kInter / TN, mt, stream);
    else
      launch_exl3_rows<3, Mode::Up, kHidden>(g, tr, 2 * kInter / TN, mt, stream);
    if (e.active_mask) {
      moe_exl3_swiglu_masked_kernel<<<mt, 256, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(e.exl3_scratch), reinterpret_cast<uint16_t*>(act), w, e.active_mask);
    } else {
      const int pairs = T * kTopK;
      moe_exl3_swiglu_kernel<<<pairs, 256, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(e.exl3_scratch), reinterpret_cast<uint16_t*>(act), pairs);
    }
    return;
  }
  const GemmArgs g = expert_args(e.w13, e.w13_scale, e.w13_factor, e.w13_ptrs, e.w13_scale_ptrs,
                                 e.active_mask, x, w, act);
  if (e.fmt == FMT_NVFP4) launch_rows<FMT_NVFP4, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
  else if (e.fmt == FMT_MXFP8) launch_rows<FMT_MXFP8, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
  else if (e.fmt == FMT_ATLAS2) launch_rows<FMT_ATLAS2, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
#if INGINE_ACT_FP16
  else if (e.fmt == FMT_GQ5) launch_rows<FMT_GQ5, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
  else if (e.fmt == FMT_GQI4) launch_rows<FMT_GQI4, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
#endif
  else launch_rows<FMT_ATLAS3, Mode::Up, kHidden>(g, moe_tile_rows(T), 2 * kInter / TN, moe_max_tiles(T), stream);
}

void moe_down(const MoeExperts& e, const ActT* act, int T, const MoeWork& w, ActT* down,
              cudaStream_t stream) {
  if (moe_down_decode(e, act, T, w, down, stream)) return;
  const int w2_fmt = e.w2_fmt >= 0 ? e.w2_fmt : e.fmt;
  if (w2_fmt == FMT_EXL3_2 || w2_fmt == FMT_EXL3_3) {
    if (!e.w2_rout && !e.w2_rout_ptrs) throw std::invalid_argument("prefill moe_down(EXL3): w2_rout required");
    Exl3GemmArgs g;
    g.codes = static_cast<const uint8_t*>(e.w2);
    g.rin = static_cast<const __half*>(e.w2_scale);
    g.rout = static_cast<const __half*>(e.w2_rout);
    g.code_ptrs = e.w2_ptrs;
    g.rin_ptrs = e.w2_scale_ptrs;
    g.rout_ptrs = e.w2_rout_ptrs;
    g.active_mask = e.active_mask;
    g.a = reinterpret_cast<const uint16_t*>(act);
    g.w = w;
    g.out = reinterpret_cast<uint16_t*>(down);
    const int tr = e.tile_rows > 0 ? e.tile_rows : moe_tile_rows(T);
    const int mt = moe_max_tiles(T, tr);
    if (w2_fmt == FMT_EXL3_2)
      launch_exl3_rows<2, Mode::Down, kInter>(g, tr, kHidden / TN, mt, stream);
    else
      launch_exl3_rows<3, Mode::Down, kInter>(g, tr, kHidden / TN, mt, stream);
    return;
  }
  const GemmArgs g = expert_args(e.w2, e.w2_scale, e.w2_factor, e.w2_ptrs, e.w2_scale_ptrs,
                                 e.active_mask, act, w, down);
  if (w2_fmt == FMT_NVFP4) launch_rows<FMT_NVFP4, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
  else if (w2_fmt == FMT_MXFP8) launch_rows<FMT_MXFP8, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
  else if (w2_fmt == FMT_ATLAS2) launch_rows<FMT_ATLAS2, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
#if INGINE_ACT_FP16
  else if (w2_fmt == FMT_GQ5) launch_rows<FMT_GQ5, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
  else if (w2_fmt == FMT_GQI4) launch_rows<FMT_GQI4, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
#endif
  else launch_rows<FMT_ATLAS3, Mode::Down, kInter>(g, moe_tile_rows(T), kHidden / TN, moe_max_tiles(T), stream);
}

}  // namespace ingine::kernels::prefill

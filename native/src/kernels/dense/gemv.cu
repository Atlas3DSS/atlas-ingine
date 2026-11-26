// proj, proj_conv, head, fp8_rows: weight-streaming GEMVs for M = 1..8 tokens.
//
// Design. Decode GEMVs are pure weight streams (M <= 8 activations, K = 2304 or 3072): the goal is to keep
// DRAM busy with the fewest instructions per weight byte. Two kernels share the epilogues:
//
// M = 1, `gemv_kernel` (CUDA cores):
// * One row per lane group: a full warp per BF16 row, a half warp per FP8 row, so every lane owns exactly
//   CH = K * bytes / (16 * LANES) 16-byte chunks (BF16 K=2304: 9; FP8 K=2304: 9; K=3072: 12). Chunk i of
//   lane l is bytes [(i * LANES + l) * 16, +16) of the row: each load instruction of a group reads one
//   contiguous 256/512-byte span (fully coalesced).
// * The CH chunk loads of a row are issued at the top of the kernel (read-only streaming loads, no L1
//   allocation; ptxas interleaves them with the FMAs of earlier chunks, leaving ~4 chunks = 2 KB in
//   flight per warp), and with ~32 resident warps per SM (low register count) that is ~64 KB per SM,
//   several times the bandwidth-delay product. (Measured alternatives, no gain: persistent grids, L2 bulk
//   prefetch of the row, 128/512-thread blocks.)
// * The activation is read with 16-byte L1-cached loads at the matching columns; FP32 FMAs, then a
//   warp reduction.
//
// M >= 2, `gemv_mma_kernel` (tensor cores): on CUDA cores every weight costs M FMAs plus the unpacking of
// M activations, which made M = 8 instruction-bound (FP8 proj lost 20-40%; staging the activations in
// shared memory or giving each lane two rows was measured and did not help: both raise register or
// shared-memory pressure and cut occupancy). Instead each warp runs mma.m16n8k16 (BF16 x BF16 -> FP32):
// A = 16 weight rows, B = the activations of up to 8 tokens (N = 8 = max M; missing tokens are zero),
// so one instruction does 16 x 8 x 16 multiply-adds with FP32 accumulation, exact BF16 products as on the
// CUDA-core path. The reduction order over k is free, so the fragments are fed straight from coalesced
// 16-byte loads: lane (g, t) loads columns [c + CPL t, +CPL) of rows g and g + 8 (CPL = 8 BF16 or 16
// FP8 converted exactly to BF16) and the same columns of token g's activation; any k-permutation that A
// and B share gives the same dot products. A block of SPLITK warps owns 16 rows; warp w covers a 1/SPLITK
// slice of K (enough warps in flight for small row counts), and the partial 16 x 8 tiles are summed in
// shared memory in a fixed order. proj blocks that straddle a segment boundary with different inputs run
// one pass per input with the other rows masked to zero.
//
// Epilogues: proj rounds to BF16 (input vector per row from the device-side segment table), head applies
// the softcap, proj_conv applies the post-attention causal conv for its channel (= row) over the call's
// M tokens, rewrites that channel's conv state and optionally records the conv inputs (conv_hist), so
// every op is one launch with no grid-wide dependency. M is a template parameter; FP8 rows multiply the
// dot by the row scale.
//
// Programmatic dependent launch (pdl=True): the M = 1 kernel issues all its row's weight loads, the
// M >= 2 kernel its first batch, before griddepcontrol.wait, so they overlap the previous kernel's tail
// (pdl_wait_fenced: a plain wait lets ptxas sink the loads below it; measured M = 1 proj 13.7 -> 11.8 us
// behind a 2 us predecessor); activations are read (coherent, non-hoistable loads: ld_act) and
// outputs/state written only after it, and the next kernel is released right after the wait.
// L2: the weight rows are read once per step, so their lines are filled evict-first (kernels/common/l2.cuh):
// they stop evicting the activations, KV rings and the small kernels' evict-last constants.
// Ported from engine/csrc/dense/gemv.cu (kernel bodies unchanged; torch launchers -> raw pointers).
#include "core/q5_k.h"
#include "kernels/common/mma_sm70.cuh"
#include "kernels/common/l2.cuh"
#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"

#include <cstdint>
#include <type_traits>

namespace ingine::kernels::dense {
namespace {

constexpr l2::Evict WEIGHT_L2 = l2::Evict::first;

// ---- epilogues ----------------------------------------------------------------------------------------
// input_of(row): input index of a row (-1: row not computed); x_of(i): that input, bf16 [M, K].
// kScatter epilogues store one (row, token) value at a time (store1); proj_conv needs every token of a
// row together (store: lane-parallel for the CUDA-core kernel; store_row: one thread).

template <int M>
struct ProjEpi {
  static constexpr int kInputs = 3;
  static constexpr bool kScatter = true;
  const ActT* xin;  // [3, M, H]
  const int* segments;       // [S, 3]
  int nseg, R;
  float* out;                // [M, R]
  __device__ int input_of(int row) const {
    for (int s = 0; s < nseg; ++s) {
      const int b = __ldg(segments + 3 * s), e = __ldg(segments + 3 * s + 1);
      if (row >= b && row < e) return __ldg(segments + 3 * s + 2);
    }
    return -1;
  }
  __device__ const ActT* x_of(int i) const { return xin + size_t(i) * M * H; }
  __device__ void store1(int row, int m, float v) const { out[size_t(m) * R + row] = round_bf16(v); }
};

template <int M>
struct HeadEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const ActT* x;
  int V;
  float* out;  // [M, V]
  __device__ int input_of(int) const { return 0; }
  __device__ const ActT* x_of(int) const { return x; }
  __device__ void store1(int row, int m, float v) const {
    out[size_t(m) * V + row] = tanhf(round_bf16(v) / 30.f) * 30.f;
  }
};

// plain GEMV (drafter): one input x [M, K] (row stride K), out [M, R] = bf16(w . x)
template <int M>
struct GemvEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = true;
  const ActT* x;
  int R;
  float* out;  // [M, R]
  __device__ int input_of(int) const { return 0; }
  __device__ const ActT* x_of(int) const { return x; }
  __device__ void store1(int row, int m, float v) const { out[size_t(m) * R + row] = round_bf16(v); }
};

template <int M>
struct ConvEpi {
  static constexpr int kInputs = 1;
  static constexpr bool kScatter = false;  // the conv of token m needs y of tokens m-3 .. m
  const ActT* x;  // [M, 3072]
  const float* conv_w;     // [H, 4], tap 3 = current token
  float* state;            // [3, H], oldest first
  float* out;              // [M, H]
  float* hist;             // [M, H] BF16-rounded conv inputs y, or null
  __device__ int input_of(int) const { return 0; }
  __device__ const ActT* x_of(int) const { return x; }
  __device__ void load_ext(int c, const float (&acc)[M], float (&ext)[M + 3]) const {
#pragma unroll
    for (int j = 0; j < 3; ++j) ext[j] = state[j * H + c];
#pragma unroll
    for (int m = 0; m < M; ++m) ext[3 + m] = round_bf16(acc[m]);
  }
  __device__ void token(int c, int m, const float (&ext)[M + 3]) const {
    const float4 cw = __ldg(reinterpret_cast<const float4*>(conv_w) + c);
    const float s = cw.x * ext[m] + cw.y * ext[m + 1] + cw.z * ext[m + 2] + cw.w * ext[m + 3];
    out[size_t(m) * H + c] = (s + ext[m + 3]) * (1.f / sqrtf(5.f));
    if (hist) hist[size_t(m) * H + c] = ext[m + 3];
  }
  // lane-parallel (all lanes of the group hold every token's sum)
  __device__ void store(int c, const float (&acc)[M], int gl, unsigned mask) const {
    float ext[M + 3];
    load_ext(c, acc, ext);
#pragma unroll
    for (int m = 0; m < M; ++m)
      if (gl == m) token(c, m, ext);
    __syncwarp(mask);  // the group has read the old state before it is overwritten
#pragma unroll
    for (int j = 0; j < 3; ++j)
      if (gl == 8 + j) state[j * H + c] = ext[M + j];
  }
  // one thread per channel
  __device__ void store_row(int c, const float (&acc)[M]) const {
    float ext[M + 3];
    load_ext(c, acc, ext);
#pragma unroll
    for (int m = 0; m < M; ++m) token(c, m, ext);
#pragma unroll
    for (int j = 0; j < 3; ++j) state[j * H + c] = ext[M + j];
  }
};

// ---- M = 1: CUDA cores --------------------------------------------------------------------------------

template <bool FP8, int K, int LANES, int BLOCK, class Epi>
__global__ void __launch_bounds__(BLOCK) gemv_kernel(const uint8_t* __restrict__ w,
                                                     const float* __restrict__ w_scale, int rows, uint32_t zero,
                                                     Epi epi) {
  constexpr int EPC = FP8 ? 16 : 8;  // elements per 16-byte chunk
  constexpr int CH = K / (EPC * LANES);
  static_assert(K % (EPC * LANES) == 0, "row must split evenly over the lane group");
  constexpr size_t ROW_BYTES = size_t(K) * (FP8 ? 1 : 2);

  const int gl = threadIdx.x % LANES;
  const int row = (blockIdx.x * BLOCK + threadIdx.x) / LANES;
#if INGINE_ACT_FP16 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  if constexpr (FP8) {
    // Pre-Ada FP16-activation fast path (Volta sm_70 / Turing sm_75 / Pascal sm_60/61):
    // Zero-SMEM full-row weight prefetch + L1-cached __ldg activations + branchless 3-op e4m3x2_to_half2_m8
    // (* 256.f folded into row scale). Maximizes active waves and eliminates SMEM carveout tail waves.
    if (row >= rows) return;
    const int in = epi.input_of(row);
    if (in < 0) return;
    const ActT* x = epi.x_of(in);
    const uint8_t* wr = w + size_t(row) * ROW_BYTES;
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
    uint4 wv[CH];
#pragma unroll
    for (int i = 0; i < CH; ++i) {
      wv[i] = l2::ld_stream<WEIGHT_L2>(wr + (size_t(i) * LANES + gl) * 16, wpol);
    }
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < CH; ++i) {
      const int c16 = i * LANES + gl;
      float wf[16], xf[16];
      unpack_e4m3x16_m8(wv[i], wf);
      unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(x + c16 * 16)), xf);
      unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(x + c16 * 16 + 8)), xf + 8);
#pragma unroll
      for (int j = 0; j < 16; ++j) acc = fmaf(wf[j], xf[j], acc);
    }
    float v[1] = {group_sum<LANES>(acc) * (__ldg(w_scale + row) * 256.f)};
    if constexpr (Epi::kScatter) {
      if (gl == 0) epi.store1(row, 0, v[0]);
    } else {
      epi.store(row, v, gl, group_mask<LANES>());
    }
    return;
  }
#endif
  // A whole group shares `row`, so groups exit together and the shuffles below stay within live groups.
  if (row >= rows) return;
  const int in = epi.input_of(row);
  if (in < 0) return;
  const ActT* x = epi.x_of(in);  // pointer arithmetic only: the segment table is a constant

  const uint8_t* wr = w + size_t(row) * ROW_BYTES;
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  uint4 wv[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) wv[i] = l2::ld_stream<WEIGHT_L2>(wr + (size_t(i) * LANES + gl) * 16, wpol);
  pdl_wait_fenced(zero);  // the whole row is in flight; the activation comes from an earlier kernel
  pdl_trigger();

  float acc = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = (i * LANES + gl) * EPC;
    float wf[EPC], xf[EPC];
    if constexpr (FP8) unpack_e4m3x16(wv[i], wf);
    else unpack_bf16x8(wv[i], wf);
    unpack_bf16x8(ld_act(x + col), xf);
    if constexpr (FP8) unpack_bf16x8(ld_act(x + col + 8), xf + 8);
#pragma unroll
    for (int j = 0; j < EPC; ++j) acc = fmaf(wf[j], xf[j], acc);
  }

  float v[1] = {group_sum<LANES>(acc) * (FP8 ? __ldg(w_scale + row) : 1.f)};
  if constexpr (Epi::kScatter) {
    if (gl == 0) epi.store1(row, 0, v[0]);
  } else {
    epi.store(row, v, gl, group_mask<LANES>());
  }
}

// ---- M >= 2: tensor cores -----------------------------------------------------------------------------

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  mma::mma_act(c, a0, a1, a2, a3, b0, b1);
}

// 2 E4M3 (low 16 bits of u) -> Act2T. On pre-Ada FP16 builds (sm_60/61/70/75), use exact 3-op E4M3 * 2^-8 -> FP16x2
// (adapted from ashalliants/ninfer-3090 @ 8ea67ce, Apache-2.0) and fold * 256.f into the epilogue scale.
__device__ __forceinline__ uint32_t e4m3x2_to_mma_act2(uint32_t u) {
#if INGINE_ACT_FP16 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  return e4m3x2_to_half2_m8(u);
#elif INGINE_ACT_FP16
  return act::e4m3x2_to_half2_bits(u);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && defined(__CUDACC_VER_MAJOR__) && __CUDACC_VER_MAJOR__ >= 13
  uint32_t r;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<unsigned short>(u)));
  return r;
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(static_cast<unsigned short>(u)));
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
  const __nv_bfloat162 b = __float22bfloat162_rn(f);
  return *reinterpret_cast<const uint32_t*>(&b);
#else
  return e4m3_to_bf16_bits_portable(u & 0xffu) | (e4m3_to_bf16_bits_portable((u >> 8) & 0xffu) << 16);
#endif
}

#ifndef MMA_U_BF16
#define MMA_U_BF16 6
#endif
#ifndef MMA_U_FP8
#define MMA_U_FP8 3
#endif
// `zero` is 0 at run time (opaque to the compiler): every mma of a batch takes an operand xor-ed with
// (xor of all the batch's loads) & zero, so ptxas cannot interleave the mma with the batch's loads and
// all U steps' loads of a warp are in flight together (otherwise it issues one step's loads at a time,
// ~1 KB per warp, and scattered 64-byte requests from 16 rows run at ~40% of DRAM bandwidth).
template <bool FP8, int K, int M, int SPLITK, int U, class Epi>
__global__ void __launch_bounds__(32 * SPLITK) gemv_mma_kernel(const uint8_t* __restrict__ w,
                                                              const float* __restrict__ w_scale, int rows,
                                                              uint32_t zero, Epi epi) {
  constexpr int CPL = FP8 ? 16 : 8;  // columns per lane per 16-byte weight load
  constexpr int SPAN = 4 * CPL;      // columns per step (4 lanes t)
  constexpr int KS = K / SPLITK;     // columns per warp
  static_assert(KS % (SPAN * U) == 0, "K slice must split into unrolled steps");
  constexpr int NJ = KS / SPAN;
  constexpr size_t ROW_BYTES = size_t(K) * (FP8 ? 1 : 2);
  __shared__ float cs[SPLITK][16][8];

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16;
  const int ra = r0 + g, rb = r0 + g + 8;
  const int ia = ra < rows ? epi.input_of(ra) : -1, ib = rb < rows ? epi.input_of(rb) : -1;
  const int c0 = warp * KS + t * CPL;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
  // Native Volta sm_70 HMMA.884: lane (qp, idx) owns ONE row rv = r0 + (qp>=2 ? 8 : 0) + idx and
  // TWO contiguous chunks (2*CPL cols) per SPAN step, feeding sm70_mma_f16_direct with ZERO shuffles!
  const int qp = (lane >> 2) & 3, idx = (lane & 3) | ((lane & 16) >> 2);
  const int rv = r0 + (qp >= 2 ? 8 : 0) + idx;
  const int iv = rv < rows ? epi.input_of(rv) : -1;
  const int c0_v = warp * KS + (qp & 1) * (2 * CPL);
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  float c8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
  float acc_a[M], acc_b[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc_a[m] = acc_b[m] = 0.f;
#else
  float c[4] = {0.f, 0.f, 0.f, 0.f};
#endif
  bool waited = false;
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  auto pass = [&](int cur) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
    const bool lv = iv == cur, lx = idx < M;
    const uint8_t* wv = w + size_t(lv ? rv : 0) * ROW_BYTES;
    const ActT* xg = epi.x_of(cur) + size_t(lx ? idx : 0) * K;
#pragma unroll 1
    for (int jb = 0; jb < NJ; jb += U) {
      uint4 W0[U], W1[U];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const size_t off0 = size_t(c0_v + (jb + u) * SPAN) * (FP8 ? 1 : 2);
        W0[u] = lv ? l2::ld_stream<WEIGHT_L2>(wv + off0, wpol) : make_uint4(0, 0, 0, 0);
        W1[u] = lv ? l2::ld_stream<WEIGHT_L2>(wv + off0 + 16, wpol) : make_uint4(0, 0, 0, 0);
      }
      if (!waited) {
        pdl_wait_fenced(zero);
        pdl_trigger();
        waited = true;
      }
      uint32_t dep = 0;
#pragma unroll
      for (int u = 0; u < U; ++u) dep ^= W0[u].x ^ W0[u].w ^ W1[u].x ^ W1[u].w;
      dep &= zero;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int col0 = c0_v + (jb + u) * SPAN;
        if constexpr (FP8) {
          const uint4 x0 = lx ? ld_act(xg + col0) : make_uint4(0, 0, 0, 0);
          const uint4 x1 = lx ? ld_act(xg + col0 + 8) : make_uint4(0, 0, 0, 0);
          const uint4 x2 = lx ? ld_act(xg + col0 + 16) : make_uint4(0, 0, 0, 0);
          const uint4 x3 = lx ? ld_act(xg + col0 + 24) : make_uint4(0, 0, 0, 0);
          mma::sm70_mma_f16_direct(c8, e4m3x2_to_mma_act2(W0[u].x), e4m3x2_to_mma_act2(W0[u].x >> 16),
                                   e4m3x2_to_mma_act2(W0[u].y), e4m3x2_to_mma_act2(W0[u].y >> 16),
                                   x0.x ^ dep, x0.y, x0.z, x0.w);
          mma::sm70_mma_f16_direct(c8, e4m3x2_to_mma_act2(W0[u].z), e4m3x2_to_mma_act2(W0[u].z >> 16),
                                   e4m3x2_to_mma_act2(W0[u].w), e4m3x2_to_mma_act2(W0[u].w >> 16),
                                   x1.x, x1.y, x1.z, x1.w);
          mma::sm70_mma_f16_direct(c8, e4m3x2_to_mma_act2(W1[u].x), e4m3x2_to_mma_act2(W1[u].x >> 16),
                                   e4m3x2_to_mma_act2(W1[u].y), e4m3x2_to_mma_act2(W1[u].y >> 16),
                                   x2.x, x2.y, x2.z, x2.w);
          mma::sm70_mma_f16_direct(c8, e4m3x2_to_mma_act2(W1[u].z), e4m3x2_to_mma_act2(W1[u].z >> 16),
                                   e4m3x2_to_mma_act2(W1[u].w), e4m3x2_to_mma_act2(W1[u].w >> 16),
                                   x3.x, x3.y, x3.z, x3.w);
        } else {
          const uint4 x0 = lx ? ld_act(xg + col0) : make_uint4(0, 0, 0, 0);
          const uint4 x1 = lx ? ld_act(xg + col0 + 8) : make_uint4(0, 0, 0, 0);
          mma::sm70_mma_f16_direct(c8, W0[u].x, W0[u].y, W0[u].z, W0[u].w, x0.x ^ dep, x0.y, x0.z, x0.w);
          mma::sm70_mma_f16_direct(c8, W1[u].x, W1[u].y, W1[u].z, W1[u].w, x1.x, x1.y, x1.z, x1.w);
        }
      }
    }
#else
    const bool la = ia == cur, lb = ib == cur, lx = g < M;
    const uint8_t* wa = w + size_t(la ? ra : 0) * ROW_BYTES;
    const uint8_t* wb = w + size_t(lb ? rb : 0) * ROW_BYTES;
    const ActT* xg = epi.x_of(cur) + size_t(lx ? g : 0) * K;
#pragma unroll 1
    for (int jb = 0; jb < NJ; jb += U) {
      uint4 A[U], B[U], X[U][CPL / 8];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const size_t off = size_t(c0 + (jb + u) * SPAN) * (FP8 ? 1 : 2);
        A[u] = la ? l2::ld_stream<WEIGHT_L2>(wa + off, wpol) : make_uint4(0, 0, 0, 0);
        B[u] = lb ? l2::ld_stream<WEIGHT_L2>(wb + off, wpol) : make_uint4(0, 0, 0, 0);
      }
      if (!waited) {  // the first batch of weights is in flight; activations come from earlier kernels
        pdl_wait_fenced(zero);
        pdl_trigger();
        waited = true;
      }
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const int col = c0 + (jb + u) * SPAN;
#pragma unroll
        for (int h = 0; h < CPL / 8; ++h) X[u][h] = lx ? ld_act(xg + col + 8 * h) : make_uint4(0, 0, 0, 0);
      }
      uint32_t dep = 0;
#pragma unroll
      for (int u = 0; u < U; ++u) {
        dep ^= A[u].x ^ A[u].w ^ B[u].x ^ B[u].w;
#pragma unroll
        for (int h = 0; h < CPL / 8; ++h) dep ^= X[u][h].x;
      }
      dep &= zero;
#pragma unroll
      for (int u = 0; u < U; ++u) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
        // Pascal sm_60/sm_61: compute ONLY active columns 0..M-1 with hoisted weight unpacks and exact FP32 fmaf.
        if constexpr (FP8) {
          const uint32_t aw[4] = {A[u].x, A[u].y, A[u].z, A[u].w}, bw[4] = {B[u].x, B[u].y, B[u].z, B[u].w};
          const uint32_t xw[8] = {X[u][0].x ^ dep, X[u][0].y, X[u][0].z, X[u][0].w,
                                  X[u][1].x,       X[u][1].y, X[u][1].z, X[u][1].w};
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const float2 fa0 = act::unpack2(e4m3x2_to_mma_act2(aw[s]));
            const float2 fa1 = act::unpack2(e4m3x2_to_mma_act2(aw[s] >> 16));
            const float2 fb0 = act::unpack2(e4m3x2_to_mma_act2(bw[s]));
            const float2 fb1 = act::unpack2(e4m3x2_to_mma_act2(bw[s] >> 16));
#pragma unroll
            for (int m = 0; m < M; ++m) {
              const int sl = 4 * m + t;
              const float2 fx0 = act::unpack2(__shfl_sync(0xffffffffu, xw[2 * s], sl));
              const float2 fx1 = act::unpack2(__shfl_sync(0xffffffffu, xw[2 * s + 1], sl));
              acc_a[m] = fmaf(fa1.y, fx1.y, fmaf(fa1.x, fx1.x, fmaf(fa0.y, fx0.y, fmaf(fa0.x, fx0.x, acc_a[m]))));
              acc_b[m] = fmaf(fb1.y, fx1.y, fmaf(fb1.x, fx1.x, fmaf(fb0.y, fx0.y, fmaf(fb0.x, fx0.x, acc_b[m]))));
            }
          }
        } else {
          const uint32_t ah[4] = {A[u].x, A[u].y, A[u].z, A[u].w}, bh[4] = {B[u].x, B[u].y, B[u].z, B[u].w};
          const uint32_t xw[4] = {X[u][0].x ^ dep, X[u][0].y, X[u][0].z, X[u][0].w};
#pragma unroll
          for (int s = 0; s < 4; ++s) {
            const float2 fa = act::unpack2(ah[s]);
            const float2 fb = act::unpack2(bh[s]);
#pragma unroll
            for (int m = 0; m < M; ++m) {
              const int sl = 4 * m + t;
              const float2 fx = act::unpack2(__shfl_sync(0xffffffffu, xw[s], sl));
              acc_a[m] = fmaf(fa.y, fx.y, fmaf(fa.x, fx.x, acc_a[m]));
              acc_b[m] = fmaf(fb.y, fx.y, fmaf(fb.x, fx.x, acc_b[m]));
            }
          }
        }
#else
        if constexpr (FP8) {
          // 16 columns = 4 k-steps of 4 columns (2 k-pairs) each
          const uint32_t aw[4] = {A[u].x, A[u].y, A[u].z, A[u].w}, bw[4] = {B[u].x, B[u].y, B[u].z, B[u].w};
          const uint32_t xw[8] = {X[u][0].x, X[u][0].y, X[u][0].z, X[u][0].w,
                                  X[u][1].x, X[u][1].y, X[u][1].z, X[u][1].w};
#pragma unroll
          for (int s = 0; s < 4; ++s)
            mma_bf16(c, e4m3x2_to_mma_act2(aw[s]), e4m3x2_to_mma_act2(bw[s]), e4m3x2_to_mma_act2(aw[s] >> 16),
                     e4m3x2_to_mma_act2(bw[s] >> 16), xw[2 * s] ^ dep, xw[2 * s + 1]);
        } else {
          mma_bf16(c, A[u].x, B[u].x, A[u].y, B[u].y, X[u][0].x ^ dep, X[u][0].y);
          mma_bf16(c, A[u].z, B[u].z, A[u].w, B[u].w, X[u][0].z ^ dep, X[u][0].w);
        }
#endif
      }
    }
#endif
  };
  if constexpr (Epi::kInputs == 1) {
    pass(0);
  } else {
    // inputs present among the block's 16 rows (warp-uniform; every warp sees the same rows)
#pragma unroll 1
    for (int i = 0; i < Epi::kInputs; ++i)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
      if (__any_sync(0xffffffffu, iv == i)) pass(i);
#else
      if (__any_sync(0xffffffffu, ia == i || ib == i)) pass(i);
#endif
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
  mma::sm70_reduce_acc8_to_d4(c8, c);
  cs[warp][g][2 * t] = c[0];
  cs[warp][g][2 * t + 1] = c[1];
  cs[warp][g + 8][2 * t] = c[2];
  cs[warp][g + 8][2 * t + 1] = c[3];
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
#pragma unroll
  for (int m = 0; m < M; ++m) {
    acc_a[m] += __shfl_xor_sync(0xffffffffu, acc_a[m], 1);
    acc_a[m] += __shfl_xor_sync(0xffffffffu, acc_a[m], 2);
    acc_b[m] += __shfl_xor_sync(0xffffffffu, acc_b[m], 1);
    acc_b[m] += __shfl_xor_sync(0xffffffffu, acc_b[m], 2);
    if (t == 0) {
      cs[warp][g][m] = acc_a[m];
      cs[warp][g + 8][m] = acc_b[m];
    }
  }
#else
  // C fragment: c0, c1 = rows g, tokens 2t, 2t+1; c2, c3 = row g + 8
  cs[warp][g][2 * t] = c[0];
  cs[warp][g][2 * t + 1] = c[1];
  cs[warp][g + 8][2 * t] = c[2];
  cs[warp][g + 8][2 * t + 1] = c[3];
#endif
  __syncthreads();
  auto total = [&](int r, int m) {
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < SPLITK; ++k) s += cs[k][r][m];
    return s;
  };
#if INGINE_ACT_FP16 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  constexpr float kFp8ScaleMul = FP8 ? 256.f : 1.f;
#else
  constexpr float kFp8ScaleMul = 1.f;
#endif
  if constexpr (Epi::kScatter) {
    for (int k = threadIdx.x; k < 16 * M; k += 32 * SPLITK) {
      const int r = k / M, m = k % M, row = r0 + r;
      if (row < rows && epi.input_of(row) >= 0)
        epi.store1(row, m, total(r, m) * (FP8 ? (__ldg(w_scale + row) * kFp8ScaleMul) : 1.f));
    }
  } else {
    const int r = threadIdx.x, row = r0 + r;
    if (r < 16 && row < rows) {
      const float scale = FP8 ? (__ldg(w_scale + row) * kFp8ScaleMul) : 1.f;
      float y[M];
#pragma unroll
      for (int m = 0; m < M; ++m) y[m] = total(r, m) * scale;
      epi.store_row(row, y);
    }
  }
}

// ---- Q5_K streaming GEMV (M = 1..8, K % 256 == 0, exact GGML BlockQ5K 176 B / 256 elems) --------------

__device__ __forceinline__ void q5k_get_scale_min(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
  if (j < 4) {
    sc = q[j] & 63u;
    m = q[j + 4] & 63u;
  } else {
    sc = static_cast<uint8_t>((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
    m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

template <int K, int M, class Epi>
__global__ void __launch_bounds__(128) gemv_q5_k_kernel(const uint8_t* __restrict__ w, int rows, uint32_t zero,
                                                        Epi epi) {
  static_assert(K % q5k::kBlockElems == 0, "Q5_K GEMV requires K divisible by 256");
  constexpr int NB = K / q5k::kBlockElems;
  constexpr size_t ROW_BYTES = size_t(NB) * q5k::kBlockBytes;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 4 + warp;
  if (row >= rows) return;
  const int in = epi.input_of(row);
  if (in < 0) return;

  const ActT* x = epi.x_of(in);
  const uint8_t* wr = w + size_t(row) * ROW_BYTES;
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();

  pdl_wait_fenced(zero);
  pdl_trigger();

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

#pragma unroll 1
  for (int b = 0; b < NB; ++b) {
    const uint8_t* blk = wr + size_t(b) * q5k::kBlockBytes;
    const uint16_t d_bits = *reinterpret_cast<const uint16_t*>(blk);
    const uint16_t dm_bits = *reinterpret_cast<const uint16_t*>(blk + 2);
    const float d = __half2float(__ushort_as_half(d_bits));
    const float dmin = __half2float(__ushort_as_half(dm_bits));
    uint8_t sc_raw[12];
#pragma unroll
    for (int i = 0; i < 12; ++i) sc_raw[i] = __ldg(blk + 4 + i);
    const uint8_t qh = __ldg(blk + 16 + lane);
    const uint8_t* qs = blk + 48;
    const int base = b * q5k::kBlockElems;
    (void)wpol;

#pragma unroll
    for (int c = 0; c < 4; ++c) {
      uint8_t sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
      q5k_get_scale_min(2 * c, sc_raw, sc0, m0);
      q5k_get_scale_min(2 * c + 1, sc_raw, sc1, m1);
      const float d0 = d * float(sc0);
      const float dm0 = dmin * float(m0);
      const float d1 = d * float(sc1);
      const float dm1 = dmin * float(m1);
      const uint8_t ql = __ldg(qs + 32 * c + lane);
      const int q0 = int(ql & 0x0Fu) + ((qh & (1u << (2 * c))) ? 16 : 0);
      const int q1 = int(ql >> 4) + ((qh & (1u << (2 * c + 1))) ? 16 : 0);
      const float w0 = fmaf(d0, float(q0), -dm0);
      const float w1 = fmaf(d1, float(q1), -dm1);
      const int col0 = base + 64 * c + lane;
      const int col1 = col0 + 32;
#pragma unroll
      for (int m = 0; m < M; ++m) {
        const ActT* xm = x + size_t(m) * K;
        const float x0 = act::to_float(__ldg(xm + col0));
        const float x1 = act::to_float(__ldg(xm + col1));
        acc[m] = fmaf(w0, x0, acc[m]);
        acc[m] = fmaf(w1, x1, acc[m]);
      }
    }
  }

#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = group_sum<32>(acc[m]);

  if constexpr (Epi::kScatter) {
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < M; ++m) epi.store1(row, m, acc[m]);
    }
  } else {
    epi.store(row, acc, lane, 0xffffffffu);
  }
}

// ---- launch helpers -----------------------------------------------------------------------------------

// BLOCK: threads per block of the M = 1 kernel (0: M = 1 runs the tensor-core kernel too, for K whose
// whole row per lane would not fit in registers); SPLITK: warps per 16-row block of the M >= 2 kernel;
// UFP8: steps per load batch of its FP8 instantiation.
template <int K, int BLOCK, int SPLITK, int UFP8, template <int> class Epi, int UBF16 = MMA_U_BF16, class Make>
void launch_gemv(const Matrix& w, int M, bool pdl, cudaStream_t stream, Make make) {
  const bool fp8 = w.scale != nullptr;
  require(w.w != nullptr && w.rows > 0, "weight matrix is empty");
  require(!(w.q5_k && fp8), "weight matrix cannot be both Q5_K and FP8");
  const int rows = w.rows;
  const float* sc = w.scale;
  const uint8_t* wp = static_cast<const uint8_t*>(w.w);
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    auto epi = make(std::integral_constant<int, MM>{});
    using E = decltype(epi);
    if (w.q5_k) {
      if constexpr (K % q5k::kBlockElems == 0) {
        launch_ex(gemv_q5_k_kernel<K, MM, E>, dim3((rows + 3) / 4), 128, 0, 1, pdl, stream, wp, rows, 0u, epi);
      } else {
        require(false, "Q5_K GEMV requires K divisible by 256");
      }
      return;
    }
    if constexpr (MM == 1 && (BLOCK > 0 || kActIsFP16)) {
      constexpr int B = BLOCK > 0 ? BLOCK : 32;  // (the branch is discarded for BLOCK = 0)
      if (fp8) {
        if constexpr (kActIsFP16 && K == 3072 && B == 32) {
          // Zero-SMEM 32-lane (CH=6) 128-thread blocks: 576 blocks across 80 SMs, full 100% occupancy (~32 regs).
          constexpr int B128 = 128;
          launch_ex(gemv_kernel<true, K, 32, B128, E>, dim3((rows * 32 + B128 - 1) / B128), B128, 0, 1, pdl, stream,
                    wp, sc, rows, 0u, epi);
        } else if constexpr (kActIsFP16 && K == 2304 && B == 256) {
          // Zero-SMEM 16-lane (CH=9) 128-thread blocks: 1076 blocks for proj (13.5/SM), full L1 activation hit.
          constexpr int B128 = 128;
          launch_ex(gemv_kernel<true, K, 16, B128, E>, dim3((rows * 16 + B128 - 1) / B128), B128, 0, 1, pdl, stream,
                    wp, sc, rows, 0u, epi);
        } else {
          launch_ex(gemv_kernel<true, K, 16, B, E>, dim3((rows * 16 + B - 1) / B), B, 0, 1, pdl, stream, wp, sc, rows,
                    0u, epi);
        }
      } else {
        launch_ex(gemv_kernel<false, K, 32, B, E>, dim3((rows * 32 + B - 1) / B), B, 0, 1, pdl, stream, wp, sc, rows,
                  0u, epi);
      }
    } else {
      const dim3 blocks((rows + 15) / 16);
      if (fp8) launch_ex(gemv_mma_kernel<true, K, MM, SPLITK, UFP8, E>, blocks, 32 * SPLITK, 0, 1, pdl, stream, wp, sc, rows, 0u, epi);
      else launch_ex(gemv_mma_kernel<false, K, MM, SPLITK, UBF16, E>, blocks, 32 * SPLITK, 0, 1, pdl, stream, wp, sc, rows, 0u, epi);
    }
  };
  switch (M) {
    case 1: go(std::integral_constant<int, 1>{}); break;
    case 2: go(std::integral_constant<int, 2>{}); break;
    case 3: go(std::integral_constant<int, 3>{}); break;
    case 4: go(std::integral_constant<int, 4>{}); break;
    case 5: go(std::integral_constant<int, 5>{}); break;
    case 6: go(std::integral_constant<int, 6>{}); break;
    case 7: go(std::integral_constant<int, 7>{}); break;
    case 8: go(std::integral_constant<int, 8>{}); break;
    default: require(false, "M must be in 1..8");
  }
}

// Warps per 16-row block of the M >= 2 kernel (K slice per warp must hold a whole number of U-step
// batches). Measured at M = 2 and 8 (shared GPU): proj 6 > 4 > 2 (FP8 +4-8%), head 6 ~ 4 >> 2 >> 1
// (+20-30%: 12.5K blocks need short K slices to keep enough rows in flight), proj_conv 4 ~ 8 ~ 16.
#ifndef PROJ_SPLITK
#define PROJ_SPLITK 6
#endif
#ifndef CONV_SPLITK
#define CONV_SPLITK 8
#endif
// FP8 steps per load batch of proj_conv's M >= 2 kernel: 6 = a warp's whole K slice. Its 144 blocks are all
// resident at once (2 per SM at 120 registers), so every weight byte is requested before the PDL wait, as in
// the M = 1 kernel (measured against 3 steps: 5.9 -> 4.9 us per call at M = 2, 6.6 -> 6.0 at M = 8). proj
// (538 blocks) loses with 6: half its blocks would wait for a slot (14.0 -> 14.5 us at M = 2).
#ifndef CONV_U_FP8
#define CONV_U_FP8 6
#endif
#ifndef HEAD_SPLITK
#define HEAD_SPLITK 6
#endif

__global__ void fp8_rows_kernel(const ActT* __restrict__ w, int rows, int K, __nv_fp8_storage_t* q,
                                float* scale) {
  const int lane = threadIdx.x & 31;
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (row >= rows) return;
  const ActT* wr = w + size_t(row) * K;
  float amax = 0.f;
  for (int k = lane; k < K; k += 32) amax = fmaxf(amax, fabsf(act::to_float(wr[k])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float s = amax / 448.f;
  if (lane == 0) scale[row] = s;
  for (int k = lane; k < K; k += 32)
    q[size_t(row) * K + k] =
        s > 0.f ? __nv_cvt_float_to_fp8(act::to_float(wr[k]) / s, __NV_SATFINITE, __NV_E4M3) : 0;
}

// weights.dequant: bf16(float(q) * scale[row]); one thread per 16 elements (K % 16 == 0).
__global__ void dequant_rows_kernel(const __nv_fp8_storage_t* __restrict__ q, const float* __restrict__ scale,
                                    int K, long long total16, ActT* __restrict__ out) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= total16) return;
  const long long e = i * 16;
  const float s = scale[e / K];
  const uint4 u = *reinterpret_cast<const uint4*>(q + e);
  float f[16];
  unpack_e4m3x16(u, f);
  __align__(16) ActT v[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) v[j] = act::from_float(f[j] * s);
  uint4* dst = reinterpret_cast<uint4*>(out + e);
  dst[0] = reinterpret_cast<const uint4*>(v)[0];
  dst[1] = reinterpret_cast<const uint4*>(v)[1];
}

// ---- Fused single-pass dual-vector lm_head GEMV + LoopCD adaptive contrastive combine -----------------

template <bool FP8, int K, int LANES, int BLOCK>
__global__ void __launch_bounds__(BLOCK) head_dual_gemv_kernel(const uint8_t* __restrict__ w,
                                                               const float* __restrict__ w_scale, int rows,
                                                               const ActT* __restrict__ x_strong,
                                                               const ActT* __restrict__ x_weak,
                                                               float* __restrict__ out_strong,
                                                               float* __restrict__ out_weak, uint32_t zero) {
  constexpr int EPC = FP8 ? 16 : 8;
  constexpr int CH = K / (EPC * LANES);
  static_assert(K % (EPC * LANES) == 0, "row must split evenly over the lane group");
  constexpr size_t ROW_BYTES = size_t(K) * (FP8 ? 1 : 2);

  const int gl = threadIdx.x % LANES;
  const int row = (blockIdx.x * BLOCK + threadIdx.x) / LANES;
  if (row >= rows) return;
  const uint8_t* wr = w + size_t(row) * ROW_BYTES;
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();

#if INGINE_ACT_FP16 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  if constexpr (FP8) {
    uint4 wv[CH];
#pragma unroll
    for (int i = 0; i < CH; ++i) {
      wv[i] = l2::ld_stream<WEIGHT_L2>(wr + (size_t(i) * LANES + gl) * 16, wpol);
    }
    float acc_s = 0.f, acc_w = 0.f;
#pragma unroll
    for (int i = 0; i < CH; ++i) {
      const int c16 = i * LANES + gl;
      float wf[16], xs[16], xw[16];
      unpack_e4m3x16_m8(wv[i], wf);
      unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(x_strong + c16 * 16)), xs);
      unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(x_strong + c16 * 16 + 8)), xs + 8);
      unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(x_weak + c16 * 16)), xw);
      unpack_bf16x8(__ldg(reinterpret_cast<const uint4*>(x_weak + c16 * 16 + 8)), xw + 8);
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        acc_s = fmaf(wf[j], xs[j], acc_s);
        acc_w = fmaf(wf[j], xw[j], acc_w);
      }
    }
    const float sc = __ldg(w_scale + row) * 256.f;
    const float vs = group_sum<LANES>(acc_s) * sc;
    const float vw = group_sum<LANES>(acc_w) * sc;
    if (gl == 0) {
      out_strong[row] = tanhf(round_bf16(vs) / 30.f) * 30.f;
      out_weak[row] = tanhf(round_bf16(vw) / 30.f) * 30.f;
    }
    return;
  }
#endif

  uint4 wv[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) wv[i] = l2::ld_stream<WEIGHT_L2>(wr + (size_t(i) * LANES + gl) * 16, wpol);
  pdl_wait_fenced(zero);
  pdl_trigger();

  float acc_s = 0.f, acc_w = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = (i * LANES + gl) * EPC;
    float wf[EPC], xs[EPC], xw[EPC];
    if constexpr (FP8) unpack_e4m3x16(wv[i], wf);
    else unpack_bf16x8(wv[i], wf);
    unpack_bf16x8(ld_act(x_strong + col), xs);
    unpack_bf16x8(ld_act(x_weak + col), xw);
    if constexpr (FP8) {
      unpack_bf16x8(ld_act(x_strong + col + 8), xs + 8);
      unpack_bf16x8(ld_act(x_weak + col + 8), xw + 8);
    }
#pragma unroll
    for (int j = 0; j < EPC; ++j) {
      acc_s = fmaf(wf[j], xs[j], acc_s);
      acc_w = fmaf(wf[j], xw[j], acc_w);
    }
  }
  const float sc = FP8 ? __ldg(w_scale + row) : 1.f;
  const float vs = group_sum<LANES>(acc_s) * sc;
  const float vw = group_sum<LANES>(acc_w) * sc;
  if (gl == 0) {
    out_strong[row] = tanhf(round_bf16(vs) / 30.f) * 30.f;
    out_weak[row] = tanhf(round_bf16(vw) / 30.f) * 30.f;
  }
}

template <bool FP8, int K, int M, int SPLITK, int U>
__global__ void __launch_bounds__(32 * SPLITK) head_dual_mma_kernel(const uint8_t* __restrict__ w,
                                                                    const float* __restrict__ w_scale, int rows,
                                                                    const ActT* __restrict__ x_strong,
                                                                    const ActT* __restrict__ x_weak,
                                                                    float* __restrict__ out_strong,
                                                                    float* __restrict__ out_weak, uint32_t zero) {
  constexpr int CPL = FP8 ? 16 : 8;
  constexpr int SPAN = 4 * CPL;
  constexpr int KS = K / SPLITK;
  static_assert(KS % (SPAN * U) == 0, "K slice must split into unrolled steps");
  constexpr int NJ = KS / SPAN;
  constexpr size_t ROW_BYTES = size_t(K) * (FP8 ? 1 : 2);
  __shared__ float cs[SPLITK][16][8];

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int r0 = blockIdx.x * 16;
  const int ra = r0 + g, rb = r0 + g + 8;
  const int c0 = warp * KS + t * CPL;
  const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
  bool waited = false;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
  const int qp = (lane >> 2) & 3, idx = (lane & 3) | ((lane & 16) >> 2);
  const int rv = r0 + (qp >= 2 ? 8 : 0) + idx;
  const bool lv = rv < rows, lx = idx < M;
  const int c0_v = warp * KS + (qp & 1) * (2 * CPL);
  const uint8_t* wv = w + size_t(lv ? rv : 0) * ROW_BYTES;
  const ActT* xs_g = x_strong + size_t(lx ? idx : 0) * K;
  const ActT* xw_g = x_weak + size_t(lx ? idx : 0) * K;
  float c8_s[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float c8_w[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
  for (int jb = 0; jb < NJ; jb += U) {
    uint4 W0[U], W1[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const size_t off0 = size_t(c0_v + (jb + u) * SPAN) * (FP8 ? 1 : 2);
      W0[u] = lv ? l2::ld_stream<WEIGHT_L2>(wv + off0, wpol) : make_uint4(0, 0, 0, 0);
      W1[u] = lv ? l2::ld_stream<WEIGHT_L2>(wv + off0 + 16, wpol) : make_uint4(0, 0, 0, 0);
    }
    if (!waited) {
      pdl_wait_fenced(zero);
      pdl_trigger();
      waited = true;
    }
    uint32_t dep = 0;
#pragma unroll
    for (int u = 0; u < U; ++u) dep ^= W0[u].x ^ W0[u].w ^ W1[u].x ^ W1[u].w;
    dep &= zero;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int col0 = c0_v + (jb + u) * SPAN;
      if constexpr (FP8) {
        const uint32_t a00 = e4m3x2_to_mma_act2(W0[u].x), a01 = e4m3x2_to_mma_act2(W0[u].x >> 16);
        const uint32_t a02 = e4m3x2_to_mma_act2(W0[u].y), a03 = e4m3x2_to_mma_act2(W0[u].y >> 16);
        const uint32_t a10 = e4m3x2_to_mma_act2(W0[u].z), a11 = e4m3x2_to_mma_act2(W0[u].z >> 16);
        const uint32_t a12 = e4m3x2_to_mma_act2(W0[u].w), a13 = e4m3x2_to_mma_act2(W0[u].w >> 16);
        const uint32_t a20 = e4m3x2_to_mma_act2(W1[u].x), a21 = e4m3x2_to_mma_act2(W1[u].x >> 16);
        const uint32_t a22 = e4m3x2_to_mma_act2(W1[u].y), a23 = e4m3x2_to_mma_act2(W1[u].y >> 16);
        const uint32_t a30 = e4m3x2_to_mma_act2(W1[u].z), a31 = e4m3x2_to_mma_act2(W1[u].z >> 16);
        const uint32_t a32 = e4m3x2_to_mma_act2(W1[u].w), a33 = e4m3x2_to_mma_act2(W1[u].w >> 16);
        const uint4 s0 = lx ? ld_act(xs_g + col0) : make_uint4(0, 0, 0, 0);
        const uint4 s1 = lx ? ld_act(xs_g + col0 + 8) : make_uint4(0, 0, 0, 0);
        const uint4 s2 = lx ? ld_act(xs_g + col0 + 16) : make_uint4(0, 0, 0, 0);
        const uint4 s3 = lx ? ld_act(xs_g + col0 + 24) : make_uint4(0, 0, 0, 0);
        const uint4 w0 = lx ? ld_act(xw_g + col0) : make_uint4(0, 0, 0, 0);
        const uint4 w1 = lx ? ld_act(xw_g + col0 + 8) : make_uint4(0, 0, 0, 0);
        const uint4 w2 = lx ? ld_act(xw_g + col0 + 16) : make_uint4(0, 0, 0, 0);
        const uint4 w3 = lx ? ld_act(xw_g + col0 + 24) : make_uint4(0, 0, 0, 0);
        mma::sm70_mma_f16_direct(c8_s, a00, a01, a02, a03, s0.x ^ dep, s0.y, s0.z, s0.w);
        mma::sm70_mma_f16_direct(c8_s, a10, a11, a12, a13, s1.x, s1.y, s1.z, s1.w);
        mma::sm70_mma_f16_direct(c8_s, a20, a21, a22, a23, s2.x, s2.y, s2.z, s2.w);
        mma::sm70_mma_f16_direct(c8_s, a30, a31, a32, a33, s3.x, s3.y, s3.z, s3.w);
        mma::sm70_mma_f16_direct(c8_w, a00, a01, a02, a03, w0.x ^ dep, w0.y, w0.z, w0.w);
        mma::sm70_mma_f16_direct(c8_w, a10, a11, a12, a13, w1.x, w1.y, w1.z, w1.w);
        mma::sm70_mma_f16_direct(c8_w, a20, a21, a22, a23, w2.x, w2.y, w2.z, w2.w);
        mma::sm70_mma_f16_direct(c8_w, a30, a31, a32, a33, w3.x, w3.y, w3.z, w3.w);
      } else {
        const uint4 s0 = lx ? ld_act(xs_g + col0) : make_uint4(0, 0, 0, 0);
        const uint4 s1 = lx ? ld_act(xs_g + col0 + 8) : make_uint4(0, 0, 0, 0);
        const uint4 w0 = lx ? ld_act(xw_g + col0) : make_uint4(0, 0, 0, 0);
        const uint4 w1 = lx ? ld_act(xw_g + col0 + 8) : make_uint4(0, 0, 0, 0);
        mma::sm70_mma_f16_direct(c8_s, W0[u].x, W0[u].y, W0[u].z, W0[u].w, s0.x ^ dep, s0.y, s0.z, s0.w);
        mma::sm70_mma_f16_direct(c8_s, W1[u].x, W1[u].y, W1[u].z, W1[u].w, s1.x, s1.y, s1.z, s1.w);
        mma::sm70_mma_f16_direct(c8_w, W0[u].x, W0[u].y, W0[u].z, W0[u].w, w0.x ^ dep, w0.y, w0.z, w0.w);
        mma::sm70_mma_f16_direct(c8_w, W1[u].x, W1[u].y, W1[u].z, W1[u].w, w1.x, w1.y, w1.z, w1.w);
      }
    }
  }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
  const bool la = ra < rows, lb = rb < rows, lx = g < M;
  const uint8_t* wa = w + size_t(la ? ra : 0) * ROW_BYTES;
  const uint8_t* wb = w + size_t(lb ? rb : 0) * ROW_BYTES;
  const ActT* xs_g = x_strong + size_t(lx ? g : 0) * K;
  const ActT* xw_g = x_weak + size_t(lx ? g : 0) * K;
  float acc_as[M], acc_bs[M], acc_aw[M], acc_bw[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc_as[m] = acc_bs[m] = acc_aw[m] = acc_bw[m] = 0.f;
#pragma unroll 1
  for (int jb = 0; jb < NJ; jb += U) {
    uint4 A[U], B[U], XS[U][CPL / 8], XW[U][CPL / 8];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const size_t off = size_t(c0 + (jb + u) * SPAN) * (FP8 ? 1 : 2);
      A[u] = la ? l2::ld_stream<WEIGHT_L2>(wa + off, wpol) : make_uint4(0, 0, 0, 0);
      B[u] = lb ? l2::ld_stream<WEIGHT_L2>(wb + off, wpol) : make_uint4(0, 0, 0, 0);
    }
    if (!waited) {
      pdl_wait_fenced(zero);
      pdl_trigger();
      waited = true;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int col = c0 + (jb + u) * SPAN;
#pragma unroll
      for (int h = 0; h < CPL / 8; ++h) {
        XS[u][h] = lx ? ld_act(xs_g + col + 8 * h) : make_uint4(0, 0, 0, 0);
        XW[u][h] = lx ? ld_act(xw_g + col + 8 * h) : make_uint4(0, 0, 0, 0);
      }
    }
    uint32_t dep = 0;
#pragma unroll
    for (int u = 0; u < U; ++u) dep ^= A[u].x ^ A[u].w ^ B[u].x ^ B[u].w;
    dep &= zero;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if constexpr (FP8) {
        const uint32_t aw[4] = {A[u].x, A[u].y, A[u].z, A[u].w}, bw[4] = {B[u].x, B[u].y, B[u].z, B[u].w};
        const uint32_t sw[8] = {XS[u][0].x ^ dep, XS[u][0].y, XS[u][0].z, XS[u][0].w,
                                XS[u][1].x,       XS[u][1].y, XS[u][1].z, XS[u][1].w};
        const uint32_t ww[8] = {XW[u][0].x ^ dep, XW[u][0].y, XW[u][0].z, XW[u][0].w,
                                XW[u][1].x,       XW[u][1].y, XW[u][1].z, XW[u][1].w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const float2 fa0 = act::unpack2(e4m3x2_to_mma_act2(aw[s]));
          const float2 fa1 = act::unpack2(e4m3x2_to_mma_act2(aw[s] >> 16));
          const float2 fb0 = act::unpack2(e4m3x2_to_mma_act2(bw[s]));
          const float2 fb1 = act::unpack2(e4m3x2_to_mma_act2(bw[s] >> 16));
#pragma unroll
          for (int m = 0; m < M; ++m) {
            const int sl = 4 * m + t;
            const float2 fs0 = act::unpack2(__shfl_sync(0xffffffffu, sw[2 * s], sl));
            const float2 fs1 = act::unpack2(__shfl_sync(0xffffffffu, sw[2 * s + 1], sl));
            acc_as[m] = fmaf(fa1.y, fs1.y, fmaf(fa1.x, fs1.x, fmaf(fa0.y, fs0.y, fmaf(fa0.x, fs0.x, acc_as[m]))));
            acc_bs[m] = fmaf(fb1.y, fs1.y, fmaf(fb1.x, fs1.x, fmaf(fb0.y, fs0.y, fmaf(fb0.x, fs0.x, acc_bs[m]))));
            const float2 fw0 = act::unpack2(__shfl_sync(0xffffffffu, ww[2 * s], sl));
            const float2 fw1 = act::unpack2(__shfl_sync(0xffffffffu, ww[2 * s + 1], sl));
            acc_aw[m] = fmaf(fa1.y, fw1.y, fmaf(fa1.x, fw1.x, fmaf(fa0.y, fw0.y, fmaf(fa0.x, fw0.x, acc_aw[m]))));
            acc_bw[m] = fmaf(fb1.y, fw1.y, fmaf(fb1.x, fw1.x, fmaf(fb0.y, fw0.y, fmaf(fb0.x, fw0.x, acc_bw[m]))));
          }
        }
      } else {
        const uint32_t ah[4] = {A[u].x, A[u].y, A[u].z, A[u].w}, bh[4] = {B[u].x, B[u].y, B[u].z, B[u].w};
        const uint32_t sw[4] = {XS[u][0].x ^ dep, XS[u][0].y, XS[u][0].z, XS[u][0].w};
        const uint32_t ww[4] = {XW[u][0].x ^ dep, XW[u][0].y, XW[u][0].z, XW[u][0].w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const float2 fa = act::unpack2(ah[s]), fb = act::unpack2(bh[s]);
#pragma unroll
          for (int m = 0; m < M; ++m) {
            const int sl = 4 * m + t;
            const float2 fs = act::unpack2(__shfl_sync(0xffffffffu, sw[s], sl));
            acc_as[m] = fmaf(fa.y, fs.y, fmaf(fa.x, fs.x, acc_as[m]));
            acc_bs[m] = fmaf(fb.y, fs.y, fmaf(fb.x, fs.x, acc_bs[m]));
            const float2 fw = act::unpack2(__shfl_sync(0xffffffffu, ww[s], sl));
            acc_aw[m] = fmaf(fa.y, fw.y, fmaf(fa.x, fw.x, acc_aw[m]));
            acc_bw[m] = fmaf(fb.y, fw.y, fmaf(fb.x, fw.x, acc_bw[m]));
          }
        }
      }
    }
  }
#else
  const bool la = ra < rows, lb = rb < rows, lx = g < M;
  const uint8_t* wa = w + size_t(la ? ra : 0) * ROW_BYTES;
  const uint8_t* wb = w + size_t(lb ? rb : 0) * ROW_BYTES;
  const ActT* xs_g = x_strong + size_t(lx ? g : 0) * K;
  const ActT* xw_g = x_weak + size_t(lx ? g : 0) * K;
  float c_s[4] = {0.f, 0.f, 0.f, 0.f};
  float c_w[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
  for (int jb = 0; jb < NJ; jb += U) {
    uint4 A[U], B[U], XS[U][CPL / 8], XW[U][CPL / 8];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const size_t off = size_t(c0 + (jb + u) * SPAN) * (FP8 ? 1 : 2);
      A[u] = la ? l2::ld_stream<WEIGHT_L2>(wa + off, wpol) : make_uint4(0, 0, 0, 0);
      B[u] = lb ? l2::ld_stream<WEIGHT_L2>(wb + off, wpol) : make_uint4(0, 0, 0, 0);
    }
    if (!waited) {
      pdl_wait_fenced(zero);
      pdl_trigger();
      waited = true;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int col = c0 + (jb + u) * SPAN;
#pragma unroll
      for (int h = 0; h < CPL / 8; ++h) {
        XS[u][h] = lx ? ld_act(xs_g + col + 8 * h) : make_uint4(0, 0, 0, 0);
        XW[u][h] = lx ? ld_act(xw_g + col + 8 * h) : make_uint4(0, 0, 0, 0);
      }
    }
    uint32_t dep = 0;
#pragma unroll
    for (int u = 0; u < U; ++u) dep ^= A[u].x ^ A[u].w ^ B[u].x ^ B[u].w;
    dep &= zero;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      if constexpr (FP8) {
        const uint32_t aw[4] = {A[u].x, A[u].y, A[u].z, A[u].w}, bw[4] = {B[u].x, B[u].y, B[u].z, B[u].w};
        const uint32_t sw[8] = {XS[u][0].x, XS[u][0].y, XS[u][0].z, XS[u][0].w,
                                XS[u][1].x, XS[u][1].y, XS[u][1].z, XS[u][1].w};
        const uint32_t ww[8] = {XW[u][0].x, XW[u][0].y, XW[u][0].z, XW[u][0].w,
                                XW[u][1].x, XW[u][1].y, XW[u][1].z, XW[u][1].w};
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          const uint32_t a0 = e4m3x2_to_mma_act2(aw[s]), b0 = e4m3x2_to_mma_act2(bw[s]);
          const uint32_t a1 = e4m3x2_to_mma_act2(aw[s] >> 16), b1 = e4m3x2_to_mma_act2(bw[s] >> 16);
          mma_bf16(c_s, a0, b0, a1, b1, sw[2 * s] ^ dep, sw[2 * s + 1]);
          mma_bf16(c_w, a0, b0, a1, b1, ww[2 * s] ^ dep, ww[2 * s + 1]);
        }
      } else {
        mma_bf16(c_s, A[u].x, B[u].x, A[u].y, B[u].y, XS[u][0].x ^ dep, XS[u][0].y);
        mma_bf16(c_s, A[u].z, B[u].z, A[u].w, B[u].w, XS[u][0].z ^ dep, XS[u][0].w);
        mma_bf16(c_w, A[u].x, B[u].x, A[u].y, B[u].y, XW[u][0].x ^ dep, XW[u][0].y);
        mma_bf16(c_w, A[u].z, B[u].z, A[u].w, B[u].w, XW[u][0].z ^ dep, XW[u][0].w);
      }
    }
  }
#endif

#if INGINE_ACT_FP16 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 890
  constexpr float kFp8ScaleMul = FP8 ? 256.f : 1.f;
#else
  constexpr float kFp8ScaleMul = 1.f;
#endif

  auto finish_pass = [&](float* out_dst, bool is_weak) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    mma::sm70_reduce_acc8_to_d4(is_weak ? c8_w : c8_s, c);
    cs[warp][g][2 * t] = c[0];
    cs[warp][g][2 * t + 1] = c[1];
    cs[warp][g + 8][2 * t] = c[2];
    cs[warp][g + 8][2 * t + 1] = c[3];
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
    float* aa = is_weak ? acc_aw : acc_as;
    float* ab = is_weak ? acc_bw : acc_bs;
#pragma unroll
    for (int m = 0; m < M; ++m) {
      aa[m] += __shfl_xor_sync(0xffffffffu, aa[m], 1);
      aa[m] += __shfl_xor_sync(0xffffffffu, aa[m], 2);
      ab[m] += __shfl_xor_sync(0xffffffffu, ab[m], 1);
      ab[m] += __shfl_xor_sync(0xffffffffu, ab[m], 2);
      if (t == 0) {
        cs[warp][g][m] = aa[m];
        cs[warp][g + 8][m] = ab[m];
      }
    }
#else
    const float* c = is_weak ? c_w : c_s;
    cs[warp][g][2 * t] = c[0];
    cs[warp][g][2 * t + 1] = c[1];
    cs[warp][g + 8][2 * t] = c[2];
    cs[warp][g + 8][2 * t + 1] = c[3];
#endif
    __syncthreads();
    for (int k = threadIdx.x; k < 16 * M; k += 32 * SPLITK) {
      const int r = k / M, m = k % M, row = r0 + r;
      if (row < rows) {
        float s = 0.f;
#pragma unroll
        for (int sk = 0; sk < SPLITK; ++sk) s += cs[sk][r][m];
        const float v = s * (FP8 ? (__ldg(w_scale + row) * kFp8ScaleMul) : 1.f);
        out_dst[size_t(m) * rows + row] = tanhf(round_bf16(v) / 30.f) * 30.f;
      }
    }
  };
  finish_pass(out_strong, false);
  __syncthreads();
  finish_pass(out_weak, true);
}

struct OnlineTop2Sum {
  float m1 = -INFINITY;
  float m2 = -INFINITY;
  float sum = 0.f;
};

__device__ __forceinline__ void online_add(OnlineTop2Sum& s, float x) {
  if (x > s.m1) {
    s.sum = (s.m1 > -INFINITY ? s.sum * expf(s.m1 - x) : 0.f) + 1.f;
    s.m2 = s.m1;
    s.m1 = x;
  } else {
    if (x > s.m2) s.m2 = x;
    s.sum += expf(x - s.m1);
  }
}

__device__ __forceinline__ OnlineTop2Sum online_merge(OnlineTop2Sum a, OnlineTop2Sum b) {
  if (b.m1 == -INFINITY) return a;
  if (a.m1 == -INFINITY) return b;
  if (a.m1 >= b.m1) {
    return {a.m1, fmaxf(a.m2, b.m1), a.sum + b.sum * expf(b.m1 - a.m1)};
  }
  return {b.m1, fmaxf(b.m2, a.m1), b.sum + a.sum * expf(a.m1 - b.m1)};
}

__global__ void loopcd_stats_blocks_kernel(const float* __restrict__ z_s, int V, float* __restrict__ ws_reduce) {
  const int m = blockIdx.y;
  const float* row = z_s + size_t(m) * V;
  OnlineTop2Sum acc;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += stride) {
    online_add(acc, row[i]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    OnlineTop2Sum other{__shfl_xor_sync(0xffffffffu, acc.m1, o), __shfl_xor_sync(0xffffffffu, acc.m2, o),
                        __shfl_xor_sync(0xffffffffu, acc.sum, o)};
    acc = online_merge(acc, other);
  }
  __shared__ float sm1[8], sm2[8], ssum[8];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane == 0) {
    sm1[warp] = acc.m1;
    sm2[warp] = acc.m2;
    ssum[warp] = acc.sum;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    OnlineTop2Sum tot;
    const int nw = blockDim.x >> 5;
    for (int w = 0; w < nw; ++w) tot = online_merge(tot, OnlineTop2Sum{sm1[w], sm2[w], ssum[w]});
    float* dst = ws_reduce + (size_t(m) * kLoopCdReduceBlocks + blockIdx.x) * 3;
    dst[0] = tot.m1;
    dst[1] = tot.m2;
    dst[2] = tot.sum;
  }
}

__global__ void loopcd_stats_final_kernel(float* __restrict__ ws_reduce, float omega_max) {
  const int m = blockIdx.x;
  const int tid = threadIdx.x;
  const float* src = ws_reduce + (size_t(m) * kLoopCdReduceBlocks + tid) * 3;
  OnlineTop2Sum acc = tid < kLoopCdReduceBlocks ? OnlineTop2Sum{src[0], src[1], src[2]} : OnlineTop2Sum{};
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    OnlineTop2Sum other{__shfl_xor_sync(0xffffffffu, acc.m1, o), __shfl_xor_sync(0xffffffffu, acc.m2, o),
                        __shfl_xor_sync(0xffffffffu, acc.sum, o)};
    acc = online_merge(acc, other);
  }
  __shared__ float sm1[4], sm2[4], ssum[4];
  const int warp = tid >> 5, lane = tid & 31;
  if (lane == 0) {
    sm1[warp] = acc.m1;
    sm2[warp] = acc.m2;
    ssum[warp] = acc.sum;
  }
  __syncthreads();
  if (tid == 0) {
    OnlineTop2Sum tot;
    for (int w = 0; w < 4; ++w) tot = online_merge(tot, OnlineTop2Sum{sm1[w], sm2[w], ssum[w]});
    const float inv_sum = tot.sum > 0.f ? (1.f / tot.sum) : 0.f;
    const float p1 = inv_sum;
    const float p2 = (tot.m2 > -INFINITY) ? (expf(tot.m2 - tot.m1) * inv_sum) : 0.f;
    const float margin = fmaxf(0.f, fminf(1.f, p1 - p2));
    ws_reduce[size_t(m) * kLoopCdReduceFloatsPerRow] = omega_max * (1.f - margin);
  }
}

__global__ void loopcd_combine_logits_kernel(float* __restrict__ out_strong, const float* __restrict__ z_weak,
                                             const float* __restrict__ ws_reduce, int V) {
  const int m = blockIdx.y;
  const float omega = __ldg(ws_reduce + size_t(m) * kLoopCdReduceFloatsPerRow);
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int v4 = idx * 4;
  float* s_row = out_strong + size_t(m) * V;
  const float* w_row = z_weak + size_t(m) * V;
  if (v4 + 3 < V && (uintptr_t(s_row) & 15) == 0 && (uintptr_t(w_row) & 15) == 0) {
    const float4 zs = reinterpret_cast<const float4*>(s_row)[idx];
    const float4 zw = reinterpret_cast<const float4*>(w_row)[idx];
    float4 o;
    o.x = fmaf(omega, zs.x - zw.x, zs.x);
    o.y = fmaf(omega, zs.y - zw.y, zs.y);
    o.z = fmaf(omega, zs.z - zw.z, zs.z);
    o.w = fmaf(omega, zs.w - zw.w, zs.w);
    reinterpret_cast<float4*>(s_row)[idx] = o;
  } else {
    for (int k = 0; k < 4; ++k) {
      const int v = v4 + k;
      if (v < V) {
        const float zs = s_row[v];
        s_row[v] = fmaf(omega, zs - w_row[v], zs);
      }
    }
  }
}

__global__ void loopcd_hidden_blend_kernel(float* __restrict__ h_strong, const float* __restrict__ h_weak, int total4,
                                           float omega) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total4) return;
  const float4 hs = reinterpret_cast<const float4*>(h_strong)[i];
  const float4 hw = reinterpret_cast<const float4*>(h_weak)[i];
  float4 o;
  o.x = fmaf(omega, hs.x - hw.x, hs.x);
  o.y = fmaf(omega, hs.y - hw.y, hs.y);
  o.z = fmaf(omega, hs.z - hw.z, hs.z);
  o.w = fmaf(omega, hs.w - hw.w, hs.w);
  reinterpret_cast<float4*>(h_strong)[i] = o;
}

__global__ void loopcd_hidden_from_residual_kernel(float* __restrict__ dst, const float* __restrict__ base,
                                                   const float* __restrict__ write, const float* __restrict__ h_weak,
                                                   int total4, float omega) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total4) return;
  const float4 b = reinterpret_cast<const float4*>(base)[i];
  const float4 w = reinterpret_cast<const float4*>(write)[i];
  const float4 hw = reinterpret_cast<const float4*>(h_weak)[i];
  const float4 hs = make_float4(b.x + w.x, b.y + w.y, b.z + w.z, b.w + w.w);
  float4 o;
  o.x = fmaf(omega, hs.x - hw.x, hs.x);
  o.y = fmaf(omega, hs.y - hw.y, hs.y);
  o.z = fmaf(omega, hs.z - hw.z, hs.z);
  o.w = fmaf(omega, hs.w - hw.w, hs.w);
  reinterpret_cast<float4*>(dst)[i] = o;
}

}  // namespace

void proj(const ActT* xin, int M, Matrix w, const int* segments, int nseg, float* out, bool pdl,
          cudaStream_t stream) {
  require(xin && segments && out && nseg > 0, "proj: null argument");
  const int R = w.rows;
  launch_gemv<H, 256, PROJ_SPLITK, MMA_U_FP8, ProjEpi>(w, M, pdl, stream, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return ProjEpi<MM>{xin, segments, nseg, R, out};
  });
}

void proj_conv(const ActT* x, int M, Matrix w, const float* conv_w, float* conv_state, float* out,
               float* conv_hist, bool pdl, cudaStream_t stream) {
  require(x && conv_w && conv_state && out, "proj_conv: null argument");
  require(w.rows == H, "proj_conv: w must have 2304 rows");
  // M = 1: 2304 rows = 1152-2304 warps; 32-thread blocks spread them evenly over the SMs (measured: equal
  // or better than 64-256)
  launch_gemv<3072, 32, CONV_SPLITK, CONV_U_FP8, ConvEpi>(w, M, pdl, stream, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return ConvEpi<MM>{x, conv_w, conv_state, out, conv_hist};
  });
}

void head(const ActT* x, int M, Matrix w, float* out, bool pdl, cudaStream_t stream) {
  require(x && out, "head: null argument");
  const int V = w.rows;
  launch_gemv<H, 256, HEAD_SPLITK, MMA_U_FP8, HeadEpi>(w, M, pdl, stream, [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return HeadEpi<MM>{x, V, out};
  });
}

void head_loopcd(const ActT* x_strong, const ActT* x_weak, int M, Matrix w, float omega_max, float* out,
                 float* ws_weak, float* ws_reduce, bool pdl, cudaStream_t stream) {
  require(x_strong && x_weak && out && ws_weak && ws_reduce, "head_loopcd: null argument");
  require(w.w != nullptr && w.rows > 0, "head_loopcd: weight matrix is empty");
  require(M >= 1 && M <= 8, "head_loopcd: M must be in 1..8");
  require(omega_max >= 0.f, "head_loopcd: omega_max must be non-negative");
  const bool fp8 = w.scale != nullptr;
  const int rows = w.rows;
  const float* sc = w.scale;
  const uint8_t* wp = static_cast<const uint8_t*>(w.w);
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    if constexpr (MM == 1) {
      if (fp8) {
        constexpr int B = kActIsFP16 ? 128 : 256;
        launch_ex(head_dual_gemv_kernel<true, H, 16, B>, dim3((rows * 16 + B - 1) / B), B, 0, 1, pdl, stream, wp, sc,
                  rows, x_strong, x_weak, out, ws_weak, 0u);
      } else {
        launch_ex(head_dual_gemv_kernel<false, H, 32, 256>, dim3((rows * 32 + 255) / 256), 256, 0, 1, pdl, stream, wp,
                  sc, rows, x_strong, x_weak, out, ws_weak, 0u);
      }
    } else {
      const dim3 blocks((rows + 15) / 16);
      if (fp8) {
        launch_ex(head_dual_mma_kernel<true, H, MM, HEAD_SPLITK, MMA_U_FP8>, blocks, 32 * HEAD_SPLITK, 0, 1, pdl,
                  stream, wp, sc, rows, x_strong, x_weak, out, ws_weak, 0u);
      } else {
        launch_ex(head_dual_mma_kernel<false, H, MM, HEAD_SPLITK, MMA_U_BF16>, blocks, 32 * HEAD_SPLITK, 0, 1, pdl,
                  stream, wp, sc, rows, x_strong, x_weak, out, ws_weak, 0u);
      }
    }
  };
  switch (M) {
    case 1: go(std::integral_constant<int, 1>{}); break;
    case 2: go(std::integral_constant<int, 2>{}); break;
    case 3: go(std::integral_constant<int, 3>{}); break;
    case 4: go(std::integral_constant<int, 4>{}); break;
    case 5: go(std::integral_constant<int, 5>{}); break;
    case 6: go(std::integral_constant<int, 6>{}); break;
    case 7: go(std::integral_constant<int, 7>{}); break;
    case 8: go(std::integral_constant<int, 8>{}); break;
  }
  loopcd_adaptive_combine(out, ws_weak, M, rows, omega_max, ws_reduce, stream);
}

void loopcd_adaptive_combine(float* out_strong, const float* z_weak, int M, int V, float omega_max, float* ws_reduce,
                             cudaStream_t stream) {
  require(out_strong && z_weak && ws_reduce && M >= 1 && M <= 8 && V > 0 && omega_max >= 0.f,
          "loopcd_adaptive_combine: bad args");
  if (omega_max == 0.f) return;
  loopcd_stats_blocks_kernel<<<dim3(kLoopCdReduceBlocks, M), 256, 0, stream>>>(out_strong, V, ws_reduce);
  loopcd_stats_final_kernel<<<M, kLoopCdReduceBlocks, 0, stream>>>(ws_reduce, omega_max);
  loopcd_combine_logits_kernel<<<dim3((V + 1023) / 1024, M), 256, 0, stream>>>(out_strong, z_weak, ws_reduce, V);
  CUDA_CHECK(cudaGetLastError());
}

void loopcd_hidden_blend(float* h_strong, const float* h_weak, int M, float omega, bool pdl, cudaStream_t stream) {
  require(h_strong && h_weak && M >= 1, "loopcd_hidden_blend: bad args");
  (void)pdl;
  const int total4 = (M * H) / 4;
  loopcd_hidden_blend_kernel<<<(total4 + 255) / 256, 256, 0, stream>>>(h_strong, h_weak, total4, omega);
  CUDA_CHECK(cudaGetLastError());
}

void loopcd_hidden_from_residual(float* dst, const float* base, const float* write, const float* h_weak, int M,
                                 float omega, bool pdl, cudaStream_t stream) {
  require(dst && base && write && h_weak && M >= 1, "loopcd_hidden_from_residual: bad args");
  (void)pdl;
  const int total4 = (M * H) / 4;
  loopcd_hidden_from_residual_kernel<<<(total4 + 255) / 256, 256, 0, stream>>>(dst, base, write, h_weak, total4, omega);
  CUDA_CHECK(cudaGetLastError());
}

void gemv(const ActT* x, int M, int K, Matrix w, float* out, bool pdl, cudaStream_t stream) {
  require(x && out, "gemv: null argument");
  const int R = w.rows;
  auto make = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    return GemvEpi<MM>{x, R, out};
  };
  // 2304 rows = 144 blocks of 16 rows: 16 warps per block (K slices of 192 / 576 columns) keep enough
  // bytes in flight (measured M = 8, BF16: K = 3072 10.4 -> 8.4 us, K = 9216 28.3 -> 26.0 us vs 8 warps).
  // For BF16, K = 3072, M = 1: as proj_conv; K = 9216: 36 chunks per lane would spill, so M = 1 uses tensor cores.
  if (K == 3072) launch_gemv<3072, 32, 16, MMA_U_FP8, GemvEpi>(w, M, pdl, stream, make);
  else if (K == 4096) launch_gemv<4096, 0, 16, 2, GemvEpi, 4>(w, M, pdl, stream, make);
  else if (K == 6912) launch_gemv<6912, 0, 12, MMA_U_FP8, GemvEpi>(w, M, pdl, stream, make);
  else if (K == 1536) launch_gemv<1536, 32, 8, MMA_U_FP8, GemvEpi>(w, M, pdl, stream, make);
  else if (K == 768) launch_gemv<768, 32, 4, MMA_U_FP8, GemvEpi>(w, M, pdl, stream, make);
  else if (K == 9216) launch_gemv<9216, 0, 16, MMA_U_FP8, GemvEpi>(w, M, pdl, stream, make);
  else require(false, "gemv: K must be 768, 1536, 3072, 4096, 6912 or 9216");
}

void fp8_rows(const ActT* w, int rows, int K, void* q, float* scale, cudaStream_t stream) {
  require(rows >= 0 && K > 0, "fp8_rows: bad shape");
  if (rows > 0)
    fp8_rows_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(w, rows, K, static_cast<__nv_fp8_storage_t*>(q), scale);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void static_conv_decode_kernel(const float* __restrict__ x_f32, int M, const float* __restrict__ conv_w,
                                          float* __restrict__ state, float* __restrict__ out, float* __restrict__ hist) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= H) return;
  const float4 cw = __ldg(reinterpret_cast<const float4*>(conv_w) + c);
  float ext[11];
  ext[0] = state[c];
  ext[1] = state[H + c];
  ext[2] = state[2 * H + c];
  for (int m = 0; m < M; ++m) ext[3 + m] = round_bf16(x_f32[size_t(m) * H + c]);
  constexpr float inv_sqrt5 = 1.f / 2.2360679774997896f;
  for (int m = 0; m < M; ++m) {
    float acc = __fmul_rn(cw.x, ext[m]);
    acc = __fadd_rn(acc, __fmul_rn(cw.y, ext[m + 1]));
    acc = __fadd_rn(acc, __fmul_rn(cw.z, ext[m + 2]));
    acc = __fadd_rn(acc, __fmul_rn(cw.w, ext[m + 3]));
    out[size_t(m) * H + c] = __fmul_rn(__fadd_rn(acc, ext[m + 3]), inv_sqrt5);
    if (hist) hist[size_t(m) * H + c] = ext[m + 3];
  }
  state[c] = ext[M];
  state[H + c] = ext[M + 1];
  state[2 * H + c] = ext[M + 2];
}

void static_conv_decode(const float* x_f32, int M, const float* conv_w, float* conv_state, float* out,
                        float* conv_hist, cudaStream_t stream) {
  require(x_f32 && conv_w && conv_state && out && M >= 1 && M <= 8, "static_conv_decode: bad args");
  static_conv_decode_kernel<<<(H + 255) / 256, 256, 0, stream>>>(x_f32, M, conv_w, conv_state, out, conv_hist);
  CUDA_CHECK(cudaGetLastError());
}

void dequant_rows(const void* q, const float* scale, int rows, int K, ActT* out, cudaStream_t stream) {
  require(rows >= 0 && K > 0 && K % 16 == 0, "dequant_rows: K must be a multiple of 16");
  const long long total16 = (long long)rows * K / 16;
  if (total16 > 0)
    dequant_rows_kernel<<<unsigned((total16 + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_fp8_storage_t*>(q), scale, K, total16, out);
  CUDA_CHECK(cudaGetLastError());
}

// 1 warp per GGML BlockQ5K (256 elements): lane l in 0..31 owns element l of all 8 sub-blocks,
// matching q5k::quantize_block_cpu bit-for-bit with zero shared memory and coalesced 32B stores.
__global__ void quantize_q5_k_rows_kernel(const ActT* __restrict__ w, int rows, int K, uint8_t* __restrict__ out) {
  const int nb = K / q5k::kBlockElems;
  const long long blk_idx = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const long long total_blks = (long long)rows * nb;
  if (blk_idx >= total_blks) return;

  const int lane = threadIdx.x & 31;
  const int row = int(blk_idx / nb);
  const int b = int(blk_idx % nb);
  const ActT* src = w + (size_t(row) * K + size_t(b) * q5k::kBlockElems);
  uint8_t* dst = out + size_t(blk_idx) * q5k::kBlockBytes;

  float vals[8];
  float sb_mn[8];
  float sb_mx[8];
  float max_min = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float v = act::to_float(src[j * 32 + lane]);
    vals[j] = v;
    float mn = fminf(0.f, v);
    float mx = fmaxf(0.f, v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, o));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    sb_mn[j] = mn;
    sb_mx[j] = mx;
    max_min = fmaxf(max_min, -mn);
  }

  const __half dm_h = __float2half_rn(__fdiv_rn(max_min, 63.0f));
  const float dm_val = __half2float(dm_h);

  uint8_t m_q[8];
  float need_sc[8];
  float max_scale = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int mq = (dm_val > 0.f) ? min(63, max(0, int(ceilf(__fdiv_rn(-sb_mn[j], dm_val))))) : 0;
    m_q[j] = static_cast<uint8_t>(mq);
    const float eff_m = dm_val * float(mq);
    const float sc = fmaxf(0.f, __fdiv_rn(sb_mx[j] + eff_m, 31.0f));
    need_sc[j] = sc;
    max_scale = fmaxf(max_scale, sc);
  }

  const __half d_h = __float2half_rn(__fdiv_rn(max_scale, 63.0f));
  const float d_val = __half2float(d_h);

  uint8_t sc_q[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int sq = (d_val > 0.f) ? min(63, max(0, int(ceilf(__fdiv_rn(need_sc[j], d_val))))) : 0;
    sc_q[j] = static_cast<uint8_t>(sq);
  }

  if (lane == 0) {
    *reinterpret_cast<uint16_t*>(dst) = __half_as_ushort(d_h);
    *reinterpret_cast<uint16_t*>(dst + 2) = __half_as_ushort(dm_h);
    for (int j = 0; j < 4; ++j) {
      dst[4 + j] = static_cast<uint8_t>((sc_q[j] & 63u) | ((sc_q[j + 4] >> 4) << 6));
      dst[8 + j] = static_cast<uint8_t>((m_q[j] & 63u) | ((m_q[j + 4] >> 4) << 6));
      dst[12 + j] = static_cast<uint8_t>((sc_q[j + 4] & 0x0Fu) | ((m_q[j + 4] & 0x0Fu) << 4));
    }
  }

  uint8_t qh_byte = 0;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const int j0 = 2 * c;
    const int j1 = 2 * c + 1;
    const float eff_s0 = d_val * float(sc_q[j0]);
    const float eff_m0 = dm_val * float(m_q[j0]);
    const float eff_s1 = d_val * float(sc_q[j1]);
    const float eff_m1 = dm_val * float(m_q[j1]);
    const float inv_s0 = (eff_s0 > 0.f) ? (1.0f / eff_s0) : 0.f;
    const float inv_s1 = (eff_s1 > 0.f) ? (1.0f / eff_s1) : 0.f;
    const int q0 = (inv_s0 > 0.f) ? min(31, max(0, int(rintf((vals[j0] + eff_m0) * inv_s0)))) : 0;
    const int q1 = (inv_s1 > 0.f) ? min(31, max(0, int(rintf((vals[j1] + eff_m1) * inv_s1)))) : 0;
    dst[48 + 32 * c + lane] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    if (q0 & 0x10) qh_byte |= static_cast<uint8_t>(1u << (2 * c));
    if (q1 & 0x10) qh_byte |= static_cast<uint8_t>(1u << (2 * c + 1));
  }
  dst[16 + lane] = qh_byte;
}

__global__ void dequant_q5_k_rows_kernel(const uint8_t* __restrict__ q, int rows, int K, ActT* __restrict__ out) {
  const int nb = K / q5k::kBlockElems;
  const long long blk_idx = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const long long total_blks = (long long)rows * nb;
  if (blk_idx >= total_blks) return;

  const int lane = threadIdx.x & 31;
  const int row = int(blk_idx / nb);
  const int b = int(blk_idx % nb);
  const uint8_t* blk = q + size_t(blk_idx) * q5k::kBlockBytes;
  ActT* dst = out + (size_t(row) * K + size_t(b) * q5k::kBlockElems);

  const float d = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(blk)));
  const float dmin = __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(blk + 2)));
  uint8_t sc_raw[12];
#pragma unroll
  for (int i = 0; i < 12; ++i) sc_raw[i] = __ldg(blk + 4 + i);
  const uint8_t qh = __ldg(blk + 16 + lane);
  const uint8_t* qs = blk + 48;

#pragma unroll
  for (int c = 0; c < 4; ++c) {
    uint8_t sc0 = 0, m0 = 0, sc1 = 0, m1 = 0;
    q5k_get_scale_min(2 * c, sc_raw, sc0, m0);
    q5k_get_scale_min(2 * c + 1, sc_raw, sc1, m1);
    const float d0 = d * float(sc0);
    const float dm0 = dmin * float(m0);
    const float d1 = d * float(sc1);
    const float dm1 = dmin * float(m1);
    const uint8_t ql = __ldg(qs + 32 * c + lane);
    const int q0 = int(ql & 0x0Fu) + ((qh & (1u << (2 * c))) ? 16 : 0);
    const int q1 = int(ql >> 4) + ((qh & (1u << (2 * c + 1))) ? 16 : 0);
    dst[64 * c + lane] = act::from_float(fmaf(d0, float(q0), -dm0));
    dst[64 * c + 32 + lane] = act::from_float(fmaf(d1, float(q1), -dm1));
  }
}

void quantize_q5_k_rows(const ActT* w, int rows, int K, void* q, cudaStream_t stream) {
  require(w && q && rows >= 0 && K > 0 && (K % q5k::kBlockElems) == 0,
          "quantize_q5_k_rows: K must be a positive multiple of 256");
  const long long total_blks = (long long)rows * (K / q5k::kBlockElems);
  if (total_blks > 0) {
    // 8 warps = 256 threads per CTA -> 8 super-blocks per CTA
    quantize_q5_k_rows_kernel<<<unsigned((total_blks + 7) / 8), 256, 0, stream>>>(
        w, rows, K, static_cast<uint8_t*>(q));
  }
  CUDA_CHECK(cudaGetLastError());
}

void dequant_q5_k_rows(const void* q, int rows, int K, ActT* out, cudaStream_t stream) {
  require(q && out && rows >= 0 && K > 0 && (K % q5k::kBlockElems) == 0,
          "dequant_q5_k_rows: K must be a positive multiple of 256");
  const long long total_blks = (long long)rows * (K / q5k::kBlockElems);
  if (total_blks > 0) {
    dequant_q5_k_rows_kernel<<<unsigned((total_blks + 7) / 8), 256, 0, stream>>>(
        static_cast<const uint8_t*>(q), rows, K, out);
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::dense

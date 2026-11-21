// message_gates: loop-pass message gates (rank-64 down projection, tanh forget/gain, gated mix, RMS norm).
//
// Design. Per gate the work is a chain of all-to-all dependencies (row norms -> 64 down outputs -> 2304
// gated channels -> their RMS norm), so each gate is one cluster of CL = 16 CTAs (non-portable size; with
// 8 the FP32 forget/gain slices would not fit in shared memory) and the dependencies are crossed with
// distributed shared memory instead of extra launches. Per-SM DRAM concurrency is the limit for such a
// small grid (~110 KB per SM at G = 3), so every byte is requested at kernel start. CTA r of gate g:
// * issues, in this order: its share of the token-0 inputs (L2 hits), cp.async copies of its forget/gain
//   row slices (rows [r * 144, +144), padded rows in shared memory), and 16-byte streaming loads of its
//   4 rows of `w_down` (2 warps per row: half 0 against n(h), half 1 against n(msg)) into registers;
// * down projection, one token at a time: a block sum of squares normalises h and msg once per CTA,
//   bf16(x * rinv) is staged in shared memory (each input element is read from L2 once per CTA), and
//   every warp dots its row half against it; the next token's inputs are loaded during the dots;
// * the 4 BF16-rounded down outputs per token are pushed into every CTA's copy of g [M, 64]; one cluster
//   barrier later every CTA holds the full g (this thread's h, msg, norm_w values load meanwhile);
// * forget/gain: one thread per channel reads its two weight rows from shared memory (16-byte chunks,
//   the 16-byte row padding puts 8 consecutive threads in distinct banks) against broadcast g, applies
//   gated = (1 - tanh) h + tanh msg in registers, a block sum gives the CTA's per-token sum of squares,
//   pushed to every CTA; after a second barrier each CTA normalises and writes its own channels.
// Every CTA sums the pushed partials in the same order, so all CTAs use identical g and norms.
// pdl=True: all weight copies/loads are issued before griddepcontrol.wait, h/msg are read after it.
// Ported from engine/csrc/dense/gates.cu (kernel bodies unchanged; torch launchers -> raw pointers).
#include <cooperative_groups.h>

#include "kernels/dense/cluster.cuh"
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"

#include <type_traits>

namespace cg = cooperative_groups;

namespace ingine::kernels::dense {
namespace {

constexpr int RANK = 64;

struct GateArgs {
  const float* h;    // [G, M, H]
  const float* msg;  // [G, M, H]
  const __nv_bfloat16* w_down;  // [G, 64, 2H]
  const void* w_forget;         // [G, H, 64] fp32 or bf16
  const void* w_gain;
  const float* norm_w;  // [G, H]
  __nv_bfloat16* out_bf16;  // [G, M, H]
  float* out_f32;           // [G, M, H] or null
};

template <int CL, bool FG32>
struct GateCfg {
  static constexpr int RPC = RANK / CL;           // down rows per CTA
  static constexpr int NW = 2 * RPC;              // warps: (row, half)
  static constexpr int NT = 32 * NW;              // threads
  static constexpr int CPC = H / CL;              // channels per CTA (one thread each in phase C)
  static constexpr int FGB = FG32 ? 4 : 2;        // bytes per forget/gain weight
  static constexpr int ROW = RANK * FGB;          // bytes of one forget/gain row
  static constexpr int RS = ROW + 16;             // padded row stride in shared memory (no bank conflicts)
  static constexpr int SLICE = CPC * RS;          // bytes of one padded matrix slice
  static_assert(CPC <= NT && ROW % 16 == 0, "uneven split");
  static constexpr int SMEM = 2 * SLICE;
};

template <int CL, bool FG32, int M>
__global__ void __launch_bounds__(GateCfg<CL, FG32>::NT) gates_kernel(GateArgs a) {
  using C = GateCfg<CL, FG32>;
  constexpr int DCH = H / (8 * 32);  // 9 chunks of 8 down weights per lane
  constexpr int XPT = H / C::NT;     // inputs per thread per half when staging x
  static_assert(H % C::NT == 0, "x staging split");
  extern __shared__ __align__(16) uint8_t fg[];  // forget slice, then gain slice (padded rows)
  __shared__ __align__(16) float gall[M][RANK];  // full down output (BF16 values)
  __shared__ __align__(16) __nv_bfloat16 xs[2 * H];  // bf16([n(h), n(msg)]) of the current token
  __shared__ float part[C::RPC][2][M];  // per-half partial dots
  __shared__ float ssall[CL][M];
  __shared__ float red[(C::NW + 1) * block_sum_slots<M>()], red2[(C::NW + 1) * 2];

  cluster_arrive_relaxed();
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int g = blockIdx.x / CL;
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int rl = warp >> 1, half = warp & 1;  // down projection: warp = (row_local, half)
  const float* hg = a.h + size_t(g) * M * H;
  const float* mg = a.msg + size_t(g) * M * H;

  // inputs: XPT of h and of msg per thread, coalesced (activations: loaded after the PDL wait)
  float xh[XPT], xm[XPT];
  auto load_x = [&](int m) {
#pragma unroll
    for (int k = 0; k < XPT; ++k) {
      xh[k] = hg[size_t(m) * H + k * C::NT + t];
      xm[k] = mg[size_t(m) * H + k * C::NT + t];
    }
  };

  // forget/gain slices (padded rows) -> shared memory, asynchronous, waited for before phase C
  {
    const uint8_t* fsrc = reinterpret_cast<const uint8_t*>(a.w_forget) + (size_t(g) * H + rank * C::CPC) * C::ROW;
    const uint8_t* gsrc = reinterpret_cast<const uint8_t*>(a.w_gain) + (size_t(g) * H + rank * C::CPC) * C::ROW;
    constexpr int RCH = C::ROW / 16;
    for (int k = t; k < C::CPC * RCH; k += C::NT) {
      const int dst = (k / RCH) * C::RS + (k % RCH) * 16;
      cp_async16(fg + dst, fsrc + size_t(k) * 16);
      cp_async16(fg + C::SLICE + dst, gsrc + size_t(k) * 16);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }

  // down projection weights -> registers
  uint4 wv[DCH];
  {
    const __nv_bfloat16* wr = a.w_down + (size_t(g) * RANK + rank * C::RPC + rl) * 2 * H + half * H;
#pragma unroll
    for (int i = 0; i < DCH; ++i) wv[i] = ld_stream(wr + (i * 32 + lane) * 8);
  }

  // all weights are in flight; h and msg come from earlier kernels
  pdl_wait();
  pdl_trigger();
  load_x(0);

  // down projection, one token at a time: the CTA normalises the token's h and msg once (block sum of
  // squares), stages bf16(x * rinv) in shared memory, and every warp dots its row half against it
#pragma unroll 1
  for (int m = 0; m < M; ++m) {
    float ss[2] = {0.f, 0.f};
#pragma unroll
    for (int k = 0; k < XPT; ++k) {
      ss[0] = fmaf(xh[k], xh[k], ss[0]);
      ss[1] = fmaf(xm[k], xm[k], ss[1]);
    }
    block_sum(ss, red2);  // its barriers also order the previous token's dots before xs is rewritten
    const float rh = rsqrtf(ss[0] / H + EPS), rm = rsqrtf(ss[1] / H + EPS);
#pragma unroll
    for (int k = 0; k < XPT; ++k) {
      xs[k * C::NT + t] = __float2bfloat16_rn(xh[k] * rh);
      xs[H + k * C::NT + t] = __float2bfloat16_rn(xm[k] * rm);
    }
    if (m + 1 < M) load_x(m + 1);  // next token's inputs fly during this token's dots
    __syncthreads();
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < DCH; ++i) {
      float wf[8], xf[8];
      unpack_bf16x8(wv[i], wf);
      unpack_bf16x8(*reinterpret_cast<const uint4*>(&xs[half * H + (i * 32 + lane) * 8]), xf);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc = fmaf(wf[j], xf[j], acc);
    }
    acc = group_sum<32>(acc);
    if (lane == 0) part[rl][half][m] = acc;
  }
  __syncthreads();

  cluster_wait();  // all CTAs resident: remote shared memory may be written
  for (int k = t; k < CL * C::RPC * M; k += C::NT) {
    const int d = k / (C::RPC * M), r = (k / M) % C::RPC, m = k % M;
    float* dst = cluster.map_shared_rank(&gall[0][0], d);
    dst[m * RANK + rank * C::RPC + r] = round_bf16(part[r][0][m] + part[r][1][m]);
  }
  cluster_arrive();
  // this thread's channel inputs fly during the barrier
  const int c = rank * C::CPC + t;
  const bool own = t < C::CPC;
  float hv[M], mv[M], nw = 0.f;
#pragma unroll
  for (int m = 0; m < M; ++m) hv[m] = mv[m] = 0.f;
  if (own) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      hv[m] = hg[size_t(m) * H + c];
      mv[m] = mg[size_t(m) * H + c];
    }
    nw = a.norm_w[size_t(g) * H + c];
  }
  asm volatile("cp.async.wait_all;" ::: "memory");
  cluster_wait();
  __syncthreads();  // forget/gain slices from all threads' copies are visible

  // forget / gain: one thread per channel; its two weight rows are read as 16-byte chunks from the padded
  // slices (8 consecutive threads hit 8 distinct bank groups), g is a broadcast read
  float sq[M];
#pragma unroll
  for (int m = 0; m < M; ++m) sq[m] = 0.f;
  if (own) {
    constexpr int EPC = 16 / C::FGB;  // weights per chunk
    const uint8_t* fr = fg + t * C::RS;
    const uint8_t* gr = fg + C::SLICE + t * C::RS;
    float df[M], dg[M];
#pragma unroll
    for (int m = 0; m < M; ++m) df[m] = dg[m] = 0.f;
#pragma unroll 4
    for (int k = 0; k < RANK / EPC; ++k) {
      const uint4 uf = *reinterpret_cast<const uint4*>(fr + k * 16);
      const uint4 ug = *reinterpret_cast<const uint4*>(gr + k * 16);
      float wf[EPC], wg[EPC];
      if constexpr (FG32) {
        wf[0] = __uint_as_float(uf.x); wf[1] = __uint_as_float(uf.y); wf[2] = __uint_as_float(uf.z); wf[3] = __uint_as_float(uf.w);
        wg[0] = __uint_as_float(ug.x); wg[1] = __uint_as_float(ug.y); wg[2] = __uint_as_float(ug.z); wg[3] = __uint_as_float(ug.w);
      } else {
        unpack_bf16x8(uf, wf);
        unpack_bf16x8(ug, wg);
      }
#pragma unroll
      for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int e = 0; e < EPC; e += 4) {
          const float4 gv = *reinterpret_cast<const float4*>(&gall[m][k * EPC + e]);
          df[m] = fmaf(wf[e], gv.x, fmaf(wf[e + 1], gv.y, fmaf(wf[e + 2], gv.z, fmaf(wf[e + 3], gv.w, df[m]))));
          dg[m] = fmaf(wg[e], gv.x, fmaf(wg[e + 1], gv.y, fmaf(wg[e + 2], gv.z, fmaf(wg[e + 3], gv.w, dg[m]))));
        }
      }
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      hv[m] = (1.f - tanhf(df[m])) * hv[m] + tanhf(dg[m]) * mv[m];  // gated
      sq[m] = hv[m] * hv[m];
      if (a.out_f32) a.out_f32[(size_t(g) * M + m) * H + c] = hv[m];
    }
  }

  // per-token sums of squares of this CTA's channels, pushed to every CTA
  block_sum(sq, red);
  if (t < CL) {
#pragma unroll
    for (int m = 0; m < M; ++m) cluster.map_shared_rank(&ssall[0][0], t)[rank * M + m] = sq[m];
  }
  cluster_arrive();
  cluster_wait();

  if (own) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float ss = 0.f;
#pragma unroll
      for (int d = 0; d < CL; ++d) ss += ssall[d][m];
      a.out_bf16[(size_t(g) * M + m) * H + c] = __float2bfloat16_rn(hv[m] * rsqrtf(ss / H + EPS) * nw);
    }
  }
}

template <int CL, bool FG32>
void launch_gates(int G, int M, const GateArgs& a, bool pdl, cudaStream_t stream) {
  using C = GateCfg<CL, FG32>;
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    launch_ex(gates_kernel<CL, FG32, MM>, dim3(G * CL), C::NT, C::SMEM, CL, pdl, stream, a);
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
    default: require(false, "message_gates: M must be in 1..8");
  }
}

}  // namespace

void message_gates(const float* h, const float* msg, int G, int M, const __nv_bfloat16* w_down,
                   const void* w_forget, const void* w_gain, bool fg_fp32, const float* norm_w,
                   __nv_bfloat16* out_bf16, float* out_f32, bool pdl, cudaStream_t stream) {
  require(G >= 1 && G <= 3 && M >= 1 && M <= 8, "message_gates: G in 1..3, M in 1..8");
  require(h && msg && w_down && w_forget && w_gain && norm_w && out_bf16, "message_gates: null argument");
  GateArgs a;
  a.h = h;
  a.msg = msg;
  a.w_down = w_down;
  a.w_forget = w_forget;
  a.w_gain = w_gain;
  a.norm_w = norm_w;
  a.out_bf16 = out_bf16;
  a.out_f32 = out_f32;
  // 16-CTA clusters (non-portable, verified schedulable on the RTX 5090): the FP32 forget/gain slices of
  // an 8-CTA split would not fit in shared memory.
  if (fg_fp32) launch_gates<16, true>(G, M, a, pdl, stream);
  else launch_gates<16, false>(G, M, a, pdl, stream);
}

}  // namespace ingine::kernels::dense

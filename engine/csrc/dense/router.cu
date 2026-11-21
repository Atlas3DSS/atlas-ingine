// router: FP32 logits, sigmoid, biased top-8, normalised x8 weights, for M = 1..8 tokens.
//
// Design. The router matrix is small (128 x 2304: 1.2 MB FP32 / 0.6 MB BF16) but top-8 needs all 128
// logits of a token, so the op is one cluster of CL = 16 CTAs (8 if the device cannot co-schedule 16):
// * every CTA copies the M BF16 activation rows into shared memory (cp.async, issued first), then CTA r
//   loads its rows [r * 128/CL, +128/CL) whole into registers (one warp per row, 16-byte streaming loads:
//   FP32 18 per lane, BF16 9), so the weight stream never waits on activation latency. Spreading the
//   1.2 MB over 16 SMs keeps per-SM traffic at ~74 KB.
// * the M logits of a row are reduce-scattered over the warp (lane m holds token m) and pushed through
//   distributed shared memory to CTA m, which owns token m's selection; one release/acquire cluster
//   barrier publishes them (a relaxed arrive at kernel start + wait before the remote stores guarantees
//   every CTA is resident).
// * selection in CTA m: thread e computes v_e = sigmoid(logit_e) + bias_e and its rank among the 128
//   values (#greater + #equal with a lower index: ties go to the lower expert) with broadcast shared
//   reads, so the 8 winners land in rank order (descending score) without a serial argmax chain; lanes
//   0..7 then write idx and wts = s / (sum s + 1e-20) * 8.
// * pdl=True: weights and bias are loaded before griddepcontrol.wait, the activations after it.
#include <cooperative_groups.h>
#include <torch/extension.h>

#include "cluster.cuh"
#include "dense.cuh"

namespace cg = cooperative_groups;

namespace dense {
namespace {

constexpr int EXPERTS = 128, TOPK = 8;

template <int CL, bool WF32, int M>
__global__ void __launch_bounds__(32 * EXPERTS / CL) router_kernel(const __nv_bfloat16* __restrict__ x,
                                                                   const void* __restrict__ w,
                                                                   const float* __restrict__ bias,
                                                                   int* __restrict__ idx, float* __restrict__ wts) {
  constexpr int RPC = EXPERTS / CL, NT = 32 * RPC;
  constexpr int EPC = WF32 ? 4 : 8;  // weights per 16-byte chunk
  constexpr int CH = H / (EPC * 32);
  __shared__ __align__(16) __nv_bfloat16 xs[M][H];
  __shared__ __align__(16) float logits[EXPERTS];  // token `rank`'s logits (CTAs rank < M)
  __shared__ __align__(16) float sv[EXPERTS];      // sigmoid + bias
  __shared__ float sel_s[TOPK];
  __shared__ int sel_e[TOPK];

  cluster_arrive_relaxed();
  cg::cluster_group cluster = cg::this_cluster();
  const int rank = cluster.block_rank();
  const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
  const int row = rank * RPC + warp;

  // weights and bias (constants) before the PDL wait, activations after it
  const uint8_t* wr = reinterpret_cast<const uint8_t*>(w) + size_t(row) * H * (WF32 ? 4 : 2);
  uint4 wv[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) wv[i] = ld_stream(wr + (size_t(i) * 32 + lane) * 16);
  const float b = t < EXPERTS ? __ldg(bias + t) : 0.f;  // for the selection phase

  pdl_wait();
  pdl_trigger();
  for (int k = t; k < M * H / 8; k += NT) cp_async16(&xs[0][0] + k * 8, x + size_t(k) * 8);
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_all;" ::: "memory");
  __syncthreads();

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; ++m) acc[m] = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = (i * 32 + lane) * EPC;
    float wf[EPC], xf[EPC];
    if constexpr (WF32) {
      wf[0] = __uint_as_float(wv[i].x); wf[1] = __uint_as_float(wv[i].y);
      wf[2] = __uint_as_float(wv[i].z); wf[3] = __uint_as_float(wv[i].w);
    } else {
      unpack_bf16x8(wv[i], wf);
    }
#pragma unroll
    for (int m = 0; m < M; ++m) {
      if constexpr (WF32) {
        const uint2 u = *reinterpret_cast<const uint2*>(&xs[m][col]);
        xf[0] = bf_lo(u.x); xf[1] = bf_hi(u.x); xf[2] = bf_lo(u.y); xf[3] = bf_hi(u.y);
      } else {
        unpack_bf16x8(*reinterpret_cast<const uint4*>(&xs[m][col]), xf);
      }
      float a = acc[m];
#pragma unroll
      for (int j = 0; j < EPC; ++j) a = fmaf(wf[j], xf[j], a);
      acc[m] = a;
    }
  }
  const float logit = reduce_scatter<32, M>(acc);  // lane m: token m

  cluster_wait();  // every CTA has started: remote shared memory is live
  if (lane < M) cluster.map_shared_rank(&logits[0], lane)[row] = logit;
  cluster_arrive();
  cluster_wait();
  if (rank >= M) return;

  const int m = rank;
  float s = 0.f, v = 0.f;
  if (t < EXPERTS) {
    s = 1.f / (1.f + expf(-logits[t]));
    v = s + b;
    sv[t] = v;
  }
  __syncthreads();
  if (t < EXPERTS) {
    int r = 0;
#pragma unroll 8
    for (int j = 0; j < EXPERTS; j += 4) {
      const float4 o = *reinterpret_cast<const float4*>(&sv[j]);
      r += (o.x > v) + (o.y > v) + (o.z > v) + (o.w > v);
      r += (o.x == v && j < t) + (o.y == v && j + 1 < t) + (o.z == v && j + 2 < t) + (o.w == v && j + 3 < t);
    }
    if (r < TOPK) {
      sel_e[r] = t;
      sel_s[r] = s;
    }
  }
  __syncthreads();
  if (t < TOPK) {
    const float my_s = sel_s[t];
    float sum = my_s;
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffu, sum, o);
    idx[m * TOPK + t] = sel_e[t];
    wts[m * TOPK + t] = my_s / (sum + 1e-20f) * 8.f;
  }
}

template <int CL, bool WF32>
void launch_router(int M, const __nv_bfloat16* x, const void* w, const float* bias, int* idx, float* wts, bool pdl) {
  auto go = [&](auto mc) {
    constexpr int MM = decltype(mc)::value;
    launch_ex(router_kernel<CL, WF32, MM>, dim3(CL), 32 * EXPERTS / CL, 0, CL, pdl, x, w, bias, idx, wts);
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
    default: TORCH_CHECK(false, "M must be in 1..8");
  }
}

// 16-CTA clusters need the non-portable opt-in and a GPC with 16 free SMs; fall back to 8 otherwise.
int router_cluster_size() {
  static const int cl = [] {
    auto k = router_kernel<16, true, 8>;
    if (cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1) != cudaSuccess) {
      (void)cudaGetLastError();
      return 8;
    }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(16);
    cfg.blockDim = dim3(32 * EXPERTS / 16);
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 16; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    int n = 0;
    if (cudaOccupancyMaxActiveClusters(&n, k, &cfg) != cudaSuccess) { (void)cudaGetLastError(); n = 0; }
    return n > 0 ? 16 : 8;
  }();
  return cl;
}

}  // namespace

void router(torch::Tensor x, torch::Tensor w, torch::Tensor bias, torch::Tensor idx, torch::Tensor wts, bool pdl) {
  for (auto* t : {&x, &w, &bias, &idx, &wts})
    TORCH_CHECK(t->is_cuda() && t->is_contiguous(), "router: tensors must be contiguous CUDA tensors");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(1) == H, "x must be bf16 [M, 2304]");
  const int M = x.size(0);
  const bool f32 = w.scalar_type() == at::kFloat;
  TORCH_CHECK(f32 || w.scalar_type() == at::kBFloat16, "w must be fp32 or bf16");
  TORCH_CHECK(w.dim() == 2 && w.size(0) == EXPERTS && w.size(1) == H, "w must be [128, 2304]");
  TORCH_CHECK(bias.scalar_type() == at::kFloat && bias.numel() == EXPERTS, "bias must be fp32 [128]");
  TORCH_CHECK(idx.scalar_type() == at::kInt && idx.numel() == M * TOPK, "idx must be int32 [M, 8]");
  TORCH_CHECK(wts.scalar_type() == at::kFloat && wts.numel() == M * TOPK, "wts must be fp32 [M, 8]");
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const float* b = bias.data_ptr<float>();
  int* ip = idx.data_ptr<int>();
  float* wp = wts.data_ptr<float>();
  const bool c16 = router_cluster_size() == 16;
  if (f32) {
    if (c16) launch_router<16, true>(M, xp, w.data_ptr(), b, ip, wp, pdl);
    else launch_router<8, true>(M, xp, w.data_ptr(), b, ip, wp, pdl);
  } else {
    if (c16) launch_router<16, false>(M, xp, w.data_ptr(), b, ip, wp, pdl);
    else launch_router<8, false>(M, xp, w.data_ptr(), b, ip, wp, pdl);
  }
}

}  // namespace dense

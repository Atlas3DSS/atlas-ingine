// Lightweight persona/mech-interp helpers in the dense kernel area:
//   - residual_steer: probe-only additive steering on FP32 residual write buffers [M, H] (d_2027 guarded)
//   - tap_ffn_inp:    read-only capture of the primary MoE operand xm [M, H] for parity checks
#include "kernels/dense/dense.cuh"
#include "kernels/dense/dense.h"

#include "core/device.h"

namespace ingine::kernels::dense {
namespace {

__global__ void residual_steer_kernel(float* __restrict__ y, int M,
                                      const persona::ResidualSteerSite* __restrict__ site) {
  if (!site) return;
  const float scale = site->scale;
  const float* vec = site->vec;
  if (scale == 0.0f || !vec) return;
  const int m = blockIdx.x;
  if (m >= M) return;
  float* row = y + std::size_t(m) * H;
  for (int c = threadIdx.x; c < H; c += blockDim.x) {
    if (c == persona::kForbiddenCoord) continue;  // standing invariant: d_2027 is never perturbed
    const float v = vec[c];
    if (v != 0.0f) {
      row[c] = __fadd_rn(row[c], __fmul_rn(scale, v));
    }
  }
}

__global__ void tap_ffn_inp_kernel(const ActT* __restrict__ xm, int M,
                                   const persona::FfnInpTapVisit* __restrict__ tap,
                                   const std::int64_t* __restrict__ pos_dev, int pos0) {
  if (!tap || !tap->enabled || (!tap->out && !tap->last_out)) return;
  const int m = blockIdx.x;
  if (m >= M) return;
  const int tok_pos = pos_dev ? int(pos_dev[0]) + m : pos0 + m;
  const ActT* src = xm + std::size_t(m) * H;
  if (tap->out && tok_pos >= 0 && tok_pos < tap->max_pos) {
    float* dst = tap->out + std::size_t(tok_pos) * H;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
      dst[c] = act::to_float(src[c]);
    }
  }
  if (tap->last_out && m == M - 1) {
    float* dst = tap->last_out;
    for (int c = threadIdx.x; c < H; c += blockDim.x) {
      dst[c] = act::to_float(src[c]);
    }
  }
}

}  // namespace

void residual_steer(float* y, int M, const persona::ResidualSteerSite* site, cudaStream_t stream) {
  if (!site || M <= 0) return;
  residual_steer_kernel<<<M, 256, 0, stream>>>(y, M, site);
  CUDA_CHECK(cudaGetLastError());
}

void tap_ffn_inp(const ActT* xm, int M, const persona::FfnInpTapVisit* tap, const std::int64_t* pos_dev, int pos0,
                 cudaStream_t stream) {
  if (!tap || M <= 0) return;
  tap_ffn_inp_kernel<<<M, 256, 0, stream>>>(xm, M, tap, pos_dev, pos0);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::dense

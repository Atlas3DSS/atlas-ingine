// Pure native AMD ROCm/HIP Q4_0 SoA embedding + RMSNorm and batch decode kernels.
// Uses portable warp32 / CDNA wave64 logical-32 reductions (kernels/common/warp.cuh) and
// canonical Q4_0 bit semantics matching CUDA and CPU oracles identically. Zero CUDA dependencies.
#include "core/act.h"
#include "core/ops.h"
#include "kernels/common/arch_ops.cuh"
#include "kernels/common/warp.cuh"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ingine::kernels::hip_dense {
namespace {

constexpr int kInklingHidden = 2304;
constexpr float kInklingEps = 1e-6f;
constexpr int kBlockThreads = 256;

__device__ __forceinline__ float decode_round_act(float x, int rounding_mode) noexcept {
  switch (rounding_mode) {
    case static_cast<int>(ops::ActRounding::Bf16Rn):
      return act::hip_round_bf16(x);
    case static_cast<int>(ops::ActRounding::Fp16RnSat): {
      const float clamped = isfinite(x) ? fminf(fmaxf(x, -act::kFp16Max), act::kFp16Max) : x;
      return __half2float(__float2half_rn(clamped));
    }
    default:
      return x;
  }
}

__device__ __forceinline__ float4 decode_q4_0_x4(std::uint16_t b2, std::uint16_t scale_bits) noexcept {
  return make_float4(arch::decode_q4_0_nibble_canonical(static_cast<std::uint8_t>(b2 & 0xfu), scale_bits),
                     arch::decode_q4_0_nibble_canonical(static_cast<std::uint8_t>((b2 >> 4) & 0xfu), scale_bits),
                     arch::decode_q4_0_nibble_canonical(static_cast<std::uint8_t>((b2 >> 8) & 0xfu), scale_bits),
                     arch::decode_q4_0_nibble_canonical(static_cast<std::uint8_t>((b2 >> 12) & 0xfu), scale_bits));
}

__global__ void decode_q4_0_hip_kernel(const std::uint8_t* __restrict__ table, int hidden_size,
                                       float* __restrict__ out) {
  const int row_idx = static_cast<int>(blockIdx.x);
  const int t = static_cast<int>(threadIdx.x);
  const int nv = hidden_size / 4;
  const std::size_t nib_bytes = static_cast<std::size_t>(hidden_size) / 2u;
  const std::size_t row_bytes = nib_bytes + static_cast<std::size_t>(hidden_size / 32) * 2u;
  const std::uint8_t* row = table + static_cast<std::size_t>(row_idx) * row_bytes;
  const std::uint8_t* scales = row + nib_bytes;
  float* out_row = out + static_cast<std::size_t>(row_idx) * static_cast<std::size_t>(hidden_size);
  for (int idx = t; idx < nv; idx += static_cast<int>(blockDim.x)) {
    const std::uint16_t b2 = static_cast<std::uint16_t>(row[2 * idx]) |
                             (static_cast<std::uint16_t>(row[2 * idx + 1]) << 8);
    const std::uint16_t sb = static_cast<std::uint16_t>(scales[2 * (idx >> 3)]) |
                             (static_cast<std::uint16_t>(scales[2 * (idx >> 3) + 1]) << 8);
    const float4 v = decode_q4_0_x4(b2, sb);
    out_row[4 * idx + 0] = v.x;
    out_row[4 * idx + 1] = v.y;
    out_row[4 * idx + 2] = v.z;
    out_row[4 * idx + 3] = v.w;
  }
}

template <int BW>
__global__ void embed_q4_hip_kernel(const std::int64_t* __restrict__ token, int vocab_size,
                                    const std::uint8_t* __restrict__ table,
                                    const float* __restrict__ norm_w, float* __restrict__ out,
                                    std::uint32_t* __restrict__ oob_flag, int rounding_mode) {
  constexpr int NV = kInklingHidden / 4;
  constexpr int PER = (NV + BW - 1) / BW;
  constexpr std::size_t NIB_BYTES = kInklingHidden / 2;
  constexpr std::size_t ROW_BYTES = NIB_BYTES + (kInklingHidden / 32) * 2;
  __shared__ float part[BW];
  __shared__ float rinv_s;
  const int m = static_cast<int>(blockIdx.x);
  const int t = static_cast<int>(threadIdx.x);

  float4 g[PER];
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    if (t + k * BW < NV) {
      g[k] = *reinterpret_cast<const float4*>(norm_w + 4 * (t + k * BW));
    }
  }

  const std::int64_t tok = token[m];
  if (tok < 0 || tok >= vocab_size) {
    if (t == 0 && oob_flag != nullptr) atomicExch(oob_flag, 1u);
    return;
  }

  const std::uint8_t* row = table + static_cast<std::size_t>(tok) * ROW_BYTES;
  const std::uint8_t* scales = row + NIB_BYTES;
  float4 v[PER];
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int idx = t + k * BW;
    if (idx < NV) {
      const std::uint16_t b2 = static_cast<std::uint16_t>(row[2 * idx]) |
                               (static_cast<std::uint16_t>(row[2 * idx + 1]) << 8);
      const std::uint16_t sb = static_cast<std::uint16_t>(scales[2 * (idx >> 3)]) |
                               (static_cast<std::uint16_t>(scales[2 * (idx >> 3) + 1]) << 8);
      v[k] = decode_q4_0_x4(b2, sb);
      acc[0] += v[k].x * v[k].x;
      acc[1] += v[k].y * v[k].y;
      acc[2] += v[k].z * v[k].z;
      acc[3] += v[k].w * v[k].w;
    }
  }

  float s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  part[t] = s;
#pragma unroll
  for (int o = BW / 2; o >= 32; o >>= 1) {
    __syncthreads();
    if (t < o) {
      s += part[t + o];
      part[t] = s;
    }
  }
  __syncthreads();
  if (t < 32) {
    s = warp::reduce_sum_32(s);
    if (t == 0) rinv_s = rsqrtf(s * (1.0f / static_cast<float>(kInklingHidden)) + kInklingEps);
  }
  __syncthreads();

  const float rinv = rinv_s;
  float* o = out + static_cast<std::size_t>(m) * static_cast<std::size_t>(kInklingHidden);
#pragma unroll
  for (int k = 0; k < PER; ++k) {
    const int idx = t + k * BW;
    if (idx < NV) {
      const float4 x = v[k];
      const float4 w = g[k];
      *reinterpret_cast<float4*>(o + 4 * idx) = make_float4(
          decode_round_act(x.x * rinv * w.x, rounding_mode), decode_round_act(x.y * rinv * w.y, rounding_mode),
          decode_round_act(x.z * rinv * w.z, rounding_mode), decode_round_act(x.w * rinv * w.w, rounding_mode));
    }
  }
}

void check_hip(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorName(err) + ": " + hipGetErrorString(err));
  }
}

}  // namespace

void decode_q4_0_hip(const void* table_q4, int rows, int hidden_size, float* out_f32, hipStream_t stream) {
  hipLaunchKernelGGL(decode_q4_0_hip_kernel, dim3(static_cast<unsigned>(rows)), dim3(kBlockThreads), 0, stream,
                     static_cast<const std::uint8_t*>(table_q4), hidden_size, out_f32);
  check_hip(hipGetLastError(), "decode_q4_0_hip_kernel launch");
}

void embed_q4_hip(const std::int64_t* token_ids, int tokens, int vocab_size, const void* table_q4,
                  const float* norm_weight, float* out, ops::ActRounding rounding, hipStream_t stream,
                  std::uint32_t* oob_flag) {
  hipLaunchKernelGGL((embed_q4_hip_kernel<kBlockThreads>), dim3(static_cast<unsigned>(tokens)), dim3(kBlockThreads), 0,
                     stream, token_ids, vocab_size, static_cast<const std::uint8_t*>(table_q4), norm_weight, out,
                     oob_flag, static_cast<int>(rounding));
  check_hip(hipGetLastError(), "embed_q4_hip_kernel launch");
}

}  // namespace ingine::kernels::hip_dense

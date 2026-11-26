// Portable wave32 / logical-32 half-wavefront primitives across CUDA, AMD ROCm/HIP (RDNA wave32 &
// CDNA wave64 compat), and host verification helpers.
// Incorporates wave64 half-wavefront ballot shift & RDNA3/4 WMMA patterns from Strata / Project Maya (MIT).
#pragma once

#include <cstdint>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace ingine::kernels::warp {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define INGINE_WARP_HD __host__ __device__ __forceinline__
#else
#define INGINE_WARP_HD inline
#endif

inline constexpr std::uint32_t kFullWave32Mask = 0xffffffffu;

// Shifts a 64-lane CDNA ballot down by 32 when executing in the upper half-wavefront, yielding the local 32-lane mask.
INGINE_WARP_HD std::uint32_t logical_32_half_ballot(std::uint64_t wave64_ballot, int lane_id) noexcept {
  return static_cast<std::uint32_t>(wave64_ballot >> (lane_id & 32));
}

#if defined(__CUDACC__) && !defined(__HIPCC__)

__device__ __forceinline__ int lane_id_32() noexcept { return static_cast<int>(threadIdx.x & 31u); }

__device__ __forceinline__ void sync_32() noexcept { __syncwarp(kFullWave32Mask); }

__device__ __forceinline__ float shfl_xor_f32(float val, int mask) noexcept {
  return __shfl_xor_sync(kFullWave32Mask, val, mask, 32);
}

__device__ __forceinline__ float shfl_down_f32(float val, int delta) noexcept {
  return __shfl_down_sync(kFullWave32Mask, val, delta, 32);
}

__device__ __forceinline__ std::uint32_t ballot_32(int pred) noexcept {
  return __ballot_sync(kFullWave32Mask, pred);
}

#elif defined(__HIPCC__)

__device__ __forceinline__ int lane_id_32() noexcept { return static_cast<int>(threadIdx.x & 31u); }

__device__ __forceinline__ void sync_32() noexcept {
#if defined(__HIP_DEVICE_COMPILE__)
#ifdef INGINE_HIP_WAVE64
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#elif defined(HIP_VERSION_MAJOR) && (HIP_VERSION_MAJOR >= 7)
  __syncwarp(kFullWave32Mask);
#else
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
#endif
#endif
}

__device__ __forceinline__ float shfl_xor_f32(float val, int mask) noexcept { return __shfl_xor(val, mask, 32); }

__device__ __forceinline__ float shfl_down_f32(float val, int delta) noexcept { return __shfl_down(val, delta, 32); }

__device__ __forceinline__ std::uint32_t ballot_32(int pred) noexcept {
#ifdef INGINE_HIP_WAVE64
  return logical_32_half_ballot(static_cast<std::uint64_t>(__ballot(pred)), static_cast<int>(__lane_id()));
#else
  return static_cast<std::uint32_t>(__ballot(pred));
#endif
}

#if defined(__GFX11__) || defined(__GFX12__)
using wmma_h16 = _Float16 __attribute__((ext_vector_type(16)));
using wmma_f8 = float __attribute__((ext_vector_type(8)));

__device__ __forceinline__ wmma_f8 wmma_f16(wmma_h16 a, wmma_h16 b, wmma_f8 c) noexcept {
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__GFX12__)
  using h8 = _Float16 __attribute__((ext_vector_type(8)));
  const int k0 = static_cast<int>((threadIdx.x & 31u) / 16u) * 8;
  h8 aa, bb;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    aa[i] = a[k0 + i];
    bb[i] = b[k0 + i];
  }
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(aa, bb, c);
#else
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
#else
  (void)a;
  (void)b;
  return c;
#endif
}

__device__ __forceinline__ void lds_sync() noexcept {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
#endif
}
#endif  // __GFX11__ || __GFX12__

#endif  // device dialects

#if defined(__CUDACC__) || defined(__HIPCC__)
__device__ __forceinline__ float reduce_sum_32(float val) noexcept {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) val += shfl_down_f32(val, o);
  return val;
}

__device__ __forceinline__ float allreduce_sum_32(float val) noexcept {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) val += shfl_xor_f32(val, mask);
  return val;
}

__device__ __forceinline__ float allreduce_max_32(float val) noexcept {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    const float other = shfl_xor_f32(val, mask);
    val = (other > val) ? other : val;
  }
  return val;
}
#endif

#undef INGINE_WARP_HD

}  // namespace ingine::kernels::warp
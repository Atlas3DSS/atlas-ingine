// Device helpers shared by the DSpark draft and speculative-acceptance kernels.
#pragma once

#include "core/act.h"
#include "core/device.h"
#include "core/gpu_compat.h"
#include "kernels/common/pdl.cuh"
#include "kernels/draft/rng.h"

#include <cstdint>
#include <utility>

namespace ingine::kernels::draft {

__device__ __forceinline__ float bf_round(float x) { return act::round_act(x); }
__device__ __forceinline__ float bf(const ActT& x) { return act::to_float(x); }

// One element of a key/value cache: BF16, or E4M3 without a scale (nearest even, saturating at +-448).
__device__ __forceinline__ void store_cache(void* cache, size_t at, float x, bool fp8) {
  if (fp8) static_cast<uint8_t*>(cache)[at] = ingine::gpu::float_to_e4m3fn_sat(x);
  else static_cast<ActT*>(cache)[at] = act::from_float(x);
}
// 2 E4M3 (low 16 bits of u) -> Act2T, exact across CUDA (sm_70..sm_120a) and HIP.
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t u) {
  return act::e4m3x2_to_act2_bits(u);
}

// 8 BF16 values (one uint4) -> 8 floats.
__device__ __forceinline__ void unpack8(const uint4& v, float* f) {
  act::unpack8(v, f);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Block-wide sum (blockDim.x a multiple of 32, <= 1024); every thread gets the total. `scratch` holds 33
// floats; contains two barriers, so consecutive calls may reuse it.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v);
  __syncthreads();  // a previous call's readers are done with scratch
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float t = 0.f;
  for (int w = 0; w < nw; ++w) t += scratch[w];
  return t;
}
__device__ __forceinline__ float block_max(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_max(v);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float t = -INFINITY;
  for (int w = 0; w < nw; ++w) t = fmaxf(t, scratch[w]);
  return t;
}

// Order-preserving map of a float to uint32 (larger float -> larger key; -0 sorts below +0).
__device__ __forceinline__ uint32_t float_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// (score, index) packed so that atomicMax / max picks the largest score and, among equal scores, the
// lowest index (torch.argmax returns the first maximum).
__device__ __forceinline__ unsigned long long pack_argmax(float score, uint32_t index) {
  return (static_cast<unsigned long long>(float_key(score)) << 32) | (0xffffffffu - index);
}
__device__ __forceinline__ uint32_t unpack_argmax_index(unsigned long long p) {
  return 0xffffffffu - static_cast<uint32_t>(p & 0xffffffffu);
}

// 16-byte weight load issued ahead of pdl_wait(): ld.global.cg in asm volatile, which ptxas keeps above the
// wait (it may sink .nc loads below it, see kernels/common/pdl.cuh), so the weights are in flight while the
// predecessor finishes.
__device__ __forceinline__ uint4 ld_weight(const void* p) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  uint4 r;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
#else
  return *static_cast<const uint4*>(p);
#endif
}

// Launches `kernel` as a programmatic dependent of its stream predecessor (kernels/common/pdl.cuh). The
// kernel calls pdl_trigger() first and pdl_wait() before it touches data another kernel writes.
template <class... Params, class... Args>
void launch_pdl(void (*kernel)(Params...), dim3 grid, dim3 block, cudaStream_t stream, Args&&... args) {
  CUDA_CHECK(pdl_launch(kernel, grid, block, 0, stream, std::forward<Args>(args)...));
}

}  // namespace ingine::kernels::draft

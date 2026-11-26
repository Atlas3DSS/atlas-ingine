// Global-to-shared staging. Callers supply 4- or 16-byte aligned addresses and
// synchronize participating threads before reading a tile produced by peers.
#pragma once

#include "core/gpu_compat.h"

namespace ingine::kernels::copy {

template <int Bytes>
__device__ __forceinline__ void sync(void* dst, const void* src, int valid = Bytes) {
  static_assert(Bytes == 4 || Bytes == 16);
  if (valid == Bytes || valid == 0) {
    if constexpr (Bytes == 16) {
      *static_cast<uint4*>(dst) = valid ? *static_cast<const uint4*>(src) : uint4{};
    } else {
      *static_cast<unsigned*>(dst) = valid ? *static_cast<const unsigned*>(src) : 0u;
    }
    return;
  }
  auto* out = static_cast<unsigned char*>(dst);
  const auto* in = static_cast<const unsigned char*>(src);
#pragma unroll
  for (int i = 0; i < Bytes; ++i) out[i] = i < valid ? in[i] : 0;
}

__device__ __forceinline__ void commit() {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <int Groups>
__device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;" :: "n"(Groups) : "memory");
#endif
}

__device__ __forceinline__ void wait_all() {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;" ::: "memory");
#endif
}

}  // namespace ingine::kernels::copy

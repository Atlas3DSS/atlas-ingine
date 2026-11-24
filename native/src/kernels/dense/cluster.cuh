// Launch helpers of the `dense` kernels: split cluster barriers, programmatic dependent launch (PDL),
// cluster / PDL kernel launch. Ported from engine/csrc/dense/cluster.cuh (torch stream -> cudaStream_t).
#pragma once

#include "core/device.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace ingine::kernels::dense {

// Host-side argument check (replaces TORCH_CHECK).
inline void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("dense: ") + what);
}

// Split cluster barrier (SM90+ only; no-ops on SM8x which use non-cluster fallbacks).
__device__ __forceinline__ void cluster_arrive_relaxed() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
#endif
}
__device__ __forceinline__ void cluster_arrive() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
#endif
}
__device__ __forceinline__ void cluster_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
}

// Programmatic dependent launch (SM90+ only; no-ops on SM8x).
__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void pdl_wait_fenced(uint32_t zero) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#pragma unroll 1
  for (uint32_t i = 0; i <= zero; ++i) asm volatile("griddepcontrol.wait;" ::: "memory");
#else
  static_cast<void>(zero);
#endif
}

// Launches `kernel` on `stream` (graph capturable): as clusters of `cl` CTAs if cl > 1 (SM90+), with
// the programmatic-stream-serialization attribute if `pdl` (SM90+).
template <class... KArgs, class... Args>
void launch_ex(void (*kernel)(KArgs...), dim3 grid, int block, int smem, int cl, bool pdl, cudaStream_t stream,
               Args&&... args) {
  const bool sm90 = current_sm_major() >= 9;
  if (sm90 && cl > 8) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  if (smem > 0) CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[2];
  int n = 0;
  if (sm90 && cl > 1) {
    attr[n].id = cudaLaunchAttributeClusterDimension;
    attr[n].val.clusterDim.x = cl;
    attr[n].val.clusterDim.y = 1;
    attr[n].val.clusterDim.z = 1;
    ++n;
  }
  if (sm90 && pdl) {
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  cfg.attrs = attr;
  cfg.numAttrs = n;
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

}  // namespace ingine::kernels::dense

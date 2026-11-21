// Launch helpers of the `dense` extension: split cluster barriers, programmatic dependent launch (PDL),
// cluster / PDL kernel launch.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace dense {

// Split cluster barrier. A relaxed arrive at kernel start + wait before the first remote shared-memory
// store guarantees every CTA of the cluster is running; release/acquire pairs publish remote stores.
__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }

// Programmatic dependent launch. Every dense kernel loads only weights/constants before pdl_wait() and
// every activation/state input, and every write, after it; pdl_wait() returns once the preceding grid has
// completed and its writes are visible. pdl_trigger() lets the next kernel of the stream start its own
// prologue (it still waits for this grid's completion before reading our outputs). Both are no-ops when
// the kernel was not launched with the PDL attribute.
__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_trigger() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }

// Launches `kernel` on the current stream (graph capturable): as clusters of `cl` CTAs if cl > 1, with
// the programmatic-stream-serialization attribute if `pdl` (captured as a programmatic graph edge).
// Cluster sizes above 8 need the non-portable opt-in; shared memory above 48 KB the dynamic-size opt-in.
template <class... KArgs, class... Args>
void launch_ex(void (*kernel)(KArgs...), dim3 grid, int block, int smem, int cl, bool pdl, Args&&... args) {
  if (cl > 8) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
  // static + dynamic shared memory above 48 KB needs the opt-in, whatever the split
  if (smem > 0) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[2];
  int n = 0;
  if (cl > 1) {
    attr[n].id = cudaLaunchAttributeClusterDimension;
    attr[n].val.clusterDim.x = cl;
    attr[n].val.clusterDim.y = 1;
    attr[n].val.clusterDim.z = 1;
    ++n;
  }
  if (pdl) {
    attr[n].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[n].val.programmaticStreamSerializationAllowed = 1;
    ++n;
  }
  cfg.attrs = attr;
  cfg.numAttrs = n;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...));
}

}  // namespace dense

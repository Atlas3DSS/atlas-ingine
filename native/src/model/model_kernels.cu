#include "model/model_kernels.h"

#include "core/device.h"

namespace ingine::model {
namespace {

__global__ void f32_to_bf16_rows_kernel(const float* __restrict__ src, int rows, int cols,
                                        __nv_bfloat16* __restrict__ dst, long long dst_stride) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= (long long)rows * cols) return;
  const long long r = i / cols, c = i % cols;
  dst[r * dst_stride + c] = __float2bfloat16_rn(src[i]);
}

}  // namespace

void f32_to_bf16_rows(const float* src, int rows, int cols, __nv_bfloat16* dst, std::int64_t dst_stride,
                      cudaStream_t stream) {
  const long long n = (long long)rows * cols;
  if (n == 0) return;
  f32_to_bf16_rows_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(src, rows, cols, dst, dst_stride);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::model

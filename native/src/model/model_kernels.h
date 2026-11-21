// Small device helpers of the model runtime (not part of the ported decode kernels).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::model {

// dst[r * dst_stride + c] = bf16(src[r * cols + c]) (round to nearest even, torch copy_ to BF16).
void f32_to_bf16_rows(const float* src, int rows, int cols, __nv_bfloat16* dst, std::int64_t dst_stride,
                      cudaStream_t stream);

}  // namespace ingine::model

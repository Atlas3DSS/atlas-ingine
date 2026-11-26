// Routed-expert kernels (the PyTorch extension's kernels ported to raw-pointer launchers; contracts in
// native/KERNELS.md, area `moe`) and the expert decode to BF16. Launchers run on `stream`, never
// allocate or synchronise, read routing from device memory and are CUDA-graph capturable. Shapes are
// row-major (Python order); H = 2304, I = 1152, 128 experts, top-8. Invalid arguments throw
// std::invalid_argument.
#pragma once

#include "core/act.h"
#include "kernels/common/persona_abi.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::moe {

enum class Format : int { NVFP4 = 0, MXFP8 = 1, ATLAS2 = 2, ATLAS3 = 3, EXL3_2 = 4, EXL3_3 = 5, GGUF_Q5L = 6, GGUF_IQ4 = 7, GGUF_Q4L = 8 };

// Optional per-expert device pointer table ([128] base pointers for one layer) for tiered expert offloading.
// When null, kernels index contiguous [128, ...] weight arrays as before.
struct ExpertPtrs {
  const std::uint8_t* const* w13 = nullptr;
  const std::uint8_t* const* w13_scale = nullptr;
  const std::uint8_t* const* w2 = nullptr;
  const std::uint8_t* const* w2_scale = nullptr;
  const std::uint8_t* const* w13_rout = nullptr;  // F16 [2I] per expert for EXL3
  const std::uint8_t* const* w2_rout = nullptr;   // F16 [H] per expert for EXL3
};

// moe_up: W13 GEMV of the selected experts + softcapped SwiGLU. x ActT [M, H], idx int32 [M, 8],
// out ActT [M, 8, I]. NVFP4: w13 u8 [128, 2I, H/2], w13_scale e4m3 [128, 2I, H/16], w13_factor fp32 [128];
// MXFP8: w13 e4m3 [128, 2I, H], w13_scale u8 E8M0 [128, 2I, H/32], factor unused (may be null).
// EXL3_2 / EXL3_3: w13 u8 [128, 144, 144, 32*bits] (stacked [gate; up]), w13_scale = w13_rin f16 [128, H],
// w13_rout = f16 [128, 2I], w13_factor unused.
// pdl: the kernel waits for its predecessor before reading anything.
void moe_up(const ActT* x, int M, const int* idx, const void* w13, const void* w13_scale,
            const float* w13_factor, Format fmt, ActT* out, bool pdl, cudaStream_t stream,
            const void* w13_rout = nullptr, const ExpertPtrs* ptrs = nullptr);

// moe_down: W2 GEMV + routing-weighted sum + post-MLP conv with state update. act ActT [M, 8, I],
// wts fp32 [M, 8], w2 [128, H, I(/2)] + scales [128, H, I/16 | I/32] (+ factor), conv_w fp32 [H, 4],
// conv_state fp32 [3, H], out fp32 [M, H], conv_hist fp32 [M, H] (ActT-rounded conv inputs) or null.
// EXL3_2 / EXL3_3: w2 u8 [128, 144, 72, 32*bits], w2_scale = w2_rin f16 [128, I], w2_rout = f16 [128, H],
// workspace = ActT [M * 8 * H] scratch buffer.
// pdl: valid only right after moe_up (idx is read before the PDL wait).
void moe_down(const ActT* act, int M, const int* idx, const float* wts, const void* w2,
              const void* w2_scale, const float* w2_factor, Format fmt, const float* conv_w, float* conv_state,
              float* out, float* conv_hist, bool pdl, cudaStream_t stream,
              const void* w2_rout = nullptr, ActT* workspace = nullptr, const ExpertPtrs* ptrs = nullptr);

// dequant_experts: decode experts[0..n) (int32 device) of one weight ([128, rows, cols] in checkpoint
// encoding: NVFP4 codes u8 [128, rows, cols/2] + e4m3 scales [128, rows, cols/16] + factor fp32 [128];
// MXFP8 codes e4m3 [128, rows, cols] + E8M0 scales [128, rows, cols/32];
// EXL3_2/EXL3_3 codes u8 [128, rows/16, cols/16, 32*bits], scales = rin f16 [128, cols], rout f16 [128, rows])
// into out ActT [n, rows, cols], exactly as model.decode_experts (FP32 product, one ActT rounding).
void dequant_experts(const void* codes, const void* scales, const float* factor, const std::int32_t* experts, int n,
                     int rows, int cols, Format fmt, ActT* out, cudaStream_t stream,
                     const void* rout = nullptr);

// apply_expert_hooks_decode: runs 3b per-neuron affine hooks followed by 3a read-only expert activation taps
// on decode `act` [M, 8, I] between moe_up and moe_down. Fixed-grid, CUDA-graph capturable, zero-write no-op
// when gain == 1.0f && offset == 0.0f.
void apply_expert_hooks_decode(ActT* act, const int* idx, int M, const persona::ExpertTapVisit* hook,
                               const std::int64_t* pos_dev, cudaStream_t stream, int pos_offset = 0);

}  // namespace ingine::kernels::moe

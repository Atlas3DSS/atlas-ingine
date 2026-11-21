// Routed-expert decode to BF16 for prefill's grouped GEMMs.
//
// Prefill multiplies many tokens per expert, so it decodes the selected experts' matrices once into a
// BF16 scratch buffer and runs torch._grouped_mm on it. Decoding follows the checkpoint semantics exactly
// (FP32 product, one BF16 rounding), matching model.decode_experts:
//   NVFP4: value = e2m1(code) * e4m3(block scale per 16) * factor[expert]
//   MXFP8: value = e4m3(code) * 2^(e8m0 per 32 - 127)
// One thread decodes one scale block (16 or 32 values) and writes it with 16-byte stores.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace {

__constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

__global__ void dequant_nvfp4(const uint8_t* __restrict__ codes, const __nv_fp8_e4m3* __restrict__ scales,
                              const float* __restrict__ factor, const int32_t* __restrict__ experts,
                              __nv_bfloat16* __restrict__ out, int64_t blocks_per_expert, int64_t total) {
    int64_t b = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (b >= total) return;
    int64_t slot = b / blocks_per_expert, within = b % blocks_per_expert;
    int64_t e = experts[slot];
    int64_t src = e * blocks_per_expert + within;
    // the reference forms e2m1 * block_scale, then multiplies by the expert factor (both FP32)
    const float scale = float(scales[src]), f = factor[e];
    uint2 c = *reinterpret_cast<const uint2*>(codes + src * 8);
    uint32_t w[2] = {c.x, c.y};
    __align__(16) __nv_bfloat16 v[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t byte = (w[i / 4] >> (8 * (i % 4))) & 0xFF;
        v[2 * i] = __float2bfloat16((kE2M1[byte & 0xF] * scale) * f);
        v[2 * i + 1] = __float2bfloat16((kE2M1[byte >> 4] * scale) * f);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + (slot * blocks_per_expert + within) * 16);
    dst[0] = reinterpret_cast<uint4*>(v)[0];
    dst[1] = reinterpret_cast<uint4*>(v)[1];
}

__global__ void dequant_mxfp8(const __nv_fp8_e4m3* __restrict__ codes, const uint8_t* __restrict__ scales,
                              const int32_t* __restrict__ experts, __nv_bfloat16* __restrict__ out,
                              int64_t blocks_per_expert, int64_t total) {
    int64_t b = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (b >= total) return;
    int64_t slot = b / blocks_per_expert, within = b % blocks_per_expert;
    int64_t src = experts[slot] * blocks_per_expert + within;
    float s = exp2f(float(scales[src]) - 127.f);
    const uint4* c = reinterpret_cast<const uint4*>(codes + src * 32);
    uint4* dst = reinterpret_cast<uint4*>(out + (slot * blocks_per_expert + within) * 32);
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        uint4 raw = c[q];
        const __nv_fp8_e4m3* f = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
        __align__(16) __nv_bfloat16 v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = __float2bfloat16(float(f[i]) * s);
        dst[2 * q] = reinterpret_cast<uint4*>(v)[0];
        dst[2 * q + 1] = reinterpret_cast<uint4*>(v)[1];
    }
}

}  // namespace

// codes [128, R, C/2] uint8 (NVFP4) or [128, R, C] float8_e4m3fn (MXFP8); experts int32 [n];
// out bf16 [n, R, C].
void dequant_experts(torch::Tensor codes, torch::Tensor scales, c10::optional<torch::Tensor> factor,
                     torch::Tensor experts, torch::Tensor out) {
    TORCH_CHECK(out.is_contiguous() && codes.is_contiguous() && scales.is_contiguous());
    int64_t n = experts.numel();
    int64_t per_expert_blocks = scales.size(1) * scales.size(2);
    int64_t total = n * per_expert_blocks;
    auto stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;
    int64_t grid = (total + threads - 1) / threads;
    if (codes.scalar_type() == at::kByte) {
        TORCH_CHECK(factor.has_value());
        dequant_nvfp4<<<grid, threads, 0, stream>>>(
            codes.data_ptr<uint8_t>(), reinterpret_cast<const __nv_fp8_e4m3*>(scales.data_ptr()),
            factor->data_ptr<float>(), experts.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), per_expert_blocks, total);
    } else {
        dequant_mxfp8<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(codes.data_ptr()), scales.data_ptr<uint8_t>(),
            experts.data_ptr<int32_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), per_expert_blocks, total);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_experts", &dequant_experts, "decode routed experts to BF16");
}

#include <atomic>
// Routed-expert decode to BF16 (ported from engine/csrc/prefill/dequant.cu; kernel bodies unchanged).
//
// A prefill that multiplies many tokens per expert can decode the selected experts' matrices once into a
// BF16 scratch buffer and run grouped GEMMs on it. Decoding follows the checkpoint semantics exactly
// (FP32 product, one BF16 rounding), matching model.decode_experts:
//   NVFP4: value = e2m1(code) * e4m3(block scale per 16) * factor[expert]
//   MXFP8: value = e4m3(code) * 2^(e8m0 per 32 - 127)
// One thread decodes one scale block (16 or 32 values) and writes it with 16-byte stores.
#include "kernels/moe/moe.h"

#include "kernels/common/exl3.cuh"
#include "core/device.h"
#include "core/gpu_compat.h"

#include <stdexcept>

namespace ingine::kernels::moe {
namespace {

__constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
__constant__ float kAtlas2[4] = {-1.5f, -0.453125f, 0.453125f, 1.5f};
__constant__ float kAtlas3[8] = {-2.0f, -1.25f, -0.703125f, -0.2265625f, 0.2265625f, 0.703125f, 1.25f, 2.0f};

__global__ void dequant_nvfp4(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                              const float* __restrict__ factor, const int32_t* __restrict__ experts,
                              ActT* __restrict__ out, int64_t blocks_per_expert, int64_t total) {
    int64_t b = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (b >= total) return;
    int64_t slot = b / blocks_per_expert, within = b % blocks_per_expert;
    int64_t e = experts[slot];
    int64_t src = e * blocks_per_expert + within;
    // the reference forms e2m1 * block_scale, then multiplies by the expert factor (both FP32)
    const float scale = ingine::gpu::e4m3fn_to_float(scales[src]), f = factor[e];
    uint2 c = *reinterpret_cast<const uint2*>(codes + src * 8);
    uint32_t w[2] = {c.x, c.y};
    __align__(16) ActT v[16];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t byte = (w[i / 4] >> (8 * (i % 4))) & 0xFF;
        v[2 * i] = act::from_float((kE2M1[byte & 0xF] * scale) * f);
        v[2 * i + 1] = act::from_float((kE2M1[byte >> 4] * scale) * f);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + (slot * blocks_per_expert + within) * 16);
    dst[0] = reinterpret_cast<uint4*>(v)[0];
    dst[1] = reinterpret_cast<uint4*>(v)[1];
}

__global__ void dequant_mxfp8(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                              const int32_t* __restrict__ experts, ActT* __restrict__ out,
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
        const uint8_t* f = reinterpret_cast<const uint8_t*>(&raw);
        __align__(16) ActT v[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = act::from_float(ingine::gpu::e4m3fn_to_float(f[i]) * s);
        dst[2 * q] = reinterpret_cast<uint4*>(v)[0];
        dst[2 * q + 1] = reinterpret_cast<uint4*>(v)[1];
    }
}

__global__ void dequant_atlas2(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                               const float* __restrict__ factor, const int32_t* __restrict__ experts,
                               ActT* __restrict__ out, int64_t blocks_per_expert, int64_t total) {
    int64_t b = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (b >= total) return;
    int64_t slot = b / blocks_per_expert, within = b % blocks_per_expert;
    int64_t e = experts[slot];
    int64_t src = e * blocks_per_expert + within;
    const float scale = ingine::gpu::e4m3fn_to_float(scales[src]), f = factor[e];
    const uint32_t w = *reinterpret_cast<const uint32_t*>(codes + src * 4);
    __align__(16) ActT v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = act::from_float((kAtlas2[(w >> (2 * i)) & 3u] * scale) * f);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + (slot * blocks_per_expert + within) * 16);
    dst[0] = reinterpret_cast<uint4*>(v)[0];
    dst[1] = reinterpret_cast<uint4*>(v)[1];
}

__global__ void dequant_atlas3(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                               const float* __restrict__ factor, const int32_t* __restrict__ experts,
                               ActT* __restrict__ out, int64_t blocks_per_expert, int64_t total) {
    int64_t b = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (b >= total) return;
    int64_t slot = b / blocks_per_expert, within = b % blocks_per_expert;
    int64_t e = experts[slot];
    int64_t src = e * blocks_per_expert + within;
    const float scale = ingine::gpu::e4m3fn_to_float(scales[src]), f = factor[e];
    const int64_t pair = src >> 1;
    const int half = int(src & 1);
    const uint8_t* blk = codes + pair * 12;
    const uint32_t lo = *reinterpret_cast<const uint32_t*>(blk + half * 4);
    const uint16_t hi = *reinterpret_cast<const uint16_t*>(blk + 8 + half * 2);
    __align__(16) ActT v[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const uint32_t code = ((lo >> (2 * i)) & 3u) | (((hi >> i) & 1u) << 2);
        v[i] = act::from_float((kAtlas3[code] * scale) * f);
    }
    uint4* dst = reinterpret_cast<uint4*>(out + (slot * blocks_per_expert + within) * 16);
    dst[0] = reinterpret_cast<uint4*>(v)[0];
    dst[1] = reinterpret_cast<uint4*>(v)[1];
}

// EXL3 / Escha W2 full-matrix decode for unit tests / validation:
// Grid: (rows / 128, cols / 128, n), 128 threads (4 warps) per block.
// Each block decodes one 128_row x 128_col super-tile: W = diag(rout) * H_128 * W_hat^T * H_128 * diag(rin).
// Input codes layout: [128, rows/16, cols/16, 32*BITS] u8; rin f16 [128, cols]; rout f16 [128, rows].
template <int BITS, typename BufT>
__global__ void dequant_exl3_kernel(const uint8_t* __restrict__ codes, const __half* __restrict__ rin,
                                    const __half* __restrict__ rout, const int32_t* __restrict__ experts,
                                    ActT* __restrict__ out, int rows, int cols) {
    constexpr int TILE_BYTES = 32 * BITS;
    extern __shared__ __align__(16) unsigned char tile_raw[];
    BufT* tile_buf = reinterpret_cast<BufT*>(tile_raw);
    auto load_tile = [&](int r, int c) -> float {
        if constexpr (sizeof(BufT) == 4) return tile_buf[r * 128 + c];
        else return __half2float(tile_buf[r * 128 + c]);
    };
    auto store_tile = [&](int r, int c, float x) {
        if constexpr (sizeof(BufT) == 4) tile_buf[r * 128 + c] = x;
        else tile_buf[r * 128 + c] = __float2half_rn(x);
    };
    const int r_blk = blockIdx.x, c_blk = blockIdx.y, slot = blockIdx.z;
    const int e = experts[slot];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, ti = lane & 3;
    const int NT = rows / 16, KT = cols / 16;

    // Step 1: Decode 8 x 8 = 64 tiles of 16x16 into tile[row][col] = W_hat^T[row, col]
    for (int idx = warp; idx < 64; idx += 4) {
        const int nr = idx >> 3, nc = idx & 7;
        const int nt = r_blk * 8 + nr, kt = c_blk * 8 + nc;
        const uint32_t* tw = reinterpret_cast<const uint32_t*>(
            codes + ((int64_t(e) * NT + nt) * KT + kt) * TILE_BYTES);
        uint32_t h01, h23, h45, h67;
        exl3::dq8_mcg<BITS>([&](int wi) { return tw[wi]; }, lane, h01, h23, h45, h67);
        const float2 f01 = __half22float2(*reinterpret_cast<const __half2*>(&h01));
        const float2 f23 = __half22float2(*reinterpret_cast<const __half2*>(&h23));
        const float2 f45 = __half22float2(*reinterpret_cast<const __half2*>(&h45));
        const float2 f67 = __half22float2(*reinterpret_cast<const __half2*>(&h67));
        const int r0 = nr * 16 + g, r1 = nr * 16 + g + 8;
        const int c0 = nc * 16 + 2 * ti, c1 = nc * 16 + 2 * ti + 8;
        store_tile(r0, c0, f01.x);
        store_tile(r0, c0 + 1, f01.y);
        store_tile(r0, c1, f23.x);
        store_tile(r0, c1 + 1, f23.y);
        store_tile(r1, c0, f45.x);
        store_tile(r1, c0 + 1, f45.y);
        store_tile(r1, c1, f67.x);
        store_tile(r1, c1 + 1, f67.y);
    }
    __syncthreads();

    // Step 2: Apply H_128 along cols (K dim) and multiply by rin[e, c_blk*128 + col]
    const __half* rin_e = rin + int64_t(e) * cols + c_blk * 128;
    float r_in[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) r_in[i] = __half2float(rin_e[lane * 4 + i]);
    for (int r = warp; r < 128; r += 4) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = load_tile(r, lane * 4 + i);
        exl3::fwht_128_warp(v, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) store_tile(r, lane * 4 + i, v[i] * r_in[i]);
    }
    __syncthreads();

    // Step 3: Apply H_128 along rows (N dim), multiply by rout[e, r_blk*128 + row], write ActT out
    const __half* rout_e = rout + int64_t(e) * rows + r_blk * 128;
    float r_out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) r_out[i] = __half2float(rout_e[lane * 4 + i]);
    ActT* out_e = out + int64_t(slot) * rows * cols;
    for (int c = warp; c < 128; c += 4) {
        float v[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) v[i] = load_tile(lane * 4 + i, c);
        exl3::fwht_128_warp(v, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int rr = r_blk * 128 + lane * 4 + i;
            const int cc = c_blk * 128 + c;
            out_e[int64_t(rr) * cols + cc] = act::from_float(v[i] * r_out[i]);
        }
    }
}

}  // namespace

// codes [128, R, C/2] (NVFP4), [128, R, C] (MXFP8), [128, R, C/4] (ATLAS2), [128, R, 3C/8] (ATLAS3),
// or [128, R/16, C/16, 32*bits] (EXL3_2 / EXL3_3); experts int32 [n]; out ActT [n, R, C].
void dequant_experts(const void* codes, const void* scales, const float* factor, const std::int32_t* experts, int n,
                     int rows, int cols, Format fmt, ActT* out, cudaStream_t stream, const void* rout) {
    if (fmt == Format::EXL3_2 || fmt == Format::EXL3_3) {
        if (n < 0 || rows <= 0 || cols <= 0 || rows % 128 != 0 || cols % 128 != 0 || !codes || !scales || !rout ||
            !experts || !out)
            throw std::invalid_argument("moe: dequant_experts(EXL3): bad arguments");
        if (n == 0) return;
        const dim3 grid(rows / 128, cols / 128, n);
        const auto launch_exl3 = [&](auto bits_tag) {
            constexpr int B = decltype(bits_tag)::value;
            constexpr int SMEM_F32 = 128 * 128 * sizeof(float);
            constexpr int SMEM_F16 = 128 * 128 * sizeof(__half);
            if (current_max_smem_optin() >= SMEM_F32) {
                static std::atomic<uint32_t> attr_done{0};
                int dev = 0;
                CUDA_CHECK(cudaGetDevice(&dev));
                const uint32_t bit = 1u << (dev & 31);
                if (!(attr_done.load(std::memory_order_acquire) & bit)) {
                    CUDA_CHECK(cudaFuncSetAttribute(
                        dequant_exl3_kernel<B, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_F32));
                    attr_done.fetch_or(bit, std::memory_order_release);
                }
                dequant_exl3_kernel<B, float><<<grid, 128, SMEM_F32, stream>>>(
                    static_cast<const uint8_t*>(codes), static_cast<const __half*>(scales),
                    static_cast<const __half*>(rout), experts, out, rows, cols);
            } else {
                // Pascal <=48 KiB SMEM: 32 KiB FP16 tile buffer (no opt-in attribute needed).
                dequant_exl3_kernel<B, __half><<<grid, 128, SMEM_F16, stream>>>(
                    static_cast<const uint8_t*>(codes), static_cast<const __half*>(scales),
                    static_cast<const __half*>(rout), experts, out, rows, cols);
            }
        };
        if (fmt == Format::EXL3_2) launch_exl3(std::integral_constant<int, 2>{});
        else launch_exl3(std::integral_constant<int, 3>{});
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool mxfp8 = fmt == Format::MXFP8;
    const int col_align = (fmt == Format::NVFP4 || fmt == Format::ATLAS2) ? 16 : 32;
    if (n < 0 || rows <= 0 || cols % col_align != 0 || (!mxfp8 && !factor))
        throw std::invalid_argument("moe: dequant_experts: bad arguments");
    if (n == 0) return;
    int64_t per_expert_blocks = int64_t(rows) * (cols / (mxfp8 ? 32 : 16));
    int64_t total = n * per_expert_blocks;
    int threads = 256;
    int64_t grid = (total + threads - 1) / threads;
    if (fmt == Format::NVFP4) {
        dequant_nvfp4<<<grid, threads, 0, stream>>>(
            static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales), factor, experts, out,
            per_expert_blocks, total);
    } else if (fmt == Format::MXFP8) {
        dequant_mxfp8<<<grid, threads, 0, stream>>>(
            static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales), experts, out,
            per_expert_blocks, total);
    } else if (fmt == Format::ATLAS2) {
        dequant_atlas2<<<grid, threads, 0, stream>>>(
            static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales), factor, experts, out,
            per_expert_blocks, total);
    } else {
        dequant_atlas3<<<grid, threads, 0, stream>>>(
            static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales), factor, experts, out,
            per_expert_blocks, total);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::moe

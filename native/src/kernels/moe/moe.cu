// Routed-expert kernels for Inkling-2-Mini decode: moe_up (W13 GEMV + softcapped SwiGLU) and moe_down
// (W2 GEMV + routing-weighted sum + post-MLP short conv). Contract: native/KERNELS.md, area `moe`.
//
// Design
// ------
// * Weights are streamed once per call in their checkpoint encoding and decoded in registers to exactly
//   the BF16 values `model.decode_experts` produces (NVFP4: bf16(fp32(code * e4m3_scale) * factor), with
//   code * scale exact in f16; MXFP8: bf16(fp8 * 2^(s-127)), exact). Products run on tensor cores
//   (mma.m16n8k16 bf16 -> fp32): A = 16 weight rows x 16 k, B = 16 k x 8 tokens, so M = 1..8 tokens cost
//   the same weight traffic and instruction count as M = 1 (tokens >= M are zero columns).
// * Work unit: a "chunk" is 32 consecutive columns of one row (NVFP4: 16 B codes + 2 B scales; MXFP8:
//   32 B codes + 1 B scale). A warp owns one 128-column group of a 16-row tile: lane (g = lane/4,
//   t = lane%4) loads chunk 4*group + t of rows g and g+8, so a quad reads 64 B (NVFP4) / 128 B (MXFP8)
//   contiguous per row with 16-byte vector loads that bypass L1. The k order inside an mma is permuted
//   (lane t owns columns 32c+4i..32c+4i+3 of mma i) and B, read with 16-byte loads from L1/L2-resident
//   activations, uses the same permutation, so fragments need no shuffles and no shared memory.
// * Latency hiding: one chunk pair per warp per step and many warps per SM. (Giving a warp several chunks
//   does not help: ptxas sinks the extra loads next to their uses, measured 64% vs 89% of DRAM bandwidth.)
// * Expert deduplication for M > 1: the slots (token, k) that select the same expert share one pass over
//   its weights; tokens that do not select it get zero B columns and their outputs are discarded.
// * moe_up: grid (144 tiles of 16 W13 rows = 8 gate/up pairs) x (M*8 slots); a block whose slot is not the
//   first occurrence of its expert exits at once. 18 warps cover the 18 column groups of K = 2304; their
//   accumulators are summed in a fixed order through shared memory, then gate/up are rounded to BF16 and
//   the activation written for every slot that selected the expert.
// * moe_down: grid of 144 tiles of 16 W2 rows (output channels). A block handles all experts of all tokens
//   for its channels, so the weighted expert sum, its BF16 rounding and the post-MLP conv (with the state
//   update) are fused without atomics, workspaces or extra launches. 18 warps = 2 experts x 9 column
//   groups of K = 1152 per round; the next round's weights and activations are loaded into registers
//   before the current round is computed (ptxas cannot sink them past the round's barriers), so a block
//   keeps 18 KB (NVFP4) / 36 KB (MXFP8) of weights in flight. Per round the 9 partial dots of each expert
//   are summed in a fixed order and rounded to BF16; the weighted sum runs in slot order k = 0..7.
//   (Measured alternatives that were slower at M = 1: one warp per expert with 9 chunk pairs per lane
//   (ptxas sinks the loads), 4-block clusters sharing the tile through DSMEM (2x slower), and L2 prefetch
//   of later rounds.) Only 144 blocks exist, so the ~2 us of prologue/epilogue latency is visible; with
//   pdl (below) the prologue and first weight loads overlap the tail of moe_up.
// * Optional programmatic dependent launch (pdl=True): moe_up waits for its predecessor before reading
//   anything, then lets dependents launch; moe_down reads idx, enumerates experts and issues its first
//   round of weight loads before griddepcontrol.wait, and reads act/wts/conv state only after it. This is
//   only valid when the kernel before moe_down cannot still be writing idx (true for moe_up, which
//   triggers after its own wait). Without pdl both are ordinary stream-ordered launches.
// * Deterministic (fixed reduction orders) and capturable: no allocation, no host sync, current stream;
//   M is a launch-shape parameter of the captured graph and all routing is read from device memory.
// Ported from engine/csrc/moe/moe.cu (kernel bodies unchanged; torch launchers -> raw pointers).
#include "kernels/moe/moe.h"

#include "kernels/common/exl3.cuh"
#include "kernels/common/l2.cuh"
#include "kernels/common/mma_sm70.cuh"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ingine::kernels::moe {
namespace {

constexpr int H = 2304;
constexpr int I = 1152;
constexpr int TOPK = 8;
constexpr int MAXM = 8;
constexpr int TILE = 16;                    // weight rows per block
constexpr int UP_WARPS = H / 128;           // 18: one column group of W13 per warp
constexpr int DOWN_GROUPS = I / 128;        // 9 column groups of W2
constexpr int DOWN_PAR = 2;                 // experts per round
constexpr int DOWN_WARPS = DOWN_GROUPS * DOWN_PAR;

// ------------------------------------------------------------------------------------------- loads

// Expert weights are read once per step: their L2 lines are filled evict-first (kernels/common/l2.cuh), so
// they stop evicting the activations, KV rings and the small kernels' evict-last constants.
constexpr l2::Evict WEIGHT_L2 = l2::Evict::first;

// L1::no_allocate and L2::256B prefetch hints are available from SM80 onward.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define INGINE_MOE_LD_HINTS ".L1::no_allocate.L2::256B"
#else
#define INGINE_MOE_LD_HINTS ""
#endif

__device__ __forceinline__ uint4 ld_stream(const void* p, std::uint64_t pol) {
    uint4 v;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (l2::hinted<WEIGHT_L2>) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p), "l"(pol));
    } else
#endif
    {
        static_cast<void>(pol);
        asm volatile("ld.global.nc" INGINE_MOE_LD_HINTS ".v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                     : "l"(p));
    }
    return v;
}
__device__ __forceinline__ uint2 ld_stream_u2(const void* p, std::uint64_t pol) {
    uint2 v;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (l2::hinted<WEIGHT_L2>) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v2.u32 {%0,%1}, [%2], %3;"
                     : "=r"(v.x), "=r"(v.y)
                     : "l"(p), "l"(pol));
    } else
#endif
    {
        static_cast<void>(pol);
        asm volatile("ld.global.nc" INGINE_MOE_LD_HINTS ".v2.u32 {%0,%1}, [%2];"
                     : "=r"(v.x), "=r"(v.y)
                     : "l"(p));
    }
    return v;
}
__device__ __forceinline__ uint32_t ld_stream_u1(const void* p, std::uint64_t pol) {
    uint32_t v;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (l2::hinted<WEIGHT_L2>) {
        asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.u32 %0, [%1], %2;"
                     : "=r"(v)
                     : "l"(p), "l"(pol));
    } else
#endif
    {
        static_cast<void>(pol);
        asm volatile("ld.global.nc" INGINE_MOE_LD_HINTS ".u32 %0, [%1];"
                     : "=r"(v)
                     : "l"(p));
    }
    return v;
}
#undef INGINE_MOE_LD_HINTS

__device__ __forceinline__ uint16_t ld_scale2(const void* p, std::uint64_t pol) {
    uint16_t v;
    if constexpr (l2::hinted<WEIGHT_L2>) asm volatile("ld.global.nc.L2::cache_hint.u16 %0, [%1], %2;" : "=h"(v) : "l"(p), "l"(pol));
    else asm volatile("ld.global.nc.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ uint8_t ld_scale1(const void* p, std::uint64_t pol) {
    uint16_t v;
    if constexpr (l2::hinted<WEIGHT_L2>) asm volatile("ld.global.nc.L2::cache_hint.u8 %0, [%1], %2;" : "=h"(v) : "l"(p), "l"(pol));
    else asm volatile("ld.global.nc.u8 %0, [%1];" : "=h"(v) : "l"(p));
    return uint8_t(v);
}

// Inputs written by earlier kernels (x, act, idx, wts, conv_state) use coherent loads in volatile asm with a
// memory clobber: with programmatic dependent launch the producer may still be running when this kernel
// starts, so these loads must neither be non-coherent (.nc requires data read-only for the kernel's
// lifetime) nor be movable above griddepcontrol.wait (an __ldg is an invariant load the compiler may hoist).
__device__ __forceinline__ uint4 ld_dep(const void* p) {
    uint4 v;
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}
__device__ __forceinline__ int ld_dep(const int* p) {
    int v;
    asm volatile("ld.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ float ld_dep(const float* p) {
    float v;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
    return v;
}

// One 32-column chunk of one row in checkpoint encoding.
template <int FMT>
struct Chunk;
template <>
struct Chunk<0> {  // NVFP4: 32 E2M1 codes (even element in the low nibble) + 2 E4M3 block scales
    uint4 q;
    uint16_t s;
};
template <>
struct Chunk<1> {  // MXFP8: 32 E4M3 codes + 1 E8M0 block scale
    uint4 q0, q1;
    uint8_t s;
};
template <>
struct Chunk<2> {  // ATLAS2: 32 2-bit Lloyd-Max codes (8 B) + 2 E4M3 block scales (2.50 b/w)
    uint2 q;
    uint16_t s;
};
template <>
struct Chunk<3> {  // ATLAS3: 32 3-bit Lloyd-Max codes (lo0 4B, lo1 4B, hi 4B = 12 B) + 2 E4M3 block scales (3.50 b/w)
    uint32_t lo0, lo1, hi;
    uint16_t s;
};

template <int FMT>
struct Layout {
    // bytes per row of codes / scales for a matrix with K columns
    static constexpr int code_bytes(int K) {
        return FMT == 0 ? K / 2 : (FMT == 1 ? K : (FMT == 2 ? K / 4 : (K * 3) / 8));
    }
    static constexpr int scale_bytes(int K) { return FMT == 1 ? K / 32 : K / 16; }
};

template <int FMT>
__device__ __forceinline__ Chunk<FMT> load_chunk(const uint8_t* codes, const uint8_t* scales, int c, std::uint64_t pol) {
    Chunk<FMT> ch;
    if constexpr (FMT == 0) {
        ch.q = ld_stream(codes + 16 * c, pol);
        ch.s = ld_scale2(scales + 2 * c, pol);
    } else if constexpr (FMT == 1) {
        ch.q0 = ld_stream(codes + 32 * c, pol);
        ch.q1 = ld_stream(codes + 32 * c + 16, pol);
        ch.s = ld_scale1(scales + c, pol);
    } else if constexpr (FMT == 2) {
        ch.q = ld_stream_u2(codes + 8 * c, pol);
        ch.s = ld_scale2(scales + 2 * c, pol);
    } else {
        ch.lo0 = ld_stream_u1(codes + 12 * c, pol);
        ch.lo1 = ld_stream_u1(codes + 12 * c + 4, pol);
        ch.hi = ld_stream_u1(codes + 12 * c + 8, pol);
        ch.s = ld_scale2(scales + 2 * c, pol);
    }
    return ch;
}

// ------------------------------------------------------------------------------------------ decode

// Adapted from ashalliants/ninfer-3090 (src/ops/moe_situ/moe_situ.cu @ 8ea67ce, Apache-2.0):
// exact E4M3 * 2^-8 -> FP16x2 in 3 ALU ops (zero branches/LUTs).
// Valid for all finite E4M3 codes (zeros, subnormals, normals); caller scales result by 256.f.
__device__ __forceinline__ uint32_t e4m3x2_to_half2_m8(uint32_t u) {
    const uint32_t t = __byte_perm(u, 0u, 0x4140u);
    return ((t & 0x00800080u) << 8) | ((t & 0x007f007fu) << 7);
}

// Adapted from ashalliants/ninfer-3090 (src/ops/moe_situ/moe_situ.cu @ 8ea67ce, Apache-2.0):
// branchless, zero-LUT decode of 8 NVFP4 E2M1 nibbles packed in `q` (even element in low nibble,
// odd in high nibble of each byte) into 8 exact FP32 values = 2 * e2m1_value.
__device__ __forceinline__ void e2m1_twice8(uint32_t q, float (&out)[8]) {
    constexpr uint32_t kPos0 = 0x0F0E0D0Cu, kPos1 = 0x18141210u;  // nibbles 0..7: 2v + 12 = 12,13,14,15,16,18,20,24
    constexpr uint32_t kNeg0 = 0x090A0B0Cu, kNeg1 = 0x00040608u;  // nibbles 8..15: 12,11,10,9,8,6,4,0
    const uint32_t index = q & 0x77777777u;
    const uint32_t pick = 0x32103210u | ((q & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t pos = __byte_perm(kPos0, kPos1, index >> (16 * i));
        const uint32_t neg = __byte_perm(kNeg0, kNeg1, index >> (16 * i));
        const uint32_t biased = __byte_perm(pos, neg, pick >> (16 * i));
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            out[4 * i + k] = __uint_as_float(__byte_perm(biased, 0x4B000000u, 0x7540u | k)) - 8388620.0f;
        }
    }
}

// Adapted from ashalliants/ninfer-3090 (src/ops/moe_situ/moe_situ.cu @ 8ea67ce, Apache-2.0):
// Branchless decode of 8 NVFP4 E2M1 nibbles into two signed int8x4 words holding exact 2 * e2m1 in [-12, +12],
// feeding __dp4a directly with zero bias subtraction.
__device__ __forceinline__ void e2m1_twice_i8x4(uint32_t q, int32_t& w0, int32_t& w1) {
    constexpr uint32_t kPos0 = 0x03020100u, kPos1 = 0x0C080604u;  //  0, 1, 2, 3,  4,  6,  8, 12
    constexpr uint32_t kNeg0 = 0xFDFEFF00u, kNeg1 = 0xF4F8FAFCu;  //  0,-1,-2,-3, -4, -6, -8,-12
    const uint32_t idx = q & 0x77777777u;
    const uint32_t pick = 0x32103210u | ((q & 0x88888888u) >> 1);
    const uint32_t p0 = __byte_perm(kPos0, kPos1, idx);
    const uint32_t n0 = __byte_perm(kNeg0, kNeg1, idx);
    w0 = static_cast<int32_t>(__byte_perm(p0, n0, pick));
    const uint32_t p1 = __byte_perm(kPos0, kPos1, idx >> 16);
    const uint32_t n1 = __byte_perm(kNeg0, kNeg1, idx >> 16);
    w1 = static_cast<int32_t>(__byte_perm(p1, n1, pick >> 16));
}

__device__ __forceinline__ int32_t pack_i8x4(int a, int b, int c, int d) {
    const uint32_t lo = __byte_perm(static_cast<uint32_t>(a), static_cast<uint32_t>(b), 0x0040u);
    const uint32_t hi = __byte_perm(static_cast<uint32_t>(c), static_cast<uint32_t>(d), 0x0040u);
    return static_cast<int32_t>(__byte_perm(lo, hi, 0x5410u));
}

__device__ __forceinline__ void quant16_act_to_i8(const ActT* src16, int32_t* dst4, float& scale_out) {
    const uint4 q0 = ld_dep(src16);
    const uint4 q1 = ld_dep(src16 + 8);
    float xf[16];
    const float2 p0 = act::unpack2(q0.x), p1 = act::unpack2(q0.y), p2 = act::unpack2(q0.z), p3 = act::unpack2(q0.w);
    const float2 p4 = act::unpack2(q1.x), p5 = act::unpack2(q1.y), p6 = act::unpack2(q1.z), p7 = act::unpack2(q1.w);
    xf[0] = p0.x; xf[1] = p0.y; xf[2] = p1.x; xf[3] = p1.y;
    xf[4] = p2.x; xf[5] = p2.y; xf[6] = p3.x; xf[7] = p3.y;
    xf[8] = p4.x; xf[9] = p4.y; xf[10] = p5.x; xf[11] = p5.y;
    xf[12] = p6.x; xf[13] = p6.y; xf[14] = p7.x; xf[15] = p7.y;
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(xf[i]));
    scale_out = amax * (1.0f / 127.0f);
    const float inv = amax > 0.f ? (127.0f / amax) : 0.f;
    int qi[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) qi[i] = max(-127, min(127, __float2int_rn(xf[i] * inv)));
    dst4[0] = pack_i8x4(qi[0], qi[1], qi[2], qi[3]);
    dst4[1] = pack_i8x4(qi[4], qi[5], qi[6], qi[7]);
    dst4[2] = pack_i8x4(qi[8], qi[9], qi[10], qi[11]);
    dst4[3] = pack_i8x4(qi[12], qi[13], qi[14], qi[15]);
}

__device__ __forceinline__ uint16_t e4m3_to_f16_bits_portable(uint32_t b) {
    const uint32_t s = (b & 0x80u) << 8;
    const uint32_t a = b & 0x7fu;
    const uint32_t norm = (a + 64u) << 7;
    const uint32_t sub = a ? (0x1800u + (((0xBA986400u >> (a * 4u)) & 0xfu) << 8)) : 0u;
    return static_cast<uint16_t>(s | (a >= 8u ? norm : sub));
}

__device__ __forceinline__ uint32_t f16x2_from_e4m3x2(uint16_t v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return r;
#else
    return static_cast<uint32_t>(e4m3_to_f16_bits_portable(v & 0xffu)) |
           (static_cast<uint32_t>(e4m3_to_f16_bits_portable((v >> 8) & 0xffu)) << 16);
#endif
}

__device__ __forceinline__ void f16x2_from_e2m1x4(uint32_t w, uint32_t (&h)[4]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(w));
#else
    // Bit-identical to cvt.rn.f16x2.e2m1x2 via single __byte_perm (zero LUT/memory traffic on SM86/SM89).
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const uint32_t byte = (w >> (8 * b)) & 0xffu;
        const uint32_t sel = ((byte & 7u) << 4) | (((byte >> 4) & 7u) << 12);
        h[b] = __byte_perm(0x3E3C3800u, 0x46444240u, sel) | ((byte & 0x08u) << 12) | ((byte & 0x80u) << 24);
    }
#endif
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    return act::pack2(lo, hi);
}

// Decodes a chunk into 16 bf16x2 words (element pairs 0..15) holding exactly decode_experts' values.
// On Pascal sm_61 (P40/1080Ti), FP16 __hmul2 runs at 1/64 rate; multiply in exact FP32 instead.
__device__ __forceinline__ float2 scale_h2_to_f2(uint32_t h2_bits, __half2 sch, float sc_f) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 610
    static_cast<void>(sch);
    const float2 hv = __half22float2(*reinterpret_cast<const __half2*>(&h2_bits));
    return make_float2(hv.x * sc_f, hv.y * sc_f);
#else
    static_cast<void>(sc_f);
    return __half22float2(__hmul2(*reinterpret_cast<const __half2*>(&h2_bits), sch));
#endif
}

__device__ __forceinline__ void decode(const Chunk<0>& ch, float factor, uint32_t (&o)[16]) {
    const uint32_t s2 = f16x2_from_e4m3x2(ch.s);
    const __half2 sc = *reinterpret_cast<const __half2*>(&s2);
    const __half2 slo = __low2half2(sc), shi = __high2half2(sc);
    const float2 scf = __half22float2(sc);
    const uint32_t words[4] = {ch.q.x, ch.q.y, ch.q.z, ch.q.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t h[4];
        f16x2_from_e2m1x4(words[w], h);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            // code * scale is exact in f16 (<= 5 significant bits, |v| in [2^-10, 2688]); the factor
            // multiply rounds once in fp32, then to bf16, as in decode_experts.
            const float2 v = scale_h2_to_f2(h[b], w < 2 ? slo : shi, w < 2 ? scf.x : scf.y);
            o[4 * w + b] = pack_bf16(v.x * factor, v.y * factor);
        }
    }
}

__device__ __forceinline__ void decode(const Chunk<1>& ch, float /*factor*/, uint32_t (&o)[16]) {
    // E8M0 -> 2^(s-127); s = 0 is 2^-127 (fp32 subnormal), s = 255 would be NaN (never produced).
    const float sc = ch.s ? __uint_as_float(uint32_t(ch.s) << 23) : __uint_as_float(0x00400000u);
    const uint32_t words[8] = {ch.q0.x, ch.q0.y, ch.q0.z, ch.q0.w, ch.q1.x, ch.q1.y, ch.q1.z, ch.q1.w};
#pragma unroll
    for (int w = 0; w < 8; ++w) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const uint32_t h2 = f16x2_from_e4m3x2(uint16_t(words[w] >> (16 * hh)));
            const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&h2));
            o[2 * w + hh] = pack_bf16(v.x * sc, v.y * sc);  // exact: fp8 value times a power of two
        }
    }
}

__device__ __forceinline__ void decode(const Chunk<2>& ch, float factor, uint32_t (&o)[16]) {
    // Exact IEEE-754 FP16 bit patterns for kAtlas2 = {-1.5, -0.453125, +0.453125, +1.5}; branchless ALU, zero LUT.
    const uint32_t s2 = f16x2_from_e4m3x2(ch.s);
    const __half2 sc = *reinterpret_cast<const __half2*>(&s2);
    const __half2 slo = __low2half2(sc), shi = __high2half2(sc);
    const float2 scf = __half22float2(sc);
    const uint32_t words[2] = {ch.q.x, ch.q.y};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint32_t w = words[half];
        const __half2 sch = half == 0 ? slo : shi;
        const float sc_f = half == 0 ? scf.x : scf.y;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const uint32_t q0 = (w >> (4 * b)) & 3u;
            const uint32_t q1 = (w >> (4 * b + 2)) & 3u;
            const uint32_t h0 = (((q0 ^ (q0 >> 1)) & 1u) ? 0x3740u : 0x3E00u) | (((~q0) & 2u) << 14);
            const uint32_t h1 = (((q1 ^ (q1 >> 1)) & 1u) ? 0x3740u : 0x3E00u) | (((~q1) & 2u) << 14);
            const uint32_t h2 = h0 | (h1 << 16);
            const float2 v = scale_h2_to_f2(h2, sch, sc_f);
            o[8 * half + b] = pack_bf16(v.x * factor, v.y * factor);
        }
    }
}

__device__ __forceinline__ void decode(const Chunk<3>& ch, float factor, uint32_t (&o)[16]) {
    // Exact IEEE-754 FP16 bit patterns for kAtlas3 = {-2.0, -1.25, -0.703125, -0.2265625, +0.2265625, +0.703125, +1.25, +2.0}.
    // Decoded via single __byte_perm per pair (zero LUT/memory traffic).
    const uint32_t s2 = f16x2_from_e4m3x2(ch.s);
    const __half2 sc = *reinterpret_cast<const __half2*>(&s2);
    const __half2 slo = __low2half2(sc), shi = __high2half2(sc);
    const float2 scf = __half22float2(sc);
    const uint32_t los[2] = {ch.lo0, ch.lo1};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint32_t lo = los[half];
        const uint32_t hi = (ch.hi >> (16 * half)) & 0xffffu;
        const __half2 sch = half == 0 ? slo : shi;
        const float sc_f = half == 0 ? scf.x : scf.y;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const uint32_t c0 = ((lo >> (4 * b)) & 3u) | (((hi >> (2 * b)) & 1u) << 2);
            const uint32_t c1 = ((lo >> (4 * b + 2)) & 3u) | (((hi >> (2 * b + 1)) & 1u) << 2);
            const uint32_t m0 = (c0 & 4u) ? (c0 & 3u) : (3u - (c0 & 3u));
            const uint32_t m1 = (c1 & 4u) ? (c1 & 3u) : (3u - (c1 & 3u));
            const uint32_t sel = m0 | ((m0 + 4u) << 4) | (m1 << 8) | ((m1 + 4u) << 12);
            const uint32_t sgn = (((~c0) & 4u) << 13) | (((~c1) & 4u) << 29);
            const uint32_t h2 = __byte_perm(0x0000A040u, 0x403D3933u, sel) | sgn;
            const float2 v = scale_h2_to_f2(h2, sch, sc_f);
            o[8 * half + b] = pack_bf16(v.x * factor, v.y * factor);
        }
    }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
// Direct float2 decoders for Pascal sm_60/sm_61: avoid 1/64-rate sm_61 __hmul2 and clamp_fp16_finite,
// while matching model.decode_experts' ActT rounding exactly for NVFP4/ATLAS2/ATLAS3.
__device__ __forceinline__ float2 round_act_f2(float a, float b) {
#if INGINE_ACT_FP16
    return __half22float2(__floats2half2_rn(a, b));
#else
    return make_float2(act::round_act(a), act::round_act(b));
#endif
}
__device__ __forceinline__ void decode_f2(const Chunk<0>& ch, float factor, float2 (&o)[16]) {
    const uint32_t s2 = f16x2_from_e4m3x2(ch.s);
    const float2 scf = __half22float2(*reinterpret_cast<const __half2*>(&s2));
    const uint32_t words[4] = {ch.q.x, ch.q.y, ch.q.z, ch.q.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t h[4];
        f16x2_from_e2m1x4(words[w], h);
        const float sc_f = w < 2 ? scf.x : scf.y;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const float2 hv = __half22float2(*reinterpret_cast<const __half2*>(&h[b]));
            o[4 * w + b] = round_act_f2((hv.x * sc_f) * factor, (hv.y * sc_f) * factor);
        }
    }
}
__device__ __forceinline__ void decode_f2(const Chunk<1>& ch, float /*factor*/, float2 (&o)[16]) {
    const float sc = ch.s ? __uint_as_float(uint32_t(ch.s) << 23) : __uint_as_float(0x00400000u);
    const uint32_t words[8] = {ch.q0.x, ch.q0.y, ch.q0.z, ch.q0.w, ch.q1.x, ch.q1.y, ch.q1.z, ch.q1.w};
#pragma unroll
    for (int w = 0; w < 8; ++w) {
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const uint32_t h2 = f16x2_from_e4m3x2(uint16_t(words[w] >> (16 * hh)));
            const float2 v = __half22float2(*reinterpret_cast<const __half2*>(&h2));
            o[2 * w + hh] = make_float2(v.x * sc, v.y * sc);
        }
    }
}
__device__ __forceinline__ void decode_f2(const Chunk<2>& ch, float factor, float2 (&o)[16]) {
    const uint32_t s2 = f16x2_from_e4m3x2(ch.s);
    const float2 scf = __half22float2(*reinterpret_cast<const __half2*>(&s2));
    const uint32_t words[2] = {ch.q.x, ch.q.y};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint32_t w = words[half];
        const float sc_f = half == 0 ? scf.x : scf.y;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const uint32_t q0 = (w >> (4 * b)) & 3u;
            const uint32_t q1 = (w >> (4 * b + 2)) & 3u;
            const uint32_t h0 = (((q0 ^ (q0 >> 1)) & 1u) ? 0x3740u : 0x3E00u) | (((~q0) & 2u) << 14);
            const uint32_t h1 = (((q1 ^ (q1 >> 1)) & 1u) ? 0x3740u : 0x3E00u) | (((~q1) & 2u) << 14);
            const uint32_t h2 = h0 | (h1 << 16);
            const float2 hv = __half22float2(*reinterpret_cast<const __half2*>(&h2));
            o[8 * half + b] = round_act_f2((hv.x * sc_f) * factor, (hv.y * sc_f) * factor);
        }
    }
}
__device__ __forceinline__ void decode_f2(const Chunk<3>& ch, float factor, float2 (&o)[16]) {
    const uint32_t s2 = f16x2_from_e4m3x2(ch.s);
    const float2 scf = __half22float2(*reinterpret_cast<const __half2*>(&s2));
    const uint32_t los[2] = {ch.lo0, ch.lo1};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint32_t lo = los[half];
        const uint32_t hi = (ch.hi >> (16 * half)) & 0xffffu;
        const float sc_f = half == 0 ? scf.x : scf.y;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const uint32_t c0 = ((lo >> (4 * b)) & 3u) | (((hi >> (2 * b)) & 1u) << 2);
            const uint32_t c1 = ((lo >> (4 * b + 2)) & 3u) | (((hi >> (2 * b + 1)) & 1u) << 2);
            const uint32_t m0 = (c0 & 4u) ? (c0 & 3u) : (3u - (c0 & 3u));
            const uint32_t m1 = (c1 & 4u) ? (c1 & 3u) : (3u - (c1 & 3u));
            const uint32_t sel = m0 | ((m0 + 4u) << 4) | (m1 << 8) | ((m1 + 4u) << 12);
            const uint32_t sgn = (((~c0) & 4u) << 13) | (((~c1) & 4u) << 29);
            const uint32_t h2 = __byte_perm(0x0000A040u, 0x403D3933u, sel) | sgn;
            const float2 hv = __half22float2(*reinterpret_cast<const __half2*>(&h2));
            o[8 * half + b] = round_act_f2((hv.x * sc_f) * factor, (hv.y * sc_f) * factor);
        }
    }
}
#endif

// ------------------------------------------------------------------------------------------- mma

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    mma::mma_act(d, a0, a1, a2, a3, b0, b1);
}

__device__ __forceinline__ void mma_f16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
    mma::mma_f16(d, a0, a1, a2, a3, b0, b1);
}

// One warp step: chunk c of rows g and g+8 of a 16-row tile, plus the lane's B column (token g) for the
// same 32 columns (zero if the token does not use this expert).
template <int FMT>
struct Frag {
    Chunk<FMT> r0, r1;
    uint32_t x[16];
    float factor;
};

// Weights: `codes`/`scales` point at row g of the tile (row g+8 is 8 rows further).
template <int FMT, int K>
__device__ __forceinline__ void load_w(Frag<FMT>& f, const uint8_t* codes, const uint8_t* scales, int c,
                                       float factor, std::uint64_t pol) {
    constexpr int CB = Layout<FMT>::code_bytes(K), SB = Layout<FMT>::scale_bytes(K);
    f.r0 = load_chunk<FMT>(codes, scales, c, pol);
    f.r1 = load_chunk<FMT>(codes + 8 * CB, scales + 8 * SB, c, pol);
    f.factor = factor;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
// Volta sm_70 direct HMMA.884 layout: lane (qp, idx) owns ONE row and TWO contiguous chunks c0, c0+1.
template <int FMT>
struct FragSm70 {
    Chunk<FMT> r0, r1;
    float factor;
};

template <int FMT>
__device__ __forceinline__ void load_w_sm70(FragSm70<FMT>& f, const uint8_t* codes, const uint8_t* scales, int c0,
                                            float factor, std::uint64_t pol) {
    f.r0 = load_chunk<FMT>(codes, scales, c0, pol);
    f.r1 = load_chunk<FMT>(codes, scales, c0 + 1, pol);
    f.factor = factor;
}

__device__ __forceinline__ void load_x_chunk(const ActT* xrow, int c, uint32_t (&xv)[16]) {
    if (xrow) {
        const uint8_t* xp = reinterpret_cast<const uint8_t*>(xrow + 32 * c);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const uint4 q = ld_dep(xp + 16 * v);
            xv[4 * v] = q.x, xv[4 * v + 1] = q.y, xv[4 * v + 2] = q.z, xv[4 * v + 3] = q.w;
        }
    } else {
#pragma unroll
        for (int v = 0; v < 16; ++v) xv[v] = 0;
    }
}

template <int FMT>
__device__ __forceinline__ void decode_sm70_fast(const Chunk<FMT>& ch, float factor, uint32_t (&o)[16]) {
    if constexpr (FMT == 1) {
        // Exact MXFP8 -> FP16x2 via 3-op E4M3*2^-8 * FP16(2^(s-119)) (avoids XU F2F & clamp_fp16_finite).
        if (ch.s <= 134) {
            const float sc256 =
                (ch.s ? __uint_as_float(uint32_t(ch.s) << 23) : __uint_as_float(0x00400000u)) * 256.0f;
            const __half2 sch = __float2half2_rn(sc256);
            const uint32_t words[8] = {ch.q0.x, ch.q0.y, ch.q0.z, ch.q0.w, ch.q1.x, ch.q1.y, ch.q1.z, ch.q1.w};
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint32_t h0 = e4m3x2_to_half2_m8(words[w]);
                const uint32_t h1 = e4m3x2_to_half2_m8(words[w] >> 16);
                const __half2 p0 = __hmul2(*reinterpret_cast<const __half2*>(&h0), sch);
                const __half2 p1 = __hmul2(*reinterpret_cast<const __half2*>(&h1), sch);
                o[2 * w + 0] = *reinterpret_cast<const uint32_t*>(&p0);
                o[2 * w + 1] = *reinterpret_cast<const uint32_t*>(&p1);
            }
            return;
        }
    }
    decode(ch, factor, o);
}

template <int FMT>
__device__ __forceinline__ void mma_frag_sm70(const FragSm70<FMT>& f, const ActT* xrow, int c0, float (&d)[4]) {
    float acc8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    uint32_t a[16], xv[16];
    decode_sm70_fast(f.r0, f.factor, a);
    load_x_chunk(xrow, c0, xv);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        mma::sm70_mma_f16_direct(acc8, a[4 * i], a[4 * i + 1], a[4 * i + 2], a[4 * i + 3],
                                 xv[4 * i], xv[4 * i + 1], xv[4 * i + 2], xv[4 * i + 3]);
    decode_sm70_fast(f.r1, f.factor, a);
    load_x_chunk(xrow, c0 + 1, xv);
#pragma unroll
    for (int i = 0; i < 4; ++i)
        mma::sm70_mma_f16_direct(acc8, a[4 * i], a[4 * i + 1], a[4 * i + 2], a[4 * i + 3],
                                 xv[4 * i], xv[4 * i + 1], xv[4 * i + 2], xv[4 * i + 3]);
    mma::sm70_reduce_acc8_to_d4(acc8, d);
}
#endif

// Activations: `xrow` is the lane's token activation row (K bf16) or nullptr (zero column).
template <int FMT>
__device__ __forceinline__ void load_x(Frag<FMT>& f, const ActT* xrow, int c) {
    if (xrow) {
        const uint8_t* xp = reinterpret_cast<const uint8_t*>(xrow + 32 * c);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const uint4 q = ld_dep(xp + 16 * v);
            f.x[4 * v] = q.x, f.x[4 * v + 1] = q.y, f.x[4 * v + 2] = q.z, f.x[4 * v + 3] = q.w;
        }
    } else {
#pragma unroll
        for (int v = 0; v < 16; ++v) f.x[v] = 0;
    }
}

// Programmatic dependent launch (no-ops when the kernel is launched without the PDL attribute).
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

template <int FMT>
__device__ __forceinline__ void mma_frag(const Frag<FMT>& f, float (&d)[4], uint64_t mask, int M) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
    // Pascal sm_60/sm_61 fast path: direct float2 decode (no FP32->clamp->FP16->FP32 round-trip, no 1/64-rate
    // __hmul2 on sm_61), iterate ONLY over tokens that actually routed to this expert, exact FP32 fmaf.
    float2 fa[16], fb[16];
    decode_f2(f.r0, f.factor, fa);
    decode_f2(f.r1, f.factor, fb);
    const int t = threadIdx.x & 3;
    for (int m = 0; m < M; ++m) {
        if (((mask >> (8 * m)) & 0xffu) == 0u) continue;
        const int src_lane = 4 * m + t;
        float pa = 0.f, pb = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float2 fx = act::unpack2(__shfl_sync(0xffffffffu, f.x[i], src_lane));
            pa = fmaf(fa[i].x, fx.x, fmaf(fa[i].y, fx.y, pa));
            pb = fmaf(fb[i].x, fx.x, fmaf(fb[i].y, fx.y, pb));
        }
        pa += __shfl_xor_sync(0xffffffffu, pa, 1);
        pa += __shfl_xor_sync(0xffffffffu, pa, 2);
        pb += __shfl_xor_sync(0xffffffffu, pb, 1);
        pb += __shfl_xor_sync(0xffffffffu, pb, 2);
        if (m == 2 * t) {
            d[0] += pa;
            d[2] += pb;
        } else if (m == 2 * t + 1) {
            d[1] += pa;
            d[3] += pb;
        }
    }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
    uint32_t a[16], b[16];
    decode(f.r0, f.factor, a);
    decode(f.r1, f.factor, b);
    // Hoist Volta HMMA.884 accumulator across all 8 steps -> 1 reduction instead of 8 (112 fewer shuffles/warp).
    float acc8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int i = 0; i < 8; ++i)
        mma::sm70_mma_f16_acc(acc8, a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1], f.x[2 * i], f.x[2 * i + 1]);
    mma::sm70_reduce_acc8_to_d4(acc8, d);
#else
    uint32_t a[16], b[16];
    decode(f.r0, f.factor, a);
    decode(f.r1, f.factor, b);
    (void)mask;
    (void)M;
#pragma unroll
    for (int i = 0; i < 8; ++i) mma_bf16(d, a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1], f.x[2 * i], f.x[2 * i + 1]);
#endif
}

// Slots 0..63 (token * 8 + k) whose expert equals e, from a warp-wide view of idx (lo = idx[lane],
// hi = idx[lane + 32], -1 beyond M*8).
__device__ __forceinline__ uint64_t slot_mask(int e, int lo, int hi) {
    const uint32_t m0 = __ballot_sync(0xffffffffu, lo == e);
    const uint32_t m1 = __ballot_sync(0xffffffffu, hi == e);
    return uint64_t(m0) | (uint64_t(m1) << 32);
}

__device__ __forceinline__ float bf16_round(float v) { return act::round_act(v); }

// --------------------------------------------------------------------------------------- moe_up

template <int FMT>
__global__ void __launch_bounds__(UP_WARPS * 32)
moe_up_kernel(const ActT* __restrict__ x, const int* __restrict__ idx, const uint8_t* __restrict__ w,
              const uint8_t* __restrict__ ws, const float* __restrict__ wfactor, int M,
              ActT* __restrict__ out, const uint8_t* const* __restrict__ w_ptrs,
              const uint8_t* const* __restrict__ ws_ptrs) {
    constexpr int K = H, R = 2 * I;
    constexpr int CB = Layout<FMT>::code_bytes(K), SB = Layout<FMT>::scale_bytes(K);
    __shared__ float red[UP_WARPS][32][4];
    __shared__ float tile_out[TILE][MAXM + 1];

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x, slot = blockIdx.y;
    const int nslots = M * TOPK;
    pdl_wait();     // x and idx come from earlier kernels
    pdl_trigger();  // moe_down may launch once every block got here (it waits for us before reading act)
    const int lo = lane < nslots ? ld_dep(idx + lane) : -1;
    const int hi = lane + 32 < nslots ? ld_dep(idx + lane + 32) : -1;
    const int e = ld_dep(idx + slot);
    if (w_ptrs && !w_ptrs[e]) return;  // EP: non-local expert
    const uint64_t mask = slot_mask(e, lo, hi);
    if (__ffsll(static_cast<long long>(mask)) - 1 != slot) return;  // an earlier slot owns this expert

    float d[4] = {0.f, 0.f, 0.f, 0.f};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
    const int qp = (lane >> 2) & 3, idx_v = (lane & 3) | ((lane & 16) >> 2);
    const int row_off = (qp >= 2 ? 8 : 0) + idx_v;
    const int c0_v = 4 * warp + (qp & 1) * 2;
    const size_t local_row = static_cast<size_t>(tile * TILE + row_off);
    const size_t row = static_cast<size_t>(e) * R + local_row;
    const uint8_t* we = w_ptrs ? (w_ptrs[e] + local_row * CB) : (w + row * CB);
    const uint8_t* wse = ws_ptrs ? (ws_ptrs[e] + local_row * SB) : (ws + row * SB);
    const bool on = idx_v < M && ((mask >> (8 * idx_v)) & 0xffu);
    FragSm70<FMT> f;
    load_w_sm70<FMT>(f, we, wse, c0_v, FMT != 1 ? wfactor[e] : 1.f, l2::policy<WEIGHT_L2>());
    mma_frag_sm70(f, on ? x + static_cast<size_t>(idx_v) * K : nullptr, c0_v, d);
#else
    const size_t local_row = static_cast<size_t>(tile * TILE + g);
    const size_t row = static_cast<size_t>(e) * R + local_row;
    const uint8_t* we = w_ptrs ? (w_ptrs[e] + local_row * CB) : (w + row * CB);
    const uint8_t* wse = ws_ptrs ? (ws_ptrs[e] + local_row * SB) : (ws + row * SB);
    const bool on = g < M && ((mask >> (8 * g)) & 0xffu);
    Frag<FMT> f;
    load_w<FMT, K>(f, we, wse, 4 * warp + t, FMT != 1 ? wfactor[e] : 1.f, l2::policy<WEIGHT_L2>());
    load_x(f, on ? x + static_cast<size_t>(g) * K : nullptr, 4 * warp + t);
    mma_frag(f, d, mask, M);
#endif

#pragma unroll
    for (int v = 0; v < 4; ++v) red[warp][lane][v] = d[v];
    __syncthreads();
    if (threadIdx.x < 32) {
        float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ww = 0; ww < UP_WARPS; ++ww)
#pragma unroll
            for (int v = 0; v < 4; ++v) s[v] += red[ww][lane][v];
        tile_out[g][2 * t] = s[0];
        tile_out[g][2 * t + 1] = s[1];
        tile_out[g + 8][2 * t] = s[2];
        tile_out[g + 8][2 * t + 1] = s[3];
    }
    __syncthreads();
    if (threadIdx.x < TILE / 2 * MAXM) {
        const int p = threadIdx.x & 7, n = threadIdx.x >> 3;  // gate/up pair p, token n
        uint32_t kmask = n < M ? uint32_t(mask >> (8 * n)) & 0xffu : 0u;
        if (kmask) {
            const float gate = bf16_round(tile_out[2 * p][n]);
            const float up = bf16_round(tile_out[2 * p + 1][n]);
            const float a = tanhf(gate / 4.f) * 4.f * (1.f / (1.f + expf(-gate))) * (tanhf(up / 25.f) * 25.f);
            const ActT ab = act::from_float(a);
            while (kmask) {
                const int k = __ffs(kmask) - 1;
                kmask &= kmask - 1;
                out[(static_cast<size_t>(n) * TOPK + k) * I + tile * (TILE / 2) + p] = ab;
            }
        }
    }
}

// ------------------------------------------------------------------------------------- moe_down

template <int FMT>
__global__ void __launch_bounds__(DOWN_WARPS * 32)
moe_down_kernel(const ActT* __restrict__ act, const int* __restrict__ idx, const float* __restrict__ wts,
                const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws, const float* __restrict__ wfactor,
                int M, const float* __restrict__ conv_w, float* __restrict__ conv_state, float* __restrict__ out,
                float* __restrict__ conv_hist, const uint8_t* const* __restrict__ w_ptrs,
                const uint8_t* const* __restrict__ ws_ptrs) {
    constexpr int K = I, R = H;
    constexpr int CB = Layout<FMT>::code_bytes(K), SB = Layout<FMT>::scale_bytes(K);
    __shared__ float part[DOWN_WARPS][32][4];        // per-warp partial dots of the current round
    __shared__ float ybuf[TOPK * MAXM][TILE][MAXM];  // BF16-rounded expert outputs [unique expert][row][token]
    __shared__ int uexp[TOPK * MAXM];                // unique experts in first-slot order
    __shared__ unsigned long long umask[TOPK * MAXM];  // slots selecting each unique expert
    __shared__ uint8_t slot_u[TOPK * MAXM];          // slot -> unique expert index
    __shared__ int nuniq;
    __shared__ float sbuf[MAXM][TILE];

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int tile = blockIdx.x;
    const int nslots = M * TOPK;

    if (warp == 0) {
        // idx is read before pdl_wait on purpose: it was written by the router, which completed before
        // moe_up passed its own wait and triggered this launch.
        const int lo = lane < nslots ? ld_dep(idx + lane) : -1;
        const int hi = lane + 32 < nslots ? ld_dep(idx + lane + 32) : -1;
        uint64_t remaining = nslots == 64 ? ~0ull : ((1ull << nslots) - 1);
        int u = 0;
        while (remaining) {
            const int s = __ffsll(static_cast<long long>(remaining)) - 1;
            const int e = __shfl_sync(0xffffffffu, s < 32 ? lo : hi, s & 31);
            const uint64_t mask = slot_mask(e, lo, hi);
            remaining &= ~mask;
            const bool local = !w_ptrs || (w_ptrs[e] != nullptr);
            if (local) {
                if (lane == 0) uexp[u] = e, umask[u] = mask;
                if ((mask >> lane) & 1) slot_u[lane] = uint8_t(u);
                if ((mask >> (lane + 32)) & 1) slot_u[lane + 32] = uint8_t(u);
                ++u;
            } else {
                if ((mask >> lane) & 1) slot_u[lane] = uint8_t(255);
                if ((mask >> (lane + 32)) & 1) slot_u[lane + 32] = uint8_t(255);
            }
        }
        if (lane == 0) nuniq = u;
    }
    __syncthreads();

    const int U = nuniq;
    const int p = warp / DOWN_GROUPS, c = 4 * (warp % DOWN_GROUPS) + t;
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
    const int qp = (lane >> 2) & 3, idx_v = (lane & 3) | ((lane & 16) >> 2);
    const int row_off = (qp >= 2 ? 8 : 0) + idx_v;
    const int c0_v = 4 * (warp % DOWN_GROUPS) + (qp & 1) * 2;
    auto xrow_v = [&](int u) -> const ActT* {
        const uint32_t kmask = idx_v < M ? uint32_t(umask[u] >> (8 * idx_v)) & 0xffu : 0u;
        return kmask ? act + (static_cast<size_t>(idx_v) * TOPK + (__ffs(kmask) - 1)) * K : nullptr;
    };
    auto load_v = [&](int u, FragSm70<FMT>& f) {
        const int e = uexp[u];
        const size_t local_row = static_cast<size_t>(tile * TILE + row_off);
        const size_t row = static_cast<size_t>(e) * R + local_row;
        const uint8_t* we = w_ptrs ? (w_ptrs[e] + local_row * CB) : (w + row * CB);
        const uint8_t* wse = ws_ptrs ? (ws_ptrs[e] + local_row * SB) : (ws + row * SB);
        load_w_sm70<FMT>(f, we, wse, c0_v, FMT != 1 ? wfactor[e] : 1.f, wpol);
    };
    FragSm70<FMT> nxt;
    if (p < U) load_v(p, nxt);
    pdl_wait();
#pragma unroll 1
    for (int base = 0; base < U; base += DOWN_PAR) {
        const FragSm70<FMT> cur = nxt;
        const int u = base + p;
        if (u + DOWN_PAR < U) load_v(u + DOWN_PAR, nxt);
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        if (u < U) mma_frag_sm70(cur, xrow_v(u), c0_v, d);
#else
    auto xrow = [&](int u) -> const ActT* {
        const uint32_t kmask = g < M ? uint32_t(umask[u] >> (8 * g)) & 0xffu : 0u;
        return kmask ? act + (static_cast<size_t>(g) * TOPK + (__ffs(kmask) - 1)) * K : nullptr;
    };
    auto load = [&](int u, Frag<FMT>& f) {
        const int e = uexp[u];
        const size_t local_row = static_cast<size_t>(tile * TILE + g);
        const size_t row = static_cast<size_t>(e) * R + local_row;
        const uint8_t* we = w_ptrs ? (w_ptrs[e] + local_row * CB) : (w + row * CB);
        const uint8_t* wse = ws_ptrs ? (ws_ptrs[e] + local_row * SB) : (ws + row * SB);
        load_w<FMT, K>(f, we, wse, c, FMT != 1 ? wfactor[e] : 1.f, wpol);
    };

    // The first round's weights do not depend on moe_up: fetch them before waiting for it (PDL).
    Frag<FMT> nxt;
    if (p < U) load(p, nxt);
    pdl_wait();
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
    // On Pascal (96-reg launch bound), avoid holding both cur.x[16] and nxt.x[16] live across mma_frag.
#pragma unroll 1
    for (int base = 0; base < U; base += DOWN_PAR) {
        const int u = base + p;
        if (u < U) load_x(nxt, xrow(u), c);
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        if (u < U) mma_frag(nxt, d, umask[u], M);
        if (u + DOWN_PAR < U) load(u + DOWN_PAR, nxt);
#else
    if (p < U) load_x(nxt, xrow(p), c);
#pragma unroll 1
    for (int base = 0; base < U; base += DOWN_PAR) {
        const Frag<FMT> cur = nxt;
        const int u = base + p;
        if (u + DOWN_PAR < U) load(u + DOWN_PAR, nxt), load_x(nxt, xrow(u + DOWN_PAR), c);  // next round
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        if (u < U) mma_frag(cur, d, umask[u], M);
#endif
#endif
#pragma unroll
        for (int v = 0; v < 4; ++v) part[warp][lane][v] = d[v];
        __syncthreads();
        if (threadIdx.x < DOWN_PAR * 32 && base + (threadIdx.x >> 5) < U) {
            const int pp = threadIdx.x >> 5;
            float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int j = 0; j < DOWN_GROUPS; ++j)
#pragma unroll
                for (int v = 0; v < 4; ++v) s[v] += part[pp * DOWN_GROUPS + j][lane][v];
            float(&y)[TILE][MAXM] = ybuf[base + pp];
            y[g][2 * t] = bf16_round(s[0]);
            y[g][2 * t + 1] = bf16_round(s[1]);
            y[g + 8][2 * t] = bf16_round(s[2]);
            y[g + 8][2 * t + 1] = bf16_round(s[3]);
        }
        __syncthreads();
    }
    pdl_trigger();  // only the short epilogue remains; dependents still wait for our completion

    if (threadIdx.x < TILE * M) {
        const int r = threadIdx.x & (TILE - 1), n = threadIdx.x / TILE;
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const int sl = n * TOPK + k;
            const uint8_t su = slot_u[sl];
            if (su != 255) s = __fadd_rn(s, __fmul_rn(ld_dep(wts + sl), ybuf[su][r][n]));
        }
        if (!conv_w) {
            out[static_cast<size_t>(n) * H + tile * TILE + r] = s;
            return;
        }
        sbuf[n][r] = bf16_round(s);
    }
    if (!conv_w) return;
    __syncthreads();

    if (threadIdx.x < TILE) {
        const int ch = tile * TILE + threadIdx.x;
        const float4 cw = *reinterpret_cast<const float4*>(conv_w + 4 * ch);
        float ext[MAXM + 3];
        ext[0] = ld_dep(conv_state + ch);
        ext[1] = ld_dep(conv_state + H + ch);
        ext[2] = ld_dep(conv_state + 2 * H + ch);
#pragma unroll
        for (int m = 0; m < MAXM; ++m) ext[m + 3] = m < M ? sbuf[m][threadIdx.x] : 0.f;
        constexpr float inv_sqrt5 = 1.f / 2.2360679774997896f;  // torch divides by a scalar via its reciprocal
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                // same op order and roundings as model.static_conv in FP32 (bit-exact)
                float acc = __fmul_rn(cw.x, ext[m]);
                acc = __fadd_rn(acc, __fmul_rn(cw.y, ext[m + 1]));
                acc = __fadd_rn(acc, __fmul_rn(cw.z, ext[m + 2]));
                acc = __fadd_rn(acc, __fmul_rn(cw.w, ext[m + 3]));
                out[static_cast<size_t>(m) * H + ch] = __fmul_rn(__fadd_rn(acc, ext[m + 3]), inv_sqrt5);
                if (conv_hist) conv_hist[static_cast<size_t>(m) * H + ch] = ext[m + 3];  // for rollback
            }
        }
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int m = 1; m <= MAXM; ++m)  // constant indices keep ext[] in registers
                if (m == M) conv_state[j * H + ch] = ext[m + j];
    }
}

// ----------------------------------------------------------------- Pre-Ampere FP16 streaming MoE kernels
// Dedicated streaming GEMV kernels for NVFP4 (FMT=0) and MXFP8 (FMT=1) on pre-Ampere FP16 builds (Volta sm_70,
// Turing sm_75, Pascal sm_60/61). Eliminates per-weight FP16 re-packing, 7/8-empty HMMA columns, and multi-round
// reduction barriers while keeping active-token-only work for speculative widths M=2..8.

__device__ __forceinline__ void load_act_chunk_l1(const ActT* row_base, int c, int word_idx, float4& x0, float4& x1) {
    const uint4 q = ld_dep(row_base + c * 32 + word_idx * 8);
    const float2 p0 = act::unpack2(q.x), p1 = act::unpack2(q.y), p2 = act::unpack2(q.z), p3 = act::unpack2(q.w);
    x0 = make_float4(p0.x, p0.y, p1.x, p1.y);
    x1 = make_float4(p2.x, p2.y, p3.x, p3.y);
}

__device__ __forceinline__ float dot8_f4(const float (&v)[8], const float4& x0, const float4& x1, float acc) {
    acc = fmaf(v[0], x0.x, fmaf(v[1], x0.y, fmaf(v[2], x0.z, fmaf(v[3], x0.w, acc))));
    acc = fmaf(v[4], x1.x, fmaf(v[5], x1.y, fmaf(v[6], x1.z, fmaf(v[7], x1.w, acc))));
    return acc;
}

template <int FMT>
__device__ __forceinline__ void dot_chunk_1tok(const Chunk<FMT>& ch, const float* sx32, float& acc) {
    if constexpr (FMT == 0) {
        const uint32_t s2 = e4m3x2_to_half2_m8(ch.s);
        const float2 sc = __half22float2(*reinterpret_cast<const __half2*>(&s2));
        float v[8];
        float d0 = 0.f, d1 = 0.f;
        e2m1_twice8(ch.q.x, v);
        d0 = dot8_f4(v, *reinterpret_cast<const float4*>(sx32 + 0), *reinterpret_cast<const float4*>(sx32 + 4), d0);
        e2m1_twice8(ch.q.y, v);
        d0 = dot8_f4(v, *reinterpret_cast<const float4*>(sx32 + 8), *reinterpret_cast<const float4*>(sx32 + 12), d0);
        acc = fmaf(d0, sc.x, acc);
        e2m1_twice8(ch.q.z, v);
        d1 = dot8_f4(v, *reinterpret_cast<const float4*>(sx32 + 16), *reinterpret_cast<const float4*>(sx32 + 20), d1);
        e2m1_twice8(ch.q.w, v);
        d1 = dot8_f4(v, *reinterpret_cast<const float4*>(sx32 + 24), *reinterpret_cast<const float4*>(sx32 + 28), d1);
        acc = fmaf(d1, sc.y, acc);
    } else {
        const float sc256 =
            (ch.s ? __uint_as_float(uint32_t(ch.s) << 23) : __uint_as_float(0x00400000u)) * 256.0f;
        const uint32_t words[8] = {ch.q0.x, ch.q0.y, ch.q0.z, ch.q0.w, ch.q1.x, ch.q1.y, ch.q1.z, ch.q1.w};
        float d = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const uint32_t hlo = e4m3x2_to_half2_m8(words[w]);
            const uint32_t hhi = e4m3x2_to_half2_m8(words[w] >> 16);
            const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&hlo));
            const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hhi));
            const float4 xv = *reinterpret_cast<const float4*>(sx32 + 4 * w);
            d = fmaf(a.x, xv.x, fmaf(a.y, xv.y, fmaf(b.x, xv.z, fmaf(b.y, xv.w, d))));
        }
        acc = fmaf(d, sc256, acc);
    }
}

template <int FMT>
__device__ __forceinline__ void dot_chunk_pair_1tok(const Chunk<FMT>& cg, const Chunk<FMT>& cu, const float* sx32,
                                                    float& acc_g, float& acc_u) {
    if constexpr (FMT == 0) {
        const uint32_t sg2 = e4m3x2_to_half2_m8(cg.s);
        const uint32_t su2 = e4m3x2_to_half2_m8(cu.s);
        const float2 scg = __half22float2(*reinterpret_cast<const __half2*>(&sg2));
        const float2 scu = __half22float2(*reinterpret_cast<const __half2*>(&su2));
        float v[8];
        float dg0 = 0.f, du0 = 0.f, dg1 = 0.f, du1 = 0.f;
        {
            const float4 x0 = *reinterpret_cast<const float4*>(sx32 + 0);
            const float4 x1 = *reinterpret_cast<const float4*>(sx32 + 4);
            e2m1_twice8(cg.q.x, v);
            dg0 = dot8_f4(v, x0, x1, dg0);
            e2m1_twice8(cu.q.x, v);
            du0 = dot8_f4(v, x0, x1, du0);
        }
        {
            const float4 x0 = *reinterpret_cast<const float4*>(sx32 + 8);
            const float4 x1 = *reinterpret_cast<const float4*>(sx32 + 12);
            e2m1_twice8(cg.q.y, v);
            dg0 = dot8_f4(v, x0, x1, dg0);
            e2m1_twice8(cu.q.y, v);
            du0 = dot8_f4(v, x0, x1, du0);
        }
        acc_g = fmaf(dg0, scg.x, acc_g);
        acc_u = fmaf(du0, scu.x, acc_u);
        {
            const float4 x0 = *reinterpret_cast<const float4*>(sx32 + 16);
            const float4 x1 = *reinterpret_cast<const float4*>(sx32 + 20);
            e2m1_twice8(cg.q.z, v);
            dg1 = dot8_f4(v, x0, x1, dg1);
            e2m1_twice8(cu.q.z, v);
            du1 = dot8_f4(v, x0, x1, du1);
        }
        {
            const float4 x0 = *reinterpret_cast<const float4*>(sx32 + 24);
            const float4 x1 = *reinterpret_cast<const float4*>(sx32 + 28);
            e2m1_twice8(cg.q.w, v);
            dg1 = dot8_f4(v, x0, x1, dg1);
            e2m1_twice8(cu.q.w, v);
            du1 = dot8_f4(v, x0, x1, du1);
        }
        acc_g = fmaf(dg1, scg.y, acc_g);
        acc_u = fmaf(du1, scu.y, acc_u);
    } else {
        const float scg =
            (cg.s ? __uint_as_float(uint32_t(cg.s) << 23) : __uint_as_float(0x00400000u)) * 256.0f;
        const float scu =
            (cu.s ? __uint_as_float(uint32_t(cu.s) << 23) : __uint_as_float(0x00400000u)) * 256.0f;
        const uint32_t wg[8] = {cg.q0.x, cg.q0.y, cg.q0.z, cg.q0.w, cg.q1.x, cg.q1.y, cg.q1.z, cg.q1.w};
        const uint32_t wu[8] = {cu.q0.x, cu.q0.y, cu.q0.z, cu.q0.w, cu.q1.x, cu.q1.y, cu.q1.z, cu.q1.w};
        float dg = 0.f, du = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            const float4 xv = *reinterpret_cast<const float4*>(sx32 + 4 * w);
            {
                const uint32_t hlo = e4m3x2_to_half2_m8(wg[w]);
                const uint32_t hhi = e4m3x2_to_half2_m8(wg[w] >> 16);
                const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&hlo));
                const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hhi));
                dg = fmaf(a.x, xv.x, fmaf(a.y, xv.y, fmaf(b.x, xv.z, fmaf(b.y, xv.w, dg))));
            }
            {
                const uint32_t hlo = e4m3x2_to_half2_m8(wu[w]);
                const uint32_t hhi = e4m3x2_to_half2_m8(wu[w] >> 16);
                const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&hlo));
                const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hhi));
                du = fmaf(a.x, xv.x, fmaf(a.y, xv.y, fmaf(b.x, xv.z, fmaf(b.y, xv.w, du))));
            }
        }
        acc_g = fmaf(dg, scg, acc_g);
        acc_u = fmaf(du, scu, acc_u);
    }
}

template <int FMT, bool M1>
__global__ void __launch_bounds__(256)
moe_up_fp16_stream_kernel(const ActT* __restrict__ x, const int* __restrict__ idx, const uint8_t* __restrict__ w,
                          const uint8_t* __restrict__ ws, const float* __restrict__ wfactor, int M,
                          ActT* __restrict__ out, const uint8_t* const* __restrict__ w_ptrs,
                          const uint8_t* const* __restrict__ ws_ptrs) {
    constexpr int K = H, R = 2 * I;
    constexpr int CB = Layout<FMT>::code_bytes(K), SB = Layout<FMT>::scale_bytes(K);
    constexpr int STAGE_T = M1 ? 1 : 2;
    __shared__ float s_x[STAGE_T][72][36];  // padded 36 floats/chunk -> bank-conflict-free float4 loads

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int slot = blockIdx.y;
    const int nslots = M * TOPK;
    const int lo = lane < nslots ? ld_dep(idx + lane) : -1;
    const int hi = lane + 32 < nslots ? ld_dep(idx + lane + 32) : -1;
    const int e = ld_dep(idx + slot);
    if (w_ptrs && !w_ptrs[e]) return;
    const uint64_t mask = slot_mask(e, lo, hi);
    if (__ffsll(static_cast<long long>(mask)) - 1 != slot) return;

    int toks[M1 ? 1 : MAXM];
    int ntok = 0;
    if constexpr (M1) {
        toks[0] = 0;
        ntok = 1;
    } else {
        for (int m = 0; m < M; ++m) {
            if ((mask >> (8 * m)) & 0xffu) toks[ntok++] = m;
        }
    }

    const int nstage = ntok < STAGE_T ? ntok : STAGE_T;
    for (int ti = 0; ti < nstage; ++ti) {
        const ActT* xm = x + static_cast<size_t>(toks[ti]) * K;
        for (int j = tid; j < 72 * 4; j += 256) {
            const int c = j >> 2;
            const int sub = (j & 3) * 8;
            const uint4 q = ld_dep(xm + c * 32 + sub);
            const float2 p0 = act::unpack2(q.x), p1 = act::unpack2(q.y), p2 = act::unpack2(q.z), p3 = act::unpack2(q.w);
            *reinterpret_cast<float4*>(&s_x[ti][c][sub]) = make_float4(p0.x, p0.y, p1.x, p1.y);
            *reinterpret_cast<float4*>(&s_x[ti][c][sub + 4]) = make_float4(p2.x, p2.y, p3.x, p3.y);
        }
    }
    __syncthreads();

    const int pair_local = tid >> 3;  // 0..31
    const int lane8 = tid & 7;        // 0..7
    const int pair_global = blockIdx.x * 32 + pair_local;
    const size_t local_row_g = static_cast<size_t>(2 * pair_global);
    const size_t row_g = static_cast<size_t>(e) * R + local_row_g;
    const uint8_t* wg_ptr = w_ptrs ? (w_ptrs[e] + local_row_g * CB) : (w + row_g * CB);
    const uint8_t* wsg_ptr = ws_ptrs ? (ws_ptrs[e] + local_row_g * SB) : (ws + row_g * SB);
    const uint8_t* wu_ptr = wg_ptr + CB;
    const uint8_t* wsu_ptr = wsg_ptr + SB;
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();

    Chunk<FMT> cg_cur = load_chunk<FMT>(wg_ptr, wsg_ptr, lane8, wpol);
    Chunk<FMT> cu_cur = load_chunk<FMT>(wu_ptr, wsu_ptr, lane8, wpol);

    if (M1 || ntok == 1) {
        float acc_g = 0.f, acc_u = 0.f;
#pragma unroll
        for (int step = 0; step < 9; ++step) {
            Chunk<FMT> cg_nxt, cu_nxt;
            if (step + 1 < 9) {
                const int cn = 8 * (step + 1) + lane8;
                cg_nxt = load_chunk<FMT>(wg_ptr, wsg_ptr, cn, wpol);
                cu_nxt = load_chunk<FMT>(wu_ptr, wsu_ptr, cn, wpol);
            }
            const int c = 8 * step + lane8;
            dot_chunk_pair_1tok<FMT>(cg_cur, cu_cur, &s_x[0][c][0], acc_g, acc_u);
            if (step + 1 < 9) {
                cg_cur = cg_nxt;
                cu_cur = cu_nxt;
            }
        }
        acc_g += __shfl_xor_sync(0xffffffffu, acc_g, 4);
        acc_g += __shfl_xor_sync(0xffffffffu, acc_g, 2);
        acc_g += __shfl_xor_sync(0xffffffffu, acc_g, 1);
        acc_u += __shfl_xor_sync(0xffffffffu, acc_u, 4);
        acc_u += __shfl_xor_sync(0xffffffffu, acc_u, 2);
        acc_u += __shfl_xor_sync(0xffffffffu, acc_u, 1);
        if (lane8 == 0) {
            const float fmul = FMT == 0 ? (__ldg(wfactor + e) * 128.0f) : 1.0f;
            const float gate = bf16_round(acc_g * fmul);
            const float up = bf16_round(acc_u * fmul);
            const float a = tanhf(gate / 4.f) * 4.f * (1.f / (1.f + expf(-gate))) * (tanhf(up / 25.f) * 25.f);
            const ActT ab = act::from_float(a);
            const int tok = toks[0];
            uint32_t kmask = uint32_t(mask >> (8 * tok)) & 0xffu;
            while (kmask) {
                const int k = __ffs(kmask) - 1;
                kmask &= kmask - 1;
                out[(static_cast<size_t>(tok) * TOPK + k) * I + pair_global] = ab;
            }
        }
    } else {
        float acc_g[MAXM], acc_u[MAXM];
#pragma unroll
        for (int ti = 0; ti < MAXM; ++ti) acc_g[ti] = acc_u[ti] = 0.f;
#pragma unroll 1
        for (int step = 0; step < 9; ++step) {
            Chunk<FMT> cg_nxt, cu_nxt;
            if (step + 1 < 9) {
                const int cn = 8 * (step + 1) + lane8;
                cg_nxt = load_chunk<FMT>(wg_ptr, wsg_ptr, cn, wpol);
                cu_nxt = load_chunk<FMT>(wu_ptr, wsu_ptr, cn, wpol);
            }
            const int c = 8 * step + lane8;
            if constexpr (FMT == 0) {
                const uint32_t sg2 = e4m3x2_to_half2_m8(cg_cur.s);
                const uint32_t su2 = e4m3x2_to_half2_m8(cu_cur.s);
                const float2 scg = __half22float2(*reinterpret_cast<const __half2*>(&sg2));
                const float2 scu = __half22float2(*reinterpret_cast<const __half2*>(&su2));
                const uint32_t gq[4] = {cg_cur.q.x, cg_cur.q.y, cg_cur.q.z, cg_cur.q.w};
                const uint32_t uq[4] = {cu_cur.q.x, cu_cur.q.y, cu_cur.q.z, cu_cur.q.w};
                float dg_h[2][MAXM], du_h[2][MAXM];
                for (int ti = 0; ti < ntok; ++ti) dg_h[0][ti] = dg_h[1][ti] = du_h[0][ti] = du_h[1][ti] = 0.f;
#pragma unroll
                for (int widx = 0; widx < 4; ++widx) {
                    float vg[8], vu[8];
                    e2m1_twice8(gq[widx], vg);
                    e2m1_twice8(uq[widx], vu);
                    const int half = widx >> 1;
                    for (int ti = 0; ti < ntok; ++ti) {
                        float4 x0, x1;
                        if (ti < STAGE_T) {
                            x0 = *reinterpret_cast<const float4*>(&s_x[ti][c][8 * widx]);
                            x1 = *reinterpret_cast<const float4*>(&s_x[ti][c][8 * widx + 4]);
                        } else {
                            load_act_chunk_l1(x + static_cast<size_t>(toks[ti]) * K, c, widx, x0, x1);
                        }
                        dg_h[half][ti] = dot8_f4(vg, x0, x1, dg_h[half][ti]);
                        du_h[half][ti] = dot8_f4(vu, x0, x1, du_h[half][ti]);
                    }
                }
                for (int ti = 0; ti < ntok; ++ti) {
                    acc_g[ti] = fmaf(dg_h[0][ti], scg.x, acc_g[ti]);
                    acc_g[ti] = fmaf(dg_h[1][ti], scg.y, acc_g[ti]);
                    acc_u[ti] = fmaf(du_h[0][ti], scu.x, acc_u[ti]);
                    acc_u[ti] = fmaf(du_h[1][ti], scu.y, acc_u[ti]);
                }
            } else {
                const float scg =
                    (cg_cur.s ? __uint_as_float(uint32_t(cg_cur.s) << 23) : __uint_as_float(0x00400000u)) * 256.0f;
                const float scu =
                    (cu_cur.s ? __uint_as_float(uint32_t(cu_cur.s) << 23) : __uint_as_float(0x00400000u)) * 256.0f;
                const uint32_t gq[8] = {cg_cur.q0.x, cg_cur.q0.y, cg_cur.q0.z, cg_cur.q0.w,
                                        cg_cur.q1.x, cg_cur.q1.y, cg_cur.q1.z, cg_cur.q1.w};
                const uint32_t uq[8] = {cu_cur.q0.x, cu_cur.q0.y, cu_cur.q0.z, cu_cur.q0.w,
                                        cu_cur.q1.x, cu_cur.q1.y, cu_cur.q1.z, cu_cur.q1.w};
                float dg[MAXM], du[MAXM];
                for (int ti = 0; ti < ntok; ++ti) dg[ti] = du[ti] = 0.f;
#pragma unroll
                for (int widx = 0; widx < 8; ++widx) {
                    const uint32_t glo = e4m3x2_to_half2_m8(gq[widx]), ghi = e4m3x2_to_half2_m8(gq[widx] >> 16);
                    const uint32_t ulo = e4m3x2_to_half2_m8(uq[widx]), uhi = e4m3x2_to_half2_m8(uq[widx] >> 16);
                    const float2 ga = __half22float2(*reinterpret_cast<const __half2*>(&glo));
                    const float2 gb = __half22float2(*reinterpret_cast<const __half2*>(&ghi));
                    const float2 ua = __half22float2(*reinterpret_cast<const __half2*>(&ulo));
                    const float2 ub = __half22float2(*reinterpret_cast<const __half2*>(&uhi));
                    for (int ti = 0; ti < ntok; ++ti) {
                        float4 xv;
                        if (ti < STAGE_T) {
                            xv = *reinterpret_cast<const float4*>(&s_x[ti][c][4 * widx]);
                        } else {
                            const ActT* xp = x + static_cast<size_t>(toks[ti]) * K + c * 32 + 4 * widx;
                            const uint2 u = *reinterpret_cast<const uint2*>(xp);
                            const float2 p0 = act::unpack2(u.x), p1 = act::unpack2(u.y);
                            xv = make_float4(p0.x, p0.y, p1.x, p1.y);
                        }
                        dg[ti] = fmaf(ga.x, xv.x, fmaf(ga.y, xv.y, fmaf(gb.x, xv.z, fmaf(gb.y, xv.w, dg[ti]))));
                        du[ti] = fmaf(ua.x, xv.x, fmaf(ua.y, xv.y, fmaf(ub.x, xv.z, fmaf(ub.y, xv.w, du[ti]))));
                    }
                }
                for (int ti = 0; ti < ntok; ++ti) {
                    acc_g[ti] = fmaf(dg[ti], scg, acc_g[ti]);
                    acc_u[ti] = fmaf(du[ti], scu, acc_u[ti]);
                }
            }
            if (step + 1 < 9) {
                cg_cur = cg_nxt;
                cu_cur = cu_nxt;
            }
        }
        const float fmul = FMT == 0 ? (__ldg(wfactor + e) * 128.0f) : 1.0f;
        for (int ti = 0; ti < ntok; ++ti) {
            float ag = acc_g[ti], au = acc_u[ti];
            ag += __shfl_xor_sync(0xffffffffu, ag, 4);
            ag += __shfl_xor_sync(0xffffffffu, ag, 2);
            ag += __shfl_xor_sync(0xffffffffu, ag, 1);
            au += __shfl_xor_sync(0xffffffffu, au, 4);
            au += __shfl_xor_sync(0xffffffffu, au, 2);
            au += __shfl_xor_sync(0xffffffffu, au, 1);
            if (lane8 == 0) {
                const float gate = bf16_round(ag * fmul);
                const float up = bf16_round(au * fmul);
                const float a = tanhf(gate / 4.f) * 4.f * (1.f / (1.f + expf(-gate))) * (tanhf(up / 25.f) * 25.f);
                const ActT ab = act::from_float(a);
                const int tok = toks[ti];
                uint32_t kmask = uint32_t(mask >> (8 * tok)) & 0xffu;
                while (kmask) {
                    const int k = __ffs(kmask) - 1;
                    kmask &= kmask - 1;
                    out[(static_cast<size_t>(tok) * TOPK + k) * I + pair_global] = ab;
                }
            }
        }
    }
}

template <int FMT, bool M1>
__global__ void __launch_bounds__(512)
moe_down_fp16_stream_kernel(const ActT* __restrict__ act, const int* __restrict__ idx, const float* __restrict__ wts,
                            const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                            const float* __restrict__ wfactor, int M, const float* __restrict__ conv_w,
                            float* __restrict__ conv_state, float* __restrict__ out, float* __restrict__ conv_hist,
                            const uint8_t* const* __restrict__ w_ptrs, const uint8_t* const* __restrict__ ws_ptrs) {
    constexpr int K = I, R = H;
    constexpr int CB = Layout<FMT>::code_bytes(K), SB = Layout<FMT>::scale_bytes(K);
    // 41,472 B padded s_act -> bank-conflict-free float4 loads across 4 lanes & 8-way broadcast across rows.
    __shared__ float s_act[TOPK][36][36];
    __shared__ float y_slot[M1 ? 1 : MAXM][TOPK][TILE];
    __shared__ int s_idx_m1[TOPK];
    __shared__ float s_wts_m1[TOPK];
    __shared__ int uexp[M1 ? 1 : TOPK * MAXM];
    __shared__ unsigned long long umask[M1 ? 1 : TOPK * MAXM];
    __shared__ uint8_t slot_u[M1 ? 1 : TOPK * MAXM];
    __shared__ int first_slot[M1 ? 1 : TOPK * MAXM];
    __shared__ int nuniq;
    __shared__ float sbuf[M1 ? 1 : MAXM][TILE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int tile = blockIdx.x;
    const int k_grp = warp >> 1;      // 0..7: expert within the round
    const int half_tile = warp & 1;   // 0..1: rows 0..7 or 8..15
    const int r_local = half_tile * 8 + (lane >> 2);
    const int lane4 = lane & 3;
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();

    if constexpr (M1) {
        if (tid < TOPK) {
            s_idx_m1[tid] = ld_dep(idx + tid);
            s_wts_m1[tid] = ld_dep(wts + tid);
        }
        // Cooperatively load all 8 experts' activations (8 * 1152 = 9216 FP16 = 1152 uint4s) into s_act.
        for (int j = tid; j < TOPK * 36 * 4; j += 512) {
            const int slot = j / 144;
            const int rem = j - slot * 144;
            const int c = rem >> 2;
            const int sub = (rem & 3) * 8;
            const uint4 q = ld_dep(act + static_cast<size_t>(slot) * K + c * 32 + sub);
            const float2 p0 = act::unpack2(q.x), p1 = act::unpack2(q.y), p2 = act::unpack2(q.z), p3 = act::unpack2(q.w);
            *reinterpret_cast<float4*>(&s_act[slot][c][sub]) = make_float4(p0.x, p0.y, p1.x, p1.y);
            *reinterpret_cast<float4*>(&s_act[slot][c][sub + 4]) = make_float4(p2.x, p2.y, p3.x, p3.y);
        }
        __syncthreads();

        const int e = s_idx_m1[k_grp];
        const bool local = !w_ptrs || (w_ptrs[e] != nullptr);
        if (local) {
            const size_t local_row = static_cast<size_t>(tile * TILE + r_local);
            const size_t row = static_cast<size_t>(e) * R + local_row;
            const uint8_t* we = w_ptrs ? (w_ptrs[e] + local_row * CB) : (w + row * CB);
            const uint8_t* wse = ws_ptrs ? (ws_ptrs[e] + local_row * SB) : (ws + row * SB);
            Chunk<FMT> cur = load_chunk<FMT>(we, wse, lane4, wpol);
            float acc = 0.f;
#pragma unroll
            for (int step = 0; step < 9; ++step) {
                Chunk<FMT> nxt;
                if (step + 1 < 9) nxt = load_chunk<FMT>(we, wse, 4 * (step + 1) + lane4, wpol);
                dot_chunk_1tok<FMT>(cur, &s_act[k_grp][4 * step + lane4][0], acc);
                if (step + 1 < 9) cur = nxt;
            }
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);
            acc += __shfl_xor_sync(0xffffffffu, acc, 2);
            if (lane4 == 0) {
                const float fmul = FMT == 0 ? (__ldg(wfactor + e) * 128.0f) : 1.0f;
                y_slot[0][k_grp][r_local] = bf16_round(acc * fmul);
            }
        } else if (lane4 == 0) {
            y_slot[0][k_grp][r_local] = 0.f;
        }
        __syncthreads();

        if (tid < TILE) {
            const int ch = tile * TILE + tid;
            float s = 0.f;
#pragma unroll
            for (int k = 0; k < TOPK; ++k) {
                if (!w_ptrs || w_ptrs[s_idx_m1[k]]) s = __fadd_rn(s, __fmul_rn(s_wts_m1[k], y_slot[0][k][tid]));
            }
            if (!conv_w) {
                out[ch] = s;
                return;
            }
            const float sr = bf16_round(s);
            const float4 cw = *reinterpret_cast<const float4*>(conv_w + 4 * ch);
            const float e0 = ld_dep(conv_state + ch);
            const float e1 = ld_dep(conv_state + H + ch);
            const float e2 = ld_dep(conv_state + 2 * H + ch);
            constexpr float inv_sqrt5 = 1.f / 2.2360679774997896f;
            float acc = __fmul_rn(cw.x, e0);
            acc = __fadd_rn(acc, __fmul_rn(cw.y, e1));
            acc = __fadd_rn(acc, __fmul_rn(cw.z, e2));
            acc = __fadd_rn(acc, __fmul_rn(cw.w, sr));
            out[ch] = __fmul_rn(__fadd_rn(acc, sr), inv_sqrt5);
            if (conv_hist) conv_hist[ch] = sr;
            conv_state[ch] = e1;
            conv_state[H + ch] = e2;
            conv_state[2 * H + ch] = sr;
        }
    } else {
        const int nslots = M * TOPK;
        if (warp == 0) {
            const int lo = lane < nslots ? ld_dep(idx + lane) : -1;
            const int hi = lane + 32 < nslots ? ld_dep(idx + lane + 32) : -1;
            uint64_t remaining = nslots == 64 ? ~0ull : ((1ull << nslots) - 1);
            int u = 0;
            while (remaining) {
                const int s = __ffsll(static_cast<long long>(remaining)) - 1;
                const int e = __shfl_sync(0xffffffffu, s < 32 ? lo : hi, s & 31);
                const uint64_t mask = slot_mask(e, lo, hi);
                remaining &= ~mask;
                const bool local = !w_ptrs || (w_ptrs[e] != nullptr);
                if (local) {
                    if (lane == 0) uexp[u] = e, umask[u] = mask, first_slot[u] = s;
                    if ((mask >> lane) & 1) slot_u[lane] = uint8_t(u);
                    if ((mask >> (lane + 32)) & 1) slot_u[lane + 32] = uint8_t(u);
                    ++u;
                } else {
                    if ((mask >> lane) & 1) slot_u[lane] = uint8_t(255);
                    if ((mask >> (lane + 32)) & 1) slot_u[lane + 32] = uint8_t(255);
                }
            }
            if (lane == 0) nuniq = u;
        }
        __syncthreads();

        const int U = nuniq;
#pragma unroll 1
        for (int base = 0; base < U; base += TOPK) {
            const int nbatch = (U - base) < TOPK ? (U - base) : TOPK;
            for (int j = tid; j < nbatch * 144; j += 512) {
                const int kb = j / 144;
                const int rem = j - kb * 144;
                const int c = rem >> 2;
                const int sub = (rem & 3) * 8;
                const int fsl = first_slot[base + kb];
                const uint4 q = ld_dep(act + static_cast<size_t>(fsl) * K + c * 32 + sub);
                const float2 p0 = act::unpack2(q.x), p1 = act::unpack2(q.y), p2 = act::unpack2(q.z), p3 = act::unpack2(q.w);
                *reinterpret_cast<float4*>(&s_act[kb][c][sub]) = make_float4(p0.x, p0.y, p1.x, p1.y);
                *reinterpret_cast<float4*>(&s_act[kb][c][sub + 4]) = make_float4(p2.x, p2.y, p3.x, p3.y);
            }
            __syncthreads();

            const int u = base + k_grp;
            if (u < U) {
                const int e = uexp[u];
                const uint64_t mask = umask[u];
                int toks[MAXM], tslot[MAXM];
                int ntok = 0;
                for (int m = 0; m < M; ++m) {
                    const uint32_t km = uint32_t(mask >> (8 * m)) & 0xffu;
                    if (km) {
                        toks[ntok] = m;
                        tslot[ntok] = m * TOPK + (__ffs(km) - 1);
                        ++ntok;
                    }
                }
                const size_t local_row = static_cast<size_t>(tile * TILE + r_local);
                const size_t row = static_cast<size_t>(e) * R + local_row;
                const uint8_t* we = w_ptrs ? (w_ptrs[e] + local_row * CB) : (w + row * CB);
                const uint8_t* wse = ws_ptrs ? (ws_ptrs[e] + local_row * SB) : (ws + row * SB);
                Chunk<FMT> cur = load_chunk<FMT>(we, wse, lane4, wpol);
                const float fmul = FMT == 0 ? (__ldg(wfactor + e) * 128.0f) : 1.0f;

                if (ntok == 1) {
                    float acc = 0.f;
#pragma unroll
                    for (int step = 0; step < 9; ++step) {
                        Chunk<FMT> nxt;
                        if (step + 1 < 9) nxt = load_chunk<FMT>(we, wse, 4 * (step + 1) + lane4, wpol);
                        dot_chunk_1tok<FMT>(cur, &s_act[k_grp][4 * step + lane4][0], acc);
                        if (step + 1 < 9) cur = nxt;
                    }
                    acc += __shfl_xor_sync(0xffffffffu, acc, 1);
                    acc += __shfl_xor_sync(0xffffffffu, acc, 2);
                    if (lane4 == 0) {
                        const float yr = bf16_round(acc * fmul);
                        const int tok = toks[0];
                        uint32_t km = uint32_t(mask >> (8 * tok)) & 0xffu;
                        while (km) {
                            const int sk = __ffs(km) - 1;
                            km &= km - 1;
                            y_slot[tok][sk][r_local] = yr;
                        }
                    }
                } else {
                    float acc[MAXM];
#pragma unroll
                    for (int ti = 0; ti < MAXM; ++ti) acc[ti] = 0.f;
#pragma unroll 1
                    for (int step = 0; step < 9; ++step) {
                        Chunk<FMT> nxt;
                        if (step + 1 < 9) nxt = load_chunk<FMT>(we, wse, 4 * (step + 1) + lane4, wpol);
                        const int c = 4 * step + lane4;
                        if constexpr (FMT == 0) {
                            const uint32_t s2 = e4m3x2_to_half2_m8(cur.s);
                            const float2 sc = __half22float2(*reinterpret_cast<const __half2*>(&s2));
                            const uint32_t qv[4] = {cur.q.x, cur.q.y, cur.q.z, cur.q.w};
                            float dh[2][MAXM];
                            for (int ti = 0; ti < ntok; ++ti) dh[0][ti] = dh[1][ti] = 0.f;
#pragma unroll
                            for (int widx = 0; widx < 4; ++widx) {
                                float v[8];
                                e2m1_twice8(qv[widx], v);
                                const int half = widx >> 1;
                                for (int ti = 0; ti < ntok; ++ti) {
                                    float4 x0, x1;
                                    if (ti == 0) {
                                        x0 = *reinterpret_cast<const float4*>(&s_act[k_grp][c][8 * widx]);
                                        x1 = *reinterpret_cast<const float4*>(&s_act[k_grp][c][8 * widx + 4]);
                                    } else {
                                        load_act_chunk_l1(act + static_cast<size_t>(tslot[ti]) * K, c, widx, x0, x1);
                                    }
                                    dh[half][ti] = dot8_f4(v, x0, x1, dh[half][ti]);
                                }
                            }
                            for (int ti = 0; ti < ntok; ++ti) {
                                acc[ti] = fmaf(dh[0][ti], sc.x, acc[ti]);
                                acc[ti] = fmaf(dh[1][ti], sc.y, acc[ti]);
                            }
                        } else {
                            const float sc256 =
                                (cur.s ? __uint_as_float(uint32_t(cur.s) << 23) : __uint_as_float(0x00400000u)) * 256.0f;
                            const uint32_t qv[8] = {cur.q0.x, cur.q0.y, cur.q0.z, cur.q0.w,
                                                    cur.q1.x, cur.q1.y, cur.q1.z, cur.q1.w};
                            float d[MAXM];
                            for (int ti = 0; ti < ntok; ++ti) d[ti] = 0.f;
#pragma unroll
                            for (int widx = 0; widx < 8; ++widx) {
                                const uint32_t hlo = e4m3x2_to_half2_m8(qv[widx]);
                                const uint32_t hhi = e4m3x2_to_half2_m8(qv[widx] >> 16);
                                const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&hlo));
                                const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&hhi));
                                for (int ti = 0; ti < ntok; ++ti) {
                                    float4 xv;
                                    if (ti == 0) {
                                        xv = *reinterpret_cast<const float4*>(&s_act[k_grp][c][4 * widx]);
                                    } else {
                                        const ActT* xp = act + static_cast<size_t>(tslot[ti]) * K + c * 32 + 4 * widx;
                                        const uint2 u2 = *reinterpret_cast<const uint2*>(xp);
                                        const float2 p0 = act::unpack2(u2.x), p1 = act::unpack2(u2.y);
                                        xv = make_float4(p0.x, p0.y, p1.x, p1.y);
                                    }
                                    d[ti] = fmaf(a.x, xv.x, fmaf(a.y, xv.y, fmaf(b.x, xv.z, fmaf(b.y, xv.w, d[ti]))));
                                }
                            }
                            for (int ti = 0; ti < ntok; ++ti) acc[ti] = fmaf(d[ti], sc256, acc[ti]);
                        }
                        if (step + 1 < 9) cur = nxt;
                    }
                    for (int ti = 0; ti < ntok; ++ti) {
                        float a = acc[ti];
                        a += __shfl_xor_sync(0xffffffffu, a, 1);
                        a += __shfl_xor_sync(0xffffffffu, a, 2);
                        if (lane4 == 0) {
                            const float yr = bf16_round(a * fmul);
                            const int tok = toks[ti];
                            uint32_t km = uint32_t(mask >> (8 * tok)) & 0xffu;
                            while (km) {
                                const int sk = __ffs(km) - 1;
                                km &= km - 1;
                                y_slot[tok][sk][r_local] = yr;
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }

        if (tid < TILE * M) {
            const int r = tid & (TILE - 1), n = tid / TILE;
            float s = 0.f;
#pragma unroll
            for (int k = 0; k < TOPK; ++k) {
                const int sl = n * TOPK + k;
                if (slot_u[sl] != 255) s = __fadd_rn(s, __fmul_rn(ld_dep(wts + sl), y_slot[n][k][r]));
            }
            if (!conv_w) {
                out[static_cast<size_t>(n) * H + tile * TILE + r] = s;
                return;
            }
            sbuf[n][r] = bf16_round(s);
        }
        if (!conv_w) return;
        __syncthreads();

        if (tid < TILE) {
            const int ch = tile * TILE + tid;
            const float4 cw = *reinterpret_cast<const float4*>(conv_w + 4 * ch);
            float ext[MAXM + 3];
            ext[0] = ld_dep(conv_state + ch);
            ext[1] = ld_dep(conv_state + H + ch);
            ext[2] = ld_dep(conv_state + 2 * H + ch);
#pragma unroll
            for (int m = 0; m < MAXM; ++m) ext[m + 3] = m < M ? sbuf[m][tid] : 0.f;
            constexpr float inv_sqrt5 = 1.f / 2.2360679774997896f;
#pragma unroll
            for (int m = 0; m < MAXM; ++m) {
                if (m < M) {
                    float acc = __fmul_rn(cw.x, ext[m]);
                    acc = __fadd_rn(acc, __fmul_rn(cw.y, ext[m + 1]));
                    acc = __fadd_rn(acc, __fmul_rn(cw.z, ext[m + 2]));
                    acc = __fadd_rn(acc, __fmul_rn(cw.w, ext[m + 3]));
                    out[static_cast<size_t>(m) * H + ch] = __fmul_rn(__fadd_rn(acc, ext[m + 3]), inv_sqrt5);
                    if (conv_hist) conv_hist[static_cast<size_t>(m) * H + ch] = ext[m + 3];
                }
            }
#pragma unroll
            for (int j = 0; j < 3; ++j)
#pragma unroll
                for (int m = 1; m <= MAXM; ++m)
                    if (m == M) conv_state[j * H + ch] = ext[m + j];
        }
    }
}

// ----------------------------------------------------------------- NVFP4 INT8-act dp4a MoE kernels
// Adapted from ashalliants/ninfer-3090 (src/ops/moe_situ/moe_situ.cu and
// src/ops/linear/common/stream_dp4a.cuh @ 8ea67ce, Apache-2.0):
// Per-16-element symmetric INT8 activation quantization + exact signed 2*e2m1 __dp4a dot products.
// Identical arithmetic across M = 1..8 (preserves M=1 vs speculative verify parity) and stays
// HBM2-bandwidth-bound across all active-token widths on sm_61 / sm_70 / sm_75.

__device__ __forceinline__ int dp4a_s8(int32_t a, int32_t b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int a0 = static_cast<int8_t>(a & 0xff), a1 = static_cast<int8_t>((a >> 8) & 0xff);
    const int a2 = static_cast<int8_t>((a >> 16) & 0xff), a3 = static_cast<int8_t>((a >> 24) & 0xff);
    const int b0 = static_cast<int8_t>(b & 0xff), b1 = static_cast<int8_t>((b >> 8) & 0xff);
    const int b2 = static_cast<int8_t>((b >> 16) & 0xff), b3 = static_cast<int8_t>((b >> 24) & 0xff);
    return c + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
#endif
}

// Layout inside caller-owned fixed-address scratch for M > 1 NVFP4 dp4a pre-quantization.
// Each Runtime/Decoder (or standalone multi-stream caller) owns its own fixed-address scratch buffer so
// concurrent generators and concurrently replayed CUDA graphs never alias intermediate INT8 activations.
constexpr std::size_t kDp4aUpQ8Elems = static_cast<std::size_t>(MAXM) * 72 * 8;
constexpr std::size_t kDp4aUpXsElems = static_cast<std::size_t>(MAXM) * 144;
constexpr std::size_t kDp4aDownQ8Elems = static_cast<std::size_t>(MAXM) * TOPK * 36 * 8;
constexpr std::size_t kDp4aDownXsElems = static_cast<std::size_t>(MAXM) * TOPK * 72;
static_assert(kDp4aUpQ8Elems * sizeof(int32_t) + kDp4aUpXsElems * sizeof(float) <= kDp4aScratchBytes);
static_assert(kDp4aDownQ8Elems * sizeof(int32_t) + kDp4aDownXsElems * sizeof(float) <= kDp4aScratchBytes);
static_assert((kDp4aUpQ8Elems * sizeof(int32_t)) % 16 == 0 && (kDp4aDownQ8Elems * sizeof(int32_t)) % 16 == 0);

template <int K_CHUNKS>
__global__ void __launch_bounds__(128)
moe_quant_i8_kernel(const ActT* __restrict__ src, int32_t* __restrict__ q8_out, float* __restrict__ xs_out) {
    constexpr int K = K_CHUNKS * 32;
    constexpr int SUBS = K_CHUNKS * 2;
    const int row = blockIdx.x;
    const ActT* row_src = src + static_cast<size_t>(row) * K;
    int32_t* row_q8 = q8_out + static_cast<size_t>(row) * (K_CHUNKS * 8);
    float* row_xs = xs_out + static_cast<size_t>(row) * SUBS;
    for (int j = threadIdx.x; j < SUBS; j += 128) {
        const int c = j >> 1;
        const int half = j & 1;
        int32_t q4[4];
        float sc;
        quant16_act_to_i8(row_src + c * 32 + half * 16, q4, sc);
        *reinterpret_cast<int4*>(row_q8 + c * 8 + half * 4) = make_int4(q4[0], q4[1], q4[2], q4[3]);
        row_xs[j] = sc;
    }
}

__global__ void __launch_bounds__(256)
moe_up_nvfp4_dp4a_m1_kernel(const ActT* __restrict__ x, const int* __restrict__ idx, const uint8_t* __restrict__ w,
                            const uint8_t* __restrict__ ws, const float* __restrict__ wfactor, int M,
                            ActT* __restrict__ out, const uint8_t* const* __restrict__ w_ptrs,
                            const uint8_t* const* __restrict__ ws_ptrs) {
    constexpr int K = H, R = 2 * I;
    constexpr int CB = Layout<0>::code_bytes(K), SB = Layout<0>::scale_bytes(K);
    __shared__ int32_t s_q8[72][9];
    __shared__ float2 s_xs[72];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int slot = blockIdx.y;
    const int lo = lane < TOPK ? ld_dep(idx + lane) : -1;
    const int e = ld_dep(idx + slot);
    if (w_ptrs && !w_ptrs[e]) return;
    const uint64_t mask = slot_mask(e, lo, -1);
    if (__ffsll(static_cast<long long>(mask)) - 1 != slot) return;

    if (tid < 144) {
        const int c = tid >> 1;
        const int half = tid & 1;
        float sc;
        quant16_act_to_i8(x + c * 32 + half * 16, &s_q8[c][half * 4], sc);
        if (half == 0) s_xs[c].x = sc;
        else s_xs[c].y = sc;
    }
    __syncthreads();

    const int pair_local = tid >> 3;  // 0..31
    const int lane8 = tid & 7;        // 0..7
    const int pair_global = blockIdx.x * 32 + pair_local;
    const size_t local_row_g = static_cast<size_t>(2 * pair_global);
    const size_t row_g = static_cast<size_t>(e) * R + local_row_g;
    const uint8_t* wg_ptr = w_ptrs ? (w_ptrs[e] + local_row_g * CB) : (w + row_g * CB);
    const uint8_t* wsg_ptr = ws_ptrs ? (ws_ptrs[e] + local_row_g * SB) : (ws + row_g * SB);
    const uint8_t* wu_ptr = wg_ptr + CB;
    const uint8_t* wsu_ptr = wsg_ptr + SB;
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();

    float acc_g = 0.f, acc_u = 0.f;
    Chunk<0> cg_cur = load_chunk<0>(wg_ptr, wsg_ptr, lane8, wpol);
    Chunk<0> cu_cur = load_chunk<0>(wu_ptr, wsu_ptr, lane8, wpol);
#pragma unroll
    for (int step = 0; step < 9; ++step) {
        Chunk<0> cg_nxt, cu_nxt;
        if (step + 1 < 9) {
            const int cn = 8 * (step + 1) + lane8;
            cg_nxt = load_chunk<0>(wg_ptr, wsg_ptr, cn, wpol);
            cu_nxt = load_chunk<0>(wu_ptr, wsu_ptr, cn, wpol);
        }
        const int c = 8 * step + lane8;
        const int32_t x0 = s_q8[c][0], x1 = s_q8[c][1], x2 = s_q8[c][2], x3 = s_q8[c][3];
        const int32_t x4 = s_q8[c][4], x5 = s_q8[c][5], x6 = s_q8[c][6], x7 = s_q8[c][7];
        const float2 xs = s_xs[c];
        int32_t w0, w1;
        int ig0 = 0, ig1 = 0, iu0 = 0, iu1 = 0;
        e2m1_twice_i8x4(cg_cur.q.x, w0, w1); ig0 = dp4a_s8(w0, x0, ig0); ig0 = dp4a_s8(w1, x1, ig0);
        e2m1_twice_i8x4(cg_cur.q.y, w0, w1); ig0 = dp4a_s8(w0, x2, ig0); ig0 = dp4a_s8(w1, x3, ig0);
        e2m1_twice_i8x4(cg_cur.q.z, w0, w1); ig1 = dp4a_s8(w0, x4, ig1); ig1 = dp4a_s8(w1, x5, ig1);
        e2m1_twice_i8x4(cg_cur.q.w, w0, w1); ig1 = dp4a_s8(w0, x6, ig1); ig1 = dp4a_s8(w1, x7, ig1);
        e2m1_twice_i8x4(cu_cur.q.x, w0, w1); iu0 = dp4a_s8(w0, x0, iu0); iu0 = dp4a_s8(w1, x1, iu0);
        e2m1_twice_i8x4(cu_cur.q.y, w0, w1); iu0 = dp4a_s8(w0, x2, iu0); iu0 = dp4a_s8(w1, x3, iu0);
        e2m1_twice_i8x4(cu_cur.q.z, w0, w1); iu1 = dp4a_s8(w0, x4, iu1); iu1 = dp4a_s8(w1, x5, iu1);
        e2m1_twice_i8x4(cu_cur.q.w, w0, w1); iu1 = dp4a_s8(w0, x6, iu1); iu1 = dp4a_s8(w1, x7, iu1);
        const uint32_t sg2 = e4m3x2_to_half2_m8(cg_cur.s);
        const uint32_t su2 = e4m3x2_to_half2_m8(cu_cur.s);
        const float2 scg = __half22float2(*reinterpret_cast<const __half2*>(&sg2));
        const float2 scu = __half22float2(*reinterpret_cast<const __half2*>(&su2));
        acc_g = fmaf(float(ig0), scg.x * xs.x, fmaf(float(ig1), scg.y * xs.y, acc_g));
        acc_u = fmaf(float(iu0), scu.x * xs.x, fmaf(float(iu1), scu.y * xs.y, acc_u));
        if (step + 1 < 9) {
            cg_cur = cg_nxt;
            cu_cur = cu_nxt;
        }
    }
    acc_g += __shfl_xor_sync(0xffffffffu, acc_g, 4);
    acc_g += __shfl_xor_sync(0xffffffffu, acc_g, 2);
    acc_g += __shfl_xor_sync(0xffffffffu, acc_g, 1);
    acc_u += __shfl_xor_sync(0xffffffffu, acc_u, 4);
    acc_u += __shfl_xor_sync(0xffffffffu, acc_u, 2);
    acc_u += __shfl_xor_sync(0xffffffffu, acc_u, 1);
    if (lane8 == 0) {
        const float fmul = __ldg(wfactor + e) * 128.0f;
        const float gate = bf16_round(acc_g * fmul);
        const float up = bf16_round(acc_u * fmul);
        const float a = tanhf(gate / 4.f) * 4.f * (1.f / (1.f + expf(-gate))) * (tanhf(up / 25.f) * 25.f);
        const ActT ab = act::from_float(a);
        uint32_t kmask = uint32_t(mask) & 0xffu;
        while (kmask) {
            const int k = __ffs(kmask) - 1;
            kmask &= kmask - 1;
            out[static_cast<size_t>(k) * I + pair_global] = ab;
        }
    }
}

template <int MAX_T, bool HAS_WS>
__global__ void __launch_bounds__(256)
moe_up_nvfp4_dp4a_l1_kernel(const ActT* __restrict__ x_in, const int32_t* __restrict__ q8_in,
                            const float* __restrict__ xs_in, const int* __restrict__ idx, const uint8_t* __restrict__ w,
                            const uint8_t* __restrict__ ws, const float* __restrict__ wfactor, int M,
                            ActT* __restrict__ out, const uint8_t* const* __restrict__ w_ptrs,
                            const uint8_t* const* __restrict__ ws_ptrs) {
    constexpr int K = H, R = 2 * I;
    constexpr int CB = Layout<0>::code_bytes(K), SB = Layout<0>::scale_bytes(K);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int slot = blockIdx.y;
    const int nslots = M * TOPK;
    const int lo = lane < nslots ? ld_dep(idx + lane) : -1;
    const int hi = lane + 32 < nslots ? ld_dep(idx + lane + 32) : -1;
    const int e = ld_dep(idx + slot);
    if (w_ptrs && !w_ptrs[e]) return;
    const uint64_t mask = slot_mask(e, lo, hi);
    if (__ffsll(static_cast<long long>(mask)) - 1 != slot) return;

    int toks[MAX_T];
    int ntok = 0;
    for (int m = 0; m < M; ++m) {
        if ((mask >> (8 * m)) & 0xffu) toks[ntok++] = m;
    }

    const int pair_local = tid >> 3;  // 0..31
    const int lane8 = tid & 7;        // 0..7
    const int pair_global = blockIdx.x * 32 + pair_local;
    const size_t local_row_g = static_cast<size_t>(2 * pair_global);
    const size_t row_g = static_cast<size_t>(e) * R + local_row_g;
    const uint8_t* wg_ptr = w_ptrs ? (w_ptrs[e] + local_row_g * CB) : (w + row_g * CB);
    const uint8_t* wsg_ptr = ws_ptrs ? (ws_ptrs[e] + local_row_g * SB) : (ws + row_g * SB);
    const uint8_t* wu_ptr = wg_ptr + CB;
    const uint8_t* wsu_ptr = wsg_ptr + SB;
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();

    if (ntok == 1) {
        const int tok = toks[0];
        float acc_g = 0.f, acc_u = 0.f;
        Chunk<0> cg_cur = load_chunk<0>(wg_ptr, wsg_ptr, lane8, wpol);
        Chunk<0> cu_cur = load_chunk<0>(wu_ptr, wsu_ptr, lane8, wpol);
#pragma unroll
        for (int step = 0; step < 9; ++step) {
            Chunk<0> cg_nxt, cu_nxt;
            if (step + 1 < 9) {
                const int cn = 8 * (step + 1) + lane8;
                cg_nxt = load_chunk<0>(wg_ptr, wsg_ptr, cn, wpol);
                cu_nxt = load_chunk<0>(wu_ptr, wsu_ptr, cn, wpol);
            }
            const int c = 8 * step + lane8;
            int4 xv0, xv1;
            float2 xs;
            if constexpr (HAS_WS) {
                const int32_t* xq_row = q8_in + tok * (72 * 8);
                const float2* xs_row = reinterpret_cast<const float2*>(xs_in + tok * 144);
                xv0 = __ldg(reinterpret_cast<const int4*>(xq_row + c * 8));
                xv1 = __ldg(reinterpret_cast<const int4*>(xq_row + c * 8 + 4));
                xs = __ldg(xs_row + c);
            } else {
                int32_t q0[4], q1[4];
                const ActT* src_c = x_in + static_cast<size_t>(tok) * H + c * 32;
                quant16_act_to_i8(src_c, q0, xs.x);
                quant16_act_to_i8(src_c + 16, q1, xs.y);
                xv0 = make_int4(q0[0], q0[1], q0[2], q0[3]);
                xv1 = make_int4(q1[0], q1[1], q1[2], q1[3]);
            }
            int32_t w0, w1;
            int ig0 = 0, ig1 = 0, iu0 = 0, iu1 = 0;
            e2m1_twice_i8x4(cg_cur.q.x, w0, w1); ig0 = dp4a_s8(w0, xv0.x, ig0); ig0 = dp4a_s8(w1, xv0.y, ig0);
            e2m1_twice_i8x4(cg_cur.q.y, w0, w1); ig0 = dp4a_s8(w0, xv0.z, ig0); ig0 = dp4a_s8(w1, xv0.w, ig0);
            e2m1_twice_i8x4(cg_cur.q.z, w0, w1); ig1 = dp4a_s8(w0, xv1.x, ig1); ig1 = dp4a_s8(w1, xv1.y, ig1);
            e2m1_twice_i8x4(cg_cur.q.w, w0, w1); ig1 = dp4a_s8(w0, xv1.z, ig1); ig1 = dp4a_s8(w1, xv1.w, ig1);
            e2m1_twice_i8x4(cu_cur.q.x, w0, w1); iu0 = dp4a_s8(w0, xv0.x, iu0); iu0 = dp4a_s8(w1, xv0.y, iu0);
            e2m1_twice_i8x4(cu_cur.q.y, w0, w1); iu0 = dp4a_s8(w0, xv0.z, iu0); iu0 = dp4a_s8(w1, xv0.w, iu0);
            e2m1_twice_i8x4(cu_cur.q.z, w0, w1); iu1 = dp4a_s8(w0, xv1.x, iu1); iu1 = dp4a_s8(w1, xv1.y, iu1);
            e2m1_twice_i8x4(cu_cur.q.w, w0, w1); iu1 = dp4a_s8(w0, xv1.z, iu1); iu1 = dp4a_s8(w1, xv1.w, iu1);
            const uint32_t sg2 = e4m3x2_to_half2_m8(cg_cur.s);
            const uint32_t su2 = e4m3x2_to_half2_m8(cu_cur.s);
            const float2 scg = __half22float2(*reinterpret_cast<const __half2*>(&sg2));
            const float2 scu = __half22float2(*reinterpret_cast<const __half2*>(&su2));
            acc_g = fmaf(float(ig0), scg.x * xs.x, fmaf(float(ig1), scg.y * xs.y, acc_g));
            acc_u = fmaf(float(iu0), scu.x * xs.x, fmaf(float(iu1), scu.y * xs.y, acc_u));
            if (step + 1 < 9) {
                cg_cur = cg_nxt;
                cu_cur = cu_nxt;
            }
        }
        acc_g += __shfl_xor_sync(0xffffffffu, acc_g, 4);
        acc_g += __shfl_xor_sync(0xffffffffu, acc_g, 2);
        acc_g += __shfl_xor_sync(0xffffffffu, acc_g, 1);
        acc_u += __shfl_xor_sync(0xffffffffu, acc_u, 4);
        acc_u += __shfl_xor_sync(0xffffffffu, acc_u, 2);
        acc_u += __shfl_xor_sync(0xffffffffu, acc_u, 1);
        if (lane8 == 0) {
            const float fmul = __ldg(wfactor + e) * 128.0f;
            const float gate = bf16_round(acc_g * fmul);
            const float up = bf16_round(acc_u * fmul);
            const float a = tanhf(gate / 4.f) * 4.f * (1.f / (1.f + expf(-gate))) * (tanhf(up / 25.f) * 25.f);
            const ActT ab = act::from_float(a);
            uint32_t kmask = uint32_t(mask >> (8 * tok)) & 0xffu;
            while (kmask) {
                const int k = __ffs(kmask) - 1;
                kmask &= kmask - 1;
                out[(static_cast<size_t>(tok) * TOPK + k) * I + pair_global] = ab;
            }
        }
    } else {
        Chunk<0> cg_cur = load_chunk<0>(wg_ptr, wsg_ptr, lane8, wpol);
        Chunk<0> cu_cur = load_chunk<0>(wu_ptr, wsu_ptr, lane8, wpol);
        float acc_g[MAX_T], acc_u[MAX_T];
#pragma unroll
        for (int ti = 0; ti < MAX_T; ++ti) acc_g[ti] = acc_u[ti] = 0.f;
#pragma unroll 1
        for (int step = 0; step < 9; ++step) {
            Chunk<0> cg_nxt, cu_nxt;
            if (step + 1 < 9) {
                const int cn = 8 * (step + 1) + lane8;
                cg_nxt = load_chunk<0>(wg_ptr, wsg_ptr, cn, wpol);
                cu_nxt = load_chunk<0>(wu_ptr, wsu_ptr, cn, wpol);
            }
            const int c = 8 * step + lane8;
            int32_t gw[8], uw[8];
            e2m1_twice_i8x4(cg_cur.q.x, gw[0], gw[1]);
            e2m1_twice_i8x4(cg_cur.q.y, gw[2], gw[3]);
            e2m1_twice_i8x4(cg_cur.q.z, gw[4], gw[5]);
            e2m1_twice_i8x4(cg_cur.q.w, gw[6], gw[7]);
            e2m1_twice_i8x4(cu_cur.q.x, uw[0], uw[1]);
            e2m1_twice_i8x4(cu_cur.q.y, uw[2], uw[3]);
            e2m1_twice_i8x4(cu_cur.q.z, uw[4], uw[5]);
            e2m1_twice_i8x4(cu_cur.q.w, uw[6], uw[7]);
            const uint32_t sg2 = e4m3x2_to_half2_m8(cg_cur.s);
            const uint32_t su2 = e4m3x2_to_half2_m8(cu_cur.s);
            const float2 scg = __half22float2(*reinterpret_cast<const __half2*>(&sg2));
            const float2 scu = __half22float2(*reinterpret_cast<const __half2*>(&su2));
            for (int ti = 0; ti < ntok; ++ti) {
                const int tok = toks[ti];
                int4 xv0, xv1;
                float2 xs;
                if constexpr (HAS_WS) {
                    xv0 = __ldg(reinterpret_cast<const int4*>(q8_in + tok * (72 * 8) + c * 8));
                    xv1 = __ldg(reinterpret_cast<const int4*>(q8_in + tok * (72 * 8) + c * 8 + 4));
                    xs = __ldg(reinterpret_cast<const float2*>(xs_in + tok * 144) + c);
                } else {
                    int32_t q0[4], q1[4];
                    const ActT* src_c = x_in + static_cast<size_t>(tok) * H + c * 32;
                    quant16_act_to_i8(src_c, q0, xs.x);
                    quant16_act_to_i8(src_c + 16, q1, xs.y);
                    xv0 = make_int4(q0[0], q0[1], q0[2], q0[3]);
                    xv1 = make_int4(q1[0], q1[1], q1[2], q1[3]);
                }
                int ig0 = dp4a_s8(gw[0], xv0.x, dp4a_s8(gw[1], xv0.y, dp4a_s8(gw[2], xv0.z, dp4a_s8(gw[3], xv0.w, 0))));
                int ig1 = dp4a_s8(gw[4], xv1.x, dp4a_s8(gw[5], xv1.y, dp4a_s8(gw[6], xv1.z, dp4a_s8(gw[7], xv1.w, 0))));
                int iu0 = dp4a_s8(uw[0], xv0.x, dp4a_s8(uw[1], xv0.y, dp4a_s8(uw[2], xv0.z, dp4a_s8(uw[3], xv0.w, 0))));
                int iu1 = dp4a_s8(uw[4], xv1.x, dp4a_s8(uw[5], xv1.y, dp4a_s8(uw[6], xv1.z, dp4a_s8(uw[7], xv1.w, 0))));
                acc_g[ti] = fmaf(float(ig0), scg.x * xs.x, fmaf(float(ig1), scg.y * xs.y, acc_g[ti]));
                acc_u[ti] = fmaf(float(iu0), scu.x * xs.x, fmaf(float(iu1), scu.y * xs.y, acc_u[ti]));
            }
            if (step + 1 < 9) {
                cg_cur = cg_nxt;
                cu_cur = cu_nxt;
            }
        }
        const float fmul = __ldg(wfactor + e) * 128.0f;
        for (int ti = 0; ti < ntok; ++ti) {
            float ag = acc_g[ti], au = acc_u[ti];
            ag += __shfl_xor_sync(0xffffffffu, ag, 4);
            ag += __shfl_xor_sync(0xffffffffu, ag, 2);
            ag += __shfl_xor_sync(0xffffffffu, ag, 1);
            au += __shfl_xor_sync(0xffffffffu, au, 4);
            au += __shfl_xor_sync(0xffffffffu, au, 2);
            au += __shfl_xor_sync(0xffffffffu, au, 1);
            if (lane8 == 0) {
                const float gate = bf16_round(ag * fmul);
                const float up = bf16_round(au * fmul);
                const float a = tanhf(gate / 4.f) * 4.f * (1.f / (1.f + expf(-gate))) * (tanhf(up / 25.f) * 25.f);
                const ActT ab = act::from_float(a);
                const int tok = toks[ti];
                uint32_t kmask = uint32_t(mask >> (8 * tok)) & 0xffu;
                while (kmask) {
                    const int k = __ffs(kmask) - 1;
                    kmask &= kmask - 1;
                    out[(static_cast<size_t>(tok) * TOPK + k) * I + pair_global] = ab;
                }
            }
        }
    }
}

__global__ void __launch_bounds__(512)
moe_down_nvfp4_dp4a_m1_kernel(const ActT* __restrict__ act, const int* __restrict__ idx, const float* __restrict__ wts,
                              const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                              const float* __restrict__ wfactor, int M, const float* __restrict__ conv_w,
                              float* __restrict__ conv_state, float* __restrict__ out, float* __restrict__ conv_hist,
                              const uint8_t* const* __restrict__ w_ptrs, const uint8_t* const* __restrict__ ws_ptrs) {
    constexpr int K = I, R = H;
    constexpr int CB = Layout<0>::code_bytes(K), SB = Layout<0>::scale_bytes(K);
    __shared__ int32_t s_q8[TOPK][36][9];
    __shared__ float2 s_xs[TOPK][36];
    __shared__ float y_slot[TOPK][TILE];
    __shared__ int s_idx_m1[TOPK];
    __shared__ float s_wts_m1[TOPK];

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int tile = blockIdx.x;
    const int k_grp = warp >> 1;
    const int r_local = (warp & 1) * 8 + (lane >> 2);
    const int lane4 = lane & 3;
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();

    if (tid < TOPK) {
        s_idx_m1[tid] = ld_dep(idx + tid);
        s_wts_m1[tid] = ld_dep(wts + tid);
    }
    for (int j = tid; j < TOPK * 72; j += 512) {
        const int slot = j / 72;
        const int rem = j - slot * 72;
        const int c = rem >> 1;
        const int half = rem & 1;
        float sc;
        quant16_act_to_i8(act + static_cast<size_t>(slot) * K + c * 32 + half * 16, &s_q8[slot][c][half * 4], sc);
        if (half == 0) s_xs[slot][c].x = sc;
        else s_xs[slot][c].y = sc;
    }
    __syncthreads();

    const int e = s_idx_m1[k_grp];
    const bool local = !w_ptrs || (w_ptrs[e] != nullptr);
    if (local) {
        const size_t local_row = static_cast<size_t>(tile * TILE + r_local);
        const size_t row = static_cast<size_t>(e) * R + local_row;
        const uint8_t* we = w_ptrs ? (w_ptrs[e] + local_row * CB) : (w + row * CB);
        const uint8_t* wse = ws_ptrs ? (ws_ptrs[e] + local_row * SB) : (ws + row * SB);
        Chunk<0> ch[3];
        float acc = 0.f;
#pragma unroll
        for (int sb = 0; sb < 9; sb += 3) {
#pragma unroll
            for (int b = 0; b < 3; ++b) ch[b] = load_chunk<0>(we, wse, 4 * (sb + b) + lane4, wpol);
#pragma unroll
            for (int b = 0; b < 3; ++b) {
                const int c = 4 * (sb + b) + lane4;
                int32_t w0, w1;
                int id0 = 0, id1 = 0;
                e2m1_twice_i8x4(ch[b].q.x, w0, w1);
                id0 = dp4a_s8(w0, s_q8[k_grp][c][0], id0);
                id0 = dp4a_s8(w1, s_q8[k_grp][c][1], id0);
                e2m1_twice_i8x4(ch[b].q.y, w0, w1);
                id0 = dp4a_s8(w0, s_q8[k_grp][c][2], id0);
                id0 = dp4a_s8(w1, s_q8[k_grp][c][3], id0);
                e2m1_twice_i8x4(ch[b].q.z, w0, w1);
                id1 = dp4a_s8(w0, s_q8[k_grp][c][4], id1);
                id1 = dp4a_s8(w1, s_q8[k_grp][c][5], id1);
                e2m1_twice_i8x4(ch[b].q.w, w0, w1);
                id1 = dp4a_s8(w0, s_q8[k_grp][c][6], id1);
                id1 = dp4a_s8(w1, s_q8[k_grp][c][7], id1);
                const uint32_t s2 = e4m3x2_to_half2_m8(ch[b].s);
                const float2 sc = __half22float2(*reinterpret_cast<const __half2*>(&s2));
                const float2 xs = s_xs[k_grp][c];
                acc = fmaf(float(id0), sc.x * xs.x, fmaf(float(id1), sc.y * xs.y, acc));
            }
        }
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        acc += __shfl_xor_sync(0xffffffffu, acc, 2);
        if (lane4 == 0) {
            y_slot[k_grp][r_local] = bf16_round(acc * (__ldg(wfactor + e) * 128.0f));
        }
    } else if (lane4 == 0) {
        y_slot[k_grp][r_local] = 0.f;
    }
    __syncthreads();

    if (tid < TILE) {
        const int ch = tile * TILE + tid;
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            if (!w_ptrs || w_ptrs[s_idx_m1[k]]) s = __fadd_rn(s, __fmul_rn(s_wts_m1[k], y_slot[k][tid]));
        }
        if (!conv_w) {
            out[ch] = s;
            return;
        }
        const float sr = bf16_round(s);
        const float4 cw = *reinterpret_cast<const float4*>(conv_w + 4 * ch);
        const float e0 = ld_dep(conv_state + ch);
        const float e1 = ld_dep(conv_state + H + ch);
        const float e2 = ld_dep(conv_state + 2 * H + ch);
        constexpr float inv_sqrt5 = 1.f / 2.2360679774997896f;
        float acc = __fmul_rn(cw.x, e0);
        acc = __fadd_rn(acc, __fmul_rn(cw.y, e1));
        acc = __fadd_rn(acc, __fmul_rn(cw.z, e2));
        acc = __fadd_rn(acc, __fmul_rn(cw.w, sr));
        out[ch] = __fmul_rn(__fadd_rn(acc, sr), inv_sqrt5);
        if (conv_hist) conv_hist[ch] = sr;
        conv_state[ch] = e1;
        conv_state[H + ch] = e2;
        conv_state[2 * H + ch] = sr;
    }
}

template <int MAX_T, bool HAS_WS>
__global__ void __launch_bounds__(512)
moe_down_nvfp4_dp4a_l1_kernel(const ActT* __restrict__ act_in, const int32_t* __restrict__ q8_in,
                              const float* __restrict__ xs_in, const int* __restrict__ idx, const float* __restrict__ wts,
                              const uint8_t* __restrict__ w, const uint8_t* __restrict__ ws,
                              const float* __restrict__ wfactor, int M, const float* __restrict__ conv_w,
                              float* __restrict__ conv_state, float* __restrict__ out, float* __restrict__ conv_hist,
                              const uint8_t* const* __restrict__ w_ptrs, const uint8_t* const* __restrict__ ws_ptrs) {
    constexpr int K = I, R = H;
    constexpr int CB = Layout<0>::code_bytes(K), SB = Layout<0>::scale_bytes(K);
    __shared__ float y_slot[MAX_T][TOPK][TILE];
    __shared__ int uexp[TOPK * MAX_T];
    __shared__ unsigned long long umask[TOPK * MAX_T];
    __shared__ uint8_t slot_u[TOPK * MAX_T];
    __shared__ uint8_t u_ntok[TOPK * MAX_T];
    __shared__ uint8_t u_tok[TOPK * MAX_T][MAX_T];
    __shared__ uint8_t u_slot[TOPK * MAX_T][MAX_T];
    __shared__ int nuniq;
    __shared__ float sbuf[MAX_T][TILE];

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int tile = blockIdx.x;
    const int k_grp = warp >> 1;
    const int r_local = (warp & 1) * 8 + (lane >> 2);
    const int lane4 = lane & 3;
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();

    const int nslots = M * TOPK;
    if (warp == 0) {
        const int lo = lane < nslots ? ld_dep(idx + lane) : -1;
        const int hi = lane + 32 < nslots ? ld_dep(idx + lane + 32) : -1;
        uint64_t remaining = nslots == 64 ? ~0ull : ((1ull << nslots) - 1);
        int u = 0;
        while (remaining) {
            const int s = __ffsll(static_cast<long long>(remaining)) - 1;
            const int e = __shfl_sync(0xffffffffu, s < 32 ? lo : hi, s & 31);
            const uint64_t mask = slot_mask(e, lo, hi);
            remaining &= ~mask;
            const bool local = !w_ptrs || (w_ptrs[e] != nullptr);
            if (local) {
                if (lane == 0) {
                    uexp[u] = e;
                    umask[u] = mask;
                    int nt = 0;
                    for (int m = 0; m < M; ++m) {
                        const uint32_t km = uint32_t(mask >> (8 * m)) & 0xffu;
                        if (km) {
                            u_tok[u][nt] = uint8_t(m);
                            u_slot[u][nt] = uint8_t(m * TOPK + (__ffs(km) - 1));
                            ++nt;
                        }
                    }
                    u_ntok[u] = uint8_t(nt);
                }
                if ((mask >> lane) & 1) slot_u[lane] = uint8_t(u);
                if ((mask >> (lane + 32)) & 1) slot_u[lane + 32] = uint8_t(u);
                ++u;
            } else {
                if ((mask >> lane) & 1) slot_u[lane] = uint8_t(255);
                if ((mask >> (lane + 32)) & 1) slot_u[lane + 32] = uint8_t(255);
            }
        }
        if (lane == 0) nuniq = u;
    }
    __syncthreads();

    const int U = nuniq;
    for (int u = k_grp; u < U; u += TOPK) {
        const int e = uexp[u];
        const uint64_t mask = umask[u];
        const int ntok = u_ntok[u];
        const size_t local_row = static_cast<size_t>(tile * TILE + r_local);
        const size_t row = static_cast<size_t>(e) * R + local_row;
        const uint8_t* we = w_ptrs ? (w_ptrs[e] + local_row * CB) : (w + row * CB);
        const uint8_t* wse = ws_ptrs ? (ws_ptrs[e] + local_row * SB) : (ws + row * SB);
        const float fmul = __ldg(wfactor + e) * 128.0f;

        if (ntok == 1) {
            const int src_slot = u_slot[u][0];
            Chunk<0> ch[3];
            float acc = 0.f;
#pragma unroll
            for (int sb = 0; sb < 9; sb += 3) {
#pragma unroll
                for (int b = 0; b < 3; ++b) ch[b] = load_chunk<0>(we, wse, 4 * (sb + b) + lane4, wpol);
#pragma unroll
                for (int b = 0; b < 3; ++b) {
                    const int c = 4 * (sb + b) + lane4;
                    int4 xv0, xv1;
                    float2 xs;
                    if constexpr (HAS_WS) {
                        const int32_t* xq_row = q8_in + src_slot * (36 * 8);
                        const float2* xs_row = reinterpret_cast<const float2*>(xs_in + src_slot * 72);
                        xv0 = __ldg(reinterpret_cast<const int4*>(xq_row + c * 8));
                        xv1 = __ldg(reinterpret_cast<const int4*>(xq_row + c * 8 + 4));
                        xs = __ldg(xs_row + c);
                    } else {
                        int32_t q0[4], q1[4];
                        const ActT* src_c = act_in + static_cast<size_t>(src_slot) * I + c * 32;
                        quant16_act_to_i8(src_c, q0, xs.x);
                        quant16_act_to_i8(src_c + 16, q1, xs.y);
                        xv0 = make_int4(q0[0], q0[1], q0[2], q0[3]);
                        xv1 = make_int4(q1[0], q1[1], q1[2], q1[3]);
                    }
                    int32_t w0, w1;
                    int id0 = 0, id1 = 0;
                    e2m1_twice_i8x4(ch[b].q.x, w0, w1);
                    id0 = dp4a_s8(w0, xv0.x, id0);
                    id0 = dp4a_s8(w1, xv0.y, id0);
                    e2m1_twice_i8x4(ch[b].q.y, w0, w1);
                    id0 = dp4a_s8(w0, xv0.z, id0);
                    id0 = dp4a_s8(w1, xv0.w, id0);
                    e2m1_twice_i8x4(ch[b].q.z, w0, w1);
                    id1 = dp4a_s8(w0, xv1.x, id1);
                    id1 = dp4a_s8(w1, xv1.y, id1);
                    e2m1_twice_i8x4(ch[b].q.w, w0, w1);
                    id1 = dp4a_s8(w0, xv1.z, id1);
                    id1 = dp4a_s8(w1, xv1.w, id1);
                    const uint32_t s2 = e4m3x2_to_half2_m8(ch[b].s);
                    const float2 sc = __half22float2(*reinterpret_cast<const __half2*>(&s2));
                    acc = fmaf(float(id0), sc.x * xs.x, fmaf(float(id1), sc.y * xs.y, acc));
                }
            }
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);
            acc += __shfl_xor_sync(0xffffffffu, acc, 2);
            if (lane4 == 0) {
                const float yr = bf16_round(acc * fmul);
                const int tok = u_tok[u][0];
                uint32_t km = uint32_t(mask >> (8 * tok)) & 0xffu;
                while (km) {
                    const int sk = __ffs(km) - 1;
                    km &= km - 1;
                    y_slot[tok][sk][r_local] = yr;
                }
            }
        } else {
            Chunk<0> cur = load_chunk<0>(we, wse, lane4, wpol);
            float acc[MAX_T];
#pragma unroll
            for (int ti = 0; ti < MAX_T; ++ti) acc[ti] = 0.f;
#pragma unroll 1
            for (int step = 0; step < 9; ++step) {
                Chunk<0> nxt;
                if (step + 1 < 9) nxt = load_chunk<0>(we, wse, 4 * (step + 1) + lane4, wpol);
                const int c = 4 * step + lane4;
                int32_t wv[8];
                e2m1_twice_i8x4(cur.q.x, wv[0], wv[1]);
                e2m1_twice_i8x4(cur.q.y, wv[2], wv[3]);
                e2m1_twice_i8x4(cur.q.z, wv[4], wv[5]);
                e2m1_twice_i8x4(cur.q.w, wv[6], wv[7]);
                const uint32_t s2 = e4m3x2_to_half2_m8(cur.s);
                const float2 sc = __half22float2(*reinterpret_cast<const __half2*>(&s2));
                for (int ti = 0; ti < ntok; ++ti) {
                    const int src_slot = u_slot[u][ti];
                    int4 xv0, xv1;
                    float2 xs;
                    if constexpr (HAS_WS) {
                        xv0 = __ldg(reinterpret_cast<const int4*>(q8_in + src_slot * (36 * 8) + c * 8));
                        xv1 = __ldg(reinterpret_cast<const int4*>(q8_in + src_slot * (36 * 8) + c * 8 + 4));
                        xs = __ldg(reinterpret_cast<const float2*>(xs_in + src_slot * 72) + c);
                    } else {
                        int32_t q0[4], q1[4];
                        const ActT* src_c = act_in + static_cast<size_t>(src_slot) * I + c * 32;
                        quant16_act_to_i8(src_c, q0, xs.x);
                        quant16_act_to_i8(src_c + 16, q1, xs.y);
                        xv0 = make_int4(q0[0], q0[1], q0[2], q0[3]);
                        xv1 = make_int4(q1[0], q1[1], q1[2], q1[3]);
                    }
                    int id0 = dp4a_s8(wv[0], xv0.x, dp4a_s8(wv[1], xv0.y, dp4a_s8(wv[2], xv0.z, dp4a_s8(wv[3], xv0.w, 0))));
                    int id1 = dp4a_s8(wv[4], xv1.x, dp4a_s8(wv[5], xv1.y, dp4a_s8(wv[6], xv1.z, dp4a_s8(wv[7], xv1.w, 0))));
                    acc[ti] = fmaf(float(id0), sc.x * xs.x, fmaf(float(id1), sc.y * xs.y, acc[ti]));
                }
                if (step + 1 < 9) cur = nxt;
            }
            for (int ti = 0; ti < ntok; ++ti) {
                float a = acc[ti];
                a += __shfl_xor_sync(0xffffffffu, a, 1);
                a += __shfl_xor_sync(0xffffffffu, a, 2);
                if (lane4 == 0) {
                    const float yr = bf16_round(a * fmul);
                    const int tok = u_tok[u][ti];
                    uint32_t km = uint32_t(mask >> (8 * tok)) & 0xffu;
                    while (km) {
                        const int sk = __ffs(km) - 1;
                        km &= km - 1;
                        y_slot[tok][sk][r_local] = yr;
                    }
                }
            }
        }
    }
    __syncthreads();

    if (tid < TILE * M) {
        const int r = tid & (TILE - 1), n = tid / TILE;
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            const int sl = n * TOPK + k;
            if (slot_u[sl] != 255) s = __fadd_rn(s, __fmul_rn(ld_dep(wts + sl), y_slot[n][k][r]));
        }
        if (!conv_w) {
            out[static_cast<size_t>(n) * H + tile * TILE + r] = s;
            return;
        }
        sbuf[n][r] = bf16_round(s);
    }
    if (!conv_w) return;
    __syncthreads();

    if (tid < TILE) {
        const int ch = tile * TILE + tid;
        const float4 cw = *reinterpret_cast<const float4*>(conv_w + 4 * ch);
        float ext[MAX_T + 3];
        ext[0] = ld_dep(conv_state + ch);
        ext[1] = ld_dep(conv_state + H + ch);
        ext[2] = ld_dep(conv_state + 2 * H + ch);
#pragma unroll
        for (int m = 0; m < MAX_T; ++m) ext[m + 3] = m < M ? sbuf[m][tid] : 0.f;
        constexpr float inv_sqrt5 = 1.f / 2.2360679774997896f;
#pragma unroll
        for (int m = 0; m < MAX_T; ++m) {
            if (m < M) {
                float acc = __fmul_rn(cw.x, ext[m]);
                acc = __fadd_rn(acc, __fmul_rn(cw.y, ext[m + 1]));
                acc = __fadd_rn(acc, __fmul_rn(cw.z, ext[m + 2]));
                acc = __fadd_rn(acc, __fmul_rn(cw.w, ext[m + 3]));
                out[static_cast<size_t>(m) * H + ch] = __fmul_rn(__fadd_rn(acc, ext[m + 3]), inv_sqrt5);
                if (conv_hist) conv_hist[static_cast<size_t>(m) * H + ch] = ext[m + 3];
            }
        }
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int m = 1; m <= MAX_T; ++m)
                if (m == M) conv_state[j * H + ch] = ext[m + j];
    }
}

// ----------------------------------------------------------------------------- EXL3 / Escha W2 decode

// EXL3 moe_up: W13 is stacked [gate(0..1151); up(1152..2303)], codes u8 [128, 144, 144, 32*BITS],
// rin f16 [128, 2304], rout f16 [128, 2304]. Grid: (9, M * TOPK), 18 warps (576 threads).
// Block (blk, slot) owns 128-row gate block `blk` + 128-row up block `9 + blk` for deduplicated expert `e`.
// M1=true specializes M==1: 18 KiB smem (2x occupancy) and a SINGLE __syncthreads() for all 16 output tiles.
template <int BITS, bool M1>
__global__ void __launch_bounds__(UP_WARPS * 32)
moe_up_exl3_kernel(const ActT* __restrict__ x, const int* __restrict__ idx, const uint8_t* __restrict__ w,
                   const __half* __restrict__ rin, const __half* __restrict__ rout, int M,
                   ActT* __restrict__ out, const uint8_t* const* __restrict__ w_ptrs,
                   const uint8_t* const* __restrict__ rin_ptrs, const uint8_t* const* __restrict__ rout_ptrs) {
    constexpr int K = H;
    constexpr int NT = (2 * I) / 16;   // 144
    constexpr int KT = K / 16;         // 144
    constexpr int TILE_BYTES = 32 * BITS;
    constexpr int SM_M = M1 ? 1 : MAXM;
    __shared__ union {
        __half x_rot[UP_WARPS][SM_M][128];   // M1: 4,608 B | M>1: 36,864 B
        float red_m1[UP_WARPS][2][128];      // M1: 18,432 B (single barrier for all 256 gate+up outputs!)
        struct {
            float red[UP_WARPS][2][32][4];   // M>1: 18,432 B (2 tiles in parallel per barrier)
            float y[2][SM_M][128];           // M>1:  8,192 B
        } r;
    } smem;

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, ti = lane & 3;
    const int blk = blockIdx.x, slot = blockIdx.y;
    const int nslots = M * TOPK;
    pdl_wait();
    pdl_trigger();
    const int lo = lane < nslots ? ld_dep(idx + lane) : -1;
    const int hi = lane + 32 < nslots ? ld_dep(idx + lane + 32) : -1;
    const int e = ld_dep(idx + slot);
    const uint64_t mask = slot_mask(e, lo, hi);
    if (__ffsll(static_cast<long long>(mask)) - 1 != slot) return;

    // Phase 1: Input rin * x -> 128-pt Hadamard per warp (each warp owns k = warp*128 .. +127)
    const __half* rin_e = (rin_ptrs ? reinterpret_cast<const __half*>(rin_ptrs[e]) : (rin + size_t(e) * K)) + warp * 128;
    float r_in[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) r_in[i] = __half2float(__ldg(rin_e + lane * 4 + i));
    const int m_lim = M1 ? 1 : M;
    for (int m = 0; m < m_lim; ++m) {
        if (M1 || ((mask >> (8 * m)) & 0xffu)) {
            const ActT* xm = x + size_t(m) * K + warp * 128 + lane * 4;
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint16_t raw;
                asm volatile("ld.global.u16 %0, [%1];" : "=h"(raw) : "l"(xm + i) : "memory");
                v[i] = act::bits_to_float(raw) * r_in[i];
            }
            exl3::fwht_128_warp(v, lane);
            *reinterpret_cast<__half2*>(&smem.x_rot[warp][m][lane * 4]) = __floats2half2_rn(v[0], v[1]);
            *reinterpret_cast<__half2*>(&smem.x_rot[warp][m][lane * 4 + 2]) = __floats2half2_rn(v[2], v[3]);
        }
    }
    __syncwarp();

    // Phase 2: 16 output tiles (8 gate + 8 up) x 8 contiguous k-tiles via mma_f16 (nt_l outer, kt_l inner!)
    const bool g_on = M1 ? (g == 0) : (g < M && ((mask >> (8 * g)) & 0xffu));
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
    uint32_t xb0[8], xb1[8];
#pragma unroll
    for (int kt_l = 0; kt_l < 8; ++kt_l) {
        if (g_on) {
            xb0[kt_l] = *reinterpret_cast<const uint32_t*>(&smem.x_rot[warp][M1 ? 0 : g][kt_l * 16 + 2 * ti]);
            xb1[kt_l] = *reinterpret_cast<const uint32_t*>(&smem.x_rot[warp][M1 ? 0 : g][kt_l * 16 + 2 * ti + 8]);
        } else {
            xb0[kt_l] = 0u;
            xb1[kt_l] = 0u;
        }
    }
    if constexpr (M1) __syncthreads();  // x_rot is now in registers! Reuse smem immediately for red_m1.

    float acc_g[8][4], acc_u[8][4];
    const uint8_t* w_e = w_ptrs ? w_ptrs[e] : (w + size_t(e) * NT * KT * TILE_BYTES);
    const __half* rout_e = rout_ptrs ? reinterpret_cast<const __half*>(rout_ptrs[e]) : (rout + size_t(e) * (2 * I));
    auto run_side = [&](int side, int nt0, float (&acc)[8][4]) {
#pragma unroll
        for (int nt_l = 0; nt_l < 8; ++nt_l) {
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            const uint8_t* w_row = w_e + (size_t(nt0 + nt_l) * KT + warp * 8) * TILE_BYTES;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
            // Pascal sm_60/sm_61 EXL3 fast path: direct float2 MCG decode + active-token-only dot products
            // (12.8x fewer shuffles and up to 8x fewer FMAs vs 8-column SIMT MMA emulation).
            if constexpr (M1) {
                float pa = 0.f, pb = 0.f;
#pragma unroll
                for (int kt_l = 0; kt_l < 8; ++kt_l) {
                    const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_row + kt_l * TILE_BYTES);
                    float2 f01, f23, f45, f67;
                    exl3::dq8_mcg_f2<BITS>([&](int wi) { return ld_stream_u1(tw + wi, wpol); }, lane, f01, f23, f45, f67);
                    const uint32_t ux0 = __shfl_sync(0xffffffffu, xb0[kt_l], ti);
                    const uint32_t ux1 = __shfl_sync(0xffffffffu, xb1[kt_l], ti);
                    const float2 fx0 = __half22float2(*reinterpret_cast<const __half2*>(&ux0));
                    const float2 fx1 = __half22float2(*reinterpret_cast<const __half2*>(&ux1));
                    pa = fmaf(f01.x, fx0.x, fmaf(f01.y, fx0.y, fmaf(f23.x, fx1.x, fmaf(f23.y, fx1.y, pa))));
                    pb = fmaf(f45.x, fx0.x, fmaf(f45.y, fx0.y, fmaf(f67.x, fx1.x, fmaf(f67.y, fx1.y, pb))));
                }
                pa += __shfl_xor_sync(0xffffffffu, pa, 1);
                pa += __shfl_xor_sync(0xffffffffu, pa, 2);
                pb += __shfl_xor_sync(0xffffffffu, pb, 1);
                pb += __shfl_xor_sync(0xffffffffu, pb, 2);
                d[0] = pa;
                d[2] = pb;
            } else {
                float pa[MAXM], pb[MAXM];
#pragma unroll
                for (int m = 0; m < MAXM; ++m) pa[m] = pb[m] = 0.f;
#pragma unroll
                for (int kt_l = 0; kt_l < 8; ++kt_l) {
                    const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_row + kt_l * TILE_BYTES);
                    float2 f01, f23, f45, f67;
                    exl3::dq8_mcg_f2<BITS>([&](int wi) { return ld_stream_u1(tw + wi, wpol); }, lane, f01, f23, f45, f67);
                    for (int m = 0; m < M; ++m) {
                        if (((mask >> (8 * m)) & 0xffu) == 0u) continue;
                        const int src_lane = 4 * m + ti;
                        const uint32_t ux0 = __shfl_sync(0xffffffffu, xb0[kt_l], src_lane);
                        const uint32_t ux1 = __shfl_sync(0xffffffffu, xb1[kt_l], src_lane);
                        const float2 fx0 = __half22float2(*reinterpret_cast<const __half2*>(&ux0));
                        const float2 fx1 = __half22float2(*reinterpret_cast<const __half2*>(&ux1));
                        pa[m] = fmaf(f01.x, fx0.x, fmaf(f01.y, fx0.y, fmaf(f23.x, fx1.x, fmaf(f23.y, fx1.y, pa[m]))));
                        pb[m] = fmaf(f45.x, fx0.x, fmaf(f45.y, fx0.y, fmaf(f67.x, fx1.x, fmaf(f67.y, fx1.y, pb[m]))));
                    }
                }
                for (int m = 0; m < M; ++m) {
                    if (((mask >> (8 * m)) & 0xffu) == 0u) continue;
                    pa[m] += __shfl_xor_sync(0xffffffffu, pa[m], 1);
                    pa[m] += __shfl_xor_sync(0xffffffffu, pa[m], 2);
                    pb[m] += __shfl_xor_sync(0xffffffffu, pb[m], 1);
                    pb[m] += __shfl_xor_sync(0xffffffffu, pb[m], 2);
                    if (m == 2 * ti) { d[0] = pa[m]; d[2] = pb[m]; }
                    else if (m == 2 * ti + 1) { d[1] = pa[m]; d[3] = pb[m]; }
                }
            }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
            float acc8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int kt_l = 0; kt_l < 8; ++kt_l) {
                const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_row + kt_l * TILE_BYTES);
                uint32_t h01, h23, h45, h67;
                exl3::dq8_mcg<BITS>([&](int wi) { return ld_stream_u1(tw + wi, wpol); }, lane, h01, h23, h45, h67);
                mma::sm70_mma_f16_acc(acc8, h01, h45, h23, h67, xb0[kt_l], xb1[kt_l]);
            }
            mma::sm70_reduce_acc8_to_d4(acc8, d);
#else
#pragma unroll
            for (int kt_l = 0; kt_l < 8; ++kt_l) {
                const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_row + kt_l * TILE_BYTES);
                uint32_t h01, h23, h45, h67;
                exl3::dq8_mcg<BITS>([&](int wi) { return ld_stream_u1(tw + wi, wpol); }, lane, h01, h23, h45, h67);
                mma_f16(d, h01, h45, h23, h67, xb0[kt_l], xb1[kt_l]);
            }
#endif
            if constexpr (M1) {
                if (ti == 0) {
                    smem.red_m1[warp][side][nt_l * 16 + g] = d[0];
                    smem.red_m1[warp][side][nt_l * 16 + g + 8] = d[2];
                }
            } else {
#pragma unroll
                for (int v = 0; v < 4; ++v) acc[nt_l][v] = d[v];
            }
        }
    };
    run_side(0, blk * 8, acc_g);
    run_side(1, 72 + blk * 8, acc_u);
    __syncthreads();

    if constexpr (M1) {
        // Single barrier total! Warp 0 sums all 18 warps in fixed order -> FWHT -> rout -> SwiGLU -> out!
        if (warp == 0) {
            float vg[4] = {0.f, 0.f, 0.f, 0.f}, vu[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ww = 0; ww < UP_WARPS; ++ww) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    vg[i] += smem.red_m1[ww][0][lane * 4 + i];
                    vu[i] += smem.red_m1[ww][1][lane * 4 + i];
                }
            }
            exl3::fwht_128_warp(vg, lane);
            exl3::fwht_128_warp(vu, lane);
            const __half* rg = rout_e + blk * 128 + lane * 4;
            const __half* ru = rout_e + I + blk * 128 + lane * 4;
            ActT ab[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float gate = bf16_round(vg[i] * __half2float(__ldg(rg + i)));
                const float up = bf16_round(vu[i] * __half2float(__ldg(ru + i)));
                const float a = tanhf(gate / 4.f) * 4.f * (1.f / (1.f + expf(-gate))) * (tanhf(up / 25.f) * 25.f);
                ab[i] = act::from_float(a);
            }
            uint32_t kmask = uint32_t(mask) & 0xffu;
            while (kmask) {
                const int k = __ffs(kmask) - 1;
                kmask &= kmask - 1;
                ActT* dst = out + size_t(k) * I + blk * 128 + lane * 4;
#pragma unroll
                for (int i = 0; i < 4; ++i) dst[i] = ab[i];
            }
        }
    } else {
        // M > 1: reduce 2 tiles per barrier using 2 warps in parallel (8 barriers total instead of 32)
        for (int s = 0; s < 2; ++s) {
            for (int nt_pair = 0; nt_pair < 8; nt_pair += 2) {
#pragma unroll
                for (int p = 0; p < 2; ++p) {
                    const float (&a)[4] = (s == 0 ? acc_g[nt_pair + p] : acc_u[nt_pair + p]);
#pragma unroll
                    for (int v = 0; v < 4; ++v) smem.r.red[warp][p][lane][v] = a[v];
                }
                __syncthreads();
                if (warp < 2) {
                    const int nt_l = nt_pair + warp;
                    float sum[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                    for (int ww = 0; ww < UP_WARPS; ++ww)
#pragma unroll
                        for (int v = 0; v < 4; ++v) sum[v] += smem.r.red[ww][warp][lane][v];
                    const int m0 = 2 * ti, m1 = 2 * ti + 1;
                    if (m0 < M) {
                        smem.r.y[s][m0][nt_l * 16 + g] = sum[0];
                        smem.r.y[s][m0][nt_l * 16 + g + 8] = sum[2];
                    }
                    if (m1 < M) {
                        smem.r.y[s][m1][nt_l * 16 + g] = sum[1];
                        smem.r.y[s][m1][nt_l * 16 + g + 8] = sum[3];
                    }
                }
                __syncthreads();
            }
        }

        if (warp < M) {
            uint32_t kmask = uint32_t(mask >> (8 * warp)) & 0xffu;
            if (kmask) {
                float vg[4], vu[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    vg[i] = smem.r.y[0][warp][lane * 4 + i];
                    vu[i] = smem.r.y[1][warp][lane * 4 + i];
                }
                exl3::fwht_128_warp(vg, lane);
                exl3::fwht_128_warp(vu, lane);
                const __half* rg = rout_e + blk * 128 + lane * 4;
                const __half* ru = rout_e + I + blk * 128 + lane * 4;
                ActT ab[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float gate = bf16_round(vg[i] * __half2float(__ldg(rg + i)));
                    const float up = bf16_round(vu[i] * __half2float(__ldg(ru + i)));
                    const float a = tanhf(gate / 4.f) * 4.f * (1.f / (1.f + expf(-gate))) * (tanhf(up / 25.f) * 25.f);
                    ab[i] = act::from_float(a);
                }
                while (kmask) {
                    const int k = __ffs(kmask) - 1;
                    kmask &= kmask - 1;
                    ActT* dst = out + (size_t(warp) * TOPK + k) * I + blk * 128 + lane * 4;
#pragma unroll
                    for (int i = 0; i < 4; ++i) dst[i] = ab[i];
                }
            }
        }
    }
}

// EXL3 moe_down Stage 1: W2 codes u8 [128, 144, 72, 32*BITS], rin f16 [128, 1152], rout f16 [128, 2304].
// Grid: (18, M * TOPK) -> 144 active blocks at M=1! 9 warps (288 threads) cover K = 1152.
// M1=true specializes M==1: 4.6 KiB smem (max occupancy) and a SINGLE __syncthreads() across all 8 tiles!
template <int BITS, bool M1>
__global__ void __launch_bounds__(DOWN_GROUPS * 32)
moe_down_exl3_kernel(const ActT* __restrict__ act, const int* __restrict__ idx, const uint8_t* __restrict__ w,
                     const __half* __restrict__ rin, const __half* __restrict__ rout, int M,
                     ActT* __restrict__ ws, const uint8_t* const* __restrict__ w_ptrs,
                     const uint8_t* const* __restrict__ rin_ptrs, const uint8_t* const* __restrict__ rout_ptrs) {
    constexpr int K = I;
    constexpr int NT = H / 16;    // 144
    constexpr int KT = K / 16;    // 72
    constexpr int TILE_BYTES = 32 * BITS;
    constexpr int SM_M = M1 ? 1 : MAXM;
    __shared__ union {
        __half x_rot[DOWN_GROUPS][SM_M][128];  // M1: 2,304 B | M>1: 18,432 B
        float red_m1[DOWN_GROUPS][128];        // M1: 4,608 B
        struct {
            float red[DOWN_GROUPS][2][32][4];  // M>1: 9,216 B
            float y[SM_M][128];                // M>1: 4,096 B
        } r;
    } smem;

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, ti = lane & 3;
    const int blk = blockIdx.x, slot = blockIdx.y;
    const int nslots = M * TOPK;
    const int lo = lane < nslots ? ld_dep(idx + lane) : -1;
    const int hi = lane + 32 < nslots ? ld_dep(idx + lane + 32) : -1;
    const int e = ld_dep(idx + slot);
    const uint64_t mask = slot_mask(e, lo, hi);
    if (__ffsll(static_cast<long long>(mask)) - 1 != slot) return;

    const __half* rin_e = (rin_ptrs ? reinterpret_cast<const __half*>(rin_ptrs[e]) : (rin + size_t(e) * K)) + warp * 128;
    float r_in[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) r_in[i] = __half2float(__ldg(rin_e + lane * 4 + i));

    pdl_wait();

    const int m_lim = M1 ? 1 : M;
    for (int m = 0; m < m_lim; ++m) {
        const uint32_t km = uint32_t(mask >> (8 * m)) & 0xffu;
        if (M1 || km) {
            const int k0 = M1 ? slot : (__ffs(km) - 1);
            const ActT* am = act + (size_t(m) * TOPK + k0) * K + warp * 128 + lane * 4;
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint16_t raw;
                asm volatile("ld.global.u16 %0, [%1];" : "=h"(raw) : "l"(am + i) : "memory");
                v[i] = act::bits_to_float(raw) * r_in[i];
            }
            exl3::fwht_128_warp(v, lane);
            *reinterpret_cast<__half2*>(&smem.x_rot[warp][m][lane * 4]) = __floats2half2_rn(v[0], v[1]);
            *reinterpret_cast<__half2*>(&smem.x_rot[warp][m][lane * 4 + 2]) = __floats2half2_rn(v[2], v[3]);
        }
    }
    __syncwarp();

    const bool g_on = M1 ? (g == 0) : (g < M && ((mask >> (8 * g)) & 0xffu));
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
    uint32_t xb0[8], xb1[8];
#pragma unroll
    for (int kt_l = 0; kt_l < 8; ++kt_l) {
        if (g_on) {
            xb0[kt_l] = *reinterpret_cast<const uint32_t*>(&smem.x_rot[warp][M1 ? 0 : g][kt_l * 16 + 2 * ti]);
            xb1[kt_l] = *reinterpret_cast<const uint32_t*>(&smem.x_rot[warp][M1 ? 0 : g][kt_l * 16 + 2 * ti + 8]);
        } else {
            xb0[kt_l] = 0u;
            xb1[kt_l] = 0u;
        }
    }
    if constexpr (M1) __syncthreads();  // x_rot is now in registers!

    float acc[8][4];
    const uint8_t* w_e = w_ptrs ? w_ptrs[e] : (w + size_t(e) * NT * KT * TILE_BYTES);
    const __half* rout_e = rout_ptrs ? reinterpret_cast<const __half*>(rout_ptrs[e]) : (rout + size_t(e) * H);
    const int nt0 = blk * 8;
#pragma unroll
    for (int nt_l = 0; nt_l < 8; ++nt_l) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        const uint8_t* w_row = w_e + (size_t(nt0 + nt_l) * KT + warp * 8) * TILE_BYTES;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700 || defined(INGINE_FORCE_SIMT_MMA))
        if constexpr (M1) {
            float pa = 0.f, pb = 0.f;
#pragma unroll
            for (int kt_l = 0; kt_l < 8; ++kt_l) {
                const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_row + kt_l * TILE_BYTES);
                float2 f01, f23, f45, f67;
                exl3::dq8_mcg_f2<BITS>([&](int wi) { return ld_stream_u1(tw + wi, wpol); }, lane, f01, f23, f45, f67);
                const uint32_t ux0 = __shfl_sync(0xffffffffu, xb0[kt_l], ti);
                const uint32_t ux1 = __shfl_sync(0xffffffffu, xb1[kt_l], ti);
                const float2 fx0 = __half22float2(*reinterpret_cast<const __half2*>(&ux0));
                const float2 fx1 = __half22float2(*reinterpret_cast<const __half2*>(&ux1));
                pa = fmaf(f01.x, fx0.x, fmaf(f01.y, fx0.y, fmaf(f23.x, fx1.x, fmaf(f23.y, fx1.y, pa))));
                pb = fmaf(f45.x, fx0.x, fmaf(f45.y, fx0.y, fmaf(f67.x, fx1.x, fmaf(f67.y, fx1.y, pb))));
            }
            pa += __shfl_xor_sync(0xffffffffu, pa, 1);
            pa += __shfl_xor_sync(0xffffffffu, pa, 2);
            pb += __shfl_xor_sync(0xffffffffu, pb, 1);
            pb += __shfl_xor_sync(0xffffffffu, pb, 2);
            d[0] = pa;
            d[2] = pb;
        } else {
            float pa[MAXM], pb[MAXM];
#pragma unroll
            for (int m = 0; m < MAXM; ++m) pa[m] = pb[m] = 0.f;
#pragma unroll
            for (int kt_l = 0; kt_l < 8; ++kt_l) {
                const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_row + kt_l * TILE_BYTES);
                float2 f01, f23, f45, f67;
                exl3::dq8_mcg_f2<BITS>([&](int wi) { return ld_stream_u1(tw + wi, wpol); }, lane, f01, f23, f45, f67);
                for (int m = 0; m < M; ++m) {
                    if (((mask >> (8 * m)) & 0xffu) == 0u) continue;
                    const int src_lane = 4 * m + ti;
                    const uint32_t ux0 = __shfl_sync(0xffffffffu, xb0[kt_l], src_lane);
                    const uint32_t ux1 = __shfl_sync(0xffffffffu, xb1[kt_l], src_lane);
                    const float2 fx0 = __half22float2(*reinterpret_cast<const __half2*>(&ux0));
                    const float2 fx1 = __half22float2(*reinterpret_cast<const __half2*>(&ux1));
                    pa[m] = fmaf(f01.x, fx0.x, fmaf(f01.y, fx0.y, fmaf(f23.x, fx1.x, fmaf(f23.y, fx1.y, pa[m]))));
                    pb[m] = fmaf(f45.x, fx0.x, fmaf(f45.y, fx0.y, fmaf(f67.x, fx1.x, fmaf(f67.y, fx1.y, pb[m]))));
                }
            }
            for (int m = 0; m < M; ++m) {
                if (((mask >> (8 * m)) & 0xffu) == 0u) continue;
                pa[m] += __shfl_xor_sync(0xffffffffu, pa[m], 1);
                pa[m] += __shfl_xor_sync(0xffffffffu, pa[m], 2);
                pb[m] += __shfl_xor_sync(0xffffffffu, pb[m], 1);
                pb[m] += __shfl_xor_sync(0xffffffffu, pb[m], 2);
                if (m == 2 * ti) { d[0] = pa[m]; d[2] = pb[m]; }
                else if (m == 2 * ti + 1) { d[1] = pa[m]; d[3] = pb[m]; }
            }
        }
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 700 && !defined(INGINE_FORCE_SIMT_MMA)
        float acc8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kt_l = 0; kt_l < 8; ++kt_l) {
            const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_row + kt_l * TILE_BYTES);
            uint32_t h01, h23, h45, h67;
            exl3::dq8_mcg<BITS>([&](int wi) { return ld_stream_u1(tw + wi, wpol); }, lane, h01, h23, h45, h67);
            mma::sm70_mma_f16_acc(acc8, h01, h45, h23, h67, xb0[kt_l], xb1[kt_l]);
        }
        mma::sm70_reduce_acc8_to_d4(acc8, d);
#else
#pragma unroll
        for (int kt_l = 0; kt_l < 8; ++kt_l) {
            const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_row + kt_l * TILE_BYTES);
            uint32_t h01, h23, h45, h67;
            exl3::dq8_mcg<BITS>([&](int wi) { return ld_stream_u1(tw + wi, wpol); }, lane, h01, h23, h45, h67);
            mma_f16(d, h01, h45, h23, h67, xb0[kt_l], xb1[kt_l]);
        }
#endif
        if constexpr (M1) {
            if (ti == 0) {
                smem.red_m1[warp][nt_l * 16 + g] = d[0];
                smem.red_m1[warp][nt_l * 16 + g + 8] = d[2];
            }
        } else {
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[nt_l][v] = d[v];
        }
    }
    __syncthreads();
    pdl_trigger();  // Stage 2 combine+conv may start its prologue

    if constexpr (M1) {
        if (warp == 0) {
            float v[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ww = 0; ww < DOWN_GROUPS; ++ww)
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] += smem.red_m1[ww][lane * 4 + i];
            exl3::fwht_128_warp(v, lane);
            const __half* ro = rout_e + blk * 128 + lane * 4;
            ActT yb[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) yb[i] = act::from_float(v[i] * __half2float(__ldg(ro + i)));
            uint32_t kmask = uint32_t(mask) & 0xffu;
            while (kmask) {
                const int k = __ffs(kmask) - 1;
                kmask &= kmask - 1;
                ActT* dst = ws + size_t(k) * H + blk * 128 + lane * 4;
#pragma unroll
                for (int i = 0; i < 4; ++i) dst[i] = yb[i];
            }
        }
    } else {
        for (int nt_pair = 0; nt_pair < 8; nt_pair += 2) {
#pragma unroll
            for (int p = 0; p < 2; ++p)
#pragma unroll
                for (int v = 0; v < 4; ++v) smem.r.red[warp][p][lane][v] = acc[nt_pair + p][v];
            __syncthreads();
            if (warp < 2) {
                const int nt_l = nt_pair + warp;
                float sum[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
                for (int ww = 0; ww < DOWN_GROUPS; ++ww)
#pragma unroll
                    for (int v = 0; v < 4; ++v) sum[v] += smem.r.red[ww][warp][lane][v];
                const int m0 = 2 * ti, m1 = 2 * ti + 1;
                if (m0 < M) {
                    smem.r.y[m0][nt_l * 16 + g] = sum[0];
                    smem.r.y[m0][nt_l * 16 + g + 8] = sum[2];
                }
                if (m1 < M) {
                    smem.r.y[m1][nt_l * 16 + g] = sum[1];
                    smem.r.y[m1][nt_l * 16 + g + 8] = sum[3];
                }
            }
            __syncthreads();
        }

        if (warp < M) {
            uint32_t kmask = uint32_t(mask >> (8 * warp)) & 0xffu;
            if (kmask) {
                float v[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) v[i] = smem.r.y[warp][lane * 4 + i];
                exl3::fwht_128_warp(v, lane);
                const __half* ro = rout_e + blk * 128 + lane * 4;
                ActT yb[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) yb[i] = act::from_float(v[i] * __half2float(__ldg(ro + i)));
                while (kmask) {
                    const int k = __ffs(kmask) - 1;
                    kmask &= kmask - 1;
                    ActT* dst = ws + (size_t(warp) * TOPK + k) * H + blk * 128 + lane * 4;
#pragma unroll
                    for (int i = 0; i < 4; ++i) dst[i] = yb[i];
                }
            }
        }
    }
}

// EXL3 moe_down Stage 2: slot-ordered weighted sum + post-MLP conv + state update (PDL-overlapped).
__global__ void __launch_bounds__(128)
moe_down_combine_conv_kernel(const ActT* __restrict__ ws, const float* __restrict__ wts, int M,
                             const float* __restrict__ conv_w, float* __restrict__ conv_state, float* __restrict__ out,
                             float* __restrict__ conv_hist) {
    const int ch = blockIdx.x * 128 + threadIdx.x;
    const float4 cw = *reinterpret_cast<const float4*>(conv_w + 4 * ch);
    pdl_wait();
    float ext[MAXM + 3];
    ext[0] = ld_dep(conv_state + ch);
    ext[1] = ld_dep(conv_state + H + ch);
    ext[2] = ld_dep(conv_state + 2 * H + ch);
    for (int m = 0; m < MAXM; ++m) {
        if (m < M) {
            float s = 0.f;
#pragma unroll
            for (int k = 0; k < TOPK; ++k) {
                const int sl = m * TOPK + k;
                uint16_t raw;
                asm volatile("ld.global.u16 %0, [%1];" : "=h"(raw) : "l"(ws + size_t(sl) * H + ch) : "memory");
                s = __fadd_rn(s, __fmul_rn(ld_dep(wts + sl), act::bits_to_float(raw)));
            }
            ext[m + 3] = bf16_round(s);
        } else {
            ext[m + 3] = 0.f;
        }
    }
    pdl_trigger();
    constexpr float inv_sqrt5 = 1.f / 2.2360679774997896f;
#pragma unroll
    for (int m = 0; m < MAXM; ++m) {
        if (m < M) {
            float acc = __fmul_rn(cw.x, ext[m]);
            acc = __fadd_rn(acc, __fmul_rn(cw.y, ext[m + 1]));
            acc = __fadd_rn(acc, __fmul_rn(cw.z, ext[m + 2]));
            acc = __fadd_rn(acc, __fmul_rn(cw.w, ext[m + 3]));
            out[size_t(m) * H + ch] = __fmul_rn(__fadd_rn(acc, ext[m + 3]), inv_sqrt5);
            if (conv_hist) conv_hist[size_t(m) * H + ch] = ext[m + 3];
        }
    }
#pragma unroll
    for (int j = 0; j < 3; ++j)
#pragma unroll
        for (int m = 1; m <= MAXM; ++m)
            if (m == M) conv_state[j * H + ch] = ext[m + j];
}

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("moe: ") + what);
}

// Launch on `stream`; with pdl the kernel may start before its predecessor finishes
// (programmatic dependent launch, captured as a programmatic edge in CUDA graphs).
template <typename... Params, typename... Args>
void launch(void (*kernel)(Params...), dim3 grid, int threads, bool pdl, cudaStream_t stream, Args... args) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = dim3(threads);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = (pdl && current_sm_major() >= 9) ? 1 : 0;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, static_cast<Params>(args)...));
}

bool use_nvfp4_dp4a() {
    static const bool disabled = [] {
        const char* v = std::getenv("INGINE_DISABLE_NVFP4_DP4A");
        return v && v[0] != '\0' && v[0] != '0';
    }();
    return !disabled && current_sm_version() >= 61;
}

template <typename KernelT>
void set_max_shared_once(KernelT kernel, std::atomic<uint32_t>& done_mask) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return;
    const uint32_t bit = 1u << (dev & 31);
    if (!(done_mask.load(std::memory_order_acquire) & bit)) {
        (void)cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                   cudaSharedmemCarveoutMaxShared);
        done_mask.fetch_or(bit, std::memory_order_release);
    }
}

}  // namespace

void moe_up(const ActT* x, int M, const int* idx, const void* w13, const void* w13_scale,
            const float* w13_factor, Format fmt, ActT* out, bool pdl, cudaStream_t stream,
            const void* w13_rout, const ExpertPtrs* ptrs, void* workspace) {
    require(M >= 1 && M <= MAXM, "M must be in 1..8");
    const uint8_t* const* wp = ptrs ? ptrs->w13 : nullptr;
    const uint8_t* const* wsp = ptrs ? ptrs->w13_scale : nullptr;
    const uint8_t* const* wroutp = ptrs ? ptrs->w13_rout : nullptr;
    require(x && idx && ((w13 && w13_scale) || (wp && wsp)) && out, "moe_up: null argument");
    if (fmt == Format::EXL3_2 || fmt == Format::EXL3_3) {
        require(w13_rout != nullptr || wroutp != nullptr, "moe_up(EXL3): w13_rout required");
        const dim3 grid(I / 128, M * TOPK);  // 9 x (M*8)
        const auto* w = static_cast<const uint8_t*>(w13);
        const auto* rin = static_cast<const __half*>(w13_scale);
        const auto* rout = static_cast<const __half*>(w13_rout);
        if (fmt == Format::EXL3_2) {
            if (M == 1) launch(moe_up_exl3_kernel<2, true>, grid, UP_WARPS * 32, pdl, stream, x, idx, w, rin, rout, M, out, wp, wsp, wroutp);
            else launch(moe_up_exl3_kernel<2, false>, grid, UP_WARPS * 32, pdl, stream, x, idx, w, rin, rout, M, out, wp, wsp, wroutp);
        } else {
            if (M == 1) launch(moe_up_exl3_kernel<3, true>, grid, UP_WARPS * 32, pdl, stream, x, idx, w, rin, rout, M, out, wp, wsp, wroutp);
            else launch(moe_up_exl3_kernel<3, false>, grid, UP_WARPS * 32, pdl, stream, x, idx, w, rin, rout, M, out, wp, wsp, wroutp);
        }
        return;
    }
    require(fmt == Format::MXFP8 || w13_factor, "scaled-block experts need a factor tensor");
    const dim3 grid(2 * I / TILE, M * TOPK);
    const auto* w = static_cast<const uint8_t*>(w13);
    const auto* ws = static_cast<const uint8_t*>(w13_scale);
    const float* wf = fmt == Format::MXFP8 ? nullptr : w13_factor;
    if constexpr (kActIsFP16) {
        if ((fmt == Format::NVFP4 || fmt == Format::MXFP8) && current_sm_major() < 8) {
            const dim3 sgrid(I / 32, M * TOPK);  // 36 x (M*8), 256 threads (32 gate/up pairs/block)
            if (fmt == Format::NVFP4) {
                if (use_nvfp4_dp4a()) {
                    if (M == 1) {
                        launch(moe_up_nvfp4_dp4a_m1_kernel, sgrid, 256, pdl, stream, x, idx, w, ws, wf, M, out, wp, wsp);
                    } else if (workspace != nullptr) {
                        require((reinterpret_cast<std::uintptr_t>(workspace) & 15u) == 0,
                                "moe_up: workspace must be 16-byte aligned");
                        int32_t* q8_ptr = static_cast<int32_t*>(workspace);
                        float* xs_ptr = reinterpret_cast<float*>(q8_ptr + kDp4aUpQ8Elems);
                        moe_quant_i8_kernel<72><<<M, 128, 0, stream>>>(x, q8_ptr, xs_ptr);
                        CUDA_CHECK(cudaGetLastError());
                        if (M == 2) {
                            launch(moe_up_nvfp4_dp4a_l1_kernel<2, true>, sgrid, 256, pdl, stream, nullptr, q8_ptr, xs_ptr, idx, w, ws, wf, M, out, wp, wsp);
                        } else if (M <= 4) {
                            launch(moe_up_nvfp4_dp4a_l1_kernel<4, true>, sgrid, 256, pdl, stream, nullptr, q8_ptr, xs_ptr, idx, w, ws, wf, M, out, wp, wsp);
                        } else {
                            launch(moe_up_nvfp4_dp4a_l1_kernel<8, true>, sgrid, 256, pdl, stream, nullptr, q8_ptr, xs_ptr, idx, w, ws, wf, M, out, wp, wsp);
                        }
                    } else {
                        // Scratch-free M > 1 path when caller provides no workspace (bit-identical numerics).
                        if (M == 2) {
                            launch(moe_up_nvfp4_dp4a_l1_kernel<2, false>, sgrid, 256, pdl, stream, x, nullptr, nullptr, idx, w, ws, wf, M, out, wp, wsp);
                        } else if (M <= 4) {
                            launch(moe_up_nvfp4_dp4a_l1_kernel<4, false>, sgrid, 256, pdl, stream, x, nullptr, nullptr, idx, w, ws, wf, M, out, wp, wsp);
                        } else {
                            launch(moe_up_nvfp4_dp4a_l1_kernel<8, false>, sgrid, 256, pdl, stream, x, nullptr, nullptr, idx, w, ws, wf, M, out, wp, wsp);
                        }
                    }
                } else {
                    // Pascal sm_60 (P100) lacks hardware __dp4a: use exact FP32 e2m1_twice8 streaming kernel.
                    if (M == 1) launch(moe_up_fp16_stream_kernel<0, true>, sgrid, 256, pdl, stream, x, idx, w, ws, wf, M, out, wp, wsp);
                    else launch(moe_up_fp16_stream_kernel<0, false>, sgrid, 256, pdl, stream, x, idx, w, ws, wf, M, out, wp, wsp);
                }
                return;
            } else {
                // MXFP8: M=1 uses streaming fp16_stream_kernel; M>=2 uses fast HMMA.884 decode_sm70_fast.
                if (M == 1) {
                    launch(moe_up_fp16_stream_kernel<1, true>, sgrid, 256, pdl, stream, x, idx, w, ws, wf, M, out, wp, wsp);
                    return;
                }
            }
        }
    }
    if (fmt == Format::NVFP4) launch(moe_up_kernel<0>, grid, UP_WARPS * 32, pdl, stream, x, idx, w, ws, wf, M, out, wp, wsp);
    else if (fmt == Format::MXFP8) launch(moe_up_kernel<1>, grid, UP_WARPS * 32, pdl, stream, x, idx, w, ws, wf, M, out, wp, wsp);
    else if (fmt == Format::ATLAS2) launch(moe_up_kernel<2>, grid, UP_WARPS * 32, pdl, stream, x, idx, w, ws, wf, M, out, wp, wsp);
    else launch(moe_up_kernel<3>, grid, UP_WARPS * 32, pdl, stream, x, idx, w, ws, wf, M, out, wp, wsp);
}

void moe_down(const ActT* act, int M, const int* idx, const float* wts, const void* w2,
              const void* w2_scale, const float* w2_factor, Format fmt, const float* conv_w, float* conv_state,
              float* out, float* conv_hist, bool pdl, cudaStream_t stream,
              const void* w2_rout, ActT* workspace, const ExpertPtrs* ptrs) {
    require(M >= 1 && M <= MAXM, "M must be in 1..8");
    const uint8_t* const* wp = ptrs ? ptrs->w2 : nullptr;
    const uint8_t* const* wsp = ptrs ? ptrs->w2_scale : nullptr;
    const uint8_t* const* wroutp = ptrs ? ptrs->w2_rout : nullptr;
    require(act && idx && wts && ((w2 && w2_scale) || (wp && wsp)) && ((conv_w && conv_state) || wp) && out, "moe_down: null argument");
    if (fmt == Format::EXL3_2 || fmt == Format::EXL3_3) {
        require((w2_rout != nullptr || wroutp != nullptr) && workspace, "moe_down(EXL3): w2_rout and workspace required");
        const dim3 grid(H / 128, M * TOPK);  // 18 x (M*8)
        const auto* w = static_cast<const uint8_t*>(w2);
        const auto* rin = static_cast<const __half*>(w2_scale);
        const auto* rout = static_cast<const __half*>(w2_rout);
        if (fmt == Format::EXL3_2) {
            if (M == 1) launch(moe_down_exl3_kernel<2, true>, grid, DOWN_GROUPS * 32, pdl, stream, act, idx, w, rin, rout, M, workspace, wp, wsp, wroutp);
            else launch(moe_down_exl3_kernel<2, false>, grid, DOWN_GROUPS * 32, pdl, stream, act, idx, w, rin, rout, M, workspace, wp, wsp, wroutp);
        } else {
            if (M == 1) launch(moe_down_exl3_kernel<3, true>, grid, DOWN_GROUPS * 32, pdl, stream, act, idx, w, rin, rout, M, workspace, wp, wsp, wroutp);
            else launch(moe_down_exl3_kernel<3, false>, grid, DOWN_GROUPS * 32, pdl, stream, act, idx, w, rin, rout, M, workspace, wp, wsp, wroutp);
        }
        launch(moe_down_combine_conv_kernel, dim3(H / 128), 128, pdl, stream, workspace, wts, M, conv_w, conv_state,
               out, conv_hist);
        return;
    }
    require(fmt == Format::MXFP8 || w2_factor, "scaled-block experts need a factor tensor");
    const auto* w = static_cast<const uint8_t*>(w2);
    const auto* ws = static_cast<const uint8_t*>(w2_scale);
    const float* wf = fmt == Format::MXFP8 ? nullptr : w2_factor;
    if constexpr (kActIsFP16) {
        if ((fmt == Format::NVFP4 || fmt == Format::MXFP8) && current_sm_major() < 8) {
            if (fmt == Format::NVFP4) {
                if (use_nvfp4_dp4a()) {
                    if (M == 1) {
                        static std::atomic<uint32_t> dn1_attr{0};
                        set_max_shared_once(moe_down_nvfp4_dp4a_m1_kernel, dn1_attr);
                        launch(moe_down_nvfp4_dp4a_m1_kernel, dim3(H / TILE), 512, pdl, stream, act, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
                    } else if (workspace != nullptr) {
                        require((reinterpret_cast<std::uintptr_t>(workspace) & 15u) == 0,
                                "moe_down: workspace must be 16-byte aligned");
                        int32_t* q8_ptr = reinterpret_cast<int32_t*>(workspace);
                        float* xs_ptr = reinterpret_cast<float*>(q8_ptr + kDp4aDownQ8Elems);
                        moe_quant_i8_kernel<36><<<M * TOPK, 128, 0, stream>>>(act, q8_ptr, xs_ptr);
                        CUDA_CHECK(cudaGetLastError());
                        if (M == 2) {
                            launch(moe_down_nvfp4_dp4a_l1_kernel<2, true>, dim3(H / TILE), 512, pdl, stream, nullptr, q8_ptr, xs_ptr, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
                        } else if (M <= 4) {
                            launch(moe_down_nvfp4_dp4a_l1_kernel<4, true>, dim3(H / TILE), 512, pdl, stream, nullptr, q8_ptr, xs_ptr, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
                        } else {
                            launch(moe_down_nvfp4_dp4a_l1_kernel<8, true>, dim3(H / TILE), 512, pdl, stream, nullptr, q8_ptr, xs_ptr, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
                        }
                    } else {
                        // Scratch-free M > 1 path when caller provides no workspace (bit-identical numerics).
                        if (M == 2) {
                            launch(moe_down_nvfp4_dp4a_l1_kernel<2, false>, dim3(H / TILE), 512, pdl, stream, act, nullptr, nullptr, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
                        } else if (M <= 4) {
                            launch(moe_down_nvfp4_dp4a_l1_kernel<4, false>, dim3(H / TILE), 512, pdl, stream, act, nullptr, nullptr, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
                        } else {
                            launch(moe_down_nvfp4_dp4a_l1_kernel<8, false>, dim3(H / TILE), 512, pdl, stream, act, nullptr, nullptr, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
                        }
                    }
                } else {
                    if (M == 1)
                        launch(moe_down_fp16_stream_kernel<0, true>, dim3(H / TILE), 512, pdl, stream, act, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
                    else
                        launch(moe_down_fp16_stream_kernel<0, false>, dim3(H / TILE), 512, pdl, stream, act, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
                }
                return;
            } else {
                if (M == 1) {
                    static std::atomic<uint32_t> mxfp8_dn1_attr{0};
                    set_max_shared_once(moe_down_fp16_stream_kernel<1, true>, mxfp8_dn1_attr);
                    launch(moe_down_fp16_stream_kernel<1, true>, dim3(H / TILE), 512, pdl, stream, act, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
                    return;
                } else {
                    static std::atomic<uint32_t> mxfp8_dn_attr{0};
                    set_max_shared_once(moe_down_kernel<1>, mxfp8_dn_attr);
                }
            }
        }
    }
    if (fmt == Format::NVFP4)
        launch(moe_down_kernel<0>, dim3(H / TILE), DOWN_WARPS * 32, pdl, stream, act, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
    else if (fmt == Format::MXFP8)
        launch(moe_down_kernel<1>, dim3(H / TILE), DOWN_WARPS * 32, pdl, stream, act, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
    else if (fmt == Format::ATLAS2)
        launch(moe_down_kernel<2>, dim3(H / TILE), DOWN_WARPS * 32, pdl, stream, act, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
    else
        launch(moe_down_kernel<3>, dim3(H / TILE), DOWN_WARPS * 32, pdl, stream, act, idx, wts, w, ws, wf, M, conv_w, conv_state, out, conv_hist, wp, wsp);
}

}  // namespace ingine::kernels::moe

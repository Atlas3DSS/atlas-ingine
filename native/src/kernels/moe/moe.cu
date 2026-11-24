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

#include <cstdint>
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
__device__ __forceinline__ void decode(const Chunk<0>& ch, float factor, uint32_t (&o)[16]) {
    const uint32_t s2 = f16x2_from_e4m3x2(ch.s);
    const __half2 sc = *reinterpret_cast<const __half2*>(&s2);
    const __half2 slo = __low2half2(sc), shi = __high2half2(sc);
    const uint32_t words[4] = {ch.q.x, ch.q.y, ch.q.z, ch.q.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t h[4];
        f16x2_from_e2m1x4(words[w], h);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            // code * scale is exact in f16 (<= 5 significant bits, |v| in [2^-10, 2688]); the factor
            // multiply rounds once in fp32, then to bf16, as in decode_experts.
            const __half2 cs = __hmul2(*reinterpret_cast<const __half2*>(&h[b]), w < 2 ? slo : shi);
            const float2 v = __half22float2(cs);
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
    const uint32_t words[2] = {ch.q.x, ch.q.y};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint32_t w = words[half];
        const __half2 sch = half == 0 ? slo : shi;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const uint32_t q0 = (w >> (4 * b)) & 3u;
            const uint32_t q1 = (w >> (4 * b + 2)) & 3u;
            const uint32_t h0 = (((q0 ^ (q0 >> 1)) & 1u) ? 0x3740u : 0x3E00u) | (((~q0) & 2u) << 14);
            const uint32_t h1 = (((q1 ^ (q1 >> 1)) & 1u) ? 0x3740u : 0x3E00u) | (((~q1) & 2u) << 14);
            const uint32_t h2 = h0 | (h1 << 16);
            const __half2 cs = __hmul2(*reinterpret_cast<const __half2*>(&h2), sch);
            const float2 v = __half22float2(cs);
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
    const uint32_t los[2] = {ch.lo0, ch.lo1};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint32_t lo = los[half];
        const uint32_t hi = (ch.hi >> (16 * half)) & 0xffffu;
        const __half2 sch = half == 0 ? slo : shi;
#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const uint32_t c0 = ((lo >> (4 * b)) & 3u) | (((hi >> (2 * b)) & 1u) << 2);
            const uint32_t c1 = ((lo >> (4 * b + 2)) & 3u) | (((hi >> (2 * b + 1)) & 1u) << 2);
            const uint32_t m0 = (c0 & 4u) ? (c0 & 3u) : (3u - (c0 & 3u));
            const uint32_t m1 = (c1 & 4u) ? (c1 & 3u) : (3u - (c1 & 3u));
            const uint32_t sel = m0 | ((m0 + 4u) << 4) | (m1 << 8) | ((m1 + 4u) << 12);
            const uint32_t sgn = (((~c0) & 4u) << 13) | (((~c1) & 4u) << 29);
            const uint32_t h2 = __byte_perm(0x0000A040u, 0x403D3933u, sel) | sgn;
            const __half2 cs = __hmul2(*reinterpret_cast<const __half2*>(&h2), sch);
            const float2 v = __half22float2(cs);
            o[8 * half + b] = pack_bf16(v.x * factor, v.y * factor);
        }
    }
}

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
__device__ __forceinline__ void mma_frag(const Frag<FMT>& f, float (&d)[4]) {
    uint32_t a[16], b[16];
    decode(f.r0, f.factor, a);
    decode(f.r1, f.factor, b);
#pragma unroll
    for (int i = 0; i < 8; ++i) mma_bf16(d, a[2 * i], b[2 * i], a[2 * i + 1], b[2 * i + 1], f.x[2 * i], f.x[2 * i + 1]);
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
    const uint64_t mask = slot_mask(e, lo, hi);
    if (__ffsll(static_cast<long long>(mask)) - 1 != slot) return;  // an earlier slot owns this expert

    const size_t local_row = static_cast<size_t>(tile * TILE + g);
    const size_t row = static_cast<size_t>(e) * R + local_row;
    const uint8_t* we = w_ptrs ? (w_ptrs[e] + local_row * CB) : (w + row * CB);
    const uint8_t* wse = ws_ptrs ? (ws_ptrs[e] + local_row * SB) : (ws + row * SB);
    const bool on = g < M && ((mask >> (8 * g)) & 0xffu);
    Frag<FMT> f;
    load_w<FMT, K>(f, we, wse, 4 * warp + t, FMT != 1 ? wfactor[e] : 1.f, l2::policy<WEIGHT_L2>());
    load_x(f, on ? x + static_cast<size_t>(g) * K : nullptr, 4 * warp + t);
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    mma_frag(f, d);

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
        for (; remaining; ++u) {
            const int s = __ffsll(static_cast<long long>(remaining)) - 1;
            const int e = __shfl_sync(0xffffffffu, s < 32 ? lo : hi, s & 31);
            const uint64_t mask = slot_mask(e, lo, hi);
            remaining &= ~mask;
            if (lane == 0) uexp[u] = e, umask[u] = mask;
            if ((mask >> lane) & 1) slot_u[lane] = uint8_t(u);
            if ((mask >> (lane + 32)) & 1) slot_u[lane + 32] = uint8_t(u);
        }
        if (lane == 0) nuniq = u;
    }
    __syncthreads();

    const int U = nuniq;
    const int p = warp / DOWN_GROUPS, c = 4 * (warp % DOWN_GROUPS) + t;
    auto xrow = [&](int u) -> const ActT* {
        const uint32_t kmask = g < M ? uint32_t(umask[u] >> (8 * g)) & 0xffu : 0u;
        return kmask ? act + (static_cast<size_t>(g) * TOPK + (__ffs(kmask) - 1)) * K : nullptr;
    };
    const std::uint64_t wpol = l2::policy<WEIGHT_L2>();
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
    if (p < U) load_x(nxt, xrow(p), c);
#pragma unroll 1
    for (int base = 0; base < U; base += DOWN_PAR) {
        const Frag<FMT> cur = nxt;
        const int u = base + p;
        if (u + DOWN_PAR < U) load(u + DOWN_PAR, nxt), load_x(nxt, xrow(u + DOWN_PAR), c);  // next round
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        if (u < U) mma_frag(cur, d);
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
            s = __fadd_rn(s, __fmul_rn(ld_dep(wts + sl), ybuf[slot_u[sl]][r][n]));
        }
        sbuf[n][r] = bf16_round(s);
    }
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
#pragma unroll
            for (int kt_l = 0; kt_l < 8; ++kt_l) {
                const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_row + kt_l * TILE_BYTES);
                uint32_t h01, h23, h45, h67;
                exl3::dq8_mcg<BITS>([&](int wi) { return ld_stream_u1(tw + wi, wpol); }, lane, h01, h23, h45, h67);
                mma_f16(d, h01, h45, h23, h67, xb0[kt_l], xb1[kt_l]);
            }
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
#pragma unroll
        for (int kt_l = 0; kt_l < 8; ++kt_l) {
            const uint32_t* tw = reinterpret_cast<const uint32_t*>(w_row + kt_l * TILE_BYTES);
            uint32_t h01, h23, h45, h67;
            exl3::dq8_mcg<BITS>([&](int wi) { return ld_stream_u1(tw + wi, wpol); }, lane, h01, h23, h45, h67);
            mma_f16(d, h01, h45, h23, h67, xb0[kt_l], xb1[kt_l]);
        }
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

}  // namespace

void moe_up(const ActT* x, int M, const int* idx, const void* w13, const void* w13_scale,
            const float* w13_factor, Format fmt, ActT* out, bool pdl, cudaStream_t stream,
            const void* w13_rout, const ExpertPtrs* ptrs) {
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
    require(act && idx && wts && ((w2 && w2_scale) || (wp && wsp)) && conv_w && conv_state && out, "moe_down: null argument");
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

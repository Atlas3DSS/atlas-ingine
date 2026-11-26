// Batched multi-sequence decode kernels for Inkling-2-Mini continuous batching (B = 1..64+ slots).
// Exact arithmetic parity with kernels/attn/attn.cu (M=1) and kernels/prefill/rows.cu.
#include "kernels/batch/batch.h"

#include "core/act.h"
#include "core/device.h"
#include "core/gpu_compat.h"
#include "kernels/common/l2.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ingine::kernels::batch {
namespace {

#ifndef INGINE_TP_SIZE
#define INGINE_TP_SIZE 1
#endif
constexpr int HQ = 24 / INGINE_TP_SIZE, HKV = 8 / INGINE_TP_SIZE, D = 128, DREL = 16, EXT = 1024, WIN = 1024, GQA = 3, CONV_C = HKV * D;
constexpr int SINK_PAD = (HQ + 7) & ~7;
constexpr int NUM_FULL_VISITS = 8;
constexpr int OFF_GATE = HQ * D, OFF_REL = 2 * HQ * D, OFF_SINK = OFF_REL + HQ * DREL,
              OFF_K = OFF_SINK + SINK_PAD, OFF_V = OFF_K + CONV_C, OFF_KDYN = OFF_V + CONV_C,
              OFF_VDYN = OFF_KDYN + HKV * 4;
constexpr int R_SLIDING = OFF_KDYN, R_FULL = OFF_VDYN + HKV * 4;
constexpr int NT = 256;
constexpr int KW = 8;  // 8 warps per block
constexpr int U = 2;   // key pairs per warp per register buffer
constexpr int kSingleSplitMaxKeys = 1024;  // B-invariant single-split threshold (covers sliding & <=1024 full KV)
constexpr int H_DIM = 2304;
constexpr int PSTR = D + 4;
constexpr float CLIP_QK = 22.627416997969522f, CLIP_REL = 8.f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float QSCALE = 0.08838834764831845f * LOG2E;
constexpr float NEG_INIT = -1e30f;
constexpr float INV_SQRT5 = 1.f / 2.2360679774997896f;
constexpr float INV_3 = 1.f / 3.f;
constexpr l2::Evict KV_L2 = l2::Evict::first;

void require(bool ok, const char* what) {
  if (!ok) throw std::invalid_argument(std::string("batch: ") + what);
}

int choose_splits(int B) {
  if (B >= 16) return 1;
  int s = (128 + B * HKV - 1) / (B * HKV);
  if (s < 1) s = 1;
  if (s > 16) s = 16;
  return s;
}

int max_total_splits_up_to(int max_b) {
  int m = 16;
  const int lim = std::min(max_b, 15);
  for (int b = 1; b <= lim; ++b) {
    const int s = choose_splits(b);
    if (s > 1) m = std::max(m, b * s);
  }
  return std::max(m, max_b);
}

constexpr std::size_t align256(std::size_t n) { return (n + 255) & ~std::size_t(255); }

// Workspace layout for up to max_b sequences (non-power-of-two B in 3..15 can have B * choose_splits(B) up to 30):
//   part:      fp32 [max_total_splits_up_to(max_b), HKV, GQA, PSTR]
//   saved_nkv: ActT [max_b, HKV, 2, D]
//   saved_nst: fp32 [max_b, HKV, 3, 2, D]
std::size_t ws_nkv_off(int max_b) {
  const int max_ts = (INGINE_TP_SIZE > 1) ? std::max(16 * INGINE_TP_SIZE + 16, max_total_splits_up_to(max_b))
                                          : max_total_splits_up_to(max_b);
  return align256(std::size_t(max_ts) * HKV * GQA * PSTR * sizeof(float));
}
std::size_t ws_nst_off(int max_b) {
  return ws_nkv_off(max_b) + align256(std::size_t(max_b) * HKV * 2 * D * sizeof(ActT));
}
std::size_t ws_total_bytes(int max_b) {
  return ws_nst_off(max_b) + align256(std::size_t(max_b) * HKV * 3 * 2 * D * sizeof(float));
}

__device__ __forceinline__ float bf16r(float x) { return act::round_act(x); }

__device__ __forceinline__ float hsum16(float x) {
#pragma unroll
  for (int o = 1; o < 16; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}

__device__ __forceinline__ float wsum(float x) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}

__device__ __forceinline__ float clip_factor(float ss, float t) {
  const float norm = ss > 0.f ? ss * rsqrtf(ss) : 0.f;
  return fminf(1.f, __fdividef(t, norm + 1e-6f));
}

__device__ __forceinline__ float2 e4m3x2_f2(uint32_t pair) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && __CUDA_ARCH__ >= 890
  return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3)));
#else
  auto dec = [](uint32_t b) -> float {
    const uint32_t s = (b & 0x80u) << 24;
    const uint32_t a = b & 0x7fu;
    const uint32_t norm = (a + 960u) << 20;
    const uint32_t sub = a ? ((0x3B00u + (((0xBA986400u >> (a * 4u)) & 0xfu) << 5)) << 16) : 0u;
    return __uint_as_float(s | (a >= 8u ? norm : sub));
  };
  return make_float2(dec(pair & 0xffu), dec((pair >> 8) & 0xffu));
#endif
}

__device__ __forceinline__ void unpack8(const uint4 u, float f[8]) { act::unpack8(u, f); }

__device__ __forceinline__ void unpack8(const uint2 u, float f[8]) {
  float2 a = e4m3x2_f2(u.x & 0xffffu), b = e4m3x2_f2(u.x >> 16);
  f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
  a = e4m3x2_f2(u.y & 0xffffu), b = e4m3x2_f2(u.y >> 16);
  f[4] = a.x, f[5] = a.y, f[6] = b.x, f[7] = b.y;
}

__device__ __forceinline__ float e4m3r(float x) {
  return ingine::gpu::e4m3fn_to_float(ingine::gpu::float_to_e4m3fn_sat(x));
}

__device__ __forceinline__ uint32_t pack_e4m3x2(float a, float b) {
  return ingine::gpu::float2_to_e4m3fnx2_sat(a, b);
}

__device__ __forceinline__ uint4 ld_kv(const void* ptr, std::uint64_t pol) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  uint4 v;
  if constexpr (l2::hinted<KV_L2>)
    asm volatile("ld.global" INGINE_L1_NO_ALLOCATE ".L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr), "l"(pol)
                 : "memory");
  else
    asm volatile("ld.global" INGINE_L1_NO_ALLOCATE ".v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr)
                 : "memory");
  return v;
#else
  static_cast<void>(pol);
  return *static_cast<const uint4*>(ptr);
#endif
}

__device__ __forceinline__ uint2 ld_kv8(const void* ptr, std::uint64_t pol) {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  uint2 v;
  if constexpr (l2::hinted<KV_L2>)
    asm volatile("ld.global" INGINE_L1_NO_ALLOCATE ".L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(ptr), "l"(pol)
                 : "memory");
  else
    asm volatile("ld.global" INGINE_L1_NO_ALLOCATE ".v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(ptr) : "memory");
  return v;
#else
  static_cast<void>(pol);
  return *static_cast<const uint2*>(ptr);
#endif
}

template <bool PROJ_FP32>
__device__ __forceinline__ float load_proj_val(const void* proj, std::size_t idx) {
  if constexpr (PROJ_FP32) {
    return static_cast<const float*>(proj)[idx];
  } else {
    return act::to_float(static_cast<const ActT*>(proj)[idx]);
  }
}

struct BatchedAttnKernelParams {
  const void* proj;
  int R;
  int B;
  int splits;
  const std::int32_t* slot_ids;
  const std::int64_t* slot_pos;
  const float* q_scale;
  const float* r_scale;
  const float* rel_proj;
  const float* k_conv;
  const float* v_conv;
  float* conv_k_visit;
  float* conv_v_visit;
  std::size_t conv_slot_stride;
  ActT* k_ring_visit;
  ActT* v_ring_visit;
  std::int64_t* ring_pos_visit;
  std::size_t ring_slot_stride;
  std::size_t ring_pos_slot_stride;
  void* k_full_pool;
  void* v_full_pool;
  const std::int32_t* page_table;
  int max_pages_per_slot;
  int full_visit_idx;
  int max_ctx;
  int log_scaling;
  float* part;
  ActT* saved_nkv;
  float* saved_nst;
  ActT* out;
};

template <typename KV>
struct KeyBuf {
  KV k[U], v[U];
  int kp[U];
  int dp[U];
  float rp[U];
};

// Shared memory layout for one (split, kv_head, batch_item) CTA (~16.4 KiB static, fits SM60..SM120a).
struct AttnSmem {
  union {
    struct {
      float q[GQA * D];
      float qs[GQA * D];
      float rel[GQA * DREL];
      float rs[GQA * DREL];
      float st[3 * 2 * D];
      float x[2 * D];
      float w[2 * D * 4];
      float dyn[2 * 4];
      float nkf[2 * D];
    } stage;
    struct {
      float o[KW * GQA * D];
      float ml[KW * GQA * 2];
    } red;
  } u;
  float sink[GQA];
  float relp0[DREL];
  __align__(16) ActT nkv[2][D];
  float nst[3 * 2 * D];
};

template <bool FULL, bool F8, bool PROJ_FP32, bool SPLIT>
__global__ void __launch_bounds__(NT, 2) batched_attn_kernel(const BatchedAttnKernelParams p) {
  static_assert(FULL || !F8, "FP8 caches are full-layer only");
  using CT = std::conditional_t<F8, uint8_t, ActT>;
  using KV = std::conditional_t<F8, uint2, uint4>;

  const int rank = blockIdx.x;
  const int g = blockIdx.y;
  const int b = blockIdx.z;
  const int slot = p.slot_ids[b];
  if (slot < 0) return;

  const int p0 = int(p.slot_pos[slot]);
  if (FULL && p0 >= p.max_ctx) return;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int half = lane >> 4;
  const int c = lane & 15;
  const bool owner = (rank == 0);
  const int R = p.R;
  const std::size_t proj_base = std::size_t(b) * R;

  __shared__ AttnSmem sm;

  // Per-slot state pointers.
  float* const conv_k_slot = p.conv_k_visit + std::size_t(slot) * p.conv_slot_stride;
  float* const conv_v_slot = p.conv_v_visit + std::size_t(slot) * p.conv_slot_stride;
  ActT* const k_ring_slot = FULL ? nullptr : (p.k_ring_visit + std::size_t(slot) * p.ring_slot_stride);
  ActT* const v_ring_slot = FULL ? nullptr : (p.v_ring_visit + std::size_t(slot) * p.ring_slot_stride);
  std::int64_t* const kpos_slot = FULL ? nullptr : (p.ring_pos_visit + std::size_t(slot) * p.ring_pos_slot_stride);
  const std::int32_t* const pt_slot = FULL ? (p.page_table + std::size_t(slot) * p.max_pages_per_slot) : nullptr;
  CT* const k_pool = FULL ? static_cast<CT*>(p.k_full_pool) : reinterpret_cast<CT*>(k_ring_slot);
  CT* const v_pool = FULL ? static_cast<CT*>(p.v_full_pool) : reinterpret_cast<CT*>(v_ring_slot);

  // Stage q, q_scale, rel, r_scale, sink, and (owner) conv state/weights/inputs.
  for (int i = tid; i < GQA * D; i += NT) {
    sm.u.stage.q[i] = load_proj_val<PROJ_FP32>(p.proj, proj_base + g * GQA * D + i);
    sm.u.stage.qs[i] = p.q_scale[g * GQA * D + i];
  }
  for (int i = tid; i < GQA * DREL; i += NT) {
    sm.u.stage.rel[i] = load_proj_val<PROJ_FP32>(p.proj, proj_base + OFF_REL + g * GQA * DREL + i);
    sm.u.stage.rs[i] = p.r_scale[g * GQA * DREL + i];
  }
  if (tid < GQA) {
    sm.sink[tid] = load_proj_val<PROJ_FP32>(p.proj, proj_base + OFF_SINK + g * GQA + tid);
  }
  if (owner) {
    for (int i = tid; i < 3 * 2 * D; i += NT) {
      const int j = i / (2 * D), kv = (i / D) & 1, ch = i % D;
      sm.u.stage.st[i] = (kv ? conv_v_slot : conv_k_slot)[j * CONV_C + g * D + ch];
    }
    for (int i = tid; i < 2 * D; i += NT) {
      const int kv = i / D, ch = i % D;
      sm.u.stage.x[i] = load_proj_val<PROJ_FP32>(p.proj, proj_base + (kv ? OFF_V : OFF_K) + g * D + ch);
    }
    for (int i = tid; i < 2 * D * 4; i += NT) {
      const int kv = i / (D * 4), q = i % (D * 4);
      const float* w = kv ? p.v_conv : p.k_conv;
      if constexpr (FULL) {
        const int lag = q / D, ch = q % D;
        sm.u.stage.w[i] = w[lag * CONV_C + g * D + ch];
      } else {
        const int ch = q / 4, tap = q & 3;
        sm.u.stage.w[i] = w[(g * D + ch) * 4 + tap];
      }
    }
    if constexpr (FULL) {
      if (tid < 8) {
        const int kv = tid >> 2, lag = tid & 3;
        sm.u.stage.dyn[tid] = load_proj_val<PROJ_FP32>(p.proj, proj_base + (kv ? OFF_VDYN : OFF_KDYN) + g * 4 + lag);
      }
    }
    if (tid < DREL) {
      sm.relp0[tid] = p.rel_proj[tid * EXT];
    }
  }
  __syncthreads();

  // ---- q and rel post-processing (exact match to attn.cu M=1)
  const float tau = p.log_scaling ? 1.f + 0.1f * logf(fmaxf(float(p0 + 1) * (1.f / 8192.f), 1.f)) : 1.f;
  if (warp < 2 * GQA) {
    const int i = warp % GQA;
    if (warp < GQA) {
      float4* qa = reinterpret_cast<float4*>(sm.u.stage.q + i * D) + lane;
      const float4 a = *qa;
      const float4 s = reinterpret_cast<const float4*>(sm.u.stage.qs + i * D)[lane];
      float x[4] = {a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w};
      const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        x[e] = bf16r(x[e] * f);
        if (p.log_scaling) x[e] = bf16r(x[e] * tau);
      }
      *qa = make_float4(x[0] * QSCALE, x[1] * QSCALE, x[2] * QSCALE, x[3] * QSCALE);
    } else {
      float* ra = sm.u.stage.rel + i * DREL + c;
      const float r = *ra * sm.u.stage.rs[i * DREL + c];
      float y = bf16r(r * clip_factor(hsum16(r * r), CLIP_REL));
      if (p.log_scaling) y = bf16r(y * tau);
      __syncwarp();
      if (half == 0) *ra = y;
    }
  }
  __syncthreads();

  float qv[GQA][8], relv[GQA];
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    const float4* qa = reinterpret_cast<const float4*>(sm.u.stage.q + i * D + c * 8);
    const float4 a0 = qa[0], a1 = qa[1];
    qv[i][0] = a0.x, qv[i][1] = a0.y, qv[i][2] = a0.z, qv[i][3] = a0.w;
    qv[i][4] = a1.x, qv[i][5] = a1.y, qv[i][6] = a1.z, qv[i][7] = a1.w;
    relv[i] = sm.u.stage.rel[i * DREL + c];
  }

  // ---- owner: new K / V conv + key clip / value RMS norm + E4M3 quant
  if (owner) {
    for (int idx = tid; idx < 2 * D; idx += NT) {
      const int kv = idx / D, ch = idx % D;
      float e[4];
      e[0] = sm.u.stage.st[(0 * 2 + kv) * D + ch];
      e[1] = sm.u.stage.st[(1 * 2 + kv) * D + ch];
      e[2] = sm.u.stage.st[(2 * 2 + kv) * D + ch];
      e[3] = bf16r(sm.u.stage.x[kv * D + ch]);
      float y;
      if constexpr (FULL) {
        float o = 0.f;
#pragma unroll
        for (int lag = 0; lag < 4; ++lag) {
          const float w = __fadd_rn(sm.u.stage.w[(kv * 4 + lag) * D + ch], sm.u.stage.dyn[kv * 4 + lag]);
          o = __fadd_rn(o, __fmul_rn(w, e[3 - lag]));
        }
        y = __fmul_rn(__fadd_rn(o, e[3]), INV_3);
      } else {
        const float* w = sm.u.stage.w + (kv * D + ch) * 4;
        float o = __fmul_rn(w[0], e[0]);
        o = __fadd_rn(o, __fmul_rn(w[1], e[1]));
        o = __fadd_rn(o, __fmul_rn(w[2], e[2]));
        o = __fadd_rn(o, __fmul_rn(w[3], e[3]));
        y = __fmul_rn(__fadd_rn(o, e[3]), INV_SQRT5);
      }
      sm.u.stage.nkf[kv * D + ch] = bf16r(y);
      sm.nst[(0 * 2 + kv) * D + ch] = e[1];
      sm.nst[(1 * 2 + kv) * D + ch] = e[2];
      sm.nst[(2 * 2 + kv) * D + ch] = e[3];
    }
    __syncthreads();
    if (warp < 2) {
      const int kv = warp;
      const float4 x = reinterpret_cast<const float4*>(sm.u.stage.nkf + kv * D)[lane];
      const float ss = wsum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
      const float f = kv == 0 ? clip_factor(ss, CLIP_QK) : rsqrtf(ss / float(D) + 1e-6f);
      float y[4] = {x.x * f, x.y * f, x.z * f, x.w * f};
      if constexpr (F8) {
#pragma unroll
        for (int e = 0; e < 4; ++e) y[e] = e4m3r(bf16r(y[e]));
      }
      reinterpret_cast<uint2*>(&sm.nkv[kv][0])[lane] = make_uint2(act::pack2(y[0], y[1]), act::pack2(y[2], y[3]));
    }
    __syncthreads();
  }

  // ---- K/V streaming over cached keys
  // Sliding ring entries are written sequentially at p & (WIN - 1) from a -1-initialized ring,
  // so for p0 < WIN all valid entries live in [0, p0) and scanning [p0, WIN) wastes KV bandwidth.
  // For old_end <= kSingleSplitMaxKeys, owner (rank 0) runs the full single-split reduction order so
  // B-varying split counts (16/8/4/3/2/1) produce 0-ULP bit-identical attention outputs across all B.
  const int old_end = FULL ? p0 : (p0 < WIN ? p0 : WIN);
  const bool single_split = (old_end <= kSingleSplitMaxKeys);
  constexpr int STEP = KW * 2 * U;
  constexpr int TILE = 2 * STEP;
  const int stride = single_split ? TILE : (p.splits * TILE);
  const int c0 = single_split ? 0 : (rank * TILE);
  const std::uint64_t kvpol = l2::policy<KV_L2>();

  auto kv_offset = [&](int j) -> std::size_t {
    if constexpr (FULL) {
      const int phys = pt_slot[j >> kPageShift];
      return ((((std::size_t(phys) * NUM_FULL_VISITS + p.full_visit_idx) * kPageSize + (j & kPageMask)) * HKV + g) * D) +
             c * 8;
    } else {
      return (std::size_t(j) * HKV + g) * D + c * 8;
    }
  };

  auto load_buf = [&](KeyBuf<KV>& buf, int base) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int j = base + (warp * U + u) * 2 + half;
      buf.k[u] = buf.v[u] = KV{};
      buf.kp[u] = -1;
      buf.dp[u] = FULL ? (p0 - j) : (1 + ((p0 - 1 - j) & (WIN - 1)));
      buf.rp[u] = (j < old_end && buf.dp[u] < EXT) ? __ldg(p.rel_proj + c * EXT + buf.dp[u]) : 0.f;
      if (j < old_end) {
        const std::size_t off = kv_offset(j);
        if constexpr (F8) {
          buf.k[u] = ld_kv8(k_pool + off, kvpol);
          buf.v[u] = ld_kv8(v_pool + off, kvpol);
        } else {
          buf.k[u] = ld_kv(k_pool + off, kvpol);
          buf.v[u] = ld_kv(v_pool + off, kvpol);
        }
        buf.kp[u] = FULL ? j : int(kpos_slot[j]);
      }
    }
  };

  float mrun[GQA], lrun[GQA], acc[GQA][8];
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    mrun[i] = NEG_INIT;
    lrun[i] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  }

  auto compute = [&](const auto& buf, bool is_new) {
    float s[U][GQA];
    bool valid[U], need[U];
    int dist[U];
    bool any_need = false;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int kp = buf.kp[u], d = p0 - kp;
      const bool ok = kp >= 0 && d >= 0 && (is_new || (kp < p0 && (FULL || d < WIN)));
      valid[u] = ok;
      need[u] = ok && d < EXT;
      dist[u] = d;
      any_need |= need[u];
      float kf[8];
      unpack8(buf.k[u], kf);
#pragma unroll
      for (int i = 0; i < GQA; ++i) {
        float t = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) t = fmaf(qv[i][e], kf[e], t);
        s[u][i] = hsum16(t);
      }
    }
    if (__any_sync(0xffffffffu, any_need)) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        float rp = buf.rp[u];
        if (need[u] && dist[u] != buf.dp[u]) rp = is_new ? sm.relp0[c] : __ldg(p.rel_proj + c * EXT + dist[u]);
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
          const float bias = hsum16(relv[i] * rp);
          if (need[u]) s[u][i] = fmaf(bf16r(bias), LOG2E, s[u][i]);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      float mx = mrun[i];
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if (!valid[u]) s[u][i] = -INFINITY;
        mx = fmaxf(mx, s[u][i]);
      }
      const float sc = exp2f(mrun[i] - mx);
      mrun[i] = mx;
      lrun[i] *= sc;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] *= sc;
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      float vf[8];
      using V = std::decay_t<decltype(buf.v[0])>;
      unpack8(valid[u] ? buf.v[u] : V{}, vf);
#pragma unroll
      for (int i = 0; i < GQA; ++i) {
        const float pu = exp2f(s[u][i] - mrun[i]);
        lrun[i] += pu;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(pu, vf[e], acc[i][e]);
      }
    }
  };

  KeyBuf<KV> A, B;
  const bool run_keys = owner || !single_split;
  if (run_keys && c0 < old_end) load_buf(A, c0);
  if (run_keys && c0 + STEP < old_end) load_buf(B, c0 + STEP);

  if (owner) {
    KeyBuf<uint4> N;
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int m2 = (warp * U + u) * 2 + half;
      N.k[u] = N.v[u] = make_uint4(0u, 0u, 0u, 0u);
      N.kp[u] = -1, N.dp[u] = -1, N.rp[u] = 0.f;
      if (m2 == 0) {
        N.k[u] = *reinterpret_cast<const uint4*>(&sm.nkv[0][c * 8]);
        N.v[u] = *reinterpret_cast<const uint4*>(&sm.nkv[1][c * 8]);
        N.kp[u] = p0;
      }
    }
    compute(N, true);
  }

  if (run_keys) {
    for (int base = c0; base < old_end; base += stride) {
      compute(A, false);
      if (base + stride < old_end) load_buf(A, base + stride);
      if (base + STEP < old_end) {
        compute(B, false);
        if (base + stride + STEP < old_end) load_buf(B, base + stride + STEP);
      }
    }
  }

  // ---- Merge the two half-warps, then the 8 warps in shared memory
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    const float mo = __shfl_xor_sync(0xffffffffu, mrun[i], 16);
    const float lo = __shfl_xor_sync(0xffffffffu, lrun[i], 16);
    const float mx = fmaxf(mrun[i], mo);
    const float f1 = exp2f(mrun[i] - mx), f2 = exp2f(mo - mx);
    lrun[i] = lrun[i] * f1 + lo * f2;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      acc[i][e] = acc[i][e] * f1 + __shfl_xor_sync(0xffffffffu, acc[i][e], 16) * f2;
    mrun[i] = mx;
  }
  __syncthreads();  // stage inputs consumed; alias sm.u.red
  if (half == 0) {
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      float4* dst = reinterpret_cast<float4*>(sm.u.red.o + (warp * GQA + i) * D + c * 8);
      dst[0] = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
      dst[1] = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
      if (c == 0) {
        sm.u.red.ml[(warp * GQA + i) * 2] = mrun[i];
        sm.u.red.ml[(warp * GQA + i) * 2 + 1] = lrun[i];
      }
    }
  }
  __syncthreads();

  if constexpr (SPLIT) {
    // Write split partial to workspace; owner saves nkv/nst for stage 2.
    float* part = p.part + std::size_t((b * HKV + g) * p.splits + rank) * GQA * PSTR;
    for (int idx = tid; idx < GQA * (D / 4); idx += NT) {
      const int row = idx / (D / 4), d4 = idx % (D / 4);
      float mx = NEG_INIT;
#pragma unroll
      for (int k = 0; k < KW; ++k) mx = fmaxf(mx, sm.u.red.ml[(k * GQA + row) * 2]);
      float l = 0.f;
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int k = 0; k < KW; ++k) {
        const int w = k * GQA + row;
        const float f = exp2f(sm.u.red.ml[w * 2] - mx);
        l += sm.u.red.ml[w * 2 + 1] * f;
        const float4 a = reinterpret_cast<const float4*>(sm.u.red.o + w * D)[d4];
        o.x += a.x * f, o.y += a.y * f, o.z += a.z * f, o.w += a.w * f;
      }
      reinterpret_cast<float4*>(part + row * PSTR)[d4] = o;
      if (d4 == 0) *reinterpret_cast<float2*>(part + row * PSTR + D) = make_float2(mx, l);
    }
    if (owner) {
      ActT* dst_nkv = p.saved_nkv + std::size_t(b * HKV + g) * 2 * D;
      for (int idx = tid; idx < 2 * 16; idx += NT) {
        const int kv = idx / 16, c8 = idx % 16;
        reinterpret_cast<uint4*>(dst_nkv + kv * D)[c8] = reinterpret_cast<const uint4*>(&sm.nkv[kv][0])[c8];
      }
      float* dst_nst = p.saved_nst + std::size_t(b * HKV + g) * 3 * 2 * D;
      for (int idx = tid; idx < 3 * 2 * D; idx += NT) dst_nst[idx] = sm.nst[idx];
    }
  } else {
    // ---- Non-split single-launch finish: match split-reduce 2-step warp->sink merge for 0-ULP invariance!
    for (int idx = tid; idx < GQA * D; idx += NT) {
      const int row = idx / D, dim = idx % D;
      float mx0 = NEG_INIT;
#pragma unroll
      for (int k = 0; k < KW; ++k) mx0 = fmaxf(mx0, sm.u.red.ml[(k * GQA + row) * 2]);
      float l0 = 0.f, a0 = 0.f;
#pragma unroll
      for (int k = 0; k < KW; ++k) {
        const int w = k * GQA + row;
        const float f = exp2f(sm.u.red.ml[w * 2] - mx0);
        l0 += sm.u.red.ml[w * 2 + 1] * f;
        a0 += sm.u.red.o[w * D + dim] * f;
      }
      const float sink = sm.sink[row] * LOG2E;
      const float mx = fmaxf(sink, mx0);
      const float f0 = exp2f(mx0 - mx);
      float l = exp2f(sink - mx), a = 0.f;
      l += l0 * f0;
      a += a0 * f0;
      const float gate_in = load_proj_val<PROJ_FP32>(p.proj, proj_base + OFF_GATE + (g * GQA + row) * D + dim);
      const float gate = 1.f / (1.f + expf(-gate_in));
      p.out[std::size_t(b) * HQ * D + (g * GQA + row) * D + dim] = act::from_float(bf16r(a / l) * gate);
    }

    // Commit new K/V row, conv state, and (sliding, head 0) ring_pos.
    for (int idx = tid; idx < 2 * 16; idx += NT) {
      const int kv = idx / 16, c8 = idx % 16;
      const uint4 u = reinterpret_cast<const uint4*>(&sm.nkv[kv][0])[c8];
      std::size_t dst_off;
      if constexpr (FULL) {
        const int phys = pt_slot[p0 >> kPageShift];
        dst_off = ((((std::size_t(phys) * NUM_FULL_VISITS + p.full_visit_idx) * kPageSize + (p0 & kPageMask)) * HKV + g) * D);
      } else {
        dst_off = (std::size_t(p0 & (WIN - 1)) * HKV + g) * D;
      }
      CT* const dst = (kv ? v_pool : k_pool) + dst_off;
      if constexpr (F8) {
        float f[8];
        unpack8(u, f);
        reinterpret_cast<uint2*>(dst)[c8] = make_uint2(pack_e4m3x2(f[0], f[1]) | (pack_e4m3x2(f[2], f[3]) << 16),
                                                       pack_e4m3x2(f[4], f[5]) | (pack_e4m3x2(f[6], f[7]) << 16));
      } else {
        reinterpret_cast<uint4*>(dst)[c8] = u;
      }
    }
    for (int idx = tid; idx < 3 * 2 * D; idx += NT) {
      const int i = idx / (2 * D), kv = (idx / D) & 1, ch = idx % D;
      (kv ? conv_v_slot : conv_k_slot)[i * CONV_C + g * D + ch] = sm.nst[idx];
    }
    if (!FULL && g == 0 && tid == 0) {
      kpos_slot[p0 & (WIN - 1)] = p0;
    }
  }
}

template <bool FULL, bool F8, bool PROJ_FP32>
__global__ void __launch_bounds__(NT) batched_attn_reduce_kernel(const BatchedAttnKernelParams p) {
  static_assert(FULL || !F8, "FP8 caches are full-layer only");
  using CT = std::conditional_t<F8, uint8_t, ActT>;

  const int g = blockIdx.y;
  const int b = blockIdx.z;
  const int slot = p.slot_ids[b];
  if (slot < 0) return;

  const int p0 = int(p.slot_pos[slot]);
  if (FULL && p0 >= p.max_ctx) return;

  const int tid = threadIdx.x;
  const int R = p.R;
  const std::size_t proj_base = std::size_t(b) * R;
  const float* base_part = p.part + std::size_t((b * HKV + g) * p.splits) * GQA * PSTR;

  for (int idx = tid; idx < GQA * D; idx += NT) {
    const int row = idx / D, dim = idx % D;
    const float sink = load_proj_val<PROJ_FP32>(p.proj, proj_base + OFF_SINK + g * GQA + row) * LOG2E;
    float mx = sink;
    for (int q = 0; q < p.splits; ++q) {
      const float2 ml = *reinterpret_cast<const float2*>(base_part + (q * GQA + row) * PSTR + D);
      mx = fmaxf(mx, ml.x);
    }
    float l = exp2f(sink - mx), a = 0.f;
    for (int q = 0; q < p.splits; ++q) {
      const float* pr = base_part + (q * GQA + row) * PSTR;
      const float2 ml = *reinterpret_cast<const float2*>(pr + D);
      const float f = exp2f(ml.x - mx);
      l += ml.y * f;
      a += pr[dim] * f;
    }
    const float gate_in = load_proj_val<PROJ_FP32>(p.proj, proj_base + OFF_GATE + (g * GQA + row) * D + dim);
    const float gate = 1.f / (1.f + expf(-gate_in));
    p.out[std::size_t(b) * HQ * D + (g * GQA + row) * D + dim] = act::from_float(bf16r(a / l) * gate);
  }

  // Commit saved K/V row, conv state, and ring_pos.
  float* const conv_k_slot = p.conv_k_visit + std::size_t(slot) * p.conv_slot_stride;
  float* const conv_v_slot = p.conv_v_visit + std::size_t(slot) * p.conv_slot_stride;
  ActT* const k_ring_slot = FULL ? nullptr : (p.k_ring_visit + std::size_t(slot) * p.ring_slot_stride);
  ActT* const v_ring_slot = FULL ? nullptr : (p.v_ring_visit + std::size_t(slot) * p.ring_slot_stride);
  std::int64_t* const kpos_slot = FULL ? nullptr : (p.ring_pos_visit + std::size_t(slot) * p.ring_pos_slot_stride);
  const std::int32_t* const pt_slot = FULL ? (p.page_table + std::size_t(slot) * p.max_pages_per_slot) : nullptr;
  CT* const k_pool = FULL ? static_cast<CT*>(p.k_full_pool) : reinterpret_cast<CT*>(k_ring_slot);
  CT* const v_pool = FULL ? static_cast<CT*>(p.v_full_pool) : reinterpret_cast<CT*>(v_ring_slot);

  const ActT* src_nkv = p.saved_nkv + std::size_t(b * HKV + g) * 2 * D;
  for (int idx = tid; idx < 2 * 16; idx += NT) {
    const int kv = idx / 16, c8 = idx % 16;
    const uint4 u = reinterpret_cast<const uint4*>(src_nkv + kv * D)[c8];
    std::size_t dst_off;
    if constexpr (FULL) {
      const int phys = pt_slot[p0 >> kPageShift];
      dst_off = ((((std::size_t(phys) * NUM_FULL_VISITS + p.full_visit_idx) * kPageSize + (p0 & kPageMask)) * HKV + g) * D);
    } else {
      dst_off = (std::size_t(p0 & (WIN - 1)) * HKV + g) * D;
    }
    CT* const dst = (kv ? v_pool : k_pool) + dst_off;
    if constexpr (F8) {
      float f[8];
      unpack8(u, f);
      reinterpret_cast<uint2*>(dst)[c8] = make_uint2(pack_e4m3x2(f[0], f[1]) | (pack_e4m3x2(f[2], f[3]) << 16),
                                                     pack_e4m3x2(f[4], f[5]) | (pack_e4m3x2(f[6], f[7]) << 16));
    } else {
      reinterpret_cast<uint4*>(dst)[c8] = u;
    }
  }
  const float* src_nst = p.saved_nst + std::size_t(b * HKV + g) * 3 * 2 * D;
  for (int idx = tid; idx < 3 * 2 * D; idx += NT) {
    const int i = idx / (2 * D), kv = (idx / D) & 1, ch = idx % D;
    (kv ? conv_v_slot : conv_k_slot)[i * CONV_C + g * D + ch] = src_nst[idx];
  }
  if (!FULL && g == 0 && tid == 0) {
    kpos_slot[p0 & (WIN - 1)] = p0;
  }
}

template <bool X_FP32>
__global__ void batched_static_conv_kernel(const void* __restrict__ x, int ld, int C,
                                           const std::int32_t* __restrict__ slot_ids,
                                           const float* __restrict__ conv_w, float* __restrict__ conv_state_visit,
                                           std::size_t conv_slot_stride, float* __restrict__ out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;
  if (c >= C) return;
  const int slot = slot_ids[b];
  if (slot < 0) return;

  float* const st = conv_state_visit + std::size_t(slot) * conv_slot_stride;
  const float e0 = st[0 * C + c];
  const float e1 = st[1 * C + c];
  const float e2 = st[2 * C + c];
  float e3;
  if constexpr (X_FP32) {
    e3 = bf16r(static_cast<const float*>(x)[std::size_t(b) * ld + c]);
  } else {
    e3 = act::to_float(static_cast<const ActT*>(x)[std::size_t(b) * ld + c]);
  }
  const float4 w4 = reinterpret_cast<const float4*>(conv_w)[c];
  float acc = __fmul_rn(w4.x, e0);
  acc = __fadd_rn(acc, __fmul_rn(w4.y, e1));
  acc = __fadd_rn(acc, __fmul_rn(w4.z, e2));
  acc = __fadd_rn(acc, __fmul_rn(w4.w, e3));
  out[std::size_t(b) * C + c] = __fmul_rn(__fadd_rn(acc, e3), INV_SQRT5);
  st[0 * C + c] = e1;
  st[1 * C + c] = e2;
  st[2 * C + c] = e3;
}

template <bool SCATTER, bool F8>
__global__ void paged_full_kv_transfer_kernel(const void* __restrict__ scratch_k, const void* __restrict__ scratch_v,
                                              void* __restrict__ pool_k, void* __restrict__ pool_v,
                                              int scratch_max_ctx, int pos0, int T, int slot,
                                              const std::int32_t* __restrict__ page_table, int max_pages_per_slot) {
  // Grid: (T, NUM_FULL_VISITS); block: 256 threads, each handles 4 channels (4 * 256 = 1024 = KV_DIM).
  const int t = blockIdx.x;
  const int fi = blockIdx.y;
  if (t >= T) return;
  const int pos = pos0 + t;
  const int phys = page_table[std::size_t(slot) * max_pages_per_slot + (pos >> kPageShift)];
  const int off_in_page = pos & kPageMask;

  constexpr int EPT = 4;  // 4 elements per thread -> 1024 elements per row
  const int ch = threadIdx.x * EPT;
  const std::size_t scratch_off = (std::size_t(fi) * scratch_max_ctx + pos) * CONV_C + ch;
  const std::size_t pool_off =
      (((std::size_t(phys) * NUM_FULL_VISITS + fi) * kPageSize + off_in_page) * CONV_C) + ch;

  if constexpr (F8) {
    if constexpr (SCATTER) {
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(pool_k) + pool_off) =
          *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(scratch_k) + scratch_off);
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(pool_v) + pool_off) =
          *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(scratch_v) + scratch_off);
    } else {
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(const_cast<void*>(scratch_k)) + scratch_off) =
          *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(pool_k) + pool_off);
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(const_cast<void*>(scratch_v)) + scratch_off) =
          *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(pool_v) + pool_off);
    }
  } else {
    if constexpr (SCATTER) {
      *reinterpret_cast<uint2*>(static_cast<ActT*>(pool_k) + pool_off) =
          *reinterpret_cast<const uint2*>(static_cast<const ActT*>(scratch_k) + scratch_off);
      *reinterpret_cast<uint2*>(static_cast<ActT*>(pool_v) + pool_off) =
          *reinterpret_cast<const uint2*>(static_cast<const ActT*>(scratch_v) + scratch_off);
    } else {
      *reinterpret_cast<uint2*>(static_cast<ActT*>(const_cast<void*>(scratch_k)) + scratch_off) =
          *reinterpret_cast<const uint2*>(static_cast<const ActT*>(pool_k) + pool_off);
      *reinterpret_cast<uint2*>(static_cast<ActT*>(const_cast<void*>(scratch_v)) + scratch_off) =
          *reinterpret_cast<const uint2*>(static_cast<const ActT*>(pool_v) + pool_off);
    }
  }
}

template <bool TO_TAIL, bool F8>
__global__ void tail_kv_scratch_kernel(const void* __restrict__ scratch_k, const void* __restrict__ scratch_v,
                                       void* __restrict__ tail_k, void* __restrict__ tail_v, int scratch_max_ctx,
                                       int pos0, int tail_tokens) {
  const int t = blockIdx.x;
  const int fi = blockIdx.y;
  if (t >= tail_tokens) return;
  const int ch = threadIdx.x * 4;
  const std::size_t scratch_off = (std::size_t(fi) * scratch_max_ctx + (pos0 + t)) * CONV_C + ch;
  const std::size_t tail_off = (std::size_t(fi) * tail_tokens + t) * CONV_C + ch;
  if constexpr (F8) {
    if constexpr (TO_TAIL) {
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(tail_k) + tail_off) =
          *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(scratch_k) + scratch_off);
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(tail_v) + tail_off) =
          *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(scratch_v) + scratch_off);
    } else {
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(const_cast<void*>(scratch_k)) + scratch_off) =
          *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(tail_k) + tail_off);
      *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(const_cast<void*>(scratch_v)) + scratch_off) =
          *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(tail_v) + tail_off);
    }
  } else {
    if constexpr (TO_TAIL) {
      *reinterpret_cast<uint2*>(static_cast<ActT*>(tail_k) + tail_off) =
          *reinterpret_cast<const uint2*>(static_cast<const ActT*>(scratch_k) + scratch_off);
      *reinterpret_cast<uint2*>(static_cast<ActT*>(tail_v) + tail_off) =
          *reinterpret_cast<const uint2*>(static_cast<const ActT*>(scratch_v) + scratch_off);
    } else {
      *reinterpret_cast<uint2*>(static_cast<ActT*>(const_cast<void*>(scratch_k)) + scratch_off) =
          *reinterpret_cast<const uint2*>(static_cast<const ActT*>(tail_k) + tail_off);
      *reinterpret_cast<uint2*>(static_cast<ActT*>(const_cast<void*>(scratch_v)) + scratch_off) =
          *reinterpret_cast<const uint2*>(static_cast<const ActT*>(tail_v) + tail_off);
    }
  }
}

template <bool F8>
__global__ void tail_kv_to_pool_page_kernel(const void* __restrict__ tail_k, const void* __restrict__ tail_v,
                                            int tail_tokens, int phys_page, void* __restrict__ pool_k,
                                            void* __restrict__ pool_v) {
  const int t = blockIdx.x;
  const int fi = blockIdx.y;
  if (t >= tail_tokens) return;
  const int ch = threadIdx.x * 4;
  const std::size_t tail_off = (std::size_t(fi) * tail_tokens + t) * CONV_C + ch;
  const std::size_t pool_off = (((std::size_t(phys_page) * NUM_FULL_VISITS + fi) * kPageSize + t) * CONV_C) + ch;
  if constexpr (F8) {
    *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(pool_k) + pool_off) =
        *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(tail_k) + tail_off);
    *reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(pool_v) + pool_off) =
        *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(tail_v) + tail_off);
  } else {
    *reinterpret_cast<uint2*>(static_cast<ActT*>(pool_k) + pool_off) =
        *reinterpret_cast<const uint2*>(static_cast<const ActT*>(tail_k) + tail_off);
    *reinterpret_cast<uint2*>(static_cast<ActT*>(pool_v) + pool_off) =
        *reinterpret_cast<const uint2*>(static_cast<const ActT*>(tail_v) + tail_off);
  }
}

__global__ void batched_head_softcap_kernel(const ActT* __restrict__ in, int B, int chunk_V, int v_offset,
                                            int total_V, float* __restrict__ out) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;
  if (v >= chunk_V || b >= B) return;
  const float x = act::to_float(in[std::size_t(b) * chunk_V + v]);
  out[std::size_t(b) * total_V + v_offset + v] = 30.f * tanhf(x * (1.f / 30.f));
}

__global__ void dequant_e4m3_unscaled_kernel(const std::uint8_t* __restrict__ q, long long total16,
                                             ActT* __restrict__ out) {
  const long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (i >= total16) return;
  const uint4 u = reinterpret_cast<const uint4*>(q)[i];
  float f0[8], f1[8];
  unpack8(make_uint2(u.x, u.y), f0);
  unpack8(make_uint2(u.z, u.w), f1);
  uint4* dst = reinterpret_cast<uint4*>(out + i * 16);
  dst[0] = make_uint4(act::pack2(f0[0], f0[1]), act::pack2(f0[2], f0[3]),
                      act::pack2(f0[4], f0[5]), act::pack2(f0[6], f0[7]));
  dst[1] = make_uint4(act::pack2(f1[0], f1[1]), act::pack2(f1[2], f1[3]),
                      act::pack2(f1[4], f1[5]), act::pack2(f1[6], f1[7]));
}

__global__ void batched_head_scale_softcap_kernel(const float* __restrict__ in, const float* __restrict__ scale, int M,
                                                  int chunk_V, int v_offset, int total_V, float* __restrict__ out) {
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  if (v >= chunk_V || m >= M) return;
  float x = in[std::size_t(m) * chunk_V + v];
  if (scale) x *= __ldg(scale + v_offset + v);
  out[std::size_t(m) * total_V + v_offset + v] = 30.f * tanhf(bf16r(x) * (1.f / 30.f));
}

// Deterministic M-invariant router logits: 16 CTAs x 8 expert rows = 128 experts; each CTA tiles 8 tokens.
// Every token/expert dot product executes the exact same 9x8 FMA + wsum order regardless of M.
__global__ void __launch_bounds__(256) batched_router_logits_kernel(const ActT* __restrict__ x,
                                                                    const ActT* __restrict__ w, int M,
                                                                    float* __restrict__ out) {
  constexpr int EXPERTS = 128;
  constexpr int RPC = 8;
  constexpr int CH = H_DIM / (8 * 32);  // 9
  constexpr int TM = 8;
  __shared__ __align__(16) ActT xs[TM][H_DIM];

  const int m0 = blockIdx.y * TM;
  const int cm = min(TM, M - m0);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row = blockIdx.x * RPC + warp;

  const int total_u4 = cm * (H_DIM / 8);
  for (int k = tid; k < total_u4; k += 256) {
    const int m = k / (H_DIM / 8);
    const int c8 = k % (H_DIM / 8);
    reinterpret_cast<uint4*>(&xs[m][0])[c8] =
        __ldg(reinterpret_cast<const uint4*>(x + std::size_t(m0 + m) * H_DIM) + c8);
  }
  const ActT* wr = w + std::size_t(row) * H_DIM;
  uint4 wv[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    wv[i] = __ldg(reinterpret_cast<const uint4*>(wr + (i * 32 + lane) * 8));
  }
  __syncthreads();

  float acc[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) acc[m] = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = (i * 32 + lane) * 8;
    float wf[8];
    unpack8(wv[i], wf);
    for (int m = 0; m < cm; ++m) {
      float xf[8];
      unpack8(*reinterpret_cast<const uint4*>(&xs[m][col]), xf);
      float a = acc[m];
#pragma unroll
      for (int j = 0; j < 8; ++j) a = fmaf(wf[j], xf[j], a);
      acc[m] = a;
    }
  }
  for (int m = 0; m < cm; ++m) {
    const float s = wsum(acc[m]);
    if (lane == 0) out[std::size_t(m0 + m) * EXPERTS + row] = s;
  }
}

// Deterministic M-invariant message gates (4 kernels, 0 ULP across all M >= 1).
__global__ void __launch_bounds__(H_DIM / 4) batched_gate_inputs_kernel(const float* __restrict__ h, bool h_shared,
                                                                        const float* __restrict__ msg, int M,
                                                                        ActT* __restrict__ gin) {
  constexpr int ROW_W = (H_DIM / 4) / 32;  // 18 warps
  __shared__ float red[ROW_W][2];

  const int t = blockIdx.x;
  const int g = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int c = tid * 4;

  const float* hg = h + (h_shared ? std::size_t(0) : std::size_t(g) * M * H_DIM) + std::size_t(t) * H_DIM;
  const float* mg = msg + std::size_t(g) * M * H_DIM + std::size_t(t) * H_DIM;
  const float4 hv = *reinterpret_cast<const float4*>(hg + c);
  const float4 mv = *reinterpret_cast<const float4*>(mg + c);

  float sh = wsum(hv.x * hv.x + hv.y * hv.y + hv.z * hv.z + hv.w * hv.w);
  float sm = wsum(mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w);
  if (lane == 0) {
    red[warp][0] = sh;
    red[warp][1] = sm;
  }
  __syncthreads();

  float tot_h = 0.f, tot_m = 0.f;
#pragma unroll
  for (int w = 0; w < ROW_W; ++w) {
    tot_h += red[w][0];
    tot_m += red[w][1];
  }
  const float rh = rsqrtf(tot_h * (1.f / float(H_DIM)) + 1e-6f);
  const float rm = rsqrtf(tot_m * (1.f / float(H_DIM)) + 1e-6f);
  ActT* dst = gin + (std::size_t(g) * M + t) * (2 * H_DIM);
  *reinterpret_cast<uint2*>(dst + c) = make_uint2(act::pack2(hv.x * rh, hv.y * rh), act::pack2(hv.z * rh, hv.w * rh));
  *reinterpret_cast<uint2*>(dst + H_DIM + c) =
      make_uint2(act::pack2(mv.x * rm, mv.y * rm), act::pack2(mv.z * rm, mv.w * rm));
}

__global__ void __launch_bounds__(256) batched_gate_down_kernel(const ActT* __restrict__ gin,
                                                                const ActT* __restrict__ w_down, int M,
                                                                ActT* __restrict__ g16) {
  constexpr int RANK = 64;
  constexpr int RPC = 4;
  constexpr int CH = H_DIM / (8 * 32);  // 9
  constexpr int TM = 4;
  __shared__ __align__(16) ActT xs[TM][2 * H_DIM];
  __shared__ float part[TM][2][RPC];

  const int rank = blockIdx.x;
  const int g = blockIdx.y;
  const int m0 = blockIdx.z * TM;
  const int cm = min(TM, M - m0);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rl = warp >> 1;
  const int half = warp & 1;

  const int total_u4 = cm * (2 * H_DIM / 8);
  const ActT* gbase = gin + (std::size_t(g) * M + m0) * (2 * H_DIM);
  for (int k = tid; k < total_u4; k += 256) {
    const int m = k / (2 * H_DIM / 8);
    const int c8 = k % (2 * H_DIM / 8);
    reinterpret_cast<uint4*>(&xs[m][0])[c8] =
        __ldg(reinterpret_cast<const uint4*>(gbase + std::size_t(m) * (2 * H_DIM)) + c8);
  }
  const ActT* wr = w_down + (std::size_t(g) * RANK + rank * RPC + rl) * (2 * H_DIM) + half * H_DIM;
  uint4 wv[CH];
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    wv[i] = __ldg(reinterpret_cast<const uint4*>(wr + (i * 32 + lane) * 8));
  }
  __syncthreads();

  float acc[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) acc[m] = 0.f;
#pragma unroll
  for (int i = 0; i < CH; ++i) {
    const int col = half * H_DIM + (i * 32 + lane) * 8;
    float wf[8];
    unpack8(wv[i], wf);
    for (int m = 0; m < cm; ++m) {
      float xf[8];
      unpack8(*reinterpret_cast<const uint4*>(&xs[m][col]), xf);
      float a = acc[m];
#pragma unroll
      for (int e = 0; e < 8; ++e) a = fmaf(wf[e], xf[e], a);
      acc[m] = a;
    }
  }
  for (int m = 0; m < cm; ++m) {
    const float s = wsum(acc[m]);
    if (lane == 0) part[m][half][rl] = s;
  }
  __syncthreads();
  if (tid < cm * RPC) {
    const int m = tid / RPC;
    const int r = tid % RPC;
    g16[(std::size_t(g) * M + m0 + m) * RANK + rank * RPC + r] = act::from_float(part[m][0][r] + part[m][1][r]);
  }
}

__global__ void __launch_bounds__(256) batched_gate_fg_kernel(const float* __restrict__ h, bool h_shared,
                                                              const float* __restrict__ msg,
                                                              const ActT* __restrict__ g16,
                                                              const ActT* __restrict__ w_forget,
                                                              const ActT* __restrict__ w_gain, int M,
                                                              float* __restrict__ gated, float* __restrict__ ssall) {
  constexpr int CL = 16;
  constexpr int CPC = H_DIM / CL;  // 144
  constexpr int RANK = 64;
  constexpr int TM = 8;
  __shared__ float gs[TM][RANK];
  __shared__ float red[8][TM];

  const int rank = blockIdx.x;
  const int g = blockIdx.y;
  const int m0 = blockIdx.z * TM;
  const int cm = min(TM, M - m0);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  for (int i = tid; i < cm * RANK; i += 256) {
    const int m = i / RANK;
    const int k = i % RANK;
    gs[m][k] = act::to_float(g16[(std::size_t(g) * M + m0 + m) * RANK + k]);
  }
  __syncthreads();

  float sq[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) sq[m] = 0.f;

  if (tid < CPC) {
    const int c = rank * CPC + tid;
    const uint4* fr = reinterpret_cast<const uint4*>(w_forget + (std::size_t(g) * H_DIM + c) * RANK);
    const uint4* gr = reinterpret_cast<const uint4*>(w_gain + (std::size_t(g) * H_DIM + c) * RANK);
    float df[TM], dg[TM];
#pragma unroll
    for (int m = 0; m < TM; ++m) df[m] = dg[m] = 0.f;
#pragma unroll
    for (int k = 0; k < RANK / 8; ++k) {
      float wf[8], wg[8];
      unpack8(__ldg(fr + k), wf);
      unpack8(__ldg(gr + k), wg);
      for (int m = 0; m < cm; ++m) {
        float a_f = df[m], a_g = dg[m];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float gv = gs[m][k * 8 + e];
          a_f = fmaf(wf[e], gv, a_f);
          a_g = fmaf(wg[e], gv, a_g);
        }
        df[m] = a_f;
        dg[m] = a_g;
      }
    }
    const float* hg = h + (h_shared ? std::size_t(0) : std::size_t(g) * M * H_DIM);
    const float* mg = msg + std::size_t(g) * M * H_DIM;
    for (int m = 0; m < cm; ++m) {
      const float hv = hg[std::size_t(m0 + m) * H_DIM + c];
      const float mv = mg[std::size_t(m0 + m) * H_DIM + c];
      const float y = __fadd_rn(__fmul_rn(__fsub_rn(1.f, tanhf(df[m])), hv), __fmul_rn(tanhf(dg[m]), mv));
      gated[(std::size_t(g) * M + m0 + m) * H_DIM + c] = y;
      sq[m] = y * y;
    }
  }
  for (int m = 0; m < cm; ++m) {
    const float s = wsum(sq[m]);
    if (lane == 0) red[warp][m] = s;
  }
  __syncthreads();
  if (tid < cm) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += red[w][tid];
    ssall[(std::size_t(g) * M + m0 + tid) * CL + rank] = s;
  }
}

__global__ void __launch_bounds__(256) batched_gate_norm_kernel(const float* __restrict__ gated,
                                                                const float* __restrict__ ssall,
                                                                const float* __restrict__ norm_w, int M,
                                                                ActT* __restrict__ out_bf16) {
  constexpr int CL = 16;
  constexpr int CPC = H_DIM / CL;  // 144
  constexpr int TM = 8;

  const int rank = blockIdx.x;
  const int g = blockIdx.y;
  const int m0 = blockIdx.z * TM;
  const int cm = min(TM, M - m0);
  const int tid = threadIdx.x;
  if (tid >= CPC) return;

  const int c = rank * CPC + tid;
  const float nw = norm_w[std::size_t(g) * H_DIM + c];
  for (int m = 0; m < cm; ++m) {
    float ss = 0.f;
    const float* sp = ssall + (std::size_t(g) * M + m0 + m) * CL;
#pragma unroll
    for (int d = 0; d < CL; ++d) ss += sp[d];
    const float r = rsqrtf(ss * (1.f / float(H_DIM)) + 1e-6f);
    const std::size_t idx = (std::size_t(g) * M + m0 + m) * H_DIM + c;
    out_bf16[idx] = act::from_float(gated[idx] * r * nw);
  }
}

__global__ void batched_advance_pos_kernel(const std::int32_t* __restrict__ slot_ids, int B,
                                           std::int64_t* __restrict__ slot_pos) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const int slot = slot_ids[b];
  if (slot >= 0) slot_pos[slot] += 1;
}

template <bool FULL, bool F8, bool PROJ_FP32>
void dispatch_attn(const BatchedAttnKernelParams& kp, cudaStream_t stream) {
  if (kp.splits == 1) {
    const dim3 grid(1, HKV, kp.B);
    batched_attn_kernel<FULL, F8, PROJ_FP32, false><<<grid, NT, 0, stream>>>(kp);
  } else {
    const dim3 grid1(kp.splits, HKV, kp.B);
    batched_attn_kernel<FULL, F8, PROJ_FP32, true><<<grid1, NT, 0, stream>>>(kp);
    const dim3 grid2(1, HKV, kp.B);
    batched_attn_reduce_kernel<FULL, F8, PROJ_FP32><<<grid2, NT, 0, stream>>>(kp);
  }
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace

std::size_t attn_workspace_bytes(int max_b) {
  require(max_b >= 1, "max_b must be >= 1");
  return ws_total_bytes(max_b);
}

void batched_attn_decode(const BatchedAttnArgs& a, cudaStream_t stream) {
  require(a.B >= 1, "B must be >= 1");
  require(a.window == 0 || a.window == WIN, "window must be 0 or 1024");
  const bool full = (a.window == 0);
  require(full || !a.cache_fp8, "FP8 caches are full-layer only");
  require(a.proj && a.slot_ids && a.slot_pos && a.q_scale && a.r_scale && a.rel_proj && a.k_conv && a.v_conv &&
              a.conv_k_visit && a.conv_v_visit && a.out,
          "null argument");
  if (full) {
    require(a.k_full_pool && a.v_full_pool && a.page_table && a.max_pages_per_slot > 0 && a.max_ctx > 0,
            "full layer needs paged pool and page_table");
  } else {
    require(a.k_ring_visit && a.v_ring_visit && a.ring_pos_visit, "sliding layer needs ring pointers");
  }
  const int splits = choose_splits(a.B);
  if (splits > 1) {
    require(a.workspace && a.workspace_size >= ws_total_bytes(a.B), "workspace too small");
  }

  BatchedAttnKernelParams kp{};
  kp.proj = a.proj;
  kp.R = full ? R_FULL : R_SLIDING;
  kp.B = a.B;
  kp.splits = splits;
  kp.slot_ids = a.slot_ids;
  kp.slot_pos = a.slot_pos;
  kp.q_scale = a.q_scale;
  kp.r_scale = a.r_scale;
  kp.rel_proj = a.rel_proj;
  kp.k_conv = a.k_conv;
  kp.v_conv = a.v_conv;
  kp.conv_k_visit = a.conv_k_visit;
  kp.conv_v_visit = a.conv_v_visit;
  kp.conv_slot_stride = a.conv_slot_stride;
  kp.k_ring_visit = a.k_ring_visit;
  kp.v_ring_visit = a.v_ring_visit;
  kp.ring_pos_visit = a.ring_pos_visit;
  kp.ring_slot_stride = a.ring_slot_stride;
  kp.ring_pos_slot_stride = a.ring_pos_slot_stride;
  kp.k_full_pool = a.k_full_pool;
  kp.v_full_pool = a.v_full_pool;
  kp.page_table = a.page_table;
  kp.max_pages_per_slot = a.max_pages_per_slot;
  kp.full_visit_idx = a.full_visit_idx;
  kp.max_ctx = a.max_ctx;
  kp.log_scaling = a.log_scaling ? 1 : 0;
  if (a.workspace) {
    char* ws = static_cast<char*>(a.workspace);
    kp.part = reinterpret_cast<float*>(ws);
    kp.saved_nkv = reinterpret_cast<ActT*>(ws + ws_nkv_off(a.B));
    kp.saved_nst = reinterpret_cast<float*>(ws + ws_nst_off(a.B));
  }
  kp.out = a.out;

  if (full) {
    if (a.cache_fp8) {
      if (a.proj_fp32) dispatch_attn<true, true, true>(kp, stream);
      else dispatch_attn<true, true, false>(kp, stream);
    } else {
      if (a.proj_fp32) dispatch_attn<true, false, true>(kp, stream);
      else dispatch_attn<true, false, false>(kp, stream);
    }
  } else {
    if (a.proj_fp32) dispatch_attn<false, false, true>(kp, stream);
    else dispatch_attn<false, false, false>(kp, stream);
  }
}

void batched_static_conv_step(const void* x, bool x_fp32, int ld, int B, int C, const std::int32_t* slot_ids,
                              const float* conv_w, float* conv_state_visit, std::size_t conv_slot_stride,
                              float* out, cudaStream_t stream) {
  require(x && slot_ids && conv_w && conv_state_visit && out && B >= 1 && C >= 1 && ld >= C,
          "batched_static_conv_step: bad args");
  const dim3 grid((C + 255) / 256, B);
  if (x_fp32) {
    batched_static_conv_kernel<true><<<grid, 256, 0, stream>>>(x, ld, C, slot_ids, conv_w, conv_state_visit,
                                                               conv_slot_stride, out);
  } else {
    batched_static_conv_kernel<false><<<grid, 256, 0, stream>>>(x, ld, C, slot_ids, conv_w, conv_state_visit,
                                                                conv_slot_stride, out);
  }
  CUDA_CHECK(cudaGetLastError());
}

void scatter_full_kv(const void* src_k, const void* src_v, int scratch_max_ctx, int pos0, int T, bool kv_fp8,
                     int slot, const std::int32_t* page_table, int max_pages_per_slot, void* dst_k_pool,
                     void* dst_v_pool, cudaStream_t stream) {
  if (T <= 0) return;
  require(src_k && src_v && page_table && dst_k_pool && dst_v_pool && pos0 >= 0 && pos0 + T <= scratch_max_ctx,
          "scatter_full_kv: bad args");
  const dim3 grid(T, NUM_FULL_VISITS);
  if (kv_fp8) {
    paged_full_kv_transfer_kernel<true, true><<<grid, CONV_C / 4, 0, stream>>>(
        src_k, src_v, dst_k_pool, dst_v_pool, scratch_max_ctx, pos0, T, slot, page_table, max_pages_per_slot);
  } else {
    paged_full_kv_transfer_kernel<true, false><<<grid, CONV_C / 4, 0, stream>>>(
        src_k, src_v, dst_k_pool, dst_v_pool, scratch_max_ctx, pos0, T, slot, page_table, max_pages_per_slot);
  }
  CUDA_CHECK(cudaGetLastError());
}

void gather_full_kv(const void* src_k_pool, const void* src_v_pool, int slot, const std::int32_t* page_table,
                    int max_pages_per_slot, int n, bool kv_fp8, void* dst_k, void* dst_v, int scratch_max_ctx,
                    cudaStream_t stream) {
  if (n <= 0) return;
  require(src_k_pool && src_v_pool && page_table && dst_k && dst_v && n <= scratch_max_ctx,
          "gather_full_kv: bad args");
  const dim3 grid(n, NUM_FULL_VISITS);
  if (kv_fp8) {
    paged_full_kv_transfer_kernel<false, true><<<grid, CONV_C / 4, 0, stream>>>(
        dst_k, dst_v, const_cast<void*>(src_k_pool), const_cast<void*>(src_v_pool), scratch_max_ctx, 0, n, slot,
        page_table, max_pages_per_slot);
  } else {
    paged_full_kv_transfer_kernel<false, false><<<grid, CONV_C / 4, 0, stream>>>(
        dst_k, dst_v, const_cast<void*>(src_k_pool), const_cast<void*>(src_v_pool), scratch_max_ctx, 0, n, slot,
        page_table, max_pages_per_slot);
  }
  CUDA_CHECK(cudaGetLastError());
}

void copy_tail_kv_from_scratch(const void* src_k, const void* src_v, int scratch_max_ctx, int src_pos0,
                               int tail_tokens, bool kv_fp8, void* dst_tail_k, void* dst_tail_v, cudaStream_t stream) {
  if (tail_tokens <= 0) return;
  require(src_k && src_v && dst_tail_k && dst_tail_v && src_pos0 >= 0 && tail_tokens < kPageSize &&
              src_pos0 + tail_tokens <= scratch_max_ctx,
          "copy_tail_kv_from_scratch: bad args");
  const dim3 grid(tail_tokens, NUM_FULL_VISITS);
  if (kv_fp8) {
    tail_kv_scratch_kernel<true, true><<<grid, 256, 0, stream>>>(src_k, src_v, dst_tail_k, dst_tail_v, scratch_max_ctx,
                                                                 src_pos0, tail_tokens);
  } else {
    tail_kv_scratch_kernel<true, false><<<grid, 256, 0, stream>>>(src_k, src_v, dst_tail_k, dst_tail_v, scratch_max_ctx,
                                                                  src_pos0, tail_tokens);
  }
  CUDA_CHECK(cudaGetLastError());
}

void copy_tail_kv_to_scratch(const void* src_tail_k, const void* src_tail_v, int tail_tokens, bool kv_fp8,
                             void* dst_k, void* dst_v, int scratch_max_ctx, int dst_pos0, cudaStream_t stream) {
  if (tail_tokens <= 0) return;
  require(src_tail_k && src_tail_v && dst_k && dst_v && dst_pos0 >= 0 && tail_tokens < kPageSize &&
              dst_pos0 + tail_tokens <= scratch_max_ctx,
          "copy_tail_kv_to_scratch: bad args");
  const dim3 grid(tail_tokens, NUM_FULL_VISITS);
  if (kv_fp8) {
    tail_kv_scratch_kernel<false, true><<<grid, 256, 0, stream>>>(dst_k, dst_v, const_cast<void*>(src_tail_k),
                                                                  const_cast<void*>(src_tail_v), scratch_max_ctx,
                                                                  dst_pos0, tail_tokens);
  } else {
    tail_kv_scratch_kernel<false, false><<<grid, 256, 0, stream>>>(dst_k, dst_v, const_cast<void*>(src_tail_k),
                                                                   const_cast<void*>(src_tail_v), scratch_max_ctx,
                                                                   dst_pos0, tail_tokens);
  }
  CUDA_CHECK(cudaGetLastError());
}

void copy_tail_kv_to_pool_page(const void* src_tail_k, const void* src_tail_v, int tail_tokens, bool kv_fp8,
                               int phys_page, void* dst_k_pool, void* dst_v_pool, cudaStream_t stream) {
  if (tail_tokens <= 0) return;
  require(src_tail_k && src_tail_v && dst_k_pool && dst_v_pool && tail_tokens < kPageSize && phys_page >= 0,
          "copy_tail_kv_to_pool_page: bad args");
  const dim3 grid(tail_tokens, NUM_FULL_VISITS);
  if (kv_fp8) {
    tail_kv_to_pool_page_kernel<true><<<grid, 256, 0, stream>>>(src_tail_k, src_tail_v, tail_tokens, phys_page,
                                                                dst_k_pool, dst_v_pool);
  } else {
    tail_kv_to_pool_page_kernel<false><<<grid, 256, 0, stream>>>(src_tail_k, src_tail_v, tail_tokens, phys_page,
                                                                 dst_k_pool, dst_v_pool);
  }
  CUDA_CHECK(cudaGetLastError());
}

void batched_head_softcap(const ActT* in, int B, int chunk_V, int v_offset, int total_V, float* out,
                          cudaStream_t stream) {
  require(in && out && B >= 1 && chunk_V >= 1 && v_offset >= 0 && v_offset + chunk_V <= total_V,
          "batched_head_softcap: bad args");
  const dim3 grid((chunk_V + 255) / 256, B);
  batched_head_softcap_kernel<<<grid, 256, 0, stream>>>(in, B, chunk_V, v_offset, total_V, out);
  CUDA_CHECK(cudaGetLastError());
}

void dequant_e4m3_unscaled(const void* q, int rows, int K, ActT* out, cudaStream_t stream) {
  require(q && out && rows >= 0 && K > 0 && K % 16 == 0, "dequant_e4m3_unscaled: bad args");
  const long long total16 = static_cast<long long>(rows) * K / 16;
  if (total16 > 0) {
    dequant_e4m3_unscaled_kernel<<<static_cast<unsigned>((total16 + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::uint8_t*>(q), total16, out);
    CUDA_CHECK(cudaGetLastError());
  }
}

void batched_head_scale_softcap(const float* in, const float* scale, int M, int chunk_V, int v_offset, int total_V,
                                float* out, cudaStream_t stream) {
  require(in && out && M >= 1 && chunk_V >= 1 && v_offset >= 0 && v_offset + chunk_V <= total_V,
          "batched_head_scale_softcap: bad args");
  const dim3 grid((chunk_V + 255) / 256, M);
  batched_head_scale_softcap_kernel<<<grid, 256, 0, stream>>>(in, scale, M, chunk_V, v_offset, total_V, out);
  CUDA_CHECK(cudaGetLastError());
}

void batched_router_logits(const ActT* x, const ActT* w, int M, float* out, cudaStream_t stream) {
  require(x && w && out && M >= 1, "batched_router_logits: bad args");
  const dim3 grid(16, (M + 7) / 8);
  batched_router_logits_kernel<<<grid, 256, 0, stream>>>(x, w, M, out);
  CUDA_CHECK(cudaGetLastError());
}

void batched_message_gates(const float* h, bool h_shared, const float* msg, int G, int M, const ActT* w_down,
                           const ActT* w_forget, const ActT* w_gain, const float* norm_w, ActT* gin, ActT* g16,
                           float* ssall, ActT* out_bf16, cudaStream_t stream) {
  require(h && msg && w_down && w_forget && w_gain && norm_w && gin && g16 && ssall && out_bf16 && G >= 1 && G <= 3 &&
              M >= 1,
          "batched_message_gates: bad args");
  batched_gate_inputs_kernel<<<dim3(M, G), H_DIM / 4, 0, stream>>>(h, h_shared, msg, M, gin);
  CUDA_CHECK(cudaGetLastError());
  batched_gate_down_kernel<<<dim3(16, G, (M + 3) / 4), 256, 0, stream>>>(gin, w_down, M, g16);
  CUDA_CHECK(cudaGetLastError());
  float* gated = reinterpret_cast<float*>(gin);
  batched_gate_fg_kernel<<<dim3(16, G, (M + 7) / 8), 256, 0, stream>>>(h, h_shared, msg, g16, w_forget, w_gain, M,
                                                                       gated, ssall);
  CUDA_CHECK(cudaGetLastError());
  batched_gate_norm_kernel<<<dim3(16, G, (M + 7) / 8), 256, 0, stream>>>(gated, ssall, norm_w, M, out_bf16);
  CUDA_CHECK(cudaGetLastError());
}

void batched_advance_pos(const std::int32_t* slot_ids, int B, std::int64_t* slot_pos, cudaStream_t stream) {
  if (B <= 0) return;
  require(slot_ids && slot_pos, "batched_advance_pos: null arg");
  batched_advance_pos_kernel<<<(B + 63) / 64, 64, 0, stream>>>(slot_ids, B, slot_pos);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::batch

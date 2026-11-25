// Batched multi-sequence decode kernels for Inkling-2-Mini continuous batching (B = 1..64+ slots).
// Exact arithmetic parity with kernels/attn/attn.cu (M=1) and kernels/prefill/rows.cu.
#include "kernels/batch/batch.h"

#include "core/act.h"
#include "core/device.h"
#include "kernels/common/l2.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

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
constexpr int H_DIM = 2304;
constexpr int SINK_PAD = (HQ + 7) & ~7;
constexpr int NUM_VISITS = 48, NUM_SLIDING_VISITS = 40, NUM_FULL_VISITS = 8;
constexpr int OFF_GATE = HQ * D, OFF_REL = 2 * HQ * D, OFF_SINK = OFF_REL + HQ * DREL,
              OFF_K = OFF_SINK + SINK_PAD, OFF_V = OFF_K + CONV_C, OFF_KDYN = OFF_V + CONV_C,
              OFF_VDYN = OFF_KDYN + HKV * 4;
constexpr int R_SLIDING = OFF_KDYN, R_FULL = OFF_VDYN + HKV * 4;
constexpr int NT = 256;
constexpr int KW = 8;  // 8 warps per block
constexpr int U = 2;   // key pairs per warp per register buffer
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

constexpr std::size_t align256(std::size_t n) { return (n + 255) & ~std::size_t(255); }

// Workspace layout for up to max_b sequences (splits * B <= max(16, max_b)):
//   part:      fp32 [max_total_splits, HKV, GQA, PSTR]
//   saved_nkv: ActT [max_b, HKV, 2, D]
//   saved_nst: fp32 [max_b, HKV, 3, 2, D]
std::size_t ws_nkv_off(int max_b) {
  const int max_ts = std::max(16 * INGINE_TP_SIZE + 16, max_b);
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
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
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3), __NV_E4M3)));
}

__device__ __forceinline__ uint32_t pack_e4m3x2(float a, float b) {
  return __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ uint4 ld_kv(const void* ptr, std::uint64_t pol) {
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
}

__device__ __forceinline__ uint2 ld_kv8(const void* ptr, std::uint64_t pol) {
  uint2 v;
  if constexpr (l2::hinted<KV_L2>)
    asm volatile("ld.global" INGINE_L1_NO_ALLOCATE ".L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
                 : "=r"(v.x), "=r"(v.y)
                 : "l"(ptr), "l"(pol)
                 : "memory");
  else
    asm volatile("ld.global" INGINE_L1_NO_ALLOCATE ".v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(ptr) : "memory");
  return v;
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
  using CT = std::conditional_t<F8, __nv_fp8_storage_t, ActT>;
  using KV = std::conditional_t<F8, uint2, uint4>;

  const int rank = blockIdx.x;
  const int g = blockIdx.y;
  const int b = blockIdx.z;
  const int slot = p.slot_ids[b];
  const int tid = threadIdx.x;
  if (slot < 0) {
    if constexpr (!SPLIT) {
      for (int idx = tid; idx < GQA * D; idx += NT) {
        const int row = idx / D, dim = idx % D;
        p.out[std::size_t(b) * HQ * D + (g * GQA + row) * D + dim] = act::from_float(0.f);
      }
    }
    return;
  }

  const int p0 = int(p.slot_pos[slot]);
  if (FULL && p0 >= p.max_ctx) return;

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
  const int old_end = FULL ? p0 : (p0 < WIN ? p0 : WIN);
  constexpr int STEP = KW * 2 * U;
  constexpr int TILE = 2 * STEP;
  const int stride = p.splits * TILE;
  const int c0 = rank * TILE;
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
  if (c0 < old_end) load_buf(A, c0);
  if (c0 + STEP < old_end) load_buf(B, c0 + STEP);

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

  for (int base = c0; base < old_end; base += stride) {
    compute(A, false);
    if (base + stride < old_end) load_buf(A, base + stride);
    if (base + STEP < old_end) {
      compute(B, false);
      if (base + stride + STEP < old_end) load_buf(B, base + stride + STEP);
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
    // ---- Non-split single-launch finish: fold sink + sigmoid(gate), write out, commit KV & conv state!
    for (int idx = tid; idx < GQA * D; idx += NT) {
      const int row = idx / D, dim = idx % D;
      const float sink = sm.sink[row] * LOG2E;
      float mx = sink;
#pragma unroll
      for (int k = 0; k < KW; ++k) mx = fmaxf(mx, sm.u.red.ml[(k * GQA + row) * 2]);
      float l = exp2f(sink - mx), a = 0.f;
#pragma unroll
      for (int k = 0; k < KW; ++k) {
        const int w = k * GQA + row;
        const float f = exp2f(sm.u.red.ml[w * 2] - mx);
        l += sm.u.red.ml[w * 2 + 1] * f;
        a += sm.u.red.o[w * D + dim] * f;
      }
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
  using CT = std::conditional_t<F8, __nv_fp8_storage_t, ActT>;

  const int g = blockIdx.y;
  const int b = blockIdx.z;
  const int slot = p.slot_ids[b];
  const int tid = threadIdx.x;
  if (slot < 0) {
    for (int idx = tid; idx < GQA * D; idx += NT) {
      const int row = idx / D, dim = idx % D;
      p.out[std::size_t(b) * HQ * D + (g * GQA + row) * D + dim] = act::from_float(0.f);
    }
    return;
  }

  const int p0 = int(p.slot_pos[slot]);
  if (FULL && p0 >= p.max_ctx) return;

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
  if (slot < 0) {
    out[std::size_t(b) * C + c] = 0.f;
    return;
  }

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

void batched_advance_pos(const std::int32_t* slot_ids, int B, std::int64_t* slot_pos, cudaStream_t stream) {
  if (B <= 0) return;
  require(slot_ids && slot_pos, "batched_advance_pos: null arg");
  batched_advance_pos_kernel<<<(B + 63) / 64, 64, 0, stream>>>(slot_ids, B, slot_pos);
  CUDA_CHECK(cudaGetLastError());
}

// ==========================================================================================================
// Batched multi-token speculative verification & rollback (W in {1, 2, 4, 8})
// ==========================================================================================================

namespace {

int choose_verify_splits(int B) {
  if (B >= 16) return 1;
  int s = 16 / B;
  if (s < 1) s = 1;
  if (s > 16) s = 16;
  return s;
}

std::size_t verify_ws_nkv_off(int /*max_b*/, int max_w) {
  return align256(std::size_t(16) * HKV * std::size_t(max_w) * GQA * PSTR * sizeof(float));
}
std::size_t verify_ws_nst_off(int max_b, int max_w) {
  return verify_ws_nkv_off(max_b, max_w) +
         align256(std::size_t(max_b) * HKV * 2 * std::size_t(max_w) * D * sizeof(ActT));
}
std::size_t verify_ws_total_bytes(int max_b, int max_w) {
  return verify_ws_nst_off(max_b, max_w) + align256(std::size_t(max_b) * HKV * 3 * 2 * D * sizeof(float));
}

struct BatchedVerifyAttnKernelParams {
  const void* proj;
  int R;
  int B;
  int W;
  int splits;
  const std::int32_t* slot_ids;
  const std::int32_t* slot_w_limit;
  const std::int64_t* slot_pos;
  const float* q_scale;
  const float* r_scale;
  const float* rel_proj;
  const float* k_conv;
  const float* v_conv;
  float* conv_k_visit;
  float* conv_v_visit;
  std::size_t conv_slot_stride;
  float* hist_k_visit;
  float* hist_v_visit;
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

constexpr int V_KT = 16;  // cached keys per shared-memory tile (amortized Wx across all W tokens of slot b)

template <int W, bool F8>
struct VerifyAttnSmem {
  using CT = std::conditional_t<F8, __nv_fp8_storage_t, ActT>;
  __align__(16) ActT q[W][GQA][D];
  float rel[W][GQA][DREL];
  float sink[W][GQA];
  __align__(16) ActT nkv[2][W][D];
  float nst[3 * 2 * D];
  float relp0[DREL][8];
  union {
    struct {
      float qs[GQA * D];
      float rs[GQA * DREL];
      float st[3 * 2 * D];
      float w[2 * D * 4];
      float dyn[W * 2 * 4];
      float nkf[2 * D];
    } stage;
    struct {
      __align__(16) CT k[V_KT][D];
      __align__(16) CT v[V_KT][D];
      int kp[V_KT];
    } tile;
    struct {
      float o[KW * GQA * D];
      float ml[KW * GQA * 2];
    } red;
  } u;
};

static_assert(sizeof(VerifyAttnSmem<8, false>) <= 48 * 1024, "VerifyAttnSmem<8, BF16> must fit 48 KiB SMEM");
static_assert(sizeof(VerifyAttnSmem<8, true>) <= 48 * 1024, "VerifyAttnSmem<8, FP8> must fit 48 KiB SMEM");

template <int W, bool FULL, bool F8, bool PROJ_FP32, bool SPLIT>
__global__ void __launch_bounds__(NT, 1) batched_attn_verify_kernel(const BatchedVerifyAttnKernelParams p) {
  static_assert(W == 1 || W == 2 || W == 4 || W == 8, "W must be 1, 2, 4, or 8");
  static_assert(FULL || !F8, "FP8 caches are full-layer only");
  using CT = std::conditional_t<F8, __nv_fp8_storage_t, ActT>;
  using KV = std::conditional_t<F8, uint2, uint4>;

  const int rank = blockIdx.x;
  const int g = blockIdx.y;
  const int b = blockIdx.z;
  const int slot = p.slot_ids[b];
  const int tid = threadIdx.x;
  if (slot < 0) {
    if constexpr (!SPLIT) {
      for (int idx = tid; idx < W * GQA * D; idx += NT) {
        const int w = idx / (GQA * D);
        const int rem = idx % (GQA * D);
        const int row = rem / D, dim = rem % D;
        p.out[std::size_t(b * W + w) * HQ * D + (g * GQA + row) * D + dim] = act::from_float(0.f);
      }
    }
    return;
  }

  const int w_lim = p.slot_w_limit ? min(W, max(1, int(p.slot_w_limit[b]))) : W;
  const int p0 = int(p.slot_pos[slot]);
  if (FULL && p0 + w_lim > p.max_ctx) return;

  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int half = lane >> 4;
  const int c = lane & 15;
  const bool owner = (rank == 0);
  const int R = p.R;

  constexpr int KW_PER_TOK = KW / W;  // 8, 4, 2, 1 warps per trial token
  const int tok = warp / KW_PER_TOK;  // 0..W-1
  const int kw = warp % KW_PER_TOK;   // 0..KW_PER_TOK-1

  __shared__ VerifyAttnSmem<W, F8> sm;

  float* const conv_k_slot = p.conv_k_visit + std::size_t(slot) * p.conv_slot_stride;
  float* const conv_v_slot = p.conv_v_visit + std::size_t(slot) * p.conv_slot_stride;
  ActT* const k_ring_slot = FULL ? nullptr : (p.k_ring_visit + std::size_t(slot) * p.ring_slot_stride);
  ActT* const v_ring_slot = FULL ? nullptr : (p.v_ring_visit + std::size_t(slot) * p.ring_slot_stride);
  std::int64_t* const kpos_slot = FULL ? nullptr : (p.ring_pos_visit + std::size_t(slot) * p.ring_pos_slot_stride);
  const std::int32_t* const pt_slot = FULL ? (p.page_table + std::size_t(slot) * p.max_pages_per_slot) : nullptr;
  CT* const k_pool = FULL ? static_cast<CT*>(p.k_full_pool) : reinterpret_cast<CT*>(k_ring_slot);
  CT* const v_pool = FULL ? static_cast<CT*>(p.v_full_pool) : reinterpret_cast<CT*>(v_ring_slot);

  // ---- Stage shared scales, sink, and owner conv weights/state/relp0.
  for (int i = tid; i < GQA * D; i += NT) sm.u.stage.qs[i] = p.q_scale[g * GQA * D + i];
  for (int i = tid; i < GQA * DREL; i += NT) sm.u.stage.rs[i] = p.r_scale[g * GQA * DREL + i];
  for (int i = tid; i < W * GQA; i += NT) {
    const int w = i / GQA, row = i % GQA;
    sm.sink[w][row] = (w < w_lim) ? load_proj_val<PROJ_FP32>(p.proj, std::size_t(b * W + w) * R + OFF_SINK + g * GQA + row)
                                  : 0.f;
  }
  if (owner) {
    for (int i = tid; i < 3 * 2 * D; i += NT) {
      const int j = i / (2 * D), kv = (i / D) & 1, ch = i % D;
      sm.u.stage.st[i] = (kv ? conv_v_slot : conv_k_slot)[j * CONV_C + g * D + ch];
    }
    for (int i = tid; i < 2 * D * 4; i += NT) {
      const int kv = i / (D * 4), q = i % (D * 4);
      const float* wptr = kv ? p.v_conv : p.k_conv;
      if constexpr (FULL) {
        const int lag = q / D, ch = q % D;
        sm.u.stage.w[i] = wptr[lag * CONV_C + g * D + ch];
      } else {
        const int ch = q / 4, tap = q & 3;
        sm.u.stage.w[i] = wptr[(g * D + ch) * 4 + tap];
      }
    }
    if constexpr (FULL) {
      for (int i = tid; i < W * 8; i += NT) {
        const int w = i / 8, rem = i & 7, kv = rem >> 2, lag = rem & 3;
        sm.u.stage.dyn[i] = (w < w_lim)
                                ? load_proj_val<PROJ_FP32>(
                                      p.proj, std::size_t(b * W + w) * R + (kv ? OFF_VDYN : OFF_KDYN) + g * 4 + lag)
                                : 0.f;
      }
    }
    for (int i = tid; i < DREL * 8; i += NT) {
      const int c2 = i >> 3, d0 = i & 7;
      sm.relp0[c2][d0] = p.rel_proj[c2 * EXT + d0];
    }
  }
  __syncthreads();

  // ---- Q and rel post-processing for all w < w_lim (bit-exact with batched_attn_decode).
  // One warp processes one (w, head) pair at a time across (W * GQA) total rows.
  for (int row_id = warp; row_id < W * GQA; row_id += KW) {
    const int w = row_id / GQA;
    const int h = row_id % GQA;
    if (w < w_lim) {
      const std::size_t pb = std::size_t(b * W + w) * R;
      const float tau = p.log_scaling ? 1.f + 0.1f * logf(fmaxf(float(p0 + w + 1) * (1.f / 8192.f), 1.f)) : 1.f;
      const float4 s = reinterpret_cast<const float4*>(sm.u.stage.qs + h * D)[lane];
      float x[4];
      x[0] = load_proj_val<PROJ_FP32>(p.proj, pb + (g * GQA + h) * D + 4 * lane + 0) * s.x;
      x[1] = load_proj_val<PROJ_FP32>(p.proj, pb + (g * GQA + h) * D + 4 * lane + 1) * s.y;
      x[2] = load_proj_val<PROJ_FP32>(p.proj, pb + (g * GQA + h) * D + 4 * lane + 2) * s.z;
      x[3] = load_proj_val<PROJ_FP32>(p.proj, pb + (g * GQA + h) * D + 4 * lane + 3) * s.w;
      const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        x[e] = bf16r(x[e] * f);
        if (p.log_scaling) x[e] = bf16r(x[e] * tau);
      }
      reinterpret_cast<uint2*>(&sm.q[w][h][0])[lane] = make_uint2(act::pack2(x[0], x[1]), act::pack2(x[2], x[3]));
    }
  }
  // Rel: one half-warp per (w, h) row. Loop trip count MUST be warp-uniform so hsum16(0xffffffffu) never diverges!
  constexpr int REL_ITERS = (W * GQA + KW * 2 - 1) / (KW * 2);
  for (int iter = 0; iter < REL_ITERS; ++iter) {
    const int row_id = iter * (KW * 2) + warp * 2 + half;
    const int w = row_id / GQA;
    const int h = row_id % GQA;
    const bool ok = (row_id < W * GQA) && (w < w_lim);
    float r = 0.f;
    float tau = 1.f;
    if (ok) {
      const std::size_t pb = std::size_t(b * W + w) * R;
      tau = p.log_scaling ? 1.f + 0.1f * logf(fmaxf(float(p0 + w + 1) * (1.f / 8192.f), 1.f)) : 1.f;
      r = load_proj_val<PROJ_FP32>(p.proj, pb + OFF_REL + (g * GQA + h) * DREL + c) * sm.u.stage.rs[h * DREL + c];
    }
    const float ss = hsum16(r * r);
    if (ok) {
      float y = bf16r(r * clip_factor(ss, CLIP_REL));
      if (p.log_scaling) y = bf16r(y * tau);
      sm.rel[w][h][c] = y;
    }
  }

  // ---- Owner: sequential w_lim-step K/V conv + key clip / value RMS norm + optional E4M3 round + hist_k/v.
  if (owner) {
    for (int w = 0; w < w_lim; ++w) {
      const std::size_t pb = std::size_t(b * W + w) * R;
      for (int idx = tid; idx < 2 * D; idx += NT) {
        const int kv = idx / D, ch = idx % D;
        const float e0 = sm.u.stage.st[(0 * 2 + kv) * D + ch];
        const float e1 = sm.u.stage.st[(1 * 2 + kv) * D + ch];
        const float e2 = sm.u.stage.st[(2 * 2 + kv) * D + ch];
        const float e3 = bf16r(load_proj_val<PROJ_FP32>(p.proj, pb + (kv ? OFF_V : OFF_K) + g * D + ch));
        if (kv == 0 && p.hist_k_visit) {
          p.hist_k_visit[std::size_t(b * W + w) * CONV_C + g * D + ch] = e3;
        } else if (kv == 1 && p.hist_v_visit) {
          p.hist_v_visit[std::size_t(b * W + w) * CONV_C + g * D + ch] = e3;
        }
        float y;
        if constexpr (FULL) {
          float o = 0.f;
          const float e_arr[4] = {e0, e1, e2, e3};
#pragma unroll
          for (int lag = 0; lag < 4; ++lag) {
            const float wval = __fadd_rn(sm.u.stage.w[(kv * 4 + lag) * D + ch], sm.u.stage.dyn[w * 8 + kv * 4 + lag]);
            o = __fadd_rn(o, __fmul_rn(wval, e_arr[3 - lag]));
          }
          y = __fmul_rn(__fadd_rn(o, e3), INV_3);
        } else {
          const float* wptr = sm.u.stage.w + (kv * D + ch) * 4;
          float o = __fmul_rn(wptr[0], e0);
          o = __fadd_rn(o, __fmul_rn(wptr[1], e1));
          o = __fadd_rn(o, __fmul_rn(wptr[2], e2));
          o = __fadd_rn(o, __fmul_rn(wptr[3], e3));
          y = __fmul_rn(__fadd_rn(o, e3), INV_SQRT5);
        }
        sm.u.stage.nkf[kv * D + ch] = bf16r(y);
        sm.u.stage.st[(0 * 2 + kv) * D + ch] = e1;
        sm.u.stage.st[(1 * 2 + kv) * D + ch] = e2;
        sm.u.stage.st[(2 * 2 + kv) * D + ch] = e3;
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
        reinterpret_cast<uint2*>(&sm.nkv[kv][w][0])[lane] = make_uint2(act::pack2(y[0], y[1]), act::pack2(y[2], y[3]));
      }
      __syncthreads();
    }
    for (int i = tid; i < 3 * 2 * D; i += NT) sm.nst[i] = sm.u.stage.st[i];
  }
  __syncthreads();  // stage inputs dead; sm.u.tile may now be used

  // ---- Load this warp's token `tok` queries into registers (30 floats total -> zero spills!).
  float qv[GQA][8], relv[GQA];
  if (tok < w_lim) {
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
      float qf[8];
      unpack8(*reinterpret_cast<const uint4*>(&sm.q[tok][i][c * 8]), qf);
#pragma unroll
      for (int e = 0; e < 8; ++e) qv[i][e] = qf[e] * QSCALE;
      relv[i] = sm.rel[tok][i][c];
    }
  } else {
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
#pragma unroll
      for (int e = 0; e < 8; ++e) qv[i][e] = 0.f;
      relv[i] = 0.f;
    }
  }

  float mrun[GQA], lrun[GQA], acc[GQA][8];
#pragma unroll
  for (int i = 0; i < GQA; ++i) {
    mrun[i] = NEG_INIT;
    lrun[i] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  }

  const int P_tok = p0 + tok;

  // ---- Owner attends causal self-keys j in [0..tok] from sm.nkv (warp-uniform trip count).
  if (owner && kw == 0 && tok < w_lim) {
    constexpr int SELF_STEPS = (W + 1) / 2;
#pragma unroll
    for (int step = 0; step < SELF_STEPS; ++step) {
      const int j = 2 * step + half;
      const bool ok = (j <= tok);
      const int d = tok - j;
      float kf[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      float vf[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      if (ok) {
        unpack8(*reinterpret_cast<const uint4*>(&sm.nkv[0][j][c * 8]), kf);
        unpack8(*reinterpret_cast<const uint4*>(&sm.nkv[1][j][c * 8]), vf);
      }
      const float rp = ok ? sm.relp0[c][d] : 0.f;
#pragma unroll
      for (int i = 0; i < GQA; ++i) {
        float t = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) t = fmaf(qv[i][e], kf[e], t);
        float s = hsum16(t);
        const float bias = hsum16(relv[i] * rp);
        if (ok) s = fmaf(bf16r(bias), LOG2E, s);
        else s = -INFINITY;
        const float mx = fmaxf(mrun[i], s);
        const float sc = exp2f(mrun[i] - mx);
        mrun[i] = mx;
        lrun[i] *= sc;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][e] *= sc;
        const float pu = exp2f(s - mrun[i]);
        lrun[i] += pu;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(pu, vf[e], acc[i][e]);
      }
    }
  }

  // ---- Stream cached keys [0, old_end) in V_KT=16-key SMEM tiles, 1x global load shared across all W tokens!
  const int old_end = FULL ? p0 : (p0 < WIN ? p0 : WIN);
  const int stride = p.splits * V_KT;
  const int c0 = rank * V_KT;
  const std::uint64_t kvpol = l2::policy<KV_L2>();

  constexpr int KEYS_PER_WARP = V_KT / KW_PER_TOK;  // 2, 4, 8, 16

  for (int base = c0; base < old_end; base += stride) {
    // Cooperative tile load: 512 vector loads (2 per thread for NT=256).
    for (int idx = tid; idx < V_KT * 2 * (D / 8); idx += NT) {
      const int k_idx = idx / (2 * (D / 8));
      const int kv = (idx / (D / 8)) & 1;
      const int c8 = idx & ((D / 8) - 1);
      const int j = base + k_idx;
      if (j < old_end) {
        std::size_t off;
        if constexpr (FULL) {
          const int phys = pt_slot[j >> kPageShift];
          off = ((((std::size_t(phys) * NUM_FULL_VISITS + p.full_visit_idx) * kPageSize + (j & kPageMask)) * HKV + g) * D) +
                c8 * 8;
        } else {
          off = (std::size_t(j) * HKV + g) * D + c8 * 8;
        }
        const CT* src = (kv ? v_pool : k_pool) + off;
        if constexpr (F8) {
          const uint2 val = ld_kv8(src, kvpol);
          reinterpret_cast<uint2*>((kv ? &sm.u.tile.v[k_idx][0] : &sm.u.tile.k[k_idx][0]))[c8] = val;
        } else {
          const uint4 val = ld_kv(src, kvpol);
          reinterpret_cast<uint4*>((kv ? &sm.u.tile.v[k_idx][0] : &sm.u.tile.k[k_idx][0]))[c8] = val;
        }
      } else {
        if constexpr (F8) {
          reinterpret_cast<uint2*>((kv ? &sm.u.tile.v[k_idx][0] : &sm.u.tile.k[k_idx][0]))[c8] = make_uint2(0u, 0u);
        } else {
          reinterpret_cast<uint4*>((kv ? &sm.u.tile.v[k_idx][0] : &sm.u.tile.k[k_idx][0]))[c8] =
              make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
    if (tid < V_KT) {
      const int j = base + tid;
      sm.u.tile.kp[tid] = (j < old_end) ? (FULL ? j : int(kpos_slot[j])) : -1;
    }
    __syncthreads();

    if (tok < w_lim) {
#pragma unroll
      for (int step = 0; step < KEYS_PER_WARP / 2; ++step) {
        const int k_idx = kw * KEYS_PER_WARP + 2 * step + half;
        const int kp = sm.u.tile.kp[k_idx];
        const int d = P_tok - kp;
        const bool ok = (kp >= 0 && kp < p0 && d >= 0 && (FULL || d < WIN));
        const bool need = (ok && d < EXT);
        float kf[8], vf[8];
        if constexpr (F8) {
          unpack8(reinterpret_cast<const uint2*>(&sm.u.tile.k[k_idx][0])[c], kf);
        } else {
          unpack8(reinterpret_cast<const uint4*>(&sm.u.tile.k[k_idx][0])[c], kf);
        }
        float s[GQA];
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
          float t = 0.f;
#pragma unroll
          for (int e = 0; e < 8; ++e) t = fmaf(qv[i][e], kf[e], t);
          s[i] = hsum16(t);
        }
        if (__any_sync(0xffffffffu, need)) {
          const float rp = need ? __ldg(p.rel_proj + c * EXT + d) : 0.f;
#pragma unroll
          for (int i = 0; i < GQA; ++i) {
            const float bias = hsum16(relv[i] * rp);
            if (need) s[i] = fmaf(bf16r(bias), LOG2E, s[i]);
          }
        }
        if (ok) {
          if constexpr (F8) {
            unpack8(reinterpret_cast<const uint2*>(&sm.u.tile.v[k_idx][0])[c], vf);
          } else {
            unpack8(reinterpret_cast<const uint4*>(&sm.u.tile.v[k_idx][0])[c], vf);
          }
        } else {
#pragma unroll
          for (int e = 0; e < 8; ++e) vf[e] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
          const float sv = ok ? s[i] : -INFINITY;
          const float mx = fmaxf(mrun[i], sv);
          const float sc = exp2f(mrun[i] - mx);
          mrun[i] = mx;
          lrun[i] *= sc;
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][e] *= sc;
          const float pu = exp2f(sv - mrun[i]);
          lrun[i] += pu;
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(pu, vf[e], acc[i][e]);
        }
      }
    }
    __syncthreads();
  }

  // ---- Merge the two half-warps in registers, then write per-warp partials to sm.u.red.
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
  __syncthreads();  // tile dead; alias sm.u.red
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
    // Stage 1 (READ-ONLY on KV/ring_pos/conv): write split partials + owner saves nkv/nst.
    float* part_base = p.part + std::size_t(((b * p.splits + rank) * HKV + g) * W) * GQA * PSTR;
    for (int idx = tid; idx < w_lim * GQA * (D / 4); idx += NT) {
      const int w = idx / (GQA * (D / 4));
      const int rem = idx % (GQA * (D / 4));
      const int row = rem / (D / 4), d4 = rem % (D / 4);
      const int w0 = w * KW_PER_TOK;
      float mx = NEG_INIT;
#pragma unroll
      for (int k = 0; k < KW_PER_TOK; ++k) mx = fmaxf(mx, sm.u.red.ml[((w0 + k) * GQA + row) * 2]);
      float l = 0.f;
      float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int k = 0; k < KW_PER_TOK; ++k) {
        const int wi = (w0 + k) * GQA + row;
        const float f = exp2f(sm.u.red.ml[wi * 2] - mx);
        l += sm.u.red.ml[wi * 2 + 1] * f;
        const float4 a = reinterpret_cast<const float4*>(sm.u.red.o + wi * D)[d4];
        o.x += a.x * f, o.y += a.y * f, o.z += a.z * f, o.w += a.w * f;
      }
      float* pr = part_base + std::size_t(w * GQA + row) * PSTR;
      reinterpret_cast<float4*>(pr)[d4] = o;
      if (d4 == 0) *reinterpret_cast<float2*>(pr + D) = make_float2(mx, l);
    }
    if (owner) {
      ActT* dst_nkv = p.saved_nkv + std::size_t(b * HKV + g) * 2 * W * D;
      for (int idx = tid; idx < 2 * w_lim * 16; idx += NT) {
        const int kv = idx / (w_lim * 16);
        const int rem = idx % (w_lim * 16);
        const int w = rem / 16, c8 = rem & 15;
        reinterpret_cast<uint4*>(dst_nkv + (kv * W + w) * D)[c8] =
            reinterpret_cast<const uint4*>(&sm.nkv[kv][w][0])[c8];
      }
      float* dst_nst = p.saved_nst + std::size_t(b * HKV + g) * 3 * 2 * D;
      for (int idx = tid; idx < 3 * 2 * D; idx += NT) dst_nst[idx] = sm.nst[idx];
    }
  } else {
    // Non-split finish: write out, commit per-head K/V and conv state.
    // Note: for !FULL (sliding), ring_pos is committed in a separate post-grid kernel so no head can race on kpos!
    for (int idx = tid; idx < w_lim * GQA * D; idx += NT) {
      const int w = idx / (GQA * D);
      const int rem = idx % (GQA * D);
      const int row = rem / D, dim = rem % D;
      const int w0 = w * KW_PER_TOK;
      const float sink = sm.sink[w][row] * LOG2E;
      float mx = sink;
#pragma unroll
      for (int k = 0; k < KW_PER_TOK; ++k) mx = fmaxf(mx, sm.u.red.ml[((w0 + k) * GQA + row) * 2]);
      float l = exp2f(sink - mx), a = 0.f;
#pragma unroll
      for (int k = 0; k < KW_PER_TOK; ++k) {
        const int wi = (w0 + k) * GQA + row;
        const float f = exp2f(sm.u.red.ml[wi * 2] - mx);
        l += sm.u.red.ml[wi * 2 + 1] * f;
        a += sm.u.red.o[wi * D + dim] * f;
      }
      const std::size_t pb = std::size_t(b * W + w) * R;
      const float gate_in = load_proj_val<PROJ_FP32>(p.proj, pb + OFF_GATE + (g * GQA + row) * D + dim);
      const float gate = 1.f / (1.f + expf(-gate_in));
      p.out[std::size_t(b * W + w) * HQ * D + (g * GQA + row) * D + dim] = act::from_float(bf16r(a / l) * gate);
    }
    for (int idx = tid; idx < (W - w_lim) * GQA * D; idx += NT) {
      const int w = w_lim + idx / (GQA * D);
      const int rem = idx % (GQA * D);
      const int row = rem / D, dim = rem % D;
      p.out[std::size_t(b * W + w) * HQ * D + (g * GQA + row) * D + dim] = act::from_float(0.f);
    }

    // Commit per-head new K/V rows (safe: this CTA is the sole reader & writer of head g for slot `slot`).
    for (int idx = tid; idx < 2 * w_lim * 16; idx += NT) {
      const int kv = idx / (w_lim * 16);
      const int rem = idx % (w_lim * 16);
      const int w = rem / 16, c8 = rem & 15;
      const uint4 u = reinterpret_cast<const uint4*>(&sm.nkv[kv][w][0])[c8];
      const int pw = p0 + w;
      std::size_t dst_off;
      if constexpr (FULL) {
        const int phys = pt_slot[pw >> kPageShift];
        dst_off = ((((std::size_t(phys) * NUM_FULL_VISITS + p.full_visit_idx) * kPageSize + (pw & kPageMask)) * HKV + g) * D);
      } else {
        dst_off = (std::size_t(pw & (WIN - 1)) * HKV + g) * D;
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
  }
}

template <int W, bool FULL, bool F8, bool PROJ_FP32>
__global__ void __launch_bounds__(NT) batched_attn_verify_reduce_kernel(const BatchedVerifyAttnKernelParams p) {
  static_assert(FULL || !F8, "FP8 caches are full-layer only");
  using CT = std::conditional_t<F8, __nv_fp8_storage_t, ActT>;

  const int g = blockIdx.y;
  const int b = blockIdx.z;
  const int slot = p.slot_ids[b];
  const int tid = threadIdx.x;
  if (slot < 0) {
    for (int idx = tid; idx < W * GQA * D; idx += NT) {
      const int w = idx / (GQA * D);
      const int rem = idx % (GQA * D);
      const int row = rem / D, dim = rem % D;
      p.out[std::size_t(b * W + w) * HQ * D + (g * GQA + row) * D + dim] = act::from_float(0.f);
    }
    return;
  }

  const int w_lim = p.slot_w_limit ? min(W, max(1, int(p.slot_w_limit[b]))) : W;
  const int p0 = int(p.slot_pos[slot]);
  if (FULL && p0 + w_lim > p.max_ctx) return;

  const int R = p.R;

  for (int idx = tid; idx < w_lim * GQA * D; idx += NT) {
    const int w = idx / (GQA * D);
    const int rem = idx % (GQA * D);
    const int row = rem / D, dim = rem % D;
    const std::size_t pb = std::size_t(b * W + w) * R;
    const float sink = load_proj_val<PROJ_FP32>(p.proj, pb + OFF_SINK + g * GQA + row) * LOG2E;
    float mx = sink;
    for (int q = 0; q < p.splits; ++q) {
      const float* pr = p.part + std::size_t((((b * p.splits + q) * HKV + g) * W + w) * GQA + row) * PSTR;
      const float2 ml = *reinterpret_cast<const float2*>(pr + D);
      mx = fmaxf(mx, ml.x);
    }
    float l = exp2f(sink - mx), a = 0.f;
    for (int q = 0; q < p.splits; ++q) {
      const float* pr = p.part + std::size_t((((b * p.splits + q) * HKV + g) * W + w) * GQA + row) * PSTR;
      const float2 ml = *reinterpret_cast<const float2*>(pr + D);
      const float f = exp2f(ml.x - mx);
      l += ml.y * f;
      a += pr[dim] * f;
    }
    const float gate_in = load_proj_val<PROJ_FP32>(p.proj, pb + OFF_GATE + (g * GQA + row) * D + dim);
    const float gate = 1.f / (1.f + expf(-gate_in));
    p.out[std::size_t(b * W + w) * HQ * D + (g * GQA + row) * D + dim] = act::from_float(bf16r(a / l) * gate);
  }
  for (int idx = tid; idx < (W - w_lim) * GQA * D; idx += NT) {
    const int w = w_lim + idx / (GQA * D);
    const int rem = idx % (GQA * D);
    const int row = rem / D, dim = rem % D;
    p.out[std::size_t(b * W + w) * HQ * D + (g * GQA + row) * D + dim] = act::from_float(0.f);
  }

  float* const conv_k_slot = p.conv_k_visit + std::size_t(slot) * p.conv_slot_stride;
  float* const conv_v_slot = p.conv_v_visit + std::size_t(slot) * p.conv_slot_stride;
  ActT* const k_ring_slot = FULL ? nullptr : (p.k_ring_visit + std::size_t(slot) * p.ring_slot_stride);
  ActT* const v_ring_slot = FULL ? nullptr : (p.v_ring_visit + std::size_t(slot) * p.ring_slot_stride);
  std::int64_t* const kpos_slot = FULL ? nullptr : (p.ring_pos_visit + std::size_t(slot) * p.ring_pos_slot_stride);
  const std::int32_t* const pt_slot = FULL ? (p.page_table + std::size_t(slot) * p.max_pages_per_slot) : nullptr;
  CT* const k_pool = FULL ? static_cast<CT*>(p.k_full_pool) : reinterpret_cast<CT*>(k_ring_slot);
  CT* const v_pool = FULL ? static_cast<CT*>(p.v_full_pool) : reinterpret_cast<CT*>(v_ring_slot);

  const ActT* src_nkv = p.saved_nkv + std::size_t(b * HKV + g) * 2 * W * D;
  for (int idx = tid; idx < 2 * w_lim * 16; idx += NT) {
    const int kv = idx / (w_lim * 16);
    const int rem = idx % (w_lim * 16);
    const int w = rem / 16, c8 = rem & 15;
    const uint4 u = reinterpret_cast<const uint4*>(src_nkv + (kv * W + w) * D)[c8];
    const int pw = p0 + w;
    std::size_t dst_off;
    if constexpr (FULL) {
      const int phys = pt_slot[pw >> kPageShift];
      dst_off = ((((std::size_t(phys) * NUM_FULL_VISITS + p.full_visit_idx) * kPageSize + (pw & kPageMask)) * HKV + g) * D);
    } else {
      dst_off = (std::size_t(pw & (WIN - 1)) * HKV + g) * D;
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
  if (!FULL && g == 0 && tid < w_lim) {
    kpos_slot[(p0 + tid) & (WIN - 1)] = p0 + tid;
  }
}

__global__ void batched_ring_pos_write_kernel(const std::int32_t* __restrict__ slot_ids,
                                              const std::int32_t* __restrict__ slot_w_limit,
                                              const std::int64_t* __restrict__ slot_pos, int W,
                                              std::int64_t* __restrict__ ring_pos_visit,
                                              std::size_t ring_pos_slot_stride) {
  const int b = blockIdx.x;
  const int w = threadIdx.x;
  const int slot = slot_ids[b];
  if (slot < 0) return;
  const int w_lim = slot_w_limit ? min(W, max(1, int(slot_w_limit[b]))) : W;
  if (w >= w_lim) return;
  const int p0 = int(slot_pos[slot]);
  ring_pos_visit[std::size_t(slot) * ring_pos_slot_stride + ((p0 + w) & (WIN - 1))] = p0 + w;
}

template <int W, bool FULL, bool F8, bool PROJ_FP32>
void dispatch_verify_attn_w(const BatchedVerifyAttnKernelParams& kp, cudaStream_t stream) {
  if (kp.splits == 1) {
    const dim3 grid(1, HKV, kp.B);
    batched_attn_verify_kernel<W, FULL, F8, PROJ_FP32, false><<<grid, NT, 0, stream>>>(kp);
    if constexpr (!FULL) {
      batched_ring_pos_write_kernel<<<kp.B, 32, 0, stream>>>(kp.slot_ids, kp.slot_w_limit, kp.slot_pos, W,
                                                             kp.ring_pos_visit, kp.ring_pos_slot_stride);
    }
  } else {
    const dim3 grid1(kp.splits, HKV, kp.B);
    batched_attn_verify_kernel<W, FULL, F8, PROJ_FP32, true><<<grid1, NT, 0, stream>>>(kp);
    const dim3 grid2(1, HKV, kp.B);
    batched_attn_verify_reduce_kernel<W, FULL, F8, PROJ_FP32><<<grid2, NT, 0, stream>>>(kp);
  }
  CUDA_CHECK(cudaGetLastError());
}

template <bool FULL, bool F8, bool PROJ_FP32>
void dispatch_verify_attn(const BatchedVerifyAttnKernelParams& kp, cudaStream_t stream) {
  switch (kp.W) {
    case 1: dispatch_verify_attn_w<1, FULL, F8, PROJ_FP32>(kp, stream); break;
    case 2: dispatch_verify_attn_w<2, FULL, F8, PROJ_FP32>(kp, stream); break;
    case 4: dispatch_verify_attn_w<4, FULL, F8, PROJ_FP32>(kp, stream); break;
    case 8: dispatch_verify_attn_w<8, FULL, F8, PROJ_FP32>(kp, stream); break;
    default: require(false, "batched_attn_verify: W must be 1, 2, 4, or 8");
  }
}

template <bool X_FP32>
__global__ void batched_static_conv_verify_kernel(const void* __restrict__ x, int ld, int W, int C,
                                                  const std::int32_t* __restrict__ slot_ids,
                                                  const std::int32_t* __restrict__ slot_w_limit,
                                                  const float* __restrict__ conv_w,
                                                  float* __restrict__ conv_state_visit, std::size_t conv_slot_stride,
                                                  float* __restrict__ out, float* __restrict__ hist) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int b = blockIdx.y;
  if (c >= C) return;
  const int slot = slot_ids[b];
  if (slot < 0) {
    for (int w = 0; w < W; ++w) {
      const std::size_t r = std::size_t(b) * W + w;
      out[r * C + c] = 0.f;
      if (hist) hist[r * C + c] = 0.f;
    }
    return;
  }

  const int w_lim = slot_w_limit ? min(W, max(1, int(slot_w_limit[b]))) : W;
  float* const st = conv_state_visit + std::size_t(slot) * conv_slot_stride;
  float e0 = st[0 * C + c];
  float e1 = st[1 * C + c];
  float e2 = st[2 * C + c];
  const float4 w4 = reinterpret_cast<const float4*>(conv_w)[c];

  for (int w = 0; w < w_lim; ++w) {
    const std::size_t r = std::size_t(b) * W + w;
    float e3;
    if constexpr (X_FP32) {
      e3 = bf16r(static_cast<const float*>(x)[r * ld + c]);
    } else {
      e3 = act::to_float(static_cast<const ActT*>(x)[r * ld + c]);
    }
    if (hist) hist[r * C + c] = e3;
    float acc = __fmul_rn(w4.x, e0);
    acc = __fadd_rn(acc, __fmul_rn(w4.y, e1));
    acc = __fadd_rn(acc, __fmul_rn(w4.z, e2));
    acc = __fadd_rn(acc, __fmul_rn(w4.w, e3));
    out[r * C + c] = __fmul_rn(__fadd_rn(acc, e3), INV_SQRT5);
    e0 = e1;
    e1 = e2;
    e2 = e3;
  }
  for (int w = w_lim; w < W; ++w) {
    const std::size_t r = std::size_t(b) * W + w;
    out[r * C + c] = 0.f;
    if (hist) hist[r * C + c] = 0.f;
  }
  st[0 * C + c] = e0;
  st[1 * C + c] = e1;
  st[2 * C + c] = e2;
}

constexpr int kRbThreads = 256;
constexpr int kRowVec = (2 * CONV_C + 2 * H_DIM) / 4;   // 1664 float4 per visit across k|v|attn|mlp
constexpr int kConvVecs = NUM_VISITS * kRowVec;          // 79872 float4 per conv row across 48 visits
constexpr int kRingVec = CONV_C * sizeof(ActT) / 16;     // 128 uint4 per ring slot row

struct RbConvAt {
  int t, v, c, cols;
};

__device__ __forceinline__ RbConvAt rb_conv_at(int q) {
  const int v = q / kRowVec, c = q % kRowVec;
  constexpr int k4 = CONV_C / 4, h4 = H_DIM / 4;
  if (c < k4) return {0, v, c, k4};
  if (c < 2 * k4) return {1, v, c - k4, k4};
  if (c < 2 * k4 + h4) return {2, v, c - 2 * k4, h4};
  return {3, v, c - 2 * k4 - h4, h4};
}

__device__ __forceinline__ float4* rb_state_conv_ptr(const BatchedVerifyStatePtrs& st, int slot, const RbConvAt& a,
                                                     int row) {
  float* base = (a.t == 0) ? (st.conv_k + std::size_t(slot) * st.conv_kv_slot_stride)
                           : (a.t == 1) ? (st.conv_v + std::size_t(slot) * st.conv_kv_slot_stride)
                                        : (a.t == 2) ? (st.conv_attn + std::size_t(slot) * st.conv_h_slot_stride)
                                                     : (st.conv_mlp + std::size_t(slot) * st.conv_h_slot_stride);
  return reinterpret_cast<float4*>(base) + (std::size_t(a.v) * 3 + row) * a.cols + a.c;
}

__device__ __forceinline__ float4* rb_bk_conv_ptr(const BatchedVerifyBackup& bk, int b, const RbConvAt& a,
                                                  int row_idx01) {
  // bk.conv_* is [max_B, NUM_VISITS, 2, cols*4], row_idx01 = 0 for conv row 1, 1 for conv row 2.
  float* base = (a.t == 0) ? bk.conv_k : (a.t == 1) ? bk.conv_v : (a.t == 2) ? bk.conv_attn : bk.conv_mlp;
  const std::size_t per_b = std::size_t(NUM_VISITS) * 2 * a.cols;
  return reinterpret_cast<float4*>(base) + std::size_t(b) * per_b + (std::size_t(a.v) * 2 + row_idx01) * a.cols + a.c;
}

__device__ __forceinline__ const float4* rb_hist_ptr(const BatchedVerifyHistory& h, int b, int W, const RbConvAt& a,
                                                     int w_row) {
  const float* base = (a.t == 0) ? h.hist_k : (a.t == 1) ? h.hist_v : (a.t == 2) ? h.hist_attn : h.hist_mlp;
  const std::size_t v_stride = (a.t < 2) ? (h.kv_visit_stride / 4) : (h.h_visit_stride / 4);
  return reinterpret_cast<const float4*>(base) + std::size_t(a.v) * v_stride +
         std::size_t(b * W + w_row) * a.cols + a.c;
}

__global__ void batched_verify_save_kernel(BatchedVerifyStatePtrs st, BatchedVerifyBackup bk,
                                           const std::int32_t* __restrict__ slot_ids,
                                           const std::int32_t* __restrict__ slot_w_limit, int W) {
  const int b = blockIdx.y;
  const int slot = slot_ids[b];
  if (slot < 0) return;

  const int w_lim = slot_w_limit ? min(W, max(1, int(slot_w_limit[b]))) : W;
  const int i = blockIdx.x * kRbThreads + threadIdx.x;
  if (i < 2 * kConvVecs) {
    const int row01 = i / kConvVecs;  // 0 -> state row 1, 1 -> state row 2
    const RbConvAt a = rb_conv_at(i % kConvVecs);
    *rb_bk_conv_ptr(bk, b, a, row01) = *rb_state_conv_ptr(st, slot, a, row01 + 1);
    return;
  }

  const int p0 = int(st.slot_pos[slot]);
  int j = i - 2 * kConvVecs;
  const int per_ring = W * 2 * kRingVec;
  if (j < NUM_SLIDING_VISITS * per_ring) {
    const int r = j / per_ring, k = j % per_ring;
    const int row = k / (2 * kRingVec), c = k % (2 * kRingVec);
    if (row >= w_lim) return;
    const int ring_slot = (p0 + row) & (WIN - 1);
    const std::size_t bk_off = ((std::size_t(b) * NUM_SLIDING_VISITS + r) * bk.max_W + row) * kRingVec;
    if (c < kRingVec) {
      const ActT* src = st.k_ring + std::size_t(slot) * st.ring_slot_stride + (std::size_t(r) * WIN + ring_slot) * CONV_C;
      reinterpret_cast<uint4*>(bk.ring_k)[bk_off + c] = reinterpret_cast<const uint4*>(src)[c];
    } else {
      const ActT* src = st.v_ring + std::size_t(slot) * st.ring_slot_stride + (std::size_t(r) * WIN + ring_slot) * CONV_C;
      reinterpret_cast<uint4*>(bk.ring_v)[bk_off + c - kRingVec] = reinterpret_cast<const uint4*>(src)[c - kRingVec];
    }
    return;
  }
  j -= NUM_SLIDING_VISITS * per_ring;
  if (j < NUM_SLIDING_VISITS * W) {
    const int r = j / W, row = j % W;
    if (row >= w_lim) return;
    const int ring_slot = (p0 + row) & (WIN - 1);
    const std::int64_t* src_rp = st.ring_pos + std::size_t(slot) * st.ring_pos_slot_stride;
    bk.ring_pos[(std::size_t(b) * NUM_SLIDING_VISITS + r) * bk.max_W + row] = src_rp[std::size_t(r) * WIN + ring_slot];
  }
}

__global__ void batched_verify_rollback_kernel(BatchedVerifyStatePtrs st, BatchedVerifyBackup bk,
                                               BatchedVerifyHistory hist, const std::int32_t* __restrict__ slot_ids,
                                               const std::int32_t* __restrict__ slot_w_limit,
                                               const std::int32_t* __restrict__ commit_counts, int W) {
  const int b = blockIdx.y;
  const int slot = slot_ids[b];
  if (slot < 0) return;

  const int w_lim = slot_w_limit ? min(W, max(1, int(slot_w_limit[b]))) : W;
  const int commit = commit_counts[b];
  if (commit >= w_lim) return;

  const int i = blockIdx.x * kRbThreads + threadIdx.x;
  if (i < 3 * kConvVecs) {
    const int row_j = i / kConvVecs;
    const int src = commit + row_j;
    const RbConvAt a = rb_conv_at(i % kConvVecs);
    const float4 val = (src < 3) ? *rb_bk_conv_ptr(bk, b, a, src - 1) : *rb_hist_ptr(hist, b, W, a, src - 3);
    *rb_state_conv_ptr(st, slot, a, row_j) = val;
    return;
  }

  const int p0 = int(st.slot_pos[slot]);
  int j = i - 3 * kConvVecs;
  const int per_ring = W * 2 * kRingVec;
  if (j < NUM_SLIDING_VISITS * per_ring) {
    const int r = j / per_ring, k = j % per_ring;
    const int row = k / (2 * kRingVec), c = k % (2 * kRingVec);
    if (row < commit || row >= w_lim) return;
    const int ring_slot = (p0 + row) & (WIN - 1);
    const std::size_t bk_off = ((std::size_t(b) * NUM_SLIDING_VISITS + r) * bk.max_W + row) * kRingVec;
    if (c < kRingVec) {
      ActT* dst = st.k_ring + std::size_t(slot) * st.ring_slot_stride + (std::size_t(r) * WIN + ring_slot) * CONV_C;
      reinterpret_cast<uint4*>(dst)[c] = reinterpret_cast<const uint4*>(bk.ring_k)[bk_off + c];
    } else {
      ActT* dst = st.v_ring + std::size_t(slot) * st.ring_slot_stride + (std::size_t(r) * WIN + ring_slot) * CONV_C;
      reinterpret_cast<uint4*>(dst)[c - kRingVec] = reinterpret_cast<const uint4*>(bk.ring_v)[bk_off + c - kRingVec];
    }
    return;
  }
  j -= NUM_SLIDING_VISITS * per_ring;
  if (j < NUM_SLIDING_VISITS * W) {
    const int r = j / W, row = j % W;
    if (row < commit || row >= w_lim) return;
    const int ring_slot = (p0 + row) & (WIN - 1);
    std::int64_t* dst_rp = st.ring_pos + std::size_t(slot) * st.ring_pos_slot_stride;
    dst_rp[std::size_t(r) * WIN + ring_slot] = bk.ring_pos[(std::size_t(b) * NUM_SLIDING_VISITS + r) * bk.max_W + row];
  }
}

__global__ void batched_advance_pos_by_kernel(const std::int32_t* __restrict__ slot_ids,
                                              const std::int32_t* __restrict__ commit_counts, int B,
                                              std::int64_t* __restrict__ slot_pos) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  const int slot = slot_ids[b];
  if (slot >= 0) slot_pos[slot] += commit_counts[b];
}

}  // namespace

std::size_t attn_verify_workspace_bytes(int max_b, int max_w) {
  require(max_b >= 1 && (max_w == 1 || max_w == 2 || max_w == 4 || max_w == 8), "attn_verify_workspace_bytes: bad args");
  return verify_ws_total_bytes(max_b, max_w);
}

void batched_attn_verify(const BatchedVerifyAttnArgs& a, cudaStream_t stream) {
  require(a.B >= 1 && (a.W == 1 || a.W == 2 || a.W == 4 || a.W == 8), "batched_attn_verify: bad B/W");
  require(a.window == 0 || a.window == WIN, "batched_attn_verify: window must be 0 or 1024");
  const bool full = (a.window == 0);
  require(full || !a.cache_fp8, "batched_attn_verify: FP8 caches are full-layer only");
  require(a.proj && a.slot_ids && a.slot_pos && a.q_scale && a.r_scale && a.rel_proj && a.k_conv && a.v_conv &&
              a.conv_k_visit && a.conv_v_visit && a.out,
          "batched_attn_verify: null argument");
  if (full) {
    require(a.k_full_pool && a.v_full_pool && a.page_table && a.max_pages_per_slot > 0 && a.max_ctx > 0,
            "batched_attn_verify: full layer needs paged pool and page_table");
  } else {
    require(a.k_ring_visit && a.v_ring_visit && a.ring_pos_visit, "batched_attn_verify: sliding layer needs rings");
  }
  const int splits = choose_verify_splits(a.B);
  if (splits > 1) {
    require(a.workspace && a.workspace_size >= verify_ws_total_bytes(a.B, a.W),
            "batched_attn_verify: workspace too small");
  }

  BatchedVerifyAttnKernelParams kp{};
  kp.proj = a.proj;
  kp.R = full ? R_FULL : R_SLIDING;
  kp.B = a.B;
  kp.W = a.W;
  kp.splits = splits;
  kp.slot_ids = a.slot_ids;
  kp.slot_w_limit = a.slot_w_limit;
  kp.slot_pos = a.slot_pos;
  kp.q_scale = a.q_scale;
  kp.r_scale = a.r_scale;
  kp.rel_proj = a.rel_proj;
  kp.k_conv = a.k_conv;
  kp.v_conv = a.v_conv;
  kp.conv_k_visit = a.conv_k_visit;
  kp.conv_v_visit = a.conv_v_visit;
  kp.conv_slot_stride = a.conv_slot_stride;
  kp.hist_k_visit = a.hist_k_visit;
  kp.hist_v_visit = a.hist_v_visit;
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
    kp.saved_nkv = reinterpret_cast<ActT*>(ws + verify_ws_nkv_off(a.B, a.W));
    kp.saved_nst = reinterpret_cast<float*>(ws + verify_ws_nst_off(a.B, a.W));
  }
  kp.out = a.out;

  if (full) {
    if (a.cache_fp8) {
      if (a.proj_fp32) dispatch_verify_attn<true, true, true>(kp, stream);
      else dispatch_verify_attn<true, true, false>(kp, stream);
    } else {
      if (a.proj_fp32) dispatch_verify_attn<true, false, true>(kp, stream);
      else dispatch_verify_attn<true, false, false>(kp, stream);
    }
  } else {
    if (a.proj_fp32) dispatch_verify_attn<false, false, true>(kp, stream);
    else dispatch_verify_attn<false, false, false>(kp, stream);
  }
}

void batched_static_conv_verify(const void* x, bool x_fp32, int ld, int B, int W, int C,
                                const std::int32_t* slot_ids, const std::int32_t* slot_w_limit,
                                const float* conv_w, float* conv_state_visit, std::size_t conv_slot_stride,
                                float* out, float* hist, cudaStream_t stream) {
  require(x && slot_ids && conv_w && conv_state_visit && out && B >= 1 && W >= 1 && W <= 8 && C >= 1 && ld >= C,
          "batched_static_conv_verify: bad args");
  const dim3 grid((C + 255) / 256, B);
  if (x_fp32) {
    batched_static_conv_verify_kernel<true><<<grid, 256, 0, stream>>>(
        x, ld, W, C, slot_ids, slot_w_limit, conv_w, conv_state_visit, conv_slot_stride, out, hist);
  } else {
    batched_static_conv_verify_kernel<false><<<grid, 256, 0, stream>>>(
        x, ld, W, C, slot_ids, slot_w_limit, conv_w, conv_state_visit, conv_slot_stride, out, hist);
  }
  CUDA_CHECK(cudaGetLastError());
}

void batched_verify_save(const BatchedVerifyStatePtrs& st, const BatchedVerifyBackup& bk,
                         const std::int32_t* slot_ids, const std::int32_t* slot_w_limit, int B, int W,
                         cudaStream_t stream) {
  if (B <= 0) return;
  require(slot_ids && st.conv_k && st.conv_v && st.conv_attn && st.conv_mlp && st.k_ring && st.v_ring && st.ring_pos &&
              st.slot_pos && bk.conv_k && bk.conv_v && bk.conv_attn && bk.conv_mlp && bk.ring_k && bk.ring_v &&
              bk.ring_pos && W >= 1 && W <= bk.max_W,
          "batched_verify_save: bad args");
  const int items = 2 * kConvVecs + NUM_SLIDING_VISITS * W * (2 * kRingVec + 1);
  const dim3 grid((items + kRbThreads - 1) / kRbThreads, B);
  batched_verify_save_kernel<<<grid, kRbThreads, 0, stream>>>(st, bk, slot_ids, slot_w_limit, W);
  CUDA_CHECK(cudaGetLastError());
}

void batched_verify_rollback(const BatchedVerifyStatePtrs& st, const BatchedVerifyBackup& bk,
                             const BatchedVerifyHistory& hist, const std::int32_t* slot_ids,
                             const std::int32_t* slot_w_limit, const std::int32_t* commit_counts, int B, int W,
                             cudaStream_t stream) {
  if (B <= 0) return;
  require(slot_ids && commit_counts && st.slot_pos && W >= 1, "batched_verify_rollback: bad args");
  if (W > 1) {
    require(st.conv_k && st.conv_v && st.conv_attn && st.conv_mlp && st.k_ring && st.v_ring && st.ring_pos &&
                bk.conv_k && bk.conv_v && bk.conv_attn && bk.conv_mlp && bk.ring_k && bk.ring_v && bk.ring_pos &&
                hist.hist_k && hist.hist_v && hist.hist_attn && hist.hist_mlp && W <= bk.max_W,
            "batched_verify_rollback: null rollback buffers");
    const int items = 3 * kConvVecs + NUM_SLIDING_VISITS * W * (2 * kRingVec + 1);
    const dim3 grid((items + kRbThreads - 1) / kRbThreads, B);
    batched_verify_rollback_kernel<<<grid, kRbThreads, 0, stream>>>(st, bk, hist, slot_ids, slot_w_limit, commit_counts,
                                                                    W);
  }
  // Separate post-grid launch guarantees every rollback block finished reading old slot_pos[s] before advancing!
  batched_advance_pos_by_kernel<<<(B + 63) / 64, 64, 0, stream>>>(slot_ids, commit_counts, B, st.slot_pos);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::batch

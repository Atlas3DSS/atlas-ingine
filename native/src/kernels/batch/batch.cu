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
constexpr int SINK_PAD = (HQ + 7) & ~7;
constexpr int NUM_FULL_VISITS = 8;
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

}  // namespace ingine::kernels::batch

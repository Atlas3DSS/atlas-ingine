// GPU kernel parity tests for kernels/batch/:
// 1. batched_static_conv_step vs exact FP32/CPU reference (B=1,8,32,64, x_fp32 true/false, padded slot=-1).
// 2. batched_attn_decode vs kernels::attn::attn_decode (M=1 single-sequence oracle) for:
//    - sliding window (1024) with mixed positions (including wrapped ring > 1024),
//    - paged full KV (BF16/FP16 & FP8 E4M3) with shuffled physical page IDs and mixed multi-page positions,
//    - both proj_fp32=true and proj_fp32=false, B=1,4,16,64, and padded slot=-1 no-op guarantee.
// 3. scatter_full_kv + gather_full_kv roundtrip (BF16 & FP8) and batched_head_softcap exactness.
#include "core/act.h"
#include "core/arena.h"
#include "core/device.h"
#include "kernels/attn/attn.h"
#include "kernels/batch/batch.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

namespace attn = ingine::kernels::attn;
namespace bk = ingine::kernels::batch;
using ingine::ActT;
using ingine::DeviceBuffer;

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, WIN = 1024, KV_DIM = HKV * D, H = 2304;
constexpr int NUM_FULL_VISITS = 8;
constexpr int R_SLIDING = 8600, R_FULL = 8664;

int g_failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

std::vector<float> rand_floats(std::mt19937& rng, std::size_t n, float scale = 0.5f) {
  std::normal_distribution<float> dist(0.f, scale);
  std::vector<float> v(n);
  for (float& x : v) x = dist(rng);
  return v;
}

std::vector<ActT> to_act(const std::vector<float>& f) {
  std::vector<ActT> a(f.size());
  for (std::size_t i = 0; i < f.size(); ++i) a[i] = ingine::act::host_from_float(f[i]);
  return a;
}

std::vector<std::uint8_t> to_e4m3(const std::vector<float>& f) {
  std::vector<std::uint8_t> out(f.size());
  for (std::size_t i = 0; i < f.size(); ++i) {
    const float c = std::clamp(ingine::act::host_round(f[i]), -448.f, 448.f);
    out[i] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(c, __NV_SATFINITE, __NV_E4M3));
  }
  return out;
}

void test_static_conv() {
  std::mt19937 rng(42);
  const int max_slots = 68;
  const std::size_t slot_stride = 3 * H;
  const auto w_host = rand_floats(rng, H * 4, 0.25f);
  const auto st_init = rand_floats(rng, std::size_t(max_slots) * slot_stride, 0.4f);

  DeviceBuffer d_w(w_host.size() * sizeof(float));
  d_w.copy_from_host(w_host.data(), d_w.bytes);

  for (bool x_fp32 : {true, false}) {
    for (int B : {1, 8, 32, 64}) {
      DeviceBuffer d_st(st_init.size() * sizeof(float));
      d_st.copy_from_host(st_init.data(), d_st.bytes);

      std::vector<std::int32_t> slot_ids(std::size_t(B), -1);
      for (int b = 0; b < B; ++b) slot_ids[std::size_t(b)] = (b % 7 == 6 && B > 1) ? -1 : (B - 1 - b + 2);
      DeviceBuffer d_slots(slot_ids.size() * sizeof(std::int32_t));
      d_slots.copy_from_host(slot_ids.data(), d_slots.bytes);

      const auto x_f32 = rand_floats(rng, std::size_t(B) * H, 0.5f);
      const auto x_act = to_act(x_f32);
      DeviceBuffer d_x(x_fp32 ? x_f32.size() * sizeof(float) : x_act.size() * sizeof(ActT));
      if (x_fp32) d_x.copy_from_host(x_f32.data(), d_x.bytes);
      else d_x.copy_from_host(x_act.data(), d_x.bytes);

      DeviceBuffer d_out(std::size_t(B) * H * sizeof(float));
      d_out.fill(0);

      bk::batched_static_conv_step(d_x.p, x_fp32, H, B, H, static_cast<const std::int32_t*>(d_slots.p),
                                   static_cast<const float*>(d_w.p), static_cast<float*>(d_st.p), slot_stride,
                                   static_cast<float*>(d_out.p), nullptr);
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<float> got_out(std::size_t(B) * H);
      std::vector<float> got_st(st_init.size());
      d_out.copy_to_host(got_out.data(), d_out.bytes);
      d_st.copy_to_host(got_st.data(), d_st.bytes);

      std::vector<float> exp_st = st_init;
      const float inv_sqrt5 = 1.f / 2.2360679774997896f;
      float max_err = 0.f;
      for (int b = 0; b < B; ++b) {
        const int s = slot_ids[std::size_t(b)];
        if (s < 0) continue;
        for (int c = 0; c < H; ++c) {
          const float e0 = st_init[std::size_t(s) * slot_stride + 0 * H + c];
          const float e1 = st_init[std::size_t(s) * slot_stride + 1 * H + c];
          const float e2 = st_init[std::size_t(s) * slot_stride + 2 * H + c];
          const float e3 = ingine::act::host_round(x_f32[std::size_t(b) * H + c]);
          const float* w4 = &w_host[std::size_t(c) * 4];
          const float acc = ((w4[0] * e0 + w4[1] * e1) + w4[2] * e2) + w4[3] * e3;
          const float exp = (acc + e3) * inv_sqrt5;
          max_err = std::max(max_err, std::fabs(got_out[std::size_t(b) * H + c] - exp));
          exp_st[std::size_t(s) * slot_stride + 0 * H + c] = e1;
          exp_st[std::size_t(s) * slot_stride + 1 * H + c] = e2;
          exp_st[std::size_t(s) * slot_stride + 2 * H + c] = e3;
        }
      }
      CHECK(max_err <= 1e-6f);
      CHECK(got_st == exp_st);
    }
  }
}

void test_batched_attn_vs_single(bool full, bool cache_fp8, bool proj_fp32, int B) {
  std::mt19937 rng(1000 + B * 17 + (full ? 100 : 0) + (cache_fp8 ? 50 : 0) + (proj_fp32 ? 7 : 0));
  const int R = full ? R_FULL : R_SLIDING;
  const int max_slots = B + 4;
  const int max_ctx = 512;
  const int max_pages_per_slot = (max_ctx + bk::kPageSize - 1) / bk::kPageSize;
  const int full_visit_idx = 3;

  // Shared layer weights
  const auto q_scale = rand_floats(rng, HQ * D, 0.3f);
  const auto r_scale = rand_floats(rng, HQ * DREL, 0.3f);
  const auto rel_proj = rand_floats(rng, DREL * EXT, 0.2f);
  const auto k_conv = rand_floats(rng, KV_DIM * 4, 0.2f);
  const auto v_conv = rand_floats(rng, KV_DIM * 4, 0.2f);

  DeviceBuffer d_qs(q_scale.size() * 4), d_rs(r_scale.size() * 4), d_rp(rel_proj.size() * 4);
  DeviceBuffer d_kc(k_conv.size() * 4), d_vc(v_conv.size() * 4);
  d_qs.copy_from_host(q_scale.data(), d_qs.bytes);
  d_rs.copy_from_host(r_scale.data(), d_rs.bytes);
  d_rp.copy_from_host(rel_proj.data(), d_rp.bytes);
  d_kc.copy_from_host(k_conv.data(), d_kc.bytes);
  d_vc.copy_from_host(v_conv.data(), d_vc.bytes);

  // Batch projections (rounded to ActT so proj_fp32=true and proj_fp32=false compare apples-to-apples with attn_decode)
  auto proj_f32 = rand_floats(rng, std::size_t(B) * R, 0.3f);
  for (float& x : proj_f32) x = ingine::act::host_round(x);
  const auto proj_act = to_act(proj_f32);
  DeviceBuffer d_proj(proj_fp32 ? proj_f32.size() * 4 : proj_act.size() * sizeof(ActT));
  if (proj_fp32) d_proj.copy_from_host(proj_f32.data(), d_proj.bytes);
  else d_proj.copy_from_host(proj_act.data(), d_proj.bytes);

  // Slot IDs (include padded -1 rows when B >= 4) and heterogeneous positions
  std::vector<std::int32_t> slot_ids(static_cast<std::size_t>(B), -1);
  std::vector<std::int64_t> slot_pos(std::size_t(max_slots), 0);
  for (int b = 0; b < B; ++b) {
    if (B >= 4 && b == B - 1) {
      slot_ids[std::size_t(b)] = -1;
    } else {
      const int s = b + 1;
      slot_ids[std::size_t(b)] = s;
      slot_pos[std::size_t(s)] = full ? (5 + (b * 19) % 240) : (3 + (b * 137) % 1400);
    }
  }
  DeviceBuffer d_slot_ids(slot_ids.size() * 4), d_slot_pos(slot_pos.size() * 8);
  d_slot_ids.copy_from_host(slot_ids.data(), d_slot_ids.bytes);
  d_slot_pos.copy_from_host(slot_pos.data(), d_slot_pos.bytes);

  // Per-slot conv states
  const std::size_t conv_stride = 3 * KV_DIM;
  const auto conv_k_init = rand_floats(rng, std::size_t(max_slots) * conv_stride, 0.3f);
  const auto conv_v_init = rand_floats(rng, std::size_t(max_slots) * conv_stride, 0.3f);
  DeviceBuffer d_conv_k(conv_k_init.size() * 4), d_conv_v(conv_v_init.size() * 4);
  d_conv_k.copy_from_host(conv_k_init.data(), d_conv_k.bytes);
  d_conv_v.copy_from_host(conv_v_init.data(), d_conv_v.bytes);

  // Allocate sliding rings or paged full KV pool
  const std::size_t ring_stride = std::size_t(WIN) * KV_DIM;
  const std::size_t ring_pos_stride = WIN;
  DeviceBuffer d_k_ring, d_v_ring, d_ring_pos;
  std::vector<ActT> k_ring_init, v_ring_init;
  std::vector<std::int64_t> ring_pos_init;

  const int num_pages = max_slots * max_pages_per_slot;
  const std::size_t es = cache_fp8 ? 1 : 2;
  DeviceBuffer d_k_pool, d_v_pool, d_pt;
  std::vector<std::int32_t> pt_host(std::size_t(max_slots) * max_pages_per_slot, -1);
  std::vector<std::uint8_t> k_full_slot_init, v_full_slot_init;  // [max_slots, max_ctx, KV_DIM] in es bytes

  if (!full) {
    k_ring_init = to_act(rand_floats(rng, std::size_t(max_slots) * ring_stride, 0.25f));
    v_ring_init = to_act(rand_floats(rng, std::size_t(max_slots) * ring_stride, 0.25f));
    ring_pos_init.assign(std::size_t(max_slots) * ring_pos_stride, -1);
    for (int b = 0; b < B; ++b) {
      const int s = slot_ids[std::size_t(b)];
      if (s < 0) continue;
      const int p0 = int(slot_pos[std::size_t(s)]);
      const int start = std::max(0, p0 - WIN);
      for (int p = start; p < p0; ++p) ring_pos_init[std::size_t(s) * ring_pos_stride + (p & (WIN - 1))] = p;
    }
    d_k_ring = DeviceBuffer(k_ring_init.size() * sizeof(ActT));
    d_v_ring = DeviceBuffer(v_ring_init.size() * sizeof(ActT));
    d_ring_pos = DeviceBuffer(ring_pos_init.size() * sizeof(std::int64_t));
    d_k_ring.copy_from_host(k_ring_init.data(), d_k_ring.bytes);
    d_v_ring.copy_from_host(v_ring_init.data(), d_v_ring.bytes);
    d_ring_pos.copy_from_host(ring_pos_init.data(), d_ring_pos.bytes);
  } else {
    // Shuffle physical page IDs so non-contiguous paging is thoroughly exercised!
    std::vector<int> perm(static_cast<std::size_t>(num_pages), 0);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int s = 0; s < max_slots; ++s)
      for (int lp = 0; lp < max_pages_per_slot; ++lp)
        pt_host[std::size_t(s) * max_pages_per_slot + lp] = perm[std::size_t(s) * max_pages_per_slot + lp];
    d_pt = DeviceBuffer(pt_host.size() * 4);
    d_pt.copy_from_host(pt_host.data(), d_pt.bytes);

    const std::size_t page_bytes = std::size_t(NUM_FULL_VISITS) * bk::kPageSize * KV_DIM * es;
    d_k_pool = DeviceBuffer(std::size_t(num_pages) * page_bytes);
    d_v_pool = DeviceBuffer(std::size_t(num_pages) * page_bytes);
    d_k_pool.fill(0);
    d_v_pool.fill(0);

    const std::size_t slot_full_bytes = std::size_t(max_ctx) * KV_DIM * es;
    k_full_slot_init.resize(std::size_t(max_slots) * slot_full_bytes, 0);
    v_full_slot_init.resize(std::size_t(max_slots) * slot_full_bytes, 0);
    DeviceBuffer d_tmp_k(std::size_t(NUM_FULL_VISITS) * slot_full_bytes);
    DeviceBuffer d_tmp_v(std::size_t(NUM_FULL_VISITS) * slot_full_bytes);

    for (int b = 0; b < B; ++b) {
      const int s = slot_ids[std::size_t(b)];
      if (s < 0) continue;
      const int p0 = int(slot_pos[std::size_t(s)]);
      if (p0 == 0) continue;
      const auto kf = rand_floats(rng, std::size_t(p0) * KV_DIM, 0.25f);
      const auto vf = rand_floats(rng, std::size_t(p0) * KV_DIM, 0.25f);
      if (cache_fp8) {
        const auto ke = to_e4m3(kf), ve = to_e4m3(vf);
        std::memcpy(k_full_slot_init.data() + std::size_t(s) * slot_full_bytes, ke.data(), ke.size());
        std::memcpy(v_full_slot_init.data() + std::size_t(s) * slot_full_bytes, ve.data(), ve.size());
      } else {
        const auto ka = to_act(kf), va = to_act(vf);
        std::memcpy(k_full_slot_init.data() + std::size_t(s) * slot_full_bytes, ka.data(), ka.size() * sizeof(ActT));
        std::memcpy(v_full_slot_init.data() + std::size_t(s) * slot_full_bytes, va.data(), va.size() * sizeof(ActT));
      }
      d_tmp_k.fill(0);
      d_tmp_v.fill(0);
      CUDA_CHECK(cudaMemcpy(static_cast<char*>(d_tmp_k.p) + std::size_t(full_visit_idx) * slot_full_bytes,
                            k_full_slot_init.data() + std::size_t(s) * slot_full_bytes, std::size_t(p0) * KV_DIM * es,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(static_cast<char*>(d_tmp_v.p) + std::size_t(full_visit_idx) * slot_full_bytes,
                            v_full_slot_init.data() + std::size_t(s) * slot_full_bytes, std::size_t(p0) * KV_DIM * es,
                            cudaMemcpyHostToDevice));
      bk::scatter_full_kv(d_tmp_k.p, d_tmp_v.p, max_ctx, 0, p0, cache_fp8, s, static_cast<const std::int32_t*>(d_pt.p),
                          max_pages_per_slot, d_k_pool.p, d_v_pool.p, nullptr);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Run batched_attn_decode
  DeviceBuffer d_ws(bk::attn_workspace_bytes(B));
  d_ws.fill(0);
  DeviceBuffer d_out(std::size_t(B) * HQ * D * sizeof(ActT));
  d_out.fill(0);

  bk::BatchedAttnArgs a;
  a.proj = d_proj.p;
  a.proj_fp32 = proj_fp32;
  a.B = B;
  a.slot_ids = static_cast<const std::int32_t*>(d_slot_ids.p);
  a.slot_pos = static_cast<const std::int64_t*>(d_slot_pos.p);
  a.q_scale = static_cast<const float*>(d_qs.p);
  a.r_scale = static_cast<const float*>(d_rs.p);
  a.rel_proj = static_cast<const float*>(d_rp.p);
  a.k_conv = static_cast<const float*>(d_kc.p);
  a.v_conv = static_cast<const float*>(d_vc.p);
  a.conv_k_visit = static_cast<float*>(d_conv_k.p);
  a.conv_v_visit = static_cast<float*>(d_conv_v.p);
  a.conv_slot_stride = conv_stride;
  if (!full) {
    a.k_ring_visit = static_cast<ActT*>(d_k_ring.p);
    a.v_ring_visit = static_cast<ActT*>(d_v_ring.p);
    a.ring_pos_visit = static_cast<std::int64_t*>(d_ring_pos.p);
    a.ring_slot_stride = ring_stride;
    a.ring_pos_slot_stride = ring_pos_stride;
    a.window = WIN;
    a.log_scaling = false;
  } else {
    a.k_full_pool = d_k_pool.p;
    a.v_full_pool = d_v_pool.p;
    a.page_table = static_cast<const std::int32_t*>(d_pt.p);
    a.max_pages_per_slot = max_pages_per_slot;
    a.full_visit_idx = full_visit_idx;
    a.max_ctx = max_ctx;
    a.cache_fp8 = cache_fp8;
    a.window = 0;
    a.log_scaling = true;
  }
  a.workspace = d_ws.p;
  a.workspace_size = d_ws.bytes;
  a.out = static_cast<ActT*>(d_out.p);
  bk::batched_attn_decode(a, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ActT> got_out(std::size_t(B) * HQ * D);
  d_out.copy_to_host(got_out.data(), d_out.bytes);
  std::vector<float> got_ck(conv_k_init.size()), got_cv(conv_v_init.size());
  d_conv_k.copy_to_host(got_ck.data(), d_conv_k.bytes);
  d_conv_v.copy_to_host(got_cv.data(), d_conv_v.bytes);

  // Compare each active row against single-sequence attn::attn_decode (M=1)!
  DeviceBuffer ref_ws(attn::workspace_bytes(1));
  ref_ws.fill(0);
  DeviceBuffer ref_proj(std::size_t(R) * 4), ref_pos(8), ref_out(std::size_t(HQ) * D * sizeof(ActT));
  DeviceBuffer ref_ck(conv_stride * 4), ref_cv(conv_stride * 4);
  DeviceBuffer ref_kc(full ? std::size_t(max_ctx) * KV_DIM * es : ring_stride * sizeof(ActT));
  DeviceBuffer ref_vc(full ? std::size_t(max_ctx) * KV_DIM * es : ring_stride * sizeof(ActT));
  DeviceBuffer ref_kpos(ring_pos_stride * 8);

  float max_err = 0.f;
  for (int b = 0; b < B; ++b) {
    const int s = slot_ids[std::size_t(b)];
    if (s < 0) continue;
    ref_proj.copy_from_host(proj_f32.data() + std::size_t(b) * R, ref_proj.bytes);
    const std::int64_t p0 = slot_pos[std::size_t(s)];
    ref_pos.copy_from_host(&p0, 8);
    ref_ck.copy_from_host(conv_k_init.data() + std::size_t(s) * conv_stride, ref_ck.bytes);
    ref_cv.copy_from_host(conv_v_init.data() + std::size_t(s) * conv_stride, ref_cv.bytes);
    if (!full) {
      ref_kc.copy_from_host(k_ring_init.data() + std::size_t(s) * ring_stride, ref_kc.bytes);
      ref_vc.copy_from_host(v_ring_init.data() + std::size_t(s) * ring_stride, ref_vc.bytes);
      ref_kpos.copy_from_host(ring_pos_init.data() + std::size_t(s) * ring_pos_stride, ref_kpos.bytes);
    } else {
      const std::size_t slot_full_bytes = std::size_t(max_ctx) * KV_DIM * es;
      ref_kc.copy_from_host(k_full_slot_init.data() + std::size_t(s) * slot_full_bytes, slot_full_bytes);
      ref_vc.copy_from_host(v_full_slot_init.data() + std::size_t(s) * slot_full_bytes, slot_full_bytes);
    }

    attn::DecodeArgs da;
    da.proj = static_cast<const float*>(ref_proj.p);
    da.M = 1;
    da.q_scale = static_cast<const float*>(d_qs.p);
    da.r_scale = static_cast<const float*>(d_rs.p);
    da.rel_proj = static_cast<const float*>(d_rp.p);
    da.k_conv = static_cast<const float*>(d_kc.p);
    da.v_conv = static_cast<const float*>(d_vc.p);
    da.conv_k_state = static_cast<float*>(ref_ck.p);
    da.conv_v_state = static_cast<float*>(ref_cv.p);
    da.k_cache = ref_kc.p;
    da.v_cache = ref_vc.p;
    da.cache_fp8 = cache_fp8;
    da.cache_rows = full ? max_ctx : WIN;
    da.kpos = full ? nullptr : static_cast<std::int64_t*>(ref_kpos.p);
    da.pos = static_cast<const std::int64_t*>(ref_pos.p);
    da.window = full ? 0 : WIN;
    da.log_scaling = full;
    da.workspace = ref_ws.p;
    da.workspace_size = ref_ws.bytes;
    da.out = static_cast<ActT*>(ref_out.p);
    attn::attn_decode(da, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<ActT> exp_out(HQ * D);
    ref_out.copy_to_host(exp_out.data(), ref_out.bytes);
    for (int i = 0; i < HQ * D; ++i) {
      const float g = ingine::act::host_to_float(got_out[std::size_t(b) * HQ * D + i]);
      const float e = ingine::act::host_to_float(exp_out[std::size_t(i)]);
      max_err = std::max(max_err, std::fabs(g - e));
    }

    // Verify conv_k and conv_v state updates match bit-for-bit!
    std::vector<float> exp_ck(conv_stride), exp_cv(conv_stride);
    ref_ck.copy_to_host(exp_ck.data(), ref_ck.bytes);
    ref_cv.copy_to_host(exp_cv.data(), ref_cv.bytes);
    for (std::size_t i = 0; i < conv_stride; ++i) {
      CHECK(got_ck[std::size_t(s) * conv_stride + i] == exp_ck[i]);
      CHECK(got_cv[std::size_t(s) * conv_stride + i] == exp_cv[i]);
    }
  }

  // Verify unused slot 0 was NEVER touched!
  for (std::size_t i = 0; i < conv_stride; ++i) {
    CHECK(got_ck[i] == conv_k_init[i]);
    CHECK(got_cv[i] == conv_v_init[i]);
  }
  CHECK(max_err <= 2e-2f);
  if (max_err > 2e-2f) {
    std::fprintf(stderr, "attn mismatch (full=%d fp8=%d proj_fp32=%d B=%d): max_err=%g\n",
                 int(full), int(cache_fp8), int(proj_fp32), B, max_err);
  }
}

void test_head_softcap_and_scatter_gather() {
  std::mt19937 rng(999);
  const int B = 16, total_V = 4096, chunk_V = 1024;
  const auto raw_f = rand_floats(rng, std::size_t(B) * total_V, 12.f);
  const auto raw_a = to_act(raw_f);
  DeviceBuffer d_out(std::size_t(B) * total_V * 4);
  d_out.fill(0);
  for (int v0 = 0; v0 < total_V; v0 += chunk_V) {
    std::vector<ActT> chunk(std::size_t(B) * chunk_V);
    for (int b = 0; b < B; ++b)
      for (int v = 0; v < chunk_V; ++v)
        chunk[std::size_t(b) * chunk_V + v] = raw_a[std::size_t(b) * total_V + v0 + v];
    DeviceBuffer d_in(chunk.size() * sizeof(ActT));
    d_in.copy_from_host(chunk.data(), d_in.bytes);
    bk::batched_head_softcap(static_cast<const ActT*>(d_in.p), B, chunk_V, v0, total_V, static_cast<float*>(d_out.p),
                             nullptr);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(std::size_t(B) * total_V);
  d_out.copy_to_host(got.data(), d_out.bytes);
  float max_err = 0.f;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float x = ingine::act::host_to_float(raw_a[i]);
    const float exp = 30.f * std::tanh(x * (1.f / 30.f));
    max_err = std::max(max_err, std::fabs(got[i] - exp));
  }
  CHECK(max_err <= 1e-5f);

  // Scatter -> gather roundtrip for both BF16 and FP8
  for (bool fp8 : {false, true}) {
    const int scratch_ctx = 256, n = 150, max_pages = 4, num_pages = 8;
    const std::size_t es = fp8 ? 1 : 2;
    const std::size_t scratch_bytes = std::size_t(NUM_FULL_VISITS) * scratch_ctx * KV_DIM * es;
    std::vector<std::uint8_t> src_k(scratch_bytes), src_v(scratch_bytes);
    for (std::size_t i = 0; i < scratch_bytes; ++i) {
      src_k[i] = std::uint8_t((i * 131 + 7) & 0xff);
      src_v[i] = std::uint8_t((i * 73 + 29) & 0xff);
    }
    DeviceBuffer d_sk(scratch_bytes), d_sv(scratch_bytes), d_gk(scratch_bytes), d_gv(scratch_bytes);
    d_sk.copy_from_host(src_k.data(), scratch_bytes);
    d_sv.copy_from_host(src_v.data(), scratch_bytes);
    d_gk.fill(0);
    d_gv.fill(0);
    const std::size_t pool_bytes = std::size_t(num_pages) * NUM_FULL_VISITS * bk::kPageSize * KV_DIM * es;
    DeviceBuffer d_pk(pool_bytes), d_pv(pool_bytes);
    d_pk.fill(0);
    d_pv.fill(0);
    const std::vector<std::int32_t> pt = {5, 2, 7, 1};
    DeviceBuffer d_pt(pt.size() * 4);
    d_pt.copy_from_host(pt.data(), d_pt.bytes);

    bk::scatter_full_kv(d_sk.p, d_sv.p, scratch_ctx, 0, n, fp8, 0, static_cast<const std::int32_t*>(d_pt.p), max_pages,
                        d_pk.p, d_pv.p, nullptr);
    bk::gather_full_kv(d_pk.p, d_pv.p, 0, static_cast<const std::int32_t*>(d_pt.p), max_pages, n, fp8, d_gk.p, d_gv.p,
                       scratch_ctx, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<std::uint8_t> got_k(scratch_bytes), got_v(scratch_bytes);
    d_gk.copy_to_host(got_k.data(), scratch_bytes);
    d_gv.copy_to_host(got_v.data(), scratch_bytes);
    for (int fi = 0; fi < NUM_FULL_VISITS; ++fi) {
      const std::size_t off = std::size_t(fi) * scratch_ctx * KV_DIM * es;
      const std::size_t len = std::size_t(n) * KV_DIM * es;
      CHECK(std::memcmp(src_k.data() + off, got_k.data() + off, len) == 0);
      CHECK(std::memcmp(src_v.data() + off, got_v.data() + off, len) == 0);
    }

    // Verify partial (<64-token) tail copy helpers: scratch -> compact tail -> scratch & pool page.
    const int tail_pos0 = 128;
    const int tail_len = 37;
    const std::size_t tail_bytes = std::size_t(NUM_FULL_VISITS) * std::size_t(tail_len) * KV_DIM * es;
    DeviceBuffer d_tk(tail_bytes), d_tv(tail_bytes);
    d_tk.fill(0);
    d_tv.fill(0);
    d_gk.fill(0);
    d_gv.fill(0);
    bk::copy_tail_kv_from_scratch(d_sk.p, d_sv.p, scratch_ctx, tail_pos0, tail_len, fp8, d_tk.p, d_tv.p, nullptr);
    bk::copy_tail_kv_to_scratch(d_tk.p, d_tv.p, tail_len, fp8, d_gk.p, d_gv.p, scratch_ctx, tail_pos0, nullptr);
    d_pk.fill(0);
    d_pv.fill(0);
    bk::copy_tail_kv_to_pool_page(d_tk.p, d_tv.p, tail_len, fp8, pt[2], d_pk.p, d_pv.p, nullptr);
    DeviceBuffer d_gk2(scratch_bytes), d_gv2(scratch_bytes);
    d_gk2.fill(0);
    d_gv2.fill(0);
    bk::gather_full_kv(d_pk.p, d_pv.p, 0, static_cast<const std::int32_t*>(d_pt.p), max_pages, tail_pos0 + tail_len, fp8,
                       d_gk2.p, d_gv2.p, scratch_ctx, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_gk.copy_to_host(got_k.data(), scratch_bytes);
    d_gv.copy_to_host(got_v.data(), scratch_bytes);
    std::vector<std::uint8_t> got_k2(scratch_bytes), got_v2(scratch_bytes);
    d_gk2.copy_to_host(got_k2.data(), scratch_bytes);
    d_gv2.copy_to_host(got_v2.data(), scratch_bytes);
    for (int fi = 0; fi < NUM_FULL_VISITS; ++fi) {
      const std::size_t off = (std::size_t(fi) * scratch_ctx + tail_pos0) * KV_DIM * es;
      const std::size_t len = std::size_t(tail_len) * KV_DIM * es;
      CHECK(std::memcmp(src_k.data() + off, got_k.data() + off, len) == 0);
      CHECK(std::memcmp(src_v.data() + off, got_v.data() + off, len) == 0);
      CHECK(std::memcmp(src_k.data() + off, got_k2.data() + off, len) == 0);
      CHECK(std::memcmp(src_v.data() + off, got_v2.data() + off, len) == 0);
    }
  }
}

void test_static_conv_verify() {
  std::mt19937 rng(4242);
  const int max_slots = 24;
  const std::size_t slot_stride = 3 * H;
  const auto w_host = rand_floats(rng, H * 4, 0.25f);
  const auto st_init = rand_floats(rng, std::size_t(max_slots) * slot_stride, 0.4f);

  DeviceBuffer d_w(w_host.size() * sizeof(float));
  d_w.copy_from_host(w_host.data(), d_w.bytes);

  for (bool x_fp32 : {true, false}) {
    for (int W : {1, 2, 4, 8}) {
      for (int B : {1, 4, 16}) {
        DeviceBuffer d_st(st_init.size() * sizeof(float));
        d_st.copy_from_host(st_init.data(), d_st.bytes);

        std::vector<std::int32_t> slot_ids(std::size_t(B), -1);
        std::vector<std::int32_t> w_limits(std::size_t(B), W);
        for (int b = 0; b < B; ++b) {
          slot_ids[std::size_t(b)] = (B >= 4 && b == B - 1) ? -1 : (b + 1);
          w_limits[std::size_t(b)] = std::max(1, std::min(W, (b % W) + 1));
        }
        DeviceBuffer d_slots(slot_ids.size() * 4), d_wlim(w_limits.size() * 4);
        d_slots.copy_from_host(slot_ids.data(), d_slots.bytes);
        d_wlim.copy_from_host(w_limits.data(), d_wlim.bytes);

        const auto x_f32 = rand_floats(rng, std::size_t(B) * W * H, 0.5f);
        const auto x_act = to_act(x_f32);
        DeviceBuffer d_x(x_fp32 ? x_f32.size() * 4 : x_act.size() * sizeof(ActT));
        if (x_fp32) d_x.copy_from_host(x_f32.data(), d_x.bytes);
        else d_x.copy_from_host(x_act.data(), d_x.bytes);

        DeviceBuffer d_out(std::size_t(B) * W * H * 4);
        DeviceBuffer d_hist(std::size_t(B) * W * H * 4);
        d_out.fill(0);
        d_hist.fill(0);

        bk::batched_static_conv_verify(d_x.p, x_fp32, H, B, W, H, static_cast<const std::int32_t*>(d_slots.p),
                                       static_cast<const std::int32_t*>(d_wlim.p), static_cast<const float*>(d_w.p),
                                       static_cast<float*>(d_st.p), slot_stride, static_cast<float*>(d_out.p),
                                       static_cast<float*>(d_hist.p), nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> got_out(std::size_t(B) * W * H), got_hist(std::size_t(B) * W * H), got_st(st_init.size());
        d_out.copy_to_host(got_out.data(), d_out.bytes);
        d_hist.copy_to_host(got_hist.data(), d_hist.bytes);
        d_st.copy_to_host(got_st.data(), d_st.bytes);

        std::vector<float> exp_st = st_init;
        const float inv_sqrt5 = 1.f / 2.2360679774997896f;
        float max_err = 0.f;
        for (int b = 0; b < B; ++b) {
          const int s = slot_ids[std::size_t(b)];
          if (s < 0) continue;
          const int wlim = w_limits[std::size_t(b)];
          for (int c = 0; c < H; ++c) {
            float e0 = st_init[std::size_t(s) * slot_stride + 0 * H + c];
            float e1 = st_init[std::size_t(s) * slot_stride + 1 * H + c];
            float e2 = st_init[std::size_t(s) * slot_stride + 2 * H + c];
            const float* w4 = &w_host[std::size_t(c) * 4];
            for (int w = 0; w < wlim; ++w) {
              const std::size_t r = std::size_t(b) * W + w;
              const float e3 = ingine::act::host_round(x_f32[r * H + c]);
              const float acc = ((w4[0] * e0 + w4[1] * e1) + w4[2] * e2) + w4[3] * e3;
              const float exp = (acc + e3) * inv_sqrt5;
              max_err = std::max(max_err, std::fabs(got_out[r * H + c] - exp));
              CHECK(got_hist[r * H + c] == e3);
              e0 = e1;
              e1 = e2;
              e2 = e3;
            }
            exp_st[std::size_t(s) * slot_stride + 0 * H + c] = e0;
            exp_st[std::size_t(s) * slot_stride + 1 * H + c] = e1;
            exp_st[std::size_t(s) * slot_stride + 2 * H + c] = e2;
          }
        }
        CHECK(max_err <= 1e-6f);
        CHECK(got_st == exp_st);
      }
    }
  }
}

void test_batched_attn_verify_and_rollback(bool full, bool cache_fp8, bool proj_fp32, int B, int W) {
  std::mt19937 rng(7000 + B * 31 + W * 13 + (full ? 500 : 0) + (cache_fp8 ? 90 : 0) + (proj_fp32 ? 3 : 0));
  constexpr int NUM_VISITS = 48, NUM_SLIDING_VISITS = 40;
  const int R = full ? R_FULL : R_SLIDING;
  const int max_slots = B + 4;
  const int max_ctx = 512;
  const int max_pages_per_slot = (max_ctx + bk::kPageSize - 1) / bk::kPageSize;
  const int full_visit_idx = 2;
  const int visit_v = 5;       // arbitrary visit tested in full 48-visit layout
  const int ring_ri = 4;       // arbitrary sliding visit tested in 40-visit layout

  const auto q_scale = rand_floats(rng, HQ * D, 0.3f);
  const auto r_scale = rand_floats(rng, HQ * DREL, 0.3f);
  const auto rel_proj = rand_floats(rng, DREL * EXT, 0.2f);
  const auto k_conv = rand_floats(rng, KV_DIM * 4, 0.2f);
  const auto v_conv = rand_floats(rng, KV_DIM * 4, 0.2f);
  const auto attn_conv = rand_floats(rng, H * 4, 0.2f);
  const auto mlp_conv = rand_floats(rng, H * 4, 0.2f);

  DeviceBuffer d_qs(q_scale.size() * 4), d_rs(r_scale.size() * 4), d_rp(rel_proj.size() * 4);
  DeviceBuffer d_kc(k_conv.size() * 4), d_vc(v_conv.size() * 4), d_ac(attn_conv.size() * 4), d_mc(mlp_conv.size() * 4);
  d_qs.copy_from_host(q_scale.data(), d_qs.bytes);
  d_rs.copy_from_host(r_scale.data(), d_rs.bytes);
  d_rp.copy_from_host(rel_proj.data(), d_rp.bytes);
  d_kc.copy_from_host(k_conv.data(), d_kc.bytes);
  d_vc.copy_from_host(v_conv.data(), d_vc.bytes);
  d_ac.copy_from_host(attn_conv.data(), d_ac.bytes);
  d_mc.copy_from_host(mlp_conv.data(), d_mc.bytes);

  auto proj_f32 = rand_floats(rng, std::size_t(B) * W * R, 0.3f);
  for (float& x : proj_f32) x = ingine::act::host_round(x);
  const auto proj_act = to_act(proj_f32);
  DeviceBuffer d_proj(proj_fp32 ? proj_f32.size() * 4 : proj_act.size() * sizeof(ActT));
  if (proj_fp32) d_proj.copy_from_host(proj_f32.data(), d_proj.bytes);
  else d_proj.copy_from_host(proj_act.data(), d_proj.bytes);

  auto h_x_attn = rand_floats(rng, std::size_t(B) * W * H, 0.3f);
  auto h_x_mlp = rand_floats(rng, std::size_t(B) * W * H, 0.3f);
  for (float& x : h_x_attn) x = ingine::act::host_round(x);
  for (float& x : h_x_mlp) x = ingine::act::host_round(x);
  DeviceBuffer d_x_attn(h_x_attn.size() * 4), d_x_mlp(h_x_mlp.size() * 4);
  d_x_attn.copy_from_host(h_x_attn.data(), d_x_attn.bytes);
  d_x_mlp.copy_from_host(h_x_mlp.data(), d_x_mlp.bytes);

  // Heterogeneous slot_ids (including padded -1), boundary positions (P=1024 wrap, page crossings 62/63/127),
  // heterogeneous slot_w_limit, and forced zero/partial/full acceptance commit counts!
  std::vector<std::int32_t> slot_ids(std::size_t(B), -1);
  std::vector<std::int32_t> w_limits(std::size_t(B), W);
  std::vector<std::int32_t> commits(std::size_t(B), 1);
  std::vector<std::int64_t> slot_pos_init(std::size_t(max_slots), 0);
  const int sliding_positions[] = {1024, 1022, 1025, 3, 2047, 512, 1023, 1500};
  const int full_positions[] = {62, 63, 127, 5, 190, 255, 64, 319};
  for (int b = 0; b < B; ++b) {
    if (B >= 4 && b == B - 1) {
      slot_ids[std::size_t(b)] = -1;
      w_limits[std::size_t(b)] = W;
      commits[std::size_t(b)] = 1;
    } else {
      const int s = b + 1;
      slot_ids[std::size_t(b)] = s;
      slot_pos_init[std::size_t(s)] = full ? full_positions[b % 8] : sliding_positions[b % 8];
      w_limits[std::size_t(b)] = (b == 0) ? W : std::max(1, std::min(W, ((b + 1) % W) + 1));
      const int wl = w_limits[std::size_t(b)];
      // Mix of zero accepted (commit=1), partial acceptance, and full acceptance (commit=wl).
      if (b % 3 == 0) commits[std::size_t(b)] = 1;
      else if (b % 3 == 1) commits[std::size_t(b)] = wl;
      else commits[std::size_t(b)] = std::max(1, (wl + 1) / 2);
    }
  }
  DeviceBuffer d_slot_ids(slot_ids.size() * 4), d_wlim(w_limits.size() * 4), d_commits(commits.size() * 4);
  DeviceBuffer d_slot_pos(slot_pos_init.size() * 8);
  d_slot_ids.copy_from_host(slot_ids.data(), d_slot_ids.bytes);
  d_wlim.copy_from_host(w_limits.data(), d_wlim.bytes);
  d_commits.copy_from_host(commits.data(), d_commits.bytes);
  d_slot_pos.copy_from_host(slot_pos_init.data(), d_slot_pos.bytes);

  const std::size_t conv_kv_stride = std::size_t(NUM_VISITS) * 3 * KV_DIM;
  const std::size_t conv_h_stride = std::size_t(NUM_VISITS) * 3 * H;
  const std::size_t ring_stride = std::size_t(NUM_SLIDING_VISITS) * WIN * KV_DIM;
  const std::size_t ring_pos_stride = std::size_t(NUM_SLIDING_VISITS) * WIN;

  // Allocate full 48-visit conv & 40-visit sliding state buffers (matching BatchStatePool layout).
  DeviceBuffer d_conv_k(std::size_t(max_slots) * conv_kv_stride * 4);
  DeviceBuffer d_conv_v(std::size_t(max_slots) * conv_kv_stride * 4);
  DeviceBuffer d_conv_a(std::size_t(max_slots) * conv_h_stride * 4);
  DeviceBuffer d_conv_m(std::size_t(max_slots) * conv_h_stride * 4);
  d_conv_k.fill(0);
  d_conv_v.fill(0);
  d_conv_a.fill(0);
  d_conv_m.fill(0);

  // Populate visit_v conv states with random floats across all slots.
  std::vector<float> ck_v_init = rand_floats(rng, std::size_t(max_slots) * 3 * KV_DIM, 0.3f);
  std::vector<float> cv_v_init = rand_floats(rng, std::size_t(max_slots) * 3 * KV_DIM, 0.3f);
  std::vector<float> ca_v_init = rand_floats(rng, std::size_t(max_slots) * 3 * H, 0.3f);
  std::vector<float> cm_v_init = rand_floats(rng, std::size_t(max_slots) * 3 * H, 0.3f);
  for (int s = 0; s < max_slots; ++s) {
    CUDA_CHECK(cudaMemcpy(static_cast<float*>(d_conv_k.p) + std::size_t(s) * conv_kv_stride + std::size_t(visit_v) * 3 * KV_DIM,
                          ck_v_init.data() + std::size_t(s) * 3 * KV_DIM, 3 * KV_DIM * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(static_cast<float*>(d_conv_v.p) + std::size_t(s) * conv_kv_stride + std::size_t(visit_v) * 3 * KV_DIM,
                          cv_v_init.data() + std::size_t(s) * 3 * KV_DIM, 3 * KV_DIM * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(static_cast<float*>(d_conv_a.p) + std::size_t(s) * conv_h_stride + std::size_t(visit_v) * 3 * H,
                          ca_v_init.data() + std::size_t(s) * 3 * H, 3 * H * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(static_cast<float*>(d_conv_m.p) + std::size_t(s) * conv_h_stride + std::size_t(visit_v) * 3 * H,
                          cm_v_init.data() + std::size_t(s) * 3 * H, 3 * H * 4, cudaMemcpyHostToDevice));
  }

  // Sliding rings (always allocated so verify_save/verify_rollback exercise ring backup & restore).
  DeviceBuffer d_k_ring(std::size_t(max_slots) * ring_stride * sizeof(ActT));
  DeviceBuffer d_v_ring(std::size_t(max_slots) * ring_stride * sizeof(ActT));
  DeviceBuffer d_ring_pos(std::size_t(max_slots) * ring_pos_stride * 8);
  d_k_ring.fill(0);
  d_v_ring.fill(0);
  d_ring_pos.fill(0xff);

  std::vector<ActT> kr_ri_init = to_act(rand_floats(rng, std::size_t(max_slots) * WIN * KV_DIM, 0.25f));
  std::vector<ActT> vr_ri_init = to_act(rand_floats(rng, std::size_t(max_slots) * WIN * KV_DIM, 0.25f));
  std::vector<std::int64_t> rp_ri_init(std::size_t(max_slots) * WIN, -1);
  for (int b = 0; b < B; ++b) {
    const int s = slot_ids[std::size_t(b)];
    if (s < 0) continue;
    const int p0 = int(slot_pos_init[std::size_t(s)]);
    const int start = std::max(0, p0 - WIN);
    for (int p = start; p < p0; ++p) rp_ri_init[std::size_t(s) * WIN + (p & (WIN - 1))] = p;
  }
  for (int s = 0; s < max_slots; ++s) {
    CUDA_CHECK(cudaMemcpy(static_cast<ActT*>(d_k_ring.p) + std::size_t(s) * ring_stride + std::size_t(ring_ri) * WIN * KV_DIM,
                          kr_ri_init.data() + std::size_t(s) * WIN * KV_DIM, WIN * KV_DIM * sizeof(ActT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(static_cast<ActT*>(d_v_ring.p) + std::size_t(s) * ring_stride + std::size_t(ring_ri) * WIN * KV_DIM,
                          vr_ri_init.data() + std::size_t(s) * WIN * KV_DIM, WIN * KV_DIM * sizeof(ActT),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(static_cast<std::int64_t*>(d_ring_pos.p) + std::size_t(s) * ring_pos_stride + std::size_t(ring_ri) * WIN,
                          rp_ri_init.data() + std::size_t(s) * WIN, WIN * 8, cudaMemcpyHostToDevice));
  }

  // Full paged KV pool (when full == true)
  const int num_pages = max_slots * max_pages_per_slot;
  const std::size_t es = cache_fp8 ? 1 : 2;
  DeviceBuffer d_k_pool, d_v_pool, d_pt;
  std::vector<std::int32_t> pt_host(std::size_t(max_slots) * max_pages_per_slot, -1);
  std::vector<std::uint8_t> k_full_slot_init, v_full_slot_init;
  if (full) {
    std::vector<int> perm(static_cast<std::size_t>(num_pages), 0);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int s = 0; s < max_slots; ++s)
      for (int lp = 0; lp < max_pages_per_slot; ++lp)
        pt_host[std::size_t(s) * max_pages_per_slot + lp] = perm[std::size_t(s) * max_pages_per_slot + lp];
    d_pt = DeviceBuffer(pt_host.size() * 4);
    d_pt.copy_from_host(pt_host.data(), d_pt.bytes);

    const std::size_t page_bytes = std::size_t(NUM_FULL_VISITS) * bk::kPageSize * KV_DIM * es;
    d_k_pool = DeviceBuffer(std::size_t(num_pages) * page_bytes);
    d_v_pool = DeviceBuffer(std::size_t(num_pages) * page_bytes);
    d_k_pool.fill(0);
    d_v_pool.fill(0);

    const std::size_t slot_full_bytes = std::size_t(max_ctx) * KV_DIM * es;
    k_full_slot_init.resize(std::size_t(max_slots) * slot_full_bytes, 0);
    v_full_slot_init.resize(std::size_t(max_slots) * slot_full_bytes, 0);
    DeviceBuffer d_tmp_k(std::size_t(NUM_FULL_VISITS) * slot_full_bytes);
    DeviceBuffer d_tmp_v(std::size_t(NUM_FULL_VISITS) * slot_full_bytes);
    for (int b = 0; b < B; ++b) {
      const int s = slot_ids[std::size_t(b)];
      if (s < 0) continue;
      const int p0 = int(slot_pos_init[std::size_t(s)]);
      const auto kf = rand_floats(rng, std::size_t(p0) * KV_DIM, 0.25f);
      const auto vf = rand_floats(rng, std::size_t(p0) * KV_DIM, 0.25f);
      if (cache_fp8) {
        const auto ke = to_e4m3(kf), ve = to_e4m3(vf);
        std::memcpy(k_full_slot_init.data() + std::size_t(s) * slot_full_bytes, ke.data(), ke.size());
        std::memcpy(v_full_slot_init.data() + std::size_t(s) * slot_full_bytes, ve.data(), ve.size());
      } else {
        const auto ka = to_act(kf), va = to_act(vf);
        std::memcpy(k_full_slot_init.data() + std::size_t(s) * slot_full_bytes, ka.data(), ka.size() * sizeof(ActT));
        std::memcpy(v_full_slot_init.data() + std::size_t(s) * slot_full_bytes, va.data(), va.size() * sizeof(ActT));
      }
      d_tmp_k.fill(0);
      d_tmp_v.fill(0);
      CUDA_CHECK(cudaMemcpy(static_cast<char*>(d_tmp_k.p) + std::size_t(full_visit_idx) * slot_full_bytes,
                            k_full_slot_init.data() + std::size_t(s) * slot_full_bytes, std::size_t(p0) * KV_DIM * es,
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(static_cast<char*>(d_tmp_v.p) + std::size_t(full_visit_idx) * slot_full_bytes,
                            v_full_slot_init.data() + std::size_t(s) * slot_full_bytes, std::size_t(p0) * KV_DIM * es,
                            cudaMemcpyHostToDevice));
      bk::scatter_full_kv(d_tmp_k.p, d_tmp_v.p, max_ctx, 0, p0, cache_fp8, s, static_cast<const std::int32_t*>(d_pt.p),
                          max_pages_per_slot, d_k_pool.p, d_v_pool.p, nullptr);
    }
  }

  // Backup and history buffers
  DeviceBuffer bk_ck(std::size_t(B) * NUM_VISITS * 2 * KV_DIM * 4);
  DeviceBuffer bk_cv(std::size_t(B) * NUM_VISITS * 2 * KV_DIM * 4);
  DeviceBuffer bk_ca(std::size_t(B) * NUM_VISITS * 2 * H * 4);
  DeviceBuffer bk_cm(std::size_t(B) * NUM_VISITS * 2 * H * 4);
  DeviceBuffer bk_rk(std::size_t(B) * NUM_SLIDING_VISITS * W * KV_DIM * sizeof(ActT));
  DeviceBuffer bk_rv(std::size_t(B) * NUM_SLIDING_VISITS * W * KV_DIM * sizeof(ActT));
  DeviceBuffer bk_rp(std::size_t(B) * NUM_SLIDING_VISITS * W * 8);

  const std::size_t kv_v_stride = std::size_t(B) * W * KV_DIM;
  const std::size_t h_v_stride = std::size_t(B) * W * H;
  DeviceBuffer d_hk(std::size_t(NUM_VISITS) * kv_v_stride * 4);
  DeviceBuffer d_hv(std::size_t(NUM_VISITS) * kv_v_stride * 4);
  DeviceBuffer d_ha(std::size_t(NUM_VISITS) * h_v_stride * 4);
  DeviceBuffer d_hm(std::size_t(NUM_VISITS) * h_v_stride * 4);
  d_hk.fill(0);
  d_hv.fill(0);
  d_ha.fill(0);
  d_hm.fill(0);

  bk::BatchedVerifyStatePtrs st_ptrs{};
  st_ptrs.conv_k = static_cast<float*>(d_conv_k.p);
  st_ptrs.conv_v = static_cast<float*>(d_conv_v.p);
  st_ptrs.conv_attn = static_cast<float*>(d_conv_a.p);
  st_ptrs.conv_mlp = static_cast<float*>(d_conv_m.p);
  st_ptrs.conv_kv_slot_stride = conv_kv_stride;
  st_ptrs.conv_h_slot_stride = conv_h_stride;
  st_ptrs.k_ring = static_cast<ActT*>(d_k_ring.p);
  st_ptrs.v_ring = static_cast<ActT*>(d_v_ring.p);
  st_ptrs.ring_pos = static_cast<std::int64_t*>(d_ring_pos.p);
  st_ptrs.ring_slot_stride = ring_stride;
  st_ptrs.ring_pos_slot_stride = ring_pos_stride;
  st_ptrs.slot_pos = static_cast<std::int64_t*>(d_slot_pos.p);

  bk::BatchedVerifyBackup bk_buf{};
  bk_buf.conv_k = static_cast<float*>(bk_ck.p);
  bk_buf.conv_v = static_cast<float*>(bk_cv.p);
  bk_buf.conv_attn = static_cast<float*>(bk_ca.p);
  bk_buf.conv_mlp = static_cast<float*>(bk_cm.p);
  bk_buf.ring_k = static_cast<ActT*>(bk_rk.p);
  bk_buf.ring_v = static_cast<ActT*>(bk_rv.p);
  bk_buf.ring_pos = static_cast<std::int64_t*>(bk_rp.p);
  bk_buf.max_W = W;

  bk::BatchedVerifyHistory hist_buf{};
  hist_buf.hist_k = static_cast<const float*>(d_hk.p);
  hist_buf.hist_v = static_cast<const float*>(d_hv.p);
  hist_buf.hist_attn = static_cast<const float*>(d_ha.p);
  hist_buf.hist_mlp = static_cast<const float*>(d_hm.p);
  hist_buf.kv_visit_stride = kv_v_stride;
  hist_buf.h_visit_stride = h_v_stride;

  // 1. Save pre-verify state
  bk::batched_verify_save(st_ptrs, bk_buf, static_cast<const std::int32_t*>(d_slot_ids.p),
                          static_cast<const std::int32_t*>(d_wlim.p), B, W, nullptr);

  // 2. Run batched_attn_verify + batched_static_conv_verify (attn & mlp) on visit_v
  DeviceBuffer d_ws(bk::attn_verify_workspace_bytes(B, W));
  d_ws.fill(0);
  DeviceBuffer d_out(std::size_t(B) * W * HQ * D * sizeof(ActT));
  DeviceBuffer d_ya(std::size_t(B) * W * H * 4), d_ym(std::size_t(B) * W * H * 4);
  d_out.fill(0);

  bk::BatchedVerifyAttnArgs va;
  va.proj = d_proj.p;
  va.proj_fp32 = proj_fp32;
  va.B = B;
  va.W = W;
  va.slot_ids = static_cast<const std::int32_t*>(d_slot_ids.p);
  va.slot_w_limit = static_cast<const std::int32_t*>(d_wlim.p);
  va.slot_pos = static_cast<const std::int64_t*>(d_slot_pos.p);
  va.q_scale = static_cast<const float*>(d_qs.p);
  va.r_scale = static_cast<const float*>(d_rs.p);
  va.rel_proj = static_cast<const float*>(d_rp.p);
  va.k_conv = static_cast<const float*>(d_kc.p);
  va.v_conv = static_cast<const float*>(d_vc.p);
  va.conv_k_visit = static_cast<float*>(d_conv_k.p) + std::size_t(visit_v) * 3 * KV_DIM;
  va.conv_v_visit = static_cast<float*>(d_conv_v.p) + std::size_t(visit_v) * 3 * KV_DIM;
  va.conv_slot_stride = conv_kv_stride;
  va.hist_k_visit = static_cast<float*>(d_hk.p) + std::size_t(visit_v) * kv_v_stride;
  va.hist_v_visit = static_cast<float*>(d_hv.p) + std::size_t(visit_v) * kv_v_stride;
  if (!full) {
    va.k_ring_visit = static_cast<ActT*>(d_k_ring.p) + std::size_t(ring_ri) * WIN * KV_DIM;
    va.v_ring_visit = static_cast<ActT*>(d_v_ring.p) + std::size_t(ring_ri) * WIN * KV_DIM;
    va.ring_pos_visit = static_cast<std::int64_t*>(d_ring_pos.p) + std::size_t(ring_ri) * WIN;
    va.ring_slot_stride = ring_stride;
    va.ring_pos_slot_stride = ring_pos_stride;
    va.window = WIN;
    va.log_scaling = false;
  } else {
    va.k_full_pool = d_k_pool.p;
    va.v_full_pool = d_v_pool.p;
    va.page_table = static_cast<const std::int32_t*>(d_pt.p);
    va.max_pages_per_slot = max_pages_per_slot;
    va.full_visit_idx = full_visit_idx;
    va.max_ctx = max_ctx;
    va.cache_fp8 = cache_fp8;
    va.window = 0;
    va.log_scaling = true;
  }
  va.workspace = d_ws.p;
  va.workspace_size = d_ws.bytes;
  va.out = static_cast<ActT*>(d_out.p);
  bk::batched_attn_verify(va, nullptr);

  bk::batched_static_conv_verify(d_x_attn.p, true, H, B, W, H, static_cast<const std::int32_t*>(d_slot_ids.p),
                                 static_cast<const std::int32_t*>(d_wlim.p), static_cast<const float*>(d_ac.p),
                                 static_cast<float*>(d_conv_a.p) + std::size_t(visit_v) * 3 * H, conv_h_stride,
                                 static_cast<float*>(d_ya.p),
                                 static_cast<float*>(d_ha.p) + std::size_t(visit_v) * h_v_stride, nullptr);
  bk::batched_static_conv_verify(d_x_mlp.p, true, H, B, W, H, static_cast<const std::int32_t*>(d_slot_ids.p),
                                 static_cast<const std::int32_t*>(d_wlim.p), static_cast<const float*>(d_mc.p),
                                 static_cast<float*>(d_conv_m.p) + std::size_t(visit_v) * 3 * H, conv_h_stride,
                                 static_cast<float*>(d_ym.p),
                                 static_cast<float*>(d_hm.p) + std::size_t(visit_v) * h_v_stride, nullptr);

  // 3. Roll back to heterogeneous commit_counts[b] and advance slot_pos once per slot.
  bk::batched_verify_rollback(st_ptrs, bk_buf, hist_buf, static_cast<const std::int32_t*>(d_slot_ids.p),
                              static_cast<const std::int32_t*>(d_wlim.p), static_cast<const std::int32_t*>(d_commits.p),
                              B, W, nullptr);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ActT> got_out(std::size_t(B) * W * HQ * D);
  d_out.copy_to_host(got_out.data(), d_out.bytes);
  std::vector<std::int64_t> got_pos(static_cast<std::size_t>(max_slots), 0);
  d_slot_pos.copy_to_host(got_pos.data(), d_slot_pos.bytes);

  // 4. Serial 1-token oracle replay for each active slot:
  //    - Replay w_lim steps to verify ALL w_lim attention outputs match!
  //    - Snapshot state at step == commit[b] to verify post-rollback conv_k/v/attn/mlp, ring K/V/pos, and slot_pos
  //      match BIT-FOR-BIT!
  DeviceBuffer ref_ws(attn::workspace_bytes(1));
  ref_ws.fill(0);
  DeviceBuffer ref_proj(std::size_t(R) * 4), ref_pos(8), ref_out(std::size_t(HQ) * D * sizeof(ActT));
  DeviceBuffer ref_ck(3 * KV_DIM * 4), ref_cv(3 * KV_DIM * 4);
  DeviceBuffer ref_kc(full ? std::size_t(max_ctx) * KV_DIM * es : std::size_t(WIN) * KV_DIM * sizeof(ActT));
  DeviceBuffer ref_vc(full ? std::size_t(max_ctx) * KV_DIM * es : std::size_t(WIN) * KV_DIM * sizeof(ActT));
  DeviceBuffer ref_kpos(WIN * 8);

  float max_err = 0.f;
  for (int b = 0; b < B; ++b) {
    const int s = slot_ids[std::size_t(b)];
    if (s < 0) continue;
    const int wlim = w_limits[std::size_t(b)];
    const int commit = commits[std::size_t(b)];
    const std::int64_t p0 = slot_pos_init[std::size_t(s)];
    CHECK(got_pos[std::size_t(s)] == p0 + commit);

    ref_ck.copy_from_host(ck_v_init.data() + std::size_t(s) * 3 * KV_DIM, ref_ck.bytes);
    ref_cv.copy_from_host(cv_v_init.data() + std::size_t(s) * 3 * KV_DIM, ref_cv.bytes);
    if (!full) {
      ref_kc.copy_from_host(kr_ri_init.data() + std::size_t(s) * WIN * KV_DIM, ref_kc.bytes);
      ref_vc.copy_from_host(vr_ri_init.data() + std::size_t(s) * WIN * KV_DIM, ref_vc.bytes);
      ref_kpos.copy_from_host(rp_ri_init.data() + std::size_t(s) * WIN, ref_kpos.bytes);
    } else {
      const std::size_t slot_full_bytes = std::size_t(max_ctx) * KV_DIM * es;
      ref_kc.copy_from_host(k_full_slot_init.data() + std::size_t(s) * slot_full_bytes, slot_full_bytes);
      ref_vc.copy_from_host(v_full_slot_init.data() + std::size_t(s) * slot_full_bytes, slot_full_bytes);
    }

    std::vector<float> exp_ck_commit, exp_cv_commit;
    std::vector<ActT> exp_kr_commit, exp_vr_commit;
    std::vector<std::int64_t> exp_rp_commit;

    for (int w = 0; w < wlim; ++w) {
      const std::int64_t pw = p0 + w;
      ref_proj.copy_from_host(proj_f32.data() + std::size_t(b * W + w) * R, ref_proj.bytes);
      ref_pos.copy_from_host(&pw, 8);
      attn::DecodeArgs da;
      da.proj = static_cast<const float*>(ref_proj.p);
      da.M = 1;
      da.q_scale = static_cast<const float*>(d_qs.p);
      da.r_scale = static_cast<const float*>(d_rs.p);
      da.rel_proj = static_cast<const float*>(d_rp.p);
      da.k_conv = static_cast<const float*>(d_kc.p);
      da.v_conv = static_cast<const float*>(d_vc.p);
      da.conv_k_state = static_cast<float*>(ref_ck.p);
      da.conv_v_state = static_cast<float*>(ref_cv.p);
      da.k_cache = ref_kc.p;
      da.v_cache = ref_vc.p;
      da.cache_fp8 = cache_fp8;
      da.cache_rows = full ? max_ctx : WIN;
      da.kpos = full ? nullptr : static_cast<std::int64_t*>(ref_kpos.p);
      da.pos = static_cast<const std::int64_t*>(ref_pos.p);
      da.window = full ? 0 : WIN;
      da.log_scaling = full;
      da.workspace = ref_ws.p;
      da.workspace_size = ref_ws.bytes;
      da.out = static_cast<ActT*>(ref_out.p);
      attn::attn_decode(da, nullptr);
      CUDA_CHECK(cudaDeviceSynchronize());

      std::vector<ActT> exp_out(HQ * D);
      ref_out.copy_to_host(exp_out.data(), ref_out.bytes);
      for (int i = 0; i < HQ * D; ++i) {
        const float g = ingine::act::host_to_float(got_out[std::size_t(b * W + w) * HQ * D + i]);
        const float e = ingine::act::host_to_float(exp_out[std::size_t(i)]);
        max_err = std::max(max_err, std::fabs(g - e));
      }
      if (w + 1 == commit) {
        exp_ck_commit.resize(3 * KV_DIM);
        exp_cv_commit.resize(3 * KV_DIM);
        ref_ck.copy_to_host(exp_ck_commit.data(), ref_ck.bytes);
        ref_cv.copy_to_host(exp_cv_commit.data(), ref_cv.bytes);
        if (!full) {
          exp_kr_commit.resize(WIN * KV_DIM);
          exp_vr_commit.resize(WIN * KV_DIM);
          exp_rp_commit.resize(WIN);
          ref_kc.copy_to_host(exp_kr_commit.data(), ref_kc.bytes);
          ref_vc.copy_to_host(exp_vr_commit.data(), ref_vc.bytes);
          ref_kpos.copy_to_host(exp_rp_commit.data(), ref_kpos.bytes);
        }
      }
    }

    // Compare post-rollback conv_k, conv_v, conv_attn, conv_mlp, and (sliding) k_ring, v_ring, ring_pos BIT-EXACT!
    std::vector<float> got_ck(3 * KV_DIM), got_cv(3 * KV_DIM), got_ca(3 * H), got_cm(3 * H);
    CUDA_CHECK(cudaMemcpy(got_ck.data(),
                          static_cast<const float*>(d_conv_k.p) + std::size_t(s) * conv_kv_stride + std::size_t(visit_v) * 3 * KV_DIM,
                          3 * KV_DIM * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_cv.data(),
                          static_cast<const float*>(d_conv_v.p) + std::size_t(s) * conv_kv_stride + std::size_t(visit_v) * 3 * KV_DIM,
                          3 * KV_DIM * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_ca.data(),
                          static_cast<const float*>(d_conv_a.p) + std::size_t(s) * conv_h_stride + std::size_t(visit_v) * 3 * H,
                          3 * H * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_cm.data(),
                          static_cast<const float*>(d_conv_m.p) + std::size_t(s) * conv_h_stride + std::size_t(visit_v) * 3 * H,
                          3 * H * 4, cudaMemcpyDeviceToHost));
    CHECK(got_ck == exp_ck_commit);
    CHECK(got_cv == exp_cv_commit);

    // Expected conv_attn & conv_mlp after exactly `commit` steps
    std::vector<float> exp_ca(ca_v_init.begin() + std::size_t(s) * 3 * H, ca_v_init.begin() + std::size_t(s + 1) * 3 * H);
    std::vector<float> exp_cm(cm_v_init.begin() + std::size_t(s) * 3 * H, cm_v_init.begin() + std::size_t(s + 1) * 3 * H);
    for (int w = 0; w < commit; ++w) {
      for (int c = 0; c < H; ++c) {
        exp_ca[0 * H + c] = exp_ca[1 * H + c];
        exp_ca[1 * H + c] = exp_ca[2 * H + c];
        exp_ca[2 * H + c] = h_x_attn[std::size_t(b * W + w) * H + c];
        exp_cm[0 * H + c] = exp_cm[1 * H + c];
        exp_cm[1 * H + c] = exp_cm[2 * H + c];
        exp_cm[2 * H + c] = h_x_mlp[std::size_t(b * W + w) * H + c];
      }
    }
    CHECK(got_ca == exp_ca);
    CHECK(got_cm == exp_cm);

    if (!full) {
      std::vector<ActT> got_kr(WIN * KV_DIM), got_vr(WIN * KV_DIM);
      std::vector<std::int64_t> got_rp(WIN);
      CUDA_CHECK(cudaMemcpy(got_kr.data(),
                            static_cast<const ActT*>(d_k_ring.p) + std::size_t(s) * ring_stride + std::size_t(ring_ri) * WIN * KV_DIM,
                            WIN * KV_DIM * sizeof(ActT), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(got_vr.data(),
                            static_cast<const ActT*>(d_v_ring.p) + std::size_t(s) * ring_stride + std::size_t(ring_ri) * WIN * KV_DIM,
                            WIN * KV_DIM * sizeof(ActT), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(got_rp.data(),
                            static_cast<const std::int64_t*>(d_ring_pos.p) + std::size_t(s) * ring_pos_stride + std::size_t(ring_ri) * WIN,
                            WIN * 8, cudaMemcpyDeviceToHost));
      CHECK(std::memcmp(got_kr.data(), exp_kr_commit.data(), got_kr.size() * sizeof(ActT)) == 0);
      CHECK(std::memcmp(got_vr.data(), exp_vr_commit.data(), got_vr.size() * sizeof(ActT)) == 0);
      CHECK(got_rp == exp_rp_commit);
    }
  }
  CHECK(got_pos[0] == 0);  // unused slot 0 untouched
  CHECK(max_err <= 2e-2f);
  if (max_err > 2e-2f) {
    std::fprintf(stderr, "verify attn mismatch (full=%d fp8=%d proj_fp32=%d B=%d W=%d): max_err=%g\n",
                 int(full), int(cache_fp8), int(proj_fp32), B, W, max_err);
  }
}

}  // namespace

int main() {
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
    std::fprintf(stderr, "batch_kernels: no CUDA device, skipping\n");
    return 77;
  }
  test_static_conv();
  for (int B : {1, 4, 16, 64}) {
    for (bool proj_fp32 : {true, false}) {
      test_batched_attn_vs_single(false, false, proj_fp32, B);  // sliding
      test_batched_attn_vs_single(true, false, proj_fp32, B);   // paged full BF16/FP16
      test_batched_attn_vs_single(true, true, proj_fp32, B);    // paged full FP8
    }
  }
  test_head_softcap_and_scatter_gather();
  test_static_conv_verify();
  for (int W : {1, 2, 4, 8}) {
    for (int B : {1, 4, 16}) {
      for (bool proj_fp32 : {true, false}) {
        test_batched_attn_verify_and_rollback(false, false, proj_fp32, B, W);  // sliding + wrap P=1024
        test_batched_attn_verify_and_rollback(true, false, proj_fp32, B, W);   // paged full BF16/FP16 + page cross
        test_batched_attn_verify_and_rollback(true, true, proj_fp32, B, W);    // paged full FP8 + page cross
      }
    }
  }
  std::printf(g_failures ? "FAILED (%d checks)\n" : "batch_kernels ok\n", g_failures);
  return g_failures ? 1 : 0;
}

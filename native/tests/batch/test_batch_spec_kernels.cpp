// Unit tests for multi-row speculative verification kernels in kernels/batch/ (§5.2):
// 1. batch_spec_conv: batched_static_conv_rows vs m sequential batched_static_conv_step calls (bit-identical),
//    including ragged m in {1..4}, padded slots (-1), and both x_fp32 true/false.
// 2. batch_spec_attn: batched_attn_decode_rows vs m sequential batched_attn_decode steps across 3 active slots +
//    padded rows, covering sliding layers crossing a ring wrap (p mod 1024 near 1020) and full layers crossing a
//    page boundary (p mod 64 near 62) in ActT, FP8, and FP8K+FP4V modes.
// 3. batch_spec_rollback: batch_verify_save + multi-row conv/attn + batch_verify_rollback + batched_advance_pos_by
//    for every accept count a in {0..m-1}, memcmp-verifying conv states, sliding rings, ring_pos, and slot_pos
//    against running a+1 single-row steps.
#include "core/act.h"
#include "core/arena.h"
#include "core/device.h"
#include "kernels/batch/batch.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

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

// ----------------------------------------------------------------------------- 1. batch_spec_conv
void test_batch_spec_conv() {
  std::mt19937 rng(101);
  const int max_slots = 6;
  const std::size_t slot_stride = 3 * H;
  const auto w_host = rand_floats(rng, H * 4, 0.25f);
  const auto st_init = rand_floats(rng, std::size_t(max_slots) * slot_stride, 0.4f);

  DeviceBuffer d_w(w_host.size() * sizeof(float));
  d_w.copy_from_host(w_host.data(), d_w.bytes);

  // 3 active slots with ragged m = {1, 4, 3} + 1 padded slot (-1) -> R = 10 (including 2 padded rows)
  const std::vector<std::int32_t> active_slots = {3, -1, 1, 4};
  const std::vector<std::int32_t> slot_row0 = {0, 1, 1, 5};
  const std::vector<std::int32_t> slot_m = {1, 0, 4, 3};
  const int S = int(active_slots.size());
  const int R = 10;

  DeviceBuffer d_as(S * 4), d_r0(S * 4), d_sm(S * 4);
  d_as.copy_from_host(active_slots.data(), d_as.bytes);
  d_r0.copy_from_host(slot_row0.data(), d_r0.bytes);
  d_sm.copy_from_host(slot_m.data(), d_sm.bytes);

  for (bool x_fp32 : {true, false}) {
    const auto x_f32 = rand_floats(rng, std::size_t(R) * H, 0.5f);
    const auto x_act = to_act(x_f32);
    DeviceBuffer d_x(x_fp32 ? x_f32.size() * 4 : x_act.size() * sizeof(ActT));
    if (x_fp32) d_x.copy_from_host(x_f32.data(), d_x.bytes);
    else d_x.copy_from_host(x_act.data(), d_x.bytes);

    DeviceBuffer d_st_multi(st_init.size() * 4), d_st_ref(st_init.size() * 4);
    d_st_multi.copy_from_host(st_init.data(), d_st_multi.bytes);
    d_st_ref.copy_from_host(st_init.data(), d_st_ref.bytes);

    DeviceBuffer d_out_multi(std::size_t(R) * H * 4), d_hist(std::size_t(R) * H * 4);
    d_out_multi.fill(0);
    d_hist.fill(0);

    bk::batched_static_conv_rows(d_x.p, x_fp32, H, S, H, static_cast<const std::int32_t*>(d_as.p),
                                 static_cast<const std::int32_t*>(d_r0.p), static_cast<const std::int32_t*>(d_sm.p),
                                 static_cast<const float*>(d_w.p), static_cast<float*>(d_st_multi.p), slot_stride,
                                 static_cast<float*>(d_out_multi.p), static_cast<float*>(d_hist.p), nullptr);

    // Reference: step-by-step batched_static_conv_step
    DeviceBuffer d_out_ref(std::size_t(R) * H * 4);
    d_out_ref.fill(0);
    DeviceBuffer d_one_slot(4);
    for (int i = 0; i < S; ++i) {
      if (active_slots[i] < 0) continue;
      d_one_slot.copy_from_host(&active_slots[i], 4);
      for (int j = 0; j < slot_m[i]; ++j) {
        const int r = slot_row0[i] + j;
        const void* xp = x_fp32 ? static_cast<const void*>(static_cast<const float*>(d_x.p) + std::size_t(r) * H)
                                : static_cast<const void*>(static_cast<const ActT*>(d_x.p) + std::size_t(r) * H);
        bk::batched_static_conv_step(xp, x_fp32, H, 1, H, static_cast<const std::int32_t*>(d_one_slot.p),
                                     static_cast<const float*>(d_w.p), static_cast<float*>(d_st_ref.p), slot_stride,
                                     static_cast<float*>(d_out_ref.p) + std::size_t(r) * H, nullptr);
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> om(R * H), oref(R * H), stm(st_init.size()), stref(st_init.size()), hist(R * H);
    d_out_multi.copy_to_host(om.data(), d_out_multi.bytes);
    d_out_ref.copy_to_host(oref.data(), d_out_ref.bytes);
    d_st_multi.copy_to_host(stm.data(), d_st_multi.bytes);
    d_st_ref.copy_to_host(stref.data(), d_st_ref.bytes);
    d_hist.copy_to_host(hist.data(), d_hist.bytes);

    CHECK(std::memcmp(om.data(), oref.data(), om.size() * 4) == 0);
    CHECK(std::memcmp(stm.data(), stref.data(), stm.size() * 4) == 0);
    for (int i = 0; i < S; ++i) {
      if (active_slots[i] < 0) continue;
      for (int j = 0; j < slot_m[i]; ++j) {
        const int r = slot_row0[i] + j;
        for (int c = 0; c < H; ++c) {
          const float exp_h = ingine::act::host_round(x_f32[std::size_t(r) * H + c]);
          if (hist[std::size_t(r) * H + c] != exp_h) {
            CHECK(false);
            return;
          }
        }
      }
    }
  }
  std::printf("batch_spec_conv: PASS\n");
}

// ----------------------------------------------------------------------------- 2. batch_spec_attn
void test_batch_spec_attn_case(bool full, bool cache_fp8, bool cache_v4, bool proj_fp32) {
  std::mt19937 rng(2026 + (full ? 100 : 0) + (cache_fp8 ? 20 : 0) + (cache_v4 ? 40 : 0) + (proj_fp32 ? 3 : 0));
  const int proj_R = full ? R_FULL : R_SLIDING;
  const int max_slots = 5;
  const int max_ctx = 512;
  const int max_pages_per_slot = (max_ctx + bk::kPageSize - 1) / bk::kPageSize;
  const int full_visit_idx = 5;

  // 3 active slots with ragged m = {4, 2, 3} + padded row -> R = 10 (total slots S = 4 with one -1)
  // Positions specifically hit ring wrap (1022, 2046) and page boundary (62, 127)!
  const std::vector<std::int32_t> active_slots = {1, 3, 2, -1};
  const std::vector<std::int32_t> slot_row0 = {0, 4, 6, 9};
  const std::vector<std::int32_t> slot_m = {4, 2, 3, 0};
  const int S = 4, R = 10;
  std::vector<std::int32_t> row_slot(R, -1), row_off(R, 0), row_base(R, 0);
  for (int i = 0; i < S; ++i) {
    if (active_slots[i] < 0) continue;
    for (int j = 0; j < slot_m[i]; ++j) {
      const int r = slot_row0[i] + j;
      row_slot[r] = active_slots[i];
      row_off[r] = j;
      row_base[r] = slot_row0[i];
    }
  }

  std::vector<std::int64_t> slot_pos_init(max_slots, 0);
  if (full) {
    slot_pos_init[1] = 62;   // crosses page boundary 63 -> 64 (m=4 -> 62,63,64,65)
    slot_pos_init[3] = 127;  // crosses page boundary 127 -> 128 (m=2 -> 127,128)
    slot_pos_init[2] = 0;    // starts at position 0 (m=3 -> 0,1,2)
  } else {
    slot_pos_init[1] = 1022;  // crosses ring wrap 1023 -> 1024 (m=4 -> 1022,1023,1024,1025)
    slot_pos_init[3] = 2046;  // full ring wrap (m=2 -> 2046,2047; overwrites 1022,1023)
    slot_pos_init[2] = 3;     // small non-full ring (m=3 -> 3,4,5)
  }

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

  auto proj_f32 = rand_floats(rng, std::size_t(R) * proj_R, 0.3f);
  for (float& x : proj_f32) x = ingine::act::host_round(x);
  const auto proj_act = to_act(proj_f32);
  DeviceBuffer d_proj(proj_fp32 ? proj_f32.size() * 4 : proj_act.size() * sizeof(ActT));
  if (proj_fp32) d_proj.copy_from_host(proj_f32.data(), d_proj.bytes);
  else d_proj.copy_from_host(proj_act.data(), d_proj.bytes);

  const std::size_t conv_stride = 3 * KV_DIM;
  const auto ck_init = rand_floats(rng, std::size_t(max_slots) * conv_stride, 0.3f);
  const auto cv_init = rand_floats(rng, std::size_t(max_slots) * conv_stride, 0.3f);

  // Two identical state copies: multi-row vs step-by-step reference
  DeviceBuffer d_ck_m(ck_init.size() * 4), d_cv_m(cv_init.size() * 4);
  DeviceBuffer d_ck_r(ck_init.size() * 4), d_cv_r(cv_init.size() * 4);
  d_ck_m.copy_from_host(ck_init.data(), d_ck_m.bytes);
  d_cv_m.copy_from_host(cv_init.data(), d_cv_m.bytes);
  d_ck_r.copy_from_host(ck_init.data(), d_ck_r.bytes);
  d_cv_r.copy_from_host(cv_init.data(), d_cv_r.bytes);

  const std::size_t ring_stride = std::size_t(WIN) * KV_DIM;
  const std::size_t ring_pos_stride = WIN;
  DeviceBuffer d_kr_m, d_vr_m, d_rp_m, d_kr_r, d_vr_r, d_rp_r;
  DeviceBuffer d_kp_m, d_vp_m, d_kp_r, d_vp_r, d_pt;

  if (!full) {
    auto kr_h = to_act(rand_floats(rng, std::size_t(max_slots) * ring_stride, 0.3f));
    auto vr_h = to_act(rand_floats(rng, std::size_t(max_slots) * ring_stride, 0.3f));
    std::vector<std::int64_t> rp_h(std::size_t(max_slots) * ring_pos_stride, -1);
    for (int s = 0; s < max_slots; ++s) {
      const std::int64_t p0 = slot_pos_init[s];
      const std::int64_t lo = std::max<std::int64_t>(0, p0 - WIN);
      for (std::int64_t pos = lo; pos < p0; ++pos) rp_h[std::size_t(s) * WIN + (pos & (WIN - 1))] = pos;
    }
    d_kr_m = DeviceBuffer(kr_h.size() * sizeof(ActT));
    d_vr_m = DeviceBuffer(vr_h.size() * sizeof(ActT));
    d_rp_m = DeviceBuffer(rp_h.size() * 8);
    d_kr_r = DeviceBuffer(kr_h.size() * sizeof(ActT));
    d_vr_r = DeviceBuffer(vr_h.size() * sizeof(ActT));
    d_rp_r = DeviceBuffer(rp_h.size() * 8);
    d_kr_m.copy_from_host(kr_h.data(), d_kr_m.bytes);
    d_vr_m.copy_from_host(vr_h.data(), d_vr_m.bytes);
    d_rp_m.copy_from_host(rp_h.data(), d_rp_m.bytes);
    d_kr_r.copy_from_host(kr_h.data(), d_kr_r.bytes);
    d_vr_r.copy_from_host(vr_h.data(), d_vr_r.bytes);
    d_rp_r.copy_from_host(rp_h.data(), d_rp_r.bytes);
  } else {
    const int num_pages = max_slots * max_pages_per_slot;
    std::vector<std::int32_t> pt_h(static_cast<std::size_t>(num_pages));
    for (int i = 0; i < num_pages; ++i) pt_h[i] = num_pages - 1 - i;  // shuffled physical pages
    d_pt = DeviceBuffer(pt_h.size() * 4);
    d_pt.copy_from_host(pt_h.data(), d_pt.bytes);

    const std::size_t k_page_elems = std::size_t(NUM_FULL_VISITS) * bk::kPageSize * KV_DIM;
    const std::size_t k_bytes = std::size_t(num_pages) * k_page_elems * (cache_fp8 ? 1 : 2);
    const std::size_t v_bytes = cache_v4 ? (k_bytes / 2) : k_bytes;
    std::vector<std::uint8_t> kh(k_bytes), vh(v_bytes);
    std::uniform_int_distribution<int> bd(0, 255);
    if (!cache_fp8) {
      auto fa = to_act(rand_floats(rng, k_bytes / 2, 0.3f));
      auto fb = to_act(rand_floats(rng, v_bytes / 2, 0.3f));
      std::memcpy(kh.data(), fa.data(), k_bytes);
      std::memcpy(vh.data(), fb.data(), v_bytes);
    } else {
      auto kf = rand_floats(rng, k_bytes, 0.4f);
      for (std::size_t i = 0; i < k_bytes; ++i)
        kh[i] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(kf[i], __NV_SATFINITE, __NV_E4M3));
      if (cache_v4) {
        for (std::size_t i = 0; i < v_bytes; ++i) vh[i] = static_cast<std::uint8_t>(bd(rng));
      } else {
        auto vf = rand_floats(rng, v_bytes, 0.4f);
        for (std::size_t i = 0; i < v_bytes; ++i)
          vh[i] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(vf[i], __NV_SATFINITE, __NV_E4M3));
      }
    }
    d_kp_m = DeviceBuffer(k_bytes);
    d_vp_m = DeviceBuffer(v_bytes);
    d_kp_r = DeviceBuffer(k_bytes);
    d_vp_r = DeviceBuffer(v_bytes);
    d_kp_m.copy_from_host(kh.data(), k_bytes);
    d_vp_m.copy_from_host(vh.data(), v_bytes);
    d_kp_r.copy_from_host(kh.data(), k_bytes);
    d_vp_r.copy_from_host(vh.data(), v_bytes);
  }

  // Run multi-row attention
  DeviceBuffer d_as(S * 4), d_r0(S * 4), d_sm(S * 4), d_rs_d(R * 4), d_ro_d(R * 4), d_rb_d(R * 4);
  d_as.copy_from_host(active_slots.data(), d_as.bytes);
  d_r0.copy_from_host(slot_row0.data(), d_r0.bytes);
  d_sm.copy_from_host(slot_m.data(), d_sm.bytes);
  d_rs_d.copy_from_host(row_slot.data(), d_rs_d.bytes);
  d_ro_d.copy_from_host(row_off.data(), d_ro_d.bytes);
  d_rb_d.copy_from_host(row_base.data(), d_rb_d.bytes);

  DeviceBuffer d_pos_m(max_slots * 8), d_pos_r(max_slots * 8);
  d_pos_m.copy_from_host(slot_pos_init.data(), d_pos_m.bytes);
  d_pos_r.copy_from_host(slot_pos_init.data(), d_pos_r.bytes);

  DeviceBuffer d_step_kv(std::size_t(R) * 2 * KV_DIM * sizeof(ActT));
  DeviceBuffer d_hk(std::size_t(R) * KV_DIM * 4), d_hv(std::size_t(R) * KV_DIM * 4);
  DeviceBuffer d_ws(bk::attn_workspace_bytes(R));
  d_ws.fill(0);
  DeviceBuffer d_out_m(std::size_t(R) * HQ * D * sizeof(ActT));
  d_out_m.fill(0);

  bk::BatchedAttnRowsArgs a{};
  a.proj = d_proj.p;
  a.proj_fp32 = proj_fp32;
  a.R = R;
  a.S = S;
  a.active_slots = static_cast<const std::int32_t*>(d_as.p);
  a.slot_row0 = static_cast<const std::int32_t*>(d_r0.p);
  a.slot_m = static_cast<const std::int32_t*>(d_sm.p);
  a.row_slot = static_cast<const std::int32_t*>(d_rs_d.p);
  a.row_off = static_cast<const std::int32_t*>(d_ro_d.p);
  a.row_base = static_cast<const std::int32_t*>(d_rb_d.p);
  a.slot_pos = static_cast<const std::int64_t*>(d_pos_m.p);
  a.q_scale = static_cast<const float*>(d_qs.p);
  a.r_scale = static_cast<const float*>(d_rs.p);
  a.rel_proj = static_cast<const float*>(d_rp.p);
  a.k_conv = static_cast<const float*>(d_kc.p);
  a.v_conv = static_cast<const float*>(d_vc.p);
  a.conv_k_visit = static_cast<float*>(d_ck_m.p);
  a.conv_v_visit = static_cast<float*>(d_cv_m.p);
  a.conv_slot_stride = conv_stride;
  if (full) {
    a.k_full_pool = d_kp_m.p;
    a.v_full_pool = d_vp_m.p;
    a.page_table = static_cast<const std::int32_t*>(d_pt.p);
    a.max_pages_per_slot = max_pages_per_slot;
    a.full_visit_idx = full_visit_idx;
    a.max_ctx = max_ctx;
    a.cache_fp8 = cache_fp8;
    a.cache_v4 = cache_v4;
    a.window = 0;
    a.log_scaling = true;
  } else {
    a.k_ring_visit = static_cast<ActT*>(d_kr_m.p);
    a.v_ring_visit = static_cast<ActT*>(d_vr_m.p);
    a.ring_pos_visit = static_cast<std::int64_t*>(d_rp_m.p);
    a.ring_slot_stride = ring_stride;
    a.ring_pos_slot_stride = ring_pos_stride;
    a.window = WIN;
    a.log_scaling = false;
  }
  a.step_kv = static_cast<ActT*>(d_step_kv.p);
  a.hist_k_visit = static_cast<float*>(d_hk.p);
  a.hist_v_visit = static_cast<float*>(d_hv.p);
  a.workspace = d_ws.p;
  a.workspace_size = d_ws.bytes;
  a.out = static_cast<ActT*>(d_out_m.p);
  bk::batched_attn_decode_rows(a, nullptr);

  // Reference: step-by-step batched_attn_decode + batched_advance_pos
  DeviceBuffer d_ws_r(bk::attn_workspace_bytes(1));
  d_ws_r.fill(0);
  DeviceBuffer d_out_r(std::size_t(R) * HQ * D * sizeof(ActT));
  d_out_r.fill(0);
  DeviceBuffer d_one_slot(4);
  for (int i = 0; i < S; ++i) {
    if (active_slots[i] < 0) continue;
    d_one_slot.copy_from_host(&active_slots[i], 4);
    for (int j = 0; j < slot_m[i]; ++j) {
      const int r = slot_row0[i] + j;
      bk::BatchedAttnArgs b{};
      b.proj = proj_fp32 ? static_cast<const void*>(static_cast<const float*>(d_proj.p) + std::size_t(r) * proj_R)
                         : static_cast<const void*>(static_cast<const ActT*>(d_proj.p) + std::size_t(r) * proj_R);
      b.proj_fp32 = proj_fp32;
      b.B = 1;
      b.slot_ids = static_cast<const std::int32_t*>(d_one_slot.p);
      b.slot_pos = static_cast<const std::int64_t*>(d_pos_r.p);
      b.q_scale = a.q_scale;
      b.r_scale = a.r_scale;
      b.rel_proj = a.rel_proj;
      b.k_conv = a.k_conv;
      b.v_conv = a.v_conv;
      b.conv_k_visit = static_cast<float*>(d_ck_r.p);
      b.conv_v_visit = static_cast<float*>(d_cv_r.p);
      b.conv_slot_stride = conv_stride;
      if (full) {
        b.k_full_pool = d_kp_r.p;
        b.v_full_pool = d_vp_r.p;
        b.page_table = a.page_table;
        b.max_pages_per_slot = max_pages_per_slot;
        b.full_visit_idx = full_visit_idx;
        b.max_ctx = max_ctx;
        b.cache_fp8 = cache_fp8;
        b.cache_v4 = cache_v4;
        b.window = 0;
        b.log_scaling = true;
      } else {
        b.k_ring_visit = static_cast<ActT*>(d_kr_r.p);
        b.v_ring_visit = static_cast<ActT*>(d_vr_r.p);
        b.ring_pos_visit = static_cast<std::int64_t*>(d_rp_r.p);
        b.ring_slot_stride = ring_stride;
        b.ring_pos_slot_stride = ring_pos_stride;
        b.window = WIN;
        b.log_scaling = false;
      }
      b.workspace = d_ws_r.p;
      b.workspace_size = d_ws_r.bytes;
      b.out = static_cast<ActT*>(d_out_r.p) + std::size_t(r) * HQ * D;
      bk::batched_attn_decode(b, nullptr);
      bk::batched_advance_pos(static_cast<const std::int32_t*>(d_one_slot.p), 1, static_cast<std::int64_t*>(d_pos_r.p),
                              nullptr);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ActT> om(R * HQ * D), oref(R * HQ * D);
  d_out_m.copy_to_host(om.data(), d_out_m.bytes);
  d_out_r.copy_to_host(oref.data(), d_out_r.bytes);

  float max_err = 0.f;
  int max_ulp = 0;
  for (int r = 0; r < R; ++r) {
    for (int k = 0; k < HQ * D; ++k) {
      const ActT x = om[std::size_t(r) * HQ * D + k];
      const ActT y = oref[std::size_t(r) * HQ * D + k];
      const float fx = ingine::act::host_to_float(x), fy = ingine::act::host_to_float(y);
      max_err = std::max(max_err, std::fabs(fx - fy));
      std::uint16_t ux, uy;
      std::memcpy(&ux, &x, 2);
      std::memcpy(&uy, &y, 2);
      const int ulp = (ux == uy) ? 0 : std::abs(int(ux) - int(uy));
      if (std::fabs(fx - fy) > 0.f) max_ulp = std::max(max_ulp, ulp);
    }
  }
  // State parity: conv_k, conv_v, ring/pool must be bit-identical
  std::vector<float> ckm(ck_init.size()), ckr(ck_init.size()), cvm(cv_init.size()), cvr(cv_init.size());
  d_ck_m.copy_to_host(ckm.data(), d_ck_m.bytes);
  d_ck_r.copy_to_host(ckr.data(), d_ck_r.bytes);
  d_cv_m.copy_to_host(cvm.data(), d_cv_m.bytes);
  d_cv_r.copy_to_host(cvr.data(), d_cv_r.bytes);
  CHECK(std::memcmp(ckm.data(), ckr.data(), d_ck_m.bytes) == 0);
  CHECK(std::memcmp(cvm.data(), cvr.data(), d_cv_m.bytes) == 0);
  if (!full) {
    std::vector<ActT> krm(d_kr_m.bytes / 2), krr(d_kr_r.bytes / 2), vrm(d_vr_m.bytes / 2), vrr(d_vr_r.bytes / 2);
    std::vector<std::int64_t> rpm(d_rp_m.bytes / 8), rpr(d_rp_r.bytes / 8);
    d_kr_m.copy_to_host(krm.data(), d_kr_m.bytes);
    d_kr_r.copy_to_host(krr.data(), d_kr_r.bytes);
    d_vr_m.copy_to_host(vrm.data(), d_vr_m.bytes);
    d_vr_r.copy_to_host(vrr.data(), d_vr_r.bytes);
    d_rp_m.copy_to_host(rpm.data(), d_rp_m.bytes);
    d_rp_r.copy_to_host(rpr.data(), d_rp_r.bytes);
    CHECK(std::memcmp(krm.data(), krr.data(), d_kr_m.bytes) == 0);
    CHECK(std::memcmp(vrm.data(), vrr.data(), d_vr_m.bytes) == 0);
    CHECK(std::memcmp(rpm.data(), rpr.data(), d_rp_m.bytes) == 0);
  } else {
    std::vector<std::uint8_t> kpm(d_kp_m.bytes), kpr(d_kp_r.bytes), vpm(d_vp_m.bytes), vpr(d_vp_r.bytes);
    d_kp_m.copy_to_host(kpm.data(), d_kp_m.bytes);
    d_kp_r.copy_to_host(kpr.data(), d_kp_r.bytes);
    d_vp_m.copy_to_host(vpm.data(), d_vp_m.bytes);
    d_vp_r.copy_to_host(vpr.data(), d_vp_r.bytes);
    CHECK(std::memcmp(kpm.data(), kpr.data(), d_kp_m.bytes) == 0);
    CHECK(std::memcmp(vpm.data(), vpr.data(), d_vp_m.bytes) == 0);
  }
  // Expect <= 1 ulp of ActT (or <= 1e-3f)
  const bool ok_tol = max_ulp <= 1 || max_err <= 2e-3f;
  if (!ok_tol) {
    std::fprintf(stderr, "batch_spec_attn(full=%d fp8=%d v4=%d proj_fp32=%d): max_err=%g max_ulp=%d\n", full,
                 cache_fp8, cache_v4, proj_fp32, max_err, max_ulp);
  }
  CHECK(ok_tol);
  std::printf("batch_spec_attn(full=%d fp8=%d v4=%d proj_fp32=%d): PASS (max_err=%g, max_ulp=%d)\n", full, cache_fp8,
              cache_v4, proj_fp32, max_err, max_ulp);
}

// ----------------------------------------------------------------------------- 3. batch_spec_rollback
void test_batch_spec_rollback() {
  std::mt19937 rng(777);
  const int max_slots = 4;
  const int num_visits = 4;
  const int num_sliding = 3;
  const int R = 12;  // 3 slots with m = 4 each
  const int S = 3;
  const std::vector<std::int32_t> active_slots = {0, 2, 3};
  const std::vector<std::int32_t> slot_row0 = {0, 4, 8};
  const std::vector<std::int32_t> slot_m = {4, 4, 4};
  std::vector<std::int32_t> row_slot(R), row_off(R), row_base(R);
  for (int i = 0; i < S; ++i)
    for (int j = 0; j < slot_m[i]; ++j) {
      const int r = slot_row0[i] + j;
      row_slot[r] = active_slots[i];
      row_off[r] = j;
      row_base[r] = slot_row0[i];
    }

  const std::vector<std::int64_t> pos_init = {1022, 17, 2046, 2};  // ring wrap + small ring
  const std::size_t kv_conv_stride = std::size_t(num_visits) * 3 * KV_DIM;
  const std::size_t h_conv_stride = std::size_t(num_visits) * 3 * H;
  const std::size_t ring_stride = std::size_t(num_sliding) * WIN * KV_DIM;
  const std::size_t ring_pos_stride = std::size_t(num_sliding) * WIN;

  const auto q_scale = rand_floats(rng, HQ * D, 0.3f);
  const auto r_scale = rand_floats(rng, HQ * DREL, 0.3f);
  const auto rel_proj = rand_floats(rng, DREL * EXT, 0.2f);
  const auto k_conv = rand_floats(rng, KV_DIM * 4, 0.2f);
  const auto v_conv = rand_floats(rng, KV_DIM * 4, 0.2f);
  const auto a_conv = rand_floats(rng, H * 4, 0.2f);
  const auto m_conv = rand_floats(rng, H * 4, 0.2f);
  DeviceBuffer d_qs(q_scale.size() * 4), d_rs(r_scale.size() * 4), d_rp_w(rel_proj.size() * 4);
  DeviceBuffer d_kc(k_conv.size() * 4), d_vc(v_conv.size() * 4), d_ac(a_conv.size() * 4), d_mc(m_conv.size() * 4);
  d_qs.copy_from_host(q_scale.data(), d_qs.bytes);
  d_rs.copy_from_host(r_scale.data(), d_rs.bytes);
  d_rp_w.copy_from_host(rel_proj.data(), d_rp_w.bytes);
  d_kc.copy_from_host(k_conv.data(), d_kc.bytes);
  d_vc.copy_from_host(v_conv.data(), d_vc.bytes);
  d_ac.copy_from_host(a_conv.data(), d_ac.bytes);
  d_mc.copy_from_host(m_conv.data(), d_mc.bytes);

  auto proj_f32 = rand_floats(rng, std::size_t(R) * R_SLIDING, 0.3f);
  for (float& x : proj_f32) x = ingine::act::host_round(x);
  auto wo_f32 = rand_floats(rng, std::size_t(R) * H, 0.3f);
  auto moe_f32 = rand_floats(rng, std::size_t(R) * H, 0.3f);
  DeviceBuffer d_proj(proj_f32.size() * 4), d_wo(wo_f32.size() * 4), d_moe(moe_f32.size() * 4);
  d_proj.copy_from_host(proj_f32.data(), d_proj.bytes);
  d_wo.copy_from_host(wo_f32.data(), d_wo.bytes);
  d_moe.copy_from_host(moe_f32.data(), d_moe.bytes);

  const auto ck0 = rand_floats(rng, std::size_t(max_slots) * kv_conv_stride, 0.3f);
  const auto cv0 = rand_floats(rng, std::size_t(max_slots) * kv_conv_stride, 0.3f);
  const auto ca0 = rand_floats(rng, std::size_t(max_slots) * h_conv_stride, 0.3f);
  const auto cm0 = rand_floats(rng, std::size_t(max_slots) * h_conv_stride, 0.3f);
  const auto kr0 = to_act(rand_floats(rng, std::size_t(max_slots) * ring_stride, 0.3f));
  const auto vr0 = to_act(rand_floats(rng, std::size_t(max_slots) * ring_stride, 0.3f));
  std::vector<std::int64_t> rpos0(std::size_t(max_slots) * ring_pos_stride, -1);
  for (int s = 0; s < max_slots; ++s) {
    const std::int64_t p0 = pos_init[s];
    const std::int64_t lo = std::max<std::int64_t>(0, p0 - WIN);
    for (int ri = 0; ri < num_sliding; ++ri)
      for (std::int64_t pos = lo; pos < p0; ++pos)
        rpos0[std::size_t(s) * ring_pos_stride + std::size_t(ri) * WIN + (pos & (WIN - 1))] = pos;
  }

  DeviceBuffer d_as(S * 4), d_r0(S * 4), d_sm(S * 4), d_rsd(R * 4), d_rod(R * 4), d_rbd(R * 4);
  d_as.copy_from_host(active_slots.data(), d_as.bytes);
  d_r0.copy_from_host(slot_row0.data(), d_r0.bytes);
  d_sm.copy_from_host(slot_m.data(), d_sm.bytes);
  d_rsd.copy_from_host(row_slot.data(), d_rsd.bytes);
  d_rod.copy_from_host(row_off.data(), d_rod.bytes);
  d_rbd.copy_from_host(row_base.data(), d_rbd.bytes);

  // Test every accept count a in {0, 1, 2, 3} (commits {1, 2, 3, 4}), with heterogeneous commits across slots!
  for (int base_a = 0; base_a < 4; ++base_a) {
    const std::vector<std::int32_t> commits = {1 + base_a, 1 + ((base_a + 1) % 4), 1 + ((base_a + 2) % 4)};
    DeviceBuffer d_commits(S * 4);
    d_commits.copy_from_host(commits.data(), d_commits.bytes);

    auto make_pool = [&](DeviceBuffer& ck, DeviceBuffer& cv, DeviceBuffer& ca, DeviceBuffer& cm, DeviceBuffer& kr,
                         DeviceBuffer& vr, DeviceBuffer& rp, DeviceBuffer& pos) {
      ck = DeviceBuffer(ck0.size() * 4);
      cv = DeviceBuffer(cv0.size() * 4);
      ca = DeviceBuffer(ca0.size() * 4);
      cm = DeviceBuffer(cm0.size() * 4);
      kr = DeviceBuffer(kr0.size() * sizeof(ActT));
      vr = DeviceBuffer(vr0.size() * sizeof(ActT));
      rp = DeviceBuffer(rpos0.size() * 8);
      pos = DeviceBuffer(pos_init.size() * 8);
      ck.copy_from_host(ck0.data(), ck.bytes);
      cv.copy_from_host(cv0.data(), cv.bytes);
      ca.copy_from_host(ca0.data(), ca.bytes);
      cm.copy_from_host(cm0.data(), cm.bytes);
      kr.copy_from_host(kr0.data(), kr.bytes);
      vr.copy_from_host(vr0.data(), vr.bytes);
      rp.copy_from_host(rpos0.data(), rp.bytes);
      pos.copy_from_host(pos_init.data(), pos.bytes);
    };

    DeviceBuffer ck_m, cv_m, ca_m, cm_m, kr_m, vr_m, rp_m, pos_m;
    DeviceBuffer ck_r, cv_r, ca_r, cm_r, kr_r, vr_r, rp_r, pos_r;
    make_pool(ck_m, cv_m, ca_m, cm_m, kr_m, vr_m, rp_m, pos_m);
    make_pool(ck_r, cv_r, ca_r, cm_r, kr_r, vr_r, rp_r, pos_r);

    bk::BatchStatePoolPtrs st_m{};
    st_m.conv[0] = static_cast<float*>(ck_m.p);
    st_m.conv[1] = static_cast<float*>(cv_m.p);
    st_m.conv[2] = static_cast<float*>(ca_m.p);
    st_m.conv[3] = static_cast<float*>(cm_m.p);
    st_m.conv_slot_stride[0] = kv_conv_stride;
    st_m.conv_slot_stride[1] = kv_conv_stride;
    st_m.conv_slot_stride[2] = h_conv_stride;
    st_m.conv_slot_stride[3] = h_conv_stride;
    st_m.k_ring = static_cast<ActT*>(kr_m.p);
    st_m.v_ring = static_cast<ActT*>(vr_m.p);
    st_m.ring_pos = static_cast<std::int64_t*>(rp_m.p);
    st_m.ring_slot_stride = ring_stride;
    st_m.ring_pos_slot_stride = ring_pos_stride;
    st_m.num_visits = num_visits;
    st_m.num_sliding_visits = num_sliding;

    // Backup & hist buffers
    DeviceBuffer bk_c0(std::size_t(max_slots) * num_visits * 2 * KV_DIM * 4);
    DeviceBuffer bk_c1(std::size_t(max_slots) * num_visits * 2 * KV_DIM * 4);
    DeviceBuffer bk_c2(std::size_t(max_slots) * num_visits * 2 * H * 4);
    DeviceBuffer bk_c3(std::size_t(max_slots) * num_visits * 2 * H * 4);
    DeviceBuffer bk_k(std::size_t(max_slots) * num_sliding * bk::kMaxVerifyRows * KV_DIM * sizeof(ActT));
    DeviceBuffer bk_v(std::size_t(max_slots) * num_sliding * bk::kMaxVerifyRows * KV_DIM * sizeof(ActT));
    DeviceBuffer bk_p(std::size_t(max_slots) * num_sliding * bk::kMaxVerifyRows * 8);
    bk::BatchVerifyBackup bk_buf{};
    bk_buf.conv[0] = static_cast<float*>(bk_c0.p);
    bk_buf.conv[1] = static_cast<float*>(bk_c1.p);
    bk_buf.conv[2] = static_cast<float*>(bk_c2.p);
    bk_buf.conv[3] = static_cast<float*>(bk_c3.p);
    bk_buf.conv_slot_stride[0] = std::size_t(num_visits) * 2 * KV_DIM;
    bk_buf.conv_slot_stride[1] = std::size_t(num_visits) * 2 * KV_DIM;
    bk_buf.conv_slot_stride[2] = std::size_t(num_visits) * 2 * H;
    bk_buf.conv_slot_stride[3] = std::size_t(num_visits) * 2 * H;
    bk_buf.k = static_cast<ActT*>(bk_k.p);
    bk_buf.v = static_cast<ActT*>(bk_v.p);
    bk_buf.pos = static_cast<std::int64_t*>(bk_p.p);
    bk_buf.kv_slot_stride = std::size_t(num_sliding) * bk::kMaxVerifyRows * KV_DIM;
    bk_buf.pos_slot_stride = std::size_t(num_sliding) * bk::kMaxVerifyRows;

    DeviceBuffer h0(std::size_t(num_visits) * R * KV_DIM * 4);
    DeviceBuffer h1(std::size_t(num_visits) * R * KV_DIM * 4);
    DeviceBuffer h2(std::size_t(num_visits) * R * H * 4);
    DeviceBuffer h3(std::size_t(num_visits) * R * H * 4);
    const float* hptrs[4] = {static_cast<const float*>(h0.p), static_cast<const float*>(h1.p),
                             static_cast<const float*>(h2.p), static_cast<const float*>(h3.p)};

    DeviceBuffer d_step_kv(std::size_t(R) * 2 * KV_DIM * sizeof(ActT));
    DeviceBuffer d_ws(bk::attn_workspace_bytes(R));
    d_ws.fill(0);
    DeviceBuffer d_aout(std::size_t(R) * HQ * D * sizeof(ActT));
    DeviceBuffer d_y(std::size_t(R) * H * 4), d_y2(std::size_t(R) * H * 4);

    // 1. Save backup
    bk::batch_verify_save(st_m, bk_buf, static_cast<const std::int32_t*>(d_as.p),
                          static_cast<const std::int32_t*>(d_sm.p), S, static_cast<const std::int64_t*>(pos_m.p),
                          nullptr);

    // 2. Run m=4 across visits (first `num_sliding` visits do sliding attn + attn_conv + mlp_conv; last does convs)
    for (int v = 0; v < num_visits; ++v) {
      if (v < num_sliding) {
        bk::BatchedAttnRowsArgs a{};
        a.proj = d_proj.p;
        a.proj_fp32 = true;
        a.R = R;
        a.S = S;
        a.active_slots = static_cast<const std::int32_t*>(d_as.p);
        a.slot_row0 = static_cast<const std::int32_t*>(d_r0.p);
        a.slot_m = static_cast<const std::int32_t*>(d_sm.p);
        a.row_slot = static_cast<const std::int32_t*>(d_rsd.p);
        a.row_off = static_cast<const std::int32_t*>(d_rod.p);
        a.row_base = static_cast<const std::int32_t*>(d_rbd.p);
        a.slot_pos = static_cast<const std::int64_t*>(pos_m.p);
        a.q_scale = static_cast<const float*>(d_qs.p);
        a.r_scale = static_cast<const float*>(d_rs.p);
        a.rel_proj = static_cast<const float*>(d_rp_w.p);
        a.k_conv = static_cast<const float*>(d_kc.p);
        a.v_conv = static_cast<const float*>(d_vc.p);
        a.conv_k_visit = st_m.conv[0] + std::size_t(v) * 3 * KV_DIM;
        a.conv_v_visit = st_m.conv[1] + std::size_t(v) * 3 * KV_DIM;
        a.conv_slot_stride = kv_conv_stride;
        a.k_ring_visit = st_m.k_ring + std::size_t(v) * WIN * KV_DIM;
        a.v_ring_visit = st_m.v_ring + std::size_t(v) * WIN * KV_DIM;
        a.ring_pos_visit = st_m.ring_pos + std::size_t(v) * WIN;
        a.ring_slot_stride = ring_stride;
        a.ring_pos_slot_stride = ring_pos_stride;
        a.window = WIN;
        a.step_kv = static_cast<ActT*>(d_step_kv.p);
        a.hist_k_visit = static_cast<float*>(h0.p) + std::size_t(v) * R * KV_DIM;
        a.hist_v_visit = static_cast<float*>(h1.p) + std::size_t(v) * R * KV_DIM;
        a.workspace = d_ws.p;
        a.workspace_size = d_ws.bytes;
        a.out = static_cast<ActT*>(d_aout.p);
        bk::batched_attn_decode_rows(a, nullptr);
      } else {
        // Run static conv on conv[0], conv[1] directly to test visit >= num_sliding rollback too
        bk::batched_static_conv_rows(d_proj.p, true, R_SLIDING, S, KV_DIM, static_cast<const std::int32_t*>(d_as.p),
                                     static_cast<const std::int32_t*>(d_r0.p), static_cast<const std::int32_t*>(d_sm.p),
                                     static_cast<const float*>(d_kc.p), st_m.conv[0] + std::size_t(v) * 3 * KV_DIM,
                                     kv_conv_stride, static_cast<float*>(d_y.p),
                                     static_cast<float*>(h0.p) + std::size_t(v) * R * KV_DIM, nullptr);
        bk::batched_static_conv_rows(d_proj.p, true, R_SLIDING, S, KV_DIM, static_cast<const std::int32_t*>(d_as.p),
                                     static_cast<const std::int32_t*>(d_r0.p), static_cast<const std::int32_t*>(d_sm.p),
                                     static_cast<const float*>(d_vc.p), st_m.conv[1] + std::size_t(v) * 3 * KV_DIM,
                                     kv_conv_stride, static_cast<float*>(d_y.p),
                                     static_cast<float*>(h1.p) + std::size_t(v) * R * KV_DIM, nullptr);
      }
      bk::batched_static_conv_rows(d_wo.p, true, H, S, H, static_cast<const std::int32_t*>(d_as.p),
                                   static_cast<const std::int32_t*>(d_r0.p), static_cast<const std::int32_t*>(d_sm.p),
                                   static_cast<const float*>(d_ac.p), st_m.conv[2] + std::size_t(v) * 3 * H,
                                   h_conv_stride, static_cast<float*>(d_y.p),
                                   static_cast<float*>(h2.p) + std::size_t(v) * R * H, nullptr);
      bk::batched_static_conv_rows(d_moe.p, true, H, S, H, static_cast<const std::int32_t*>(d_as.p),
                                   static_cast<const std::int32_t*>(d_r0.p), static_cast<const std::int32_t*>(d_sm.p),
                                   static_cast<const float*>(d_mc.p), st_m.conv[3] + std::size_t(v) * 3 * H,
                                   h_conv_stride, static_cast<float*>(d_y2.p),
                                   static_cast<float*>(h3.p) + std::size_t(v) * R * H, nullptr);
    }

    // 3. Rollback & advance pos by commits
    bk::batch_verify_rollback(st_m, bk_buf, hptrs, R, static_cast<const std::int32_t*>(d_as.p),
                              static_cast<const std::int32_t*>(d_r0.p), static_cast<const std::int32_t*>(d_sm.p),
                              static_cast<const std::int32_t*>(d_commits.p), S,
                              static_cast<const std::int64_t*>(pos_m.p), nullptr);
    bk::batched_advance_pos_by(static_cast<const std::int32_t*>(d_as.p), static_cast<const std::int32_t*>(d_commits.p),
                               S, static_cast<std::int64_t*>(pos_m.p), nullptr);

    // Reference: run exactly `commits[i]` single-row steps for each slot i!
    DeviceBuffer d_ws_r(bk::attn_workspace_bytes(1));
    d_ws_r.fill(0);
    DeviceBuffer d_one_slot(4);
    for (int i = 0; i < S; ++i) {
      d_one_slot.copy_from_host(&active_slots[i], 4);
      for (int j = 0; j < commits[i]; ++j) {
        const int r = slot_row0[i] + j;
        for (int v = 0; v < num_visits; ++v) {
          if (v < num_sliding) {
            bk::BatchedAttnArgs b{};
            b.proj = static_cast<const float*>(d_proj.p) + std::size_t(r) * R_SLIDING;
            b.proj_fp32 = true;
            b.B = 1;
            b.slot_ids = static_cast<const std::int32_t*>(d_one_slot.p);
            b.slot_pos = static_cast<const std::int64_t*>(pos_r.p);
            b.q_scale = static_cast<const float*>(d_qs.p);
            b.r_scale = static_cast<const float*>(d_rs.p);
            b.rel_proj = static_cast<const float*>(d_rp_w.p);
            b.k_conv = static_cast<const float*>(d_kc.p);
            b.v_conv = static_cast<const float*>(d_vc.p);
            b.conv_k_visit = static_cast<float*>(ck_r.p) + std::size_t(v) * 3 * KV_DIM;
            b.conv_v_visit = static_cast<float*>(cv_r.p) + std::size_t(v) * 3 * KV_DIM;
            b.conv_slot_stride = kv_conv_stride;
            b.k_ring_visit = static_cast<ActT*>(kr_r.p) + std::size_t(v) * WIN * KV_DIM;
            b.v_ring_visit = static_cast<ActT*>(vr_r.p) + std::size_t(v) * WIN * KV_DIM;
            b.ring_pos_visit = static_cast<std::int64_t*>(rp_r.p) + std::size_t(v) * WIN;
            b.ring_slot_stride = ring_stride;
            b.ring_pos_slot_stride = ring_pos_stride;
            b.window = WIN;
            b.workspace = d_ws_r.p;
            b.workspace_size = d_ws_r.bytes;
            b.out = static_cast<ActT*>(d_aout.p);
            bk::batched_attn_decode(b, nullptr);
          } else {
            bk::batched_static_conv_step(static_cast<const float*>(d_proj.p) + std::size_t(r) * R_SLIDING, true,
                                         R_SLIDING, 1, KV_DIM, static_cast<const std::int32_t*>(d_one_slot.p),
                                         static_cast<const float*>(d_kc.p),
                                         static_cast<float*>(ck_r.p) + std::size_t(v) * 3 * KV_DIM, kv_conv_stride,
                                         static_cast<float*>(d_y.p), nullptr);
            bk::batched_static_conv_step(static_cast<const float*>(d_proj.p) + std::size_t(r) * R_SLIDING, true,
                                         R_SLIDING, 1, KV_DIM, static_cast<const std::int32_t*>(d_one_slot.p),
                                         static_cast<const float*>(d_vc.p),
                                         static_cast<float*>(cv_r.p) + std::size_t(v) * 3 * KV_DIM, kv_conv_stride,
                                         static_cast<float*>(d_y.p), nullptr);
          }
          bk::batched_static_conv_step(static_cast<const float*>(d_wo.p) + std::size_t(r) * H, true, H, 1, H,
                                       static_cast<const std::int32_t*>(d_one_slot.p), static_cast<const float*>(d_ac.p),
                                       static_cast<float*>(ca_r.p) + std::size_t(v) * 3 * H, h_conv_stride,
                                       static_cast<float*>(d_y.p), nullptr);
          bk::batched_static_conv_step(static_cast<const float*>(d_moe.p) + std::size_t(r) * H, true, H, 1, H,
                                       static_cast<const std::int32_t*>(d_one_slot.p), static_cast<const float*>(d_mc.p),
                                       static_cast<float*>(cm_r.p) + std::size_t(v) * 3 * H, h_conv_stride,
                                       static_cast<float*>(d_y2.p), nullptr);
        }
        bk::batched_advance_pos(static_cast<const std::int32_t*>(d_one_slot.p), 1, static_cast<std::int64_t*>(pos_r.p),
                                nullptr);
      }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto cmp_buf = [&](const char* name, const DeviceBuffer& a, const DeviceBuffer& b) {
      std::vector<std::uint8_t> ha(a.bytes), hb(b.bytes);
      a.copy_to_host(ha.data(), a.bytes);
      b.copy_to_host(hb.data(), b.bytes);
      const bool eq = (std::memcmp(ha.data(), hb.data(), a.bytes) == 0);
      if (!eq) std::fprintf(stderr, "batch_spec_rollback(base_a=%d): %s MISMATCH!\n", base_a, name);
      CHECK(eq);
    };
    cmp_buf("conv_k", ck_m, ck_r);
    cmp_buf("conv_v", cv_m, cv_r);
    cmp_buf("conv_attn", ca_m, ca_r);
    cmp_buf("conv_mlp", cm_m, cm_r);
    cmp_buf("k_ring", kr_m, kr_r);
    cmp_buf("v_ring", vr_m, vr_r);
    cmp_buf("ring_pos", rp_m, rp_r);
    cmp_buf("slot_pos", pos_m, pos_r);
  }
  std::printf("batch_spec_rollback: PASS\n");
}

}  // namespace

int main() {
  test_batch_spec_conv();
  test_batch_spec_attn_case(/*full=*/false, false, false, true);
  test_batch_spec_attn_case(/*full=*/false, false, false, false);
  test_batch_spec_attn_case(/*full=*/true, false, false, true);
  test_batch_spec_attn_case(/*full=*/true, true, false, true);
  test_batch_spec_attn_case(/*full=*/true, true, true, true);
  test_batch_spec_attn_case(/*full=*/true, true, true, false);
  test_batch_spec_rollback();
  if (g_failures == 0) {
    std::printf("ALL batch_spec_kernels TESTS PASSED\n");
    return 0;
  }
  std::fprintf(stderr, "%d FAILURE(S)\n", g_failures);
  return 1;
}

#include "backends/qwen38/qwen38_block_cpu.h"
#include "backends/qwen38/qwen38_cpu_ops.h"
#include "backends/qwen38/qwen38_manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <sys/resource.h>
#include <vector>

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

#define CHECK_THROWS(expr, ex_type)                                                         \
  do {                                                                                      \
    bool threw_ = false;                                                                    \
    try {                                                                                   \
      (void)(expr);                                                                         \
    } catch (const ex_type&) {                                                              \
      threw_ = true;                                                                        \
    }                                                                                       \
    if (!threw_) {                                                                          \
      std::fprintf(stderr, "%s:%d: expected exception %s: %s\n", __FILE__, __LINE__,        \
                   #ex_type, #expr);                                                        \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

namespace {

using namespace ingine::backends::qwen38;

float max_abs_diff(std::span<const float> a, std::span<const float> b) {
  CHECK(a.size() == b.size());
  float m = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
    m = std::max(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

void test_fast_e4m3_parity() {
  const std::vector<float> edges = {
      0.0f, -0.0f, 1e-12f, -1e-12f, 0.0009765625f, 0.0009765626f, 0.001953125f,
      0.015625f, 0.0166015625f, 1.0f, -1.0f, 1.0625f, 1.125f, 416.0f, 431.99f, 432.0f, 447.99f, 448.0f, 500.0f, -500.0f,
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
  };
  for (float x : edges) {
    CHECK(quantize_e4m3fn_fast(x) == quantize_e4m3fn(x));
  }

  std::mt19937 rng(20261125u);
  std::normal_distribution<float> dist(0.0f, 30.0f);
  std::uniform_real_distribution<float> tiny(-0.05f, 0.05f);
  for (int i = 0; i < 50000; ++i) {
    const float v1 = dist(rng);
    const float v2 = tiny(rng);
    CHECK(quantize_e4m3fn_fast(v1) == quantize_e4m3fn(v1));
    CHECK(quantize_e4m3fn_fast(v2) == quantize_e4m3fn(v2));
  }
}

void test_synthetic_block_harness() {
  const std::array<int, 1> layer0 = {0};
  const std::array<int, 1> layer3 = {3};
  const std::array<int, 4> layers03 = {0, 1, 2, 3};
  const std::array<std::int32_t, 6> toks = {3, 17, 29, 41, 7, 19};

  for (Qwen38WeightPrecision prec : {Qwen38WeightPrecision::Bf16, Qwen38WeightPrecision::MixedFp8Nvfp4}) {
    for (std::span<const int> ls :
         std::vector<std::span<const int>>{layer0, layer3, layers03}) {
      const auto bundle = make_synthetic_qwen38_block_bundle(prec, ls, toks, 4242u);
      const auto sw_1t = run_qwen38_blocks_cpu(bundle, true, false, 4, true, 1);
      const auto sw_4t = run_qwen38_blocks_cpu(bundle, true, false, 4, true, 4);
      const auto ch_4t = run_qwen38_blocks_cpu(bundle, true, true, 3, true, 4);
      const auto sim_4t = run_qwen38_blocks_cpu(bundle, true, ls.size() == 1 && ls[0] == 0, 4, true, 4, true);
      CHECK(max_abs_diff(sw_1t.hidden_seq, sw_4t.hidden_seq) == 0.0f);
      CHECK(max_abs_diff(sw_1t.hidden_seq, ch_4t.hidden_seq) <= 1e-4f);
      CHECK(max_abs_diff(sw_1t.hidden_seq, sim_4t.hidden_seq) <= 1e-3f);
      for (std::size_t i = 0; i < bundle.layers.size(); ++i) {
        if (bundle.layers[i].kind == Qwen38LayerKind::LinearAttention) {
          CHECK(max_abs_diff(sw_1t.recurrent_states[i], ch_4t.recurrent_states[i]) <= 1e-5f);
          CHECK(max_abs_diff(sw_1t.conv_states[i], ch_4t.conv_states[i]) <= 1e-5f);
          CHECK(max_abs_diff(sw_1t.recurrent_states[i], sim_4t.recurrent_states[i]) <= 2e-4f);
        }
      }
    }
    const auto b4 = make_synthetic_qwen38_block_bundle(prec, layers03, toks, 777u);
    const auto b4_l0 = slice_qwen38_block_layer(b4, 0);
    const auto b4_l3 = slice_qwen38_block_layer(b4, 3);
    const auto b1_l0 = make_synthetic_qwen38_block_bundle(prec, layer0, toks, 777u);
    CHECK(b4_l0.layers[0].layer_index == 0);
    CHECK(b4_l3.layers[0].layer_index == 3);
    CHECK(b4_l0.prompt_embeddings_bf16 == b1_l0.prompt_embeddings_bf16);
  }
}

void test_gate1_linear_fixtures_cpu() {
  CHECK(bf16_ulp_distance(0x0000u, 0x8000u) == 0u);
  CHECK(bf16_ulp_distance(0x3f80u, 0x3f81u) == 1u);
  CHECK(bf16_ulp_distance(0x7fc0u, 0x0000u) == std::numeric_limits<std::uint32_t>::max());
  CHECK(bf16_ulp_distance(0x7f80u, 0x3f80u) == std::numeric_limits<std::uint32_t>::max());

  const auto fx = make_gate1_linear_fixtures(20261125u);
  const std::size_t out_n = static_cast<std::size_t>(fx.m * fx.n);

  // Confirm raw FP32 oracle vs nearest-BF16 reproduces the documented >1e-3 excursions (657/1024 and 57/1024).
  auto count_nearest_bf16_excursions = [&](std::span<const float> ref_f32, std::span<const std::uint16_t> ref_bf16,
                                           std::size_t& count_out, float& max_err_out) {
    count_out = 0;
    max_err_out = 0.0f;
    for (std::size_t i = 0; i < ref_f32.size(); ++i) {
      const float err = std::fabs(bf16_to_f32(ref_bf16[i]) - ref_f32[i]);
      max_err_out = std::max(max_err_out, err);
      if (err > 1e-3f) ++count_out;
    }
  };
  std::size_t nv_exc = 0, fp8_cal_exc = 0;
  float nv_max_err = 0.0f, fp8_cal_max_err = 0.0f;
  count_nearest_bf16_excursions(fx.nvfp4_ref_f32, fx.nvfp4_ref_bf16, nv_exc, nv_max_err);
  count_nearest_bf16_excursions(fx.fp8_cal_ref_f32, fx.fp8_cal_ref_bf16, fp8_cal_exc, fp8_cal_max_err);
  CHECK(nv_exc == 657u && nv_max_err > 0.025f);
  CHECK(fp8_cal_exc == 57u && fp8_cal_max_err > 0.0019f);

  const auto exact_nv = compare_bf16_against_rounded_oracle(fx.nvfp4_ref_bf16, fx.nvfp4_ref_f32, 0u, 0.0f);
  CHECK(exact_nv.passed && exact_nv.exact_bf16_matches == out_n && exact_nv.criterion_violations == 0);

  auto to_bf16_bits = [](std::span<const float> src) {
    std::vector<std::uint16_t> bits(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) bits[i] = f32_to_bf16(src[i]);
    return bits;
  };

  // NVFP4 warp FP32 simulation & rounded BF16 oracle check.
  std::vector<float> sim_nv(out_n);
  simulate_warp_f32_nvfp4_linear(fx.nvfp4_x_q, fx.nvfp4_packed, fx.nvfp4_scales, fx.nvfp4_ws2, fx.m, fx.n, fx.k,
                                 sim_nv);
  CHECK(max_abs_diff(sim_nv, fx.nvfp4_ref_f32) <= 1e-3f);
  auto sim_nv_bits = to_bf16_bits(sim_nv);
  const auto cmp_nv = compare_bf16_against_rounded_oracle(sim_nv_bits, fx.nvfp4_ref_f32, 1u, 1e-3f);
  CHECK(cmp_nv.passed && cmp_nv.criterion_violations == 0 && cmp_nv.max_bf16_ulp <= 1u);

  // FP8 weight-only & calibrated warp FP32 simulation & rounded BF16 oracle checks.
  std::vector<float> sim_wo(out_n), sim_cal(out_n);
  simulate_warp_f32_fp8_linear(fx.fp8_x_q, fx.fp8_w, fx.fp8_ws, 0.0f, fx.m, fx.n, fx.k, sim_wo);
  simulate_warp_f32_fp8_linear(fx.fp8_x_q, fx.fp8_w, fx.fp8_ws, fx.fp8_in_sc, fx.m, fx.n, fx.k, sim_cal);
  CHECK(max_abs_diff(sim_wo, fx.fp8_wo_ref_f32) <= 1e-3f);
  CHECK(max_abs_diff(sim_cal, fx.fp8_cal_ref_f32) <= 1e-3f);
  const auto cmp_wo = compare_bf16_against_rounded_oracle(to_bf16_bits(sim_wo), fx.fp8_wo_ref_f32, 1u, 1e-3f);
  const auto cmp_cal = compare_bf16_against_rounded_oracle(to_bf16_bits(sim_cal), fx.fp8_cal_ref_f32, 1u, 1e-3f);
  CHECK(cmp_wo.passed && cmp_cal.passed);

  // Negative test: perturb one large-magnitude element by >= 2 BF16 ULPs -> fail closed with 1 violation.
  std::size_t large_idx = out_n;
  for (std::size_t i = 0; i < out_n; ++i) {
    if (std::fabs(fx.nvfp4_ref_f32[i]) > 1.0f) {
      large_idx = i;
      break;
    }
  }
  CHECK(large_idx < out_n);
  auto bad_bits = to_bf16_bits(sim_nv);
  const std::uint16_t oracle_bits = fx.nvfp4_ref_bf16[large_idx];
  const std::uint16_t mag = static_cast<std::uint16_t>(oracle_bits & 0x7fffu);
  bad_bits[large_idx] = static_cast<std::uint16_t>((oracle_bits & 0x8000u) | (mag > 4u ? (mag - 3u) : (mag + 3u)));
  const auto bad_cmp = compare_bf16_against_rounded_oracle(bad_bits, fx.nvfp4_ref_f32, 1u, 1e-3f);
  CHECK(!bad_cmp.passed && bad_cmp.criterion_violations == 1u && bad_cmp.max_bf16_ulp >= 2u);

  // Floor-exempt case regression (review.log: actual ULP=14851 within abs_floor=1e-3 vs 0.0f):
  // MUST retain true raw maximum (14851), report 0 above-floor max, and count floor_exempt_count=1.
  {
    const std::array<std::uint16_t, 1> got_floor = {14851u};
    const std::array<float, 1> ref_zero = {0.0f};
    const auto fl_cmp = compare_bf16_against_rounded_oracle(got_floor, ref_zero, 1u, 1e-3f);
    CHECK(fl_cmp.passed);
    CHECK(fl_cmp.max_bf16_ulp == 14851u);
    CHECK(fl_cmp.max_bf16_ulp_above_floor == 0u);
    CHECK(fl_cmp.floor_exempt_count == 1u);
    CHECK(fl_cmp.criterion_violations == 0u);
  }
}

void test_gate1_coverage_enforcement() {
  const auto& req = qwen38_gate1_required_row_names();
  CHECK(req.size() == 29u);

  std::vector<Qwen38Gate1RowStatus> all_ok;
  all_ok.reserve(req.size());
  for (const auto& name : req) all_ok.push_back({name, "pass", true});

  const auto qual = evaluate_gate1_coverage(all_ok, false, true);
  CHECK(qual.qualification_passed && qual.all_passed && qual.complete_coverage && qual.executed_checks_passed);
  CHECK(qual.coverage_status == "complete_qualified" && qual.missing_required_rows.empty() && qual.failed_rows.empty());

  const auto unpinned = evaluate_gate1_coverage(all_ok, false, false);
  CHECK(!unpinned.qualification_passed && !unpinned.all_passed && unpinned.complete_coverage);
  CHECK(unpinned.coverage_status == "failed");

  const auto diag_all = evaluate_gate1_coverage(all_ok, true, true);
  CHECK(!diag_all.qualification_passed && !diag_all.all_passed && diag_all.coverage_status == "incomplete_diagnostic");

  // Missing 0.8B rows fails qualification even if everything else passed.
  {
    auto rows = all_ok;
    for (auto& r : rows) {
      if (r.name.starts_with("qwen08b_")) {
        r.status = "missing_checkpoint";
        r.passed = false;
      }
    }
    const auto s_qual = evaluate_gate1_coverage(rows, false, true);
    CHECK(!s_qual.complete_coverage && !s_qual.qualification_passed && !s_qual.all_passed);
    CHECK(s_qual.missing_required_rows.size() == 4u && s_qual.coverage_status == "failed");
    const auto s_diag = evaluate_gate1_coverage(rows, true, true);
    CHECK(!s_diag.complete_coverage && s_diag.executed_checks_passed && !s_diag.all_passed &&
          s_diag.coverage_status == "incomplete_diagnostic");
  }

  // Omitted 27B rows fails qualification.
  {
    std::vector<Qwen38Gate1RowStatus> no27;
    for (const auto& r : all_ok) {
      if (!r.name.starts_with("qwen27b_")) no27.push_back(r);
    }
    const auto s_qual = evaluate_gate1_coverage(no27, false, false);
    CHECK(!s_qual.complete_coverage && !s_qual.qualification_passed && !s_qual.all_passed);
    CHECK(s_qual.missing_required_rows.size() == 4u);
  }

  // Single failed row fails both qualification and diagnostic execution status.
  {
    auto rows = all_ok;
    rows[0].status = "fail";
    rows[0].passed = false;
    const auto s = evaluate_gate1_coverage(rows, true, true);
    CHECK(!s.executed_checks_passed && !s.qualification_passed && !s.all_passed && s.coverage_status == "failed");
    CHECK(s.failed_rows.size() == 1u);
  }
}

void test_block_parity_evaluator() {
  const std::vector<float> v_ref = {1.0f, -2.0f, 3.0f, 4.0f};
  const std::vector<float> v_eq = {1.0f, -2.0f, 3.0f, 4.0f};
  const auto eq_stats = compare_f32_vectors(v_eq, v_ref);
  CHECK(eq_stats.finite && eq_stats.max_abs == 0.0f && eq_stats.rel_l2 == 0.0f && eq_stats.cosine_sim == 1.0);

  const std::vector<float> v_nan = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 4.0f};
  CHECK(!compare_f32_vectors(v_nan, v_ref).finite);

  const std::array<int, 1> l0 = {0};
  const std::array<int, 4> l4 = {0, 1, 2, 3};
  const std::array<std::int32_t, 3> toks = {2, 5, 9};

  const auto b1 = make_synthetic_qwen38_block_bundle(Qwen38WeightPrecision::Bf16, l0, toks, 101u);
  const auto r1_ref = run_qwen38_blocks_cpu(b1, true, true, 4, true, 2, false);
  const auto r1_sim = run_qwen38_blocks_cpu(b1, true, true, 4, true, 2, true);
  CHECK(evaluate_block_parity(b1, r1_sim, r1_ref, Qwen38BlockParityTier::SyntheticStrict).passed);
  CHECK(evaluate_block_parity(b1, r1_sim, r1_ref, Qwen38BlockParityTier::RealSingleLayer).passed);

  {
    auto bad_gdn = r1_ref;
    bad_gdn.recurrent_states[0][0] += 5e-4f;
    CHECK(!evaluate_block_parity(b1, bad_gdn, r1_ref, Qwen38BlockParityTier::RealSingleLayer).passed);
  }
  {
    auto bad_cos = r1_ref;
    for (float& x : bad_cos.hidden_seq) x = -x;
    const auto p = evaluate_block_parity(b1, bad_cos, r1_ref, Qwen38BlockParityTier::RealSingleLayer);
    CHECK(!p.passed && p.hidden.cosine_sim < 0.0);
  }
  {
    auto bad_nan = r1_ref;
    bad_nan.hidden_seq[0] = std::numeric_limits<float>::infinity();
    CHECK(!evaluate_block_parity(b1, bad_nan, r1_ref, Qwen38BlockParityTier::SyntheticStrict).passed);
  }

  const auto b4 = make_synthetic_qwen38_block_bundle(Qwen38WeightPrecision::MixedFp8Nvfp4, l4, toks, 202u);
  const auto r4_ref = run_qwen38_blocks_cpu(b4, true, false, 4, true, 2, false);
  const auto r4_sim = run_qwen38_blocks_cpu(b4, true, false, 4, true, 2, true);
  CHECK(evaluate_block_parity(b4, r4_sim, r4_ref, Qwen38BlockParityTier::SyntheticStrict).passed);
  CHECK(evaluate_block_parity(b4, r4_sim, r4_ref, Qwen38BlockParityTier::RealStacked4Layer).passed);

  {
    auto bad_l0 = r4_ref;
    bad_l0.recurrent_states[0][0] += 3e-4f;
    CHECK(!evaluate_block_parity(b4, bad_l0, r4_ref, Qwen38BlockParityTier::RealStacked4Layer).passed);
  }
  {
    auto bad_rel = r4_ref;
    for (float& x : bad_rel.hidden_seq) x *= 1.03f;
    CHECK(!evaluate_block_parity(b4, bad_rel, r4_ref, Qwen38BlockParityTier::RealStacked4Layer).passed);
  }

  // Regression for review.log line 2: valid 4-layer hidden & GDN output MUST FAIL when conv_states / K / V
  // caches are damaged (nonfinite, missing/truncated, finite drift, or stray KV on linear layers).
  {
    auto damaged_combo = r4_ref;
    damaged_combo.conv_states[0][0] = std::numeric_limits<float>::quiet_NaN();
    damaged_combo.k_caches[3][0] = std::numeric_limits<float>::infinity();
    damaged_combo.v_caches[3].clear();
    const auto p = evaluate_block_parity(b4, damaged_combo, r4_ref, Qwen38BlockParityTier::RealStacked4Layer);
    CHECK(!p.passed && !p.state_shapes_valid && !p.conv_finite && !p.kv_finite);
  }
  {
    auto bad_conv_nan = r4_ref;
    bad_conv_nan.conv_states[0][0] = std::numeric_limits<float>::quiet_NaN();
    const auto p = evaluate_block_parity(b4, bad_conv_nan, r4_ref, Qwen38BlockParityTier::RealStacked4Layer);
    CHECK(!p.passed && !p.conv_finite && p.state_shapes_valid);
  }
  {
    auto bad_conv_trunc = r4_ref;
    bad_conv_trunc.conv_states[0].pop_back();
    const auto p = evaluate_block_parity(b4, bad_conv_trunc, r4_ref, Qwen38BlockParityTier::RealStacked4Layer);
    CHECK(!p.passed && !p.state_shapes_valid);
  }
  {
    auto bad_k_inf = r4_ref;
    bad_k_inf.k_caches[3][0] = -std::numeric_limits<float>::infinity();
    const auto p = evaluate_block_parity(b4, bad_k_inf, r4_ref, Qwen38BlockParityTier::RealStacked4Layer);
    CHECK(!p.passed && !p.kv_finite && p.state_shapes_valid);
  }
  {
    auto bad_v_missing = r4_ref;
    bad_v_missing.v_caches[3].clear();
    const auto p = evaluate_block_parity(b4, bad_v_missing, r4_ref, Qwen38BlockParityTier::RealStacked4Layer);
    CHECK(!p.passed && !p.state_shapes_valid);
  }
  {
    auto bad_conv_val = r4_ref;
    bad_conv_val.conv_states[0][0] += 0.1f;
    CHECK(!evaluate_block_parity(b4, bad_conv_val, r4_ref, Qwen38BlockParityTier::RealStacked4Layer).passed);
  }
  {
    auto bad_stacked_conv = r4_ref;
    bad_stacked_conv.conv_states[1][0] += 0.2f;
    CHECK(!evaluate_block_parity(b4, bad_stacked_conv, r4_ref, Qwen38BlockParityTier::RealStacked4Layer).passed);
  }
  {
    auto bad_k_val = r4_ref;
    bad_k_val.k_caches[3][0] += 0.5f;
    CHECK(!evaluate_block_parity(b4, bad_k_val, r4_ref, Qwen38BlockParityTier::RealStacked4Layer).passed);
  }
  {
    auto bad_v_val = r4_ref;
    bad_v_val.v_caches[3][0] += 0.5f;
    CHECK(!evaluate_block_parity(b4, bad_v_val, r4_ref, Qwen38BlockParityTier::RealStacked4Layer).passed);
  }
  {
    auto stray_kv = r4_ref;
    stray_kv.k_caches[0].push_back(0.0f);
    const auto p = evaluate_block_parity(b4, stray_kv, r4_ref, Qwen38BlockParityTier::RealStacked4Layer);
    CHECK(!p.passed && !p.state_shapes_valid);
  }
}

Qwen38BlockBundleHost slice_qwen38_block_tokens(const Qwen38BlockBundleHost& bundle, std::size_t tok_begin,
                                               std::size_t tok_count) {
  CHECK(tok_count > 0 && tok_begin + tok_count <= bundle.token_ids.size());
  const std::size_t H = static_cast<std::size_t>(bundle.cfg.hidden_size);
  Qwen38BlockBundleHost out;
  out.cfg = bundle.cfg;
  out.token_ids.assign(bundle.token_ids.begin() + static_cast<std::ptrdiff_t>(tok_begin),
                       bundle.token_ids.begin() + static_cast<std::ptrdiff_t>(tok_begin + tok_count));
  out.prompt_embeddings_bf16.assign(
      bundle.prompt_embeddings_bf16.begin() + static_cast<std::ptrdiff_t>(tok_begin * H),
      bundle.prompt_embeddings_bf16.begin() + static_cast<std::ptrdiff_t>((tok_begin + tok_count) * H));
  out.layers = bundle.layers;
  return out;
}

void bench_27b_pinned_cpu_dry_run(bool verify_pins_27b) {
  const auto t_start = std::chrono::steady_clock::now();
  auto wall_ms = [&]() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
  };
  const std::filesystem::path q08_dir("/mnt/storage/models/Qwen3.5-0.8B");
  const std::filesystem::path q27_dir("/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f");
  CHECK(std::filesystem::is_directory(q08_dir));
  CHECK(std::filesystem::is_directory(q27_dir));

  std::printf("[phase] start=qwen08b_inspect wall_ms=%.1f\n", wall_ms());
  const auto t_i08_0 = std::chrono::steady_clock::now();
  const auto m08 = Qwen38Manifest::inspect(q08_dir, false, "local");
  const double qwen08b_inspect_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_i08_0).count();
  std::printf("[phase] done=qwen08b_inspect elapsed_ms=%.1f wall_ms=%.1f\n", qwen08b_inspect_ms, wall_ms());

  const std::array<int, 4> l4 = {0, 1, 2, 3};
  const std::array<std::int32_t, 6> toks08 = {9707, 15, 220, 16, 17, 198};
  const std::array<std::int32_t, 6> toks_holdout = {785, 279, 1493, 311, 458, 19};

  std::printf("[phase] start=qwen08b_load wall_ms=%.1f\n", wall_ms());
  const auto t_l08_0 = std::chrono::steady_clock::now();
  const auto b08_4l = load_qwen38_block_bundle(q08_dir, m08, l4, toks08);
  const auto b08_holdout_4l = load_qwen38_block_bundle(q08_dir, m08, l4, toks_holdout);
  const double qwen08b_load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_l08_0).count();
  std::printf("[phase] done=qwen08b_load elapsed_ms=%.1f wall_ms=%.1f\n", qwen08b_load_ms, wall_ms());

  std::printf("[phase] start=qwen27b_manifest_sha wall_ms=%.1f\n", wall_ms());
  const auto t_sha0 = std::chrono::steady_clock::now();
  const auto m27 = verify_pins_27b ? Qwen38Manifest::inspect_pinned(q27_dir, qwen38_27b_nvfp4_pin_spec(), "482ca0f")
                                   : Qwen38Manifest::inspect(q27_dir, false, "482ca0f");
  const double qwen27b_manifest_sha_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sha0).count();
  std::printf("[phase] done=qwen27b_manifest_sha elapsed_ms=%.1f wall_ms=%.1f\n", qwen27b_manifest_sha_ms, wall_ms());

  // Extract primary (first 4) + disjoint holdout (last 6) token embeddings in a single pinned 4L stream pass.
  const std::array<std::int32_t, 10> toks27_both = {9707, 15, 220, 16, 785, 279, 1493, 311, 458, 19};
  std::printf("[phase] start=qwen27b_stream_load wall_ms=%.1f\n", wall_ms());
  const auto t_l27_0 = std::chrono::steady_clock::now();
  const auto b27_both = load_qwen38_block_bundle(q27_dir, m27, l4, toks27_both);
  const double qwen27b_stream_load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_l27_0).count();
  std::printf("[phase] done=qwen27b_stream_load elapsed_ms=%.1f wall_ms=%.1f\n", qwen27b_stream_load_ms, wall_ms());

  const auto b27_4l = slice_qwen38_block_tokens(b27_both, 0, 4);
  const auto b27_holdout_4l = slice_qwen38_block_tokens(b27_both, 4, 6);

  double qwen08b_cpu_fp64_ms = 0.0, qwen08b_cpu_warp_ms = 0.0;
  double qwen27b_cpu_fp64_ms = 0.0, qwen27b_cpu_warp_ms = 0.0;
  double holdout_cpu_ms = 0.0;
  bool all_ok = true;

  auto check_sim_pair = [&](const char* label, const Qwen38BlockBundleHost& b, bool chunk, bool cal_fp8,
                            Qwen38BlockParityTier tier, double* fp64_acc_ms, double* warp_acc_ms) {
    const auto r_fp64 = run_qwen38_blocks_cpu(b, true, chunk, 64, cal_fp8, 8, false);
    const auto r_fp32 = run_qwen38_blocks_cpu(b, true, chunk, 64, cal_fp8, 8, true);
    if (fp64_acc_ms) *fp64_acc_ms += r_fp64.elapsed_ms;
    if (warp_acc_ms) *warp_acc_ms += r_fp32.elapsed_ms;
    const auto p = evaluate_block_parity(b, r_fp32, r_fp64, tier);
    const float conv_rel_l2 = std::max(p.layer0_conv_rel_l2, p.stacked_conv_rel_l2);
    const float conv_rel_max = std::max(p.layer0_conv_rel_max_abs, p.stacked_conv_rel_max_abs);
    std::printf(
        "  [%s] rel_l2=%.6g cos=%.8f rel_max=%.6g l0_gdn=%.6f stacked_gdn=%.6f "
        "conv_rel_l2=%.6g conv_rel_max=%.6g k_rel_l2=%.6g k_rel_max=%.6g k_cos=%.8f "
        "v_rel_l2=%.6g v_rel_max=%.6g v_cos=%.8f raw_ulp=%u above_floor_ulp=%u "
        "exempt=%zu fp64_ms=%.1f warp_ms=%.1f -> %s\n",
        label, p.hidden.rel_l2, p.hidden.cosine_sim, p.hidden.rel_max_abs, p.layer0_gdn_max_abs,
        p.stacked_gdn_max_abs, conv_rel_l2, conv_rel_max, p.k_cache_rel_l2, p.k_cache_rel_max_abs, p.k_cache_cosine,
        p.v_cache_rel_l2, p.v_cache_rel_max_abs, p.v_cache_cosine, p.hidden_bf16.max_bf16_ulp,
        p.hidden_bf16.max_bf16_ulp_above_floor, p.hidden_bf16.floor_exempt_count, r_fp64.elapsed_ms, r_fp32.elapsed_ms,
        p.passed ? "PASS" : "FAIL");
    all_ok = all_ok && p.passed;
  };

  std::printf("[phase] start=qwen08b_blocks wall_ms=%.1f\n", wall_ms());
  check_sim_pair("qwen08b_1L_gdn", slice_qwen38_block_layer(b08_4l, 0), true, false,
                 Qwen38BlockParityTier::RealSingleLayer, &qwen08b_cpu_fp64_ms, &qwen08b_cpu_warp_ms);
  check_sim_pair("qwen08b_1L_full", slice_qwen38_block_layer(b08_4l, 3), false, false,
                 Qwen38BlockParityTier::RealSingleLayer, &qwen08b_cpu_fp64_ms, &qwen08b_cpu_warp_ms);
  check_sim_pair("qwen08b_4L_stepwise", b08_4l, false, false, Qwen38BlockParityTier::RealStacked4Layer,
                 &qwen08b_cpu_fp64_ms, &qwen08b_cpu_warp_ms);
  check_sim_pair("qwen08b_4L_chunk", b08_4l, true, false, Qwen38BlockParityTier::RealStacked4Layer,
                 &qwen08b_cpu_fp64_ms, &qwen08b_cpu_warp_ms);
  std::printf("[phase] done=qwen08b_blocks fp64_ms=%.1f warp_ms=%.1f wall_ms=%.1f\n",
              qwen08b_cpu_fp64_ms, qwen08b_cpu_warp_ms, wall_ms());

  std::printf("[phase] start=qwen27b_blocks wall_ms=%.1f\n", wall_ms());
  check_sim_pair("qwen27b_1L_gdn", slice_qwen38_block_layer(b27_4l, 0), true, true,
                 Qwen38BlockParityTier::RealSingleLayer, &qwen27b_cpu_fp64_ms, &qwen27b_cpu_warp_ms);
  check_sim_pair("qwen27b_1L_full", slice_qwen38_block_layer(b27_4l, 3), false, true,
                 Qwen38BlockParityTier::RealSingleLayer, &qwen27b_cpu_fp64_ms, &qwen27b_cpu_warp_ms);
  check_sim_pair("qwen27b_4L_stepwise", b27_4l, false, true, Qwen38BlockParityTier::RealStacked4Layer,
                 &qwen27b_cpu_fp64_ms, &qwen27b_cpu_warp_ms);
  check_sim_pair("qwen27b_4L_chunk", b27_4l, true, true, Qwen38BlockParityTier::RealStacked4Layer,
                 &qwen27b_cpu_fp64_ms, &qwen27b_cpu_warp_ms);
  std::printf("[phase] done=qwen27b_blocks fp64_ms=%.1f warp_ms=%.1f wall_ms=%.1f\n",
              qwen27b_cpu_fp64_ms, qwen27b_cpu_warp_ms, wall_ms());

  // Independent completely disjoint holdout sequence {785, 279, 1493, 311, 458, 19}.
  std::printf("[phase] start=holdout_disjoint_blocks wall_ms=%.1f\n", wall_ms());
  const auto t_ho0 = std::chrono::steady_clock::now();
  check_sim_pair("holdout_08b_1L_gdn", slice_qwen38_block_layer(b08_holdout_4l, 0), true, false,
                 Qwen38BlockParityTier::RealSingleLayer, &holdout_cpu_ms, nullptr);
  check_sim_pair("holdout_08b_1L_full", slice_qwen38_block_layer(b08_holdout_4l, 3), false, false,
                 Qwen38BlockParityTier::RealSingleLayer, &holdout_cpu_ms, nullptr);
  check_sim_pair("holdout_08b_4L_stepwise", b08_holdout_4l, false, false, Qwen38BlockParityTier::RealStacked4Layer,
                 &holdout_cpu_ms, nullptr);
  check_sim_pair("holdout_08b_4L_chunk", b08_holdout_4l, true, false, Qwen38BlockParityTier::RealStacked4Layer,
                 &holdout_cpu_ms, nullptr);

  check_sim_pair("holdout_27b_1L_gdn", slice_qwen38_block_layer(b27_holdout_4l, 0), true, true,
                 Qwen38BlockParityTier::RealSingleLayer, &holdout_cpu_ms, nullptr);
  check_sim_pair("holdout_27b_1L_full", slice_qwen38_block_layer(b27_holdout_4l, 3), false, true,
                 Qwen38BlockParityTier::RealSingleLayer, &holdout_cpu_ms, nullptr);
  check_sim_pair("holdout_27b_4L_stepwise", b27_holdout_4l, false, true, Qwen38BlockParityTier::RealStacked4Layer,
                 &holdout_cpu_ms, nullptr);
  check_sim_pair("holdout_27b_4L_chunk", b27_holdout_4l, true, true, Qwen38BlockParityTier::RealStacked4Layer,
                 &holdout_cpu_ms, nullptr);
  const double holdout_total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ho0).count();
  std::printf("[phase] done=holdout_disjoint_blocks elapsed_ms=%.1f wall_ms=%.1f\n", holdout_total_ms, wall_ms());

  struct rusage ru {};
  std::uint64_t rss_kib = (::getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_maxrss > 0)
                               ? static_cast<std::uint64_t>(ru.ru_maxrss)
                               : 0ull;
  std::printf(
      "bench_27b_%s summary: q08_inspect_ms=%.1f q08_load_ms=%.1f q08_fp64_ms=%.1f q08_warp_ms=%.1f "
      "q27_manifest_sha_ms=%.1f q27_stream_load_ms=%.1f q27_fp64_ms=%.1f q27_warp_ms=%.1f "
      "holdout_total_ms=%.1f vmhwm_kib=%llu total_wall_ms=%.1f all_16_passed=%s\n",
      verify_pins_27b ? "pinned" : "fast", qwen08b_inspect_ms, qwen08b_load_ms, qwen08b_cpu_fp64_ms,
      qwen08b_cpu_warp_ms, qwen27b_manifest_sha_ms, qwen27b_stream_load_ms, qwen27b_cpu_fp64_ms, qwen27b_cpu_warp_ms,
      holdout_total_ms, static_cast<unsigned long long>(rss_kib), wall_ms(), all_ok ? "true" : "false");
  CHECK(all_ok);
}

struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const char* tag) {
    path = std::filesystem::temp_directory_path() / (std::string("ingine_p1a_prep_") + tag);
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

void write_bytes(const std::filesystem::path& p, const void* d, std::size_t n) {
  std::ofstream out(p, std::ios::binary);
  CHECK(out.good());
  out.write(static_cast<const char*>(d), static_cast<std::streamsize>(n));
}

void write_str(const std::filesystem::path& p, const std::string& s) { write_bytes(p, s.data(), s.size()); }

void test_pinned_safetensors_block_extract() {
  TempDir dir("pinned");
  const std::string cfg = R"({
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5_text",
    "tie_word_embeddings": true,
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 4,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "full_attention_interval": 4,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "attention_bias": false,
    "attn_output_gate": true,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 4,
    "linear_value_head_dim": 4,
    "linear_num_key_heads": 1,
    "linear_num_value_heads": 2,
    "mamba_ssm_dtype": "float32",
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
  })";
  write_str(dir.path / "config.json", cfg);
  write_str(dir.path / "tokenizer.json", R"({"version":"1.0"})");
  write_str(dir.path / "chat_template.jinja", "{{ messages }}");

  struct Spec {
    std::string name;
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::size_t elem_bytes;
  };
  std::vector<Spec> specs = {
      {"model.language_model.embed_tokens.weight", "BF16", {32, 16}, 2},
      {"model.language_model.norm.weight", "BF16", {16}, 2},
  };
  for (int l = 0; l < 4; ++l) {
    const std::string lp = "model.language_model.layers." + std::to_string(l);
    specs.push_back({lp + ".input_layernorm.weight", "BF16", {16}, 2});
    specs.push_back({lp + ".post_attention_layernorm.weight", "BF16", {16}, 2});
    if (l < 3) {
      const std::string a = lp + ".linear_attn";
      specs.push_back({a + ".A_log", "F32", {2}, 4});
      specs.push_back({a + ".dt_bias", "BF16", {2}, 2});
      specs.push_back({a + ".conv1d.weight", "BF16", {16, 1, 4}, 2});
      specs.push_back({a + ".in_proj_a.weight", "BF16", {2, 16}, 2});
      specs.push_back({a + ".in_proj_b.weight", "BF16", {2, 16}, 2});
      specs.push_back({a + ".norm.weight", "BF16", {4}, 2});
      specs.push_back({a + ".in_proj_qkv.weight", "BF16", {16, 16}, 2});
      specs.push_back({a + ".in_proj_z.weight", "BF16", {8, 16}, 2});
      specs.push_back({a + ".out_proj.weight", "BF16", {16, 8}, 2});
    } else {
      const std::string a = lp + ".self_attn";
      specs.push_back({a + ".q_norm.weight", "BF16", {8}, 2});
      specs.push_back({a + ".k_norm.weight", "BF16", {8}, 2});
      specs.push_back({a + ".q_proj.weight", "BF16", {32, 16}, 2});
      specs.push_back({a + ".k_proj.weight", "BF16", {8, 16}, 2});
      specs.push_back({a + ".v_proj.weight", "BF16", {8, 16}, 2});
      specs.push_back({a + ".o_proj.weight", "BF16", {16, 16}, 2});
    }
    specs.push_back({lp + ".mlp.gate_proj.weight", "BF16", {32, 16}, 2});
    specs.push_back({lp + ".mlp.up_proj.weight", "BF16", {32, 16}, 2});
    specs.push_back({lp + ".mlp.down_proj.weight", "BF16", {16, 32}, 2});
  }

  nlohmann::ordered_json hdr;
  std::uint64_t off = 0;
  for (const auto& sp : specs) {
    std::uint64_t elems = 1;
    for (auto d : sp.shape) elems *= static_cast<std::uint64_t>(d);
    const std::uint64_t bytes = elems * sp.elem_bytes;
    hdr[sp.name] = {{"dtype", sp.dtype}, {"shape", sp.shape}, {"data_offsets", {off, off + bytes}}};
    off += bytes;
  }
  const std::string hdr_s = hdr.dump();
  const std::uint64_t hdr_len = hdr_s.size();
  std::vector<std::uint8_t> st(8 + hdr_len + off, 0x3cu);
  for (int i = 0; i < 8; ++i) st[i] = static_cast<std::uint8_t>((hdr_len >> (8 * i)) & 0xffu);
  std::memcpy(st.data() + 8, hdr_s.data(), hdr_len);
  write_bytes(dir.path / "model.safetensors", st.data(), st.size());

  const auto m_self = Qwen38Manifest::inspect(dir.path, true, "p1a");
  ingine::model::CheckpointPinSpec pins;
  pins.config_sha256 = m_self.checkpoint.config_sha256;
  pins.tokenizer_sha256 = m_self.checkpoint.tokenizer_sha256;
  pins.chat_template_sha256 = m_self.checkpoint.chat_template_sha256;
  pins.shards = {{"model.safetensors", m_self.shard_sizes.at("model.safetensors"),
                  ingine::model::parse_sha256_hex(m_self.shard_sha256.at("model.safetensors"))}};
  const auto m_pin = Qwen38Manifest::inspect_pinned(dir.path, pins, "p1a");

  const std::array<int, 4> layers = {0, 1, 2, 3};
  const std::array<std::int32_t, 3> toks = {2, 5, 9};
  const auto bundle = load_qwen38_block_bundle(dir.path, m_pin, layers, toks);
  CHECK(bundle.layers.size() == 4);
  const auto res_sw = run_qwen38_blocks_cpu(bundle, true, false, 4, true, 2);
  const auto res_ch = run_qwen38_blocks_cpu(bundle, true, true, 2, true, 2);
  CHECK(max_abs_diff(res_sw.hidden_seq, res_ch.hidden_seq) <= 1e-4f);

  // Real Qwen3.5-0.8B 1-layer extraction + CPU block smoke test when present locally.
  const std::filesystem::path q08("/mnt/storage/models/Qwen3.5-0.8B");
  if (std::filesystem::is_directory(q08)) {
    const auto m08 = Qwen38Manifest::inspect(q08, false, "local");
    const std::array<int, 1> l0 = {0};
    const std::array<std::int32_t, 2> t2 = {10, 20};
    const auto b08 = load_qwen38_block_bundle(q08, m08, l0, t2);
    const auto sw08 = run_qwen38_blocks_cpu(b08, true, false, 4, false, 4);
    const auto ch08 = run_qwen38_blocks_cpu(b08, true, true, 2, false, 4);
    CHECK(max_abs_diff(sw08.hidden_seq, ch08.hidden_seq) <= 1e-3f);
    std::printf("  0.8B 1-layer CPU extraction+parity OK (sw_ms=%.1f, ch_ms=%.1f)\n", sw08.elapsed_ms,
                ch08.elapsed_ms);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::setvbuf(stderr, nullptr, _IOLBF, 0);
  bool bench_27b_pinned = false;
  bool bench_27b_fast = false;
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--bench-27b-pinned") {
      bench_27b_pinned = true;
    } else if (std::string_view(argv[i]) == "--bench-27b-fast") {
      bench_27b_fast = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }
  test_fast_e4m3_parity();
  test_synthetic_block_harness();
  test_gate1_linear_fixtures_cpu();
  test_gate1_coverage_enforcement();
  test_block_parity_evaluator();
  test_pinned_safetensors_block_extract();
  if (bench_27b_fast) bench_27b_pinned_cpu_dry_run(false);
  if (bench_27b_pinned) bench_27b_pinned_cpu_dry_run(true);
  std::printf("qwen38_phase1a_cpu_prep: PASS\n");
  return 0;
}

#include "backends/qwen38/qwen38_gate2_metrics.h"
#include "backends/qwen38/qwen38_manifest.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include <vector>

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

namespace {

using namespace ingine::backends::qwen38;

void test_kld_and_top1() {
  // Identical distributions -> KLD == 0, top1 matches, max_abs_diff == 0.
  const std::vector<float> a = {1.5f, -0.5f, 3.25f, 0.0f};
  const auto p_eq = compute_logit_kld_and_top1(a, a, 7);
  CHECK(p_eq.finite);
  CHECK(p_eq.position == 7);
  CHECK(p_eq.kld == 0.0);
  CHECK(p_eq.ref_top1 == 2 && p_eq.cand_top1 == 2 && p_eq.top1_match);
  CHECK(p_eq.max_abs_diff == 0.0f);

  // Shift invariance: adding constant to candidate logits does not change KLD or argmax.
  const std::vector<float> a_shift = {101.5f, 99.5f, 103.25f, 100.0f};
  const auto p_shift = compute_logit_kld_and_top1(a, a_shift, 0);
  CHECK(p_shift.finite);
  CHECK(std::fabs(p_shift.kld) < 1e-10);
  CHECK(p_shift.top1_match && p_shift.cand_top1 == 2);

  // Analytic D_KL([0.7, 0.3] || [0.4, 0.6]).
  const std::vector<float> r_logits = {std::log(0.7f), std::log(0.3f)};
  const std::vector<float> c_logits = {std::log(0.4f), std::log(0.6f)};
  const double expected_kl = 0.7 * std::log(0.7 / 0.4) + 0.3 * std::log(0.3 / 0.6);
  const auto p_kl = compute_logit_kld_and_top1(r_logits, c_logits, 1);
  CHECK(p_kl.finite);
  CHECK(std::fabs(p_kl.kld - expected_kl) < 1e-6);
  CHECK(p_kl.ref_top1 == 0 && p_kl.cand_top1 == 1 && !p_kl.top1_match);

  // Fail closed on NaN, Inf, empty, or size mismatch.
  const std::vector<float> b_nan = {1.0f, std::numeric_limits<float>::quiet_NaN(), 3.0f, 0.0f};
  CHECK(!compute_logit_kld_and_top1(a, b_nan, 0).finite);
  const std::vector<float> b_inf = {1.0f, 0.0f, std::numeric_limits<float>::infinity(), 0.0f};
  CHECK(!compute_logit_kld_and_top1(a, b_inf, 0).finite);
  const std::vector<float> short_v = {1.0f, 2.0f};
  CHECK(!compute_logit_kld_and_top1(a, short_v, 0).finite);
  const std::vector<float> empty_v;
  CHECK(!compute_logit_kld_and_top1(empty_v, empty_v, 0).finite);
}

void test_kld_summary_and_verdict() {
  std::vector<Qwen38LogitParityPoint> pts;
  for (int i = 0; i < 101; ++i) {
    Qwen38LogitParityPoint p;
    p.position = i;
    p.kld = static_cast<double>(i) * 0.0001;  // 0.0000 .. 0.0100
    p.top1_match = (i < 98);
    p.finite = true;
    pts.push_back(p);
  }
  const auto s = summarize_kld_metrics(pts);
  CHECK(s.all_finite && s.positions == 101u);
  CHECK(std::fabs(s.kld_mean - 0.005) < 1e-12);
  CHECK(std::fabs(s.kld_p99 - 0.0099) < 1e-12);
  CHECK(std::fabs(s.kld_max - 0.01) < 1e-12);
  CHECK(s.top1_matches == 98u);

  Qwen38Gate2Input base;
  base.diagnostic_mode = false;
  base.qwen27b_pins_verified = true;
  base.short_ref_sha_verified = true;
  base.long_ref_sha_verified = true;
  base.short_prompt_count = 3;
  base.prompt0_greedy_matches = 16;
  base.prompt0_greedy_total = 16;
  base.short_summary = Qwen38KldSummary{141, 0.0018, 0.015, 0.020, 136, 136.0 / 141.0, true};
  base.long_prompt_count = 6;
  base.long_summary = Qwen38KldSummary{288, 0.0030, 0.025, 0.060, 282, 282.0 / 288.0, true};
  base.chunk_vs_step_prompt_count = 3;
  base.chunk_vs_step_summary = Qwen38KldSummary{3, 1e-6, 2e-6, 2e-6, 3, 1.0, true};
  base.chunk_vs_step_greedy_matches = 48;
  base.chunk_vs_step_greedy_total = 48;

  const auto pass_v = evaluate_gate2_verdict(base);
  CHECK(pass_v.short_prompts_passed && pass_v.long_code_passed && pass_v.chunked_vs_stepwise_passed &&
        pass_v.executed_checks_passed && pass_v.qualification_passed && pass_v.failures.empty());

  // 10+ fail-closed negative tests:
  {
    auto d = base;
    d.diagnostic_mode = true;
    const auto v = evaluate_gate2_verdict(d);
    CHECK(v.executed_checks_passed && !v.qualification_passed);
  }
  {
    auto d = base;
    d.qwen27b_pins_verified = false;
    CHECK(!evaluate_gate2_verdict(d).qualification_passed);
  }
  {
    auto d = base;
    d.short_ref_sha_verified = false;
    CHECK(!evaluate_gate2_verdict(d).qualification_passed);
  }
  {
    auto d = base;
    d.long_ref_sha_verified = false;
    CHECK(!evaluate_gate2_verdict(d).qualification_passed);
  }
  {
    auto d = base;
    d.prompt0_greedy_matches = 15;
    CHECK(!evaluate_gate2_verdict(d).short_prompts_passed && !evaluate_gate2_verdict(d).qualification_passed);
  }
  {
    auto d = base;
    d.short_summary.kld_mean = 0.0051;
    CHECK(!evaluate_gate2_verdict(d).short_prompts_passed);
  }
  {
    auto d = base;
    d.short_summary.kld_max = 0.051;
    CHECK(!evaluate_gate2_verdict(d).short_prompts_passed);
  }
  {
    auto d = base;
    d.short_summary.top1_agreement = 0.89;
    CHECK(!evaluate_gate2_verdict(d).short_prompts_passed);
  }
  {
    auto d = base;
    d.long_summary.kld_mean = 0.011;
    CHECK(!evaluate_gate2_verdict(d).long_code_passed);
  }
  {
    auto d = base;
    d.long_summary.kld_p99 = 0.081;
    CHECK(!evaluate_gate2_verdict(d).long_code_passed);
  }
  {
    auto d = base;
    d.long_summary.top1_agreement = 0.94;
    CHECK(!evaluate_gate2_verdict(d).long_code_passed);
  }
  {
    auto d = base;
    d.chunk_vs_step_summary.kld_max = 2e-4;
    CHECK(!evaluate_gate2_verdict(d).chunked_vs_stepwise_passed);
  }
  {
    auto d = base;
    d.chunk_vs_step_greedy_matches = 47;
    CHECK(!evaluate_gate2_verdict(d).chunked_vs_stepwise_passed);
  }
  {
    auto d = base;
    d.long_summary.all_finite = false;
    CHECK(!evaluate_gate2_verdict(d).long_code_passed);
  }
  // Regression for hand-built summaries claiming all_finite=true with NaN/Inf/out-of-range scalars:
  {
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for (int tc = 0; tc < 8; ++tc) {
      auto d = base;
      if (tc == 0) d.short_summary.kld_mean = qnan;
      if (tc == 1) d.short_summary.kld_max = qnan;
      if (tc == 2) d.short_summary.top1_agreement = qnan;
      if (tc == 3) d.long_summary.kld_p99 = qnan;
      if (tc == 4) d.chunk_vs_step_summary.kld_max = qnan;
      if (tc == 5) d.short_summary.kld_mean = inf;
      if (tc == 6) d.long_summary.kld_mean = -0.001;
      if (tc == 7) d.short_summary.top1_agreement = 1.01;
      const auto v = evaluate_gate2_verdict(d);
      CHECK(!v.executed_checks_passed && !v.qualification_passed && !v.failures.empty());
    }
  }
}

void test_streaming_reference_reader() {
  const std::filesystem::path ref_dir("/mnt/storage/flare/release-references");
  const std::filesystem::path alt_b1_dir("/mnt/storage/flare/john-repro-b1");
  const std::filesystem::path short_path = ref_dir / "reference.safetensors";
  const std::filesystem::path short_meta_path = ref_dir / "reference.json";
  const std::filesystem::path alt_b1_path = alt_b1_dir / "reference.safetensors";
  const std::filesystem::path alt_b1_meta_path = alt_b1_dir / "reference.json";
  const std::filesystem::path long_path = ref_dir / "long-code-reference.safetensors";
  if (!std::filesystem::is_regular_file(short_path) || !std::filesystem::is_regular_file(short_meta_path) ||
      !std::filesystem::is_regular_file(alt_b1_path) || !std::filesystem::is_regular_file(alt_b1_meta_path) ||
      !std::filesystem::is_regular_file(long_path)) {
    std::printf("  skipping on-disk reference safetensors test (files absent)\n");
    return;
  }

  std::string sha_chk;
  CHECK(verify_file_sha256(short_meta_path, kGate2ShortRefMetaSha256, &sha_chk));
  CHECK(sha_chk == kGate2ShortRefMetaSha256);
  CHECK(verify_file_sha256(alt_b1_meta_path, kGate2AltShortRefB1MetaSha256, &sha_chk));
  CHECK(sha_chk == kGate2AltShortRefB1MetaSha256);
  CHECK(!verify_file_sha256(ref_dir / "missing_nonexistent.json", kGate2ShortRefMetaSha256, &sha_chk));

  Qwen38RefSafetensors sr(short_path);
  CHECK(sr.verify_sha256(kGate2ShortRefSha256));
  const auto p0_ids = sr.read_i64("prompt_0.prompt_ids");
  const auto p0_gen = sr.read_i64("prompt_0.generated_ids");
  CHECK(p0_ids.size() == 32u && p0_gen.size() == 16u);
  const auto& pf0_meta = sr.at("prompt_0.prefill_logits");
  CHECK(pf0_meta.shape.size() == 2 && pf0_meta.shape[0] == 32 && pf0_meta.shape[1] == 248320);
  std::vector<float> row(static_cast<std::size_t>(pf0_meta.shape[1]));
  sr.read_f32_row("prompt_0.prefill_logits", 31, row);
  const auto p0_step0_top2 = compute_top2_logits(row);
  CHECK(p0_step0_top2.finite && p0_step0_top2.top1_id == static_cast<std::int32_t>(p0_gen[0]));
  CHECK(p0_gen[0] == 5215);

  sr.read_f32_row("prompt_0.decode_logits", 0, row);
  const auto b3_dec0_top2 = compute_top2_logits(row);
  CHECK(b3_dec0_top2.finite && b3_dec0_top2.top1_id == 303 && b3_dec0_top2.top2_id == 321);
  CHECK(row[303] > row[321] && std::fabs((row[303] - row[321]) - 0.026779f) < 1e-4f);

  Qwen38RefSafetensors b1_ref(alt_b1_path);
  CHECK(b1_ref.verify_sha256(kGate2AltShortRefB1Sha256));
  const auto b1_p0_ids = b1_ref.read_i64("prompt_0.prompt_ids");
  const auto b1_p0_gen = b1_ref.read_i64("prompt_0.generated_ids");
  CHECK(b1_p0_ids == p0_ids);
  CHECK(b1_p0_gen.size() == 16u && b1_p0_gen[0] == 5215 && b1_p0_gen[1] == 321);
  b1_ref.read_f32_row("prompt_0.decode_logits", 0, row);
  const auto b1_dec0_top2 = compute_top2_logits(row);
  CHECK(b1_dec0_top2.finite && b1_dec0_top2.top1_id == 321 && b1_dec0_top2.top2_id == 303);
  CHECK(row[303] < row[321] && std::fabs((row[303] - row[321]) + 0.004461f) < 1e-4f);

  std::vector<std::int32_t> b3_32(16), b1_32(16);
  for (std::size_t i = 0; i < 16; ++i) {
    b3_32[i] = static_cast<std::int32_t>(p0_gen[i]);
    b1_32[i] = static_cast<std::int32_t>(b1_p0_gen[i]);
  }
  CHECK(evaluate_prompt0_canonical_trajectory(b3_32, b3_32, b1_32).chosen_realization == "release_b3");
  CHECK(evaluate_prompt0_canonical_trajectory(b1_32, b3_32, b1_32).chosen_realization == "canonical_b1");
  auto hybrid = b3_32;
  hybrid[1] = b1_32[1];
  CHECK(evaluate_prompt0_canonical_trajectory(hybrid, b3_32, b1_32).chosen_realization == "none");
  auto tail_mut = b1_32;
  tail_mut[15] += 1;
  CHECK(evaluate_prompt0_canonical_trajectory(tail_mut, b3_32, b1_32).chosen_realization == "none");

  Qwen38RefSafetensors lr(long_path);
  CHECK(lr.verify_sha256(kGate2LongCodeRefSha256));
  const auto l0_ids = lr.read_i64("prompt_0.input_ids");
  const auto l0_pos = lr.read_i64("prompt_0.positions");
  CHECK(l0_ids.size() == 4128u && l0_pos.size() == 48u);
  const auto& l0_meta = lr.at("prompt_0.logits");
  CHECK(l0_meta.shape.size() == 2 && l0_meta.shape[0] == 48 && l0_meta.shape[1] == 248320);
  lr.read_f32_row("prompt_0.logits", 0, row);
  for (float x : row) CHECK(std::isfinite(x));
  std::printf("  reference safetensors pins & Option 1 dual-canonical check verified (b3=%d, b1=%d)\n",
              b3_dec0_top2.top1_id, b1_dec0_top2.top1_id);
}

void test_option1_addendum_replay() {
  const std::vector<std::int32_t> kRelB3 = {5215, 303, 279, 795, 321, 6805, 279, 1727,
                                            1330, 37780, 13, 271, 760, 5215, 303, 279};
  const std::vector<std::int32_t> kCanB1 = {5215, 321, 6805, 279, 1727, 1330, 37780, 13,
                                            271, 760, 5215, 303, 279, 9039, 37780, 369};

  auto make_base = [&]() {
    Qwen38Gate2Input d;
    d.diagnostic_mode = false;
    d.qwen27b_pins_verified = true;
    d.short_ref_sha_verified = true;
    d.long_ref_sha_verified = true;
    d.allow_canonical_b1_addendum = true;
    d.short_ref_meta_sha_verified = true;
    d.alt_short_ref_b1_sha_verified = true;
    d.alt_short_ref_b1_meta_sha_verified = true;
    d.alt_short_ref_b1_prompt0_ids_match = true;
    d.short_prompt_count = 3;
    d.short_summary = Qwen38KldSummary{141, 0.000594, 0.004903, 0.010555, 137, 137.0 / 141.0, true};
    d.long_prompt_count = 6;
    d.long_summary = Qwen38KldSummary{288, 0.001496, 0.035922, 0.049572, 286, 286.0 / 288.0, true};
    d.chunk_vs_step_prompt_count = 3;
    d.chunk_vs_step_summary = Qwen38KldSummary{3, 0.0, 0.0, 0.0, 3, 1.0, true};
    d.chunk_vs_step_greedy_matches = 48;
    d.chunk_vs_step_greedy_total = 48;
    d.prompt0_b1_eval_executed = true;
    d.prompt0_b1_summary = Qwen38KldSummary{47, 0.00031, 0.0012, 0.0015, 47, 1.0, true};
    return d;
  };

  // 1. Canonical positive A: release-b3 exact 16/16 (both release-only and versioned pass).
  {
    const auto ev = evaluate_prompt0_canonical_trajectory(kRelB3, kRelB3, kCanB1);
    CHECK(ev.matched_intact_canonical && ev.matches_release_b3_intact && !ev.matches_canonical_b1_intact &&
          ev.chosen_realization == "release_b3" && ev.release_b3_matches == 16u && ev.release_b3_first_diff_idx == -1);
    auto d = make_base();
    d.prompt0_greedy_matches = ev.release_b3_matches;
    d.prompt0_greedy_total = 16;
    d.prompt0_b1_greedy_matches = ev.canonical_b1_matches;
    d.prompt0_b1_greedy_total = 16;
    const auto v = evaluate_gate2_verdict(d);
    CHECK(v.contract_version == kGate2ContractVersion);
    CHECK(v.chosen_prompt0_realization == "release_b3");
    CHECK(v.release_only_short_passed && v.release_only_qualification_passed && v.release_only_failures.empty());
    CHECK(v.short_prompts_passed && v.qualification_passed && v.failures.empty());
  }

  // 2. Canonical positive B: canonical-b1 exact 16/16 + valid 47-row B1 distribution + valid pins.
  //    Strict release-only remains FAILED while versioned addendum verdict PASSES.
  {
    const auto ev = evaluate_prompt0_canonical_trajectory(kCanB1, kRelB3, kCanB1);
    CHECK(ev.matched_intact_canonical && !ev.matches_release_b3_intact && ev.matches_canonical_b1_intact &&
          ev.chosen_realization == "canonical_b1" && ev.release_b3_matches == 1u && ev.release_b3_first_diff_idx == 1 &&
          ev.canonical_b1_matches == 16u && ev.canonical_b1_first_diff_idx == -1);
    auto d = make_base();
    d.prompt0_greedy_matches = ev.release_b3_matches;
    d.prompt0_greedy_total = 16;
    d.prompt0_b1_greedy_matches = ev.canonical_b1_matches;
    d.prompt0_b1_greedy_total = 16;
    const auto v = evaluate_gate2_verdict(d);
    CHECK(v.contract_version == kGate2ContractVersion);
    CHECK(v.chosen_prompt0_realization == "canonical_b1");
    CHECK(!v.release_only_short_passed && !v.release_only_qualification_passed && !v.release_only_failures.empty());
    CHECK(v.canonical_b1_branch_passed && v.short_prompts_passed && v.qualification_passed && v.failures.empty());
  }

  // 3. Per-token union hybrid rejection (every token matches at least one canonical ID, neither intact).
  {
    auto hybrid = kRelB3;
    hybrid[1] = kCanB1[1];   // token from b1
    hybrid[15] = kRelB3[15]; // tail back to b3
    const auto ev = evaluate_prompt0_canonical_trajectory(hybrid, kRelB3, kCanB1);
    CHECK(!ev.matched_intact_canonical && ev.chosen_realization == "none");
    auto d = make_base();
    d.prompt0_greedy_matches = ev.release_b3_matches;
    d.prompt0_greedy_total = 16;
    d.prompt0_b1_greedy_matches = ev.canonical_b1_matches;
    d.prompt0_b1_greedy_total = 16;
    const auto v = evaluate_gate2_verdict(d);
    CHECK(v.chosen_prompt0_realization == "none" && !v.short_prompts_passed && !v.qualification_passed);
  }

  // 4. Single altered tail ID rejection (index 15 changed on b3 or b1 -> 15/16 -> rejected).
  {
    for (int which = 0; which < 2; ++which) {
      auto cand = (which == 0) ? kRelB3 : kCanB1;
      cand[15] += 1;
      const auto ev = evaluate_prompt0_canonical_trajectory(cand, kRelB3, kCanB1);
      CHECK(!ev.matched_intact_canonical && ev.chosen_realization == "none");
      auto d = make_base();
      d.prompt0_greedy_matches = ev.release_b3_matches;
      d.prompt0_greedy_total = 16;
      d.prompt0_b1_greedy_matches = ev.canonical_b1_matches;
      d.prompt0_b1_greedy_total = 16;
      const auto v = evaluate_gate2_verdict(d);
      CHECK(!v.short_prompts_passed && !v.qualification_passed);
    }
  }

  // 5. Absent / bad pins & malformed reference fail closed.
  {
    for (int tc = 0; tc < 4; ++tc) {
      auto d = make_base();
      d.prompt0_greedy_matches = 1;
      d.prompt0_greedy_total = 16;
      d.prompt0_b1_greedy_matches = 16;
      d.prompt0_b1_greedy_total = 16;
      if (tc == 0) d.short_ref_meta_sha_verified = false;
      if (tc == 1) d.alt_short_ref_b1_sha_verified = false;
      if (tc == 2) d.alt_short_ref_b1_meta_sha_verified = false;
      if (tc == 3) d.alt_short_ref_b1_prompt0_ids_match = false;
      const auto v = evaluate_gate2_verdict(d);
      CHECK(!v.canonical_b1_branch_passed && !v.qualification_passed && !v.failures.empty());
    }
    // Even if release-b3 matches 16/16, missing bound metadata/alt pin in v1.1 contract fails closed.
    auto d_b3 = make_base();
    d_b3.prompt0_greedy_matches = 16;
    d_b3.prompt0_greedy_total = 16;
    d_b3.alt_short_ref_b1_sha_verified = false;
    CHECK(!evaluate_gate2_verdict(d_b3).qualification_passed);

    // Malformed safetensors file rejected by Qwen38RefSafetensors, and missing file rejected by verify_file_sha256.
    const auto bad_path =
        std::filesystem::temp_directory_path() / ("qwen38_bad_ref_" + std::to_string(::getpid()) + ".safetensors");
    {
      std::ofstream ofs(bad_path, std::ios::binary | std::ios::trunc);
      ofs << "not_a_valid_safetensors_file";
    }
    bool threw = false;
    try {
      Qwen38RefSafetensors bad_ref(bad_path);
    } catch (const std::exception&) {
      threw = true;
    }
    std::error_code ec;
    std::filesystem::remove(bad_path, ec);
    CHECK(threw);
    CHECK(!verify_file_sha256(bad_path, kGate2AltShortRefB1Sha256));
  }

  // 6. Matching B1 IDs (16/16) with corrupted B1 logits / distribution or corrupted release distribution rejected.
  {
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    for (int tc = 0; tc < 6; ++tc) {
      auto d = make_base();
      d.prompt0_greedy_matches = 1;
      d.prompt0_greedy_total = 16;
      d.prompt0_b1_greedy_matches = 16;
      d.prompt0_b1_greedy_total = 16;
      if (tc == 0) d.prompt0_b1_eval_executed = false;
      if (tc == 1) d.prompt0_b1_summary.positions = 46;
      if (tc == 2) d.prompt0_b1_summary.kld_mean = 0.0051;
      if (tc == 3) d.prompt0_b1_summary.kld_max = 0.051;
      if (tc == 4) d.prompt0_b1_summary.top1_agreement = 0.89;
      if (tc == 5) d.prompt0_b1_summary.kld_mean = qnan;  // all_finite still true
      const auto v = evaluate_gate2_verdict(d);
      CHECK(!v.canonical_b1_branch_passed && !v.short_prompts_passed && !v.qualification_passed);
    }
    // Also reject matching B1 IDs if release-b3 141-row distribution fails.
    auto d_rel_bad = make_base();
    d_rel_bad.prompt0_greedy_matches = 1;
    d_rel_bad.prompt0_greedy_total = 16;
    d_rel_bad.prompt0_b1_greedy_matches = 16;
    d_rel_bad.prompt0_b1_greedy_total = 16;
    d_rel_bad.short_summary.kld_mean = 0.006;
    CHECK(!evaluate_gate2_verdict(d_rel_bad).short_prompts_passed &&
          !evaluate_gate2_verdict(d_rel_bad).qualification_passed);

    // Nonfinite logits fail closed in compute_top2_logits.
    const std::vector<float> bad_logits = {1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f};
    CHECK(!compute_top2_logits(bad_logits).finite);
  }
}

}  // namespace

int main() {
  test_kld_and_top1();
  test_kld_summary_and_verdict();
  test_option1_addendum_replay();
  test_streaming_reference_reader();
  std::printf("qwen38_phase1b_cpu_prep: PASS\n");
  return 0;
}

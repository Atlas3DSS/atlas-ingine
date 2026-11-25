#include "backends/qwen38/qwen38_gate2_metrics.h"
#include "backends/qwen38/qwen38_manifest.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
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
}

void test_streaming_reference_reader() {
  const std::filesystem::path ref_dir("/mnt/storage/flare/release-references");
  const std::filesystem::path short_path = ref_dir / "reference.safetensors";
  const std::filesystem::path long_path = ref_dir / "long-code-reference.safetensors";
  if (!std::filesystem::is_regular_file(short_path) || !std::filesystem::is_regular_file(long_path)) {
    std::printf("  skipping on-disk reference safetensors test (files absent)\n");
    return;
  }

  Qwen38RefSafetensors sr(short_path);
  CHECK(sr.verify_sha256(kGate2ShortRefSha256));
  const auto p0_ids = sr.read_i64("prompt_0.prompt_ids");
  const auto p0_gen = sr.read_i64("prompt_0.generated_ids");
  CHECK(p0_ids.size() == 32u && p0_gen.size() == 16u);
  const auto& pf0_meta = sr.at("prompt_0.prefill_logits");
  CHECK(pf0_meta.shape.size() == 2 && pf0_meta.shape[0] == 32 && pf0_meta.shape[1] == 248320);
  std::vector<float> row(static_cast<std::size_t>(pf0_meta.shape[1]));
  sr.read_f32_row("prompt_0.prefill_logits", 31, row);
  double max_v = -1e30;
  std::int32_t argmax_31 = -1;
  for (std::size_t i = 0; i < row.size(); ++i) {
    CHECK(std::isfinite(row[i]));
    if (row[i] > max_v) {
      max_v = row[i];
      argmax_31 = static_cast<std::int32_t>(i);
    }
  }
  CHECK(argmax_31 == static_cast<std::int32_t>(p0_gen[0]));

  Qwen38RefSafetensors lr(long_path);
  CHECK(lr.verify_sha256(kGate2LongCodeRefSha256));
  const auto l0_ids = lr.read_i64("prompt_0.input_ids");
  const auto l0_pos = lr.read_i64("prompt_0.positions");
  CHECK(l0_ids.size() == 4128u && l0_pos.size() == 48u);
  const auto& l0_meta = lr.at("prompt_0.logits");
  CHECK(l0_meta.shape.size() == 2 && l0_meta.shape[0] == 48 && l0_meta.shape[1] == 248320);
  lr.read_f32_row("prompt_0.logits", 0, row);
  for (float x : row) CHECK(std::isfinite(x));
  std::printf("  reference safetensors pins & streaming reader verified (argmax_p0_31=%d == %lld)\n",
              argmax_31, static_cast<long long>(p0_gen[0]));
}

}  // namespace

int main() {
  test_kld_and_top1();
  test_kld_summary_and_verdict();
  test_streaming_reference_reader();
  std::printf("qwen38_phase1b_cpu_prep: PASS\n");
  return 0;
}

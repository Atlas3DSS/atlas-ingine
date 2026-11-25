// Verifies build-aware three-argument randn_bf16 rounding against ActT under both BF16 and FP16
// compilation modes, while keeping explicit 4-argument precision overrides exact.
#include "common/test_util.h"

#include <cstdio>
#include <cstring>

namespace {

bool bits_equal(float a, float b) {
  std::uint32_t ab = 0, bb = 0;
  std::memcpy(&ab, &a, sizeof(ab));
  std::memcpy(&bb, &b, sizeof(bb));
  return ab == bb;
}

}  // namespace

int main() {
  using namespace ingine::test;
  static_assert(ingine::kActIsFP16 == kScalarActIsFP16);
  static_assert(kScalarActIsFP16 == (INGINE_ACT_FP16 != 0));

  Rng base(835), candidate(835), r_bf(835), r_fp(835);
  const auto raw = randn(base, 2304, 1.5);
  const auto three_arg = randn_bf16(candidate, 2304, 1.5);
  const auto explicit_bf16 = randn_bf16(r_bf, 2304, 1.5, false);
  const auto explicit_fp16 = randn_bf16(r_fp, 2304, 1.5, true);

  int changed_vs_act = 0;
  int mismatch_bf = 0;
  int mismatch_fp = 0;
  int bf_vs_fp = 0;
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (!bits_equal(three_arg[i], round_bf16(raw[i]))) ++changed_vs_act;
    if (!bits_equal(explicit_bf16[i], round_bf16_scalar(raw[i]))) ++mismatch_bf;
    if (!bits_equal(explicit_fp16[i], round_fp16_sat_scalar(raw[i]))) ++mismatch_fp;
    if (!bits_equal(explicit_bf16[i], explicit_fp16[i])) ++bf_vs_fp;
  }

  INGINE_CHECK(changed_vs_act == 0);
  INGINE_CHECK(mismatch_bf == 0);
  INGINE_CHECK(mismatch_fp == 0);
  INGINE_CHECK(bf_vs_fp > 0);
  std::printf("PASS core_oracle_precision (%s): three-arg diff=%d/2304 bf16_vs_fp16_diff=%d/2304\n",
              kScalarActIsFP16 ? "FP16" : "BF16", changed_vs_act, bf_vs_fp);
  return 0;
}

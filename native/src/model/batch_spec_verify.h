// Pure CPU reference validators for batched speculative decoding (#851):
// 1. Exact Philox4x32-10 draft proposal reconstruction (matching kernels/draft/batch_draft.cu:1066 & draft_device.cuh)
//    + exact splitmix64 uniform/Gumbel acceptance & residual verification (matching kernels/head/head.cu & accept.cu).
// 2. Strict cross-gear common-prefix logit divergence validation (binds both plain & spec argmax winners, enforces
//    finiteness, non-empty full vocab, nonnegative margins, and tight numerical near-tie bounds).
// CUDA-free (host-callable) so BOTH CPU CTests (test_batch_scheduler) and GPU parity (test_batch_spec_parity)
// execute the exact same validators and negative/corrupt controls.
#pragma once

#include "model/batch_spec_trace.h"

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace ingine::model::spec_verify {

inline constexpr int kGamma = 7;
inline constexpr int kCandidates = 1024;

// ---- Exact CPU mirror of kernels/draft/rng.h (Philox4x32-10 + Gumbel) & draft_device.cuh (pack_argmax) ----

inline void philox_round(std::uint32_t (&c)[4], const std::uint32_t (&k)[2]) {
  const std::uint64_t p0 = static_cast<std::uint64_t>(0xD2511F53u) * c[0];
  const std::uint64_t p1 = static_cast<std::uint64_t>(0xCD9E8D57u) * c[2];
  const std::uint32_t hi0 = static_cast<std::uint32_t>(p0 >> 32), lo0 = static_cast<std::uint32_t>(p0);
  const std::uint32_t hi1 = static_cast<std::uint32_t>(p1 >> 32), lo1 = static_cast<std::uint32_t>(p1);
  c[0] = hi1 ^ c[1] ^ k[0];
  c[1] = lo1;
  c[2] = hi0 ^ c[3] ^ k[1];
  c[3] = lo0;
}

inline void philox_generate(std::uint32_t (&c)[4], std::uint64_t seed) {
  std::uint32_t k[2] = {static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32)};
  for (int r = 0; r < 10; ++r) {
    philox_round(c, k);
    k[0] += 0x9E3779B9u;
    k[1] += 0xBB67AE85u;
  }
}

inline float gumbel_from_bits_host(std::uint64_t k) {
  const int z = std::min(int(std::countl_zero(k)), 40);
  const float m = static_cast<float>(static_cast<std::uint32_t>(k >> (40 - z))) + (z == 40 ? 0.5f : 0.0f);
  const float v = m * std::bit_cast<float>(static_cast<std::uint32_t>((127 - 24 - z) << 23));
  const float e = v < 0.5f ? -std::log1p(-v) : -std::log(1.0f - v);
  return -std::log(e);
}

inline float philox_gumbel(std::uint64_t seed, std::uint64_t stream, std::uint32_t row, std::uint32_t col) {
  std::uint32_t c[4] = {static_cast<std::uint32_t>(stream), static_cast<std::uint32_t>(stream >> 32), row, col};
  philox_generate(c, seed);
  return gumbel_from_bits_host((static_cast<std::uint64_t>(c[0]) << 32) | c[1]);
}

inline std::uint32_t draft_float_key(float f) {
  const std::uint32_t u = std::bit_cast<std::uint32_t>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

inline std::uint64_t draft_pack_argmax(float score, std::uint32_t col) {
  return (static_cast<std::uint64_t>(draft_float_key(score)) << 32) | (0xffffffffu - col);
}

inline std::uint32_t draft_unpack_col(std::uint64_t packed) {
  return 0xffffffffu - static_cast<std::uint32_t>(packed & 0xffffffffu);
}

// Reconstructs the EXACT selected shortlist column in [0, kCandidates) for draft row `row` using
// kernels/draft/batch_draft.cu:1066 (Philox4x32-10 Gumbel-max + pack_argmax lowest-column tie break).
inline int reconstruct_draft_column(std::span<const float> row_logits, float temperature, std::uint64_t draft_seed,
                                    std::uint64_t draft_draws, int row) {
  std::uint64_t best = 0;
  const float denom = std::max(temperature, 1e-6f);
  for (int k = 0; k < int(row_logits.size()); ++k) {
    const float l = row_logits[std::size_t(k)];
    const float score =
        (temperature > 0.f) ? (l / denom + philox_gumbel(draft_seed, draft_draws, std::uint32_t(row), std::uint32_t(k)))
                            : l;
    const std::uint64_t p = draft_pack_argmax(score, std::uint32_t(k));
    if (k == 0 || p > best) best = p;
  }
  return int(draft_unpack_col(best));
}

// ---- Exact CPU mirror of kernels/head/head.cu (splitmix64 uniform + Gumbel-max residual draw) ----

inline std::uint64_t head_splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

inline std::uint64_t head_random64(std::uint64_t seed, std::uint64_t counter, std::uint32_t row, std::uint32_t index) {
  const std::uint64_t k = head_splitmix64(seed ^ (counter * 0xD1B54A32D192ED03ull));
  return head_splitmix64(k ^ ((static_cast<std::uint64_t>(row) << 32) | index));
}

inline float head_uniform(std::uint64_t seed, std::uint64_t counter, std::uint32_t i) {
  const std::uint32_t bits = static_cast<std::uint32_t>(head_random64(seed, counter, 0xffffffffu, i) >> 40);
  return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

inline float head_gumbel(std::uint64_t seed, std::uint64_t counter, std::uint32_t row, std::uint32_t index) {
  return gumbel_from_bits_host(head_random64(seed, counter, row, index));
}

inline std::int32_t head_sample_from_probs(std::span<const float> probs, std::uint64_t seed, std::uint64_t counter) {
  float bv = -std::numeric_limits<float>::infinity();
  int bi = INT_MAX;
  for (int v = 0; v < int(probs.size()); ++v) {
    const float p = probs[std::size_t(v)];
    if (p > 0.f) {
      const float score = std::log(p) + head_gumbel(seed, counter, 0u, std::uint32_t(v));
      if (score > bv || (score == bv && v < bi)) {
        bv = score;
        bi = v;
      }
    }
  }
  return bi == INT_MAX ? 0 : bi;
}

inline std::int32_t head_argmax(std::span<const float> logits) {
  float bv = -std::numeric_limits<float>::infinity();
  int bi = INT_MAX;
  for (int v = 0; v < int(logits.size()); ++v) {
    const float x = logits[std::size_t(v)];
    if (x > bv || (x == bv && v < bi)) {
      bv = x;
      bi = v;
    }
  }
  return bi == INT_MAX ? 0 : bi;
}

struct ValidationResult {
  bool ok = true;
  std::string reason;
  float max_diff = 0.f;
  float margin_plain = 0.f;
  float margin_spec = 0.f;
  std::int32_t argmax_plain = -1;
  std::int32_t argmax_spec = -1;
};

inline ValidationResult fail_res(std::string why) {
  ValidationResult r;
  r.ok = false;
  r.reason = std::move(why);
  return r;
}

// Validates a complete sampled speculative round against the exact fixed-draw CPU reference:
//   - Finiteness of all logits, probs, qstats, uniforms, residuals
//   - Exact Philox4x32-10 draft column & token ID reconstruction for ALL 7 draft rows
//   - Exact qstats/conf match vs candidate logits
//   - Exact splitmix64 uniform draws bit-for-bit
//   - Exact speculative rejection accept count
//   - Exact residual distribution (including all-zero fallback) and Gumbel-max residual choice
inline ValidationResult validate_sampled_trace(const BatchSpecSlotTrace& tr, int vocab_size) {
  if (!tr.has_draft || !tr.has_sampled_details) return fail_res("missing draft or sampled trace details");
  if (tr.w_cap < 2 || tr.w_cap > 8) return fail_res("invalid w_cap for sampled accept");
  if (!(tr.temperature > 0.f) || !std::isfinite(tr.temperature)) return fail_res("non-positive or nonfinite temperature");
  const int n = tr.w_cap - 1;
  if (int(tr.uniforms.size()) != n) return fail_res("uniforms size mismatch");
  if (int(tr.target_probs.size()) != tr.w_cap * vocab_size) return fail_res("target_probs size mismatch");
  if (int(tr.residual_probs.size()) != vocab_size) return fail_res("residual_probs size mismatch");
  if (int(tr.draft_cand_idx.size()) != kGamma * kCandidates ||
      int(tr.draft_cand_logits.size()) != kGamma * kCandidates)
    return fail_res("candidate idx/logits size mismatch");

  auto all_fin = [](std::span<const float> s) {
    for (float x : s)
      if (!std::isfinite(x)) return false;
    return true;
  };
  if (!all_fin(tr.uniforms)) return fail_res("nonfinite uniforms");
  if (!all_fin(tr.target_probs)) return fail_res("nonfinite target_probs");
  if (!all_fin(tr.residual_probs)) return fail_res("nonfinite residual_probs");
  if (!all_fin(tr.draft_cand_logits)) return fail_res("nonfinite draft_cand_logits");

  const float T = tr.temperature;
  std::vector<std::unordered_map<std::int32_t, float>> q_maps(static_cast<std::size_t>(kGamma));

  // 1. Reconstruct EVERY draft row 0..kGamma-1 using exact Philox4x32-10 Gumbel-max + pack_argmax tie-break!
  for (int r = 0; r < kGamma; ++r) {
    const std::span<const float> row_l(tr.draft_cand_logits.data() + std::size_t(r) * kCandidates, kCandidates);
    const std::span<const std::int32_t> row_id(tr.draft_cand_idx.data() + std::size_t(r) * kCandidates, kCandidates);
    const int exp_col = reconstruct_draft_column(row_l, T, tr.draft_seed, tr.draft_draws, r);
    const std::int32_t exp_tok = row_id[std::size_t(exp_col)];
    if (tr.draft_tokens[r] != exp_tok) {
      return fail_res("Philox draft token mismatch at row " + std::to_string(r) + ": expected " +
                      std::to_string(exp_tok) + " (col " + std::to_string(exp_col) + "), got " +
                      std::to_string(tr.draft_tokens[r]));
    }

    float mx_l = -std::numeric_limits<float>::infinity();
    float mx_t = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < kCandidates; ++k) {
      if (row_l[std::size_t(k)] > mx_l) mx_l = row_l[std::size_t(k)];
      const float x = row_l[std::size_t(k)] / T;
      if (x > mx_t) mx_t = x;
    }
    double sum_l = 0.0, sum_t = 0.0;
    for (int k = 0; k < kCandidates; ++k) {
      sum_l += std::exp(double(row_l[std::size_t(k)] - mx_l));
      sum_t += std::exp(double(row_l[std::size_t(k)] / T - mx_t));
    }
    const float ref_conf = float(1.0 / sum_l);
    const float ref_qs = float(sum_t);
    if (!std::isfinite(tr.draft_conf[r]) || std::fabs(tr.draft_conf[r] - ref_conf) > 1e-4f)
      return fail_res("draft_conf mismatch at row " + std::to_string(r));
    if (!std::isfinite(tr.draft_qstats[r][0]) || std::fabs(tr.draft_qstats[r][0] - mx_t) > 1e-4f)
      return fail_res("draft_qstats max mismatch at row " + std::to_string(r));
    if (!std::isfinite(tr.draft_qstats[r][1]) ||
        std::fabs(tr.draft_qstats[r][1] - ref_qs) > 1e-2f * std::max(1.f, ref_qs))
      return fail_res("draft_qstats sum mismatch at row " + std::to_string(r));

    for (int k = 0; k < kCandidates; ++k) {
      const float qv = std::exp((row_l[std::size_t(k)] / T) - tr.draft_qstats[r][0]) / tr.draft_qstats[r][1];
      q_maps[std::size_t(r)][row_id[std::size_t(k)]] = qv;
    }
  }

  // 2. Verify exact splitmix64 uniform RNG draws bit-for-bit.
  for (int r = 0; r < n; ++r) {
    const float exp_u = head_uniform(tr.accept_rng_seed, tr.accept_rng_counter_u, std::uint32_t(r));
    if (tr.uniforms[std::size_t(r)] != exp_u) {
      return fail_res("uniform RNG draw mismatch at r=" + std::to_string(r));
    }
  }

  // 3. Verify exact speculative rejection accept count.
  int ref_acc = 0;
  for (; ref_acc < n; ++ref_acc) {
    const std::int32_t d = tr.draft_tokens[ref_acc];
    const float qd = q_maps[std::size_t(ref_acc)].at(d);
    const float pd = tr.target_probs[std::size_t(ref_acc) * vocab_size + std::size_t(d)];
    if (!(tr.uniforms[std::size_t(ref_acc)] * qd < pd)) break;
  }
  if (ref_acc != tr.accepted) {
    return fail_res("accepted count mismatch: ref=" + std::to_string(ref_acc) + " got=" + std::to_string(tr.accepted));
  }

  // 4. Verify exact residual distribution (including all-zero fallback) and Gumbel-max residual choice.
  std::vector<float> ref_resid(static_cast<std::size_t>(vocab_size), 0.f);
  double resid_sum = 0.0;
  const float* pa = tr.target_probs.data() + std::size_t(ref_acc) * vocab_size;
  for (int x = 0; x < vocab_size; ++x) {
    float w = pa[x];
    if (ref_acc < n) {
      auto it = q_maps[std::size_t(ref_acc)].find(x);
      const float qx = (it != q_maps[std::size_t(ref_acc)].end()) ? it->second : 0.f;
      w = std::max(pa[x] - qx, 0.f);
    }
    ref_resid[std::size_t(x)] = w;
    resid_sum += double(w);
  }
  if (!(resid_sum > 0.0)) {
    for (int x = 0; x < vocab_size; ++x) ref_resid[std::size_t(x)] = pa[x];
  }

  float max_rdiff = 0.f;
  for (int x = 0; x < vocab_size; ++x) {
    const float d = std::fabs(tr.residual_probs[std::size_t(x)] - ref_resid[std::size_t(x)]);
    if (!std::isfinite(d)) return fail_res("nonfinite residual diff");
    if (d > max_rdiff) max_rdiff = d;
  }
  if (max_rdiff > 1e-5f) return fail_res("residual distribution mismatch: max_diff=" + std::to_string(max_rdiff));

  const std::int32_t next_gpu_resid =
      head_sample_from_probs(tr.residual_probs, tr.accept_rng_seed, tr.accept_rng_counter_resid);
  const std::int32_t next_cpu_resid =
      head_sample_from_probs(ref_resid, tr.accept_rng_seed, tr.accept_rng_counter_resid);
  if (tr.next_token != next_gpu_resid || tr.next_token != next_cpu_resid) {
    return fail_res("residual Gumbel-max draw mismatch: got=" + std::to_string(tr.next_token) +
                    " ref_gpu=" + std::to_string(next_gpu_resid) + " ref_cpu=" + std::to_string(next_cpu_resid));
  }

  ValidationResult ok;
  ok.ok = true;
  ok.max_diff = max_rdiff;
  return ok;
}

// Strictly validates a claimed cross-gear token divergence at a common prefix:
//   - Rejects empty / wrong-sized / non-finite captures
//   - Rejects equal or out-of-range tokens
//   - Binds BOTH winners: argmax(l_plain) MUST equal plain_tok AND argmax(l_spec) MUST equal spec_tok
//   - Enforces BOTH margins nonnegative (margin_plain >= 0 && margin_spec >= 0) and consistent with max_logit_diff!
inline ValidationResult validate_cross_gear_divergence(std::span<const float> l_plain, std::span<const float> l_spec,
                                                       std::int32_t plain_tok, std::int32_t spec_tok,
                                                       int expected_vocab, float max_allowed_diff = 0.6f) {
  if (expected_vocab <= 0 || int(l_plain.size()) != expected_vocab || int(l_spec.size()) != expected_vocab)
    return fail_res("incomplete or empty logit capture");
  if (plain_tok < 0 || plain_tok >= expected_vocab || spec_tok < 0 || spec_tok >= expected_vocab)
    return fail_res("token id out of vocab range");
  if (plain_tok == spec_tok) return fail_res("claimed divergent tokens are identical");

  float max_diff = 0.f;
  for (int v = 0; v < expected_vocab; ++v) {
    const float a = l_plain[std::size_t(v)];
    const float b = l_spec[std::size_t(v)];
    if (!std::isfinite(a) || !std::isfinite(b)) return fail_res("nonfinite logit in cross-gear capture");
    const float d = std::fabs(b - a);
    if (!std::isfinite(d)) return fail_res("nonfinite logit difference");
    if (d > max_diff) max_diff = d;
  }

  const std::int32_t am_plain = head_argmax(l_plain);
  const std::int32_t am_spec = head_argmax(l_spec);
  if (am_plain != plain_tok) {
    return fail_res("plain logit capture argmax (" + std::to_string(am_plain) +
                    ") does not match observed plain_tok (" + std::to_string(plain_tok) + ")");
  }
  if (am_spec != spec_tok) {
    return fail_res("spec logit capture argmax (" + std::to_string(am_spec) +
                    ") does not match observed spec_tok (" + std::to_string(spec_tok) + ")");
  }

  const float mp = l_plain[std::size_t(plain_tok)] - l_plain[std::size_t(spec_tok)];
  const float ms = l_spec[std::size_t(spec_tok)] - l_spec[std::size_t(plain_tok)];
  if (!(mp >= 0.f) || !(ms >= 0.f)) return fail_res("negative winner margin");
  if (max_diff > max_allowed_diff) {
    return fail_res("cross-gear max_logit_diff exceeds bound: " + std::to_string(max_diff));
  }
  if (mp + ms > 2.f * max_diff + 1e-4f) {
    return fail_res("inconsistent margins relative to max_logit_diff");
  }

  ValidationResult r;
  r.ok = true;
  r.max_diff = max_diff;
  r.margin_plain = mp;
  r.margin_spec = ms;
  r.argmax_plain = am_plain;
  r.argmax_spec = am_spec;
  return r;
}

}  // namespace ingine::model::spec_verify

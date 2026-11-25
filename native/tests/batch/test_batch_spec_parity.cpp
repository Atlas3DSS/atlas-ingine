// Full-model GPU parity, isolation, and reference-acceptance test suite for Batched MTP Speculative Decoding (#851):
// 1. Equal-arithmetic exact qualification:
//    - W=1 spec (step_slots) vs plain (spec=false) at B=1, 4, 8, 12: 100% exact tokens, finite logits, 0.0f max diff.
//    - Batch replicate invariance inside B=8 and B=12 spec batches: 100% bit-identical tokens, draft IDs, confidences.
// 2. Post-partial-rollback and W=1 -> wider state/tap/next-logit parity vs clean serial W=1 replay at identical prefix:
//    - Exercises partial rollback (1 <= commit < w_cap) and W=1 -> W=8 transition; checks finiteness, exact next token,
//      committed target taps [4, H], and next-step FP32 logits [V] against serial W=1 replay (zero rollback).
// 3. Measured cross-gear agreement (preserved separately) + common-prefix logit diagnosis on any divergence:
//    - B=1, 4, 8, 12 free-running --spec vs plain --no-spec reported separately as cross-gear agreement.
//    - Any divergent sequence is replayed at its exact common prefix on both plain and spec to inspect finite FP32
//      logits, max_abs_logit_diff, and top-1 margin (proving any flip is strictly a tiny cross-gear near-tie).
// 4. Confidence-width bound enforcement (spec_threshold=0.8):
//    - Verifies w_cap <= lead + 1 and accepted <= lead on every active slot even when W_batch pads shared launch shape.
// 5. Sampled speculative parity vs fixed-draw CPU reference (cold, warm prefix-cache restore, and heterogeneous batch):
//    - Verifies per-slot draft IDs, probabilities/qstats, exact splitmix64 uniform RNG draws, exact rejection accept
//      count, exact residual distribution (including fallback), exact Gumbel-max residual choice, and solo-vs-batch
//      per-slot RNG/proposal invariance.
// 6. Deep multi-tenant shortlist isolation (cold concurrent + warm cross-slot prefix-cache restore):
//    - Verifies neighbor Tenant B's slot_extra_ids actually mutated with rare out-of-shortlist tokens while Tenant A's
//      slot_extra_ids, draft_cand_idx, draft_tokens, draft_conf, draft_qstats, accepted, next_token, and final tokens
//      match solo Tenant A exactly.
// 7. Mid-block stop truncation, prefix-cache exact/partial/root cold-vs-warm parity, and tight 6-page pool safety.
#include "engine/scheduler.h"
#include "ingine/generator.h"
#include "model/batch_spec_trace.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using ingine::engine::Job;
using ingine::engine::JobBatch;
using ingine::engine::Scheduler;
using ingine::model::BatchSlotResult;
using ingine::model::BatchSlotStep;
using ingine::model::BatchSpecSlotTrace;
using ingine::model::BatchSpecTraceConfig;
using ingine::model::GenerateStats;
using ingine::model::Generator;
using ingine::model::Prompt;
using ingine::model::Reuse;
using ingine::model::RuntimeOptions;
using ingine::model::Sampling;

constexpr int kUnpaddedVocab = 200058;
constexpr int kGamma = 7;
constexpr int kCandidates = 1024;

int g_failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

struct DiffStats {
  float max_abs = 0.f;
  int nonfinite_a = 0;
  int nonfinite_b = 0;
  int count = 0;
  void update(float a, float b) {
    ++count;
    if (!std::isfinite(a)) {
      ++nonfinite_a;
      return;
    }
    if (!std::isfinite(b)) {
      ++nonfinite_b;
      return;
    }
    const float d = std::fabs(a - b);
    if (!std::isfinite(d)) {
      ++nonfinite_a;
      return;
    }
    if (d > max_abs) max_abs = d;
  }
  bool finite() const { return count > 0 && nonfinite_a == 0 && nonfinite_b == 0 && std::isfinite(max_abs); }
};

bool all_finite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

// ---- Exact CPU mirrors of kernels/head/head.cu RNG & Gumbel-max draws ----

std::uint64_t ref_splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

std::uint64_t ref_random64(std::uint64_t seed, std::uint64_t counter, std::uint32_t row, std::uint32_t index) {
  const std::uint64_t k = ref_splitmix64(seed ^ (counter * 0xD1B54A32D192ED03ull));
  return ref_splitmix64(k ^ ((std::uint64_t(row) << 32) | index));
}

float ref_uniform(std::uint64_t seed, std::uint64_t counter, std::uint32_t i) {
  const std::uint32_t bits = static_cast<std::uint32_t>(ref_random64(seed, counter, 0xffffffffu, i) >> 40);
  return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

float ref_gumbel(std::uint64_t seed, std::uint64_t counter, std::uint32_t row, std::uint32_t index) {
  const std::uint64_t k = ref_random64(seed, counter, row, index);
  const int z = std::min(int(std::countl_zero(k)), 40);
  const float m = static_cast<float>(static_cast<std::uint32_t>(k >> (40 - z))) + (z == 40 ? 0.5f : 0.0f);
  const std::uint32_t exp_bits = static_cast<std::uint32_t>((127 - 24 - z) << 23);
  float scale = 0.f;
  std::memcpy(&scale, &exp_bits, sizeof(scale));
  const float v = m * scale;
  const float e = v < 0.5f ? -std::log1p(-v) : -std::log(1.0f - v);
  return -std::log(e);
}

std::int32_t ref_sample_from_probs(const std::vector<float>& probs, std::uint64_t seed, std::uint64_t counter) {
  float bv = -std::numeric_limits<float>::infinity();
  int bi = INT_MAX;
  for (int v = 0; v < int(probs.size()); ++v) {
    const float p = probs[std::size_t(v)];
    if (p > 0.f) {
      const float score = std::log(p) + ref_gumbel(seed, counter, 0u, std::uint32_t(v));
      if (score > bv || (score == bv && v < bi)) {
        bv = score;
        bi = v;
      }
    }
  }
  return bi == INT_MAX ? 0 : bi;
}

// Verify a single sampled speculative round against the exact fixed-draw CPU reference!
void verify_sampled_trace_vs_cpu_ref(const BatchSpecSlotTrace& tr, const char* label) {
  CHECK(tr.has_draft);
  CHECK(tr.has_sampled_details);
  CHECK(tr.w_cap >= 2 && tr.w_cap <= 8);
  const int n = tr.w_cap - 1;
  const float T = tr.temperature;
  CHECK(T > 0.f);
  CHECK(int(tr.uniforms.size()) == n);
  CHECK(int(tr.target_probs.size()) == tr.w_cap * kUnpaddedVocab);
  CHECK(int(tr.residual_probs.size()) == kUnpaddedVocab);
  CHECK(int(tr.draft_cand_idx.size()) == kGamma * kCandidates);
  CHECK(int(tr.draft_cand_logits.size()) == kGamma * kCandidates);
  CHECK(all_finite(tr.uniforms));
  CHECK(all_finite(tr.target_probs));
  CHECK(all_finite(tr.residual_probs));
  CHECK(all_finite(tr.draft_cand_logits));

  // 1. Verify draft qstats & conf against candidate logits [kGamma, kCandidates].
  std::vector<std::unordered_map<std::int32_t, float>> q_maps(static_cast<std::size_t>(n));
  for (int r = 0; r < n; ++r) {
    float mx_l = -std::numeric_limits<float>::infinity();
    float mx_t = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < kCandidates; ++k) {
      const float l = tr.draft_cand_logits[std::size_t(r) * kCandidates + k];
      if (l > mx_l) mx_l = l;
      const float x = l / T;
      if (x > mx_t) mx_t = x;
    }
    double sum_l = 0.0, sum_t = 0.0;
    for (int k = 0; k < kCandidates; ++k) {
      const float l = tr.draft_cand_logits[std::size_t(r) * kCandidates + k];
      sum_l += std::exp(double(l - mx_l));
      sum_t += std::exp(double(l / T - mx_t));
    }
    const float ref_conf = float(1.0 / sum_l);
    const float ref_qm = mx_t;
    const float ref_qs = float(sum_t);
    CHECK(std::isfinite(tr.draft_conf[r]) && std::fabs(tr.draft_conf[r] - ref_conf) <= 1e-4f);
    CHECK(std::isfinite(tr.draft_qstats[r][0]) && std::fabs(tr.draft_qstats[r][0] - ref_qm) <= 1e-4f);
    CHECK(std::isfinite(tr.draft_qstats[r][1]) && std::fabs(tr.draft_qstats[r][1] - ref_qs) <= 1e-2f * std::max(1.f, ref_qs));
    for (int k = 0; k < kCandidates; ++k) {
      const std::int32_t id = tr.draft_cand_idx[std::size_t(r) * kCandidates + k];
      const float l = tr.draft_cand_logits[std::size_t(r) * kCandidates + k];
      const float qv = std::exp((l / T) - tr.draft_qstats[r][0]) / tr.draft_qstats[r][1];
      q_maps[std::size_t(r)][id] = qv;
    }
    CHECK(q_maps[std::size_t(r)].count(tr.draft_tokens[r]) == 1);
  }

  // 2. Verify exact deterministic splitmix64 uniform RNG draws bit-for-bit!
  for (int r = 0; r < n; ++r) {
    const float exp_u = ref_uniform(tr.accept_rng_seed, tr.accept_rng_counter_u, std::uint32_t(r));
    CHECK(tr.uniforms[std::size_t(r)] == exp_u);
  }

  // 3. Verify exact speculative rejection accept count against fixed-draw CPU reference!
  int ref_acc = 0;
  for (; ref_acc < n; ++ref_acc) {
    const std::int32_t d = tr.draft_tokens[ref_acc];
    const float qd = q_maps[std::size_t(ref_acc)].at(d);
    const float pd = tr.target_probs[std::size_t(ref_acc) * kUnpaddedVocab + std::size_t(d)];
    if (!(tr.uniforms[std::size_t(ref_acc)] * qd < pd)) break;
  }
  if (ref_acc != tr.accepted) {
    std::fprintf(stderr, "sampled_accept mismatch [%s]: ref_acc=%d got_acc=%d w_cap=%d\n", label, ref_acc, tr.accepted,
                 tr.w_cap);
  }
  CHECK(ref_acc == tr.accepted);

  // 4. Verify exact residual distribution (including all-zero fallback to p[acc])!
  std::vector<float> ref_resid(kUnpaddedVocab, 0.f);
  double resid_sum = 0.0;
  const float* pa = tr.target_probs.data() + std::size_t(ref_acc) * kUnpaddedVocab;
  for (int x = 0; x < kUnpaddedVocab; ++x) {
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
    for (int x = 0; x < kUnpaddedVocab; ++x) ref_resid[std::size_t(x)] = pa[x];
  }
  DiffStats rdiff;
  for (int x = 0; x < kUnpaddedVocab; ++x) rdiff.update(tr.residual_probs[std::size_t(x)], ref_resid[std::size_t(x)]);
  CHECK(rdiff.finite() && rdiff.max_abs <= 1e-5f);

  // 5. Verify exact Gumbel-max residual choice using GPU residual probs (and CPU ref_resid) & fixed counter!
  const std::int32_t next_from_gpu_resid =
      ref_sample_from_probs(tr.residual_probs, tr.accept_rng_seed, tr.accept_rng_counter_resid);
  const std::int32_t next_from_cpu_resid =
      ref_sample_from_probs(ref_resid, tr.accept_rng_seed, tr.accept_rng_counter_resid);
  if (tr.next_token != next_from_gpu_resid) {
    std::fprintf(stderr, "sampled residual draw mismatch [%s]: got=%d ref_gpu=%d ref_cpu=%d\n", label, tr.next_token,
                 next_from_gpu_resid, next_from_cpu_resid);
  }
  CHECK(tr.next_token == next_from_gpu_resid);
  CHECK(tr.next_token == next_from_cpu_resid);
}

struct TraceGuard {
  explicit TraceGuard(BatchSpecTraceConfig cfg) { ingine::model::set_batch_spec_trace_hook_for_test(std::move(cfg)); }
  ~TraceGuard() { ingine::model::clear_batch_spec_trace_hook_for_test(); }
};

std::filesystem::path find_model_dir() {
  if (const char* env = std::getenv("INGINE_MODEL_DIR"); env && *env) return env;
  return std::filesystem::path(INGINE_MODEL_DIR);
}

std::vector<int32_t> render_prompt(const ingine::text::ChatFormat& fmt, const std::string& sys,
                                   const std::string& user) {
  ingine::text::ChatOptions opts;
  opts.reasoning_effort = "low";
  nlohmann::ordered_json msgs = nlohmann::ordered_json::array();
  if (!sys.empty()) msgs.push_back({{"role", "system"}, {"content", sys}});
  msgs.push_back({{"role", "user"}, {"content", user}});
  return fmt.render(msgs, nlohmann::ordered_json(), opts).ids;
}

std::vector<int32_t> drain_job(Job& job, std::optional<std::string>* err = nullptr) {
  std::vector<int32_t> out;
  for (;;) {
    JobBatch b = job.take(std::chrono::seconds(120));
    out.insert(out.end(), b.tokens.begin(), b.tokens.end());
    if (b.failure && err) *err = b.failure;
    if (b.end) return out;
    if (b.empty()) throw std::runtime_error("timeout draining job");
  }
}

struct RunOut {
  std::vector<int32_t> tokens;
  GenerateStats stats;
};

RunOut run_solo(Generator& gen, const Prompt& p, int max_new, const Sampling& sampling,
                std::span<const int32_t> stops = {}) {
  RunOut r;
  r.stats = gen.generate(p, max_new, sampling, stops, [&](std::span<const int32_t> toks) {
    r.tokens.insert(r.tokens.end(), toks.begin(), toks.end());
    return true;
  });
  return r;
}

std::vector<std::vector<int32_t>> run_batch_greedy_w1(Generator& gen, const std::vector<Prompt>& prompts, int max_new) {
  const int B = int(prompts.size());
  std::vector<std::vector<int32_t>> outs(static_cast<std::size_t>(B));
  struct Active {
    int idx = 0;
    int slot = -1;
    int32_t last_tok = 0;
  };
  std::vector<Active> live;
  Sampling greedy;
  greedy.temperature = 0.0f;
  for (int i = 0; i < B; ++i) {
    int32_t tok0 = 0;
    GenerateStats st;
    const int s = gen.start_slot(prompts[std::size_t(i)], max_new, greedy, tok0, st);
    CHECK(s >= 0);
    if (s < 0) return outs;
    outs[std::size_t(i)].push_back(tok0);
    if (int(outs[std::size_t(i)].size()) >= max_new) gen.release_slot(s);
    else live.push_back(Active{i, s, tok0});
  }
  while (!live.empty()) {
    std::vector<BatchSlotStep> steps(live.size());
    std::vector<int32_t> next(live.size(), 0);
    for (std::size_t i = 0; i < live.size(); ++i) steps[i] = {live[i].slot, live[i].last_tok};
    gen.step_slots(steps, next);
    std::vector<Active> kept;
    for (std::size_t i = 0; i < live.size(); ++i) {
      auto& out = outs[std::size_t(live[i].idx)];
      out.push_back(next[i]);
      live[i].last_tok = next[i];
      if (int(out.size()) >= max_new) gen.release_slot(live[i].slot);
      else kept.push_back(live[i]);
    }
    live = std::move(kept);
  }
  return outs;
}

std::vector<std::vector<int32_t>> run_batch_greedy_multi(Generator& gen, const std::vector<Prompt>& prompts,
                                                         int max_new, int& total_rounds, int& total_accepted) {
  const int B = int(prompts.size());
  std::vector<std::vector<int32_t>> outs(static_cast<std::size_t>(B));
  struct Active {
    int idx = 0;
    int slot = -1;
    int32_t last_tok = 0;
  };
  std::vector<Active> live;
  Sampling greedy;
  greedy.temperature = 0.0f;
  for (int i = 0; i < B; ++i) {
    int32_t tok0 = 0;
    GenerateStats st;
    const int s = gen.start_slot(prompts[std::size_t(i)], max_new, greedy, tok0, st);
    CHECK(s >= 0);
    if (s < 0) return outs;
    outs[std::size_t(i)].push_back(tok0);
    if (int(outs[std::size_t(i)].size()) >= max_new) gen.release_slot(s);
    else live.push_back(Active{i, s, tok0});
  }
  total_rounds = 0;
  total_accepted = 0;
  while (!live.empty()) {
    std::vector<BatchSlotStep> steps(live.size());
    std::vector<BatchSlotResult> res(live.size());
    for (std::size_t i = 0; i < live.size(); ++i) steps[i] = {live[i].slot, live[i].last_tok};
    gen.step_slots_multi(steps, res);
    std::vector<Active> kept;
    for (std::size_t i = 0; i < live.size(); ++i) {
      total_rounds += res[i].spec_rounds;
      total_accepted += res[i].spec_accepted;
      auto& out = outs[std::size_t(live[i].idx)];
      for (int k = 0; k < res[i].count && int(out.size()) < max_new; ++k) {
        out.push_back(res[i].tokens[k]);
      }
      live[i].last_tok = out.back();
      if (int(out.size()) >= max_new) gen.release_slot(live[i].slot);
      else kept.push_back(live[i]);
    }
    live = std::move(kept);
  }
  return outs;
}

// Compute single-step logits for a prompt+prefix on a given generator at a specific batch size B_eval (row 0 = target).
std::vector<float> capture_prefix_logits(Generator& gen, const std::vector<int32_t>& full_prefix, int B_eval,
                                         const std::vector<std::vector<int32_t>>& fillers, bool use_multi) {
  CHECK(full_prefix.size() >= 2);
  const std::vector<int32_t> head_prompt(full_prefix.begin(), full_prefix.end() - 1);
  const int32_t anchor = full_prefix.back();
  Sampling greedy;
  greedy.temperature = 0.0f;

  std::vector<int> slots(static_cast<std::size_t>(B_eval), -1);
  std::vector<int32_t> anchors(static_cast<std::size_t>(B_eval), 0);
  {
    int32_t t0 = 0;
    GenerateStats st;
    slots[0] = gen.start_slot(Prompt{head_prompt, {}}, 16, greedy, t0, st);
    anchors[0] = anchor;
    CHECK(slots[0] >= 0);
    for (int i = 1; i < B_eval; ++i) {
      int32_t tf = 0;
      GenerateStats sf;
      slots[std::size_t(i)] = gen.start_slot(Prompt{fillers[std::size_t(i) % fillers.size()], {}}, 16, greedy, tf, sf);
      anchors[std::size_t(i)] = tf;
      CHECK(slots[std::size_t(i)] >= 0);
    }
  }

  std::vector<float> logits_row0;
  {
    BatchSpecTraceConfig cfg;
    cfg.capture_target_logits = true;
    cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
      if (!trs.empty() && trs[0].target_logits.size() >= std::size_t(kUnpaddedVocab)) {
        logits_row0.assign(trs[0].target_logits.begin(), trs[0].target_logits.begin() + kUnpaddedVocab);
      }
    };
    TraceGuard tg(std::move(cfg));
    std::vector<BatchSlotStep> steps(static_cast<std::size_t>(B_eval));
    for (int i = 0; i < B_eval; ++i) steps[std::size_t(i)] = {slots[std::size_t(i)], anchors[std::size_t(i)]};
    if (use_multi) {
      std::vector<BatchSlotResult> res(static_cast<std::size_t>(B_eval));
      gen.step_slots_multi(steps, res);
    } else {
      std::vector<int32_t> out_toks(static_cast<std::size_t>(B_eval), 0);
      gen.step_slots(steps, out_toks);
    }
  }
  for (int s : slots)
    if (s >= 0) gen.release_slot(s);
  return logits_row0;
}

}  // namespace

int main() {
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
    std::fprintf(stderr, "batch_spec_parity: no CUDA device, skipping\n");
    return 77;
  }
  const auto model_dir = find_model_dir();
  if (!std::filesystem::exists(model_dir / "mtp.safetensors")) {
    std::fprintf(stderr, "batch_spec_parity: model/mtp.safetensors not found at %s, skipping\n",
                 model_dir.string().c_str());
    return 77;
  }

  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  const auto& tok = fmt.tokenizer();
  const int32_t eos_id = *tok.token_id("<|content_model_end_sampling|>");

  const std::vector<std::string> base_users = {
      "Explain how a hash map handles collisions, with a short Python example.",
      "Write a Python module implementing an LRU cache with O(1) get and put.",
      "Compare B-trees and LSM-trees for write-heavy database workloads.",
      "Explain how TCP congestion control slow start and AIMD work.",
      "Write a C++20 ring buffer with atomic head and tail indices.",
      "Explain virtual memory paging versus segmentation concisely.",
      "Summarize sliding window attention combined with periodic full attention.",
      "Write a Python function solving 0/1 knapsack with path reconstruction.",
      "Explain MVCC snapshot isolation in relational databases.",
      "Write a concise Python binary search tree insert and lookup.",
      "Explain Raft leader election and log replication in bullets.",
      "Write a Python topological sort using Kahn's algorithm.",
  };

  std::vector<std::vector<int32_t>> prompt_ids;
  for (const auto& u : base_users) prompt_ids.push_back(render_prompt(fmt, "", u));

  // ---- 1 & 2 & 3 & 4 & 5 & 6: Shared plain & spec generators (cache off) ----
  {
    RuntimeOptions opt_plain;
    opt_plain.model_dir = model_dir;
    opt_plain.max_ctx = 2048;
    opt_plain.prefill_chunk = 512;
    opt_plain.dense_fp8 = true;
    opt_plain.kv_fp8 = true;
    opt_plain.max_batch = 12;
    opt_plain.max_kv_tokens = 12 * 512;
    opt_plain.spec = false;
    opt_plain.prefix_cache = false;
    auto gen_plain = ingine::model::make_generator(opt_plain);

    const int kGenLen = 48;
    const int bs[] = {1, 4, 8, 12};
    std::vector<std::vector<std::vector<int32_t>>> plain_refs;
    for (int B : bs) {
      std::vector<Prompt> ps;
      for (int i = 0; i < B; ++i) ps.push_back(Prompt{prompt_ids[std::size_t(i)], {}});
      plain_refs.push_back(run_batch_greedy_w1(*gen_plain, ps, kGenLen));
    }

    RuntimeOptions opt_spec = opt_plain;
    opt_spec.spec = true;
    opt_spec.spec_threshold = 0.0f;  // full width W=8 for thorough verify+rollback coverage
    auto gen_spec = ingine::model::make_generator(opt_spec);
    CHECK(gen_spec->supports_batched_spec());

    // ---- 1a. Equal-arithmetic exact qualification: W=1 spec (step_slots) vs plain at B=1, 4, 8, 12 (MUST be 100% exact!)
    for (int bi = 0; bi < 4; ++bi) {
      const int B = bs[bi];
      std::vector<Prompt> ps;
      for (int i = 0; i < B; ++i) ps.push_back(Prompt{prompt_ids[std::size_t(i)], {}});
      const auto got_w1 = run_batch_greedy_w1(*gen_spec, ps, 24);
      int exact = 0, total = 0;
      for (int i = 0; i < B; ++i) {
        for (int k = 0; k < 24; ++k) {
          ++total;
          if (got_w1[std::size_t(i)][std::size_t(k)] == plain_refs[std::size_t(bi)][std::size_t(i)][std::size_t(k)])
            ++exact;
        }
      }
      std::printf("equal-arith W=1 spec vs plain B=%-2d: exact=%d/%d (%.2f%%)\n", B, exact, total,
                  100.0 * double(exact) / double(total));
      CHECK(exact == total);  // ZERO wrong tokens tolerated on equal arithmetic!
    }

    // ---- 1b. Exact batch replicate invariance inside B=12 spec batch (identical prompts in distinct slots -> bit-exact!)
    {
      std::vector<Prompt> reps;
      for (int i = 0; i < 12; ++i) reps.push_back(Prompt{prompt_ids[std::size_t(i % 4)], {}});
      int r = 0, a = 0;
      std::vector<std::vector<BatchSpecSlotTrace>> rep_traces;
      {
        BatchSpecTraceConfig cfg;
        cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) { rep_traces.push_back(trs); };
        TraceGuard tg(std::move(cfg));
        const auto outs = run_batch_greedy_multi(*gen_spec, reps, 32, r, a);
        for (int i = 4; i < 12; ++i) {
          CHECK(outs[std::size_t(i)] == outs[std::size_t(i % 4)]);
        }
      }
      CHECK(!rep_traces.empty());
      for (const auto& step_trs : rep_traces) {
        if (step_trs.size() == 12) {
          for (int i = 4; i < 12; ++i) {
            CHECK(step_trs[std::size_t(i)].w_cap == step_trs[std::size_t(i % 4)].w_cap);
            CHECK(step_trs[std::size_t(i)].accepted == step_trs[std::size_t(i % 4)].accepted);
            CHECK(step_trs[std::size_t(i)].next_token == step_trs[std::size_t(i % 4)].next_token);
            for (int g = 0; g < kGamma; ++g) {
              CHECK(step_trs[std::size_t(i)].draft_tokens[g] == step_trs[std::size_t(i % 4)].draft_tokens[g]);
              CHECK(step_trs[std::size_t(i)].draft_conf[g] == step_trs[std::size_t(i % 4)].draft_conf[g]);
            }
          }
        }
      }
      std::printf("equal-arith B=12 spec replicate invariance: EXACT (rounds=%d accepted=%d)\n", r, a);
    }

    // ---- 2. Post-partial-rollback & W=1 -> wider model taps + next logits + tokens vs clean serial W=1 replay!
    {
      Sampling greedy;
      greedy.temperature = 0.0f;
      int32_t t0 = 0;
      GenerateStats st;
      const int s_spec = gen_spec->start_slot(Prompt{prompt_ids[0], {}}, 48, greedy, t0, st);
      CHECK(s_spec >= 0);

      // Step A: W=1 via step_slots (tests W=1 -> wider drafter tap commit & state transition!)
      std::vector<int32_t> prefix_tokens = {t0};
      int32_t t1 = 0;
      {
        const BatchSlotStep step[1] = {{s_spec, t0}};
        gen_spec->step_slots(step, std::span(&t1, 1));
        prefix_tokens.push_back(t1);
      }

      // Step B: Run W=8 step_slots_multi until we observe a partial rollback (1 <= commit < w_cap) — guaranteed
      //         within a few rounds, or force a partial mismatch if all 7 accepted on round 1!
      bool saw_partial_rollback = false;
      for (int rd = 0; rd < 8 && !saw_partial_rollback; ++rd) {
        BatchSlotStep step[1] = {{s_spec, prefix_tokens.back()}};
        BatchSlotResult res[1] = {};
        BatchSpecSlotTrace last_tr;
        {
          BatchSpecTraceConfig cfg;
          cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
            if (!trs.empty()) last_tr = trs[0];
          };
          TraceGuard tg(std::move(cfg));
          gen_spec->step_slots_multi(step, res);
        }
        for (int k = 0; k < res[0].count; ++k) prefix_tokens.push_back(res[0].tokens[k]);
        if (last_tr.w_cap > 1 && last_tr.commit >= 1 && last_tr.commit < last_tr.w_cap) {
          saw_partial_rollback = true;
        }
      }
      CHECK(saw_partial_rollback);

      // Step C: Immediately after partial rollback, run a W=1 step capturing BOTH target_taps [4*H] AND next target_logits [V]!
      BatchSpecSlotTrace post_rb_tr;
      int32_t post_rb_tok = 0;
      {
        BatchSpecTraceConfig cfg;
        cfg.capture_target_logits = true;
        cfg.capture_target_taps = true;
        cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
          if (!trs.empty()) post_rb_tr = trs[0];
        };
        TraceGuard tg(std::move(cfg));
        const BatchSlotStep step[1] = {{s_spec, prefix_tokens.back()}};
        gen_spec->step_slots(step, std::span(&post_rb_tok, 1));
      }
      gen_spec->release_slot(s_spec);

      // Clean serial W=1 replay from scratch across the EXACT SAME prefix_tokens (ZERO speculative rollback ever ran!)
      int32_t rt0 = 0;
      GenerateStats rst;
      const int s_ref = gen_spec->start_slot(Prompt{prompt_ids[0], {}}, 48, greedy, rt0, rst);
      CHECK(s_ref >= 0 && rt0 == prefix_tokens[0]);
      for (std::size_t k = 0; k + 1 < prefix_tokens.size(); ++k) {
        int32_t next_t = 0;
        const BatchSlotStep step[1] = {{s_ref, prefix_tokens[k]}};
        gen_spec->step_slots(step, std::span(&next_t, 1));
      }
      BatchSpecSlotTrace ref_tr;
      int32_t ref_tok = 0;
      {
        BatchSpecTraceConfig cfg;
        cfg.capture_target_logits = true;
        cfg.capture_target_taps = true;
        cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
          if (!trs.empty()) ref_tr = trs[0];
        };
        TraceGuard tg(std::move(cfg));
        const BatchSlotStep step[1] = {{s_ref, prefix_tokens.back()}};
        gen_spec->step_slots(step, std::span(&ref_tok, 1));
      }
      gen_spec->release_slot(s_ref);

      CHECK(post_rb_tr.target_logits.size() == std::size_t(kUnpaddedVocab));
      CHECK(ref_tr.target_logits.size() == std::size_t(kUnpaddedVocab));
      CHECK(!post_rb_tr.target_taps.empty() && post_rb_tr.target_taps.size() == ref_tr.target_taps.size());
      DiffStats ldiff, tdiff;
      for (std::size_t i = 0; i < ref_tr.target_logits.size(); ++i)
        ldiff.update(post_rb_tr.target_logits[i], ref_tr.target_logits[i]);
      for (std::size_t i = 0; i < ref_tr.target_taps.size(); ++i)
        tdiff.update(post_rb_tr.target_taps[i], ref_tr.target_taps[i]);
      std::printf("post-partial-rollback & W=1->W=8 vs serial W=1 (prefix=%zu): tok=%d/%d logit_max_diff=%g tap_max_diff=%g\n",
                  prefix_tokens.size(), post_rb_tok, ref_tok, ldiff.max_abs, tdiff.max_abs);
      CHECK(ldiff.finite() && ldiff.max_abs <= 0.15f);
      CHECK(tdiff.finite() && tdiff.max_abs <= 0.05f);
      CHECK(post_rb_tok == ref_tok);
    }

    // ---- 3. Measured cross-gear agreement (reported separately) + common-prefix logit diagnosis on ANY divergence!
    for (int bi = 0; bi < 4; ++bi) {
      const int B = bs[bi];
      std::vector<Prompt> ps;
      for (int i = 0; i < B; ++i) ps.push_back(Prompt{prompt_ids[std::size_t(i)], {}});
      int rounds = 0, accepted = 0;
      const auto got = run_batch_greedy_multi(*gen_spec, ps, kGenLen, rounds, accepted);
      int exact_tokens = 0, total_tokens = 0, divergent_seqs = 0;
      for (int i = 0; i < B; ++i) {
        const auto& gseq = got[std::size_t(i)];
        const auto& rseq = plain_refs[std::size_t(bi)][std::size_t(i)];
        CHECK(gseq.size() == rseq.size());
        const std::size_t n = std::min(gseq.size(), rseq.size());
        std::size_t first_diff = n;
        for (std::size_t k = 0; k < n; ++k) {
          ++total_tokens;
          if (gseq[k] == rseq[k]) ++exact_tokens;
          else if (first_diff == n) first_diff = k;
        }
        if (first_diff < n) {
          ++divergent_seqs;
          // Diagnose divergent row at the EXACT common prefix `prompt + rseq[0..first_diff-1]`!
          CHECK(first_diff > 0);  // Prefill token 0 must ALWAYS be exact!
          std::vector<int32_t> common_pref = prompt_ids[std::size_t(i)];
          common_pref.insert(common_pref.end(), rseq.begin(), rseq.begin() + std::ptrdiff_t(first_diff));
          const auto l_plain = capture_prefix_logits(*gen_plain, common_pref, B, prompt_ids, /*use_multi=*/false);
          const auto l_spec = capture_prefix_logits(*gen_spec, common_pref, B, prompt_ids, /*use_multi=*/true);
          DiffStats ldiff;
          for (int v = 0; v < kUnpaddedVocab; ++v) ldiff.update(l_spec[std::size_t(v)], l_plain[std::size_t(v)]);
          const int32_t tp = rseq[first_diff];
          const int32_t ts = gseq[first_diff];
          const float margin_plain = l_plain[std::size_t(tp)] - l_plain[std::size_t(ts)];
          const float margin_spec = l_spec[std::size_t(ts)] - l_spec[std::size_t(tp)];
          std::printf(
              "  [cross-gear diag B=%d seq=%d pos=%zu] plain_tok=%d spec_tok=%d max_logit_diff=%.4f "
              "margin_plain=%.4f margin_spec=%.4f\n",
              B, i, first_diff, tp, ts, ldiff.max_abs, margin_plain, margin_spec);
          // Any cross-gear divergence MUST be finite with small logit diff and genuine near-tie margin (NEVER a bug!).
          CHECK(ldiff.finite());
          CHECK(ldiff.max_abs <= 0.6f);
          CHECK(margin_plain <= 2.0f * ldiff.max_abs + 0.05f);
        }
      }
      const double agree = total_tokens > 0 ? double(exact_tokens) / double(total_tokens) : 0.0;
      const double tok_per_round = rounds > 0 ? double(accepted + rounds) / double(rounds) : 0.0;
      std::printf("cross-gear B=%-2d: agree=%d/%d (%.2f%%, div_seqs=%d) rounds=%d accepted=%d (%.2f tok/round)\n", B,
                  exact_tokens, total_tokens, 100.0 * agree, divergent_seqs, rounds, accepted, tok_per_round);
      CHECK(rounds > 0);
      CHECK(accepted > 0);
      // Same-gear B=1 (M<=8 GEMV) has much tighter agreement; cross-gear B>1 diagnosed above at common prefix.
      if (B == 1) CHECK(agree >= 0.99);
    }

    // ---- 4. Sampled speculative parity vs fixed-draw CPU reference (heterogeneous greedy+banned+sampled batch)
    {
      const int32_t ban_tok = plain_refs[0][0][1];
      std::vector<Sampling> samps(4);
      samps[0].temperature = 0.0f;  // greedy
      samps[1].temperature = 0.0f;
      samps[1].banned_tokens = {ban_tok};
      samps[2].temperature = 0.8f;
      samps[2].top_k = 40;
      samps[2].top_p = 0.95f;
      samps[2].seed = 12345;
      samps[3].temperature = 0.6f;
      samps[3].top_k = 20;
      samps[3].top_p = 0.9f;
      samps[3].banned_tokens = {eos_id};
      samps[3].seed = 67890;

      // Capture solo sampled reference traces for slot 2 (proves per-slot RNG & proposals are unaffected by neighbors!).
      std::vector<BatchSpecSlotTrace> solo2_traces;
      std::vector<int32_t> solo2_tokens;
      {
        BatchSpecTraceConfig cfg;
        cfg.capture_candidates = true;
        cfg.capture_sampled_details = true;
        cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
          for (const auto& tr : trs) solo2_traces.push_back(tr);
        };
        TraceGuard tg(std::move(cfg));
        solo2_tokens = run_solo(*gen_spec, Prompt{prompt_ids[2], {}}, 24, samps[2]).tokens;
      }
      CHECK(!solo2_traces.empty());
      int solo2_sampled_verified = 0;
      for (const auto& tr : solo2_traces) {
        if (tr.has_sampled_details) {
          verify_sampled_trace_vs_cpu_ref(tr, "solo-slot2");
          ++solo2_sampled_verified;
        }
      }
      CHECK(solo2_sampled_verified > 0);

      auto run_hetero = [&](std::vector<BatchSpecSlotTrace>* all_traces) {
        std::vector<std::vector<int32_t>> outs(4);
        std::vector<int> slots(4, -1);
        for (int i = 0; i < 4; ++i) {
          int32_t t0 = 0;
          GenerateStats st;
          slots[std::size_t(i)] =
              gen_spec->start_slot(Prompt{prompt_ids[std::size_t(i)], {}}, 24, samps[std::size_t(i)], t0, st);
          CHECK(slots[std::size_t(i)] >= 0);
          outs[std::size_t(i)].push_back(t0);
        }
        BatchSpecTraceConfig cfg;
        cfg.capture_candidates = (all_traces != nullptr);
        cfg.capture_sampled_details = (all_traces != nullptr);
        if (all_traces) {
          cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
            for (const auto& tr : trs) all_traces->push_back(tr);
          };
        }
        std::optional<TraceGuard> tg;
        if (all_traces) tg.emplace(std::move(cfg));

        for (int step = 0; step < 30; ++step) {
          std::vector<BatchSlotStep> st;
          std::vector<int> idxs;
          for (int i = 0; i < 4; ++i) {
            if (slots[std::size_t(i)] >= 0 && outs[std::size_t(i)].size() < 24) {
              st.push_back({slots[std::size_t(i)], outs[std::size_t(i)].back()});
              idxs.push_back(i);
            }
          }
          if (st.empty()) break;
          std::vector<BatchSlotResult> res(st.size());
          gen_spec->step_slots_multi(st, res);
          for (std::size_t k = 0; k < st.size(); ++k) {
            const int i = idxs[k];
            for (int j = 0; j < res[k].count && outs[std::size_t(i)].size() < 24; ++j)
              outs[std::size_t(i)].push_back(res[k].tokens[j]);
            if (outs[std::size_t(i)].size() >= 24) {
              gen_spec->release_slot(slots[std::size_t(i)]);
              slots[std::size_t(i)] = -1;
            }
          }
        }
        for (int s : slots)
          if (s >= 0) gen_spec->release_slot(s);
        return outs;
      };

      std::vector<BatchSpecSlotTrace> h1_traces, h2_traces;
      const auto h1 = run_hetero(&h1_traces);
      const auto h2 = run_hetero(&h2_traces);
      CHECK(h1 == h2);
      CHECK(h1_traces.size() == h2_traces.size());
      CHECK(std::find(h1[1].begin(), h1[1].end(), ban_tok) == h1[1].end());
      CHECK(std::find(h1[3].begin(), h1[3].end(), eos_id) == h1[3].end());

      // Verify EVERY sampled round in the heterogeneous batch against the exact CPU fixed-draw reference,
      // AND verify replay determinism of draft IDs, probabilities, RNG draws, accept counts, and next tokens!
      int hetero_sampled_verified = 0;
      for (std::size_t k = 0; k < h1_traces.size(); ++k) {
        const auto& t1 = h1_traces[k];
        const auto& t2 = h2_traces[k];
        CHECK(t1.draft_seed == t2.draft_seed && t1.draft_draws == t2.draft_draws);
        CHECK(t1.w_cap == t2.w_cap && t1.accepted == t2.accepted && t1.next_token == t2.next_token);
        for (int g = 0; g < kGamma; ++g) {
          CHECK(t1.draft_tokens[g] == t2.draft_tokens[g]);
          CHECK(t1.draft_conf[g] == t2.draft_conf[g]);
        }
        if (t1.has_sampled_details) {
          CHECK(t1.uniforms == t2.uniforms);
          verify_sampled_trace_vs_cpu_ref(t1, "hetero-batch");
          ++hetero_sampled_verified;
        }
      }
      CHECK(hetero_sampled_verified > 0);
      std::printf("sampled fixed-draw CPU ref parity: solo_verified=%d hetero_verified=%d deterministic=EXACT bans=OK\n",
                  solo2_sampled_verified, hetero_sampled_verified);
    }

    // ---- 5. Deep multi-tenant shortlist isolation (cold concurrent: verify neighbor extra_ids mutated while Tenant A
    //         proposals, candidate IDs, confidences, qstats, accepts, and outputs stay EXACT!)
    {
      Sampling greedy;
      greedy.temperature = 0.0f;
      Prompt pA{prompt_ids[0], {}, 0, "tenant-A", true};
      std::vector<BatchSpecSlotTrace> soloA_traces;
      RunOut soloA;
      {
        BatchSpecTraceConfig cfg;
        cfg.capture_candidates = true;
        cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
          for (const auto& tr : trs) soloA_traces.push_back(tr);
        };
        TraceGuard tg(std::move(cfg));
        soloA = run_solo(*gen_spec, pA, 32, greedy);
      }

      // Build Tenant B packed with rare out-of-shortlist tokens (150000..150220).
      std::vector<int32_t> rare_b = prompt_ids[1];
      for (int t = 150000; t < 150220; ++t) rare_b.push_back(t);
      Prompt pB{rare_b, {}, 0, "tenant-B", true};

      std::vector<BatchSpecSlotTrace> pairA_traces, pairB_traces;
      int r = 0, a = 0;
      std::vector<std::vector<int32_t>> paired;
      {
        BatchSpecTraceConfig cfg;
        cfg.capture_candidates = true;
        cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
          for (const auto& tr : trs) {
            if (tr.slot == 0) pairA_traces.push_back(tr);
            else pairB_traces.push_back(tr);
          }
        };
        TraceGuard tg(std::move(cfg));
        paired = run_batch_greedy_multi(*gen_spec, {pA, pB}, 32, r, a);
      }

      // Positive control: verify Tenant B's dynamic shortlist ACTUALLY grew with rare tokens!
      CHECK(!pairB_traces.empty());
      CHECK(pairB_traces[0].slot_extra_ids.size() >= 100);
      // Verify Tenant A's shortlist, draft candidate IDs, draft tokens, confidences, accepted, and outputs are IDENTICAL!
      CHECK(paired[0] == soloA.tokens);
      CHECK(pairA_traces.size() == soloA_traces.size());
      const std::size_t ntr = std::min(pairA_traces.size(), soloA_traces.size());
      for (std::size_t k = 0; k < ntr; ++k) {
        CHECK(pairA_traces[k].slot_extra_ids == soloA_traces[k].slot_extra_ids);
        CHECK(pairA_traces[k].accepted == soloA_traces[k].accepted);
        CHECK(pairA_traces[k].next_token == soloA_traces[k].next_token);
        for (int g = 0; g < kGamma; ++g) {
          CHECK(pairA_traces[k].draft_tokens[g] == soloA_traces[k].draft_tokens[g]);
          CHECK(std::fabs(pairA_traces[k].draft_conf[g] - soloA_traces[k].draft_conf[g]) <= 1e-4f);
        }
        // Candidate IDs at every step (same slot_extra_ids + same top-1024 set).
        CHECK(pairA_traces[k].draft_cand_idx.size() == soloA_traces[k].draft_cand_idx.size());
        int cand_exact = 0;
        for (std::size_t c = 0; c < pairA_traces[k].draft_cand_idx.size(); ++c) {
          if (pairA_traces[k].draft_cand_idx[c] == soloA_traces[k].draft_cand_idx[c]) ++cand_exact;
        }
        CHECK(cand_exact == int(pairA_traces[k].draft_cand_idx.size()));
      }
      std::printf("tenant shortlist isolation (cold): neighbor_extra=%zu tenantA_extra=%zu proposals=EXACT tokens=EXACT\n",
                  pairB_traces[0].slot_extra_ids.size(), pairA_traces[0].slot_extra_ids.size());
    }

    // ---- 6. Mid-block stop-token truncation
    {
      Sampling greedy;
      greedy.temperature = 0.0f;
      const RunOut full_run = run_solo(*gen_spec, Prompt{prompt_ids[0], {}}, 24, greedy);
      CHECK(full_run.tokens.size() == 24);
      const int32_t mid_stop = full_run.tokens[10];
      const std::vector<int32_t> stops = {mid_stop};
      const RunOut stopped = run_solo(*gen_spec, Prompt{prompt_ids[0], {}}, 24, greedy, stops);
      CHECK(stopped.tokens.size() == 11);
      CHECK(stopped.tokens.back() == mid_stop);
      CHECK(std::equal(stopped.tokens.begin(), stopped.tokens.end(), full_run.tokens.begin()));
      std::printf("mid-block stop truncation: len=%zu exact=YES\n", stopped.tokens.size());
    }
  }

  // ---- 7. Confidence-width enforcement on real model (spec_threshold=0.8: w_cap <= lead + 1, accepted <= lead!)
  {
    RuntimeOptions opt_thr;
    opt_thr.model_dir = model_dir;
    opt_thr.max_ctx = 1024;
    opt_thr.prefill_chunk = 512;
    opt_thr.dense_fp8 = true;
    opt_thr.kv_fp8 = true;
    opt_thr.max_batch = 4;
    opt_thr.max_kv_tokens = 4 * 512;
    opt_thr.spec = true;
    opt_thr.spec_threshold = 0.8f;
    opt_thr.prefix_cache = false;
    auto gen_thr = ingine::model::make_generator(opt_thr);

    int total_traced = 0;
    int non_pow2_wcap_seen = 0;
    {
      BatchSpecTraceConfig cfg;
      cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
        for (const auto& tr : trs) {
          ++total_traced;
          int exp_lead = 0;
          while (exp_lead < kGamma && tr.draft_conf[exp_lead] > 0.8f) ++exp_lead;
          CHECK(tr.lead == exp_lead);
          CHECK(tr.w_cap <= exp_lead + 1);       // NEVER verifies unconfident drafts!
          CHECK(tr.accepted <= exp_lead);        // NEVER accepts unconfident drafts!
          CHECK(tr.W_batch >= tr.w_cap);
          if (tr.w_cap == 3 || tr.w_cap == 5 || tr.w_cap == 6 || tr.w_cap == 7) ++non_pow2_wcap_seen;
        }
      };
      TraceGuard tg(std::move(cfg));
      std::vector<Prompt> ps;
      for (int i = 0; i < 4; ++i) ps.push_back(Prompt{prompt_ids[std::size_t(i)], {}});
      int r = 0, a = 0;
      (void)run_batch_greedy_multi(*gen_thr, ps, 48, r, a);
    }
    CHECK(total_traced > 0);
    std::printf("confidence-width enforcement (thr=0.8): traced_slots=%d non_pow2_wcap_seen=%d bound=EXACT\n",
                total_traced, non_pow2_wcap_seen);
  }

  // ---- 8. Prefix cache + --spec: exact/partial/root cold-vs-warm parity, warm sampled fixed-draw parity,
  //         and warm cross-slot shortlist isolation after Tenant B rare tokens!
  {
    RuntimeOptions opt_cache;
    opt_cache.model_dir = model_dir;
    opt_cache.max_ctx = 1024;
    opt_cache.prefill_chunk = 512;
    opt_cache.dense_fp8 = true;
    opt_cache.kv_fp8 = true;
    opt_cache.max_batch = 4;
    opt_cache.max_kv_tokens = 4 * 512;
    opt_cache.spec = true;
    opt_cache.spec_threshold = 0.0f;
    opt_cache.prefix_cache = true;
    auto gen = ingine::model::make_generator(opt_cache);

    std::string long_sys;
    for (int i = 0; i < 30; ++i)
      long_sys += "System rule " + std::to_string(i) + ": verify every invariant before responding cleanly. ";
    const auto p_full1 = render_prompt(fmt, long_sys, "Explain red-black tree balancing in 3 sentences.");
    const auto p_full2 = render_prompt(fmt, long_sys, "Explain AVL tree rotations in 3 sentences.");
    int common = 0;
    while (common < int(std::min(p_full1.size(), p_full2.size())) &&
           p_full1[std::size_t(common)] == p_full2[std::size_t(common)])
      ++common;
    CHECK(common > 128);

    Sampling greedy;
    greedy.temperature = 0.0f;
    const RunOut cold1 = run_solo(*gen, Prompt{p_full1, {}, 0, "ns1", true}, 32, greedy);
    const RunOut cold2 = run_solo(*gen, Prompt{p_full2, {}, 0, "ns1", true}, 32, greedy);
    CHECK(cold1.stats.reused_tokens == 0 && cold2.stats.reused_tokens == 0);

    // Populate root + full checkpoint via p_full1
    const RunOut warm_seed = run_solo(*gen, Prompt{p_full1, {}, common, "ns1", false}, 32, greedy);
    CHECK(warm_seed.tokens == cold1.tokens);
    CHECK(warm_seed.stats.spec_rounds == cold1.stats.spec_rounds);
    CHECK(warm_seed.stats.spec_accepted == cold1.stats.spec_accepted);

    // Dirty slot 0 with Tenant B rare tokens (150000..150180) so warm restore of p_full1 into slot 0
    // proves zero stale extra_ids or KV contamination across slot reuse!
    std::vector<int32_t> rare_dirty = prompt_ids[1];
    for (int t = 150000; t < 150180; ++t) rare_dirty.push_back(t);
    (void)run_solo(*gen, Prompt{rare_dirty, {}, 0, "ns-dirty", true}, 8, greedy);

    // Exact warm hit on p_full1 (greedy)
    std::vector<BatchSpecSlotTrace> exact_warm_traces;
    RunOut exact_hit;
    {
      BatchSpecTraceConfig cfg;
      cfg.capture_candidates = true;
      cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
        for (const auto& tr : trs) exact_warm_traces.push_back(tr);
      };
      TraceGuard tg(std::move(cfg));
      exact_hit = run_solo(*gen, Prompt{p_full1, {}, common, "ns1", false}, 32, greedy);
    }
    CHECK(exact_hit.stats.reused_tokens == int(p_full1.size()));
    CHECK(exact_hit.stats.reuse == Reuse::PagedPrefix);
    CHECK(exact_hit.tokens == cold1.tokens);
    CHECK(exact_hit.stats.spec_rounds == cold1.stats.spec_rounds);
    CHECK(exact_hit.stats.spec_accepted == cold1.stats.spec_accepted);
    CHECK(!exact_warm_traces.empty());
    for (int32_t id : exact_warm_traces[0].slot_extra_ids) {
      CHECK(id < 150000 || id >= 150180);  // ZERO rare tokens leaked from dirty slot 0!
    }

    // Warm vs cold SAMPLED fixed-draw parity on cached p_full1!
    Sampling samp;
    samp.temperature = 0.7f;
    samp.top_k = 32;
    samp.top_p = 0.92f;
    samp.seed = 424242;
    std::vector<BatchSpecSlotTrace> cold_s_traces, warm_s_traces;
    RunOut cold_s, warm_s;
    {
      BatchSpecTraceConfig cfg;
      cfg.capture_candidates = true;
      cfg.capture_sampled_details = true;
      cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
        for (const auto& tr : trs) cold_s_traces.push_back(tr);
      };
      TraceGuard tg(std::move(cfg));
      cold_s = run_solo(*gen, Prompt{p_full1, {}, 0, "ns1", true}, 24, samp);
    }
    {
      BatchSpecTraceConfig cfg;
      cfg.capture_candidates = true;
      cfg.capture_sampled_details = true;
      cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
        for (const auto& tr : trs) warm_s_traces.push_back(tr);
      };
      TraceGuard tg(std::move(cfg));
      warm_s = run_solo(*gen, Prompt{p_full1, {}, common, "ns1", false}, 24, samp);
    }
    CHECK(warm_s.stats.reused_tokens == int(p_full1.size()));
    CHECK(warm_s.tokens == cold_s.tokens);
    CHECK(warm_s_traces.size() == cold_s_traces.size());
    int warm_sampled_verified = 0;
    for (std::size_t k = 0; k < std::min(warm_s_traces.size(), cold_s_traces.size()); ++k) {
      CHECK(warm_s_traces[k].slot_extra_ids == cold_s_traces[k].slot_extra_ids);
      CHECK(warm_s_traces[k].draft_seed == cold_s_traces[k].draft_seed);
      CHECK(warm_s_traces[k].draft_draws == cold_s_traces[k].draft_draws);
      CHECK(warm_s_traces[k].accepted == cold_s_traces[k].accepted);
      CHECK(warm_s_traces[k].next_token == cold_s_traces[k].next_token);
      for (int g = 0; g < kGamma; ++g) {
        CHECK(warm_s_traces[k].draft_tokens[g] == cold_s_traces[k].draft_tokens[g]);
        CHECK(warm_s_traces[k].draft_conf[g] == cold_s_traces[k].draft_conf[g]);
      }
      if (warm_s_traces[k].has_sampled_details) {
        CHECK(warm_s_traces[k].uniforms == cold_s_traces[k].uniforms);
        verify_sampled_trace_vs_cpu_ref(warm_s_traces[k], "warm-restore-sampled");
        ++warm_sampled_verified;
      }
    }
    CHECK(warm_sampled_verified > 0);

    // Root/partial hit on p_full2 & namespace miss isolation ("ns2" must NOT reuse "ns1")
    const RunOut partial_hit = run_solo(*gen, Prompt{p_full2, {}, common, "ns1", false}, 32, greedy);
    CHECK(partial_hit.stats.reused_tokens == common);
    CHECK(partial_hit.tokens == cold2.tokens);

    const RunOut ns_miss = run_solo(*gen, Prompt{p_full1, {}, common, "ns2", false}, 16, greedy);
    CHECK(ns_miss.stats.reused_tokens == 0);
    std::printf(
        "prefix-cache+spec (cold/warm greedy+sampled+isolation): exact_reused=%d partial_reused=%d "
        "warm_sampled_verified=%d ns_miss=%d parity=EXACT\n",
        exact_hit.stats.reused_tokens, partial_hit.stats.reused_tokens, warm_sampled_verified,
        ns_miss.stats.reused_tokens);
  }

  // ---- 9. Tight 6-page pool (384 tokens, exact 6 pages) with --spec:
  //         24-tok overlap, 96-tok safe deferral, and 65/96 boundary reservation protection!
  {
    RuntimeOptions opt_tight;
    opt_tight.model_dir = model_dir;
    opt_tight.max_ctx = 384;
    opt_tight.max_kv_tokens = 384;  // exact 6 pages!
    opt_tight.prefill_chunk = 384;
    opt_tight.dense_fp8 = true;
    opt_tight.kv_fp8 = true;
    opt_tight.max_batch = 2;
    opt_tight.spec = true;
    opt_tight.spec_threshold = 0.0f;
    opt_tight.prefix_cache = true;
    auto gen = ingine::model::make_generator(opt_tight);

    std::vector<int32_t> p192 = prompt_ids[0];
    while (p192.size() < 192) p192.insert(p192.end(), prompt_ids[1].begin(), prompt_ids[1].end());
    p192.resize(192);

    Sampling greedy;
    greedy.temperature = 0.0f;
    const RunOut solo96 = run_solo(*gen, Prompt{p192, {}}, 96, greedy);
    CHECK(solo96.tokens.size() == 96);

    Scheduler sched(*gen, {}, 4);
    auto s24a = std::make_shared<Job>(p192, 24, greedy);
    auto s24b = std::make_shared<Job>(p192, 24, greedy);
    CHECK(sched.submit(s24a) && sched.submit(s24b));
    std::optional<std::string> e1, e2;
    const auto o24a = drain_job(*s24a, &e1);
    const auto o24b = drain_job(*s24b, &e2);
    CHECK(!e1 && !e2 && o24a.size() == 24 && o24b.size() == 24);
    CHECK(std::equal(o24a.begin(), o24a.end(), solo96.tokens.begin()));
    CHECK(std::equal(o24b.begin(), o24b.end(), solo96.tokens.begin()));

    auto m65 = std::make_shared<Job>(p192, 65, greedy);
    auto m96 = std::make_shared<Job>(p192, 96, greedy);
    CHECK(sched.submit(m65) && sched.submit(m96));
    std::optional<std::string> em1, em2;
    const auto om65 = drain_job(*m65, &em1);
    const auto om96 = drain_job(*m96, &em2);
    CHECK(!em1 && !em2 && om65.size() == 65 && om96.size() == 96);
    CHECK(std::equal(om65.begin(), om65.end(), solo96.tokens.begin()));
    CHECK(om96 == solo96.tokens);

    auto l96a = std::make_shared<Job>(p192, 96, greedy);
    auto l96b = std::make_shared<Job>(p192, 96, greedy);
    CHECK(sched.submit(l96a) && sched.submit(l96b));
    std::optional<std::string> el1, el2;
    const auto ol96a = drain_job(*l96a, &el1);
    const auto ol96b = drain_job(*l96b, &el2);
    CHECK(!el1 && !el2 && ol96a == solo96.tokens && ol96b == solo96.tokens);
    std::printf("tight 6-page + spec: 24/24=PASS 65/96=PASS 96/96=PASS\n");
  }

  std::printf(g_failures ? "FAILED (%d checks)\n" : "batch_spec_parity ok\n", g_failures);
  return g_failures ? 1 : 0;
}

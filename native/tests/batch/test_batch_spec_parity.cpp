// Full-model GPU parity, isolation, and reference-acceptance test suite for Batched MTP Speculative Decoding (#851):
// - Enforces single-resident-generator sequencing (AT MOST ONE BatchGenerator loaded at any time -> fits 28 GiB lease!)
//   and supports process-wide --vram-limit-mb / INGINE_VRAM_LIMIT_MB lease enforcement + peak VRAM tracking.
// 1. Equal-arithmetic exact qualification:
//    - W=1 spec (step_slots) vs plain (spec=false) at B=1, 4, 8, 12: 100% exact tokens, 0 wrong.
//    - Batch replicate invariance inside B=12 spec batch: 100% bit-identical tokens, w_cap, accepted, draft IDs, conf.
// 2. Post-partial-rollback and W=1 -> wider state/tap/next-logit parity vs clean serial W=1 replay at identical prefix.
// 3. Cross-gear agreement preserved separately + exact live-row common-prefix logit binding on any divergence:
//    - Captures the ACTUAL spec verify logit row during Section 3, replays the EXACT plain B-batch in Phase C,
//      and binds argmax(l_plain)==plain_tok, argmax(l_spec)==spec_tok, nonnegative margins, and tight near-tie bound.
// 4. Sampled speculative parity vs fixed-draw CPU reference (exact Philox4x32-10 draft column reconstruction across
//    all 7 rows + splitmix64 uniform draws + rejection accept + residual/fallback + Gumbel-max residual draw), plus
//    inline wrong-member Philox control (420 vs 421) and identical-row [0,0,3] cross-gear negative control.
// 5. Deep multi-tenant shortlist isolation (cold concurrent + warm cross-slot restore after Tenant B rare tokens).
// 6. Confidence-width bound enforcement (spec_threshold=0.8: w_cap <= lead + 1, accepted <= lead).
// 7. Mid-block stop truncation, prefix-cache exact/partial/root cold-vs-warm parity, and tight 6-page pool safety.
#include "core/act.h"
#include "engine/scheduler.h"
#include "ingine/generator.h"
#include "model/batch_spec_trace.h"
#include "model/batch_spec_verify.h"
#include "text/chat.h"

namespace ingine::model {
void set_batch_prefill_splits_for_test(std::vector<int> splits);
}  // namespace ingine::model

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
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
namespace sv = ingine::model::spec_verify;

constexpr int kUnpaddedVocab = 200058;

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

struct VramTracker {
  std::size_t total = 0;
  std::size_t free0 = 0;
  std::size_t min_free = 0;
  void init() {
    cudaFree(nullptr);
    cudaMemGetInfo(&free0, &total);
    min_free = free0;
  }
  void sample() {
    std::size_t f = 0, t = 0;
    if (cudaMemGetInfo(&f, &t) == cudaSuccess && f < min_free) min_free = f;
  }
  double peak_delta_mib() const { return double(free0 > min_free ? free0 - min_free : 0) / double(1 << 20); }
  double peak_device_used_gb() const { return double(total > min_free ? total - min_free : 0) / 1e9; }
};

struct TraceGuard {
  explicit TraceGuard(BatchSpecTraceConfig cfg) { ingine::model::set_batch_spec_trace_hook_for_test(std::move(cfg)); }
  ~TraceGuard() { ingine::model::clear_batch_spec_trace_hook_for_test(); }
};

void run_validator_negative_controls() {
  // 1. Philox wrong-member control (seed=12345, draws=7, uniform shortlist -> col 420; substituting 421 MUST fail!)
  constexpr int V = 1024;
  BatchSpecSlotTrace tr{};
  tr.has_draft = true;
  tr.has_sampled_details = true;
  tr.temperature = 0.8f;
  tr.draft_seed = 12345ull;
  tr.draft_draws = 7ull;
  tr.w_cap = 2;
  tr.W_batch = 2;
  tr.draft_cand_idx.resize(sv::kGamma * sv::kCandidates);
  tr.draft_cand_logits.assign(sv::kGamma * sv::kCandidates, 0.0f);
  for (int r = 0; r < sv::kGamma; ++r) {
    for (int k = 0; k < sv::kCandidates; ++k) tr.draft_cand_idx[std::size_t(r) * sv::kCandidates + k] = k;
    const int col = sv::reconstruct_draft_column(
        std::span<const float>(tr.draft_cand_logits.data() + std::size_t(r) * sv::kCandidates, sv::kCandidates),
        tr.temperature, tr.draft_seed, tr.draft_draws, r);
    tr.draft_tokens[r] = tr.draft_cand_idx[std::size_t(r) * sv::kCandidates + col];
    tr.draft_conf[r] = 1.0f / float(sv::kCandidates);
    tr.draft_qstats[r][0] = 0.0f;
    tr.draft_qstats[r][1] = float(sv::kCandidates);
  }
  CHECK(tr.draft_tokens[0] == 420);
  tr.accept_rng_seed = 99999ull;
  tr.accept_rng_counter_u = 3ull;
  tr.accept_rng_counter_resid = 4ull;
  tr.uniforms = {sv::head_uniform(tr.accept_rng_seed, tr.accept_rng_counter_u, 0u)};
  tr.target_probs.assign(std::size_t(tr.w_cap) * V, 0.f);
  tr.target_probs[420] = 1.0f;
  for (int v = 0; v < V; ++v) tr.target_probs[std::size_t(V + v)] = 1.0f / float(V);
  tr.accepted = 1;
  tr.residual_probs.assign(tr.target_probs.begin() + V, tr.target_probs.begin() + 2 * V);
  tr.next_token = sv::head_sample_from_probs(tr.residual_probs, tr.accept_rng_seed, tr.accept_rng_counter_resid);
  tr.commit = 2;
  CHECK(sv::validate_sampled_trace(tr, V).ok);
  BatchSpecSlotTrace bad = tr;
  bad.draft_tokens[0] = 421;
  CHECK(!sv::validate_sampled_trace(bad, V).ok);

  // All-zero target tensor + zero residual + token 0 -> MUST fail!
  {
    BatchSpecSlotTrace zall = tr;
    zall.target_probs.assign(std::size_t(zall.w_cap) * V, 0.f);
    zall.accepted = 0;
    zall.residual_probs.assign(V, 0.f);
    zall.next_token = 0;
    CHECK(!sv::validate_sampled_trace(zall, V).ok);
  }
  // Valid first target row + all-zero bonus row + zero residual + token 0 -> MUST fail!
  {
    BatchSpecSlotTrace zbonus = tr;
    for (int v = 0; v < V; ++v) zbonus.target_probs[std::size_t(V + v)] = 0.f;
    zbonus.accepted = 1;
    zbonus.residual_probs.assign(V, 0.f);
    zbonus.next_token = 0;
    CHECK(!sv::validate_sampled_trace(zbonus, V).ok);
  }
  // Valid zero-residual fallback to normalized target row -> MUST pass!
  {
    BatchSpecSlotTrace fb = tr;
    const float q_unif = 1.0f / float(sv::kCandidates);
    const float p_sub = q_unif * 0.9995f;
    std::uint64_t high_u_ctr = 0;
    while (sv::head_uniform(fb.accept_rng_seed, high_u_ctr, 0u) < 0.9996f) ++high_u_ctr;
    fb.accept_rng_counter_u = high_u_ctr;
    fb.uniforms = {sv::head_uniform(fb.accept_rng_seed, fb.accept_rng_counter_u, 0u)};
    for (int v = 0; v < V; ++v) fb.target_probs[std::size_t(v)] = p_sub;
    fb.accepted = 0;
    fb.residual_probs.assign(fb.target_probs.begin(), fb.target_probs.begin() + V);
    fb.next_token = sv::head_sample_from_probs(fb.residual_probs, fb.accept_rng_seed, fb.accept_rng_counter_resid);
    fb.commit = 1;
    CHECK(sv::validate_sampled_trace(fb, V).ok);
  }

  // 2. Cross-gear identical-row [0, 0, 3] control for claimed tokens 0 vs 1 MUST fail!
  const std::vector<float> id_row = {0.f, 0.f, 3.f};
  CHECK(!sv::validate_cross_gear_divergence(id_row, id_row, 0, 1, 3, 0.6f).ok);
  CHECK(!sv::validate_cross_gear_divergence({}, {}, 0, 1, 3, 0.6f).ok);
}

void verify_sampled_trace(const BatchSpecSlotTrace& tr, const char* label) {
  const auto res = sv::validate_sampled_trace(tr, kUnpaddedVocab);
  if (!res.ok) {
    std::fprintf(stderr, "sampled reference parity FAILED [%s]: %s\n", label, res.reason.c_str());
  }
  CHECK(res.ok);
}

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

struct DivergentCase {
  int B = 0;
  int seq_idx = 0;
  std::size_t pos = 0;  // index in generated output (pos >= 1)
  int32_t plain_tok = -1;
  int32_t spec_tok = -1;
  std::vector<float> l_spec;   // ACTUAL live verify logit row [kUnpaddedVocab] captured during spec run
  std::vector<float> l_plain;  // ACTUAL live plain logit row [kUnpaddedVocab] captured on exact plain batch replay
};

uint16_t float_to_fp16_bits_host(float f) {
  const __half h = __float2half_rn(f);
  uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

int fp16_ulp_distance(float a, float b, uint16_t& bits_a, uint16_t& bits_b, float& step_at_b) {
  bits_a = float_to_fp16_bits_host(a);
  bits_b = float_to_fp16_bits_host(b);
  auto to_ordered = [](uint16_t u) -> int {
    return (u & 0x8000u) ? -int(u & 0x7fffu) : int(u & 0x7fffu);
  };
  const uint16_t mag_b = uint16_t(bits_b & 0x7fffu);
  const uint16_t next_b = (mag_b < 0x7bffu) ? uint16_t((bits_b & 0x8000u) | (mag_b + 1u)) : bits_b;
  __half hb{}, hnb{};
  std::memcpy(&hb, &bits_b, sizeof(hb));
  std::memcpy(&hnb, &next_b, sizeof(hnb));
  step_at_b = std::fabs(__half2float(hnb) - __half2float(hb));
  return std::abs(to_ordered(bits_a) - to_ordered(bits_b));
}

// Independent target-only small (B=6/12, M<=32 SIMT FP8 GEMV) vs large (B=36, M=36>32 cuBLASLt GEMM) control
// WITHOUT speculative drafting, verify kernels, or rollback. Evaluates 6 frozen independent prompts over 47 steps.
int run_fp16_geometry_control(const std::filesystem::path& model_dir, const ingine::text::ChatFormat& fmt,
                              int vram_limit_mb, VramTracker& vram) {
  const std::vector<std::string> ctrl_users = {
      "Explain how a Bloom filter achieves space-efficient probabilistic set membership.",
      "Describe the difference between optimistic and pessimistic concurrency control.",
      "Write a short Python generator that yields Fibonacci numbers indefinitely.",
      "Explain how TLS 1.3 handshake reduces round trips compared to TLS 1.2.",
      "Summarize how garbage collection mark-and-sweep handles cyclic references.",
      "Explain how consistent hashing minimizes key remapping when nodes join.",
  };
  std::vector<std::vector<int32_t>> ctrl_prompts;
  for (const auto& u : ctrl_users) ctrl_prompts.push_back(render_prompt(fmt, "", u));

  constexpr int kS = 6;
  constexpr int kSteps = 48;  // pos 0 from prefill, pos 1..47 from teacher-forced step_slots (282 logit rows)

  RuntimeOptions opt;
  opt.model_dir = model_dir;
  opt.max_ctx = 512;
  opt.prefill_chunk = 512;
  opt.dense_fp8 = true;
  opt.kv_fp8 = true;
  opt.max_batch = 36;
  opt.max_kv_tokens = 36 * 128;
  opt.vram_limit_mb = vram_limit_mb;
  opt.spec = false;
  opt.prefix_cache = false;
  auto gen = ingine::model::make_generator(opt);
  vram.sample();

  // Helper: run B = kS * reps slots on the 6 frozen prompts. If teacher_force is empty, records greedy ref_toks;
  // otherwise feeds exact teacher_force[s][pos-1] so every geometry evaluates identical prefixes.
  auto run_plain_geom = [&](int reps, const std::vector<std::vector<int32_t>>& teacher_force,
                            std::vector<std::vector<int32_t>>& out_toks,
                            std::vector<std::vector<std::vector<float>>>& out_logits) {
    const int B = kS * reps;
    Sampling greedy;
    greedy.temperature = 0.0f;
    std::vector<int> slots(std::size_t(B), -1);
    std::vector<int32_t> last_toks(std::size_t(B), 0);
    out_toks.assign(kS, {});
    out_logits.assign(kS, std::vector<std::vector<float>>(kSteps));
    for (int i = 0; i < B; ++i) {
      const int s = i % kS;
      int32_t tok0 = 0;
      GenerateStats st;
      slots[std::size_t(i)] = gen->start_slot(Prompt{ctrl_prompts[std::size_t(s)], {}}, kSteps + 2, greedy, tok0, st);
      CHECK(slots[std::size_t(i)] >= 0);
      last_toks[std::size_t(i)] = teacher_force.empty() ? tok0 : teacher_force[std::size_t(s)][0];
      if (i < kS) out_toks[std::size_t(s)].push_back(tok0);
    }
    for (int pos = 1; pos < kSteps; ++pos) {
      std::vector<BatchSlotStep> steps(static_cast<std::size_t>(B));
      std::vector<int32_t> next(std::size_t(B), 0);
      for (int i = 0; i < B; ++i) {
        const int s = i % kS;
        const int32_t in_tok = teacher_force.empty() ? last_toks[std::size_t(i)] : teacher_force[std::size_t(s)][std::size_t(pos - 1)];
        steps[std::size_t(i)] = {slots[std::size_t(i)], in_tok};
      }
      std::vector<BatchSpecSlotTrace> trs;
      {
        BatchSpecTraceConfig cfg;
        cfg.capture_target_logits = true;
        cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& t) { trs = t; };
        TraceGuard tg(std::move(cfg));
        gen->step_slots(steps, next);
      }
      CHECK(trs.size() == std::size_t(B));
      for (int s = 0; s < kS; ++s) {
        out_logits[std::size_t(s)][std::size_t(pos)] = trs[std::size_t(s)].target_logits;
        out_toks[std::size_t(s)].push_back(next[std::size_t(s)]);
      }
      last_toks = next;
    }
    for (int sl : slots)
      if (sl >= 0) gen->release_slot(sl);
    vram.sample();
  };

  std::vector<std::vector<int32_t>> ref_toks, toks_b12, toks_b36;
  std::vector<std::vector<std::vector<float>>> logits_b6, logits_b12, logits_b36;
  run_plain_geom(1, {}, ref_toks, logits_b6);        // B=6  (M=6:  all GEMV + dense::message_gates)
  run_plain_geom(2, ref_toks, toks_b12, logits_b12); // B=12 (M=12: gates GEMM + proj/wo/head GEMV)
  run_plain_geom(6, ref_toks, toks_b36, logits_b36); // B=36 (M=36: ALL cuBLASLt GEMM + dequant_rows + softcap)

  auto summarize = [&](const char* label, const std::vector<std::vector<std::vector<float>>>& a_log,
                       const std::vector<std::vector<std::vector<float>>>& b_log) {
    std::vector<float> row_maxes;
    row_maxes.reserve(kS * (kSteps - 1));
    int gt_06 = 0, gt_15 = 0, flips = 0;
    float global_max = 0.f;
    int worst_s = -1, worst_pos = -1;
    for (int s = 0; s < kS; ++s) {
      float seq_max = 0.f;
      for (int pos = 1; pos < kSteps; ++pos) {
        const auto& la = a_log[std::size_t(s)][std::size_t(pos)];
        const auto& lb = b_log[std::size_t(s)][std::size_t(pos)];
        CHECK(la.size() == std::size_t(kUnpaddedVocab) && lb.size() == std::size_t(kUnpaddedVocab));
        float md = 0.f;
        for (int v = 0; v < kUnpaddedVocab; ++v) {
          const float d = std::fabs(lb[std::size_t(v)] - la[std::size_t(v)]);
          CHECK(std::isfinite(d));
          if (d > md) md = d;
        }
        row_maxes.push_back(md);
        if (md > seq_max) seq_max = md;
        if (md > global_max) {
          global_max = md;
          worst_s = s;
          worst_pos = pos;
        }
        if (md > 0.6f) ++gt_06;
        if (md > 1.5f) ++gt_15;
        const int32_t ama = sv::head_argmax(la);
        const int32_t amb = sv::head_argmax(lb);
        if (ama != amb) {
          ++flips;
          const float ma = la[std::size_t(ama)] - la[std::size_t(amb)];
          const float mb = lb[std::size_t(amb)] - lb[std::size_t(ama)];
          std::printf("  [%s argmax flip seq=%d pos=%d] small=%d large=%d max_diff=%.6f margin_small=%.6f margin_large=%.6f\n",
                      label, s, pos, ama, amb, md, ma, mb);
        }
      }
      std::printf("  [%s seq=%d] max_logit_diff=%.6f\n", label, s, seq_max);
    }
    std::sort(row_maxes.begin(), row_maxes.end());
    auto pct = [&](double q) {
      const std::size_t idx = std::min(row_maxes.size() - 1, std::size_t(q * double(row_maxes.size() - 1)));
      return row_maxes[idx];
    };
    std::printf(
        "TARGET-ONLY GEOMETRY CONTROL [%s] (act=%s, rows=%zu, spec=false, rollback=0): "
        "p50=%.6f p95=%.6f p99=%.6f max=%.6f (seq=%d pos=%d) | >0.6=%d/%zu >1.5=%d/%zu flips=%d\n",
        label, INGINE_ACT_FP16 ? "FP16" : "BF16", row_maxes.size(), pct(0.50), pct(0.95), pct(0.99), global_max,
        worst_s, worst_pos, gt_06, row_maxes.size(), gt_15, row_maxes.size(), flips);
  };

  summarize("B6_vs_B12", logits_b6, logits_b12);
  summarize("B6_vs_B36", logits_b6, logits_b36);
  summarize("B12_vs_B36", logits_b12, logits_b36);

  std::printf("geometry-control VRAM summary: peak_delta=%.1f MiB (%.2f GB), peak_device_used=%.2f GB, limit=%d MiB [%s]\n",
              vram.peak_delta_mib(), vram.peak_delta_mib() / 1024.0, vram.peak_device_used_gb(),
              vram_limit_mb, (vram_limit_mb <= 0 || vram.peak_delta_mib() <= double(vram_limit_mb)) ? "OK" : "EXCEEDED");
  if (vram_limit_mb > 0) CHECK(vram.peak_delta_mib() <= double(vram_limit_mb));
  std::printf(g_failures ? "FAILED (%d checks)\n" : "fp16_geometry_control ok\n", g_failures);
  return g_failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  run_validator_negative_controls();

  int vram_limit_mb = 0;
  if (const char* env_lim = std::getenv("INGINE_VRAM_LIMIT_MB"); env_lim && *env_lim) {
    vram_limit_mb = std::atoi(env_lim);
  }
  bool fp16_geometry_control = false;
  for (int a = 1; a < argc; ++a) {
    if (std::strcmp(argv[a], "--vram-limit-mb") == 0 && a + 1 < argc) {
      vram_limit_mb = std::atoi(argv[++a]);
    } else if (std::strcmp(argv[a], "--fp16-geometry-control") == 0) {
      fp16_geometry_control = true;
    }
  }

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

  VramTracker vram;
  vram.init();

  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  if (fp16_geometry_control) {
    return run_fp16_geometry_control(model_dir, fmt, vram_limit_mb, vram);
  }
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

  const int kGenLen = 48;
  const int bs[] = {1, 4, 8, 12};

  // ---- PHASE 0: Near-cap VRAM preflight rejection (MUST fail closed BEFORE load_weights crosses the cap!) ----
  {
    RuntimeOptions opt_tiny;
    opt_tiny.model_dir = model_dir;
    opt_tiny.max_ctx = 2048;
    opt_tiny.prefill_chunk = 512;
    opt_tiny.dense_fp8 = true;
    opt_tiny.kv_fp8 = true;
    opt_tiny.max_batch = 12;
    opt_tiny.max_kv_tokens = 12 * 512;
    opt_tiny.vram_limit_mb = 2048;  // far below weights peak; must reject before allocating weights!
    opt_tiny.spec = true;
    bool rejected = false;
    try {
      auto gen_fail = ingine::model::make_generator(opt_tiny);
      (void)gen_fail;
    } catch (const std::runtime_error& e) {
      rejected = (std::string(e.what()).find("VRAM lease cap exceeded before load_weights") != std::string::npos);
    }
    vram.sample();
    CHECK(rejected);
    CHECK(vram.peak_delta_mib() < 512.0);
    std::printf("preflight VRAM near-cap rejection (limit=2048 MiB): rejected=%d peak_delta=%.1f MiB\n",
                int(rejected), vram.peak_delta_mib());
  }

  // ---- PHASE A: Single resident plain generator (collects plain references, then FULLY RELEASES before spec!) ----
  std::vector<std::vector<std::vector<int32_t>>> plain_refs;
  {
    RuntimeOptions opt_plain;
    opt_plain.model_dir = model_dir;
    opt_plain.max_ctx = 2048;
    opt_plain.prefill_chunk = 512;
    opt_plain.dense_fp8 = true;
    opt_plain.kv_fp8 = true;
    opt_plain.max_batch = 12;
    opt_plain.max_kv_tokens = 12 * 512;
    opt_plain.vram_limit_mb = vram_limit_mb;
    opt_plain.spec = false;
    opt_plain.prefix_cache = false;
    auto gen_plain = ingine::model::make_generator(opt_plain);
    vram.sample();
    for (int B : bs) {
      std::vector<Prompt> ps;
      for (int i = 0; i < B; ++i) ps.push_back(Prompt{prompt_ids[std::size_t(i)], {}});
      plain_refs.push_back(run_batch_greedy_w1(*gen_plain, ps, kGenLen));
      vram.sample();
      std::printf("Phase A plain reference B=%-2d: %d seqs x %d tokens ok\n", B, B, kGenLen);
    }
  }  // gen_plain destroyed here -> 100% of plain weights & KV rings freed BEFORE gen_spec loads!

  // ---- PHASE B: Single resident spec generator (cache off, spec_threshold=0.0) ----
  std::vector<DivergentCase> divergent_cases;
  {
    RuntimeOptions opt_spec;
    opt_spec.model_dir = model_dir;
    opt_spec.max_ctx = 2048;
    opt_spec.prefill_chunk = 512;
    opt_spec.dense_fp8 = true;
    opt_spec.kv_fp8 = true;
    opt_spec.max_batch = 12;
    opt_spec.max_kv_tokens = 12 * 512;
    opt_spec.vram_limit_mb = vram_limit_mb;
    opt_spec.spec = true;
    opt_spec.spec_threshold = 0.0f;
    opt_spec.prefix_cache = false;
    auto gen_spec = ingine::model::make_generator(opt_spec);
    CHECK(gen_spec->supports_batched_spec());
    vram.sample();

    // ---- 1a. Equal-arithmetic exact qualification: W=1 spec (step_slots) vs plain at B=1, 4, 8, 12 (100% exact!)
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
      CHECK(exact == total);
    }

    // ---- 1b. Exact batch replicate invariance inside B=12 spec batch
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
        for (int i = 4; i < 12; ++i) CHECK(outs[std::size_t(i)] == outs[std::size_t(i % 4)]);
      }
      CHECK(!rep_traces.empty());
      for (const auto& step_trs : rep_traces) {
        if (step_trs.size() == 12) {
          for (int i = 4; i < 12; ++i) {
            CHECK(step_trs[std::size_t(i)].w_cap == step_trs[std::size_t(i % 4)].w_cap);
            CHECK(step_trs[std::size_t(i)].accepted == step_trs[std::size_t(i % 4)].accepted);
            CHECK(step_trs[std::size_t(i)].next_token == step_trs[std::size_t(i % 4)].next_token);
            for (int g = 0; g < sv::kGamma; ++g) {
              CHECK(step_trs[std::size_t(i)].draft_tokens[g] == step_trs[std::size_t(i % 4)].draft_tokens[g]);
              CHECK(step_trs[std::size_t(i)].draft_conf[g] == step_trs[std::size_t(i % 4)].draft_conf[g]);
            }
          }
        }
      }
      std::printf("equal-arith B=12 spec replicate invariance: EXACT (rounds=%d accepted=%d)\n", r, a);
    }

    // ---- 2. Post-partial-rollback & W=1 -> wider model taps + next logits + tokens vs clean serial W=1 replay
    {
      Sampling greedy;
      greedy.temperature = 0.0f;
      int32_t t0 = 0;
      GenerateStats st;
      const int s_spec = gen_spec->start_slot(Prompt{prompt_ids[0], {}}, 48, greedy, t0, st);
      CHECK(s_spec >= 0);

      std::vector<int32_t> prefix_tokens = {t0};
      int32_t t1 = 0;
      {
        const BatchSlotStep step[1] = {{s_spec, t0}};
        gen_spec->step_slots(step, std::span(&t1, 1));
        prefix_tokens.push_back(t1);
      }

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
        if (last_tr.w_cap > 1 && last_tr.commit >= 1 && last_tr.commit < last_tr.w_cap) saw_partial_rollback = true;
      }
      CHECK(saw_partial_rollback);

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
      std::size_t worst_tap_idx = 0;
      float worst_tap_got = 0.f, worst_tap_ref = 0.f, worst_tap_abs = -1.f;
      for (std::size_t i = 0; i < ref_tr.target_taps.size(); ++i) {
        const float gv = post_rb_tr.target_taps[i];
        const float rv = ref_tr.target_taps[i];
        tdiff.update(gv, rv);
        const float d = std::fabs(gv - rv);
        if (std::isfinite(d) && d > worst_tap_abs) {
          worst_tap_abs = d;
          worst_tap_idx = i;
          worst_tap_got = gv;
          worst_tap_ref = rv;
        }
      }
      uint16_t bits_got = 0, bits_ref = 0;
      float fp16_step_ref = 0.f;
      const int fp16_ulps = fp16_ulp_distance(worst_tap_got, worst_tap_ref, bits_got, bits_ref, fp16_step_ref);
      const bool tap_strict_ok = tdiff.finite() && tdiff.max_abs <= 0.05f;
      const bool tap_diag07_ok = tdiff.finite() && tdiff.max_abs <= 0.07f;
      std::printf(
          "post-partial-rollback & W=1->W=8 vs serial W=1 (prefix=%zu): tok=%d/%d logit_max_diff=%.6f "
          "tap_max_diff=%.6f (idx=%zu tap=%zu col=%zu got=%.6f ref=%.6f fp16_bits=0x%04x/0x%04x "
          "fp16_step_at_ref=%.6f fp16_ulps=%d | strict_0.05=%s diag_0.07=%s)\n",
          prefix_tokens.size(), post_rb_tok, ref_tok, ldiff.max_abs, tdiff.max_abs, worst_tap_idx,
          worst_tap_idx / 2304u, worst_tap_idx % 2304u, worst_tap_got, worst_tap_ref, unsigned(bits_got),
          unsigned(bits_ref), fp16_step_ref, fp16_ulps, tap_strict_ok ? "PASS" : "FAIL",
          tap_diag07_ok ? "PASS" : "FAIL");
      CHECK(ldiff.finite() && ldiff.max_abs <= 0.15f);
      CHECK(tap_strict_ok);
      CHECK(post_rb_tok == ref_tok);
    }

    // ---- 3. Measured cross-gear agreement (reported separately), capturing ACTUAL live verify logits at any
    //         first divergence for exact common-prefix binding in Phase C!
    for (int bi = 0; bi < 4; ++bi) {
      const int B = bs[bi];
      std::vector<Prompt> ps;
      for (int i = 0; i < B; ++i) ps.push_back(Prompt{prompt_ids[std::size_t(i)], {}});

      // Run batched spec while recording the actual emitted logit row for the first divergence of each sequence i.
      std::vector<std::vector<int32_t>> outs(static_cast<std::size_t>(B));
      struct Active {
        int idx = 0;
        int slot = -1;
        int32_t last_tok = 0;
      };
      std::vector<Active> live;
      std::vector<bool> recorded_div(static_cast<std::size_t>(B), false);
      std::vector<int> slot_to_seq(16, -1);
      Sampling greedy;
      greedy.temperature = 0.0f;
      for (int i = 0; i < B; ++i) {
        int32_t tok0 = 0;
        GenerateStats st;
        const int s = gen_spec->start_slot(ps[std::size_t(i)], kGenLen, greedy, tok0, st);
        CHECK(s >= 0);
        outs[std::size_t(i)].push_back(tok0);
        CHECK(tok0 == plain_refs[std::size_t(bi)][std::size_t(i)][0]);  // prefill token 0 MUST be exact!
        slot_to_seq[std::size_t(s)] = i;
        live.push_back(Active{i, s, tok0});
      }
      int rounds = 0, accepted = 0;
      while (!live.empty()) {
        std::vector<BatchSlotStep> steps(live.size());
        std::vector<BatchSlotResult> res(live.size());
        for (std::size_t i = 0; i < live.size(); ++i) steps[i] = {live[i].slot, live[i].last_tok};

        std::vector<BatchSpecSlotTrace> step_trs;
        {
          BatchSpecTraceConfig cfg;
          cfg.capture_target_logits = true;
          cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) { step_trs = trs; };
          TraceGuard tg(std::move(cfg));
          gen_spec->step_slots_multi(steps, res);
        }
        CHECK(step_trs.size() == live.size());

        std::vector<Active> kept;
        for (std::size_t i = 0; i < live.size(); ++i) {
          rounds += res[i].spec_rounds;
          accepted += res[i].spec_accepted;
          const int seq_i = live[i].idx;
          auto& out = outs[std::size_t(seq_i)];
          const auto& ref = plain_refs[std::size_t(bi)][std::size_t(seq_i)];
          for (int k = 0; k < res[i].count && int(out.size()) < kGenLen; ++k) {
            const std::size_t out_pos = out.size();
            const int32_t got_tok = res[i].tokens[k];
            if (!recorded_div[std::size_t(seq_i)] && out_pos < ref.size() && got_tok != ref[out_pos]) {
              recorded_div[std::size_t(seq_i)] = true;
              // Bind the ACTUAL verify row `k` logits that produced `got_tok`!
              DivergentCase dc;
              dc.B = B;
              dc.seq_idx = seq_i;
              dc.pos = out_pos;
              dc.plain_tok = ref[out_pos];
              dc.spec_tok = got_tok;
              if (step_trs[i].target_logits.size() >= std::size_t(k + 1) * kUnpaddedVocab) {
                const float* row_p = step_trs[i].target_logits.data() + std::size_t(k) * kUnpaddedVocab;
                dc.l_spec.assign(row_p, row_p + kUnpaddedVocab);
              }
              divergent_cases.push_back(std::move(dc));
            }
            out.push_back(got_tok);
          }
          live[i].last_tok = out.back();
          if (int(out.size()) >= kGenLen) gen_spec->release_slot(live[i].slot);
          else kept.push_back(live[i]);
        }
        live = std::move(kept);
      }

      int exact_tokens = 0, total_tokens = 0, div_seqs = 0;
      for (int i = 0; i < B; ++i) {
        if (recorded_div[std::size_t(i)]) ++div_seqs;
        const auto& gseq = outs[std::size_t(i)];
        const auto& rseq = plain_refs[std::size_t(bi)][std::size_t(i)];
        CHECK(gseq.size() == rseq.size());
        for (std::size_t k = 0; k < std::min(gseq.size(), rseq.size()); ++k) {
          ++total_tokens;
          if (gseq[k] == rseq[k]) ++exact_tokens;
        }
      }
      const double agree = total_tokens > 0 ? double(exact_tokens) / double(total_tokens) : 0.0;
      const double tok_per_round = rounds > 0 ? double(accepted + rounds) / double(rounds) : 0.0;
      std::printf("cross-gear B=%-2d: agree=%d/%d (%.2f%%, div_seqs=%d) rounds=%d accepted=%d (%.2f tok/round)\n", B,
                  exact_tokens, total_tokens, 100.0 * agree, div_seqs, rounds, accepted, tok_per_round);
      CHECK(rounds > 0);
      CHECK(accepted > 0);
      if (B == 1) CHECK(agree >= 0.99);
    }

    // ---- 4. Sampled speculative parity vs fixed-draw CPU reference (exact Philox4x32-10 + splitmix64!)
    {
      const int32_t ban_tok = plain_refs[0][0][1];
      std::vector<Sampling> samps(4);
      samps[0].temperature = 0.0f;
      samps[0].seed = 11111;
      samps[1].temperature = 0.0f;
      samps[1].banned_tokens = {ban_tok};
      samps[1].seed = 22222;
      samps[2].temperature = 0.8f;
      samps[2].top_k = 40;
      samps[2].top_p = 0.95f;
      samps[2].seed = 12345;
      samps[3].temperature = 0.6f;
      samps[3].top_k = 20;
      samps[3].top_p = 0.9f;
      samps[3].banned_tokens = {eos_id};
      samps[3].seed = 67890;

      std::vector<BatchSpecSlotTrace> solo2_traces;
      {
        BatchSpecTraceConfig cfg;
        cfg.capture_candidates = true;
        cfg.capture_sampled_details = true;
        cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
          for (const auto& tr : trs) solo2_traces.push_back(tr);
        };
        TraceGuard tg(std::move(cfg));
        (void)run_solo(*gen_spec, Prompt{prompt_ids[2], {}}, 24, samps[2]);
      }
      int solo2_sampled_verified = 0;
      for (const auto& tr : solo2_traces) {
        if (tr.has_sampled_details) {
          verify_sampled_trace(tr, "solo-slot2");
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

      int hetero_sampled_verified = 0;
      for (std::size_t k = 0; k < h1_traces.size(); ++k) {
        const auto& t1 = h1_traces[k];
        const auto& t2 = h2_traces[k];
        CHECK(t1.draft_seed == t2.draft_seed && t1.draft_draws == t2.draft_draws);
        CHECK(t1.w_cap == t2.w_cap && t1.accepted == t2.accepted && t1.next_token == t2.next_token);
        for (int g = 0; g < sv::kGamma; ++g) {
          CHECK(t1.draft_tokens[g] == t2.draft_tokens[g]);
          CHECK(t1.draft_conf[g] == t2.draft_conf[g]);
        }
        if (t1.has_sampled_details) {
          CHECK(t1.uniforms == t2.uniforms);
          verify_sampled_trace(t1, "hetero-batch");
          ++hetero_sampled_verified;
        }
      }
      CHECK(hetero_sampled_verified > 0);
      std::printf("sampled Philox+splitmix fixed-draw parity: solo_verified=%d hetero_verified=%d deterministic=EXACT\n",
                  solo2_sampled_verified, hetero_sampled_verified);
    }

    // ---- 5. Deep multi-tenant shortlist isolation (cold concurrent, strict lockstep B=2 until Tenant A hits 32)
    {
      Sampling greedy;
      greedy.temperature = 0.0f;
      greedy.seed = 55555;
      Prompt pA{prompt_ids[0], {}, 0, "tenant-A", true};
      Prompt pCleanB{prompt_ids[1], {}, 0, "tenant-B-clean", true};
      const RunOut soloA = run_solo(*gen_spec, pA, 32, greedy);

      // Keep neighbor slot B active (budget=96) for EVERY round until Tenant A emits 32 tokens, asserting B==2
      // on every step and releasing both slots cleanly at the end.
      auto run_lockstep_b2 = [&](const Prompt& neighbor, std::vector<BatchSpecSlotTrace>& outA_traces,
                                 std::vector<BatchSpecSlotTrace>* outB_traces, std::vector<int>& out_step_bs) {
        std::vector<int32_t> outA, outB;
        int32_t tA0 = 0, tB0 = 0;
        GenerateStats stA, stB;
        const int sA = gen_spec->start_slot(pA, 32, greedy, tA0, stA);
        const int sB = gen_spec->start_slot(neighbor, 96, greedy, tB0, stB);
        CHECK(sA >= 0 && sB >= 0 && sA != sB);
        outA.push_back(tA0);
        outB.push_back(tB0);
        {
          BatchSpecTraceConfig cfg;
          cfg.capture_candidates = true;
          cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
            const int step_b = int(trs.size());
            for (const auto& tr : trs) {
              if (tr.slot == sA) {
                outA_traces.push_back(tr);
                out_step_bs.push_back(step_b);
              } else if (tr.slot == sB && outB_traces) {
                outB_traces->push_back(tr);
              }
            }
          };
          TraceGuard tg(std::move(cfg));
          while (int(outA.size()) < 32) {
            CHECK(int(outB.size()) < 96);
            const BatchSlotStep steps[2] = {{sA, outA.back()}, {sB, outB.back()}};
            BatchSlotResult res[2] = {};
            gen_spec->step_slots_multi(steps, res);
            for (int k = 0; k < res[0].count && int(outA.size()) < 32; ++k) outA.push_back(res[0].tokens[k]);
            for (int k = 0; k < res[1].count && int(outB.size()) < 96; ++k) outB.push_back(res[1].tokens[k]);
          }
        }
        gen_spec->release_slot(sA);
        gen_spec->release_slot(sB);
        return outA;
      };

      std::vector<BatchSpecSlotTrace> cleanA_traces;
      std::vector<int> clean_step_bs;
      const auto clean_outA = run_lockstep_b2(pCleanB, cleanA_traces, nullptr, clean_step_bs);

      // Dirty Tenant B with 350 rare tail token IDs (guaranteed >= 100 outside immutable 32768 shortlist):
      std::vector<int32_t> rare_b = prompt_ids[1];
      for (int t = 185000; t < 185350; ++t) rare_b.push_back(t);
      Prompt pB{rare_b, {}, 0, "tenant-B", true};

      std::vector<BatchSpecSlotTrace> pairA_traces, pairB_traces;
      std::vector<int> pair_step_bs;
      const auto dirty_outA = run_lockstep_b2(pB, pairA_traces, &pairB_traces, pair_step_bs);

      CHECK(!pairB_traces.empty() && pairB_traces[0].slot_extra_ids.size() >= 100);
      const bool tokens_exact = (dirty_outA == soloA.tokens) && (dirty_outA == clean_outA);
      CHECK(dirty_outA == soloA.tokens);
      CHECK(dirty_outA == clean_outA);
      bool proposals_exact =
          !pairA_traces.empty() && (pairA_traces.size() == cleanA_traces.size()) &&
          (pair_step_bs.size() == pairA_traces.size()) && (clean_step_bs.size() == cleanA_traces.size());
      CHECK(proposals_exact);
      for (std::size_t k = 0; k < std::min(pairA_traces.size(), cleanA_traces.size()); ++k) {
        const bool b_ok = (pair_step_bs[k] == 2 && clean_step_bs[k] == 2);
        const bool w_ok =
            (pairA_traces[k].W_batch == cleanA_traces[k].W_batch) && (pairA_traces[k].w_cap == cleanA_traces[k].w_cap);
        const bool ex_ok = (pairA_traces[k].slot_extra_ids == cleanA_traces[k].slot_extra_ids);
        const bool acc_ok = (pairA_traces[k].accepted == cleanA_traces[k].accepted);
        const bool nxt_ok = (pairA_traces[k].next_token == cleanA_traces[k].next_token);
        const bool cand_ok = (pairA_traces[k].draft_cand_idx == cleanA_traces[k].draft_cand_idx);
        CHECK(b_ok);
        CHECK(w_ok);
        CHECK(ex_ok);
        CHECK(acc_ok);
        CHECK(nxt_ok);
        bool g_ok = true;
        for (int g = 0; g < sv::kGamma; ++g) {
          const bool dt_ok = (pairA_traces[k].draft_tokens[g] == cleanA_traces[k].draft_tokens[g]);
          const bool dc_ok = (pairA_traces[k].draft_conf[g] == cleanA_traces[k].draft_conf[g]);
          CHECK(dt_ok);
          CHECK(dc_ok);
          g_ok = g_ok && dt_ok && dc_ok;
        }
        CHECK(cand_ok);
        proposals_exact = proposals_exact && b_ok && w_ok && ex_ok && acc_ok && nxt_ok && cand_ok && g_ok;
      }
      std::printf(
          "tenant shortlist isolation (cold, lockstep B=2 rounds=%zu): neighbor_extra=%zu tenantA_extra=%zu "
          "proposals=%s tokens=%s\n",
          pairA_traces.size(), pairB_traces.empty() ? std::size_t(0) : pairB_traces[0].slot_extra_ids.size(),
          pairA_traces.empty() ? std::size_t(0) : pairA_traces[0].slot_extra_ids.size(),
          proposals_exact ? "EXACT" : "MISMATCH", tokens_exact ? "EXACT" : "MISMATCH");
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
    vram.sample();
  }  // gen_spec destroyed here -> 100% freed!

  // ---- PHASE C: Exact live plain batch replay for any cross-gear divergences recorded in Phase B ----
  if (!divergent_cases.empty()) {
    RuntimeOptions opt_plain;
    opt_plain.model_dir = model_dir;
    opt_plain.max_ctx = 2048;
    opt_plain.prefill_chunk = 512;
    opt_plain.dense_fp8 = true;
    opt_plain.kv_fp8 = true;
    opt_plain.max_batch = 12;
    opt_plain.max_kv_tokens = 12 * 512;
    opt_plain.vram_limit_mb = vram_limit_mb;
    opt_plain.spec = false;
    opt_plain.prefix_cache = false;
    auto gen_plain = ingine::model::make_generator(opt_plain);
    vram.sample();

    std::map<int, std::vector<DivergentCase*>> by_B;
    for (auto& dc : divergent_cases) by_B[dc.B].push_back(&dc);

    for (auto& [B, cases] : by_B) {
      std::size_t max_pos = 0;
      for (const auto* dc : cases) max_pos = std::max(max_pos, dc->pos);

      Sampling greedy;
      greedy.temperature = 0.0f;
      std::vector<int> slots(static_cast<std::size_t>(B), -1);
      std::vector<int32_t> last_toks(static_cast<std::size_t>(B), 0);
      for (int i = 0; i < B; ++i) {
        GenerateStats st;
        slots[std::size_t(i)] =
            gen_plain->start_slot(Prompt{prompt_ids[std::size_t(i)], {}}, int(max_pos) + 2, greedy, last_toks[std::size_t(i)], st);
        CHECK(slots[std::size_t(i)] >= 0);
      }
      for (std::size_t pos = 1; pos <= max_pos; ++pos) {
        std::vector<BatchSlotStep> steps(static_cast<std::size_t>(B));
        std::vector<int32_t> next(static_cast<std::size_t>(B), 0);
        for (int i = 0; i < B; ++i) steps[std::size_t(i)] = {slots[std::size_t(i)], last_toks[std::size_t(i)]};

        bool need_capture = false;
        for (const auto* dc : cases)
          if (dc->pos == pos) need_capture = true;

        std::vector<BatchSpecSlotTrace> trs;
        if (need_capture) {
          BatchSpecTraceConfig cfg;
          cfg.capture_target_logits = true;
          cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& t) { trs = t; };
          TraceGuard tg(std::move(cfg));
          gen_plain->step_slots(steps, next);
        } else {
          gen_plain->step_slots(steps, next);
        }
        if (need_capture) {
          CHECK(trs.size() == std::size_t(B));
          for (auto* dc : cases) {
            if (dc->pos == pos) dc->l_plain = trs[std::size_t(dc->seq_idx)].target_logits;
          }
        }
        last_toks = next;
      }
      for (int s : slots)
        if (s >= 0) gen_plain->release_slot(s);
    }

    int strict06_pass = 0, diag15_pass = 0;
    for (const auto& dc : divergent_cases) {
      const auto vr_strict = sv::validate_cross_gear_divergence(dc.l_plain, dc.l_spec, dc.plain_tok, dc.spec_tok,
                                                                kUnpaddedVocab, 0.6f);
      const auto vr_diag15 = sv::validate_cross_gear_divergence(dc.l_plain, dc.l_spec, dc.plain_tok, dc.spec_tok,
                                                                kUnpaddedVocab, 1.5f);
      if (vr_strict.ok) ++strict06_pass;
      if (vr_diag15.ok) ++diag15_pass;
      std::printf(
          "  [cross-gear bound diag B=%d seq=%d pos=%zu] plain=%d(am=%d) spec=%d(am=%d) max_diff=%.6f "
          "margin_p=%.6f margin_s=%.6f | strict_0.6=%s diag_1.5=%s\n",
          dc.B, dc.seq_idx, dc.pos, dc.plain_tok, vr_strict.argmax_plain, dc.spec_tok, vr_strict.argmax_spec,
          vr_strict.max_diff, vr_strict.margin_plain, vr_strict.margin_spec,
          vr_strict.ok ? "PASS" : vr_strict.reason.c_str(),
          vr_diag15.ok ? "PASS" : vr_diag15.reason.c_str());
      CHECK(vr_strict.ok);
    }
    std::printf("cross-gear bound summary (act=%s): total=%zu strict_0.6=%d/%zu diag_1.5=%d/%zu\n",
                INGINE_ACT_FP16 ? "FP16" : "BF16", divergent_cases.size(), strict06_pass, divergent_cases.size(),
                diag15_pass, divergent_cases.size());
    vram.sample();
  }

  // ---- PHASE D: Confidence-width enforcement on real model (spec_threshold=0.8) ----
  {
    RuntimeOptions opt_thr;
    opt_thr.model_dir = model_dir;
    opt_thr.max_ctx = 1024;
    opt_thr.prefill_chunk = 512;
    opt_thr.dense_fp8 = true;
    opt_thr.kv_fp8 = true;
    opt_thr.max_batch = 4;
    opt_thr.max_kv_tokens = 4 * 512;
    opt_thr.vram_limit_mb = vram_limit_mb;
    opt_thr.spec = true;
    opt_thr.spec_threshold = 0.8f;
    opt_thr.prefix_cache = false;
    auto gen_thr = ingine::model::make_generator(opt_thr);
    vram.sample();

    int total_traced = 0;
    int non_pow2_wcap_seen = 0;
    {
      BatchSpecTraceConfig cfg;
      cfg.on_step = [&](const std::vector<BatchSpecSlotTrace>& trs) {
        for (const auto& tr : trs) {
          ++total_traced;
          int exp_lead = 0;
          while (exp_lead < sv::kGamma && tr.draft_conf[exp_lead] > 0.8f) ++exp_lead;
          CHECK(tr.lead == exp_lead);
          CHECK(tr.w_cap <= exp_lead + 1);
          CHECK(tr.accepted <= exp_lead);
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
    vram.sample();
  }

  // ---- PHASE E: Prefix cache + --spec (exact/partial/root cold-vs-warm greedy+sampled Philox parity & isolation) ----
  {
    RuntimeOptions opt_cache;
    opt_cache.model_dir = model_dir;
    opt_cache.max_ctx = 1024;
    opt_cache.prefill_chunk = 512;
    opt_cache.dense_fp8 = true;
    opt_cache.kv_fp8 = true;
    opt_cache.max_batch = 4;
    opt_cache.max_kv_tokens = 4 * 512;
    opt_cache.vram_limit_mb = vram_limit_mb;
    opt_cache.spec = true;
    opt_cache.spec_threshold = 0.0f;
    opt_cache.prefix_cache = true;
    auto gen = ingine::model::make_generator(opt_cache);
    vram.sample();

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
    // Match the root-prefix chunk boundary (`common`) on cold controls so prefill cuBLASLt GEMM shapes are
    // identical between cold and root-cached/warm runs (same pattern as test_batch_prefix_parity.cpp).
    ingine::model::set_batch_prefill_splits_for_test({common});
    const RunOut cold1 = run_solo(*gen, Prompt{p_full1, {}, 0, "ns1", true}, 32, greedy);
    const RunOut cold2 = run_solo(*gen, Prompt{p_full2, {}, 0, "ns1", true}, 32, greedy);
    ingine::model::set_batch_prefill_splits_for_test({});
    CHECK(cold1.stats.reused_tokens == 0 && cold2.stats.reused_tokens == 0);

    const RunOut warm_seed = run_solo(*gen, Prompt{p_full1, {}, common, "ns1", false}, 32, greedy);
    CHECK(warm_seed.tokens == cold1.tokens);
    CHECK(warm_seed.stats.spec_rounds == cold1.stats.spec_rounds);
    CHECK(warm_seed.stats.spec_accepted == cold1.stats.spec_accepted);

    std::vector<int32_t> rare_dirty = prompt_ids[1];
    for (int t = 150000; t < 150180; ++t) rare_dirty.push_back(t);
    (void)run_solo(*gen, Prompt{rare_dirty, {}, 0, "ns-dirty", true}, 8, greedy);

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
    for (int32_t id : exact_warm_traces[0].slot_extra_ids) CHECK(id < 150000 || id >= 150180);

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
      ingine::model::set_batch_prefill_splits_for_test({common});
      cold_s = run_solo(*gen, Prompt{p_full1, {}, 0, "ns1", true}, 24, samp);
      ingine::model::set_batch_prefill_splits_for_test({});
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
      for (int g = 0; g < sv::kGamma; ++g) {
        CHECK(warm_s_traces[k].draft_tokens[g] == cold_s_traces[k].draft_tokens[g]);
        CHECK(warm_s_traces[k].draft_conf[g] == cold_s_traces[k].draft_conf[g]);
      }
      if (warm_s_traces[k].has_sampled_details) {
        CHECK(warm_s_traces[k].uniforms == cold_s_traces[k].uniforms);
        verify_sampled_trace(warm_s_traces[k], "warm-restore-sampled");
        ++warm_sampled_verified;
      }
    }
    CHECK(warm_sampled_verified > 0);

    const RunOut partial_hit = run_solo(*gen, Prompt{p_full2, {}, common, "ns1", false}, 32, greedy);
    CHECK(partial_hit.stats.reused_tokens == common);
    CHECK(partial_hit.tokens == cold2.tokens);

    const RunOut ns_miss = run_solo(*gen, Prompt{p_full1, {}, common, "ns2", false}, 16, greedy);
    CHECK(ns_miss.stats.reused_tokens == 0);
    std::printf(
        "prefix-cache+spec (cold/warm greedy+sampled Philox+isolation): exact_reused=%d partial_reused=%d "
        "warm_sampled_verified=%d ns_miss=%d parity=EXACT\n",
        exact_hit.stats.reused_tokens, partial_hit.stats.reused_tokens, warm_sampled_verified,
        ns_miss.stats.reused_tokens);
    vram.sample();
  }

  // ---- PHASE F: Tight 6-page pool (384 tokens, exact 6 pages) with --spec ----
  {
    RuntimeOptions opt_tight;
    opt_tight.model_dir = model_dir;
    opt_tight.max_ctx = 384;
    opt_tight.max_kv_tokens = 384;
    opt_tight.prefill_chunk = 384;
    opt_tight.dense_fp8 = true;
    opt_tight.kv_fp8 = true;
    opt_tight.max_batch = 2;
    opt_tight.vram_limit_mb = vram_limit_mb;
    opt_tight.spec = true;
    opt_tight.spec_threshold = 0.0f;
    opt_tight.prefix_cache = true;
    auto gen = ingine::model::make_generator(opt_tight);
    vram.sample();

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
    vram.sample();
  }

  vram.sample();
  const bool vram_ok = (vram_limit_mb <= 0) || (vram.peak_delta_mib() <= double(vram_limit_mb));
  std::printf("parity VRAM summary: peak_delta=%.1f MiB (%.2f GB), peak_device_used=%.2f GB, limit=%d MiB [%s]\n",
              vram.peak_delta_mib(), vram.peak_delta_mib() / 1024.0, vram.peak_device_used_gb(), vram_limit_mb,
              vram_ok ? "OK" : "EXCEEDED");
  CHECK(vram_ok);

  std::printf(g_failures ? "FAILED (%d checks)\n" : "batch_spec_parity ok\n", g_failures);
  return g_failures ? 1 : 0;
}

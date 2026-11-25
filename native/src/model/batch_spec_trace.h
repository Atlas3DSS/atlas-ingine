// Test-only observability hook for BatchGenerator speculative decoding (#851).
// Inactive (nullptr callback, zero allocations/copies/syncs) in production.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace ingine::model {

struct BatchSpecSlotTrace {
  int slot = -1;
  int pos = 0;
  std::int32_t anchor = 0;
  float temperature = 0.f;
  int top_k = 0;
  float top_p = 1.f;
  std::vector<std::int32_t> banned;

  // Per-slot drafter state & proposals (populated on step_slots_multi when drafter active)
  bool has_draft = false;
  std::uint64_t draft_seed = 0;
  std::uint64_t draft_draws = 0;
  std::vector<std::int32_t> slot_extra_ids;
  std::int32_t draft_tokens[7] = {};
  float draft_conf[7] = {};
  float draft_qstats[7][2] = {};
  std::vector<std::int32_t> draft_cand_idx;  // [7 * 1024] when capture_candidates
  std::vector<float> draft_cand_logits;      // [7 * 1024] when capture_candidates

  // Width selection & acceptance
  int lead = 0;
  int w_cap = 1;
  int W_batch = 1;
  int accepted = 0;
  std::int32_t next_token = 0;
  int commit = 1;

  // Sampled-accept fixed-draw reference data (when capture_sampled_details && temp > 0 && w_cap > 1)
  bool has_sampled_details = false;
  std::uint64_t accept_rng_seed = 0;
  std::uint64_t accept_rng_counter_u = 0;
  std::uint64_t accept_rng_counter_resid = 0;
  std::vector<float> uniforms;        // [w_cap - 1]
  std::vector<float> target_probs;    // [w_cap * V]
  std::vector<float> residual_probs;  // [V]

  // Optional full target logits [w_cap * V] and committed target taps [commit * 4 * H] (converted to fp32)
  std::vector<float> target_logits;
  std::vector<float> target_taps;
};

struct BatchSpecTraceConfig {
  bool capture_candidates = false;
  bool capture_sampled_details = false;
  bool capture_target_logits = false;
  bool capture_target_taps = false;
  std::function<void(const std::vector<BatchSpecSlotTrace>&)> on_step;
};

void set_batch_spec_trace_hook_for_test(BatchSpecTraceConfig cfg);
void clear_batch_spec_trace_hook_for_test();

}  // namespace ingine::model

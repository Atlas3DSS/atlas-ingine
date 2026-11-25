// Pure CPU planning helpers for batched speculative verify widths and staggered arrival benchmarks.
// CUDA-free so both BatchGenerator (production) and CPU unit tests (test_batch_scheduler) execute the exact same logic.
#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <span>
#include <vector>

namespace ingine::model {

struct VerifySlotInput {
  int lead_confident = 0;       // leading drafts strictly exceeding spec_threshold (0..7)
  int remaining_emissions = 1;  // remaining token budget for this slot
  int pos = 0;                  // current slot position before this verify step
  int max_ctx = 0;              // model context limit
  int max_pos_res = 0;          // reserved page-aligned lifetime position ceiling
  int current_pages = 0;        // pages currently attached to this slot
};

struct VerifyWidthPlan {
  std::vector<int> w_cap;      // exact per-slot active verify width in [1, max_verify_width] (1 anchor + w_cap-1 drafts)
  int W_batch = 1;             // shared canonical launch bucket in {1, 2, 4, 8}
  int total_extra_pages = 0;   // total additional KV pages required across all slots in this batch
};

// Returns the smallest canonical verify launch width in {1, 2, 4, 8} that is >= max_w_active
// and <= max_verify_width (used ONLY for the shared launch bucket shape, never to inflate an individual slot's drafts).
inline int canonical_verify_width(int max_w_active, int max_verify_width) {
  int W_batch = std::max(1, max_verify_width);
  for (int cw : {8, 4, 2, 1}) {
    if (cw <= max_verify_width && cw >= max_w_active) W_batch = cw;
  }
  return std::clamp(W_batch, 1, std::max(1, max_verify_width));
}

inline int pages_needed_for_tokens(int total_tokens, int page_size) {
  return (total_tokens + page_size - 1) / page_size;
}

// Plans per-slot active verify widths w_cap[b] and shared launch bucket W_batch.
// Invariants:
//   - Each active slot is clamped to lead_confident + 1 (at most lead_confident active drafts).
//   - Each slot respects remaining_emissions, max_ctx, and its reserved lifetime page ceiling.
//   - Aggregate extra KV page demand never exceeds spec_page_budget.
//   - Shared launch bucket W_batch in {1, 2, 4, 8} pads only the shared launch shape >= max_b(w_cap[b]).
inline VerifyWidthPlan plan_batch_verify_widths(std::span<const VerifySlotInput> slots, int max_verify_width,
                                                int page_size, int spec_page_budget) {
  VerifyWidthPlan plan;
  const int B = int(slots.size());
  const int max_w = std::max(1, max_verify_width);
  plan.w_cap.assign(static_cast<std::size_t>(B), 1);
  if (B == 0) return plan;

  for (int b = 0; b < B; ++b) {
    const VerifySlotInput& s = slots[std::size_t(b)];
    int w_lim = std::min(max_w, std::max(0, s.lead_confident) + 1);
    if (s.remaining_emissions > 0) {
      w_lim = std::min(w_lim, s.remaining_emissions);
    } else {
      w_lim = 1;
    }
    if (s.max_ctx > 0) {
      w_lim = std::min(w_lim, std::max(1, s.max_ctx - s.pos));
    }
    if (s.max_pos_res > 0) {
      w_lim = std::min(w_lim, std::max(1, s.max_pos_res - s.pos));
    }
    plan.w_cap[std::size_t(b)] = std::clamp(w_lim, 1, max_w);
  }

  auto extra_for = [&](int b) -> int {
    const VerifySlotInput& s = slots[std::size_t(b)];
    const int need = pages_needed_for_tokens(s.pos + plan.w_cap[std::size_t(b)], page_size);
    return need > s.current_pages ? (need - s.current_pages) : 0;
  };
  auto calc_total_extra = [&]() -> int {
    int sum = 0;
    for (int b = 0; b < B; ++b) sum += extra_for(b);
    return sum;
  };

  int total_extra = calc_total_extra();
  const int budget = std::max(0, spec_page_budget);
  while (total_extra > budget) {
    int best_b = -1;
    int best_w = 1;
    for (int b = 0; b < B; ++b) {
      if (extra_for(b) > 0 && plan.w_cap[std::size_t(b)] > best_w) {
        best_w = plan.w_cap[std::size_t(b)];
        best_b = b;
      }
    }
    if (best_b < 0) break;
    const VerifySlotInput& s = slots[std::size_t(best_b)];
    const int fit_in_cur_pages = s.current_pages * page_size - s.pos;
    plan.w_cap[std::size_t(best_b)] = std::max(1, std::min(plan.w_cap[std::size_t(best_b)] - 1, fit_in_cur_pages));
    total_extra = calc_total_extra();
  }

  int max_w_active = 1;
  for (int b = 0; b < B; ++b) max_w_active = std::max(max_w_active, plan.w_cap[std::size_t(b)]);
  plan.W_batch = canonical_verify_width(max_w_active, max_w);
  plan.total_extra_pages = total_extra;
  return plan;
}

// Deterministic output-milestone staggered arrival controller for continuous-batching benchmarks.
// Schedules cohort 2 arrival when cohort 1's cumulative emitted tokens reach `join_milestone_tokens`
// (or when cohort 1 drains early), guaranteeing plain (1 tok/round) and spec (k tok/round) replay
// the same mid-generation overlap point instead of drifting by round index.
struct StaggeredBenchmarkState {
  int total_requests = 0;
  int cohort1_size = 0;
  int join_milestone_tokens = 0;
  bool cohort2_offered = false;

  int offered = 0;
  int admitted = 0;
  int completed = 0;
  int dropped = 0;
  int deferred_attempts = 0;
  int eos_first_token = 0;
  int eos_stopped = 0;
  int max_token_stopped = 0;

  int cohort1_emitted = 0;
  int cohort1_emitted_at_join = 0;
  int cohort1_active_at_join = 0;
  int join_round = -1;

  std::deque<int> pending;

  void init(int C, int tokens_per_req, int join_tokens_per_req = 0) {
    *this = StaggeredBenchmarkState{};
    total_requests = std::max(0, C);
    cohort1_size = std::max(1, total_requests / 2);
    if (cohort1_size > total_requests) cohort1_size = total_requests;
    const int per_req = join_tokens_per_req > 0 ? join_tokens_per_req : std::max(1, tokens_per_req / 4);
    join_milestone_tokens = cohort1_size * per_req;
    for (int i = 0; i < cohort1_size; ++i) {
      pending.push_back(i);
      ++offered;
    }
    if (cohort1_size >= total_requests) cohort2_offered = true;
  }

  void maybe_offer_cohort2(int current_round, int cohort1_active_now) {
    if (cohort2_offered) return;
    if (cohort1_emitted >= join_milestone_tokens || (cohort1_active_now == 0 && pending.empty())) {
      cohort2_offered = true;
      cohort1_emitted_at_join = cohort1_emitted;
      cohort1_active_at_join = cohort1_active_now;
      join_round = current_round;
      for (int i = cohort1_size; i < total_requests; ++i) {
        pending.push_back(i);
        ++offered;
      }
    }
  }

  void note_emitted(int req_idx, int count) {
    if (req_idx < cohort1_size) cohort1_emitted += count;
  }

  bool accounting_consistent() const {
    return offered == total_requests && (admitted + dropped) == offered && completed == admitted && pending.empty();
  }
};

}  // namespace ingine::model

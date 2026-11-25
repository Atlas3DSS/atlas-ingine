// Pure CPU planning helpers for batched speculative verify widths, fail-closed VRAM lease accounting,
// and exact staggered arrival benchmarks.
// CUDA-free so both BatchGenerator (production) and CPU unit tests (test_batch_scheduler) execute the exact same logic.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ingine::model {

struct VerifySlotInput {
  int lead_confident = 0;       // leading drafts strictly exceeding spec_threshold (0..7)
  int remaining_emissions = 1;  // remaining token budget for this slot
  int pos = 0;                  // current slot position before this verify step
  int max_ctx = 0;              // model context limit
  int max_pos_res = 0;          // reserved page-aligned lifetime position ceiling
  int current_pages = 0;        // pages currently attached to this slot
  int step_max_tokens = 0;      // optional per-step cap (0 = unconstrained; >0 clamps w_cap <= step_max_tokens)
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

// Fail-closed VRAM lease check: std::nullopt means unbounded; a present value (including 0!) is a hard ceiling.
struct VramLeaseCheck {
  bool ok = true;
  std::size_t planned_bytes = 0;
  std::size_t remaining_bytes = 0;
};

inline VramLeaseCheck check_vram_lease_fit(std::optional<std::size_t> remaining_budget, std::size_t planned_bytes) {
  if (!remaining_budget.has_value()) return {true, planned_bytes, 0};
  return {planned_bytes <= *remaining_budget, planned_bytes, *remaining_budget};
}

// Conservative preflight peak VRAM bound for load_weights() BEFORE any GPU tensor is allocated.
// Accounts for all non-vision checkpoint tensors plus transient BF16->FP8 quantization coexistence (unembed + layer)
// and CUDA runtime/allocator overhead.
inline std::size_t estimate_weights_peak_bytes(std::size_t safetensors_non_vision_bytes, bool dense_fp8, int vocab,
                                               int hidden) {
  const std::size_t transient_fp8 =
      dense_fp8 ? (std::size_t(vocab) * std::size_t(hidden) * 2 + (std::size_t(128) << 20)) : 0;
  const std::size_t runtime_slack = std::size_t(256) << 20;
  return safetensors_non_vision_bytes + transient_fp8 + runtime_slack;
}

// Conservative preflight peak VRAM bound for Prefiller + auxiliary prefill buffers + cuBLASLt workspace.
inline std::size_t estimate_prefiller_peak_bytes(int prefill_chunk, int vocab, int max_batch) {
  const std::size_t C = std::size_t(std::max(1, prefill_chunk));
  const std::size_t arena_and_persist = C * std::size_t(280) * 1024 + (std::size_t(64) << 20);
  const std::size_t aux = C * sizeof(int) + std::size_t(vocab) * sizeof(float) +
                          std::size_t(std::max(1, max_batch)) * sizeof(int);
  const std::size_t cublas_slack = std::size_t(64) << 20;
  return arena_and_persist + aux + cublas_slack;
}

struct DrafterVramPeakEstimate {
  std::size_t load_peak_bytes = 0;   // peak while raw mtp.safetensors + cat + persistent MTP weights + fold_head coexist
  std::size_t steady_bytes = 0;      // steady-state resident MTP weights + shortlist + KV rings + scratch + accept bufs
  std::size_t peak_bytes = 0;        // max(load_peak_bytes, steady_bytes)
};

// Conservative preflight peak VRAM bound for BatchDrafter BEFORE allocating the raw mtp.safetensors buffer.
// Matches BatchDrafter::load() and BatchDrafter::BatchDrafter() exact buffer geometry.
inline DrafterVramPeakEstimate estimate_drafter_peak_bytes(std::size_t mtp_raw_bytes, int inter, int max_batch,
                                                           int draft_ctx, int vocab, int shortlist_rows,
                                                           int max_verify_width, bool fp8_head) {
  constexpr std::size_t kH = 2048;
  constexpr std::size_t kRank = 256;
  constexpr std::size_t kGamma = 7;
  constexpr std::size_t kBlock = 8;
  constexpr std::size_t kKvDim = 512;
  constexpr std::size_t kQDim = 2048;
  constexpr std::size_t kQkvRows = kQDim + 2 * kKvDim;
  constexpr std::size_t kLocalSlots = 2048;
  constexpr std::size_t kGlobalSlotBytes = 1024;  // FP8 KV_DIM + scale/metadata
  constexpr std::size_t kMaxSlotExtra = 1024;
  constexpr std::size_t kShortChunk = 2048;
  const std::size_t B = std::size_t(std::max(1, max_batch));
  const std::size_t I = std::size_t(std::max(1, inter));
  const std::size_t V = std::size_t(std::max(1, vocab));
  const std::size_t W = std::size_t(std::max(1, max_verify_width));
  const std::size_t ctx = std::size_t(std::max(16, draft_ctx));
  const std::size_t short_K = std::size_t(std::max(0, shortlist_rows)) + 256;

  // During BatchDrafter::load(), `raw` (mtp_raw_bytes) and `cat` (2 * I * H * 2) coexist with all persisted
  // MTP weights (<= mtp_raw_bytes + 16 MiB for wkv_all FP8) plus fold_w_ and fold_b_.
  const std::size_t cat_bytes = 2 * I * kH * 2;
  const std::size_t persistent_mtp_upper = mtp_raw_bytes + (std::size_t(16) << 20);
  const std::size_t fold_head_bytes = V * (kRank + 1) * 2;
  const std::size_t slack = std::size_t(64) << 20;
  const std::size_t load_peak = mtp_raw_bytes + cat_bytes + persistent_mtp_upper + fold_head_bytes + slack;

  // Steady-state after `raw` and `cat` are freed:
  const std::size_t shortlist_bytes = short_K * (fp8_head ? (kH + 4) : (2 * kH)) + short_K * 8;
  const std::size_t local_kv = B * 4 * kLocalSlots * kKvDim * 2 + B * 2 * kLocalSlots * 4;
  const std::size_t global_kv = B * 2 * ctx * kGlobalSlotBytes;
  const std::size_t max_rows = B * kBlock;
  const std::size_t max_gamma = B * kGamma;
  const std::size_t max_commit = B * W;
  const std::size_t act_scratch =
      max_rows * (2 * kH + 2 * kQDim + I + kQkvRows + kH + 2 * I) * 2 +
      max_gamma * (kH + kShortChunk) * 2 +
      max_gamma * (short_K + kMaxSlotExtra) * 4 +
      std::max({kQkvRows * kH, 2 * I * kH, kH * kQDim, kH * I, kShortChunk * kH}) * 2 +
      max_commit * (3 * kH + 2 * kKvDim) * 2 +
      (std::size_t(32) << 20);  // attn_ws + topk_ws + commit_piece + candidates
  const std::size_t accept_bufs = (kBlock + kGamma + 1) * V * 4 + (std::size_t(16) << 20);
  const std::size_t steady =
      persistent_mtp_upper + fold_head_bytes + shortlist_bytes + local_kv + global_kv + act_scratch + accept_bufs +
      slack;

  return {load_peak, steady, std::max(load_peak, steady)};
}

// Plans per-slot active verify widths w_cap[b] and shared launch bucket W_batch.
// Invariants:
//   - Each active slot is clamped to lead_confident + 1 (at most lead_confident active drafts).
//   - Each slot respects remaining_emissions, step_max_tokens, max_ctx, and its reserved lifetime page ceiling.
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
    if (s.step_max_tokens > 0) {
      w_lim = std::min(w_lim, s.step_max_tokens);
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

// Deterministic staggered arrival controller supporting TWO exact modes:
//   1. Exact emission-point mode (default): caps each cohort-1 request's per-step width so every cohort-1
//      request lands on EXACTLY `join_tokens_per_req` (zero overshoot between 1 tok/round plain and k tok/round spec!),
//      then offers cohort 2 at the exact common emission state.
//   2. External arrival-trace mode: replays an identical frozen arrival schedule `arrival_ms[0..C-1]` (in ms from t=0)
//      across both plain and spec arms without artificial step caps.
// Zero-result steps below token budget are classified separately in `aborted_zero_result` (never `completed`).
struct StaggeredBenchmarkState {
  int total_requests = 0;
  int cohort1_size = 0;
  int join_tokens_per_req = 0;
  int join_milestone_tokens = 0;
  bool use_external_trace = false;
  std::vector<double> arrival_ms;
  int next_trace_idx = 0;
  bool cohort2_offered = false;

  int offered = 0;
  int admitted = 0;
  int completed = 0;
  int aborted_zero_result = 0;
  int dropped = 0;
  int deferred_attempts = 0;
  int eos_first_token = 0;
  int eos_stopped = 0;
  int max_token_stopped = 0;

  int cohort1_emitted = 0;
  int cohort1_emitted_at_join = 0;
  int cohort1_active_at_join = 0;
  int cohort1_parked_steps = 0;
  int join_round = -1;
  double join_elapsed_ms = 0.0;

  std::deque<int> pending;

  void init(int C, int tokens_per_req, int join_per_req = 0) {
    *this = StaggeredBenchmarkState{};
    total_requests = std::max(0, C);
    cohort1_size = std::max(1, total_requests / 2);
    if (cohort1_size > total_requests) cohort1_size = total_requests;
    join_tokens_per_req = join_per_req > 0 ? join_per_req : std::max(1, tokens_per_req / 4);
    join_tokens_per_req = std::min(join_tokens_per_req, std::max(1, tokens_per_req));
    join_milestone_tokens = cohort1_size * join_tokens_per_req;
    for (int i = 0; i < cohort1_size; ++i) {
      pending.push_back(i);
      ++offered;
    }
    if (cohort1_size >= total_requests) cohort2_offered = true;
  }

  void init_external_trace(std::vector<double> trace_ms) {
    for (std::size_t i = 0; i < trace_ms.size(); ++i) {
      if (!std::isfinite(trace_ms[i]) || trace_ms[i] < 0.0) {
        throw std::invalid_argument("arrival trace timestamps must be finite and >= 0");
      }
      if (i > 0 && trace_ms[i] < trace_ms[i - 1]) {
        throw std::invalid_argument("arrival trace timestamps must be nondecreasing");
      }
    }
    *this = StaggeredBenchmarkState{};
    use_external_trace = true;
    arrival_ms = std::move(trace_ms);
    total_requests = int(arrival_ms.size());
    cohort1_size = 0;
    for (double t : arrival_ms)
      if (t <= 0.0) ++cohort1_size;
    if (cohort1_size == 0 && total_requests > 0) cohort1_size = 1;
    next_trace_idx = 0;
    offer_arrivals_at_ms(0.0, 0, 0);
  }

  // Returns true when cohort-1 lane `req_idx` has already reached `join_tokens_per_req` and must be parked
  // until the rest of cohort 1 catches up so heterogeneous acceptance widths NEVER overshoot the join milestone.
  // Never parks while cohort-1 requests remain deferred in `pending` (admitted < cohort1_size) to prevent deadlock.
  bool is_parked_at_milestone(int req_idx, int generated) const {
    if (use_external_trace || cohort2_offered) return false;
    if (!pending.empty() || admitted < cohort1_size) return false;
    return req_idx < cohort1_size && generated >= join_tokens_per_req;
  }

  bool should_step_lane(int req_idx, int generated) const { return !is_parked_at_milestone(req_idx, generated); }

  // Per-step emission cap for slot `req_idx` (currently at `generated` tokens).
  // In exact emission-point mode, caps cohort-1 slots before join so they NEVER overshoot `join_tokens_per_req`.
  // Parked lanes must be excluded from stepping via `should_step_lane()`. Returns 0 when unconstrained.
  int step_max_tokens_for(int req_idx, int generated) const {
    if (use_external_trace || cohort2_offered) return 0;
    if (req_idx >= cohort1_size) return 0;
    if (!pending.empty() || admitted < cohort1_size) return 0;
    if (generated >= join_tokens_per_req) return 0;
    return join_tokens_per_req - generated;
  }

  // Offers arrivals due at `elapsed_ms` (external trace mode).
  void offer_arrivals_at_ms(double elapsed_ms, int current_round, int cohort1_active_now) {
    if (!use_external_trace) return;
    while (next_trace_idx < total_requests && arrival_ms[std::size_t(next_trace_idx)] <= elapsed_ms) {
      if (!cohort2_offered && next_trace_idx >= cohort1_size) {
        cohort2_offered = true;
        cohort1_emitted_at_join = cohort1_emitted;
        cohort1_active_at_join = cohort1_active_now;
        join_round = current_round;
        join_elapsed_ms = elapsed_ms;
      }
      pending.push_back(next_trace_idx);
      ++offered;
      ++next_trace_idx;
    }
    if (next_trace_idx >= total_requests) cohort2_offered = true;
  }

  // In exact emission-point mode, offers cohort 2 as soon as every live cohort-1 slot has reached
  // `join_tokens_per_req` (so total cohort-1 emissions equal EXACTLY `join_milestone_tokens` when none hit early EOS),
  // or when cohort 1 has drained early.
  void maybe_offer_cohort2(int current_round, int cohort1_active_now, int cohort1_below_join_now,
                           double elapsed_ms = 0.0) {
    if (use_external_trace) {
      offer_arrivals_at_ms(elapsed_ms, current_round, cohort1_active_now);
      return;
    }
    if (cohort2_offered) return;
    if (pending.empty() && cohort1_below_join_now == 0) {
      cohort2_offered = true;
      cohort1_emitted_at_join = cohort1_emitted;
      cohort1_active_at_join = cohort1_active_now;
      join_round = current_round;
      join_elapsed_ms = elapsed_ms;
      for (int i = cohort1_size; i < total_requests; ++i) {
        pending.push_back(i);
        ++offered;
      }
    }
  }

  bool has_unoffered() const {
    return use_external_trace ? (next_trace_idx < total_requests) : !cohort2_offered;
  }

  double next_arrival_ms() const {
    if (use_external_trace && next_trace_idx < total_requests) return arrival_ms[std::size_t(next_trace_idx)];
    return 0.0;
  }

  void note_emitted(int req_idx, int count) {
    if (req_idx < cohort1_size) cohort1_emitted += count;
  }

  bool accounting_consistent() const {
    return offered == total_requests && (admitted + dropped) == offered &&
           (completed + aborted_zero_result) == admitted && pending.empty() && !has_unoffered();
  }
};

}  // namespace ingine::model

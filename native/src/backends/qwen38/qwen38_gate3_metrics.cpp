#include "backends/qwen38/qwen38_gate3_metrics.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace ingine::backends::qwen38 {

Qwen38CheckpointIndex::Qwen38CheckpointIndex(std::size_t max_checkpoints) : max_checkpoints_(max_checkpoints) {
  if (max_checkpoints_ == 0) {
    throw std::invalid_argument("Qwen38CheckpointIndex max_checkpoints must be positive");
  }
}

void Qwen38CheckpointIndex::reset(const ingine::model::BackendStateIdentity& default_id) {
  session_identity_ = default_id;
  committed_tokens_.clear();
  committed_token_mask_.clear();
  checkpoint_lengths_.clear();
  clear_tip_logits();
}

void Qwen38CheckpointIndex::append_committed(std::span<const std::int32_t> tokens, std::span<const std::uint8_t> mask) {
  if (tokens.empty()) {
    if (!mask.empty()) throw std::invalid_argument("mask must be empty when tokens are empty");
    return;
  }
  if (!mask.empty() && mask.size() != tokens.size()) {
    throw std::invalid_argument("append_committed mask length mismatch");
  }
  if (!committed_tokens_.empty() && (committed_token_mask_.empty() != mask.empty())) {
    throw std::logic_error("cannot mix masked and unmasked commits on live session without reset");
  }
  committed_tokens_.insert(committed_tokens_.end(), tokens.begin(), tokens.end());
  if (!mask.empty()) {
    committed_token_mask_.insert(committed_token_mask_.end(), mask.begin(), mask.end());
  }
}

void Qwen38CheckpointIndex::truncate_committed(std::size_t new_len) {
  if (new_len > committed_tokens_.size()) {
    throw std::out_of_range("truncate_committed exceeds committed_tokens size");
  }
  committed_tokens_.resize(new_len);
  if (!committed_token_mask_.empty()) {
    committed_token_mask_.resize(new_len);
  }
  clear_tip_logits();
}

std::optional<std::size_t> Qwen38CheckpointIndex::record_checkpoint(std::size_t token_count) {
  if (token_count == 0 || token_count > committed_tokens_.size()) {
    throw std::out_of_range("record_checkpoint token_count out of range");
  }
  if (std::find(checkpoint_lengths_.begin(), checkpoint_lengths_.end(), token_count) != checkpoint_lengths_.end()) {
    return std::nullopt;
  }
  std::optional<std::size_t> evicted;
  if (checkpoint_lengths_.size() >= max_checkpoints_) {
    // Evict oldest saved checkpoint strictly shorter than the live tip if available, else oldest entry.
    auto victim = checkpoint_lengths_.begin();
    const std::size_t tip = committed_tokens_.size();
    for (auto it = checkpoint_lengths_.begin(); it != checkpoint_lengths_.end(); ++it) {
      if (*it < tip) {
        victim = it;
        break;
      }
    }
    evicted = *victim;
    checkpoint_lengths_.erase(victim);
  }
  checkpoint_lengths_.push_back(token_count);
  std::sort(checkpoint_lengths_.begin(), checkpoint_lengths_.end());
  return evicted;
}

std::vector<std::size_t> Qwen38CheckpointIndex::prune_above(std::size_t max_len) {
  std::vector<std::size_t> removed;
  auto it = std::remove_if(checkpoint_lengths_.begin(), checkpoint_lengths_.end(), [&](std::size_t v) {
    if (v > max_len) {
      removed.push_back(v);
      return true;
    }
    return false;
  });
  checkpoint_lengths_.erase(it, checkpoint_lengths_.end());
  return removed;
}

std::vector<std::size_t> Qwen38CheckpointIndex::evict_all_checkpoints() {
  std::vector<std::size_t> evicted = std::move(checkpoint_lengths_);
  checkpoint_lengths_.clear();
  return evicted;
}

std::size_t Qwen38CheckpointIndex::restorable_prefix_len(std::size_t max_prefix_len) const noexcept {
  const std::size_t tip = committed_tokens_.size();
  if (tip == 0 || max_prefix_len == 0) return 0;
  if (max_prefix_len >= tip) return tip;
  std::size_t best = 0;
  for (const std::size_t ckpt : checkpoint_lengths_) {
    if (ckpt > 0 && ckpt <= max_prefix_len && ckpt < tip && ckpt > best) {
      best = ckpt;
    }
  }
  return best;
}

bool Qwen38CheckpointIndex::can_truncate_to(std::size_t new_len) const noexcept {
  const std::size_t tip = committed_tokens_.size();
  if (new_len == 0 || new_len == tip) return true;
  if (new_len > tip) return false;
  return std::find(checkpoint_lengths_.begin(), checkpoint_lengths_.end(), new_len) != checkpoint_lengths_.end();
}

Gate3StepComparison compare_logits_bit_exact(std::span<const float> cold_logits, std::span<const float> warm_logits,
                                             int position) {
  return compare_logits_bit_exact(cold_logits, warm_logits, position, position);
}

Gate3StepComparison compare_logits_bit_exact(std::span<const float> cold_logits, std::span<const float> warm_logits,
                                             int cold_position, int warm_position) {
  Gate3StepComparison cmp;
  cmp.cold_position = cold_position;
  cmp.warm_position = warm_position;
  cmp.position = cold_position;
  cmp.cold_vocab_size = cold_logits.size();
  cmp.warm_vocab_size = warm_logits.size();
  cmp.vocab_size = cold_logits.size();
  if (cold_logits.empty() || cold_logits.size() != warm_logits.size()) {
    cmp.all_finite = false;
    cmp.memcmp_equal = false;
    cmp.argmax_match = false;
    cmp.max_abs_diff = std::numeric_limits<float>::infinity();
    return cmp;
  }

  float best_cold = -std::numeric_limits<float>::infinity();
  float best_warm = -std::numeric_limits<float>::infinity();
  std::int32_t cold_id = 0;
  std::int32_t warm_id = 0;
  float max_abs = 0.0f;
  bool any_nonfinite = false;
  bool any_raw_diff = false;
  std::size_t first_diff = cold_logits.size();

  for (std::size_t i = 0; i < cold_logits.size(); ++i) {
    const float c = cold_logits[i];
    const float w = warm_logits[i];
    if (!std::isfinite(c) || !std::isfinite(w)) {
      any_nonfinite = true;
    } else {
      const float ad = std::fabs(c - w);
      if (ad > max_abs) max_abs = ad;
      if (c > best_cold) {
        best_cold = c;
        cold_id = static_cast<std::int32_t>(i);
      }
      if (w > best_warm) {
        best_warm = w;
        warm_id = static_cast<std::int32_t>(i);
      }
    }
    std::uint32_t cb = 0;
    std::uint32_t wb = 0;
    std::memcpy(&cb, &c, 4);
    std::memcpy(&wb, &w, 4);
    if (cb != wb && !any_raw_diff) {
      any_raw_diff = true;
      first_diff = i;
    }
  }

  if (any_nonfinite) {
    cmp.all_finite = false;
    cmp.memcmp_equal = false;
    cmp.first_diff_vocab_idx = first_diff;
    cmp.max_abs_diff = std::numeric_limits<float>::infinity();
    cmp.argmax_match = false;
    return cmp;
  }

  cmp.all_finite = true;
  cmp.cold_argmax = cold_id;
  cmp.warm_argmax = warm_id;
  cmp.argmax_match = (cold_id == warm_id);
  cmp.max_abs_diff = max_abs;
  cmp.first_diff_vocab_idx = first_diff;
  cmp.memcmp_equal = (std::memcmp(cold_logits.data(), warm_logits.data(), cold_logits.size() * sizeof(float)) == 0);
  if (cmp.memcmp_equal && any_raw_diff) {
    cmp.memcmp_equal = false;
  }
  if (cold_position < 0 || warm_position < 0 || cold_position != warm_position) {
    cmp.memcmp_equal = false;
    cmp.argmax_match = false;
  }
  return cmp;
}

Qwen38Gate3Verdict evaluate_gate3_verdict(const Qwen38Gate3Input& input) {
  Qwen38Gate3Verdict v;

  if (!input.qwen27b_pins_verified) {
    v.failures.push_back("qwen27b_pins_verified is false");
  }
  if (!input.short_ref_json_sha_verified) {
    v.failures.push_back("short reference.json SHA-256 not verified");
  }
  if (!input.long_ref_json_sha_verified) {
    v.failures.push_back("long-code-reference.json SHA-256 not verified");
  }
  if (!input.self_check_passed) {
    v.failures.push_back("self_check_gemm_parity failed or not executed");
  }

  bool reuse_ok = true;
  std::unordered_map<std::string_view, const Gate3ReuseCaseResult*> reuse_by_name;
  for (const auto& rc : input.reuse_cases) {
    const bool known = std::any_of(kGate3CaseContracts.begin(), kGate3CaseContracts.end(),
                                   [&](const Gate3CaseContract& cc) { return cc.name == rc.name; });
    if (!known) {
      reuse_ok = false;
      v.failures.push_back("unexpected reuse case: " + rc.name);
      continue;
    }
    if (!reuse_by_name.emplace(rc.name, &rc).second) {
      reuse_ok = false;
      v.failures.push_back("duplicate reuse case: " + rc.name);
    }
  }
  if (input.reuse_cases.size() != kGate3CaseContracts.size()) {
    reuse_ok = false;
    v.failures.push_back("reuse case count mismatch: expected " + std::to_string(kGate3CaseContracts.size()) +
                         ", got " + std::to_string(input.reuse_cases.size()));
  }

  for (const Gate3CaseContract& cc : kGate3CaseContracts) {
    const auto it = reuse_by_name.find(cc.name);
    if (it == reuse_by_name.end()) {
      reuse_ok = false;
      v.failures.push_back("missing required reuse/isolation case: " + std::string(cc.name));
      continue;
    }
    const Gate3ReuseCaseResult& rc = *it->second;
    if (!rc.executed) {
      reuse_ok = false;
      v.failures.push_back("reuse case not executed: " + rc.name);
      continue;
    }
    if (!std::isfinite(rc.restore_seconds) || rc.restore_seconds < 0.0 || !std::isfinite(rc.cold_elapsed_ms) ||
        rc.cold_elapsed_ms < 0.0 || !std::isfinite(rc.warm_elapsed_ms) || rc.warm_elapsed_ms < 0.0 ||
        !std::isfinite(rc.warm_prefill_seconds) || rc.warm_prefill_seconds < 0.0) {
      reuse_ok = false;
      v.failures.push_back("invalid or non-finite timing in reuse case: " + rc.name);
    }
    if (!rc.failure_note.empty()) {
      reuse_ok = false;
      v.failures.push_back("reuse case " + rc.name + " error: " + rc.failure_note);
    }
    if (rc.expected_reuse != cc.required_reuse || rc.actual_reuse != cc.required_reuse) {
      reuse_ok = false;
      v.failures.push_back("reuse case " + rc.name + " contract reuse mismatch: required " +
                           ingine::model::to_string(cc.required_reuse) + ", expected=" +
                           ingine::model::to_string(rc.expected_reuse) + ", actual=" +
                           ingine::model::to_string(rc.actual_reuse));
    }
    if (rc.expected_reused_tokens != cc.required_reused_tokens ||
        rc.actual_reused_tokens != cc.required_reused_tokens) {
      reuse_ok = false;
      v.failures.push_back("reuse case " + rc.name + " contract reused_tokens mismatch: required " +
                           std::to_string(cc.required_reused_tokens) + ", expected=" +
                           std::to_string(rc.expected_reused_tokens) + ", actual=" +
                           std::to_string(rc.actual_reused_tokens));
    }
    if (rc.prompt_len != cc.required_prompt_len) {
      reuse_ok = false;
      v.failures.push_back("reuse case " + rc.name + " prompt_len mismatch: required " +
                           std::to_string(cc.required_prompt_len) + ", got " + std::to_string(rc.prompt_len));
    }
    if (cc.require_nonzero_restore_bytes && rc.restored_bytes == 0) {
      reuse_ok = false;
      v.failures.push_back("reuse case " + rc.name + " Checkpoint reported zero restored_bytes");
    }
    if (!cc.require_nonzero_restore_bytes && rc.restored_bytes != 0) {
      reuse_ok = false;
      v.failures.push_back("reuse case " + rc.name + " non-Checkpoint reported nonzero restored_bytes=" +
                           std::to_string(rc.restored_bytes));
    }
    if (cc.require_zero_prefill_seconds && rc.warm_prefill_seconds != 0.0) {
      reuse_ok = false;
      v.failures.push_back("reuse case " + rc.name + " expected zero suffix prefill seconds on exact tip");
    }
    if (rc.cold_step_count != cc.required_steps || rc.warm_step_count != cc.required_steps ||
        rc.step_comparisons.size() != cc.required_steps) {
      reuse_ok = false;
      v.failures.push_back("reuse case " + rc.name + " step count mismatch: required " +
                           std::to_string(cc.required_steps) + ", cold=" + std::to_string(rc.cold_step_count) +
                           ", warm=" + std::to_string(rc.warm_step_count) +
                           ", comparisons=" + std::to_string(rc.step_comparisons.size()));
    }
    if (rc.cold_greedy_ids.size() != cc.required_steps || rc.warm_greedy_ids.size() != cc.required_steps ||
        rc.cold_greedy_ids != rc.warm_greedy_ids) {
      reuse_ok = false;
      v.failures.push_back("reuse case " + rc.name + " greedy token sequence mismatch or wrong count");
    }
    const std::size_t n_cmp = std::min(cc.required_steps, rc.step_comparisons.size());
    for (std::size_t si = 0; si < n_cmp; ++si) {
      const auto& sc = rc.step_comparisons[si];
      const int exp_pos = cc.first_step_position + static_cast<int>(si);
      if (sc.cold_position != exp_pos || sc.warm_position != exp_pos || sc.position != exp_pos) {
        reuse_ok = false;
        v.failures.push_back("reuse case " + rc.name + " step " + std::to_string(si) +
                             " position mismatch: required=" + std::to_string(exp_pos) +
                             ", cold=" + std::to_string(sc.cold_position) +
                             ", warm=" + std::to_string(sc.warm_position));
      }
      if (sc.cold_vocab_size != kGate3PinnedVocabSize || sc.warm_vocab_size != kGate3PinnedVocabSize ||
          sc.vocab_size != kGate3PinnedVocabSize) {
        reuse_ok = false;
        v.failures.push_back("reuse case " + rc.name + " step " + std::to_string(si) +
                             " vocab_size mismatch: required=" + std::to_string(kGate3PinnedVocabSize) +
                             ", cold=" + std::to_string(sc.cold_vocab_size) +
                             ", warm=" + std::to_string(sc.warm_vocab_size));
      }
      if (!sc.all_finite || !sc.memcmp_equal || !sc.argmax_match || !std::isfinite(sc.max_abs_diff) ||
          sc.max_abs_diff != 0.0f) {
        reuse_ok = false;
        v.failures.push_back("reuse case " + rc.name + " step " + std::to_string(si) +
                             " failed bitwise parity (finite=" + std::to_string(sc.all_finite) +
                             ", memcmp=" + std::to_string(sc.memcmp_equal) +
                             ", argmax_match=" + std::to_string(sc.argmax_match) + ")");
      }
      if (sc.cold_argmax < 0 || sc.cold_argmax >= static_cast<std::int32_t>(kGate3PinnedVocabSize) ||
          sc.warm_argmax < 0 || sc.warm_argmax >= static_cast<std::int32_t>(kGate3PinnedVocabSize)) {
        reuse_ok = false;
        v.failures.push_back("reuse case " + rc.name + " step " + std::to_string(si) + " argmax out of range");
      } else if (si < rc.cold_greedy_ids.size() && si < rc.warm_greedy_ids.size() &&
                 (sc.cold_argmax != rc.cold_greedy_ids[si] || sc.warm_argmax != rc.warm_greedy_ids[si])) {
        reuse_ok = false;
        v.failures.push_back("reuse case " + rc.name + " step " + std::to_string(si) +
                             " greedy ID does not match logit argmax");
      }
    }
  }
  v.reuse_cases_passed = reuse_ok;

  bool miss_ok = true;
  std::unordered_map<std::string_view, const Gate3ProtectedMissResult*> miss_by_name;
  for (const auto& pm : input.protected_misses) {
    const bool known = std::any_of(kGate3ProtectedMissContracts.begin(), kGate3ProtectedMissContracts.end(),
                                   [&](const Gate3ProtectedMissContract& mc) { return mc.name == pm.name; });
    if (!known) {
      miss_ok = false;
      v.failures.push_back("unexpected protected miss case: " + pm.name);
      continue;
    }
    if (!miss_by_name.emplace(pm.name, &pm).second) {
      miss_ok = false;
      v.failures.push_back("duplicate protected miss case: " + pm.name);
    }
  }
  if (input.protected_misses.size() != kGate3ProtectedMissContracts.size()) {
    miss_ok = false;
    v.failures.push_back("protected miss count mismatch: expected " +
                         std::to_string(kGate3ProtectedMissContracts.size()) + ", got " +
                         std::to_string(input.protected_misses.size()));
  }

  for (const Gate3ProtectedMissContract& mc : kGate3ProtectedMissContracts) {
    const auto it = miss_by_name.find(mc.name);
    if (it == miss_by_name.end()) {
      miss_ok = false;
      v.failures.push_back("missing required protected miss case: " + std::string(mc.name));
      continue;
    }
    const Gate3ProtectedMissResult& pm = *it->second;
    if (!pm.executed) {
      miss_ok = false;
      v.failures.push_back("protected miss case not executed: " + pm.name);
      continue;
    }
    if (pm.reason.empty()) {
      miss_ok = false;
      v.failures.push_back("protected miss case missing lookup reason: " + pm.name);
    }
    if (!std::isfinite(pm.max_abs_diff) || pm.max_abs_diff != 0.0) {
      miss_ok = false;
      v.failures.push_back("non-zero or non-finite diff in protected miss case: " + pm.name);
    }
    if (!pm.failure_note.empty()) {
      miss_ok = false;
      v.failures.push_back("protected miss case " + pm.name + " error: " + pm.failure_note);
    }
    if (pm.hit || pm.reuse != ingine::model::Reuse::None || pm.reused_tokens != 0) {
      miss_ok = false;
      v.failures.push_back("protected miss case " + pm.name + " failed closed check (hit=" + std::to_string(pm.hit) +
                           ", reused=" + std::to_string(pm.reused_tokens) + ")");
    }
    if (pm.cold_position != mc.expected_position || pm.warm_position != mc.expected_position ||
        pm.post_miss_position != mc.expected_position) {
      miss_ok = false;
      v.failures.push_back("protected miss case " + pm.name + " position mismatch: expected=" +
                           std::to_string(mc.expected_position) + ", cold=" + std::to_string(pm.cold_position) +
                           ", warm=" + std::to_string(pm.warm_position) +
                           ", post=" + std::to_string(pm.post_miss_position));
    }
    if (pm.post_miss_vocab_size != kGate3PinnedVocabSize) {
      miss_ok = false;
      v.failures.push_back("protected miss case " + pm.name + " vocab_size mismatch: expected=" +
                           std::to_string(kGate3PinnedVocabSize) + ", got=" + std::to_string(pm.post_miss_vocab_size));
    }
    if (pm.cold_argmax < 0 || pm.cold_argmax >= static_cast<std::int32_t>(kGate3PinnedVocabSize) ||
        pm.cold_argmax != pm.warm_argmax || pm.cold_argmax != pm.cold_greedy_id ||
        pm.warm_argmax != pm.warm_greedy_id) {
      miss_ok = false;
      v.failures.push_back("protected miss case " + pm.name + " argmax/greedy ID mismatch");
    }
    if (!pm.post_miss_bit_exact) {
      miss_ok = false;
      v.failures.push_back("protected miss case " + pm.name + " post-miss execution not bitwise exact");
    }
  }
  v.protected_misses_passed = miss_ok;

  v.executed_checks_passed = input.self_check_passed && v.reuse_cases_passed && v.protected_misses_passed;
  if (input.diagnostic_mode) {
    v.failures.push_back("diagnostic mode cannot qualify for Gate 3");
  }
  v.qualification_passed = !input.diagnostic_mode && input.qwen27b_pins_verified && input.short_ref_json_sha_verified &&
                           input.long_ref_json_sha_verified && v.executed_checks_passed;
  return v;
}

}  // namespace ingine::backends::qwen38

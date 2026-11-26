#pragma once

#include "ingine/generator.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ingine::backends::qwen38::test_883 {

struct FollowupRestoreCheck {
  std::size_t expected_reused_tokens = 0;
  int actual_reused_tokens = 0;
  ingine::model::Reuse actual_reuse = ingine::model::Reuse::None;
  std::uint64_t restored_bytes = 0;
  bool passed = false;
};

// Verifies that a follow-up turn after a prior turn whose pre-suffix boundary was
// `prev_pre_suffix_tokens` actually restored that exact pre-suffix GDN checkpoint
// (`Reuse::Checkpoint`, exact `reused_tokens`, and nonzero `restored_bytes`).
inline FollowupRestoreCheck check_presuffix_followup_restore(std::size_t prev_pre_suffix_tokens,
                                                             const ingine::model::GenerateStats& stats) noexcept {
  FollowupRestoreCheck out;
  out.expected_reused_tokens = prev_pre_suffix_tokens;
  out.actual_reused_tokens = stats.reused_tokens;
  out.actual_reuse = stats.reuse;
  out.restored_bytes = stats.restored_bytes.value_or(0ull);
  out.passed = (prev_pre_suffix_tokens > 0) && (stats.reuse == ingine::model::Reuse::Checkpoint) &&
               (stats.reused_tokens >= 0) &&
               (static_cast<std::size_t>(stats.reused_tokens) == prev_pre_suffix_tokens) &&
               (out.restored_bytes > 0ull);
  return out;
}

inline bool check_multiturn_presuffix_restores(std::span<const std::size_t> pre_suffix_boundaries,
                                               std::span<const ingine::model::GenerateStats> turn_stats) noexcept {
  if (pre_suffix_boundaries.size() != turn_stats.size() || turn_stats.size() < 2) return false;
  for (std::size_t ti = 1; ti < turn_stats.size(); ++ti) {
    if (!check_presuffix_followup_restore(pre_suffix_boundaries[ti - 1], turn_stats[ti]).passed) return false;
  }
  return true;
}

}  // namespace ingine::backends::qwen38::test_883

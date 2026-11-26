#pragma once

#include "ingine/model_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::backends::qwen38 {

inline constexpr std::string_view kGate3ContractVersion = "gate3-v1-bit-exact";
inline constexpr std::string_view kGate3ShortRefJsonSha256 =
    "277f47825bac44feba200dfab6ab555d637b66a30d31c2bfdfa2f0f09afcc5a3";
inline constexpr std::string_view kGate3LongRefJsonSha256 =
    "663ab9262b33ae26a4f07dbbd08ac02ea188990dbc1cc7c8df1effa474e0f15a";
inline constexpr std::size_t kGate3PinnedVocabSize = 248320;
inline constexpr std::size_t kGate3RequiredSteps = 8;

struct Gate3CaseContract {
  std::string_view name;
  ingine::model::Reuse required_reuse;
  std::size_t required_reused_tokens;
  std::size_t required_prompt_len;
  std::size_t required_steps;
  int first_step_position;
  bool require_nonzero_restore_bytes;
  bool require_zero_prefill_seconds;
};

inline constexpr std::array<Gate3CaseContract, 9> kGate3CaseContracts = {{
    {"extend_suffix", ingine::model::Reuse::Extend, 20, 32, 8, 31, false, false},
    {"extend_exact_tip", ingine::model::Reuse::Extend, 32, 32, 8, 31, false, true},
    {"checkpoint_after_decode_fork", ingine::model::Reuse::Checkpoint, 32, 44, 8, 43, true, false},
    {"checkpoint_multiturn_rollback", ingine::model::Reuse::Checkpoint, 20, 32, 8, 31, true, false},
    {"checkpoint_midfork_earlier_ckpt", ingine::model::Reuse::Checkpoint, 20, 32, 8, 31, true, false},
    {"multichunk_checkpoint_fork", ingine::model::Reuse::Checkpoint, 320, 608, 8, 607, true, false},
    {"multichunk_extend_suffix", ingine::model::Reuse::Extend, 320, 608, 8, 607, false, false},
    {"session_reset_isolation", ingine::model::Reuse::None, 0, 32, 8, 31, false, false},
    {"multi_session_isolation", ingine::model::Reuse::Extend, 20, 32, 8, 31, false, false},
}};

inline constexpr std::array<std::string_view, 9> kGate3RequiredReuseCaseNames = {
    "extend_suffix",
    "extend_exact_tip",
    "checkpoint_after_decode_fork",
    "checkpoint_multiturn_rollback",
    "checkpoint_midfork_earlier_ckpt",
    "multichunk_checkpoint_fork",
    "multichunk_extend_suffix",
    "session_reset_isolation",
    "multi_session_isolation",
};

struct Gate3ProtectedMissContract {
  std::string_view name;
  int expected_position;
};

inline constexpr std::array<Gate3ProtectedMissContract, 14> kGate3ProtectedMissContracts = {{
    {"miss_explicit_bypass", 31},
    {"miss_empty_common_prefix", 31},
    {"miss_cache_namespace_mismatch", 31},
    {"miss_token_mask_presence_mismatch", 31},
    {"miss_token_mask_bit_flip_at_pos0", 31},
    {"miss_uncheckpointed_short_fork", 19},
    {"miss_unverified_identity", 31},
    {"miss_headers_only_identity", 31},
    {"miss_self_hashed_identity", 31},
    {"miss_mutated_shard_set_sha256", 31},
    {"miss_mutated_weight_manifest_sha256", 31},
    {"miss_mutated_numeric_abi", 31},
    {"miss_mutated_media_sha256", 31},
    {"miss_evicted_checkpoint_fork", 43},
}};

inline constexpr std::array<std::string_view, 14> kGate3RequiredProtectedMissNames = {
    "miss_explicit_bypass",
    "miss_empty_common_prefix",
    "miss_cache_namespace_mismatch",
    "miss_token_mask_presence_mismatch",
    "miss_token_mask_bit_flip_at_pos0",
    "miss_uncheckpointed_short_fork",
    "miss_unverified_identity",
    "miss_headers_only_identity",
    "miss_self_hashed_identity",
    "miss_mutated_shard_set_sha256",
    "miss_mutated_weight_manifest_sha256",
    "miss_mutated_numeric_abi",
    "miss_mutated_media_sha256",
    "miss_evicted_checkpoint_fork",
};

// Pure-CPU bookkeeper for Qwen3.8 live session tip + sparse saved checkpoints.
class Qwen38CheckpointIndex {
 public:
  explicit Qwen38CheckpointIndex(std::size_t max_checkpoints = 4);

  void reset(const ingine::model::BackendStateIdentity& default_id);
  void set_identity(ingine::model::BackendStateIdentity id) { session_identity_ = std::move(id); }
  [[nodiscard]] const ingine::model::BackendStateIdentity& identity() const noexcept { return session_identity_; }

  void append_committed(std::span<const std::int32_t> tokens, std::span<const std::uint8_t> mask);
  void truncate_committed(std::size_t new_len);

  void set_tip_logits(int pos) noexcept {
    has_tip_logits_ = true;
    tip_logits_pos_ = pos;
  }
  void clear_tip_logits() noexcept {
    has_tip_logits_ = false;
    tip_logits_pos_ = -1;
  }
  [[nodiscard]] bool has_tip_logits() const noexcept { return has_tip_logits_; }
  [[nodiscard]] int tip_logits_pos() const noexcept { return tip_logits_pos_; }

  // Registers a saved checkpoint at `token_count` (must satisfy 0 < token_count <= committed_tokens_.size()).
  // Returns the oldest evicted checkpoint length if capacity was exceeded, else std::nullopt.
  std::optional<std::size_t> record_checkpoint(std::size_t token_count);
  std::vector<std::size_t> prune_above(std::size_t max_len);
  std::vector<std::size_t> evict_all_checkpoints();

  [[nodiscard]] std::size_t restorable_prefix_len(std::size_t max_prefix_len) const noexcept;
  [[nodiscard]] bool can_truncate_to(std::size_t new_len) const noexcept;

  [[nodiscard]] std::span<const std::int32_t> committed_tokens() const noexcept { return committed_tokens_; }
  [[nodiscard]] std::span<const std::uint8_t> committed_token_mask() const noexcept { return committed_token_mask_; }
  [[nodiscard]] const std::vector<std::size_t>& saved_checkpoint_lengths() const noexcept {
    return checkpoint_lengths_;
  }
  [[nodiscard]] std::size_t max_checkpoints() const noexcept { return max_checkpoints_; }

 private:
  std::size_t max_checkpoints_ = 4;
  ingine::model::BackendStateIdentity session_identity_;
  std::vector<std::int32_t> committed_tokens_;
  std::vector<std::uint8_t> committed_token_mask_;
  std::vector<std::size_t> checkpoint_lengths_;
  bool has_tip_logits_ = false;
  int tip_logits_pos_ = -1;
};

struct Gate3StepComparison {
  int position = -1;
  int cold_position = -1;
  int warm_position = -1;
  std::size_t vocab_size = 0;
  std::size_t cold_vocab_size = 0;
  std::size_t warm_vocab_size = 0;
  bool all_finite = false;
  bool memcmp_equal = false;
  std::size_t first_diff_vocab_idx = 0;
  float max_abs_diff = 0.0f;
  std::int32_t cold_argmax = -1;
  std::int32_t warm_argmax = -1;
  bool argmax_match = false;
};

// Compares full-vocabulary logit vectors strictly bitwise (memcmp == 0 + zero NaN/Inf)
// while preserving explicit cold and warm step positions verbatim.
Gate3StepComparison compare_logits_bit_exact(std::span<const float> cold_logits, std::span<const float> warm_logits,
                                             int cold_position, int warm_position);
Gate3StepComparison compare_logits_bit_exact(std::span<const float> cold_logits, std::span<const float> warm_logits,
                                             int position = 0);

struct Gate3ReuseCaseResult {
  std::string name;
  bool executed = false;
  ingine::model::Reuse expected_reuse = ingine::model::Reuse::None;
  ingine::model::Reuse actual_reuse = ingine::model::Reuse::None;
  std::size_t expected_reused_tokens = 0;
  std::size_t actual_reused_tokens = 0;
  std::size_t prompt_len = 0;
  std::size_t cold_step_count = 0;
  std::size_t warm_step_count = 0;
  double warm_prefill_seconds = 0.0;
  std::uint64_t restored_bytes = 0;
  double restore_seconds = 0.0;
  double cold_elapsed_ms = 0.0;
  double warm_elapsed_ms = 0.0;
  std::vector<std::int32_t> cold_greedy_ids;
  std::vector<std::int32_t> warm_greedy_ids;
  std::vector<Gate3StepComparison> step_comparisons;
  std::string failure_note;
};

struct Gate3ProtectedMissResult {
  std::string name;
  bool executed = false;
  bool hit = true;
  ingine::model::Reuse reuse = ingine::model::Reuse::Extend;
  std::size_t reused_tokens = 0;
  std::string reason;
  int cold_position = -1;
  int warm_position = -1;
  int post_miss_position = -1;
  std::size_t post_miss_vocab_size = 0;
  std::int32_t cold_argmax = -1;
  std::int32_t warm_argmax = -1;
  std::int32_t cold_greedy_id = -1;
  std::int32_t warm_greedy_id = -1;
  bool post_miss_bit_exact = false;
  double max_abs_diff = 0.0;
  std::string failure_note;
};

struct Qwen38Gate3Input {
  bool diagnostic_mode = false;
  bool qwen27b_pins_verified = false;
  bool short_ref_json_sha_verified = false;
  bool long_ref_json_sha_verified = false;
  bool self_check_passed = false;

  std::vector<Gate3ReuseCaseResult> reuse_cases;
  std::vector<Gate3ProtectedMissResult> protected_misses;
};

struct Qwen38Gate3Verdict {
  std::string contract_version = std::string(kGate3ContractVersion);
  bool reuse_cases_passed = false;
  bool protected_misses_passed = false;
  bool executed_checks_passed = false;
  bool qualification_passed = false;
  std::vector<std::string> failures;
};

Qwen38Gate3Verdict evaluate_gate3_verdict(const Qwen38Gate3Input& input);

}  // namespace ingine::backends::qwen38

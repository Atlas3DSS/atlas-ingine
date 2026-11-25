#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ingine::backends::qwen38 {

inline constexpr std::string_view kGate2ContractVersion = "v1.1-option1-addendum";
inline constexpr std::string_view kGate2ShortRefSha256 =
    "30dd669b1baf6a1ec517b635c083b275dd72417ca07521bf33cecdd954f02a3f";
inline constexpr std::string_view kGate2ShortRefMetaSha256 =
    "277f47825bac44feba200dfab6ab555d637b66a30d31c2bfdfa2f0f09afcc5a3";
inline constexpr std::string_view kGate2AltShortRefB1Sha256 =
    "272619388f38c5f621cffc93519c2571579fb28e4e399048c43da4f8b3fd9d71";
inline constexpr std::string_view kGate2AltShortRefB1MetaSha256 =
    "866d1ef4f63db7a2183fab7aa492f911dc081273f2f629e2b624e8cca182b019";
inline constexpr std::string_view kGate2LongCodeRefSha256 =
    "d4479957d5ec101c8274d7a7fe59032d0171423ebacd8607a6aeb61e00f1afc9";
inline constexpr std::string_view kGate2ExpectedNumericMode =
    "eager_fp32_nvfp4_weight_only_calibrated_fp8_act";

struct Qwen38Top2Logits {
  std::int32_t top1_id = -1;
  float top1_logit = 0.0f;
  std::int32_t top2_id = -1;
  float top2_logit = 0.0f;
  bool finite = false;
};

struct Qwen38Prompt0TrajectoryEval {
  std::size_t release_b3_matches = 0;
  int release_b3_first_diff_idx = -1;
  std::size_t canonical_b1_matches = 0;
  int canonical_b1_first_diff_idx = -1;
  bool matches_release_b3_intact = false;
  bool matches_canonical_b1_intact = false;
  bool matched_intact_canonical = false;
  std::string chosen_realization = "none";  // "release_b3" | "canonical_b1" | "none"
};

struct Qwen38LogitParityPoint {
  int position = 0;
  double kld = 0.0;  // D_KL(p_ref || p_cand)
  std::int32_t ref_top1 = -1;
  std::int32_t cand_top1 = -1;
  bool top1_match = false;
  float max_abs_diff = 0.0f;
  bool finite = true;
};

struct Qwen38KldSummary {
  std::size_t positions = 0;
  double kld_mean = 0.0;
  double kld_p99 = 0.0;
  double kld_max = 0.0;
  std::size_t top1_matches = 0;
  double top1_agreement = 0.0;
  bool all_finite = true;
};

struct Qwen38Gate2Input {
  bool diagnostic_mode = false;
  bool qwen27b_pins_verified = false;
  bool short_ref_sha_verified = false;
  bool long_ref_sha_verified = false;

  // Prospective Option 1 addendum (att-ba5deaaf): when enabled, binds both canonical short captures
  // and allows exact 16/16 intact prompt_0 trajectory match against either release-b3 or canonical-b1.
  bool allow_canonical_b1_addendum = false;
  bool short_ref_meta_sha_verified = false;
  bool alt_short_ref_b1_sha_verified = false;
  bool alt_short_ref_b1_meta_sha_verified = false;
  bool alt_short_ref_b1_prompt0_ids_match = false;

  std::size_t short_prompt_count = 0;
  std::size_t prompt0_greedy_matches = 0;
  std::size_t prompt0_greedy_total = 0;
  Qwen38KldSummary short_summary;

  std::size_t prompt0_b1_greedy_matches = 0;
  std::size_t prompt0_b1_greedy_total = 0;
  bool prompt0_b1_eval_executed = false;
  Qwen38KldSummary prompt0_b1_summary;

  std::size_t long_prompt_count = 0;
  Qwen38KldSummary long_summary;

  std::size_t chunk_vs_step_prompt_count = 0;
  Qwen38KldSummary chunk_vs_step_summary;
  std::size_t chunk_vs_step_greedy_matches = 0;
  std::size_t chunk_vs_step_greedy_total = 0;
};

struct Qwen38Gate2Verdict {
  std::string contract_version = "v1.0-release-only";
  std::string chosen_prompt0_realization = "none";  // "release_b3" | "canonical_b1" | "none"
  bool release_only_short_passed = false;
  bool release_only_qualification_passed = false;
  std::vector<std::string> release_only_failures;
  bool canonical_b1_branch_passed = false;

  bool short_prompts_passed = false;
  bool long_code_passed = false;
  bool chunked_vs_stepwise_passed = false;
  bool qualification_passed = false;
  bool executed_checks_passed = false;
  std::vector<std::string> failures;
};

// Computes exact FP64 D_KL(p_ref || p_cand) and top-1 argmax agreement for one logit row.
// Fails closed (finite=false, kld=+inf) on empty/mismatched spans or any NaN/Inf.
Qwen38LogitParityPoint compute_logit_kld_and_top1(std::span<const float> ref_logits,
                                                  std::span<const float> cand_logits, int position = 0);

Qwen38Top2Logits compute_top2_logits(std::span<const float> logits);

bool verify_file_sha256(const std::filesystem::path& path, std::string_view expected_hex,
                        std::string* actual_hex_out = nullptr);

Qwen38Prompt0TrajectoryEval evaluate_prompt0_canonical_trajectory(std::span<const std::int32_t> cand_ids,
                                                                 std::span<const std::int32_t> release_b3_ids,
                                                                 std::span<const std::int32_t> canonical_b1_ids);

Qwen38KldSummary summarize_kld_metrics(std::span<const Qwen38LogitParityPoint> points);

Qwen38Gate2Verdict evaluate_gate2_verdict(const Qwen38Gate2Input& input);

struct Qwen38RefTensorEntry {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::uint64_t abs_offset = 0;
  std::uint64_t byte_size = 0;
};

// Streaming read-only safetensors reader for large Gate 2 reference logit packs (POSIX pread).
class Qwen38RefSafetensors {
 public:
  explicit Qwen38RefSafetensors(const std::filesystem::path& path);
  ~Qwen38RefSafetensors();

  Qwen38RefSafetensors(const Qwen38RefSafetensors&) = delete;
  Qwen38RefSafetensors& operator=(const Qwen38RefSafetensors&) = delete;
  Qwen38RefSafetensors(Qwen38RefSafetensors&& other) noexcept;
  Qwen38RefSafetensors& operator=(Qwen38RefSafetensors&& other) noexcept;

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
  [[nodiscard]] const std::string& sha256_hex() const;
  [[nodiscard]] bool verify_sha256(std::string_view expected_hex) const;
  [[nodiscard]] const Qwen38RefTensorEntry* find(std::string_view name) const;
  [[nodiscard]] const Qwen38RefTensorEntry& at(std::string_view name) const;

  [[nodiscard]] std::vector<std::int64_t> read_i64(std::string_view name) const;
  void read_f32_row(std::string_view name, std::size_t row_idx, std::span<float> out_row) const;

 private:
  void close_fd() noexcept;

  std::filesystem::path path_;
  int fd_ = -1;
  std::uint64_t file_size_ = 0;
  std::unordered_map<std::string, Qwen38RefTensorEntry> entries_;
  mutable std::string cached_sha256_;
};

}  // namespace ingine::backends::qwen38

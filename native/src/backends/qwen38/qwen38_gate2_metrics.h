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

inline constexpr std::string_view kGate2ShortRefSha256 =
    "30dd669b1baf6a1ec517b635c083b275dd72417ca07521bf33cecdd954f02a3f";
inline constexpr std::string_view kGate2LongCodeRefSha256 =
    "d4479957d5ec101c8274d7a7fe59032d0171423ebacd8607a6aeb61e00f1afc9";

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

  std::size_t short_prompt_count = 0;
  std::size_t prompt0_greedy_matches = 0;
  std::size_t prompt0_greedy_total = 0;
  Qwen38KldSummary short_summary;

  std::size_t long_prompt_count = 0;
  Qwen38KldSummary long_summary;

  std::size_t chunk_vs_step_prompt_count = 0;
  Qwen38KldSummary chunk_vs_step_summary;
  std::size_t chunk_vs_step_greedy_matches = 0;
  std::size_t chunk_vs_step_greedy_total = 0;
};

struct Qwen38Gate2Verdict {
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

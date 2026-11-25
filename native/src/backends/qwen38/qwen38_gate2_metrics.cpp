#include "backends/qwen38/qwen38_gate2_metrics.h"

#include "backends/qwen38/qwen38_manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace ingine::backends::qwen38 {
namespace {

void pread_all(int fd, void* dst, std::size_t bytes, std::uint64_t offset, const std::filesystem::path& path) {
  auto* out = static_cast<std::uint8_t*>(dst);
  std::size_t done = 0;
  while (done < bytes) {
    const ssize_t got = ::pread(fd, out + done, bytes - done, static_cast<off_t>(offset + done));
    if (got < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("pread failed on " + path.string() + ": " + std::strerror(errno));
    }
    if (got == 0) {
      throw std::runtime_error("unexpected EOF in pread on " + path.string());
    }
    done += static_cast<std::size_t>(got);
  }
}

std::string lower_ascii(std::string_view s) {
  std::string out(s);
  for (char& ch : out) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return out;
}

}  // namespace

Qwen38LogitParityPoint compute_logit_kld_and_top1(std::span<const float> ref_logits,
                                                  std::span<const float> cand_logits, int position) {
  Qwen38LogitParityPoint pt;
  pt.position = position;
  if (ref_logits.empty() || ref_logits.size() != cand_logits.size()) {
    pt.finite = false;
    pt.kld = std::numeric_limits<double>::infinity();
    pt.max_abs_diff = std::numeric_limits<float>::infinity();
    return pt;
  }

  double max_r = -std::numeric_limits<double>::infinity();
  double max_c = -std::numeric_limits<double>::infinity();
  std::int32_t best_r = 0;
  std::int32_t best_c = 0;
  float max_abs_d = 0.0f;

  for (std::size_t i = 0; i < ref_logits.size(); ++i) {
    const float rv = ref_logits[i];
    const float cv = cand_logits[i];
    if (!std::isfinite(rv) || !std::isfinite(cv)) {
      pt.finite = false;
      pt.kld = std::numeric_limits<double>::infinity();
      pt.max_abs_diff = std::numeric_limits<float>::infinity();
      return pt;
    }
    const double dr = static_cast<double>(rv);
    const double dc = static_cast<double>(cv);
    if (dr > max_r) {
      max_r = dr;
      best_r = static_cast<std::int32_t>(i);
    }
    if (dc > max_c) {
      max_c = dc;
      best_c = static_cast<std::int32_t>(i);
    }
    const float ad = std::fabs(rv - cv);
    if (ad > max_abs_d) max_abs_d = ad;
  }

  double sum_exp_r = 0.0;
  double sum_exp_c = 0.0;
  for (std::size_t i = 0; i < ref_logits.size(); ++i) {
    sum_exp_r += std::exp(static_cast<double>(ref_logits[i]) - max_r);
    sum_exp_c += std::exp(static_cast<double>(cand_logits[i]) - max_c);
  }
  if (!std::isfinite(sum_exp_r) || !std::isfinite(sum_exp_c) || sum_exp_r <= 0.0 || sum_exp_c <= 0.0) {
    pt.finite = false;
    pt.kld = std::numeric_limits<double>::infinity();
    pt.max_abs_diff = max_abs_d;
    return pt;
  }

  const double logz_r = max_r + std::log(sum_exp_r);
  const double logz_c = max_c + std::log(sum_exp_c);
  double kld = 0.0;
  for (std::size_t i = 0; i < ref_logits.size(); ++i) {
    const double lp_r = static_cast<double>(ref_logits[i]) - logz_r;
    const double lp_c = static_cast<double>(cand_logits[i]) - logz_c;
    const double p_r = std::exp(lp_r);
    if (p_r > 0.0) kld += p_r * (lp_r - lp_c);
  }
  if (!std::isfinite(kld)) {
    pt.finite = false;
    pt.kld = std::numeric_limits<double>::infinity();
    pt.max_abs_diff = max_abs_d;
    return pt;
  }
  if (kld < 0.0 && kld >= -1e-12) kld = 0.0;
  if (kld < 0.0) {
    pt.finite = false;
    pt.kld = std::numeric_limits<double>::infinity();
    pt.max_abs_diff = max_abs_d;
    return pt;
  }

  pt.kld = kld;
  pt.ref_top1 = best_r;
  pt.cand_top1 = best_c;
  pt.top1_match = (best_r == best_c);
  pt.max_abs_diff = max_abs_d;
  pt.finite = true;
  return pt;
}

Qwen38KldSummary summarize_kld_metrics(std::span<const Qwen38LogitParityPoint> points) {
  Qwen38KldSummary s;
  s.positions = points.size();
  if (points.empty()) {
    s.all_finite = false;
    return s;
  }

  std::vector<double> vals;
  vals.reserve(points.size());
  double sum = 0.0;
  double mx = 0.0;
  for (const auto& pt : points) {
    if (!pt.finite || !std::isfinite(pt.kld)) {
      s.all_finite = false;
      s.kld_mean = std::numeric_limits<double>::infinity();
      s.kld_p99 = std::numeric_limits<double>::infinity();
      s.kld_max = std::numeric_limits<double>::infinity();
      return s;
    }
    vals.push_back(pt.kld);
    sum += pt.kld;
    if (pt.kld > mx) mx = pt.kld;
    if (pt.top1_match) ++s.top1_matches;
  }

  std::sort(vals.begin(), vals.end());
  const double pos = 0.99 * static_cast<double>(vals.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
  const std::size_t hi = static_cast<std::size_t>(std::ceil(pos));
  const double frac = pos - static_cast<double>(lo);

  s.kld_mean = sum / static_cast<double>(vals.size());
  s.kld_p99 = vals[lo] * (1.0 - frac) + vals[hi] * frac;
  s.kld_max = mx;
  s.top1_agreement = static_cast<double>(s.top1_matches) / static_cast<double>(points.size());
  s.all_finite = true;
  return s;
}

Qwen38Top2Logits compute_top2_logits(std::span<const float> logits) {
  Qwen38Top2Logits out;
  if (logits.empty()) return out;
  double b1 = -std::numeric_limits<double>::infinity();
  double b2 = -std::numeric_limits<double>::infinity();
  std::int32_t i1 = -1;
  std::int32_t i2 = -1;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const float v = logits[i];
    if (!std::isfinite(v)) return Qwen38Top2Logits{};
    const double dv = static_cast<double>(v);
    if (dv > b1) {
      b2 = b1;
      i2 = i1;
      b1 = dv;
      i1 = static_cast<std::int32_t>(i);
    } else if (dv > b2) {
      b2 = dv;
      i2 = static_cast<std::int32_t>(i);
    }
  }
  out.top1_id = i1;
  out.top1_logit = static_cast<float>(b1);
  out.top2_id = i2;
  out.top2_logit = static_cast<float>(i2 >= 0 ? b2 : 0.0);
  out.finite = true;
  return out;
}

bool verify_file_sha256(const std::filesystem::path& path, std::string_view expected_hex,
                        std::string* actual_hex_out) {
  if (actual_hex_out) actual_hex_out->clear();
  try {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec) || ec) return false;
    const std::string actual = sha256_file_hex_streaming(path);
    if (actual_hex_out) *actual_hex_out = actual;
    return lower_ascii(actual) == lower_ascii(expected_hex);
  } catch (...) {
    return false;
  }
}

Qwen38Prompt0TrajectoryEval evaluate_prompt0_canonical_trajectory(std::span<const std::int32_t> cand_ids,
                                                                 std::span<const std::int32_t> release_b3_ids,
                                                                 std::span<const std::int32_t> canonical_b1_ids) {
  Qwen38Prompt0TrajectoryEval ev;
  auto cmp = [](std::span<const std::int32_t> a, std::span<const std::int32_t> b, std::size_t& matches_out,
                int& first_diff_out, bool& intact_out) {
    matches_out = 0;
    first_diff_out = -1;
    const std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
      if (a[i] == b[i]) {
        ++matches_out;
      } else if (first_diff_out < 0) {
        first_diff_out = static_cast<int>(i);
      }
    }
    if (a.size() != b.size() && first_diff_out < 0) {
      first_diff_out = static_cast<int>(n);
    }
    intact_out = (a.size() == 16u && b.size() == 16u && matches_out == 16u && first_diff_out == -1);
  };
  cmp(cand_ids, release_b3_ids, ev.release_b3_matches, ev.release_b3_first_diff_idx, ev.matches_release_b3_intact);
  cmp(cand_ids, canonical_b1_ids, ev.canonical_b1_matches, ev.canonical_b1_first_diff_idx,
      ev.matches_canonical_b1_intact);
  if (ev.matches_release_b3_intact) {
    ev.matched_intact_canonical = true;
    ev.chosen_realization = "release_b3";
  } else if (ev.matches_canonical_b1_intact) {
    ev.matched_intact_canonical = true;
    ev.chosen_realization = "canonical_b1";
  } else {
    ev.matched_intact_canonical = false;
    ev.chosen_realization = "none";
  }
  return ev;
}

Qwen38Gate2Verdict evaluate_gate2_verdict(const Qwen38Gate2Input& in) {
  Qwen38Gate2Verdict v;

  auto summary_scalars_valid = [](const Qwen38KldSummary& s) noexcept {
    return s.all_finite && std::isfinite(s.kld_mean) && std::isfinite(s.kld_p99) && std::isfinite(s.kld_max) &&
           std::isfinite(s.top1_agreement) && s.kld_mean >= 0.0 && s.kld_p99 >= 0.0 && s.kld_max >= 0.0 &&
           s.top1_agreement >= 0.0 && s.top1_agreement <= 1.0 && s.top1_matches <= s.positions;
  };

  auto check_short_distribution = [&](std::vector<std::string>& errs) {
    if (in.short_prompt_count != 3 || in.short_summary.positions != 141) {
      errs.push_back("short_prompts incomplete count/positions");
      return false;
    }
    if (!summary_scalars_valid(in.short_summary)) {
      errs.push_back("short_prompts nonfinite or out-of-range summary");
      return false;
    }
    bool ok = true;
    if (in.short_summary.kld_mean > 0.005) {
      errs.push_back("short_prompts kld_mean > 0.005");
      ok = false;
    }
    if (in.short_summary.kld_max > 0.05) {
      errs.push_back("short_prompts kld_max > 0.05");
      ok = false;
    }
    if (in.short_summary.top1_agreement < 0.90) {
      errs.push_back("short_prompts top1_agreement < 0.90");
      ok = false;
    }
    return ok;
  };

  auto check_b1_distribution = [&](std::vector<std::string>& errs) {
    if (!in.prompt0_b1_eval_executed || in.prompt0_b1_summary.positions != 47) {
      errs.push_back("canonical-b1 prompt_0 teacher-forced evaluation incomplete/missing (expected 47 rows)");
      return false;
    }
    if (!summary_scalars_valid(in.prompt0_b1_summary)) {
      errs.push_back("canonical-b1 prompt_0 nonfinite or out-of-range summary");
      return false;
    }
    bool ok = true;
    if (in.prompt0_b1_summary.kld_mean > 0.005) {
      errs.push_back("canonical-b1 prompt_0 kld_mean > 0.005");
      ok = false;
    }
    if (in.prompt0_b1_summary.kld_max > 0.05) {
      errs.push_back("canonical-b1 prompt_0 kld_max > 0.05");
      ok = false;
    }
    if (in.prompt0_b1_summary.top1_agreement < 0.90) {
      errs.push_back("canonical-b1 prompt_0 top1_agreement < 0.90");
      ok = false;
    }
    return ok;
  };

  auto check_long = [&](std::vector<std::string>& errs) {
    if (in.long_prompt_count != 6 || in.long_summary.positions != 288) {
      errs.push_back("long_code incomplete count/positions");
      return false;
    }
    if (!summary_scalars_valid(in.long_summary)) {
      errs.push_back("long_code nonfinite or out-of-range summary");
      return false;
    }
    bool ok = true;
    if (in.long_summary.kld_mean > 0.01) {
      errs.push_back("long_code kld_mean > 0.01");
      ok = false;
    }
    if (in.long_summary.kld_p99 > 0.08) {
      errs.push_back("long_code kld_p99 > 0.08");
      ok = false;
    }
    if (in.long_summary.top1_agreement < 0.95) {
      errs.push_back("long_code top1_agreement < 0.95");
      ok = false;
    }
    return ok;
  };

  auto check_chunk = [&](std::vector<std::string>& errs) {
    if (in.chunk_vs_step_prompt_count != 3 || in.chunk_vs_step_summary.positions != 3) {
      errs.push_back("chunked_vs_stepwise incomplete count/positions");
      return false;
    }
    if (!summary_scalars_valid(in.chunk_vs_step_summary)) {
      errs.push_back("chunked_vs_stepwise nonfinite or out-of-range summary");
      return false;
    }
    bool ok = true;
    if (in.chunk_vs_step_summary.kld_max > 1e-4) {
      errs.push_back("chunked_vs_stepwise kld_max > 1e-4");
      ok = false;
    }
    if (in.chunk_vs_step_greedy_total != 48 || in.chunk_vs_step_greedy_matches != 48) {
      errs.push_back("chunked_vs_stepwise short greedy match < 48/48");
      ok = false;
    }
    return ok;
  };

  const bool matched_b3 = (in.prompt0_greedy_total == 16 && in.prompt0_greedy_matches == 16);
  const bool matched_b1 = (in.prompt0_b1_greedy_total == 16 && in.prompt0_b1_greedy_matches == 16);

  // Strict release-only verdict (always computed and preserved).
  {
    bool rel_short_ok = check_short_distribution(v.release_only_failures);
    if (!matched_b3) {
      v.release_only_failures.push_back("short prompt_0 greedy match < 16/16");
      rel_short_ok = false;
    }
    v.release_only_short_passed = rel_short_ok;
    const bool rel_long_ok = check_long(v.release_only_failures);
    const bool rel_chunk_ok = check_chunk(v.release_only_failures);
    if (in.diagnostic_mode) v.release_only_failures.push_back("diagnostic_mode forbids Gate 2 qualification");
    if (!in.qwen27b_pins_verified) v.release_only_failures.push_back("qwen27b_pins_verified is false");
    if (!in.short_ref_sha_verified) v.release_only_failures.push_back("short reference SHA-256 mismatch");
    if (!in.long_ref_sha_verified) v.release_only_failures.push_back("long-code reference SHA-256 mismatch");
    v.release_only_qualification_passed = !in.diagnostic_mode && in.qwen27b_pins_verified &&
                                          in.short_ref_sha_verified && in.long_ref_sha_verified && rel_short_ok &&
                                          rel_long_ok && rel_chunk_ok;
  }

  v.long_code_passed = check_long(v.failures);
  v.chunked_vs_stepwise_passed = check_chunk(v.failures);

  const bool addendum_pins_ok = in.short_ref_sha_verified && in.short_ref_meta_sha_verified &&
                                in.alt_short_ref_b1_sha_verified && in.alt_short_ref_b1_meta_sha_verified &&
                                in.alt_short_ref_b1_prompt0_ids_match;

  if (!in.allow_canonical_b1_addendum) {
    v.contract_version = "v1.0-release-only";
    v.chosen_prompt0_realization = matched_b3 ? "release_b3" : "none";
    v.failures.clear();
    bool short_ok = check_short_distribution(v.failures);
    if (!matched_b3) {
      v.failures.push_back("short prompt_0 greedy match < 16/16");
      short_ok = false;
    }
    v.short_prompts_passed = short_ok;
    v.long_code_passed = check_long(v.failures);
    v.chunked_vs_stepwise_passed = check_chunk(v.failures);
    v.executed_checks_passed = v.short_prompts_passed && v.long_code_passed && v.chunked_vs_stepwise_passed;
    if (in.diagnostic_mode) v.failures.push_back("diagnostic_mode forbids Gate 2 qualification");
    if (!in.qwen27b_pins_verified) v.failures.push_back("qwen27b_pins_verified is false");
    if (!in.short_ref_sha_verified) v.failures.push_back("short reference SHA-256 mismatch");
    if (!in.long_ref_sha_verified) v.failures.push_back("long-code reference SHA-256 mismatch");
    v.qualification_passed = !in.diagnostic_mode && in.qwen27b_pins_verified && in.short_ref_sha_verified &&
                             in.long_ref_sha_verified && v.executed_checks_passed;
    return v;
  }

  v.contract_version = std::string(kGate2ContractVersion);
  std::vector<std::string> b1_report_errs;
  const bool b1_dist_ok = check_b1_distribution(b1_report_errs);
  v.canonical_b1_branch_passed = addendum_pins_ok && matched_b1 && b1_dist_ok;

  // Always retain the 141 release-b3 distribution rows and bounds unconditionally.
  bool short_ok = check_short_distribution(v.failures);

  if (matched_b3) {
    v.chosen_prompt0_realization = "release_b3";
  } else if (matched_b1) {
    v.chosen_prompt0_realization = "canonical_b1";
    if (!addendum_pins_ok) short_ok = false;
    if (!check_b1_distribution(v.failures)) short_ok = false;
  } else {
    v.chosen_prompt0_realization = "none";
    v.failures.push_back("short prompt_0 greedy did not match intact 16/16 trajectory of release-b3 or canonical-b1");
    short_ok = false;
  }

  v.short_prompts_passed = short_ok;
  v.executed_checks_passed = v.short_prompts_passed && v.long_code_passed && v.chunked_vs_stepwise_passed;

  if (in.diagnostic_mode) v.failures.push_back("diagnostic_mode forbids Gate 2 qualification");
  if (!in.qwen27b_pins_verified) v.failures.push_back("qwen27b_pins_verified is false");
  if (!in.short_ref_sha_verified) v.failures.push_back("short reference SHA-256 mismatch");
  if (!in.short_ref_meta_sha_verified) v.failures.push_back("short reference metadata SHA-256 mismatch");
  if (!in.alt_short_ref_b1_sha_verified) v.failures.push_back("canonical-b1 short reference SHA-256 mismatch");
  if (!in.alt_short_ref_b1_meta_sha_verified) {
    v.failures.push_back("canonical-b1 short reference metadata SHA-256 mismatch");
  }
  if (!in.alt_short_ref_b1_prompt0_ids_match) v.failures.push_back("canonical-b1 prompt_0 input IDs mismatch");
  if (!in.long_ref_sha_verified) v.failures.push_back("long-code reference SHA-256 mismatch");

  v.qualification_passed = !in.diagnostic_mode && in.qwen27b_pins_verified && addendum_pins_ok &&
                           in.long_ref_sha_verified && v.executed_checks_passed;
  return v;
}

Qwen38RefSafetensors::Qwen38RefSafetensors(const std::filesystem::path& path) : path_(path) {
  fd_ = ::open(path_.c_str(), O_RDONLY);
  if (fd_ < 0) {
    throw std::invalid_argument("cannot open reference safetensors " + path_.string() + ": " + std::strerror(errno));
  }
  struct stat st {};
  if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 8) {
    close_fd();
    throw std::invalid_argument("invalid safetensors file: " + path_.string());
  }
  file_size_ = static_cast<std::uint64_t>(st.st_size);

  std::array<std::uint8_t, 8> hdr_len_bytes{};
  pread_all(fd_, hdr_len_bytes.data(), 8, 0, path_);
  std::uint64_t hdr_len = 0;
  for (int i = 0; i < 8; ++i) hdr_len |= (static_cast<std::uint64_t>(hdr_len_bytes[i]) << (8 * i));
  if (hdr_len == 0 || hdr_len > (64ull * 1024ull * 1024ull) || 8ull + hdr_len > file_size_) {
    close_fd();
    throw std::invalid_argument("invalid safetensors header length in " + path_.string());
  }
  const std::uint64_t data_base = 8ull + hdr_len;

  std::string hdr_str(static_cast<std::size_t>(hdr_len), '\0');
  pread_all(fd_, hdr_str.data(), hdr_str.size(), 8, path_);
  const auto hdr_json = nlohmann::json::parse(hdr_str);
  if (!hdr_json.is_object()) {
    close_fd();
    throw std::invalid_argument("safetensors header is not a JSON object: " + path_.string());
  }

  for (auto it = hdr_json.begin(); it != hdr_json.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const auto& obj = it.value();
    if (!obj.is_object() || !obj.contains("dtype") || !obj.contains("shape") || !obj.contains("data_offsets")) {
      close_fd();
      throw std::invalid_argument("malformed tensor entry '" + it.key() + "' in " + path_.string());
    }
    const auto& offsets = obj.at("data_offsets");
    if (!offsets.is_array() || offsets.size() != 2) {
      close_fd();
      throw std::invalid_argument("malformed data_offsets for '" + it.key() + "' in " + path_.string());
    }
    const std::uint64_t rel_b = offsets.at(0).get<std::uint64_t>();
    const std::uint64_t rel_e = offsets.at(1).get<std::uint64_t>();
    if (rel_e < rel_b || data_base + rel_e > file_size_) {
      close_fd();
      throw std::invalid_argument("out-of-bounds data_offsets for '" + it.key() + "' in " + path_.string());
    }
    Qwen38RefTensorEntry e;
    e.name = it.key();
    e.dtype = obj.at("dtype").get<std::string>();
    e.shape = obj.at("shape").get<std::vector<std::int64_t>>();
    e.abs_offset = data_base + rel_b;
    e.byte_size = rel_e - rel_b;
    entries_.emplace(e.name, std::move(e));
  }
}

Qwen38RefSafetensors::~Qwen38RefSafetensors() { close_fd(); }

Qwen38RefSafetensors::Qwen38RefSafetensors(Qwen38RefSafetensors&& other) noexcept
    : path_(std::move(other.path_)),
      fd_(std::exchange(other.fd_, -1)),
      file_size_(std::exchange(other.file_size_, 0)),
      entries_(std::move(other.entries_)),
      cached_sha256_(std::move(other.cached_sha256_)) {}

Qwen38RefSafetensors& Qwen38RefSafetensors::operator=(Qwen38RefSafetensors&& other) noexcept {
  if (this != &other) {
    close_fd();
    path_ = std::move(other.path_);
    fd_ = std::exchange(other.fd_, -1);
    file_size_ = std::exchange(other.file_size_, 0);
    entries_ = std::move(other.entries_);
    cached_sha256_ = std::move(other.cached_sha256_);
  }
  return *this;
}

void Qwen38RefSafetensors::close_fd() noexcept {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

const std::string& Qwen38RefSafetensors::sha256_hex() const {
  if (cached_sha256_.empty()) {
    cached_sha256_ = sha256_file_hex_streaming(path_);
  }
  return cached_sha256_;
}

bool Qwen38RefSafetensors::verify_sha256(std::string_view expected_hex) const {
  return lower_ascii(sha256_hex()) == lower_ascii(expected_hex);
}

const Qwen38RefTensorEntry* Qwen38RefSafetensors::find(std::string_view name) const {
  const auto it = entries_.find(std::string(name));
  return it == entries_.end() ? nullptr : &it->second;
}

const Qwen38RefTensorEntry& Qwen38RefSafetensors::at(std::string_view name) const {
  const auto* e = find(name);
  if (!e) throw std::out_of_range("reference safetensors missing tensor: " + std::string(name));
  return *e;
}

std::vector<std::int64_t> Qwen38RefSafetensors::read_i64(std::string_view name) const {
  const auto& e = at(name);
  if (e.dtype != "I64") {
    throw std::invalid_argument("tensor '" + e.name + "' expected I64, got " + e.dtype);
  }
  if (e.byte_size % sizeof(std::int64_t) != 0) {
    throw std::invalid_argument("tensor '" + e.name + "' byte_size not multiple of 8");
  }
  std::vector<std::int64_t> out(e.byte_size / sizeof(std::int64_t));
  if (!out.empty()) pread_all(fd_, out.data(), e.byte_size, e.abs_offset, path_);
  return out;
}

void Qwen38RefSafetensors::read_f32_row(std::string_view name, std::size_t row_idx, std::span<float> out_row) const {
  const auto& e = at(name);
  if (e.dtype != "F32") {
    throw std::invalid_argument("tensor '" + e.name + "' expected F32, got " + e.dtype);
  }
  if (e.shape.size() != 2 || e.shape[0] <= 0 || e.shape[1] <= 0) {
    throw std::invalid_argument("tensor '" + e.name + "' must be non-empty 2D");
  }
  const auto rows = static_cast<std::size_t>(e.shape[0]);
  const auto cols = static_cast<std::size_t>(e.shape[1]);
  if (row_idx >= rows || out_row.size() != cols) {
    throw std::invalid_argument("read_f32_row bounds/size mismatch for '" + e.name + "'");
  }
  const std::uint64_t row_bytes = static_cast<std::uint64_t>(cols) * sizeof(float);
  if (static_cast<std::uint64_t>(rows) * row_bytes != e.byte_size) {
    throw std::invalid_argument("tensor '" + e.name + "' byte_size mismatch with shape");
  }
  pread_all(fd_, out_row.data(), static_cast<std::size_t>(row_bytes),
            e.abs_offset + static_cast<std::uint64_t>(row_idx) * row_bytes, path_);
}

}  // namespace ingine::backends::qwen38

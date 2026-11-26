#include "backends/qwen38/qwen38_dense_gpu.h"
#include "backends/qwen38/qwen38_gate2_metrics.h"
#include "backends/qwen38/qwen38_gate3_metrics.h"
#include "backends/qwen38/qwen38_manifest.h"
#include "ingine/model_backend.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

#define CUDA_OK(expr)                                                                       \
  do {                                                                                      \
    cudaError_t err_ = (expr);                                                              \
    if (err_ != cudaSuccess) {                                                              \
      std::fprintf(stderr, "%s:%d: CUDA error %s: %s\n", __FILE__, __LINE__, #expr,         \
                   cudaGetErrorString(err_));                                               \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

namespace {

using namespace ingine::backends::qwen38;
using namespace ingine::model;

volatile std::sig_atomic_t g_stop_signal = 0;

void handle_stop_signal(int sig) { g_stop_signal = sig; }

void install_stop_signal_handlers() {
  struct sigaction sa {};
  sa.sa_handler = handle_stop_signal;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  ::sigaction(SIGTERM, &sa, nullptr);
  ::sigaction(SIGINT, &sa, nullptr);
}

bool write_json_atomic(const std::string& path_str, const nlohmann::ordered_json& doc) {
  if (path_str.empty()) return true;
  const std::string payload = doc.dump(2) + "\n";
  const std::filesystem::path target(path_str);
  const std::filesystem::path parent = target.parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
  }
  const std::string tmp_str = path_str + ".tmp";
  int fd = ::open(tmp_str.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) return false;
  std::size_t off = 0;
  while (off < payload.size()) {
    const ssize_t w = ::write(fd, payload.data() + off, payload.size() - off);
    if (w < 0) {
      if (errno == EINTR) continue;
      ::close(fd);
      std::error_code ec;
      std::filesystem::remove(tmp_str, ec);
      return false;
    }
    off += static_cast<std::size_t>(w);
  }
  (void)::fsync(fd);
  ::close(fd);
  std::error_code ec;
  std::filesystem::rename(tmp_str, target, ec);
  if (ec) {
    std::filesystem::remove(tmp_str, ec);
    return false;
  }
  return true;
}

std::string format_cuda_uuid(const cudaUUID_t& u) {
  char buf[64]{};
  const unsigned char* b = reinterpret_cast<const unsigned char*>(u.bytes);
  std::snprintf(buf, sizeof(buf),
                "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14],
                b[15]);
  return std::string(buf);
}

std::string digest_to_hex(const std::array<std::uint8_t, 32>& d) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out(64, '0');
  for (std::size_t i = 0; i < 32; ++i) {
    out[2 * i] = kHex[(d[i] >> 4) & 0xf];
    out[2 * i + 1] = kHex[d[i] & 0xf];
  }
  return out;
}

std::uint64_t process_max_rss_bytes() {
  struct rusage ru {};
  if (::getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_maxrss > 0) {
    return static_cast<std::uint64_t>(ru.ru_maxrss) * 1024ull;
  }
  return 0ull;
}

std::vector<std::int32_t> json_prompt_ids(const nlohmann::json& j, int vocab_size) {
  if (!j.is_array() || j.empty()) throw std::runtime_error("prompt_ids JSON array missing or empty");
  std::vector<std::int32_t> out;
  out.reserve(j.size());
  for (const auto& item : j) {
    const std::int64_t v = item.get<std::int64_t>();
    if (v < 0 || v >= static_cast<std::int64_t>(vocab_size)) {
      throw std::runtime_error("prompt_ids token out of vocab range");
    }
    out.push_back(static_cast<std::int32_t>(v));
  }
  return out;
}

std::string read_text_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + p.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct CapturedRun {
  GenerateStats stats;
  std::vector<std::int32_t> greedy_ids;
  std::vector<int> step_positions;
  std::vector<std::vector<float>> step_logits;
  double elapsed_ms = 0.0;
};

CapturedRun run_session_request(Qwen38DenseBackend& backend, Qwen38DenseSession& sess, const BackendRequest& req,
                                int steps) {
  CapturedRun out;
  Sampling samp;
  samp.temperature = 0.0f;
  samp.top_k = 1;
  out.step_positions.reserve(static_cast<std::size_t>(steps));
  out.step_logits.reserve(static_cast<std::size_t>(steps));
  out.greedy_ids.reserve(static_cast<std::size_t>(steps));
  const auto t0 = std::chrono::steady_clock::now();
  out.stats = backend.generate_request_on_session(
      sess, req, steps, samp, {},
      [&](std::span<const std::int32_t> toks) {
        out.greedy_ids.insert(out.greedy_ids.end(), toks.begin(), toks.end());
        return true;
      },
      {},
      [&](const LogitStep& st) {
        out.step_positions.push_back(st.position);
        out.step_logits.emplace_back(st.logits.begin(), st.logits.end());
      });
  out.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

std::vector<Gate3StepComparison> compare_runs(const CapturedRun& cold, const CapturedRun& warm, std::string& note) {
  if (cold.step_positions.size() != cold.step_logits.size() || warm.step_positions.size() != warm.step_logits.size()) {
    if (!note.empty()) note += "; ";
    note += "step_positions count does not match step_logits count";
  }
  if (cold.step_logits.size() != warm.step_logits.size()) {
    if (!note.empty()) note += "; ";
    note += "cold step_logits count (" + std::to_string(cold.step_logits.size()) + ") != warm step_logits count (" +
            std::to_string(warm.step_logits.size()) + ")";
  }
  std::vector<Gate3StepComparison> cmps;
  const std::size_t n = std::min({cold.step_logits.size(), warm.step_logits.size(), cold.step_positions.size(),
                                  warm.step_positions.size()});
  cmps.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    cmps.push_back(
        compare_logits_bit_exact(cold.step_logits[i], warm.step_logits[i], cold.step_positions[i], warm.step_positions[i]));
  }
  return cmps;
}

Gate3ReuseCaseResult assemble_reuse_case(std::string name, Reuse expected_reuse, std::size_t expected_reused,
                                         std::size_t prompt_len, const CapturedRun& cold, const CapturedRun& warm,
                                         std::string note = {}) {
  Gate3ReuseCaseResult r;
  r.name = std::move(name);
  r.executed = true;
  r.expected_reuse = expected_reuse;
  r.actual_reuse = warm.stats.reuse;
  r.expected_reused_tokens = expected_reused;
  r.actual_reused_tokens = static_cast<std::size_t>(warm.stats.reused_tokens);
  r.prompt_len = prompt_len;
  r.cold_step_count = cold.step_logits.size();
  r.warm_step_count = warm.step_logits.size();
  r.warm_prefill_seconds = warm.stats.prefill_seconds;
  r.restored_bytes = warm.stats.restored_bytes.value_or(0ull);
  r.restore_seconds = warm.stats.restore_seconds.value_or(0.0);
  r.cold_elapsed_ms = cold.elapsed_ms;
  r.warm_elapsed_ms = warm.elapsed_ms;
  r.cold_greedy_ids = cold.greedy_ids;
  r.warm_greedy_ids = warm.greedy_ids;
  r.failure_note = std::move(note);
  r.step_comparisons = compare_runs(cold, warm, r.failure_note);
  return r;
}

std::uint64_t sample_vram_used_bytes() {
  std::size_t free_b = 0;
  std::size_t total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess || total_b < free_b) return 0ull;
  return static_cast<std::uint64_t>(total_b - free_b);
}

nlohmann::ordered_json reuse_case_to_json(const Gate3ReuseCaseResult& rc) {
  nlohmann::ordered_json steps = nlohmann::json::array();
  bool all_memcmp = !rc.step_comparisons.empty();
  bool all_finite = !rc.step_comparisons.empty();
  float max_abs = 0.0f;
  for (const auto& sc : rc.step_comparisons) {
    all_memcmp = all_memcmp && sc.memcmp_equal;
    all_finite = all_finite && sc.all_finite;
    if (std::isfinite(sc.max_abs_diff)) {
      max_abs = std::max(max_abs, sc.max_abs_diff);
    } else {
      max_abs = std::numeric_limits<float>::infinity();
    }
    steps.push_back({
        {"position", sc.position},
        {"cold_position", sc.cold_position},
        {"warm_position", sc.warm_position},
        {"vocab_size", sc.vocab_size},
        {"cold_vocab_size", sc.cold_vocab_size},
        {"warm_vocab_size", sc.warm_vocab_size},
        {"all_finite", sc.all_finite},
        {"memcmp_equal", sc.memcmp_equal},
        {"argmax_match", sc.argmax_match},
        {"cold_argmax", sc.cold_argmax},
        {"warm_argmax", sc.warm_argmax},
        {"first_diff_vocab_idx", sc.first_diff_vocab_idx},
        {"max_abs_diff", sc.max_abs_diff},
    });
  }
  return {
      {"name", rc.name},
      {"executed", rc.executed},
      {"expected_reuse", to_string(rc.expected_reuse)},
      {"actual_reuse", to_string(rc.actual_reuse)},
      {"expected_reused_tokens", rc.expected_reused_tokens},
      {"actual_reused_tokens", rc.actual_reused_tokens},
      {"prompt_len", rc.prompt_len},
      {"cold_step_count", rc.cold_step_count},
      {"warm_step_count", rc.warm_step_count},
      {"warm_prefill_seconds", rc.warm_prefill_seconds},
      {"restored_bytes", rc.restored_bytes},
      {"restore_seconds", rc.restore_seconds},
      {"cold_elapsed_ms", rc.cold_elapsed_ms},
      {"warm_elapsed_ms", rc.warm_elapsed_ms},
      {"steps_compared", rc.step_comparisons.size()},
      {"all_steps_finite", all_finite},
      {"all_steps_memcmp_equal", all_memcmp},
      {"max_abs_diff", max_abs},
      {"cold_greedy_ids", rc.cold_greedy_ids},
      {"warm_greedy_ids", rc.warm_greedy_ids},
      {"step_comparisons", std::move(steps)},
      {"failure_note", rc.failure_note},
  };
}

int main_impl(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::setvbuf(stderr, nullptr, _IOLBF, 0);

  std::string q27_dir = "/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f";
  std::string ref_dir = "/mnt/storage/flare/release-references";
  std::string json_out;
  int prefill_chunk = 256;
  int max_ctx = 2048;
  bool verify_pins = true;
  bool diagnostic_mode = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--qwen27-dir") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing argument for --qwen27-dir\n");
        return 2;
      }
      q27_dir = argv[++i];
    } else if (a == "--ref-dir") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing argument for --ref-dir\n");
        return 2;
      }
      ref_dir = argv[++i];
    } else if (a == "--prefill-chunk") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing argument for --prefill-chunk\n");
        return 2;
      }
      prefill_chunk = std::atoi(argv[++i]);
      if (prefill_chunk <= 0) {
        std::fprintf(stderr, "--prefill-chunk must be > 0\n");
        return 2;
      }
    } else if (a == "--max-ctx") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing argument for --max-ctx\n");
        return 2;
      }
      max_ctx = std::atoi(argv[++i]);
      if (max_ctx < 512) {
        std::fprintf(stderr, "--max-ctx must be >= 512\n");
        return 2;
      }
    } else if (a == "--verify-pins") {
      verify_pins = true;
    } else if (a == "--no-verify-pins") {
      verify_pins = false;
    } else if (a == "--diagnostic") {
      diagnostic_mode = true;
    } else if (a == "--json-out") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing argument for --json-out\n");
        return 2;
      }
      json_out = argv[++i];
    } else {
      std::fprintf(stderr, "unknown option: %s\n", a.c_str());
      return 2;
    }
  }

  if (!diagnostic_mode) {
    if (!verify_pins) {
      std::fprintf(stderr, "--no-verify-pins requires --diagnostic (qualification mode fails closed)\n");
      return 2;
    }
    if (prefill_chunk != 256) {
      std::fprintf(stderr, "qualification mode requires --prefill-chunk 256 (pass --diagnostic to override)\n");
      return 2;
    }
  }

  int dev_count = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
  if (dev_err != cudaSuccess || dev_count <= 0) {
    std::fprintf(stderr, "qwen38_gpu_gate3: no CUDA device found, skipping (rc=77)\n");
    return 77;
  }

  install_stop_signal_handlers();

  cudaDeviceProp props{};
  CUDA_OK(cudaGetDeviceProperties(&props, 0));
  const std::string gpu_uuid = format_cuda_uuid(props.uuid);
  std::printf("Gate 3 running [%s] on %s (%s, sm_%d%d, vram=%.2f GiB, verify_pins=%s, chunk=%d, max_ctx=%d)\n",
              diagnostic_mode ? "DIAGNOSTIC" : "QUALIFICATION", props.name, gpu_uuid.c_str(), props.major, props.minor,
              static_cast<double>(props.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
              verify_pins ? "true" : "false", prefill_chunk, max_ctx);

  const auto wall_t0 = std::chrono::steady_clock::now();
  std::optional<Qwen38GemmParityReport> gemm_report;
  std::optional<Qwen38DenseLoadStats> load_stats;
  std::optional<ResidencyReport> residency;
  std::optional<BackendStateIdentity> state_id;
  bool q27_pins_verified_actual = false;
  bool short_ref_json_sha_verified = false;
  bool long_ref_json_sha_verified = false;
  std::string short_ref_json_sha_actual;
  std::string long_ref_json_sha_actual;

  double gemm_self_check_ms = 0.0;
  double ref_sha_verify_ms = 0.0;
  double reuse_cases_ms = 0.0;
  double protected_misses_ms = 0.0;
  double warm_prefix_lookup_ms = 0.0;
  double warm_reset_ms = 0.0;
  std::uint64_t load_only_peak_vram_bytes = 0;
  std::uint64_t two_session_simultaneous_vram_bytes = 0;
  std::uint64_t two_session_state_device_bytes = 0;
  std::uint64_t two_session_residency_device_bytes = 0;
  std::uint64_t max_runtime_vram_bytes = 0;

  std::vector<Gate3ReuseCaseResult> reuse_cases;
  std::vector<Gate3ProtectedMissResult> protected_misses;

  auto build_json = [&](bool in_progress, const std::string& current_phase, bool terminated_early, int stop_sig,
                        const std::string& error_message, const Qwen38Gate3Verdict* final_verdict) {
    const std::uint64_t rss_bytes = process_max_rss_bytes();
    const double total_wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_t0).count();

    nlohmann::ordered_json root;
    root["device"] = props.name;
    root["gpu_uuid"] = gpu_uuid;
    root["pci_bus_id"] =
        std::to_string(props.pciDomainID) + ":" + std::to_string(props.pciBusID) + ":" + std::to_string(props.pciDeviceID);
    root["sm"] = std::to_string(props.major) + std::to_string(props.minor);
    root["total_vram_bytes"] = static_cast<std::uint64_t>(props.totalGlobalMem);
    root["mode"] = diagnostic_mode ? "diagnostic" : "qualification";
    root["in_progress"] = in_progress;
    root["current_phase"] = current_phase;
    root["terminated_early"] = terminated_early;
    if (stop_sig != 0) root["signal"] = stop_sig;
    if (!error_message.empty()) root["error"] = error_message;

    root["qwen27b_dir"] = q27_dir;
    root["ref_dir"] = ref_dir;
    root["prefill_chunk"] = prefill_chunk;
    root["max_ctx"] = max_ctx;
    root["verify_pins_requested"] = verify_pins;
    root["qwen27b_pins_verified"] = q27_pins_verified_actual;
    root["short_ref_json_sha256_expected"] = std::string(kGate3ShortRefJsonSha256);
    root["short_ref_json_sha256_actual"] = short_ref_json_sha_actual;
    root["short_ref_json_sha_verified"] = short_ref_json_sha_verified;
    root["long_ref_json_sha256_expected"] = std::string(kGate3LongRefJsonSha256);
    root["long_ref_json_sha256_actual"] = long_ref_json_sha_actual;
    root["long_ref_json_sha_verified"] = long_ref_json_sha_verified;
    root["gate3_contract_version"] = std::string(kGate3ContractVersion);

    if (state_id.has_value()) {
      root["checkpoint_identity"] = {
          {"verification_level", to_string(state_id->checkpoint.verification_level)},
          {"shard_digests_verified", state_id->checkpoint.shard_digests_verified},
          {"weight_manifest_verified", state_id->checkpoint.weight_manifest_verified},
          {"shard_count", state_id->checkpoint.shard_count},
          {"shard_set_sha256", digest_to_hex(state_id->checkpoint.shard_set_sha256)},
          {"config_sha256", digest_to_hex(state_id->checkpoint.config_sha256)},
          {"tokenizer_sha256", digest_to_hex(state_id->checkpoint.tokenizer_sha256)},
          {"chat_template_sha256", digest_to_hex(state_id->checkpoint.chat_template_sha256)},
          {"weight_manifest_sha256", digest_to_hex(state_id->checkpoint.weight_manifest_sha256)},
          {"numeric_abi", state_id->numeric_abi},
          {"kv_format", state_id->kv_format},
          {"family_state_fingerprint", state_id->family_state_fingerprint},
      };
    }

    if (gemm_report.has_value()) {
      root["gemm_parity_self_check"] = {
          {"nvfp4_max_abs", gemm_report->nvfp4_max_abs},
          {"fp8_cal_max_abs", gemm_report->fp8_cal_max_abs},
          {"bf16_max_abs", gemm_report->bf16_max_abs},
          {"batch_invariance_max_abs", gemm_report->batch_invariance_max_abs},
          {"gdn_prefill_vs_step_max_abs", gemm_report->gdn_prefill_vs_step_max_abs},
          {"gdn_batch_invariance_max_abs", gemm_report->gdn_batch_invariance_max_abs},
          {"passed", gemm_report->passed},
      };
    }

    if (load_stats.has_value()) {
      root["load_stats"] = {
          {"cold_manifest_sha_ms", load_stats->cold_manifest_sha_ms},
          {"cold_safetensors_load_ms", load_stats->cold_safetensors_load_ms},
          {"weights_device_bytes", load_stats->weights_device_bytes},
          {"state_device_bytes", load_stats->state_device_bytes},
          {"scratch_device_bytes", load_stats->scratch_device_bytes},
          {"load_only_peak_vram_bytes", load_stats->peak_vram_bytes},
          {"peak_vram_bytes", load_stats->peak_vram_bytes},
          {"host_rss_bytes", std::max(load_stats->host_rss_bytes, rss_bytes)},
          {"pins_verified", load_stats->pins_verified},
      };
    }
    if (residency.has_value()) {
      root["residency_report"] = {
          {"weights_device_bytes", residency->weights_device_bytes},
          {"state_device_bytes", residency->state_device_bytes},
          {"device_bytes", residency->device_bytes},
          {"host_ram_bytes", residency->host_ram_bytes},
          {"process_rss_bytes", std::max(residency->process_rss_bytes, rss_bytes)},
      };
    }
    root["host_rss_bytes"] = rss_bytes;
    root["load_only_peak_vram_bytes"] = load_only_peak_vram_bytes;
    root["two_session_simultaneous_vram_bytes"] = two_session_simultaneous_vram_bytes;
    root["two_session_state_device_bytes"] = two_session_state_device_bytes;
    root["two_session_residency_device_bytes"] = two_session_residency_device_bytes;
    root["peak_vram_bytes"] =
        std::max({load_only_peak_vram_bytes, two_session_simultaneous_vram_bytes, max_runtime_vram_bytes});

    root["timings_ms"] = {
        {"gemm_self_check_ms", gemm_self_check_ms},
        {"ref_sha_verify_ms", ref_sha_verify_ms},
        {"cold_manifest_sha_ms", load_stats.has_value() ? load_stats->cold_manifest_sha_ms : 0.0},
        {"cold_safetensors_load_ms", load_stats.has_value() ? load_stats->cold_safetensors_load_ms : 0.0},
        {"reuse_cases_ms", reuse_cases_ms},
        {"protected_misses_ms", protected_misses_ms},
        {"warm_prefix_cache_lookup_ms", warm_prefix_lookup_ms},
        {"warm_reset_ms", warm_reset_ms},
        {"total_wall_ms", total_wall_ms},
    };

    nlohmann::ordered_json rc_arr = nlohmann::json::array();
    for (const auto& rc : reuse_cases) rc_arr.push_back(reuse_case_to_json(rc));
    root["reuse_cases"] = std::move(rc_arr);

    nlohmann::ordered_json pm_arr = nlohmann::json::array();
    for (const auto& pm : protected_misses) {
      pm_arr.push_back({
          {"name", pm.name},
          {"executed", pm.executed},
          {"hit", pm.hit},
          {"reuse", to_string(pm.reuse)},
          {"reused_tokens", pm.reused_tokens},
          {"reason", pm.reason},
          {"cold_position", pm.cold_position},
          {"warm_position", pm.warm_position},
          {"post_miss_position", pm.post_miss_position},
          {"post_miss_vocab_size", pm.post_miss_vocab_size},
          {"cold_argmax", pm.cold_argmax},
          {"warm_argmax", pm.warm_argmax},
          {"cold_greedy_id", pm.cold_greedy_id},
          {"warm_greedy_id", pm.warm_greedy_id},
          {"post_miss_bit_exact", pm.post_miss_bit_exact},
          {"max_abs_diff", pm.max_abs_diff},
          {"failure_note", pm.failure_note},
      });
    }
    root["protected_misses"] = std::move(pm_arr);

    if (final_verdict) {
      root["reuse_cases_passed"] = final_verdict->reuse_cases_passed;
      root["protected_misses_passed"] = final_verdict->protected_misses_passed;
      root["executed_checks_passed"] = final_verdict->executed_checks_passed;
      root["qualification_passed"] = final_verdict->qualification_passed;
      root["all_passed"] = final_verdict->qualification_passed;
      root["coverage_status"] =
          terminated_early
              ? "incomplete_terminated"
              : (in_progress ? "incomplete_partial"
                             : (final_verdict->qualification_passed
                                    ? "complete_qualified"
                                    : (diagnostic_mode && final_verdict->executed_checks_passed
                                           ? "diagnostic_executed_pass"
                                           : "failed")));
      root["failures"] = final_verdict->failures;
    } else {
      root["reuse_cases_passed"] = false;
      root["protected_misses_passed"] = false;
      root["executed_checks_passed"] = false;
      root["qualification_passed"] = false;
      root["all_passed"] = false;
      root["coverage_status"] = terminated_early ? "incomplete_terminated" : "incomplete_partial";
      root["failures"] = nlohmann::json::array();
    }
    return root;
  };

  auto checkpoint_json = [&](const std::string& phase) {
    if (json_out.empty()) return;
    (void)write_json_atomic(json_out, build_json(true, phase, false, 0, "", nullptr));
  };

  auto check_interrupted = [&](const std::string& phase) {
    if (g_stop_signal != 0) {
      const int sig = static_cast<int>(g_stop_signal);
      if (!json_out.empty()) {
        (void)write_json_atomic(json_out, build_json(false, phase, true, sig, "interrupted by signal", nullptr));
      }
      std::fprintf(stderr, "qwen38_gpu_gate3: terminated early at phase=%s signal=%d\n", phase.c_str(), sig);
      std::exit(128 + sig);
    }
  };

  try {
    // Phase 1: Self-check
    std::printf("[phase] start=gemm_parity_self_check\n");
    checkpoint_json("gemm_parity_self_check");
    const auto tg0 = std::chrono::steady_clock::now();
    gemm_report = Qwen38DenseBackend::self_check_gemm_parity(0);
    gemm_self_check_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tg0).count();
    std::printf("  [gemm_parity_self_check] batch_inv=%g gdn_vs_step=%g gdn_batch_inv=%g passed=%s elapsed_ms=%.1f\n",
                gemm_report->batch_invariance_max_abs, gemm_report->gdn_prefill_vs_step_max_abs,
                gemm_report->gdn_batch_invariance_max_abs, gemm_report->passed ? "true" : "false", gemm_self_check_ms);
    check_interrupted("gemm_parity_self_check");

    // Phase 2: Verify reference JSON hashes and load pinned prompts
    std::printf("[phase] start=verify_references\n");
    checkpoint_json("verify_references");
    const auto tr0 = std::chrono::steady_clock::now();
    const std::filesystem::path short_json_path = std::filesystem::path(ref_dir) / "reference.json";
    const std::filesystem::path long_json_path = std::filesystem::path(ref_dir) / "long-code-reference.json";
    short_ref_json_sha_verified = verify_file_sha256(short_json_path, kGate3ShortRefJsonSha256, &short_ref_json_sha_actual);
    long_ref_json_sha_verified = verify_file_sha256(long_json_path, kGate3LongRefJsonSha256, &long_ref_json_sha_actual);
    ref_sha_verify_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr0).count();
    std::printf("  [verify_references] short_json_sha=%s long_json_sha=%s elapsed_ms=%.1f\n",
                short_ref_json_sha_verified ? "OK" : "FAIL", long_ref_json_sha_verified ? "OK" : "FAIL",
                ref_sha_verify_ms);
    if (!diagnostic_mode && (!short_ref_json_sha_verified || !long_ref_json_sha_verified)) {
      throw std::runtime_error("reference JSON SHA-256 verification failed");
    }

    const nlohmann::json short_doc = nlohmann::json::parse(read_text_file(short_json_path));
    const nlohmann::json long_doc = nlohmann::json::parse(read_text_file(long_json_path));
    check_interrupted("verify_references");

    // Phase 3: Load pinned 27B backend
    std::printf("[phase] start=load_qwen27b verify_pins=%s\n", verify_pins ? "true" : "false");
    checkpoint_json("load_qwen27b");
    Qwen38DenseLoadOptions opts;
    opts.model_dir = q27_dir;
    opts.max_ctx = max_ctx;
    opts.prefill_chunk = prefill_chunk;
    opts.verify_pins = verify_pins;
    opts.use_calibrated_fp8 = true;
    opts.device_id = 0;

    Qwen38DenseBackend backend(opts);
    load_stats = backend.load_stats();
    load_only_peak_vram_bytes = load_stats->peak_vram_bytes;
    max_runtime_vram_bytes = std::max(load_only_peak_vram_bytes, sample_vram_used_bytes());
    residency = backend.residency_report();
    state_id = backend.state_identity();
    q27_pins_verified_actual = load_stats->pins_verified;
    const int vocab_size = backend.config().vocab_size;

    std::printf("  [load_qwen27b] pins=%s sha_ms=%.1f load_ms=%.1f load_vram=%.1f MiB\n",
                q27_pins_verified_actual ? "verified" : "UNVERIFIED", load_stats->cold_manifest_sha_ms,
                load_stats->cold_safetensors_load_ms,
                static_cast<double>(load_stats->peak_vram_bytes) / (1024.0 * 1024.0));
    if (!diagnostic_mode && !q27_pins_verified_actual) {
      throw std::runtime_error("Qwen3.8-27B pins were not verified");
    }
    check_interrupted("load_qwen27b");

    // Extract pinned real prompt slices
    const auto p0 = json_prompt_ids(short_doc.at("prompts").at(0).at("prompt_ids"), vocab_size);
    const auto p1 = json_prompt_ids(short_doc.at("prompts").at(1).at("prompt_ids"), vocab_size);
    const auto p2 = json_prompt_ids(short_doc.at("prompts").at(2).at("prompt_ids"), vocab_size);
    const auto l0 = json_prompt_ids(long_doc.at("prompts").at(0).at("prompt_ids"), vocab_size);
    const auto l1 = json_prompt_ids(long_doc.at("prompts").at(1).at("prompt_ids"), vocab_size);
    if (p0.size() < 32 || p1.size() < 12 || p2.size() < 12 || l0.size() < 608 || l1.size() < 288) {
      throw std::runtime_error("unexpected short/long reference prompt lengths");
    }

    const std::vector<std::int32_t> p0_20(p0.begin(), p0.begin() + 20);
    std::vector<std::int32_t> fork_20 = p0_20;
    {
      const std::int32_t f20_div = (p2[0] != p0[20]) ? p2[0] : ((p0[20] + 1) % vocab_size);
      fork_20.push_back(f20_div);
      fork_20.insert(fork_20.end(), p2.begin() + 1, p2.begin() + 12);
    }
    // Construct a fork diverging strictly at position 26 while checkpoints exist only at 20 and 32.
    std::vector<std::int32_t> fork_26(p0.begin(), p0.begin() + 26);
    {
      const std::int32_t div_tok = (p1[0] != p0[26]) ? p1[0] : ((p0[26] + 1) % vocab_size);
      fork_26.push_back(div_tok);
      fork_26.insert(fork_26.end(), p1.begin() + 1, p1.begin() + 6);
    }

    // Multi-chunk prompts crossing prefill_chunk=256 (320 prefix + 288 suffix -> len 608).
    const std::vector<std::int32_t> code_320(l0.begin(), l0.begin() + 320);
    const std::vector<std::int32_t> code_ext_608(l0.begin(), l0.begin() + 608);

    auto* sess0 = dynamic_cast<Qwen38DenseSession*>(backend.acquire_session(0));
    if (!sess0) throw std::runtime_error("acquire_session(0) returned null");

    auto make_req = [](std::span<const std::int32_t> ids, bool use_cache, std::span<const std::uint8_t> mask = {},
                       std::string ns = {}) {
      BackendRequest r;
      r.prompt.ids = ids;
      r.token_mask = mask;
      r.use_prefix_cache = use_cache;
      r.cache_namespace = std::move(ns);
      return r;
    };

    // Determine decode-divergent fork_32 guaranteed distinct from p0's first generated token.
    backend.reset_session();
    const auto seed_p0 = run_session_request(backend, *sess0, make_req(p0, true), 8);
    std::vector<std::int32_t> fork_32 = p0;
    {
      std::int32_t div_32 = p1[0];
      if (!seed_p0.greedy_ids.empty() && div_32 == seed_p0.greedy_ids[0]) {
        div_32 = (div_32 + 1) % vocab_size;
      }
      fork_32.push_back(div_32);
      fork_32.insert(fork_32.end(), p1.begin() + 1, p1.begin() + 12);
    }

    // Determine multi-chunk checkpoint fork guaranteed distinct from both l0[320] and code_320 decode step 0.
    backend.reset_session();
    const auto seed_code = run_session_request(backend, *sess0, make_req(code_320, true), 4);
    std::vector<std::int32_t> code_fork_608 = code_320;
    {
      std::int32_t div_mc = l1[0];
      while (div_mc == l0[320] || (!seed_code.greedy_ids.empty() && div_mc == seed_code.greedy_ids[0])) {
        div_mc = (div_mc + 1) % vocab_size;
      }
      code_fork_608.push_back(div_mc);
      code_fork_608.insert(code_fork_608.end(), l1.begin() + 1, l1.begin() + 288);
    }

    // Phase 4: Execute all 9 Reuse & Isolation Cases
    std::printf("[phase] start=reuse_and_isolation\n");
    checkpoint_json("reuse_and_isolation");
    const auto tru0 = std::chrono::steady_clock::now();

    // Case 1: extend_suffix (20 -> 32, 8 resumed steps)
    {
      backend.reset_session();
      const auto cold = run_session_request(backend, *sess0, make_req(p0, false), 8);
      backend.reset_session();
      (void)run_session_request(backend, *sess0, make_req(p0_20, true), 1);
      const auto warm = run_session_request(backend, *sess0, make_req(p0, true), 8);
      warm_prefix_lookup_ms = std::max(warm_prefix_lookup_ms, backend.last_lookup_ms());
      reuse_cases.push_back(assemble_reuse_case("extend_suffix", Reuse::Extend, 20, p0.size(), cold, warm));
    }
    check_interrupted("reuse_and_isolation");

    // Case 2: extend_exact_tip (32 == tip via teacher-forced empty callback, 0 suffix prefill, 8 resumed steps)
    {
      backend.reset_session();
      const auto cold = run_session_request(backend, *sess0, make_req(p0, false), 8);
      backend.reset_session();
      backend.evaluate_teacher_forced_on_session(*sess0, make_req(p0, true), std::array<int, 1>{31}, {});
      std::string tip_note;
      if (sess0->committed_tokens().size() != p0.size() || !sess0->checkpoint_index().has_tip_logits() ||
          sess0->checkpoint_index().tip_logits_pos() != static_cast<int>(p0.size()) - 1) {
        tip_note = "exact-tip seed did not populate committed tip_logits";
      }
      const auto warm = run_session_request(backend, *sess0, make_req(p0, true), 8);
      reuse_cases.push_back(
          assemble_reuse_case("extend_exact_tip", Reuse::Extend, 32, p0.size(), cold, warm, std::move(tip_note)));
    }
    check_interrupted("reuse_and_isolation");

    // Case 3: checkpoint_after_decode_fork (prefill 32 + decode 8 -> fork to 44 via ckpt 32 + 8 steps)
    {
      backend.reset_session();
      const auto cold = run_session_request(backend, *sess0, make_req(fork_32, false), 8);
      backend.reset_session();
      (void)run_session_request(backend, *sess0, make_req(p0, true), 8);
      const auto warm = run_session_request(backend, *sess0, make_req(fork_32, true), 8);
      reuse_cases.push_back(
          assemble_reuse_case("checkpoint_after_decode_fork", Reuse::Checkpoint, 32, fork_32.size(), cold, warm));
    }
    check_interrupted("reuse_and_isolation");

    // Case 4: checkpoint_multiturn_rollback (ckpt 20 -> extend 32 + decode -> fork rollback to ckpt 20 + 8 steps)
    {
      backend.reset_session();
      const auto cold = run_session_request(backend, *sess0, make_req(fork_20, false), 8);
      backend.reset_session();
      (void)run_session_request(backend, *sess0, make_req(p0_20, true), 1);
      (void)run_session_request(backend, *sess0, make_req(p0, true), 4);
      const auto warm = run_session_request(backend, *sess0, make_req(fork_20, true), 8);
      reuse_cases.push_back(
          assemble_reuse_case("checkpoint_multiturn_rollback", Reuse::Checkpoint, 20, fork_20.size(), cold, warm));
    }
    check_interrupted("reuse_and_isolation");

    // Case 5: checkpoint_midfork_earlier_ckpt (ckpts {20, 32}, common prefix 26 -> restores to 20, not 26)
    {
      backend.reset_session();
      const auto cold = run_session_request(backend, *sess0, make_req(fork_26, false), 8);
      backend.reset_session();
      (void)run_session_request(backend, *sess0, make_req(p0_20, true), 1);
      (void)run_session_request(backend, *sess0, make_req(p0, true), 2);
      const auto warm = run_session_request(backend, *sess0, make_req(fork_26, true), 8);
      reuse_cases.push_back(
          assemble_reuse_case("checkpoint_midfork_earlier_ckpt", Reuse::Checkpoint, 20, fork_26.size(), cold, warm));
    }
    check_interrupted("reuse_and_isolation");

    // Case 6: multichunk_checkpoint_fork (ckpt @ 320 -> decode -> fork 288 suffix crossing prefill_chunk=256 -> len 608 + 8 steps)
    {
      backend.reset_session();
      const auto cold = run_session_request(backend, *sess0, make_req(code_fork_608, false), 8);
      backend.reset_session();
      (void)run_session_request(backend, *sess0, make_req(code_320, true), 4);
      const auto warm = run_session_request(backend, *sess0, make_req(code_fork_608, true), 8);
      reuse_cases.push_back(
          assemble_reuse_case("multichunk_checkpoint_fork", Reuse::Checkpoint, 320, code_fork_608.size(), cold, warm));
    }
    check_interrupted("reuse_and_isolation");

    // Case 7: multichunk_extend_suffix (tip @ 320 -> extend 288 suffix crossing prefill_chunk=256 -> len 608 + 8 steps)
    {
      backend.reset_session();
      const auto cold = run_session_request(backend, *sess0, make_req(code_ext_608, false), 8);
      backend.reset_session();
      (void)run_session_request(backend, *sess0, make_req(code_320, true), 1);
      const auto warm = run_session_request(backend, *sess0, make_req(code_ext_608, true), 8);
      reuse_cases.push_back(
          assemble_reuse_case("multichunk_extend_suffix", Reuse::Extend, 320, code_ext_608.size(), cold, warm));
    }
    check_interrupted("reuse_and_isolation");

    // Case 8: session_reset_isolation
    {
      backend.reset_session();
      const auto cold = run_session_request(backend, *sess0, make_req(p0, false), 8);
      backend.reset_session();
      (void)run_session_request(backend, *sess0, make_req(p2, true), 4);
      backend.reset_session();
      warm_reset_ms = std::max(warm_reset_ms, backend.last_reset_ms());
      const auto warm = run_session_request(backend, *sess0, make_req(p0, true), 8);
      reuse_cases.push_back(assemble_reuse_case("session_reset_isolation", Reuse::None, 0, p0.size(), cold, warm));
    }
    check_interrupted("reuse_and_isolation");

    // Case 9: multi_session_isolation (standalone Session B live simultaneously; does not contaminate Session A)
    {
      backend.reset_session();
      const auto cold = run_session_request(backend, *sess0, make_req(p0, false), 8);
      {
        auto owned_b = backend.create_session();
        auto* sess_b = dynamic_cast<Qwen38DenseSession*>(owned_b.get());
        if (!sess_b) throw std::runtime_error("create_session() did not return Qwen38DenseSession");
        backend.reset_session();
        (void)run_session_request(backend, *sess0, make_req(p0_20, true), 1);
        (void)run_session_request(backend, *sess_b, make_req(p1, true), 4);
        two_session_simultaneous_vram_bytes = sample_vram_used_bytes();
        max_runtime_vram_bytes = std::max(max_runtime_vram_bytes, two_session_simultaneous_vram_bytes);
        two_session_state_device_bytes = sess0->state_bytes() + sess_b->state_bytes();
        two_session_residency_device_bytes =
            load_stats->weights_device_bytes + two_session_state_device_bytes + load_stats->scratch_device_bytes;
        sess_b->reset();
        const auto warm = run_session_request(backend, *sess0, make_req(p0, true), 8);
        reuse_cases.push_back(assemble_reuse_case("multi_session_isolation", Reuse::Extend, 20, p0.size(), cold, warm));
      }
    }
    max_runtime_vram_bytes = std::max(max_runtime_vram_bytes, sample_vram_used_bytes());
    reuse_cases_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tru0).count();
    for (const auto& rc : reuse_cases) {
      std::size_t eq_cnt = 0;
      for (const auto& sc : rc.step_comparisons) eq_cnt += (sc.memcmp_equal && sc.all_finite) ? 1u : 0u;
      std::printf("  [reuse:%s] expected=%s(%zu) actual=%s(%zu) memcmp=%zu/%zu greedy=%s\n", rc.name.c_str(),
                  to_string(rc.expected_reuse), rc.expected_reused_tokens, to_string(rc.actual_reuse),
                  rc.actual_reused_tokens, eq_cnt, rc.step_comparisons.size(),
                  (rc.cold_greedy_ids == rc.warm_greedy_ids) ? "EXACT" : "DIFF");
    }
    check_interrupted("reuse_and_isolation");

    // Phase 5: Execute all 14 Protected Miss Controls
    std::printf("[phase] start=protected_misses\n");
    checkpoint_json("protected_misses");
    const auto tpm0 = std::chrono::steady_clock::now();

    backend.reset_session();
    const auto cold_p0 = run_session_request(backend, *sess0, make_req(p0, false), 1);

    auto run_e2e_miss = [&](std::string name, auto setup_fn, const BackendRequest& miss_req,
                            const CapturedRun& cold_ref) {
      setup_fn();
      const auto warm = run_session_request(backend, *sess0, miss_req, 1);
      const int cpos = cold_ref.step_positions.empty() ? -1 : cold_ref.step_positions[0];
      const int wpos = warm.step_positions.empty() ? -1 : warm.step_positions[0];
      const auto cmp = compare_logits_bit_exact(cold_ref.step_logits[0], warm.step_logits[0], cpos, wpos);
      Gate3ProtectedMissResult pm;
      pm.name = std::move(name);
      pm.executed = true;
      pm.hit = (warm.stats.reuse != Reuse::None);
      pm.reuse = warm.stats.reuse;
      pm.reused_tokens = static_cast<std::size_t>(warm.stats.reused_tokens);
      pm.reason = backend.last_lookup().reason;
      pm.cold_position = cmp.cold_position;
      pm.warm_position = cmp.warm_position;
      pm.post_miss_position = cmp.position;
      pm.post_miss_vocab_size = cmp.vocab_size;
      pm.cold_argmax = cmp.cold_argmax;
      pm.warm_argmax = cmp.warm_argmax;
      pm.cold_greedy_id = cold_ref.greedy_ids.empty() ? -1 : cold_ref.greedy_ids[0];
      pm.warm_greedy_id = warm.greedy_ids.empty() ? -1 : warm.greedy_ids[0];
      pm.post_miss_bit_exact = cmp.all_finite && cmp.memcmp_equal && cmp.argmax_match &&
                               (pm.cold_position == pm.warm_position) && (pm.cold_greedy_id == pm.warm_greedy_id);
      pm.max_abs_diff = cmp.max_abs_diff;
      protected_misses.push_back(std::move(pm));
    };

    // 1. miss_explicit_bypass
    run_e2e_miss(
        "miss_explicit_bypass",
        [&] {
          backend.reset_session();
          (void)run_session_request(backend, *sess0, make_req(p0_20, true), 1);
        },
        make_req(p0, false), cold_p0);

    // 2. miss_empty_common_prefix
    {
      std::vector<std::int32_t> diff0 = p1;
      if (diff0[0] == p0[0]) diff0[0] = (p0[0] + 1) % vocab_size;
      run_e2e_miss(
          "miss_empty_common_prefix",
          [&] {
            backend.reset_session();
            (void)run_session_request(backend, *sess0, make_req(diff0, true), 1);
          },
          make_req(p0, true), cold_p0);
    }

    // 3. miss_cache_namespace_mismatch
    {
      backend.reset_session();
      const auto cold_nsb = run_session_request(backend, *sess0, make_req(p0, false, {}, "ns-b"), 1);
      run_e2e_miss(
          "miss_cache_namespace_mismatch",
          [&] {
            backend.reset_session();
            (void)run_session_request(backend, *sess0, make_req(p0_20, true, {}, "ns-a"), 1);
          },
          make_req(p0, true, {}, "ns-b"), cold_nsb);
    }

    // 4. miss_token_mask_presence_mismatch
    {
      const std::vector<std::uint8_t> mask_all1(p0.size(), 1u);
      run_e2e_miss(
          "miss_token_mask_presence_mismatch",
          [&] {
            backend.reset_session();
            (void)run_session_request(backend, *sess0, make_req(p0_20, true), 1);
          },
          make_req(p0, true, mask_all1), cold_p0);
    }

    // 5. miss_token_mask_bit_flip_at_pos0
    {
      const std::vector<std::uint8_t> mask_cached(p0_20.size(), 1u);
      std::vector<std::uint8_t> mask_flip0(p0.size(), 1u);
      mask_flip0[0] = 0u;
      backend.reset_session();
      const auto cold_flip = run_session_request(backend, *sess0, make_req(p0, false, mask_flip0), 1);
      run_e2e_miss(
          "miss_token_mask_bit_flip_at_pos0",
          [&] {
            backend.reset_session();
            (void)run_session_request(backend, *sess0, make_req(p0_20, true, mask_cached), 1);
          },
          make_req(p0, true, mask_flip0), cold_flip);
    }

    // 6. miss_uncheckpointed_short_fork (ckpt only @ 32, fork @ 10 -> no ckpt @ 10 -> cold fallback)
    {
      std::vector<std::int32_t> short_fork(p0.begin(), p0.begin() + 10);
      short_fork.insert(short_fork.end(), p1.begin(), p1.begin() + 10);
      if (short_fork[10] == p0[10]) short_fork[10] = (p0[10] + 1) % vocab_size;
      backend.reset_session();
      const auto cold_sf = run_session_request(backend, *sess0, make_req(short_fork, false), 1);
      run_e2e_miss(
          "miss_uncheckpointed_short_fork",
          [&] {
            backend.reset_session();
            (void)run_session_request(backend, *sess0, make_req(p0, true), 1);
          },
          make_req(short_fork, true), cold_sf);
    }

    // 7. miss_evicted_checkpoint_fork (ckpt @ 32 evicted via evict_caches() while live tip is @ 35)
    {
      backend.reset_session();
      const auto cold_fk = run_session_request(backend, *sess0, make_req(fork_32, false), 1);
      std::uint64_t evicted_bytes = 0;
      run_e2e_miss(
          "miss_evicted_checkpoint_fork",
          [&] {
            backend.reset_session();
            (void)run_session_request(backend, *sess0, make_req(p0, true), 4);
            evicted_bytes = backend.evict_caches(1ull << 60);
          },
          make_req(fork_32, true), cold_fk);
      if (evicted_bytes == 0) {
        protected_misses.back().failure_note = "evict_caches returned 0 bytes";
      }
    }

    // 8..14: Identity/verification/ABI/media mutation controls against warm pinned session
    backend.reset_session();
    (void)run_session_request(backend, *sess0, make_req(p0_20, true), 1);
    const BackendStateIdentity live_id = sess0->identity();
    auto run_identity_miss = [&](std::string name, BackendStateIdentity qid) {
      const auto lk = restore_session_prefix(*sess0, SessionLookupQuery{qid, p0, {}, {}, true});
      const auto post_warm = run_session_request(backend, *sess0, make_req(p0, false), 1);
      const int cpos = cold_p0.step_positions.empty() ? -1 : cold_p0.step_positions[0];
      const int wpos = post_warm.step_positions.empty() ? -1 : post_warm.step_positions[0];
      const auto cmp = compare_logits_bit_exact(cold_p0.step_logits[0], post_warm.step_logits[0], cpos, wpos);
      Gate3ProtectedMissResult pm;
      pm.name = std::move(name);
      pm.executed = true;
      pm.hit = lk.hit;
      pm.reuse = lk.reuse;
      pm.reused_tokens = lk.reused_tokens;
      pm.reason = lk.reason;
      pm.cold_position = cmp.cold_position;
      pm.warm_position = cmp.warm_position;
      pm.post_miss_position = cmp.position;
      pm.post_miss_vocab_size = cmp.vocab_size;
      pm.cold_argmax = cmp.cold_argmax;
      pm.warm_argmax = cmp.warm_argmax;
      pm.cold_greedy_id = cold_p0.greedy_ids.empty() ? -1 : cold_p0.greedy_ids[0];
      pm.warm_greedy_id = post_warm.greedy_ids.empty() ? -1 : post_warm.greedy_ids[0];
      pm.post_miss_bit_exact = cmp.all_finite && cmp.memcmp_equal && cmp.argmax_match &&
                               (pm.cold_position == pm.warm_position) && (pm.cold_greedy_id == pm.warm_greedy_id);
      pm.max_abs_diff = cmp.max_abs_diff;
      protected_misses.push_back(std::move(pm));
      // Re-warm session for next identity check.
      backend.reset_session();
      (void)run_session_request(backend, *sess0, make_req(p0_20, true), 1);
    };

    {
      auto id = live_id;
      id.checkpoint.verification_level = VerificationLevel::Unverified;
      id.checkpoint.shard_digests_verified = false;
      run_identity_miss("miss_unverified_identity", id);
    }
    {
      auto id = live_id;
      id.checkpoint.verification_level = VerificationLevel::HeadersOnly;
      id.checkpoint.shard_digests_verified = false;
      run_identity_miss("miss_headers_only_identity", id);
    }
    {
      auto id = live_id;
      id.checkpoint.verification_level = VerificationLevel::SelfHashed;
      run_identity_miss("miss_self_hashed_identity", id);
    }
    {
      auto id = live_id;
      id.checkpoint.shard_set_sha256[0] ^= 0xffu;
      run_identity_miss("miss_mutated_shard_set_sha256", id);
    }
    {
      auto id = live_id;
      id.checkpoint.weight_manifest_sha256[0] ^= 0xffu;
      run_identity_miss("miss_mutated_weight_manifest_sha256", id);
    }
    {
      auto id = live_id;
      id.numeric_abi += ";tamper";
      run_identity_miss("miss_mutated_numeric_abi", id);
    }
    {
      auto id = live_id;
      id.media_sha256[0] ^= 0xffu;
      run_identity_miss("miss_mutated_media_sha256", id);
    }

    protected_misses_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tpm0).count();
    for (const auto& pm : protected_misses) {
      std::printf("  [miss:%s] hit=%s reused=%zu reason=%s bit_exact=%s\n", pm.name.c_str(),
                  pm.hit ? "TRUE" : "false", pm.reused_tokens, pm.reason.c_str(),
                  pm.post_miss_bit_exact ? "true" : "false");
    }

    residency = backend.residency_report();

    // Final Verdict
    Qwen38Gate3Input gi;
    gi.diagnostic_mode = diagnostic_mode;
    gi.qwen27b_pins_verified = q27_pins_verified_actual;
    gi.short_ref_json_sha_verified = short_ref_json_sha_verified;
    gi.long_ref_json_sha_verified = long_ref_json_sha_verified;
    gi.self_check_passed = gemm_report.has_value() && gemm_report->passed;
    gi.reuse_cases = reuse_cases;
    gi.protected_misses = protected_misses;

    const Qwen38Gate3Verdict verdict = evaluate_gate3_verdict(gi);
    const nlohmann::ordered_json final_doc = build_json(false, "complete", false, 0, "", &verdict);
    if (!json_out.empty() && !write_json_atomic(json_out, final_doc)) {
      std::fprintf(stderr, "failed to write final atomic json output path: %s\n", json_out.c_str());
      return 1;
    }

    const bool ok = diagnostic_mode ? verdict.executed_checks_passed : verdict.qualification_passed;
    std::printf(
        "qwen38_gpu_gate3: %s | mode=%s reuse=%s(%zu/9) misses=%s(%zu/14) pins=%s load_vram=%.1f MiB two_sess_vram=%.1f MiB wall_ms=%.1f\n",
        verdict.qualification_passed ? "PASS" : (diagnostic_mode && verdict.executed_checks_passed ? "DIAG_PASS" : "FAIL"),
        diagnostic_mode ? "diagnostic" : "qualification", verdict.reuse_cases_passed ? "PASS" : "FAIL",
        reuse_cases.size(), verdict.protected_misses_passed ? "PASS" : "FAIL", protected_misses.size(),
        q27_pins_verified_actual ? "verified" : "unverified",
        static_cast<double>(load_only_peak_vram_bytes) / (1024.0 * 1024.0),
        static_cast<double>(two_session_simultaneous_vram_bytes) / (1024.0 * 1024.0),
        final_doc["timings_ms"]["total_wall_ms"].get<double>());
    if (!verdict.failures.empty()) {
      for (const auto& f : verdict.failures) std::fprintf(stderr, "  failure: %s\n", f.c_str());
    }
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    if (!json_out.empty()) {
      (void)write_json_atomic(json_out, build_json(false, "error", false, 0, e.what(), nullptr));
    }
    std::fprintf(stderr, "qwen38_gpu_gate3: ERROR: %s\n", e.what());
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) { return main_impl(argc, argv); }

#include "backends/qwen38/qwen38_dense_gpu.h"
#include "backends/qwen38/qwen38_gate2_metrics.h"
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
#include <optional>
#include <span>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <unordered_map>
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

std::vector<std::int32_t> i64_to_i32(std::span<const std::int64_t> src, int vocab_size) {
  std::vector<std::int32_t> out(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    if (src[i] < 0 || src[i] >= static_cast<std::int64_t>(vocab_size)) {
      throw std::runtime_error("reference token id out of vocabulary range");
    }
    out[i] = static_cast<std::int32_t>(src[i]);
  }
  return out;
}

nlohmann::ordered_json kld_summary_to_json(const Qwen38KldSummary& s) {
  return {
      {"positions", s.positions},
      {"kld_mean", s.kld_mean},
      {"kld_p99", s.kld_p99},
      {"kld_max", s.kld_max},
      {"top1_matches", s.top1_matches},
      {"top1_agreement", s.top1_agreement},
      {"all_finite", s.all_finite},
  };
}

struct SplitLogitTop2 {
  bool valid = false;
  Qwen38Top2Logits top2;
  float logit_303 = 0.0f;
  float logit_321 = 0.0f;
};

SplitLogitTop2 extract_split_top2(std::span<const float> logits) {
  SplitLogitTop2 out;
  out.top2 = compute_top2_logits(logits);
  if (!out.top2.finite || logits.size() <= 321u) return out;
  out.logit_303 = logits[303];
  out.logit_321 = logits[321];
  out.valid = std::isfinite(out.logit_303) && std::isfinite(out.logit_321);
  return out;
}

nlohmann::ordered_json split_top2_to_json(const SplitLogitTop2& s) {
  if (!s.valid) return {{"valid", false}};
  return {
      {"valid", true},
      {"top1_id", s.top2.top1_id},
      {"top1_logit", s.top2.top1_logit},
      {"top2_id", s.top2.top2_id},
      {"top2_logit", s.top2.top2_logit},
      {"logit_303", s.logit_303},
      {"logit_321", s.logit_321},
      {"diff_303_minus_321", s.logit_303 - s.logit_321},
  };
}

struct ShortPromptRecord {
  std::string name;
  std::size_t prompt_len = 0;
  std::size_t positions_evaluated = 0;
  Qwen38KldSummary summary;
  std::size_t greedy_matches = 0;
  std::size_t greedy_total = 0;
  std::vector<std::int32_t> ref_generated_ids;
  std::vector<std::int32_t> cand_greedy_ids;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double greedy_ms = 0.0;
};

struct ChunkVsStepRecord {
  std::string name;
  Qwen38LogitParityPoint final_logit;
  std::size_t greedy_matches = 0;
  std::size_t greedy_total = 0;
  double chunk_prefill_ms = 0.0;
  double step_prefill_ms = 0.0;
  std::vector<std::int32_t> chunk_greedy_ids;
  std::vector<std::int32_t> step_greedy_ids;
};

struct LongPromptRecord {
  std::string name;
  std::size_t seq_len = 0;
  std::size_t positions_evaluated = 0;
  Qwen38KldSummary summary;
  double prefill_ms = 0.0;
  double prefill_tok_s = 0.0;
};

struct Gate2TimingsMs {
  double gemm_self_check_ms = 0.0;
  double ref_sha_verify_ms = 0.0;
  double cold_manifest_sha_ms = 0.0;
  double cold_safetensors_load_ms = 0.0;
  double short_prompts_ms = 0.0;
  double chunk_vs_step_ms = 0.0;
  double long_code_ms = 0.0;
  double warm_prefix_cache_lookup_ms = 0.0;
  double warm_reset_ms = 0.0;
  double smoke_08b_ms = 0.0;
  double total_wall_ms = 0.0;
};

int main_impl(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::setvbuf(stderr, nullptr, _IOLBF, 0);

  std::string q27_dir = "/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f";
  std::string ref_dir = "/mnt/storage/flare/release-references";
  std::string alt_short_ref_dir = "/mnt/storage/flare/john-repro-b1";
  std::string q08_dir = "/mnt/storage/models/Qwen3.5-0.8B";
  std::string json_out;
  int prefill_chunk = 256;
  bool verify_pins = true;  // Fail-closed qualification default.
  bool diagnostic_mode = false;
  bool smoke_08b = false;

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
    } else if (a == "--alt-short-ref-dir") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing argument for --alt-short-ref-dir\n");
        return 2;
      }
      alt_short_ref_dir = argv[++i];
    } else if (a == "--qwen08-dir") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing argument for --qwen08-dir\n");
        return 2;
      }
      q08_dir = argv[++i];
    } else if (a == "--prefill-chunk") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing argument for --prefill-chunk\n");
        return 2;
      }
      char* endp = nullptr;
      const long parsed = std::strtol(argv[++i], &endp, 10);
      if (endp == nullptr || *endp != '\0' || parsed <= 0 || parsed > 4096) {
        std::fprintf(stderr, "invalid --prefill-chunk value\n");
        return 2;
      }
      prefill_chunk = static_cast<int>(parsed);
    } else if (a == "--json-out") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing argument for --json-out\n");
        return 2;
      }
      json_out = argv[++i];
    } else if (a == "--verify-pins") {
      verify_pins = true;
    } else if (a == "--no-verify-pins") {
      verify_pins = false;
    } else if (a == "--diagnostic") {
      diagnostic_mode = true;
    } else if (a == "--smoke-08b") {
      smoke_08b = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }

  if (!diagnostic_mode && (smoke_08b || !verify_pins)) {
    std::fprintf(stderr,
                 "error: --smoke-08b and --no-verify-pins require --diagnostic and are forbidden in "
                 "Gate 2 qualification mode\n");
    return 2;
  }

  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
    std::printf("skip: no CUDA device visible\n");
    return 77;
  }

  install_stop_signal_handlers();
  const auto t_program_start = std::chrono::steady_clock::now();
  auto wall_ms = [&]() -> double {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_program_start).count();
  };

  CUDA_OK(cudaSetDevice(0));
  cudaDeviceProp props{};
  CUDA_OK(cudaGetDeviceProperties(&props, 0));
  const std::string gpu_uuid = format_cuda_uuid(props.uuid);
  const double total_vram_gib = static_cast<double>(props.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0);

  std::printf("Gate 2 running [%s] on %s (%s, sm_%d%d, vram=%.2f GiB, verify_pins=%s, chunk=%d)\n",
              smoke_08b ? "DIAGNOSTIC_SMOKE_08B" : (diagnostic_mode ? "DIAGNOSTIC" : "QUALIFICATION"),
              props.name, gpu_uuid.c_str(), props.major, props.minor, total_vram_gib,
              verify_pins ? "true" : "false", prefill_chunk);

  Gate2TimingsMs timings;
  std::optional<Qwen38GemmParityReport> gemm_report;
  std::optional<Qwen38DenseLoadStats> load_stats;
  std::optional<ingine::model::ResidencyReport> residency;
  std::optional<ingine::model::BackendStateIdentity> state_id;

  bool q27_pins_verified_actual = false;
  bool short_ref_sha_verified = false;
  bool short_ref_meta_sha_verified = false;
  bool alt_short_ref_b1_sha_verified = false;
  bool alt_short_ref_b1_meta_sha_verified = false;
  bool alt_short_ref_b1_prompt0_ids_match = false;
  bool long_ref_sha_verified = false;
  std::string short_ref_sha_actual;
  std::string short_ref_meta_sha_actual;
  std::string alt_short_ref_b1_sha_actual;
  std::string alt_short_ref_b1_meta_sha_actual;
  std::string long_ref_sha_actual;
  nlohmann::ordered_json rel_b3_meta_summary;
  nlohmann::ordered_json can_b1_meta_summary;

  std::vector<ShortPromptRecord> short_records;
  std::vector<ChunkVsStepRecord> chunk_step_records;
  std::vector<LongPromptRecord> long_records;
  std::vector<Qwen38LogitParityPoint> short_points;
  std::vector<Qwen38LogitParityPoint> chunk_step_points;
  std::vector<Qwen38LogitParityPoint> long_points;

  std::vector<std::int32_t> p0_alt_b1_ref_ids;
  std::optional<Qwen38Prompt0TrajectoryEval> p0_traj_eval;
  SplitLogitTop2 p0_cand_dec0_split_top2;
  SplitLogitTop2 p0_rel_b3_dec0_split_top2;
  SplitLogitTop2 p0_can_b1_dec0_split_top2;
  bool p0_b1_eval_executed = false;
  std::vector<Qwen38LogitParityPoint> p0_b1_points;

  bool prefix_lookup_hit = false;
  std::size_t prefix_lookup_reused_tokens = 0;

  // 0.8B smoke state
  bool smoke08_executed = false;
  bool smoke08_passed = false;
  std::optional<Qwen38LogitParityPoint> smoke08_chunk_vs_step_logit;
  std::size_t smoke08_greedy_matches = 0;
  std::size_t smoke08_greedy_total = 0;
  std::vector<std::int32_t> smoke08_chunk_greedy;
  std::vector<std::int32_t> smoke08_step_greedy;

  std::string error_message;

  auto build_json = [&](bool in_progress, const std::string& current_phase, bool terminated_early, int stop_sig,
                        Qwen38Gate2Verdict* out_verdict = nullptr) {
    timings.total_wall_ms = wall_ms();
    const std::uint64_t rss_bytes = process_max_rss_bytes();

    nlohmann::ordered_json root;
    root["device"] = props.name;
    root["gpu_uuid"] = gpu_uuid;
    root["pci_bus_id"] =
        std::to_string(props.pciDomainID) + ":" + std::to_string(props.pciBusID) + ":" + std::to_string(props.pciDeviceID);
    root["sm"] = std::to_string(props.major) + std::to_string(props.minor);
    root["total_vram_bytes"] = static_cast<std::uint64_t>(props.totalGlobalMem);
    root["mode"] = smoke_08b ? "diagnostic_smoke_08b" : (diagnostic_mode ? "diagnostic" : "qualification");
    root["in_progress"] = in_progress;
    root["current_phase"] = current_phase;
    root["terminated_early"] = terminated_early;
    if (stop_sig != 0) root["signal"] = stop_sig;
    if (!error_message.empty()) root["error"] = error_message;

    root["qwen27b_dir"] = q27_dir;
    root["qwen08b_dir"] = q08_dir;
    root["ref_dir"] = ref_dir;
    root["alt_short_ref_dir"] = alt_short_ref_dir;
    root["prefill_chunk"] = prefill_chunk;
    root["verify_pins_requested"] = verify_pins;
    root["qwen27b_pins_verified"] = q27_pins_verified_actual;
    root["short_ref_sha256_expected"] = std::string(kGate2ShortRefSha256);
    root["short_ref_sha256_actual"] = short_ref_sha_actual;
    root["short_ref_sha_verified"] = short_ref_sha_verified;
    root["short_ref_meta_sha_verified"] = short_ref_meta_sha_verified;
    root["alt_short_ref_b1_sha_verified"] = alt_short_ref_b1_sha_verified;
    root["alt_short_ref_b1_meta_sha_verified"] = alt_short_ref_b1_meta_sha_verified;
    root["alt_short_ref_b1_prompt0_ids_match"] = alt_short_ref_b1_prompt0_ids_match;
    root["long_ref_sha256_expected"] = std::string(kGate2LongCodeRefSha256);
    root["long_ref_sha256_actual"] = long_ref_sha_actual;
    root["long_ref_sha_verified"] = long_ref_sha_verified;
    root["reference_contract"] = {
        {"version", std::string(kGate2ContractVersion)},
        {"numeric_mode", std::string(kGate2ExpectedNumericMode)},
        {"release_b3_sha256_expected", std::string(kGate2ShortRefSha256)},
        {"release_b3_sha256_actual", short_ref_sha_actual},
        {"release_b3_sha_verified", short_ref_sha_verified},
        {"release_b3_meta_sha256_expected", std::string(kGate2ShortRefMetaSha256)},
        {"release_b3_meta_sha256_actual", short_ref_meta_sha_actual},
        {"release_b3_meta_sha_verified", short_ref_meta_sha_verified},
        {"canonical_b1_sha256_expected", std::string(kGate2AltShortRefB1Sha256)},
        {"canonical_b1_sha256_actual", alt_short_ref_b1_sha_actual},
        {"canonical_b1_sha_verified", alt_short_ref_b1_sha_verified},
        {"canonical_b1_meta_sha256_expected", std::string(kGate2AltShortRefB1MetaSha256)},
        {"canonical_b1_meta_sha256_actual", alt_short_ref_b1_meta_sha_actual},
        {"canonical_b1_meta_sha_verified", alt_short_ref_b1_meta_sha_verified},
        {"prompt0_input_ids_identical", alt_short_ref_b1_prompt0_ids_match},
        {"release_b3_meta", rel_b3_meta_summary},
        {"canonical_b1_meta", can_b1_meta_summary},
    };

    if (state_id.has_value()) {
      root["checkpoint_identity"] = {
          {"verification_level", ingine::model::to_string(state_id->checkpoint.verification_level)},
          {"shard_digests_verified", state_id->checkpoint.shard_digests_verified},
          {"weight_manifest_verified", state_id->checkpoint.weight_manifest_verified},
          {"shard_count", state_id->checkpoint.shard_count},
          {"shard_set_sha256", digest_to_hex(state_id->checkpoint.shard_set_sha256)},
          {"config_sha256", digest_to_hex(state_id->checkpoint.config_sha256)},
          {"tokenizer_sha256", digest_to_hex(state_id->checkpoint.tokenizer_sha256)},
          {"chat_template_sha256", digest_to_hex(state_id->checkpoint.chat_template_sha256)},
          {"weight_manifest_sha256", digest_to_hex(state_id->checkpoint.weight_manifest_sha256)},
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
    root["peak_vram_bytes"] = load_stats.has_value() ? load_stats->peak_vram_bytes : 0ull;

    root["timings_ms"] = {
        {"gemm_self_check_ms", timings.gemm_self_check_ms},
        {"ref_sha_verify_ms", timings.ref_sha_verify_ms},
        {"cold_manifest_sha_ms", timings.cold_manifest_sha_ms},
        {"cold_safetensors_load_ms", timings.cold_safetensors_load_ms},
        {"short_prompts_ms", timings.short_prompts_ms},
        {"chunk_vs_step_ms", timings.chunk_vs_step_ms},
        {"long_code_ms", timings.long_code_ms},
        {"warm_prefix_cache_lookup_ms", timings.warm_prefix_cache_lookup_ms},
        {"warm_reset_ms", timings.warm_reset_ms},
        {"smoke_08b_ms", timings.smoke_08b_ms},
        {"total_wall_ms", timings.total_wall_ms},
    };

    if (smoke_08b) {
      const bool gemm_ok = gemm_report.has_value() && gemm_report->passed;
      const bool ok_exec = !in_progress && !terminated_early && error_message.empty() && gemm_ok && smoke08_executed &&
                           smoke08_passed;
      root["coverage_status"] =
          terminated_early ? "incomplete_terminated"
                           : (in_progress ? "incomplete_partial"
                                          : (ok_exec ? "diagnostic_smoke_08b_pass" : "diagnostic_smoke_08b_fail"));
      root["executed_checks_passed"] = ok_exec;
      root["qualification_passed"] = false;
      root["all_passed"] = false;
      nlohmann::ordered_json sm;
      sm["executed"] = smoke08_executed;
      sm["passed"] = smoke08_passed;
      sm["greedy_matches"] = smoke08_greedy_matches;
      sm["greedy_total"] = smoke08_greedy_total;
      sm["chunk_greedy_ids"] = smoke08_chunk_greedy;
      sm["step_greedy_ids"] = smoke08_step_greedy;
      if (smoke08_chunk_vs_step_logit.has_value()) {
        sm["final_logit_kld"] = smoke08_chunk_vs_step_logit->kld;
        sm["final_logit_top1_match"] = smoke08_chunk_vs_step_logit->top1_match;
        sm["final_logit_finite"] = smoke08_chunk_vs_step_logit->finite;
        sm["final_logit_max_abs"] = smoke08_chunk_vs_step_logit->max_abs_diff;
      }
      root["smoke_08b"] = std::move(sm);
      return root;
    }

    const Qwen38KldSummary short_sum = summarize_kld_metrics(short_points);
    const Qwen38KldSummary chunk_sum = summarize_kld_metrics(chunk_step_points);
    const Qwen38KldSummary long_sum = summarize_kld_metrics(long_points);
    const Qwen38KldSummary p0_b1_sum = p0_b1_points.empty() ? Qwen38KldSummary{} : summarize_kld_metrics(p0_b1_points);

    std::size_t p0_matches = 0;
    std::size_t p0_total = 0;
    if (!short_records.empty()) {
      p0_matches = short_records[0].greedy_matches;
      p0_total = short_records[0].greedy_total;
    }
    std::size_t cs_greedy_matches = 0;
    std::size_t cs_greedy_total = 0;
    for (const auto& cs : chunk_step_records) {
      cs_greedy_matches += cs.greedy_matches;
      cs_greedy_total += cs.greedy_total;
    }

    Qwen38Gate2Input gi;
    gi.diagnostic_mode = diagnostic_mode;
    gi.qwen27b_pins_verified = q27_pins_verified_actual;
    gi.short_ref_sha_verified = short_ref_sha_verified;
    gi.long_ref_sha_verified = long_ref_sha_verified;
    gi.allow_canonical_b1_addendum = true;
    gi.short_ref_meta_sha_verified = short_ref_meta_sha_verified;
    gi.alt_short_ref_b1_sha_verified = alt_short_ref_b1_sha_verified;
    gi.alt_short_ref_b1_meta_sha_verified = alt_short_ref_b1_meta_sha_verified;
    gi.alt_short_ref_b1_prompt0_ids_match = alt_short_ref_b1_prompt0_ids_match;
    gi.short_prompt_count = short_records.size();
    gi.prompt0_greedy_matches = p0_matches;
    gi.prompt0_greedy_total = p0_total;
    gi.short_summary = short_sum;
    gi.prompt0_b1_greedy_matches = p0_traj_eval.has_value() ? p0_traj_eval->canonical_b1_matches : 0u;
    gi.prompt0_b1_greedy_total = p0_alt_b1_ref_ids.size();
    gi.prompt0_b1_eval_executed = p0_b1_eval_executed;
    gi.prompt0_b1_summary = p0_b1_sum;
    gi.long_prompt_count = long_records.size();
    gi.long_summary = long_sum;
    gi.chunk_vs_step_prompt_count = chunk_step_records.size();
    gi.chunk_vs_step_summary = chunk_sum;
    gi.chunk_vs_step_greedy_matches = cs_greedy_matches;
    gi.chunk_vs_step_greedy_total = cs_greedy_total;

    Qwen38Gate2Verdict verdict = evaluate_gate2_verdict(gi);
    const bool gemm_ok = gemm_report.has_value() && gemm_report->passed;
    if (!gemm_ok) {
      verdict.executed_checks_passed = false;
      verdict.qualification_passed = false;
      verdict.release_only_qualification_passed = false;
      verdict.failures.push_back("self_check_gemm_parity failed or not executed");
      verdict.release_only_failures.push_back("self_check_gemm_parity failed or not executed");
    }
    if (!error_message.empty() || terminated_early || in_progress) {
      verdict.executed_checks_passed = false;
      verdict.qualification_passed = false;
      verdict.release_only_qualification_passed = false;
    }
    if (out_verdict) *out_verdict = verdict;

    const std::string cov_status =
        terminated_early
            ? "incomplete_terminated"
            : (in_progress
                   ? "incomplete_partial"
                   : (verdict.qualification_passed
                          ? "complete_qualified"
                          : (diagnostic_mode && verdict.executed_checks_passed ? "diagnostic_executed_pass" : "failed")));

    root["coverage_status"] = cov_status;
    root["gate2_contract_version"] = verdict.contract_version;
    root["chosen_prompt0_realization"] = verdict.chosen_prompt0_realization;
    root["canonical_b1_branch_passed"] = verdict.canonical_b1_branch_passed;
    root["release_only_short_passed"] = verdict.release_only_short_passed;
    root["release_only_qualification_passed"] = verdict.release_only_qualification_passed;
    root["release_only_failures"] = verdict.release_only_failures;
    root["short_prompts_passed"] = verdict.short_prompts_passed;
    root["long_code_passed"] = verdict.long_code_passed;
    root["chunked_vs_stepwise_passed"] = verdict.chunked_vs_stepwise_passed;
    root["executed_checks_passed"] = verdict.executed_checks_passed;
    root["qualification_passed"] = verdict.qualification_passed;
    root["all_passed"] = verdict.qualification_passed;
    root["failures"] = verdict.failures;

    root["prefix_cache_lookup"] = {
        {"hit", prefix_lookup_hit},
        {"reused_tokens", prefix_lookup_reused_tokens},
        {"warm_prefix_cache_lookup_ms", timings.warm_prefix_cache_lookup_ms},
    };

    root["short_summary"] = kld_summary_to_json(short_sum);
    root["short_summary"]["prompt0_greedy_matches"] = p0_matches;
    root["short_summary"]["prompt0_greedy_total"] = p0_total;
    root["short_summary"]["prompt0_b1_greedy_matches"] =
        p0_traj_eval.has_value() ? p0_traj_eval->canonical_b1_matches : 0u;
    root["short_summary"]["prompt0_b1_greedy_total"] = p0_alt_b1_ref_ids.size();
    root["prompt0_addendum"] = {
        {"contract_version", verdict.contract_version},
        {"chosen_realization", verdict.chosen_prompt0_realization},
        {"release_b3_generated_ids",
         short_records.empty() ? std::vector<std::int32_t>{} : short_records[0].ref_generated_ids},
        {"canonical_b1_generated_ids", p0_alt_b1_ref_ids},
        {"candidate_greedy_ids",
         short_records.empty() ? std::vector<std::int32_t>{} : short_records[0].cand_greedy_ids},
        {"release_b3_greedy_matches", p0_traj_eval.has_value() ? p0_traj_eval->release_b3_matches : 0u},
        {"release_b3_greedy_total", p0_total},
        {"release_b3_first_diff_idx", p0_traj_eval.has_value() ? p0_traj_eval->release_b3_first_diff_idx : -1},
        {"canonical_b1_greedy_matches", p0_traj_eval.has_value() ? p0_traj_eval->canonical_b1_matches : 0u},
        {"canonical_b1_greedy_total", p0_alt_b1_ref_ids.size()},
        {"canonical_b1_first_diff_idx", p0_traj_eval.has_value() ? p0_traj_eval->canonical_b1_first_diff_idx : -1},
        {"first_canonical_split_idx", 1},
        {"dec0_split_top2",
         {{"candidate", split_top2_to_json(p0_cand_dec0_split_top2)},
          {"release_b3", split_top2_to_json(p0_rel_b3_dec0_split_top2)},
          {"canonical_b1", split_top2_to_json(p0_can_b1_dec0_split_top2)}}},
        {"canonical_b1_eval_executed", p0_b1_eval_executed},
        {"canonical_b1_distribution_summary", kld_summary_to_json(p0_b1_sum)},
        {"canonical_b1_branch_passed", verdict.canonical_b1_branch_passed},
    };
    root["short_prompts"] = nlohmann::json::array();
    for (const auto& sr : short_records) {
      root["short_prompts"].push_back({
          {"name", sr.name},
          {"prompt_len", sr.prompt_len},
          {"positions", sr.positions_evaluated},
          {"summary", kld_summary_to_json(sr.summary)},
          {"greedy_matches", sr.greedy_matches},
          {"greedy_total", sr.greedy_total},
          {"ref_generated_ids", sr.ref_generated_ids},
          {"cand_greedy_ids", sr.cand_greedy_ids},
          {"prefill_ms", sr.prefill_ms},
          {"decode_ms", sr.decode_ms},
          {"greedy_ms", sr.greedy_ms},
      });
    }

    root["chunk_vs_step_summary"] = kld_summary_to_json(chunk_sum);
    root["chunk_vs_step_summary"]["greedy_matches"] = cs_greedy_matches;
    root["chunk_vs_step_summary"]["greedy_total"] = cs_greedy_total;
    root["chunk_vs_step_prompts"] = nlohmann::json::array();
    for (const auto& cs : chunk_step_records) {
      root["chunk_vs_step_prompts"].push_back({
          {"name", cs.name},
          {"kld", cs.final_logit.kld},
          {"top1_match", cs.final_logit.top1_match},
          {"finite", cs.final_logit.finite},
          {"max_abs_diff", cs.final_logit.max_abs_diff},
          {"greedy_matches", cs.greedy_matches},
          {"greedy_total", cs.greedy_total},
          {"chunk_prefill_ms", cs.chunk_prefill_ms},
          {"step_prefill_ms", cs.step_prefill_ms},
          {"chunk_greedy_ids", cs.chunk_greedy_ids},
          {"step_greedy_ids", cs.step_greedy_ids},
      });
    }

    root["long_summary"] = kld_summary_to_json(long_sum);
    root["long_prompts"] = nlohmann::json::array();
    for (const auto& lr : long_records) {
      root["long_prompts"].push_back({
          {"name", lr.name},
          {"seq_len", lr.seq_len},
          {"positions", lr.positions_evaluated},
          {"summary", kld_summary_to_json(lr.summary)},
          {"prefill_ms", lr.prefill_ms},
          {"prefill_tok_s", lr.prefill_tok_s},
      });
    }

    return root;
  };

  auto checkpoint_json = [&](const std::string& phase) {
    if (json_out.empty()) return;
    (void)write_json_atomic(json_out, build_json(true, phase, false, 0));
  };

  auto check_early_stop = [&](const std::string& phase) -> std::optional<int> {
    const int sig = static_cast<int>(g_stop_signal);
    if (sig == 0) return std::nullopt;
    std::fprintf(stderr, "early stop requested by signal=%d during phase=%s (wall_ms=%.1f)\n", sig, phase.c_str(),
                 wall_ms());
    if (!json_out.empty()) {
      (void)write_json_atomic(json_out, build_json(false, phase, true, sig));
    }
    return 1;
  };

  try {
    // Phase 1: cuBLASLt vs Gate 1 warp GEMM self-check (fast preflight on GPU).
    std::printf("[phase] start=gemm_parity_self_check wall_ms=%.1f\n", wall_ms());
    checkpoint_json("gemm_parity_self_check");
    {
      const auto t0 = std::chrono::steady_clock::now();
      gemm_report = Qwen38DenseBackend::self_check_gemm_parity(0);
      timings.gemm_self_check_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf(
          "  [gemm_parity_self_check] nvfp4_max_abs=%.6g fp8_cal_max_abs=%.6g bf16_max_abs=%.6g "
          "batch_inv=%.6g gdn_vs_step=%.6g gdn_batch_inv=%.6g elapsed_ms=%.1f -> %s\n",
          gemm_report->nvfp4_max_abs, gemm_report->fp8_cal_max_abs, gemm_report->bf16_max_abs,
          gemm_report->batch_invariance_max_abs, gemm_report->gdn_prefill_vs_step_max_abs,
          gemm_report->gdn_batch_invariance_max_abs, timings.gemm_self_check_ms,
          gemm_report->passed ? "PASS" : "FAIL");
    }
    if (auto rc = check_early_stop("gemm_parity_self_check")) return *rc;

    if (smoke_08b) {
      std::printf("[phase] start=smoke_08b wall_ms=%.1f\n", wall_ms());
      checkpoint_json("smoke_08b");
      if (!std::filesystem::is_directory(q08_dir)) {
        throw std::runtime_error("missing Qwen3.5-0.8B directory: " + q08_dir);
      }
      const auto t0 = std::chrono::steady_clock::now();
      Qwen38DenseLoadOptions opts08;
      opts08.model_dir = q08_dir;
      opts08.max_ctx = 1024;
      opts08.prefill_chunk = std::min(prefill_chunk, 256);
      opts08.verify_pins = false;
      opts08.use_calibrated_fp8 = false;
      opts08.revision = "local";
      opts08.device_id = 0;

      Qwen38DenseBackend backend08(opts08);
      load_stats = backend08.load_stats();
      residency = backend08.residency_report();
      state_id = backend08.state_identity();
      timings.cold_manifest_sha_ms = load_stats->cold_manifest_sha_ms;
      timings.cold_safetensors_load_ms = load_stats->cold_safetensors_load_ms;

      const std::array<std::int32_t, 16> prompt08 = {9707, 15,  220, 16,  17,  198, 220, 18,
                                                     198,  791, 4128, 374, 220, 17,  10,  18};
      const std::array<int, 1> last_pos = {static_cast<int>(prompt08.size()) - 1};
      std::vector<float> logits_chunk, logits_step;

      backend08.evaluate_teacher_forced_mode(
          prompt08, last_pos, true,
          [&](const ingine::model::LogitStep& st) { logits_chunk.assign(st.logits.begin(), st.logits.end()); },
          nullptr, nullptr, true);
      backend08.evaluate_teacher_forced_mode(
          prompt08, last_pos, false,
          [&](const ingine::model::LogitStep& st) { logits_step.assign(st.logits.begin(), st.logits.end()); },
          nullptr, nullptr, true);

      smoke08_chunk_vs_step_logit = compute_logit_kld_and_top1(logits_step, logits_chunk, last_pos[0]);
      backend08.generate_greedy_exact(prompt08, 16, true, false, smoke08_chunk_greedy);
      backend08.generate_greedy_exact(prompt08, 16, false, false, smoke08_step_greedy);

      smoke08_greedy_total = 16;
      smoke08_greedy_matches = 0;
      for (std::size_t k = 0; k < 16 && k < smoke08_chunk_greedy.size() && k < smoke08_step_greedy.size(); ++k) {
        if (smoke08_chunk_greedy[k] == smoke08_step_greedy[k]) ++smoke08_greedy_matches;
      }

      const auto t_rst0 = std::chrono::steady_clock::now();
      backend08.reset_session();
      timings.warm_reset_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_rst0).count();

      smoke08_executed = true;
      smoke08_passed = smoke08_chunk_vs_step_logit->finite && smoke08_chunk_vs_step_logit->kld <= 1e-4 &&
                       smoke08_greedy_matches == 16 && smoke08_chunk_greedy.size() == 16;
      timings.smoke_08b_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

      std::printf(
          "  [smoke_08b] load_ms=%.1f kld=%.6g top1_match=%s greedy=%zu/%zu peak_vram=%.1f MiB wall_ms=%.1f -> %s\n",
          timings.cold_safetensors_load_ms, smoke08_chunk_vs_step_logit->kld,
          smoke08_chunk_vs_step_logit->top1_match ? "true" : "false", smoke08_greedy_matches, smoke08_greedy_total,
          static_cast<double>(load_stats->peak_vram_bytes) / (1024.0 * 1024.0), wall_ms(),
          smoke08_passed ? "PASS" : "FAIL");

      const auto final_doc = build_json(false, "complete", false, 0);
      if (!json_out.empty() && !write_json_atomic(json_out, final_doc)) {
        std::fprintf(stderr, "failed to write final atomic json output path: %s\n", json_out.c_str());
        return 1;
      }
      const bool ok_all = gemm_report->passed && smoke08_passed;
      return ok_all ? 0 : 1;
    }

    // Phase 2: Verify release-reference & canonical-b1 safetensors + metadata SHA-256 digests and contract bindings.
    std::printf("[phase] start=verify_references wall_ms=%.1f\n", wall_ms());
    checkpoint_json("verify_references");
    const std::filesystem::path ref_root(ref_dir);
    const std::filesystem::path short_path = ref_root / "reference.safetensors";
    const std::filesystem::path short_meta_path = ref_root / "reference.json";
    const std::filesystem::path long_path = ref_root / "long-code-reference.safetensors";
    const std::filesystem::path alt_b1_root(alt_short_ref_dir);
    const std::filesystem::path alt_b1_path = alt_b1_root / "reference.safetensors";
    const std::filesystem::path alt_b1_meta_path = alt_b1_root / "reference.json";
    if (!std::filesystem::is_regular_file(short_path) || !std::filesystem::is_regular_file(short_meta_path) ||
        !std::filesystem::is_regular_file(long_path) || !std::filesystem::is_regular_file(alt_b1_path) ||
        !std::filesystem::is_regular_file(alt_b1_meta_path)) {
      throw std::runtime_error("missing pinned reference or canonical-b1 files under ref-dir / alt-short-ref-dir");
    }

    const auto t_ref0 = std::chrono::steady_clock::now();
    Qwen38RefSafetensors short_ref(short_path);
    Qwen38RefSafetensors alt_b1_ref(alt_b1_path);
    Qwen38RefSafetensors long_ref(long_path);
    short_ref_sha_actual = short_ref.sha256_hex();
    short_ref_sha_verified = (short_ref_sha_actual == kGate2ShortRefSha256);
    short_ref_meta_sha_verified =
        verify_file_sha256(short_meta_path, kGate2ShortRefMetaSha256, &short_ref_meta_sha_actual);
    alt_short_ref_b1_sha_actual = alt_b1_ref.sha256_hex();
    alt_short_ref_b1_sha_verified = (alt_short_ref_b1_sha_actual == kGate2AltShortRefB1Sha256);
    alt_short_ref_b1_meta_sha_verified =
        verify_file_sha256(alt_b1_meta_path, kGate2AltShortRefB1MetaSha256, &alt_short_ref_b1_meta_sha_actual);
    long_ref_sha_actual = long_ref.sha256_hex();
    long_ref_sha_verified = (long_ref_sha_actual == kGate2LongCodeRefSha256);

    auto parse_and_bind_meta = [&](const std::filesystem::path& mp, int expected_bs, int expected_threads,
                                   const Qwen38RefSafetensors& sf, nlohmann::ordered_json& summary_out) -> bool {
      std::ifstream ifs(mp);
      if (!ifs.is_open()) return false;
      const auto j = nlohmann::json::parse(ifs, nullptr, false);
      if (j.is_discarded() || !j.is_object()) return false;
      if (j.value("model_id", "") != "nvidia/Qwen3.8-27B-NVFP4" ||
          j.value("revision", "") != "482ca0f3832238542f8f5295dde86b5f22711d80" ||
          j.value("status", "") != "complete" ||
          j.value("config_sha256", "") != "71fa01bb64de20971045cd82a82c0ebb25a54e1b0a3f09ef23cf5479e9c54df8" ||
          j.value("index_sha256", "") != "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2") {
        return false;
      }
      if (!j.contains("options") || !j["options"].is_object()) return false;
      const int bs = j["options"].value("prompt_batch_size", -1);
      const int th = j["options"].value("threads", -1);
      if (bs != expected_bs || th != expected_threads) return false;
      if (!j.contains("prompts") || !j["prompts"].is_array() || j["prompts"].size() != 3u) return false;
      const auto& p0 = j["prompts"][0];
      if (!p0.contains("prompt_ids") || !p0["prompt_ids"].is_array() || !p0.contains("generated_ids") ||
          !p0["generated_ids"].is_array()) {
        return false;
      }
      const auto sf_p0_ids = sf.read_i64("prompt_0.prompt_ids");
      const auto sf_p0_gen = sf.read_i64("prompt_0.generated_ids");
      const auto meta_p0_ids = p0["prompt_ids"].get<std::vector<std::int64_t>>();
      const auto meta_p0_gen = p0["generated_ids"].get<std::vector<std::int64_t>>();
      if (sf_p0_ids != meta_p0_ids || sf_p0_gen != meta_p0_gen) return false;
      summary_out = {
          {"prompt_batch_size", bs},
          {"threads", th},
          {"torch", j.value("torch", "")},
          {"transformers", j.value("transformers", "")},
          {"device_name", j.value("device_name", "")},
          {"generated_ids", meta_p0_gen},
      };
      return true;
    };

    const bool rel_meta_bound = parse_and_bind_meta(short_meta_path, 3, 30, short_ref, rel_b3_meta_summary);
    const bool b1_meta_bound = parse_and_bind_meta(alt_b1_meta_path, 1, 16, alt_b1_ref, can_b1_meta_summary);
    const auto rel_p0_ids64 = short_ref.read_i64("prompt_0.prompt_ids");
    const auto rel_p0_gen64 = short_ref.read_i64("prompt_0.generated_ids");
    const auto b1_p0_ids64 = alt_b1_ref.read_i64("prompt_0.prompt_ids");
    const auto b1_p0_gen64 = alt_b1_ref.read_i64("prompt_0.generated_ids");
    const auto& rel_p0_pf = short_ref.at("prompt_0.prefill_logits");
    const auto& rel_p0_dc = short_ref.at("prompt_0.decode_logits");
    const auto& b1_p0_pf = alt_b1_ref.at("prompt_0.prefill_logits");
    const auto& b1_p0_dc = alt_b1_ref.at("prompt_0.decode_logits");
    const bool shapes_bound = (rel_p0_ids64.size() == 32u && rel_p0_gen64.size() == 16u &&
                               b1_p0_ids64.size() == 32u && b1_p0_gen64.size() == 16u &&
                               rel_p0_pf.shape.size() == 2 && rel_p0_pf.shape[0] == 32 &&
                               rel_p0_dc.shape.size() == 2 && rel_p0_dc.shape[0] == 15 &&
                               b1_p0_pf.shape == rel_p0_pf.shape && b1_p0_dc.shape == rel_p0_dc.shape);
    if (shapes_bound) {
      std::vector<float> tmp_row(static_cast<std::size_t>(rel_p0_dc.shape[1]));
      short_ref.read_f32_row("prompt_0.decode_logits", 0, tmp_row);
      p0_rel_b3_dec0_split_top2 = extract_split_top2(tmp_row);
      alt_b1_ref.read_f32_row("prompt_0.decode_logits", 0, tmp_row);
      p0_can_b1_dec0_split_top2 = extract_split_top2(tmp_row);
    }
    alt_short_ref_b1_prompt0_ids_match =
        shapes_bound && rel_meta_bound && b1_meta_bound && (rel_p0_ids64 == b1_p0_ids64) &&
        (rel_p0_gen64[0] == b1_p0_gen64[0]) && p0_rel_b3_dec0_split_top2.valid && p0_can_b1_dec0_split_top2.valid;

    timings.ref_sha_verify_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ref0).count();
    std::printf(
        "  [verify_references] short_sha=%s short_meta=%s b1_sha=%s b1_meta=%s p0_bind=%s long_sha=%s elapsed_ms=%.1f "
        "wall_ms=%.1f\n",
        short_ref_sha_verified ? "OK" : "MISMATCH", short_ref_meta_sha_verified ? "OK" : "MISMATCH",
        alt_short_ref_b1_sha_verified ? "OK" : "MISMATCH", alt_short_ref_b1_meta_sha_verified ? "OK" : "MISMATCH",
        alt_short_ref_b1_prompt0_ids_match ? "OK" : "MISMATCH", long_ref_sha_verified ? "OK" : "MISMATCH",
        timings.ref_sha_verify_ms, wall_ms());
    if (!short_ref_sha_verified || !short_ref_meta_sha_verified || !alt_short_ref_b1_sha_verified ||
        !alt_short_ref_b1_meta_sha_verified || !alt_short_ref_b1_prompt0_ids_match || !long_ref_sha_verified) {
      throw std::runtime_error("reference safetensors or Option 1 addendum metadata/pin mismatch (fail closed)");
    }
    if (auto rc = check_early_stop("verify_references")) return *rc;

    // Phase 3: Load Qwen3.8-27B-NVFP4 into VRAM with strict expected-pin verification.
    if (!std::filesystem::is_directory(q27_dir)) {
      throw std::runtime_error("missing Qwen3.8-27B-NVFP4 directory: " + q27_dir);
    }
    std::printf("[phase] start=load_qwen27b verify_pins=%s wall_ms=%.1f\n", verify_pins ? "true" : "false", wall_ms());
    checkpoint_json("load_qwen27b");

    Qwen38DenseLoadOptions opts27;
    opts27.model_dir = q27_dir;
    opts27.max_ctx = 4224;
    opts27.prefill_chunk = prefill_chunk;
    opts27.verify_pins = verify_pins;
    opts27.use_calibrated_fp8 = true;
    opts27.revision = "482ca0f";
    opts27.device_id = 0;
    if (verify_pins) opts27.expected_pins = qwen38_27b_nvfp4_pin_spec();

    Qwen38DenseBackend backend(opts27);
    load_stats = backend.load_stats();
    residency = backend.residency_report();
    state_id = backend.state_identity();
    q27_pins_verified_actual = load_stats->pins_verified;
    timings.cold_manifest_sha_ms = load_stats->cold_manifest_sha_ms;
    timings.cold_safetensors_load_ms = load_stats->cold_safetensors_load_ms;

    const int V = backend.config().vocab_size;
    p0_alt_b1_ref_ids = i64_to_i32(b1_p0_gen64, V);
    std::printf(
        "  [load_qwen27b] pins=%s sha_ms=%.1f load_ms=%.1f weights=%.1f MiB state=%.1f MiB scratch=%.1f MiB "
        "peak_vram=%.1f MiB wall_ms=%.1f\n",
        q27_pins_verified_actual ? "verified" : "UNVERIFIED", timings.cold_manifest_sha_ms,
        timings.cold_safetensors_load_ms, static_cast<double>(load_stats->weights_device_bytes) / (1024.0 * 1024.0),
        static_cast<double>(load_stats->state_device_bytes) / (1024.0 * 1024.0),
        static_cast<double>(load_stats->scratch_device_bytes) / (1024.0 * 1024.0),
        static_cast<double>(load_stats->peak_vram_bytes) / (1024.0 * 1024.0), wall_ms());
    if (auto rc = check_early_stop("load_qwen27b")) return *rc;

    std::vector<float> ref_row(static_cast<std::size_t>(V));

    // Phase 4: Short prompts (prompt_0..prompt_2 prefill + stateful decode logits + 16-tok greedy).
    std::printf("[phase] start=short_prompts wall_ms=%.1f\n", wall_ms());
    checkpoint_json("short_prompts");
    const auto t_short0 = std::chrono::steady_clock::now();
    for (int pi = 0; pi < 3; ++pi) {
      if (auto rc = check_early_stop("short_prompts")) return *rc;
      const std::string pfx = "prompt_" + std::to_string(pi);
      const auto p_ids_64 = short_ref.read_i64(pfx + ".prompt_ids");
      const auto g_ids_64 = short_ref.read_i64(pfx + ".generated_ids");
      const auto p_ids = i64_to_i32(p_ids_64, V);
      const auto g_ids = i64_to_i32(g_ids_64, V);
      const auto& pf_entry = short_ref.at(pfx + ".prefill_logits");
      const auto& dc_entry = short_ref.at(pfx + ".decode_logits");
      if (p_ids.size() != 32 || g_ids.size() != 16 || pf_entry.shape.size() != 2 || pf_entry.shape[0] != 32 ||
          pf_entry.shape[1] != V || dc_entry.shape.size() != 2 || dc_entry.shape[0] != 15 || dc_entry.shape[1] != V) {
        throw std::runtime_error("unexpected short prompt shape for " + pfx);
      }

      std::vector<int> pf_caps(p_ids.size());
      for (std::size_t k = 0; k < p_ids.size(); ++k) pf_caps[k] = static_cast<int>(k);

      std::vector<Qwen38LogitParityPoint> prompt_pts;
      prompt_pts.reserve(47);

      double pf_ms = 0.0;
      backend.evaluate_teacher_forced_mode(
          p_ids, pf_caps, true,
          [&](const ingine::model::LogitStep& st) {
            short_ref.read_f32_row(pfx + ".prefill_logits", static_cast<std::size_t>(st.position), ref_row);
            const auto pt = compute_logit_kld_and_top1(ref_row, st.logits, st.position);
            prompt_pts.push_back(pt);
            short_points.push_back(pt);
          },
          &pf_ms, nullptr, true);

      const std::span<const std::int32_t> dec_ids(g_ids.data(), 15);
      std::vector<int> dec_caps(15);
      for (int k = 0; k < 15; ++k) dec_caps[static_cast<std::size_t>(k)] = k;

      double dec_ms = 0.0;
      backend.evaluate_teacher_forced_mode(
          dec_ids, dec_caps, false,
          [&](const ingine::model::LogitStep& st) {
            const int local_dec = st.position - static_cast<int>(p_ids.size());
            if (local_dec < 0 || local_dec >= 15) {
              throw std::runtime_error("unexpected decode position from evaluate_teacher_forced_mode");
            }
            short_ref.read_f32_row(pfx + ".decode_logits", static_cast<std::size_t>(local_dec), ref_row);
            const auto pt = compute_logit_kld_and_top1(ref_row, st.logits, st.position);
            prompt_pts.push_back(pt);
            short_points.push_back(pt);
          },
          &dec_ms, nullptr, false);

      if (prompt_pts.size() != 47u) {
        throw std::runtime_error("incomplete short prompt logit count on " + pfx);
      }

      const auto t_rst0 = std::chrono::steady_clock::now();
      backend.reset_session();
      const double rst_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_rst0).count();
      if (timings.warm_reset_ms == 0.0) timings.warm_reset_ms = rst_ms;

      double g_pf_ms = 0.0, g_dec_ms = 0.0;
      std::vector<std::int32_t> cand_greedy;
      backend.generate_greedy_exact(
          p_ids, 16, true, false, cand_greedy, &g_pf_ms, &g_dec_ms, nullptr,
          [&](const ingine::model::LogitStep& st) {
            if (pi == 0 && st.position == static_cast<int>(p_ids.size())) {
              p0_cand_dec0_split_top2 = extract_split_top2(st.logits);
            }
          });
      std::size_t matches = 0;
      for (std::size_t k = 0; k < 16 && k < cand_greedy.size(); ++k) {
        if (cand_greedy[k] == g_ids[k]) ++matches;
      }

      if (pi == 0) {
        p0_traj_eval = evaluate_prompt0_canonical_trajectory(cand_greedy, g_ids, p0_alt_b1_ref_ids);
        // Always evaluate canonical-b1's own 47-row teacher-forced distribution separately from short_points.
        const auto alt_p_ids = i64_to_i32(b1_p0_ids64, V);
        p0_b1_points.clear();
        p0_b1_points.reserve(47);
        backend.evaluate_teacher_forced_mode(
            alt_p_ids, pf_caps, true,
            [&](const ingine::model::LogitStep& st) {
              alt_b1_ref.read_f32_row("prompt_0.prefill_logits", static_cast<std::size_t>(st.position), ref_row);
              p0_b1_points.push_back(compute_logit_kld_and_top1(ref_row, st.logits, st.position));
            },
            nullptr, nullptr, true);
        const std::span<const std::int32_t> b1_dec_ids(p0_alt_b1_ref_ids.data(), 15);
        backend.evaluate_teacher_forced_mode(
            b1_dec_ids, dec_caps, false,
            [&](const ingine::model::LogitStep& st) {
              const int local_dec = st.position - static_cast<int>(alt_p_ids.size());
              if (local_dec < 0 || local_dec >= 15) {
                throw std::runtime_error("unexpected decode position in canonical-b1 prompt_0 evaluation");
              }
              alt_b1_ref.read_f32_row("prompt_0.decode_logits", static_cast<std::size_t>(local_dec), ref_row);
              p0_b1_points.push_back(compute_logit_kld_and_top1(ref_row, st.logits, st.position));
            },
            nullptr, nullptr, false);
        backend.reset_session();
        if (p0_b1_points.size() != 47u) {
          throw std::runtime_error("incomplete canonical-b1 prompt_0 logit count");
        }
        p0_b1_eval_executed = true;
        const auto b1_sum = summarize_kld_metrics(p0_b1_points);
        std::printf(
            "  [prompt_0:addendum] chosen=%s rel_b3=%zu/16(diff=%d) can_b1=%zu/16(diff=%d) "
            "dec0_303_321 cand=(%.4f,%.4f) b3=(%.4f,%.4f) b1=(%.4f,%.4f) b1_dist(mean=%.6f,max=%.6f,top1=%.4f)\n",
            p0_traj_eval->chosen_realization.c_str(), p0_traj_eval->release_b3_matches,
            p0_traj_eval->release_b3_first_diff_idx, p0_traj_eval->canonical_b1_matches,
            p0_traj_eval->canonical_b1_first_diff_idx, p0_cand_dec0_split_top2.logit_303,
            p0_cand_dec0_split_top2.logit_321, p0_rel_b3_dec0_split_top2.logit_303,
            p0_rel_b3_dec0_split_top2.logit_321, p0_can_b1_dec0_split_top2.logit_303,
            p0_can_b1_dec0_split_top2.logit_321, b1_sum.kld_mean, b1_sum.kld_max, b1_sum.top1_agreement);
      }

      ShortPromptRecord rec;
      rec.name = pfx;
      rec.prompt_len = p_ids.size();
      rec.positions_evaluated = prompt_pts.size();
      rec.summary = summarize_kld_metrics(prompt_pts);
      rec.greedy_matches = matches;
      rec.greedy_total = 16;
      rec.ref_generated_ids = g_ids;
      rec.cand_greedy_ids = cand_greedy;
      rec.prefill_ms = pf_ms;
      rec.decode_ms = dec_ms;
      rec.greedy_ms = g_pf_ms + g_dec_ms;

      std::printf(
          "  [%s] kld_mean=%.6f kld_max=%.6f top1=%.4f (%zu/%zu) greedy=%zu/%zu pf_ms=%.1f dec_ms=%.1f wall_ms=%.1f\n",
          rec.name.c_str(), rec.summary.kld_mean, rec.summary.kld_max, rec.summary.top1_agreement,
          rec.summary.top1_matches, rec.summary.positions, rec.greedy_matches, rec.greedy_total, rec.prefill_ms,
          rec.decode_ms, wall_ms());
      short_records.push_back(std::move(rec));
      checkpoint_json("short_prompts");
    }
    timings.short_prompts_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_short0).count();

    // Phase 5: Chunked vs stepwise prefill + greedy equivalence across prompt_0..prompt_2.
    std::printf("[phase] start=chunk_vs_step wall_ms=%.1f\n", wall_ms());
    checkpoint_json("chunk_vs_step");
    const auto t_cs0 = std::chrono::steady_clock::now();
    for (int pi = 0; pi < 3; ++pi) {
      if (auto rc = check_early_stop("chunk_vs_step")) return *rc;
      const std::string pfx = "prompt_" + std::to_string(pi);
      const auto p_ids = i64_to_i32(short_ref.read_i64(pfx + ".prompt_ids"), V);
      const std::array<int, 1> last_cap = {static_cast<int>(p_ids.size()) - 1};

      std::vector<float> logits_chunk, logits_step;
      double chunk_ms = 0.0, step_ms = 0.0;
      backend.evaluate_teacher_forced_mode(
          p_ids, last_cap, true,
          [&](const ingine::model::LogitStep& st) { logits_chunk.assign(st.logits.begin(), st.logits.end()); },
          &chunk_ms, nullptr, true);
      backend.evaluate_teacher_forced_mode(
          p_ids, last_cap, false,
          [&](const ingine::model::LogitStep& st) { logits_step.assign(st.logits.begin(), st.logits.end()); },
          &step_ms, nullptr, true);

      const auto pt = compute_logit_kld_and_top1(logits_step, logits_chunk, pi);
      chunk_step_points.push_back(pt);

      std::vector<std::int32_t> chunk_greedy, step_greedy;
      backend.generate_greedy_exact(p_ids, 16, true, false, chunk_greedy);
      backend.generate_greedy_exact(p_ids, 16, false, false, step_greedy);
      std::size_t gmatch = 0;
      for (std::size_t k = 0; k < 16 && k < chunk_greedy.size() && k < step_greedy.size(); ++k) {
        if (chunk_greedy[k] == step_greedy[k]) ++gmatch;
      }

      ChunkVsStepRecord cs;
      cs.name = pfx;
      cs.final_logit = pt;
      cs.greedy_matches = gmatch;
      cs.greedy_total = 16;
      cs.chunk_prefill_ms = chunk_ms;
      cs.step_prefill_ms = step_ms;
      cs.chunk_greedy_ids = std::move(chunk_greedy);
      cs.step_greedy_ids = std::move(step_greedy);

      std::printf(
          "  [chunk_vs_step:%s] kld=%.6g top1_match=%s max_abs=%.6g greedy=%zu/%zu chunk_ms=%.1f step_ms=%.1f\n",
          cs.name.c_str(), cs.final_logit.kld, cs.final_logit.top1_match ? "true" : "false",
          cs.final_logit.max_abs_diff, cs.greedy_matches, cs.greedy_total, cs.chunk_prefill_ms, cs.step_prefill_ms);
      chunk_step_records.push_back(std::move(cs));
      checkpoint_json("chunk_vs_step");
    }
    timings.chunk_vs_step_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cs0).count();

    // Phase 6: Long-code reference prompts (prompt_0..prompt_5, ~4,096 tokens each, 48 positions per prompt).
    std::printf("[phase] start=long_code wall_ms=%.1f\n", wall_ms());
    checkpoint_json("long_code");
    const auto t_long0 = std::chrono::steady_clock::now();
    for (int pi = 0; pi < 6; ++pi) {
      if (auto rc = check_early_stop("long_code")) return *rc;
      const std::string pfx = "prompt_" + std::to_string(pi);
      const auto ids = i64_to_i32(long_ref.read_i64(pfx + ".input_ids"), V);
      const auto pos64 = long_ref.read_i64(pfx + ".positions");
      const auto& l_entry = long_ref.at(pfx + ".logits");
      if (pos64.empty() || l_entry.shape.size() != 2 || l_entry.shape[0] != static_cast<std::int64_t>(pos64.size()) ||
          l_entry.shape[1] != V) {
        throw std::runtime_error("unexpected long-code shape for " + pfx);
      }

      std::vector<int> caps;
      caps.reserve(pos64.size());
      std::unordered_map<int, std::size_t> pos_to_row;
      pos_to_row.reserve(pos64.size());
      for (std::size_t r = 0; r < pos64.size(); ++r) {
        if (pos64[r] < 0 || pos64[r] >= static_cast<std::int64_t>(ids.size())) {
          throw std::runtime_error("long-code capture position out of bounds on " + pfx);
        }
        const int p = static_cast<int>(pos64[r]);
        caps.push_back(p);
        pos_to_row.emplace(p, r);
      }

      std::vector<Qwen38LogitParityPoint> prompt_pts;
      prompt_pts.reserve(pos64.size());
      double pf_ms = 0.0;
      backend.evaluate_teacher_forced_mode(
          ids, caps, true,
          [&](const ingine::model::LogitStep& st) {
            const auto it = pos_to_row.find(st.position);
            if (it == pos_to_row.end()) {
              throw std::runtime_error("unexpected callback position on " + pfx);
            }
            long_ref.read_f32_row(pfx + ".logits", it->second, ref_row);
            const auto pt = compute_logit_kld_and_top1(ref_row, st.logits, st.position);
            prompt_pts.push_back(pt);
            long_points.push_back(pt);
          },
          &pf_ms, nullptr, true);

      if (prompt_pts.size() != pos64.size()) {
        throw std::runtime_error("incomplete long-code logit count on " + pfx);
      }

      LongPromptRecord lr;
      lr.name = pfx;
      lr.seq_len = ids.size();
      lr.positions_evaluated = prompt_pts.size();
      lr.summary = summarize_kld_metrics(prompt_pts);
      lr.prefill_ms = pf_ms;
      lr.prefill_tok_s = pf_ms > 0.0 ? (static_cast<double>(ids.size()) / (pf_ms / 1000.0)) : 0.0;

      std::printf(
          "  [long:%s] len=%zu kld_mean=%.6f kld_p99=%.6f kld_max=%.6f top1=%.4f (%zu/%zu) pf_ms=%.1f "
          "tok/s=%.1f wall_ms=%.1f\n",
          lr.name.c_str(), lr.seq_len, lr.summary.kld_mean, lr.summary.kld_p99, lr.summary.kld_max,
          lr.summary.top1_agreement, lr.summary.top1_matches, lr.summary.positions, lr.prefill_ms, lr.prefill_tok_s,
          wall_ms());
      long_records.push_back(std::move(lr));
      checkpoint_json("long_code");
    }
    timings.long_code_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_long0).count();

    // Measure warm prefix cache lookup on committed session state + warm reset + residency report.
    if (auto* sess = backend.acquire_session(0)) {
      const auto committed = sess->committed_tokens();
      if (!committed.empty()) {
        std::vector<std::int32_t> ext(committed.begin(), committed.end());
        ext.push_back(committed.back());
        ingine::model::SessionLookupQuery q;
        q.identity = backend.state_identity();
        q.prompt_ids = ext;
        q.use_prefix_cache = true;
        const auto t_lk0 = std::chrono::steady_clock::now();
        const auto lk = ingine::model::lookup_session_prefix(*sess, q);
        timings.warm_prefix_cache_lookup_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_lk0).count();
        prefix_lookup_hit = lk.hit;
        prefix_lookup_reused_tokens = lk.reused_tokens;
      }
    }
    residency = backend.residency_report();

    const auto t_rst_end0 = std::chrono::steady_clock::now();
    backend.reset_session();
    const double final_rst_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_rst_end0).count();
    timings.warm_reset_ms = timings.warm_reset_ms > 0.0 ? std::max(timings.warm_reset_ms, final_rst_ms) : final_rst_ms;
  } catch (const std::exception& ex) {
    error_message = ex.what();
    std::fprintf(stderr, "qwen38_gpu_gate2 error: %s\n", error_message.c_str());
    if (!json_out.empty()) {
      (void)write_json_atomic(json_out, build_json(false, "error", false, 0));
    }
    return 1;
  }

  Qwen38Gate2Verdict verdict;
  const auto final_doc = build_json(false, "complete", false, 0, &verdict);
  if (!json_out.empty() && !write_json_atomic(json_out, final_doc)) {
    std::fprintf(stderr, "failed to write final atomic json output path: %s\n", json_out.c_str());
    return 1;
  }

  const auto short_sum = summarize_kld_metrics(short_points);
  const auto long_sum = summarize_kld_metrics(long_points);
  std::printf(
      "qwen38_gpu_gate2: %s | contract=%s chosen=%s rel_only=%s short(mean=%.6f,max=%.6f,top1=%.4f,p0_b3=%zu/%zu,p0_b1=%zu/%zu) "
      "long(mean=%.6f,p99=%.6f,top1=%.4f) chunk_vs_step=%s pins=%s peak_vram=%.1f MiB wall_ms=%.1f\n",
      verdict.qualification_passed ? "PASS" : (diagnostic_mode && verdict.executed_checks_passed ? "DIAGNOSTIC_PASS" : "FAIL"),
      verdict.contract_version.c_str(), verdict.chosen_prompt0_realization.c_str(),
      verdict.release_only_qualification_passed ? "PASS" : "FAIL",
      short_sum.kld_mean, short_sum.kld_max, short_sum.top1_agreement,
      short_records.empty() ? 0u : short_records[0].greedy_matches,
      short_records.empty() ? 0u : short_records[0].greedy_total,
      p0_traj_eval.has_value() ? p0_traj_eval->canonical_b1_matches : 0u,
      p0_alt_b1_ref_ids.size(),
      long_sum.kld_mean, long_sum.kld_p99, long_sum.top1_agreement,
      verdict.chunked_vs_stepwise_passed ? "PASS" : "FAIL",
      q27_pins_verified_actual ? "verified" : "UNVERIFIED",
      load_stats.has_value() ? static_cast<double>(load_stats->peak_vram_bytes) / (1024.0 * 1024.0) : 0.0,
      timings.total_wall_ms);

  if (diagnostic_mode) return verdict.executed_checks_passed ? 0 : 1;
  return verdict.qualification_passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) { return main_impl(argc, argv); }

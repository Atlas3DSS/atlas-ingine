#include "backends/qwen38/qwen38_block_cpu.h"
#include "backends/qwen38/qwen38_cpu_ops.h"
#include "backends/qwen38/qwen38_cuda_ops.h"
#include "backends/qwen38/qwen38_manifest.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <optional>
#include <random>
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

template <typename T>
struct DevVec {
  T* ptr = nullptr;
  std::size_t n = 0;
  DevVec() = default;
  explicit DevVec(std::size_t count) : n(count) {
    if (n > 0) CUDA_OK(cudaMalloc(&ptr, n * sizeof(T)));
  }
  explicit DevVec(const std::vector<T>& host) : DevVec(host.size()) {
    if (n > 0) CUDA_OK(cudaMemcpy(ptr, host.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  DevVec(const DevVec&) = delete;
  DevVec& operator=(const DevVec&) = delete;
  ~DevVec() {
    if (ptr) cudaFree(ptr);
  }
  std::vector<T> download() const {
    std::vector<T> out(n);
    if (n > 0) CUDA_OK(cudaMemcpy(out.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return out;
  }
};

std::vector<__nv_bfloat16> to_bf16_nv(std::span<const float> src) {
  std::vector<__nv_bfloat16> out(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    const std::uint16_t b = f32_to_bf16(src[i]);
    std::memcpy(&out[i], &b, 2);
  }
  return out;
}

std::vector<__nv_bfloat16> bits_to_bf16_nv(std::span<const std::uint16_t> src) {
  std::vector<__nv_bfloat16> out(src.size());
  if (!src.empty()) std::memcpy(out.data(), src.data(), src.size() * 2);
  return out;
}

std::vector<std::uint16_t> to_bf16_bits(std::span<const __nv_bfloat16> src) {
  std::vector<std::uint16_t> out(src.size());
  if (!src.empty()) std::memcpy(out.data(), src.data(), src.size() * 2);
  return out;
}

std::vector<float> from_bf16_nv(std::span<const __nv_bfloat16> src) {
  std::vector<float> out(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    std::uint16_t b = 0;
    std::memcpy(&b, &src[i], 2);
    out[i] = bf16_to_f32(b);
  }
  return out;
}

float max_abs(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size()) return std::numeric_limits<float>::infinity();
  float m = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<float>::infinity();
    m = std::max(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

struct MatchedWarpMetrics {
  float hidden_max_abs = 0.0f;
  float hidden_rel_l2 = 0.0f;
  double hidden_cosine = 1.0;
  float layer0_gdn_max_abs = 0.0f;
  float stacked_gdn_max_abs = 0.0f;
  float conv_max_abs = 0.0f;
  float conv_rel_l2 = 0.0f;
  float k_cache_max_abs = 0.0f;
  float k_cache_rel_l2 = 0.0f;
  float v_cache_max_abs = 0.0f;
  float v_cache_rel_l2 = 0.0f;
  std::string criterion;
  bool passed = false;
};

struct BlockStateMetrics {
  bool state_shapes_valid = true;
  bool conv_finite = true;
  bool gdn_finite = true;
  bool kv_finite = true;
  float layer0_conv_max_abs = 0.0f;
  float layer0_conv_rel_l2 = 0.0f;
  float layer0_conv_rel_max_abs = 0.0f;
  double layer0_conv_cosine = 1.0;
  float stacked_conv_max_abs = 0.0f;
  float stacked_conv_rel_l2 = 0.0f;
  float stacked_conv_rel_max_abs = 0.0f;
  double stacked_conv_cosine = 1.0;
  float k_cache_max_abs = 0.0f;
  float k_cache_rel_l2 = 0.0f;
  float k_cache_rel_max_abs = 0.0f;
  double k_cache_cosine = 1.0;
  float v_cache_max_abs = 0.0f;
  float v_cache_rel_l2 = 0.0f;
  float v_cache_rel_max_abs = 0.0f;
  double v_cache_cosine = 1.0;
};

struct Gate1Record {
  std::string name;
  std::string category;  // "strict_operator", "strict_synthetic_block", "real_block_envelope"
  std::string status = "pass";
  std::string criterion;
  float max_abs_diff = 0.0f;
  float aux_max_abs = 0.0f;
  std::optional<float> ref_max_abs;
  std::optional<float> rel_max_abs;
  std::optional<float> rel_l2;
  std::optional<double> cosine_sim;
  std::optional<float> layer0_gdn_max_abs;
  std::optional<float> stacked_gdn_rel_l2;
  std::optional<std::uint32_t> max_bf16_ulp;
  std::optional<std::uint32_t> max_bf16_ulp_above_floor;
  std::optional<std::size_t> floor_exempt_count;
  std::optional<std::size_t> bf16_vs_gpu_f32_bit_mismatches;
  std::optional<BlockStateMetrics> state_metrics;
  std::optional<MatchedWarpMetrics> matched_warp_f32;
  double elapsed_ms = 0.0;
  bool passed = false;
};

struct PhaseTimingsMs {
  double unit_operators_ms = 0.0;
  double synthetic_blocks_ms = 0.0;
  double qwen08b_inspect_ms = 0.0;
  double qwen08b_load_ms = 0.0;
  double qwen08b_cpu_fp64_ms = 0.0;
  double qwen08b_cpu_warp_ms = 0.0;
  double qwen08b_gpu_ms = 0.0;
  double qwen27b_manifest_sha_ms = 0.0;
  double qwen27b_stream_load_ms = 0.0;
  double qwen27b_cpu_fp64_ms = 0.0;
  double qwen27b_cpu_warp_ms = 0.0;
  double qwen27b_gpu_ms = 0.0;
  double total_wall_ms = 0.0;
};

int main_impl(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::setvbuf(stderr, nullptr, _IOLBF, 0);

  std::string q08_dir = "/mnt/storage/models/Qwen3.5-0.8B";
  std::string q27_dir = "/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f";
  std::string json_out;
  bool verify_pins = true;  // Fail-closed qualification default.
  bool diagnostic_mode = false;
  bool skip_08b = false;
  bool skip_27b = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--qwen08-dir") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing argument for --qwen08-dir\n");
        return 2;
      }
      q08_dir = argv[++i];
    } else if (a == "--qwen27-dir") {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing argument for --qwen27-dir\n");
        return 2;
      }
      q27_dir = argv[++i];
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
    } else if (a == "--skip-08b") {
      skip_08b = true;
    } else if (a == "--skip-27b") {
      skip_27b = true;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }

  if (!diagnostic_mode && (skip_08b || skip_27b || !verify_pins)) {
    std::fprintf(stderr,
                 "error: --skip-08b, --skip-27b, and --no-verify-pins require --diagnostic and are forbidden in "
                 "Gate 1 qualification mode\n");
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
  std::printf("Gate 1 running [%s] on %s (sm_%d%d, vram=%.2f GiB, verify_pins=%s)\n",
              diagnostic_mode ? "DIAGNOSTIC" : "QUALIFICATION", props.name, props.major, props.minor,
              static_cast<double>(props.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
              verify_pins ? "true" : "false");

  std::vector<Gate1Record> records;
  PhaseTimingsMs timings;
  std::uint64_t peak_vram_bytes = 0;
  bool q27_pins_verified_actual = false;

  auto build_json = [&](bool in_progress, const std::string& current_phase, bool terminated_early, int stop_sig,
                        Qwen38Gate1CoverageSummary* out_summary = nullptr) {
    std::vector<Qwen38Gate1RowStatus> statuses;
    statuses.reserve(records.size());
    for (const auto& r : records) statuses.push_back({r.name, r.status, r.passed});
    auto summary = evaluate_gate1_coverage(statuses, diagnostic_mode, q27_pins_verified_actual);
    if (out_summary) *out_summary = summary;

    struct rusage ru {};
    std::uint64_t host_rss_bytes = 0;
    if (::getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_maxrss > 0) {
      host_rss_bytes = static_cast<std::uint64_t>(ru.ru_maxrss) * 1024ull;
    }

    auto category_all_passed = [&](std::string_view cat, std::size_t expected_count, bool require_warp = false) {
      if (in_progress || terminated_early) return false;
      std::size_t passed_count = 0;
      for (const auto& r : records) {
        if (r.category != cat) continue;
        if (r.status != "pass" || !r.passed) return false;
        if (require_warp && (!r.matched_warp_f32.has_value() || !r.matched_warp_f32->passed)) return false;
        ++passed_count;
      }
      return passed_count == expected_count;
    };

    const bool strict_op_ok = category_all_passed("strict_operator", 13);
    const bool strict_synth_ok = category_all_passed("strict_synthetic_block", 8);
    const bool real_env_ok = category_all_passed("real_block_envelope", 8, false);
    const bool real_warp_ok = category_all_passed("real_block_envelope", 8, true);

    const std::string cov_status =
        terminated_early ? "incomplete_terminated" : (in_progress ? "incomplete_partial" : summary.coverage_status);
    const bool final_qual = (!in_progress && !terminated_early) ? summary.qualification_passed : false;
    const bool final_complete = (!in_progress && !terminated_early) ? summary.complete_coverage : false;

    timings.total_wall_ms = wall_ms();

    nlohmann::ordered_json root;
    root["device"] = props.name;
    root["sm"] = std::to_string(props.major) + std::to_string(props.minor);
    root["mode"] = summary.mode;
    root["in_progress"] = in_progress;
    root["current_phase"] = current_phase;
    root["terminated_early"] = terminated_early;
    if (stop_sig != 0) root["signal"] = stop_sig;
    root["coverage_status"] = cov_status;
    root["complete_coverage"] = final_complete;
    root["executed_checks_passed"] = summary.executed_checks_passed;
    root["qwen27b_pins_verified"] = summary.qwen27b_pins_verified;
    root["qualification_passed"] = final_qual;
    root["all_passed"] = final_qual;
    root["strict_operator_all_passed"] = strict_op_ok;
    root["strict_synthetic_all_passed"] = strict_synth_ok;
    root["real_block_envelope_all_passed"] = real_env_ok;
    root["real_block_matched_warp_all_passed"] = real_warp_ok;
    root["required_row_count"] = summary.required_row_count;
    root["passed_required_row_count"] = summary.passed_required_row_count;
    root["missing_required_rows"] = summary.missing_required_rows;
    root["failed_rows"] = summary.failed_rows;
    root["qwen08b_dir"] = q08_dir;
    root["qwen27b_dir"] = q27_dir;
    root["cold_manifest_sha_ms"] = timings.qwen27b_manifest_sha_ms;
    root["cold_safetensors_load_ms"] = timings.qwen27b_stream_load_ms;
    root["phase_timings_ms"] = {
        {"unit_operators_ms", timings.unit_operators_ms},
        {"synthetic_blocks_ms", timings.synthetic_blocks_ms},
        {"qwen08b_inspect_ms", timings.qwen08b_inspect_ms},
        {"qwen08b_load_ms", timings.qwen08b_load_ms},
        {"qwen08b_cpu_fp64_ms", timings.qwen08b_cpu_fp64_ms},
        {"qwen08b_cpu_warp_ms", timings.qwen08b_cpu_warp_ms},
        {"qwen08b_gpu_ms", timings.qwen08b_gpu_ms},
        {"qwen27b_manifest_sha_ms", timings.qwen27b_manifest_sha_ms},
        {"qwen27b_stream_load_ms", timings.qwen27b_stream_load_ms},
        {"qwen27b_cpu_fp64_ms", timings.qwen27b_cpu_fp64_ms},
        {"qwen27b_cpu_warp_ms", timings.qwen27b_cpu_warp_ms},
        {"qwen27b_gpu_ms", timings.qwen27b_gpu_ms},
        {"total_wall_ms", timings.total_wall_ms},
    };
    root["peak_vram_bytes"] = peak_vram_bytes;
    root["host_rss_bytes"] = host_rss_bytes;
    root["checks"] = nlohmann::json::array();
    for (const auto& r : records) {
      nlohmann::ordered_json j;
      j["name"] = r.name;
      j["category"] = r.category;
      j["status"] = r.status;
      j["criterion"] = r.criterion;
      j["max_abs"] = r.max_abs_diff;
      j["aux_max_abs"] = r.aux_max_abs;
      if (r.ref_max_abs.has_value()) j["ref_max_abs"] = *r.ref_max_abs;
      if (r.rel_max_abs.has_value()) j["rel_max_abs"] = *r.rel_max_abs;
      if (r.rel_l2.has_value()) j["rel_l2"] = *r.rel_l2;
      if (r.cosine_sim.has_value()) j["cosine_sim"] = *r.cosine_sim;
      if (r.layer0_gdn_max_abs.has_value()) j["layer0_gdn_max_abs"] = *r.layer0_gdn_max_abs;
      if (r.stacked_gdn_rel_l2.has_value()) j["stacked_gdn_rel_l2"] = *r.stacked_gdn_rel_l2;
      if (r.max_bf16_ulp.has_value()) j["max_bf16_ulp"] = *r.max_bf16_ulp;
      if (r.max_bf16_ulp_above_floor.has_value()) j["max_bf16_ulp_above_floor"] = *r.max_bf16_ulp_above_floor;
      if (r.floor_exempt_count.has_value()) j["floor_exempt_count"] = *r.floor_exempt_count;
      if (r.bf16_vs_gpu_f32_bit_mismatches.has_value()) {
        j["bf16_vs_gpu_f32_bit_mismatches"] = *r.bf16_vs_gpu_f32_bit_mismatches;
      }
      if (r.state_metrics.has_value()) {
        const auto& sm = *r.state_metrics;
        j["state_metrics"] = {
            {"state_shapes_valid", sm.state_shapes_valid},
            {"conv_finite", sm.conv_finite},
            {"gdn_finite", sm.gdn_finite},
            {"kv_finite", sm.kv_finite},
            {"layer0_conv_max_abs", sm.layer0_conv_max_abs},
            {"layer0_conv_rel_l2", sm.layer0_conv_rel_l2},
            {"layer0_conv_rel_max_abs", sm.layer0_conv_rel_max_abs},
            {"layer0_conv_cosine", sm.layer0_conv_cosine},
            {"stacked_conv_max_abs", sm.stacked_conv_max_abs},
            {"stacked_conv_rel_l2", sm.stacked_conv_rel_l2},
            {"stacked_conv_rel_max_abs", sm.stacked_conv_rel_max_abs},
            {"stacked_conv_cosine", sm.stacked_conv_cosine},
            {"k_cache_max_abs", sm.k_cache_max_abs},
            {"k_cache_rel_l2", sm.k_cache_rel_l2},
            {"k_cache_rel_max_abs", sm.k_cache_rel_max_abs},
            {"k_cache_cosine", sm.k_cache_cosine},
            {"v_cache_max_abs", sm.v_cache_max_abs},
            {"v_cache_rel_l2", sm.v_cache_rel_l2},
            {"v_cache_rel_max_abs", sm.v_cache_rel_max_abs},
            {"v_cache_cosine", sm.v_cache_cosine},
        };
      }
      if (r.matched_warp_f32.has_value()) {
        const auto& mw = *r.matched_warp_f32;
        j["matched_warp_f32"] = {
            {"criterion", mw.criterion},
            {"hidden_max_abs", mw.hidden_max_abs},
            {"hidden_rel_l2", mw.hidden_rel_l2},
            {"hidden_cosine", mw.hidden_cosine},
            {"layer0_gdn_max_abs", mw.layer0_gdn_max_abs},
            {"stacked_gdn_max_abs", mw.stacked_gdn_max_abs},
            {"conv_max_abs", mw.conv_max_abs},
            {"conv_rel_l2", mw.conv_rel_l2},
            {"k_cache_max_abs", mw.k_cache_max_abs},
            {"k_cache_rel_l2", mw.k_cache_rel_l2},
            {"v_cache_max_abs", mw.v_cache_max_abs},
            {"v_cache_rel_l2", mw.v_cache_rel_l2},
            {"passed", mw.passed},
        };
      }
      j["elapsed_ms"] = r.elapsed_ms;
      j["passed"] = r.passed;
      root["checks"].push_back(std::move(j));
    }
    return root;
  };

  auto checkpoint_json = [&](const std::string& phase) {
    if (json_out.empty()) return;
    (void)write_json_atomic(json_out, build_json(true, phase, false, 0));
  };

  auto check_early_stop = [&](const std::string& phase) -> std::optional<int> {
    if (g_stop_signal == 0) return std::nullopt;
    const int sig = static_cast<int>(g_stop_signal);
    std::printf("qwen38_gpu_gate1: TERMINATED_EARLY signal=%d phase=%s wall_ms=%.1f (qualification=false)\n", sig,
                phase.c_str(), wall_ms());
    if (!json_out.empty()) {
      (void)write_json_atomic(json_out, build_json(false, phase, true, sig));
    }
    return 128 + sig;
  };

  auto record_exec = [&](const std::string& name, const std::string& criterion, float max_d, float aux_d, double ms,
                         bool ok, const std::string& phase_label,
                         std::optional<std::uint32_t> max_ulp = std::nullopt,
                         std::optional<std::uint32_t> max_ulp_above = std::nullopt,
                         std::optional<std::size_t> floor_exempt = std::nullopt,
                         std::optional<std::size_t> f32_bf16_mm = std::nullopt) {
    Gate1Record r;
    r.name = name;
    r.category = "strict_operator";
    r.status = ok ? "pass" : "fail";
    r.criterion = criterion;
    r.max_abs_diff = max_d;
    r.aux_max_abs = aux_d;
    r.max_bf16_ulp = max_ulp;
    r.max_bf16_ulp_above_floor = max_ulp_above;
    r.floor_exempt_count = floor_exempt;
    r.bf16_vs_gpu_f32_bit_mismatches = f32_bf16_mm;
    r.elapsed_ms = ms;
    r.passed = ok;
    records.push_back(std::move(r));
    std::printf("  [%s] max_abs=%.6g aux=%.6g wall_ms=%.1f -> %s\n", name.c_str(), max_d, aux_d, wall_ms(),
                ok ? "PASS" : "FAIL");
    checkpoint_json(phase_label);
  };

  // Phase 1: Strict unit operators.
  const auto t_unit0 = std::chrono::steady_clock::now();
  std::printf("[phase] start=unit_operators wall_ms=%.1f\n", wall_ms());
  checkpoint_json("unit_operators");
  {
    const auto lin = make_gate1_linear_fixtures(20261125u);
    const std::size_t out_n = static_cast<std::size_t>(lin.m * lin.n);

    auto count_bit_mismatches = [&](std::span<const std::uint16_t> got_bits, std::span<const float> got_f32) {
      std::size_t mm = 0;
      for (std::size_t i = 0; i < out_n; ++i) {
        if (got_bits[i] != f32_to_bf16(got_f32[i])) ++mm;
      }
      return mm;
    };

    // 1. nvfp4_linear (F32 + BF16)
    {
      DevVec<float> d_x_f32(lin.nvfp4_x_q), d_out_f32(out_n);
      DevVec<__nv_bfloat16> d_x_bf16(bits_to_bf16_nv(lin.nvfp4_x_bf16)), d_out_bf16(out_n);
      DevVec<std::uint8_t> d_p(lin.nvfp4_packed), d_s(lin.nvfp4_scales);

      qwen38_cuda_nvfp4_linear_f32(d_x_f32.ptr, d_p.ptr, d_s.ptr, lin.nvfp4_ws2, lin.m, lin.n, lin.k, d_out_f32.ptr);
      qwen38_cuda_nvfp4_linear_bf16(d_x_bf16.ptr, d_p.ptr, d_s.ptr, lin.nvfp4_ws2, lin.m, lin.n, lin.k,
                                     d_out_bf16.ptr);
      CUDA_OK(cudaDeviceSynchronize());

      const auto got_f32 = d_out_f32.download();
      const auto got_bits = to_bf16_bits(d_out_bf16.download());
      const float d_f32 = max_abs(got_f32, lin.nvfp4_ref_f32);
      const std::size_t mm = count_bit_mismatches(got_bits, got_f32);
      const auto cmp = compare_bf16_against_rounded_oracle(got_bits, lin.nvfp4_ref_f32, 1u, 1e-3f);

      record_exec("unit_nvfp4_linear_f32", "raw_f32_max_abs <= 1e-3", d_f32, 0.0f, 0.0, d_f32 <= 1e-3f,
                  "unit_operators");
      record_exec("unit_nvfp4_linear_bf16", "bf16_oracle_ulp <= 1 || abs <= 1e-3; exact_gpu_f32_to_bf16 == 0",
                  cmp.max_abs_vs_bf16_oracle, static_cast<float>(cmp.criterion_violations), 0.0,
                  mm == 0 && cmp.passed, "unit_operators", cmp.max_bf16_ulp, cmp.max_bf16_ulp_above_floor,
                  cmp.floor_exempt_count, mm);
    }

    // 2A & 2B. fp8_linear weight-only (input_scale = 0) and calibrated (input_scale = 0.125), both F32 + BF16.
    {
      DevVec<float> d_x_f32(lin.fp8_x_q), d_out_wo_f32(out_n), d_out_cal_f32(out_n);
      DevVec<__nv_bfloat16> d_x_bf16(bits_to_bf16_nv(lin.fp8_x_bf16)), d_out_wo_bf16(out_n), d_out_cal_bf16(out_n);
      DevVec<std::uint8_t> d_w(lin.fp8_w);

      qwen38_cuda_fp8_linear_f32(d_x_f32.ptr, d_w.ptr, lin.fp8_ws, 0.0f, lin.m, lin.n, lin.k, d_out_wo_f32.ptr);
      qwen38_cuda_fp8_linear_bf16(d_x_bf16.ptr, d_w.ptr, lin.fp8_ws, 0.0f, lin.m, lin.n, lin.k, d_out_wo_bf16.ptr);
      qwen38_cuda_fp8_linear_f32(d_x_f32.ptr, d_w.ptr, lin.fp8_ws, lin.fp8_in_sc, lin.m, lin.n, lin.k,
                                 d_out_cal_f32.ptr);
      qwen38_cuda_fp8_linear_bf16(d_x_bf16.ptr, d_w.ptr, lin.fp8_ws, lin.fp8_in_sc, lin.m, lin.n, lin.k,
                                   d_out_cal_bf16.ptr);
      CUDA_OK(cudaDeviceSynchronize());

      const auto wo_f32 = d_out_wo_f32.download();
      const auto wo_bits = to_bf16_bits(d_out_wo_bf16.download());
      const float d_wo_f32 = max_abs(wo_f32, lin.fp8_wo_ref_f32);
      const std::size_t mm_wo = count_bit_mismatches(wo_bits, wo_f32);
      const auto cmp_wo = compare_bf16_against_rounded_oracle(wo_bits, lin.fp8_wo_ref_f32, 1u, 1e-3f);
      record_exec("unit_fp8_linear_weight_only_f32", "raw_f32_max_abs <= 1e-3", d_wo_f32, 0.0f, 0.0,
                  d_wo_f32 <= 1e-3f, "unit_operators");
      record_exec("unit_fp8_linear_weight_only_bf16",
                  "bf16_oracle_ulp <= 1 || abs <= 1e-3; exact_gpu_f32_to_bf16 == 0", cmp_wo.max_abs_vs_bf16_oracle,
                  static_cast<float>(cmp_wo.criterion_violations), 0.0, mm_wo == 0 && cmp_wo.passed,
                  "unit_operators", cmp_wo.max_bf16_ulp, cmp_wo.max_bf16_ulp_above_floor, cmp_wo.floor_exempt_count,
                  mm_wo);

      const auto cal_f32 = d_out_cal_f32.download();
      const auto cal_bits = to_bf16_bits(d_out_cal_bf16.download());
      const float d_cal_f32 = max_abs(cal_f32, lin.fp8_cal_ref_f32);
      const std::size_t mm_cal = count_bit_mismatches(cal_bits, cal_f32);
      const auto cmp_cal = compare_bf16_against_rounded_oracle(cal_bits, lin.fp8_cal_ref_f32, 1u, 1e-3f);
      record_exec("unit_fp8_linear_calibrated_f32", "raw_f32_max_abs <= 1e-3", d_cal_f32, 0.0f, 0.0,
                  d_cal_f32 <= 1e-3f, "unit_operators");
      record_exec("unit_fp8_linear_calibrated_bf16",
                  "bf16_oracle_ulp <= 1 || abs <= 1e-3; exact_gpu_f32_to_bf16 == 0", cmp_cal.max_abs_vs_bf16_oracle,
                  static_cast<float>(cmp_cal.criterion_violations), 0.0, mm_cal == 0 && cmp_cal.passed,
                  "unit_operators", cmp_cal.max_bf16_ulp, cmp_cal.max_bf16_ulp_above_floor,
                  cmp_cal.floor_exempt_count, mm_cal);
    }
  }

  std::mt19937 rng(20261125u ^ 0x51a7u);
  auto rand_vec = [&](std::size_t n, float stddev = 0.4f) {
    std::normal_distribution<float> d(0.0f, stddev);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
  };

  // 3. rmsnorm_zero_centered (27B H=5120).
  {
    constexpr int M = 8, D = 5120;
    const auto x_f = rand_vec(M * D, 1.2f);
    const auto w_b = to_bf16_nv(rand_vec(D, 0.15f));
    const auto w_f = from_bf16_nv(w_b);
    std::vector<float> ref(M * D);
    for (int m = 0; m < M; ++m) {
      rmsnorm_zero_centered(std::span<const float>(x_f.data() + m * D, D), w_f, 1e-6f,
                            std::span<float>(ref.data() + m * D, D));
    }
    DevVec<float> d_x(x_f), d_out(M * D);
    DevVec<__nv_bfloat16> d_w(w_b);
    qwen38_cuda_rmsnorm_zero_centered_f32(d_x.ptr, d_w.ptr, 1e-6f, M, D, d_out.ptr);
    CUDA_OK(cudaDeviceSynchronize());
    const float d_rms = max_abs(d_out.download(), ref);
    record_exec("unit_rmsnorm_zero_centered", "raw_f32_max_abs <= 1e-5", d_rms, 0.0f, 0.0, d_rms <= 1e-5f,
                "unit_operators");
  }

  // 4. rmsnorm_gated_silu (27B v_dim=128, M=48).
  {
    constexpr int M = 48, D = 128;
    const auto x_f = rand_vec(M * D, 0.8f);
    const auto z_f = rand_vec(M * D, 0.8f);
    const auto w_f = rand_vec(D, 0.5f);
    std::vector<float> ref(M * D);
    for (int m = 0; m < M; ++m) {
      rmsnorm_gated_silu(std::span<const float>(x_f.data() + m * D, D),
                         std::span<const float>(z_f.data() + m * D, D), w_f, 1e-6f,
                         std::span<float>(ref.data() + m * D, D));
    }
    DevVec<float> d_x(x_f), d_z(z_f), d_w(w_f), d_out(M * D);
    qwen38_cuda_rmsnorm_gated_silu_f32(d_x.ptr, d_z.ptr, d_w.ptr, 1e-6f, M, D, d_out.ptr);
    CUDA_OK(cudaDeviceSynchronize());
    const float d_gr = max_abs(d_out.download(), ref);
    record_exec("unit_rmsnorm_gated_silu", "raw_f32_max_abs <= 1e-5", d_gr, 0.0f, 0.0, d_gr <= 1e-5f,
                "unit_operators");
  }

  // 5. partial_rope_neox (27B H=24, head_dim=256, rotary_dim=64, pos_offset=37).
  {
    constexpr int M = 4, H = 24, HD = 256, RD = 64, POS0 = 37;
    const auto x_f = rand_vec(M * H * HD, 1.0f);
    auto ref = x_f;
    for (int m = 0; m < M; ++m) {
      apply_partial_rope_neox(std::span<float>(ref.data() + m * H * HD, H * HD), H, HD, RD, POS0 + m, 10000000.0f);
    }
    DevVec<float> d_qk(x_f);
    qwen38_cuda_partial_rope_neox_f32(d_qk.ptr, M, H, HD, RD, POS0, 10000000.0f);
    CUDA_OK(cudaDeviceSynchronize());
    const auto got = d_qk.download();
    const float d_rope = max_abs(got, ref);
    bool slice_exact = true;
    for (int m = 0; m < M && slice_exact; ++m) {
      for (int h = 0; h < H && slice_exact; ++h) {
        for (int d = RD; d < HD; ++d) {
          const std::size_t idx = (static_cast<std::size_t>(m * H + h)) * HD + d;
          if (got[idx] != x_f[idx]) {
            slice_exact = false;
            break;
          }
        }
      }
    }
    record_exec("unit_partial_rope_neox", "raw_f32_max_abs <= 1e-5 && passthrough_exact", d_rope,
                slice_exact ? 0.0f : 1.0f, 0.0, d_rope <= 1e-5f && slice_exact, "unit_operators");
  }

  // 6. causal_conv1d_silu (step & prefill, 27B conv_dim=10240, kernel=4, seq=8, non-zero init state).
  {
    constexpr int S = 8, CD = 10240, K = 4;
    const auto x_f = rand_vec(S * CD, 0.5f);
    const auto w_f = rand_vec(CD * K, 0.3f);
    const auto st_init = rand_vec(CD * K, 0.2f);
    auto st_cpu = st_init;
    std::vector<float> out_cpu(S * CD);
    for (int t = 0; t < S; ++t) {
      causal_conv1d_silu_step(std::span<const float>(x_f.data() + t * CD, CD), w_f, CD, K, st_cpu,
                              std::span<float>(out_cpu.data() + t * CD, CD));
    }
    DevVec<float> d_x(x_f), d_w(w_f), d_st(st_init), d_out(S * CD);
    qwen38_cuda_causal_conv1d_silu_prefill_f32(d_x.ptr, d_w.ptr, S, CD, K, d_st.ptr, d_out.ptr);
    CUDA_OK(cudaDeviceSynchronize());
    const float d_out_v = max_abs(d_out.download(), out_cpu);
    const float d_st_v = max_abs(d_st.download(), st_cpu);
    record_exec("unit_causal_conv1d_silu", "raw_f32_out <= 1e-5 && state <= 1e-6", d_out_v, d_st_v, 0.0,
                d_out_v <= 1e-5f && d_st_v <= 1e-6f, "unit_operators");
  }

  // 7 & 8. gdn_recurrent_step & gdn_chunk_prefill (27B 16K/48V heads, 128x128, seq=70 across 2 chunks 64+6).
  {
    constexpr int S = 70, CS = 64, NKH = 16, NVH = 48, KD = 128, VD = 128;
    const auto q_f = rand_vec(S * NKH * KD, 0.3f);
    const auto k_f = rand_vec(S * NKH * KD, 0.3f);
    const auto v_f = rand_vec(S * NVH * VD, 0.3f);
    const auto z_f = rand_vec(S * NVH * VD, 0.3f);
    const auto a_f = rand_vec(S * NVH, 0.2f);
    const auto b_f = rand_vec(S * NVH, 0.2f);
    const auto alog = rand_vec(NVH, 0.2f);
    const auto dtb = rand_vec(NVH, 0.1f);
    const auto nw = rand_vec(VD, 0.5f);
    const auto st_init = rand_vec(NVH * KD * VD, 0.05f);

    auto st_cpu_step = st_init;
    std::vector<float> out_cpu_step(NVH * VD);
    gdn_recurrent_step(std::span<const float>(q_f.data(), NKH * KD), std::span<const float>(k_f.data(), NKH * KD),
                       std::span<const float>(v_f.data(), NVH * VD), std::span<const float>(z_f.data(), NVH * VD),
                       std::span<const float>(a_f.data(), NVH), std::span<const float>(b_f.data(), NVH), alog, dtb,
                       nw, 1e-6f, NKH, NVH, KD, VD, st_cpu_step, out_cpu_step);

    DevVec<float> d_q(q_f), d_k(k_f), d_v(v_f), d_z(z_f), d_a(a_f), d_b(b_f), d_al(alog), d_dt(dtb), d_nw(nw);
    DevVec<float> d_st_step(st_init), d_out_step(NVH * VD);
    DevVec<float> d_ws(qwen38_cuda_gdn_chunk_scratch_floats(S, CS, NKH, NVH, KD, VD));
    qwen38_cuda_gdn_recurrent_step_f32(d_q.ptr, d_k.ptr, d_v.ptr, d_z.ptr, d_a.ptr, d_b.ptr, d_al.ptr, d_dt.ptr,
                                       d_nw.ptr, 1e-6f, NKH, NVH, KD, VD, d_st_step.ptr, d_ws.ptr, d_out_step.ptr);
    CUDA_OK(cudaDeviceSynchronize());
    const float d_step_o = max_abs(d_out_step.download(), out_cpu_step);
    const float d_step_s = max_abs(d_st_step.download(), st_cpu_step);
    record_exec("unit_gdn_recurrent_step", "raw_f32_out <= 1e-4 && state <= 2e-4", d_step_o, d_step_s, 0.0,
                d_step_o <= 1e-4f && d_step_s <= 2e-4f, "unit_operators");

    auto st_cpu_ch = st_init;
    std::vector<float> out_cpu_ch(S * NVH * VD);
    gdn_chunk_prefill(q_f, k_f, v_f, z_f, a_f, b_f, alog, dtb, nw, 1e-6f, S, CS, NKH, NVH, KD, VD, st_cpu_ch,
                      out_cpu_ch);

    DevVec<float> d_st_ch(st_init), d_out_ch(S * NVH * VD);
    qwen38_cuda_gdn_chunk_prefill_f32(d_q.ptr, d_k.ptr, d_v.ptr, d_z.ptr, d_a.ptr, d_b.ptr, d_al.ptr, d_dt.ptr,
                                      d_nw.ptr, 1e-6f, S, CS, NKH, NVH, KD, VD, d_st_ch.ptr, d_ws.ptr, d_out_ch.ptr);
    CUDA_OK(cudaDeviceSynchronize());
    const float d_ch_o = max_abs(d_out_ch.download(), out_cpu_ch);
    const float d_ch_s = max_abs(d_st_ch.download(), st_cpu_ch);
    record_exec("unit_gdn_chunk_prefill", "raw_f32_out <= 1e-4 && state <= 2e-4", d_ch_o, d_ch_s, 0.0,
                d_ch_o <= 1e-4f && d_ch_s <= 2e-4f, "unit_operators");
  }

  // 9. full_attn_gated (27B 24H / 4KVH, head_dim=256, rotary=64, prefill 8 + decode step 1 at pos=8).
  {
    constexpr int S_PF = 8, S_DEC = 1, NH = 24, NKVH = 4, HD = 256, RD = 64;
    const auto qg = rand_vec((S_PF + S_DEC) * NH * HD * 2, 0.35f);
    const auto kr = rand_vec((S_PF + S_DEC) * NKVH * HD, 0.35f);
    const auto vr = rand_vec((S_PF + S_DEC) * NKVH * HD, 0.35f);
    const auto qnw_b = to_bf16_nv(rand_vec(HD, 0.2f));
    const auto knw_b = to_bf16_nv(rand_vec(HD, 0.2f));
    const auto qnw = from_bf16_nv(qnw_b), knw = from_bf16_nv(knw_b);

    std::vector<float> kc_cpu, vc_cpu, out_cpu((S_PF + S_DEC) * NH * HD);
    for (int t = 0; t < S_PF + S_DEC; ++t) {
      full_attn_gated_step(std::span<const float>(qg.data() + t * NH * HD * 2, NH * HD * 2),
                           std::span<const float>(kr.data() + t * NKVH * HD, NKVH * HD),
                           std::span<const float>(vr.data() + t * NKVH * HD, NKVH * HD), qnw, knw, 1e-6f, 10000000.0f,
                           NH, NKVH, HD, RD, t, kc_cpu, vc_cpu,
                           std::span<float>(out_cpu.data() + t * NH * HD, NH * HD));
    }

    DevVec<float> d_qg(qg), d_kr(kr), d_vr(vr), d_kc((S_PF + S_DEC) * NKVH * HD), d_vc((S_PF + S_DEC) * NKVH * HD),
        d_qtmp((S_PF + S_DEC) * NH * HD), d_gtmp((S_PF + S_DEC) * NH * HD), d_out((S_PF + S_DEC) * NH * HD);
    DevVec<__nv_bfloat16> d_qnw(qnw_b), d_knw(knw_b);
    qwen38_cuda_full_attn_gated_f32(d_qg.ptr, d_kr.ptr, d_vr.ptr, d_qnw.ptr, d_knw.ptr, 1e-6f, 10000000.0f, S_PF, 0,
                                    NH, NKVH, HD, RD, d_kc.ptr, d_vc.ptr, S_PF + S_DEC, d_qtmp.ptr, d_gtmp.ptr,
                                    d_out.ptr);
    qwen38_cuda_full_attn_gated_f32(d_qg.ptr + S_PF * NH * HD * 2, d_kr.ptr + S_PF * NKVH * HD,
                                    d_vr.ptr + S_PF * NKVH * HD, d_qnw.ptr, d_knw.ptr, 1e-6f, 10000000.0f, S_DEC, S_PF,
                                    NH, NKVH, HD, RD, d_kc.ptr, d_vc.ptr, S_PF + S_DEC, d_qtmp.ptr, d_gtmp.ptr,
                                    d_out.ptr + S_PF * NH * HD);
    CUDA_OK(cudaDeviceSynchronize());
    const float d_fa = max_abs(d_out.download(), out_cpu);
    record_exec("unit_full_attn_gated", "raw_f32_max_abs <= 1e-4", d_fa, 0.0f, 0.0, d_fa <= 1e-4f, "unit_operators");
  }
  timings.unit_operators_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_unit0).count();
  std::printf("[phase] done=unit_operators elapsed_ms=%.1f wall_ms=%.1f\n", timings.unit_operators_ms, wall_ms());
  if (auto rc = check_early_stop("unit_operators")) return *rc;

  auto run_block_check = [&](const std::string& tag, const std::string& category, const Qwen38BlockBundleHost& bundle,
                             bool chunk_prefill, bool calibrated_fp8, Qwen38BlockParityTier tier,
                             bool run_matched_warp, double* fp64_ms_acc, double* warp_ms_acc, double* gpu_ms_acc,
                             const std::string& phase_label) {
    const auto cpu_bf16 = run_qwen38_blocks_cpu(bundle, true, chunk_prefill, 64, calibrated_fp8, 0, false);
    if (fp64_ms_acc) *fp64_ms_acc += cpu_bf16.elapsed_ms;
    const auto gpu = run_qwen38_blocks_gpu(bundle, chunk_prefill, 64, calibrated_fp8);
    if (gpu_ms_acc) *gpu_ms_acc += gpu.elapsed_ms;
    peak_vram_bytes = std::max(peak_vram_bytes, gpu.peak_vram_bytes);
    const auto p = evaluate_block_parity(bundle, gpu, cpu_bf16, tier);

    std::optional<MatchedWarpMetrics> mw_opt;
    bool warp_ok = true;
    if (run_matched_warp) {
      const auto cpu_warp = run_qwen38_blocks_cpu(bundle, true, chunk_prefill, 64, calibrated_fp8, 0, true);
      if (warp_ms_acc) *warp_ms_acc += cpu_warp.elapsed_ms;
      const auto pw = evaluate_block_parity(bundle, gpu, cpu_warp, tier);
      MatchedWarpMetrics mw;
      mw.hidden_max_abs = pw.hidden.max_abs;
      mw.hidden_rel_l2 = pw.hidden.rel_l2;
      mw.hidden_cosine = pw.hidden.cosine_sim;
      mw.layer0_gdn_max_abs = pw.layer0_gdn_max_abs;
      mw.stacked_gdn_max_abs = pw.stacked_gdn_max_abs;
      mw.conv_max_abs = std::max(pw.layer0_conv_max_abs, pw.stacked_conv_max_abs);
      mw.conv_rel_l2 = std::max(pw.layer0_conv_rel_l2, pw.stacked_conv_rel_l2);
      mw.k_cache_max_abs = pw.k_cache_max_abs;
      mw.k_cache_rel_l2 = pw.k_cache_rel_l2;
      mw.v_cache_max_abs = pw.v_cache_max_abs;
      mw.v_cache_rel_l2 = pw.v_cache_rel_l2;
      if (tier == Qwen38BlockParityTier::RealSingleLayer) {
        mw.criterion =
            "matched_warp_1L: triangle-safe 2x real_1L (l0_gdn <= 4e-4 && hidden/conv/k/v rel_l2 <= 4e-3, cos >= "
            "0.99998, rel_max <= 2e-2)";
        mw.passed = pw.state_shapes_valid && pw.hidden.finite && pw.conv_finite && pw.gdn_finite && pw.kv_finite &&
                    pw.layer0_gdn_max_abs <= 4e-4f &&
                    pw.hidden.rel_l2 <= 4e-3f && pw.hidden.cosine_sim >= 0.99998 && pw.hidden.rel_max_abs <= 2e-2f &&
                    pw.layer0_conv_rel_l2 <= 4e-3f && pw.layer0_conv_cosine >= 0.99998 &&
                    pw.layer0_conv_rel_max_abs <= 2e-2f &&
                    pw.k_cache_rel_l2 <= 4e-3f && pw.k_cache_cosine >= 0.99998 && pw.k_cache_rel_max_abs <= 2e-2f &&
                    pw.v_cache_rel_l2 <= 4e-3f && pw.v_cache_cosine >= 0.99998 && pw.v_cache_rel_max_abs <= 2e-2f;
      } else {
        mw.criterion = pw.criterion;
        mw.passed = pw.passed;
      }
      warp_ok = mw.passed;
      mw_opt = std::move(mw);
    }

    BlockStateMetrics sm;
    sm.state_shapes_valid = p.state_shapes_valid;
    sm.conv_finite = p.conv_finite;
    sm.gdn_finite = p.gdn_finite;
    sm.kv_finite = p.kv_finite;
    sm.layer0_conv_max_abs = p.layer0_conv_max_abs;
    sm.layer0_conv_rel_l2 = p.layer0_conv_rel_l2;
    sm.layer0_conv_rel_max_abs = p.layer0_conv_rel_max_abs;
    sm.layer0_conv_cosine = p.layer0_conv_cosine;
    sm.stacked_conv_max_abs = p.stacked_conv_max_abs;
    sm.stacked_conv_rel_l2 = p.stacked_conv_rel_l2;
    sm.stacked_conv_rel_max_abs = p.stacked_conv_rel_max_abs;
    sm.stacked_conv_cosine = p.stacked_conv_cosine;
    sm.k_cache_max_abs = p.k_cache_max_abs;
    sm.k_cache_rel_l2 = p.k_cache_rel_l2;
    sm.k_cache_rel_max_abs = p.k_cache_rel_max_abs;
    sm.k_cache_cosine = p.k_cache_cosine;
    sm.v_cache_max_abs = p.v_cache_max_abs;
    sm.v_cache_rel_l2 = p.v_cache_rel_l2;
    sm.v_cache_rel_max_abs = p.v_cache_rel_max_abs;
    sm.v_cache_cosine = p.v_cache_cosine;

    const bool ok = p.passed && warp_ok;
    Gate1Record r;
    r.name = tag;
    r.category = category;
    r.status = ok ? "pass" : "fail";
    r.criterion = p.criterion;
    r.max_abs_diff = p.hidden.max_abs;
    r.aux_max_abs = std::max(p.layer0_gdn_max_abs, p.stacked_gdn_max_abs);
    r.ref_max_abs = p.hidden.ref_max_abs;
    r.rel_max_abs = p.hidden.rel_max_abs;
    r.rel_l2 = p.hidden.rel_l2;
    r.cosine_sim = p.hidden.cosine_sim;
    r.layer0_gdn_max_abs = p.layer0_gdn_max_abs;
    r.stacked_gdn_rel_l2 = p.stacked_gdn_rel_l2;
    r.max_bf16_ulp = p.hidden_bf16.max_bf16_ulp;
    r.max_bf16_ulp_above_floor = p.hidden_bf16.max_bf16_ulp_above_floor;
    r.floor_exempt_count = p.hidden_bf16.floor_exempt_count;
    r.state_metrics = sm;
    r.matched_warp_f32 = mw_opt;
    r.elapsed_ms = gpu.elapsed_ms;
    r.passed = ok;
    records.push_back(std::move(r));

    if (mw_opt.has_value()) {
      std::printf(
          "  [%s] rel_l2=%.6g cos=%.8f l0_gdn=%.6f conv_rel_l2=%.6g k_rel_l2=%.6g v_rel_l2=%.6g "
          "warp_rel_l2=%.6g warp_v_rel_l2=%.6g raw_ulp=%u exempt=%zu gpu_ms=%.2f cpu_ms=%.1f vram=%.1f MiB "
          "wall_ms=%.1f -> %s\n",
          tag.c_str(), p.hidden.rel_l2, p.hidden.cosine_sim, p.layer0_gdn_max_abs,
          std::max(p.layer0_conv_rel_l2, p.stacked_conv_rel_l2), p.k_cache_rel_l2, p.v_cache_rel_l2,
          mw_opt->hidden_rel_l2, mw_opt->v_cache_rel_l2, p.hidden_bf16.max_bf16_ulp,
          p.hidden_bf16.floor_exempt_count, gpu.elapsed_ms, cpu_bf16.elapsed_ms,
          static_cast<double>(gpu.peak_vram_bytes) / (1024.0 * 1024.0), wall_ms(), ok ? "PASS" : "FAIL");
    } else {
      std::printf(
          "  [%s] rel_l2=%.6g cos=%.8f l0_gdn=%.6f conv_rel_l2=%.6g k_rel_l2=%.6g v_rel_l2=%.6g "
          "raw_ulp=%u exempt=%zu gpu_ms=%.2f cpu_ms=%.1f vram=%.1f MiB wall_ms=%.1f -> %s\n",
          tag.c_str(), p.hidden.rel_l2, p.hidden.cosine_sim, p.layer0_gdn_max_abs,
          std::max(p.layer0_conv_rel_l2, p.stacked_conv_rel_l2), p.k_cache_rel_l2, p.v_cache_rel_l2,
          p.hidden_bf16.max_bf16_ulp, p.hidden_bf16.floor_exempt_count, gpu.elapsed_ms, cpu_bf16.elapsed_ms,
          static_cast<double>(gpu.peak_vram_bytes) / (1024.0 * 1024.0), wall_ms(), ok ? "PASS" : "FAIL");
    }
    checkpoint_json(phase_label);
  };

  auto record_non_executed_group = [&](std::initializer_list<const char*> names, const std::string& category,
                                       const std::string& status, const std::string& reason,
                                       const std::string& phase_label) {
    for (const char* nm : names) {
      Gate1Record r;
      r.name = nm;
      r.category = category;
      r.status = status;
      r.criterion = reason;
      r.passed = false;
      records.push_back(std::move(r));
      std::printf("  [%s] status=%s (%s)\n", nm, status.c_str(), reason.c_str());
    }
    checkpoint_json(phase_label);
  };

  // Phase 2A: Synthetic 1-layer and 4-layer blocks (strict tolerance category).
  const auto t_synth0 = std::chrono::steady_clock::now();
  std::printf("[phase] start=synthetic_blocks wall_ms=%.1f\n", wall_ms());
  checkpoint_json("synthetic_blocks");
  {
    const std::array<int, 4> l4 = {0, 1, 2, 3};
    const std::array<std::int32_t, 6> toks = {5, 11, 23, 37, 2, 49};
    for (Qwen38WeightPrecision prec : {Qwen38WeightPrecision::Bf16, Qwen38WeightPrecision::MixedFp8Nvfp4}) {
      const auto b4 = make_synthetic_qwen38_block_bundle(prec, l4, toks, 777u);
      const std::string ptag = prec == Qwen38WeightPrecision::Bf16 ? "synth_bf16" : "synth_nvfp4_fp8";
      run_block_check(ptag + "_1L_gdn", "strict_synthetic_block", slice_qwen38_block_layer(b4, 0), true, true,
                      Qwen38BlockParityTier::SyntheticStrict, false, nullptr, nullptr, nullptr, "synthetic_blocks");
      run_block_check(ptag + "_1L_full", "strict_synthetic_block", slice_qwen38_block_layer(b4, 3), false, true,
                      Qwen38BlockParityTier::SyntheticStrict, false, nullptr, nullptr, nullptr, "synthetic_blocks");
      run_block_check(ptag + "_4L_stepwise", "strict_synthetic_block", b4, false, true,
                      Qwen38BlockParityTier::SyntheticStrict, false, nullptr, nullptr, nullptr, "synthetic_blocks");
      run_block_check(ptag + "_4L_chunk", "strict_synthetic_block", b4, true, true,
                      Qwen38BlockParityTier::SyntheticStrict, false, nullptr, nullptr, nullptr, "synthetic_blocks");
    }
  }
  timings.synthetic_blocks_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_synth0).count();
  std::printf("[phase] done=synthetic_blocks elapsed_ms=%.1f wall_ms=%.1f\n", timings.synthetic_blocks_ms, wall_ms());
  if (auto rc = check_early_stop("synthetic_blocks")) return *rc;

  // Phase 2B: Real Qwen3.5-0.8B (BF16) 1L & 4L blocks (single 4L stream load + 1L slices).
  if (skip_08b) {
    record_non_executed_group({"qwen08b_1L_gdn", "qwen08b_1L_full", "qwen08b_4L_stepwise", "qwen08b_4L_chunk"},
                              "real_block_envelope", "skipped_diagnostic", "--skip-08b specified in diagnostic mode",
                              "qwen08b_blocks");
  } else if (!std::filesystem::is_directory(q08_dir)) {
    record_non_executed_group({"qwen08b_1L_gdn", "qwen08b_1L_full", "qwen08b_4L_stepwise", "qwen08b_4L_chunk"},
                              "real_block_envelope", "missing_checkpoint", "missing qwen08b directory: " + q08_dir,
                              "qwen08b_blocks");
  } else {
    std::printf("[phase] start=qwen08b_inspect wall_ms=%.1f\n", wall_ms());
    checkpoint_json("qwen08b_inspect");
    const auto t_i08_0 = std::chrono::steady_clock::now();
    const auto m08 = Qwen38Manifest::inspect(q08_dir, false, "local");
    timings.qwen08b_inspect_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_i08_0).count();
    std::printf("[phase] done=qwen08b_inspect elapsed_ms=%.1f wall_ms=%.1f\n", timings.qwen08b_inspect_ms, wall_ms());
    if (auto rc = check_early_stop("qwen08b_inspect")) return *rc;

    const std::array<std::int32_t, 6> toks = {9707, 15, 220, 16, 17, 198};
    const std::array<int, 4> l4 = {0, 1, 2, 3};
    std::printf("[phase] start=qwen08b_load wall_ms=%.1f\n", wall_ms());
    checkpoint_json("qwen08b_load");
    const auto t_l08_0 = std::chrono::steady_clock::now();
    const auto b08_4l = load_qwen38_block_bundle(q08_dir, m08, l4, toks);
    timings.qwen08b_load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_l08_0).count();
    std::printf("[phase] done=qwen08b_load elapsed_ms=%.1f wall_ms=%.1f\n", timings.qwen08b_load_ms, wall_ms());
    if (auto rc = check_early_stop("qwen08b_load")) return *rc;

    std::printf("[phase] start=qwen08b_blocks wall_ms=%.1f\n", wall_ms());
    checkpoint_json("qwen08b_blocks");
    const auto b08_l0 = slice_qwen38_block_layer(b08_4l, 0);
    const auto b08_l3 = slice_qwen38_block_layer(b08_4l, 3);
    struct Job08 {
      const char* name;
      const Qwen38BlockBundleHost* bundle;
      bool chunk;
      Qwen38BlockParityTier tier;
    };
    const std::array<Job08, 4> jobs08 = {{
        {"qwen08b_1L_gdn", &b08_l0, true, Qwen38BlockParityTier::RealSingleLayer},
        {"qwen08b_1L_full", &b08_l3, false, Qwen38BlockParityTier::RealSingleLayer},
        {"qwen08b_4L_stepwise", &b08_4l, false, Qwen38BlockParityTier::RealStacked4Layer},
        {"qwen08b_4L_chunk", &b08_4l, true, Qwen38BlockParityTier::RealStacked4Layer},
    }};
    for (const auto& job : jobs08) {
      if (auto rc = check_early_stop("qwen08b_blocks")) return *rc;
      run_block_check(job.name, "real_block_envelope", *job.bundle, job.chunk, false, job.tier, true,
                      &timings.qwen08b_cpu_fp64_ms, &timings.qwen08b_cpu_warp_ms, &timings.qwen08b_gpu_ms,
                      "qwen08b_blocks");
    }
    std::printf("[phase] done=qwen08b_blocks cpu_fp64_ms=%.1f cpu_warp_ms=%.1f gpu_ms=%.1f wall_ms=%.1f\n",
                timings.qwen08b_cpu_fp64_ms, timings.qwen08b_cpu_warp_ms, timings.qwen08b_gpu_ms, wall_ms());
  }
  if (auto rc = check_early_stop("qwen08b_blocks")) return *rc;

  // Phase 2C: Real Qwen3.8-27B-NVFP4 (MixedFp8Nvfp4) 1L & 4L blocks (single pinned 4L stream load + slices).
  if (skip_27b) {
    record_non_executed_group({"qwen27b_1L_gdn", "qwen27b_1L_full", "qwen27b_4L_stepwise", "qwen27b_4L_chunk"},
                              "real_block_envelope", "skipped_diagnostic", "--skip-27b specified in diagnostic mode",
                              "qwen27b_blocks");
  } else if (!std::filesystem::is_directory(q27_dir)) {
    record_non_executed_group({"qwen27b_1L_gdn", "qwen27b_1L_full", "qwen27b_4L_stepwise", "qwen27b_4L_chunk"},
                              "real_block_envelope", "missing_checkpoint", "missing qwen27b directory: " + q27_dir,
                              "qwen27b_blocks");
  } else {
    std::printf("[phase] start=qwen27b_manifest_sha wall_ms=%.1f\n", wall_ms());
    checkpoint_json("qwen27b_manifest_sha");
    const auto t_sha0 = std::chrono::steady_clock::now();
    const auto m27 = verify_pins ? Qwen38Manifest::inspect_pinned(q27_dir, qwen38_27b_nvfp4_pin_spec(), "482ca0f")
                                 : Qwen38Manifest::inspect(q27_dir, false, "482ca0f");
    timings.qwen27b_manifest_sha_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sha0).count();
    std::printf("[phase] done=qwen27b_manifest_sha elapsed_ms=%.1f wall_ms=%.1f\n",
                timings.qwen27b_manifest_sha_ms, wall_ms());
    if (auto rc = check_early_stop("qwen27b_manifest_sha")) return *rc;

    q27_pins_verified_actual =
        verify_pins && m27.checkpoint.verification_level == ingine::model::VerificationLevel::PinnedVerified &&
        m27.checkpoint.shard_digests_verified;

    const std::array<std::int32_t, 4> toks = {9707, 15, 220, 16};
    const std::array<int, 4> l4 = {0, 1, 2, 3};
    std::printf("[phase] start=qwen27b_stream_load wall_ms=%.1f\n", wall_ms());
    checkpoint_json("qwen27b_stream_load");
    const auto t_ld0 = std::chrono::steady_clock::now();
    const auto b27_4l = load_qwen38_block_bundle(q27_dir, m27, l4, toks);
    timings.qwen27b_stream_load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ld0).count();
    std::printf("[phase] done=qwen27b_stream_load elapsed_ms=%.1f wall_ms=%.1f\n",
                timings.qwen27b_stream_load_ms, wall_ms());
    if (auto rc = check_early_stop("qwen27b_stream_load")) return *rc;

    std::printf("[phase] start=qwen27b_blocks wall_ms=%.1f\n", wall_ms());
    checkpoint_json("qwen27b_blocks");
    const auto b27_l0 = slice_qwen38_block_layer(b27_4l, 0);
    const auto b27_l3 = slice_qwen38_block_layer(b27_4l, 3);
    struct Job27 {
      const char* name;
      const Qwen38BlockBundleHost* bundle;
      bool chunk;
      Qwen38BlockParityTier tier;
    };
    const std::array<Job27, 4> jobs27 = {{
        {"qwen27b_1L_gdn", &b27_l0, true, Qwen38BlockParityTier::RealSingleLayer},
        {"qwen27b_1L_full", &b27_l3, false, Qwen38BlockParityTier::RealSingleLayer},
        {"qwen27b_4L_stepwise", &b27_4l, false, Qwen38BlockParityTier::RealStacked4Layer},
        {"qwen27b_4L_chunk", &b27_4l, true, Qwen38BlockParityTier::RealStacked4Layer},
    }};
    for (const auto& job : jobs27) {
      if (auto rc = check_early_stop("qwen27b_blocks")) return *rc;
      run_block_check(job.name, "real_block_envelope", *job.bundle, job.chunk, true, job.tier, true,
                      &timings.qwen27b_cpu_fp64_ms, &timings.qwen27b_cpu_warp_ms, &timings.qwen27b_gpu_ms,
                      "qwen27b_blocks");
    }
    std::printf("[phase] done=qwen27b_blocks cpu_fp64_ms=%.1f cpu_warp_ms=%.1f gpu_ms=%.1f wall_ms=%.1f\n",
                timings.qwen27b_cpu_fp64_ms, timings.qwen27b_cpu_warp_ms, timings.qwen27b_gpu_ms, wall_ms());
  }

  Qwen38Gate1CoverageSummary summary;
  const auto final_doc = build_json(false, "complete", false, 0, &summary);
  if (!json_out.empty() && !write_json_atomic(json_out, final_doc)) {
    std::fprintf(stderr, "failed to write final atomic json output path: %s\n", json_out.c_str());
    return 1;
  }

  if (diagnostic_mode) {
    std::printf("qwen38_gpu_gate1: DIAGNOSTIC (INCOMPLETE - NOT GATE 1 QUALIFIED) "
                "status=%s (%zu/%zu required passed, missing=%zu, failed=%zu, peak_vram=%.1f MiB, wall_ms=%.1f)\n",
                summary.coverage_status.c_str(), summary.passed_required_row_count, summary.required_row_count,
                summary.missing_required_rows.size(), summary.failed_rows.size(),
                static_cast<double>(peak_vram_bytes) / (1024.0 * 1024.0), timings.total_wall_ms);
    return summary.executed_checks_passed ? 0 : 1;
  }

  std::printf("qwen38_gpu_gate1: %s (%zu/%zu required passed, missing=%zu, failed=%zu, pins=%s, peak_vram=%.1f MiB, "
              "wall_ms=%.1f)\n",
              summary.qualification_passed ? "PASS" : "FAIL", summary.passed_required_row_count,
              summary.required_row_count, summary.missing_required_rows.size(), summary.failed_rows.size(),
              summary.qwen27b_pins_verified ? "verified" : "UNVERIFIED",
              static_cast<double>(peak_vram_bytes) / (1024.0 * 1024.0), timings.total_wall_ms);
  return summary.qualification_passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) { return main_impl(argc, argv); }

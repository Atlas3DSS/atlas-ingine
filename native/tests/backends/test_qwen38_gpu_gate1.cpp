#include "backends/qwen38/qwen38_block_cpu.h"
#include "backends/qwen38/qwen38_cpu_ops.h"
#include "backends/qwen38/qwen38_cuda_ops.h"
#include "backends/qwen38/qwen38_manifest.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <sys/resource.h>
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

struct Gate1Record {
  std::string name;
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
  std::optional<std::size_t> bf16_vs_gpu_f32_bit_mismatches;
  double elapsed_ms = 0.0;
  bool passed = false;
};

int main_impl(int argc, char** argv) {
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

  CUDA_OK(cudaSetDevice(0));
  cudaDeviceProp props{};
  CUDA_OK(cudaGetDeviceProperties(&props, 0));
  std::printf("Gate 1 running [%s] on %s (sm_%d%d, vram=%.2f GiB, verify_pins=%s)\n",
              diagnostic_mode ? "DIAGNOSTIC" : "QUALIFICATION", props.name, props.major, props.minor,
              static_cast<double>(props.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0),
              verify_pins ? "true" : "false");

  std::vector<Gate1Record> records;
  std::uint64_t peak_vram_bytes = 0;

  auto record_exec = [&](const std::string& name, const std::string& criterion, float max_d, float aux_d,
                         double ms, bool ok,
                         std::optional<std::uint32_t> max_ulp = std::nullopt,
                         std::optional<std::size_t> f32_bf16_mm = std::nullopt) {
    Gate1Record r;
    r.name = name;
    r.status = ok ? "pass" : "fail";
    r.criterion = criterion;
    r.max_abs_diff = max_d;
    r.aux_max_abs = aux_d;
    r.max_bf16_ulp = max_ulp;
    r.bf16_vs_gpu_f32_bit_mismatches = f32_bf16_mm;
    r.elapsed_ms = ms;
    r.passed = ok;
    records.push_back(std::move(r));
    std::printf("  [%s] max_abs=%.6g aux=%.6g %s\n", name.c_str(), max_d, aux_d, ok ? "PASS" : "FAIL");
  };

  // Part 1.1 & 1.2: Shared Gate 1 NVFP4 + FP8 linear unit checks (both F32 oracle and rounded BF16 oracle).
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

      record_exec("unit_nvfp4_linear_f32", "raw_f32_max_abs <= 1e-3", d_f32, 0.0f, 0.0,
                  d_f32 <= 1e-3f);
      record_exec("unit_nvfp4_linear_bf16", "bf16_oracle_ulp <= 1 || abs <= 1e-3; exact_gpu_f32_to_bf16 == 0",
                  cmp.max_abs_vs_bf16_oracle, static_cast<float>(cmp.criterion_violations), 0.0,
                  mm == 0 && cmp.passed, cmp.max_bf16_ulp, mm);
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
                  d_wo_f32 <= 1e-3f);
      record_exec("unit_fp8_linear_weight_only_bf16",
                  "bf16_oracle_ulp <= 1 || abs <= 1e-3; exact_gpu_f32_to_bf16 == 0", cmp_wo.max_abs_vs_bf16_oracle,
                  static_cast<float>(cmp_wo.criterion_violations), 0.0, mm_wo == 0 && cmp_wo.passed,
                  cmp_wo.max_bf16_ulp, mm_wo);

      const auto cal_f32 = d_out_cal_f32.download();
      const auto cal_bits = to_bf16_bits(d_out_cal_bf16.download());
      const float d_cal_f32 = max_abs(cal_f32, lin.fp8_cal_ref_f32);
      const std::size_t mm_cal = count_bit_mismatches(cal_bits, cal_f32);
      const auto cmp_cal = compare_bf16_against_rounded_oracle(cal_bits, lin.fp8_cal_ref_f32, 1u, 1e-3f);
      record_exec("unit_fp8_linear_calibrated_f32", "raw_f32_max_abs <= 1e-3", d_cal_f32, 0.0f, 0.0,
                  d_cal_f32 <= 1e-3f);
      record_exec("unit_fp8_linear_calibrated_bf16",
                  "bf16_oracle_ulp <= 1 || abs <= 1e-3; exact_gpu_f32_to_bf16 == 0", cmp_cal.max_abs_vs_bf16_oracle,
                  static_cast<float>(cmp_cal.criterion_violations), 0.0, mm_cal == 0 && cmp_cal.passed,
                  cmp_cal.max_bf16_ulp, mm_cal);
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
    record_exec("unit_rmsnorm_zero_centered", "raw_f32_max_abs <= 1e-5", d_rms, 0.0f, 0.0, d_rms <= 1e-5f);
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
    record_exec("unit_rmsnorm_gated_silu", "raw_f32_max_abs <= 1e-5", d_gr, 0.0f, 0.0, d_gr <= 1e-5f);
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
                slice_exact ? 0.0f : 1.0f, 0.0, d_rope <= 1e-5f && slice_exact);
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
                d_out_v <= 1e-5f && d_st_v <= 1e-6f);
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
                d_step_o <= 1e-4f && d_step_s <= 2e-4f);

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
                d_ch_o <= 1e-4f && d_ch_s <= 2e-4f);
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
    record_exec("unit_full_attn_gated", "raw_f32_max_abs <= 1e-4", d_fa, 0.0f, 0.0, d_fa <= 1e-4f);
  }

  auto run_block_check = [&](const std::string& tag, const Qwen38BlockBundleHost& bundle, bool chunk_prefill,
                             bool calibrated_fp8, Qwen38BlockParityTier tier) {
    const auto cpu_bf16 = run_qwen38_blocks_cpu(bundle, true, chunk_prefill, 64, calibrated_fp8);
    const auto gpu = run_qwen38_blocks_gpu(bundle, chunk_prefill, 64, calibrated_fp8);
    peak_vram_bytes = std::max(peak_vram_bytes, gpu.peak_vram_bytes);
    const auto p = evaluate_block_parity(bundle, gpu, cpu_bf16, tier);

    Gate1Record r;
    r.name = tag;
    r.status = p.passed ? "pass" : "fail";
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
    r.elapsed_ms = gpu.elapsed_ms;
    r.passed = p.passed;
    records.push_back(std::move(r));

    std::printf(
        "  [%s] max_abs=%.6f ref_max=%.6f rel_l2=%.6g cos=%.8f l0_gdn=%.6f stacked_gdn=%.6f gpu_ms=%.2f cpu_ms=%.1f "
        "vram=%.1f MiB -> %s\n",
        tag.c_str(), p.hidden.max_abs, p.hidden.ref_max_abs, p.hidden.rel_l2, p.hidden.cosine_sim, p.layer0_gdn_max_abs,
        p.stacked_gdn_max_abs, gpu.elapsed_ms, cpu_bf16.elapsed_ms,
        static_cast<double>(gpu.peak_vram_bytes) / (1024.0 * 1024.0), p.passed ? "PASS" : "FAIL");
  };

  auto record_non_executed_group = [&](std::initializer_list<const char*> names, const std::string& status,
                                       const std::string& reason) {
    for (const char* nm : names) {
      Gate1Record r;
      r.name = nm;
      r.status = status;
      r.criterion = reason;
      r.passed = false;
      records.push_back(std::move(r));
      std::printf("  [%s] status=%s (%s)\n", nm, status.c_str(), reason.c_str());
    }
  };

  // Part 2A: Synthetic 1-layer and 4-layer blocks (BF16 & MixedFp8Nvfp4).
  {
    const std::array<int, 4> l4 = {0, 1, 2, 3};
    const std::array<std::int32_t, 6> toks = {5, 11, 23, 37, 2, 49};
    for (Qwen38WeightPrecision prec : {Qwen38WeightPrecision::Bf16, Qwen38WeightPrecision::MixedFp8Nvfp4}) {
      const auto b4 = make_synthetic_qwen38_block_bundle(prec, l4, toks, 777u);
      const std::string ptag = prec == Qwen38WeightPrecision::Bf16 ? "synth_bf16" : "synth_nvfp4_fp8";
      run_block_check(ptag + "_1L_gdn", slice_qwen38_block_layer(b4, 0), true, true,
                      Qwen38BlockParityTier::SyntheticStrict);
      run_block_check(ptag + "_1L_full", slice_qwen38_block_layer(b4, 3), false, true,
                      Qwen38BlockParityTier::SyntheticStrict);
      run_block_check(ptag + "_4L_stepwise", b4, false, true, Qwen38BlockParityTier::SyntheticStrict);
      run_block_check(ptag + "_4L_chunk", b4, true, true, Qwen38BlockParityTier::SyntheticStrict);
    }
  }

  // Part 2B: Real Qwen3.5-0.8B (BF16) 1-layer & 4-layer blocks (single 4L stream load + 1L slices).
  if (skip_08b) {
    record_non_executed_group({"qwen08b_1L_gdn", "qwen08b_1L_full", "qwen08b_4L_stepwise", "qwen08b_4L_chunk"},
                              "skipped_diagnostic", "--skip-08b specified in diagnostic mode");
  } else if (!std::filesystem::is_directory(q08_dir)) {
    record_non_executed_group({"qwen08b_1L_gdn", "qwen08b_1L_full", "qwen08b_4L_stepwise", "qwen08b_4L_chunk"},
                              "missing_checkpoint", "missing qwen08b directory: " + q08_dir);
  } else {
    const auto m08 = Qwen38Manifest::inspect(q08_dir, false, "local");
    const std::array<std::int32_t, 6> toks = {9707, 15, 220, 16, 17, 198};
    const std::array<int, 4> l4 = {0, 1, 2, 3};
    const auto b08_4l = load_qwen38_block_bundle(q08_dir, m08, l4, toks);
    run_block_check("qwen08b_1L_gdn", slice_qwen38_block_layer(b08_4l, 0), true, false,
                    Qwen38BlockParityTier::RealSingleLayer);
    run_block_check("qwen08b_1L_full", slice_qwen38_block_layer(b08_4l, 3), false, false,
                    Qwen38BlockParityTier::RealSingleLayer);
    run_block_check("qwen08b_4L_stepwise", b08_4l, false, false, Qwen38BlockParityTier::RealStacked4Layer);
    run_block_check("qwen08b_4L_chunk", b08_4l, true, false, Qwen38BlockParityTier::RealStacked4Layer);
  }

  // Part 2C: Real Qwen3.8-27B-NVFP4 (MixedFp8Nvfp4) 1-layer & 4-layer blocks (single pinned 4L stream load + slices).
  double cold_manifest_sha_ms = 0.0;
  double cold_safetensors_load_ms = 0.0;
  bool q27_pins_verified_actual = false;
  if (skip_27b) {
    record_non_executed_group({"qwen27b_1L_gdn", "qwen27b_1L_full", "qwen27b_4L_stepwise", "qwen27b_4L_chunk"},
                              "skipped_diagnostic", "--skip-27b specified in diagnostic mode");
  } else if (!std::filesystem::is_directory(q27_dir)) {
    record_non_executed_group({"qwen27b_1L_gdn", "qwen27b_1L_full", "qwen27b_4L_stepwise", "qwen27b_4L_chunk"},
                              "missing_checkpoint", "missing qwen27b directory: " + q27_dir);
  } else {
    const auto t_sha0 = std::chrono::steady_clock::now();
    const auto m27 = verify_pins ? Qwen38Manifest::inspect_pinned(q27_dir, qwen38_27b_nvfp4_pin_spec(), "482ca0f")
                                 : Qwen38Manifest::inspect(q27_dir, false, "482ca0f");
    const auto t_sha1 = std::chrono::steady_clock::now();
    cold_manifest_sha_ms = std::chrono::duration<double, std::milli>(t_sha1 - t_sha0).count();
    q27_pins_verified_actual =
        verify_pins && m27.checkpoint.verification_level == ingine::model::VerificationLevel::PinnedVerified &&
        m27.checkpoint.shard_digests_verified;

    const std::array<std::int32_t, 4> toks = {9707, 15, 220, 16};
    const std::array<int, 4> l4 = {0, 1, 2, 3};
    const auto t_ld0 = std::chrono::steady_clock::now();
    const auto b27_4l = load_qwen38_block_bundle(q27_dir, m27, l4, toks);
    const auto t_ld1 = std::chrono::steady_clock::now();
    cold_safetensors_load_ms = std::chrono::duration<double, std::milli>(t_ld1 - t_ld0).count();

    run_block_check("qwen27b_1L_gdn", slice_qwen38_block_layer(b27_4l, 0), true, true,
                    Qwen38BlockParityTier::RealSingleLayer);
    run_block_check("qwen27b_1L_full", slice_qwen38_block_layer(b27_4l, 3), false, true,
                    Qwen38BlockParityTier::RealSingleLayer);
    run_block_check("qwen27b_4L_stepwise", b27_4l, false, true, Qwen38BlockParityTier::RealStacked4Layer);
    run_block_check("qwen27b_4L_chunk", b27_4l, true, true, Qwen38BlockParityTier::RealStacked4Layer);
  }

  std::vector<Qwen38Gate1RowStatus> statuses;
  statuses.reserve(records.size());
  for (const auto& r : records) statuses.push_back({r.name, r.status, r.passed});
  const auto summary = evaluate_gate1_coverage(statuses, diagnostic_mode, q27_pins_verified_actual);

  struct rusage ru {};
  std::uint64_t host_rss_bytes = 0;
  if (::getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_maxrss > 0) {
    host_rss_bytes = static_cast<std::uint64_t>(ru.ru_maxrss) * 1024ull;
  }

  if (!json_out.empty()) {
    nlohmann::ordered_json root;
    root["device"] = props.name;
    root["sm"] = std::to_string(props.major) + std::to_string(props.minor);
    root["mode"] = summary.mode;
    root["coverage_status"] = summary.coverage_status;
    root["complete_coverage"] = summary.complete_coverage;
    root["executed_checks_passed"] = summary.executed_checks_passed;
    root["qwen27b_pins_verified"] = summary.qwen27b_pins_verified;
    root["qualification_passed"] = summary.qualification_passed;
    root["all_passed"] = summary.all_passed;
    root["required_row_count"] = summary.required_row_count;
    root["passed_required_row_count"] = summary.passed_required_row_count;
    root["missing_required_rows"] = summary.missing_required_rows;
    root["failed_rows"] = summary.failed_rows;
    root["qwen08b_dir"] = q08_dir;
    root["qwen27b_dir"] = q27_dir;
    root["cold_manifest_sha_ms"] = cold_manifest_sha_ms;
    root["cold_safetensors_load_ms"] = cold_safetensors_load_ms;
    root["peak_vram_bytes"] = peak_vram_bytes;
    root["host_rss_bytes"] = host_rss_bytes;
    root["checks"] = nlohmann::json::array();
    for (const auto& r : records) {
      nlohmann::ordered_json j;
      j["name"] = r.name;
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
      if (r.bf16_vs_gpu_f32_bit_mismatches.has_value()) {
        j["bf16_vs_gpu_f32_bit_mismatches"] = *r.bf16_vs_gpu_f32_bit_mismatches;
      }
      j["elapsed_ms"] = r.elapsed_ms;
      j["passed"] = r.passed;
      root["checks"].push_back(std::move(j));
    }
    std::ofstream out(json_out, std::ios::binary);
    if (!out.good()) {
      std::fprintf(stderr, "failed to open json output path: %s\n", json_out.c_str());
      return 1;
    }
    out << root.dump(2) << '\n';
  }

  if (diagnostic_mode) {
    std::printf("qwen38_gpu_gate1: DIAGNOSTIC (INCOMPLETE - NOT GATE 1 QUALIFIED) "
                "status=%s (%zu/%zu required passed, missing=%zu, failed=%zu, peak_vram=%.1f MiB)\n",
                summary.coverage_status.c_str(), summary.passed_required_row_count, summary.required_row_count,
                summary.missing_required_rows.size(), summary.failed_rows.size(),
                static_cast<double>(peak_vram_bytes) / (1024.0 * 1024.0));
    return summary.executed_checks_passed ? 0 : 1;
  }

  std::printf("qwen38_gpu_gate1: %s (%zu/%zu required passed, missing=%zu, failed=%zu, pins=%s, peak_vram=%.1f MiB)\n",
              summary.qualification_passed ? "PASS" : "FAIL", summary.passed_required_row_count,
              summary.required_row_count, summary.missing_required_rows.size(), summary.failed_rows.size(),
              summary.qwen27b_pins_verified ? "verified" : "UNVERIFIED",
              static_cast<double>(peak_vram_bytes) / (1024.0 * 1024.0));
  return summary.qualification_passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) { return main_impl(argc, argv); }

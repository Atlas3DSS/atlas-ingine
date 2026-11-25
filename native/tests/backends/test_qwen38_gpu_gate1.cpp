#include "backends/qwen38/qwen38_block_cpu.h"
#include "backends/qwen38/qwen38_cpu_ops.h"
#include "backends/qwen38/qwen38_cuda_ops.h"
#include "backends/qwen38/qwen38_manifest.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

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

std::vector<__nv_bfloat16> to_bf16(std::span<const float> src) {
  std::vector<__nv_bfloat16> out(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    const std::uint16_t b = f32_to_bf16(src[i]);
    std::memcpy(&out[i], &b, 2);
  }
  return out;
}

std::vector<float> from_bf16(std::span<const __nv_bfloat16> src) {
  std::vector<float> out(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    std::uint16_t b = 0;
    std::memcpy(&b, &src[i], 2);
    out[i] = bf16_to_f32(b);
  }
  return out;
}

float max_abs(std::span<const float> a, std::span<const float> b) {
  CHECK(a.size() == b.size());
  float m = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
    m = std::max(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

struct Gate1Record {
  std::string name;
  float max_abs_diff = 0.0f;
  float aux_max_abs = 0.0f;
  double elapsed_ms = 0.0;
  bool passed = false;
};

int main_impl(int argc, char** argv) {
  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
    std::printf("skip: no CUDA device visible\n");
    return 77;
  }

  std::string q08_dir = "/mnt/storage/models/Qwen3.5-0.8B";
  std::string q27_dir = "/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f";
  std::string json_out;
  bool verify_pins = false;
  bool skip_08b = false;
  bool skip_27b = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--qwen08-dir" && i + 1 < argc) q08_dir = argv[++i];
    else if (a == "--qwen27-dir" && i + 1 < argc) q27_dir = argv[++i];
    else if (a == "--json-out" && i + 1 < argc) json_out = argv[++i];
    else if (a == "--verify-pins") verify_pins = true;
    else if (a == "--skip-08b") skip_08b = true;
    else if (a == "--skip-27b") skip_27b = true;
  }

  CUDA_OK(cudaSetDevice(0));
  cudaDeviceProp props{};
  CUDA_OK(cudaGetDeviceProperties(&props, 0));
  std::printf("Gate 1 running on %s (sm_%d%d, vram=%.2f GiB)\n", props.name, props.major, props.minor,
              static_cast<double>(props.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

  std::mt19937 rng(20261125u);
  auto rand_vec = [&](std::size_t n, float stddev = 0.4f) {
    std::normal_distribution<float> d(0.0f, stddev);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
  };

  std::vector<Gate1Record> records;
  std::uint64_t peak_vram_bytes = 0;

  // 1. nvfp4_linear (27B K=5120 width).
  {
    constexpr int M = 4, N = 256, K = 5120;
    const auto x_f = rand_vec(M * K, 0.25f);
    const auto x_b = to_bf16(x_f);
    const auto x_q = from_bf16(x_b);
    std::vector<std::uint8_t> packed(N * (K / 2));
    std::vector<std::uint8_t> scales(N * (K / 16));
    for (auto& p : packed) p = static_cast<std::uint8_t>(rng() & 0xffu);
    for (auto& s : scales) s = static_cast<std::uint8_t>(0x38u + (rng() % 20));
    const float ws2 = 0.015f;

    std::vector<float> ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m) {
      nvfp4_weight_only_linear(std::span<const float>(x_q.data() + m * K, K), packed, scales, ws2, N, K,
                               std::span<float>(ref.data() + m * N, N));
    }
    DevVec<__nv_bfloat16> d_x(x_b), d_out(M * N);
    DevVec<std::uint8_t> d_p(packed), d_s(scales);
    qwen38_cuda_nvfp4_linear_bf16(d_x.ptr, d_p.ptr, d_s.ptr, ws2, M, N, K, d_out.ptr);
    CUDA_OK(cudaDeviceSynchronize());
    const float d_bf16 = max_abs(from_bf16(d_out.download()), ref);
    CHECK(d_bf16 <= 1e-3f);
    records.push_back({"unit_nvfp4_linear", d_bf16, 0.0f, 0.0, true});
  }

  // 2. fp8_linear (weight-only and calibrated input_scale > 0, K=5120).
  {
    constexpr int M = 4, N = 256, K = 5120;
    const auto x_f = rand_vec(M * K, 0.3f);
    const auto x_b = to_bf16(x_f);
    const auto x_q = from_bf16(x_b);
    std::vector<std::uint8_t> w(N * K);
    for (auto& b : w) b = quantize_e4m3fn_fast(static_cast<float>(static_cast<int>(rng() % 200) - 100) * 0.01f);
    const float ws = 0.025f, in_sc = 0.125f;

    std::vector<float> ref_cal(M * N);
    for (int m = 0; m < M; ++m) {
      fp8_calibrated_linear(std::span<const float>(x_q.data() + m * K, K), in_sc, w, ws, N, K,
                            std::span<float>(ref_cal.data() + m * N, N));
    }
    DevVec<__nv_bfloat16> d_x(x_b), d_out(M * N);
    DevVec<std::uint8_t> d_w(w);
    qwen38_cuda_fp8_linear_bf16(d_x.ptr, d_w.ptr, ws, in_sc, M, N, K, d_out.ptr);
    CUDA_OK(cudaDeviceSynchronize());
    const float d_cal = max_abs(from_bf16(d_out.download()), ref_cal);
    CHECK(d_cal <= 1e-3f);
    records.push_back({"unit_fp8_linear_calibrated", d_cal, 0.0f, 0.0, true});
  }

  // 3. rmsnorm_zero_centered (27B H=5120).
  {
    constexpr int M = 8, D = 5120;
    const auto x_f = rand_vec(M * D, 1.2f);
    const auto w_b = to_bf16(rand_vec(D, 0.15f));
    const auto w_f = from_bf16(w_b);
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
    CHECK(d_rms <= 1e-5f);
    records.push_back({"unit_rmsnorm_zero_centered", d_rms, 0.0f, 0.0, true});
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
    CHECK(d_gr <= 1e-5f);
    records.push_back({"unit_rmsnorm_gated_silu", d_gr, 0.0f, 0.0, true});
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
    CHECK(d_rope <= 1e-5f);
    for (int m = 0; m < M; ++m) {
      for (int h = 0; h < H; ++h) {
        for (int d = RD; d < HD; ++d) {
          const std::size_t idx = (static_cast<std::size_t>(m * H + h)) * HD + d;
          CHECK(got[idx] == x_f[idx]);
        }
      }
    }
    records.push_back({"unit_partial_rope_neox", d_rope, 0.0f, 0.0, true});
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
    CHECK(d_out_v <= 1e-5f && d_st_v <= 1e-6f);
    records.push_back({"unit_causal_conv1d_silu", d_out_v, d_st_v, 0.0, true});
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

    // Step 1 on CPU vs GPU recurrent_step
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
    CHECK(d_step_o <= 1e-4f && d_step_s <= 2e-4f);
    records.push_back({"unit_gdn_recurrent_step", d_step_o, d_step_s, 0.0, true});

    // Multi-chunk prefill (S=70, chunk_size=64) vs CPU gdn_chunk_prefill
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
    CHECK(d_ch_o <= 1e-4f && d_ch_s <= 2e-4f);
    records.push_back({"unit_gdn_chunk_prefill", d_ch_o, d_ch_s, 0.0, true});
  }

  // 9. full_attn_gated (27B 24H / 4KVH, head_dim=256, rotary=64, prefill 8 + decode step 1 at pos=8).
  {
    constexpr int S_PF = 8, S_DEC = 1, NH = 24, NKVH = 4, HD = 256, RD = 64;
    const auto qg = rand_vec((S_PF + S_DEC) * NH * HD * 2, 0.35f);
    const auto kr = rand_vec((S_PF + S_DEC) * NKVH * HD, 0.35f);
    const auto vr = rand_vec((S_PF + S_DEC) * NKVH * HD, 0.35f);
    const auto qnw_b = to_bf16(rand_vec(HD, 0.2f));
    const auto knw_b = to_bf16(rand_vec(HD, 0.2f));
    const auto qnw = from_bf16(qnw_b), knw = from_bf16(knw_b);

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
    // 8-token prefill
    qwen38_cuda_full_attn_gated_f32(d_qg.ptr, d_kr.ptr, d_vr.ptr, d_qnw.ptr, d_knw.ptr, 1e-6f, 10000000.0f, S_PF, 0,
                                    NH, NKVH, HD, RD, d_kc.ptr, d_vc.ptr, S_PF + S_DEC, d_qtmp.ptr, d_gtmp.ptr,
                                    d_out.ptr);
    // 1-token decode step at pos=8
    qwen38_cuda_full_attn_gated_f32(d_qg.ptr + S_PF * NH * HD * 2, d_kr.ptr + S_PF * NKVH * HD,
                                    d_vr.ptr + S_PF * NKVH * HD, d_qnw.ptr, d_knw.ptr, 1e-6f, 10000000.0f, S_DEC, S_PF,
                                    NH, NKVH, HD, RD, d_kc.ptr, d_vc.ptr, S_PF + S_DEC, d_qtmp.ptr, d_gtmp.ptr,
                                    d_out.ptr + S_PF * NH * HD);
    CUDA_OK(cudaDeviceSynchronize());
    const float d_fa = max_abs(d_out.download(), out_cpu);
    CHECK(d_fa <= 1e-4f);
    records.push_back({"unit_full_attn_gated", d_fa, 0.0f, 0.0, true});
  }

  auto run_block_check = [&](const std::string& tag, const Qwen38BlockBundleHost& bundle, bool chunk_prefill,
                             bool calibrated_fp8) {
    const auto cpu_bf16 = run_qwen38_blocks_cpu(bundle, true, chunk_prefill, 64, calibrated_fp8);
    const auto gpu = run_qwen38_blocks_gpu(bundle, chunk_prefill, 64, calibrated_fp8);
    peak_vram_bytes = std::max(peak_vram_bytes, gpu.peak_vram_bytes);
    const float d_h = max_abs(gpu.hidden_seq, cpu_bf16.hidden_seq);
    float d_gdn = 0.0f;
    for (std::size_t li = 0; li < bundle.layers.size(); ++li) {
      if (bundle.layers[li].kind == Qwen38LayerKind::LinearAttention) {
        d_gdn = std::max(d_gdn, max_abs(gpu.recurrent_states[li], cpu_bf16.recurrent_states[li]));
      }
    }
    const bool ok = (d_h <= 1e-3f && d_gdn <= 2e-4f);
    std::printf("  [%s] hidden_max_abs=%.6f gdn_state_max_abs=%.6f gpu_ms=%.2f cpu_ms=%.1f vram=%.1f MiB -> %s\n",
                tag.c_str(), d_h, d_gdn, gpu.elapsed_ms, cpu_bf16.elapsed_ms,
                static_cast<double>(gpu.peak_vram_bytes) / (1024.0 * 1024.0), ok ? "PASS" : "FAIL");
    CHECK(ok);
    records.push_back({tag, d_h, d_gdn, gpu.elapsed_ms, ok});
  };

  // Part 2A: Synthetic 1-layer and 4-layer blocks (BF16 & MixedFp8Nvfp4).
  {
    const std::array<int, 4> l4 = {0, 1, 2, 3};
    const std::array<std::int32_t, 6> toks = {5, 11, 23, 37, 2, 49};
    for (Qwen38WeightPrecision prec : {Qwen38WeightPrecision::Bf16, Qwen38WeightPrecision::MixedFp8Nvfp4}) {
      const auto b = make_synthetic_qwen38_block_bundle(prec, l4, toks, 777u);
      const std::string ptag = prec == Qwen38WeightPrecision::Bf16 ? "synth_bf16" : "synth_nvfp4_fp8";
      run_block_check(ptag + "_4L_stepwise", b, false, true);
      run_block_check(ptag + "_4L_chunk", b, true, true);
    }
  }

  // Part 2B: Real Qwen3.5-0.8B (BF16) 1-layer & 4-layer blocks.
  if (!skip_08b && std::filesystem::is_directory(q08_dir)) {
    const auto m08 = Qwen38Manifest::inspect(q08_dir, false, "local");
    const std::array<std::int32_t, 6> toks = {9707, 15, 220, 16, 17, 198};
    const std::array<int, 1> l0 = {0}, l3 = {3};
    const std::array<int, 4> l4 = {0, 1, 2, 3};
    run_block_check("qwen08b_1L_gdn", load_qwen38_block_bundle(q08_dir, m08, l0, toks), true, false);
    run_block_check("qwen08b_1L_full", load_qwen38_block_bundle(q08_dir, m08, l3, toks), false, false);
    run_block_check("qwen08b_4L_stepwise", load_qwen38_block_bundle(q08_dir, m08, l4, toks), false, false);
    run_block_check("qwen08b_4L_chunk", load_qwen38_block_bundle(q08_dir, m08, l4, toks), true, false);
  }

  // Part 2C: Real Qwen3.8-27B-NVFP4 (MixedFp8Nvfp4) 1-layer & 4-layer blocks.
  if (!skip_27b && std::filesystem::is_directory(q27_dir)) {
    const auto m27 = verify_pins ? Qwen38Manifest::inspect_pinned(q27_dir, qwen38_27b_nvfp4_pin_spec(), "482ca0f")
                                 : Qwen38Manifest::inspect(q27_dir, false, "482ca0f");
    const std::array<std::int32_t, 4> toks = {9707, 15, 220, 16};
    const std::array<int, 1> l0 = {0}, l3 = {3};
    const std::array<int, 4> l4 = {0, 1, 2, 3};
    run_block_check("qwen27b_1L_gdn", load_qwen38_block_bundle(q27_dir, m27, l0, toks), true, true);
    run_block_check("qwen27b_1L_full", load_qwen38_block_bundle(q27_dir, m27, l3, toks), false, true);
    run_block_check("qwen27b_4L_stepwise", load_qwen38_block_bundle(q27_dir, m27, l4, toks), false, true);
    run_block_check("qwen27b_4L_chunk", load_qwen38_block_bundle(q27_dir, m27, l4, toks), true, true);
  }

  if (!json_out.empty()) {
    nlohmann::ordered_json root;
    root["device"] = props.name;
    root["sm"] = std::to_string(props.major) + std::to_string(props.minor);
    root["peak_vram_bytes"] = peak_vram_bytes;
    root["all_passed"] = true;
    root["checks"] = nlohmann::json::array();
    for (const auto& r : records) {
      root["checks"].push_back({{"name", r.name},
                                {"max_abs", r.max_abs_diff},
                                {"aux_max_abs", r.aux_max_abs},
                                {"elapsed_ms", r.elapsed_ms},
                                {"passed", r.passed}});
    }
    std::ofstream out(json_out, std::ios::binary);
    CHECK(out.good());
    out << root.dump(2) << '\n';
  }

  std::printf("qwen38_gpu_gate1: PASS (%zu checks, peak_vram=%.1f MiB)\n", records.size(),
              static_cast<double>(peak_vram_bytes) / (1024.0 * 1024.0));
  return 0;
}

}  // namespace

int main(int argc, char** argv) { return main_impl(argc, argv); }

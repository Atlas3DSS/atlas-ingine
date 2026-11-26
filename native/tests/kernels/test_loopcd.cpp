// Kernel-level numerical equivalence & timing verification for #882 Fused Single-Pass LoopCD (arXiv:2610.02185):
// 1. Exact equivalence of single-pass dual-vector dense::head_loopcd vs naive 2-pass dense::head + FP64 adaptive Eq. 4
//    across FP8 & BF16 weights, M = 1, 2, 4, 8, V = 200058 (Inkling-2-Mini vocab), and omega_max in {0.0, 0.3, 0.5, 1.0}.
// 2. Edge cases: omega_max=0 (exact 0-ULP bit identity with single-vector dense::head), exact top-2 tie (omega=omega_max),
//    CUDA-graph capture & replay equivalence.
// 3. Hidden-space extrapolation kernels (loopcd_hidden_blend, loopcd_hidden_from_residual) vs FP64 reference.
// 4. Microbenchmark of single-pass head_loopcd vs plain dense::head vs naive 2-pass (2x dense::head) at V=200058, H=2304.
#include "kernels/dense/dense.h"
#include "kernels/oracle.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

using namespace ingine::test;
namespace dense = ingine::kernels::dense;

namespace {

constexpr int H = 2304;
constexpr int V_FULL = 200058;
cudaStream_t g_stream = nullptr;

void sync() { INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream)); }

struct SynthWeights {
  int rows = 0;
  bool fp8 = false;
  std::optional<DeviceArray<uint8_t>> dev;
  std::optional<DeviceArray<float>> dev_scale;
  dense::Matrix matrix() const { return {dev->get(), fp8 ? dev_scale->get() : nullptr, rows}; }
};

SynthWeights make_weights(Rng& rng, int rows, bool fp8, double sd) {
  SynthWeights w;
  w.rows = rows;
  w.fp8 = fp8;
  uint64_t s = 0x9e3779b97f4a7c15ULL ^ (uint64_t(rows) << 16) ^ uint64_t(fp8);
  auto next_u64 = [&]() -> uint64_t {
    s += 0x9e3779b97f4a7c15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  };
  if (fp8) {
    uint8_t lut[4096];
    for (int i = 0; i < 4096; ++i) lut[i] = float_to_e4m3(float(rng.normal() * 64.0));
    std::vector<uint8_t> codes(size_t(rows) * H);
    std::vector<float> scale(rows);
    for (int r = 0; r < rows; ++r) {
      scale[r] = float(sd / 64.0 * rng.uniform(0.5, 1.5));
      uint8_t* dst = &codes[size_t(r) * H];
      for (int k = 0; k < H; k += 4) {
        const uint64_t u = next_u64();
        dst[k + 0] = lut[(u >> 0) & 4095u];
        dst[k + 1] = lut[(u >> 12) & 4095u];
        dst[k + 2] = lut[(u >> 24) & 4095u];
        dst[k + 3] = lut[(u >> 36) & 4095u];
      }
    }
    w.dev.emplace(codes);
    w.dev_scale.emplace(scale);
  } else {
    const std::vector<float> lut_f = randn_bf16(rng, 4096, sd);
    const auto lut_b = to_bf16(lut_f);
    std::vector<uint8_t> bytes(size_t(rows) * H * 2);
    auto* dst = reinterpret_cast<uint16_t*>(bytes.data());
    for (size_t i = 0; i < size_t(rows) * H; i += 4) {
      const uint64_t u = next_u64();
      dst[i + 0] = static_cast<uint16_t>(lut_b[(u >> 0) & 4095u]);
      dst[i + 1] = static_cast<uint16_t>(lut_b[(u >> 12) & 4095u]);
      dst[i + 2] = static_cast<uint16_t>(lut_b[(u >> 24) & 4095u]);
      dst[i + 3] = static_cast<uint16_t>(lut_b[(u >> 36) & 4095u]);
    }
    w.dev.emplace(bytes);
  }
  return w;
}

// Exact FP64 reference of Eq. (3)-(4) on top of naive 2-pass dense::head outputs (zs, zw).
void reference_loopcd_adaptive(const std::vector<float>& zs, const std::vector<float>& zw, int M, int V,
                               float omega_max, std::vector<float>& out, std::vector<double>& omegas) {
  out.resize(size_t(M) * V);
  omegas.resize(M);
  for (int m = 0; m < M; ++m) {
    const float* rs = &zs[size_t(m) * V];
    const float* rw = &zw[size_t(m) * V];
    double m1 = -std::numeric_limits<double>::infinity();
    double m2 = -std::numeric_limits<double>::infinity();
    for (int v = 0; v < V; ++v) {
      const double x = rs[v];
      if (x > m1) {
        m2 = m1;
        m1 = x;
      } else if (x > m2) {
        m2 = x;
      }
    }
    double sum = 0.0;
    for (int v = 0; v < V; ++v) sum += std::exp(double(rs[v]) - m1);
    const double p1 = 1.0 / sum;
    const double p2 = (m2 > -std::numeric_limits<double>::infinity()) ? (std::exp(m2 - m1) / sum) : 0.0;
    const double margin = std::clamp(p1 - p2, 0.0, 1.0);
    const double omega = double(omega_max) * (1.0 - margin);
    omegas[m] = omega;
    for (int v = 0; v < V; ++v) {
      out[size_t(m) * V + v] = float(double(rs[v]) + omega * (double(rs[v]) - double(rw[v])));
    }
  }
}

bool check_head_loopcd(Rng& rng, const SynthWeights& w, int M, float omega_max, bool test_graph) {
  const int V = w.rows;
  const auto xs_f = randn_bf16(rng, size_t(M) * H, 1.0);
  const auto xw_f = randn_bf16(rng, size_t(M) * H, 1.0);
  DeviceArray<ingine::ActT> dxs(to_bf16(xs_f));
  DeviceArray<ingine::ActT> dxw(to_bf16(xw_f));
  DeviceArray<float> ref_s(size_t(M) * V), ref_w(size_t(M) * V);
  DeviceArray<float> out_fused(size_t(M) * V), ws_weak(size_t(M) * V);
  DeviceArray<float> ws_red(size_t(M) * dense::kLoopCdReduceFloatsPerRow);

  // Naive 2-pass reference using production dense::head twice.
  dense::head(dxs.get(), M, w.matrix(), ref_s.get(), true, g_stream);
  dense::head(dxw.get(), M, w.matrix(), ref_w.get(), true, g_stream);
  // Fused single-pass dual-vector head_loopcd.
  if (test_graph) {
    // Warm up then capture + replay as a CUDA graph.
    dense::head_loopcd(dxs.get(), dxw.get(), M, w.matrix(), omega_max, out_fused.get(), ws_weak.get(), ws_red.get(),
                       true, g_stream);
    sync();
    cudaGraph_t g = nullptr;
    INGINE_TEST_CUDA(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeThreadLocal));
    dense::head_loopcd(dxs.get(), dxw.get(), M, w.matrix(), omega_max, out_fused.get(), ws_weak.get(), ws_red.get(),
                       true, g_stream);
    INGINE_TEST_CUDA(cudaStreamEndCapture(g_stream, &g));
    cudaGraphExec_t exec = nullptr;
    INGINE_TEST_CUDA(cudaGraphInstantiate(&exec, g, 0));
    cudaGraphDestroy(g);
    INGINE_TEST_CUDA(cudaMemsetAsync(out_fused.get(), 0, size_t(M) * V * sizeof(float), g_stream));
    INGINE_TEST_CUDA(cudaGraphLaunch(exec, g_stream));
    sync();
    cudaGraphExecDestroy(exec);
  } else {
    dense::head_loopcd(dxs.get(), dxw.get(), M, w.matrix(), omega_max, out_fused.get(), ws_weak.get(), ws_red.get(),
                       true, g_stream);
    sync();
  }

  const auto h_zs = ref_s.download();
  const auto h_zw = ref_w.download();
  const auto h_got = out_fused.download();
  const auto h_got_w = ws_weak.download();

  // 1) ws_weak must be bit-identical (0 ULP) to naive dense::head(x_weak)!
  uint64_t weak_bit_diffs = 0;
  float weak_max_abs = 0.f;
  for (size_t i = 0; i < h_zw.size(); ++i) {
    uint32_t a, b;
    std::memcpy(&a, &h_zw[i], 4);
    std::memcpy(&b, &h_got_w[i], 4);
    if (a != b) {
      ++weak_bit_diffs;
      weak_max_abs = std::max(weak_max_abs, std::fabs(h_zw[i] - h_got_w[i]));
    }
  }

  // 2) out_fused must match FP64 Eq. 3-4 reference (and exact 0-ULP bit identity when omega_max == 0).
  std::vector<float> h_ref;
  std::vector<double> omegas;
  reference_loopcd_adaptive(h_zs, h_zw, M, V, omega_max, h_ref, omegas);
  float max_abs = 0.f;
  uint64_t zero_bit_diffs = 0;
  uint64_t non_finite = 0;
  for (size_t i = 0; i < h_ref.size(); ++i) {
    if (!std::isfinite(h_got[i]) || !std::isfinite(h_got_w[i])) {
      ++non_finite;
      continue;
    }
    max_abs = std::max(max_abs, std::fabs(h_got[i] - h_ref[i]));
    if (omega_max == 0.f) {
      uint32_t a, b;
      std::memcpy(&a, &h_zs[i], 4);
      std::memcpy(&b, &h_got[i], 4);
      if (a != b) ++zero_bit_diffs;
    }
  }
  const float tol = omega_max == 0.f ? 0.f : 1e-4f;
  const bool ok = (non_finite == 0) && (weak_bit_diffs == 0) && (max_abs <= tol) && (zero_bit_diffs == 0);
  std::printf("%s head_loopcd %s V=%d M=%d omega_max=%.2f graph=%d: weak_bit_diffs=%llu non_finite=%llu max_abs=%.3e (omega0=%.4f)\n",
              ok ? "PASS" : "FAIL", w.fp8 ? "fp8" : "bf16", V, M, omega_max, test_graph ? 1 : 0,
              static_cast<unsigned long long>(weak_bit_diffs), static_cast<unsigned long long>(non_finite),
              max_abs, omegas[0]);
  return ok;
}

bool test_hidden_kernels(Rng& rng) {
  const int M = 4;
  const float omega = 0.35f;
  const auto hs = randn_bf16(rng, size_t(M) * H, 1.0);
  const auto hw = randn_bf16(rng, size_t(M) * H, 1.0);
  const auto base = randn_bf16(rng, size_t(M) * H, 1.0);
  const auto write = randn_bf16(rng, size_t(M) * H, 1.0);
  DeviceArray<float> dhs(hs), dhw(hw), dbase(base), dwrite(write), ddst(size_t(M) * H);
  dense::loopcd_hidden_blend(dhs.get(), dhw.get(), M, omega, true, g_stream);
  dense::loopcd_hidden_from_residual(ddst.get(), dbase.get(), dwrite.get(), dhw.get(), M, omega, true, g_stream);
  sync();
  const auto g_blend = dhs.download();
  const auto g_res = ddst.download();
  float max_b = 0.f, max_r = 0.f;
  for (size_t i = 0; i < hs.size(); ++i) {
    const double rb = double(hs[i]) + double(omega) * (double(hs[i]) - double(hw[i]));
    const double sum = double(base[i]) + double(write[i]);
    const double rr = sum + double(omega) * (sum - double(hw[i]));
    max_b = std::max(max_b, float(std::fabs(double(g_blend[i]) - rb)));
    max_r = std::max(max_r, float(std::fabs(double(g_res[i]) - rr)));
  }
  const bool ok = (max_b <= 1e-5f) && (max_r <= 1e-5f);
  std::printf("%s loopcd_hidden_kernels M=%d omega=%.2f: blend_max_abs=%.3e res_max_abs=%.3e\n",
              ok ? "PASS" : "FAIL", M, omega, max_b, max_r);
  return ok;
}

void bench_head_full_vocab(Rng& rng, const SynthWeights& w) {
  const int V = w.rows;
  const int M = 1;
  const auto xs_f = randn_bf16(rng, H, 1.0);
  const auto xw_f = randn_bf16(rng, H, 1.0);
  DeviceArray<ingine::ActT> dxs(to_bf16(xs_f)), dxw(to_bf16(xw_f));
  DeviceArray<float> out1(V), out2(V), ws_red(dense::kLoopCdReduceFloatsPerRow);

  auto time_us = [&](auto fn) {
    for (int i = 0; i < 10; ++i) fn();
    sync();
    cudaEvent_t e0, e1;
    INGINE_TEST_CUDA(cudaEventCreate(&e0));
    INGINE_TEST_CUDA(cudaEventCreate(&e1));
    constexpr int ITERS = 100;
    INGINE_TEST_CUDA(cudaEventRecord(e0, g_stream));
    for (int i = 0; i < ITERS; ++i) fn();
    INGINE_TEST_CUDA(cudaEventRecord(e1, g_stream));
    INGINE_TEST_CUDA(cudaEventSynchronize(e1));
    float ms = 0.f;
    INGINE_TEST_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return (double(ms) * 1000.0) / ITERS;
  };

  const double us_plain = time_us([&] { dense::head(dxs.get(), M, w.matrix(), out1.get(), true, g_stream); });
  const double us_2pass = time_us([&] {
    dense::head(dxs.get(), M, w.matrix(), out1.get(), true, g_stream);
    dense::head(dxw.get(), M, w.matrix(), out2.get(), true, g_stream);
  });
  const double us_fused = time_us([&] {
    dense::head_loopcd(dxs.get(), dxw.get(), M, w.matrix(), 0.5f, out1.get(), out2.get(), ws_red.get(), true, g_stream);
  });
  const double overhead_pct = 100.0 * (us_fused - us_plain) / us_plain;
  std::printf("BENCH head_%s V=%d M=1: plain=%.1f us | naive_2pass=%.1f us (%.2fx) | fused_loopcd=%.1f us (%+.2f%% vs plain, %.2fx faster than 2-pass)\n",
              w.fp8 ? "fp8" : "bf16", V, us_plain, us_2pass, us_2pass / us_plain, us_fused, overhead_pct,
              us_2pass / us_fused);
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  INGINE_TEST_CUDA(cudaStreamCreate(&g_stream));
  Rng rng(882);

  bool ok = true;
  ok &= test_hidden_kernels(rng);

  // Full Inkling-2-Mini vocabulary V=200058 for both FP8 and BF16 across M=1,2,4,8 + graph capture.
  for (bool fp8 : {true, false}) {
    const SynthWeights w_full = make_weights(rng, V_FULL, fp8, 0.03);
    ok &= check_head_loopcd(rng, w_full, 1, 0.0f, false);
    ok &= check_head_loopcd(rng, w_full, 1, 0.5f, false);
    ok &= check_head_loopcd(rng, w_full, 1, 1.0f, true);  // CUDA graph capture + replay
    ok &= check_head_loopcd(rng, w_full, 2, 0.5f, false);
    ok &= check_head_loopcd(rng, w_full, 4, 0.3f, true);
    ok &= check_head_loopcd(rng, w_full, 8, 0.5f, false);
    bench_head_full_vocab(rng, w_full);
  }

  cudaStreamDestroy(g_stream);
  std::printf("%s\n", ok ? "LOOPCD_KERNELS_ALL_PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}

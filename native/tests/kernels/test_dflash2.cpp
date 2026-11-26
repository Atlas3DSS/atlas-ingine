// Native DFlash2 drafter kernel oracle tests against FP64 host references:
// - kdense::gemv K=4096 and K=6912 (BF16/FP16 and FP8 rows)
// - dflash2_rmsnorm / dflash2_rmsnorm_f32
// - dflash2_conv_prepare & dflash2_conv_finish_add_rmsnorm (2-tap grouped causal conv, group=16)
// - dflash2_qkv_rope (theta=10000) & dflash2_ctx_kv_store
// - dflash2_block_attention (QH=4, S=2048 exact per-query |p_key - p_query| < 2048 mask across wraparound)
// - dflash2_topk16 (ties, -inf, map remapping)
// - dflash2_select_path (greedy & sampled, after_correction & before_correction softcap, sparse draft_logits clear)
#include "kernels/oracle.h"

#include "kernels/dense/dense.h"
#include "kernels/draft/draft.h"
#include "kernels/draft/rng.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

using namespace ingine::test;
namespace kd = ingine::kernels::draft;
namespace kdense = ingine::kernels::dense;

namespace {

constexpr int H = kd::H;
constexpr int BLOCK = kd::BLOCK;
constexpr int GAMMA = kd::GAMMA;
cudaStream_t g_stream = nullptr;

void sync() { INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream)); }

bool check_vec(const char* name, const std::vector<float>& got, const std::vector<double>& ref, double atol,
               double rtol) {
  if (got.size() != ref.size()) {
    std::printf("FAIL %s: size mismatch %zu vs %zu\n", name, got.size(), ref.size());
    return false;
  }
  double max_abs = 0.0, max_rel = 0.0;
  std::size_t worst = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double g = double(got[i]), r = ref[i];
    if (std::isinf(r)) {
      if (!(std::isinf(g) && (std::signbit(g) == std::signbit(r)))) {
        std::printf("FAIL %s: inf mismatch at %zu (got %g ref %g)\n", name, i, g, r);
        return false;
      }
      continue;
    }
    const double d = std::abs(g - r);
    const double rel = d / std::max(1e-6, std::abs(r));
    if (d > max_abs) {
      max_abs = d;
      worst = i;
    }
    if (rel > max_rel) max_rel = rel;
  }
  const double tol = atol + rtol * std::abs(ref[worst]);
  const bool ok = (max_abs <= atol) || (std::abs(double(got[worst]) - ref[worst]) <= tol && max_rel <= rtol + atol);
  // Strict per-element check:
  bool elem_ok = true;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double r = ref[i];
    if (std::isinf(r)) continue;
    const double g = double(got[i]);
    if (!std::isfinite(g) || std::abs(g - r) > atol + rtol * std::abs(r)) {
      std::printf("FAIL %s: idx %zu got=%g ref=%g abs=%g (atol=%g rtol=%g)\n", name, i, g, r, std::abs(g - r), atol,
                  rtol);
      elem_ok = false;
      break;
    }
  }
  if (elem_ok) {
    std::printf("PASS %s (max_abs=%.4g)\n", name, max_abs);
  }
  (void)ok;
  return elem_ok;
}

bool test_gemv_k(Rng& rng, int K, int M, bool fp8) {
  const int R = 256;
  const auto x_f = randn_bf16(rng, std::size_t(M) * K, 0.5);
  DeviceArray<ingine::ActT> d_x(to_bf16(x_f));
  std::vector<double> w_eff(std::size_t(R) * K);
  std::optional<DeviceArray<std::uint8_t>> d_w;
  std::optional<DeviceArray<float>> d_sc;
  if (fp8) {
    std::vector<std::uint8_t> codes(std::size_t(R) * K);
    std::vector<float> scales(R);
    for (int r = 0; r < R; ++r) {
      scales[r] = float(0.01 * rng.uniform(0.5, 1.5));
      for (int k = 0; k < K; ++k) {
        const std::uint8_t c = float_to_e4m3(float(rng.normal() * 48.0));
        codes[std::size_t(r) * K + k] = c;
        w_eff[std::size_t(r) * K + k] = double(e4m3_to_float(c)) * double(scales[r]);
      }
    }
    d_w.emplace(codes);
    d_sc.emplace(scales);
  } else {
    const auto wf = randn_bf16(rng, std::size_t(R) * K, 0.05);
    for (std::size_t i = 0; i < wf.size(); ++i) w_eff[i] = wf[i];
    const auto wb = to_bf16(wf);
    std::vector<std::uint8_t> bytes(wb.size() * 2);
    std::memcpy(bytes.data(), wb.data(), bytes.size());
    d_w.emplace(bytes);
  }
  DeviceArray<float> d_out(std::size_t(M) * R);
  kdense::Matrix mat{d_w->get(), fp8 ? d_sc->get() : nullptr, R};
  kdense::gemv(d_x.get(), M, K, mat, d_out.get(), true, g_stream);
  sync();
  const auto got = d_out.download();
  std::vector<double> ref(std::size_t(M) * R);
  for (int m = 0; m < M; ++m) {
    for (int r = 0; r < R; ++r) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += w_eff[std::size_t(r) * K + k] * double(x_f[std::size_t(m) * K + k]);
      ref[std::size_t(m) * R + r] = acc;
    }
  }
  char label[96];
  std::snprintf(label, sizeof label, "gemv_K%d_M%d_%s", K, M, fp8 ? "fp8" : "act");
  return check_vec(label, got, ref, kScalarActIsFP16 ? 2e-2 : 5e-2, kScalarActIsFP16 ? 5e-3 : 2e-2);
}

bool test_conv_and_rmsnorm(Rng& rng) {
  const auto x_f = randn_bf16(rng, BLOCK * H, 0.7);
  const auto base_f = randn_bf16(rng, 4 * H, 0.5);
  const auto dyn_f = randn_bf16(rng, BLOCK * kd::DF2_CONV_PROJ_ROWS, 0.2);
  const auto res_f = randn_bf16(rng, BLOCK * H, 0.7);
  const auto delta_f = randn_bf16(rng, BLOCK * H, 0.5);
  const auto nw_f = randn_bf16(rng, H, 1.0);

  DeviceArray<ingine::ActT> d_x(to_bf16(x_f)), d_base(to_bf16(base_f)), d_out(BLOCK * H);
  DeviceArray<float> d_dyn(dyn_f), d_delta(delta_f);
  DeviceArray<ingine::ActT> d_res(to_bf16(res_f)), d_nw(to_bf16(nw_f)), d_xn(BLOCK * H);

  kd::dflash2_conv_prepare(d_x.get(), d_base.get(), d_dyn.get(), d_out.get(), BLOCK, g_stream);
  kd::dflash2_conv_finish_add_rmsnorm(d_res.get(), d_delta.get(), d_base.get(), d_dyn.get(), d_nw.get(), d_xn.get(),
                                      BLOCK, g_stream);
  sync();

  std::vector<double> ref_prep(BLOCK * H);
  for (int r = 0; r < BLOCK; ++r) {
    for (int h = 0; h < H; ++h) {
      const int g = h / kd::DF2_CONV_GROUP;
      const double c0 = double(base_f[h]) + double(dyn_f[r * kd::DF2_CONV_PROJ_ROWS + g]);
      double v = c0 * double(x_f[r * H + h]);
      if (r > 0) {
        const double c1 = double(base_f[H + h]) + double(dyn_f[r * kd::DF2_CONV_PROJ_ROWS + kd::DF2_CONV_GROUPS + g]);
        v += c1 * double(x_f[(r - 1) * H + h]);
      }
      ref_prep[r * H + h] = v;
    }
  }
  bool ok = check_vec("dflash2_conv_prepare", bf16_download(d_out), ref_prep, 1e-2, 2e-2);

  std::vector<double> ref_res(BLOCK * H), ref_xn(BLOCK * H);
  for (int r = 0; r < BLOCK; ++r) {
    double ss = 0.0;
    for (int h = 0; h < H; ++h) {
      const int g = h / kd::DF2_CONV_GROUP;
      const double c0 = double(base_f[2 * H + h]) + double(dyn_f[r * kd::DF2_CONV_PROJ_ROWS + 2 * kd::DF2_CONV_GROUPS + g]);
      double v = c0 * double(delta_f[r * H + h]);
      if (r > 0) {
        const double c1 =
            double(base_f[3 * H + h]) + double(dyn_f[r * kd::DF2_CONV_PROJ_ROWS + 3 * kd::DF2_CONV_GROUPS + g]);
        v += c1 * double(delta_f[(r - 1) * H + h]);
      }
      const double rv = rbf(double(res_f[r * H + h]) + v);
      ref_res[r * H + h] = rv;
      ss += rv * rv;
    }
    const double rs = 1.0 / std::sqrt(ss / double(H) + double(kd::EPS));
    for (int h = 0; h < H; ++h) ref_xn[r * H + h] = ref_res[r * H + h] * rs * double(nw_f[h]);
  }
  ok &= check_vec("dflash2_conv_finish_res", bf16_download(d_res), ref_res, 2e-2, 2e-2);
  ok &= check_vec("dflash2_conv_finish_xn", bf16_download(d_xn), ref_xn, 2e-2, 2e-2);
  return ok;
}

bool test_block_attention_sliding_mask(Rng& rng) {
  // Position P = 2050 so ring slots wrap around and exact boundary |p_key - (P+r)| < 2048 differs per row r!
  // For r=0 (p_q=2050): cp=2 is distance 2048 (EXCLUDED), cp=3 is distance 2047 (INCLUDED).
  // For r=7 (p_q=2057): cp=9 is distance 2048 (EXCLUDED), cp=10 is distance 2047 (INCLUDED).
  const int P = 2050;
  const int S = kd::DF2_WINDOW;
  const int SLOTS = kd::DF2_LOCAL_SLOTS;

  const auto q_f = randn_bf16(rng, std::size_t(BLOCK) * kd::DF2_Q_DIM, 0.4);
  const auto k_f = randn_bf16(rng, std::size_t(SLOTS) * kd::KV_DIM, 0.4);
  const auto v_f = randn_bf16(rng, std::size_t(SLOTS) * kd::KV_DIM, 0.4);
  std::vector<std::int32_t> ctx_pos(S, -1);
  // Populate all 2048 ring slots with positions [P - 2048, P) = [2, 2050), stored at pos % 2048.
  for (int pos = P - S; pos < P; ++pos) ctx_pos[pos % S] = pos;
  // Also put a huge spike in K/V at cp = 2..8 so any masking bug for rows r=0..7 explodes the error immediately!
  std::vector<float> k_mut = k_f, v_mut = v_f;
  for (int pos = 2; pos <= 9; ++pos) {
    const int slot = pos % S;
    for (int d = 0; d < kd::KV_DIM; ++d) {
      k_mut[std::size_t(slot) * kd::KV_DIM + d] = float(rbf(3.0 * rng.normal()));
      v_mut[std::size_t(slot) * kd::KV_DIM + d] = float(rbf(5.0 * (pos - 1)));
    }
  }

  DeviceArray<ingine::ActT> d_q(to_bf16(q_f)), d_k(to_bf16(k_mut)), d_v(to_bf16(v_mut));
  DeviceArray<std::int32_t> d_pos(ctx_pos);
  DeviceArray<kd::DraftParams> d_p(1);
  kd::set_draft_inputs(d_p.get(), 0, P, 0.f, g_stream);
  DeviceArray<float> d_ws(kd::dflash2_attention_workspace_floats());
  DeviceArray<ingine::ActT> d_out(std::size_t(BLOCK) * kd::DF2_Q_DIM);

  kd::dflash2_block_attention(d_q.get(), d_k.get(), d_v.get(), d_pos.get(), d_p.get(), d_ws.get(), d_out.get(), g_stream);
  sync();

  const double scale = 1.0 / std::sqrt(double(kd::HEAD_DIM));
  std::vector<double> ref(std::size_t(BLOCK) * kd::DF2_Q_DIM, 0.0);
  for (int r = 0; r < BLOCK; ++r) {
    const int pq = P + r;
    for (int h = 0; h < kd::DF2_HEADS; ++h) {
      const int g = h / (kd::DF2_HEADS / kd::KV_HEADS);
      std::vector<double> sc(SLOTS, -std::numeric_limits<double>::infinity());
      double mx = -std::numeric_limits<double>::infinity();
      for (int s = 0; s < SLOTS; ++s) {
        const bool ok_key = (s >= S) || (ctx_pos[s] >= 0 && ctx_pos[s] < P && ctx_pos[s] + S > pq);
        if (!ok_key) continue;
        double dot = 0.0;
        for (int d = 0; d < kd::HEAD_DIM; ++d) {
          dot += double(q_f[(std::size_t(r) * kd::DF2_HEADS + h) * kd::HEAD_DIM + d]) *
                 double(k_mut[(std::size_t(s) * kd::KV_HEADS + g) * kd::HEAD_DIM + d]);
        }
        sc[s] = dot * scale;
        mx = std::max(mx, sc[s]);
      }
      double sum = 0.0;
      std::vector<double> pr(SLOTS, 0.0);
      for (int s = 0; s < SLOTS; ++s) {
        if (std::isfinite(sc[s])) {
          pr[s] = std::exp(sc[s] - mx);
          sum += pr[s];
        }
      }
      for (int d = 0; d < kd::HEAD_DIM; ++d) {
        double o = 0.0;
        for (int s = 0; s < SLOTS; ++s) {
          if (pr[s] > 0.0) o += pr[s] * double(v_mut[(std::size_t(s) * kd::KV_HEADS + g) * kd::HEAD_DIM + d]);
        }
        ref[(std::size_t(r) * kd::DF2_HEADS + h) * kd::HEAD_DIM + d] = o / sum;
      }
    }
  }
  return check_vec("dflash2_block_attention_sliding_mask", bf16_download(d_out), ref, 3e-2, 3e-2);
}

bool test_topk16_and_select_path(Rng& rng) {
  const int V = 4096;
  const int rows = GAMMA;
  std::vector<float> base(std::size_t(rows) * V);
  for (int r = 0; r < rows; ++r) {
    for (int v = 0; v < V; ++v) base[std::size_t(r) * V + v] = float(rng.normal() * 5.0);
    // Inject explicit tie values and -inf entries to verify deterministic tie-breaking by lowest index.
    base[std::size_t(r) * V + 17] = 100.0f;
    base[std::size_t(r) * V + 91] = 100.0f;
    base[std::size_t(r) * V + 5] = -std::numeric_limits<float>::infinity();
  }
  std::vector<std::int32_t> map(V);
  for (int v = 0; v < V; ++v) map[v] = v;

  DeviceArray<float> d_base(base);
  DeviceArray<std::int32_t> d_map(map);
  DeviceArray<float> d_cval(rows * kd::DF2_TOPK);
  DeviceArray<std::int32_t> d_cidx(rows * kd::DF2_TOPK);
  DeviceArray<std::uint8_t> d_ws(kd::dflash2_topk16_workspace_bytes(rows));

  kd::dflash2_topk16(d_base.get(), rows, V, d_map.get(), d_cval.get(), d_cidx.get(), d_ws.get(), g_stream);
  sync();
  const auto got_val = d_cval.download();
  const auto got_idx = d_cidx.download();

  for (int r = 0; r < rows; ++r) {
    std::vector<int> ord(V);
    std::iota(ord.begin(), ord.end(), 0);
    std::partial_sort(ord.begin(), ord.begin() + kd::DF2_TOPK, ord.end(), [&](int a, int b) {
      const float va = base[std::size_t(r) * V + a], vb = base[std::size_t(r) * V + b];
      return va != vb ? va > vb : a < b;
    });
    for (int k = 0; k < kd::DF2_TOPK; ++k) {
      const int exp_i = ord[k];
      if (got_idx[r * kd::DF2_TOPK + k] != exp_i || got_val[r * kd::DF2_TOPK + k] != base[std::size_t(r) * V + exp_i]) {
        std::printf("FAIL dflash2_topk16: row=%d k=%d got=(%d,%g) ref=(%d,%g)\n", r, k, got_idx[r * kd::DF2_TOPK + k],
                    got_val[r * kd::DF2_TOPK + k], exp_i, base[std::size_t(r) * V + exp_i]);
        return false;
      }
    }
  }
  std::printf("PASS dflash2_topk16\n");

  // Test dflash2_select_path for greedy (T=0, after_correction) AND sampled (T=0.7, after_correction)
  // including verifying that previous sparse entries in draft_logits are cleared back to -inf!
  const auto g_f = randn_bf16(rng, GAMMA * kd::DF2_RANK, 0.5);
  const auto pred_f = randn_bf16(rng, std::size_t(V) * kd::DF2_RANK, 0.3);
  const auto succ_f = randn_bf16(rng, std::size_t(V) * kd::DF2_RANK, 0.3);
  DeviceArray<float> d_g(g_f);
  DeviceArray<ingine::ActT> d_pred(to_bf16(pred_f)), d_succ(to_bf16(succ_f));
  DeviceArray<std::int32_t> d_prev(GAMMA * kd::DF2_TOPK);
  INGINE_TEST_CUDA(cudaMemsetAsync(d_prev.get(), 0, GAMMA * kd::DF2_TOPK * 4, g_stream));
  DeviceArray<float> d_dlogits(std::size_t(GAMMA) * V);
  kd::fill(d_dlogits.get(), std::size_t(GAMMA) * V, -std::numeric_limits<float>::infinity(), g_stream);
  DeviceArray<std::int32_t> d_toks(GAMMA);
  DeviceArray<float> d_conf(GAMMA), d_qstats(2 * GAMMA);
  DeviceArray<kd::DraftParams> d_p(1);

  const int anchor = 42;
  const std::uint64_t seed = 0x123456789abcdef0ull;
  kd::set_seed(d_p.get(), seed, g_stream);

  for (int pass = 0; pass < 2; ++pass) {
    const float T = pass == 0 ? 0.f : 0.7f;
    const int P = 100 + pass * 10;
    kd::set_draft_inputs(d_p.get(), anchor, P, T, g_stream);
    kd::dflash2_select_path(d_cval.get(), d_cidx.get(), d_g.get(), d_pred.get(), d_succ.get(), d_prev.get(),
                            d_dlogits.get(), V, 30.f, /*softcap_after_correction=*/true, d_toks.get(), d_conf.get(),
                            d_qstats.get(), d_p.get(), g_stream);
    sync();

    const auto toks = d_toks.download();
    const auto conf = d_conf.download();
    const auto qstats = d_qstats.download();
    const auto dlogits = d_dlogits.download();
    const auto params = d_p.download();

    if (params[0].draws != std::uint64_t(pass + 1) || params[0].round_start != P) {
      std::printf("FAIL dflash2_select_path: draws/round_start mismatch\n");
      return false;
    }

    int prev = anchor;
    for (int i = 0; i < GAMMA; ++i) {
      double logits[kd::DF2_TOPK];
      double mx = -std::numeric_limits<double>::infinity();
      int best_c = 0;
      double best_score = -std::numeric_limits<double>::infinity();
      for (int c = 0; c < kd::DF2_TOPK; ++c) {
        const int tok = got_idx[i * kd::DF2_TOPK + c];
        double dot = 0.0;
        for (int r = 0; r < kd::DF2_RANK; ++r) {
          dot += double(pred_f[std::size_t(prev) * kd::DF2_RANK + r]) * double(g_f[i * kd::DF2_RANK + r]) *
                 double(succ_f[std::size_t(tok) * kd::DF2_RANK + r]);
        }
        const double raw = double(got_val[i * kd::DF2_TOPK + c]) + dot;
        logits[c] = 30.0 * std::tanh(raw / 30.0);
        mx = std::max(mx, logits[c]);
        double score = logits[c];
        if (T > 0.f) {
          score = logits[c] / double(T) + double(kd::gumbel(seed, std::uint64_t(pass), std::uint32_t(i), std::uint32_t(c)));
        }
        if (score > best_score) {
          best_score = score;
          best_c = c;
        }
      }
      if (toks[i] != got_idx[i * kd::DF2_TOPK + best_c]) {
        std::printf("FAIL dflash2_select_path pass=%d step=%d tok=%d ref=%d\n", pass, i, toks[i],
                    got_idx[i * kd::DF2_TOPK + best_c]);
        return false;
      }
      double s = 0.0;
      for (int c = 0; c < kd::DF2_TOPK; ++c) s += std::exp(logits[c] - mx);
      const double ref_conf = 1.0 / s;
      if (std::abs(double(conf[i]) - ref_conf) > 1e-4) {
        std::printf("FAIL dflash2_select_path conf pass=%d step=%d got=%g ref=%g\n", pass, i, conf[i], ref_conf);
        return false;
      }
      // Verify exact sparse population of draft_logits[i, :].
      int finite_cnt = 0;
      for (int v = 0; v < V; ++v) {
        if (std::isfinite(dlogits[std::size_t(i) * V + v])) ++finite_cnt;
      }
      if (finite_cnt != kd::DF2_TOPK) {
        std::printf("FAIL dflash2_select_path sparse count pass=%d step=%d got=%d\n", pass, i, finite_cnt);
        return false;
      }
      for (int c = 0; c < kd::DF2_TOPK; ++c) {
        const int tok = got_idx[i * kd::DF2_TOPK + c];
        if (std::abs(double(dlogits[std::size_t(i) * V + tok]) - logits[c]) > 1e-3) {
          std::printf("FAIL dflash2_select_path logit mismatch pass=%d step=%d c=%d\n", pass, i, c);
          return false;
        }
      }
      if (T > 0.f) {
        double qm = -std::numeric_limits<double>::infinity(), qs = 0.0;
        for (int c = 0; c < kd::DF2_TOPK; ++c) qm = std::max(qm, logits[c] / double(T));
        for (int c = 0; c < kd::DF2_TOPK; ++c) qs += std::exp(logits[c] / double(T) - qm);
        if (std::abs(double(qstats[2 * i]) - qm) > 2e-3 || std::abs(double(qstats[2 * i + 1]) - qs) > 2e-3) {
          std::printf("FAIL dflash2_select_path qstats pass=%d step=%d got=(%g,%g) ref=(%g,%g)\n", pass, i,
                      qstats[2 * i], qstats[2 * i + 1], qm, qs);
          return false;
        }
      }
      prev = toks[i];
    }
    std::printf("PASS dflash2_select_path_%s\n", T > 0.f ? "sampled" : "greedy");
  }
  return true;
}

}  // namespace

int main() {
  INGINE_TEST_CUDA(cudaStreamCreate(&g_stream));
  Rng rng(20261126ull);
  bool ok = true;
  for (int K : {4096, 6912}) {
    for (int M : {1, 8}) {
      ok &= test_gemv_k(rng, K, M, false);
      ok &= test_gemv_k(rng, K, M, true);
    }
  }
  ok &= test_conv_and_rmsnorm(rng);
  ok &= test_block_attention_sliding_mask(rng);
  ok &= test_topk16_and_select_path(rng);
  INGINE_TEST_CUDA(cudaStreamDestroy(g_stream));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}

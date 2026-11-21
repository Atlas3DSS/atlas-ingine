// CPU-oracle tests of the DSpark draft kernels (src/kernels/draft) at small sizes. The oracles repeat
// dspark.py's rounding points in FP32 (reductions in FP64), so results agree to about one BF16 ulp; discrete
// outputs (candidate sets, picks, tokens, slots) must match exactly or be consistent with the kernel's own
// logits.
#include "common/test_util.h"
#include "kernels/draft/draft.h"
#include "kernels/draft/rng.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace ingine::test;
namespace kd = ingine::kernels::draft;
using bf16 = __nv_bfloat16;

namespace {

constexpr int H = kd::H, D = kd::HEAD_DIM;
int g_failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++g_failures;
  }
}

std::vector<bf16> rand_bf16(Rng& r, size_t n, double scale = 1.0) {
  std::vector<bf16> v(n);
  for (auto& x : v) x = float_to_bf16(float(r.normal() * scale));
  return v;
}
std::vector<float> to_float(const std::vector<bf16>& v) {
  std::vector<float> f(v.size());
  for (size_t i = 0; i < v.size(); ++i) f[i] = bf16_to_float(v[i]);
  return f;
}
float bfr(float x) { return round_bf16(x); }

// every |got - ref| within `ulps` BF16 ulps of max(|ref|, floor)
bool check_close(const std::vector<float>& got, const std::vector<float>& ref, const char* what, double ulps = 2.0,
           double floor = 1e-3) {
  double worst = 0;
  size_t wi = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double scale = std::max(std::fabs(double(ref[i])), floor);
    const double e = std::fabs(double(got[i]) - double(ref[i])) / (scale * std::ldexp(1.0, -7));
    if (!(e <= worst)) {
      worst = e;
      wi = i;
    }
  }
  const bool ok = worst <= ulps;
  std::printf("%s %s: worst %.2f ulp at %zu (got %.6g ref %.6g)\n", ok ? "PASS" : "FAIL", what, worst, wi,
              ref.empty() ? 0.0 : double(got[wi]), ref.empty() ? 0.0 : double(ref[wi]));
  if (!ok) ++g_failures;
  return ok;
}

std::vector<float> download_bf16(const DeviceArray<bf16>& a) { return to_float(a.download()); }

// dspark.rms over a row (FP64 sum of squares), times w, rounded to BF16
std::vector<float> rms_row(const float* x, int n, const float* w) {
  double ss = 0;
  for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
  const float rs = 1.f / std::sqrt(float(ss / n) + kd::EPS);
  std::vector<float> y(n);
  for (int i = 0; i < n; ++i) y[i] = bfr((x[i] * rs) * w[i]);
  return y;
}

// neox rope of one BF16 head row at `pos`, rounded to BF16
std::vector<float> rope(const std::vector<float>& y, int pos) {
  std::vector<float> o(D);
  for (int i = 0; i < D / 2; ++i) {
    const float inv = std::pow(kd::ROPE_THETA, (-float(i) * 2.f) / float(D));
    const float ang = float(pos) * inv;
    const float c = std::cos(ang), s = std::sin(ang);
    o[i] = bfr(y[i] * c - y[i + D / 2] * s);
    o[i + D / 2] = bfr(y[i + D / 2] * c + y[i] * s);
  }
  return o;
}

std::vector<float> norm_rope(const float* x, const float* w, int pos) { return rope(rms_row(x, D, w), pos); }

void set_params(kd::DraftParams* p, int anchor, int P, float T, int start, uint64_t seed) {
  kd::set_seed(p, seed, nullptr);
  kd::set_draft_inputs(p, anchor, P, T, nullptr);
  kd::set_commit_start(p, start, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
}

// ---- tests -------------------------------------------------------------------------------------------

void test_mix_rmsnorm(Rng& r) {
  const int n = 5;
  auto feats = rand_bf16(r, size_t(n) * 4 * H);
  auto alpha = rand_bf16(r, 4, 0.5);
  auto w = rand_bf16(r, H, 0.3);
  DeviceArray<bf16> d_feats(feats), d_alpha(alpha), d_w(w), d_out(size_t(n) * H);
  kd::mix_rmsnorm(d_feats.get(), n, d_alpha.get(), d_w.get(), d_out.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto f = to_float(feats), a = to_float(alpha), wf = to_float(w);
  std::vector<float> ref;
  for (int i = 0; i < n; ++i) {
    std::vector<float> m(H);
    for (int d = 0; d < H; ++d) {
      const float* fr = f.data() + size_t(i) * 4 * H;
      m[d] = ((fr[d] * a[0] + fr[H + d] * a[1]) + fr[2 * H + d] * a[2]) + fr[3 * H + d] * a[3];
    }
    auto y = rms_row(m.data(), H, wf.data());
    ref.insert(ref.end(), y.begin(), y.end());
  }
  check_close(download_bf16(d_out), ref, "mix_rmsnorm");
}

void test_ctx_kv_store(Rng& r) {
  DeviceArray<kd::DraftParams> p(1);
  for (const int n : {6, 20}) {
    const int size = 16, start = 1000, row0 = 3;
    const int keep_from = n - std::min(n, size);
    set_params(p.get(), 0, 0, 0.f, start, 1);
    auto kv = rand_bf16(r, size_t(n) * 2 * kd::KV_DIM);
    auto kn = rand_bf16(r, D, 0.5);
    DeviceArray<bf16> d_kv(kv), d_kn(kn), d_k(size_t(size) * kd::KV_DIM), d_v(size_t(size) * kd::KV_DIM);
    DeviceArray<int32_t> d_pos(std::vector<int32_t>(size, -1));
    kd::ctx_kv_store(d_kv.get(), n, p.get(), row0, keep_from, d_kn.get(), d_k.get(), d_v.get(), size, d_pos.get(),
                     nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto kvf = to_float(kv), knf = to_float(kn);
    std::vector<float> ref_k(size_t(size) * kd::KV_DIM, 0.f), ref_v(size_t(size) * kd::KV_DIM, 0.f);
    std::vector<int32_t> ref_pos(size, -1);
    for (int i = keep_from; i < n; ++i) {
      const int pos = start + row0 + i, slot = pos % size;
      ref_pos[slot] = pos;
      for (int g = 0; g < kd::KV_HEADS; ++g) {
        auto k = norm_rope(kvf.data() + size_t(i) * 2 * kd::KV_DIM + g * D, knf.data(), pos);
        for (int d = 0; d < D; ++d) {
          ref_k[(size_t(slot) * kd::KV_HEADS + g) * D + d] = k[d];
          ref_v[(size_t(slot) * kd::KV_HEADS + g) * D + d] = kvf[size_t(i) * 2 * kd::KV_DIM + kd::KV_DIM + g * D + d];
        }
      }
    }
    check_close(download_bf16(d_k), ref_k, n > size ? "ctx_kv_store k (wrapped)" : "ctx_kv_store k");
    check_close(download_bf16(d_v), ref_v, "ctx_kv_store v", 0.0, 1e-30);
    expect(d_pos.download() == ref_pos, "ctx_kv_store ctx_pos");
  }
}

void test_embed_block(Rng& r) {
  const int V = 50, anchor = 7;
  auto embed = rand_bf16(r, size_t(V) * H);
  std::vector<float> en(H);
  for (auto& x : en) x = float(r.uniform(0.5, 1.5));
  auto mask = rand_bf16(r, H);
  auto in_norm = rand_bf16(r, H, 0.5);
  DeviceArray<bf16> d_embed(embed), d_mask(mask), d_in(in_norm), d_res(size_t(8) * H), d_xn(size_t(8) * H);
  DeviceArray<float> d_en(en);
  DeviceArray<unsigned long long> d_picks(std::vector<unsigned long long>(7, 123));
  DeviceArray<kd::DraftParams> p(1);
  set_params(p.get(), anchor, 0, 0.f, 0, 1);
  kd::embed_block(d_embed.get(), d_en.get(), d_mask.get(), d_in.get(), p.get(), d_res.get(), d_xn.get(),
                  d_picks.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto ef = to_float(embed), mf = to_float(mask), inf = to_float(in_norm);
  std::vector<float> ref_res, ref_xn;
  auto x0 = rms_row(ef.data() + size_t(anchor) * H, H, en.data());
  for (int i = 0; i < 8; ++i) {
    const std::vector<float>& x = i == 0 ? x0 : mf;
    ref_res.insert(ref_res.end(), x.begin(), x.end());
    auto y = rms_row(x.data(), H, inf.data());
    ref_xn.insert(ref_xn.end(), y.begin(), y.end());
  }
  check_close(download_bf16(d_res), ref_res, "embed_block res");
  check_close(download_bf16(d_xn), ref_xn, "embed_block xn");
  const auto picks = d_picks.download();
  expect(std::all_of(picks.begin(), picks.end(), [](auto v) { return v == 0; }), "embed_block clears picks");
}

void test_qkv_rope(Rng& r) {
  const int P = 37, ctx = 64;
  std::vector<float> qkv(size_t(8) * kd::QKV_ROWS);
  for (auto& x : qkv) x = bfr(float(r.normal()));
  auto qn = rand_bf16(r, D, 0.5), kn = rand_bf16(r, D, 0.5);
  DeviceArray<float> d_qkv(qkv);
  DeviceArray<bf16> d_qn(qn), d_kn(kn), d_q(size_t(8) * kd::Q_DIM);
  DeviceArray<kd::DraftParams> p(1);
  set_params(p.get(), 0, P, 0.f, 0, 1);
  const auto qnf = to_float(qn), knf = to_float(kn);
  for (const bool local : {true, false}) {
    const int slots = local ? kd::WINDOW + 8 : ctx;
    DeviceArray<bf16> d_k(size_t(slots) * kd::KV_DIM), d_v(size_t(slots) * kd::KV_DIM);
    kd::qkv_rope(d_qkv.get(), d_qn.get(), d_kn.get(), p.get(), d_q.get(), d_k.get(), d_v.get(), local, ctx, nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<float> ref_q(size_t(8) * kd::Q_DIM), ref_k(size_t(slots) * kd::KV_DIM, 0.f),
        ref_v(size_t(slots) * kd::KV_DIM, 0.f);
    for (int i = 0; i < 8; ++i) {
      const float* row = qkv.data() + size_t(i) * kd::QKV_ROWS;
      for (int h = 0; h < kd::HEADS; ++h) {
        auto q = norm_rope(row + h * D, qnf.data(), P + i);
        std::copy(q.begin(), q.end(), ref_q.begin() + (size_t(i) * kd::HEADS + h) * D);
      }
      const int slot = local ? kd::WINDOW + i : (P + i) % ctx;
      for (int g = 0; g < kd::KV_HEADS; ++g) {
        auto k = norm_rope(row + kd::Q_DIM + g * D, knf.data(), P + i);
        std::copy(k.begin(), k.end(), ref_k.begin() + (size_t(slot) * kd::KV_HEADS + g) * D);
        std::copy(row + kd::Q_DIM + kd::KV_DIM + g * D, row + kd::Q_DIM + kd::KV_DIM + (g + 1) * D,
                  ref_v.begin() + (size_t(slot) * kd::KV_HEADS + g) * D);
      }
    }
    check_close(download_bf16(d_q), ref_q, local ? "qkv_rope q (local)" : "qkv_rope q (global)");
    check_close(download_bf16(d_k), ref_k, local ? "qkv_rope k (local)" : "qkv_rope k (global)");
    check_close(download_bf16(d_v), ref_v, "qkv_rope v", 0.0, 1e-30);
  }
}

void test_block_attention(Rng& r) {
  struct Case {
    bool local;
    int nkeys, P;
    const char* name;
  };
  const Case cases[] = {{true, kd::WINDOW + 8, 1500, "attention local (wrapped ring)"},
                        {true, kd::WINDOW + 8, 300, "attention local (partial ring)"},
                        {false, 1536, 1300, "attention global (masked tail)"},
                        {false, 1024, 2000, "attention global (full ring)"},
                        {false, 2048, 5, "attention global (short context)"}};
  DeviceArray<kd::DraftParams> p(1);
  for (const Case& c : cases) {
    set_params(p.get(), 0, c.P, 0.f, 0, 1);
    auto q = rand_bf16(r, size_t(8) * kd::Q_DIM);
    auto k = rand_bf16(r, size_t(c.nkeys) * kd::KV_DIM);
    auto v = rand_bf16(r, size_t(c.nkeys) * kd::KV_DIM);
    std::vector<int32_t> cpos(kd::WINDOW);
    for (int j = 0; j < kd::WINDOW; ++j) {
      // the ring as commits leave it: slot j holds the latest position = j mod 1024 below P, some empty
      int pos = (c.P - 1) - ((c.P - 1 - j) % kd::WINDOW + kd::WINDOW) % kd::WINDOW;
      cpos[j] = pos >= 0 && r.uniform() > 0.05 ? pos : -1;
    }
    DeviceArray<bf16> d_q(q), d_k(k), d_v(v), d_o(size_t(8) * kd::Q_DIM);
    DeviceArray<int32_t> d_cpos(cpos);
    DeviceArray<float> ws(kd::attention_workspace_floats(c.nkeys));
    kd::block_attention(d_q.get(), d_k.get(), d_v.get(), c.nkeys, c.local ? d_cpos.get() : nullptr, p.get(), ws.get(),
                        d_o.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto qf = to_float(q), kf = to_float(k), vf = to_float(v);
    std::vector<float> ref(size_t(8) * kd::Q_DIM);
    for (int h = 0; h < kd::HEADS; ++h) {
      const int g = h / 3;
      for (int i = 0; i < 8; ++i) {
        std::vector<double> s(c.nkeys, -INFINITY);
        double m = -INFINITY;
        for (int j = 0; j < c.nkeys; ++j) {
          bool valid;
          if (c.local) valid = j >= kd::WINDOW || (cpos[j] >= 0 && cpos[j] < c.P && cpos[j] >= c.P - kd::WINDOW);
          else valid = j < c.P + 8;
          if (!valid) continue;
          double dot = 0;
          for (int d = 0; d < D; ++d)
            dot += double(qf[(size_t(i) * kd::HEADS + h) * D + d]) * kf[(size_t(j) * kd::KV_HEADS + g) * D + d];
          s[j] = dot / std::sqrt(double(D));
          m = std::max(m, s[j]);
        }
        double l = 0;
        std::vector<double> o(D, 0.0);
        for (int j = 0; j < c.nkeys; ++j) {
          if (s[j] == -INFINITY) continue;
          const double e = std::exp(s[j] - m);
          l += e;
          for (int d = 0; d < D; ++d) o[d] += e * vf[(size_t(j) * kd::KV_HEADS + g) * D + d];
        }
        for (int d = 0; d < D; ++d) ref[size_t(i) * kd::Q_DIM + h * D + d] = bfr(float(o[d] / l));
      }
    }
    check_close(download_bf16(d_o), ref, c.name, 2.0, 1e-2);
  }
}

void test_add_rmsnorm_silu(Rng& r) {
  const int rows = 3, inter = 640;
  auto res = rand_bf16(r, size_t(rows) * H), delta = rand_bf16(r, size_t(rows) * H), w = rand_bf16(r, H, 0.5);
  DeviceArray<bf16> d_res(res), d_delta(delta), d_w(w), d_xn(size_t(rows) * H);
  kd::add_rmsnorm(d_res.get(), d_delta.get(), d_w.get(), d_xn.get(), rows, nullptr);
  std::vector<float> gu(size_t(rows) * 2 * inter);
  for (auto& x : gu) x = bfr(float(r.normal() * 3));
  DeviceArray<float> d_gu(gu);
  DeviceArray<bf16> d_act(size_t(rows) * inter);
  kd::silu_mul(d_gu.get(), rows, inter, d_act.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto rf = to_float(res), df = to_float(delta), wf = to_float(w);
  std::vector<float> ref_res(rf.size()), ref_xn;
  for (size_t i = 0; i < rf.size(); ++i) ref_res[i] = bfr(rf[i] + df[i]);
  for (int i = 0; i < rows; ++i) {
    auto y = rms_row(ref_res.data() + size_t(i) * H, H, wf.data());
    ref_xn.insert(ref_xn.end(), y.begin(), y.end());
  }
  check_close(download_bf16(d_res), ref_res, "add_rmsnorm res", 0.0, 1e-30);
  check_close(download_bf16(d_xn), ref_xn, "add_rmsnorm xn");
  std::vector<float> ref_act(size_t(rows) * inter);
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < inter; ++j) {
      const float g = gu[size_t(i) * 2 * inter + j], u = gu[size_t(i) * 2 * inter + inter + j];
      ref_act[size_t(i) * inter + j] = bfr((g / (1.f + std::exp(-g))) * u);
    }
  check_close(download_bf16(d_act), ref_act, "silu_mul");
}

// reference top-k: value descending, index ascending
std::vector<int> topk_ref(const float* x, int V) {
  std::vector<int> idx(V);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return x[a] > x[b]; });
  idx.resize(kd::CANDIDATES);
  return idx;
}

void test_topk(Rng& r) {
  const int rows = 3, V = 5003;
  std::vector<float> base(size_t(rows) * V);
  for (int i = 0; i < rows; ++i)
    for (int x = 0; x < V; ++x) {
      double v = r.normal() * 4;
      if (i == 1) v = std::round(v * 2) / 2;  // heavy ties
      if (i == 2) v = -std::fabs(v);          // negative values (key order)
      base[size_t(i) * V + x] = bfr(float(v));
    }
  DeviceArray<float> d_base(base), d_val(size_t(rows) * kd::CANDIDATES);
  DeviceArray<int32_t> d_idx(size_t(rows) * kd::CANDIDATES);
  kd::topk_candidates(d_base.get(), rows, V, d_val.get(), d_idx.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto val = d_val.download();
  const auto idx = d_idx.download();
  bool ok = true;
  for (int i = 0; i < rows; ++i) {
    const auto ref = topk_ref(base.data() + size_t(i) * V, V);
    for (int k = 0; k < kd::CANDIDATES; ++k) {
      ok &= idx[size_t(i) * kd::CANDIDATES + k] == ref[k];
      ok &= val[size_t(i) * kd::CANDIDATES + k] == base[size_t(i) * V + ref[k]];
    }
  }
  std::printf("%s topk_candidates (sorted, ties by index)\n", ok ? "PASS" : "FAIL");
  expect(ok, "topk_candidates");
}

void test_markov(Rng& r) {
  const int V = 3000;
  auto rnn_w = rand_bf16(r, size_t(kd::RNN_OUT) * kd::RNN_IN, 0.02);
  auto rnn_b = rand_bf16(r, kd::RNN_OUT, 0.5);
  auto w1 = rand_bf16(r, size_t(V) * kd::RANK);
  auto fold_w = rand_bf16(r, size_t(V) * kd::RANK, 0.2);
  auto fold_b = rand_bf16(r, V);
  auto h = rand_bf16(r, size_t(kd::GAMMA) * H);
  auto f_rnn = rand_bf16(r, H);
  std::vector<float> base(size_t(kd::GAMMA) * V);
  for (auto& x : base) x = bfr(float(r.normal() * 5));
  const int anchor = 1234;

  DeviceArray<bf16> d_rnn_w(rnn_w), d_rnn_b(rnn_b), d_w1(w1), d_fw(fold_w), d_fb(fold_b), d_h(h), d_f(f_rnn);
  DeviceArray<bf16> d_state(size_t(kd::GAMMA + 1) * kd::RANK), d_latent(kd::RANK);
  DeviceArray<float> d_base(base), d_val(size_t(kd::GAMMA) * kd::CANDIDATES), d_part(size_t(kd::GAMMA) * kd::RNN_OUT);
  DeviceArray<float> d_logits(size_t(kd::GAMMA) * kd::CANDIDATES), d_conf(kd::GAMMA);
  DeviceArray<float> d_draft(std::vector<float>(size_t(kd::GAMMA) * V, -INFINITY));
  DeviceArray<int32_t> d_idx(size_t(kd::GAMMA) * kd::CANDIDATES), d_prev(size_t(kd::GAMMA) * kd::CANDIDATES),
      d_tokens(kd::GAMMA);
  DeviceArray<unsigned long long> d_picks(kd::GAMMA);
  DeviceArray<kd::DraftParams> p(1);
  kd::topk_candidates(d_base.get(), kd::GAMMA, V, d_val.get(), d_idx.get(), nullptr);
  kd::markov_pre(d_rnn_w.get(), d_h.get(), d_f.get(), d_part.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto cand_val = d_val.download();
  const auto cand_idx = d_idx.download();

  const auto wf = to_float(rnn_w), bfv = to_float(rnn_b), w1f = to_float(w1), fwf = to_float(fold_w),
             fbf = to_float(fold_b), hf = to_float(h), ff = to_float(f_rnn);
  // markov_pre against FP64
  {
    const auto part = d_part.download();
    std::vector<double> ref(part.size());
    for (int i = 0; i < kd::GAMMA; ++i)
      for (int row = 0; row < kd::RNN_OUT; ++row) {
        double s = 0;
        for (int c = 0; c < H; ++c) s += double(wf[size_t(row) * kd::RNN_IN + 512 + c]) * hf[size_t(i) * H + c];
        for (int c = 0; c < H; ++c) s += double(wf[size_t(row) * kd::RNN_IN + 512 + H + c]) * ff[c];
        ref[size_t(i) * kd::RNN_OUT + row] = s;
      }
    expect(report("markov_pre", compare(part.data(), ref.data(), ref.size()), 1e-5), "markov_pre");
  }

  for (const float T : {0.f, 0.7f}) {
    set_params(p.get(), anchor, 100, T, 0, 42);
    INGINE_TEST_CUDA(cudaMemset(d_state.get(), 0, sizeof(bf16) * (kd::GAMMA + 1) * kd::RANK));
    INGINE_TEST_CUDA(cudaMemset(d_picks.get(), 0, sizeof(unsigned long long) * kd::GAMMA));
    std::vector<float> state(kd::RANK, 0.f);
    int prev = anchor;
    std::vector<int> picks_host(kd::GAMMA);
    bool picks_ok = true;
    for (int i = 0; i < kd::GAMMA; ++i) {
      kd::markov_step(i, d_rnn_w.get(), d_rnn_b.get(), d_w1.get(), d_part.get(), d_picks.get(), d_idx.get(), p.get(),
                      d_state.get(), d_latent.get(), nullptr);
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
      const auto part = d_part.download();
      std::vector<float> raw(kd::RNN_OUT);
      for (int row = 0; row < kd::RNN_OUT; ++row) {
        double s = part[size_t(i) * kd::RNN_OUT + row];
        for (int c = 0; c < kd::RANK; ++c) s += double(wf[size_t(row) * kd::RNN_IN + c]) * state[c];
        for (int c = 0; c < kd::RANK; ++c)
          s += double(wf[size_t(row) * kd::RNN_IN + kd::RANK + c]) * w1f[size_t(prev) * kd::RANK + c];
        raw[row] = bfr(bfr(float(s)) + bfv[row]);
      }
      std::vector<float> ref_state(kd::RANK), ref_latent(kd::RANK);
      for (int j = 0; j < kd::RANK; ++j) {
        const float gate = bfr(1.f / (1.f + std::exp(-raw[j])));
        const float cell = bfr(std::tanh(raw[kd::RANK + j]));
        ref_state[j] = bfr(bfr(gate * state[j]) + bfr(bfr(1.f - gate) * cell));
        ref_latent[j] = bfr(std::tanh(raw[2 * kd::RANK + j]));
      }
      const auto all_state = download_bf16(d_state);
      const std::vector<float> got_state(all_state.begin() + size_t(i + 1) * kd::RANK,
                                         all_state.begin() + size_t(i + 2) * kd::RANK);
      const auto got_latent = download_bf16(d_latent);
      char name[64];
      std::snprintf(name, sizeof(name), "markov_step %d state (T=%.1f)", i, T);
      check_close(got_state, ref_state, name, 2.0, 1e-2);
      std::snprintf(name, sizeof(name), "markov_step %d latent (T=%.1f)", i, T);
      check_close(got_latent, ref_latent, name, 2.0, 1e-2);

      kd::markov_logits(i, d_fw.get(), d_fb.get(), d_val.get(), d_idx.get(), d_latent.get(), p.get(), d_logits.get(),
                        d_picks.get(), nullptr);
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
      const auto logits = d_logits.download();
      std::vector<float> ref_logits(kd::CANDIDATES), got_logits(kd::CANDIDATES);
      for (int k = 0; k < kd::CANDIDATES; ++k) {
        const int c = cand_idx[size_t(i) * kd::CANDIDATES + k];
        double dot = 0;
        for (int e = 0; e < kd::RANK; ++e) dot += double(fwf[size_t(c) * kd::RANK + e]) * got_latent[e];
        const float bias = bfr(fbf[c] + bfr(float(dot)));
        const float t = bfr(cand_val[size_t(i) * kd::CANDIDATES + k] + bias);
        ref_logits[k] = bfr(30.f * std::tanh(t / 30.f));
        got_logits[k] = logits[size_t(i) * kd::CANDIDATES + k];
      }
      std::snprintf(name, sizeof(name), "markov_logits %d (T=%.1f)", i, T);
      check_close(got_logits, ref_logits, name, 2.0, 1e-2);
      // the pick must be the argmax (first maximum) of the kernel's own scores
      const auto picks = d_picks.download();
      const int pick = int(0xffffffffu - uint32_t(picks[i] & 0xffffffffu));
      int best = 0;
      float best_s = -INFINITY;
      for (int k = 0; k < kd::CANDIDATES; ++k) {
        float s = got_logits[k];
        if (T > 0.f) {
          const float u = std::max(kd::uniform01(42, 0, uint32_t(i), uint32_t(k)), 1e-20f);
          s = got_logits[k] / std::max(T, 1e-6f) - std::log(-std::log(u));
        }
        if (s > best_s) {
          best_s = s;
          best = k;
        }
      }
      picks_ok &= pick == best;
      picks_host[i] = pick;
      prev = cand_idx[size_t(i) * kd::CANDIDATES + pick];
      state = got_state;
    }
    std::printf("%s markov picks (T=%.1f)\n", picks_ok ? "PASS" : "FAIL", T);
    expect(picks_ok, "markov picks");

    // finish twice: the second call must clear the first call's candidates
    for (int round = 0; round < 2; ++round) {
      if (round == 1) {
        std::vector<int32_t> shifted(cand_idx.size());
        for (size_t k = 0; k < shifted.size(); ++k) shifted[k] = (cand_idx[k] + 7) % V;
        d_idx.upload(shifted);
      }
      kd::markov_finish(d_idx.get(), d_logits.get(), d_picks.get(), d_prev.get(), d_draft.get(), V, d_tokens.get(),
                        d_conf.get(), p.get(), nullptr);
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
    }
    const auto idx_now = d_idx.download();
    d_idx.upload(cand_idx);
    const auto logits = d_logits.download();
    const auto draft = d_draft.download();
    const auto tokens = d_tokens.download();
    const auto conf = d_conf.download();
    bool ok = true;
    for (int i = 0; i < kd::GAMMA; ++i) {
      ok &= tokens[i] == idx_now[size_t(i) * kd::CANDIDATES + picks_host[i]];
      std::vector<float> expect_row(V, -INFINITY);
      double m = -INFINITY, s = 0;
      for (int k = 0; k < kd::CANDIDATES; ++k) {
        expect_row[idx_now[size_t(i) * kd::CANDIDATES + k]] = logits[size_t(i) * kd::CANDIDATES + k];
        m = std::max(m, double(logits[size_t(i) * kd::CANDIDATES + k]));
      }
      for (int k = 0; k < kd::CANDIDATES; ++k) s += std::exp(double(logits[size_t(i) * kd::CANDIDATES + k]) - m);
      ok &= std::equal(expect_row.begin(), expect_row.end(), draft.begin() + size_t(i) * V);
      ok &= std::fabs(conf[i] - 1.0 / s) <= 1e-5 * (1.0 / s);
    }
    const auto params = p.download();
    ok &= params[0].draws == 2;
    std::printf("%s markov_finish (T=%.1f)\n", ok ? "PASS" : "FAIL", T);
    expect(ok, "markov_finish");
  }
}

void test_fold(Rng& r) {
  const int rows = 100;
  auto hw = rand_bf16(r, size_t(H) * kd::RANK, 0.05);
  auto hb = rand_bf16(r, H, 0.05);
  const auto hwf = to_float(hw), hbf = to_float(hb);
  for (const bool fp8 : {false, true}) {
    std::vector<float> u(size_t(rows) * H);
    std::vector<uint8_t> codes(size_t(rows) * H);
    std::vector<float> scale(rows);
    std::vector<bf16> ubf;
    if (fp8) {
      for (int i = 0; i < rows; ++i) scale[i] = float(r.uniform(0.001, 0.01));
      for (size_t i = 0; i < codes.size(); ++i) {
        uint8_t c;
        do c = uint8_t(r.next() & 0xff);
        while ((c & 0x7f) == 0x7f);
        codes[i] = c;
        u[i] = bfr(e4m3_to_float(c) * scale[i / H]);
      }
    } else {
      ubf = rand_bf16(r, size_t(rows) * H);
      u = to_float(ubf);
    }
    DeviceArray<uint8_t> d_codes(codes);
    DeviceArray<bf16> d_ubf(fp8 ? std::vector<bf16>(1) : ubf), d_hw(hw), d_hb(hb), d_fw(size_t(rows) * kd::RANK),
        d_fb(rows);
    DeviceArray<float> d_scale(scale);
    kd::fold_head(fp8 ? static_cast<const void*>(d_codes.get()) : d_ubf.get(), fp8 ? d_scale.get() : nullptr, rows,
                  d_hw.get(), d_hb.get(), d_fw.get(), d_fb.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    std::vector<float> ref_w(size_t(rows) * kd::RANK), ref_b(rows);
    for (int i = 0; i < rows; ++i) {
      double sb = 0;
      for (int k = 0; k < H; ++k) sb += double(u[size_t(i) * H + k]) * hbf[k];
      ref_b[i] = bfr(float(sb));
      for (int j = 0; j < kd::RANK; ++j) {
        double s = 0;
        for (int k = 0; k < H; ++k) s += double(u[size_t(i) * H + k]) * hwf[size_t(k) * kd::RANK + j];
        ref_w[size_t(i) * kd::RANK + j] = bfr(float(s));
      }
    }
    check_close(download_bf16(d_fw), ref_w, fp8 ? "fold_w (fp8 rows)" : "fold_w (bf16 rows)", 1.0, 1e-30);
    check_close(download_bf16(d_fb), ref_b, fp8 ? "fold_b (fp8 rows)" : "fold_b (bf16 rows)", 1.0, 1e-30);
  }
}

void test_greedy_accept() {
  struct Case {
    std::vector<int32_t> drafts, target;
    int acc, next;
  };
  const Case cases[] = {{{5, 6, 7}, {5, 6, 7, 8}, 3, 8},
                        {{5, 6, 7}, {5, 9, 7, 8}, 1, 9},
                        {{5, 6, 7}, {4, 6, 7, 8}, 0, 4},
                        {{1}, {1, 2}, 1, 2},
                        {{1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 0, 9}, 6, 0}};
  DeviceArray<kd::AcceptResult> out(1);
  bool ok = true;
  for (const Case& c : cases) {
    DeviceArray<int32_t> d(c.drafts), t(c.target);
    kd::greedy_accept(t.get(), d.get(), int(c.drafts.size()), out.get(), nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    const auto res = out.download()[0];
    ok &= res.accepted == c.acc && res.next == c.next;
  }
  std::printf("%s greedy_accept (longest prefix equal to the target argmax)\n", ok ? "PASS" : "FAIL");
  expect(ok, "greedy_accept");
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Rng r(20261120);
  test_mix_rmsnorm(r);
  test_ctx_kv_store(r);
  test_embed_block(r);
  test_qkv_rope(r);
  test_block_attention(r);
  test_add_rmsnorm_silu(r);
  test_topk(r);
  test_markov(r);
  test_fold(r);
  test_greedy_accept();
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}

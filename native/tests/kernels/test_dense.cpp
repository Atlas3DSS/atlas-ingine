// Dense decode kernels (src/kernels/dense) against FP64 oracles of SPEC.md area `dense`, at the model's
// shapes with synthetic weights (adapted from engine/tests/test_dense.py: proj BF16/FP8 for several M and
// both projection layouts, proj_conv with its state update and history, head BF16/FP8, router with FP32 and
// BF16 weights, mix over its block/query/optional-tensor variants, message gates G = 1/3 with FP32/BF16
// forget/gain weights, embed, fp8_rows and dequant_rows).
#include "kernels/oracle.h"

#include "kernels/dense/dense.h"

#include <cuda_runtime.h>

#include <array>
#include <numeric>

using namespace ingine::test;
namespace dense = ingine::kernels::dense;

namespace {

constexpr int H = 2304;
cudaStream_t g_stream = nullptr;

void sync() { INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream)); }

// A random weight matrix [rows, K] as the kernels take it: BF16 values, or FP8 codes + row scales.
struct Weights {
  int rows = 0, K = 0;
  bool fp8 = false;
  std::vector<double> value;  // effective value per element (code * scale applied at the end, see dot())
  std::vector<float> scale;   // fp8: per-row scale
  std::vector<double> code;   // fp8: decoded E4M3 value; bf16: the value
  std::optional<DeviceArray<uint8_t>> dev;
  std::optional<DeviceArray<float>> dev_scale;
  dense::Matrix matrix() const { return {dev->get(), fp8 ? dev_scale->get() : nullptr, rows}; }
  double dot(int r, const float* x) const {
    double s = 0;
    const double* c = &code[size_t(r) * K];
    for (int k = 0; k < K; ++k) s += c[k] * double(x[k]);
    return fp8 ? s * double(scale[r]) : s;
  }
};

Weights make_weights(Rng& rng, int rows, int K, bool fp8, double sd) {
  Weights w;
  w.rows = rows, w.K = K, w.fp8 = fp8;
  w.code.resize(size_t(rows) * K);
  if (fp8) {
    std::vector<uint8_t> codes(size_t(rows) * K);
    w.scale.resize(rows);
    for (int r = 0; r < rows; ++r) {
      w.scale[r] = float(sd / 64.0 * rng.uniform(0.5, 1.5));
      for (int k = 0; k < K; ++k) {
        const uint8_t c = float_to_e4m3(float(rng.normal() * 64.0));
        codes[size_t(r) * K + k] = c;
        w.code[size_t(r) * K + k] = e4m3_to_float(c);
      }
    }
    w.dev.emplace(codes);
    w.dev_scale.emplace(w.scale);
  } else {
    const std::vector<float> v = randn_bf16(rng, size_t(rows) * K, sd);
    for (size_t i = 0; i < v.size(); ++i) w.code[i] = v[i];
    const auto b = to_bf16(v);
    std::vector<uint8_t> bytes(b.size() * 2);
    std::memcpy(bytes.data(), b.data(), bytes.size());
    w.dev.emplace(bytes);
  }
  return w;
}

std::vector<int> layout(bool full, bool single) {
  const int rows[8] = {3072, 3072, 384, 24, 1024, 1024, 32, 32};
  const int input[8] = {0, 0, 0, 0, 1, 2, 1, 2};
  std::vector<int> t;
  int b = 0;
  for (int s = 0; s < (full ? 8 : 6); ++s) {
    t.insert(t.end(), {b, b + rows[s], single ? 0 : input[s]});
    b += rows[s];
  }
  return t;
}

bool test_proj(Rng& rng, int M, bool fp8, bool full, bool single) {
  const std::vector<int> seg = layout(full, single);
  const int nseg = int(seg.size() / 3), R = seg[seg.size() - 2];
  const Weights w = make_weights(rng, R, H, fp8, 0.03);
  const std::vector<float> x = randn_bf16(rng, size_t(3) * M * H, 1.0);
  DeviceArray<__nv_bfloat16> dx(to_bf16(x));
  DeviceArray<int> dseg(seg);
  DeviceArray<float> out(size_t(M) * R);
  dense::proj(dx.get(), M, w.matrix(), dseg.get(), nseg, out.get(), true, g_stream);
  sync();
  const auto got = out.download();
  std::vector<double> ref(got.size()), rnd(got.size());
  for (int s = 0; s < nseg; ++s)
    for (int r = seg[3 * s]; r < seg[3 * s + 1]; ++r)
      for (int m = 0; m < M; ++m) {
        ref[size_t(m) * R + r] = w.dot(r, &x[(size_t(seg[3 * s + 2]) * M + m) * H]);
        rnd[size_t(m) * R + r] = rbf(ref[size_t(m) * R + r]);
      }
  char what[96];
  std::snprintf(what, sizeof what, "proj M=%d %s %s%s", M, fp8 ? "fp8" : "bf16", full ? "full" : "sliding",
                single ? " single-input" : "");
  return Check{what, 1.0 / 128, 1e-6, 0.01}(got, ref, &rnd);
}

bool test_proj_conv(Rng& rng, int M, bool fp8) {
  const int K = 3072;
  const Weights w = make_weights(rng, H, K, fp8, 0.02);
  const std::vector<float> x = randn_bf16(rng, size_t(M) * K, 1.0);
  const std::vector<float> cw = randn(rng, size_t(H) * 4, 0.5);
  const std::vector<float> st0 = randn_bf16(rng, size_t(3) * H, 1.0);
  DeviceArray<__nv_bfloat16> dx(to_bf16(x));
  DeviceArray<float> dcw(cw), dst(st0), out(size_t(M) * H), hist(size_t(M) * H);
  dense::proj_conv(dx.get(), M, w.matrix(), dcw.get(), dst.get(), out.get(), hist.get(), true, g_stream);
  sync();
  const auto y = hist.download(), o = out.download(), st = dst.download();
  // the conv inputs: BF16-rounded projections
  std::vector<double> ry(y.size()), rr(y.size());
  for (int m = 0; m < M; ++m)
    for (int c = 0; c < H; ++c) {
      ry[size_t(m) * H + c] = w.dot(c, &x[size_t(m) * K]);
      rr[size_t(m) * H + c] = rbf(ry[size_t(m) * H + c]);
    }
  bool ok = Check{"proj_conv conv inputs", 1.0 / 128, 1e-6, 0.01}(y, ry, &rr);
  // the conv over [state, y] (oracle from the kernel's own y: checks the epilogue tightly)
  std::vector<double> ro(o.size()), rs(st.size());
  for (int c = 0; c < H; ++c) {
    std::vector<double> ext(M + 3);
    for (int j = 0; j < 3; ++j) ext[j] = st0[size_t(j) * H + c];
    for (int m = 0; m < M; ++m) ext[3 + m] = y[size_t(m) * H + c];
    for (int m = 0; m < M; ++m) {
      double s = 0;
      for (int j = 0; j < 4; ++j) s += double(cw[size_t(c) * 4 + j]) * ext[m + j];
      ro[size_t(m) * H + c] = (s + ext[m + 3]) / std::sqrt(5.0);
    }
    for (int j = 0; j < 3; ++j) rs[size_t(j) * H + c] = ext[M + j];
  }
  char what[64];
  std::snprintf(what, sizeof what, "proj_conv M=%d %s out", M, fp8 ? "fp8" : "bf16");
  ok &= Check{what, 1e-5, 1e-6, 1.0}(o, ro);
  ok &= Check{"proj_conv state", 0, 0, 0}(st, rs);
  return ok;
}

bool test_head(Rng& rng, int M, bool fp8) {
  const int V = 5000;
  const Weights w = make_weights(rng, V, H, fp8, 0.05);
  const std::vector<float> x = randn_bf16(rng, size_t(M) * H, 1.0);
  DeviceArray<__nv_bfloat16> dx(to_bf16(x));
  DeviceArray<float> out(size_t(M) * V);
  dense::head(dx.get(), M, w.matrix(), out.get(), true, g_stream);
  sync();
  const auto got = out.download();
  std::vector<double> ref(got.size());
  for (int m = 0; m < M; ++m)
    for (int v = 0; v < V; ++v) ref[size_t(m) * V + v] = 30.0 * std::tanh(w.dot(v, &x[size_t(m) * H]) / 30.0);
  char what[48];
  std::snprintf(what, sizeof what, "head M=%d %s", M, fp8 ? "fp8" : "bf16");
  return Check{what, 1.0 / 128, 1e-5, 1.0}(got, ref);
}

bool test_router(Rng& rng, int M, bool wf32) {
  const std::vector<float> w = randn_bf16(rng, size_t(128) * H, 0.02);
  const std::vector<float> bias = randn(rng, 128, 0.1);
  const std::vector<float> x = randn_bf16(rng, size_t(M) * H, 1.0);
  DeviceArray<__nv_bfloat16> dx(to_bf16(x)), dwb(to_bf16(w));
  DeviceArray<float> dwf(w), db(bias), wts(size_t(M) * 8);
  DeviceArray<int> idx(size_t(M) * 8);
  dense::router(dx.get(), M, wf32 ? static_cast<const void*>(dwf.get()) : static_cast<const void*>(dwb.get()), wf32,
                db.get(), idx.get(), wts.get(), true, g_stream);
  sync();
  const auto gi = idx.download();
  const auto gw = wts.download();
  bool ok = true;
  std::vector<double> rw(gw.size());
  for (int m = 0; m < M; ++m) {
    std::vector<double> s(128), v(128);
    for (int e = 0; e < 128; ++e) {
      double l = 0;
      for (int k = 0; k < H; ++k) l += double(w[size_t(e) * H + k]) * x[size_t(m) * H + k];
      s[e] = 1.0 / (1.0 + std::exp(-l));
      v[e] = s[e] + bias[e];
    }
    std::vector<int> order(128);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return v[a] > v[b]; });
    double sum = 0;
    for (int k = 0; k < 8; ++k) sum += s[order[k]];
    for (int k = 0; k < 8; ++k) {
      if (gi[size_t(m) * 8 + k] != order[k]) ok = false;
      rw[size_t(m) * 8 + k] = s[order[k]] / (sum + 1e-20) * 8.0;
    }
  }
  std::printf("%s router M=%d %s: expert ids %s\n", ok ? "PASS" : "FAIL", M, wf32 ? "fp32" : "bf16",
              ok ? "match" : "differ");
  return Check{"router weights", 1e-5, 1e-6, 1.0}(gw, rw) && ok;
}

struct MixCase {
  int nb, Q;
  bool base, write, abl, bf16, f32, partial;
};

bool test_mix(Rng& rng, int M, const MixCase& c) {
  const size_t n = size_t(M) * H;
  std::vector<std::vector<float>> blocks;
  std::vector<DeviceArray<float>> dblocks;
  for (int i = 0; i < c.nb; ++i) {
    blocks.push_back(randn(rng, n, rng.uniform(0.5, 3.0)));
    dblocks.emplace_back(blocks.back());
  }
  std::vector<const float*> ptrs;
  for (auto& d : dblocks) ptrs.push_back(d.get());
  std::vector<uint64_t> ptr_bits(ptrs.size());
  std::memcpy(ptr_bits.data(), ptrs.data(), ptrs.size() * 8);
  DeviceArray<uint64_t> table(ptr_bits.empty() ? std::vector<uint64_t>{0} : ptr_bits);
  const std::vector<float> base = randn(rng, n, 1.0), write = randn(rng, n, 0.7);
  std::vector<float> r = randn(rng, H, 1.0);
  double rn = 0;
  for (float v : r) rn += double(v) * v;
  for (auto& v : r) v = float(v / std::sqrt(rn));
  const std::vector<float> aw = {0.8f};
  const std::vector<float> q = randn(rng, size_t(c.Q) * H, 0.05), g = randn(rng, size_t(c.Q) * H, 0.1);
  std::vector<float> gw(g);
  for (auto& v : gw) v += 1.f;
  DeviceArray<float> dbase(base), dwrite(write), dr(r), daw(aw), dq(q), dg(gw), part(n), of(size_t(c.Q) * n);
  DeviceArray<__nv_bfloat16> ob(size_t(c.Q) * n);
  dense::MixParams p;
  p.blocks = c.nb ? reinterpret_cast<const float* const*>(table.get()) : nullptr;
  p.nblocks = c.nb;
  p.base = c.base ? dbase.get() : nullptr;
  p.write = c.write ? dwrite.get() : nullptr;
  p.abl_r = c.abl ? dr.get() : nullptr;
  p.abl_w = c.abl ? daw.get() : nullptr;
  p.partial_out = c.partial ? part.get() : nullptr;
  const bool skip = c.nb == 0 && c.Q == 1;
  p.queries = skip ? nullptr : dq.get();
  p.Q = c.Q;
  p.norm_w = dg.get();
  p.out_bf16 = c.bf16 ? ob.get() : nullptr;
  p.out_f32 = c.f32 ? of.get() : nullptr;
  p.M = M;
  dense::mix(p, true, g_stream);
  sync();
  // oracle
  std::vector<double> nw(n), mixr(size_t(c.Q) * n), outb(size_t(c.Q) * n), outbr(size_t(c.Q) * n);
  for (int m = 0; m < M; ++m) {
    const float* y = &write[size_t(m) * H];
    double d = 0;
    for (int k = 0; k < H; ++k) d += double(y[k]) * r[k];
    for (int k = 0; k < H; ++k) {
      double v = c.base ? base[size_t(m) * H + k] : 0.0;
      if (c.write) v += c.abl ? y[k] - aw[0] * d * r[k] : y[k];
      nw[size_t(m) * H + k] = v;
    }
    std::vector<const double*> states;
    std::vector<std::vector<double>> blk(c.nb);
    for (int i = 0; i < c.nb; ++i) {
      blk[i].assign(blocks[i].begin() + std::ptrdiff_t(m) * H, blocks[i].begin() + std::ptrdiff_t(m + 1) * H);
      states.push_back(blk[i].data());
    }
    states.push_back(&nw[size_t(m) * H]);
    for (int j = 0; j < c.Q; ++j) {
      std::vector<double> mixv(H);
      if (skip) {
        for (int k = 0; k < H; ++k) mixv[k] = nw[size_t(m) * H + k];
      } else {
        const int NS = int(states.size());
        std::vector<double> rinv(NS), lg(NS);
        double mx = -1e300;
        for (int i = 0; i < NS; ++i) {
          double ss = 0, dq = 0;
          for (int k = 0; k < H; ++k) ss += states[i][k] * states[i][k], dq += double(q[size_t(j) * H + k]) * states[i][k];
          rinv[i] = 1.0 / std::sqrt(ss / H + 1e-6);
          lg[i] = dq * rinv[i] / 48.0;
          mx = std::max(mx, lg[i]);
        }
        double den = 0;
        for (auto& l : lg) den += (l = std::exp(l - mx));
        for (int k = 0; k < H; ++k) {
          double s = 0;
          for (int i = 0; i < NS; ++i) s += lg[i] / den * states[i][k] * rinv[i];
          mixv[k] = s;
        }
      }
      double ss = 0;
      for (double v : mixv) ss += v * v;
      const double ri = 1.0 / std::sqrt(ss / H + 1e-6);
      for (int k = 0; k < H; ++k) {
        const size_t o = (size_t(j) * M + m) * H + k;
        mixr[o] = mixv[k];
        outb[o] = mixv[k] * ri * gw[size_t(j) * H + k];
        outbr[o] = rbf(outb[o]);
      }
    }
  }
  char what[96];
  std::snprintf(what, sizeof what, "mix M=%d blocks=%d Q=%d%s%s%s", M, c.nb, c.Q, c.base ? " base" : "",
                c.write ? " write" : "", c.abl ? " abl" : "");
  bool ok = true;
  if (c.partial) ok &= Check{(std::string(what) + " partial").c_str(), 1e-5, 1e-5, 1.0}(part.download(), nw);
  if (c.f32) ok &= Check{(std::string(what) + " f32").c_str(), 1e-4, 1e-5, 1.0}(of.download(), mixr);
  if (c.bf16) ok &= Check{(std::string(what) + " bf16").c_str(), 1.0 / 128, 1e-5, 0.01}(bf16_download(ob), outb, &outbr);
  return ok;
}

bool test_gates(Rng& rng, int G, int M, bool fg32) {
  const size_t n = size_t(G) * M * H;
  const std::vector<float> h = randn(rng, n, 1.0), msg = randn(rng, n, 1.0);
  const std::vector<float> wd = randn_bf16(rng, size_t(G) * 64 * 2 * H, 0.02);
  const std::vector<float> wf = randn_bf16(rng, size_t(G) * H * 64, 0.1), wg = randn_bf16(rng, size_t(G) * H * 64, 0.1);
  std::vector<float> nw = randn(rng, size_t(G) * H, 0.1);
  for (auto& v : nw) v += 1.f;
  DeviceArray<float> dh(h), dm(msg), dwf32(wf), dwg32(wg), dn(nw), of(n);
  DeviceArray<__nv_bfloat16> dwd(to_bf16(wd)), dwf16(to_bf16(wf)), dwg16(to_bf16(wg)), ob(n);
  dense::message_gates(dh.get(), dm.get(), G, M, dwd.get(), fg32 ? (const void*)dwf32.get() : (const void*)dwf16.get(),
                       fg32 ? (const void*)dwg32.get() : (const void*)dwg16.get(), fg32, dn.get(), ob.get(), of.get(),
                       true, g_stream);
  sync();
  std::vector<double> gated(n), out(n), outr(n);
  for (int g = 0; g < G; ++g)
    for (int m = 0; m < M; ++m) {
      const float* hv = &h[(size_t(g) * M + m) * H];
      const float* mv = &msg[(size_t(g) * M + m) * H];
      double sh = 0, sm = 0;
      for (int k = 0; k < H; ++k) sh += double(hv[k]) * hv[k], sm += double(mv[k]) * mv[k];
      const double rh = 1.0 / std::sqrt(sh / H + 1e-6), rm = 1.0 / std::sqrt(sm / H + 1e-6);
      std::vector<double> xs(2 * H), gd(64);
      for (int k = 0; k < H; ++k) xs[k] = rbf(hv[k] * rh), xs[H + k] = rbf(mv[k] * rm);
      for (int d = 0; d < 64; ++d) {
        double s = 0;
        for (int k = 0; k < 2 * H; ++k) s += double(wd[(size_t(g) * 64 + d) * 2 * H + k]) * xs[k];
        gd[d] = rbf(s);
      }
      double ss = 0;
      for (int c = 0; c < H; ++c) {
        double df = 0, dg = 0;
        for (int d = 0; d < 64; ++d)
          df += double(wf[(size_t(g) * H + c) * 64 + d]) * gd[d], dg += double(wg[(size_t(g) * H + c) * 64 + d]) * gd[d];
        const double v = (1 - std::tanh(df)) * hv[c] + std::tanh(dg) * mv[c];
        gated[(size_t(g) * M + m) * H + c] = v;
        ss += v * v;
      }
      const double r = 1.0 / std::sqrt(ss / H + 1e-6);
      for (int c = 0; c < H; ++c) {
        const size_t o = (size_t(g) * M + m) * H + c;
        out[o] = gated[o] * r * nw[size_t(g) * H + c];
        outr[o] = rbf(out[o]);
      }
    }
  char what[64];
  std::snprintf(what, sizeof what, "message_gates G=%d M=%d %s", G, M, fg32 ? "fp32" : "bf16");
  // the BF16 roundings of the normalised inputs and of g may flip against the oracle: loose bounds
  bool ok = Check{(std::string(what) + " gated").c_str(), 2e-3, 2e-3, 1.0}(of.download(), gated);
  ok &= Check{(std::string(what) + " out").c_str(), 1.0 / 64, 4e-3, 0.05}(bf16_download(ob), out, &outr);
  return ok;
}

bool test_embed(Rng& rng, int M) {
  const int V = 1000;
  const std::vector<float> table = randn_bf16(rng, size_t(V) * H, 0.05);
  std::vector<float> nw = randn(rng, H, 0.2);
  for (auto& v : nw) v += 1.f;
  std::vector<int64_t> tok(M);
  for (auto& t : tok) t = int64_t(rng.next() % V);
  DeviceArray<__nv_bfloat16> dt(to_bf16(table));
  DeviceArray<float> dn(nw), out(size_t(M) * H);
  DeviceArray<int64_t> dtok(tok);
  dense::embed(dtok.get(), M, dt.get(), dn.get(), out.get(), true, g_stream);
  sync();
  std::vector<double> ref(size_t(M) * H), rnd(ref.size());
  for (int m = 0; m < M; ++m) {
    const float* x = &table[size_t(tok[m]) * H];
    double ss = 0;
    for (int k = 0; k < H; ++k) ss += double(x[k]) * x[k];
    const double r = 1.0 / std::sqrt(ss / H + 1e-6);
    for (int k = 0; k < H; ++k) ref[size_t(m) * H + k] = x[k] * r * nw[k], rnd[size_t(m) * H + k] = rbf(ref[size_t(m) * H + k]);
  }
  char what[32];
  std::snprintf(what, sizeof what, "embed M=%d", M);
  return Check{what, 1.0 / 128, 1e-6, 0.01}(out.download(), ref, &rnd);
}

bool test_fp8_rows(Rng& rng) {
  const int rows = 64, K = H;
  std::vector<float> w = randn_bf16(rng, size_t(rows) * K, 0.03);
  for (int k = 0; k < K; ++k) w[size_t(5) * K + k] = 0.f;  // an all-zero row: scale 0, codes 0
  DeviceArray<__nv_bfloat16> dw(to_bf16(w));
  DeviceArray<uint8_t> q(size_t(rows) * K);
  DeviceArray<float> s(rows);
  DeviceArray<__nv_bfloat16> back(size_t(rows) * K);
  dense::fp8_rows(dw.get(), rows, K, q.get(), s.get(), g_stream);
  dense::dequant_rows(q.get(), s.get(), rows, K, back.get(), g_stream);
  sync();
  const auto gq = q.download();
  const auto gs = s.download();
  const auto gb = bf16_download(back);
  size_t bad_scale = 0, bad_code = 0, bad_back = 0;
  for (int r = 0; r < rows; ++r) {
    float amax = 0;
    for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(w[size_t(r) * K + k]));
    const float sc = amax / 448.f;
    bad_scale += gs[r] != sc;
    for (int k = 0; k < K; ++k) {
      const size_t i = size_t(r) * K + k;
      const uint8_t want = sc > 0 ? float_to_e4m3(w[i] / sc) : 0;
      bad_code += gq[i] != want;
      bad_back += gb[i] != round_bf16(e4m3_to_float(gq[i]) * gs[r]);
    }
  }
  const bool ok = bad_scale == 0 && bad_code == 0 && bad_back == 0;
  std::printf("%s fp8_rows/dequant_rows: scale mismatches %zu, code mismatches %zu, dequant mismatches %zu\n",
              ok ? "PASS" : "FAIL", bad_scale, bad_code, bad_back);
  return ok;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  Rng rng(20261120);
  bool ok = true;
  for (int M : {1, 3, 8}) {
    ok &= test_proj(rng, M, false, false, true);
    ok &= test_proj(rng, M, true, true, false);
  }
  ok &= test_proj(rng, 2, false, false, false);
  ok &= test_proj(rng, 5, true, false, true);
  for (int M : {1, 4}) {
    ok &= test_proj_conv(rng, M, false);
    ok &= test_proj_conv(rng, M, true);
  }
  for (int M : {1, 8}) {
    ok &= test_head(rng, M, false);
    ok &= test_head(rng, M, true);
    ok &= test_router(rng, M, true);
    ok &= test_router(rng, M, false);
  }
  ok &= test_router(rng, 3, false);
  const MixCase cases[] = {
      {0, 1, true, false, false, true, false, false},   // layer 0: base only, no mix
      {0, 1, true, true, true, true, true, true},       // skip path with an ablated write
      {1, 1, true, true, true, true, false, true},      // sliding attention read
      {3, 3, false, true, true, true, true, true},      // full layer at a block boundary
      {1, 4, true, true, true, false, true, true},      // loop messages (Q = 4)
      {5, 1, true, true, false, true, false, false},    // the most blocks a mix takes (5)
  };
  for (int M : {1, 3})
    for (const auto& c : cases) ok &= test_mix(rng, M, c);
  ok &= test_gates(rng, 3, 1, false);
  ok &= test_gates(rng, 3, 2, true);
  ok &= test_gates(rng, 1, 8, false);
  for (int M : {1, 3, 8}) ok &= test_embed(rng, M);
  ok &= test_fp8_rows(rng);
  INGINE_TEST_CUDA(cudaStreamDestroy(g_stream));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}

// Prefill attention against an FP64 oracle of model.Model._attend (+ flex's sink factor and the output gate):
// full-attention caches (BF16 and FP8 E4M3, keys past the relative extent), sliding rings (partially filled,
// wrapped, a stale slot, chunks longer than the window), ring_gather / key_tile_bounds / ring_update, and full
// visits read through a TP batch pool's page table (FP8 K with FP8 or FP4 V).
#include "kernels/prefill/prefill.h"
#include "prefill/prefill_test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace ingine::test;
using namespace ingine::test::prefill;
namespace pk = ingine::kernels::prefill;

namespace {

constexpr int HQ = 24, HK = 8, D = 128, REL = 1024, W = 1024;
constexpr int PROJ_LD = 8664, GATE = 3072, SINK = 6528;

struct Inputs {
  int T = 0, pos0 = 0;
  std::vector<uint16_t> q, rel, proj;  // [T,24,128], [T,24,1024], [T,PROJ_LD]
};

Inputs make_inputs(Rng& rng, int T, int pos0) {
  Inputs in;
  in.T = T, in.pos0 = pos0;
  in.q.resize(size_t(T) * HQ * D);
  for (auto& x : in.q) x = bf16_bits(float(rng.normal() * 1.2));
  in.rel.resize(size_t(T) * HQ * REL);
  for (auto& x : in.rel) x = bf16_bits(float(rng.normal() * 1.5));
  in.proj.assign(size_t(T) * PROJ_LD, 0);
  for (int t = 0; t < T; ++t) {
    for (int c = 0; c < HQ * D; ++c) in.proj[size_t(t) * PROJ_LD + GATE + c] = bf16_bits(float(rng.normal()));
    for (int h = 0; h < HQ; ++h) in.proj[size_t(t) * PROJ_LD + SINK + h] = bf16_bits(float(rng.normal() * 2.0));
  }
  return in;
}

// One key/value row [8, 128]: keys of norm <= ~22, values RMS-normalised per head.
void make_kv_row(Rng& rng, float* k, float* v) {
  for (int h = 0; h < HK; ++h) {
    double ss = 0;
    for (int d = 0; d < D; ++d) {
      k[h * D + d] = rbf(rng.normal() * 1.4);
      const double x = rng.normal();
      v[h * D + d] = float(x);
      ss += x * x;
    }
    const double r = 1.0 / std::sqrt(ss / D + 1e-6);
    for (int d = 0; d < D; ++d) v[h * D + d] = rbf(v[h * D + d] * r);
  }
}

// Reference output (BF16 roundings of model._visit emulated) for rows (t, h) with t % stride == 0.
std::vector<double> oracle(const Inputs& in, const std::vector<float>& K, const std::vector<float>& V,
                           const std::vector<int>& kpos, int window, int stride, std::vector<size_t>& index) {
  std::vector<double> out;
  const int L = int(kpos.size());
  std::vector<double> s(L);
  for (int t = 0; t < in.T; t += stride) {
    const int qp = in.pos0 + t;
    for (int h = 0; h < HQ; ++h) {
      const int g = h / 3;
      const uint16_t* q = &in.q[(size_t(t) * HQ + h) * D];
      double mx = -INFINITY;
      for (int j = 0; j < L; ++j) {
        const int d = qp - kpos[j];
        const bool ok = kpos[j] >= 0 && d >= 0 && (window == 0 || d < window);
        if (!ok) {
          s[j] = -INFINITY;
          continue;
        }
        double dot = 0;
        const float* k = &K[(size_t(j) * HK + g) * D];
        for (int e = 0; e < D; ++e) dot += double(bf16_val(q[e])) * k[e];
        s[j] = dot / std::sqrt(128.0) + (d < REL ? double(bf16_val(in.rel[(size_t(t) * HQ + h) * REL + d])) : 0.0);
        mx = std::max(mx, s[j]);
      }
      const double sink = bf16_val(in.proj[size_t(t) * PROJ_LD + SINK + h]);
      double z = 0;
      std::vector<double> o(D, 0.0);
      for (int j = 0; j < L; ++j) {
        if (s[j] == -INFINITY) continue;
        const double p = std::exp(s[j] - mx);
        z += p;
        const float* v = &V[(size_t(j) * HK + g) * D];
        for (int e = 0; e < D; ++e) o[e] += p * v[e];
      }
      const double lse = mx + std::log(z);
      const double factor = 1.0 / (1.0 + std::exp(sink - lse));
      for (int e = 0; e < D; ++e) {
        const double a = rbf(o[e] / z * factor);
        const double gate = bf16_val(in.proj[size_t(t) * PROJ_LD + GATE + h * D + e]);
        out.push_back(a / (1.0 + std::exp(-gate)));
        index.push_back((size_t(t) * HQ + h) * D + e);
      }
    }
  }
  return out;
}

bool run_attention(const char* name, const Inputs& in, const void* k_dev, const void* v_dev, bool fp8, bool q4,
                   const int* kpos_dev, const int4* bounds_dev, int L, int window, const std::vector<float>& K,
                   const std::vector<float>& V, const std::vector<int>& kpos, int stride) {
  DeviceArray<uint16_t> q(in.q), rel(in.rel), proj(in.proj), out(size_t(in.T) * HQ * D);
  pk::AttnParams p;
  p.q = reinterpret_cast<const ingine::ActT*>(q.get());
  p.k = k_dev, p.v = v_dev, p.kv_fp8 = fp8, p.kv_q4 = q4;
  p.kpos = kpos_dev, p.bounds = bounds_dev, p.L = L;
  p.T = in.T, p.pos0 = in.pos0, p.window = window;
  p.rel = reinterpret_cast<const ingine::ActT*>(rel.get());
  p.proj = reinterpret_cast<const ingine::ActT*>(proj.get()), p.ld = PROJ_LD;
  p.out = reinterpret_cast<ingine::ActT*>(out.get());
  pk::attention(p, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto got_bits = out.download();
  std::vector<size_t> index;
  const auto ref = oracle(in, K, V, kpos, window, stride, index);
  std::vector<float> got(index.size());
  for (size_t i = 0; i < index.size(); ++i) got[i] = bf16_val(got_bits[index[i]]);
  // one BF16 rounding of the output (plus a rare flip of the rounded pre-gate value) and FP16 probabilities
  return check_close(name, got, ref, 1.2e-2, 2e-4);
}

bool full_case(Rng& rng, bool fp8, bool q4, int pos0, int T) {
  const int L = pos0 + T, rows = L + 37;  // extra rows past L: never read
  std::vector<float> K(size_t(rows) * HK * D), V(K.size());
  for (int j = 0; j < rows; ++j) make_kv_row(rng, &K[size_t(j) * HK * D], &V[size_t(j) * HK * D]);
  const size_t row_bytes = q4 ? size_t(kTestKvQ4RowBytes) : size_t(HK * D) * (fp8 ? 1 : 2);
  DeviceArray<uint8_t> kd(size_t(rows) * row_bytes), vd(size_t(rows) * row_bytes);
  if (q4) {
    std::vector<uint8_t> kb(size_t(rows) * kTestKvQ4RowBytes), vb(size_t(rows) * kTestKvQ4RowBytes);
    for (int j = 0; j < rows; ++j) {
      q4_0_pack_row(&K[size_t(j) * HK * D], &kb[size_t(j) * kTestKvQ4RowBytes], &K[size_t(j) * HK * D]);
      q4_0_pack_row(&V[size_t(j) * HK * D], &vb[size_t(j) * kTestKvQ4RowBytes], &V[size_t(j) * HK * D]);
      // Prefill Q.K^T runs on BF16 MMA (store_tile rounds dequantized Q4_0 K to BF16 and V to FP16).
      for (int c = 0; c < HK * D; ++c) {
        K[size_t(j) * HK * D + c] = rbf(K[size_t(j) * HK * D + c]);
        V[size_t(j) * HK * D + c] = __half2float(__float2half_rn(V[size_t(j) * HK * D + c]));
      }
    }
    kd.upload(kb), vd.upload(vb);
  } else if (fp8) {
    std::vector<uint8_t> kb(K.size()), vb(K.size());
    for (size_t i = 0; i < K.size(); ++i) {
      kb[i] = e4m3_bits(K[i]), vb[i] = e4m3_bits(V[i]);
      K[i] = e4m3_to_float(kb[i]), V[i] = e4m3_to_float(vb[i]);
    }
    kd.upload(kb), vd.upload(vb);
  } else {
    std::vector<uint16_t> kb(K.size()), vb(K.size());
    for (size_t i = 0; i < K.size(); ++i) kb[i] = bf16_bits(K[i]), vb[i] = bf16_bits(V[i]);
    std::vector<uint8_t> kbytes(K.size() * 2), vbytes(K.size() * 2);
    std::memcpy(kbytes.data(), kb.data(), kbytes.size());
    std::memcpy(vbytes.data(), vb.data(), vbytes.size());
    kd.upload(kbytes), vd.upload(vbytes);
  }
  std::vector<int> kpos(L);
  for (int j = 0; j < L; ++j) kpos[j] = j;
  const Inputs in = make_inputs(rng, T, pos0);
  char name[96];
  std::snprintf(name, sizeof name, "full %s pos0=%d T=%d", q4 ? "q4" : (fp8 ? "fp8" : "bf16"), pos0, T);
  return run_attention(name, in, kd.get(), vd.get(), fp8, q4, nullptr, nullptr, L, 0, K, V, kpos, 1);
}

// Full visit read from a TP batch pool: 64-token pages in shuffled physical order, `visits` visits a page, FP8 K
// and FP8 or FP4 V (E2M1 code of v * 1.5, two codes a byte, low nibble first).
uint32_t e2m1_code(float x) {
  const uint32_t sgn = x < 0.f ? 8u : 0u;
  const float a = std::fabs(x);
  return sgn | uint32_t((a >= 0.25f) + (a >= 0.75f) + (a >= 1.25f) + (a >= 1.75f) + (a >= 2.5f) + (a >= 3.5f) +
                        (a >= 5.0f));
}
float e2m1_val(uint32_t c) {
  static const float kT[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float v = kT[c & 7u] * (1.f / 1.5f);
  return (c & 8u) ? -v : v;
}

bool paged_case(Rng& rng, bool v4, int pos0, int T) {
  constexpr int PAGE = 64, VISITS = 3, VISIT = 1, RD = HK * D;
  const int L = pos0 + T, npages = (L + PAGE - 1) / PAGE, phys_pages = npages + 3;
  std::vector<int> table(npages);
  for (int i = 0; i < npages; ++i) table[i] = (i * 5 + 2) % phys_pages;  // distinct (phys_pages coprime to 5 here)
  std::vector<float> K(size_t(L) * RD), V(K.size());
  for (int j = 0; j < L; ++j) make_kv_row(rng, &K[size_t(j) * RD], &V[size_t(j) * RD]);
  const size_t pool_elems = size_t(phys_pages) * VISITS * PAGE * RD;
  std::vector<uint8_t> kb(pool_elems, 0), vb(v4 ? pool_elems / 2 : pool_elems, 0);
  for (int j = 0; j < L; ++j) {
    const size_t row = ((size_t(table[j / PAGE]) * VISITS + VISIT) * PAGE + j % PAGE) * RD;
    for (int c = 0; c < RD; ++c) {
      float& k = K[size_t(j) * RD + c];
      float& v = V[size_t(j) * RD + c];
      kb[row + c] = e4m3_bits(k);
      k = e4m3_to_float(kb[row + c]);
      if (v4) {
        const uint32_t code = e2m1_code(v * 1.5f);
        vb[(row + c) / 2] |= uint8_t(code << (4 * ((row + c) & 1)));
        v = __half2float(__float2half_rn(e2m1_val(code)));  // the kernel stages values as FP16
      } else {
        vb[row + c] = e4m3_bits(v);
        v = e4m3_to_float(vb[row + c]);
      }
    }
  }
  DeviceArray<uint8_t> kd(kb.size()), vd(vb.size());
  kd.upload(kb), vd.upload(vb);
  DeviceArray<int> td(table);
  std::vector<int> kpos(L);
  for (int j = 0; j < L; ++j) kpos[j] = j;
  const Inputs in = make_inputs(rng, T, pos0);
  DeviceArray<uint16_t> q(in.q), rel(in.rel), proj(in.proj), out(size_t(T) * HQ * D);
  pk::AttnParams p;
  p.q = reinterpret_cast<const ingine::ActT*>(q.get());
  p.k = kd.get(), p.v = vd.get(), p.kv_fp8 = true, p.kv_v4 = v4;
  p.kv_page_row = td.get(), p.kv_page_visit = VISIT, p.kv_page_visits = VISITS;
  p.L = L, p.T = T, p.pos0 = pos0, p.window = 0;
  p.rel = reinterpret_cast<const ingine::ActT*>(rel.get());
  p.proj = reinterpret_cast<const ingine::ActT*>(proj.get()), p.ld = PROJ_LD;
  p.out = reinterpret_cast<ingine::ActT*>(out.get());
  pk::attention(p, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto got_bits = out.download();
  std::vector<size_t> index;
  const auto ref = oracle(in, K, V, kpos, 0, 1, index);
  std::vector<float> got(index.size());
  for (size_t i = 0; i < index.size(); ++i) got[i] = bf16_val(got_bits[index[i]]);
  char name[96];
  std::snprintf(name, sizeof name, "paged fp8k+%s pos0=%d T=%d", v4 ? "fp4v" : "fp8v", pos0, T);
  return check_close(name, got, ref, 1.2e-2, 2e-4);
}

// Sliding visit: ring state before the chunk (ring_pos per slot), chunk keys, gathered rows.
bool sliding_case(Rng& rng, int pos0, int T, bool stale, int stride) {
  std::vector<float> ringK(size_t(W) * HK * D), ringV(ringK.size());
  std::vector<int64_t> ring_pos(W, -1);
  for (int s = 0; s < W; ++s) make_kv_row(rng, &ringK[size_t(s) * HK * D], &ringV[size_t(s) * HK * D]);
  for (int p = std::max(0, pos0 - W); p < pos0; ++p) ring_pos[p % W] = p;
  if (stale) ring_pos[(pos0 + 5) % W] = pos0 + 5;  // a slot holding a position at/after the chunk start
  std::vector<uint16_t> rk(ringK.size()), rv(ringK.size());
  for (size_t i = 0; i < rk.size(); ++i) rk[i] = bf16_bits(ringK[i]), rv[i] = bf16_bits(ringV[i]);

  const int L = W + T, kpos_rows = (L + 63) / 64 * 64;
  std::vector<float> K(size_t(L) * HK * D), V(K.size());
  std::vector<int> kpos(L);
  for (int i = 0; i < W; ++i) {
    const int slot = (pos0 + i) % W;
    std::copy_n(&ringK[size_t(slot) * HK * D], HK * D, &K[size_t(i) * HK * D]);
    std::copy_n(&ringV[size_t(slot) * HK * D], HK * D, &V[size_t(i) * HK * D]);
    kpos[i] = int(ring_pos[slot]);
  }
  std::vector<uint16_t> rows_k(size_t(L) * HK * D, 0), rows_v(rows_k.size(), 0);
  for (int t = 0; t < T; ++t) {
    float* k = &K[size_t(W + t) * HK * D];
    float* v = &V[size_t(W + t) * HK * D];
    make_kv_row(rng, k, v);
    for (int c = 0; c < HK * D; ++c)
      rows_k[size_t(W + t) * HK * D + c] = bf16_bits(k[c]), rows_v[size_t(W + t) * HK * D + c] = bf16_bits(v[c]);
    kpos[W + t] = pos0 + t;
  }
  DeviceArray<uint16_t> ring_k(rk), ring_v(rv), kr(rows_k), vr(rows_v);
  DeviceArray<int64_t> rpos(ring_pos);
  DeviceArray<int> kp(kpos_rows);
  DeviceArray<int4> bounds(kpos_rows / 64);
  pk::ring_gather(reinterpret_cast<const ingine::ActT*>(ring_k.get()),
                  reinterpret_cast<const ingine::ActT*>(ring_v.get()), rpos.get(), pos0, T,
                  reinterpret_cast<ingine::ActT*>(kr.get()), reinterpret_cast<ingine::ActT*>(vr.get()), kp.get(),
                  kpos_rows, nullptr);
  pk::key_tile_bounds(kp.get(), L, bounds.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  bool ok = true;
  {
    const auto gk = kp.download();
    const auto gr = kr.download();
    size_t bad = 0;
    for (int i = 0; i < kpos_rows; ++i) bad += gk[i] != (i < L ? kpos[i] : -1);
    for (size_t i = 0; i < size_t(W) * HK * D; ++i) bad += gr[i] != bf16_bits(K[i]);
    std::printf("%s ring_gather rows/positions: %zu mismatches\n", bad ? "FAIL" : "PASS", bad);
    ok &= bad == 0;
  }
  const Inputs in = make_inputs(rng, T, pos0);
  char name[96];
  std::snprintf(name, sizeof name, "sliding pos0=%d T=%d%s", pos0, T, stale ? " stale" : "");
  ok &= run_attention(name, in, kr.get(), vr.get(), false, false, kp.get(), bounds.get(), L, W, K, V, kpos, stride);

  pk::ring_update(reinterpret_cast<const ingine::ActT*>(kr.get()), reinterpret_cast<const ingine::ActT*>(vr.get()),
                  pos0, T, reinterpret_cast<ingine::ActT*>(ring_k.get()),
                  reinterpret_cast<ingine::ActT*>(ring_v.get()), rpos.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  {
    auto exp_pos = ring_pos;
    auto exp_k = rk;
    const int n = std::min(T, W);
    for (int c = T - n; c < T; ++c) {
      const int slot = (pos0 + c) % W;
      exp_pos[slot] = pos0 + c;
      std::copy_n(&rows_k[size_t(W + c) * HK * D], HK * D, &exp_k[size_t(slot) * HK * D]);
    }
    const auto gp = rpos.download();
    const auto gk = ring_k.download();
    size_t bad = 0;
    for (int s = 0; s < W; ++s) bad += gp[s] != exp_pos[s];
    for (size_t i = 0; i < exp_k.size(); ++i) bad += gk[i] != exp_k[i];
    std::printf("%s ring_update: %zu mismatches\n", bad ? "FAIL" : "PASS", bad);
    ok &= bad == 0;
  }
  return ok;
}

// 128-CTA split-K Q4_0 decode chunk kernel (single-batch and multi-batch partial_o/m/l merge) vs FP64 oracle.
bool decode_q4_split_case(Rng& rng, int pos0, int T, int batches) {
  const int L = pos0 + T;
  std::vector<float> K(size_t(L) * HK * D), V(K.size());
  std::vector<uint8_t> kb(size_t(L) * kTestKvQ4RowBytes), vb(size_t(L) * kTestKvQ4RowBytes);
  for (int j = 0; j < L; ++j) {
    make_kv_row(rng, &K[size_t(j) * HK * D], &V[size_t(j) * HK * D]);
    q4_0_pack_row(&K[size_t(j) * HK * D], &kb[size_t(j) * kTestKvQ4RowBytes], &K[size_t(j) * HK * D]);
    q4_0_pack_row(&V[size_t(j) * HK * D], &vb[size_t(j) * kTestKvQ4RowBytes], &V[size_t(j) * HK * D]);
  }
  DeviceArray<uint8_t> kd(kb), vd(vb);
  const Inputs in = make_inputs(rng, T, pos0);
  DeviceArray<uint16_t> q(in.q), rel(in.rel), proj(in.proj), out(size_t(T) * HQ * D);
  constexpr int kSplits = 16;
  DeviceArray<float> ws(size_t(kSplits) * T * HQ * (D + 2));
  DeviceArray<float> po(size_t(T) * HQ * D), pm(size_t(T) * HQ), pl(size_t(T) * HQ);

  const int chunk = (L + batches - 1) / batches;
  for (int b = 0; b < batches; ++b) {
    const int k0 = b * chunk;
    const int bl = std::min(chunk, L - k0);
    if (bl <= 0) break;
    pk::AttnParams p;
    p.q = reinterpret_cast<const ingine::ActT*>(q.get());
    p.k = kd.get() + size_t(k0) * kTestKvQ4RowBytes;
    p.v = vd.get() + size_t(k0) * kTestKvQ4RowBytes;
    p.kv_q4 = true;
    p.k_pos_base = k0;
    p.L = bl;
    p.T = T;
    p.pos0 = pos0;
    p.window = 0;
    p.rel = reinterpret_cast<const ingine::ActT*>(rel.get());
    p.proj = reinterpret_cast<const ingine::ActT*>(proj.get());
    p.ld = PROJ_LD;
    p.out = reinterpret_cast<ingine::ActT*>(out.get());
    p.decode_ws = ws.get();
    p.decode_splits = kSplits;
    if (batches > 1) {
      p.partial_o = po.get();
      p.partial_m = pm.get();
      p.partial_l = pl.get();
      p.partial_init = (b == 0);
      p.partial_final = (k0 + bl >= L);
    }
    pk::attention(p, nullptr);
  }
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const auto got_bits = out.download();
  std::vector<int> kpos(L);
  for (int j = 0; j < L; ++j) kpos[j] = j;
  std::vector<size_t> index;
  const auto ref = oracle(in, K, V, kpos, 0, 1, index);
  std::vector<float> got(index.size());
  for (size_t i = 0; i < index.size(); ++i) got[i] = bf16_val(got_bits[index[i]]);
  char name[96];
  std::snprintf(name, sizeof name, "decode_q4_split pos0=%d T=%d batches=%d", pos0, T, batches);
  return check_close(name, got, ref, 1.2e-2, 2e-4);
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Rng rng(1234);
  bool ok = true;
  ok &= full_case(rng, false, false, 0, 45);
  ok &= full_case(rng, false, false, 1200, 77);
  ok &= full_case(rng, true, false, 1200, 77);
  ok &= full_case(rng, true, false, 3, 130);
  ok &= paged_case(rng, false, 1200, 77);
  ok &= paged_case(rng, true, 1200, 77);
  ok &= paged_case(rng, true, 3, 130);
  ok &= full_case(rng, false, true, 0, 45);
  ok &= full_case(rng, false, true, 1200, 77);
  ok &= full_case(rng, false, true, 3, 130);
  ok &= decode_q4_split_case(rng, 2500, 1, 1);
  ok &= decode_q4_split_case(rng, 2500, 2, 1);
  ok &= decode_q4_split_case(rng, 2500, 4, 3);
  ok &= decode_q4_split_case(rng, 2500, 8, 3);
  ok &= sliding_case(rng, 0, 50, false, 1);
  ok &= sliding_case(rng, 700, 300, false, 3);
  ok &= sliding_case(rng, 5000, 1100, true, 7);
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}

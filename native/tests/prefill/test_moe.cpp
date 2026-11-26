// Prefill MoE against FP64 oracles: router top-8 and weights, expert-sorted pair lists and tiles, grouped
// gate/up and down GEMMs over NVFP4 and MXFP8 experts (decoded as model.decode_experts), slot-ordered combine.
// Only the first kAlloc experts hold weights; the router bias steers every token to them (several tiles per
// expert, partial tiles).
#include "kernels/prefill/prefill.h"
#include "prefill/prefill_test.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

using namespace ingine::test;
using namespace ingine::test::prefill;
namespace pk = ingine::kernels::prefill;

namespace {

constexpr int H = 2304, I = 1152, E = 128, TOPK = 8;
constexpr int kAlloc = 10;
constexpr float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
constexpr float kAtlas2[4] = {-1.5f, -0.453125f, 0.453125f, 1.5f};
constexpr float kAtlas3[8] = {-2.0f, -1.25f, -0.703125f, -0.2265625f, 0.2265625f, 0.703125f, 1.25f, 2.0f};

float fp16_bits_to_f32(uint16_t h) {
  const uint32_t s = uint32_t(h >> 15) << 31;
  const uint32_t e = (h >> 10) & 31u;
  const uint32_t m = h & 1023u;
  uint32_t f;
  if (e == 0) {
    if (m == 0) f = s;
    else {
      int sh = 0;
      uint32_t mm = m;
      while ((mm & 1024u) == 0) { mm <<= 1; ++sh; }
      f = s | (uint32_t(127 - 14 - sh) << 23) | ((mm & 1023u) << 13);
    }
  } else if (e == 31) {
    f = s | 0x7f800000u | (m << 13);
  } else {
    f = s | ((e + 112u) << 23) | (m << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

uint16_t f32_to_fp16_bits(float v) { return __half_as_ushort(__float2half_rn(v)); }

float mcg_host(uint32_t st16) {
  const uint32_t p = st16 * 0xCBAC1FEDu;
  const uint32_t u = (p & 0x8fff8fffu) ^ 0x3b603b60u;
  const float lo = fp16_bits_to_f32(uint16_t(u & 0xffffu));
  const float hi = fp16_bits_to_f32(uint16_t(u >> 16));
  return fp16_bits_to_f32(f32_to_fp16_bits(lo + hi));
}

void fwht128_host(double* x) {
  for (int h = 1; h < 128; h <<= 1) {
    for (int i = 0; i < 128; i += 2 * h) {
      for (int j = i; j < i + h; ++j) {
        const double a = x[j], b = x[j + h];
        x[j] = a + b;
        x[j + h] = a - b;
      }
    }
  }
  const double inv = 1.0 / std::sqrt(128.0);
  for (int i = 0; i < 128; ++i) x[i] *= inv;
}

struct Mat {  // one expert-stacked matrix [kAlloc, rows, cols] in checkpoint encoding
  int fmt, rows, cols;
  std::vector<uint8_t> codes, scales;
  std::vector<uint16_t> rin, rout;
  std::vector<float> factor;
  std::vector<float> dec;  // decoded BF16 values [kAlloc, rows, cols]
  float value(int e, int r, int k) const { return dec[(size_t(e) * rows + r) * cols + k]; }
  void decode() {
    if (fmt == 4 || fmt == 5) {
      const int bits = fmt == 4 ? 2 : 3;
      const int nt = rows / 16, kt = cols / 16, tb = 32 * bits, nw = 8 * bits;
      dec.resize(size_t(kAlloc) * rows * cols);
      for (int e = 0; e < kAlloc; ++e) {
        std::vector<double> mat(size_t(rows) * cols, 0.0);
        for (int n_t = 0; n_t < nt; ++n_t) {
          for (int k_t = 0; k_t < kt; ++k_t) {
            const uint8_t* tp = codes.data() + ((size_t(e) * nt + n_t) * kt + k_t) * tb;
            uint32_t words[24];
            std::memcpy(words, tp, tb);
            for (int lane = 0; lane < 32; ++lane) {
              for (int half = 0; half < 2; ++half) {
                const int t_off = lane * 8 + half * 4;
                const int b0 = (t_off + 257) * bits - 16;
                const int w0 = (b0 >> 5) % nw, w1 = (w0 + 1) % nw, sh = b0 & 31;
                const uint64_t pair = (uint64_t(words[w0]) << 32) | uint64_t(words[w1]);
                const uint32_t win = uint32_t((pair << sh) >> 32);
                for (int m = 0; m < 4; ++m) {
                  const uint32_t st = (win >> (16 - m * bits)) & 0xffffu;
                  const int j = half * 4 + m;
                  const int kr = (lane & 3) * 2 + (j & 1) + ((j & 2) ? 8 : 0);
                  const int nc = (lane >> 2) + ((j & 4) ? 8 : 0);
                  mat[size_t(n_t * 16 + nc) * cols + (k_t * 16 + kr)] = double(mcg_host(st));
                }
              }
            }
          }
        }
        for (int r = 0; r < rows; ++r) {
          for (int cb = 0; cb < cols; cb += 128) {
            fwht128_host(&mat[size_t(r) * cols + cb]);
            for (int i = 0; i < 128; ++i)
              mat[size_t(r) * cols + cb + i] *= double(fp16_bits_to_f32(rin[size_t(e) * cols + cb + i]));
          }
        }
        for (int c = 0; c < cols; ++c) {
          for (int rb = 0; rb < rows; rb += 128) {
            double col[128];
            for (int i = 0; i < 128; ++i) col[i] = mat[size_t(rb + i) * cols + c];
            fwht128_host(col);
            for (int i = 0; i < 128; ++i)
              dec[(size_t(e) * rows + rb + i) * cols + c] =
                  float(col[i] * double(fp16_bits_to_f32(rout[size_t(e) * rows + rb + i])));
          }
        }
      }
      return;
    }
    float e4[256];
    for (int c = 0; c < 256; ++c) e4[c] = e4m3_to_float(uint8_t(c));
    dec.resize(size_t(kAlloc) * rows * cols);
    for (int e = 0; e < kAlloc; ++e)
      for (int r = 0; r < rows; ++r)
        for (int k = 0; k < cols; ++k) {
          float v;
          if (fmt == 0) {
            const uint8_t b = codes[(size_t(e) * rows + r) * (cols / 2) + k / 2];
            const float s = e4[scales[(size_t(e) * rows + r) * (cols / 16) + k / 16]];
#if INGINE_ACT_FP16
            // Volta (FP16): code x block scale is exact in FP16 and the factor scales the FP32 sums (moe_decode_v70,
            // wgemm's Volta path); rounding scale x factor to FP16 underflowed for small block scales.
            v = double(kE2M1[(k & 1) ? b >> 4 : b & 0xf] * s) * double(factor[e]);
#else
            v = rbf((kE2M1[(k & 1) ? b >> 4 : b & 0xf] * s) * factor[e]);
#endif
          } else if (fmt == 1) {
            const float c = e4[codes[(size_t(e) * rows + r) * cols + k]];
            const int s = scales[(size_t(e) * rows + r) * (cols / 32) + k / 32];
            v = rbf(c * std::ldexp(1.f, s - 127));
          } else if (fmt == 2) {
            const uint8_t b = codes[(size_t(e) * rows + r) * (cols / 4) + k / 4];
            const float s = e4[scales[(size_t(e) * rows + r) * (cols / 16) + k / 16]];
            v = rbf((kAtlas2[(b >> (2 * (k & 3))) & 3] * s) * factor[e]);
          } else {
            // ATLAS3: 32 elements -> 12 bytes [lo0(4B), lo1(4B), hi01(4B)]
            const size_t blk = (size_t(e) * rows + r) * (cols / 32) + k / 32;
            const int j = k & 31;
            const uint8_t* p = codes.data() + blk * 12;
            const uint8_t lo = (p[j / 4] >> (2 * (j & 3))) & 3;
            const uint8_t hi = (p[8 + j / 8] >> (j & 7)) & 1;
            const float s = e4[scales[(size_t(e) * rows + r) * (cols / 16) + k / 16]];
            v = rbf((kAtlas3[lo | (hi << 2)] * s) * factor[e]);
          }
          dec[(size_t(e) * rows + r) * cols + k] = v;
        }
  }
};

Mat make_mat(Rng& rng, int fmt, int rows, int cols) {
  Mat m{fmt, rows, cols, {}, {}, {}, {}, {}, {}};
  if (fmt == 0) {
    m.codes.resize(size_t(kAlloc) * rows * cols / 2);
    for (auto& b : m.codes) b = uint8_t(rng.next());
    m.scales.resize(size_t(kAlloc) * rows * cols / 16);
    for (auto& s : m.scales) s = uint8_t(((5 + rng.next() % 4) << 3) | (rng.next() & 7));  // 2^-2 .. 2^2
    m.factor.resize(E, 0.f);
    for (int e = 0; e < kAlloc; ++e) m.factor[e] = float(rng.uniform(0.01, 0.03));
  } else if (fmt == 1) {
    m.codes.resize(size_t(kAlloc) * rows * cols);
    for (auto& b : m.codes) b = uint8_t((rng.next() & 0x80) | ((1 + rng.next() % 9) << 3) | (rng.next() & 7));
    m.scales.resize(size_t(kAlloc) * rows * cols / 32);
    for (auto& s : m.scales) s = uint8_t(127 - 6 + int(rng.uniform(0, 3)));
  } else if (fmt == 2) {
    m.codes.resize(size_t(kAlloc) * rows * cols / 4);
    for (auto& b : m.codes) b = uint8_t(rng.next());
    m.scales.resize(size_t(kAlloc) * rows * cols / 16);
    for (auto& s : m.scales) s = uint8_t(((5 + rng.next() % 4) << 3) | (rng.next() & 7));
    m.factor.resize(E, 0.f);
    for (int e = 0; e < kAlloc; ++e) m.factor[e] = float(rng.uniform(0.01, 0.03));
  } else if (fmt == 4 || fmt == 5) {
    const int bits = fmt == 4 ? 2 : 3;
    m.codes.resize(size_t(kAlloc) * (rows / 16) * (cols / 16) * (32 * bits));
    for (auto& b : m.codes) b = uint8_t(rng.next());
    m.rin.resize(size_t(kAlloc) * cols);
    for (auto& r : m.rin) r = f32_to_fp16_bits(float(rng.uniform(-0.12, 0.12)));
    m.rout.resize(size_t(kAlloc) * rows);
    for (auto& r : m.rout) r = f32_to_fp16_bits(float(rng.uniform(-0.12, 0.12)));
  } else {
    m.codes.resize(size_t(kAlloc) * rows * (cols / 32) * 12);
    for (auto& b : m.codes) b = uint8_t(rng.next());
    m.scales.resize(size_t(kAlloc) * rows * cols / 16);
    for (auto& s : m.scales) s = uint8_t(((5 + rng.next() % 4) << 3) | (rng.next() & 7));
    m.factor.resize(E, 0.f);
    for (int e = 0; e < kAlloc; ++e) m.factor[e] = float(rng.uniform(0.01, 0.03));
  }
  m.decode();
  return m;
}

double act_ref(double g, double u) {
  g = rbf(g), u = rbf(u);
  return std::tanh(g / 4) * 4 * (1 / (1 + std::exp(-g))) * std::tanh(u / 25) * 25;
}

bool run(Rng& rng, int fmt, int T) {
  static constexpr const char* kFmtName[6] = {"NVFP4", "MXFP8", "ATLAS2", "ATLAS3", "EXL3_2", "EXL3_3"};
  const bool is_exl3 = fmt == 4 || fmt == 5;
  const bool has_factor = fmt != 1 && !is_exl3;
  std::printf("-- %s, T=%d\n", kFmtName[fmt], T);
  const Mat w13 = make_mat(rng, fmt, 2 * I, H), w2 = make_mat(rng, fmt, H, I);
  std::vector<float> logits(size_t(T) * E), bias(E);
  for (auto& x : logits) x = float(rng.normal() * 2);
  for (int e = 0; e < E; ++e) bias[e] = e < kAlloc ? float(rng.uniform(-0.1, 0.1)) : -1e30f;
  std::vector<uint16_t> x(size_t(T) * H);
  for (auto& v : x) v = bf16_bits(float(rng.normal()));

  DeviceArray<float> d_logits(logits), d_bias(bias), d_wts(size_t(T) * TOPK);
  DeviceArray<int> d_idx(size_t(T) * TOPK), counts(E), tiles_n(1), pair_token(size_t(T) * TOPK),
      pair_dst(size_t(T) * TOPK);
  DeviceArray<int4> tiles(pk::moe_tiles_capacity(T));
  DeviceArray<uint8_t> c13(w13.codes), s13(is_exl3 ? std::vector<uint8_t>(1) : w13.scales), c2(w2.codes),
      s2(is_exl3 ? std::vector<uint8_t>(1) : w2.scales);
  DeviceArray<uint16_t> rin13(is_exl3 ? w13.rin : std::vector<uint16_t>(1)),
      rout13(is_exl3 ? w13.rout : std::vector<uint16_t>(1)), rin2(is_exl3 ? w2.rin : std::vector<uint16_t>(1)),
      rout2(is_exl3 ? w2.rout : std::vector<uint16_t>(1));
  DeviceArray<float> f13(has_factor ? w13.factor : std::vector<float>(1)),
      f2(has_factor ? w2.factor : std::vector<float>(1));
  DeviceArray<uint16_t> d_x(x), act(size_t(T) * TOPK * I), down(size_t(T) * TOPK * H), out(size_t(T) * H);
  pk::MoeWork wk{counts.get(), tiles_n.get(), tiles.get(), pair_token.get(), pair_dst.get()};
  pk::MoeExperts ex{fmt,
                    c13.get(),
                    is_exl3 ? static_cast<const void*>(rin13.get()) : static_cast<const void*>(s13.get()),
                    has_factor ? f13.get() : nullptr,
                    c2.get(),
                    is_exl3 ? static_cast<const void*>(rin2.get()) : static_cast<const void*>(s2.get()),
                    has_factor ? f2.get() : nullptr,
                    fmt,
                    is_exl3 ? static_cast<const void*>(rout13.get()) : nullptr,
                    is_exl3 ? static_cast<const void*>(rout2.get()) : nullptr,
                    is_exl3 ? reinterpret_cast<ingine::ActT*>(down.get()) : nullptr};

  pk::moe_topk(d_logits.get(), d_bias.get(), T, d_idx.get(), d_wts.get(), counts.get(), nullptr);
  pk::moe_route(d_idx.get(), T, wk, nullptr);
  pk::moe_gate_up(ex, reinterpret_cast<const ingine::ActT*>(d_x.get()), T, wk,
                  reinterpret_cast<ingine::ActT*>(act.get()), nullptr);
  pk::moe_down(ex, reinterpret_cast<const ingine::ActT*>(act.get()), T, wk,
               reinterpret_cast<ingine::ActT*>(down.get()), nullptr);
  pk::moe_combine(reinterpret_cast<const ingine::ActT*>(down.get()), d_wts.get(), T,
                  reinterpret_cast<ingine::ActT*>(out.get()), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  bool ok = true;

  // router
  const auto idx = d_idx.download();
  const auto wts = d_wts.download();
  {
    size_t bad = 0;
    double max_rel = 0;
    for (int t = 0; t < T; ++t) {
      std::vector<int> order(E);
      std::iota(order.begin(), order.end(), 0);
      std::vector<double> s(E);
      for (int e = 0; e < E; ++e) s[e] = 1 / (1 + std::exp(-double(logits[size_t(t) * E + e])));
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return s[a] + bias[a] > s[b] + bias[b]; });
      double sum = 0;
      for (int k = 0; k < TOPK; ++k) sum += s[order[k]];
      for (int k = 0; k < TOPK; ++k) {
        bad += idx[size_t(t) * TOPK + k] != order[k];
        const double ref = s[order[k]] / (sum + 1e-20) * 8;
        max_rel = std::max(max_rel, std::fabs(wts[size_t(t) * TOPK + k] - ref) / ref);
      }
    }
    const bool pass = bad == 0 && max_rel < 1e-5;
    std::printf("%s topk: %zu index mismatches, max rel weight error %.2e\n", pass ? "PASS" : "FAIL", bad, max_rel);
    ok &= pass;
  }
  // routing
  const auto ptok = pair_token.download();
  const auto pdst = pair_dst.download();
  {
    const int nt = tiles_n.download()[0];
    const auto tl = tiles.download();
    std::vector<int> seen(size_t(T) * TOPK, 0);
    size_t bad = 0;
    int covered = 0;
    for (int i = 0; i < nt; ++i) {
      const int4 tile = tl[i];
      covered += tile.z;
      bad += tile.z <= 0 || tile.z > pk::moe_tile_rows(T);
      for (int r = 0; r < tile.z; ++r) {
        const int pp = tile.y + r, dst = pdst[pp];
        bad += dst / TOPK != ptok[pp] || idx[dst] != tile.x;
        ++seen[dst];
      }
    }
    for (int v : seen) bad += v != 1;
    bad += covered != T * TOPK;
    std::printf("%s route: %d tiles, %zu inconsistencies\n", bad ? "FAIL" : "PASS", nt, bad);
    ok &= bad == 0;
  }
  // gate/up and down on a subset of tokens; each stage from the GPU's input to it
  const auto act_g = act.download();
  const auto down_g = down.download();
  std::vector<int> pos_of(size_t(T) * TOPK);
  for (int pp = 0; pp < T * TOPK; ++pp) pos_of[pdst[pp]] = pp;
  {
    // allowances: FP32 accumulation (1e-6 of the sum of |terms|) and, for gate/up, a flip of the BF16 rounding of
    // g or u propagated through the activation; then one BF16 rounding of the output (rel 4e-3)
    std::vector<float> got_a, got_d;
    std::vector<double> ref_a, ref_d, abs_a, abs_d;
    auto act_raw = [](double g, double u) {
      return std::tanh(g / 4) * 4 * (1 / (1 + std::exp(-g))) * std::tanh(u / 25) * 25;
    };
    // Output quantization floor: 0.5 ULP of the smallest subnormal (FP16: 2^-25; BF16: ~1e-30).
#if INGINE_ACT_FP16
    constexpr double kSubnormalHalfStep = 2.98023223876953125e-8;  // 2^-25
#else
    constexpr double kSubnormalHalfStep = 1e-30;
#endif
    for (int t = 0; t < T; t += 7) {
      for (int k = 0; k < TOPK; ++k) {
        const int dst = t * TOPK + k, e = idx[dst], pp = pos_of[dst];
        for (int i = 0; i < I; i += 5) {
          double g = 0, u = 0, sg = 0, su = 0;
          for (int c = 0; c < H; ++c) {
            const double xv = bf16_val(x[size_t(t) * H + c]);
            const int gr_row = is_exl3 ? i : 2 * i;
            const int ur_row = is_exl3 ? (I + i) : (2 * i + 1);
            const double tg = w13.value(e, gr_row, c) * xv, tu = w13.value(e, ur_row, c) * xv;
            g += tg, u += tu, sg += std::fabs(tg), su += std::fabs(tu);
          }
          const double a = act_ref(g, u);
          const double gr = rbf(g), ur = rbf(u);
          const double acc_tol = is_exl3 ? 4e-4 : 1e-6;
          const double dg = acc_tol * sg + std::ldexp(std::fabs(gr), -7), du = acc_tol * su + std::ldexp(std::fabs(ur), -7);
          abs_a.push_back(std::max(std::fabs(act_raw(gr + dg, ur) - a), std::fabs(act_raw(gr - dg, ur) - a)) +
                          std::max(std::fabs(act_raw(gr, ur + du) - a), std::fabs(act_raw(gr, ur - du) - a)) +
                          kSubnormalHalfStep);
          ref_a.push_back(a);
          got_a.push_back(bf16_val(act_g[size_t(pp) * I + i]));
        }
        for (int n = 0; n < H; n += 9) {
          double acc = 0, s = 0;
          for (int c = 0; c < I; ++c) {
            const double term = w2.value(e, n, c) * double(bf16_val(act_g[size_t(pp) * I + c]));
            acc += term, s += std::fabs(term);
          }
          ref_d.push_back(acc);
          abs_d.push_back((is_exl3 ? 4e-4 : 1e-6) * s + kSubnormalHalfStep);
          got_d.push_back(bf16_val(down_g[size_t(dst) * H + n]));
        }
      }
    }
    ok &= check_close("gate_up", got_a, ref_a, is_exl3 ? 1.5e-2 : 4e-3, abs_a);
    ok &= check_close("down", got_d, ref_d, is_exl3 ? 1.5e-2 : 4e-3, abs_d);
  }
  {
    const auto out_g = out.download();
    std::vector<float> got;
    std::vector<double> ref;
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < H; ++c) {
        float acc = 0.f;
        for (int k = 0; k < TOPK; ++k) {
          const float p = bf16_val(down_g[(size_t(t) * TOPK + k) * H + c]) * wts[size_t(t) * TOPK + k];
          acc = k == 0 ? p : acc + p;
        }
        ref.push_back(rbf(acc));
        got.push_back(bf16_val(out_g[size_t(t) * H + c]));
      }
    ok &= check_close("combine (exact)", got, ref, 0.0, 0.0);
  }
  return ok;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  Rng rng(77);
  bool ok = true;
  ok &= run(rng, 0, 200);
  ok &= run(rng, 1, 200);
  ok &= run(rng, 2, 200);
  ok &= run(rng, 3, 200);
  ok &= run(rng, 0, 9);
  ok &= run(rng, 2, 9);
  ok &= run(rng, 3, 9);
  ok &= run(rng, 0, 700);  // 128-row tiles (T > 512)
  ok &= run(rng, 1, 700);
  ok &= run(rng, 2, 700);
  ok &= run(rng, 3, 700);
  ok &= run(rng, 4, 9);
  ok &= run(rng, 5, 9);
  ok &= run(rng, 4, 200);
  ok &= run(rng, 5, 200);
  ok &= run(rng, 4, 700);
  ok &= run(rng, 5, 700);
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}

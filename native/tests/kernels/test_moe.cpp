// Routed-expert kernels (src/kernels/moe) against FP64 oracles of KERNELS.md area `moe` (adapted from the
// PyTorch engine's test_moe.py): moe_up and moe_down for NVFP4 and MXFP8 experts, M = 1, 3, 8 with experts
// shared between tokens (deduplicated passes), the post-MLP conv state update and conv history, and
// dequant_experts. Synthetic experts in checkpoint encoding; only experts 0..E-1 are allocated (routing
// stays below E), which keeps GPU memory small.
#include "core/device.h"
#include "kernels/oracle.h"

#include "kernels/moe/moe.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <numeric>

using namespace ingine::test;
namespace moe = ingine::kernels::moe;

namespace {

constexpr int H = 2304, I = 1152, TOPK = 8, E = 10;
cudaStream_t g_stream = nullptr;

void sync() { INGINE_TEST_CUDA(cudaStreamSynchronize(g_stream)); }

constexpr float kAtlas2[4] = {-1.5f, -0.453125f, 0.453125f, 1.5f};
constexpr float kAtlas3[8] = {-2.0f, -1.25f, -0.703125f, -0.2265625f, 0.2265625f, 0.703125f, 1.25f, 2.0f};

const char* fmt_name(moe::Format fmt) {
  switch (fmt) {
    case moe::Format::NVFP4: return "nvfp4";
    case moe::Format::MXFP8: return "mxfp8";
    case moe::Format::ATLAS2: return "atlas2";
    case moe::Format::ATLAS3: return "atlas3";
    case moe::Format::EXL3_2: return "exl3_2";
    case moe::Format::EXL3_3: return "exl3_3";
  }
  return "?";
}

// Host FP64 reference for EXL3 MCG codebook + 128-pt normalized Sylvester Hadamard.
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

uint16_t f32_to_fp16_bits(float v) {
  return __half_as_ushort(__float2half_rn(v));
}

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

// One expert weight [E, rows, cols] in checkpoint encoding plus its decoded values (model.decode_experts).
struct ExpertWeight {
  int rows = 0, cols = 0;
  moe::Format fmt = moe::Format::NVFP4;
  std::vector<float> value;        // [E, rows, cols] decoded ActT values (matches dequant_experts)
  std::vector<float> value_exact;  // [E, rows, cols] unrounded NVFP4 (code * scale * factor) for streaming oracle
  std::optional<DeviceArray<uint8_t>> codes, scales;
  std::optional<DeviceArray<uint16_t>> rin, rout;
  std::optional<DeviceArray<float>> factor;  // [128]
};

// Per-16-element symmetric INT8 activation quantization matching moe_{up,down}_nvfp4_dp4a_kernel on sm_61/70/75.
std::vector<float> quant16_act_host(const std::vector<float>& x) {
  std::vector<float> qx(x.size());
  for (size_t base = 0; base < x.size(); base += 16) {
    float amax = 0.f;
    for (int j = 0; j < 16; ++j) amax = std::max(amax, std::fabs(x[base + j]));
    const float sc = amax * (1.0f / 127.0f);
    const float inv = amax > 0.f ? (127.0f / amax) : 0.f;
    for (int j = 0; j < 16; ++j) {
      const int q = std::max(-127, std::min(127, int(std::nearbyint(x[base + j] * inv))));
      qx[base + j] = float(q) * sc;
    }
  }
  return qx;
}

bool nvfp4_pre_sm80_stream() {
  return ingine::kActIsFP16 && ingine::current_sm_major() < 8;
}

bool nvfp4_pre_sm80_dp4a() {
  const char* v = std::getenv("INGINE_DISABLE_NVFP4_DP4A");
  const bool disabled = v && v[0] != '\0' && v[0] != '0';
  return nvfp4_pre_sm80_stream() && !disabled && ingine::current_sm_version() >= 61;
}

ExpertWeight make_expert_weight(Rng& rng, int rows, int cols, moe::Format fmt) {
  ExpertWeight w;
  w.rows = rows, w.cols = cols, w.fmt = fmt;
  const size_t n = size_t(E) * rows * cols;
  w.value.resize(n);
  std::vector<float> factor(128, 0.f);
  if (fmt == moe::Format::NVFP4) {
    std::vector<uint8_t> codes(n / 2), scales(n / 16);
    for (auto& c : codes) c = uint8_t(rng.next());
    for (auto& s : scales) s = float_to_e4m3(float(rng.uniform(16.0, 448.0)));
    for (int e = 0; e < E; ++e) factor[e] = float(rng.uniform(0.5, 1.5) * 0.002 / 448.0 * 64.0);
    w.value_exact.resize(n);
    for (size_t i = 0; i < n; ++i) {
      const int e = int(i / (size_t(rows) * cols));
      const uint8_t byte = codes[i / 2];
      const float code = kE2M1[(i & 1) ? byte >> 4 : byte & 0xF];
      const float cs = code * e4m3_to_float(scales[i / 16]);  // exact (as in FP16)
      w.value_exact[i] = cs * factor[e];
      w.value[i] = round_bf16(w.value_exact[i]);
    }
    w.codes.emplace(codes);
    w.scales.emplace(scales);
  } else if (fmt == moe::Format::MXFP8) {
    std::vector<uint8_t> codes(n), scales(n / 32);
    for (auto& c : codes) {
      c = uint8_t(rng.next());
      if ((c & 0x7f) == 0x7f) c ^= 1;  // no NaN
    }
    for (auto& s : scales) s = uint8_t(112 + rng.next() % 6);  // weights of real magnitude (|w| < 0.5)
    for (size_t i = 0; i < n; ++i) w.value[i] = round_bf16(e4m3_to_float(codes[i]) * std::ldexp(1.0f, scales[i / 32] - 127));
    w.codes.emplace(codes);
    w.scales.emplace(scales);
  } else if (fmt == moe::Format::ATLAS2) {
    std::vector<uint8_t> codes(n / 4), scales(n / 16);
    for (auto& c : codes) c = uint8_t(rng.next());
    for (auto& s : scales) s = float_to_e4m3(float(rng.uniform(16.0, 448.0)));
    for (int e = 0; e < E; ++e) factor[e] = float(rng.uniform(0.5, 1.5) * 0.002 / 448.0 * 64.0);
    for (size_t i = 0; i < n; ++i) {
      const int e = int(i / (size_t(rows) * cols));
      const uint8_t q = (codes[i / 4] >> (2 * (i & 3))) & 3u;
      const float cs = kAtlas2[q] * e4m3_to_float(scales[i / 16]);
      w.value[i] = round_bf16(cs * factor[e]);
    }
    w.codes.emplace(codes);
    w.scales.emplace(scales);
  } else if (fmt == moe::Format::EXL3_2 || fmt == moe::Format::EXL3_3) {
    const int bits = fmt == moe::Format::EXL3_2 ? 2 : 3;
    const int nt = rows / 16, kt = cols / 16, tb = 32 * bits, nw = 8 * bits;
    std::vector<uint8_t> codes(size_t(E) * nt * kt * tb);
    for (auto& c : codes) c = uint8_t(rng.next());
    std::vector<uint16_t> rin(size_t(E) * cols), rout(size_t(E) * rows);
    for (auto& r : rin) r = f32_to_fp16_bits(float(rng.uniform(-0.12, 0.12)));
    for (auto& r : rout) r = f32_to_fp16_bits(float(rng.uniform(-0.12, 0.12)));
    for (int e = 0; e < E; ++e) {
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
      // Apply H_128 along cols + rin, then H_128 along rows + rout
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
          for (int i = 0; i < 128; ++i) {
            const float val = float(col[i] * double(fp16_bits_to_f32(rout[size_t(e) * rows + rb + i])));
            w.value[(size_t(e) * rows + rb + i) * cols + c] = val;  // exact FP32 W (not BF16-materialized)
          }
        }
      }
    }
    w.codes.emplace(codes);
    w.rin.emplace(rin);
    w.rout.emplace(rout);
  } else {
    std::vector<uint8_t> codes((n * 3) / 8), scales(n / 16);
    for (auto& c : codes) c = uint8_t(rng.next());
    for (auto& s : scales) s = float_to_e4m3(float(rng.uniform(16.0, 448.0)));
    for (int e = 0; e < E; ++e) factor[e] = float(rng.uniform(0.5, 1.5) * 0.002 / 448.0 * 64.0);
    for (size_t i = 0; i < n; ++i) {
      const int e = int(i / (size_t(rows) * cols));
      const size_t blk = i / 32;
      const size_t within = i & 31;
      const size_t half = within >> 4;
      const size_t j = within & 15;
      const uint8_t* p = codes.data() + blk * 12;
      uint32_t lo;
      uint16_t hi;
      std::memcpy(&lo, p + half * 4, 4);
      std::memcpy(&hi, p + 8 + half * 2, 2);
      const uint32_t q = ((lo >> (2 * j)) & 3u) | (((hi >> j) & 1u) << 2);
      const float cs = kAtlas3[q] * e4m3_to_float(scales[i / 16]);
      w.value[i] = round_bf16(cs * factor[e]);
    }
    w.codes.emplace(codes);
    w.scales.emplace(scales);
  }
  w.factor.emplace(factor);
  return w;
}

// Routing: 8 distinct experts of 0..E-1 per token (tokens share experts).
std::vector<int> routing(Rng& rng, int M) {
  std::vector<int> idx;
  for (int m = 0; m < M; ++m) {
    std::vector<int> p(E);
    std::iota(p.begin(), p.end(), 0);
    for (int i = E - 1; i > 0; --i) std::swap(p[i], p[rng.next() % (i + 1)]);
    idx.insert(idx.end(), p.begin(), p.begin() + TOPK);
  }
  return idx;
}

double softcap(double x, double c) { return std::tanh(x / c) * c; }

bool test_up(Rng& rng, const ExpertWeight& w13, int M) {
  const std::vector<float> x = randn_bf16(rng, size_t(M) * H, 1.0);
  const std::vector<int> idx = routing(rng, M);
  DeviceArray<ingine::ActT> dx(to_bf16(x)), out(size_t(M) * TOPK * I);
  DeviceArray<int> didx(idx);
  const bool is_exl3 = w13.fmt == moe::Format::EXL3_2 || w13.fmt == moe::Format::EXL3_3;
  moe::moe_up(dx.get(), M, didx.get(), w13.codes->get(),
              is_exl3 ? static_cast<const void*>(w13.rin->get()) : static_cast<const void*>(w13.scales->get()),
              (!is_exl3 && w13.fmt != moe::Format::MXFP8) ? w13.factor->get() : nullptr, w13.fmt, out.get(), true,
              g_stream, is_exl3 ? static_cast<const void*>(w13.rout->get()) : nullptr);
  sync();
  const bool is_nvfp4_stream = (w13.fmt == moe::Format::NVFP4) && nvfp4_pre_sm80_stream();
  const bool is_nvfp4_dp4a = (w13.fmt == moe::Format::NVFP4) && nvfp4_pre_sm80_dp4a();
  const std::vector<float> x_oracle = is_nvfp4_dp4a ? quant16_act_host(x) : x;
  const std::vector<float>& w_oracle = is_nvfp4_stream ? w13.value_exact : w13.value;
  std::vector<double> ref(size_t(M) * TOPK * I), rnd(ref.size());
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < TOPK; ++k) {
      const int e = idx[size_t(m) * TOPK + k];
      for (int i = 0; i < I; ++i) {
        double gu[2] = {0, 0};
        for (int j = 0; j < 2; ++j) {
          const int wrow = is_exl3 ? (j * I + i) : (2 * i + j);
          const float* row = &w_oracle[(size_t(e) * 2 * I + wrow) * H];
          for (int h = 0; h < H; ++h) gu[j] += double(row[h]) * x_oracle[size_t(m) * H + h];
        }
        const double g = rbf(gu[0]), u = rbf(gu[1]);
        const size_t o = (size_t(m) * TOPK + k) * I + i;
        ref[o] = softcap(g, 4.0) / (1.0 + std::exp(-g)) * softcap(u, 25.0);
        rnd[o] = rbf(ref[o]);
      }
    }
  char what[48];
  std::snprintf(what, sizeof what, "moe_up %s M=%d", fmt_name(w13.fmt), M);
  // EXL3 computes rout * H_128 * (W_hat^T * FP16(H_128 * rin * x)) directly without ActT-materializing W,
  // so FP16 rounding of x_rot can flip ~18% of BF16 ULPs (~60% of 8x finer FP16 ULPs) by 1 ULP while
  // staying well inside strict tol (worst err/tol < 0.10).
  const double exl3_off = ingine::kActIsFP16 ? 0.70 : 0.25;
  const auto got = bf16_download(out);
  bool ok = Check{what, 1.0 / 64, is_exl3 ? 3e-3 : 2e-3, is_exl3 ? exl3_off : 0.02}(got, ref, &rnd);
  if (is_nvfp4_dp4a) {
    // Also verify end-to-end fidelity against unquantized FP64 reference (per-16 INT8 quant stays < 1.5% RMS).
    double err2 = 0, ref2 = 0;
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < TOPK; ++k) {
        const int e = idx[size_t(m) * TOPK + k];
        for (int i = 0; i < I; ++i) {
          double gu[2] = {0, 0};
          for (int j = 0; j < 2; ++j) {
            const float* row = &w13.value[(size_t(e) * 2 * I + 2 * i + j) * H];
            for (int h = 0; h < H; ++h) gu[j] += double(row[h]) * x[size_t(m) * H + h];
          }
          const double g = rbf(gu[0]), u = rbf(gu[1]);
          const double r0 = softcap(g, 4.0) / (1.0 + std::exp(-g)) * softcap(u, 25.0);
          const double d = double(got[(size_t(m) * TOPK + k) * I + i]) - r0;
          err2 += d * d;
          ref2 += r0 * r0;
        }
      }
    const double rel_rms = std::sqrt(err2 / std::max(ref2, 1e-30));
    const bool fid_ok = rel_rms < 0.015;
    std::printf("%s %s (dp4a vs unquantized fp64): rel_rms=%.5f (limit 0.015)\n",
                fid_ok ? "PASS" : "FAIL", what, rel_rms);
    ok &= fid_ok;
  }
  return ok;
}

bool test_down(Rng& rng, const ExpertWeight& w2, int M) {
  const std::vector<float> act = randn_bf16(rng, size_t(M) * TOPK * I, 0.5);
  const std::vector<int> idx = routing(rng, M);
  std::vector<float> wts(size_t(M) * TOPK);
  for (auto& v : wts) v = float(rng.uniform(0.2, 2.0));
  const std::vector<float> cw = randn(rng, size_t(H) * 4, 0.5);
  const std::vector<float> st0 = randn_bf16(rng, size_t(3) * H, 0.3);
  DeviceArray<ingine::ActT> dact(to_bf16(act)), dws(size_t(M) * TOPK * H);
  DeviceArray<int> didx(idx);
  DeviceArray<float> dw(wts), dcw(cw), dst(st0), out(size_t(M) * H), hist(size_t(M) * H);
  const bool is_exl3 = w2.fmt == moe::Format::EXL3_2 || w2.fmt == moe::Format::EXL3_3;
  moe::moe_down(dact.get(), M, didx.get(), dw.get(), w2.codes->get(),
                is_exl3 ? static_cast<const void*>(w2.rin->get()) : static_cast<const void*>(w2.scales->get()),
                (!is_exl3 && w2.fmt != moe::Format::MXFP8) ? w2.factor->get() : nullptr, w2.fmt, dcw.get(), dst.get(),
                out.get(), hist.get(), true, g_stream,
                is_exl3 ? static_cast<const void*>(w2.rout->get()) : nullptr, is_exl3 ? dws.get() : nullptr);
  sync();
  const auto s = hist.download(), o = out.download(), st = dst.download();
  const bool is_nvfp4_stream = (w2.fmt == moe::Format::NVFP4) && nvfp4_pre_sm80_stream();
  const bool is_nvfp4_dp4a = (w2.fmt == moe::Format::NVFP4) && nvfp4_pre_sm80_dp4a();
  const std::vector<float> act_oracle = is_nvfp4_dp4a ? quant16_act_host(act) : act;
  const std::vector<float>& w_oracle = is_nvfp4_stream ? w2.value_exact : w2.value;
  std::vector<double> rs(s.size()), rsr(s.size());
  for (int m = 0; m < M; ++m)
    for (int c = 0; c < H; ++c) {
      double sum = 0;
      for (int k = 0; k < TOPK; ++k) {
        const int e = idx[size_t(m) * TOPK + k];
        const float* row = &w_oracle[(size_t(e) * H + c) * I];
        double y = 0;
        for (int i = 0; i < I; ++i) y += double(row[i]) * act_oracle[(size_t(m) * TOPK + k) * I + i];
        sum += double(wts[size_t(m) * TOPK + k]) * rbf(y);
      }
      rs[size_t(m) * H + c] = sum;
      rsr[size_t(m) * H + c] = rbf(sum);
    }
  char what[64];
  std::snprintf(what, sizeof what, "moe_down %s M=%d expert sum", fmt_name(w2.fmt), M);
  // Each of the 8 experts rounds its FP32 dot product to BF16 before the weighted sum; when an expert dot sits at a
  // BF16 halfway tie and the 8 experts cancel near zero, a 1-ULP tie flip shifts the sum by ~|w_k * y_k| / 128.
  double max_term = 0;
  for (int m = 0; m < M; ++m)
    for (int c = 0; c < H; ++c)
      for (int k = 0; k < TOPK; ++k) {
        const int e = idx[size_t(m) * TOPK + k];
        const float* row = &w_oracle[(size_t(e) * H + c) * I];
        double y = 0;
        for (int i = 0; i < I; ++i) y += double(row[i]) * act_oracle[(size_t(m) * TOPK + k) * I + i];
        max_term = std::max(max_term, std::fabs(double(wts[size_t(m) * TOPK + k]) * y));
      }
  const double exl3_down_off = ingine::kActIsFP16 ? 0.60 : 0.15;
  bool ok = Check{what, 1.0 / 64, 1e-4 + max_term / 256.0, is_exl3 ? exl3_down_off : 0.02}(s, rs, &rsr);
  if (is_nvfp4_dp4a) {
    double err2 = 0, ref2 = 0;
    for (int m = 0; m < M; ++m)
      for (int c = 0; c < H; ++c) {
        double sum = 0;
        for (int k = 0; k < TOPK; ++k) {
          const int e = idx[size_t(m) * TOPK + k];
          const float* row = &w2.value[(size_t(e) * H + c) * I];
          double y = 0;
          for (int i = 0; i < I; ++i) y += double(row[i]) * act[(size_t(m) * TOPK + k) * I + i];
          sum += double(wts[size_t(m) * TOPK + k]) * rbf(y);
        }
        const double d = double(s[size_t(m) * H + c]) - sum;
        err2 += d * d;
        ref2 += sum * sum;
      }
    const double rel_rms = std::sqrt(err2 / std::max(ref2, 1e-30));
    const bool fid_ok = rel_rms < 0.015;
    std::printf("%s %s (dp4a vs unquantized fp64): rel_rms=%.5f (limit 0.015)\n",
                fid_ok ? "PASS" : "FAIL", what, rel_rms);
    ok &= fid_ok;
  }
  std::vector<double> ro(o.size()), rst(st.size());
  for (int c = 0; c < H; ++c) {
    std::vector<double> ext(M + 3);
    for (int j = 0; j < 3; ++j) ext[j] = st0[size_t(j) * H + c];
    for (int m = 0; m < M; ++m) ext[3 + m] = s[size_t(m) * H + c];
    for (int m = 0; m < M; ++m) {
      double a = 0;
      for (int j = 0; j < 4; ++j) a += double(cw[size_t(c) * 4 + j]) * ext[m + j];
      ro[size_t(m) * H + c] = (a + ext[m + 3]) / std::sqrt(5.0);
    }
    for (int j = 0; j < 3; ++j) rst[size_t(j) * H + c] = ext[M + j];
  }
  // FP32 sums of up to five terms: the absolute error follows the largest input, not the (cancelling) result
  double big = 0;
  for (float v : s) big = std::max(big, double(std::fabs(v)));
  for (float v : st0) big = std::max(big, double(std::fabs(v)));
  ok &= Check{"moe_down conv out", 1e-5, 4e-7 * big + 1e-6, 1.0}(o, ro);
  ok &= Check{"moe_down conv state", 0, 0, 0}(st, rst);
  return ok;
}

bool test_dequant(const ExpertWeight& w) {
  const std::vector<int> experts = {E - 1, 0, 3};
  DeviceArray<int> de(experts);
  DeviceArray<ingine::ActT> out(experts.size() * w.rows * w.cols);
  const bool is_exl3 = w.fmt == moe::Format::EXL3_2 || w.fmt == moe::Format::EXL3_3;
  moe::dequant_experts(w.codes->get(),
                       is_exl3 ? static_cast<const void*>(w.rin->get()) : static_cast<const void*>(w.scales->get()),
                       (!is_exl3 && w.fmt != moe::Format::MXFP8) ? w.factor->get() : nullptr, de.get(),
                       int(experts.size()), w.rows, w.cols, w.fmt, out.get(), g_stream,
                       is_exl3 ? static_cast<const void*>(w.rout->get()) : nullptr);
  sync();
  const auto got = bf16_download(out);
  size_t bad = 0;
  double max_diff = 0.0;
  for (size_t s = 0; s < experts.size(); ++s)
    for (size_t i = 0; i < size_t(w.rows) * w.cols; ++i) {
      const float g = got[s * w.rows * w.cols + i], r = w.value[size_t(experts[s]) * w.rows * w.cols + i];
      if (is_exl3) {
        const double d = std::fabs(double(g) - double(round_bf16(r)));
        max_diff = std::max(max_diff, d);
        bad += d > std::max(2e-4, 0.02 * std::fabs(double(r)));
      } else {
        bad += g != r;
      }
    }
  std::printf("%s dequant_experts %s [%d, %d]: %zu mismatches (max_diff=%.2e)\n", bad ? "FAIL" : "PASS",
              fmt_name(w.fmt), w.rows, w.cols, bad, max_diff);
  return bad == 0;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
  Rng rng(1152);
  bool ok = true;
  for (moe::Format fmt : {moe::Format::NVFP4, moe::Format::MXFP8, moe::Format::ATLAS2, moe::Format::ATLAS3,
                          moe::Format::EXL3_2, moe::Format::EXL3_3}) {
    const ExpertWeight w13 = make_expert_weight(rng, 2 * I, H, fmt);
    for (int M : {1, 2, 3, 8})
      if (M != 2 || fmt == moe::Format::NVFP4) ok &= test_up(rng, w13, M);
    ok &= test_dequant(w13);
  }
  for (moe::Format fmt : {moe::Format::NVFP4, moe::Format::MXFP8, moe::Format::ATLAS2, moe::Format::ATLAS3,
                          moe::Format::EXL3_2, moe::Format::EXL3_3}) {
    const ExpertWeight w2 = make_expert_weight(rng, H, I, fmt);
    for (int M : {1, 2, 3, 8})
      if (M != 2 || fmt == moe::Format::NVFP4) ok &= test_down(rng, w2, M);
    ok &= test_dequant(w2);
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(g_stream));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}

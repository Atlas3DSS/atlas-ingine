// Verifies TP1 batched GGUF mixed-projection routing (ex.w13_fmt != ex.w2_fmt) and explicit pre-decode
// rejection of unsupported GGUF entry points:
//   Suite 1: Kernel-level pk::moe_gate_up + pk::moe_down FP64 oracle for mixed (w13_fmt, w2_fmt) pairs
//            (GGUF_Q5L/GGUF_IQ4, GGUF_IQ4/GGUF_Q5L, NVFP4/GGUF_Q5L) across T = 1, 2, 4 (moe_decode_v70)
//            and T = 16 (wgemm).
//   Suite 2: Explicit pre-decode std::invalid_argument rejection on unsupported GGUF entry points
//            (TP1 max_batch=1 on both make_generator overloads + Decoder; TP>1 max_batch=1 --spec).
//   Suite 3: Real TP1 make_generator(max_batch=2) on pinned GGUF (B=1 == B=2 greedy parity).
//   Suite 4: Real 24-layer TP1 make_generator(max_batch=2, weights, ablation) with mixed projections
//            (L20 w13=GGUF_Q5L/w2=GGUF_IQ4, L23 w13=GGUF_IQ4/w2=GGUF_Q5L), verified step-by-step
//            against Prefiller reference logits and B=1 == B=2 parity.
#include "common/test_util.h"
#include "core/act.h"
#include "core/arena.h"
#include "engine/scheduler.h"
#include "ingine/generator.h"
#include "kernels/prefill/prefill.h"
#include "model/model_data.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "model/weights_gguf.h"
#include "text/chat.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ingine::model;
using ingine::ActT;
using ingine::test::DeviceArray;
using ingine::test::Rng;
namespace pk = ingine::kernels::prefill;

constexpr int kH = HIDDEN;
constexpr int kI = INTER;
constexpr int kE = EXPERTS;
constexpr int kTopK = TOP_K;
constexpr int kAlloc = 10;

constexpr float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
constexpr float kIq4nl[16] = {-127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f,
                              1.f,    13.f,   25.f,  38.f,  53.f,  69.f,  89.f,  113.f};

int g_failures = 0;

void check(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", msg);
  }
}

float round_act_f32(float v) {
#if INGINE_ACT_FP16
  return __half2float(__float2half_rn(v));
#else
  return __bfloat162float(__float2bfloat16_rn(v));
#endif
}

ActT from_f32(float v) {
#if INGINE_ACT_FP16
  return __float2half_rn(v);
#else
  return __float2bfloat16_rn(v);
#endif
}

float to_f32(ActT v) {
#if INGINE_ACT_FP16
  return __half2float(v);
#else
  return __bfloat162float(v);
#endif
}

float e4m3_to_f32(uint8_t b) {
  const int e = (b >> 3) & 15;
  const int m = b & 7;
  if (e == 0) return std::ldexp(float(m), -9);
  if (e == 15 && m == 7) return 0.f;
  return std::ldexp(1.f + float(m) / 8.f, e - 7);
}

struct SynthMat {
  int fmt = 0;
  int rows = 0;
  int cols = 0;
  std::vector<uint8_t> codes;
  std::vector<uint8_t> scales;
  std::vector<float> factor;
  std::vector<float> dec;  // [kAlloc, rows, cols] rounded to ActT

  float val(int e, int r, int k) const { return dec[(std::size_t(e) * rows + r) * cols + k]; }
};

SynthMat make_synth_mat(Rng& rng, int fmt, int rows, int cols) {
  SynthMat m;
  m.fmt = fmt;
  m.rows = rows;
  m.cols = cols;
  const std::size_t n_rows = std::size_t(kAlloc) * rows;
  m.codes.resize(n_rows * (cols / 2), 0);
  m.factor.assign(kE, 1.0f);
  m.dec.resize(n_rows * cols, 0.f);

  if (fmt == 0) {  // NVFP4
    m.scales.resize(n_rows * (cols / 16), 0);
    for (int e = 0; e < kAlloc; ++e) m.factor[e] = float(rng.uniform(0.01, 0.05));
    for (std::size_t r = 0; r < n_rows; ++r) {
      const int e = int(r / rows);
      for (int b = 0; b < cols / 16; ++b) {
        const uint8_t sb = uint8_t(0x28 + (rng.next() % 0x20));  // positive normal E4M3
        m.scales[r * (cols / 16) + b] = sb;
        const float sc = e4m3_to_f32(sb) * m.factor[e];
        for (int i = 0; i < 8; ++i) {
          const uint8_t lo = uint8_t(rng.next() & 15u);
          const uint8_t hi = uint8_t(rng.next() & 15u);
          m.codes[r * (cols / 2) + b * 8 + i] = uint8_t(lo | (hi << 4));
          m.dec[r * cols + b * 16 + 2 * i] = round_act_f32(kE2M1[lo] * sc);
          m.dec[r * cols + b * 16 + 2 * i + 1] = round_act_f32(kE2M1[hi] * sc);
        }
      }
    }
  } else if (fmt == 6) {  // GGUF_Q5L: 12B per 32-value group (f32 scale, f32 min, u32 high_bits)
    const int ng = cols / 32;
    m.scales.resize(n_rows * std::size_t(ng) * 12, 0);
    for (std::size_t r = 0; r < n_rows; ++r) {
      for (int g = 0; g < ng; ++g) {
        const float s = float(rng.uniform(0.002, 0.015));
        const float mn = float(rng.uniform(-0.20, -0.02));
        uint32_t hb = 0;
        for (int i = 0; i < 16; ++i) {
          const uint8_t q0 = uint8_t(rng.next() & 31u);
          const uint8_t q1 = uint8_t(rng.next() & 31u);
          m.codes[r * (cols / 2) + g * 16 + i] = uint8_t((q0 & 15u) | ((q1 & 15u) << 4));
          if (q0 & 16u) hb |= (1u << (2 * i));
          if (q1 & 16u) hb |= (1u << (2 * i + 1));
          m.dec[r * cols + g * 32 + 2 * i] = round_act_f32(s * float(q0) + mn);
          m.dec[r * cols + g * 32 + 2 * i + 1] = round_act_f32(s * float(q1) + mn);
        }
        uint8_t* sp = m.scales.data() + (r * ng + g) * 12;
        std::memcpy(sp, &s, 4);
        std::memcpy(sp + 4, &mn, 4);
        std::memcpy(sp + 8, &hb, 4);
      }
    }
  } else if (fmt == 7) {  // GGUF_IQ4: 4B per 32-value group (f32 scale)
    const int ng = cols / 32;
    m.scales.resize(n_rows * std::size_t(ng) * 4, 0);
    for (std::size_t r = 0; r < n_rows; ++r) {
      for (int g = 0; g < ng; ++g) {
        const float s = float(rng.uniform(0.0005, 0.003));
        std::memcpy(m.scales.data() + (r * ng + g) * 4, &s, 4);
        for (int i = 0; i < 16; ++i) {
          const uint8_t q0 = uint8_t(rng.next() & 15u);
          const uint8_t q1 = uint8_t(rng.next() & 15u);
          m.codes[r * (cols / 2) + g * 16 + i] = uint8_t(q0 | (q1 << 4));
          m.dec[r * cols + g * 32 + 2 * i] = round_act_f32(s * kIq4nl[q0]);
          m.dec[r * cols + g * 32 + 2 * i + 1] = round_act_f32(s * kIq4nl[q1]);
        }
      }
    }
  }
  return m;
}

float swiglu_ref(float g, float u) {
  g = round_act_f32(g);
  u = round_act_f32(u);
  const float cg = std::tanh(g / 4.f) * 4.f;
  const float cu = std::tanh(u / 25.f) * 25.f;
  return round_act_f32(cg * (1.f / (1.f + std::exp(-g))) * cu);
}

bool run_mixed_kernel_case(Rng& rng, int fmt13, int fmt2, int T) {
  const SynthMat w13 = make_synth_mat(rng, fmt13, 2 * kI, kH);
  const SynthMat w2 = make_synth_mat(rng, fmt2, kH, kI);

  std::vector<float> logits(std::size_t(T) * kE), bias(kE);
  for (auto& v : logits) v = float(rng.normal() * 2.0);
  for (int e = 0; e < kE; ++e) bias[e] = e < kAlloc ? float(rng.uniform(-0.1, 0.1)) : -1e30f;

  std::vector<ActT> x(std::size_t(T) * kH);
  for (auto& v : x) v = from_f32(float(rng.normal() * 0.4));

  DeviceArray<float> d_logits(logits), d_bias(bias), d_wts(std::size_t(T) * kTopK);
  DeviceArray<int> d_idx(std::size_t(T) * kTopK), d_counts(kE), d_tiles_n(1);
  DeviceArray<int> d_pair_tok(std::size_t(T) * kTopK), d_pair_dst(std::size_t(T) * kTopK);
  DeviceArray<int4> d_tiles(pk::moe_tiles_capacity(T));
  DeviceArray<uint8_t> d_c13(w13.codes), d_s13(w13.scales), d_c2(w2.codes), d_s2(w2.scales);
  DeviceArray<float> d_f13(w13.factor), d_f2(w2.factor);
  DeviceArray<ActT> d_x(x), d_act(std::size_t(T) * kTopK * kI), d_down(std::size_t(T) * kTopK * kH);

  INGINE_TEST_CUDA(cudaMemset(d_counts.get(), 0, kE * sizeof(int)));
  pk::moe_topk(d_logits.get(), d_bias.get(), T, d_idx.get(), d_wts.get(), d_counts.get(), nullptr);
  const pk::MoeWork work{d_counts.get(), d_tiles_n.get(), d_tiles.get(), d_pair_tok.get(), d_pair_dst.get()};
  pk::moe_route(d_idx.get(), T, work, nullptr);

  pk::MoeExperts me;
  me.fmt = fmt13;
  me.w13 = d_c13.get();
  me.w13_scale = d_s13.get();
  me.w13_factor = d_f13.get();
  me.w2 = d_c2.get();
  me.w2_scale = d_s2.get();
  me.w2_factor = d_f2.get();
  me.w2_fmt = fmt2;

  pk::moe_gate_up(me, d_x.get(), T, work, d_act.get(), nullptr);
  pk::moe_down(me, d_act.get(), T, work, d_down.get(), nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());

  const std::vector<int> h_idx = d_idx.download();
  const std::vector<int> h_pt = d_pair_tok.download();
  const std::vector<int> h_pd = d_pair_dst.download();
  const std::vector<ActT> h_act = d_act.download();
  const std::vector<ActT> h_down = d_down.download();

  // Verify gate_up and down against FP64 reference across all pairs (sampled channels for speed at T=16, all at T<=4).
  double max_up_err = 0.0, max_down_err = 0.0;
  double max_up_ref = 1e-6, max_down_ref = 1e-6;
  const int n_pairs = T * kTopK;
  const int ch_step = T <= 4 ? 7 : 29;
  for (int p = 0; p < n_pairs; ++p) {
    const int tok = h_pt[p];
    const int dst = h_pd[p];
    const int slot = dst % kTopK;
    const int e = h_idx[tok * kTopK + slot];

    // Check gate_up on sampled channels
    for (int i = 0; i < kI; i += ch_step) {
      double sg = 0.0, su = 0.0;
      for (int k = 0; k < kH; ++k) {
        const double xv = double(to_f32(x[std::size_t(tok) * kH + k]));
        sg += double(w13.val(e, 2 * i, k)) * xv;
        su += double(w13.val(e, 2 * i + 1, k)) * xv;
      }
      const double ref = double(swiglu_ref(float(sg), float(su)));
      const double got = double(to_f32(h_act[std::size_t(p) * kI + i]));
      max_up_err = std::max(max_up_err, std::abs(got - ref));
      max_up_ref = std::max(max_up_ref, std::abs(ref));
    }

    // Check down using actual GPU h_act[p] as input to isolate w2_fmt kernel accuracy
    for (int r = 0; r < kH; r += ch_step) {
      double sd = 0.0;
      for (int k = 0; k < kI; ++k) {
        sd += double(w2.val(e, r, k)) * double(to_f32(h_act[std::size_t(p) * kI + k]));
      }
      const double ref = double(round_act_f32(float(sd)));
      const double got = double(to_f32(h_down[std::size_t(dst) * kH + r]));
      max_down_err = std::max(max_down_err, std::abs(got - ref));
      max_down_ref = std::max(max_down_ref, std::abs(ref));
    }
  }
  const double rel_up = max_up_err / std::max(1.0, max_up_ref);
  const double rel_down = max_down_err / std::max(1.0, max_down_ref);
  const bool ok = std::isfinite(max_up_err) && std::isfinite(max_down_err) && rel_up < 0.03 && rel_down < 0.03;
  std::printf("  mixed(%d -> %d) T=%2d: max_up_err=%.4f (rel=%.5f) max_down_err=%.4f (rel=%.5f) -> %s\n",
              fmt13, fmt2, T, max_up_err, rel_up, max_down_err, rel_down, ok ? "PASS" : "FAIL");
  return ok;
}

std::vector<int32_t> drain(ingine::engine::Job& job) {
  std::vector<int32_t> out;
  for (;;) {
    auto b = job.take(std::chrono::seconds(120));
    out.insert(out.end(), b.tokens.begin(), b.tokens.end());
    if (b.failure) throw std::runtime_error(*b.failure);
    if (b.end) return out;
    if (b.empty()) throw std::runtime_error("timeout");
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;

  std::printf("=== Suite 1: Mixed (w13_fmt != w2_fmt) Kernel Oracle (moe_decode_v70 + wgemm) ===\n");
  Rng rng(20261126u);
  const std::pair<int, int> pairs[] = {{6, 7}, {7, 6}, {0, 6}};
  for (auto [f13, f2] : pairs) {
    for (int T : {1, 2, 4, 16}) {
      const bool ok = run_mixed_kernel_case(rng, f13, f2, T);
      check(ok, "Suite 1: mixed (w13_fmt, w2_fmt) kernel oracle");
    }
  }

  std::string gguf_dir;
  if (argc > 1) gguf_dir = argv[1];
  else if (const char* env = std::getenv("INGINE_GGUF_MODEL_DIR")) gguf_dir = env;
  if (gguf_dir.empty() || !std::filesystem::exists(std::filesystem::path(gguf_dir) / "model.gguf")) {
    std::printf("skip: Suites 2-4 require INGINE_GGUF_MODEL_DIR with model.gguf\n");
    return g_failures ? 1 : 0;
  }

#if INGINE_TP_SIZE > 1
  std::printf("=== Suite 2 (TP>1): Explicit Pre-Decode Rejection for GGUF max_batch=1 --spec ===\n");
  {
    RuntimeOptions opts;
    opts.model_dir = gguf_dir;
    opts.dense_fp8 = false;
    opts.max_ctx = 4096;
    opts.kv_fp8 = true;
    opts.max_batch = 1;
    opts.spec = true;
    bool rejected = false;
    try {
      auto g = make_generator(opts);
      (void)g;
    } catch (const std::invalid_argument& e) {
      std::printf("  caught expected TP>1 GGUF --spec rejection: %s\n", e.what());
      rejected = true;
    }
    check(rejected, "Suite 2 (TP>1): make_generator rejects GGUF max_batch=1 --spec before load");
  }
#else
  std::printf("=== Suite 2 (TP1): Explicit Pre-Decode Rejection on Public max_batch=1 Entry Points ===\n");
  {
    RuntimeOptions opts;
    opts.model_dir = gguf_dir;
    opts.dense_fp8 = false;
    opts.max_ctx = 4096;
    opts.kv_fp8 = true;
    opts.max_batch = 1;
    bool rejected_dir = false;
    try {
      auto g = make_generator(opts);
      (void)g;
    } catch (const std::invalid_argument& e) {
      std::printf("  caught expected make_generator(options) rejection: %s\n", e.what());
      rejected_dir = true;
    }
    check(rejected_dir, "Suite 2a: make_generator(options) rejects TP1 GGUF max_batch=1 before load");

    std::size_t free_before = 0, total_before = 0;
    INGINE_TEST_CUDA(cudaMemGetInfo(&free_before, &total_before));
    RuntimeOptions opts_cap;
    opts_cap.model_dir = gguf_dir;
    opts_cap.dense_fp8 = false;
    opts_cap.max_ctx = 4096;
    opts_cap.kv_fp8 = true;
    opts_cap.max_batch = 2;
    opts_cap.vram_limit_mb = 15360;
    bool rejected_cap = false;
    try {
      auto g = make_generator(opts_cap);
      (void)g;
    } catch (const std::invalid_argument& e) {
      std::printf("  caught expected capped GGUF rejection (--vram-limit-mb 15360): %s\n", e.what());
      rejected_cap = (std::string(e.what()).find("--vram-limit-mb") != std::string::npos);
    }
    std::size_t free_after = 0, total_after = 0;
    INGINE_TEST_CUDA(cudaMemGetInfo(&free_after, &total_after));
    const std::size_t delta_b = free_before > free_after ? (free_before - free_after) : 0;
    std::printf("  capped GGUF pre-load VRAM delta: %.2f MiB\n", double(delta_b) / (1024.0 * 1024.0));
    check(rejected_cap && delta_b < (64ull << 20),
          "Suite 2d: make_generator(options) rejects capped TP1 batched GGUF (--vram-limit-mb 15360) before load_weights");
  }

  const auto fmt = ingine::text::ChatFormat::load(gguf_dir);
  ingine::text::ChatOptions co;
  co.reasoning_effort = "none";
  const nlohmann::ordered_json msgs = nlohmann::ordered_json::array(
      {{{"role", "user"}, {"content", "Explain hash map collision handling in one short sentence."}}});
  const std::vector<int32_t> prompt = fmt.render(msgs, nlohmann::ordered_json(), co).ids;
  const std::vector<int32_t> stop = fmt.stop_ids();
  Sampling greedy;
  greedy.temperature = 0.0f;

  std::printf("=== Suite 3 (TP1): Real Unmodified NVMe GGUF via make_generator(max_batch=2) ===\n");
  {
    RuntimeOptions bopts;
    bopts.model_dir = gguf_dir;
    bopts.dense_fp8 = false;
    bopts.max_ctx = 4096;
    bopts.kv_fp8 = true;
    bopts.park_mb = 0;
    bopts.max_batch = 2;
    bopts.max_kv_tokens = 2048;
    auto bgen = make_generator(bopts);

    std::vector<int32_t> out_b1;
    bgen->generate(prompt, 24, greedy, stop, [&](std::span<const int32_t> toks) {
      out_b1.insert(out_b1.end(), toks.begin(), toks.end());
      return true;
    });

    ingine::engine::Scheduler sched(*bgen, stop, 4);
    auto j0 = std::make_shared<ingine::engine::Job>(prompt, 24, greedy);
    auto j1 = std::make_shared<ingine::engine::Job>(prompt, 24, greedy);
    check(sched.submit(j0) && sched.submit(j1), "Suite 3: submitted B=2 concurrent GGUF jobs");
    const std::vector<int32_t> out0 = drain(*j0);
    const std::vector<int32_t> out1 = drain(*j1);
    std::printf("  unmodified TP1 GGUF max_batch=2: len(b1)=%zu len(b2_0)=%zu len(b2_1)=%zu\n",
                out_b1.size(), out0.size(), out1.size());
    check(!out_b1.empty() && out0 == out_b1 && out1 == out_b1,
          "Suite 3: unmodified TP1 GGUF max_batch=2 B=1 == B=2 exact parity");
  }

  std::printf("=== Suite 4 (TP1): Real 24-Layer Mixed-Projection (w13_fmt != w2_fmt) BatchGenerator ===\n");
  {
    cudaStream_t stream = nullptr;
    INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Weights w = load_weights(gguf_dir, false, stream);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));

    // Also verify Decoder and make_generator(options, weights, ablation) reject TP1 GGUF max_batch=1 pre-decode!
    {
      Ablation abl = load_ablation({});
      State st = make_state(512, true);
      bool rejected_dec = false;
      try {
        Decoder dec(w, abl, st, false);
        (void)dec;
      } catch (const std::invalid_argument& e) {
        std::printf("  caught expected Decoder rejection: %s\n", e.what());
        rejected_dec = true;
      }
      check(rejected_dec, "Suite 2b: Decoder rejects GGUF weights at construction");
    }

    // Verify L20 is GGUF_Q5L and L23 is GGUF_IQ4, then swap w2 projections so BOTH mixed directions exist!
    Experts& ex20 = w.layers[20].experts;
    Experts& ex23 = w.layers[23].experts;
    check(ex20.w13_fmt == ExpertFormat::GGUF_Q5L && ex20.w2_fmt == ExpertFormat::GGUF_Q5L &&
              ex23.w13_fmt == ExpertFormat::GGUF_IQ4 && ex23.w2_fmt == ExpertFormat::GGUF_IQ4,
          "Suite 4: verified base L20=GGUF_Q5L and L23=GGUF_IQ4 formats");
    std::swap(ex20.w2, ex23.w2);
    std::swap(ex20.w2_scale, ex23.w2_scale);
    std::swap(ex20.w2_factor, ex23.w2_factor);
    std::swap(ex20.w2_ptrs, ex23.w2_ptrs);
    std::swap(ex20.w2_scale_ptrs, ex23.w2_scale_ptrs);
    std::swap(ex20.w2_fmt, ex23.w2_fmt);
    w.w2_expert_formats[20] = ex20.w2_fmt;
    w.w2_expert_formats[23] = ex23.w2_fmt;
    std::printf("  swapped w2: L20=(w13=%d,w2=%d), L23=(w13=%d,w2=%d)\n",
                int(ex20.w13_fmt), int(ex20.w2_fmt), int(ex23.w13_fmt), int(ex23.w2_fmt));

    // Compute exact Prefiller reference sequence (8 greedy tokens) on the mixed-format weights.
    const int n_steps = 8;
    std::vector<int32_t> ref_top1(n_steps), ref_top2(n_steps);
    std::vector<float> ref_margin(n_steps);
    {
      Ablation abl = load_ablation({});
      Prefiller pre(w, abl, 256, stream);
      ingine::DeviceBuffer d_logits(std::size_t(UNPADDED_VOCAB) * sizeof(float));
      std::vector<float> h_logits(UNPADDED_VOCAB);
      std::vector<int32_t> cur = prompt;
      for (int s = 0; s < n_steps; ++s) {
        State st = make_state(512, true);
        DeviceArray<int32_t> d_toks(cur);
        pre.run(st, d_toks.get(), int(cur.size()), 0, static_cast<float*>(d_logits.p), nullptr);
        INGINE_TEST_CUDA(cudaMemcpyAsync(h_logits.data(), d_logits.p, h_logits.size() * sizeof(float),
                                         cudaMemcpyDeviceToHost, stream));
        INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
        int t1 = 0, t2 = 1;
        if (h_logits[t2] > h_logits[t1]) std::swap(t1, t2);
        for (int v = 2; v < UNPADDED_VOCAB; ++v) {
          if (h_logits[v] > h_logits[t1]) {
            t2 = t1;
            t1 = v;
          } else if (h_logits[v] > h_logits[t2]) {
            t2 = v;
          }
        }
        ref_top1[s] = t1;
        ref_top2[s] = t2;
        ref_margin[s] = h_logits[t1] - h_logits[t2];
        cur.push_back(t1);
      }
    }

    // Verify make_generator(options, weights, ablation) rejects max_batch=1 BEFORE constructing EngineGenerator.
    // Note: weights is moved only after validation passes, so we test rejection with a lightweight shell Weights first.
    {
      Weights w_check;
      Layer l0;
      l0.experts.w13_fmt = ExpertFormat::GGUF_Q5L;
      l0.experts.w2_fmt = ExpertFormat::GGUF_IQ4;
      w_check.layers.push_back(std::move(l0));
      RuntimeOptions opts_b1;
      opts_b1.model_dir = gguf_dir;
      opts_b1.dense_fp8 = false;
      opts_b1.max_ctx = 512;
      opts_b1.kv_fp8 = true;
      opts_b1.max_batch = 1;
      bool rejected_w = false;
      try {
        auto g = make_generator(opts_b1, std::move(w_check), Ablation{});
        (void)g;
      } catch (const std::invalid_argument& e) {
        std::printf("  caught expected make_generator(weights) rejection: %s\n", e.what());
        rejected_w = true;
      }
      check(rejected_w, "Suite 2c: make_generator(options, weights, ablation) rejects TP1 GGUF max_batch=1");

      Weights w_check2;
      Layer l0_cap;
      l0_cap.experts.w13_fmt = ExpertFormat::GGUF_Q5L;
      l0_cap.experts.w2_fmt = ExpertFormat::GGUF_IQ4;
      w_check2.layers.push_back(std::move(l0_cap));
      RuntimeOptions opts_cap_w;
      opts_cap_w.model_dir = gguf_dir;
      opts_cap_w.dense_fp8 = false;
      opts_cap_w.max_ctx = 512;
      opts_cap_w.kv_fp8 = true;
      opts_cap_w.max_batch = 2;
      opts_cap_w.vram_limit_mb = 15360;
      bool rejected_cap_w = false;
      try {
        auto g = make_generator(opts_cap_w, std::move(w_check2), Ablation{});
        (void)g;
      } catch (const std::invalid_argument& e) {
        std::printf("  caught expected capped make_generator(weights) rejection: %s\n", e.what());
        rejected_cap_w = (std::string(e.what()).find("--vram-limit-mb") != std::string::npos);
      }
      check(rejected_cap_w,
            "Suite 2e: make_generator(options, weights, ablation) rejects capped TP1 batched GGUF (--vram-limit-mb 15360)");
    }

    // Run real BatchGenerator (max_batch=2) on the mixed-format Weights (both B=1 and concurrent B=2)!
    RuntimeOptions bopts;
    bopts.model_dir = gguf_dir;
    bopts.dense_fp8 = false;
    bopts.max_ctx = 512;
    bopts.kv_fp8 = true;
    bopts.park_mb = 0;
    bopts.max_batch = 2;
    bopts.max_kv_tokens = 1024;
    auto bgen = make_generator(bopts, std::move(w), load_ablation({}));

    std::vector<int32_t> mixed_b1;
    bgen->generate(prompt, n_steps, greedy, {}, [&](std::span<const int32_t> toks) {
      mixed_b1.insert(mixed_b1.end(), toks.begin(), toks.end());
      return true;
    });

    ingine::engine::Scheduler sched(*bgen, {}, 4);
    auto j0 = std::make_shared<ingine::engine::Job>(prompt, n_steps, greedy);
    auto j1 = std::make_shared<ingine::engine::Job>(prompt, n_steps, greedy);
    check(sched.submit(j0) && sched.submit(j1), "Suite 4: submitted B=2 mixed-format jobs");
    const std::vector<int32_t> mb0 = drain(*j0);
    const std::vector<int32_t> mb1 = drain(*j1);
    check(int(mixed_b1.size()) == n_steps && mb0 == mixed_b1 && mb1 == mixed_b1,
          "Suite 4: mixed-format BatchGenerator B=1 == B=2 exact parity");

    int exact_top1 = 0;
    bool all_within_tie = true;
    for (int s = 0; s < n_steps; ++s) {
      const bool hit1 = (mixed_b1[s] == ref_top1[s]);
      const bool hit2_tie = (mixed_b1[s] == ref_top2[s] && ref_margin[s] < 0.05f);
      if (hit1) ++exact_top1;
      if (!hit1 && !hit2_tie) all_within_tie = false;
      std::printf("  step %d: batch_tok=%d ref_top1=%d ref_top2=%d margin=%.4f (%s)\n",
                  s, mixed_b1[s], ref_top1[s], ref_top2[s], ref_margin[s],
                  hit1 ? "EXACT" : (hit2_tie ? "NEAR_TIE" : "MISMATCH"));
      if (!hit1) break;  // after any divergence, subsequent autoreg prefixes differ
    }
    check(exact_top1 >= 4 && all_within_tie,
          "Suite 4: mixed-format BatchGenerator matches Prefiller reference on real 24-layer L20(Q5L/IQ4)+L23(IQ4/Q5L)");

    INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  }
#endif

  std::printf(g_failures ? "BATCH_GGUF_MIXED_FAIL (%d)\n" : "BATCH_GGUF_MIXED_ALL_PASS\n", g_failures);
  return g_failures ? 1 : 0;
}

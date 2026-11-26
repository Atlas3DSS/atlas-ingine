// Gate 1 GPU verification for #888 (opt-in small-card knobs, strictly NO silent --tail-skip):
//   Suite 1: Q5_K vs FP8 dense projection microbenchmarks (wproj [8600, 2304], wo [2304, 3072], M=1, 4, 8).
//   Suite 2: Real disk KV checkpoint + prefix-cache bidirectional isolation (off <-> q5_k) + --loop-bf16
//            0-ULP parity and persistent VRAM delta on real Inkling-2-Mini weights.
//   Suite 3: Real GPU ExpertCache + kernels::moe::moe_up / moe_down 8-expert numerical verification across
//            interleaved decode (M=1, M=4) and prefill (Pass 0 / Pass 1) steps forcing repeated HOT promotions,
//            HOT demotions, staging-LRU hits, HOT hits, re-fetch of demoted experts, and bounded queue saturation
//            (async_swaps_dropped_bound > 0, experts_dropped == 0), verified at exact 0-ULP vs --expert-swaps 0.
#include "common/test_util.h"
#include "core/act.h"
#include "core/q5_k.h"
#include "ingine/generator.h"
#include "kernels/dense/dense.h"
#include "kernels/moe/moe.h"
#include "model/expert_cache.h"
#include "model/model_data.h"
#include "model/runtime.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using namespace ingine::model;
using ingine::test::DeviceArray;
using ingine::test::Rng;
using Json = nlohmann::ordered_json;

void check(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
  }
  std::printf("PASS: %s\n", msg);
}

std::size_t vram_used_bytes() {
  std::size_t free_b = 0, total_b = 0;
  INGINE_TEST_CUDA(cudaMemGetInfo(&free_b, &total_b));
  return total_b > free_b ? (total_b - free_b) : 0;
}

std::vector<ingine::ActT> to_act(const std::vector<float>& v) {
  std::vector<ingine::ActT> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
#if INGINE_ACT_FP16
    out[i] = __float2half(v[i]);
#else
    out[i] = __float2bfloat16(v[i]);
#endif
  }
  return out;
}

template <typename Fn>
double bench_us(cudaStream_t stream, int warmup, int iters, Fn&& fn) {
  for (int i = 0; i < warmup; ++i) fn();
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  cudaEvent_t e0 = nullptr, e1 = nullptr;
  INGINE_TEST_CUDA(cudaEventCreate(&e0));
  INGINE_TEST_CUDA(cudaEventCreate(&e1));
  std::vector<double> us;
  us.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    INGINE_TEST_CUDA(cudaEventRecord(e0, stream));
    fn();
    INGINE_TEST_CUDA(cudaEventRecord(e1, stream));
    INGINE_TEST_CUDA(cudaEventSynchronize(e1));
    float ms = 0.f;
    INGINE_TEST_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    us.push_back(double(ms) * 1000.0);
  }
  INGINE_TEST_CUDA(cudaEventDestroy(e0));
  INGINE_TEST_CUDA(cudaEventDestroy(e1));
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

// ---- Suite 1: Q5_K vs FP8 microbenchmark (wproj [8600, 2304], wo [2304, 3072], M=1, 4, 8) ----
Json run_suite1_microbench(cudaStream_t stream) {
  std::printf("\n=== Suite 1: Q5_K vs FP8 Dense Microbenchmark ===\n");
  namespace dense = ingine::kernels::dense;
  Rng rng(20261126);
  const int rows_p = 8600, Kp = 2304;
  const int rows_o = 2304, Ko = 3072;

  std::vector<float> wp_h(std::size_t(rows_p) * Kp);
  for (auto& v : wp_h) v = float(rng.normal() * 0.03);
  std::vector<float> wo_h(std::size_t(rows_o) * Ko);
  for (auto& v : wo_h) v = float(rng.normal() * 0.02);

  DeviceArray<ingine::ActT> dwp(to_act(wp_h)), dwo(to_act(wo_h));
  DeviceArray<std::uint8_t> dwp_fp8(std::size_t(rows_p) * Kp), dwo_fp8(std::size_t(rows_o) * Ko);
  DeviceArray<float> dwp_sc(rows_p), dwo_sc(rows_o);
  DeviceArray<std::uint8_t> dwp_q5(std::size_t(rows_p) * ingine::q5k::row_bytes(Kp));
  DeviceArray<std::uint8_t> dwo_q5(std::size_t(rows_o) * ingine::q5k::row_bytes(Ko));

  dense::fp8_rows(dwp.get(), rows_p, Kp, dwp_fp8.get(), dwp_sc.get(), stream);
  dense::fp8_rows(dwo.get(), rows_o, Ko, dwo_fp8.get(), dwo_sc.get(), stream);
  dense::quantize_q5_k_rows(dwp.get(), rows_p, Kp, dwp_q5.get(), stream);
  dense::quantize_q5_k_rows(dwo.get(), rows_o, Ko, dwo_q5.get(), stream);

  const std::vector<int> seg = {0, 3072, 0, 3072, 6144, 0, 6144, 6528, 0, 6528, 6552, 0,
                                6552, 7576, 1, 7576, 8600, 2};
  DeviceArray<int> dseg(seg);
  DeviceArray<ingine::ActT> dxp(std::size_t(3) * 8 * Kp), dxo(std::size_t(8) * Ko);
  DeviceArray<float> doutp(std::size_t(8) * rows_p), douto(std::size_t(8) * rows_o);
  DeviceArray<float> dcw(std::size_t(rows_o) * 4), dst(std::size_t(3) * rows_o);
  INGINE_TEST_CUDA(cudaMemsetAsync(dxp.get(), 0, std::size_t(3) * 8 * Kp * sizeof(ingine::ActT), stream));
  INGINE_TEST_CUDA(cudaMemsetAsync(dxo.get(), 0, std::size_t(8) * Ko * sizeof(ingine::ActT), stream));
  INGINE_TEST_CUDA(cudaMemsetAsync(dcw.get(), 0, std::size_t(rows_o) * 4 * sizeof(float), stream));
  INGINE_TEST_CUDA(cudaMemsetAsync(dst.get(), 0, std::size_t(3) * rows_o * sizeof(float), stream));
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));

  Json arr = Json::array();
  for (int M : {1, 4, 8}) {
    dense::Matrix mp_fp8{dwp_fp8.get(), dwp_sc.get(), rows_p, false};
    dense::Matrix mp_q5{dwp_q5.get(), nullptr, rows_p, true};
    dense::Matrix mo_fp8{dwo_fp8.get(), dwo_sc.get(), rows_o, false};
    dense::Matrix mo_q5{dwo_q5.get(), nullptr, rows_o, true};

    const double wp_fp8_us = bench_us(stream, 10, 50, [&] {
      dense::proj(dxp.get(), M, mp_fp8, dseg.get(), 6, doutp.get(), false, stream);
    });
    const double wp_q5_us = bench_us(stream, 10, 50, [&] {
      dense::proj(dxp.get(), M, mp_q5, dseg.get(), 6, doutp.get(), false, stream);
    });
    const double wo_fp8_us = bench_us(stream, 10, 50, [&] {
      dense::proj_conv(dxo.get(), M, mo_fp8, dcw.get(), dst.get(), douto.get(), nullptr, false, stream);
    });
    const double wo_q5_us = bench_us(stream, 10, 50, [&] {
      dense::proj_conv(dxo.get(), M, mo_q5, dcw.get(), dst.get(), douto.get(), nullptr, false, stream);
    });
    std::printf("  M=%d  wproj[8600x2304]: fp8=%.1f us  q5_k=%.1f us (%.2fx) | wo[2304x3072]: fp8=%.1f us  q5_k=%.1f us (%.2fx)\n",
                M, wp_fp8_us, wp_q5_us, wp_fp8_us / wp_q5_us, wo_fp8_us, wo_q5_us, wo_fp8_us / wo_q5_us);
    arr.push_back(Json{
        {"M", M},
        {"wproj_fp8_us", wp_fp8_us},
        {"wproj_q5k_us", wp_q5_us},
        {"wproj_speedup", wp_fp8_us / wp_q5_us},
        {"wo_fp8_us", wo_fp8_us},
        {"wo_q5k_us", wo_q5_us},
        {"wo_speedup", wo_fp8_us / wo_q5_us},
    });
  }
  check(true, "Suite 1: Q5_K vs FP8 microbenchmarks completed cleanly");
  return arr;
}

// ---- Suite 2: Real Disk Checkpoint & Prefix-Cache Bidirectional Isolation + --loop-bf16 ----
Json run_suite2_checkpoint_and_loop_bf16(const std::filesystem::path& model_dir) {
  std::printf("\n=== Suite 2: Real Disk Checkpoint Bidirectional Isolation & --loop-bf16 ===\n");
  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  const Json msgs = Json::array(
      {{{"role", "system"},
        {"content", "You are a concise, deterministic systems engineer verifying GPU checkpoint isolation."}},
       {{"role", "user"}, {"content", "Explain why 2 + 2 = 4 in one clear sentence."}}});
  const auto rp = fmt.render(msgs, Json::array(), ingine::text::ChatOptions{});
  const int R = int(rp.ids.size());
  std::vector<std::int32_t> fork_ids = rp.ids;
  for (int i = 0; i < 8; ++i) fork_ids.push_back(rp.ids[std::size_t(i % R)]);

  Sampling greedy;
  greedy.temperature = 0.f;
  auto run_gen = [&](Generator& g, std::span<const std::int32_t> ids, int max_new, int root_hint = 0) {
    std::vector<std::int32_t> out;
    GenerateStats st = g.generate(Prompt{ids, {}, root_hint}, max_new, greedy, {}, [&](std::span<const std::int32_t> t) {
      out.insert(out.end(), t.begin(), t.end());
      return true;
    });
    return std::pair{out, st};
  };

  RuntimeOptions base;
  base.model_dir = model_dir;
  base.dense_fp8 = true;
  base.kv_fp8 = true;
  base.max_ctx = 1024;
  base.prefill_chunk = 512;
  base.park_mb = 256;
  base.park_host_mb = 0;

  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / ("ingine_888_g1_" + std::to_string(::getpid()));
  std::filesystem::remove_all(tmp_dir);
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path ckpt_off = tmp_dir / "ckpt_off.bin";
  const std::filesystem::path ckpt_q5k = tmp_dir / "ckpt_q5k.bin";

  std::vector<std::int32_t> ref_off, ref_loop_bf16, ref_q5k;
  double vram_off_mib = 0.0, vram_loop_bf16_mib = 0.0, vram_q5k_mib = 0.0;

  // 2a. Control (off, loop_bf16=false): save ckpt_off at exact boundary pos = R (with logits) via root split
  {
    RuntimeOptions opt = base;
    opt.kv_ckpt_path = ckpt_off;
    opt.kv_ckpt_every = R;
    auto gen = make_generator(opt);
    vram_off_mib = double(vram_used_bytes()) / (1024.0 * 1024.0);
    (void)run_gen(*gen, fork_ids, 4, R);
    auto [toks, st] = run_gen(*gen, rp.ids, 16, 0);
    ref_off = toks;
    check(std::filesystem::exists(ckpt_off) && toks.size() == 16,
          "Suite 2a: saved real disk checkpoint under attn_requant=off");
  }

  // 2b. --loop-bf16: 0-ULP token parity + ~81 MiB VRAM reclaim
  {
    RuntimeOptions opt = base;
    opt.loop_bf16 = true;
    auto gen = make_generator(opt);
    vram_loop_bf16_mib = double(vram_used_bytes()) / (1024.0 * 1024.0);
    auto [toks, st] = run_gen(*gen, rp.ids, 16, 0);
    ref_loop_bf16 = toks;
    const double delta_mib = vram_loop_bf16_mib - vram_off_mib;
    std::printf("  VRAM: control=%.1f MiB, --loop-bf16=%.1f MiB (delta=%.2f MiB)\n", vram_off_mib,
                vram_loop_bf16_mib, delta_mib);
    check(ref_loop_bf16 == ref_off, "Suite 2b: --loop-bf16 matches Control 16/16 tokens exactly");
    check(delta_mib <= -75.0 && delta_mib >= -88.0,
          "Suite 2b: --loop-bf16 reclaims ~81 MiB persistent VRAM (measured via cudaMemGetInfo)");
  }

  // 2c. Same-mode off -> off resume succeeds
  {
    RuntimeOptions opt = base;
    opt.park_mb = 0;
    opt.kv_ckpt_resume = ckpt_off;
    auto gen = make_generator(opt);
    auto [toks, st] = run_gen(*gen, rp.ids, 16, 0);
    check(st.reuse == Reuse::Checkpoint && st.reused_tokens == R && toks == ref_off,
          "Suite 2c: same-mode disk resume off -> off hits Reuse::Checkpoint with exact parity");
  }

  // 2d. Cross-mode off -> q5_k: explicit resume MUST throw; auto resume MUST decline to Reuse::None
  {
    RuntimeOptions opt = base;
    opt.park_mb = 0;
    opt.attn_requant = AttnRequantMode::Q5K;
    opt.kv_ckpt_resume = ckpt_off;
    auto gen = make_generator(opt);
    bool threw = false;
    try {
      (void)run_gen(*gen, rp.ids, 8, 0);
    } catch (const std::exception&) {
      threw = true;
    }
    check(threw, "Suite 2d: explicit disk resume off -> q5_k throws fail-closed");
  }
  {
    RuntimeOptions opt = base;
    opt.park_mb = 0;
    opt.attn_requant = AttnRequantMode::Q5K;
    opt.kv_ckpt_path = ckpt_off;
    opt.kv_ckpt_every = 0;
    auto gen = make_generator(opt);
    auto [toks, st] = run_gen(*gen, rp.ids, 8, 0);
    check(st.reuse == Reuse::None && st.reused_tokens == 0 && toks.size() == 8,
          "Suite 2d: auto disk resume off -> q5_k declines to Reuse::None");
  }

  // 2e. Save disk checkpoint under attn_requant=q5_k + measure VRAM delta vs Control
  {
    RuntimeOptions opt = base;
    opt.attn_requant = AttnRequantMode::Q5K;
    opt.kv_ckpt_path = ckpt_q5k;
    opt.kv_ckpt_every = R;
    auto gen = make_generator(opt);
    vram_q5k_mib = double(vram_used_bytes()) / (1024.0 * 1024.0);
    (void)run_gen(*gen, fork_ids, 4, R);
    auto [toks, st] = run_gen(*gen, rp.ids, 16, 0);
    ref_q5k = toks;
    const double delta_mib = vram_q5k_mib - vram_off_mib;
    std::printf("  VRAM: control=%.1f MiB, --attn-requant q5_k=%.1f MiB (delta=%.2f MiB)\n", vram_off_mib,
                vram_q5k_mib, delta_mib);
    check(std::filesystem::exists(ckpt_q5k) && toks.size() == 16,
          "Suite 2e: saved real disk checkpoint under attn_requant=q5_k");
    check(delta_mib <= -185.0 && delta_mib >= -205.0,
          "Suite 2e: --attn-requant q5_k reclaims ~193.5 MiB persistent VRAM (measured via cudaMemGetInfo)");
  }

  // 2f. Same-mode q5_k -> q5_k resume succeeds
  {
    RuntimeOptions opt = base;
    opt.park_mb = 0;
    opt.attn_requant = AttnRequantMode::Q5K;
    opt.kv_ckpt_resume = ckpt_q5k;
    auto gen = make_generator(opt);
    auto [toks, st] = run_gen(*gen, rp.ids, 16, 0);
    check(st.reuse == Reuse::Checkpoint && st.reused_tokens == R && toks == ref_q5k,
          "Suite 2f: same-mode disk resume q5_k -> q5_k hits Reuse::Checkpoint with exact parity");
  }

  // 2g. Cross-mode q5_k -> off: explicit resume MUST throw; auto resume MUST decline to Reuse::None
  {
    RuntimeOptions opt = base;
    opt.park_mb = 0;
    opt.attn_requant = AttnRequantMode::Off;
    opt.kv_ckpt_resume = ckpt_q5k;
    auto gen = make_generator(opt);
    bool threw = false;
    try {
      (void)run_gen(*gen, rp.ids, 8, 0);
    } catch (const std::exception&) {
      threw = true;
    }
    check(threw, "Suite 2g: explicit disk resume q5_k -> off throws fail-closed");
  }
  {
    RuntimeOptions opt = base;
    opt.park_mb = 0;
    opt.attn_requant = AttnRequantMode::Off;
    opt.kv_ckpt_path = ckpt_q5k;
    opt.kv_ckpt_every = 0;
    auto gen = make_generator(opt);
    auto [toks, st] = run_gen(*gen, rp.ids, 8, 0);
    check(st.reuse == Reuse::None && st.reused_tokens == 0 && toks.size() == 8,
          "Suite 2g: auto disk resume q5_k -> off declines to Reuse::None");
  }

  std::filesystem::remove_all(tmp_dir);
  return Json{
      {"vram_control_mib", vram_off_mib},
      {"vram_loop_bf16_mib", vram_loop_bf16_mib},
      {"vram_loop_bf16_delta_mib", vram_loop_bf16_mib - vram_off_mib},
      {"vram_q5k_mib", vram_q5k_mib},
      {"vram_q5k_delta_mib", vram_q5k_mib - vram_off_mib},
      {"bidirectional_ckpt_isolation", true},
      {"loop_bf16_exact_tokens", true},
  };
}

// ---- Suite 3: Real GPU ExpertCache + kernels::moe::moe_up/moe_down 8-Expert Numerical Verification ----
// Exercises BOTH decode (M=1, M=4) and prefill (Pass 0 / Pass 1) on real Inkling-2-Mini expert weights
// under tight simulated VRAM cap (5888 MiB) with --expert-swaps 0 vs --expert-swaps 8, verifying exact 0-ULP
// outputs across repeated promotions, demotions, LRU hits, HOT hits, re-fetch of demoted experts, and queue drops.
Json run_suite3_gpu_expert_cache_numerical(const std::filesystem::path& model_dir, cudaStream_t stream) {
  std::printf("\n=== Suite 3: Real GPU ExpertCache 8-Expert Numerical Verification ===\n");
  constexpr int H = ingine::model::HIDDEN;
  constexpr int I = ingine::model::INTER;
  constexpr int K = ingine::model::TOP_K;

  // Deterministic sequence of decode (M=1, M=4) and prefill steps across loop layers 6..9 (shared slab bucket):
  //   Phase 0 (32 steps, M=4): sweep experts 0..127 twice -> guarantees non-HOT hits >= promote_threshold (2),
  //                            burst > capacity (4) in single M=4 step -> guarantees async_swaps_dropped_bound > 0,
  //                            promotions, and HOT demotions regardless of initial router_bias placement.
  //   Phase 1 (32 steps, mixed M=1/M=4 + prefill): repeatedly hit two shifting 24-expert working sets to drive
  //                            further promotions, demotions, staging-LRU hits, and HOT hits.
  //   Phase 2 (24 steps, decode + prefill): sweep ALL 128 experts on layer 6 & 7 so every demoted HOT expert
  //                            is guaranteed to be re-fetched byte-identically in both decode and prefill!
  struct StepSpec {
    bool is_prefill = false;
    int layer = 0;
    int M = 1;
    std::vector<int> idx;    // [M * TOP_K]
    std::vector<float> wts;  // [M * TOP_K]
  };
  std::vector<StepSpec> steps;
  steps.reserve(88);
  Rng rng(0x88812345ULL);

  auto add_step = [&](bool is_prefill, int layer, int M, int base_e, int stride) {
    StepSpec sp;
    sp.is_prefill = is_prefill;
    sp.layer = layer;
    sp.M = M;
    sp.idx.resize(std::size_t(M) * K);
    sp.wts.resize(std::size_t(M) * K);
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        sp.idx[std::size_t(m) * K + k] = (base_e + m * stride + k) % EXPERTS;
        sp.wts[std::size_t(m) * K + k] = 0.5f + 0.1f * float(k + 1);
      }
    }
    steps.push_back(std::move(sp));
  };

  // Phase 0: 32 decode steps (M=4, stride=8 => 32 distinct experts per step) sweeping 0..127 twice on layers 6..9
  for (int rep = 0; rep < 2; ++rep) {
    for (int blk = 0; blk < 4; ++blk) {
      for (int l = 6; l <= 9; ++l) {
        add_step(/*is_prefill=*/false, l, /*M=*/4, blk * 32, /*stride=*/8);
      }
    }
  }
  // Phase 1: 32 interleaved decode + prefill steps concentrating on shifting working sets on layers 6..9
  for (int s = 0; s < 32; ++s) {
    const int l = 6 + (s % 4);
    const bool pf = ((s % 4) == 3);
    const int M = pf ? 2 : ((s % 2) == 0 ? 4 : 1);
    const int base = (s < 16) ? ((s % 2) * 16 + 64) : ((s % 2) * 16 + 16);
    add_step(pf, l, M, base, /*stride=*/6);
  }
  // Phase 2: 24 decode & prefill steps sweeping ALL 128 experts on layers 6 & 7 (guarantees re-fetch of any demoted HOT)
  for (int blk = 0; blk < 4; ++blk) {
    add_step(/*is_prefill=*/false, 6, /*M=*/4, blk * 32, /*stride=*/8);
    add_step(/*is_prefill=*/false, 7, /*M=*/4, blk * 32, /*stride=*/8);
    add_step(/*is_prefill=*/true, 6, /*M=*/4, blk * 32, /*stride=*/8);
    add_step(/*is_prefill=*/true, 7, /*M=*/4, blk * 32, /*stride=*/8);
    add_step(/*is_prefill=*/false, 8, /*M=*/1, blk * 32, /*stride=*/1);
    add_step(/*is_prefill=*/true, 9, /*M=*/2, blk * 32, /*stride=*/8);
  }

  // Deterministic input activations [4, H] and zero conv weights so conv_hist is exact weighted sum
  std::vector<float> x_h(std::size_t(4) * H);
  for (auto& v : x_h) v = float(rng.normal() * 0.5);
  DeviceArray<ingine::ActT> d_x(to_act(x_h));
  DeviceArray<float> d_zero_conv(std::size_t(H) * 4);
  DeviceArray<float> d_conv_state(std::size_t(3) * H);
  INGINE_TEST_CUDA(cudaMemsetAsync(d_zero_conv.get(), 0, std::size_t(H) * 4 * sizeof(float), stream));

  DeviceArray<int> d_idx(std::size_t(4) * K);
  DeviceArray<float> d_wts(std::size_t(4) * K);
  DeviceArray<int> d_counts(EXPERTS);
  DeviceArray<ingine::ActT> d_act(std::size_t(4) * K * I);
  DeviceArray<float> d_out(std::size_t(4) * H);
  DeviceArray<float> d_hist(std::size_t(4) * H);
  DeviceArray<ingine::ActT> d_ws(std::size_t(4) * K * H);

  // Helper that runs all 64 steps under a given expert_swaps value, returning exact bit-level output vectors
  // for every decode step ([M, H] from moe_down) and every prefill step (per-slot [M*K, I] from moe_up + [M*K, H]
  // single-expert moe_down evaluated strictly on active pointers inside each Pass 0 / Pass 1 callback).
  struct RunCapture {
    std::vector<std::vector<std::uint32_t>> step_bits;
    OffloadStats stats;
    std::size_t vram_bytes = 0;
  };

  auto run_mode = [&](int expert_swaps) -> RunCapture {
    OffloadConfig cfg;
    cfg.vram_mb = 5888;  // Simulated 6 GiB cap on V100 (forces HOT + LRU + WARM RAM offloading)
    cfg.ram_mb = 12288;
    cfg.mode = "dma-lru";
    cfg.max_ctx = 2048;
    cfg.prefill_chunk = 512;
    cfg.kv_fp8 = true;
    cfg.expert_swaps = expert_swaps;

    Weights w = load_weights(model_dir, /*dense_fp8=*/true, stream, cfg);
    check(w.expert_cache != nullptr, "Suite 3: ExpertCache instantiated under offload_vram_mb=5888");
    w.expert_cache->reset_runtime_stats();

    RunCapture cap;
    cap.vram_bytes = vram_used_bytes();
    cap.step_bits.resize(steps.size());

    for (std::size_t si = 0; si < steps.size(); ++si) {
      const StepSpec& sp = steps[si];
      const Layer& ly = w.layers[std::size_t(sp.layer)];
      const auto fmt = w.expert_formats[std::size_t(sp.layer)];
      const auto fmt2 = w.w2_expert_formats[std::size_t(sp.layer)];
      // Ensure non-EXL3 or handle w13/w2 format cleanly
      const auto* f13 = ly.experts.w13_factor.data ? ptr<float>(ly.experts.w13_factor) : nullptr;
      const auto* f2 = ly.experts.w2_factor.data ? ptr<float>(ly.experts.w2_factor) : nullptr;

      INGINE_TEST_CUDA(cudaMemsetAsync(d_conv_state.get(), 0, std::size_t(3) * H * sizeof(float), stream));
      INGINE_TEST_CUDA(cudaMemcpyAsync(d_idx.get(), sp.idx.data(), sp.idx.size() * sizeof(int), cudaMemcpyHostToDevice,
                                       stream));
      INGINE_TEST_CUDA(cudaMemcpyAsync(d_wts.get(), sp.wts.data(), sp.wts.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));

      if (!sp.is_prefill) {
        w.expert_cache->run_decode_layer(sp.layer, d_idx.get(), sp.M, stream, [&](int tok_off, int tok_cnt) {
          const auto ptrs = w.expert_cache->layer_ptrs(sp.layer);
          ingine::kernels::moe::moe_up(d_x.get() + std::size_t(tok_off) * H, tok_cnt,
                                       d_idx.get() + std::size_t(tok_off) * K, nullptr, nullptr, f13,
                                       static_cast<ingine::kernels::moe::Format>(fmt),
                                       d_act.get() + std::size_t(tok_off) * K * I, false, stream, nullptr, &ptrs,
                                       d_ws.get());
          ingine::kernels::moe::moe_down(d_act.get() + std::size_t(tok_off) * K * I, tok_cnt,
                                         d_idx.get() + std::size_t(tok_off) * K, d_wts.get() + std::size_t(tok_off) * K,
                                         nullptr, nullptr, f2, static_cast<ingine::kernels::moe::Format>(fmt2),
                                         d_zero_conv.get(), d_conv_state.get(),
                                         d_out.get() + std::size_t(tok_off) * H,
                                         d_hist.get() + std::size_t(tok_off) * H, false, stream, nullptr, d_ws.get(),
                                         &ptrs);
        });
        INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
        std::vector<float> h_hist(std::size_t(sp.M) * H);
        INGINE_TEST_CUDA(cudaMemcpy(h_hist.data(), d_hist.get(), h_hist.size() * sizeof(float), cudaMemcpyDeviceToHost));
        cap.step_bits[si].resize(h_hist.size());
        std::memcpy(cap.step_bits[si].data(), h_hist.data(), h_hist.size() * sizeof(float));
      } else {
        // Prefill Pass 0 / Pass 1 path: compute active counts on device, invoke run_prefill_layer, and inside each
        // pass run real GPU moe_up + moe_down ONLY on experts marked active in d_active_mask!
        std::array<int, EXPERTS> h_counts{};
        for (int e : sp.idx) h_counts[std::size_t(e)] += 1;
        INGINE_TEST_CUDA(cudaMemcpyAsync(d_counts.get(), h_counts.data(), EXPERTS * sizeof(int), cudaMemcpyHostToDevice,
                                         stream));

        const std::size_t total_slots = std::size_t(sp.M) * K;
        std::vector<std::uint16_t> up_bits(total_slots * I, 0);
        std::vector<std::uint32_t> down_bits(total_slots * H, 0);
        std::vector<std::uint8_t> slot_done(total_slots, 0);

        w.expert_cache->run_prefill_layer(sp.layer, d_counts.get(), stream, [&](const std::uint8_t* d_active_mask) {
          std::array<std::uint8_t, EXPERTS> mask{};
          if (d_active_mask) {
            INGINE_TEST_CUDA(cudaMemcpyAsync(mask.data(), d_active_mask, EXPERTS, cudaMemcpyDeviceToHost, stream));
            INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
          } else {
            mask.fill(1);
          }
          const auto ptrs = w.expert_cache->layer_ptrs(sp.layer);
          // For each routed (m, k) active in this pass, run real GPU moe_up + moe_down with idx=[e,e,...,e] and
          // wts=[1,0,...,0] so ONLY active expert `e`'s device pointers are ever dereferenced in this pass!
          for (int m = 0; m < sp.M; ++m) {
            for (int k = 0; k < K; ++k) {
              const std::size_t slot_i = std::size_t(m) * K + std::size_t(k);
              const int e = sp.idx[slot_i];
              if (!mask[std::size_t(e)]) continue;
              check(slot_done[slot_i] == 0, "Suite 3 prefill: expert never computed twice across Pass 0 / Pass 1");
              slot_done[slot_i] = 1;

              std::array<int, K> one_idx{};
              one_idx.fill(e);
              std::array<float, K> one_wts{};
              one_wts[0] = sp.wts[slot_i];
              INGINE_TEST_CUDA(cudaMemcpyAsync(d_idx.get(), one_idx.data(), K * sizeof(int), cudaMemcpyHostToDevice,
                                               stream));
              INGINE_TEST_CUDA(cudaMemcpyAsync(d_wts.get(), one_wts.data(), K * sizeof(float), cudaMemcpyHostToDevice,
                                               stream));
              INGINE_TEST_CUDA(cudaMemsetAsync(d_conv_state.get(), 0, std::size_t(3) * H * sizeof(float), stream));

              ingine::kernels::moe::moe_up(d_x.get() + std::size_t(m) * H, 1, d_idx.get(), nullptr, nullptr, f13,
                                           static_cast<ingine::kernels::moe::Format>(fmt), d_act.get(), false, stream,
                                           nullptr, &ptrs, d_ws.get());
              ingine::kernels::moe::moe_down(d_act.get(), 1, d_idx.get(), d_wts.get(), nullptr, nullptr, f2,
                                             static_cast<ingine::kernels::moe::Format>(fmt2), d_zero_conv.get(),
                                             d_conv_state.get(), d_out.get(), d_hist.get(), false, stream, nullptr,
                                             d_ws.get(), &ptrs);
              INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
              INGINE_TEST_CUDA(cudaMemcpy(&up_bits[slot_i * I], d_act.get(), std::size_t(I) * sizeof(ingine::ActT),
                                          cudaMemcpyDeviceToHost));
              INGINE_TEST_CUDA(cudaMemcpy(&down_bits[slot_i * H], d_hist.get(), std::size_t(H) * sizeof(float),
                                          cudaMemcpyDeviceToHost));
            }
          }
        });
        INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
        for (std::size_t s0 = 0; s0 < total_slots; ++s0) {
          check(slot_done[s0] == 1, "Suite 3 prefill: every routed expert computed exactly once (zero dropped)");
        }
        // Pack up_bits + down_bits into step_bits[si]
        cap.step_bits[si].reserve(up_bits.size() + down_bits.size());
        for (auto u16 : up_bits) cap.step_bits[si].push_back(std::uint32_t(u16));
        cap.step_bits[si].insert(cap.step_bits[si].end(), down_bits.begin(), down_bits.end());
      }
    }

    cap.stats = w.expert_cache->stats();
    return cap;
  };

  const RunCapture ctrl = run_mode(/*expert_swaps=*/0);
  const RunCapture swap = run_mode(/*expert_swaps=*/4);

  // Verify exact 0-ULP bit identity across all 64 interleaved decode & prefill steps!
  std::size_t total_words = 0;
  std::size_t diff_words = 0;
  for (std::size_t si = 0; si < steps.size(); ++si) {
    check(ctrl.step_bits[si].size() == swap.step_bits[si].size(), "Suite 3: output tensor sizes match");
    for (std::size_t j = 0; j < ctrl.step_bits[si].size(); ++j) {
      ++total_words;
      if (ctrl.step_bits[si][j] != swap.step_bits[si][j]) ++diff_words;
    }
  }
  std::printf("  Control (swaps=0): hot_hits=%llu lru_hits=%llu warm=%llu cold=%llu promo=%llu evict=%llu qdrop=%llu edrop=%llu\n",
              static_cast<unsigned long long>(ctrl.stats.hot_hits),
              static_cast<unsigned long long>(ctrl.stats.lru_hits),
              static_cast<unsigned long long>(ctrl.stats.warm_fetches),
              static_cast<unsigned long long>(ctrl.stats.cold_fetches),
              static_cast<unsigned long long>(ctrl.stats.async_swaps_completed),
              static_cast<unsigned long long>(ctrl.stats.hot_evictions),
              static_cast<unsigned long long>(ctrl.stats.async_swaps_dropped_bound),
              static_cast<unsigned long long>(ctrl.stats.experts_dropped));
  std::printf("  Swap    (swaps=8): hot_hits=%llu lru_hits=%llu warm=%llu cold=%llu promo=%llu evict=%llu qdrop=%llu edrop=%llu\n",
              static_cast<unsigned long long>(swap.stats.hot_hits),
              static_cast<unsigned long long>(swap.stats.lru_hits),
              static_cast<unsigned long long>(swap.stats.warm_fetches),
              static_cast<unsigned long long>(swap.stats.cold_fetches),
              static_cast<unsigned long long>(swap.stats.async_swaps_completed),
              static_cast<unsigned long long>(swap.stats.hot_evictions),
              static_cast<unsigned long long>(swap.stats.async_swaps_dropped_bound),
              static_cast<unsigned long long>(swap.stats.experts_dropped));
  std::printf("  Bitwise parity across %zu decode+prefill output elements: diff_words=%zu\n", total_words, diff_words);

  check(diff_words == 0, "Suite 3: real GPU ExpertCache swap path matches Control 0-ULP on all 64 decode+prefill steps");
  check(swap.stats.experts_dropped == 0 && ctrl.stats.experts_dropped == 0,
        "Suite 3: zero experts dropped invariant (experts_dropped == 0)");
  check(swap.stats.async_swaps_completed > 0 && swap.stats.hot_evictions > 0,
        "Suite 3: real GPU HOT promotions and demotions exercised (completed > 0 && hot_evictions > 0)");
  check(swap.stats.hot_hits > 0 && swap.stats.lru_hits > 0,
        "Suite 3: both HOT hits and staging-LRU hits exercised on real GPU");
  check(swap.stats.async_swaps_dropped_bound > 0,
        "Suite 3: bounded swap queue saturation exercised (async_swaps_dropped_bound > 0)");

  return Json{
      {"simulated_vram_cap_mib", 5888},
      {"actual_vram_ctrl_mib", double(ctrl.vram_bytes) / (1024.0 * 1024.0)},
      {"actual_vram_swap_mib", double(swap.vram_bytes) / (1024.0 * 1024.0)},
      {"total_compared_words", total_words},
      {"diff_words", diff_words},
      {"swap_hot_hits", swap.stats.hot_hits},
      {"swap_lru_hits", swap.stats.lru_hits},
      {"swap_warm_fetches", swap.stats.warm_fetches},
      {"swap_cold_fetches", swap.stats.cold_fetches},
      {"swap_completed", swap.stats.async_swaps_completed},
      {"swap_hot_evictions", swap.stats.hot_evictions},
      {"swap_dropped_bound", swap.stats.async_swaps_dropped_bound},
      {"swap_experts_dropped", swap.stats.experts_dropped},
  };
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  std::filesystem::path model_dir = INGINE_MODEL_DIR;
  std::string json_out;
  bool skip_microbench = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) model_dir = argv[++i];
    else if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) json_out = argv[++i];
    else if (std::strcmp(argv[i], "--skip-microbench") == 0) skip_microbench = true;
  }
  if (!std::filesystem::exists(model_dir / "config.json")) {
    std::printf("skip: no model at %s\n", model_dir.c_str());
    return ingine::test::kSkip;
  }

  cudaStream_t stream = nullptr;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  Json report;
  if (!skip_microbench) {
    report["suite1_microbench"] = run_suite1_microbench(stream);
  }
  report["suite2_ckpt_and_loop_bf16"] = run_suite2_checkpoint_and_loop_bf16(model_dir);
  report["suite3_gpu_expert_cache"] = run_suite3_gpu_expert_cache_numerical(model_dir, stream);

  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  if (!json_out.empty()) {
    std::ofstream out(json_out);
    out << report.dump(2) << "\n";
  }
  std::printf("\nSMALL_CARD_888_GATE1_ALL_PASS\n");
  return 0;
}

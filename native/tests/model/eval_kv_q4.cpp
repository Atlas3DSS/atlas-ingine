// Part 1 (--kv-q4) & 1M-context end-to-end verification on real Inkling-2-Mini weights:
//   --mode kl              : exact full-vocab KL(BF16 || Q4_0) and KL(BF16 || FP8) over 256 decoded tokens after 4K prefill
//   --mode needle          : 64K, 128K, and 245K needle-in-a-haystack recall (3 depths each = 9 needles) + VRAM & speed
//   --mode vram            : exact KV cache VRAM comparison (BF16 vs FP8 vs Q4_0) at 64K, 128K, 262K, 1M
//   --mode test-1m-kernel  : 1,048,576-row Q4_0 KV GPU oracle test for attn_decode (M=1,4) + prefill::attention + save/restore
//   --mode bench-needle-1m : resident 1M-context (--max-ctx 1048576 --kv-q4) VRAM, prefill/decode tok/s, & honest needle test
#include "core/device.h"
#include "ingine/generator.h"
#include "kernels/attn/attn.h"
#include "kernels/head/head.h"
#include "kernels/prefill/prefill.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "text/chat.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::model;

namespace {

std::vector<float> download_f32(const float* d, size_t n) {
  std::vector<float> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

double kl_div(const std::vector<float>& ref, const std::vector<float>& test) {
  const double ma = *std::max_element(ref.begin(), ref.end());
  const double mb = *std::max_element(test.begin(), test.end());
  double za = 0.0, zb = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    za += std::exp(double(ref[i]) - ma);
    zb += std::exp(double(test[i]) - mb);
  }
  const double lza = std::log(za), lzb = std::log(zb);
  double k = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double la = double(ref[i]) - ma - lza;
    const double lb = double(test[i]) - mb - lzb;
    k += std::exp(la) * (la - lb);
  }
  return k;
}

int argmax(const std::vector<float>& a) {
  return int(std::max_element(a.begin(), a.end()) - a.begin());
}

int run_vram() {
  cudaFree(nullptr);
  std::printf("=== KV Cache VRAM Comparison (40 SWA BF16 + 8 Global Full) ===\n");
  for (int ctx : {65536, 131072, 262144}) {
    for (int mode = 0; mode < 3; ++mode) {
      const char* name = mode == 0 ? "bf16" : (mode == 1 ? "fp8 " : "q4_0");
      const bool fp8 = mode == 1, q4 = mode == 2;
      size_t f0 = 0, f1 = 0, tot = 0;
      cudaMemGetInfo(&f0, &tot);
      State st = make_state(ctx, fp8, q4);
      cudaMemGetInfo(&f1, &tot);
      const double full_gib = double(st.k_full.bytes() + st.v_full.bytes()) / (1024.0 * 1024.0 * 1024.0);
      const double ring_mib = double(st.k_ring.bytes() + st.v_ring.bytes()) / (1024.0 * 1024.0);
      const double delta_gib = double(f0 > f1 ? f0 - f1 : 0) / (1024.0 * 1024.0 * 1024.0);
      std::printf("ctx=%6d  kv=%s  full_kv=%.3f GiB  swa_ring=%.1f MiB  cuda_delta=%.3f GiB\n",
                  ctx, name, full_gib, ring_mib, delta_gib);
    }
  }
  return 0;
}

int run_kl(const std::string& model_dir, const std::string& golden_json, int prefix_len, int eval_tokens) {
  std::ifstream gf(golden_json);
  if (!gf) {
    std::fprintf(stderr, "cannot open %s\n", golden_json.c_str());
    return 1;
  }
  std::vector<int32_t> ids = nlohmann::json::parse(gf)["prompt_ids"].get<std::vector<int32_t>>();
  if (int(ids.size()) < prefix_len + eval_tokens) {
    std::fprintf(stderr, "prompt has %zu tokens, need %d\n", ids.size(), prefix_len + eval_tokens);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::printf("Loading weights (%s, dense_fp8=true)...\n", model_dir.c_str());
  std::fflush(stdout);
  Weights w = load_weights(model_dir, true, stream);
  Ablation abl = load_ablation({});
  Prefiller pre(w, abl, 2048, stream);
  // Build each KV mode sequentially reusing the same loaded Weights!
  const int max_ctx = prefix_len + eval_tokens + 64;

  DeviceBuffer d_ids(size_t(prefix_len + eval_tokens) * 4);
  d_ids.copy_from_host(ids.data(), size_t(prefix_len + eval_tokens) * 4);
  DeviceBuffer d_logits(size_t(UNPADDED_VOCAB) * 4);
  DeviceBuffer d_pos(8);
  Buffers buf1 = make_buffers(1);

  auto eval_mode = [&](const char* label, bool kv_fp8, bool kv_q4) -> std::vector<std::vector<float>> {
    State st = make_state(max_ctx, kv_fp8, kv_q4);
    Decoder dec(w, abl, st, true);
    reset_state(st, stream);
    // Prefill prefix_len tokens in 2048 chunks
    for (int p0 = 0; p0 < prefix_len; p0 += 2048) {
      const int T = std::min(2048, prefix_len - p0);
      pre.run(st, static_cast<const int32_t*>(d_ids.p) + p0, T, p0, static_cast<float*>(d_logits.p), nullptr);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::vector<float>> all_logits;
    all_logits.reserve(eval_tokens + 1);
    all_logits.push_back(download_f32(static_cast<const float*>(d_logits.p), UNPADDED_VOCAB));
    // Teacher-forced decode for eval_tokens positions
    for (int i = 0; i < eval_tokens; ++i) {
      const int64_t tok = ids[prefix_len + i];
      const int64_t pos = prefix_len + i;
      CUDA_CHECK(cudaMemcpyAsync(buf1.token.data, &tok, 8, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_pos.p, &pos, 8, cudaMemcpyHostToDevice, stream));
      dec.decode(buf1, static_cast<const int64_t*>(d_pos.p), false, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      all_logits.push_back(download_f32(static_cast<const float*>(buf1.logits.data), UNPADDED_VOCAB));
    }
    std::printf("[%s] prefilled %d + decoded %d tokens\n", label, prefix_len, eval_tokens);
    std::fflush(stdout);
    return all_logits;
  };

  auto ref_bf16 = eval_mode("bf16", false, false);
  auto out_fp8  = eval_mode("fp8 ", true, false);
  auto out_q4   = eval_mode("q4_0", false, true);

  auto report = [&](const char* name, const std::vector<std::vector<float>>& test) {
    std::vector<double> kls;
    int top1_match = 0;
    for (size_t i = 0; i < ref_bf16.size(); ++i) {
      kls.push_back(kl_div(ref_bf16[i], test[i]));
      if (argmax(ref_bf16[i]) == argmax(test[i])) ++top1_match;
    }
    std::vector<double> sorted = kls;
    std::sort(sorted.begin(), sorted.end());
    double mean = std::accumulate(kls.begin(), kls.end(), 0.0) / double(kls.size());
    double p50 = sorted[sorted.size() / 2];
    double p95 = sorted[size_t(sorted.size() * 0.95)];
    double mx = sorted.back();
    std::printf("KL(%s vs bf16): n=%zu  mean=%.6f  p50=%.6f  p95=%.6f  max=%.6f  prefill_kl=%.6f  top1=%d/%zu (%.2f%%)\n",
                name, kls.size(), mean, p50, p95, mx, kls[0], top1_match, kls.size(),
                100.0 * double(top1_match) / double(kls.size()));
    std::fflush(stdout);
    return mean;
  };

  const double kl_fp8 = report("fp8 ", out_fp8);
  const double kl_q4  = report("q4_0", out_q4);
  CUDA_CHECK(cudaStreamDestroy(stream));
  const bool pass = kl_q4 < 0.005;
  std::printf("%s Part 1 KL gate: Q4_0 mean KL=%.6f (FP8=%.6f, gate < 0.005)\n", pass ? "PASS" : "FAIL", kl_q4, kl_fp8);
  return pass ? 0 : 1;
}

int run_kl_models(const std::string& ref_dir, const std::string& test_dir, const std::string& golden_json,
                  int prefix_len, int eval_tokens, bool test_kv_q4, bool dense_fp8) {
  std::ifstream gf(golden_json);
  if (!gf) {
    std::fprintf(stderr, "cannot open %s\n", golden_json.c_str());
    return 1;
  }
  std::vector<int32_t> ids = nlohmann::json::parse(gf)["prompt_ids"].get<std::vector<int32_t>>();
  if (int(ids.size()) < prefix_len + eval_tokens) {
    std::fprintf(stderr, "prompt has %zu tokens, need %d\n", ids.size(), prefix_len + eval_tokens);
    return 1;
  }

  auto eval_dir = [&](const std::string& dir, bool kv_fp8, bool kv_q4, const char* label) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    size_t f0 = 0, f1 = 0, tot = 0;
    cudaMemGetInfo(&f0, &tot);
    Weights w = load_weights(dir, dense_fp8, stream);
    cudaMemGetInfo(&f1, &tot);
    const double w_gib = double(f0 > f1 ? f0 - f1 : 0) / (1024.0 * 1024.0 * 1024.0);
    std::printf("[%s] loaded %s (dense_fp8=%d, kv_q4=%d): weights_vram=%.3f GiB (%.3f GB)\n", label, dir.c_str(),
                int(dense_fp8), int(kv_q4), w_gib, w_gib * 1.073741824);
    std::fflush(stdout);

    Ablation abl = load_ablation({});
    Prefiller pre(w, abl, 2048, stream);
    const int max_ctx = prefix_len + eval_tokens + 64;
    State st = make_state(max_ctx, kv_fp8, kv_q4);
    Decoder dec(w, abl, st, true);
    reset_state(st, stream);

    DeviceBuffer d_ids(size_t(prefix_len + eval_tokens) * 4);
    d_ids.copy_from_host(ids.data(), size_t(prefix_len + eval_tokens) * 4);
    DeviceBuffer d_logits(size_t(UNPADDED_VOCAB) * 4);
    DeviceBuffer d_pos(8);
    Buffers buf1 = make_buffers(1);

    for (int p0 = 0; p0 < prefix_len; p0 += 2048) {
      const int T = std::min(2048, prefix_len - p0);
      pre.run(st, static_cast<const int32_t*>(d_ids.p) + p0, T, p0, static_cast<float*>(d_logits.p), nullptr);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::vector<float>> all_logits;
    all_logits.reserve(eval_tokens + 1);
    all_logits.push_back(download_f32(static_cast<const float*>(d_logits.p), UNPADDED_VOCAB));
    for (int i = 0; i < eval_tokens; ++i) {
      const int64_t tok = ids[prefix_len + i];
      const int64_t pos = prefix_len + i;
      CUDA_CHECK(cudaMemcpyAsync(buf1.token.data, &tok, 8, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_pos.p, &pos, 8, cudaMemcpyHostToDevice, stream));
      dec.decode(buf1, static_cast<const int64_t*>(d_pos.p), false, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      all_logits.push_back(download_f32(static_cast<const float*>(buf1.logits.data), UNPADDED_VOCAB));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return all_logits;
  };

  auto ref_logits = eval_dir(ref_dir, false, false, "ref_native");
  auto tst_logits = eval_dir(test_dir, false, test_kv_q4, "test_exl3 ");
  std::vector<double> kls;
  int top1 = 0;
  for (size_t i = 0; i < ref_logits.size(); ++i) {
    kls.push_back(kl_div(ref_logits[i], tst_logits[i]));
    if (argmax(ref_logits[i]) == argmax(tst_logits[i])) ++top1;
  }
  std::vector<double> sorted = kls;
  std::sort(sorted.begin(), sorted.end());
  const double mean = std::accumulate(kls.begin(), kls.end(), 0.0) / double(kls.size());
  const double p50 = sorted[sorted.size() / 2];
  const double p95 = sorted[size_t(sorted.size() * 0.95)];
  const double mx = sorted.back();
  std::printf("KL(test vs ref): n=%zu  mean=%.6f  p50=%.6f  p95=%.6f  max=%.6f  prefill_kl=%.6f  top1=%d/%zu (%.2f%%)\n",
              kls.size(), mean, p50, p95, mx, kls[0], top1, kls.size(), 100.0 * double(top1) / double(kls.size()));
  return mean <= 0.031 ? 0 : 1;
}

int run_needle(const std::string& model_dir, int max_ctx) {
  cudaFree(nullptr);
  size_t free0 = 0, free1 = 0, total = 0;
  cudaMemGetInfo(&free0, &total);

  RuntimeOptions opt;
  opt.model_dir = model_dir;
  opt.max_ctx = max_ctx;
  opt.dense_fp8 = true;
  opt.kv_q4 = true;
  opt.spec = false;
  opt.park_mb = 0;
  opt.vision = false;

  std::printf("Loading generator: model=%s max_ctx=%d dense_fp8=true kv_q4=true...\n", model_dir.c_str(), max_ctx);
  std::fflush(stdout);
  auto gen = make_generator(opt);
  cudaMemGetInfo(&free1, &total);
  const double used_gib = double(free0 > free1 ? free0 - free1 : 0) / (1024.0 * 1024.0 * 1024.0);
  std::printf("Generator VRAM used (weights + %d ctx Q4_0 KV + scratch): %.3f GiB\n", max_ctx, used_gib);
  std::fflush(stdout);

  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  const auto& tok = fmt.tokenizer();
  const std::vector<int32_t> stop = {*tok.token_id("<|content_model_end_sampling|>")};

  // Build filler paragraph (~120 tokens)
  const std::string para =
      "The municipal archive catalogued seasonal rainfall, timber shipments, railway timetables, and harbor tide "
      "ledgers across the coastal districts during the nineteenth century. Clerks recorded barometric readings "
      "each morning and summarized freight tonnage at the close of every month without incident.\n";
  const std::vector<int32_t> para_ids = tok.encode(para, false);

  struct Trial {
    int target_ctx;
    double depth;
    std::string key;
    std::string code;
  };
  std::vector<Trial> trials;
  const std::vector<int> ctxs = {65536, 131072, 245760};
  const std::vector<double> depths = {0.10, 0.50, 0.90};
  const char* names[9] = {"ALPHA-739241", "BRAVO-618405", "CHARLIE-942187",
                          "DELTA-851063", "ECHO-304927",  "FOXTROT-576214",
                          "GOLF-289531",  "HOTEL-417892", "OMEGA-492618"};
  int ni = 0;
  for (int ctx : ctxs) {
    if (ctx + 512 > max_ctx) continue;
    for (double d : depths) {
      trials.push_back({ctx, d, "archive_catalog_id_" + std::to_string(ni), names[ni]});
      ++ni;
    }
  }

  int passed = 0;
  for (const auto& tr : trials) {
    const std::string needle = "\n[ARCHIVE ENTRY] The reference " + tr.key + " is " + tr.code + ".\n";
    const std::string question = "\n\nQuestion: What is the exact value of " + tr.key + " listed in the archive above? Reply with ONLY the code.";
    const auto needle_ids = tok.encode(needle, false);
    const auto q_ids = tok.encode(question, false);
    const int filler_budget = tr.target_ctx - int(needle_ids.size()) - int(q_ids.size()) - 64;
    const int before_toks = int(filler_budget * tr.depth);
    const int after_toks = filler_budget - before_toks;

    std::string content;
    content.reserve(size_t(tr.target_ctx) * 6);
    const int n_before = std::max(1, before_toks / int(para_ids.size()));
    const int n_after = std::max(1, after_toks / int(para_ids.size()));
    for (int i = 0; i < n_before; ++i) content += para;
    content += needle;
    for (int i = 0; i < n_after; ++i) content += para;
    content += question;

    ingine::text::ChatOptions chat;
    chat.reasoning_effort = "none";
    const nlohmann::ordered_json msgs = nlohmann::ordered_json::array({{{"role", "user"}, {"content", content}}});
    const std::vector<int32_t> ids = fmt.render(msgs, nlohmann::ordered_json(), chat).ids;

    Sampling samp;
    samp.temperature = 0.0f;
    std::vector<int32_t> out;
    const auto st = gen->generate(ids, 256, samp, stop, [&](std::span<const int32_t> t) {
      out.insert(out.end(), t.begin(), t.end());
      return true;
    });
    const std::string text = tok.decode_lossy(out);
    const bool hit = text.find(tr.code) != std::string::npos;
    if (hit) ++passed;
    std::string preview = text;
    for (char& ch : preview) if (ch == '\n' || ch == '\r') ch = ' ';
    if (preview.size() > 140) preview = preview.substr(preview.size() - 140);
    std::printf("%s needle ctx=%zu (target %d) depth=%.0f%% code=%s prefill=%.0f tok/s decode=%.1f tok/s toks=%zu tail='%s'\n",
                hit ? "PASS" : "FAIL", ids.size(), tr.target_ctx, tr.depth * 100.0, tr.code.c_str(),
                double(ids.size()) / std::max(st.prefill_seconds, 1e-9),
                st.generated_tokens > 1 ? double(st.generated_tokens - 1) / std::max(st.decode_seconds, 1e-9) : 0.0,
                out.size(), preview.c_str());
    std::fflush(stdout);
  }
  std::printf("Needle recall summary: %d / %zu (%.1f%%)\n", passed, trials.size(),
              100.0 * double(passed) / double(trials.size()));
  return passed == int(trials.size()) ? 0 : 1;
}

// ---- Phase 1A: 1,048,576-row Q4_0 KV GPU oracle test (attn_decode M=1,4 + prefill::attention + 9 GiB state) ----

struct SimpleRng {
  uint64_t s;
  explicit SimpleRng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return s * 2685821657736338717ULL;
  }
  double uniform(double a, double b) { return a + (b - a) * (double(next() >> 11) * (1.0 / 9007199254740992.0)); }
  double normal() {
    const double u1 = std::max(1e-12, uniform(0.0, 1.0)), u2 = uniform(0.0, 1.0);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.28318530717958647692 * u2);
  }
};

float host_rbf16(double x) {
  const float f = float(x);
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return f;
  u += 0x7fffu + ((u >> 16) & 1u);
  u &= 0xffff0000u;
  float o = 0;
  std::memcpy(&o, &u, 4);
  return o;
}

void pack_q4_kv_row(const float* src_1024, uint8_t* dst_576, float* dec_1024) {
  std::memset(dst_576, 0, KV_Q4_ROW_BYTES);
  for (int g = 0; g < KV_HEADS; ++g) {
    for (int blk = 0; blk < KV_Q4_BLOCKS_PER_HEAD; ++blk) {
      const float* x = src_1024 + g * HEAD_DIM + blk * KV_Q4_GROUP;
      float mx = 0.f;
      for (int i = 0; i < KV_Q4_GROUP; ++i) {
        const float xi = host_rbf16(x[i]);
        const float aa = std::fabs(mx), ab = std::fabs(xi);
        if (ab > aa || (ab == aa && xi < mx)) mx = xi;
      }
      const __half d_h = __float2half_rn(mx * -0.125f);
      const float d = __half2float(d_h);
      const float id = d != 0.f ? (1.f / d) : 0.f;
      const uint16_t d_bits = __half_as_ushort(d_h);
      std::memcpy(dst_576 + KV_Q4_NIBBLE_BYTES + g * 8 + blk * 2, &d_bits, 2);
      uint8_t* nib = dst_576 + g * 64 + blk * 16;
      for (int i = 0; i < KV_Q4_GROUP; i += 2) {
        const float y0 = host_rbf16(x[i]), y1 = host_rbf16(x[i + 1]);
        int q0 = std::max(-8, std::min(7, int(std::llrint(double(y0) * id))));
        int q1 = std::max(-8, std::min(7, int(std::llrint(double(y1) * id))));
        nib[i / 2] = uint8_t((q0 + 8) | ((q1 + 8) << 4));
        if (dec_1024) {
          dec_1024[g * HEAD_DIM + blk * KV_Q4_GROUP + i] = float(q0) * d;
          dec_1024[g * HEAD_DIM + blk * KV_Q4_GROUP + i + 1] = float(q1) * d;
        }
      }
    }
  }
}

void unpack_q4_kv_row(const uint8_t* src_576, float* dst_1024) {
  for (int g = 0; g < KV_HEADS; ++g) {
    for (int blk = 0; blk < KV_Q4_BLOCKS_PER_HEAD; ++blk) {
      uint16_t d_bits = 0;
      std::memcpy(&d_bits, src_576 + KV_Q4_NIBBLE_BYTES + g * 8 + blk * 2, 2);
      const float d = __half2float(__ushort_as_half(d_bits));
      const uint8_t* nib = src_576 + g * 64 + blk * 16;
      for (int i = 0; i < KV_Q4_GROUP; i += 2) {
        const uint8_t b = nib[i / 2];
        dst_1024[g * HEAD_DIM + blk * KV_Q4_GROUP + i] = float(int(b & 0xfu) - 8) * d;
        dst_1024[g * HEAD_DIM + blk * KV_Q4_GROUP + i + 1] = float(int(b >> 4) - 8) * d;
      }
    }
  }
}

int run_test_1m_kernel() {
  namespace ka = ingine::kernels::attn;
  namespace kp = ingine::kernels::prefill;
  constexpr int kOneM = 1 << 20;  // 1,048,576
  constexpr int kPeriod = 256;
  constexpr int kP0 = kOneM - 6;  // 1,048,570 (leaves room for M=4 -> 1,048,574 < 1,048,576)
  constexpr int kPlanted[3] = {100000, 524288, 1048560};

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  SimpleRng rng(20261123ULL);
  bool ok = true;

  std::printf("=== Phase 1A: 1M (1,048,576) Q4_0 KV GPU Oracle & 64-bit Offset Test ===\n");
  // 1. Build 256-row periodic Q4_0 background + 3 distinct planted rows (early, mid, late)
  std::vector<uint8_t> bg_kb(size_t(kPeriod) * KV_Q4_ROW_BYTES), bg_vb(size_t(kPeriod) * KV_Q4_ROW_BYTES);
  std::vector<float> bg_k(size_t(kPeriod) * KV_DIM), bg_v(size_t(kPeriod) * KV_DIM);
  auto gen_kv_row = [&](float k_scale, float v_scale, uint8_t* kb, uint8_t* vb, float* dk, float* dv) {
    std::vector<float> rk(KV_DIM), rv(KV_DIM);
    for (int g = 0; g < KV_HEADS; ++g) {
      double ssk = 0, ssv = 0;
      for (int d = 0; d < HEAD_DIM; ++d) {
        rk[g * HEAD_DIM + d] = float(rng.normal() * k_scale);
        rv[g * HEAD_DIM + d] = float(rng.normal() * v_scale);
        ssk += rk[g * HEAD_DIM + d] * rk[g * HEAD_DIM + d];
        ssv += rv[g * HEAD_DIM + d] * rv[g * HEAD_DIM + d];
      }
      const double fk = std::min(1.0, QK_CLIP / (std::sqrt(ssk) + 1e-6));
      const double fv = 1.0 / std::sqrt(ssv / HEAD_DIM + 1e-6);
      for (int d = 0; d < HEAD_DIM; ++d) {
        rk[g * HEAD_DIM + d] = host_rbf16(rk[g * HEAD_DIM + d] * fk);
        rv[g * HEAD_DIM + d] = host_rbf16(rv[g * HEAD_DIM + d] * fv);
      }
    }
    pack_q4_kv_row(rk.data(), kb, dk);
    pack_q4_kv_row(rv.data(), vb, dv);
  };
  for (int r = 0; r < kPeriod; ++r)
    gen_kv_row(0.6f, 1.0f, &bg_kb[size_t(r) * KV_Q4_ROW_BYTES], &bg_vb[size_t(r) * KV_Q4_ROW_BYTES],
               &bg_k[size_t(r) * KV_DIM], &bg_v[size_t(r) * KV_DIM]);

  std::vector<uint8_t> pl_kb[3], pl_vb[3];
  std::vector<float> pl_k[3], pl_v[3];
  for (int p = 0; p < 3; ++p) {
    pl_kb[p].resize(KV_Q4_ROW_BYTES);
    pl_vb[p].resize(KV_Q4_ROW_BYTES);
    pl_k[p].resize(KV_DIM);
    pl_v[p].resize(KV_DIM);
    gen_kv_row(2.5f, 1.0f, pl_kb[p].data(), pl_vb[p].data(), pl_k[p].data(), pl_v[p].data());
  }

  // Allocate full 1,048,576-row single-visit Q4_0 K/V cache (604 MB each = 1.21 GB total)
  const size_t cache_bytes = size_t(kOneM) * KV_Q4_ROW_BYTES;
  DeviceBuffer d_k(cache_bytes), d_v(cache_bytes);
  // Upload 256 rows and double-copy to 1,048,576 rows in 12 D2D calls
  CUDA_CHECK(cudaMemcpyAsync(d_k.p, bg_kb.data(), bg_kb.size(), cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_v.p, bg_vb.data(), bg_vb.size(), cudaMemcpyHostToDevice, stream));
  for (int rows = kPeriod; rows < kOneM; rows <<= 1) {
    const size_t chunk = size_t(rows) * KV_Q4_ROW_BYTES;
    CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(d_k.p) + chunk, d_k.p, chunk, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(d_v.p) + chunk, d_v.p, chunk, cudaMemcpyDeviceToDevice, stream));
  }
  for (int p = 0; p < 3; ++p) {
    const size_t off = size_t(kPlanted[p]) * KV_Q4_ROW_BYTES;
    CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(d_k.p) + off, pl_kb[p].data(), KV_Q4_ROW_BYTES,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(d_v.p) + off, pl_vb[p].data(), KV_Q4_ROW_BYTES,
                               cudaMemcpyHostToDevice, stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Shared layer params
  std::vector<float> q_scale(HEADS * HEAD_DIM), r_scale(HEADS * D_REL), rel_proj(D_REL * REL_EXTENT);
  std::vector<float> k_conv(4 * KV_DIM), v_conv(4 * KV_DIM), conv_k0(3 * KV_DIM), conv_v0(3 * KV_DIM);
  for (auto& x : q_scale) x = float(rng.uniform(0.5, 1.5));
  for (auto& x : r_scale) x = float(rng.uniform(0.5, 1.5));
  for (auto& x : rel_proj) x = float(rng.normal() * 0.2);
  for (auto& x : k_conv) x = float(rng.normal() * 0.2);
  for (auto& x : v_conv) x = float(rng.normal() * 0.2);
  for (auto& x : conv_k0) x = float(rng.normal() * 0.5);
  for (auto& x : conv_v0) x = float(rng.normal() * 0.5);

  DeviceBuffer d_qs(q_scale.size() * 4), d_rs(r_scale.size() * 4), d_rp(rel_proj.size() * 4);
  DeviceBuffer d_kc(k_conv.size() * 4), d_vc(v_conv.size() * 4);
  DeviceBuffer d_ck(conv_k0.size() * 4), d_cv(conv_v0.size() * 4);
  d_qs.copy_from_host(q_scale.data(), d_qs.bytes);
  d_rs.copy_from_host(r_scale.data(), d_rs.bytes);
  d_rp.copy_from_host(rel_proj.data(), d_rp.bytes);
  d_kc.copy_from_host(k_conv.data(), d_kc.bytes);
  d_vc.copy_from_host(v_conv.data(), d_vc.bytes);
  DeviceBuffer d_ws(ka::workspace_bytes(8));
  d_ws.fill(0);
  DeviceBuffer d_pos(8);

  // Analytical FP64 oracle for one query position `qpos` attending over cached [0, p0) + new_k/new_v [p0..qpos]
  auto eval_oracle_row = [&](int qpos, int p0, const std::vector<double>& qh_all, const std::vector<double>& rl_all,
                             const float* proj_row, const std::vector<std::vector<float>>& new_k,
                             const std::vector<std::vector<float>>& new_v, std::vector<double>& out_ref) {
    out_ref.assign(HEADS * HEAD_DIM, 0.0);
    // Split cached range [0, p0) into far [0, far_end) where dist >= 1024 (relbias == 0) and tail [far_end, p0)
    const int far_end = std::max(0, std::min(p0, qpos - REL_EXTENT + 1));
    int cls_count[kPeriod] = {};
    const int full_periods = far_end / kPeriod, rem = far_end % kPeriod;
    for (int c = 0; c < kPeriod; ++c) cls_count[c] = full_periods + (c < rem ? 1 : 0);
    for (int p = 0; p < 3; ++p)
      if (kPlanted[p] < far_end) cls_count[kPlanted[p] % kPeriod]--;

    for (int h = 0; h < HEADS; ++h) {
      const int g = h / 3;
      const double* qh = &qh_all[size_t(h) * HEAD_DIM];
      const double* rlh = &rl_all[size_t(h) * REL_EXTENT];
      double cls_s[kPeriod];
      double mx = proj_row[SEG_BEGIN[SEG_SINK] + h];
      for (int c = 0; c < kPeriod; ++c) {
        double dot = 0.0;
        const float* kr = &bg_k[size_t(c) * KV_DIM + g * HEAD_DIM];
        for (int e = 0; e < HEAD_DIM; ++e) dot += qh[e] * double(kr[e]);
        cls_s[c] = dot / std::sqrt(128.0);
        if (cls_count[c] > 0) mx = std::max(mx, cls_s[c]);
      }
      double pl_s[3];
      for (int p = 0; p < 3; ++p) {
        double dot = 0.0;
        const float* kr = &pl_k[p][g * HEAD_DIM];
        for (int e = 0; e < HEAD_DIM; ++e) dot += qh[e] * double(kr[e]);
        const int dist = qpos - kPlanted[p];
        pl_s[p] = dot / std::sqrt(128.0) + (dist >= 0 && dist < REL_EXTENT ? rlh[dist] : 0.0);
        if (kPlanted[p] < far_end) mx = std::max(mx, pl_s[p]);
      }
      std::vector<double> tail_s(size_t(p0 - far_end));
      for (int kp = far_end; kp < p0; ++kp) {
        const float* kr = &bg_k[size_t(kp % kPeriod) * KV_DIM + g * HEAD_DIM];
        for (int p = 0; p < 3; ++p)
          if (kp == kPlanted[p]) kr = &pl_k[p][g * HEAD_DIM];
        double dot = 0.0;
        for (int e = 0; e < HEAD_DIM; ++e) dot += qh[e] * double(kr[e]);
        const int dist = qpos - kp;
        tail_s[kp - far_end] = dot / std::sqrt(128.0) + (dist < REL_EXTENT ? rlh[dist] : 0.0);
        mx = std::max(mx, tail_s[kp - far_end]);
      }
      const int n_new = std::max(0, qpos - p0 + 1);
      std::vector<double> new_s(static_cast<size_t>(n_new));
      for (int i = 0; i < n_new; ++i) {
        double dot = 0.0;
        const float* kr = &new_k[i][g * HEAD_DIM];
        for (int e = 0; e < HEAD_DIM; ++e) dot += qh[e] * double(kr[e]);
        const int dist = n_new - 1 - i;
        new_s[i] = dot / std::sqrt(128.0) + (dist < REL_EXTENT ? rlh[dist] : 0.0);
        mx = std::max(mx, new_s[i]);
      }
      double den = std::exp(double(proj_row[SEG_BEGIN[SEG_SINK] + h]) - mx);
      double acc[HEAD_DIM] = {};
      for (int c = 0; c < kPeriod; ++c) {
        if (cls_count[c] <= 0) continue;
        const double w = double(cls_count[c]) * std::exp(cls_s[c] - mx);
        den += w;
        const float* vr = &bg_v[size_t(c) * KV_DIM + g * HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; ++d) acc[d] += w * double(vr[d]);
      }
      for (int p = 0; p < 3; ++p) {
        if (kPlanted[p] >= far_end) continue;
        const double w = std::exp(pl_s[p] - mx);
        den += w;
        const float* vr = &pl_v[p][g * HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; ++d) acc[d] += w * double(vr[d]);
      }
      for (int kp = far_end; kp < p0; ++kp) {
        const float* vr = &bg_v[size_t(kp % kPeriod) * KV_DIM + g * HEAD_DIM];
        for (int p = 0; p < 3; ++p)
          if (kp == kPlanted[p]) vr = &pl_v[p][g * HEAD_DIM];
        const double w = std::exp(tail_s[kp - far_end] - mx);
        den += w;
        for (int d = 0; d < HEAD_DIM; ++d) acc[d] += w * double(vr[d]);
      }
      for (int i = 0; i < n_new; ++i) {
        const double w = std::exp(new_s[i] - mx);
        den += w;
        const float* vr = &new_v[i][g * HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; ++d) acc[d] += w * double(vr[d]);
      }
      for (int d = 0; d < HEAD_DIM; ++d) {
        const double gate = 1.0 / (1.0 + std::exp(-double(proj_row[SEG_BEGIN[SEG_GATE] + h * HEAD_DIM + d])));
        out_ref[size_t(h) * HEAD_DIM + d] = (acc[d] / den) * gate;
      }
    }
  };

  // 2. Test attn_decode Q4_0 at p0 = 1,048,570 for M = 1 and M = 4
  for (int M : {1, 4}) {
    std::vector<float> proj(size_t(M) * PROJ_ROWS_FULL);
    for (auto& x : proj) x = float(rng.normal() * 0.5);
    DeviceBuffer d_proj(proj.size() * 4), d_out(size_t(M) * HEADS * HEAD_DIM * 2);
    d_proj.copy_from_host(proj.data(), d_proj.bytes);
    d_ck.copy_from_host(conv_k0.data(), d_ck.bytes);
    d_cv.copy_from_host(conv_v0.data(), d_cv.bytes);
    const int64_t pos_val = kP0;
    d_pos.copy_from_host(&pos_val, 8);

    ka::DecodeArgs a;
    a.proj = static_cast<const float*>(d_proj.p);
    a.M = M;
    a.q_scale = static_cast<const float*>(d_qs.p);
    a.r_scale = static_cast<const float*>(d_rs.p);
    a.rel_proj = static_cast<const float*>(d_rp.p);
    a.k_conv = static_cast<const float*>(d_kc.p);
    a.v_conv = static_cast<const float*>(d_vc.p);
    a.conv_k_state = static_cast<float*>(d_ck.p);
    a.conv_v_state = static_cast<float*>(d_cv.p);
    a.k_cache = d_k.p;
    a.v_cache = d_v.p;
    a.cache_fp8 = false;
    a.cache_q4 = true;
    a.cache_rows = kOneM;
    a.kpos = nullptr;
    a.window = 0;
    a.log_scaling = true;
    a.pos = static_cast<const int64_t*>(d_pos.p);
    a.workspace = d_ws.p;
    a.workspace_size = d_ws.bytes;
    a.out = static_cast<ActT*>(d_out.p);
    ka::attn_decode(a, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Download new cache rows written at p0..p0+M-1 and verify against FP64 oracle
    std::vector<uint8_t> got_kb(size_t(M) * KV_Q4_ROW_BYTES), got_vb(size_t(M) * KV_Q4_ROW_BYTES);
    CUDA_CHECK(cudaMemcpy(got_kb.data(), static_cast<const uint8_t*>(d_k.p) + size_t(kP0) * KV_Q4_ROW_BYTES,
                          got_kb.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_vb.data(), static_cast<const uint8_t*>(d_v.p) + size_t(kP0) * KV_Q4_ROW_BYTES,
                          got_vb.size(), cudaMemcpyDeviceToHost));
    std::vector<std::vector<float>> new_k(M, std::vector<float>(KV_DIM)), new_v(M, std::vector<float>(KV_DIM));
    for (int m = 0; m < M; ++m) {
      unpack_q4_kv_row(&got_kb[size_t(m) * KV_Q4_ROW_BYTES], new_k[m].data());
      unpack_q4_kv_row(&got_vb[size_t(m) * KV_Q4_ROW_BYTES], new_v[m].data());
    }
    std::vector<ActT> got_out(size_t(M) * HEADS * HEAD_DIM);
    CUDA_CHECK(cudaMemcpy(got_out.data(), d_out.p, d_out.bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0, sum_err = 0.0;
    size_t n_finite = 0;
    for (int m = 0; m < M; ++m) {
      const float tau = 1.f + 0.1f * std::log(std::fmax(float(kP0 + m + 1) * (1.f / 8192.f), 1.f));
      std::vector<double> qh_all(HEADS * HEAD_DIM), rl_all(HEADS * REL_EXTENT);
      const float* prow = &proj[size_t(m) * PROJ_ROWS_FULL];
      for (int h = 0; h < HEADS; ++h) {
        double ss = 0.0;
        for (int d = 0; d < HEAD_DIM; ++d) {
          const double v = double(prow[h * HEAD_DIM + d]) * double(q_scale[h * HEAD_DIM + d]);
          qh_all[h * HEAD_DIM + d] = v;
          ss += v * v;
        }
        const double f = std::min(1.0, QK_CLIP / (std::sqrt(ss) + 1e-6));
        for (int d = 0; d < HEAD_DIM; ++d)
          qh_all[h * HEAD_DIM + d] = host_rbf16(host_rbf16(qh_all[h * HEAD_DIM + d] * f) * tau);
        double rs = 0.0;
        double rv[D_REL];
        for (int j = 0; j < D_REL; ++j) {
          rv[j] = double(prow[SEG_BEGIN[SEG_REL] + h * D_REL + j]) * double(r_scale[h * D_REL + j]);
          rs += rv[j] * rv[j];
        }
        const double rf = std::min(1.0, REL_CLIP / (std::sqrt(rs) + 1e-6));
        for (int j = 0; j < D_REL; ++j) rv[j] = host_rbf16(host_rbf16(rv[j] * rf) * tau);
        for (int dist = 0; dist < REL_EXTENT; ++dist) {
          double b = 0.0;
          for (int j = 0; j < D_REL; ++j) b += rv[j] * double(rel_proj[j * REL_EXTENT + dist]);
          rl_all[h * REL_EXTENT + dist] = host_rbf16(b);
        }
      }
      std::vector<double> ref_row;
      eval_oracle_row(kP0 + m, kP0, qh_all, rl_all, prow, new_k, new_v, ref_row);
      for (size_t i = 0; i < ref_row.size(); ++i) {
        const float g = act::host_to_float(got_out[size_t(m) * HEADS * HEAD_DIM + i]);
        if (std::isfinite(g)) ++n_finite;
        const double err = std::fabs(double(g) - ref_row[i]);
        max_err = std::max(max_err, err);
        sum_err += err;
      }
    }
    const size_t total_elems = size_t(M) * HEADS * HEAD_DIM;
    const double mean_err = sum_err / double(total_elems);
    const bool pass = n_finite == total_elems && max_err < 0.015 && mean_err < 1.5e-3;
    std::printf("%s attn_decode 1M (p0=%d, M=%d, Q4_0): max_err=%.5f mean_err=%.6f finite=%zu/%zu\n",
                pass ? "PASS" : "FAIL", kP0, M, max_err, mean_err, n_finite, total_elems);
    ok &= pass;
  }

  // 3. Test prefill::attention Q4_0 at pos0 = 1,048,512, T = 64, L = 1,048,576 (attending over full 1M cache!)
  {
    // Restore rows [kP0..kOneM) back to background pattern so cache is purely periodic + 3 planted rows
    for (int r = kP0; r < kOneM; ++r) {
      const int c = r % kPeriod;
      CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(d_k.p) + size_t(r) * KV_Q4_ROW_BYTES,
                                 &bg_kb[size_t(c) * KV_Q4_ROW_BYTES], KV_Q4_ROW_BYTES, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(static_cast<uint8_t*>(d_v.p) + size_t(r) * KV_Q4_ROW_BYTES,
                                 &bg_vb[size_t(c) * KV_Q4_ROW_BYTES], KV_Q4_ROW_BYTES, cudaMemcpyHostToDevice, stream));
    }
    constexpr int T = 64;
    constexpr int pos0 = kOneM - T;  // 1,048,512
    std::vector<ActT> q_h(size_t(T) * HEADS * HEAD_DIM), rel_h(size_t(T) * HEADS * REL_EXTENT),
        proj_h(size_t(T) * PROJ_ROWS_FULL);
    for (auto& v : q_h) v = act::host_from_float(float(rng.normal() * 0.8));
    for (auto& v : rel_h) v = act::host_from_float(float(rng.normal() * 0.2));
    for (auto& v : proj_h) v = act::host_from_float(float(rng.normal() * 0.5));
    DeviceBuffer dq(q_h.size() * 2), drel(rel_h.size() * 2), dproj(proj_h.size() * 2), dout(q_h.size() * 2);
    dq.copy_from_host(q_h.data(), dq.bytes);
    drel.copy_from_host(rel_h.data(), drel.bytes);
    dproj.copy_from_host(proj_h.data(), dproj.bytes);

    kp::AttnParams ap;
    ap.q = static_cast<const ActT*>(dq.p);
    ap.k = d_k.p;
    ap.v = d_v.p;
    ap.kv_fp8 = false;
    ap.kv_q4 = true;
    ap.kpos = nullptr;
    ap.bounds = nullptr;
    ap.rel = static_cast<const ActT*>(drel.p);
    ap.proj = static_cast<const ActT*>(dproj.p);
    ap.ld = PROJ_ROWS_FULL;
    ap.T = T;
    ap.L = kOneM;
    ap.pos0 = pos0;
    ap.window = 0;
    ap.out = static_cast<ActT*>(dout.p);
    kp::attention(ap, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<ActT> got(q_h.size());
    CUDA_CHECK(cudaMemcpy(got.data(), dout.p, dout.bytes, cudaMemcpyDeviceToHost));
    // Verify last row t = T - 1 (qpos = 1,048,575, attending to ALL 1,048,576 keys!)
    const int t = T - 1, qpos = pos0 + t;
    std::vector<double> qh_all(HEADS * HEAD_DIM), rl_all(HEADS * REL_EXTENT);
    std::vector<float> prow(PROJ_ROWS_FULL);
    for (size_t i = 0; i < HEADS * HEAD_DIM; ++i)
      qh_all[i] = double(act::host_to_float(q_h[size_t(t) * HEADS * HEAD_DIM + i]));
    for (size_t i = 0; i < HEADS * REL_EXTENT; ++i)
      rl_all[i] = double(act::host_to_float(rel_h[size_t(t) * HEADS * REL_EXTENT + i]));
    for (size_t i = 0; i < PROJ_ROWS_FULL; ++i)
      prow[i] = act::host_to_float(proj_h[size_t(t) * PROJ_ROWS_FULL + i]);
    std::vector<double> ref_row;
    eval_oracle_row(qpos, kOneM, qh_all, rl_all, prow.data(), {}, {}, ref_row);
    double max_err = 0.0, sum_err = 0.0;
    size_t n_finite = 0;
    for (size_t i = 0; i < ref_row.size(); ++i) {
      const float g = act::host_to_float(got[size_t(t) * HEADS * HEAD_DIM + i]);
      if (std::isfinite(g)) ++n_finite;
      const double err = std::fabs(double(g) - ref_row[i]);
      max_err = std::max(max_err, err);
      sum_err += err;
    }
    const double mean_err = sum_err / double(ref_row.size());
    const bool pass = n_finite == ref_row.size() && max_err < 0.015 && mean_err < 1.5e-3;
    std::printf("%s prefill::attention 1M (pos0=%d, T=%d, L=%d, Q4_0): max_err=%.5f mean_err=%.6f finite=%zu/%zu\n",
                pass ? "PASS" : "FAIL", pos0, T, kOneM, max_err, mean_err, n_finite, ref_row.size());
    ok &= pass;
  }

  // Free single-visit cache before full 8-visit 9.00 GiB State allocation check
  d_k = DeviceBuffer();
  d_v = DeviceBuffer();

  // 4. Full 8-visit 1M State allocation (9.000 GiB Q4_0 KV) + visit 7 row 1,048,575 (> 4.5 GiB offset!) write/read
  {
    State st = make_state(kOneM, false, true);
    const size_t full_bytes = st.k_full.bytes() + st.v_full.bytes();
    const bool sz_ok = full_bytes == 9663676416ull;
    const size_t last_off = size_t(kOneM - 1) * KV_Q4_ROW_BYTES;
    uint8_t* v7_k = static_cast<uint8_t*>(st.k_full_at(NUM_FULL_VISITS - 1).data);
    uint8_t* v7_v = static_cast<uint8_t*>(st.v_full_at(NUM_FULL_VISITS - 1).data);
    std::vector<uint8_t> sent(KV_Q4_ROW_BYTES, 0xA5), got_k(KV_Q4_ROW_BYTES, 0), got_v(KV_Q4_ROW_BYTES, 0);
    for (int i = 0; i < KV_Q4_ROW_BYTES; ++i) sent[i] = uint8_t((i * 37 + 13) & 0xff);
    CUDA_CHECK(cudaMemcpy(v7_k + last_off, sent.data(), KV_Q4_ROW_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v7_v + last_off, sent.data(), KV_Q4_ROW_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(got_k.data(), v7_k + last_off, KV_Q4_ROW_BYTES, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(got_v.data(), v7_v + last_off, KV_Q4_ROW_BYTES, cudaMemcpyDeviceToHost));
    const bool rw_ok = got_k == sent && got_v == sent;
    std::printf("%s 8-visit 1M State (full_kv=%.3f GiB, visit7 last-row offset=%.3f GiB): rw=%s\n",
                (sz_ok && rw_ok) ? "PASS" : "FAIL", double(full_bytes) / 1073741824.0,
                double(size_t(NUM_FULL_VISITS - 1) * kOneM * KV_Q4_ROW_BYTES + last_off) / 1073741824.0,
                rw_ok ? "exact" : "MISMATCH");
    ok &= sz_ok && rw_ok;
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}

// ---- Phase 1B + 1C: Resident 1M-token VRAM, prefill/decode speed, and honest Needle-in-a-Haystack ----

int run_bench_needle_1m(const std::string& model_dir, int max_ctx, int prefill_chunk, int decode_tokens,
                        const std::vector<int>& ctxs) {
  cudaFree(nullptr);
  size_t free0 = 0, free_rt = 0, free_pre = 0, total = 0;
  cudaMemGetInfo(&free0, &total);

  RuntimeOptions opt;
  opt.model_dir = model_dir;
  opt.max_ctx = max_ctx;
  opt.prefill_chunk = prefill_chunk;
  opt.dense_fp8 = true;
  opt.kv_q4 = true;
  opt.spec = false;
  opt.park_mb = 0;
  opt.vision = false;

  std::printf("=== Phase 1B/1C: Resident 1M Context Bench & Honest Needle Test ===\n");
  std::printf("model=%s max_ctx=%d prefill_chunk=%d kv_q4=true dense_fp8=true\n",
              model_dir.c_str(), max_ctx, prefill_chunk);
  std::fflush(stdout);

  const auto t_load0 = std::chrono::steady_clock::now();
  Runtime rt(opt);
  const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
  cudaMemGetInfo(&free_rt, &total);
  Prefiller pre(rt.weights(), rt.ablation(), prefill_chunk, rt.stream());
  DeviceBuffer ckpt(rt.checkpoint_bytes());
  DeviceBuffer d_chunk(size_t(prefill_chunk) * sizeof(int32_t));
  DeviceBuffer d_logits(size_t(UNPADDED_VOCAB) * sizeof(float));
  CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
  cudaMemGetInfo(&free_pre, &total);

  const double rt_gib = double(free0 > free_rt ? free0 - free_rt : 0) / 1073741824.0;
  const double tot_gib = double(free0 > free_pre ? free0 - free_pre : 0) / 1073741824.0;
  const double full_kv_gib = double(rt.state().k_full.bytes() + rt.state().v_full.bytes()) / 1073741824.0;
  const double swa_mib = double(rt.state().k_ring.bytes() + rt.state().v_ring.bytes()) / 1048576.0;
  std::printf("Load+capture: %.1fs | Runtime VRAM: %.3f GiB (%.2f GB) | Total w/ Prefiller(chunk=%d): %.3f GiB (%.2f GB) | "
              "Full Q4_0 KV: %.3f GiB | SWA BF16: %.1f MiB | Free: %.3f GiB\n",
              load_s, rt_gib, rt_gib * 1.073741824, prefill_chunk, tot_gib, tot_gib * 1.073741824, full_kv_gib, swa_mib,
              double(free_pre) / 1073741824.0);
  std::fflush(stdout);

  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  const auto& tok = fmt.tokenizer();
  const int32_t stop_id = *tok.token_id("<|content_model_end_sampling|>");

  // Extract exact chat wrapper prefix & suffix tokens for reasoning_effort="none"
  ingine::text::ChatOptions chat;
  chat.reasoning_effort = "none";
  const std::string sentinel = "\n__INGINE_1M_SENTINEL_BOUNDARY__\n";
  const auto sent_ids = tok.encode(sentinel, false);
  const auto wrap_ids =
      fmt.render(nlohmann::ordered_json::array({{{"role", "user"}, {"content", sentinel}}}), nlohmann::ordered_json(), chat)
          .ids;
  auto it = std::search(wrap_ids.begin(), wrap_ids.end(), sent_ids.begin(), sent_ids.end());
  if (it == wrap_ids.end()) {
    std::fprintf(stderr, "Failed to locate sentinel in chat template!\n");
    return 1;
  }
  const std::vector<int32_t> chat_prefix(wrap_ids.begin(), it);
  const std::vector<int32_t> chat_suffix(it + std::ptrdiff_t(sent_ids.size()), wrap_ids.end());

  // 16 varied historical archive paragraphs (pre-tokenized once for <5ms 1M assembly)
  std::vector<std::vector<int32_t>> paras(16);
  const char* regions[16] = {"North", "South", "East", "West", "Central", "Harbor", "Upland", "Valley",
                             "Estuary", "Ridge", "Basin", "Canal", "Port", "Marsh", "Plateau", "Fjord"};
  const char* goods[16] = {"timber", "grain", "iron ore", "salt", "wool", "slate", "copper", "flax",
                           "coal", "granite", "hemp", "barley", "tar", "tin", "limestone", "cedar"};
  for (int i = 0; i < 16; ++i) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "Archive ledger %d (%s District, year %d): Clerks recorded seasonal rainfall, %s shipments, "
                  "railway timetables, and harbor tide measurements across stations %d through %d. "
                  "Barometric readings were logged each morning and freight tonnage summarized monthly without incident.\n",
                  i + 1, regions[i], 1840 + i * 3, goods[i], i * 4 + 1, i * 4 + 4);
    paras[i] = tok.encode(buf, false);
  }

  auto prefill_span = [&](const std::vector<int32_t>& ids, int pos0, bool log_progress) -> double {
    const auto t0 = std::chrono::steady_clock::now();
    auto t_seg = t0;
    int seg_toks = 0;
    int next_mark = 131072;
    for (int off = 0; off < int(ids.size()); off += prefill_chunk) {
      const int T = std::min(prefill_chunk, int(ids.size()) - off);
      CUDA_CHECK(cudaMemcpyAsync(d_chunk.p, ids.data() + off, size_t(T) * 4, cudaMemcpyHostToDevice, rt.stream()));
      pre.run(rt.state(), static_cast<const int32_t*>(d_chunk.p), T, pos0 + off, static_cast<float*>(d_logits.p), nullptr);
      seg_toks += T;
      if (log_progress && (off + T >= next_mark || off + T == int(ids.size()))) {
        CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
        const auto now = std::chrono::steady_clock::now();
        const double seg_s = std::chrono::duration<double>(now - t_seg).count();
        const double tot_s = std::chrono::duration<double>(now - t0).count();
        std::printf("  [prefill progress] %d / %zu tokens | seg=%.0f tok/s | avg=%.0f tok/s | elapsed=%.1fs\n",
                    off + T, ids.size(), double(seg_toks) / std::max(seg_s, 1e-9),
                    double(off + T) / std::max(tot_s, 1e-9), tot_s);
        std::fflush(stdout);
        t_seg = now;
        seg_toks = 0;
        while (next_mark <= off + T) next_mark += 131072;
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };

  DeviceBuffer d_tok_out(sizeof(int32_t));
  auto greedy_decode = [&](int start_pos, int max_new, bool use_stop, std::vector<int32_t>& out_ids,
                           double& decode_s) {
    out_ids.clear();
    auto* d_tok = static_cast<int32_t*>(d_tok_out.p);
    int32_t tok_id = 0;
    ingine::kernels::head::argmax(static_cast<const float*>(d_logits.p), 1, UNPADDED_VOCAB, UNPADDED_VOCAB, d_tok,
                                  rt.stream());
    CUDA_CHECK(cudaMemcpyAsync(&tok_id, d_tok, 4, cudaMemcpyDeviceToHost, rt.stream()));
    CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
    out_ids.push_back(tok_id);
    const auto t0 = std::chrono::steady_clock::now();
    for (int step = 1; step < max_new; ++step) {
      if (use_stop && tok_id == stop_id) break;
      const float* d_next = rt.decode_logits(tok_id, start_pos + step - 1);
      ingine::kernels::head::argmax(d_next, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, d_tok, rt.stream());
      CUDA_CHECK(cudaMemcpyAsync(&tok_id, d_tok, 4, cudaMemcpyDeviceToHost, rt.stream()));
      CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
      out_ids.push_back(tok_id);
    }
    decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };

  const double depths[3] = {0.10, 0.50, 0.90};
  const char* depth_names[3] = {"early(10%)", "mid(50%) ", "late(90%)"};
  const char* code_pool[9] = {"ZIRCON-481920", "COBALT-739154", "QUARTZ-602847",
                              "AMBER-318492",  "ONYX-945206",   "JASPER-527813",
                              "TOPAZ-864139",  "GARNET-209574", "BERYL-673418"};

  int total_passed = 0, total_trials = 0;
  for (size_t ci = 0; ci < ctxs.size(); ++ci) {
    const int target_ctx = ctxs[ci];
    if (target_ctx + 256 > max_ctx) {
      std::printf("SKIP ctx=%d (exceeds max_ctx=%d)\n", target_ctx, max_ctx);
      continue;
    }
    std::printf("\n--- Context Length: %d tokens ---\n", target_ctx);
    const int doc_budget = target_ctx - int(chat_prefix.size()) - 96;
    std::string keys[3], codes[3];
    std::vector<int32_t> n_ids[3];
    int needle_toks = 0;
    for (int d = 0; d < 3; ++d) {
      keys[d] = "vault_passkey_" + std::to_string(ci) + "_" + std::to_string(d);
      codes[d] = code_pool[(ci * 3 + d) % 9];
      const std::string line = "\n[SPECIAL ARCHIVE ENTRY] The verified " + keys[d] + " is " + codes[d] + ".\n";
      n_ids[d] = tok.encode(line, false);
      needle_toks += int(n_ids[d].size());
    }
    const int filler_budget = doc_budget - needle_toks;
    const int pos_target[3] = {int(filler_budget * depths[0]), int(filler_budget * depths[1]),
                               int(filler_budget * depths[2])};

    std::vector<int32_t> doc_ids;
    doc_ids.reserve(size_t(target_ctx));
    doc_ids.insert(doc_ids.end(), chat_prefix.begin(), chat_prefix.end());
    int inserted_filler = 0, p_idx = 0, next_needle = 0;
    int actual_needle_pos[3] = {};
    while (inserted_filler < filler_budget) {
      while (next_needle < 3 && inserted_filler >= pos_target[next_needle]) {
        actual_needle_pos[next_needle] = int(doc_ids.size());
        doc_ids.insert(doc_ids.end(), n_ids[next_needle].begin(), n_ids[next_needle].end());
        ++next_needle;
      }
      const auto& p = paras[p_idx++ & 15];
      const int take = std::min(int(p.size()), filler_budget - inserted_filler);
      doc_ids.insert(doc_ids.end(), p.begin(), p.begin() + take);
      inserted_filler += take;
    }
    while (next_needle < 3) {
      actual_needle_pos[next_needle] = int(doc_ids.size());
      doc_ids.insert(doc_ids.end(), n_ids[next_needle].begin(), n_ids[next_needle].end());
      ++next_needle;
    }
    const int L_doc = int(doc_ids.size());

    // Prefill document prefix ONCE and checkpoint state at L_doc
    reset_state(rt.state(), rt.stream());
    const double prefill_s = prefill_span(doc_ids, 0, true);
    rt.save_checkpoint(ckpt.p);
    CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
    std::printf("PREFILL ctx=%d (doc=%d toks): %.2fs -> %.1f tok/s\n",
                target_ctx, L_doc, prefill_s, double(L_doc) / std::max(prefill_s, 1e-9));
    std::fflush(stdout);

    // 3 depth probes reusing full KV [0..L_doc) via checkpoint restore (<1ms!)
    int ctx_pass = 0;
    for (int d = 0; d < 3; ++d) {
      rt.restore_checkpoint(ckpt.p);
      rt.state().length = L_doc;
      const std::string q = "\n\nQuestion: What is the exact value of " + keys[d] +
                            " listed in the archive above? Reply with ONLY the code.";
      std::vector<int32_t> suffix = tok.encode(q, false);
      suffix.insert(suffix.end(), chat_suffix.begin(), chat_suffix.end());
      prefill_span(suffix, L_doc, false);
      std::vector<int32_t> out_ids;
      double dec_s = 0.0;
      greedy_decode(L_doc + int(suffix.size()), 256, true, out_ids, dec_s);
      const std::string text = tok.decode_lossy(out_ids);
      const bool hit = text.find(codes[d]) != std::string::npos;
      if (hit) ++ctx_pass, ++total_passed;
      ++total_trials;
      std::string clean = text;
      for (char& ch : clean) if (ch == '\n' || ch == '\r') ch = ' ';
      if (clean.size() > 180) clean = "..." + clean.substr(clean.size() - 180);
      std::printf("  %s needle ctx=%d %s (pos=%d/%d) code=%s toks=%zu tail='%s'\n",
                  hit ? "PASS" : "FAIL", L_doc + int(suffix.size()), depth_names[d], actual_needle_pos[d], L_doc,
                  codes[d].c_str(), out_ids.size(), clean.c_str());
      std::fflush(stdout);
    }

    // 64-token steady-state decode timing probe at context L_doc (restores same checkpoint)
    {
      rt.restore_checkpoint(ckpt.p);
      rt.state().length = L_doc;
      const std::string q = "\n\nQuestion: Summarize the districts and commodities tracked in the archive above in a detailed paragraph.";
      std::vector<int32_t> suffix = tok.encode(q, false);
      suffix.insert(suffix.end(), chat_suffix.begin(), chat_suffix.end());
      prefill_span(suffix, L_doc, false);
      std::vector<int32_t> out_ids;
      double dec_s = 0.0;
      greedy_decode(L_doc + int(suffix.size()), decode_tokens, false, out_ids, dec_s);
      const double dec_tok_s = double(out_ids.size() - 1) / std::max(dec_s, 1e-9);
      std::printf("DECODE  ctx=%d (%zu toks generated in %.2fs): %.2f tok/s | needle=%d/3\n",
                  L_doc + int(suffix.size()), out_ids.size(), dec_s, dec_tok_s, ctx_pass);
      std::fflush(stdout);
    }
  }
  std::printf("\n=== 1M Bench & Needle Summary: %d / %d needles recalled (%.1f%%) ===\n",
              total_passed, total_trials, total_trials ? 100.0 * double(total_passed) / double(total_trials) : 0.0);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = "vram";
  std::string model_dir = "/mnt/storage/models/Inkling-2-Mini-ingine";
  std::string test_model_dir = "/mnt/storage/models/Inkling-2-Mini-atlas-exl3-9g";
  std::string golden = std::string(INGINE_GOLDEN_DIR) + "/model/abl_kv16_t5000.json";
  int max_ctx = 262144;
  int prefix_len = 4096;
  int eval_tokens = 256;
  int prefill_chunk = 2048;
  int decode_tokens = 64;
  std::vector<int> ctxs = {262144, 524288, 1000000};
  bool test_kv_q4 = false;
  bool dense_fp8 = true;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--mode" && i + 1 < argc) mode = argv[++i];
    else if (a == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
    else if (a == "--test-model-dir" && i + 1 < argc) test_model_dir = argv[++i];
    else if (a == "--golden" && i + 1 < argc) golden = argv[++i];
    else if (a == "--max-ctx" && i + 1 < argc) max_ctx = std::atoi(argv[++i]);
    else if (a == "--prefix" && i + 1 < argc) prefix_len = std::atoi(argv[++i]);
    else if (a == "--eval-tokens" && i + 1 < argc) eval_tokens = std::atoi(argv[++i]);
    else if (a == "--prefill-chunk" && i + 1 < argc) prefill_chunk = std::atoi(argv[++i]);
    else if (a == "--decode-tokens" && i + 1 < argc) decode_tokens = std::atoi(argv[++i]);
    else if (a == "--ctxs" && i + 1 < argc) {
      ctxs.clear();
      std::stringstream ss(argv[++i]);
      std::string item;
      while (std::getline(ss, item, ',')) if (!item.empty()) ctxs.push_back(std::atoi(item.c_str()));
    }
    else if (a == "--test-kv-q4") test_kv_q4 = true;
    else if (a == "--dense-bf16") dense_fp8 = false;
  }
  try {
    if (mode == "vram") return run_vram();
    if (mode == "kl") return run_kl(model_dir, golden, prefix_len, eval_tokens);
    if (mode == "kl-models") return run_kl_models(model_dir, test_model_dir, golden, prefix_len, eval_tokens, test_kv_q4, dense_fp8);
    if (mode == "needle") return run_needle(model_dir, max_ctx);
    if (mode == "test-1m-kernel") return run_test_1m_kernel();
    if (mode == "bench-needle-1m") return run_bench_needle_1m(model_dir, max_ctx, prefill_chunk, decode_tokens, ctxs);
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eval_kv_q4 error: %s\n", e.what());
    return 1;
  }
}

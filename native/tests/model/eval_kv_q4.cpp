// Part 1 (--kv-q4) end-to-end verification on real Inkling-2-Mini weights:
//   --mode kl     : exact full-vocab KL(BF16 || Q4_0) and KL(BF16 || FP8) over 256 decoded tokens after 4K prefill
//   --mode needle : 64K, 128K, and 245K needle-in-a-haystack recall (3 depths each = 9 needles) + VRAM & speed
//   --mode vram   : exact KV cache VRAM comparison (BF16 vs FP8 vs Q4_0) at 64K, 128K, 262K
#include "core/device.h"
#include "ingine/generator.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

}  // namespace

int main(int argc, char** argv) {
  std::string mode = "vram";
  std::string model_dir = "/mnt/storage/models/Inkling-2-Mini-ingine";
  std::string test_model_dir = "/mnt/storage/models/Inkling-2-Mini-atlas-exl3-9g";
  std::string golden = std::string(INGINE_GOLDEN_DIR) + "/model/abl_kv16_t5000.json";
  int max_ctx = 262144;
  int prefix_len = 4096;
  int eval_tokens = 256;
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
    else if (a == "--test-kv-q4") test_kv_q4 = true;
    else if (a == "--dense-bf16") dense_fp8 = false;
  }
  try {
    if (mode == "vram") return run_vram();
    if (mode == "kl") return run_kl(model_dir, golden, prefix_len, eval_tokens);
    if (mode == "kl-models") return run_kl_models(model_dir, test_model_dir, golden, prefix_len, eval_tokens, test_kv_q4, dense_fp8);
    if (mode == "needle") return run_needle(model_dir, max_ctx);
    std::fprintf(stderr, "unknown mode: %s\n", mode.c_str());
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eval_kv_q4 error: %s\n", e.what());
    return 1;
  }
}

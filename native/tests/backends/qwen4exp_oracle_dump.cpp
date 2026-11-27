// Standalone 48-layer CPU oracle dumper for Qwen3.8-Flash-Next (qwen4exp, #849).
// Runs exact streaming FP32 forward directly over real multi-shard GGUF files and dumps
// per-layer L2 norms, final result_norm, full FP32 logits, and top-20 tokens for direct
// numerical comparison against upstream llama.cpp-qwen38-flash-next-035e2273.
#include "backends/qwen4exp/qwen4exp_config.h"
#include "backends/qwen4exp/qwen4exp_cpu_ops.h"
#include "backends/qwen4exp/qwen4exp_gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::int32_t> parse_csv_tokens(const std::string& csv) {
  std::vector<std::int32_t> out;
  std::stringstream ss(csv);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(std::stoi(tok));
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path model_dir;
  std::vector<std::int32_t> tokens;
  std::filesystem::path out_logits;
  std::filesystem::path out_norm;
  std::filesystem::path out_json;
  int threads = 14;
  bool verify_pin = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model-dir" && i + 1 < argc) model_dir = argv[++i];
    else if (arg == "--tokens" && i + 1 < argc) tokens = parse_csv_tokens(argv[++i]);
    else if (arg == "--out-logits" && i + 1 < argc) out_logits = argv[++i];
    else if (arg == "--out-norm" && i + 1 < argc) out_norm = argv[++i];
    else if (arg == "--out-json" && i + 1 < argc) out_json = argv[++i];
    else if (arg == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
    else if (arg == "--verify-pin") verify_pin = true;
    else {
      std::fprintf(stderr, "unknown arg: %s\n", arg.c_str());
      return 2;
    }
  }
  if (model_dir.empty() || tokens.empty()) {
    std::fprintf(stderr, "usage: qwen4exp_oracle_dump --model-dir <dir> --tokens <csv> [--out-logits <bin>] [--out-norm <bin>] [--out-json <json>] [--threads N] [--verify-pin]\n");
    return 2;
  }

  using namespace ingine::backends::qwen4exp;
  const auto t0 = std::chrono::steady_clock::now();
  const auto gguf = verify_pin ? Qwen4ExpGgufSet::inspect_pinned(model_dir, qwen4exp_ud_q6_k_xl_pin_spec())
                               : Qwen4ExpGgufSet::inspect_headers(model_dir);
  const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  Qwen4ExpSessionCpu session(gguf.config());
  std::vector<double> last_layer_l2;
  std::vector<float> last_norm;
  std::vector<float> logits;

  const auto t1 = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const bool is_last = (i + 1 == tokens.size());
    const auto ts = std::chrono::steady_clock::now();
    logits = forward_gguf_token_logits_cpu(gguf, session, tokens[i], /*compute_logits=*/is_last,
                                           is_last ? &last_layer_l2 : nullptr,
                                           is_last ? &last_norm : nullptr, threads);
    const double step_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts).count();
    std::printf("step %zu/%zu tok=%d compute_logits=%s elapsed=%.2fs\n", i + 1, tokens.size(), tokens[i],
                is_last ? "true" : "false", step_s);
    std::fflush(stdout);
  }
  const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

  if (!out_logits.empty()) {
    std::ofstream f(out_logits, std::ios::binary);
    f.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
  }
  if (!out_norm.empty()) {
    std::ofstream f(out_norm, std::ios::binary);
    f.write(reinterpret_cast<const char*>(last_norm.data()),
            static_cast<std::streamsize>(last_norm.size() * sizeof(float)));
  }

  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  const std::size_t topk = std::min<std::size_t>(20, order.size());
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(topk), order.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });

  nlohmann::ordered_json top20 = nlohmann::ordered_json::array();
  for (std::size_t i = 0; i < topk; ++i) {
    top20.push_back({{"id", order[i]}, {"logit", logits[order[i]]}});
  }
  double norm_l2 = 0.0;
  for (float v : last_norm) norm_l2 += double(v) * double(v);
  norm_l2 = std::sqrt(norm_l2);

  nlohmann::ordered_json report = {
      {"model_dir", model_dir.string()},
      {"verify_pin", verify_pin},
      {"shards", gguf.shards().size()},
      {"tensors", gguf.tensors().size()},
      {"vocab_size", gguf.config().vocab_size},
      {"num_hidden_layers", gguf.config().num_hidden_layers},
      {"tokens", tokens},
      {"load_seconds", load_s},
      {"forward_seconds", total_s},
      {"result_norm_l2", norm_l2},
      {"layer_l2", last_layer_l2},
      {"top20", top20},
  };
  if (!out_json.empty()) {
    std::ofstream f(out_json, std::ios::binary);
    f << report.dump(2) << "\n";
  }
  std::printf("top5: ");
  for (std::size_t i = 0; i < std::min<std::size_t>(5, topk); ++i) {
    std::printf("%d(%.4f) ", order[i], logits[order[i]]);
  }
  std::printf("\nresult_norm_l2=%.6f total_forward=%.2fs\n", norm_l2, total_s);
  return 0;
}

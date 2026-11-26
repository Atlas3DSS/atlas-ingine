// Standalone GPU parity & speed benchmark for Qwen3.8-27B Elastic Quant packs (#849).
// Compares a baseline Qwen3.8-27B-NVFP4 checkpoint against one or more Elastic packs (atlas-14g, atlas-11g, atlas-17g)
// measuring exact VRAM residency, prefill tok/s, decode tok/s, teacher-forced KLD/cosine/top-1 vs baseline,
// and real greedy generation coherence.
#include "backends/qwen38/qwen38_dense_gpu.h"
#include "backends/qwen38/qwen38_gate2_metrics.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using namespace ingine::backends::qwen38;
using namespace ingine::model;
using Json = nlohmann::ordered_json;

struct EvalRun {
  std::string label;
  std::string path;
  std::string schema_variant;
  std::uint64_t weights_bytes = 0;
  std::uint64_t state_bytes = 0;
  std::uint64_t scratch_bytes = 0;
  std::uint64_t peak_vram_bytes = 0;
  double load_ms = 0.0;
  double prefill_tok_s = 0.0;
  double decode_tok_s = 0.0;
  std::vector<std::vector<float>> tf_logits;
  std::vector<std::int32_t> greedy_tokens;
  std::string greedy_text;
};

EvalRun run_single(const std::string& label, const std::filesystem::path& dir, const std::vector<std::int32_t>& tf_ids,
                   const std::vector<int>& tf_caps, const std::vector<std::int32_t>& gen_ids, int decode_tokens,
                   int max_ctx) {
  EvalRun out;
  out.label = label;
  out.path = dir.string();

  const auto fmt = ingine::text::ChatFormat::load(dir);

  Qwen38DenseLoadOptions opt;
  opt.model_dir = dir;
  opt.max_ctx = max_ctx;
  opt.prefill_chunk = 256;
  opt.verify_pins = false;
  opt.prefix_cache = true;
  opt.pre_suffix_checkpoint = true;
  opt.use_fast_attn = true;

  Qwen38DenseBackend backend(opt);
  const auto& ls = backend.load_stats();
  out.schema_variant = backend.state_identity().checkpoint.schema_variant;
  out.weights_bytes = ls.weights_device_bytes;
  out.state_bytes = ls.state_device_bytes;
  out.scratch_bytes = ls.scratch_device_bytes;
  out.peak_vram_bytes = ls.peak_vram_bytes;
  out.load_ms = ls.cold_safetensors_load_ms;

  auto* sess = dynamic_cast<Qwen38DenseSession*>(backend.acquire_session(0));
  if (!sess) std::exit(2);

  // 1. Teacher-forced logits capture on calibration/parity sequence.
  out.tf_logits.reserve(tf_caps.size());
  double tf_ms = 0.0;
  backend.evaluate_teacher_forced_mode(
      tf_ids, tf_caps, true,
      [&](const LogitStep& st) { out.tf_logits.emplace_back(st.logits.begin(), st.logits.end()); }, &tf_ms, sess, true);
  out.prefill_tok_s = tf_ms > 0.0 ? (static_cast<double>(tf_ids.size()) / (tf_ms / 1000.0)) : 0.0;

  // 2. Greedy generation speed + coherent answer.
  sess->reset();
  BackendRequest req;
  req.prompt.ids = gen_ids;
  req.use_prefix_cache = false;
  Sampling samp;
  samp.temperature = 0.0f;
  samp.top_k = 1;
  const std::array<std::int32_t, 2> stops = {248044, 248046};
  const auto st = backend.generate_request_on_session(
      *sess, req, decode_tokens, samp, stops,
      [&](std::span<const std::int32_t> toks) {
        out.greedy_tokens.insert(out.greedy_tokens.end(), toks.begin(), toks.end());
        return true;
      },
      {});
  out.decode_tok_s = st.decode_seconds > 0.0 ? (static_cast<double>(st.generated_tokens) / st.decode_seconds) : 0.0;
  out.greedy_text = fmt.tokenizer().decode(out.greedy_tokens);

  std::printf("[%s] schema=%s weights=%.2f GiB peak_vram=%.2f GiB load=%.1fs prefill=%.1f tok/s decode=%.1f tok/s\n",
              label.c_str(), out.schema_variant.c_str(), double(out.weights_bytes) / double(1ull << 30),
              double(out.peak_vram_bytes) / double(1ull << 30), out.load_ms / 1000.0, out.prefill_tok_s,
              out.decode_tok_s);
  std::printf("  greedy (%zu tok): %s\n", out.greedy_tokens.size(), out.greedy_text.c_str());
  return out;
}

double cosine_f32(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += double(a[i]) * double(b[i]);
    na += double(a[i]) * double(a[i]);
    nb += double(b[i]) * double(b[i]);
  }
  return dot / std::max(1e-12, std::sqrt(na) * std::sqrt(nb));
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  std::filesystem::path baseline_dir;
  std::vector<std::pair<std::string, std::filesystem::path>> targets;
  std::string out_json;
  int decode_tokens = 48;
  int max_ctx = 4096;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--baseline" && i + 1 < argc) {
      baseline_dir = argv[++i];
    } else if (a == "--target" && i + 2 < argc) {
      std::string lbl = argv[++i];
      std::filesystem::path p = argv[++i];
      targets.emplace_back(std::move(lbl), std::move(p));
    } else if (a == "--output" && i + 1 < argc) {
      out_json = argv[++i];
    } else if (a == "--decode-tokens" && i + 1 < argc) {
      decode_tokens = std::atoi(argv[++i]);
    } else if (a == "--max-ctx" && i + 1 < argc) {
      max_ctx = std::atoi(argv[++i]);
    }
  }
  if (baseline_dir.empty()) {
    std::fprintf(stderr, "Usage: %s --baseline <nvfp4_dir> --target <label> <dir> [--output file.json]\n", argv[0]);
    return 2;
  }

  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
    std::printf("skip: no CUDA device visible\n");
    return 77;
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);

  const auto fmt = ingine::text::ChatFormat::load(baseline_dir);
  ingine::text::ChatOptions copts;
  copts.enable_thinking = false;

  // 1. 256+ token technical passage for teacher-forced KLD/cosine parity.
  const std::string tf_text =
      "In modern GPU inference engines, mixed-precision quantization balances memory bandwidth against "
      "numerical fidelity across transformer and state-space layers. Boundary layers near the embedding and "
      "output projection carry higher sensitivity to weight quantization noise, while core MLP blocks tolerate "
      "non-uniform Lloyd-Max codebooks such as 3-bit and 2-bit symmetric centroids paired with per-16-element "
      "FP8 E4M3 block scales. When every parameter fits inside local GPU VRAM without host-to-device paging, "
      "single-token autoregressive latency becomes strictly bounded by GPU memory bandwidth rather than PCIe "
      "transfer stalls. Explain concisely why boundary-layer precision protection preserves output distribution.";
  const auto tf_prompt = fmt.render(Json::array({{{"role", "user"}, {"content", tf_text}}}), Json::array(), copts);
  const auto& tf_ids = tf_prompt.ids;
  const int n_tf = static_cast<int>(tf_ids.size());
  const int n_cap = std::min(32, n_tf);
  std::vector<int> tf_caps;
  for (int i = n_tf - n_cap; i < n_tf; ++i) tf_caps.push_back(i);

  // 2. Clean factual + reasoning prompt for greedy generation verification.
  const auto gen_prompt = fmt.render(
      Json::array({{{"role", "user"},
                    {"content", "Answer in two concise sentences: What is the capital of France, and why is "
                                "keeping model weights completely inside GPU VRAM faster than PCIe offloading?"}}}),
      Json::array(), copts);

  Json report;
  report["gpu_name"] = prop.name;
  report["sm"] = std::to_string(prop.major) + "." + std::to_string(prop.minor);
  report["tf_prompt_tokens"] = tf_ids.size();
  report["tf_captured_positions"] = tf_caps.size();

  const EvalRun base = run_single("baseline_nvfp4", baseline_dir, tf_ids, tf_caps, gen_prompt.ids, decode_tokens, max_ctx);
  report["baseline"] = {
      {"label", base.label},
      {"path", base.path},
      {"schema_variant", base.schema_variant},
      {"weights_gib", double(base.weights_bytes) / double(1ull << 30)},
      {"peak_vram_gib", double(base.peak_vram_bytes) / double(1ull << 30)},
      {"load_ms", base.load_ms},
      {"prefill_tok_s", base.prefill_tok_s},
      {"decode_tok_s", base.decode_tok_s},
      {"greedy_text", base.greedy_text},
  };

  Json t_arr = Json::array();
  for (const auto& [lbl, dir] : targets) {
    const EvalRun cur = run_single(lbl, dir, tf_ids, tf_caps, gen_prompt.ids, decode_tokens, max_ctx);
    std::vector<Qwen38LogitParityPoint> pts;
    double cos_sum = 0.0, cos_min = 1.0;
    for (std::size_t i = 0; i < tf_caps.size(); ++i) {
      pts.push_back(compute_logit_kld_and_top1(base.tf_logits[i], cur.tf_logits[i], tf_caps[i]));
      const double c = cosine_f32(base.tf_logits[i], cur.tf_logits[i]);
      cos_sum += c;
      cos_min = std::min(cos_min, c);
    }
    const auto ks = summarize_kld_metrics(pts);
    const double cos_mean = tf_caps.empty() ? 0.0 : cos_sum / double(tf_caps.size());
    std::size_t tok_match = 0;
    const std::size_t tok_min = std::min(base.greedy_tokens.size(), cur.greedy_tokens.size());
    for (std::size_t i = 0; i < tok_min; ++i) {
      if (base.greedy_tokens[i] == cur.greedy_tokens[i]) ++tok_match;
    }
    std::printf("  [%s vs baseline] kld_mean=%.4f kld_p99=%.4f cos_mean=%.6f cos_min=%.6f top1=%.3f (%zu/%zu) "
                "greedy_match=%zu/%zu vram_saved=%.2f GiB\n",
                lbl.c_str(), ks.kld_mean, ks.kld_p99, cos_mean, cos_min, ks.top1_agreement, ks.top1_matches,
                ks.positions, tok_match, tok_min,
                (double(base.weights_bytes) - double(cur.weights_bytes)) / double(1ull << 30));
    t_arr.push_back({
        {"label", cur.label},
        {"path", cur.path},
        {"schema_variant", cur.schema_variant},
        {"weights_gib", double(cur.weights_bytes) / double(1ull << 30)},
        {"peak_vram_gib", double(cur.peak_vram_bytes) / double(1ull << 30)},
        {"vram_saved_gib", (double(base.weights_bytes) - double(cur.weights_bytes)) / double(1ull << 30)},
        {"load_ms", cur.load_ms},
        {"prefill_tok_s", cur.prefill_tok_s},
        {"decode_tok_s", cur.decode_tok_s},
        {"kld_mean", ks.kld_mean},
        {"kld_p99", ks.kld_p99},
        {"kld_max", ks.kld_max},
        {"cosine_mean", cos_mean},
        {"cosine_min", cos_min},
        {"top1_agreement", ks.top1_agreement},
        {"greedy_token_matches", tok_match},
        {"greedy_token_total", tok_min},
        {"greedy_text", cur.greedy_text},
        {"all_finite", ks.all_finite},
    });
  }
  report["targets"] = t_arr;

  if (!out_json.empty()) {
    std::ofstream f(out_json);
    f << report.dump(2) << "\n";
  }
  std::printf("qwen38_elastic_eval: DONE\n");
  return 0;
}

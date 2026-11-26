// Live GPU end-to-end warm-start verification across real Inkling B=1, Inkling B>1, and pinned Qwen3.8 (#887).
// Skips cleanly with exit code 77 when no CUDA GPU is visible (e.g. CUDA_VISIBLE_DEVICES="").
//
// For each selected real backend arm, verifies via live ChatServer HTTP:
//   1. Cold baseline (unset warm-start): /health warm_start=false, warm_start_tokens=0, Turn-1 cached_tokens=0.
//   2. Warm-started server #1 (matching namespace): /health warm_start=true, warm_start_tokens==W > 0,
//      first matching request reports cached_tokens == W and produces bit-identical raw greedy token IDs vs cold.
//   3. Fresh warm-started server #2 (mismatched namespace): /health warm_start=true, warm_start_tokens==W,
//      first mismatched request reports cached_tokens == 0 and produces bit-identical raw greedy token IDs vs cold.
#include "ingine/generator.h"
#include "serve/chat_server.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ingine::model::BatchSlotResult;
using ingine::model::BatchSlotStep;
using ingine::model::GenerateStats;
using ingine::model::Generator;
using ingine::model::KvCacheMode;
using ingine::model::LoopCdMode;
using ingine::model::PrefillCallback;
using ingine::model::PrefixCacheStats;
using ingine::model::Prompt;
using ingine::model::RuntimeOptions;
using ingine::model::Sampling;
using ingine::serve::ChatServer;
using ingine::serve::ServerOptions;
using ingine::text::ChatFormat;

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

// Transparent proxy forwarding every Generator method to the real GPU backend while capturing the exact
// raw greedy output token IDs emitted by B=1 generate() or B>1 start_slot()/step_slots().
class RecordingGenerator final : public Generator {
 public:
  explicit RecordingGenerator(std::unique_ptr<Generator> inner) : inner_(std::move(inner)) {}

  std::string model_id() const override { return inner_->model_id(); }
  int max_model_len() const override { return inner_->max_model_len(); }
  bool supports_images() const override { return inner_->supports_images(); }
  bool supports_logit_masker() const override { return inner_->supports_logit_masker(); }
  std::string model_family() const override { return inner_->model_family(); }
  KvCacheMode kv_cache_mode() const override { return inner_->kv_cache_mode(); }
  std::vector<KvCacheMode> supported_kv_cache_modes() const override { return inner_->supported_kv_cache_modes(); }
  int loop_passes() const override { return inner_->loop_passes(); }
  LoopCdMode loopcd_mode() const override { return inner_->loopcd_mode(); }
  std::string offload_summary() const override { return inner_->offload_summary(); }
  PrefixCacheStats cache_stats() const override { return inner_->cache_stats(); }
  int max_batch_size() const override { return inner_->max_batch_size(); }
  bool can_admit_slot(int p, int m) const override { return inner_->can_admit_slot(p, m); }
  bool can_admit_slot(const Prompt& p, int m, const Sampling& s) const override {
    return inner_->can_admit_slot(p, m, s);
  }
  bool supports_batched_spec() const override { return inner_->supports_batched_spec(); }

  GenerateStats warm_start(const Prompt& prompt, const Sampling& sampling) override {
    const GenerateStats st = inner_->warm_start(prompt, sampling);
    std::lock_guard lock(mu_);
    last_warm_start_stats_ = st;
    return st;
  }

  GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const int32_t> stop_ids,
                         const std::function<bool(std::span<const int32_t>)>& on_tokens) override {
    clear_recorded();
    auto wrap = [&](std::span<const int32_t> toks) {
      record(toks);
      return on_tokens(toks);
    };
    return inner_->generate(prompt, max_new_tokens, sampling, stop_ids, wrap);
  }

  GenerateStats generate(const Prompt& prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const int32_t> stop_ids,
                         const std::function<bool(std::span<const int32_t>)>& on_tokens) override {
    clear_recorded();
    auto wrap = [&](std::span<const int32_t> toks) {
      record(toks);
      return on_tokens(toks);
    };
    return inner_->generate(prompt, max_new_tokens, sampling, stop_ids, wrap);
  }

  GenerateStats generate(const Prompt& prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const int32_t> stop_ids,
                         const std::function<bool(std::span<const int32_t>)>& on_tokens,
                         const PrefillCallback& on_prefill) override {
    clear_recorded();
    auto wrap = [&](std::span<const int32_t> toks) {
      record(toks);
      return on_tokens(toks);
    };
    return inner_->generate(prompt, max_new_tokens, sampling, stop_ids, wrap, on_prefill);
  }

  int start_slot(const Prompt& prompt, int max_new_tokens, const Sampling& sampling, int32_t& first_token,
                 GenerateStats& stats) override {
    const int s = inner_->start_slot(prompt, max_new_tokens, sampling, first_token, stats);
    if (s >= 0) {
      clear_recorded();
      record(std::span<const int32_t>(&first_token, 1));
    }
    return s;
  }

  void step_slots(std::span<const BatchSlotStep> inputs, std::span<int32_t> out_tokens) override {
    inner_->step_slots(inputs, out_tokens);
    if (!inputs.empty()) record(std::span<const int32_t>(out_tokens.data(), inputs.size()));
  }

  void step_slots_multi(std::span<const BatchSlotStep> inputs, std::span<BatchSlotResult> out) override {
    inner_->step_slots_multi(inputs, out);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      record(std::span<const int32_t>(out[i].tokens, static_cast<std::size_t>(out[i].count)));
    }
  }

  void release_slot(int slot) override { inner_->release_slot(slot); }

  std::vector<int32_t> last_emitted_tokens() const {
    std::lock_guard lock(mu_);
    return emitted_;
  }

  GenerateStats last_warm_start_stats() const {
    std::lock_guard lock(mu_);
    return last_warm_start_stats_;
  }

 private:
  void clear_recorded() {
    std::lock_guard lock(mu_);
    emitted_.clear();
  }
  void record(std::span<const int32_t> toks) {
    std::lock_guard lock(mu_);
    emitted_.insert(emitted_.end(), toks.begin(), toks.end());
  }

  std::unique_ptr<Generator> inner_;
  mutable std::mutex mu_;
  std::vector<int32_t> emitted_;
  GenerateStats last_warm_start_stats_{};
};

struct RunCapture {
  Json health;
  Json response;
  int warm_start_reused_tokens = 0;
  int cached_tokens = -1;
  int prompt_tokens = -1;
  int completion_tokens = -1;
  std::vector<int32_t> token_ids;
  double http_ms = 0.0;
};

RunCapture run_server_on_generator(RecordingGenerator& gen, const ChatFormat& fmt, const std::string& warm_prompt,
                                   const std::string& warm_ns, const std::string& req_ns, const Json& req_body) {
  ServerOptions so;
  so.images.enabled = false;
  so.cache_namespace_header = "X-Ingine-Cache-Namespace";
  so.warm_start_prompt = warm_prompt;
  so.warm_start_namespace = warm_ns;
  ChatServer srv(gen, fmt, so);
  CHECK(srv.bind("127.0.0.1", 0));
  srv.start();

  httplib::Client cli("127.0.0.1", srv.port());
  cli.set_read_timeout(120, 0);

  const auto rh = cli.Get("/health");
  CHECK(rh && rh->status == 200);
  RunCapture cap;
  cap.health = Json::parse(rh->body);
  if (!warm_prompt.empty()) {
    cap.warm_start_reused_tokens = gen.last_warm_start_stats().reused_tokens;
  }

  const auto t0 = std::chrono::steady_clock::now();
  const auto rc = cli.Post("/v1/chat/completions", httplib::Headers{{"X-Ingine-Cache-Namespace", req_ns}},
                           req_body.dump(), "application/json");
  cap.http_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  CHECK(rc && rc->status == 200);
  cap.response = Json::parse(rc->body);
  cap.prompt_tokens = cap.response["usage"]["prompt_tokens"].get<int>();
  cap.completion_tokens = cap.response["usage"]["completion_tokens"].get<int>();
  cap.cached_tokens = cap.response["usage"]["prompt_tokens_details"]["cached_tokens"].get<int>();
  cap.token_ids = gen.last_emitted_tokens();

  srv.stop();
  return cap;
}

Json run_arm(const std::string& arm_name, const RuntimeOptions& rt_opts, const std::string& sys_prompt,
             bool disable_thinking, int max_tokens) {
  const ChatFormat fmt = ChatFormat::load(rt_opts.model_dir);
  const int expected_ws_tokens = int(fmt.render_system_prefix(sys_prompt).ids.size());
  CHECK(expected_ws_tokens > 0);

  Json body = {
      {"messages",
       Json::array({{{"role", "system"}, {"content", sys_prompt}},
                    {{"role", "user"},
                     {"content", "Give a concise two-sentence explanation of why prefix caching reduces time to first token."}}})},
      {"temperature", 0.0},
      {"top_k", 1},
      {"seed", 42},
      {"max_tokens", max_tokens},
  };
  if (disable_thinking) {
    body["chat_template_kwargs"] = {{"enable_thinking", false}};
  }

  std::printf("=== ARM %s (model=%s, max_batch=%d, expected_ws_tokens=%d) ===\n", arm_name.c_str(),
              rt_opts.model_dir.c_str(), rt_opts.max_batch, expected_ws_tokens);
  std::fflush(stdout);

  // Load immutable GPU weights once per arm, then construct three separate ChatServer lifecycles with
  // mutually disjoint namespaces (cold -> ns-cold, warm_hit -> ns-warm-1, fresh_mismatch -> ns-warm-2 vs ns-mismatch).
  auto raw_gen = ingine::model::make_generator(rt_opts);
  RecordingGenerator gen(std::move(raw_gen));

  // 1. Cold unset baseline
  const RunCapture cold = run_server_on_generator(gen, fmt, "", "", "ns-cold", body);
  cudaDeviceSynchronize();
  CHECK(cold.health["status"] == "ok");
  CHECK(cold.health["warm_start"] == false);
  CHECK(cold.health["warm_start_tokens"] == 0);
  CHECK(cold.cached_tokens == 0);
  CHECK(!cold.token_ids.empty());

  // 2. Warm-started server #1: disjoint namespace ns-warm-1 -> warm_start prefills fresh (reused=0),
  //    first matching request reports cached_tokens == warm_start_tokens and matches cold greedy token IDs.
  const RunCapture warm_hit = run_server_on_generator(gen, fmt, sys_prompt, "ns-warm-1", "ns-warm-1", body);
  cudaDeviceSynchronize();
  CHECK(warm_hit.health["status"] == "ok");
  CHECK(warm_hit.health["warm_start"] == true);
  CHECK(warm_hit.health["warm_start_tokens"] == expected_ws_tokens);
  CHECK(warm_hit.warm_start_reused_tokens == 0);
  CHECK(warm_hit.cached_tokens == expected_ws_tokens);
  CHECK(warm_hit.token_ids == cold.token_ids);
  CHECK(warm_hit.response["choices"][0]["message"] == cold.response["choices"][0]["message"]);

  // 3. Fresh warm-started server #2: brand-new namespace ns-warm-2 -> warm_start prefills fresh (reused=0),
  //    first request in unseen mismatched namespace ns-mismatch reports cached_tokens == 0 and matches cold greedy IDs.
  const RunCapture warm_miss = run_server_on_generator(gen, fmt, sys_prompt, "ns-warm-2", "ns-mismatch", body);
  cudaDeviceSynchronize();
  CHECK(warm_miss.health["status"] == "ok");
  CHECK(warm_miss.health["warm_start"] == true);
  CHECK(warm_miss.health["warm_start_tokens"] == expected_ws_tokens);
  CHECK(warm_miss.warm_start_reused_tokens == 0);
  CHECK(warm_miss.cached_tokens == 0);
  CHECK(warm_miss.token_ids == cold.token_ids);
  CHECK(warm_miss.response["choices"][0]["message"] == cold.response["choices"][0]["message"]);

  std::printf(
      "PASS %s: ws_tokens=%d cold_cached=%d warm_hit_cached=%d fresh_mismatch_cached=%d greedy_ids=%zu/%zu exact_match=true "
      "cold_ms=%.1f warm_hit_ms=%.1f\n",
      arm_name.c_str(), expected_ws_tokens, cold.cached_tokens, warm_hit.cached_tokens, warm_miss.cached_tokens,
      warm_hit.token_ids.size(), cold.token_ids.size(), cold.http_ms, warm_hit.http_ms);
  std::fflush(stdout);

  return Json{
      {"arm", arm_name},
      {"model_dir", rt_opts.model_dir.string()},
      {"max_batch", rt_opts.max_batch},
      {"expected_warm_start_tokens", expected_ws_tokens},
      {"cold",
       {{"health", cold.health},
        {"prompt_tokens", cold.prompt_tokens},
        {"cached_tokens", cold.cached_tokens},
        {"completion_tokens", cold.completion_tokens},
        {"http_ms", cold.http_ms},
        {"token_ids", cold.token_ids},
        {"message", cold.response["choices"][0]["message"]}}},
      {"warm_hit",
       {{"health", warm_hit.health},
        {"warm_start_reused_tokens", warm_hit.warm_start_reused_tokens},
        {"prompt_tokens", warm_hit.prompt_tokens},
        {"cached_tokens", warm_hit.cached_tokens},
        {"completion_tokens", warm_hit.completion_tokens},
        {"http_ms", warm_hit.http_ms},
        {"token_ids", warm_hit.token_ids},
        {"exact_token_id_match_vs_cold", warm_hit.token_ids == cold.token_ids},
        {"message", warm_hit.response["choices"][0]["message"]}}},
      {"fresh_namespace_mismatch",
       {{"health", warm_miss.health},
        {"warm_start_reused_tokens", warm_miss.warm_start_reused_tokens},
        {"prompt_tokens", warm_miss.prompt_tokens},
        {"cached_tokens", warm_miss.cached_tokens},
        {"completion_tokens", warm_miss.completion_tokens},
        {"http_ms", warm_miss.http_ms},
        {"token_ids", warm_miss.token_ids},
        {"exact_token_id_match_vs_cold", warm_miss.token_ids == cold.token_ids}}},
  };
}

}  // namespace

int main(int argc, char** argv) {
  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
    std::printf("skip: no CUDA device available\n");
    return 77;
  }

  std::string arm = "all";
  const char* env_ink = std::getenv("INGINE_MODEL_DIR");
  const char* env_qwen = std::getenv("INGINE_QWEN38_MODEL_DIR");
  std::filesystem::path ink_dir =
      (env_ink && *env_ink) ? std::filesystem::path(env_ink) : std::filesystem::path("/mnt/storage/models/Inkling-2-Mini-ingine");
  std::filesystem::path qwen_dir = (env_qwen && *env_qwen)
                                       ? std::filesystem::path(env_qwen)
                                       : std::filesystem::path("/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f");
  int max_tokens = 24;
  std::string out_path;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--arm" && i + 1 < argc) arm = argv[++i];
    else if (arg == "--inkling-dir" && i + 1 < argc) ink_dir = argv[++i];
    else if (arg == "--qwen38-dir" && i + 1 < argc) qwen_dir = argv[++i];
    else if (arg == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
    else if (arg == "--out" && i + 1 < argc) out_path = argv[++i];
    else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      return 2;
    }
  }

  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
  std::printf("CUDA device 0: %s (sm_%d%d, %.2f GiB)\n", prop.name, prop.major, prop.minor,
              double(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));

  const std::string sys_prompt =
      "You are an autonomous C++ and CUDA systems engineer maintaining atlas-ingine. "
      "Follow every invariant strictly: verify all boundary conditions, preserve exact prefix cache isolation across "
      "tenant namespaces, never fabricate measurements, report exact token counts, and keep every response concise, "
      "deterministic, and factually grounded.";

  Json report = {
      {"gpu", prop.name},
      {"sm", prop.major * 10 + prop.minor},
      {"arms", Json::array()},
  };

  int ran = 0;
  if (arm == "all" || arm == "inkling-b1") {
    if (std::filesystem::is_directory(ink_dir)) {
      RuntimeOptions opts;
      opts.model_dir = ink_dir;
      opts.max_ctx = 4096;
      opts.prefill_chunk = 1024;
      opts.dense_fp8 = true;
      opts.kv_fp8 = true;
      opts.vision = false;
      opts.prefix_cache = true;
      opts.park_mb = 256;
      opts.max_batch = 1;
      report["arms"].push_back(run_arm("inkling-b1", opts, sys_prompt, /*disable_thinking=*/true, max_tokens));
      ++ran;
    } else if (arm == "inkling-b1") {
      std::fprintf(stderr, "inkling model dir missing: %s\n", ink_dir.c_str());
      return 1;
    }
  }

  if (arm == "all" || arm == "inkling-b2") {
    if (std::filesystem::is_directory(ink_dir)) {
      RuntimeOptions opts;
      opts.model_dir = ink_dir;
      opts.max_ctx = 4096;
      opts.max_kv_tokens = 4096;
      opts.prefill_chunk = 1024;
      opts.dense_fp8 = true;
      opts.kv_fp8 = true;
      opts.vision = false;
      opts.prefix_cache = true;
      opts.park_mb = 256;
      opts.max_batch = 2;
      report["arms"].push_back(run_arm("inkling-b2", opts, sys_prompt, /*disable_thinking=*/true, max_tokens));
      ++ran;
    } else if (arm == "inkling-b2") {
      std::fprintf(stderr, "inkling model dir missing: %s\n", ink_dir.c_str());
      return 1;
    }
  }

  if (arm == "all" || arm == "qwen38") {
    if (std::filesystem::is_directory(qwen_dir) && prop.major >= 8) {
      RuntimeOptions opts;
      opts.model_dir = qwen_dir;
      opts.max_ctx = 2048;
      opts.prefill_chunk = 256;
      opts.dense_fp8 = false;  // normalized to calibrated FP8 + PinnedVerified by resolve_qwen38_runtime_options
      opts.vision = false;
      opts.prefix_cache = true;
      opts.park_mb = 0;
      opts.max_batch = 1;
      report["arms"].push_back(run_arm("qwen38-pinned", opts, sys_prompt, /*disable_thinking=*/false, max_tokens));
      ++ran;
    } else if (arm == "qwen38") {
      std::fprintf(stderr, "qwen38 model dir missing or GPU < sm_80: %s\n", qwen_dir.c_str());
      return 1;
    }
  }

  if (ran == 0) {
    std::printf("skip: no compatible model directories found for arm=%s\n", arm.c_str());
    return 77;
  }

  if (!out_path.empty()) {
    std::ofstream out(out_path, std::ios::binary);
    CHECK(out.good());
    out << report.dump(2) << "\n";
  }
  std::printf("serve_gpu_warm_start: PASS (%d arms)\n", ran);
  return 0;
}

// Canned-output generator (mirror of engine/ingine/server.py FakeGenerator) for tests and `ingine-serve --fake`.
#pragma once

#include "ingine/generator.h"

#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace ingine::text {
class Tokenizer;
}

namespace ingine::engine {

// Replays canned token ids, honouring max_new_tokens, stop_ids and cancellation. responder(prompt_ids) returns
// the full output as token ids or as text (encoded with added tokens recognised); the default echoes the last
// user message as thinking + answer, or as a tool call when it reads `call NAME {json args}`. calls() records
// every request and how it ended.
class FakeGenerator final : public model::Generator {
 public:
  using Output = std::variant<std::vector<int32_t>, std::string>;
  using Responder = std::function<Output(std::span<const int32_t>)>;
  // Optional test hook invoked after prompt prefill (before emitting token 0) to populate reuse stats.
  using PrefillStatsFn = std::function<void(std::span<const int32_t>, model::GenerateStats&)>;

  struct Call {
    std::vector<int32_t> prompt_ids;
    int max_new_tokens = 0;
    model::Sampling sampling;
    int images = 0, image_tokens = 0;  // of the prompt
    int root_prefix_tokens = 0;
    std::string cache_namespace;
    bool bypass_prefix_cache = false;
    int yielded = 0;
    bool cancelled = false;
    bool done = false;
  };

  explicit FakeGenerator(const text::Tokenizer& tokenizer, std::string model_id = "Inkling-2-Mini-fake",
                         int max_model_len = 1 << 20, Responder responder = {}, double token_delay = 0.0,
                         int loop_passes = 3, model::KvCacheMode kv_cache_mode = model::KvCacheMode::PerPass,
                         bool allow_last_full = true);

  std::string model_id() const override { return model_id_; }
  int max_model_len() const override { return max_model_len_; }
  int loop_passes() const override { return loop_passes_; }
  model::GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens,
                                const model::Sampling& sampling, std::span<const int32_t> stop_ids,
                                const std::function<bool(std::span<const int32_t>)>& on_tokens) override {
    return generate(model::Prompt{prompt, {}}, max_new_tokens, sampling, stop_ids, on_tokens, model::PrefillCallback{});
  }
  // Images are recorded, not looked at (their tokens are negative ids, which echo() skips).
  model::GenerateStats generate(const model::Prompt& prompt, int max_new_tokens, const model::Sampling& sampling,
                                std::span<const int32_t> stop_ids,
                                const std::function<bool(std::span<const int32_t>)>& on_tokens) override {
    return generate(prompt, max_new_tokens, sampling, stop_ids, on_tokens, model::PrefillCallback{});
  }
  model::GenerateStats generate(const model::Prompt& prompt, int max_new_tokens, const model::Sampling& sampling,
                                std::span<const int32_t> stop_ids,
                                const std::function<bool(std::span<const int32_t>)>& on_tokens,
                                const model::PrefillCallback& on_prefill) override;
  bool supports_images() const override { return true; }
  model::KvCacheMode kv_cache_mode() const override { return kv_cache_mode_; }
  std::vector<model::KvCacheMode> supported_kv_cache_modes() const override { return supported_kv_cache_modes_; }
  void set_kv_cache_mode(model::KvCacheMode mode);
  void set_supported_kv_cache_modes(std::vector<model::KvCacheMode> modes) {
    supported_kv_cache_modes_ = std::move(modes);
  }

  void set_prefill_stats_hook(PrefillStatsFn hook);
  std::string echo(std::span<const int32_t> prompt_ids) const;
  std::vector<Call> calls() const;

 private:
  const text::Tokenizer& tokenizer_;
  std::string model_id_;
  int max_model_len_;
  int loop_passes_;
  Responder responder_;
  double token_delay_;
  model::KvCacheMode kv_cache_mode_ = model::KvCacheMode::PerPass;
  std::vector<model::KvCacheMode> supported_kv_cache_modes_ = {model::KvCacheMode::PerPass,
                                                                model::KvCacheMode::LastFull};
  mutable std::mutex mutex_;
  PrefillStatsFn prefill_stats_hook_;
  std::vector<Call> calls_;
};

}  // namespace ingine::engine

// The Generator (engine.Engine): one resident model, one request at a time. Prefill runs in chunks through
// the Prefiller; plain decoding replays the width-1 graph per token; with options.spec the speculative loop
// (generate_spec) drafts with DSpark and verifies with the width-2/4/8 graphs. A prompt that extends the
// tokens already in the state only prefills the new suffix; any other prompt starts from an empty state.
#include "ingine/generator.h"

#include "core/device.h"
#include "kernels/head/head.h"
#include "model/drafter.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "model/spec.h"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace {

namespace head = kernels::head;

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

class EngineGenerator final : public Generator {
 public:
  explicit EngineGenerator(const RuntimeOptions& options)
      : options_(options),
        rt_(options),
        prefiller_(rt_.weights(), rt_.ablation(), options.prefill_chunk, rt_.stream()),
        chunk_tokens_(std::size_t(options.prefill_chunk) * sizeof(std::int32_t)),
        logits_(std::size_t(UNPADDED_VOCAB) * sizeof(float)),
        token_(sizeof(std::int32_t)),
        host_token_(sizeof(std::int32_t)) {
    if (options.prefill_chunk < 1) throw std::invalid_argument("prefill_chunk must be positive");
    if (options.spec) {
      drafter_ = std::make_unique<Drafter>(options.model_dir, rt_.weights(), options.max_ctx, rt_.stream());
      drafter_->capture();
      taps_ = DeviceBuffer(std::size_t(options.prefill_chunk) * NUM_TAPS * HIDDEN * 2);
    }
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
  }

  std::string model_id() const override {
    return options_.ablation.empty() ? "inkling-2-mini" : "inkling-2-mini-abliterated";
  }

  int max_model_len() const override { return options_.max_ctx; }

  GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    if (prompt.empty()) throw std::invalid_argument("prompt is empty");
    const int n = int(prompt.size());
    // a verify call writes up to max(verify widths) positions
    const int reserve = drafter_ ? *std::max_element(rt_.verify_widths().begin(), rt_.verify_widths().end()) : 0;
    if (n + std::int64_t(max_new_tokens) + reserve > options_.max_ctx) {
      max_new_tokens = options_.max_ctx - n - reserve;
      if (max_new_tokens <= 0)
        throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                    std::to_string(options_.max_ctx));
    }
    GenerateStats stats;
    head::SamplerRng rng;
    rng.seed = sampling.seed ? *sampling.seed : (std::uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
    std::lock_guard<std::mutex> lock(mutex_);

    auto t = std::chrono::steady_clock::now();
    const int reused = prefill(prompt);
    stats.prompt_tokens = n;
    stats.reused_tokens = reused;
    std::int32_t token = sample(static_cast<const float*>(logits_.p), sampling, rng);
    stats.prefill_seconds = seconds_since(t);

    t = std::chrono::steady_clock::now();
    auto is_stop = [&](std::int32_t tok) { return std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end(); };
    if (max_new_tokens > 0) {
      if (drafter_) {
        drafter_->set_seed(rng.seed);
        generate_spec(rt_, *drafter_, token, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats,
                      tokens_in_state_);
      } else {
        int pos = n;
        for (int i = 0; i < max_new_tokens; ++i) {
          stats.generated_tokens = i + 1;
          const std::int32_t out[1] = {token};
          const bool go_on = on_tokens(out);
          if (is_stop(token) || i + 1 == max_new_tokens || !go_on) break;
          const float* logits = rt_.decode_logits(token, pos);
          tokens_in_state_.push_back(token);
          ++pos;
          token = sample(logits, sampling, rng);
        }
      }
    }
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    stats.decode_seconds = seconds_since(t);
    return stats;
  }

 private:
  // engine._prefill: brings the state to `ids` (reusing a matching prefix); last logits in logits_.
  // Returns the number of reused tokens.
  int prefill(std::span<const std::int32_t> ids) {
    const std::vector<std::int32_t>& prev = tokens_in_state_;
    bool reusable = !prev.empty() && prev.size() < ids.size() && std::equal(prev.begin(), prev.end(), ids.begin());
    if (drafter_ && drafter_->length() != int(prev.size()))
      reusable = false;  // the drafter missed rows (a non-speculative request ran in between)
    State& st = rt_.state();
    int start;
    if (reusable) {
      start = int(prev.size());
      st.length = start;  // decode steps advance the state without its host-side length
    } else {
      reset_state(st, rt_.stream());
      if (drafter_) drafter_->reset();
      tokens_in_state_.clear();
      start = 0;
    }
    const int chunk = options_.prefill_chunk;
    for (int s = start; s < int(ids.size()); s += chunk) {
      const int T = std::min(chunk, int(ids.size()) - s);
      CUDA_CHECK(cudaMemcpyAsync(chunk_tokens_.p, ids.data() + s, std::size_t(T) * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, rt_.stream()));
      Tensor taps;
      if (drafter_) taps = rm(taps_.p, DType::BF16, {T, NUM_TAPS, HIDDEN});
      prefiller_.run(st, static_cast<const std::int32_t*>(chunk_tokens_.p), T, s, static_cast<float*>(logits_.p),
                     drafter_ ? &taps : nullptr);
      if (drafter_) drafter_->commit(taps, s);
    }
    tokens_in_state_.assign(ids.begin(), ids.end());
    return start;
  }

  // engine._sample: argmax at temperature <= 0, else one draw from engine._probs.
  std::int32_t sample(const float* logits, const Sampling& sampling, head::SamplerRng& rng) {
    head::SamplingParams p;
    p.temperature = sampling.temperature;
    p.top_k = sampling.top_k;
    p.top_p = sampling.top_p;
    head::sample(logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, p, rng, static_cast<std::int32_t*>(token_.p),
                 rt_.stream());
    CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), token_.p, sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               rt_.stream()));
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    return *static_cast<const std::int32_t*>(host_token_.data());
  }

  RuntimeOptions options_;
  Runtime rt_;
  Prefiller prefiller_;
  std::unique_ptr<Drafter> drafter_;
  DeviceBuffer chunk_tokens_, logits_, token_, taps_;
  PinnedHostBuffer host_token_;
  std::vector<std::int32_t> tokens_in_state_;
  std::mutex mutex_;
};

}  // namespace

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options) {
  return std::make_unique<EngineGenerator>(options);
}

}  // namespace ingine::model

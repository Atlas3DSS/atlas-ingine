// The token generator interface the server drives (mirror of engine.Engine.generate). CUDA-free so the
// scheduler, the HTTP layer and their tests build without the model; ingine_model implements make_generator.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace ingine::model {

struct Sampling {
  float temperature = 1.0f;  // 0 => greedy
  float top_p = 1.0f;
  int top_k = 0;  // 0 => disabled
  std::optional<uint64_t> seed;
};

struct GenerateStats {
  int prompt_tokens = 0, reused_tokens = 0, generated_tokens = 0;
  double prefill_seconds = 0, decode_seconds = 0;
  int spec_rounds = 0, spec_accepted = 0;
};

struct RuntimeOptions {
  std::filesystem::path model_dir, ablation;  // empty ablation: original model
  int max_ctx = 65536;
  int prefill_chunk = 2048;
  bool dense_fp8 = true, kv_fp8 = false, spec = false;
  float spec_threshold = 0.8f;
};

class Generator {
 public:
  virtual ~Generator() = default;
  virtual std::string model_id() const = 0;  // "inkling-2-mini-abliterated" with an ablation, else "inkling-2-mini"
  virtual int max_model_len() const = 0;     // = max_ctx
  // One request at a time. Emits sampled tokens in order through on_tokens (a speculative round emits several);
  // stops after a token in stop_ids, after max_new_tokens, or when on_tokens returns false (cancel).
  // Throws std::invalid_argument when the prompt does not fit max_ctx (reserve as engine.py does).
  virtual GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) = 0;
};

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options);  // the model (ingine_model)

}  // namespace ingine::model

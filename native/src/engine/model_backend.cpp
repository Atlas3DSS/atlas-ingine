#include "ingine/model_backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace ingine::model {
namespace {

std::uint64_t fnv1a_mix(std::uint64_t h, const void* data, std::size_t bytes) noexcept {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    h ^= std::uint64_t(p[i]);
    h *= 1099511628211ull;
  }
  return h;
}

void fnv1a_mix_string(std::uint64_t& h, std::string_view s) noexcept {
  const std::size_t len = s.size();
  h = fnv1a_mix(h, &len, sizeof(len));
  if (len > 0) h = fnv1a_mix(h, s.data(), len);
}

std::string to_lower_ascii(std::string_view s) {
  std::string out(s);
  for (char& ch : out) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return out;
}

std::string read_file_bytes(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::invalid_argument("cannot open " + p.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

ModelFamily classify_model_tag(std::string_view tag) {
  const std::string lower = to_lower_ascii(tag);
  if (lower.starts_with("qwen3_5") || lower.starts_with("qwen3.5") || lower.starts_with("qwen3.8")) {
    return ModelFamily::Qwen3_5;
  }
  if (lower.starts_with("inkling")) return ModelFamily::Inkling2;
  if (lower.starts_with("deepseek")) return ModelFamily::DeepSeekV4;
  if (lower.starts_with("glm")) return ModelFamily::GlmFlash;
  return ModelFamily::Unknown;
}

struct RegistryState {
  std::mutex mu;
  std::unordered_map<std::uint8_t, BackendRegistration> by_family;
};

RegistryState& registry_state() {
  static RegistryState state;
  return state;
}

}  // namespace

const char* to_string(ModelFamily family) noexcept {
  switch (family) {
    case ModelFamily::Inkling2: return "inkling2";
    case ModelFamily::Qwen3_5: return "qwen3_5";
    case ModelFamily::DeepSeekV4: return "deepseek_v4";
    case ModelFamily::GlmFlash: return "glm_flash";
    default: return "unknown";
  }
}

const char* to_string(CheckpointFormat fmt) noexcept {
  switch (fmt) {
    case CheckpointFormat::Safetensors: return "safetensors";
    case CheckpointFormat::Gguf: return "gguf";
    default: return "unknown";
  }
}

const char* to_string(ActivationDtype dt) noexcept {
  switch (dt) {
    case ActivationDtype::Bf16: return "bf16";
    case ActivationDtype::Fp16: return "fp16";
    case ActivationDtype::Fp32: return "fp32";
    default: return "unknown";
  }
}

ModelFamily parse_model_family(std::string_view s) {
  const std::string lower = to_lower_ascii(s);
  if (lower == "inkling2" || lower == "inkling-2") return ModelFamily::Inkling2;
  if (lower == "qwen3_5" || lower == "qwen3.5" || lower == "qwen3_5_text" || lower == "qwen38" || lower == "qwen3.8") {
    return ModelFamily::Qwen3_5;
  }
  if (lower == "deepseek_v4" || lower == "deepseekv4" || lower == "deepseek_v41") {
    return ModelFamily::DeepSeekV4;
  }
  if (lower == "glm_flash" || lower == "glm53" || lower == "glm_53_flash") {
    return ModelFamily::GlmFlash;
  }
  throw std::invalid_argument("unknown model family '" + std::string(s) + "'");
}

ModelFamily detect_model_family(const std::filesystem::path& model_dir) {
  if (!std::filesystem::is_directory(model_dir)) {
    throw std::invalid_argument("model directory does not exist: " + model_dir.string());
  }
  const std::filesystem::path cfg_path = model_dir / "config.json";
  if (std::filesystem::is_regular_file(cfg_path)) {
    nlohmann::json cfg;
    try {
      cfg = nlohmann::json::parse(read_file_bytes(cfg_path));
    } catch (const nlohmann::json::exception& e) {
      throw std::invalid_argument("invalid config.json in " + model_dir.string() + ": " + e.what());
    }
    if (!cfg.is_object()) {
      throw std::invalid_argument("config.json in " + model_dir.string() + " must be a JSON object");
    }
    if (auto it = cfg.find("architectures"); it != cfg.end() && it->is_array()) {
      for (const auto& arch : *it) {
        if (arch.is_string()) {
          const ModelFamily fam = classify_model_tag(arch.get<std::string>());
          if (fam != ModelFamily::Unknown) return fam;
        }
      }
    }
    if (auto it = cfg.find("model_type"); it != cfg.end() && it->is_string()) {
      const ModelFamily fam = classify_model_tag(it->get<std::string>());
      if (fam != ModelFamily::Unknown) return fam;
    }
    if (auto tc = cfg.find("text_config"); tc != cfg.end() && tc->is_object()) {
      if (auto mt = tc->find("model_type"); mt != tc->end() && mt->is_string()) {
        const ModelFamily fam = classify_model_tag(mt->get<std::string>());
        if (fam != ModelFamily::Unknown) return fam;
      }
    }
    throw std::invalid_argument("unrecognized model family in " + cfg_path.string());
  }

  // Inkling checkpoints may omit config.json and ship tokenizer.json + chat_template.jinja.
  const std::filesystem::path tok_path = model_dir / "tokenizer.json";
  if (std::filesystem::is_regular_file(tok_path)) {
    nlohmann::json tok;
    try {
      tok = nlohmann::json::parse(read_file_bytes(tok_path));
    } catch (const nlohmann::json::exception& e) {
      throw std::invalid_argument("invalid tokenizer.json in " + model_dir.string() + ": " + e.what());
    }
    if (auto added = tok.find("added_tokens"); added != tok.end() && added->is_array()) {
      for (const auto& item : *added) {
        if (item.is_object() && item.value("content", std::string{}) == "<|message_model|>") {
          return ModelFamily::Inkling2;
        }
      }
    }
  }
  throw std::invalid_argument("cannot detect model family in " + model_dir.string() + " (missing config.json)");
}

CheckpointFormat detect_checkpoint_format(const std::filesystem::path& model_dir) {
  if (!std::filesystem::is_directory(model_dir)) {
    throw std::invalid_argument("checkpoint directory does not exist: " + model_dir.string());
  }
  bool has_safetensors = false;
  bool has_gguf = false;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(model_dir, ec)) {
    if (ec) {
      throw std::invalid_argument("cannot scan checkpoint directory " + model_dir.string() + ": " + ec.message());
    }
    if (!entry.is_regular_file(ec)) continue;
    const std::string name = entry.path().filename().string();
    if (name == "model.safetensors" || name == "model.safetensors.index.json" ||
        entry.path().extension() == ".safetensors") {
      has_safetensors = true;
    } else if (entry.path().extension() == ".gguf") {
      has_gguf = true;
    }
  }
  if (has_safetensors && has_gguf) {
    throw std::invalid_argument("ambiguous checkpoint directory contains both safetensors and gguf: " +
                                model_dir.string());
  }
  if (has_safetensors) return CheckpointFormat::Safetensors;
  if (has_gguf) return CheckpointFormat::Gguf;
  throw std::invalid_argument("no safetensors or gguf weights found in " + model_dir.string());
}

bool BackendStateIdentity::is_reusable_with(const BackendStateIdentity& other) const noexcept {
  if (family == ModelFamily::Unknown || other.family == ModelFamily::Unknown) return false;
  if (state_layout_id.empty() || other.state_layout_id.empty()) return false;
  if (act_dtype == ActivationDtype::Unknown || other.act_dtype == ActivationDtype::Unknown) return false;
  if (checkpoint.format == CheckpointFormat::Unknown || other.checkpoint.format == CheckpointFormat::Unknown) {
    return false;
  }
  if (checkpoint.shard_count == 0 || other.checkpoint.shard_count == 0) return false;
  if (!checkpoint.weight_manifest_verified || !other.checkpoint.weight_manifest_verified) return false;
  return *this == other;
}

std::uint64_t hash_backend_state_identity(const BackendStateIdentity& id) noexcept {
  std::uint64_t h = 14695981039346656037ull;
  const auto fam = static_cast<std::uint8_t>(id.family);
  const auto fmt = static_cast<std::uint8_t>(id.checkpoint.format);
  const auto act = static_cast<std::uint8_t>(id.act_dtype);
  h = fnv1a_mix(h, &fam, sizeof(fam));
  fnv1a_mix_string(h, id.backend_name);
  h = fnv1a_mix(h, &fmt, sizeof(fmt));
  fnv1a_mix_string(h, id.checkpoint.schema_variant);
  fnv1a_mix_string(h, id.checkpoint.model_id);
  fnv1a_mix_string(h, id.checkpoint.revision);
  h = fnv1a_mix(h, &id.checkpoint.shard_count, sizeof(id.checkpoint.shard_count));
  h = fnv1a_mix(h, id.checkpoint.shard_set_sha256.data(), id.checkpoint.shard_set_sha256.size());
  h = fnv1a_mix(h, id.checkpoint.config_sha256.data(), id.checkpoint.config_sha256.size());
  h = fnv1a_mix(h, id.checkpoint.tokenizer_sha256.data(), id.checkpoint.tokenizer_sha256.size());
  h = fnv1a_mix(h, id.checkpoint.chat_template_sha256.data(), id.checkpoint.chat_template_sha256.size());
  h = fnv1a_mix(h, id.checkpoint.weight_manifest_sha256.data(), id.checkpoint.weight_manifest_sha256.size());
  h = fnv1a_mix(h, &id.checkpoint.weight_manifest_verified, sizeof(id.checkpoint.weight_manifest_verified));
  h = fnv1a_mix(h, &id.checkpoint.shard_digests_verified, sizeof(id.checkpoint.shard_digests_verified));
  fnv1a_mix_string(h, id.state_layout_id);
  h = fnv1a_mix(h, &act, sizeof(act));
  fnv1a_mix_string(h, id.weight_precision);
  fnv1a_mix_string(h, id.kv_format);
  fnv1a_mix_string(h, id.kv_cache_mode);
  fnv1a_mix_string(h, id.gear);
  h = fnv1a_mix(h, &id.max_ctx, sizeof(id.max_ctx));
  h = fnv1a_mix(h, &id.prefill_chunk, sizeof(id.prefill_chunk));
  h = fnv1a_mix(h, &id.spec_enabled, sizeof(id.spec_enabled));
  h = fnv1a_mix(h, &id.mtp_enabled, sizeof(id.mtp_enabled));
  h = fnv1a_mix(h, &id.media_enabled, sizeof(id.media_enabled));
  h = fnv1a_mix(h, &id.persona_key, sizeof(id.persona_key));
  fnv1a_mix_string(h, id.persona_fingerprint);
  fnv1a_mix_string(h, id.cache_namespace);
  fnv1a_mix_string(h, id.rendering_fingerprint);
  fnv1a_mix_string(h, id.family_state_fingerprint);
  return h;
}

bool prefix_matches_exact(std::span<const std::int32_t> cached_ids, std::span<const std::int32_t> prompt_ids,
                          std::size_t prefix_len) noexcept {
  if (prefix_len > cached_ids.size() || prefix_len > prompt_ids.size()) return false;
  return std::equal(cached_ids.begin(), cached_ids.begin() + static_cast<std::ptrdiff_t>(prefix_len),
                    prompt_ids.begin());
}

bool prefix_matches_exact_with_mask(std::span<const std::int32_t> cached_ids,
                                    std::span<const std::uint8_t> cached_mask,
                                    std::span<const std::int32_t> prompt_ids,
                                    std::span<const std::uint8_t> prompt_mask, std::size_t prefix_len) noexcept {
  if (prefix_len > cached_ids.size() || prefix_len > cached_mask.size() || prefix_len > prompt_ids.size() ||
      prefix_len > prompt_mask.size()) {
    return false;
  }
  return std::equal(cached_ids.begin(), cached_ids.begin() + static_cast<std::ptrdiff_t>(prefix_len),
                    prompt_ids.begin()) &&
         std::equal(cached_mask.begin(), cached_mask.begin() + static_cast<std::ptrdiff_t>(prefix_len),
                    prompt_mask.begin());
}

void validate_runtime_options_against_capabilities(const BackendCapabilities& caps, const RuntimeOptions& options) {
  if (caps.family == ModelFamily::Unknown) {
    throw std::invalid_argument("backend capabilities must declare a known ModelFamily");
  }
  if (options.max_ctx < 1) {
    throw std::invalid_argument("max_ctx must be positive");
  }
  if (caps.max_position_embeddings > 0 && options.max_ctx > caps.max_position_embeddings) {
    throw std::invalid_argument("max_ctx " + std::to_string(options.max_ctx) + " exceeds model limit " +
                                std::to_string(caps.max_position_embeddings));
  }
  if (options.prefill_chunk < 1) {
    throw std::invalid_argument("prefill_chunk must be positive");
  }
  if (options.max_batch < 1) {
    throw std::invalid_argument("max_batch must be at least 1");
  }
  if (options.max_batch > 1 && !caps.supports_continuous_batching) {
    throw std::invalid_argument(caps.backend_name + ": continuous batching (max_batch > 1) is not supported");
  }
  if (options.max_batch > std::max(1, caps.max_supported_batch)) {
    throw std::invalid_argument(caps.backend_name + ": max_batch " + std::to_string(options.max_batch) +
                                " exceeds max_supported_batch " + std::to_string(caps.max_supported_batch));
  }
  if (options.spec && !caps.supports_speculation) {
    throw std::invalid_argument(caps.backend_name + ": speculative decoding is not supported");
  }
  if (options.vision && !caps.supports_images) {
    throw std::invalid_argument(caps.backend_name + ": vision/image input is not supported");
  }
  if (!options.ablation.empty() && !caps.supports_ablation) {
    throw std::invalid_argument(caps.backend_name + ": ablation files are not supported");
  }
  if (options.enable_persona_hooks && !caps.supports_persona_hooks) {
    throw std::invalid_argument(caps.backend_name + ": persona hooks are not supported");
  }
  if ((!options.kv_ckpt_path.empty() || !options.kv_ckpt_resume.empty() || options.kv_ckpt_every > 0) &&
      !caps.supports_disk_checkpoint) {
    throw std::invalid_argument(caps.backend_name + ": disk KV checkpoints are not supported");
  }
}

void validate_request_against_capabilities(const BackendCapabilities& caps, int max_model_len, const Prompt& prompt,
                                           int max_new_tokens, const Sampling& sampling) {
  if (prompt.ids.empty()) {
    throw std::invalid_argument("prompt must not be empty");
  }
  if (max_new_tokens < 1) {
    throw std::invalid_argument("max_new_tokens must be positive");
  }
  if (!prompt.images.empty() && !caps.supports_images) {
    throw std::invalid_argument(caps.backend_name + ": image prompts are not supported");
  }
  if (sampling.persona != nullptr && !caps.supports_persona_hooks) {
    throw std::invalid_argument(caps.backend_name + ": per-request persona profiles are not supported");
  }
  if (max_model_len <= 0 || static_cast<std::int64_t>(prompt.ids.size()) >= static_cast<std::int64_t>(max_model_len)) {
    throw std::invalid_argument("prompt of " + std::to_string(prompt.ids.size()) +
                                " tokens does not fit max_model_len " + std::to_string(max_model_len));
  }
  if (caps.vocab_size > 0) {
    for (std::size_t i = 0; i < prompt.ids.size(); ++i) {
      const std::int32_t id = prompt.ids[i];
      if (id < 0 && caps.supports_images) continue;
      if (id < 0 || id >= caps.vocab_size) {
        throw std::out_of_range("prompt token id " + std::to_string(id) + " at index " + std::to_string(i) +
                                " out of vocab [0, " + std::to_string(caps.vocab_size) + ")");
      }
    }
    for (std::size_t i = 0; i < sampling.forced.size(); ++i) {
      const std::int32_t id = sampling.forced[i];
      if (id < 0 || id >= caps.vocab_size) {
        throw std::out_of_range("forced token id " + std::to_string(id) + " at index " + std::to_string(i) +
                                " out of vocab [0, " + std::to_string(caps.vocab_size) + ")");
      }
    }
    for (std::size_t i = 0; i < sampling.banned_tokens.size(); ++i) {
      const std::int32_t id = sampling.banned_tokens[i];
      if (id < 0 || id >= caps.vocab_size) {
        throw std::out_of_range("banned token id " + std::to_string(id) + " at index " + std::to_string(i) +
                                " out of vocab [0, " + std::to_string(caps.vocab_size) + ")");
      }
    }
  }
}

BackendRegistry& BackendRegistry::instance() {
  static BackendRegistry registry;
  return registry;
}

void BackendRegistry::register_backend(BackendRegistration reg) {
  if (reg.family == ModelFamily::Unknown || reg.name.empty() || !reg.create) {
    throw std::invalid_argument("BackendRegistration requires known family, non-empty name, and factory");
  }
  RegistryState& st = registry_state();
  std::lock_guard lock(st.mu);
  st.by_family[static_cast<std::uint8_t>(reg.family)] = std::move(reg);
}

const BackendRegistration* BackendRegistry::find(ModelFamily family) const {
  RegistryState& st = registry_state();
  std::lock_guard lock(st.mu);
  const auto it = st.by_family.find(static_cast<std::uint8_t>(family));
  return it == st.by_family.end() ? nullptr : &it->second;
}

std::unique_ptr<ModelBackend> BackendRegistry::create(ModelFamily family, const RuntimeOptions& options) const {
  std::function<std::unique_ptr<ModelBackend>(const RuntimeOptions&)> factory;
  {
    RegistryState& st = registry_state();
    std::lock_guard lock(st.mu);
    const auto it = st.by_family.find(static_cast<std::uint8_t>(family));
    if (it == st.by_family.end() || !it->second.create) {
      throw std::invalid_argument(std::string("no backend registered for model family ") + to_string(family));
    }
    factory = it->second.create;
  }
  return factory(options);
}

std::vector<ModelFamily> BackendRegistry::registered_families() const {
  RegistryState& st = registry_state();
  std::lock_guard lock(st.mu);
  std::vector<ModelFamily> out;
  out.reserve(st.by_family.size());
  for (const auto& [k, _] : st.by_family) out.push_back(static_cast<ModelFamily>(k));
  std::sort(out.begin(), out.end(),
            [](ModelFamily a, ModelFamily b) { return static_cast<std::uint8_t>(a) < static_cast<std::uint8_t>(b); });
  return out;
}

BackendCapabilities inkling_default_capabilities() {
  BackendCapabilities c;
  c.family = ModelFamily::Inkling2;
  c.backend_name = "inkling2-cuda";
  c.architecture = "Inkling2ForConditionalGeneration";
  c.supports_text = true;
  c.supports_images = true;
  c.supports_speculation = true;
  c.supports_continuous_batching = true;
  c.max_supported_batch = 64;
  c.supports_prefix_cache = true;
  c.supports_host_park = true;
  c.supports_disk_checkpoint = true;
  c.supports_persona_hooks = true;
  c.supports_ablation = true;
  c.supports_logit_capture = true;
  c.supported_gears = {"loop_1", "loop_2", "loop_3", "loop_4", "loop_5", "loop_6", "loop_7", "loop_8"};
  c.supported_kv_cache_modes = {"per_pass", "last_full"};
  c.supported_quant_layouts = {"bf16", "dense_fp8", "nvfp4", "mxfp8", "atlas2", "atlas3", "exl3"};
  c.vocab_size = 201024;
  c.max_position_embeddings = 1048576;
  return c;
}

}  // namespace ingine::model

#include "backends/qwen38/qwen38_backend.h"

#include <algorithm>
#include <initializer_list>
#include <stdexcept>

namespace ingine::backends::qwen38 {
namespace {

bool contains_mode(const std::vector<std::string>& allowed, std::string_view value) noexcept {
  return std::find(allowed.begin(), allowed.end(), value) != allowed.end();
}

void validate_qwen38_profile_selection(const ingine::model::BackendCapabilities& caps,
                                       std::string_view kv_cache_mode, std::string_view gear) {
  if (!contains_mode(caps.supported_kv_cache_modes, kv_cache_mode)) {
    throw std::invalid_argument("qwen38-dense unsupported kv_cache_mode: '" + std::string(kv_cache_mode) + "'");
  }
  if (!contains_mode(caps.supported_gears, gear)) {
    throw std::invalid_argument("qwen38-dense unsupported gear: '" + std::string(gear) + "'");
  }
}

}  // namespace

ingine::model::BackendCapabilities qwen38_dense_capabilities(const Qwen38Config& config, int max_supported_batch) {
  config.validate();
  ingine::model::BackendCapabilities caps;
  caps.family = ingine::model::ModelFamily::Qwen3_5;
  caps.backend_name = "qwen38-dense";
  caps.architecture = !config.architectures.empty() ? config.architectures.front() : "Qwen3_5ForConditionalGeneration";
  caps.supports_text = true;
  caps.supports_images = false;
  caps.supports_speculation = false;
  caps.supports_continuous_batching = max_supported_batch > 1;
  caps.max_supported_batch = max_supported_batch > 0 ? max_supported_batch : 1;
  caps.supports_prefix_cache = true;
  caps.supports_host_park = false;
  caps.supports_disk_checkpoint = false;
  caps.supports_persona_hooks = false;
  caps.supports_ablation = false;
  caps.supports_logit_capture = true;
  caps.supported_gears = {"dense"};
  caps.supported_kv_cache_modes = {"dense_bf16"};
  if (config.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4) {
    caps.supported_quant_layouts = {"mixed_fp8_nvfp4_w4a16"};
  } else {
    caps.supported_quant_layouts = {"bf16"};
  }
  if (config.vocab_size > 248057) {
    caps.media_token_ids = {248053, 248054, 248055, 248056, 248057};
  }
  caps.vocab_size = config.vocab_size;
  caps.max_position_embeddings = config.max_position_embeddings;
  return caps;
}

ingine::model::BackendStateIdentity make_qwen38_state_identity(const Qwen38Manifest& manifest,
                                                              const ingine::model::RuntimeOptions& options,
                                                              std::string_view kv_cache_mode, std::string_view gear,
                                                              std::string_view cache_namespace) {
  validate_qwen38_runtime_options(manifest.config, options);
  const auto caps = qwen38_dense_capabilities(manifest.config, options.max_batch);
  validate_qwen38_profile_selection(caps, kv_cache_mode, gear);

  const auto live_cfg_fp = qwen38_config_fingerprint(manifest.config);
  const std::string live_cfg_hex = qwen38_config_fingerprint_hex(manifest.config);

  ingine::model::BackendStateIdentity id;
  id.family = ingine::model::ModelFamily::Qwen3_5;
  id.backend_name = "qwen38-dense";
  id.checkpoint = manifest.checkpoint;
  // Bind the effective parsed execution config into verification status: if config fields were
  // mutated in memory after inspection, revoke verification fail-closed.
  if (manifest.parsed_config_fingerprint != live_cfg_fp) {
    id.checkpoint.verification_level = ingine::model::VerificationLevel::Unverified;
    id.checkpoint.shard_digests_verified = false;
  }
  id.state_layout_id = "qwen3_5_gdn_kv_v1";
  id.act_dtype = ingine::model::ActivationDtype::Bf16;
  id.numeric_abi = manifest.config.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4
                       ? "act=bf16;attn=fp8_e4m3_cal;mlp=nvfp4_w4a16;gdn=f32;kv=bf16"
                       : "act=bf16;attn=bf16;mlp=bf16;gdn=f32;kv=bf16";
  id.weight_precision = to_string(manifest.config.weight_precision);
  id.kv_format = "gdn_f32_conv_bf16_kv_bf16";
  id.kv_cache_mode = std::string(kv_cache_mode);
  id.gear = std::string(gear);
  id.max_ctx = options.max_ctx;
  id.prefill_chunk = options.prefill_chunk;
  id.spec_enabled = false;
  id.mtp_enabled = false;
  id.media_enabled = false;
  id.persona_key = 0;
  id.persona_fingerprint = "";
  id.cache_namespace = std::string(cache_namespace);
  id.family_state_fingerprint = "gdn" + std::to_string(manifest.config.num_linear_layers) + "_full" +
                                std::to_string(manifest.config.num_full_layers) + "_conv" +
                                std::to_string(manifest.config.linear_conv_kernel_dim) + "_rope" +
                                std::to_string(manifest.config.rotary_dim) + "_v1_cfg_" + live_cfg_hex;
  return id;
}

ingine::model::WorkerProfile make_qwen38_worker_profile(const Qwen38Manifest& manifest,
                                                        const ingine::model::RuntimeOptions& options,
                                                        std::string_view kv_cache_mode, std::string_view gear) {
  validate_qwen38_runtime_options(manifest.config, options);
  ingine::model::WorkerProfile prof;
  prof.family = ingine::model::ModelFamily::Qwen3_5;
  prof.model_id = manifest.checkpoint.model_id;
  prof.gear = std::string(gear);
  prof.kv_cache_mode = std::string(kv_cache_mode);
  prof.tokenizer_format = "qwen2_bpe_nfc";
  prof.chat_template_family = "qwen3_5_chatml_xml";
  prof.max_ctx = options.max_ctx;
  prof.max_batch = options.max_batch;
  prof.capabilities = qwen38_dense_capabilities(manifest.config, options.max_batch);
  validate_qwen38_profile_selection(prof.capabilities, kv_cache_mode, gear);
  if (!manifest.media_token_ids.empty()) {
    prof.capabilities.media_token_ids.insert(prof.capabilities.media_token_ids.end(),
                                             manifest.media_token_ids.begin(), manifest.media_token_ids.end());
    std::sort(prof.capabilities.media_token_ids.begin(), prof.capabilities.media_token_ids.end());
    prof.capabilities.media_token_ids.erase(
        std::unique(prof.capabilities.media_token_ids.begin(), prof.capabilities.media_token_ids.end()),
        prof.capabilities.media_token_ids.end());
  }
  prof.state_identity = make_qwen38_state_identity(manifest, options, kv_cache_mode, gear);
  return prof;
}

void validate_qwen38_runtime_options(const Qwen38Config& config, const ingine::model::RuntimeOptions& options) {
  const auto caps = qwen38_dense_capabilities(config);
  ingine::model::validate_runtime_options_against_capabilities(caps, options);

  if (config.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4 && !options.dense_fp8) {
    throw std::invalid_argument("qwen38-dense MixedFp8Nvfp4 requires dense_fp8=true (attention uses calibrated FP8)");
  }
  if (config.weight_precision == Qwen38WeightPrecision::Bf16 && options.dense_fp8) {
    throw std::invalid_argument("qwen38-dense BF16 checkpoint rejects dense_fp8=true (effective numeric ABI mismatch)");
  }
  if (options.loop_passes_explicit || options.loop_passes != 3) {
    throw std::invalid_argument("qwen38-dense is a non-looped dense architecture and does not support --loop-passes");
  }
  if (options.kv_cache_mode != ingine::model::KvCacheMode::PerPass) {
    throw std::invalid_argument("qwen38-dense does not support kv_cache_mode '" +
                                std::string(ingine::model::to_string(options.kv_cache_mode)) +
                                "' (only per_pass/dense_bf16 is supported)");
  }
  if (options.kv_fp8) {
    throw std::invalid_argument("qwen38-dense does not yet support kv_fp8");
  }
  if (options.kv_q4) {
    throw std::invalid_argument("qwen38-dense does not support kv_q4");
  }
  if (options.kv_paged) {
    throw std::invalid_argument("qwen38-dense Phase 1 requires contiguous dense KV/GDN slots (kv_paged unsupported)");
  }
  if (options.kv_vram_mb != 0 || !options.kv_swap_dir.empty()) {
    throw std::invalid_argument("qwen38-dense does not support tiered KV swap options");
  }
  if (options.offload_vram_mb != 0) {
    throw std::invalid_argument("qwen38-dense does not support offload_vram_mb in Phase 1");
  }
  if (options.park_mb > 0) {
    throw std::invalid_argument("qwen38-dense does not support host/VRAM conversation parking yet");
  }
  if (options.park_host_mb > 0) {
    throw std::invalid_argument("qwen38-dense does not support host conversation parking (park_host_mb)");
  }
  if (options.max_kv_tokens != 0) {
    throw std::invalid_argument("qwen38-dense does not support shared paged pool max_kv_tokens in Phase 1");
  }
  if (!options.expert_ranking.empty() || !options.dump_router_stats.empty()) {
    throw std::invalid_argument("qwen38-dense has no MoE router/expert ranking hooks");
  }
  if (!options.refusal_map.empty()) {
    throw std::invalid_argument("qwen38-dense does not support refusal_map hooks");
  }
  if (!options.dflash2_dir.empty()) {
    throw std::invalid_argument("qwen38-dense does not support --dflash2-dir (speculative decoding unsupported)");
  }
  if (options.loopcd_mode != ingine::model::LoopCdMode::Off) {
    throw std::invalid_argument("qwen38-dense is a non-looped dense architecture and does not support --loopcd");
  }
  if (options.loop_bf16) {
    throw std::invalid_argument("qwen38-dense is a non-looped dense architecture and does not support --loop-bf16");
  }
  if (options.attn_requant != ingine::model::AttnRequantMode::Off) {
    throw std::invalid_argument("qwen38-dense does not support --attn-requant");
  }
  if (options.expert_swaps != 0) {
    throw std::invalid_argument("qwen38-dense has no MoE experts and does not support --expert-swaps");
  }
}

Qwen38RuntimeDispatchPlan resolve_qwen38_runtime_options(const ingine::model::RuntimeOptions& options,
                                                         bool allow_unpinned) {
  if (options.loopcd_mode != ingine::model::LoopCdMode::Off) {
    throw std::invalid_argument("qwen38-dense is a non-looped dense architecture and does not support --loopcd");
  }
  if (options.model_dir.empty()) {
    throw std::invalid_argument("qwen38-dense dispatch requires a non-empty model_dir");
  }
  const Qwen38Config config = Qwen38Config::load(options.model_dir / "config.json");
  ingine::model::RuntimeOptions normalized = options;
  // MixedFp8Nvfp4 checkpoints use calibrated FP8 attention natively even when the CLI omits --dense-fp8.
  if (config.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4) {
    normalized.dense_fp8 = true;
  }
  validate_qwen38_runtime_options(config, normalized);

  Qwen38RuntimeDispatchPlan plan;
  plan.normalized_options = normalized;
  plan.max_ctx = normalized.max_ctx;
  plan.prefill_chunk = normalized.prefill_chunk;
  plan.verify_pins = !allow_unpinned;
  plan.use_calibrated_fp8 = (config.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4);
  plan.prefix_cache = normalized.prefix_cache;
  return plan;
}

std::size_t find_qwen35_pre_suffix_checkpoint(std::span<const std::int32_t> prompt_ids) noexcept {
  // Pinned Qwen3.5 / Qwen3.8 BPE + special token IDs (verified from tokenizer.json):
  //   <|im_start|>=248045, <|im_end|>=248046, assistant=74455, \n=198, \n\n=271,
  //   <think>=248068, </think>=248069.
  constexpr std::int32_t kImStart = 248045;
  constexpr std::int32_t kImEnd = 248046;
  constexpr std::int32_t kAssistant = 74455;
  constexpr std::int32_t kNewline = 198;
  constexpr std::int32_t kDoubleNewline = 271;
  constexpr std::int32_t kThink = 248068;
  constexpr std::int32_t kEndThink = 248069;

  auto match_tail = [&](std::initializer_list<std::int32_t> suffix) -> std::size_t {
    const std::size_t slen = suffix.size();
    if (prompt_ids.size() <= slen) return 0;
    const std::size_t start = prompt_ids.size() - slen;
    std::size_t idx = start;
    for (std::int32_t expected : suffix) {
      if (prompt_ids[idx++] != expected) return 0;
    }
    // Require a completed prior turn boundary (`<|im_end|>\n` or `<|im_end|>`) directly before
    // `<|im_start|>assistant...` so synthetic non-chat prompts never trigger unintended checkpoints.
    if (start >= 2 && prompt_ids[start - 1] == kNewline && prompt_ids[start - 2] == kImEnd) {
      return start;
    }
    if (start >= 1 && prompt_ids[start - 1] == kImEnd) {
      return start;
    }
    return 0;
  };

  // Check longest suffixes first:
  // 1. enable_thinking=false: <|im_start|>assistant\n<think>\n\n</think>\n\n (7 tokens)
  if (const std::size_t p =
          match_tail({kImStart, kAssistant, kNewline, kThink, kDoubleNewline, kEndThink, kDoubleNewline});
      p > 0) {
    return p;
  }
  // 2. enable_thinking=true: <|im_start|>assistant\n<think>\n (5 tokens)
  if (const std::size_t p = match_tail({kImStart, kAssistant, kNewline, kThink, kNewline}); p > 0) {
    return p;
  }
  // 3. Bare assistant generation prompt: <|im_start|>assistant\n (3 tokens)
  if (const std::size_t p = match_tail({kImStart, kAssistant, kNewline}); p > 0) {
    return p;
  }
  return 0;
}

}  // namespace ingine::backends::qwen38

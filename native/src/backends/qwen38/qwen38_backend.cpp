#include "backends/qwen38/qwen38_backend.h"

#include <stdexcept>

namespace ingine::backends::qwen38 {

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
  caps.vocab_size = config.vocab_size;
  caps.max_position_embeddings = config.max_position_embeddings;
  return caps;
}

ingine::model::BackendStateIdentity make_qwen38_state_identity(const Qwen38Manifest& manifest,
                                                              const ingine::model::RuntimeOptions& options,
                                                              std::string_view kv_cache_mode, std::string_view gear,
                                                              std::string_view cache_namespace) {
  manifest.config.validate();
  ingine::model::BackendStateIdentity id;
  id.family = ingine::model::ModelFamily::Qwen3_5;
  id.backend_name = "qwen38-dense";
  id.checkpoint = manifest.checkpoint;
  id.state_layout_id = "qwen3_5_gdn_kv_v1";
  id.act_dtype = ingine::model::ActivationDtype::Bf16;
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
                                std::to_string(manifest.config.rotary_dim) + "_v1";
  return id;
}

ingine::model::WorkerProfile make_qwen38_worker_profile(const Qwen38Manifest& manifest,
                                                        const ingine::model::RuntimeOptions& options,
                                                        std::string_view kv_cache_mode, std::string_view gear) {
  ingine::model::WorkerProfile prof;
  prof.family = ingine::model::ModelFamily::Qwen3_5;
  prof.model_id = manifest.checkpoint.model_id;
  prof.gear = std::string(gear);
  prof.kv_cache_mode = std::string(kv_cache_mode);
  prof.tokenizer_format = "qwen2_bpe_nfc";
  prof.chat_template_family = "qwen3_5_chatml_xml";
  prof.max_ctx = options.max_ctx;
  prof.max_batch = options.max_batch;
  prof.capabilities = qwen38_dense_capabilities(manifest.config);
  prof.state_identity = make_qwen38_state_identity(manifest, options, kv_cache_mode, gear);
  return prof;
}

void validate_qwen38_runtime_options(const Qwen38Config& config, const ingine::model::RuntimeOptions& options) {
  const auto caps = qwen38_dense_capabilities(config);
  ingine::model::validate_runtime_options_against_capabilities(caps, options);

  if (options.kv_fp8) {
    throw std::invalid_argument("qwen38-dense does not yet support kv_fp8");
  }
  if (options.kv_q4) {
    throw std::invalid_argument("qwen38-dense does not support kv_q4");
  }
  if (options.kv_paged) {
    throw std::invalid_argument("qwen38-dense Phase 1 requires contiguous dense KV/GDN slots (kv_paged unsupported)");
  }
  if (options.offload_vram_mb > 0) {
    throw std::invalid_argument("qwen38-dense does not support offload_vram_mb in Phase 1");
  }
}

}  // namespace ingine::backends::qwen38

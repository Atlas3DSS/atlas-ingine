#pragma once

#include "backends/qwen38/qwen38_config.h"
#include "backends/qwen38/qwen38_manifest.h"
#include "ingine/model_backend.h"

#include <string_view>

namespace ingine::backends::qwen38 {

// Returns conservative Phase 0/Phase 1 capabilities for the dense Qwen3.8 text worker.
ingine::model::BackendCapabilities qwen38_dense_capabilities(const Qwen38Config& config, int max_supported_batch = 1);

// Builds an exact BackendStateIdentity for dense Qwen3.8 state (GDN recurrent+conv + partial-RoPE full-attn KV).
ingine::model::BackendStateIdentity make_qwen38_state_identity(const Qwen38Manifest& manifest,
                                                              const ingine::model::RuntimeOptions& options,
                                                              std::string_view kv_cache_mode = "dense_bf16",
                                                              std::string_view gear = "dense",
                                                              std::string_view cache_namespace = "");

// Builds a host-visible WorkerProfile for Qwen3.8 dispatch (#821/#822).
ingine::model::WorkerProfile make_qwen38_worker_profile(const Qwen38Manifest& manifest,
                                                        const ingine::model::RuntimeOptions& options,
                                                        std::string_view kv_cache_mode = "dense_bf16",
                                                        std::string_view gear = "dense");

// Fail-closed validation rejecting unsupported options (Inkling speculation, persona hooks, ablation,
// vision, max_batch > 1 when disabled, KV FP8/Q4/paged/checkpoint/offload).
void validate_qwen38_runtime_options(const Qwen38Config& config, const ingine::model::RuntimeOptions& options);

}  // namespace ingine::backends::qwen38

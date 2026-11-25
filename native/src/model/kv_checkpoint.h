// Atomic streaming disk checkpointing and resume for long-context (up to 1M+) KV states.
// Persists full-attention KV (both resident and 3-tier paged VRAM/RAM/NVMe), sliding-window rings,
// convolution states, sequence cursor, optional last prefill logits, and SHA-256 pins over both the
// model config and exact prompt prefix [0, pos) so mismatched resumes fail loudly.
#pragma once

#include "ingine/prefix_cache.h"
#include "model/model_data.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace ingine::model {

struct KvCheckpointInfo {
  int pos = 0;
  int max_ctx_at_save = 0;
  std::uint32_t act_dtype = 0;  // raw static_cast<uint32_t>(ingine::DType): BF16=0, FP16=6
  std::uint32_t kv_format = 0;  // 0=ACT, 1=FP8, 2=Q4_0
  bool has_logits = false;
  // Optional v1 extended producer identity inside the zero-padded 4096-byte header page.
  // Legacy checkpoints leave these bytes zero (has_producer_identity=false) and cannot establish weights,
  // dense_fp8, or prefill_chunk beyond config.json; auto-resume and root promotion require verified identity.
  bool has_producer_identity = false;
  bool producer_dense_fp8 = false;
  int producer_prefill_chunk = 0;
  std::array<std::uint8_t, 32> prompt_sha256{};
  std::array<std::uint8_t, 32> model_sha256{};     // legacy config.json SHA-256
  std::array<std::uint8_t, 32> producer_sha256{};  // verified producer identity SHA-256 (when has_producer_identity)
};

// Producer metadata embedded into newly saved v1 disk checkpoints so auto-resume and root promotion can verify
// exact weights package, ablation, act_dtype, dense_fp8, kv_format, and prefill_chunk.
struct KvCheckpointProducer {
  bool valid = false;
  bool dense_fp8 = true;
  int prefill_chunk = 0;
  std::array<std::uint8_t, 32> producer_sha256{};
};

// Result of a single-fd checked checkpoint load (pos/has_logits/producer_identity all come from the exact fd restored).
struct KvCheckpointLoadResult {
  bool loaded = false;
  KvCheckpointInfo info;
};

// Map between StateIdentity::act_dtype (0=BF16, 1=FP16) and the on-disk RawHeader::act_dtype encoding
// (static_cast<uint32_t>(ingine::DType): BF16=0, FP16=6). Fails closed on any unknown or mismatched code.
inline constexpr bool kv_checkpoint_act_dtype_matches(std::uint32_t disk_act_dtype,
                                                      std::uint32_t identity_act_dtype) noexcept {
  if (identity_act_dtype == 0u) return disk_act_dtype == static_cast<std::uint32_t>(DType::BF16);
  if (identity_act_dtype == 1u) return disk_act_dtype == static_cast<std::uint32_t>(DType::FP16);
  return false;
}

inline constexpr bool kv_checkpoint_runtime_matches(const KvCheckpointInfo& info,
                                                    const StateIdentity& req_identity) noexcept {
  return kv_checkpoint_act_dtype_matches(info.act_dtype, req_identity.act_dtype) &&
         info.kv_format == req_identity.kv_format;
}

// Auto-resume eligibility predicate (evaluated on the same opened checkpoint fd used to restore):
//   * Requires verified producer identity (weights package + ablation + act_dtype + dense_fp8 + kv_format +
//     prefill_chunk), declining legacy config.json-only checkpoints and any producer mismatch.
//   * Deliberate compatible case: `max_ctx_at_save` need NOT equal `max_ctx` as long as `pos <= max_ctx`, because
//     full-attention KV rows are serialized compactly over [0, pos) and attention geometry (WINDOW=1024,
//     REL_EXTENT=1024, LOG_FLOOR=8192) is invariant to backing `max_ctx` allocation capacity.
inline bool kv_checkpoint_auto_resume_eligible(const KvCheckpointInfo& info, const StateIdentity& req_identity,
                                               const std::array<std::uint8_t, 32>& expected_producer_sha,
                                               std::size_t prompt_tokens, int max_ctx,
                                               bool has_media_in_prefix) noexcept {
  return !req_identity.spec_enabled && info.pos > 0 && std::size_t(info.pos) <= prompt_tokens &&
         kv_checkpoint_runtime_matches(info, req_identity) && info.has_producer_identity &&
         info.producer_sha256 == expected_producer_sha && info.pos <= max_ctx && !has_media_in_prefix &&
         !(info.pos == int(prompt_tokens) && !info.has_logits);
}

// Install a SIGTERM handler that sets an atomic flag checked at chunk boundaries during long prefill.
void install_kv_checkpoint_sigterm_handler() noexcept;
bool kv_checkpoint_sigterm_requested() noexcept;
void clear_kv_checkpoint_sigterm() noexcept;

// Compute SHA-256 over raw bytes (self-contained, no external crypto dependency).
std::array<std::uint8_t, 32> sha256_bytes(const void* data, std::size_t len) noexcept;
std::array<std::uint8_t, 32> sha256_file_or_zero(const std::filesystem::path& path);

// Combine per-32MiB-chunk SHA-256 digests of one safetensors tensor into its canonical content digest.
std::array<std::uint8_t, 32> sha256_tensor_from_chunk_digests(
    std::string_view name, std::string_view dtype, std::span<const std::int64_t> shape, std::uint64_t bytes,
    std::span<const std::array<std::uint8_t, 32>> chunk_shas) noexcept;

// Combine model metadata (config.json, hf_quant_config.json, model.safetensors.index.json; never trusting
// optional SHA256SUMS text) with sorted per-tensor content digests over 100% of weight bytes.
std::array<std::uint8_t, 32> compute_model_package_sha256_from_tensors(
    const std::filesystem::path& model_dir,
    const std::map<std::string, std::array<std::uint8_t, 32>>& tensor_shas);

// Pure in-memory digest strictly over the exact bytes consumed during load_weights (config.json SHA,
// consumed hf_quant_config.json SHA, and every consumed weight tensor's digest — zero post-load file re-reads).
std::array<std::uint8_t, 32> compute_consumed_weights_sha256(
    const std::array<std::uint8_t, 32>& config_sha, const std::array<std::uint8_t, 32>& quant_config_sha,
    const std::map<std::string, std::array<std::uint8_t, 32>>& tensor_shas) noexcept;

// Standalone on-disk model-package fingerprint (INGINE_MODEL_PKG_V2) over config.json, hf_quant_config.json,
// model.safetensors.index.json, and 100% of shard weight tensor bytes (32 MiB chunked SHA-256). Distinct from
// load_weights()'s in-memory consumed-bytes digest (INGINE_CONSUMED_WEIGHTS_V1, compute_consumed_weights_sha256).
// Returns std::nullopt when model_dir is empty or has no readable weight tensors.
std::optional<std::array<std::uint8_t, 32>> compute_model_package_sha256_opt(const std::filesystem::path& model_dir);
std::array<std::uint8_t, 32> compute_model_package_sha256(const std::filesystem::path& model_dir);

// Compute the verified disk-checkpoint producer digest over (model_sha256, act_dtype, dense_fp8, kv_format,
// prefill_chunk). Deliberately excludes max_ctx so checkpoints with pos <= runtime.max_ctx remain portable across
// backing context capacities when all producer-affecting fields match.
std::array<std::uint8_t, 32> compute_disk_producer_sha256(const StateIdentity& identity) noexcept;

// Inspect header metadata of an existing checkpoint file (throws on invalid magic/version/truncation).
KvCheckpointInfo inspect_kv_checkpoint(const std::filesystem::path& path);

// Atomically save `st` at exact token cursor `pos = prompt_prefix.size()` to `out_path` via
// `<out_path>.tmp.<pid>` -> fsync -> atomic rename -> parent dir fsync. Uses O(1) ~4.5 MiB streaming buffer.
void save_kv_checkpoint(State& st, const std::filesystem::path& model_dir,
                        std::span<const std::int32_t> prompt_prefix, const float* d_logits,
                        const std::filesystem::path& out_path, cudaStream_t stream,
                        const KvCheckpointProducer* producer = nullptr);

// Opens `path` once, validates the header from that fd, invokes `eligibility(info)` before modifying `st`, and
// restores from that SAME fd if eligible. If `eligibility` returns false, returns `{loaded=false, info}` with `st`
// untouched. If an error occurs after state copying begins, resets `st` (rings/conv/paged_kv/length) before rethrowing.
KvCheckpointLoadResult load_kv_checkpoint_checked(
    State& st, const std::filesystem::path& model_dir, std::span<const std::int32_t> full_prompt, float* d_logits,
    const std::filesystem::path& path, cudaStream_t stream,
    const std::function<bool(const KvCheckpointInfo&)>& eligibility);

// Validate header + model SHA-256 + exact prompt prefix SHA-256, then stream-load `path` into `st`
// (and `d_logits` if non-null and present in the checkpoint). Sets `st.length = pos` and returns `pos`.
// Fails loudly with std::runtime_error on any config, format, or prompt-prefix mismatch.
int load_kv_checkpoint(State& st, const std::filesystem::path& model_dir,
                       std::span<const std::int32_t> full_prompt, float* d_logits,
                       const std::filesystem::path& path, cudaStream_t stream);

}  // namespace ingine::model

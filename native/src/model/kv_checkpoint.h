// Atomic streaming disk checkpointing and resume for long-context (up to 1M+) KV states.
// Persists full-attention KV (both resident and 3-tier paged VRAM/RAM/NVMe), sliding-window rings,
// convolution states, sequence cursor, optional last prefill logits, and SHA-256 pins over both the
// model config and exact prompt prefix [0, pos) so mismatched resumes fail loudly.
#pragma once

#include "model/model_data.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>

namespace ingine::model {

struct KvCheckpointInfo {
  int pos = 0;
  int max_ctx_at_save = 0;
  std::uint32_t act_dtype = 0;
  std::uint32_t kv_format = 0;  // 0=ACT, 1=FP8, 2=Q4_0
  bool has_logits = false;
  std::array<std::uint8_t, 32> prompt_sha256{};
  std::array<std::uint8_t, 32> model_sha256{};
};

// Install a SIGTERM handler that sets an atomic flag checked at chunk boundaries during long prefill.
void install_kv_checkpoint_sigterm_handler() noexcept;
bool kv_checkpoint_sigterm_requested() noexcept;
void clear_kv_checkpoint_sigterm() noexcept;

// Compute SHA-256 over raw bytes (self-contained, no external crypto dependency).
std::array<std::uint8_t, 32> sha256_bytes(const void* data, std::size_t len) noexcept;
std::array<std::uint8_t, 32> sha256_file_or_zero(const std::filesystem::path& path);

// Inspect header metadata of an existing checkpoint file (throws on invalid magic/version/truncation).
KvCheckpointInfo inspect_kv_checkpoint(const std::filesystem::path& path);

// Atomically save `st` at exact token cursor `pos = prompt_prefix.size()` to `out_path` via
// `<out_path>.tmp.<pid>` -> fsync -> atomic rename -> parent dir fsync. Uses O(1) ~4.5 MiB streaming buffer.
void save_kv_checkpoint(State& st, const std::filesystem::path& model_dir,
                        std::span<const std::int32_t> prompt_prefix, const float* d_logits,
                        const std::filesystem::path& out_path, cudaStream_t stream);

// Validate header + model SHA-256 + exact prompt prefix SHA-256, then stream-load `path` into `st`
// (and `d_logits` if non-null and present in the checkpoint). Sets `st.length = pos` and returns `pos`.
// Fails loudly with std::runtime_error on any config, format, or prompt-prefix mismatch.
int load_kv_checkpoint(State& st, const std::filesystem::path& model_dir,
                       std::span<const std::int32_t> full_prompt, float* d_logits,
                       const std::filesystem::path& path, cudaStream_t stream);

}  // namespace ingine::model

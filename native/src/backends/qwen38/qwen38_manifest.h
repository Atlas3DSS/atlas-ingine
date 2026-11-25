#pragma once

#include "backends/qwen38/qwen38_config.h"
#include "ingine/model_backend.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ingine::backends::qwen38 {

struct Qwen38VerifiedFile {
  std::string filename;
  std::uint64_t dev = 0;
  std::uint64_t ino = 0;
  std::uint64_t byte_size = 0;
  std::array<std::uint8_t, 32> sha256{};
  bool payload_hashed = false;
};

struct Qwen38StreamResult {
  std::uint64_t bytes_delivered = 0;
  std::array<std::uint8_t, 32> sha256{};
  bool admitted = false;  // true ONLY after full stream EOF + exact size + expected SHA-256 pin match
};

struct Qwen38TensorInfo {
  std::string name;
  std::string shard_file;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::uint64_t data_offset_begin = 0;
  std::uint64_t data_offset_end = 0;

  [[nodiscard]] std::uint64_t byte_size() const noexcept { return data_offset_end - data_offset_begin; }
};

struct Qwen38MemoryBudget {
  std::uint64_t text_weight_bytes = 0;
  std::uint64_t excluded_visual_bytes = 0;
  std::uint64_t excluded_mtp_bytes = 0;
  std::uint64_t gdn_recurrent_bytes = 0;  // num_linear * V_heads * K_dim * V_dim * 4 (F32 state)
  std::uint64_t gdn_conv_bytes = 0;       // num_linear * conv_dim * kernel_dim * 2 (BF16 sliding state)
  std::uint64_t kv_bytes_per_token = 0;   // num_full * 2 * kv_heads * head_dim * 2 (BF16 K+V)
};

struct Qwen38Manifest {
  Qwen38Config config;
  ingine::model::CheckpointIdentity checkpoint;
  std::vector<std::string> shard_files;
  std::unordered_map<std::string, std::uint64_t> shard_sizes;
  std::unordered_map<std::string, std::string> shard_sha256;  // populated when hash_shards=true or inspect_pinned()
  std::unordered_map<std::string, Qwen38TensorInfo> active_tensors;
  std::size_t total_index_tensors = 0;
  std::size_t active_text_tensor_count = 0;
  std::size_t excluded_visual_tensor_count = 0;
  std::size_t excluded_mtp_tensor_count = 0;
  Qwen38MemoryBudget memory_budget;
  bool shard_payloads_verified = false;
  std::unordered_map<std::string, Qwen38VerifiedFile> verified_files;
  std::array<std::uint8_t, 32> parsed_config_fingerprint{};
  std::vector<std::int32_t> media_token_ids;

  // Verifies an already-open file descriptor against the pinned manifest: requires PinnedVerified,
  // matching fstat (S_ISREG, dev, ino, byte_size), and a full bounded streaming SHA-256 verification
  // over the FD so same-size in-place mutations are rejected fail-closed at any file size.
  void verify_open_fd(int fd, std::string_view filename, std::size_t chunk_bytes = 1024ull * 1024ull) const;

  // Streams exact bytes from an open FD (starting at offset 0 via pread) in bounded chunks
  // (default 1 MiB, max 16 MiB), hashing the exact delivered bytes with SHA-256 in the same pass.
  // Calls `on_admit(res)` ONLY after all `byte_size` bytes are delivered, EOF has no trailing
  // bytes, and computed SHA-256 matches the expected pin. On any mismatch/error (including
  // unpinned manifests or same-size mutations), `on_admit` is never called, `on_abort` is
  // invoked to discard staged state, and std::invalid_argument is thrown fail-closed.
  Qwen38StreamResult stream_verified_open_fd(
      int fd, std::string_view filename,
      const std::function<void(std::uint64_t offset, std::span<const std::uint8_t> chunk)>& on_chunk,
      const std::function<void(const Qwen38StreamResult&)>& on_admit = {},
      const std::function<void()>& on_abort = {},
      std::size_t chunk_bytes = 1024ull * 1024ull) const;

  // Opens `model_dir / filename` once and delegates to stream_verified_open_fd().
  Qwen38StreamResult stream_verified_file(
      const std::filesystem::path& model_dir, std::string_view filename,
      const std::function<void(std::uint64_t offset, std::span<const std::uint8_t> chunk)>& on_chunk,
      const std::function<void(const Qwen38StreamResult&)>& on_admit = {},
      const std::function<void()>& on_abort = {},
      std::size_t chunk_bytes = 1024ull * 1024ull) const;

  // Bounded metadata helper (<= 256 MiB): opens `model_dir / filename` via stream_verified_file() and
  // admits buffered bytes ONLY after full expected-pin verification. Rejects > 256 MiB shards
  // fail-closed (use stream_verified_file() / stream_verified_open_fd() for production shards).
  std::string read_verified_file_bytes(const std::filesystem::path& model_dir, std::string_view filename) const;

  // Header inspection (hash_shards=false -> HeadersOnly; hash_shards=true -> SelfHashed).
  // Neither sets VerificationLevel::PinnedVerified nor shard_digests_verified=true.
  static Qwen38Manifest inspect(const std::filesystem::path& model_dir, bool hash_shards = false,
                                const std::string& revision = "");

  // Expected-vs-consumed shard/config verification: hashes all shards on disk and verifies exact
  // filenames, byte sizes, and SHA-256 digests against `expected_pins`. Sets PinnedVerified on pass;
  // throws std::invalid_argument fail-closed on any mismatch.
  static Qwen38Manifest inspect_pinned(const std::filesystem::path& model_dir,
                                       const ingine::model::CheckpointPinSpec& expected_pins,
                                       const std::string& revision = "");
};

// Canonical SHA-256 over all effective execution fields of Qwen38Config (including rms_norm_eps bits,
// RoPE parameters, layer schedule, and quantization parameters).
std::array<std::uint8_t, 32> qwen38_config_fingerprint(const Qwen38Config& config);
std::string qwen38_config_fingerprint_hex(const Qwen38Config& config);

ingine::model::CheckpointPinSpec qwen38_27b_nvfp4_pin_spec();

std::string sha256_bytes_hex(const void* data, std::size_t size);
std::string sha256_file_hex_streaming(const std::filesystem::path& path);
std::array<std::uint8_t, 32> sha256_file_digest(const std::filesystem::path& path);
std::array<std::uint8_t, 32> sha256_bytes_digest(const void* data, std::size_t size);

}  // namespace ingine::backends::qwen38

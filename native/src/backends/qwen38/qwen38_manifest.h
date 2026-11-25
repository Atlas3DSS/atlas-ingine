#pragma once

#include "backends/qwen38/qwen38_config.h"
#include "ingine/model_backend.h"

#include <array>
#include <cstdint>
#include <filesystem>
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

  // Verifies an already-open file descriptor matches the fstat (dev, ino, size) recorded at inspection.
  void verify_open_fd(int fd, std::string_view filename) const;
  // Opens `model_dir / filename` once, verifies FD identity and full-byte SHA-256 (if hashed),
  // and returns exact consumed bytes. Throws std::invalid_argument fail-closed on any mismatch.
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

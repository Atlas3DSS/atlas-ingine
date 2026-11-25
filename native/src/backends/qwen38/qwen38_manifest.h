#pragma once

#include "backends/qwen38/qwen38_config.h"
#include "ingine/model_backend.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace ingine::backends::qwen38 {

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
  std::unordered_map<std::string, std::string> shard_sha256;  // populated when verify_shard_sha256=true
  std::unordered_map<std::string, Qwen38TensorInfo> active_tensors;
  std::size_t total_index_tensors = 0;
  std::size_t active_text_tensor_count = 0;
  std::size_t excluded_visual_tensor_count = 0;
  std::size_t excluded_mtp_tensor_count = 0;
  Qwen38MemoryBudget memory_budget;
  bool shard_payloads_verified = false;

  static Qwen38Manifest inspect(const std::filesystem::path& model_dir, bool verify_shard_sha256 = false,
                                const std::string& revision = "");
};

std::string sha256_bytes_hex(const void* data, std::size_t size);
std::string sha256_file_hex_streaming(const std::filesystem::path& path);
std::array<std::uint8_t, 32> sha256_file_digest(const std::filesystem::path& path);
std::array<std::uint8_t, 32> sha256_bytes_digest(const void* data, std::size_t size);

}  // namespace ingine::backends::qwen38

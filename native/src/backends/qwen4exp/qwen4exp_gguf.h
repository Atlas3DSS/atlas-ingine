// CUDA-free read-only multi-shard GGUF v2/v3 reader, pinned SHA-256 verifier, and GGML dequantizer
// for Qwen3.8-Flash-Next (qwen4exp, #849).
#pragma once

#include "backends/qwen4exp/qwen4exp_config.h"
#include "ingine/model_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

namespace ingine::backends::qwen4exp {

enum class GgmlType : std::uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q5_1 = 7,
  Q8_0 = 8,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  IQ4_NL = 20,
  BF16 = 30,
};

struct GgmlBlockGeometry {
  std::uint64_t block_elems = 0;
  std::uint64_t block_bytes = 0;
};

const char* ggml_type_name(std::uint32_t type_id) noexcept;
bool ggml_block_geometry(std::uint32_t type_id, GgmlBlockGeometry& out) noexcept;

// Exact GGML row dequantizers into float32 (rows have `cols` elements, divisible by block_elems).
void dequantize_row_to_f32(std::uint32_t type_id, std::span<const std::uint8_t> raw_bytes, std::span<float> out_f32);

struct GgufValue {
  using Variant = std::variant<
      std::uint64_t,
      std::int64_t,
      double,
      bool,
      std::string,
      std::vector<std::uint64_t>,
      std::vector<std::int64_t>,
      std::vector<double>,
      std::vector<bool>,
      std::vector<std::string>>;

  Variant value;

  [[nodiscard]] std::int64_t as_i64() const;
  [[nodiscard]] std::uint64_t as_u64() const;
  [[nodiscard]] double as_f64() const;
  [[nodiscard]] bool as_bool() const;
  [[nodiscard]] const std::string& as_string() const;
  [[nodiscard]] std::vector<std::int64_t> as_i64_array() const;
  [[nodiscard]] std::vector<std::uint64_t> as_u64_array() const;
};

struct GgufTensorEntry {
  std::string name;
  std::vector<std::uint64_t> dims;  // GGML order: dims[0] varies fastest (inner/row length)
  std::uint32_t type_id = 0;
  std::uint64_t data_offset = 0;    // absolute byte offset inside shard file
  std::uint64_t byte_size = 0;
  std::size_t shard_index = 0;

  [[nodiscard]] std::uint64_t numel() const noexcept;
  [[nodiscard]] std::uint64_t row_bytes() const noexcept;
  [[nodiscard]] std::uint64_t num_rows() const noexcept;
};

struct GgufShardInfo {
  std::filesystem::path path;
  std::string filename;
  std::uint64_t file_size = 0;
  std::uint64_t data_start = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t kv_count = 0;
  std::uint32_t alignment = 32;
  std::array<std::uint8_t, 32> sha256{};
  bool sha256_verified = false;
  std::shared_ptr<int> cached_fd;
};

class Qwen4ExpGgufSet {
 public:
  // Inspect headers + tensor table only (VerificationLevel::HeadersOnly; zero full-file SHA-256 hashing).
  static Qwen4ExpGgufSet inspect_headers(const std::filesystem::path& path);

  // Inspect headers + compute SHA-256 of every shard + verify exact expected-vs-consumed match against `pin`.
  // Fails closed on any shard count, filename, byte size, or SHA-256 mismatch.
  static Qwen4ExpGgufSet inspect_pinned(const std::filesystem::path& path, const ingine::model::CheckpointPinSpec& pin);

  [[nodiscard]] const Qwen4ExpConfig& config() const noexcept { return config_; }
  [[nodiscard]] const ingine::model::CheckpointIdentity& identity() const noexcept { return identity_; }
  [[nodiscard]] const std::vector<GgufShardInfo>& shards() const noexcept { return shards_; }
  [[nodiscard]] const std::unordered_map<std::string, GgufTensorEntry>& tensors() const noexcept { return tensors_; }
  [[nodiscard]] const std::map<std::string, GgufValue, std::less<>>& metadata() const noexcept { return kv_; }
  [[nodiscard]] std::uint64_t total_tensor_bytes() const noexcept { return total_tensor_bytes_; }
  [[nodiscard]] std::uint64_t ple_table_rows() const noexcept { return ple_table_rows_; }

  [[nodiscard]] const GgufTensorEntry* find_tensor(std::string_view name) const noexcept;
  [[nodiscard]] const GgufTensorEntry& require_tensor(std::string_view name) const;

  // Reads raw bytes of an entire tensor via POSIX pread (bounds-checked).
  std::vector<std::uint8_t> read_tensor_raw(const GgufTensorEntry& entry) const;
  // Reads and dequantizes an entire tensor into float32.
  std::vector<float> read_tensor_f32(const GgufTensorEntry& entry) const;
  // Reads and dequantizes a single row [row_index, dims[0]] (or 2D slice of a 3D expert tensor) via POSIX pread.
  // Essential for O(1) sparse gather from 54 GB per_layer_token_embd.weight or single-expert reads!
  void read_tensor_row_f32(const GgufTensorEntry& entry, std::uint64_t row_index, std::span<float> out_f32) const;
  // Reads and dequantizes a contiguous slice of `count` rows starting at `first_row` via one POSIX pread.
  std::vector<float> read_tensor_rows_f32(const GgufTensorEntry& entry, std::uint64_t first_row,
                                          std::uint64_t count) const;
  // Reads raw quantized bytes for `count` contiguous rows starting at `first_row` via POSIX pread.
  void read_tensor_rows_raw(const GgufTensorEntry& entry, std::uint64_t first_row, std::uint64_t count,
                            std::span<std::uint8_t> out_raw) const;

 private:
  static Qwen4ExpGgufSet load_impl(const std::filesystem::path& path,
                                   const ingine::model::CheckpointPinSpec* pin);
  void parse_config_from_kv();
  void validate_qwen4exp_tensor_schema();
  void compute_manifest_digest();

  Qwen4ExpConfig config_;
  ingine::model::CheckpointIdentity identity_;
  std::vector<GgufShardInfo> shards_;
  std::map<std::string, GgufValue, std::less<>> kv_;
  std::unordered_map<std::string, GgufTensorEntry> tensors_;
  std::uint64_t total_tensor_bytes_ = 0;
  std::uint64_t ple_table_rows_ = 0;
};

// Pinned SHA-256 spec for unsloth/Qwen3.8-Flash-Next-GGUF @ 766911a6 UD-Q6_K_XL (6 shards, 169.16 GB).
ingine::model::CheckpointPinSpec qwen4exp_ud_q6_k_xl_pin_spec();

// Standalone FIPS 180-4 SHA-256 helpers (also used by unit tests).
std::array<std::uint8_t, 32> sha256_bytes(const void* data, std::size_t len) noexcept;
std::array<std::uint8_t, 32> sha256_file(const std::filesystem::path& path);
std::string sha256_to_hex(const std::array<std::uint8_t, 32>& digest);

}  // namespace ingine::backends::qwen4exp

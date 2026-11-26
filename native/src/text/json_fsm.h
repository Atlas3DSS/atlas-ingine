// Token-level RFC 8259 JSON and JSON-Schema pushdown automaton for constrained decoding (CUDA-free).
#pragma once

#include "ingine/token_masker.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::text {

enum class JsonFormatMode : std::uint8_t {
  None = 0,
  JsonObject = 1,
  JsonSchema = 2,
};

struct JsonFormatSpec {
  JsonFormatMode mode = JsonFormatMode::None;
  nlohmann::ordered_json schema;  // populated when mode == JsonSchema (optional if raw_schema_json is set)
  std::string raw_schema_json;    // verbatim wire JSON bytes for exact decimal precision without binary64 rounding
};

// Extracts the verbatim JSON substring at `path` (e.g. {"response_format", "json_schema", "schema"}) from raw wire
// JSON bytes without parsing numbers through binary64 double.
std::optional<std::string> extract_raw_json_field_path(std::string_view raw_json,
                                                       std::initializer_list<std::string_view> path);

// Precomputed per-tokenizer byte index shared across all constrained requests (thread-safe immutable).
class VocabByteIndex {
 public:
  static std::shared_ptr<const VocabByteIndex> build(const Tokenizer& tokenizer,
                                                     std::span<const std::int32_t> stop_ids);

  int vocab_size() const noexcept { return vocab_size_; }
  std::size_t mask_words() const noexcept { return mask_words_; }

  std::int32_t content_text_id() const noexcept { return content_text_id_; }
  std::int32_t end_message_id() const noexcept { return end_message_id_; }
  std::int32_t end_sampling_id() const noexcept { return end_sampling_id_; }
  std::int32_t endoftext_id() const noexcept { return endoftext_id_; }

  const std::string& token_bytes(std::int32_t id) const noexcept { return bytes_[std::size_t(id)]; }
  bool is_special(std::int32_t id) const noexcept { return is_special_[std::size_t(id)] != 0; }
  const std::vector<std::int32_t>& stop_ids() const noexcept { return stop_ids_; }

  // Fast precomputed bitsets & first-byte buckets.
  const std::vector<std::uint32_t>& ws_mask() const noexcept { return ws_mask_; }
  const std::vector<std::uint32_t>& string_safe_mask() const noexcept { return string_safe_mask_; }
  const std::vector<std::int32_t>& string_special_candidates() const noexcept { return string_special_candidates_; }
  const std::vector<std::int32_t>& bucket(std::uint8_t first_byte) const noexcept { return by_first_byte_[first_byte]; }

 private:
  int vocab_size_ = 0;
  std::size_t mask_words_ = 0;
  std::int32_t content_text_id_ = -1;
  std::int32_t end_message_id_ = -1;
  std::int32_t end_sampling_id_ = -1;
  std::int32_t endoftext_id_ = -1;
  std::vector<std::int32_t> stop_ids_;
  std::vector<std::string> bytes_;
  std::vector<std::uint8_t> is_special_;
  std::vector<std::uint32_t> ws_mask_;
  std::vector<std::uint32_t> string_safe_mask_;
  std::vector<std::int32_t> string_special_candidates_;
  std::array<std::vector<std::int32_t>, 256> by_first_byte_;
};

// Compiles a JSON Schema into an immutable node graph and creates a per-request TokenMasker.
// Channel-aware for Inkling-2: stays inactive while the model emits <|content_thinking|>...
// and activates automatically upon entering <|content_text|> (or immediately if prefill_ids
// already entered <|content_text|>).
std::shared_ptr<model::TokenMasker> make_json_fsm(std::shared_ptr<const VocabByteIndex> vocab,
                                                  const JsonFormatSpec& spec,
                                                  std::span<const std::int32_t> prefill_ids);

}  // namespace ingine::text

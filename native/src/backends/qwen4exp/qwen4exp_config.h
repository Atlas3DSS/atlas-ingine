// Qwen3.8-Flash-Next (GGUF general.architecture == "qwen4exp") execution config & validator (#849).
// Strictly CUDA-free host header.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ingine::backends::qwen4exp {

enum class Qwen4ExpLayerKind : std::uint8_t {
  LinearAttention = 0,
  FullAttention = 1,
};

const char* to_string(Qwen4ExpLayerKind kind) noexcept;

struct Qwen4ExpConfig {
  std::string architecture = "qwen4exp";
  std::string model_name;
  int vocab_size = 0;
  int hidden_size = 0;         // embedding_length (2560)
  int num_hidden_layers = 0;   // block_count (48)
  int max_position_embeddings = 0;  // context_length (262144)
  float rms_norm_eps = 1e-6f;

  // Hyper-Connections (low-rank multi-stream residual mixer).
  int hc_mult = 0;      // hyper_connection.count (4)
  int hc_low_rank = 0;  // hyper_connection.low_rank (320)

  // Full attention + partial IMRoPE / NeoX + QSA block-sparse indexer.
  int full_attention_interval = 4;
  int num_attention_heads = 0;      // 24
  int num_key_value_heads = 0;      // 2
  int head_dim = 0;                 // 256 (key_length == value_length)
  float rope_theta = 10000000.0f;
  int rotary_dim = 64;
  std::array<int, 4> rope_sections{11, 11, 10, 0};

  int indexer_head_count = 0;       // 4
  int indexer_key_length = 0;       // 128
  int indexer_top_k = 0;            // 2048
  std::vector<int> compress_ratios; // length == num_hidden_layers (4 on full-attn layers, 0 elsewhere)

  // Gated DeltaNet (linear attention; sigmoid output gate vs Qwen3.5 silu).
  int ssm_conv_kernel = 4;
  int ssm_state_size = 128;         // head_k_dim == head_v_dim
  int ssm_group_count = 16;         // num_k_heads
  int ssm_time_step_rank = 48;      // num_v_heads
  int ssm_inner_size = 6144;        // num_v_heads * head_v_dim

  // MoE + sigmoid-gated shared expert.
  int expert_count = 0;             // 512
  int expert_used_count = 0;        // 10
  int expert_ffn_length = 0;        // 640
  int shared_expert_ffn_length = 0; // 640
  float expert_weights_scale = 1.0f;

  // Per-Layer N-Gram Embeddings (PLE / Engram hash table).
  std::vector<int> ple_layers;      // e.g. {1}
  int ple_ngram_size = 0;           // 3
  int ple_heads_per_ngram = 0;      // 8
  int ple_conv_kernel = 0;          // 4
  int ple_head_dim = 0;             // 160 (embedding_length_per_layer_input)
  std::int32_t ple_eos_token_id = 248044;
  std::int32_t ple_image_token_id = 0;
  std::vector<std::uint64_t> ple_layer_multipliers;
  std::vector<std::uint64_t> ple_head_offsets;
  std::vector<std::uint64_t> ple_head_vocab_sizes;

  // Derived layer schedule.
  std::vector<Qwen4ExpLayerKind> layer_types;
  std::vector<std::uint8_t> is_ple_layer;
  int num_linear_layers = 0;
  int num_full_layers = 0;

  [[nodiscard]] int hc_dim() const noexcept { return hc_mult * hidden_size; }
  [[nodiscard]] int gdn_key_dim() const noexcept { return ssm_group_count * ssm_state_size; }
  [[nodiscard]] int gdn_value_dim() const noexcept { return ssm_time_step_rank * ssm_state_size; }
  [[nodiscard]] int gdn_conv_dim() const noexcept { return gdn_key_dim() * 2 + gdn_value_dim(); }
  [[nodiscard]] int full_q_out_dim() const noexcept { return num_attention_heads * head_dim * 2; }
  [[nodiscard]] int full_kv_out_dim() const noexcept { return num_key_value_heads * head_dim; }
  [[nodiscard]] int ple_n_heads() const noexcept {
    return (ple_ngram_size >= 2 && ple_heads_per_ngram > 0) ? (ple_ngram_size - 1) * ple_heads_per_ngram : 0;
  }
  [[nodiscard]] int ple_conv_history_cols() const noexcept {
    return (ple_conv_kernel > 1 && ple_ngram_size > 0) ? (ple_conv_kernel - 1) * ple_ngram_size : 0;
  }
  [[nodiscard]] bool is_ple(int layer) const noexcept {
    return layer >= 0 && static_cast<std::size_t>(layer) < is_ple_layer.size() && is_ple_layer[layer] != 0;
  }
  [[nodiscard]] bool is_linear(int layer) const noexcept {
    return layer >= 0 && static_cast<std::size_t>(layer) < layer_types.size() &&
           layer_types[layer] == Qwen4ExpLayerKind::LinearAttention;
  }
  [[nodiscard]] bool has_qsa(int layer) const noexcept {
    return layer >= 0 && static_cast<std::size_t>(layer) < compress_ratios.size() && compress_ratios[layer] > 0 &&
           indexer_head_count > 0 && indexer_key_length > 0 && indexer_top_k > 0;
  }

  // Canonical fingerprint string bound into BackendStateIdentity::family_state_fingerprint.
  [[nodiscard]] std::string canonical_fingerprint() const;

  void finalize_and_validate();
};

}  // namespace ingine::backends::qwen4exp

#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::backends::qwen38 {

enum class Qwen38LayerKind : std::uint8_t {
  LinearAttention = 0,
  FullAttention = 1,
};

enum class Qwen38WeightPrecision : std::uint8_t {
  Bf16 = 0,
  MixedFp8Nvfp4 = 1,
};

enum class Qwen38LinearQuantKind : std::uint8_t {
  Bf16 = 0,
  Fp8E4M3 = 1,
  Nvfp4 = 2,
};

const char* to_string(Qwen38LayerKind kind) noexcept;
const char* to_string(Qwen38WeightPrecision prec) noexcept;
const char* to_string(Qwen38LinearQuantKind quant) noexcept;

struct Qwen38Config {
  std::string model_type;
  std::vector<std::string> architectures;
  int vocab_size = 0;
  int hidden_size = 0;
  int intermediate_size = 0;
  int num_hidden_layers = 0;
  int max_position_embeddings = 0;
  float rms_norm_eps = 1e-6f;
  bool tie_word_embeddings = false;

  // Full attention + partial NeoX RoPE.
  int full_attention_interval = 4;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  bool attention_bias = false;
  bool attn_output_gate = true;
  float rope_theta = 10000000.0f;
  float partial_rotary_factor = 0.25f;
  int rotary_dim = 0;
  std::array<int, 3> mrope_section{0, 0, 0};
  bool mrope_interleaved = false;

  // Gated DeltaNet (linear attention).
  int linear_conv_kernel_dim = 4;
  int linear_key_head_dim = 0;
  int linear_value_head_dim = 0;
  int linear_num_key_heads = 0;
  int linear_num_value_heads = 0;
  std::string mamba_ssm_dtype = "float32";

  // Layer schedule.
  std::vector<Qwen38LayerKind> layer_types;
  int num_linear_layers = 0;
  int num_full_layers = 0;

  // Quantization metadata (BF16 or ModelOpt mixed FP8/NVFP4).
  Qwen38WeightPrecision weight_precision = Qwen38WeightPrecision::Bf16;
  std::string quant_method;
  std::string quant_algo;
  int nvfp4_group_size = 16;

  [[nodiscard]] int linear_key_dim() const noexcept { return linear_num_key_heads * linear_key_head_dim; }
  [[nodiscard]] int linear_value_dim() const noexcept { return linear_num_value_heads * linear_value_head_dim; }
  [[nodiscard]] int linear_conv_dim() const noexcept { return linear_key_dim() * 2 + linear_value_dim(); }
  [[nodiscard]] int full_q_proj_out_dim() const noexcept {
    return num_attention_heads * head_dim * (attn_output_gate ? 2 : 1);
  }
  [[nodiscard]] int full_kv_proj_out_dim() const noexcept { return num_key_value_heads * head_dim; }

  [[nodiscard]] Qwen38LinearQuantKind linear_attn_quant() const noexcept {
    return weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4 ? Qwen38LinearQuantKind::Fp8E4M3
                                                                   : Qwen38LinearQuantKind::Bf16;
  }
  [[nodiscard]] Qwen38LinearQuantKind full_attn_quant() const noexcept {
    return weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4 ? Qwen38LinearQuantKind::Fp8E4M3
                                                                   : Qwen38LinearQuantKind::Bf16;
  }
  [[nodiscard]] Qwen38LinearQuantKind mlp_quant() const noexcept {
    return weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4 ? Qwen38LinearQuantKind::Nvfp4
                                                                   : Qwen38LinearQuantKind::Bf16;
  }
  [[nodiscard]] Qwen38LinearQuantKind lm_head_quant() const noexcept {
    return weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4 ? Qwen38LinearQuantKind::Nvfp4
                                                                   : Qwen38LinearQuantKind::Bf16;
  }

  void validate() const;

  static Qwen38Config load(const std::filesystem::path& config_path);
  static Qwen38Config from_json_str(std::string_view json_text);
};

}  // namespace ingine::backends::qwen38

#include "backends/qwen38/qwen38_config.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ingine::backends::qwen38 {
namespace {

int read_required_int(const nlohmann::json& obj, const char* key) {
  auto it = obj.find(key);
  if (it == obj.end() || !it->is_number_integer()) {
    throw std::invalid_argument(std::string("qwen38 config missing integer field '") + key + "'");
  }
  return it->get<int>();
}

float read_required_float(const nlohmann::json& obj, const char* key) {
  auto it = obj.find(key);
  if (it == obj.end() || !it->is_number()) {
    throw std::invalid_argument(std::string("qwen38 config missing numeric field '") + key + "'");
  }
  return it->get<float>();
}

bool read_bool_or(const nlohmann::json& obj, const char* key, bool def_val) {
  auto it = obj.find(key);
  if (it == obj.end()) return def_val;
  if (!it->is_boolean()) {
    throw std::invalid_argument(std::string("qwen38 config field '") + key + "' must be boolean");
  }
  return it->get<bool>();
}

void validate_quant_group_entry(const std::string& name, const nlohmann::json& group_obj, int expected_nvfp4_group) {
  if (!group_obj.is_object()) {
    throw std::invalid_argument("quantization_config.config_groups entry '" + name + "' must be object");
  }
  const std::string algo = group_obj.value("quant_algo", std::string{});
  const bool is_linear_or_full =
      name.find(".linear_attn.in_proj_qkv") != std::string::npos ||
      name.find(".linear_attn.in_proj_z") != std::string::npos ||
      name.find(".linear_attn.out_proj") != std::string::npos ||
      name.find(".self_attn.q_proj") != std::string::npos ||
      name.find(".self_attn.k_proj") != std::string::npos ||
      name.find(".self_attn.v_proj") != std::string::npos ||
      name.find(".self_attn.o_proj") != std::string::npos;
  const bool is_mlp_or_head =
      name.find(".mlp.gate_proj") != std::string::npos ||
      name.find(".mlp.up_proj") != std::string::npos ||
      name.find(".mlp.down_proj") != std::string::npos ||
      name == "lm_head";

  if (is_linear_or_full) {
    if (algo != "FP8") {
      throw std::invalid_argument("expected quant_algo 'FP8' for '" + name + "', got '" + algo + "'");
    }
    return;
  }
  if (is_mlp_or_head) {
    if (algo != "NVFP4") {
      throw std::invalid_argument("expected quant_algo 'NVFP4' for '" + name + "', got '" + algo + "'");
    }
    const int gs = group_obj.value("group_size", 0);
    if (gs != expected_nvfp4_group) {
      throw std::invalid_argument("expected NVFP4 group_size " + std::to_string(expected_nvfp4_group) + " for '" +
                                  name + "', got " + std::to_string(gs));
    }
    return;
  }
}

}  // namespace

const char* to_string(Qwen38LayerKind kind) noexcept {
  switch (kind) {
    case Qwen38LayerKind::LinearAttention: return "linear_attention";
    case Qwen38LayerKind::FullAttention: return "full_attention";
    default: return "unknown";
  }
}

const char* to_string(Qwen38WeightPrecision prec) noexcept {
  switch (prec) {
    case Qwen38WeightPrecision::Bf16: return "bf16";
    case Qwen38WeightPrecision::MixedFp8Nvfp4: return "mixed_fp8_nvfp4";
    default: return "unknown";
  }
}

const char* to_string(Qwen38LinearQuantKind quant) noexcept {
  switch (quant) {
    case Qwen38LinearQuantKind::Bf16: return "bf16";
    case Qwen38LinearQuantKind::Fp8E4M3: return "fp8_e4m3";
    case Qwen38LinearQuantKind::Nvfp4: return "nvfp4";
    default: return "unknown";
  }
}

void Qwen38Config::validate() const {
  const bool valid_model_type = (model_type == "qwen3_5" || model_type == "qwen3_5_text");
  if (!valid_model_type) {
    throw std::invalid_argument("unsupported qwen38 model_type: '" + model_type + "'");
  }
  if (vocab_size <= 0 || hidden_size <= 0 || intermediate_size <= 0 || num_hidden_layers <= 0 ||
      max_position_embeddings <= 0) {
    throw std::invalid_argument("qwen38 config dimensions must be strictly positive");
  }
  if (!(rms_norm_eps > 0.0f && rms_norm_eps < 1.0f)) {
    throw std::invalid_argument("qwen38 config rms_norm_eps out of range");
  }
  if (full_attention_interval <= 0 || num_attention_heads <= 0 || num_key_value_heads <= 0 || head_dim <= 0) {
    throw std::invalid_argument("qwen38 config attention dimensions must be strictly positive");
  }
  if (num_attention_heads % num_key_value_heads != 0) {
    throw std::invalid_argument("qwen38 num_attention_heads must be divisible by num_key_value_heads");
  }
  if (head_dim % 2 != 0 || rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0) {
    throw std::invalid_argument("qwen38 invalid head_dim/rotary_dim");
  }
  if (!(rope_theta > 1.0f)) {
    throw std::invalid_argument("qwen38 rope_theta must be > 1.0");
  }
  if (attention_bias) {
    throw std::invalid_argument("qwen38 attention_bias=true is unsupported");
  }
  if (linear_conv_kernel_dim != 4) {
    throw std::invalid_argument("qwen38 only supports linear_conv_kernel_dim=4");
  }
  if (linear_key_head_dim <= 0 || linear_value_head_dim <= 0 || linear_num_key_heads <= 0 ||
      linear_num_value_heads <= 0) {
    throw std::invalid_argument("qwen38 linear_attn dimensions must be strictly positive");
  }
  if (linear_num_value_heads % linear_num_key_heads != 0) {
    throw std::invalid_argument("qwen38 linear_num_value_heads must be divisible by linear_num_key_heads");
  }
  if (mamba_ssm_dtype != "float32") {
    throw std::invalid_argument("qwen38 requires mamba_ssm_dtype == 'float32', got '" + mamba_ssm_dtype + "'");
  }
  if (static_cast<int>(layer_types.size()) != num_hidden_layers) {
    throw std::invalid_argument("qwen38 layer_types length must equal num_hidden_layers");
  }
  if (weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4) {
    if (quant_method != "modelopt" || quant_algo != "MIXED_PRECISION") {
      throw std::invalid_argument("qwen38 mixed quantization requires quant_method=modelopt and quant_algo=MIXED_PRECISION");
    }
    if (nvfp4_group_size != 16) {
      throw std::invalid_argument("qwen38 NVFP4 requires group_size=16");
    }
    if (hidden_size % nvfp4_group_size != 0 || intermediate_size % nvfp4_group_size != 0) {
      throw std::invalid_argument("qwen38 NVFP4 requires hidden_size and intermediate_size divisible by group_size 16");
    }
  }
}

Qwen38Config Qwen38Config::load(const std::filesystem::path& config_path) {
  std::ifstream in(config_path, std::ios::binary);
  if (!in) {
    throw std::invalid_argument("cannot open qwen38 config.json: " + config_path.string());
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return from_json_str(ss.str());
}

Qwen38Config Qwen38Config::from_json_str(std::string_view json_text) {
  nlohmann::json root;
  try {
    root = nlohmann::json::parse(json_text);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument(std::string("invalid qwen38 config JSON: ") + e.what());
  }
  if (!root.is_object()) {
    throw std::invalid_argument("qwen38 config root must be a JSON object");
  }

  Qwen38Config cfg;
  if (auto it = root.find("architectures"); it != root.end()) {
    if (!it->is_array()) throw std::invalid_argument("qwen38 config 'architectures' must be an array");
    for (const auto& entry : *it) {
      if (!entry.is_string()) throw std::invalid_argument("qwen38 architecture entries must be strings");
      cfg.architectures.push_back(entry.get<std::string>());
    }
  }

  const bool has_text_cfg = root.contains("text_config");
  const nlohmann::json& text = has_text_cfg ? root.at("text_config") : root;
  if (!text.is_object()) {
    throw std::invalid_argument("qwen38 text_config must be a JSON object");
  }

  if (root.contains("model_type") && root["model_type"].is_string()) {
    cfg.model_type = root["model_type"].get<std::string>();
  } else if (text.contains("model_type") && text["model_type"].is_string()) {
    cfg.model_type = text["model_type"].get<std::string>();
  }

  if (!cfg.architectures.empty()) {
    const std::string& arch0 = cfg.architectures.front();
    if (arch0 != "Qwen3_5ForConditionalGeneration" && arch0 != "Qwen3_5ForCausalLM") {
      throw std::invalid_argument("unsupported qwen38 architecture: '" + arch0 + "'");
    }
  }

  cfg.vocab_size = read_required_int(text, "vocab_size");
  cfg.hidden_size = read_required_int(text, "hidden_size");
  cfg.intermediate_size = read_required_int(text, "intermediate_size");
  cfg.num_hidden_layers = read_required_int(text, "num_hidden_layers");
  cfg.max_position_embeddings = read_required_int(text, "max_position_embeddings");
  cfg.rms_norm_eps = read_required_float(text, "rms_norm_eps");

  const bool root_tie = read_bool_or(root, "tie_word_embeddings", false);
  const bool text_tie = read_bool_or(text, "tie_word_embeddings", root_tie);
  if (root.contains("tie_word_embeddings") && text.contains("tie_word_embeddings") && root_tie != text_tie) {
    throw std::invalid_argument("conflicting tie_word_embeddings between root and text_config");
  }
  cfg.tie_word_embeddings = text_tie;

  cfg.full_attention_interval = text.contains("full_attention_interval") ? read_required_int(text, "full_attention_interval") : 4;
  cfg.num_attention_heads = read_required_int(text, "num_attention_heads");
  cfg.num_key_value_heads = read_required_int(text, "num_key_value_heads");
  cfg.head_dim = read_required_int(text, "head_dim");
  cfg.attention_bias = read_bool_or(text, "attention_bias", false);
  cfg.attn_output_gate = read_bool_or(text, "attn_output_gate", true);

  const nlohmann::json& rope = text.contains("rope_parameters") ? text.at("rope_parameters") : text;
  if (!rope.is_object()) {
    throw std::invalid_argument("qwen38 rope_parameters must be an object");
  }
  cfg.rope_theta = read_required_float(rope, "rope_theta");
  cfg.partial_rotary_factor = rope.contains("partial_rotary_factor") ? read_required_float(rope, "partial_rotary_factor") : 0.25f;
  cfg.rotary_dim = static_cast<int>(std::llround(static_cast<double>(cfg.head_dim) * cfg.partial_rotary_factor));
  cfg.mrope_interleaved = read_bool_or(rope, "mrope_interleaved", false);
  if (auto ms = rope.find("mrope_section"); ms != rope.end()) {
    if (!ms->is_array() || ms->size() != 3) {
      throw std::invalid_argument("qwen38 mrope_section must be a 3-element integer array");
    }
    for (std::size_t i = 0; i < 3; ++i) {
      if (!(*ms)[i].is_number_integer()) throw std::invalid_argument("qwen38 mrope_section entry must be int");
      cfg.mrope_section[i] = (*ms)[i].get<int>();
    }
  }

  cfg.linear_conv_kernel_dim = read_required_int(text, "linear_conv_kernel_dim");
  cfg.linear_key_head_dim = read_required_int(text, "linear_key_head_dim");
  cfg.linear_value_head_dim = read_required_int(text, "linear_value_head_dim");
  cfg.linear_num_key_heads = read_required_int(text, "linear_num_key_heads");
  cfg.linear_num_value_heads = read_required_int(text, "linear_num_value_heads");
  cfg.mamba_ssm_dtype = text.value("mamba_ssm_dtype", std::string("float32"));

  if (auto lt = text.find("layer_types"); lt != text.end()) {
    if (!lt->is_array()) throw std::invalid_argument("qwen38 layer_types must be an array");
    cfg.layer_types.reserve(lt->size());
    for (const auto& entry : *lt) {
      if (!entry.is_string()) throw std::invalid_argument("qwen38 layer_types elements must be strings");
      const std::string s = entry.get<std::string>();
      if (s == "linear_attention") {
        cfg.layer_types.push_back(Qwen38LayerKind::LinearAttention);
        ++cfg.num_linear_layers;
      } else if (s == "full_attention") {
        cfg.layer_types.push_back(Qwen38LayerKind::FullAttention);
        ++cfg.num_full_layers;
      } else {
        throw std::invalid_argument("unknown qwen38 layer_type: '" + s + "'");
      }
    }
  } else {
    for (int i = 0; i < cfg.num_hidden_layers; ++i) {
      const bool is_full = ((i + 1) % cfg.full_attention_interval) == 0;
      cfg.layer_types.push_back(is_full ? Qwen38LayerKind::FullAttention : Qwen38LayerKind::LinearAttention);
      if (is_full) {
        ++cfg.num_full_layers;
      } else {
        ++cfg.num_linear_layers;
      }
    }
  }

  if (auto qc = root.find("quantization_config"); qc != root.end() && !qc->is_null()) {
    if (!qc->is_object()) throw std::invalid_argument("qwen38 quantization_config must be an object");
    cfg.weight_precision = Qwen38WeightPrecision::MixedFp8Nvfp4;
    cfg.quant_method = qc->value("quant_method", std::string{});
    cfg.quant_algo = qc->value("quant_algo", std::string{});
    cfg.nvfp4_group_size = 16;

    if (auto cg = qc->find("config_groups"); cg != qc->end()) {
      if (!cg->is_object()) throw std::invalid_argument("qwen38 config_groups must be an object");
      for (auto it = cg->begin(); it != cg->end(); ++it) {
        validate_quant_group_entry(it.key(), it.value(), cfg.nvfp4_group_size);
      }
    }
  } else {
    cfg.weight_precision = Qwen38WeightPrecision::Bf16;
  }

  cfg.validate();
  return cfg;
}

}  // namespace ingine::backends::qwen38

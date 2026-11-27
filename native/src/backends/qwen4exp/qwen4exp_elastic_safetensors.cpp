#include "backends/qwen4exp/qwen4exp_elastic_safetensors.h"
#include "backends/qwen4exp/qwen4exp_gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>

namespace ingine::backends::qwen4exp {
namespace {

constexpr float kE2M1Host[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

float decode_e4m3fn_host(std::uint8_t b) noexcept {
  const std::uint32_t s = (b >> 7) & 1u;
  const std::uint32_t e = (b >> 3) & 0xfu;
  const std::uint32_t m = b & 0x7u;
  float v = 0.0f;
  if (e == 0) {
    v = std::ldexp(static_cast<float>(m), -9);
  } else if (e == 15 && m == 7) {
    return 0.0f;
  } else {
    v = std::ldexp(static_cast<float>(8u + m), static_cast<int>(e) - 10);
  }
  return s ? -v : v;
}

float bf16_to_f32_host(std::uint16_t b) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b, const char* what) {
  if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
    throw std::overflow_error(std::string("qwen4exp elastic: integer overflow in ") + what);
  }
  return a * b;
}

std::uint64_t checked_add_u64(std::uint64_t a, std::uint64_t b, const char* what) {
  if (a > std::numeric_limits<std::uint64_t>::max() - b) {
    throw std::overflow_error(std::string("qwen4exp elastic: integer overflow in ") + what);
  }
  return a + b;
}

std::filesystem::path validate_shard_path(const std::filesystem::path& dir,
                                          const std::filesystem::path& canonical_dir,
                                          const std::string& shard_name) {
  if (shard_name.empty() || shard_name == "." || shard_name == ".." ||
      shard_name.find('/') != std::string::npos || shard_name.find('\\') != std::string::npos ||
      shard_name.find('\0') != std::string::npos) {
    throw std::invalid_argument("invalid elastic shard filename: '" + shard_name + "'");
  }
  const std::filesystem::path rel(shard_name);
  if (rel.is_absolute() || rel.has_parent_path() || rel.filename().string() != shard_name) {
    throw std::invalid_argument("elastic shard path must be a plain filename: '" + shard_name + "'");
  }
  const std::filesystem::path sp = dir / rel;
  std::error_code ec;
  const auto st = std::filesystem::symlink_status(sp, ec);
  if (ec || !std::filesystem::exists(sp, ec)) {
    throw std::invalid_argument("elastic shard does not exist: " + sp.string());
  }
  if (std::filesystem::is_symlink(st)) {
    const auto canon_sp = std::filesystem::canonical(sp, ec);
    if (ec || canon_sp.parent_path() != canonical_dir) {
      throw std::invalid_argument("elastic shard symlink escapes checkpoint directory: " + shard_name);
    }
  }
  if (!std::filesystem::is_regular_file(sp, ec) || ec) {
    throw std::invalid_argument("elastic shard is not a regular file: " + sp.string());
  }
  return sp;
}

ElasticTensorKind parse_algo(const std::string& algo, const std::string& layer_name) {
  if (algo == "FP8") return ElasticTensorKind::Fp8;
  if (algo == "NVFP4") return ElasticTensorKind::Nvfp4;
  if (algo == "ATLAS3") return ElasticTensorKind::Atlas3;
  if (algo == "ATLAS2") return ElasticTensorKind::Atlas2;
  if (algo == "BF16") return ElasticTensorKind::Bf16;
  if (algo == "F32") return ElasticTensorKind::F32;
  throw std::invalid_argument("unsupported quant_algo '" + algo + "' for layer '" + layer_name + "'");
}

bool is_packed_codebook_kind(ElasticTensorKind kind) noexcept {
  return kind == ElasticTensorKind::Nvfp4 || kind == ElasticTensorKind::Atlas3 || kind == ElasticTensorKind::Atlas2;
}

std::string strip_weight_suffix(const std::string& name) {
  if (name.size() > 7 && name.ends_with(".weight")) {
    return name.substr(0, name.size() - 7);
  }
  return name;
}

std::uint64_t packed_code_width_for_kind(ElasticTensorKind kind, int in_features, const std::string& ctx) {
  if (in_features <= 0) {
    throw std::invalid_argument(ctx + ": in_features must be positive");
  }
  const std::uint64_t in_u = static_cast<std::uint64_t>(in_features);
  if (kind == ElasticTensorKind::Nvfp4) {
    if ((in_u % 16u) != 0) {
      throw std::invalid_argument(ctx + ": NVFP4 in_features (" + std::to_string(in_features) +
                                  ") must be divisible by 16");
    }
    return in_u / 2u;
  }
  if (kind == ElasticTensorKind::Atlas3) {
    // ATLAS3 kernel packs 32 elements into 12-byte pairs (lo0[4B], lo1[4B], hi01[4B]).
    if ((in_u % 32u) != 0) {
      throw std::invalid_argument(ctx + ": ATLAS3 in_features (" + std::to_string(in_features) +
                                  ") must be divisible by 32");
    }
    return checked_mul_u64(in_u / 8u, 3u, "atlas3 packed width");
  }
  if (kind == ElasticTensorKind::Atlas2) {
    if ((in_u % 16u) != 0) {
      throw std::invalid_argument(ctx + ": ATLAS2 in_features (" + std::to_string(in_features) +
                                  ") must be divisible by 16");
    }
    return in_u / 4u;
  }
  throw std::invalid_argument(ctx + ": expected packed codebook format (NVFP4/ATLAS3/ATLAS2)");
}

void validate_scalar_entry(const SafetensorsEntry& e, const std::string& ctx) {
  if (e.dtype != "F32" && e.dtype != "BF16") {
    throw std::invalid_argument(ctx + ": scalar '" + e.name + "' must be F32 or BF16, got '" + e.dtype + "'");
  }
  const bool valid_shape = e.shape.empty() || (e.shape.size() == 1 && e.shape[0] == 1);
  if (!valid_shape) {
    throw std::invalid_argument(ctx + ": scalar '" + e.name + "' must have shape [] or [1]");
  }
  const std::uint64_t elem_bytes = (e.dtype == "F32") ? 4u : 2u;
  if (e.byte_size != elem_bytes) {
    throw std::invalid_argument(ctx + ": scalar '" + e.name + "' byte size mismatch (" +
                                std::to_string(e.byte_size) + " != " + std::to_string(elem_bytes) + ")");
  }
}

}  // namespace

bool Qwen4ExpElasticSet::is_elastic_dir(const std::filesystem::path& path) {
  return std::filesystem::is_directory(path) && std::filesystem::is_regular_file(path / "hf_quant_config.json") &&
         std::filesystem::is_regular_file(path / "model.safetensors.index.json") &&
         std::filesystem::is_regular_file(path / "config.json");
}

Qwen4ExpElasticSet Qwen4ExpElasticSet::inspect_headers(const std::filesystem::path& dir) {
  if (!is_elastic_dir(dir)) {
    throw std::invalid_argument("not a Qwen4Exp Elastic safetensors directory: " + dir.string());
  }
  auto read_json = [](const std::filesystem::path& p) {
    std::ifstream in(p);
    if (!in) throw std::invalid_argument("cannot open " + p.string());
    return nlohmann::json::parse(in);
  };

  const auto cfg_j = read_json(dir / "config.json");
  const auto qcfg_j = read_json(dir / "hf_quant_config.json");
  const auto idx_j = read_json(dir / "model.safetensors.index.json");

  if (!cfg_j.is_object()) {
    throw std::invalid_argument("config.json is not a JSON object");
  }
  if (!qcfg_j.is_object() || !qcfg_j.contains("atlas_elastic") || !qcfg_j["atlas_elastic"].is_object()) {
    throw std::invalid_argument("hf_quant_config.json missing atlas_elastic section");
  }
  if (!idx_j.is_object() || !idx_j.contains("weight_map") || !idx_j["weight_map"].is_object() ||
      idx_j["weight_map"].empty()) {
    throw std::invalid_argument("model.safetensors.index.json missing non-empty weight_map object");
  }

  const auto& ae = qcfg_j["atlas_elastic"];
  if (ae.value("family", std::string{}) != "qwen4exp") {
    throw std::invalid_argument("hf_quant_config.json atlas_elastic.family != qwen4exp");
  }

  Qwen4ExpElasticSet set;
  set.profile_ = ae.value("profile", std::string("elastic"));
  if (ae.contains("ple")) {
    if (!ae["ple"].is_object()) {
      throw std::invalid_argument("hf_quant_config.json atlas_elastic.ple must be an object");
    }
    set.ple_format_ = ae["ple"].value("format", std::string("external"));
    set.ple_num_shards_ = ae["ple"].value("num_shards", 0);
    set.ple_chunk_rows_ = ae["ple"].value("chunk_rows", 2000000ull);
    if (set.ple_num_shards_ < 0) {
      throw std::invalid_argument("hf_quant_config.json atlas_elastic.ple.num_shards must be >= 0");
    }
    if (set.ple_format_ != "external" && set.ple_format_ != "nvfp4" && set.ple_format_ != "q8_0" &&
        set.ple_format_ != "f32") {
      throw std::invalid_argument("unsupported atlas_elastic.ple.format: '" + set.ple_format_ + "'");
    }
    if (set.ple_format_ == "external" && set.ple_num_shards_ > 0) {
      throw std::invalid_argument("hf_quant_config.json atlas_elastic.ple.format == external with num_shards > 0");
    }
    if (set.ple_num_shards_ > 0 && set.ple_format_ != "external" && set.ple_chunk_rows_ == 0) {
      throw std::invalid_argument("hf_quant_config.json atlas_elastic.ple.chunk_rows must be > 0");
    }
  }

  // Populate Qwen4ExpConfig from config.json["gguf_kv"].
  if (!cfg_j.contains("gguf_kv") || !cfg_j["gguf_kv"].is_object()) {
    throw std::invalid_argument("config.json missing gguf_kv metadata for qwen4exp Elastic checkpoint");
  }
  const auto& kv = cfg_j["gguf_kv"];
  auto& c = set.config_;
  c.architecture = kv.value("general.architecture", std::string("qwen4exp"));
  c.model_name = kv.value("general.name", std::string("Qwen3.8-Flash-Next"));
  const std::string p = c.architecture + ".";
  c.num_hidden_layers = kv.at(p + "block_count").get<int>();
  c.hidden_size = kv.at(p + "embedding_length").get<int>();
  c.max_position_embeddings = kv.at(p + "context_length").get<int>();
  c.rms_norm_eps = kv.at(p + "attention.layer_norm_rms_epsilon").get<float>();
  c.hc_mult = kv.at(p + "hyper_connection.count").get<int>();
  c.hc_low_rank = kv.at(p + "hyper_connection.low_rank").get<int>();
  c.num_attention_heads = kv.at(p + "attention.head_count").get<int>();
  c.num_key_value_heads = kv.at(p + "attention.head_count_kv").get<int>();
  c.head_dim = kv.at(p + "attention.key_length").get<int>();
  c.full_attention_interval = kv.value(p + "full_attention_interval", 4);
  c.rope_theta = kv.value(p + "rope.freq_base", 10000000.0f);
  c.rotary_dim = kv.value(p + "rope.dimension_count", 64);
  if (kv.contains(p + "rope.dimension_sections") && kv[p + "rope.dimension_sections"].is_array()) {
    const auto& rs = kv[p + "rope.dimension_sections"];
    for (std::size_t i = 0; i < std::min<std::size_t>(4, rs.size()); ++i) c.rope_sections[i] = rs[i].get<int>();
  }
  c.indexer_head_count = kv.value(p + "attention.indexer.head_count", 0);
  c.indexer_key_length = kv.value(p + "attention.indexer.key_length", 0);
  c.indexer_top_k = kv.value(p + "attention.indexer.top_k", 0);
  if (kv.contains(p + "attention.compress_ratios") && kv[p + "attention.compress_ratios"].is_array()) {
    c.compress_ratios = kv[p + "attention.compress_ratios"].get<std::vector<int>>();
  }
  c.ssm_conv_kernel = kv.at(p + "ssm.conv_kernel").get<int>();
  c.ssm_state_size = kv.at(p + "ssm.state_size").get<int>();
  c.ssm_group_count = kv.at(p + "ssm.group_count").get<int>();
  c.ssm_time_step_rank = kv.at(p + "ssm.time_step_rank").get<int>();
  c.ssm_inner_size = kv.at(p + "ssm.inner_size").get<int>();
  c.expert_count = kv.at(p + "expert_count").get<int>();
  c.expert_used_count = kv.at(p + "expert_used_count").get<int>();
  c.expert_ffn_length = kv.at(p + "expert_feed_forward_length").get<int>();
  c.shared_expert_ffn_length = kv.value(p + "expert_shared_feed_forward_length", 0);
  c.expert_weights_scale = kv.value(p + "expert_weights_scale", 1.0f);
  if (c.expert_weights_scale == 0.0f) c.expert_weights_scale = 1.0f;

  if (kv.contains(p + "ple.layers") && kv[p + "ple.layers"].is_array()) {
    c.ple_layers = kv[p + "ple.layers"].get<std::vector<int>>();
    c.ple_ngram_size = kv.at(p + "ple.ngram_size").get<int>();
    c.ple_heads_per_ngram = kv.at(p + "ple.heads_per_ngram").get<int>();
    c.ple_conv_kernel = kv.at(p + "ple.conv_kernel").get<int>();
    c.ple_head_dim = kv.at(p + "embedding_length_per_layer_input").get<int>();
    c.ple_eos_token_id = kv.at(p + "ple.eos_token_id").get<std::int32_t>();
    c.ple_image_token_id = kv.value(p + "ple.image_token_id", 0);
    c.ple_layer_multipliers = kv.at(p + "ple.layer_multipliers").get<std::vector<std::uint64_t>>();
    c.ple_head_offsets = kv.at(p + "ple.head_offsets").get<std::vector<std::uint64_t>>();
    c.ple_head_vocab_sizes = kv.at(p + "ple.head_vocab_sizes").get<std::vector<std::uint64_t>>();
  }

  // Parse quantized_layers map fail-closed (no silent fallback on unknown quant_algo or group_size != 16).
  if (qcfg_j.contains("quantization") && qcfg_j["quantization"].is_object() &&
      qcfg_j["quantization"].contains("quantized_layers") && qcfg_j["quantization"]["quantized_layers"].is_object()) {
    for (const auto& [k, v] : qcfg_j["quantization"]["quantized_layers"].items()) {
      if (!v.is_object()) {
        throw std::invalid_argument("hf_quant_config.json quantized_layers entry is not an object: " + k);
      }
      if (!v.contains("quant_algo") || !v["quant_algo"].is_string()) {
        throw std::invalid_argument("hf_quant_config.json quantized_layers entry missing string quant_algo: " + k);
      }
      const ElasticTensorKind kind = parse_algo(v["quant_algo"].get<std::string>(), k);
      if (is_packed_codebook_kind(kind) && v.contains("group_size")) {
        if (!v["group_size"].is_number_integer() || v["group_size"].get<int>() != 16) {
          throw std::invalid_argument("hf_quant_config.json layer '" + k + "' requires group_size == 16");
        }
      }
      set.layer_kinds_[k] = kind;
    }
  }

  // Validate weight_map shard filenames and inspect every referenced shard via ingine::SafetensorsFile.
  std::error_code ec;
  const std::filesystem::path canonical_dir = std::filesystem::canonical(dir, ec);
  if (ec) {
    throw std::invalid_argument("cannot canonicalize elastic checkpoint directory: " + dir.string());
  }

  std::set<std::string> shard_names;
  std::unordered_map<std::string, std::string> declared_map;
  for (const auto& [tname, sname_val] : idx_j.at("weight_map").items()) {
    if (tname.empty() || !sname_val.is_string()) {
      throw std::invalid_argument("invalid weight_map entry in model.safetensors.index.json: '" + tname + "'");
    }
    const std::string sname = sname_val.get<std::string>();
    (void)validate_shard_path(dir, canonical_dir, sname);
    shard_names.insert(sname);
    declared_map.emplace(tname, sname);
  }

  for (const std::string& sname : shard_names) {
    const std::filesystem::path sp = validate_shard_path(dir, canonical_dir, sname);
    auto sf = std::make_shared<ingine::SafetensorsFile>(sp);
    if (!sf->open_stat().valid) {
      throw std::invalid_argument("elastic shard is not a regular file: " + sp.string());
    }
    if (sf->is_weight_pack()) {
      throw std::invalid_argument("elastic shard unexpectedly uses INGPACK1 format: " + sp.string());
    }

    ElasticShardInfo si;
    si.filename = sname;
    si.full_path = sp;
    si.file_size = sf->file_size();
    si.data_start = sf->data_offset();
    si.file = sf;

    const std::size_t sidx = set.shards_.size();
    for (const ingine::SafetensorsTensor& t : sf->tensors()) {
      // Fail closed on unknown/unsupported dtypes (SafetensorsFile skips size check when dtype is unknown).
      const std::size_t elem_bytes = ingine::safetensors_dtype_size(t.dtype);
      std::uint64_t expected_bytes = static_cast<std::uint64_t>(elem_bytes);
      for (const std::int64_t dim : t.shape) {
        if (dim < 0) {
          throw std::invalid_argument("tensor '" + t.name + "' has negative/overflowed dimension in " + sname);
        }
        expected_bytes = checked_mul_u64(expected_bytes, static_cast<std::uint64_t>(dim), "tensor shape bytes");
      }
      if (t.bytes != expected_bytes) {
        throw std::invalid_argument("tensor '" + t.name + "' byte size " + std::to_string(t.bytes) +
                                    " disagrees with dtype/shape size " + std::to_string(expected_bytes) + " in " +
                                    sname);
      }
      if (t.offset < si.data_start || t.offset > si.file_size || t.bytes > si.file_size - t.offset) {
        throw std::invalid_argument("tensor '" + t.name + "' span out of bounds in " + sname);
      }

      SafetensorsEntry e;
      e.name = t.name;
      e.dtype = t.dtype;
      e.shape = t.shape;
      e.shard_index = sidx;
      e.data_offset = t.offset;
      e.byte_size = t.bytes;

      set.total_tensor_bytes_ = checked_add_u64(set.total_tensor_bytes_, e.byte_size, "total_tensor_bytes");
      if (!set.tensors_.emplace(e.name, std::move(e)).second) {
        throw std::invalid_argument("duplicate tensor '" + t.name + "' across elastic safetensors shards");
      }
    }
    set.shards_.push_back(std::move(si));
  }

  // Verify every tensor listed in weight_map actually resides in its declared shard.
  for (const auto& [tname, expected_shard] : declared_map) {
    const auto* e = set.find_tensor(tname);
    if (!e) {
      throw std::invalid_argument("weight_map tensor '" + tname + "' missing from shard '" + expected_shard + "'");
    }
    if (set.shards_.at(e->shard_index).filename != expected_shard) {
      throw std::invalid_argument("weight_map tensor '" + tname + "' declared in '" + expected_shard +
                                  "' but found in '" + set.shards_.at(e->shard_index).filename + "'");
    }
  }

  // Derive vocab_size from token_embd.weight and validate config.
  const auto& te = set.require_tensor("token_embd.weight");
  if (te.shape.size() != 2 || te.shape[0] <= 0 ||
      te.shape[0] > static_cast<std::int64_t>(std::numeric_limits<int>::max()) || te.shape[1] <= 0) {
    throw std::invalid_argument("token_embd.weight must have a valid positive 2D shape");
  }
  c.vocab_size = static_cast<int>(te.shape[0]);
  c.finalize_and_validate();

  // Validate full consumer-expected model tensor schema (global dense, per-layer dense/norm/conv,
  // per-layer routed MoE gate/up/down formats/shapes/scale counts/packed spans).
  set.validate_qwen4exp_tensor_schema();

  const ElasticLinearLayout emb_lay = set.validate_dense_linear("token_embd.weight", c.vocab_size, c.hidden_size);
  set.embed_kind_ = emb_lay.kind;
  if (emb_lay.kind == ElasticTensorKind::Fp8) {
    set.embed_scale_ = set.read_scalar_f32(*emb_lay.scale);
  } else if (emb_lay.kind == ElasticTensorKind::Nvfp4) {
    set.embed_scale_ = set.read_scalar_f32(*emb_lay.scale_2);
  } else {
    set.embed_scale_ = 1.0f;
  }

  // Validate embedded host-only PLE shard tensors at ingress against config PLE geometry and row bounds.
  if (set.ple_num_shards_ > 0 && set.ple_format_ != "external") {
    if (c.ple_n_heads() <= 0 || c.ple_head_dim <= 0) {
      throw std::invalid_argument("embedded PLE shards require valid PLE config (ple_n_heads and ple_head_dim > 0)");
    }
    const std::int64_t ple_dim = static_cast<std::int64_t>(c.ple_head_dim);
    std::uint64_t total_ple_rows = 0;

    auto check_chunk_rows = [&](const std::string& pref, int idx, std::int64_t rows) {
      if (rows <= 0) {
        throw std::invalid_argument(pref + ": PLE chunk row count must be positive");
      }
      const std::uint64_t rows_u = static_cast<std::uint64_t>(rows);
      if (idx + 1 < set.ple_num_shards_) {
        if (rows_u != set.ple_chunk_rows_) {
          throw std::invalid_argument(pref + ": non-final PLE chunk rows (" + std::to_string(rows_u) +
                                      ") != chunk_rows (" + std::to_string(set.ple_chunk_rows_) + ")");
        }
      } else if (rows_u > set.ple_chunk_rows_) {
        throw std::invalid_argument(pref + ": final PLE chunk rows (" + std::to_string(rows_u) +
                                    ") exceeds chunk_rows (" + std::to_string(set.ple_chunk_rows_) + ")");
      }
      total_ple_rows = checked_add_u64(total_ple_rows, rows_u, "total_ple_rows");
    };

    if (set.ple_format_ == "nvfp4") {
      if ((c.ple_head_dim % 16) != 0) {
        throw std::invalid_argument("NVFP4 PLE requires ple_head_dim divisible by 16");
      }
      set.ple_chunk_scale2_.resize(static_cast<std::size_t>(set.ple_num_shards_), 1.0f);
      for (int i = 0; i < set.ple_num_shards_; ++i) {
        const std::string pref = "ple.chunk_" + std::to_string(i);
        const auto& w = set.require_tensor(pref + ".weight");
        const auto& sc = set.require_tensor(pref + ".weight_scale");
        const auto& sc2 = set.require_tensor(pref + ".weight_scale_2");
        if (w.dtype != "U8" || w.shape.size() != 2 || w.shape[1] != ple_dim / 2) {
          throw std::invalid_argument(pref + ".weight NVFP4 dtype/shape mismatch against ple_head_dim");
        }
        check_chunk_rows(pref, i, w.shape[0]);
        if (sc.dtype != "F8_E4M3" || sc.shape.size() != 2 || sc.shape[0] != w.shape[0] ||
            sc.shape[1] != ple_dim / 16) {
          throw std::invalid_argument(pref + ".weight_scale NVFP4 dtype/shape mismatch against ple_head_dim");
        }
        validate_scalar_entry(sc2, pref);
        set.ple_chunk_scale2_[static_cast<std::size_t>(i)] = set.read_scalar_f32(sc2);
      }
    } else if (set.ple_format_ == "q8_0") {
      if ((c.ple_head_dim % 32) != 0) {
        throw std::invalid_argument("Q8_0 PLE requires ple_head_dim divisible by 32");
      }
      const std::int64_t expected_row_bytes = (ple_dim / 32) * 34;
      for (int i = 0; i < set.ple_num_shards_; ++i) {
        const std::string pref = "ple.chunk_" + std::to_string(i);
        const auto& raw = set.require_tensor(pref + ".q8_0_raw");
        if (raw.dtype != "U8" || raw.shape.size() != 2 || raw.shape[1] != expected_row_bytes) {
          throw std::invalid_argument(pref + ".q8_0_raw dtype/shape mismatch against ple_head_dim");
        }
        check_chunk_rows(pref, i, raw.shape[0]);
      }
    } else if (set.ple_format_ == "f32") {
      for (int i = 0; i < set.ple_num_shards_; ++i) {
        const std::string pref = "ple.chunk_" + std::to_string(i);
        const auto& f32 = set.require_tensor(pref + ".f32");
        if (f32.dtype != "F32" || f32.shape.size() != 2 || f32.shape[1] != ple_dim) {
          throw std::invalid_argument(pref + ".f32 dtype/shape mismatch against ple_head_dim");
        }
        check_chunk_rows(pref, i, f32.shape[0]);
      }
    }

    for (int h = 0; h < c.ple_n_heads(); ++h) {
      const std::uint64_t end_row = checked_add_u64(
          c.ple_head_offsets[static_cast<std::size_t>(h)],
          c.ple_head_vocab_sizes[static_cast<std::size_t>(h)],
          "ple head end_row");
      if (end_row > total_ple_rows) {
        throw std::invalid_argument("PLE head vocab exceeds embedded PLE shard row count");
      }
    }
  }

  set.identity_.format = ingine::model::CheckpointFormat::Safetensors;
  set.identity_.schema_variant = "qwen4exp_atlas_elastic_" + set.profile_;
  set.identity_.model_id = c.model_name;
  set.identity_.shard_count = static_cast<std::uint32_t>(set.shards_.size());
  set.identity_.weight_manifest_verified = true;
  set.identity_.verification_level = ingine::model::VerificationLevel::HeadersOnly;
  return set;
}

const SafetensorsEntry* Qwen4ExpElasticSet::find_tensor(const std::string& name) const noexcept {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const SafetensorsEntry& Qwen4ExpElasticSet::require_tensor(const std::string& name) const {
  const auto* e = find_tensor(name);
  if (!e) throw std::invalid_argument("missing Elastic safetensors entry: " + name);
  return *e;
}

std::optional<ElasticTensorKind> Qwen4ExpElasticSet::find_layer_kind(const std::string& prefix) const noexcept {
  const auto it = layer_kinds_.find(prefix);
  if (it == layer_kinds_.end()) return std::nullopt;
  return it->second;
}

ElasticTensorKind Qwen4ExpElasticSet::layer_kind(const std::string& prefix) const noexcept {
  const auto it = layer_kinds_.find(prefix);
  return it == layer_kinds_.end() ? ElasticTensorKind::F32 : it->second;
}

const SafetensorsEntry& Qwen4ExpElasticSet::validate_f32_tensor(
    const std::string& name, std::span<const std::int64_t> expected_shape) const {
  const auto& e = require_tensor(name);
  if (e.dtype != "F32" && e.dtype != "BF16") {
    throw std::invalid_argument("tensor '" + name + "' expected F32 or BF16 dtype, got '" + e.dtype + "'");
  }
  const std::string prefix = strip_weight_suffix(name);
  if (const auto declared = find_layer_kind(prefix); declared.has_value()) {
    if (*declared == ElasticTensorKind::F32 && e.dtype != "F32") {
      throw std::invalid_argument("tensor '" + name + "' declared F32 in quantized_layers but stored as '" + e.dtype +
                                  "'");
    }
    if (*declared == ElasticTensorKind::Bf16 && e.dtype != "BF16") {
      throw std::invalid_argument("tensor '" + name + "' declared BF16 in quantized_layers but stored as '" + e.dtype +
                                  "'");
    }
    if (*declared != ElasticTensorKind::F32 && *declared != ElasticTensorKind::Bf16) {
      throw std::invalid_argument("tensor '" + name + "' unexpectedly marked with quantized algo in quantized_layers");
    }
  }
  if (e.shape.size() != expected_shape.size()) {
    throw std::invalid_argument("tensor '" + name + "' rank mismatch: expected " +
                                std::to_string(expected_shape.size()) + ", got " + std::to_string(e.shape.size()));
  }
  std::uint64_t numel = 1;
  for (std::size_t i = 0; i < expected_shape.size(); ++i) {
    if (expected_shape[i] <= 0) {
      throw std::invalid_argument("tensor '" + name + "' invalid expected dim[" + std::to_string(i) + "]");
    }
    if (e.shape[i] != expected_shape[i]) {
      throw std::invalid_argument("tensor '" + name + "' dim[" + std::to_string(i) + "] mismatch: expected " +
                                  std::to_string(expected_shape[i]) + ", got " + std::to_string(e.shape[i]));
    }
    numel = checked_mul_u64(numel, static_cast<std::uint64_t>(expected_shape[i]), "validate_f32_tensor numel");
  }
  const std::uint64_t elem_bytes = (e.dtype == "F32") ? 4u : 2u;
  const std::uint64_t expected_bytes = checked_mul_u64(numel, elem_bytes, "validate_f32_tensor bytes");
  if (e.byte_size != expected_bytes) {
    throw std::invalid_argument("tensor '" + name + "' byte size mismatch: expected " +
                                std::to_string(expected_bytes) + ", got " + std::to_string(e.byte_size));
  }
  return e;
}

ElasticLinearLayout Qwen4ExpElasticSet::validate_dense_linear(const std::string& weight_name, int expected_out_features,
                                                              int expected_in_features) const {
  if (expected_out_features <= 0 || expected_in_features <= 0) {
    throw std::invalid_argument("validate_dense_linear invalid expected dimensions on '" + weight_name + "'");
  }
  const std::string prefix = strip_weight_suffix(weight_name);
  const auto declared = find_layer_kind(prefix);
  const auto& ew = require_tensor(weight_name);
  const std::uint64_t out_u = static_cast<std::uint64_t>(expected_out_features);
  const std::uint64_t in_u = static_cast<std::uint64_t>(expected_in_features);

  ElasticLinearLayout lay;
  lay.weight = &ew;
  lay.out_features = expected_out_features;
  lay.in_features = expected_in_features;

  if (declared.has_value() && *declared == ElasticTensorKind::Fp8) {
    if (ew.dtype != "F8_E4M3") {
      throw std::invalid_argument("dense FP8 weight '" + weight_name + "' must have dtype F8_E4M3, got '" + ew.dtype +
                                  "'");
    }
    if (ew.shape.size() != 2 || ew.shape[0] != expected_out_features || ew.shape[1] != expected_in_features) {
      throw std::invalid_argument("dense FP8 weight '" + weight_name + "' shape mismatch against expected [" +
                                  std::to_string(expected_out_features) + ", " + std::to_string(expected_in_features) +
                                  "]");
    }
    const std::uint64_t expected_w_bytes = checked_mul_u64(out_u, in_u, "dense fp8 bytes");
    if (ew.byte_size != expected_w_bytes) {
      throw std::invalid_argument("dense FP8 weight '" + weight_name + "' byte size mismatch");
    }
    const auto& esc = require_tensor(prefix + ".weight_scale");
    validate_scalar_entry(esc, prefix);
    lay.kind = ElasticTensorKind::Fp8;
    lay.scale = &esc;
    lay.expected_weight_bytes = expected_w_bytes;
    lay.expected_scale_bytes = esc.byte_size;
    return lay;
  }

  if (declared.has_value() && is_packed_codebook_kind(*declared)) {
    const std::uint64_t packed_in = packed_code_width_for_kind(*declared, expected_in_features, prefix);
    const std::uint64_t scale_in = in_u / 16u;
    if (ew.dtype != "U8") {
      throw std::invalid_argument("dense packed weight '" + weight_name + "' must have dtype U8, got '" + ew.dtype +
                                  "'");
    }
    if (ew.shape.size() != 2 || ew.shape[0] != expected_out_features ||
        ew.shape[1] != static_cast<std::int64_t>(packed_in)) {
      throw std::invalid_argument("dense packed weight '" + weight_name + "' shape mismatch against expected [" +
                                  std::to_string(expected_out_features) + ", " + std::to_string(packed_in) + "]");
    }
    const std::uint64_t expected_w_bytes = checked_mul_u64(out_u, packed_in, "dense packed weight bytes");
    if (ew.byte_size != expected_w_bytes) {
      throw std::invalid_argument("dense packed weight '" + weight_name + "' byte size mismatch");
    }

    const auto& esc = require_tensor(prefix + ".weight_scale");
    if (esc.dtype != "F8_E4M3") {
      throw std::invalid_argument("dense packed scale '" + esc.name + "' must have dtype F8_E4M3, got '" + esc.dtype +
                                  "'");
    }
    if (esc.shape.size() != 2 || esc.shape[0] != expected_out_features ||
        esc.shape[1] != static_cast<std::int64_t>(scale_in)) {
      throw std::invalid_argument("dense packed scale '" + esc.name + "' shape mismatch against expected [" +
                                  std::to_string(expected_out_features) + ", " + std::to_string(scale_in) + "]");
    }
    const std::uint64_t expected_s_bytes = checked_mul_u64(out_u, scale_in, "dense packed scale bytes");
    if (esc.byte_size != expected_s_bytes) {
      throw std::invalid_argument("dense packed scale '" + esc.name + "' byte size mismatch");
    }

    const auto& esc2 = require_tensor(prefix + ".weight_scale_2");
    validate_scalar_entry(esc2, prefix);

    lay.kind = *declared;
    lay.scale = &esc;
    lay.scale_2 = &esc2;
    lay.expected_weight_bytes = expected_w_bytes;
    lay.expected_scale_bytes = expected_s_bytes;
    return lay;
  }

  // Unquantized 2D dense weight: strictly F32 or BF16 only (never treat arbitrary non-BF16 dtypes as F32).
  if (ew.dtype != "F32" && ew.dtype != "BF16") {
    throw std::invalid_argument("dense weight '" + weight_name + "' unsupported unquantized dtype '" + ew.dtype +
                                "' (expected F32 or BF16)");
  }
  if (declared.has_value()) {
    if (*declared == ElasticTensorKind::F32 && ew.dtype != "F32") {
      throw std::invalid_argument("dense weight '" + weight_name + "' declared F32 but stored as '" + ew.dtype + "'");
    }
    if (*declared == ElasticTensorKind::Bf16 && ew.dtype != "BF16") {
      throw std::invalid_argument("dense weight '" + weight_name + "' declared BF16 but stored as '" + ew.dtype + "'");
    }
  }
  if (ew.shape.size() != 2 || ew.shape[0] != expected_out_features || ew.shape[1] != expected_in_features) {
    throw std::invalid_argument("dense weight '" + weight_name + "' shape mismatch against expected [" +
                                std::to_string(expected_out_features) + ", " + std::to_string(expected_in_features) +
                                "]");
  }
  const std::uint64_t elem_bytes = (ew.dtype == "BF16") ? 2u : 4u;
  const std::uint64_t expected_w_bytes =
      checked_mul_u64(checked_mul_u64(out_u, in_u, "dense numel"), elem_bytes, "dense bytes");
  if (ew.byte_size != expected_w_bytes) {
    throw std::invalid_argument("dense weight '" + weight_name + "' byte size mismatch");
  }
  lay.kind = (ew.dtype == "BF16") ? ElasticTensorKind::Bf16 : ElasticTensorKind::F32;
  lay.expected_weight_bytes = expected_w_bytes;
  lay.expected_scale_bytes = 0;
  return lay;
}

ElasticExpertLayout Qwen4ExpElasticSet::validate_moe_experts(const std::string& prefix, int expected_expert_count,
                                                             int expected_out_features,
                                                             int expected_in_features) const {
  if (expected_expert_count <= 0 || expected_out_features <= 0 || expected_in_features <= 0) {
    throw std::invalid_argument("validate_moe_experts invalid expected dimensions on '" + prefix + "'");
  }
  const auto declared = find_layer_kind(prefix);
  if (!declared.has_value() || !is_packed_codebook_kind(*declared)) {
    throw std::invalid_argument("MoE expert tensor '" + prefix +
                                "' must declare quant_algo in {NVFP4, ATLAS3, ATLAS2}");
  }

  const std::uint64_t e_u = static_cast<std::uint64_t>(expected_expert_count);
  const std::uint64_t out_u = static_cast<std::uint64_t>(expected_out_features);
  const std::uint64_t in_u = static_cast<std::uint64_t>(expected_in_features);
  const std::uint64_t packed_in = packed_code_width_for_kind(*declared, expected_in_features, prefix);
  const std::uint64_t scale_in = in_u / 16u;

  const std::uint64_t per_exp_code_bytes = checked_mul_u64(out_u, packed_in, "moe per-expert code bytes");
  const std::uint64_t total_code_bytes = checked_mul_u64(e_u, per_exp_code_bytes, "moe total code bytes");
  const std::uint64_t per_exp_scale_bytes = checked_mul_u64(out_u, scale_in, "moe per-expert scale bytes");
  const std::uint64_t total_scale_bytes = checked_mul_u64(e_u, per_exp_scale_bytes, "moe total scale bytes");

  const auto& ew = require_tensor(prefix + ".weight");
  if (ew.dtype != "U8") {
    throw std::invalid_argument("MoE weight '" + ew.name + "' must have dtype U8, got '" + ew.dtype + "'");
  }
  if (ew.shape.size() != 3 || ew.shape[0] != expected_expert_count || ew.shape[1] != expected_out_features ||
      ew.shape[2] != static_cast<std::int64_t>(packed_in)) {
    throw std::invalid_argument("MoE weight '" + ew.name + "' shape mismatch against expected [" +
                                std::to_string(expected_expert_count) + ", " + std::to_string(expected_out_features) +
                                ", " + std::to_string(packed_in) + "]");
  }
  if (ew.byte_size != total_code_bytes) {
    throw std::invalid_argument("MoE weight '" + ew.name + "' byte size mismatch: expected " +
                                std::to_string(total_code_bytes) + ", got " + std::to_string(ew.byte_size));
  }

  const auto& esc = require_tensor(prefix + ".weight_scale");
  if (esc.dtype != "F8_E4M3") {
    throw std::invalid_argument("MoE scale '" + esc.name + "' must have dtype F8_E4M3, got '" + esc.dtype + "'");
  }
  if (esc.shape.size() != 3 || esc.shape[0] != expected_expert_count || esc.shape[1] != expected_out_features ||
      esc.shape[2] != static_cast<std::int64_t>(scale_in)) {
    throw std::invalid_argument("MoE scale '" + esc.name + "' shape mismatch against expected [" +
                                std::to_string(expected_expert_count) + ", " + std::to_string(expected_out_features) +
                                ", " + std::to_string(scale_in) + "]");
  }
  if (esc.byte_size != total_scale_bytes) {
    throw std::invalid_argument("MoE scale '" + esc.name + "' byte size mismatch: expected " +
                                std::to_string(total_scale_bytes) + ", got " + std::to_string(esc.byte_size));
  }

  const auto& esc2 = require_tensor(prefix + ".weight_scale_2");
  if (esc2.dtype != "F32" && esc2.dtype != "BF16") {
    throw std::invalid_argument("MoE scale_2 '" + esc2.name + "' must have dtype F32 or BF16, got '" + esc2.dtype +
                                "'");
  }
  if (esc2.shape.size() != 1 || esc2.shape[0] != expected_expert_count) {
    throw std::invalid_argument("MoE scale_2 '" + esc2.name + "' shape mismatch against expected [" +
                                std::to_string(expected_expert_count) + "]");
  }
  const std::uint64_t esc2_elem_bytes = (esc2.dtype == "F32") ? 4u : 2u;
  const std::uint64_t expected_esc2_bytes = checked_mul_u64(e_u, esc2_elem_bytes, "moe scale_2 bytes");
  if (esc2.byte_size != expected_esc2_bytes) {
    throw std::invalid_argument("MoE scale_2 '" + esc2.name + "' byte size mismatch");
  }

  ElasticExpertLayout lay;
  lay.kind = *declared;
  lay.weight = &ew;
  lay.scale = &esc;
  lay.scale_2 = &esc2;
  lay.expert_count = expected_expert_count;
  lay.out_features = expected_out_features;
  lay.in_features = expected_in_features;
  lay.per_expert_code_bytes = per_exp_code_bytes;
  lay.per_expert_scale_bytes = per_exp_scale_bytes;
  return lay;
}

void Qwen4ExpElasticSet::validate_qwen4exp_tensor_schema() const {
  const auto& c = config_;
  const int H = c.hidden_size;
  const int V = c.vocab_size;
  const int hc = c.hc_mult;
  const int hc_dim = c.hc_dim();
  const int hc_lr = c.hc_low_rank;
  const int E = c.expert_count;
  const int ff_e = c.expert_ffn_length;
  const int ff_s = c.shared_expert_ffn_length;
  const int conv_dim = c.gdn_conv_dim();
  const int val_dim = c.gdn_value_dim();
  const int v_heads = c.ssm_time_step_rank;

  const auto check_1d = [&](const std::string& name, int n) {
    const std::array<std::int64_t, 1> sh{static_cast<std::int64_t>(n)};
    (void)validate_f32_tensor(name, sh);
  };
  const auto check_2d_f32 = [&](const std::string& name, int d0, int d1) {
    const std::array<std::int64_t, 2> sh{static_cast<std::int64_t>(d0), static_cast<std::int64_t>(d1)};
    (void)validate_f32_tensor(name, sh);
  };

  // Global tensors.
  const ElasticLinearLayout emb = validate_dense_linear("token_embd.weight", V, H);
  if (emb.kind != ElasticTensorKind::F32 && emb.kind != ElasticTensorKind::Bf16 &&
      emb.kind != ElasticTensorKind::Fp8 && emb.kind != ElasticTensorKind::Nvfp4) {
    throw std::invalid_argument("token_embd.weight supports only F32, BF16, FP8, or NVFP4");
  }
  check_1d("output_hc_norm.weight", hc_dim);
  (void)validate_dense_linear("output_hc_down.weight", hc_lr, hc_dim);
  (void)validate_dense_linear("output_hc_up.weight", hc_dim, hc_lr);
  if (find_tensor("output.weight") != nullptr) {
    (void)validate_dense_linear("output.weight", V, H);
  }

  for (int il = 0; il < c.num_hidden_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";

    check_1d(p + "hc_attn_norm.weight", hc_dim);
    (void)validate_dense_linear(p + "hc_attn_down.weight", hc_lr, hc_dim);
    (void)validate_dense_linear(p + "hc_attn_up.weight", hc_dim, hc_lr);
    (void)validate_dense_linear(p + "hc_attn_inject.weight", hc, hc_dim);
    check_1d(p + "hc_ffn_norm.weight", hc_dim);
    (void)validate_dense_linear(p + "hc_ffn_down.weight", hc_lr, hc_dim);
    (void)validate_dense_linear(p + "hc_ffn_up.weight", hc_dim, hc_lr);
    (void)validate_dense_linear(p + "hc_ffn_inject.weight", hc, hc_dim);

    if (c.is_ple(il)) {
      (void)validate_dense_linear(p + "ple_key.weight", hc_dim, H);
      (void)validate_dense_linear(p + "ple_value.weight", H, H);
      check_1d(p + "ple_norm_key.weight", hc_dim);
      check_1d(p + "ple_norm_query.weight", hc_dim);
      check_1d(p + "ple_norm_conv.weight", hc_dim);
      check_2d_f32(p + "ple_conv1d.weight", hc_dim, c.ple_conv_kernel);
    }

    if (c.is_linear(il)) {
      (void)validate_dense_linear(p + "attn_qkv.weight", conv_dim, H);
      (void)validate_dense_linear(p + "attn_gate.weight", val_dim, H);
      (void)validate_dense_linear(p + "ssm_alpha.weight", v_heads, H);
      (void)validate_dense_linear(p + "ssm_beta.weight", v_heads, H);
      (void)validate_dense_linear(p + "ssm_out.weight", H, val_dim);
      check_2d_f32(p + "ssm_conv1d.weight", conv_dim, c.ssm_conv_kernel);
      check_1d(p + "ssm_dt.bias", v_heads);
      check_1d(p + "ssm_a", v_heads);
      check_1d(p + "ssm_norm.weight", c.ssm_state_size);
    } else {
      (void)validate_dense_linear(p + "attn_q.weight", c.full_q_out_dim(), H);
      (void)validate_dense_linear(p + "attn_k.weight", c.full_kv_out_dim(), H);
      (void)validate_dense_linear(p + "attn_v.weight", c.full_kv_out_dim(), H);
      (void)validate_dense_linear(p + "attn_output.weight", H, c.num_attention_heads * c.head_dim);
      check_1d(p + "attn_q_norm.weight", c.head_dim);
      check_1d(p + "attn_k_norm.weight", c.head_dim);
      if (c.has_qsa(il)) {
        (void)validate_dense_linear(p + "indexer.q_proj.weight", c.indexer_head_count * c.indexer_key_length, H);
        (void)validate_dense_linear(p + "indexer.k_proj.weight", c.indexer_key_length, H);
        check_1d(p + "indexer.q_norm.weight", c.indexer_key_length);
        check_1d(p + "indexer.k_norm.weight", c.indexer_key_length);
      }
    }

    (void)validate_dense_linear(p + "ffn_gate_inp.weight", E, H);
    if (ff_s > 0) {
      check_1d(p + "ffn_gate_inp_shexp.weight", H);
      (void)validate_dense_linear(p + "ffn_gate_shexp.weight", ff_s, H);
      (void)validate_dense_linear(p + "ffn_up_shexp.weight", ff_s, H);
      (void)validate_dense_linear(p + "ffn_down_shexp.weight", H, ff_s);
    }

    (void)validate_moe_experts(p + "ffn_gate_exps", E, ff_e, H);
    (void)validate_moe_experts(p + "ffn_up_exps", E, ff_e, H);
    (void)validate_moe_experts(p + "ffn_down_exps", E, H, ff_e);
  }
}

void Qwen4ExpElasticSet::read_tensor_bytes(const SafetensorsEntry& entry, std::uint64_t rel_offset,
                                           std::span<std::uint8_t> dst) const {
  if (entry.shard_index >= shards_.size() || !shards_[entry.shard_index].file) {
    throw std::out_of_range("read_tensor_bytes invalid shard index on " + entry.name);
  }
  if (rel_offset > entry.byte_size || dst.size() > entry.byte_size - rel_offset) {
    throw std::out_of_range("read_tensor_bytes out of bounds on " + entry.name);
  }
  const std::uint64_t abs_off = checked_add_u64(entry.data_offset, rel_offset, "read_tensor_bytes offset");
  if (dst.empty()) return;
  shards_[entry.shard_index].file->read_range(abs_off, dst.size(), dst.data());
}

std::vector<std::uint8_t> Qwen4ExpElasticSet::read_tensor_raw(const SafetensorsEntry& entry) const {
  if (entry.byte_size > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("read_tensor_raw size exceeds address space on " + entry.name);
  }
  std::vector<std::uint8_t> out(static_cast<std::size_t>(entry.byte_size));
  read_tensor_bytes(entry, 0, out);
  return out;
}

std::vector<float> Qwen4ExpElasticSet::read_tensor_f32(const SafetensorsEntry& entry) const {
  if (entry.dtype == "F32") {
    if ((entry.byte_size % sizeof(float)) != 0) {
      throw std::invalid_argument("read_tensor_f32 misaligned F32 byte size (" + std::to_string(entry.byte_size) +
                                  ") on " + entry.name);
    }
    const auto raw = read_tensor_raw(entry);
    std::vector<float> out(raw.size() / sizeof(float));
    if (!raw.empty()) {
      std::memcpy(out.data(), raw.data(), raw.size());
    }
    return out;
  }
  if (entry.dtype == "BF16") {
    if ((entry.byte_size % 2u) != 0) {
      throw std::invalid_argument("read_tensor_f32 misaligned BF16 byte size (" + std::to_string(entry.byte_size) +
                                  ") on " + entry.name);
    }
    const auto raw = read_tensor_raw(entry);
    const std::size_t n = raw.size() / 2u;
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t b = 0;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      out[i] = bf16_to_f32_host(b);
    }
    return out;
  }
  throw std::invalid_argument("read_tensor_f32 unsupported dtype '" + entry.dtype + "' on " + entry.name);
}

float Qwen4ExpElasticSet::read_scalar_f32(const SafetensorsEntry& entry) const {
  validate_scalar_entry(entry, "read_scalar_f32");
  const auto v = read_tensor_f32(entry);
  if (v.size() != 1) throw std::invalid_argument("expected scalar F32 on " + entry.name);
  return v[0];
}

void Qwen4ExpElasticSet::read_embed_row_f32(std::uint64_t row_idx, std::span<float> out) const {
  const std::size_t H = out.size();
  if (row_idx >= static_cast<std::uint64_t>(config_.vocab_size) ||
      H != static_cast<std::size_t>(config_.hidden_size)) {
    throw std::out_of_range("read_embed_row_f32 row or hidden size out of range");
  }
  const auto& w = require_tensor("token_embd.weight");
  if (embed_kind_ == ElasticTensorKind::Nvfp4) {
    if ((H % 16) != 0) {
      throw std::invalid_argument("read_embed_row_f32 NVFP4 hidden size must be divisible by 16");
    }
    const auto& sc_e = require_tensor("token_embd.weight_scale");
    std::vector<std::uint8_t> codes(H / 2), scales(H / 16);
    read_tensor_bytes(w, checked_mul_u64(row_idx, H / 2, "embed nvfp4 code offset"), codes);
    read_tensor_bytes(sc_e, checked_mul_u64(row_idx, H / 16, "embed nvfp4 scale offset"), scales);
    for (std::size_t b = 0; b < H / 16; ++b) {
      const float sc = decode_e4m3fn_host(scales[b]) * embed_scale_;
      for (int i = 0; i < 8; ++i) {
        const std::uint8_t pb = codes[b * 8 + i];
        out[b * 16 + 2 * i] = kE2M1Host[pb & 0x0fu] * sc;
        out[b * 16 + 2 * i + 1] = kE2M1Host[pb >> 4] * sc;
      }
    }
    return;
  }
  if (embed_kind_ == ElasticTensorKind::Fp8) {
    std::vector<std::uint8_t> row(H);
    read_tensor_bytes(w, checked_mul_u64(row_idx, H, "embed fp8 offset"), row);
    for (std::size_t i = 0; i < H; ++i) out[i] = decode_e4m3fn_host(row[i]) * embed_scale_;
    return;
  }
  if (w.dtype != "F32" && w.dtype != "BF16") {
    throw std::invalid_argument("read_embed_row_f32 unsupported dense dtype '" + w.dtype + "'");
  }
  const std::uint64_t elem_w = (w.dtype == "F32") ? 4u : 2u;
  const std::uint64_t row_bytes = checked_mul_u64(H, elem_w, "embed dense row bytes");
  std::vector<std::uint8_t> row(static_cast<std::size_t>(row_bytes));
  read_tensor_bytes(w, checked_mul_u64(row_idx, row_bytes, "embed dense offset"), row);
  if (w.dtype == "F32") {
    std::memcpy(out.data(), row.data(), row.size());
  } else {
    for (std::size_t i = 0; i < H; ++i) {
      std::uint16_t b = 0;
      std::memcpy(&b, row.data() + 2 * i, 2);
      out[i] = bf16_to_f32_host(b);
    }
  }
}

void Qwen4ExpElasticSet::read_ple_row_f32(std::uint64_t row_idx, std::span<float> out) const {
  if (ple_num_shards_ <= 0 || ple_chunk_rows_ == 0 || ple_format_ == "external") {
    throw std::invalid_argument("read_ple_row_f32 called without embedded PLE shards");
  }
  const std::size_t D = out.size();
  if (config_.ple_head_dim <= 0 || D != static_cast<std::size_t>(config_.ple_head_dim)) {
    throw std::invalid_argument("read_ple_row_f32 output span size disagrees with ple_head_dim");
  }
  const std::uint64_t si_u64 = row_idx / ple_chunk_rows_;
  const std::uint64_t local_row = row_idx % ple_chunk_rows_;
  if (si_u64 >= static_cast<std::uint64_t>(ple_num_shards_)) {
    throw std::out_of_range("read_ple_row_f32 row_idx exceeds embedded PLE shard range");
  }
  const std::size_t si = static_cast<std::size_t>(si_u64);
  const std::string p = "ple.chunk_" + std::to_string(si);
  if (ple_format_ == "f32") {
    const auto& e = require_tensor(p + ".f32");
    if (local_row >= static_cast<std::uint64_t>(e.shape.at(0)) ||
        static_cast<std::int64_t>(D) != e.shape.at(1)) {
      throw std::out_of_range("read_ple_row_f32 f32 row/dim mismatch");
    }
    const std::uint64_t row_bytes = checked_mul_u64(D, sizeof(float), "ple f32 row bytes");
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(row_bytes));
    read_tensor_bytes(e, checked_mul_u64(local_row, row_bytes, "ple f32 offset"), raw);
    std::memcpy(out.data(), raw.data(), raw.size());
    return;
  }
  if (ple_format_ == "q8_0") {
    const auto& e = require_tensor(p + ".q8_0_raw");
    if (local_row >= static_cast<std::uint64_t>(e.shape.at(0)) || (D % 32) != 0) {
      throw std::out_of_range("read_ple_row_f32 q8_0 row/dim mismatch");
    }
    const std::size_t rb = static_cast<std::size_t>(e.shape.at(1));
    if (rb != (D / 32) * 34) {
      throw std::invalid_argument("read_ple_row_f32 q8_0 row byte width mismatch");
    }
    std::vector<std::uint8_t> raw(rb);
    read_tensor_bytes(e, checked_mul_u64(local_row, rb, "ple q8_0 offset"), raw);
    dequantize_row_to_f32(8u, raw, out);
    return;
  }
  if ((D % 16) != 0) {
    throw std::invalid_argument("read_ple_row_f32 NVFP4 dim must be divisible by 16");
  }
  const auto& w = require_tensor(p + ".weight");
  const auto& sc_e = require_tensor(p + ".weight_scale");
  if (local_row >= static_cast<std::uint64_t>(w.shape.at(0)) ||
      static_cast<std::int64_t>(D / 2) != w.shape.at(1) ||
      static_cast<std::int64_t>(D / 16) != sc_e.shape.at(1)) {
    throw std::out_of_range("read_ple_row_f32 NVFP4 row/dim mismatch");
  }
  const float fac = ple_chunk_scale2_.at(si);
  std::vector<std::uint8_t> codes(D / 2), scales(D / 16);
  read_tensor_bytes(w, checked_mul_u64(local_row, D / 2, "ple nvfp4 code offset"), codes);
  read_tensor_bytes(sc_e, checked_mul_u64(local_row, D / 16, "ple nvfp4 scale offset"), scales);
  for (std::size_t b = 0; b < D / 16; ++b) {
    const float sc = decode_e4m3fn_host(scales[b]) * fac;
    for (int i = 0; i < 8; ++i) {
      const std::uint8_t pb = codes[b * 8 + i];
      out[b * 16 + 2 * i] = kE2M1Host[pb & 0x0fu] * sc;
      out[b * 16 + 2 * i + 1] = kE2M1Host[pb >> 4] * sc;
    }
  }
}

}  // namespace ingine::backends::qwen4exp

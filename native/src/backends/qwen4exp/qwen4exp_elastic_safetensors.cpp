#include "backends/qwen4exp/qwen4exp_elastic_safetensors.h"
#include "backends/qwen4exp/qwen4exp_gguf.h"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>

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

std::shared_ptr<int> open_fd_ro(const std::filesystem::path& p) {
  const int fd = ::open(p.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open elastic shard " + p.string() + ": " + std::strerror(errno));
  return std::shared_ptr<int>(new int(fd), [](int* fp) {
    if (fp) {
      if (*fp >= 0) ::close(*fp);
      delete fp;
    }
  });
}

void pread_all(int fd, const std::filesystem::path& p, std::uint64_t off, std::uint8_t* dst, std::size_t len) {
  std::size_t done = 0;
  while (done < len) {
    const ssize_t n = ::pread(fd, dst + done, len - done, static_cast<off_t>(off + done));
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("pread failed on " + p.string() + ": " + std::strerror(errno));
    }
    if (n == 0) throw std::runtime_error("unexpected EOF in " + p.string());
    done += static_cast<std::size_t>(n);
  }
}

ElasticTensorKind parse_algo(const std::string& algo) {
  if (algo == "FP8") return ElasticTensorKind::Fp8;
  if (algo == "NVFP4") return ElasticTensorKind::Nvfp4;
  if (algo == "ATLAS3") return ElasticTensorKind::Atlas3;
  if (algo == "ATLAS2") return ElasticTensorKind::Atlas2;
  if (algo == "BF16") return ElasticTensorKind::Bf16;
  return ElasticTensorKind::F32;
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

  if (!qcfg_j.contains("atlas_elastic") || !qcfg_j["atlas_elastic"].is_object()) {
    throw std::invalid_argument("hf_quant_config.json missing atlas_elastic section");
  }
  const auto& ae = qcfg_j["atlas_elastic"];
  if (ae.value("family", std::string{}) != "qwen4exp") {
    throw std::invalid_argument("hf_quant_config.json atlas_elastic.family != qwen4exp");
  }

  Qwen4ExpElasticSet set;
  set.profile_ = ae.value("profile", std::string("elastic"));
  if (ae.contains("ple") && ae["ple"].is_object()) {
    set.ple_format_ = ae["ple"].value("format", std::string("external"));
    set.ple_num_shards_ = ae["ple"].value("num_shards", 0);
    set.ple_chunk_rows_ = ae["ple"].value("chunk_rows", 2000000ull);
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

  // Parse quantized_layers map.
  if (qcfg_j.contains("quantization") && qcfg_j["quantization"].contains("quantized_layers")) {
    for (const auto& [k, v] : qcfg_j["quantization"]["quantized_layers"].items()) {
      set.layer_kinds_[k] = parse_algo(v.value("quant_algo", std::string{}));
    }
  }

  // Inspect all safetensors shard headers.
  std::set<std::string> shard_names;
  for (const auto& [tname, sname] : idx_j.at("weight_map").items()) {
    (void)tname;
    shard_names.insert(sname.get<std::string>());
  }
  for (const std::string& sname : shard_names) {
    const std::filesystem::path sp = dir / sname;
    ElasticShardInfo si;
    si.filename = sname;
    si.full_path = sp;
    si.file_size = std::filesystem::file_size(sp);
    si.cached_fd = open_fd_ro(sp);

    std::uint8_t hdr_len_b[8]{};
    pread_all(*si.cached_fd, sp, 0, hdr_len_b, 8);
    std::uint64_t hlen = 0;
    for (int i = 0; i < 8; ++i) hlen |= static_cast<std::uint64_t>(hdr_len_b[i]) << (8 * i);
    std::string hdr_str(hlen, '\0');
    pread_all(*si.cached_fd, sp, 8, reinterpret_cast<std::uint8_t*>(hdr_str.data()), hlen);
    si.data_start = 8ull + hlen;

    const auto hj = nlohmann::json::parse(hdr_str);
    const std::size_t sidx = set.shards_.size();
    for (const auto& [tname, meta] : hj.items()) {
      if (tname == "__metadata__") continue;
      SafetensorsEntry e;
      e.name = tname;
      e.dtype = meta.at("dtype").get<std::string>();
      e.shape = meta.at("shape").get<std::vector<std::int64_t>>();
      const auto offs = meta.at("data_offsets").get<std::vector<std::uint64_t>>();
      e.shard_index = sidx;
      e.data_offset = si.data_start + offs[0];
      e.byte_size = offs[1] - offs[0];
      set.total_tensor_bytes_ += e.byte_size;
      set.tensors_.emplace(tname, std::move(e));
    }
    set.shards_.push_back(std::move(si));
  }

  // Derive vocab_size from token_embd.weight.
  const auto& te = set.require_tensor("token_embd.weight");
  c.vocab_size = static_cast<int>(te.shape.at(0));
  c.finalize_and_validate();

  set.embed_kind_ = set.layer_kind("token_embd");
  if (set.embed_kind_ == ElasticTensorKind::Fp8) {
    set.embed_scale_ = set.read_scalar_f32(set.require_tensor("token_embd.weight_scale"));
  } else if (set.embed_kind_ == ElasticTensorKind::Nvfp4) {
    set.embed_scale_ = set.read_scalar_f32(set.require_tensor("token_embd.weight_scale_2"));
  }

  if (set.ple_num_shards_ > 0 && set.ple_format_ == "nvfp4") {
    set.ple_chunk_scale2_.resize(static_cast<std::size_t>(set.ple_num_shards_), 1.0f);
    for (int i = 0; i < set.ple_num_shards_; ++i) {
      set.ple_chunk_scale2_[static_cast<std::size_t>(i)] =
          set.read_scalar_f32(set.require_tensor("ple.chunk_" + std::to_string(i) + ".weight_scale_2"));
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

ElasticTensorKind Qwen4ExpElasticSet::layer_kind(const std::string& prefix) const noexcept {
  const auto it = layer_kinds_.find(prefix);
  return it == layer_kinds_.end() ? ElasticTensorKind::F32 : it->second;
}

void Qwen4ExpElasticSet::read_tensor_bytes(const SafetensorsEntry& entry, std::uint64_t rel_offset,
                                           std::span<std::uint8_t> dst) const {
  if (rel_offset > entry.byte_size || dst.size() > entry.byte_size - rel_offset) {
    throw std::out_of_range("read_tensor_bytes out of bounds on " + entry.name);
  }
  const auto& sh = shards_.at(entry.shard_index);
  pread_all(*sh.cached_fd, sh.full_path, entry.data_offset + rel_offset, dst.data(), dst.size());
}

std::vector<std::uint8_t> Qwen4ExpElasticSet::read_tensor_raw(const SafetensorsEntry& entry) const {
  std::vector<std::uint8_t> out(entry.byte_size);
  read_tensor_bytes(entry, 0, out);
  return out;
}

std::vector<float> Qwen4ExpElasticSet::read_tensor_f32(const SafetensorsEntry& entry) const {
  const auto raw = read_tensor_raw(entry);
  if (entry.dtype == "F32") {
    std::vector<float> out(raw.size() / 4);
    std::memcpy(out.data(), raw.data(), raw.size());
    return out;
  }
  if (entry.dtype == "BF16") {
    const std::size_t n = raw.size() / 2;
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
  const auto v = read_tensor_f32(entry);
  if (v.size() != 1) throw std::invalid_argument("expected scalar F32 on " + entry.name);
  return v[0];
}

void Qwen4ExpElasticSet::read_embed_row_f32(std::uint64_t row_idx, std::span<float> out) const {
  const std::size_t H = out.size();
  const auto& w = require_tensor("token_embd.weight");
  if (embed_kind_ == ElasticTensorKind::Nvfp4) {
    const auto& sc_e = require_tensor("token_embd.weight_scale");
    std::vector<std::uint8_t> codes(H / 2), scales(H / 16);
    read_tensor_bytes(w, row_idx * (H / 2), codes);
    read_tensor_bytes(sc_e, row_idx * (H / 16), scales);
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
    read_tensor_bytes(w, row_idx * H, row);
    for (std::size_t i = 0; i < H; ++i) out[i] = decode_e4m3fn_host(row[i]) * embed_scale_;
    return;
  }
  std::vector<std::uint8_t> row(H * (w.dtype == "F32" ? 4 : 2));
  read_tensor_bytes(w, row_idx * row.size(), row);
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
  const std::size_t D = out.size();
  const std::size_t si = static_cast<std::size_t>(row_idx / ple_chunk_rows_);
  const std::uint64_t local_row = row_idx % ple_chunk_rows_;
  const std::string p = "ple.chunk_" + std::to_string(si);
  if (ple_format_ == "f32") {
    const auto& e = require_tensor(p + ".f32");
    std::vector<std::uint8_t> raw(D * 4);
    read_tensor_bytes(e, local_row * D * 4, raw);
    std::memcpy(out.data(), raw.data(), D * 4);
    return;
  }
  if (ple_format_ == "q8_0") {
    const auto& e = require_tensor(p + ".q8_0_raw");
    const std::size_t rb = static_cast<std::size_t>(e.shape.at(1));
    std::vector<std::uint8_t> raw(rb);
    read_tensor_bytes(e, local_row * rb, raw);
    dequantize_row_to_f32(8u, raw, out);
    return;
  }
  const auto& w = require_tensor(p + ".weight");
  const auto& sc_e = require_tensor(p + ".weight_scale");
  const float fac = ple_chunk_scale2_.at(si);
  std::vector<std::uint8_t> codes(D / 2), scales(D / 16);
  read_tensor_bytes(w, local_row * (D / 2), codes);
  read_tensor_bytes(sc_e, local_row * (D / 16), scales);
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

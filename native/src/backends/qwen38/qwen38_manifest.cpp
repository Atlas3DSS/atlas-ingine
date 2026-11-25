#include "backends/qwen38/qwen38_manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace ingine::backends::qwen38 {
namespace {

class Sha256 {
 public:
  Sha256() { reset(); }

  void reset() {
    state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    total_bits_ = 0;
    buf_len_ = 0;
  }

  void update(const void* data, std::size_t len) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    total_bits_ += static_cast<std::uint64_t>(len) * 8ull;
    while (len > 0) {
      const std::size_t take = std::min<std::size_t>(len, 64 - buf_len_);
      std::memcpy(buf_.data() + buf_len_, p, take);
      buf_len_ += take;
      p += take;
      len -= take;
      if (buf_len_ == 64) {
        compress(buf_.data());
        buf_len_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> finalize() {
    buf_[buf_len_++] = 0x80u;
    if (buf_len_ > 56) {
      while (buf_len_ < 64) buf_[buf_len_++] = 0;
      compress(buf_.data());
      buf_len_ = 0;
    }
    while (buf_len_ < 56) buf_[buf_len_++] = 0;
    for (int i = 0; i < 8; ++i) {
      buf_[56 + i] = static_cast<std::uint8_t>((total_bits_ >> (56 - 8 * i)) & 0xffu);
    }
    compress(buf_.data());

    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
      out[4 * i + 0] = static_cast<std::uint8_t>((state_[i] >> 24) & 0xffu);
      out[4 * i + 1] = static_cast<std::uint8_t>((state_[i] >> 16) & 0xffu);
      out[4 * i + 2] = static_cast<std::uint8_t>((state_[i] >> 8) & 0xffu);
      out[4 * i + 3] = static_cast<std::uint8_t>(state_[i] & 0xffu);
    }
    return out;
  }

 private:
  static inline std::uint32_t rotr(std::uint32_t x, std::uint32_t n) noexcept {
    return (x >> n) | (x << (32 - n));
  }

  void compress(const std::uint8_t* block) {
    static constexpr std::array<std::uint32_t, 64> kK = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(block[4 * i + 0]) << 24) |
             (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
             (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) |
             static_cast<std::uint32_t>(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = state_;
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + s1 + ch + kK[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::uint64_t total_bits_ = 0;
  std::array<std::uint8_t, 64> buf_{};
  std::size_t buf_len_ = 0;
};

std::string to_hex(const std::array<std::uint8_t, 32>& d) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out(64, '0');
  for (std::size_t i = 0; i < 32; ++i) {
    out[2 * i] = kHex[(d[i] >> 4) & 0xf];
    out[2 * i + 1] = kHex[d[i] & 0xf];
  }
  return out;
}

struct ParsedSafetensorHeader {
  std::uint64_t data_base_offset = 0;
  std::uint64_t file_size = 0;
  std::unordered_map<std::string, Qwen38TensorInfo> tensors;
};

ParsedSafetensorHeader parse_safetensors_header(const std::filesystem::path& path, const std::string& shard_name) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) {
    throw std::invalid_argument("cannot open safetensors shard: " + path.string());
  }
  const std::streamoff end_pos = in.tellg();
  if (end_pos < 8) {
    throw std::invalid_argument("safetensors shard truncated (< 8 bytes): " + path.string());
  }
  const std::uint64_t file_size = static_cast<std::uint64_t>(end_pos);
  in.seekg(0, std::ios::beg);

  std::array<std::uint8_t, 8> len_bytes{};
  in.read(reinterpret_cast<char*>(len_bytes.data()), 8);
  std::uint64_t header_len = 0;
  for (int i = 0; i < 8; ++i) {
    header_len |= (static_cast<std::uint64_t>(len_bytes[i]) << (8 * i));
  }
  if (header_len == 0 || header_len > (64ull * 1024ull * 1024ull) || 8ull + header_len > file_size) {
    throw std::invalid_argument("invalid safetensors header length in " + path.string());
  }

  std::string header_json(header_len, '\0');
  in.read(header_json.data(), static_cast<std::streamsize>(header_len));
  nlohmann::json hdr;
  try {
    hdr = nlohmann::json::parse(header_json);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument("invalid safetensors JSON header in " + path.string() + ": " + e.what());
  }
  if (!hdr.is_object()) {
    throw std::invalid_argument("safetensors header is not a JSON object: " + path.string());
  }

  ParsedSafetensorHeader parsed;
  parsed.data_base_offset = 8ull + header_len;
  parsed.file_size = file_size;
  const std::uint64_t payload_bytes = file_size - parsed.data_base_offset;

  for (auto it = hdr.begin(); it != hdr.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const auto& obj = it.value();
    if (!obj.is_object() || !obj.contains("dtype") || !obj.contains("shape") || !obj.contains("data_offsets")) {
      throw std::invalid_argument("malformed safetensors entry '" + it.key() + "' in " + path.string());
    }
    Qwen38TensorInfo info;
    info.name = it.key();
    info.shard_file = shard_name;
    info.dtype = obj.at("dtype").get<std::string>();
    for (const auto& d : obj.at("shape")) {
      info.shape.push_back(d.get<std::int64_t>());
    }
    const auto& offs = obj.at("data_offsets");
    if (!offs.is_array() || offs.size() != 2) {
      throw std::invalid_argument("invalid data_offsets for tensor '" + info.name + "'");
    }
    info.data_offset_begin = offs[0].get<std::uint64_t>();
    info.data_offset_end = offs[1].get<std::uint64_t>();
    if (info.data_offset_begin > info.data_offset_end || info.data_offset_end > payload_bytes) {
      throw std::invalid_argument("out-of-bounds data_offsets for tensor '" + info.name + "' in " + shard_name);
    }
    parsed.tensors.emplace(info.name, std::move(info));
  }
  return parsed;
}

bool is_excluded_tensor(std::string_view name, bool& is_visual, bool& is_mtp) {
  is_visual = name.starts_with("model.visual.");
  is_mtp = name.starts_with("mtp.");
  return is_visual || is_mtp;
}

void assert_tensor(const std::unordered_map<std::string, Qwen38TensorInfo>& tensors, const std::string& name,
                   std::initializer_list<std::string_view> allowed_dtypes,
                   std::initializer_list<std::int64_t> expected_shape, std::unordered_set<std::string>& seen) {
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    throw std::invalid_argument("qwen38 manifest missing required tensor '" + name + "'");
  }
  const Qwen38TensorInfo& t = it->second;
  bool dtype_ok = false;
  for (std::string_view dt : allowed_dtypes) {
    if (t.dtype == dt) {
      dtype_ok = true;
      break;
    }
  }
  if (!dtype_ok) {
    throw std::invalid_argument("qwen38 tensor '" + name + "' has unexpected dtype '" + t.dtype + "'");
  }
  if (t.shape.size() != expected_shape.size() ||
      !std::equal(t.shape.begin(), t.shape.end(), expected_shape.begin())) {
    std::ostringstream ss;
    ss << "qwen38 tensor '" << name << "' has unexpected shape [";
    for (std::size_t i = 0; i < t.shape.size(); ++i) ss << (i ? "," : "") << t.shape[i];
    ss << "]";
    throw std::invalid_argument(ss.str());
  }
  seen.insert(name);
}

void assert_linear_proj(const std::unordered_map<std::string, Qwen38TensorInfo>& tensors, const std::string& prefix,
                        int out_f, int in_f, Qwen38LinearQuantKind quant, std::unordered_set<std::string>& seen) {
  if (quant == Qwen38LinearQuantKind::Bf16) {
    assert_tensor(tensors, prefix + ".weight", {"BF16"}, {out_f, in_f}, seen);
    return;
  }
  if (quant == Qwen38LinearQuantKind::Fp8E4M3) {
    assert_tensor(tensors, prefix + ".weight", {"F8_E4M3"}, {out_f, in_f}, seen);
    assert_tensor(tensors, prefix + ".weight_scale", {"F32"}, {}, seen);
    assert_tensor(tensors, prefix + ".input_scale", {"F32"}, {}, seen);
    return;
  }
  if (in_f % 16 != 0) {
    throw std::invalid_argument("NVFP4 tensor '" + prefix + "' requires in_features divisible by 16");
  }
  assert_tensor(tensors, prefix + ".weight", {"U8"}, {out_f, in_f / 2}, seen);
  assert_tensor(tensors, prefix + ".weight_scale", {"F8_E4M3"}, {out_f, in_f / 16}, seen);
  assert_tensor(tensors, prefix + ".weight_scale_2", {"F32"}, {}, seen);
  assert_tensor(tensors, prefix + ".input_scale", {"F32"}, {}, seen);
}

}  // namespace

std::array<std::uint8_t, 32> sha256_bytes_digest(const void* data, std::size_t size) {
  Sha256 h;
  h.update(data, size);
  return h.finalize();
}

std::string sha256_bytes_hex(const void* data, std::size_t size) { return to_hex(sha256_bytes_digest(data, size)); }

std::array<std::uint8_t, 32> sha256_file_digest(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::invalid_argument("cannot open file for sha256: " + path.string());
  }
  Sha256 h;
  std::array<char, 65536> buf{};
  while (in.good()) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize got = in.gcount();
    if (got > 0) h.update(buf.data(), static_cast<std::size_t>(got));
  }
  if (in.bad()) {
    throw std::invalid_argument("read error while hashing file: " + path.string());
  }
  return h.finalize();
}

std::string sha256_file_hex_streaming(const std::filesystem::path& path) {
  return to_hex(sha256_file_digest(path));
}

Qwen38Manifest Qwen38Manifest::inspect(const std::filesystem::path& model_dir, bool verify_shard_sha256,
                                        const std::string& revision) {
  if (!std::filesystem::is_directory(model_dir)) {
    throw std::invalid_argument("qwen38 checkpoint directory does not exist: " + model_dir.string());
  }
  const std::filesystem::path cfg_path = model_dir / "config.json";
  const std::filesystem::path tok_path = model_dir / "tokenizer.json";
  const std::filesystem::path tmpl_path = model_dir / "chat_template.jinja";

  if (!std::filesystem::is_regular_file(cfg_path)) {
    throw std::invalid_argument("missing config.json in " + model_dir.string());
  }
  if (!std::filesystem::is_regular_file(tok_path)) {
    throw std::invalid_argument("missing tokenizer.json in " + model_dir.string());
  }

  Qwen38Manifest manifest;
  manifest.config = Qwen38Config::load(cfg_path);
  manifest.checkpoint.format = ingine::model::CheckpointFormat::Safetensors;
  manifest.checkpoint.schema_variant =
      manifest.config.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4 ? "qwen3_5_mixed_fp8_nvfp4"
                                                                              : "qwen3_5_bf16";
  manifest.checkpoint.model_id = model_dir.filename().string();
  manifest.checkpoint.revision = revision;
  manifest.checkpoint.config_sha256 = sha256_file_digest(cfg_path);
  manifest.checkpoint.tokenizer_sha256 = sha256_file_digest(tok_path);
  if (std::filesystem::is_regular_file(tmpl_path)) {
    manifest.checkpoint.chat_template_sha256 = sha256_file_digest(tmpl_path);
  }

  const std::filesystem::path index_path = model_dir / "model.safetensors.index.json";
  const std::filesystem::path single_path = model_dir / "model.safetensors";

  std::vector<std::pair<std::string, std::string>> index_entries;  // (tensor_name, shard_file)
  std::unordered_set<std::string> shard_set;
  if (std::filesystem::is_regular_file(index_path)) {
    std::ifstream in(index_path, std::ios::binary);
    nlohmann::json idx;
    try {
      in >> idx;
    } catch (const nlohmann::json::exception& e) {
      throw std::invalid_argument("invalid model.safetensors.index.json: " + std::string(e.what()));
    }
    if (!idx.is_object() || !idx.contains("weight_map") || !idx.at("weight_map").is_object()) {
      throw std::invalid_argument("model.safetensors.index.json missing weight_map object");
    }
    const auto& wm = idx.at("weight_map");
    for (auto it = wm.begin(); it != wm.end(); ++it) {
      if (!it.value().is_string()) {
        throw std::invalid_argument("weight_map value must be string for tensor '" + it.key() + "'");
      }
      const std::string shard = it.value().get<std::string>();
      index_entries.emplace_back(it.key(), shard);
      shard_set.insert(shard);
    }
    manifest.shard_files.assign(shard_set.begin(), shard_set.end());
    std::sort(manifest.shard_files.begin(), manifest.shard_files.end());
  } else if (std::filesystem::is_regular_file(single_path)) {
    manifest.shard_files.push_back("model.safetensors");
  } else {
    throw std::invalid_argument("neither model.safetensors.index.json nor model.safetensors found in " +
                                model_dir.string());
  }

  std::unordered_map<std::string, ParsedSafetensorHeader> parsed_shards;
  Sha256 shard_set_hash;
  for (const std::string& shard : manifest.shard_files) {
    const std::filesystem::path sp = model_dir / shard;
    if (!std::filesystem::is_regular_file(sp)) {
      throw std::invalid_argument("missing safetensors shard: " + sp.string());
    }
    auto hdr = parse_safetensors_header(sp, shard);
    std::string digest_note = "header_verified";
    if (verify_shard_sha256) {
      digest_note = sha256_file_hex_streaming(sp);
      manifest.shard_sha256.emplace(shard, digest_note);
    }
    const std::string line = shard + '|' + std::to_string(hdr.file_size) + '|' + digest_note + '\n';
    shard_set_hash.update(line.data(), line.size());
    parsed_shards.emplace(shard, std::move(hdr));
  }
  manifest.checkpoint.shard_set_sha256 = shard_set_hash.finalize();
  manifest.shard_payloads_verified = verify_shard_sha256;

  // Reconcile index weight_map against shard headers, partitioning active text vs visual/MTP.
  if (!index_entries.empty()) {
    manifest.total_index_tensors = index_entries.size();
    for (const auto& [name, shard] : index_entries) {
      const auto& hdr = parsed_shards.at(shard);
      auto tit = hdr.tensors.find(name);
      if (tit == hdr.tensors.end()) {
        throw std::invalid_argument("index references tensor '" + name + "' missing from shard '" + shard + "'");
      }
      bool vis = false;
      bool mtp = false;
      if (is_excluded_tensor(name, vis, mtp)) {
        if (vis) {
          ++manifest.excluded_visual_tensor_count;
          manifest.memory_budget.excluded_visual_bytes += tit->second.byte_size();
        } else if (mtp) {
          ++manifest.excluded_mtp_tensor_count;
          manifest.memory_budget.excluded_mtp_bytes += tit->second.byte_size();
        }
        continue;
      }
      manifest.memory_budget.text_weight_bytes += tit->second.byte_size();
      manifest.active_tensors.emplace(name, tit->second);
    }
  } else {
    const auto& hdr = parsed_shards.at("model.safetensors");
    manifest.total_index_tensors = hdr.tensors.size();
    for (const auto& [name, info] : hdr.tensors) {
      bool vis = false;
      bool mtp = false;
      if (is_excluded_tensor(name, vis, mtp)) {
        if (vis) {
          ++manifest.excluded_visual_tensor_count;
          manifest.memory_budget.excluded_visual_bytes += info.byte_size();
        } else if (mtp) {
          ++manifest.excluded_mtp_tensor_count;
          manifest.memory_budget.excluded_mtp_bytes += info.byte_size();
        }
        continue;
      }
      manifest.memory_budget.text_weight_bytes += info.byte_size();
      manifest.active_tensors.emplace(name, info);
    }
  }
  manifest.active_text_tensor_count = manifest.active_tensors.size();

  // Verify exact expected text tensor inventory against Qwen38Config.
  const Qwen38Config& c = manifest.config;
  std::unordered_set<std::string> seen;
  seen.reserve(manifest.active_tensors.size());

  assert_tensor(manifest.active_tensors, "model.language_model.embed_tokens.weight", {"BF16"},
                {c.vocab_size, c.hidden_size}, seen);
  assert_tensor(manifest.active_tensors, "model.language_model.norm.weight", {"BF16"}, {c.hidden_size}, seen);
  if (!c.tie_word_embeddings) {
    assert_linear_proj(manifest.active_tensors, "lm_head", c.vocab_size, c.hidden_size, c.lm_head_quant(), seen);
  }

  for (int i = 0; i < c.num_hidden_layers; ++i) {
    const std::string lp = "model.language_model.layers." + std::to_string(i);
    assert_tensor(manifest.active_tensors, lp + ".input_layernorm.weight", {"BF16"}, {c.hidden_size}, seen);
    assert_tensor(manifest.active_tensors, lp + ".post_attention_layernorm.weight", {"BF16"}, {c.hidden_size}, seen);

    if (c.layer_types[static_cast<std::size_t>(i)] == Qwen38LayerKind::LinearAttention) {
      const std::string attn = lp + ".linear_attn";
      assert_tensor(manifest.active_tensors, attn + ".A_log", {"BF16", "F32"}, {c.linear_num_value_heads}, seen);
      assert_tensor(manifest.active_tensors, attn + ".dt_bias", {"BF16", "F32"}, {c.linear_num_value_heads}, seen);
      assert_tensor(manifest.active_tensors, attn + ".conv1d.weight", {"BF16", "F32"},
                    {c.linear_conv_dim(), 1, c.linear_conv_kernel_dim}, seen);
      assert_tensor(manifest.active_tensors, attn + ".in_proj_a.weight", {"BF16", "F32"},
                    {c.linear_num_value_heads, c.hidden_size}, seen);
      assert_tensor(manifest.active_tensors, attn + ".in_proj_b.weight", {"BF16", "F32"},
                    {c.linear_num_value_heads, c.hidden_size}, seen);
      assert_tensor(manifest.active_tensors, attn + ".norm.weight", {"BF16", "F32"}, {c.linear_value_head_dim}, seen);

      assert_linear_proj(manifest.active_tensors, attn + ".in_proj_qkv", c.linear_conv_dim(), c.hidden_size,
                         c.linear_attn_quant(), seen);
      assert_linear_proj(manifest.active_tensors, attn + ".in_proj_z", c.linear_value_dim(), c.hidden_size,
                         c.linear_attn_quant(), seen);
      assert_linear_proj(manifest.active_tensors, attn + ".out_proj", c.hidden_size, c.linear_value_dim(),
                         c.linear_attn_quant(), seen);
    } else {
      const std::string attn = lp + ".self_attn";
      assert_tensor(manifest.active_tensors, attn + ".q_norm.weight", {"BF16", "F32"}, {c.head_dim}, seen);
      assert_tensor(manifest.active_tensors, attn + ".k_norm.weight", {"BF16", "F32"}, {c.head_dim}, seen);

      assert_linear_proj(manifest.active_tensors, attn + ".q_proj", c.full_q_proj_out_dim(), c.hidden_size,
                         c.full_attn_quant(), seen);
      assert_linear_proj(manifest.active_tensors, attn + ".k_proj", c.full_kv_proj_out_dim(), c.hidden_size,
                         c.full_attn_quant(), seen);
      assert_linear_proj(manifest.active_tensors, attn + ".v_proj", c.full_kv_proj_out_dim(), c.hidden_size,
                         c.full_attn_quant(), seen);
      assert_linear_proj(manifest.active_tensors, attn + ".o_proj", c.hidden_size, c.num_attention_heads * c.head_dim,
                         c.full_attn_quant(), seen);
    }

    assert_linear_proj(manifest.active_tensors, lp + ".mlp.gate_proj", c.intermediate_size, c.hidden_size,
                       c.mlp_quant(), seen);
    assert_linear_proj(manifest.active_tensors, lp + ".mlp.up_proj", c.intermediate_size, c.hidden_size,
                       c.mlp_quant(), seen);
    assert_linear_proj(manifest.active_tensors, lp + ".mlp.down_proj", c.hidden_size, c.intermediate_size,
                       c.mlp_quant(), seen);
  }

  if (seen.size() != manifest.active_tensors.size()) {
    for (const auto& [name, _] : manifest.active_tensors) {
      if (seen.find(name) == seen.end()) {
        throw std::invalid_argument("unexpected active text tensor in qwen38 manifest: '" + name + "'");
      }
    }
  }

  // Canonical digest over active tensor inventory (deterministic regardless of JSON iteration order).
  std::vector<std::string> sorted_names;
  sorted_names.reserve(manifest.active_tensors.size());
  for (const auto& [name, _] : manifest.active_tensors) sorted_names.push_back(name);
  std::sort(sorted_names.begin(), sorted_names.end());

  Sha256 mh;
  for (const std::string& name : sorted_names) {
    const Qwen38TensorInfo& t = manifest.active_tensors.at(name);
    std::ostringstream line;
    line << name << '|' << t.shard_file << '|' << t.dtype << '|';
    for (std::size_t i = 0; i < t.shape.size(); ++i) line << (i ? "," : "") << t.shape[i];
    line << '|' << t.data_offset_begin << ':' << t.data_offset_end << '\n';
    const std::string s = line.str();
    mh.update(s.data(), s.size());
  }
  manifest.checkpoint.weight_manifest_sha256 = mh.finalize();
  manifest.checkpoint.weight_manifest_verified = true;
  manifest.checkpoint.shard_count = static_cast<std::uint32_t>(manifest.shard_files.size());
  manifest.checkpoint.shard_digests_verified = verify_shard_sha256;

  manifest.memory_budget.gdn_recurrent_bytes =
      static_cast<std::uint64_t>(c.num_linear_layers) * static_cast<std::uint64_t>(c.linear_num_value_heads) *
      static_cast<std::uint64_t>(c.linear_key_head_dim) * static_cast<std::uint64_t>(c.linear_value_head_dim) * 4ull;
  manifest.memory_budget.gdn_conv_bytes =
      static_cast<std::uint64_t>(c.num_linear_layers) * static_cast<std::uint64_t>(c.linear_conv_dim()) *
      static_cast<std::uint64_t>(c.linear_conv_kernel_dim) * 2ull;
  manifest.memory_budget.kv_bytes_per_token =
      static_cast<std::uint64_t>(c.num_full_layers) * 2ull * static_cast<std::uint64_t>(c.num_key_value_heads) *
      static_cast<std::uint64_t>(c.head_dim) * 2ull;

  return manifest;
}

}  // namespace ingine::backends::qwen38

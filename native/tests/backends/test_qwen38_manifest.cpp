#include "backends/qwen38/qwen38_config.h"
#include "backends/qwen38/qwen38_manifest.h"
#include "backends/qwen38/qwen38_backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

#define CHECK_THROWS(expr, ex_type)                                                         \
  do {                                                                                      \
    bool threw_ = false;                                                                    \
    try {                                                                                   \
      (void)(expr);                                                                         \
    } catch (const ex_type&) {                                                              \
      threw_ = true;                                                                        \
    }                                                                                       \
    if (!threw_) {                                                                          \
      std::fprintf(stderr, "%s:%d: expected exception %s: %s\n", __FILE__, __LINE__,        \
                   #ex_type, #expr);                                                        \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

namespace {

using namespace ingine::backends::qwen38;

struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const char* tag) {
    path = std::filesystem::temp_directory_path() / (std::string("ingine_test_manifest_") + tag);
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

void write_bytes(const std::filesystem::path& p, const void* data, std::size_t n) {
  std::ofstream out(p, std::ios::binary);
  CHECK(out.good());
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
}

void write_str(const std::filesystem::path& p, const std::string& s) { write_bytes(p, s.data(), s.size()); }

std::string hex_digest(const std::array<std::uint8_t, 32>& d) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out(64, '0');
  for (std::size_t i = 0; i < 32; ++i) {
    out[2 * i] = kHex[(d[i] >> 4) & 0xf];
    out[2 * i + 1] = kHex[d[i] & 0xf];
  }
  return out;
}

std::string minimal_bf16_config() {
  return R"({
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": true,
    "text_config": {
      "model_type": "qwen3_5_text",
      "vocab_size": 32,
      "hidden_size": 16,
      "intermediate_size": 32,
      "num_hidden_layers": 2,
      "max_position_embeddings": 128,
      "rms_norm_eps": 1e-6,
      "full_attention_interval": 2,
      "num_attention_heads": 2,
      "num_key_value_heads": 1,
      "head_dim": 8,
      "attention_bias": false,
      "attn_output_gate": true,
      "linear_conv_kernel_dim": 4,
      "linear_key_head_dim": 4,
      "linear_value_head_dim": 4,
      "linear_num_key_heads": 1,
      "linear_num_value_heads": 2,
      "mamba_ssm_dtype": "float32",
      "layer_types": ["linear_attention", "full_attention"],
      "rope_parameters": {
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25
      }
    }
  })";
}

void write_synthetic_safetensors_dir(const std::filesystem::path& dir, bool corrupt_extra_tensor = false) {
  write_str(dir / "config.json", minimal_bf16_config());
  write_str(dir / "tokenizer.json", R"({"version":"1.0"})");
  write_str(dir / "chat_template.jinja", "{{ messages }}");

  struct Spec {
    std::string name;
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::size_t elem_bytes;
  };
  std::vector<Spec> specs = {
      {"model.language_model.embed_tokens.weight", "BF16", {32, 16}, 2},
      {"model.language_model.norm.weight", "BF16", {16}, 2},
      // Layer 0 (linear_attention)
      {"model.language_model.layers.0.input_layernorm.weight", "BF16", {16}, 2},
      {"model.language_model.layers.0.post_attention_layernorm.weight", "BF16", {16}, 2},
      {"model.language_model.layers.0.linear_attn.A_log", "F32", {2}, 4},
      {"model.language_model.layers.0.linear_attn.dt_bias", "BF16", {2}, 2},
      {"model.language_model.layers.0.linear_attn.conv1d.weight", "BF16", {16, 1, 4}, 2},
      {"model.language_model.layers.0.linear_attn.in_proj_a.weight", "BF16", {2, 16}, 2},
      {"model.language_model.layers.0.linear_attn.in_proj_b.weight", "BF16", {2, 16}, 2},
      {"model.language_model.layers.0.linear_attn.norm.weight", "F32", {4}, 4},
      {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "BF16", {16, 16}, 2},
      {"model.language_model.layers.0.linear_attn.in_proj_z.weight", "BF16", {8, 16}, 2},
      {"model.language_model.layers.0.linear_attn.out_proj.weight", "BF16", {16, 8}, 2},
      {"model.language_model.layers.0.mlp.gate_proj.weight", "BF16", {32, 16}, 2},
      {"model.language_model.layers.0.mlp.up_proj.weight", "BF16", {32, 16}, 2},
      {"model.language_model.layers.0.mlp.down_proj.weight", "BF16", {16, 32}, 2},
      // Layer 1 (full_attention)
      {"model.language_model.layers.1.input_layernorm.weight", "BF16", {16}, 2},
      {"model.language_model.layers.1.post_attention_layernorm.weight", "BF16", {16}, 2},
      {"model.language_model.layers.1.self_attn.q_norm.weight", "BF16", {8}, 2},
      {"model.language_model.layers.1.self_attn.k_norm.weight", "BF16", {8}, 2},
      {"model.language_model.layers.1.self_attn.q_proj.weight", "BF16", {32, 16}, 2},
      {"model.language_model.layers.1.self_attn.k_proj.weight", "BF16", {8, 16}, 2},
      {"model.language_model.layers.1.self_attn.v_proj.weight", "BF16", {8, 16}, 2},
      {"model.language_model.layers.1.self_attn.o_proj.weight", "BF16", {16, 16}, 2},
      {"model.language_model.layers.1.mlp.gate_proj.weight", "BF16", {32, 16}, 2},
      {"model.language_model.layers.1.mlp.up_proj.weight", "BF16", {32, 16}, 2},
      {"model.language_model.layers.1.mlp.down_proj.weight", "BF16", {16, 32}, 2},
      // Excluded non-text components (must be skipped cleanly).
      {"model.visual.patch_embed.proj.weight", "BF16", {4}, 2},
      {"mtp.layers.0.norm.weight", "BF16", {16}, 2},
  };
  if (corrupt_extra_tensor) {
    specs.push_back({"model.language_model.stray.weight", "BF16", {4}, 2});
  }

  nlohmann::ordered_json hdr;
  std::uint64_t offset = 0;
  for (const auto& sp : specs) {
    std::uint64_t elems = 1;
    for (std::int64_t d : sp.shape) elems *= static_cast<std::uint64_t>(d);
    const std::uint64_t bytes = elems * sp.elem_bytes;
    hdr[sp.name] = {{"dtype", sp.dtype}, {"shape", sp.shape}, {"data_offsets", {offset, offset + bytes}}};
    offset += bytes;
  }
  const std::string hdr_str = hdr.dump();
  const std::uint64_t hdr_len = hdr_str.size();
  std::vector<std::uint8_t> st(8 + hdr_len + offset, 0u);
  for (int i = 0; i < 8; ++i) st[i] = static_cast<std::uint8_t>((hdr_len >> (8 * i)) & 0xffu);
  std::memcpy(st.data() + 8, hdr_str.data(), hdr_len);
  write_bytes(dir / "model.safetensors", st.data(), st.size());
}

void test_config_validation() {
  const auto cfg = Qwen38Config::from_json_str(minimal_bf16_config());
  CHECK(cfg.num_hidden_layers == 2);
  CHECK(cfg.num_linear_layers == 1);
  CHECK(cfg.num_full_layers == 1);
  CHECK(cfg.linear_conv_dim() == 16);
  CHECK(cfg.full_q_proj_out_dim() == 32);
  CHECK(cfg.rotary_dim == 2);
  CHECK(cfg.tie_word_embeddings);

  CHECK_THROWS(Qwen38Config::from_json_str("{bad"), std::invalid_argument);
  CHECK_THROWS(Qwen38Config::from_json_str(R"({"model_type":"llama","text_config":{}})"), std::invalid_argument);
  {
    auto j = nlohmann::json::parse(minimal_bf16_config());
    j["text_config"]["mamba_ssm_dtype"] = "bfloat16";
    CHECK_THROWS(Qwen38Config::from_json_str(j.dump()), std::invalid_argument);
  }
  {
    auto j = nlohmann::json::parse(minimal_bf16_config());
    j["text_config"]["linear_conv_kernel_dim"] = 5;
    CHECK_THROWS(Qwen38Config::from_json_str(j.dump()), std::invalid_argument);
  }
  {
    auto j = nlohmann::json::parse(minimal_bf16_config());
    j["text_config"]["attention_bias"] = true;
    CHECK_THROWS(Qwen38Config::from_json_str(j.dump()), std::invalid_argument);
  }
  {
    auto j = nlohmann::json::parse(minimal_bf16_config());
    j["quantization_config"] = {
        {"quant_method", "modelopt"},
        {"quant_algo", "MIXED_PRECISION"},
        {"config_groups", {{"lm_head", {{"quant_algo", "NVFP4"}, {"group_size", 32}}}}},
    };
    CHECK_THROWS(Qwen38Config::from_json_str(j.dump()), std::invalid_argument);
  }
}

void test_synthetic_manifest() {
  TempDir ok("ok");
  write_synthetic_safetensors_dir(ok.path, false);
  const auto m_hdr = Qwen38Manifest::inspect(ok.path, false, "synth-r1");
  const auto m_hdr2 = Qwen38Manifest::inspect(ok.path, false, "synth-r1");
  const auto m_self = Qwen38Manifest::inspect(ok.path, true, "synth-r1");
  CHECK(m_hdr.checkpoint.format == ingine::model::CheckpointFormat::Safetensors);
  CHECK(m_hdr.checkpoint.schema_variant == "qwen3_5_bf16");
  CHECK(m_hdr.checkpoint.verification_level == ingine::model::VerificationLevel::HeadersOnly);
  CHECK(m_hdr.checkpoint.weight_manifest_verified);
  CHECK(!m_hdr.checkpoint.shard_digests_verified);
  CHECK(m_self.checkpoint.verification_level == ingine::model::VerificationLevel::SelfHashed);
  // Self-computed hashes NEVER unlock shard_digests_verified or state reuse without an external pin spec.
  CHECK(!m_self.checkpoint.shard_digests_verified);
  CHECK(m_self.checkpoint.weight_manifest_sha256 == m_hdr.checkpoint.weight_manifest_sha256);
  CHECK(m_hdr.checkpoint.shard_set_sha256 == m_hdr2.checkpoint.shard_set_sha256);
  CHECK(m_self.checkpoint.shard_set_sha256 != m_hdr.checkpoint.shard_set_sha256);
  CHECK((m_self.checkpoint.shard_set_sha256 != std::array<std::uint8_t, 32>{}));
  CHECK(m_self.shard_sha256.at("model.safetensors").size() == 64);
  CHECK(m_self.active_text_tensor_count == 27);
  CHECK(m_self.excluded_visual_tensor_count == 1);
  CHECK(m_self.excluded_mtp_tensor_count == 1);

  ingine::model::RuntimeOptions bf16_opts;
  bf16_opts.dense_fp8 = false;
  bf16_opts.max_ctx = 128;
  // HeadersOnly and SelfHashed profiles are strictly non-reusable.
  CHECK(!make_qwen38_state_identity(m_hdr, bf16_opts).is_reusable_with(make_qwen38_state_identity(m_hdr, bf16_opts)));
  CHECK(!make_qwen38_state_identity(m_self, bf16_opts).is_reusable_with(make_qwen38_state_identity(m_self, bf16_opts)));

  // Reject dense_fp8=true on BF16, max_ctx > model limit, and unsupported KV/offload/hook options fail-closed.
  {
    ingine::model::RuntimeOptions bad_fp8;
    bad_fp8.dense_fp8 = true;
    bad_fp8.max_ctx = 128;
    CHECK_THROWS(make_qwen38_worker_profile(m_self, bad_fp8), std::invalid_argument);
  }
  {
    ingine::model::RuntimeOptions bad_ctx = bf16_opts;
    bad_ctx.max_ctx = 256;
    CHECK_THROWS(validate_qwen38_runtime_options(m_self.config, bad_ctx), std::invalid_argument);
  }
  for (const auto& mutator : std::vector<std::function<void(ingine::model::RuntimeOptions&)>>{
           [](auto& o) { o.kv_vram_mb = 256; },
           [](auto& o) { o.park_mb = 512; },
           [](auto& o) { o.max_kv_tokens = 1024; },
           [](auto& o) { o.refusal_map = "/tmp/refusal.json"; },
       }) {
    ingine::model::RuntimeOptions bad = bf16_opts;
    mutator(bad);
    CHECK_THROWS(validate_qwen38_runtime_options(m_self.config, bad), std::invalid_argument);
  }

  // External expected-vs-consumed pinning: ONLY inspect_pinned() unlocks reuse!
  ingine::model::CheckpointPinSpec pins;
  pins.config_sha256 = m_self.checkpoint.config_sha256;
  pins.tokenizer_sha256 = m_self.checkpoint.tokenizer_sha256;
  pins.chat_template_sha256 = m_self.checkpoint.chat_template_sha256;
  pins.shards = {
      {"model.safetensors", m_self.shard_sizes.at("model.safetensors"),
       ingine::model::parse_sha256_hex(m_self.shard_sha256.at("model.safetensors"))},
  };
  const auto m_pinned = Qwen38Manifest::inspect_pinned(ok.path, pins, "synth-r1");
  CHECK(m_pinned.checkpoint.verification_level == ingine::model::VerificationLevel::PinnedVerified);
  CHECK(m_pinned.checkpoint.shard_digests_verified);
  const auto prof = make_qwen38_worker_profile(m_pinned, bf16_opts);
  CHECK(prof.family == ingine::model::ModelFamily::Qwen3_5);
  CHECK(prof.state_identity.state_layout_id == "qwen3_5_gdn_kv_v1");
  CHECK(prof.state_identity.act_dtype == ingine::model::ActivationDtype::Bf16);
  CHECK(prof.state_identity.is_reusable_with(prof.state_identity));

  // Mutate 1 byte inside active embed_tokens.weight payload on disk (leaving headers untouched):
  //   - HeadersOnly inspect(false) has identical header metadata yet MUST NOT reuse.
  //   - SelfHashed inspect(true) changes shard_set_sha256 and MUST NOT reuse.
  //   - inspect_pinned(pins) catches the 1-byte mutation fail-closed.
  {
    const std::filesystem::path st_path = ok.path / "model.safetensors";
    const auto st_size = std::filesystem::file_size(st_path);
    std::fstream rw(st_path, std::ios::in | std::ios::out | std::ios::binary);
    CHECK(rw.good());
    rw.seekg(static_cast<std::streamoff>(st_size - 128), std::ios::beg);
    char b = 0;
    rw.read(&b, 1);
    b ^= 0x01;
    rw.seekp(static_cast<std::streamoff>(st_size - 128), std::ios::beg);
    rw.write(&b, 1);
    rw.flush();
    rw.close();

    const auto mut_hdr = Qwen38Manifest::inspect(ok.path, false, "synth-r1");
    CHECK(!make_qwen38_state_identity(mut_hdr, bf16_opts).is_reusable_with(prof.state_identity));
    const auto mut_self = Qwen38Manifest::inspect(ok.path, true, "synth-r1");
    CHECK(mut_self.checkpoint.shard_set_sha256 != m_pinned.checkpoint.shard_set_sha256);
    CHECK(!make_qwen38_state_identity(mut_self, bf16_opts).is_reusable_with(prof.state_identity));
    CHECK_THROWS(Qwen38Manifest::inspect_pinned(ok.path, pins, "synth-r1"), std::invalid_argument);
  }

  // Verify pinned 27B spec carries 3 expected shards + config/tokenizer/chat_template digests.
  const auto spec27 = qwen38_27b_nvfp4_pin_spec();
  CHECK(spec27.shards.size() == 3);
  CHECK(spec27.config_sha256.has_value() && spec27.tokenizer_sha256.has_value() &&
        spec27.chat_template_sha256.has_value());

  TempDir stray("stray");
  write_synthetic_safetensors_dir(stray.path, true);
  CHECK_THROWS(Qwen38Manifest::inspect(stray.path, false), std::invalid_argument);
}

void test_real_checkpoints() {
  const std::filesystem::path q27("/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f");
  if (std::filesystem::is_directory(q27)) {
    const auto m = Qwen38Manifest::inspect(q27, false, "482ca0f3832238542f8f5295dde86b5f22711d80");
    CHECK(m.config.num_hidden_layers == 64);
    CHECK(m.config.num_linear_layers == 48);
    CHECK(m.config.num_full_layers == 16);
    CHECK(m.config.hidden_size == 5120);
    CHECK(m.config.intermediate_size == 17408);
    CHECK(m.config.vocab_size == 248320);
    CHECK(m.config.rotary_dim == 64);
    CHECK(!m.config.tie_word_embeddings);
    CHECK(m.config.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4);
    CHECK(m.shard_files.size() == 3);
    CHECK(m.total_index_tensors == 2194);
    CHECK(m.active_text_tensor_count == 1846);
    CHECK(m.excluded_visual_tensor_count == 333);
    CHECK(m.excluded_mtp_tensor_count == 15);
    CHECK(m.memory_budget.text_weight_bytes == 20150569096ull);
    CHECK(m.memory_budget.excluded_visual_bytes == 921460192ull);
    CHECK(m.memory_budget.excluded_mtp_bytes == 849398784ull);
    CHECK(m.memory_budget.gdn_recurrent_bytes == 150994944ull);
    CHECK(m.memory_budget.gdn_conv_bytes == 3932160ull);
    CHECK(m.memory_budget.kv_bytes_per_token == 65536ull);
    CHECK(m.checkpoint.format == ingine::model::CheckpointFormat::Safetensors);
    CHECK(m.checkpoint.schema_variant == "qwen3_5_mixed_fp8_nvfp4");
    CHECK(m.checkpoint.shard_count == 3);
    CHECK(m.checkpoint.weight_manifest_verified);
    CHECK(!m.checkpoint.shard_digests_verified);
    CHECK(hex_digest(m.checkpoint.config_sha256) ==
          "71fa01bb64de20971045cd82a82c0ebb25a54e1b0a3f09ef23cf5479e9c54df8");
    CHECK(hex_digest(m.checkpoint.tokenizer_sha256) ==
          "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3");
    CHECK(hex_digest(m.checkpoint.chat_template_sha256) ==
          "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041");
    std::printf("  27B manifest verified: active=%zu text_bytes=%llu\n", m.active_text_tensor_count,
                static_cast<unsigned long long>(m.memory_budget.text_weight_bytes));
  }

  const std::filesystem::path q08("/mnt/storage/models/Qwen3.5-0.8B");
  if (std::filesystem::is_directory(q08)) {
    const auto m = Qwen38Manifest::inspect(q08, false, "local");
    CHECK(m.config.num_hidden_layers == 24);
    CHECK(m.config.num_linear_layers == 18);
    CHECK(m.config.num_full_layers == 6);
    CHECK(m.config.tie_word_embeddings);
    CHECK(m.config.weight_precision == Qwen38WeightPrecision::Bf16);
    CHECK(m.total_index_tensors == 488);
    CHECK(m.active_text_tensor_count == 320);
    CHECK(m.excluded_visual_tensor_count == 153);
    CHECK(m.excluded_mtp_tensor_count == 15);
    CHECK(m.memory_budget.text_weight_bytes == 1504791232ull);
    std::printf("  0.8B manifest verified: active=%zu text_bytes=%llu\n", m.active_text_tensor_count,
                static_cast<unsigned long long>(m.memory_budget.text_weight_bytes));
  }
}

}  // namespace

int main() {
  test_config_validation();
  test_synthetic_manifest();
  test_real_checkpoints();
  std::printf("qwen38_manifest: PASS\n");
  return 0;
}

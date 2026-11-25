#include "backends/qwen38/qwen38_config.h"
#include "backends/qwen38/qwen38_manifest.h"
#include "backends/qwen38/qwen38_backend.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <unistd.h>
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

void write_synthetic_safetensors_dir(const std::filesystem::path& dir, bool corrupt_extra_tensor = false,
                                     std::uint64_t extra_excluded_visual_bytes = 0) {
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
  if (extra_excluded_visual_bytes > 0) {
    specs.push_back(
        {"model.visual.large_blob.weight", "U8", {static_cast<std::int64_t>(extra_excluded_visual_bytes)}, 1});
  }
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
  const std::filesystem::path out_path = dir / "model.safetensors";
  if (extra_excluded_visual_bytes == 0) {
    std::vector<std::uint8_t> st(8 + hdr_len + offset, 0u);
    for (int i = 0; i < 8; ++i) st[i] = static_cast<std::uint8_t>((hdr_len >> (8 * i)) & 0xffu);
    std::memcpy(st.data() + 8, hdr_str.data(), hdr_len);
    write_bytes(out_path, st.data(), st.size());
  } else {
    const int fd = ::open(out_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK(fd >= 0);
    std::array<std::uint8_t, 8> len_buf{};
    for (int i = 0; i < 8; ++i) len_buf[i] = static_cast<std::uint8_t>((hdr_len >> (8 * i)) & 0xffu);
    CHECK(::write(fd, len_buf.data(), 8) == 8);
    CHECK(::write(fd, hdr_str.data(), hdr_len) == static_cast<ssize_t>(hdr_len));
    const std::uint64_t total_size = 8ull + hdr_len + offset;
    CHECK(::ftruncate(fd, static_cast<off_t>(total_size)) == 0);
    const std::uint8_t marker = 0x5au;
    CHECK(::pwrite(fd, &marker, 1, static_cast<off_t>(total_size - 1)) == 1);
    ::close(fd);
  }
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

  // Unadvertised gear or kv_cache_mode must reject fail-closed (never return reusable=true).
  CHECK_THROWS(make_qwen38_state_identity(m_pinned, bf16_opts, "dense_bf16", "sparse"), std::invalid_argument);
  CHECK_THROWS(make_qwen38_state_identity(m_pinned, bf16_opts, "paged_q4", "dense"), std::invalid_argument);
  CHECK_THROWS(make_qwen38_worker_profile(m_pinned, bf16_opts, "dense_bf16", "sparse"), std::invalid_argument);
  CHECK_THROWS(make_qwen38_worker_profile(m_pinned, bf16_opts, "paged_q4", "dense"), std::invalid_argument);

  // Effective execution config (e.g. rms_norm_eps) MUST bind into BackendStateIdentity and fail reuse if altered.
  {
    Qwen38Manifest m_eps = m_pinned;
    m_eps.config.rms_norm_eps = 2e-6f;
    const auto id_eps = make_qwen38_state_identity(m_eps, bf16_opts);
    CHECK(id_eps != prof.state_identity);
    CHECK(!prof.state_identity.is_reusable_with(id_eps));
    CHECK(!id_eps.is_reusable_with(prof.state_identity));
    CHECK(!id_eps.is_reusable_with(id_eps));
  }

  // Verified FD/bytes tracking from single-pass inspect_pinned.
  CHECK(m_pinned.verified_files.count("config.json") == 1 && m_pinned.verified_files.at("config.json").payload_hashed);
  CHECK(m_pinned.verified_files.count("tokenizer.json") == 1 &&
        m_pinned.verified_files.at("tokenizer.json").payload_hashed);
  CHECK(m_pinned.verified_files.count("model.safetensors") == 1 &&
        m_pinned.verified_files.at("model.safetensors").payload_hashed);
  CHECK(!m_pinned.read_verified_file_bytes(ok.path, "config.json").empty());

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

  // Concurrent replacement with identical bytes at a new inode MUST fail open-FD identity verification.
  {
    const std::string cfg_bytes = m_pinned.read_verified_file_bytes(ok.path, "config.json");
    write_str(ok.path / "config.json.new", cfg_bytes);
    std::filesystem::rename(ok.path / "config.json.new", ok.path / "config.json");
    CHECK_THROWS(m_pinned.read_verified_file_bytes(ok.path, "config.json"), std::invalid_argument);
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
    for (std::int32_t mid : {248053, 248054, 248055, 248056, 248057, 248070, 248071, 248076}) {
      CHECK(std::find(m.media_token_ids.begin(), m.media_token_ids.end(), mid) != m.media_token_ids.end());
    }
    const auto real_prof = make_qwen38_worker_profile(m, ingine::model::RuntimeOptions{});
    for (std::int32_t blocked_id : {248053, 248054, 248055, 248056, 248057, 248076}) {
      ingine::model::Prompt p;
      const std::array<std::int32_t, 3> ids = {10, blocked_id, 11};
      p.ids = ids;
      CHECK_THROWS(ingine::model::validate_request_against_capabilities(
                       real_prof.capabilities, real_prof.max_ctx, p, 8, ingine::model::Sampling{}),
                   std::invalid_argument);
    }
    CHECK(std::find(m.shard_files.begin(), m.shard_files.end(), "model-00001-of-00003.safetensors") !=
          m.shard_files.end());
    for (const std::string& shard : m.shard_files) {
      CHECK(m.shard_sizes.at(shard) > 256ull * 1024ull * 1024ull);
      // Header-only manifest MUST fail closed under both read_verified_file_bytes and stream_verified_file.
      CHECK_THROWS(m.read_verified_file_bytes(q27, shard), std::invalid_argument);
      bool admitted = false;
      bool aborted = false;
      CHECK_THROWS(m.stream_verified_file(
                       q27, shard, [](std::uint64_t, std::span<const std::uint8_t>) {},
                       [&](const Qwen38StreamResult&) { admitted = true; }, [&]() { aborted = true; }),
                   std::invalid_argument);
      CHECK(!admitted && aborted);
    }
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

void test_bounded_streaming_and_same_size_mutation() {
  // 1. Unpinned manifests (HeadersOnly / SelfHashed) MUST NOT admit verified reads/streams or verify_open_fd.
  {
    TempDir unpinned("stream_unpinned");
    write_synthetic_safetensors_dir(unpinned.path, false);
    const auto m_hdr = Qwen38Manifest::inspect(unpinned.path, false, "r1");
    const auto m_self = Qwen38Manifest::inspect(unpinned.path, true, "r1");
    for (const auto* mp : {&m_hdr, &m_self}) {
      bool admitted = false;
      bool aborted = false;
      CHECK_THROWS(mp->stream_verified_file(
                       unpinned.path, "model.safetensors", [](std::uint64_t, std::span<const std::uint8_t>) {},
                       [&](const Qwen38StreamResult&) { admitted = true; }, [&]() { aborted = true; }),
                   std::invalid_argument);
      CHECK(!admitted && aborted);
      CHECK_THROWS(mp->read_verified_file_bytes(unpinned.path, "config.json"), std::invalid_argument);
      const int fd = ::open((unpinned.path / "model.safetensors").c_str(), O_RDONLY);
      CHECK(fd >= 0);
      CHECK_THROWS(mp->verify_open_fd(fd, "model.safetensors"), std::invalid_argument);
      ::close(fd);
    }
  }

  // 2. Ordinary same-size in-place shard update MUST fail both verify_open_fd and stream_verified_file (no admit).
  {
    TempDir same_sz("stream_same_size");
    write_synthetic_safetensors_dir(same_sz.path, false);
    const auto m_self = Qwen38Manifest::inspect(same_sz.path, true, "r1");
    ingine::model::CheckpointPinSpec pins;
    pins.config_sha256 = m_self.checkpoint.config_sha256;
    pins.tokenizer_sha256 = m_self.checkpoint.tokenizer_sha256;
    pins.chat_template_sha256 = m_self.checkpoint.chat_template_sha256;
    pins.shards = {
        {"model.safetensors", m_self.shard_sizes.at("model.safetensors"),
         ingine::model::parse_sha256_hex(m_self.shard_sha256.at("model.safetensors"))},
    };
    const auto m_pin = Qwen38Manifest::inspect_pinned(same_sz.path, pins, "r1");

    // Good state: both verify_open_fd and stream_verified_file pass and fire on_admit.
    const std::filesystem::path st_path = same_sz.path / "model.safetensors";
    {
      const int fd = ::open(st_path.c_str(), O_RDONLY);
      CHECK(fd >= 0);
      m_pin.verify_open_fd(fd, "model.safetensors");
      ::close(fd);
    }
    {
      std::uint64_t seen = 0;
      bool admitted = false;
      bool aborted = false;
      const auto res = m_pin.stream_verified_file(
          same_sz.path, "model.safetensors",
          [&](std::uint64_t off, std::span<const std::uint8_t> ch) {
            CHECK(off == seen);
            seen += ch.size();
          },
          [&](const Qwen38StreamResult& r) {
            CHECK(r.admitted && r.bytes_delivered == seen);
            admitted = true;
          },
          [&]() { aborted = true; }, 1024);
      CHECK(admitted && !aborted && res.admitted && res.bytes_delivered == m_pin.shard_sizes.at("model.safetensors"));
    }

    // Mutate 1 byte in-place keeping the exact same inode and byte size.
    const int rwfd = ::open(st_path.c_str(), O_RDWR);
    CHECK(rwfd >= 0);
    std::uint8_t byte = 0;
    const off_t mut_off = static_cast<off_t>(m_pin.shard_sizes.at("model.safetensors") - 64);
    CHECK(::pread(rwfd, &byte, 1, mut_off) == 1);
    byte ^= 0x01u;
    CHECK(::pwrite(rwfd, &byte, 1, mut_off) == 1);
    ::close(rwfd);

    const int rofd = ::open(st_path.c_str(), O_RDONLY);
    CHECK(rofd >= 0);
    CHECK_THROWS(m_pin.verify_open_fd(rofd, "model.safetensors"), std::invalid_argument);
    ::close(rofd);

    bool admitted = false;
    bool aborted = false;
    CHECK_THROWS(m_pin.stream_verified_file(
                     same_sz.path, "model.safetensors", [](std::uint64_t, std::span<const std::uint8_t>) {},
                     [&](const Qwen38StreamResult&) { admitted = true; }, [&]() { aborted = true; }, 1024),
                 std::invalid_argument);
    CHECK(!admitted && aborted);
  }

  // 3. >256 MiB shard bounded streaming: rejects read_verified_file_bytes(), streams in <=1 MiB chunks,
  //    admits only on full pin match, and catches same-size in-place mutation past 256 MiB.
  {
    TempDir big("stream_over_256m");
    constexpr std::uint64_t kLargeBlobBytes = 260ull * 1024ull * 1024ull;
    write_synthetic_safetensors_dir(big.path, false, kLargeBlobBytes);
    const auto m_self = Qwen38Manifest::inspect(big.path, true, "r1");
    CHECK(m_self.shard_sizes.at("model.safetensors") > 256ull * 1024ull * 1024ull);

    ingine::model::CheckpointPinSpec pins;
    pins.config_sha256 = m_self.checkpoint.config_sha256;
    pins.tokenizer_sha256 = m_self.checkpoint.tokenizer_sha256;
    pins.chat_template_sha256 = m_self.checkpoint.chat_template_sha256;
    pins.shards = {
        {"model.safetensors", m_self.shard_sizes.at("model.safetensors"),
         ingine::model::parse_sha256_hex(m_self.shard_sha256.at("model.safetensors"))},
    };
    const auto m_pin = Qwen38Manifest::inspect_pinned(big.path, pins, "r1");

    // 256 MiB metadata reader rejects >256 MiB production shards fail-closed.
    CHECK_THROWS(m_pin.read_verified_file_bytes(big.path, "model.safetensors"), std::invalid_argument);

    constexpr std::size_t kChunkBytes = 1024ull * 1024ull;
    std::uint64_t seen = 0;
    std::size_t max_chunk = 0;
    int admit_calls = 0;
    bool aborted = false;
    const auto res = m_pin.stream_verified_file(
        big.path, "model.safetensors",
        [&](std::uint64_t off, std::span<const std::uint8_t> ch) {
          CHECK(off == seen);
          CHECK(!ch.empty() && ch.size() <= kChunkBytes);
          seen += ch.size();
          max_chunk = std::max(max_chunk, ch.size());
        },
        [&](const Qwen38StreamResult& r) {
          CHECK(r.admitted && r.bytes_delivered == m_pin.shard_sizes.at("model.safetensors"));
          ++admit_calls;
        },
        [&]() { aborted = true; }, kChunkBytes);
    CHECK(res.admitted && admit_calls == 1 && !aborted);
    CHECK(seen == m_pin.shard_sizes.at("model.safetensors") && max_chunk <= kChunkBytes);

    const std::filesystem::path st_path = big.path / "model.safetensors";
    {
      const int fd = ::open(st_path.c_str(), O_RDONLY);
      CHECK(fd >= 0);
      m_pin.verify_open_fd(fd, "model.safetensors", kChunkBytes);
      ::close(fd);
    }

    // Mutate 1 byte past 256 MiB in-place (same inode, same size).
    const int rwfd = ::open(st_path.c_str(), O_RDWR);
    CHECK(rwfd >= 0);
    std::uint8_t byte = 0;
    const off_t mut_off = static_cast<off_t>(m_pin.shard_sizes.at("model.safetensors") - 1);
    CHECK(::pread(rwfd, &byte, 1, mut_off) == 1);
    byte ^= 0x01u;
    CHECK(::pwrite(rwfd, &byte, 1, mut_off) == 1);
    ::close(rwfd);

    const int rofd = ::open(st_path.c_str(), O_RDONLY);
    CHECK(rofd >= 0);
    CHECK_THROWS(m_pin.verify_open_fd(rofd, "model.safetensors", kChunkBytes), std::invalid_argument);
    ::close(rofd);

    bool mut_admitted = false;
    bool mut_aborted = false;
    CHECK_THROWS(m_pin.stream_verified_file(
                     big.path, "model.safetensors", [](std::uint64_t, std::span<const std::uint8_t>) {},
                     [&](const Qwen38StreamResult&) { mut_admitted = true; }, [&]() { mut_aborted = true; },
                     kChunkBytes),
                 std::invalid_argument);
    CHECK(!mut_admitted && mut_aborted);
  }
}

}  // namespace

int main() {
  test_config_validation();
  test_synthetic_manifest();
  test_real_checkpoints();
  test_bounded_streaming_and_same_size_mutation();
  std::printf("qwen38_manifest: PASS\n");
  return 0;
}

#include "backends/qwen38/qwen38_backend.h"
#include "backends/qwen38/qwen38_config.h"
#include "backends/qwen38/qwen38_manifest.h"
#include "ingine/model_backend.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const char* tag) {
    path = std::filesystem::temp_directory_path() / (std::string("ingine_test_backend_") + tag);
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

void write_text_file(const std::filesystem::path& p, const std::string& body) {
  std::ofstream out(p, std::ios::binary);
  CHECK(out.good());
  out.write(body.data(), static_cast<std::streamsize>(body.size()));
}

using namespace ingine::model;
using namespace ingine::backends::qwen38;

void test_family_enum_and_detection() {
  CHECK(std::string(to_string(ModelFamily::Inkling2)) == "inkling2");
  CHECK(std::string(to_string(ModelFamily::Qwen3_5)) == "qwen3_5");
  CHECK(std::string(to_string(ModelFamily::DeepSeekV4)) == "deepseek_v4");
  CHECK(std::string(to_string(ModelFamily::GlmFlash)) == "glm_flash");
  CHECK(std::string(to_string(ModelFamily::Unknown)) == "unknown");

  CHECK(parse_model_family("inkling-2") == ModelFamily::Inkling2);
  CHECK(parse_model_family("qwen3.8") == ModelFamily::Qwen3_5);
  CHECK(parse_model_family("deepseek_v41") == ModelFamily::DeepSeekV4);
  CHECK(parse_model_family("glm53") == ModelFamily::GlmFlash);
  CHECK_THROWS(parse_model_family("llama99"), std::invalid_argument);

  CHECK(std::string(to_string(CheckpointFormat::Safetensors)) == "safetensors");
  CHECK(std::string(to_string(CheckpointFormat::Gguf)) == "gguf");
  CHECK(std::string(to_string(CheckpointFormat::Unknown)) == "unknown");
  CHECK(std::string(to_string(ActivationDtype::Bf16)) == "bf16");
  CHECK(std::string(to_string(ActivationDtype::Fp16)) == "fp16");
  CHECK(std::string(to_string(ActivationDtype::Fp32)) == "fp32");
  CHECK(std::string(to_string(ActivationDtype::Unknown)) == "unknown");

  const std::filesystem::path q27("/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f");
  if (std::filesystem::is_directory(q27)) {
    CHECK(detect_model_family(q27) == ModelFamily::Qwen3_5);
    CHECK(detect_checkpoint_format(q27) == CheckpointFormat::Safetensors);
  }
  const std::filesystem::path q08("/mnt/storage/models/Qwen3.5-0.8B");
  if (std::filesystem::is_directory(q08)) {
    CHECK(detect_model_family(q08) == ModelFamily::Qwen3_5);
    CHECK(detect_checkpoint_format(q08) == CheckpointFormat::Safetensors);
  }

  TempDir gguf_dir("gguf_fmt");
  write_text_file(gguf_dir.path / "model-00001-of-00002.gguf", "GGUF");
  CHECK(detect_checkpoint_format(gguf_dir.path) == CheckpointFormat::Gguf);

  TempDir mixed_dir("mixed_fmt");
  write_text_file(mixed_dir.path / "model.safetensors", "ST");
  write_text_file(mixed_dir.path / "model.gguf", "GGUF");
  CHECK_THROWS(detect_checkpoint_format(mixed_dir.path), std::invalid_argument);

  TempDir empty_dir("empty_fmt");
  CHECK_THROWS(detect_checkpoint_format(empty_dir.path), std::invalid_argument);
  CHECK_THROWS(detect_checkpoint_format(empty_dir.path / "missing"), std::invalid_argument);

  TempDir ds_dir("ds");
  write_text_file(ds_dir.path / "config.json", R"({"model_type":"deepseek_v41"})");
  CHECK(detect_model_family(ds_dir.path) == ModelFamily::DeepSeekV4);

  TempDir glm_dir("glm");
  write_text_file(glm_dir.path / "config.json", R"({"architectures":["GlmForCausalLM"]})");
  CHECK(detect_model_family(glm_dir.path) == ModelFamily::GlmFlash);

  TempDir ink_dir("ink");
  write_text_file(ink_dir.path / "tokenizer.json", R"({"added_tokens":[{"id":1,"content":"<|message_model|>"}]})");
  CHECK(detect_model_family(ink_dir.path) == ModelFamily::Inkling2);

  TempDir bad_dir("bad");
  write_text_file(bad_dir.path / "config.json", R"({"model_type":"mystery_arch"})");
  CHECK_THROWS(detect_model_family(bad_dir.path), std::invalid_argument);
  CHECK_THROWS(detect_model_family(bad_dir.path / "missing"), std::invalid_argument);
}

BackendStateIdentity make_verified_sample_identity() {
  BackendStateIdentity id;
  id.family = ModelFamily::Qwen3_5;
  id.backend_name = "qwen38-dense";
  id.checkpoint.format = CheckpointFormat::Safetensors;
  id.checkpoint.schema_variant = "qwen3_5_mixed_fp8_nvfp4";
  id.checkpoint.model_id = "Qwen3.8-27B-NVFP4-482ca0f";
  id.checkpoint.revision = "482ca0f3832238542f8f5295dde86b5f22711d80";
  id.checkpoint.shard_count = 3;
  id.checkpoint.shard_set_sha256[0] = 0x55;
  id.checkpoint.config_sha256[0] = 0x11;
  id.checkpoint.tokenizer_sha256[0] = 0x22;
  id.checkpoint.chat_template_sha256[0] = 0x33;
  id.checkpoint.weight_manifest_sha256[0] = 0x44;
  id.checkpoint.weight_manifest_verified = true;
  id.checkpoint.shard_digests_verified = true;
  id.state_layout_id = "qwen3_5_gdn_kv_v1";
  id.act_dtype = ActivationDtype::Bf16;
  id.weight_precision = "mixed_fp8_nvfp4";
  id.kv_format = "gdn_f32_conv_bf16_kv_bf16";
  id.kv_cache_mode = "dense_bf16";
  id.gear = "dense";
  id.max_ctx = 32768;
  id.prefill_chunk = 2048;
  id.spec_enabled = false;
  id.mtp_enabled = false;
  id.media_enabled = false;
  id.persona_key = 0;
  id.persona_fingerprint = "";
  id.cache_namespace = "ns0";
  id.rendering_fingerprint = "chatml_nothink";
  id.family_state_fingerprint = "gdn48_full16_conv4_rope64_v1";
  return id;
}

void test_identity_reusability_and_prefix_checks() {
  const BackendStateIdentity a = make_verified_sample_identity();
  CHECK(a.is_reusable_with(a));

  auto check_miss = [&](auto mutator) {
    BackendStateIdentity b = a;
    mutator(b);
    CHECK(!a.is_reusable_with(b));
    CHECK(hash_backend_state_identity(a) != hash_backend_state_identity(b));
  };

  {
    BackendStateIdentity unverified = a;
    unverified.checkpoint.weight_manifest_verified = false;
    CHECK(!unverified.is_reusable_with(a));
    CHECK(!unverified.is_reusable_with(unverified));
  }
  {
    BackendStateIdentity unknown_fam = a;
    unknown_fam.family = ModelFamily::Unknown;
    CHECK(!unknown_fam.is_reusable_with(unknown_fam));
  }
  {
    BackendStateIdentity empty_layout = a;
    empty_layout.state_layout_id.clear();
    CHECK(!empty_layout.is_reusable_with(empty_layout));
  }
  {
    BackendStateIdentity unknown_act = a;
    unknown_act.act_dtype = ActivationDtype::Unknown;
    CHECK(!unknown_act.is_reusable_with(unknown_act));
  }
  {
    BackendStateIdentity unknown_fmt = a;
    unknown_fmt.checkpoint.format = CheckpointFormat::Unknown;
    CHECK(!unknown_fmt.is_reusable_with(unknown_fmt));
  }
  {
    BackendStateIdentity zero_shards = a;
    zero_shards.checkpoint.shard_count = 0;
    CHECK(!zero_shards.is_reusable_with(zero_shards));
  }

  check_miss([](BackendStateIdentity& b) { b.checkpoint.format = CheckpointFormat::Gguf; });
  check_miss([](BackendStateIdentity& b) { b.checkpoint.schema_variant = "deepseek41_ple"; });
  check_miss([](BackendStateIdentity& b) { b.checkpoint.shard_count = 4; });
  check_miss([](BackendStateIdentity& b) { b.checkpoint.shard_set_sha256[0] ^= 1u; });
  check_miss([](BackendStateIdentity& b) { b.checkpoint.shard_digests_verified = false; });
  check_miss([](BackendStateIdentity& b) { b.checkpoint.weight_manifest_sha256[31] ^= 1u; });
  check_miss([](BackendStateIdentity& b) { b.act_dtype = ActivationDtype::Fp16; });
  check_miss([](BackendStateIdentity& b) { b.kv_cache_mode = "last_full"; });
  check_miss([](BackendStateIdentity& b) { b.gear = "qsa_v2"; });
  check_miss([](BackendStateIdentity& b) { b.prefill_chunk = 3072; });
  check_miss([](BackendStateIdentity& b) { b.mtp_enabled = true; });
  check_miss([](BackendStateIdentity& b) { b.media_enabled = true; });
  check_miss([](BackendStateIdentity& b) { b.persona_key = 42; });
  check_miss([](BackendStateIdentity& b) { b.cache_namespace = "ns1"; });
  check_miss([](BackendStateIdentity& b) { b.rendering_fingerprint = "chatml_think"; });
  check_miss([](BackendStateIdentity& b) { b.family_state_fingerprint = "gdn48_full16_conv4_rope64_v2"; });

  const std::vector<std::int32_t> cached_ids = {10, 20, 30, 40};
  const std::vector<std::int32_t> prompt_match = {10, 20, 30, 40, 50};
  const std::vector<std::int32_t> prompt_mismatch = {10, 20, 99, 40, 50};
  CHECK(prefix_matches_exact(cached_ids, prompt_match, 4));
  CHECK(!prefix_matches_exact(cached_ids, prompt_mismatch, 4));
  CHECK(!prefix_matches_exact(cached_ids, prompt_match, 5));

  const std::vector<std::uint8_t> cached_mask = {1, 1, 0, 1};
  const std::vector<std::uint8_t> prompt_mask_ok = {1, 1, 0, 1, 1};
  const std::vector<std::uint8_t> prompt_mask_bad = {1, 1, 1, 1, 1};
  CHECK(prefix_matches_exact_with_mask(cached_ids, cached_mask, prompt_match, prompt_mask_ok, 4));
  CHECK(!prefix_matches_exact_with_mask(cached_ids, cached_mask, prompt_match, prompt_mask_bad, 4));
  CHECK(!prefix_matches_exact_with_mask(cached_ids, cached_mask, prompt_mismatch, prompt_mask_ok, 4));
  CHECK(!prefix_matches_exact_with_mask(cached_ids, cached_mask, prompt_match, prompt_mask_ok, 5));
}

void test_capabilities_and_fail_closed_validation() {
  const auto ink = inkling_default_capabilities();
  CHECK(ink.family == ModelFamily::Inkling2);
  CHECK(ink.supports_images);
  CHECK(ink.supports_speculation);
  CHECK(ink.supports_continuous_batching);
  CHECK(ink.max_supported_batch == 64);

  Qwen38Config qcfg;
  qcfg.model_type = "qwen3_5";
  qcfg.architectures = {"Qwen3_5ForConditionalGeneration"};
  qcfg.vocab_size = 248320;
  qcfg.hidden_size = 5120;
  qcfg.intermediate_size = 17408;
  qcfg.num_hidden_layers = 4;
  qcfg.max_position_embeddings = 262144;
  qcfg.num_attention_heads = 24;
  qcfg.num_key_value_heads = 4;
  qcfg.head_dim = 256;
  qcfg.rotary_dim = 64;
  qcfg.linear_conv_kernel_dim = 4;
  qcfg.linear_key_head_dim = 128;
  qcfg.linear_value_head_dim = 128;
  qcfg.linear_num_key_heads = 16;
  qcfg.linear_num_value_heads = 48;
  qcfg.layer_types = {Qwen38LayerKind::LinearAttention, Qwen38LayerKind::LinearAttention,
                      Qwen38LayerKind::LinearAttention, Qwen38LayerKind::FullAttention};
  qcfg.num_linear_layers = 3;
  qcfg.num_full_layers = 1;
  qcfg.weight_precision = Qwen38WeightPrecision::MixedFp8Nvfp4;
  qcfg.quant_method = "modelopt";
  qcfg.quant_algo = "MIXED_PRECISION";

  const BackendCapabilities qcaps = qwen38_dense_capabilities(qcfg);
  CHECK(qcaps.family == ModelFamily::Qwen3_5);
  CHECK(qcaps.supports_text);
  CHECK(!qcaps.supports_images);
  CHECK(!qcaps.supports_speculation);
  CHECK(!qcaps.supports_continuous_batching);
  CHECK(qcaps.max_supported_batch == 1);

  RuntimeOptions base_opts;
  base_opts.max_ctx = 32768;
  base_opts.prefill_chunk = 2048;
  validate_qwen38_runtime_options(qcfg, base_opts);

  auto reject_opts = [&](auto mutator) {
    RuntimeOptions bad = base_opts;
    mutator(bad);
    CHECK_THROWS(validate_qwen38_runtime_options(qcfg, bad), std::invalid_argument);
  };
  reject_opts([](RuntimeOptions& o) { o.spec = true; });
  reject_opts([](RuntimeOptions& o) { o.vision = true; });
  reject_opts([](RuntimeOptions& o) { o.ablation = "/tmp/ablate.json"; });
  reject_opts([](RuntimeOptions& o) { o.enable_persona_hooks = true; });
  reject_opts([](RuntimeOptions& o) { o.max_batch = 2; });
  reject_opts([](RuntimeOptions& o) { o.kv_fp8 = true; });
  reject_opts([](RuntimeOptions& o) { o.kv_q4 = true; });
  reject_opts([](RuntimeOptions& o) { o.kv_paged = true; });
  reject_opts([](RuntimeOptions& o) { o.offload_vram_mb = 1024; });
  reject_opts([](RuntimeOptions& o) { o.kv_ckpt_path = "/tmp/ckpt"; });
  reject_opts([](RuntimeOptions& o) { o.max_ctx = 262145; });

  std::vector<std::int32_t> ok_ids = {1, 2, 3};
  Prompt ok_prompt;
  ok_prompt.ids = ok_ids;
  Sampling ok_samp;
  validate_request_against_capabilities(qcaps, 1024, ok_prompt, 16, ok_samp);

  Prompt empty_p;
  CHECK_THROWS(validate_request_against_capabilities(qcaps, 1024, empty_p, 16, ok_samp), std::invalid_argument);
  CHECK_THROWS(validate_request_against_capabilities(qcaps, 1024, ok_prompt, 0, ok_samp), std::invalid_argument);

  Prompt img_p = ok_prompt;
  std::vector<PromptImage> imgs(1);
  img_p.images = imgs;
  CHECK_THROWS(validate_request_against_capabilities(qcaps, 1024, img_p, 16, ok_samp), std::invalid_argument);

  CHECK_THROWS(validate_request_against_capabilities(qcaps, 3, ok_prompt, 16, ok_samp), std::invalid_argument);

  std::vector<std::int32_t> oov_ids = {1, 248320};
  Prompt oov_p;
  oov_p.ids = oov_ids;
  CHECK_THROWS(validate_request_against_capabilities(qcaps, 1024, oov_p, 16, ok_samp), std::out_of_range);

  Sampling bad_banned;
  bad_banned.banned_tokens = {-1};
  CHECK_THROWS(validate_request_against_capabilities(qcaps, 1024, ok_prompt, 16, bad_banned), std::out_of_range);

  std::vector<std::int32_t> forced_oov = {999999};
  Sampling bad_forced;
  bad_forced.forced = forced_oov;
  CHECK_THROWS(validate_request_against_capabilities(qcaps, 1024, ok_prompt, 16, bad_forced), std::out_of_range);
}

void test_registry() {
  auto& reg = BackendRegistry::instance();
  CHECK_THROWS(reg.register_backend({ModelFamily::Unknown, "bad", nullptr}), std::invalid_argument);
  CHECK_THROWS(reg.create(ModelFamily::Unknown, RuntimeOptions{}), std::invalid_argument);
  if (reg.find(ModelFamily::DeepSeekV4) == nullptr) {
    CHECK_THROWS(reg.create(ModelFamily::DeepSeekV4, RuntimeOptions{}), std::invalid_argument);
  }
}

}  // namespace

int main() {
  test_family_enum_and_detection();
  test_identity_reusability_and_prefix_checks();
  test_capabilities_and_fail_closed_validation();
  test_registry();
  std::printf("backend_model_contract: PASS\n");
  return 0;
}

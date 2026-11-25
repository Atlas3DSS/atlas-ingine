#include "backends/qwen38/qwen38_backend.h"
#include "backends/qwen38/qwen38_config.h"
#include "backends/qwen38/qwen38_manifest.h"
#include "ingine/model_backend.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
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

void append_u32(std::vector<std::uint8_t>& buf, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) buf.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu));
}

void append_u64(std::vector<std::uint8_t>& buf, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) buf.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu));
}

void append_gguf_str(std::vector<std::uint8_t>& buf, std::string_view s) {
  append_u64(buf, static_cast<std::uint64_t>(s.size()));
  buf.insert(buf.end(), s.begin(), s.end());
}

void write_synthetic_gguf(const std::filesystem::path& p, const std::string& arch_value, bool include_arch = true) {
  std::vector<std::uint8_t> buf = {'G', 'G', 'U', 'F'};
  append_u32(buf, 3);  // version
  append_u64(buf, 0);  // tensor_count
  append_u64(buf, include_arch ? 2 : 1);
  append_gguf_str(buf, "split.no");
  append_u32(buf, 4);  // UINT32
  append_u32(buf, 1);
  if (include_arch) {
    append_gguf_str(buf, "general.architecture");
    append_u32(buf, 8);  // STRING
    append_gguf_str(buf, arch_value);
  }
  std::ofstream out(p, std::ios::binary);
  CHECK(out.good());
  out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

using namespace ingine::model;
using namespace ingine::backends::qwen38;

BackendStateIdentity make_verified_sample_identity() {
  BackendStateIdentity id;
  id.family = ModelFamily::Qwen3_5;
  id.backend_name = "qwen38-dense";
  id.checkpoint.format = CheckpointFormat::Safetensors;
  id.checkpoint.verification_level = VerificationLevel::PinnedVerified;
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
  id.numeric_abi = "act=bf16;attn=fp8_e4m3_cal;mlp=nvfp4_w4a16;gdn=f32;kv=bf16";
  id.weight_precision = "mixed_fp8_nvfp4";
  id.kv_format = "gdn_f32_conv_bf16_kv_bf16";
  id.kv_cache_mode = "dense_bf16";
  id.gear = "dense";
  id.max_ctx = 32768;
  id.prefill_chunk = 2048;
  id.spec_enabled = false;
  id.mtp_enabled = false;
  id.media_enabled = false;
  id.media_sha256 = {};
  id.persona_key = 0;
  id.persona_fingerprint = "";
  id.cache_namespace = "ns0";
  id.rendering_fingerprint = "chatml_nothink";
  id.family_state_fingerprint = "gdn48_full16_conv4_rope64_v1";
  return id;
}

// Deterministic CPU fake BackendSession & ModelBackend for contract/reuse verification (#827/#833/#834).
class FakeCpuSession final : public BackendSession {
 public:
  static constexpr int kVocab = 64;

  explicit FakeCpuSession(BackendStateIdentity id) : id_(std::move(id)) { reset(); }

  void reset() override {
    tokens_.clear();
    mask_.clear();
    acc_ = {0x9e3779b97f4a7c15ull, 0xbf58476d1ce4e5b9ull};
    history_.clear();
  }

  std::size_t state_bytes() const noexcept override {
    return tokens_.size() * (sizeof(std::int32_t) + sizeof(std::uint8_t) + sizeof(AccState));
  }

  std::span<const std::int32_t> committed_tokens() const noexcept override { return tokens_; }
  std::span<const std::uint8_t> committed_token_mask() const noexcept override { return mask_; }
  const BackendStateIdentity& identity() const noexcept override { return id_; }

  void truncate(std::size_t new_token_count) override {
    if (new_token_count > tokens_.size()) {
      throw std::out_of_range("truncate exceeds committed token count");
    }
    if (new_token_count == 0) {
      reset();
      return;
    }
    tokens_.resize(new_token_count);
    mask_.resize(new_token_count);
    history_.resize(new_token_count);
    acc_ = history_.back();
  }

  void set_identity(BackendStateIdentity id) { id_ = std::move(id); }

  void advance(std::int32_t tok, std::uint8_t mask_bit) {
    // Incorporate mask + media digest deterministically so any mismatch changes downstream logits.
    std::uint64_t media_mix = 0;
    for (std::size_t i = 0; i < 8; ++i) media_mix |= static_cast<std::uint64_t>(id_.media_sha256[i]) << (8 * i);
    acc_.s0 ^= static_cast<std::uint64_t>(tok) * 0x9e3779b97f4a7c15ull ^ (static_cast<std::uint64_t>(mask_bit) << 56);
    acc_.s0 = (acc_.s0 ^ (acc_.s0 >> 30)) * 0xbf58476d1ce4e5b9ull + media_mix;
    acc_.s1 += acc_.s0 ^ (static_cast<std::uint64_t>(tokens_.size()) * 0x94d049bb133111ebull);
    tokens_.push_back(tok);
    mask_.push_back(mask_bit);
    history_.push_back(acc_);
  }

  std::vector<float> current_logits() const {
    std::vector<float> logits(kVocab, 0.0f);
    for (int v = 0; v < kVocab; ++v) {
      const double phase = static_cast<double>((acc_.s0 ^ (static_cast<std::uint64_t>(v) * 0x517cc1b727220a95ull)) &
                                               0xffffu) /
                           65536.0;
      const double freq = static_cast<double>((acc_.s1 + static_cast<std::uint64_t>(v) * 17u) % 97u) * 0.05;
      logits[static_cast<std::size_t>(v)] = static_cast<float>(std::sin(phase * 6.2831853 + freq) * 4.0);
    }
    return logits;
  }

  std::int32_t greedy_next() const {
    const auto logits = current_logits();
    int best = 0;
    for (int v = 1; v < kVocab; ++v) {
      if (logits[static_cast<std::size_t>(v)] > logits[static_cast<std::size_t>(best)]) best = v;
    }
    return best;
  }

 private:
  struct AccState {
    std::uint64_t s0 = 0;
    std::uint64_t s1 = 0;
  };

  BackendStateIdentity id_;
  std::vector<std::int32_t> tokens_;
  std::vector<std::uint8_t> mask_;
  AccState acc_{};
  std::vector<AccState> history_;
};

class FakeCpuModelBackend final : public ModelBackend {
 public:
  explicit FakeCpuModelBackend(bool support_images = true) {
    caps_.family = ModelFamily::DeepSeekV4;
    caps_.backend_name = "fake-cpu-ds";
    caps_.architecture = "DeepseekV41ForCausalLM";
    caps_.supports_text = true;
    caps_.supports_images = support_images;
    caps_.supports_prefix_cache = true;
    caps_.supports_logit_capture = true;
    caps_.supported_gears = {"dense"};
    caps_.supported_kv_cache_modes = {"dense_bf16"};
    caps_.supported_quant_layouts = {"q8_0"};
    caps_.media_token_ids = {129264};
    caps_.vocab_size = FakeCpuSession::kVocab;
    caps_.max_position_embeddings = 4096;

    id_ = make_verified_sample_identity();
    id_.family = ModelFamily::DeepSeekV4;
    id_.backend_name = caps_.backend_name;
    id_.checkpoint.format = CheckpointFormat::Gguf;
    id_.checkpoint.schema_variant = "deepseek41_engram";
    id_.checkpoint.model_id = "fake-ds-v41";
    id_.media_enabled = support_images;

    profile_.family = caps_.family;
    profile_.model_id = id_.checkpoint.model_id;
    profile_.gear = id_.gear;
    profile_.kv_cache_mode = id_.kv_cache_mode;
    profile_.max_ctx = 4096;
    profile_.max_batch = 1;
    profile_.capabilities = caps_;
    profile_.state_identity = id_;

    session_ = std::make_unique<FakeCpuSession>(id_);
  }

  std::string model_id() const override { return id_.checkpoint.model_id; }
  int max_model_len() const override { return 4096; }
  bool supports_images() const override { return caps_.supports_images; }

  const BackendCapabilities& capabilities() const override { return caps_; }
  const WorkerProfile& worker_profile() const override { return profile_; }
  const BackendStateIdentity& state_identity() const override { return id_; }
  void reset_session() override { session_->reset(); }

  std::unique_ptr<BackendSession> create_session() override { return std::make_unique<FakeCpuSession>(id_); }
  BackendSession* acquire_session(int slot = 0) override {
    if (slot != 0) throw std::invalid_argument("FakeCpuModelBackend supports slot 0 only");
    return session_.get();
  }
  void release_session(int /*slot*/ = 0) override {}

  GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    BackendRequest req;
    req.prompt.ids = prompt;
    req.use_prefix_cache = true;
    return generate_request(req, max_new_tokens, sampling, stop_ids, on_tokens, nullptr);
  }

  GenerateStats generate(const Prompt& prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                         const PrefillCallback& on_prefill) override {
    BackendRequest req;
    req.prompt = prompt;
    req.use_prefix_cache = !prompt.images.empty() || true;
    return generate_request(req, max_new_tokens, sampling, stop_ids, on_tokens, on_prefill);
  }

  GenerateStats generate_request(const BackendRequest& req, int max_new_tokens, const Sampling& sampling,
                                 std::span<const std::int32_t> stop_ids,
                                 const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                                 const PrefillCallback& on_prefill) override {
    validate_request_against_capabilities(caps_, max_model_len(), req.prompt, max_new_tokens, sampling);
    if (!req.token_mask.empty() && req.token_mask.size() != req.prompt.ids.size()) {
      throw std::invalid_argument("token_mask length must equal prompt length");
    }

    auto* sess = dynamic_cast<FakeCpuSession*>(acquire_session(0));
    CHECK(sess != nullptr);

    BackendStateIdentity req_id = id_;
    req_id.cache_namespace = req.cache_namespace.empty() ? id_.cache_namespace : req.cache_namespace;
    req_id.media_sha256 = hash_prompt_media(req.prompt.images);

    const std::vector<std::uint8_t> default_mask(req.prompt.ids.size(), 1u);
    const std::span<const std::uint8_t> eff_mask =
        req.token_mask.empty() ? std::span<const std::uint8_t>(default_mask) : req.token_mask;

    const SessionLookupResult lookup = lookup_session_prefix(
        *sess, SessionLookupQuery{req_id, req.prompt.ids, eff_mask, req.prompt.images, req.use_prefix_cache});

    GenerateStats stats;
    stats.prompt_tokens = static_cast<int>(req.prompt.ids.size());
    stats.reuse = lookup.reuse;
    stats.reused_tokens = static_cast<int>(lookup.reused_tokens);

    if (!lookup.hit) {
      sess->reset();
      sess->set_identity(req_id);
    } else if (lookup.reused_tokens < sess->committed_tokens().size()) {
      sess->truncate(lookup.reused_tokens);
    }

    for (std::size_t i = lookup.reused_tokens; i < req.prompt.ids.size(); ++i) {
      sess->advance(req.prompt.ids[i], eff_mask[i]);
    }
    if (on_prefill) on_prefill(stats);

    for (int step = 0; step < max_new_tokens; ++step) {
      const std::int32_t next_tok = sess->greedy_next();
      const std::int32_t out_arr[1] = {next_tok};
      ++stats.generated_tokens;
      sess->advance(next_tok, 1u);
      const bool cont = on_tokens(out_arr);
      if (!cont || std::find(stop_ids.begin(), stop_ids.end(), next_tok) != stop_ids.end()) break;
    }
    release_session(0);
    return stats;
  }

  void evaluate_teacher_forced(const BackendRequest& req, std::span<const int> capture_positions,
                               const LogitCallback& on_logits) override {
    validate_request_against_capabilities(caps_, max_model_len(), req.prompt, 1, Sampling{});
    if (!req.token_mask.empty() && req.token_mask.size() != req.prompt.ids.size()) {
      throw std::invalid_argument("token_mask length must equal prompt length");
    }
    auto* sess = dynamic_cast<FakeCpuSession*>(acquire_session(0));
    CHECK(sess != nullptr);

    BackendStateIdentity req_id = id_;
    req_id.cache_namespace = req.cache_namespace.empty() ? id_.cache_namespace : req.cache_namespace;
    req_id.media_sha256 = hash_prompt_media(req.prompt.images);

    const std::vector<std::uint8_t> default_mask(req.prompt.ids.size(), 1u);
    const std::span<const std::uint8_t> eff_mask =
        req.token_mask.empty() ? std::span<const std::uint8_t>(default_mask) : req.token_mask;

    const SessionLookupResult lookup = lookup_session_prefix(
        *sess, SessionLookupQuery{req_id, req.prompt.ids, eff_mask, req.prompt.images, req.use_prefix_cache});

    if (!lookup.hit) {
      sess->reset();
      sess->set_identity(req_id);
    } else if (lookup.reused_tokens < sess->committed_tokens().size()) {
      sess->truncate(lookup.reused_tokens);
    }
    last_lookup_ = lookup;

    std::size_t start_pos = lookup.reused_tokens;
    for (std::size_t i = start_pos; i < req.prompt.ids.size(); ++i) {
      sess->advance(req.prompt.ids[i], eff_mask[i]);
      if (std::find(capture_positions.begin(), capture_positions.end(), static_cast<int>(i)) !=
          capture_positions.end()) {
        const auto logits = sess->current_logits();
        on_logits(LogitStep{static_cast<int>(i), logits});
      }
    }
    release_session(0);
  }

  const SessionLookupResult& last_lookup() const noexcept { return last_lookup_; }

 private:
  BackendCapabilities caps_;
  BackendStateIdentity id_;
  WorkerProfile profile_;
  std::unique_ptr<FakeCpuSession> session_;
  SessionLookupResult last_lookup_;
};

void test_family_enum_and_detection() {
  CHECK(std::string(to_string(ModelFamily::Inkling2)) == "inkling2");
  CHECK(std::string(to_string(ModelFamily::Qwen3_5)) == "qwen3_5");
  CHECK(std::string(to_string(ModelFamily::DeepSeekV4)) == "deepseek_v4");
  CHECK(std::string(to_string(ModelFamily::GlmFlash)) == "glm_flash");
  CHECK(std::string(to_string(ModelFamily::Unknown)) == "unknown");

  CHECK(parse_model_family("inkling-2") == ModelFamily::Inkling2);
  CHECK(parse_model_family("qwen3.8") == ModelFamily::Qwen3_5);
  CHECK(parse_model_family("qwen35") == ModelFamily::Qwen3_5);
  CHECK(parse_model_family("deepseek_v41") == ModelFamily::DeepSeekV4);
  CHECK(parse_model_family("deepseek41") == ModelFamily::DeepSeekV4);
  CHECK(parse_model_family("glm53") == ModelFamily::GlmFlash);
  CHECK(parse_model_family("glm5-next") == ModelFamily::GlmFlash);
  for (ModelFamily fam : {ModelFamily::Inkling2, ModelFamily::Qwen3_5, ModelFamily::DeepSeekV4, ModelFamily::GlmFlash}) {
    CHECK(parse_model_family(to_string(fam)) == fam);
  }
  CHECK_THROWS(parse_model_family(to_string(ModelFamily::Unknown)), std::invalid_argument);
  CHECK_THROWS(parse_model_family("deepseek_v3"), std::invalid_argument);
  CHECK_THROWS(parse_model_family("llama99"), std::invalid_argument);
  CHECK_THROWS(parse_model_family("dflash"), std::invalid_argument);

  CHECK(std::string(to_string(VerificationLevel::PinnedVerified)) == "pinned_verified");
  CHECK(std::string(to_string(VerificationLevel::SelfHashed)) == "self_hashed");
  CHECK(std::string(to_string(VerificationLevel::HeadersOnly)) == "headers_only");

  // Exact SF & GGUF packs on disk.
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
  for (const char* ds_dir : {"/home/orwel/dev_genius/models/gguf/dsv41/mrfakename-Q8_0",
                             "/home/orwel/dev_genius/models/gguf/dsv41/acrogenesis-IQ2_XXS",
                             "/mnt/storage/models/Mix-256G"}) {
    if (std::filesystem::is_directory(ds_dir)) {
      CHECK(detect_checkpoint_format(ds_dir) == CheckpointFormat::Gguf);
      CHECK(detect_model_family(ds_dir) == ModelFamily::DeepSeekV4);
    }
  }
  const std::filesystem::path ds_hf("/mnt/storage/models/DeepSeek-V4.1-Flash-1017");
  if (std::filesystem::is_directory(ds_hf)) {
    CHECK(detect_checkpoint_format(ds_hf) == CheckpointFormat::Safetensors);
    CHECK(detect_model_family(ds_hf) == ModelFamily::DeepSeekV4);
  }
  const std::filesystem::path ds_mtp("/home/orwel/dev_genius/models/gguf/dsv41/DeepSeek-V4.1-Flash-1017-MTP-Q8_0.gguf");
  if (std::filesystem::is_regular_file(ds_mtp)) {
    CHECK(detect_checkpoint_format(ds_mtp) == CheckpointFormat::Gguf);
    CHECK_THROWS(detect_model_family(ds_mtp), std::invalid_argument);
  }
  const std::filesystem::path glm_synth("/home/orwel/dev_genius/engines/atlas-v6-refs/project-maya/data/glm5-synth.gguf");
  if (std::filesystem::is_regular_file(glm_synth)) {
    CHECK(detect_checkpoint_format(glm_synth) == CheckpointFormat::Gguf);
    CHECK(detect_model_family(glm_synth) == ModelFamily::GlmFlash);
  }

  // Synthetic GGUF shard + auxiliary rejection.
  TempDir gguf_dir("gguf_fmt");
  write_synthetic_gguf(gguf_dir.path / "model-00001-of-00002.gguf", "deepseek41", true);
  write_synthetic_gguf(gguf_dir.path / "model-00002-of-00002.gguf", "", false);
  CHECK(detect_checkpoint_format(gguf_dir.path) == CheckpointFormat::Gguf);
  CHECK(detect_model_family(gguf_dir.path) == ModelFamily::DeepSeekV4);
  CHECK(detect_model_family(gguf_dir.path / "model-00001-of-00002.gguf") == ModelFamily::DeepSeekV4);
  CHECK_THROWS(detect_model_family(gguf_dir.path / "model-00002-of-00002.gguf"), std::invalid_argument);

  TempDir mixed_base_draft("mixed_draft");
  write_synthetic_gguf(mixed_base_draft.path / "01-base.gguf", "deepseek41", true);
  write_synthetic_gguf(mixed_base_draft.path / "02-draft.gguf", "dflash", true);
  CHECK_THROWS(detect_model_family(mixed_base_draft.path), std::invalid_argument);

  // Reject legacy/unsupported HF tags cleanly (e.g. DeepSeek-V3 must NOT map to DeepSeekV4, GLM-4 not to GlmFlash).
  TempDir v3_dir("dsv3");
  write_text_file(v3_dir.path / "config.json", R"({"architectures":["DeepseekV3ForCausalLM"],"model_type":"deepseek_v3"})");
  CHECK_THROWS(detect_model_family(v3_dir.path), std::invalid_argument);

  TempDir glm4_dir("glm4");
  write_text_file(glm4_dir.path / "config.json", R"({"architectures":["GlmForCausalLM"],"model_type":"glm4"})");
  CHECK_THROWS(detect_model_family(glm4_dir.path), std::invalid_argument);

  TempDir qwen2_dir("qwen2");
  write_text_file(qwen2_dir.path / "config.json", R"({"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2"})");
  CHECK_THROWS(detect_model_family(qwen2_dir.path), std::invalid_argument);

  // Valid supported HF configs.
  TempDir glm5_dir("glm5_sf");
  write_text_file(glm5_dir.path / "config.json",
                  R"({"architectures":["Glm5NextForConditionalGeneration"],"model_type":"glm5_next"})");
  CHECK(detect_model_family(glm5_dir.path) == ModelFamily::GlmFlash);

  TempDir ink_dir("ink");
  write_text_file(ink_dir.path / "tokenizer.json", R"({"added_tokens":[{"id":1,"content":"<|message_model|>"}]})");
  CHECK(detect_model_family(ink_dir.path) == ModelFamily::Inkling2);

  // Ambiguous config.json + GGUF dir must fail closed.
  TempDir cfg_gguf_conflict("cfg_gguf");
  write_text_file(cfg_gguf_conflict.path / "config.json", R"({"model_type":"qwen3_5"})");
  write_synthetic_gguf(cfg_gguf_conflict.path / "m.gguf", "deepseek41", true);
  CHECK_THROWS(detect_model_family(cfg_gguf_conflict.path), std::invalid_argument);
}

void test_identity_reusability_and_prefix_checks() {
  const BackendStateIdentity a = make_verified_sample_identity();
  CHECK(a.is_reusable_with(a));

  auto check_fail = [&](auto mutator) {
    BackendStateIdentity b = a;
    mutator(b);
    CHECK(!b.is_reusable_with(b));
    CHECK(!a.is_reusable_with(b));
  };

  check_fail([](BackendStateIdentity& b) { b.checkpoint.verification_level = VerificationLevel::HeadersOnly; });
  check_fail([](BackendStateIdentity& b) { b.checkpoint.verification_level = VerificationLevel::SelfHashed; });
  check_fail([](BackendStateIdentity& b) { b.checkpoint.shard_digests_verified = false; });
  check_fail([](BackendStateIdentity& b) { b.checkpoint.weight_manifest_verified = false; });
  check_fail([](BackendStateIdentity& b) { b.checkpoint.weight_manifest_sha256.fill(0u); });
  check_fail([](BackendStateIdentity& b) { b.checkpoint.shard_set_sha256.fill(0u); });
  check_fail([](BackendStateIdentity& b) { b.family = ModelFamily::Unknown; });
  check_fail([](BackendStateIdentity& b) { b.backend_name.clear(); });
  check_fail([](BackendStateIdentity& b) { b.checkpoint.format = CheckpointFormat::Unknown; });
  check_fail([](BackendStateIdentity& b) { b.checkpoint.model_id.clear(); });
  check_fail([](BackendStateIdentity& b) { b.checkpoint.schema_variant.clear(); });
  check_fail([](BackendStateIdentity& b) { b.checkpoint.shard_count = 0; });
  check_fail([](BackendStateIdentity& b) { b.state_layout_id.clear(); });
  check_fail([](BackendStateIdentity& b) { b.act_dtype = ActivationDtype::Unknown; });
  check_fail([](BackendStateIdentity& b) { b.numeric_abi.clear(); });
  check_fail([](BackendStateIdentity& b) { b.weight_precision.clear(); });
  check_fail([](BackendStateIdentity& b) { b.kv_format.clear(); });
  check_fail([](BackendStateIdentity& b) { b.kv_cache_mode.clear(); });
  check_fail([](BackendStateIdentity& b) { b.gear.clear(); });
  check_fail([](BackendStateIdentity& b) { b.max_ctx = 0; });
  check_fail([](BackendStateIdentity& b) { b.prefill_chunk = 0; });
  check_fail([](BackendStateIdentity& b) { b.persona_key = 7; /* empty persona_fingerprint */ });
  check_fail([](BackendStateIdentity& b) { b.persona_fingerprint = "p7"; /* zero persona_key */ });

  const std::vector<std::int32_t> cached_ids = {10, 20, 30, 40};
  const std::vector<std::int32_t> prompt_match = {10, 20, 30, 40, 50};
  const std::vector<std::int32_t> prompt_mismatch = {10, 20, 99, 40, 50};
  CHECK(!prefix_matches_exact(cached_ids, prompt_match, 0));  // 0 prefix is never a cache hit
  CHECK(prefix_matches_exact(cached_ids, prompt_match, 4));
  CHECK(!prefix_matches_exact(cached_ids, prompt_mismatch, 4));
  CHECK(!prefix_matches_exact(cached_ids, prompt_match, 5));

  const std::vector<std::uint8_t> cached_mask = {1, 1, 0, 1};
  const std::vector<std::uint8_t> prompt_mask_ok = {1, 1, 0, 1, 1};
  const std::vector<std::uint8_t> prompt_mask_bad = {1, 1, 1, 1, 1};
  CHECK(!prefix_matches_exact_with_mask(cached_ids, cached_mask, prompt_match, prompt_mask_ok, 0));
  CHECK(prefix_matches_exact_with_mask(cached_ids, cached_mask, prompt_match, prompt_mask_ok, 4));
  CHECK(!prefix_matches_exact_with_mask(cached_ids, cached_mask, prompt_match, prompt_mask_bad, 4));
  CHECK(!prefix_matches_exact_with_mask(cached_ids, cached_mask, prompt_mismatch, prompt_mask_ok, 4));
}

void test_capabilities_and_fail_closed_validation() {
  BackendCapabilities caps_zero_vocab;
  caps_zero_vocab.family = ModelFamily::DeepSeekV4;
  caps_zero_vocab.backend_name = "ds-zero-vocab";
  caps_zero_vocab.supports_images = false;
  caps_zero_vocab.media_token_ids = {129264};
  caps_zero_vocab.vocab_size = 0;

  std::vector<std::int32_t> ds_media = {5, 129264, 6};
  Prompt p;
  p.ids = ds_media;
  CHECK_THROWS(validate_request_against_capabilities(caps_zero_vocab, 1024, p, 8, Sampling{}), std::invalid_argument);
}

void test_session_lifecycle_reuse_and_registry() {
  // 1. Register FakeCpuModelBackend in BackendRegistry (rejects duplicate registration, returns stable optional).
  auto& reg = BackendRegistry::instance();
  CHECK_THROWS(reg.register_backend({ModelFamily::Unknown, "bad", nullptr}), std::invalid_argument);
  reg.register_backend(BackendRegistration{
      ModelFamily::DeepSeekV4, "fake-cpu-ds",
      [](const RuntimeOptions&) -> std::unique_ptr<ModelBackend> {
        return std::make_unique<FakeCpuModelBackend>(true);
      }});
  const auto snap_reg = reg.find(ModelFamily::DeepSeekV4);
  CHECK(snap_reg.has_value() && snap_reg->name == "fake-cpu-ds");
  CHECK_THROWS(reg.register_backend(BackendRegistration{
                   ModelFamily::DeepSeekV4, "dup",
                   [](const RuntimeOptions&) -> std::unique_ptr<ModelBackend> { return nullptr; }}),
               std::invalid_argument);

  auto backend_ptr = reg.create(ModelFamily::DeepSeekV4, RuntimeOptions{});
  auto& backend = *backend_ptr;
  auto* fake = dynamic_cast<FakeCpuModelBackend*>(backend_ptr.get());
  CHECK(fake != nullptr);

  // 2. Existing Generator& callers still work untouched.
  Generator& gen = backend;
  const std::vector<std::int32_t> prefix = {3, 17, 29, 41};
  const std::vector<std::uint8_t> prefix_mask = {1, 1, 0, 1};
  const std::vector<std::int32_t> suffix = {12, 34, 56};
  const std::vector<std::uint8_t> suffix_mask = {1, 1, 1};

  std::vector<std::int32_t> full_tokens = prefix;
  full_tokens.insert(full_tokens.end(), suffix.begin(), suffix.end());
  std::vector<std::uint8_t> full_mask = prefix_mask;
  full_mask.insert(full_mask.end(), suffix_mask.begin(), suffix_mask.end());

  // Cold uncached reference: teacher-forced final prompt logits + 5 greedy tokens.
  std::vector<float> cold_tf_logits;
  std::vector<std::int32_t> cold_greedy;
  {
    backend.reset_session();
    BackendRequest cold_req;
    cold_req.prompt.ids = full_tokens;
    cold_req.token_mask = full_mask;
    cold_req.use_prefix_cache = false;
    const std::array<int, 1> cap = {static_cast<int>(full_tokens.size() - 1)};
    backend.evaluate_teacher_forced(cold_req, cap, [&](const LogitStep& s) {
      cold_tf_logits.assign(s.logits.begin(), s.logits.end());
    });
    CHECK(cold_tf_logits.size() == FakeCpuSession::kVocab);
    for (float x : cold_tf_logits) CHECK(std::isfinite(x));

    backend.reset_session();
    const auto st = backend.generate_request(
        cold_req, 5, Sampling{}, {}, [&](std::span<const std::int32_t> toks) {
          cold_greedy.insert(cold_greedy.end(), toks.begin(), toks.end());
          return true;
        });
    CHECK(st.reuse == Reuse::None && st.reused_tokens == 0);
    CHECK(cold_greedy.size() == 5);
  }

  // Warm prefix reuse via public create_session / acquire_session + generate_request:
  // Populate session with prefix + divergent suffix, then resume with full_tokens (Checkpoint truncate + reuse).
  {
    backend.reset_session();
    std::vector<std::int32_t> other = prefix;
    other.push_back(61);
    std::vector<std::uint8_t> other_mask = prefix_mask;
    other_mask.push_back(1);
    BackendRequest warm_req;
    warm_req.prompt.ids = other;
    warm_req.token_mask = other_mask;
    warm_req.use_prefix_cache = true;
    (void)backend.generate_request(warm_req, 1, Sampling{}, {}, [](std::span<const std::int32_t>) { return true; });

    BackendRequest resume_req;
    resume_req.prompt.ids = full_tokens;
    resume_req.token_mask = full_mask;
    resume_req.use_prefix_cache = true;
    std::vector<float> resumed_tf_logits;
    const std::array<int, 1> cap = {static_cast<int>(full_tokens.size() - 1)};
    backend.evaluate_teacher_forced(resume_req, cap, [&](const LogitStep& s) {
      resumed_tf_logits.assign(s.logits.begin(), s.logits.end());
    });
    CHECK(fake->last_lookup().hit);
    CHECK(fake->last_lookup().reuse == Reuse::Checkpoint);
    CHECK(fake->last_lookup().reused_tokens == prefix.size());
    CHECK(resumed_tf_logits.size() == cold_tf_logits.size());
    CHECK(std::memcmp(resumed_tf_logits.data(), cold_tf_logits.data(),
                      cold_tf_logits.size() * sizeof(float)) == 0);

    // Also test acquire_session + truncate + resumed 5 greedy steps match byte-for-byte.
    auto* sess = backend.acquire_session(0);
    CHECK(sess != nullptr);
    sess->truncate(prefix.size());
    std::vector<std::int32_t> resumed_greedy;
    const auto st2 = backend.generate_request(
        resume_req, 5, Sampling{}, {}, [&](std::span<const std::int32_t> toks) {
          resumed_greedy.insert(resumed_greedy.end(), toks.begin(), toks.end());
          return true;
        });
    CHECK(st2.reuse == Reuse::Extend && st2.reused_tokens == static_cast<int>(prefix.size()));
    CHECK(resumed_greedy == cold_greedy);
  }

  // Flip 1 mask bit at position 2 -> must stop reuse before index 2 (reused_tokens == 2) and stay exact vs cold!
  {
    std::vector<std::uint8_t> flipped_mask = full_mask;
    flipped_mask[2] = 1;  // was 0
    BackendRequest req_flip;
    req_flip.prompt.ids = full_tokens;
    req_flip.token_mask = flipped_mask;
    req_flip.use_prefix_cache = true;
    std::vector<float> flip_reused_logits;
    const std::array<int, 1> cap = {static_cast<int>(full_tokens.size() - 1)};
    backend.evaluate_teacher_forced(req_flip, cap, [&](const LogitStep& s) {
      flip_reused_logits.assign(s.logits.begin(), s.logits.end());
    });
    CHECK(fake->last_lookup().hit);
    CHECK(fake->last_lookup().reused_tokens == 2);

    // Cold run with flipped_mask must produce identical finite full-vocab bytes.
    backend.reset_session();
    req_flip.use_prefix_cache = false;
    std::vector<float> flip_cold_logits;
    backend.evaluate_teacher_forced(req_flip, cap, [&](const LogitStep& s) {
      flip_cold_logits.assign(s.logits.begin(), s.logits.end());
    });
    CHECK(std::memcmp(flip_reused_logits.data(), flip_cold_logits.data(),
                      flip_cold_logits.size() * sizeof(float)) == 0);
  }

  // Media-sensitive lookup: identical image hits; 1-byte mutated RGB image misses cleanly.
  {
    backend.reset_session();
    auto img_rgb = std::make_shared<std::vector<std::uint8_t>>(40 * 40 * 3, 0x2a);
    PromptImage img{1, 40, 40, img_rgb};
    std::vector<std::int32_t> media_toks = {10, -1, 11, 12};
    std::vector<std::uint8_t> media_mask = {1, 0, 1, 1};
    std::vector<PromptImage> imgs = {img};

    BackendRequest r1;
    r1.prompt.ids = media_toks;
    r1.prompt.images = imgs;
    r1.token_mask = media_mask;
    r1.use_prefix_cache = true;
    const auto s1 = backend.generate_request(r1, 2, Sampling{}, {}, [](std::span<const std::int32_t>) { return true; });
    CHECK(s1.reuse == Reuse::None);

    // Same image -> reuse hits!
    std::vector<std::int32_t> media_toks2 = {10, -1, 11, 12, 13};
    std::vector<std::uint8_t> media_mask2 = {1, 0, 1, 1, 1};
    BackendRequest r2 = r1;
    r2.prompt.ids = media_toks2;
    r2.token_mask = media_mask2;
    const auto s2 = backend.generate_request(r2, 2, Sampling{}, {}, [](std::span<const std::int32_t>) { return true; });
    CHECK(s2.reuse == Reuse::Checkpoint && s2.reused_tokens == 4);

    // 1-byte mutated image -> MUST miss cleanly!
    auto img_rgb_mut = std::make_shared<std::vector<std::uint8_t>>(*img_rgb);
    (*img_rgb_mut)[77] ^= 1u;
    std::vector<PromptImage> imgs_mut = {PromptImage{1, 40, 40, img_rgb_mut}};
    BackendRequest r3 = r2;
    r3.prompt.images = imgs_mut;
    const auto s3 = backend.generate_request(r3, 2, Sampling{}, {}, [](std::span<const std::int32_t>) { return true; });
    CHECK(s3.reuse == Reuse::None && s3.reused_tokens == 0);
  }

  // Standalone create_session + Generator::generate(ids, ...) backwards-compatibility check.
  {
    auto owned_sess = backend.create_session();
    CHECK(owned_sess != nullptr && owned_sess->committed_tokens().empty());
    std::vector<std::int32_t> gen_ids;
    const auto gst = gen.generate(std::span<const std::int32_t>(prefix), 2, Sampling{}, {},
                                  [&](std::span<const std::int32_t> t) {
                                    gen_ids.insert(gen_ids.end(), t.begin(), t.end());
                                    return true;
                                  });
    CHECK(gst.generated_tokens == 2 && gen_ids.size() == 2);
  }

  // Base ModelBackend::generate_request() fallback: delegates plain requests and fails closed on
  // non-empty token_mask, use_prefix_cache=false, or non-empty cache_namespace.
  {
    class FallbackOnlyBackend final : public ModelBackend {
     public:
      FallbackOnlyBackend() {
        caps_.family = ModelFamily::Qwen3_5;
        caps_.backend_name = "fallback-only";
        caps_.architecture = "Qwen3_5ForConditionalGeneration";
        caps_.supports_text = true;
        caps_.vocab_size = 64;
        caps_.max_position_embeddings = 128;

        id_ = make_verified_sample_identity();
        id_.max_ctx = 128;
        profile_.family = ModelFamily::Qwen3_5;
        profile_.model_id = "fallback-only";
        profile_.gear = "dense";
        profile_.kv_cache_mode = "dense_bf16";
        profile_.max_ctx = 128;
        profile_.max_batch = 1;
        profile_.capabilities = caps_;
        profile_.state_identity = id_;
      }

      std::string model_id() const override { return "fallback-only"; }
      int max_model_len() const override { return 128; }
      bool supports_images() const override { return false; }

      const BackendCapabilities& capabilities() const override { return caps_; }
      const WorkerProfile& worker_profile() const override { return profile_; }
      const BackendStateIdentity& state_identity() const override { return id_; }
      void reset_session() override {}

      GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                             std::span<const std::int32_t> stop_ids,
                             const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
        Prompt p;
        p.ids = prompt;
        return generate(p, max_new_tokens, sampling, stop_ids, on_tokens, nullptr);
      }

      GenerateStats generate(const Prompt& prompt, int /*max_new_tokens*/, const Sampling& /*sampling*/,
                             std::span<const std::int32_t> /*stop_ids*/,
                             const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                             const PrefillCallback& on_prefill) override {
        GenerateStats st;
        st.prompt_tokens = static_cast<int>(prompt.ids.size());
        if (on_prefill) on_prefill(st);
        const std::int32_t tok = 7;
        st.generated_tokens = 1;
        (void)on_tokens(std::span<const std::int32_t>(&tok, 1));
        return st;
      }

     private:
      BackendCapabilities caps_;
      BackendStateIdentity id_;
      WorkerProfile profile_;
    };

    FallbackOnlyBackend fb;
    const std::array<std::int32_t, 2> fb_ids = {1, 2};
    BackendRequest ok_req;
    ok_req.prompt.ids = fb_ids;
    ok_req.use_prefix_cache = true;
    std::vector<std::int32_t> out_toks;
    const auto fb_st = fb.generate_request(
        ok_req, 1, Sampling{}, {}, [&](std::span<const std::int32_t> toks) {
          out_toks.insert(out_toks.end(), toks.begin(), toks.end());
          return true;
        });
    CHECK(fb_st.generated_tokens == 1 && out_toks == std::vector<std::int32_t>({7}));

    const std::array<std::uint8_t, 1> short_mask = {1};
    BackendRequest bad_len = ok_req;
    bad_len.token_mask = short_mask;
    CHECK_THROWS(
        fb.generate_request(bad_len, 1, Sampling{}, {}, [](std::span<const std::int32_t>) { return true; }),
        std::invalid_argument);

    const std::array<std::uint8_t, 2> valid_mask = {1, 0};
    BackendRequest with_mask = ok_req;
    with_mask.token_mask = valid_mask;
    CHECK_THROWS(
        fb.generate_request(with_mask, 1, Sampling{}, {}, [](std::span<const std::int32_t>) { return true; }),
        std::logic_error);

    BackendRequest cold_req = ok_req;
    cold_req.use_prefix_cache = false;
    CHECK_THROWS(
        fb.generate_request(cold_req, 1, Sampling{}, {}, [](std::span<const std::int32_t>) { return true; }),
        std::logic_error);

    BackendRequest ns_req = ok_req;
    ns_req.cache_namespace = "isolated-ns";
    CHECK_THROWS(
        fb.generate_request(ns_req, 1, Sampling{}, {}, [](std::span<const std::int32_t>) { return true; }),
        std::logic_error);
  }
}

}  // namespace

int main() {
  test_family_enum_and_detection();
  test_identity_reusability_and_prefix_checks();
  test_capabilities_and_fail_closed_validation();
  test_session_lifecycle_reuse_and_registry();
  std::printf("backend_model_contract: PASS\n");
  return 0;
}

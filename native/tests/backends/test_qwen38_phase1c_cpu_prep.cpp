#include "backends/qwen38/qwen38_gate2_metrics.h"
#include "backends/qwen38/qwen38_gate3_metrics.h"
#include "ingine/model_backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ingine::backends::qwen38;
using namespace ingine::model;

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: CHECK FAILED: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

BackendStateIdentity make_verified_qwen_id(std::string ns = {}) {
  BackendStateIdentity id;
  id.family = ModelFamily::Qwen3_5;
  id.backend_name = "qwen38-dense";
  id.checkpoint.format = CheckpointFormat::Safetensors;
  id.checkpoint.verification_level = VerificationLevel::PinnedVerified;
  id.checkpoint.schema_variant = "qwen3_5_text";
  id.checkpoint.model_id = "Qwen3.8-27B-NVFP4";
  id.checkpoint.revision = "482ca0f";
  id.checkpoint.shard_count = 3;
  id.checkpoint.shard_set_sha256.fill(0x5a);
  id.checkpoint.config_sha256.fill(0x11);
  id.checkpoint.tokenizer_sha256.fill(0x22);
  id.checkpoint.chat_template_sha256.fill(0x33);
  id.checkpoint.weight_manifest_sha256.fill(0x7c);
  id.checkpoint.weight_manifest_verified = true;
  id.checkpoint.shard_digests_verified = true;
  id.state_layout_id = "qwen3_5_gdn_kv_v1";
  id.act_dtype = ActivationDtype::Fp32;
  id.numeric_abi = "act=f32;attn=fp8_e4m3_cal;mlp=nvfp4_w4a32;gdn=f32;kv=bf16";
  id.weight_precision = "mixed_fp8_nvfp4";
  id.kv_format = "gdn_f32_conv_f32_kv_bf16";
  id.kv_cache_mode = "dense_bf16";
  id.gear = "dense";
  id.max_ctx = 2048;
  id.prefill_chunk = 256;
  id.cache_namespace = std::move(ns);
  id.family_state_fingerprint = "gdn48_full16_conv4_rope64_v1";
  return id;
}

// Deterministic CPU emulator exercising real Qwen38CheckpointIndex + restore_session_prefix().
class CpuEmulatedQwenSession final : public BackendSession {
 public:
  static constexpr int kActiveVocab = 64;

  explicit CpuEmulatedQwenSession(BackendStateIdentity default_id, std::size_t max_ckpts = 4)
      : default_id_(std::move(default_id)), index_(max_ckpts) {
    reset();
  }

  void reset() override {
    index_.reset(default_id_);
    checkpoints_.clear();
    acc0_ = 0x9e3779b97f4a7c15ull;
    acc1_ = 0xbf58476d1ce4e5b9ull;
    tip_logits_.clear();
  }

  std::size_t state_bytes() const noexcept override { return index_.committed_tokens().size() * 16ull; }
  std::span<const std::int32_t> committed_tokens() const noexcept override { return index_.committed_tokens(); }
  std::span<const std::uint8_t> committed_token_mask() const noexcept override { return index_.committed_token_mask(); }
  const BackendStateIdentity& identity() const noexcept override { return index_.identity(); }
  std::size_t restorable_prefix_len(std::size_t max_prefix_len) const noexcept override {
    return index_.restorable_prefix_len(max_prefix_len);
  }

  void truncate(std::size_t new_len) override {
    const std::size_t old_len = index_.committed_tokens().size();
    if (new_len == 0) {
      reset();
      return;
    }
    if (new_len == old_len) return;
    for (const auto& ck : checkpoints_) {
      if (ck.len == new_len) {
        acc0_ = ck.acc0;
        acc1_ = ck.acc1;
        tip_logits_ = ck.logits;
        index_.truncate_committed(new_len);
        (void)index_.prune_above(new_len);
        if (!tip_logits_.empty()) index_.set_tip_logits(static_cast<int>(new_len) - 1);
        return;
      }
    }
    throw std::logic_error("no saved checkpoint");
  }

  void set_identity(BackendStateIdentity id) { index_.set_identity(std::move(id)); }

  void forward(std::span<const std::int32_t> toks, std::span<const std::uint8_t> mask) {
    for (std::size_t i = 0; i < toks.size(); ++i) {
      const std::uint8_t m = mask.empty() ? 1u : mask[i];
      acc0_ ^= static_cast<std::uint64_t>(toks[i]) * 0x9e3779b97f4a7c15ull ^ (static_cast<std::uint64_t>(m) << 56);
      acc0_ = (acc0_ ^ (acc0_ >> 30)) * 0xbf58476d1ce4e5b9ull;
      acc1_ += acc0_ ^ (static_cast<std::uint64_t>(index_.committed_tokens().size() + i) * 0x94d049bb133111ebull);
    }
    index_.append_committed(toks, mask);
    tip_logits_ = compute_logits();
    index_.set_tip_logits(static_cast<int>(index_.committed_tokens().size()) - 1);
  }

  void save_ckpt() {
    const std::size_t len = index_.committed_tokens().size();
    if (const auto ev = index_.record_checkpoint(len); ev.has_value()) {
      checkpoints_.erase(std::remove_if(checkpoints_.begin(), checkpoints_.end(),
                                        [&](const Ckpt& c) { return c.len == *ev; }),
                         checkpoints_.end());
    }
    checkpoints_.push_back(Ckpt{len, acc0_, acc1_, tip_logits_});
  }

  void evict_ckpts() {
    (void)index_.evict_all_checkpoints();
    checkpoints_.clear();
  }

  std::vector<float> compute_logits() const {
    // Full pinned vocab width (248320) with active deterministic distribution in [0..kActiveVocab).
    std::vector<float> out(kGate3PinnedVocabSize, -10.0f);
    for (int v = 0; v < kActiveVocab; ++v) {
      const double phase = static_cast<double>((acc0_ ^ (static_cast<std::uint64_t>(v) * 0x517cc1b727220a95ull)) &
                                               0xffffu) /
                           65536.0;
      const double freq = static_cast<double>((acc1_ + static_cast<std::uint64_t>(v) * 17u) % 97u) * 0.05;
      out[static_cast<std::size_t>(v)] = static_cast<float>(std::sin(phase * 6.2831853 + freq) * 4.0);
    }
    return out;
  }

  const std::vector<float>& tip_logits() const { return tip_logits_; }
  const Qwen38CheckpointIndex& idx() const { return index_; }

 private:
  struct Ckpt {
    std::size_t len = 0;
    std::uint64_t acc0 = 0;
    std::uint64_t acc1 = 0;
    std::vector<float> logits;
  };
  BackendStateIdentity default_id_;
  Qwen38CheckpointIndex index_;
  std::vector<Ckpt> checkpoints_;
  std::uint64_t acc0_ = 0;
  std::uint64_t acc1_ = 0;
  std::vector<float> tip_logits_;
};

void test_compare_logits_bit_exact() {
  std::vector<float> a = {1.0f, -2.5f, 3.25f, 0.0f};
  std::vector<float> b = a;
  const auto eq = compare_logits_bit_exact(a, b, 5, 5);
  CHECK(eq.all_finite && eq.memcmp_equal && eq.argmax_match && eq.cold_argmax == 2 && eq.warm_argmax == 2 &&
        eq.max_abs_diff == 0.0f && eq.position == 5 && eq.cold_position == 5 && eq.warm_position == 5 &&
        eq.vocab_size == 4 && eq.cold_vocab_size == 4 && eq.warm_vocab_size == 4);

  // 1-ULP difference must fail memcmp even when argmax matches.
  b[0] = std::nextafter(a[0], 2.0f);
  const auto ulp = compare_logits_bit_exact(a, b, 5, 5);
  CHECK(ulp.all_finite && !ulp.memcmp_equal && ulp.argmax_match && ulp.first_diff_vocab_idx == 0 &&
        ulp.max_abs_diff > 0.0f);

  // Signed -0.0f vs +0.0f must fail memcmp cleanly.
  b = a;
  b[3] = -0.0f;
  const auto sz = compare_logits_bit_exact(a, b, 0, 0);
  CHECK(sz.all_finite && !sz.memcmp_equal && sz.max_abs_diff == 0.0f && sz.first_diff_vocab_idx == 3);

  // Identical NaN bit patterns in both must still fail closed.
  a[1] = std::numeric_limits<float>::quiet_NaN();
  b = a;
  const auto nan_cmp = compare_logits_bit_exact(a, b, 0, 0);
  CHECK(!nan_cmp.all_finite && !nan_cmp.memcmp_equal && !nan_cmp.argmax_match);

  // Identical +Inf must fail closed.
  a[1] = std::numeric_limits<float>::infinity();
  b = a;
  const auto inf_cmp = compare_logits_bit_exact(a, b, 0, 0);
  CHECK(!inf_cmp.all_finite && !inf_cmp.memcmp_equal);

  // Empty and size mismatch fail closed.
  CHECK(!compare_logits_bit_exact({}, {}, 0, 0).all_finite);
  CHECK(!compare_logits_bit_exact(a, std::span<const float>(b.data(), 3), 0, 0).memcmp_equal);
}

void test_checkpoint_index() {
  const auto id = make_verified_qwen_id();
  Qwen38CheckpointIndex idx(4);
  CHECK(idx.restorable_prefix_len(5) == 0);

  const std::vector<std::int32_t> toks(45, 7);
  idx.reset(id);
  idx.append_committed(std::span<const std::int32_t>(toks.data(), 10), {});
  CHECK(!idx.record_checkpoint(10).has_value());
  idx.append_committed(std::span<const std::int32_t>(toks.data() + 10, 10), {});
  CHECK(!idx.record_checkpoint(20).has_value());
  idx.append_committed(std::span<const std::int32_t>(toks.data() + 20, 12), {});
  CHECK(!idx.record_checkpoint(32).has_value());
  idx.append_committed(std::span<const std::int32_t>(toks.data() + 32, 4), {});
  CHECK(!idx.record_checkpoint(36).has_value());  // 4 saved: {10, 20, 32, 36}
  idx.append_committed(std::span<const std::int32_t>(toks.data() + 36, 9), {});
  const auto evicted = idx.record_checkpoint(45);
  CHECK(evicted.has_value() && *evicted == 10);
  CHECK(idx.saved_checkpoint_lengths() == std::vector<std::size_t>({20, 32, 36, 45}));

  CHECK(idx.restorable_prefix_len(45) == 45);
  CHECK(idx.restorable_prefix_len(100) == 45);
  CHECK(idx.restorable_prefix_len(32) == 32);
  CHECK(idx.restorable_prefix_len(26) == 20);
  CHECK(idx.restorable_prefix_len(19) == 0);
  CHECK(idx.restorable_prefix_len(0) == 0);

  const auto pruned = idx.prune_above(20);
  CHECK(pruned == std::vector<std::size_t>({32, 36, 45}));
  CHECK(idx.saved_checkpoint_lengths() == std::vector<std::size_t>({20}));

  const auto ev_all = idx.evict_all_checkpoints();
  CHECK(ev_all == std::vector<std::size_t>({20}));
  CHECK(idx.committed_tokens().size() == 45);
  CHECK(idx.restorable_prefix_len(44) == 0);
}

struct EmulatedTrace {
  SessionLookupResult lk;
  std::vector<int> positions;
  std::vector<std::vector<float>> logits;
  std::vector<std::int32_t> greedy_ids;
  double prefill_seconds = 0.0;
};

int argmax_vec(std::span<const float> lv) {
  int best = 0;
  for (int v = 1; v < static_cast<int>(lv.size()); ++v) {
    if (lv[v] > lv[best]) best = v;
  }
  return best;
}

EmulatedTrace emulate_cold(const BackendStateIdentity& id, std::span<const std::int32_t> prompt,
                           std::span<const std::uint8_t> mask, int steps) {
  CpuEmulatedQwenSession c(id);
  c.forward(prompt, mask);
  EmulatedTrace out;
  for (int s = 0; s < steps; ++s) {
    out.positions.push_back(static_cast<int>(c.committed_tokens().size()) - 1);
    const auto& lv = c.tip_logits();
    out.logits.push_back(lv);
    const int best = argmax_vec(lv);
    out.greedy_ids.push_back(best);
    if (s + 1 < steps) {
      const std::int32_t one[1] = {best};
      const std::uint8_t om[1] = {1u};
      c.forward(one, mask.empty() ? std::span<const std::uint8_t>{} : std::span<const std::uint8_t>(om, 1));
    }
  }
  return out;
}

EmulatedTrace emulate_warm(CpuEmulatedQwenSession& s, const BackendStateIdentity& req_id,
                           std::span<const std::int32_t> prompt, std::span<const std::uint8_t> mask, bool use_cache,
                           int steps) {
  EmulatedTrace out;
  out.lk = restore_session_prefix(s, SessionLookupQuery{req_id, prompt, mask, {}, use_cache});
  if (!out.lk.hit) {
    s.reset();
    s.set_identity(req_id);
    s.forward(prompt, mask);
    out.prefill_seconds = 0.01;
  } else if (out.lk.reused_tokens < prompt.size()) {
    const auto sub_m = mask.empty() ? std::span<const std::uint8_t>{} : mask.subspan(out.lk.reused_tokens);
    s.forward(prompt.subspan(out.lk.reused_tokens), sub_m);
    out.prefill_seconds = 0.005;
  } else {
    out.prefill_seconds = 0.0;
  }
  s.save_ckpt();
  for (int st = 0; st < steps; ++st) {
    out.positions.push_back(static_cast<int>(s.committed_tokens().size()) - 1);
    const auto& lv = s.tip_logits();
    out.logits.push_back(lv);
    const int best = argmax_vec(lv);
    out.greedy_ids.push_back(best);
    if (st + 1 < steps) {
      const std::int32_t one[1] = {best};
      const std::uint8_t om[1] = {1u};
      s.forward(one, mask.empty() ? std::span<const std::uint8_t>{} : std::span<const std::uint8_t>(om, 1));
    }
  }
  return out;
}

Gate3ReuseCaseResult make_reuse_case(std::string name, Reuse exp_r, std::size_t exp_toks, std::size_t prompt_len,
                                     const EmulatedTrace& cold, const EmulatedTrace& warm) {
  Gate3ReuseCaseResult r;
  r.name = std::move(name);
  r.executed = true;
  r.expected_reuse = exp_r;
  r.actual_reuse = warm.lk.reuse;
  r.expected_reused_tokens = exp_toks;
  r.actual_reused_tokens = warm.lk.reused_tokens;
  r.prompt_len = prompt_len;
  r.cold_step_count = cold.logits.size();
  r.warm_step_count = warm.logits.size();
  r.warm_prefill_seconds = warm.prefill_seconds;
  r.restored_bytes = (warm.lk.reuse == Reuse::Checkpoint) ? 158859264ull : 0ull;
  r.restore_seconds = (warm.lk.reuse == Reuse::Checkpoint) ? 0.012 : 0.0;
  r.cold_elapsed_ms = 120.0;
  r.warm_elapsed_ms = 25.0;
  r.cold_greedy_ids = cold.greedy_ids;
  r.warm_greedy_ids = warm.greedy_ids;
  const std::size_t n = std::min(cold.logits.size(), warm.logits.size());
  for (std::size_t i = 0; i < n; ++i) {
    r.step_comparisons.push_back(
        compare_logits_bit_exact(cold.logits[i], warm.logits[i], cold.positions[i], warm.positions[i]));
  }
  return r;
}

Qwen38Gate3Input run_passing_emulated_gate3_input() {
  const auto base_id = make_verified_qwen_id();
  CpuEmulatedQwenSession sess(base_id);

  std::vector<std::int32_t> p0(32);
  for (int i = 0; i < 32; ++i) p0[i] = 10 + i;
  std::vector<std::int32_t> p0_20(p0.begin(), p0.begin() + 20);
  std::vector<std::int32_t> p1 = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  std::vector<std::int32_t> fork_32 = p0;
  fork_32.insert(fork_32.end(), p1.begin(), p1.end());
  std::vector<std::int32_t> fork_20 = p0_20;
  fork_20.insert(fork_20.end(), p1.begin(), p1.end());
  std::vector<std::int32_t> fork_26(p0.begin(), p0.begin() + 26);
  fork_26.insert(fork_26.end(), p1.begin(), p1.begin() + 6);
  std::vector<std::int32_t> code_320(320);
  for (int i = 0; i < 320; ++i) code_320[i] = 5 + (i % 40);
  std::vector<std::int32_t> code_fork_608 = code_320;
  for (int i = 0; i < 288; ++i) code_fork_608.push_back(100 + (i % 30));
  std::vector<std::int32_t> code_ext_608 = code_320;
  for (int i = 0; i < 288; ++i) code_ext_608.push_back(200 + (i % 30));

  Qwen38Gate3Input in;
  in.qwen27b_pins_verified = true;
  in.short_ref_json_sha_verified = true;
  in.long_ref_json_sha_verified = true;
  in.self_check_passed = true;

  // 1. extend_suffix
  {
    sess.reset();
    (void)emulate_warm(sess, base_id, p0_20, {}, true, 1);
    const auto warm = emulate_warm(sess, base_id, p0, {}, true, 8);
    const auto cold = emulate_cold(base_id, p0, {}, 8);
    in.reuse_cases.push_back(make_reuse_case("extend_suffix", Reuse::Extend, 20, p0.size(), cold, warm));
  }
  // 2. extend_exact_tip
  {
    sess.reset();
    (void)emulate_warm(sess, base_id, p0, {}, true, 1);
    const auto warm = emulate_warm(sess, base_id, p0, {}, true, 8);
    const auto cold = emulate_cold(base_id, p0, {}, 8);
    in.reuse_cases.push_back(make_reuse_case("extend_exact_tip", Reuse::Extend, 32, p0.size(), cold, warm));
  }
  // 3. checkpoint_after_decode_fork
  {
    sess.reset();
    (void)emulate_warm(sess, base_id, p0, {}, true, 8);
    const auto warm = emulate_warm(sess, base_id, fork_32, {}, true, 8);
    const auto cold = emulate_cold(base_id, fork_32, {}, 8);
    in.reuse_cases.push_back(
        make_reuse_case("checkpoint_after_decode_fork", Reuse::Checkpoint, 32, fork_32.size(), cold, warm));
  }
  // 4. checkpoint_multiturn_rollback
  {
    sess.reset();
    (void)emulate_warm(sess, base_id, p0_20, {}, true, 1);
    (void)emulate_warm(sess, base_id, p0, {}, true, 4);
    const auto warm = emulate_warm(sess, base_id, fork_20, {}, true, 8);
    const auto cold = emulate_cold(base_id, fork_20, {}, 8);
    in.reuse_cases.push_back(
        make_reuse_case("checkpoint_multiturn_rollback", Reuse::Checkpoint, 20, fork_20.size(), cold, warm));
  }
  // 5. checkpoint_midfork_earlier_ckpt
  {
    sess.reset();
    (void)emulate_warm(sess, base_id, p0_20, {}, true, 1);
    (void)emulate_warm(sess, base_id, p0, {}, true, 2);
    const auto warm = emulate_warm(sess, base_id, fork_26, {}, true, 8);
    const auto cold = emulate_cold(base_id, fork_26, {}, 8);
    in.reuse_cases.push_back(
        make_reuse_case("checkpoint_midfork_earlier_ckpt", Reuse::Checkpoint, 20, fork_26.size(), cold, warm));
  }
  // 6. multichunk_checkpoint_fork (independently seeded, 608 total -> 8 steps 607..614)
  {
    sess.reset();
    (void)emulate_warm(sess, base_id, code_320, {}, true, 4);
    const auto warm = emulate_warm(sess, base_id, code_fork_608, {}, true, 8);
    const auto cold = emulate_cold(base_id, code_fork_608, {}, 8);
    in.reuse_cases.push_back(
        make_reuse_case("multichunk_checkpoint_fork", Reuse::Checkpoint, 320, code_fork_608.size(), cold, warm));
  }
  // 7. multichunk_extend_suffix (independently seeded, 608 total -> 8 steps 607..614)
  {
    sess.reset();
    (void)emulate_warm(sess, base_id, code_320, {}, true, 1);
    const auto warm = emulate_warm(sess, base_id, code_ext_608, {}, true, 8);
    const auto cold = emulate_cold(base_id, code_ext_608, {}, 8);
    in.reuse_cases.push_back(
        make_reuse_case("multichunk_extend_suffix", Reuse::Extend, 320, code_ext_608.size(), cold, warm));
  }
  // 8. session_reset_isolation
  {
    sess.reset();
    (void)emulate_warm(sess, base_id, fork_32, {}, true, 3);
    sess.reset();
    const auto warm = emulate_warm(sess, base_id, p0, {}, true, 8);
    const auto cold = emulate_cold(base_id, p0, {}, 8);
    in.reuse_cases.push_back(make_reuse_case("session_reset_isolation", Reuse::None, 0, p0.size(), cold, warm));
  }
  // 9. multi_session_isolation
  {
    sess.reset();
    (void)emulate_warm(sess, base_id, p0_20, {}, true, 1);
    CpuEmulatedQwenSession sess_b(base_id);
    (void)emulate_warm(sess_b, base_id, p1, {}, true, 4);
    sess_b.reset();
    const auto warm = emulate_warm(sess, base_id, p0, {}, true, 8);
    const auto cold = emulate_cold(base_id, p0, {}, 8);
    in.reuse_cases.push_back(make_reuse_case("multi_session_isolation", Reuse::Extend, 20, p0.size(), cold, warm));
  }

  // Protected misses (all 14 controls).
  for (const Gate3ProtectedMissContract& mc : kGate3ProtectedMissContracts) {
    sess.reset();
    (void)emulate_warm(sess, base_id, p0, {}, true, 1);
    BackendStateIdentity qid = base_id;
    bool use_cache = true;
    std::vector<std::int32_t> qtoks = p0;
    std::vector<std::uint8_t> qmask;
    if (mc.name == "miss_explicit_bypass") {
      use_cache = false;
    } else if (mc.name == "miss_empty_common_prefix") {
      qtoks[0] = 99;
    } else if (mc.name == "miss_cache_namespace_mismatch") {
      qid.cache_namespace = "other-ns";
    } else if (mc.name == "miss_token_mask_presence_mismatch") {
      qmask.assign(qtoks.size(), 1u);
    } else if (mc.name == "miss_token_mask_bit_flip_at_pos0") {
      std::vector<std::uint8_t> m_cached(p0_20.size(), 1u);
      sess.reset();
      (void)emulate_warm(sess, base_id, p0_20, m_cached, true, 1);
      qmask.assign(qtoks.size(), 1u);
      qmask[0] = 0u;
    } else if (mc.name == "miss_uncheckpointed_short_fork") {
      qtoks.resize(20);
      qtoks[10] = 99;
    } else if (mc.name == "miss_unverified_identity") {
      qid.checkpoint.verification_level = VerificationLevel::Unverified;
    } else if (mc.name == "miss_headers_only_identity") {
      qid.checkpoint.verification_level = VerificationLevel::HeadersOnly;
      qid.checkpoint.shard_digests_verified = false;
    } else if (mc.name == "miss_self_hashed_identity") {
      qid.checkpoint.verification_level = VerificationLevel::SelfHashed;
    } else if (mc.name == "miss_mutated_shard_set_sha256") {
      qid.checkpoint.shard_set_sha256[0] ^= 1u;
    } else if (mc.name == "miss_mutated_weight_manifest_sha256") {
      qid.checkpoint.weight_manifest_sha256[0] ^= 1u;
    } else if (mc.name == "miss_mutated_numeric_abi") {
      qid.numeric_abi += ";tamper";
    } else if (mc.name == "miss_mutated_media_sha256") {
      qid.media_sha256[0] ^= 1u;
    } else if (mc.name == "miss_evicted_checkpoint_fork") {
      (void)emulate_warm(sess, base_id, p0, {}, true, 4);
      sess.evict_ckpts();
      qtoks = fork_32;
    }
    const auto lk = restore_session_prefix(sess, SessionLookupQuery{qid, qtoks, qmask, {}, use_cache});
    const auto cold = emulate_cold(qid, qtoks, qmask, 1);
    const auto warm = emulate_cold(qid, qtoks, qmask, 1);
    const auto cmp = compare_logits_bit_exact(cold.logits[0], warm.logits[0], cold.positions[0], warm.positions[0]);

    Gate3ProtectedMissResult pm;
    pm.name = std::string(mc.name);
    pm.executed = true;
    pm.hit = lk.hit;
    pm.reuse = lk.reuse;
    pm.reused_tokens = lk.reused_tokens;
    pm.reason = lk.reason;
    pm.cold_position = cmp.cold_position;
    pm.warm_position = cmp.warm_position;
    pm.post_miss_position = cmp.position;
    pm.post_miss_vocab_size = cmp.vocab_size;
    pm.cold_argmax = cmp.cold_argmax;
    pm.warm_argmax = cmp.warm_argmax;
    pm.cold_greedy_id = cold.greedy_ids[0];
    pm.warm_greedy_id = warm.greedy_ids[0];
    pm.post_miss_bit_exact = cmp.all_finite && cmp.memcmp_equal && cmp.argmax_match;
    pm.max_abs_diff = cmp.max_abs_diff;
    in.protected_misses.push_back(std::move(pm));
  }
  return in;
}

void test_gate3_verdict_and_negative_controls() {
  const Qwen38Gate3Input base = run_passing_emulated_gate3_input();
  const auto pass_v = evaluate_gate3_verdict(base);
  CHECK(pass_v.reuse_cases_passed && pass_v.protected_misses_passed && pass_v.executed_checks_passed &&
        pass_v.qualification_passed && pass_v.failures.empty());

  // Reproduce att-cdd05f43 adversarial spoof: all named cases present, but all report None(0),
  // single scalar logit at pos 777, and 1 unrelated ID -> MUST fail closed.
  {
    auto spoof = base;
    for (auto& rc : spoof.reuse_cases) {
      rc.expected_reuse = Reuse::None;
      rc.actual_reuse = Reuse::None;
      rc.expected_reused_tokens = 0;
      rc.actual_reused_tokens = 0;
      rc.cold_step_count = 1;
      rc.warm_step_count = 1;
      rc.cold_greedy_ids = {999};
      rc.warm_greedy_ids = {999};
      rc.restored_bytes = 0;
      const std::array<float, 1> scalar = {1.25f};
      rc.step_comparisons = {compare_logits_bit_exact(scalar, scalar, 777, 777)};
    }
    const auto v = evaluate_gate3_verdict(spoof);
    CHECK(!v.reuse_cases_passed && !v.qualification_passed && !v.failures.empty());
  }

  // Reproduce att-cdd05f43 transfer spoofs: Extend with 4096 restored_bytes & Checkpoint with 0 bytes.
  {
    auto ext_nz = base;
    ext_nz.reuse_cases[0].restored_bytes = 4096;
    CHECK(!evaluate_gate3_verdict(ext_nz).qualification_passed);

    auto ck_z = base;
    ck_z.reuse_cases[2].restored_bytes = 0;
    CHECK(!evaluate_gate3_verdict(ck_z).qualification_passed);

    auto exact_pf = base;
    exact_pf.reuse_cases[1].warm_prefill_seconds = 0.01;
    CHECK(!evaluate_gate3_verdict(exact_pf).qualification_passed);
  }

  // Truncated/mismatched step counts, wrong positions, non-pinned vocab width, unrelated IDs.
  {
    auto trunc = base;
    trunc.reuse_cases[0].cold_step_count = 7;
    CHECK(!evaluate_gate3_verdict(trunc).qualification_passed);

    trunc = base;
    trunc.reuse_cases[0].warm_step_count = 7;
    trunc.reuse_cases[0].step_comparisons.pop_back();
    trunc.reuse_cases[0].cold_greedy_ids.pop_back();
    trunc.reuse_cases[0].warm_greedy_ids.pop_back();
    CHECK(!evaluate_gate3_verdict(trunc).qualification_passed);

    auto bad_pos = base;
    bad_pos.reuse_cases[0].step_comparisons[0].warm_position = 32;
    CHECK(!evaluate_gate3_verdict(bad_pos).qualification_passed);

    auto bad_vocab = base;
    bad_vocab.reuse_cases[0].step_comparisons[0].vocab_size = 248319;
    bad_vocab.reuse_cases[0].step_comparisons[0].cold_vocab_size = 248319;
    bad_vocab.reuse_cases[0].step_comparisons[0].warm_vocab_size = 248319;
    CHECK(!evaluate_gate3_verdict(bad_vocab).qualification_passed);

    auto bad_id = base;
    bad_id.reuse_cases[0].cold_greedy_ids[0] = (bad_id.reuse_cases[0].step_comparisons[0].cold_argmax + 1) % 100;
    bad_id.reuse_cases[0].warm_greedy_ids[0] = bad_id.reuse_cases[0].cold_greedy_ids[0];
    CHECK(!evaluate_gate3_verdict(bad_id).qualification_passed);
  }

  // Diagnostic mode rejects qualification_passed while preserving executed_checks_passed.
  {
    auto d = base;
    d.diagnostic_mode = true;
    const auto v = evaluate_gate3_verdict(d);
    CHECK(v.executed_checks_passed && !v.qualification_passed);
  }

  // Unverified model pins, missing JSON pins, failed self check.
  for (int k = 0; k < 4; ++k) {
    auto neg = base;
    if (k == 0) neg.qwen27b_pins_verified = false;
    if (k == 1) neg.short_ref_json_sha_verified = false;
    if (k == 2) neg.long_ref_json_sha_verified = false;
    if (k == 3) neg.self_check_passed = false;
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);
  }

  // 1-ULP drift even when argmax & greedy tokens match must FAIL Gate 3.
  {
    auto neg = base;
    neg.reuse_cases[0].step_comparisons[0].memcmp_equal = false;
    neg.reuse_cases[0].step_comparisons[0].max_abs_diff = 1e-6f;
    const auto v = evaluate_gate3_verdict(neg);
    CHECK(!v.reuse_cases_passed && !v.qualification_passed);
  }

  // Non-finite logit (NaN) fails closed.
  {
    auto neg = base;
    neg.reuse_cases[1].step_comparisons[0].all_finite = false;
    const auto v = evaluate_gate3_verdict(neg);
    CHECK(!v.reuse_cases_passed && !v.qualification_passed);
  }

  // Wrong Reuse enum / reused_tokens / prompt_len.
  {
    auto neg = base;
    neg.reuse_cases[2].actual_reuse = Reuse::Extend;
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);
    neg = base;
    neg.reuse_cases[2].actual_reused_tokens = 31;
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);
    neg = base;
    neg.reuse_cases[2].prompt_len = 32;
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);
  }

  // Missing, duplicate, or extra unexpected reuse case / protected miss.
  {
    auto neg = base;
    neg.reuse_cases.pop_back();
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);

    neg = base;
    neg.reuse_cases.push_back(neg.reuse_cases[0]);
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);

    neg = base;
    auto extra = neg.reuse_cases[0];
    extra.name = "extra_bogus_case";
    neg.reuse_cases.push_back(std::move(extra));
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);

    neg = base;
    neg.protected_misses.pop_back();
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);

    neg = base;
    auto extra_pm = neg.protected_misses[0];
    extra_pm.name = "extra_bogus_miss";
    neg.protected_misses.push_back(std::move(extra_pm));
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);
  }

  // Protected miss contract checks: hit, empty reason, wrong pos/vocab, non-zero diff, unrelated greedy ID.
  {
    auto neg = base;
    neg.protected_misses[0].hit = true;
    neg.protected_misses[0].reuse = Reuse::Extend;
    neg.protected_misses[0].reused_tokens = 12;
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);

    neg = base;
    neg.protected_misses[0].reason.clear();
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);

    neg = base;
    neg.protected_misses[0].post_miss_position = 777;
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);

    neg = base;
    neg.protected_misses[0].post_miss_vocab_size = 1;
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);

    neg = base;
    neg.protected_misses[0].max_abs_diff = 1e-7;
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);

    neg = base;
    neg.protected_misses[0].cold_greedy_id += 1;
    neg.protected_misses[0].warm_greedy_id += 1;
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);
  }

  // Non-finite timing fails closed.
  {
    auto neg = base;
    neg.reuse_cases[0].warm_elapsed_ms = std::numeric_limits<double>::quiet_NaN();
    CHECK(!evaluate_gate3_verdict(neg).qualification_passed);
  }
}

void test_on_disk_gate3_json_pins() {
  const std::filesystem::path short_json("/mnt/storage/flare/release-references/reference.json");
  const std::filesystem::path long_json("/mnt/storage/flare/release-references/long-code-reference.json");
  if (std::filesystem::is_regular_file(short_json)) {
    std::string actual;
    CHECK(verify_file_sha256(short_json, kGate3ShortRefJsonSha256, &actual));
  }
  if (std::filesystem::is_regular_file(long_json)) {
    std::string actual;
    CHECK(verify_file_sha256(long_json, kGate3LongRefJsonSha256, &actual));
  }
}

}  // namespace

int main() {
  test_compare_logits_bit_exact();
  test_checkpoint_index();
  test_gate3_verdict_and_negative_controls();
  test_on_disk_gate3_json_pins();
  std::printf("qwen38_phase1c_cpu_prep: PASS\n");
  return 0;
}

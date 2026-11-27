// Phase 0 CPU qualification tests for Qwen3.8-Flash-Next (qwen4exp, #849):
//   1. ModelFamily::Qwen4Exp detection & synthetic 2-shard GGUF header/pin verification (positive + fail-closed)
//   2. GGML row dequantizers (F32, F16, BF16, Q4_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K)
//   3. Exact PLE 64-bit XOR-multiply n-gram hash + EOS segment reset + dilated depthwise causal conv1d
//   4. QSA block-4 indexer dense equivalence & sparse top-k selection (always-visible tail + top blocks)
//   5. End-to-end 4-layer Qwen4ExpMiniModelCpu (GDN + PLE + GDN + FullAttn+QSA + Top-2 MoE + shared expert)
//   6. Real Quincy 6-shard UD-Q6_K_XL schema + streaming pread Layer 0..3 CPU forward (when INGINE_QWEN4EXP_GGUF_DIR set)
#include "backends/qwen4exp/qwen4exp_config.h"
#include "backends/qwen4exp/qwen4exp_cpu_ops.h"
#include "backends/qwen4exp/qwen4exp_elastic_safetensors.h"
#include "backends/qwen4exp/qwen4exp_gguf.h"
#include "ingine/model_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace {

using namespace ingine::backends::qwen4exp;
using ingine::model::CheckpointFormat;
using ingine::model::CheckpointPinSpec;
using ingine::model::ExpectedShardDigest;
using ingine::model::ModelFamily;
using ingine::model::VerificationLevel;
using ingine::model::detect_checkpoint_format;
using ingine::model::detect_model_family;
using ingine::model::parse_model_family;
using ingine::model::to_string;

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

#define CHECK_THROWS(expr)                                                                  \
  do {                                                                                      \
    bool threw_ = false;                                                                    \
    try {                                                                                   \
      (void)(expr);                                                                         \
    } catch (const std::exception&) {                                                       \
      threw_ = true;                                                                        \
    }                                                                                       \
    if (!threw_) {                                                                          \
      std::fprintf(stderr, "%s:%d: expected exception: %s\n", __FILE__, __LINE__, #expr);   \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const char* tag) {
    path = std::filesystem::temp_directory_path() /
           (std::string("ingine_test_q4exp_") + tag + "_" + std::to_string(::getpid()));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

struct GgufWriter {
  std::vector<std::uint8_t> buf;

  template <typename T>
  void put(T v) {
    const std::size_t off = buf.size();
    buf.resize(off + sizeof(T));
    std::memcpy(buf.data() + off, &v, sizeof(T));
  }

  void put_bytes(const void* p, std::size_t n) {
    const std::size_t off = buf.size();
    buf.resize(off + n);
    if (n > 0) std::memcpy(buf.data() + off, p, n);
  }

  void put_str(std::string_view s) {
    put<std::uint64_t>(s.size());
    put_bytes(s.data(), s.size());
  }

  void kv_str(std::string_view k, std::string_view v) {
    put_str(k);
    put<std::uint32_t>(8);
    put_str(v);
  }
  void kv_u32(std::string_view k, std::uint32_t v) {
    put_str(k);
    put<std::uint32_t>(4);
    put<std::uint32_t>(v);
  }
  void kv_f32(std::string_view k, float v) {
    put_str(k);
    put<std::uint32_t>(6);
    put<float>(v);
  }
  void kv_u32_arr(std::string_view k, const std::vector<std::uint32_t>& a) {
    put_str(k);
    put<std::uint32_t>(9);
    put<std::uint32_t>(4);
    put<std::uint64_t>(a.size());
    for (std::uint32_t x : a) put<std::uint32_t>(x);
  }
  void kv_u64_arr(std::string_view k, const std::vector<std::uint64_t>& a) {
    put_str(k);
    put<std::uint32_t>(9);
    put<std::uint32_t>(10);
    put<std::uint64_t>(a.size());
    for (std::uint64_t x : a) put<std::uint64_t>(x);
  }
  void align32() {
    while ((buf.size() % 32) != 0) buf.push_back(0);
  }
};

std::uint16_t f32_to_fp16_simple(float x) {
  // Exact for small normal values like 1.0, 0.5, 0.25, 2.0.
  std::uint32_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
  const std::uint32_t mant = (bits >> 13) & 0x03ffu;
  if (exp <= 0) return static_cast<std::uint16_t>(sign);
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | mant);
}

void write_synthetic_qwen4exp_shards(const std::filesystem::path& dir, std::string_view arch = "qwen4exp") {
  // Tiny 2-layer qwen4exp (L0 linear, L1 full+QSA, PLE on L0) split across 2 shards.
  // H=32, hc=2 (hc_dim=64), hc_lr=8, V=16, E=4, K=2, ff_e=32, ff_s=32, ssm_k=4, state=8, ng=2, dt=4, inner=32,
  // attn heads=4, kv=2, hd=16, rot=8, idx_h=2, idx_k=16, ple: ngram=3, hpg=4 -> 8 heads * 4 = 32 == H.
  struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type_id;
    std::vector<std::uint8_t> payload;
  };
  auto make_f32 = [](std::string name, std::vector<std::uint64_t> dims, float val) {
    std::uint64_t n = 1;
    for (auto d : dims) n *= d;
    std::vector<std::uint8_t> raw(n * 4);
    for (std::uint64_t i = 0; i < n; ++i) std::memcpy(raw.data() + i * 4, &val, 4);
    return TensorSpec{std::move(name), std::move(dims), 0u, std::move(raw)};
  };
  auto make_q8_0 = [](std::string name, std::vector<std::uint64_t> dims, std::int8_t qv) {
    std::uint64_t n = 1;
    for (auto d : dims) n *= d;
    CHECK(n % 32 == 0);
    const std::uint64_t nb = n / 32;
    std::vector<std::uint8_t> raw(nb * 34);
    const std::uint16_t d = f32_to_fp16_simple(0.25f);
    for (std::uint64_t b = 0; b < nb; ++b) {
      std::uint8_t* p = raw.data() + b * 34;
      p[0] = static_cast<std::uint8_t>(d & 0xffu);
      p[1] = static_cast<std::uint8_t>(d >> 8);
      std::memset(p + 2, static_cast<std::uint8_t>(qv), 32);
    }
    return TensorSpec{std::move(name), std::move(dims), 8u, std::move(raw)};
  };

  std::vector<TensorSpec> s1, s2;
  s1.push_back(make_q8_0("token_embd.weight", {32, 16}, 2));
  s1.push_back(make_q8_0("output.weight", {32, 16}, 1));
  s1.push_back(make_f32("output_hc_norm.weight", {64}, 1.0f));
  s1.push_back(make_q8_0("output_hc_down.weight", {64, 8}, 1));
  s1.push_back(make_f32("output_hc_up.weight", {8, 64}, 0.1f));
  s1.push_back(make_f32("per_layer_token_embd.weight", {4, 240}, 0.05f));

  auto add_layer = [&](std::vector<TensorSpec>& dst, int il, bool linear, bool ple) {
    const std::string p = "blk." + std::to_string(il) + ".";
    dst.push_back(make_f32(p + "hc_attn_norm.weight", {64}, 1.0f));
    dst.push_back(make_q8_0(p + "hc_attn_down.weight", {64, 8}, 1));
    dst.push_back(make_f32(p + "hc_attn_up.weight", {8, 64}, 0.1f));
    dst.push_back(make_f32(p + "hc_attn_inject.weight", {64, 2}, 0.0f));
    dst.push_back(make_f32(p + "hc_ffn_norm.weight", {64}, 1.0f));
    dst.push_back(make_q8_0(p + "hc_ffn_down.weight", {64, 8}, 1));
    dst.push_back(make_f32(p + "hc_ffn_up.weight", {8, 64}, 0.1f));
    dst.push_back(make_f32(p + "hc_ffn_inject.weight", {64, 2}, 0.0f));
    if (linear) {
      dst.push_back(make_q8_0(p + "attn_qkv.weight", {32, 64}, 1));
      dst.push_back(make_q8_0(p + "attn_gate.weight", {32, 32}, 1));
      dst.push_back(make_f32(p + "ssm_conv1d.weight", {4, 64}, 0.25f));
      dst.push_back(make_f32(p + "ssm_dt.bias", {4}, 0.0f));
      dst.push_back(make_f32(p + "ssm_a", {4}, -0.5f));
      dst.push_back(make_f32(p + "ssm_beta.weight", {32, 4}, 0.05f));
      dst.push_back(make_f32(p + "ssm_alpha.weight", {32, 4}, 0.05f));
      dst.push_back(make_f32(p + "ssm_norm.weight", {8}, 1.0f));
      dst.push_back(make_q8_0(p + "ssm_out.weight", {32, 32}, 1));
    } else {
      dst.push_back(make_q8_0(p + "attn_q.weight", {32, 128}, 1));
      dst.push_back(make_q8_0(p + "attn_k.weight", {32, 32}, 1));
      dst.push_back(make_q8_0(p + "attn_v.weight", {32, 32}, 1));
      dst.push_back(make_q8_0(p + "attn_output.weight", {64, 32}, 1));
      dst.push_back(make_f32(p + "attn_q_norm.weight", {16}, 1.0f));
      dst.push_back(make_f32(p + "attn_k_norm.weight", {16}, 1.0f));
      dst.push_back(make_f32(p + "indexer.q_proj.weight", {32, 32}, 0.1f));
      dst.push_back(make_f32(p + "indexer.k_proj.weight", {32, 16}, 0.1f));
      dst.push_back(make_f32(p + "indexer.q_norm.weight", {16}, 1.0f));
      dst.push_back(make_f32(p + "indexer.k_norm.weight", {16}, 1.0f));
    }
    if (ple) {
      dst.push_back(make_q8_0(p + "ple_key.weight", {32, 64}, 1));
      dst.push_back(make_q8_0(p + "ple_value.weight", {32, 32}, 1));
      dst.push_back(make_f32(p + "ple_norm_key.weight", {64}, 1.0f));
      dst.push_back(make_f32(p + "ple_norm_query.weight", {64}, 1.0f));
      dst.push_back(make_f32(p + "ple_norm_conv.weight", {64}, 1.0f));
      dst.push_back(make_f32(p + "ple_conv1d.weight", {4, 64}, 0.25f));
    }
    dst.push_back(make_f32(p + "ffn_gate_inp.weight", {32, 4}, 0.1f));
    dst.push_back(make_q8_0(p + "ffn_gate_exps.weight", {32, 32, 4}, 1));
    dst.push_back(make_q8_0(p + "ffn_up_exps.weight", {32, 32, 4}, 1));
    dst.push_back(make_q8_0(p + "ffn_down_exps.weight", {32, 32, 4}, 1));
    dst.push_back(make_f32(p + "ffn_gate_inp_shexp.weight", {32}, 0.1f));
    dst.push_back(make_q8_0(p + "ffn_gate_shexp.weight", {32, 32}, 1));
    dst.push_back(make_q8_0(p + "ffn_up_shexp.weight", {32, 32}, 1));
    dst.push_back(make_q8_0(p + "ffn_down_shexp.weight", {32, 32}, 1));
  };
  add_layer(s1, 0, true, true);
  add_layer(s2, 1, false, false);

  auto emit_shard = [&](const std::filesystem::path& out_path, const std::vector<TensorSpec>& ts, bool with_kv) {
    GgufWriter w;
    w.put_bytes("GGUF", 4);
    w.put<std::uint32_t>(3);
    w.put<std::uint64_t>(ts.size());
    w.put<std::uint64_t>(with_kv ? 34 : 1);
    if (with_kv) {
      w.kv_str("general.architecture", arch);
      w.kv_str("general.name", "Qwen3.8-Flash-Next-Synth");
      w.kv_u32("qwen4exp.block_count", 2);
      w.kv_u32("qwen4exp.embedding_length", 32);
      w.kv_u32("qwen4exp.context_length", 1024);
      w.kv_f32("qwen4exp.attention.layer_norm_rms_epsilon", 1e-6f);
      w.kv_u32("qwen4exp.hyper_connection.count", 2);
      w.kv_u32("qwen4exp.hyper_connection.low_rank", 8);
      w.kv_u32("qwen4exp.full_attention_interval", 2);
      w.kv_u32("qwen4exp.attention.head_count", 4);
      w.kv_u32("qwen4exp.attention.head_count_kv", 2);
      w.kv_u32("qwen4exp.attention.key_length", 16);
      w.kv_u32("qwen4exp.attention.value_length", 16);
      w.kv_u32("qwen4exp.rope.dimension_count", 8);
      w.kv_f32("qwen4exp.rope.freq_base", 10000000.0f);
      w.kv_u32_arr("qwen4exp.rope.dimension_sections", {2, 2, 0, 0});
      w.kv_u32("qwen4exp.attention.indexer.head_count", 2);
      w.kv_u32("qwen4exp.attention.indexer.key_length", 16);
      w.kv_u32("qwen4exp.attention.indexer.top_k", 8);
      w.kv_u32_arr("qwen4exp.attention.compress_ratios", {0, 4});
      w.kv_u32("qwen4exp.ssm.conv_kernel", 4);
      w.kv_u32("qwen4exp.ssm.state_size", 8);
      w.kv_u32("qwen4exp.ssm.group_count", 2);
      w.kv_u32("qwen4exp.ssm.time_step_rank", 4);
      w.kv_u32("qwen4exp.ssm.inner_size", 32);
      w.kv_u32("qwen4exp.expert_count", 4);
      w.kv_u32("qwen4exp.expert_used_count", 2);
      w.kv_u32("qwen4exp.expert_feed_forward_length", 32);
      w.kv_u32("qwen4exp.expert_shared_feed_forward_length", 32);
      w.kv_u32_arr("qwen4exp.ple.layers", {0});
      w.kv_u32("qwen4exp.ple.ngram_size", 3);
      w.kv_u32("qwen4exp.ple.heads_per_ngram", 4);
      w.kv_u32("qwen4exp.ple.conv_kernel", 4);
      w.kv_u32("qwen4exp.ple.eos_token_id", 0);
      w.kv_u32("qwen4exp.embedding_length_per_layer_input", 4);
      w.kv_u64_arr("qwen4exp.ple.layer_multipliers", {23703573157769ull, 20109073645365ull, 8052911324071ull});
      w.kv_u64_arr("qwen4exp.ple.head_offsets", {0, 20, 40, 60, 80, 100, 120, 140});
      w.kv_u64_arr("qwen4exp.ple.head_vocab_sizes", {19, 19, 19, 19, 19, 19, 19, 19});
      // Fix KV count in header (38 keys total).
      const std::uint64_t real_kv = 38;
      std::memcpy(w.buf.data() + 16, &real_kv, 8);
    } else {
      w.kv_str("general.architecture", arch);
    }

    std::uint64_t rel_off = 0;
    for (const auto& t : ts) {
      w.put_str(t.name);
      w.put<std::uint32_t>(static_cast<std::uint32_t>(t.dims.size()));
      for (auto d : t.dims) w.put<std::uint64_t>(d);
      w.put<std::uint32_t>(t.type_id);
      w.put<std::uint64_t>(rel_off);
      rel_off = ((rel_off + t.payload.size()) + 31ull) & ~31ull;
    }
    w.align32();
    const std::size_t data_base = w.buf.size();
    rel_off = 0;
    for (const auto& t : ts) {
      w.buf.resize(data_base + rel_off, 0);
      w.put_bytes(t.payload.data(), t.payload.size());
      rel_off = ((rel_off + t.payload.size()) + 31ull) & ~31ull;
    }
    std::ofstream out(out_path, std::ios::binary);
    CHECK(out.good());
    out.write(reinterpret_cast<const char*>(w.buf.data()), static_cast<std::streamsize>(w.buf.size()));
  };

  emit_shard(dir / "q4exp-00001-of-00002.gguf", s1, true);
  emit_shard(dir / "q4exp-00002-of-00002.gguf", s2, false);
}

void test_family_and_synthetic_gguf() {
  CHECK(std::string(to_string(ModelFamily::Qwen4Exp)) == "qwen4exp");
  CHECK(parse_model_family("qwen4exp") == ModelFamily::Qwen4Exp);
  CHECK(parse_model_family("qwen3.8-flash-next") == ModelFamily::Qwen4Exp);

  TempDir td("synth");
  write_synthetic_qwen4exp_shards(td.path);
  CHECK(detect_checkpoint_format(td.path) == CheckpointFormat::Gguf);
  CHECK(detect_model_family(td.path) == ModelFamily::Qwen4Exp);

  const auto hdr = Qwen4ExpGgufSet::inspect_headers(td.path);
  CHECK(hdr.shards().size() == 2);
  CHECK(hdr.config().num_hidden_layers == 2);
  CHECK(hdr.config().num_linear_layers == 1 && hdr.config().num_full_layers == 1);
  CHECK(hdr.config().is_ple(0) && !hdr.config().is_ple(1));
  CHECK(hdr.identity().verification_level == VerificationLevel::HeadersOnly);
  CHECK(!hdr.identity().shard_digests_verified);
  CHECK(hdr.ple_table_rows() == 240);

  // Build exact pin spec and verify PinnedVerified succeeds.
  CheckpointPinSpec pin;
  for (const auto& s : hdr.shards()) {
    pin.shards.push_back(ExpectedShardDigest{s.filename, s.file_size, sha256_file(s.path)});
  }
  const auto pinned = Qwen4ExpGgufSet::inspect_pinned(td.path, pin);
  CHECK(pinned.identity().verification_level == VerificationLevel::PinnedVerified);
  CHECK(pinned.identity().shard_digests_verified);

  // Verify streaming pread forward on both synthetic GGUF layers!
  Qwen4ExpSessionCpu sess(pinned.config());
  std::vector<float> res_in(pinned.config().hc_dim(), 0.25f);
  std::vector<float> res_mid(pinned.config().hc_dim(), 0.0f);
  std::vector<float> res_out(pinned.config().hc_dim(), 0.0f);
  forward_gguf_layer_cpu(pinned, sess, 0, 5, 0, res_in, res_mid);
  forward_gguf_layer_cpu(pinned, sess, 1, 5, 0, res_mid, res_out);
  for (float v : res_out) CHECK(std::isfinite(v));

  // Mutated pin digest must fail closed!
  CheckpointPinSpec bad_pin = pin;
  bad_pin.shards[1].sha256[0] ^= 0xffu;
  CHECK_THROWS(Qwen4ExpGgufSet::inspect_pinned(td.path, bad_pin));

  // Missing shard must fail closed!
  std::filesystem::remove(td.path / "q4exp-00002-of-00002.gguf");
  CHECK_THROWS(Qwen4ExpGgufSet::inspect_headers(td.path));

  std::printf("PASS test_family_and_synthetic_gguf\n");
}

void test_ggml_dequantizers() {
  // Q8_0: d=0.5, qs[i]=i-16
  {
    std::vector<std::uint8_t> raw(34, 0);
    const std::uint16_t d = f32_to_fp16_simple(0.5f);
    raw[0] = static_cast<std::uint8_t>(d & 0xffu);
    raw[1] = static_cast<std::uint8_t>(d >> 8);
    for (int i = 0; i < 32; ++i) raw[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(i - 16));
    std::vector<float> out(32);
    dequantize_row_to_f32(8, raw, out);
    for (int i = 0; i < 32; ++i) CHECK(std::fabs(out[i] - 0.5f * float(i - 16)) < 1e-6f);
  }
  // Q6_K: ql=0, qh=0 (so q_val = -32), sc=1, d=0.25 -> every elem == -8.0
  {
    std::vector<std::uint8_t> raw(210, 0);
    for (int i = 0; i < 16; ++i) raw[192 + i] = 1;
    const std::uint16_t d = f32_to_fp16_simple(0.25f);
    raw[208] = static_cast<std::uint8_t>(d & 0xffu);
    raw[209] = static_cast<std::uint8_t>(d >> 8);
    std::vector<float> out(256);
    dequantize_row_to_f32(14, raw, out);
    for (int i = 0; i < 256; ++i) CHECK(std::fabs(out[i] - (-8.0f)) < 1e-6f);
  }
  // BF16: 1.5f == 0x3fc0
  {
    std::vector<std::uint8_t> raw = {0xc0, 0x3f, 0x00, 0xbf};
    std::vector<float> out(2);
    dequantize_row_to_f32(30, raw, out);
    CHECK(std::fabs(out[0] - 1.5f) < 1e-6f && std::fabs(out[1] - (-0.5f)) < 1e-6f);
  }
  // Q4_0: d=0.5, all nibbles=10 -> (10-8)*0.5 = 1.0
  {
    std::vector<std::uint8_t> raw(18, 0xaa);
    const std::uint16_t d = f32_to_fp16_simple(0.5f);
    raw[0] = static_cast<std::uint8_t>(d & 0xffu);
    raw[1] = static_cast<std::uint8_t>(d >> 8);
    std::vector<float> out(32);
    dequantize_row_to_f32(2, raw, out);
    for (float v : out) CHECK(std::fabs(v - 1.0f) < 1e-6f);
  }
  std::printf("PASS test_ggml_dequantizers\n");
}

void test_ple_hash_and_eos_semantics() {
  const auto m = Qwen4ExpMiniModelCpu::make_synthetic(77u);
  const auto& c = m.config();
  std::vector<std::int32_t> hist;
  std::vector<std::int32_t> r0(c.ple_n_heads()), r_after_eos(c.ple_n_heads()), r_on_eos(c.ple_n_heads());

  // Fresh sequence: token 7 sees predecessors (eos, eos).
  ple_hash_step(c, 7, hist, r0);
  // Feed token 9, then EOS (0), then token 7: token 7 sees (eos, eos) -> MUST match r0 bit-for-bit!
  std::vector<std::int32_t> tmp(c.ple_n_heads());
  ple_hash_step(c, 9, hist, tmp);
  ple_hash_step(c, 0, hist, r_on_eos);  // token's own EOS does NOT reset predecessors (9, 7)!
  ple_hash_step(c, 7, hist, r_after_eos);
  CHECK(r0 == r_after_eos);

  // Verify token's own EOS kept (9, 7) predecessors (differs from fresh EOS seeing (eos, eos)).
  std::vector<std::int32_t> fresh_hist, r_fresh_eos(c.ple_n_heads());
  ple_hash_step(c, 0, fresh_hist, r_fresh_eos);
  CHECK(r_on_eos != r_fresh_eos);
  std::printf("PASS test_ple_hash_and_eos_semantics\n");
}

void test_qsa_dense_and_sparse() {
  const int idx_h = 2, idx_d = 16, rot = 8, r = 4, top_k = 8;
  std::vector<float> q_norm(idx_d, 1.0f), k_norm(idx_d, 1.0f);
  std::vector<float> idx_cache;
  // Width = top_k + r - 1 = 11.
  // For pos = 0..10 (n_kv <= 11): exact dense [0..pos].
  for (int pos = 0; pos <= 10; ++pos) {
    std::vector<float> q(idx_h * idx_d, 0.25f), k(idx_d, 0.25f);
    const auto sel =
        qsa_select_positions_f32(q, k, q_norm, k_norm, 1e-6f, 1e7f, rot, idx_h, idx_d, r, top_k, pos, idx_cache);
    CHECK(static_cast<int>(sel.size()) == pos + 1);
    for (int i = 0; i <= pos; ++i) CHECK(sel[i] == i);
  }
  // At pos = 13 (n_kv = 14 > 11): complete blocks are 0=[0..3], 1=[4..7], 2=[8..11], tail=[12, 13].
  // Tail [12, 13] MUST always be included, and total count == 11 (2 complete blocks + 2 tail? Wait: 11 = 2*4 + 2 + 1).
  for (int pos = 11; pos <= 13; ++pos) {
    std::vector<float> q(idx_h * idx_d, 0.25f), k(idx_d, (pos >= 8 && pos < 12) ? 2.0f : -1.0f);
    const auto sel =
        qsa_select_positions_f32(q, k, q_norm, k_norm, 1e-6f, 1e7f, rot, idx_h, idx_d, r, top_k, pos, idx_cache);
    CHECK(static_cast<int>(sel.size()) == 11);
    if (pos == 13) {
      CHECK(std::find(sel.begin(), sel.end(), 12) != sel.end());
      CHECK(std::find(sel.begin(), sel.end(), 13) != sel.end());
    }
  }
  std::printf("PASS test_qsa_dense_and_sparse\n");
}

void test_mini_model_e2e() {
  const auto model = Qwen4ExpMiniModelCpu::make_synthetic(20261126u, /*top_k_budget=*/8);
  const std::vector<std::int32_t> seq = {3, 7, 11, 2, 19, 5, 13, 17, 23, 29, 31, 37, 41, 43, 47, 53};

  Qwen4ExpSessionCpu s1(model.config());
  const auto logits1 = model.forward_sequence_last_logits(s1, seq);
  CHECK(logits1.size() == static_cast<std::size_t>(model.config().vocab_size));
  for (float v : logits1) CHECK(std::isfinite(v));

  // Resetting session reproduces bit-identical logits.
  s1.reset();
  const auto logits2 = model.forward_sequence_last_logits(s1, seq);
  CHECK(logits1 == logits2);
  std::printf("PASS test_mini_model_e2e\n");
}

void test_real_quincy_ud_q6_k_xl() {
  const char* env = std::getenv("INGINE_QWEN4EXP_GGUF_DIR");
  if (!env || !*env || !std::filesystem::is_directory(env)) {
    std::printf("SKIP test_real_quincy_ud_q6_k_xl (INGINE_QWEN4EXP_GGUF_DIR unset or absent)\n");
    return;
  }
  const std::filesystem::path dir(env);
  CHECK(detect_checkpoint_format(dir) == CheckpointFormat::Gguf);
  CHECK(detect_model_family(dir) == ModelFamily::Qwen4Exp);

  const auto gguf = Qwen4ExpGgufSet::inspect_headers(dir);
  const auto& c = gguf.config();
  CHECK(gguf.shards().size() == 6);
  CHECK(gguf.tensors().size() == 1224);
  CHECK(c.num_hidden_layers == 48);
  CHECK(c.hidden_size == 2560);
  CHECK(c.vocab_size == 248320);
  CHECK(c.expert_count == 512 && c.expert_used_count == 10);
  CHECK(c.hc_mult == 4 && c.hc_low_rank == 320);
  CHECK(c.num_linear_layers == 36 && c.num_full_layers == 12);
  CHECK(c.is_ple(1) && !c.is_ple(0));
  CHECK(gguf.ple_table_rows() == 320001536ull);

  // Execute real streaming pread Layer 0 (GDN), Layer 1 (GDN+PLE), Layer 2 (GDN), Layer 3 (FullAttn+QSA)
  // on 2 tokens directly from the 169 GB UD-Q6_K_XL shards!
  Qwen4ExpSessionCpu sess(c);
  const auto& tok_t = gguf.require_tensor("token_embd.weight");
  const std::vector<std::int32_t> prompt = {9707, 1917};  // real Qwen tokens ("Hello", " world")
  std::vector<float> emb(static_cast<std::size_t>(c.hidden_size));
  std::vector<float> res_a(static_cast<std::size_t>(c.hc_dim()));
  std::vector<float> res_b(static_cast<std::size_t>(c.hc_dim()));

  for (int pos = 0; pos < static_cast<int>(prompt.size()); ++pos) {
    gguf.read_tensor_row_f32(tok_t, static_cast<std::uint64_t>(prompt[pos]), emb);
    for (int stream = 0; stream < c.hc_mult; ++stream) {
      std::copy_n(emb.data(), c.hidden_size, res_a.data() + static_cast<std::size_t>(stream) * c.hidden_size);
    }
    for (int il = 0; il < 4; ++il) {
      forward_gguf_layer_cpu(gguf, sess, il, prompt[pos], pos, res_a, res_b);
      res_a.swap(res_b);
    }
    double l2 = 0.0;
    for (float v : res_a) {
      CHECK(std::isfinite(v));
      l2 += double(v) * double(v);
    }
    CHECK(l2 > 0.0);
    std::printf("  real UD-Q6_K_XL pos=%d tok=%d L0..L3 out_l2=%.4f\n", pos, prompt[pos], std::sqrt(l2));
  }
  std::printf("PASS test_real_quincy_ud_q6_k_xl (6 shards, 1224 tensors, L0..L3 streaming forward verified)\n");
}

void write_tiny_safetensors(const std::filesystem::path& path,
                            const std::vector<std::tuple<std::string, std::string, std::vector<std::int64_t>,
                                                         std::vector<std::uint8_t>>>& entries) {
  std::string hdr = "{";
  std::uint64_t off = 0;
  std::vector<std::uint8_t> payload;
  for (std::size_t i = 0; i < entries.size(); ++i) {
    const auto& [name, dtype, shape, bytes] = entries[i];
    if (i > 0) hdr += ",";
    hdr += "\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":[";
    for (std::size_t j = 0; j < shape.size(); ++j) {
      if (j > 0) hdr += ",";
      hdr += std::to_string(shape[j]);
    }
    hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes.size()) + "]}";
    off += bytes.size();
    payload.insert(payload.end(), bytes.begin(), bytes.end());
  }
  hdr += "}";
  std::ofstream out(path, std::ios::binary);
  CHECK(out.good());
  const std::uint64_t hlen = hdr.size();
  out.write(reinterpret_cast<const char*>(&hlen), 8);
  out.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
  out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
}

void test_synthetic_elastic_safetensors() {
  TempDir td("elastic_st");
  const auto f32_bytes = [](float v) {
    std::vector<std::uint8_t> b(4);
    std::memcpy(b.data(), &v, 4);
    return b;
  };
  // E4M3FN byte 0x38 == 1.0f; E2M1 nibble 0x02 == +1.0f, 0x0a == -1.0f (packed 0xa2 -> [+1.0, -1.0]).
  std::vector<std::uint8_t> emb_codes(4 * 16, 0x22u);  // 4 rows x H=32 (16 bytes/row), all +1.0
  std::vector<std::uint8_t> emb_scales(4 * 2, 0x38u);  // scale=1.0
  write_tiny_safetensors(td.path / "model-00001-of-00001.safetensors", {
      {"token_embd.weight", "U8", {4, 16}, emb_codes},
      {"token_embd.weight_scale", "F8_E4M3", {4, 2}, emb_scales},
      {"token_embd.weight_scale_2", "F32", {}, f32_bytes(0.5f)},
  });

  // Two PLE shards, chunk_rows = 2, D = 32 (16 bytes codes + 2 bytes scales per row).
  std::vector<std::uint8_t> ple0_codes(2 * 16, 0x22u);  // +1.0 * 0.25 = +0.25
  std::vector<std::uint8_t> ple0_scales(2 * 2, 0x38u);
  write_tiny_safetensors(td.path / "ple-00001-of-00002.safetensors", {
      {"ple.chunk_0.weight", "U8", {2, 16}, ple0_codes},
      {"ple.chunk_0.weight_scale", "F8_E4M3", {2, 2}, ple0_scales},
      {"ple.chunk_0.weight_scale_2", "F32", {}, f32_bytes(0.25f)},
  });
  std::vector<std::uint8_t> ple1_codes(2 * 16, 0xa2u);  // alternating +1.0, -1.0 * 0.5 = +/-0.5
  std::vector<std::uint8_t> ple1_scales(2 * 2, 0x38u);
  write_tiny_safetensors(td.path / "ple-00002-of-00002.safetensors", {
      {"ple.chunk_1.weight", "U8", {2, 16}, ple1_codes},
      {"ple.chunk_1.weight_scale", "F8_E4M3", {2, 2}, ple1_scales},
      {"ple.chunk_1.weight_scale_2", "F32", {}, f32_bytes(0.5f)},
  });

  {
    std::ofstream cfg(td.path / "config.json");
    cfg << R"({
      "model_type": "qwen4exp",
      "architectures": ["Qwen4ExpForCausalLM"],
      "gguf_kv": {
        "general.architecture": "qwen4exp",
        "general.name": "Qwen3.8-Flash-Next-SynthElastic",
        "qwen4exp.block_count": 2,
        "qwen4exp.embedding_length": 32,
        "qwen4exp.context_length": 128,
        "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
        "qwen4exp.hyper_connection.count": 2,
        "qwen4exp.hyper_connection.low_rank": 8,
        "qwen4exp.full_attention_interval": 2,
        "qwen4exp.attention.head_count": 4,
        "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 16,
        "qwen4exp.attention.value_length": 16,
        "qwen4exp.rope.dimension_count": 8,
        "qwen4exp.rope.freq_base": 10000000.0,
        "qwen4exp.rope.dimension_sections": [2, 2, 0, 0],
        "qwen4exp.ssm.conv_kernel": 4,
        "qwen4exp.ssm.state_size": 8,
        "qwen4exp.ssm.group_count": 2,
        "qwen4exp.ssm.time_step_rank": 4,
        "qwen4exp.ssm.inner_size": 32,
        "qwen4exp.expert_count": 4,
        "qwen4exp.expert_used_count": 2,
        "qwen4exp.expert_feed_forward_length": 32,
        "qwen4exp.expert_shared_feed_forward_length": 32,
        "qwen4exp.ple.layers": [0],
        "qwen4exp.ple.ngram_size": 3,
        "qwen4exp.ple.heads_per_ngram": 4,
        "qwen4exp.ple.conv_kernel": 4,
        "qwen4exp.embedding_length_per_layer_input": 4,
        "qwen4exp.ple.eos_token_id": 0,
        "qwen4exp.ple.layer_multipliers": [11, 13, 17],
        "qwen4exp.ple.head_offsets": [0, 1, 2, 3, 4, 5, 6, 7],
        "qwen4exp.ple.head_vocab_sizes": [2, 2, 2, 2, 2, 2, 2, 2]
      }
    })";
  }
  {
    std::ofstream idx(td.path / "model.safetensors.index.json");
    idx << R"({
      "metadata": {"total_size": 1024},
      "weight_map": {
        "token_embd.weight": "model-00001-of-00001.safetensors",
        "token_embd.weight_scale": "model-00001-of-00001.safetensors",
        "token_embd.weight_scale_2": "model-00001-of-00001.safetensors",
        "ple.chunk_0.weight": "ple-00001-of-00002.safetensors",
        "ple.chunk_0.weight_scale": "ple-00001-of-00002.safetensors",
        "ple.chunk_0.weight_scale_2": "ple-00001-of-00002.safetensors",
        "ple.chunk_1.weight": "ple-00002-of-00002.safetensors",
        "ple.chunk_1.weight_scale": "ple-00002-of-00002.safetensors",
        "ple.chunk_1.weight_scale_2": "ple-00002-of-00002.safetensors"
      }
    })";
  }
  {
    std::ofstream qcfg(td.path / "hf_quant_config.json");
    qcfg << R"({
      "producer": {"name": "atlas-ingine-elastic", "version": "1.0"},
      "quantization": {
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
          "token_embd": {"quant_algo": "NVFP4", "group_size": 16},
          "blk.0.ffn_gate_exps": {"quant_algo": "ATLAS3", "group_size": 16},
          "blk.1.ffn_gate_exps": {"quant_algo": "ATLAS2", "group_size": 16}
        }
      },
      "atlas_elastic": {
        "family": "qwen4exp",
        "profile": "atlas-56g",
        "ple": {"format": "nvfp4", "num_shards": 2, "chunk_rows": 2}
      }
    })";
  }

  CHECK(detect_checkpoint_format(td.path) == CheckpointFormat::Safetensors);
  CHECK(detect_model_family(td.path) == ModelFamily::Qwen4Exp);
  CHECK(Qwen4ExpElasticSet::is_elastic_dir(td.path));

  const auto eset = Qwen4ExpElasticSet::inspect_headers(td.path);
  CHECK(eset.profile() == "atlas-56g");
  CHECK(eset.config().vocab_size == 4);
  CHECK(eset.config().is_ple(0));
  CHECK(eset.layer_kind("token_embd") == ElasticTensorKind::Nvfp4);
  CHECK(eset.layer_kind("blk.0.ffn_gate_exps") == ElasticTensorKind::Atlas3);
  CHECK(eset.layer_kind("blk.1.ffn_gate_exps") == ElasticTensorKind::Atlas2);

  std::vector<float> row(32, 0.0f);
  eset.read_embed_row_f32(2, row);
  for (float v : row) CHECK(std::fabs(v - 0.5f) < 1e-6f);

  eset.read_ple_row_f32(1, row);  // chunk 0 -> all +0.25
  for (float v : row) CHECK(std::fabs(v - 0.25f) < 1e-6f);

  eset.read_ple_row_f32(3, row);  // chunk 1 -> alternating +0.5, -0.5
  for (std::size_t i = 0; i < row.size(); ++i) {
    CHECK(std::fabs(row[i] - ((i % 2 == 0) ? 0.5f : -0.5f)) < 1e-6f);
  }

  // Out-of-bounds row / dimension reads must throw fail-closed.
  CHECK_THROWS(eset.read_embed_row_f32(4, row));
  CHECK_THROWS(eset.read_ple_row_f32(4, row));
  std::vector<float> bad_dim(16, 0.0f);
  CHECK_THROWS(eset.read_embed_row_f32(0, bad_dim));
  CHECK_THROWS(eset.read_ple_row_f32(0, bad_dim));

  // Direct defense-in-depth check on read_tensor_f32 for non-word-aligned F32 / BF16 byte_size.
  {
    SafetensorsEntry mis_f32 = eset.require_tensor("token_embd.weight_scale_2");
    mis_f32.byte_size = 6;
    CHECK_THROWS(eset.read_tensor_f32(mis_f32));
    SafetensorsEntry mis_bf16 = mis_f32;
    mis_bf16.dtype = "BF16";
    mis_bf16.byte_size = 3;
    CHECK_THROWS(eset.read_tensor_f32(mis_bf16));
  }

  // Fail-closed when atlas_elastic.family mismatches.
  {
    std::ofstream bad(td.path / "hf_quant_config.json");
    bad << R"({"atlas_elastic": {"family": "qwen3_5", "profile": "atlas-14g"}})";
  }
  CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  std::printf("PASS test_synthetic_elastic_safetensors (NVFP4 embed + multi-shard PLE + fail-closed)\n");
}

void write_raw_safetensors(const std::filesystem::path& path, std::uint64_t declared_hlen, const std::string& hdr,
                           const std::vector<std::uint8_t>& payload) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  CHECK(out.good());
  out.write(reinterpret_cast<const char*>(&declared_hlen), 8);
  if (!hdr.empty()) out.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
  if (!payload.empty()) out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
}

void test_elastic_safetensors_ingress_rejections() {
  auto init_base_dir = [](const std::filesystem::path& dir) {
    {
      std::ofstream cfg(dir / "config.json");
      cfg << R"({
        "model_type": "qwen4exp",
        "architectures": ["Qwen4ExpForCausalLM"],
        "gguf_kv": {
          "general.architecture": "qwen4exp",
          "general.name": "Qwen3.8-Flash-Next-ValTest",
          "qwen4exp.block_count": 2,
          "qwen4exp.embedding_length": 32,
          "qwen4exp.context_length": 128,
          "qwen4exp.attention.layer_norm_rms_epsilon": 1e-6,
          "qwen4exp.hyper_connection.count": 2,
          "qwen4exp.hyper_connection.low_rank": 8,
          "qwen4exp.full_attention_interval": 2,
          "qwen4exp.attention.head_count": 4,
          "qwen4exp.attention.head_count_kv": 2,
          "qwen4exp.attention.key_length": 16,
          "qwen4exp.attention.value_length": 16,
          "qwen4exp.rope.dimension_count": 8,
          "qwen4exp.rope.freq_base": 10000000.0,
          "qwen4exp.rope.dimension_sections": [2, 2, 0, 0],
          "qwen4exp.ssm.conv_kernel": 4,
          "qwen4exp.ssm.state_size": 8,
          "qwen4exp.ssm.group_count": 2,
          "qwen4exp.ssm.time_step_rank": 4,
          "qwen4exp.ssm.inner_size": 32,
          "qwen4exp.expert_count": 4,
          "qwen4exp.expert_used_count": 2,
          "qwen4exp.expert_feed_forward_length": 32,
          "qwen4exp.expert_shared_feed_forward_length": 32
        }
      })";
    }
    {
      std::ofstream qcfg(dir / "hf_quant_config.json");
      qcfg << R"({
        "producer": {"name": "atlas-ingine-elastic", "version": "1.0"},
        "quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": {}},
        "atlas_elastic": {"family": "qwen4exp", "profile": "atlas-test", "ple": {"format": "external", "num_shards": 0}}
      })";
    }
    {
      std::ofstream idx(dir / "model.safetensors.index.json");
      idx << R"({"metadata": {"total_size": 256}, "weight_map": {"token_embd.weight": "s.safetensors"}})";
    }
  };

  // 1. Truncated shard (< 8 bytes).
  {
    TempDir td("el_trunc");
    init_base_dir(td.path);
    std::ofstream f(td.path / "s.safetensors", std::ios::binary);
    f.write("1234", 4);
    f.close();
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }

  // 2. Huge unchecked header length (UINT64_MAX & >100 MiB & past EOF) and zero header length.
  for (std::uint64_t bad_hlen : {0ull, 200ull << 20, 0xFFFFFFFFFFFFFFFFull, 512ull}) {
    TempDir td("el_hlen");
    init_base_dir(td.path);
    write_raw_safetensors(td.path / "s.safetensors", bad_hlen, "{}", std::vector<std::uint8_t>(16, 0));
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }

  // 3. Shard path traversal, subdirectory, and absolute path in weight_map.
  for (const char* bad_shard : {"../escape.safetensors", "sub/s.safetensors", "/etc/passwd", "", ".."}) {
    TempDir td("el_path");
    init_base_dir(td.path);
    std::ofstream idx(td.path / "model.safetensors.index.json");
    idx << "{\"weight_map\":{\"token_embd.weight\":\"" << bad_shard << "\"}}";
    idx.close();
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }

  // 4. Symlink escaping checkpoint directory.
  {
    TempDir td("el_sym");
    init_base_dir(td.path);
    TempDir ext("el_sym_ext");
    write_tiny_safetensors(ext.path / "ext.safetensors",
                           {{"token_embd.weight", "F32", {2, 32}, std::vector<std::uint8_t>(2 * 32 * 4, 0)}});
    std::filesystem::create_symlink(ext.path / "ext.safetensors", td.path / "s.safetensors");
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }

  // 5. Malformed data_offsets: wrong arity, reversed [end, start], out-of-bounds, negative.
  for (const char* bad_offs : {"[0]", "[0,256,512]", "[256,0]", "[0,9999]", "[-4,256]"}) {
    TempDir td("el_offs");
    init_base_dir(td.path);
    const std::string hdr = std::string(R"({"token_embd.weight":{"dtype":"F32","shape":[2,32],"data_offsets":)") +
                            bad_offs + "}}";
    write_raw_safetensors(td.path / "s.safetensors", hdr.size(), hdr, std::vector<std::uint8_t>(256, 0));
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }

  // 6. Dtype/shape/byte-size mismatches including non-word-aligned F32 (6 bytes) & unknown dtype & overflow shape.
  {
    TempDir td("el_misalign_f32");
    init_base_dir(td.path);
    const std::string hdr = R"({"token_embd.weight":{"dtype":"F32","shape":[2,32],"data_offsets":[0,6]}})";
    write_raw_safetensors(td.path / "s.safetensors", hdr.size(), hdr, std::vector<std::uint8_t>(16, 0));
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }
  {
    TempDir td("el_bad_dtype");
    init_base_dir(td.path);
    const std::string hdr = R"({"token_embd.weight":{"dtype":"F99","shape":[2,32],"data_offsets":[0,256]}})";
    write_raw_safetensors(td.path / "s.safetensors", hdr.size(), hdr, std::vector<std::uint8_t>(256, 0));
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }
  {
    TempDir td("el_neg_shape");
    init_base_dir(td.path);
    const std::string hdr = R"({"token_embd.weight":{"dtype":"F32","shape":[-2,32],"data_offsets":[0,256]}})";
    write_raw_safetensors(td.path / "s.safetensors", hdr.size(), hdr, std::vector<std::uint8_t>(256, 0));
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }
  {
    TempDir td("el_ovf_shape");
    init_base_dir(td.path);
    const std::string hdr =
        R"({"token_embd.weight":{"dtype":"F32","shape":[10000000000,10000000000],"data_offsets":[0,256]}})";
    write_raw_safetensors(td.path / "s.safetensors", hdr.size(), hdr, std::vector<std::uint8_t>(256, 0));
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }

  // 7. Duplicate tensor name across two shards.
  {
    TempDir td("el_dup");
    init_base_dir(td.path);
    write_tiny_safetensors(td.path / "a.safetensors",
                           {{"token_embd.weight", "F32", {2, 32}, std::vector<std::uint8_t>(256, 0)}});
    write_tiny_safetensors(td.path / "b.safetensors",
                           {{"token_embd.weight", "F32", {2, 32}, std::vector<std::uint8_t>(256, 0)},
                            {"extra.weight", "F32", {1}, std::vector<std::uint8_t>(4, 0)}});
    std::ofstream idx(td.path / "model.safetensors.index.json");
    idx << R"({"weight_map":{"token_embd.weight":"a.safetensors","extra.weight":"b.safetensors"}})";
    idx.close();
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }

  // 8. Dangling weight_map entry missing from declared shard.
  {
    TempDir td("el_dangling");
    init_base_dir(td.path);
    write_tiny_safetensors(td.path / "s.safetensors",
                           {{"token_embd.weight", "F32", {2, 32}, std::vector<std::uint8_t>(256, 0)}});
    std::ofstream idx(td.path / "model.safetensors.index.json");
    idx << R"({"weight_map":{"token_embd.weight":"s.safetensors","ghost.weight":"s.safetensors"}})";
    idx.close();
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }

  // 9. Invalid PLE chunk_rows == 0.
  {
    TempDir td("el_ple0");
    init_base_dir(td.path);
    write_tiny_safetensors(td.path / "s.safetensors",
                           {{"token_embd.weight", "F32", {2, 32}, std::vector<std::uint8_t>(256, 0)}});
    std::ofstream qcfg(td.path / "hf_quant_config.json");
    qcfg << R"({"atlas_elastic":{"family":"qwen4exp","profile":"atlas-test","ple":{"format":"nvfp4","num_shards":1,"chunk_rows":0}}})";
    qcfg.close();
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }

  // 10. Valid F32 pack loads cleanly and returns exact float values.
  {
    TempDir td("el_valid_f32");
    init_base_dir(td.path);
    std::vector<float> vals(2 * 32);
    for (std::size_t i = 0; i < vals.size(); ++i) vals[i] = static_cast<float>(i) * 0.25f;
    std::vector<std::uint8_t> raw(vals.size() * sizeof(float));
    std::memcpy(raw.data(), vals.data(), raw.size());
    write_tiny_safetensors(td.path / "s.safetensors", {{"token_embd.weight", "F32", {2, 32}, raw}});
    const auto eset = Qwen4ExpElasticSet::inspect_headers(td.path);
    const auto loaded = eset.read_tensor_f32(eset.require_tensor("token_embd.weight"));
    CHECK(loaded.size() == vals.size());
    for (std::size_t i = 0; i < vals.size(); ++i) CHECK(loaded[i] == vals[i]);
    std::vector<float> row1(32, 0.0f);
    eset.read_embed_row_f32(1, row1);
    for (std::size_t i = 0; i < 32; ++i) CHECK(row1[i] == vals[32 + i]);
  }

  std::printf("PASS test_elastic_safetensors_ingress_rejections (all malformed + valid F32 cases verified)\n");
}

void test_real_quincy_elastic_packs() {
  for (const char* p : {
           "/mnt/storage16tb/models/Qwen3.8-Flash-Next-atlas-70g-ingine",
           "/mnt/storage16tb/models/Qwen3.8-Flash-Next-atlas-56g-ingine",
       }) {
    const std::filesystem::path dir(p);
    if (!Qwen4ExpElasticSet::is_elastic_dir(dir)) {
      std::printf("SKIP test_real_quincy_elastic_packs (%s not present)\n", p);
      return;
    }
    const auto eset = Qwen4ExpElasticSet::inspect_headers(dir);
    CHECK(eset.config().num_hidden_layers == 48);
    CHECK(eset.config().hidden_size == 2560);
    CHECK(eset.config().vocab_size == 248320);
    CHECK(eset.has_embedded_ple());
    std::vector<float> emb(static_cast<std::size_t>(eset.config().hidden_size), 0.0f);
    eset.read_embed_row_f32(760, emb);
    double l2 = 0.0;
    for (float v : emb) {
      CHECK(std::isfinite(v));
      l2 += double(v) * double(v);
    }
    CHECK(l2 > 0.0);
    std::vector<float> ple(static_cast<std::size_t>(eset.config().ple_head_dim), 0.0f);
    eset.read_ple_row_f32(0, ple);
    for (float v : ple) CHECK(std::isfinite(v));
    std::printf("  real elastic pack OK: %s (profile=%s, shards=%u, total_bytes=%llu, emb_l2=%.4f)\n", p,
                eset.profile().c_str(), eset.identity().shard_count,
                static_cast<unsigned long long>(eset.total_tensor_bytes()), std::sqrt(l2));
  }
  std::printf("PASS test_real_quincy_elastic_packs (130-shard atlas-70g & atlas-56g validated)\n");
}

}  // namespace

int main() {
  test_family_and_synthetic_gguf();
  test_synthetic_elastic_safetensors();
  test_elastic_safetensors_ingress_rejections();
  test_ggml_dequantizers();
  test_ple_hash_and_eos_semantics();
  test_qsa_dense_and_sparse();
  test_mini_model_e2e();
  test_real_quincy_ud_q6_k_xl();
  test_real_quincy_elastic_packs();
  std::printf("qwen4exp_phase0: PASS\n");
  return 0;
}

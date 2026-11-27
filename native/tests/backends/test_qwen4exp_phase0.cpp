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
#include <limits>
#include <map>
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

struct StTensor {
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::uint8_t> bytes;
};

struct SyntheticElasticPack {
  std::map<std::string, StTensor> model_tensors;
  std::vector<std::map<std::string, StTensor>> ple_chunks;
  std::map<std::string, std::string> qlayers_json;  // prefix -> raw JSON object string
  std::string ple_format = "nvfp4";
  int ple_chunk_rows = 2;
  int hidden_size = 32;
  int expert_count = 4;
  int expert_ffn_length = 32;
  int shared_expert_ffn_length = 32;
  int ple_head_dim = 16;

  static std::vector<std::uint8_t> f32_bytes(std::size_t n, float v) {
    std::vector<std::uint8_t> b(n * sizeof(float));
    for (std::size_t i = 0; i < n; ++i) std::memcpy(b.data() + i * sizeof(float), &v, sizeof(float));
    return b;
  }

  static std::vector<std::uint8_t> bf16_bytes(std::size_t n, float v) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    const std::uint16_t h = static_cast<std::uint16_t>(bits >> 16);
    std::vector<std::uint8_t> b(n * 2);
    for (std::size_t i = 0; i < n; ++i) std::memcpy(b.data() + i * 2, &h, 2);
    return b;
  }

  void add_f32(const std::string& name, std::vector<std::int64_t> shape, float val = 1.0f) {
    std::size_t n = 1;
    for (auto d : shape) n *= static_cast<std::size_t>(d);
    model_tensors[name] = StTensor{"F32", std::move(shape), f32_bytes(n, val)};
  }

  void add_bf16(const std::string& name, std::vector<std::int64_t> shape, float val = 0.25f) {
    std::size_t n = 1;
    for (auto d : shape) n *= static_cast<std::size_t>(d);
    model_tensors[name] = StTensor{"BF16", std::move(shape), bf16_bytes(n, val)};
  }

  void add_fp8(const std::string& prefix, int out_f, int in_f, float scale = 1.0f) {
    model_tensors[prefix + ".weight"] =
        StTensor{"F8_E4M3", {out_f, in_f}, std::vector<std::uint8_t>(static_cast<std::size_t>(out_f) * in_f, 0x38u)};
    model_tensors[prefix + ".weight_scale"] = StTensor{"F32", {}, f32_bytes(1, scale)};
    qlayers_json[prefix] = R"({"quant_algo":"FP8"})";
  }

  void add_packed_2d(const std::string& prefix, const std::string& algo, int out_f, int in_f,
                     std::uint8_t code_fill = 0x22u, float scale2 = 0.5f) {
    const int packed_in = (algo == "NVFP4") ? (in_f / 2) : (algo == "ATLAS3" ? (in_f * 3) / 8 : in_f / 4);
    const int scale_in = in_f / 16;
    model_tensors[prefix + ".weight"] = StTensor{
        "U8", {out_f, packed_in}, std::vector<std::uint8_t>(static_cast<std::size_t>(out_f) * packed_in, code_fill)};
    model_tensors[prefix + ".weight_scale"] = StTensor{
        "F8_E4M3", {out_f, scale_in}, std::vector<std::uint8_t>(static_cast<std::size_t>(out_f) * scale_in, 0x38u)};
    model_tensors[prefix + ".weight_scale_2"] = StTensor{"F32", {}, f32_bytes(1, scale2)};
    qlayers_json[prefix] = std::string(R"({"quant_algo":")") + algo + R"(","group_size":16})";
  }

  void add_moe_3d(const std::string& prefix, const std::string& algo, int E, int out_f, int in_f) {
    const int packed_in = (algo == "NVFP4") ? (in_f / 2) : (algo == "ATLAS3" ? (in_f * 3) / 8 : in_f / 4);
    const int scale_in = in_f / 16;
    model_tensors[prefix + ".weight"] =
        StTensor{"U8",
                 {E, out_f, packed_in},
                 std::vector<std::uint8_t>(static_cast<std::size_t>(E) * out_f * packed_in, 0x22u)};
    model_tensors[prefix + ".weight_scale"] =
        StTensor{"F8_E4M3",
                 {E, out_f, scale_in},
                 std::vector<std::uint8_t>(static_cast<std::size_t>(E) * out_f * scale_in, 0x38u)};
    model_tensors[prefix + ".weight_scale_2"] = StTensor{"F32", {E}, f32_bytes(static_cast<std::size_t>(E), 0.5f)};
    qlayers_json[prefix] =
        std::string(R"({"quant_algo":")") + algo + R"(","group_size":16,"per_expert_scale_2":true})";
  }

  static SyntheticElasticPack make_valid(bool with_ple = true, bool f32_embed = false) {
    SyntheticElasticPack pk;
    const int V = 4, H = 32, hc = 2, hc_dim = 64, hc_lr = 8;
    const int E = 4, ff_e = 32, ff_s = 32;
    const int conv_dim = 64, val_dim = 32, v_heads = 4, ssm_k = 4, ssm_state = 8;
    const int q_dim = 128, kv_dim = 32, attn_in = 64, hd = 16, idx_h = 2, idx_d = 16;

    if (f32_embed) {
      std::vector<float> vals(static_cast<std::size_t>(V) * H);
      for (std::size_t i = 0; i < vals.size(); ++i) vals[i] = static_cast<float>(i) * 0.25f;
      std::vector<std::uint8_t> raw(vals.size() * sizeof(float));
      std::memcpy(raw.data(), vals.data(), raw.size());
      pk.model_tensors["token_embd.weight"] = StTensor{"F32", {V, H}, std::move(raw)};
    } else {
      pk.add_packed_2d("token_embd", "NVFP4", V, H, 0x22u, 0.5f);
    }
    pk.add_packed_2d("output", "NVFP4", V, H, 0x22u, 0.5f);
    pk.add_f32("output_hc_norm.weight", {hc_dim}, 1.0f);
    pk.add_bf16("output_hc_down.weight", {hc_lr, hc_dim}, 0.25f);
    pk.add_f32("output_hc_up.weight", {hc_dim, hc_lr}, 0.25f);

    for (int il = 0; il < 2; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      pk.add_f32(p + "hc_attn_norm.weight", {hc_dim}, 1.0f);
      pk.add_bf16(p + "hc_attn_down.weight", {hc_lr, hc_dim});
      pk.add_f32(p + "hc_attn_up.weight", {hc_dim, hc_lr});
      pk.add_bf16(p + "hc_attn_inject.weight", {hc, hc_dim});
      pk.add_f32(p + "hc_ffn_norm.weight", {hc_dim}, 1.0f);
      pk.add_bf16(p + "hc_ffn_down.weight", {hc_lr, hc_dim});
      pk.add_f32(p + "hc_ffn_up.weight", {hc_dim, hc_lr});
      pk.add_bf16(p + "hc_ffn_inject.weight", {hc, hc_dim});

      if (il == 0) {
        if (with_ple) {
          pk.add_fp8(p + "ple_key", hc_dim, H);
          pk.add_fp8(p + "ple_value", H, H);
          pk.add_f32(p + "ple_norm_key.weight", {hc_dim});
          pk.add_f32(p + "ple_norm_query.weight", {hc_dim});
          pk.add_f32(p + "ple_norm_conv.weight", {hc_dim});
          pk.add_f32(p + "ple_conv1d.weight", {hc_dim, 4});
        }
        pk.add_fp8(p + "attn_qkv", conv_dim, H);
        pk.add_fp8(p + "attn_gate", val_dim, H);
        pk.add_bf16(p + "ssm_alpha.weight", {v_heads, H});
        pk.add_f32(p + "ssm_beta.weight", {v_heads, H});
        pk.add_fp8(p + "ssm_out", H, val_dim);
        pk.add_f32(p + "ssm_conv1d.weight", {conv_dim, ssm_k});
        pk.add_f32(p + "ssm_dt.bias", {v_heads});
        pk.add_f32(p + "ssm_a", {v_heads}, -0.5f);
        pk.add_f32(p + "ssm_norm.weight", {ssm_state});
      } else {
        pk.add_fp8(p + "attn_q", q_dim, H);
        pk.add_fp8(p + "attn_k", kv_dim, H);
        pk.add_fp8(p + "attn_v", kv_dim, H);
        pk.add_fp8(p + "attn_output", H, attn_in);
        pk.add_f32(p + "attn_q_norm.weight", {hd});
        pk.add_f32(p + "attn_k_norm.weight", {hd});
        pk.add_fp8(p + "indexer.q_proj", idx_h * idx_d, H);
        pk.add_fp8(p + "indexer.k_proj", idx_d, H);
        pk.add_f32(p + "indexer.q_norm.weight", {idx_d});
        pk.add_f32(p + "indexer.k_norm.weight", {idx_d});
      }

      pk.add_bf16(p + "ffn_gate_inp.weight", {E, H});
      pk.add_f32(p + "ffn_gate_inp_shexp.weight", {H});
      if (il == 0) {
        // Exercise NVFP4, ATLAS3, and ATLAS2 packed dense 2D weights as well as ATLAS3 routed MoE.
        pk.add_packed_2d(p + "ffn_gate_shexp", "NVFP4", ff_s, H);
        pk.add_packed_2d(p + "ffn_up_shexp", "ATLAS3", ff_s, H);
        pk.add_packed_2d(p + "ffn_down_shexp", "ATLAS2", H, ff_s);
        pk.add_moe_3d(p + "ffn_gate_exps", "ATLAS3", E, ff_e, H);
        pk.add_moe_3d(p + "ffn_up_exps", "ATLAS3", E, ff_e, H);
        pk.add_moe_3d(p + "ffn_down_exps", "ATLAS3", E, H, ff_e);
      } else {
        // Exercise FP8 shared experts + mixed NVFP4/ATLAS2/ATLAS3 routed MoE per projection.
        pk.add_fp8(p + "ffn_gate_shexp", ff_s, H);
        pk.add_fp8(p + "ffn_up_shexp", ff_s, H);
        pk.add_fp8(p + "ffn_down_shexp", H, ff_s);
        pk.add_moe_3d(p + "ffn_gate_exps", "NVFP4", E, ff_e, H);
        pk.add_moe_3d(p + "ffn_up_exps", "ATLAS2", E, ff_e, H);
        pk.add_moe_3d(p + "ffn_down_exps", "ATLAS3", E, H, ff_e);
      }
    }

    if (with_ple) {
      // Two NVFP4 PLE chunks, chunk_rows = 2, ple_head_dim = 16 (8B codes + 1B scales per row).
      pk.ple_format = "nvfp4";
      pk.ple_chunk_rows = 2;
      std::map<std::string, StTensor> ch0;
      ch0["ple.chunk_0.weight"] = StTensor{"U8", {2, 8}, std::vector<std::uint8_t>(2 * 8, 0x22u)};
      ch0["ple.chunk_0.weight_scale"] = StTensor{"F8_E4M3", {2, 1}, std::vector<std::uint8_t>(2 * 1, 0x38u)};
      ch0["ple.chunk_0.weight_scale_2"] = StTensor{"F32", {}, f32_bytes(1, 0.25f)};
      std::map<std::string, StTensor> ch1;
      ch1["ple.chunk_1.weight"] = StTensor{"U8", {2, 8}, std::vector<std::uint8_t>(2 * 8, 0xa2u)};
      ch1["ple.chunk_1.weight_scale"] = StTensor{"F8_E4M3", {2, 1}, std::vector<std::uint8_t>(2 * 1, 0x38u)};
      ch1["ple.chunk_1.weight_scale_2"] = StTensor{"F32", {}, f32_bytes(1, 0.5f)};
      pk.ple_chunks.push_back(std::move(ch0));
      pk.ple_chunks.push_back(std::move(ch1));
    } else {
      pk.ple_format = "external";
    }
    return pk;
  }

  void write_to(const std::filesystem::path& dir) const {
    using EntryTuple = std::tuple<std::string, std::string, std::vector<std::int64_t>, std::vector<std::uint8_t>>;
    std::vector<EntryTuple> m_entries;
    std::map<std::string, std::string> weight_map;
    for (const auto& [k, t] : model_tensors) {
      m_entries.emplace_back(k, t.dtype, t.shape, t.bytes);
      weight_map[k] = "model-00001-of-00001.safetensors";
    }
    write_tiny_safetensors(dir / "model-00001-of-00001.safetensors", m_entries);

    for (std::size_t i = 0; i < ple_chunks.size(); ++i) {
      const std::string fname = "ple-0000" + std::to_string(i + 1) + "-of-0000" + std::to_string(ple_chunks.size()) +
                                ".safetensors";
      std::vector<EntryTuple> p_entries;
      for (const auto& [k, t] : ple_chunks[i]) {
        p_entries.emplace_back(k, t.dtype, t.shape, t.bytes);
        weight_map[k] = fname;
      }
      write_tiny_safetensors(dir / fname, p_entries);
    }

    {
      std::ofstream cfg(dir / "config.json");
      cfg << "{\n"
          << "  \"model_type\": \"qwen4exp\",\n"
          << "  \"architectures\": [\"Qwen4ExpForCausalLM\"],\n"
          << "  \"gguf_kv\": {\n"
          << "    \"general.architecture\": \"qwen4exp\",\n"
          << "    \"general.name\": \"Qwen3.8-Flash-Next-SynthElastic\",\n"
          << "    \"qwen4exp.block_count\": 2,\n"
          << "    \"qwen4exp.embedding_length\": " << hidden_size << ",\n"
          << "    \"qwen4exp.context_length\": 128,\n"
          << "    \"qwen4exp.attention.layer_norm_rms_epsilon\": 1e-6,\n"
          << "    \"qwen4exp.hyper_connection.count\": 2,\n"
          << "    \"qwen4exp.hyper_connection.low_rank\": 8,\n"
          << "    \"qwen4exp.full_attention_interval\": 2,\n"
          << "    \"qwen4exp.attention.head_count\": 4,\n"
          << "    \"qwen4exp.attention.head_count_kv\": 2,\n"
          << "    \"qwen4exp.attention.key_length\": 16,\n"
          << "    \"qwen4exp.attention.value_length\": 16,\n"
          << "    \"qwen4exp.rope.dimension_count\": 8,\n"
          << "    \"qwen4exp.rope.freq_base\": 10000000.0,\n"
          << "    \"qwen4exp.rope.dimension_sections\": [2, 2, 0, 0],\n"
          << "    \"qwen4exp.attention.indexer.head_count\": 2,\n"
          << "    \"qwen4exp.attention.indexer.key_length\": 16,\n"
          << "    \"qwen4exp.attention.indexer.top_k\": 8,\n"
          << "    \"qwen4exp.attention.compress_ratios\": [0, 4],\n"
          << "    \"qwen4exp.ssm.conv_kernel\": 4,\n"
          << "    \"qwen4exp.ssm.state_size\": 8,\n"
          << "    \"qwen4exp.ssm.group_count\": 2,\n"
          << "    \"qwen4exp.ssm.time_step_rank\": 4,\n"
          << "    \"qwen4exp.ssm.inner_size\": 32,\n"
          << "    \"qwen4exp.expert_count\": " << expert_count << ",\n"
          << "    \"qwen4exp.expert_used_count\": 2,\n"
          << "    \"qwen4exp.expert_feed_forward_length\": " << expert_ffn_length << ",\n"
          << "    \"qwen4exp.expert_shared_feed_forward_length\": " << shared_expert_ffn_length;
      if (ple_format != "external") {
        cfg << ",\n"
            << "    \"qwen4exp.ple.layers\": [0],\n"
            << "    \"qwen4exp.ple.ngram_size\": 2,\n"
            << "    \"qwen4exp.ple.heads_per_ngram\": 2,\n"
            << "    \"qwen4exp.ple.conv_kernel\": 4,\n"
            << "    \"qwen4exp.embedding_length_per_layer_input\": " << ple_head_dim << ",\n"
            << "    \"qwen4exp.ple.eos_token_id\": 0,\n"
            << "    \"qwen4exp.ple.layer_multipliers\": [11, 13],\n"
            << "    \"qwen4exp.ple.head_offsets\": [0, 2],\n"
            << "    \"qwen4exp.ple.head_vocab_sizes\": [2, 2]";
      }
      cfg << "\n  }\n}\n";
    }
    {
      std::ofstream idx(dir / "model.safetensors.index.json");
      idx << "{\"metadata\":{\"total_size\":1024},\"weight_map\":{";
      bool first = true;
      for (const auto& [k, sf] : weight_map) {
        if (!first) idx << ",";
        first = false;
        idx << "\"" << k << "\":\"" << sf << "\"";
      }
      idx << "}}\n";
    }
    {
      std::ofstream qcfg(dir / "hf_quant_config.json");
      qcfg << "{\n"
           << "  \"producer\": {\"name\": \"atlas-ingine-elastic\", \"version\": \"1.0\"},\n"
           << "  \"quantization\": {\n"
           << "    \"quant_algo\": \"MIXED_PRECISION\",\n"
           << "    \"quantized_layers\": {";
      bool first = true;
      for (const auto& [k, vj] : qlayers_json) {
        if (!first) qcfg << ",";
        first = false;
        qcfg << "\"" << k << "\":" << vj;
      }
      qcfg << "}\n"
           << "  },\n"
           << "  \"atlas_elastic\": {\n"
           << "    \"family\": \"qwen4exp\",\n"
           << "    \"profile\": \"atlas-56g\",\n"
           << "    \"ple\": {\"format\": \"" << ple_format << "\", \"num_shards\": " << ple_chunks.size()
           << ", \"chunk_rows\": " << ple_chunk_rows << "}\n"
           << "  }\n"
           << "}\n";
    }
  }
};

void test_synthetic_elastic_safetensors() {
  TempDir td("elastic_st");
  const auto pk = SyntheticElasticPack::make_valid(/*with_ple=*/true, /*f32_embed=*/false);
  pk.write_to(td.path);

  CHECK(detect_checkpoint_format(td.path) == CheckpointFormat::Safetensors);
  CHECK(detect_model_family(td.path) == ModelFamily::Qwen4Exp);
  CHECK(Qwen4ExpElasticSet::is_elastic_dir(td.path));

  const auto eset = Qwen4ExpElasticSet::inspect_headers(td.path);
  CHECK(eset.profile() == "atlas-56g");
  CHECK(eset.config().vocab_size == 4);
  CHECK(eset.config().is_ple(0));
  CHECK(eset.layer_kind("token_embd") == ElasticTensorKind::Nvfp4);
  CHECK(eset.layer_kind("blk.0.ffn_gate_exps") == ElasticTensorKind::Atlas3);
  CHECK(eset.layer_kind("blk.1.ffn_gate_exps") == ElasticTensorKind::Nvfp4);
  CHECK(eset.layer_kind("blk.1.ffn_up_exps") == ElasticTensorKind::Atlas2);
  CHECK(eset.layer_kind("blk.1.ffn_down_exps") == ElasticTensorKind::Atlas3);

  // Verify exact validated dense & MoE layout descriptors.
  {
    const auto d_nvfp4 = eset.validate_dense_linear("blk.0.ffn_gate_shexp.weight", 32, 32);
    CHECK(d_nvfp4.kind == ElasticTensorKind::Nvfp4 && d_nvfp4.expected_weight_bytes == 32 * 16 &&
          d_nvfp4.expected_scale_bytes == 32 * 2);
    const auto d_a3 = eset.validate_dense_linear("blk.0.ffn_up_shexp.weight", 32, 32);
    CHECK(d_a3.kind == ElasticTensorKind::Atlas3 && d_a3.expected_weight_bytes == 32 * 12 &&
          d_a3.expected_scale_bytes == 32 * 2);
    const auto d_a2 = eset.validate_dense_linear("blk.0.ffn_down_shexp.weight", 32, 32);
    CHECK(d_a2.kind == ElasticTensorKind::Atlas2 && d_a2.expected_weight_bytes == 32 * 8 &&
          d_a2.expected_scale_bytes == 32 * 2);
    const auto d_bf16 = eset.validate_dense_linear("blk.0.ffn_gate_inp.weight", 4, 32);
    CHECK(d_bf16.kind == ElasticTensorKind::Bf16 && d_bf16.expected_weight_bytes == 4 * 32 * 2);
    const auto d_f32 = eset.validate_dense_linear("blk.0.ssm_beta.weight", 4, 32);
    CHECK(d_f32.kind == ElasticTensorKind::F32 && d_f32.expected_weight_bytes == 4 * 32 * 4);
    const auto d_fp8 = eset.validate_dense_linear("blk.0.attn_qkv.weight", 64, 32);
    CHECK(d_fp8.kind == ElasticTensorKind::Fp8 && d_fp8.expected_weight_bytes == 64 * 32);

    const auto m_a3 = eset.validate_moe_experts("blk.0.ffn_gate_exps", 4, 32, 32);
    CHECK(m_a3.kind == ElasticTensorKind::Atlas3 && m_a3.per_expert_code_bytes == 32 * 12 &&
          m_a3.per_expert_scale_bytes == 32 * 2);
    const auto m_a2 = eset.validate_moe_experts("blk.1.ffn_up_exps", 4, 32, 32);
    CHECK(m_a2.kind == ElasticTensorKind::Atlas2 && m_a2.per_expert_code_bytes == 32 * 8 &&
          m_a2.per_expert_scale_bytes == 32 * 2);

    // Caller dimension mismatches on an otherwise valid pack must throw fail-closed!
    CHECK_THROWS(eset.validate_dense_linear("blk.0.ffn_gate_shexp.weight", 16, 32));
    CHECK_THROWS(eset.validate_dense_linear("blk.0.ffn_gate_shexp.weight", 32, 64));
    CHECK_THROWS(eset.validate_dense_linear("blk.0.attn_qkv.weight", 32, 32));
    CHECK_THROWS(eset.validate_dense_linear("blk.0.ffn_gate_inp.weight", 8, 32));
    CHECK_THROWS(eset.validate_moe_experts("blk.0.ffn_gate_exps", 8, 32, 32));
    CHECK_THROWS(eset.validate_moe_experts("blk.0.ffn_gate_exps", 4, 16, 32));
    CHECK_THROWS(eset.validate_moe_experts("blk.0.ffn_down_exps", 4, 32, 16));
  }

  std::vector<float> emb_row(32, 0.0f);
  eset.read_embed_row_f32(2, emb_row);
  for (float v : emb_row) CHECK(std::fabs(v - 0.5f) < 1e-6f);

  std::vector<float> ple_row(16, 0.0f);
  eset.read_ple_row_f32(1, ple_row);  // chunk 0 -> all +0.25
  for (float v : ple_row) CHECK(std::fabs(v - 0.25f) < 1e-6f);

  eset.read_ple_row_f32(3, ple_row);  // chunk 1 -> alternating +0.5, -0.5
  for (std::size_t i = 0; i < ple_row.size(); ++i) {
    CHECK(std::fabs(ple_row[i] - ((i % 2 == 0) ? 0.5f : -0.5f)) < 1e-6f);
  }

  // Out-of-bounds row / dimension reads must throw fail-closed.
  CHECK_THROWS(eset.read_embed_row_f32(4, emb_row));
  CHECK_THROWS(eset.read_ple_row_f32(4, ple_row));
  std::vector<float> bad_dim(8, 0.0f);
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
  std::printf("PASS test_synthetic_elastic_safetensors (full 2-layer Elastic + multi-shard PLE + fail-closed)\n");
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

  // 10. Valid full F32-embedding pack loads cleanly and returns exact float values.
  {
    TempDir td("el_valid_f32");
    const auto pk = SyntheticElasticPack::make_valid(/*with_ple=*/false, /*f32_embed=*/true);
    pk.write_to(td.path);
    const auto eset = Qwen4ExpElasticSet::inspect_headers(td.path);
    const auto loaded = eset.read_tensor_f32(eset.require_tensor("token_embd.weight"));
    CHECK(loaded.size() == 4 * 32);
    for (std::size_t i = 0; i < loaded.size(); ++i) CHECK(loaded[i] == static_cast<float>(i) * 0.25f);
    std::vector<float> row1(32, 0.0f);
    eset.read_embed_row_f32(1, row1);
    for (std::size_t i = 0; i < 32; ++i) CHECK(row1[i] == static_cast<float>(32 + i) * 0.25f);
  }

  std::printf("PASS test_elastic_safetensors_ingress_rejections (all malformed + valid F32 cases verified)\n");
}

void test_elastic_consumer_layout_rejections() {
  auto expect_rejected = [](const char* tag, auto&& mutator) {
    TempDir td(tag);
    auto pk = SyntheticElasticPack::make_valid(/*with_ple=*/true, /*f32_embed=*/false);
    mutator(pk);
    pk.write_to(td.path);
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  };

  // --- A. Dense packed upload rejections (NVFP4 / ATLAS3 / ATLAS2) ---
  // 1. Self-consistent packed dense NVFP4 weight + scale shrunk below config [32, 32].
  expect_rejected("el_dense_nvfp4_short_both", [](SyntheticElasticPack& pk) {
    pk.add_packed_2d("blk.0.ffn_gate_shexp", "NVFP4", 16, 32);
  });
  // 2. Packed dense NVFP4 weight bytes too short while weight_scale keeps full config shape [32, 2].
  expect_rejected("el_dense_nvfp4_short_w", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.0.ffn_gate_shexp.weight"] = StTensor{"U8", {32, 8}, std::vector<std::uint8_t>(32 * 8, 0)};
  });
  // 3. Packed dense NVFP4 weight_scale too short while weight keeps full config shape [32, 16].
  expect_rejected("el_dense_nvfp4_short_s", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.0.ffn_gate_shexp.weight_scale"] =
        StTensor{"F8_E4M3", {32, 1}, std::vector<std::uint8_t>(32 * 1, 0x38u)};
  });
  // 4. Packed dense ATLAS3 weight packed with ATLAS2 width [32, 8] instead of [32, 12].
  expect_rejected("el_dense_a3_wrong_width", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.0.ffn_up_shexp.weight"] = StTensor{"U8", {32, 8}, std::vector<std::uint8_t>(32 * 8, 0)};
  });
  // 5. Packed dense ATLAS2 wrong weight dtype (F8_E4M3 instead of U8).
  expect_rejected("el_dense_a2_bad_wdtype", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.0.ffn_down_shexp.weight"] =
        StTensor{"F8_E4M3", {32, 8}, std::vector<std::uint8_t>(32 * 8, 0)};
  });
  // 6. Packed dense wrong scale dtype (U8 instead of F8_E4M3).
  expect_rejected("el_dense_packed_bad_sdtype", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.0.ffn_gate_shexp.weight_scale"] =
        StTensor{"U8", {32, 2}, std::vector<std::uint8_t>(32 * 2, 0x38u)};
  });
  // 7. Packed dense non-scalar weight_scale_2 (shape [2] instead of scalar).
  expect_rejected("el_dense_packed_nonscalar_s2", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.0.ffn_gate_shexp.weight_scale_2"] =
        StTensor{"F32", {2}, SyntheticElasticPack::f32_bytes(2, 0.5f)};
  });
  // 8. Packed dense ATLAS3 with in_features not divisible by 32 (in_features=16 -> hc_low_rank=8 won't work;
  //    set shared_expert_ffn_length=16 and make ffn_down_shexp [32, 16] declared ATLAS3).
  expect_rejected("el_dense_a3_in_not_div32", [](SyntheticElasticPack& pk) {
    pk.shared_expert_ffn_length = 16;
    pk.add_fp8("blk.0.ffn_gate_shexp", 16, 32);
    pk.add_fp8("blk.0.ffn_up_shexp", 16, 32);
    pk.add_fp8("blk.1.ffn_gate_shexp", 16, 32);
    pk.add_fp8("blk.1.ffn_up_shexp", 16, 32);
    pk.add_fp8("blk.1.ffn_down_shexp", 32, 16);
    pk.model_tensors["blk.0.ffn_down_shexp.weight"] = StTensor{"U8", {32, 6}, std::vector<std::uint8_t>(32 * 6, 0)};
    pk.model_tensors["blk.0.ffn_down_shexp.weight_scale"] =
        StTensor{"F8_E4M3", {32, 1}, std::vector<std::uint8_t>(32, 0x38u)};
    pk.model_tensors["blk.0.ffn_down_shexp.weight_scale_2"] =
        StTensor{"F32", {}, SyntheticElasticPack::f32_bytes(1, 0.5f)};
    pk.qlayers_json["blk.0.ffn_down_shexp"] = R"({"quant_algo":"ATLAS3","group_size":16})";
  });

  // --- B. Dense FP8 & fallback unquantized dtype/shape rejections ---
  // 9. FP8 projection self-consistent but shorter than config [64, 32].
  expect_rejected("el_dense_fp8_short", [](SyntheticElasticPack& pk) {
    pk.add_fp8("blk.0.attn_qkv", 32, 32);
  });
  // 10. FP8 wrong weight dtype (U8 instead of F8_E4M3).
  expect_rejected("el_dense_fp8_bad_dtype", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.0.attn_qkv.weight"] = StTensor{"U8", {64, 32}, std::vector<std::uint8_t>(64 * 32, 0)};
  });
  // 11. FP8 non-scalar weight_scale (shape [4]).
  expect_rejected("el_dense_fp8_bad_scale", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.0.attn_qkv.weight_scale"] =
        StTensor{"F32", {4}, SyntheticElasticPack::f32_bytes(4, 1.0f)};
  });
  // 12. Dense fallback MUST reject every non-F32/BF16 dtype (F16, U8, I8, F8_E4M3, I32) instead of treating as F32.
  for (const auto& [dt, bpe] : std::vector<std::pair<const char*, std::size_t>>{
           {"F16", 2}, {"U8", 1}, {"I8", 1}, {"F8_E4M3", 1}, {"I32", 4}}) {
    expect_rejected("el_dense_fallback_bad_dtype", [&](SyntheticElasticPack& pk) {
      pk.model_tensors["blk.0.ffn_gate_inp.weight"] =
          StTensor{dt, {4, 32}, std::vector<std::uint8_t>(4 * 32 * bpe, 0)};
    });
  }
  // 13. Dense unquantized weight self-consistent but wrong shape ([4, 64] instead of [8, 64]).
  expect_rejected("el_dense_bf16_short", [](SyntheticElasticPack& pk) {
    pk.add_bf16("blk.0.hc_attn_down.weight", {4, 64});
  });
  // 14. Explicit BF16 quant_algo disagrees with on-disk F32 tensor (and vice versa).
  expect_rejected("el_dense_bf16_mismatch", [](SyntheticElasticPack& pk) {
    pk.qlayers_json["blk.0.ssm_beta"] = R"({"quant_algo":"BF16"})";  // tensor is F32
  });
  expect_rejected("el_dense_f32_mismatch", [](SyntheticElasticPack& pk) {
    pk.qlayers_json["blk.0.ssm_alpha"] = R"({"quant_algo":"F32"})";  // tensor is BF16
  });
  // 15. Unknown quant_algo and unsupported group_size in hf_quant_config.json.
  expect_rejected("el_bad_quant_algo", [](SyntheticElasticPack& pk) {
    pk.qlayers_json["blk.0.attn_qkv"] = R"({"quant_algo":"INT4"})";
  });
  expect_rejected("el_bad_group_size", [](SyntheticElasticPack& pk) {
    pk.qlayers_json["blk.0.ffn_gate_shexp"] = R"({"quant_algo":"NVFP4","group_size":32})";
  });
  // 16. 1D/conv dense tensor short shape & invalid dtype.
  expect_rejected("el_norm_short", [](SyntheticElasticPack& pk) {
    pk.add_f32("blk.0.hc_attn_norm.weight", {32});
  });
  expect_rejected("el_conv_short", [](SyntheticElasticPack& pk) {
    pk.add_f32("blk.0.ssm_conv1d.weight", {32, 4});
  });
  expect_rejected("el_norm_bad_dtype", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.0.hc_attn_norm.weight"] = StTensor{"U8", {64}, std::vector<std::uint8_t>(64, 0)};
  });

  // --- C. Routed MoE gate/up/down rejections (self-consistent spans too short or mismatched for config) ---
  // 17. Too few expert scale_2 entries ([2] instead of [4]) on EACH of gate, up, and down projections.
  for (const char* proj : {"blk.0.ffn_gate_exps", "blk.0.ffn_up_exps", "blk.0.ffn_down_exps"}) {
    expect_rejected("el_moe_short_scale2", [&](SyntheticElasticPack& pk) {
      pk.model_tensors[std::string(proj) + ".weight_scale_2"] =
          StTensor{"F32", {2}, SyntheticElasticPack::f32_bytes(2, 0.5f)};
    });
  }
  // 18. Mutually self-consistent MoE tensors with too few experts (2 instead of config expert_count=4).
  for (const char* proj : {"blk.0.ffn_gate_exps", "blk.0.ffn_up_exps", "blk.0.ffn_down_exps"}) {
    expect_rejected("el_moe_few_experts", [&](SyntheticElasticPack& pk) {
      pk.add_moe_3d(proj, "ATLAS3", /*E=*/2, 32, 32);
    });
  }
  // 19. Self-consistent MoE tensors with too short out_features (16 instead of 32) on gate, up, and down.
  for (const char* proj : {"blk.0.ffn_gate_exps", "blk.0.ffn_up_exps", "blk.0.ffn_down_exps"}) {
    expect_rejected("el_moe_short_out", [&](SyntheticElasticPack& pk) {
      pk.add_moe_3d(proj, "ATLAS3", /*E=*/4, /*out_f=*/16, /*in_f=*/32);
    });
  }
  // 20. Packed code span too short for declared format (ATLAS2 width 8 packed under declared NVFP4/ATLAS3) on gate/up/down.
  for (const char* proj : {"blk.0.ffn_gate_exps", "blk.0.ffn_up_exps", "blk.0.ffn_down_exps"}) {
    expect_rejected("el_moe_short_packed", [&](SyntheticElasticPack& pk) {
      pk.model_tensors[std::string(proj) + ".weight"] =
          StTensor{"U8", {4, 32, 8}, std::vector<std::uint8_t>(4 * 32 * 8, 0)};
    });
  }
  // 21. MoE weight_scale inner dim too short ([4, 32, 1] instead of [4, 32, 2]).
  expect_rejected("el_moe_short_scale", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.1.ffn_gate_exps.weight_scale"] =
        StTensor{"F8_E4M3", {4, 32, 1}, std::vector<std::uint8_t>(4 * 32, 0x38u)};
  });
  // 22. MoE wrong dtypes on weight, weight_scale, and weight_scale_2.
  expect_rejected("el_moe_bad_wdtype", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.1.ffn_gate_exps.weight"] =
        StTensor{"F8_E4M3", {4, 32, 16}, std::vector<std::uint8_t>(4 * 32 * 16, 0)};
  });
  expect_rejected("el_moe_bad_sdtype", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.1.ffn_gate_exps.weight_scale"] =
        StTensor{"U8", {4, 32, 2}, std::vector<std::uint8_t>(4 * 32 * 2, 0x38u)};
  });
  expect_rejected("el_moe_bad_s2dtype", [](SyntheticElasticPack& pk) {
    pk.model_tensors["blk.1.ffn_gate_exps.weight_scale_2"] = StTensor{"U8", {4}, std::vector<std::uint8_t>(4, 1)};
  });
  // 23. Missing or non-codebook quant_algo on ffn_down_exps / ffn_up_exps (even when ffn_gate_exps is valid).
  expect_rejected("el_moe_missing_algo", [](SyntheticElasticPack& pk) {
    pk.qlayers_json.erase("blk.0.ffn_down_exps");
  });
  expect_rejected("el_moe_fp8_algo", [](SyntheticElasticPack& pk) {
    pk.qlayers_json["blk.0.ffn_up_exps"] = R"({"quant_algo":"FP8"})";
  });
  expect_rejected("el_moe_f32_algo", [](SyntheticElasticPack& pk) {
    pk.qlayers_json["blk.0.ffn_down_exps"] = R"({"quant_algo":"F32"})";
  });

  // --- D. Embedded PLE chunk layout rejections ---
  // 24. PLE chunk width disagrees with ple_head_dim=16 (shape [2, 16] instead of [2, 8]).
  expect_rejected("el_ple_bad_width", [](SyntheticElasticPack& pk) {
    pk.ple_chunks[0]["ple.chunk_0.weight"] = StTensor{"U8", {2, 16}, std::vector<std::uint8_t>(2 * 16, 0x22u)};
    pk.ple_chunks[0]["ple.chunk_0.weight_scale"] = StTensor{"F8_E4M3", {2, 2}, std::vector<std::uint8_t>(2 * 2, 0x38u)};
  });
  // 25. Embedded PLE row count too short to cover ple_head_offsets + ple_head_vocab_sizes (2 rows < 4 required).
  expect_rejected("el_ple_short_rows", [](SyntheticElasticPack& pk) {
    pk.ple_chunks.resize(1);
  });
  // 26. token_embd declared as ATLAS3 (read_embed_row_f32 supports F32/BF16/FP8/NVFP4 only).
  expect_rejected("el_embed_atlas3", [](SyntheticElasticPack& pk) {
    pk.add_packed_2d("token_embd", "ATLAS3", 4, 32);
  });
  // 27. PLE format == external with non-zero num_shards.
  expect_rejected("el_ple_ext_nonzero", [](SyntheticElasticPack& pk) {
    pk.ple_format = "external";  // ple_chunks still has 2 entries -> num_shards=2
  });
  // 28. Embedded PLE shards with empty ple_layers (and empty/short head_offsets/vocab_sizes) must be rejected before indexing.
  {
    TempDir td("el_ple_empty_layers");
    auto pk = SyntheticElasticPack::make_valid(/*with_ple=*/true, /*f32_embed=*/false);
    pk.write_to(td.path);
    // Rewrite config.json with ple.layers=[] alongside stray PLE metadata and empty head_offsets/head_vocab_sizes.
    std::ifstream in(td.path / "config.json");
    std::string cfg((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const auto pos = cfg.find("\"qwen4exp.ple.layers\": [0]");
    CHECK(pos != std::string::npos);
    cfg.replace(pos, std::strlen("\"qwen4exp.ple.layers\": [0]"), "\"qwen4exp.ple.layers\": []");
    const auto p_off = cfg.find("\"qwen4exp.ple.head_offsets\": [0, 2]");
    CHECK(p_off != std::string::npos);
    cfg.replace(p_off, std::strlen("\"qwen4exp.ple.head_offsets\": [0, 2]"), "\"qwen4exp.ple.head_offsets\": []");
    const auto p_voc = cfg.find("\"qwen4exp.ple.head_vocab_sizes\": [2, 2]");
    CHECK(p_voc != std::string::npos);
    cfg.replace(p_voc, std::strlen("\"qwen4exp.ple.head_vocab_sizes\": [2, 2]"),
                "\"qwen4exp.ple.head_vocab_sizes\": []");
    std::ofstream out(td.path / "config.json", std::ios::trunc);
    out << cfg;
    out.close();
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }
  // 29. Embedded PLE shards when config.json omits ple.layers completely must also be rejected.
  {
    TempDir td("el_ple_missing_layers");
    auto pk = SyntheticElasticPack::make_valid(/*with_ple=*/false, /*f32_embed=*/false);
    pk.write_to(td.path);
    // Inject embedded PLE config into hf_quant_config.json + a PLE shard while config.json has PLE disabled.
    write_tiny_safetensors(td.path / "ple-00001-of-00001.safetensors", {
        {"ple.chunk_0.weight", "U8", {2, 8}, std::vector<std::uint8_t>(16, 0x22u)},
        {"ple.chunk_0.weight_scale", "F8_E4M3", {2, 1}, std::vector<std::uint8_t>(2, 0x38u)},
        {"ple.chunk_0.weight_scale_2", "F32", {}, SyntheticElasticPack::f32_bytes(1, 0.25f)},
    });
    {
      std::ifstream in(td.path / "model.safetensors.index.json");
      std::string idx((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      in.close();
      const auto p = idx.rfind("}}");
      CHECK(p != std::string::npos);
      idx.insert(p,
                 ",\"ple.chunk_0.weight\":\"ple-00001-of-00001.safetensors\""
                 ",\"ple.chunk_0.weight_scale\":\"ple-00001-of-00001.safetensors\""
                 ",\"ple.chunk_0.weight_scale_2\":\"ple-00001-of-00001.safetensors\"");
      std::ofstream out(td.path / "model.safetensors.index.json", std::ios::trunc);
      out << idx;
    }
    {
      std::ifstream in(td.path / "hf_quant_config.json");
      std::string qc((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      in.close();
      const auto p = qc.find("\"format\": \"external\", \"num_shards\": 0");
      CHECK(p != std::string::npos);
      qc.replace(p, std::strlen("\"format\": \"external\", \"num_shards\": 0"),
                 "\"format\": \"nvfp4\", \"num_shards\": 1");
      std::ofstream out(td.path / "hf_quant_config.json", std::ios::trunc);
      out << qc;
    }
    CHECK_THROWS(Qwen4ExpElasticSet::inspect_headers(td.path));
  }
  // 30. Valid PLE-disabled Elastic model (explicit ple.layers=[], ple.format=external, num_shards=0) MUST load cleanly.
  {
    TempDir td("el_ple_disabled_ok");
    auto pk = SyntheticElasticPack::make_valid(/*with_ple=*/false, /*f32_embed=*/false);
    pk.write_to(td.path);
    std::ifstream in(td.path / "config.json");
    std::string cfg((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    const auto p = cfg.find("\"qwen4exp.expert_shared_feed_forward_length\": 32");
    CHECK(p != std::string::npos);
    cfg.insert(p + std::strlen("\"qwen4exp.expert_shared_feed_forward_length\": 32"),
               ",\n    \"qwen4exp.ple.layers\": []");
    std::ofstream out(td.path / "config.json", std::ios::trunc);
    out << cfg;
    out.close();
    const auto eset = Qwen4ExpElasticSet::inspect_headers(td.path);
    CHECK(!eset.has_embedded_ple());
    CHECK(eset.config().ple_layers.empty());
    CHECK(eset.config().ple_n_heads() == 0);
    CHECK(eset.config().ple_conv_history_cols() == 0);
  }
  // 31. Direct Qwen4ExpConfig wide-checked PLE overflow & inconsistent PLE-disabled rejection tests.
  {
    const auto base_model = Qwen4ExpMiniModelCpu::make_synthetic(123u);
    const Qwen4ExpConfig base = base_model.config();

    // Valid PLE-disabled config passes finalize_and_validate().
    {
      Qwen4ExpConfig disabled = base;
      disabled.ple_layers.clear();
      disabled.ple_ngram_size = 0;
      disabled.ple_heads_per_ngram = 0;
      disabled.ple_conv_kernel = 0;
      disabled.ple_head_dim = 0;
      disabled.ple_layer_multipliers.clear();
      disabled.ple_head_offsets.clear();
      disabled.ple_head_vocab_sizes.clear();
      disabled.finalize_and_validate();
      CHECK(disabled.ple_n_heads() == 0 && disabled.ple_conv_history_cols() == 0);
    }
    // Inconsistent PLE-disabled config (empty ple_layers with non-zero PLE hyperparameters) must throw.
    {
      Qwen4ExpConfig bad = base;
      bad.ple_layers.clear();
      CHECK_THROWS(bad.finalize_and_validate());
    }
    // Signed int overflow in (ple_ngram_size - 1) * ple_heads_per_ngram must throw (wide checked arithmetic).
    {
      Qwen4ExpConfig ovf_heads = base;
      ovf_heads.ple_ngram_size = 8;
      ovf_heads.ple_heads_per_ngram = std::numeric_limits<int>::max();
      CHECK_THROWS(ovf_heads.finalize_and_validate());
    }
    // Signed int overflow in (ple_conv_kernel - 1) * ple_ngram_size must throw (wide checked arithmetic).
    {
      Qwen4ExpConfig ovf_conv = base;
      ovf_conv.ple_ngram_size = 4;
      ovf_conv.ple_conv_kernel = std::numeric_limits<int>::max();
      CHECK_THROWS(ovf_conv.finalize_and_validate());
    }
    // Signed int overflow in ple_conv_history_cols * hc_dim must throw.
    {
      Qwen4ExpConfig ovf_hist_hc = base;
      ovf_hist_hc.ple_conv_kernel = 100000000;
      CHECK_THROWS(ovf_hist_hc.finalize_and_validate());
    }
    // Unsigned 64-bit overflow in ple_head_offsets[h] + ple_head_vocab_sizes[h] must throw.
    {
      Qwen4ExpConfig ovf_vocab = base;
      ovf_vocab.ple_head_offsets[0] = std::numeric_limits<std::uint64_t>::max() - 1;
      ovf_vocab.ple_head_vocab_sizes[0] = 4;
      CHECK_THROWS(ovf_vocab.finalize_and_validate());
    }
  }

  std::printf("PASS test_elastic_consumer_layout_rejections (31 dense/MoE/PLE/config layout & overflow cases)\n");
}

void test_real_quincy_elastic_packs() {
  int checked = 0;
  for (const char* p : {
           "/mnt/storage16tb/models/Qwen3.8-Flash-Next-atlas-70g-ingine",
           "/mnt/storage16tb/models/Qwen3.8-Flash-Next-atlas-56g-ingine",
           "/mnt/storage16tb/models/Qwen3.8-Flash-Next-atlas-42g-ingine",
       }) {
    const std::filesystem::path dir(p);
    if (!Qwen4ExpElasticSet::is_elastic_dir(dir)) {
      std::printf("SKIP test_real_quincy_elastic_packs (%s not present)\n", p);
      continue;
    }
    const auto eset = Qwen4ExpElasticSet::inspect_headers(dir);
    const auto& c = eset.config();
    CHECK(c.num_hidden_layers == 48);
    CHECK(c.hidden_size == 2560);
    CHECK(c.vocab_size == 248320);
    CHECK(c.expert_count == 512 && c.expert_ffn_length == 640);
    CHECK(eset.has_embedded_ple());
    // Spot-check validated expert layouts on boundary (L0) and core (L4) layers.
    const auto g0 = eset.validate_moe_experts("blk.0.ffn_gate_exps", c.expert_count, c.expert_ffn_length, c.hidden_size);
    const auto d4 = eset.validate_moe_experts("blk.4.ffn_down_exps", c.expert_count, c.hidden_size, c.expert_ffn_length);
    CHECK(g0.per_expert_code_bytes > 0 && g0.per_expert_scale_bytes == static_cast<std::uint64_t>(c.expert_ffn_length) * (c.hidden_size / 16));
    CHECK(d4.per_expert_code_bytes > 0 && d4.per_expert_scale_bytes == static_cast<std::uint64_t>(c.hidden_size) * (c.expert_ffn_length / 16));
    std::vector<float> emb(static_cast<std::size_t>(c.hidden_size), 0.0f);
    eset.read_embed_row_f32(760, emb);
    double l2 = 0.0;
    for (float v : emb) {
      CHECK(std::isfinite(v));
      l2 += double(v) * double(v);
    }
    CHECK(l2 > 0.0);
    std::vector<float> ple(static_cast<std::size_t>(c.ple_head_dim), 0.0f);
    eset.read_ple_row_f32(0, ple);
    for (float v : ple) CHECK(std::isfinite(v));
    ++checked;
    std::printf("  real elastic pack OK: %s (profile=%s, shards=%u, total_bytes=%llu, emb_l2=%.4f)\n", p,
                eset.profile().c_str(), eset.identity().shard_count,
                static_cast<unsigned long long>(eset.total_tensor_bytes()), std::sqrt(l2));
  }
  if (checked > 0) {
    std::printf("PASS test_real_quincy_elastic_packs (%d real 130-shard packs passed full 48-layer schema validation)\n",
                checked);
  }
}

}  // namespace

int main() {
  test_family_and_synthetic_gguf();
  test_synthetic_elastic_safetensors();
  test_elastic_safetensors_ingress_rejections();
  test_elastic_consumer_layout_rejections();
  test_ggml_dequantizers();
  test_ple_hash_and_eos_semantics();
  test_qsa_dense_and_sparse();
  test_mini_model_e2e();
  test_real_quincy_ud_q6_k_xl();
  test_real_quincy_elastic_packs();
  std::printf("qwen4exp_phase0: PASS\n");
  return 0;
}

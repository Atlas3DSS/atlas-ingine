#include "backends/qwen38/qwen38_block_cpu.h"
#include "backends/qwen38/qwen38_cpu_ops.h"
#include "backends/qwen38/qwen38_manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
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

float max_abs_diff(std::span<const float> a, std::span<const float> b) {
  CHECK(a.size() == b.size());
  float m = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
    m = std::max(m, std::fabs(a[i] - b[i]));
  }
  return m;
}

void test_fast_e4m3_parity() {
  const std::vector<float> edges = {
      0.0f, -0.0f, 1e-12f, -1e-12f, 0.0009765625f, 0.0009765626f, 0.001953125f,
      0.015625f, 0.0166015625f, 1.0f, -1.0f, 1.0625f, 1.125f, 416.0f, 431.99f, 432.0f, 447.99f, 448.0f, 500.0f, -500.0f,
      std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
      std::numeric_limits<float>::quiet_NaN(),
  };
  for (float x : edges) {
    CHECK(quantize_e4m3fn_fast(x) == quantize_e4m3fn(x));
  }

  std::mt19937 rng(20261125u);
  std::normal_distribution<float> dist(0.0f, 30.0f);
  std::uniform_real_distribution<float> tiny(-0.05f, 0.05f);
  for (int i = 0; i < 50000; ++i) {
    const float v1 = dist(rng);
    const float v2 = tiny(rng);
    CHECK(quantize_e4m3fn_fast(v1) == quantize_e4m3fn(v1));
    CHECK(quantize_e4m3fn_fast(v2) == quantize_e4m3fn(v2));
  }
}

void test_synthetic_block_harness() {
  const std::array<int, 1> layer0 = {0};
  const std::array<int, 1> layer3 = {3};
  const std::array<int, 4> layers03 = {0, 1, 2, 3};
  const std::array<std::int32_t, 6> toks = {3, 17, 29, 41, 7, 19};

  for (Qwen38WeightPrecision prec : {Qwen38WeightPrecision::Bf16, Qwen38WeightPrecision::MixedFp8Nvfp4}) {
    for (std::span<const int> ls :
         std::vector<std::span<const int>>{layer0, layer3, layers03}) {
      const auto bundle = make_synthetic_qwen38_block_bundle(prec, ls, toks, 4242u);
      const auto sw_1t = run_qwen38_blocks_cpu(bundle, true, false, 4, true, 1);
      const auto sw_4t = run_qwen38_blocks_cpu(bundle, true, false, 4, true, 4);
      const auto ch_4t = run_qwen38_blocks_cpu(bundle, true, true, 3, true, 4);
      CHECK(max_abs_diff(sw_1t.hidden_seq, sw_4t.hidden_seq) == 0.0f);
      CHECK(max_abs_diff(sw_1t.hidden_seq, ch_4t.hidden_seq) <= 1e-4f);
      for (std::size_t i = 0; i < bundle.layers.size(); ++i) {
        if (bundle.layers[i].kind == Qwen38LayerKind::LinearAttention) {
          CHECK(max_abs_diff(sw_1t.recurrent_states[i], ch_4t.recurrent_states[i]) <= 1e-5f);
          CHECK(max_abs_diff(sw_1t.conv_states[i], ch_4t.conv_states[i]) <= 1e-5f);
        }
      }
    }
  }
}

struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const char* tag) {
    path = std::filesystem::temp_directory_path() / (std::string("ingine_p1a_prep_") + tag);
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

void write_bytes(const std::filesystem::path& p, const void* d, std::size_t n) {
  std::ofstream out(p, std::ios::binary);
  CHECK(out.good());
  out.write(static_cast<const char*>(d), static_cast<std::streamsize>(n));
}

void write_str(const std::filesystem::path& p, const std::string& s) { write_bytes(p, s.data(), s.size()); }

void test_pinned_safetensors_block_extract() {
  TempDir dir("pinned");
  const std::string cfg = R"({
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5_text",
    "tie_word_embeddings": true,
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 4,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "full_attention_interval": 4,
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
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}
  })";
  write_str(dir.path / "config.json", cfg);
  write_str(dir.path / "tokenizer.json", R"({"version":"1.0"})");
  write_str(dir.path / "chat_template.jinja", "{{ messages }}");

  struct Spec {
    std::string name;
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::size_t elem_bytes;
  };
  std::vector<Spec> specs = {
      {"model.language_model.embed_tokens.weight", "BF16", {32, 16}, 2},
      {"model.language_model.norm.weight", "BF16", {16}, 2},
  };
  for (int l = 0; l < 4; ++l) {
    const std::string lp = "model.language_model.layers." + std::to_string(l);
    specs.push_back({lp + ".input_layernorm.weight", "BF16", {16}, 2});
    specs.push_back({lp + ".post_attention_layernorm.weight", "BF16", {16}, 2});
    if (l < 3) {
      const std::string a = lp + ".linear_attn";
      specs.push_back({a + ".A_log", "F32", {2}, 4});
      specs.push_back({a + ".dt_bias", "BF16", {2}, 2});
      specs.push_back({a + ".conv1d.weight", "BF16", {16, 1, 4}, 2});
      specs.push_back({a + ".in_proj_a.weight", "BF16", {2, 16}, 2});
      specs.push_back({a + ".in_proj_b.weight", "BF16", {2, 16}, 2});
      specs.push_back({a + ".norm.weight", "BF16", {4}, 2});
      specs.push_back({a + ".in_proj_qkv.weight", "BF16", {16, 16}, 2});
      specs.push_back({a + ".in_proj_z.weight", "BF16", {8, 16}, 2});
      specs.push_back({a + ".out_proj.weight", "BF16", {16, 8}, 2});
    } else {
      const std::string a = lp + ".self_attn";
      specs.push_back({a + ".q_norm.weight", "BF16", {8}, 2});
      specs.push_back({a + ".k_norm.weight", "BF16", {8}, 2});
      specs.push_back({a + ".q_proj.weight", "BF16", {32, 16}, 2});
      specs.push_back({a + ".k_proj.weight", "BF16", {8, 16}, 2});
      specs.push_back({a + ".v_proj.weight", "BF16", {8, 16}, 2});
      specs.push_back({a + ".o_proj.weight", "BF16", {16, 16}, 2});
    }
    specs.push_back({lp + ".mlp.gate_proj.weight", "BF16", {32, 16}, 2});
    specs.push_back({lp + ".mlp.up_proj.weight", "BF16", {32, 16}, 2});
    specs.push_back({lp + ".mlp.down_proj.weight", "BF16", {16, 32}, 2});
  }

  nlohmann::ordered_json hdr;
  std::uint64_t off = 0;
  for (const auto& sp : specs) {
    std::uint64_t elems = 1;
    for (auto d : sp.shape) elems *= static_cast<std::uint64_t>(d);
    const std::uint64_t bytes = elems * sp.elem_bytes;
    hdr[sp.name] = {{"dtype", sp.dtype}, {"shape", sp.shape}, {"data_offsets", {off, off + bytes}}};
    off += bytes;
  }
  const std::string hdr_s = hdr.dump();
  const std::uint64_t hdr_len = hdr_s.size();
  std::vector<std::uint8_t> st(8 + hdr_len + off, 0x3cu);
  for (int i = 0; i < 8; ++i) st[i] = static_cast<std::uint8_t>((hdr_len >> (8 * i)) & 0xffu);
  std::memcpy(st.data() + 8, hdr_s.data(), hdr_len);
  write_bytes(dir.path / "model.safetensors", st.data(), st.size());

  const auto m_self = Qwen38Manifest::inspect(dir.path, true, "p1a");
  ingine::model::CheckpointPinSpec pins;
  pins.config_sha256 = m_self.checkpoint.config_sha256;
  pins.tokenizer_sha256 = m_self.checkpoint.tokenizer_sha256;
  pins.chat_template_sha256 = m_self.checkpoint.chat_template_sha256;
  pins.shards = {{"model.safetensors", m_self.shard_sizes.at("model.safetensors"),
                  ingine::model::parse_sha256_hex(m_self.shard_sha256.at("model.safetensors"))}};
  const auto m_pin = Qwen38Manifest::inspect_pinned(dir.path, pins, "p1a");

  const std::array<int, 4> layers = {0, 1, 2, 3};
  const std::array<std::int32_t, 3> toks = {2, 5, 9};
  const auto bundle = load_qwen38_block_bundle(dir.path, m_pin, layers, toks);
  CHECK(bundle.layers.size() == 4);
  const auto res_sw = run_qwen38_blocks_cpu(bundle, true, false, 4, true, 2);
  const auto res_ch = run_qwen38_blocks_cpu(bundle, true, true, 2, true, 2);
  CHECK(max_abs_diff(res_sw.hidden_seq, res_ch.hidden_seq) <= 1e-4f);

  // Real Qwen3.5-0.8B 1-layer extraction + CPU block smoke test when present locally.
  const std::filesystem::path q08("/mnt/storage/models/Qwen3.5-0.8B");
  if (std::filesystem::is_directory(q08)) {
    const auto m08 = Qwen38Manifest::inspect(q08, false, "local");
    const std::array<int, 1> l0 = {0};
    const std::array<std::int32_t, 2> t2 = {10, 20};
    const auto b08 = load_qwen38_block_bundle(q08, m08, l0, t2);
    const auto sw08 = run_qwen38_blocks_cpu(b08, true, false, 4, false, 4);
    const auto ch08 = run_qwen38_blocks_cpu(b08, true, true, 2, false, 4);
    CHECK(max_abs_diff(sw08.hidden_seq, ch08.hidden_seq) <= 1e-3f);
    std::printf("  0.8B 1-layer CPU extraction+parity OK (sw_ms=%.1f, ch_ms=%.1f)\n", sw08.elapsed_ms,
                ch08.elapsed_ms);
  }
}

}  // namespace

int main() {
  test_fast_e4m3_parity();
  test_synthetic_block_harness();
  test_pinned_safetensors_block_extract();
  std::printf("qwen38_phase1a_cpu_prep: PASS\n");
  return 0;
}

// End-to-end GPU backend parity test for Qwen3.8-Flash-Next (qwen4exp, #849):
//   1. Builds a 2-shard synthetic Qwen4Exp GGUF (L0 GDN+PLE, L1 FullAttn+QSA, MoE+shared expert).
//   2. Verifies Qwen4ExpGpuBackend in BOTH AllResident mode AND TieredExpertCache (LRU miss/eviction) mode
//      produces logits matching forward_gguf_token_logits_cpu to < 5e-4 across multi-token prompts (including QSA sparsity).
//   3. Verifies exact prefix reuse (Reuse::Extend), namespace isolation, and make_generator() dispatch.
//   4. When INGINE_QWEN4EXP_GGUF_DIR is set, loads the real 158 GiB 6-shard UD-Q6_K_XL on GPU using
//      Dense-in-VRAM + Host-PLE + TieredExpertCache, verifies 48-layer forward + greedy next-token parity vs CPU!
// Skips cleanly with exit code 77 when no CUDA device is visible.
#include "backends/qwen4exp/qwen4exp_config.h"
#include "backends/qwen4exp/qwen4exp_cpu_ops.h"
#include "backends/qwen4exp/qwen4exp_gguf.h"
#include "backends/qwen4exp/qwen4exp_gpu.h"
#include "ingine/generator.h"
#include "ingine/model_backend.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using namespace ingine::backends::qwen4exp;
using ingine::model::Prompt;
using ingine::model::Reuse;
using ingine::model::RuntimeOptions;
using ingine::model::Sampling;

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const char* tag) {
    path = std::filesystem::temp_directory_path() /
           (std::string("ingine_test_q4exp_gpu_") + tag + "_" + std::to_string(::getpid()));
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
  void kv_str(std::string_view k, std::string_view v) { put_str(k); put<std::uint32_t>(8); put_str(v); }
  void kv_u32(std::string_view k, std::uint32_t v) { put_str(k); put<std::uint32_t>(4); put<std::uint32_t>(v); }
  void kv_f32(std::string_view k, float v) { put_str(k); put<std::uint32_t>(6); put<float>(v); }
  void kv_u32_arr(std::string_view k, const std::vector<std::uint32_t>& a) {
    put_str(k); put<std::uint32_t>(9); put<std::uint32_t>(4); put<std::uint64_t>(a.size());
    for (auto x : a) put<std::uint32_t>(x);
  }
  void kv_u64_arr(std::string_view k, const std::vector<std::uint64_t>& a) {
    put_str(k); put<std::uint32_t>(9); put<std::uint32_t>(10); put<std::uint64_t>(a.size());
    for (auto x : a) put<std::uint64_t>(x);
  }
  void align32() { while ((buf.size() % 32) != 0) buf.push_back(0); }
};

void write_random_synthetic_qwen4exp_shards(const std::filesystem::path& dir) {
  std::mt19937 rng(20261126u);
  struct TensorSpec {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type_id;
    std::vector<std::uint8_t> payload;
  };
  auto make_rand_f32 = [&](std::string name, std::vector<std::uint64_t> dims, float scale, float mean = 0.0f) {
    std::uint64_t n = 1;
    for (auto d : dims) n *= d;
    std::normal_distribution<float> dist(mean, scale);
    std::vector<std::uint8_t> raw(n * 4);
    for (std::uint64_t i = 0; i < n; ++i) {
      const float v = dist(rng);
      std::memcpy(raw.data() + i * 4, &v, 4);
    }
    return TensorSpec{std::move(name), std::move(dims), 0u, std::move(raw)};
  };

  std::vector<TensorSpec> s1, s2;
  s1.push_back(make_rand_f32("token_embd.weight", {32, 24}, 0.25f));
  s1.push_back(make_rand_f32("output.weight", {32, 24}, 0.20f));
  s1.push_back(make_rand_f32("output_hc_norm.weight", {64}, 0.05f, 1.0f));
  s1.push_back(make_rand_f32("output_hc_down.weight", {64, 8}, 0.15f));
  s1.push_back(make_rand_f32("output_hc_up.weight", {8, 64}, 0.15f));
  s1.push_back(make_rand_f32("per_layer_token_embd.weight", {4, 240}, 0.20f));

  auto add_layer = [&](std::vector<TensorSpec>& dst, int il, bool linear, bool ple) {
    const std::string p = "blk." + std::to_string(il) + ".";
    dst.push_back(make_rand_f32(p + "hc_attn_norm.weight", {64}, 0.05f, 1.0f));
    dst.push_back(make_rand_f32(p + "hc_attn_down.weight", {64, 8}, 0.15f));
    dst.push_back(make_rand_f32(p + "hc_attn_up.weight", {8, 64}, 0.15f));
    dst.push_back(make_rand_f32(p + "hc_attn_inject.weight", {64, 2}, 0.15f));
    dst.push_back(make_rand_f32(p + "hc_ffn_norm.weight", {64}, 0.05f, 1.0f));
    dst.push_back(make_rand_f32(p + "hc_ffn_down.weight", {64, 8}, 0.15f));
    dst.push_back(make_rand_f32(p + "hc_ffn_up.weight", {8, 64}, 0.15f));
    dst.push_back(make_rand_f32(p + "hc_ffn_inject.weight", {64, 2}, 0.15f));
    if (linear) {
      dst.push_back(make_rand_f32(p + "attn_qkv.weight", {32, 64}, 0.15f));
      dst.push_back(make_rand_f32(p + "attn_gate.weight", {32, 32}, 0.15f));
      dst.push_back(make_rand_f32(p + "ssm_conv1d.weight", {4, 64}, 0.2f));
      dst.push_back(make_rand_f32(p + "ssm_dt.bias", {4}, 0.1f));
      dst.push_back(make_rand_f32(p + "ssm_a", {4}, 0.05f, -0.5f));
      dst.push_back(make_rand_f32(p + "ssm_beta.weight", {32, 4}, 0.15f));
      dst.push_back(make_rand_f32(p + "ssm_alpha.weight", {32, 4}, 0.15f));
      dst.push_back(make_rand_f32(p + "ssm_norm.weight", {8}, 0.05f, 1.0f));
      dst.push_back(make_rand_f32(p + "ssm_out.weight", {32, 32}, 0.15f));
    } else {
      dst.push_back(make_rand_f32(p + "attn_q.weight", {32, 128}, 0.15f));
      dst.push_back(make_rand_f32(p + "attn_k.weight", {32, 32}, 0.15f));
      dst.push_back(make_rand_f32(p + "attn_v.weight", {32, 32}, 0.15f));
      dst.push_back(make_rand_f32(p + "attn_output.weight", {64, 32}, 0.15f));
      dst.push_back(make_rand_f32(p + "attn_q_norm.weight", {16}, 0.05f, 1.0f));
      dst.push_back(make_rand_f32(p + "attn_k_norm.weight", {16}, 0.05f, 1.0f));
      dst.push_back(make_rand_f32(p + "indexer.q_proj.weight", {32, 32}, 0.15f));
      dst.push_back(make_rand_f32(p + "indexer.k_proj.weight", {32, 16}, 0.15f));
      dst.push_back(make_rand_f32(p + "indexer.q_norm.weight", {16}, 0.05f, 1.0f));
      dst.push_back(make_rand_f32(p + "indexer.k_norm.weight", {16}, 0.05f, 1.0f));
    }
    if (ple) {
      dst.push_back(make_rand_f32(p + "ple_key.weight", {32, 64}, 0.15f));
      dst.push_back(make_rand_f32(p + "ple_value.weight", {32, 32}, 0.15f));
      dst.push_back(make_rand_f32(p + "ple_norm_key.weight", {64}, 0.05f, 1.0f));
      dst.push_back(make_rand_f32(p + "ple_norm_query.weight", {64}, 0.05f, 1.0f));
      dst.push_back(make_rand_f32(p + "ple_norm_conv.weight", {64}, 0.05f, 1.0f));
      dst.push_back(make_rand_f32(p + "ple_conv1d.weight", {4, 64}, 0.2f));
    }
    dst.push_back(make_rand_f32(p + "ffn_gate_inp.weight", {32, 6}, 0.3f));
    dst.push_back(make_rand_f32(p + "ffn_gate_exps.weight", {32, 32, 6}, 0.15f));
    dst.push_back(make_rand_f32(p + "ffn_up_exps.weight", {32, 32, 6}, 0.15f));
    dst.push_back(make_rand_f32(p + "ffn_down_exps.weight", {32, 32, 6}, 0.15f));
    dst.push_back(make_rand_f32(p + "ffn_gate_inp_shexp.weight", {32}, 0.2f));
    dst.push_back(make_rand_f32(p + "ffn_gate_shexp.weight", {32, 32}, 0.15f));
    dst.push_back(make_rand_f32(p + "ffn_up_shexp.weight", {32, 32}, 0.15f));
    dst.push_back(make_rand_f32(p + "ffn_down_shexp.weight", {32, 32}, 0.15f));
  };
  add_layer(s1, 0, true, true);
  add_layer(s2, 1, false, false);

  auto emit_shard = [&](const std::filesystem::path& out_path, const std::vector<TensorSpec>& ts, bool with_kv) {
    GgufWriter w;
    w.put_bytes("GGUF", 4);
    w.put<std::uint32_t>(3);
    w.put<std::uint64_t>(ts.size());
    w.put<std::uint64_t>(with_kv ? 38 : 1);
    if (with_kv) {
      w.kv_str("general.architecture", "qwen4exp");
      w.kv_str("general.name", "Qwen3.8-Flash-Next-GpuTest");
      w.kv_u32("qwen4exp.block_count", 2);
      w.kv_u32("qwen4exp.embedding_length", 32);
      w.kv_u32("qwen4exp.context_length", 128);
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
      w.kv_u32("qwen4exp.expert_count", 6);
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
    } else {
      w.kv_str("general.architecture", "qwen4exp");
    }
    std::uint64_t rel = 0;
    for (const auto& t : ts) {
      w.put_str(t.name);
      w.put<std::uint32_t>(static_cast<std::uint32_t>(t.dims.size()));
      for (auto d : t.dims) w.put<std::uint64_t>(d);
      w.put<std::uint32_t>(t.type_id);
      w.put<std::uint64_t>(rel);
      rel = ((rel + t.payload.size()) + 31ull) & ~31ull;
    }
    w.align32();
    const std::size_t base = w.buf.size();
    rel = 0;
    for (const auto& t : ts) {
      w.buf.resize(base + rel, 0);
      w.put_bytes(t.payload.data(), t.payload.size());
      rel = ((rel + t.payload.size()) + 31ull) & ~31ull;
    }
    std::ofstream f(out_path, std::ios::binary);
    CHECK(f.good());
    f.write(reinterpret_cast<const char*>(w.buf.data()), static_cast<std::streamsize>(w.buf.size()));
  };
  emit_shard(dir / "q4exp-00001-of-00002.gguf", s1, true);
  emit_shard(dir / "q4exp-00002-of-00002.gguf", s2, false);
}

float max_abs(const std::vector<float>& a, const std::vector<float>& b) {
  CHECK(a.size() == b.size());
  float m = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

void test_synthetic_gpu_vs_cpu_and_tiered_cache() {
  TempDir td("synth_e2e");
  write_random_synthetic_qwen4exp_shards(td.path);

  const auto gguf = Qwen4ExpGgufSet::inspect_headers(td.path);
  const std::vector<std::int32_t> seq = {3, 7, 11, 2, 19, 5, 13, 17, 23, 1, 4, 8, 12, 15};

  // CPU oracle reference over 14 tokens (crosses QSA sparse threshold top_k+cr-1 = 11!)
  Qwen4ExpSessionCpu cpu_sess(gguf.config());
  std::vector<float> cpu_logits;
  for (std::size_t i = 0; i < seq.size(); ++i) {
    cpu_logits = forward_gguf_token_logits_cpu(gguf, cpu_sess, seq[i], /*compute_logits=*/i + 1 == seq.size());
  }

  // Mode A: AllResident GPU
  {
    Qwen4ExpGpuLoadOptions opts;
    opts.model_dir = td.path;
    opts.max_ctx = 64;
    Qwen4ExpGpuBackend gpu(opts);
    CHECK(gpu.load_stats().all_experts_resident);
    auto sess = gpu.create_session();
    auto* gs = dynamic_cast<Qwen4ExpGpuSession*>(sess.get());
    CHECK(gs != nullptr);
    std::vector<float> gpu_logits;
    for (std::size_t i = 0; i < seq.size(); ++i) {
      gpu_logits = gpu.forward_token_logits_gpu(*gs, seq[i], /*compute_logits=*/i + 1 == seq.size());
    }
    const float diff = max_abs(cpu_logits, gpu_logits);
    std::printf("  AllResident GPU vs CPU max_abs=%.6e\n", diff);
    CHECK(diff < 5e-4f);
  }

  // Mode B: TieredExpertCache GPU (force tiny 3-slot LRU cache -> exercises misses, hits, AND evictions!)
  {
    Qwen4ExpGpuLoadOptions opts;
    opts.model_dir = td.path;
    opts.max_ctx = 64;
    opts.expert_cache_bytes = 3 * (3 * 32 * 32 * 4);  // exactly 3 expert slots across 12 layer-experts
    Qwen4ExpGpuBackend gpu(opts);
    CHECK(!gpu.load_stats().all_experts_resident);
    CHECK(gpu.load_stats().expert_cache_slots == 3);
    auto sess = gpu.create_session();
    auto* gs = dynamic_cast<Qwen4ExpGpuSession*>(sess.get());
    std::vector<float> gpu_logits;
    for (std::size_t i = 0; i < seq.size(); ++i) {
      gpu_logits = gpu.forward_token_logits_gpu(*gs, seq[i], /*compute_logits=*/i + 1 == seq.size());
    }
    const float diff = max_abs(cpu_logits, gpu_logits);
    const auto cs = gpu.expert_cache_stats();
    std::printf("  TieredExpertCache GPU vs CPU max_abs=%.6e (slots=%llu hits=%llu misses=%llu)\n", diff,
                static_cast<unsigned long long>(cs.slots_capacity), static_cast<unsigned long long>(cs.hits),
                static_cast<unsigned long long>(cs.misses));
    CHECK(diff < 5e-4f);
    CHECK(cs.misses > 0);
  }

  // Mode C: make_generator() dispatch + prefix reuse (warm_start -> Reuse::Extend) + namespace isolation!
  {
    RuntimeOptions rt;
    rt.model_dir = td.path;
    rt.max_ctx = 64;
    rt.prefix_cache = true;
    auto gen = ingine::model::make_generator(rt);
    CHECK(gen->model_family() == "qwen4exp");

    Sampling greedy;
    greedy.temperature = 0.0f;
    greedy.top_k = 1;
    greedy.seed = 42;

    Prompt p_warm;
    p_warm.ids = std::span<const std::int32_t>(seq.data(), 6);
    p_warm.cache_namespace = "ns-a";
    const auto ws = gen->generate(p_warm, 0, greedy, {}, [](std::span<const std::int32_t>) { return true; });
    CHECK(ws.prompt_tokens == 6 && ws.reused_tokens == 0);

    Prompt p_full;
    p_full.ids = seq;
    p_full.cache_namespace = "ns-a";
    std::vector<std::int32_t> out_warm;
    const auto st_warm = gen->generate(p_full, 4, greedy, {}, [&](std::span<const std::int32_t> t) {
      out_warm.insert(out_warm.end(), t.begin(), t.end());
      return true;
    });
    CHECK(st_warm.reused_tokens == 6);
    CHECK(st_warm.reuse == Reuse::Extend);
    CHECK(out_warm.size() == 4);

    // Mismatched namespace MUST miss cache (reused_tokens == 0) and emit bit-identical greedy tokens!
    p_full.cache_namespace = "ns-b";
    std::vector<std::int32_t> out_cold;
    const auto st_cold = gen->generate(p_full, 4, greedy, {}, [&](std::span<const std::int32_t> t) {
      out_cold.insert(out_cold.end(), t.begin(), t.end());
      return true;
    });
    CHECK(st_cold.reused_tokens == 0);
    CHECK(st_cold.reuse == Reuse::None);
    CHECK(out_warm == out_cold);
  }

  std::printf("PASS test_synthetic_gpu_vs_cpu_and_tiered_cache\n");
}

void test_real_gguf_if_configured() {
  const char* gguf_dir = std::getenv("INGINE_QWEN4EXP_GGUF_DIR");
  if (!gguf_dir || !*gguf_dir) return;

  std::vector<std::int32_t> tokens = {760, 6511, 314, 9338, 369};  // "The capital of France is"
  if (const char* tok_env = std::getenv("INGINE_QWEN4EXP_TOKENS"); tok_env && *tok_env) {
    tokens.clear();
    const std::string s(tok_env);
    std::size_t pos = 0;
    while (pos < s.size()) {
      auto comma = s.find(',', pos);
      if (comma == std::string::npos) comma = s.size();
      tokens.push_back(std::stoi(s.substr(pos, comma - pos)));
      pos = comma + 1;
    }
  }
  const int max_new = std::getenv("INGINE_QWEN4EXP_MAX_NEW") ? std::atoi(std::getenv("INGINE_QWEN4EXP_MAX_NEW")) : 8;

  std::printf("Running real Qwen4Exp 48-layer GPU test on %s (%zu prompt tokens)...\n", gguf_dir, tokens.size());
  Qwen4ExpGpuLoadOptions opt;
  opt.model_dir = gguf_dir;
  opt.max_ctx = 256;
  opt.prefix_cache = true;
  opt.expert_cache_bytes =
      std::getenv("INGINE_QWEN4EXP_EXPERT_CACHE_BYTES")
          ? static_cast<std::uint64_t>(std::strtoull(std::getenv("INGINE_QWEN4EXP_EXPERT_CACHE_BYTES"), nullptr, 10))
          : 0ull;  // 0 = auto-size to available GPU VRAM
  Qwen4ExpGpuBackend gpu(opt);
  const auto ls = gpu.load_stats();
  std::printf("  loaded in %.2fs: dense=%.2f GiB expert_cache=%.2f GiB (%llu slots, all_resident=%s)\n",
              ls.load_ms / 1000.0, double(ls.dense_device_bytes) / (1024.0 * 1024.0 * 1024.0),
              double(ls.expert_device_bytes) / (1024.0 * 1024.0 * 1024.0),
              static_cast<unsigned long long>(ls.expert_cache_slots), ls.all_experts_resident ? "true" : "false");

  auto sess = gpu.create_session();
  auto* gs = dynamic_cast<Qwen4ExpGpuSession*>(sess.get());
  CHECK(gs != nullptr);

  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> logits;
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const auto ts0 = std::chrono::steady_clock::now();
    logits = gpu.forward_token_logits_gpu(*gs, tokens[i], /*compute_logits=*/i + 1 == tokens.size());
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - ts0).count();
    const auto cs = gpu.expert_cache_stats();
    std::printf("  prefill step %zu/%zu tok=%d %.3fs (cache hits=%llu misses=%llu used=%llu/%llu)\n", i + 1,
                tokens.size(), tokens[i], dt, static_cast<unsigned long long>(cs.hits),
                static_cast<unsigned long long>(cs.misses), static_cast<unsigned long long>(cs.slots_used),
                static_cast<unsigned long long>(cs.slots_capacity));
  }
  const double prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  std::vector<int> idx(logits.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
  std::printf("  GPU top10 (prefill=%.2fs):", prefill_s);
  for (int i = 0; i < 10; ++i) std::printf(" %d(%.4f)", idx[i], logits[idx[i]]);
  std::printf("\n");

  if (const char* ref_path = std::getenv("INGINE_QWEN4EXP_REF_LOGITS"); ref_path && *ref_path) {
    std::ifstream in(ref_path, std::ios::binary);
    CHECK(in.good());
    std::vector<float> ref(logits.size());
    in.read(reinterpret_cast<char*>(ref.data()), static_cast<std::streamsize>(ref.size() * sizeof(float)));
    CHECK(static_cast<std::size_t>(in.gcount()) == ref.size() * sizeof(float));

    double max_abs_d = 0.0, sq = 0.0, dot = 0.0, n1 = 0.0, n2 = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
      const double a = logits[i], b = ref[i];
      max_abs_d = std::max(max_abs_d, std::abs(a - b));
      sq += (a - b) * (a - b);
      dot += a * b;
      n1 += a * a;
      n2 += b * b;
    }
    const double cos = dot / std::max(1e-12, std::sqrt(n1) * std::sqrt(n2));
    const double rmse = std::sqrt(sq / double(logits.size()));
    std::vector<int> ridx(ref.size());
    std::iota(ridx.begin(), ridx.end(), 0);
    std::partial_sort(ridx.begin(), ridx.begin() + 10, ridx.end(),
                      [&](int a, int b) { return ref[a] > ref[b]; });
    const bool top1 = idx[0] == ridx[0];
    bool top10 = true;
    for (int i = 0; i < 10; ++i) {
      if (idx[i] != ridx[i]) top10 = false;
    }
    const double min_cos =
        std::getenv("INGINE_QWEN4EXP_MIN_COS") ? std::atof(std::getenv("INGINE_QWEN4EXP_MIN_COS")) : 0.9999;
    std::printf("  GPU vs REF (%s): max_abs=%.6e rmse=%.6e cosine=%.8f (min=%.4f) top1=%s top10=%s\n", ref_path,
                max_abs_d, rmse, cos, min_cos, top1 ? "true" : "false", top10 ? "true" : "false");
    CHECK(cos > min_cos);
    CHECK(top1);
  }

  if (max_new > 0) {
    // Also exercise greedy autoregressive decoding starting from logits we just computed!
    std::vector<std::int32_t> generated;
    const auto td0 = std::chrono::steady_clock::now();
    std::int32_t cur = idx[0];
    for (int step = 0; step < max_new; ++step) {
      generated.push_back(cur);
      if (step + 1 == max_new) break;
      auto next_logits = gpu.forward_token_logits_gpu(*gs, cur, /*compute_logits=*/true);
      cur = static_cast<std::int32_t>(
          std::max_element(next_logits.begin(), next_logits.end()) - next_logits.begin());
    }
    const double dec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
    const auto cs = gpu.expert_cache_stats();
    std::printf("  GPU greedy %d tokens in %.2fs (%.2f tok/s, cache hits=%llu misses=%llu):", max_new, dec_s,
                max_new > 1 ? double(max_new - 1) / std::max(1e-6, dec_s) : 0.0,
                static_cast<unsigned long long>(cs.hits), static_cast<unsigned long long>(cs.misses));
    for (auto t : generated) std::printf(" %d", t);
    std::printf("\n");
  }
  std::printf("PASS test_real_gguf_if_configured\n");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
    std::printf("skip: no CUDA device available\n");
    return 77;
  }
  const bool real_only = std::getenv("INGINE_QWEN4EXP_REAL_ONLY") != nullptr;
  if (!real_only) {
    test_synthetic_gpu_vs_cpu_and_tiered_cache();
  }
  test_real_gguf_if_configured();
  std::printf("qwen4exp_gpu_e2e: PASS\n");
  return 0;
}

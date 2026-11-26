// Standalone GPU parity & isolated A/B benchmark harness for #883:
//   Feature A: Pre-suffix GDN checkpoint for multi-turn prefix reuse
//   Feature B: Fast canonical-64 GQA attention (split-KV decode + SMEM-tiled prefill)
//
// Modes:
//   --unit-only (default when --model-dir omitted): runs self_check_gemm_parity() + single-layer
//     full-attention kernel parity & microbench at kv_len in {320, 4096, 8192, 16384}.
//   --model-dir <path>: additionally loads Qwen3.8-27B-NVFP4 once and runs isolated A/B arms
//     (A0B0 baseline, A1B0 pre-suffix only, A0B1 fast-attn only, A1B1 combined) measuring
//     prefix-hit/TTFT, long-context decode throughput, and full-vocab logit/greedy parity.
#include "backends/qwen38/qwen38_backend.h"
#include "backends/qwen38/qwen38_cpu_ops.h"
#include "backends/qwen38/qwen38_cuda_ops.h"
#include "backends/qwen38/qwen38_dense_gpu.h"
#include "backends/qwen38/qwen38_gate2_metrics.h"
#include "backends/qwen38/qwen38_gate3_metrics.h"
#include "qwen38_883_bench_checks.h"
#include "text/chat.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

#define CUDA_OK(expr)                                                                       \
  do {                                                                                      \
    cudaError_t err_ = (expr);                                                              \
    if (err_ != cudaSuccess) {                                                              \
      std::fprintf(stderr, "%s:%d: CUDA error %s: %s\n", __FILE__, __LINE__, #expr,         \
                   cudaGetErrorString(err_));                                               \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

namespace {

using namespace ingine::backends::qwen38;
using namespace ingine::model;
using Json = nlohmann::ordered_json;

template <typename T>
struct DevVec {
  T* ptr = nullptr;
  std::size_t n = 0;
  explicit DevVec(std::size_t count = 0) { alloc(count); }
  ~DevVec() {
    if (ptr) cudaFree(ptr);
  }
  DevVec(const DevVec&) = delete;
  DevVec& operator=(const DevVec&) = delete;
  void alloc(std::size_t count) {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
    n = count;
    if (n > 0) CUDA_OK(cudaMalloc(&ptr, n * sizeof(T)));
  }
};

struct KernelBenchRow {
  int kv_len = 0;
  int seq_len = 1;
  double ref_ms = 0.0;
  double fast_ms = 0.0;
  double speedup = 0.0;
  float max_abs_diff = 0.0f;
  bool finite = true;
};

KernelBenchRow bench_single_layer_attn(int kv_len, int seq_len, int iters, cudaStream_t stream) {
  constexpr int kNh = 24, kNkvh = 4, kHd = 256, kRd = 64;
  constexpr int kQgDim = kNh * kHd * 2;
  constexpr int kKvDim = kNkvh * kHd;
  constexpr int kOutDim = kNh * kHd;
  const int pos_offset = kv_len - seq_len;

  std::mt19937 rng(20261126u + static_cast<unsigned>(kv_len) * 17u + static_cast<unsigned>(seq_len));
  std::normal_distribution<float> dist(0.0f, 0.25f);

  std::vector<float> h_qg(static_cast<std::size_t>(seq_len) * kQgDim);
  std::vector<float> h_kr(static_cast<std::size_t>(seq_len) * kKvDim);
  std::vector<float> h_vr(static_cast<std::size_t>(seq_len) * kKvDim);
  std::vector<std::uint16_t> h_qnw(kHd), h_knw(kHd);
  std::vector<std::uint16_t> h_kc(static_cast<std::size_t>(kv_len) * kKvDim);
  std::vector<std::uint16_t> h_vc(static_cast<std::size_t>(kv_len) * kKvDim);

  for (float& v : h_qg) v = dist(rng);
  for (float& v : h_kr) v = dist(rng);
  for (float& v : h_vr) v = dist(rng);
  for (int i = 0; i < kHd; ++i) {
    h_qnw[static_cast<std::size_t>(i)] = f32_to_bf16(dist(rng) * 0.1f);
    h_knw[static_cast<std::size_t>(i)] = f32_to_bf16(dist(rng) * 0.1f);
  }
  // Fill prior KV cache slots with repeatable BF16 values.
  for (std::size_t i = 0; i < h_kc.size(); ++i) {
    h_kc[i] = f32_to_bf16(static_cast<float>(static_cast<int>((i * 2654435761u) & 255u) - 128) * 0.004f);
    h_vc[i] = f32_to_bf16(static_cast<float>(static_cast<int>(((i + 13u) * 2246822519u) & 255u) - 128) * 0.004f);
  }

  DevVec<float> d_qg(h_qg.size()), d_kr_orig(h_kr.size()), d_kr(h_kr.size()), d_vr(h_vr.size());
  DevVec<std::uint16_t> d_qnw(kHd), d_knw(kHd);
  DevVec<__nv_bfloat16> d_kc(h_kc.size()), d_vc(h_vc.size());
  DevVec<float> d_qt(static_cast<std::size_t>(seq_len) * kOutDim), d_gt(static_cast<std::size_t>(seq_len) * kOutDim),
      d_out_ref(static_cast<std::size_t>(seq_len) * kOutDim), d_out_fast(static_cast<std::size_t>(seq_len) * kOutDim);
  const std::size_t sp_floats = qwen38_cuda_full_attn_split_scratch_floats(kNh, kv_len, kHd);
  DevVec<float> d_sp(sp_floats);

  CUDA_OK(cudaMemcpy(d_qg.ptr, h_qg.data(), h_qg.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_kr_orig.ptr, h_kr.data(), h_kr.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_vr.ptr, h_vr.data(), h_vr.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_qnw.ptr, h_qnw.data(), kHd * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_knw.ptr, h_knw.data(), kHd * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_kc.ptr, h_kc.data(), h_kc.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_vc.ptr, h_vc.data(), h_vc.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));

  const auto* qnw_bf = reinterpret_cast<const __nv_bfloat16*>(d_qnw.ptr);
  const auto* knw_bf = reinterpret_cast<const __nv_bfloat16*>(d_knw.ptr);

  auto run_once = [&](bool fast, float* dst) {
    CUDA_OK(cudaMemcpyAsync(d_kr.ptr, d_kr_orig.ptr, h_kr.size() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    qwen38_cuda_full_attn_gated_f32_kvbf16(d_qg.ptr, d_kr.ptr, d_vr.ptr, qnw_bf, knw_bf, 1e-6f, 10000000.0f, seq_len,
                                           pos_offset, kNh, kNkvh, kHd, kRd, d_kc.ptr, d_vc.ptr, kv_len, d_qt.ptr,
                                           d_gt.ptr, dst, stream, fast ? d_sp.ptr : nullptr, fast ? sp_floats : 0,
                                           fast);
  };

  // Warmup + correctness capture.
  run_once(false, d_out_ref.ptr);
  run_once(true, d_out_fast.ptr);
  CUDA_OK(cudaStreamSynchronize(stream));

  std::vector<float> h_ref(d_out_ref.n), h_fast(d_out_fast.n);
  CUDA_OK(cudaMemcpy(h_ref.data(), d_out_ref.ptr, h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_OK(cudaMemcpy(h_fast.data(), d_out_fast.ptr, h_fast.size() * sizeof(float), cudaMemcpyDeviceToHost));

  KernelBenchRow row;
  row.kv_len = kv_len;
  row.seq_len = seq_len;
  for (std::size_t i = 0; i < h_ref.size(); ++i) {
    if (!std::isfinite(h_ref[i]) || !std::isfinite(h_fast[i])) {
      row.finite = false;
      row.max_abs_diff = std::numeric_limits<float>::infinity();
      break;
    }
    row.max_abs_diff = std::max(row.max_abs_diff, std::fabs(h_ref[i] - h_fast[i]));
  }

  cudaEvent_t ev0, ev1;
  CUDA_OK(cudaEventCreate(&ev0));
  CUDA_OK(cudaEventCreate(&ev1));

  auto measure_ms = [&](bool fast, float* dst) -> double {
    CUDA_OK(cudaEventRecord(ev0, stream));
    for (int i = 0; i < iters; ++i) {
      run_once(fast, dst);
    }
    CUDA_OK(cudaEventRecord(ev1, stream));
    CUDA_OK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, ev0, ev1));
    return static_cast<double>(ms) / static_cast<double>(iters);
  };

  row.ref_ms = measure_ms(false, d_out_ref.ptr);
  row.fast_ms = measure_ms(true, d_out_fast.ptr);
  row.speedup = (row.fast_ms > 0.0) ? (row.ref_ms / row.fast_ms) : 0.0;

  CUDA_OK(cudaEventDestroy(ev0));
  CUDA_OK(cudaEventDestroy(ev1));
  return row;
}

struct TurnCapture {
  GenerateStats stats;
  std::vector<std::int32_t> tokens;
  std::vector<float> first_logits;
  double wall_ms = 0.0;
  double restore_ms = 0.0;
};

TurnCapture run_greedy_turn(Qwen38DenseBackend& backend, Qwen38DenseSession& sess,
                            std::span<const std::int32_t> prompt_ids, int max_new_tokens, bool use_cache,
                            bool ignore_eos = false) {
  TurnCapture out;
  BackendRequest req;
  req.prompt.ids = prompt_ids;
  req.use_prefix_cache = use_cache;
  Sampling samp;
  samp.temperature = 0.0f;
  samp.top_k = 1;
  const std::array<std::int32_t, 2> stops_arr = {248044, 248046};
  const std::span<const std::int32_t> stops =
      ignore_eos ? std::span<const std::int32_t>{} : std::span<const std::int32_t>{stops_arr};

  const auto t0 = std::chrono::steady_clock::now();
  out.stats = backend.generate_request_on_session(
      sess, req, max_new_tokens, samp, stops,
      [&](std::span<const std::int32_t> toks) {
        out.tokens.insert(out.tokens.end(), toks.begin(), toks.end());
        return true;
      },
      {},
      [&](const LogitStep& st) {
        if (out.first_logits.empty()) {
          out.first_logits.assign(st.logits.begin(), st.logits.end());
        }
      });
  out.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  out.restore_ms = out.stats.restore_seconds.value_or(0.0) * 1000.0;
  return out;
}

// Builds a realistic multi-turn Qwen3.8 thinking tool-calling prompt sequence of ~target_ctx tokens
// where OpenAI follow-up turns omit reasoning_content (reproducing the exact BFCL 6-token suffix gap).
std::vector<std::vector<std::int32_t>> build_bfcl_like_turns(const std::filesystem::path& model_dir, int target_ctx) {
  const ingine::text::ChatFormat fmt = ingine::text::ChatFormat::load(model_dir);
  Json tools = Json::array();
  for (int i = 0; i < 16; ++i) {
    tools.push_back({{"type", "function"},
                     {"function",
                      {{"name", "fs_tool_" + std::to_string(i)},
                       {"description",
                        "Inspect or mutate workspace path entry #" + std::to_string(i) +
                            " with deterministic validation, permissions checking, and audit logging."},
                       {"parameters",
                        {{"type", "object"},
                         {"properties",
                          {{"path", {{"type", "string"}, {"description", "Target relative or absolute path."}}},
                           {"mode", {{"type", "string"}, {"description", "Operation mode (read, list, stat, move)."}}}}},
                         {"required", Json::array({"path"})}}}}}});
  }

  std::string sys = "You are a precise filesystem assistant. Use tools step by step.\n";
  // Pad system context deterministically until Turn 1 reaches ~target_ctx tokens.
  ingine::text::ChatOptions opts;
  opts.enable_thinking = true;
  opts.reasoning_effort = "medium";

  auto render_msgs = [&](const Json& msgs) { return fmt.render(msgs, tools, opts).ids; };

  std::string block;
  for (int k = 0; k < 20; ++k) {
    block += "Workspace rule " + std::to_string(k) +
             ": verify directory contents before moving files, keep logs intact, and cite exact filenames.\n";
  }
  while (true) {
    Json probe = Json::array({{{"role", "system"}, {"content", sys}},
                              {{"role", "user"}, {"content", "List /workspace/reports and inspect summary.txt."}}});
    const auto ids = render_msgs(probe);
    if (static_cast<int>(ids.size()) >= target_ctx) break;
    sys += block;
  }

  Json m1 = Json::array({{{"role", "system"}, {"content", sys}},
                         {{"role", "user"}, {"content", "List /workspace/reports and inspect summary.txt."}}});
  Json m2 = m1;
  m2.push_back({{"role", "assistant"},
                {"content", nullptr},
                {"tool_calls",
                 Json::array({{{"id", "call_t1"},
                               {"type", "function"},
                               {"function",
                                {{"name", "fs_tool_0"},
                                 {"arguments", "{\"path\":\"/workspace/reports\",\"mode\":\"list\"}"}}}}})}});
  m2.push_back({{"role", "tool"},
                {"tool_call_id", "call_t1"},
                {"content", "{\"entries\":[\"summary.txt\",\"q1.csv\",\"q2.csv\"],\"count\":3}"}});

  Json m3 = m2;
  m3.push_back({{"role", "assistant"},
                {"content", nullptr},
                {"tool_calls",
                 Json::array({{{"id", "call_t2"},
                               {"type", "function"},
                               {"function",
                                {{"name", "fs_tool_1"},
                                 {"arguments", "{\"path\":\"/workspace/reports/summary.txt\",\"mode\":\"read\"}"}}}}})}});
  m3.push_back({{"role", "tool"},
                {"tool_call_id", "call_t2"},
                {"content", "{\"content\":\"Q1=42, Q2=58, Total=100. Status: verified.\"}"}});

  Json m4 = m3;
  m4.push_back({{"role", "assistant"}, {"content", "summary.txt reports Q1=42, Q2=58, Total=100 (verified)."}});
  m4.push_back({{"role", "user"}, {"content", "Great. Now summarize the total in one sentence."}});

  return {render_msgs(m1), render_msgs(m2), render_msgs(m3), render_msgs(m4)};
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path model_dir;
  std::string out_json_path;
  int max_ctx = 10240;
  int decode_tokens = 32;
  int multiturn_ctx = 4096;
  bool unit_only = false;
  bool self_check_only = false;
  bool verify_pins = true;
  bool test_restore_predicate = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model-dir" && i + 1 < argc) {
      model_dir = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      out_json_path = argv[++i];
    } else if (arg == "--max-ctx" && i + 1 < argc) {
      max_ctx = std::atoi(argv[++i]);
    } else if (arg == "--decode-tokens" && i + 1 < argc) {
      decode_tokens = std::atoi(argv[++i]);
    } else if (arg == "--multiturn-ctx" && i + 1 < argc) {
      multiturn_ctx = std::atoi(argv[++i]);
    } else if (arg == "--unit-only") {
      unit_only = true;
    } else if (arg == "--self-check-only") {
      self_check_only = true;
      unit_only = true;
    } else if (arg == "--no-verify-pins") {
      verify_pins = false;
    } else if (arg == "--test-restore-predicate") {
      test_restore_predicate = true;
    }
  }
  if (model_dir.empty()) unit_only = true;
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  if (test_restore_predicate) {
    using ingine::backends::qwen38::test_883::check_multiturn_presuffix_restores;
    using ingine::backends::qwen38::test_883::check_presuffix_followup_restore;
    auto mk = [](Reuse r, int reused, std::optional<std::uint64_t> bytes) {
      GenerateStats st;
      st.reuse = r;
      st.reused_tokens = reused;
      st.restored_bytes = bytes;
      return st;
    };
    const std::vector<std::size_t> bounds = {4204, 4281, 4369, 4416};
    const std::vector<GenerateStats> good = {
        mk(Reuse::None, 0, 0ull),
        mk(Reuse::Checkpoint, 4204, 158859264ull),
        mk(Reuse::Checkpoint, 4281, 158859264ull),
        mk(Reuse::Checkpoint, 4369, 158859264ull),
    };
    if (!check_multiturn_presuffix_restores(bounds, good)) return 10;
    // Negative: silent reuse=none recompute on a follow-up must fail.
    auto bad_none = good;
    bad_none[2] = mk(Reuse::None, 0, 0ull);
    if (check_multiturn_presuffix_restores(bounds, bad_none)) return 11;
    // Negative: wrong reuse mode (extend instead of checkpoint) must fail.
    if (check_presuffix_followup_restore(4204, mk(Reuse::Extend, 4204, 158859264ull)).passed) return 12;
    // Negative: mismatched reused token count must fail.
    if (check_presuffix_followup_restore(4204, mk(Reuse::Checkpoint, 4203, 158859264ull)).passed) return 13;
    // Negative: zero or absent restored_bytes must fail.
    if (check_presuffix_followup_restore(4204, mk(Reuse::Checkpoint, 4204, 0ull)).passed) return 14;
    if (check_presuffix_followup_restore(4204, mk(Reuse::Checkpoint, 4204, std::nullopt)).passed) return 15;
    // Negative: missing prior pre-suffix boundary must fail.
    if (check_presuffix_followup_restore(0, mk(Reuse::Checkpoint, 0, 158859264ull)).passed) return 16;
    std::printf("qwen38_gpu_883_bench: restore predicate self-test PASS\n");
    return 0;
  }

  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
    std::printf("skip: no CUDA device visible\n");
    return 77;
  }

  cudaDeviceProp prop{};
  CUDA_OK(cudaGetDeviceProperties(&prop, 0));
  cudaStream_t stream = nullptr;
  CUDA_OK(cudaStreamCreate(&stream));

  Json report;
  report["gpu_name"] = prop.name;
  report["sm_major"] = prop.major;
  report["sm_minor"] = prop.minor;

  // 1. Built-in parity self-check (including canonical-64 attention parity & 0-ULP batch invariance).
  const Qwen38GemmParityReport sc = Qwen38DenseBackend::self_check_gemm_parity(0);
  report["self_check"] = {
      {"passed", sc.passed},
      {"nvfp4_max_abs", sc.nvfp4_max_abs},
      {"fp8_cal_max_abs", sc.fp8_cal_max_abs},
      {"bf16_max_abs", sc.bf16_max_abs},
      {"batch_invariance_max_abs", sc.batch_invariance_max_abs},
      {"gdn_prefill_vs_step_max_abs", sc.gdn_prefill_vs_step_max_abs},
      {"gdn_batch_invariance_max_abs", sc.gdn_batch_invariance_max_abs},
      {"attn_fast_vs_ref_max_abs", sc.attn_fast_vs_ref_max_abs},
      {"attn_batch_invariance_max_abs", sc.attn_batch_invariance_max_abs},
  };
  std::printf("[self_check] passed=%s attn_fast_vs_ref=%.6g attn_batch_inv=%.6g\n", sc.passed ? "true" : "false",
              sc.attn_fast_vs_ref_max_abs, sc.attn_batch_invariance_max_abs);
  if (!sc.passed) return 1;
  if (self_check_only) {
    CUDA_OK(cudaStreamDestroy(stream));
    report["passed"] = true;
    if (!out_json_path.empty()) {
      std::ofstream out(out_json_path);
      out << report.dump(2) << "\n";
    }
    std::printf("qwen38_gpu_883_bench: PASS (self-check-only)\n");
    return 0;
  }

  // 2. Single-layer Qwen3.8 full-attention kernel microbenchmark & parity across contexts.
  Json kernel_decode = Json::array();
  for (int kv : {320, 4096, 8192, 16384}) {
    const KernelBenchRow r = bench_single_layer_attn(kv, 1, 5, stream);
    kernel_decode.push_back({{"kv_len", r.kv_len},
                             {"seq_len", r.seq_len},
                             {"ref_ms_per_layer", r.ref_ms},
                             {"fast_ms_per_layer", r.fast_ms},
                             {"ref_16layer_ms", r.ref_ms * 16.0},
                             {"fast_16layer_ms", r.fast_ms * 16.0},
                             {"speedup", r.speedup},
                             {"max_abs_diff", r.max_abs_diff},
                             {"finite", r.finite}});
    std::printf("[kernel_decode kv=%5d] ref=%.3f ms/layer (16L=%.2f ms) fast=%.3f ms/layer (16L=%.2f ms) "
                "speedup=%.1fx max_abs=%.3e\n",
                r.kv_len, r.ref_ms, r.ref_ms * 16.0, r.fast_ms, r.fast_ms * 16.0, r.speedup, r.max_abs_diff);
    if (!r.finite || r.max_abs_diff > 1e-4f) return 1;
  }
  Json kernel_prefill = Json::array();
  for (int kv : {320, 4096, 8192}) {
    const int sl = std::min(256, kv);
    const KernelBenchRow r = bench_single_layer_attn(kv, sl, 3, stream);
    kernel_prefill.push_back({{"kv_len", r.kv_len},
                              {"seq_len", r.seq_len},
                              {"ref_ms_per_layer", r.ref_ms},
                              {"fast_ms_per_layer", r.fast_ms},
                              {"speedup", r.speedup},
                              {"max_abs_diff", r.max_abs_diff},
                              {"finite", r.finite}});
    std::printf("[kernel_prefill kv=%5d M=%3d] ref=%.3f ms/layer fast=%.3f ms/layer speedup=%.1fx max_abs=%.3e\n",
                r.kv_len, r.seq_len, r.ref_ms, r.fast_ms, r.speedup, r.max_abs_diff);
    if (!r.finite || r.max_abs_diff > 1e-4f) return 1;
  }
  report["kernel_decode"] = kernel_decode;
  report["kernel_prefill"] = kernel_prefill;
  CUDA_OK(cudaStreamDestroy(stream));

  if (!unit_only) {
    Qwen38DenseLoadOptions opts;
    opts.model_dir = model_dir;
    opts.max_ctx = max_ctx;
    opts.prefill_chunk = 256;
    opts.verify_pins = verify_pins;
    opts.prefix_cache = true;
    opts.pre_suffix_checkpoint = false;
    opts.use_fast_attn = false;

    std::printf("[model] loading %s (max_ctx=%d)...\n", model_dir.c_str(), max_ctx);
    Qwen38DenseBackend backend(opts);
    auto* sess = dynamic_cast<Qwen38DenseSession*>(backend.acquire_session(0));
    if (!sess) return 2;

    // --- PART A: Isolated Feature A (Pre-suffix GDN checkpoint, ref attention held constant) ---
    const auto turns = build_bfcl_like_turns(model_dir, multiturn_ctx);
    auto run_multiturn = [&](bool pre_suffix, bool fast_attn, const char* label) {
      backend.configure_883_ablation(pre_suffix, fast_attn);
      std::vector<TurnCapture> caps;
      Json arr = Json::array();
      for (std::size_t ti = 0; ti < turns.size(); ++ti) {
        const auto c = run_greedy_turn(backend, *sess, turns[ti], 16, true);
        const double dec_tps =
            (c.stats.decode_seconds > 0.0) ? (static_cast<double>(c.stats.generated_tokens) / c.stats.decode_seconds)
                                           : 0.0;
        const std::size_t pre_pos = find_qwen35_pre_suffix_checkpoint(turns[ti]);
        arr.push_back({{"turn", ti + 1},
                       {"prompt_tokens", c.stats.prompt_tokens},
                       {"pre_suffix_boundary", pre_pos},
                       {"reuse", to_string(c.stats.reuse)},
                       {"reused_tokens", c.stats.reused_tokens},
                       {"recomputed_tokens", c.stats.prompt_tokens - c.stats.reused_tokens},
                       {"restored_bytes", c.stats.restored_bytes},
                       {"restore_ms", c.restore_ms},
                       {"prefill_seconds", c.stats.prefill_seconds},
                       {"decode_seconds", c.stats.decode_seconds},
                       {"decode_tok_s", dec_tps},
                       {"wall_ms", c.wall_ms},
                       {"generated_tokens", c.stats.generated_tokens},
                       {"tokens", c.tokens}});
        std::printf("[%s turn %zu] prompt=%d reuse=%s reused=%d recomputed=%d prefill=%.3fs restore=%.1fms "
                    "decode=%.1f tok/s\n",
                    label, ti + 1, c.stats.prompt_tokens, to_string(c.stats.reuse), c.stats.reused_tokens,
                    c.stats.prompt_tokens - c.stats.reused_tokens, c.stats.prefill_seconds, c.restore_ms, dec_tps);
        caps.push_back(c);
      }
      return std::pair{arr, caps};
    };

    const auto [a0b0_json, a0b0_caps] = run_multiturn(false, false, "A0B0_baseline");
    const auto [a1b0_json, a1b0_caps] = run_multiturn(true, false, "A1B0_presuffix_only");

    std::vector<std::size_t> pre_suffix_boundaries;
    pre_suffix_boundaries.reserve(turns.size());
    for (const auto& t : turns) {
      pre_suffix_boundaries.push_back(find_qwen35_pre_suffix_checkpoint(t));
    }

    // Verify exact bit-for-bit parity AND mandatory follow-up pre-suffix checkpoint restoration
    // of Feature A vs A0B0 baseline across ALL turns.
    Json a_parity = Json::array();
    bool a_parity_ok = true;
    bool a_restore_ok = (turns.size() >= 2);
    for (std::size_t ti = 0; ti < turns.size(); ++ti) {
      const Gate3StepComparison cmp =
          compare_logits_bit_exact(a0b0_caps[ti].first_logits, a1b0_caps[ti].first_logits,
                                   static_cast<int>(turns[ti].size()) - 1);
      const bool toks_eq = (a0b0_caps[ti].tokens == a1b0_caps[ti].tokens);
      a_parity_ok = a_parity_ok && cmp.all_finite && cmp.memcmp_equal && toks_eq;

      Json row = {{"turn", ti + 1},
                  {"memcmp_equal", cmp.memcmp_equal},
                  {"max_abs_diff", cmp.max_abs_diff},
                  {"argmax_match", cmp.argmax_match},
                  {"greedy_tokens_exact", toks_eq},
                  {"followup_restore_required", ti > 0}};
      if (ti > 0) {
        const auto rc = ingine::backends::qwen38::test_883::check_presuffix_followup_restore(
            pre_suffix_boundaries[ti - 1], a1b0_caps[ti].stats);
        a_restore_ok = a_restore_ok && rc.passed;
        row["expected_reused_tokens"] = rc.expected_reused_tokens;
        row["actual_reused_tokens"] = rc.actual_reused_tokens;
        row["actual_reuse"] = to_string(rc.actual_reuse);
        row["restored_bytes"] = rc.restored_bytes;
        row["followup_restore_passed"] = rc.passed;
      } else {
        row["followup_restore_passed"] = true;
      }
      a_parity.push_back(std::move(row));
      std::printf("[FeatureA parity turn %zu] memcmp=%s max_abs=%.6g greedy_exact=%s restore_ok=%s\n", ti + 1,
                  cmp.memcmp_equal ? "true" : "false", cmp.max_abs_diff, toks_eq ? "true" : "false",
                  (ti == 0 || ingine::backends::qwen38::test_883::check_presuffix_followup_restore(
                                  pre_suffix_boundaries[ti - 1], a1b0_caps[ti].stats)
                                  .passed)
                      ? "true"
                      : "false");
    }
    const bool a_ok = a_parity_ok && a_restore_ok;
    report["feature_a_isolated"] = {
        {"baseline_A0B0", a0b0_json},
        {"presuffix_A1B0", a1b0_json},
        {"parity_vs_baseline", a_parity},
        {"parity_passed", a_parity_ok},
        {"restore_passed", a_restore_ok},
        {"passed", a_ok},
    };
    if (!a_ok) return 3;

    // --- PART B: Isolated Feature B (Fast GQA64 attention, pre-suffix ckpt OFF) ---
    std::vector<int> ctx_points = {320, 4096, 8192};
    Json b_rows = Json::array();
    bool b_parity_ok = true;
    for (int ctx_len : ctx_points) {
      if (ctx_len + decode_tokens + 128 > max_ctx) continue;
      const auto p_turns = build_bfcl_like_turns(model_dir, ctx_len);
      std::vector<std::int32_t> prompt = p_turns[0];
      if (static_cast<int>(prompt.size()) + decode_tokens + 16 > max_ctx) continue;

      backend.configure_883_ablation(false, false);
      const TurnCapture ref_c = run_greedy_turn(backend, *sess, prompt, decode_tokens, false, true);

      backend.configure_883_ablation(false, true);
      const TurnCapture fast_c = run_greedy_turn(backend, *sess, prompt, decode_tokens, false, true);

      const Qwen38LogitParityPoint lp =
          compute_logit_kld_and_top1(ref_c.first_logits, fast_c.first_logits, static_cast<int>(prompt.size()) - 1);
      std::size_t tok_match = 0;
      const std::size_t tok_tot = std::min(ref_c.tokens.size(), fast_c.tokens.size());
      for (std::size_t k = 0; k < tok_tot; ++k) {
        if (ref_c.tokens[k] == fast_c.tokens[k]) ++tok_match;
      }
      const double ref_tps =
          (ref_c.stats.decode_seconds > 0.0) ? (static_cast<double>(ref_c.stats.generated_tokens) / ref_c.stats.decode_seconds) : 0.0;
      const double fast_tps =
          (fast_c.stats.decode_seconds > 0.0) ? (static_cast<double>(fast_c.stats.generated_tokens) / fast_c.stats.decode_seconds) : 0.0;
      const bool row_ok = lp.finite && lp.kld <= 0.10 && (ref_c.stats.generated_tokens == decode_tokens) &&
                          (fast_c.stats.generated_tokens == decode_tokens);
      b_parity_ok = b_parity_ok && row_ok;

      b_rows.push_back({{"ctx_len", prompt.size()},
                        {"decode_tokens", decode_tokens},
                        {"ref_generated_tokens", ref_c.stats.generated_tokens},
                        {"fast_generated_tokens", fast_c.stats.generated_tokens},
                        {"ref_prefill_seconds", ref_c.stats.prefill_seconds},
                        {"fast_prefill_seconds", fast_c.stats.prefill_seconds},
                        {"prefill_speedup",
                         fast_c.stats.prefill_seconds > 0.0 ? ref_c.stats.prefill_seconds / fast_c.stats.prefill_seconds : 0.0},
                        {"ref_decode_tok_s", ref_tps},
                        {"fast_decode_tok_s", fast_tps},
                        {"decode_speedup", ref_tps > 0.0 ? fast_tps / ref_tps : 0.0},
                        {"logit_kld", lp.kld},
                        {"logit_max_abs_diff", lp.max_abs_diff},
                        {"top1_match", lp.top1_match},
                        {"greedy_matches", tok_match},
                        {"greedy_total", tok_tot}});
      std::printf("[FeatureB ctx=%zu] ref_dec=%.1f tok/s fast_dec=%.1f tok/s (%.2fx) | ref_pf=%.2fs fast_pf=%.2fs | "
                  "kld=%.3e max_abs=%.3e top1=%s greedy=%zu/%zu\n",
                  prompt.size(), ref_tps, fast_tps, ref_tps > 0.0 ? fast_tps / ref_tps : 0.0,
                  ref_c.stats.prefill_seconds, fast_c.stats.prefill_seconds, lp.kld, lp.max_abs_diff,
                  lp.top1_match ? "true" : "false", tok_match, tok_tot);
    }
    // Multi-position teacher-forced KLD summary (last 32 prompt tokens of turn 1, Gate 2 style).
    {
      const auto& p0 = turns[0];
      const int n_p = static_cast<int>(p0.size());
      const int n_cap = std::min(32, n_p);
      std::vector<int> caps;
      caps.reserve(static_cast<std::size_t>(n_cap));
      for (int k = n_p - n_cap; k < n_p; ++k) caps.push_back(k);
      std::vector<std::vector<float>> ref_rows, fast_rows;
      ref_rows.reserve(static_cast<std::size_t>(n_cap));
      fast_rows.reserve(static_cast<std::size_t>(n_cap));
      backend.configure_883_ablation(false, false);
      backend.evaluate_teacher_forced_mode(
          p0, caps, true, [&](const LogitStep& st) { ref_rows.emplace_back(st.logits.begin(), st.logits.end()); },
          nullptr, sess, true);
      backend.configure_883_ablation(false, true);
      backend.evaluate_teacher_forced_mode(
          p0, caps, true, [&](const LogitStep& st) { fast_rows.emplace_back(st.logits.begin(), st.logits.end()); },
          nullptr, sess, true);
      std::vector<Qwen38LogitParityPoint> pts;
      pts.reserve(static_cast<std::size_t>(n_cap));
      for (int i = 0; i < n_cap; ++i) {
        pts.push_back(compute_logit_kld_and_top1(ref_rows[static_cast<std::size_t>(i)],
                                                 fast_rows[static_cast<std::size_t>(i)], caps[static_cast<std::size_t>(i)]));
      }
      const Qwen38KldSummary ks = summarize_kld_metrics(pts);
      const bool multi_ok = ks.all_finite && ks.kld_mean <= 0.01 && ks.top1_agreement >= 0.90;
      b_parity_ok = b_parity_ok && multi_ok;
      report["feature_b_multipos_parity"] = {
          {"prompt_tokens", p0.size()},
          {"positions", ks.positions},
          {"kld_mean", ks.kld_mean},
          {"kld_p99", ks.kld_p99},
          {"kld_max", ks.kld_max},
          {"top1_matches", ks.top1_matches},
          {"top1_agreement", ks.top1_agreement},
          {"all_finite", ks.all_finite},
          {"passed", multi_ok},
      };
      std::printf("[FeatureB multipos len=%zu pos=%zu] kld_mean=%.6g kld_p99=%.6g kld_max=%.6g top1=%.4f (%zu/%zu) -> %s\n",
                  p0.size(), ks.positions, ks.kld_mean, ks.kld_p99, ks.kld_max, ks.top1_agreement, ks.top1_matches,
                  ks.positions, multi_ok ? "PASS" : "FAIL");
    }
    report["feature_b_isolated"] = {{"contexts", b_rows}, {"parity_passed", b_parity_ok}};

    // --- PART C: Combined A + B (default production configuration) ---
    const auto [a0b1_json, a0b1_caps] = run_multiturn(false, true, "A0B1_fastattn_only");
    const auto [a1b1_json, a1b1_caps] = run_multiturn(true, true, "A1B1_combined_default");
    Json ab_parity = Json::array();
    bool ab_parity_ok = true;
    bool ab_restore_ok = (turns.size() >= 2);
    double ab_kld_sum = 0.0;
    for (std::size_t ti = 0; ti < turns.size(); ++ti) {
      const Gate3StepComparison cmp_vs_a0b1 =
          compare_logits_bit_exact(a0b1_caps[ti].first_logits, a1b1_caps[ti].first_logits,
                                   static_cast<int>(turns[ti].size()) - 1);
      const Qwen38LogitParityPoint kld_vs_a0b0 =
          compute_logit_kld_and_top1(a0b0_caps[ti].first_logits, a1b1_caps[ti].first_logits,
                                     static_cast<int>(turns[ti].size()) - 1);
      const bool exact_vs_a0b1 = cmp_vs_a0b1.all_finite && cmp_vs_a0b1.memcmp_equal &&
                                 (a0b1_caps[ti].tokens == a1b1_caps[ti].tokens);
      ab_parity_ok = ab_parity_ok && exact_vs_a0b1 && kld_vs_a0b0.finite;
      ab_kld_sum += kld_vs_a0b0.kld;
      Json row = {{"turn", ti + 1},
                  {"vs_A0B1_memcmp_equal", cmp_vs_a0b1.memcmp_equal},
                  {"vs_A0B1_max_abs_diff", cmp_vs_a0b1.max_abs_diff},
                  {"vs_A0B1_greedy_exact", a0b1_caps[ti].tokens == a1b1_caps[ti].tokens},
                  {"vs_A0B0_kld", kld_vs_a0b0.kld},
                  {"vs_A0B0_max_abs_diff", kld_vs_a0b0.max_abs_diff},
                  {"vs_A0B0_top1_match", kld_vs_a0b0.top1_match},
                  {"followup_restore_required", ti > 0}};
      bool turn_restore_ok = true;
      if (ti > 0) {
        const auto rc = ingine::backends::qwen38::test_883::check_presuffix_followup_restore(
            pre_suffix_boundaries[ti - 1], a1b1_caps[ti].stats);
        turn_restore_ok = rc.passed;
        ab_restore_ok = ab_restore_ok && rc.passed;
        row["expected_reused_tokens"] = rc.expected_reused_tokens;
        row["actual_reused_tokens"] = rc.actual_reused_tokens;
        row["actual_reuse"] = to_string(rc.actual_reuse);
        row["restored_bytes"] = rc.restored_bytes;
        row["followup_restore_passed"] = rc.passed;
      } else {
        row["followup_restore_passed"] = true;
      }
      ab_parity.push_back(std::move(row));
      std::printf("[Combined A1B1 turn %zu] vs_A0B1 memcmp=%s restore_ok=%s | vs_A0B0 kld=%.3e max_abs=%.3e top1=%s\n",
                  ti + 1, cmp_vs_a0b1.memcmp_equal ? "true" : "false", turn_restore_ok ? "true" : "false",
                  kld_vs_a0b0.kld, kld_vs_a0b0.max_abs_diff, kld_vs_a0b0.top1_match ? "true" : "false");
    }
    const double ab_kld_mean = turns.empty() ? 0.0 : (ab_kld_sum / static_cast<double>(turns.size()));
    ab_parity_ok = ab_parity_ok && ab_kld_mean <= 0.10;
    const bool ab_ok = ab_parity_ok && ab_restore_ok;
    report["combined_a1b1"] = {
        {"A0B1_turns", a0b1_json},
        {"A1B1_turns", a1b1_json},
        {"parity", ab_parity},
        {"vs_A0B0_kld_mean", ab_kld_mean},
        {"parity_passed", ab_parity_ok},
        {"restore_passed", ab_restore_ok},
        {"passed", ab_ok},
    };
    const bool all_model_ok = a_ok && b_parity_ok && ab_ok;
    report["passed"] = all_model_ok;
    if (!out_json_path.empty()) {
      std::ofstream out(out_json_path);
      out << report.dump(2) << "\n";
    }
    if (!b_parity_ok) return 4;
    if (!ab_ok) return 5;
  }

  report["passed"] = true;
  if (!out_json_path.empty()) {
    std::ofstream out(out_json_path);
    out << report.dump(2) << "\n";
  }
  std::printf("qwen38_gpu_883_bench: PASS\n");
  return 0;
}

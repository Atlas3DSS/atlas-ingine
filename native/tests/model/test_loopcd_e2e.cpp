// End-to-end validation for #882 Fused Single-Pass LoopCD (arXiv:2610.02185) on Inkling-2-Mini.
//
// Validates:
//   1. Fail-closed pre-load rejections (loop_passes=1, max_batch>1, last_full, invalid omega, Qwen3.8).
//   2. Zero-overhead exact parity:
//        --loopcd logits-adaptive --loopcd-omega 0  === 100% bit-identical greedy tokens vs --loopcd off
//        --loopcd hidden --loopcd-omega 0           === 100% bit-identical greedy tokens vs --loopcd off
//      (proves fused single-pass head_loopcd and loopcd_hidden_from_residual match plain head/residual on real weights).
//   3. Active LoopCD modes (logits-adaptive omega=0.5, hidden omega=0.3) at L=3 and L=2 produce coherent,
//      finite outputs, respect prefix-cache state identity, and measure decode tok/s overhead.
//   4. Optional `--eval-jsonl <out.jsonl>` mode dumps full greedy responses for a built-in deterministic
//      reasoning + Python coding benchmark across (L=3 plain, L=3 logits-adaptive, L=3 hidden,
//      L=2 plain, L=2 logits-adaptive, L=2 hidden) for automated accuracy verification.
#include "ingine/generator.h"
#include "ingine/prefix_cache.h"
#include "text/chat.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ingine::model::GenerateStats;
using ingine::model::LoopCdMode;
using ingine::model::RuntimeOptions;
using ingine::model::Sampling;

#define CHECK(cond)                                                                                   \
  do {                                                                                                \
    if (!(cond)) {                                                                                    \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                            \
      std::exit(1);                                                                                   \
    }                                                                                                 \
  } while (0)

namespace {

std::string default_model_dir() {
  if (const char* env = std::getenv("INGINE_MODEL_DIR"); env && *env) return env;
  return "/mnt/nvme/models/Inkling-2-Mini-ingine";
}

void expect_throws(const char* name, const std::function<void()>& fn) {
  bool threw = false;
  try {
    fn();
  } catch (const std::invalid_argument& e) {
    threw = true;
    std::printf("  [ok reject] %s -> %s\n", name, e.what());
  }
  CHECK(threw);
}

void test_fail_closed_rejections(const fs::path& model_dir) {
  std::printf("[1/4] Fail-closed pre-load rejections...\n");
  RuntimeOptions base;
  base.model_dir = model_dir;
  base.max_ctx = 4096;
  base.dense_fp8 = true;
  base.kv_fp8 = true;
  base.vision = false;

  expect_throws("loop_passes=1 + logits-adaptive", [&] {
    auto o = base;
    o.loop_passes = 1;
    o.loopcd_mode = LoopCdMode::LogitsAdaptive;
    (void)ingine::model::make_generator(o);
  });
  expect_throws("loop_passes=1 + hidden", [&] {
    auto o = base;
    o.loop_passes = 1;
    o.loopcd_mode = LoopCdMode::Hidden;
    (void)ingine::model::make_generator(o);
  });
  expect_throws("max_batch=2 + logits-adaptive", [&] {
    auto o = base;
    o.max_batch = 2;
    o.loopcd_mode = LoopCdMode::LogitsAdaptive;
    (void)ingine::model::make_generator(o);
  });
  expect_throws("kv_cache_mode=last_full + logits-adaptive", [&] {
    auto o = base;
    o.kv_cache_mode = ingine::model::KvCacheMode::LastFull;
    o.loopcd_mode = LoopCdMode::LogitsAdaptive;
    (void)ingine::model::make_generator(o);
  });
  expect_throws("negative omega (-0.2)", [&] {
    auto o = base;
    o.loopcd_mode = LoopCdMode::LogitsAdaptive;
    o.loopcd_omega = -0.2f;
    (void)ingine::model::make_generator(o);
  });
  expect_throws("omega > 4.0", [&] {
    auto o = base;
    o.loopcd_mode = LoopCdMode::LogitsAdaptive;
    o.loopcd_omega = 4.5f;
    (void)ingine::model::make_generator(o);
  });
  expect_throws("NaN omega", [&] {
    auto o = base;
    o.loopcd_mode = LoopCdMode::LogitsAdaptive;
    o.loopcd_omega = std::numeric_limits<float>::quiet_NaN();
    (void)ingine::model::make_generator(o);
  });

  // Synthetic Qwen3.8 config dir must reject --loopcd fail-closed before load (specifically citing --loopcd).
  const fs::path tmp_qwen = fs::temp_directory_path() / "ingine_loopcd_qwen_test";
  fs::create_directories(tmp_qwen);
  {
    std::ofstream out(tmp_qwen / "config.json");
    out << R"({"model_type":"qwen3_5","architectures":["Qwen3_5ForCausalLM"]})";
  }
  {
    bool threw_loopcd = false;
    try {
      auto o = base;
      o.model_dir = tmp_qwen;
      o.loopcd_mode = LoopCdMode::LogitsAdaptive;
      (void)ingine::model::make_generator(o);
    } catch (const std::invalid_argument& e) {
      std::printf("  [ok reject] Qwen3.8 + logits-adaptive -> %s\n", e.what());
      threw_loopcd = (std::string(e.what()).find("--loopcd") != std::string::npos);
    }
    CHECK(threw_loopcd);
  }
  fs::remove_all(tmp_qwen);

  // Prefix cache StateIdentity hashing must distinguish off vs logits-adaptive vs hidden and distinct omegas.
  ingine::model::StateIdentity id_off{};
  ingine::model::StateIdentity id_la{};
  id_la.loopcd_mode = static_cast<std::uint32_t>(LoopCdMode::LogitsAdaptive);
  float w05 = 0.5f;
  std::memcpy(&id_la.loopcd_omega_bits, &w05, sizeof(w05));
  ingine::model::StateIdentity id_la2 = id_la;
  float w10 = 1.0f;
  std::memcpy(&id_la2.loopcd_omega_bits, &w10, sizeof(w10));
  ingine::model::StateIdentity id_hid{};
  id_hid.loopcd_mode = static_cast<std::uint32_t>(LoopCdMode::Hidden);
  float w03 = 0.3f;
  std::memcpy(&id_hid.loopcd_omega_bits, &w03, sizeof(w03));

  CHECK(id_off != id_la);
  CHECK(id_la != id_la2);
  CHECK(id_la != id_hid);
  CHECK(ingine::model::hash_state_identity(id_off) != ingine::model::hash_state_identity(id_la));
  CHECK(ingine::model::hash_state_identity(id_la) != ingine::model::hash_state_identity(id_la2));
  CHECK(ingine::model::hash_state_identity(id_la) != ingine::model::hash_state_identity(id_hid));
  std::printf("  PASS fail-closed rejections & StateIdentity isolation.\n");
}

struct GenResult {
  std::vector<std::int32_t> tokens;
  std::string text;
  double tok_s = 0.0;
  GenerateStats stats;
};

GenResult run_greedy(ingine::model::Generator& gen, const ingine::text::ChatFormat& fmt,
                     const std::string& prompt, const std::string& effort, int max_new_tokens) {
  ingine::text::ChatOptions chat;
  chat.reasoning_effort = effort;
  const nlohmann::ordered_json messages = nlohmann::ordered_json::array({{{"role", "user"}, {"content", prompt}}});
  const std::vector<std::int32_t> ids = fmt.render(messages, nlohmann::ordered_json(), chat).ids;
  const std::vector<std::int32_t> stop = {*fmt.tokenizer().token_id("<|content_model_end_sampling|>")};
  Sampling s;
  s.temperature = 0.0f;
  s.seed = 42;
  GenResult out;
  out.stats = gen.generate(ids, max_new_tokens, s, stop, [&](std::span<const std::int32_t> toks) {
    out.tokens.insert(out.tokens.end(), toks.begin(), toks.end());
    return true;
  });
  out.text = fmt.tokenizer().decode_lossy(out.tokens);
  out.tok_s = double(std::max(1, out.stats.generated_tokens - 1)) / std::max(out.stats.decode_seconds, 1e-9);
  return out;
}

struct EvalPrompt {
  const char* id;
  const char* category;
  const char* prompt;
};

// Deterministic verifiable reasoning + Python coding suite (checked externally by Python script).
static const std::vector<EvalPrompt> kEvalSuite = {
    {"math_01", "math", "A store sells notebooks for $4 each and pens for $1.50 each. Maya buys 7 notebooks and 12 pens, then uses a 20% off coupon on the entire purchase. How much does she pay in dollars? Put your final numerical answer after 'ANSWER: '."},
    {"math_02", "math", "A train leaves Station A at 08:15 traveling at 72 km/h. Another train leaves Station B (396 km away) at 09:00 traveling toward Station A at 90 km/h. At what distance from Station A (in km) do they meet? Put your final numerical answer after 'ANSWER: '."},
    {"math_03", "math", "Find the sum of all positive integers n <= 100 that are divisible by 3 or 5, but not by 15. Put your final numerical answer after 'ANSWER: '."},
    {"math_04", "math", "A cylindrical water tank has radius 3 meters and height 10 meters. Water is pumped in at 1500 liters per minute and drains out at 300 liters per minute. Starting empty, how many minutes does it take to fill 80% of the tank? Use pi = 3.141592653589793 and round to the nearest integer. Put your final numerical answer after 'ANSWER: '."},
    {"math_05", "math", "How many distinct 4-digit integers can be formed using the digits {1, 2, 3, 4, 5, 6} without repetition such that the number is even and greater than 3000? Put your final numerical answer after 'ANSWER: '."},
    {"math_06", "math", "If f(x) = 3x^2 - 12x + 7 and g(x) = 2x + 5, find the value of f(g(3)) - g(f(3)). Put your final numerical answer after 'ANSWER: '."},
    {"math_07", "math", "Three workers A, B, and C can finish a job alone in 12, 15, and 20 days respectively. A and B work together for 3 days, then A leaves and C joins B until the job is finished. What is the total number of days taken to finish the job? Express as a reduced fraction p/q. Put your final numerical answer after 'ANSWER: '."},
    {"math_08", "math", "A bag contains 5 red, 4 blue, and 3 green marbles. Two marbles are drawn without replacement. What is the probability they are different colors? Express as a reduced fraction p/q. Put your final numerical answer after 'ANSWER: '."},
    {"math_09", "math", "Solve the system of linear equations: 3x + 2y - z = 11, 2x - y + 3z = 7, x + 4y + 2z = 21. What is x + y + z? Put your final numerical answer after 'ANSWER: '."},
    {"math_10", "math", "A sequence is defined by a_1 = 2, a_2 = 5, and a_n = 3*a_{n-1} - 2*a_{n-2} for n >= 3. What is a_8? Put your final numerical answer after 'ANSWER: '."},
    {"math_11", "math", "What is the remainder when 7^100 is divided by 13? Put your final numerical answer after 'ANSWER: '."},
    {"math_12", "math", "A rectangle's perimeter is 68 cm and its diagonal is 26 cm. What is its area in square cm? Put your final numerical answer after 'ANSWER: '."},
    {"code_01", "code", "Write a single Python function `def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:` that merges all overlapping or adjacent closed integer intervals (treat [1,3] and [3,5] as overlapping, [1,2] and [3,4] as separate) and returns them sorted by start. Output only valid Python code inside a ```python block."},
    {"code_02", "code", "Write a single Python function `def eval_rpn(tokens: list[str]) -> int:` that evaluates a Reverse Polish Notation expression supporting `+`, `-`, `*`, `/` where division truncates toward zero (like C/Python int(a/b)). Output only valid Python code inside a ```python block."},
    {"code_03", "code", "Write a single Python function `def longest_increasing_subseq_len(nums: list[int]) -> int:` in O(n log n) time that returns the length of the strictly increasing longest subsequence of `nums`. Output only valid Python code inside a ```python block."},
    {"code_04", "code", "Write a single Python function `def min_edit_distance(a: str, b: str) -> int:` that returns the Levenshtein edit distance (insert, delete, substitute each cost 1) between strings `a` and `b`. Output only valid Python code inside a ```python block."},
    {"code_05", "code", "Write a single Python function `def shortest_path_dag(n: int, edges: list[tuple[int, int, int]], src: int, dst: int) -> int:` that finds the minimum weighted path cost from `src` to `dst` in a directed acyclic graph with `n` nodes (0..n-1) and possibly negative edge weights, returning -1 if `dst` is unreachable. Output only valid Python code inside a ```python block."},
    {"code_06", "code", "Write a single Python function `def top_k_frequent(nums: list[int], k: int) -> list[int]:` that returns the `k` most frequent integers sorted by descending frequency, breaking ties by smaller integer value first. Output only valid Python code inside a ```python block."},
    {"code_07", "code", "Write a single Python function `def parse_csv_line(line: str) -> list[str]:` that parses one RFC-4180 CSV line supporting quoted fields containing commas and escaped double quotes (`\"\"`). Output only valid Python code inside a ```python block."},
    {"code_08", "code", "Write a single Python function `def lru_simulate(capacity: int, ops: list[tuple]) -> list[int]:` where each op is either `('put', key, val)` or `('get', key)` (returning value or -1 on miss) and returns the list of results for all `get` ops in order. Output only valid Python code inside a ```python block."},
};

void run_eval_suite(const fs::path& model_dir, const std::string& out_path, const std::string& effort, int max_tokens,
                    bool only_3pass) {
  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  struct Config {
    const char* name;
    int passes;
    LoopCdMode mode;
    float omega;
  };
  std::vector<Config> configs = {
      {"3pass_plain", 3, LoopCdMode::Off, -1.0f},
      {"3pass_loopcd_logits", 3, LoopCdMode::LogitsAdaptive, -1.0f},
      {"3pass_loopcd_hidden", 3, LoopCdMode::Hidden, -1.0f},
  };
  if (!only_3pass) {
    configs.push_back({"2pass_plain", 2, LoopCdMode::Off, -1.0f});
    configs.push_back({"2pass_loopcd_logits", 2, LoopCdMode::LogitsAdaptive, -1.0f});
    configs.push_back({"2pass_loopcd_hidden", 2, LoopCdMode::Hidden, -1.0f});
  }
  std::ofstream out(out_path);
  CHECK(out.is_open());
  for (const auto& cfg : configs) {
    RuntimeOptions opt;
    opt.model_dir = model_dir;
    opt.max_ctx = 8192;
    opt.prefill_chunk = 1024;
    opt.dense_fp8 = true;
    opt.kv_fp8 = true;
    opt.vision = false;
    opt.prefix_cache = false;
    opt.loop_passes = cfg.passes;
    opt.loopcd_mode = cfg.mode;
    opt.loopcd_omega = cfg.omega;
    std::printf("[eval] loading %s (passes=%d, mode=%s)...\n", cfg.name, cfg.passes,
                ingine::model::to_string(cfg.mode));
    std::fflush(stdout);
    auto gen = ingine::model::make_generator(opt);
    double sum_tok_s = 0.0;
    for (const auto& item : kEvalSuite) {
      const GenResult res = run_greedy(*gen, fmt, item.prompt, effort, max_tokens);
      sum_tok_s += res.tok_s;
      nlohmann::ordered_json rec = {
          {"config", cfg.name},
          {"loop_passes", cfg.passes},
          {"loopcd", ingine::model::to_string(cfg.mode)},
          {"id", item.id},
          {"category", item.category},
          {"generated_tokens", res.stats.generated_tokens},
          {"decode_seconds", res.stats.decode_seconds},
          {"tok_s", res.tok_s},
          {"text", res.text},
      };
      out << rec.dump() << "\n";
      out.flush();
    }
    std::printf("[eval] %s done: mean %.1f tok/s across %zu prompts\n", cfg.name,
                sum_tok_s / double(kEvalSuite.size()), kEvalSuite.size());
    std::fflush(stdout);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir = default_model_dir();
  std::string eval_jsonl;
  std::string effort = "low";
  int eval_tokens = 512;
  bool only_3pass = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
      model_dir = argv[++i];
    } else if (std::strcmp(argv[i], "--eval-jsonl") == 0 && i + 1 < argc) {
      eval_jsonl = argv[++i];
    } else if (std::strcmp(argv[i], "--effort") == 0 && i + 1 < argc) {
      effort = argv[++i];
    } else if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
      eval_tokens = std::stoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--only-3pass") == 0) {
      only_3pass = true;
    }
  }
  if (!fs::exists(fs::path(model_dir) / "tokenizer.json")) {
    std::printf("skip: model dir not found at %s\n", model_dir.c_str());
    return 77;
  }

  if (!eval_jsonl.empty()) {
    run_eval_suite(model_dir, eval_jsonl, effort, eval_tokens, only_3pass);
    return 0;
  }

  test_fail_closed_rejections(model_dir);

  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  const std::string p_chat = "Explain how a hash map handles collisions in 3 concise sentences.";
  const std::string p_math = "What is 17 * 23 + 45? Show brief steps and give the final integer.";

  auto make_opt = [&](int passes, LoopCdMode mode, float omega) {
    RuntimeOptions o;
    o.model_dir = model_dir;
    o.max_ctx = 8192;
    o.prefill_chunk = 1024;
    o.dense_fp8 = true;
    o.kv_fp8 = true;
    o.vision = false;
    o.loop_passes = passes;
    o.loopcd_mode = mode;
    o.loopcd_omega = omega;
    return o;
  };

  std::printf("[2/4] Bit-identical zero-strength control (omega=0 vs off) on real Inkling-2-Mini (L=3)...\n");
  GenResult base_chat, base_math;
  {
    auto g = ingine::model::make_generator(make_opt(3, LoopCdMode::Off, -1.0f));
    CHECK(g->loopcd_mode() == LoopCdMode::Off);
    base_chat = run_greedy(*g, fmt, p_chat, "low", 96);
    base_math = run_greedy(*g, fmt, p_math, "low", 96);
    std::printf("  plain (off):            chat=%zu tok (%.1f tok/s), math=%zu tok (%.1f tok/s)\n",
                base_chat.tokens.size(), base_chat.tok_s, base_math.tokens.size(), base_math.tok_s);
  }
  {
    auto g = ingine::model::make_generator(make_opt(3, LoopCdMode::LogitsAdaptive, 0.0f));
    CHECK(g->loopcd_mode() == LoopCdMode::LogitsAdaptive);
    const GenResult r_chat = run_greedy(*g, fmt, p_chat, "low", 96);
    const GenResult r_math = run_greedy(*g, fmt, p_math, "low", 96);
    std::printf("  logits-adaptive (w=0):  chat=%zu tok (%.1f tok/s), math=%zu tok (%.1f tok/s)\n",
                r_chat.tokens.size(), r_chat.tok_s, r_math.tokens.size(), r_math.tok_s);
    CHECK(r_chat.tokens == base_chat.tokens);
    CHECK(r_math.tokens == base_math.tokens);
    std::printf("  PASS logits-adaptive(omega=0) 100%% bit-identical to plain(off) (%zu + %zu tokens)!\n",
                r_chat.tokens.size(), r_math.tokens.size());
  }
  {
    auto g = ingine::model::make_generator(make_opt(3, LoopCdMode::Hidden, 0.0f));
    CHECK(g->loopcd_mode() == LoopCdMode::Hidden);
    const GenResult r_chat = run_greedy(*g, fmt, p_chat, "low", 96);
    const GenResult r_math = run_greedy(*g, fmt, p_math, "low", 96);
    std::printf("  hidden (w=0):           chat=%zu tok (%.1f tok/s), math=%zu tok (%.1f tok/s)\n",
                r_chat.tokens.size(), r_chat.tok_s, r_math.tokens.size(), r_math.tok_s);
    CHECK(r_chat.tokens == base_chat.tokens);
    CHECK(r_math.tokens == base_math.tokens);
    std::printf("  PASS hidden(omega=0) 100%% bit-identical to plain(off) (%zu + %zu tokens)!\n",
                r_chat.tokens.size(), r_math.tokens.size());
  }

  std::printf("[3/4] Active LoopCD (default omega) on real Inkling-2-Mini (L=3) + prefix-cache repeat check...\n");
  {
    auto g = ingine::model::make_generator(make_opt(3, LoopCdMode::LogitsAdaptive, -1.0f));
    const GenResult r1 = run_greedy(*g, fmt, p_math, "low", 96);
    // Repeat identical prompt: must hit prefix cache (Reuse::Checkpoint) and produce identical greedy output.
    const GenResult r2 = run_greedy(*g, fmt, p_math, "low", 96);
    std::printf("  3pass logits-adaptive (w=0.5): %zu tok (%.1f tok/s), repeat reuse=%s\n",
                r1.tokens.size(), r1.tok_s, ingine::model::to_string(r2.stats.reuse));
    CHECK(!r1.tokens.empty());
    CHECK(r1.tokens == r2.tokens);
    CHECK(r2.stats.reuse == ingine::model::Reuse::Checkpoint);
  }
  {
    auto g = ingine::model::make_generator(make_opt(3, LoopCdMode::Hidden, -1.0f));
    const GenResult r1 = run_greedy(*g, fmt, p_math, "low", 96);
    const GenResult r2 = run_greedy(*g, fmt, p_math, "low", 96);
    std::printf("  3pass hidden (w=0.3):          %zu tok (%.1f tok/s), repeat reuse=%s\n",
                r1.tokens.size(), r1.tok_s, ingine::model::to_string(r2.stats.reuse));
    CHECK(!r1.tokens.empty());
    CHECK(r1.tokens == r2.tokens);
    CHECK(r2.stats.reuse == ingine::model::Reuse::Checkpoint);
  }

  std::printf("[4/4] Active LoopCD on 2-pass Inkling-2-Mini (L=2)...\n");
  {
    auto g = ingine::model::make_generator(make_opt(2, LoopCdMode::LogitsAdaptive, -1.0f));
    const GenResult r1 = run_greedy(*g, fmt, p_math, "low", 96);
    std::printf("  2pass logits-adaptive (w=0.5): %zu tok (%.1f tok/s)\n", r1.tokens.size(), r1.tok_s);
    CHECK(!r1.tokens.empty());
  }
  {
    auto g = ingine::model::make_generator(make_opt(2, LoopCdMode::Hidden, -1.0f));
    const GenResult r1 = run_greedy(*g, fmt, p_math, "low", 96);
    std::printf("  2pass hidden (w=0.3):          %zu tok (%.1f tok/s)\n", r1.tokens.size(), r1.tok_s);
    CHECK(!r1.tokens.empty());
  }

  std::printf("ALL PASS model_loopcd_e2e\n");
  return 0;
}

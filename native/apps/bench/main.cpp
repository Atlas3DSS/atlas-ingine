// ingine-bench: end-to-end speed, prefill tokens/s and decode tokens/s for one request (port of
// engine/bench/bench_e2e.py; same flags, lines and JSON).
//
//     ingine-bench --tokens 512 --dense-fp8
//     ingine-bench --spec --forced agentic.jsonl   (teacher-forced replay, see forced.h)
//     ingine-bench --spec --forced agentic.jsonl --forced-free 1 --temperature 1   (free generation from each
//         item's prompt for len(output) tokens at the temperature: sampled A/B, where forced acceptance by
//         equality under-counts)
#include "forced.h"
#include "ingine/generator.h"
#include "serve/cli.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <utf8proc/utf8proc.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

const std::vector<std::pair<std::string, std::string>> kPrompts = {
    {"chat", "Explain how a hash map handles collisions, with a short Python example."},
    {"code", "Write a Python module implementing an LRU cache with O(1) get/put, type hints and unit tests."},
};

const std::vector<std::string> kBatchPrompts = {
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a Python module implementing an LRU cache with O(1) get/put, type hints and unit tests.",
    "Compare B-trees and LSM-trees for write-heavy database workloads in clear bullet points.",
    "Explain how TCP congestion control (AIMD and slow start) works with a concise numeric trace.",
    "Write a C++20 lock-free single-producer single-consumer ring buffer with memory ordering comments.",
    "Explain the difference between virtual memory paging and segmentation in modern operating systems.",
    "Summarize how attention with a sliding window and periodic global layers bounds KV cache growth.",
    "Write a Python function to solve the 0/1 knapsack problem with path reconstruction and tests.",
};

std::vector<int> parse_csv_ints(const std::string& s) {
  std::vector<int> out;
  std::size_t start = 0;
  while (start < s.size()) {
    const std::size_t comma = s.find(',', start);
    const std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!tok.empty()) out.push_back(std::stoi(tok));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return out;
}

// Python repr(float).
std::string py_float(double v) {
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
  char buf[64];
  auto [end, ec] = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  std::string s(buf, end);  // [-]d[.ddd]e[+-]XX, shortest round-trip digits
  std::string sign;
  if (s[0] == '-') {
    sign = "-";
    s.erase(0, 1);
  }
  const size_t e = s.find('e');
  const int exp = std::stoi(s.substr(e + 1));
  std::string digits = s.substr(0, e);
  digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
  if (exp < -4 || exp >= 16) {
    std::string mant = digits.substr(0, 1);
    if (digits.size() > 1) mant += "." + digits.substr(1);
    char ebuf[16];
    std::snprintf(ebuf, sizeof ebuf, "e%c%02d", exp < 0 ? '-' : '+', std::abs(exp));
    return sign + mant + ebuf;
  }
  std::string out;
  if (exp < 0) {
    out = "0." + std::string(size_t(-exp - 1), '0') + digits;
  } else if (size_t(exp) + 1 >= digits.size()) {
    out = digits + std::string(size_t(exp) + 1 - digits.size(), '0') + ".0";
  } else {
    out = digits.substr(0, size_t(exp) + 1) + "." + digits.substr(size_t(exp) + 1);
  }
  return sign + out;
}

// Python repr(str) of (possibly partial) UTF-8 text; invalid bytes decode to U+FFFD as in tokenizer.decode.
std::string py_repr(const std::string& text) {
  std::vector<int32_t> cps;
  const auto* p = reinterpret_cast<const utf8proc_uint8_t*>(text.data());
  utf8proc_ssize_t left = utf8proc_ssize_t(text.size());
  while (left > 0) {
    utf8proc_int32_t cp;
    utf8proc_ssize_t n = utf8proc_iterate(p, left, &cp);
    if (n <= 0) {
      cp = 0xFFFD;
      n = 1;
    }
    cps.push_back(cp);
    p += n;
    left -= n;
  }
  const bool has_single = std::find(cps.begin(), cps.end(), '\'') != cps.end();
  const bool has_double = std::find(cps.begin(), cps.end(), '"') != cps.end();
  const char quote = has_single && !has_double ? '"' : '\'';
  std::string out(1, quote);
  for (int32_t cp : cps) {
    if (cp == quote || cp == '\\') {
      out += '\\';
      out += char(cp);
    } else if (cp == '\n') {
      out += "\\n";
    } else if (cp == '\r') {
      out += "\\r";
    } else if (cp == '\t') {
      out += "\\t";
    } else {
      const utf8proc_category_t cat = utf8proc_category(cp);
      const bool printable =
          cp == ' ' || !(cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF || cat == UTF8PROC_CATEGORY_CS ||
                         cat == UTF8PROC_CATEGORY_CO || cat == UTF8PROC_CATEGORY_CN || cat == UTF8PROC_CATEGORY_ZL ||
                         cat == UTF8PROC_CATEGORY_ZP || cat == UTF8PROC_CATEGORY_ZS);
      if (printable) {
        utf8proc_uint8_t enc[4];
        const utf8proc_ssize_t n = utf8proc_encode_char(cp, enc);
        out.append(reinterpret_cast<const char*>(enc), size_t(n));
      } else {
        char buf[16];
        if (cp < 0x100) std::snprintf(buf, sizeof buf, "\\x%02x", cp);
        else if (cp < 0x10000) std::snprintf(buf, sizeof buf, "\\u%04x", cp);
        else std::snprintf(buf, sizeof buf, "\\U%08x", cp);
        out += buf;
      }
    }
  }
  out += quote;
  return out;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

std::string default_model_dir() {
  if (const char* env = std::getenv("INGINE_MODEL_DIR")) return env;
  return std::string(INGINE_REPO_DIR) + "/models/Inkling-2-Mini";
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir = default_model_dir(), backend = "fast", ablation, effort = "low", forced, forced_trace;
  std::string offload_mode = "auto", expert_ranking, dump_router_stats, kv_swap_dir, kv_ckpt_path, kv_ckpt_resume;
  std::string kv_cache_mode = "per_pass";
  std::string concurrency_csv;
  int max_ctx = 32768, prefill_chunk = 3072, tokens = 512, runs = 3, long_prompt = 0, forced_items = 0, forced_skip = 0,
      forced_free = -1;
  int max_batch = 0, max_kv_tokens = 0, park_mb = -1, park_host_mb = 0, loop_passes = 3;
  int offload_vram_mb = 0, offload_ram_mb = 12288;
  int kv_vram_mb = 0, kv_ram_mb = 12288, kv_ckpt_every = 0;
  bool dense_fp8 = false, kv_fp8 = false, kv_q4 = false, kv_paged = false, spec = false, vision = false, atlas_6g = false,
       ignore_eos = false, staggered = false, no_prefix_cache = false;
  double spec_threshold = 0.8, temperature = 0.0;

  ingine::serve::ArgParser ap("ingine-bench",
                              "End-to-end ingine speed: prefill tokens/s and greedy decode tokens/s for one or many concurrent requests.");
  ap.option("--model-dir", "", "", &model_dir);
  ap.option("--backend", "{fast}", "", &backend, {"fast"});
  ap.option("--ablation", "", "", &ablation);
  ap.option("--max-ctx", "", "", &max_ctx);
  ap.option("--prefill-chunk", "", "tokens per prefill chunk (default 3072; 512 in offload mode / 12 GB GPUs)",
            &prefill_chunk);
  ap.option("--tokens", "", "", &tokens);
  ap.option("--runs", "", "", &runs);
  ap.option("--effort", "", "", &effort);
  ap.flag("--dense-fp8", "", &dense_fp8);
  ap.flag("--vision", "also load the vision tower", &vision);
  ap.flag("--kv-fp8", "FP8 full-attention KV caches", &kv_fp8);
  ap.flag("--kv-q4", "Q4_0 4-bit full-attention KV caches (9 KiB/tok per_pass, 4.5 KiB/tok last_full; SWA stays BF16)", &kv_q4);
  ap.option("--kv-cache-mode", "{per_pass,last_full}",
            "full-attention KV mode: per_pass (default, 8 visit slots) or last_full (opt-in L11/L17 shared, 4 slots; B=1 non-spec)",
            &kv_cache_mode, {"per_pass", "last_full"});
  ap.flag("--kv-paged", "3-tier paged Q4_0 full-attention KV cache (VRAM LRU + pinned UVA RAM + NVMe SwapFile)", &kv_paged);
  ap.option("--kv-vram-mb", "", "VRAM cap in MiB for paged Q4_0 KV cache (implies --kv-paged --kv-q4)", &kv_vram_mb);
  ap.option("--kv-ram-mb", "", "pinned UVA host RAM cap in MiB for paged Q4_0 KV before NVMe SwapFile spill (default 12288)",
            &kv_ram_mb);
  ap.option("--kv-swap-dir", "", "optional directory for Tier-2 NVMe KV swap file", &kv_swap_dir);
  ap.option("--kv-ckpt-path", "", "atomic disk KV checkpoint file path (periodic & auto-resume on prefix match)",
            &kv_ckpt_path);
  ap.option("--kv-ckpt-every", "", "save atomic disk KV checkpoint every N tokens during long prefill (0 = off)",
            &kv_ckpt_every);
  ap.option("--kv-ckpt-resume", "", "explicit disk KV checkpoint file to resume from (fails loudly on mismatch)",
            &kv_ckpt_resume);
  ap.exclusive({"--kv-fp8", "--kv-q4"}, false);
  ap.flag("--spec", "DSpark speculative decoding", &spec);
  ap.flag("--atlas-6g", "6 GB VRAM preset (offload-vram-mb=5888, offload-ram-mb=12288, dense-fp8, kv-fp8, prefill-chunk=1024)",
          &atlas_6g);
  ap.option("--offload-vram-mb", "", "tiered MoE offloading VRAM cap in MiB (0 = off, e.g. 5888 for 6 GB GPU)",
            &offload_vram_mb);
  ap.option("--offload-ram-mb", "", "pinned host RAM cap in MiB for WARM experts (default 12288)", &offload_ram_mb);
  ap.option("--offload-mode", "{auto,dma-lru,uva}", "tiered offload transport mode", &offload_mode,
            {"auto", "dma-lru", "uva"});
  ap.option("--expert-ranking", "", "optional JSON with per-layer [24][128] homology/frequency scores", &expert_ranking);
  ap.option("--dump-router-stats", "", "write observed [24][128] router hits JSON on exit", &dump_router_stats);
  ap.option("--spec-threshold", "", "verify the leading drafts whose draft probability exceeds this (0 = always all 7)",
            &spec_threshold);
  ap.option("--park-mb", "", "VRAM cap (MiB) for Tier-0 parked checkpoints (-1 = auto, 0 = off)", &park_mb);
  ap.option("--park-host-mb", "", "pinned host RAM cap (MiB) for Tier-1 parked checkpoints (default 0 = off)",
            &park_host_mb);
  ap.flag("--no-prefix-cache", "disable all prefix/root/parked checkpoint reuse (cache-off parity baseline)",
          &no_prefix_cache);
  ap.option("--temperature", "", "", &temperature);
  ap.option("--long-prompt", "", "also time prefill of a synthetic prompt of N tokens", &long_prompt);
  ap.option("--forced", "", "teacher-forced replay of the outputs in FILE (JSON lines {name, prompt, output})",
            &forced);
  ap.option("--forced-items", "", "replay at most N items of --forced", &forced_items);
  ap.option("--forced-skip", "", "skip the first N items of --forced", &forced_skip);
  ap.option("--forced-trace", "", "append every replayed item and round to FILE (JSON lines)", &forced_trace);
  ap.option("--forced-free", "", "with --forced: sample freely from each item's prompt with this seed (no replay)", &forced_free);
  ap.option("--concurrency", "", "comma-separated concurrent request counts for continuous batching (e.g. 1,4,8,16,32,64)",
            &concurrency_csv);
  ap.option("--max-batch", "", "override max batch slots (default: max of --concurrency, or 1)", &max_batch);
  ap.option("--max-kv-tokens", "", "shared paged full-KV token pool capacity (0 = auto)", &max_kv_tokens);
  ap.option("--loop-passes", "", "recurrent loop passes over layers 6..17 (1..8, default 3 = 48 visits)", &loop_passes);
  ap.flag("--ignore-eos", "with --concurrency: decode full --tokens per request (steady-state peak throughput)", &ignore_eos);
  ap.flag("--staggered", "with --concurrency: also run a staggered mid-generation continuous-batching arrival pass", &staggered);
  ap.flag("--no-prefix-cache", "disable prefix checkpoint caching (cache-off parity baseline)", &no_prefix_cache);
  ap.parse(argc, argv);

  try {
    using Clock = std::chrono::steady_clock;
    const std::vector<int> concurrencies = parse_csv_ints(concurrency_csv);
    int resolved_max_batch = max_batch;
    if (!concurrencies.empty()) {
      for (int c : concurrencies) {
        if (c < 1) throw std::invalid_argument("--concurrency values must be >= 1");
        resolved_max_batch = std::max(resolved_max_batch, c);
      }
      if (resolved_max_batch == 1) resolved_max_batch = 1;
    } else if (resolved_max_batch < 1) {
      resolved_max_batch = 1;
    }

    cudaFree(nullptr);  // create the context before measuring the model's memory
    size_t free0 = 0, total = 0;
    cudaMemGetInfo(&free0, &total);
    const auto t0 = Clock::now();
    ingine::model::RuntimeOptions options;
    if (atlas_6g) {
      if (offload_vram_mb == 0) offload_vram_mb = 5888;
      dense_fp8 = true;
      if (kv_paged || kv_vram_mb > 0) kv_q4 = true;
      if (!kv_q4) kv_fp8 = true;
      if (prefill_chunk > 1024) prefill_chunk = 1024;
    }
    options.model_dir = model_dir;
    options.ablation = ablation;
    options.max_ctx = max_ctx;
    options.prefill_chunk = prefill_chunk;
    options.dense_fp8 = dense_fp8;
    options.kv_fp8 = kv_fp8;
    options.kv_q4 = kv_q4;
    options.kv_cache_mode = ingine::model::parse_kv_cache_mode(kv_cache_mode);
    options.kv_paged = kv_paged || kv_vram_mb > 0;
    options.kv_vram_mb = kv_vram_mb;
    options.kv_ram_mb = kv_ram_mb;
    options.kv_swap_dir = kv_swap_dir;
    options.kv_ckpt_path = kv_ckpt_path;
    options.kv_ckpt_every = kv_ckpt_every;
    options.kv_ckpt_resume = kv_ckpt_resume;
    options.spec = spec;
    options.spec_threshold = float(spec_threshold);
    options.park_mb = park_mb;
    options.park_host_mb = park_host_mb;
    options.prefix_cache = !no_prefix_cache;
    options.vision = vision;
    options.offload_vram_mb = offload_vram_mb;
    options.offload_ram_mb = offload_ram_mb;
    options.offload_mode = offload_mode;
    options.expert_ranking = expert_ranking;
    options.dump_router_stats = dump_router_stats;
    options.max_batch = resolved_max_batch;
    options.max_kv_tokens = max_kv_tokens;
    options.prefix_cache = !no_prefix_cache;
    options.loop_passes = loop_passes;
    auto gen = ingine::model::make_generator(options);
    const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
    size_t free1 = 0;
    cudaMemGetInfo(&free1, &total);
    const double vram_gb = double(free0 > free1 ? free0 - free1 : 0) / 1e9;
    std::printf("load+capture %.1fs, GPU %.2f GB (max_batch=%d, kv_cache_mode=%s, loop_passes=%d)\n", load_s, vram_gb,
                gen->max_batch_size(), ingine::model::to_string(gen->kv_cache_mode()), gen->loop_passes());
    std::fflush(stdout);
    if (!forced.empty()) {
      std::printf("%s\n", ingine::bench::run_forced(*gen, forced, temperature, forced_skip, forced_items, forced_trace, forced_free).c_str());
      return 0;
    }

    const auto fmt = ingine::text::ChatFormat::load(model_dir);
    const auto& tok = fmt.tokenizer();
    const std::vector<int32_t> stop = {*tok.token_id("<|content_model_end_sampling|>")};

    if (!concurrencies.empty()) {
      // Build distinct rendered prompts for up to resolved_max_batch concurrent sequences.
      std::vector<std::vector<int32_t>> prompt_bank;
      prompt_bank.reserve(std::size_t(resolved_max_batch));
      for (int i = 0; i < resolved_max_batch; ++i) {
        ingine::text::ChatOptions chat;
        chat.reasoning_effort = effort;
        const std::string text =
            "[Request #" + std::to_string(i + 1) + "] " + kBatchPrompts[std::size_t(i) % kBatchPrompts.size()];
        const nlohmann::ordered_json messages = nlohmann::ordered_json::array({{{"role", "user"}, {"content", text}}});
        prompt_bank.push_back(fmt.render(messages, nlohmann::ordered_json(), chat).ids);
      }

      nlohmann::ordered_json report;
      report["vram_gb"] = vram_gb;
      report["max_batch"] = gen->max_batch_size();
      report["spec"] = spec;
      report["batched_spec"] = gen->supports_batched_spec();
      report["prefix_cache"] = !no_prefix_cache;
      report["tokens_per_req"] = tokens;
      report["ignore_eos"] = ignore_eos;
      report["results"] = nlohmann::ordered_json::array();

      for (int C : concurrencies) {
        std::vector<double> agg_rates;
        double last_prefill_s = 0.0, last_decode_s = 0.0;
        int last_prompt_toks = 0, last_gen_toks = 0, last_spec_rounds = 0, last_spec_accepted = 0;
        for (int r = 0; r < runs; ++r) {
          if (C == 1 && gen->max_batch_size() == 1) {
            ingine::model::Sampling sampling;
            sampling.temperature = float(temperature);
            sampling.seed = uint64_t(r);
            const auto st = gen->generate(prompt_bank[0], tokens, sampling,
                                          ignore_eos ? std::span<const int32_t>{} : std::span<const int32_t>(stop),
                                          [](std::span<const int32_t>) { return true; });
            last_prompt_toks = st.prompt_tokens;
            last_gen_toks = st.generated_tokens;
            last_prefill_s = st.prefill_seconds;
            last_decode_s = st.decode_seconds;
            last_spec_rounds = st.spec_rounds;
            last_spec_accepted = st.spec_accepted;
            agg_rates.push_back(double(std::max(1, st.generated_tokens - 1)) / std::max(st.decode_seconds, 1e-9));
            continue;
          }
          struct Active {
            int slot = -1;
            int32_t last_tok = 0;
            int generated = 0;
          };
          std::vector<Active> live;
          live.reserve(std::size_t(C));
          int total_prompt = 0;
          int total_gen = 0;
          int total_spec_rounds = 0;
          int total_spec_accepted = 0;
          const auto tp0 = Clock::now();
          for (int i = 0; i < C; ++i) {
            ingine::model::Sampling sampling;
            sampling.temperature = float(temperature);
            sampling.seed = uint64_t(r * 1000 + i);
            int32_t first_tok = 0;
            ingine::model::GenerateStats st;
            const int s = gen->start_slot(ingine::model::Prompt{prompt_bank[std::size_t(i)], {}}, tokens, sampling,
                                          first_tok, st);
            if (s < 0) throw std::runtime_error("start_slot returned -1 for concurrency " + std::to_string(C));
            total_prompt += st.prompt_tokens;
            ++total_gen;
            const bool done = (tokens <= 1) || (!ignore_eos && std::find(stop.begin(), stop.end(), first_tok) != stop.end());
            if (done) {
              gen->release_slot(s);
            } else {
              live.push_back(Active{s, first_tok, 1});
            }
          }
          const double prefill_s = std::chrono::duration<double>(Clock::now() - tp0).count();

          const auto td0 = Clock::now();
          int decode_tokens = 0;
          std::vector<ingine::model::BatchSlotStep> steps;
          std::vector<ingine::model::BatchSlotResult> results;
          while (!live.empty()) {
            steps.resize(live.size());
            results.resize(live.size());
            for (std::size_t i = 0; i < live.size(); ++i) steps[i] = {live[i].slot, live[i].last_tok};
            gen->step_slots_multi(steps, results);
            std::vector<Active> kept;
            kept.reserve(live.size());
            for (std::size_t i = 0; i < live.size(); ++i) {
              const auto& res = results[i];
              total_spec_rounds += res.spec_rounds;
              total_spec_accepted += res.spec_accepted;
              bool done = false;
              for (int k = 0; k < res.count; ++k) {
                const int32_t tok_k = res.tokens[k];
                live[i].last_tok = tok_k;
                ++live[i].generated;
                ++total_gen;
                ++decode_tokens;
                if (live[i].generated >= tokens ||
                    (!ignore_eos && std::find(stop.begin(), stop.end(), tok_k) != stop.end())) {
                  done = true;
                  break;
                }
              }
              if (done || res.count == 0) gen->release_slot(live[i].slot);
              else kept.push_back(live[i]);
            }
            live = std::move(kept);
          }
          const double decode_s = std::chrono::duration<double>(Clock::now() - td0).count();
          last_prompt_toks = total_prompt;
          last_gen_toks = total_gen;
          last_prefill_s = prefill_s;
          last_decode_s = decode_s;
          last_spec_rounds = total_spec_rounds;
          last_spec_accepted = total_spec_accepted;
          agg_rates.push_back(double(decode_tokens) / std::max(decode_s, 1e-9));
        }

        const double med_agg = median(agg_rates);
        const double per_req = med_agg / double(C);
        const double prefill_tok_s = double(last_prompt_toks) / std::max(last_prefill_s, 1e-9);
        const double mean_toks_per_round =
            last_spec_rounds > 0 ? double(last_spec_accepted + last_spec_rounds) / double(last_spec_rounds) : 1.0;
        std::printf("B=%-2d: aggregate decode %7.1f tok/s (%6.1f tok/s/req) | prefill %6.0f tok/s (%d prompt, %d gen, %.2fs decode%s)\n",
                    C, med_agg, per_req, prefill_tok_s, last_prompt_toks, last_gen_toks, last_decode_s,
                    last_spec_rounds > 0
                        ? (", " + std::to_string(mean_toks_per_round).substr(0, 4) + " tok/round").c_str()
                        : "");
        std::fflush(stdout);
        report["results"].push_back({{"concurrency", C},
                                     {"aggregate_decode_tok_s", med_agg},
                                     {"per_req_decode_tok_s", per_req},
                                     {"prefill_tok_s", prefill_tok_s},
                                     {"prompt_tokens", last_prompt_toks},
                                     {"generated_tokens", last_gen_toks},
                                     {"prefill_seconds", last_prefill_s},
                                     {"decode_seconds", last_decode_s},
                                     {"spec_rounds", last_spec_rounds},
                                     {"spec_accepted", last_spec_accepted},
                                     {"mean_tokens_per_round", mean_toks_per_round},
                                     {"runs", agg_rates}});
      }

      if (staggered && gen->max_batch_size() > 1) {
        // Staggered continuous-batching pass: start C/2 requests, inject C/2 more after tokens/4 steps.
        const int C = resolved_max_batch;
        const int half = std::max(1, C / 2);
        const int join_step = std::max(1, tokens / 4);
        struct Active {
          int slot = -1;
          int32_t last_tok = 0;
          int generated = 0;
        };
        std::vector<Active> live;
        int next_idx = 0, total_gen = 0, step_n = 0, total_spec_rounds = 0, total_spec_accepted = 0;
        auto admit = [&](int count) {
          for (int k = 0; k < count && next_idx < C; ++k, ++next_idx) {
            ingine::model::Sampling s;
            s.temperature = float(temperature);
            s.seed = uint64_t(777 + next_idx);
            int32_t tok0 = 0;
            ingine::model::GenerateStats st;
            const int slot = gen->start_slot(ingine::model::Prompt{prompt_bank[std::size_t(next_idx)], {}}, tokens, s, tok0, st);
            if (slot >= 0) {
              ++total_gen;
              live.push_back(Active{slot, tok0, 1});
            }
          }
        };
        const auto ts0 = Clock::now();
        admit(half);
        std::vector<ingine::model::BatchSlotStep> steps;
        std::vector<ingine::model::BatchSlotResult> results;
        while (!live.empty() || next_idx < C) {
          if (step_n == join_step && next_idx < C) admit(C - next_idx);
          if (live.empty()) {
            admit(C - next_idx);
            continue;
          }
          steps.resize(live.size());
          results.resize(live.size());
          for (std::size_t i = 0; i < live.size(); ++i) steps[i] = {live[i].slot, live[i].last_tok};
          gen->step_slots_multi(steps, results);
          ++step_n;
          std::vector<Active> kept;
          for (std::size_t i = 0; i < live.size(); ++i) {
            const auto& res = results[i];
            total_spec_rounds += res.spec_rounds;
            total_spec_accepted += res.spec_accepted;
            bool done = false;
            for (int k = 0; k < res.count; ++k) {
              const int32_t tok_k = res.tokens[k];
              live[i].last_tok = tok_k;
              ++live[i].generated;
              ++total_gen;
              if (live[i].generated >= tokens ||
                  (!ignore_eos && std::find(stop.begin(), stop.end(), tok_k) != stop.end())) {
                done = true;
                break;
              }
            }
            if (done || res.count == 0) gen->release_slot(live[i].slot);
            else kept.push_back(live[i]);
          }
          live = std::move(kept);
        }
        const double wall_s = std::chrono::duration<double>(Clock::now() - ts0).count();
        const double eff_tok_s = double(total_gen) / std::max(wall_s, 1e-9);
        std::printf("staggered (C=%d, join@%d): %d tokens in %.2fs = %.1f tok/s effective (spec_rounds=%d, spec_accepted=%d)\n",
                    C, join_step, total_gen, wall_s, eff_tok_s, total_spec_rounds, total_spec_accepted);
        report["staggered"] = {{"concurrency", C},
                               {"join_step", join_step},
                               {"generated_tokens", total_gen},
                               {"wall_seconds", wall_s},
                               {"effective_tok_s", eff_tok_s},
                               {"spec_rounds", total_spec_rounds},
                               {"spec_accepted", total_spec_accepted}};
      }

      std::printf("%s\n", report.dump().c_str());
      return 0;
    }

    std::string json = "{";
    for (const auto& [name, prompt] : kPrompts) {
      ingine::text::ChatOptions chat;
      chat.reasoning_effort = effort;
      const nlohmann::ordered_json messages = nlohmann::ordered_json::array({{{"role", "user"}, {"content", prompt}}});
      const std::vector<int32_t> ids = fmt.render(messages, nlohmann::ordered_json(), chat).ids;
      std::vector<double> rates;
      ingine::model::GenerateStats st;
      for (int r = 0; r < runs; ++r) {
        std::vector<int32_t> out;
        ingine::model::Sampling sampling;
        sampling.temperature = float(temperature);
        sampling.seed = uint64_t(r);
        // every run stops at the end-of-turn token: tokens after it are not what a request generates (and
        // speculative decoding accepts few drafts there, which made later runs look slow)
        st = gen->generate(ids, tokens, sampling, std::span<const int32_t>(stop),
                           [&](std::span<const int32_t> t) {
                             out.insert(out.end(), t.begin(), t.end());
                             return true;
                           });
        rates.push_back((st.generated_tokens - 1) / st.decode_seconds);
        if (r == 0) {
          out.resize(std::min<size_t>(out.size(), 120));
          std::printf("--- %s: %s\n", name.c_str(), py_repr(tok.decode_lossy(out)).c_str());
          std::fflush(stdout);
        }
      }
      const double med = median(rates);
      std::string runs_json, runs_text;
      for (double x : rates) {
        runs_json += (runs_json.empty() ? "" : ", ") + py_float(x);
        char buf[64];
        std::snprintf(buf, sizeof buf, "%s%.1f", runs_text.empty() ? "" : ", ", x);
        runs_text += buf;
      }
      const std::string spec_json =
          spec ? "{\"rounds\": " + std::to_string(st.spec_rounds) + ", \"accepted\": " + std::to_string(st.spec_accepted) + "}"
               : "{}";
      if (json.size() > 1) json += ", ";
      json += "\"" + name + "\": {\"decode_tok_s\": " + py_float(med) + ", \"runs\": [" + runs_json +
              "], \"spec\": " + spec_json + "}";
      std::string spec_text;
      if (spec && st.spec_rounds) {
        char buf[96];
        std::snprintf(buf, sizeof buf, ", %.2f tokens/verify (last run)",
                      double(st.spec_accepted + st.spec_rounds) / st.spec_rounds);
        spec_text = buf;
      }
      std::printf("%s: decode %.1f tok/s (runs %s)%s\n", name.c_str(), med, runs_text.c_str(), spec_text.c_str());
      std::fflush(stdout);
    }
    if (long_prompt) {
      std::ifstream in(__FILE__, std::ios::binary);
      std::stringstream source;
      source << in.rdbuf();
      const std::vector<int32_t> once = tok.encode(source.str(), /*special=*/true);
      std::vector<int32_t> ids;
      ids.reserve(size_t(long_prompt) + once.size());
      while (!once.empty() && int(ids.size()) < long_prompt) ids.insert(ids.end(), once.begin(), once.end());
      ids.resize(std::min<size_t>(ids.size(), size_t(long_prompt)));
      ingine::model::Sampling greedy;
      greedy.temperature = 0.0f;
      const ingine::model::GenerateStats st =
          gen->generate(ids, 1, greedy, {}, [](std::span<const int32_t>) { return true; });
      const double tok_s = double(ids.size()) / st.prefill_seconds;
      json += ", \"prefill\": {\"tokens\": " + std::to_string(ids.size()) + ", \"tok_s\": " + py_float(tok_s) + "}";
      std::printf("prefill %zu tokens: %.0f tok/s\n", ids.size(), tok_s);
      std::fflush(stdout);
    }
    if (const std::string osum = gen->offload_summary(); !osum.empty()) {
      std::printf("%s\n", osum.c_str());
    }
    std::printf("%s}\n", json.c_str());
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ingine-bench: %s\n", e.what());
    return 1;
  }
}

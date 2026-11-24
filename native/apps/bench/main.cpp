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
  int max_ctx = 32768, prefill_chunk = 3072, tokens = 512, runs = 3, long_prompt = 0, forced_items = 0, forced_skip = 0,
      forced_free = -1;
  int offload_vram_mb = 0, offload_ram_mb = 12288;
  int kv_vram_mb = 0, kv_ram_mb = 12288, kv_ckpt_every = 0;
  bool dense_fp8 = false, kv_fp8 = false, kv_q4 = false, kv_paged = false, spec = false, vision = false, atlas_6g = false;
  double spec_threshold = 0.8, temperature = 0.0;

  ingine::serve::ArgParser ap("ingine-bench",
                              "End-to-end ingine speed: prefill tokens/s and greedy decode tokens/s for one request.");
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
  ap.flag("--kv-q4", "Q4_0 4-bit full-attention KV caches (9 KiB/tok; SWA stays BF16)", &kv_q4);
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
  ap.flag("--atlas-6g", "6 GB VRAM preset (offload-vram-mb=5888, offload-ram-mb=12288, dense-fp8, kv-fp8, prefill-chunk=512)",
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
  ap.option("--temperature", "", "", &temperature);
  ap.option("--long-prompt", "", "also time prefill of a synthetic prompt of N tokens", &long_prompt);
  ap.option("--forced", "", "teacher-forced replay of the outputs in FILE (JSON lines {name, prompt, output})",
            &forced);
  ap.option("--forced-items", "", "replay at most N items of --forced", &forced_items);
  ap.option("--forced-skip", "", "skip the first N items of --forced", &forced_skip);
  ap.option("--forced-trace", "", "append every replayed item and round to FILE (JSON lines)", &forced_trace);
  ap.option("--forced-free", "", "with --forced: sample freely from each item's prompt with this seed (no replay)", &forced_free);
  ap.parse(argc, argv);

  try {
    using Clock = std::chrono::steady_clock;
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
      if (prefill_chunk > 512) prefill_chunk = 512;
    }
    options.model_dir = model_dir;
    options.ablation = ablation;
    options.max_ctx = max_ctx;
    options.prefill_chunk = prefill_chunk;
    options.dense_fp8 = dense_fp8;
    options.kv_fp8 = kv_fp8;
    options.kv_q4 = kv_q4;
    options.kv_paged = kv_paged || kv_vram_mb > 0;
    options.kv_vram_mb = kv_vram_mb;
    options.kv_ram_mb = kv_ram_mb;
    options.kv_swap_dir = kv_swap_dir;
    options.kv_ckpt_path = kv_ckpt_path;
    options.kv_ckpt_every = kv_ckpt_every;
    options.kv_ckpt_resume = kv_ckpt_resume;
    options.spec = spec;
    options.spec_threshold = float(spec_threshold);
    options.vision = vision;
    options.offload_vram_mb = offload_vram_mb;
    options.offload_ram_mb = offload_ram_mb;
    options.offload_mode = offload_mode;
    options.expert_ranking = expert_ranking;
    options.dump_router_stats = dump_router_stats;
    auto gen = ingine::model::make_generator(options);
    const double load_s = std::chrono::duration<double>(Clock::now() - t0).count();
    size_t free1 = 0;
    cudaMemGetInfo(&free1, &total);
    std::printf("load+capture %.1fs, GPU %.2f GB\n", load_s, double(free0 > free1 ? free0 - free1 : 0) / 1e9);
    std::fflush(stdout);
    if (!forced.empty()) {
      std::printf("%s\n", ingine::bench::run_forced(*gen, forced, temperature, forced_skip, forced_items, forced_trace, forced_free).c_str());
      return 0;
    }

    const auto fmt = ingine::text::ChatFormat::load(model_dir);
    const auto& tok = fmt.tokenizer();
    const std::vector<int32_t> stop = {*tok.token_id("<|content_model_end_sampling|>")};

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

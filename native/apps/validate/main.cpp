// ingine-validate: parity of the C++ model against the Python engine's goldens (tools/goldens_model.py).
//
// Default ("pieces"): for every golden case, runs the C++ Prefiller over the prompt (chunked as the goldens
// were) and the width-1 decode graph for the golden greedy tokens (teacher forced), and reports per stage
// the KL divergence KL(python || c++) over the golden top-256 (+ the remaining mass as one bucket), top-1
// agreement and the largest logit difference among the golden top-256. --generate instead runs the full
// Generator greedily and compares its tokens with the golden greedy continuation; --generate --spec does the
// same with DSpark speculative decoding (greedy speculation must reproduce plain greedy decoding).
// Exit status 0 when the mean KL over all stages is within --max-mean-kl, every stage within --max-kl and top-1
// agreement is at least --min-top1, else 1. The defaults sit at the Python reference's own reproducibility: on
// the long golden prompts, Python against itself (prefill chunk 2048 vs 1536, or flex TF32 vs exact FP32
// attention) already differs by KL 6e-3 .. 8e-2 per stage with top-1 flips at near ties, while a real bug gives
// KL >= 1 on most stages. --generate needs every selected case's greedy tokens to match exactly, so it is meant
// for prompts without near ties (the CTest entries pick the BF16-cache chat and 300-token cases).
#include "ingine/generator.h"

#include "core/arena.h"
#include "core/device.h"
#include "model/prefill.h"
#include "model/runtime.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using ingine::DeviceBuffer;
using namespace ingine::model;

namespace {

struct Args {
  fs::path model_dir = "/mnt/data/inkling/hf";
  fs::path ablation = fs::path(INGINE_REPO_DIR) / "ablation" / "inkling-2-mini-abliterated.safetensors";
  fs::path golden = fs::path(INGINE_REPO_DIR) / "native" / "tests" / "golden" / "model";
  std::string filter;
  bool generate = false;
  bool spec = false;  // --generate with DSpark speculative decoding: greedy must reproduce the goldens
  double max_kl = 0.25;
  double max_mean_kl = 0.01;
  double min_top1 = 0.95;
};

void usage() {
  std::printf(
      "usage: ingine-validate [--model-dir DIR] [--ablation FILE] [--golden DIR] [--cases SUBSTR] [--generate [--spec]]\n"
      "                       [--max-kl X] [--max-mean-kl X] [--min-top1 F]\n"
      "--cases takes comma-separated name substrings\n");
}

struct Stage {
  double kl = 0, max_diff = 0;
  bool top1 = false;
};

// KL(p || q) with p from the golden top-256 + logsumexp and q from the full C++ row; the mass outside the
// golden top-256 is compared as one bucket.
Stage compare(const nlohmann::json& g, const std::vector<float>& c) {
  double mx = -INFINITY;
  int arg = 0;
  for (std::size_t i = 0; i < c.size(); ++i)
    if (c[i] > mx) mx = c[i], arg = int(i);
  double s = 0;
  for (float v : c) s += std::exp(double(v) - mx);
  const double lse_c = mx + std::log(s);
  const double lse_g = g.at("logsumexp").get<double>();
  const auto& ids = g.at("top_ids");
  const auto& vals = g.at("top_logits");
  Stage st;
  double pin = 0, qin = 0;
  for (std::size_t k = 0; k < ids.size(); ++k) {
    const int id = ids[k].get<int>();
    const double lp = vals[k].get<double>() - lse_g, lq = double(c[std::size_t(id)]) - lse_c;
    const double p = std::exp(lp);
    st.kl += p * (lp - lq);
    pin += p;
    qin += std::exp(lq);
    st.max_diff = std::max(st.max_diff, std::fabs(vals[k].get<double>() - double(c[std::size_t(id)])));
  }
  const double pr = std::max(0.0, 1.0 - pin), qr = std::max(1e-30, 1.0 - qin);
  if (pr > 0) st.kl += pr * std::log(pr / qr);
  st.top1 = arg == ids[0].get<int>();
  return st;
}

nlohmann::json load_json(const fs::path& p) {
  std::ifstream in(p);
  if (!in) throw std::runtime_error("cannot open " + p.string());
  return nlohmann::json::parse(in);
}

struct Totals {
  int stages = 0, top1 = 0;
  double max_kl = 0, sum_kl = 0;
  void add(const Stage& s) {
    ++stages;
    top1 += s.top1;
    sum_kl += s.kl;
    max_kl = std::max(max_kl, s.kl);
  }
};

void run_pieces(const RuntimeOptions& opt, const std::vector<nlohmann::json>& cases, Totals& tot) {
  Runtime rt(opt);
  Prefiller prefiller(rt.weights(), rt.ablation(), opt.prefill_chunk, rt.stream());
  DeviceBuffer tokens(std::size_t(opt.prefill_chunk) * 4), logits(std::size_t(UNPADDED_VOCAB) * 4);
  std::vector<float> host(UNPADDED_VOCAB);
  auto fetch = [&](const float* d) {
    CUDA_CHECK(cudaMemcpyAsync(host.data(), d, host.size() * 4, cudaMemcpyDeviceToHost, rt.stream()));
    CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
  };
  for (const auto& c : cases) {
    const std::string name = c.at("name");
    const std::vector<std::int32_t> ids = c.at("prompt_ids").get<std::vector<std::int32_t>>();
    reset_state(rt.state(), rt.stream());
    for (int s = 0; s < int(ids.size()); s += opt.prefill_chunk) {
      const int T = std::min(opt.prefill_chunk, int(ids.size()) - s);
      CUDA_CHECK(cudaMemcpyAsync(tokens.p, ids.data() + s, std::size_t(T) * 4, cudaMemcpyHostToDevice, rt.stream()));
      prefiller.run(rt.state(), static_cast<const std::int32_t*>(tokens.p), T, s, static_cast<float*>(logits.p),
                    nullptr);
    }
    fetch(static_cast<const float*>(logits.p));
    const Stage pre = compare(c.at("prefill"), host);
    tot.add(pre);
    std::printf("%-22s prefill (%5zu tokens): KL %.3e  top1 %s  max|dlogit| %.4f\n", name.c_str(), ids.size(), pre.kl,
                pre.top1 ? "ok" : "MISS", pre.max_diff);
    double sum_kl = 0, max_kl = 0, max_diff = 0;
    int agree = 0, steps = 0;
    for (const auto& step : c.at("decode")) {
      const float* d = rt.decode_logits(step.at("input_token").get<std::int32_t>(), step.at("pos").get<int>());
      fetch(d);
      const Stage s = compare(step, host);
      tot.add(s);
      sum_kl += s.kl;
      max_kl = std::max(max_kl, s.kl);
      max_diff = std::max(max_diff, s.max_diff);
      agree += s.top1;
      ++steps;
    }
    std::printf("%-22s decode  (%2d steps):     KL mean %.3e max %.3e  top1 %d/%d  max|dlogit| %.4f\n", name.c_str(),
                steps, steps ? sum_kl / steps : 0.0, max_kl, agree, steps, max_diff);
  }
}

void run_generate(const RuntimeOptions& opt, const std::vector<nlohmann::json>& cases, Totals& tot) {
  auto gen = make_generator(opt);
  for (const auto& c : cases) {
    const std::string name = c.at("name");
    const std::vector<std::int32_t> ids = c.at("prompt_ids").get<std::vector<std::int32_t>>();
    const std::vector<std::int32_t> want = c.at("greedy_tokens").get<std::vector<std::int32_t>>();
    std::vector<std::int32_t> got;
    Sampling greedy;
    greedy.temperature = 0.f;
    const GenerateStats st = gen->generate(ids, int(want.size()), greedy, {}, [&](std::span<const std::int32_t> t) {
      got.insert(got.end(), t.begin(), t.end());
      return true;
    });
    std::size_t match = 0;
    while (match < want.size() && match < got.size() && want[match] == got[match]) ++match;
    Stage s;
    s.top1 = match == want.size();
    tot.add(s);
    std::printf("%-22s generate: %zu/%zu greedy tokens match  (prefill %.3f s, decode %.3f s, %d tokens)\n",
                name.c_str(), match, want.size(), st.prefill_seconds, st.decode_seconds, st.generated_tokens);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto val = [&]() -> std::string {
      if (i + 1 >= argc) {
        usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--model-dir") a.model_dir = val();
    else if (k == "--ablation") a.ablation = val();
    else if (k == "--golden") a.golden = val();
    else if (k == "--cases") a.filter = val();
    else if (k == "--generate") a.generate = true;
    else if (k == "--spec") a.spec = true;
    else if (k == "--max-kl") a.max_kl = std::stod(val());
    else if (k == "--max-mean-kl") a.max_mean_kl = std::stod(val());
    else if (k == "--min-top1") a.min_top1 = std::stod(val());
    else {
      usage();
      return k == "--help" || k == "-h" ? 0 : 2;
    }
  }
  try {
    const nlohmann::json manifest = load_json(a.golden / "manifest.json");
    // one runtime per (ablated, kv_fp8) configuration, loaded one at a time
    std::map<std::pair<bool, bool>, std::vector<nlohmann::json>> groups;
    for (const auto& c : manifest.at("cases")) {
      const std::string name = c.at("name");
      if (!a.filter.empty()) {
        bool hit = false;
        for (std::size_t b = 0; b <= a.filter.size();) {
          const std::size_t e = std::min(a.filter.find(',', b), a.filter.size());
          hit |= e > b && name.find(a.filter.substr(b, e - b)) != std::string::npos;
          b = e + 1;
        }
        if (!hit) continue;
      }
      groups[{c.at("ablated").get<bool>(), c.at("kv_fp8").get<bool>()}].push_back(
          load_json(a.golden / c.at("file").get<std::string>()));
    }
    if (groups.empty()) {
      std::printf("no golden cases selected\n");
      return 1;
    }
    Totals tot;
    for (const auto& [key, cases] : groups) {
      RuntimeOptions opt;
      opt.model_dir = a.model_dir;
      opt.ablation = key.first ? a.ablation : fs::path();
      opt.kv_fp8 = key.second;
      opt.dense_fp8 = manifest.value("dense_fp8", true);
      opt.max_ctx = manifest.value("max_ctx", 8192);
      opt.prefill_chunk = manifest.value("prefill_chunk", 2048);
      opt.spec = a.generate && a.spec;
      std::printf("== %s model, %s full-attention KV\n", key.first ? "abliterated" : "original",
                  key.second ? "FP8" : "BF16");
      if (a.generate) run_generate(opt, cases, tot);
      else run_pieces(opt, cases, tot);
    }
    const double top1 = tot.stages ? double(tot.top1) / tot.stages : 0.0;
    const double mean_kl = tot.stages ? tot.sum_kl / tot.stages : 0.0;
    const bool ok = (a.generate || (tot.max_kl <= a.max_kl && mean_kl <= a.max_mean_kl)) && top1 >= a.min_top1;
    std::printf("%s: %d stages, top1 %.3f, KL mean %.3e max %.3e (limits: KL mean %.3e max %.3e, top1 %.3f)\n",
                ok ? "PASS" : "FAIL", tot.stages, top1, mean_kl, tot.max_kl, a.max_mean_kl, a.max_kl, a.min_top1);
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ingine-validate: %s\n", e.what());
    return 1;
  }
}

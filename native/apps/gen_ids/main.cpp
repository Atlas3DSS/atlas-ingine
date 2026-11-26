// ingine-gen-ids: greedy continuations of raw prompt token ids through the real generator (engine parity tool).
// stdin: JSON lines {"name": str, "ids": [int]}; stdout: JSON lines {"name": str, "out": [int], "decode_s": float}.
// Items run `--concurrency` at a time from separate threads, so a batching generator decodes them together.
#include "ingine/generator.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
  ingine::model::RuntimeOptions o;
  int tokens = 128, concurrency = 4;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--model-dir") o.model_dir = next();
    else if (a == "--max-ctx") o.max_ctx = std::stoi(next());
    else if (a == "--max-batch") o.max_batch = std::stoi(next());
    else if (a == "--tokens") tokens = std::stoi(next());
    else if (a == "--concurrency") concurrency = std::stoi(next());
    else if (a == "--kv-fp8") o.kv_fp8 = true;
    else if (a == "--kv-fp4v") o.kv_fp8 = o.kv_fp4v = true;
    else if (a == "--prefill-chunk") o.prefill_chunk = std::stoi(next());
    else if (a == "--spec") o.spec = true;
    else if (a == "--spec-threshold") o.spec_threshold = std::stof(next());
    else if (a == "--draft-ctx") o.draft_ctx = std::stoi(next());
    else if (a == "--spec-max-m") o.spec_max_m = std::stoi(next());
    else if (a == "--spec-shortlist") o.spec_shortlist = std::stoi(next());
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  o.dense_fp8 = false;
  o.vision = false;
  o.prefix_cache = false;
  auto gen = ingine::model::make_generator(o);
  struct Item { std::string name; std::vector<int32_t> ids, out; double dec = 0; };
  std::vector<Item> items;
  for (std::string line; std::getline(std::cin, line);) {
    if (line.empty()) continue;
    const auto j = nlohmann::json::parse(line);
    items.push_back({j.at("name").get<std::string>(), j.at("ids").get<std::vector<int32_t>>(), {}, 0});
  }
  if (concurrency > 1 && gen->max_batch_size() > 1) {
    // True multi-slot continuous-batching decode (tests batched verify + speculation across `concurrency` slots).
    const int C = std::min(concurrency, gen->max_batch_size());
    struct Active {
      size_t idx = 0;
      int slot = -1;
      int32_t last_tok = 0;
      std::chrono::steady_clock::time_point t0{};
    };
    std::vector<Active> live;
    size_t next_item = 0;
    auto admit = [&]() {
      while (int(live.size()) < C && next_item < items.size()) {
        const size_t idx = next_item++;
        Item& it = items[idx];
        ingine::model::Sampling s;
        s.temperature = 0.0f;
        int32_t tok0 = 0;
        ingine::model::GenerateStats st;
        const int slot = gen->start_slot(ingine::model::Prompt{it.ids, {}}, tokens, s, tok0, st);
        if (slot < 0) break;
        it.out.push_back(tok0);
        if (int(it.out.size()) >= tokens) {
          gen->release_slot(slot);
        } else {
          live.push_back(Active{idx, slot, tok0, std::chrono::steady_clock::now()});
        }
      }
    };
    gen->reset_spec_metrics();
    admit();
    std::vector<ingine::model::BatchSlotStep> steps;
    std::vector<ingine::model::BatchSlotOutput> outs;
    while (!live.empty()) {
      steps.resize(live.size());
      outs.resize(live.size());
      for (size_t i = 0; i < live.size(); ++i) steps[i] = {live[i].slot, live[i].last_tok};
      gen->step_slots_multi(steps, outs);
      std::vector<Active> kept;
      kept.reserve(live.size());
      for (size_t i = 0; i < live.size(); ++i) {
        Item& it = items[live[i].idx];
        bool done = false;
        for (int k = 0; k < outs[i].count; ++k) {
          const int32_t tok = outs[i].tokens[k];
          it.out.push_back(tok);
          live[i].last_tok = tok;
          if (int(it.out.size()) >= tokens) {
            done = true;
            break;
          }
        }
        if (done) {
          it.dec = std::chrono::duration<double>(std::chrono::steady_clock::now() - live[i].t0).count();
          gen->release_slot(live[i].slot);
        } else {
          kept.push_back(live[i]);
        }
      }
      live = std::move(kept);
      admit();
    }
    if (const std::string sm = gen->spec_metrics_summary(); !sm.empty()) {
      std::fprintf(stderr, "ingine-gen-ids spec: %s\n", sm.c_str());
    }
  } else {
    std::atomic<size_t> next{0};
    std::vector<std::thread> th;
    for (int t = 0; t < concurrency; ++t)
      th.emplace_back([&] {
        for (size_t i; (i = next.fetch_add(1)) < items.size();) {
          ingine::model::Sampling s;
          s.temperature = 0.0f;
          Item& it = items[i];
          const auto st = gen->generate(it.ids, tokens, s, {}, [&](std::span<const int32_t> toks) {
            it.out.insert(it.out.end(), toks.begin(), toks.end());
            return true;
          });
          it.dec = st.decode_seconds;
        }
      });
    for (auto& t : th) t.join();
  }
  for (const auto& it : items)
    std::cout << nlohmann::json{{"name", it.name}, {"out", it.out}, {"decode_s", it.dec}}.dump() << "\n";
  return 0;
}

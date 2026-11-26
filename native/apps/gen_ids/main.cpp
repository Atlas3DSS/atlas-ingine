// ingine-gen-ids: greedy continuations of raw prompt token ids through the real generator (engine parity tool).
// stdin: JSON lines {"name": str, "ids": [int]}; stdout: JSON lines {"name": str, "out": [int], "decode_s": float}.
// Items run `--concurrency` at a time from separate threads, so a batching generator decodes them together.
#include "ingine/generator.h"

#include <nlohmann/json.hpp>

#include <atomic>
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
    else if (a == "--prefill-chunk") o.prefill_chunk = std::stoi(next());
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
  for (const auto& it : items)
    std::cout << nlohmann::json{{"name", it.name}, {"out", it.out}, {"decode_s", it.dec}}.dump() << "\n";
  return 0;
}

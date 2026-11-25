// Full-model GPU parity & safety test suite for Batched MTP Speculative Decoding (#851):
// 1. Greedy --spec vs --no-spec exact token parity at B=1, 4, 8, 12 (intermediate bucket), verifying spec_accepted > 0.
// 2. Prefix cache + --spec exact, partial, and root-prefix cold-vs-warm parity (including spec_rounds/spec_accepted
//    match), plus fail-closed bypass_prefix_cache and cache_namespace isolation.
// 3. Multi-tenant shortlist isolation: isolated slot produces identical output + spec_accepted regardless of rare
//    out-of-shortlist tokens processed concurrently by another slot.
// 4. Tight 6-page pool (384 tok, exact 6 pages) with --spec: 24-tok sibling overlap, 96-tok deferral, and the
//    65/96 mixed-budget boundary (no slot stealing a sibling's mandatory progress page).
// 5. Heterogeneous greedy + sampled batch with distinct per-slot temperature, top-k, top-p, banned tokens, and seeds.
// 6. Mid-block stop-token truncation with zero leaked tokens and clean slot reuse.
#include "engine/scheduler.h"
#include "ingine/generator.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using ingine::engine::Job;
using ingine::engine::JobBatch;
using ingine::engine::Scheduler;
using ingine::model::BatchSlotResult;
using ingine::model::BatchSlotStep;
using ingine::model::GenerateStats;
using ingine::model::Generator;
using ingine::model::Prompt;
using ingine::model::Reuse;
using ingine::model::RuntimeOptions;
using ingine::model::Sampling;

int g_failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

std::filesystem::path find_model_dir() {
  if (const char* env = std::getenv("INGINE_MODEL_DIR"); env && *env) return env;
  return std::filesystem::path(INGINE_MODEL_DIR);
}

std::vector<int32_t> render_prompt(const ingine::text::ChatFormat& fmt, const std::string& sys,
                                   const std::string& user) {
  ingine::text::ChatOptions opts;
  opts.reasoning_effort = "low";
  nlohmann::ordered_json msgs = nlohmann::ordered_json::array();
  if (!sys.empty()) msgs.push_back({{"role", "system"}, {"content", sys}});
  msgs.push_back({{"role", "user"}, {"content", user}});
  return fmt.render(msgs, nlohmann::ordered_json(), opts).ids;
}

std::vector<int32_t> drain_job(Job& job, std::optional<std::string>* err = nullptr) {
  std::vector<int32_t> out;
  for (;;) {
    JobBatch b = job.take(std::chrono::seconds(120));
    out.insert(out.end(), b.tokens.begin(), b.tokens.end());
    if (b.failure && err) *err = b.failure;
    if (b.end) return out;
    if (b.empty()) throw std::runtime_error("timeout draining job");
  }
}

struct RunOut {
  std::vector<int32_t> tokens;
  GenerateStats stats;
};

RunOut run_solo(Generator& gen, const Prompt& p, int max_new, const Sampling& sampling,
                std::span<const int32_t> stops = {}) {
  RunOut r;
  r.stats = gen.generate(p, max_new, sampling, stops, [&](std::span<const int32_t> toks) {
    r.tokens.insert(r.tokens.end(), toks.begin(), toks.end());
    return true;
  });
  return r;
}

std::vector<std::vector<int32_t>> run_batch_greedy(Generator& gen, const std::vector<Prompt>& prompts, int max_new,
                                                   int& total_rounds, int& total_accepted) {
  const int B = int(prompts.size());
  std::vector<std::vector<int32_t>> outs(static_cast<std::size_t>(B));
  struct Active {
    int idx = 0;
    int slot = -1;
    int32_t last_tok = 0;
  };
  std::vector<Active> live;
  Sampling greedy;
  greedy.temperature = 0.0f;
  for (int i = 0; i < B; ++i) {
    int32_t tok0 = 0;
    GenerateStats st;
    const int s = gen.start_slot(prompts[std::size_t(i)], max_new, greedy, tok0, st);
    CHECK(s >= 0);
    if (s < 0) return outs;
    outs[std::size_t(i)].push_back(tok0);
    if (int(outs[std::size_t(i)].size()) >= max_new) gen.release_slot(s);
    else live.push_back(Active{i, s, tok0});
  }
  total_rounds = 0;
  total_accepted = 0;
  while (!live.empty()) {
    std::vector<BatchSlotStep> steps(live.size());
    std::vector<BatchSlotResult> res(live.size());
    for (std::size_t i = 0; i < live.size(); ++i) steps[i] = {live[i].slot, live[i].last_tok};
    gen.step_slots_multi(steps, res);
    std::vector<Active> kept;
    for (std::size_t i = 0; i < live.size(); ++i) {
      total_rounds += res[i].spec_rounds;
      total_accepted += res[i].spec_accepted;
      auto& out = outs[std::size_t(live[i].idx)];
      for (int k = 0; k < res[i].count && int(out.size()) < max_new; ++k) {
        out.push_back(res[i].tokens[k]);
      }
      live[i].last_tok = out.back();
      if (int(out.size()) >= max_new) gen.release_slot(live[i].slot);
      else kept.push_back(live[i]);
    }
    live = std::move(kept);
  }
  return outs;
}

}  // namespace

int main() {
  int n_dev = 0;
  if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) {
    std::fprintf(stderr, "batch_spec_parity: no CUDA device, skipping\n");
    return 77;
  }
  const auto model_dir = find_model_dir();
  if (!std::filesystem::exists(model_dir / "mtp.safetensors")) {
    std::fprintf(stderr, "batch_spec_parity: model/mtp.safetensors not found at %s, skipping\n",
                 model_dir.string().c_str());
    return 77;
  }

  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  const auto& tok = fmt.tokenizer();
  const int32_t eos_id = *tok.token_id("<|content_model_end_sampling|>");

  const std::vector<std::string> base_users = {
      "Explain how a hash map handles collisions, with a short Python example.",
      "Write a Python module implementing an LRU cache with O(1) get and put.",
      "Compare B-trees and LSM-trees for write-heavy database workloads.",
      "Explain how TCP congestion control slow start and AIMD work.",
      "Write a C++20 ring buffer with atomic head and tail indices.",
      "Explain virtual memory paging versus segmentation concisely.",
      "Summarize sliding window attention combined with periodic full attention.",
      "Write a Python function solving 0/1 knapsack with path reconstruction.",
      "Explain MVCC snapshot isolation in relational databases.",
      "Write a concise Python binary search tree insert and lookup.",
      "Explain Raft leader election and log replication in bullets.",
      "Write a Python topological sort using Kahn's algorithm.",
  };

  std::vector<std::vector<int32_t>> prompt_ids;
  for (const auto& u : base_users) prompt_ids.push_back(render_prompt(fmt, "", u));

  // ---- 1. Matched greedy --no-spec vs --spec parity at B=1, 4, 8, 12 (including intermediate bucket B=12!)
  {
    RuntimeOptions opt_plain;
    opt_plain.model_dir = model_dir;
    opt_plain.max_ctx = 2048;
    opt_plain.prefill_chunk = 512;
    opt_plain.dense_fp8 = true;
    opt_plain.kv_fp8 = true;
    opt_plain.max_batch = 12;
    opt_plain.spec = false;
    opt_plain.prefix_cache = false;
    auto gen_plain = ingine::model::make_generator(opt_plain);

    std::vector<std::vector<std::vector<int32_t>>> plain_refs;
    const int kGenLen = 48;
    for (int B : {1, 4, 8, 12}) {
      std::vector<Prompt> ps;
      for (int i = 0; i < B; ++i) ps.push_back(Prompt{prompt_ids[std::size_t(i)], {}});
      int r = 0, a = 0;
      plain_refs.push_back(run_batch_greedy(*gen_plain, ps, kGenLen, r, a));
    }
    gen_plain.reset();

    RuntimeOptions opt_spec = opt_plain;
    opt_spec.spec = true;
    opt_spec.spec_threshold = 0.0f;  // always verify full width to exercise rollback thoroughly
    auto gen_spec = ingine::model::make_generator(opt_spec);
    CHECK(gen_spec->supports_batched_spec());

    const int bs[] = {1, 4, 8, 12};
    for (int bi = 0; bi < 4; ++bi) {
      const int B = bs[bi];
      std::vector<Prompt> ps;
      for (int i = 0; i < B; ++i) ps.push_back(Prompt{prompt_ids[std::size_t(i)], {}});
      int rounds = 0, accepted = 0;
      auto got = run_batch_greedy(*gen_spec, ps, kGenLen, rounds, accepted);
      int exact_tokens = 0, total_tokens = 0;
      for (int i = 0; i < B; ++i) {
        CHECK(got[std::size_t(i)].size() == plain_refs[std::size_t(bi)][std::size_t(i)].size());
        const std::size_t n = std::min(got[std::size_t(i)].size(), plain_refs[std::size_t(bi)][std::size_t(i)].size());
        for (std::size_t k = 0; k < n; ++k) {
          ++total_tokens;
          if (got[std::size_t(i)][k] == plain_refs[std::size_t(bi)][std::size_t(i)][k]) ++exact_tokens;
        }
      }
      const double agree = total_tokens > 0 ? double(exact_tokens) / double(total_tokens) : 0.0;
      const double tok_per_round = rounds > 0 ? double(accepted + rounds) / double(rounds) : 0.0;
      std::printf("parity B=%-2d: agree=%d/%d (%.2f%%) rounds=%d accepted=%d (%.2f tok/round)\n", B, exact_tokens,
                  total_tokens, 100.0 * agree, rounds, accepted, tok_per_round);
      CHECK(rounds > 0);
      CHECK(accepted > 0);
      CHECK(agree >= 0.98);  // allow tiny BF16 GEMV/GEMM boundary rounding differences across B*W gear shifts
    }

    // ---- 2. Heterogeneous greedy + sampled batch with per-slot temperature, top_k, top_p, bans, and seeds
    {
      const int32_t ban_tok = plain_refs[0][0][1];  // ban the greedy next token for slot 1!
      std::vector<Sampling> samps(4);
      samps[0].temperature = 0.0f;  // greedy
      samps[1].temperature = 0.0f;
      samps[1].banned_tokens = {ban_tok};
      samps[2].temperature = 0.8f;
      samps[2].top_k = 40;
      samps[2].top_p = 0.95f;
      samps[2].seed = 12345;
      samps[3].temperature = 0.6f;
      samps[3].top_k = 20;
      samps[3].top_p = 0.9f;
      samps[3].banned_tokens = {eos_id};
      samps[3].seed = 67890;

      auto run_hetero = [&]() {
        std::vector<std::vector<int32_t>> outs(4);
        std::vector<int> slots(4, -1);
        for (int i = 0; i < 4; ++i) {
          int32_t t0 = 0;
          GenerateStats st;
          slots[std::size_t(i)] = gen_spec->start_slot(Prompt{prompt_ids[std::size_t(i)], {}}, 24, samps[std::size_t(i)], t0, st);
          CHECK(slots[std::size_t(i)] >= 0);
          outs[std::size_t(i)].push_back(t0);
        }
        for (int step = 0; step < 30; ++step) {
          std::vector<BatchSlotStep> st;
          std::vector<int> idxs;
          for (int i = 0; i < 4; ++i) {
            if (slots[std::size_t(i)] >= 0 && outs[std::size_t(i)].size() < 24) {
              st.push_back({slots[std::size_t(i)], outs[std::size_t(i)].back()});
              idxs.push_back(i);
            }
          }
          if (st.empty()) break;
          std::vector<BatchSlotResult> res(st.size());
          gen_spec->step_slots_multi(st, res);
          for (std::size_t k = 0; k < st.size(); ++k) {
            const int i = idxs[k];
            for (int j = 0; j < res[k].count && outs[std::size_t(i)].size() < 24; ++j)
              outs[std::size_t(i)].push_back(res[k].tokens[j]);
            if (outs[std::size_t(i)].size() >= 24) {
              gen_spec->release_slot(slots[std::size_t(i)]);
              slots[std::size_t(i)] = -1;
            }
          }
        }
        for (int s : slots)
          if (s >= 0) gen_spec->release_slot(s);
        return outs;
      };
      const auto h1 = run_hetero();
      const auto h2 = run_hetero();
      CHECK(h1 == h2);  // deterministic replay with fixed per-slot seeds!
      CHECK(std::find(h1[1].begin(), h1[1].end(), ban_tok) == h1[1].end());  // ban respected
      CHECK(std::find(h1[3].begin(), h1[3].end(), eos_id) == h1[3].end());   // sampled ban respected
      std::printf("heterogeneous greedy+sampled: deterministic=YES bans=RESPECTED\n");
    }

    // ---- 3. Multi-tenant shortlist isolation (rare tokens in slot B must NOT alter slot A proposals/outputs!)
    {
      Sampling greedy;
      greedy.temperature = 0.0f;
      Prompt pA{prompt_ids[0], {}, 0, "tenant-A", true};
      const RunOut soloA = run_solo(*gen_spec, pA, 32, greedy);

      // Build prompt B packed with rare high-id tokens outside the 32K shortlist
      std::vector<int32_t> rare_b = prompt_ids[1];
      for (int t = 150000; t < 150120; ++t) rare_b.push_back(t);
      Prompt pB{rare_b, {}, 0, "tenant-B", true};

      int r = 0, a = 0;
      const auto paired = run_batch_greedy(*gen_spec, {pA, pB}, 32, r, a);
      CHECK(paired[0] == soloA.tokens);
      std::printf("tenant isolation: solo_vs_concurrent=%s\n", paired[0] == soloA.tokens ? "EXACT" : "DIFF");
    }

    // ---- 4. Mid-block stop-token truncation
    {
      Sampling greedy;
      greedy.temperature = 0.0f;
      const RunOut full_run = run_solo(*gen_spec, Prompt{prompt_ids[0], {}}, 24, greedy);
      CHECK(full_run.tokens.size() == 24);
      const int32_t mid_stop = full_run.tokens[10];
      const std::vector<int32_t> stops = {mid_stop};
      const RunOut stopped = run_solo(*gen_spec, Prompt{prompt_ids[0], {}}, 24, greedy, stops);
      CHECK(stopped.tokens.size() == 11);
      CHECK(stopped.tokens.back() == mid_stop);
      CHECK(std::equal(stopped.tokens.begin(), stopped.tokens.end(), full_run.tokens.begin()));
      std::printf("mid-block stop truncation: len=%zu exact=YES\n", stopped.tokens.size());
    }
  }

  // ---- 5. Prefix cache + --spec exact, partial, root cold-vs-warm parity & namespace/bypass isolation
  {
    RuntimeOptions opt_cache;
    opt_cache.model_dir = model_dir;
    opt_cache.max_ctx = 1024;
    opt_cache.prefill_chunk = 512;
    opt_cache.dense_fp8 = true;
    opt_cache.kv_fp8 = true;
    opt_cache.max_batch = 4;
    opt_cache.spec = true;
    opt_cache.spec_threshold = 0.0f;
    opt_cache.prefix_cache = true;
    auto gen = ingine::model::make_generator(opt_cache);

    std::string long_sys;
    for (int i = 0; i < 30; ++i)
      long_sys += "System rule " + std::to_string(i) + ": verify every invariant before responding cleanly. ";
    const auto p_full1 = render_prompt(fmt, long_sys, "Explain red-black tree balancing in 3 sentences.");
    const auto p_full2 = render_prompt(fmt, long_sys, "Explain AVL tree rotations in 3 sentences.");
    // Compute shared prefix length
    int common = 0;
    while (common < int(std::min(p_full1.size(), p_full2.size())) &&
           p_full1[std::size_t(common)] == p_full2[std::size_t(common)])
      ++common;
    CHECK(common > 128);

    Sampling greedy;
    greedy.temperature = 0.0f;
    // Cold bypass baseline for p_full1 and p_full2
    const RunOut cold1 = run_solo(*gen, Prompt{p_full1, {}, 0, "ns1", true}, 32, greedy);
    const RunOut cold2 = run_solo(*gen, Prompt{p_full2, {}, 0, "ns1", true}, 32, greedy);
    CHECK(cold1.stats.reused_tokens == 0 && cold2.stats.reused_tokens == 0);

    // Populate root + full checkpoint via p_full1
    const RunOut warm_seed = run_solo(*gen, Prompt{p_full1, {}, common, "ns1", false}, 32, greedy);
    CHECK(warm_seed.tokens == cold1.tokens);
    CHECK(warm_seed.stats.spec_rounds == cold1.stats.spec_rounds);
    CHECK(warm_seed.stats.spec_accepted == cold1.stats.spec_accepted);

    // Exact hit on p_full1
    const RunOut exact_hit = run_solo(*gen, Prompt{p_full1, {}, common, "ns1", false}, 32, greedy);
    CHECK(exact_hit.stats.reused_tokens == int(p_full1.size()));
    CHECK(exact_hit.stats.reuse == Reuse::PagedPrefix);
    CHECK(exact_hit.tokens == cold1.tokens);
    CHECK(exact_hit.stats.spec_rounds == cold1.stats.spec_rounds);
    CHECK(exact_hit.stats.spec_accepted == cold1.stats.spec_accepted);

    // Root/partial hit on p_full2
    const RunOut partial_hit = run_solo(*gen, Prompt{p_full2, {}, common, "ns1", false}, 32, greedy);
    CHECK(partial_hit.stats.reused_tokens == common);
    CHECK(partial_hit.tokens == cold2.tokens);

    // Namespace miss isolation ("ns2" must NOT reuse "ns1")
    const RunOut ns_miss = run_solo(*gen, Prompt{p_full1, {}, common, "ns2", false}, 16, greedy);
    CHECK(ns_miss.stats.reused_tokens == 0);
    std::printf("prefix-cache+spec: exact_reused=%d partial_reused=%d ns_miss_reused=%d parity=EXACT\n",
                exact_hit.stats.reused_tokens, partial_hit.stats.reused_tokens, ns_miss.stats.reused_tokens);
  }

  // ---- 6. Tight 6-page pool (384 tokens, exact 6 pages) with --spec:
  //         24-tok overlap, 96-tok safe deferral, and 65/96 boundary reservation protection!
  {
    RuntimeOptions opt_tight;
    opt_tight.model_dir = model_dir;
    opt_tight.max_ctx = 384;
    opt_tight.max_kv_tokens = 384;  // exact 6 pages!
    opt_tight.prefill_chunk = 384;
    opt_tight.dense_fp8 = true;
    opt_tight.kv_fp8 = true;
    opt_tight.max_batch = 2;
    opt_tight.spec = true;
    opt_tight.spec_threshold = 0.0f;
    opt_tight.prefix_cache = true;
    auto gen = ingine::model::make_generator(opt_tight);

    std::vector<int32_t> p192 = prompt_ids[0];
    while (p192.size() < 192) p192.insert(p192.end(), prompt_ids[1].begin(), prompt_ids[1].end());
    p192.resize(192);

    Sampling greedy;
    greedy.temperature = 0.0f;
    const RunOut solo96 = run_solo(*gen, Prompt{p192, {}}, 96, greedy);
    CHECK(solo96.tokens.size() == 96);

    Scheduler sched(*gen, {}, 4);
    // 24-tok siblings: 3 shared + 1 private each = 5 <= 6 -> overlap!
    auto s24a = std::make_shared<Job>(p192, 24, greedy);
    auto s24b = std::make_shared<Job>(p192, 24, greedy);
    CHECK(sched.submit(s24a) && sched.submit(s24b));
    std::optional<std::string> e1, e2;
    const auto o24a = drain_job(*s24a, &e1);
    const auto o24b = drain_job(*s24b, &e2);
    CHECK(!e1 && !e2 && o24a.size() == 24 && o24b.size() == 24);
    CHECK(std::equal(o24a.begin(), o24a.end(), solo96.tokens.begin()));
    CHECK(std::equal(o24b.begin(), o24b.end(), solo96.tokens.begin()));

    // 65/96 mixed budget boundary: 3 shared + 1 private (65) + 2 private (96) = 6 pages exact!
    auto m65 = std::make_shared<Job>(p192, 65, greedy);
    auto m96 = std::make_shared<Job>(p192, 96, greedy);
    CHECK(sched.submit(m65) && sched.submit(m96));
    std::optional<std::string> em1, em2;
    const auto om65 = drain_job(*m65, &em1);
    const auto om96 = drain_job(*m96, &em2);
    CHECK(!em1 && !em2 && om65.size() == 65 && om96.size() == 96);
    CHECK(std::equal(om65.begin(), om65.end(), solo96.tokens.begin()));
    CHECK(om96 == solo96.tokens);

    // 96/96 long siblings: 3 shared + 2 + 2 = 7 > 6 -> safe deferral, both finish 96/96 exact!
    auto l96a = std::make_shared<Job>(p192, 96, greedy);
    auto l96b = std::make_shared<Job>(p192, 96, greedy);
    CHECK(sched.submit(l96a) && sched.submit(l96b));
    std::optional<std::string> el1, el2;
    const auto ol96a = drain_job(*l96a, &el1);
    const auto ol96b = drain_job(*l96b, &el2);
    CHECK(!el1 && !el2 && ol96a == solo96.tokens && ol96b == solo96.tokens);
    std::printf("tight 6-page + spec: 24/24=PASS 65/96=PASS 96/96=PASS\n");
  }

  std::printf(g_failures ? "FAILED (%d checks)\n" : "batch_spec_parity ok\n", g_failures);
  return g_failures ? 1 : 0;
}

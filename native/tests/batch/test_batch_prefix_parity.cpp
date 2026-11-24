// GPU end-to-end parity & edge-case test for Stage 2 (#825) batched (B>1) prefix checkpoint/restore:
// Compares cache-off (prefix_cache=false) vs cache-on (prefix_cache=true) at identical checkpoint, build,
// GPU, batch shape, and prefill chunking across:
//   1. Exact identical prompt hit (L == n -> Reuse::PagedPrefix, zero prefill compute).
//   2. Page boundaries 63, 64, 65, 128, 129 (committed pages + private partial tails) at matched chunk boundaries.
//   3. Mid-prefill verified root_prefix_tokens snapshot + concurrent B=4 forks sharing committed pages.
//   4. Sliding-ring wraparound (>1024 tokens: 1088-token prefix + 40-token suffix).
//   5. Isolation: mismatched cache_namespace -> clean miss (reused_tokens == 0).
//   6. Early slot cancellation & recycle, plus tight-pool eviction with live siblings (non-starvation).
//
// Usage: batch_prefix_parity [model_dir]
#include "core/device.h"
#include "engine/scheduler.h"
#include "ingine/generator.h"
#include "text/chat.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

void print_vec_diff(const char* label, const std::vector<int32_t>& got, const std::vector<int32_t>& ref) {
  int match = 0;
  const std::size_t n = std::min(got.size(), ref.size());
  for (std::size_t i = 0; i < n; ++i)
    if (got[i] == ref[i]) ++match;
  std::fprintf(stderr, "  [%s] match=%d/%zu got_sz=%zu ref_sz=%zu first_got=%d first_ref=%d\n", label, match, n,
               got.size(), ref.size(), got.empty() ? -1 : got[0], ref.empty() ? -1 : ref[0]);
}

std::vector<int32_t> drain(ingine::engine::Job& job) {
  std::vector<int32_t> out;
  for (;;) {
    auto b = job.take(std::chrono::seconds(60));
    out.insert(out.end(), b.tokens.begin(), b.tokens.end());
    if (b.failure) {
      std::fprintf(stderr, "job failed: %s\n", b.failure->c_str());
      ++g_failures;
      break;
    }
    if (b.end) break;
  }
  return out;
}

std::vector<int32_t> pad_to_len(const std::vector<int32_t>& base, int target_len, const std::vector<int32_t>& filler) {
  std::vector<int32_t> out = base;
  std::size_t idx = 0;
  while (int(out.size()) < target_len) {
    out.push_back(filler[idx % filler.size()]);
    ++idx;
  }
  out.resize(static_cast<std::size_t>(target_len));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  const char* env_dir = std::getenv("INGINE_MODEL_DIR");
  const std::string model_dir = argc > 1 ? argv[1] : (env_dir ? env_dir : "");
  if (model_dir.empty()) {
    std::fprintf(stderr, "batch_prefix_parity: pass model_dir or set INGINE_MODEL_DIR\n");
    return 2;
  }

  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  const auto& tok = fmt.tokenizer();
  const std::vector<int32_t> stop = {*tok.token_id("<|content_model_end_sampling|>")};

  ingine::text::ChatOptions chat;
  chat.reasoning_effort = "low";
  auto render = [&](const std::string& text) {
    const nlohmann::ordered_json msgs = nlohmann::ordered_json::array({{{"role", "user"}, {"content", text}}});
    return fmt.render(msgs, nlohmann::ordered_json(), chat).ids;
  };
  const std::vector<int32_t> filler = tok.encode(
      " Explain memory coalescing, bank conflicts, warp divergence, and stream synchronization in CUDA kernels. ",
      false);

  // Build test prompts covering exact page boundaries (63, 64, 65, 128, 129), forks, and sliding-ring wrap (>1024).
  const auto seed_base = render("You are a concise CUDA systems engineering assistant.");
  const auto p63 = pad_to_len(seed_base, 63, filler);
  const auto p64 = pad_to_len(p63, 64, filler);
  const auto p65 = pad_to_len(p64, 65, filler);
  auto p128_a = pad_to_len(p65, 128, filler);
  auto p128_b = pad_to_len(p65, 128, filler);
  p128_b[70] = filler[3];  // fork right after 65-token shared prefix
  auto p129_a = pad_to_len(p128_a, 129, filler);
  auto p_wrap_prefix = pad_to_len(p129_a, 1088, filler);  // 17*64 = 1088 > 1024 triggers sliding-ring wrap
  auto p_wrap_full = pad_to_len(p_wrap_prefix, 1128, filler);

  const int gen_tokens = 16;
  ingine::model::Sampling greedy;
  greedy.temperature = 0.0f;

  // 1. Cache-OFF baseline (prefix_cache = false) on BatchGenerator (max_batch = 4, prefill_chunk = 64).
  // Matching prefill_chunk = 64 guarantees identical GEMM chunk boundaries at every page multiple (64, 128, 1088),
  // isolating prefix-cache state fidelity from cuBLASLt chunk-size tiling differences (cf. test_reuse.cpp).
  std::vector<int32_t> ref_p63, ref_p64, ref_p65, ref_p128_a, ref_p129, ref_wrap;
  std::vector<std::vector<int32_t>> ref_forks(4);
  {
    ingine::model::RuntimeOptions opts;
    opts.model_dir = model_dir;
    opts.max_ctx = 2048;
    opts.prefill_chunk = 64;
    opts.dense_fp8 = true;
    opts.kv_fp8 = true;
    opts.park_mb = 0;
    opts.prefix_cache = false;
    opts.max_batch = 4;
    opts.max_kv_tokens = 4 * 1280;
    auto gen = ingine::model::make_generator(opts);

    auto run_one = [&](const std::vector<int32_t>& ids) {
      std::vector<int32_t> out;
      auto st = gen->generate(ids, gen_tokens, greedy, stop, [&](std::span<const int32_t> t) {
        out.insert(out.end(), t.begin(), t.end());
        return true;
      });
      CHECK(st.reused_tokens == 0 && st.reuse == ingine::model::Reuse::None);
      return out;
    };
    ref_p63 = run_one(p63);
    ref_p64 = run_one(p64);
    ref_p65 = run_one(p65);
    ref_p128_a = run_one(p128_a);
    ref_p129 = run_one(p129_a);
    ref_wrap = run_one(p_wrap_full);

    ingine::engine::Scheduler sched(*gen, stop, 16);
    std::vector<std::vector<int32_t>> fork_prompts = {p128_a, p128_b, p129_a, p64};
    std::vector<std::shared_ptr<ingine::engine::Job>> jobs;
    for (const auto& fp : fork_prompts) {
      auto j = std::make_shared<ingine::engine::Job>(fp, gen_tokens, greedy);
      CHECK(sched.submit(j));
      jobs.push_back(std::move(j));
    }
    for (std::size_t i = 0; i < jobs.size(); ++i) ref_forks[i] = drain(*jobs[i]);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 2. Cache-ON (prefix_cache = true) on BatchGenerator (max_batch = 4, prefill_chunk = 64).
  {
    ingine::model::RuntimeOptions opts;
    opts.model_dir = model_dir;
    opts.max_ctx = 2048;
    opts.prefill_chunk = 64;
    opts.dense_fp8 = true;
    opts.kv_fp8 = true;
    opts.park_mb = 768;
    opts.prefix_cache = true;
    opts.max_batch = 4;
    opts.max_kv_tokens = 4 * 1280;
    auto gen = ingine::model::make_generator(opts);

    auto run_cached = [&](const std::vector<int32_t>& ids, const std::string& ns, int root_toks,
                          ingine::model::GenerateStats& out_st) {
      std::vector<int32_t> out;
      ingine::model::Prompt pr{ids, {}, root_toks, ns};
      out_st = gen->generate(pr, gen_tokens, greedy, stop, [&](std::span<const int32_t> t) {
        out.insert(out.end(), t.begin(), t.end());
        return true;
      });
      return out;
    };

    // 2a. Exact hit (63 -> 63), namespace isolation (ns-b miss), and page-aligned boundary chain
    //     (64 -> 65 -> 128 -> 129, exact matched chunk boundaries!).
    ingine::model::GenerateStats st;
    auto out_63_cold = run_cached(p63, "ns-tail", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::None && st.reused_tokens == 0);
    CHECK(out_63_cold == ref_p63);

    auto out_63_warm = run_cached(p63, "ns-tail", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 63 &&
          st.restored_bytes.has_value() && *st.restored_bytes > 0 && st.restore_seconds.has_value());
    CHECK(out_63_warm == ref_p63);

    // Partial tail hit (63 -> 64): verifies tail_k/tail_v restore into scratch + scatter of page 0.
    auto out_64_from_63 = run_cached(p64, "ns-tail", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 63 &&
          st.restored_bytes.has_value() && *st.restored_bytes > 0);
    CHECK(!out_64_from_63.empty() && out_64_from_63[0] == ref_p64[0]);

    // Namespace isolation: identical p63 in "ns-b" MUST MISS!
    auto out_63_other_ns = run_cached(p63, "ns-b", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::None && st.reused_tokens == 0 &&
          st.restored_bytes.has_value() && *st.restored_bytes == 0);
    CHECK(out_63_other_ns == ref_p63);

    // Matched-chunk-boundary chain in "ns-aligned": cold 64 -> warm 65 (64+1) -> warm 128 (64+64) -> warm 129 (128+1).
    auto out_64_cold = run_cached(p64, "ns-aligned", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::None && st.reused_tokens == 0);
    if (out_64_cold != ref_p64) print_vec_diff("p64_cold", out_64_cold, ref_p64);
    CHECK(out_64_cold == ref_p64);

    auto out_65 = run_cached(p65, "ns-aligned", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 64);
    if (out_65 != ref_p65) print_vec_diff("p65_from_64", out_65, ref_p65);
    CHECK(out_65 == ref_p65);

    auto out_128 = run_cached(p128_a, "ns-aligned-128", 64, st);  // cold w/ root_prefix_tokens = 64 -> pins root!
    CHECK(st.reuse == ingine::model::Reuse::None && st.reused_tokens == 0);
    if (out_128 != ref_p128_a) print_vec_diff("p128_cold_root64", out_128, ref_p128_a);
    CHECK(out_128 == ref_p128_a);

    auto out_129 = run_cached(p129_a, "ns-aligned-128", 64, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 128);
    if (out_129 != ref_p129) print_vec_diff("p129_from_128", out_129, ref_p129);
    CHECK(out_129 == ref_p129);

    // 2b. Sliding-ring wraparound (>1024 tokens): prime 1088-tok prefix (17*64), then extend to 1128.
    auto out_wrap_prime = run_cached(p_wrap_prefix, "ns-wrap", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::None && st.reused_tokens == 0);
    const double cold_wrap_prefill_ms = st.prefill_seconds * 1e3;
    (void)out_wrap_prime;
    auto out_wrap_hit = run_cached(p_wrap_full, "ns-wrap", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 1088 &&
          st.restored_bytes.has_value() && *st.restored_bytes > 0);
    if (out_wrap_hit != ref_wrap) print_vec_diff("wrap_1088_to_1128", out_wrap_hit, ref_wrap);
    int wrap_match = 0;
    for (std::size_t k = 0; k < std::min(out_wrap_hit.size(), ref_wrap.size()); ++k)
      if (out_wrap_hit[k] == ref_wrap[k]) ++wrap_match;
    std::printf("wrap (>1024) parity: %d/%zu tokens matched (reused=%d, restored_bytes=%llu, cold_prefill=%.1fms, "
                "warm_prefill=%.1fms, restore=%.2fms)\n",
                wrap_match, ref_wrap.size(), st.reused_tokens,
                static_cast<unsigned long long>(st.restored_bytes.value_or(0)), cold_wrap_prefill_ms,
                st.prefill_seconds * 1e3, st.restore_seconds.value_or(0.0) * 1e3);
    CHECK(out_wrap_hit == ref_wrap);

    // 2c. Concurrent B=4 forks sharing committed pages + verified root boundary (64) via Scheduler::loop_batched()
    ingine::engine::Scheduler sched(*gen, stop, 16);
    std::vector<std::vector<int32_t>> fork_prompts = {p128_a, p128_b, p129_a, p64};
    std::vector<std::shared_ptr<ingine::engine::Job>> jobs;
    for (const auto& fp : fork_prompts) {
      auto j = std::make_shared<ingine::engine::Job>(fp, gen_tokens, greedy, std::vector<ingine::model::PromptImage>{},
                                                     64, "ns-aligned-128");
      CHECK(sched.submit(j));
      jobs.push_back(std::move(j));
    }
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const auto out = drain(*jobs[i]);
      const auto pst = jobs[i]->prefill_stats();
      CHECK(pst.has_value() && pst->reuse == ingine::model::Reuse::PagedPrefix && pst->reused_tokens >= 64);
      if (out != ref_forks[i]) print_vec_diff("fork", out, ref_forks[i]);
      CHECK(out == ref_forks[i]);
    }

    // 2d. Early cancellation & slot recycle (must release active refs cleanly with zero leaks).
    {
      auto cancel_job = std::make_shared<ingine::engine::Job>(p129_a, 64, greedy,
                                                              std::vector<ingine::model::PromptImage>{}, 64,
                                                              "ns-aligned-128");
      CHECK(sched.submit(cancel_job));
      auto b = cancel_job->take(std::chrono::seconds(30));
      CHECK(!b.tokens.empty() && !b.failure);
      cancel_job->cancel();
      auto pst = cancel_job->wait_prefill_stats();
      CHECK(pst.has_value() && pst->reuse == ingine::model::Reuse::PagedPrefix && pst->reused_tokens >= 64);
      // Follow-up request after cancellation must succeed with identical output.
      auto follow_job = std::make_shared<ingine::engine::Job>(p128_a, gen_tokens, greedy,
                                                              std::vector<ingine::model::PromptImage>{}, 64,
                                                              "ns-aligned-128");
      CHECK(sched.submit(follow_job));
      CHECK(drain(*follow_job) == ref_forks[0]);
    }

    const auto cstats = gen->cache_stats();
    std::printf("batch_prefix_parity cache_stats: entries=%llu root=%llu vram_used=%zu vram_peak=%zu d2d_restore=%llu\n",
                static_cast<unsigned long long>(cstats.vram_entries),
                static_cast<unsigned long long>(cstats.root_entries), cstats.vram_used_bytes, cstats.vram_peak_bytes,
                static_cast<unsigned long long>(cstats.d2d_restore_bytes));
    CHECK(cstats.vram_entries > 0 && cstats.root_entries >= 1 && cstats.d2d_restore_bytes > 0);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 3. Tight KV pool eviction with live siblings (max_ctx = 384, max_kv_tokens = 384 -> exactly 6 pages total).
  // Prime tight-b (LRU, 2 pages) and tight-a (MRU, 2 pages): 4 of 6 pages pinned in cache, 2 free.
  // Run live sibling sharing tight-a (attaches 2 committed + 1 decode = 3 active pages, 1 free) concurrently with
  // an unrelated 160-tok prompt needing 3 pages -> MUST evict tight-b (2 pages) without starving admission or
  // corrupting the live sibling!
  {
    ingine::model::RuntimeOptions opts;
    opts.model_dir = model_dir;
    opts.max_ctx = 384;
    opts.prefill_chunk = 64;
    opts.dense_fp8 = true;
    opts.kv_fp8 = true;
    opts.park_mb = 512;
    opts.prefix_cache = true;
    opts.max_batch = 2;
    opts.max_kv_tokens = 384;  // exactly 6 pages (max_pages_per_slot = 6)
    auto gen = ingine::model::make_generator(opts);

    auto p_alt_b = pad_to_len(render("Distinct topic about distributed consensus algorithms."), 128, filler);
    auto p_pressure = pad_to_len(render("Unrelated long prompt forcing KV page eviction under pressure."), 160, filler);
    auto run_plain = [&](const std::vector<int32_t>& ids, const std::string& ns) {
      std::vector<int32_t> out;
      ingine::model::Prompt pr{ids, {}, 0, ns};
      (void)gen->generate(pr, gen_tokens, greedy, stop, [&](std::span<const int32_t> t) {
        out.insert(out.end(), t.begin(), t.end());
        return true;
      });
      return out;
    };
    (void)run_plain(p_alt_b, "tight-b");
    CHECK(run_plain(p128_a, "tight-a") == ref_forks[0]);

    ingine::engine::Scheduler sched(*gen, stop, 8);
    auto j_sib = std::make_shared<ingine::engine::Job>(p128_a, gen_tokens, greedy,
                                                       std::vector<ingine::model::PromptImage>{}, 0, "tight-a");
    auto j_press = std::make_shared<ingine::engine::Job>(p_pressure, gen_tokens, greedy,
                                                         std::vector<ingine::model::PromptImage>{}, 0, "tight-c");
    CHECK(sched.submit(j_sib));
    CHECK(sched.submit(j_press));
    const auto out_sib = drain(*j_sib);
    const auto out_press = drain(*j_press);
    CHECK(out_sib == ref_forks[0]);
    CHECK(!out_press.empty());
    const auto st_sib = j_sib->prefill_stats();
    CHECK(st_sib.has_value() && st_sib->reuse == ingine::model::Reuse::PagedPrefix && st_sib->reused_tokens == 128);
    const auto cstats = gen->cache_stats();
    std::printf("tight-pool live-sibling eviction: vram_evictions=%llu entries=%llu\n",
                static_cast<unsigned long long>(cstats.vram_evictions),
                static_cast<unsigned long long>(cstats.vram_entries));
    CHECK(cstats.vram_evictions > 0);
  }

  std::printf(g_failures ? "FAILED (%d checks)\n" : "batch_prefix_parity ok\n", g_failures);
  return g_failures ? 1 : 0;
}

// GPU end-to-end parity & edge-case test for Stage 2 (#825) batched (B>1) prefix checkpoint/restore:
// Compares cache-off (prefix_cache=false) vs cache-on (prefix_cache=true) at identical checkpoint, build,
// GPU, batch shape, and matched prefill chunk boundaries across:
//   1. Exact identical prompt hit (L == n -> Reuse::PagedPrefix, zero prefill compute).
//   2. Arbitrary-tail & page-boundary restoration (63->64, 64->65, 65->129, 128->129): whole 16-token
//      continuation compared against cold execution with IDENTICAL prefill split (prefill_chunk = 63, 64, 65),
//      plus preserved original 512-chunk unsplit diagnostic ("alpha beta gamma ...").
//   3. Mid-prefill verified root_prefix_tokens snapshot + concurrent B=4 forks sharing committed pages.
//   4. Sliding-ring wraparound (>1024 tokens: 1088-token prefix + 40-token suffix) with matched 1128-token
//      cold vs warm prefill timing and CUDA-event GPU restore_seconds.
//   5. Isolation: mismatched cache_namespace and bypass_prefix_cache -> clean miss (reused_tokens == 0).
//   6. Recoverable failure cleanup (fail_next_allocations + throwing on_prefill) & early cancel/recycle.
//   7. Tight 6-page KV pool: (a) overlapping live-sibling prompt-aware admission via Scheduler (192-tok
//      exact prefix: 3 shared + 1 private active, 2 free -> 2nd identical sibling admits concurrently!)
//      and (b) live-sibling eviction under pressure (vram_evictions > 0, zero admission starvation).
//
// Usage: batch_prefix_parity [model_dir]
#include "core/arena.h"
#include "core/device.h"
#include "engine/scheduler.h"
#include "ingine/generator.h"
#include "text/chat.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::model {
void set_batch_prefill_splits_for_test(std::vector<int> splits);
}  // namespace ingine::model

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

int count_matching(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  int m = 0;
  for (std::size_t i = 0; i < n; ++i)
    if (a[i] == b[i]) ++m;
  return m;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
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

  // Preserve the EXACT original run-1 prompt ("alpha beta gamma ...") as an explicit diagnostic + matched-split check!
  const std::vector<int32_t> orig_filler =
      tok.encode(" alpha beta gamma delta epsilon zeta eta theta iota kappa ", false);
  const auto orig_seed = render("You are a concise systems engineering assistant.");
  const auto orig_p63 = pad_to_len(orig_seed, 63, orig_filler);
  const auto orig_p64 = pad_to_len(orig_p63, 64, orig_filler);
  const auto orig_p65 = pad_to_len(orig_p64, 65, orig_filler);
  const auto orig_p129 = pad_to_len(orig_p65, 129, orig_filler);

  // Realistic technical prompt chain for page boundaries, forks, wraparound (>1024), and 192-tok overlapping siblings.
  const std::vector<int32_t> filler = tok.encode(
      " Explain memory coalescing, bank conflicts, warp divergence, and stream synchronization in CUDA kernels. ",
      false);
  const auto seed_base = render("You are a concise CUDA systems engineering assistant.");
  const auto p63 = pad_to_len(seed_base, 63, filler);
  const auto p64 = pad_to_len(p63, 64, filler);
  const auto p65 = pad_to_len(p64, 65, filler);
  auto p128_a = pad_to_len(p65, 128, filler);
  auto p128_b = pad_to_len(p65, 128, filler);
  p128_b[70] = filler[3];
  auto p129_a = pad_to_len(p128_a, 129, filler);
  auto p192 = pad_to_len(p129_a, 192, filler);
  auto p_wrap_prefix = pad_to_len(p129_a, 1088, filler);  // 17*64 = 1088 > 1024 triggers sliding-ring wrap
  auto p_wrap_full = pad_to_len(p_wrap_prefix, 1128, filler);

  const int gen_tokens = 16;
  ingine::model::Sampling greedy;
  greedy.temperature = 0.0f;

  auto run_cold_with_chunk = [&](int chunk, const std::vector<std::vector<int32_t>>& prompts,
                                 std::vector<double>* prefill_ms_out = nullptr) {
    ingine::model::RuntimeOptions opts;
    opts.model_dir = model_dir;
    opts.max_ctx = 2048;
    opts.prefill_chunk = chunk;
    opts.dense_fp8 = true;
    opts.kv_fp8 = true;
    opts.park_mb = 0;
    opts.prefix_cache = false;
    opts.max_batch = 4;
    opts.max_kv_tokens = 4 * 1280;
    auto gen = ingine::model::make_generator(opts);
    std::vector<std::vector<int32_t>> outs;
    for (const auto& ids : prompts) {
      std::vector<int32_t> out;
      auto st = gen->generate(ids, gen_tokens, greedy, stop, [&](std::span<const int32_t> t) {
        out.insert(out.end(), t.begin(), t.end());
        return true;
      });
      CHECK(st.reused_tokens == 0 && st.reuse == ingine::model::Reuse::None);
      if (prefill_ms_out) prefill_ms_out->push_back(st.prefill_seconds * 1e3);
      outs.push_back(std::move(out));
    }
    return outs;
  };

  // 1. Cold references with MATCHED prefill splits + unsplit 512-chunk diagnostic baseline.
  //    - chunk=63 -> cold [0,63)+[63,64) matches arbitrary-tail restore 63->64!
  //    - chunk=65 -> cold [0,65)+[65,129) matches page+tail restore 65->129!
  //    - chunk=64 -> cold [0,64)+[64,65), [0,64)+[64,128), [0,1088)+[1088,1128) match page-aligned restores!
  //    - chunk=512 -> ordinary unsplit cold baseline (diagnostic comparison for chunk-geometry sensitivity).
  const auto ref_split63 = run_cold_with_chunk(63, {orig_p64, p64});
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto ref_split65 = run_cold_with_chunk(65, {orig_p129, p129_a});
  CUDA_CHECK(cudaDeviceSynchronize());
  const auto ref_unsplit512 = run_cold_with_chunk(512, {orig_p64, orig_p65, orig_p129});
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> cold64_ms;
  const auto ref_split64 =
      run_cold_with_chunk(64, {orig_p64, orig_p65, p63, p64, p65, p128_a, p128_b, p129_a, p192, p_wrap_full}, &cold64_ms);
  const auto& ref_orig_p64_s64 = ref_split64[0];
  const auto& ref_orig_p65_s64 = ref_split64[1];
  const auto& ref_p63 = ref_split64[2];
  const auto& ref_p64 = ref_split64[3];
  const auto& ref_p65 = ref_split64[4];
  const auto& ref_p128_a = ref_split64[5];
  const auto& ref_p128_b = ref_split64[6];
  const auto& ref_p129 = ref_split64[7];
  const auto& ref_p192 = ref_split64[8];
  const auto& ref_wrap = ref_split64[9];
  const double cold_1128_prefill_ms = cold64_ms[9];
  CUDA_CHECK(cudaDeviceSynchronize());

  // 2. Cache-ON (prefix_cache = true) on BatchGenerator (max_batch = 4, prefill_chunk = 512 for arbitrary-tail
  //    exact-split verification AND original 512-chunk diagnostic!).
  {
    ingine::model::RuntimeOptions opts;
    opts.model_dir = model_dir;
    opts.max_ctx = 2048;
    opts.prefill_chunk = 512;
    opts.dense_fp8 = true;
    opts.kv_fp8 = true;
    opts.park_mb = 768;
    opts.prefix_cache = true;
    opts.max_batch = 4;
    opts.max_kv_tokens = 4 * 1280;
    auto gen = ingine::model::make_generator(opts);

    auto run_cached = [&](const std::vector<int32_t>& ids, const std::string& ns, int root_toks,
                          ingine::model::GenerateStats& out_st, bool bypass = false) {
      std::vector<int32_t> out;
      ingine::model::Prompt pr{ids, {}, root_toks, ns, bypass};
      out_st = gen->generate(pr, gen_tokens, greedy, stop, [&](std::span<const int32_t> t) {
        out.insert(out.end(), t.begin(), t.end());
        return true;
      });
      return out;
    };

    ingine::model::GenerateStats st;

    // 2a. Arbitrary-tail restoration (63->64, tail=63) compared WHOLE 16-token continuation against matched-split cold!
    (void)run_cached(orig_p63, "orig-tail63", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::None && st.reused_tokens == 0);
    auto orig_64_from_63 = run_cached(orig_p64, "orig-tail63", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 63 &&
          st.restore_seconds.has_value() && *st.restore_seconds > 0.0);
    if (orig_64_from_63 != ref_split63[0]) print_vec_diff("orig_63->64_vs_split63", orig_64_from_63, ref_split63[0]);
    CHECK(orig_64_from_63 == ref_split63[0]);

    (void)run_cached(p63, "tech-tail63", 0, st);
    auto tech_64_from_63 = run_cached(p64, "tech-tail63", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 63);
    if (tech_64_from_63 != ref_split63[1]) print_vec_diff("tech_63->64_vs_split63", tech_64_from_63, ref_split63[1]);
    CHECK(tech_64_from_63 == ref_split63[1]);

    // 2b. Original run-1 diagnostic:
    //     - Matched split (64+1): cold orig_p64 -> warm orig_p65 MUST match ref_orig_p65_s64 16/16!
    //     - Also record ordinary unsplit 512-chunk comparison and 63+1+1 chain diagnostic.
    auto orig_64_cold = run_cached(orig_p64, "orig-page64", 0, st);
    CHECK(orig_64_cold == ref_orig_p64_s64);
    auto orig_65_from_64 = run_cached(orig_p65, "orig-page64", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 64);
    if (orig_65_from_64 != ref_orig_p65_s64)
      print_vec_diff("orig_64->65_vs_split64", orig_65_from_64, ref_orig_p65_s64);
    CHECK(orig_65_from_64 == ref_orig_p65_s64);
    auto orig_65_chain_63_64_65 = run_cached(orig_p65, "orig-tail63", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 64);
    // Matched-split fresh zero-restore control (63+1+1): MUST match chained 63->64->65 bit-for-bit (16/16)!
    ingine::model::set_batch_prefill_splits_for_test({63, 1, 1});
    auto orig_65_fresh_63_1_1 = run_cached(orig_p65, "orig-tail63", 0, st, /*bypass=*/true);
    ingine::model::set_batch_prefill_splits_for_test({});
    CHECK(st.reuse == ingine::model::Reuse::None && st.reused_tokens == 0);
    if (orig_65_chain_63_64_65 != orig_65_fresh_63_1_1)
      print_vec_diff("orig_63->64->65_vs_fresh_63+1+1", orig_65_chain_63_64_65, orig_65_fresh_63_1_1);
    CHECK(orig_65_chain_63_64_65 == orig_65_fresh_63_1_1);
    // Re-running orig_p65 as an exact warm hit must reproduce the chained 63->64->65 continuation bit-for-bit.
    auto orig_65_chain_exact = run_cached(orig_p65, "orig-tail63", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 65);
    CHECK(orig_65_chain_exact == orig_65_chain_63_64_65);
    std::printf("orig 512-chunk diagnostic (alpha beta gamma): 63->64->65 vs fresh_63+1+1=%d/16, "
                "64->65 vs matched-split64=%d/16, 64->65 vs unsplit512=%d/16, 63->64->65 vs unsplit512=%d/16, "
                "split64 vs unsplit512 (both cold!)=%d/16\n",
                count_matching(orig_65_chain_63_64_65, orig_65_fresh_63_1_1),
                count_matching(orig_65_from_64, ref_orig_p65_s64), count_matching(orig_65_from_64, ref_unsplit512[1]),
                count_matching(orig_65_chain_63_64_65, ref_unsplit512[1]),
                count_matching(ref_orig_p65_s64, ref_unsplit512[1]));

    // 2c. Page+tail restoration (65->129, F=1, tail=1) compared WHOLE 16-token continuation against matched-split65 cold!
    (void)run_cached(orig_p65, "orig-tail65", 0, st);
    auto orig_129_from_65 = run_cached(orig_p129, "orig-tail65", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 65);
    if (orig_129_from_65 != ref_split65[0])
      print_vec_diff("orig_65->129_vs_split65", orig_129_from_65, ref_split65[0]);
    CHECK(orig_129_from_65 == ref_split65[0]);

    (void)run_cached(p65, "tech-tail65", 0, st);
    auto tech_129_from_65 = run_cached(p129_a, "tech-tail65", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 65);
    if (tech_129_from_65 != ref_split65[1])
      print_vec_diff("tech_65->129_vs_split65", tech_129_from_65, ref_split65[1]);
    CHECK(tech_129_from_65 == ref_split65[1]);

    // 2d. Exact hit, namespace isolation, and explicit bypass_prefix_cache isolation.
    auto out_63_warm = run_cached(p63, "tech-tail63", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 63 &&
          st.restored_bytes.has_value() && *st.restored_bytes > 0 && st.restore_seconds.has_value() &&
          *st.restore_seconds > 0.0);
    CHECK(out_63_warm == ref_p63);

    auto out_63_other_ns = run_cached(p63, "ns-other", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::None && st.reused_tokens == 0 &&
          st.restored_bytes.has_value() && *st.restored_bytes == 0);
    CHECK(out_63_other_ns == ref_p63);

    auto out_63_bypass = run_cached(p63, "tech-tail63", 0, st, /*bypass=*/true);
    CHECK(st.reuse == ingine::model::Reuse::None && st.reused_tokens == 0);
    CHECK(out_63_bypass == ref_p63);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 3. Cache-ON (prefill_chunk = 64) for matched-prompt wraparound (>1024), mid-prefill root64, B=4 forks,
  //    recoverable failure cleanup, and early slot cancellation/recycle.
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

    ingine::model::GenerateStats st;

    // 3a. Page-aligned chain (64->65, cold 128 w/ root_prefix_tokens=64, 128->129).
    CHECK(run_cached(p64, "ns-aligned", 0, st) == ref_p64);
    CHECK(run_cached(p65, "ns-aligned", 0, st) == ref_p65);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 64);

    CHECK(run_cached(p128_a, "ns-root64", 64, st) == ref_p128_a);
    CHECK(st.reuse == ingine::model::Reuse::None && st.reused_tokens == 0);
    CHECK(run_cached(p129_a, "ns-root64", 64, st) == ref_p129);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 128);

    // 3b. Sliding-ring wraparound (>1024) with MATCHED 1128-token prompt timing (cold_1128 vs warm_1128)
    //     and true CUDA-event GPU restore_seconds!
    (void)run_cached(p_wrap_prefix, "ns-wrap", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::None && st.reused_tokens == 0);
    auto out_wrap_hit = run_cached(p_wrap_full, "ns-wrap", 0, st);
    CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 1088 &&
          st.restored_bytes.has_value() && *st.restored_bytes > 0 && st.restore_seconds.has_value() &&
          *st.restore_seconds > 0.0);
    if (out_wrap_hit != ref_wrap) print_vec_diff("wrap_1088_to_1128", out_wrap_hit, ref_wrap);
    const double warm_1128_prefill_ms = st.prefill_seconds * 1e3;
    const double restore_gpu_ms = st.restore_seconds.value_or(0.0) * 1e3;
    std::printf("wrap (>1024) matched-1128 parity: %d/%zu tokens matched (reused=%d, restored_bytes=%llu, "
                "cold_1128_prefill=%.1fms, warm_1128_prefill=%.1fms, speedup=%.2fx, gpu_event_restore=%.2fms)\n",
                count_matching(out_wrap_hit, ref_wrap), ref_wrap.size(), st.reused_tokens,
                static_cast<unsigned long long>(st.restored_bytes.value_or(0)), cold_1128_prefill_ms,
                warm_1128_prefill_ms, cold_1128_prefill_ms / std::max(warm_1128_prefill_ms, 0.01), restore_gpu_ms);
    CHECK(out_wrap_hit == ref_wrap);

    // 3c. Concurrent B=4 forks sharing committed pages + verified root64 via Scheduler::loop_batched()
    ingine::engine::Scheduler sched(*gen, stop, 16);
    std::vector<std::vector<int32_t>> fork_prompts = {p128_a, p128_b, p129_a, p64};
    std::vector<std::vector<int32_t>> fork_refs = {ref_p128_a, ref_p128_b, ref_p129, ref_p64};
    std::vector<std::shared_ptr<ingine::engine::Job>> jobs;
    for (const auto& fp : fork_prompts) {
      auto j = std::make_shared<ingine::engine::Job>(fp, gen_tokens, greedy, std::vector<ingine::model::PromptImage>{},
                                                     64, "ns-root64");
      CHECK(sched.submit(j));
      jobs.push_back(std::move(j));
    }
    for (std::size_t i = 0; i < jobs.size(); ++i) {
      const auto out = drain(*jobs[i]);
      const auto pst = jobs[i]->prefill_stats();
      CHECK(pst.has_value() && pst->reuse == ingine::model::Reuse::PagedPrefix && pst->reused_tokens >= 64);
      if (out != fork_refs[i]) print_vec_diff("fork", out, fork_refs[i]);
      CHECK(out == fork_refs[i]);
    }

    // 3d. Early cancellation & slot recycle + recoverable failure cleanup (throwing on_prefill + OOM during snapshot).
    {
      auto cancel_job = std::make_shared<ingine::engine::Job>(p129_a, 64, greedy,
                                                              std::vector<ingine::model::PromptImage>{}, 64,
                                                              "ns-root64");
      CHECK(sched.submit(cancel_job));
      auto b = cancel_job->take(std::chrono::seconds(30));
      CHECK(!b.tokens.empty() && !b.failure);
      cancel_job->cancel();
      auto pst = cancel_job->wait_prefill_stats();
      CHECK(pst.has_value() && pst->reuse == ingine::model::Reuse::PagedPrefix && pst->reused_tokens >= 64);

      auto follow_job = std::make_shared<ingine::engine::Job>(p128_a, gen_tokens, greedy,
                                                              std::vector<ingine::model::PromptImage>{}, 64,
                                                              "ns-root64");
      CHECK(sched.submit(follow_job));
      CHECK(drain(*follow_job) == ref_p128_a);
    }

    // Recoverable failure: throwing on_prefill callback must release slot & active_refs cleanly!
    {
      bool threw = false;
      try {
        ingine::model::Prompt pr{p128_a, {}, 64, "ns-root64"};
        (void)gen->generate(
            pr, gen_tokens, greedy, stop, [](std::span<const int32_t>) { return true; },
            [](const ingine::model::GenerateStats&) { throw std::runtime_error("injected on_prefill failure"); });
      } catch (const std::runtime_error&) {
        threw = true;
      }
      CHECK(threw);
      CHECK(run_cached(p128_a, "ns-root64", 64, st) == ref_p128_a);
      CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 128);
    }

    // Verify mid-prefill root64 snapshot was tracked before the OOM pressure-release test clears the cache.
    CHECK(gen->cache_stats().root_entries >= 1);

    // Recoverable failure: injected device allocation failure during checkpoint snapshot must release cache
    // gracefully without failing the live request, and subsequent caching recovers cleanly!
    {
      ingine::fail_next_allocations(1);
      auto out_oom = run_cached(p192, "ns-oom", 0, st);
      CHECK(out_oom == ref_p192);
      const auto post_oom = gen->cache_stats();
      CHECK(post_oom.vram_pressure_releases >= 1);
      // Re-prime and hit after pressure release to prove slot & cache fully recovered.
      CHECK(run_cached(p192, "ns-oom", 0, st) == ref_p192);
      CHECK(run_cached(p192, "ns-oom", 0, st) == ref_p192);
      CHECK(st.reuse == ingine::model::Reuse::PagedPrefix && st.reused_tokens == 192);
    }

    const auto cstats = gen->cache_stats();
    std::printf("batch_prefix_parity cache_stats: entries=%llu root=%llu vram_used=%zu vram_peak=%zu d2d_restore=%llu "
                "pressure_releases=%llu\n",
                static_cast<unsigned long long>(cstats.vram_entries),
                static_cast<unsigned long long>(cstats.root_entries), cstats.vram_used_bytes, cstats.vram_peak_bytes,
                static_cast<unsigned long long>(cstats.d2d_restore_bytes),
                static_cast<unsigned long long>(cstats.vram_pressure_releases));
    CHECK(cstats.vram_entries > 0 && cstats.d2d_restore_bytes > 0);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 4. Tight 6-page KV pool (max_ctx = 384, max_kv_tokens = 384 -> exactly 6 pages total):
  //    4a. Prompt-aware Scheduler admission with overlapping live 192-tok identical siblings:
  //        Prime p192 (3 shared committed pages). Submit 2 identical p192 jobs (gen_tokens = 24) simultaneously:
  //        Sibling 1 takes 3 shared + 1 private = 4 pages (2 free). Sibling 2 needs only 1 fresh page (<= 2 free)
  //        and MUST be admitted concurrently (slot 0 + slot 1 active together) without stalling!
  //    4b. Live-sibling eviction under pressure: prime tight-b (2 pages) + tight-a (2 pages), run live tight-a
  //        sibling alongside unrelated 160-tok prompt -> evicts tight-b cleanly (vram_evictions > 0).
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

    auto run_plain = [&](const std::vector<int32_t>& ids, const std::string& ns) {
      std::vector<int32_t> out;
      ingine::model::Prompt pr{ids, {}, 0, ns};
      (void)gen->generate(pr, gen_tokens, greedy, stop, [&](std::span<const int32_t> t) {
        out.insert(out.end(), t.begin(), t.end());
        return true;
      });
      return out;
    };

    // 4a. Overlapping live 192-tok identical siblings on a 6-page pool via ordinary Scheduler path!
    CHECK(run_plain(p192, "overlap-192") == ref_p192);
    {
      ingine::engine::Scheduler sched(*gen, stop, 8);
      auto j1 = std::make_shared<ingine::engine::Job>(p192, 24, greedy, std::vector<ingine::model::PromptImage>{}, 0,
                                                      "overlap-192");
      auto j2 = std::make_shared<ingine::engine::Job>(p192, 24, greedy, std::vector<ingine::model::PromptImage>{}, 0,
                                                      "overlap-192");
      CHECK(sched.submit(j1));
      CHECK(sched.submit(j2));
      // Drain token 1 of j1, then wait for j2 to ALSO start while j1 is still live (overlapping admission!).
      const auto t_start_before_j1_done = std::chrono::steady_clock::now();
      (void)t_start_before_j1_done;
      const auto out1 = drain(*j1);
      const auto t_j1_done = std::chrono::steady_clock::now();
      const auto out2 = drain(*j2);
      const auto st1 = j1->prefill_stats();
      const auto st2 = j2->prefill_stats();
      CHECK(st1.has_value() && st1->reuse == ingine::model::Reuse::PagedPrefix && st1->reused_tokens == 192);
      CHECK(st2.has_value() && st2->reuse == ingine::model::Reuse::PagedPrefix && st2->reused_tokens == 192);
      CHECK(out1 == out2);
      CHECK(std::equal(ref_p192.begin(), ref_p192.end(), out1.begin()));
      // Verify j2 started BEFORE j1 finished (i.e. prompt-aware admission admitted both concurrently on 6 pages!).
      CHECK(j2->t_start().has_value() && *j2->t_start() < t_j1_done);
      std::printf("tight 6-page overlapping 192-tok siblings (24 tok): overlapped=%s reused=%d/%d\n",
                  (j2->t_start().has_value() && *j2->t_start() < t_j1_done) ? "YES" : "NO", st1->reused_tokens,
                  st2->reused_tokens);
    }

    // 4a-long. Long-decode sibling commitment on the same 6-page pool (96 tokens, stopless):
    // Each 192-tok request needs 3 shared + 2 private pages = 5 pages -> two concurrent 96-tok siblings would
    // need 3 + 2 + 2 = 7 > 6 pages! Admission MUST safely defer the 2nd sibling until the 1st finishes so BOTH
    // complete all 96/96 tokens matching solo output with reused=192 and zero "paged KV pool exhausted" errors!
    {
      std::vector<int32_t> solo_96;
      {
        ingine::model::Prompt pr{p192, {}, 0, "overlap-192-96"};
        (void)gen->generate(pr, 96, greedy, {}, [&](std::span<const int32_t> t) {
          solo_96.insert(solo_96.end(), t.begin(), t.end());
          return true;
        });
      }
      CHECK(solo_96.size() == 96);

      ingine::engine::Scheduler sched(*gen, {}, 8);  // stopless so all 96 tokens are generated
      auto j1 = std::make_shared<ingine::engine::Job>(p192, 96, greedy, std::vector<ingine::model::PromptImage>{}, 0,
                                                      "overlap-192-96");
      auto j2 = std::make_shared<ingine::engine::Job>(p192, 96, greedy, std::vector<ingine::model::PromptImage>{}, 0,
                                                      "overlap-192-96");
      CHECK(sched.submit(j1));
      CHECK(sched.submit(j2));
      const auto out1 = drain(*j1);
      const auto out2 = drain(*j2);
      const auto st1 = j1->prefill_stats();
      const auto st2 = j2->prefill_stats();
      CHECK(st1.has_value() && st1->reuse == ingine::model::Reuse::PagedPrefix && st1->reused_tokens == 192);
      CHECK(st2.has_value() && st2->reuse == ingine::model::Reuse::PagedPrefix && st2->reused_tokens == 192);
      CHECK(out1 == solo_96 && out2 == solo_96);
      std::printf("tight 6-page 96-tok siblings (safe deferral): emitted=%zu/%zu reused=%d/%d exact_vs_solo=%s\n",
                  out1.size(), out2.size(), st1 ? st1->reused_tokens : -1, st2 ? st2->reused_tokens : -1,
                  (out1 == solo_96 && out2 == solo_96) ? "YES" : "NO");
    }

    // 4b. Tight-pool live-sibling eviction under pressure.
    auto p_alt_b = pad_to_len(render("Distinct topic about distributed consensus algorithms."), 128, filler);
    auto p_pressure = pad_to_len(render("Unrelated long prompt forcing KV page eviction under pressure."), 160, filler);
    (void)run_plain(p_alt_b, "tight-b");
    CHECK(run_plain(p128_a, "tight-a") == ref_p128_a);

    ingine::engine::Scheduler sched(*gen, stop, 8);
    auto j_sib = std::make_shared<ingine::engine::Job>(p128_a, gen_tokens, greedy,
                                                       std::vector<ingine::model::PromptImage>{}, 0, "tight-a");
    auto j_press = std::make_shared<ingine::engine::Job>(p_pressure, gen_tokens, greedy,
                                                         std::vector<ingine::model::PromptImage>{}, 0, "tight-c");
    CHECK(sched.submit(j_sib));
    CHECK(sched.submit(j_press));
    const auto out_sib = drain(*j_sib);
    const auto out_press = drain(*j_press);
    CHECK(out_sib == ref_p128_a);
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

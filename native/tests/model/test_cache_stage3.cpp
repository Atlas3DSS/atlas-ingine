// Stage 3 (#826) full-model verification & restore-vs-recompute latency crossover benchmark:
//   1. Verified root-prefix checkpoint on real ChatFormat-rendered system+tools prompts:
//      - Fork diverging immediately after the root boundary (Reuse::RootCheckpoint, exact greedy parity)
//      - Exact-root request (n == R, 0 tokens prefilled, exact greedy parity, even with root hint=0)
//      - Child conversation extension past R never erases the root checkpoint
//      - Tenant/branch cache_namespace isolation (fail-closed miss)
//      - Symmetric --no-prefix-cache toggle (cache-off parity baseline)
//   2. Bounded pinned host-RAM parking tier (park_host_mb):
//      - Direct host-only parking (park_mb=0, park_host_mb>0): Reuse::HostParked, 0 VRAM parked, exact parity
//      - Strict byte cap + LRU eviction (host_used <= host_budget, host_peak <= host_budget, evictions > 0)
//      - Tier-0 VRAM -> Tier-1 pinned host RAM demotion on VRAM LRU eviction
//      - Graceful fallback under host pinned OOM (fail_next_pinned_allocations) and VRAM OOM (host entries survive)
//      - DSpark --spec parity with host-parked root + fork
//   3. Measured restore-vs-recompute latency crossover sweep across L in {256, 512, 1024, 2048, 4096, 8192, 16384}.
#include "common/test_util.h"
#include "core/arena.h"
#include "ingine/generator.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace ingine::model;
using Json = nlohmann::ordered_json;

namespace {

struct RunOut {
  std::vector<std::int32_t> tokens;
  GenerateStats stats;
  GenerateStats prefill_cb_stats;
  bool prefill_cb_before_tokens = true;
  bool saw_prefill_cb = false;
};

RunOut run_prompt(Generator& gen, const Prompt& p, int max_new, const Sampling& sp) {
  RunOut o;
  bool saw_any_tok = false;
  o.stats = gen.generate(
      p, max_new, sp, {},
      [&](std::span<const std::int32_t> t) {
        saw_any_tok = true;
        o.tokens.insert(o.tokens.end(), t.begin(), t.end());
        return true;
      },
      [&](const GenerateStats& st) {
        o.saw_prefill_cb = true;
        o.prefill_cb_stats = st;
        if (saw_any_tok) o.prefill_cb_before_tokens = false;
      });
  return o;
}

std::size_t current_vram_used_bytes() {
  std::size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  return total_b > free_b ? (total_b - free_b) : 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  std::filesystem::path dir = INGINE_MODEL_DIR;
  bool run_crossover = true;
  bool use_kv_q4 = false;
  int offload_vram_mb = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) dir = argv[++i];
    else if (std::strcmp(argv[i], "--no-crossover") == 0) run_crossover = false;
    else if (std::strcmp(argv[i], "--kv-q4") == 0) use_kv_q4 = true;
    else if (std::strcmp(argv[i], "--offload-vram-mb") == 0 && i + 1 < argc) offload_vram_mb = std::atoi(argv[++i]);
  }
  if (!std::filesystem::exists(dir / "config.json")) {
    std::printf("skip: no model at %s\n", dir.c_str());
    return ingine::test::kSkip;
  }

  const auto fmt = ingine::text::ChatFormat::load(dir);
  Sampling greedy;
  greedy.temperature = 0.f;

  // Build a realistic system + tools root (~600-900 tokens) and two forks diverging immediately after root.
  std::string sys_text =
      "You are an autonomous software engineering agent operating inside a verified build harness.\n"
      "Follow these immutable architectural rules strictly on every turn:\n";
  for (int i = 0; i < 40; ++i) {
    sys_text += "Rule " + std::to_string(i + 1) +
                ": Always verify compiled artifacts, exact token boundaries, memory budgets, and parity gates "
                "before publishing any cache entry or claiming performance improvements.\n";
  }
  const Json tools = Json::array(
      {{{"type", "function"},
        {"function",
         {{"name", "read_file"},
          {"description", "Read UTF-8 file contents by line range from the workspace."},
          {"parameters",
           {{"type", "object"},
            {"properties",
             {{"path", {{"type", "string"}}}, {"start_line", {{"type", "integer"}}}, {"end_line", {{"type", "integer"}}}}},
            {"required", Json::array({"path"})}}}}}},
       {{"type", "function"},
        {"function",
         {{"name", "run_tests"},
          {"description", "Execute a CTest filter and return stdout, stderr, and exit code."},
          {"parameters",
           {{"type", "object"},
            {"properties", {{"filter", {{"type", "string"}}}, {"timeout_s", {{"type", "integer"}}}}},
            {"required", Json::array({"filter"})}}}}}}});

  const auto render_fork = [&](const std::string& user_msg) {
    return fmt.render(Json::array({{{"role", "system"}, {"content", sys_text}},
                                   {{"role", "user"}, {"content", user_msg}}}),
                      tools, ingine::text::ChatOptions{});
  };
  const auto rp_a = render_fork("Analyze cache eviction policies in native/src/model/generator.cpp and list risks.");
  const auto rp_b = render_fork("Explain how sliding-window rings interact with full-attention KV row restore.");
  const auto rp_c = render_fork("Write a short C++ helper to verify token prefix equality without hash trust.");
  const auto rp_root = fmt.render(Json::array({{{"role", "system"}, {"content", sys_text}}}), tools,
                                  ingine::text::ChatOptions{});
  const int R = rp_a.root_prefix_tokens;
  std::printf("[setup] rendered root_prefix_tokens=%d | fork_a=%zu fork_b=%zu fork_c=%zu exact_root=%zu\n", R,
              rp_a.ids.size(), rp_b.ids.size(), rp_c.ids.size(), rp_root.ids.size());

  bool ok = true;
  auto check = [&](bool cond, const char* label) {
    std::printf("%s %s\n", cond ? "PASS" : "FAIL", label);
    ok &= cond;
  };
  check(R >= 256 && R == rp_b.root_prefix_tokens && R == rp_c.root_prefix_tokens && R == int(rp_root.ids.size()) &&
            std::equal(rp_root.ids.begin(), rp_root.ids.end(), rp_a.ids.begin()) &&
            std::equal(rp_root.ids.begin(), rp_root.ids.end(), rp_b.ids.begin()),
        "rendered prompts share exact verified root boundary");

  RuntimeOptions base_opt;
  base_opt.model_dir = dir;
  base_opt.max_ctx = 20480;
  base_opt.prefill_chunk = 512;  // R (~700-900) straddles chunk=512, exercising chunk-split root capture!
  base_opt.dense_fp8 = true;
  base_opt.kv_fp8 = !use_kv_q4;
  base_opt.kv_q4 = use_kv_q4;
  base_opt.spec = false;
  base_opt.offload_vram_mb = offload_vram_mb;

  // ---- Reference outputs (strict exact-root -> Extend parity oracle, no parking) ----
  RunOut ref_fork_b, ref_fork_c, ref_exact_root;
  {
    RuntimeOptions opt = base_opt;
    opt.park_mb = 0;
    opt.park_host_mb = 0;
    auto gen = make_generator(opt);
    ref_exact_root = run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 32, greedy);
    // Strict Extend from exact-root prefill (0 generated tokens so state is at exact R) -> same chunk boundaries as root restore.
    run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 0, greedy);
    ref_fork_b = run_prompt(*gen, Prompt{rp_b.ids, {}, 0}, 32, greedy);
    // Reset with unrelated prompt, then exact-root -> Extend for fork C.
    const std::vector<std::int32_t> noise(64, 1234);
    run_prompt(*gen, Prompt{noise, {}, 0}, 1, greedy);
    run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 0, greedy);
    ref_fork_c = run_prompt(*gen, Prompt{rp_c.ids, {}, 0}, 32, greedy);
    check(ref_fork_b.stats.reuse == Reuse::Extend && ref_fork_b.stats.reused_tokens == R &&
              ref_fork_c.stats.reuse == Reuse::Extend && ref_fork_c.stats.reused_tokens == R,
          "reference oracle established strict Extend from exact root R");
  }

  // ---- Suite 1: B=1 Verified Root-Prefix Checkpoint (VRAM tier), forks, exact-root, child extension, namespace & --no-prefix-cache ----
  {
    RuntimeOptions opt = base_opt;
    opt.park_mb = 1024;
    opt.park_host_mb = 0;
    auto gen = make_generator(opt);

    const RunOut oa = run_prompt(*gen, Prompt{rp_a.ids, {}, R}, 16, greedy);
    const PrefixCacheStats cs1 = gen->cache_stats();
    check(oa.stats.reuse == Reuse::None && oa.stats.reused_tokens == 0 && cs1.root_entries == 1 &&
              oa.saw_prefill_cb && oa.prefill_cb_before_tokens && oa.prefill_cb_stats.reused_tokens == 0,
          "Suite 1a: cold fork A captures verified root checkpoint (root_entries=1, prefill_cb before tokens)");

    // Fork B diverges immediately after R -> must hit RootCheckpoint with exact greedy parity!
    const RunOut ob = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 32, greedy);
    check(ob.stats.reuse == Reuse::RootCheckpoint && ob.stats.reused_tokens == R &&
              ob.stats.restored_bytes.value_or(0) > 0 && ob.tokens == ref_fork_b.tokens && ob.saw_prefill_cb &&
              ob.prefill_cb_before_tokens && ob.prefill_cb_stats.reused_tokens == R &&
              ob.prefill_cb_stats.reuse == Reuse::RootCheckpoint,
          "Suite 1b: fork B diverging right after R hits Reuse::RootCheckpoint with exact greedy parity");

    // Extend fork A past R (simulating multi-turn child growth) -> root checkpoint MUST NOT be erased as superseded!
    std::vector<std::int32_t> child_a = rp_a.ids;
    child_a.insert(child_a.end(), oa.tokens.begin(), oa.tokens.end());
    const std::vector<std::int32_t> turn2_tail(60, 4242);
    child_a.insert(child_a.end(), turn2_tail.begin(), turn2_tail.end());
    const RunOut oca = run_prompt(*gen, Prompt{child_a, {}, R}, 8, greedy);
    const PrefixCacheStats cs_child = gen->cache_stats();
    check(oca.stats.reuse == Reuse::Parked && oca.stats.reused_tokens == int(rp_a.ids.size()) &&
              cs_child.root_entries == 1,
          "Suite 1c: child A resumes from parked A and extending past R preserves root checkpoint (root_entries=1)");

    // Fork C diverging right after R STILL hits the preserved root checkpoint!
    const RunOut oc = run_prompt(*gen, Prompt{rp_c.ids, {}, R}, 32, greedy);
    check(oc.stats.reuse == Reuse::RootCheckpoint && oc.stats.reused_tokens == R && oc.tokens == ref_fork_c.tokens,
          "Suite 1d: fork C hits preserved root checkpoint after sibling child growth (exact greedy parity)");

    // Exact-root request (n == R, even with root_prefix_tokens=0 so lookup uses exact token IDs, never trusting hint)
    const RunOut oroot = run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 32, greedy);
    check(oroot.stats.reuse == Reuse::RootCheckpoint && oroot.stats.reused_tokens == R &&
              oroot.tokens == ref_exact_root.tokens,
          "Suite 1e: exact-root request (hint=0) hits RootCheckpoint with 0 tokens prefilled & exact parity");

    // Namespace isolation: same tokens under a different cache_namespace MUST miss (fail-closed)
    const RunOut ons = run_prompt(*gen, Prompt{rp_b.ids, {}, R, "tenant-isolated"}, 8, greedy);
    check(ons.stats.reuse == Reuse::None && ons.stats.reused_tokens == 0,
          "Suite 1f: different cache_namespace fails closed to Reuse::None");

    std::printf("[Suite 1 stats] %s\n", gen->offload_summary().c_str());
  }

  // ---- Suite 1g: Symmetric --no-prefix-cache toggle (cache-off parity baseline) ----
  {
    RuntimeOptions opt = base_opt;
    opt.prefix_cache = false;
    opt.park_mb = 1024;
    opt.park_host_mb = 1024;
    auto gen = make_generator(opt);
    const RunOut r1 = run_prompt(*gen, Prompt{rp_root.ids, {}, R}, 16, greedy);
    const RunOut r2 = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 16, greedy);
    const PrefixCacheStats cs = gen->cache_stats();
    check(r1.stats.reuse == Reuse::None && r2.stats.reuse == Reuse::None && r2.stats.reused_tokens == 0 &&
              cs.vram_used_bytes == 0 && cs.host_used_bytes == 0 && cs.root_entries == 0,
          "Suite 1g: prefix_cache=false disables all reuse & parking cleanly (cache-off baseline)");
  }

  // ---- Suite 2: Bounded Pinned Host-RAM Parking (park_mb=0, LRU eviction, VRAM->Host demotion, OOM fallback) ----
  {
    // 2a: Direct host-only parking (0 VRAM parked) + exact fork & exact-root parity + LRU eviction
    RuntimeOptions opt = base_opt;
    opt.park_mb = 0;
    opt.park_host_mb = 450;  // fits ~2 checkpoints (~175-190 MiB each), forces LRU eviction on 3rd!
    auto gen = make_generator(opt);

    const RunOut ha = run_prompt(*gen, Prompt{rp_a.ids, {}, R}, 8, greedy);
    const RunOut hb = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 32, greedy);
    const PrefixCacheStats cs_h1 = gen->cache_stats();
    check(ha.stats.reuse == Reuse::None && hb.stats.reuse == Reuse::HostParked && hb.stats.reused_tokens == R &&
              hb.stats.restored_bytes.value_or(0) > 0 && hb.stats.restore_seconds.value_or(0.0) > 0.0 &&
              hb.tokens == ref_fork_b.tokens && cs_h1.vram_used_bytes == 0 && cs_h1.vram_peak_bytes == 0 &&
              cs_h1.host_used_bytes <= cs_h1.host_budget_bytes && cs_h1.host_peak_bytes <= cs_h1.host_budget_bytes,
          "Suite 2a: direct host-only parking (park_mb=0) hits Reuse::HostParked with 0 VRAM parked & exact parity");

    // Exact-root from pinned host RAM
    const RunOut hroot = run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 32, greedy);
    check(hroot.stats.reuse == Reuse::HostParked && hroot.stats.reused_tokens == R &&
              hroot.tokens == ref_exact_root.tokens,
          "Suite 2b: exact-root request hits Reuse::HostParked with exact greedy parity");

    // Force host LRU eviction with 3 unrelated 400-token conversations (each ~170 MiB > 450 MiB for 3)
    ingine::test::Rng rng(77);
    auto make_conv = [&](int len) {
      std::vector<std::int32_t> v(static_cast<std::size_t>(len));
      for (auto& x : v) x = std::int32_t(1000 + rng.next() % 50000);
      return v;
    };
    const auto u1 = make_conv(400), u2 = make_conv(400), u3 = make_conv(400), u4 = make_conv(400);
    run_prompt(*gen, Prompt{u1, {}, 0}, 4, greedy);
    run_prompt(*gen, Prompt{u2, {}, 0}, 4, greedy);  // parks u1 in host
    run_prompt(*gen, Prompt{u3, {}, 0}, 4, greedy);  // parks u2 in host (evicts older LRU)
    run_prompt(*gen, Prompt{u4, {}, 0}, 4, greedy);  // parks u3 in host (evicts u1!)
    // Follow-up on u3 (MRU) should HIT host; follow-up on u1 (LRU evicted) should MISS!
    auto u3_follow = u3;
    u3_follow.push_back(5555);
    auto u1_follow = u1;
    u1_follow.push_back(5555);
    const RunOut hu3 = run_prompt(*gen, Prompt{u3_follow, {}, 0}, 4, greedy);
    const RunOut hu1 = run_prompt(*gen, Prompt{u1_follow, {}, 0}, 4, greedy);
    const PrefixCacheStats cs_ev = gen->cache_stats();
    check(hu3.stats.reuse == Reuse::HostParked && hu3.stats.reused_tokens == 400 &&
              hu1.stats.reuse == Reuse::None && cs_ev.host_evictions > 0 &&
              cs_ev.host_used_bytes <= cs_ev.host_budget_bytes && cs_ev.host_peak_bytes <= cs_ev.host_budget_bytes,
          "Suite 2c: bounded host-RAM tier enforces byte cap with deterministic LRU eviction");

    // Graceful fallback under injected host pinned allocation failure (request MUST succeed with exact output!)
    ingine::fail_next_pinned_allocations(2);
    const RunOut hfail = run_prompt(*gen, Prompt{rp_c.ids, {}, R}, 8, greedy);
    const PrefixCacheStats cs_fail = gen->cache_stats();
    check(hfail.tokens.size() == 8 && cs_fail.host_alloc_failures >= 1,
          "Suite 2d: host pinned OOM degrades gracefully (host_alloc_failures>=1, request succeeds)");

    std::printf("[Suite 2a-2d stats] %s\n", gen->offload_summary().c_str());
  }

  // 2e: Tier-0 VRAM -> Tier-1 Host demotion + VRAM OOM pressure preserves Host entries
  {
    RuntimeOptions opt = base_opt;
    opt.park_mb = 240;       // fits exactly 1 checkpoint in VRAM (~175-190 MiB)
    opt.park_host_mb = 1024; // demoted entries land in pinned host RAM
    auto gen = make_generator(opt);

    run_prompt(*gen, Prompt{rp_a.ids, {}, R}, 4, greedy);  // captures root in VRAM (slot 1/1), live=A
    // An unrelated prompt has no parked match (keep=nullptr), so parking live A into Tier-0 VRAM (240 MiB)
    // evicts the LRU Tier-0 entry (root) and demotes it to Tier-1 pinned host RAM.
    std::vector<std::int32_t> unrelated(300, 9999);
    unrelated[0] = 8888;
    run_prompt(*gen, Prompt{unrelated, {}, 0}, 4, greedy);
    const PrefixCacheStats cs_dem = gen->cache_stats();
    check(cs_dem.host_demotions >= 1 && cs_dem.vram_entries == 1 && cs_dem.host_entries >= 1 &&
              cs_dem.root_entries >= 1 && cs_dem.vram_peak_bytes <= cs_dem.vram_budget_bytes &&
              cs_dem.host_peak_bytes <= cs_dem.host_budget_bytes,
          "Suite 2e: Tier-0 VRAM LRU eviction demotes checkpoint to Tier-1 pinned host RAM");

    // Inject GPU VRAM allocation failure while both Tier-0 VRAM (A) and Tier-1 Host (demoted root) exist:
    // VRAM entries release, Host entries SURVIVE (0 bytes of VRAM), and fork C still hits demoted root!
    const std::uint64_t host_before = cs_dem.host_entries;
    Sampling forced = greedy;
    const std::vector<std::int32_t> forced_toks = {100, 101, 102, 103};
    forced.forced = forced_toks;
    ingine::fail_next_allocations(1);
    const RunOut opress = run_prompt(*gen, Prompt{unrelated, {}, 0}, 4, forced);
    const PrefixCacheStats cs_press = gen->cache_stats();
    const RunOut post_oom_c = run_prompt(*gen, Prompt{rp_c.ids, {}, R}, 32, greedy);
    check(opress.tokens == forced_toks && cs_press.vram_pressure_releases >= 1 && cs_press.vram_entries == 0 &&
              cs_press.host_entries == host_before && post_oom_c.stats.reuse == Reuse::HostParked &&
              post_oom_c.stats.reused_tokens == R && post_oom_c.tokens == ref_fork_c.tokens,
          "Suite 2f: VRAM OOM releases Tier-0 VRAM parked entries while preserving Tier-1 Host entries & root parity");

    std::printf("[Suite 2e-2f stats] %s\n", gen->offload_summary().c_str());
  }

  // ---- Suite 2g: DSpark --spec with host-parked root + fork parity ----
  if (std::filesystem::exists(dir / "mtp.safetensors")) {
    RuntimeOptions opt = base_opt;
    opt.spec = true;
    opt.park_mb = 0;
    opt.park_host_mb = 0;
    RunOut spec_ref_b;
    {
      auto gen = make_generator(opt);
      run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 0, greedy);
      spec_ref_b = run_prompt(*gen, Prompt{rp_b.ids, {}, 0}, 32, greedy);
    }
    opt.park_host_mb = 1024;
    auto gen = make_generator(opt);
    run_prompt(*gen, Prompt{rp_a.ids, {}, R}, 8, greedy);
    const RunOut spec_hb = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 32, greedy);
    check(spec_hb.stats.reuse == Reuse::HostParked && spec_hb.stats.reused_tokens == R &&
              spec_hb.tokens == spec_ref_b.tokens,
          "Suite 2g: DSpark --spec root capture + HostParked fork restore exact greedy parity");
    std::printf("[Suite 2g spec stats] %s\n", gen->offload_summary().c_str());
  }

  // ---- Suite 3: Measured Restore-vs-Recompute Latency Crossover Sweep ----
  if (run_crossover) {
    RuntimeOptions opt = base_opt;
    opt.max_ctx = 20480;
    opt.prefill_chunk = 3072;
    opt.park_mb = 0;
    opt.park_host_mb = 2048;

    RuntimeOptions opt_vram = opt;
    opt_vram.park_mb = 2048;
    opt_vram.park_host_mb = 0;

    std::printf("\n=== MEASURED RESTORE-VS-RECOMPUTE CROSSOVER (kv=%s, chunk=%d) ===\n",
                use_kv_q4 ? "q4_0" : "fp8", opt.prefill_chunk);
    std::printf("%-7s | %-10s | %-11s | %-11s | %-11s | %-11s | %-10s | %-9s\n",
                "Len", "Ckpt_MiB", "Cold_ms", "VramRes_ms", "HostRes_ms", "HostEff_GBs", "Speedup_H", "Parity");

    ingine::test::Rng rng(2026);
    std::vector<std::int32_t> master(16500);
    for (auto& x : master) x = std::int32_t(1000 + rng.next() % 50000);
    const std::vector<std::int32_t> flush_ids(128, 7777);
    const std::vector<int> lengths = {256, 512, 1024, 2048, 4096, 8192, 16384};

    struct Row {
      int L = 0;
      std::uint64_t bytes = 0;
      double cold_ms = 0;
      double host_ms = 0;
      double vram_ms = 0;
      std::vector<std::int32_t> cold_tok;
      bool host_ok = false;
      bool vram_ok = false;
    };
    std::vector<Row> rows(lengths.size());
    PrefixCacheStats ch, cv;
    std::size_t vram_delta_host = 0, vram_delta_vram = 0;

    // Pass 1: Cold recompute + Tier-1 Pinned Host-RAM restore (one model resident at a time so 24 GB GPUs fit)
    {
      const std::size_t before = current_vram_used_bytes();
      auto gen_host = make_generator(opt);
      const std::size_t after = current_vram_used_bytes();
      vram_delta_host = after > before ? (after - before) : 0;
      run_prompt(*gen_host, Prompt{std::span(master).subspan(0, 256), {}, 0}, 1, greedy);  // warm up
      for (std::size_t i = 0; i < lengths.size(); ++i) {
        const int L = lengths[i];
        std::vector<std::int32_t> p(master.begin(), master.begin() + L);
        p[0] = 20000 + L;
        run_prompt(*gen_host, Prompt{flush_ids, {}, 0}, 1, greedy);
        const RunOut cold = run_prompt(*gen_host, Prompt{p, {}, 0}, 1, greedy);
        run_prompt(*gen_host, Prompt{flush_ids, {}, 0}, 1, greedy);  // parks p in pinned host RAM
        const std::uint64_t h2d_before = gen_host->cache_stats().h2d_restore_bytes;
        const RunOut hres = run_prompt(*gen_host, Prompt{p, {}, 0}, 1, greedy);
        const std::uint64_t h2d_after = gen_host->cache_stats().h2d_restore_bytes;
        rows[i].L = L;
        rows[i].bytes = h2d_after - h2d_before;  // exact PCIe H2D bytes copied for this checkpoint
        rows[i].cold_ms = cold.stats.prefill_seconds * 1e3;
        rows[i].host_ms = hres.stats.restore_seconds.value_or(0.0) * 1e3;
        rows[i].cold_tok = cold.tokens;
        rows[i].host_ok = (hres.stats.reuse == Reuse::HostParked && hres.stats.reused_tokens == L &&
                           hres.tokens == cold.tokens);
      }
      ch = gen_host->cache_stats();
    }

    // Pass 2: Tier-0 VRAM parked restore (separate model instance so peak VRAM stays single-instance)
    {
      const std::size_t before = current_vram_used_bytes();
      auto gen_vram = make_generator(opt_vram);
      const std::size_t after = current_vram_used_bytes();
      vram_delta_vram = after > before ? (after - before) : 0;
      run_prompt(*gen_vram, Prompt{std::span(master).subspan(0, 256), {}, 0}, 1, greedy);  // warm up
      for (std::size_t i = 0; i < lengths.size(); ++i) {
        const int L = lengths[i];
        std::vector<std::int32_t> p(master.begin(), master.begin() + L);
        p[0] = 20000 + L;
        run_prompt(*gen_vram, Prompt{flush_ids, {}, 0}, 1, greedy);
        run_prompt(*gen_vram, Prompt{p, {}, 0}, 1, greedy);
        run_prompt(*gen_vram, Prompt{flush_ids, {}, 0}, 1, greedy);  // parks p in Tier-0 VRAM
        const RunOut vres = run_prompt(*gen_vram, Prompt{p, {}, 0}, 1, greedy);
        rows[i].vram_ms = vres.stats.restore_seconds.value_or(0.0) * 1e3;
        rows[i].vram_ok = (vres.stats.reuse == Reuse::Parked && vres.stats.reused_tokens == L &&
                           vres.tokens == rows[i].cold_tok);
      }
      cv = gen_vram->cache_stats();
    }

    for (const Row& r : rows) {
      const double ckpt_mib = double(r.bytes) / (1024.0 * 1024.0);
      const double host_gbs = r.host_ms > 0.0 ? (double(r.bytes) / (r.host_ms * 1e6)) : 0.0;
      const double speedup = r.host_ms > 0.0 ? (r.cold_ms / r.host_ms) : 0.0;
      const bool par = r.host_ok && r.vram_ok;
      std::printf("%-7d | %10.1f | %11.2f | %11.3f | %11.3f | %11.2f | %9.2fx | %s\n", r.L, ckpt_mib, r.cold_ms,
                  r.vram_ms, r.host_ms, host_gbs, speedup, par ? "PASS" : "FAIL");
      ok &= par;
    }
    std::printf("[crossover host-tier] device_staging=%.1f MiB, vram_park_peak=%.1f MiB, host_park_peak=%.1f MiB, "
                "h2d=%.2f MB, d2h=%.2f MB, host_evictions=%llu, model_vram_delta=%.2f GiB\n",
                double(ch.device_staging_bytes) / (1024.0 * 1024.0), double(ch.vram_peak_bytes) / (1024.0 * 1024.0),
                double(ch.host_peak_bytes) / (1024.0 * 1024.0), double(ch.h2d_restore_bytes) / 1e6,
                double(ch.d2h_park_bytes) / 1e6, static_cast<unsigned long long>(ch.host_evictions),
                double(vram_delta_host) / (1024.0 * 1024.0 * 1024.0));
    std::printf("[crossover vram-tier] device_staging=%.1f MiB, vram_park_peak=%.1f MiB, host_park_peak=%.1f MiB, "
                "d2d_r=%.2f MB, d2d_p=%.2f MB, vram_evictions=%llu, model_vram_delta=%.2f GiB\n",
                double(cv.device_staging_bytes) / (1024.0 * 1024.0), double(cv.vram_peak_bytes) / (1024.0 * 1024.0),
                double(cv.host_peak_bytes) / (1024.0 * 1024.0), double(cv.d2d_restore_bytes) / 1e6,
                double(cv.d2d_park_bytes) / 1e6, static_cast<unsigned long long>(cv.vram_evictions),
                double(vram_delta_vram) / (1024.0 * 1024.0 * 1024.0));
  }

  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}

// Stage 3 (#826) full-model verification & restore-vs-recompute latency crossover benchmark:
//   1. Verified root-prefix checkpoint on real ChatFormat-rendered system+tools prompts:
//      - True immediate-divergence fork with asserted LCP(A, B) == R (Reuse::RootCheckpoint, exact greedy parity)
//      - Sibling forks diverging after R, exact-root request (n == R, hint=0, 0 prefilled), child growth past R
//      - Tenant/branch cache_namespace isolation (fail-closed miss) & Prompt::bypass_prefix_cache
//      - Symmetric --no-prefix-cache baseline (counters + true cache-off continuation comparison vs matched-boundary Extend)
//   2. Bounded pinned host-RAM parking tier (park_host_mb):
//      - Direct host-only parking (park_mb=0, park_host_mb>0): Reuse::HostParked, 0 VRAM parked, exact parity
//      - Strict byte cap + deterministic LRU eviction
//      - Simultaneous Tier-0 VRAM + Tier-1 Host pressure (demotion evicts earlier Tier-1 entry, victim preserved intact)
//      - Pin-failure rollback under simultaneous tier pressure (victim cleanly dropped, surviving state uncorrupted)
//      - VRAM OOM releasing Tier-0 while preserving Tier-1 Host entries & exact demoted root parity
//      - DSpark --spec root capture + HostParked fork parity
//      - Disk checkpoint resume & promotion gating (has_logits=false exact-root & non-promotion; --spec rejection/decline)
//      - Verified media SHA-256 identity binding (identical negative token IDs with different RGB pixels fail closed)
//   3. Measured apples-to-apples wall TTFT crossover (kReps=3 medians) across L in {64, 128, 256..16384}, with
//      separate CUDA transfer-event timing and separate prefix-cache vs device-wide VRAM reporting.
#include "common/test_util.h"
#include "core/act.h"
#include "core/arena.h"
#include "core/safetensors.h"
#include "ingine/generator.h"
#include "model/kv_checkpoint.h"
#include "model/model_data.h"
#include "model/runtime.h"
#include "text/chat.h"

#include "core/gpu_compat.h"
#include <nlohmann/json.hpp>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace ingine::model;
using Json = nlohmann::ordered_json;

#if defined(__linux__)
namespace {
constexpr std::string_view kToctouNormName = "model.llm.embed_norm.weight";
std::filesystem::path g_toctou_priv_shard;
bool g_toctou_armed = false;
bool g_toctou_flipped = false;
bool g_observe_loads = false;
std::vector<float> g_observed_norm0;
std::vector<std::optional<std::array<std::uint8_t, 32>>> g_observed_shas;
}  // namespace

extern "C" std::vector<std::byte> s3_real_st_read(const ingine::SafetensorsModel*, std::string_view)
    asm("__real__ZNK6ingine16SafetensorsModel4readESt17basic_string_viewIcSt11char_traitsIcEE");
extern "C" std::vector<std::byte> s3_wrap_st_read(const ingine::SafetensorsModel*, std::string_view)
    asm("__wrap__ZNK6ingine16SafetensorsModel4readESt17basic_string_viewIcSt11char_traitsIcEE");

std::vector<std::byte> s3_wrap_st_read(const ingine::SafetensorsModel* model, std::string_view name) {
  auto original = s3_real_st_read(model, name);
  if (!g_toctou_armed || name != kToctouNormName) return original;
  const ingine::SafetensorsRef ref = model->at(name);
  if (ref.file->path() != g_toctou_priv_shard) return original;
  auto flipped = original;
  if (ref->dtype == "BF16") {
    for (std::size_t i = 1; i < flipped.size(); i += 2) flipped[i] ^= std::byte{0x80};
  } else if (ref->dtype == "F32") {
    for (std::size_t i = 3; i < flipped.size(); i += 4) flipped[i] ^= std::byte{0x80};
  } else {
    throw std::runtime_error("unexpected embed_norm dtype for during-load TOCTOU test: " + ref->dtype);
  }
  std::fstream out(g_toctou_priv_shard, std::ios::in | std::ios::out | std::ios::binary);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.seekp(static_cast<std::streamoff>(ref->offset));
  out.write(reinterpret_cast<const char*>(flipped.data()), std::streamsize(flipped.size()));
  out.close();
  g_toctou_armed = false;
  g_toctou_flipped = true;
  return original;
}

#if defined(INGINE_BACKEND_HIP) && INGINE_BACKEND_HIP
extern "C" Weights s3_real_load_weights(const std::filesystem::path&, bool, cudaStream_t, const OffloadConfig&)
    asm("__real__ZN6ingine5model12load_weightsERKNSt10filesystem7__cxx114pathEbP12ihipStream_tRKNS0_13OffloadConfigE");
extern "C" Weights s3_wrap_load_weights(const std::filesystem::path&, bool, cudaStream_t, const OffloadConfig&)
    asm("__wrap__ZN6ingine5model12load_weightsERKNSt10filesystem7__cxx114pathEbP12ihipStream_tRKNS0_13OffloadConfigE");
#else
extern "C" Weights s3_real_load_weights(const std::filesystem::path&, bool, cudaStream_t, const OffloadConfig&)
    asm("__real__ZN6ingine5model12load_weightsERKNSt10filesystem7__cxx114pathEbP11CUstream_stRKNS0_13OffloadConfigE");
extern "C" Weights s3_wrap_load_weights(const std::filesystem::path&, bool, cudaStream_t, const OffloadConfig&)
    asm("__wrap__ZN6ingine5model12load_weightsERKNSt10filesystem7__cxx114pathEbP11CUstream_stRKNS0_13OffloadConfigE");
#endif

Weights s3_wrap_load_weights(const std::filesystem::path& dir, bool dense, cudaStream_t stream,
                             const OffloadConfig& cfg) {
  Weights w = s3_real_load_weights(dir, dense, stream, cfg);
  if (g_observe_loads) {
    float first = 0.f;
    INGINE_TEST_CUDA(cudaMemcpy(&first, w.embed_norm.data, sizeof(first), cudaMemcpyDeviceToHost));
    g_observed_norm0.push_back(first);
    g_observed_shas.push_back(w.content_sha256);
  }
  return w;
}
#endif

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

std::size_t common_prefix_len(std::span<const std::int32_t> a, std::span<const std::int32_t> b) {
  const std::size_t m = std::min(a.size(), b.size());
  std::size_t i = 0;
  while (i < m && a[i] == b[i]) ++i;
  return i;
}

double median_of(std::vector<double> vals) {
  if (vals.empty()) return 0.0;
  std::sort(vals.begin(), vals.end());
  const std::size_t mid = vals.size() / 2;
  return (vals.size() % 2 == 1) ? vals[mid] : 0.5 * (vals[mid - 1] + vals[mid]);
}

std::size_t device_wide_vram_used_bytes() {
  std::size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  return total_b > free_b ? (total_b - free_b) : 0;
}

// Create a valid disk KV checkpoint copy with has_logits=0 (no logits payload), matching kv_checkpoint.cpp RawHeader.
void strip_logits_from_kv_checkpoint(const std::filesystem::path& src, const std::filesystem::path& dst) {
  std::ifstream in(src, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open source checkpoint: " + src.string());
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const std::size_t logits_bytes = std::size_t(UNPADDED_VOCAB) * sizeof(float);
  if (bytes.size() < 4096 + logits_bytes) {
    throw std::runtime_error("source checkpoint too small to strip logits");
  }
  // RawHeader::has_logits sits at byte offset 52 (8-byte magic + 11 x 4-byte scalar fields).
  std::uint32_t zero = 0;
  std::memcpy(bytes.data() + 52, &zero, sizeof(zero));
  bytes.resize(bytes.size() - logits_bytes);
  std::ofstream out(dst, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write stripped checkpoint: " + dst.string());
  out.write(bytes.data(), std::streamsize(bytes.size()));
  out.close();
}

// Create a disk KV checkpoint copy with RawHeader::act_dtype (byte offset 16) patched to `new_act_dtype`.
void patch_act_dtype_in_kv_checkpoint(const std::filesystem::path& src, const std::filesystem::path& dst,
                                      std::uint32_t new_act_dtype) {
  std::ifstream in(src, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open source checkpoint: " + src.string());
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (bytes.size() < 4096) throw std::runtime_error("source checkpoint smaller than 4096-byte header");
  std::memcpy(bytes.data() + 16, &new_act_dtype, sizeof(new_act_dtype));
  std::ofstream out(dst, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write patched checkpoint: " + dst.string());
  out.write(bytes.data(), std::streamsize(bytes.size()));
  out.close();
}

// Create a legacy disk KV checkpoint copy with bytes 144..4095 zeroed (no v1 producer identity trailer).
void strip_producer_from_kv_checkpoint(const std::filesystem::path& src, const std::filesystem::path& dst) {
  std::ifstream in(src, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open source checkpoint: " + src.string());
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (bytes.size() < 4096) throw std::runtime_error("source checkpoint smaller than 4096-byte header");
  std::memset(bytes.data() + 144, 0, 4096 - 144);
  std::ofstream out(dst, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write legacy checkpoint: " + dst.string());
  out.write(bytes.data(), std::streamsize(bytes.size()));
  out.close();
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

  // Build a realistic system + tools root (~1600 tokens) and forks diverging both at exact index R (LCP == R)
  // and after user role tokens.
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
  // Fork diverging immediately at token index R itself: assistant-prefill (enable_thinking=false) begins with
  // <|message_model|> at index R, whereas rp_a begins with <|message_user|> at index R -> LCP(rp_a, rp_fork_r) == R.
  ingine::text::ChatOptions no_think_opts;
  no_think_opts.enable_thinking = false;
  const auto rp_fork_r =
      fmt.render(Json::array({{{"role", "system"}, {"content", sys_text}}}), tools, no_think_opts);

  const int R = rp_a.root_prefix_tokens;
  const std::size_t lcp_ar = common_prefix_len(rp_a.ids, rp_fork_r.ids);
  const std::size_t lcp_ab = common_prefix_len(rp_a.ids, rp_b.ids);
  std::printf("[setup] rendered root_prefix_tokens=%d | fork_a=%zu fork_b=%zu fork_c=%zu fork_r=%zu exact_root=%zu | "
              "LCP(A,fork_r)=%zu LCP(A,B)=%zu\n",
              R, rp_a.ids.size(), rp_b.ids.size(), rp_c.ids.size(), rp_fork_r.ids.size(), rp_root.ids.size(), lcp_ar,
              lcp_ab);

  bool ok = true;
  auto check = [&](bool cond, const char* label) {
    std::printf("%s %s\n", cond ? "PASS" : "FAIL", label);
    ok &= cond;
  };
  check(R >= 256 && R == rp_b.root_prefix_tokens && R == rp_c.root_prefix_tokens && R == rp_fork_r.root_prefix_tokens &&
            R == int(rp_root.ids.size()) && lcp_ar == std::size_t(R) && rp_fork_r.ids.size() > std::size_t(R) &&
            std::equal(rp_root.ids.begin(), rp_root.ids.end(), rp_a.ids.begin()) &&
            std::equal(rp_root.ids.begin(), rp_root.ids.end(), rp_b.ids.begin()),
        "rendered prompts share exact verified root boundary with LCP(A, fork_r) == R");

  RuntimeOptions base_opt;
  base_opt.model_dir = dir;
  base_opt.max_ctx = 20480;
  base_opt.prefill_chunk = 512;  // R (~1629) straddles chunk=512, exercising chunk-split root capture
  base_opt.dense_fp8 = true;
  base_opt.kv_fp8 = !use_kv_q4;
  base_opt.kv_q4 = use_kv_q4;
  base_opt.spec = false;
  base_opt.offload_vram_mb = offload_vram_mb;

  // ---- Reference outputs (strict exact-root -> Extend parity oracle, no parking) ----
  RunOut ref_fork_b, ref_fork_c, ref_fork_r, ref_exact_root;
  {
    RuntimeOptions opt = base_opt;
    opt.park_mb = 0;
    opt.park_host_mb = 0;
    auto gen = make_generator(opt);
    const std::vector<std::int32_t> noise(64, 1234);
    ref_exact_root = run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 32, greedy);
    run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 0, greedy);
    ref_fork_b = run_prompt(*gen, Prompt{rp_b.ids, {}, 0}, 32, greedy);
    run_prompt(*gen, Prompt{noise, {}, 0}, 1, greedy);
    run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 0, greedy);
    ref_fork_c = run_prompt(*gen, Prompt{rp_c.ids, {}, 0}, 32, greedy);
    run_prompt(*gen, Prompt{noise, {}, 0}, 1, greedy);
    run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 0, greedy);
    ref_fork_r = run_prompt(*gen, Prompt{rp_fork_r.ids, {}, 0}, 32, greedy);
    check(ref_fork_b.stats.reuse == Reuse::Extend && ref_fork_b.stats.reused_tokens == R &&
              ref_fork_c.stats.reuse == Reuse::Extend && ref_fork_c.stats.reused_tokens == R &&
              ref_fork_r.stats.reuse == Reuse::Extend && ref_fork_r.stats.reused_tokens == R,
          "reference oracle established strict Extend from exact root R (including LCP==R fork)");
  }

  // ---- Suite 1: B=1 Verified Root-Prefix Checkpoint (VRAM tier), forks, exact-root, child extension, namespace & --no-prefix-cache ----
  RunOut s1_ob, s1_ofr;
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

    // Immediate-divergence fork (LCP(A, fork_r) == R) -> must hit RootCheckpoint with exact greedy parity!
    const std::uint64_t d2d_before = gen->cache_stats().d2d_restore_bytes;
    s1_ofr = run_prompt(*gen, Prompt{rp_fork_r.ids, {}, R}, 32, greedy);
    const std::uint64_t d2d_after = gen->cache_stats().d2d_restore_bytes;
    check(s1_ofr.stats.reuse == Reuse::RootCheckpoint && s1_ofr.stats.reused_tokens == R &&
              s1_ofr.stats.restored_bytes.value_or(0) > 0 &&
              (d2d_after - d2d_before) == s1_ofr.stats.restored_bytes.value_or(0) &&
              s1_ofr.tokens == ref_fork_r.tokens,
          "Suite 1b-exact-LCP: fork diverging at exact index R (LCP==R) hits RootCheckpoint with exact parity & D2D byte match");

    // Sibling user fork B -> must hit RootCheckpoint with exact greedy parity!
    s1_ob = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 32, greedy);
    check(s1_ob.stats.reuse == Reuse::RootCheckpoint && s1_ob.stats.reused_tokens == R &&
              s1_ob.stats.restored_bytes.value_or(0) > 0 && s1_ob.tokens == ref_fork_b.tokens && s1_ob.saw_prefill_cb &&
              s1_ob.prefill_cb_before_tokens && s1_ob.prefill_cb_stats.reused_tokens == R &&
              s1_ob.prefill_cb_stats.reuse == Reuse::RootCheckpoint,
          "Suite 1b: fork B hits Reuse::RootCheckpoint with exact greedy parity");

    // Extend fork A past R (multi-turn child growth) -> root checkpoint MUST NOT be erased as superseded!
    std::vector<std::int32_t> child_a = rp_a.ids;
    child_a.insert(child_a.end(), oa.tokens.begin(), oa.tokens.end());
    const std::vector<std::int32_t> turn2_tail(60, 4242);
    child_a.insert(child_a.end(), turn2_tail.begin(), turn2_tail.end());
    const RunOut oca = run_prompt(*gen, Prompt{child_a, {}, R}, 8, greedy);
    const PrefixCacheStats cs_child = gen->cache_stats();
    check(oca.stats.reuse == Reuse::Parked && oca.stats.reused_tokens == int(rp_a.ids.size()) &&
              cs_child.root_entries == 1,
          "Suite 1c: child A resumes from parked A and extending past R preserves root checkpoint (root_entries=1)");

    // Fork C STILL hits the preserved root checkpoint!
    const RunOut oc = run_prompt(*gen, Prompt{rp_c.ids, {}, R}, 32, greedy);
    check(oc.stats.reuse == Reuse::RootCheckpoint && oc.stats.reused_tokens == R && oc.tokens == ref_fork_c.tokens,
          "Suite 1d: fork C hits preserved root checkpoint after sibling child growth (exact greedy parity)");

    // Exact-root request (n == R, even with hint=0 so lookup verifies exact token IDs, never trusting hint)
    const RunOut oroot = run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 32, greedy);
    check(oroot.stats.reuse == Reuse::RootCheckpoint && oroot.stats.reused_tokens == R &&
              oroot.tokens == ref_exact_root.tokens,
          "Suite 1e: exact-root request (hint=0) hits RootCheckpoint with 0 tokens prefilled & exact parity");

    // Namespace isolation + per-request bypass_prefix_cache: both MUST fail closed to Reuse::None
    const RunOut ons = run_prompt(*gen, Prompt{rp_b.ids, {}, R, "tenant-isolated", false}, 8, greedy);
    const RunOut obyp = run_prompt(*gen, Prompt{rp_b.ids, {}, R, "", true}, 8, greedy);
    check(ons.stats.reuse == Reuse::None && ons.stats.reused_tokens == 0 && obyp.stats.reuse == Reuse::None &&
              obyp.stats.reused_tokens == 0,
          "Suite 1f: different cache_namespace and Prompt::bypass_prefix_cache fail closed to Reuse::None");

    std::printf("[Suite 1 stats] %s\n", gen->offload_summary().c_str());
  }

  // ---- Suite 1g: True cache-off baseline (--no-prefix-cache) counters + continuation comparison ----
  {
    RuntimeOptions opt = base_opt;
    opt.prefix_cache = false;
    opt.park_mb = 1024;
    opt.park_host_mb = 1024;
    auto gen = make_generator(opt);
    const RunOut off_root = run_prompt(*gen, Prompt{rp_root.ids, {}, R}, 32, greedy);
    const RunOut off_b = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 32, greedy);
    const RunOut off_fr = run_prompt(*gen, Prompt{rp_fork_r.ids, {}, R}, 32, greedy);
    const PrefixCacheStats cs = gen->cache_stats();
    const std::size_t b_match = common_prefix_len(off_b.tokens, s1_ob.tokens);
    const std::size_t fr_match = common_prefix_len(off_fr.tokens, s1_ofr.tokens);
    std::printf("[Suite 1g cache-off comparison] exact_root_match=%s | fork_b_unsplit_vs_split_root=%zu/32 | "
                "fork_r_unsplit_vs_split_root=%zu/32 (matched-boundary Extend parity=32/32; unsplit cold may differ "
                "only by BF16/FP8 chunk-shape rounding at R=%d vs chunk=%d)\n",
                (off_root.tokens == ref_exact_root.tokens) ? "32/32" : "DIFF", b_match, fr_match, R,
                base_opt.prefill_chunk);
    check(off_root.stats.reuse == Reuse::None && off_b.stats.reuse == Reuse::None && off_b.stats.reused_tokens == 0 &&
              off_root.tokens == ref_exact_root.tokens && cs.vram_used_bytes == 0 && cs.host_used_bytes == 0 &&
              cs.root_entries == 0,
          "Suite 1g: prefix_cache=false disables all reuse & parking and matches exact-root cold continuation");
  }

  // ---- Suite 2: Bounded Pinned Host-RAM Parking, Simultaneous Tier Pressure, Rollback, Spec, Disk, Media ----
  {
    RuntimeOptions opt = base_opt;
    opt.park_mb = 0;
    opt.park_host_mb = 450;  // fits ~2 checkpoints (~175-190 MiB each), forces LRU eviction on 3rd
    auto gen = make_generator(opt);

    const RunOut ha = run_prompt(*gen, Prompt{rp_a.ids, {}, R}, 8, greedy);
    const std::uint64_t h2d_0 = gen->cache_stats().h2d_restore_bytes;
    const std::uint64_t d2d_0 = gen->cache_stats().d2d_restore_bytes;
    const RunOut hb = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 32, greedy);
    const PrefixCacheStats cs_h1 = gen->cache_stats();
    const std::uint64_t dma_delta = (cs_h1.h2d_restore_bytes - h2d_0) + (cs_h1.d2d_restore_bytes - d2d_0);
    check(ha.stats.reuse == Reuse::None && hb.stats.reuse == Reuse::HostParked && hb.stats.reused_tokens == R &&
              hb.stats.restored_bytes.value_or(0) > 0 && dma_delta == hb.stats.restored_bytes.value_or(0) &&
              hb.stats.restore_seconds.value_or(0.0) > 0.0 && hb.tokens == ref_fork_b.tokens &&
              cs_h1.vram_used_bytes == 0 && cs_h1.vram_peak_bytes == 0 && cs_h1.d2d_park_bytes > 0 &&
              cs_h1.d2h_park_bytes > 0 && cs_h1.host_used_bytes <= cs_h1.host_budget_bytes &&
              cs_h1.host_peak_bytes <= cs_h1.host_budget_bytes,
          "Suite 2a: direct host-only parking (park_mb=0) hits Reuse::HostParked with exact parity & copy-site DMA match");

    const RunOut hroot = run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 32, greedy);
    check(hroot.stats.reuse == Reuse::HostParked && hroot.stats.reused_tokens == R &&
              hroot.tokens == ref_exact_root.tokens,
          "Suite 2b: exact-root request hits Reuse::HostParked with exact greedy parity");

    ingine::test::Rng rng(77);
    auto make_conv = [&](int len) {
      std::vector<std::int32_t> v(static_cast<std::size_t>(len));
      for (auto& x : v) x = std::int32_t(1000 + rng.next() % 50000);
      return v;
    };
    const auto u1 = make_conv(400), u2 = make_conv(400), u3 = make_conv(400), u4 = make_conv(400);
    run_prompt(*gen, Prompt{u1, {}, 0}, 4, greedy);
    run_prompt(*gen, Prompt{u2, {}, 0}, 4, greedy);
    run_prompt(*gen, Prompt{u3, {}, 0}, 4, greedy);
    run_prompt(*gen, Prompt{u4, {}, 0}, 4, greedy);
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

    ingine::fail_next_pinned_allocations(2);
    const RunOut hfail = run_prompt(*gen, Prompt{rp_c.ids, {}, R}, 8, greedy);
    ingine::fail_next_pinned_allocations(0);
    const PrefixCacheStats cs_fail = gen->cache_stats();
    check(hfail.tokens.size() == 8 && cs_fail.host_alloc_failures >= 1,
          "Suite 2d: host pinned OOM degrades gracefully (host_alloc_failures>=1, request succeeds)");

    std::printf("[Suite 2a-2d stats] %s\n", gen->offload_summary().c_str());
  }

  // 2e + 2e-rollback + 2f: Simultaneous Tier-0 VRAM + Tier-1 Host pressure, pin-failure rollback, and VRAM OOM
  {
    RuntimeOptions opt = base_opt;
    opt.park_mb = 240;       // fits exactly 1 checkpoint in Tier-0 VRAM (~175-190 MiB)
    opt.park_host_mb = 240;  // fits exactly 1 checkpoint in Tier-1 Host -> forces simultaneous tier eviction!
    auto gen = make_generator(opt);

    // 1. Capture root in Tier-0 VRAM (slot 1/1), live = A.
    run_prompt(*gen, Prompt{rp_a.ids, {}, R}, 4, greedy);
    // 2. Fork B restores pinned root from VRAM -> live A spills directly into Tier-1 Host (slot 1/1; Tier-1 FULL!).
    run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 4, greedy);
    // 3. Unrelated prompt (keep=nullptr) parks live B into Tier-0 VRAM -> demotes root from Tier-0 to Tier-1,
    //    which MUST simultaneously evict earlier Tier-1 entry A while keeping root's iterator & data valid!
    std::vector<std::int32_t> unrelated(300, 9999);
    unrelated[0] = 8888;
    run_prompt(*gen, Prompt{unrelated, {}, 0}, 4, greedy);
    const PrefixCacheStats cs_dem = gen->cache_stats();
    // Verify the demoted root survived simultaneous Tier-1 eviction intact with exact parity!
    const RunOut dem_c = run_prompt(*gen, Prompt{rp_c.ids, {}, R}, 32, greedy);
    check(cs_dem.host_demotions >= 1 && cs_dem.host_evictions >= 1 && cs_dem.vram_entries == 1 &&
              cs_dem.host_entries == 1 && cs_dem.root_entries == 1 &&
              cs_dem.vram_peak_bytes <= cs_dem.vram_budget_bytes &&
              cs_dem.host_peak_bytes <= cs_dem.host_budget_bytes && dem_c.stats.reuse == Reuse::HostParked &&
              dem_c.stats.reused_tokens == R && dem_c.tokens == ref_fork_c.tokens,
          "Suite 2e: simultaneous Tier-0/Tier-1 pressure demotes victim cleanly while evicting earlier Tier-1 entry (exact parity)");

    // 2f: VRAM OOM releases Tier-0 VRAM while preserving Tier-1 Host root intact
    const std::uint64_t host_before = gen->cache_stats().host_entries;
    Sampling forced = greedy;
    const std::vector<std::int32_t> forced_toks = {100, 101, 102, 103};
    forced.forced = forced_toks;
    ingine::fail_next_allocations(1);
    const RunOut opress = run_prompt(*gen, Prompt{rp_c.ids, {}, 0}, 4, forced);
    const PrefixCacheStats cs_press = gen->cache_stats();
    const RunOut post_oom_b = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 32, greedy);
    check(opress.tokens == forced_toks && cs_press.vram_pressure_releases >= 1 && cs_press.vram_entries == 0 &&
              cs_press.host_entries == host_before && post_oom_b.stats.reuse == Reuse::HostParked &&
              post_oom_b.stats.reused_tokens == R && post_oom_b.tokens == ref_fork_b.tokens,
          "Suite 2f: VRAM OOM releases Tier-0 VRAM parked entries while preserving Tier-1 Host entries & root parity");

    // 2e-rollback: With BOTH Tier-0 VRAM (C) and Tier-1 Host (root) populated, inject pin failure during
    // Tier-0 -> Tier-1 demotion: alloc_host_park_slot evicts earlier Tier-1 entry on retry AND pin still fails;
    // verify the selected VRAM victim is cleanly dropped without list corruption and request completes.
    std::vector<std::int32_t> unrelated2(300, 7777);
    unrelated2[0] = 6666;
    const std::uint64_t v_ev_before = gen->cache_stats().vram_evictions;
    const std::uint64_t h_fail_before = gen->cache_stats().host_alloc_failures;
    ingine::fail_next_pinned_allocations(2);
    const RunOut oroll = run_prompt(*gen, Prompt{unrelated2, {}, 0}, 4, greedy);
    ingine::fail_next_pinned_allocations(0);
    const PrefixCacheStats cs_roll = gen->cache_stats();
    check(oroll.tokens.size() == 4 && cs_roll.host_alloc_failures > h_fail_before &&
              cs_roll.vram_evictions > v_ev_before && cs_roll.vram_used_bytes <= cs_roll.vram_budget_bytes &&
              cs_roll.host_used_bytes <= cs_roll.host_budget_bytes,
          "Suite 2e-rollback: pin failure after Tier-1 eviction rolls back demotion cleanly & drops exact VRAM victim");

    std::printf("[Suite 2e-2f stats] %s\n", gen->offload_summary().c_str());
  }

  // ---- Suite 2h: Disk checkpoint gating (verified producer, legacy handling, mixed-dtype, has_logits=false) ----
  const std::filesystem::path ckpt_with_logits =
      std::filesystem::temp_directory_path() / ("s3_ckpt_logits_" + std::string(use_kv_q4 ? "q4" : "fp8") + ".bin");
  const std::filesystem::path ckpt_no_logits =
      std::filesystem::temp_directory_path() / ("s3_ckpt_nologits_" + std::string(use_kv_q4 ? "q4" : "fp8") + ".bin");
  const std::filesystem::path ckpt_wrong_dtype =
      std::filesystem::temp_directory_path() / ("s3_ckpt_wrongdtype_" + std::string(use_kv_q4 ? "q4" : "fp8") + ".bin");
  const std::filesystem::path ckpt_legacy =
      std::filesystem::temp_directory_path() / ("s3_ckpt_legacy_" + std::string(use_kv_q4 ? "q4" : "fp8") + ".bin");
  std::filesystem::remove(ckpt_with_logits);
  std::filesystem::remove(ckpt_no_logits);
  std::filesystem::remove(ckpt_wrong_dtype);
  std::filesystem::remove(ckpt_legacy);
  {
    // Save a real disk checkpoint at exact pos = R (with logits) via periodic checkpoint on rp_b (root split hits R)
    RuntimeOptions opt = base_opt;
    opt.park_mb = 512;
    opt.park_host_mb = 0;
    opt.kv_ckpt_path = ckpt_with_logits;
    opt.kv_ckpt_every = R;
    {
      auto gen = make_generator(opt);
      run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 4, greedy);
    }
    const bool saved_ok = std::filesystem::exists(ckpt_with_logits) &&
                          inspect_kv_checkpoint(ckpt_with_logits).pos == R &&
                          inspect_kv_checkpoint(ckpt_with_logits).has_logits &&
                          inspect_kv_checkpoint(ckpt_with_logits).act_dtype ==
                              static_cast<std::uint32_t>(ingine::kActDType);
    if (saved_ok) strip_logits_from_kv_checkpoint(ckpt_with_logits, ckpt_no_logits);
    const bool stripped_ok = saved_ok && std::filesystem::exists(ckpt_no_logits) &&
                             inspect_kv_checkpoint(ckpt_no_logits).pos == R &&
                             !inspect_kv_checkpoint(ckpt_no_logits).has_logits;

    // 2h-0: Same-dtype explicit & auto resume succeed with exact parity; mixed-dtype explicit throws & auto declines
    bool same_dtype_resume_ok = false;
    bool mixed_dtype_explicit_threw = false;
    bool mixed_dtype_auto_declined = false;
    if (saved_ok) {
      {
        RuntimeOptions ropt = base_opt;
        ropt.park_mb = 0;
        ropt.park_host_mb = 0;
        ropt.kv_ckpt_resume = ckpt_with_logits;
        auto gen = make_generator(ropt);
        const RunOut er = run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 32, greedy);
        same_dtype_resume_ok =
            (er.stats.reuse == Reuse::Checkpoint && er.stats.reused_tokens == R && er.tokens == ref_exact_root.tokens);
      }
      {
        RuntimeOptions aopt = base_opt;
        aopt.park_mb = 0;
        aopt.park_host_mb = 0;
        aopt.kv_ckpt_path = ckpt_with_logits;
        aopt.kv_ckpt_every = 0;
        auto gen = make_generator(aopt);
        const RunOut er = run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 32, greedy);
        same_dtype_resume_ok &=
            (er.stats.reuse == Reuse::Checkpoint && er.stats.reused_tokens == R && er.tokens == ref_exact_root.tokens);
      }
      const std::uint32_t other_dtype = (ingine::kActDType == ingine::DType::FP16)
                                            ? static_cast<std::uint32_t>(ingine::DType::BF16)
                                            : static_cast<std::uint32_t>(ingine::DType::FP16);
      patch_act_dtype_in_kv_checkpoint(ckpt_with_logits, ckpt_wrong_dtype, other_dtype);
      {
        RuntimeOptions ropt = base_opt;
        ropt.park_mb = 0;
        ropt.park_host_mb = 0;
        ropt.kv_ckpt_resume = ckpt_wrong_dtype;
        auto gen = make_generator(ropt);
        try {
          run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 8, greedy);
        } catch (const std::exception&) {
          mixed_dtype_explicit_threw = true;
        }
      }
      {
        RuntimeOptions aopt = base_opt;
        aopt.park_mb = 0;
        aopt.park_host_mb = 0;
        aopt.kv_ckpt_path = ckpt_wrong_dtype;
        aopt.kv_ckpt_every = 0;
        auto gen = make_generator(aopt);
        const RunOut er = run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 32, greedy);
        mixed_dtype_auto_declined =
            (er.stats.reuse == Reuse::None && er.stats.reused_tokens == 0 && er.tokens == ref_exact_root.tokens);
      }
    }
    check(saved_ok && same_dtype_resume_ok && mixed_dtype_explicit_threw && mixed_dtype_auto_declined,
          "Suite 2h-dtype: same-dtype disk resume succeeds & mixed-dtype throws on explicit / declines on auto");

    // 2h-legacy: Legacy checkpoint (no producer identity trailer): explicit preserved without root promotion; auto declined
    bool legacy_explicit_no_promote = false;
    bool legacy_auto_declined = false;
    if (saved_ok) {
      strip_producer_from_kv_checkpoint(ckpt_with_logits, ckpt_legacy);
      {
        RuntimeOptions ropt = base_opt;
        ropt.park_mb = 512;
        ropt.park_host_mb = 0;
        ropt.kv_ckpt_resume = ckpt_legacy;
        auto gen = make_generator(ropt);
        const RunOut fb = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 32, greedy);
        legacy_explicit_no_promote = (fb.stats.reuse == Reuse::Checkpoint && fb.stats.reused_tokens == R &&
                                      fb.tokens == ref_fork_b.tokens && gen->cache_stats().root_entries == 0);
      }
      {
        RuntimeOptions aopt = base_opt;
        aopt.park_mb = 512;
        aopt.park_host_mb = 0;
        aopt.kv_ckpt_path = ckpt_legacy;
        aopt.kv_ckpt_every = 0;
        auto gen = make_generator(aopt);
        const RunOut fb = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 32, greedy);
        legacy_auto_declined =
            (fb.stats.reuse == Reuse::None && fb.stats.reused_tokens == 0 && fb.tokens == ref_fork_b.tokens);
      }
    }
    check(saved_ok && legacy_explicit_no_promote && legacy_auto_declined,
          "Suite 2h-legacy: legacy disk ckpt preserved on explicit (no root promotion) & declined on auto");

    // 2h-1: Explicit --kv-ckpt-resume on exact-root with has_logits=false MUST throw
    bool explicit_nologits_threw = false;
    if (stripped_ok) {
      RuntimeOptions ropt = base_opt;
      ropt.park_mb = 512;
      ropt.park_host_mb = 0;
      ropt.kv_ckpt_resume = ckpt_no_logits;
      auto gen = make_generator(ropt);
      try {
        run_prompt(*gen, Prompt{rp_root.ids, {}, R}, 8, greedy);
      } catch (const std::exception&) {
        explicit_nologits_threw = true;
      }
    }

    // 2h-2 & 2h-3: Auto --kv-ckpt-path with has_logits=false:
    //   - On exact-root (n == R): declines disk resume (Reuse::None) and recomputes exact ref_exact_root
    //   - On fork B (n > R, root_prefix_tokens = R): resumes R tokens (Reuse::Checkpoint), computes suffix logits
    //     matching ref_fork_b, and MUST NOT promote the logit-less disk resume into a reusable root (root_entries==0)!
    bool auto_exact_declined = false;
    bool auto_fork_no_promote = false;
    if (stripped_ok) {
      RuntimeOptions aopt = base_opt;
      aopt.park_mb = 512;
      aopt.park_host_mb = 0;
      aopt.kv_ckpt_path = ckpt_no_logits;
      aopt.kv_ckpt_every = 0;
      {
        auto gen = make_generator(aopt);
        const RunOut er = run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 32, greedy);
        auto_exact_declined = (er.stats.reuse == Reuse::None && er.stats.reused_tokens == 0 &&
                               er.tokens == ref_exact_root.tokens);
      }
      {
        auto gen = make_generator(aopt);
        const RunOut fb = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 32, greedy);
        const PrefixCacheStats cs = gen->cache_stats();
        auto_fork_no_promote = (fb.stats.reuse == Reuse::Checkpoint && fb.stats.reused_tokens == R &&
                                fb.tokens == ref_fork_b.tokens && cs.root_entries == 0);
      }
    }
    check(saved_ok && stripped_ok && explicit_nologits_threw && auto_exact_declined && auto_fork_no_promote,
          "Suite 2h: has_logits=false disk checkpoint rejected on exact-root & never promoted to reusable root");

    // 2h-supplied-ablation: given-weights overload with real load_weights + supplied active Ablation (options.ablation
    // empty) MUST NOT write a checkpoint, MUST throw on explicit resume, and MUST decline auto-resume!
    const std::filesystem::path abl_path = dir / "inkling-2-mini-abliterated.safetensors";
    if (saved_ok && std::filesystem::exists(abl_path)) {
      const std::filesystem::path ckpt_abl_out =
          std::filesystem::temp_directory_path() / ("s3_ckpt_abl_out_" + std::string(use_kv_q4 ? "q4" : "fp8") + ".bin");
      std::filesystem::remove(ckpt_abl_out);
      auto make_supplied_abl_gen = [&](const RuntimeOptions& ropt) {
        cudaStream_t s;
        INGINE_TEST_CUDA(cudaStreamCreate(&s));
        Weights w = load_weights(dir, ropt.dense_fp8, s);
        Ablation abl = load_ablation(abl_path);
        INGINE_TEST_CUDA(cudaStreamDestroy(s));
        return make_generator(ropt, std::move(w), std::move(abl));
      };
      bool abl_save_blocked = false;
      bool abl_explicit_threw = false;
      bool abl_auto_declined = false;
      {
        RuntimeOptions aopt = base_opt;
        aopt.ablation.clear();  // empty options.ablation, active supplied Ablation!
        aopt.park_mb = 0;
        aopt.park_host_mb = 0;
        aopt.kv_ckpt_path = ckpt_abl_out;
        aopt.kv_ckpt_every = R;
        auto gen = make_supplied_abl_gen(aopt);
        run_prompt(*gen, Prompt{rp_b.ids, {}, 0}, 4, greedy);
        abl_save_blocked = !std::filesystem::exists(ckpt_abl_out);
      }
      {
        RuntimeOptions ropt = base_opt;
        ropt.ablation.clear();
        ropt.park_mb = 0;
        ropt.park_host_mb = 0;
        ropt.kv_ckpt_resume = ckpt_with_logits;
        auto gen = make_supplied_abl_gen(ropt);
        try {
          run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 4, greedy);
        } catch (const std::invalid_argument&) {
          abl_explicit_threw = true;
        }
      }
      {
        RuntimeOptions aopt = base_opt;
        aopt.ablation.clear();
        aopt.park_mb = 0;
        aopt.park_host_mb = 0;
        aopt.kv_ckpt_path = ckpt_with_logits;
        aopt.kv_ckpt_every = 0;
        auto gen = make_supplied_abl_gen(aopt);
        const RunOut er = run_prompt(*gen, Prompt{rp_root.ids, {}, 0}, 16, greedy);
        abl_auto_declined = (er.stats.reuse == Reuse::None && er.stats.reused_tokens == 0 && er.tokens.size() == 16);
      }
      std::filesystem::remove(ckpt_abl_out);
      check(abl_save_blocked && abl_explicit_threw && abl_auto_declined,
            "Suite 2h-supplied-ablation: active supplied Ablation (empty options.ablation) blocks save, throws on explicit, declines auto");
    }

    // 2h-consumed-weight-toctou + unknown-reader explicit-resume gating:
    //   1. Deterministically flip the private 4608-byte embed_norm shard on disk DURING Load 1 right after
    //      SafetensorsModel::read("model.llm.embed_norm.weight") returns its +norm payload (before flush/digest).
    //      Assert actual GPU device norm[0] differs between Load 1 (+norm) and Load 2 (-norm), their verified
    //      consumed-weight digests differ, Load 2 declines auto-resume, and Load 3 throws on explicit resume.
    //   2. Unknown-reader explicit resume (tiered expert offload AND caller-supplied given weights) MUST reject
    //      Load 1's stamped checkpoint BEFORE import (even on mismatched prompts), while preserving explicit resume
    //      for unstamped legacy checkpoints.
#if defined(__linux__)
    if (std::filesystem::exists(dir / "model.safetensors.index.json")) {
      const std::string pid_tag = std::to_string(int(::getpid()));
      const std::filesystem::path priv_dir =
          std::filesystem::temp_directory_path() /
          ("s3_priv_toctou_" + std::string(use_kv_q4 ? "q4" : "fp8") + "_" + pid_tag);
      const std::filesystem::path ckpt_toctou = priv_dir / "toctou_pos64.bin";
      const std::filesystem::path ckpt_toctou_legacy = priv_dir / "toctou_pos64_legacy.bin";
      std::error_code ec;
      std::filesystem::remove_all(priv_dir, ec);
      std::filesystem::create_directories(priv_dir);
      // Symlink large weight shard(s) and config/quant files; override embed_norm into a tiny private shard.
      for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const std::string fname = entry.path().filename().string();
        if (fname == "model.safetensors.index.json") continue;
        std::filesystem::create_symlink(entry.path(), priv_dir / fname, ec);
      }
      const std::string en_name(kToctouNormName);
      const std::filesystem::path priv_shard = priv_dir / "embed_norm_priv.safetensors";
      std::vector<std::byte> en_raw;
      std::string en_dtype;
      std::vector<std::int64_t> en_shape;
      {
        const ingine::SafetensorsModel orig_m(dir);
        const ingine::SafetensorsRef eref = orig_m.at(en_name);
        en_dtype = eref->dtype;
        en_shape = eref->shape;
        en_raw = orig_m.read(en_name);
      }
      {
        Json sh = Json::array();
        for (auto d : en_shape) sh.push_back(d);
        Json hdr = {{en_name, {{"dtype", en_dtype}, {"shape", sh}, {"data_offsets", Json::array({0, en_raw.size()})}}}};
        const std::string hdr_s = hdr.dump();
        const std::uint64_t hlen = std::uint64_t(hdr_s.size());
        std::ofstream out(priv_shard, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
        out.write(hdr_s.data(), std::streamsize(hdr_s.size()));
        out.write(reinterpret_cast<const char*>(en_raw.data()), std::streamsize(en_raw.size()));
      }
      {
        std::ifstream in(dir / "model.safetensors.index.json");
        Json idx = Json::parse(in);
        idx["weight_map"][en_name] = "embed_norm_priv.safetensors";
        std::ofstream out(priv_dir / "model.safetensors.index.json", std::ios::trunc);
        out << idx.dump();
      }

      const std::vector<std::int32_t> short_p(rp_root.ids.begin(), rp_root.ids.begin() + 100);
      const std::vector<std::int32_t> exact_p(rp_root.ids.begin(), rp_root.ids.begin() + 64);
      std::vector<std::int32_t> other_exact_p = exact_p;
      other_exact_p[0] = (other_exact_p[0] == 1000) ? 1001 : 1000;

      RuntimeOptions toctou_base = base_opt;
      toctou_base.model_dir = priv_dir;
      toctou_base.max_ctx = 512;
      toctou_base.prefill_chunk = 64;
      toctou_base.park_mb = 0;
      toctou_base.park_host_mb = 0;

      g_toctou_priv_shard = priv_shard;
      g_toctou_armed = true;
      g_toctou_flipped = false;
      g_observe_loads = true;
      g_observed_norm0.clear();
      g_observed_shas.clear();

      RunOut out1_exact, out2_auto;
      bool toctou_saved = false;
      bool toctou_explicit_threw = false;
      {
        RuntimeOptions topt = toctou_base;
        topt.kv_ckpt_path = ckpt_toctou;
        topt.kv_ckpt_every = 64;
        auto gen1 = make_generator(topt);  // consumes +embed_norm; wrapper flips shard to -embed_norm mid-load
        (void)run_prompt(*gen1, Prompt{short_p, {}, 0}, 8, greedy);
        out1_exact = run_prompt(*gen1, Prompt{exact_p, {}, 0, "", true}, 16, greedy);
        toctou_saved = std::filesystem::exists(ckpt_toctou) && inspect_kv_checkpoint(ckpt_toctou).pos == 64 &&
                       inspect_kv_checkpoint(ckpt_toctou).has_producer_identity;
      }
      {
        RuntimeOptions topt = toctou_base;
        topt.kv_ckpt_path = ckpt_toctou;
        topt.kv_ckpt_every = 0;
        auto gen2 = make_generator(topt);  // consumes -embed_norm -> different consumed producer digest
        out2_auto = run_prompt(*gen2, Prompt{exact_p, {}, 0}, 16, greedy);
      }
      {
        RuntimeOptions topt = toctou_base;
        topt.kv_ckpt_resume = ckpt_toctou;
        auto gen3 = make_generator(topt);
        try {
          run_prompt(*gen3, Prompt{exact_p, {}, 0}, 8, greedy);
        } catch (const std::runtime_error& e) {
          toctou_explicit_threw =
              std::string(e.what()).find("producer identity mismatch") != std::string::npos;
        }
      }
      const bool resident_norms_and_shas_distinct =
          g_toctou_flipped && !g_toctou_armed && g_observed_norm0.size() == 3 && g_observed_shas.size() == 3 &&
          g_observed_norm0[0] != g_observed_norm0[1] && g_observed_norm0[1] == g_observed_norm0[2] &&
          g_observed_shas[0].has_value() && g_observed_shas[1].has_value() && g_observed_shas[2].has_value() &&
          *g_observed_shas[0] != *g_observed_shas[1] && *g_observed_shas[1] == *g_observed_shas[2];
      std::printf("[Suite 2h-toctou] flipped_during_load=%d norm0=(%.6g, %.6g, %.6g) distinct_sha=%d auto_reuse=%d\n",
                  int(g_toctou_flipped), g_observed_norm0.size() > 0 ? g_observed_norm0[0] : 0.f,
                  g_observed_norm0.size() > 1 ? g_observed_norm0[1] : 0.f,
                  g_observed_norm0.size() > 2 ? g_observed_norm0[2] : 0.f,
                  int(g_observed_shas.size() > 1 && g_observed_shas[0] && g_observed_shas[1] &&
                      *g_observed_shas[0] != *g_observed_shas[1]),
                  int(out2_auto.stats.reuse));
      check(resident_norms_and_shas_distinct && toctou_saved && out2_auto.stats.reuse == Reuse::None &&
                out2_auto.stats.reused_tokens == 0 && out2_auto.tokens != out1_exact.tokens && toctou_explicit_threw,
            "Suite 2h-consumed-weight-toctou: during-load embed_norm mutation yields distinct resident norm & consumed producer SHA (auto declines & explicit throws)");

      // Unknown-reader explicit-resume gating on revision B (-embed_norm) against revision A's stamped checkpoint:
      //   * Branch 1: tiered expert offload (offload_vram_mb=12288, offload_ram_mb=512) -> content_sha256 unknown
      //   * Branch 2: caller-supplied given-weights overload (make_generator(opt, load_weights(...), load_ablation({})))
      //   * Unstamped legacy checkpoint remains explicitly resumable (isolated, root_entries==0).
      bool tiered_stamped_rejected = false;
      bool given_stamped_rejected = false;
      bool unknown_legacy_preserved = false;
      if (toctou_saved && resident_norms_and_shas_distinct) {
        {
          RuntimeOptions topt = toctou_base;
          topt.offload_vram_mb = 12288;
          topt.offload_ram_mb = 512;
          topt.kv_ckpt_resume = ckpt_toctou;
          auto gen_t = make_generator(topt);
          const bool tiered_obs_ok = !g_observed_norm0.empty() && g_observed_norm0.back() == g_observed_norm0[1] &&
                                     !g_observed_shas.empty() && !g_observed_shas.back().has_value();
          bool threw_match = false, threw_other = false;
          try {
            run_prompt(*gen_t, Prompt{exact_p, {}, 0}, 16, greedy);
          } catch (const std::runtime_error& e) {
            threw_match = std::string(e.what()).find("cannot verify stamped checkpoint producer") != std::string::npos;
          }
          try {
            run_prompt(*gen_t, Prompt{other_exact_p, {}, 0}, 16, greedy);
          } catch (const std::runtime_error& e) {
            threw_other = std::string(e.what()).find("cannot verify stamped checkpoint producer") != std::string::npos;
          }
          const PrefixCacheStats cs = gen_t->cache_stats();
          tiered_stamped_rejected = tiered_obs_ok && threw_match && threw_other && cs.root_entries == 0 &&
                                    cs.vram_entries == 0 && cs.host_entries == 0;
        }
        {
          RuntimeOptions gopt = toctou_base;
          gopt.kv_ckpt_resume = ckpt_toctou;
          cudaStream_t s;
          INGINE_TEST_CUDA(cudaStreamCreate(&s));
          Weights w = load_weights(priv_dir, gopt.dense_fp8, s);
          INGINE_TEST_CUDA(cudaStreamDestroy(s));
          const bool w_obs_ok = w.content_sha256.has_value() && *w.content_sha256 == *g_observed_shas[1] &&
                                !g_observed_norm0.empty() && g_observed_norm0.back() == g_observed_norm0[1];
          auto gen_g = make_generator(gopt, std::move(w), load_ablation({}));
          bool threw_match = false, threw_other = false;
          try {
            run_prompt(*gen_g, Prompt{exact_p, {}, 0}, 16, greedy);
          } catch (const std::runtime_error& e) {
            threw_match = std::string(e.what()).find("cannot verify stamped checkpoint producer") != std::string::npos;
          }
          try {
            run_prompt(*gen_g, Prompt{other_exact_p, {}, 0}, 16, greedy);
          } catch (const std::runtime_error& e) {
            threw_other = std::string(e.what()).find("cannot verify stamped checkpoint producer") != std::string::npos;
          }
          const PrefixCacheStats cs = gen_g->cache_stats();
          given_stamped_rejected = w_obs_ok && threw_match && threw_other && cs.root_entries == 0 &&
                                   cs.vram_entries == 0 && cs.host_entries == 0;
        }
        strip_producer_from_kv_checkpoint(ckpt_toctou, ckpt_toctou_legacy);
        {
          RuntimeOptions lopt = toctou_base;
          lopt.park_mb = 512;
          lopt.kv_ckpt_resume = ckpt_toctou_legacy;
          cudaStream_t s;
          INGINE_TEST_CUDA(cudaStreamCreate(&s));
          Weights w = load_weights(priv_dir, lopt.dense_fp8, s);
          INGINE_TEST_CUDA(cudaStreamDestroy(s));
          auto gen_l = make_generator(lopt, std::move(w), load_ablation({}));
          const RunOut leg = run_prompt(*gen_l, Prompt{short_p, {}, 64}, 8, greedy);
          unknown_legacy_preserved = (leg.stats.reuse == Reuse::Checkpoint && leg.stats.reused_tokens == 64 &&
                                      gen_l->cache_stats().root_entries == 0);
        }
      }
      g_observe_loads = false;
      g_toctou_armed = false;
      std::filesystem::remove_all(priv_dir, ec);
      check(tiered_stamped_rejected && given_stamped_rejected && unknown_legacy_preserved,
            "Suite 2h-unknown-reader-explicit: tiered offload & given-weights reject stamped ckpt pre-import while preserving unstamped legacy");
    }
#endif
  }

  // ---- Suite 2g: DSpark --spec host-parked root + fork parity & speculative disk-resume rejection/decline ----
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
    // Speculative explicit --kv-ckpt-resume MUST throw; auto --kv-ckpt-path MUST decline & match spec_ref_b
    bool spec_explicit_threw = false;
    bool spec_auto_declined = false;
    if (std::filesystem::exists(ckpt_with_logits)) {
      {
        RuntimeOptions sopt = opt;
        sopt.kv_ckpt_resume = ckpt_with_logits;
        auto gen = make_generator(sopt);
        try {
          run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 8, greedy);
        } catch (const std::exception&) {
          spec_explicit_threw = true;
        }
      }
      {
        RuntimeOptions sopt = opt;
        sopt.park_host_mb = 1024;
        sopt.kv_ckpt_path = ckpt_with_logits;
        sopt.kv_ckpt_every = 0;
        auto gen = make_generator(sopt);
        const RunOut sa = run_prompt(*gen, Prompt{rp_a.ids, {}, R}, 8, greedy);
        const RunOut sb = run_prompt(*gen, Prompt{rp_b.ids, {}, R}, 32, greedy);
        spec_auto_declined = (sa.stats.reuse == Reuse::None && sa.stats.reused_tokens == 0 &&
                              sb.stats.reuse == Reuse::HostParked && sb.stats.reused_tokens == R &&
                              sb.tokens == spec_ref_b.tokens);
        std::printf("[Suite 2g spec stats] %s\n", gen->offload_summary().c_str());
      }
    }
    check(spec_explicit_threw && spec_auto_declined,
          "Suite 2g: DSpark --spec rejects/declines incompatible disk ckpt & passes HostParked fork exact parity");
  }
  std::filesystem::remove(ckpt_with_logits);
  std::filesystem::remove(ckpt_no_logits);
  std::filesystem::remove(ckpt_wrong_dtype);
  std::filesystem::remove(ckpt_legacy);

  // ---- Suite 2i: Verified media SHA-256 identity binding (same negative token ID, different RGB fails closed) ----
  {
    RuntimeOptions vopt = base_opt;
    vopt.vision = true;
    vopt.park_mb = 512;
    vopt.park_host_mb = 0;
    try {
      auto gen = make_generator(vopt);
      if (gen->supports_images()) {
        std::vector<std::int32_t> img_ids(300, 1234);
        img_ids[40] = -42;  // identical negative image token ID in both prompts!
        auto red_rgb = std::make_shared<std::vector<std::uint8_t>>(40 * 40 * 3, 220);
        auto blue_rgb = std::make_shared<std::vector<std::uint8_t>>(40 * 40 * 3, 25);
        std::array<PromptImage, 1> img_red = {PromptImage{40, 40, 40, red_rgb}};
        std::array<PromptImage, 1> img_blue = {PromptImage{40, 40, 40, blue_rgb}};
        const RunOut m_red1 = run_prompt(*gen, Prompt{img_ids, img_red, 0}, 8, greedy);
        const RunOut m_blue = run_prompt(*gen, Prompt{img_ids, img_blue, 0}, 8, greedy);
        const RunOut m_red2 = run_prompt(*gen, Prompt{img_ids, img_red, 0}, 8, greedy);
        check(m_red1.stats.reuse == Reuse::None && m_blue.stats.reuse == Reuse::None &&
                  m_blue.stats.reused_tokens == 0 && m_red2.stats.reuse == Reuse::Parked &&
                  m_red2.stats.reused_tokens == 300 && m_red2.tokens == m_red1.tokens,
              "Suite 2i: verified media SHA-256 fails closed on different RGB with identical negative token IDs");
      }
    } catch (const std::exception& e) {
      std::printf("skip Suite 2i (vision tower unavailable: %s)\n", e.what());
    }
  }

  // ---- Suite 3: Apples-to-Apples End-to-End Wall TTFT Crossover (kReps=3 medians) ----
  // Brackets below kMinParked=256 using eligible verified root checkpoints (kMinRootTokens=64: L=64, 128).
  // Compares the EXACT SAME wall TTFT boundary (stats.prefill_seconds: start of generate() -> first sampled token)
  // across Cold, Tier-0 VRAM, and Tier-1 Host, while reporting CUDA transfer-event timing separately.
  if (run_crossover) {
    RuntimeOptions opt = base_opt;
    opt.max_ctx = 20480;
    opt.prefill_chunk = 3072;
    opt.park_mb = 0;
    opt.park_host_mb = 2048;

    RuntimeOptions opt_vram = opt;
    opt_vram.park_mb = 2048;
    opt_vram.park_host_mb = 0;

    constexpr int kReps = 3;
    std::printf("\n=== MEASURED WALL-TTFT RESTORE-VS-RECOMPUTE CROSSOVER (kv=%s, chunk=%d, reps=%d median) ===\n",
                use_kv_q4 ? "q4_0" : "fp8", opt.prefill_chunk, kReps);
    std::printf("%-6s | %-4s | %-8s | %-11s | %-11s | %-11s | %-10s | %-10s | %-11s | %-11s | %-10s | %-6s\n",
                "Len", "Path", "CkptMiB", "ColdTTFTms", "VramTTFTms", "HostTTFTms", "TTFT_SpdV", "TTFT_SpdH",
                "VramXferms", "HostXferms", "HostGB/s", "Parity");

    ingine::test::Rng rng(2026);
    std::vector<std::int32_t> master(16500);
    for (auto& x : master) x = std::int32_t(1000 + rng.next() % 50000);
    const std::vector<std::int32_t> flush_ids(128, 7777);
    const std::vector<int> lengths = {64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    struct Row {
      int L = 0;
      bool is_root_bracket = false;
      std::uint64_t h2d_bytes = 0;
      double cold_ttft_ms = 0;
      double host_ttft_ms = 0;
      double vram_ttft_ms = 0;
      double host_xfer_ms = 0;
      double vram_xfer_ms = 0;
      std::array<std::vector<std::int32_t>, kReps> cold_toks;
      bool host_ok = true;
      bool vram_ok = true;
    };
    std::vector<Row> rows(lengths.size());
    PrefixCacheStats ch, cv;
    std::size_t dev_vram_host_snap = 0, dev_vram_vram_snap = 0;

    // Pass 1: Pure cold recompute (bypass_prefix_cache=true) + Tier-1 Pinned Host-RAM restore on IDENTICAL in-vocab IDs
    {
      auto gen_host = make_generator(opt);
      dev_vram_host_snap = device_wide_vram_used_bytes();
      run_prompt(*gen_host, Prompt{std::span(master).subspan(0, 256), {}, 0}, 1, greedy);  // warm up
      for (std::size_t i = 0; i < lengths.size(); ++i) {
        const int L = lengths[i];
        const bool use_root = (L < 256);  // bracket below kMinParked=256 where root parking (kMinRootTokens=64) applies
        const int root_hint = use_root ? L : 0;
        std::vector<std::int32_t> p(master.begin(), master.begin() + L);
        rows[i].L = L;
        rows[i].is_root_bracket = use_root;

        std::vector<double> cold_samples, host_ttft_samples, host_xfer_samples;
        for (int rep = 0; rep < kReps; ++rep) {
          // Strictly in-vocabulary IDs (< UNPADDED_VOCAB) identical across pure cold, host restore, and VRAM restore
          std::vector<std::int32_t> pc = p;
          pc[0] = 1000 + int(i) * 10 + rep;

          // 1) Pure cold recompute measurement on identical IDs (bypass_prefix_cache=true -> zero pin/root overhead)
          const RunOut pure_cold = run_prompt(*gen_host, Prompt{pc, {}, 0, "", /*bypass_prefix_cache=*/true}, 1, greedy);
          cold_samples.push_back(pure_cold.stats.prefill_seconds * 1e3);
          rows[i].cold_toks[static_cast<std::size_t>(rep)] = pure_cold.tokens;

          // 2) Populate parked entry for the exact same `pc`, flush live, then time HostParked restore
          run_prompt(*gen_host, Prompt{flush_ids, {}, 0}, 1, greedy);
          const RunOut pop = run_prompt(*gen_host, Prompt{pc, {}, root_hint}, 1, greedy);

          run_prompt(*gen_host, Prompt{flush_ids, {}, 0}, 1, greedy);  // parks pc (non-root) or flushes live (root already parked)
          const std::uint64_t h2d_before = gen_host->cache_stats().h2d_restore_bytes;
          const RunOut hres = run_prompt(*gen_host, Prompt{pc, {}, 0}, 1, greedy);
          const std::uint64_t h2d_after = gen_host->cache_stats().h2d_restore_bytes;
          if (rep == 0) rows[i].h2d_bytes = h2d_after - h2d_before;
          host_ttft_samples.push_back(hres.stats.prefill_seconds * 1e3);
          host_xfer_samples.push_back(hres.stats.restore_seconds.value_or(0.0) * 1e3);
          rows[i].host_ok &= (pure_cold.stats.reuse == Reuse::None && pure_cold.stats.reused_tokens == 0 &&
                              hres.stats.reuse == Reuse::HostParked && hres.stats.reused_tokens == L &&
                              pop.tokens == pure_cold.tokens && hres.tokens == pure_cold.tokens);
        }
        rows[i].cold_ttft_ms = median_of(cold_samples);
        rows[i].host_ttft_ms = median_of(host_ttft_samples);
        rows[i].host_xfer_ms = median_of(host_xfer_samples);
      }
      ch = gen_host->cache_stats();
    }

    // Pass 2: Tier-0 VRAM parked restore on the EXACT SAME in-vocab IDs, directly compared to Pass 1 pure_cold.tokens
    {
      auto gen_vram = make_generator(opt_vram);
      dev_vram_vram_snap = device_wide_vram_used_bytes();
      run_prompt(*gen_vram, Prompt{std::span(master).subspan(0, 256), {}, 0}, 1, greedy);  // warm up
      for (std::size_t i = 0; i < lengths.size(); ++i) {
        const int L = lengths[i];
        const bool use_root = (L < 256);
        const int root_hint = use_root ? L : 0;
        std::vector<std::int32_t> p(master.begin(), master.begin() + L);
        std::vector<double> vram_ttft_samples, vram_xfer_samples;
        for (int rep = 0; rep < kReps; ++rep) {
          std::vector<std::int32_t> pc = p;
          pc[0] = 1000 + int(i) * 10 + rep;
          run_prompt(*gen_vram, Prompt{flush_ids, {}, 0}, 1, greedy);
          const RunOut pop = run_prompt(*gen_vram, Prompt{pc, {}, root_hint}, 1, greedy);
          run_prompt(*gen_vram, Prompt{flush_ids, {}, 0}, 1, greedy);
          const RunOut vres = run_prompt(*gen_vram, Prompt{pc, {}, 0}, 1, greedy);
          vram_ttft_samples.push_back(vres.stats.prefill_seconds * 1e3);
          vram_xfer_samples.push_back(vres.stats.restore_seconds.value_or(0.0) * 1e3);
          const Reuse expected_vram = use_root ? Reuse::RootCheckpoint : Reuse::Parked;
          const auto& cold_ref = rows[i].cold_toks[static_cast<std::size_t>(rep)];
          rows[i].vram_ok &= (vres.stats.reuse == expected_vram && vres.stats.reused_tokens == L &&
                              pop.tokens == cold_ref && vres.tokens == cold_ref);
        }
        rows[i].vram_ttft_ms = median_of(vram_ttft_samples);
        rows[i].vram_xfer_ms = median_of(vram_xfer_samples);
      }
      cv = gen_vram->cache_stats();
    }

    for (const Row& r : rows) {
      const double ckpt_mib = double(r.h2d_bytes) / (1024.0 * 1024.0);
      const double spd_v = r.vram_ttft_ms > 0.0 ? (r.cold_ttft_ms / r.vram_ttft_ms) : 0.0;
      const double spd_h = r.host_ttft_ms > 0.0 ? (r.cold_ttft_ms / r.host_ttft_ms) : 0.0;
      const double host_gbs = r.host_xfer_ms > 0.0 ? (double(r.h2d_bytes) / (r.host_xfer_ms * 1e6)) : 0.0;
      const bool par = r.host_ok && r.vram_ok;
      std::printf("%-6d | %-4s | %8.1f | %11.2f | %11.3f | %11.3f | %9.2fx | %9.2fx | %11.3f | %11.3f | %10.2f | %s\n",
                  r.L, r.is_root_bracket ? "root" : "park", ckpt_mib, r.cold_ttft_ms, r.vram_ttft_ms, r.host_ttft_ms,
                  spd_v, spd_h, r.vram_xfer_ms, r.host_xfer_ms, host_gbs, par ? "PASS" : "FAIL");
      ok &= par;
    }
    std::printf("[crossover prefix-cache host-tier] staging=%.1f MiB, vram_park_peak=%.1f MiB, host_park_peak=%.1f MiB, "
                "dma(h2d_r=%.2f MB, d2h_p=%.2f MB, d2d_r=%.2f MB, d2d_p=%.2f MB), host_evict=%llu, demote=%llu\n",
                double(ch.device_staging_bytes) / (1024.0 * 1024.0), double(ch.vram_peak_bytes) / (1024.0 * 1024.0),
                double(ch.host_peak_bytes) / (1024.0 * 1024.0), double(ch.h2d_restore_bytes) / 1e6,
                double(ch.d2h_park_bytes) / 1e6, double(ch.d2d_restore_bytes) / 1e6, double(ch.d2d_park_bytes) / 1e6,
                static_cast<unsigned long long>(ch.host_evictions), static_cast<unsigned long long>(ch.host_demotions));
    std::printf("[crossover prefix-cache vram-tier] staging=%.1f MiB, vram_park_peak=%.1f MiB, host_park_peak=%.1f MiB, "
                "dma(h2d_r=%.2f MB, d2h_p=%.2f MB, d2d_r=%.2f MB, d2d_p=%.2f MB), vram_evict=%llu\n",
                double(cv.device_staging_bytes) / (1024.0 * 1024.0), double(cv.vram_peak_bytes) / (1024.0 * 1024.0),
                double(cv.host_peak_bytes) / (1024.0 * 1024.0), double(cv.h2d_restore_bytes) / 1e6,
                double(cv.d2h_park_bytes) / 1e6, double(cv.d2d_restore_bytes) / 1e6, double(cv.d2d_park_bytes) / 1e6,
                static_cast<unsigned long long>(cv.vram_evictions));
    std::printf("[device-wide cudaMemGetInfo snapshot (non-exclusive GPU, not per-process peak)] "
                "host_pass=%.2f GiB, vram_pass=%.2f GiB\n",
                double(dev_vram_host_snap) / (1024.0 * 1024.0 * 1024.0),
                double(dev_vram_vram_snap) / (1024.0 * 1024.0 * 1024.0));
  }

  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}

// Unit tests for multi-GPU TP startup byte-exact P2P probe, fail-closed IOMMU guard, and option validation.
// Uses explicit runtime checks (never assert()) so all assertions stay active under Release (-DNDEBUG) builds.
#include "model/tp_engine.h"

#include "core/gpu_compat.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace ingine::model;

namespace {

int g_failures = 0;

void expect(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  }
}

void expect_rejects(const RuntimeOptions& opts, int tp_size, const char* what) {
  bool threw = false;
  try {
    validate_tp_runtime_options(opts, tp_size);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  expect(threw, what);
}

void expect_accepts(const RuntimeOptions& opts, int tp_size, const char* what) {
  bool ok = true;
  try {
    validate_tp_runtime_options(opts, tp_size);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s (unexpected exception: %s)\n", what, e.what());
    ok = false;
  }
  if (!ok) ++g_failures;
}

}  // namespace

int main() {
  // 1. Deterministic non-zero pattern sanity (CPU-only, always runs).
  const std::uint32_t w00 = tp_p2p_probe_word(0, 0);
  const std::uint32_t w10 = tp_p2p_probe_word(1, 0);
  const std::uint32_t w01 = tp_p2p_probe_word(0, 1);
  expect(w00 != 0 && w00 != 0xDEADBEEFu, "probe word is non-zero and distinct from poison");
  expect(w00 != w10, "probe word varies across source ranks");
  expect(w00 != w01, "probe word varies across word indices");

  // 2. tp_size <= 1 is a clean no-op.
  {
    const TpP2pProbeResult r1 = probe_and_enable_tp_p2p(1, false);
    expect(r1.capable_pairs == 0 && r1.verified_pairs == 0 && !r1.p2p_enabled && !r1.corrupted,
           "tp_size=1 probe is a no-op");
  }

  // 3. TP RuntimeOptions validation rejects unsupported cache/feature flags before GPU allocation.
  {
    RuntimeOptions base;
    base.max_ctx = 65536;
    base.max_batch = 1;
    base.spec = true;
    expect_accepts(base, 2, "TP2 accepts default BF16/FP16 spec options");
    base.kv_fp8 = true;
    expect_accepts(base, 4, "TP4 accepts --kv-fp8 spec options");
    base.max_batch = 64;
    base.spec = false;
    expect_accepts(base, 2, "TP2 accepts --kv-fp8 batch options");

    auto with = [&](auto fn) {
      RuntimeOptions o;
      o.max_ctx = 65536;
      o.max_batch = 1;
      o.spec = true;
      fn(o);
      return o;
    };
    expect_rejects(with([](RuntimeOptions& o) { o.kv_q4 = true; }), 2, "TP rejects --kv-q4");
    expect_rejects(with([](RuntimeOptions& o) { o.kv_paged = true; }), 4, "TP rejects --kv-paged");
    expect_rejects(with([](RuntimeOptions& o) { o.kv_vram_mb = 512; }), 4, "TP rejects --kv-vram-mb");
    expect_rejects(with([](RuntimeOptions& o) { o.kv_swap_dir = "/tmp/kv_swap"; }), 2, "TP rejects --kv-swap-dir");
    expect_rejects(with([](RuntimeOptions& o) { o.kv_cache_mode = KvCacheMode::LastFull; }), 2,
                   "TP rejects --kv-cache-mode last_full");
    expect_rejects(with([](RuntimeOptions& o) { o.kv_cache_mode = KvCacheMode::LastFull; }), 4,
                   "TP4 rejects --kv-cache-mode last_full");
    expect_rejects(with([](RuntimeOptions& o) { o.vision = true; }), 2, "TP rejects --vision");
    expect_rejects(with([](RuntimeOptions& o) { o.offload_vram_mb = 4096; }), 4, "TP rejects --offload-vram-mb");
    expect_rejects(with([](RuntimeOptions& o) { o.kv_ckpt_path = "/tmp/ckpt"; }), 4, "TP rejects --kv-ckpt-path");
    expect_rejects(with([](RuntimeOptions& o) { o.enable_persona_hooks = true; }), 2, "TP rejects persona hooks");
    expect_rejects(with([](RuntimeOptions& o) { o.max_ctx = 8; }), 2, "TP rejects max_ctx < 16");
    expect_rejects(with([](RuntimeOptions& o) { o.max_batch = 0; }), 2, "TP rejects max_batch < 1");

#if INGINE_TP_SIZE > 1
    // Verify the public model::make_generator() entry point rejects unsupported KV/feature options before allocation
    // on both B=1 spec (TpSpecGenerator) and batch/plain (TpBatchGenerator) routes.
    auto expect_make_gen_rejects = [&](const RuntimeOptions& o, const char* what) {
      bool threw = false;
      try {
        (void)make_generator(o);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      expect(threw, what);
    };
    for (bool spec_mode : {true, false}) {
      auto mk = [&](auto fn) {
        RuntimeOptions o;
        o.max_ctx = 65536;
        o.max_batch = spec_mode ? 1 : 4;
        o.spec = spec_mode;
        fn(o);
        return o;
      };
      expect_make_gen_rejects(mk([](RuntimeOptions& o) { o.kv_q4 = true; }),
                              spec_mode ? "make_generator(spec) rejects --kv-q4" : "make_generator(batch) rejects --kv-q4");
      expect_make_gen_rejects(mk([](RuntimeOptions& o) { o.kv_paged = true; }),
                              spec_mode ? "make_generator(spec) rejects --kv-paged" : "make_generator(batch) rejects --kv-paged");
      expect_make_gen_rejects(mk([](RuntimeOptions& o) { o.kv_vram_mb = 512; }),
                              spec_mode ? "make_generator(spec) rejects --kv-vram-mb" : "make_generator(batch) rejects --kv-vram-mb");
      expect_make_gen_rejects(mk([](RuntimeOptions& o) { o.kv_cache_mode = KvCacheMode::LastFull; }),
                              spec_mode ? "make_generator(spec) rejects --kv-cache-mode last_full"
                                        : "make_generator(batch) rejects --kv-cache-mode last_full");
      expect_make_gen_rejects(mk([](RuntimeOptions& o) { o.vision = true; }),
                              spec_mode ? "make_generator(spec) rejects --vision" : "make_generator(batch) rejects --vision");
    }
#endif
  }

  // 4. When >= 2 CUDA GPUs are visible, run live P2P probe + fail-closed synthetic corruption test.
  //    Distinguish live multi-GPU coverage from host-only coverage explicitly in output (and fail if
  //    INGINE_TP_REQUIRE_LIVE_P2P=1 is set without >= 2 P2P-capable GPUs).
  const char* req_live_env = std::getenv("INGINE_TP_REQUIRE_LIVE_P2P");
  const bool require_live = (req_live_env && req_live_env[0] == '1');
  int dev_count = 0;
  bool ran_live_p2p = false;
  int live_gpus = 0;
  int live_capable = 0;
  int live_verified = 0;
  if (cudaGetDeviceCount(&dev_count) == cudaSuccess && dev_count >= 2) {
    const int n = dev_count >= 4 ? 4 : 2;
    const TpP2pProbeResult ok = probe_and_enable_tp_p2p(n, false);
    std::cout << "live P2P probe (" << n << " GPUs): capable=" << ok.capable_pairs
              << " verified=" << ok.verified_pairs << " enabled=" << ok.p2p_enabled
              << " corrupted=" << ok.corrupted << "\n";
    expect(!ok.corrupted, "live P2P probe reports no corruption");
    expect(ok.verified_pairs == ok.capable_pairs, "live P2P probe verifies every capable pair");
    expect(ok.p2p_enabled == (ok.capable_pairs > 0), "live P2P enabled matches capable pair presence");
    if (require_live) expect(ok.capable_pairs > 0, "require-live mode found >= 1 P2P-capable pair");

    if (ok.capable_pairs > 0) {
      ran_live_p2p = true;
      live_gpus = n;
      live_capable = ok.capable_pairs;
      live_verified = ok.verified_pairs;
      const TpP2pProbeResult bad = probe_and_enable_tp_p2p(n, true);
      std::cout << "forced-fail P2P probe: corrupted=" << bad.corrupted << " enabled=" << bad.p2p_enabled
                << " bad_pair=(" << bad.first_bad_src << "->" << bad.first_bad_dst << ") mismatches="
                << bad.first_bad_mismatches << " detail=" << bad.detail << "\n";
      expect(bad.corrupted, "forced-fail P2P probe flags corruption");
      expect(!bad.p2p_enabled, "forced-fail P2P probe disables peer access");
      expect(bad.first_bad_src >= 0 && bad.first_bad_dst >= 0, "forced-fail P2P probe reports bad pair");
      expect(bad.first_bad_mismatches == 1, "forced-fail P2P probe detects exact injected mismatch");
    }
  } else {
    std::cout << "CUDA multi-GPU probe not run (visible GPUs=" << dev_count << ")\n";
    if (require_live) expect(false, "INGINE_TP_REQUIRE_LIVE_P2P=1 requires >= 2 visible CUDA GPUs");
  }

  if (g_failures > 0) {
    std::fprintf(stderr, "model_tp_p2p: FAIL (%d failure(s))\n", g_failures);
    return 1;
  }
  if (ran_live_p2p) {
    std::cout << "model_tp_p2p: LIVE+HOST PASS (" << live_gpus << " GPUs, " << live_verified << "/" << live_capable
              << " directed pairs verified + injected failure caught)\n";
  } else {
    std::cout << "model_tp_p2p: HOST-ONLY PASS (live multi-GPU P2P probe not run)\n";
  }
  return 0;
}

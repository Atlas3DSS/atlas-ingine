// DSpark conditional-graph fallback test (GPU MODEL): forces INGINE_DISABLE_CONDITIONAL_GRAPHS=1 and verifies
// bit-exact parity against normal conditional-graph execution across multi-turn greedy decode, teacher-forced
// replay (widths, accepted/rejected counts, spec_trace), rollback state continuation, hooked persona verify,
// actual SWITCH vs eager launch assertions, and induced unsupported-SWITCH fallback after probe success.
#include "common/test_util.h"
#include "core/hal.h"
#include "ingine/generator.h"
#include "ingine/persona.h"
#include "model/runtime.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace ingine::test;
namespace model = ingine::model;
namespace persona = ingine::persona;

namespace {

int g_failures = 0;

void expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

const std::vector<std::int32_t> kPrompt = {
    200002, 200004, 1735,   149914, 220,    65281,  6,      82,     194368, 163872, 171159, 13,     68014,  2,
    154675, 198,    12,     167508, 1255,   11,     150126, 151894, 12,     2409,   152725, 69137,  13,     165179,
    154393, 152059, 26,     152293, 220,    67211,  150665, 149882, 150261, 1528,   183763, 13,     198,    12,
    194063, 67,     150406, 25,     199531, 152297, 784,    168047, 11,     152145, 279,    11,     167527, 13,
    198,    12,     150058, 139526, 166183, 151122, 149914, 149876, 150409, 6,      82,     26,     200010, 200000,
    200004, 3,      200010, 200001};

struct EnvRestorer {
  std::optional<std::string> prev;
  explicit EnvRestorer(const char* name) : name_(name) {
    if (const char* v = std::getenv(name)) prev = std::string(v);
  }
  ~EnvRestorer() {
    if (prev) setenv(name_, prev->c_str(), 1);
    else unsetenv(name_);
  }
  const char* name_;
};

struct FailSwitchGuard {
  ~FailSwitchGuard() { model::fail_next_switch_capture(cudaSuccess); }
};

struct RunCapture {
  bool conditional_graphs_enabled = false;
  std::vector<std::int32_t> turn1_out;
  model::GenerateStats turn1_stats;
  model::SpecExecutionCounters turn1_exec;
  std::vector<std::int32_t> turn2_out;
  model::GenerateStats turn2_stats;
  model::SpecExecutionCounters turn2_exec;
  std::vector<std::int32_t> forced_out;
  model::GenerateStats forced_stats;
  model::SpecExecutionCounters forced_exec;
  std::vector<std::int32_t> forced_followup_out;
  model::GenerateStats forced_followup_stats;
  model::SpecExecutionCounters forced_followup_exec;
  std::vector<std::int32_t> hooked_out;
  model::GenerateStats hooked_stats;
  model::SpecExecutionCounters hooked_exec;
  std::vector<std::int32_t> unhooked_after_out;
  model::GenerateStats unhooked_after_stats;
  model::SpecExecutionCounters unhooked_after_exec;
  std::vector<std::int32_t> induced_forced_out;
  model::GenerateStats induced_forced_stats;
  model::SpecExecutionCounters induced_forced_exec;
  std::vector<std::int32_t> induced_followup_out;
  model::GenerateStats induced_followup_stats;
  model::SpecExecutionCounters induced_followup_exec;
};

model::GenerateStats collect(model::Generator& gen, std::span<const std::int32_t> prompt, int max_tokens,
                             const model::Sampling& s, std::vector<std::int32_t>& out,
                             model::SpecExecutionCounters& exec_out) {
  out.clear();
  model::reset_spec_execution_counters();
  auto st = gen.generate(prompt, max_tokens, s, {}, [&](std::span<const std::int32_t> toks) {
    out.insert(out.end(), toks.begin(), toks.end());
    return true;
  });
  exec_out = model::spec_execution_counters();
  return st;
}

bool used_switch_only(const model::SpecExecutionCounters& c) {
  return c.switch_verify_launches > 0 && c.switch_accept_launches > 0 &&
         c.eager_verify_launches == 0 && c.eager_accept_launches == 0;
}

bool used_eager_only(const model::SpecExecutionCounters& c) {
  return c.switch_verify_launches == 0 && c.switch_accept_launches == 0 &&
         c.eager_verify_launches > 0 && c.eager_accept_launches > 0;
}

RunCapture run_suite(const std::filesystem::path& dir, bool disable_conditional_graphs,
                     std::shared_ptr<const persona::PersonaProfile> prof, bool inject_switch_failure) {
  if (disable_conditional_graphs) setenv("INGINE_DISABLE_CONDITIONAL_GRAPHS", "1", 1);
  else unsetenv("INGINE_DISABLE_CONDITIONAL_GRAPHS");

  int active_dev = 0;
  INGINE_TEST_CUDA(cudaGetDevice(&active_dev));
  const bool cond_enabled = ingine::hal::conditional_graphs_enabled(active_dev);
  FailSwitchGuard fail_guard;

  model::RuntimeOptions opt;
  opt.model_dir = dir;
  opt.max_ctx = 8192;
  opt.dense_fp8 = true;
  opt.kv_fp8 = true;
  opt.spec = true;
  opt.enable_persona_hooks = true;

  auto gen = model::make_generator(opt);
  RunCapture cap;
  cap.conditional_graphs_enabled = cond_enabled;

  model::Sampling greedy;
  greedy.temperature = 0.f;

  // 1) Turn 1 greedy speculative decode
  cap.turn1_stats = collect(*gen, kPrompt, 128, greedy, cap.turn1_out, cap.turn1_exec);

  // 2) Turn 2 continuation extending Turn 1's state (exercises post-rollback / end-of-turn KV+conv state)
  std::vector<std::int32_t> cont_prompt = kPrompt;
  cont_prompt.insert(cont_prompt.end(), cap.turn1_out.begin(), cap.turn1_out.end());
  cap.turn2_stats = collect(*gen, cont_prompt, 64, greedy, cap.turn2_out, cap.turn2_exec);

  // 3) Forced replay of a truncated slice followed by immediate continuation decode
  std::vector<std::int32_t> forced_target(
      cap.turn1_out.begin(), cap.turn1_out.begin() + std::min<std::size_t>(48, cap.turn1_out.size()));
  if (forced_target.size() >= 12) {
    forced_target[10] = (forced_target[10] + 1) % 200000;  // force a draft rejection & rollback event
  }
  model::Sampling forced_s = greedy;
  forced_s.forced = forced_target;
  cap.forced_stats = collect(*gen, kPrompt, int(forced_target.size()), forced_s, cap.forced_out, cap.forced_exec);

  std::vector<std::int32_t> post_forced_prompt = kPrompt;
  post_forced_prompt.insert(post_forced_prompt.end(), cap.forced_out.begin(), cap.forced_out.end());
  cap.forced_followup_stats =
      collect(*gen, post_forced_prompt, 32, greedy, cap.forced_followup_out, cap.forced_followup_exec);

  // 4) Hooked persona speculative decode + return to unhooked decode
  model::Sampling hooked_s = greedy;
  hooked_s.persona = prof;
  cap.hooked_stats = collect(*gen, kPrompt, 64, hooked_s, cap.hooked_out, cap.hooked_exec);
  cap.unhooked_after_stats = collect(*gen, kPrompt, 64, greedy, cap.unhooked_after_out, cap.unhooked_after_exec);

  // 5) Induced unsupported SWITCH error after HAL probe succeeded (uses banned_tokens to trigger prepare_accept):
  //    exercises controlled transition from SWITCH -> eager fallback, rollback, and continuation.
  model::Sampling induced_s = forced_s;
  induced_s.banned_tokens = {0};
  if (inject_switch_failure) model::fail_next_switch_capture(cudaErrorUnsupportedPtxVersion);
  cap.induced_forced_stats =
      collect(*gen, kPrompt, int(forced_target.size()), induced_s, cap.induced_forced_out, cap.induced_forced_exec);
  std::vector<std::int32_t> post_induced_prompt = kPrompt;
  post_induced_prompt.insert(post_induced_prompt.end(), cap.induced_forced_out.begin(),
                             cap.induced_forced_out.end());
  cap.induced_followup_stats =
      collect(*gen, post_induced_prompt, 32, greedy, cap.induced_followup_out, cap.induced_followup_exec);

  return cap;
}

bool traces_equal(const std::vector<model::SpecRoundRecord>& a, const std::vector<model::SpecRoundRecord>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].width != b[i].width || a[i].accepted != b[i].accepted || a[i].accepted_all != b[i].accepted_all) {
      return false;
    }
  }
  return true;
}

bool has_reject_or_partial(const std::vector<model::SpecRoundRecord>& trace) {
  for (const auto& r : trace) {
    if (r.accepted < r.width - 1) return true;
  }
  return false;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  const char* env_dir = std::getenv("INGINE_MODEL_DIR");
  const std::filesystem::path dir = (env_dir && *env_dir) ? std::filesystem::path(env_dir)
                                                          : std::filesystem::path(INGINE_MODEL_DIR);
  if (!std::filesystem::exists(dir / "mtp.safetensors")) {
    std::printf("skip: no model in %s\n", dir.c_str());
    return kSkip;
  }

  EnvRestorer guard("INGINE_DISABLE_CONDITIONAL_GRAPHS");

  auto prof = std::make_shared<persona::PersonaProfile>();
  prof->name = "spec_fallback_router_force";
  prof->fingerprint_sha256 = persona::sha256_hex(prof->name);
  prof->cache_key = persona::cache_key_from_hex(prof->fingerprint_sha256);
  for (int visit : {10, 14, 18, 22, 26, 30}) {
    persona::InterventionSpec iv;
    iv.type = persona::InterventionType::RouterForce;
    iv.visit = visit;
    iv.expert = 5;
    iv.bias = 100.0f;
    iv.gain = 12.0f;
    prof->interventions.push_back(iv);
  }
  persona::validate_profile_safety(*prof, persona::RefusalMap::default_map());

  unsetenv("INGINE_DISABLE_CONDITIONAL_GRAPHS");
  int active_dev = 0;
  INGINE_TEST_CUDA(cudaGetDevice(&active_dev));
  const bool hw_supports_switch = ingine::hal::conditional_graphs_enabled(active_dev);

  if (!hw_supports_switch) {
    std::printf("INFO paired SWITCH-vs-eager comparison unavailable on this device (conditional SWITCH unsupported); "
                "eager path qualified\n");
    const RunCapture eager =
        run_suite(dir, /*disable_conditional_graphs=*/false, prof, /*inject_switch_failure=*/false);
    expect(!eager.conditional_graphs_enabled, "device reports conditional SWITCH unavailable");
    expect(eager.turn1_stats.spec_rounds > 0 && !eager.turn1_out.empty(), "eager run executed speculative rounds");
    expect(used_eager_only(eager.turn1_exec) && used_eager_only(eager.turn2_exec) &&
               used_eager_only(eager.forced_exec) && used_eager_only(eager.forced_followup_exec) &&
               used_eager_only(eager.hooked_exec) && used_eager_only(eager.unhooked_after_exec) &&
               used_eager_only(eager.induced_forced_exec) && used_eager_only(eager.induced_followup_exec),
           "all phases executed on the eager speculative path without SWITCH");
    expect(has_reject_or_partial(eager.forced_stats.spec_trace), "eager forced replay exercised draft rejection / rollback");
    const std::vector<std::int32_t> t1_prefix(
        eager.turn1_out.begin(), eager.turn1_out.begin() + std::min<std::size_t>(64, eager.turn1_out.size()));
    expect(eager.hooked_out != t1_prefix, "hooked persona profile alters output vs unhooked baseline on eager path");
    expect(eager.unhooked_after_out == t1_prefix, "unhooked decode after hooked request restores baseline on eager path");
    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASSED" : "FAILED", g_failures);
    return g_failures == 0 ? 0 : 1;
  }

  // Run sequentially so only one model instance occupies GPU VRAM at a time.
  const RunCapture normal = run_suite(dir, /*disable_conditional_graphs=*/false, prof, /*inject_switch_failure=*/true);
  const RunCapture fallback = run_suite(dir, /*disable_conditional_graphs=*/true, prof, /*inject_switch_failure=*/false);

  expect(normal.conditional_graphs_enabled && !fallback.conditional_graphs_enabled,
         "env toggle disabled conditional SWITCH graphs for fallback run");
  expect(normal.turn1_stats.spec_rounds > 0 && !normal.turn1_out.empty(), "normal run executed speculative rounds");
  expect(used_switch_only(normal.turn1_exec) && used_switch_only(normal.turn2_exec) &&
             used_switch_only(normal.forced_exec) && used_switch_only(normal.forced_followup_exec) &&
             used_switch_only(normal.hooked_exec) && used_switch_only(normal.unhooked_after_exec),
         "normal arm phases 1-6 executed actual SWITCH verify/accept graph launches");
  expect(used_eager_only(fallback.turn1_exec) && used_eager_only(fallback.turn2_exec) &&
             used_eager_only(fallback.forced_exec) && used_eager_only(fallback.forced_followup_exec) &&
             used_eager_only(fallback.hooked_exec) && used_eager_only(fallback.unhooked_after_exec) &&
             used_eager_only(fallback.induced_forced_exec) && used_eager_only(fallback.induced_followup_exec),
         "disabled arm executed strictly eager verify/accept paths");

  expect(normal.turn1_out == fallback.turn1_out &&
             normal.turn1_stats.generated_tokens == fallback.turn1_stats.generated_tokens &&
             normal.turn1_stats.spec_rounds == fallback.turn1_stats.spec_rounds &&
             normal.turn1_stats.spec_accepted == fallback.turn1_stats.spec_accepted,
         "turn1 token ids, spec_rounds, spec_accepted match bit-for-bit under fallback");

  expect(normal.turn2_out == fallback.turn2_out &&
             normal.turn2_stats.reuse == fallback.turn2_stats.reuse &&
             normal.turn2_stats.reused_tokens == fallback.turn2_stats.reused_tokens &&
             normal.turn2_stats.spec_rounds == fallback.turn2_stats.spec_rounds &&
             normal.turn2_stats.spec_accepted == fallback.turn2_stats.spec_accepted,
         "turn2 extended continuation & reuse match bit-for-bit under fallback");

  expect(normal.forced_out == fallback.forced_out &&
             normal.forced_stats.spec_rounds == fallback.forced_stats.spec_rounds &&
             normal.forced_stats.spec_accepted == fallback.forced_stats.spec_accepted &&
             traces_equal(normal.forced_stats.spec_trace, fallback.forced_stats.spec_trace),
         "forced replay token ids, accepted counts, widths, and spec_trace match under fallback");

  expect(has_reject_or_partial(fallback.forced_stats.spec_trace), "forced replay exercised draft rejection / rollback");
  expect(normal.forced_followup_out == fallback.forced_followup_out &&
             normal.forced_followup_stats.spec_rounds == fallback.forced_followup_stats.spec_rounds &&
             normal.forced_followup_stats.spec_accepted == fallback.forced_followup_stats.spec_accepted,
         "post-rollback follow-up decode matches bit-for-bit under fallback");

  expect(normal.hooked_out == fallback.hooked_out &&
             normal.hooked_stats.spec_rounds == fallback.hooked_stats.spec_rounds &&
             normal.hooked_stats.spec_accepted == fallback.hooked_stats.spec_accepted,
         "hooked persona speculative verify matches bit-for-bit under fallback");

  const std::vector<std::int32_t> t1_prefix(
      normal.turn1_out.begin(), normal.turn1_out.begin() + std::min<std::size_t>(64, normal.turn1_out.size()));
  std::size_t hooked_diff = 0;
  const std::size_t cmp_len = std::min(normal.hooked_out.size(), t1_prefix.size());
  for (std::size_t i = 0; i < cmp_len; ++i) {
    if (normal.hooked_out[i] != t1_prefix[i]) ++hooked_diff;
  }
  if (normal.hooked_out.size() != t1_prefix.size()) {
    hooked_diff += (normal.hooked_out.size() > t1_prefix.size() ? normal.hooked_out.size() - t1_prefix.size()
                                                                : t1_prefix.size() - normal.hooked_out.size());
  }
  std::printf("INFO hooked vs unhooked differing tokens: %zu / %zu\n", hooked_diff, t1_prefix.size());
  expect(normal.hooked_out != t1_prefix, "hooked persona profile alters output vs unhooked baseline");
  expect(fallback.unhooked_after_out == normal.unhooked_after_out && fallback.unhooked_after_out == t1_prefix,
         "unhooked speculative decode after hooked request matches baseline under fallback");

  // Phase 5 checks: induced unsupported SWITCH construction after successful probe.
  expect(normal.induced_forced_exec.switch_capture_fallbacks >= 1 &&
             used_eager_only(normal.induced_forced_exec) &&
             used_eager_only(normal.induced_followup_exec),
         "induced unsupported SWITCH construction transitions cleanly to eager verify/accept + follow-up");
  expect(has_reject_or_partial(normal.induced_forced_stats.spec_trace) &&
             normal.induced_forced_out == fallback.induced_forced_out &&
             traces_equal(normal.induced_forced_stats.spec_trace, fallback.induced_forced_stats.spec_trace) &&
             normal.induced_followup_out == fallback.induced_followup_out,
         "induced SWITCH->eager fallback preserves forced rollback and continuation bit-for-bit");

  std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASSED" : "FAILED", g_failures);
  return g_failures == 0 ? 0 : 1;
}

// DSpark conditional-graph fallback test (GPU MODEL): forces INGINE_DISABLE_CONDITIONAL_GRAPHS=1 and verifies
// bit-exact parity against normal conditional-graph execution across multi-turn greedy decode, teacher-forced
// replay (widths, accepted/rejected counts, spec_trace), rollback state continuation, and hooked persona verify.
#include "common/test_util.h"
#include "ingine/generator.h"
#include "ingine/persona.h"

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

struct RunCapture {
  std::vector<std::int32_t> turn1_out;
  model::GenerateStats turn1_stats;
  std::vector<std::int32_t> turn2_out;
  model::GenerateStats turn2_stats;
  std::vector<std::int32_t> forced_out;
  model::GenerateStats forced_stats;
  std::vector<std::int32_t> forced_followup_out;
  model::GenerateStats forced_followup_stats;
  std::vector<std::int32_t> hooked_out;
  model::GenerateStats hooked_stats;
  std::vector<std::int32_t> unhooked_after_out;
  model::GenerateStats unhooked_after_stats;
};

model::GenerateStats collect(model::Generator& gen, std::span<const std::int32_t> prompt, int max_tokens,
                             const model::Sampling& s, std::vector<std::int32_t>& out) {
  out.clear();
  return gen.generate(prompt, max_tokens, s, {}, [&](std::span<const std::int32_t> toks) {
    out.insert(out.end(), toks.begin(), toks.end());
    return true;
  });
}

RunCapture run_suite(const std::filesystem::path& dir, bool disable_conditional_graphs,
                     std::shared_ptr<const persona::PersonaProfile> prof) {
  if (disable_conditional_graphs) setenv("INGINE_DISABLE_CONDITIONAL_GRAPHS", "1", 1);
  else unsetenv("INGINE_DISABLE_CONDITIONAL_GRAPHS");

  model::RuntimeOptions opt;
  opt.model_dir = dir;
  opt.max_ctx = 8192;
  opt.dense_fp8 = true;
  opt.kv_fp8 = true;
  opt.spec = true;
  opt.enable_persona_hooks = true;

  auto gen = model::make_generator(opt);
  RunCapture cap;

  model::Sampling greedy;
  greedy.temperature = 0.f;

  // 1) Turn 1 greedy speculative decode
  cap.turn1_stats = collect(*gen, kPrompt, 128, greedy, cap.turn1_out);

  // 2) Turn 2 continuation extending Turn 1's state (exercises post-rollback / end-of-turn KV+conv state)
  std::vector<std::int32_t> cont_prompt = kPrompt;
  cont_prompt.insert(cont_prompt.end(), cap.turn1_out.begin(), cap.turn1_out.end());
  cap.turn2_stats = collect(*gen, cont_prompt, 64, greedy, cap.turn2_out);

  // 3) Forced replay of a truncated slice followed by immediate continuation decode
  std::vector<std::int32_t> forced_target(
      cap.turn1_out.begin(), cap.turn1_out.begin() + std::min<std::size_t>(48, cap.turn1_out.size()));
  if (forced_target.size() >= 12) {
    forced_target[10] = (forced_target[10] + 1) % 200000;  // force a draft rejection & rollback event
  }
  model::Sampling forced_s = greedy;
  forced_s.forced = forced_target;
  cap.forced_stats = collect(*gen, kPrompt, int(forced_target.size()), forced_s, cap.forced_out);

  std::vector<std::int32_t> post_forced_prompt = kPrompt;
  post_forced_prompt.insert(post_forced_prompt.end(), cap.forced_out.begin(), cap.forced_out.end());
  cap.forced_followup_stats = collect(*gen, post_forced_prompt, 32, greedy, cap.forced_followup_out);

  // 4) Hooked persona speculative decode + return to unhooked decode
  model::Sampling hooked_s = greedy;
  hooked_s.persona = prof;
  cap.hooked_stats = collect(*gen, kPrompt, 64, hooked_s, cap.hooked_out);
  cap.unhooked_after_stats = collect(*gen, kPrompt, 64, greedy, cap.unhooked_after_out);

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
  {
    persona::InterventionSpec iv;
    iv.type = persona::InterventionType::RouterForce;
    iv.visit = 10;
    iv.expert = 5;
    iv.bias = 25.0f;
    iv.gain = 2.5f;
    prof->interventions.push_back(iv);
  }
  persona::validate_profile_safety(*prof, persona::RefusalMap::default_map());

  // Run sequentially so only one model instance occupies GPU VRAM at a time.
  const RunCapture normal = run_suite(dir, /*disable_conditional_graphs=*/false, prof);
  const RunCapture fallback = run_suite(dir, /*disable_conditional_graphs=*/true, prof);

  expect(normal.turn1_stats.spec_rounds > 0 && !normal.turn1_out.empty(), "normal run executed speculative rounds");
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

  // Confirm at least one rollback occurred in forced replay or turn1 and post-rollback continuation matches.
  bool saw_partial_or_reject = false;
  for (const auto& r : fallback.forced_stats.spec_trace) {
    if (r.accepted < r.width - 1) saw_partial_or_reject = true;
  }
  expect(saw_partial_or_reject, "forced replay exercised draft rejection / rollback");
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
  expect(normal.hooked_out != t1_prefix, "hooked persona profile alters output vs unhooked baseline");
  expect(fallback.unhooked_after_out == normal.unhooked_after_out && fallback.unhooked_after_out == t1_prefix,
         "unhooked speculative decode after hooked request matches baseline under fallback");

  std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PASSED" : "FAILED", g_failures);
  return g_failures == 0 ? 0 : 1;
}

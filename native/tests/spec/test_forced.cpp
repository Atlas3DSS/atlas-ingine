// Teacher-forced replay (Sampling::forced) with the full model: replaying a speculative greedy generation's
// own output must emit exactly that output and repeat its rounds (same drafts, widths and acceptances: the
// forced token after each accepted run is the target's argmax there); a plain generator replays it too.
#include "common/test_util.h"
#include "ingine/generator.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <numeric>
#include <vector>

using namespace ingine::test;
namespace model = ingine::model;

namespace {

int g_failures = 0;

void expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

// the start of a captured agent prompt (system message)
const std::vector<std::int32_t> kPrompt = {
    200002, 200004, 1735,   149914, 220,    65281,  6,      82,     194368, 163872, 171159, 13,     68014,  2,
    154675, 198,    12,     167508, 1255,   11,     150126, 151894, 12,     2409,   152725, 69137,  13,     165179,
    154393, 152059, 26,     152293, 220,    67211,  150665, 149882, 150261, 1528,   183763, 13,     198,    12,
    194063, 67,     150406, 25,     199531, 152297, 784,    168047, 11,     152145, 279,    11,     167527, 13,
    198,    12,     150058, 139526, 166183, 151122, 149914, 149876, 150409, 6,      82,     26,     200010, 200000,
    200004, 3,      200010, 200001};

model::GenerateStats run(model::Generator& gen, int tokens, const model::Sampling& s, std::vector<std::int32_t>& out) {
  out.clear();
  return gen.generate(kPrompt, tokens, s, {}, [&](std::span<const std::int32_t> t) {
    out.insert(out.end(), t.begin(), t.end());
    return true;
  });
}

#if INGINE_TP_SIZE > 1
void check_tp_checkpoint_scope(model::Generator& gen, const model::Sampling& sampling) {
  auto prefill = [&](const model::Prompt& prompt) {
    return gen.generate(prompt, 1, sampling, {}, [](auto) { return true; }).reused_tokens;
  };
  model::Prompt a{kPrompt, {}, 0, "a"};
  model::Prompt b{kPrompt, {}, 0, "b"};
  const int n = int(kPrompt.size());
  expect(prefill(a) == 0 && prefill(a) == n, "TP checkpoint warms within one namespace");
  expect(prefill(b) == 0 && prefill(b) == n, "same text starts cold in a second namespace");
  auto extended = kPrompt;
  extended.push_back(42);
  a.ids = extended;
  expect(prefill(a) == 0, "extended prompt starts cold across namespaces");
  a.bypass_prefix_cache = true;
  expect(prefill(a) == 0 && prefill(a) == 0, "bypassed requests always prefill");
  a.bypass_prefix_cache = false;
  expect(prefill(a) == 0 && prefill(a) == int(extended.size()), "bypassed prefill leaves no reusable checkpoint");
}

void check_tp_token_bounds(model::Generator& gen, bool batch) {
  const auto rejected = [](auto&& call) {
    try { call(); } catch (const std::out_of_range&) { return true; }
    return false;
  };
  const std::int32_t outside[] = {std::numeric_limits<std::int32_t>::max()};
  const std::int32_t negative[] = {-1};
  model::Sampling sampling;
  auto generate = [&](std::span<const std::int32_t> ids) {
    gen.generate(ids, 1, sampling, {}, [](auto) { return true; });
  };
  expect(rejected([&] { generate(outside); }), "TP validates prompt token bounds on the host");
  sampling.forced = negative;
  expect(rejected([&] { generate(kPrompt); }), "TP validates forced token bounds on the host");
  if (!batch) return;
  model::GenerateStats stats;
  std::int32_t first;
  sampling.forced = {};
  const int slot = gen.start_slot(model::Prompt{kPrompt, {}}, 2, sampling, first, stats);
  expect(slot >= 0, "TP batch slot available after rejected inputs");
  if (slot < 0) return;
  const model::BatchSlotStep invalid[] = {{slot, outside[0]}};
  const model::BatchSlotStep valid[] = {{slot, first}};
  std::int32_t out[1]{};
  expect(rejected([&] { gen.step_slots(invalid, out); }), "TP validates batch step token bounds before advancing");
  gen.step_slots(valid, out);
  gen.release_slot(slot);
}
#endif

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  const std::filesystem::path dir = INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "mtp.safetensors")) {
    std::printf("skip: no model in %s\n", dir.c_str());
    return kSkip;
  }
  model::RuntimeOptions o;
  o.model_dir = dir;
  o.max_ctx = 8192;
  o.dense_fp8 = true;
  o.kv_fp8 = true;
  const int tokens = 160;
  std::vector<std::int32_t> greedy_out, out;
  {
    o.spec = true;
    auto gen = model::make_generator(o);
    model::Sampling greedy;
    greedy.temperature = 0.f;
    const model::GenerateStats ref = run(*gen, tokens, greedy, greedy_out);
    model::Sampling forced = greedy;
    forced.forced = greedy_out;
    const model::GenerateStats st = run(*gen, tokens, forced, out);
    std::printf("greedy: %d rounds, %d accepted; forced: %d rounds, %d accepted, %zu traced\n", ref.spec_rounds,
                ref.spec_accepted, st.spec_rounds, st.spec_accepted, st.spec_trace.size());
    expect(out == greedy_out, "speculative forced replay emits the forced output");
    // The greedy run's last round may accept drafts past the token budget (1 + rounds + accepted tokens were
    // produced, the first `tokens` emitted); the replay has no forced tokens there to accept.
    const int past_budget = std::max(0, ref.spec_rounds + ref.spec_accepted - tokens);
    expect(st.spec_rounds == ref.spec_rounds && st.spec_accepted == ref.spec_accepted - past_budget,
           "forced replay of the greedy output repeats its rounds");
    int traced = 0;
    for (const auto& r : st.spec_trace) traced += r.accepted;
    bool sane = st.spec_trace.size() == std::size_t(st.spec_rounds) && traced == st.spec_accepted;
    for (const auto& r : st.spec_trace)
      sane &= (r.width == 2 || r.width == 4 || r.width == 8) && r.accepted < r.width && r.accepted <= r.accepted_all &&
              (r.accepted_all == r.accepted || r.accepted == r.width - 1) && r.verify_ms > 0 && r.draft_ms > 0;
    expect(sane, "trace: one record per round, widths 2/4/8, accepted <= accepted_all, timings");
    // a forced output the model would not produce: every token emitted as given
    std::vector<std::int32_t> other(greedy_out.rbegin(), greedy_out.rend());
    forced.forced = other;
    run(*gen, tokens, forced, out);
    expect(out == other, "speculative forced replay of an arbitrary output");
    // forced length cap: requesting more tokens than forced.size() caps emission at forced.size()
    const std::vector<std::int32_t> short_forced(other.begin(), other.begin() + 17);
    forced.forced = short_forced;
    const model::GenerateStats cap_st = run(*gen, tokens, forced, out);
    expect(out == short_forced && cap_st.generated_tokens == int(short_forced.size()) &&
               cap_st.spec_trace.size() == std::size_t(cap_st.spec_rounds),
           "speculative forced replay caps max_new_tokens at forced.size()");
#if INGINE_TP_SIZE > 1
    check_tp_checkpoint_scope(*gen, greedy);
    check_tp_token_bounds(*gen, false);
#endif
  }
  {
    o.spec = false;
    auto gen = model::make_generator(o);
    model::Sampling forced;
    forced.temperature = 1.f;
    forced.seed = 7;
    forced.forced = greedy_out;
    const model::GenerateStats st = run(*gen, tokens, forced, out);
    expect(out == greedy_out && st.spec_trace.empty(), "plain forced replay emits the forced output");
    const std::vector<std::int32_t> short_forced(greedy_out.begin(), greedy_out.begin() + 17);
    forced.forced = short_forced;
    const model::GenerateStats cap_st = run(*gen, tokens, forced, out);
    expect(out == short_forced && cap_st.generated_tokens == int(short_forced.size()) && cap_st.spec_trace.empty(),
           "plain forced replay caps max_new_tokens at forced.size()");
#if INGINE_TP_SIZE > 1
    check_tp_token_bounds(*gen, true);
#endif
  }
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}

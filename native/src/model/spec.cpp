#include "model/spec.h"

#include "core/device.h"
#include "kernels/draft/draft.h"
#include "model/model_kernels.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ingine::model {

namespace kd = kernels::draft;
namespace kh = kernels::head;

namespace {

std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static_assert(sizeof(VerifyRecord::drafts) / sizeof(std::int32_t) == kd::GAMMA);

Widths verify_widths(const Runtime& rt) {
  const std::span<const int> w = rt.verify_widths();
  if (w.empty() || w.size() > 3) throw std::invalid_argument("SpecLoop: the runtime has no verify widths (spec off)");
  Widths r;
  r.n = int(w.size());
  std::copy(w.begin(), w.end(), r.w);
  if (r.w[r.n - 1] < kd::BLOCK) throw std::invalid_argument("SpecLoop: the widest verify call must hold a block");
  return r;
}

}  // namespace

SpecLoop::SpecLoop(Runtime& rt, Drafter& drafter)
    : rt_(rt), drafter_(drafter), width_(sizeof(std::int32_t)), width_host_(sizeof(std::int32_t)),
      record_(sizeof(VerifyRecord)), forced_window_(kd::BLOCK * sizeof(std::int32_t)) {
  if (drafter.stream() != rt.stream()) throw std::invalid_argument("SpecLoop: drafter and runtime streams differ");
  const Widths widths = verify_widths(rt);
  // the draft outputs live at fixed addresses: one draft names them (its scratch rows are reset)
  draft_ = drafter.draft(0, 0, 0.f);
  drafter.reset();
  width_.fill(0);
  const float threshold = rt.options().spec_threshold;
  if (rt.has_verify_graph()) {
    verify_ = rt.try_capture_switch(
        [&](cudaStream_t s, cudaGraphConditionalHandle h) {
          return verify_select(draft_.conf, draft_.tokens, threshold, widths, rt_.tokens(),
                               static_cast<std::int32_t*>(width_.p), h, s);
        },
        [&](cudaStream_t s, int m) {
          rt_.record_save_verify(m, s);
          rt_.record_step(m, s);
        });
  }
  // where a round's outcome lives on the device: after the acceptance the drafter commits the accepted
  // rows and drafts for the next anchor from these (one graph, no host round trip in between)
  {
    const Drafter::AcceptScratch& scratch = drafter.accept_scratch();
    kd::RoundInputs round;
    round.nwidths = widths.n;
    for (int i = 0; i < widths.n; ++i) {
      round.widths[i] = widths.w[i];
      round.taps[i] = ptr<const ActT>(rt.buffers(widths.w[i]).taps);
    }
    round.m = static_cast<const std::int32_t*>(width_.p);
    round.accepted = &scratch.result->accepted;
    round.next_greedy = &scratch.result->next;
    round.next_sampled = &static_cast<VerifyRecord*>(record_.device_ptr())->next;  // sample_from_probs
    drafter.bind_round(round);
  }
  CUDA_CHECK(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&width_ready_, cudaEventDisableTiming));
  CUDA_CHECK(cudaStreamSynchronize(rt.stream()));
}

SpecLoop::~SpecLoop() {
  if (done_) cudaEventDestroy(done_);
  if (width_ready_) cudaEventDestroy(width_ready_);
}

void SpecLoop::ensure_verify_hooked() {
  if (verify_hooked_) return;
  rt_.ensure_hooked_graphs();
  if (!rt_.has_verify_graph()) return;
  const Widths widths = verify_widths(rt_);
  const float threshold = rt_.options().spec_threshold;
  verify_hooked_ = rt_.try_capture_switch(
      [&](cudaStream_t s, cudaGraphConditionalHandle h) {
        return verify_select(draft_.conf, draft_.tokens, threshold, widths, rt_.tokens(),
                             static_cast<std::int32_t*>(width_.p), h, s);
      },
      [&](cudaStream_t s, int m) {
        rt_.record_save_verify(m, s);
        rt_.record_step(m, s, /*hooked=*/true);
      });
  if (!verify_hooked_) {
    verify_ = GraphExec();
    accept_ = GraphExec();
  }
}

void SpecLoop::record_accept_for_width(int m, cudaStream_t stream) {
  const Widths widths = verify_widths(rt_);
  bool valid_width = false;
  for (int i = 0; i < widths.n; ++i) {
    if (widths.w[i] == m) {
      valid_width = true;
      break;
    }
  }
  if (!valid_width) {
    throw std::invalid_argument("SpecLoop::record_accept_for_width: invalid verify width " + std::to_string(m));
  }
  const int V = drafter_.vocab();
  const Drafter::AcceptScratch& scratch = drafter_.accept_scratch();
  const auto* width = static_cast<const std::int32_t*>(width_.p);
  auto* record = static_cast<VerifyRecord*>(record_.device_ptr());
  // forced replay: the drafts are compared with the forced tokens (the argmax still runs, for the timing)
  const auto* target =
      accept_key_.forced ? static_cast<const std::int32_t*>(forced_window_.device_ptr()) : scratch.target;
  const float* logits = static_cast<const float*>(rt_.buffers(m).logits.data);
  const int n = m - 1;
  if (accept_key_.greedy) {
    kh::argmax(logits, m, V, V, scratch.target, stream, accept_params_.banned);
    kd::greedy_accept(target, draft_.tokens, n, scratch.result, stream);
  } else {
    kh::sampling_probs(logits, m, V, V, accept_params_, scratch.probs, stream);
    kd::sampled_accept(scratch.probs, draft_.logits, V, draft_.tokens, n, accept_params_.temperature, draft_.qstats,
                       scratch.uniforms, scratch.ws, scratch.resid, scratch.result, stream);
  }
  rt_.record_rollback(m, &scratch.result->accepted, stream);
  verify_publish(width, &scratch.result->accepted, accept_key_.greedy ? &scratch.result->next : nullptr, rt_.tokens(),
                 record, stream);
}

void SpecLoop::prepare_accept(const Sampling& sampling) {
  const bool forced = !sampling.forced.empty();
  const bool greedy = sampling.temperature <= 0.f || forced;
  kh::BannedTokens banned{};
  for (std::int32_t id : sampling.banned_tokens) {
    if (id >= 0 && banned.count < kh::BannedTokens::kMax) banned.ids[banned.count++] = id;
  }
  std::array<std::int32_t, kh::BannedTokens::kMax> banned_arr{};
  for (int i = 0; i < kh::BannedTokens::kMax; ++i) banned_arr[i] = banned.ids[i];
  const AcceptKey key = greedy ? AcceptKey{true, forced, 0.f, 1.f, 0, banned.count, banned_arr}
                               : AcceptKey{false, false, sampling.temperature, sampling.top_p, sampling.top_k,
                                           banned.count, banned_arr};
  if (accept_prepared_ && key == accept_key_ && (!rt_.supports_conditional_graphs() || accept_)) return;
  const Widths widths = verify_widths(rt_);
  const kh::SamplingParams params{sampling.temperature, sampling.top_k, sampling.top_p, banned};
  const auto* width = static_cast<const std::int32_t*>(width_.p);
  accept_key_ = key;
  accept_params_ = params;
  accept_prepared_ = true;
  accept_ = GraphExec();
  if (!rt_.supports_conditional_graphs()) return;
  accept_ = rt_.try_capture_switch(
      [&](cudaStream_t s, cudaGraphConditionalHandle h) { return width_select(width, widths, h, s); },
      [&](cudaStream_t s, int m) { record_accept_for_width(m, s); });
  if (!accept_) {
    verify_ = GraphExec();
    verify_hooked_ = GraphExec();
  }
}

void SpecLoop::generate(std::int32_t token, int max_new_tokens, const Sampling& sampling, kh::SamplerRng& rng,
                        std::span<const std::int32_t> stop_ids,
                        const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats,
                        std::vector<std::int32_t>& tokens_in_state) {
  const cudaStream_t stream = rt_.stream();
  int P = int(tokens_in_state.size());  // position of the anchor (sampled, not yet processed)
  if (drafter_.length() != P) throw std::logic_error("SpecLoop: drafter context does not match the state");

  const bool forced = !sampling.forced.empty();
  const bool greedy = sampling.temperature <= 0.f || forced;  // forced: acceptance by equality
  const int V = drafter_.vocab();
  const Drafter::AcceptScratch& scratch = drafter_.accept_scratch();
  auto* record_dev = static_cast<VerifyRecord*>(record_.device_ptr());
  const auto* record = static_cast<const VerifyRecord*>(record_.data());
  prepare_accept(sampling);
  if (sampling.temperature > 0.f) drafter_.set_seed(splitmix64(rng.seed ^ splitmix64(rng.counter++)));

  int emitted = 0, rounds = 0, accepted = 0;
  std::vector<std::int32_t> out{token};
  // the last verify call (none before the first round): its position, width and accepted drafts
  int round_P = -1, round_m = 0, round_acc = 0;
  // forced replay: per round, events around the verify + acceptance and around the next draft
  struct RoundEvents {
    cudaEvent_t e[3];
  };
  std::vector<RoundEvents> events;
  auto record_event = [&](int i) {
    CUDA_CHECK(cudaEventCreate(&events.back().e[i]));
    CUDA_CHECK(cudaEventRecord(events.back().e[i], stream));
  };
  // the first draft; later ones are enqueued by the previous round (they leave the target state alone, so
  // the last one is wasted when the generation ends) and run while the host emits
  drafter_.draft(token, P, sampling.temperature);
  for (;;) {
    const std::int32_t anchor = out.back();

    // emit this round's tokens up to a stop token or the budget
    std::size_t k = 0;
    bool done = false;
    while (k < out.size()) {
      const std::int32_t tok = out[k++];
      ++emitted;
      if (std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end() || emitted == max_new_tokens) {
        done = true;
        break;
      }
    }
    stats.generated_tokens = emitted;
    const bool go_on = on_tokens(std::span<const std::int32_t>(out.data(), k));
    drafter_.note_tokens(std::span<const std::int32_t>(out.data(), k));  // for the drafts after the next one
    if (done || !go_on) {
      // The last emitted token stays unprocessed, as in plain decoding. If the generation ended inside the
      // round's accepted drafts (out[k - 1] with k - 1 < accepted), the state, the drafter and
      // tokens_in_state hold drafts past it: keep the verify call's first k rows (anchor + k - 1 drafts).
      const int keep = int(k) - 1;
      if (round_P >= 0 && keep < round_acc) {
        rt_.rollback(round_P, keep + 1, round_m);
        drafter_.commit(ptr<const ActT>(rt_.buffers(round_m).taps), keep + 1, round_P);
        tokens_in_state.resize(tokens_in_state.size() - std::size_t(round_acc - keep));
      }
      break;
    }

    // verify the confident drafts in the width the device picks, accept, roll back the rejected rows; then
    // commit the accepted rows and draft for the next anchor, all from device values
    const std::int32_t inputs[1] = {anchor};
    rt_.set_inputs(inputs, P);
    if (rt_.state().paged_kv) {
      // A verify call writes up to BLOCK (=8) consecutive positions; ensure pages across page boundaries exist.
      rt_.state().paged_kv->ensure_decode_pages(P, kd::BLOCK, stream);
    }
    if (forced) {  // the forced tokens after the anchor (forced[emitted - 1]); the accept graph reads them
      auto* w = static_cast<std::int32_t*>(forced_window_.data());
      for (int i = 0; i < kd::BLOCK; ++i)
        w[i] = std::size_t(emitted + i) < sampling.forced.size() ? sampling.forced[emitted + i] : 0;
      events.emplace_back();
      record_event(0);
    }
    if (!greedy) kh::uniform(kd::GAMMA, rng, scratch.uniforms, stream);
    if (rt_.hooked_active()) ensure_verify_hooked();
    int selected_m = 0;
    const bool use_verify_graph =
        rt_.has_verify_graph() && (rt_.hooked_active() ? bool(verify_hooked_) : bool(verify_));
    if (!use_verify_graph) {
      record_spec_execution(true, false);
      verify_select_eager(draft_.conf, draft_.tokens, rt_.options().spec_threshold, verify_widths(rt_), rt_.tokens(),
                          static_cast<std::int32_t*>(width_.p), static_cast<std::int32_t*>(width_host_.device_ptr()),
                          stream);
      CUDA_CHECK(cudaEventRecord(width_ready_, stream));
      CUDA_CHECK(cudaEventSynchronize(width_ready_));
      selected_m = *static_cast<const volatile std::int32_t*>(width_host_.data());
      rt_.record_save_verify(selected_m, stream);
      rt_.run_step(selected_m);
    } else if (rt_.hooked_active()) {
      record_spec_execution(true, true);
      verify_hooked_.launch(stream);
    } else {
      record_spec_execution(true, true);
      verify_.launch(stream);
    }
    if (rt_.supports_conditional_graphs() && accept_) {
      record_spec_execution(false, true);
      accept_.launch(stream);
    } else {
      record_spec_execution(false, false);
      if (selected_m == 0) {
        CUDA_CHECK(cudaMemcpyAsync(&selected_m, width_.p, sizeof(selected_m), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      record_accept_for_width(selected_m, stream);
    }
    if (!greedy) kh::sample_from_probs(scratch.resid, 1, V, V, rng, &record_dev->next, stream);
    CUDA_CHECK(cudaEventRecord(done_, stream));
    if (forced) record_event(1);
    drafter_.advance(P);
    if (forced) record_event(2);
    CUDA_CHECK(cudaEventSynchronize(done_));  // the round's only host synchronisation
    VerifyRecord r;
    std::memcpy(&r, record, sizeof r);
    const int acc = r.accepted;
    const int commit = acc + 1;
    round_P = P;
    round_m = r.m;
    round_acc = acc;
    if (forced) {  // the run all 7 drafts would have reached
      int all = 0;
      while (all < kd::GAMMA && std::size_t(emitted + all) < sampling.forced.size() &&
             r.drafts[all] == sampling.forced[emitted + all])
        ++all;
      stats.spec_trace.push_back({r.m, acc, all, 0.f, 0.f});
    }

    drafter_.advanced(commit);
    tokens_in_state.push_back(anchor);
    tokens_in_state.insert(tokens_in_state.end(), r.drafts, r.drafts + acc);
    P += commit;
    ++rounds;
    accepted += acc;
    out.assign(r.drafts, r.drafts + acc);
    out.push_back(r.next);
  }
  if (forced) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (std::size_t i = 0; i < events.size(); ++i) {
      if (i < stats.spec_trace.size()) {
        CUDA_CHECK(cudaEventElapsedTime(&stats.spec_trace[i].verify_ms, events[i].e[0], events[i].e[1]));
        CUDA_CHECK(cudaEventElapsedTime(&stats.spec_trace[i].draft_ms, events[i].e[1], events[i].e[2]));
      }
      for (cudaEvent_t e : events[i].e) cudaEventDestroy(e);
    }
  }
  stats.spec_rounds = rounds;
  stats.spec_accepted = accepted;
}

}  // namespace ingine::model

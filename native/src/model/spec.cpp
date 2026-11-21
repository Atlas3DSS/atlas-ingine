#include "model/spec.h"

#include "core/device.h"
#include "kernels/draft/draft.h"

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

// pinned host layout of Drafter::AcceptScratch::host
struct HostSync {
  kd::AcceptResult result;
  std::int32_t tokens[kd::GAMMA];
  float conf[kd::GAMMA];
};

}  // namespace

int verify_width(std::span<const int> widths, int lead) {
  for (int w : widths)
    if (w >= lead + 1) return w;
  throw std::logic_error("verify_width: no verify width holds the anchor and the confident drafts");
}

int confident_lead(std::span<const float> conf, float threshold) {
  int lead = 0;
  while (lead < int(conf.size()) && conf[lead] > threshold) ++lead;
  return lead;
}

void generate_spec(Runtime& rt, Drafter& drafter, std::int32_t token, int max_new_tokens, const Sampling& sampling,
                   kh::SamplerRng& rng, std::span<const std::int32_t> stop_ids,
                   const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats,
                   std::vector<std::int32_t>& tokens_in_state) {
  static_assert(sizeof(HostSync) <= 256, "pinned scratch too small");
  const cudaStream_t stream = rt.stream();
  if (drafter.stream() != stream) throw std::invalid_argument("generate_spec: drafter and runtime streams differ");
  const std::span<const int> widths = rt.verify_widths();
  if (widths.empty()) throw std::invalid_argument("generate_spec: the runtime has no verify widths (spec off)");
  int P = int(tokens_in_state.size());  // position of the anchor (sampled, not yet processed)
  if (drafter.length() != P) throw std::logic_error("generate_spec: drafter context does not match the state");

  const bool greedy = sampling.temperature <= 0.f;
  const float threshold = rt.options().spec_threshold;
  const int V = drafter.vocab();
  const Drafter::AcceptScratch& scratch = drafter.accept_scratch();
  auto* host = static_cast<HostSync*>(scratch.host);
  const kh::SamplingParams params{sampling.temperature, sampling.top_k, sampling.top_p};
  if (!greedy) drafter.set_seed(splitmix64(rng.seed ^ splitmix64(rng.counter++)));

  int emitted = 0, rounds = 0, accepted = 0;
  std::vector<std::int32_t> out{token};
  std::vector<std::int32_t> verify;
  verify.reserve(kd::BLOCK);
  for (;;) {
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
    if (done || !go_on) break;

    // draft; verify only the leading drafts the drafter is confident about (a wider call reads more experts)
    const std::int32_t anchor = out.back();
    const Drafter::Draft d = drafter.draft(anchor, P, sampling.temperature);
    CUDA_CHECK(cudaMemcpyAsync(host->tokens, d.tokens, sizeof(host->tokens), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host->conf, d.conf, sizeof(host->conf), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // one sync
    std::int32_t drafted[kd::GAMMA];
    std::memcpy(drafted, host->tokens, sizeof(drafted));
    const int lead = confident_lead(std::span<const float>(host->conf, kd::GAMMA), threshold);
    const int m = verify_width(widths, lead);
    const int n = m - 1;
    verify.assign(1, anchor);
    verify.insert(verify.end(), drafted, drafted + n);
    rt.set_inputs(verify, P);
    rt.save_verify(P, m);
    const float* logits = rt.run_step(m);

    // engine._accept
    if (greedy) {
      kh::argmax(logits, m, V, V, scratch.target, stream);
      kd::greedy_accept(scratch.target, d.tokens, n, scratch.result, stream);
    } else {
      kh::sampling_probs(logits, m, V, V, params, scratch.probs, stream);
      kh::uniform(n, rng, scratch.uniforms, stream);
      kd::sampled_accept(scratch.probs, d.logits, V, d.tokens, n, sampling.temperature, scratch.uniforms, scratch.ws,
                         scratch.resid, scratch.result, stream);
      kh::sample_from_probs(scratch.resid, 1, V, V, rng, &scratch.result->next, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(&host->result, scratch.result, sizeof(kd::AcceptResult), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));  // one sync
    const int acc = host->result.accepted;
    const std::int32_t nxt = host->result.next;
    const int commit = acc + 1;

    rt.rollback(P, commit, m);
    drafter.commit(ptr<const __nv_bfloat16>(rt.buffers(m).taps), commit, P);
    tokens_in_state.push_back(anchor);
    tokens_in_state.insert(tokens_in_state.end(), drafted, drafted + acc);
    P += commit;
    ++rounds;
    accepted += acc;
    out.assign(drafted, drafted + acc);
    out.push_back(nxt);
  }
  stats.spec_rounds = rounds;
  stats.spec_accepted = accepted;
}

}  // namespace ingine::model

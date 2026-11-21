// DSpark speculative decoding loop: C++ port of engine.Engine._generate_spec / _accept / _rollback.
//
// Per round: the drafter proposes 7 tokens after the anchor (the last sampled, not yet processed token at
// position P); only the leading drafts whose confidence (max softmax probability) exceeds
// options.spec_threshold are verified, in the smallest verify width of {2, 4, 8} holding anchor + them; the
// verify call runs the target's decode graph of that width (history + taps); acceptance is the longest
// prefix equal to the target argmax (greedy) or speculative rejection sampling against the drafts' own
// distributions (sampled, exact for the target's distribution); rejected rows are rolled back (conv states
// from the call's history, ring slots from the backup), the accepted rows' taps become draft context, and
// the accepted drafts plus the correction/bonus token are emitted. One host synchronisation after the draft
// and one after the acceptance.
#pragma once

#include "kernels/head/head.h"
#include "model/drafter.h"
#include "model/runtime.h"

#include "ingine/generator.h"

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ingine::model {

// Generates after a prefill: `token` is the first sampled token (not yet processed), at position
// P = tokens_in_state.size(); the drafter must hold every position before P (drafter.length() == P).
// Emits tokens through on_tokens (several per round; stops after a token in stop_ids, after
// max_new_tokens tokens, or when on_tokens returns false). Appends every processed token to
// tokens_in_state and sets stats.generated_tokens, spec_rounds and spec_accepted. Sampled acceptance draws
// from `rng` (and seeds the drafter's Gumbel noise from it). The caller reserves BLOCK positions of
// max_ctx for the verify calls (engine.generate: prompt + max_new_tokens + 8 <= max_ctx).
void generate_spec(Runtime& rt, Drafter& drafter, std::int32_t token, int max_new_tokens, const Sampling& sampling,
                   kernels::head::SamplerRng& rng, std::span<const std::int32_t> stop_ids,
                   const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats,
                   std::vector<std::int32_t>& tokens_in_state);

// The verify width for `lead` confident drafts: the smallest of `widths` (ascending) >= lead + 1.
int verify_width(std::span<const int> widths, int lead);
// engine._generate_spec's lead: number of leading rows with conf > threshold.
int confident_lead(std::span<const float> conf, float threshold);

}  // namespace ingine::model

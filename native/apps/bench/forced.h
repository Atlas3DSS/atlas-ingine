// ingine-bench --forced: teacher-forced replay of fixed outputs (see forced.cpp).
#pragma once

#include "ingine/generator.h"

#include <string>

namespace ingine::bench {

// Replays every item of `path` (JSON lines {"name", "prompt": [ids], "output": [ids]}) through `gen` with
// Sampling::forced = the item's output and prints per item and in total: tokens/s, tokens/verify, the width
// histogram, draft and verify ms per round; with a speculative generator also per output kind (reasoning,
// text, tool call), the share of rounds whose first draft is rejected and the oracle-width estimate.
// Items [skip, skip + items) are replayed (items <= 0: all after skip). With `trace_path`, every item and round
// is appended there as JSON lines ({"item", "tokens", "seconds", "rounds": [[width, accepted, accepted_all,
// verify_ms, draft_ms, kind], ...]}) for aggregation across runs. Returns the summary as a JSON object.
std::string run_forced(model::Generator& gen, const std::string& path, double temperature, int skip, int items,
                       const std::string& trace_path);

}  // namespace ingine::bench

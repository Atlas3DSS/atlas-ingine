// Continuous-batching runtime and generator for Inkling-2-Mini (B = 1..64+ concurrent sequences).
// Reuses Weights, Ablation, Prefiller, cuBLASLt Gemm, and grouped-expert wgemm.cu alongside the per-slot
// batched attention / static-conv / paged full-KV kernels in kernels/batch/.
#pragma once

#include "ingine/generator.h"
#include "model/model_data.h"

#include <memory>

namespace ingine::model {

// Constructs a continuous-batching Generator for options.max_batch >= 1 (called by make_generator when
// options.max_batch > 1; options.max_batch == 1 keeps EngineGenerator untouched for zero B=1 regression).
std::unique_ptr<Generator> make_batch_generator(const RuntimeOptions& options);

// Same over pre-loaded weights and ablation (for unit/parity tests without reloading weights).
std::unique_ptr<Generator> make_batch_generator(const RuntimeOptions& options, Weights weights, Ablation ablation);

}  // namespace ingine::model

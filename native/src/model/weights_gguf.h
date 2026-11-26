// Weights from a llama.cpp GGUF file of Inkling-2-Mini (see weights_gguf.cpp): `--model-dir` may hold a
// `model.gguf` (plus tokenizer.json / tokenizer_config.json / chat_template.jinja / config.json) instead of the
// Hugging Face safetensors checkpoint; load_weights and load_weights_sharded then read it.
#pragma once

#include "model/model_data.h"

#include <cuda_runtime.h>

#include <array>
#include <filesystem>

namespace ingine::model {

// model_dir/model.gguf, or model_dir itself when it is a .gguf file; empty when there is none.
std::filesystem::path gguf_model_file(const std::filesystem::path& model_dir);
bool is_gguf_model(const std::filesystem::path& model_dir);
// Expert formats per layer of the w13 (gate/up) or w2 (down) projections the GGUF file will load as.
std::array<ExpertFormat, LAYERS> gguf_expert_formats(const std::filesystem::path& model_dir, bool w2);
// The model for one tensor/expert-parallel rank (tp_size 1: everything), dense in the activation type.
Weights load_weights_gguf(const std::filesystem::path& model_dir, cudaStream_t stream, int tp_rank, int tp_size,
                          bool keep_full_unembed = false);

}  // namespace ingine::model

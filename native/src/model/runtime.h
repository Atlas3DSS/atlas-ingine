// The model runtime shared by the Generator, the Prefiller and the speculative loop: one CUDA stream, the
// weights and ablation, the request state, and the fused decode step (fast.FastDecoder.decode) per call
// width M in {1, 2, 4, 8} replayed as CUDA graphs (engine.Engine.steps / graphs), with the state
// accessors engine.Engine._snapshot/_restore/_rollback need.
//
// Decode inputs live in device buffers the graphs read: buffers(m).token (int64 [m]) and pos() (int64 [1],
// the first row's position; row i is at pos + i). Outputs: buffers(m).logits (fp32 [m, UNPADDED_VOCAB],
// softcapped), buffers(m).taps (bf16 [m, 4, H]) and, for widths captured with history, buffers(m).hist_*.
// Width 1 runs without history; widths 2, 4 and 8 (created when spec is on) with history (engine.py).
#pragma once

#include "model/model_data.h"

#include "ingine/generator.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <vector>

namespace ingine::model {

// ---- loading (weights.load + FastDecoder._concat_projections/_stack_layer_params/_fp8_dense,
// model.Ablation.load/none, model.State)
Weights load_weights(const std::filesystem::path& model_dir, bool dense_fp8, cudaStream_t stream);
// One layer as weights.load_layer (+ the decoder's derived tensors), for kernel tests; owns its memory.
Layer load_layer(const std::filesystem::path& model_dir, int l, bool dense_fp8, cudaStream_t stream,
                 std::vector<DeviceBuffer>& storage);
Ablation load_ablation(const std::filesystem::path& path);  // empty path: Ablation.none()
State make_state(int max_ctx, bool kv_fp8);
Buffers make_buffers(int m);

// model.State.reset: ring_pos = -1, conv states = 0, length = 0 (caches keep their contents).
void reset_state(State& state, cudaStream_t stream);

// ---- fast.FastDecoder
class Decoder {
 public:
  Decoder(const Weights& w, const Ablation& abl, State& st, bool pdl = true);
  ~Decoder();
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  // fast.FastDecoder.decode: runs b.m consecutive tokens b.token at positions *pos, *pos + 1, ...;
  // logits in b.logits, taps in b.taps, conv inputs in b.hist_* when `history`. Graph capturable once it
  // ran eagerly for `b` (pointer tables are built on first use).
  void decode(Buffers& b, const std::int64_t* pos, bool history, cudaStream_t stream);

 private:
  const std::int64_t* block_table(const std::vector<const float*>& blocks);

  const Weights& w_;
  const Ablation& abl_;
  State& st_;
  bool pdl_;
  DeviceBuffer attn_ws_;  // attn workspace for up to 8 rows (zeroed once)
  std::map<std::vector<const float*>, DeviceBuffer> tables_;  // device pointer tables for mix
  std::array<DeviceBuffer, 2> segments_;  // proj segment tables: [single-input sliding, three-input]
  DeviceBuffer full_segments_;
};

// ---- Runtime
class Runtime {
 public:
  // Loads the weights (options.model_dir, dense_fp8) and ablation (options.ablation).
  explicit Runtime(const RuntimeOptions& options);
  // Uses the given weights and ablation (tests, tools); options.model_dir and ablation are not read.
  Runtime(const RuntimeOptions& options, Weights weights, Ablation ablation);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

  const RuntimeOptions& options() const { return options_; }
  cudaStream_t stream() const { return stream_; }
  const Weights& weights() const { return weights_; }
  const Ablation& ablation() const { return ablation_; }
  State& state() { return state_; }
  const State& state() const { return state_; }

  // Call widths available: {1}, or {1, 2, 4, 8} with options.spec.
  bool has_width(int m) const;
  std::span<const int> verify_widths() const;  // {2, 4, 8} with spec, else empty
  Buffers& buffers(int m);
  std::int64_t* pos() { return static_cast<std::int64_t*>(pos_.p); }

  // Stream-ordered upload of the step inputs: buffers(tokens.size()).token = tokens, pos = position.
  void set_inputs(std::span<const std::int32_t> tokens, int position);
  // Replays width m's graph; returns buffers(m).logits (device). Stream-ordered, no host sync.
  const float* run_step(int m);
  // set_inputs + run_step for one token (engine._decode_logits).
  const float* decode_logits(std::int32_t token, int position);
  // The same step without graphs (validation, tests).
  const float* run_step_eager(int m);

  // engine._snapshot/_restore over ring_pos + conv states (device copies, stream-ordered).
  struct Snapshot {
    std::vector<DeviceBuffer> tensors;
  };
  Snapshot snapshot();
  void restore(const Snapshot& snap);

  // Speculative rollback (engine._generate_spec / _rollback). save_verify before a verify call of m rows at
  // P..P+m-1 copies the conv states and the ring slots (P + i) % WINDOW the call will overwrite;
  // rollback(P, commit, m) after it leaves the state as after its first `commit` rows (no-op if
  // commit == m). Uses buffers(m).hist_* (the call must run with history).
  void save_verify(int P, int m);
  void rollback(int P, int commit, int m);

 private:
  void init();
  void capture(int m, bool history);

  RuntimeOptions options_;
  cudaStream_t stream_ = nullptr;
  Weights weights_;
  Ablation ablation_;
  State state_;
  std::unique_ptr<Decoder> decoder_;
  DeviceBuffer pos_;
  std::map<int, Buffers> buffers_;
  std::map<int, bool> history_;
  std::map<int, cudaGraphExec_t> graphs_;
  std::vector<int> verify_widths_;
  // rollback scratch: old conv states [k, v, attn, mlp] and ring backups [40, 8, ...]
  std::array<DeviceBuffer, 4> old_conv_;
  DeviceBuffer ring_k_backup_, ring_v_backup_, ring_pos_backup_;
};

}  // namespace ingine::model

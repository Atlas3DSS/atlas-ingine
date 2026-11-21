// DSpark drafter for Inkling-2-Mini (mtp.safetensors): C++ port of engine/ingine/dspark.py Drafter.
//
// * Target taps per processed row (bf16 [n, 4, H]: the partial block entering visits 5, 23, 41 and the final
//   normed hidden) become draft context through commit(): per draft layer l, c = mix_rmsnorm(feats,
//   alpha[l]), k = rope(rmsnorm_128(c Wk_ctx)), v = c Wv_ctx. Layers 0 and 2 keep the last WINDOW = 1024
//   positions in a ring (plus 8 rows for the block being drafted); layer 1 a ring of ctx = min(max_ctx,
//   DRAFT_CTX) slots (slot = position % ctx).
// * draft() runs the 8-row block (anchor embedding + 7 mask rows) through 3 bidirectional layers with
//   FP8-row q/k/v and gate/up projections (the dense GEMV kernel) and BF16 o/down projections (cuBLASLt),
//   then the Markov head over each step's top-1024 base-logit candidates (base = FP8/BF16 head GEMV).
// * CUDA graphs: one draft graph per context bucket (1K, 2K, ... < ctx, then ctx; the global layer attends
//   to the first `bucket` slots) and one commit graph per row count 1..8.
#pragma once

#include "kernels/dense/dense.h"
#include "kernels/draft/draft.h"
#include "kernels/draft/gemm.h"
#include "model/model_data.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <vector>

namespace ingine::model {

class Drafter {
 public:
  static constexpr int BLOCK = kernels::draft::BLOCK;  // 8 rows per drafted block
  static constexpr int GAMMA = kernels::draft::GAMMA;  // 7 drafts per round
  static constexpr int DRAFT_CTX = 65536;              // global-layer ring slots (at most)

  // Loads mtp.safetensors from model_dir; `w` supplies the embedding, its norm and the unembedding (the
  // base logits use w.unembed as stored: FP8 rows with dense_fp8). `w` must outlive the drafter.
  Drafter(const std::filesystem::path& model_dir, const Weights& w, int max_ctx, cudaStream_t stream);
  ~Drafter();
  Drafter(const Drafter&) = delete;
  Drafter& operator=(const Drafter&) = delete;

  // Captures the draft graphs (per bucket) and commit graphs (1..8 rows), then reset()s.
  void capture();
  // Empty context (local rings invalidated, length 0). Stream-ordered.
  void reset();
  // Positions committed so far (start + n of the last commit).
  int length() const { return length_; }
  int ctx() const { return ctx_; }

  // Committed target rows (feats bf16 [n, 4, H] on the device, positions start..start+n-1) become draft
  // context. n of 1..8 replays a commit graph; larger n (prefill chunks) runs eagerly in pieces of <= 1024.
  void commit(const Tensor& feats, int start);
  void commit(const __nv_bfloat16* feats, int n, int start);

  // Drafts for positions P+1..P+7 after the anchor token at P. Greedy at temperature <= 0, otherwise
  // Gumbel-max at the temperature. Outputs are fixed device buffers, overwritten by the next draft.
  struct Draft {
    const std::int32_t* tokens;  // int32 [7]
    const float* logits;         // fp32 [7, V]: softcapped corrected logits, -inf outside the candidates
    const float* conf;           // fp32 [7]: max softmax probability per row (softmax(logits).amax(-1))
  };
  Draft draft(int anchor, int P, float temperature);

  // Seed of the Gumbel noise of sampled drafts (resets the draw counter). Stream-ordered.
  void set_seed(std::uint64_t seed);

  int vocab() const { return V_; }

  // Device scratch of the acceptance step (engine._accept), owned here so it is allocated once.
  struct AcceptScratch {
    std::int32_t* target;                    // int32 [8]: argmax of the verify rows
    float* probs;                            // fp32 [8, V]: engine._probs of the verify rows
    float* resid;                            // fp32 [V]
    float* uniforms;                         // fp32 [7]
    float* ws;                               // fp32 [2 * GAMMA + 1]
    kernels::draft::AcceptResult* result;    // device
    void* host;                              // pinned: AcceptResult, then tokens [7] + conf [7]
  };
  const AcceptScratch& accept_scratch() const { return accept_; }
  cudaStream_t stream() const { return stream_; }

 private:
  struct DraftLayer {
    const __nv_bfloat16 *in_norm, *post_norm, *ctx_norm, *q_norm, *k_norm, *k_norm_ctx;
    kernels::dense::Matrix qkv, gate_up;  // FP8 rows + scales
    const __nv_bfloat16* wo;              // [H, 3072]
    const __nv_bfloat16* w_down;          // [H, inter]
    const __nv_bfloat16* wkv_ctx;         // [2048, H]: k_proj_ctx | v_proj_ctx
    __nv_bfloat16 *k_ctx, *v_ctx;         // [slots, 8, 128]
    std::int32_t* ctx_pos;                // local layers: int32 [WINDOW]; null for the global layer
  };

  void* alloc(std::size_t bytes);
  void load(const std::filesystem::path& model_dir);
  void run_draft(int bucket);
  void run_commit(const __nv_bfloat16* feats, int n, int row0, bool last_piece);
  int bucket_for(int P) const;

  const Weights& w_;
  cudaStream_t stream_;
  int ctx_ = 0;
  int V_ = 0;
  int inter_ = 0;
  int length_ = 0;
  std::vector<DeviceBuffer> storage_;
  kernels::draft::Bf16Gemm gemm_;

  std::array<DraftLayer, 3> layers_{};
  const __nv_bfloat16* alpha_ = nullptr;  // bf16 [4, 4] softmax(fuse_weight)
  const __nv_bfloat16 *final_norm_ = nullptr, *mask_embed_ = nullptr, *f_norm_ = nullptr;
  const __nv_bfloat16 *w1_ = nullptr, *rnn_w_ = nullptr, *rnn_b_ = nullptr;
  __nv_bfloat16 *fold_w_ = nullptr, *fold_b_ = nullptr;
  kernels::dense::Matrix head_{};

  // per-call state and scratch
  kernels::draft::DraftParams* params_ = nullptr;
  __nv_bfloat16 *res_ = nullptr, *xn_ = nullptr, *q_ = nullptr, *attn_o_ = nullptr, *delta_ = nullptr,
                *act_ = nullptr, *f_rnn_ = nullptr, *state_ = nullptr, *latent_ = nullptr;
  float *qkv_out_ = nullptr, *gu_out_ = nullptr, *base_ = nullptr, *attn_ws_ = nullptr;
  float *cand_val_ = nullptr, *cand_logits_ = nullptr, *part_ = nullptr;
  std::int32_t *cand_idx_ = nullptr, *prev_cand_ = nullptr;
  unsigned long long* picks_ = nullptr;
  float *draft_logits_ = nullptr, *conf_ = nullptr;
  std::int32_t* tokens_ = nullptr;
  int *seg_qkv_ = nullptr, *seg_gate_up_ = nullptr, *seg_head_ = nullptr;
  // commit scratch
  __nv_bfloat16 *commit_feats_ = nullptr, *ctx_c_ = nullptr, *ctx_kv_ = nullptr;

  AcceptScratch accept_{};
  PinnedHostBuffer pinned_{256};

  std::map<int, cudaGraphExec_t> draft_graphs_;   // bucket -> graph, ascending
  std::map<int, cudaGraphExec_t> commit_graphs_;  // rows -> graph
};

}  // namespace ingine::model

// Multi-slot batched DSpark drafter (B = 1..64+ concurrent slots) sharing immutable mtp.safetensors weights
// and immutable ranked shortlist across all slots, while keeping KV rings, f_rnn, dynamic shortlist extra ids,
// RNG state, and candidate proposals strictly slot-local.
#pragma once

#include "core/act.h"
#include "core/arena.h"
#include "core/host_memory.h"
#include "kernels/dense/dense.h"
#include "kernels/draft/batch_draft.h"
#include "kernels/draft/draft.h"
#include "model/model_data.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <unordered_set>
#include <vector>

namespace ingine::kernels::prefill {
class Gemm;
}

namespace ingine::model {

// Compact GPU snapshot of one slot's drafter context at prefix length `length` (stored inside PrefixCheckpointGpu).
struct DraftCheckpointGpu {
  int length = 0;
  int local_rows = 0;   // min(length, WINDOW)
  int global_rows = 0;  // min(length, draft_ctx)
  std::array<DeviceBuffer, 2> local_k, local_v, local_pos;  // layers 0 and 2
  DeviceBuffer global_k, global_v;                          // layer 1 (E4M3)
  DeviceBuffer f_rnn;                                       // ActT [H]
  std::vector<std::int32_t> extra_ids;                      // slot's exact dynamic shortlist tokens up to `length`
};

class BatchDrafter {
 public:
  static constexpr int BLOCK = kernels::draft::BLOCK;        // 8
  static constexpr int GAMMA = kernels::draft::GAMMA;        // 7
  static constexpr int SHORTLIST_ROWS = 32768;
  static constexpr int MAX_SLOT_EXTRA = kernels::draft::kMaxSlotExtra;  // 2048

  struct DraftRequest {
    int slot = -1;
    std::int32_t anchor = 0;
    float temperature = 0.f;
  };

  struct BatchDraftOutput {
    const std::int32_t* tokens = nullptr;   // device int32 [B, 7]
    const float* conf = nullptr;            // device fp32 [B, 7]
    const float* qstats = nullptr;          // device fp32 [B, 7, 2]
    const std::int32_t* cand_idx = nullptr; // device int32 [B, 7, 1024]
    const float* cand_logits = nullptr;     // device fp32 [B, 7, 1024]
  };

  struct AcceptScratch {
    std::int32_t* target_argmax = nullptr;  // int32 [max_B * max_W]
    std::int32_t* accepted = nullptr;       // int32 [max_B]
    std::int32_t* next = nullptr;           // int32 [max_B]
    std::int32_t* commit = nullptr;         // int32 [max_B]
    // Reusable single-slot buffers for exact sampled rejection acceptance (saves ~770 MiB at B=64!):
    float* probs = nullptr;                 // fp32 [BLOCK, V]
    float* draft_logits = nullptr;          // fp32 [GAMMA, V] (maintained all -INFINITY between calls)
    float* resid = nullptr;                 // fp32 [V]
    float* uniforms = nullptr;              // fp32 [GAMMA]
    float* ws = nullptr;                    // fp32 [sampled_accept_workspace_floats(V)]
    kernels::draft::AcceptResult* one_res = nullptr;
  };

  BatchDrafter(const std::filesystem::path& model_dir, const Weights& w, int max_batch, int max_ctx,
               int max_verify_width, cudaStream_t stream, int shortlist_rows = SHORTLIST_ROWS,
               std::size_t vram_budget_bytes = 0);
  ~BatchDrafter();
  BatchDrafter(const BatchDrafter&) = delete;
  BatchDrafter& operator=(const BatchDrafter&) = delete;

  int max_batch() const { return max_batch_; }
  int draft_ctx() const { return ctx_; }
  int vocab() const { return V_; }
  std::size_t device_bytes() const;

  // Per-slot lifecycle & state isolation
  void reset_slot(int slot);
  void set_slot_seed(int slot, std::uint64_t seed);
  int slot_length(int slot) const { return slot_length_.at(std::size_t(slot)); }
  std::uint64_t slot_seed(int slot) const { return slot_seed_.at(std::size_t(slot)); }
  std::uint64_t slot_draws(int slot) const { return slot_draws_.at(std::size_t(slot)); }
  const std::vector<std::int32_t>& slot_extra_ids(int slot) const { return slot_extra_ids_host_.at(std::size_t(slot)); }

  // Appends unseen rare tokens (outside immutable ranked shortlist) to THIS slot's private extra_ids list only.
  void note_tokens_slot(int slot, std::span<const std::int32_t> ids);

  // Commits prefill taps [n, 4, H] for a single slot at positions [start, start + n).
  void commit_slot_prefill(int slot, const ActT* taps, int n, int start);

  // Commits accepted verify taps [B * W, 4, H] across B active slots in one batched pass.
  void commit_batch(std::span<const kernels::draft::BatchSlotCommitMeta> commits, const ActT* taps, int W);

  // Runs one batched 3-layer draft + Markov head for B active slots simultaneously.
  BatchDraftOutput draft_batch(std::span<const DraftRequest> reqs);

  // Exact compact prefix checkpointing (unconditional full copy across slots; safe for cross-slot restores).
  std::size_t estimate_checkpoint_bytes(int length) const;
  std::size_t save_slot_checkpoint(int slot, int length, DraftCheckpointGpu& out);
  std::size_t restore_slot_checkpoint(int slot, const DraftCheckpointGpu& ckpt);

  const AcceptScratch& accept_scratch() const { return accept_; }

 private:
  struct DraftLayer {
    const ActT *in_norm, *post_norm, *ctx_norm, *q_norm, *k_norm, *k_norm_ctx;
    kernels::dense::Matrix qkv, gate_up;
    kernels::dense::Matrix wo, w_down;
    const ActT* wkv_ctx;                  // BF16 [2048, H]: k_proj_ctx | v_proj_ctx
    void *k_ctx, *v_ctx;                  // [max_batch, slots, 8, 128]
    std::size_t kv_slot_bytes;
    std::int32_t* ctx_pos;                // local layers: [max_batch, WINDOW]; nullptr for global
  };

  void* alloc(std::size_t bytes);
  void load(const std::filesystem::path& model_dir);
  const ActT* dequant_to_scratch(const kernels::dense::Matrix& m, int K);

  const Weights& w_;
  cudaStream_t stream_;
  int max_batch_ = 1;
  int max_verify_width_ = 8;
  int ctx_ = 0;
  int V_ = 0;
  int inter_ = 0;

  std::vector<DeviceBuffer> storage_;
  std::unique_ptr<kernels::prefill::Gemm> gemm_;

  std::array<DraftLayer, 3> layers_{};
  const ActT* alpha_ = nullptr;
  const ActT *final_norm_ = nullptr, *mask_embed_ = nullptr, *f_norm_ = nullptr;
  const ActT *w1_ = nullptr, *rnn_w_ = nullptr, *rnn_b_ = nullptr;
  ActT *fold_w_ = nullptr, *fold_b_ = nullptr;
  kernels::dense::Matrix head_{};
  kernels::dense::Matrix short_head_{};   // IMMUTABLE ranked + special rows [short_K_, H]
  std::int32_t* short_map_ = nullptr;     // [short_K_]
  int short_K_ = 0;
  int base_stride_ = 0;                   // short_K_ + MAX_SLOT_EXTRA
  std::vector<std::uint8_t> short_base_in_;

  kernels::dense::Matrix wkv_all_{};
  kernels::draft::BatchCtxLayers ctx_layers_{};
  int *seg_qkv_ = nullptr, *seg_gate_up_ = nullptr, *seg_short_ = nullptr, *seg_kv_ = nullptr;

  // Per-slot host tracking & device pools
  std::vector<int> slot_length_;
  std::vector<std::uint64_t> slot_seed_;
  std::vector<std::uint64_t> slot_draws_;
  std::vector<std::vector<std::int32_t>> slot_extra_ids_host_;
  std::vector<std::unordered_set<std::int32_t>> slot_extra_set_;
  std::int32_t* slot_extra_ids_dev_ = nullptr;  // [max_batch, MAX_SLOT_EXTRA]
  ActT* f_rnn_pool_ = nullptr;                  // [max_batch, H]

  // Batched draft & commit scratch
  PinnedHostBuffer draft_meta_host_, commit_meta_host_;
  kernels::draft::BatchSlotDraftMeta* draft_meta_dev_ = nullptr;
  kernels::draft::BatchSlotCommitMeta* commit_meta_dev_ = nullptr;
  kernels::draft::DraftParams* prefill_params_ = nullptr;

  ActT *res_ = nullptr, *xn_ = nullptr, *q_ = nullptr, *attn_o_ = nullptr, *act_ = nullptr, *h_gamma_ = nullptr;
  ActT *qkv_act_ = nullptr, *delta_act_ = nullptr, *gu_act_ = nullptr, *base_act_ = nullptr, *wscratch_ = nullptr;
  float *qkv_f32_ = nullptr, *delta_f32_ = nullptr, *gu_f32_ = nullptr, *base_tmp_f32_ = nullptr, *base_ = nullptr;
  float *attn_ws_ = nullptr;
  void* topk_ws_ = nullptr;
  float *cand_val_ = nullptr, *cand_logits_ = nullptr, *part_ = nullptr;
  std::int32_t* cand_idx_ = nullptr;
  ActT *state_ = nullptr, *latent_ = nullptr;
  unsigned long long* picks_ = nullptr;
  std::int32_t* tokens_ = nullptr;
  float *conf_ = nullptr, *qstats_ = nullptr;

  // Commit scratch
  ActT *ctx_c_ = nullptr, *ctx_kv_ = nullptr, *commit_xin_ = nullptr, *commit_kv_act_ = nullptr;
  float* commit_kv_f32_ = nullptr;

  AcceptScratch accept_{};
};

}  // namespace ingine::model

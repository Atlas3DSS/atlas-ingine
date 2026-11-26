// Batched multi-slot DSpark drafter (up to kMaxBatchSlots = 4 concurrent slots).
//
// Shares all drafter weights and a single union shortlist head across slots while maintaining per-slot
// local (1024-token BF16) and global (draft_ctx-token E4M3) KV rings, DraftParams, and Markov head state.
// Every GEMV reads weights ONCE across all S active slots (M = 8*S for block layers, M = 7*S for shortlist
// head, M = max_commit*S for small commits). Prefill commits chunk through FP8 wkv_all_ (eliminating the
// 28 MiB BF16 wkv_ctx copy and 32 MiB cuBLASLt workspace). Greedy-only (omits [7, V] draft_logits).
//
// CUDA graphs are captured per (S, global_bucket) and read per-slot pointers from a device BatchSlotDesc*,
// so one captured graph handles any subset or ordering of S active physical slots.
#pragma once

#include "core/act.h"
#include "core/arena.h"
#include "kernels/dense/dense.h"
#include "kernels/draft/draft.h"
#include "model/model_data.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <utility>
#include <vector>

namespace ingine::model {

class BatchDrafter {
 public:
  static constexpr int MAX_SLOTS = kernels::draft::kMaxBatchSlots;  // 4
  static constexpr int BLOCK = kernels::draft::BLOCK;               // 8
  static constexpr int GAMMA = kernels::draft::GAMMA;               // 7
  static constexpr int DRAFT_CTX = 65536;                           // global ring cap
  static constexpr int DEFAULT_DRAFT_CTX = 16384;                   // default global ring per slot (32 MiB/slot)
  static constexpr int SHORTLIST_ROWS = 32768;
  static constexpr int CONTEXT_ROWS = 16384;

  // draft_ctx <= 0 uses min(max_ctx, DRAFT_CTX); otherwise min({max_ctx, DRAFT_CTX, draft_ctx}).
  // draft_len (1..GAMMA) controls how many draft tokens are scored through the shortlist head + Markov head
  // (e.g. 3 when verify width per slot <= 4: bit-identical first 3 drafts at ~40% lower head/Markov cost).
  BatchDrafter(const std::filesystem::path& model_dir, const Weights& w, int max_slots, int max_ctx, int draft_ctx,
               cudaStream_t stream, int shortlist_rows = SHORTLIST_ROWS,
               const kernels::draft::HeadShards& head_shards = {}, int max_commit = 4, int draft_len = 3);
  ~BatchDrafter();
  BatchDrafter(const BatchDrafter&) = delete;
  BatchDrafter& operator=(const BatchDrafter&) = delete;

  // Captures draft and commit+draft CUDA graphs for all S in 1..max_slots and global context buckets, then reset_all().
  void capture();

  // Resets one physical slot's local ring positions, f_rnn, and length (stream-ordered).
  void reset_slot(int slot);
  // Per-slot host snapshot/restore for TP2 prefix-cache reuse (zero VRAM; stream-ordered D2H / H2D).
  std::size_t slot_snapshot_bytes() const noexcept;
  std::size_t snapshot_slot_to_host(int slot, void* dst) const;
  std::size_t restore_slot_from_host(int slot, const void* src, int length);
  // Resets all slots and clears context tokens added to the shared shortlist (stream-ordered).
  void reset_all();
  // Clears context tokens added to the shared shortlist (call when all slots are idle).
  void reset_context_shortlist();

  // Adds unseen context tokens to the shared shortlist union (up to CONTEXT_ROWS). Stream-ordered.
  std::size_t note_tokens(std::span<const std::int32_t> ids);

  // Commits `n` target tap rows (feats [n, 4, H] on device) for `slot` at positions [start, start + n).
  void commit_slot(int slot, const ActT* feats, int n, int start);

  // Uploads BatchSlotDesc for `active_slots` (1..max_slots) into slots_dev_ (stream-ordered).
  // Call before batch_greedy_accept when fusing accept -> step_commit_and_draft.
  void upload_active_desc(std::span<const int> active_slots);

  // Standalone draft across `active_slots` (e.g. initial draft right after prefill, or testing).
  void draft_slots(std::span<const int> active_slots, std::span<const std::int32_t> anchors,
                   std::span<const int> positions, float temperature = 0.f);

  // Batched small commit (1..max_commit rows per slot, already written into each slot's DraftParams on device
  // by batch_greedy_accept) followed immediately by batched draft across `active_slots`.
  // `max_P` is an upper bound on P across the active slots (used to pick the global attention bucket).
  void step_commit_and_draft(std::span<const int> active_slots, int max_P, bool desc_already_uploaded = false);

  const kernels::draft::BatchSlotDesc* slots_dev() const { return slots_dev_; }
  kernels::draft::DraftParams* slot_params_dev(int slot) const { return slots_[slot].params; }
  const std::int32_t* slot_tokens_dev(int slot) const { return slots_[slot].tokens; }  // int32 [GAMMA]
  const float* slot_conf_dev(int slot) const { return slots_[slot].conf; }             // fp32 [GAMMA]

  int slot_length(int slot) const { return slots_[slot].length; }
  void set_slot_length(int slot, int len) { slots_[slot].length = len; }

  int max_slots() const { return max_slots_; }
  int max_commit() const { return max_commit_; }
  int draft_len() const { return draft_len_; }
  int ctx() const { return ctx_; }
  int vocab() const { return V_; }
  int shortlist_size() const { return short_K_ + short_E_; }
  cudaStream_t stream() const { return stream_; }

 private:
  struct DraftLayerWeights {
    const ActT *in_norm, *post_norm, *ctx_norm, *q_norm, *k_norm, *k_norm_ctx;
    kernels::dense::Matrix qkv, gate_up;
    kernels::dense::Matrix wo, w_down;
  };

  struct SlotState {
    kernels::draft::DraftParams* params = nullptr;
    kernels::draft::CtxLayers ctx{};
    ActT* f_rnn = nullptr;
    ActT* state = nullptr;
    ActT* latent = nullptr;
    unsigned long long* picks = nullptr;
    std::int32_t* tokens = nullptr;
    float* conf = nullptr;
    int length = 0;
  };

  void* alloc(std::size_t bytes);
  void load(const std::filesystem::path& model_dir);
  void load_safetensors(const std::filesystem::path& path);
  void load_gguf(const std::filesystem::path& path);
  void run_draft(int S, int bucket);
  void run_step(int S, int bucket);
  int bucket_for(int max_P) const;
  std::size_t set_context_rows(int rows);

  const Weights& w_;
  cudaStream_t stream_;
  int max_slots_ = 0;
  int max_commit_ = 4;
  int draft_len_ = 3;
  int ctx_ = 0;
  int V_ = 0;
  int inter_ = 0;
  std::vector<DeviceBuffer> storage_;

  std::array<DraftLayerWeights, 3> layers_{};
  const ActT* alpha_ = nullptr;
  const ActT *final_norm_ = nullptr, *mask_embed_ = nullptr, *f_norm_ = nullptr;
  const ActT *w1_ = nullptr, *rnn_w_ = nullptr, *rnn_b_ = nullptr;
  ActT *fold_w_ = nullptr, *fold_b_ = nullptr;
  kernels::draft::HeadShards head_shards_{};
  kernels::dense::Matrix head_{};
  kernels::dense::Matrix short_head_{};
  std::int64_t* short_blk_off_ = nullptr;
  std::int32_t* short_blk_soff_ = nullptr;
  std::uint16_t* short_q6_scales_ = nullptr;
  std::int32_t* short_map_ = nullptr;
  std::int32_t* short_ids_ = nullptr;
  int short_K_ = 0, short_E_ = 0;
  std::vector<std::uint8_t> short_in_;
  std::vector<std::int32_t> short_extra_;
  kernels::dense::Matrix wkv_all_{};

  std::array<SlotState, MAX_SLOTS> slots_{};
  kernels::draft::BatchSlotDesc* slots_dev_ = nullptr;

  // Shared scratch sized for max_slots_ (attention & topk workspaces sized for 1 slot and reused sequentially).
  ActT *res_ = nullptr, *xn_ = nullptr, *q_ = nullptr, *attn_o_ = nullptr, *act_ = nullptr, *h_ = nullptr;
  float *qkv_out_ = nullptr, *gu_out_ = nullptr, *delta_ = nullptr, *base_ = nullptr, *attn_ws_ = nullptr;
  float *cand_val_ = nullptr, *cand_logits_ = nullptr, *part_ = nullptr;
  void* topk_ws_ = nullptr;
  std::int32_t* cand_idx_ = nullptr;
  int *seg_qkv_ = nullptr, *seg_gate_up_ = nullptr, *seg_short_ = nullptr, *seg_kv_ = nullptr;
  ActT *warm_feats_ = nullptr, *commit_xin_ = nullptr;
  float* commit_kv_ = nullptr;

  std::vector<int> buckets_;
  std::map<std::pair<int, int>, cudaGraphExec_t> draft_graphs_;
  std::map<std::pair<int, int>, cudaGraphExec_t> step_graphs_;
};

}  // namespace ingine::model

// Batched multi-slot DSpark drafter kernels (B = 1..64+ concurrent slots, SM60..SM120a, BF16 & INGINE_ACT_FP16).
// Every kernel is stream-ordered, allocates nothing, never synchronizes, keeps all per-slot state (KV rings,
// f_rnn, dynamic shortlist extra ids, RNG seed/draws, proposals) strictly slot-local, and skips slot < 0.
#pragma once

#include "core/act.h"
#include "kernels/draft/draft.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::draft {

inline constexpr int kMaxSlotExtra = 2048;

struct BatchSlotDraftMeta {
  std::int32_t slot = -1;       // physical slot in [0, max_slots), < 0 = padded inactive row
  std::int32_t anchor = 0;      // anchor token at position P
  std::int32_t P = 0;           // position of anchor
  float temperature = 0.f;      // <= 0 greedy, > 0 Gumbel-max
  std::uint64_t seed = 0;       // per-slot Gumbel RNG seed
  std::uint64_t draws = 0;      // per-slot draw counter
  std::int32_t extra_count = 0; // slot-local dynamic shortlist tokens in [0, kMaxSlotExtra]
  std::int32_t pad = 0;
};

struct BatchSlotCommitMeta {
  std::int32_t slot = -1;     // physical slot in [0, max_slots), < 0 = skip
  std::int32_t start = 0;     // starting sequence position of committed rows
  std::int32_t ncommit = 0;   // rows to commit in [1, W]
  std::int32_t pad = 0;
};

struct BatchCtxLayers {
  const ActT* ctx_norm[3] = {};
  const ActT* k_norm_ctx[3] = {};
  void* k_cache[3] = {};            // slot s at static_cast<uint8_t*>(k_cache[l]) + s * kv_slot_bytes[l]
  void* v_cache[3] = {};
  std::size_t kv_slot_bytes[3] = {};
  bool fp8[3] = {};
  std::int32_t* ctx_pos[3] = {};    // local layers: slot s at ctx_pos[l] + s * WINDOW; nullptr for global
  int size[3] = {};                 // WINDOW (1024) for local, draft_ctx for global
};

// ---- Batched context commit (from [B * W, 4, H] target verify taps) ----------------------------------------

// Computes xin[l, b*W + r] = mix_rmsnorm(taps[b*W + r], alpha[l], ctx_norm[l]) for r < ncommit (zero above)
// and updates f_rnn_pool[slot] = mix_rmsnorm(taps[b*W + ncommit - 1], alpha[3], f_norm).
void batch_commit_mix(const BatchSlotCommitMeta* meta, int B, int W, const ActT* taps, const ActT* alpha,
                      const BatchCtxLayers& L, const ActT* f_norm, ActT* xin, ActT* f_rnn_pool, cudaStream_t stream);

// Stores layer `l` projected K/V rows (either fp32 with ld_kv >= col_off + 2048, or ActT with ld_kv = 2048)
// into each slot's own context cache at positions start + r (0 <= r < ncommit).
void batch_commit_store(const void* kv, bool kv_fp32, int ld_kv, int col_off, int l, const BatchSlotCommitMeta* meta,
                        int B, int W, const BatchCtxLayers& L, cudaStream_t stream);

// ---- Batched draft block -----------------------------------------------------------------------------------

void batch_embed_block(const ActT* embed, const float* embed_norm, const ActT* mask_embed, const ActT* in_norm,
                       const BatchSlotDraftMeta* meta, int B, ActT* res, ActT* xn, unsigned long long* picks,
                       cudaStream_t stream, bool embed_q4 = false);

void batch_qkv_rope(const void* qkv, bool qkv_fp32, const ActT* q_norm, const ActT* k_norm,
                    const BatchSlotDraftMeta* meta, int B, ActT* q_out, void* k_cache, void* v_cache,
                    std::size_t kv_slot_bytes, bool fp8, bool local, int ctx, cudaStream_t stream);

std::size_t batch_attention_workspace_floats(int max_b, int max_nkeys);
void batch_block_attention(const ActT* q, const void* k_cache, const void* v_cache, std::size_t kv_slot_bytes,
                           int nkeys, const std::int32_t* ctx_pos_base, bool fp8, const BatchSlotDraftMeta* meta,
                           int B, float* ws, ActT* out, cudaStream_t stream);

void batch_add_rmsnorm(ActT* res, const void* delta, bool delta_fp32, const ActT* w, ActT* xn, int rows,
                       cudaStream_t stream);

void batch_silu_mul(const void* gu, bool gu_fp32, int rows, int inter, ActT* out, cudaStream_t stream);

// Copies rows 1..7 of each slot's xn [B, 8, H] into contiguous h_gamma [B * 7, H].
void batch_extract_gamma_rows(const ActT* xn, int B, ActT* h_gamma, cudaStream_t stream);

// Copies a chunk [rows, chunk_V] of ActT GEMM output into fp32 dst [rows, dst_stride] at column offset v_offset.
void batch_cast_base_chunk(const ActT* src, int rows, int chunk_V, int v_offset, int dst_stride, float* dst,
                           cudaStream_t stream);

// Copies a chunk [rows, chunk_V] of fp32 GEMV output into fp32 dst [rows, dst_stride] at column offset 0.
void batch_copy_base_rows(const float* src, int rows, int src_stride, int dst_stride, float* dst, cudaStream_t stream);

// Scores each slot's own dynamic shortlist tokens (extra_ids[slot, 0..extra_count)) directly against resident
// unembedding weights (FP8 + scale or ActT), writing base[(b*7 + g) * base_stride + short_K + e].
// Zero cross-slot reads or writes.
void batch_score_slot_extra(const ActT* h_gamma, int B, const void* unembed_w, const float* unembed_scale,
                            const std::int32_t* slot_extra_ids, int max_extra, const BatchSlotDraftMeta* meta,
                            int short_K, int base_stride, float* base, cudaStream_t stream);

// ---- Batched Markov head & acceptance ----------------------------------------------------------------------

std::size_t batch_topk_workspace_bytes(int max_rows);
void batch_topk_candidates(const float* base, int B, int base_stride, int short_K, const std::int32_t* short_map,
                           const std::int32_t* slot_extra_ids, int max_extra, const BatchSlotDraftMeta* meta,
                           float* cand_val, std::int32_t* cand_idx, void* ws, cudaStream_t stream);

void batch_markov_pre(const ActT* rnn_w, const ActT* h_gamma, const ActT* f_rnn_pool, const BatchSlotDraftMeta* meta,
                      int B, float* part, cudaStream_t stream);

void batch_markov_step(int i, const ActT* rnn_w, const ActT* rnn_b, const ActT* w1, const float* part,
                       const unsigned long long* picks, const std::int32_t* cand_idx, const BatchSlotDraftMeta* meta,
                       int B, ActT* state, ActT* latent, cudaStream_t stream);

void batch_markov_logits(int i, const ActT* fold_w, const ActT* fold_b, const float* cand_val,
                         const std::int32_t* cand_idx, const ActT* latent, const BatchSlotDraftMeta* meta, int B,
                         float* cand_logits, unsigned long long* picks, cudaStream_t stream);

// Computes per-slot tokens [B, 7], conf [B, 7], qstats [B, 7, 2] directly from compact [B, 7, 1024] candidates
// (no giant [B, 7, V] dense logits buffer).
void batch_markov_finish(const std::int32_t* cand_idx, const float* cand_logits, const unsigned long long* picks,
                         const BatchSlotDraftMeta* meta, int B, std::int32_t* tokens, float* conf, float* qstats,
                         cudaStream_t stream);

// Scatters or clears ONE slot's [GAMMA, CANDIDATES] sparse candidate logits into a reusable [GAMMA, V] buffer
// (initialized to -INFINITY) for exact sampled_accept() without allocating [B, GAMMA, V] VRAM.
void scatter_slot_draft_logits(const std::int32_t* cand_idx, const float* cand_logits, int V, float* dst_logits,
                               cudaStream_t stream);
void clear_slot_draft_logits(const std::int32_t* cand_idx, int V, float* dst_logits, cudaStream_t stream);

// Batched greedy acceptance for B slots: compares drafts[b, 0..w_lim-2] against target_argmax[b*W + 0..w_lim-1].
// Writes accepted[b], next[b], commit[b] = accepted[b] + 1.
void batch_greedy_accept(const std::int32_t* target_argmax, const std::int32_t* drafts,
                         const std::int32_t* slot_w_limit, int B, int W, std::int32_t* accepted_out,
                         std::int32_t* next_out, std::int32_t* commit_out, cudaStream_t stream);

}  // namespace ingine::kernels::draft

// Batched multi-sequence decode kernels for continuous batching (B = 1..64+ independent requests).
// Every kernel is stream-ordered, allocates nothing, never synchronizes, skips padded rows (slot_ids[b] < 0),
// and is CUDA-graph capturable across SM60..SM120a (BF16 and INGINE_ACT_FP16).
#pragma once

#include "core/act.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::batch {

inline constexpr int kPageSize = 64;
inline constexpr int kPageShift = 6;
inline constexpr int kPageMask = kPageSize - 1;

struct BatchedAttnArgs {
  const void* proj = nullptr;  // [B, R] fp32 (proj_fp32=true) or ActT (proj_fp32=false), SEG order
  bool proj_fp32 = false;
  int B = 1;
  const std::int32_t* slot_ids = nullptr;  // device int32 [B], < 0 = padded inactive row
  const std::int64_t* slot_pos = nullptr;  // device int64 [max_slots], current token position per slot

  const float* q_scale = nullptr;   // fp32 [24, 128]
  const float* r_scale = nullptr;   // fp32 [24, 16]
  const float* rel_proj = nullptr;  // fp32 [16, 1024]
  const float* k_conv = nullptr;    // sliding: fp32 [1024, 4]; full: k_static [4, 1024]
  const float* v_conv = nullptr;

  // Per-slot conv state pointers offset to the current visit `v`; slot `s` is at `ptr + s * conv_slot_stride`.
  float* conv_k_visit = nullptr;
  float* conv_v_visit = nullptr;
  std::size_t conv_slot_stride = 0;  // floats per slot (NUM_VISITS * 3 * KV_DIM)

  // Sliding layer (window == 1024): per-slot ring pointers offset to sliding visit `ri`.
  ActT* k_ring_visit = nullptr;
  ActT* v_ring_visit = nullptr;
  std::int64_t* ring_pos_visit = nullptr;
  std::size_t ring_slot_stride = 0;      // ActT elements per slot (NUM_SLIDING_VISITS * 1024 * 1024)
  std::size_t ring_pos_slot_stride = 0;  // int64 elements per slot (NUM_SLIDING_VISITS * 1024)

  // Full layer (window == 0): paged KV pool [num_pages, NUM_FULL_VISITS, kPageSize, 8, 128].
  void* k_full_pool = nullptr;
  void* v_full_pool = nullptr;
  const std::int32_t* page_table = nullptr;  // device int32 [max_slots, max_pages_per_slot]
  int max_pages_per_slot = 0;
  int full_visit_idx = 0;  // 0..7
  int max_ctx = 0;
  bool cache_fp8 = false;

  int window = 0;  // 1024 (sliding) or 0 (full)
  bool log_scaling = false;

  void* workspace = nullptr;  // workspace_bytes(max_B)
  std::size_t workspace_size = 0;
  ActT* out = nullptr;  // [B, 3072]
};

// Workspace bytes required for batched_attn_decode up to max_b rows.
std::size_t attn_workspace_bytes(int max_b);

// Launches batched 1-token-per-slot attention for B independent sequences.
void batched_attn_decode(const BatchedAttnArgs& a, cudaStream_t stream);

// Per-slot 1-step 4-tap static convolution + state shift (attn_conv / mlp_conv, C = 2304).
// x is [B, ld] either fp32 (x_fp32=true, rounded to ActT before conv) or ActT (x_fp32=false).
// conv_state_visit is offset to visit `v`; slot `s` lives at `conv_state_visit + s * conv_slot_stride`.
// Padded rows (slot_ids[b] < 0) are skipped without mutating state.
void batched_static_conv_step(const void* x, bool x_fp32, int ld, int B, int C, const std::int32_t* slot_ids,
                              const float* conv_w, float* conv_state_visit, std::size_t conv_slot_stride,
                              float* out, cudaStream_t stream);

// Scatters newly prefilled full-attention KV rows [pos0, pos0 + T) of all 8 full visits from a scratch
// contiguous buffer `[8, scratch_max_ctx, 8, 128]` into the paged pool `[num_pages, 8, 64, 8, 128]`.
void scatter_full_kv(const void* src_k, const void* src_v, int scratch_max_ctx, int pos0, int T, bool kv_fp8,
                     int slot, const std::int32_t* page_table, int max_pages_per_slot,
                     void* dst_k_pool, void* dst_v_pool, cudaStream_t stream);

// Gathers [0, n) full-attention KV rows of all 8 full visits from the paged pool back into a scratch buffer.
void gather_full_kv(const void* src_k_pool, const void* src_v_pool, int slot, const std::int32_t* page_table,
                    int max_pages_per_slot, int n, bool kv_fp8, void* dst_k, void* dst_v, int scratch_max_ctx,
                    cudaStream_t stream);

// Copies a partial (<64-token) full-attention KV tail [src_pos0, src_pos0 + tail_tokens) across all 8 full visits
// from scratch `[8, scratch_max_ctx, 8, 128]` into compact checkpoint tail `[8, tail_tokens, 8, 128]`.
void copy_tail_kv_from_scratch(const void* src_k, const void* src_v, int scratch_max_ctx, int src_pos0,
                               int tail_tokens, bool kv_fp8, void* dst_tail_k, void* dst_tail_v, cudaStream_t stream);

// Copies a compact checkpoint tail `[8, tail_tokens, 8, 128]` into scratch `[8, scratch_max_ctx, 8, 128]`
// at `[dst_pos0, dst_pos0 + tail_tokens)`.
void copy_tail_kv_to_scratch(const void* src_tail_k, const void* src_tail_v, int tail_tokens, bool kv_fp8,
                             void* dst_k, void* dst_v, int scratch_max_ctx, int dst_pos0, cudaStream_t stream);

// Copies a compact checkpoint tail `[8, tail_tokens, 8, 128]` into rows `[0, tail_tokens)` of private physical
// pool page `phys_page` inside `[num_pages, 8, 64, 8, 128]` (used on exact prompt hits with non-64-aligned length).
void copy_tail_kv_to_pool_page(const void* src_tail_k, const void* src_tail_v, int tail_tokens, bool kv_fp8,
                               int phys_page, void* dst_k_pool, void* dst_v_pool, cudaStream_t stream);

// Converts a slice `[v_offset, v_offset + chunk_V)` of ActT raw head matmul outputs `[B, chunk_V]` into
// softcapped FP32 logits `out[b, v_offset + v] = 30 * tanh(float(in[b, v]) / 30)` (stride total_V).
void batched_head_softcap(const ActT* in, int B, int chunk_V, int v_offset, int total_V, float* out,
                          cudaStream_t stream);

// Advances slot_pos[slot_ids[b]] += 1 on device for every b in [0, B) with slot_ids[b] >= 0.
void batched_advance_pos(const std::int32_t* slot_ids, int B, std::int64_t* slot_pos, cudaStream_t stream);

// ---- Batched multi-token speculative verification & rollback (W in {1, 2, 4, 8}) ---------------------------

struct BatchedVerifyAttnArgs {
  const void* proj = nullptr;  // [B * W, R] fp32 (proj_fp32=true) or ActT (proj_fp32=false), row r = b * W + w
  bool proj_fp32 = false;
  int B = 1;
  int W = 1;                                    // 1, 2, 4, or 8
  const std::int32_t* slot_ids = nullptr;       // device int32 [B], < 0 = padded inactive slot
  const std::int32_t* slot_w_limit = nullptr;   // optional device int32 [B], active trial width per slot in [1, W]
  const std::int64_t* slot_pos = nullptr;       // device int64 [max_slots], starting pos p0 (NOT modified here)

  const float* q_scale = nullptr;   // fp32 [24, 128]
  const float* r_scale = nullptr;   // fp32 [24, 16]
  const float* rel_proj = nullptr;  // fp32 [16, 1024]
  const float* k_conv = nullptr;    // sliding: fp32 [1024, 4]; full: k_static [4, 1024]
  const float* v_conv = nullptr;

  float* conv_k_visit = nullptr;
  float* conv_v_visit = nullptr;
  std::size_t conv_slot_stride = 0;

  // Optional conv input history [B * W, 1024] for speculative rollback (nullptr when W == 1).
  float* hist_k_visit = nullptr;
  float* hist_v_visit = nullptr;

  ActT* k_ring_visit = nullptr;
  ActT* v_ring_visit = nullptr;
  std::int64_t* ring_pos_visit = nullptr;
  std::size_t ring_slot_stride = 0;
  std::size_t ring_pos_slot_stride = 0;

  void* k_full_pool = nullptr;
  void* v_full_pool = nullptr;
  const std::int32_t* page_table = nullptr;
  int max_pages_per_slot = 0;
  int full_visit_idx = 0;
  int max_ctx = 0;
  bool cache_fp8 = false;

  int window = 0;  // 1024 (sliding) or 0 (full)
  bool log_scaling = false;

  void* workspace = nullptr;
  std::size_t workspace_size = 0;
  ActT* out = nullptr;  // [B * W, 3072]
};

// Workspace bytes required for batched_attn_verify up to (max_b, max_w).
std::size_t attn_verify_workspace_bytes(int max_b, int max_w);

// Launches race-free batched multi-token causal verify attention for B slots x W trial tokens.
// Does NOT modify slot_pos; caller commits slot_pos[s] += commit[b] once via batched_verify_rollback.
void batched_attn_verify(const BatchedVerifyAttnArgs& a, cudaStream_t stream);

// Multi-token per-slot 4-tap static conv for B slots x W trial rows (row r = b * W + w).
// Optionally records BF16-rounded inputs into hist [B * W, C] for rollback.
void batched_static_conv_verify(const void* x, bool x_fp32, int ld, int B, int W, int C,
                                const std::int32_t* slot_ids, const std::int32_t* slot_w_limit,
                                const float* conv_w, float* conv_state_visit, std::size_t conv_slot_stride,
                                float* out, float* hist, cudaStream_t stream);

// Pre-verify state snapshots for up to max_B batch items, width up to max_W (<= 8).
struct BatchedVerifyBackup {
  float* conv_k = nullptr;     // [max_B, 48, 2, 1024]
  float* conv_v = nullptr;     // [max_B, 48, 2, 1024]
  float* conv_attn = nullptr;  // [max_B, 48, 2, 2304]
  float* conv_mlp = nullptr;   // [max_B, 48, 2, 2304]
  ActT* ring_k = nullptr;      // [max_B, 40, max_W, 1024]
  ActT* ring_v = nullptr;      // [max_B, 40, max_W, 1024]
  std::int64_t* ring_pos = nullptr;  // [max_B, 40, max_W]
  int max_W = 8;
};

struct BatchedVerifyStatePtrs {
  float* conv_k = nullptr;
  float* conv_v = nullptr;
  float* conv_attn = nullptr;
  float* conv_mlp = nullptr;
  std::size_t conv_kv_slot_stride = 0;
  std::size_t conv_h_slot_stride = 0;
  ActT* k_ring = nullptr;
  ActT* v_ring = nullptr;
  std::int64_t* ring_pos = nullptr;
  std::size_t ring_slot_stride = 0;
  std::size_t ring_pos_slot_stride = 0;
  std::int64_t* slot_pos = nullptr;
};

struct BatchedVerifyHistory {
  const float* hist_k = nullptr;     // [48, visit_kv_stride] (row b * W + w at offset (b * W + w) * 1024)
  const float* hist_v = nullptr;
  const float* hist_attn = nullptr;  // [48, visit_h_stride] (row b * W + w at offset (b * W + w) * 2304)
  const float* hist_mlp = nullptr;
  std::size_t kv_visit_stride = 0;
  std::size_t h_visit_stride = 0;
};

// Snapshots conv rows [1, 2] and sliding-ring entries [(p0 + w) % 1024] for w < slot_w_limit[b] before verify.
void batched_verify_save(const BatchedVerifyStatePtrs& st, const BatchedVerifyBackup& bk,
                         const std::int32_t* slot_ids, const std::int32_t* slot_w_limit, int B, int W,
                         cudaStream_t stream);

// Rolls back conv states & sliding-ring rows for w in [commit[b], slot_w_limit[b]) and advances
// slot_pos[slot_ids[b]] += commit[b] exactly once per active slot.
void batched_verify_rollback(const BatchedVerifyStatePtrs& st, const BatchedVerifyBackup& bk,
                             const BatchedVerifyHistory& hist, const std::int32_t* slot_ids,
                             const std::int32_t* slot_w_limit, const std::int32_t* commit_counts, int B, int W,
                             cudaStream_t stream);

}  // namespace ingine::kernels::batch

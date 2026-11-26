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
  bool cache_v4 = false;  // with cache_fp8: V pool holds E2M1 nibbles of v * 1.5 ([pages, 8, 64, 8, 64] bytes), K stays FP8

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
                     void* dst_k_pool, void* dst_v_pool, cudaStream_t stream, bool v4 = false);

// Writes T contiguous rows [T, KV_DIM] (positions pos0..pos0+T) of one full visit into the slot's pool pages
// (prefill straight into the pool; v4: FP8 rows -> FP4 V pages).
void write_full_kv_rows(const void* k_rows, const void* v_rows, int pos0, int T, int full_visit, bool kv_fp8, bool v4,
                        int slot, const std::int32_t* page_table, int max_pages_per_slot, void* dst_k_pool,
                        void* dst_v_pool, cudaStream_t stream);

// Gathers [0, n) full-attention KV rows of all 8 full visits from the paged pool back into a scratch buffer.
void gather_full_kv(const void* src_k_pool, const void* src_v_pool, int slot, const std::int32_t* page_table,
                    int max_pages_per_slot, int n, bool kv_fp8, void* dst_k, void* dst_v, int scratch_max_ctx,
                    cudaStream_t stream, bool v4 = false);

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
                               int phys_page, void* dst_k_pool, void* dst_v_pool, cudaStream_t stream, bool v4 = false);

// (v4: the pools hold FP4 V - KVF 2 - and the scratch / tails FP8 V: converted on the way.)
// Converts a slice `[v_offset, v_offset + chunk_V)` of ActT raw head matmul outputs `[B, chunk_V]` into
// softcapped FP32 logits `out[b, v_offset + v] = 30 * tanh(float(in[b, v]) / 30)` (stride total_V).
void batched_head_softcap(const ActT* in, int B, int chunk_V, int v_offset, int total_V, float* out,
                          cudaStream_t stream);

// Advances slot_pos[slot_ids[b]] += 1 on device for every b in [0, B) with slot_ids[b] >= 0.
void batched_advance_pos(const std::int32_t* slot_ids, int B, std::int64_t* slot_pos, cudaStream_t stream);

// Advances slot_pos[slot_ids[i]] += deltas[i] on device for every i in [0, S) with slot_ids[i] >= 0.
void batched_advance_pos_by(const std::int32_t* slot_ids, const std::int32_t* deltas, int S, std::int64_t* slot_pos,
                            cudaStream_t stream);

// Multi-row per-slot 4-tap static convolution (attn_conv / mlp_conv, C = 2304).
// Active slot i in [0, S) has id `active_slots[i]` (< 0 = padded skip), `m = slot_m[i]` (1..8) consecutive rows
// starting at `slot_row0[i]`. Walks those m rows in order, updates `conv_state_visit`, writes `out[R, C]`, and
// optionally records the BF16-rounded conv inputs into `hist[R, C]` for speculative rollback.
void batched_static_conv_rows(const void* x, bool x_fp32, int ld, int S, int C, const std::int32_t* active_slots,
                              const std::int32_t* slot_row0, const std::int32_t* slot_m, const float* conv_w,
                              float* conv_state_visit, std::size_t conv_slot_stride, float* out, float* hist,
                              cudaStream_t stream);

// Multi-row per-slot attention for speculative verification: R total rows across S active slots (m_s in 1..8
// consecutive rows per slot, padded rows have row_slot[r] < 0). Does NOT advance slot_pos.
// Pre-pass computes all m_s new K/V rows sequentially through K/V conv into `step_kv` [R, 2, KV_DIM] (recording
// optional `hist_k_visit`/`hist_v_visit` [R, KV_DIM] and shifting conv state). Full layers write all m_s rows into
// the paged pool before attention; sliding layers keep old ring intact during attention (reading old ring +
// `step_kv[base..r]`) and commit all m_s rows to the ring after attention.
struct BatchedAttnRowsArgs {
  const void* proj = nullptr;  // [R, proj_R] fp32 (proj_fp32=true) or ActT
  bool proj_fp32 = false;
  int R = 1;  // total rows (including any padded rows)
  int S = 1;  // active slot descriptor entries

  // Per-slot descriptors [S]
  const std::int32_t* active_slots = nullptr;  // slot id (< 0 = skip)
  const std::int32_t* slot_row0 = nullptr;     // starting row index in [0, R)
  const std::int32_t* slot_m = nullptr;        // rows for this slot (1..8)

  // Per-row descriptors [R]
  const std::int32_t* row_slot = nullptr;  // slot id (< 0 = padded skip)
  const std::int32_t* row_off = nullptr;   // offset 0..m_s-1 within slot
  const std::int32_t* row_base = nullptr;  // starting row index of this row's slot (= slot_row0)

  const std::int64_t* slot_pos = nullptr;  // [max_slots] committed position BEFORE this verify step

  const float* q_scale = nullptr;
  const float* r_scale = nullptr;
  const float* rel_proj = nullptr;
  const float* k_conv = nullptr;
  const float* v_conv = nullptr;

  float* conv_k_visit = nullptr;
  float* conv_v_visit = nullptr;
  std::size_t conv_slot_stride = 0;

  // Sliding layer (window == 1024)
  ActT* k_ring_visit = nullptr;
  ActT* v_ring_visit = nullptr;
  std::int64_t* ring_pos_visit = nullptr;
  std::size_t ring_slot_stride = 0;
  std::size_t ring_pos_slot_stride = 0;

  // Full layer (window == 0)
  void* k_full_pool = nullptr;
  void* v_full_pool = nullptr;
  const std::int32_t* page_table = nullptr;
  int max_pages_per_slot = 0;
  int full_visit_idx = 0;
  int max_ctx = 0;
  bool cache_fp8 = false;
  bool cache_v4 = false;

  int window = 0;
  bool log_scaling = false;

  // Multi-row step scratch & optional conv rollback history for this visit
  ActT* step_kv = nullptr;         // [R, 2, KV_DIM] ActT (reused across visits)
  float* hist_k_visit = nullptr;   // optional [R, KV_DIM] fp32
  float* hist_v_visit = nullptr;   // optional [R, KV_DIM] fp32

  void* workspace = nullptr;  // attn_workspace_bytes(max_R)
  std::size_t workspace_size = 0;
  ActT* out = nullptr;  // [R, HQ * 128]
};

void batched_attn_decode_rows(const BatchedAttnRowsArgs& a, cudaStream_t stream);

// Per-slot verify state backup & rollback for multi-row speculative verification (up to kMaxVerifyRows = 8 per slot).
inline constexpr int kMaxVerifyRows = 8;

struct BatchStatePoolPtrs {
  float* conv[4] = {};                  // conv_k, conv_v, conv_attn, conv_mlp bases
  std::size_t conv_slot_stride[4] = {}; // floats per slot (kv_stride, kv_stride, h_stride, h_stride)
  ActT* k_ring = nullptr;               // [max_slots, num_sliding, 1024, KV_DIM]
  ActT* v_ring = nullptr;
  std::int64_t* ring_pos = nullptr;     // [max_slots, num_sliding, 1024]
  std::size_t ring_slot_stride = 0;
  std::size_t ring_pos_slot_stride = 0;
  int num_visits = 48;
  int num_sliding_visits = 40;
};

struct BatchVerifyBackup {
  // conv[t]: [max_slots, num_visits, 2, C_t] (rows 1 and 2 of the 3-row conv state before the step)
  float* conv[4] = {};
  std::size_t conv_slot_stride[4] = {}; // floats per slot (= num_visits * 2 * C_t)
  // k, v: [max_slots, num_sliding, kMaxVerifyRows, KV_DIM]; pos: [max_slots, num_sliding, kMaxVerifyRows]
  ActT* k = nullptr;
  ActT* v = nullptr;
  std::int64_t* pos = nullptr;
  std::size_t kv_slot_stride = 0;       // ActT elements per slot (= num_sliding * 8 * KV_DIM)
  std::size_t pos_slot_stride = 0;      // int64 elements per slot (= num_sliding * 8)
};

// Saves conv rows 1..2 and sliding ring slots (slot_pos[s] + 0..m_s-1) % 1024 for each active slot in [0, S).
void batch_verify_save(const BatchStatePoolPtrs& st, const BatchVerifyBackup& bk, const std::int32_t* active_slots,
                       const std::int32_t* slot_m, int S, const std::int64_t* slot_pos, cudaStream_t stream);

// Restores conv states and sliding ring slots past `slot_commit[i]` (= accepted + 1) for each active slot in [0, S).
// `hist[t]` is [num_visits, hist_row_stride, C_t] fp32 recording each row's BF16-rounded conv input.
// Call BEFORE batched_advance_pos_by (reads pre-step slot_pos[s]).
void batch_verify_rollback(const BatchStatePoolPtrs& st, const BatchVerifyBackup& bk, const float* const hist[4],
                           int hist_row_stride, const std::int32_t* active_slots, const std::int32_t* slot_row0,
                           const std::int32_t* slot_m, const std::int32_t* slot_commit, int S,
                           const std::int64_t* slot_pos, cudaStream_t stream);

// Two-GPU in-place FP32 all-reduce over CUDA peer access (TP2 decode): `data` [n] (n % 4 == 0, n <= cap) on this
// rank; slot s of a ring of slots that both ranks use in the same order. Block b pushes its chunk into the peer's
// slot, fences, raises the peer's flag (s, b) to the next sequence number, waits for its own, then adds the peer's
// chunk: data = local + peer on both ranks (IEEE addition commutes: bit-identical on both).
struct P2pAllReduce2 {
  static constexpr int kBlocks = 8;
  float* my_recv = nullptr;        // [slots][cap] on this device
  float* peer_recv = nullptr;      // [slots][cap] on the peer
  unsigned* my_flag = nullptr;     // [slots][kBlocks] on this device (the peer raises them)
  unsigned* peer_flag = nullptr;   // [slots][kBlocks] on the peer
  unsigned* my_seq = nullptr;      // [slots][kBlocks] on this device
  int slots = 0, cap = 0;          // cap: floats per slot
  bool active() const { return my_recv != nullptr; }
};
void p2p_allreduce2(float* data, int n, const P2pAllReduce2& ar, int slot, cudaStream_t stream);

// Per-row banned token list for distributed greedy argmax (matches head::BannedTokens).
struct BannedList {
  static constexpr int kMax = 4;
  int count = 0;
  std::int32_t ids[kMax] = {-1, -1, -1, -1};
};

// Two-GPU distributed greedy argmax over sharded FP32 logits [rows, stride]. Each rank scans [0, local_V)
// with global token ids [v_offset, v_offset + local_V), honours optional per-row `banned` [rows] (global ids),
// exchanges (max_val, argmax_id) over `ar` at `slot`, and writes global argmax ids (ties -> lowest id, matching
// head::argmax identically) into `out_tokens` [rows]. `rows` must be in [1, 128].
std::size_t p2p_argmax2_workspace_bytes(int max_rows);
void p2p_argmax2(const float* local_logits, int rows, int local_V, std::int64_t stride, int v_offset,
                 const BannedList* banned, void* ws, const P2pAllReduce2& ar, int slot,
                 std::int32_t* out_tokens, cudaStream_t stream);

}  // namespace ingine::kernels::batch

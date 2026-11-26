// Prefill kernels (new for ingine): the pieces of model.Model.prefill for a chunk of T tokens that the decode
// kernels do not cover. Launchers are raw-pointer, run on `stream`, never allocate or synchronise. Shapes are
// row-major (Python order). H = 2304; positions of chunk row t are pos0 + t.
#pragma once

#include "core/act.h"
#include "core/gpu_compat.h"
#include "kernels/common/persona_abi.h"

#include <cstddef>
#include <cstdint>

namespace ingine::kernels::prefill {

#ifndef INGINE_TP_SIZE
#define INGINE_TP_SIZE 1
#endif
inline constexpr int kTpSize = INGINE_TP_SIZE;
inline constexpr int kHidden = 2304;
inline constexpr int kHeads = 24 / kTpSize, kKvHeads = 8 / kTpSize, kHeadDim = 128, kRel = 16, kRelExtent = 1024;
inline constexpr int kSinkPad = (kHeads + 7) & ~7;
inline constexpr int kKvDim = kKvHeads * kHeadDim;  // 1024
inline constexpr int kKvQ4NibbleBytes = kKvDim / 2;              // 512
inline constexpr int kKvQ4ScaleBytes = kKvHeads * (kHeadDim / 32) * 2;  // 64
inline constexpr int kKvQ4RowBytes = kKvQ4NibbleBytes + kKvQ4ScaleBytes;  // 576
inline constexpr int kInter = 1152, kExperts = 128, kTopK = 8;
inline constexpr int kAttnKeyTile = 64;    // keys per attention tile (tile bounds granularity)
inline constexpr int kAttnMaxTiles = 64;   // explicit-position key tiles per call (ring + chunk)

// ------------------------------------------------------------------------------------------------- GEMM

// cuBLASLt BF16 GEMM: C[M, N] (row-major, ldc) = A[M, K] (bf16, lda) . B[N, K]^T (bf16, ldb), FP32 accumulation
// and FP32 split-K reductions; C is bf16 (rounded once) or fp32.
class Gemm {
 public:
  Gemm();
  ~Gemm();
  Gemm(const Gemm&) = delete;
  Gemm& operator=(const Gemm&) = delete;
  void run(const ActT* A, int lda, const ActT* B, int ldb, void* C, int ldc, bool c_fp32, int M,
           int N, int K, cudaStream_t stream);

 private:
  struct Impl;
  Impl* impl_;
};

// ------------------------------------------------------------------------------------------- attention

// Keys/values of one sliding visit for the chunk: rows [0, 1024) are the ring before the chunk, rotated so
// row i holds slot (pos0 + i) % 1024 (position order for a filled ring), rows [1024, 1024 + T) the chunk.
// ring_gather fills rows [0, 1024) and the positions of every row (kpos int32 [kpos_rows], -1 = empty, padding
// rows -1); attn_prep writes the chunk rows.
void ring_gather(const ActT* k_ring, const ActT* v_ring, const int64_t* ring_pos, int pos0, int T,
                 ActT* k_rows, ActT* v_rows, int* kpos, int kpos_rows, cudaStream_t stream);

// model.prefill's ring update: the last min(T, 1024) chunk rows (rows 1024.. of k_rows/v_rows) into their slots
// (pos % 1024), ring_pos = pos.
void ring_update(const ActT* k_rows, const ActT* v_rows, int pos0, int T, ActT* k_ring,
                 ActT* v_ring, int64_t* ring_pos, cudaStream_t stream);

// Per key tile of kAttnKeyTile rows: int4 {min valid position, max valid position, all rows valid, any valid}.
void key_tile_bounds(const int* kpos, int L, int4* bounds, cudaStream_t stream);

struct AttnPrepParams {
  const ActT* proj = nullptr;  // [T, ld] BF16-rounded projections in SEG order (fast.proj_layout)
  int ld = 0;
  int T = 0, pos0 = 0;
  bool full = false;
  const float* q_scale = nullptr;  // [24, 128]
  const float* r_scale = nullptr;  // [24, 16]
  const float* k_w = nullptr;      // full: k_static [4, 1024] by lag; sliding: k_conv [1024, 4] conv1d order
  const float* v_w = nullptr;
  const float* conv_k = nullptr;   // [3, 1024] previous conv inputs (read only; see conv_state_update)
  const float* conv_v = nullptr;
  ActT* q = nullptr;               // out [T, 24, 128]
  ActT* rel = nullptr;             // out [T, 24, 16]
  void* k_out = nullptr;           // out [T, 8, 128] ActT/fp8, or [T, 576] u8 Q4_0 (out_q4: full caches)
  void* v_out = nullptr;
  void* const* k_pages = nullptr;  // optional [num_pages] Q4_0 page pointers (indexed by (pos0+t) >> page_shift)
  void* const* v_pages = nullptr;
  int page_shift = 0;
  bool out_fp8 = false;
  bool out_q4 = false;
};
// model._qkv per token: q (scale, clip, tau, BF16), rel (scale, clip, tau, BF16), keys/values through the
// static (sliding) or dynamic (full) conv, BF16, key clip / value RMS norm, stored at the cache precision.
void attn_prep(const AttnPrepParams& p, cudaStream_t stream);

// rel_logits[t, h, d] = bf16(sum_j rel[t, h, j] * bf16(rel_proj[j, d])), rel bf16 [T, 24, 16], rel_proj fp32
// [16, 1024], out bf16 [T, 24, 1024].
void rel_logits(const ActT* rel, const float* rel_proj, int T, ActT* out, cudaStream_t stream);

struct AttnParams {
  const ActT* q = nullptr;  // [T, 24, 128]
  const void* k = nullptr;  // key rows [L, 8, 128] ActT/fp8, or [L, 576] u8 Q4_0 (kv_q4)
  const void* v = nullptr;
  void* const* k_pages = nullptr;  // optional [num_pages] Q4_0 K/V page pointers (indexed by j >> page_shift)
  void* const* v_pages = nullptr;
  int page_shift = 0;
  int k_pos_base = 0;              // absolute token position of key row 0 in k/v (0 for full prefix)
  bool kv_fp8 = false;
  bool kv_q4 = false;
  const int* kpos = nullptr;         // positions of the L keys (padded to tiles with -1), or null: key j at k_pos_base + j
  const int4* bounds = nullptr;      // key_tile_bounds of kpos (with kpos)
  int L = 0;
  int T = 0, pos0 = 0;
  int window = 0;                    // 0: causal; else keys with 0 <= qpos - kpos < window
  const ActT* rel = nullptr;  // rel_logits [T, 24, 1024]
  const ActT* proj = nullptr; // [T, ld]: gate (sigmoid input) and sink logits in SEG order
  int ld = 0;
  ActT* out = nullptr;        // [T, 3072] = bf16(bf16(attention * sigmoid(lse - sink)) * sigmoid(gate))
  // Optional chunked online-softmax merge across staged KV batches (non-null only during paged KV streaming):
  float* partial_o = nullptr;        // [T, 24, 128] FP32 running unscaled numerator O (2^15 scaled in prefill, 1x in split-K decode)
  float* partial_m = nullptr;        // [T, 24] FP32 running log2 row max
  float* partial_l = nullptr;        // [T, 24] FP32 running row sum
  bool partial_init = true;          // true on first batch; false resumes from partial_*
  bool partial_final = true;         // true on final batch (writes out); false saves partial_*
  // Optional split-K workspace for low-M (T <= 8) Q4_0 streamed decode: [decode_splits, T, 24, 130] FP32.
  float* decode_ws = nullptr;
  int decode_splits = 0;
};
// flex.attend + the output gate of model._visit: scores (q.k)/sqrt(128) + rel bias (distance < 1024), masked,
// softmax with the per-head sink folded in through the log-sum-exp. Q.K in BF16 MMA with FP32 accumulation
// (exact products), probabilities in FP16 (scaled by 2^15) times FP16 values (exact for BF16/E4M3 values in
// range), FP32 accumulation; FP32 softmax statistics.
void attention(const AttnParams& p, cudaStream_t stream);

// ----------------------------------------------------------------------------------------- convolutions

// model.static_conv on BF16-valued inputs: out[t, c] = (sum_j w[c, j] * ext[t + j] + x[t, c]) / sqrt(5),
// ext = [state (3 rows), x]; x bf16 [T, ld], w fp32 [C, 4], state fp32 [3, C], out fp32 [T, C].
void static_conv(const ActT* x, int ld, int T, int C, const float* w, const float* state, float* out,
                 cudaStream_t stream);

// state <- the last three rows of [state, x] (x bf16 [T, ld], C channels). Run after every reader of state.
void conv_state_update(float* state, const ActT* x, int ld, int T, int C, cudaStream_t stream);
void static_conv_f32(const float* x, int ld, int T, int C, const float* w, const float* state, float* out, cudaStream_t stream);
void conv_state_update_f32(float* state, const float* x, int ld, int T, int C, cudaStream_t stream);

// ---------------------------------------------------------------------------------------- message gates

// gin[g, t] = bf16([rms_norm(h[g, t]), rms_norm(msg[g, t])]) (bf16 [G, T, 2H]); h_g / msg_g fp32 [T, H].
struct GateRows {
  const float* h[3] = {};
  const float* msg[3] = {};
};
void gate_inputs(const GateRows& rows, int G, int T, ActT* gin, cudaStream_t stream);

// out[t] = bf16(rms_norm((1 - tanh(f)) * h + tanh(g) * msg) * norm_w); f, g fp32 [T, H] (the forget / gain
// products of the BF16 down projection).
void gate_apply(const float* h, const float* msg, const float* f, const float* g, const float* norm_w, int T,
                ActT* out, cudaStream_t stream);

// ------------------------------------------------------------------------------------------------- MoE

struct MoeWork {
  int* counts = nullptr;      // [128] (zeroed by moe_topk's caller)
  int* tiles_n = nullptr;     // [1]
  int4* tiles = nullptr;      // [moe_max_tiles(T)] {expert, first pair, pairs, 0}
  int* pair_token = nullptr;  // [T * 8] token of each expert-sorted pair
  int* pair_dst = nullptr;    // [T * 8] t * 8 + slot of each expert-sorted pair
};
// (token, expert) pairs per grouped-GEMM tile: 32 for small chunks (few pairs per expert: deeper weight
// prefetch), 128 otherwise. INGINE_MOE_TILE_ROWS (32 or 128) overrides it (benchmarks).
int moe_tile_rows(int T);
inline int moe_max_tiles(int T, int tile_rows = 0) {
  const int tr = tile_rows > 0 ? tile_rows : moe_tile_rows(T);
  return (T * kTopK + tr - 1) / tr + kExperts;
}
// tile-list capacity for every chunk size up to max_T
int moe_tiles_capacity(int max_T);

// Router selection on FP32 logits [T, 128] (x . router^T): s = sigmoid, top-8 of s + bias (descending),
// wts = 8 s / (sum s + 1e-20); counts[e] += hits.
void moe_topk(const float* logits, const float* bias, int T, int* idx, float* wts, int* counts, cudaStream_t stream);
// Fused mech-interp / persona variant of moe_topk (ablate, slot-preserving force, route-replay, slot-permute,
// read-only routing tap); exact 0.0 bit-identical on every no-op form.
void moe_topk_hooked(const float* logits, const float* bias, int T, int pos0, int* idx, float* wts, int* counts,
                     const persona::RouterHookVisit* hook, cudaStream_t stream);
// Expert-sorted pair lists and tile_rows (default moe_tile_rows(T)) tiles from idx and counts.
void moe_route(const int* idx, int T, const MoeWork& w, cudaStream_t stream, int tile_rows = 0);

// Runs 3b per-neuron affine hooks followed by 3a read-only expert activation taps on prefill sorted-pair `act`
// [T * 8, 1152] right between moe_gate_up and moe_down. Optional `active_mask` ([128] u8) restricts execution
// to the current expert-cache batch in tiered offload mode.
void apply_expert_hooks_prefill(ActT* act, const MoeWork& w, int T, int pos0, const persona::ExpertTapVisit* hook,
                                cudaStream_t stream, const std::uint8_t* active_mask = nullptr);

struct MoeExperts {
  int fmt = 0;  // w13 format: 0 NVFP4, 1 MXFP8, 2 ATLAS2, 3 ATLAS3, 4 EXL3_2, 5 EXL3_3
  const void* w13 = nullptr;
  const void* w13_scale = nullptr;  // block scales, or w13_rin (F16 [128, H]) for EXL3
  const float* w13_factor = nullptr;
  const void* w2 = nullptr;
  const void* w2_scale = nullptr;   // block scales, or w2_rin (F16 [128, I]) for EXL3
  const float* w2_factor = nullptr;
  int w2_fmt = -1;                  // w2 format (-1 -> same as fmt)
  const void* w13_rout = nullptr;   // F16 [128, 2I] for EXL3
  const void* w2_rout = nullptr;    // F16 [128, H] for EXL3
  ActT* exl3_scratch = nullptr;     // ActT [T * 8, 2I] scratch for EXL3 stacked gate/up (e.g. down_)
  // Optional per-expert pointer tables ([128]) and active-expert mask ([128] u8) for tiered offloading.
  const std::uint8_t* const* w13_ptrs = nullptr;
  const std::uint8_t* const* w13_scale_ptrs = nullptr;
  const std::uint8_t* const* w13_rout_ptrs = nullptr;
  const std::uint8_t* const* w2_ptrs = nullptr;
  const std::uint8_t* const* w2_scale_ptrs = nullptr;
  const std::uint8_t* const* w2_rout_ptrs = nullptr;
  const std::uint8_t* active_mask = nullptr;
  int tile_rows = 0;  // 0 = moe_tile_rows(T); EXL3 uses 32 (3-CTA/SM, zero register spills)
};
// act[pair] = bf16(softcap(g, 4) * sigmoid(g) * softcap(u, 25)), g/u = bf16(W13_e . x[token]) (experts decoded
// exactly as model.decode_experts). x bf16 [T, H], act bf16 [T * 8, 1152] by sorted pair.
void moe_gate_up(const MoeExperts& e, const ActT* x, int T, const MoeWork& w, ActT* act,
                 cudaStream_t stream);
// down[t * 8 + slot] = bf16(W2_e . act[pair]), bf16 [T * 8, H].
void moe_down(const MoeExperts& e, const ActT* act, int T, const MoeWork& w, ActT* down,
              cudaStream_t stream);
// out[t] = bf16(sum over slots in order of down[t, slot] * wts[t, slot]), bf16 [T, H].
void moe_combine(const ActT* down, const float* wts, int T, ActT* out, cudaStream_t stream);
// EP partial combine: sums only slots whose expert has active_mask[e] != 0 into FP32 [T, H].
void moe_combine_ep(const ActT* down, const float* wts, const int* idx, const std::uint8_t* active_mask, int T, float* out_f32, cudaStream_t stream);

// ------------------------------------------------------------------------------------------------ misc

// int32 -> int64 token ids; a negative id (an image token, whose embedding row put_rows replaces) becomes
// `negative_as`, so the embedding lookup stays inside its table.
void widen_ids(const int32_t* in, int T, int64_t* out, int32_t negative_as, cudaStream_t stream);
// dst[r] = float(src[r]) for `count` rows of H: precomputed embedding rows (image tokens) over the rows of
// the embedding block. src bf16 [count, H], dst f32 [count, H].
void put_rows(const ActT* src, int count, float* dst, cudaStream_t stream);
// out[t * out_stride] = bf16(x[t]) per row, x fp32 [T, H] (DSpark taps).
void store_tap(const float* x, int T, ActT* out, int64_t out_stride, cudaStream_t stream);
// model.logits for one row: out[v] = 30 tanh(bf16(w_v . x) / 30), w bf16 [V, H] or FP8 rows dequantized
// as weights.dequant (bf16(fp8 * scale[v])); x bf16 [H].
void head_logits(const ActT* x, const void* w, const float* scale, int V, float* out, cudaStream_t stream);
// Round FP32 buffer to ActT (BF16/FP16) elementwise (used by paged KV Tier-2 NVMe decode fallback).
void f32_to_act(const float* src, ActT* dst, std::size_t n, cudaStream_t stream);

}  // namespace ingine::kernels::prefill

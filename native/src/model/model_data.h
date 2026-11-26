// C++ mirrors of the Python engine's model data: weights.py (constants, visit plan, Weights/Layer/Experts),
// model.py (State, Ablation, TAP_VISITS) and fast.py (projection layout, Buffers per call width).
//
// Tensors are ingine::Tensor views over device memory. Shape convention: Tensor::ne[] lists dimensions
// innermost first (ggml order), so a Python tensor of shape [a, b, c] has ne = {c, b, a}; every field
// comment below gives the Python (row-major) shape. All tensors are contiguous unless noted.
// `rm(data, dtype, {a, b, c})` builds a view from a Python-order shape.
#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::model {

class ExpertCache;

// ------------------------------------------------------------------------------------ weights.py constants

#ifndef INGINE_TP_SIZE
#define INGINE_TP_SIZE 1
#endif
inline constexpr int TP_SIZE = INGINE_TP_SIZE;
static_assert(TP_SIZE == 1 || TP_SIZE == 2 || TP_SIZE == 4, "INGINE_TP_SIZE must be 1, 2, or 4");
inline constexpr int HIDDEN = 2304;
inline constexpr int TOTAL_HEADS = 24, TOTAL_KV_HEADS = 8, HEAD_DIM = 128;
inline constexpr int HEADS = TOTAL_HEADS / TP_SIZE, KV_HEADS = TOTAL_KV_HEADS / TP_SIZE;
inline constexpr int SEG_SINK_PAD = (HEADS + 7) & ~7;
inline constexpr int D_REL = 16, REL_EXTENT = 1024, WINDOW = 1024;
inline constexpr int INTER = 1152, EXPERTS = 128, TOP_K = 8;
inline constexpr int CONV_K = 4;
inline constexpr int LAYERS = 24, BLOCK = 6;
inline constexpr int LOOP_START = 6, LOOP_END = 18, PASSES = 3, MAX_PASSES = 8;
inline constexpr int VOCAB = 201024, UNPADDED_VOCAB = 200058;
inline constexpr int LOCAL_VOCAB_CHUNK = TP_SIZE == 1 ? UNPADDED_VOCAB : (((UNPADDED_VOCAB + TP_SIZE - 1) / TP_SIZE + 15) & ~15);
inline constexpr int GATE_RANK = 64;
inline constexpr float EPS = 1e-6f;
inline constexpr double QK_CLIP = 22.627416997969522, REL_CLIP = 8.0;
inline constexpr double LOG_ALPHA = 0.1, LOG_FLOOR = 8192.0;
inline constexpr float GATE_SOFTCAP = 4.0f, UP_SOFTCAP = 25.0f, LOGIT_SOFTCAP = 30.0f;
inline constexpr float ROUTE_SCALE = 8.0f;
inline constexpr float FP8_MAX = 448.0f;  // largest finite float8_e4m3fn (model.FP8_MAX)
inline constexpr int KV_DIM = KV_HEADS * HEAD_DIM;  // 1024
// Q4_0 per-row layout for full-attention caches: [8 heads x 64B nibbles | 8 heads x 4 FP16 scales] = 576B/row
// (18B per 32-element block, 64-byte aligned row stride, 9 KiB/tok across 8 full visits x K/V).
inline constexpr int KV_Q4_GROUP = 32;
inline constexpr int KV_Q4_BLOCKS_PER_HEAD = HEAD_DIM / KV_Q4_GROUP;  // 4
inline constexpr int KV_Q4_NIBBLE_BYTES = KV_DIM / 2;                 // 512
inline constexpr int KV_Q4_SCALE_BYTES = KV_HEADS * KV_Q4_BLOCKS_PER_HEAD * 2;  // 64
inline constexpr int KV_Q4_ROW_BYTES = KV_Q4_NIBBLE_BYTES + KV_Q4_SCALE_BYTES;  // 576
// Q4_0 per-row layout for token embeddings: [1152B nibbles | 72 FP16 scales = 144B] = 1296B/row (16-byte aligned).
inline constexpr int EMBED_Q4_NIBBLE_BYTES = HIDDEN / 2;            // 1152
inline constexpr int EMBED_Q4_SCALE_BYTES = (HIDDEN / 32) * 2;      // 144
inline constexpr int EMBED_Q4_ROW_BYTES = EMBED_Q4_NIBBLE_BYTES + EMBED_Q4_SCALE_BYTES;  // 1296

inline constexpr bool is_full_layer(int l) { return l == 5 || l == 11 || l == 17 || l == 23; }
inline constexpr bool is_loop_layer(int l) { return LOOP_START <= l && l < LOOP_END; }

// --------------------------------------------------------------------------------------------- visit plan

struct Visit {
  int layer;
  int pass;  // loop pass (0 outside the loop)
};

inline constexpr int NUM_VISITS = 48;
inline constexpr int NUM_FULL_VISITS = 8;
inline constexpr int NUM_FULL_SLOTS_LAST_FULL = 4;  // physical full layers: L5, L11 (shared across passes 0..2), L17 (shared), L23
inline constexpr int NUM_SLIDING_VISITS = NUM_VISITS - NUM_FULL_VISITS;

// weights.visit_plan(): layers 0-5, layers 6-17 three times, layers 18-23
inline constexpr std::array<Visit, NUM_VISITS> VISITS = [] {
  std::array<Visit, NUM_VISITS> plan{};
  int v = 0;
  for (int l = 0; l < LOOP_START; ++l) plan[v++] = {l, 0};
  for (int p = 0; p < PASSES; ++p)
    for (int l = LOOP_START; l < LOOP_END; ++l) plan[v++] = {l, p};
  for (int l = LOOP_END; l < LAYERS; ++l) plan[v++] = {l, 0};
  return plan;
}();

// model.State.ring_index / full_index: visit -> index into the sliding rings / full caches (-1: other kind)
inline constexpr std::array<int, NUM_VISITS> RING_INDEX = [] {
  std::array<int, NUM_VISITS> r{};
  int n = 0;
  for (int v = 0; v < NUM_VISITS; ++v) r[v] = is_full_layer(VISITS[v].layer) ? -1 : n++;
  return r;
}();
inline constexpr std::array<int, NUM_VISITS> FULL_INDEX = [] {
  std::array<int, NUM_VISITS> r{};
  int n = 0;
  for (int v = 0; v < NUM_VISITS; ++v) r[v] = is_full_layer(VISITS[v].layer) ? n++ : -1;
  return r;
}();
// LastFull mode: L5 -> 0, L11 (passes 0,1,2) -> 1, L17 (passes 0,1,2) -> 2, L23 -> 3; sliding visits -> -1.
inline constexpr std::array<int, NUM_VISITS> FULL_INDEX_LAST_FULL = [] {
  std::array<int, NUM_VISITS> r{};
  for (int v = 0; v < NUM_VISITS; ++v) {
    const int l = VISITS[v].layer;
    r[v] = (l == 5) ? 0 : (l == 11) ? 1 : (l == 17) ? 2 : (l == 23) ? 3 : -1;
  }
  return r;
}();
inline constexpr int full_slot_count(bool kv_last_full) noexcept {
  return kv_last_full ? NUM_FULL_SLOTS_LAST_FULL : NUM_FULL_VISITS;
}
inline constexpr int full_slot_index(int v, bool kv_last_full) noexcept {
  return kv_last_full ? FULL_INDEX_LAST_FULL[v] : FULL_INDEX[v];
}

// Runtime visit schedule for variable loop depth L in [1, MAX_PASSES] (default L = PASSES = 3 -> 48 visits).
struct VisitPlan {
  int passes = PASSES;
  int num_visits = NUM_VISITS;
  int num_full_visits = NUM_FULL_VISITS;
  int num_sliding_visits = NUM_SLIDING_VISITS;
  std::vector<Visit> visits;
  std::vector<int> ring_index;
  std::vector<int> full_index;

  int last_pass() const noexcept { return passes - 1; }
};

inline VisitPlan make_visit_plan(int passes) {
  if (passes < 1 || passes > MAX_PASSES) {
    throw std::invalid_argument("loop_passes must be in 1.." + std::to_string(MAX_PASSES) + ", got " +
                                std::to_string(passes));
  }
  VisitPlan plan;
  plan.passes = passes;
  const int loop_len = LOOP_END - LOOP_START;
  plan.num_visits = LOOP_START + passes * loop_len + (LAYERS - LOOP_END);
  plan.visits.reserve(std::size_t(plan.num_visits));
  for (int l = 0; l < LOOP_START; ++l) plan.visits.push_back({l, 0});
  for (int p = 0; p < passes; ++p)
    for (int l = LOOP_START; l < LOOP_END; ++l) plan.visits.push_back({l, p});
  for (int l = LOOP_END; l < LAYERS; ++l) plan.visits.push_back({l, 0});
  plan.ring_index.resize(std::size_t(plan.num_visits), -1);
  plan.full_index.resize(std::size_t(plan.num_visits), -1);
  int nf = 0, ns = 0;
  for (int v = 0; v < plan.num_visits; ++v) {
    if (is_full_layer(plan.visits[std::size_t(v)].layer)) plan.full_index[std::size_t(v)] = nf++;
    else plan.ring_index[std::size_t(v)] = ns++;
  }
  plan.num_full_visits = nf;
  plan.num_sliding_visits = ns;
  return plan;
}

// Prefiller per-pass persistent block states: B2[p] + E[p] = 2 * chunk * HIDDEN * sizeof(float) per pass.
inline constexpr std::size_t prefill_loop_pass_bytes(int prefill_chunk, int loop_passes) {
  return std::size_t(loop_passes) * 2 * std::size_t(prefill_chunk) * std::size_t(HIDDEN) * sizeof(float);
}

// Offload VRAM budget estimate for Prefiller (aliased phase arena + persistent block states for `loop_passes`).
// Matches the original L=3 budget (chunk * 280 KiB + 48 MiB) exactly while tracking +/- 18 KiB/tok per pass.
inline constexpr std::size_t offload_prefill_vram_bytes(int prefill_chunk, int loop_passes) {
  const std::size_t base_per_tok = std::size_t(280) * 1024 - std::size_t(PASSES) * 2 * std::size_t(HIDDEN) * sizeof(float);
  return std::size_t(prefill_chunk) * base_per_tok + (std::size_t(48) << 20) +
         prefill_loop_pass_bytes(prefill_chunk, loop_passes);
}

// model.TAP_VISITS: DSpark features = the partial block entering these visits, then the final normed hidden
inline constexpr std::array<int, 3> TAP_VISITS = {5, 23, 41};
inline constexpr int NUM_TAPS = 4;

// ----------------------------------------------------------------------------- fast.py projection layout

// SEG_ROWS order (KERNELS.md): q | gate | rel | sink | k | v | (full layers) k_dyn | v_dyn
enum Seg : int { SEG_Q = 0, SEG_GATE, SEG_REL, SEG_SINK, SEG_K, SEG_V, SEG_K_DYN, SEG_V_DYN, NUM_SEGS };
inline constexpr std::array<int, NUM_SEGS> SEG_ROWS = {HEADS * HEAD_DIM, HEADS * HEAD_DIM, HEADS * D_REL, SEG_SINK_PAD,
                                                       KV_DIM,           KV_DIM,           KV_HEADS * CONV_K,
                                                       KV_HEADS * CONV_K};
inline constexpr std::array<int, NUM_SEGS> SEG_INPUT = {0, 0, 0, 0, 1, 2, 1, 2};  // xin index (q, k, v inputs)
inline constexpr std::array<int, NUM_SEGS + 1> SEG_BEGIN = [] {
  std::array<int, NUM_SEGS + 1> b{};
  for (int s = 0; s < NUM_SEGS; ++s) b[s + 1] = b[s] + SEG_ROWS[s];
  return b;
}();
inline constexpr int PROJ_ROWS_SLIDING = SEG_BEGIN[SEG_K_DYN];  // 8600
inline constexpr int PROJ_ROWS_FULL = SEG_BEGIN[NUM_SEGS];      // 8664
inline constexpr int num_segs(bool full) { return full ? NUM_SEGS : SEG_K_DYN; }
inline constexpr int proj_rows(bool full) { return full ? PROJ_ROWS_FULL : PROJ_ROWS_SLIDING; }

// --------------------------------------------------------------------------------------------- helpers

// View with a Python-order (row-major) shape.
inline Tensor rm(void* data, DType dtype, std::initializer_list<std::int32_t> python_shape) {
  std::int32_t s[4] = {1, 1, 1, 1};
  const int n = static_cast<int>(python_shape.size());
  int i = n - 1;
  for (std::int32_t d : python_shape) s[i--] = d;
  switch (n) {
    case 1: return Tensor(data, dtype, {s[0]});
    case 2: return Tensor(data, dtype, {s[0], s[1]});
    case 3: return Tensor(data, dtype, {s[0], s[1], s[2]});
    default: return Tensor(data, dtype, {s[0], s[1], s[2], s[3]});
  }
}

template <class T>
inline T* ptr(const Tensor& t) {
  return static_cast<T*>(t.data);
}

// ----------------------------------------------------------------------------------------------- weights

// A dense matrix as the decoder stores it: BF16 rows, or float8_e4m3fn rows with FP32 per-row scales
// (`scale.data != nullptr`; value = fp8 * scale, see weights.dequant / dense fp8_rows).
struct Dense {
  Tensor w;      // [rows, cols] bf16 or fp8
  Tensor scale;  // [rows] fp32, or empty (data == nullptr) for BF16
  bool fp8() const { return scale.data != nullptr; }
  int rows() const { return w.ne[1]; }
  int cols() const { return w.ne[0]; }
};

// GGUF_Q5L / GGUF_IQ4: experts of GGUF files (weights_gguf.cpp), 32-value groups. Codes: 4 bits a value (value 2i in
// the low nibble of byte i, as NVFP4); scales per group: Q5L 12 B = FP32 scale, FP32 min, 32 fifth bits (value =
// scale * code + min, code 0..31: Q5_K, Q4_K, Q5_0), IQ4 4 B = FP32 scale (value = scale * iq4nl[code]: IQ4_XS, IQ4_NL).
enum class ExpertFormat : int { NVFP4 = 0, MXFP8 = 1, ATLAS2 = 2, ATLAS3 = 3, EXL3_2 = 4, EXL3_3 = 5, GGUF_Q5L = 6, GGUF_IQ4 = 7 };

inline constexpr bool is_exl3(ExpertFormat f) { return f == ExpertFormat::EXL3_2 || f == ExpertFormat::EXL3_3; }

// weights.Experts (checkpoint encoding; see weights.py docstring)
struct Experts {
  ExpertFormat fmt = ExpertFormat::NVFP4;       // w13 format (legacy alias)
  ExpertFormat w13_fmt = ExpertFormat::NVFP4;   // w13 format (may differ from w2_fmt, e.g. EXL3 Map M2)
  ExpertFormat w2_fmt = ExpertFormat::NVFP4;    // w2 format
  // NVFP4:  w13 u8 [128, 2304, 1152], w13_scale e4m3 [128, 2304, 144], w2 u8 [128, 2304, 576],
  //         w2_scale e4m3 [128, 2304, 72], factors fp32 [128] (amax / outer_scale_denominator)
  // MXFP8:  w13 e4m3 [128, 2304, 2304], w13_scale u8 (E8M0) [128, 2304, 72], w2 e4m3 [128, 2304, 1152],
  //         w2_scale u8 [128, 2304, 36], factors empty
  // ATLAS2: 2-bit symmetric Lloyd-Max (group 16, 2.50 b/w): w13 u8 [128, 2304, 576], w2 u8 [128, 2304, 288],
  //         e4m3 scales & fp32 factors same as NVFP4
  // ATLAS3: 3-bit symmetric Lloyd-Max (group 16, 3.50 b/w): w13 u8 [128, 2304, 864], w2 u8 [128, 2304, 432],
  //         e4m3 scales & fp32 factors same as NVFP4
  // EXL3_2 / EXL3_3: EXL3 MCG trellis + 128-pt Hadamard (2 or 3 b/w):
  //         w13 u8 [128, 144, 144, 32*bits] (stacked [gate; up]), w13_rin f16 [128, 2304], w13_rout f16 [128, 2304];
  //         w2  u8 [128, 144,  72, 32*bits],                      w2_rin  f16 [128, 1152], w2_rout  f16 [128, 2304]
  Tensor w13, w13_scale, w2, w2_scale;
  Tensor w13_factor, w2_factor;
  Tensor w13_rin, w13_rout, w2_rin, w2_rout;
  // EP sharding (non-null when ep_size > 1): [128] device pointer tables & active mask
  Tensor w13_ptrs, w13_scale_ptrs, w2_ptrs, w2_scale_ptrs;
  Tensor active_mask;
  int ep_rank = 0, ep_size = 1;
};

// weights.Layer plus the per-layer derived tensors of fast.FastDecoder (_concat_projections,
// _stack_layer_params, _fp8_dense). Absent fields have data == nullptr.
struct Layer {
  int idx = 0;
  bool full = false, loop = false;

  // norms (fp32 [H])
  Tensor q_norm, k_norm, v_norm;  // full layers
  Tensor attn_norm;               // sliding layers
  Tensor mlp_norm;

  // attention projections: one [R, H] matrix in SEG order (R = proj_rows(full)); the per-projection
  // entries are row views of it (with the matching rows of its scales when FP8)
  Dense wproj;
  Dense wq, wgate, wr, wsink, wk, wv, wk_dyn, wv_dyn;
  Dense wo;  // [H, 3072]

  Tensor q_scale;   // fp32 [24, 128]
  Tensor r_scale;   // fp32 [24, 16]
  Tensor rel_proj;  // fp32 [16, 1024]
  Tensor k_static, v_static;  // full: fp32 [4, 1024] by lag (row 0 = current token)
  Tensor k_conv, v_conv;      // sliding: fp32 [1024, 4] conv1d order (tap 3 = current token)
  Tensor attn_conv, mlp_conv;  // fp32 [H, 4]
  Tensor router;       // fp32 [128, H]
  Tensor router_bias;  // fp32 [128]
  Tensor router_bf16;  // bf16 [128, H] (exact copy of the checkpoint values)
  Tensor attn_query;   // fp32 [Q, H] (Q = 3 full, 1 sliding), layers > 0
  Tensor mlp_query;    // fp32 [1, H]

  // loop layers: message gates. Stacks the kernels take, with per-gate views into them.
  Tensor qkv_gate_down;                      // bf16 [3, 64, 2H]
  Tensor qkv_gate_forget, qkv_gate_gain;     // fp32 [3, H, 64]
  Tensor qkv_gate_forget16, qkv_gate_gain16; // bf16 [3, H, 64]
  Tensor mlp1_gate_down;                     // bf16 [1, 64, 2H] (= mlp_gate_down)
  Tensor mlp1_gate_forget, mlp1_gate_gain;   // fp32 [1, H, 64]
  Tensor mlp1_gate_forget16, mlp1_gate_gain16;
  Tensor q_gate_down, k_gate_down, v_gate_down, mlp_gate_down;          // bf16 [64, 2H] views
  Tensor q_gate_forget, k_gate_forget, v_gate_forget, mlp_gate_forget;  // fp32 [H, 64] views
  Tensor q_gate_gain, k_gate_gain, v_gate_gain, mlp_gate_gain;          // fp32 [H, 64] views

  // stacked norm weights (fast._stack_layer_params)
  Tensor attn_norm3;     // fp32 [3, H]: (q, k, v) norms, or attn_norm three times
  Tensor attn_norm_mix;  // fp32 [3, H] (full) or [1, H] (sliding)
  Tensor mlp_norm1;      // fp32 [1, H]

  Experts experts;
};

// weights.Weights (+ fast.FastDecoder.final_norm1 / head_w). Owns its device memory.
struct Weights {
  std::vector<Layer> layers;  // LAYERS
  Tensor embed;               // bf16 [VOCAB, H] or q4_0 u8 [VOCAB, EMBED_Q4_ROW_BYTES]
  Tensor embed_norm;          // fp32 [H]
  Tensor final_query;         // fp32 [1, H]
  Tensor final_norm;          // fp32 [H] (also final_norm1 = [1, H] view of the same memory)
  Dense unembed;              // [UNPADDED_VOCAB, H] bf16, or FP8 rows + scales (dense_fp8)
  Tensor message_queries;     // fp32 [12, 4, H]: q, k, v, mlp per loop layer
  std::array<ExpertFormat, LAYERS> expert_formats{};
  std::array<ExpertFormat, LAYERS> w2_expert_formats{};
  Dense full_unembed;         // optional full [UNPADDED_VOCAB, H] on TP rank 0 for DSpark drafter
  int tp_rank = 0, tp_size = 1;
  int vocab_begin = 0, vocab_rows = UNPADDED_VOCAB;
  bool dense_fp8 = false;     // wproj (+ views), wo and unembed are FP8 rows
  bool embed_q4() const { return embed.dtype == DType::U8; }
  std::shared_ptr<ExpertCache> expert_cache;  // non-null when tiered MoE offloading is enabled
  // Verified cryptographic digest over model metadata + 100% of weight tensor contents, populated only by
  // load_weights(model_dir, ...). Absent (std::nullopt) for caller-supplied/synthetic Weights with no identity.
  std::optional<std::array<std::uint8_t, 32>> content_sha256;

  std::vector<DeviceBuffer> storage;  // device allocations behind the views
};

// model.Ablation: per visit y <- y - w * r * max(r . y - offset, floor) on the attention / MLP writes; offset 0
// with floor -FLT_MAX is the plain projection y - w * r * (r . y), floor 0 removes only what exceeds the
// offset. With detectors a write is changed only where det . y exceeds the visit's threshold. Without an
// ablation file (Ablation.none) directions and weights are zero.
struct Ablation {
  Tensor attn_dirs;  // fp32 [96, H]: rows 0..47 unit directions, rows 48..95 detectors (zero without them)
  Tensor attn_w;     // fp32 [48, 4] rows: weight, offset, floor, detector threshold
  Tensor mlp_dirs;   // fp32 [96, H]
  Tensor mlp_w;      // fp32 [48, 4]
  bool active = false;  // loaded from a file
  bool detect = false;  // the file has detectors

  std::vector<DeviceBuffer> storage;
};

// ------------------------------------------------------------------------------------------------- state

// model.State: per-request caches.
class PagedKvCache;

struct State {
  int max_ctx = 0;
  bool kv_fp8 = false;
  bool kv_q4 = false;
  bool kv_last_full = false;
  VisitPlan plan = make_visit_plan(PASSES);
  std::shared_ptr<PagedKvCache> paged_kv;  // non-null when 3-tier paged Q4_0 full-attention KV is enabled
  Tensor k_ring, v_ring;      // bf16 [plan.num_sliding_visits, 1024, 8, 128] (sliding visits)
  Tensor ring_pos;            // int64 [plan.num_sliding_visits, 1024], absolute position per slot, -1 = empty
  Tensor k_full, v_full;      // bf16/fp8 [num_full_slots, max_ctx, 8, 128] or q4_0 u8 [num_full_slots, max_ctx, 576]
  Tensor conv_k, conv_v;      // fp32 [plan.num_visits, 3, 1024] previous 3 key/value conv inputs (oldest first)
  Tensor conv_attn, conv_mlp; // fp32 [plan.num_visits, 3, H]
  int length = 0;             // tokens in the context (host copy; decode steps do not advance it)

  std::vector<DeviceBuffer> storage;

  int num_full_slots() const noexcept { return kv_last_full ? NUM_FULL_SLOTS_LAST_FULL : plan.num_full_visits; }
  int full_slot(int v) const noexcept { return kv_last_full ? FULL_INDEX_LAST_FULL[v] : plan.full_index[std::size_t(v)]; }
  std::size_t full_row_bytes() const { return kv_q4 ? KV_Q4_ROW_BYTES : std::size_t(KV_DIM) * (kv_fp8 ? 1 : 2); }

  // per-visit views
  Tensor k_ring_at(int ring) const;
  Tensor v_ring_at(int ring) const;
  std::int64_t* ring_pos_at(int ring) const;
  Tensor k_full_at(int full) const;
  Tensor v_full_at(int full) const;
  float* conv_k_at(int v) const { return ptr<float>(conv_k) + std::size_t(v) * 3 * KV_DIM; }
  float* conv_v_at(int v) const { return ptr<float>(conv_v) + std::size_t(v) * 3 * KV_DIM; }
  float* conv_attn_at(int v) const { return ptr<float>(conv_attn) + std::size_t(v) * 3 * HIDDEN; }
  float* conv_mlp_at(int v) const { return ptr<float>(conv_mlp) + std::size_t(v) * 3 * HIDDEN; }
};

// ----------------------------------------------------------------------------------------------- buffers

// fast.Buffers: fixed device buffers of one call width M (one set per captured CUDA graph).
//   B0 = embedding; B1 = hidden entering layer 6 (pass 0); B2[p] = hidden entering layer 12 in pass p;
//   E[p] = hidden after layer 17 in pass p (E[passes - 1] is also the block entering layer 18).
struct Buffers {
  int m = 0;
  int passes = PASSES;
  Tensor token;                         // int64 [m] input token ids (read by embed)
  Tensor B0, B1;                        // fp32 [m, H]
  std::array<Tensor, MAX_PASSES> B2, E; // fp32 [m, H] (0..passes-1 allocated)
  Tensor hid, part, y, y2;          // fp32 [m, H]
  Tensor xin;                       // bf16 [3, m, H]
  Tensor mixf;                      // fp32 [3, m, H]
  Tensor msgs;                      // fp32 [4, m, H]
  Tensor xm;                        // bf16 [1, m, H]
  Tensor mixm;                      // fp32 [1, m, H]
  Tensor proj_sliding, proj_full;   // fp32 [m, 8600], [m, 8664] (fast: proj_out keyed by row count)
  Tensor attn_out;                  // bf16 [m, 3072]
  Tensor idx;                       // int32 [m, 8]
  Tensor wts;                       // fp32 [m, 8]
  Tensor act;                       // bf16 [m, 8, I]
  Tensor xf;                        // bf16 [1, m, H]
  Tensor xf_weak;                   // bf16 [1, m, H] (allocated only when LoopCdMode::LogitsAdaptive)
  Tensor logits;                    // fp32 [m, UNPADDED_VOCAB]
  Tensor logits_weak;               // fp32 [m, UNPADDED_VOCAB] (allocated only when LoopCdMode::LogitsAdaptive)
  Tensor loopcd_reduce;             // fp32 [m * 384] (allocated only when LoopCdMode::LogitsAdaptive)
  Tensor taps;                      // bf16 [m, 4, H]: partial block entering TAP_VISITS + final normed hidden
  // convolution inputs of each row per visit, for rolling back rejected rows (history = true)
  Tensor hist_attn, hist_mlp;       // fp32 [48, m, H]
  Tensor hist_k, hist_v;            // fp32 [48, m, 1024]

  std::vector<DeviceBuffer> storage;

  // fast.Buffers.blocks: completed blocks visible to layer l's attention read in pass p
  std::vector<const float*> blocks(int l, int p) const;
};

}  // namespace ingine::model

// Dense decode kernels (the PyTorch extension's kernels ported to raw-pointer launchers; contracts in
// native/KERNELS.md, area `dense`). Every launcher runs on `stream`, never allocates or synchronises, reads token
// ids, positions and routing from device memory, and is CUDA-graph capturable. pdl = programmatic dependent
// launch: the kernel prefetches weights/constants, then waits for its predecessor before reading activations or
// state.
// M (tokens per call) is 1..8 (mix and embed: any M >= 1, one CTA per row). Shapes are row-major (Python
// order). Invalid arguments throw std::invalid_argument.
#pragma once

#include "core/act.h"
#include "kernels/common/persona_abi.h"
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::dense {

// A weight matrix [rows, K]: BF16 rows (scale == nullptr) or float8_e4m3fn rows with fp32 per-row scales.
struct Matrix {
  const void* w = nullptr;
  const float* scale = nullptr;
  int rows = 0;
  // Optional 16-row-block table (GGUF dense weights, Volta): block b's rows start at w + blk_off[b]; blk_soff[b] >= 0:
  // Q8_0 rows (int8 [16, K] + FP16 scales q8_scale[blk_soff[b] + r (K / 32) + k / 32]), -1: activation-type rows,
  // <= -2: int8 codes with FP32 scales a 16 columns (Q6_K rows) at ((const float*)(q8_scale + (-2 - blk_soff[b])))
  // [r (K / 16) + k / 16].
  const std::int64_t* blk_off = nullptr;
  const std::int32_t* blk_soff = nullptr;
  const std::uint16_t* q8_scale = nullptr;
};

// proj: multi-segment GEMV. xin bf16 [3, M, H]; segments int32 device [nseg, 3] = (row_begin, row_end,
// input index); out fp32 [M, R] = bf16(w[r] . xin[seg(r).input, m]), R = w.rows.
void proj(const ActT* xin, int M, Matrix w, const int* segments, int nseg, float* out, bool pdl,
          cudaStream_t stream);

// proj_conv: attention output projection (w [H, 3072]) + post-attention conv with state update.
// x bf16 [M, 3072]; conv_w fp32 [H, 4]; conv_state fp32 [3, H]; out fp32 [M, H]; conv_hist fp32 [M, H]
// (BF16-rounded conv inputs, for speculative rollback) or nullptr.
void proj_conv(const ActT* x, int M, Matrix w, const float* conv_w, float* conv_state, float* out,
               float* conv_hist, bool pdl, cudaStream_t stream);

// head: out[m, v] = 30 tanh(bf16(w_v . x[m]) / 30), x bf16 [M, H], out fp32 [M, V], V = w.rows.
void head(const ActT* x, int M, Matrix w, float* out, bool pdl, cudaStream_t stream);

// gemv: plain GEMV (drafter o_proj / down_proj, and TP2/TP4 sharded wo). x bf16 [M, K] (row stride K),
// out fp32 [M, R] = bf16(w[r] . x[m]), R = w.rows; K = 768 (TP4), 1536 (TP2), 3072 or 9216; same PDL contract as proj.
void gemv(const ActT* x, int M, int K, Matrix w, float* out, bool pdl, cudaStream_t stream);
// Standalone 4-tap static conv + state shift for M=1..8 FP32 inputs (after TP/EP AllReduce).
void static_conv_decode(const float* x_f32, int M, const float* conv_w, float* conv_state, float* out,
                        float* conv_hist, cudaStream_t stream);

// fp8_rows: bf16 [rows, K] -> float8_e4m3fn [rows, K] (q) + fp32 scale [rows] = amax / 448.
void fp8_rows(const ActT* w, int rows, int K, void* q, float* scale, cudaStream_t stream);

// dequant_rows: weights.dequant: out = bf16(float(q) * scale[row]) for FP8 rows [rows, K].
void dequant_rows(const void* q, const float* scale, int rows, int K, ActT* out, cudaStream_t stream);
// dequant_blocked: a block-table matrix (Matrix::blk_off) as activation-type rows [rows, K] (Q8_0: act(q * d)).
void dequant_blocked(const Matrix& w, int K, ActT* out, cudaStream_t stream);

// router: x bf16 [M, H]; w [128, H] fp32 (w_fp32) or bf16 holding BF16 values; bias fp32 [128];
// idx int32 [M, 8] (descending score), wts fp32 [M, 8] = 8 s / (sum s + 1e-20).
void router(const ActT* x, int M, const void* w, bool w_fp32, const float* bias, int* idx, float* wts,
            bool pdl, cudaStream_t stream);

// router_hooked: fused mech-interp / persona router path (ablate, slot-preserving force, route-replay,
// slot-permute, read-only routing tap) reading hook descriptors from fixed device pointer `hook`.
void router_hooked(const ActT* x, int M, const void* w, bool w_fp32, const float* bias, int* idx, float* wts,
                   const persona::RouterHookVisit* hook, const std::int64_t* pos_dev, bool pdl, cudaStream_t stream);

// mix: ablated residual update + attention residual over block states + RMS norm (KERNELS.md).
// Side copies: FP32 rows copied by the same launch (dst + m * dst_stride <- src + m * src_stride, `cols`
// floats; 16-byte aligned), so the decode step needs no separate memcpy node.
struct MixRowCopy {
  const float* src = nullptr;
  long long src_stride = 0;
  float* dst = nullptr;  // null: no copy
  long long dst_stride = 0;
  int cols = 0;
};
struct MixParams {
  const float* const* blocks = nullptr;  // device table of nblocks device pointers, each fp32 [M, H]
  int nblocks = 0;                       // 0..5
  const float* base = nullptr;           // [M, H] or null
  const float* write = nullptr;          // [M, H] or null
  const float* abl_r = nullptr;          // [H] or null (no ablation of `write`)
  const float* abl_d = nullptr;          // [H] detector or null: with it `write` is ablated only where write . abl_d
                                         // exceeds the threshold in abl_w
  const float* abl_w = nullptr;          // [4] (device, 16-byte aligned): weight, offset, floor, detector threshold;
                                         // required with abl_r
  float* partial_out = nullptr;          // [M, H] or null
  const float* queries = nullptr;        // [Q, H]; null only for the layer-0 path (no blocks, Q = 1)
  int Q = 1;                             // 1..4
  const float* norm_w = nullptr;         // [Q, H], required with out_bf16
  ActT* out_bf16 = nullptr;     // [Q, M, H] or null
  float* out_f32 = nullptr;              // [Q, M, H] or null
  ActT* new_bf16 = nullptr;     // bf16(new), row m at new_bf16 + m * new_bf16_stride, or null
  long long new_bf16_stride = 0;
  ActT* out_copy = nullptr;     // Q == 1: copy of out_bf16, row m at out_copy + m * out_copy_stride
  long long out_copy_stride = 0;
  MixRowCopy copies[2];
  int M = 1;
};
void mix(const MixParams& p, bool pdl, cudaStream_t stream);
// mix_pair: `p` (1..4 blocks, Q = 1 or 3) and the independent message mix `msg` (one block, Q = 4) of a
// message visit in one launch; same results as two mix calls. Neither may read what the other writes.
void mix_pair(const MixParams& p, const MixParams& msg, bool pdl, cudaStream_t stream);

// Fused depth-replace / depth-tap variants of mix and mix_pair (read hook descriptors from fixed device memory;
// pos_dev != nullptr reads *pos_dev + m for CUDA-graph decode, else uses pos0 + m for prefill).
void mix_hooked(const MixParams& p, const persona::DepthHookSite* depth_hook, const std::int64_t* pos_dev, int pos0,
                bool pdl, cudaStream_t stream);
void mix_pair_hooked(const MixParams& p, const persona::DepthHookSite* depth_p, const MixParams& msg,
                     const persona::DepthHookSite* depth_msg, const std::int64_t* pos_dev, int pos0, bool pdl,
                     cudaStream_t stream);

// Probe-only residual steering on FP32 write buffers [M, H]; skips cleanly when site->scale == 0.f or vec == null,
// and unconditionally leaves coordinate d_2027 untouched.
void residual_steer(float* y, int M, const persona::ResidualSteerSite* site, cudaStream_t stream);

// Read-only tap of the primary MoE operand xm (ActT [M, H]) into FP32 [max_pos, H] at tok_pos.
void tap_ffn_inp(const ActT* xm, int M, const persona::FfnInpTapVisit* tap, const std::int64_t* pos_dev, int pos0,
                 cudaStream_t stream);

// message_gates (loop passes 1-2): h, msg fp32 [G, M, H] (G <= 3; h_shared: every gate reads h[0], i.e. h is
// [M, H]); w_down bf16 [G, 64, 2H]; w_forget/w_gain [G, H, 64] fp32 (fg_fp32) or bf16; norm_w fp32 [G, H];
// out_bf16 [G, M, H]; out_f32 [G, M, H] (gated, before the norm) or null.
void message_gates(const float* h, bool h_shared, const float* msg, int G, int M, const ActT* w_down,
                   const void* w_forget, const void* w_gain, bool fg_fp32, const float* norm_w,
                   ActT* out_bf16, float* out_f32, bool pdl, cudaStream_t stream);

// embed: out[m] = float(bf16(rms_norm(float(table[token[m]])) * norm_w)), bit-exact with model.Model.embed.
// token int64 [M] (device), table bf16 [V, H], norm_w fp32 [H], out fp32 [M, H].
void embed(const std::int64_t* token, int M, const ActT* table, const float* norm_w, float* out,
           bool pdl, cudaStream_t stream);

// embed_q8: GGUF Q8_0 embedding rows u8 [V, 2448] ([FP16 d | 32 int8] x 72): act(q * d) values, embed's RMS norm.
void embed_q8(const std::int64_t* token, int M, const void* table_q8, const float* norm_w, float* out, bool pdl,
              cudaStream_t stream);
// embed_q4: Q4_0 SoA embedding table u8 [V, 1296] ([1152B nibbles | 72 FP16 scales]) -> identical RMS norm output.
// Guards out-of-range token IDs against [0, vocab_size) and sets *oob_flag = 1 if non-null.
void embed_q4(const std::int64_t* token, int M, int vocab_size, const void* table_q4, const float* norm_w, float* out,
              bool pdl, cudaStream_t stream, std::uint32_t* oob_flag = nullptr);

// decode_q4_0: Q4_0 SoA table u8 [rows, row_bytes] -> float[rows * hidden_size] via canonical fmaf lane decode.
void decode_q4_0(const void* table_q4, int rows, int hidden_size, float* out_f32, cudaStream_t stream);

}  // namespace ingine::kernels::dense

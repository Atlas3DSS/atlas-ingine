// Device-side POD ABI for mech-interp and persona hooks in atlas-ingine.
// Shared across dense, moe, prefill, and model targets so all hook descriptors live in one fixed-address
// device table (PersonaDeviceTable) that CUDA graphs read without recapture when switching profiles.
#pragma once

#include <cstdint>

namespace ingine::kernels::persona {

inline constexpr int kVisits = 48;
inline constexpr int kExperts = 128;
inline constexpr int kTopK = 8;
inline constexpr int kHidden = 2304;
inline constexpr int kInter = 1152;
inline constexpr int kForbiddenCoord = 2027;  // standing invariant: coordinate 2027 is never perturbed

// Router hook flags (bitmask in RouterHookVisit::flags).
inline constexpr std::uint32_t kRouterAblate = 1u << 0;
inline constexpr std::uint32_t kRouterForce = 1u << 1;
inline constexpr std::uint32_t kRouterSlotPermute = 1u << 2;
inline constexpr std::uint32_t kRouterReplayIds = 1u << 3;
inline constexpr std::uint32_t kRouterReplayWts = 1u << 4;  // copy weights verbatim (requires kRouterReplayIds)
inline constexpr std::uint32_t kRouterTap = 1u << 5;        // read-only routing capture

struct RouterHookVisit {
  std::uint32_t flags = 0;
  std::uint32_t visit = 0;
  std::uint64_t ablate_mask[2] = {0, 0};  // bit e set => expert e excluded from top-k
  std::int32_t force_expert = -1;
  float force_bias = 0.0f;
  float force_gain = 1.0f;
  std::int8_t slot_perm[kTopK] = {0, 1, 2, 3, 4, 5, 6, 7};
  // Position-addressed route replay (works identically in prefill chunks and decode steps).
  const std::int32_t* replay_ids = nullptr;  // [replay_n_pos, kTopK]
  const float* replay_wts = nullptr;         // [replay_n_pos, kTopK] or null (recompute from recipient unbiased s)
  std::int32_t replay_pos_start = 0;
  std::int32_t replay_n_pos = 0;
  // Optional read-only routing tap (position-addressed for this visit).
  std::int32_t* tap_idx = nullptr;     // [tap_max_pos, kTopK]
  float* tap_wts = nullptr;            // [tap_max_pos, kTopK]
  float* tap_logits = nullptr;         // [tap_max_pos, kExperts] or null
  std::int32_t tap_max_pos = 0;
};

// 3b Per-neuron post-activation affine hook: y = gain * y + offset.
// Identity (gain == 1.0f && offset == 0.0f) never writes dst -> exact 0.0 bit-identical control.
struct NeuronAffineEntry {
  std::int32_t global_idx = 0;
  std::int32_t expert = -1;
  std::int32_t neuron = -1;
  float gain = 1.0f;
  float offset = 0.0f;
};

struct NeuronAffineStat {
  std::uint32_t hits = 0;
  float pre_sum = 0.0f;
  float post_sum = 0.0f;
  std::int32_t last_pos = -1;
  float pre_last = 0.0f;
  float post_last = 0.0f;
};

// 3a Read-only expert post-activation hidden tap for one (visit, expert) unit.
struct ExpertTapTarget {
  std::int32_t expert = -1;
  std::int32_t tap_idx = 0;
  std::uint32_t per_token = 0;  // 0 = mean + last only; 1 = also append sparse per-token rows
};

struct ExpertTapVisit {
  const NeuronAffineEntry* neuron_entries = nullptr;
  NeuronAffineStat* neuron_stats = nullptr;
  std::int32_t n_neuron_entries = 0;
  const ExpertTapTarget* tap_targets = nullptr;
  std::int32_t n_tap_targets = 0;
  std::uint32_t* tap_hits = nullptr;     // [n_total_taps]
  std::int32_t* tap_last_pos = nullptr;  // [n_total_taps], initialized to -1
  float* tap_sum = nullptr;              // [n_total_taps, kInter]
  float* tap_last = nullptr;             // [n_total_taps, kInter]
  std::int32_t* tap_tok_count = nullptr; // [1] atomic counter
  std::int32_t* tap_tok_meta = nullptr;  // [tap_tok_cap, 3] = (tap_idx, pos, slot)
  float* tap_tok_act = nullptr;          // [tap_tok_cap, kInter]
  std::int32_t tap_tok_cap = 0;
};

// Depth-mix replace & tap site inside dense::mix_hooked / dense::mix_pair_hooked.
// mode: 0 = off, 1 = read (replace softmax scores), 2 = value (replace normalized states), 3 = both.
struct DepthHookSite {
  std::uint32_t mode = 0;
  std::uint32_t src_mask = 0;  // 0 = all sources; else bit i selects source i
  std::int32_t pos_start = 0;
  std::int32_t n_pos = 0;
  std::int32_t n_src = 0;
  std::int32_t n_q = 0;
  const float* ov_scores = nullptr;  // [n_pos, Q, NS]
  const float* ov_states = nullptr;  // [n_pos, NS, H] (RMS-normalized source vectors)
  // Optional read-only tap of exact fused FP32 scores and normalized states.
  std::uint32_t tap_enabled = 0;
  std::int32_t tap_max_pos = 0;
  float* tap_scores = nullptr;  // [tap_max_pos, Q, NS]
  float* tap_states = nullptr;  // [tap_max_pos, NS, H]
};

// Residual steering vector hook (probe/diagnostic only; vec[2027] == 0.0f enforced).
struct ResidualSteerSite {
  const float* vec = nullptr;  // [kHidden] FP32
  float scale = 0.0f;          // 0.0f => exact no-op skip
};

// Primary MoE operand (xm / ffn_moe_inp) read-only tap for parity checks and per-pass (n=1,2,3) observation
// accumulated on-device with zero per-token GPU->CPU syncs.
struct FfnInpTapVisit {
  std::uint32_t enabled = 0;
  std::int32_t max_pos = 0;
  float* out = nullptr;       // optional [max_pos, kHidden] FP32 per-token ring/sequence buffer
  float* last_out = nullptr;  // optional [kHidden] FP32 latest token buffer (O(1) device memory)
};

// Fixed-address device table read by hooked decode CUDA graphs and hooked prefill.
struct PersonaDeviceTable {
  std::uint32_t active = 0;
  RouterHookVisit router[kVisits];
  ExpertTapVisit expert[kVisits];
  DepthHookSite depth_attn[kVisits];
  DepthHookSite depth_ffn[kVisits];
  DepthHookSite depth_msg[kVisits];
  DepthHookSite depth_output;
  ResidualSteerSite steer_attn[kVisits];
  ResidualSteerSite steer_mlp[kVisits];
  FfnInpTapVisit ffn_inp_tap[kVisits];
};

}  // namespace ingine::kernels::persona

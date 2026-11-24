// Decode-side 3b per-neuron affine hook and 3a read-only expert post-activation tap.
// Runs on decode `act` (ActT [M, 8, 1152]) right between moe_up and moe_down ONLY in the hooked path.
// Fixed grid + device-side descriptor pointer -> 100% CUDA-graph capturable across arbitrary profile switches.
// Identity (gain == 1.0f && offset == 0.0f) never writes `act` -> exact 0.0 bit-identical control.
#include "kernels/moe/moe.h"

#include "core/device.h"

namespace ingine::kernels::moe {
namespace {

constexpr int I = persona::kInter;
constexpr int TOPK = persona::kTopK;
constexpr int kMaxNeuronBlocks = 64;
constexpr int kMaxTapBlocks = 32;

__global__ void decode_neuron_affine_kernel(ActT* __restrict__ act, const int* __restrict__ idx, int M,
                                            const persona::ExpertTapVisit* __restrict__ hook,
                                            const std::int64_t* __restrict__ pos_dev) {
  if (!hook || hook->n_neuron_entries <= 0 || !hook->neuron_entries) return;
  const int n_entries = hook->n_neuron_entries;
  const int base_pos = pos_dev ? int(pos_dev[0]) : 0;

  for (int ei = blockIdx.x; ei < n_entries; ei += gridDim.x) {
    if (threadIdx.x == 0) {
      const persona::NeuronAffineEntry ent = hook->neuron_entries[ei];
      if (ent.neuron >= 0 && ent.neuron < I && ent.expert >= 0 && ent.expert < persona::kExperts) {
        const bool do_modify = (ent.gain != 1.0f || ent.offset != 0.0f);
        std::uint32_t hits = 0;
        float pre_sum = 0.0f;
        float post_sum = 0.0f;
        int last_pos = -1;
        float pre_last = 0.0f;
        float post_last = 0.0f;

        for (int m = 0; m < M; ++m) {
          const int tok_pos = base_pos + m;
          const int* row_ids = idx + m * TOPK;
          for (int k = 0; k < TOPK; ++k) {
            if (row_ids[k] == ent.expert) {
              ActT* cell = act + (std::size_t(m) * TOPK + k) * I + ent.neuron;
              const float pre = act::to_float(*cell);
              float post = pre;
              if (do_modify) {
                if (ent.gain == 1.0f) {
                  post = __fadd_rn(pre, ent.offset);
                } else if (ent.offset == 0.0f) {
                  post = __fmul_rn(pre, ent.gain);
                } else {
                  post = __fadd_rn(__fmul_rn(pre, ent.gain), ent.offset);
                }
                const ActT q = act::from_float(post);
                *cell = q;
                post = act::to_float(q);
              }
              ++hits;
              pre_sum += pre;
              post_sum += post;
              if (tok_pos >= last_pos) {
                last_pos = tok_pos;
                pre_last = pre;
                post_last = post;
              }
            }
          }
        }
        if (hits > 0u && hook->neuron_stats) {
          persona::NeuronAffineStat& st = hook->neuron_stats[ent.global_idx];
          st.hits += hits;
          st.pre_sum += pre_sum;
          st.post_sum += post_sum;
          if (last_pos >= st.last_pos) {
            st.last_pos = last_pos;
            st.pre_last = pre_last;
            st.post_last = post_last;
          }
        }
      }
    }
  }
}

__global__ void decode_expert_tap_kernel(const ActT* __restrict__ act, const int* __restrict__ idx, int M,
                                         const persona::ExpertTapVisit* __restrict__ hook,
                                         const std::int64_t* __restrict__ pos_dev) {
  if (!hook || hook->n_tap_targets <= 0 || !hook->tap_targets || !hook->tap_hits || !hook->tap_sum || !hook->tap_last) {
    return;
  }
  const int n_targets = hook->n_tap_targets;
  const int base_pos = pos_dev ? int(pos_dev[0]) : 0;
  const int tid = threadIdx.x;
  const int bsz = blockDim.x;

  for (int ti = blockIdx.x; ti < n_targets; ti += gridDim.x) {
    const persona::ExpertTapTarget tg = hook->tap_targets[ti];
    if (tg.expert < 0 || tg.expert >= persona::kExperts || tg.tap_idx < 0) continue;

    std::uint32_t added_hits = 0;
    int best_pos = -1;
    int best_slot_linear = -1;

    // Small scan (M <= 8, TOPK = 8 => at most 64 slots) done identically by all threads so no broadcast needed.
    for (int m = 0; m < M; ++m) {
      const int tok_pos = base_pos + m;
      const int* row_ids = idx + m * TOPK;
      for (int k = 0; k < TOPK; ++k) {
        if (row_ids[k] == tg.expert) {
          const int linear = m * TOPK + k;
          const ActT* src = act + std::size_t(linear) * I;
          float* dst_sum = hook->tap_sum + std::size_t(tg.tap_idx) * I;
          for (int i = tid; i < I; i += bsz) {
            dst_sum[i] += act::to_float(src[i]);
          }
          if (tg.per_token && hook->tap_tok_count && hook->tap_tok_meta && hook->tap_tok_act) {
            __shared__ int tok_row;
            if (tid == 0) {
              const int r = atomicAdd(hook->tap_tok_count, 1);
              tok_row = r < hook->tap_tok_cap ? r : -1;
              if (tok_row >= 0) {
                hook->tap_tok_meta[tok_row * 3 + 0] = tg.tap_idx;
                hook->tap_tok_meta[tok_row * 3 + 1] = tok_pos;
                hook->tap_tok_meta[tok_row * 3 + 2] = k;
              }
            }
            __syncthreads();
            if (tok_row >= 0) {
              float* dst_tok = hook->tap_tok_act + std::size_t(tok_row) * I;
              for (int i = tid; i < I; i += bsz) dst_tok[i] = act::to_float(src[i]);
            }
            __syncthreads();
          }
          ++added_hits;
          if (tok_pos >= best_pos) {
            best_pos = tok_pos;
            best_slot_linear = linear;
          }
        }
      }
    }

    if (added_hits > 0u) {
      const int prev_last = hook->tap_last_pos ? hook->tap_last_pos[tg.tap_idx] : -1;
      if (best_pos >= prev_last && best_slot_linear >= 0) {
        const ActT* src = act + std::size_t(best_slot_linear) * I;
        float* dst_last = hook->tap_last + std::size_t(tg.tap_idx) * I;
        for (int i = tid; i < I; i += bsz) {
          dst_last[i] = act::to_float(src[i]);
        }
      }
      __syncthreads();
      if (tid == 0) {
        hook->tap_hits[tg.tap_idx] += added_hits;
        if (hook->tap_last_pos && best_pos >= prev_last) {
          hook->tap_last_pos[tg.tap_idx] = best_pos;
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace

void apply_expert_hooks_decode(ActT* act, const int* idx, int M, const persona::ExpertTapVisit* hook,
                               const std::int64_t* pos_dev, cudaStream_t stream) {
  if (!hook || M <= 0) return;
  decode_neuron_affine_kernel<<<kMaxNeuronBlocks, 32, 0, stream>>>(act, idx, M, hook, pos_dev);
  CUDA_CHECK(cudaGetLastError());
  decode_expert_tap_kernel<<<kMaxTapBlocks, 256, 0, stream>>>(act, idx, M, hook, pos_dev);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ingine::kernels::moe

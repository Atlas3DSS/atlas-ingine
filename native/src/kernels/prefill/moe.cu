// Prefill MoE routing (model._moe / _moe_grouped for T tokens): router selection, expert-sorted pair lists and
// tiles, and the deterministic combine. The grouped expert GEMMs are in wgemm.cu.
#include "core/device.h"
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

// Warp per token: sigmoid scores, 8 rounds of arg-max over s + bias (lowest expert on ties).
__global__ void topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int T,
                            int* __restrict__ idx, float* __restrict__ wts, int* __restrict__ counts) {
  const int t = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (t >= T) return;
  float sc[4], key[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e = lane + 32 * i;
    sc[i] = 1.f / (1.f + expf(-logits[size_t(t) * kExperts + e]));
    key[i] = sc[i] + bias[e];
  }
  int sel_e[kTopK];
  float sel_s[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float best = -INFINITY;
    int be = kExperts;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (key[i] > best) best = key[i], be = lane + 32 * i;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (ob > best || (ob == best && oe < be)) best = ob, be = oe;
    }
    sel_e[k] = be;
    float s_be = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (lane + 32 * i == be) s_be = sc[i], key[i] = -INFINITY;
    sel_s[k] = __shfl_sync(0xffffffffu, s_be, be & 31);
  }
  if (lane == 0) {
    float sum = sel_s[0];
#pragma unroll
    for (int k = 1; k < kTopK; ++k) sum += sel_s[k];
    const float den = sum + 1e-20f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      idx[size_t(t) * kTopK + k] = sel_e[k];
      wts[size_t(t) * kTopK + k] = sel_s[k] / den * 8.f;
      atomicAdd(counts + sel_e[k], 1);
    }
  }
}

// One CTA: expert offsets, tiles, scatter of the pairs into expert order (order within an expert does not
// affect any result: every pair row is computed independently).
__global__ void __launch_bounds__(1024) route_kernel(const int* __restrict__ idx, int T, int TR, MoeWork w) {
  __shared__ int offs[kExperts], tile0[kExperts + 1], cursor[kExperts];
  const int tid = threadIdx.x;
  if (tid == 0) {
    int o = 0, tl = 0;
    for (int e = 0; e < kExperts; ++e) {
      const int c = w.counts[e];
      offs[e] = o;
      tile0[e] = tl;
      o += c;
      tl += (c + TR - 1) / TR;
    }
    tile0[kExperts] = tl;
    *w.tiles_n = tl;
  }
  if (tid < kExperts) cursor[tid] = 0;
  __syncthreads();
  if (tid < kExperts) {
    const int c = w.counts[tid];
    for (int i = 0; i * TR < c; ++i) w.tiles[tile0[tid] + i] = make_int4(tid, offs[tid] + i * TR, min(TR, c - i * TR), 0);
  }
  for (int p = tid; p < T * kTopK; p += blockDim.x) {
    const int e = idx[p];
    const int pos = offs[e] + atomicAdd(&cursor[e], 1);
    w.pair_token[pos] = p / kTopK;
    w.pair_dst[pos] = p;
  }
}

// out[t] = bf16(sum_slot down[t, slot] * wts[t, slot]), 4 channels per thread.
__global__ void __launch_bounds__(kHidden / 4) combine_kernel(const uint2* __restrict__ down, const float* __restrict__ wts,
                                                            uint2* __restrict__ out) {
  const int t = blockIdx.x, c = threadIdx.x;
  float acc[4];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const uint2 d = down[(size_t(t) * kTopK + k) * (kHidden / 4) + c];
    const float wk = wts[size_t(t) * kTopK + k];
    const float p[4] = {__fmul_rn(bf_lo(d.x), wk), __fmul_rn(bf_hi(d.x), wk), __fmul_rn(bf_lo(d.y), wk),
                        __fmul_rn(bf_hi(d.y), wk)};
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = k == 0 ? p[i] : __fadd_rn(acc[i], p[i]);
  }
  out[size_t(t) * (kHidden / 4) + c] = make_uint2(pack_bf16x2(acc[0], acc[1]), pack_bf16x2(acc[2], acc[3]));
}

// Hooked warp-per-token top-k kernel (ablate, slot-preserving force, fused route-replay, slot-permute, routing tap).
// Keeps exact bit-identity (0 ULP) with topk_kernel for every no-op form.
__global__ void topk_hooked_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int T, int pos0,
                                   int* __restrict__ idx, float* __restrict__ wts, int* __restrict__ counts,
                                   const persona::RouterHookVisit* __restrict__ hook) {
  const int t = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (t >= T) return;
  const int tok_pos = pos0 + t;
  const std::uint32_t flags = hook ? hook->flags : 0u;

  float sc[4], key[4], raw_key[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e = lane + 32 * i;
    const float lg = logits[size_t(t) * kExperts + e];
    sc[i] = 1.f / (1.f + expf(-lg));
    raw_key[i] = sc[i] + bias[e];
    const bool ablated =
        ((flags & persona::kRouterAblate) != 0u) && (((hook->ablate_mask[e >> 6] >> (e & 63)) & 1ull) != 0ull);
    key[i] = ablated ? -INFINITY : raw_key[i];
    if ((flags & persona::kRouterTap) != 0u && hook->tap_logits && tok_pos >= 0 && tok_pos < hook->tap_max_pos) {
      hook->tap_logits[size_t(tok_pos) * kExperts + e] = lg;
    }
  }

  const bool do_replay = ((flags & persona::kRouterReplayIds) != 0u) && hook->replay_ids &&
                         tok_pos >= hook->replay_pos_start && tok_pos < hook->replay_pos_start + hook->replay_n_pos;
  const bool replay_copy = do_replay && ((flags & persona::kRouterReplayWts) != 0u) && hook->replay_wts != nullptr;

  int sel_e[kTopK];
  float sel_s[kTopK];
  float last_sel = -INFINITY;

  if (do_replay) {
    const size_t r_base = size_t(tok_pos - hook->replay_pos_start) * kTopK;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const int e = hook->replay_ids[r_base + k];
      sel_e[k] = e;
      if (replay_copy) {
        sel_s[k] = hook->replay_wts[r_base + k];
      } else {
        float s_e = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          if (lane + 32 * i == e) s_e = sc[i];
        sel_s[k] = __shfl_sync(0xffffffffu, s_e, e & 31);
      }
    }
  } else {
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      float best = -INFINITY;
      int be = kExperts;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (key[i] > best) best = key[i], be = lane + 32 * i;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oe = __shfl_xor_sync(0xffffffffu, be, o);
        if (ob > best || (ob == best && oe < be)) best = ob, be = oe;
      }
      sel_e[k] = be;
      if (k == kTopK - 1) last_sel = best;
      float s_be = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        if (lane + 32 * i == be) s_be = sc[i], key[i] = -INFINITY;
      sel_s[k] = __shfl_sync(0xffffffffu, s_be, be & 31);
    }

    if ((flags & persona::kRouterForce) != 0u) {
      const int fe = hook->force_expert;
      if (fe >= 0 && fe < kExperts) {
        float fe_s_lane = 0.f, fe_sel_lane = -INFINITY;
#pragma unroll
        for (int i = 0; i < 4; ++i)
          if (lane + 32 * i == fe) fe_s_lane = sc[i], fe_sel_lane = raw_key[i];
        const float fe_s = __shfl_sync(0xffffffffu, fe_s_lane, fe & 31);
        const float fe_sel = __shfl_sync(0xffffffffu, fe_sel_lane, fe & 31);
        if (lane == 0) {
          int force_slot = -1;
          for (int k = 0; k < kTopK; ++k) {
            if (sel_e[k] == fe) {
              force_slot = k;
              break;
            }
          }
          if (force_slot >= 0) {
            if (hook->force_gain != 1.0f) {
              sel_s[force_slot] *= hook->force_gain;
            }
          } else {
            const float boosted = fe_sel + hook->force_bias;
            const int last_e = sel_e[kTopK - 1];
            if (boosted > last_sel || (boosted == last_sel && fe < last_e)) {
              sel_e[kTopK - 1] = fe;
              sel_s[kTopK - 1] = (hook->force_gain == 1.0f) ? fe_s : (fe_s * hook->force_gain);
            }
          }
        }
      }
    }
  }

  if (lane == 0) {
    float out_w[kTopK];
    if (replay_copy) {
#pragma unroll
      for (int k = 0; k < kTopK; ++k) out_w[k] = sel_s[k];
    } else {
      float sum = sel_s[0];
#pragma unroll
      for (int k = 1; k < kTopK; ++k) sum += sel_s[k];
      const float den = sum + 1e-20f;
#pragma unroll
      for (int k = 0; k < kTopK; ++k) out_w[k] = sel_s[k] / den * 8.f;
    }
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      const int dst_k = ((flags & persona::kRouterSlotPermute) != 0u) ? int(hook->slot_perm[k]) : k;
      idx[size_t(t) * kTopK + dst_k] = sel_e[k];
      wts[size_t(t) * kTopK + dst_k] = out_w[k];
      atomicAdd(counts + sel_e[k], 1);
      if ((flags & persona::kRouterTap) != 0u && tok_pos >= 0 && tok_pos < hook->tap_max_pos) {
        if (hook->tap_idx) hook->tap_idx[size_t(tok_pos) * kTopK + dst_k] = sel_e[k];
        if (hook->tap_wts) hook->tap_wts[size_t(tok_pos) * kTopK + dst_k] = out_w[k];
      }
    }
  }
}

// Prefill 3b per-neuron affine hook over expert-grouped sorted pairs (`act [T * 8, kInter]`).
// Identity (gain == 1.0f && offset == 0.0f) never writes `act` -> exact 0.0 bit-identical control.
__global__ void prefill_neuron_affine_kernel(ActT* __restrict__ act, MoeWork w, int pos0,
                                             const persona::ExpertTapVisit* __restrict__ hook) {
  if (!hook || hook->n_neuron_entries <= 0 || !hook->neuron_entries) return;
  const int n_entries = hook->n_neuron_entries;
  const int n_tiles = *w.tiles_n;
  const int tid = threadIdx.x;
  const int bsz = blockDim.x;

  __shared__ std::uint32_t sh_hits[256];
  __shared__ float sh_pre_sum[256];
  __shared__ float sh_post_sum[256];
  __shared__ int sh_last_pos[256];
  __shared__ float sh_pre_last[256];
  __shared__ float sh_post_last[256];

  for (int ei = blockIdx.x; ei < n_entries; ei += gridDim.x) {
    const persona::NeuronAffineEntry ent = hook->neuron_entries[ei];
    std::uint32_t my_hits = 0;
    float my_pre_sum = 0.0f;
    float my_post_sum = 0.0f;
    int my_last_pos = -1;
    float my_pre_last = 0.0f;
    float my_post_last = 0.0f;

    if (ent.neuron >= 0 && ent.neuron < kInter && ent.expert >= 0 && ent.expert < kExperts) {
      const bool do_modify = (ent.gain != 1.0f || ent.offset != 0.0f);
      for (int ti = 0; ti < n_tiles; ++ti) {
        const int4 tile = w.tiles[ti];
        if (tile.x != ent.expert) continue;
        const int pb = tile.y, n = tile.z;
        for (int r = tid; r < n; r += bsz) {
          const int pair = pb + r;
          const int tok_pos = pos0 + w.pair_token[pair];
          ActT* cell = act + size_t(pair) * kInter + ent.neuron;
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
          ++my_hits;
          my_pre_sum += pre;
          my_post_sum += post;
          if (tok_pos >= my_last_pos) {
            my_last_pos = tok_pos;
            my_pre_last = pre;
            my_post_last = post;
          }
        }
      }
    }

    sh_hits[tid] = my_hits;
    sh_pre_sum[tid] = my_pre_sum;
    sh_post_sum[tid] = my_post_sum;
    sh_last_pos[tid] = my_last_pos;
    sh_pre_last[tid] = my_pre_last;
    sh_post_last[tid] = my_post_last;
    __syncthreads();

    for (int stride = bsz >> 1; stride > 0; stride >>= 1) {
      if (tid < stride) {
        sh_hits[tid] += sh_hits[tid + stride];
        sh_pre_sum[tid] += sh_pre_sum[tid + stride];
        sh_post_sum[tid] += sh_post_sum[tid + stride];
        if (sh_last_pos[tid + stride] > sh_last_pos[tid]) {
          sh_last_pos[tid] = sh_last_pos[tid + stride];
          sh_pre_last[tid] = sh_pre_last[tid + stride];
          sh_post_last[tid] = sh_post_last[tid + stride];
        }
      }
      __syncthreads();
    }

    if (tid == 0 && sh_hits[0] > 0u && hook->neuron_stats) {
      persona::NeuronAffineStat& st = hook->neuron_stats[ent.global_idx];
      st.hits += sh_hits[0];
      st.pre_sum += sh_pre_sum[0];
      st.post_sum += sh_post_sum[0];
      if (sh_last_pos[0] >= st.last_pos) {
        st.last_pos = sh_last_pos[0];
        st.pre_last = sh_pre_last[0];
        st.post_last = sh_post_last[0];
      }
    }
    __syncthreads();
  }
}

// Prefill 3a read-only expert activation tap over sorted-pair `act [T * 8, kInter]`.
// Runs immediately after prefill_neuron_affine_kernel on the same stream so 3a observes 3b edits.
__global__ void prefill_expert_tap_kernel(const ActT* __restrict__ act, MoeWork w, int pos0,
                                          const persona::ExpertTapVisit* __restrict__ hook) {
  if (!hook || hook->n_tap_targets <= 0 || !hook->tap_targets || !hook->tap_hits || !hook->tap_sum || !hook->tap_last) {
    return;
  }
  const int n_targets = hook->n_tap_targets;
  const int n_tiles = *w.tiles_n;
  const int tid = threadIdx.x;
  const int bsz = blockDim.x;

  __shared__ std::uint32_t s_hits;
  __shared__ int s_best_pos;
  __shared__ int s_best_pair;
  __shared__ int s_tok_row;

  for (int ti = blockIdx.x; ti < n_targets; ti += gridDim.x) {
    const persona::ExpertTapTarget tg = hook->tap_targets[ti];
    if (tg.expert < 0 || tg.expert >= kExperts || tg.tap_idx < 0) continue;

    if (tid == 0) {
      std::uint32_t hits = 0;
      int best_pos = -1;
      int best_pair = -1;
      for (int k = 0; k < n_tiles; ++k) {
        const int4 tile = w.tiles[k];
        if (tile.x != tg.expert) continue;
        const int pb = tile.y, n = tile.z;
        hits += std::uint32_t(n);
        for (int r = 0; r < n; ++r) {
          const int pair = pb + r;
          const int tp = pos0 + w.pair_token[pair];
          if (tp >= best_pos) {
            best_pos = tp;
            best_pair = pair;
          }
        }
      }
      s_hits = hits;
      s_best_pos = best_pos;
      s_best_pair = best_pair;
    }
    __syncthreads();

    if (s_hits > 0u) {
      float* dst_sum = hook->tap_sum + size_t(tg.tap_idx) * kInter;
      for (int k = 0; k < n_tiles; ++k) {
        const int4 tile = w.tiles[k];
        if (tile.x != tg.expert) continue;
        const int pb = tile.y, n = tile.z;
        for (int r = 0; r < n; ++r) {
          const int pair = pb + r;
          const ActT* src = act + size_t(pair) * kInter;
          for (int i = tid; i < kInter; i += bsz) {
            dst_sum[i] += act::to_float(src[i]);
          }
          if (tg.per_token && hook->tap_tok_count && hook->tap_tok_meta && hook->tap_tok_act) {
            if (tid == 0) {
              const int row = atomicAdd(hook->tap_tok_count, 1);
              s_tok_row = row < hook->tap_tok_cap ? row : -1;
              if (s_tok_row >= 0) {
                hook->tap_tok_meta[s_tok_row * 3 + 0] = tg.tap_idx;
                hook->tap_tok_meta[s_tok_row * 3 + 1] = pos0 + w.pair_token[pair];
                hook->tap_tok_meta[s_tok_row * 3 + 2] = w.pair_dst[pair] % kTopK;
              }
            }
            __syncthreads();
            if (s_tok_row >= 0) {
              float* dst_tok = hook->tap_tok_act + size_t(s_tok_row) * kInter;
              for (int i = tid; i < kInter; i += bsz) dst_tok[i] = act::to_float(src[i]);
            }
            __syncthreads();
          }
        }
      }
      const int prev_last = hook->tap_last_pos ? hook->tap_last_pos[tg.tap_idx] : -1;
      if (s_best_pos >= prev_last && s_best_pair >= 0) {
        const ActT* src = act + size_t(s_best_pair) * kInter;
        float* dst_last = hook->tap_last + size_t(tg.tap_idx) * kInter;
        for (int i = tid; i < kInter; i += bsz) {
          dst_last[i] = act::to_float(src[i]);
        }
      }
      __syncthreads();
      if (tid == 0) {
        hook->tap_hits[tg.tap_idx] += s_hits;
        if (hook->tap_last_pos && s_best_pos >= prev_last) {
          hook->tap_last_pos[tg.tap_idx] = s_best_pos;
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace

int moe_tile_rows(int T) {
  // Turing (64 KiB opt-in) and Pascal (48 KiB) use the <=48 KiB TM=32 wgemm path for all T.
  if (current_max_smem_optin() < 90000) return 32;
  static const int forced = [] {
    const char* e = std::getenv("INGINE_MOE_TILE_ROWS");
    const int v = e ? std::atoi(e) : 0;
    return v == 32 || v == 128 ? v : 0;
  }();
  return forced ? forced : T <= 512 ? 32 : 128;
}

int moe_tiles_capacity(int max_T) { return std::max(moe_max_tiles(std::min(max_T, 512)), moe_max_tiles(max_T)); }

void moe_topk(const float* logits, const float* bias, int T, int* idx, float* wts, int* counts, cudaStream_t stream) {
  topk_kernel<<<(T + 7) / 8, 256, 0, stream>>>(logits, bias, T, idx, wts, counts);
}

void moe_topk_hooked(const float* logits, const float* bias, int T, int pos0, int* idx, float* wts, int* counts,
                     const persona::RouterHookVisit* hook, cudaStream_t stream) {
  topk_hooked_kernel<<<(T + 7) / 8, 256, 0, stream>>>(logits, bias, T, pos0, idx, wts, counts, hook);
  CUDA_CHECK(cudaGetLastError());
}

void apply_expert_hooks_prefill(ActT* act, const MoeWork& w, int T, int pos0, const persona::ExpertTapVisit* hook,
                                cudaStream_t stream) {
  if (!hook || T <= 0) return;
  prefill_neuron_affine_kernel<<<64, 256, 0, stream>>>(act, w, pos0, hook);
  CUDA_CHECK(cudaGetLastError());
  prefill_expert_tap_kernel<<<32, 256, 0, stream>>>(act, w, pos0, hook);
  CUDA_CHECK(cudaGetLastError());
}

void moe_route(const int* idx, int T, const MoeWork& w, cudaStream_t stream) {
  route_kernel<<<1, 1024, 0, stream>>>(idx, T, moe_tile_rows(T), w);
}

void moe_combine(const ActT* down, const float* wts, int T, ActT* out, cudaStream_t stream) {
  combine_kernel<<<T, kHidden / 4, 0, stream>>>(reinterpret_cast<const uint2*>(down), wts,
                                                reinterpret_cast<uint2*>(out));
}

}  // namespace ingine::kernels::prefill

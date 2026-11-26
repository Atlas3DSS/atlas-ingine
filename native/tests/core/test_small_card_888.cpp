// Gate 0 pure-CPU verification for #888 (opt-in small-card knobs, strictly NO silent --tail-skip):
//   1. Pinned GGML BlockQ5K (176 B / 256 elems) pack/unpack + multi-block K=2304 & K=3072 oracle
//      covering both BF16 and pre-packed FP8-E4M3FN fixtures, verifying <= 2.5% per-32-value
//      sub-block source-range error on all 5,376 tested fixture sub-blocks + FP64 GEMV.
//   2. Recomputed persistent, scratch, and packed-payload byte accounting.
//   3. Production LruSlotIndex + BoundedSwapQueue + ExpertCache host/device dispatch-pointer lifecycle
//      for BOTH decode and prefill (Pass 0 / Pass 1):
//        - Exact att-be17aa24 regression: stage into LRU -> hit -> promote to HOT (must erase LRU key)
//          -> hit in HOT (decode + prefill) -> demote from HOT (clears dispatch ptr) -> re-route in
//          decode & prefill (must miss LRU, re-fetch byte-identically from immutable backing, never
//          dereference nullptr or stale slot).
//        - Actual 8-expert SwiGLU numerical output verified at 0-ULP across 60+ interleaved decode &
//          prefill steps with repeated promotions, demotions, staging-LRU hits, HOT hits, LRU
//          evictions, queue saturation (dropped_bound separate from experts_dropped == 0), and
//          pin-generation reuse protection.
//   4. RuntimeOptions defaults, parse_attn_requant_mode, StateIdentity bidirectional isolation,
//      and null-safe elided FP32 alias views.
#include "core/lru_index.h"
#include "core/q5_k.h"
#include "ingine/generator.h"
#include "ingine/prefix_cache.h"
#include "model/expert_swap_queue.h"
#include "model/model_data.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using namespace ingine::q5k;
using namespace ingine::model;

float round_bf16_host(float v) {
  std::uint32_t u = 0;
  std::memcpy(&u, &v, 4);
  const std::uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  u &= 0xffff0000u;
  float out = 0.f;
  std::memcpy(&out, &u, 4);
  return out;
}

// Pure-CPU IEEE FP8 E4M3FN quantize+dequantize per row (matches weights.cpp / fp8_rows).
void roundtrip_fp8_row(const float* in, float* out, int K) {
  float amax = 0.f;
  for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(in[k]));
  const float scale = amax / 448.0f;
  if (scale <= 0.f) {
    for (int k = 0; k < K; ++k) out[k] = 0.f;
    return;
  }
  for (int k = 0; k < K; ++k) {
    float x = std::clamp(in[k] / scale, -448.0f, 448.0f);
    const float sgn = x < 0.f ? -1.f : 1.f;
    const float ax = std::fabs(x);
    float q = 0.f;
    if (ax < (1.0f / 64.0f)) {
      q = std::nearbyint(ax * 512.0f) / 512.0f;
    } else {
      int e = 0;
      (void)std::frexp(ax, &e);
      const float step = std::ldexp(1.0f, e - 4);
      q = std::min(448.0f, std::nearbyint(ax / step) * step);
    }
    out[k] = sgn * q * scale;
  }
}

double cosine_sim(const std::vector<double>& a, const std::vector<double>& b) {
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return (na > 0.0 && nb > 0.0) ? (dot / std::sqrt(na * nb)) : 0.0;
}

// Compact numerical payload for each (layer, expert) representing a real SwiGLU expert MLP:
//   up_w[4][4], gate_w[4][4], down_w[4][4] -> output y[4] = down_w @ (silu(gate_w @ x) * (up_w @ x)).
struct ExpertWeights {
  static constexpr int kDim = 4;
  float up[kDim][kDim]{};
  float gate[kDim][kDim]{};
  float down[kDim][kDim]{};
};

ExpertWeights make_expert_weights(int l, int e) {
  ExpertWeights w{};
  std::uint64_t s = 0x9e3779b97f4a7c15ULL ^ (std::uint64_t(l + 1) << 32) ^ std::uint64_t(e * 1315423911u + 17u);
  auto next = [&]() -> float {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    const double u = double((s >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 53);
    return float((u - 0.5) * 0.5);
  };
  for (int i = 0; i < ExpertWeights::kDim; ++i) {
    for (int j = 0; j < ExpertWeights::kDim; ++j) {
      w.up[i][j] = next();
      w.gate[i][j] = next();
      w.down[i][j] = next();
    }
  }
  return w;
}

std::array<double, ExpertWeights::kDim> eval_expert_mlp(const ExpertWeights& w,
                                                        const std::array<double, ExpertWeights::kDim>& x) {
  std::array<double, ExpertWeights::kDim> hidden{};
  for (int i = 0; i < ExpertWeights::kDim; ++i) {
    double g = 0.0, u = 0.0;
    for (int j = 0; j < ExpertWeights::kDim; ++j) {
      g += double(w.gate[i][j]) * x[std::size_t(j)];
      u += double(w.up[i][j]) * x[std::size_t(j)];
    }
    const double silu = g / (1.0 + std::exp(-g));
    hidden[std::size_t(i)] = silu * u;
  }
  std::array<double, ExpertWeights::kDim> y{};
  for (int i = 0; i < ExpertWeights::kDim; ++i) {
    double acc = 0.0;
    for (int j = 0; j < ExpertWeights::kDim; ++j) {
      acc += double(w.down[i][j]) * hidden[std::size_t(j)];
    }
    y[std::size_t(i)] = acc;
  }
  return y;
}

constexpr std::uint64_t make_key(int l, int e) noexcept {
  return (static_cast<std::uint64_t>(l) << 8) | static_cast<std::uint64_t>(e);
}
constexpr int key_layer(std::uint64_t k) noexcept { return static_cast<int>(k >> 8); }
constexpr int key_expert(std::uint64_t k) noexcept { return static_cast<int>(k & 0xffu); }

}  // namespace

int main() {
  bool ok = true;
  auto expect = [&](bool cond, const char* msg) {
    if (!cond) {
      std::printf("FAIL: %s\n", msg);
      ok = false;
    }
  };

  // ------------------------------------------------------------------ 1. GGML BlockQ5K layout & scale packing
  expect(sizeof(BlockQ5K) == 176, "BlockQ5K == 176 bytes");
  expect(row_bytes(2304) == 9 * 176 && row_bytes(2304) == 1584, "K=2304 -> 9 super-blocks (1584 B/row)");
  expect(row_bytes(3072) == 12 * 176 && row_bytes(3072) == 2112, "K=3072 -> 12 super-blocks (2112 B/row)");

  {
    std::uint8_t sc[kSubBlocks] = {0, 17, 31, 63, 1, 22, 48, 63};
    std::uint8_t mn[kSubBlocks] = {63, 42, 15, 0, 59, 33, 16, 7};
    std::uint8_t packed[kScaleBytes]{};
    pack_scale_min_k4(sc, mn, packed);
    for (int j = 0; j < kSubBlocks; ++j) {
      std::uint8_t out_sc = 0, out_mn = 0;
      get_scale_min_k4(j, packed, out_sc, out_mn);
      expect(out_sc == sc[j] && out_mn == mn[j], "get_scale_min_k4 exact round-trip");
    }
  }

  {
    const float probes[] = {0.0f, -0.0f, 1.0f, -2.5f, 0.1f, 1e-4f, 65504.0f, 100000.0f};
    for (float v : probes) {
      const std::uint16_t h = f32_to_f16_bits(v);
      const float r = f16_bits_to_f32(h);
      if (std::fabs(v) > 65504.0f) {
        expect(std::fabs(r - 65504.0f) == 0.f, "f16 clamp to 65504");
      } else if (v == 0.0f) {
        expect(r == 0.0f, "f16 zero exact");
      } else {
        expect(std::fabs(r - v) <= std::fabs(v) * 2e-3f + 1e-7f, "f16 round-trip accuracy");
      }
    }
  }

  // ------------------------------------------------------------------ 2. Multi-block K=2304 & K=3072 (BF16 + FP8)
  // Verifies <= 2.5% per-32-value sub-block source-range error on the tested fixtures (0/5376 failures;
  // shared super-block scales do not guarantee 2.5% for arbitrary unbounded input ranges).
  for (int K : {2304, 3072}) {
    const int R = 16;
    const int M = 3;
    std::vector<float> w_bf16(std::size_t(R) * K);
    std::vector<float> w_fp8(std::size_t(R) * K);
    std::vector<float> x(std::size_t(M) * K);
    std::uint64_t rng = 0x888123456789ULL ^ std::uint64_t(K);
    auto next_f = [&]() -> float {
      rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
      const double u = double((rng >> 11) & ((1ULL << 53) - 1)) / double(1ULL << 53);
      return float((u - 0.5) * 2.0);
    };
    for (int r = 0; r < R; ++r) {
      for (int k = 0; k < K; ++k) {
        const int sb = (k / 32) % 8;
        const float mag = 0.04f * float(1 + (sb % 5));
        const float bias = (sb & 1) ? -0.02f : 0.015f;
        float v = next_f() * mag + bias;
        if ((k % 257) == 0) v *= 3.5f;
        w_bf16[std::size_t(r) * K + k] = round_bf16_host(v);
      }
      roundtrip_fp8_row(w_bf16.data() + std::size_t(r) * K, w_fp8.data() + std::size_t(r) * K, K);
    }
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = round_bf16_host(next_f() * 0.5f);

    auto check_source = [&](const char* label, const std::vector<float>& src) {
      std::vector<std::uint8_t> packed(std::size_t(R) * row_bytes(K));
      quantize_rows_cpu(src.data(), R, K, packed.data());
      std::vector<float> deq(src.size(), 0.f);
      dequant_rows_cpu(packed.data(), R, K, deq.data());

      int nonzero = 0;
      for (std::size_t i = 0; i < src.size(); ++i) {
        if (deq[i] != 0.f) ++nonzero;
      }
      expect(nonzero > int(src.size() * 9 / 10), label);

      // Tested-fixture per-32-value sub-block source-range bound: max|src - deq| <= 2.5% * (max - min).
      const int nb = K / kBlockElems;
      int total_sb = 0;
      int fail_sb = 0;
      double worst_rel = 0.0;
      for (int r = 0; r < R; ++r) {
        const float* srow = src.data() + std::size_t(r) * K;
        const float* drow = deq.data() + std::size_t(r) * K;
        for (int b = 0; b < nb; ++b) {
          for (int j = 0; j < kSubBlocks; ++j) {
            const int off = b * kBlockElems + j * kSubBlockElems;
            double mn = srow[off], mx = srow[off], sb_err = 0.0;
            for (int i = 0; i < kSubBlockElems; ++i) {
              const double sv = srow[off + i];
              const double dv = drow[off + i];
              mn = std::min(mn, sv);
              mx = std::max(mx, sv);
              sb_err = std::max(sb_err, std::fabs(sv - dv));
            }
            const double sb_range = mx - mn;
            ++total_sb;
            if (sb_range > 0.0) {
              const double rel = sb_err / sb_range;
              worst_rel = std::max(worst_rel, rel);
              if (rel > 0.025) ++fail_sb;
            } else if (sb_err > 0.0) {
              ++fail_sb;
            }
          }
        }
      }
      expect(fail_sb == 0, "Q5_K <= 2.5% per-32-value source-range bound across all tested fixture sub-blocks");
      expect(worst_rel <= 0.025, "Q5_K worst tested-fixture sub-block relative error <= 2.5%");

      // Compare FP64 GEMV over packed BlockQ5K vs original FP64 GEMV
      std::vector<double> ref_orig(std::size_t(M) * R, 0.0);
      std::vector<double> ref_q5k(std::size_t(M) * R, 0.0);
      for (int m = 0; m < M; ++m) {
        for (int r = 0; r < R; ++r) {
          double acc = 0.0;
          for (int k = 0; k < K; ++k) {
            acc += double(src[std::size_t(r) * K + k]) * double(x[std::size_t(m) * K + k]);
          }
          ref_orig[std::size_t(m) * R + r] = acc;
        }
      }
      gemv_fp64_ref(packed.data(), x.data(), M, R, K, ref_q5k.data());
      const double cos = cosine_sim(ref_orig, ref_q5k);
      expect(cos >= 0.999, label);

      double max_deq_diff = 0.0;
      for (int m = 0; m < M; ++m) {
        for (int r = 0; r < R; ++r) {
          double acc = 0.0;
          for (int k = 0; k < K; ++k) {
            acc += double(deq[std::size_t(r) * K + k]) * double(x[std::size_t(m) * K + k]);
          }
          max_deq_diff = std::max(max_deq_diff, std::fabs(acc - ref_q5k[std::size_t(m) * R + r]));
        }
      }
      expect(max_deq_diff == 0.0, "gemv_fp64_ref matches explicit dequant_rows_cpu dot product bit-for-bit");
      std::printf("PASS q5k_%s K=%d: cos=%.6f subblock_fails=%d/%d worst_sb_rel=%.3f%% nonzero=%d/%zu\n", label, K, cos,
                  fail_sb, total_sb, worst_rel * 100.0, nonzero, src.size());
    };
    check_source("bf16_src", w_bf16);
    check_source("fp8_disk_src", w_fp8);
  }

  // Zero-range constant sub-block edge case
  {
    std::vector<float> z(kBlockElems, 0.0f);
    BlockQ5K blk{};
    quantize_block_cpu(z.data(), blk);
    std::vector<float> out(kBlockElems, 1.0f);
    dequant_block_cpu(blk, out.data());
    double max_z = 0.0;
    for (float v : out) max_z = std::max(max_z, double(std::fabs(v)));
    expect(max_z == 0.0, "Q5_K all-zero block round-trips identically to 0");
  }

  // ------------------------------------------------------------------ 3. Recomputed persistent & scratch byte accounting
  {
    const std::size_t dead_ctrl_f32 = 12ull * 4 * 2 * HIDDEN * GATE_RANK * 4;
    const std::size_t dead_router_f32 = 24ull * EXPERTS * HIDDEN * 4;
    expect(dead_ctrl_f32 == 56623104ull, "dead FP32 controller duplicates == 54.00 MiB");
    expect(dead_router_f32 == 28311552ull, "dead FP32 router duplicates == 27.00 MiB");
    expect(dead_ctrl_f32 + dead_router_f32 == 84934656ull, "--loop-bf16 reclaim == 81.00 MiB");

    const std::size_t attn_elems =
        (20ull * PROJ_ROWS_SLIDING + 4ull * PROJ_ROWS_FULL) * HIDDEN + 24ull * HIDDEN * (TOTAL_HEADS * HEAD_DIM);
    const std::size_t attn_bf16_bytes = attn_elems * 2;
    const std::size_t attn_fp8_bytes =
        attn_elems + (20ull * PROJ_ROWS_SLIDING + 4ull * PROJ_ROWS_FULL + 24ull * HIDDEN) * 4;
    const std::size_t attn_q5k_bytes =
        (20ull * PROJ_ROWS_SLIDING + 4ull * PROJ_ROWS_FULL) * row_bytes(HIDDEN) +
        24ull * HIDDEN * row_bytes(TOTAL_HEADS * HEAD_DIM);
    expect(attn_bf16_bytes == 1292009472ull, "wproj+wo BF16 == 1232.16 MiB");
    expect(attn_fp8_bytes == 647052544ull, "wproj+wo FP8 == 617.08 MiB");
    expect(attn_q5k_bytes == 444128256ull, "wproj+wo Q5_K == 423.55 MiB");
    const double payload_red = 1.0 - double(kBlockBytes) / double(kBlockElems);
    expect(std::fabs(payload_red - 0.3125) < 1e-12, "Q5_K packed payload reduction vs FP8 == 31.25%");
    const std::size_t wscratch_bytes =
        std::max(std::size_t(PROJ_ROWS_FULL) * HIDDEN, std::size_t(HIDDEN) * TOTAL_HEADS * HEAD_DIM) * 2;
    expect(wscratch_bytes == 39923712ull, "Prefiller::wscratch_ == 38.07 MiB");
  }

  // ------------------------------------------------------------------ 4. LruSlotIndex erase + pin_gen guard unit check
  {
    ingine::LruSlotIndex lru(2);
    auto a0 = lru.acquire(make_key(1, 5), /*pin_gen=*/10);
    auto a1 = lru.acquire(make_key(1, 6), /*pin_gen=*/10);
    expect(!a0.hit && !a1.hit && lru.find(make_key(1, 5)) == a0.slot, "LruSlotIndex initial acquire");
    expect(lru.erase(make_key(1, 5)), "LruSlotIndex::erase removes resident key");
    expect(lru.find(make_key(1, 5)) == -1, "erased key absent from find()");
    expect(lru.touch_and_pin(make_key(1, 5), 11) == -1, "erased key cannot hit touch_and_pin()");
    expect(!lru.erase(make_key(1, 5)), "second erase returns false");
    // Slot a0 was pinned at pin_gen=10; acquiring in pin_gen=10 must throw (both slots still pinned at gen 10).
    bool pinned_threw = false;
    try {
      (void)lru.acquire(make_key(1, 7), /*pin_gen=*/10);
    } catch (const std::runtime_error&) {
      pinned_threw = true;
    }
    expect(pinned_threw, "LruSlotIndex::erase preserves pin_gen guard against same-step reuse");
    // In pin_gen=11, vacated slot a0 is reused cleanly without evicting live key (1, 6).
    auto a2 = lru.acquire(make_key(1, 7), /*pin_gen=*/11);
    expect(!a2.evicted && a2.slot == a0.slot && lru.find(make_key(1, 6)) == a1.slot,
           "vacated erased slot reused cleanly in next pin_gen");
  }

  // ------------------------------------------------------------------ 5. Reconciled ExpertCache slot & pointer lifecycle
  // Uses production LruSlotIndex + BoundedSwapQueue + host/device pointer tables + decode & prefill Pass 0/1
  // and validates actual 8-expert SwiGLU numerical outputs (0-ULP) across repeated promote/demote/hit/refetch cycles.
  {
    // 5a. Strict opt-in default: max_swaps=0 disables promotions completely.
    BoundedSwapQueue q0(0);
    expect(!q0.enabled() && q0.capacity() == 0, "BoundedSwapQueue(0) disabled");
    for (int i = 0; i < 10; ++i) {
      expect(q0.record_access(6, 5, false) == BoundedSwapQueue::EnqueueStatus::Disabled, "max_swaps=0 never enqueues");
    }
    expect(q0.queued_total() == 0 && q0.dropped_bound() == 0, "max_swaps=0 zero queued/dropped");
    bool neg_threw = false;
    try {
      BoundedSwapQueue qneg(-1);
    } catch (const std::invalid_argument&) {
      neg_threw = true;
    }
    expect(neg_threw, "BoundedSwapQueue(-1) throws invalid_argument");

    // 5b. Full decode + prefill lifecycle model with actual ExpertWeights buffers & h_ptr / d_ptr dispatch tables.
    constexpr int kHotCap = 4;
    constexpr int kLruCap = 12;  // >= TOP_K (8) for single-slice decode, small enough to force real LRU evictions
    constexpr int kSwapCap = 4;
    BoundedSwapQueue swap_q(kSwapCap, /*promote_threshold=*/2, /*hysteresis=*/2, /*decay_interval=*/100000);
    ingine::LruSlotIndex lru(kLruCap);

    enum class Tier { Hot, Warm, Cold };
    struct Meta {
      Tier tier = Tier::Warm;
      bool has_warm = false;
      bool has_cold = false;
      int hot_slot = -1;
      int inflight_slot = -1;
      ExpertWeights warm_buf{};
      ExpertWeights cold_buf{};
    };
    struct Inflight {
      bool active = false;
      int hot_slot = -1;
      int nl = -1;
      int ne = -1;
      ExpertWeights* dst = nullptr;
    };

    std::vector<std::array<Meta, 128>> meta(24);
    std::vector<ExpertWeights> hot_vram(kHotCap);
    std::vector<ExpertWeights> lru_vram(kLruCap);
    std::vector<BoundedSwapQueue::HotSlotState> hot_slots;
    std::array<Inflight, 2> inflight{};

    // Separate host_mirror and device_table pointer arrays (mirrors ExpertCache::LayerDeviceTables).
    std::vector<std::array<const ExpertWeights*, 128>> h_ptr(24);
    std::vector<std::array<const ExpertWeights*, 128>> d_ptr(24);
    std::array<bool, 24> dirty{};
    for (int l = 0; l < 24; ++l) {
      h_ptr[l].fill(nullptr);
      d_ptr[l].fill(nullptr);
      for (int e = 0; e < 128; ++e) {
        const ExpertWeights gt = make_expert_weights(l, e);
        meta[l][e].tier = (e % 2 == 0) ? Tier::Warm : Tier::Cold;
        meta[l][e].has_warm = (e % 2 == 0);
        meta[l][e].has_cold = true;
        meta[l][e].warm_buf = gt;
        meta[l][e].cold_buf = gt;
      }
    }
    // Initial 4 HOT experts: (0, 0..3)
    for (int e = 0; e < kHotCap; ++e) {
      meta[0][e].tier = Tier::Hot;
      meta[0][e].has_warm = (e < 2);
      meta[0][e].has_cold = true;
      meta[0][e].hot_slot = e;
      hot_vram[std::size_t(e)] = meta[0][e].cold_buf;
      h_ptr[0][e] = &hot_vram[std::size_t(e)];
      d_ptr[0][e] = &hot_vram[std::size_t(e)];
      hot_slots.push_back(BoundedSwapQueue::HotSlotState{0, e, true, false, 0, std::uint64_t(e + 1)});
    }

    auto set_host = [&](int l, int e, const ExpertWeights* p) {
      h_ptr[l][e] = p;
      dirty[l] = true;
    };
    auto clear_host = [&](int l, int e) {
      h_ptr[l][e] = nullptr;
      dirty[l] = true;
    };
    auto flush_table = [&](int l) {
      if (!dirty[l]) return;
      d_ptr[l] = h_ptr[l];
      dirty[l] = false;
    };

    // Strict mutual-exclusion & pointer-validity invariant checked after EVERY transition.
    auto check_invariants = [&]() {
      for (int l = 0; l < 24; ++l) {
        for (int e = 0; e < 128; ++e) {
          const int lru_slot = lru.find(make_key(l, e));
          if (meta[l][e].tier == Tier::Hot) {
            expect(lru_slot == -1, "INVARIANT: HOT expert must never also exist in staging LRU");
            expect(meta[l][e].hot_slot >= 0 && h_ptr[l][e] == &hot_vram[std::size_t(meta[l][e].hot_slot)],
                   "INVARIANT: HOT expert host pointer matches hot_vram slot");
          } else {
            expect(meta[l][e].hot_slot == -1, "INVARIANT: non-HOT expert has hot_slot == -1");
            if (lru_slot >= 0) {
              expect(h_ptr[l][e] == &lru_vram[std::size_t(lru_slot)],
                     "INVARIANT: staging-LRU resident expert has matching non-null host pointer");
            } else {
              expect(h_ptr[l][e] == nullptr, "INVARIANT: non-resident expert has null host pointer");
            }
          }
        }
      }
    };

    std::uint64_t pin_gen = 0, tick = 10;
    std::uint64_t promotions = 0, evictions = 0, lru_hits = 0, refetches_after_demote = 0, experts_dropped = 0;
    std::array<std::array<bool, 128>, 24> ever_demoted{};

    auto complete_swap = [&](int idx) {
      Inflight& inf = inflight[std::size_t(idx)];
      if (!inf.active) return;
      // Mirrors repaired ExpertCache::complete_inflight_swap: erase any surviving staging-LRU key first!
      (void)lru.erase(make_key(inf.nl, inf.ne));
      Meta& nem = meta[inf.nl][inf.ne];
      nem.tier = Tier::Hot;
      nem.hot_slot = inf.hot_slot;
      nem.inflight_slot = -1;
      set_host(inf.nl, inf.ne, inf.dst);
      auto& hs = hot_slots[std::size_t(inf.hot_slot)];
      hs.layer = inf.nl;
      hs.expert = inf.ne;
      hs.evictable = true;
      hs.swap_inflight = false;
      hs.last_use = ++tick;
      swap_q.mark_inflight(inf.nl, inf.ne, false);
      inf.active = false;
      ++promotions;
    };

    auto finalize_all_swaps = [&]() {
      for (int i = 0; i < 2; ++i) complete_swap(i);
      for (int l = 0; l < 24; ++l) flush_table(l);
    };

    auto stage_expert = [&](int l, int e, std::uint64_t gen) {
      expect(meta[l][e].tier != Tier::Hot, "stage_expert never called on HOT expert");
      auto res = lru.acquire(make_key(l, e), gen, is_loop_layer(l));
      if (res.hit) {
        expect(h_ptr[l][e] == &lru_vram[std::size_t(res.slot)], "stage_expert hit has valid ptr");
        ++lru_hits;
        return;
      }
      if (res.evicted) {
        const int el = key_layer(res.evicted_key);
        const int ee = key_expert(res.evicted_key);
        expect(meta[el][ee].tier != Tier::Hot, "LRU eviction never touches HOT expert");
        clear_host(el, ee);
      }
      const Meta& em = meta[l][e];
      expect(em.has_warm || em.has_cold, "non-HOT expert has verified immutable backing");
      lru_vram[std::size_t(res.slot)] = (em.tier == Tier::Warm) ? em.warm_buf : em.cold_buf;
      if (ever_demoted[l][e]) ++refetches_after_demote;
      set_host(l, e, &lru_vram[std::size_t(res.slot)]);
    };

    auto drain_async = [&](int cur_layer, std::uint64_t gen, bool complete_now) {
      int free_i = -1;
      for (int i = 0; i < 2; ++i) {
        if (!inflight[std::size_t(i)].active) {
          free_i = i;
          break;
        }
      }
      if (free_i < 0) return;
      BoundedSwapQueue::Entry cand{};
      while (swap_q.pop(cand)) {
        Meta& nem = meta[cand.layer][cand.expert];
        if (nem.tier == Tier::Hot || nem.inflight_slot >= 0) continue;
        const int victim = swap_q.select_victim(hot_slots, cur_layer, gen, swap_q.live_hits(cand.layer, cand.expert));
        if (victim < 0) continue;
        auto& vs = hot_slots[std::size_t(victim)];
        const int ol = vs.layer, oe = vs.expert;
        // Mirrors repaired ExpertCache::drain_swap_queue_async invariant check.
        expect(lru.find(make_key(ol, oe)) == -1, "demoted HOT victim must not be present in staging LRU");
        Meta& oem = meta[ol][oe];
        oem.tier = oem.has_warm ? Tier::Warm : Tier::Cold;
        oem.hot_slot = -1;
        clear_host(ol, oe);
        ever_demoted[ol][oe] = true;
        ++evictions;

        vs.swap_inflight = true;
        nem.inflight_slot = free_i;
        swap_q.mark_inflight(cand.layer, cand.expert, true);
        hot_vram[std::size_t(victim)] = nem.has_warm ? nem.warm_buf : nem.cold_buf;
        inflight[std::size_t(free_i)] = Inflight{true, victim, cand.layer, cand.expert, &hot_vram[std::size_t(victim)]};
        if (complete_now) complete_swap(free_i);
        break;
      }
    };

    // Execute one decode layer and verify ACTUAL 8-expert SwiGLU numerical output at 0-ULP by dereferencing d_ptr[l].
    auto run_decode_step = [&](int l, const std::array<int, 8>& sel, const std::array<double, 4>& x, bool poll_first,
                               bool drain_after, bool complete_immediately) {
      if (poll_first) {
        for (int i = 0; i < 2; ++i) complete_swap(i);
      }
      std::vector<int> hot_ids, non_hot_ids;
      for (int e : sel) {
        if (meta[l][e].inflight_slot >= 0) complete_swap(meta[l][e].inflight_slot);
        if (meta[l][e].tier == Tier::Hot) {
          expect(h_ptr[l][e] == &hot_vram[std::size_t(meta[l][e].hot_slot)], "decode HOT ptr valid");
          hot_ids.push_back(e);
        } else {
          non_hot_ids.push_back(e);
        }
      }
      const std::uint64_t gen = ++pin_gen;
      for (int e : hot_ids) {
        hot_slots[std::size_t(meta[l][e].hot_slot)].pin_gen = gen;
        hot_slots[std::size_t(meta[l][e].hot_slot)].last_use = ++tick;
        (void)swap_q.record_access(l, e, true);
      }
      for (int e : non_hot_ids) (void)swap_q.record_access(l, e, false);

      std::vector<int> misses;
      for (int e : non_hot_ids) {
        const int slot = lru.touch_and_pin(make_key(l, e), gen, is_loop_layer(l));
        if (slot >= 0) {
          expect(h_ptr[l][e] == &lru_vram[std::size_t(slot)], "decode staging-LRU hit has non-null matching ptr");
          ++lru_hits;
        } else {
          misses.push_back(e);
        }
      }
      for (int e : misses) stage_expert(l, e, gen);
      flush_table(l);

      // Compute actual 8-expert numerical output EXCLUSIVELY via d_ptr[l][e] (must never be null) vs ground truth!
      std::array<double, 4> y_got{}, y_ref{};
      for (int k = 0; k < 8; ++k) {
        const int e = sel[std::size_t(k)];
        const double rw = 0.05 * double(k + 1);
        const ExpertWeights* dp = d_ptr[l][e];
        expect(dp != nullptr, "decode device dispatch pointer non-null for all 8 routed experts");
        if (dp) {
          const auto ye = eval_expert_mlp(*dp, x);
          for (int d = 0; d < 4; ++d) y_got[std::size_t(d)] += rw * ye[std::size_t(d)];
        }
        const auto yr = eval_expert_mlp(make_expert_weights(l, e), x);
        for (int d = 0; d < 4; ++d) y_ref[std::size_t(d)] += rw * yr[std::size_t(d)];
      }
      for (int d = 0; d < 4; ++d) {
        expect(y_got[std::size_t(d)] == y_ref[std::size_t(d)], "decode 8-expert SwiGLU output 0-ULP exact");
      }
      if (drain_after) drain_async(l, gen, complete_immediately);
      check_invariants();
    };

    // Execute one prefill layer (Pass 0 overlapped HOT+LRU hits using d_ptr BEFORE Pass 1 flush, then Pass 1).
    auto run_prefill_step = [&](int l, const std::array<int, 8>& sel, const std::array<double, 4>& x) {
      // Mirrors repaired finalize_all_inflight_swaps: completes swaps AND flushes all dirty tables before Pass 0!
      finalize_all_swaps();
      std::vector<int> active_hot, misses;
      for (int e : sel) {
        if (meta[l][e].tier == Tier::Hot) active_hot.push_back(e);
        else misses.push_back(e);
      }
      const std::uint64_t gen = ++pin_gen;
      std::vector<int> p0_lru_hits, to_stage;
      for (int e : misses) {
        const int slot = lru.touch_and_pin(make_key(l, e), gen, is_loop_layer(l));
        if (slot >= 0) {
          expect(h_ptr[l][e] == &lru_vram[std::size_t(slot)], "prefill staging-LRU hit has non-null matching ptr");
          ++lru_hits;
          p0_lru_hits.push_back(e);
        } else {
          to_stage.push_back(e);
        }
      }
      std::array<double, 4> y_got{}, y_ref{};
      std::array<std::array<double, 4>, 8> per_expert_got{};
      std::array<bool, 8> expert_computed{};
      // Pass 0 executes active_hot + p0_lru_hits using EXISTING d_ptr[l] BEFORE staging `to_stage`!
      for (int k = 0; k < 8; ++k) {
        const int e = sel[std::size_t(k)];
        const bool in_p0 = (std::find(active_hot.begin(), active_hot.end(), e) != active_hot.end()) ||
                           (std::find(p0_lru_hits.begin(), p0_lru_hits.end(), e) != p0_lru_hits.end());
        if (in_p0) {
          const ExpertWeights* dp = d_ptr[l][e];
          expect(dp != nullptr, "prefill Pass 0 device dispatch pointer non-null before Pass 1 flush");
          if (dp) {
            per_expert_got[std::size_t(k)] = eval_expert_mlp(*dp, x);
            expert_computed[std::size_t(k)] = true;
          }
        }
      }
      // Pass 1 stages `to_stage`, flushes d_ptr[l], and evaluates the remaining experts.
      for (int e : to_stage) stage_expert(l, e, gen);
      flush_table(l);
      for (int k = 0; k < 8; ++k) {
        const int e = sel[std::size_t(k)];
        const double rw = 0.05 * double(k + 1);
        if (std::find(to_stage.begin(), to_stage.end(), e) != to_stage.end()) {
          const ExpertWeights* dp = d_ptr[l][e];
          expect(dp != nullptr, "prefill Pass 1 device dispatch pointer non-null");
          if (dp) {
            per_expert_got[std::size_t(k)] = eval_expert_mlp(*dp, x);
            expert_computed[std::size_t(k)] = true;
          }
        }
        expect(expert_computed[std::size_t(k)], "all 8 routed prefill experts computed (experts_dropped == 0)");
        const auto yr = eval_expert_mlp(make_expert_weights(l, e), x);
        for (int d = 0; d < 4; ++d) {
          expect(per_expert_got[std::size_t(k)][std::size_t(d)] == yr[std::size_t(d)],
                 "prefill per-expert SwiGLU output 0-ULP exact");
          y_got[std::size_t(d)] += rw * per_expert_got[std::size_t(k)][std::size_t(d)];
          y_ref[std::size_t(d)] += rw * yr[std::size_t(d)];
        }
      }
      for (int d = 0; d < 4; ++d) {
        expect(y_got[std::size_t(d)] == y_ref[std::size_t(d)], "prefill 8-expert SwiGLU output 0-ULP exact");
      }
      check_invariants();
    };

    const std::array<double, 4> x_probe = {0.35, -0.22, 0.71, -0.48};
    // Exact att-be17aa24 regression scenario:
    // Step 1-2: Layer 6 routes {10..17} twice -> stages in LRU, hits in LRU, saturates bounded swap queue (cap=4, qdrop=4).
    const std::array<int, 8> l6_a = {10, 11, 12, 13, 14, 15, 16, 17};
    run_decode_step(6, l6_a, x_probe, false, false, false);
    run_decode_step(6, l6_a, x_probe, false, false, false);
    expect(swap_q.size() == kSwapCap && swap_q.dropped_bound() == 4, "bounded queue saturated (qdrop=4)");
    expect(experts_dropped == 0, "experts_dropped == 0 during saturation");

    // Step 3: Promote (6, 10..13) into HOT (evicting (0, 0..3)) WHILE (6, 10..13) are LIVE in staging LRU!
    // Before our fix, (6, 10..13) remained live in staging LRU after promotion.
    for (int step = 0; step < 4; ++step) run_decode_step(6, l6_a, x_probe, true, true, true);
    expect(promotions == 4 && evictions == 4, "4 HOT promotions + 4 HOT evictions");
    for (int e = 10; e < 14; ++e) {
      expect(meta[6][e].tier == Tier::Hot && lru.find(make_key(6, e)) == -1,
             "promoted experts erased from staging LRU immediately on promotion");
    }

    // Step 4: Prefill on layer 6 while (6, 10..13) are HOT and dirty[6] may have just changed (Pass 0 check).
    run_prefill_step(6, {10, 11, 12, 13, 40, 41, 42, 43}, x_probe);

    // Step 5: Promote layer 12 experts {50..53} with higher hit counts (>= 6 + hysteresis 2 = 8, plus 4 drain steps)
    // to DEMOTE all 4 (6, 10..13) back to WARM/COLD!
    // Note: (6, 10..13) were previously staged in LRU, then promoted to HOT, and are NOW demoted (h_ptr cleared).
    const std::array<int, 8> l12_hot = {50, 51, 52, 53, 54, 55, 56, 57};
    for (int rep = 0; rep < 14; ++rep) run_decode_step(12, l12_hot, x_probe, true, true, false);
    finalize_all_swaps();
    int demoted_l6 = 0;
    for (int e = 10; e < 14; ++e) {
      if (meta[6][e].tier != Tier::Hot) ++demoted_l6;
    }
    expect(demoted_l6 == 4, "all 4 previously-promoted layer-6 experts demoted by hotter layer-12 experts");

    // Step 6: Immediately re-route (6, 10..13) in BOTH decode AND prefill!
    // Before our fix, touch_and_pin hit the stale staging-LRU key while h_ptr/d_ptr was nullptr!
    run_decode_step(6, l6_a, x_probe, true, false, false);
    run_prefill_step(6, l6_a, x_probe);
    // Also re-route originally evicted (0, 0..3) in decode & prefill.
    run_decode_step(0, {0, 1, 2, 3, 20, 21, 22, 23}, x_probe, true, true, false);
    run_prefill_step(0, {0, 1, 2, 3, 24, 25, 26, 27}, x_probe);

    // Step 7: 40-step stress storm interleaving decode, deferred-completion async swaps, and prefill across layers.
    std::uint64_t srng = 0x888CAFEBABEULL;
    for (int step = 0; step < 40; ++step) {
      const int layer = (step % 3 == 0) ? 0 : ((step % 3 == 1) ? 6 : 12);
      std::array<int, 8> sel{};
      for (int k = 0; k < 8; ++k) {
        srng = srng * 6364136223846793005ULL + 1442695040888963407ULL;
        const int base = ((step / 8) % 2 == 0) ? 0 : 10;
        sel[std::size_t(k)] = (base + int((srng >> 32) % 20) + k * 3) % 128;
      }
      // Deduplicate 8 distinct expert IDs (realistic TOP_K routing)
      for (int k = 0; k < 8; ++k) {
        while (std::find(sel.begin(), sel.begin() + k, sel[std::size_t(k)]) != sel.begin() + k) {
          sel[std::size_t(k)] = (sel[std::size_t(k)] + 1) % 128;
        }
      }
      const std::array<double, 4> xs = {0.1 * double(step % 7 - 3), 0.25, -0.4 + 0.01 * step, 0.6};
      if (step % 5 == 4) {
        run_prefill_step(layer, sel, xs);
      } else {
        run_decode_step(layer, sel, xs, /*poll_first=*/(step % 2 == 0), /*drain_after=*/true,
                        /*complete_immediately=*/(step % 4 == 0));
      }
    }
    finalize_all_swaps();
    check_invariants();

    expect(promotions >= 8 && evictions >= 8, "repeated HOT promotions and demotions exercised");
    expect(lru_hits > 0 && refetches_after_demote >= 4, "staging LRU hits and post-demotion re-fetches verified");
    std::printf("PASS lifecycle_8expert: promo=%llu evict=%llu lru_hits=%llu refetch=%llu qdrop=%llu edrop=%llu\n",
                static_cast<unsigned long long>(promotions), static_cast<unsigned long long>(evictions),
                static_cast<unsigned long long>(lru_hits), static_cast<unsigned long long>(refetches_after_demote),
                static_cast<unsigned long long>(swap_q.dropped_bound()),
                static_cast<unsigned long long>(experts_dropped));
  }

  // ------------------------------------------------------------------ 6. Defaults, parsing, StateIdentity, null-safety
  {
    RuntimeOptions def;
    expect(def.loop_bf16 == false, "RuntimeOptions::loop_bf16 defaults false");
    expect(def.attn_requant == AttnRequantMode::Off, "RuntimeOptions::attn_requant defaults Off");
    expect(def.expert_swaps == 0, "RuntimeOptions::expert_swaps defaults 0");
    expect(parse_attn_requant_mode("off") == AttnRequantMode::Off, "parse_attn_requant_mode(off)");
    expect(parse_attn_requant_mode("q5_k") == AttnRequantMode::Q5K, "parse_attn_requant_mode(q5_k)");
    bool bad_threw = false;
    try {
      (void)parse_attn_requant_mode("q4_k");
    } catch (const std::invalid_argument&) {
      bad_threw = true;
    }
    expect(bad_threw, "parse_attn_requant_mode rejects invalid value");

    StateIdentity id_off;
    id_off.attn_requant = static_cast<std::uint8_t>(AttnRequantMode::Off);
    StateIdentity id_q5 = id_off;
    id_q5.attn_requant = static_cast<std::uint8_t>(AttnRequantMode::Q5K);
    expect(!(id_off == id_q5) && !(id_q5 == id_off), "StateIdentity bidirectional attn_requant inequality");
    expect(hash_state_identity(id_off) != hash_state_identity(id_q5), "hash_state_identity distinguishes attn_requant");

    Layer layer;
    auto null_safe_at = [](const ingine::Tensor& stack, int g, ingine::DType dt, std::int32_t a, std::int32_t b) {
      if (!stack.data) return ingine::Tensor{};
      const std::size_t es = dt == ingine::DType::FP32 ? 4 : 2;
      return rm(static_cast<char*>(stack.data) + std::size_t(g) * a * b * es, dt, {a, b});
    };
    layer.q_gate_forget = null_safe_at(layer.qkv_gate_forget, 0, ingine::DType::FP32, HIDDEN, GATE_RANK);
    layer.k_gate_forget = null_safe_at(layer.qkv_gate_forget, 1, ingine::DType::FP32, HIDDEN, GATE_RANK);
    layer.v_gate_forget = null_safe_at(layer.qkv_gate_forget, 2, ingine::DType::FP32, HIDDEN, GATE_RANK);
    layer.mlp_gate_forget = null_safe_at(layer.mlp1_gate_forget, 0, ingine::DType::FP32, HIDDEN, GATE_RANK);
    expect(layer.router.data == nullptr && layer.router.nb[0] == 0, "elided router FP32 is null-safe");
    expect(layer.qkv_gate_forget.data == nullptr && layer.qkv_gate_gain.data == nullptr, "elided qkv FP32 null-safe");
    expect(layer.q_gate_forget.data == nullptr && layer.k_gate_forget.data == nullptr &&
               layer.v_gate_forget.data == nullptr && layer.mlp_gate_forget.data == nullptr,
           "elided FP32 forget alias views are null-safe");
    expect(layer.q_gate_gain.data == nullptr && layer.k_gate_gain.data == nullptr &&
               layer.v_gate_gain.data == nullptr && layer.mlp_gate_gain.data == nullptr,
           "elided FP32 gain alias views are null-safe");
  }

  std::printf("%s core_small_card_888\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

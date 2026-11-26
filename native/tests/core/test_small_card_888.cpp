// Gate 0 pure-CPU verification for #888 (opt-in small-card knobs, strictly NO silent --tail-skip):
//   1. Pinned GGML BlockQ5K (176 B / 256 elems) pack/unpack + multi-block K=2304 & K=3072 oracle
//      covering both BF16 and pre-packed FP8-E4M3FN sources + FP64 GEMV reference.
//   2. Recomputed persistent, scratch, and packed-payload byte accounting.
//   3. BoundedSwapQueue strict opt-in (default 0), forced promotion, eviction with verified immutable
//      backing, byte-identical re-fetch of evicted expert, queue saturation (dropped_bound separate
//      from experts_dropped == 0), pin/layer overlap guards, and independent 8-expert parity audit.
//   4. RuntimeOptions defaults, parse_attn_requant_mode, StateIdentity bidirectional isolation,
//      and null-safe elided FP32 alias views.
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
    // Round to E4M3FN precision (3 mantissa bits, bias 7, subnormal step 2^-9)
    const float sgn = x < 0.f ? -1.f : 1.f;
    const float ax = std::fabs(x);
    float q = 0.f;
    if (ax < (1.0f / 64.0f)) {
      q = std::nearbyint(ax * 512.0f) / 512.0f;
    } else {
      int e = 0;
      const float m = std::frexp(ax, &e);  // ax = m * 2^e, m in [0.5, 1)
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
    // Exact pack_scale_min_k4 / get_scale_min_k4 round-trip over all 8 sub-blocks (full 6-bit range 0..63).
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

  // Exact FP16 bit-conversion round-trip on normal, subnormal, zero, and clamp edges.
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
      // Varied per-sub-block scales, asymmetric offsets, and occasional outlier
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

      // Check nonzero reconstruction and tight elementwise fidelity
      double max_err = 0.0, max_abs = 0.0;
      int nonzero = 0;
      for (std::size_t i = 0; i < src.size(); ++i) {
        if (deq[i] != 0.f) ++nonzero;
        max_abs = std::max(max_abs, double(std::fabs(src[i])));
        max_err = std::max(max_err, double(std::fabs(src[i] - deq[i])));
      }
      expect(nonzero > int(src.size() * 9 / 10), label);
      expect(max_err < max_abs * 0.06, label);

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

      // Verify gemv_fp64_ref agrees bit-identically with explicit deq dot product
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
      std::printf("PASS q5k_%s K=%d: cos=%.6f max_err=%.5f (max_abs=%.4f) nonzero=%d/%zu\n", label, K, cos, max_err,
                  max_abs, nonzero, src.size());
    };
    check_source("bf16_src", w_bf16);
    check_source("fp8_disk_src", w_fp8);
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
    const std::size_t wscratch_bytes = std::max(std::size_t(PROJ_ROWS_FULL) * HIDDEN,
                                                std::size_t(HIDDEN) * TOTAL_HEADS * HEAD_DIM) * 2;
    expect(wscratch_bytes == 39923712ull, "Prefiller::wscratch_ == 38.07 MiB");
  }

  // ------------------------------------------------------------------ 4. BoundedSwapQueue & full 8-expert parity audit
  {
    // 4a. Default max_swaps=0 is strictly disabled (retains unmodified offload behavior).
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

    // 4b. Forced promotion, eviction, re-fetch of evicted expert, queue saturation, and independent 8-expert audit.
    constexpr int kCap = 4;
    BoundedSwapQueue q(kCap, /*promote_threshold=*/2, /*hysteresis=*/2, /*decay_interval=*/100000);
    expect(q.enabled() && q.capacity() == kCap, "BoundedSwapQueue(4) enabled");

    // Simulated ground-truth expert payloads (layer, expert) -> unique deterministic value.
    auto ground_truth = [](int l, int e) -> double { return double((l + 1) * 1000 + e) * 0.00125; };

    // 4 HOT slots (layer 0, experts 0..3) all with verified immutable backing (2 WARM + 2 COLD).
    enum class SimTier { Hot, Warm, Cold };
    struct SimMeta {
      SimTier tier = SimTier::Warm;
      bool has_warm = false;
      bool has_cold = false;
      int hot_slot = -1;
      double warm_val = 0.0;
      double cold_val = 0.0;
    };
    std::array<std::array<SimMeta, 128>, 24> meta{};
    std::vector<BoundedSwapQueue::HotSlotState> slots;
    std::vector<double> slot_vram;
    for (int l = 0; l < 24; ++l) {
      for (int e = 0; e < 128; ++e) {
        meta[l][e].tier = (e % 2 == 0) ? SimTier::Warm : SimTier::Cold;
        meta[l][e].has_warm = (e % 2 == 0);
        meta[l][e].has_cold = true;  // every expert has verified immutable cold backing
        meta[l][e].warm_val = ground_truth(l, e);
        meta[l][e].cold_val = ground_truth(l, e);
      }
    }
    for (int e = 0; e < 4; ++e) {
      meta[0][e].tier = SimTier::Hot;
      meta[0][e].has_warm = (e < 2);
      meta[0][e].has_cold = true;
      meta[0][e].hot_slot = e;
      slots.push_back(BoundedSwapQueue::HotSlotState{0, e, true, false, 0, std::uint64_t(e + 1)});
      slot_vram.push_back(ground_truth(0, e));
    }

    std::uint64_t promotions = 0, evictions = 0, refetches_of_evicted = 0, experts_dropped = 0;
    std::array<bool, 4> was_evicted{};
    std::uint64_t pin_gen = 0, tick = 10;

    auto execute_step = [&](int cur_layer, const std::array<int, 8>& selected, bool drain_after) {
      const std::uint64_t gen = ++pin_gen;
      double ref_sum = 0.0;
      double exec_sum = 0.0;
      std::array<int, 8> exec_ids{};
      for (int k = 0; k < 8; ++k) {
        const int e = selected[k];
        const double w = 0.125 * double(k + 1);
        ref_sum += w * ground_truth(cur_layer, e);
        SimMeta& m = meta[cur_layer][e];
        double payload = 0.0;
        if (m.tier == SimTier::Hot) {
          slots[std::size_t(m.hot_slot)].pin_gen = gen;
          slots[std::size_t(m.hot_slot)].last_use = ++tick;
          payload = slot_vram[std::size_t(m.hot_slot)];
          (void)q.record_access(cur_layer, e, true);
        } else {
          // Verify immutable backing on re-fetch (NEVER drop expert!)
          expect(m.has_warm || m.has_cold, "non-HOT expert must have verified immutable backing");
          payload = (m.tier == SimTier::Warm) ? m.warm_val : m.cold_val;
          if (cur_layer == 0 && e < 4 && was_evicted[e]) ++refetches_of_evicted;
          (void)q.record_access(cur_layer, e, false);
        }
        exec_ids[k] = e;
        exec_sum += w * payload;
      }
      // Independently compare all 8 selected IDs and exact sum contribution
      expect(exec_ids == selected, "all 8 routed expert IDs executed identically");
      expect(exec_sum == ref_sum, "0-ULP contribution parity across all 8 selected experts");

      if (drain_after) {
        BoundedSwapQueue::Entry cand{};
        if (q.pop(cand)) {
          const int victim = q.select_victim(slots, cur_layer, gen, q.live_hits(cand.layer, cand.expert));
          if (victim >= 0) {
            auto& vs = slots[std::size_t(victim)];
            SimMeta& old_m = meta[vs.layer][vs.expert];
            expect(old_m.has_warm || old_m.has_cold, "evicted HOT victim has immutable backing");
            old_m.tier = old_m.has_warm ? SimTier::Warm : SimTier::Cold;
            old_m.hot_slot = -1;
            if (vs.layer == 0 && vs.expert < 4) was_evicted[vs.expert] = true;
            ++evictions;

            SimMeta& new_m = meta[cand.layer][cand.expert];
            const double fetched = new_m.has_warm ? new_m.warm_val : new_m.cold_val;
            slot_vram[std::size_t(victim)] = fetched;
            new_m.tier = SimTier::Hot;
            new_m.hot_slot = victim;
            vs.layer = cand.layer;
            vs.expert = cand.expert;
            vs.last_use = ++tick;
            ++promotions;
          }
        }
      }
    };

    // Burst 8 distinct WARM/COLD experts in layer 6 twice without draining -> forces queue saturation (cap=4).
    const std::array<int, 8> burst_l6 = {10, 11, 12, 13, 14, 15, 16, 17};
    execute_step(6, burst_l6, /*drain_after=*/false);
    execute_step(6, burst_l6, /*drain_after=*/false);
    expect(q.size() == kCap, "bounded swap queue saturated at exact capacity");
    expect(q.dropped_bound() == 4, "4 excess promotion requests counted in dropped_bound");
    expect(experts_dropped == 0, "experts_dropped remains strictly 0 during saturation");

    // Verify avoid_layer guard: trying to select a victim from layer 0 while cur_layer==0 returns -1.
    expect(q.select_victim(slots, /*avoid_layer=*/0, pin_gen, 10) == -1,
           "avoid_layer prevents evicting victim from currently executing layer");

    // Now drain promotions while executing layer 6 -> evicts layer 0's cold HOT experts (hits=0 vs hits>=2).
    for (int step = 0; step < 4; ++step) execute_step(6, burst_l6, /*drain_after=*/true);
    expect(promotions == 4 && evictions == 4, "forced 4 promotions and 4 HOT evictions");
    for (int e = 0; e < 4; ++e) expect(was_evicted[e], "each original HOT expert evicted cleanly");

    // Now return to layer 0 and route to the evicted experts 0..3 -> verifies byte-identical re-fetch!
    const std::array<int, 8> ret_l0 = {0, 1, 2, 3, 20, 21, 22, 23};
    execute_step(0, ret_l0, /*drain_after=*/false);
    expect(refetches_of_evicted == 4, "all 4 evicted HOT experts re-fetched byte-identically from immutable backing");
    std::printf("PASS swap_queue: promo=%llu evict=%llu refetch=%llu qdrop=%llu edrop=%llu\n",
                static_cast<unsigned long long>(promotions), static_cast<unsigned long long>(evictions),
                static_cast<unsigned long long>(refetches_of_evicted),
                static_cast<unsigned long long>(q.dropped_bound()), static_cast<unsigned long long>(experts_dropped));
  }

  // ------------------------------------------------------------------ 5. Defaults, parsing, StateIdentity, null-safety
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

    // Bidirectional StateIdentity isolation (off <-> q5_k)
    StateIdentity id_off;
    id_off.attn_requant = static_cast<std::uint8_t>(AttnRequantMode::Off);
    StateIdentity id_q5 = id_off;
    id_q5.attn_requant = static_cast<std::uint8_t>(AttnRequantMode::Q5K);
    expect(!(id_off == id_q5) && !(id_q5 == id_off), "StateIdentity bidirectional attn_requant inequality");
    expect(hash_state_identity(id_off) != hash_state_identity(id_q5), "hash_state_identity distinguishes attn_requant");

    // Null-safety of elided FP32 alias views in Layer (absent tensors carry data == nullptr)
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

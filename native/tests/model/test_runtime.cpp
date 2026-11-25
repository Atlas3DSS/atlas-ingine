// The decode runtime on a synthetic model (random weights in the real layouts; experts shared by all layers
// and routing pinned to 8 experts, so it needs well under 2 GB of GPU memory): graph replay equals the eager
// step bit for bit; the verify widths 2/4/8 reproduce consecutive width-1 steps; the speculative rollback
// (save_verify / rollback) leaves the state a shorter call would; taps and logits are finite; pipelined
// width-1 steps (chain_advance) equal explicit inputs and chain_undo restores the state bit for bit; the
// device-selected verify (switch graph, positions and accepted count on the device) equals the host-driven
// call. The Generator (plain, pipelined decoding): stop tokens, cancellation and max_new_tokens end the
// output exactly, and a follow-up request reusing the prefix generates what it generates after a request
// that never ran past its end. The ablation file layouts through prefill and decode: zero offsets without the
// gate equal the plain file, a gate that is never reached equals no ablation, a reachable one differs.
#include "common/test_util.h"
#include "core/fixture.h"
#include "core/hal.h"

#include "kernels/dense/dense.h"
#include "model/kv_checkpoint.h"
#include "model/model_kernels.h"
#include "model/prefill.h"
#include "model/runtime.h"

#include "ingine/generator.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ingine;
using namespace ingine::model;
using ingine::test::float_to_bf16;
using ingine::test::Rng;

namespace {

constexpr int kVocab = 4096;  // unembedding rows used by the synthetic model (the rest are zero; token ids stay below)
constexpr int kExperts = 8;   // experts actually allocated (the router bias pins routing to them)
constexpr int H = HIDDEN;

struct Gen {
  Rng rng{77};
  std::vector<DeviceBuffer>& st;
  cudaStream_t stream;

  void* up(const void* host, std::size_t bytes) {
    st.emplace_back(bytes);
    st.back().copy_from_host(host, bytes);
    return st.back().p;
  }
  Tensor f32(std::initializer_list<std::int32_t> shape, double sd, double mean = 0) {
    std::size_t n = 1;
    for (auto d : shape) n *= std::size_t(d);
    std::vector<float> h(n);
    for (auto& x : h) x = float(mean + rng.normal() * sd);
    return rm(up(h.data(), n * 4), DType::FP32, shape);
  }
  Tensor bf16(std::initializer_list<std::int32_t> shape, double sd) {
    std::size_t n = 1;
    for (auto d : shape) n *= std::size_t(d);
    std::vector<ingine::ActT> h(n);
    for (auto& x : h) x = float_to_bf16(float(rng.normal() * sd));
    return rm(up(h.data(), n * 2), ingine::kActDType, shape);
  }
  Dense fp8(int rows, int cols, double sd) {
    const Tensor b = bf16({rows, cols}, sd);
    st.emplace_back(std::size_t(rows) * cols);
    void* q = st.back().p;
    st.emplace_back(std::size_t(rows) * 4);
    float* s = static_cast<float*>(st.back().p);
    kernels::dense::fp8_rows(static_cast<const ingine::ActT*>(b.data), rows, cols, q, s, stream);
    INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
    return Dense{rm(q, DType::FP8_E4M3FN, {rows, cols}), rm(s, DType::FP32, {rows})};
  }
  Tensor bytes(std::initializer_list<std::int32_t> shape, DType dt, int lo, int hi) {
    std::size_t n = 1;
    for (auto d : shape) n *= std::size_t(d);
    std::vector<std::uint8_t> h(n);
    for (auto& x : h) x = std::uint8_t(lo + int(rng.next() % std::uint64_t(hi - lo + 1)));
    return rm(up(h.data(), n), dt, shape);
  }
};

Weights synthetic_weights(cudaStream_t stream) {
  Weights w;
  w.storage.reserve(1024);
  Gen g{Rng(77), w.storage, stream};
  w.dense_fp8 = true;
  // experts shared by every layer (NVFP4); only experts 0..7 exist, the router bias selects exactly them
  Experts ex;
  ex.fmt = ExpertFormat::NVFP4;
  ex.w13 = g.bytes({kExperts, 2 * INTER, H / 2}, DType::U8, 0, 255);
  ex.w13_scale = g.bytes({kExperts, 2 * INTER, H / 16}, DType::FP8_E4M3FN, 0x30, 0x48);  // E4M3 ~ 0.5..4
  ex.w2 = g.bytes({kExperts, H, INTER / 2}, DType::U8, 0, 255);
  ex.w2_scale = g.bytes({kExperts, H, INTER / 16}, DType::FP8_E4M3FN, 0x30, 0x48);
  ex.w13_factor = g.f32({EXPERTS}, 0.0, 0.004);
  ex.w2_factor = g.f32({EXPERTS}, 0.0, 0.004);
  std::vector<float> bias(EXPERTS, 0.f);
  for (int e = 0; e < kExperts; ++e) bias[e] = 10.f;
  const Tensor router_bias = rm(g.up(bias.data(), bias.size() * 4), DType::FP32, {EXPERTS});

  const Dense wproj[2] = {g.fp8(PROJ_ROWS_SLIDING, H, 0.02), g.fp8(PROJ_ROWS_FULL, H, 0.02)};
  const Dense wo = g.fp8(H, HEADS * HEAD_DIM, 0.02);
  const Tensor ones3 = g.f32({3, H}, 0.05, 1.0), ones1 = g.f32({1, H}, 0.05, 1.0);
  for (int l = 0; l < LAYERS; ++l) {
    Layer L;
    L.idx = l;
    L.full = is_full_layer(l);
    L.loop = is_loop_layer(l);
    L.wproj = wproj[L.full];
    L.wo = wo;
    L.q_scale = g.f32({HEADS, HEAD_DIM}, 0.1, 1.0);
    L.r_scale = g.f32({HEADS, D_REL}, 0.1, 1.0);
    L.rel_proj = g.f32({D_REL, REL_EXTENT}, 0.3);
    if (L.full) {
      L.k_static = g.f32({CONV_K, KV_DIM}, 0.3);
      L.v_static = g.f32({CONV_K, KV_DIM}, 0.3);
    } else {
      L.k_conv = g.f32({KV_DIM, CONV_K}, 0.3);
      L.v_conv = g.f32({KV_DIM, CONV_K}, 0.3);
    }
    L.attn_conv = g.f32({H, CONV_K}, 0.3);
    L.mlp_conv = g.f32({H, CONV_K}, 0.3);
    L.router_bf16 = g.bf16({EXPERTS, H}, 0.02);
    L.router_bias = router_bias;
    if (l > 0) L.attn_query = g.f32({L.full ? 3 : 1, H}, 0.05);
    L.mlp_query = g.f32({1, H}, 0.05);
    L.attn_norm3 = ones3;
    L.attn_norm_mix = L.full ? ones3 : rm(ones3.data, DType::FP32, {1, H});
    L.mlp_norm1 = ones1;
    if (L.loop) {
      L.qkv_gate_down = g.bf16({3, GATE_RANK, 2 * H}, 0.02);
      L.qkv_gate_forget16 = g.bf16({3, H, GATE_RANK}, 0.1);
      L.qkv_gate_gain16 = g.bf16({3, H, GATE_RANK}, 0.1);
      L.mlp1_gate_down = g.bf16({1, GATE_RANK, 2 * H}, 0.02);
      L.mlp1_gate_forget16 = g.bf16({1, H, GATE_RANK}, 0.1);
      L.mlp1_gate_gain16 = g.bf16({1, H, GATE_RANK}, 0.1);
    }
    L.experts = ex;
    w.layers.push_back(L);
    w.expert_formats[l] = ExpertFormat::NVFP4;
  }
  w.embed = g.bf16({kVocab, H}, 0.05);
  w.embed_norm = g.f32({H}, 0.05, 1.0);
  w.final_query = g.f32({1, H}, 0.05);
  w.final_norm = g.f32({H}, 0.05, 1.0);
  w.message_queries = g.f32({LOOP_END - LOOP_START, 4, H}, 0.05);
  // the head and the prefill read every vocabulary row: rows >= kVocab are zero (logit 0), below the best of
  // the random rows, so greedy and top-k sampled tokens stay below kVocab
  const Dense used = g.fp8(kVocab, H, 0.05);
  const std::size_t q_bytes = std::size_t(UNPADDED_VOCAB) * H, s_bytes = std::size_t(UNPADDED_VOCAB) * 4;
  w.storage.emplace_back(q_bytes);
  void* q = w.storage.back().p;
  w.storage.emplace_back(s_bytes);
  void* sc = w.storage.back().p;
  INGINE_TEST_CUDA(cudaMemsetAsync(q, 0, q_bytes, stream));
  INGINE_TEST_CUDA(cudaMemsetAsync(sc, 0, s_bytes, stream));
  INGINE_TEST_CUDA(cudaMemcpyAsync(q, used.w.data, std::size_t(kVocab) * H, cudaMemcpyDeviceToDevice, stream));
  INGINE_TEST_CUDA(cudaMemcpyAsync(sc, used.scale.data, std::size_t(kVocab) * 4, cudaMemcpyDeviceToDevice, stream));
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  w.unembed = Dense{rm(q, DType::FP8_E4M3FN, {UNPADDED_VOCAB, H}), rm(sc, DType::FP32, {UNPADDED_VOCAB})};
  return w;
}

Weights synthetic_weights_q4_equiv(cudaStream_t stream, bool packed_q4) {
  Weights w = synthetic_weights(stream);
  Rng rng(177);
  constexpr float kScale = 0.0625f;
  const std::uint16_t scale_bits = ingine::act::host_float_to_fp16_bits(kScale);
  if (packed_q4) {
    std::vector<std::uint8_t> packed(std::size_t(kVocab) * EMBED_Q4_ROW_BYTES, 0);
    for (int r = 0; r < kVocab; ++r) {
      std::uint8_t* dst = packed.data() + std::size_t(r) * EMBED_Q4_ROW_BYTES;
      for (int j = 0; j < H; j += 2) {
        const std::uint8_t q0 = static_cast<std::uint8_t>(rng.next() & 0xfu);
        const std::uint8_t q1 = static_cast<std::uint8_t>(rng.next() & 0xfu);
        dst[j >> 1] = static_cast<std::uint8_t>(q0 | (q1 << 4));
      }
      for (int b = 0; b < H / 32; ++b) {
        std::memcpy(dst + EMBED_Q4_NIBBLE_BYTES + std::size_t(b) * 2, &scale_bits, 2);
      }
    }
    w.storage.emplace_back(packed.size());
    w.storage.back().copy_from_host(packed.data(), packed.size());
    w.embed = rm(w.storage.back().p, DType::U8, {kVocab, EMBED_Q4_ROW_BYTES});
  } else {
    std::vector<ingine::ActT> dense_emb(std::size_t(kVocab) * H);
    for (int r = 0; r < kVocab; ++r) {
      ingine::ActT* dst = dense_emb.data() + std::size_t(r) * H;
      for (int j = 0; j < H; j += 2) {
        const std::uint8_t q0 = static_cast<std::uint8_t>(rng.next() & 0xfu);
        const std::uint8_t q1 = static_cast<std::uint8_t>(rng.next() & 0xfu);
        dst[j] = float_to_bf16((float(q0) - 8.0f) * kScale);
        dst[j + 1] = float_to_bf16((float(q1) - 8.0f) * kScale);
      }
    }
    w.storage.emplace_back(dense_emb.size() * sizeof(ingine::ActT));
    w.storage.back().copy_from_host(dense_emb.data(), dense_emb.size() * sizeof(ingine::ActT));
    w.embed = rm(w.storage.back().p, ingine::kActDType, {kVocab, H});
  }
  return w;
}

std::vector<float> host(const float* d, std::size_t n, cudaStream_t s) {
  std::vector<float> h(n);
  INGINE_TEST_CUDA(cudaMemcpyAsync(h.data(), d, n * 4, cudaMemcpyDeviceToHost, s));
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  return h;
}

template <class T>
std::vector<T> host_t(const Tensor& t, cudaStream_t s) {
  std::vector<T> h(std::size_t(t.numel()));
  INGINE_TEST_CUDA(cudaMemcpyAsync(h.data(), t.data, h.size() * sizeof(T), cudaMemcpyDeviceToHost, s));
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  return h;
}

// row `row` of width-m logits ([m, vocabulary rows] buffer, kVocab columns used)
std::vector<float> logits_row(Runtime& rt, int m, int row) {
  const std::size_t stride = std::size_t(rt.weights().unembed.rows());
  return host(static_cast<const float*>(rt.buffers(m).logits.data) + row * stride, kVocab, rt.stream());
}

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double d = 0, n = 0;
  for (std::size_t i = 0; i < a.size(); ++i) d += (double(a[i]) - b[i]) * (double(a[i]) - b[i]), n += double(b[i]) * b[i];
  return std::sqrt(d / std::max(n, 1e-30));
}

bool finite(const std::vector<float>& v) {
  for (float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

bool report(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
  return ok;
}

// Every state tensor a decode step changes besides the cache rows at its own positions.
struct StateCopy {
  std::vector<float> conv_k, conv_v, conv_attn, conv_mlp;
  std::vector<std::int64_t> ring_pos;
  std::vector<std::uint16_t> k_ring, v_ring;
  bool operator==(const StateCopy&) const = default;
};

StateCopy copy_state(Runtime& rt) {
  const State& st = rt.state();
  const cudaStream_t s = rt.stream();
  return {host_t<float>(st.conv_k, s),         host_t<float>(st.conv_v, s),         host_t<float>(st.conv_attn, s),
          host_t<float>(st.conv_mlp, s),       host_t<std::int64_t>(st.ring_pos, s), host_t<std::uint16_t>(st.k_ring, s),
          host_t<std::uint16_t>(st.v_ring, s)};
}

// all m rows of width-m logits (kVocab columns each)
std::vector<float> logits_rows(Runtime& rt, int m) {
  std::vector<float> all;
  for (int r = 0; r < m; ++r) {
    const auto row = logits_row(rt, m, r);
    all.insert(all.end(), row.begin(), row.end());
  }
  return all;
}

bool run(bool kv_fp8) {
  const std::string tag = kv_fp8 ? " (fp8 kv)" : " (bf16 kv)";
  RuntimeOptions opt;
  opt.max_ctx = 2048;
  opt.spec = true;
  opt.kv_fp8 = kv_fp8;
  cudaStream_t s0;
  INGINE_TEST_CUDA(cudaStreamCreate(&s0));
  Weights w = synthetic_weights(s0);
  Runtime rt(opt, std::move(w), load_ablation({}));
  INGINE_TEST_CUDA(cudaStreamDestroy(s0));
  const cudaStream_t s = rt.stream();
  const std::int32_t toks[8] = {11, 523, 7, 4000, 19, 2048, 3, 77};
  bool ok = true;

  // 1. graph replay == eager step (bit for bit), from the same state
  reset_state(rt.state(), s);
  {
    const Runtime::Snapshot snap = rt.snapshot();
    rt.set_inputs({toks, 1}, 0);
    rt.run_step_eager(1);
    const auto eager = logits_row(rt, 1, 0);
    rt.restore(snap);
    rt.set_inputs({toks, 1}, 0);
    rt.run_step(1);
    const auto graph = logits_row(rt, 1, 0);
    ok &= report(finite(eager) && eager == graph, "graph replay equals the eager step" + tag);
  }

  // 2. six width-1 steps vs one width-2 and one width-4 call
  std::vector<std::vector<float>> seq;
  reset_state(rt.state(), s);
  for (int i = 0; i < 6; ++i) {
    rt.decode_logits(toks[i], i);
    seq.push_back(logits_row(rt, 1, 0));
  }
  const auto taps1 = host_t<ingine::ActT>(rt.buffers(1).taps, s);
  reset_state(rt.state(), s);
  rt.set_inputs({toks, 2}, 0);
  rt.run_step(2);
  std::vector<std::vector<float>> wide = {logits_row(rt, 2, 0), logits_row(rt, 2, 1)};
  rt.set_inputs({toks + 2, 4}, 2);
  rt.run_step(4);
  for (int i = 0; i < 4; ++i) wide.push_back(logits_row(rt, 4, i));
  double worst = 0;
  bool all_finite = true;
  for (int i = 0; i < 6; ++i) worst = std::max(worst, rel_l2(wide[i], seq[i])), all_finite &= finite(wide[i]);
  ok &= report(all_finite && worst < 0.03,
               "widths 2 + 4 reproduce six width-1 steps (worst relative L2 " + std::to_string(worst) + ")" + tag);
  {
    const auto taps4 = host_t<ingine::ActT>(rt.buffers(4).taps, s);
    bool f = true;
    for (const auto* t : {&taps1, &taps4})
      for (auto x : *t) f &= std::isfinite(ingine::test::bf16_to_float(x));
    ok &= report(f, "taps finite" + tag);
  }

  // 3. rollback: a width-8 verify at P = 0 rolled back to its first 3 rows against three width-1 steps
  //    (conv states and ring positions), then one more step on both
  reset_state(rt.state(), s);
  for (int i = 0; i < 3; ++i) rt.decode_logits(toks[i], i);
  const auto ref_ck = host_t<float>(rt.state().conv_k, s), ref_cm = host_t<float>(rt.state().conv_mlp, s);
  const auto ref_rp = host_t<std::int64_t>(rt.state().ring_pos, s);
  rt.decode_logits(toks[3], 3);
  const auto ref_next = logits_row(rt, 1, 0);

  reset_state(rt.state(), s);
  rt.save_verify(0, 8);
  rt.set_inputs({toks, 8}, 0);
  rt.run_step(8);
  rt.rollback(0, 3, 8);
  const auto ck = host_t<float>(rt.state().conv_k, s), cm = host_t<float>(rt.state().conv_mlp, s);
  const auto rp = host_t<std::int64_t>(rt.state().ring_pos, s);
  ok &= report(rp == ref_rp, "rollback restores the ring positions" + tag);
  ok &= report(rel_l2(ck, ref_ck) < 0.03 && rel_l2(cm, ref_cm) < 0.03,
               "rollback leaves the conv states of three steps (relative L2 " + std::to_string(rel_l2(ck, ref_ck)) +
                   ", " + std::to_string(rel_l2(cm, ref_cm)) + ")" + tag);
  rt.decode_logits(toks[3], 3);
  const auto next = logits_row(rt, 1, 0);
  ok &= report(finite(next) && rel_l2(next, ref_next) < 0.03,
               "decoding after the rollback matches (relative L2 " + std::to_string(rel_l2(next, ref_next)) + ")" + tag);

  // 4. a rollback across the ring wrap (P = 1020, slots 1020..1027 mod 1024)
  reset_state(rt.state(), s);
  const auto before = host_t<std::int64_t>(rt.state().ring_pos, s);
  rt.save_verify(1020, 8);
  rt.set_inputs({toks, 8}, 1020);
  rt.run_step(8);
  rt.rollback(1020, 2, 8);
  const auto after = host_t<std::int64_t>(rt.state().ring_pos, s);
  bool wrap_ok = true;
  for (int ring = 0; ring < NUM_SLIDING_VISITS; ++ring)
    for (int slot = 0; slot < WINDOW; ++slot) {
      const std::int64_t got = after[std::size_t(ring) * WINDOW + slot];
      const std::int64_t want = slot == 1020 ? 1020 : slot == 1021 ? 1021 : before[std::size_t(ring) * WINDOW + slot];
      wrap_ok &= got == want;
    }
  ok &= report(wrap_ok, "rollback across the ring wrap" + tag);

  // 5. pipelined width-1 steps (input token from device memory) equal explicitly set inputs bit for bit
  DeviceBuffer sampled(sizeof(std::int32_t));
  auto put_token = [&](std::int32_t t) {
    INGINE_TEST_CUDA(cudaMemcpyAsync(sampled.p, &t, sizeof t, cudaMemcpyHostToDevice, s));
  };
  {
    std::vector<std::vector<float>> explicit_steps, chained;
    reset_state(rt.state(), s);
    for (int i = 0; i < 4; ++i) {
      rt.decode_logits(toks[i], 1000 + i);
      explicit_steps.push_back(logits_row(rt, 1, 0));
    }
    reset_state(rt.state(), s);
    bool tokens_ok = true;
    for (int i = 0; i < 4; ++i) {
      put_token(toks[i]);
      rt.chain_advance(static_cast<const std::int32_t*>(sampled.p), 1000 + i);
      rt.run_step(1);
      tokens_ok &= rt.chain_token(1000 + i) == toks[i];
      chained.push_back(logits_row(rt, 1, 0));
    }
    ok &= report(tokens_ok && chained == explicit_steps, "chained width-1 steps equal explicit inputs" + tag);
  }

  // 6. chain_undo restores the state before the step bit for bit (slot 0 after the ring wrap: position 1024)
  {
    reset_state(rt.state(), s);
    for (int i = 0; i < 3; ++i) rt.decode_logits(toks[i], 1021 + i);
    const StateCopy before_step = copy_state(rt);
    put_token(toks[3]);
    rt.chain_advance(static_cast<const std::int32_t*>(sampled.p), 1024);
    rt.run_step(1);
    const bool changed = !(copy_state(rt) == before_step);
    rt.chain_undo(1024);
    ok &= report(changed && copy_state(rt) == before_step, "chain_undo restores the state bit for bit" + tag);
  }

  // 7. the device-selected verify (switch over the widths: save, decode, rollback with the width, position and
  //    accepted count read on the device) equals the host-driven call bit for bit
  {
    DeviceBuffer width(sizeof(std::int32_t)), accepted(sizeof(std::int32_t));
    const std::span<const int> vw = rt.verify_widths();
    Widths widths;
    widths.n = int(vw.size());
    std::copy(vw.begin(), vw.end(), widths.w);
    GraphExec verify = rt.capture_switch(
        [&](cudaStream_t cs, cudaGraphConditionalHandle h) {
          return width_select(static_cast<const std::int32_t*>(width.p), widths, h, cs);
        },
        [&](cudaStream_t cs, int m) {
          rt.record_save_verify(m, cs);
          rt.record_step(m, cs);
          rt.record_rollback(m, static_cast<const std::int32_t*>(accepted.p), cs);
        });
    const int P = 1018;  // rows cross the ring wrap
    for (auto [m, acc] : {std::pair{8, 2}, std::pair{4, 3}, std::pair{2, 0}}) {
      auto prologue = [&] {
        reset_state(rt.state(), s);
        for (int i = 0; i < 2; ++i) rt.decode_logits(toks[7 - i], P - 2 + i);
      };
      prologue();
      rt.save_verify(P, m);
      rt.set_inputs({toks, std::size_t(m)}, P);
      rt.run_step(m);
      const auto host_logits = logits_rows(rt, m);
      rt.rollback(P, acc + 1, m);
      const StateCopy host_state = copy_state(rt);

      prologue();
      rt.set_inputs({toks, std::size_t(m)}, P);
      INGINE_TEST_CUDA(cudaMemcpyAsync(width.p, &m, 4, cudaMemcpyHostToDevice, s));
      INGINE_TEST_CUDA(cudaMemcpyAsync(accepted.p, &acc, 4, cudaMemcpyHostToDevice, s));
      verify.launch(s);
      const auto device_logits = logits_rows(rt, m);
      ok &= report(device_logits == host_logits && copy_state(rt) == host_state,
                   "device-selected verify width " + std::to_string(m) + ", commit " + std::to_string(acc + 1) +
                       " equals the host-driven call" + tag);
    }
  }

  // 8. prompt checkpoint round trip: save, decode 1100 steps (the rings wrap), restore: bit-identical state
  {
    reset_state(rt.state(), s);
    for (int i = 0; i < 3; ++i) rt.decode_logits(toks[i], i);
    const StateCopy saved_state = copy_state(rt);
    DeviceBuffer ck(rt.checkpoint_bytes());
    cudaEvent_t e0, e1, e2, e3;
    for (cudaEvent_t* e : {&e0, &e1, &e2, &e3}) INGINE_TEST_CUDA(cudaEventCreate(e));
    INGINE_TEST_CUDA(cudaEventRecord(e0, s));
    rt.save_checkpoint(ck.p);
    INGINE_TEST_CUDA(cudaEventRecord(e1, s));
    for (int i = 3; i < 1103; ++i) rt.decode_logits(toks[i % 8], i);
    INGINE_TEST_CUDA(cudaEventRecord(e2, s));
    rt.restore_checkpoint(ck.p);
    INGINE_TEST_CUDA(cudaEventRecord(e3, s));
    INGINE_TEST_CUDA(cudaEventSynchronize(e3));
    float save_ms = 0, restore_ms = 0;
    INGINE_TEST_CUDA(cudaEventElapsedTime(&save_ms, e0, e1));
    INGINE_TEST_CUDA(cudaEventElapsedTime(&restore_ms, e2, e3));
    for (cudaEvent_t e : {e0, e1, e2, e3}) INGINE_TEST_CUDA(cudaEventDestroy(e));
    char buf[160];
    std::snprintf(buf, sizeof buf, "checkpoint round trip restores the state bit for bit (%.0f MB, save %.3f ms, restore %.3f ms)",
                  double(rt.checkpoint_bytes()) / 1e6, save_ms, restore_ms);
    ok &= report(copy_state(rt) == saved_state, buf + tag);
  }

  // 8b. conversation parking: A decodes 1100 positions, parks (checkpoint + full rows [0, 1100)); B starts
  //     over and decodes 1500 positions (every full row A used and the rings overwritten); A restored decodes
  //     on with the same logits, bit for bit, as A never displaced
  {
    const int n = 1100, more = 20;
    auto run_a = [&](int from, int to, std::vector<float>* logits) {
      for (int i = from; i < to; ++i) {
        rt.decode_logits(toks[i % 8], i);
        if (logits) {
          const auto row = logits_row(rt, 1, 0);
          logits->insert(logits->end(), row.begin(), row.end());
        }
      }
    };
    reset_state(rt.state(), s);
    run_a(0, n, nullptr);
    std::vector<float> want, got;
    run_a(n, n + more, &want);
    reset_state(rt.state(), s);
    run_a(0, n, nullptr);
    DeviceBuffer ck(rt.checkpoint_bytes()), rows(rt.full_rows_bytes(n));
    rt.save_checkpoint(ck.p);
    rt.save_full_rows(rows.p, n);
    reset_state(rt.state(), s);
    for (int i = 0; i < 1500; ++i) rt.decode_logits(toks[(3 * i + 1) % 8], i);
    rt.restore_checkpoint(ck.p);
    rt.restore_full_rows(rows.p, n);
    run_a(n, n + more, &got);
    char buf[160];
    std::snprintf(buf, sizeof buf, "parked conversation restores bit for bit after another one (%.1f KB per position)",
                  double(rt.full_rows_bytes(n)) / n / 1e3);
    ok &= report(!want.empty() && got == want, buf + tag);
  }

  // 9. device width selection: verify_select (confidences -> lead -> width, drafts into token[1..7]) for
  //    leads 0, 1, 2, 4, 7; width_select of a width that is not a verify width runs no branch; verify_publish
  //    writes the host record
  {
    const std::span<const int> vw = rt.verify_widths();
    Widths widths;
    widths.n = int(vw.size());
    std::copy(vw.begin(), vw.end(), widths.w);
    DeviceBuffer conf(7 * 4), drafts(7 * 4), m_out(4), marker(4), m_in(4), acc(4), next(4);
    auto mark = [&](cudaStream_t cs, int m) { INGINE_TEST_CUDA(cudaMemsetAsync(marker.p, m, 1, cs)); };
    GraphExec select = rt.capture_switch(
        [&](cudaStream_t cs, cudaGraphConditionalHandle h) {
          return verify_select(static_cast<const float*>(conf.p), static_cast<const std::int32_t*>(drafts.p), 0.8f,
                               widths, rt.tokens(), static_cast<std::int32_t*>(m_out.p), h, cs);
        },
        mark);
    GraphExec by_width = rt.capture_switch(
        [&](cudaStream_t cs, cudaGraphConditionalHandle h) {
          return width_select(static_cast<const std::int32_t*>(m_in.p), widths, h, cs);
        },
        mark);
    const std::int32_t d[7] = {101, 102, 103, 104, 105, 106, 107};
    INGINE_TEST_CUDA(cudaMemcpy(drafts.p, d, sizeof d, cudaMemcpyHostToDevice));
    bool sel_ok = true;
    for (auto [lead, want] : {std::pair{0, 2}, std::pair{1, 2}, std::pair{2, 4}, std::pair{4, 8}, std::pair{7, 8}}) {
      float c[7];
      for (int i = 0; i < 7; ++i) c[i] = i < lead ? 0.9f : (i == lead ? 0.8f : 0.95f);  // 0.8 is not > 0.8
      INGINE_TEST_CUDA(cudaMemcpy(conf.p, c, sizeof c, cudaMemcpyHostToDevice));
      INGINE_TEST_CUDA(cudaMemset(marker.p, 0, 4));
      INGINE_TEST_CUDA(cudaMemset(rt.tokens(), 0, 8 * 8));
      INGINE_TEST_CUDA(cudaDeviceSynchronize());  // the uploads above are not ordered with stream s
      select.launch(s);
      INGINE_TEST_CUDA(cudaStreamSynchronize(s));
      std::int32_t m = 0, mk = 0;
      std::int64_t tk[8];
      INGINE_TEST_CUDA(cudaMemcpy(&m, m_out.p, 4, cudaMemcpyDeviceToHost));
      INGINE_TEST_CUDA(cudaMemcpy(&mk, marker.p, 4, cudaMemcpyDeviceToHost));
      INGINE_TEST_CUDA(cudaMemcpy(tk, rt.tokens(), sizeof tk, cudaMemcpyDeviceToHost));
      bool tok_ok = tk[0] == 0;
      for (int i = 0; i < 7; ++i) tok_ok &= tk[1 + i] == d[i];
      sel_ok &= m == want && (mk & 0xff) == want && tok_ok;
    }
    ok &= report(sel_ok, "verify_select picks the width of leads 0/1/2/4/7 and copies the drafts" + tag);
    bool none_ok = true;
    for (auto [m, want] : {std::pair{4, 4}, std::pair{5, 0}}) {
      INGINE_TEST_CUDA(cudaMemset(marker.p, 0, 4));
      INGINE_TEST_CUDA(cudaMemcpy(m_in.p, &m, 4, cudaMemcpyHostToDevice));
      INGINE_TEST_CUDA(cudaDeviceSynchronize());
      by_width.launch(s);
      INGINE_TEST_CUDA(cudaStreamSynchronize(s));
      std::int32_t mk = 0;
      INGINE_TEST_CUDA(cudaMemcpy(&mk, marker.p, 4, cudaMemcpyDeviceToHost));
      none_ok &= (mk & 0xff) == want;
    }
    ok &= report(none_ok, "width_select runs the branch of a verify width and none for another width" + tag);
    MappedHostBuffer rec(sizeof(VerifyRecord));
    const std::int32_t m = 4, a = 2, nx = 777;
    INGINE_TEST_CUDA(cudaMemcpy(m_in.p, &m, 4, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(acc.p, &a, 4, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(next.p, &nx, 4, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    auto* r = static_cast<VerifyRecord*>(rec.data());
    r->next = -5;
    verify_publish(static_cast<const std::int32_t*>(m_in.p), static_cast<const std::int32_t*>(acc.p), nullptr,
                   rt.tokens(), static_cast<VerifyRecord*>(rec.device_ptr()), s);
    INGINE_TEST_CUDA(cudaStreamSynchronize(s));
    bool pub_ok = r->m == 4 && r->accepted == 2 && r->next == -5;  // next untouched without a source
    for (int i = 0; i < 7; ++i) pub_ok &= r->drafts[i] == d[i];
    verify_publish(static_cast<const std::int32_t*>(m_in.p), static_cast<const std::int32_t*>(acc.p),
                   static_cast<const std::int32_t*>(next.p), rt.tokens(), static_cast<VerifyRecord*>(rec.device_ptr()), s);
    INGINE_TEST_CUDA(cudaStreamSynchronize(s));
    pub_ok &= r->next == 777;
    ok &= report(pub_ok, "verify_publish writes the width, accepted count, drafts and next token" + tag);
  }
  return ok;
}

// The Generator over the synthetic model (plain pipelined decoding, FP8 caches, 64-token prefill chunks).
bool run_generator() {
  RuntimeOptions opt;
  opt.max_ctx = 2048;
  opt.kv_fp8 = true;
  opt.prefill_chunk = 64;
  cudaStream_t s0;
  INGINE_TEST_CUDA(cudaStreamCreate(&s0));
  Weights w = synthetic_weights(s0);
  INGINE_TEST_CUDA(cudaStreamDestroy(s0));
  const auto gen = make_generator(opt, std::move(w), load_ablation({}));

  Rng rng(5);
  std::vector<std::int32_t> prompt(100);
  for (auto& t : prompt) t = std::int32_t(rng.next() % kVocab);
  struct Out {
    std::vector<std::int32_t> tokens;
    GenerateStats stats;
  };
  // cancel_after >= 0: on_tokens returns false once that many tokens arrived
  auto run_gen = [&](const std::vector<std::int32_t>& ids, int max_new, const Sampling& sampling,
                     std::vector<std::int32_t> stop, int cancel_after = -1) {
    Out o;
    o.stats = gen->generate(ids, max_new, sampling, stop, [&](std::span<const std::int32_t> t) {
      o.tokens.insert(o.tokens.end(), t.begin(), t.end());
      return cancel_after < 0 || int(o.tokens.size()) < cancel_after;
    });
    return o;
  };
  auto head = [](const std::vector<std::int32_t>& v, int n) { return std::vector<std::int32_t>(v.begin(), v.begin() + n); };

  bool ok = true;
  Sampling greedy;
  greedy.temperature = 0.f;
  Sampling sampled;
  sampled.temperature = 1.f;
  sampled.top_k = 50;
  sampled.seed = 7;
  for (const auto& [name, sp] : {std::pair{"greedy", greedy}, std::pair{"sampled", sampled}}) {
    const std::string tag = std::string(" (") + name + ")";
    const Out ref = run_gen(prompt, 48, sp, {});
    bool ref_ok = ref.tokens.size() == 48 && ref.stats.generated_tokens == 48;
    for (auto t : ref.tokens) ref_ok &= t >= 0 && t < kVocab;
    ok &= report(ref_ok, "48 tokens below the used vocabulary" + tag);
    const Out again = run_gen(prompt, 48, sp, {});
    ok &= report(again.tokens == ref.tokens, "a repeated request generates the same tokens" + tag);

    const Out capped = run_gen(prompt, 9, sp, {});
    ok &= report(capped.tokens == head(ref.tokens, 9) && capped.stats.generated_tokens == 9,
                 "max_new_tokens ends the output exactly" + tag);

    // follow-ups after an end at token k (stop token / cancel, the step for token k was already launched)
    // against a request that ended there by its budget (no step launched past it)
    Sampling follow_sp = sp;
    follow_sp.seed = 11;
    std::vector<int> ends = {0};
    for (int k = 12; k < 48 && ends.size() < 2; ++k)
      if (std::find(ref.tokens.begin(), ref.tokens.begin() + k, ref.tokens[k]) == ref.tokens.begin() + k)
        ends.push_back(k);
    ok &= report(ends.size() == 2, "a token after the 12th that did not occur before (stop candidate)" + tag);
    for (int k : ends) {
      const std::string at = " at token " + std::to_string(k) + tag;
      std::vector<std::int32_t> follow = prompt;
      follow.insert(follow.end(), ref.tokens.begin(), ref.tokens.begin() + k + 1);
      for (int i = 0; i < 6; ++i) follow.push_back(std::int32_t(rng.next() % kVocab));
      const int reuse = int(prompt.size()) + k;

      const Out budget = run_gen(prompt, k + 1, sp, {});
      const Out fresh = run_gen(follow, 24, follow_sp, {});
      ok &= report(budget.tokens == head(ref.tokens, k + 1) && fresh.stats.reused_tokens == reuse &&
                       fresh.tokens.size() == 24,
                   "reference follow-up after a budget end" + at);

      const Out stopped = run_gen(prompt, 48, sp, {ref.tokens[k]});
      const Out after_stop = run_gen(follow, 24, follow_sp, {});
      ok &= report(stopped.tokens == head(ref.tokens, k + 1) && stopped.stats.generated_tokens == k + 1,
                   "a stop token ends the output exactly" + at);
      ok &= report(after_stop.stats.reused_tokens == reuse && after_stop.tokens == fresh.tokens,
                   "the follow-up after a stop token reuses the prefix and matches" + at);

      const Out cancelled = run_gen(prompt, 48, sp, {}, k + 1);
      const Out after_cancel = run_gen(follow, 24, follow_sp, {});
      ok &= report(cancelled.tokens == head(ref.tokens, k + 1) && cancelled.stats.generated_tokens == k + 1,
                   "cancellation ends the output exactly" + at);
      ok &= report(after_cancel.stats.reused_tokens == reuse && after_cancel.tokens == fresh.tokens,
                   "the follow-up after a cancel reuses the prefix and matches" + at);
    }

    // prompt checkpoints: X prefilled the prompt P, generated G and gets P + Y (Y diverges from G); Z prefilled
    // P with no new tokens and gets P + Y through the strict-prefix path. Same chunk boundaries, so X (restored
    // checkpoint) and Z must generate identical tokens. G longer than the 1024-slot ring as well.
    std::vector<std::int32_t> other(50);
    for (auto& t : other) t = std::int32_t(rng.next() % kVocab);
    struct Case {
      int g, common;
    };
    for (const Case c : {Case{40, 0}, Case{40, 3}, Case{1100, 0}}) {
      const std::string at = " (G " + std::to_string(c.g) + " tokens, Y shares " + std::to_string(c.common) + ")" + tag;
      run_gen(other, 2, sp, {});
      const Out g = run_gen(prompt, c.g, sp, {});
      std::vector<std::int32_t> follow = prompt;
      follow.insert(follow.end(), g.tokens.begin(), g.tokens.begin() + c.common);
      follow.push_back((g.tokens[c.common] + 1) % kVocab);
      for (int i = 0; i < 5; ++i) follow.push_back(std::int32_t(rng.next() % kVocab));
      const Out x = run_gen(follow, 24, follow_sp, {});
      run_gen(other, 2, sp, {});
      const Out z0 = run_gen(prompt, 0, sp, {});
      const Out z = run_gen(follow, 24, follow_sp, {});
      ok &= report(g.tokens.size() == std::size_t(c.g) && z0.tokens.empty() && z.stats.reuse == Reuse::Extend &&
                       x.stats.reuse == Reuse::Checkpoint && x.stats.reused_tokens == int(prompt.size()) &&
                       z.stats.reused_tokens == int(prompt.size()) && x.tokens == z.tokens,
                   "a follow-up restoring the prompt checkpoint equals the strict-prefix path" + at);
    }
    // the same prompt again after a generation: the checkpoint holds its state and last logits
    run_gen(other, 2, sp, {});
    run_gen(prompt, 30, sp, {});
    const Out retry = run_gen(prompt, 48, sp, {});
    ok &= report(retry.stats.reuse == Reuse::Checkpoint && retry.stats.reused_tokens == int(prompt.size()) &&
                     retry.tokens == ref.tokens,
                 "the same prompt again restores its checkpoint and generates the same tokens" + tag);
    // a request that fails mid-generation (on_tokens throws) leaves nothing reusable: the next request starts
    // from an empty state and generates what a fresh one does
    bool threw = false;
    int count = 0;
    try {
      gen->generate(prompt, 48, sp, {}, [&](std::span<const std::int32_t>) {
        if (++count == 5) throw std::runtime_error("client gone");
        return true;
      });
    } catch (const std::runtime_error&) {
      threw = true;
    }
    const Out after_failure = run_gen(prompt, 48, sp, {});
    ok &= report(threw && after_failure.stats.reuse == Reuse::None && after_failure.tokens == ref.tokens,
                 "after a failed request the next one starts from an empty state" + tag);
  }
  return ok;
}

// The ablation file layouts (weights.cpp load_ablation) through the whole model: greedy tokens after a prefill.
bool run_ablation_layouts() {
  using ingine::test::FixtureTensor;
  const ingine::test::TempDir dir;
  Rng rng(9);
  const std::size_t nd = std::size_t(NUM_VISITS) * H;
  std::vector<float> attn_dirs(nd), mlp_dirs(nd), attn_w(NUM_VISITS), mlp_w(NUM_VISITS);
  for (auto& x : attn_dirs) x = float(rng.normal());
  for (auto& x : mlp_dirs) x = float(rng.normal());
  for (auto& x : attn_w) x = float(rng.uniform(0.5, 1.5));
  for (auto& x : mlp_w) x = float(rng.uniform(0.5, 1.5));
  auto bytes = [](const std::vector<float>& v) {
    std::vector<std::uint8_t> b(v.size() * 4);
    std::memcpy(b.data(), v.data(), b.size());
    return b;
  };
  const std::vector<std::int64_t> dshape = {NUM_VISITS, H}, vshape = {NUM_VISITS};
  // threshold: detectors (random vectors) with this threshold for every visit; NaN = a file without detectors
  auto file = [&](const char* name, bool plain, float offset, float gate, float threshold = std::nanf("")) {
    std::vector<FixtureTensor> t = {{"attn_dirs", "F32", dshape, bytes(attn_dirs)},
                                    {"mlp_dirs", "F32", dshape, bytes(mlp_dirs)}};
    if (plain) {
      t.push_back({"attn_w", "F32", vshape, bytes(attn_w)});
      t.push_back({"mlp_w", "F32", vshape, bytes(mlp_w)});
    } else {
      const std::vector<float> off(NUM_VISITS, offset);
      t.push_back({"attn_weight", "F32", vshape, bytes(attn_w)});
      t.push_back({"mlp_weight", "F32", vshape, bytes(mlp_w)});
      t.push_back({"attn_offset", "F32", vshape, bytes(off)});
      t.push_back({"mlp_offset", "F32", vshape, bytes(off)});
      t.push_back({"gate", "F32", {1}, bytes({gate})});
      if (!std::isnan(threshold)) {
        const std::vector<float> tau(NUM_VISITS, threshold);
        t.push_back({"attn_det", "F32", dshape, bytes(mlp_dirs)});  // any vectors serve as detectors here
        t.push_back({"mlp_det", "F32", dshape, bytes(attn_dirs)});
        t.push_back({"attn_threshold", "F32", vshape, bytes(tau)});
        t.push_back({"mlp_threshold", "F32", vshape, bytes(tau)});
      }
    }
    const std::filesystem::path p = dir.path / name;
    ingine::test::write_safetensors(p, t);
    return p;
  };
  auto tokens = [&](const std::filesystem::path& ablation) {
    RuntimeOptions opt;
    opt.max_ctx = 2048;
    opt.kv_fp8 = true;
    opt.prefill_chunk = 64;
    cudaStream_t s0;
    INGINE_TEST_CUDA(cudaStreamCreate(&s0));
    Weights w = synthetic_weights(s0);
    INGINE_TEST_CUDA(cudaStreamDestroy(s0));
    const auto gen = make_generator(opt, std::move(w), load_ablation(ablation));
    Rng r(5);
    std::vector<std::int32_t> prompt(100);
    for (auto& t : prompt) t = std::int32_t(r.next() % kVocab);
    Sampling greedy;
    greedy.temperature = 0.f;
    std::vector<std::int32_t> out;
    gen->generate(prompt, 32, greedy, {}, [&](std::span<const std::int32_t> t) {
      out.insert(out.end(), t.begin(), t.end());
      return true;
    });
    return out;
  };
  const auto none = tokens({});
  const auto plain = tokens(file("plain.safetensors", true, 0.f, 0.f));
  const auto zero = tokens(file("zero.safetensors", false, 0.f, 0.f));
  const auto never = tokens(file("never.safetensors", false, 1e30f, 1.f));
  const auto gated = tokens(file("gated.safetensors", false, 0.f, 1.f));
  const auto det_open = tokens(file("det_open.safetensors", false, 0.f, 0.f, -1e30f));
  const auto det_shut = tokens(file("det_shut.safetensors", false, 0.f, 0.f, 1e30f));
  const auto det_some = tokens(file("det_some.safetensors", false, 0.f, 0.f, 0.f));
  bool ok = report(none.size() == 32 && plain != none, "an ablation changes the greedy tokens");
  ok &= report(zero == plain, "zero offsets without the gate equal the plain file");
  ok &= report(never == none, "a gate that is never reached equals no ablation");
  ok &= report(gated != plain && gated != none, "a gate at offset 0 differs from both");
  ok &= report(det_open == plain, "detectors that always fire equal the plain file");
  ok &= report(det_shut == none, "detectors that never fire equal no ablation");
  ok &= report(det_some != plain && det_some != none, "detectors at threshold 0 differ from both");
  return ok;
}

// Mech-interp / persona hooks on the synthetic model:
//   - exact 0.0 bit-identical no-op across all 6 required visits {3, 6, 10, 18, 29, 44} (prefill + hooked CUDA graph)
//   - hooked CUDA graph replay == hooked eager step bit-for-bit
//   - active causal interventions (router ablate/force, 3b affine + 3a tap, residual steer with d_2027 guard)
//   - cross-request cache-hit & parked-conversation persona isolation
bool run_persona_hooks() {
  using namespace ingine::persona;
  bool ok = true;
  RuntimeOptions opt;
  opt.max_ctx = 2048;
  opt.prefill_chunk = 64;
  opt.enable_persona_hooks = true;
  cudaStream_t s0;
  INGINE_TEST_CUDA(cudaStreamCreate(&s0));
  Weights w = synthetic_weights(s0);
  Runtime rt(opt, std::move(w), load_ablation({}));
  INGINE_TEST_CUDA(cudaStreamDestroy(s0));
  const cudaStream_t s = rt.stream();
  Prefiller prefiller(rt.weights(), rt.ablation(), 64, s);

  const std::vector<std::int32_t> prompt = {11, 523, 7, 4000, 19, 2048, 3, 77};
  DeviceBuffer tok_dev(prompt.size() * sizeof(std::int32_t));
  DeviceBuffer logits_dev(std::size_t(UNPADDED_VOCAB) * sizeof(float));
  INGINE_TEST_CUDA(cudaMemcpyAsync(tok_dev.p, prompt.data(), prompt.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, s));

  auto run_pf_and_dec = [&](std::vector<float>& pf_out, std::vector<float>& dec_out) {
    reset_state(rt.state(), s);
    prefiller.run(rt.state(), static_cast<const std::int32_t*>(tok_dev.p), int(prompt.size()), 0,
                  static_cast<float*>(logits_dev.p), nullptr, {}, rt.active_hooks());
    pf_out.resize(kVocab);
    INGINE_TEST_CUDA(cudaMemcpyAsync(pf_out.data(), logits_dev.p, kVocab * sizeof(float), cudaMemcpyDeviceToHost, s));
    rt.decode_logits(42, int(prompt.size()));
    dec_out = logits_row(rt, 1, 0);
  };

  // Baseline unhooked
  rt.set_persona(nullptr);
  std::vector<float> base_pf, base_dec;
  run_pf_and_dec(base_pf, base_dec);

  // Tap routing and ffn_res depth at all 6 required visits {3, 6, 10, 18, 29, 44} (plus loop_msg at {18, 29})
  const std::array<int, 6> kReqVisits = {3, 6, 10, 18, 29, 44};
  PersonaTapConfig tap_all;
  tap_all.max_pos = 16;
  for (int v : kReqVisits) {
    tap_all.routing_visits.push_back(v);
    tap_all.ffn_inp_visits.push_back(v);
    tap_all.depth_taps.push_back({v, DepthSiteKind::FfnRes, true, true});
    if (v == 18 || v == 29) tap_all.depth_taps.push_back({v, DepthSiteKind::LoopMsg, true, true});
  }
  rt.set_persona(nullptr, tap_all);
  std::vector<float> tap_pf, tap_dec;
  run_pf_and_dec(tap_pf, tap_dec);
  ok &= report(tap_pf == base_pf && tap_dec == base_dec,
               "persona read-only taps across all 6 visits are bit-identical to unhooked baseline");
  std::map<int, RoutingTapReadout> saved_rtaps;
  std::map<int, DepthTapReadout> saved_dtaps;
  for (int v : kReqVisits) {
    saved_rtaps[v] = rt.persona_runtime().read_routing_tap(v, 9, s);
    saved_dtaps[v] = rt.persona_runtime().read_depth_tap(v, DepthSiteKind::FfnRes, 9, s);
  }

  // Hooked CUDA graph replay == hooked eager bit-for-bit
  {
    PersonaProfile p;
    p.name = "force_active";
    p.fingerprint_sha256 = sha256_hex("force_active");
    p.cache_key = cache_key_from_hex(p.fingerprint_sha256);
    InterventionSpec iv;
    iv.type = InterventionType::RouterForce;
    iv.visit = 10;
    iv.expert = 5;
    iv.bias = 100.0f;
    iv.gain = 2.5f;
    p.interventions.push_back(iv);
    rt.set_persona(&p);
    reset_state(rt.state(), s);
    const auto snap = rt.snapshot();
    const std::int32_t tk[1] = {42};
    rt.set_inputs(tk, 0);
    rt.run_step_eager(1);
    const auto eager = logits_row(rt, 1, 0);
    rt.restore(snap);
    rt.set_inputs(tk, 0);
    rt.run_step(1);
    const auto graph = logits_row(rt, 1, 0);
    ok &= report(finite(eager) && eager == graph, "hooked CUDA graph replay equals hooked eager step bit for bit");
  }

  // Exact 0.0 bit-identical no-op across all 6 required visits for every hook type
  bool all6_noop_ok = true;
  for (int v : kReqVisits) {
    const auto& rtap = saved_rtaps.at(v);
    // Find an active expert and a never-selected expert at visit v
    std::array<int, EXPERTS> cnt{};
    for (int id : rtap.idx)
      if (id >= 0 && id < EXPERTS) ++cnt[std::size_t(id)];
    int active_e = 0, never_e = -1;
    for (int e = 0; e < EXPERTS; ++e) {
      if (cnt[std::size_t(e)] > cnt[std::size_t(active_e)]) active_e = e;
      if (cnt[std::size_t(e)] == 0 && never_e < 0) never_e = e;
    }
    const auto& dtap = saved_dtaps.at(v);

    auto check_noop = [&](const PersonaProfile& prof) {
      rt.set_persona(&prof);
      std::vector<float> pf, dec;
      run_pf_and_dec(pf, dec);
      return pf == base_pf && dec == base_dec;
    };

    // 1) force bias=0, gain=1
    {
      PersonaProfile p;
      p.name = "noop_force_b0_v" + std::to_string(v);
      p.fingerprint_sha256 = sha256_hex(p.name);
      p.cache_key = cache_key_from_hex(p.fingerprint_sha256);
      InterventionSpec iv;
      iv.type = InterventionType::RouterForce;
      iv.visit = v;
      iv.expert = active_e;
      iv.bias = 0.0f;
      iv.gain = 1.0f;
      p.interventions.push_back(iv);
      all6_noop_ok &= check_noop(p);
    }
    // 2) ablate never-selected expert
    if (never_e >= 0) {
      PersonaProfile p;
      p.name = "noop_ablate_never_v" + std::to_string(v);
      p.fingerprint_sha256 = sha256_hex(p.name);
      p.cache_key = cache_key_from_hex(p.fingerprint_sha256);
      InterventionSpec iv;
      iv.type = InterventionType::RouterAblate;
      iv.visit = v;
      iv.expert = never_e;
      p.interventions.push_back(iv);
      all6_noop_ok &= check_noop(p);
    }
    // 3) slot permute identity
    {
      PersonaProfile p;
      p.name = "noop_perm_v" + std::to_string(v);
      p.fingerprint_sha256 = sha256_hex(p.name);
      p.cache_key = cache_key_from_hex(p.fingerprint_sha256);
      InterventionSpec iv;
      iv.type = InterventionType::RouterSlotPermute;
      iv.visit = v;
      p.interventions.push_back(iv);
      all6_noop_ok &= check_noop(p);
    }
    // 4) self route replay (recompute + copy)
    for (bool copy_w : {false, true}) {
      PersonaProfile p;
      p.name = "noop_replay_" + std::string(copy_w ? "copy_" : "recomp_") + std::to_string(v);
      p.fingerprint_sha256 = sha256_hex(p.name);
      p.cache_key = cache_key_from_hex(p.fingerprint_sha256);
      InterventionSpec iv;
      iv.type = InterventionType::RouteReplay;
      iv.visit = v;
      iv.replay_copy_weights = copy_w;
      iv.inline_replay_ids = rtap.idx;
      if (copy_w) iv.inline_replay_wts = rtap.wts;
      p.interventions.push_back(iv);
      all6_noop_ok &= check_noop(p);
    }
    // 5) 3b neuron affine identity (gain=1, offset=0)
    {
      PersonaProfile p;
      p.name = "noop_3b_v" + std::to_string(v);
      p.fingerprint_sha256 = sha256_hex(p.name);
      p.cache_key = cache_key_from_hex(p.fingerprint_sha256);
      InterventionSpec iv;
      iv.type = InterventionType::NeuronAffine;
      iv.visit = v;
      iv.expert = active_e;
      iv.neuron = 17;
      iv.gain = 1.0f;
      iv.offset = 0.0f;
      p.interventions.push_back(iv);
      all6_noop_ok &= check_noop(p);
    }
    // 6) self depth replace (ffn_res, mode=both)
    {
      PersonaProfile p;
      p.name = "noop_depth_v" + std::to_string(v);
      p.fingerprint_sha256 = sha256_hex(p.name);
      p.cache_key = cache_key_from_hex(p.fingerprint_sha256);
      InterventionSpec iv;
      iv.type = InterventionType::DepthReplace;
      iv.visit = v;
      iv.depth_site = DepthSiteKind::FfnRes;
      iv.depth_mode = 3u;
      iv.inline_depth_nsrc = dtap.n_src;
      iv.inline_depth_nq = dtap.n_q;
      iv.inline_depth_scores = dtap.scores;
      iv.inline_depth_states = dtap.states;
      p.interventions.push_back(iv);
      all6_noop_ok &= check_noop(p);
    }
    // 7) residual steer scale=0
    {
      PersonaProfile p;
      p.name = "noop_steer_v" + std::to_string(v);
      p.fingerprint_sha256 = sha256_hex(p.name);
      p.cache_key = cache_key_from_hex(p.fingerprint_sha256);
      InterventionSpec iv;
      iv.type = InterventionType::ResidualSteer;
      iv.visit = v;
      iv.steer_site = SteerSiteKind::MlpWrite;
      iv.scale = 0.0f;
      iv.vector.assign(HIDDEN, 0.25f);
      iv.vector[kForbiddenCoord] = 0.0f;
      p.interventions.push_back(iv);
      all6_noop_ok &= check_noop(p);
    }
  }
  ok &= report(all6_noop_ok,
               "all 7 hook types are exact-0.0 bit-identical no-ops across all 6 visits {3, 6, 10, 18, 29, 44}");

  // Active 3b neuron affine + simultaneous 3a tap verification
  {
    const auto& rtap10 = saved_rtaps.at(10);
    const int active_e = rtap10.idx[0];
    PersonaProfile p;
    p.name = "active_3b_3a";
    p.fingerprint_sha256 = sha256_hex(p.name);
    p.cache_key = cache_key_from_hex(p.fingerprint_sha256);
    InterventionSpec iv;
    iv.type = InterventionType::NeuronAffine;
    iv.visit = 10;
    iv.expert = active_e;
    iv.neuron = 17;
    iv.gain = 3.0f;
    iv.offset = 2.5f;
    p.interventions.push_back(iv);
    PersonaTapConfig tc;
    tc.expert_taps.push_back({10, active_e, true});
    rt.set_persona(&p, tc);
    std::vector<float> pf, dec;
    run_pf_and_dec(pf, dec);
    const auto ns = rt.persona_runtime().read_neuron_stats(s);
    const auto et = rt.persona_runtime().read_expert_taps(s);
    const bool match_3a = ns.size() == 1 && et.targets.size() == 1 && ns[0].hits == et.targets[0].hits &&
                          ns[0].hits > 0 && std::abs(ns[0].post_mean - et.targets[0].mean[17]) < 2e-2f &&
                          pf != base_pf;
    ok &= report(match_3a, "active 3b neuron affine changes logits and matches simultaneous 3a tap");
  }

  // Cross-request cache-hit & parked-conversation persona isolation via EngineGenerator
  {
    RuntimeOptions gopt;
    gopt.max_ctx = 2048;
    gopt.kv_fp8 = true;
    gopt.prefill_chunk = 64;
    gopt.enable_persona_hooks = true;
    gopt.park_mb = 256;
    cudaStream_t s1;
    INGINE_TEST_CUDA(cudaStreamCreate(&s1));
    Weights wg = synthetic_weights(s1);
    INGINE_TEST_CUDA(cudaStreamDestroy(s1));
    const auto gen = make_generator(gopt, std::move(wg), load_ablation({}));

    auto prof_a = std::make_shared<PersonaProfile>();
    prof_a->name = "A";
    prof_a->fingerprint_sha256 = sha256_hex("A");
    prof_a->cache_key = cache_key_from_hex(prof_a->fingerprint_sha256);
    {
      InterventionSpec iv;
      iv.type = InterventionType::RouterForce;
      iv.visit = 10;
      iv.expert = 5;
      iv.bias = 100.0f;
      iv.gain = 4.0f;
      prof_a->interventions.push_back(iv);
    }

    Rng rng(17);
    std::vector<std::int32_t> long_p(300);
    for (auto& t : long_p) t = std::int32_t(rng.next() % kVocab);
    auto run_g = [&](const std::vector<std::int32_t>& ids, std::shared_ptr<const PersonaProfile> p) {
      Sampling sp;
      sp.temperature = 0.f;
      sp.persona = std::move(p);
      std::vector<std::int32_t> out;
      const GenerateStats st = gen->generate(ids, 16, sp, {}, [&](std::span<const std::int32_t> t) {
        out.insert(out.end(), t.begin(), t.end());
        return true;
      });
      return std::pair{out, st};
    };
    const auto [b_alone, st_b0] = run_g(long_p, nullptr);
    const auto [a_out, st_a] = run_g(long_p, prof_a);
    const auto [b_after_a, st_b1] = run_g(long_p, nullptr);
    // Notice: b_alone parked under key=0; a_out did NOT reuse b_alone (Reuse::None);
    // b_after_a unparked b_alone (Reuse::Parked) and generated bit-identical tokens to b_alone!
    const bool iso_ok = (a_out != b_alone) && (st_a.reuse == Reuse::None) && (b_after_a == b_alone) &&
                        (st_b1.reuse == Reuse::Parked);
    ok &= report(iso_ok, "cross-request cache & parked-conversation persona isolation");
  }
  return ok;
}

bool run_switch_error_and_fallback_tests() {
  bool ok = true;
  // Verify classifier covers 12.8+ driver/PTX SWITCH errors without #ifdef constant-masking bugs.
  const bool cls_ok = is_conditional_switch_unsupported_error(cudaErrorCallRequiresNewerDriver) &&
                      is_conditional_switch_unsupported_error(cudaErrorUnsupportedPtxVersion) &&
                      is_conditional_switch_unsupported_error(cudaErrorNotSupported) &&
                      is_conditional_switch_unsupported_error(cudaErrorInsufficientDriver) &&
                      is_conditional_switch_unsupported_error(cudaErrorNotPermitted) &&
                      is_conditional_switch_unsupported_error(cudaErrorOperatingSystem) &&
                      !is_conditional_switch_unsupported_error(cudaSuccess) &&
                      !is_conditional_switch_unsupported_error(cudaErrorMemoryAllocation) &&
                      !is_conditional_switch_unsupported_error(cudaErrorIllegalAddress) &&
                      !is_conditional_switch_unsupported_error(cudaErrorInvalidValue);
  ok &= report(cls_ok, "conditional SWITCH unsupported error classifier handles enum constants");

  struct ResetInjectGuard {
    ~ResetInjectGuard() { fail_next_switch_capture(cudaSuccess); }
  } guard;

  const bool hal_switch_ok = hal::conditional_graphs_enabled(0);
  if (!hal_switch_ok) {
    std::printf("INFO conditional graphs disabled/unsupported by HAL; skipping induced capture injection\n");
    return ok;
  }

  for (const cudaError_t injected : {cudaErrorCallRequiresNewerDriver, cudaErrorUnsupportedPtxVersion}) {
    reset_spec_execution_counters();
    RuntimeOptions opt;
    opt.max_ctx = 512;
    opt.prefill_chunk = 64;
    opt.kv_fp8 = true;
    opt.spec = true;
    cudaStream_t s0 = nullptr;
    INGINE_TEST_CUDA(cudaStreamCreate(&s0));
    Weights w = synthetic_weights(s0);
    Runtime rt(opt, std::move(w), load_ablation({}));
    INGINE_TEST_CUDA(cudaStreamDestroy(s0));

    DeviceBuffer width(sizeof(std::int32_t));
    const std::span<const int> vw = rt.verify_widths();
    Widths widths;
    widths.n = int(vw.size());
    std::copy(vw.begin(), vw.end(), widths.w);

    cudaStream_t captured_stream = nullptr;
    fail_next_switch_capture(injected);
    GraphExec exec = rt.try_capture_switch(
        [&](cudaStream_t cs, cudaGraphConditionalHandle h) {
          captured_stream = cs;
          return width_select(static_cast<const std::int32_t*>(width.p), widths, h, cs);
        },
        [&](cudaStream_t cs, int m) {
          (void)cs;
          (void)m;
        });

    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    const cudaError_t cap_q = captured_stream ? cudaStreamIsCapturing(captured_stream, &cap_status) : cudaSuccess;
    const SpecExecutionCounters cnt = spec_execution_counters();
    const bool probe_after = !exec && !rt.supports_conditional_graphs() && !rt.has_verify_graph() &&
                             cnt.switch_capture_fallbacks == 1 && cap_q == cudaSuccess &&
                             cap_status == cudaStreamCaptureStatusNone && cudaGetLastError() == cudaSuccess;
    reset_state(rt.state(), rt.stream());
    rt.decode_logits(1, 0);
    const bool eager_ok = finite(logits_row(rt, 1, 0));
    ok &= report(probe_after && eager_ok,
                 std::string("induced SWITCH failure after HAL probe falls back cleanly (") +
                     cudaGetErrorName(injected) + ")");
  }

  // Non-recoverable CUDA errors must still surface rather than downgrading to eager fallback.
  {
    reset_spec_execution_counters();
    RuntimeOptions opt;
    opt.max_ctx = 512;
    opt.prefill_chunk = 64;
    opt.kv_fp8 = true;
    opt.spec = true;
    cudaStream_t s0 = nullptr;
    INGINE_TEST_CUDA(cudaStreamCreate(&s0));
    Weights w = synthetic_weights(s0);
    Runtime rt(opt, std::move(w), load_ablation({}));
    INGINE_TEST_CUDA(cudaStreamDestroy(s0));

    DeviceBuffer width(sizeof(std::int32_t));
    Widths widths;
    const std::span<const int> vw = rt.verify_widths();
    widths.n = int(vw.size());
    std::copy(vw.begin(), vw.end(), widths.w);

    bool threw = false;
    fail_next_switch_capture(cudaErrorMemoryAllocation);
    try {
      (void)rt.try_capture_switch(
          [&](cudaStream_t cs, cudaGraphConditionalHandle h) {
            return width_select(static_cast<const std::int32_t*>(width.p), widths, h, cs);
          },
          [&](cudaStream_t cs, int m) {
            (void)cs;
            (void)m;
          });
    } catch (const std::runtime_error&) {
      threw = true;
    }
    (void)cudaGetLastError();
    const SpecExecutionCounters cnt = spec_execution_counters();
    ok &= report(threw && rt.supports_conditional_graphs() && cnt.switch_capture_fallbacks == 0,
                 "non-recoverable CUDA error during try_capture_switch rethrows");
  }
  return ok;
}

template <class Ex, class Fn>
bool expect_throws(Fn&& fn) {
  try {
    fn();
  } catch (const Ex&) {
    return true;
  }
  return false;
}

bool run_batch_admission_and_q4() {
  bool ok = true;
  RuntimeOptions opt;
  opt.max_ctx = 512;
  opt.prefill_chunk = 64;
  opt.kv_fp8 = true;
  opt.max_batch = 4;
  opt.max_kv_tokens = 2048;

  const std::vector<std::int32_t> p0 = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<std::int32_t> p1 = {9, 10, 11, 12, 13, 14, 15, 16};
  const std::vector<std::int32_t> bad_high = {1, VOCAB};
  const std::vector<std::int32_t> bad_neg = {1, -1};
  const std::vector<std::int32_t> forced_bad_high = {5, VOCAB};
  const std::vector<std::int32_t> forced_bad_neg = {5, -1};
  Sampling greedy;
  greedy.temperature = 0.f;

  auto collect_gen = [](Generator& g, std::span<const std::int32_t> prompt, int max_new, const Sampling& sp) {
    std::vector<std::int32_t> out;
    g.generate(prompt, max_new, sp, {}, [&](std::span<const std::int32_t> t) {
      out.insert(out.end(), t.begin(), t.end());
      return true;
    });
    return out;
  };

  // Capture clean reference outputs first so peak VRAM never holds two full synthetic models at once.
  std::int32_t ref_f0 = -1, ref_f1 = -1;
  std::vector<std::int32_t> ref_step(2, -1);
  {
    cudaStream_t s0 = nullptr;
    INGINE_TEST_CUDA(cudaStreamCreate(&s0));
    Weights w = synthetic_weights(s0);
    INGINE_TEST_CUDA(cudaStreamDestroy(s0));
    auto bref = make_generator(opt, std::move(w), load_ablation({}));
    GenerateStats st0, st1;
    const int r0 = bref->start_slot(Prompt{p0, {}}, 8, greedy, ref_f0, st0);
    const int r1 = bref->start_slot(Prompt{p1, {}}, 8, greedy, ref_f1, st1);
    const BatchSlotStep ref_steps[2] = {{r0, ref_f0}, {r1, ref_f1}};
    bref->step_slots(ref_steps, ref_step);
    bref->release_slot(r0);
    bref->release_slot(r1);
  }

  // Test BatchGenerator admission via make_generator(max_batch > 1).
  {
    cudaStream_t s0 = nullptr;
    INGINE_TEST_CUDA(cudaStreamCreate(&s0));
    Weights w = synthetic_weights(s0);
    INGINE_TEST_CUDA(cudaStreamDestroy(s0));
    auto bgen = make_generator(opt, std::move(w), load_ablation({}));
    bool batch_check = bgen->max_batch_size() == 4;

    Sampling bad_fhi = greedy;
    bad_fhi.forced = forced_bad_high;
    Sampling bad_flo = greedy;
    bad_flo.forced = forced_bad_neg;

    batch_check &= expect_throws<std::out_of_range>([&] { (void)collect_gen(*bgen, bad_high, 4, greedy); });
    batch_check &= expect_throws<std::invalid_argument>([&] { (void)collect_gen(*bgen, bad_neg, 4, greedy); });
    batch_check &= expect_throws<std::out_of_range>([&] { (void)collect_gen(*bgen, p0, 4, bad_fhi); });
    batch_check &= expect_throws<std::out_of_range>([&] { (void)collect_gen(*bgen, p0, 4, bad_flo); });
    const auto post_gen = collect_gen(*bgen, p0, 4, greedy);
    batch_check &= post_gen.size() == 4 && post_gen[0] == ref_f0 && post_gen[0] >= 0 && post_gen[0] < kVocab;
    ok &= report(batch_check, "BatchGenerator::generate rejects OOB prompt/forced tokens and recovers cleanly");

    // start_slot rejection must leak zero slots (all max_batch slots remain available afterwards).
    std::int32_t dummy_tok = -1;
    GenerateStats dummy_st;
    bool slot_check = expect_throws<std::out_of_range>(
        [&] { (void)bgen->start_slot(Prompt{bad_high, {}}, 4, greedy, dummy_tok, dummy_st); });
    slot_check &= expect_throws<std::invalid_argument>(
        [&] { (void)bgen->start_slot(Prompt{bad_neg, {}}, 4, greedy, dummy_tok, dummy_st); });
    slot_check &= expect_throws<std::out_of_range>(
        [&] { (void)bgen->start_slot(Prompt{p0, {}}, 4, bad_fhi, dummy_tok, dummy_st); });
    slot_check &= expect_throws<std::out_of_range>(
        [&] { (void)bgen->start_slot(Prompt{p0, {}}, 4, bad_flo, dummy_tok, dummy_st); });
    std::vector<int> all_slots;
    for (int i = 0; i < 4; ++i) {
      std::int32_t tok = -1;
      GenerateStats st;
      const int s = bgen->start_slot(Prompt{p0, {}}, 4, greedy, tok, st);
      slot_check &= (s >= 0 && tok == ref_f0);
      if (s >= 0) all_slots.push_back(s);
    }
    slot_check &= bgen->start_slot(Prompt{p0, {}}, 4, greedy, dummy_tok, dummy_st) == -1;
    for (int s : all_slots) bgen->release_slot(s);
    ok &= report(slot_check, "BatchGenerator::start_slot rejects OOB tokens without leaking slots");

    // step_slots must validate the whole batch before mutating any slot state.
    std::int32_t f0 = -1, f1 = -1;
    GenerateStats s0_st, s1_st;
    const int s0_id = bgen->start_slot(Prompt{p0, {}}, 8, greedy, f0, s0_st);
    const int s1_id = bgen->start_slot(Prompt{p1, {}}, 8, greedy, f1, s1_st);
    bool step_atomic = (s0_id >= 0 && s1_id >= 0 && f0 == ref_f0 && f1 == ref_f1);
    std::vector<std::int32_t> out_step(2, -999);
    const BatchSlotStep bad_hi_step[2] = {{s0_id, f0}, {s1_id, VOCAB}};
    const BatchSlotStep bad_lo_step[2] = {{s0_id, f0}, {s1_id, -1}};
    const BatchSlotStep dup_slot_step[2] = {{s0_id, f0}, {s0_id, f0}};
    const BatchSlotStep bad_slot_step[2] = {{s0_id, f0}, {99, f1}};
    step_atomic &= expect_throws<std::out_of_range>([&] { bgen->step_slots(bad_hi_step, out_step); });
    step_atomic &= expect_throws<std::out_of_range>([&] { bgen->step_slots(bad_lo_step, out_step); });
    step_atomic &= expect_throws<std::invalid_argument>([&] { bgen->step_slots(dup_slot_step, out_step); });
    step_atomic &= expect_throws<std::invalid_argument>([&] { bgen->step_slots(bad_slot_step, out_step); });

    const BatchSlotStep good_step[2] = {{s0_id, f0}, {s1_id, f1}};
    bgen->step_slots(good_step, out_step);
    step_atomic &= (out_step == ref_step);
    bgen->release_slot(s0_id);
    bgen->release_slot(s1_id);
    const auto after_atomic = collect_gen(*bgen, p1, 4, greedy);
    step_atomic &= after_atomic.size() == 4 && after_atomic[0] == ref_f1;
    ok &= report(step_atomic, "BatchGenerator::step_slots validates atomically and preserves clean recovery");
  }

  // Packed Q4_0 embeddings through make_generator(max_batch=1) and make_generator(max_batch=4)
  // vs bit-exact dequantized BF16 BatchGenerator reference.
  {
    const std::vector<std::int32_t> qp0 = {1, 2, 3, 4, 5, 6, 7, 4090};
    const std::vector<std::int32_t> qp1 = {9, 10, 11, 12, 13, 14, 15, 4091};

    std::vector<std::int32_t> q4_b1_p0, q4_b1_p1;
    {
      RuntimeOptions opt1 = opt;
      opt1.max_batch = 1;
      opt1.max_kv_tokens = 0;
      cudaStream_t s0 = nullptr;
      INGINE_TEST_CUDA(cudaStreamCreate(&s0));
      Weights w = synthetic_weights_q4_equiv(s0, /*packed_q4=*/true);
      INGINE_TEST_CUDA(cudaStreamDestroy(s0));
      auto gen1 = make_generator(opt1, std::move(w), load_ablation({}));
      q4_b1_p0 = collect_gen(*gen1, qp0, 5, greedy);
      q4_b1_p1 = collect_gen(*gen1, qp1, 5, greedy);
    }

    std::vector<std::int32_t> bf16_b4_p0, bf16_b4_p1, bf16_got0, bf16_got1;
    {
      cudaStream_t s0 = nullptr;
      INGINE_TEST_CUDA(cudaStreamCreate(&s0));
      Weights wb = synthetic_weights_q4_equiv(s0, /*packed_q4=*/false);
      INGINE_TEST_CUDA(cudaStreamDestroy(s0));
      auto bgen_bf16 = make_generator(opt, std::move(wb), load_ablation({}));
      bf16_b4_p0 = collect_gen(*bgen_bf16, qp0, 5, greedy);
      bf16_b4_p1 = collect_gen(*bgen_bf16, qp1, 5, greedy);

      std::int32_t f0 = -1, f1 = -1;
      GenerateStats st0, st1;
      const int s0_id = bgen_bf16->start_slot(Prompt{qp0, {}}, 6, greedy, f0, st0);
      const int s1_id = bgen_bf16->start_slot(Prompt{qp1, {}}, 6, greedy, f1, st1);
      bf16_got0 = {f0};
      bf16_got1 = {f1};
      std::vector<std::int32_t> step_out(2, -1);
      for (int k = 1; k < 5; ++k) {
        const BatchSlotStep cur[2] = {{s0_id, bf16_got0.back()}, {s1_id, bf16_got1.back()}};
        bgen_bf16->step_slots(cur, step_out);
        bf16_got0.push_back(step_out[0]);
        bf16_got1.push_back(step_out[1]);
      }
      bgen_bf16->release_slot(s0_id);
      bgen_bf16->release_slot(s1_id);
    }

    cudaStream_t s0 = nullptr;
    INGINE_TEST_CUDA(cudaStreamCreate(&s0));
    Weights w4 = synthetic_weights_q4_equiv(s0, /*packed_q4=*/true);
    INGINE_TEST_CUDA(cudaStreamDestroy(s0));
    auto bgen_q4 = make_generator(opt, std::move(w4), load_ablation({}));
    const auto q4_b4_p0 = collect_gen(*bgen_q4, qp0, 5, greedy);
    const auto q4_b4_p1 = collect_gen(*bgen_q4, qp1, 5, greedy);

    bool q4_ok = true;
    auto check_sub = [&](bool cond, const char* msg) {
      if (!cond) std::fprintf(stderr, "Q4_0 batch subcheck failed: %s\n", msg);
      q4_ok &= cond;
    };
    check_sub(q4_b1_p0.size() == 5 && q4_b1_p1.size() == 5 && q4_b4_p0.size() == 5 && q4_b4_p1.size() == 5,
              "sequence lengths == 5");
    check_sub(q4_b4_p0 == bf16_b4_p0 && q4_b4_p1 == bf16_b4_p1,
              "BatchGenerator packed Q4_0 matches exact-dequantized BF16 ref solo");
    check_sub(!q4_b4_p0.empty() && !q4_b1_p0.empty() && q4_b4_p0[0] == q4_b1_p0[0] &&
                  !q4_b4_p1.empty() && !q4_b1_p1.empty() && q4_b4_p1[0] == q4_b1_p1[0],
              "Q4_0 first token matches B=1 EngineGenerator prefill");
    bool in_vocab = true;
    for (auto t : q4_b4_p0) in_vocab &= (t >= 0 && t < kVocab);
    for (auto t : q4_b4_p1) in_vocab &= (t >= 0 && t < kVocab);
    check_sub(in_vocab, "generated Q4_0 tokens in [0, kVocab)");

    // Concurrent B=2 batched Q4_0 decode across start_slot + step_slots + post-OOB recovery.
    std::int32_t c0 = -1, c1 = -1;
    GenerateStats cs0, cs1;
    const int s0_id = bgen_q4->start_slot(Prompt{qp0, {}}, 6, greedy, c0, cs0);
    const int s1_id = bgen_q4->start_slot(Prompt{qp1, {}}, 6, greedy, c1, cs1);
    check_sub(s0_id >= 0 && s1_id >= 0 && c0 == q4_b4_p0[0] && c1 == q4_b4_p1[0], "start_slot Q4_0 first tokens");
    std::vector<std::int32_t> q4_got0 = {c0}, q4_got1 = {c1};
    std::vector<std::int32_t> step_out(2, -1);
    const BatchSlotStep bad_step[2] = {{s0_id, c0}, {s1_id, VOCAB}};
    check_sub(expect_throws<std::out_of_range>([&] { bgen_q4->step_slots(bad_step, step_out); }),
              "Q4_0 step_slots rejects OOB");
    for (int k = 1; k < 5; ++k) {
      const BatchSlotStep cur[2] = {{s0_id, q4_got0.back()}, {s1_id, q4_got1.back()}};
      bgen_q4->step_slots(cur, step_out);
      q4_got0.push_back(step_out[0]);
      q4_got1.push_back(step_out[1]);
    }
    bgen_q4->release_slot(s0_id);
    bgen_q4->release_slot(s1_id);
    check_sub(q4_got0 == bf16_got0 && q4_got1 == bf16_got1,
              "concurrent B=2 Q4_0 step_slots matches exact-dequantized BF16 ref");

    // Replicate invariance on Q4_0 B=2 concurrent decode.
    std::int32_t rp0 = -1, rp1 = -1;
    GenerateStats rs0, rs1;
    const int r0_id = bgen_q4->start_slot(Prompt{qp0, {}}, 6, greedy, rp0, rs0);
    const int r1_id = bgen_q4->start_slot(Prompt{qp0, {}}, 6, greedy, rp1, rs1);
    std::vector<std::int32_t> rep0 = {rp0}, rep1 = {rp1};
    for (int k = 1; k < 5; ++k) {
      const BatchSlotStep cur[2] = {{r0_id, rep0.back()}, {r1_id, rep1.back()}};
      bgen_q4->step_slots(cur, step_out);
      rep0.push_back(step_out[0]);
      rep1.push_back(step_out[1]);
    }
    bgen_q4->release_slot(r0_id);
    bgen_q4->release_slot(r1_id);
    check_sub(rep0 == rep1 && rep0.size() == 5, "Q4_0 concurrent replicates match identically");

    check_sub(expect_throws<std::out_of_range>([&] { (void)collect_gen(*bgen_q4, bad_high, 4, greedy); }),
              "Q4_0 generate rejects OOB");
    check_sub(collect_gen(*bgen_q4, qp0, 5, greedy) == q4_b4_p0, "Q4_0 post-OOB recovery generate matches");
    ok &= report(q4_ok, "BatchGenerator dispatches packed Q4_0 embed with B=1/B>1 parity and OOB admission");
  }
  return ok;
}

// End-to-end EngineGenerator disk KV checkpoint save + same-dtype explicit/auto resume + mixed-dtype rejection/decline
// on the synthetic model (runs on both SM80+ BF16 and SM70 FP16 builds without requiring the full 22 GB checkpoint).
bool run_disk_checkpoint_dtype() {
  bool ok = true;
  const std::string tag = (kActDType == DType::FP16) ? " (FP16)" : " (BF16)";
  const std::filesystem::path tmp_dir =
      std::filesystem::temp_directory_path() / ("ingine_rt_ckpt_dtype_" + std::to_string(kActIsFP16 ? 16 : 80));
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path ckpt_same = tmp_dir / "same.bin";
  const std::filesystem::path ckpt_mixed = tmp_dir / "mixed.bin";
  const std::filesystem::path ckpt_fp32 = tmp_dir / "fp32_bogus.bin";
  const std::filesystem::path ckpt_legacy = tmp_dir / "legacy.bin";
  const std::filesystem::path ckpt_legacy_desc = tmp_dir / "legacy_desc_96.bin";
  const std::filesystem::path ckpt_unverified = tmp_dir / "unverified_given.bin";
  const std::filesystem::path ckpt_stamped = tmp_dir / "stamped_verified.bin";
  const std::filesystem::path ckpt_stamped_bad_prompt = tmp_dir / "stamped_bad_prompt.bin";
  std::filesystem::remove(ckpt_same);
  std::filesystem::remove(ckpt_mixed);
  std::filesystem::remove(ckpt_fp32);
  std::filesystem::remove(ckpt_legacy);
  std::filesystem::remove(ckpt_legacy_desc);
  std::filesystem::remove(ckpt_unverified);
  std::filesystem::remove(ckpt_stamped);
  std::filesystem::remove(ckpt_stamped_bad_prompt);

  auto patch_act_dtype = [](const std::filesystem::path& src, const std::filesystem::path& dst,
                            std::uint32_t new_act_dtype) {
    std::ifstream in(src, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open checkpoint: " + src.string());
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() < 4096) throw std::runtime_error("checkpoint smaller than 4096-byte header");
    // RawHeader::act_dtype sits at byte offset 16 (8-byte magic + 4-byte version + 4-byte header_bytes).
    std::memcpy(bytes.data() + 16, &new_act_dtype, sizeof(new_act_dtype));
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write patched checkpoint: " + dst.string());
    out.write(bytes.data(), std::streamsize(bytes.size()));
    out.close();
  };

  auto strip_producer_trailer = [](const std::filesystem::path& src, const std::filesystem::path& dst) {
    std::ifstream in(src, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open checkpoint: " + src.string());
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() < 4096) throw std::runtime_error("checkpoint smaller than 4096-byte header");
    // Legacy INGCKPT1 files zero bytes 144..4095 (no v1 producer trailer).
    std::memset(bytes.data() + 144, 0, 4096 - 144);
    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write legacy checkpoint: " + dst.string());
    out.write(bytes.data(), std::streamsize(bytes.size()));
    out.close();
  };

  const std::array<std::uint8_t, 32> synth_verified_sha = sha256_bytes("SYNTH_VERIFIED_TEST_WEIGHTS_77", 30);
  auto make_synth_gen = [&](const RuntimeOptions& opt, bool set_sha = false, const std::filesystem::path& abl = {}) {
    cudaStream_t s;
    INGINE_TEST_CUDA(cudaStreamCreate(&s));
    Weights w = synthetic_weights(s);
    INGINE_TEST_CUDA(cudaStreamDestroy(s));
    if (set_sha) w.content_sha256 = synth_verified_sha;
    return make_generator(opt, std::move(w), load_ablation(abl));
  };

  Rng rng(91);
  std::vector<std::int32_t> prompt(144);
  for (auto& t : prompt) t = std::int32_t(rng.next() % kVocab);
  const std::span<const std::int32_t> prompt100(prompt.data(), 100);  // saves only at pos=64 when kv_ckpt_every=64
  std::vector<std::int32_t> sibling = prompt;
  sibling[96] = (sibling[96] + 1) % kVocab;  // diverges at R = 96 (> pos=64)
  Sampling greedy;
  greedy.temperature = 0.f;

  auto run_once = [&](Generator& gen, std::span<const std::int32_t> ids) {
    std::vector<std::int32_t> out;
    const GenerateStats st = gen.generate(ids, 16, greedy, {}, [&](std::span<const std::int32_t> t) {
      out.insert(out.end(), t.begin(), t.end());
      return true;
    });
    return std::pair{out, st};
  };

  RuntimeOptions base;
  base.max_ctx = 2048;
  base.kv_fp8 = true;
  base.prefill_chunk = 64;
  base.park_mb = 0;
  base.park_host_mb = 0;

  // 0. Caller-supplied given-weights overload MUST stay unverified even if w.content_sha256 is set
  //    (saves !has_producer_identity and declines auto-resume).
  for (bool set_sha : {false, true}) {
    std::filesystem::remove(ckpt_unverified);
    RuntimeOptions opt = base;
    opt.kv_ckpt_path = ckpt_unverified;
    opt.kv_ckpt_every = 64;
    auto gen = make_synth_gen(opt, set_sha);
    (void)run_once(*gen, prompt100);
    const KvCheckpointInfo ui = inspect_kv_checkpoint(ckpt_unverified);

    RuntimeOptions aopt = base;
    aopt.kv_ckpt_path = ckpt_unverified;
    aopt.kv_ckpt_every = 0;
    auto gen2 = make_synth_gen(aopt, set_sha);
    const auto [toks2, st2] = run_once(*gen2, prompt100);
    ok &= report(ui.pos == 64 && !ui.has_producer_identity && st2.reuse == Reuse::None && st2.reused_tokens == 0 &&
                     toks2.size() == 16,
                 std::string("given-weights overload stays unverified (set_sha=") + (set_sha ? "true" : "false") + ")" +
                     tag);
  }

  // 1. Save real disk checkpoint at pos = 64 (< 100, with boundary logits) and record greedy reference for 144-tok prompt
  std::vector<std::int32_t> ref_tokens;
  {
    RuntimeOptions opt = base;
    opt.kv_ckpt_path = ckpt_same;
    opt.kv_ckpt_every = 64;
    auto gen = make_synth_gen(opt);
    (void)run_once(*gen, prompt100);
  }
  {
    auto gen = make_synth_gen(base);
    const auto [toks, st] = run_once(*gen, prompt);
    ref_tokens = toks;
  }
  const KvCheckpointInfo saved_info = inspect_kv_checkpoint(ckpt_same);
  const std::uint32_t expected_disk_dtype = static_cast<std::uint32_t>(kActDType);
  ok &= report(saved_info.pos == 64 && saved_info.has_logits && !saved_info.has_producer_identity &&
                   saved_info.act_dtype == expected_disk_dtype,
               "disk checkpoint writes raw kActDType (" + std::to_string(expected_disk_dtype) + ") at pos=64" + tag);

  // 2. Same-dtype explicit --kv-ckpt-resume on unstamped checkpoint MUST succeed and reproduce exact greedy tokens
  //    (including deliberate max_ctx=4096).
  {
    RuntimeOptions opt = base;
    opt.max_ctx = 4096;  // deliberate compatible case: pos=64 <= max_ctx
    opt.kv_ckpt_resume = ckpt_same;
    auto gen = make_synth_gen(opt);
    const auto [toks, st] = run_once(*gen, prompt);
    ok &= report(st.reuse == Reuse::Checkpoint && st.reused_tokens == 64 && toks == ref_tokens,
                 "same-dtype explicit --kv-ckpt-resume succeeds with exact parity (deliberate max_ctx=4096)" + tag);
  }

  // 2b. Stamped checkpoint on unknown caller-supplied given-weights reader MUST throw BEFORE import (both set_sha=false
  //     and set_sha=true), even when prompt_sha256 is zeroed, leaving live state clean for cold fallback.
  {
    auto stamp_producer = [&](const std::filesystem::path& src, const std::filesystem::path& dst, bool zero_prompt_sha) {
      std::ifstream in(src, std::ios::binary);
      if (!in) throw std::runtime_error("cannot open checkpoint: " + src.string());
      std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
      if (bytes.size() < 4096) throw std::runtime_error("checkpoint smaller than 4096-byte header");
      if (zero_prompt_sha) std::memset(bytes.data() + 80, 0, 32);  // RawHeader::prompt_sha256
      const std::uint32_t flags = 1u;
      const std::uint32_t dfp8 = 1u;
      const std::int32_t chunk = 64;
      const std::uint32_t resv = 0u;
      std::memcpy(bytes.data() + 144, &flags, 4);
      std::memcpy(bytes.data() + 148, &dfp8, 4);
      std::memcpy(bytes.data() + 152, &chunk, 4);
      std::memcpy(bytes.data() + 156, &resv, 4);
      std::memcpy(bytes.data() + 160, synth_verified_sha.data(), 32);
      std::ofstream out(dst, std::ios::binary | std::ios::trunc);
      if (!out) throw std::runtime_error("cannot write stamped checkpoint: " + dst.string());
      out.write(bytes.data(), std::streamsize(bytes.size()));
    };
    stamp_producer(ckpt_same, ckpt_stamped, false);
    stamp_producer(ckpt_same, ckpt_stamped_bad_prompt, true);
    const KvCheckpointInfo si = inspect_kv_checkpoint(ckpt_stamped);
    bool all_rejected_pre_import = si.has_producer_identity;
    for (bool set_sha : {false, true}) {
      for (const auto& spath : {ckpt_stamped, ckpt_stamped_bad_prompt}) {
        RuntimeOptions ropt = base;
        ropt.kv_ckpt_resume = spath;
        auto gen = make_synth_gen(ropt, set_sha);
        bool threw_producer = false;
        try {
          (void)run_once(*gen, prompt);
        } catch (const std::runtime_error& e) {
          threw_producer = std::string(e.what()).find("cannot verify stamped checkpoint producer") != std::string::npos;
        }
        const PrefixCacheStats cs = gen->cache_stats();
        all_rejected_pre_import &=
            (threw_producer && cs.root_entries == 0 && cs.vram_entries == 0 && cs.host_entries == 0);
      }
    }
    // Direct pre-import proof: throwing inside load_kv_checkpoint_checked's eligibility callback leaves State untouched
    // (st.length stays at its pre-call value 0 and zero payload/logits bytes are imported).
    {
      cudaStream_t s;
      INGINE_TEST_CUDA(cudaStreamCreate(&s));
      State st = make_state(base.max_ctx, true, false);
      DeviceBuffer d_logits(std::size_t(UNPADDED_VOCAB) * sizeof(float));
      d_logits.fill(0);
      bool threw_pre = false;
      try {
        (void)load_kv_checkpoint_checked(st, {}, prompt, static_cast<float*>(d_logits.p), ckpt_stamped_bad_prompt, s,
                                         [](const KvCheckpointInfo& info) -> bool {
                                           if (info.has_producer_identity) {
                                             throw std::runtime_error("pre-import reject");
                                           }
                                           return true;
                                         });
      } catch (const std::runtime_error& e) {
        threw_pre = std::string(e.what()) == "pre-import reject";
      }
      std::vector<float> h_logits(64, 1.f);
      INGINE_TEST_CUDA(cudaMemcpy(h_logits.data(), d_logits.p, h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
      bool logits_untouched = true;
      for (float v : h_logits) logits_untouched &= (v == 0.f);
      INGINE_TEST_CUDA(cudaStreamDestroy(s));
      all_rejected_pre_import &= (threw_pre && st.length == 0 && logits_untouched);
    }
    ok &= report(all_rejected_pre_import,
                 "stamped disk checkpoint rejected pre-import on given-weights unknown reader (set_sha=false/true)" +
                     tag);
  }

  // 3. Supplied active Ablation with empty options.ablation MUST block periodic save, throw on explicit, decline auto
  {
    const std::filesystem::path abl_file = tmp_dir / "supplied_abl.safetensors";
    const std::filesystem::path ckpt_abl_out = tmp_dir / "abl_should_not_exist.bin";
    std::filesystem::remove(ckpt_abl_out);
    Rng arng(31);
    auto fbytes = [](const std::vector<float>& v) {
      std::vector<std::uint8_t> b(v.size() * 4);
      std::memcpy(b.data(), v.data(), b.size());
      return b;
    };
    std::vector<float> dirs(std::size_t(NUM_VISITS) * H);
    for (auto& x : dirs) x = float(arng.normal());
    std::vector<float> wt(NUM_VISITS, 1.5f);
    ingine::test::write_safetensors(abl_file, {{"attn_dirs", "F32", {NUM_VISITS, H}, fbytes(dirs)},
                                               {"mlp_dirs", "F32", {NUM_VISITS, H}, fbytes(dirs)},
                                               {"attn_w", "F32", {NUM_VISITS}, fbytes(wt)},
                                               {"mlp_w", "F32", {NUM_VISITS}, fbytes(wt)}});

    RuntimeOptions opt = base;  // opt.ablation is EMPTY!
    opt.kv_ckpt_path = ckpt_abl_out;
    opt.kv_ckpt_every = 64;
    auto gen_save = make_synth_gen(opt, /*set_sha=*/true, abl_file);
    (void)run_once(*gen_save, prompt100);
    const bool save_blocked = !std::filesystem::exists(ckpt_abl_out);

    bool explicit_threw = false;
    {
      RuntimeOptions ropt = base;
      ropt.kv_ckpt_resume = ckpt_same;
      auto gen_exp = make_synth_gen(ropt, /*set_sha=*/true, abl_file);
      try {
        (void)run_once(*gen_exp, prompt);
      } catch (const std::invalid_argument&) {
        explicit_threw = true;
      }
    }
    bool auto_declined = false;
    {
      RuntimeOptions aopt = base;
      aopt.kv_ckpt_path = ckpt_same;
      aopt.kv_ckpt_every = 0;
      auto gen_auto = make_synth_gen(aopt, /*set_sha=*/true, abl_file);
      const auto [toks, st] = run_once(*gen_auto, prompt);
      auto_declined = (st.reuse == Reuse::None && st.reused_tokens == 0 && toks.size() == 16);
    }
    ok &= report(save_blocked && explicit_threw && auto_declined,
                 "supplied active Ablation (empty options.ablation) blocks save, throws on explicit, declines auto" +
                     tag);
  }

  // 4. Legacy checkpoint (zeroed producer trailer at pos=64):
  //    - explicit resume with R=96 and 144-token prompt MUST preserve legacy execution while isolating descendants:
  //      root_entries==0, no Extend/Checkpoint/Parked reuse, periodic pos-96 save has !has_producer_identity,
  //      and a fresh generator MUST decline auto-resume on that pos-96 descendant!
  strip_producer_trailer(ckpt_same, ckpt_legacy);
  {
    const KvCheckpointInfo li = inspect_kv_checkpoint(ckpt_legacy);
    bool explicit_isolated = false;
    {
      RuntimeOptions opt = base;
      opt.prefill_chunk = 32;  // hits done=96 cleanly after pos=64
      opt.park_mb = 256;
      opt.kv_ckpt_resume = ckpt_legacy;
      opt.kv_ckpt_path = ckpt_legacy_desc;
      opt.kv_ckpt_every = 96;
      auto gen = make_synth_gen(opt);
      std::vector<std::int32_t> out;
      const GenerateStats st =
          gen->generate(Prompt{prompt, {}, 96}, 16, greedy, {}, [&](std::span<const std::int32_t> t) {
            out.insert(out.end(), t.begin(), t.end());
            return true;
          });
      // Sibling sharing [0, 96) MUST NOT hit RootCheckpoint/Parked/Checkpoint/Extend from the legacy descendant!
      std::vector<std::int32_t> sib_out;
      const GenerateStats sib_st =
          gen->generate(Prompt{sibling, {}, 96}, 4, greedy, {}, [&](std::span<const std::int32_t> t) {
            sib_out.insert(sib_out.end(), t.begin(), t.end());
            return true;
          });
      const PrefixCacheStats cs = gen->cache_stats();
      const KvCheckpointInfo di = inspect_kv_checkpoint(ckpt_legacy_desc);
      explicit_isolated = (st.reuse == Reuse::Checkpoint && st.reused_tokens == 64 && out.size() == 16 &&
                           cs.root_entries == 0 && cs.vram_entries == 0 && cs.host_entries == 0 &&
                           sib_st.reuse == Reuse::Checkpoint && sib_st.reused_tokens == 64 &&
                           di.pos == 96 && !di.has_producer_identity);
    }
    bool desc_auto_declined = false;
    {
      RuntimeOptions opt = base;
      opt.prefill_chunk = 32;
      opt.kv_ckpt_path = ckpt_legacy_desc;
      opt.kv_ckpt_every = 0;
      auto gen = make_synth_gen(opt);
      const auto [toks, st] = run_once(*gen, prompt);
      desc_auto_declined = (st.reuse == Reuse::None && st.reused_tokens == 0 && toks.size() == 16);
    }
    ok &= report(!li.has_producer_identity && explicit_isolated && desc_auto_declined,
                 "legacy disk checkpoint & pos-96 descendant isolated from root/parked/auto-resume & producer stamp" +
                     tag);
  }

  // 5. Host-only root capture books live-to-staging D2D capture bytes into d2d_park_bytes (> 0) as well as d2h_park_bytes
  {
    RuntimeOptions opt = base;
    opt.park_mb = 0;
    opt.park_host_mb = 256;
    auto gen = make_synth_gen(opt);
    std::vector<std::int32_t> out;
    gen->generate(Prompt{prompt, {}, 64}, 4, greedy, {}, [&](std::span<const std::int32_t> t) {
      out.insert(out.end(), t.begin(), t.end());
      return true;
    });
    const PrefixCacheStats cs = gen->cache_stats();
    ok &= report(cs.root_entries == 1 && cs.host_entries == 1 && cs.vram_entries == 0 &&
                     cs.d2d_park_bytes > 0 && cs.d2h_park_bytes > 0,
                 "host-only root capture books live-to-staging d2d_park_bytes > 0 and d2h_park_bytes > 0" + tag);
  }

  // 6. Mixed-dtype (opposite BF16/FP16) & bogus FP32(1) checkpoints: explicit MUST throw, auto MUST decline cleanly
  const std::uint32_t other_disk_dtype =
      (kActDType == DType::FP16) ? static_cast<std::uint32_t>(DType::BF16) : static_cast<std::uint32_t>(DType::FP16);
  patch_act_dtype(ckpt_same, ckpt_mixed, other_disk_dtype);
  patch_act_dtype(ckpt_same, ckpt_fp32, static_cast<std::uint32_t>(DType::FP32));

  for (const auto& [label, path, want_code] :
       {std::tuple{"mixed-dtype", ckpt_mixed, other_disk_dtype},
        std::tuple{"raw-FP32(1)", ckpt_fp32, static_cast<std::uint32_t>(DType::FP32)}}) {
    const KvCheckpointInfo pi = inspect_kv_checkpoint(path);
    bool explicit_threw = false;
    {
      RuntimeOptions opt = base;
      opt.kv_ckpt_resume = path;
      auto gen = make_synth_gen(opt);
      try {
        (void)run_once(*gen, prompt);
      } catch (const std::exception&) {
        explicit_threw = true;
      }
    }
    bool auto_declined = false;
    {
      RuntimeOptions opt = base;
      opt.kv_ckpt_path = path;
      opt.kv_ckpt_every = 0;
      auto gen = make_synth_gen(opt);
      const auto [toks, st] = run_once(*gen, prompt);
      auto_declined = (st.reuse == Reuse::None && st.reused_tokens == 0 && toks == ref_tokens);
    }
    ok &= report(pi.act_dtype == want_code && explicit_threw && auto_declined,
                 std::string(label) + " disk checkpoint rejected on explicit resume & declined on auto resume" + tag);
  }

  std::error_code ec;
  std::filesystem::remove_all(tmp_dir, ec);
  return ok;
}

}  // namespace

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  std::size_t free = 0, total = 0;
  INGINE_TEST_CUDA(cudaMemGetInfo(&free, &total));
  if (free < (std::size_t(7) << 28)) {  // peak use ~1.6 GB (the full-size unembedding is 0.46 GB)
    std::printf("skip: %.2f GB of GPU memory free, need 1.75\n", double(free) / (1 << 30));
    return ingine::test::kSkip;
  }
  bool ok = run(false);
  ok &= run(true);
  ok &= run_generator();
  ok &= run_disk_checkpoint_dtype();
  ok &= run_ablation_layouts();
  ok &= run_persona_hooks();
  ok &= run_switch_error_and_fallback_tests();
  ok &= run_batch_admission_and_q4();
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}

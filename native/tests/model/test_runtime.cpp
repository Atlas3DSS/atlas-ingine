// The decode runtime on a synthetic model (random weights in the real layouts; experts shared by all layers
// and routing pinned to 8 experts, so it needs well under 2 GB of GPU memory): graph replay equals the eager
// step bit for bit; the verify widths 2/4/8 reproduce consecutive width-1 steps; the speculative rollback
// (save_verify / rollback) leaves the state a shorter call would; taps and logits are finite.
#include "common/test_util.h"

#include "kernels/dense/dense.h"
#include "model/runtime.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <string>

using namespace ingine;
using namespace ingine::model;
using ingine::test::float_to_bf16;
using ingine::test::Rng;

namespace {

constexpr int kVocab = 4096;  // unembedding rows of the synthetic model (token ids stay below)
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
    std::vector<__nv_bfloat16> h(n);
    for (auto& x : h) x = float_to_bf16(float(rng.normal() * sd));
    return rm(up(h.data(), n * 2), DType::BF16, shape);
  }
  Dense fp8(int rows, int cols, double sd) {
    const Tensor b = bf16({rows, cols}, sd);
    st.emplace_back(std::size_t(rows) * cols);
    void* q = st.back().p;
    st.emplace_back(std::size_t(rows) * 4);
    float* s = static_cast<float*>(st.back().p);
    kernels::dense::fp8_rows(static_cast<const __nv_bfloat16*>(b.data), rows, cols, q, s, stream);
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
  w.unembed = g.fp8(kVocab, H, 0.05);
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

// rows m of width-m logits ([m, UNPADDED_VOCAB] buffer, kVocab columns used)
std::vector<float> logits_row(Runtime& rt, int m, int row) {
  return host(static_cast<const float*>(rt.buffers(m).logits.data) + std::size_t(row) * kVocab, kVocab, rt.stream());
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
  const auto taps1 = host_t<__nv_bfloat16>(rt.buffers(1).taps, s);
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
    const auto taps4 = host_t<__nv_bfloat16>(rt.buffers(4).taps, s);
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
  return ok;
}

}  // namespace

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  std::size_t free = 0, total = 0;
  INGINE_TEST_CUDA(cudaMemGetInfo(&free, &total));
  if (free < (std::size_t(3) << 29)) {
    std::printf("skip: %.2f GB of GPU memory free, need 1.5\n", double(free) / (1 << 30));
    return ingine::test::kSkip;
  }
  bool ok = run(false);
  ok &= run(true);
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}

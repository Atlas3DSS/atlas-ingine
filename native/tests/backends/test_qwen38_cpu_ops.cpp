#include "backends/qwen38/qwen38_cpu_ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                               \
  do {                                                                                      \
    const double da_ = static_cast<double>(a);                                              \
    const double db_ = static_cast<double>(b);                                              \
    if (!(std::fabs(da_ - db_) <= (tol))) {                                                 \
      std::fprintf(stderr, "%s:%d: near failed: %.9g vs %.9g (diff=%.3g > tol=%.3g)\n",     \
                   __FILE__, __LINE__, da_, db_, std::fabs(da_ - db_), double(tol));        \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

namespace {

using namespace ingine::backends::qwen38;

void test_fp8_and_nvfp4() {
  CHECK(decode_e4m3fn(0x00u) == 0.0f);
  CHECK(decode_e4m3fn(0x80u) == -0.0f);
  CHECK(std::signbit(decode_e4m3fn(0x80u)));
  CHECK(std::isnan(decode_e4m3fn(0x7fu)));
  CHECK(std::isnan(decode_e4m3fn(0xffu)));
  CHECK_NEAR(decode_e4m3fn(0x7eu), 448.0f, 1e-6);
  CHECK_NEAR(decode_e4m3fn(0x01u), std::ldexp(1.0 / 8.0, -6), 1e-9);

  for (int code = 0; code < 256; ++code) {
    const auto u = static_cast<std::uint8_t>(code);
    if (u == 0x7fu || u == 0xffu) continue;
    const float v = decode_e4m3fn(u);
    CHECK(quantize_e4m3fn(v) == u);
  }
  CHECK(quantize_e4m3fn(1000.0f) == 0x7eu);
  CHECK(quantize_e4m3fn(-1000.0f) == 0xfeu);

  // 16-element NVFP4 row: low nibble = even idx, high nibble = odd idx.
  // Block scale = 1.0 (0x38), global scale2 = 0.25 -> effective scale = 0.25.
  std::vector<std::uint8_t> packed(8);
  // Nibble 4 -> +2.0, nibble 12 (8|4) -> -2.0; nibble 7 -> +6.0, nibble 9 -> -0.5.
  packed[0] = static_cast<std::uint8_t>((7u << 4) | 4u);    // [0]=+2.0*0.25=0.5, [1]=+6.0*0.25=1.5
  packed[1] = static_cast<std::uint8_t>((9u << 4) | 12u);   // [2]=-2.0*0.25=-0.5, [3]=-0.5*0.25=-0.125
  for (int i = 2; i < 8; ++i) packed[i] = 0x11u;            // +0.5*0.25=0.125
  std::vector<std::uint8_t> bscale = {0x38u};
  std::vector<float> decoded(16);
  decode_nvfp4_row(packed, bscale, 0.25f, decoded);
  CHECK_NEAR(decoded[0], 0.5f, 1e-6);
  CHECK_NEAR(decoded[1], 1.5f, 1e-6);
  CHECK_NEAR(decoded[2], -0.5f, 1e-6);
  CHECK_NEAR(decoded[3], -0.125f, 1e-6);
  for (int i = 4; i < 16; ++i) CHECK_NEAR(decoded[i], 0.125f, 1e-6);

  std::vector<float> x(16, 1.0f);
  std::vector<float> out(1, 0.0f);
  nvfp4_weight_only_linear(x, packed, bscale, 0.25f, 1, 16, out);
  CHECK_NEAR(out[0], 0.5 + 1.5 - 0.5 - 0.125 + 12.0 * 0.125, 1e-5);

  // Calibrated FP8 linear check: 4 inputs, 2 outputs.
  const std::vector<float> fx = {0.5f, -1.0f, 0.25f, 0.75f};
  const float in_scale = 0.0625f;
  const float w_scale = 0.125f;
  const std::vector<std::uint8_t> fw = {
      quantize_e4m3fn(2.0f), quantize_e4m3fn(-1.5f), quantize_e4m3fn(4.0f), quantize_e4m3fn(0.5f),
      quantize_e4m3fn(1.0f), quantize_e4m3fn(1.0f),  quantize_e4m3fn(1.0f), quantize_e4m3fn(1.0f),
  };
  std::vector<float> fout(2);
  fp8_calibrated_linear(fx, in_scale, fw, w_scale, 2, 4, fout);
  double expected0 = 0.0, expected1 = 0.0;
  for (int c = 0; c < 4; ++c) {
    const double xq = decode_e4m3fn(quantize_e4m3fn(fx[c] / in_scale)) * in_scale;
    expected0 += xq * decode_e4m3fn(fw[c]) * w_scale;
    expected1 += xq * decode_e4m3fn(fw[4 + c]) * w_scale;
  }
  CHECK_NEAR(fout[0], expected0, 1e-5);
  CHECK_NEAR(fout[1], expected1, 1e-5);
}

void test_python_math_oracles() {
  const std::vector<float> x = {1.0f, -0.5f, 2.0f, -1.5f, 0.25f, 0.75f, -1.25f, 0.5f};
  const std::vector<float> w_zero = {0.1f, -0.2f, 0.05f, 0.0f, 0.3f, -0.1f, 0.15f, -0.05f};
  const std::vector<float> exp_zero = {0.9869586f, -0.358894f, 1.8841937f, -1.3458526f,
                                       0.2916014f, 0.6056337f, -1.2897754f, 0.4261867f};
  std::vector<float> out_zero(8);
  rmsnorm_zero_centered(x, w_zero, 1e-6f, out_zero);
  for (int i = 0; i < 8; ++i) CHECK_NEAR(out_zero[i], exp_zero[i], 1e-5);

  const std::vector<float> w_gated = {1.0f, 0.8f, 1.2f, 0.9f, 1.1f, 0.7f, 1.3f, 1.0f};
  const std::vector<float> z = {0.5f, -1.0f, 1.5f, 0.0f, -0.5f, 2.0f, 0.25f, -0.25f};
  const std::vector<float> exp_gated = {0.2792462f, 0.0965215f, 2.6408034f, 0.0f,
                                        -0.0465771f, 0.8297961f, -0.2049143f, -0.0491038f};
  std::vector<float> out_gated(8);
  rmsnorm_gated_silu(x, z, w_gated, 1e-6f, out_gated);
  for (int i = 0; i < 8; ++i) CHECK_NEAR(out_gated[i], exp_gated[i], 1e-5);

  std::vector<float> q_in(16);
  for (int i = 0; i < 16; ++i) q_in[i] = -1.0f + 2.0f * static_cast<float>(i) / 15.0f;
  const std::vector<float> exp_rope = {-0.2721121f, -0.8653364f, -1.2098483f, -0.601917f,
                                       -0.4666667f, -0.3333333f, -0.2f,       -0.0666667f,
                                       0.0666667f,  0.2f,        0.3333333f,  0.4666667f,
                                       0.6f,        0.7333333f,  0.8666667f,  1.0f};
  apply_partial_rope_neox(q_in, 1, 16, 4, 7, 10000000.0f);
  for (int i = 0; i < 16; ++i) CHECK_NEAR(q_in[i], exp_rope[i], 1e-5);
}

void test_conv_and_gdn_chunk_vs_recurrent() {
  std::mt19937 rng(2026);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  auto gen = [&](std::size_t n) {
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
  };

  // Conv1D prefill vs step-by-step and prefix continuation.
  const int conv_dim = 12;
  const int ksize = 4;
  const int seq = 7;
  const auto x_seq = gen(seq * conv_dim);
  const auto conv_w = gen(conv_dim * ksize);
  std::vector<float> state_full;
  std::vector<float> out_full(seq * conv_dim);
  causal_conv1d_silu_prefill(x_seq, conv_w, seq, conv_dim, ksize, state_full, out_full);

  std::vector<float> state_split(conv_dim * ksize, 0.0f);
  std::vector<float> out_split(seq * conv_dim);
  causal_conv1d_silu_prefill(std::span<const float>(x_seq.data(), 3 * conv_dim), conv_w, 3, conv_dim, ksize,
                             state_split, std::span<float>(out_split.data(), 3 * conv_dim));
  for (int t = 3; t < seq; ++t) {
    causal_conv1d_silu_step(std::span<const float>(x_seq.data() + t * conv_dim, conv_dim), conv_w, conv_dim, ksize,
                            state_split, std::span<float>(out_split.data() + t * conv_dim, conv_dim));
  }
  for (std::size_t i = 0; i < out_full.size(); ++i) CHECK_NEAR(out_full[i], out_split[i], 1e-6);
  for (std::size_t i = 0; i < state_full.size(); ++i) CHECK_NEAR(state_full[i], state_split[i], 1e-6);

  // GDN recurrent vs chunk across multiple chunk sizes and non-zero initial state.
  const int T = 10, KH = 2, VH = 4, KD = 8, VD = 8;
  const auto q = gen(T * KH * KD);
  const auto k = gen(T * KH * KD);
  const auto v = gen(T * VH * VD);
  const auto z = gen(T * VH * VD);
  const auto a = gen(T * VH);
  const auto b = gen(T * VH);
  const auto a_log = gen(VH);
  const auto dt_bias = gen(VH);
  std::vector<float> norm_w(VD, 1.0f);
  const auto init_state = gen(VH * KD * VD);

  std::vector<float> state_rec = init_state;
  std::vector<float> out_rec(T * VH * VD);
  gdn_recurrent_prefill(q, k, v, z, a, b, a_log, dt_bias, norm_w, 1e-6f, T, KH, VH, KD, VD, state_rec, out_rec);

  for (int cs : {2, 3, 4, 8}) {
    std::vector<float> state_chk = init_state;
    std::vector<float> out_chk(T * VH * VD);
    gdn_chunk_prefill(q, k, v, z, a, b, a_log, dt_bias, norm_w, 1e-6f, T, cs, KH, VH, KD, VD, state_chk, out_chk);
    for (std::size_t i = 0; i < out_rec.size(); ++i) CHECK_NEAR(out_rec[i], out_chk[i], 1e-4);
    for (std::size_t i = 0; i < state_rec.size(); ++i) CHECK_NEAR(state_rec[i], state_chk[i], 1e-4);
  }
}

void test_mini_model_prefill_reuse_and_isolation() {
  const auto model = Qwen38MiniModel::make_synthetic(20261125u);
  const std::vector<std::int32_t> prefix = {3, 17, 42, 8, 55};
  const std::vector<std::int32_t> suffix_a = {9, 14};
  const std::vector<std::int32_t> suffix_b = {21, 6, 33};

  auto argmax_tok = [](const std::vector<float>& logits) -> std::int32_t {
    CHECK(!logits.empty());
    for (float x : logits) CHECK(std::isfinite(x));
    std::int32_t best = 0;
    for (std::size_t i = 1; i < logits.size(); ++i) {
      if (logits[i] > logits[static_cast<std::size_t>(best)]) best = static_cast<std::int32_t>(i);
    }
    return best;
  };
  auto assert_logits_byte_identical = [&](const std::vector<float>& a, const std::vector<float>& b) {
    CHECK(a.size() == b.size());
    for (float x : a) CHECK(std::isfinite(x));
    for (float x : b) CHECK(std::isfinite(x));
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
  };

  // Cold full sequence prefix + suffix_b plus 4 subsequent greedy steps.
  std::vector<std::int32_t> full_b = prefix;
  full_b.insert(full_b.end(), suffix_b.begin(), suffix_b.end());
  Qwen38MiniSession s_cold(model.config());
  const auto logits_cold = s_cold.prefill_last_logits(model, full_b);
  std::vector<std::vector<float>> cold_steps = {logits_cold};
  std::vector<std::int32_t> cold_greedy = {argmax_tok(logits_cold)};
  for (int k = 0; k < 4; ++k) {
    const auto nxt = s_cold.step(model, cold_greedy.back());
    cold_steps.push_back(nxt);
    cold_greedy.push_back(argmax_tok(nxt));
  }

  // Prefix -> branch A -> restore snapshot -> branch B + 4 greedy decode steps MUST match cold bytes bit-for-bit.
  Qwen38MiniSession s_reuse(model.config());
  (void)s_reuse.prefill_last_logits(model, prefix);
  const auto snap = s_reuse.snapshot();
  CHECK(s_reuse.position() == static_cast<int>(prefix.size()));
  CHECK(s_reuse.state_bytes() > 0);
  (void)s_reuse.prefill_last_logits(model, suffix_a);
  s_reuse.restore(snap);
  auto logits_reused = s_reuse.prefill_last_logits(model, suffix_b);
  assert_logits_byte_identical(cold_steps[0], logits_reused);
  std::vector<std::int32_t> reused_greedy = {argmax_tok(logits_reused)};
  for (int k = 0; k < 4; ++k) {
    logits_reused = s_reuse.step(model, reused_greedy.back());
    assert_logits_byte_identical(cold_steps[static_cast<std::size_t>(k + 1)], logits_reused);
    reused_greedy.push_back(argmax_tok(logits_reused));
  }
  CHECK(reused_greedy == cold_greedy);

  // Fresh-session isolation after unrelated conversation + reset() matches cold full-vocab bytes & greedy IDs.
  Qwen38MiniSession s_dirty(model.config());
  (void)s_dirty.prefill_last_logits(model, std::vector<std::int32_t>{1, 2, 3, 4, 5, 6, 7});
  s_dirty.reset();
  CHECK(s_dirty.position() == 0);
  auto logits_clean = s_dirty.prefill_last_logits(model, full_b);
  assert_logits_byte_identical(cold_steps[0], logits_clean);
  std::vector<std::int32_t> clean_greedy = {argmax_tok(logits_clean)};
  for (int k = 0; k < 4; ++k) {
    logits_clean = s_dirty.step(model, clean_greedy.back());
    assert_logits_byte_identical(cold_steps[static_cast<std::size_t>(k + 1)], logits_clean);
    clean_greedy.push_back(argmax_tok(logits_clean));
  }
  CHECK(clean_greedy == cold_greedy);

  // Out-of-vocab token fails closed.
  bool threw = false;
  try {
    (void)s_dirty.step(model, model.config().vocab_size);
  } catch (const std::out_of_range&) {
    threw = true;
  }
  CHECK(threw);
}

}  // namespace

int main() {
  test_fp8_and_nvfp4();
  test_python_math_oracles();
  test_conv_and_gdn_chunk_vs_recurrent();
  test_mini_model_prefill_reuse_and_isolation();
  std::printf("qwen38_cpu_ops: PASS\n");
  return 0;
}

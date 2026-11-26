#include "backends/qwen4exp/qwen4exp_cpu_ops.h"

#include "backends/qwen38/qwen38_cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace ingine::backends::qwen4exp {
namespace {

inline float sigmoid_f32(float x) noexcept {
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

inline float silu_f32(float x) noexcept { return x * sigmoid_f32(x); }

inline float softplus_f32(float x) noexcept {
  if (x > 20.0f) return x;
  if (x < -20.0f) return std::exp(x);
  return std::log1p(std::exp(x));
}

void l2_normalize_heads(std::span<float> x, int num_heads, int head_dim, float eps) {
  for (int h = 0; h < num_heads; ++h) {
    float* ptr = x.data() + static_cast<std::size_t>(h) * head_dim;
    double sum_sq = 0.0;
    for (int d = 0; d < head_dim; ++d) sum_sq += double(ptr[d]) * double(ptr[d]);
    const float inv = 1.0f / std::max(static_cast<float>(std::sqrt(sum_sq)), eps);
    for (int d = 0; d < head_dim; ++d) ptr[d] *= inv;
  }
}

}  // namespace

void rmsnorm_f32(std::span<const float> x, std::span<const float> weight, float eps, std::span<float> out) {
  if (x.size() != weight.size() || x.size() != out.size() || x.empty()) {
    throw std::invalid_argument("rmsnorm_f32: dimension mismatch");
  }
  double ss = 0.0;
  for (float v : x) ss += double(v) * double(v);
  const float inv = static_cast<float>(1.0 / std::sqrt(ss / double(x.size()) + double(eps)));
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] * inv * weight[i];
  }
}

void grouped_rmsnorm_f32(std::span<const float> x, std::span<const float> weight, int groups, int group_dim, float eps,
                         std::span<float> out) {
  const std::size_t total = static_cast<std::size_t>(groups) * static_cast<std::size_t>(group_dim);
  if (groups <= 0 || group_dim <= 0 || x.size() != total || weight.size() != total || out.size() != total) {
    throw std::invalid_argument("grouped_rmsnorm_f32: dimension mismatch");
  }
  for (int g = 0; g < groups; ++g) {
    const std::size_t off = static_cast<std::size_t>(g) * static_cast<std::size_t>(group_dim);
    rmsnorm_f32(x.subspan(off, group_dim), weight.subspan(off, group_dim), eps, out.subspan(off, group_dim));
  }
}

void matvec_f32(std::span<const float> w, std::span<const float> x, int out_features, int in_features,
                std::span<float> y) {
  if (out_features <= 0 || in_features <= 0 || static_cast<int>(x.size()) != in_features ||
      static_cast<int>(y.size()) != out_features ||
      w.size() != static_cast<std::size_t>(out_features) * static_cast<std::size_t>(in_features)) {
    throw std::invalid_argument("matvec_f32: dimension mismatch");
  }
  const std::size_t work = static_cast<std::size_t>(out_features) * static_cast<std::size_t>(in_features);
  const unsigned hw = std::max(1u, std::min(14u, std::thread::hardware_concurrency()));
  if (out_features < 512 || work < 1000000ull || hw <= 1u) {
    for (int r = 0; r < out_features; ++r) {
      const float* row = w.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_features);
      double acc = 0.0;
      for (int c = 0; c < in_features; ++c) acc += double(row[c]) * double(x[c]);
      y[static_cast<std::size_t>(r)] = static_cast<float>(acc);
    }
    return;
  }
  const int n_threads = static_cast<int>(std::min<unsigned>(hw, static_cast<unsigned>(out_features / 64)));
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(n_threads));
  for (int t = 0; t < n_threads; ++t) {
    const int r0 = (out_features * t) / n_threads;
    const int r1 = (out_features * (t + 1)) / n_threads;
    pool.emplace_back([&, r0, r1]() {
      for (int r = r0; r < r1; ++r) {
        const float* row = w.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_features);
        double acc = 0.0;
        for (int c = 0; c < in_features; ++c) acc += double(row[c]) * double(x[c]);
        y[static_cast<std::size_t>(r)] = static_cast<float>(acc);
      }
    });
  }
  for (auto& th : pool) th.join();
}

void hc_mix_f32(std::span<const float> x, std::span<const float> w_norm, std::span<const float> w_down,
                std::span<const float> w_up, std::span<const float> w_inject, int hc, int hidden, int hc_lr, float eps,
                std::span<float> out_mixed, std::span<float> out_inject) {
  const int hc_dim = hc * hidden;
  std::vector<float> xn(static_cast<std::size_t>(hc_dim));
  grouped_rmsnorm_f32(x, w_norm, hc, hidden, eps, xn);

  std::vector<float> lo(static_cast<std::size_t>(hc_lr));
  matvec_f32(w_down, xn, hc_lr, hc_dim, lo);
  const float inv_hc = 1.0f / static_cast<float>(hc);
  for (float& v : lo) v = silu_f32(v * inv_hc);

  std::vector<float> gate(static_cast<std::size_t>(hc_dim));
  matvec_f32(w_up, lo, hc_dim, hc_lr, gate);
  for (float& v : gate) v = sigmoid_f32(v);

  std::fill(out_mixed.begin(), out_mixed.end(), 0.0f);
  for (int c = 0; c < hc; ++c) {
    const std::size_t off = static_cast<std::size_t>(c) * static_cast<std::size_t>(hidden);
    for (int i = 0; i < hidden; ++i) {
      out_mixed[static_cast<std::size_t>(i)] += xn[off + i] * gate[off + i];
    }
  }
  for (int i = 0; i < hidden; ++i) out_mixed[static_cast<std::size_t>(i)] *= inv_hc;

  if (!w_inject.empty()) {
    matvec_f32(w_inject, xn, hc, hc_dim, out_inject);
  }
}

void hc_combine_f32(std::span<const float> residual, std::span<const float> block_out, std::span<const float> inject,
                    int hc, int hidden, std::span<float> out) {
  const std::size_t hc_dim = static_cast<std::size_t>(hc) * static_cast<std::size_t>(hidden);
  if (residual.size() != hc_dim || out.size() != hc_dim || static_cast<int>(block_out.size()) != hidden ||
      static_cast<int>(inject.size()) != hc) {
    throw std::invalid_argument("hc_combine_f32: dimension mismatch");
  }
  const float inv_hc = 1.0f / static_cast<float>(hc);
  for (int c = 0; c < hc; ++c) {
    const float w = 2.0f * sigmoid_f32(inject[static_cast<std::size_t>(c)] * inv_hc);
    const std::size_t off = static_cast<std::size_t>(c) * static_cast<std::size_t>(hidden);
    for (int i = 0; i < hidden; ++i) {
      out[off + i] = residual[off + i] + block_out[static_cast<std::size_t>(i)] * w;
    }
  }
}

void ple_hash_step(const Qwen4ExpConfig& cfg, std::int32_t token_id, std::vector<std::int32_t>& token_history,
                   std::span<std::int32_t> out_rows) {
  const int n_gram = cfg.ple_ngram_size;
  const int per_gram = cfg.ple_heads_per_ngram;
  const int n_heads = cfg.ple_n_heads();
  if (n_heads <= 0 || static_cast<int>(out_rows.size()) != n_heads) {
    throw std::invalid_argument("ple_hash_step: invalid PLE config or output size");
  }
  const std::int32_t eos = cfg.ple_eos_token_id;
  if (static_cast<int>(token_history.size()) != n_gram - 1) {
    token_history.assign(static_cast<std::size_t>(n_gram - 1), eos);
  }

  // ctx[0] = current token; ctx[s] = s-th predecessor, reset to EOS at or before any strictly earlier EOS.
  std::vector<std::int64_t> ctx(static_cast<std::size_t>(n_gram), eos);
  ctx[0] = static_cast<std::int64_t>(token_id);
  bool cut = false;
  for (int s = 1; s < n_gram; ++s) {
    const std::int32_t prev_tok = token_history[static_cast<std::size_t>((n_gram - 1) - s)];
    ctx[static_cast<std::size_t>(s)] = cut ? eos : static_cast<std::int64_t>(prev_tok);
    if (ctx[static_cast<std::size_t>(s)] == eos) cut = true;
  }

  for (int n = 2; n <= n_gram; ++n) {
    std::uint64_t mixed = static_cast<std::uint64_t>(ctx[0]) * cfg.ple_layer_multipliers[0];
    for (int j = 1; j < n; ++j) {
      mixed ^= static_cast<std::uint64_t>(ctx[static_cast<std::size_t>(j)]) *
               cfg.ple_layer_multipliers[static_cast<std::size_t>(j)];
    }
    const int base = (n - 2) * per_gram;
    for (int g = 0; g < per_gram; ++g) {
      const int hi = base + g;
      const std::uint64_t row =
          (mixed % cfg.ple_head_vocab_sizes[static_cast<std::size_t>(hi)]) +
          cfg.ple_head_offsets[static_cast<std::size_t>(hi)];
      out_rows[static_cast<std::size_t>(hi)] = static_cast<std::int32_t>(row);
    }
  }

  // Advance history: drop oldest, append current token.
  for (int i = 0; i + 1 < n_gram - 1; ++i) {
    token_history[static_cast<std::size_t>(i)] = token_history[static_cast<std::size_t>(i + 1)];
  }
  if (n_gram > 1) {
    token_history.back() = token_id;
  }
}

void ple_step_f32(const Qwen4ExpConfig& cfg, std::span<const float> hidden_in, std::span<const float> ple_emb,
                  std::span<const float> w_key, std::span<const float> w_value, std::span<const float> w_norm_key,
                  std::span<const float> w_norm_query, std::span<const float> w_norm_conv,
                  std::span<const float> w_conv1d, std::vector<float>& ple_conv_state, std::span<float> hidden_out) {
  const int H = cfg.hidden_size;
  const int hc = cfg.hc_mult;
  const int hc_dim = cfg.hc_dim();
  const int kern = cfg.ple_conv_kernel;
  const int dil = cfg.ple_ngram_size;
  const int hist = cfg.ple_conv_history_cols();

  std::vector<float> key_raw(static_cast<std::size_t>(hc_dim));
  std::vector<float> value(static_cast<std::size_t>(H));
  matvec_f32(w_key, ple_emb, hc_dim, H, key_raw);
  matvec_f32(w_value, ple_emb, H, H, value);

  std::vector<float> key(static_cast<std::size_t>(hc_dim));
  std::vector<float> query(static_cast<std::size_t>(hc_dim));
  grouped_rmsnorm_f32(key_raw, w_norm_key, hc, H, cfg.rms_norm_eps, key);
  grouped_rmsnorm_f32(hidden_in, w_norm_query, hc, H, cfg.rms_norm_eps, query);

  const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(H));
  std::vector<float> gated(static_cast<std::size_t>(hc_dim));
  for (int c = 0; c < hc; ++c) {
    const std::size_t off = static_cast<std::size_t>(c) * static_cast<std::size_t>(H);
    double dot = 0.0;
    for (int i = 0; i < H; ++i) dot += double(key[off + i]) * double(query[off + i]);
    const float s = static_cast<float>(dot) * inv_sqrt_h;
    const float mag = std::sqrt(std::max(std::fabs(s), 1e-6f));
    const float sgn = (s > 0.0f) ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
    const float g = sigmoid_f32(sgn * mag);
    for (int i = 0; i < H; ++i) gated[off + i] = value[static_cast<std::size_t>(i)] * g;
  }

  std::vector<float> normed(static_cast<std::size_t>(hc_dim));
  grouped_rmsnorm_f32(gated, w_norm_conv, hc, H, cfg.rms_norm_eps, normed);

  const std::size_t need_state = static_cast<std::size_t>(hist) * static_cast<std::size_t>(hc_dim);
  if (ple_conv_state.size() != need_state) {
    ple_conv_state.assign(need_state, 0.0f);
  }

  // Dilated causal depthwise conv:
  //   window[0 .. hist-1] = ple_conv_state, window[hist] = normed
  //   out[ch] = silu(sum_{k=0..kern-1} w_conv1d[ch * kern + k] * window[hist - (kern - 1 - k) * dil][ch])
  std::vector<float> conv_out(static_cast<std::size_t>(hc_dim), 0.0f);
  for (int ch = 0; ch < hc_dim; ++ch) {
    double acc = 0.0;
    const float* wc = w_conv1d.data() + static_cast<std::size_t>(ch) * static_cast<std::size_t>(kern);
    for (int k = 0; k < kern; ++k) {
      const int idx = hist - (kern - 1 - k) * dil;
      const float xval = (idx == hist)
                             ? normed[static_cast<std::size_t>(ch)]
                             : ple_conv_state[static_cast<std::size_t>(idx) * static_cast<std::size_t>(hc_dim) + ch];
      acc += double(wc[k]) * double(xval);
    }
    conv_out[static_cast<std::size_t>(ch)] = silu_f32(static_cast<float>(acc));
  }

  // Shift conv history left by 1 column and store `normed` at the tail.
  if (hist > 0) {
    for (int col = 0; col + 1 < hist; ++col) {
      std::copy_n(ple_conv_state.data() + static_cast<std::size_t>(col + 1) * hc_dim, hc_dim,
                  ple_conv_state.data() + static_cast<std::size_t>(col) * hc_dim);
    }
    std::copy_n(normed.data(), hc_dim, ple_conv_state.data() + static_cast<std::size_t>(hist - 1) * hc_dim);
  }

  for (int i = 0; i < hc_dim; ++i) {
    hidden_out[static_cast<std::size_t>(i)] =
        hidden_in[static_cast<std::size_t>(i)] + gated[static_cast<std::size_t>(i)] +
        conv_out[static_cast<std::size_t>(i)];
  }
}

void gdn_step_sigmoid_f32(std::span<const float> qkv_mixed, std::span<const float> z_vdim,
                          std::span<const float> alpha_vheads, std::span<const float> beta_raw_vheads,
                          std::span<const float> ssm_conv_weight, std::span<const float> ssm_dt_bias,
                          std::span<const float> ssm_a_neg, std::span<const float> ssm_norm_weight, float rms_eps,
                          int conv_kernel, int num_k_heads, int num_v_heads, int head_dim,
                          std::vector<float>& conv_state, std::vector<float>& recurrent_state,
                          std::span<float> out_vdim) {
  const int k_dim_total = num_k_heads * head_dim;
  const int v_dim_total = num_v_heads * head_dim;
  const int conv_dim = 2 * k_dim_total + v_dim_total;

  // 1. Causal depthwise conv1d + SiLU (reusing qwen38_cpu_ops exact reference).
  const std::size_t need_conv = static_cast<std::size_t>(std::max(0, conv_kernel)) * static_cast<std::size_t>(conv_dim);
  if (conv_state.size() != need_conv) {
    conv_state.assign(need_conv, 0.0f);
  }
  std::vector<float> conv_out(static_cast<std::size_t>(conv_dim));
  ingine::backends::qwen38::causal_conv1d_silu_step(qkv_mixed, ssm_conv_weight, conv_dim, conv_kernel, conv_state,
                                                    conv_out);

  std::vector<float> q(conv_out.begin(), conv_out.begin() + k_dim_total);
  std::vector<float> k(conv_out.begin() + k_dim_total, conv_out.begin() + 2 * k_dim_total);
  std::span<const float> v(conv_out.data() + 2 * k_dim_total, static_cast<std::size_t>(v_dim_total));

  l2_normalize_heads(q, num_k_heads, head_dim, rms_eps);
  l2_normalize_heads(k, num_k_heads, head_dim, rms_eps);

  const float q_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const std::size_t need_rec =
      static_cast<std::size_t>(num_v_heads) * static_cast<std::size_t>(head_dim) * static_cast<std::size_t>(head_dim);
  if (recurrent_state.size() != need_rec) {
    recurrent_state.assign(need_rec, 0.0f);
  }

  std::vector<float> raw_out(static_cast<std::size_t>(head_dim));
  std::vector<float> normed_out(static_cast<std::size_t>(head_dim));
  for (int vh = 0; vh < num_v_heads; ++vh) {
    // GGUF qwen4exp tiles Q/K heads across V heads via ggml_repeat_4d (iq1 = iv1 % num_k_heads).
    const int kh = vh % num_k_heads;
    const float* qh = q.data() + static_cast<std::size_t>(kh) * head_dim;
    const float* khp = k.data() + static_cast<std::size_t>(kh) * head_dim;
    const float* vhp = v.data() + static_cast<std::size_t>(vh) * head_dim;
    const float* zhp = z_vdim.data() + static_cast<std::size_t>(vh) * head_dim;
    float* state = recurrent_state.data() + static_cast<std::size_t>(vh) * head_dim * head_dim;

    // g = exp(softplus(alpha + dt_bias) * ssm_a_neg), beta = sigmoid(beta_raw).
    const float log_g = softplus_f32(alpha_vheads[vh] + ssm_dt_bias[vh]) * ssm_a_neg[vh];
    const float decay = std::exp(log_g);
    const float beta = sigmoid_f32(beta_raw_vheads[vh]);

    // Decay state: S *= decay
    for (int i = 0; i < head_dim * head_dim; ++i) state[i] *= decay;

    // kv_mem[j] = sum_i S[i, j] * k[i]; delta[j] = (v[j] - kv_mem[j]) * beta; S[i, j] += k[i] * delta[j]
    // out[j] = sum_i S[i, j] * (q[i] * q_scale)
    for (int j = 0; j < head_dim; ++j) {
      double kv_mem = 0.0;
      for (int i = 0; i < head_dim; ++i) kv_mem += double(state[i * head_dim + j]) * double(khp[i]);
      const double delta = (double(vhp[j]) - kv_mem) * double(beta);
      double o = 0.0;
      for (int i = 0; i < head_dim; ++i) {
        const float ns = static_cast<float>(double(state[i * head_dim + j]) + double(khp[i]) * delta);
        state[i * head_dim + j] = ns;
        o += double(ns) * double(qh[i] * q_scale);
      }
      raw_out[static_cast<std::size_t>(j)] = static_cast<float>(o);
    }

    // qwen4exp build_norm_gated: RMSNorm(raw_out, ssm_norm_weight) * sigmoid(z)!
    rmsnorm_f32(raw_out, ssm_norm_weight, rms_eps, normed_out);
    float* dst = out_vdim.data() + static_cast<std::size_t>(vh) * head_dim;
    for (int j = 0; j < head_dim; ++j) {
      dst[j] = normed_out[static_cast<std::size_t>(j)] * sigmoid_f32(zhp[j]);
    }
  }
}

std::vector<int> qsa_select_positions_f32(std::span<const float> q_idx_raw, std::span<const float> k_idx_raw,
                                          std::span<const float> idx_q_norm, std::span<const float> idx_k_norm,
                                          float rms_eps, float rope_theta, int rotary_dim, int idx_heads, int idx_dim,
                                          int compress_ratio, int top_k_budget, int pos,
                                          std::vector<float>& idx_k_cache) {
  if (compress_ratio <= 0 || top_k_budget <= 0 || pos < 0) {
    throw std::invalid_argument("qsa_select_positions_f32: invalid QSA parameters");
  }
  const std::size_t need = static_cast<std::size_t>(pos + 1) * static_cast<std::size_t>(idx_dim);
  if (idx_k_cache.size() < need) idx_k_cache.resize(need, 0.0f);
  std::copy_n(k_idx_raw.data(), idx_dim, idx_k_cache.data() + static_cast<std::size_t>(pos) * idx_dim);

  const int n_kv = pos + 1;
  const int width = std::min(n_kv, top_k_budget + compress_ratio - 1);
  if (n_kv <= width) {
    std::vector<int> all(static_cast<std::size_t>(n_kv));
    std::iota(all.begin(), all.end(), 0);
    return all;
  }

  // Normalize & RoPE indexer Q at `pos`.
  std::vector<float> q_idx(static_cast<std::size_t>(idx_heads) * idx_dim);
  for (int h = 0; h < idx_heads; ++h) {
    rmsnorm_f32(q_idx_raw.subspan(static_cast<std::size_t>(h) * idx_dim, idx_dim), idx_q_norm, rms_eps,
                std::span<float>(q_idx.data() + static_cast<std::size_t>(h) * idx_dim, idx_dim));
  }
  ingine::backends::qwen38::apply_partial_rope_neox(q_idx, idx_heads, idx_dim, rotary_dim, pos, rope_theta);

  // Score every complete block b in [0, n_complete_blocks).
  const int n_complete_blocks = n_kv / compress_ratio;
  const int tail_start = n_complete_blocks * compress_ratio;
  std::vector<float> block_scores(static_cast<std::size_t>(n_complete_blocks), 0.0f);
  std::vector<float> k_pool(static_cast<std::size_t>(idx_dim));
  std::vector<float> k_rot(static_cast<std::size_t>(idx_dim));
  const float inv_r = 1.0f / static_cast<float>(compress_ratio);
  for (int b = 0; b < n_complete_blocks; ++b) {
    std::fill(k_pool.begin(), k_pool.end(), 0.0f);
    for (int off = 0; off < compress_ratio; ++off) {
      const float* kp = idx_k_cache.data() + static_cast<std::size_t>(b * compress_ratio + off) * idx_dim;
      for (int d = 0; d < idx_dim; ++d) k_pool[static_cast<std::size_t>(d)] += kp[d];
    }
    for (int d = 0; d < idx_dim; ++d) k_pool[static_cast<std::size_t>(d)] *= inv_r;
    rmsnorm_f32(k_pool, idx_k_norm, rms_eps, k_rot);
    ingine::backends::qwen38::apply_partial_rope_neox(k_rot, 1, idx_dim, rotary_dim, b * compress_ratio, rope_theta);
    double sum = 0.0;
    for (int h = 0; h < idx_heads; ++h) {
      const float* qh = q_idx.data() + static_cast<std::size_t>(h) * idx_dim;
      double dot = 0.0;
      for (int d = 0; d < idx_dim; ++d) dot += double(qh[d]) * double(k_rot[static_cast<std::size_t>(d)]);
      if (dot > 0.0) sum += dot;
    }
    block_scores[static_cast<std::size_t>(b)] = static_cast<float>(sum);
  }

  // Assign each token its block score (or +1e9f for the always-visible incomplete tail).
  std::vector<std::pair<float, int>> scored;
  scored.reserve(static_cast<std::size_t>(n_kv));
  for (int p = 0; p < n_kv; ++p) {
    const float sc = (p >= tail_start) ? 1e9f : block_scores[static_cast<std::size_t>(p / compress_ratio)];
    scored.emplace_back(sc, p);
  }
  std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
    if (a.first != b.first) return a.first > b.first;
    return a.second < b.second;
  });

  std::vector<int> selected;
  selected.reserve(static_cast<std::size_t>(width));
  for (int i = 0; i < width; ++i) selected.push_back(scored[static_cast<std::size_t>(i)].second);
  std::sort(selected.begin(), selected.end());
  return selected;
}

void full_attn_qsa_step_f32(std::span<const float> q_with_gate, std::span<const float> k_raw, std::span<const float> v,
                            std::span<const float> q_norm_w, std::span<const float> k_norm_w, float rms_eps,
                            float rope_theta, int num_heads, int num_kv_heads, int head_dim, int rotary_dim, int pos,
                            std::span<const int> selected_positions, std::vector<float>& k_cache,
                            std::vector<float>& v_cache, std::span<float> out_attn) {
  std::vector<float> q(static_cast<std::size_t>(num_heads) * head_dim);
  std::vector<float> gate(static_cast<std::size_t>(num_heads) * head_dim);
  for (int h = 0; h < num_heads; ++h) {
    const std::size_t src = static_cast<std::size_t>(h) * (2 * head_dim);
    std::copy_n(q_with_gate.data() + src, head_dim, q.data() + static_cast<std::size_t>(h) * head_dim);
    std::copy_n(q_with_gate.data() + src + head_dim, head_dim, gate.data() + static_cast<std::size_t>(h) * head_dim);
  }
  for (int h = 0; h < num_heads; ++h) {
    auto slice = std::span<float>(q.data() + static_cast<std::size_t>(h) * head_dim, head_dim);
    rmsnorm_f32(slice, q_norm_w, rms_eps, slice);
  }
  std::vector<float> k(static_cast<std::size_t>(num_kv_heads) * head_dim);
  for (int h = 0; h < num_kv_heads; ++h) {
    rmsnorm_f32(k_raw.subspan(static_cast<std::size_t>(h) * head_dim, head_dim), k_norm_w, rms_eps,
                std::span<float>(k.data() + static_cast<std::size_t>(h) * head_dim, head_dim));
  }

  ingine::backends::qwen38::apply_partial_rope_neox(q, num_heads, head_dim, rotary_dim, pos, rope_theta);
  ingine::backends::qwen38::apply_partial_rope_neox(k, num_kv_heads, head_dim, rotary_dim, pos, rope_theta);

  const std::size_t kv_stride = static_cast<std::size_t>(num_kv_heads) * head_dim;
  const std::size_t need = static_cast<std::size_t>(pos + 1) * kv_stride;
  if (k_cache.size() < need) k_cache.resize(need, 0.0f);
  if (v_cache.size() < need) v_cache.resize(need, 0.0f);
  std::copy_n(k.data(), kv_stride, k_cache.data() + static_cast<std::size_t>(pos) * kv_stride);
  std::copy_n(v.data(), kv_stride, v_cache.data() + static_cast<std::size_t>(pos) * kv_stride);

  std::vector<int> dense_pos;
  if (selected_positions.empty()) {
    dense_pos.resize(static_cast<std::size_t>(pos + 1));
    std::iota(dense_pos.begin(), dense_pos.end(), 0);
    selected_positions = dense_pos;
  }

  const int gqa_group = num_heads / num_kv_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<double> scores(selected_positions.size());

  for (int h = 0; h < num_heads; ++h) {
    const int kvh = h / gqa_group;
    const float* qh = q.data() + static_cast<std::size_t>(h) * head_dim;
    double max_s = -std::numeric_limits<double>::infinity();
    for (std::size_t idx = 0; idx < selected_positions.size(); ++idx) {
      const int p = selected_positions[idx];
      const float* kp = k_cache.data() + static_cast<std::size_t>(p) * kv_stride + static_cast<std::size_t>(kvh) * head_dim;
      double dot = 0.0;
      for (int d = 0; d < head_dim; ++d) dot += double(qh[d]) * double(kp[d]);
      const double sc = dot * double(scale);
      scores[idx] = sc;
      if (sc > max_s) max_s = sc;
    }
    double sum_e = 0.0;
    for (double& s : scores) {
      s = std::exp(s - max_s);
      sum_e += s;
    }
    float* dst = out_attn.data() + static_cast<std::size_t>(h) * head_dim;
    const float* gh = gate.data() + static_cast<std::size_t>(h) * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      double acc = 0.0;
      for (std::size_t idx = 0; idx < selected_positions.size(); ++idx) {
        const int p = selected_positions[idx];
        const float* vp =
            v_cache.data() + static_cast<std::size_t>(p) * kv_stride + static_cast<std::size_t>(kvh) * head_dim;
        acc += (scores[idx] / sum_e) * double(vp[d]);
      }
      dst[d] = static_cast<float>(acc) * sigmoid_f32(gh[d]);
    }
  }
}

std::vector<RouterChoice> route_top_k_softmax(std::span<const float> router_logits, int top_k, float weights_scale) {
  if (router_logits.empty() || top_k <= 0 || top_k > static_cast<int>(router_logits.size())) {
    throw std::invalid_argument("route_top_k_softmax: invalid router dimensions");
  }
  double max_v = -std::numeric_limits<double>::infinity();
  for (float v : router_logits) max_v = std::max(max_v, double(v));
  std::vector<double> probs(router_logits.size());
  double sum_e = 0.0;
  for (std::size_t i = 0; i < router_logits.size(); ++i) {
    probs[i] = std::exp(double(router_logits[i]) - max_v);
    sum_e += probs[i];
  }
  std::vector<int> order(router_logits.size());
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + top_k, order.end(), [&](int a, int b) {
    if (probs[a] != probs[b]) return probs[a] > probs[b];
    return a < b;
  });

  double top_sum = 0.0;
  for (int i = 0; i < top_k; ++i) top_sum += probs[order[i]] / sum_e;
  std::vector<RouterChoice> out;
  out.reserve(static_cast<std::size_t>(top_k));
  for (int i = 0; i < top_k; ++i) {
    const int e = order[static_cast<std::size_t>(i)];
    const float w = static_cast<float>(((probs[e] / sum_e) / std::max(top_sum, 1e-12)) * double(weights_scale));
    out.push_back(RouterChoice{e, w});
  }
  return out;
}

void swiglu_ffn_f32(std::span<const float> x, std::span<const float> w_gate, std::span<const float> w_up,
                    std::span<const float> w_down, int hidden, int ffn_dim, std::span<float> out) {
  std::vector<float> g(static_cast<std::size_t>(ffn_dim));
  std::vector<float> u(static_cast<std::size_t>(ffn_dim));
  matvec_f32(w_gate, x, ffn_dim, hidden, g);
  matvec_f32(w_up, x, ffn_dim, hidden, u);
  for (int i = 0; i < ffn_dim; ++i) g[static_cast<std::size_t>(i)] = silu_f32(g[i]) * u[i];
  matvec_f32(w_down, g, hidden, ffn_dim, out);
}

void moe_ffn_step_f32(std::span<const float> x, std::span<const float> w_router, std::span<const float> w_gate_exps,
                      std::span<const float> w_up_exps, std::span<const float> w_down_exps,
                      std::span<const float> w_sh_gate_inp, std::span<const float> w_sh_gate,
                      std::span<const float> w_sh_up, std::span<const float> w_sh_down, int hidden, int expert_count,
                      int expert_used, int expert_ffn, int shared_ffn, float weights_scale, std::span<float> out) {
  std::vector<float> r_logits(static_cast<std::size_t>(expert_count));
  matvec_f32(w_router, x, expert_count, hidden, r_logits);
  const auto chosen = route_top_k_softmax(r_logits, expert_used, weights_scale);

  std::fill(out.begin(), out.end(), 0.0f);
  std::vector<float> e_out(static_cast<std::size_t>(hidden));
  const std::size_t gu_stride = static_cast<std::size_t>(expert_ffn) * static_cast<std::size_t>(hidden);
  for (const auto& ch : chosen) {
    const std::size_t off = static_cast<std::size_t>(ch.expert) * gu_stride;
    swiglu_ffn_f32(x, w_gate_exps.subspan(off, gu_stride), w_up_exps.subspan(off, gu_stride),
                   w_down_exps.subspan(off, gu_stride), hidden, expert_ffn, e_out);
    for (int i = 0; i < hidden; ++i) out[static_cast<std::size_t>(i)] += ch.weight * e_out[i];
  }

  if (shared_ffn > 0 && !w_sh_gate.empty()) {
    std::vector<float> sh_out(static_cast<std::size_t>(hidden));
    swiglu_ffn_f32(x, w_sh_gate, w_sh_up, w_sh_down, hidden, shared_ffn, sh_out);
    double sg_dot = 0.0;
    for (int i = 0; i < hidden; ++i) sg_dot += double(w_sh_gate_inp[i]) * double(x[i]);
    const float sg = sigmoid_f32(static_cast<float>(sg_dot));
    for (int i = 0; i < hidden; ++i) out[static_cast<std::size_t>(i)] += sg * sh_out[i];
  }
}

Qwen4ExpSessionCpu::Qwen4ExpSessionCpu(const Qwen4ExpConfig& c) : cfg(c) { reset(); }

void Qwen4ExpSessionCpu::reset() {
  committed_tokens.clear();
  if (cfg.ple_ngram_size >= 2) {
    ple_token_history.assign(static_cast<std::size_t>(cfg.ple_ngram_size - 1), cfg.ple_eos_token_id);
  } else {
    ple_token_history.clear();
  }
  ple_conv_states.assign(static_cast<std::size_t>(cfg.num_hidden_layers), {});
  gdn_conv_states.assign(static_cast<std::size_t>(cfg.num_hidden_layers), {});
  gdn_recurrent_states.assign(static_cast<std::size_t>(cfg.num_hidden_layers), {});
  k_caches.assign(static_cast<std::size_t>(cfg.num_hidden_layers), {});
  v_caches.assign(static_cast<std::size_t>(cfg.num_hidden_layers), {});
  idx_k_caches.assign(static_cast<std::size_t>(cfg.num_hidden_layers), {});
}

Qwen4ExpMiniModelCpu Qwen4ExpMiniModelCpu::make_synthetic(std::uint32_t seed, int top_k_budget) {
  Qwen4ExpMiniModelCpu m;
  Qwen4ExpConfig& c = m.cfg_;
  c.architecture = "qwen4exp";
  c.model_name = "Qwen4Exp-Synthetic-Mini";
  c.vocab_size = 64;
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // L0 GDN, L1 GDN+PLE, L2 GDN, L3 FullAttn+QSA
  c.max_position_embeddings = 256;
  c.rms_norm_eps = 1e-6f;
  c.hc_mult = 4;
  c.hc_low_rank = 8;
  c.full_attention_interval = 4;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.rotary_dim = 8;
  c.rope_theta = 10000000.0f;
  c.indexer_head_count = 2;
  c.indexer_key_length = 16;
  c.indexer_top_k = top_k_budget;
  c.compress_ratios = {0, 0, 0, 4};
  c.ssm_conv_kernel = 4;
  c.ssm_state_size = 8;
  c.ssm_group_count = 2;
  c.ssm_time_step_rank = 4;
  c.ssm_inner_size = 32;
  c.expert_count = 8;
  c.expert_used_count = 2;
  c.expert_ffn_length = 16;
  c.shared_expert_ffn_length = 16;
  c.ple_layers = {1};
  c.ple_ngram_size = 3;
  c.ple_heads_per_ngram = 4;  // 2 * 4 = 8 heads * ple_head_dim 4 = 32 == hidden_size
  c.ple_conv_kernel = 4;
  c.ple_head_dim = 4;
  c.ple_eos_token_id = 0;
  c.ple_layer_multipliers = {23703573157769ull, 20109073645365ull, 8052911324071ull};
  c.ple_head_vocab_sizes = {17, 19, 23, 29, 31, 37, 41, 43};
  c.ple_head_offsets.resize(8);
  std::uint64_t off = 0;
  for (int i = 0; i < 8; ++i) {
    c.ple_head_offsets[i] = off;
    off += c.ple_head_vocab_sizes[i];
  }
  c.finalize_and_validate();

  std::mt19937 rng(seed);
  auto rand_vec = [&](std::size_t n, float scale) {
    std::normal_distribution<float> dist(0.0f, scale);
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
  };
  auto ones_perturbed = [&](std::size_t n) {
    std::normal_distribution<float> dist(1.0f, 0.05f);
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
  };

  const int H = c.hidden_size;
  const int hc = c.hc_mult;
  const int hc_dim = c.hc_dim();
  const int hc_lr = c.hc_low_rank;

  m.token_embd_ = rand_vec(static_cast<std::size_t>(c.vocab_size) * H, 0.2f);
  m.ple_table_ = rand_vec(off * static_cast<std::uint64_t>(c.ple_head_dim), 0.15f);
  m.output_hc_norm_ = ones_perturbed(hc_dim);
  m.output_hc_down_ = rand_vec(static_cast<std::size_t>(hc_lr) * hc_dim, 0.1f);
  m.output_hc_up_ = rand_vec(static_cast<std::size_t>(hc_dim) * hc_lr, 0.1f);
  m.output_weight_ = rand_vec(static_cast<std::size_t>(c.vocab_size) * H, 0.15f);

  m.layers_.resize(static_cast<std::size_t>(c.num_hidden_layers));
  for (int il = 0; il < c.num_hidden_layers; ++il) {
    auto& L = m.layers_[static_cast<std::size_t>(il)];
    L.kind = c.layer_types[static_cast<std::size_t>(il)];
    L.has_ple = c.is_ple(il);
    L.hc_attn_norm = ones_perturbed(hc_dim);
    L.hc_attn_down = rand_vec(static_cast<std::size_t>(hc_lr) * hc_dim, 0.1f);
    L.hc_attn_up = rand_vec(static_cast<std::size_t>(hc_dim) * hc_lr, 0.1f);
    L.hc_attn_inject = rand_vec(static_cast<std::size_t>(hc) * hc_dim, 0.1f);
    L.hc_ffn_norm = ones_perturbed(hc_dim);
    L.hc_ffn_down = rand_vec(static_cast<std::size_t>(hc_lr) * hc_dim, 0.1f);
    L.hc_ffn_up = rand_vec(static_cast<std::size_t>(hc_dim) * hc_lr, 0.1f);
    L.hc_ffn_inject = rand_vec(static_cast<std::size_t>(hc) * hc_dim, 0.1f);

    if (L.has_ple) {
      L.ple_key = rand_vec(static_cast<std::size_t>(hc_dim) * H, 0.1f);
      L.ple_value = rand_vec(static_cast<std::size_t>(H) * H, 0.1f);
      L.ple_norm_key = ones_perturbed(hc_dim);
      L.ple_norm_query = ones_perturbed(hc_dim);
      L.ple_norm_conv = ones_perturbed(hc_dim);
      L.ple_conv1d = rand_vec(static_cast<std::size_t>(hc_dim) * c.ple_conv_kernel, 0.15f);
    }

    if (L.kind == Qwen4ExpLayerKind::LinearAttention) {
      L.w_qkv = rand_vec(static_cast<std::size_t>(c.gdn_conv_dim()) * H, 0.12f);
      L.w_gate = rand_vec(static_cast<std::size_t>(c.gdn_value_dim()) * H, 0.12f);
      L.ssm_conv1d = rand_vec(static_cast<std::size_t>(c.gdn_conv_dim()) * c.ssm_conv_kernel, 0.15f);
      L.ssm_dt_bias = rand_vec(c.ssm_time_step_rank, 0.1f);
      L.ssm_a.assign(static_cast<std::size_t>(c.ssm_time_step_rank), -0.5f);
      L.ssm_beta = rand_vec(static_cast<std::size_t>(c.ssm_time_step_rank) * H, 0.12f);
      L.ssm_alpha = rand_vec(static_cast<std::size_t>(c.ssm_time_step_rank) * H, 0.12f);
      L.ssm_norm = ones_perturbed(c.ssm_state_size);
      L.ssm_out = rand_vec(static_cast<std::size_t>(H) * c.gdn_value_dim(), 0.12f);
    } else {
      L.w_q = rand_vec(static_cast<std::size_t>(c.full_q_out_dim()) * H, 0.12f);
      L.w_k = rand_vec(static_cast<std::size_t>(c.full_kv_out_dim()) * H, 0.12f);
      L.w_v = rand_vec(static_cast<std::size_t>(c.full_kv_out_dim()) * H, 0.12f);
      L.w_o = rand_vec(static_cast<std::size_t>(H) * c.num_attention_heads * c.head_dim, 0.12f);
      L.attn_q_norm = ones_perturbed(c.head_dim);
      L.attn_k_norm = ones_perturbed(c.head_dim);
      L.idx_q_proj = rand_vec(static_cast<std::size_t>(c.indexer_head_count) * c.indexer_key_length * H, 0.12f);
      L.idx_k_proj = rand_vec(static_cast<std::size_t>(c.indexer_key_length) * H, 0.12f);
      L.idx_q_norm = ones_perturbed(c.indexer_key_length);
      L.idx_k_norm = ones_perturbed(c.indexer_key_length);
    }

    const std::size_t e_sz = static_cast<std::size_t>(c.expert_count) * c.expert_ffn_length * H;
    L.ffn_gate_inp = rand_vec(static_cast<std::size_t>(c.expert_count) * H, 0.15f);
    L.ffn_gate_exps = rand_vec(e_sz, 0.1f);
    L.ffn_up_exps = rand_vec(e_sz, 0.1f);
    L.ffn_down_exps = rand_vec(e_sz, 0.1f);
    L.ffn_gate_inp_shexp = rand_vec(H, 0.15f);
    L.ffn_gate_shexp = rand_vec(static_cast<std::size_t>(c.shared_expert_ffn_length) * H, 0.1f);
    L.ffn_up_shexp = rand_vec(static_cast<std::size_t>(c.shared_expert_ffn_length) * H, 0.1f);
    L.ffn_down_shexp = rand_vec(static_cast<std::size_t>(H) * c.shared_expert_ffn_length, 0.1f);
  }
  return m;
}

std::vector<float> Qwen4ExpMiniModelCpu::forward_token(Qwen4ExpSessionCpu& session, std::int32_t token_id) const {
  if (token_id < 0 || token_id >= cfg_.vocab_size) {
    throw std::out_of_range("Qwen4ExpMiniModelCpu: token_id out of range");
  }
  const int pos = session.position();
  const int H = cfg_.hidden_size;
  const int hc = cfg_.hc_mult;
  const int hc_dim = cfg_.hc_dim();

  // Initialize wide residual as `hc` copies of token embedding.
  std::vector<float> res_hc(static_cast<std::size_t>(hc_dim));
  const float* emb = token_embd_.data() + static_cast<std::size_t>(token_id) * H;
  for (int c = 0; c < hc; ++c) {
    std::copy_n(emb, H, res_hc.data() + static_cast<std::size_t>(c) * H);
  }

  // Compute PLE hash once per token if any PLE layer exists.
  std::vector<float> ple_emb;
  if (cfg_.ple_n_heads() > 0) {
    std::vector<std::int32_t> ple_rows(static_cast<std::size_t>(cfg_.ple_n_heads()));
    ple_hash_step(cfg_, token_id, session.ple_token_history, ple_rows);
    ple_emb.resize(static_cast<std::size_t>(H));
    for (int h = 0; h < cfg_.ple_n_heads(); ++h) {
      std::copy_n(ple_table_.data() + static_cast<std::size_t>(ple_rows[h]) * cfg_.ple_head_dim, cfg_.ple_head_dim,
                  ple_emb.data() + static_cast<std::size_t>(h) * cfg_.ple_head_dim);
    }
  }

  std::vector<float> cur(static_cast<std::size_t>(H));
  std::vector<float> block_out(static_cast<std::size_t>(H));
  std::vector<float> inject(static_cast<std::size_t>(hc));
  std::vector<float> next_res(static_cast<std::size_t>(hc_dim));

  for (int il = 0; il < cfg_.num_hidden_layers; ++il) {
    const auto& L = layers_[static_cast<std::size_t>(il)];
    if (L.has_ple) {
      ple_step_f32(cfg_, res_hc, ple_emb, L.ple_key, L.ple_value, L.ple_norm_key, L.ple_norm_query, L.ple_norm_conv,
                   L.ple_conv1d, session.ple_conv_states[static_cast<std::size_t>(il)], next_res);
      res_hc.swap(next_res);
    }

    hc_mix_f32(res_hc, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, L.hc_attn_inject, hc, H, cfg_.hc_low_rank,
               cfg_.rms_norm_eps, cur, inject);

    if (L.kind == Qwen4ExpLayerKind::LinearAttention) {
      std::vector<float> qkv(static_cast<std::size_t>(cfg_.gdn_conv_dim()));
      std::vector<float> z(static_cast<std::size_t>(cfg_.gdn_value_dim()));
      std::vector<float> alpha(static_cast<std::size_t>(cfg_.ssm_time_step_rank));
      std::vector<float> beta(static_cast<std::size_t>(cfg_.ssm_time_step_rank));
      matvec_f32(L.w_qkv, cur, cfg_.gdn_conv_dim(), H, qkv);
      matvec_f32(L.w_gate, cur, cfg_.gdn_value_dim(), H, z);
      matvec_f32(L.ssm_alpha, cur, cfg_.ssm_time_step_rank, H, alpha);
      matvec_f32(L.ssm_beta, cur, cfg_.ssm_time_step_rank, H, beta);
      std::vector<float> gdn_out(static_cast<std::size_t>(cfg_.gdn_value_dim()));
      gdn_step_sigmoid_f32(qkv, z, alpha, beta, L.ssm_conv1d, L.ssm_dt_bias, L.ssm_a, L.ssm_norm, cfg_.rms_norm_eps,
                           cfg_.ssm_conv_kernel, cfg_.ssm_group_count, cfg_.ssm_time_step_rank, cfg_.ssm_state_size,
                           session.gdn_conv_states[il], session.gdn_recurrent_states[il], gdn_out);
      matvec_f32(L.ssm_out, gdn_out, H, cfg_.gdn_value_dim(), block_out);
    } else {
      std::vector<int> sel;
      if (cfg_.has_qsa(il)) {
        std::vector<float> iq(static_cast<std::size_t>(cfg_.indexer_head_count) * cfg_.indexer_key_length);
        std::vector<float> ik(static_cast<std::size_t>(cfg_.indexer_key_length));
        matvec_f32(L.idx_q_proj, cur, cfg_.indexer_head_count * cfg_.indexer_key_length, H, iq);
        matvec_f32(L.idx_k_proj, cur, cfg_.indexer_key_length, H, ik);
        sel = qsa_select_positions_f32(iq, ik, L.idx_q_norm, L.idx_k_norm, cfg_.rms_norm_eps, cfg_.rope_theta,
                                       cfg_.rotary_dim, cfg_.indexer_head_count, cfg_.indexer_key_length,
                                       cfg_.compress_ratios[il], cfg_.indexer_top_k, pos, session.idx_k_caches[il]);
      }
      std::vector<float> qg(static_cast<std::size_t>(cfg_.full_q_out_dim()));
      std::vector<float> kr(static_cast<std::size_t>(cfg_.full_kv_out_dim()));
      std::vector<float> vr(static_cast<std::size_t>(cfg_.full_kv_out_dim()));
      matvec_f32(L.w_q, cur, cfg_.full_q_out_dim(), H, qg);
      matvec_f32(L.w_k, cur, cfg_.full_kv_out_dim(), H, kr);
      matvec_f32(L.w_v, cur, cfg_.full_kv_out_dim(), H, vr);
      std::vector<float> attn_out(static_cast<std::size_t>(cfg_.num_attention_heads) * cfg_.head_dim);
      full_attn_qsa_step_f32(qg, kr, vr, L.attn_q_norm, L.attn_k_norm, cfg_.rms_norm_eps, cfg_.rope_theta,
                             cfg_.num_attention_heads, cfg_.num_key_value_heads, cfg_.head_dim, cfg_.rotary_dim, pos,
                             sel, session.k_caches[il], session.v_caches[il], attn_out);
      matvec_f32(L.w_o, attn_out, H, cfg_.num_attention_heads * cfg_.head_dim, block_out);
    }

    hc_combine_f32(res_hc, block_out, inject, hc, H, next_res);
    res_hc.swap(next_res);

    hc_mix_f32(res_hc, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, L.hc_ffn_inject, hc, H, cfg_.hc_low_rank,
               cfg_.rms_norm_eps, cur, inject);
    moe_ffn_step_f32(cur, L.ffn_gate_inp, L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps, L.ffn_gate_inp_shexp,
                     L.ffn_gate_shexp, L.ffn_up_shexp, L.ffn_down_shexp, H, cfg_.expert_count, cfg_.expert_used_count,
                     cfg_.expert_ffn_length, cfg_.shared_expert_ffn_length, cfg_.expert_weights_scale, block_out);
    hc_combine_f32(res_hc, block_out, inject, hc, H, next_res);
    res_hc.swap(next_res);
  }

  hc_mix_f32(res_hc, output_hc_norm_, output_hc_down_, output_hc_up_, {}, hc, H, cfg_.hc_low_rank, cfg_.rms_norm_eps,
             cur, {});
  std::vector<float> logits(static_cast<std::size_t>(cfg_.vocab_size));
  matvec_f32(output_weight_, cur, cfg_.vocab_size, H, logits);
  session.committed_tokens.push_back(token_id);
  return logits;
}

std::vector<float> Qwen4ExpMiniModelCpu::forward_sequence_last_logits(Qwen4ExpSessionCpu& session,
                                                                      std::span<const std::int32_t> token_ids) const {
  if (token_ids.empty()) throw std::invalid_argument("forward_sequence_last_logits: empty sequence");
  std::vector<float> last;
  for (std::int32_t t : token_ids) last = forward_token(session, t);
  return last;
}

void forward_gguf_layer_cpu(const Qwen4ExpGgufSet& gguf, Qwen4ExpSessionCpu& session, int layer_idx,
                            std::int32_t token_id, int pos, std::span<const float> res_hc_in,
                            std::span<float> res_hc_out) {
  const Qwen4ExpConfig& c = gguf.config();
  const int H = c.hidden_size;
  const int hc = c.hc_mult;
  const int hc_dim = c.hc_dim();
  const std::string p = "blk." + std::to_string(layer_idx) + ".";
  auto load = [&](const std::string& name) { return gguf.read_tensor_f32(gguf.require_tensor(name)); };

  std::vector<float> res_hc(res_hc_in.begin(), res_hc_in.end());
  std::vector<float> next_res(static_cast<std::size_t>(hc_dim));

  if (c.is_ple(layer_idx)) {
    std::vector<std::int32_t> ple_rows(static_cast<std::size_t>(c.ple_n_heads()));
    ple_hash_step(c, token_id, session.ple_token_history, ple_rows);
    std::vector<float> ple_emb(static_cast<std::size_t>(H));
    const auto& ple_t = gguf.require_tensor("per_layer_token_embd.weight");
    for (int h = 0; h < c.ple_n_heads(); ++h) {
      gguf.read_tensor_row_f32(ple_t, static_cast<std::uint64_t>(ple_rows[h]),
                               std::span<float>(ple_emb.data() + static_cast<std::size_t>(h) * c.ple_head_dim,
                                                static_cast<std::size_t>(c.ple_head_dim)));
    }
    ple_step_f32(c, res_hc, ple_emb, load(p + "ple_key.weight"), load(p + "ple_value.weight"),
                 load(p + "ple_norm_key.weight"), load(p + "ple_norm_query.weight"), load(p + "ple_norm_conv.weight"),
                 load(p + "ple_conv1d.weight"), session.ple_conv_states[static_cast<std::size_t>(layer_idx)], next_res);
    res_hc.swap(next_res);
  }

  std::vector<float> cur(static_cast<std::size_t>(H));
  std::vector<float> block_out(static_cast<std::size_t>(H));
  std::vector<float> inject(static_cast<std::size_t>(hc));

  hc_mix_f32(res_hc, load(p + "hc_attn_norm.weight"), load(p + "hc_attn_down.weight"), load(p + "hc_attn_up.weight"),
             load(p + "hc_attn_inject.weight"), hc, H, c.hc_low_rank, c.rms_norm_eps, cur, inject);

  if (c.is_linear(layer_idx)) {
    std::vector<float> qkv(static_cast<std::size_t>(c.gdn_conv_dim()));
    std::vector<float> z(static_cast<std::size_t>(c.gdn_value_dim()));
    std::vector<float> alpha(static_cast<std::size_t>(c.ssm_time_step_rank));
    std::vector<float> beta(static_cast<std::size_t>(c.ssm_time_step_rank));
    matvec_f32(load(p + "attn_qkv.weight"), cur, c.gdn_conv_dim(), H, qkv);
    matvec_f32(load(p + "attn_gate.weight"), cur, c.gdn_value_dim(), H, z);
    matvec_f32(load(p + "ssm_alpha.weight"), cur, c.ssm_time_step_rank, H, alpha);
    matvec_f32(load(p + "ssm_beta.weight"), cur, c.ssm_time_step_rank, H, beta);
    std::vector<float> gdn_out(static_cast<std::size_t>(c.gdn_value_dim()));
    gdn_step_sigmoid_f32(qkv, z, alpha, beta, load(p + "ssm_conv1d.weight"), load(p + "ssm_dt.bias"),
                         load(p + "ssm_a"), load(p + "ssm_norm.weight"), c.rms_norm_eps, c.ssm_conv_kernel,
                         c.ssm_group_count, c.ssm_time_step_rank, c.ssm_state_size,
                         session.gdn_conv_states[layer_idx], session.gdn_recurrent_states[layer_idx], gdn_out);
    matvec_f32(load(p + "ssm_out.weight"), gdn_out, H, c.gdn_value_dim(), block_out);
  } else {
    std::vector<int> sel;
    if (c.has_qsa(layer_idx)) {
      std::vector<float> iq(static_cast<std::size_t>(c.indexer_head_count) * c.indexer_key_length);
      std::vector<float> ik(static_cast<std::size_t>(c.indexer_key_length));
      matvec_f32(load(p + "indexer.q_proj.weight"), cur, c.indexer_head_count * c.indexer_key_length, H, iq);
      matvec_f32(load(p + "indexer.k_proj.weight"), cur, c.indexer_key_length, H, ik);
      sel = qsa_select_positions_f32(iq, ik, load(p + "indexer.q_norm.weight"), load(p + "indexer.k_norm.weight"),
                                     c.rms_norm_eps, c.rope_theta, c.rotary_dim, c.indexer_head_count,
                                     c.indexer_key_length, c.compress_ratios[layer_idx], c.indexer_top_k, pos,
                                     session.idx_k_caches[layer_idx]);
    }
    std::vector<float> qg(static_cast<std::size_t>(c.full_q_out_dim()));
    std::vector<float> kr(static_cast<std::size_t>(c.full_kv_out_dim()));
    std::vector<float> vr(static_cast<std::size_t>(c.full_kv_out_dim()));
    matvec_f32(load(p + "attn_q.weight"), cur, c.full_q_out_dim(), H, qg);
    matvec_f32(load(p + "attn_k.weight"), cur, c.full_kv_out_dim(), H, kr);
    matvec_f32(load(p + "attn_v.weight"), cur, c.full_kv_out_dim(), H, vr);
    std::vector<float> attn_out(static_cast<std::size_t>(c.num_attention_heads) * c.head_dim);
    full_attn_qsa_step_f32(qg, kr, vr, load(p + "attn_q_norm.weight"), load(p + "attn_k_norm.weight"), c.rms_norm_eps,
                           c.rope_theta, c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rotary_dim, pos,
                           sel, session.k_caches[layer_idx], session.v_caches[layer_idx], attn_out);
    matvec_f32(load(p + "attn_output.weight"), attn_out, H, c.num_attention_heads * c.head_dim, block_out);
  }

  hc_combine_f32(res_hc, block_out, inject, hc, H, next_res);
  res_hc.swap(next_res);

  hc_mix_f32(res_hc, load(p + "hc_ffn_norm.weight"), load(p + "hc_ffn_down.weight"), load(p + "hc_ffn_up.weight"),
             load(p + "hc_ffn_inject.weight"), hc, H, c.hc_low_rank, c.rms_norm_eps, cur, inject);

  // Streaming top-K MoE: read ONLY the selected experts via pread (10/512 = 1.95% of MoE weights!).
  std::vector<float> r_logits(static_cast<std::size_t>(c.expert_count));
  matvec_f32(load(p + "ffn_gate_inp.weight"), cur, c.expert_count, H, r_logits);
  const auto chosen = route_top_k_softmax(r_logits, c.expert_used_count, c.expert_weights_scale);

  const auto& t_gate = gguf.require_tensor(p + "ffn_gate_exps.weight");
  const auto& t_up = gguf.require_tensor(p + "ffn_up_exps.weight");
  const auto& t_down = gguf.require_tensor(p + "ffn_down_exps.weight");
  std::fill(block_out.begin(), block_out.end(), 0.0f);
  std::vector<float> e_out(static_cast<std::size_t>(H));
  for (const auto& ch : chosen) {
    const auto wg = gguf.read_tensor_rows_f32(
        t_gate, static_cast<std::uint64_t>(ch.expert) * c.expert_ffn_length, c.expert_ffn_length);
    const auto wu = gguf.read_tensor_rows_f32(
        t_up, static_cast<std::uint64_t>(ch.expert) * c.expert_ffn_length, c.expert_ffn_length);
    const auto wd = gguf.read_tensor_rows_f32(t_down, static_cast<std::uint64_t>(ch.expert) * H, H);
    swiglu_ffn_f32(cur, wg, wu, wd, H, c.expert_ffn_length, e_out);
    for (int i = 0; i < H; ++i) block_out[static_cast<std::size_t>(i)] += ch.weight * e_out[i];
  }
  if (c.shared_expert_ffn_length > 0) {
    std::vector<float> sh_out(static_cast<std::size_t>(H));
    swiglu_ffn_f32(cur, load(p + "ffn_gate_shexp.weight"), load(p + "ffn_up_shexp.weight"),
                   load(p + "ffn_down_shexp.weight"), H, c.shared_expert_ffn_length, sh_out);
    const auto w_sg = load(p + "ffn_gate_inp_shexp.weight");
    double sg_dot = 0.0;
    for (int i = 0; i < H; ++i) sg_dot += double(w_sg[i]) * double(cur[i]);
    const float sg = sigmoid_f32(static_cast<float>(sg_dot));
    for (int i = 0; i < H; ++i) block_out[static_cast<std::size_t>(i)] += sg * sh_out[i];
  }

  hc_combine_f32(res_hc, block_out, inject, hc, H, res_hc_out);
}

void matvec_gguf_tensor_f32(const Qwen4ExpGgufSet& gguf, const GgufTensorEntry& entry, std::span<const float> x,
                            std::span<float> y, int num_threads) {
  if (entry.dims.size() != 2) {
    throw std::invalid_argument("matvec_gguf_tensor_f32: expected 2D tensor " + entry.name);
  }
  const std::uint64_t in_features = entry.dims[0];
  const std::uint64_t out_features = entry.dims[1];
  if (x.size() != in_features || y.size() != out_features) {
    throw std::invalid_argument("matvec_gguf_tensor_f32: dimension mismatch on " + entry.name);
  }
  const int n_threads = num_threads > 0 ? num_threads
                                        : static_cast<int>(std::max(1u, std::min(14u, std::thread::hardware_concurrency())));
  const std::uint64_t rb = entry.row_bytes();
  const std::uint64_t slab_rows = 4096ull;
  std::vector<std::uint8_t> raw(std::min(out_features, slab_rows) * rb);

  for (std::uint64_t base = 0; base < out_features; base += slab_rows) {
    const std::uint64_t cnt = std::min(slab_rows, out_features - base);
    gguf.read_tensor_rows_raw(entry, base, cnt, std::span<std::uint8_t>(raw.data(), cnt * rb));
    const int active_threads = static_cast<int>(std::min<std::uint64_t>(static_cast<std::uint64_t>(n_threads), cnt));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(active_threads));
    for (int t = 0; t < active_threads; ++t) {
      const std::uint64_t r0 = (cnt * static_cast<std::uint64_t>(t)) / static_cast<std::uint64_t>(active_threads);
      const std::uint64_t r1 = (cnt * static_cast<std::uint64_t>(t + 1)) / static_cast<std::uint64_t>(active_threads);
      pool.emplace_back([&, r0, r1]() {
        std::vector<float> row_f32(in_features);
        for (std::uint64_t r = r0; r < r1; ++r) {
          dequantize_row_to_f32(entry.type_id, std::span<const std::uint8_t>(raw.data() + r * rb, rb), row_f32);
          double acc = 0.0;
          for (std::uint64_t c = 0; c < in_features; ++c) acc += double(row_f32[c]) * double(x[c]);
          y[base + r] = static_cast<float>(acc);
        }
      });
    }
    for (auto& th : pool) th.join();
  }
}

std::vector<float> forward_gguf_token_logits_cpu(const Qwen4ExpGgufSet& gguf, Qwen4ExpSessionCpu& session,
                                                 std::int32_t token_id, bool compute_logits,
                                                 std::vector<double>* layer_l2_out,
                                                 std::vector<float>* result_norm_out, int num_threads) {
  const Qwen4ExpConfig& c = gguf.config();
  if (token_id < 0 || token_id >= c.vocab_size) {
    throw std::out_of_range("forward_gguf_token_logits_cpu: token_id out of range");
  }
  const int pos = session.position();
  const int H = c.hidden_size;
  const int hc = c.hc_mult;
  const int hc_dim = c.hc_dim();

  std::vector<float> emb(static_cast<std::size_t>(H));
  gguf.read_tensor_row_f32(gguf.require_tensor("token_embd.weight"), static_cast<std::uint64_t>(token_id), emb);
  std::vector<float> res_a(static_cast<std::size_t>(hc_dim));
  std::vector<float> res_b(static_cast<std::size_t>(hc_dim));
  for (int s = 0; s < hc; ++s) {
    std::copy_n(emb.data(), H, res_a.data() + static_cast<std::size_t>(s) * H);
  }

  if (layer_l2_out) {
    layer_l2_out->clear();
    layer_l2_out->reserve(static_cast<std::size_t>(c.num_hidden_layers));
  }
  for (int il = 0; il < c.num_hidden_layers; ++il) {
    forward_gguf_layer_cpu(gguf, session, il, token_id, pos, res_a, res_b);
    res_a.swap(res_b);
    if (layer_l2_out) {
      double ss = 0.0;
      for (float v : res_a) ss += double(v) * double(v);
      layer_l2_out->push_back(std::sqrt(ss));
    }
  }

  std::vector<float> cur(static_cast<std::size_t>(H));
  hc_mix_f32(res_a, gguf.read_tensor_f32(gguf.require_tensor("output_hc_norm.weight")),
             gguf.read_tensor_f32(gguf.require_tensor("output_hc_down.weight")),
             gguf.read_tensor_f32(gguf.require_tensor("output_hc_up.weight")), {}, hc, H, c.hc_low_rank,
             c.rms_norm_eps, cur, {});
  if (result_norm_out) {
    *result_norm_out = cur;
  }

  session.committed_tokens.push_back(token_id);

  std::vector<float> logits;
  if (compute_logits) {
    logits.resize(static_cast<std::size_t>(c.vocab_size));
    const GgufTensorEntry* out_t = gguf.find_tensor("output.weight");
    if (!out_t) out_t = &gguf.require_tensor("token_embd.weight");
    matvec_gguf_tensor_f32(gguf, *out_t, cur, logits, num_threads);
  }
  return logits;
}

}  // namespace ingine::backends::qwen4exp

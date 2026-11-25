#include "backends/qwen38/qwen38_cpu_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace ingine::backends::qwen38 {
namespace {

inline float silu_f32(float x) noexcept { return x / (1.0f + std::exp(-x)); }

inline float softplus_f32(float x) noexcept {
  if (x > 20.0f) return x;
  if (x < -20.0f) return std::exp(x);
  return std::log1p(std::exp(x));
}

void matvec_f32(std::span<const float> x, std::span<const float> w, int out_f, int in_f, std::span<float> out) {
  for (int r = 0; r < out_f; ++r) {
    const float* row = w.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_f);
    double acc = 0.0;
    for (int c = 0; c < in_f; ++c) acc += static_cast<double>(x[c]) * static_cast<double>(row[c]);
    out[r] = static_cast<float>(acc);
  }
}

void quantize_nvfp4_matrix(std::span<const float> w, int out_f, int in_f, std::vector<std::uint8_t>& packed_out,
                            std::vector<std::uint8_t>& block_scales_out, float& global_scale2_out) {
  static constexpr float kE2M1Pos[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  if (in_f % 16 != 0) throw std::invalid_argument("NVFP4 requires in_f divisible by 16");

  float max_abs = 0.0f;
  for (float v : w) max_abs = std::max(max_abs, std::fabs(v));
  global_scale2_out = max_abs > 0.0f ? (max_abs / (6.0f * 32.0f)) : 1.0f;

  packed_out.assign(static_cast<std::size_t>(out_f) * static_cast<std::size_t>(in_f / 2), 0u);
  block_scales_out.assign(static_cast<std::size_t>(out_f) * static_cast<std::size_t>(in_f / 16), 0u);

  const int blocks_per_row = in_f / 16;
  for (int r = 0; r < out_f; ++r) {
    const float* row = w.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_f);
    for (int b = 0; b < blocks_per_row; ++b) {
      float blk_max = 0.0f;
      for (int i = 0; i < 16; ++i) blk_max = std::max(blk_max, std::fabs(row[b * 16 + i]));
      const float raw_bs = blk_max / (6.0f * global_scale2_out);
      const std::uint8_t qbs = quantize_e4m3fn(raw_bs);
      block_scales_out[static_cast<std::size_t>(r) * static_cast<std::size_t>(blocks_per_row) + b] = qbs;
      const float eff_scale = std::max(1e-12f, decode_e4m3fn(qbs) * global_scale2_out);

      for (int i = 0; i < 16; i += 2) {
        auto encode_one = [&](float v) -> std::uint8_t {
          const std::uint8_t sign = v < 0.0f ? 8u : 0u;
          const float av = std::fabs(v) / eff_scale;
          std::uint8_t best = 0;
          float best_err = std::fabs(av - kE2M1Pos[0]);
          for (std::uint8_t code = 1; code < 8; ++code) {
            const float err = std::fabs(av - kE2M1Pos[code]);
            if (err < best_err) {
              best_err = err;
              best = code;
            }
          }
          return static_cast<std::uint8_t>(sign | best);
        };
        const std::uint8_t lo = encode_one(row[b * 16 + i]);
        const std::uint8_t hi = encode_one(row[b * 16 + i + 1]);
        packed_out[static_cast<std::size_t>(r) * static_cast<std::size_t>(in_f / 2) + (b * 8 + i / 2)] =
            static_cast<std::uint8_t>((hi << 4) | lo);
      }
    }
  }
}

void quantize_fp8_matrix(std::span<const float> w, int out_f, int in_f, std::vector<std::uint8_t>& w_e4m3_out,
                           float& wscale_out) {
  float max_abs = 0.0f;
  for (float v : w) max_abs = std::max(max_abs, std::fabs(v));
  wscale_out = max_abs > 0.0f ? (max_abs / 240.0f) : 1.0f;
  w_e4m3_out.resize(static_cast<std::size_t>(out_f) * static_cast<std::size_t>(in_f));
  for (std::size_t i = 0; i < w_e4m3_out.size(); ++i) {
    w_e4m3_out[i] = quantize_e4m3fn(w[i] / wscale_out);
  }
}

}  // namespace

float bf16_to_f32(std::uint16_t bits) noexcept {
  const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
  float f = 0.0f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

std::uint16_t f32_to_bf16(float x) noexcept {
  std::uint32_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
  const std::uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<std::uint16_t>(u >> 16);
}

float decode_e4m3fn(std::uint8_t bits) noexcept {
  const std::uint8_t sign = (bits >> 7) & 1u;
  const std::uint8_t exp = (bits >> 3) & 0x0fu;
  const std::uint8_t mant = bits & 0x07u;
  if (exp == 0x0fu && mant == 0x07u) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  float val = 0.0f;
  if (exp == 0) {
    val = std::ldexp(static_cast<float>(mant) / 8.0f, 1 - 7);
  } else {
    val = std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, static_cast<int>(exp) - 7);
  }
  return sign ? -val : val;
}

std::uint8_t quantize_e4m3fn(float x) noexcept {
  if (std::isnan(x)) return 0x7fu;
  const std::uint8_t sign = std::signbit(x) ? 0x80u : 0x00u;
  float ax = std::fabs(x);
  if (ax == 0.0f) return sign;
  if (ax >= 448.0f) return static_cast<std::uint8_t>(sign | 0x7eu);

  // Brute search over all 127 finite non-negative E4M3FN magnitudes guarantees exact nearest-even ties.
  std::uint8_t best = 0;
  float best_err = std::numeric_limits<float>::infinity();
  for (std::uint8_t code = 0; code <= 0x7eu; ++code) {
    const float candidate = decode_e4m3fn(code);
    const float err = std::fabs(ax - candidate);
    if (err < best_err || (err == best_err && ((code & 1u) == 0u))) {
      best_err = err;
      best = code;
    }
  }
  return static_cast<std::uint8_t>(sign | best);
}

float decode_e2m1(std::uint8_t nibble) noexcept {
  static constexpr float kTable[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return kTable[nibble & 0x0fu];
}

void decode_nvfp4_row(std::span<const std::uint8_t> packed_u8, std::span<const std::uint8_t> block_scale_e4m3,
                      float weight_scale_2, std::span<float> out_f32) {
  const std::size_t in_f = out_f32.size();
  if (in_f % 16 != 0 || packed_u8.size() != in_f / 2 || block_scale_e4m3.size() != in_f / 16) {
    throw std::invalid_argument("decode_nvfp4_row shape mismatch (must satisfy 16-element NVFP4 blocks)");
  }
  for (std::size_t b = 0; b < in_f / 16; ++b) {
    const float scale = decode_e4m3fn(block_scale_e4m3[b]) * weight_scale_2;
    for (std::size_t i = 0; i < 8; ++i) {
      const std::uint8_t byte = packed_u8[b * 8 + i];
      out_f32[b * 16 + 2 * i] = decode_e2m1(byte & 0x0fu) * scale;
      out_f32[b * 16 + 2 * i + 1] = decode_e2m1((byte >> 4) & 0x0fu) * scale;
    }
  }
}

void fp8_weight_only_linear(std::span<const float> x_in, std::span<const std::uint8_t> w_e4m3, float weight_scale,
                            int out_features, int in_features, std::span<float> out) {
  if (static_cast<int>(x_in.size()) != in_features || static_cast<int>(out.size()) != out_features ||
      w_e4m3.size() != static_cast<std::size_t>(out_features) * static_cast<std::size_t>(in_features)) {
    throw std::invalid_argument("fp8_weight_only_linear dimension mismatch");
  }
  for (int r = 0; r < out_features; ++r) {
    const std::size_t off = static_cast<std::size_t>(r) * static_cast<std::size_t>(in_features);
    double acc = 0.0;
    for (int c = 0; c < in_features; ++c) {
      acc += static_cast<double>(x_in[c]) * static_cast<double>(decode_e4m3fn(w_e4m3[off + c]));
    }
    out[r] = static_cast<float>(acc * static_cast<double>(weight_scale));
  }
}

void fp8_calibrated_linear(std::span<const float> x_in, float input_scale, std::span<const std::uint8_t> w_e4m3,
                           float weight_scale, int out_features, int in_features, std::span<float> out) {
  if (static_cast<int>(x_in.size()) != in_features || static_cast<int>(out.size()) != out_features ||
      w_e4m3.size() != static_cast<std::size_t>(out_features) * static_cast<std::size_t>(in_features) ||
      !(input_scale > 0.0f)) {
    throw std::invalid_argument("fp8_calibrated_linear dimension or scale mismatch");
  }
  std::vector<float> x_q(static_cast<std::size_t>(in_features));
  for (int c = 0; c < in_features; ++c) {
    x_q[c] = decode_e4m3fn(quantize_e4m3fn(x_in[c] / input_scale)) * input_scale;
  }
  fp8_weight_only_linear(x_q, w_e4m3, weight_scale, out_features, in_features, out);
}

void nvfp4_weight_only_linear(std::span<const float> x_in, std::span<const std::uint8_t> packed_u8,
                              std::span<const std::uint8_t> block_scales_e4m3, float weight_scale_2, int out_features,
                              int in_features, std::span<float> out) {
  if (in_features % 16 != 0 || static_cast<int>(x_in.size()) != in_features ||
      static_cast<int>(out.size()) != out_features ||
      packed_u8.size() != static_cast<std::size_t>(out_features) * static_cast<std::size_t>(in_features / 2) ||
      block_scales_e4m3.size() != static_cast<std::size_t>(out_features) * static_cast<std::size_t>(in_features / 16)) {
    throw std::invalid_argument("nvfp4_weight_only_linear dimension mismatch");
  }
  std::vector<float> row_w(static_cast<std::size_t>(in_features));
  for (int r = 0; r < out_features; ++r) {
    const auto packed_row =
        packed_u8.subspan(static_cast<std::size_t>(r) * static_cast<std::size_t>(in_features / 2), in_features / 2);
    const auto bs_row = block_scales_e4m3.subspan(
        static_cast<std::size_t>(r) * static_cast<std::size_t>(in_features / 16), in_features / 16);
    decode_nvfp4_row(packed_row, bs_row, weight_scale_2, row_w);
    double acc = 0.0;
    for (int c = 0; c < in_features; ++c) acc += static_cast<double>(x_in[c]) * static_cast<double>(row_w[c]);
    out[r] = static_cast<float>(acc);
  }
}

void rmsnorm_zero_centered(std::span<const float> x, std::span<const float> weight, float eps, std::span<float> out) {
  if (x.size() != weight.size() || x.size() != out.size() || x.empty()) {
    throw std::invalid_argument("rmsnorm_zero_centered dimension mismatch");
  }
  double sq = 0.0;
  for (float v : x) sq += static_cast<double>(v) * static_cast<double>(v);
  const float inv_rms = static_cast<float>(1.0 / std::sqrt(sq / static_cast<double>(x.size()) + static_cast<double>(eps)));
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = x[i] * inv_rms * (1.0f + weight[i]);
  }
}

void rmsnorm_gated_silu(std::span<const float> x, std::span<const float> gate_z, std::span<const float> weight,
                        float eps, std::span<float> out) {
  if (x.size() != gate_z.size() || x.size() != weight.size() || x.size() != out.size() || x.empty()) {
    throw std::invalid_argument("rmsnorm_gated_silu dimension mismatch");
  }
  double sq = 0.0;
  for (float v : x) sq += static_cast<double>(v) * static_cast<double>(v);
  const float inv_rms = static_cast<float>(1.0 / std::sqrt(sq / static_cast<double>(x.size()) + static_cast<double>(eps)));
  for (std::size_t i = 0; i < x.size(); ++i) {
    out[i] = weight[i] * (x[i] * inv_rms) * silu_f32(gate_z[i]);
  }
}

void apply_partial_rope_neox(std::span<float> qk_heads, int num_heads, int head_dim, int rotary_dim, int pos,
                             float rope_theta) {
  if (rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 ||
      qk_heads.size() != static_cast<std::size_t>(num_heads) * static_cast<std::size_t>(head_dim)) {
    throw std::invalid_argument("apply_partial_rope_neox dimension mismatch");
  }
  const int half = rotary_dim / 2;
  for (int h = 0; h < num_heads; ++h) {
    float* head = qk_heads.data() + static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim);
    for (int j = 0; j < half; ++j) {
      const double exponent = -2.0 * static_cast<double>(j) / static_cast<double>(rotary_dim);
      const double angle = static_cast<double>(pos) * std::pow(static_cast<double>(rope_theta), exponent);
      const float c = static_cast<float>(std::cos(angle));
      const float s = static_cast<float>(std::sin(angle));
      const float x0 = head[j];
      const float x1 = head[j + half];
      head[j] = x0 * c - x1 * s;
      head[j + half] = x1 * c + x0 * s;
    }
  }
}

void causal_conv1d_silu_step(std::span<const float> x_conv_in, std::span<const float> conv_weight, int conv_dim,
                             int kernel_size, std::vector<float>& conv_state, std::span<float> out) {
  if (kernel_size < 1 || static_cast<int>(x_conv_in.size()) != conv_dim ||
      static_cast<int>(out.size()) != conv_dim ||
      conv_weight.size() != static_cast<std::size_t>(conv_dim) * static_cast<std::size_t>(kernel_size) ||
      conv_state.size() != static_cast<std::size_t>(conv_dim) * static_cast<std::size_t>(kernel_size)) {
    throw std::invalid_argument("causal_conv1d_silu_step dimension mismatch");
  }
  for (int c = 0; c < conv_dim; ++c) {
    float* st = conv_state.data() + static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel_size);
    const float* w = conv_weight.data() + static_cast<std::size_t>(c) * static_cast<std::size_t>(kernel_size);
    for (int k = 0; k + 1 < kernel_size; ++k) st[k] = st[k + 1];
    st[kernel_size - 1] = x_conv_in[c];
    double acc = 0.0;
    for (int k = 0; k < kernel_size; ++k) acc += static_cast<double>(st[k]) * static_cast<double>(w[k]);
    out[c] = silu_f32(static_cast<float>(acc));
  }
}

void causal_conv1d_silu_prefill(std::span<const float> x_seq, std::span<const float> conv_weight, int seq_len,
                                int conv_dim, int kernel_size, std::vector<float>& conv_state,
                                std::span<float> out_seq) {
  if (seq_len < 0 || x_seq.size() != static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(conv_dim) ||
      out_seq.size() != x_seq.size()) {
    throw std::invalid_argument("causal_conv1d_silu_prefill dimension mismatch");
  }
  conv_state.assign(static_cast<std::size_t>(conv_dim) * static_cast<std::size_t>(kernel_size), 0.0f);
  for (int t = 0; t < seq_len; ++t) {
    const auto x_t = x_seq.subspan(static_cast<std::size_t>(t) * static_cast<std::size_t>(conv_dim), conv_dim);
    const auto out_t = out_seq.subspan(static_cast<std::size_t>(t) * static_cast<std::size_t>(conv_dim), conv_dim);
    causal_conv1d_silu_step(x_t, conv_weight, conv_dim, kernel_size, conv_state, out_t);
  }
}

void gdn_recurrent_step(std::span<const float> q_kheads, std::span<const float> k_kheads,
                        std::span<const float> v_vheads, std::span<const float> z_vheads,
                        std::span<const float> a_vheads, std::span<const float> b_vheads, std::span<const float> a_log,
                        std::span<const float> dt_bias, std::span<const float> norm_weight, float rms_eps,
                        int num_k_heads, int num_v_heads, int k_dim, int v_dim, std::vector<float>& recurrent_state,
                        std::span<float> out_vdim) {
  if (num_k_heads <= 0 || num_v_heads <= 0 || num_v_heads % num_k_heads != 0 ||
      q_kheads.size() != static_cast<std::size_t>(num_k_heads * k_dim) ||
      k_kheads.size() != static_cast<std::size_t>(num_k_heads * k_dim) ||
      v_vheads.size() != static_cast<std::size_t>(num_v_heads * v_dim) ||
      z_vheads.size() != static_cast<std::size_t>(num_v_heads * v_dim) ||
      out_vdim.size() != static_cast<std::size_t>(num_v_heads * v_dim) ||
      static_cast<int>(a_vheads.size()) != num_v_heads || static_cast<int>(b_vheads.size()) != num_v_heads ||
      static_cast<int>(a_log.size()) != num_v_heads || static_cast<int>(dt_bias.size()) != num_v_heads ||
      static_cast<int>(norm_weight.size()) != v_dim ||
      recurrent_state.size() != static_cast<std::size_t>(num_v_heads) * static_cast<std::size_t>(k_dim) *
                                    static_cast<std::size_t>(v_dim)) {
    throw std::invalid_argument("gdn_recurrent_step dimension mismatch");
  }

  const float q_scale = 1.0f / std::sqrt(static_cast<float>(k_dim));
  std::vector<float> q_norm(static_cast<std::size_t>(num_k_heads * k_dim));
  std::vector<float> k_norm(static_cast<std::size_t>(num_k_heads * k_dim));
  for (int kh = 0; kh < num_k_heads; ++kh) {
    double q_sq = 0.0;
    double k_sq = 0.0;
    const std::size_t base = static_cast<std::size_t>(kh) * static_cast<std::size_t>(k_dim);
    for (int d = 0; d < k_dim; ++d) {
      q_sq += static_cast<double>(q_kheads[base + d]) * static_cast<double>(q_kheads[base + d]);
      k_sq += static_cast<double>(k_kheads[base + d]) * static_cast<double>(k_kheads[base + d]);
    }
    const float q_inv = static_cast<float>(1.0 / std::sqrt(q_sq + 1e-6)) * q_scale;
    const float k_inv = static_cast<float>(1.0 / std::sqrt(k_sq + 1e-6));
    for (int d = 0; d < k_dim; ++d) {
      q_norm[base + d] = q_kheads[base + d] * q_inv;
      k_norm[base + d] = k_kheads[base + d] * k_inv;
    }
  }

  const int repeat_factor = num_v_heads / num_k_heads;
  std::vector<float> kv_mem(static_cast<std::size_t>(v_dim), 0.0f);
  std::vector<float> delta(static_cast<std::size_t>(v_dim), 0.0f);
  std::vector<float> core_out(static_cast<std::size_t>(v_dim), 0.0f);

  for (int vh = 0; vh < num_v_heads; ++vh) {
    const int kh = vh / repeat_factor;
    const float* qh = q_norm.data() + static_cast<std::size_t>(kh) * static_cast<std::size_t>(k_dim);
    const float* kh_ptr = k_norm.data() + static_cast<std::size_t>(kh) * static_cast<std::size_t>(k_dim);
    const float* vh_ptr = v_vheads.data() + static_cast<std::size_t>(vh) * static_cast<std::size_t>(v_dim);
    const float* zh_ptr = z_vheads.data() + static_cast<std::size_t>(vh) * static_cast<std::size_t>(v_dim);
    float* state = recurrent_state.data() +
                   static_cast<std::size_t>(vh) * static_cast<std::size_t>(k_dim) * static_cast<std::size_t>(v_dim);

    const float beta = 1.0f / (1.0f + std::exp(-b_vheads[vh]));
    const float g = -std::exp(a_log[vh]) * softplus_f32(a_vheads[vh] + dt_bias[vh]);
    const float decay = std::exp(g);

    for (std::size_t idx = 0; idx < static_cast<std::size_t>(k_dim) * static_cast<std::size_t>(v_dim); ++idx) {
      state[idx] *= decay;
    }

    std::fill(kv_mem.begin(), kv_mem.end(), 0.0f);
    for (int kd = 0; kd < k_dim; ++kd) {
      const float kk = kh_ptr[kd];
      const float* row = state + static_cast<std::size_t>(kd) * static_cast<std::size_t>(v_dim);
      for (int vd = 0; vd < v_dim; ++vd) kv_mem[vd] += row[vd] * kk;
    }

    for (int vd = 0; vd < v_dim; ++vd) delta[vd] = (vh_ptr[vd] - kv_mem[vd]) * beta;

    for (int kd = 0; kd < k_dim; ++kd) {
      const float kk = kh_ptr[kd];
      float* row = state + static_cast<std::size_t>(kd) * static_cast<std::size_t>(v_dim);
      for (int vd = 0; vd < v_dim; ++vd) row[vd] += kk * delta[vd];
    }

    std::fill(core_out.begin(), core_out.end(), 0.0f);
    for (int kd = 0; kd < k_dim; ++kd) {
      const float qk = qh[kd];
      const float* row = state + static_cast<std::size_t>(kd) * static_cast<std::size_t>(v_dim);
      for (int vd = 0; vd < v_dim; ++vd) core_out[vd] += row[vd] * qk;
    }

    auto out_vh = out_vdim.subspan(static_cast<std::size_t>(vh) * static_cast<std::size_t>(v_dim), v_dim);
    rmsnorm_gated_silu(core_out, std::span<const float>(zh_ptr, v_dim), norm_weight, rms_eps, out_vh);
  }
}

void gdn_recurrent_prefill(std::span<const float> q_seq, std::span<const float> k_seq, std::span<const float> v_seq,
                           std::span<const float> z_seq, std::span<const float> a_seq, std::span<const float> b_seq,
                           std::span<const float> a_log, std::span<const float> dt_bias,
                           std::span<const float> norm_weight, float rms_eps, int seq_len, int num_k_heads,
                           int num_v_heads, int k_dim, int v_dim, std::vector<float>& recurrent_state,
                           std::span<float> out_seq) {
  const std::size_t k_stride = static_cast<std::size_t>(num_k_heads * k_dim);
  const std::size_t v_stride = static_cast<std::size_t>(num_v_heads * v_dim);
  for (int t = 0; t < seq_len; ++t) {
    gdn_recurrent_step(q_seq.subspan(t * k_stride, k_stride), k_seq.subspan(t * k_stride, k_stride),
                       v_seq.subspan(t * v_stride, v_stride), z_seq.subspan(t * v_stride, v_stride),
                       a_seq.subspan(t * num_v_heads, num_v_heads), b_seq.subspan(t * num_v_heads, num_v_heads),
                       a_log, dt_bias, norm_weight, rms_eps, num_k_heads, num_v_heads, k_dim, v_dim, recurrent_state,
                       out_seq.subspan(t * v_stride, v_stride));
  }
}

void gdn_chunk_prefill(std::span<const float> q_seq, std::span<const float> k_seq, std::span<const float> v_seq,
                       std::span<const float> z_seq, std::span<const float> a_seq, std::span<const float> b_seq,
                       std::span<const float> a_log, std::span<const float> dt_bias,
                       std::span<const float> norm_weight, float rms_eps, int seq_len, int chunk_size,
                       int num_k_heads, int num_v_heads, int k_dim, int v_dim, std::vector<float>& recurrent_state,
                       std::span<float> out_seq) {
  if (chunk_size < 1) throw std::invalid_argument("gdn_chunk_prefill chunk_size must be positive");
  const std::size_t k_stride = static_cast<std::size_t>(num_k_heads * k_dim);
  const std::size_t v_stride = static_cast<std::size_t>(num_v_heads * v_dim);
  if (seq_len == 0) return;
  if (q_seq.size() != static_cast<std::size_t>(seq_len) * k_stride ||
      k_seq.size() != static_cast<std::size_t>(seq_len) * k_stride ||
      v_seq.size() != static_cast<std::size_t>(seq_len) * v_stride ||
      z_seq.size() != static_cast<std::size_t>(seq_len) * v_stride ||
      out_seq.size() != static_cast<std::size_t>(seq_len) * v_stride) {
    throw std::invalid_argument("gdn_chunk_prefill dimension mismatch");
  }

  // Normalize Q/K and precompute per-step gates, then solve within each chunk via WY lower-triangular matrix.
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(k_dim));
  std::vector<float> qn(q_seq.size());
  std::vector<float> kn(k_seq.size());
  for (int t = 0; t < seq_len; ++t) {
    for (int kh = 0; kh < num_k_heads; ++kh) {
      const std::size_t base = static_cast<std::size_t>(t) * k_stride + static_cast<std::size_t>(kh * k_dim);
      double qsq = 0.0, ksq = 0.0;
      for (int d = 0; d < k_dim; ++d) {
        qsq += static_cast<double>(q_seq[base + d]) * static_cast<double>(q_seq[base + d]);
        ksq += static_cast<double>(k_seq[base + d]) * static_cast<double>(k_seq[base + d]);
      }
      const float qinv = static_cast<float>(1.0 / std::sqrt(qsq + 1e-6)) * q_scale;
      const float kinv = static_cast<float>(1.0 / std::sqrt(ksq + 1e-6));
      for (int d = 0; d < k_dim; ++d) {
        qn[base + d] = q_seq[base + d] * qinv;
        kn[base + d] = k_seq[base + d] * kinv;
      }
    }
  }

  const int repeat_factor = num_v_heads / num_k_heads;
  for (int vh = 0; vh < num_v_heads; ++vh) {
    const int kh = vh / repeat_factor;
    float* state = recurrent_state.data() +
                   static_cast<std::size_t>(vh) * static_cast<std::size_t>(k_dim) * static_cast<std::size_t>(v_dim);
    for (int c0 = 0; c0 < seq_len; c0 += chunk_size) {
      const int L = std::min(chunk_size, seq_len - c0);
      std::vector<float> g_cum(static_cast<std::size_t>(L), 0.0f);
      std::vector<float> beta(static_cast<std::size_t>(L), 0.0f);
      for (int i = 0; i < L; ++i) {
        const int t = c0 + i;
        const float gi = -std::exp(a_log[vh]) * softplus_f32(a_seq[t * num_v_heads + vh] + dt_bias[vh]);
        g_cum[i] = (i == 0 ? 0.0f : g_cum[i - 1]) + gi;
        beta[i] = 1.0f / (1.0f + std::exp(-b_seq[t * num_v_heads + vh]));
      }

      // Solve (I + strict_lower(decay_kk * beta)) u = beta * v and w = beta * k * exp(g_cum) via forward sub.
      std::vector<float> v_corr(static_cast<std::size_t>(L * v_dim), 0.0f);
      std::vector<float> u_corr(static_cast<std::size_t>(L * v_dim), 0.0f);
      std::vector<float> k_w(static_cast<std::size_t>(L * k_dim), 0.0f);
      std::vector<float> w_mat(static_cast<std::size_t>(L * k_dim), 0.0f);

      for (int i = 0; i < L; ++i) {
        const int t = c0 + i;
        const float* ki = kn.data() + static_cast<std::size_t>(t) * k_stride + static_cast<std::size_t>(kh * k_dim);
        const float* vi = v_seq.data() + static_cast<std::size_t>(t) * v_stride + static_cast<std::size_t>(vh * v_dim);
        for (int vd = 0; vd < v_dim; ++vd) v_corr[i * v_dim + vd] = vi[vd] * beta[i];
        const float kg_scale = std::exp(g_cum[i]) * beta[i];
        for (int kd = 0; kd < k_dim; ++kd) k_w[i * k_dim + kd] = ki[kd] * kg_scale;

        for (int vd = 0; vd < v_dim; ++vd) u_corr[i * v_dim + vd] = v_corr[i * v_dim + vd];
        for (int kd = 0; kd < k_dim; ++kd) w_mat[i * k_dim + kd] = k_w[i * k_dim + kd];
        for (int j = 0; j < i; ++j) {
          const int tj = c0 + j;
          const float* kj = kn.data() + static_cast<std::size_t>(tj) * k_stride + static_cast<std::size_t>(kh * k_dim);
          double dot_k = 0.0;
          for (int kd = 0; kd < k_dim; ++kd) dot_k += static_cast<double>(ki[kd]) * static_cast<double>(kj[kd]);
          const float coeff = static_cast<float>(dot_k) * std::exp(g_cum[i] - g_cum[j]) * beta[i];
          for (int vd = 0; vd < v_dim; ++vd) u_corr[i * v_dim + vd] -= coeff * u_corr[j * v_dim + vd];
          for (int kd = 0; kd < k_dim; ++kd) w_mat[i * k_dim + kd] -= coeff * w_mat[j * k_dim + kd];
        }
      }

      // Compute chunk new values v_new = u - w @ state, then causal core output, then inter-chunk state update.
      std::vector<float> v_new(static_cast<std::size_t>(L * v_dim), 0.0f);
      for (int i = 0; i < L; ++i) {
        for (int vd = 0; vd < v_dim; ++vd) {
          double ws = 0.0;
          for (int kd = 0; kd < k_dim; ++kd) {
            ws += static_cast<double>(w_mat[i * k_dim + kd]) *
                  static_cast<double>(state[static_cast<std::size_t>(kd * v_dim + vd)]);
          }
          v_new[i * v_dim + vd] = u_corr[i * v_dim + vd] - static_cast<float>(ws);
        }
      }

      std::vector<float> core(static_cast<std::size_t>(v_dim), 0.0f);
      for (int i = 0; i < L; ++i) {
        const int t = c0 + i;
        const float* qi = qn.data() + static_cast<std::size_t>(t) * k_stride + static_cast<std::size_t>(kh * k_dim);
        const float* zi = z_seq.data() + static_cast<std::size_t>(t) * v_stride + static_cast<std::size_t>(vh * v_dim);
        const float g_i_exp = std::exp(g_cum[i]);
        for (int vd = 0; vd < v_dim; ++vd) {
          double inter = 0.0;
          for (int kd = 0; kd < k_dim; ++kd) {
            inter += static_cast<double>(qi[kd]) *
                     static_cast<double>(state[static_cast<std::size_t>(kd * v_dim + vd)]);
          }
          double intra = 0.0;
          for (int j = 0; j <= i; ++j) {
            const int tj = c0 + j;
            const float* kj = kn.data() + static_cast<std::size_t>(tj) * k_stride + static_cast<std::size_t>(kh * k_dim);
            double qk = 0.0;
            for (int kd = 0; kd < k_dim; ++kd) qk += static_cast<double>(qi[kd]) * static_cast<double>(kj[kd]);
            intra += qk * static_cast<double>(std::exp(g_cum[i] - g_cum[j])) * static_cast<double>(v_new[j * v_dim + vd]);
          }
          core[vd] = static_cast<float>(inter * static_cast<double>(g_i_exp) + intra);
        }
        auto out_vh = out_seq.subspan(static_cast<std::size_t>(t) * v_stride + static_cast<std::size_t>(vh * v_dim), v_dim);
        rmsnorm_gated_silu(core, std::span<const float>(zi, v_dim), norm_weight, rms_eps, out_vh);
      }

      const float end_decay = std::exp(g_cum[L - 1]);
      for (int kd = 0; kd < k_dim; ++kd) {
        for (int vd = 0; vd < v_dim; ++vd) {
          double acc = static_cast<double>(state[static_cast<std::size_t>(kd * v_dim + vd)]) *
                       static_cast<double>(end_decay);
          for (int i = 0; i < L; ++i) {
            const int ti = c0 + i;
            const float* ki = kn.data() + static_cast<std::size_t>(ti) * k_stride + static_cast<std::size_t>(kh * k_dim);
            acc += static_cast<double>(ki[kd]) * static_cast<double>(std::exp(g_cum[L - 1] - g_cum[i])) *
                   static_cast<double>(v_new[i * v_dim + vd]);
          }
          state[static_cast<std::size_t>(kd * v_dim + vd)] = static_cast<float>(acc);
        }
      }
    }
  }
}

void full_attn_gated_step(std::span<const float> q_with_gate, std::span<const float> k_raw, std::span<const float> v,
                          std::span<const float> q_norm_w, std::span<const float> k_norm_w, float rms_eps,
                          float rope_theta, int num_heads, int num_kv_heads, int head_dim, int rotary_dim, int pos,
                          std::vector<float>& k_cache, std::vector<float>& v_cache, std::span<float> out_attn) {
  if (num_heads <= 0 || num_kv_heads <= 0 || num_heads % num_kv_heads != 0 ||
      q_with_gate.size() != static_cast<std::size_t>(num_heads * head_dim * 2) ||
      k_raw.size() != static_cast<std::size_t>(num_kv_heads * head_dim) ||
      v.size() != static_cast<std::size_t>(num_kv_heads * head_dim) ||
      out_attn.size() != static_cast<std::size_t>(num_heads * head_dim)) {
    throw std::invalid_argument("full_attn_gated_step dimension mismatch");
  }

  std::vector<float> q(static_cast<std::size_t>(num_heads * head_dim));
  std::vector<float> gate(static_cast<std::size_t>(num_heads * head_dim));
  for (int h = 0; h < num_heads; ++h) {
    const float* src = q_with_gate.data() + static_cast<std::size_t>(h * 2 * head_dim);
    std::copy_n(src, head_dim, q.data() + static_cast<std::size_t>(h * head_dim));
    std::copy_n(src + head_dim, head_dim, gate.data() + static_cast<std::size_t>(h * head_dim));
    auto qh = std::span<float>(q.data() + static_cast<std::size_t>(h * head_dim), head_dim);
    rmsnorm_zero_centered(qh, q_norm_w, rms_eps, qh);
  }
  apply_partial_rope_neox(q, num_heads, head_dim, rotary_dim, pos, rope_theta);

  std::vector<float> k_step(k_raw.begin(), k_raw.end());
  for (int kvh = 0; kvh < num_kv_heads; ++kvh) {
    auto kh = std::span<float>(k_step.data() + static_cast<std::size_t>(kvh * head_dim), head_dim);
    rmsnorm_zero_centered(kh, k_norm_w, rms_eps, kh);
  }
  apply_partial_rope_neox(k_step, num_kv_heads, head_dim, rotary_dim, pos, rope_theta);

  const std::size_t kv_stride = static_cast<std::size_t>(num_kv_heads * head_dim);
  k_cache.insert(k_cache.end(), k_step.begin(), k_step.end());
  v_cache.insert(v_cache.end(), v.begin(), v.end());
  const int cache_len = static_cast<int>(k_cache.size() / kv_stride);
  if (cache_len != pos + 1) {
    throw std::invalid_argument("full_attn_gated_step pos does not match KV cache length");
  }

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int group = num_heads / num_kv_heads;
  std::vector<float> scores(static_cast<std::size_t>(cache_len), 0.0f);

  for (int h = 0; h < num_heads; ++h) {
    const int kvh = h / group;
    const float* qh = q.data() + static_cast<std::size_t>(h * head_dim);
    float max_s = -std::numeric_limits<float>::infinity();
    for (int t = 0; t < cache_len; ++t) {
      const float* kt = k_cache.data() + static_cast<std::size_t>(t) * kv_stride + static_cast<std::size_t>(kvh * head_dim);
      double dot = 0.0;
      for (int d = 0; d < head_dim; ++d) dot += static_cast<double>(qh[d]) * static_cast<double>(kt[d]);
      scores[t] = static_cast<float>(dot) * scale;
      max_s = std::max(max_s, scores[t]);
    }
    double denom = 0.0;
    for (int t = 0; t < cache_len; ++t) {
      scores[t] = std::exp(scores[t] - max_s);
      denom += static_cast<double>(scores[t]);
    }
    const float inv_denom = static_cast<float>(1.0 / denom);
    float* oh = out_attn.data() + static_cast<std::size_t>(h * head_dim);
    const float* gh = gate.data() + static_cast<std::size_t>(h * head_dim);
    std::fill_n(oh, head_dim, 0.0f);
    for (int t = 0; t < cache_len; ++t) {
      const float w = scores[t] * inv_denom;
      const float* vt = v_cache.data() + static_cast<std::size_t>(t) * kv_stride + static_cast<std::size_t>(kvh * head_dim);
      for (int d = 0; d < head_dim; ++d) oh[d] += w * vt[d];
    }
    for (int d = 0; d < head_dim; ++d) oh[d] *= (1.0f / (1.0f + std::exp(-gh[d])));
  }
}

Qwen38MiniSession::Qwen38MiniSession(const Qwen38Config& cfg) : cfg_(cfg) {
  cfg_.validate();
  reset();
}

void Qwen38MiniSession::reset() {
  committed_tokens_.clear();
  conv_states_.clear();
  recurrent_states_.clear();
  k_caches_.clear();
  v_caches_.clear();
  conv_states_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
  recurrent_states_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
  k_caches_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
  v_caches_.resize(static_cast<std::size_t>(cfg_.num_hidden_layers));
  for (int i = 0; i < cfg_.num_hidden_layers; ++i) {
    if (cfg_.layer_types[i] == Qwen38LayerKind::LinearAttention) {
      conv_states_[i].assign(
          static_cast<std::size_t>(cfg_.linear_conv_dim()) * static_cast<std::size_t>(cfg_.linear_conv_kernel_dim),
          0.0f);
      recurrent_states_[i].assign(static_cast<std::size_t>(cfg_.linear_num_value_heads) *
                                      static_cast<std::size_t>(cfg_.linear_key_head_dim) *
                                      static_cast<std::size_t>(cfg_.linear_value_head_dim),
                                  0.0f);
    }
  }
}

std::size_t Qwen38MiniSession::state_bytes() const noexcept {
  std::size_t total = 0;
  for (const auto& s : conv_states_) total += s.size() * sizeof(float);
  for (const auto& s : recurrent_states_) total += s.size() * sizeof(float);
  for (const auto& s : k_caches_) total += s.size() * sizeof(float);
  for (const auto& s : v_caches_) total += s.size() * sizeof(float);
  return total;
}

Qwen38MiniSessionSnapshot Qwen38MiniSession::snapshot() const {
  return Qwen38MiniSessionSnapshot{committed_tokens_, conv_states_, recurrent_states_, k_caches_, v_caches_};
}

void Qwen38MiniSession::restore(const Qwen38MiniSessionSnapshot& snap) {
  if (snap.conv_states.size() != static_cast<std::size_t>(cfg_.num_hidden_layers)) {
    throw std::invalid_argument("snapshot layer count mismatch");
  }
  committed_tokens_ = snap.committed_tokens;
  conv_states_ = snap.conv_states;
  recurrent_states_ = snap.recurrent_states;
  k_caches_ = snap.k_caches;
  v_caches_ = snap.v_caches;
}

std::vector<float> Qwen38MiniSession::step(const Qwen38MiniModel& model, std::int32_t token_id) {
  return model.forward_token(*this, token_id);
}

std::vector<float> Qwen38MiniSession::prefill_last_logits(const Qwen38MiniModel& model,
                                                          std::span<const std::int32_t> token_ids) {
  if (token_ids.empty()) throw std::invalid_argument("prefill_last_logits requires non-empty tokens");
  std::vector<float> logits;
  for (std::int32_t tid : token_ids) logits = step(model, tid);
  return logits;
}

Qwen38MiniModel Qwen38MiniModel::make_synthetic(std::uint32_t seed) {
  Qwen38MiniModel m;
  Qwen38Config& c = m.cfg_;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForCausalLM"};
  c.vocab_size = 64;
  c.hidden_size = 32;
  c.intermediate_size = 64;
  c.num_hidden_layers = 4;
  c.max_position_embeddings = 256;
  c.rms_norm_eps = 1e-6f;
  c.tie_word_embeddings = false;
  c.full_attention_interval = 4;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 16;
  c.attention_bias = false;
  c.attn_output_gate = true;
  c.rope_theta = 10000000.0f;
  c.partial_rotary_factor = 0.25f;
  c.rotary_dim = 4;
  c.linear_conv_kernel_dim = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.mamba_ssm_dtype = "float32";
  c.layer_types = {Qwen38LayerKind::LinearAttention, Qwen38LayerKind::LinearAttention,
                   Qwen38LayerKind::LinearAttention, Qwen38LayerKind::FullAttention};
  c.num_linear_layers = 3;
  c.num_full_layers = 1;
  c.weight_precision = Qwen38WeightPrecision::MixedFp8Nvfp4;
  c.quant_method = "modelopt";
  c.quant_algo = "MIXED_PRECISION";
  c.nvfp4_group_size = 16;
  c.validate();

  std::mt19937 rng(seed);
  auto rand_vec = [&](std::size_t n, float stddev) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
  };

  m.embed_tokens_ = rand_vec(static_cast<std::size_t>(c.vocab_size * c.hidden_size), 0.1f);
  m.final_norm_ = rand_vec(static_cast<std::size_t>(c.hidden_size), 0.02f);
  const auto lm_head_f = rand_vec(static_cast<std::size_t>(c.vocab_size * c.hidden_size), 0.08f);
  quantize_nvfp4_matrix(lm_head_f, c.vocab_size, c.hidden_size, m.lm_head_nvfp4_, m.lm_head_scales_, m.lm_head_scale2_);

  m.layers_.resize(static_cast<std::size_t>(c.num_hidden_layers));
  for (int i = 0; i < c.num_hidden_layers; ++i) {
    auto& lw = m.layers_[i];
    lw.kind = c.layer_types[i];
    lw.input_layernorm = rand_vec(static_cast<std::size_t>(c.hidden_size), 0.02f);
    lw.post_attention_layernorm = rand_vec(static_cast<std::size_t>(c.hidden_size), 0.02f);

    if (lw.kind == Qwen38LayerKind::LinearAttention) {
      quantize_fp8_matrix(rand_vec(static_cast<std::size_t>(c.linear_conv_dim() * c.hidden_size), 0.08f),
                          c.linear_conv_dim(), c.hidden_size, lw.in_proj_qkv_fp8, lw.in_proj_qkv_wscale);
      quantize_fp8_matrix(rand_vec(static_cast<std::size_t>(c.linear_value_dim() * c.hidden_size), 0.08f),
                          c.linear_value_dim(), c.hidden_size, lw.in_proj_z_fp8, lw.in_proj_z_wscale);
      quantize_fp8_matrix(rand_vec(static_cast<std::size_t>(c.hidden_size * c.linear_value_dim()), 0.08f),
                          c.hidden_size, c.linear_value_dim(), lw.out_proj_fp8, lw.out_proj_wscale);
      lw.in_proj_a = rand_vec(static_cast<std::size_t>(c.linear_num_value_heads * c.hidden_size), 0.05f);
      lw.in_proj_b = rand_vec(static_cast<std::size_t>(c.linear_num_value_heads * c.hidden_size), 0.05f);
      lw.A_log = rand_vec(static_cast<std::size_t>(c.linear_num_value_heads), 0.1f);
      lw.dt_bias = rand_vec(static_cast<std::size_t>(c.linear_num_value_heads), 0.05f);
      lw.conv1d_weight = rand_vec(static_cast<std::size_t>(c.linear_conv_dim() * c.linear_conv_kernel_dim), 0.1f);
      lw.linear_norm_weight = std::vector<float>(static_cast<std::size_t>(c.linear_value_head_dim), 1.0f);
    } else {
      quantize_fp8_matrix(rand_vec(static_cast<std::size_t>(c.full_q_proj_out_dim() * c.hidden_size), 0.08f),
                          c.full_q_proj_out_dim(), c.hidden_size, lw.q_proj_fp8, lw.q_proj_wscale);
      quantize_fp8_matrix(rand_vec(static_cast<std::size_t>(c.full_kv_proj_out_dim() * c.hidden_size), 0.08f),
                          c.full_kv_proj_out_dim(), c.hidden_size, lw.k_proj_fp8, lw.k_proj_wscale);
      quantize_fp8_matrix(rand_vec(static_cast<std::size_t>(c.full_kv_proj_out_dim() * c.hidden_size), 0.08f),
                          c.full_kv_proj_out_dim(), c.hidden_size, lw.v_proj_fp8, lw.v_proj_wscale);
      quantize_fp8_matrix(
          rand_vec(static_cast<std::size_t>(c.hidden_size * c.num_attention_heads * c.head_dim), 0.08f), c.hidden_size,
          c.num_attention_heads * c.head_dim, lw.o_proj_fp8, lw.o_proj_wscale);
      lw.q_norm = rand_vec(static_cast<std::size_t>(c.head_dim), 0.02f);
      lw.k_norm = rand_vec(static_cast<std::size_t>(c.head_dim), 0.02f);
    }

    quantize_nvfp4_matrix(rand_vec(static_cast<std::size_t>(c.intermediate_size * c.hidden_size), 0.08f),
                          c.intermediate_size, c.hidden_size, lw.gate_nvfp4, lw.gate_scales, lw.gate_scale2);
    quantize_nvfp4_matrix(rand_vec(static_cast<std::size_t>(c.intermediate_size * c.hidden_size), 0.08f),
                          c.intermediate_size, c.hidden_size, lw.up_nvfp4, lw.up_scales, lw.up_scale2);
    quantize_nvfp4_matrix(rand_vec(static_cast<std::size_t>(c.hidden_size * c.intermediate_size), 0.08f), c.hidden_size,
                          c.intermediate_size, lw.down_nvfp4, lw.down_scales, lw.down_scale2);
  }
  return m;
}

std::vector<float> Qwen38MiniModel::forward_token(Qwen38MiniSession& session, std::int32_t token_id) const {
  if (token_id < 0 || token_id >= cfg_.vocab_size) {
    throw std::out_of_range("token_id out of vocab range");
  }
  const int pos = session.position();
  const int H = cfg_.hidden_size;
  std::vector<float> hidden(embed_tokens_.begin() + static_cast<std::ptrdiff_t>(token_id * H),
                            embed_tokens_.begin() + static_cast<std::ptrdiff_t>((token_id + 1) * H));
  std::vector<float> normed(static_cast<std::size_t>(H));
  std::vector<float> residual(static_cast<std::size_t>(H));

  for (int l = 0; l < cfg_.num_hidden_layers; ++l) {
    const auto& lw = layers_[l];
    rmsnorm_zero_centered(hidden, lw.input_layernorm, cfg_.rms_norm_eps, normed);

    if (lw.kind == Qwen38LayerKind::LinearAttention) {
      const int conv_dim = cfg_.linear_conv_dim();
      const int k_dim = cfg_.linear_key_dim();
      const int v_dim = cfg_.linear_value_dim();
      std::vector<float> mixed_qkv(static_cast<std::size_t>(conv_dim));
      std::vector<float> conv_out(static_cast<std::size_t>(conv_dim));
      std::vector<float> z_vec(static_cast<std::size_t>(v_dim));
      std::vector<float> a_vec(static_cast<std::size_t>(cfg_.linear_num_value_heads));
      std::vector<float> b_vec(static_cast<std::size_t>(cfg_.linear_num_value_heads));

      fp8_weight_only_linear(normed, lw.in_proj_qkv_fp8, lw.in_proj_qkv_wscale, conv_dim, H, mixed_qkv);
      fp8_weight_only_linear(normed, lw.in_proj_z_fp8, lw.in_proj_z_wscale, v_dim, H, z_vec);
      matvec_f32(normed, lw.in_proj_a, cfg_.linear_num_value_heads, H, a_vec);
      matvec_f32(normed, lw.in_proj_b, cfg_.linear_num_value_heads, H, b_vec);
      causal_conv1d_silu_step(mixed_qkv, lw.conv1d_weight, conv_dim, cfg_.linear_conv_kernel_dim,
                              session.conv_states_[l], conv_out);

      std::vector<float> core_out(static_cast<std::size_t>(v_dim));
      gdn_recurrent_step(std::span<const float>(conv_out.data(), k_dim),
                         std::span<const float>(conv_out.data() + k_dim, k_dim),
                         std::span<const float>(conv_out.data() + 2 * k_dim, v_dim), z_vec, a_vec, b_vec, lw.A_log,
                         lw.dt_bias, lw.linear_norm_weight, cfg_.rms_norm_eps, cfg_.linear_num_key_heads,
                         cfg_.linear_num_value_heads, cfg_.linear_key_head_dim, cfg_.linear_value_head_dim,
                         session.recurrent_states_[l], core_out);
      fp8_weight_only_linear(core_out, lw.out_proj_fp8, lw.out_proj_wscale, H, v_dim, residual);
    } else {
      std::vector<float> q_gate(static_cast<std::size_t>(cfg_.full_q_proj_out_dim()));
      std::vector<float> k_raw(static_cast<std::size_t>(cfg_.full_kv_proj_out_dim()));
      std::vector<float> v_raw(static_cast<std::size_t>(cfg_.full_kv_proj_out_dim()));
      fp8_weight_only_linear(normed, lw.q_proj_fp8, lw.q_proj_wscale, cfg_.full_q_proj_out_dim(), H, q_gate);
      fp8_weight_only_linear(normed, lw.k_proj_fp8, lw.k_proj_wscale, cfg_.full_kv_proj_out_dim(), H, k_raw);
      fp8_weight_only_linear(normed, lw.v_proj_fp8, lw.v_proj_wscale, cfg_.full_kv_proj_out_dim(), H, v_raw);

      const int attn_out_dim = cfg_.num_attention_heads * cfg_.head_dim;
      std::vector<float> attn_out(static_cast<std::size_t>(attn_out_dim));
      full_attn_gated_step(q_gate, k_raw, v_raw, lw.q_norm, lw.k_norm, cfg_.rms_norm_eps, cfg_.rope_theta,
                           cfg_.num_attention_heads, cfg_.num_key_value_heads, cfg_.head_dim, cfg_.rotary_dim, pos,
                           session.k_caches_[l], session.v_caches_[l], attn_out);
      fp8_weight_only_linear(attn_out, lw.o_proj_fp8, lw.o_proj_wscale, H, attn_out_dim, residual);
    }

    for (int i = 0; i < H; ++i) hidden[i] += residual[i];

    rmsnorm_zero_centered(hidden, lw.post_attention_layernorm, cfg_.rms_norm_eps, normed);
    std::vector<float> gate_buf(static_cast<std::size_t>(cfg_.intermediate_size));
    std::vector<float> up_buf(static_cast<std::size_t>(cfg_.intermediate_size));
    nvfp4_weight_only_linear(normed, lw.gate_nvfp4, lw.gate_scales, lw.gate_scale2, cfg_.intermediate_size, H,
                             gate_buf);
    nvfp4_weight_only_linear(normed, lw.up_nvfp4, lw.up_scales, lw.up_scale2, cfg_.intermediate_size, H, up_buf);
    for (int i = 0; i < cfg_.intermediate_size; ++i) gate_buf[i] = silu_f32(gate_buf[i]) * up_buf[i];
    nvfp4_weight_only_linear(gate_buf, lw.down_nvfp4, lw.down_scales, lw.down_scale2, H, cfg_.intermediate_size,
                             residual);
    for (int i = 0; i < H; ++i) hidden[i] += residual[i];
  }

  session.committed_tokens_.push_back(token_id);
  rmsnorm_zero_centered(hidden, final_norm_, cfg_.rms_norm_eps, normed);
  std::vector<float> logits(static_cast<std::size_t>(cfg_.vocab_size));
  nvfp4_weight_only_linear(normed, lm_head_nvfp4_, lm_head_scales_, lm_head_scale2_, cfg_.vocab_size, H, logits);
  return logits;
}

}  // namespace ingine::backends::qwen38

#include "backends/qwen38/qwen38_block_cpu.h"

#include "backends/qwen38/qwen38_cpu_ops.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <random>
#include <stdexcept>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

namespace ingine::backends::qwen38 {
namespace {

struct ScopedPosixFd {
  int fd = -1;
  ScopedPosixFd() = default;
  explicit ScopedPosixFd(int f) noexcept : fd(f) {}
  ScopedPosixFd(const ScopedPosixFd&) = delete;
  ScopedPosixFd& operator=(const ScopedPosixFd&) = delete;
  ~ScopedPosixFd() {
    if (fd >= 0) ::close(fd);
  }
  [[nodiscard]] int get() const noexcept { return fd; }
};

void pread_exact(int fd, void* dst, std::size_t n, std::uint64_t offset, const std::filesystem::path& path) {
  auto* out = static_cast<std::uint8_t*>(dst);
  std::size_t done = 0;
  while (done < n) {
    const ssize_t got = ::pread(fd, out + done, n - done, static_cast<off_t>(offset + done));
    if (got < 0) {
      if (errno == EINTR) continue;
      throw std::invalid_argument("pread error on " + path.string() + ": " + std::strerror(errno));
    }
    if (got == 0) {
      throw std::invalid_argument("unexpected EOF in pread on " + path.string());
    }
    done += static_cast<std::size_t>(got);
  }
}

std::uint64_t parse_header_len_le(std::span<const std::uint8_t> eight_bytes) {
  std::uint64_t header_len = 0;
  for (int i = 0; i < 8; ++i) {
    header_len |= (static_cast<std::uint64_t>(eight_bytes[static_cast<std::size_t>(i)]) << (8 * i));
  }
  if (header_len == 0 || header_len > (64ull * 1024ull * 1024ull)) {
    throw std::invalid_argument("invalid safetensors header_len in shard");
  }
  return header_len;
}

template <typename Fn>
void parallel_rows(int rows, std::size_t work_estimate, int num_threads, Fn&& fn) {
  int threads = num_threads > 0 ? num_threads : static_cast<int>(std::min<unsigned>(8u, std::thread::hardware_concurrency()));
  if (threads < 1) threads = 1;
  if (threads == 1 || rows < 8 || work_estimate < 65536ull) {
    fn(0, rows);
    return;
  }
  threads = std::min(threads, rows);
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads - 1));
  for (int t = 0; t < threads - 1; ++t) {
    const int r0 = (rows * t) / threads;
    const int r1 = (rows * (t + 1)) / threads;
    pool.emplace_back([&, r0, r1]() { fn(r0, r1); });
  }
  const int r0 = (rows * (threads - 1)) / threads;
  fn(r0, rows);
  for (auto& th : pool) th.join();
}

void bf16_round_vec(std::vector<float>& v) noexcept {
  for (float& x : v) x = bf16_to_f32(f32_to_bf16(x));
}

std::vector<float> decode_bf16_vec(std::span<const std::uint16_t> src) {
  std::vector<float> out(src.size());
  for (std::size_t i = 0; i < src.size(); ++i) out[i] = bf16_to_f32(src[i]);
  return out;
}

float silu_f32(float x) noexcept { return x / (1.0f + std::exp(-x)); }

template <typename WarpFn>
float warp_reduce_32(WarpFn&& fn) {
  float lane_acc[32]{};
  for (int lane = 0; lane < 32; ++lane) lane_acc[lane] = fn(lane);
  for (int off = 16; off > 0; off >>= 1) {
    for (int lane = 0; lane < off; ++lane) lane_acc[lane] += lane_acc[lane + off];
  }
  return lane_acc[0];
}

void run_proj_host(const Qwen38ProjWeightsHost& w, std::span<const float> x_row, std::span<float> out_row,
                   bool bf16_boundaries, bool use_calibrated_fp8, int num_threads, bool fp32_accum_sim) {
  if (static_cast<int>(x_row.size()) != w.in_features || static_cast<int>(out_row.size()) != w.out_features) {
    throw std::invalid_argument("run_proj_host dimension mismatch");
  }
  const int N = w.out_features;
  const int K = w.in_features;

  std::vector<float> x_eff(x_row.begin(), x_row.end());
  if (w.quant == Qwen38LinearQuantKind::Fp8E4M3 && use_calibrated_fp8 && w.input_scale > 0.0f) {
    const float inv = 1.0f / w.input_scale;
    for (int k = 0; k < K; ++k) {
      x_eff[static_cast<std::size_t>(k)] =
          decode_e4m3fn(quantize_e4m3fn_fast(x_eff[static_cast<std::size_t>(k)] * inv)) * w.input_scale;
    }
  }

  const std::size_t work = static_cast<std::size_t>(N) * static_cast<std::size_t>(K);
  if (w.quant == Qwen38LinearQuantKind::Bf16) {
    parallel_rows(N, work, num_threads, [&](int r0, int r1) {
      for (int r = r0; r < r1; ++r) {
        const std::uint16_t* row = w.bf16_w.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(K);
        float val = 0.0f;
        if (fp32_accum_sim) {
          val = warp_reduce_32([&](int lane) {
            float acc = 0.0f;
            for (int k = lane; k < K; k += 32) {
              acc += x_eff[static_cast<std::size_t>(k)] * bf16_to_f32(row[k]);
            }
            return acc;
          });
        } else {
          double acc = 0.0;
          for (int k = 0; k < K; ++k) {
            acc += static_cast<double>(x_eff[static_cast<std::size_t>(k)]) * static_cast<double>(bf16_to_f32(row[k]));
          }
          val = static_cast<float>(acc);
        }
        out_row[static_cast<std::size_t>(r)] = bf16_boundaries ? bf16_to_f32(f32_to_bf16(val)) : val;
      }
    });
  } else if (w.quant == Qwen38LinearQuantKind::Fp8E4M3) {
    parallel_rows(N, work, num_threads, [&](int r0, int r1) {
      for (int r = r0; r < r1; ++r) {
        const std::uint8_t* row = w.fp8_w.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(K);
        float val = 0.0f;
        if (fp32_accum_sim) {
          const float acc_f = warp_reduce_32([&](int lane) {
            float acc = 0.0f;
            for (int k = lane; k < K; k += 32) {
              acc += x_eff[static_cast<std::size_t>(k)] * decode_e4m3fn(row[k]);
            }
            return acc;
          });
          val = acc_f * w.weight_scale;
        } else {
          double acc = 0.0;
          for (int k = 0; k < K; ++k) {
            acc += static_cast<double>(x_eff[static_cast<std::size_t>(k)]) * static_cast<double>(decode_e4m3fn(row[k]));
          }
          val = static_cast<float>(acc * static_cast<double>(w.weight_scale));
        }
        out_row[static_cast<std::size_t>(r)] = bf16_boundaries ? bf16_to_f32(f32_to_bf16(val)) : val;
      }
    });
  } else {
    if (K % 16 != 0) throw std::invalid_argument("NVFP4 linear requires in_features divisible by 16");
    const int half_k = K / 2;
    const int blocks_per_row = K / 16;
    parallel_rows(N, work, num_threads, [&](int r0, int r1) {
      for (int r = r0; r < r1; ++r) {
        const std::uint8_t* prow = w.nvfp4_packed.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(half_k);
        const std::uint8_t* srow =
            w.nvfp4_scales.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(blocks_per_row);
        float val = 0.0f;
        if (fp32_accum_sim) {
          val = warp_reduce_32([&](int lane) {
            float acc = 0.0f;
            for (int b = lane; b < blocks_per_row; b += 32) {
              const float scale = decode_e4m3fn(srow[b]) * w.weight_scale_2;
              float bsum = 0.0f;
              const int kbase = b * 16;
              for (int i = 0; i < 8; ++i) {
                const std::uint8_t byte = prow[b * 8 + i];
                bsum += x_eff[static_cast<std::size_t>(kbase + 2 * i)] * decode_e2m1(byte & 0x0fu);
                bsum += x_eff[static_cast<std::size_t>(kbase + 2 * i + 1)] * decode_e2m1((byte >> 4) & 0x0fu);
              }
              acc += bsum * scale;
            }
            return acc;
          });
        } else {
          double acc = 0.0;
          for (int b = 0; b < blocks_per_row; ++b) {
            const float scale = decode_e4m3fn(srow[b]) * w.weight_scale_2;
            double bsum = 0.0;
            const int kbase = b * 16;
            for (int i = 0; i < 8; ++i) {
              const std::uint8_t byte = prow[b * 8 + i];
              bsum += static_cast<double>(x_eff[static_cast<std::size_t>(kbase + 2 * i)]) *
                      static_cast<double>(decode_e2m1(byte & 0x0fu));
              bsum += static_cast<double>(x_eff[static_cast<std::size_t>(kbase + 2 * i + 1)]) *
                      static_cast<double>(decode_e2m1((byte >> 4) & 0x0fu));
            }
            acc += bsum * static_cast<double>(scale);
          }
          val = static_cast<float>(acc);
        }
        out_row[static_cast<std::size_t>(r)] = bf16_boundaries ? bf16_to_f32(f32_to_bf16(val)) : val;
      }
    });
  }
}

void quantize_fp8_host_proj(std::span<const float> w_f32, int out_f, int in_f, float input_scale,
                            Qwen38ProjWeightsHost& dst) {
  dst.out_features = out_f;
  dst.in_features = in_f;
  dst.quant = Qwen38LinearQuantKind::Fp8E4M3;
  dst.input_scale = input_scale;
  float max_abs = 0.0f;
  for (float v : w_f32) max_abs = std::max(max_abs, std::fabs(v));
  dst.weight_scale = max_abs > 0.0f ? (max_abs / 240.0f) : 1.0f;
  dst.fp8_w.resize(static_cast<std::size_t>(out_f) * static_cast<std::size_t>(in_f));
  for (std::size_t i = 0; i < dst.fp8_w.size(); ++i) {
    dst.fp8_w[i] = quantize_e4m3fn_fast(w_f32[i] / dst.weight_scale);
  }
}

void quantize_bf16_host_proj(std::span<const float> w_f32, int out_f, int in_f, Qwen38ProjWeightsHost& dst) {
  dst.out_features = out_f;
  dst.in_features = in_f;
  dst.quant = Qwen38LinearQuantKind::Bf16;
  dst.bf16_w.resize(static_cast<std::size_t>(out_f) * static_cast<std::size_t>(in_f));
  for (std::size_t i = 0; i < dst.bf16_w.size(); ++i) {
    dst.bf16_w[i] = f32_to_bf16(w_f32[i]);
  }
}

void quantize_nvfp4_host_proj(std::span<const float> w_f32, int out_f, int in_f, float input_scale,
                              Qwen38ProjWeightsHost& dst) {
  static constexpr float kE2M1Pos[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  if (in_f % 16 != 0) throw std::invalid_argument("NVFP4 requires in_f divisible by 16");
  dst.out_features = out_f;
  dst.in_features = in_f;
  dst.quant = Qwen38LinearQuantKind::Nvfp4;
  dst.input_scale = input_scale;

  float max_abs = 0.0f;
  for (float v : w_f32) max_abs = std::max(max_abs, std::fabs(v));
  dst.weight_scale_2 = max_abs > 0.0f ? (max_abs / (6.0f * 32.0f)) : 1.0f;

  dst.nvfp4_packed.assign(static_cast<std::size_t>(out_f) * static_cast<std::size_t>(in_f / 2), 0u);
  dst.nvfp4_scales.assign(static_cast<std::size_t>(out_f) * static_cast<std::size_t>(in_f / 16), 0u);

  const int blocks_per_row = in_f / 16;
  for (int r = 0; r < out_f; ++r) {
    const float* row = w_f32.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_f);
    for (int b = 0; b < blocks_per_row; ++b) {
      float blk_max = 0.0f;
      for (int i = 0; i < 16; ++i) blk_max = std::max(blk_max, std::fabs(row[b * 16 + i]));
      const float raw_bs = blk_max / (6.0f * dst.weight_scale_2);
      const std::uint8_t qbs = quantize_e4m3fn_fast(raw_bs);
      dst.nvfp4_scales[static_cast<std::size_t>(r) * static_cast<std::size_t>(blocks_per_row) + b] = qbs;
      const float eff_scale = std::max(1e-12f, decode_e4m3fn(qbs) * dst.weight_scale_2);
      for (int i = 0; i < 16; i += 2) {
        auto enc = [&](float v) -> std::uint8_t {
          const std::uint8_t s = v < 0.0f ? 8u : 0u;
          const float av = std::fabs(v) / eff_scale;
          std::uint8_t best = 0;
          float best_e = std::fabs(av - kE2M1Pos[0]);
          for (std::uint8_t c = 1; c < 8; ++c) {
            const float err = std::fabs(av - kE2M1Pos[c]);
            if (err < best_e) {
              best_e = err;
              best = c;
            }
          }
          return static_cast<std::uint8_t>(s | best);
        };
        const std::uint8_t lo = enc(row[b * 16 + i]);
        const std::uint8_t hi = enc(row[b * 16 + i + 1]);
        dst.nvfp4_packed[static_cast<std::size_t>(r) * static_cast<std::size_t>(in_f / 2) + (b * 8 + i / 2)] =
            static_cast<std::uint8_t>((hi << 4) | lo);
      }
    }
  }
}

struct SliceTarget {
  std::uint64_t file_begin = 0;
  std::uint64_t file_end = 0;
  std::vector<std::uint8_t>* dst = nullptr;
  std::size_t dst_offset = 0;
};

struct TensorReadSpec {
  std::string name;
  const Qwen38TensorInfo* info = nullptr;
  std::vector<std::uint8_t>* dst = nullptr;
};

std::vector<float> decode_bf16_or_f32_bytes(std::span<const std::uint8_t> raw, std::string_view dtype,
                                            std::size_t expected_elems, std::string_view name) {
  std::vector<float> out(expected_elems);
  if (dtype == "BF16") {
    if (raw.size() != expected_elems * 2ull) {
      throw std::invalid_argument("unexpected BF16 byte size for '" + std::string(name) + "'");
    }
    for (std::size_t i = 0; i < expected_elems; ++i) {
      const std::uint16_t bits = static_cast<std::uint16_t>(raw[2 * i]) |
                                 (static_cast<std::uint16_t>(raw[2 * i + 1]) << 8);
      out[i] = bf16_to_f32(bits);
    }
  } else if (dtype == "F32") {
    if (raw.size() != expected_elems * 4ull) {
      throw std::invalid_argument("unexpected F32 byte size for '" + std::string(name) + "'");
    }
    std::memcpy(out.data(), raw.data(), raw.size());
  } else {
    throw std::invalid_argument("expected BF16/F32 tensor for '" + std::string(name) + "', got '" +
                                std::string(dtype) + "'");
  }
  return out;
}

std::vector<std::uint16_t> copy_bf16_bytes(std::span<const std::uint8_t> raw, std::string_view dtype,
                                           std::size_t expected_elems, std::string_view name) {
  if (dtype != "BF16" || raw.size() != expected_elems * 2ull) {
    throw std::invalid_argument("expected BF16 tensor of " + std::to_string(expected_elems) + " elems for '" +
                                std::string(name) + "'");
  }
  std::vector<std::uint16_t> out(expected_elems);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

float decode_scalar_f32(std::span<const std::uint8_t> raw, std::string_view dtype, std::string_view name) {
  if (dtype != "F32" || raw.size() != 4) {
    throw std::invalid_argument("expected scalar F32 tensor for '" + std::string(name) + "'");
  }
  float v = 0.0f;
  std::memcpy(&v, raw.data(), 4);
  return v;
}

}  // namespace

std::uint8_t quantize_e4m3fn_fast(float x) noexcept {
  std::uint32_t u = 0;
  std::memcpy(&u, &x, sizeof(u));
  const std::uint8_t sign = static_cast<std::uint8_t>((u >> 24) & 0x80u);
  const std::uint32_t abs_u = u & 0x7fffffffu;
  if (abs_u > 0x7f800000u) return 0x7fu;  // NaN matches quantize_e4m3fn()
  if (abs_u >= 0x43e00000u) return static_cast<std::uint8_t>(sign | 0x7eu);
  if (abs_u <= 0x3a800000u) return sign;
  if (abs_u < 0x3c800000u) {
    const std::uint32_t fexp = abs_u >> 23;
    const std::uint32_t sig = 0x800000u | (abs_u & 0x7fffffu);
    const std::uint32_t shift = 141u - fexp;
    std::uint32_t q = sig >> shift;
    const std::uint32_t rem = sig & ((1u << shift) - 1u);
    const std::uint32_t half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (q & 1u) != 0u)) ++q;
    return static_cast<std::uint8_t>(sign | q);
  }
  const std::uint32_t e4 = (abs_u >> 23) - 120u;
  const std::uint32_t fmant = abs_u & 0x7fffffu;
  std::uint32_t code = (e4 << 3) | (fmant >> 20);
  const std::uint32_t low20 = fmant & 0xfffffu;
  if (low20 > 0x80000u || (low20 == 0x80000u && (code & 1u) != 0u)) ++code;
  if (code > 0x7eu) code = 0x7eu;
  return static_cast<std::uint8_t>(sign | code);
}

Qwen38BlockBundleHost make_synthetic_qwen38_block_bundle(Qwen38WeightPrecision prec,
                                                         std::span<const int> layer_indices,
                                                         std::span<const std::int32_t> token_ids,
                                                         std::uint32_t seed) {
  if (layer_indices.empty() || token_ids.empty()) {
    throw std::invalid_argument("make_synthetic_qwen38_block_bundle requires non-empty layers and tokens");
  }
  Qwen38BlockBundleHost b;
  Qwen38Config& c = b.cfg;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForCausalLM"};
  c.vocab_size = 64;
  c.hidden_size = 64;
  c.intermediate_size = 128;
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
  c.weight_precision = prec;
  if (prec == Qwen38WeightPrecision::MixedFp8Nvfp4) {
    c.quant_method = "modelopt";
    c.quant_algo = "MIXED_PRECISION";
    c.nvfp4_group_size = 16;
  }
  c.validate();

  b.token_ids.assign(token_ids.begin(), token_ids.end());

  std::mt19937 rng(seed);
  auto rand_vec = [&](std::size_t n, float stddev) {
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> v(n);
    for (float& x : v) x = dist(rng);
    return v;
  };
  auto rand_bf16 = [&](std::size_t n, float stddev) {
    const auto f = rand_vec(n, stddev);
    std::vector<std::uint16_t> out(n);
    for (std::size_t i = 0; i < n; ++i) out[i] = f32_to_bf16(f[i]);
    return out;
  };

  const int H = c.hidden_size;
  const int I = c.intermediate_size;
  b.prompt_embeddings_bf16.resize(token_ids.size() * static_cast<std::size_t>(H));
  const auto full_embed = rand_bf16(static_cast<std::size_t>(c.vocab_size * H), 0.1f);
  for (std::size_t t = 0; t < token_ids.size(); ++t) {
    const int tid = token_ids[t];
    if (tid < 0 || tid >= c.vocab_size) throw std::out_of_range("synthetic token_id out of vocab range");
    std::copy_n(full_embed.data() + static_cast<std::size_t>(tid * H), H,
                b.prompt_embeddings_bf16.data() + t * static_cast<std::size_t>(H));
  }

  for (int layer_idx : layer_indices) {
    if (layer_idx < 0 || layer_idx >= c.num_hidden_layers) {
      throw std::out_of_range("layer index out of range in synthetic bundle");
    }
    Qwen38BlockLayerHost lw;
    lw.layer_index = layer_idx;
    lw.kind = c.layer_types[static_cast<std::size_t>(layer_idx)];
    lw.input_layernorm = rand_bf16(static_cast<std::size_t>(H), 0.02f);
    lw.post_attention_layernorm = rand_bf16(static_cast<std::size_t>(H), 0.02f);

    if (lw.kind == Qwen38LayerKind::LinearAttention) {
      const int cd = c.linear_conv_dim();
      const int vd = c.linear_value_dim();
      const int nhv = c.linear_num_value_heads;
      if (prec == Qwen38WeightPrecision::MixedFp8Nvfp4) {
        quantize_fp8_host_proj(rand_vec(static_cast<std::size_t>(cd * H), 0.08f), cd, H, 0.25f, lw.in_proj_qkv);
        quantize_fp8_host_proj(rand_vec(static_cast<std::size_t>(vd * H), 0.08f), vd, H, 0.25f, lw.in_proj_z);
        quantize_fp8_host_proj(rand_vec(static_cast<std::size_t>(H * vd), 0.08f), H, vd, 0.25f, lw.out_proj);
      } else {
        quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(cd * H), 0.08f), cd, H, lw.in_proj_qkv);
        quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(vd * H), 0.08f), vd, H, lw.in_proj_z);
        quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(H * vd), 0.08f), H, vd, lw.out_proj);
      }
      quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(nhv * H), 0.05f), nhv, H, lw.in_proj_a);
      quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(nhv * H), 0.05f), nhv, H, lw.in_proj_b);
      lw.A_log = rand_vec(static_cast<std::size_t>(nhv), 0.1f);
      lw.dt_bias = rand_vec(static_cast<std::size_t>(nhv), 0.05f);
      lw.conv1d_weight = rand_vec(static_cast<std::size_t>(cd * c.linear_conv_kernel_dim), 0.1f);
      lw.linear_norm_weight = std::vector<float>(static_cast<std::size_t>(c.linear_value_head_dim), 1.0f);
    } else {
      const int q_out = c.full_q_proj_out_dim();
      const int kv_out = c.full_kv_proj_out_dim();
      const int attn_out = c.num_attention_heads * c.head_dim;
      if (prec == Qwen38WeightPrecision::MixedFp8Nvfp4) {
        quantize_fp8_host_proj(rand_vec(static_cast<std::size_t>(q_out * H), 0.08f), q_out, H, 0.25f, lw.q_proj);
        quantize_fp8_host_proj(rand_vec(static_cast<std::size_t>(kv_out * H), 0.08f), kv_out, H, 0.25f, lw.k_proj);
        quantize_fp8_host_proj(rand_vec(static_cast<std::size_t>(kv_out * H), 0.08f), kv_out, H, 0.25f, lw.v_proj);
        quantize_fp8_host_proj(rand_vec(static_cast<std::size_t>(H * attn_out), 0.08f), H, attn_out, 0.25f, lw.o_proj);
      } else {
        quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(q_out * H), 0.08f), q_out, H, lw.q_proj);
        quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(kv_out * H), 0.08f), kv_out, H, lw.k_proj);
        quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(kv_out * H), 0.08f), kv_out, H, lw.v_proj);
        quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(H * attn_out), 0.08f), H, attn_out, lw.o_proj);
      }
      lw.q_norm = rand_bf16(static_cast<std::size_t>(c.head_dim), 0.02f);
      lw.k_norm = rand_bf16(static_cast<std::size_t>(c.head_dim), 0.02f);
    }

    if (prec == Qwen38WeightPrecision::MixedFp8Nvfp4) {
      quantize_nvfp4_host_proj(rand_vec(static_cast<std::size_t>(I * H), 0.08f), I, H, 0.25f, lw.gate_proj);
      quantize_nvfp4_host_proj(rand_vec(static_cast<std::size_t>(I * H), 0.08f), I, H, 0.25f, lw.up_proj);
      quantize_nvfp4_host_proj(rand_vec(static_cast<std::size_t>(H * I), 0.08f), H, I, 0.25f, lw.down_proj);
    } else {
      quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(I * H), 0.08f), I, H, lw.gate_proj);
      quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(I * H), 0.08f), I, H, lw.up_proj);
      quantize_bf16_host_proj(rand_vec(static_cast<std::size_t>(H * I), 0.08f), H, I, lw.down_proj);
    }
    b.layers.push_back(std::move(lw));
  }
  return b;
}

Qwen38BlockBundleHost load_qwen38_block_bundle(const std::filesystem::path& model_dir,
                                               const Qwen38Manifest& manifest,
                                               std::span<const int> layer_indices,
                                               std::span<const std::int32_t> token_ids) {
  if (layer_indices.empty() || token_ids.empty()) {
    throw std::invalid_argument("load_qwen38_block_bundle requires non-empty layers and tokens");
  }
  const Qwen38Config& c = manifest.config;
  c.validate();
  for (std::int32_t tid : token_ids) {
    if (tid < 0 || tid >= c.vocab_size) {
      throw std::out_of_range("load_qwen38_block_bundle token_id out of vocab range");
    }
  }

  std::vector<std::string> needed_names;
  auto need_tensor = [&](const std::string& name) {
    if (manifest.active_tensors.find(name) == manifest.active_tensors.end()) {
      throw std::invalid_argument("manifest missing tensor '" + name + "'");
    }
    needed_names.push_back(name);
  };
  auto need_proj = [&](const std::string& prefix, Qwen38LinearQuantKind q) {
    need_tensor(prefix + ".weight");
    if (q == Qwen38LinearQuantKind::Fp8E4M3) {
      need_tensor(prefix + ".weight_scale");
      need_tensor(prefix + ".input_scale");
    } else if (q == Qwen38LinearQuantKind::Nvfp4) {
      need_tensor(prefix + ".weight_scale");
      need_tensor(prefix + ".weight_scale_2");
      need_tensor(prefix + ".input_scale");
    }
  };

  for (int layer_idx : layer_indices) {
    if (layer_idx < 0 || layer_idx >= c.num_hidden_layers) {
      throw std::out_of_range("layer_index out of bounds");
    }
    const std::string lp = "model.language_model.layers." + std::to_string(layer_idx);
    need_tensor(lp + ".input_layernorm.weight");
    need_tensor(lp + ".post_attention_layernorm.weight");
    if (c.layer_types[static_cast<std::size_t>(layer_idx)] == Qwen38LayerKind::LinearAttention) {
      const std::string attn = lp + ".linear_attn";
      need_tensor(attn + ".A_log");
      need_tensor(attn + ".dt_bias");
      need_tensor(attn + ".conv1d.weight");
      need_tensor(attn + ".norm.weight");
      need_tensor(attn + ".in_proj_a.weight");
      need_tensor(attn + ".in_proj_b.weight");
      need_proj(attn + ".in_proj_qkv", c.linear_attn_quant());
      need_proj(attn + ".in_proj_z", c.linear_attn_quant());
      need_proj(attn + ".out_proj", c.linear_attn_quant());
    } else {
      const std::string attn = lp + ".self_attn";
      need_tensor(attn + ".q_norm.weight");
      need_tensor(attn + ".k_norm.weight");
      need_proj(attn + ".q_proj", c.full_attn_quant());
      need_proj(attn + ".k_proj", c.full_attn_quant());
      need_proj(attn + ".v_proj", c.full_attn_quant());
      need_proj(attn + ".o_proj", c.full_attn_quant());
    }
    need_proj(lp + ".mlp.gate_proj", c.mlp_quant());
    need_proj(lp + ".mlp.up_proj", c.mlp_quant());
    need_proj(lp + ".mlp.down_proj", c.mlp_quant());
  }

  // Group requests by shard file (extract only the requested embedding rows + requested layer tensors).
  const auto embed_it = manifest.active_tensors.find("model.language_model.embed_tokens.weight");
  if (embed_it == manifest.active_tensors.end()) {
    throw std::invalid_argument("manifest missing embed_tokens.weight");
  }
  const Qwen38TensorInfo& embed_info = embed_it->second;
  if (embed_info.dtype != "BF16") {
    throw std::invalid_argument("embed_tokens.weight must be BF16");
  }
  const std::uint64_t row_bytes = static_cast<std::uint64_t>(c.hidden_size) * 2ull;

  std::unordered_map<std::string, std::vector<std::uint8_t>> raw_tensors;
  for (const std::string& name : needed_names) {
    raw_tensors.emplace(name, std::vector<std::uint8_t>(manifest.active_tensors.at(name).byte_size()));
  }
  std::vector<std::uint8_t> embed_rows_raw(token_ids.size() * static_cast<std::size_t>(row_bytes));

  std::unordered_map<std::string, std::vector<TensorReadSpec>> by_shard;
  for (const std::string& name : needed_names) {
    const Qwen38TensorInfo& info = manifest.active_tensors.at(name);
    by_shard[info.shard_file].push_back(TensorReadSpec{name, &info, &raw_tensors.at(name)});
  }
  // Ensure embed_tokens shard is visited even if no layer happens to sit on it.
  (void)by_shard[embed_info.shard_file];

  const bool is_pinned = manifest.checkpoint.verification_level == ingine::model::VerificationLevel::PinnedVerified &&
                         manifest.checkpoint.shard_digests_verified;

  for (auto& [shard, specs] : by_shard) {
    const bool has_embed = (shard == embed_info.shard_file);
    if (is_pinned) {
      std::array<std::uint8_t, 8> hdr_bytes{};
      std::uint8_t hdr_have = 0;
      bool targets_built = false;
      std::vector<SliceTarget> targets;
      std::vector<std::vector<std::uint8_t>> staged_tensors(specs.size());
      std::vector<std::uint8_t> staged_embed(has_embed ? embed_rows_raw.size() : 0);

      (void)manifest.stream_verified_file(
          model_dir, shard,
          [&](std::uint64_t off, std::span<const std::uint8_t> chunk) {
            if (off < 8) {
              const std::uint64_t take = std::min<std::uint64_t>(8 - off, chunk.size());
              std::memcpy(hdr_bytes.data() + off, chunk.data(), static_cast<std::size_t>(take));
              hdr_have = static_cast<std::uint8_t>(std::max<std::uint64_t>(hdr_have, off + take));
            }
            if (!targets_built && hdr_have == 8) {
              const std::uint64_t data_base = 8ull + parse_header_len_le(hdr_bytes);
              targets.reserve(specs.size() + (has_embed ? token_ids.size() : 0));
              for (std::size_t i = 0; i < specs.size(); ++i) {
                staged_tensors[i].assign(specs[i].info->byte_size(), 0u);
                targets.push_back(SliceTarget{data_base + specs[i].info->data_offset_begin,
                                              data_base + specs[i].info->data_offset_end, &staged_tensors[i], 0});
              }
              if (has_embed) {
                for (std::size_t t = 0; t < token_ids.size(); ++t) {
                  const std::uint64_t rb = data_base + embed_info.data_offset_begin +
                                           static_cast<std::uint64_t>(token_ids[t]) * row_bytes;
                  targets.push_back(SliceTarget{rb, rb + row_bytes, &staged_embed, t * static_cast<std::size_t>(row_bytes)});
                }
              }
              std::sort(targets.begin(), targets.end(),
                        [](const SliceTarget& a, const SliceTarget& b) { return a.file_begin < b.file_begin; });
              targets_built = true;
            }
            if (!targets_built) return;
            const std::uint64_t c0 = off;
            const std::uint64_t c1 = off + chunk.size();
            for (const SliceTarget& tg : targets) {
              if (tg.file_begin >= c1) break;
              if (tg.file_end <= c0) continue;
              const std::uint64_t s0 = std::max(c0, tg.file_begin);
              const std::uint64_t s1 = std::min(c1, tg.file_end);
              std::memcpy(tg.dst->data() + tg.dst_offset + static_cast<std::size_t>(s0 - tg.file_begin),
                          chunk.data() + static_cast<std::size_t>(s0 - c0), static_cast<std::size_t>(s1 - s0));
            }
          },
          [&](const Qwen38StreamResult&) {
            for (std::size_t i = 0; i < specs.size(); ++i) {
              *specs[i].dst = std::move(staged_tensors[i]);
            }
            if (has_embed) embed_rows_raw = std::move(staged_embed);
          },
          [&]() {
            for (auto& s : staged_tensors) s.clear();
            staged_embed.clear();
          });
    } else {
      const std::filesystem::path sp = model_dir / shard;
      int raw_fd = ::open(sp.c_str(), O_RDONLY);
      if (raw_fd < 0) {
        throw std::invalid_argument("cannot open shard " + sp.string() + ": " + std::strerror(errno));
      }
      ScopedPosixFd fd(raw_fd);
      const auto vf_it = manifest.verified_files.find(shard);
      if (vf_it != manifest.verified_files.end()) {
        struct stat st {};
        if (::fstat(fd.get(), &st) != 0 || !S_ISREG(st.st_mode) ||
            static_cast<std::uint64_t>(st.st_dev) != vf_it->second.dev ||
            static_cast<std::uint64_t>(st.st_ino) != vf_it->second.ino ||
            static_cast<std::uint64_t>(st.st_size) != vf_it->second.byte_size) {
          throw std::invalid_argument("shard identity mismatch on open FD for '" + shard + "'");
        }
      }
      std::array<std::uint8_t, 8> hdr_bytes{};
      pread_exact(fd.get(), hdr_bytes.data(), 8, 0, sp);
      const std::uint64_t data_base = 8ull + parse_header_len_le(hdr_bytes);
      for (const TensorReadSpec& spc : specs) {
        pread_exact(fd.get(), spc.dst->data(), spc.dst->size(), data_base + spc.info->data_offset_begin, sp);
      }
      if (has_embed) {
        for (std::size_t t = 0; t < token_ids.size(); ++t) {
          const std::uint64_t rb = data_base + embed_info.data_offset_begin +
                                   static_cast<std::uint64_t>(token_ids[t]) * row_bytes;
          pread_exact(fd.get(), embed_rows_raw.data() + t * static_cast<std::size_t>(row_bytes),
                      static_cast<std::size_t>(row_bytes), rb, sp);
        }
      }
    }
  }

  Qwen38BlockBundleHost bundle;
  bundle.cfg = c;
  bundle.token_ids.assign(token_ids.begin(), token_ids.end());
  bundle.prompt_embeddings_bf16 =
      copy_bf16_bytes(embed_rows_raw, "BF16", token_ids.size() * static_cast<std::size_t>(c.hidden_size),
                      "embed_tokens_slice");

  auto build_proj = [&](const std::string& prefix, int out_f, int in_f, Qwen38LinearQuantKind q) {
    Qwen38ProjWeightsHost w;
    w.out_features = out_f;
    w.in_features = in_f;
    w.quant = q;
    const auto& wt_info = manifest.active_tensors.at(prefix + ".weight");
    const auto& wt_raw = raw_tensors.at(prefix + ".weight");
    if (q == Qwen38LinearQuantKind::Bf16) {
      w.bf16_w = copy_bf16_bytes(wt_raw, wt_info.dtype, static_cast<std::size_t>(out_f * in_f), prefix + ".weight");
    } else if (q == Qwen38LinearQuantKind::Fp8E4M3) {
      w.fp8_w = wt_raw;
      w.weight_scale = decode_scalar_f32(raw_tensors.at(prefix + ".weight_scale"),
                                         manifest.active_tensors.at(prefix + ".weight_scale").dtype,
                                         prefix + ".weight_scale");
      w.input_scale = decode_scalar_f32(raw_tensors.at(prefix + ".input_scale"),
                                        manifest.active_tensors.at(prefix + ".input_scale").dtype,
                                        prefix + ".input_scale");
    } else {
      w.nvfp4_packed = wt_raw;
      w.nvfp4_scales = raw_tensors.at(prefix + ".weight_scale");
      w.weight_scale_2 = decode_scalar_f32(raw_tensors.at(prefix + ".weight_scale_2"),
                                           manifest.active_tensors.at(prefix + ".weight_scale_2").dtype,
                                           prefix + ".weight_scale_2");
      w.input_scale = decode_scalar_f32(raw_tensors.at(prefix + ".input_scale"),
                                        manifest.active_tensors.at(prefix + ".input_scale").dtype,
                                        prefix + ".input_scale");
    }
    return w;
  };

  for (int layer_idx : layer_indices) {
    Qwen38BlockLayerHost lw;
    lw.layer_index = layer_idx;
    lw.kind = c.layer_types[static_cast<std::size_t>(layer_idx)];
    const std::string lp = "model.language_model.layers." + std::to_string(layer_idx);
    lw.input_layernorm = copy_bf16_bytes(raw_tensors.at(lp + ".input_layernorm.weight"),
                                         manifest.active_tensors.at(lp + ".input_layernorm.weight").dtype,
                                         static_cast<std::size_t>(c.hidden_size), lp + ".input_layernorm.weight");
    lw.post_attention_layernorm =
        copy_bf16_bytes(raw_tensors.at(lp + ".post_attention_layernorm.weight"),
                        manifest.active_tensors.at(lp + ".post_attention_layernorm.weight").dtype,
                        static_cast<std::size_t>(c.hidden_size), lp + ".post_attention_layernorm.weight");
    if (lw.kind == Qwen38LayerKind::LinearAttention) {
      const std::string attn = lp + ".linear_attn";
      lw.A_log = decode_bf16_or_f32_bytes(raw_tensors.at(attn + ".A_log"),
                                          manifest.active_tensors.at(attn + ".A_log").dtype,
                                          static_cast<std::size_t>(c.linear_num_value_heads), attn + ".A_log");
      lw.dt_bias = decode_bf16_or_f32_bytes(raw_tensors.at(attn + ".dt_bias"),
                                            manifest.active_tensors.at(attn + ".dt_bias").dtype,
                                            static_cast<std::size_t>(c.linear_num_value_heads), attn + ".dt_bias");
      lw.conv1d_weight = decode_bf16_or_f32_bytes(
          raw_tensors.at(attn + ".conv1d.weight"), manifest.active_tensors.at(attn + ".conv1d.weight").dtype,
          static_cast<std::size_t>(c.linear_conv_dim()) * static_cast<std::size_t>(c.linear_conv_kernel_dim),
          attn + ".conv1d.weight");
      lw.linear_norm_weight = decode_bf16_or_f32_bytes(
          raw_tensors.at(attn + ".norm.weight"), manifest.active_tensors.at(attn + ".norm.weight").dtype,
          static_cast<std::size_t>(c.linear_value_head_dim), attn + ".norm.weight");
      lw.in_proj_a = build_proj(attn + ".in_proj_a", c.linear_num_value_heads, c.hidden_size, Qwen38LinearQuantKind::Bf16);
      lw.in_proj_b = build_proj(attn + ".in_proj_b", c.linear_num_value_heads, c.hidden_size, Qwen38LinearQuantKind::Bf16);
      lw.in_proj_qkv = build_proj(attn + ".in_proj_qkv", c.linear_conv_dim(), c.hidden_size, c.linear_attn_quant());
      lw.in_proj_z = build_proj(attn + ".in_proj_z", c.linear_value_dim(), c.hidden_size, c.linear_attn_quant());
      lw.out_proj = build_proj(attn + ".out_proj", c.hidden_size, c.linear_value_dim(), c.linear_attn_quant());
    } else {
      const std::string attn = lp + ".self_attn";
      lw.q_norm = copy_bf16_bytes(raw_tensors.at(attn + ".q_norm.weight"),
                                  manifest.active_tensors.at(attn + ".q_norm.weight").dtype,
                                  static_cast<std::size_t>(c.head_dim), attn + ".q_norm.weight");
      lw.k_norm = copy_bf16_bytes(raw_tensors.at(attn + ".k_norm.weight"),
                                  manifest.active_tensors.at(attn + ".k_norm.weight").dtype,
                                  static_cast<std::size_t>(c.head_dim), attn + ".k_norm.weight");
      lw.q_proj = build_proj(attn + ".q_proj", c.full_q_proj_out_dim(), c.hidden_size, c.full_attn_quant());
      lw.k_proj = build_proj(attn + ".k_proj", c.full_kv_proj_out_dim(), c.hidden_size, c.full_attn_quant());
      lw.v_proj = build_proj(attn + ".v_proj", c.full_kv_proj_out_dim(), c.hidden_size, c.full_attn_quant());
      lw.o_proj = build_proj(attn + ".o_proj", c.hidden_size, c.num_attention_heads * c.head_dim, c.full_attn_quant());
    }
    lw.gate_proj = build_proj(lp + ".mlp.gate_proj", c.intermediate_size, c.hidden_size, c.mlp_quant());
    lw.up_proj = build_proj(lp + ".mlp.up_proj", c.intermediate_size, c.hidden_size, c.mlp_quant());
    lw.down_proj = build_proj(lp + ".mlp.down_proj", c.hidden_size, c.intermediate_size, c.mlp_quant());
    bundle.layers.push_back(std::move(lw));
  }
  return bundle;
}

Qwen38BlockRunOutput run_qwen38_blocks_cpu(const Qwen38BlockBundleHost& bundle, bool bf16_boundaries,
                                           bool use_chunk_prefill, int chunk_size, bool use_calibrated_fp8_input_scale,
                                           int num_threads, bool fp32_accum_sim) {
  const auto t0 = std::chrono::steady_clock::now();
  const Qwen38Config& c = bundle.cfg;
  c.validate();
  const int S = static_cast<int>(bundle.token_ids.size());
  const int H = c.hidden_size;
  const int I = c.intermediate_size;
  if (S == 0 || bundle.layers.empty() ||
      bundle.prompt_embeddings_bf16.size() != static_cast<std::size_t>(S * H)) {
    throw std::invalid_argument("run_qwen38_blocks_cpu invalid bundle dimensions");
  }

  Qwen38BlockRunOutput out;
  out.hidden_seq = decode_bf16_vec(bundle.prompt_embeddings_bf16);
  out.conv_states.resize(bundle.layers.size());
  out.recurrent_states.resize(bundle.layers.size());
  out.k_caches.resize(bundle.layers.size());
  out.v_caches.resize(bundle.layers.size());

  std::vector<float> normed(static_cast<std::size_t>(S * H));
  std::vector<float> residual(static_cast<std::size_t>(S * H));

  for (std::size_t li = 0; li < bundle.layers.size(); ++li) {
    const auto& lw = bundle.layers[li];
    const auto in_norm_w = decode_bf16_vec(lw.input_layernorm);
    const auto post_norm_w = decode_bf16_vec(lw.post_attention_layernorm);

    for (int t = 0; t < S; ++t) {
      rmsnorm_zero_centered(std::span<const float>(out.hidden_seq.data() + static_cast<std::size_t>(t * H), H),
                            in_norm_w, c.rms_norm_eps,
                            std::span<float>(normed.data() + static_cast<std::size_t>(t * H), H));
    }
    if (bf16_boundaries) bf16_round_vec(normed);

    if (lw.kind == Qwen38LayerKind::LinearAttention) {
      const int cd = c.linear_conv_dim();
      const int kd = c.linear_key_dim();
      const int vd = c.linear_value_dim();
      const int nhv = c.linear_num_value_heads;
      std::vector<float> qkv_seq(static_cast<std::size_t>(S * cd));
      std::vector<float> z_seq(static_cast<std::size_t>(S * vd));
      std::vector<float> a_seq(static_cast<std::size_t>(S * nhv));
      std::vector<float> b_seq(static_cast<std::size_t>(S * nhv));

      for (int t = 0; t < S; ++t) {
        const auto x_t = std::span<const float>(normed.data() + static_cast<std::size_t>(t * H), H);
        run_proj_host(lw.in_proj_qkv, x_t, std::span<float>(qkv_seq.data() + static_cast<std::size_t>(t * cd), cd),
                      bf16_boundaries, use_calibrated_fp8_input_scale, num_threads, fp32_accum_sim);
        run_proj_host(lw.in_proj_z, x_t, std::span<float>(z_seq.data() + static_cast<std::size_t>(t * vd), vd),
                      bf16_boundaries, use_calibrated_fp8_input_scale, num_threads, fp32_accum_sim);
        run_proj_host(lw.in_proj_a, x_t, std::span<float>(a_seq.data() + static_cast<std::size_t>(t * nhv), nhv),
                      bf16_boundaries, false, num_threads, fp32_accum_sim);
        run_proj_host(lw.in_proj_b, x_t, std::span<float>(b_seq.data() + static_cast<std::size_t>(t * nhv), nhv),
                      bf16_boundaries, false, num_threads, fp32_accum_sim);
      }

      out.conv_states[li].assign(static_cast<std::size_t>(cd * c.linear_conv_kernel_dim), 0.0f);
      out.recurrent_states[li].assign(static_cast<std::size_t>(nhv) * static_cast<std::size_t>(c.linear_key_head_dim) *
                                          static_cast<std::size_t>(c.linear_value_head_dim),
                                      0.0f);

      std::vector<float> conv_out(static_cast<std::size_t>(S * cd));
      std::vector<float> core_out(static_cast<std::size_t>(S * vd));

      if (use_chunk_prefill) {
        causal_conv1d_silu_prefill(qkv_seq, lw.conv1d_weight, S, cd, c.linear_conv_kernel_dim, out.conv_states[li],
                                   conv_out);
        if (bf16_boundaries) bf16_round_vec(conv_out);
        std::vector<float> q_sep(static_cast<std::size_t>(S * kd));
        std::vector<float> k_sep(static_cast<std::size_t>(S * kd));
        std::vector<float> v_sep(static_cast<std::size_t>(S * vd));
        for (int t = 0; t < S; ++t) {
          const float* src = conv_out.data() + static_cast<std::size_t>(t * cd);
          std::copy_n(src, kd, q_sep.data() + static_cast<std::size_t>(t * kd));
          std::copy_n(src + kd, kd, k_sep.data() + static_cast<std::size_t>(t * kd));
          std::copy_n(src + 2 * kd, vd, v_sep.data() + static_cast<std::size_t>(t * vd));
        }
        gdn_chunk_prefill(q_sep, k_sep, v_sep, z_seq, a_seq, b_seq, lw.A_log, lw.dt_bias, lw.linear_norm_weight,
                          c.rms_norm_eps, S, chunk_size, c.linear_num_key_heads, nhv, c.linear_key_head_dim,
                          c.linear_value_head_dim, out.recurrent_states[li], core_out);
        if (bf16_boundaries) bf16_round_vec(core_out);
      } else {
        for (int t = 0; t < S; ++t) {
          auto co_t = std::span<float>(conv_out.data() + static_cast<std::size_t>(t * cd), cd);
          causal_conv1d_silu_step(std::span<const float>(qkv_seq.data() + static_cast<std::size_t>(t * cd), cd),
                                  lw.conv1d_weight, cd, c.linear_conv_kernel_dim, out.conv_states[li], co_t);
          if (bf16_boundaries) {
            for (float& x : co_t) x = bf16_to_f32(f32_to_bf16(x));
          }
          auto core_t = std::span<float>(core_out.data() + static_cast<std::size_t>(t * vd), vd);
          gdn_recurrent_step(
              std::span<const float>(co_t.data(), kd), std::span<const float>(co_t.data() + kd, kd),
              std::span<const float>(co_t.data() + 2 * kd, vd),
              std::span<const float>(z_seq.data() + static_cast<std::size_t>(t * vd), vd),
              std::span<const float>(a_seq.data() + static_cast<std::size_t>(t * nhv), nhv),
              std::span<const float>(b_seq.data() + static_cast<std::size_t>(t * nhv), nhv), lw.A_log, lw.dt_bias,
              lw.linear_norm_weight, c.rms_norm_eps, c.linear_num_key_heads, nhv, c.linear_key_head_dim,
              c.linear_value_head_dim, out.recurrent_states[li], core_t);
          if (bf16_boundaries) {
            for (float& x : core_t) x = bf16_to_f32(f32_to_bf16(x));
          }
        }
      }

      for (int t = 0; t < S; ++t) {
        run_proj_host(lw.out_proj, std::span<const float>(core_out.data() + static_cast<std::size_t>(t * vd), vd),
                      std::span<float>(residual.data() + static_cast<std::size_t>(t * H), H), bf16_boundaries,
                      use_calibrated_fp8_input_scale, num_threads, fp32_accum_sim);
      }
    } else {
      const int qg_dim = c.full_q_proj_out_dim();
      const int kv_dim = c.full_kv_proj_out_dim();
      const int attn_dim = c.num_attention_heads * c.head_dim;
      const auto qnw = decode_bf16_vec(lw.q_norm);
      const auto knw = decode_bf16_vec(lw.k_norm);
      out.k_caches[li].clear();
      out.v_caches[li].clear();

      std::vector<float> q_gate(static_cast<std::size_t>(qg_dim));
      std::vector<float> k_raw(static_cast<std::size_t>(kv_dim));
      std::vector<float> v_raw(static_cast<std::size_t>(kv_dim));
      std::vector<float> attn_out(static_cast<std::size_t>(attn_dim));
      std::vector<float> q_step(static_cast<std::size_t>(attn_dim));
      std::vector<float> gate_step(static_cast<std::size_t>(attn_dim));

      for (int t = 0; t < S; ++t) {
        const auto x_t = std::span<const float>(normed.data() + static_cast<std::size_t>(t * H), H);
        run_proj_host(lw.q_proj, x_t, q_gate, bf16_boundaries, use_calibrated_fp8_input_scale, num_threads,
                      fp32_accum_sim);
        run_proj_host(lw.k_proj, x_t, k_raw, bf16_boundaries, use_calibrated_fp8_input_scale, num_threads,
                      fp32_accum_sim);
        run_proj_host(lw.v_proj, x_t, v_raw, bf16_boundaries, use_calibrated_fp8_input_scale, num_threads,
                      fp32_accum_sim);
        if (bf16_boundaries) {
          for (int h = 0; h < c.num_attention_heads; ++h) {
            const float* src = q_gate.data() + static_cast<std::size_t>(h * 2 * c.head_dim);
            std::copy_n(src, c.head_dim, q_step.data() + static_cast<std::size_t>(h * c.head_dim));
            std::copy_n(src + c.head_dim, c.head_dim, gate_step.data() + static_cast<std::size_t>(h * c.head_dim));
            auto qh = std::span<float>(q_step.data() + static_cast<std::size_t>(h * c.head_dim), c.head_dim);
            rmsnorm_zero_centered(qh, qnw, c.rms_norm_eps, qh);
          }
          bf16_round_vec(q_step);
          apply_partial_rope_neox(q_step, c.num_attention_heads, c.head_dim, c.rotary_dim, t, c.rope_theta);
          bf16_round_vec(q_step);

          std::vector<float> k_step = k_raw;
          for (int kvh = 0; kvh < c.num_key_value_heads; ++kvh) {
            auto kh = std::span<float>(k_step.data() + static_cast<std::size_t>(kvh * c.head_dim), c.head_dim);
            rmsnorm_zero_centered(kh, knw, c.rms_norm_eps, kh);
          }
          bf16_round_vec(k_step);
          apply_partial_rope_neox(k_step, c.num_key_value_heads, c.head_dim, c.rotary_dim, t, c.rope_theta);
          bf16_round_vec(k_step);

          out.k_caches[li].insert(out.k_caches[li].end(), k_step.begin(), k_step.end());
          out.v_caches[li].insert(out.v_caches[li].end(), v_raw.begin(), v_raw.end());

          const float scale = 1.0f / std::sqrt(static_cast<float>(c.head_dim));
          const int group = c.num_attention_heads / c.num_key_value_heads;
          const int cache_len = t + 1;
          std::vector<float> scores(static_cast<std::size_t>(cache_len), 0.0f);
          for (int h = 0; h < c.num_attention_heads; ++h) {
            const int kvh = h / group;
            const float* qh = q_step.data() + static_cast<std::size_t>(h * c.head_dim);
            float max_s = -std::numeric_limits<float>::infinity();
            for (int ti = 0; ti < cache_len; ++ti) {
              const float* kt = out.k_caches[li].data() + static_cast<std::size_t>(ti * kv_dim + kvh * c.head_dim);
              double dot = 0.0;
              for (int d = 0; d < c.head_dim; ++d) dot += static_cast<double>(qh[d]) * static_cast<double>(kt[d]);
              scores[static_cast<std::size_t>(ti)] = static_cast<float>(dot) * scale;
              max_s = std::max(max_s, scores[static_cast<std::size_t>(ti)]);
            }
            double denom = 0.0;
            for (int ti = 0; ti < cache_len; ++ti) {
              scores[static_cast<std::size_t>(ti)] = std::exp(scores[static_cast<std::size_t>(ti)] - max_s);
              denom += static_cast<double>(scores[static_cast<std::size_t>(ti)]);
            }
            const float inv_denom = static_cast<float>(1.0 / denom);
            float* oh = attn_out.data() + static_cast<std::size_t>(h * c.head_dim);
            const float* gh = gate_step.data() + static_cast<std::size_t>(h * c.head_dim);
            std::fill_n(oh, c.head_dim, 0.0f);
            for (int ti = 0; ti < cache_len; ++ti) {
              const float w = scores[static_cast<std::size_t>(ti)] * inv_denom;
              const float* vt = out.v_caches[li].data() + static_cast<std::size_t>(ti * kv_dim + kvh * c.head_dim);
              for (int d = 0; d < c.head_dim; ++d) oh[d] += w * vt[d];
            }
            for (int d = 0; d < c.head_dim; ++d) oh[d] *= (1.0f / (1.0f + std::exp(-gh[d])));
          }
          bf16_round_vec(attn_out);
        } else {
          full_attn_gated_step(q_gate, k_raw, v_raw, qnw, knw, c.rms_norm_eps, c.rope_theta, c.num_attention_heads,
                               c.num_key_value_heads, c.head_dim, c.rotary_dim, t, out.k_caches[li], out.v_caches[li],
                               attn_out);
        }
        run_proj_host(lw.o_proj, attn_out, std::span<float>(residual.data() + static_cast<std::size_t>(t * H), H),
                      bf16_boundaries, use_calibrated_fp8_input_scale, num_threads, fp32_accum_sim);
      }
    }

    for (std::size_t i = 0; i < out.hidden_seq.size(); ++i) {
      out.hidden_seq[i] += residual[i];
      if (bf16_boundaries) out.hidden_seq[i] = bf16_to_f32(f32_to_bf16(out.hidden_seq[i]));
    }

    for (int t = 0; t < S; ++t) {
      rmsnorm_zero_centered(std::span<const float>(out.hidden_seq.data() + static_cast<std::size_t>(t * H), H),
                            post_norm_w, c.rms_norm_eps,
                            std::span<float>(normed.data() + static_cast<std::size_t>(t * H), H));
    }
    if (bf16_boundaries) bf16_round_vec(normed);

    std::vector<float> gate_buf(static_cast<std::size_t>(I));
    std::vector<float> up_buf(static_cast<std::size_t>(I));
    for (int t = 0; t < S; ++t) {
      const auto x_t = std::span<const float>(normed.data() + static_cast<std::size_t>(t * H), H);
      run_proj_host(lw.gate_proj, x_t, gate_buf, bf16_boundaries, use_calibrated_fp8_input_scale, num_threads,
                    fp32_accum_sim);
      run_proj_host(lw.up_proj, x_t, up_buf, bf16_boundaries, use_calibrated_fp8_input_scale, num_threads,
                    fp32_accum_sim);
      for (int i = 0; i < I; ++i) {
        const float sw = silu_f32(gate_buf[static_cast<std::size_t>(i)]) * up_buf[static_cast<std::size_t>(i)];
        gate_buf[static_cast<std::size_t>(i)] = bf16_boundaries ? bf16_to_f32(f32_to_bf16(sw)) : sw;
      }
      run_proj_host(lw.down_proj, gate_buf, std::span<float>(residual.data() + static_cast<std::size_t>(t * H), H),
                    bf16_boundaries, use_calibrated_fp8_input_scale, num_threads, fp32_accum_sim);
    }

    for (std::size_t i = 0; i < out.hidden_seq.size(); ++i) {
      out.hidden_seq[i] += residual[i];
      if (bf16_boundaries) out.hidden_seq[i] = bf16_to_f32(f32_to_bf16(out.hidden_seq[i]));
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  out.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return out;
}

Qwen38BlockBundleHost slice_qwen38_block_layer(const Qwen38BlockBundleHost& bundle, std::size_t layer_pos) {
  if (layer_pos >= bundle.layers.size()) {
    throw std::out_of_range("slice_qwen38_block_layer layer_pos out of range");
  }
  Qwen38BlockBundleHost out;
  out.cfg = bundle.cfg;
  out.token_ids = bundle.token_ids;
  out.prompt_embeddings_bf16 = bundle.prompt_embeddings_bf16;
  out.layers.push_back(bundle.layers[layer_pos]);
  return out;
}

std::uint32_t bf16_ulp_distance(std::uint16_t a, std::uint16_t b) noexcept {
  auto is_nonfinite = [](std::uint16_t u) noexcept -> bool { return (u & 0x7f80u) == 0x7f80u; };
  if (is_nonfinite(a) || is_nonfinite(b)) {
    return std::numeric_limits<std::uint32_t>::max();
  }
  auto ord = [](std::uint16_t u) noexcept -> std::int32_t {
    const std::int32_t mag = static_cast<std::int32_t>(u & 0x7fffu);
    return (u & 0x8000u) ? -mag : mag;
  };
  const std::int32_t diff = ord(a) - ord(b);
  return static_cast<std::uint32_t>(diff < 0 ? -diff : diff);
}

Qwen38Bf16CompareStats compare_bf16_against_rounded_oracle(std::span<const std::uint16_t> got_bf16_bits,
                                                           std::span<const float> ref_f32, std::uint32_t max_ulp,
                                                           float abs_floor) {
  if (got_bf16_bits.size() != ref_f32.size()) {
    throw std::invalid_argument("compare_bf16_against_rounded_oracle size mismatch");
  }
  Qwen38Bf16CompareStats stats;
  for (std::size_t i = 0; i < ref_f32.size(); ++i) {
    const std::uint16_t ref_bits = f32_to_bf16(ref_f32[i]);
    const float ref_b = bf16_to_f32(ref_bits);
    const float got_b = bf16_to_f32(got_bf16_bits[i]);
    if (!std::isfinite(ref_f32[i]) || !std::isfinite(ref_b) || !std::isfinite(got_b)) {
      ++stats.criterion_violations;
      ++stats.non_exact_bf16_count;
      stats.max_bf16_ulp = std::numeric_limits<std::uint32_t>::max();
      continue;
    }
    const float abs_d = std::fabs(got_b - ref_b);
    const std::uint32_t ulp = bf16_ulp_distance(got_bf16_bits[i], ref_bits);
    stats.max_abs_vs_bf16_oracle = std::max(stats.max_abs_vs_bf16_oracle, abs_d);
    if (ulp == 0) {
      ++stats.exact_bf16_matches;
    } else {
      ++stats.non_exact_bf16_count;
    }
    if (abs_d > abs_floor || ulp <= max_ulp) {
      stats.max_bf16_ulp = std::max(stats.max_bf16_ulp, ulp);
    }
    if (ulp > max_ulp && abs_d > abs_floor) {
      ++stats.criterion_violations;
    }
  }
  stats.passed = (stats.criterion_violations == 0);
  return stats;
}

Qwen38Gate1LinearFixtures make_gate1_linear_fixtures(std::uint32_t seed) {
  Qwen38Gate1LinearFixtures fx;
  const int M = fx.m;
  const int N = fx.n;
  const int K = fx.k;
  std::mt19937 rng(seed);
  auto rand_vec = [&](std::size_t n, float stddev) {
    std::normal_distribution<float> d(0.0f, stddev);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
  };
  auto to_bf16_vec = [](std::span<const float> src) {
    std::vector<std::uint16_t> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) out[i] = f32_to_bf16(src[i]);
    return out;
  };

  // 1. NVFP4 fixture (M=4, N=256, K=5120) — exact RNG sequence preserved.
  {
    const auto x_f = rand_vec(static_cast<std::size_t>(M * K), 0.25f);
    fx.nvfp4_x_bf16 = to_bf16_vec(x_f);
    fx.nvfp4_x_q = decode_bf16_vec(fx.nvfp4_x_bf16);
    fx.nvfp4_packed.resize(static_cast<std::size_t>(N * (K / 2)));
    fx.nvfp4_scales.resize(static_cast<std::size_t>(N * (K / 16)));
    for (auto& p : fx.nvfp4_packed) p = static_cast<std::uint8_t>(rng() & 0xffu);
    for (auto& s : fx.nvfp4_scales) s = static_cast<std::uint8_t>(0x38u + (rng() % 20));
    fx.nvfp4_ref_f32.assign(static_cast<std::size_t>(M * N), 0.0f);
    for (int m = 0; m < M; ++m) {
      nvfp4_weight_only_linear(std::span<const float>(fx.nvfp4_x_q.data() + static_cast<std::size_t>(m * K), K),
                               fx.nvfp4_packed, fx.nvfp4_scales, fx.nvfp4_ws2, N, K,
                               std::span<float>(fx.nvfp4_ref_f32.data() + static_cast<std::size_t>(m * N), N));
    }
    fx.nvfp4_ref_bf16 = to_bf16_vec(fx.nvfp4_ref_f32);
  }

  // 2. FP8 fixture (M=4, N=256, K=5120) — weight-only and calibrated references.
  {
    const auto x_f = rand_vec(static_cast<std::size_t>(M * K), 0.3f);
    fx.fp8_x_bf16 = to_bf16_vec(x_f);
    fx.fp8_x_q = decode_bf16_vec(fx.fp8_x_bf16);
    fx.fp8_w.resize(static_cast<std::size_t>(N * K));
    for (auto& b : fx.fp8_w) b = quantize_e4m3fn_fast(static_cast<float>(static_cast<int>(rng() % 200) - 100) * 0.01f);
    fx.fp8_wo_ref_f32.assign(static_cast<std::size_t>(M * N), 0.0f);
    fx.fp8_cal_ref_f32.assign(static_cast<std::size_t>(M * N), 0.0f);
    for (int m = 0; m < M; ++m) {
      const auto xm = std::span<const float>(fx.fp8_x_q.data() + static_cast<std::size_t>(m * K), K);
      fp8_weight_only_linear(xm, fx.fp8_w, fx.fp8_ws, N, K,
                             std::span<float>(fx.fp8_wo_ref_f32.data() + static_cast<std::size_t>(m * N), N));
      fp8_calibrated_linear(xm, fx.fp8_in_sc, fx.fp8_w, fx.fp8_ws, N, K,
                            std::span<float>(fx.fp8_cal_ref_f32.data() + static_cast<std::size_t>(m * N), N));
    }
    fx.fp8_wo_ref_bf16 = to_bf16_vec(fx.fp8_wo_ref_f32);
    fx.fp8_cal_ref_bf16 = to_bf16_vec(fx.fp8_cal_ref_f32);
  }
  return fx;
}

void simulate_warp_f32_nvfp4_linear(std::span<const float> x_in, std::span<const std::uint8_t> packed_u8,
                                    std::span<const std::uint8_t> block_scales_e4m3, float weight_scale_2, int m,
                                    int out_features, int in_features, std::span<float> out) {
  if (m <= 0 || out_features <= 0 || in_features <= 0 || in_features % 16 != 0 ||
      x_in.size() != static_cast<std::size_t>(m * in_features) ||
      packed_u8.size() != static_cast<std::size_t>(out_features * (in_features / 2)) ||
      block_scales_e4m3.size() != static_cast<std::size_t>(out_features * (in_features / 16)) ||
      out.size() != static_cast<std::size_t>(m * out_features)) {
    throw std::invalid_argument("simulate_warp_f32_nvfp4_linear shape mismatch");
  }
  const int blocks = in_features / 16;
  const int half_k = in_features / 2;
  for (int row = 0; row < m; ++row) {
    const float* x_row = x_in.data() + static_cast<std::size_t>(row * in_features);
    parallel_rows(out_features, static_cast<std::size_t>(out_features * in_features), 0, [&](int r0, int r1) {
      for (int n = r0; n < r1; ++n) {
        const std::uint8_t* prow = packed_u8.data() + static_cast<std::size_t>(n * half_k);
        const std::uint8_t* srow = block_scales_e4m3.data() + static_cast<std::size_t>(n * blocks);
        out[static_cast<std::size_t>(row * out_features + n)] = warp_reduce_32([&](int lane) {
          float acc = 0.0f;
          for (int b = lane; b < blocks; b += 32) {
            const float sc = decode_e4m3fn(srow[b]) * weight_scale_2;
            float bsum = 0.0f;
            const int k0 = b * 16;
            const int p0 = b * 8;
            for (int i = 0; i < 8; ++i) {
              const std::uint8_t pb = prow[p0 + i];
              bsum += x_row[k0 + 2 * i] * decode_e2m1(pb & 0x0fu);
              bsum += x_row[k0 + 2 * i + 1] * decode_e2m1((pb >> 4) & 0x0fu);
            }
            acc += bsum * sc;
          }
          return acc;
        });
      }
    });
  }
}

void simulate_warp_f32_fp8_linear(std::span<const float> x_in, std::span<const std::uint8_t> w_e4m3,
                                  float weight_scale, float input_scale, int m, int out_features, int in_features,
                                  std::span<float> out) {
  if (m <= 0 || out_features <= 0 || in_features <= 0 ||
      x_in.size() != static_cast<std::size_t>(m * in_features) ||
      w_e4m3.size() != static_cast<std::size_t>(out_features * in_features) ||
      out.size() != static_cast<std::size_t>(m * out_features)) {
    throw std::invalid_argument("simulate_warp_f32_fp8_linear shape mismatch");
  }
  const bool calibrated = (input_scale > 0.0f);
  const float inv_in = calibrated ? (1.0f / input_scale) : 1.0f;
  for (int row = 0; row < m; ++row) {
    std::vector<float> x_eff(static_cast<std::size_t>(in_features));
    const float* x_row = x_in.data() + static_cast<std::size_t>(row * in_features);
    for (int k = 0; k < in_features; ++k) {
      x_eff[static_cast<std::size_t>(k)] =
          calibrated ? (decode_e4m3fn(quantize_e4m3fn_fast(x_row[k] * inv_in)) * input_scale) : x_row[k];
    }
    parallel_rows(out_features, static_cast<std::size_t>(out_features * in_features), 0, [&](int r0, int r1) {
      for (int n = r0; n < r1; ++n) {
        const std::uint8_t* wrow = w_e4m3.data() + static_cast<std::size_t>(n * in_features);
        const float acc_f = warp_reduce_32([&](int lane) {
          float acc = 0.0f;
          for (int k = lane; k < in_features; k += 32) {
            acc += x_eff[static_cast<std::size_t>(k)] * decode_e4m3fn(wrow[k]);
          }
          return acc;
        });
        out[static_cast<std::size_t>(row * out_features + n)] = acc_f * weight_scale;
      }
    });
  }
}

const std::vector<std::string>& qwen38_gate1_required_row_names() {
  static const std::vector<std::string> kRows = {
      "unit_nvfp4_linear_f32",
      "unit_nvfp4_linear_bf16",
      "unit_fp8_linear_weight_only_f32",
      "unit_fp8_linear_weight_only_bf16",
      "unit_fp8_linear_calibrated_f32",
      "unit_fp8_linear_calibrated_bf16",
      "unit_rmsnorm_zero_centered",
      "unit_rmsnorm_gated_silu",
      "unit_partial_rope_neox",
      "unit_causal_conv1d_silu",
      "unit_gdn_recurrent_step",
      "unit_gdn_chunk_prefill",
      "unit_full_attn_gated",
      "synth_bf16_1L_gdn",
      "synth_bf16_1L_full",
      "synth_bf16_4L_stepwise",
      "synth_bf16_4L_chunk",
      "synth_nvfp4_fp8_1L_gdn",
      "synth_nvfp4_fp8_1L_full",
      "synth_nvfp4_fp8_4L_stepwise",
      "synth_nvfp4_fp8_4L_chunk",
      "qwen08b_1L_gdn",
      "qwen08b_1L_full",
      "qwen08b_4L_stepwise",
      "qwen08b_4L_chunk",
      "qwen27b_1L_gdn",
      "qwen27b_1L_full",
      "qwen27b_4L_stepwise",
      "qwen27b_4L_chunk",
  };
  return kRows;
}

Qwen38Gate1CoverageSummary evaluate_gate1_coverage(std::span<const Qwen38Gate1RowStatus> rows, bool diagnostic_mode,
                                                   bool qwen27b_pins_verified) {
  Qwen38Gate1CoverageSummary summary;
  const auto& required = qwen38_gate1_required_row_names();
  summary.mode = diagnostic_mode ? "diagnostic" : "qualification";
  summary.qwen27b_pins_verified = qwen27b_pins_verified;
  summary.required_row_count = required.size();

  std::unordered_map<std::string, const Qwen38Gate1RowStatus*> by_name;
  std::unordered_set<std::string> failed_set;
  auto push_failed = [&](const std::string& name) {
    if (failed_set.insert(name).second) summary.failed_rows.push_back(name);
  };

  std::size_t executed_count = 0;
  for (const auto& r : rows) {
    if (!by_name.emplace(r.name, &r).second) {
      push_failed(r.name);
      continue;
    }
    if (r.status == "pass") {
      ++executed_count;
      if (!r.passed) push_failed(r.name);
    } else if (r.status == "fail") {
      ++executed_count;
      push_failed(r.name);
    } else if (r.status == "skipped_diagnostic" || r.status == "missing_checkpoint") {
      if (r.passed) push_failed(r.name);
    } else {
      push_failed(r.name);
    }
  }

  for (const std::string& req : required) {
    const auto it = by_name.find(req);
    if (it == by_name.end()) {
      summary.missing_required_rows.push_back(req);
      continue;
    }
    const auto* r = it->second;
    if (r->status == "pass" && r->passed) {
      ++summary.passed_required_row_count;
    } else if (r->status == "skipped_diagnostic" || r->status == "missing_checkpoint") {
      summary.missing_required_rows.push_back(req);
    }
  }

  summary.complete_coverage = summary.missing_required_rows.empty();
  summary.executed_checks_passed = summary.failed_rows.empty() && (executed_count > 0);
  summary.qualification_passed =
      !diagnostic_mode && summary.complete_coverage && summary.executed_checks_passed && qwen27b_pins_verified;
  summary.all_passed = summary.qualification_passed;
  if (summary.qualification_passed) {
    summary.coverage_status = "complete_qualified";
  } else if (diagnostic_mode && summary.executed_checks_passed) {
    summary.coverage_status = "incomplete_diagnostic";
  } else {
    summary.coverage_status = "failed";
  }
  return summary;
}

Qwen38VectorCompareStats compare_f32_vectors(std::span<const float> got, std::span<const float> ref) {
  Qwen38VectorCompareStats s;
  if (got.size() != ref.size()) {
    s.finite = false;
    s.max_abs = std::numeric_limits<float>::infinity();
    s.rel_max_abs = std::numeric_limits<float>::infinity();
    s.rel_l2 = std::numeric_limits<float>::infinity();
    s.cosine_sim = 0.0;
    return s;
  }
  double diff_sq = 0.0;
  double got_sq = 0.0;
  double ref_sq = 0.0;
  double dot = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float g = got[i];
    const float r = ref[i];
    if (!std::isfinite(g) || !std::isfinite(r)) {
      s.finite = false;
      s.max_abs = std::numeric_limits<float>::infinity();
      s.rel_max_abs = std::numeric_limits<float>::infinity();
      s.rel_l2 = std::numeric_limits<float>::infinity();
      s.cosine_sim = 0.0;
      return s;
    }
    const double dg = static_cast<double>(g);
    const double dr = static_cast<double>(r);
    const double d = dg - dr;
    s.max_abs = std::max(s.max_abs, std::fabs(g - r));
    s.ref_max_abs = std::max(s.ref_max_abs, std::fabs(r));
    diff_sq += d * d;
    got_sq += dg * dg;
    ref_sq += dr * dr;
    dot += dg * dr;
  }
  const double diff_l2 = std::sqrt(diff_sq);
  const double got_l2 = std::sqrt(got_sq);
  const double ref_l2 = std::sqrt(ref_sq);
  s.rel_max_abs =
      static_cast<float>(static_cast<double>(s.max_abs) / std::max(1e-6, static_cast<double>(s.ref_max_abs)));
  s.rel_l2 = static_cast<float>(diff_l2 / std::max(1e-6, ref_l2));
  if (got_l2 == 0.0 && ref_l2 == 0.0) {
    s.cosine_sim = 1.0;
  } else if (got_l2 == 0.0 || ref_l2 == 0.0) {
    s.cosine_sim = 0.0;
  } else {
    s.cosine_sim = dot / (got_l2 * ref_l2);
  }
  return s;
}

Qwen38BlockParityResult evaluate_block_parity(const Qwen38BlockBundleHost& bundle, const Qwen38BlockRunOutput& got,
                                              const Qwen38BlockRunOutput& ref, Qwen38BlockParityTier tier) {
  Qwen38BlockParityResult res;
  res.hidden = compare_f32_vectors(got.hidden_seq, ref.hidden_seq);
  if (got.hidden_seq.size() == ref.hidden_seq.size()) {
    std::vector<std::uint16_t> got_bits(got.hidden_seq.size());
    for (std::size_t i = 0; i < got.hidden_seq.size(); ++i) got_bits[i] = f32_to_bf16(got.hidden_seq[i]);
    res.hidden_bf16 = compare_bf16_against_rounded_oracle(got_bits, ref.hidden_seq, 1u, 1e-3f);
  }
  if (got.recurrent_states.size() != bundle.layers.size() || ref.recurrent_states.size() != bundle.layers.size()) {
    res.gdn_finite = false;
  } else {
    for (std::size_t li = 0; li < bundle.layers.size(); ++li) {
      if (bundle.layers[li].kind != Qwen38LayerKind::LinearAttention) continue;
      const auto st = compare_f32_vectors(got.recurrent_states[li], ref.recurrent_states[li]);
      if (!st.finite) {
        res.gdn_finite = false;
        continue;
      }
      if (li == 0) {
        res.layer0_gdn_max_abs = std::max(res.layer0_gdn_max_abs, st.max_abs);
      } else {
        res.stacked_gdn_max_abs = std::max(res.stacked_gdn_max_abs, st.max_abs);
        res.stacked_gdn_rel_l2 = std::max(res.stacked_gdn_rel_l2, st.rel_l2);
      }
    }
  }
  if (tier == Qwen38BlockParityTier::SyntheticStrict) {
    res.criterion = "synth_strict: hidden_max_abs <= 1e-3 && gdn_max_abs <= 2e-4 && bf16_ulp <= 1 || abs <= 1e-3";
    res.passed = res.hidden.finite && res.gdn_finite && res.hidden.max_abs <= 1e-3f &&
                 res.layer0_gdn_max_abs <= 2e-4f && res.stacked_gdn_max_abs <= 2e-4f && res.hidden_bf16.passed;
  } else if (tier == Qwen38BlockParityTier::RealSingleLayer) {
    res.criterion =
        "real_1L_bf16: layer0_gdn_max_abs <= 2e-4 && hidden_rel_l2 <= 2e-3 && hidden_cosine >= 0.99999 && "
        "hidden_rel_max_abs <= 1e-2";
    res.passed = res.hidden.finite && res.gdn_finite && res.layer0_gdn_max_abs <= 2e-4f &&
                 res.hidden.rel_l2 <= 2e-3f && res.hidden.cosine_sim >= 0.99999 && res.hidden.rel_max_abs <= 1e-2f;
  } else {
    res.criterion =
        "real_4L_bf16: layer0_gdn_max_abs <= 2e-4 && stacked_gdn_max_abs <= 1e-2 && stacked_gdn_rel_l2 <= 2.5e-2 && "
        "hidden_rel_l2 <= 1.5e-2 && hidden_cosine >= 0.99985 && hidden_rel_max_abs <= 2.5e-2";
    res.passed = res.hidden.finite && res.gdn_finite && res.layer0_gdn_max_abs <= 2e-4f &&
                 res.stacked_gdn_max_abs <= 1e-2f && res.stacked_gdn_rel_l2 <= 2.5e-2f &&
                 res.hidden.rel_l2 <= 1.5e-2f && res.hidden.cosine_sim >= 0.99985 && res.hidden.rel_max_abs <= 2.5e-2f;
  }
  return res;
}

}  // namespace ingine::backends::qwen38

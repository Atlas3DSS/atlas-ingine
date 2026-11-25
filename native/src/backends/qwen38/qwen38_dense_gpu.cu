#include "backends/qwen38/qwen38_dense_gpu.h"

#include "backends/qwen38/qwen38_cpu_ops.h"
#include "backends/qwen38/qwen38_cuda_ops.h"
#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>

namespace ingine::backends::qwen38 {
namespace {

template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  std::size_t count = 0;
  DevBuf() = default;
  explicit DevBuf(std::size_t n) { allocate(n); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : ptr(std::exchange(o.ptr, nullptr)), count(std::exchange(o.count, 0)) {}
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) {
      free();
      ptr = std::exchange(o.ptr, nullptr);
      count = std::exchange(o.count, 0);
    }
    return *this;
  }
  ~DevBuf() { free(); }
  void allocate(std::size_t n) {
    free();
    count = n;
    if (n > 0) CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
  }
  void free() noexcept {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
      count = 0;
    }
  }
};

struct DevProj {
  int out_features = 0;
  int in_features = 0;
  Qwen38LinearQuantKind quant = Qwen38LinearQuantKind::Bf16;
  const __nv_bfloat16* w_bf16 = nullptr;
  const std::uint8_t* w_u8 = nullptr;
  const std::uint8_t* w_scales = nullptr;
  float weight_scale = 1.0f;
  float weight_scale_2 = 1.0f;
  float input_scale = 0.0f;
};

struct DevLayer {
  Qwen38LayerKind kind = Qwen38LayerKind::LinearAttention;
  const __nv_bfloat16* input_layernorm = nullptr;
  const __nv_bfloat16* post_attention_layernorm = nullptr;

  DevProj in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj;
  const float* a_log = nullptr;
  const float* dt_bias = nullptr;
  const float* conv1d_weight = nullptr;
  const float* linear_norm_weight = nullptr;

  DevProj q_proj, k_proj, v_proj, o_proj;
  const __nv_bfloat16* q_norm = nullptr;
  const __nv_bfloat16* k_norm = nullptr;

  DevProj gate_proj, up_proj, down_proj;
};

std::uint64_t align_up(std::uint64_t x, std::uint64_t a = 256) noexcept { return (x + (a - 1)) & ~(a - 1); }

std::uint64_t parse_header_len_le(std::span<const std::uint8_t> eight_bytes) {
  std::uint64_t h = 0;
  for (int i = 0; i < 8; ++i) h |= (static_cast<std::uint64_t>(eight_bytes[i]) << (8 * i));
  if (h == 0 || h > (64ull * 1024ull * 1024ull)) {
    throw std::invalid_argument("invalid safetensors header length during GPU load");
  }
  return h;
}

float fp16_to_f32(std::uint16_t bits) noexcept {
  const std::uint32_t sign = (static_cast<std::uint32_t>(bits) & 0x8000u) << 16;
  const std::uint32_t exp = (bits >> 10) & 0x1fu;
  const std::uint32_t frac = bits & 0x3ffu;
  if (exp == 0) {
    if (frac == 0) {
      float out = 0.0f;
      std::memcpy(&out, &sign, 4);
      return out;
    }
    return std::ldexp(static_cast<float>(frac) / 1024.0f, -14) * (sign ? -1.0f : 1.0f);
  }
  if (exp == 0x1fu) {
    const std::uint32_t inf_or_nan = sign | 0x7f800000u | (frac << 13);
    float out = 0.0f;
    std::memcpy(&out, &inf_or_nan, 4);
    return out;
  }
  const std::uint32_t f_bits = sign | ((exp + 112u) << 23) | (frac << 13);
  float out = 0.0f;
  std::memcpy(&out, &f_bits, 4);
  return out;
}

float decode_scalar_bytes(std::span<const std::uint8_t> raw, std::string_view dtype, std::string_view name) {
  if (dtype == "F32") {
    if (raw.size() != 4) throw std::invalid_argument("scalar F32 byte size != 4 for " + std::string(name));
    float v = 0.0f;
    std::memcpy(&v, raw.data(), 4);
    return v;
  }
  if (dtype == "BF16") {
    if (raw.size() != 2) throw std::invalid_argument("scalar BF16 byte size != 2 for " + std::string(name));
    std::uint16_t b = 0;
    std::memcpy(&b, raw.data(), 2);
    return bf16_to_f32(b);
  }
  if (dtype == "F16") {
    if (raw.size() != 2) throw std::invalid_argument("scalar F16 byte size != 2 for " + std::string(name));
    std::uint16_t b = 0;
    std::memcpy(&b, raw.data(), 2);
    return fp16_to_f32(b);
  }
  throw std::invalid_argument("unsupported scalar dtype '" + std::string(dtype) + "' for " + std::string(name));
}

void decode_f32_vec_bytes(std::span<const std::uint8_t> raw, std::string_view dtype, std::size_t expected_elems,
                          std::string_view name, std::vector<float>& dst, std::size_t dst_off) {
  if (dtype == "F32") {
    if (raw.size() != expected_elems * 4) throw std::invalid_argument("F32 byte mismatch for " + std::string(name));
    std::memcpy(dst.data() + dst_off, raw.data(), raw.size());
    return;
  }
  if (dtype == "BF16") {
    if (raw.size() != expected_elems * 2) throw std::invalid_argument("BF16 byte mismatch for " + std::string(name));
    for (std::size_t i = 0; i < expected_elems; ++i) {
      std::uint16_t b = 0;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      dst[dst_off + i] = bf16_to_f32(b);
    }
    return;
  }
  throw std::invalid_argument("unsupported 1D GDN dtype '" + std::string(dtype) + "' for " + std::string(name));
}

void stream_unpinned_shard(const Qwen38Manifest& manifest, const std::filesystem::path& model_dir,
                           const std::string& shard,
                           const std::function<void(std::uint64_t, std::span<const std::uint8_t>)>& on_chunk,
                           const std::function<void()>& on_abort) {
  const std::filesystem::path sp = model_dir / shard;
  int fd = ::open(sp.c_str(), O_RDONLY);
  if (fd < 0) {
    if (on_abort) on_abort();
    throw std::invalid_argument("cannot open shard " + sp.string() + ": " + std::strerror(errno));
  }
  try {
    struct stat st {};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
      throw std::invalid_argument("not a regular file: " + sp.string());
    }
    const auto vf_it = manifest.verified_files.find(shard);
    if (vf_it != manifest.verified_files.end()) {
      if (static_cast<std::uint64_t>(st.st_dev) != vf_it->second.dev ||
          static_cast<std::uint64_t>(st.st_ino) != vf_it->second.ino ||
          static_cast<std::uint64_t>(st.st_size) != vf_it->second.byte_size) {
        throw std::invalid_argument("open FD mismatch for unpinned shard " + shard);
      }
    }
    std::vector<std::uint8_t> buf(4ull * 1024ull * 1024ull);
    std::uint64_t offset = 0;
    while (true) {
      const ssize_t got = ::pread(fd, buf.data(), buf.size(), static_cast<off_t>(offset));
      if (got < 0) {
        if (errno == EINTR) continue;
        throw std::invalid_argument("pread failed on " + sp.string() + ": " + std::strerror(errno));
      }
      if (got == 0) break;
      on_chunk(offset, std::span<const std::uint8_t>(buf.data(), static_cast<std::size_t>(got)));
      offset += static_cast<std::uint64_t>(got);
    }
    ::close(fd);
  } catch (...) {
    ::close(fd);
    if (on_abort) on_abort();
    throw;
  }
}

struct CopySpec {
  const Qwen38TensorInfo* info = nullptr;
  bool to_device = true;
  std::uint64_t dev_offset = 0;
  std::vector<std::uint8_t>* host_dst = nullptr;
};

struct ActiveTarget {
  std::uint64_t file_begin = 0;
  std::uint64_t file_end = 0;
  bool to_device = true;
  std::uint64_t dev_offset = 0;
  std::vector<std::uint8_t>* host_dst = nullptr;
};

struct HostScalarEntry {
  float* target_field = nullptr;
  std::string dtype;
  std::vector<std::uint8_t> raw;
};

struct HostF32VecEntry {
  std::size_t f32_offset = 0;
  std::size_t elems = 0;
  std::string dtype;
  std::string name;
  std::vector<std::uint8_t> raw;
};

}  // namespace

struct Qwen38DenseSession::StateImpl {
  std::vector<DevBuf<float>> conv_states;
  std::vector<DevBuf<float>> rec_states;
  std::vector<DevBuf<__nv_bfloat16>> k_caches;
  std::vector<DevBuf<__nv_bfloat16>> v_caches;
  std::size_t total_state_bytes = 0;
};

struct Qwen38DenseBackend::Impl {
  DeviceContext ctx;
  Qwen38Manifest manifest;
  Qwen38DenseLoadOptions opts;

  DevBuf<std::uint8_t> weights_arena;
  DevBuf<float> f32_arena;

  const __nv_bfloat16* embed_tokens = nullptr;
  const __nv_bfloat16* final_norm = nullptr;
  DevProj lm_head;
  std::vector<DevLayer> layers;

  // Shared execution scratch buffers (FP32 activations for bit-exact batch invariance across M=1..256).
  DevBuf<float> act_quant_scratch;
  DevBuf<std::int32_t> token_ids_dev;
  DevBuf<int> gather_rows_dev;
  DevBuf<float> hidden;
  DevBuf<float> normed;
  DevBuf<float> residual;
  DevBuf<float> qkv;
  DevBuf<float> conv_out;
  DevBuf<float> z_buf;
  DevBuf<float> a_buf;
  DevBuf<float> b_buf;
  DevBuf<float> core_out;
  DevBuf<float> gdn_scratch;

  DevBuf<float> q_gate;
  DevBuf<float> k_raw;
  DevBuf<float> v_raw;
  DevBuf<float> q_tmp;
  DevBuf<float> gate_tmp;
  DevBuf<float> attn_out;

  DevBuf<float> mlp_gate;
  DevBuf<float> mlp_up;

  static constexpr int kLogitBatchMax = 64;
  DevBuf<float> cap_hidden;
  DevBuf<float> cap_normed;
  DevBuf<float> logits_f32;
  std::vector<float> host_logits_stage;
  std::vector<std::int32_t> decode_single_tok;

  std::unique_ptr<Qwen38DenseSession> default_session;
  std::uint64_t scratch_bytes = 0;
  std::uint64_t max_state_bytes = 0;

  explicit Impl(int dev_id) : ctx(dev_id) { ctx.bind_to_current_thread(); }

  void proj_forward_f32(const DevProj& p, const float* x, int m, float* out) {
    if (m <= 0) throw std::invalid_argument("proj_forward_f32 m <= 0");
    if (p.quant == Qwen38LinearQuantKind::Bf16) {
      qwen38_cuda_bf16_linear_f32_batched(x, p.w_bf16, m, p.out_features, p.in_features, out, ctx.stream);
    } else if (p.quant == Qwen38LinearQuantKind::Fp8E4M3) {
      const float* x_use = x;
      if (opts.use_calibrated_fp8 && p.input_scale > 0.0f) {
        const std::size_t x_elems = static_cast<std::size_t>(m) * static_cast<std::size_t>(p.in_features);
        if (x_elems > act_quant_scratch.count) throw std::runtime_error("act_quant_scratch overflow");
        qwen38_cuda_quant_fp8_act_f32(x, p.input_scale, x_elems, act_quant_scratch.ptr, ctx.stream);
        x_use = act_quant_scratch.ptr;
      }
      qwen38_cuda_fp8_linear_f32_batched(x_use, p.w_u8, p.weight_scale, 0.0f, m, p.out_features, p.in_features, out,
                                         ctx.stream);
    } else {
      qwen38_cuda_nvfp4_linear_f32_batched(x, p.w_u8, p.w_scales, p.weight_scale_2, m, p.out_features, p.in_features,
                                           out, ctx.stream);
    }
  }

  void lm_head_forward_f32(const float* x, int m, float* out_f32) {
    const DevProj& p = lm_head;
    if (m <= 0) throw std::invalid_argument("lm_head_forward_f32 m <= 0");
    if (p.quant == Qwen38LinearQuantKind::Bf16) {
      qwen38_cuda_bf16_linear_f32_batched(x, p.w_bf16, m, p.out_features, p.in_features, out_f32, ctx.stream);
    } else if (p.quant == Qwen38LinearQuantKind::Nvfp4) {
      qwen38_cuda_nvfp4_linear_f32_batched(x, p.w_u8, p.w_scales, p.weight_scale_2, m, p.out_features, p.in_features,
                                           out_f32, ctx.stream);
    } else {
      throw std::invalid_argument("unsupported lm_head quant");
    }
  }

  void forward_range(Qwen38DenseSession& sess, std::span<const std::int32_t> token_ids,
                     std::span<const int> sorted_capture_positions, bool use_chunk_prefill,
                     const ingine::model::LogitCallback& on_logits, double* prefill_ms_out) {
    ctx.bind_to_current_thread();
    const auto& c = manifest.config;
    const int S = static_cast<int>(token_ids.size());
    const int H = c.hidden_size;
    const int I = c.intermediate_size;
    const int V = c.vocab_size;
    const int pos_start = static_cast<int>(sess.committed_tokens_.size());
    if (S == 0) return;
    if (pos_start + S > opts.max_ctx) {
      throw std::invalid_argument("Qwen38DenseBackend sequence exceeds max_ctx");
    }

    const int step_cap = use_chunk_prefill ? opts.prefill_chunk : 1;
    std::size_t cap_idx = 0;
    while (cap_idx < sorted_capture_positions.size() && sorted_capture_positions[cap_idx] < pos_start) {
      ++cap_idx;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const int cd = c.linear_conv_dim();
    const int nhv = c.linear_num_value_heads;
    const int nkh = c.linear_num_key_heads;
    const int k_dim = c.linear_key_head_dim;
    const int v_dim = c.linear_value_head_dim;

    for (int off = 0; off < S; off += step_cap) {
      const int M = std::min(step_cap, S - off);
      const int global_pos = pos_start + off;
      CUDA_CHECK(cudaMemcpyAsync(token_ids_dev.ptr, token_ids.data() + off,
                                 static_cast<std::size_t>(M) * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                 ctx.stream));
      qwen38_cuda_embed_lookup_f32(embed_tokens, token_ids_dev.ptr, M, H, V, hidden.ptr, ctx.stream);

      std::size_t lin_idx = 0;
      std::size_t full_idx = 0;
      for (std::size_t li = 0; li < layers.size(); ++li) {
        const auto& l = layers[li];
        qwen38_cuda_rmsnorm_zero_centered_f32(hidden.ptr, l.input_layernorm, c.rms_norm_eps, M, H, normed.ptr,
                                              ctx.stream);
        if (l.kind == Qwen38LayerKind::LinearAttention) {
          proj_forward_f32(l.in_proj_qkv, normed.ptr, M, qkv.ptr);
          proj_forward_f32(l.in_proj_z, normed.ptr, M, z_buf.ptr);
          proj_forward_f32(l.in_proj_a, normed.ptr, M, a_buf.ptr);
          proj_forward_f32(l.in_proj_b, normed.ptr, M, b_buf.ptr);

          float* conv_st = sess.state_->conv_states[lin_idx].ptr;
          float* rec_st = sess.state_->rec_states[lin_idx].ptr;
          qwen38_cuda_causal_conv1d_silu_prefill_f32(qkv.ptr, l.conv1d_weight, M, cd, c.linear_conv_kernel_dim,
                                                     conv_st, conv_out.ptr, ctx.stream);
          qwen38_cuda_gdn_recurrent_prefill_f32(conv_out.ptr, z_buf.ptr, a_buf.ptr, b_buf.ptr, l.a_log, l.dt_bias,
                                                l.linear_norm_weight, c.rms_norm_eps, M, nkh, nhv, k_dim, v_dim, rec_st,
                                                gdn_scratch.ptr, core_out.ptr, ctx.stream);
          proj_forward_f32(l.out_proj, core_out.ptr, M, residual.ptr);
          ++lin_idx;
        } else {
          proj_forward_f32(l.q_proj, normed.ptr, M, q_gate.ptr);
          proj_forward_f32(l.k_proj, normed.ptr, M, k_raw.ptr);
          proj_forward_f32(l.v_proj, normed.ptr, M, v_raw.ptr);
          qwen38_cuda_full_attn_gated_f32_kvbf16(
              q_gate.ptr, k_raw.ptr, v_raw.ptr, l.q_norm, l.k_norm, c.rms_norm_eps, c.rope_theta, M, global_pos,
              c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rotary_dim,
              sess.state_->k_caches[full_idx].ptr, sess.state_->v_caches[full_idx].ptr, opts.max_ctx, q_tmp.ptr,
              gate_tmp.ptr, attn_out.ptr, ctx.stream);
          proj_forward_f32(l.o_proj, attn_out.ptr, M, residual.ptr);
          ++full_idx;
        }

        qwen38_cuda_add_residual_f32(hidden.ptr, residual.ptr, static_cast<std::size_t>(M * H), ctx.stream);
        qwen38_cuda_rmsnorm_zero_centered_f32(hidden.ptr, l.post_attention_layernorm, c.rms_norm_eps, M, H,
                                              normed.ptr, ctx.stream);
        proj_forward_f32(l.gate_proj, normed.ptr, M, mlp_gate.ptr);
        proj_forward_f32(l.up_proj, normed.ptr, M, mlp_up.ptr);
        qwen38_cuda_swiglu_mul_f32(mlp_gate.ptr, mlp_up.ptr, mlp_gate.ptr, static_cast<std::size_t>(M * I),
                                   ctx.stream);
        proj_forward_f32(l.down_proj, mlp_gate.ptr, M, residual.ptr);
        qwen38_cuda_add_residual_f32(hidden.ptr, residual.ptr, static_cast<std::size_t>(M * H), ctx.stream);
      }

      std::vector<int> local_rows;
      std::vector<int> global_rows;
      while (cap_idx < sorted_capture_positions.size() && sorted_capture_positions[cap_idx] < global_pos + M) {
        const int gp = sorted_capture_positions[cap_idx];
        local_rows.push_back(gp - global_pos);
        global_rows.push_back(gp);
        ++cap_idx;
      }

      if (!local_rows.empty() && on_logits) {
        for (std::size_t b0 = 0; b0 < local_rows.size(); b0 += kLogitBatchMax) {
          const int bc = static_cast<int>(std::min<std::size_t>(kLogitBatchMax, local_rows.size() - b0));
          const float* x_cap = nullptr;
          if (bc == 1 && M == 1) {
            x_cap = hidden.ptr;
          } else {
            CUDA_CHECK(cudaMemcpyAsync(gather_rows_dev.ptr, local_rows.data() + b0,
                                       static_cast<std::size_t>(bc) * sizeof(int), cudaMemcpyHostToDevice, ctx.stream));
            qwen38_cuda_gather_rows_f32(hidden.ptr, gather_rows_dev.ptr, bc, H, cap_hidden.ptr, ctx.stream);
            x_cap = cap_hidden.ptr;
          }
          qwen38_cuda_rmsnorm_zero_centered_f32(x_cap, final_norm, c.rms_norm_eps, bc, H, cap_normed.ptr, ctx.stream);
          lm_head_forward_f32(cap_normed.ptr, bc, logits_f32.ptr);
          const std::size_t copy_elems = static_cast<std::size_t>(bc) * static_cast<std::size_t>(V);
          CUDA_CHECK(cudaMemcpyAsync(host_logits_stage.data(), logits_f32.ptr, copy_elems * sizeof(float),
                                     cudaMemcpyDeviceToHost, ctx.stream));
          ctx.synchronize();
          for (int r = 0; r < bc; ++r) {
            on_logits(ingine::model::LogitStep{
                global_rows[b0 + static_cast<std::size_t>(r)],
                std::span<const float>(host_logits_stage.data() + static_cast<std::size_t>(r) * V,
                                       static_cast<std::size_t>(V))});
          }
        }
      }

      sess.committed_tokens_.insert(sess.committed_tokens_.end(), token_ids.begin() + off,
                                    token_ids.begin() + off + M);
    }

    ctx.synchronize();
    if (prefill_ms_out) {
      *prefill_ms_out += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
  }
};

Qwen38DenseSession::Qwen38DenseSession(Qwen38DenseBackend& owner)
    : owner_(owner), state_(std::make_unique<StateImpl>()) {
  const auto& c = owner_.config();
  const int max_ctx = owner_.impl_->opts.max_ctx;
  const int cd = c.linear_conv_dim();
  const std::size_t conv_elems = static_cast<std::size_t>(cd) * static_cast<std::size_t>(c.linear_conv_kernel_dim);
  const std::size_t rec_elems = static_cast<std::size_t>(c.linear_num_value_heads) *
                                static_cast<std::size_t>(c.linear_key_head_dim) *
                                static_cast<std::size_t>(c.linear_value_head_dim);
  const std::size_t kv_elems = static_cast<std::size_t>(max_ctx) * static_cast<std::size_t>(c.full_kv_proj_out_dim());

  owner_.impl_->ctx.bind_to_current_thread();
  state_->conv_states.resize(static_cast<std::size_t>(c.num_linear_layers));
  state_->rec_states.resize(static_cast<std::size_t>(c.num_linear_layers));
  for (int i = 0; i < c.num_linear_layers; ++i) {
    state_->conv_states[static_cast<std::size_t>(i)].allocate(conv_elems);
    state_->rec_states[static_cast<std::size_t>(i)].allocate(rec_elems);
    state_->total_state_bytes += (conv_elems + rec_elems) * sizeof(float);
  }
  state_->k_caches.resize(static_cast<std::size_t>(c.num_full_layers));
  state_->v_caches.resize(static_cast<std::size_t>(c.num_full_layers));
  for (int i = 0; i < c.num_full_layers; ++i) {
    state_->k_caches[static_cast<std::size_t>(i)].allocate(kv_elems);
    state_->v_caches[static_cast<std::size_t>(i)].allocate(kv_elems);
    state_->total_state_bytes += kv_elems * sizeof(__nv_bfloat16) * 2ull;
  }
  reset();
}

Qwen38DenseSession::~Qwen38DenseSession() = default;

void Qwen38DenseSession::reset() {
  owner_.impl_->ctx.bind_to_current_thread();
  for (auto& b : state_->conv_states) {
    if (b.count > 0) CUDA_CHECK(cudaMemsetAsync(b.ptr, 0, b.count * sizeof(float), owner_.impl_->ctx.stream));
  }
  for (auto& b : state_->rec_states) {
    if (b.count > 0) CUDA_CHECK(cudaMemsetAsync(b.ptr, 0, b.count * sizeof(float), owner_.impl_->ctx.stream));
  }
  owner_.impl_->ctx.synchronize();
  committed_tokens_.clear();
}

std::size_t Qwen38DenseSession::state_bytes() const noexcept { return state_->total_state_bytes; }

const ingine::model::BackendStateIdentity& Qwen38DenseSession::identity() const noexcept {
  return owner_.state_identity();
}

Qwen38GemmParityReport Qwen38DenseBackend::self_check_gemm_parity(int device_id) {
  DeviceContext ctx(device_id);
  ctx.bind_to_current_thread();

  constexpr int M = 17, N = 256, K = 5120;
  std::mt19937 rng(20261125u);
  std::normal_distribution<float> dist(0.0f, 0.25f);
  std::vector<float> x_f32(static_cast<std::size_t>(M * K));
  for (float& v : x_f32) v = dist(rng);

  std::vector<std::uint8_t> nv_p(static_cast<std::size_t>(N * (K / 2)));
  std::vector<std::uint8_t> nv_s(static_cast<std::size_t>(N * (K / 16)));
  for (auto& b : nv_p) b = static_cast<std::uint8_t>(rng() & 0xffu);
  for (auto& s : nv_s) s = static_cast<std::uint8_t>(0x38u + (rng() % 20));
  constexpr float kWs2 = 0.015f;

  std::vector<std::uint8_t> fp8_w(static_cast<std::size_t>(N * K));
  for (auto& b : fp8_w) b = quantize_e4m3fn(static_cast<float>(static_cast<int>(rng() % 200) - 100) * 0.01f);
  constexpr float kFp8Ws = 0.025f;
  constexpr float kFp8In = 0.125f;

  std::vector<std::uint16_t> bf_w(static_cast<std::size_t>(N * K));
  for (auto& b : bf_w) b = f32_to_bf16(dist(rng));

  DevBuf<float> d_x(x_f32.size()), d_xq(x_f32.size()), d_out_ref(static_cast<std::size_t>(M * N)),
      d_out_batched(static_cast<std::size_t>(M * N)), d_out_m1_row0(N), d_out_m1_row16(N);
  DevBuf<std::uint8_t> d_nv_p(nv_p.size()), d_nv_s(nv_s.size()), d_fp8_w(fp8_w.size());
  DevBuf<std::uint16_t> d_bf_w(bf_w.size());

  CUDA_CHECK(cudaMemcpy(d_x.ptr, x_f32.data(), x_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nv_p.ptr, nv_p.data(), nv_p.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_nv_s.ptr, nv_s.data(), nv_s.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fp8_w.ptr, fp8_w.data(), fp8_w.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bf_w.ptr, bf_w.data(), bf_w.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));

  auto dev_max_abs = [&](const float* a_dev, const float* b_dev, std::size_t count) -> float {
    std::vector<float> a(count), b(count);
    CUDA_CHECK(cudaMemcpy(a.data(), a_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), b_dev, count * sizeof(float), cudaMemcpyDeviceToHost));
    float mx = 0.0f;
    for (std::size_t i = 0; i < count; ++i) {
      if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<float>::infinity();
      mx = std::max(mx, std::fabs(a[i] - b[i]));
    }
    return mx;
  };

  Qwen38GemmParityReport rep;
  float max_batch_inv = 0.0f;

  // 1. NVFP4 batched vs Gate 1 reference F32 + M=1 invariance check on rows 0 and 16.
  qwen38_cuda_nvfp4_linear_f32(d_x.ptr, d_nv_p.ptr, d_nv_s.ptr, kWs2, M, N, K, d_out_ref.ptr, ctx.stream);
  qwen38_cuda_nvfp4_linear_f32_batched(d_x.ptr, d_nv_p.ptr, d_nv_s.ptr, kWs2, M, N, K, d_out_batched.ptr, ctx.stream);
  qwen38_cuda_nvfp4_linear_f32_batched(d_x.ptr, d_nv_p.ptr, d_nv_s.ptr, kWs2, 1, N, K, d_out_m1_row0.ptr, ctx.stream);
  qwen38_cuda_nvfp4_linear_f32_batched(d_x.ptr + static_cast<std::size_t>(16 * K), d_nv_p.ptr, d_nv_s.ptr, kWs2, 1, N,
                                       K, d_out_m1_row16.ptr, ctx.stream);
  ctx.synchronize();
  rep.nvfp4_max_abs = dev_max_abs(d_out_ref.ptr, d_out_batched.ptr, static_cast<std::size_t>(M * N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row0.ptr, d_out_batched.ptr, N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row16.ptr, d_out_batched.ptr + 16 * N, N));

  // 2. Calibrated FP8 batched vs Gate 1 reference F32 + M=1 invariance check on rows 0 and 16.
  qwen38_cuda_fp8_linear_f32(d_x.ptr, d_fp8_w.ptr, kFp8Ws, kFp8In, M, N, K, d_out_ref.ptr, ctx.stream);
  qwen38_cuda_quant_fp8_act_f32(d_x.ptr, kFp8In, static_cast<std::size_t>(M * K), d_xq.ptr, ctx.stream);
  qwen38_cuda_fp8_linear_f32_batched(d_xq.ptr, d_fp8_w.ptr, kFp8Ws, 0.0f, M, N, K, d_out_batched.ptr, ctx.stream);
  qwen38_cuda_fp8_linear_f32_batched(d_xq.ptr, d_fp8_w.ptr, kFp8Ws, 0.0f, 1, N, K, d_out_m1_row0.ptr, ctx.stream);
  qwen38_cuda_fp8_linear_f32_batched(d_xq.ptr + static_cast<std::size_t>(16 * K), d_fp8_w.ptr, kFp8Ws, 0.0f, 1, N, K,
                                     d_out_m1_row16.ptr, ctx.stream);
  ctx.synchronize();
  rep.fp8_cal_max_abs = dev_max_abs(d_out_ref.ptr, d_out_batched.ptr, static_cast<std::size_t>(M * N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row0.ptr, d_out_batched.ptr, N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row16.ptr, d_out_batched.ptr + 16 * N, N));

  // 3. BF16 batched vs Gate 1 reference F32 + M=1 invariance check on rows 0 and 16.
  const auto* d_bf_w_ptr = reinterpret_cast<const __nv_bfloat16*>(d_bf_w.ptr);
  qwen38_cuda_bf16_linear_f32(d_x.ptr, d_bf_w_ptr, M, N, K, d_out_ref.ptr, ctx.stream);
  qwen38_cuda_bf16_linear_f32_batched(d_x.ptr, d_bf_w_ptr, M, N, K, d_out_batched.ptr, ctx.stream);
  qwen38_cuda_bf16_linear_f32_batched(d_x.ptr, d_bf_w_ptr, 1, N, K, d_out_m1_row0.ptr, ctx.stream);
  qwen38_cuda_bf16_linear_f32_batched(d_x.ptr + static_cast<std::size_t>(16 * K), d_bf_w_ptr, 1, N, K,
                                      d_out_m1_row16.ptr, ctx.stream);
  ctx.synchronize();
  rep.bf16_max_abs = dev_max_abs(d_out_ref.ptr, d_out_batched.ptr, static_cast<std::size_t>(M * N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row0.ptr, d_out_batched.ptr, N));
  max_batch_inv = std::max(max_batch_inv, dev_max_abs(d_out_m1_row16.ptr, d_out_batched.ptr + 16 * N, N));

  // 4. GDN exact recurrent prefill invariance from non-zero initial state:
  // whole (273, 64 KiB/head smem across 64/256 boundaries), ragged+boundary splits {17,47,192,17},
  // forced same-order global fallback {64,192,17}, 273x M=1 seq prefill, and 273x Gate 1 recurrent_step.
  constexpr int kGdnSeq = 273, kGdnKh = 2, kGdnVh = 6, kGdnKd = 128, kGdnVd = 128;
  constexpr int kGdnKdTot = kGdnKh * kGdnKd;
  constexpr int kGdnVdTot = kGdnVh * kGdnVd;
  constexpr int kGdnCd = 2 * kGdnKdTot + kGdnVdTot;
  constexpr std::size_t kGdnStateElems =
      static_cast<std::size_t>(kGdnVh) * static_cast<std::size_t>(kGdnKd) * static_cast<std::size_t>(kGdnVd);

  std::vector<float> h_gdn_conv(static_cast<std::size_t>(kGdnSeq * kGdnCd));
  std::vector<float> h_gdn_z(static_cast<std::size_t>(kGdnSeq * kGdnVdTot));
  std::vector<float> h_gdn_a(static_cast<std::size_t>(kGdnSeq * kGdnVh));
  std::vector<float> h_gdn_b(static_cast<std::size_t>(kGdnSeq * kGdnVh));
  std::vector<float> h_gdn_alog(static_cast<std::size_t>(kGdnVh));
  std::vector<float> h_gdn_dtb(static_cast<std::size_t>(kGdnVh));
  std::vector<float> h_gdn_nw(static_cast<std::size_t>(kGdnVd));
  std::vector<float> h_gdn_st_init(kGdnStateElems);
  for (float& v : h_gdn_conv) v = dist(rng);
  for (float& v : h_gdn_z) v = dist(rng);
  for (float& v : h_gdn_a) v = dist(rng) * 4.0f;
  for (float& v : h_gdn_b) v = dist(rng) * 4.0f;
  for (int i = 0; i < kGdnVh; ++i) {
    h_gdn_alog[static_cast<std::size_t>(i)] = -2.0f + 1.3f * static_cast<float>(i);
    h_gdn_dtb[static_cast<std::size_t>(i)] = -3.0f + 2.6f * static_cast<float>(i);
  }
  for (float& v : h_gdn_nw) v = 1.0f + dist(rng) * 0.2f;
  for (float& v : h_gdn_st_init) v = dist(rng) * 0.1f;

  DevBuf<float> d_gdn_conv(h_gdn_conv.size()), d_gdn_z(h_gdn_z.size()), d_gdn_a(h_gdn_a.size()),
      d_gdn_b(h_gdn_b.size()), d_gdn_alog(h_gdn_alog.size()), d_gdn_dtb(h_gdn_dtb.size()), d_gdn_nw(h_gdn_nw.size()),
      d_gdn_st_whole(kGdnStateElems), d_gdn_st_split(kGdnStateElems), d_gdn_st_fb(kGdnStateElems),
      d_gdn_st_seq1(kGdnStateElems), d_gdn_st_gate1(kGdnStateElems),
      d_gdn_scratch(static_cast<std::size_t>(2 * kGdnSeq * kGdnKdTot)), d_gdn_out_whole(h_gdn_z.size()),
      d_gdn_out_split(h_gdn_z.size()), d_gdn_out_fb(h_gdn_z.size()), d_gdn_out_seq1(h_gdn_z.size()),
      d_gdn_out_gate1(h_gdn_z.size());

  CUDA_CHECK(cudaMemcpy(d_gdn_conv.ptr, h_gdn_conv.data(), h_gdn_conv.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_z.ptr, h_gdn_z.data(), h_gdn_z.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_a.ptr, h_gdn_a.data(), h_gdn_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_b.ptr, h_gdn_b.data(), h_gdn_b.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_alog.ptr, h_gdn_alog.data(), h_gdn_alog.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_dtb.ptr, h_gdn_dtb.data(), h_gdn_dtb.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gdn_nw.ptr, h_gdn_nw.data(), h_gdn_nw.size() * sizeof(float), cudaMemcpyHostToDevice));
  for (float* st_dst :
       {d_gdn_st_whole.ptr, d_gdn_st_split.ptr, d_gdn_st_fb.ptr, d_gdn_st_seq1.ptr, d_gdn_st_gate1.ptr}) {
    CUDA_CHECK(cudaMemcpy(st_dst, h_gdn_st_init.data(), kGdnStateElems * sizeof(float), cudaMemcpyHostToDevice));
  }

  // Whole sequence across 64/256 boundaries.
  qwen38_cuda_gdn_recurrent_prefill_f32(d_gdn_conv.ptr, d_gdn_z.ptr, d_gdn_a.ptr, d_gdn_b.ptr, d_gdn_alog.ptr,
                                        d_gdn_dtb.ptr, d_gdn_nw.ptr, 1e-6f, kGdnSeq, kGdnKh, kGdnVh, kGdnKd, kGdnVd,
                                        d_gdn_st_whole.ptr, d_gdn_scratch.ptr, d_gdn_out_whole.ptr, ctx.stream);

  // Ragged + 64/256 boundary splits: {17, 47, 192, 17} -> cumulative [17, 64, 256, 273].
  {
    int off = 0;
    for (int slen : {17, 47, 192, 17}) {
      qwen38_cuda_gdn_recurrent_prefill_f32(
          d_gdn_conv.ptr + static_cast<std::size_t>(off * kGdnCd),
          d_gdn_z.ptr + static_cast<std::size_t>(off * kGdnVdTot),
          d_gdn_a.ptr + static_cast<std::size_t>(off * kGdnVh),
          d_gdn_b.ptr + static_cast<std::size_t>(off * kGdnVh), d_gdn_alog.ptr, d_gdn_dtb.ptr, d_gdn_nw.ptr, 1e-6f,
          slen, kGdnKh, kGdnVh, kGdnKd, kGdnVd, d_gdn_st_split.ptr, d_gdn_scratch.ptr,
          d_gdn_out_split.ptr + static_cast<std::size_t>(off * kGdnVdTot), ctx.stream);
      off += slen;
    }
  }

  // Forced same-order global fallback splits {64, 192, 17}.
  {
    int off = 0;
    for (int slen : {64, 192, 17}) {
      qwen38_cuda_gdn_recurrent_prefill_f32(
          d_gdn_conv.ptr + static_cast<std::size_t>(off * kGdnCd),
          d_gdn_z.ptr + static_cast<std::size_t>(off * kGdnVdTot),
          d_gdn_a.ptr + static_cast<std::size_t>(off * kGdnVh),
          d_gdn_b.ptr + static_cast<std::size_t>(off * kGdnVh), d_gdn_alog.ptr, d_gdn_dtb.ptr, d_gdn_nw.ptr, 1e-6f,
          slen, kGdnKh, kGdnVh, kGdnKd, kGdnVd, d_gdn_st_fb.ptr, d_gdn_scratch.ptr,
          d_gdn_out_fb.ptr + static_cast<std::size_t>(off * kGdnVdTot), ctx.stream, true);
      off += slen;
    }
  }

  // M=1 sequential prefill + Gate 1 recurrent_step reference.
  for (int t = 0; t < kGdnSeq; ++t) {
    const float* conv_t = d_gdn_conv.ptr + static_cast<std::size_t>(t * kGdnCd);
    const float* z_t = d_gdn_z.ptr + static_cast<std::size_t>(t * kGdnVdTot);
    const float* a_t = d_gdn_a.ptr + static_cast<std::size_t>(t * kGdnVh);
    const float* b_t = d_gdn_b.ptr + static_cast<std::size_t>(t * kGdnVh);
    qwen38_cuda_gdn_recurrent_prefill_f32(conv_t, z_t, a_t, b_t, d_gdn_alog.ptr, d_gdn_dtb.ptr, d_gdn_nw.ptr, 1e-6f, 1,
                                          kGdnKh, kGdnVh, kGdnKd, kGdnVd, d_gdn_st_seq1.ptr, d_gdn_scratch.ptr,
                                          d_gdn_out_seq1.ptr + static_cast<std::size_t>(t * kGdnVdTot), ctx.stream);
    qwen38_cuda_gdn_recurrent_step_f32(conv_t, conv_t + kGdnKdTot, conv_t + 2 * kGdnKdTot, z_t, a_t, b_t,
                                       d_gdn_alog.ptr, d_gdn_dtb.ptr, d_gdn_nw.ptr, 1e-6f, kGdnKh, kGdnVh, kGdnKd,
                                       kGdnVd, d_gdn_st_gate1.ptr, d_gdn_scratch.ptr,
                                       d_gdn_out_gate1.ptr + static_cast<std::size_t>(t * kGdnVdTot), ctx.stream);
  }
  ctx.synchronize();

  rep.batch_invariance_max_abs = max_batch_inv;
  float gdn_bi = 0.0f;
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_out_whole.ptr, d_gdn_out_seq1.ptr, h_gdn_z.size()));
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_st_whole.ptr, d_gdn_st_seq1.ptr, kGdnStateElems));
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_out_split.ptr, d_gdn_out_seq1.ptr, h_gdn_z.size()));
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_st_split.ptr, d_gdn_st_seq1.ptr, kGdnStateElems));
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_out_fb.ptr, d_gdn_out_seq1.ptr, h_gdn_z.size()));
  gdn_bi = std::max(gdn_bi, dev_max_abs(d_gdn_st_fb.ptr, d_gdn_st_seq1.ptr, kGdnStateElems));
  rep.gdn_batch_invariance_max_abs = gdn_bi;
  rep.gdn_prefill_vs_step_max_abs = std::max(dev_max_abs(d_gdn_out_whole.ptr, d_gdn_out_gate1.ptr, h_gdn_z.size()),
                                             dev_max_abs(d_gdn_st_whole.ptr, d_gdn_st_gate1.ptr, kGdnStateElems));
  rep.passed = (rep.nvfp4_max_abs <= 1e-4f && rep.fp8_cal_max_abs <= 1e-4f && rep.bf16_max_abs <= 1e-4f &&
                rep.batch_invariance_max_abs == 0.0f && rep.gdn_prefill_vs_step_max_abs <= 1e-6f &&
                rep.gdn_batch_invariance_max_abs == 0.0f);
  return rep;
}

Qwen38DenseBackend::Qwen38DenseBackend(const Qwen38DenseLoadOptions& options)
    : impl_(std::make_unique<Impl>(options.device_id)) {
  if (options.max_ctx <= 0 || options.prefill_chunk <= 0) {
    throw std::invalid_argument("Qwen38DenseBackend max_ctx and prefill_chunk must be > 0");
  }
  impl_->opts = options;

  const auto t_sha0 = std::chrono::steady_clock::now();
  if (options.verify_pins) {
    const auto spec = options.expected_pins.has_value() ? *options.expected_pins : qwen38_27b_nvfp4_pin_spec();
    impl_->manifest = Qwen38Manifest::inspect_pinned(options.model_dir, spec, options.revision);
  } else {
    impl_->manifest = Qwen38Manifest::inspect(options.model_dir, false, options.revision);
  }
  const double sha_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_sha0).count();

  const auto& m = impl_->manifest;
  const auto& c = m.config;
  c.validate();
  if (options.max_ctx > c.max_position_embeddings) {
    throw std::invalid_argument("max_ctx exceeds model max_position_embeddings");
  }

  ingine::model::RuntimeOptions ropt;
  ropt.model_dir = options.model_dir;
  ropt.max_ctx = options.max_ctx;
  ropt.prefill_chunk = options.prefill_chunk;
  ropt.dense_fp8 = (c.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4);
  ropt.max_batch = 1;
  profile_ = make_qwen38_worker_profile(m, ropt, "dense_bf16", "dense");
  profile_.state_identity.act_dtype = ingine::model::ActivationDtype::Fp32;
  profile_.state_identity.numeric_abi =
      (c.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4)
          ? "act=f32;attn=fp8_e4m3_cal;mlp=nvfp4_w4a32;gdn=f32;kv=bf16"
          : "act=f32;attn=bf16;mlp=bf16;gdn=f32;kv=bf16";
  profile_.state_identity.kv_format = "gdn_f32_conv_f32_kv_bf16";

  load_stats_.pins_verified =
      options.verify_pins && m.checkpoint.verification_level == ingine::model::VerificationLevel::PinnedVerified &&
      m.checkpoint.shard_digests_verified;
  load_stats_.cold_manifest_sha_ms = sha_ms;

  std::size_t vram_free_before = 0, vram_total = 0;
  CUDA_CHECK(cudaMemGetInfo(&vram_free_before, &vram_total));

  const auto t_ld0 = std::chrono::steady_clock::now();

  // Plan device arena + small F32/scalar staging across every active text tensor.
  std::uint64_t dev_arena_bytes = 0;
  std::unordered_map<std::string, CopySpec> copy_specs;
  std::deque<HostScalarEntry> scalar_entries;
  std::deque<HostF32VecEntry> f32_entries;

  auto plan_dev = [&](const std::string& name) -> std::uint64_t {
    const auto it = m.active_tensors.find(name);
    if (it == m.active_tensors.end()) throw std::invalid_argument("missing active tensor: " + name);
    const std::uint64_t off = align_up(dev_arena_bytes);
    dev_arena_bytes = off + it->second.byte_size();
    copy_specs.emplace(name, CopySpec{&it->second, true, off, nullptr});
    return off;
  };

  auto plan_scalar = [&](const std::string& name, float* dst_field) {
    const auto it = m.active_tensors.find(name);
    if (it == m.active_tensors.end()) throw std::invalid_argument("missing scalar tensor: " + name);
    scalar_entries.push_back(HostScalarEntry{dst_field, it->second.dtype, {}});
    copy_specs.emplace(name, CopySpec{&it->second, false, 0, &scalar_entries.back().raw});
  };

  std::size_t f32_total_elems = 0;
  auto plan_f32_vec = [&](const std::string& name, std::size_t elems) -> std::size_t {
    const auto it = m.active_tensors.find(name);
    if (it == m.active_tensors.end()) throw std::invalid_argument("missing 1D GDN tensor: " + name);
    const std::size_t off = f32_total_elems;
    f32_total_elems += elems;
    f32_entries.push_back(HostF32VecEntry{off, elems, it->second.dtype, name, {}});
    copy_specs.emplace(name, CopySpec{&it->second, false, 0, &f32_entries.back().raw});
    return off;
  };

  struct PlannedProj {
    DevProj* dst = nullptr;
    std::uint64_t main_off = 0;
    std::uint64_t scale_off = 0;
  };
  std::vector<PlannedProj> planned_projs;
  planned_projs.reserve(static_cast<std::size_t>(c.num_hidden_layers) * 9u + 1u);

  auto plan_proj = [&](const std::string& prefix, int out_f, int in_f, Qwen38LinearQuantKind q, DevProj* dst) {
    dst->out_features = out_f;
    dst->in_features = in_f;
    dst->quant = q;
    PlannedProj pp;
    pp.dst = dst;
    pp.main_off = plan_dev(prefix + ".weight");
    if (q == Qwen38LinearQuantKind::Fp8E4M3) {
      plan_scalar(prefix + ".weight_scale", &dst->weight_scale);
      plan_scalar(prefix + ".input_scale", &dst->input_scale);
    } else if (q == Qwen38LinearQuantKind::Nvfp4) {
      pp.scale_off = plan_dev(prefix + ".weight_scale");
      plan_scalar(prefix + ".weight_scale_2", &dst->weight_scale_2);
      plan_scalar(prefix + ".input_scale", &dst->input_scale);
    }
    planned_projs.push_back(pp);
  };

  const std::uint64_t embed_off = plan_dev("model.language_model.embed_tokens.weight");
  const std::uint64_t norm_off = plan_dev("model.language_model.norm.weight");
  if (!c.tie_word_embeddings) {
    plan_proj("lm_head", c.vocab_size, c.hidden_size, c.lm_head_quant(), &impl_->lm_head);
  }

  impl_->layers.resize(static_cast<std::size_t>(c.num_hidden_layers));
  struct LayerOffsets {
    std::uint64_t in_norm = 0, post_norm = 0, q_norm = 0, k_norm = 0;
    std::size_t a_log = 0, dt_bias = 0, conv1d = 0, lin_norm = 0;
  };
  std::vector<LayerOffsets> layer_offs(static_cast<std::size_t>(c.num_hidden_layers));

  for (int li = 0; li < c.num_hidden_layers; ++li) {
    auto& l = impl_->layers[static_cast<std::size_t>(li)];
    auto& lo = layer_offs[static_cast<std::size_t>(li)];
    l.kind = c.layer_types[static_cast<std::size_t>(li)];
    const std::string lp = "model.language_model.layers." + std::to_string(li);
    lo.in_norm = plan_dev(lp + ".input_layernorm.weight");
    lo.post_norm = plan_dev(lp + ".post_attention_layernorm.weight");
    if (l.kind == Qwen38LayerKind::LinearAttention) {
      const std::string attn = lp + ".linear_attn";
      lo.a_log = plan_f32_vec(attn + ".A_log", static_cast<std::size_t>(c.linear_num_value_heads));
      lo.dt_bias = plan_f32_vec(attn + ".dt_bias", static_cast<std::size_t>(c.linear_num_value_heads));
      lo.conv1d = plan_f32_vec(
          attn + ".conv1d.weight",
          static_cast<std::size_t>(c.linear_conv_dim()) * static_cast<std::size_t>(c.linear_conv_kernel_dim));
      lo.lin_norm = plan_f32_vec(attn + ".norm.weight", static_cast<std::size_t>(c.linear_value_head_dim));
      plan_proj(attn + ".in_proj_a", c.linear_num_value_heads, c.hidden_size, Qwen38LinearQuantKind::Bf16, &l.in_proj_a);
      plan_proj(attn + ".in_proj_b", c.linear_num_value_heads, c.hidden_size, Qwen38LinearQuantKind::Bf16, &l.in_proj_b);
      plan_proj(attn + ".in_proj_qkv", c.linear_conv_dim(), c.hidden_size, c.linear_attn_quant(), &l.in_proj_qkv);
      plan_proj(attn + ".in_proj_z", c.linear_value_dim(), c.hidden_size, c.linear_attn_quant(), &l.in_proj_z);
      plan_proj(attn + ".out_proj", c.hidden_size, c.linear_value_dim(), c.linear_attn_quant(), &l.out_proj);
    } else {
      const std::string attn = lp + ".self_attn";
      lo.q_norm = plan_dev(attn + ".q_norm.weight");
      lo.k_norm = plan_dev(attn + ".k_norm.weight");
      plan_proj(attn + ".q_proj", c.full_q_proj_out_dim(), c.hidden_size, c.full_attn_quant(), &l.q_proj);
      plan_proj(attn + ".k_proj", c.full_kv_proj_out_dim(), c.hidden_size, c.full_attn_quant(), &l.k_proj);
      plan_proj(attn + ".v_proj", c.full_kv_proj_out_dim(), c.hidden_size, c.full_attn_quant(), &l.v_proj);
      plan_proj(attn + ".o_proj", c.hidden_size, c.num_attention_heads * c.head_dim, c.full_attn_quant(), &l.o_proj);
    }
    plan_proj(lp + ".mlp.gate_proj", c.intermediate_size, c.hidden_size, c.mlp_quant(), &l.gate_proj);
    plan_proj(lp + ".mlp.up_proj", c.intermediate_size, c.hidden_size, c.mlp_quant(), &l.up_proj);
    plan_proj(lp + ".mlp.down_proj", c.hidden_size, c.intermediate_size, c.mlp_quant(), &l.down_proj);
  }

  impl_->weights_arena.allocate(dev_arena_bytes);
  impl_->f32_arena.allocate(f32_total_elems);

  // Group CopySpecs by shard and stream directly to device / tiny staging vectors.
  std::unordered_map<std::string, std::vector<const CopySpec*>> by_shard;
  for (const auto& [name, spec] : copy_specs) {
    by_shard[spec.info->shard_file].push_back(&spec);
  }

  const bool is_pinned = load_stats_.pins_verified;
  for (const std::string& shard : m.shard_files) {
    const auto shard_it = by_shard.find(shard);
    if (shard_it == by_shard.end() || shard_it->second.empty()) {
      if (is_pinned) {
        (void)m.stream_verified_file(options.model_dir, shard, {}, {}, {}, 4ull << 20);
      }
      continue;
    }

    const auto& specs = shard_it->second;
    std::array<std::uint8_t, 8> hdr_bytes{};
    std::uint8_t hdr_have = 0;
    bool targets_built = false;
    std::vector<ActiveTarget> targets;
    std::size_t next_target = 0;

    auto on_chunk = [&](std::uint64_t off, std::span<const std::uint8_t> chunk) {
      if (off < 8) {
        const std::uint64_t take = std::min<std::uint64_t>(8 - off, chunk.size());
        std::memcpy(hdr_bytes.data() + off, chunk.data(), static_cast<std::size_t>(take));
        hdr_have = static_cast<std::uint8_t>(std::max<std::uint64_t>(hdr_have, off + take));
      }
      if (!targets_built && hdr_have == 8) {
        const std::uint64_t data_base = 8ull + parse_header_len_le(hdr_bytes);
        targets.reserve(specs.size());
        for (const CopySpec* sp : specs) {
          if (!sp->to_device) sp->host_dst->assign(sp->info->byte_size(), 0u);
          targets.push_back(ActiveTarget{data_base + sp->info->data_offset_begin,
                                         data_base + sp->info->data_offset_end, sp->to_device, sp->dev_offset,
                                         sp->host_dst});
        }
        std::sort(targets.begin(), targets.end(),
                  [](const ActiveTarget& a, const ActiveTarget& b) { return a.file_begin < b.file_begin; });
        targets_built = true;
      }
      if (!targets_built) return;
      const std::uint64_t c0 = off;
      const std::uint64_t c1 = off + chunk.size();
      while (next_target < targets.size() && targets[next_target].file_end <= c0) ++next_target;
      for (std::size_t ti = next_target; ti < targets.size(); ++ti) {
        const ActiveTarget& tg = targets[ti];
        if (tg.file_begin >= c1) break;
        const std::uint64_t s0 = std::max(c0, tg.file_begin);
        const std::uint64_t s1 = std::min(c1, tg.file_end);
        const std::size_t n = static_cast<std::size_t>(s1 - s0);
        const std::uint8_t* src = chunk.data() + static_cast<std::size_t>(s0 - c0);
        const std::uint64_t dst_off = s0 - tg.file_begin;
        if (tg.to_device) {
          CUDA_CHECK(cudaMemcpy(impl_->weights_arena.ptr + tg.dev_offset + dst_off, src, n, cudaMemcpyHostToDevice));
        } else {
          std::memcpy(tg.host_dst->data() + dst_off, src, n);
        }
      }
    };

    auto on_abort = [&]() {
      impl_->weights_arena.free();
      impl_->f32_arena.free();
    };

    if (is_pinned) {
      (void)m.stream_verified_file(options.model_dir, shard, on_chunk, {}, on_abort, 4ull << 20);
    } else {
      stream_unpinned_shard(m, options.model_dir, shard, on_chunk, on_abort);
    }
  }

  // Decode staged scalar scales and 1D F32 GDN arrays (ONLY after all shards verified/loaded).
  for (const auto& se : scalar_entries) {
    *se.target_field = decode_scalar_bytes(se.raw, se.dtype, "scalar");
  }
  std::vector<float> host_f32_all(f32_total_elems, 0.0f);
  for (const auto& fe : f32_entries) {
    decode_f32_vec_bytes(fe.raw, fe.dtype, fe.elems, fe.name, host_f32_all, fe.f32_offset);
  }
  if (f32_total_elems > 0) {
    CUDA_CHECK(cudaMemcpy(impl_->f32_arena.ptr, host_f32_all.data(), f32_total_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
  }

  // Resolve device pointers inside arenas.
  const std::uint8_t* wbase = impl_->weights_arena.ptr;
  const float* fbase = impl_->f32_arena.ptr;
  impl_->embed_tokens = reinterpret_cast<const __nv_bfloat16*>(wbase + embed_off);
  impl_->final_norm = reinterpret_cast<const __nv_bfloat16*>(wbase + norm_off);
  for (const auto& pp : planned_projs) {
    if (pp.dst->quant == Qwen38LinearQuantKind::Bf16) {
      pp.dst->w_bf16 = reinterpret_cast<const __nv_bfloat16*>(wbase + pp.main_off);
    } else if (pp.dst->quant == Qwen38LinearQuantKind::Fp8E4M3) {
      pp.dst->w_u8 = wbase + pp.main_off;
    } else {
      pp.dst->w_u8 = wbase + pp.main_off;
      pp.dst->w_scales = wbase + pp.scale_off;
    }
  }
  if (c.tie_word_embeddings) {
    impl_->lm_head.out_features = c.vocab_size;
    impl_->lm_head.in_features = c.hidden_size;
    impl_->lm_head.quant = Qwen38LinearQuantKind::Bf16;
    impl_->lm_head.w_bf16 = impl_->embed_tokens;
  }

  for (int li = 0; li < c.num_hidden_layers; ++li) {
    auto& l = impl_->layers[static_cast<std::size_t>(li)];
    const auto& lo = layer_offs[static_cast<std::size_t>(li)];
    l.input_layernorm = reinterpret_cast<const __nv_bfloat16*>(wbase + lo.in_norm);
    l.post_attention_layernorm = reinterpret_cast<const __nv_bfloat16*>(wbase + lo.post_norm);
    if (l.kind == Qwen38LayerKind::LinearAttention) {
      l.a_log = fbase + lo.a_log;
      l.dt_bias = fbase + lo.dt_bias;
      l.conv1d_weight = fbase + lo.conv1d;
      l.linear_norm_weight = fbase + lo.lin_norm;
    } else {
      l.q_norm = reinterpret_cast<const __nv_bfloat16*>(wbase + lo.q_norm);
      l.k_norm = reinterpret_cast<const __nv_bfloat16*>(wbase + lo.k_norm);
    }
  }

  // Allocate reusable execution scratch + default session.
  const int P = options.prefill_chunk;
  const int H = c.hidden_size;
  const int I = c.intermediate_size;
  const int V = c.vocab_size;
  const int cd = c.linear_conv_dim();
  const int kd = c.linear_key_dim();
  const int vd = c.linear_value_dim();
  const int nhv = c.linear_num_value_heads;
  const int qg_dim = c.full_q_proj_out_dim();
  const int kv_dim = c.full_kv_proj_out_dim();
  const int attn_dim = c.num_attention_heads * c.head_dim;

  const std::size_t max_k_elems =
      static_cast<std::size_t>(P) * static_cast<std::size_t>(std::max({H, I, vd, attn_dim}));

  impl_->act_quant_scratch.allocate(max_k_elems);
  impl_->token_ids_dev.allocate(static_cast<std::size_t>(P));
  impl_->gather_rows_dev.allocate(static_cast<std::size_t>(Impl::kLogitBatchMax));
  impl_->hidden.allocate(static_cast<std::size_t>(P * H));
  impl_->normed.allocate(static_cast<std::size_t>(P * H));
  impl_->residual.allocate(static_cast<std::size_t>(P * H));
  impl_->qkv.allocate(static_cast<std::size_t>(P * cd));
  impl_->conv_out.allocate(static_cast<std::size_t>(P * cd));
  impl_->z_buf.allocate(static_cast<std::size_t>(P * vd));
  impl_->a_buf.allocate(static_cast<std::size_t>(P * nhv));
  impl_->b_buf.allocate(static_cast<std::size_t>(P * nhv));
  impl_->core_out.allocate(static_cast<std::size_t>(P * vd));
  impl_->gdn_scratch.allocate(static_cast<std::size_t>(P) * static_cast<std::size_t>(kd) * 2ull);

  impl_->q_gate.allocate(static_cast<std::size_t>(P * qg_dim));
  impl_->k_raw.allocate(static_cast<std::size_t>(P * kv_dim));
  impl_->v_raw.allocate(static_cast<std::size_t>(P * kv_dim));
  impl_->q_tmp.allocate(static_cast<std::size_t>(P * attn_dim));
  impl_->gate_tmp.allocate(static_cast<std::size_t>(P * attn_dim));
  impl_->attn_out.allocate(static_cast<std::size_t>(P * attn_dim));
  impl_->mlp_gate.allocate(static_cast<std::size_t>(P * I));
  impl_->mlp_up.allocate(static_cast<std::size_t>(P * I));
  impl_->cap_hidden.allocate(static_cast<std::size_t>(Impl::kLogitBatchMax * H));
  impl_->cap_normed.allocate(static_cast<std::size_t>(Impl::kLogitBatchMax * H));
  impl_->logits_f32.allocate(static_cast<std::size_t>(Impl::kLogitBatchMax) * static_cast<std::size_t>(V));
  impl_->host_logits_stage.resize(static_cast<std::size_t>(Impl::kLogitBatchMax) * static_cast<std::size_t>(V));
  impl_->decode_single_tok.resize(1);

  impl_->scratch_bytes =
      (impl_->act_quant_scratch.count + impl_->hidden.count + impl_->normed.count + impl_->residual.count +
       impl_->qkv.count + impl_->conv_out.count + impl_->z_buf.count + impl_->a_buf.count + impl_->b_buf.count +
       impl_->core_out.count + impl_->q_gate.count + impl_->k_raw.count + impl_->v_raw.count + impl_->q_tmp.count +
       impl_->gate_tmp.count + impl_->attn_out.count + impl_->mlp_gate.count + impl_->mlp_up.count +
       impl_->cap_hidden.count + impl_->cap_normed.count + impl_->gdn_scratch.count + impl_->logits_f32.count) *
          sizeof(float) +
      (impl_->token_ids_dev.count + impl_->gather_rows_dev.count) * sizeof(std::int32_t);

  impl_->default_session.reset(new Qwen38DenseSession(*this));
  impl_->max_state_bytes = impl_->default_session->state_bytes();

  std::size_t vram_free_after = 0;
  CUDA_CHECK(cudaMemGetInfo(&vram_free_after, &vram_total));

  struct rusage ru {};
  if (::getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_maxrss > 0) {
    load_stats_.host_rss_bytes = static_cast<std::uint64_t>(ru.ru_maxrss) * 1024ull;
  }
  load_stats_.cold_safetensors_load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ld0).count();
  load_stats_.weights_device_bytes = dev_arena_bytes + f32_total_elems * sizeof(float);
  load_stats_.state_device_bytes = impl_->max_state_bytes;
  load_stats_.scratch_device_bytes = impl_->scratch_bytes;
  load_stats_.peak_vram_bytes = vram_free_before > vram_free_after ? (vram_free_before - vram_free_after) : 0ull;
}

Qwen38DenseBackend::~Qwen38DenseBackend() = default;

const Qwen38Manifest& Qwen38DenseBackend::manifest() const noexcept { return impl_->manifest; }

std::string Qwen38DenseBackend::model_id() const { return profile_.model_id; }

int Qwen38DenseBackend::max_model_len() const { return profile_.max_ctx; }

void Qwen38DenseBackend::reset_session() { impl_->default_session->reset(); }

std::unique_ptr<ingine::model::BackendSession> Qwen38DenseBackend::create_session() {
  return std::unique_ptr<ingine::model::BackendSession>(new Qwen38DenseSession(*this));
}

ingine::model::BackendSession* Qwen38DenseBackend::acquire_session(int slot) {
  if (slot != 0) return nullptr;
  return impl_->default_session.get();
}

void Qwen38DenseBackend::release_session(int /*slot*/) {}

ingine::model::ResidencyReport Qwen38DenseBackend::residency_report() const {
  ingine::model::ResidencyReport r;
  r.weights_device_bytes = load_stats_.weights_device_bytes;
  r.state_device_bytes = impl_->default_session ? impl_->default_session->state_bytes() : impl_->max_state_bytes;
  r.device_bytes = r.weights_device_bytes + r.state_device_bytes + load_stats_.scratch_device_bytes;
  struct rusage ru {};
  if (::getrusage(RUSAGE_SELF, &ru) == 0 && ru.ru_maxrss > 0) {
    r.process_rss_bytes = static_cast<std::uint64_t>(ru.ru_maxrss) * 1024ull;
    r.host_ram_bytes = r.process_rss_bytes;
  }
  return r;
}

ingine::model::AdmissionEstimate Qwen38DenseBackend::admission_estimate(int max_ctx, int max_batch) const {
  ingine::model::AdmissionEstimate est;
  const auto& c = manifest().config;
  const auto& mb = manifest().memory_budget;
  if (max_batch != 1 || max_ctx <= 0 || max_ctx > c.max_position_embeddings) {
    est.fits_current_budget = false;
    est.reason = "qwen38-dense Phase 1 supports max_batch=1 and 0 < max_ctx <= max_position_embeddings";
    return est;
  }
  const std::uint64_t kv_bytes = static_cast<std::uint64_t>(max_ctx) * mb.kv_bytes_per_token;
  const std::uint64_t state_bytes = mb.gdn_recurrent_bytes + mb.gdn_conv_bytes * 2ull + kv_bytes;
  est.required_device_bytes = load_stats_.weights_device_bytes + state_bytes + load_stats_.scratch_device_bytes;
  std::size_t vram_free = 0, vram_total = 0;
  CUDA_CHECK(cudaMemGetInfo(&vram_free, &vram_total));
  est.fits_current_budget = (est.required_device_bytes <= vram_total);
  return est;
}

void Qwen38DenseBackend::evaluate_teacher_forced_mode(std::span<const std::int32_t> token_ids,
                                                      std::span<const int> capture_positions, bool use_chunk_prefill,
                                                      const ingine::model::LogitCallback& on_logits,
                                                      double* prefill_ms_out, Qwen38DenseSession* session,
                                                      bool reset_before) {
  if (token_ids.empty()) throw std::invalid_argument("evaluate_teacher_forced empty token_ids");
  const int V = manifest().config.vocab_size;
  for (std::int32_t tid : token_ids) {
    if (tid < 0 || tid >= V) throw std::invalid_argument("evaluate_teacher_forced token_id out of range");
  }
  std::vector<int> caps(capture_positions.begin(), capture_positions.end());
  std::sort(caps.begin(), caps.end());
  caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
  for (int pos : caps) {
    if (pos < 0 || pos >= static_cast<int>(token_ids.size())) {
      throw std::invalid_argument("evaluate_teacher_forced capture position out of bounds");
    }
  }
  Qwen38DenseSession& sess = session ? *session : *impl_->default_session;
  if (reset_before) sess.reset();
  if (reset_before) {
    impl_->forward_range(sess, token_ids, caps, use_chunk_prefill, on_logits, prefill_ms_out);
  } else {
    const int base = static_cast<int>(sess.committed_tokens_.size());
    for (int& p : caps) p += base;
    impl_->forward_range(sess, token_ids, caps, use_chunk_prefill, on_logits, prefill_ms_out);
  }
}

void Qwen38DenseBackend::evaluate_teacher_forced(std::span<const std::int32_t> token_ids,
                                                 std::span<const int> capture_positions, bool use_prefix_cache,
                                                 const ingine::model::LogitCallback& on_logits,
                                                 std::span<const std::uint8_t> token_mask) {
  if (!token_mask.empty() && token_mask.size() != token_ids.size()) {
    throw std::invalid_argument("token_mask size mismatch");
  }
  Qwen38DenseSession& sess = *impl_->default_session;
  std::size_t start_pos = 0;
  if (use_prefix_cache && token_mask.empty()) {
    ingine::model::SessionLookupQuery q;
    q.identity = profile_.state_identity;
    q.prompt_ids = token_ids;
    q.use_prefix_cache = true;
    const auto lk = ingine::model::restore_session_prefix(sess, q);
    if (lk.hit && lk.reused_tokens > 0 && lk.reused_tokens < token_ids.size()) {
      bool need_earlier_cap = false;
      for (int p : capture_positions) {
        if (p < static_cast<int>(lk.reused_tokens)) {
          need_earlier_cap = true;
          break;
        }
      }
      if (!need_earlier_cap) start_pos = lk.reused_tokens;
    }
  }
  if (start_pos == 0) {
    evaluate_teacher_forced_mode(token_ids, capture_positions, true, on_logits, nullptr, &sess, true);
  } else {
    std::vector<int> caps(capture_positions.begin(), capture_positions.end());
    std::sort(caps.begin(), caps.end());
    caps.erase(std::unique(caps.begin(), caps.end()), caps.end());
    impl_->forward_range(sess, token_ids.subspan(start_pos), caps, true, on_logits, nullptr);
  }
}

void Qwen38DenseBackend::evaluate_teacher_forced(const ingine::model::BackendRequest& req,
                                                 std::span<const int> capture_positions,
                                                 const ingine::model::LogitCallback& on_logits) {
  ingine::model::validate_request_against_capabilities(profile_.capabilities, profile_.max_ctx, req.prompt, 0,
                                                       ingine::model::Sampling{});
  evaluate_teacher_forced(req.prompt.ids, capture_positions, req.use_prefix_cache, on_logits, req.token_mask);
}

void Qwen38DenseBackend::generate_greedy_exact(std::span<const std::int32_t> prompt_ids, int max_new_tokens,
                                               bool use_chunk_prefill, bool stop_on_eos,
                                               std::vector<std::int32_t>& out_tokens, double* prefill_ms_out,
                                               double* decode_ms_out, Qwen38DenseSession* session,
                                               const ingine::model::LogitCallback& on_step_logits) {
  out_tokens.clear();
  if (prompt_ids.empty() || max_new_tokens <= 0) return;
  const int V = manifest().config.vocab_size;
  for (std::int32_t tid : prompt_ids) {
    if (tid < 0 || tid >= V) throw std::invalid_argument("generate_greedy_exact prompt token out of range");
  }
  Qwen38DenseSession& sess = session ? *session : *impl_->default_session;
  sess.reset();

  std::int32_t next_tok = -1;
  auto pick_argmax = [&](const ingine::model::LogitStep& step) {
    float best_v = -std::numeric_limits<float>::infinity();
    std::int32_t best_id = 0;
    for (int i = 0; i < V; ++i) {
      const float lv = step.logits[static_cast<std::size_t>(i)];
      if (!std::isfinite(lv)) throw std::runtime_error("nonfinite logit in greedy generation");
      if (lv > best_v) {
        best_v = lv;
        best_id = i;
      }
    }
    next_tok = best_id;
    if (on_step_logits) on_step_logits(step);
  };

  const std::array<int, 1> last_cap = {static_cast<int>(prompt_ids.size()) - 1};
  impl_->forward_range(sess, prompt_ids, last_cap, use_chunk_prefill, pick_argmax, prefill_ms_out);

  const auto td0 = std::chrono::steady_clock::now();
  for (int step = 0; step < max_new_tokens; ++step) {
    out_tokens.push_back(next_tok);
    if (stop_on_eos && (next_tok == 248044 || next_tok == 248046)) break;
    if (step + 1 == max_new_tokens) break;
    impl_->decode_single_tok[0] = next_tok;
    const std::array<int, 1> cap_cur = {static_cast<int>(sess.committed_tokens_.size())};
    impl_->forward_range(sess, impl_->decode_single_tok, cap_cur, false, pick_argmax, nullptr);
  }
  if (decode_ms_out) {
    *decode_ms_out += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count();
  }
}

ingine::model::GenerateStats Qwen38DenseBackend::generate(
    std::span<const std::int32_t> prompt, int max_new_tokens, const ingine::model::Sampling& sampling,
    std::span<const std::int32_t> stop_ids, const std::function<bool(std::span<const std::int32_t>)>& on_tokens) {
  ingine::model::Prompt p;
  p.ids = prompt;
  ingine::model::validate_request_against_capabilities(profile_.capabilities, profile_.max_ctx, p, max_new_tokens,
                                                       sampling);
  if (prompt.empty()) throw std::invalid_argument("Qwen38DenseBackend::generate empty prompt");

  const int eff_max_new =
      sampling.forced.empty() ? max_new_tokens : std::min<int>(max_new_tokens, static_cast<int>(sampling.forced.size()));
  ingine::model::GenerateStats stats;
  stats.prompt_tokens = static_cast<int>(prompt.size());
  if (eff_max_new <= 0) return stats;

  Qwen38DenseSession& sess = *impl_->default_session;
  sess.reset();

  const int V = manifest().config.vocab_size;
  std::mt19937_64 rng(sampling.seed.value_or(0x9e3779b97f4a7c15ull));

  auto select_token = [&](std::span<const float> logits) -> std::int32_t {
    if (sampling.temperature <= 0.0f || sampling.top_k == 1) {
      float best_v = -std::numeric_limits<float>::infinity();
      std::int32_t best_id = 0;
      for (int i = 0; i < V; ++i) {
        if (std::find(sampling.banned_tokens.begin(), sampling.banned_tokens.end(), i) != sampling.banned_tokens.end()) {
          continue;
        }
        const float lv = logits[static_cast<std::size_t>(i)];
        if (!std::isfinite(lv)) throw std::runtime_error("nonfinite logit during generate()");
        if (lv > best_v) {
          best_v = lv;
          best_id = i;
        }
      }
      return best_id;
    }

    std::vector<std::pair<float, std::int32_t>> cand;
    cand.reserve(static_cast<std::size_t>(sampling.top_k > 0 ? std::min(sampling.top_k, V) : 256));
    const float inv_t = 1.0f / sampling.temperature;
    for (int i = 0; i < V; ++i) {
      if (std::find(sampling.banned_tokens.begin(), sampling.banned_tokens.end(), i) != sampling.banned_tokens.end()) {
        continue;
      }
      const float lv = logits[static_cast<std::size_t>(i)];
      if (!std::isfinite(lv)) throw std::runtime_error("nonfinite logit during generate()");
      cand.emplace_back(lv * inv_t, i);
    }
    if (sampling.top_k > 0 && sampling.top_k < static_cast<int>(cand.size())) {
      std::partial_sort(cand.begin(), cand.begin() + sampling.top_k, cand.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
      cand.resize(static_cast<std::size_t>(sampling.top_k));
    } else {
      std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    }
    const float max_l = cand.front().first;
    std::vector<double> probs(cand.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < cand.size(); ++i) {
      probs[i] = std::exp(static_cast<double>(cand[i].first - max_l));
      sum += probs[i];
    }
    double cum = 0.0;
    std::size_t keep = cand.size();
    if (sampling.top_p > 0.0f && sampling.top_p < 1.0f) {
      for (std::size_t i = 0; i < probs.size(); ++i) {
        cum += probs[i] / sum;
        if (cum >= static_cast<double>(sampling.top_p)) {
          keep = i + 1;
          break;
        }
      }
    }
    double sub_sum = 0.0;
    for (std::size_t i = 0; i < keep; ++i) sub_sum += probs[i];
    std::uniform_real_distribution<double> u(0.0, sub_sum);
    double r = u(rng);
    for (std::size_t i = 0; i < keep; ++i) {
      r -= probs[i];
      if (r <= 0.0) return cand[i].second;
    }
    return cand[keep - 1].second;
  };

  std::int32_t next_tok = 0;
  double pf_ms = 0.0;
  const std::array<int, 1> last_cap = {static_cast<int>(prompt.size()) - 1};
  impl_->forward_range(sess, prompt, last_cap, true,
                       [&](const ingine::model::LogitStep& st) { next_tok = select_token(st.logits); }, &pf_ms);
  stats.prefill_seconds = pf_ms / 1000.0;

  const auto td0 = std::chrono::steady_clock::now();
  for (int i = 0; i < eff_max_new; ++i) {
    const std::int32_t emit = sampling.forced.empty() ? next_tok : sampling.forced[static_cast<std::size_t>(i)];
    stats.generated_tokens = i + 1;
    const std::array<std::int32_t, 1> one = {emit};
    const bool cont = on_tokens ? on_tokens(one) : true;
    const bool stopped = std::find(stop_ids.begin(), stop_ids.end(), emit) != stop_ids.end();
    if (i + 1 == eff_max_new || stopped || !cont) break;
    impl_->decode_single_tok[0] = emit;
    const std::array<int, 1> cur_cap = {static_cast<int>(sess.committed_tokens_.size())};
    impl_->forward_range(sess, impl_->decode_single_tok, cur_cap, false,
                         [&](const ingine::model::LogitStep& st) { next_tok = select_token(st.logits); }, nullptr);
  }
  stats.decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
  return stats;
}

}  // namespace ingine::backends::qwen38

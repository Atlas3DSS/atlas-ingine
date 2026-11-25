#include "backends/qwen38/qwen38_cuda_ops.h"
#include "backends/qwen38/qwen38_cpu_ops.h"

#include "core/device.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ingine::backends::qwen38 {
namespace {

__device__ __forceinline__ float dev_bf16_to_f32(std::uint16_t b) {
  return __uint_as_float(static_cast<std::uint32_t>(b) << 16);
}

__device__ __forceinline__ std::uint16_t dev_f32_to_bf16(float x) {
  std::uint32_t u = __float_as_uint(x);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<std::uint16_t>((u >> 16) | 0x0040u);
  const std::uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<std::uint16_t>(u >> 16);
}

__device__ __forceinline__ float dev_decode_e4m3fn(std::uint8_t bits) {
  const std::uint8_t sign = (bits >> 7) & 1u;
  const std::uint8_t exp = (bits >> 3) & 0x0fu;
  const std::uint8_t mant = bits & 0x07u;
  if (exp == 0x0fu && mant == 0x07u) return __uint_as_float(0x7fc00000u);
  float val = 0.0f;
  if (exp == 0) {
    val = ldexpf(static_cast<float>(mant) * 0.125f, -6);
  } else {
    val = ldexpf(1.0f + static_cast<float>(mant) * 0.125f, static_cast<int>(exp) - 7);
  }
  return sign ? -val : val;
}

__device__ __forceinline__ std::uint8_t dev_quantize_e4m3fn(float x) {
  const std::uint32_t u = __float_as_uint(x);
  const std::uint8_t sign = static_cast<std::uint8_t>((u >> 24) & 0x80u);
  const std::uint32_t abs_u = u & 0x7fffffffu;
  if (abs_u > 0x7f800000u) return 0x7fu;
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

__device__ __forceinline__ float dev_decode_e2m1(std::uint8_t nibble) {
  static constexpr float kTable[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  return kTable[nibble & 0x0fu];
}

__device__ __forceinline__ float dev_silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float dev_softplus(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return expf(x);
  return log1pf(expf(x));
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, off);
  }
  return v;
}

template <typename T>
__device__ __forceinline__ float load_act(const T* p, std::size_t idx);

template <>
__device__ __forceinline__ float load_act<float>(const float* p, std::size_t idx) {
  return p[idx];
}

template <>
__device__ __forceinline__ float load_act<std::uint16_t>(const std::uint16_t* p, std::size_t idx) {
  return dev_bf16_to_f32(p[idx]);
}

template <typename T>
__device__ __forceinline__ void store_act(T* p, std::size_t idx, float v);

template <>
__device__ __forceinline__ void store_act<float>(float* p, std::size_t idx, float v) {
  p[idx] = v;
}

template <>
__device__ __forceinline__ void store_act<std::uint16_t>(std::uint16_t* p, std::size_t idx, float v) {
  p[idx] = dev_f32_to_bf16(v);
}

__global__ void f32_to_bf16_kernel(const float* src, std::uint16_t* dst, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = dev_f32_to_bf16(src[i]);
}

__global__ void bf16_to_f32_kernel(const std::uint16_t* src, float* dst, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = dev_bf16_to_f32(src[i]);
}

template <typename InIO, typename OutIO>
__global__ void nvfp4_linear_kernel(const InIO* __restrict__ x, const std::uint8_t* __restrict__ w_packed,
                                    const std::uint8_t* __restrict__ block_scales, float weight_scale_2, int m,
                                    int out_features, int in_features, OutIO* __restrict__ out) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp_in_block;
  if (row >= m || n >= out_features) return;

  const int blocks = in_features >> 4;
  const std::size_t x_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features);
  const std::size_t w_base = static_cast<std::size_t>(n) * static_cast<std::size_t>(in_features >> 1);
  const std::size_t s_base = static_cast<std::size_t>(n) * static_cast<std::size_t>(blocks);

  float acc = 0.0f;
  for (int b = lane; b < blocks; b += 32) {
    const float sc = dev_decode_e4m3fn(block_scales[s_base + b]) * weight_scale_2;
    float bsum = 0.0f;
    const int k0 = b << 4;
    const int p0 = b << 3;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      const std::uint8_t pb = w_packed[w_base + p0 + i];
      bsum += load_act(x, x_base + k0 + 2 * i) * dev_decode_e2m1(pb & 0x0fu);
      bsum += load_act(x, x_base + k0 + 2 * i + 1) * dev_decode_e2m1((pb >> 4) & 0x0fu);
    }
    acc += bsum * sc;
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    store_act(out, static_cast<std::size_t>(row) * static_cast<std::size_t>(out_features) + n, acc);
  }
}

template <typename ActIO>
__global__ void fp8_linear_kernel(const ActIO* __restrict__ x, const std::uint8_t* __restrict__ w_e4m3,
                                  float weight_scale, float input_scale, int m, int out_features, int in_features,
                                  ActIO* __restrict__ out) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp_in_block;
  if (row >= m || n >= out_features) return;

  const std::size_t x_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features);
  const std::size_t w_base = static_cast<std::size_t>(n) * static_cast<std::size_t>(in_features);
  const bool calibrated = (input_scale > 0.0f);
  const float inv_in = calibrated ? (1.0f / input_scale) : 1.0f;

  float acc = 0.0f;
  for (int k = lane; k < in_features; k += 32) {
    float xv = load_act(x, x_base + k);
    if (calibrated) xv = dev_decode_e4m3fn(dev_quantize_e4m3fn(xv * inv_in)) * input_scale;
    acc += xv * dev_decode_e4m3fn(w_e4m3[w_base + k]);
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    store_act(out, static_cast<std::size_t>(row) * static_cast<std::size_t>(out_features) + n, acc * weight_scale);
  }
}

template <typename InIO, typename OutIO>
__global__ void bf16_linear_kernel(const InIO* __restrict__ x, const std::uint16_t* __restrict__ w_bf16, int m,
                                   int out_features, int in_features, OutIO* __restrict__ out) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.y;
  const int n = blockIdx.x * 4 + warp_in_block;
  if (row >= m || n >= out_features) return;

  const std::size_t x_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features);
  const std::size_t w_base = static_cast<std::size_t>(n) * static_cast<std::size_t>(in_features);
  float acc = 0.0f;
  for (int k = lane; k < in_features; k += 32) {
    acc += load_act(x, x_base + k) * dev_bf16_to_f32(w_bf16[w_base + k]);
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    store_act(out, static_cast<std::size_t>(row) * static_cast<std::size_t>(out_features) + n, acc);
  }
}

template <typename ActIO>
__global__ void rmsnorm_zero_centered_kernel(const ActIO* __restrict__ x, const std::uint16_t* __restrict__ weight,
                                             float eps, int m, int dim, ActIO* __restrict__ out) {
  const int row = blockIdx.x;
  if (row >= m) return;
  const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim);

  float sq = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float v = load_act(x, base + d);
    sq += v * v;
  }
  sq = warp_sum(sq);
  __shared__ float smem[8];
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = sq;
  __syncthreads();
  float tot = 0.0f;
  if (threadIdx.x < 8) tot = smem[threadIdx.x];
  if (threadIdx.x < 32) {
    tot = warp_sum(tot);
    if (threadIdx.x == 0) smem[0] = rsqrtf(tot / static_cast<float>(dim) + eps);
  }
  __syncthreads();
  const float inv_rms = smem[0];

  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float xv = load_act(x, base + d);
    const float wv = 1.0f + dev_bf16_to_f32(weight[d]);
    store_act(out, base + d, xv * inv_rms * wv);
  }
}

template <typename ActIO>
__global__ void rmsnorm_gated_silu_kernel(const ActIO* __restrict__ x, const ActIO* __restrict__ gate_z,
                                          const float* __restrict__ weight, float eps, int m, int dim,
                                          ActIO* __restrict__ out) {
  const int row = blockIdx.x;
  if (row >= m) return;
  const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim);

  float sq = 0.0f;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float v = load_act(x, base + d);
    sq += v * v;
  }
  sq = warp_sum(sq);
  __shared__ float smem[8];
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = sq;
  __syncthreads();
  float tot = 0.0f;
  if (threadIdx.x < 8) tot = smem[threadIdx.x];
  if (threadIdx.x < 32) {
    tot = warp_sum(tot);
    if (threadIdx.x == 0) smem[0] = rsqrtf(tot / static_cast<float>(dim) + eps);
  }
  __syncthreads();
  const float inv_rms = smem[0];

  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    const float xv = load_act(x, base + d);
    const float zv = load_act(gate_z, base + d);
    store_act(out, base + d, weight[d] * (xv * inv_rms) * dev_silu(zv));
  }
}

template <typename ActIO>
__global__ void partial_rope_neox_kernel(ActIO* __restrict__ qk, int m, int num_heads, int head_dim, int rotary_dim,
                                         int pos_offset, float rope_theta) {
  const int half = rotary_dim >> 1;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  if (row >= m || tid >= num_heads * half) return;

  const int h = tid / half;
  const int j = tid % half;
  const std::size_t off = (static_cast<std::size_t>(row) * static_cast<std::size_t>(num_heads) + h) *
                          static_cast<std::size_t>(head_dim);

  const double exponent = -2.0 * static_cast<double>(j) / static_cast<double>(rotary_dim);
  const double angle = static_cast<double>(pos_offset + row) * pow(static_cast<double>(rope_theta), exponent);
  double sd = 0.0, cd = 0.0;
  sincos(angle, &sd, &cd);
  const float s = static_cast<float>(sd);
  const float c = static_cast<float>(cd);

  const float x0 = load_act(qk, off + j);
  const float x1 = load_act(qk, off + j + half);
  store_act(qk, off + j, x0 * c - x1 * s);
  store_act(qk, off + j + half, x1 * c + x0 * s);
}

template <typename ActIO>
__global__ void causal_conv1d_silu_prefill_kernel(const ActIO* __restrict__ x_seq,
                                                  const float* __restrict__ conv_weight, int seq_len, int conv_dim,
                                                  int kernel_size, float* __restrict__ conv_state,
                                                  ActIO* __restrict__ out_seq) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;

  float st[8]{};
  float w[8]{};
  const int base = c * kernel_size;
  for (int k = 0; k < kernel_size; ++k) {
    st[k] = conv_state[base + k];
    w[k] = conv_weight[base + k];
  }
  for (int t = 0; t < seq_len; ++t) {
    for (int k = 0; k + 1 < kernel_size; ++k) st[k] = st[k + 1];
    st[kernel_size - 1] = load_act(x_seq, static_cast<std::size_t>(t) * static_cast<std::size_t>(conv_dim) + c);
    float acc = 0.0f;
    for (int k = 0; k < kernel_size; ++k) acc += st[k] * w[k];
    store_act(out_seq, static_cast<std::size_t>(t) * static_cast<std::size_t>(conv_dim) + c, dev_silu(acc));
  }
  for (int k = 0; k < kernel_size; ++k) conv_state[base + k] = st[k];
}

template <typename ActIO>
__global__ void gdn_normalize_qk_kernel(const ActIO* __restrict__ q_seq, const ActIO* __restrict__ k_seq, int seq_len,
                                        int num_k_heads, int k_dim, float* __restrict__ q_norm,
                                        float* __restrict__ k_norm) {
  const int kh = blockIdx.x;
  const int t = blockIdx.y;
  if (t >= seq_len || kh >= num_k_heads) return;

  const std::size_t base =
      (static_cast<std::size_t>(t) * static_cast<std::size_t>(num_k_heads) + kh) * static_cast<std::size_t>(k_dim);
  float qsq = 0.0f, ksq = 0.0f;
  for (int d = threadIdx.x; d < k_dim; d += blockDim.x) {
    const float qv = load_act(q_seq, base + d);
    const float kv = load_act(k_seq, base + d);
    qsq += qv * qv;
    ksq += kv * kv;
  }
  qsq = warp_sum(qsq);
  ksq = warp_sum(ksq);
  __shared__ float rq[4], rk[4];
  if ((threadIdx.x & 31) == 0) {
    rq[threadIdx.x >> 5] = qsq;
    rk[threadIdx.x >> 5] = ksq;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float qt = 0.0f, kt = 0.0f;
    const int wcount = blockDim.x >> 5;
    for (int i = 0; i < wcount; ++i) {
      qt += rq[i];
      kt += rk[i];
    }
    rq[0] = rsqrtf(qt + 1e-6f) * rsqrtf(static_cast<float>(k_dim));
    rk[0] = rsqrtf(kt + 1e-6f);
  }
  __syncthreads();
  const float q_inv = rq[0];
  const float k_inv = rk[0];
  for (int d = threadIdx.x; d < k_dim; d += blockDim.x) {
    q_norm[base + d] = load_act(q_seq, base + d) * q_inv;
    k_norm[base + d] = load_act(k_seq, base + d) * k_inv;
  }
}

template <typename ActIO>
__global__ void gdn_recurrent_step_exact_kernel(const float* __restrict__ q_norm, const float* __restrict__ k_norm,
                                                const ActIO* __restrict__ v_vheads, const ActIO* __restrict__ z_vheads,
                                                const ActIO* __restrict__ a_vheads, const ActIO* __restrict__ b_vheads,
                                                const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                                const float* __restrict__ norm_weight, float rms_eps, int num_k_heads,
                                                int num_v_heads, int k_dim, int v_dim,
                                                float* __restrict__ recurrent_state, ActIO* __restrict__ out_vdim) {
  const int vh = blockIdx.x;
  if (vh >= num_v_heads) return;
  const int repeat_factor = num_v_heads / num_k_heads;
  const int kh = vh / repeat_factor;

  __shared__ float sqh[256];
  __shared__ float skh[256];
  __shared__ float s_core[512];
  __shared__ float s_red[8];

  const float* qh_g = q_norm + static_cast<std::size_t>(kh * k_dim);
  const float* kh_g = k_norm + static_cast<std::size_t>(kh * k_dim);
  for (int d = threadIdx.x; d < k_dim; d += blockDim.x) {
    sqh[d] = qh_g[d];
    skh[d] = kh_g[d];
  }
  __syncthreads();

  const float beta = 1.0f / (1.0f + expf(-load_act(b_vheads, vh)));
  const float g = -expf(a_log[vh]) * dev_softplus(load_act(a_vheads, vh) + dt_bias[vh]);
  const float decay = expf(g);

  float* state = recurrent_state + static_cast<std::size_t>(vh) * static_cast<std::size_t>(k_dim) *
                                       static_cast<std::size_t>(v_dim);
  const std::size_t v_off = static_cast<std::size_t>(vh * v_dim);

  float sq_sum = 0.0f;
  for (int vd = threadIdx.x; vd < v_dim; vd += blockDim.x) {
    float kv_mem = 0.0f;
    for (int kd = 0; kd < k_dim; ++kd) {
      kv_mem += (state[static_cast<std::size_t>(kd) * static_cast<std::size_t>(v_dim) + vd] * decay) * skh[kd];
    }
    const float delta = (load_act(v_vheads, v_off + vd) - kv_mem) * beta;
    float core_vd = 0.0f;
    for (int kd = 0; kd < k_dim; ++kd) {
      const std::size_t idx = static_cast<std::size_t>(kd) * static_cast<std::size_t>(v_dim) + vd;
      const float ns = state[idx] * decay + skh[kd] * delta;
      state[idx] = ns;
      core_vd += ns * sqh[kd];
    }
    s_core[vd] = core_vd;
    sq_sum += core_vd * core_vd;
  }

  sq_sum = warp_sum(sq_sum);
  if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = sq_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float tot = 0.0f;
    const int wcount = blockDim.x >> 5;
    for (int i = 0; i < wcount; ++i) tot += s_red[i];
    s_red[0] = rsqrtf(tot / static_cast<float>(v_dim) + rms_eps);
  }
  __syncthreads();

  const float inv_rms = s_red[0];
  for (int vd = threadIdx.x; vd < v_dim; vd += blockDim.x) {
    const float zv = load_act(z_vheads, v_off + vd);
    store_act(out_vdim, v_off + vd, norm_weight[vd] * (s_core[vd] * inv_rms) * dev_silu(zv));
  }
}

template <typename ActIO>
__global__ void gdn_chunk_prefill_kernel(const float* __restrict__ qn, const float* __restrict__ kn,
                                         const ActIO* __restrict__ v_seq, const ActIO* __restrict__ z_seq,
                                         const ActIO* __restrict__ a_seq, const ActIO* __restrict__ b_seq,
                                         const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                         const float* __restrict__ norm_weight, float rms_eps, int seq_len,
                                         int chunk_size, int num_k_heads, int num_v_heads, int k_dim, int v_dim,
                                         float* __restrict__ recurrent_state, float* __restrict__ w_mat_ws,
                                         float* __restrict__ v_new_ws, ActIO* __restrict__ out_seq) {
  const int vh = blockIdx.x;
  if (vh >= num_v_heads) return;
  const int repeat_factor = num_v_heads / num_k_heads;
  const int kh = vh / repeat_factor;
  const int k_stride = num_k_heads * k_dim;
  const int v_stride = num_v_heads * v_dim;

  float* state = recurrent_state + static_cast<std::size_t>(vh) * static_cast<std::size_t>(k_dim) *
                                       static_cast<std::size_t>(v_dim);
  float* w_mat = w_mat_ws + static_cast<std::size_t>(vh) * static_cast<std::size_t>(chunk_size) *
                                static_cast<std::size_t>(k_dim);
  float* v_new = v_new_ws + static_cast<std::size_t>(vh) * static_cast<std::size_t>(chunk_size) *
                                static_cast<std::size_t>(v_dim);

  __shared__ float g_cum[64];
  __shared__ float beta[64];
  __shared__ float coeffs[64];
  __shared__ float k_row[256];
  __shared__ float v_core[512];
  __shared__ float s_red[8];

  const float exp_alog = expf(a_log[vh]);
  const float bias = dt_bias[vh];

  for (int c0 = 0; c0 < seq_len; c0 += chunk_size) {
    const int L = min(chunk_size, seq_len - c0);

    // 1. Precompute g_cum and beta on thread 0 (L <= 64).
    if (threadIdx.x == 0) {
      float run_g = 0.0f;
      for (int i = 0; i < L; ++i) {
        const int t = c0 + i;
        const float av = load_act(a_seq, static_cast<std::size_t>(t * num_v_heads + vh));
        const float bv = load_act(b_seq, static_cast<std::size_t>(t * num_v_heads + vh));
        run_g += -exp_alog * dev_softplus(av + bias);
        g_cum[i] = run_g;
        beta[i] = 1.0f / (1.0f + expf(-bv));
      }
    }
    __syncthreads();

    // 2. Initialize v_corr -> v_new and k_w -> w_mat, then solve WY lower triangular system.
    for (int i = 0; i < L; ++i) {
      const int ti = c0 + i;
      const float bi = beta[i];
      const float kg_scale = expf(g_cum[i]) * bi;
      const float* ki_g = kn + static_cast<std::size_t>(ti) * k_stride + static_cast<std::size_t>(kh * k_dim);
      const std::size_t vi_off = static_cast<std::size_t>(ti) * v_stride + static_cast<std::size_t>(vh * v_dim);

      for (int vd = threadIdx.x; vd < v_dim; vd += blockDim.x) {
        v_new[i * v_dim + vd] = load_act(v_seq, vi_off + vd) * bi;
      }
      for (int kd = threadIdx.x; kd < k_dim; kd += blockDim.x) {
        const float kv = ki_g[kd];
        k_row[kd] = kv;
        w_mat[i * k_dim + kd] = kv * kg_scale;
      }
      __syncthreads();

      for (int j = 0; j < i; ++j) {
        const int tj = c0 + j;
        const float* kj_g = kn + static_cast<std::size_t>(tj) * k_stride + static_cast<std::size_t>(kh * k_dim);
        float dot_k = 0.0f;
        for (int kd = threadIdx.x; kd < k_dim; kd += blockDim.x) dot_k += k_row[kd] * kj_g[kd];
        dot_k = warp_sum(dot_k);
        if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = dot_k;
        __syncthreads();
        if (threadIdx.x == 0) {
          float tot = 0.0f;
          const int wc = blockDim.x >> 5;
          for (int w = 0; w < wc; ++w) tot += s_red[w];
          s_red[0] = tot * expf(g_cum[i] - g_cum[j]) * bi;
        }
        __syncthreads();
        const float coeff = s_red[0];
        for (int vd = threadIdx.x; vd < v_dim; vd += blockDim.x) {
          v_new[i * v_dim + vd] -= coeff * v_new[j * v_dim + vd];
        }
        for (int kd = threadIdx.x; kd < k_dim; kd += blockDim.x) {
          w_mat[i * k_dim + kd] -= coeff * w_mat[j * k_dim + kd];
        }
        __syncthreads();
      }
    }

    // 3. Subtract w_mat @ state from v_new (tiles of 8 rows keep register count low).
    for (int i0 = 0; i0 < L; i0 += 8) {
      const int ir = min(8, L - i0);
      for (int vd = threadIdx.x; vd < v_dim; vd += blockDim.x) {
        float ws[8]{};
        for (int kd = 0; kd < k_dim; ++kd) {
          const float st_v = state[static_cast<std::size_t>(kd) * static_cast<std::size_t>(v_dim) + vd];
          for (int r = 0; r < ir; ++r) {
            ws[r] += w_mat[(i0 + r) * k_dim + kd] * st_v;
          }
        }
        for (int r = 0; r < ir; ++r) {
          v_new[(i0 + r) * v_dim + vd] -= ws[r];
        }
      }
    }
    __syncthreads();

    // 4. Causal core output + gated RMSNorm for each token in chunk.
    for (int i = 0; i < L; ++i) {
      const int ti = c0 + i;
      const float* qi_g = qn + static_cast<std::size_t>(ti) * k_stride + static_cast<std::size_t>(kh * k_dim);
      for (int kd = threadIdx.x; kd < k_dim; kd += blockDim.x) k_row[kd] = qi_g[kd];
      __syncthreads();

      for (int j = 0; j <= i; ++j) {
        const int tj = c0 + j;
        const float* kj_g = kn + static_cast<std::size_t>(tj) * k_stride + static_cast<std::size_t>(kh * k_dim);
        float qk = 0.0f;
        for (int kd = threadIdx.x; kd < k_dim; kd += blockDim.x) qk += k_row[kd] * kj_g[kd];
        qk = warp_sum(qk);
        if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = qk;
        __syncthreads();
        if (threadIdx.x == 0) {
          float tot = 0.0f;
          const int wc = blockDim.x >> 5;
          for (int w = 0; w < wc; ++w) tot += s_red[w];
          coeffs[j] = tot * expf(g_cum[i] - g_cum[j]);
        }
        __syncthreads();
      }

      const float g_i_exp = expf(g_cum[i]);
      float sq = 0.0f;
      for (int vd = threadIdx.x; vd < v_dim; vd += blockDim.x) {
        float inter = 0.0f;
        for (int kd = 0; kd < k_dim; ++kd) {
          inter += k_row[kd] * state[static_cast<std::size_t>(kd) * static_cast<std::size_t>(v_dim) + vd];
        }
        float intra = 0.0f;
        for (int j = 0; j <= i; ++j) {
          intra += coeffs[j] * v_new[j * v_dim + vd];
        }
        const float cv = inter * g_i_exp + intra;
        v_core[vd] = cv;
        sq += cv * cv;
      }
      sq = warp_sum(sq);
      if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = sq;
      __syncthreads();
      if (threadIdx.x == 0) {
        float tot = 0.0f;
        const int wc = blockDim.x >> 5;
        for (int w = 0; w < wc; ++w) tot += s_red[w];
        s_red[0] = rsqrtf(tot / static_cast<float>(v_dim) + rms_eps);
      }
      __syncthreads();

      const float inv_rms = s_red[0];
      const std::size_t out_off = static_cast<std::size_t>(ti) * v_stride + static_cast<std::size_t>(vh * v_dim);
      for (int vd = threadIdx.x; vd < v_dim; vd += blockDim.x) {
        const float zv = load_act(z_seq, out_off + vd);
        store_act(out_seq, out_off + vd, norm_weight[vd] * (v_core[vd] * inv_rms) * dev_silu(zv));
      }
      __syncthreads();
    }

    // 5. Inter-chunk recurrent state update: state = state * exp(g_cum[L-1]) + sum_i k[i]*exp(g_end - g_i)*v_new[i]
    if (threadIdx.x < static_cast<unsigned>(L)) {
      coeffs[threadIdx.x] = expf(g_cum[L - 1] - g_cum[threadIdx.x]);
    }
    const float end_decay = expf(g_cum[L - 1]);
    __syncthreads();

    for (int vd = threadIdx.x; vd < v_dim; vd += blockDim.x) {
      for (int kd = 0; kd < k_dim; ++kd) {
        const std::size_t sidx = static_cast<std::size_t>(kd) * static_cast<std::size_t>(v_dim) + vd;
        float acc = state[sidx] * end_decay;
        for (int i = 0; i < L; ++i) {
          const float* ki_g = kn + static_cast<std::size_t>(c0 + i) * k_stride + static_cast<std::size_t>(kh * k_dim);
          acc += ki_g[kd] * coeffs[i] * v_new[i * v_dim + vd];
        }
        state[sidx] = acc;
      }
    }
    __syncthreads();
  }
}

template <typename ActIO>
__global__ void split_q_gate_kernel(const ActIO* __restrict__ q_with_gate, ActIO* __restrict__ q_out,
                                    ActIO* __restrict__ gate_out, int seq_len, int num_heads, int head_dim) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y;
  const int total = num_heads * head_dim;
  if (row >= seq_len || tid >= total) return;
  const int h = tid / head_dim;
  const int d = tid % head_dim;
  const std::size_t src_base =
      (static_cast<std::size_t>(row) * static_cast<std::size_t>(num_heads) + h) * static_cast<std::size_t>(2 * head_dim);
  const std::size_t dst_idx =
      static_cast<std::size_t>(row) * static_cast<std::size_t>(total) + static_cast<std::size_t>(tid);
  q_out[dst_idx] = q_with_gate[src_base + d];
  gate_out[dst_idx] = q_with_gate[src_base + head_dim + d];
}

template <typename ActIO, typename KvIO = ActIO>
__global__ void causal_gqa_gated_attn_kernel(const ActIO* __restrict__ q, const ActIO* __restrict__ gate,
                                             const KvIO* __restrict__ k_cache, const KvIO* __restrict__ v_cache,
                                             int seq_len, int pos_offset, int num_heads, int num_kv_heads,
                                             int head_dim, ActIO* __restrict__ out_attn) {
  const int h = blockIdx.x;
  const int row = blockIdx.y;
  if (h >= num_heads || row >= seq_len) return;

  const int d = threadIdx.x;
  const int group = num_heads / num_kv_heads;
  const int kvh = h / group;
  const int kv_stride = num_kv_heads * head_dim;
  const int query_pos = pos_offset + row;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  const std::size_t q_base =
      (static_cast<std::size_t>(row) * static_cast<std::size_t>(num_heads) + h) * static_cast<std::size_t>(head_dim);
  const float qd = (d < head_dim) ? load_act(q, q_base + d) : 0.0f;

  float m_i = -1e38f;
  float l_i = 0.0f;
  float o_d = 0.0f;
  __shared__ float s_red[8];
  __shared__ float s_score;

  for (int t = 0; t <= query_pos; ++t) {
    const std::size_t k_off =
        static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_stride) + static_cast<std::size_t>(kvh * head_dim);
    float prod = (d < head_dim) ? (qd * load_act(k_cache, k_off + d)) : 0.0f;
    prod = warp_sum(prod);
    if ((d & 31) == 0) s_red[d >> 5] = prod;
    __syncthreads();
    if (d == 0) {
      float tot = 0.0f;
      const int wc = blockDim.x >> 5;
      for (int w = 0; w < wc; ++w) tot += s_red[w];
      s_score = tot * scale;
    }
    __syncthreads();

    const float score = s_score;
    const float m_new = fmaxf(m_i, score);
    const float alpha = (t == 0) ? 0.0f : expf(m_i - m_new);
    const float p = expf(score - m_new);
    l_i = l_i * alpha + p;
    if (d < head_dim) {
      o_d = o_d * alpha + p * load_act(v_cache, k_off + d);
    }
    m_i = m_new;
    __syncthreads();
  }

  if (d < head_dim) {
    const float gv = 1.0f / (1.0f + expf(-load_act(gate, q_base + d)));
    store_act(out_attn, q_base + d, (o_d / l_i) * gv);
  }
}

__global__ void swiglu_mul_bf16_kernel(const std::uint16_t* gate, const std::uint16_t* up, std::uint16_t* out,
                                       std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float gv = dev_bf16_to_f32(gate[i]);
    const float uv = dev_bf16_to_f32(up[i]);
    out[i] = dev_f32_to_bf16(dev_silu(gv) * uv);
  }
}

__global__ void add_residual_bf16_kernel(std::uint16_t* hidden, const std::uint16_t* residual, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    hidden[i] = dev_f32_to_bf16(dev_bf16_to_f32(hidden[i]) + dev_bf16_to_f32(residual[i]));
  }
}

__global__ void split_qkv_sep_bf16_kernel(const std::uint16_t* mixed, std::uint16_t* q_sep, std::uint16_t* k_sep,
                                          std::uint16_t* v_sep, int seq_len, int kd, int vd) {
  const int cd = 2 * kd + vd;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  if (t >= seq_len || idx >= cd) return;
  const std::uint16_t val = mixed[static_cast<std::size_t>(t * cd + idx)];
  if (idx < kd) {
    q_sep[static_cast<std::size_t>(t * kd + idx)] = val;
  } else if (idx < 2 * kd) {
    k_sep[static_cast<std::size_t>(t * kd + (idx - kd))] = val;
  } else {
    v_sep[static_cast<std::size_t>(t * vd + (idx - 2 * kd))] = val;
  }
}

template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  std::size_t count = 0;
  DevBuf() = default;
  explicit DevBuf(std::size_t n) { allocate(n); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : ptr(o.ptr), count(o.count) {
    o.ptr = nullptr;
    o.count = 0;
  }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) {
      free();
      ptr = o.ptr;
      count = o.count;
      o.ptr = nullptr;
      o.count = 0;
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
  void upload(const T* host_src, std::size_t n) {
    allocate(n);
    if (n > 0) CUDA_CHECK(cudaMemcpy(ptr, host_src, n * sizeof(T), cudaMemcpyHostToDevice));
  }
};

struct DevProjWeights {
  int out_features = 0;
  int in_features = 0;
  Qwen38LinearQuantKind quant = Qwen38LinearQuantKind::Bf16;
  DevBuf<std::uint16_t> bf16_w;
  DevBuf<std::uint8_t> fp8_w;
  DevBuf<std::uint8_t> nvfp4_packed;
  DevBuf<std::uint8_t> nvfp4_scales;
  float weight_scale = 1.0f;
  float weight_scale_2 = 1.0f;
  float input_scale = 0.0f;

  static DevProjWeights from_host(const Qwen38ProjWeightsHost& h) {
    DevProjWeights d;
    d.out_features = h.out_features;
    d.in_features = h.in_features;
    d.quant = h.quant;
    d.weight_scale = h.weight_scale;
    d.weight_scale_2 = h.weight_scale_2;
    d.input_scale = h.input_scale;
    if (h.quant == Qwen38LinearQuantKind::Bf16) {
      d.bf16_w.upload(h.bf16_w.data(), h.bf16_w.size());
    } else if (h.quant == Qwen38LinearQuantKind::Fp8E4M3) {
      d.fp8_w.upload(h.fp8_w.data(), h.fp8_w.size());
    } else {
      d.nvfp4_packed.upload(h.nvfp4_packed.data(), h.nvfp4_packed.size());
      d.nvfp4_scales.upload(h.nvfp4_scales.data(), h.nvfp4_scales.size());
    }
    return d;
  }

  void forward_bf16(const __nv_bfloat16* x, int m, __nv_bfloat16* out, bool use_calibrated_fp8,
                    cudaStream_t stream) const {
    if (quant == Qwen38LinearQuantKind::Bf16) {
      qwen38_cuda_bf16_linear_bf16(x, reinterpret_cast<const __nv_bfloat16*>(bf16_w.ptr), m, out_features, in_features,
                                   out, stream);
    } else if (quant == Qwen38LinearQuantKind::Fp8E4M3) {
      qwen38_cuda_fp8_linear_bf16(x, fp8_w.ptr, weight_scale, use_calibrated_fp8 ? input_scale : 0.0f, m, out_features,
                                  in_features, out, stream);
    } else {
      qwen38_cuda_nvfp4_linear_bf16(x, nvfp4_packed.ptr, nvfp4_scales.ptr, weight_scale_2, m, out_features, in_features,
                                    out, stream);
    }
  }
};

struct DevBlockLayer {
  Qwen38LayerKind kind = Qwen38LayerKind::LinearAttention;
  DevBuf<std::uint16_t> input_layernorm;
  DevBuf<std::uint16_t> post_attention_layernorm;

  DevProjWeights in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj;
  DevBuf<float> A_log, dt_bias, conv1d_weight, linear_norm_weight;

  DevProjWeights q_proj, k_proj, v_proj, o_proj;
  DevBuf<std::uint16_t> q_norm, k_norm;

  DevProjWeights gate_proj, up_proj, down_proj;
};

template <typename ActIO>
void dispatch_gdn_step(const ActIO* q_kheads, const ActIO* k_kheads, const ActIO* v_vheads, const ActIO* z_vheads,
                       const ActIO* a_vheads, const ActIO* b_vheads, const float* a_log, const float* dt_bias,
                       const float* norm_weight, float rms_eps, int num_k_heads, int num_v_heads, int k_dim, int v_dim,
                       float* recurrent_state, float* qk_norm_scratch, ActIO* out_vdim, cudaStream_t stream) {
  if (v_dim > 512 || k_dim > 256 || num_v_heads % num_k_heads != 0) {
    throw std::invalid_argument("qwen38_cuda_gdn_recurrent_step unsupported head dims");
  }
  const int k_total = num_k_heads * k_dim;
  float* qn = qk_norm_scratch;
  float* kn = qk_norm_scratch + k_total;
  gdn_normalize_qk_kernel<ActIO><<<dim3(num_k_heads, 1), 128, 0, stream>>>(q_kheads, k_kheads, 1, num_k_heads, k_dim,
                                                                           qn, kn);
  CUDA_CHECK(cudaGetLastError());
  const int block = ((std::max(32, std::min(256, v_dim)) + 31) / 32) * 32;
  gdn_recurrent_step_exact_kernel<ActIO><<<num_v_heads, block, 0, stream>>>(
      qn, kn, v_vheads, z_vheads, a_vheads, b_vheads, a_log, dt_bias, norm_weight, rms_eps, num_k_heads, num_v_heads,
      k_dim, v_dim, recurrent_state, out_vdim);
  CUDA_CHECK(cudaGetLastError());
}

template <typename ActIO>
void dispatch_gdn_chunk(const ActIO* q_seq, const ActIO* k_seq, const ActIO* v_seq, const ActIO* z_seq,
                        const ActIO* a_seq, const ActIO* b_seq, const float* a_log, const float* dt_bias,
                        const float* norm_weight, float rms_eps, int seq_len, int chunk_size, int num_k_heads,
                        int num_v_heads, int k_dim, int v_dim, float* recurrent_state, float* scratch, ActIO* out_seq,
                        cudaStream_t stream) {
  if (seq_len == 0) return;
  if (chunk_size < 1 || chunk_size > 64 || k_dim > 256 || v_dim > 512 || num_v_heads % num_k_heads != 0) {
    throw std::invalid_argument("qwen38_cuda_gdn_chunk_prefill invalid chunk/head dimensions");
  }
  const std::size_t qk_elems = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(num_k_heads * k_dim);
  const std::size_t w_elems =
      static_cast<std::size_t>(num_v_heads) * static_cast<std::size_t>(chunk_size) * static_cast<std::size_t>(k_dim);
  float* qn = scratch;
  float* kn = qn + qk_elems;
  float* w_mat_ws = kn + qk_elems;
  float* v_new_ws = w_mat_ws + w_elems;

  gdn_normalize_qk_kernel<ActIO><<<dim3(num_k_heads, seq_len), 128, 0, stream>>>(q_seq, k_seq, seq_len, num_k_heads,
                                                                                 k_dim, qn, kn);
  CUDA_CHECK(cudaGetLastError());
  const int block = ((std::max(64, std::min(256, std::max(k_dim, v_dim))) + 31) / 32) * 32;
  gdn_chunk_prefill_kernel<ActIO><<<num_v_heads, block, 0, stream>>>(
      qn, kn, v_seq, z_seq, a_seq, b_seq, a_log, dt_bias, norm_weight, rms_eps, seq_len, chunk_size, num_k_heads,
      num_v_heads, k_dim, v_dim, recurrent_state, w_mat_ws, v_new_ws, out_seq);
  CUDA_CHECK(cudaGetLastError());
}

template <typename ActIO>
void dispatch_full_attn_gated(const ActIO* q_with_gate, const ActIO* k_raw, const ActIO* v_in,
                              const std::uint16_t* q_norm_w, const std::uint16_t* k_norm_w, float rms_eps,
                              float rope_theta, int seq_len, int pos_offset, int num_heads, int num_kv_heads,
                              int head_dim, int rotary_dim, ActIO* k_cache, ActIO* v_cache, int max_cache_len,
                              ActIO* q_scratch, ActIO* gate_scratch, ActIO* out_attn, cudaStream_t stream) {
  if (seq_len < 1 || pos_offset + seq_len > max_cache_len || head_dim > 256 || num_heads % num_kv_heads != 0) {
    throw std::invalid_argument("qwen38_cuda_full_attn_gated invalid dimensions or cache bounds");
  }
  const int q_total = num_heads * head_dim;
  const int kv_stride = num_kv_heads * head_dim;

  split_q_gate_kernel<ActIO><<<dim3((q_total + 255) / 256, seq_len), 256, 0, stream>>>(
      q_with_gate, q_scratch, gate_scratch, seq_len, num_heads, head_dim);
  CUDA_CHECK(cudaGetLastError());

  // Normalize Q heads and K heads in-place/copy to cache staging.
  rmsnorm_zero_centered_kernel<ActIO><<<seq_len * num_heads, 256, 0, stream>>>(
      q_scratch, q_norm_w, rms_eps, seq_len * num_heads, head_dim, q_scratch);
  CUDA_CHECK(cudaGetLastError());

  ActIO* k_dst = k_cache + static_cast<std::size_t>(pos_offset) * static_cast<std::size_t>(kv_stride);
  ActIO* v_dst = v_cache + static_cast<std::size_t>(pos_offset) * static_cast<std::size_t>(kv_stride);
  rmsnorm_zero_centered_kernel<ActIO><<<seq_len * num_kv_heads, 256, 0, stream>>>(
      k_raw, k_norm_w, rms_eps, seq_len * num_kv_heads, head_dim, k_dst);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(v_dst, v_in, static_cast<std::size_t>(seq_len * kv_stride) * sizeof(ActIO),
                             cudaMemcpyDeviceToDevice, stream));

  const int half = rotary_dim >> 1;
  partial_rope_neox_kernel<ActIO><<<dim3((num_heads * half + 127) / 128, seq_len), 128, 0, stream>>>(
      q_scratch, seq_len, num_heads, head_dim, rotary_dim, pos_offset, rope_theta);
  CUDA_CHECK(cudaGetLastError());
  partial_rope_neox_kernel<ActIO><<<dim3((num_kv_heads * half + 127) / 128, seq_len), 128, 0, stream>>>(
      k_dst, seq_len, num_kv_heads, head_dim, rotary_dim, pos_offset, rope_theta);
  CUDA_CHECK(cudaGetLastError());

  causal_gqa_gated_attn_kernel<ActIO><<<dim3(num_heads, seq_len), 256, 0, stream>>>(
      q_scratch, gate_scratch, k_cache, v_cache, seq_len, pos_offset, num_heads, num_kv_heads, head_dim, out_attn);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void dequant_nvfp4_to_bf16_kernel(const std::uint8_t* __restrict__ w_packed,
                                             const std::uint8_t* __restrict__ block_scales, int n_rows,
                                             int in_features, std::uint16_t* __restrict__ out_bf16) {
  const int blocks = in_features >> 4;
  const int r = blockIdx.y;
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n_rows || b >= blocks) return;
  const float sc = dev_decode_e4m3fn(block_scales[static_cast<std::size_t>(r) * blocks + b]);
  const std::size_t p_base = static_cast<std::size_t>(r) * static_cast<std::size_t>(in_features >> 1) +
                             static_cast<std::size_t>(b * 8);
  const std::size_t o_base = static_cast<std::size_t>(r) * static_cast<std::size_t>(in_features) +
                             static_cast<std::size_t>(b * 16);
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const std::uint8_t pb = w_packed[p_base + i];
    out_bf16[o_base + 2 * i] = dev_f32_to_bf16(dev_decode_e2m1(pb & 0x0fu) * sc);
    out_bf16[o_base + 2 * i + 1] = dev_f32_to_bf16(dev_decode_e2m1((pb >> 4) & 0x0fu) * sc);
  }
}

__global__ void dequant_fp8_to_bf16_kernel(const std::uint8_t* __restrict__ w_e4m3, std::size_t n,
                                           std::uint16_t* __restrict__ out_bf16) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out_bf16[i] = dev_f32_to_bf16(dev_decode_e4m3fn(w_e4m3[i]));
}

__global__ void quant_fp8_act_to_bf16_kernel(const std::uint16_t* __restrict__ x_in, float inv_scale, std::size_t n,
                                             std::uint16_t* __restrict__ x_out) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float xv = dev_bf16_to_f32(x_in[i]) * inv_scale;
    x_out[i] = dev_f32_to_bf16(dev_decode_e4m3fn(dev_quantize_e4m3fn(xv)));
  }
}

__global__ void embed_lookup_bf16_kernel(const std::uint16_t* __restrict__ embed_table,
                                         const std::int32_t* __restrict__ token_ids, int seq_len, int hidden_size,
                                         int vocab_size, std::uint16_t* __restrict__ out_hidden) {
  const int row = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= seq_len || d >= hidden_size) return;
  const std::int32_t tid = token_ids[row];
  const std::uint16_t val =
      (tid >= 0 && tid < vocab_size) ? embed_table[static_cast<std::size_t>(tid) * hidden_size + d] : 0u;
  out_hidden[static_cast<std::size_t>(row) * hidden_size + d] = val;
}

__global__ void gather_rows_bf16_kernel(const std::uint16_t* __restrict__ src, const int* __restrict__ row_indices,
                                        int num_rows, int hidden_size, std::uint16_t* __restrict__ dst) {
  const int row = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows || d >= hidden_size) return;
  const int src_row = row_indices[row];
  dst[static_cast<std::size_t>(row) * hidden_size + d] =
      src[static_cast<std::size_t>(src_row) * hidden_size + d];
}

__global__ void quant_fp8_act_f32_kernel(const float* __restrict__ x_in, float inv_scale, float input_scale,
                                         std::size_t n, float* __restrict__ x_out) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x_out[i] = dev_decode_e4m3fn(dev_quantize_e4m3fn(x_in[i] * inv_scale)) * input_scale;
  }
}

template <int TM, bool kAligned16>
__global__ void nvfp4_linear_f32_batched_kernel(const float* __restrict__ x, const std::uint8_t* __restrict__ w_packed,
                                                const std::uint8_t* __restrict__ block_scales, float weight_scale_2,
                                                int m, int out_features, int in_features, float* __restrict__ out) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int m0 = blockIdx.y * TM;
  const int n = blockIdx.x * 4 + warp_in_block;
  if (m0 >= m || n >= out_features) return;
  const int tm = min(TM, m - m0);

  const int blocks = in_features >> 4;
  const std::size_t w_base = static_cast<std::size_t>(n) * static_cast<std::size_t>(in_features >> 1);
  const std::size_t s_base = static_cast<std::size_t>(n) * static_cast<std::size_t>(blocks);

  float acc[TM]{};
  for (int b = lane; b < blocks; b += 32) {
    const float sc = dev_decode_e4m3fn(block_scales[s_base + b]) * weight_scale_2;
    std::uint32_t packed[2]{};
    std::memcpy(packed, w_packed + w_base + static_cast<std::size_t>(b << 3), 8);
    const std::uint16_t words[4] = {
        static_cast<std::uint16_t>(packed[0] & 0xffffu),
        static_cast<std::uint16_t>(packed[0] >> 16),
        static_cast<std::uint16_t>(packed[1] & 0xffffu),
        static_cast<std::uint16_t>(packed[1] >> 16),
    };
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
      const std::uint16_t word = words[g];
      const float w0 = dev_decode_e2m1(static_cast<std::uint8_t>(word & 0x0fu)) * sc;
      const float w1 = dev_decode_e2m1(static_cast<std::uint8_t>((word >> 4) & 0x0fu)) * sc;
      const float w2 = dev_decode_e2m1(static_cast<std::uint8_t>((word >> 8) & 0x0fu)) * sc;
      const float w3 = dev_decode_e2m1(static_cast<std::uint8_t>((word >> 12) & 0x0fu)) * sc;
      const int k_off = (b << 4) + (g << 2);
      if (tm == TM) {
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
          const std::size_t x_off = static_cast<std::size_t>(m0 + r) * static_cast<std::size_t>(in_features) + k_off;
          float4 xv;
          if constexpr (kAligned16) {
            xv = *reinterpret_cast<const float4*>(x + x_off);
          } else {
            xv = make_float4(x[x_off], x[x_off + 1], x[x_off + 2], x[x_off + 3]);
          }
          acc[r] = fmaf(xv.x, w0, acc[r]);
          acc[r] = fmaf(xv.y, w1, acc[r]);
          acc[r] = fmaf(xv.z, w2, acc[r]);
          acc[r] = fmaf(xv.w, w3, acc[r]);
        }
      } else {
        for (int r = 0; r < tm; ++r) {
          const std::size_t x_off = static_cast<std::size_t>(m0 + r) * static_cast<std::size_t>(in_features) + k_off;
          float4 xv;
          if constexpr (kAligned16) {
            xv = *reinterpret_cast<const float4*>(x + x_off);
          } else {
            xv = make_float4(x[x_off], x[x_off + 1], x[x_off + 2], x[x_off + 3]);
          }
          acc[r] = fmaf(xv.x, w0, acc[r]);
          acc[r] = fmaf(xv.y, w1, acc[r]);
          acc[r] = fmaf(xv.z, w2, acc[r]);
          acc[r] = fmaf(xv.w, w3, acc[r]);
        }
      }
    }
  }
  for (int r = 0; r < tm; ++r) {
    const float s = warp_sum(acc[r]);
    if (lane == 0) {
      out[static_cast<std::size_t>(m0 + r) * static_cast<std::size_t>(out_features) + n] = s;
    }
  }
}

template <int TM, bool kVec4, bool kAligned16>
__global__ void fp8_linear_f32_batched_kernel(const float* __restrict__ x, const std::uint8_t* __restrict__ w_e4m3,
                                              float weight_scale, float input_scale, int m, int out_features,
                                              int in_features, float* __restrict__ out) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int m0 = blockIdx.y * TM;
  const int n = blockIdx.x * 4 + warp_in_block;
  if (m0 >= m || n >= out_features) return;
  const int tm = min(TM, m - m0);

  const std::size_t w_base = static_cast<std::size_t>(n) * static_cast<std::size_t>(in_features);
  const bool calibrated = (input_scale > 0.0f);
  const float inv_in = calibrated ? (1.0f / input_scale) : 1.0f;

  float acc[TM]{};
  if constexpr (kVec4) {
    const int k4s = in_features >> 2;
    for (int k4 = lane; k4 < k4s; k4 += 32) {
      std::uint8_t wb[4]{};
      std::memcpy(wb, w_e4m3 + w_base + static_cast<std::size_t>(k4 << 2), 4);
      const float w0 = dev_decode_e4m3fn(wb[0]);
      const float w1 = dev_decode_e4m3fn(wb[1]);
      const float w2 = dev_decode_e4m3fn(wb[2]);
      const float w3 = dev_decode_e4m3fn(wb[3]);
      const int k_off = k4 << 2;
      if (tm == TM) {
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
          const std::size_t x_off = static_cast<std::size_t>(m0 + r) * static_cast<std::size_t>(in_features) + k_off;
          float4 xv;
          if constexpr (kAligned16) {
            xv = *reinterpret_cast<const float4*>(x + x_off);
          } else {
            xv = make_float4(x[x_off], x[x_off + 1], x[x_off + 2], x[x_off + 3]);
          }
          if (calibrated) {
            xv.x = dev_decode_e4m3fn(dev_quantize_e4m3fn(xv.x * inv_in)) * input_scale;
            xv.y = dev_decode_e4m3fn(dev_quantize_e4m3fn(xv.y * inv_in)) * input_scale;
            xv.z = dev_decode_e4m3fn(dev_quantize_e4m3fn(xv.z * inv_in)) * input_scale;
            xv.w = dev_decode_e4m3fn(dev_quantize_e4m3fn(xv.w * inv_in)) * input_scale;
          }
          acc[r] = fmaf(xv.x, w0, acc[r]);
          acc[r] = fmaf(xv.y, w1, acc[r]);
          acc[r] = fmaf(xv.z, w2, acc[r]);
          acc[r] = fmaf(xv.w, w3, acc[r]);
        }
      } else {
        for (int r = 0; r < tm; ++r) {
          const std::size_t x_off = static_cast<std::size_t>(m0 + r) * static_cast<std::size_t>(in_features) + k_off;
          float4 xv;
          if constexpr (kAligned16) {
            xv = *reinterpret_cast<const float4*>(x + x_off);
          } else {
            xv = make_float4(x[x_off], x[x_off + 1], x[x_off + 2], x[x_off + 3]);
          }
          if (calibrated) {
            xv.x = dev_decode_e4m3fn(dev_quantize_e4m3fn(xv.x * inv_in)) * input_scale;
            xv.y = dev_decode_e4m3fn(dev_quantize_e4m3fn(xv.y * inv_in)) * input_scale;
            xv.z = dev_decode_e4m3fn(dev_quantize_e4m3fn(xv.z * inv_in)) * input_scale;
            xv.w = dev_decode_e4m3fn(dev_quantize_e4m3fn(xv.w * inv_in)) * input_scale;
          }
          acc[r] = fmaf(xv.x, w0, acc[r]);
          acc[r] = fmaf(xv.y, w1, acc[r]);
          acc[r] = fmaf(xv.z, w2, acc[r]);
          acc[r] = fmaf(xv.w, w3, acc[r]);
        }
      }
    }
  } else {
    for (int k = lane; k < in_features; k += 32) {
      const float wv = dev_decode_e4m3fn(w_e4m3[w_base + k]);
      for (int r = 0; r < tm; ++r) {
        float xv = x[static_cast<std::size_t>(m0 + r) * static_cast<std::size_t>(in_features) + k];
        if (calibrated) xv = dev_decode_e4m3fn(dev_quantize_e4m3fn(xv * inv_in)) * input_scale;
        acc[r] = fmaf(xv, wv, acc[r]);
      }
    }
  }
  for (int r = 0; r < tm; ++r) {
    const float s = warp_sum(acc[r]);
    if (lane == 0) {
      out[static_cast<std::size_t>(m0 + r) * static_cast<std::size_t>(out_features) + n] = s * weight_scale;
    }
  }
}

template <int TM, bool kVec4, bool kAligned16>
__global__ void bf16_linear_f32_batched_kernel(const float* __restrict__ x, const std::uint16_t* __restrict__ w_bf16,
                                               int m, int out_features, int in_features, float* __restrict__ out) {
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int m0 = blockIdx.y * TM;
  const int n = blockIdx.x * 4 + warp_in_block;
  if (m0 >= m || n >= out_features) return;
  const int tm = min(TM, m - m0);

  const std::size_t w_base = static_cast<std::size_t>(n) * static_cast<std::size_t>(in_features);
  float acc[TM]{};
  if constexpr (kVec4) {
    const int k4s = in_features >> 2;
    for (int k4 = lane; k4 < k4s; k4 += 32) {
      std::uint16_t wb[4]{};
      std::memcpy(wb, w_bf16 + w_base + static_cast<std::size_t>(k4 << 2), 8);
      const float w0 = dev_bf16_to_f32(wb[0]);
      const float w1 = dev_bf16_to_f32(wb[1]);
      const float w2 = dev_bf16_to_f32(wb[2]);
      const float w3 = dev_bf16_to_f32(wb[3]);
      const int k_off = k4 << 2;
      if (tm == TM) {
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
          const std::size_t x_off = static_cast<std::size_t>(m0 + r) * static_cast<std::size_t>(in_features) + k_off;
          float4 xv;
          if constexpr (kAligned16) {
            xv = *reinterpret_cast<const float4*>(x + x_off);
          } else {
            xv = make_float4(x[x_off], x[x_off + 1], x[x_off + 2], x[x_off + 3]);
          }
          acc[r] = fmaf(xv.x, w0, acc[r]);
          acc[r] = fmaf(xv.y, w1, acc[r]);
          acc[r] = fmaf(xv.z, w2, acc[r]);
          acc[r] = fmaf(xv.w, w3, acc[r]);
        }
      } else {
        for (int r = 0; r < tm; ++r) {
          const std::size_t x_off = static_cast<std::size_t>(m0 + r) * static_cast<std::size_t>(in_features) + k_off;
          float4 xv;
          if constexpr (kAligned16) {
            xv = *reinterpret_cast<const float4*>(x + x_off);
          } else {
            xv = make_float4(x[x_off], x[x_off + 1], x[x_off + 2], x[x_off + 3]);
          }
          acc[r] = fmaf(xv.x, w0, acc[r]);
          acc[r] = fmaf(xv.y, w1, acc[r]);
          acc[r] = fmaf(xv.z, w2, acc[r]);
          acc[r] = fmaf(xv.w, w3, acc[r]);
        }
      }
    }
  } else {
    for (int k = lane; k < in_features; k += 32) {
      const float wv = dev_bf16_to_f32(w_bf16[w_base + k]);
      for (int r = 0; r < tm; ++r) {
        const float xv = x[static_cast<std::size_t>(m0 + r) * static_cast<std::size_t>(in_features) + k];
        acc[r] = fmaf(xv, wv, acc[r]);
      }
    }
  }
  for (int r = 0; r < tm; ++r) {
    const float s = warp_sum(acc[r]);
    if (lane == 0) {
      out[static_cast<std::size_t>(m0 + r) * static_cast<std::size_t>(out_features) + n] = s;
    }
  }
}

__global__ void embed_lookup_f32_kernel(const std::uint16_t* __restrict__ embed_table,
                                        const std::int32_t* __restrict__ token_ids, int seq_len, int hidden_size,
                                        int vocab_size, float* __restrict__ out_hidden) {
  const int row = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= seq_len || d >= hidden_size) return;
  const std::int32_t tid = token_ids[row];
  const float val = (tid >= 0 && tid < vocab_size)
                        ? dev_bf16_to_f32(embed_table[static_cast<std::size_t>(tid) * hidden_size + d])
                        : 0.0f;
  out_hidden[static_cast<std::size_t>(row) * hidden_size + d] = val;
}

__global__ void gather_rows_f32_kernel(const float* __restrict__ src, const int* __restrict__ row_indices,
                                       int num_rows, int hidden_size, float* __restrict__ dst) {
  const int row = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows || d >= hidden_size) return;
  const int src_row = row_indices[row];
  dst[static_cast<std::size_t>(row) * hidden_size + d] =
      src[static_cast<std::size_t>(src_row) * hidden_size + d];
}

__global__ void swiglu_mul_f32_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                      float* __restrict__ out, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = dev_silu(gate[i]) * up[i];
}

__global__ void add_residual_f32_kernel(float* __restrict__ hidden, const float* __restrict__ residual, std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) hidden[i] += residual[i];
}

__global__ void split_qkv_sep_f32_kernel(const float* __restrict__ mixed, float* __restrict__ q_sep,
                                         float* __restrict__ k_sep, float* __restrict__ v_sep, int seq_len, int kd,
                                         int vd) {
  const int cd = 2 * kd + vd;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  if (t >= seq_len || idx >= cd) return;
  const float val = mixed[static_cast<std::size_t>(t * cd + idx)];
  if (idx < kd) {
    q_sep[static_cast<std::size_t>(t * kd + idx)] = val;
  } else if (idx < 2 * kd) {
    k_sep[static_cast<std::size_t>(t * kd + (idx - kd))] = val;
  } else {
    v_sep[static_cast<std::size_t>(t * vd + (idx - 2 * kd))] = val;
  }
}

__global__ void gdn_normalize_qk_packed_f32_kernel(const float* __restrict__ conv_out_packed, int seq_len,
                                                   int num_k_heads, int num_v_heads, int k_dim, int v_dim,
                                                   float* __restrict__ q_norm, float* __restrict__ k_norm) {
  const int kh = blockIdx.x;
  const int t = blockIdx.y;
  if (t >= seq_len || kh >= num_k_heads) return;

  const int kd_total = num_k_heads * k_dim;
  const int vd_total = num_v_heads * v_dim;
  const int cd = 2 * kd_total + vd_total;
  const std::size_t q_src =
      static_cast<std::size_t>(t) * static_cast<std::size_t>(cd) + static_cast<std::size_t>(kh * k_dim);
  const std::size_t k_src = q_src + static_cast<std::size_t>(kd_total);
  const std::size_t dst_base =
      (static_cast<std::size_t>(t) * static_cast<std::size_t>(num_k_heads) + kh) * static_cast<std::size_t>(k_dim);

  float qsq = 0.0f, ksq = 0.0f;
  for (int d = threadIdx.x; d < k_dim; d += blockDim.x) {
    const float qv = conv_out_packed[q_src + d];
    const float kv = conv_out_packed[k_src + d];
    qsq += qv * qv;
    ksq += kv * kv;
  }
  qsq = warp_sum(qsq);
  ksq = warp_sum(ksq);
  __shared__ float rq[4], rk[4];
  if ((threadIdx.x & 31) == 0) {
    rq[threadIdx.x >> 5] = qsq;
    rk[threadIdx.x >> 5] = ksq;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    float qt = 0.0f, kt = 0.0f;
    const int wcount = blockDim.x >> 5;
    for (int i = 0; i < wcount; ++i) {
      qt += rq[i];
      kt += rk[i];
    }
    rq[0] = rsqrtf(qt + 1e-6f) * rsqrtf(static_cast<float>(k_dim));
    rk[0] = rsqrtf(kt + 1e-6f);
  }
  __syncthreads();
  const float q_inv = rq[0];
  const float k_inv = rk[0];
  for (int d = threadIdx.x; d < k_dim; d += blockDim.x) {
    q_norm[dst_base + d] = conv_out_packed[q_src + d] * q_inv;
    k_norm[dst_base + d] = conv_out_packed[k_src + d] * k_inv;
  }
}

template <bool kUseSmemState>
__global__ void gdn_recurrent_prefill_exact_f32_kernel(
    const float* __restrict__ q_norm, const float* __restrict__ k_norm, const float* __restrict__ conv_out_packed,
    const float* __restrict__ z_seq, const float* __restrict__ a_seq, const float* __restrict__ b_seq,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias, const float* __restrict__ norm_weight,
    float rms_eps, int seq_len, int num_k_heads, int num_v_heads, int k_dim, int v_dim,
    float* __restrict__ recurrent_state, float* __restrict__ out_seq) {
  const int vh = blockIdx.x;
  if (vh >= num_v_heads) return;
  const int repeat_factor = num_v_heads / num_k_heads;
  const int kh = vh / repeat_factor;
  const int kd_total = num_k_heads * k_dim;
  const int vd_total = num_v_heads * v_dim;
  const int cd = 2 * kd_total + vd_total;

  __shared__ float sqh[256];
  __shared__ float skh[256];
  __shared__ float s_core[512];
  __shared__ float s_red[8];
  extern __shared__ float s_state[];

  const std::size_t state_elems = static_cast<std::size_t>(k_dim) * static_cast<std::size_t>(v_dim);
  float* g_state = recurrent_state + static_cast<std::size_t>(vh) * state_elems;
  float* state = kUseSmemState ? s_state : g_state;
  if constexpr (kUseSmemState) {
    for (std::size_t idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      s_state[idx] = g_state[idx];
    }
    __syncthreads();
  }

  const float neg_exp_alog = -expf(a_log[vh]);
  const float dt_b = dt_bias[vh];

  for (int t = 0; t < seq_len; ++t) {
    const std::size_t qk_off =
        (static_cast<std::size_t>(t) * static_cast<std::size_t>(num_k_heads) + kh) * static_cast<std::size_t>(k_dim);
    const float* qh_g = q_norm + qk_off;
    const float* kh_g = k_norm + qk_off;
    for (int d = threadIdx.x; d < k_dim; d += blockDim.x) {
      sqh[d] = qh_g[d];
      skh[d] = kh_g[d];
    }
    __syncthreads();

    const float beta = 1.0f / (1.0f + expf(-b_seq[static_cast<std::size_t>(t) * static_cast<std::size_t>(num_v_heads) + vh]));
    const float g =
        neg_exp_alog * dev_softplus(a_seq[static_cast<std::size_t>(t) * static_cast<std::size_t>(num_v_heads) + vh] + dt_b);
    const float decay = expf(g);

    const std::size_t v_in_off =
        static_cast<std::size_t>(t) * static_cast<std::size_t>(cd) + static_cast<std::size_t>(2 * kd_total + vh * v_dim);
    const std::size_t zv_off =
        static_cast<std::size_t>(t) * static_cast<std::size_t>(vd_total) + static_cast<std::size_t>(vh * v_dim);

    float sq_sum = 0.0f;
    for (int vd = threadIdx.x; vd < v_dim; vd += blockDim.x) {
      float kv_mem = 0.0f;
      for (int kd = 0; kd < k_dim; ++kd) {
        kv_mem += (state[static_cast<std::size_t>(kd) * static_cast<std::size_t>(v_dim) + vd] * decay) * skh[kd];
      }
      const float delta = (conv_out_packed[v_in_off + vd] - kv_mem) * beta;
      float core_vd = 0.0f;
      for (int kd = 0; kd < k_dim; ++kd) {
        const std::size_t idx = static_cast<std::size_t>(kd) * static_cast<std::size_t>(v_dim) + vd;
        const float ns = state[idx] * decay + skh[kd] * delta;
        state[idx] = ns;
        core_vd += ns * sqh[kd];
      }
      s_core[vd] = core_vd;
      sq_sum += core_vd * core_vd;
    }

    sq_sum = warp_sum(sq_sum);
    if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = sq_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
      float tot = 0.0f;
      const int wcount = blockDim.x >> 5;
      for (int i = 0; i < wcount; ++i) tot += s_red[i];
      s_red[0] = rsqrtf(tot / static_cast<float>(v_dim) + rms_eps);
    }
    __syncthreads();

    const float inv_rms = s_red[0];
    for (int vd = threadIdx.x; vd < v_dim; vd += blockDim.x) {
      const float zv = z_seq[zv_off + vd];
      out_seq[zv_off + vd] = norm_weight[vd] * (s_core[vd] * inv_rms) * dev_silu(zv);
    }
    if (t + 1 < seq_len) __syncthreads();
  }

  if constexpr (kUseSmemState) {
    __syncthreads();
    for (std::size_t idx = threadIdx.x; idx < state_elems; idx += blockDim.x) {
      g_state[idx] = s_state[idx];
    }
  }
}

}  // namespace

void qwen38_cuda_f32_to_bf16(const float* src, __nv_bfloat16* dst, std::size_t n, cudaStream_t stream) {
  if (n == 0) return;
  f32_to_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(src, reinterpret_cast<std::uint16_t*>(dst), n);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_bf16_to_f32(const __nv_bfloat16* src, float* dst, std::size_t n, cudaStream_t stream) {
  if (n == 0) return;
  bf16_to_f32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(reinterpret_cast<const std::uint16_t*>(src), dst, n);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_nvfp4_linear_f32(const float* x, const std::uint8_t* w_packed, const std::uint8_t* block_scales_e4m3,
                                  float weight_scale_2, int m, int out_features, int in_features, float* out,
                                  cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features % 16 != 0) throw std::invalid_argument("nvfp4_linear bad shape");
  nvfp4_linear_kernel<float, float><<<dim3((out_features + 3) / 4, m), 128, 0, stream>>>(
      x, w_packed, block_scales_e4m3, weight_scale_2, m, out_features, in_features, out);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_nvfp4_linear_bf16(const __nv_bfloat16* x, const std::uint8_t* w_packed,
                                   const std::uint8_t* block_scales_e4m3, float weight_scale_2, int m, int out_features,
                                   int in_features, __nv_bfloat16* out, cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features % 16 != 0) throw std::invalid_argument("nvfp4_linear bad shape");
  nvfp4_linear_kernel<std::uint16_t, std::uint16_t><<<dim3((out_features + 3) / 4, m), 128, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(x), w_packed, block_scales_e4m3, weight_scale_2, m, out_features,
      in_features, reinterpret_cast<std::uint16_t*>(out));
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_fp8_linear_f32(const float* x, const std::uint8_t* w_e4m3, float weight_scale, float input_scale,
                                int m, int out_features, int in_features, float* out, cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features <= 0) throw std::invalid_argument("fp8_linear bad shape");
  fp8_linear_kernel<float><<<dim3((out_features + 3) / 4, m), 128, 0, stream>>>(
      x, w_e4m3, weight_scale, input_scale, m, out_features, in_features, out);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_fp8_linear_bf16(const __nv_bfloat16* x, const std::uint8_t* w_e4m3, float weight_scale,
                                 float input_scale, int m, int out_features, int in_features, __nv_bfloat16* out,
                                 cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features <= 0) throw std::invalid_argument("fp8_linear bad shape");
  fp8_linear_kernel<std::uint16_t><<<dim3((out_features + 3) / 4, m), 128, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(x), w_e4m3, weight_scale, input_scale, m, out_features, in_features,
      reinterpret_cast<std::uint16_t*>(out));
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_bf16_linear_f32(const float* x, const __nv_bfloat16* w_bf16, int m, int out_features, int in_features,
                                 float* out, cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features <= 0) throw std::invalid_argument("bf16_linear bad shape");
  bf16_linear_kernel<float, float><<<dim3((out_features + 3) / 4, m), 128, 0, stream>>>(
      x, reinterpret_cast<const std::uint16_t*>(w_bf16), m, out_features, in_features, out);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_bf16_linear_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w_bf16, int m, int out_features,
                                  int in_features, __nv_bfloat16* out, cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features <= 0) throw std::invalid_argument("bf16_linear bad shape");
  bf16_linear_kernel<std::uint16_t, std::uint16_t><<<dim3((out_features + 3) / 4, m), 128, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(x), reinterpret_cast<const std::uint16_t*>(w_bf16), m, out_features,
      in_features, reinterpret_cast<std::uint16_t*>(out));
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_rmsnorm_zero_centered_f32(const float* x, const __nv_bfloat16* weight_bf16, float eps, int m, int dim,
                                           float* out, cudaStream_t stream) {
  rmsnorm_zero_centered_kernel<float><<<m, 256, 0, stream>>>(
      x, reinterpret_cast<const std::uint16_t*>(weight_bf16), eps, m, dim, out);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_rmsnorm_zero_centered_bf16(const __nv_bfloat16* x, const __nv_bfloat16* weight_bf16, float eps, int m,
                                            int dim, __nv_bfloat16* out, cudaStream_t stream) {
  rmsnorm_zero_centered_kernel<std::uint16_t><<<m, 256, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(x), reinterpret_cast<const std::uint16_t*>(weight_bf16), eps, m, dim,
      reinterpret_cast<std::uint16_t*>(out));
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_rmsnorm_gated_silu_f32(const float* x, const float* gate_z, const float* weight, float eps, int m,
                                        int dim, float* out, cudaStream_t stream) {
  rmsnorm_gated_silu_kernel<float><<<m, 256, 0, stream>>>(x, gate_z, weight, eps, m, dim, out);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_rmsnorm_gated_silu_bf16(const __nv_bfloat16* x, const __nv_bfloat16* gate_z, const float* weight,
                                         float eps, int m, int dim, __nv_bfloat16* out, cudaStream_t stream) {
  rmsnorm_gated_silu_kernel<std::uint16_t><<<m, 256, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(x), reinterpret_cast<const std::uint16_t*>(gate_z), weight, eps, m, dim,
      reinterpret_cast<std::uint16_t*>(out));
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_partial_rope_neox_f32(float* qk, int m, int num_heads, int head_dim, int rotary_dim, int pos_offset,
                                       float rope_theta, cudaStream_t stream) {
  const int half = rotary_dim >> 1;
  partial_rope_neox_kernel<float><<<dim3((num_heads * half + 127) / 128, m), 128, 0, stream>>>(
      qk, m, num_heads, head_dim, rotary_dim, pos_offset, rope_theta);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_partial_rope_neox_bf16(__nv_bfloat16* qk, int m, int num_heads, int head_dim, int rotary_dim,
                                        int pos_offset, float rope_theta, cudaStream_t stream) {
  const int half = rotary_dim >> 1;
  partial_rope_neox_kernel<std::uint16_t><<<dim3((num_heads * half + 127) / 128, m), 128, 0, stream>>>(
      reinterpret_cast<std::uint16_t*>(qk), m, num_heads, head_dim, rotary_dim, pos_offset, rope_theta);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_causal_conv1d_silu_step_f32(const float* x_in, const float* conv_weight, int conv_dim, int kernel_size,
                                             float* conv_state, float* out, cudaStream_t stream) {
  qwen38_cuda_causal_conv1d_silu_prefill_f32(x_in, conv_weight, 1, conv_dim, kernel_size, conv_state, out, stream);
}

void qwen38_cuda_causal_conv1d_silu_step_bf16(const __nv_bfloat16* x_in, const float* conv_weight, int conv_dim,
                                              int kernel_size, float* conv_state, __nv_bfloat16* out,
                                              cudaStream_t stream) {
  qwen38_cuda_causal_conv1d_silu_prefill_bf16(x_in, conv_weight, 1, conv_dim, kernel_size, conv_state, out, stream);
}

void qwen38_cuda_causal_conv1d_silu_prefill_f32(const float* x_seq, const float* conv_weight, int seq_len, int conv_dim,
                                                int kernel_size, float* conv_state, float* out_seq,
                                                cudaStream_t stream) {
  if (kernel_size < 1 || kernel_size > 8) throw std::invalid_argument("causal_conv1d_silu unsupported kernel_size");
  causal_conv1d_silu_prefill_kernel<float><<<(conv_dim + 255) / 256, 256, 0, stream>>>(
      x_seq, conv_weight, seq_len, conv_dim, kernel_size, conv_state, out_seq);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_causal_conv1d_silu_prefill_bf16(const __nv_bfloat16* x_seq, const float* conv_weight, int seq_len,
                                                 int conv_dim, int kernel_size, float* conv_state,
                                                 __nv_bfloat16* out_seq, cudaStream_t stream) {
  if (kernel_size < 1 || kernel_size > 8) throw std::invalid_argument("causal_conv1d_silu unsupported kernel_size");
  causal_conv1d_silu_prefill_kernel<std::uint16_t><<<(conv_dim + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(x_seq), conv_weight, seq_len, conv_dim, kernel_size, conv_state,
      reinterpret_cast<std::uint16_t*>(out_seq));
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_gdn_recurrent_step_f32(const float* q_kheads, const float* k_kheads, const float* v_vheads,
                                        const float* z_vheads, const float* a_vheads, const float* b_vheads,
                                        const float* a_log, const float* dt_bias, const float* norm_weight,
                                        float rms_eps, int num_k_heads, int num_v_heads, int k_dim, int v_dim,
                                        float* recurrent_state, float* qk_norm_scratch, float* out_vdim,
                                        cudaStream_t stream) {
  dispatch_gdn_step<float>(q_kheads, k_kheads, v_vheads, z_vheads, a_vheads, b_vheads, a_log, dt_bias, norm_weight,
                           rms_eps, num_k_heads, num_v_heads, k_dim, v_dim, recurrent_state, qk_norm_scratch, out_vdim,
                           stream);
}

void qwen38_cuda_gdn_recurrent_step_bf16(const __nv_bfloat16* q_kheads, const __nv_bfloat16* k_kheads,
                                         const __nv_bfloat16* v_vheads, const __nv_bfloat16* z_vheads,
                                         const __nv_bfloat16* a_vheads, const __nv_bfloat16* b_vheads,
                                         const float* a_log, const float* dt_bias, const float* norm_weight,
                                         float rms_eps, int num_k_heads, int num_v_heads, int k_dim, int v_dim,
                                         float* recurrent_state, float* qk_norm_scratch, __nv_bfloat16* out_vdim,
                                         cudaStream_t stream) {
  dispatch_gdn_step<std::uint16_t>(
      reinterpret_cast<const std::uint16_t*>(q_kheads), reinterpret_cast<const std::uint16_t*>(k_kheads),
      reinterpret_cast<const std::uint16_t*>(v_vheads), reinterpret_cast<const std::uint16_t*>(z_vheads),
      reinterpret_cast<const std::uint16_t*>(a_vheads), reinterpret_cast<const std::uint16_t*>(b_vheads), a_log,
      dt_bias, norm_weight, rms_eps, num_k_heads, num_v_heads, k_dim, v_dim, recurrent_state, qk_norm_scratch,
      reinterpret_cast<std::uint16_t*>(out_vdim), stream);
}

std::size_t qwen38_cuda_gdn_chunk_scratch_floats(int seq_len, int chunk_size, int num_k_heads, int num_v_heads,
                                                 int k_dim, int v_dim) noexcept {
  const std::size_t qk = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(num_k_heads * k_dim) * 2ull;
  const std::size_t w =
      static_cast<std::size_t>(num_v_heads) * static_cast<std::size_t>(chunk_size) * static_cast<std::size_t>(k_dim);
  const std::size_t vn =
      static_cast<std::size_t>(num_v_heads) * static_cast<std::size_t>(chunk_size) * static_cast<std::size_t>(v_dim);
  return qk + w + vn;
}

void qwen38_cuda_gdn_chunk_prefill_f32(const float* q_seq, const float* k_seq, const float* v_seq, const float* z_seq,
                                       const float* a_seq, const float* b_seq, const float* a_log, const float* dt_bias,
                                       const float* norm_weight, float rms_eps, int seq_len, int chunk_size,
                                       int num_k_heads, int num_v_heads, int k_dim, int v_dim, float* recurrent_state,
                                       float* scratch, float* out_seq, cudaStream_t stream) {
  dispatch_gdn_chunk<float>(q_seq, k_seq, v_seq, z_seq, a_seq, b_seq, a_log, dt_bias, norm_weight, rms_eps, seq_len,
                            chunk_size, num_k_heads, num_v_heads, k_dim, v_dim, recurrent_state, scratch, out_seq,
                            stream);
}

void qwen38_cuda_gdn_chunk_prefill_bf16(const __nv_bfloat16* q_seq, const __nv_bfloat16* k_seq,
                                        const __nv_bfloat16* v_seq, const __nv_bfloat16* z_seq,
                                        const __nv_bfloat16* a_seq, const __nv_bfloat16* b_seq, const float* a_log,
                                        const float* dt_bias, const float* norm_weight, float rms_eps, int seq_len,
                                        int chunk_size, int num_k_heads, int num_v_heads, int k_dim, int v_dim,
                                        float* recurrent_state, float* scratch, __nv_bfloat16* out_seq,
                                        cudaStream_t stream) {
  dispatch_gdn_chunk<std::uint16_t>(
      reinterpret_cast<const std::uint16_t*>(q_seq), reinterpret_cast<const std::uint16_t*>(k_seq),
      reinterpret_cast<const std::uint16_t*>(v_seq), reinterpret_cast<const std::uint16_t*>(z_seq),
      reinterpret_cast<const std::uint16_t*>(a_seq), reinterpret_cast<const std::uint16_t*>(b_seq), a_log, dt_bias,
      norm_weight, rms_eps, seq_len, chunk_size, num_k_heads, num_v_heads, k_dim, v_dim, recurrent_state, scratch,
      reinterpret_cast<std::uint16_t*>(out_seq), stream);
}

void qwen38_cuda_full_attn_gated_f32(const float* q_with_gate, const float* k_raw, const float* v_in,
                                     const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w, float rms_eps,
                                     float rope_theta, int seq_len, int pos_offset, int num_heads, int num_kv_heads,
                                     int head_dim, int rotary_dim, float* k_cache, float* v_cache, int max_cache_len,
                                     float* q_scratch, float* gate_scratch, float* out_attn, cudaStream_t stream) {
  dispatch_full_attn_gated<float>(
      q_with_gate, k_raw, v_in, reinterpret_cast<const std::uint16_t*>(q_norm_w),
      reinterpret_cast<const std::uint16_t*>(k_norm_w), rms_eps, rope_theta, seq_len, pos_offset, num_heads,
      num_kv_heads, head_dim, rotary_dim, k_cache, v_cache, max_cache_len, q_scratch, gate_scratch, out_attn, stream);
}

void qwen38_cuda_full_attn_gated_bf16(const __nv_bfloat16* q_with_gate, const __nv_bfloat16* k_raw,
                                      const __nv_bfloat16* v_in, const __nv_bfloat16* q_norm_w,
                                      const __nv_bfloat16* k_norm_w, float rms_eps, float rope_theta, int seq_len,
                                      int pos_offset, int num_heads, int num_kv_heads, int head_dim, int rotary_dim,
                                      __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int max_cache_len,
                                      __nv_bfloat16* q_scratch, __nv_bfloat16* gate_scratch, __nv_bfloat16* out_attn,
                                      cudaStream_t stream) {
  dispatch_full_attn_gated<std::uint16_t>(
      reinterpret_cast<const std::uint16_t*>(q_with_gate), reinterpret_cast<const std::uint16_t*>(k_raw),
      reinterpret_cast<const std::uint16_t*>(v_in), reinterpret_cast<const std::uint16_t*>(q_norm_w),
      reinterpret_cast<const std::uint16_t*>(k_norm_w), rms_eps, rope_theta, seq_len, pos_offset, num_heads,
      num_kv_heads, head_dim, rotary_dim, reinterpret_cast<std::uint16_t*>(k_cache),
      reinterpret_cast<std::uint16_t*>(v_cache), max_cache_len, reinterpret_cast<std::uint16_t*>(q_scratch),
      reinterpret_cast<std::uint16_t*>(gate_scratch), reinterpret_cast<std::uint16_t*>(out_attn), stream);
}

void qwen38_cuda_swiglu_mul_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                                 std::size_t n, cudaStream_t stream) {
  if (n == 0) return;
  swiglu_mul_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(reinterpret_cast<const std::uint16_t*>(gate),
                                                               reinterpret_cast<const std::uint16_t*>(up),
                                                               reinterpret_cast<std::uint16_t*>(out), n);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_add_residual_bf16(__nv_bfloat16* hidden, const __nv_bfloat16* residual, std::size_t n,
                                   cudaStream_t stream) {
  if (n == 0) return;
  add_residual_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(reinterpret_cast<std::uint16_t*>(hidden),
                                                                 reinterpret_cast<const std::uint16_t*>(residual), n);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_split_qkv_sep_bf16(const __nv_bfloat16* mixed, __nv_bfloat16* q_sep, __nv_bfloat16* k_sep,
                                    __nv_bfloat16* v_sep, int seq_len, int kd, int vd, cudaStream_t stream) {
  if (seq_len <= 0 || kd <= 0 || vd <= 0) throw std::invalid_argument("split_qkv_sep bad shape");
  const int cd = 2 * kd + vd;
  split_qkv_sep_bf16_kernel<<<dim3((cd + 255) / 256, seq_len), 256, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(mixed), reinterpret_cast<std::uint16_t*>(q_sep),
      reinterpret_cast<std::uint16_t*>(k_sep), reinterpret_cast<std::uint16_t*>(v_sep), seq_len, kd, vd);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_dequant_nvfp4_to_bf16(const std::uint8_t* w_packed, const std::uint8_t* block_scales_e4m3, int n_rows,
                                       int in_features, __nv_bfloat16* out_bf16, cudaStream_t stream) {
  if (n_rows <= 0 || in_features <= 0 || in_features % 16 != 0) {
    throw std::invalid_argument("dequant_nvfp4_to_bf16 bad shape");
  }
  const int blocks = in_features >> 4;
  dequant_nvfp4_to_bf16_kernel<<<dim3((blocks + 255) / 256, n_rows), 256, 0, stream>>>(
      w_packed, block_scales_e4m3, n_rows, in_features, reinterpret_cast<std::uint16_t*>(out_bf16));
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_dequant_fp8_to_bf16(const std::uint8_t* w_e4m3, int n_rows, int in_features, __nv_bfloat16* out_bf16,
                                     cudaStream_t stream) {
  if (n_rows <= 0 || in_features <= 0) throw std::invalid_argument("dequant_fp8_to_bf16 bad shape");
  const std::size_t n = static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(in_features);
  dequant_fp8_to_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      w_e4m3, n, reinterpret_cast<std::uint16_t*>(out_bf16));
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_quant_fp8_act_to_bf16(const __nv_bfloat16* x_in, float input_scale, std::size_t n,
                                       __nv_bfloat16* x_out, cudaStream_t stream) {
  if (n == 0) return;
  if (!(input_scale > 0.0f)) throw std::invalid_argument("quant_fp8_act_to_bf16 requires positive input_scale");
  quant_fp8_act_to_bf16_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(x_in), 1.0f / input_scale, n, reinterpret_cast<std::uint16_t*>(x_out));
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_embed_lookup_bf16(const __nv_bfloat16* embed_table, const std::int32_t* token_ids, int seq_len,
                                   int hidden_size, int vocab_size, __nv_bfloat16* out_hidden, cudaStream_t stream) {
  if (seq_len <= 0 || hidden_size <= 0 || vocab_size <= 0) throw std::invalid_argument("embed_lookup bad shape");
  embed_lookup_bf16_kernel<<<dim3((hidden_size + 255) / 256, seq_len), 256, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(embed_table), token_ids, seq_len, hidden_size, vocab_size,
      reinterpret_cast<std::uint16_t*>(out_hidden));
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_gather_rows_bf16(const __nv_bfloat16* src, const int* row_indices, int num_rows, int hidden_size,
                                  __nv_bfloat16* dst, cudaStream_t stream) {
  if (num_rows <= 0 || hidden_size <= 0) return;
  gather_rows_bf16_kernel<<<dim3((hidden_size + 255) / 256, num_rows), 256, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(src), row_indices, num_rows, hidden_size,
      reinterpret_cast<std::uint16_t*>(dst));
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_nvfp4_linear_bf16_to_f32(const __nv_bfloat16* x, const std::uint8_t* w_packed,
                                          const std::uint8_t* block_scales_e4m3, float weight_scale_2, int m,
                                          int out_features, int in_features, float* out_f32, cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features % 16 != 0) throw std::invalid_argument("nvfp4_bf16_f32 bad shape");
  nvfp4_linear_kernel<std::uint16_t, float><<<dim3((out_features + 3) / 4, m), 128, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(x), w_packed, block_scales_e4m3, weight_scale_2, m, out_features,
      in_features, out_f32);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_bf16_linear_bf16_to_f32(const __nv_bfloat16* x, const __nv_bfloat16* w_bf16, int m, int out_features,
                                         int in_features, float* out_f32, cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features <= 0) throw std::invalid_argument("bf16_bf16_f32 bad shape");
  bf16_linear_kernel<std::uint16_t, float><<<dim3((out_features + 3) / 4, m), 128, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(x), reinterpret_cast<const std::uint16_t*>(w_bf16), m, out_features,
      in_features, out_f32);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_quant_fp8_act_f32(const float* x_in, float input_scale, std::size_t n, float* x_out,
                                   cudaStream_t stream) {
  if (n == 0) return;
  if (!(input_scale > 0.0f)) throw std::invalid_argument("quant_fp8_act_f32 requires positive input_scale");
  quant_fp8_act_f32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(x_in, 1.0f / input_scale, input_scale, n, x_out);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_nvfp4_linear_f32_batched(const float* x, const std::uint8_t* w_packed,
                                          const std::uint8_t* block_scales_e4m3, float weight_scale_2, int m,
                                          int out_features, int in_features, float* out, cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features <= 0 || in_features % 16 != 0) {
    throw std::invalid_argument("nvfp4_linear_f32_batched bad shape");
  }
  const bool aligned16 = ((reinterpret_cast<std::uintptr_t>(x) & 15u) == 0u);
  const int gx = (out_features + 3) / 4;
  if (m == 1) {
    if (aligned16) {
      nvfp4_linear_f32_batched_kernel<1, true><<<dim3(gx, 1), 128, 0, stream>>>(
          x, w_packed, block_scales_e4m3, weight_scale_2, m, out_features, in_features, out);
    } else {
      nvfp4_linear_f32_batched_kernel<1, false><<<dim3(gx, 1), 128, 0, stream>>>(
          x, w_packed, block_scales_e4m3, weight_scale_2, m, out_features, in_features, out);
    }
  } else {
    const int gy = (m + 15) / 16;
    if (aligned16) {
      nvfp4_linear_f32_batched_kernel<16, true><<<dim3(gx, gy), 128, 0, stream>>>(
          x, w_packed, block_scales_e4m3, weight_scale_2, m, out_features, in_features, out);
    } else {
      nvfp4_linear_f32_batched_kernel<16, false><<<dim3(gx, gy), 128, 0, stream>>>(
          x, w_packed, block_scales_e4m3, weight_scale_2, m, out_features, in_features, out);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_fp8_linear_f32_batched(const float* x, const std::uint8_t* w_e4m3, float weight_scale,
                                        float input_scale, int m, int out_features, int in_features, float* out,
                                        cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features <= 0) {
    throw std::invalid_argument("fp8_linear_f32_batched bad shape");
  }
  const bool vec4 = ((in_features & 3) == 0);
  const bool aligned16 = vec4 && ((reinterpret_cast<std::uintptr_t>(x) & 15u) == 0u);
  const int gx = (out_features + 3) / 4;
  if (m == 1) {
    if (vec4 && aligned16) {
      fp8_linear_f32_batched_kernel<1, true, true><<<dim3(gx, 1), 128, 0, stream>>>(
          x, w_e4m3, weight_scale, input_scale, m, out_features, in_features, out);
    } else if (vec4) {
      fp8_linear_f32_batched_kernel<1, true, false><<<dim3(gx, 1), 128, 0, stream>>>(
          x, w_e4m3, weight_scale, input_scale, m, out_features, in_features, out);
    } else {
      fp8_linear_f32_batched_kernel<1, false, false><<<dim3(gx, 1), 128, 0, stream>>>(
          x, w_e4m3, weight_scale, input_scale, m, out_features, in_features, out);
    }
  } else {
    const int gy = (m + 15) / 16;
    if (vec4 && aligned16) {
      fp8_linear_f32_batched_kernel<16, true, true><<<dim3(gx, gy), 128, 0, stream>>>(
          x, w_e4m3, weight_scale, input_scale, m, out_features, in_features, out);
    } else if (vec4) {
      fp8_linear_f32_batched_kernel<16, true, false><<<dim3(gx, gy), 128, 0, stream>>>(
          x, w_e4m3, weight_scale, input_scale, m, out_features, in_features, out);
    } else {
      fp8_linear_f32_batched_kernel<16, false, false><<<dim3(gx, gy), 128, 0, stream>>>(
          x, w_e4m3, weight_scale, input_scale, m, out_features, in_features, out);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_bf16_linear_f32_batched(const float* x, const __nv_bfloat16* w_bf16, int m, int out_features,
                                         int in_features, float* out, cudaStream_t stream) {
  if (m <= 0 || out_features <= 0 || in_features <= 0) {
    throw std::invalid_argument("bf16_linear_f32_batched bad shape");
  }
  const auto* w_u16 = reinterpret_cast<const std::uint16_t*>(w_bf16);
  const bool vec4 = ((in_features & 3) == 0);
  const bool aligned16 = vec4 && ((reinterpret_cast<std::uintptr_t>(x) & 15u) == 0u);
  const int gx = (out_features + 3) / 4;
  if (m == 1) {
    if (vec4 && aligned16) {
      bf16_linear_f32_batched_kernel<1, true, true><<<dim3(gx, 1), 128, 0, stream>>>(
          x, w_u16, m, out_features, in_features, out);
    } else if (vec4) {
      bf16_linear_f32_batched_kernel<1, true, false><<<dim3(gx, 1), 128, 0, stream>>>(
          x, w_u16, m, out_features, in_features, out);
    } else {
      bf16_linear_f32_batched_kernel<1, false, false><<<dim3(gx, 1), 128, 0, stream>>>(
          x, w_u16, m, out_features, in_features, out);
    }
  } else {
    const int gy = (m + 15) / 16;
    if (vec4 && aligned16) {
      bf16_linear_f32_batched_kernel<16, true, true><<<dim3(gx, gy), 128, 0, stream>>>(
          x, w_u16, m, out_features, in_features, out);
    } else if (vec4) {
      bf16_linear_f32_batched_kernel<16, true, false><<<dim3(gx, gy), 128, 0, stream>>>(
          x, w_u16, m, out_features, in_features, out);
    } else {
      bf16_linear_f32_batched_kernel<16, false, false><<<dim3(gx, gy), 128, 0, stream>>>(
          x, w_u16, m, out_features, in_features, out);
    }
  }
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_embed_lookup_f32(const __nv_bfloat16* embed_table, const std::int32_t* token_ids, int seq_len,
                                  int hidden_size, int vocab_size, float* out_hidden, cudaStream_t stream) {
  if (seq_len <= 0 || hidden_size <= 0 || vocab_size <= 0) throw std::invalid_argument("embed_lookup_f32 bad shape");
  embed_lookup_f32_kernel<<<dim3((hidden_size + 255) / 256, seq_len), 256, 0, stream>>>(
      reinterpret_cast<const std::uint16_t*>(embed_table), token_ids, seq_len, hidden_size, vocab_size, out_hidden);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_gather_rows_f32(const float* src, const int* row_indices, int num_rows, int hidden_size,
                                 float* dst, cudaStream_t stream) {
  if (num_rows <= 0 || hidden_size <= 0) return;
  gather_rows_f32_kernel<<<dim3((hidden_size + 255) / 256, num_rows), 256, 0, stream>>>(
      src, row_indices, num_rows, hidden_size, dst);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_swiglu_mul_f32(const float* gate, const float* up, float* out, std::size_t n, cudaStream_t stream) {
  if (n == 0) return;
  swiglu_mul_f32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(gate, up, out, n);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_add_residual_f32(float* hidden, const float* residual, std::size_t n, cudaStream_t stream) {
  if (n == 0) return;
  add_residual_f32_kernel<<<(n + 255) / 256, 256, 0, stream>>>(hidden, residual, n);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_split_qkv_sep_f32(const float* mixed, float* q_sep, float* k_sep, float* v_sep, int seq_len,
                                   int kd, int vd, cudaStream_t stream) {
  if (seq_len <= 0 || kd <= 0 || vd <= 0) throw std::invalid_argument("split_qkv_sep_f32 bad shape");
  const int cd = 2 * kd + vd;
  split_qkv_sep_f32_kernel<<<dim3((cd + 255) / 256, seq_len), 256, 0, stream>>>(
      mixed, q_sep, k_sep, v_sep, seq_len, kd, vd);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_full_attn_gated_f32_kvbf16(const float* q_with_gate, float* k_raw_inout, const float* v_in,
                                            const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
                                            float rms_eps, float rope_theta, int seq_len, int pos_offset,
                                            int num_heads, int num_kv_heads, int head_dim, int rotary_dim,
                                            __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int max_cache_len,
                                            float* q_scratch, float* gate_scratch, float* out_attn,
                                            cudaStream_t stream) {
  if (seq_len < 1 || pos_offset + seq_len > max_cache_len || head_dim > 256 || num_heads % num_kv_heads != 0) {
    throw std::invalid_argument("qwen38_cuda_full_attn_gated_f32_kvbf16 invalid dimensions or cache bounds");
  }
  const int q_total = num_heads * head_dim;
  const int kv_stride = num_kv_heads * head_dim;

  split_q_gate_kernel<float><<<dim3((q_total + 255) / 256, seq_len), 256, 0, stream>>>(
      q_with_gate, q_scratch, gate_scratch, seq_len, num_heads, head_dim);
  CUDA_CHECK(cudaGetLastError());

  rmsnorm_zero_centered_kernel<float><<<seq_len * num_heads, 256, 0, stream>>>(
      q_scratch, reinterpret_cast<const std::uint16_t*>(q_norm_w), rms_eps, seq_len * num_heads, head_dim, q_scratch);
  CUDA_CHECK(cudaGetLastError());

  rmsnorm_zero_centered_kernel<float><<<seq_len * num_kv_heads, 256, 0, stream>>>(
      k_raw_inout, reinterpret_cast<const std::uint16_t*>(k_norm_w), rms_eps, seq_len * num_kv_heads, head_dim,
      k_raw_inout);
  CUDA_CHECK(cudaGetLastError());

  const int half = rotary_dim >> 1;
  partial_rope_neox_kernel<float><<<dim3((num_heads * half + 127) / 128, seq_len), 128, 0, stream>>>(
      q_scratch, seq_len, num_heads, head_dim, rotary_dim, pos_offset, rope_theta);
  CUDA_CHECK(cudaGetLastError());
  partial_rope_neox_kernel<float><<<dim3((num_kv_heads * half + 127) / 128, seq_len), 128, 0, stream>>>(
      k_raw_inout, seq_len, num_kv_heads, head_dim, rotary_dim, pos_offset, rope_theta);
  CUDA_CHECK(cudaGetLastError());

  const std::size_t cache_off = static_cast<std::size_t>(pos_offset) * static_cast<std::size_t>(kv_stride);
  const std::size_t kv_elems = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(kv_stride);
  f32_to_bf16_kernel<<<(kv_elems + 255) / 256, 256, 0, stream>>>(
      k_raw_inout, reinterpret_cast<std::uint16_t*>(k_cache) + cache_off, kv_elems);
  CUDA_CHECK(cudaGetLastError());
  f32_to_bf16_kernel<<<(kv_elems + 255) / 256, 256, 0, stream>>>(
      v_in, reinterpret_cast<std::uint16_t*>(v_cache) + cache_off, kv_elems);
  CUDA_CHECK(cudaGetLastError());

  causal_gqa_gated_attn_kernel<float, std::uint16_t><<<dim3(num_heads, seq_len), 256, 0, stream>>>(
      q_scratch, gate_scratch, reinterpret_cast<const std::uint16_t*>(k_cache),
      reinterpret_cast<const std::uint16_t*>(v_cache), seq_len, pos_offset, num_heads, num_kv_heads, head_dim,
      out_attn);
  CUDA_CHECK(cudaGetLastError());
}

void qwen38_cuda_gdn_recurrent_prefill_f32(const float* conv_out_packed, const float* z_seq, const float* a_seq,
                                           const float* b_seq, const float* a_log, const float* dt_bias,
                                           const float* norm_weight, float rms_eps, int seq_len, int num_k_heads,
                                           int num_v_heads, int k_dim, int v_dim, float* recurrent_state,
                                           float* qk_norm_scratch, float* out_seq, cudaStream_t stream,
                                           bool force_global_fallback) {
  if (seq_len == 0) return;
  if (seq_len < 0 || num_k_heads <= 0 || num_v_heads <= 0 || num_v_heads % num_k_heads != 0 || k_dim <= 0 ||
      k_dim > 256 || v_dim <= 0 || v_dim > 512) {
    throw std::invalid_argument("qwen38_cuda_gdn_recurrent_prefill_f32 invalid dimensions");
  }
  const std::size_t qk_elems = static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(num_k_heads * k_dim);
  float* qn = qk_norm_scratch;
  float* kn = qn + qk_elems;

  gdn_normalize_qk_packed_f32_kernel<<<dim3(num_k_heads, seq_len), 128, 0, stream>>>(
      conv_out_packed, seq_len, num_k_heads, num_v_heads, k_dim, v_dim, qn, kn);
  CUDA_CHECK(cudaGetLastError());

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int max_optin_smem = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));

  cudaFuncAttributes attr{};
  CUDA_CHECK(cudaFuncGetAttributes(&attr, gdn_recurrent_prefill_exact_f32_kernel<true>));
  const std::size_t state_bytes = static_cast<std::size_t>(k_dim) * static_cast<std::size_t>(v_dim) * sizeof(float);
  const bool use_smem = !force_global_fallback && (seq_len > 1) &&
                        (static_cast<std::size_t>(attr.sharedSizeBytes) + state_bytes <=
                         static_cast<std::size_t>(max_optin_smem));
  const int block = ((std::max(32, std::min(256, v_dim)) + 31) / 32) * 32;

  if (use_smem) {
    CUDA_CHECK(cudaFuncSetAttribute(gdn_recurrent_prefill_exact_f32_kernel<true>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(state_bytes)));
    gdn_recurrent_prefill_exact_f32_kernel<true><<<num_v_heads, block, state_bytes, stream>>>(
        qn, kn, conv_out_packed, z_seq, a_seq, b_seq, a_log, dt_bias, norm_weight, rms_eps, seq_len, num_k_heads,
        num_v_heads, k_dim, v_dim, recurrent_state, out_seq);
  } else {
    gdn_recurrent_prefill_exact_f32_kernel<false><<<num_v_heads, block, 0, stream>>>(
        qn, kn, conv_out_packed, z_seq, a_seq, b_seq, a_log, dt_bias, norm_weight, rms_eps, seq_len, num_k_heads,
        num_v_heads, k_dim, v_dim, recurrent_state, out_seq);
  }
  CUDA_CHECK(cudaGetLastError());
}

Qwen38BlockRunOutput run_qwen38_blocks_gpu(const Qwen38BlockBundleHost& bundle, bool use_chunk_prefill, int chunk_size,
                                           bool use_calibrated_fp8_input_scale) {
  const Qwen38Config& c = bundle.cfg;
  c.validate();
  const int S = static_cast<int>(bundle.token_ids.size());
  const int H = c.hidden_size;
  const int I = c.intermediate_size;
  if (S == 0 || bundle.layers.empty() ||
      bundle.prompt_embeddings_bf16.size() != static_cast<std::size_t>(S * H)) {
    throw std::invalid_argument("run_qwen38_blocks_gpu invalid bundle");
  }

  DeviceContext ctx(0);
  ctx.bind_to_current_thread();

  std::size_t vram_free_before = 0, vram_total = 0;
  CUDA_CHECK(cudaMemGetInfo(&vram_free_before, &vram_total));

  // Upload layer weights + allocate per-layer states.
  std::vector<DevBlockLayer> dl(bundle.layers.size());
  std::vector<DevBuf<float>> d_conv_states(bundle.layers.size());
  std::vector<DevBuf<float>> d_rec_states(bundle.layers.size());
  std::vector<DevBuf<std::uint16_t>> d_k_caches(bundle.layers.size());
  std::vector<DevBuf<std::uint16_t>> d_v_caches(bundle.layers.size());

  const int cd = c.linear_conv_dim();
  const int kd = c.linear_key_dim();
  const int vd = c.linear_value_dim();
  const int nhv = c.linear_num_value_heads;
  const int qg_dim = c.full_q_proj_out_dim();
  const int kv_dim = c.full_kv_proj_out_dim();
  const int attn_dim = c.num_attention_heads * c.head_dim;

  for (std::size_t li = 0; li < bundle.layers.size(); ++li) {
    const auto& hl = bundle.layers[li];
    dl[li].kind = hl.kind;
    dl[li].input_layernorm.upload(hl.input_layernorm.data(), hl.input_layernorm.size());
    dl[li].post_attention_layernorm.upload(hl.post_attention_layernorm.data(), hl.post_attention_layernorm.size());
    if (hl.kind == Qwen38LayerKind::LinearAttention) {
      dl[li].in_proj_qkv = DevProjWeights::from_host(hl.in_proj_qkv);
      dl[li].in_proj_z = DevProjWeights::from_host(hl.in_proj_z);
      dl[li].in_proj_a = DevProjWeights::from_host(hl.in_proj_a);
      dl[li].in_proj_b = DevProjWeights::from_host(hl.in_proj_b);
      dl[li].out_proj = DevProjWeights::from_host(hl.out_proj);
      dl[li].A_log.upload(hl.A_log.data(), hl.A_log.size());
      dl[li].dt_bias.upload(hl.dt_bias.data(), hl.dt_bias.size());
      dl[li].conv1d_weight.upload(hl.conv1d_weight.data(), hl.conv1d_weight.size());
      dl[li].linear_norm_weight.upload(hl.linear_norm_weight.data(), hl.linear_norm_weight.size());
      d_conv_states[li].allocate(static_cast<std::size_t>(cd * c.linear_conv_kernel_dim));
      CUDA_CHECK(cudaMemset(d_conv_states[li].ptr, 0, d_conv_states[li].count * sizeof(float)));
      d_rec_states[li].allocate(static_cast<std::size_t>(nhv) * static_cast<std::size_t>(c.linear_key_head_dim) *
                                static_cast<std::size_t>(c.linear_value_head_dim));
      CUDA_CHECK(cudaMemset(d_rec_states[li].ptr, 0, d_rec_states[li].count * sizeof(float)));
    } else {
      dl[li].q_proj = DevProjWeights::from_host(hl.q_proj);
      dl[li].k_proj = DevProjWeights::from_host(hl.k_proj);
      dl[li].v_proj = DevProjWeights::from_host(hl.v_proj);
      dl[li].o_proj = DevProjWeights::from_host(hl.o_proj);
      dl[li].q_norm.upload(hl.q_norm.data(), hl.q_norm.size());
      dl[li].k_norm.upload(hl.k_norm.data(), hl.k_norm.size());
      d_k_caches[li].allocate(static_cast<std::size_t>(S * kv_dim));
      d_v_caches[li].allocate(static_cast<std::size_t>(S * kv_dim));
    }
    dl[li].gate_proj = DevProjWeights::from_host(hl.gate_proj);
    dl[li].up_proj = DevProjWeights::from_host(hl.up_proj);
    dl[li].down_proj = DevProjWeights::from_host(hl.down_proj);
  }

  DevBuf<std::uint16_t> d_hidden(static_cast<std::size_t>(S * H));
  CUDA_CHECK(cudaMemcpy(d_hidden.ptr, bundle.prompt_embeddings_bf16.data(),
                        static_cast<std::size_t>(S * H) * sizeof(std::uint16_t), cudaMemcpyHostToDevice));
  DevBuf<std::uint16_t> d_normed(static_cast<std::size_t>(S * H));
  DevBuf<std::uint16_t> d_residual(static_cast<std::size_t>(S * H));
  DevBuf<std::uint16_t> d_qkv(static_cast<std::size_t>(S * cd));
  DevBuf<std::uint16_t> d_conv_out(static_cast<std::size_t>(S * cd));
  DevBuf<std::uint16_t> d_q_sep(static_cast<std::size_t>(S * kd));
  DevBuf<std::uint16_t> d_k_sep(static_cast<std::size_t>(S * kd));
  DevBuf<std::uint16_t> d_v_sep(static_cast<std::size_t>(S * vd));
  DevBuf<std::uint16_t> d_z(static_cast<std::size_t>(S * vd));
  DevBuf<std::uint16_t> d_a(static_cast<std::size_t>(S * nhv));
  DevBuf<std::uint16_t> d_b(static_cast<std::size_t>(S * nhv));
  DevBuf<std::uint16_t> d_core(static_cast<std::size_t>(S * vd));
  const std::size_t gdn_ws = qwen38_cuda_gdn_chunk_scratch_floats(S, std::max(1, std::min(64, chunk_size)),
                                                                  c.linear_num_key_heads, nhv, c.linear_key_head_dim,
                                                                  c.linear_value_head_dim);
  DevBuf<float> d_gdn_scratch(std::max<std::size_t>(gdn_ws, static_cast<std::size_t>(2 * kd)));

  DevBuf<std::uint16_t> d_q_gate(static_cast<std::size_t>(qg_dim));
  DevBuf<std::uint16_t> d_k_raw(static_cast<std::size_t>(kv_dim));
  DevBuf<std::uint16_t> d_v_raw(static_cast<std::size_t>(kv_dim));
  DevBuf<std::uint16_t> d_q_tmp(static_cast<std::size_t>(attn_dim));
  DevBuf<std::uint16_t> d_gate_tmp(static_cast<std::size_t>(attn_dim));
  DevBuf<std::uint16_t> d_attn_out(static_cast<std::size_t>(attn_dim));

  DevBuf<std::uint16_t> d_mlp_gate(static_cast<std::size_t>(S * I));
  DevBuf<std::uint16_t> d_mlp_up(static_cast<std::size_t>(S * I));

  std::size_t vram_free_after = 0;
  CUDA_CHECK(cudaMemGetInfo(&vram_free_after, &vram_total));

  CudaEventTimer timer(ctx, ctx.stream);
  timer.start();

  for (std::size_t li = 0; li < dl.size(); ++li) {
    auto& l = dl[li];
    qwen38_cuda_rmsnorm_zero_centered_bf16(
        reinterpret_cast<const __nv_bfloat16*>(d_hidden.ptr),
        reinterpret_cast<const __nv_bfloat16*>(l.input_layernorm.ptr), c.rms_norm_eps, S, H,
        reinterpret_cast<__nv_bfloat16*>(d_normed.ptr), ctx.stream);

    if (l.kind == Qwen38LayerKind::LinearAttention) {
      l.in_proj_qkv.forward_bf16(reinterpret_cast<const __nv_bfloat16*>(d_normed.ptr), S,
                                 reinterpret_cast<__nv_bfloat16*>(d_qkv.ptr), use_calibrated_fp8_input_scale,
                                 ctx.stream);
      l.in_proj_z.forward_bf16(reinterpret_cast<const __nv_bfloat16*>(d_normed.ptr), S,
                               reinterpret_cast<__nv_bfloat16*>(d_z.ptr), use_calibrated_fp8_input_scale, ctx.stream);
      l.in_proj_a.forward_bf16(reinterpret_cast<const __nv_bfloat16*>(d_normed.ptr), S,
                               reinterpret_cast<__nv_bfloat16*>(d_a.ptr), false, ctx.stream);
      l.in_proj_b.forward_bf16(reinterpret_cast<const __nv_bfloat16*>(d_normed.ptr), S,
                               reinterpret_cast<__nv_bfloat16*>(d_b.ptr), false, ctx.stream);

      if (use_chunk_prefill) {
        qwen38_cuda_causal_conv1d_silu_prefill_bf16(
            reinterpret_cast<const __nv_bfloat16*>(d_qkv.ptr), l.conv1d_weight.ptr, S, cd, c.linear_conv_kernel_dim,
            d_conv_states[li].ptr, reinterpret_cast<__nv_bfloat16*>(d_conv_out.ptr), ctx.stream);
        split_qkv_sep_bf16_kernel<<<dim3((cd + 255) / 256, S), 256, 0, ctx.stream>>>(
            d_conv_out.ptr, d_q_sep.ptr, d_k_sep.ptr, d_v_sep.ptr, S, kd, vd);
        CUDA_CHECK(cudaGetLastError());
        qwen38_cuda_gdn_chunk_prefill_bf16(
            reinterpret_cast<const __nv_bfloat16*>(d_q_sep.ptr), reinterpret_cast<const __nv_bfloat16*>(d_k_sep.ptr),
            reinterpret_cast<const __nv_bfloat16*>(d_v_sep.ptr), reinterpret_cast<const __nv_bfloat16*>(d_z.ptr),
            reinterpret_cast<const __nv_bfloat16*>(d_a.ptr), reinterpret_cast<const __nv_bfloat16*>(d_b.ptr),
            l.A_log.ptr, l.dt_bias.ptr, l.linear_norm_weight.ptr, c.rms_norm_eps, S, chunk_size,
            c.linear_num_key_heads, nhv, c.linear_key_head_dim, c.linear_value_head_dim, d_rec_states[li].ptr,
            d_gdn_scratch.ptr, reinterpret_cast<__nv_bfloat16*>(d_core.ptr), ctx.stream);
      } else {
        for (int t = 0; t < S; ++t) {
          qwen38_cuda_causal_conv1d_silu_step_bf16(
              reinterpret_cast<const __nv_bfloat16*>(d_qkv.ptr + static_cast<std::size_t>(t * cd)),
              l.conv1d_weight.ptr, cd, c.linear_conv_kernel_dim, d_conv_states[li].ptr,
              reinterpret_cast<__nv_bfloat16*>(d_conv_out.ptr), ctx.stream);
          qwen38_cuda_gdn_recurrent_step_bf16(
              reinterpret_cast<const __nv_bfloat16*>(d_conv_out.ptr),
              reinterpret_cast<const __nv_bfloat16*>(d_conv_out.ptr + kd),
              reinterpret_cast<const __nv_bfloat16*>(d_conv_out.ptr + 2 * kd),
              reinterpret_cast<const __nv_bfloat16*>(d_z.ptr + static_cast<std::size_t>(t * vd)),
              reinterpret_cast<const __nv_bfloat16*>(d_a.ptr + static_cast<std::size_t>(t * nhv)),
              reinterpret_cast<const __nv_bfloat16*>(d_b.ptr + static_cast<std::size_t>(t * nhv)), l.A_log.ptr,
              l.dt_bias.ptr, l.linear_norm_weight.ptr, c.rms_norm_eps, c.linear_num_key_heads, nhv,
              c.linear_key_head_dim, c.linear_value_head_dim, d_rec_states[li].ptr, d_gdn_scratch.ptr,
              reinterpret_cast<__nv_bfloat16*>(d_core.ptr + static_cast<std::size_t>(t * vd)), ctx.stream);
        }
      }

      l.out_proj.forward_bf16(reinterpret_cast<const __nv_bfloat16*>(d_core.ptr), S,
                              reinterpret_cast<__nv_bfloat16*>(d_residual.ptr), use_calibrated_fp8_input_scale,
                              ctx.stream);
    } else {
      for (int t = 0; t < S; ++t) {
        const auto* x_t = reinterpret_cast<const __nv_bfloat16*>(d_normed.ptr + static_cast<std::size_t>(t * H));
        l.q_proj.forward_bf16(x_t, 1, reinterpret_cast<__nv_bfloat16*>(d_q_gate.ptr), use_calibrated_fp8_input_scale,
                              ctx.stream);
        l.k_proj.forward_bf16(x_t, 1, reinterpret_cast<__nv_bfloat16*>(d_k_raw.ptr), use_calibrated_fp8_input_scale,
                              ctx.stream);
        l.v_proj.forward_bf16(x_t, 1, reinterpret_cast<__nv_bfloat16*>(d_v_raw.ptr), use_calibrated_fp8_input_scale,
                              ctx.stream);
        qwen38_cuda_full_attn_gated_bf16(
            reinterpret_cast<const __nv_bfloat16*>(d_q_gate.ptr), reinterpret_cast<const __nv_bfloat16*>(d_k_raw.ptr),
            reinterpret_cast<const __nv_bfloat16*>(d_v_raw.ptr),
            reinterpret_cast<const __nv_bfloat16*>(l.q_norm.ptr),
            reinterpret_cast<const __nv_bfloat16*>(l.k_norm.ptr), c.rms_norm_eps, c.rope_theta, 1, t,
            c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.rotary_dim,
            reinterpret_cast<__nv_bfloat16*>(d_k_caches[li].ptr), reinterpret_cast<__nv_bfloat16*>(d_v_caches[li].ptr),
            S, reinterpret_cast<__nv_bfloat16*>(d_q_tmp.ptr), reinterpret_cast<__nv_bfloat16*>(d_gate_tmp.ptr),
            reinterpret_cast<__nv_bfloat16*>(d_attn_out.ptr), ctx.stream);
        l.o_proj.forward_bf16(reinterpret_cast<const __nv_bfloat16*>(d_attn_out.ptr), 1,
                              reinterpret_cast<__nv_bfloat16*>(d_residual.ptr + static_cast<std::size_t>(t * H)),
                              use_calibrated_fp8_input_scale, ctx.stream);
      }
    }

    qwen38_cuda_add_residual_bf16(reinterpret_cast<__nv_bfloat16*>(d_hidden.ptr),
                                  reinterpret_cast<const __nv_bfloat16*>(d_residual.ptr),
                                  static_cast<std::size_t>(S * H), ctx.stream);

    qwen38_cuda_rmsnorm_zero_centered_bf16(
        reinterpret_cast<const __nv_bfloat16*>(d_hidden.ptr),
        reinterpret_cast<const __nv_bfloat16*>(l.post_attention_layernorm.ptr), c.rms_norm_eps, S, H,
        reinterpret_cast<__nv_bfloat16*>(d_normed.ptr), ctx.stream);

    l.gate_proj.forward_bf16(reinterpret_cast<const __nv_bfloat16*>(d_normed.ptr), S,
                             reinterpret_cast<__nv_bfloat16*>(d_mlp_gate.ptr), use_calibrated_fp8_input_scale,
                             ctx.stream);
    l.up_proj.forward_bf16(reinterpret_cast<const __nv_bfloat16*>(d_normed.ptr), S,
                           reinterpret_cast<__nv_bfloat16*>(d_mlp_up.ptr), use_calibrated_fp8_input_scale, ctx.stream);
    qwen38_cuda_swiglu_mul_bf16(reinterpret_cast<const __nv_bfloat16*>(d_mlp_gate.ptr),
                                reinterpret_cast<const __nv_bfloat16*>(d_mlp_up.ptr),
                                reinterpret_cast<__nv_bfloat16*>(d_mlp_gate.ptr), static_cast<std::size_t>(S * I),
                                ctx.stream);
    l.down_proj.forward_bf16(reinterpret_cast<const __nv_bfloat16*>(d_mlp_gate.ptr), S,
                             reinterpret_cast<__nv_bfloat16*>(d_residual.ptr), use_calibrated_fp8_input_scale,
                             ctx.stream);
    qwen38_cuda_add_residual_bf16(reinterpret_cast<__nv_bfloat16*>(d_hidden.ptr),
                                  reinterpret_cast<const __nv_bfloat16*>(d_residual.ptr),
                                  static_cast<std::size_t>(S * H), ctx.stream);
  }

  timer.record_stop();
  ctx.synchronize();

  Qwen38BlockRunOutput out;
  out.elapsed_ms = timer.elapsed_ms();
  out.peak_vram_bytes = vram_free_before > vram_free_after ? (vram_free_before - vram_free_after) : 0ull;

  std::vector<std::uint16_t> h_hidden_bf16(static_cast<std::size_t>(S * H));
  CUDA_CHECK(cudaMemcpy(h_hidden_bf16.data(), d_hidden.ptr,
                        static_cast<std::size_t>(S * H) * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
  out.hidden_seq.resize(h_hidden_bf16.size());
  for (std::size_t i = 0; i < h_hidden_bf16.size(); ++i) out.hidden_seq[i] = bf16_to_f32(h_hidden_bf16[i]);

  out.conv_states.resize(bundle.layers.size());
  out.recurrent_states.resize(bundle.layers.size());
  out.k_caches.resize(bundle.layers.size());
  out.v_caches.resize(bundle.layers.size());
  for (std::size_t li = 0; li < bundle.layers.size(); ++li) {
    if (bundle.layers[li].kind == Qwen38LayerKind::LinearAttention) {
      out.conv_states[li].resize(d_conv_states[li].count);
      CUDA_CHECK(cudaMemcpy(out.conv_states[li].data(), d_conv_states[li].ptr,
                            d_conv_states[li].count * sizeof(float), cudaMemcpyDeviceToHost));
      out.recurrent_states[li].resize(d_rec_states[li].count);
      CUDA_CHECK(cudaMemcpy(out.recurrent_states[li].data(), d_rec_states[li].ptr,
                            d_rec_states[li].count * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
      std::vector<std::uint16_t> kb(d_k_caches[li].count), vb(d_v_caches[li].count);
      CUDA_CHECK(cudaMemcpy(kb.data(), d_k_caches[li].ptr, kb.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(vb.data(), d_v_caches[li].ptr, vb.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
      out.k_caches[li].resize(kb.size());
      out.v_caches[li].resize(vb.size());
      for (std::size_t i = 0; i < kb.size(); ++i) {
        out.k_caches[li][i] = bf16_to_f32(kb[i]);
        out.v_caches[li][i] = bf16_to_f32(vb[i]);
      }
    }
  }
  return out;
}

}  // namespace ingine::backends::qwen38

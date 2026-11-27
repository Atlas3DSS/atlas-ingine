// Phase 1 CUDA kernel parity tests for Qwen3.8-Flash-Next (qwen4exp, #849).
// Verifies every qwen4exp CUDA kernel (GGML linear F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K,
// Elastic NVFP4/ATLAS3/ATLAS2 linear, Hyper-Connections, PLE dilated conv+gate,
// sigmoid-gated GDN, QSA block-4 indexer + gated GQA, Top-K Softmax router, SwiGLU + shared expert)
// bit-closely against the FP64-accumulated CPU reference in qwen4exp_cpu_ops.
// Skips cleanly with exit code 77 when no CUDA device is visible.
#include "backends/qwen4exp/qwen4exp_config.h"
#include "backends/qwen4exp/qwen4exp_cpu_ops.h"
#include "backends/qwen4exp/qwen4exp_cuda_ops.h"
#include "backends/qwen4exp/qwen4exp_gguf.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace {

using namespace ingine::backends::qwen4exp;

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

template <typename T>
struct DevVec {
  T* p = nullptr;
  std::size_t n = 0;
  DevVec() = default;
  explicit DevVec(std::size_t count) : n(count) {
    if (n > 0) CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  }
  explicit DevVec(const std::vector<T>& host) : DevVec(host.size()) {
    if (n > 0) CUDA_CHECK(cudaMemcpy(p, host.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~DevVec() {
    if (p) cudaFree(p);
  }
  DevVec(const DevVec&) = delete;
  DevVec& operator=(const DevVec&) = delete;
  std::vector<T> to_host() const {
    std::vector<T> out(n);
    if (n > 0) CUDA_CHECK(cudaMemcpy(out.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return out;
  }
};

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  CHECK(!a.empty() && a.size() == b.size());
  float m = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
    const float d = std::fabs(a[i] - b[i]);
    CHECK(std::isfinite(d));
    if (d > m) m = d;
  }
  return m;
}

std::vector<float> rand_f32(std::mt19937& rng, std::size_t n, float scale = 0.25f) {
  std::normal_distribution<float> dist(0.0f, scale);
  std::vector<float> v(n);
  for (float& x : v) x = dist(rng);
  return v;
}

std::uint16_t f32_to_fp16(float x) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &x, 4);
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
  const std::uint32_t mant = (bits >> 13) & 0x03ffu;
  if (exp <= 0) return static_cast<std::uint16_t>(sign);
  if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | mant);
}

void test_ggml_and_elastic_linear(std::mt19937& rng, cudaStream_t stream) {
  const int out_f = 64;
  const int in_f = 512;  // divisible by 256 (Q4_K/Q5_K/Q6_K) and 32 (Q8_0/ATLAS3)
  const auto x = rand_f32(rng, static_cast<std::size_t>(in_f), 0.3f);
  DevVec<float> dx(x), dy(out_f);

  // Test GGML types: F32(0), F16(1), Q4_0(2), Q5_1(7), Q8_0(8), Q4_K(12), Q5_K(13), Q6_K(14), BF16(30)
  const std::vector<std::uint32_t> types = {0, 1, 2, 7, 8, 12, 13, 14, 30};
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for (std::uint32_t tid : types) {
    GgmlBlockGeometry geom{};
    CHECK(ggml_block_geometry(tid, geom));
    const std::size_t nb = static_cast<std::size_t>(in_f) / geom.block_elems;
    const std::size_t rb = nb * geom.block_bytes;
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(out_f) * rb);
    for (auto& b : raw) b = static_cast<std::uint8_t>(byte_dist(rng));
    // Sanitize FP16/BF16/F32 scale/value fields so weights have realistic O(1) scale without NaN/Inf.
    const std::uint16_t safe_d = f32_to_fp16(0.0625f);
    const std::uint16_t safe_dk = f32_to_fp16(0.004f);
    for (int r = 0; r < out_f; ++r) {
      std::uint8_t* rowp = raw.data() + static_cast<std::size_t>(r) * rb;
      if (tid == 0) {
        auto rf = rand_f32(rng, in_f, 0.2f);
        std::memcpy(rowp, rf.data(), rb);
      } else if (tid == 1 || tid == 30) {
        auto rf = rand_f32(rng, in_f, 0.2f);
        for (int c = 0; c < in_f; ++c) {
          std::uint16_t h = (tid == 1) ? f32_to_fp16(rf[c]) : static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(rf[c]) >> 16);
          rowp[2 * c] = static_cast<std::uint8_t>(h & 0xffu);
          rowp[2 * c + 1] = static_cast<std::uint8_t>(h >> 8);
        }
      } else if (tid == 2 || tid == 8) {
        const std::uint16_t sd = (tid == 8) ? safe_dk : safe_d;
        for (std::size_t b = 0; b < nb; ++b) {
          rowp[b * geom.block_bytes + 0] = static_cast<std::uint8_t>(sd & 0xffu);
          rowp[b * geom.block_bytes + 1] = static_cast<std::uint8_t>(sd >> 8);
        }
      } else if (tid == 7 || tid == 12 || tid == 13) {
        const std::uint16_t sd = (tid == 7) ? safe_d : safe_dk;
        for (std::size_t b = 0; b < nb; ++b) {
          rowp[b * geom.block_bytes + 0] = static_cast<std::uint8_t>(sd & 0xffu);
          rowp[b * geom.block_bytes + 1] = static_cast<std::uint8_t>(sd >> 8);
          rowp[b * geom.block_bytes + 2] = static_cast<std::uint8_t>(sd & 0xffu);
          rowp[b * geom.block_bytes + 3] = static_cast<std::uint8_t>(sd >> 8);
        }
      } else if (tid == 14) {
        for (std::size_t b = 0; b < nb; ++b) {
          rowp[b * geom.block_bytes + 208] = static_cast<std::uint8_t>(safe_dk & 0xffu);
          rowp[b * geom.block_bytes + 209] = static_cast<std::uint8_t>(safe_dk >> 8);
        }
      }
    }

    std::vector<float> w_f32(static_cast<std::size_t>(out_f) * in_f);
    for (int r = 0; r < out_f; ++r) {
      dequantize_row_to_f32(tid, std::span<const std::uint8_t>(raw.data() + static_cast<std::size_t>(r) * rb, rb),
                            std::span<float>(w_f32.data() + static_cast<std::size_t>(r) * in_f, in_f));
    }
    std::vector<float> y_cpu(out_f);
    matvec_f32(w_f32, x, out_f, in_f, y_cpu);

    DevVec<std::uint8_t> dw(raw);
    ggml_linear_f32_cuda(tid, dw.p, dx.p, dy.p, 1, out_f, in_f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto y_gpu = dy.to_host();
    const float diff = max_abs_diff(y_cpu, y_gpu);
    float ref_max = 1.0f;
    for (float v : y_cpu) ref_max = std::max(ref_max, std::fabs(v));
    std::printf("  ggml tid=%u(%s) max_abs=%.3e ref_max=%.3e rel=%.3e\n", tid, ggml_type_name(tid), diff, ref_max,
                diff / ref_max);
    CHECK(diff < 5e-4f * ref_max);
  }

  // Test Elastic NVFP4 / ATLAS3 / ATLAS2 linear against exact CPU dequant!
  static constexpr float kE2M1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  static constexpr float kA2[4] = {-1.5f, -0.453125f, 0.453125f, 1.5f};
  static constexpr float kA3[8] = {-2.0f, -1.25f, -0.703125f, -0.2265625f, 0.2265625f, 0.703125f, 1.25f, 2.0f};
  auto decode_e4m3 = [](std::uint8_t bits) -> float {
    const std::uint8_t s = (bits >> 7) & 1u, e = (bits >> 3) & 0x0fu, m = bits & 7u;
    const float v = (e == 0) ? std::ldexp(float(m) * 0.125f, -6) : std::ldexp(1.0f + float(m) * 0.125f, int(e) - 7);
    return s ? -v : v;
  };
  const int nb16 = in_f / 16;
  std::vector<std::uint8_t> scales(static_cast<std::size_t>(out_f) * nb16, 0x38u);  // 1.0 in E4M3FN
  const float factor = 0.125f;
  DevVec<std::uint8_t> dscales(scales);

  for (ElasticWeightFormat fmt : {ElasticWeightFormat::Nvfp4, ElasticWeightFormat::Atlas3, ElasticWeightFormat::Atlas2}) {
    const std::size_t row_bytes = (fmt == ElasticWeightFormat::Nvfp4)
                                      ? static_cast<std::size_t>(in_f / 2)
                                      : (fmt == ElasticWeightFormat::Atlas3 ? static_cast<std::size_t>(in_f * 3 / 8)
                                                                            : static_cast<std::size_t>(in_f / 4));
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(out_f) * row_bytes);
    for (auto& b : codes) b = static_cast<std::uint8_t>(byte_dist(rng));

    std::vector<float> w_f32(static_cast<std::size_t>(out_f) * in_f);
    for (int r = 0; r < out_f; ++r) {
      const std::uint8_t* cp = codes.data() + static_cast<std::size_t>(r) * row_bytes;
      const std::uint8_t* sp = scales.data() + static_cast<std::size_t>(r) * nb16;
      float* dst = w_f32.data() + static_cast<std::size_t>(r) * in_f;
      for (int b = 0; b < nb16; ++b) {
        const float sc = decode_e4m3(sp[b]) * factor;
        if (fmt == ElasticWeightFormat::Nvfp4) {
          for (int i = 0; i < 8; ++i) {
            const std::uint8_t pb = cp[b * 8 + i];
            dst[b * 16 + 2 * i] = kE2M1[pb & 0x0fu] * sc;
            dst[b * 16 + 2 * i + 1] = kE2M1[pb >> 4] * sc;
          }
        } else if (fmt == ElasticWeightFormat::Atlas3) {
          const std::uint8_t* blk = cp + (b / 2) * 12;
          const int half = b & 1;
          std::uint32_t lo = 0;
          std::uint16_t hi = 0;
          std::memcpy(&lo, blk + half * 4, 4);
          std::memcpy(&hi, blk + 8 + half * 2, 2);
          for (int i = 0; i < 16; ++i) {
            const std::uint32_t c = ((lo >> (2 * i)) & 3u) | (((hi >> i) & 1u) << 2);
            dst[b * 16 + i] = kA3[c] * sc;
          }
        } else {
          std::uint32_t w = 0;
          std::memcpy(&w, cp + b * 4, 4);
          for (int i = 0; i < 16; ++i) dst[b * 16 + i] = kA2[(w >> (2 * i)) & 3u] * sc;
        }
      }
    }
    std::vector<float> y_cpu(out_f);
    matvec_f32(w_f32, x, out_f, in_f, y_cpu);
    DevVec<std::uint8_t> dcodes(codes);
    elastic_linear_f32_cuda(fmt, dcodes.p, dscales.p, nullptr, factor, dx.p, dy.p, 1, out_f, in_f, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(max_abs_diff(y_cpu, dy.to_host()) < 2e-4f);
  }
  std::printf("PASS test_ggml_and_elastic_linear (9 GGML + 3 Elastic formats verified)\n");
}

void test_hc_ple_gdn_qsa_moe(std::mt19937& rng, cudaStream_t stream) {
  const auto mini = Qwen4ExpMiniModelCpu::make_synthetic(99u, /*top_k_budget=*/8);
  const auto& c = mini.config();
  const int H = c.hidden_size, hc = c.hc_mult, hc_dim = c.hc_dim(), hc_lr = c.hc_low_rank;

  // 1. Hyper-Connections mix + combine
  {
    const auto x = rand_f32(rng, hc_dim), wn = rand_f32(rng, hc_dim, 1.0f), wd = rand_f32(rng, hc_lr * hc_dim),
               wu = rand_f32(rng, hc_dim * hc_lr), wi = rand_f32(rng, hc * hc_dim), bo = rand_f32(rng, H);
    std::vector<float> m_cpu(H), inj_cpu(hc), comb_cpu(hc_dim);
    hc_mix_f32(x, wn, wd, wu, wi, hc, H, hc_lr, c.rms_norm_eps, m_cpu, inj_cpu);
    hc_combine_f32(x, bo, inj_cpu, hc, H, comb_cpu);

    DevVec<float> dx(x), dwn(wn), dwd(wd), dwu(wu), dwi(wi), dbo(bo);
    DevVec<float> dxn(hc_dim), dlo(hc_lr), dgate(hc_dim), dm(H), dinj(hc), dcomb(hc_dim);
    grouped_rmsnorm_f32_cuda(dx.p, dwn.p, dxn.p, hc, H, c.rms_norm_eps, stream);
    ggml_linear_f32_cuda(0, dwd.p, dxn.p, dlo.p, 1, hc_lr, hc_dim, stream);
    hc_silu_scale_inplace_f32_cuda(dlo.p, hc_lr, hc, stream);
    ggml_linear_f32_cuda(0, dwu.p, dlo.p, dgate.p, 1, hc_dim, hc_lr, stream);
    hc_gate_collapse_f32_cuda(dxn.p, dgate.p, dm.p, hc, H, stream);
    ggml_linear_f32_cuda(0, dwi.p, dxn.p, dinj.p, 1, hc, hc_dim, stream);
    hc_combine_f32_cuda(dx.p, dbo.p, dinj.p, dcomb.p, hc, H, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CHECK(max_abs_diff(m_cpu, dm.to_host()) < 2e-5f);
    CHECK(max_abs_diff(inj_cpu, dinj.to_host()) < 2e-5f);
    CHECK(max_abs_diff(comb_cpu, dcomb.to_host()) < 2e-5f);
  }

  // 2. PLE step across 5 consecutive tokens (exercises dilated conv history!)
  {
    const auto wk = rand_f32(rng, hc_dim * H), wv = rand_f32(rng, H * H), wnk = rand_f32(rng, hc_dim, 1.0f),
               wnq = rand_f32(rng, hc_dim, 1.0f), wnc = rand_f32(rng, hc_dim, 1.0f),
               wconv = rand_f32(rng, hc_dim * c.ple_conv_kernel);
    std::vector<float> cpu_st(static_cast<std::size_t>(c.ple_conv_history_cols()) * hc_dim, 0.0f);
    DevVec<float> dwk(wk), dwv(wv), dwnk(wnk), dwnq(wnq), dwnc(wnc), dwconv(wconv), dst(cpu_st);
    DevVec<float> dkey_raw(hc_dim), dval(H), dkey(hc_dim), dq(hc_dim), dgated(hc_dim), dnorm(hc_dim), dout(hc_dim);
    for (int step = 0; step < 5; ++step) {
      const auto hin = rand_f32(rng, hc_dim), pemb = rand_f32(rng, H);
      std::vector<float> hout_cpu(hc_dim);
      ple_step_f32(c, hin, pemb, wk, wv, wnk, wnq, wnc, wconv, cpu_st, hout_cpu);

      DevVec<float> dhin(hin), dpemb(pemb);
      ggml_linear_f32_cuda(0, dwk.p, dpemb.p, dkey_raw.p, 1, hc_dim, H, stream);
      ggml_linear_f32_cuda(0, dwv.p, dpemb.p, dval.p, 1, H, H, stream);
      grouped_rmsnorm_f32_cuda(dkey_raw.p, dwnk.p, dkey.p, hc, H, c.rms_norm_eps, stream);
      grouped_rmsnorm_f32_cuda(dhin.p, dwnq.p, dq.p, hc, H, c.rms_norm_eps, stream);
      ple_gate_value_f32_cuda(dkey.p, dq.p, dval.p, dgated.p, hc, H, stream);
      grouped_rmsnorm_f32_cuda(dgated.p, dwnc.p, dnorm.p, hc, H, c.rms_norm_eps, stream);
      ple_conv_residual_step_f32_cuda(dhin.p, dgated.p, dnorm.p, dwconv.p, dst.p, dout.p, hc_dim, c.ple_conv_kernel,
                                      c.ple_ngram_size, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(max_abs_diff(hout_cpu, dout.to_host()) < 5e-5f);
    }
    CHECK(max_abs_diff(cpu_st, dst.to_host()) < 5e-5f);
  }

  // 3. GDN sigmoid step across 5 tokens
  {
    const int kh = 4, vh = 8, hd = 32, ck = 4;
    const int conv_dim = 2 * kh * hd + vh * hd;
    const auto wconv = rand_f32(rng, conv_dim * ck), dtb = rand_f32(rng, vh), normw = rand_f32(rng, hd, 1.0f);
    std::vector<float> ssm_a(vh, -0.5f);
    std::vector<float> cst(ck * conv_dim, 0.0f), rst(vh * hd * hd, 0.0f);
    DevVec<float> dwconv(wconv), ddtb(dtb), da(ssm_a), dnw(normw), dcst(cst), drst(rst), dtmp(conv_dim), dout(vh * hd);
    for (int step = 0; step < 5; ++step) {
      const auto qkv = rand_f32(rng, conv_dim), z = rand_f32(rng, vh * hd), al = rand_f32(rng, vh), be = rand_f32(rng, vh);
      std::vector<float> o_cpu(vh * hd);
      gdn_step_sigmoid_f32(qkv, z, al, be, wconv, dtb, ssm_a, normw, 1e-6f, ck, kh, vh, hd, cst, rst, o_cpu);
      DevVec<float> dqkv(qkv), dz(z), dal(al), dbe(be);
      gdn_step_sigmoid_f32_cuda(dqkv.p, dz.p, dal.p, dbe.p, dwconv.p, ddtb.p, da.p, dnw.p, 1e-6f, ck, kh, vh, hd,
                                dcst.p, drst.p, dtmp.p, dout.p, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(max_abs_diff(o_cpu, dout.to_host()) < 1e-4f);
    }
  }

  // 4. Full Attn + QSA indexer across 14 tokens (exercises dense + sparse top-k!)
  {
    const int nh = 4, nkvh = 2, hd = 32, rot = 16, idx_h = 2, idx_d = 32, cr = 4, topk = 8, max_ctx = 32;
    const auto qnw = rand_f32(rng, hd, 1.0f), knw = rand_f32(rng, hd, 1.0f), iqnw = rand_f32(rng, idx_d, 1.0f),
               iknw = rand_f32(rng, idx_d, 1.0f);
    std::vector<float> k_cpu, v_cpu, ik_cpu;
    DevVec<float> dqnw(qnw), dknw(knw), diqnw(iqnw), diknw(iknw);
    DevVec<float> dkc(max_ctx * nkvh * hd), dvc(max_ctx * nkvh * hd), dikc(max_ctx * idx_d);
    DevVec<float> dq_sc(nh * hd), dg_sc(nh * hd), dout(nh * hd), dscores(max_ctx);
    for (int pos = 0; pos < 14; ++pos) {
      const auto qg = rand_f32(rng, nh * 2 * hd), kr = rand_f32(rng, nkvh * hd), vr = rand_f32(rng, nkvh * hd);
      const auto iq = rand_f32(rng, idx_h * idx_d), ik = rand_f32(rng, idx_d);
      const auto sel = qsa_select_positions_f32(iq, ik, iqnw, iknw, 1e-6f, 1e7f, rot, idx_h, idx_d, cr, topk, pos, ik_cpu);
      std::vector<float> o_cpu(nh * hd);
      full_attn_qsa_step_f32(qg, kr, vr, qnw, knw, 1e-6f, 1e7f, nh, nkvh, hd, rot, pos, sel, k_cpu, v_cpu, o_cpu);

      DevVec<float> dqg(qg), dkr(kr), dvr(vr), diq(iq), dik(ik);
      qsa_indexer_append_and_score_f32_cuda(diq.p, dik.p, diqnw.p, diknw.p, 1e-6f, 1e7f, rot, idx_h, idx_d, cr, pos,
                                           dikc.p, dscores.p, stream);
      std::vector<int32_t> sel_i32(sel.begin(), sel.end());
      DevVec<int32_t> dsel(sel_i32);
      full_attn_qsa_step_f32_cuda(dqg.p, dkr.p, dvr.p, dqnw.p, dknw.p, 1e-6f, 1e7f, nh, nkvh, hd, rot, pos, dsel.p,
                                  static_cast<int>(sel_i32.size()), dkc.p, dvc.p, dq_sc.p, dg_sc.p, dout.p, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CHECK(max_abs_diff(o_cpu, dout.to_host()) < 2e-4f);
    }
  }

  // 5. Top-10 Softmax router + shared expert gate add
  {
    const auto rlog = rand_f32(rng, 512, 1.5f);
    const auto cpu_ch = route_top_k_softmax(rlog, 10, 1.0f);
    DevVec<float> drlog(rlog), dw(10);
    DevVec<int32_t> dexp(10);
    route_top_k_softmax_f32_cuda(drlog.p, 512, 10, 1.0f, dexp.p, dw.p, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto gexp = dexp.to_host();
    const auto gw = dw.to_host();
    for (int k = 0; k < 10; ++k) {
      CHECK(gexp[k] == cpu_ch[k].expert);
      CHECK(std::fabs(gw[k] - cpu_ch[k].weight) < 1e-6f);
    }
  }

  std::printf("PASS test_hc_ple_gdn_qsa_moe\n");
}

}  // namespace

int main() {
  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
    std::printf("skip: no CUDA device available\n");
    return 77;
  }
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));
  std::mt19937 rng(20261126u);
  test_ggml_and_elastic_linear(rng, stream);
  test_hc_ple_gdn_qsa_moe(rng, stream);
  CUDA_CHECK(cudaStreamDestroy(stream));
  std::printf("qwen4exp_cuda_ops: PASS\n");
  return 0;
}

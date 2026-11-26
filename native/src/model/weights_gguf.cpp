// Weights from a llama.cpp GGUF file of Inkling-2-Mini (arch "inkling2_loop", unsloth's GGUF_shard_rewrite layout:
// gate/up experts fused gate-rows-first, convolution weights raw), in the decoder's device layouts, optionally
// tensor/expert-parallel sharded exactly like load_weights_sharded.
//
// Nothing is re-quantized:
//   * NVFP4 experts (layers 0-19) are repacked bit for bit into ingine's NVFP4 layout: GGUF blocks of 64 values
//     [4 UE4M3 sub-block scales | 32 code bytes, byte k of sub-block s = values 16s+k (low) and 16s+k+8 (high)] become
//     E2M1 bytes (value 2i low, 2i+1 high) + one E4M3 scale per 16 (the same 7 bits), and the tensor's per-expert
//     ".scale" is the expert factor. w13 rows are interleaved (2i gate, 2i+1 up) as the kernels expect.
//   * K-quant / legacy-quant experts (layers 20-23: Q5_K, Q4_K, Q5_0 -> GGUF_Q5L; IQ4_XS, IQ4_NL -> GGUF_IQ4) become
//     32-value groups with the group's exact FP32 scale (and min): every value is the one ggml's dequantize_row_*
//     forms (scale * code + min, or scale * iq4nl[code]).
//   * Dense projections, the embedding and the head (Q8_0 / Q6_K / Q5_K / BF16 in the file) are expanded to the
//     activation type (FP16 on Volta) on load: exactly the values ggml computes, rounded once.
#include "model/weights_gguf.h"

#include "core/act.h"
#include "core/gguf.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace ingine::model {
namespace fs = std::filesystem;
namespace {

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("weights_gguf: ") + what + ": " + cudaGetErrorString(e));
}

template <class F>
void parallel_for(std::int64_t n, F&& f) {
  const int nt = std::max(1, std::min<int>(16, int(std::thread::hardware_concurrency()) / 2));
  std::atomic<std::int64_t> next{0};
  std::vector<std::thread> th;
  for (int t = 0; t < nt; ++t)
    th.emplace_back([&] {
      for (std::int64_t i; (i = next.fetch_add(1)) < n;) f(i);
    });
  for (auto& t : th) t.join();
}

std::uint16_t act_bits(float f) {
#if INGINE_ACT_FP16
  const __half h = __float2half_rn(f);
#else
  const __nv_bfloat16 h = __float2bfloat16_rn(f);
#endif
  std::uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}

class GgufLoader {
 public:
  GgufLoader(const GgufFile& f, Weights& w) : f_(f), w_(w) {}

  void* alloc(std::size_t bytes) {
    w_.storage.emplace_back(std::max<std::size_t>(bytes, 16));
    return w_.storage.back().p;
  }
  void* upload(const void* src, std::size_t bytes) {
    void* d = alloc(bytes);
    ck(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice), "upload");
    return d;
  }
  // rows [r0, r0 + n) of a 2-D (or flattened) tensor as FP32
  std::vector<float> rows_f32(const std::string& name, std::int64_t r0, std::int64_t n) const {
    const GgufTensor& t = f_.at(name);
    if (r0 < 0 || r0 + n > t.rows()) throw std::runtime_error("weights_gguf: rows out of range in " + name);
    std::vector<float> out(std::size_t(n) * std::size_t(t.row_len()));
    parallel_for(n, [&](std::int64_t i) {
      ggml_dequantize(t.type, t.row(r0 + i), t.row_len(), out.data() + std::size_t(i) * t.row_len());
    });
    return out;
  }
  std::vector<float> all_f32(const std::string& name) const { return rows_f32(name, 0, f_.at(name).rows()); }
  std::int64_t numel(const std::string& name) const { return f_.at(name).numel(); }
  Tensor f32(const std::vector<float>& v, std::initializer_list<std::int32_t> shape) {
    return rm(upload(v.data(), v.size() * 4), DType::FP32, shape);
  }
  Tensor f32(const std::string& name, std::initializer_list<std::int32_t> shape) {
    std::int64_t n = 1;
    for (auto s : shape) n *= s;
    if (numel(name) != n) throw std::runtime_error("weights_gguf: " + name + " has an unexpected shape");
    return f32(all_f32(name), shape);
  }
  Tensor act(const std::vector<float>& v, std::initializer_list<std::int32_t> shape) {
    std::vector<std::uint16_t> h(v.size());
    parallel_for(std::int64_t(v.size() + 65535) / 65536, [&](std::int64_t c) {
      const std::size_t a = std::size_t(c) * 65536, b = std::min(v.size(), a + 65536);
      for (std::size_t i = a; i < b; ++i) h[i] = act_bits(v[i]);
    });
    return rm(upload(h.data(), h.size() * 2), kActDType, shape);
  }
  const GgufFile& file() const { return f_; }
  void note_q8(bool q8) { w_.dense_q8 = w_.dense_q8 || q8; }

 private:
  const GgufFile& f_;
  Weights& w_;
};

void check_metadata(const GgufFile& f) {
  const auto arch = f.get_string("general.architecture");
  if (!arch || *arch != "inkling2_loop")
    throw std::runtime_error("weights_gguf: expected general.architecture inkling2_loop, got " + arch.value_or("none"));
  const std::string a = "inkling2_loop.";
  auto eq = [&](const char* key, std::int64_t want) {
    const auto v = f.get_int(a + key);
    if (!v || *v != want)
      throw std::runtime_error("weights_gguf: " + a + key + " = " + (v ? std::to_string(*v) : "missing") +
                               ", the engine is built for " + std::to_string(want));
  };
  auto near = [&](const char* key, double want) {
    const auto v = f.get_float(a + key);
    if (!v || std::fabs(*v - want) > 1e-4 * std::max(1.0, std::fabs(want)))
      throw std::runtime_error("weights_gguf: " + a + key + " differs from the engine's constant");
  };
  eq("block_count", LAYERS);
  eq("embedding_length", HIDDEN);
  eq("expert_count", EXPERTS);
  eq("expert_used_count", TOP_K);
  eq("expert_feed_forward_length", INTER);
  eq("attention.key_length", HEAD_DIM);
  eq("attention.sliding_window", WINDOW);
  eq("attention.relative_length", D_REL);
  eq("attention.relative_extent", REL_EXTENT);
  eq("loop.count", PASSES);
  eq("loop.start", LOOP_START);
  eq("loop.end", LOOP_END);
  eq("loop.gate_rank", GATE_RANK);
  eq("attn_res.block_size", BLOCK);
  eq("vocab_size", VOCAB);
  eq("vocab_size_unpadded", UNPADDED_VOCAB);
  near("attention.qk_clip", QK_CLIP);
  near("attention.relative_clip", REL_CLIP);
  near("attention.log_scale_alpha", LOG_ALPHA);
  near("attention.log_scale_floor", LOG_FLOOR);
  near("expert_weights_scale", ROUTE_SCALE);
  near("activation.situ_beta", GATE_SOFTCAP);
  near("activation.situ_linear_beta", UP_SOFTCAP);
  near("final_logit_softcapping", LOGIT_SOFTCAP);
}

// --------------------------------------------------------------------------------------------- experts

ExpertFormat gguf_expert_format(GgmlType t) {
  switch (t) {
    case GgmlType::NVFP4: return ExpertFormat::NVFP4;
    case GgmlType::Q5_K: case GgmlType::Q4_K: case GgmlType::Q5_0: return ExpertFormat::GGUF_Q5L;
    case GgmlType::IQ4_XS: case GgmlType::IQ4_NL: return ExpertFormat::GGUF_IQ4;
    default:
      throw std::runtime_error(std::string("weights_gguf: expert type ") + ggml_type_name(t) + " is not supported");
  }
}

// Codes / scales bytes of one row of K values in an ingine expert format (gguf_expert_row_bytes in the header).
void row_bytes(ExpertFormat fmt, int K, std::size_t& code, std::size_t& scale) {
  switch (fmt) {
    case ExpertFormat::NVFP4: code = K / 2, scale = K / 16; return;
    case ExpertFormat::GGUF_Q5L: code = K / 2, scale = std::size_t(K / 32) * 12; return;
    case ExpertFormat::GGUF_IQ4: code = K / 2, scale = std::size_t(K / 32) * 4; return;
    default: throw std::runtime_error("weights_gguf: format has no GGUF row layout");
  }
}

// One GGUF row of K values -> ingine row bytes (codes, scales). Exact: see the file comment.
void repack_row(GgmlType t, const std::uint8_t* src, int K, std::uint8_t* code, std::uint8_t* scale) {
  auto f16 = [](const std::uint8_t* p) {
    std::uint16_t h;
    std::memcpy(&h, p, 2);
    __half x;
    std::memcpy(&x, &h, 2);
    return __half2float(x);
  };
  // a group of 32 values: nibbles (value 2i low, 2i + 1 high), 5th bits, scale and min
  auto put_group = [&](int g, const std::uint8_t (&q)[32], float s, float m, bool five, bool has_min) {
    std::uint32_t hb = 0;
    for (int i = 0; i < 16; ++i) code[16 * g + i] = std::uint8_t((q[2 * i] & 15) | ((q[2 * i + 1] & 15) << 4));
    for (int i = 0; i < 32; ++i) hb |= std::uint32_t((q[i] >> 4) & 1) << i;
    if (has_min) {
      std::memcpy(scale + 12 * g, &s, 4);
      std::memcpy(scale + 12 * g + 4, &m, 4);
      std::memcpy(scale + 12 * g + 8, &hb, 4);
    } else {
      std::memcpy(scale + 4 * g, &s, 4);
    }
    (void) five;
  };
  switch (t) {
    case GgmlType::NVFP4:
      for (int b = 0; b < K / 64; ++b) {
        const std::uint8_t* blk = src + 36 * b;
        for (int s = 0; s < 4; ++s) {
          std::uint8_t v[16];
          for (int k = 0; k < 8; ++k) v[k] = blk[4 + 8 * s + k] & 15, v[k + 8] = blk[4 + 8 * s + k] >> 4;
          for (int i = 0; i < 8; ++i) code[32 * b + 8 * s + i] = std::uint8_t(v[2 * i] | (v[2 * i + 1] << 4));
          const std::uint8_t sc = blk[s] & 0x7f;
          scale[4 * b + s] = sc == 0x7f ? 0 : sc;  // (ggml reads the UE4M3 NaN as 0)
        }
      }
      return;
    case GgmlType::Q5_0:
      for (int g = 0; g < K / 32; ++g) {
        const std::uint8_t* blk = src + 22 * g;
        const float d = f16(blk);
        std::uint32_t qh;
        std::memcpy(&qh, blk + 2, 4);
        std::uint8_t q[32];
        for (int j = 0; j < 16; ++j) {
          q[j] = std::uint8_t((blk[6 + j] & 15) | (((qh >> j) << 4) & 0x10));
          q[j + 16] = std::uint8_t((blk[6 + j] >> 4) | ((qh >> (j + 12)) & 0x10));
        }
        put_group(g, q, d, -16.0f * d, true, true);  // d * (q - 16) = d q - 16 d (16 d exact)
      }
      return;
    case GgmlType::Q4_K:
    case GgmlType::Q5_K:
      for (int sb = 0; sb < K / 256; ++sb) {
        const bool q5 = t == GgmlType::Q5_K;
        const std::uint8_t* blk = src + (q5 ? 176 : 144) * sb;
        const float d = f16(blk), dmin = f16(blk + 2);
        const std::uint8_t* sc = blk + 4;
        const std::uint8_t* qh = q5 ? blk + 16 : nullptr;
        const std::uint8_t* ql = q5 ? blk + 48 : blk + 16;
        for (int j = 0; j < 4; ++j) {  // 64 values: groups 2 j (low nibbles) and 2 j + 1 (high nibbles)
          for (int h = 0; h < 2; ++h) {
            const int is = 2 * j + h;
            std::uint8_t s, m;
            if (is < 4) {
              s = sc[is] & 63, m = sc[is + 4] & 63;
            } else {
              s = std::uint8_t((sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4));
              m = std::uint8_t((sc[is + 4] >> 4) | ((sc[is] >> 6) << 4));
            }
            std::uint8_t q[32];
            const std::uint8_t u = std::uint8_t(1u << is);
            for (int l = 0; l < 32; ++l) {
              const std::uint8_t lo = h ? (ql[32 * j + l] >> 4) : (ql[32 * j + l] & 15);
              q[l] = std::uint8_t(lo | (q5 && (qh[l] & u) ? 16 : 0));
            }
            put_group(8 * sb + is, q, d * float(s), -(dmin * float(m)), q5, true);
          }
        }
      }
      return;
    case GgmlType::IQ4_NL:
      for (int g = 0; g < K / 32; ++g) {
        const std::uint8_t* blk = src + 18 * g;
        std::uint8_t q[32];
        for (int j = 0; j < 16; ++j) q[j] = blk[2 + j] & 15, q[j + 16] = blk[2 + j] >> 4;
        put_group(g, q, f16(blk), 0.f, false, false);
      }
      return;
    case GgmlType::IQ4_XS:
      for (int sb = 0; sb < K / 256; ++sb) {
        const std::uint8_t* blk = src + 136 * sb;
        const float d = f16(blk);
        std::uint16_t sh;
        std::memcpy(&sh, blk + 2, 2);
        for (int ib = 0; ib < 8; ++ib) {
          const int ls = ((blk[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
          const std::uint8_t* qs = blk + 8 + 16 * ib;
          std::uint8_t q[32];
          for (int j = 0; j < 16; ++j) q[j] = qs[j] & 15, q[j + 16] = qs[j] >> 4;
          put_group(8 * sb + ib, q, d * float(ls - 32), 0.f, false, false);
        }
      }
      return;
    default:
      throw std::runtime_error(std::string("weights_gguf: cannot repack ") + ggml_type_name(t));
  }
}

// EP-sharded experts of one projection ("gate_up" -> w13 with interleaved rows, "down" -> w2).
void load_experts(GgufLoader& ld, const std::string& b, const char* proj, int e_begin, int e_per, ExpertFormat& fmt,
                  Tensor& wt, Tensor& sc, Tensor& fac, Tensor& wptrs, Tensor& sptrs) {
  const GgufFile& f = ld.file();
  const GgufTensor& t = f.at(b + "ffn_" + proj + "_exps.weight");
  const bool up = std::strcmp(proj, "gate_up") == 0;
  const int K = int(t.row_len());
  const int rows = up ? 2 * INTER : HIDDEN;
  if (K != (up ? HIDDEN : INTER) || t.rows() != std::int64_t(rows) * EXPERTS)
    throw std::runtime_error("weights_gguf: " + t.name + " has an unexpected shape");
  fmt = gguf_expert_format(t.type);
  std::size_t cb, sbytes;
  row_bytes(fmt, K, cb, sbytes);
  const std::size_t c_exp = cb * rows, s_exp = sbytes * rows;
  std::vector<std::uint8_t> codes(c_exp * e_per), scales(s_exp * e_per);
  parallel_for(std::int64_t(e_per) * rows, [&](std::int64_t i) {
    const int el = int(i / rows), r = int(i % rows);
    const int e = e_begin + el;
    const int src_r = up ? ((r & 1) ? INTER + r / 2 : r / 2) : r;  // w13: row 2i gate i, 2i + 1 up i
    repack_row(t.type, reinterpret_cast<const std::uint8_t*>(t.row(std::int64_t(e) * rows + src_r)), K,
               codes.data() + el * c_exp + std::size_t(r) * cb, scales.data() + el * s_exp + std::size_t(r) * sbytes);
  });
  void* wd = ld.upload(codes.data(), codes.size());
  void* sd = ld.upload(scales.data(), scales.size());
  wt = rm(wd, DType::U8, {e_per, std::int32_t(c_exp)});
  sc = rm(sd, fmt == ExpertFormat::NVFP4 ? DType::FP8_E4M3FN : DType::U8, {e_per, std::int32_t(s_exp)});
  std::vector<const std::uint8_t*> wp(EXPERTS, nullptr), sp(EXPERTS, nullptr);
  for (int i = 0; i < e_per; ++i) {
    wp[e_begin + i] = static_cast<const std::uint8_t*>(wd) + std::size_t(i) * c_exp;
    sp[e_begin + i] = static_cast<const std::uint8_t*>(sd) + std::size_t(i) * s_exp;
  }
  wptrs = rm(ld.upload(wp.data(), EXPERTS * sizeof(void*)), DType::I64, {EXPERTS});
  sptrs = rm(ld.upload(sp.data(), EXPERTS * sizeof(void*)), DType::I64, {EXPERTS});
  std::vector<float> factor(EXPERTS, 1.0f);
  const std::string sn = b + "ffn_" + proj + "_exps.scale";
  if (f.find(sn)) {
    factor = ld.all_f32(sn);
  } else if (fmt == ExpertFormat::NVFP4) {
    throw std::runtime_error("weights_gguf: NVFP4 experts without " + sn);
  }
  fac = ld.f32(factor, {EXPERTS});
}


// ---- block-table dense matrices (Volta): Q8_0 16-row blocks kept as int8 + FP16 scales, other blocks expanded -----
bool q8_dense_enabled() {
#if INGINE_ACT_FP16
  static const bool on = [] {
    const char* e = std::getenv("INGINE_GGUF_Q8_DENSE");
    return !(e && e[0] == '0');
  }();
  return on;
#else
  return false;
#endif
}

struct BlockedBuilder {
  int K = 0;
  std::vector<std::uint8_t> bytes;
  std::vector<std::uint16_t> scales;
  std::vector<std::int64_t> off;
  std::vector<std::int32_t> soff;
  bool any_q8 = false;
  // 16 rows of a Q8_0 tensor, columns [32 cb0, 32 (cb0 + nb)) of rows r0 .. r0 + 15
  void q8(const GgufTensor& t, std::int64_t r0, int cb0, int nb) {
    if (nb * 32 != K) throw std::runtime_error("weights_gguf: Q8_0 block width");
    off.push_back(std::int64_t(bytes.size()));
    soff.push_back(std::int32_t(scales.size()));
    for (int r = 0; r < 16; ++r) {
      const std::uint8_t* row = reinterpret_cast<const std::uint8_t*>(t.row(r0 + r));
      for (int b = 0; b < nb; ++b) {
        const std::uint8_t* blk = row + 34 * std::size_t(cb0 + b);
        std::uint16_t d;
        std::memcpy(&d, blk, 2);
        scales.push_back(d);
        bytes.insert(bytes.end(), blk + 2, blk + 34);
      }
    }
    any_q8 = true;
  }
  // 16 rows of FP32 values (zeros for padding) as activation-type rows
  void act(const float* v) {
    off.push_back(std::int64_t(bytes.size()));
    soff.push_back(-1);
    for (int i = 0; i < 16 * K; ++i) {
      const std::uint16_t h = act_bits(v[i]);
      bytes.push_back(std::uint8_t(h & 0xff));
      bytes.push_back(std::uint8_t(h >> 8));
    }
  }
  Dense upload(GgufLoader& ld, int rows) {
    while (bytes.size() % 16) bytes.push_back(0);
    Dense d;
    d.w = rm(ld.upload(bytes.data(), bytes.size()), DType::U8, {rows, K});
    d.blk_off = rm(ld.upload(off.data(), off.size() * 8), DType::I64, {std::int32_t(off.size())});
    d.blk_soff = rm(ld.upload(soff.data(), soff.size() * 4), DType::I32, {std::int32_t(soff.size())});
    if (scales.empty()) scales.push_back(0);
    d.q8_scale = rm(ld.upload(scales.data(), scales.size() * 2), DType::FP16, {std::int32_t(scales.size())});
    return d;
  }
};

// --------------------------------------------------------------------------------------------- layers

Layer load_layer_gguf(GgufLoader& ld, int l, int tp_rank, int tp_size) {
  const GgufFile& f = ld.file();
  const std::string b = "blk." + std::to_string(l) + ".";
  const int H = HIDDEN;
  Layer layer;
  layer.idx = l;
  layer.full = is_full_layer(l);
  layer.loop = is_loop_layer(l);

  if (layer.full) {
    layer.q_norm = ld.f32(b + "attn_q_inp_norm.weight", {H});
    layer.k_norm = ld.f32(b + "attn_k_inp_norm.weight", {H});
    layer.v_norm = ld.f32(b + "attn_v_inp_norm.weight", {H});
  } else {
    layer.attn_norm = ld.f32(b + "attn_norm.weight", {H});
  }
  layer.mlp_norm = ld.f32(b + "ffn_norm.weight", {H});

  // projections in SEG order, this rank's heads (q, gate, rel, sink) and KV heads (k, v, k_dyn, v_dyn)
  {
    const int R = proj_rows(layer.full);
    std::vector<float> wp(std::size_t(R) * H, 0.0f);
    struct Src { const char* name; int rows; };
    const Src src[NUM_SEGS] = {{"attn_q.weight", HEADS * HEAD_DIM},       {"attn_gate.weight", HEADS * HEAD_DIM},
                               {"attn_rel.weight", HEADS * D_REL},        {"attn_sink_proj.weight", HEADS},
                               {"attn_k.weight", KV_DIM},                  {"attn_v.weight", KV_DIM},
                               {"attn_k_conv_proj.weight", KV_HEADS * CONV_K}, {"attn_v_conv_proj.weight", KV_HEADS * CONV_K}};
    if (q8_dense_enabled()) {  // a format per 16-row block: Q8_0 tensors' blocks raw, everything else expanded
      BlockedBuilder bb;
      bb.K = H;
      for (int r0 = 0; r0 < R; r0 += 16) {
        int sg = 0;
        while (SEG_BEGIN[sg + 1] <= r0) ++sg;
        const int lr = r0 - SEG_BEGIN[sg];
        const GgufTensor& t = f.at(b + src[sg].name);
        if (t.type == GgmlType::Q8_0 && lr + 16 <= src[sg].rows) {
          bb.q8(t, std::int64_t(tp_rank) * src[sg].rows + lr, 0, H / 32);
        } else {
          std::vector<float> v(std::size_t(16) * H, 0.0f);
          const int n = std::max(0, std::min(16, src[sg].rows - lr));
          if (n > 0) {
            const std::vector<float> rr = ld.rows_f32(b + src[sg].name, std::int64_t(tp_rank) * src[sg].rows + lr, n);
            std::copy(rr.begin(), rr.end(), v.begin());
          }
          bb.act(v.data());
        }
      }
      layer.wproj = bb.upload(ld, R);
      ld.note_q8(bb.any_q8);
    } else {
      for (int s = 0; s < num_segs(layer.full); ++s) {
        const std::vector<float> v = ld.rows_f32(b + src[s].name, std::int64_t(tp_rank) * src[s].rows, src[s].rows);
        std::copy(v.begin(), v.end(), wp.begin() + std::size_t(SEG_BEGIN[s]) * H);
      }
      layer.wproj = Dense{ld.act(wp, {R, H}), {}};
    }
    Dense* views[NUM_SEGS] = {&layer.wq, &layer.wgate, &layer.wr, &layer.wsink,
                              &layer.wk, &layer.wv,    &layer.wk_dyn, &layer.wv_dyn};
    const std::size_t es = 2;
    for (int s = 0; s < num_segs(layer.full) && !layer.wproj.blocked(); ++s) {  // (no row views of a block table)
      const int r0 = SEG_BEGIN[s], r1 = SEG_BEGIN[s + 1];
      views[s]->w = rm(static_cast<char*>(layer.wproj.w.data) + std::size_t(r0) * H * es, kActDType, {r1 - r0, H});
    }
  }
  // wo [H, 3072] column-sharded to this rank's heads
  {
    const int full_cols = TOTAL_HEADS * HEAD_DIM, local = HEADS * HEAD_DIM, c0 = tp_rank * local;
    const GgufTensor& wot = f.at(b + "attn_output.weight");
    if (wot.row_len() != full_cols) throw std::runtime_error("weights_gguf: attn_output shape");
    if (q8_dense_enabled() && wot.type == GgmlType::Q8_0) {  // this rank's columns are whole Q8_0 blocks
      BlockedBuilder bb;
      bb.K = local;
      for (int r0 = 0; r0 < H; r0 += 16) bb.q8(wot, r0, c0 / 32, local / 32);
      layer.wo = bb.upload(ld, H);
      ld.note_q8(true);
      goto wo_done;
    }
    {
    const std::vector<float> wo = ld.rows_f32(b + "attn_output.weight", 0, H);
    std::vector<float> sl(std::size_t(H) * local);
    for (int r = 0; r < H; ++r)
      std::copy(wo.begin() + std::size_t(r) * full_cols + c0, wo.begin() + std::size_t(r) * full_cols + c0 + local,
                sl.begin() + std::size_t(r) * local);
    layer.wo = Dense{ld.act(sl, {H, local}), {}};
    }
  wo_done:;
  }
  {  // clip scales of this rank's heads: file [24][128] and [24][16]
    layer.q_scale = ld.f32(ld.rows_f32(b + "attn_q_clip.weight", std::int64_t(tp_rank) * HEADS, HEADS), {HEADS, HEAD_DIM});
    layer.r_scale = ld.f32(ld.rows_f32(b + "attn_rel_clip.weight", std::int64_t(tp_rank) * HEADS, HEADS), {HEADS, D_REL});
  }
  {  // rel_proj [16, 1024] from the file's [1024][16]
    const std::vector<float> v = ld.all_f32(b + "attn_rel_bias.weight");
    if (v.size() != std::size_t(D_REL) * REL_EXTENT) throw std::runtime_error("weights_gguf: attn_rel_bias shape");
    std::vector<float> t(v.size());
    for (int e = 0; e < REL_EXTENT; ++e)
      for (int r = 0; r < D_REL; ++r) t[std::size_t(r) * REL_EXTENT + e] = v[std::size_t(e) * D_REL + r];
    layer.rel_proj = ld.f32(t, {D_REL, REL_EXTENT});
  }
  {  // K/V short convs: file [1024 channels][4 taps], tap 3 = the current token; this rank's KV channels
    const int c0 = tp_rank * KV_DIM;
    auto conv = [&](const char* n) { return ld.rows_f32(b + n, c0, KV_DIM); };
    const std::vector<float> kc = conv("attn_k_conv.weight"), vc = conv("attn_v_conv.weight");
    if (layer.full) {  // static weights by lag (row 0 = current): lag l = tap 3 - l
      auto by_lag = [&](const std::vector<float>& w) {
        std::vector<float> o(std::size_t(CONV_K) * KV_DIM);
        for (int c = 0; c < KV_DIM; ++c)
          for (int lag = 0; lag < CONV_K; ++lag) o[std::size_t(lag) * KV_DIM + c] = w[std::size_t(c) * CONV_K + 3 - lag];
        return o;
      };
      layer.k_static = ld.f32(by_lag(kc), {CONV_K, KV_DIM});
      layer.v_static = ld.f32(by_lag(vc), {CONV_K, KV_DIM});
    } else {
      layer.k_conv = ld.f32(kc, {KV_DIM, CONV_K});
      layer.v_conv = ld.f32(vc, {KV_DIM, CONV_K});
    }
  }
  layer.attn_conv = ld.f32(b + "attn_post_conv.weight", {H, CONV_K});
  layer.mlp_conv = ld.f32(b + "ffn_post_conv.weight", {H, CONV_K});
  {
    const std::vector<float> r = ld.all_f32(b + "ffn_gate_inp.weight");
    if (r.size() != std::size_t(EXPERTS) * H) throw std::runtime_error("weights_gguf: router shape");
    layer.router = ld.f32(r, {EXPERTS, H});
    layer.router_bf16 = ld.act(r, {EXPERTS, H});
  }
  layer.router_bias = ld.f32(b + "exp_probs_b.bias", {EXPERTS});
  if (l > 0) {
    const std::string qn = b + "attn_res_query.weight";
    const std::int32_t nq = std::int32_t(ld.numel(qn) / H);
    layer.attn_query = ld.f32(ld.all_f32(qn), {nq, H});
  }
  layer.mlp_query = ld.f32(b + "ffn_res_query.weight", {1, H});
  {
    std::vector<float> n3(std::size_t(3) * H);
    if (layer.full) {
      const char* names[3] = {"attn_q_inp_norm.weight", "attn_k_inp_norm.weight", "attn_v_inp_norm.weight"};
      for (int g = 0; g < 3; ++g) {
        const std::vector<float> v = ld.all_f32(b + names[g]);
        std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
      }
    } else {
      const std::vector<float> v = ld.all_f32(b + "attn_norm.weight");
      for (int g = 0; g < 3; ++g) std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
    }
    layer.attn_norm3 = ld.f32(n3, {3, H});
    layer.attn_norm_mix = layer.full ? layer.attn_norm3 : rm(layer.attn_norm3.data, DType::FP32, {1, H});
    layer.mlp_norm1 = rm(layer.mlp_norm.data, DType::FP32, {1, H});
  }
  if (layer.loop) {  // message gates: q, k, v, mlp controllers; down [64][2H], forget / gain [H][64]
    const char* ctrl[4] = {"q_ctrl", "k_ctrl", "v_ctrl", "ffn_ctrl"};
    std::vector<float> down[4], forget[4], gain[4];
    for (int g = 0; g < 4; ++g) {
      const std::string c = b + "loop_" + ctrl[g] + "_";
      down[g] = ld.all_f32(c + "down.weight");
      forget[g] = ld.all_f32(c + "forget.weight");
      gain[g] = ld.all_f32(c + "gain.weight");
      if (down[g].size() != std::size_t(GATE_RANK) * 2 * H || forget[g].size() != std::size_t(H) * GATE_RANK)
        throw std::runtime_error("weights_gguf: loop gate shapes in layer " + std::to_string(l));
    }
    auto stack = [](std::vector<float>* parts, int n) {
      std::vector<float> out;
      for (int i = 0; i < n; ++i) out.insert(out.end(), parts[i].begin(), parts[i].end());
      return out;
    };
    const int D2 = 2 * H;
    layer.qkv_gate_down = ld.act(stack(down, 3), {3, GATE_RANK, D2});
    layer.qkv_gate_forget = ld.f32(stack(forget, 3), {3, H, GATE_RANK});
    layer.qkv_gate_gain = ld.f32(stack(gain, 3), {3, H, GATE_RANK});
    layer.qkv_gate_forget16 = ld.act(stack(forget, 3), {3, H, GATE_RANK});
    layer.qkv_gate_gain16 = ld.act(stack(gain, 3), {3, H, GATE_RANK});
    layer.mlp1_gate_down = ld.act(down[3], {1, GATE_RANK, D2});
    layer.mlp1_gate_forget = ld.f32(forget[3], {1, H, GATE_RANK});
    layer.mlp1_gate_gain = ld.f32(gain[3], {1, H, GATE_RANK});
    layer.mlp1_gate_forget16 = ld.act(forget[3], {1, H, GATE_RANK});
    layer.mlp1_gate_gain16 = ld.act(gain[3], {1, H, GATE_RANK});
    auto at = [](const Tensor& st, int g, DType dt, std::int32_t a, std::int32_t c) {
      const std::size_t es = dt == DType::FP32 ? 4 : 2;
      return rm(static_cast<char*>(st.data) + std::size_t(g) * a * c * es, dt, {a, c});
    };
    Tensor* downs[3] = {&layer.q_gate_down, &layer.k_gate_down, &layer.v_gate_down};
    Tensor* forgets[3] = {&layer.q_gate_forget, &layer.k_gate_forget, &layer.v_gate_forget};
    Tensor* gains[3] = {&layer.q_gate_gain, &layer.k_gate_gain, &layer.v_gate_gain};
    for (int g = 0; g < 3; ++g) {
      *downs[g] = at(layer.qkv_gate_down, g, kActDType, GATE_RANK, D2);
      *forgets[g] = at(layer.qkv_gate_forget, g, DType::FP32, H, GATE_RANK);
      *gains[g] = at(layer.qkv_gate_gain, g, DType::FP32, H, GATE_RANK);
    }
    layer.mlp_gate_down = at(layer.mlp1_gate_down, 0, kActDType, GATE_RANK, D2);
    layer.mlp_gate_forget = at(layer.mlp1_gate_forget, 0, DType::FP32, H, GATE_RANK);
    layer.mlp_gate_gain = at(layer.mlp1_gate_gain, 0, DType::FP32, H, GATE_RANK);
  }

  Experts& ex = layer.experts;
  ex.ep_rank = tp_rank;
  ex.ep_size = tp_size;
  const int e_per = EXPERTS / tp_size, e_begin = tp_rank * e_per;
  load_experts(ld, b, "gate_up", e_begin, e_per, ex.w13_fmt, ex.w13, ex.w13_scale, ex.w13_factor, ex.w13_ptrs,
               ex.w13_scale_ptrs);
  load_experts(ld, b, "down", e_begin, e_per, ex.w2_fmt, ex.w2, ex.w2_scale, ex.w2_factor, ex.w2_ptrs,
               ex.w2_scale_ptrs);
  ex.fmt = ex.w13_fmt;
  {
    std::vector<std::uint8_t> mask(EXPERTS, 0);
    for (int i = 0; i < e_per; ++i) mask[e_begin + i] = 1;
    ex.active_mask = rm(ld.upload(mask.data(), EXPERTS), DType::U8, {EXPERTS});
  }
  return layer;
}

}  // namespace

fs::path gguf_model_file(const fs::path& model_dir) {
  std::error_code ec;
  if (fs::is_regular_file(model_dir, ec) && model_dir.extension() == ".gguf") return model_dir;
  const fs::path p = model_dir / "model.gguf";
  if (fs::exists(p, ec)) return p;
  return {};
}

bool is_gguf_model(const fs::path& model_dir) { return !gguf_model_file(model_dir).empty(); }

std::array<ExpertFormat, LAYERS> gguf_expert_formats(const fs::path& model_dir, bool w2) {
  const GgufFile f(gguf_model_file(model_dir));
  std::array<ExpertFormat, LAYERS> out{};
  for (int l = 0; l < LAYERS; ++l)
    out[l] = gguf_expert_format(f.at("blk." + std::to_string(l) + (w2 ? ".ffn_down_exps.weight" : ".ffn_gate_up_exps.weight")).type);
  return out;
}

Weights load_weights_gguf(const fs::path& model_dir, cudaStream_t stream, int tp_rank, int tp_size,
                          bool keep_full_unembed) {
  if (tp_size != TP_SIZE || tp_rank < 0 || tp_rank >= tp_size)
    throw std::invalid_argument("load_weights_gguf: invalid tp_rank/tp_size");
  if (keep_full_unembed && tp_size > 1)
    throw std::invalid_argument("load_weights_gguf: the drafter's full unembedding is not supported yet");
  const fs::path path = gguf_model_file(model_dir);
  if (path.empty()) throw std::runtime_error("load_weights_gguf: no model.gguf in " + model_dir.string());
  const GgufFile f(path);
  check_metadata(f);
  Weights w;
  w.tp_rank = tp_rank;
  w.tp_size = tp_size;
  w.dense_fp8 = false;
  w.storage.reserve(4096);
  GgufLoader ld(f, w);
  const int H = HIDDEN;
  for (int l = 0; l < LAYERS; ++l) {
    w.layers.push_back(load_layer_gguf(ld, l, tp_rank, tp_size));
    w.expert_formats[l] = w.layers.back().experts.w13_fmt;
    w.w2_expert_formats[l] = w.layers.back().experts.w2_fmt;
  }
  {  // message queries [12, 4, H]: per loop layer (q, k, v, mlp)
    std::vector<float> mq;
    for (int l = LOOP_START; l < LOOP_END; ++l) {
      const std::vector<float> v = ld.all_f32("blk." + std::to_string(l) + ".loop_msg_query.weight");
      if (v.size() != std::size_t(4) * H) throw std::runtime_error("weights_gguf: loop_msg_query shape");
      mq.insert(mq.end(), v.begin(), v.end());
    }
    w.message_queries = ld.f32(mq, {LOOP_END - LOOP_START, 4, H});
  }
  {
    const GgufTensor& e = f.at("token_embd.weight");
    if (e.row_len() != H || e.rows() != VOCAB) throw std::runtime_error("weights_gguf: token_embd shape");
    const char* q8e = std::getenv("INGINE_GGUF_Q8_EMBED");
    if (e.type == GgmlType::Q8_0 && !(q8e && q8e[0] == '0')) {  // rows read 4 a step: kept as Q8_0 (dense::embed_q8)
      w.embed = rm(ld.upload(e.data, e.bytes), DType::I8, {VOCAB, std::int32_t(e.row_bytes())});
    } else {
      w.embed = ld.act(ld.all_f32("token_embd.weight"), {VOCAB, H});
    }
  }
  w.embed_norm = ld.f32("token_embd_norm.weight", {H});
  w.final_query = ld.f32("output_res_query.weight", {1, H});
  w.final_norm = ld.f32("output_norm.weight", {H});
  {  // row-sharded unembedding [LOCAL_VOCAB_CHUNK, H] (rows past the unpadded vocabulary zero)
    const int v0 = tp_rank * LOCAL_VOCAB_CHUNK;
    const int v1 = std::min(UNPADDED_VOCAB, v0 + LOCAL_VOCAB_CHUNK);
    const int vr = std::max(0, v1 - v0);
    w.vocab_begin = v0;
    w.vocab_rows = vr;
    std::vector<float> u(std::size_t(LOCAL_VOCAB_CHUNK) * H, 0.0f);
    if (vr > 0) {
      const std::vector<float> rows = ld.rows_f32("output.weight", v0, vr);
      std::copy(rows.begin(), rows.end(), u.begin());
    }
    w.unembed = Dense{ld.act(u, {LOCAL_VOCAB_CHUNK, H}), {}};
  }
  ck(cudaStreamSynchronize(stream), "sync");
  std::size_t bytes = 0;
  for (const auto& s : w.storage) bytes += s.bytes;
  // stderr, not the server log: the model library also links into tools without the engine library
  std::fprintf(stderr, "weights_gguf: %s (rank %d/%d): %.2f GiB on the device\n", path.filename().string().c_str(),
               tp_rank, tp_size, double(bytes) / double(1ull << 30));
  return w;
}

}  // namespace ingine::model

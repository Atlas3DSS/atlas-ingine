// Weights in the decoder's device layouts, read from the Hugging Face checkpoint: weights.load / load_layer,
// plus fast.FastDecoder._concat_projections (one [R, H] projection matrix per layer, the per-projection
// matrices are row views of it), _stack_layer_params (stacked norms and gate weights) and _fp8_dense
// (wproj, wo and the unembedding as FP8 rows + FP32 row scales, the BF16 copies freed), and
// model.Ablation.load / none.
//
// Large tensors are uploaded straight from the file with the core DeviceUploader; small per-channel
// parameters are read on the host, widened to FP32 (exact from BF16) and copied.
#include "model/runtime.h"

#include "core/act.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "kernels/dense/dense.h"
#include "model/model_kernels.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace {

namespace fs = std::filesystem;

float bf16_bits_to_float(std::uint16_t b) {
  const std::uint32_t w = std::uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &w, 4);
  return f;
}

std::uint16_t float_to_bf16_bits(float f) {  // round to nearest even (torch .to(bfloat16))
  std::uint32_t w;
  std::memcpy(&w, &f, 4);
  if ((w & 0x7fffffffu) > 0x7f800000u) return std::uint16_t((w >> 16) | 0x40);
  return std::uint16_t((w + 0x7fffu + ((w >> 16) & 1u)) >> 16);
}

// Checkpoint values widened to FP32 on the host (BF16 -> FP32 exact, F32 as is).
std::vector<float> host_f32(const SafetensorsModel& m, const std::string& name) {
  const SafetensorsRef ref = m.at(name);
  const std::vector<std::byte> raw = m.read(name);
  const std::size_t n = std::size_t(ref->numel());
  std::vector<float> out(n);
  if (ref->dtype == "F32") {
    std::memcpy(out.data(), raw.data(), n * 4);
  } else if (ref->dtype == "BF16") {
    for (std::size_t i = 0; i < n; ++i) {
      std::uint16_t b;
      std::memcpy(&b, raw.data() + 2 * i, 2);
      out[i] = bf16_bits_to_float(b);
    }
  } else {
    throw std::runtime_error("weights: " + name + " has unsupported dtype " + ref->dtype);
  }
  return out;
}

class Loader {
 public:
  Loader(const SafetensorsModel& m, DeviceUploader& up, cudaStream_t stream, std::vector<DeviceBuffer>& storage)
      : m_(m), up_(up), stream_(stream), storage_(storage) {}

  void* alloc(std::size_t bytes) {
    storage_.emplace_back(bytes);
    return storage_.back().p;
  }

  Tensor upload_f32(const std::vector<float>& host, std::initializer_list<std::int32_t> shape) {
    void* d = alloc(host.size() * 4);
    CUDA_CHECK(cudaMemcpy(d, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    return rm(d, DType::FP32, shape);
  }

  Tensor upload_bf16(const std::vector<float>& host, std::initializer_list<std::int32_t> shape) {
    std::vector<std::uint16_t> b(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) b[i] = act::host_float_to_bits(host[i]);
    void* d = alloc(b.size() * 2);
    CUDA_CHECK(cudaMemcpy(d, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
    return rm(d, kActDType, shape);
  }

  // f32(name): checkpoint tensor as FP32 (weights.load_layer's `f32`), viewed with `shape`.
  Tensor f32(const std::string& name, std::initializer_list<std::int32_t> shape) {
    return upload_f32(host_f32(m_, name), shape);
  }

  // Queues the raw bytes of `name` (rows [0, rows) if rows >= 0) for upload to `dst`.
  void queue(const std::string& name, void* dst, std::int64_t rows = -1) {
    const SafetensorsRef ref = m_.at(name);
    UploadRequest r = upload_request(ref, dst);
    if (rows >= 0) r.bytes = ref->bytes / std::uint64_t(ref->shape[0]) * std::uint64_t(rows);
    pending_.push_back(r);
  }

  // Queues rows [row_begin, row_begin + row_count) of `name` for upload to `dst`.
  void queue_rows(const std::string& name, std::int64_t row_begin, std::int64_t row_count, void* dst) {
    const SafetensorsRef ref = m_.at(name);
    const std::uint64_t row_bytes = ref->bytes / std::uint64_t(ref->shape[0]);
    UploadRequest r = upload_request(ref, dst);
    r.offset += std::uint64_t(row_begin) * row_bytes;
    r.bytes = std::uint64_t(row_count) * row_bytes;
    pending_.push_back(r);
  }

  // Raw checkpoint tensor (any dtype) in new device memory, uploaded at flush().
  Tensor raw(const std::string& name) {
    const SafetensorsRef ref = m_.at(name);
    void* d = alloc(ref->bytes);
    queue(name, d);
    return ref->view(d);
  }

  void flush() {
    if (!pending_.empty()) up_.upload(pending_, stream_);
    pending_.clear();
  }

  const SafetensorsModel& model() const { return m_; }
  cudaStream_t stream() const { return stream_; }

 private:
  const SafetensorsModel& m_;
  DeviceUploader& up_;
  cudaStream_t stream_;
  std::vector<DeviceBuffer>& storage_;
  std::vector<UploadRequest> pending_;
};

struct LayerExpertFormats {
  ExpertFormat w13 = ExpertFormat::NVFP4;
  ExpertFormat w2 = ExpertFormat::NVFP4;
};

ExpertFormat parse_expert_dtype(const std::string& dt) {
  if (dt == "nvfp4") return ExpertFormat::NVFP4;
  if (dt == "mxfp8") return ExpertFormat::MXFP8;
  if (dt == "atlas2") return ExpertFormat::ATLAS2;
  if (dt == "atlas3") return ExpertFormat::ATLAS3;
  if (dt == "exl3_2") return ExpertFormat::EXL3_2;
  if (dt == "exl3_3") return ExpertFormat::EXL3_3;
  throw std::runtime_error("weights: unsupported expert format " + dt);
}

std::array<LayerExpertFormats, LAYERS> expert_formats(const fs::path& model_dir) {
  std::ifstream in(model_dir / "hf_quant_config.json");
  if (!in) throw std::runtime_error("weights: cannot open " + (model_dir / "hf_quant_config.json").string());
  const nlohmann::json q = nlohmann::json::parse(in);
  std::array<LayerExpertFormats, LAYERS> fmt{};
  std::array<bool, LAYERS> seen13{}, seen2{};
  const std::string s13 = "mlp.experts.w13_weight";
  const std::string s2 = "mlp.experts.w2_weight";
  const std::string pfx = "model.llm.layers.";
  for (const auto& group : q.at("model").at("parameter_groups")) {
    const std::string dt = group.at("compute_dtype").get<std::string>();
    for (const auto& t : group.at("tensors")) {
      const std::string name = t.get<std::string>();
      const bool is13 = name.size() >= s13.size() && name.compare(name.size() - s13.size(), s13.size(), s13) == 0;
      const bool is2 = !is13 && name.size() >= s2.size() && name.compare(name.size() - s2.size(), s2.size(), s2) == 0;
      if (!is13 && !is2) continue;
      const int l = std::stoi(name.substr(pfx.size(), name.find('.', pfx.size()) - pfx.size()));
      if (l < 0 || l >= LAYERS) continue;
      if (is13) {
        fmt[l].w13 = parse_expert_dtype(dt);
        seen13[l] = true;
      } else {
        fmt[l].w2 = parse_expert_dtype(dt);
        seen2[l] = true;
      }
    }
  }
  for (int l = 0; l < LAYERS; ++l) {
    if (!seen13[l]) throw std::runtime_error("weights: no w13 expert format for layer " + std::to_string(l));
    if (!seen2[l]) fmt[l].w2 = fmt[l].w13;
  }
  return fmt;
}

// FP8 rows + scales of an ActT matrix (dense fp8_rows), into new storage.
Dense to_fp8(Loader& ld, const Tensor& bf16, int rows, int cols) {
  void* q = ld.alloc(std::size_t(rows) * cols);
  float* s = static_cast<float*>(ld.alloc(std::size_t(rows) * 4));
  kernels::dense::fp8_rows(static_cast<const ActT*>(bf16.data), rows, cols, q, s, ld.stream());
  return Dense{rm(q, DType::FP8_E4M3FN, {rows, cols}), rm(s, DType::FP32, {rows})};
}

Dense row_view(const Dense& d, int begin, int end) {
  const int cols = d.cols();
  const std::size_t esize = d.fp8() ? 1 : 2;
  Dense v;
  v.w = rm(static_cast<char*>(d.w.data) + std::size_t(begin) * cols * esize, d.w.dtype, {end - begin, cols});
  if (d.fp8()) v.scale = rm(static_cast<float*>(d.scale.data) + begin, DType::FP32, {end - begin});
  return v;
}

Layer load_layer_into(Loader& ld, int l, LayerExpertFormats ef, bool dense_fp8, bool load_experts,
                      std::vector<float>* router_bias_out = nullptr) {
  const SafetensorsModel& m = ld.model();
  const std::string p = "model.llm.layers." + std::to_string(l) + ".";
  Layer layer;
  layer.idx = l;
  layer.full = is_full_layer(l);
  layer.loop = is_loop_layer(l);
  const int H = HIDDEN;

  if (layer.full) {
    layer.q_norm = ld.f32(p + "q_input_norm.weight", {H});
    layer.k_norm = ld.f32(p + "k_input_norm.weight", {H});
    layer.v_norm = ld.f32(p + "v_input_norm.weight", {H});
  } else {
    layer.attn_norm = ld.f32(p + "attn_norm.weight", {H});
  }
  layer.mlp_norm = ld.f32(p + "mlp_norm.weight", {H});

  // projections: one [R, H] matrix in SEG order (fast._concat_projections).
  // Supports pre-packed on-disk FP8 E4M3 (+ F32 *_scale) directly without BF16 staging, or BF16 on disk.
  static const char* seg_names[NUM_SEGS] = {"attn.wq_du.weight",        "attn.gate_attn.weight",
                                            "attn.wr_du.weight",        "attn.attn_sink_proj.weight",
                                            "attn.wk_dv.weight",        "attn.wv_dv.weight",
                                            "attn.k_sconv.weight_proj", "attn.v_sconv.weight_proj"};
  const int R = proj_rows(layer.full);
  const bool proj_fp8_disk = m.at(p + seg_names[0])->dtype == "F8_E4M3";
  const bool wo_fp8_disk = m.at(p + "attn.wo_ud.weight")->dtype == "F8_E4M3";
  DeviceBuffer proj_tmp, wo_tmp;
  Tensor wproj_bf16, wo_bf16;
  auto queue_bf16_rows = [&](const std::string& name, void* dst, int rows, int cols) {
    const SafetensorsRef ref = m.at(name);
    if (ref->dtype != "BF16" || ref->numel() != std::int64_t(rows) * cols)
      throw std::runtime_error("weights: " + name + " must be BF16 [" + std::to_string(rows) + ", " +
                               std::to_string(cols) + "]");
    ld.queue(name, dst);
  };
  auto queue_fp8_rows = [&](const std::string& name, void* q_dst, float* s_dst, int rows, int cols) {
    const SafetensorsRef ref = m.at(name);
    const SafetensorsRef sref = m.at(name + "_scale");
    if (ref->dtype != "F8_E4M3" || ref->numel() != std::int64_t(rows) * cols)
      throw std::runtime_error("weights: " + name + " must be F8_E4M3 [" + std::to_string(rows) + ", " +
                               std::to_string(cols) + "]");
    if (sref->dtype != "F32" || sref->numel() != std::int64_t(rows))
      throw std::runtime_error("weights: " + name + "_scale must be F32 [" + std::to_string(rows) + "]");
    ld.queue(name, q_dst);
    ld.queue(name + "_scale", s_dst);
  };
  if (proj_fp8_disk) {
    void* q = ld.alloc(std::size_t(R) * H);
    float* sc = static_cast<float*>(ld.alloc(std::size_t(R) * 4));
    for (int s = 0; s < num_segs(layer.full); ++s) {
      queue_fp8_rows(p + seg_names[s], static_cast<char*>(q) + std::size_t(SEG_BEGIN[s]) * H, sc + SEG_BEGIN[s],
                     SEG_ROWS[s], H);
    }
    layer.wproj = Dense{rm(q, DType::FP8_E4M3FN, {R, H}), rm(sc, DType::FP32, {R})};
  } else {
    void* d = dense_fp8 ? (proj_tmp = DeviceBuffer(std::size_t(R) * H * 2)).p : ld.alloc(std::size_t(R) * H * 2);
    for (int s = 0; s < num_segs(layer.full); ++s)
      queue_bf16_rows(p + seg_names[s], static_cast<char*>(d) + std::size_t(SEG_BEGIN[s]) * H * 2, SEG_ROWS[s], H);
    wproj_bf16 = rm(d, kActDType, {R, H});
  }
  if (wo_fp8_disk) {
    void* oq = ld.alloc(std::size_t(H) * HEADS * HEAD_DIM);
    float* osc = static_cast<float*>(ld.alloc(std::size_t(H) * 4));
    queue_fp8_rows(p + "attn.wo_ud.weight", oq, osc, H, HEADS * HEAD_DIM);
    layer.wo = Dense{rm(oq, DType::FP8_E4M3FN, {H, HEADS * HEAD_DIM}), rm(osc, DType::FP32, {H})};
  } else {
    void* o = dense_fp8 ? (wo_tmp = DeviceBuffer(std::size_t(H) * 3072 * 2)).p : ld.alloc(std::size_t(H) * 3072 * 2);
    queue_bf16_rows(p + "attn.wo_ud.weight", o, H, HEADS * HEAD_DIM);
    wo_bf16 = rm(o, kActDType, {H, HEADS * HEAD_DIM});
  }

  layer.q_scale = ld.f32(p + "attn.q_l2_clip.scale", {HEADS, HEAD_DIM});
  layer.r_scale = ld.f32(p + "attn.rel_l2_clip.scale", {HEADS, D_REL});
  layer.rel_proj = ld.f32(p + "attn.rel_logits_proj.proj", {D_REL, REL_EXTENT});
  if (layer.full) {
    layer.k_static = ld.f32(p + "attn.k_sconv.static_w", {CONV_K, KV_DIM});
    layer.v_static = ld.f32(p + "attn.v_sconv.static_w", {CONV_K, KV_DIM});
  } else {
    layer.k_conv = ld.f32(p + "attn.k_sconv.weight", {KV_DIM, CONV_K});  // [C, 1, K] squeezed
    layer.v_conv = ld.f32(p + "attn.v_sconv.weight", {KV_DIM, CONV_K});
  }
  layer.attn_conv = ld.f32(p + "attn_sconv.weight", {H, CONV_K});
  layer.mlp_conv = ld.f32(p + "mlp_sconv.weight", {H, CONV_K});
  layer.router = ld.f32(p + "mlp.gate.weight", {EXPERTS, H});
  {
    const std::vector<float> rb = host_f32(m, p + "mlp.gate.bias");
    if (router_bias_out) *router_bias_out = rb;
    layer.router_bias = ld.upload_f32(rb, {EXPERTS});
  }
  layer.router_bf16 = ld.upload_bf16(host_f32(m, p + "mlp.gate.weight"), {EXPERTS, H});  // exact
  if (l > 0) {
    const std::string qn = "model.llm.attn_res.attn_queries." + std::to_string(l - 1);
    const std::int32_t nq = std::int32_t(m.at(qn)->numel() / H);
    layer.attn_query = ld.f32(qn, {nq, H});
  }
  layer.mlp_query = ld.f32("model.llm.attn_res.mlp_queries." + std::to_string(l), {1, H});

  // stacked norm weights (fast._stack_layer_params)
  {
    std::vector<float> n3(std::size_t(3) * H);
    if (layer.full) {
      const char* names[3] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
      for (int g = 0; g < 3; ++g) {
        const std::vector<float> v = host_f32(m, p + names[g]);
        std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
      }
    } else {
      const std::vector<float> v = host_f32(m, p + "attn_norm.weight");
      for (int g = 0; g < 3; ++g) std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
    }
    layer.attn_norm3 = ld.upload_f32(n3, {3, H});
    layer.attn_norm_mix = layer.full ? layer.attn_norm3 : rm(layer.attn_norm3.data, DType::FP32, {1, H});
    layer.mlp_norm1 = rm(layer.mlp_norm.data, DType::FP32, {1, H});
  }

  if (layer.loop) {
    // checkpoint controllers: attn (q), k, v, mlp; stacks [q, k, v] and [mlp]
    const char* ckpt[4] = {"attn", "k", "v", "mlp"};
    std::vector<float> down[4], forget[4], gain[4];
    for (int g = 0; g < 4; ++g) {
      const std::string c = p + "controllers." + ckpt[g] + ".";
      down[g] = host_f32(m, c + "down_weight");
      forget[g] = host_f32(m, c + "forget_weight");
      gain[g] = host_f32(m, c + "gain_weight");
    }
    auto stack = [](std::vector<float>* parts, int n) {
      std::vector<float> out;
      for (int i = 0; i < n; ++i) out.insert(out.end(), parts[i].begin(), parts[i].end());
      return out;
    };
    const int D2 = 2 * H;
    layer.qkv_gate_down = ld.upload_bf16(stack(down, 3), {3, GATE_RANK, D2});  // exact (BF16 checkpoint)
    layer.qkv_gate_forget = ld.upload_f32(stack(forget, 3), {3, H, GATE_RANK});
    layer.qkv_gate_gain = ld.upload_f32(stack(gain, 3), {3, H, GATE_RANK});
    layer.qkv_gate_forget16 = ld.upload_bf16(stack(forget, 3), {3, H, GATE_RANK});
    layer.qkv_gate_gain16 = ld.upload_bf16(stack(gain, 3), {3, H, GATE_RANK});
    layer.mlp1_gate_down = ld.upload_bf16(down[3], {1, GATE_RANK, D2});
    layer.mlp1_gate_forget = ld.upload_f32(forget[3], {1, H, GATE_RANK});
    layer.mlp1_gate_gain = ld.upload_f32(gain[3], {1, H, GATE_RANK});
    layer.mlp1_gate_forget16 = ld.upload_bf16(forget[3], {1, H, GATE_RANK});
    layer.mlp1_gate_gain16 = ld.upload_bf16(gain[3], {1, H, GATE_RANK});
    auto at = [](const Tensor& stack, int g, DType dt, std::int32_t a, std::int32_t b) {
      const std::size_t es = dt == DType::FP32 ? 4 : 2;
      return rm(static_cast<char*>(stack.data) + std::size_t(g) * a * b * es, dt, {a, b});
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

  // routed experts in checkpoint encoding (skipped here when ExpertCache manages tiered offloading)
  const std::string e = p + "mlp.experts.";
  Experts& ex = layer.experts;
  ex.fmt = ef.w13;
  ex.w13_fmt = ef.w13;
  ex.w2_fmt = ef.w2;
  auto load_expert_proj = [&](const char* proj, ExpertFormat pfmt, Tensor& wt, Tensor& sc, Tensor& fac, Tensor& rin,
                              Tensor& rout) {
    if (is_exl3(pfmt)) {
      if (m.at(e + proj + "_weight")->dtype != "U8" || m.at(e + proj + "_rin")->dtype != "F16" ||
          m.at(e + proj + "_rout")->dtype != "F16")
        throw std::runtime_error("weights: layer " + std::to_string(l) + " EXL3 " + proj + " dtype mismatch");
      if (load_experts) {
        wt = ld.raw(e + proj + "_weight");
        rin = ld.raw(e + proj + "_rin");
        rout = ld.raw(e + proj + "_rout");
      }
      return;
    }
    const bool mxfp8 = pfmt == ExpertFormat::MXFP8;
    const std::string want_codes = mxfp8 ? "F8_E4M3" : "U8", want_scales = mxfp8 ? "U8" : "F8_E4M3";
    if (m.at(e + proj + "_weight")->dtype != want_codes || m.at(e + proj + "_weight_scale")->dtype != want_scales)
      throw std::runtime_error("weights: layer " + std::to_string(l) + " experts do not match their format");
    if (load_experts) {
      wt = ld.raw(e + proj + "_weight");
      sc = ld.raw(e + proj + "_weight_scale");
    }
    if (!mxfp8) {
      const std::vector<float> amax = host_f32(m, e + proj + "_weight_amax");
      const std::vector<float> den = host_f32(m, e + proj + "_weight_outer_scale_denominator");
      std::vector<float> f(EXPERTS);
      for (int i = 0; i < EXPERTS; ++i) f[i] = amax[i] / den[den.size() == 1 ? 0 : i];  // FP32 division
      fac = ld.upload_f32(f, {EXPERTS});
    }
  };
  load_expert_proj("w13", ef.w13, ex.w13, ex.w13_scale, ex.w13_factor, ex.w13_rin, ex.w13_rout);
  load_expert_proj("w2", ef.w2, ex.w2, ex.w2_scale, ex.w2_factor, ex.w2_rin, ex.w2_rout);

  if (!proj_fp8_disk || !wo_fp8_disk) {
    ld.flush();
    if (!proj_fp8_disk) {
      bf16_to_act_inplace(wproj_bf16.data, std::size_t(R) * H, ld.stream());
      layer.wproj = dense_fp8 ? to_fp8(ld, wproj_bf16, R, H) : Dense{wproj_bf16, {}};
    }
    if (!wo_fp8_disk) {
      bf16_to_act_inplace(wo_bf16.data, std::size_t(H) * HEADS * HEAD_DIM, ld.stream());
      layer.wo = dense_fp8 ? to_fp8(ld, wo_bf16, H, HEADS * HEAD_DIM) : Dense{wo_bf16, {}};
    }
    if (dense_fp8) CUDA_CHECK(cudaStreamSynchronize(ld.stream()));  // before the BF16 temporaries are freed
  }
  Dense* views[NUM_SEGS] = {&layer.wq, &layer.wgate, &layer.wr, &layer.wsink,
                            &layer.wk, &layer.wv,    &layer.wk_dyn, &layer.wv_dyn};
  for (int s = 0; s < num_segs(layer.full); ++s) *views[s] = row_view(layer.wproj, SEG_BEGIN[s], SEG_BEGIN[s + 1]);
  return layer;
}

}  // namespace

Weights load_weights(const fs::path& model_dir, bool dense_fp8, cudaStream_t stream,
                     const OffloadConfig& offload) {
  const SafetensorsModel m(model_dir);
  DeviceUploader up;
  Weights w;
  w.dense_fp8 = dense_fp8;
  const auto efs = expert_formats(model_dir);
  for (int l = 0; l < LAYERS; ++l) {
    w.expert_formats[l] = efs[l].w13;
    w.w2_expert_formats[l] = efs[l].w2;
  }
  w.storage.reserve(4096);
  Loader ld(m, up, stream, w.storage);
  const int H = HIDDEN;
  const bool tiered = offload.vram_mb > 0;
  std::array<std::vector<float>, LAYERS> router_biases;
  for (int l = 0; l < LAYERS; ++l) {
    w.layers.push_back(load_layer_into(ld, l, efs[l], dense_fp8, !tiered, &router_biases[l]));
    w.dense_fp8 = w.dense_fp8 || w.layers.back().wproj.fp8() || w.layers.back().wo.fp8();
  }

  // message_queries [12, 4, H] = cat([queries[:, :1], kv_queries, queries[:, 1:]], dim=1) with queries and
  // kv_queries [24, H] unflattened to [12, 2, H]
  {
    const std::vector<float> q = host_f32(m, "model.llm.middle_cycle_cross_pass_queries");
    const std::vector<float> kv = host_f32(m, "model.llm.middle_cycle_cross_pass_kv_queries");
    const int n = int(q.size() / H / 2);
    std::vector<float> mq(std::size_t(n) * 4 * H);
    for (int i = 0; i < n; ++i) {
      const float* rows[4] = {&q[std::size_t(2 * i) * H], &kv[std::size_t(2 * i) * H], &kv[std::size_t(2 * i + 1) * H],
                              &q[std::size_t(2 * i + 1) * H]};
      for (int j = 0; j < 4; ++j) std::copy(rows[j], rows[j] + H, mq.begin() + (std::ptrdiff_t(i) * 4 + j) * H);
    }
    w.message_queries = ld.upload_f32(mq, {n, 4, H});
  }
  w.embed = ld.raw("model.llm.embed.weight");
  const bool embed_bf16 = w.embed.dtype == DType::BF16;
  const bool embed_q4 = w.embed.dtype == DType::U8 && w.embed.ne[0] == EMBED_Q4_ROW_BYTES && w.embed.ne[1] == VOCAB;
  if (!embed_bf16 && !embed_q4)
    throw std::runtime_error("weights: embed must be BF16 [VOCAB, H] or Q4_0 U8 [VOCAB, 1296]");
  w.embed_norm = ld.f32("model.llm.embed_norm.weight", {H});
  w.final_query = ld.f32("model.llm.attn_res.output_query", {1, H});
  w.final_norm = ld.f32("model.llm.norm.weight", {H});
  DeviceBuffer unembed_tmp;
  {
    const std::string name = "model.llm.unembed.weight";
    const SafetensorsRef uref = m.at(name);
    if (uref->dtype == "F8_E4M3") {
      const SafetensorsRef sref = m.at(name + "_scale");
      if (uref->shape.size() != 2 || uref->shape[0] < UNPADDED_VOCAB || uref->shape[1] != H)
        throw std::runtime_error("weights: unembed F8_E4M3 shape mismatch");
      if (sref->dtype != "F32" || sref->numel() < UNPADDED_VOCAB)
        throw std::runtime_error("weights: unembed_scale must be F32 [>=UNPADDED_VOCAB]");
      void* q = ld.alloc(std::size_t(UNPADDED_VOCAB) * H);
      float* sc = static_cast<float*>(ld.alloc(std::size_t(UNPADDED_VOCAB) * 4));
      ld.queue(name, q, UNPADDED_VOCAB);
      ld.queue(name + "_scale", sc, UNPADDED_VOCAB);
      ld.flush();
      if (embed_bf16) {
        bf16_to_act_inplace(w.embed.data, std::size_t(w.embed.numel()), stream);
        w.embed.dtype = kActDType;
      }
      w.unembed = Dense{rm(q, DType::FP8_E4M3FN, {UNPADDED_VOCAB, H}), rm(sc, DType::FP32, {UNPADDED_VOCAB})};
      w.dense_fp8 = true;
    } else if (uref->dtype == "BF16") {
      const std::size_t bytes = std::size_t(UNPADDED_VOCAB) * H * 2;
      void* d = dense_fp8 ? (unembed_tmp = DeviceBuffer(bytes)).p : ld.alloc(bytes);
      ld.queue(name, d, UNPADDED_VOCAB);  // unembed[:UNPADDED_VOCAB]
      ld.flush();
      if (embed_bf16) {
        bf16_to_act_inplace(w.embed.data, std::size_t(w.embed.numel()), stream);
        w.embed.dtype = kActDType;
      }
      bf16_to_act_inplace(d, std::size_t(UNPADDED_VOCAB) * H, stream);
      const Tensor bf = rm(d, kActDType, {UNPADDED_VOCAB, H});
      if (dense_fp8) {
        w.unembed = to_fp8(ld, bf, UNPADDED_VOCAB, H);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      } else {
        w.unembed = Dense{bf, {}};
      }
    } else {
      throw std::runtime_error("weights: unembed must be BF16 or F8_E4M3");
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (tiered) {
    std::size_t non_expert_bytes = 0;
    for (const auto& b : w.storage) non_expert_bytes += b.bytes;
    // Estimate post-load fixed VRAM (KV cache + prefiller scratch + checkpoint + runtime/cuBLAS headroom)
    // so total model GPU memory stays strictly within offload.vram_mb MiB.
    const std::size_t kv_ring_bytes =
        std::size_t(NUM_SLIDING_VISITS) * WINDOW * KV_DIM * 4 + std::size_t(NUM_SLIDING_VISITS) * WINDOW * 8;
    const std::size_t kv_full_bytes =
        offload.kv_vram_mb > 0
            ? (std::size_t(offload.kv_vram_mb) << 20)
            : (std::size_t(NUM_FULL_VISITS) * std::size_t(offload.max_ctx) *
               (offload.kv_q4 ? (2 * std::size_t(KV_Q4_ROW_BYTES)) : (std::size_t(KV_DIM) * (offload.kv_fp8 ? 2 : 4))));
    const std::size_t checkpoint_bytes = kv_ring_bytes + (std::size_t(16) << 20);
    // Prefiller aliases Attention vs MLP scratch (including EXL3 scratch) in one phase arena (~269 KB/tok + 42 MiB).
    const std::size_t prefill_bytes =
        std::size_t(offload.prefill_chunk) * std::size_t(280) * 1024 + (std::size_t(48) << 20);
    // DSpark drafter weights + shortlist + FP8 global KV ring (min(max_ctx, 65536) * 2 KiB) +
    // verify buffers (widths 2/4/8) + prefill taps when --spec is active (measured ~1.09 GB at 64K/1024).
    const std::size_t spec_bytes =
        offload.spec ? ((std::size_t(928) << 20) +
                        std::size_t(std::min(offload.max_ctx, 65536)) * std::size_t(2048) +
                        std::size_t(offload.prefill_chunk) * std::size_t(18) * 1024)
                     : 0;
    const std::size_t runtime_headroom = std::size_t(384) << 20;
    const std::size_t fixed_bytes =
        non_expert_bytes + kv_ring_bytes + kv_full_bytes + checkpoint_bytes + prefill_bytes + spec_bytes + runtime_headroom;
    const std::size_t total_budget = std::size_t(offload.vram_mb) << 20;
    const std::size_t expert_vram_budget =
        total_budget > fixed_bytes ? (total_budget - fixed_bytes) : (std::size_t(256) << 20);
    const std::size_t warm_ram_budget =
        offload.ram_mb > 512 ? (std::size_t(offload.ram_mb - 256) << 20) : std::size_t(0);
    w.expert_cache = ExpertCache::create(model_dir, w.expert_formats, w.w2_expert_formats, router_biases,
                                         expert_vram_budget, warm_ram_budget, offload, stream);
  }
  return w;
}

Layer load_layer_sharded(Loader& ld, int l, LayerExpertFormats ef, bool dense_fp8, int tp_rank, int tp_size) {
  const SafetensorsModel& m = ld.model();
  const std::string p = "model.llm.layers." + std::to_string(l) + ".";
  Layer layer;
  layer.idx = l;
  layer.full = is_full_layer(l);
  layer.loop = is_loop_layer(l);
  const int H = HIDDEN;
  if (tp_size != TP_SIZE) throw std::invalid_argument("load_layer_sharded: tp_size != compile-time TP_SIZE");

  if (layer.full) {
    layer.q_norm = ld.f32(p + "q_input_norm.weight", {H});
    layer.k_norm = ld.f32(p + "k_input_norm.weight", {H});
    layer.v_norm = ld.f32(p + "v_input_norm.weight", {H});
  } else {
    layer.attn_norm = ld.f32(p + "attn_norm.weight", {H});
  }
  layer.mlp_norm = ld.f32(p + "mlp_norm.weight", {H});

  static const char* seg_names[NUM_SEGS] = {"attn.wq_du.weight",        "attn.gate_attn.weight",
                                            "attn.wr_du.weight",        "attn.attn_sink_proj.weight",
                                            "attn.wk_dv.weight",        "attn.wv_dv.weight",
                                            "attn.k_sconv.weight_proj", "attn.v_sconv.weight_proj"};
  const int seg_real[NUM_SEGS] = {HEADS * HEAD_DIM, HEADS * HEAD_DIM, HEADS * D_REL, HEADS,
                                  KV_DIM, KV_DIM, KV_HEADS * CONV_K, KV_HEADS * CONV_K};
  const int R = proj_rows(layer.full);
  const bool proj_fp8_disk = m.at(p + seg_names[0])->dtype == "F8_E4M3";
  const bool wo_fp8_disk = m.at(p + "attn.wo_ud.weight")->dtype == "F8_E4M3";
  DeviceBuffer proj_tmp, wo_tmp;
  Tensor wproj_bf16, wo_bf16;
  if (proj_fp8_disk) {
    void* q = ld.alloc(std::size_t(R) * H);
    float* sc = static_cast<float*>(ld.alloc(std::size_t(R) * 4));
    CUDA_CHECK(cudaMemsetAsync(q, 0, std::size_t(R) * H, ld.stream()));
    CUDA_CHECK(cudaMemsetAsync(sc, 0, std::size_t(R) * 4, ld.stream()));
    for (int s = 0; s < num_segs(layer.full); ++s) {
      const std::int64_t r0 = std::int64_t(tp_rank) * seg_real[s];
      ld.queue_rows(p + seg_names[s], r0, seg_real[s], static_cast<char*>(q) + std::size_t(SEG_BEGIN[s]) * H);
      ld.queue_rows(p + std::string(seg_names[s]) + "_scale", r0, seg_real[s], sc + SEG_BEGIN[s]);
    }
    layer.wproj = Dense{rm(q, DType::FP8_E4M3FN, {R, H}), rm(sc, DType::FP32, {R})};
  } else {
    void* d = dense_fp8 ? (proj_tmp = DeviceBuffer(std::size_t(R) * H * 2)).p : ld.alloc(std::size_t(R) * H * 2);
    CUDA_CHECK(cudaMemsetAsync(d, 0, std::size_t(R) * H * 2, ld.stream()));
    for (int s = 0; s < num_segs(layer.full); ++s) {
      const std::int64_t r0 = std::int64_t(tp_rank) * seg_real[s];
      ld.queue_rows(p + seg_names[s], r0, seg_real[s], static_cast<char*>(d) + std::size_t(SEG_BEGIN[s]) * H * 2);
    }
    wproj_bf16 = rm(d, kActDType, {R, H});
  }
  // Column-sharded wo: [H, TOTAL_HEADS * HEAD_DIM] -> [H, HEADS * HEAD_DIM]
  {
    const int full_cols = TOTAL_HEADS * HEAD_DIM;
    const int local_cols = HEADS * HEAD_DIM;
    const int col0 = tp_rank * local_cols;
    const std::string wn = p + "attn.wo_ud.weight";
    const std::vector<std::byte> raw = m.read(wn);
    const std::size_t es = wo_fp8_disk ? 1 : 2;
    std::vector<std::byte> sliced(std::size_t(H) * local_cols * es);
    for (int r = 0; r < H; ++r) {
      std::memcpy(sliced.data() + std::size_t(r) * local_cols * es,
                  raw.data() + (std::size_t(r) * full_cols + col0) * es,
                  std::size_t(local_cols) * es);
    }
    if (wo_fp8_disk) {
      void* oq = ld.alloc(sliced.size());
      CUDA_CHECK(cudaMemcpyAsync(oq, sliced.data(), sliced.size(), cudaMemcpyHostToDevice, ld.stream()));
      Tensor osc = ld.f32(wn + "_scale", {H});
      layer.wo = Dense{rm(oq, DType::FP8_E4M3FN, {H, local_cols}), osc};
    } else {
      void* o = dense_fp8 ? (wo_tmp = DeviceBuffer(sliced.size())).p : ld.alloc(sliced.size());
      CUDA_CHECK(cudaMemcpyAsync(o, sliced.data(), sliced.size(), cudaMemcpyHostToDevice, ld.stream()));
      wo_bf16 = rm(o, kActDType, {H, local_cols});
    }
  }
  // Head-sharded scales & KV convolutions
  {
    const std::vector<float> qs = host_f32(m, p + "attn.q_l2_clip.scale");
    std::vector<float> qsl(qs.begin() + std::ptrdiff_t(tp_rank) * HEADS * HEAD_DIM,
                           qs.begin() + std::ptrdiff_t(tp_rank + 1) * HEADS * HEAD_DIM);
    layer.q_scale = ld.upload_f32(qsl, {HEADS, HEAD_DIM});
    const std::vector<float> rs = host_f32(m, p + "attn.rel_l2_clip.scale");
    std::vector<float> rsl(rs.begin() + std::ptrdiff_t(tp_rank) * HEADS * D_REL,
                           rs.begin() + std::ptrdiff_t(tp_rank + 1) * HEADS * D_REL);
    layer.r_scale = ld.upload_f32(rsl, {HEADS, D_REL});
  }
  layer.rel_proj = ld.f32(p + "attn.rel_logits_proj.proj", {D_REL, REL_EXTENT});
  if (layer.full) {
    const int full_kv = TOTAL_KV_HEADS * HEAD_DIM;
    auto slice_static = [&](const std::string& name) {
      const std::vector<float> full = host_f32(m, name);
      std::vector<float> out(std::size_t(CONV_K) * KV_DIM);
      for (int lag = 0; lag < CONV_K; ++lag) {
        std::copy(full.begin() + std::ptrdiff_t(lag) * full_kv + std::ptrdiff_t(tp_rank) * KV_DIM,
                  full.begin() + std::ptrdiff_t(lag) * full_kv + std::ptrdiff_t(tp_rank + 1) * KV_DIM,
                  out.begin() + std::ptrdiff_t(lag) * KV_DIM);
      }
      return ld.upload_f32(out, {CONV_K, KV_DIM});
    };
    layer.k_static = slice_static(p + "attn.k_sconv.static_w");
    layer.v_static = slice_static(p + "attn.v_sconv.static_w");
  } else {
    auto slice_conv = [&](const std::string& name) {
      const std::vector<float> full = host_f32(m, name);
      std::vector<float> out(full.begin() + std::ptrdiff_t(tp_rank) * KV_DIM * CONV_K,
                             full.begin() + std::ptrdiff_t(tp_rank + 1) * KV_DIM * CONV_K);
      return ld.upload_f32(out, {KV_DIM, CONV_K});
    };
    layer.k_conv = slice_conv(p + "attn.k_sconv.weight");
    layer.v_conv = slice_conv(p + "attn.v_sconv.weight");
  }
  layer.attn_conv = ld.f32(p + "attn_sconv.weight", {H, CONV_K});
  layer.mlp_conv = ld.f32(p + "mlp_sconv.weight", {H, CONV_K});
  layer.router = ld.f32(p + "mlp.gate.weight", {EXPERTS, H});
  layer.router_bias = ld.f32(p + "mlp.gate.bias", {EXPERTS});
  layer.router_bf16 = ld.upload_bf16(host_f32(m, p + "mlp.gate.weight"), {EXPERTS, H});
  if (l > 0) {
    const std::string qn = "model.llm.attn_res.attn_queries." + std::to_string(l - 1);
    const std::int32_t nq = std::int32_t(m.at(qn)->numel() / H);
    layer.attn_query = ld.f32(qn, {nq, H});
  }
  layer.mlp_query = ld.f32("model.llm.attn_res.mlp_queries." + std::to_string(l), {1, H});
  {
    std::vector<float> n3(std::size_t(3) * H);
    if (layer.full) {
      const char* names[3] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
      for (int g = 0; g < 3; ++g) {
        const std::vector<float> v = host_f32(m, p + names[g]);
        std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
      }
    } else {
      const std::vector<float> v = host_f32(m, p + "attn_norm.weight");
      for (int g = 0; g < 3; ++g) std::copy(v.begin(), v.end(), n3.begin() + std::ptrdiff_t(g) * H);
    }
    layer.attn_norm3 = ld.upload_f32(n3, {3, H});
    layer.attn_norm_mix = layer.full ? layer.attn_norm3 : rm(layer.attn_norm3.data, DType::FP32, {1, H});
    layer.mlp_norm1 = rm(layer.mlp_norm.data, DType::FP32, {1, H});
  }
  if (layer.loop) {
    const char* ckpt[4] = {"attn", "k", "v", "mlp"};
    std::vector<float> down[4], forget[4], gain[4];
    for (int g = 0; g < 4; ++g) {
      const std::string c = p + "controllers." + ckpt[g] + ".";
      down[g] = host_f32(m, c + "down_weight");
      forget[g] = host_f32(m, c + "forget_weight");
      gain[g] = host_f32(m, c + "gain_weight");
    }
    auto stack = [](std::vector<float>* parts, int n) {
      std::vector<float> out;
      for (int i = 0; i < n; ++i) out.insert(out.end(), parts[i].begin(), parts[i].end());
      return out;
    };
    const int D2 = 2 * H;
    layer.qkv_gate_down = ld.upload_bf16(stack(down, 3), {3, GATE_RANK, D2});
    layer.qkv_gate_forget = ld.upload_f32(stack(forget, 3), {3, H, GATE_RANK});
    layer.qkv_gate_gain = ld.upload_f32(stack(gain, 3), {3, H, GATE_RANK});
    layer.qkv_gate_forget16 = ld.upload_bf16(stack(forget, 3), {3, H, GATE_RANK});
    layer.qkv_gate_gain16 = ld.upload_bf16(stack(gain, 3), {3, H, GATE_RANK});
    layer.mlp1_gate_down = ld.upload_bf16(down[3], {1, GATE_RANK, D2});
    layer.mlp1_gate_forget = ld.upload_f32(forget[3], {1, H, GATE_RANK});
    layer.mlp1_gate_gain = ld.upload_f32(gain[3], {1, H, GATE_RANK});
    layer.mlp1_gate_forget16 = ld.upload_bf16(forget[3], {1, H, GATE_RANK});
    layer.mlp1_gate_gain16 = ld.upload_bf16(gain[3], {1, H, GATE_RANK});
    auto at = [](const Tensor& stack, int g, DType dt, std::int32_t a, std::int32_t b) {
      const std::size_t es = dt == DType::FP32 ? 4 : 2;
      return rm(static_cast<char*>(stack.data) + std::size_t(g) * a * b * es, dt, {a, b});
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

  // EP-sharded routed experts: load only [e_begin, e_begin + e_per) into device memory
  const std::string e = p + "mlp.experts.";
  Experts& ex = layer.experts;
  ex.fmt = ef.w13;
  ex.w13_fmt = ef.w13;
  ex.w2_fmt = ef.w2;
  ex.ep_rank = tp_rank;
  ex.ep_size = tp_size;
  const int e_per = EXPERTS / tp_size;
  const int e_begin = tp_rank * e_per;
  auto load_ep_proj = [&](const char* proj, ExpertFormat pfmt, Tensor& wt, Tensor& sc, Tensor& fac,
                          Tensor& w_ptrs_t, Tensor& s_ptrs_t) {
    if (is_exl3(pfmt)) throw std::runtime_error("EP sharding supports NVFP4/MXFP8/ATLAS2/ATLAS3 (not EXL3)");
    const bool mxfp8 = pfmt == ExpertFormat::MXFP8;
    const SafetensorsRef wref = m.at(e + proj + "_weight");
    const SafetensorsRef sref = m.at(e + proj + "_weight_scale");
    const std::uint64_t w_per_exp = wref->bytes / std::uint64_t(EXPERTS);
    const std::uint64_t s_per_exp = sref->bytes / std::uint64_t(EXPERTS);
    void* wd = ld.alloc(w_per_exp * std::uint64_t(e_per));
    void* sd = ld.alloc(s_per_exp * std::uint64_t(e_per));
    ld.queue_rows(e + proj + "_weight", e_begin, e_per, wd);
    ld.queue_rows(e + proj + "_weight_scale", e_begin, e_per, sd);
    wt = rm(wd, wref->ingine_dtype(), {e_per, std::int32_t(w_per_exp)});
    sc = rm(sd, sref->ingine_dtype(), {e_per, std::int32_t(s_per_exp)});
    std::vector<const std::uint8_t*> wp(EXPERTS, nullptr), sp(EXPERTS, nullptr);
    for (int i = 0; i < e_per; ++i) {
      wp[e_begin + i] = static_cast<const std::uint8_t*>(wd) + std::size_t(i) * w_per_exp;
      sp[e_begin + i] = static_cast<const std::uint8_t*>(sd) + std::size_t(i) * s_per_exp;
    }
    void* wpd = ld.alloc(EXPERTS * sizeof(void*));
    void* spd = ld.alloc(EXPERTS * sizeof(void*));
    CUDA_CHECK(cudaMemcpyAsync(wpd, wp.data(), EXPERTS * sizeof(void*), cudaMemcpyHostToDevice, ld.stream()));
    CUDA_CHECK(cudaMemcpyAsync(spd, sp.data(), EXPERTS * sizeof(void*), cudaMemcpyHostToDevice, ld.stream()));
    w_ptrs_t = rm(wpd, DType::I64, {EXPERTS});
    s_ptrs_t = rm(spd, DType::I64, {EXPERTS});
    if (!mxfp8) {
      const std::vector<float> amax = host_f32(m, e + proj + "_weight_amax");
      const std::vector<float> den = host_f32(m, e + proj + "_weight_outer_scale_denominator");
      std::vector<float> f(EXPERTS);
      for (int i = 0; i < EXPERTS; ++i) f[i] = amax[i] / den[den.size() == 1 ? 0 : i];
      fac = ld.upload_f32(f, {EXPERTS});
    }
  };
  load_ep_proj("w13", ef.w13, ex.w13, ex.w13_scale, ex.w13_factor, ex.w13_ptrs, ex.w13_scale_ptrs);
  load_ep_proj("w2", ef.w2, ex.w2, ex.w2_scale, ex.w2_factor, ex.w2_ptrs, ex.w2_scale_ptrs);
  {
    std::vector<std::uint8_t> mask(EXPERTS, 0);
    for (int i = 0; i < e_per; ++i) mask[e_begin + i] = 1;
    void* md = ld.alloc(EXPERTS);
    CUDA_CHECK(cudaMemcpyAsync(md, mask.data(), EXPERTS, cudaMemcpyHostToDevice, ld.stream()));
    ex.active_mask = rm(md, DType::U8, {EXPERTS});
  }

  if (!proj_fp8_disk || !wo_fp8_disk) {
    ld.flush();
    if (!proj_fp8_disk) {
      bf16_to_act_inplace(wproj_bf16.data, std::size_t(R) * H, ld.stream());
      layer.wproj = dense_fp8 ? to_fp8(ld, wproj_bf16, R, H) : Dense{wproj_bf16, {}};
    }
    if (!wo_fp8_disk) {
      bf16_to_act_inplace(wo_bf16.data, std::size_t(H) * HEADS * HEAD_DIM, ld.stream());
      layer.wo = dense_fp8 ? to_fp8(ld, wo_bf16, H, HEADS * HEAD_DIM) : Dense{wo_bf16, {}};
    }
    if (dense_fp8) CUDA_CHECK(cudaStreamSynchronize(ld.stream()));
  }
  Dense* views[NUM_SEGS] = {&layer.wq, &layer.wgate, &layer.wr, &layer.wsink,
                            &layer.wk, &layer.wv,    &layer.wk_dyn, &layer.wv_dyn};
  for (int s = 0; s < num_segs(layer.full); ++s) *views[s] = row_view(layer.wproj, SEG_BEGIN[s], SEG_BEGIN[s + 1]);
  return layer;
}

Weights load_weights_sharded(const fs::path& model_dir, bool dense_fp8, cudaStream_t stream,
                             int tp_rank, int tp_size, bool keep_full_unembed) {
  if (tp_size == 1) return load_weights(model_dir, dense_fp8, stream);
  if (tp_size != TP_SIZE || tp_rank < 0 || tp_rank >= tp_size)
    throw std::invalid_argument("load_weights_sharded: invalid tp_rank/tp_size");
  const SafetensorsModel m(model_dir);
  DeviceUploader up;
  Weights w;
  w.dense_fp8 = dense_fp8;
  w.tp_rank = tp_rank;
  w.tp_size = tp_size;
  const auto efs = expert_formats(model_dir);
  for (int l = 0; l < LAYERS; ++l) {
    w.expert_formats[l] = efs[l].w13;
    w.w2_expert_formats[l] = efs[l].w2;
  }
  w.storage.reserve(4096);
  Loader ld(m, up, stream, w.storage);
  const int H = HIDDEN;
  for (int l = 0; l < LAYERS; ++l) {
    w.layers.push_back(load_layer_sharded(ld, l, efs[l], dense_fp8, tp_rank, tp_size));
    w.dense_fp8 = w.dense_fp8 || w.layers.back().wproj.fp8() || w.layers.back().wo.fp8();
  }
  {
    const std::vector<float> q = host_f32(m, "model.llm.middle_cycle_cross_pass_queries");
    const std::vector<float> kv = host_f32(m, "model.llm.middle_cycle_cross_pass_kv_queries");
    const int n = int(q.size() / H / 2);
    std::vector<float> mq(std::size_t(n) * 4 * H);
    for (int i = 0; i < n; ++i) {
      const float* rows[4] = {&q[std::size_t(2 * i) * H], &kv[std::size_t(2 * i) * H], &kv[std::size_t(2 * i + 1) * H],
                              &q[std::size_t(2 * i + 1) * H]};
      for (int j = 0; j < 4; ++j) std::copy(rows[j], rows[j] + H, mq.begin() + (std::ptrdiff_t(i) * 4 + j) * H);
    }
    w.message_queries = ld.upload_f32(mq, {n, 4, H});
  }
  w.embed = ld.raw("model.llm.embed.weight");
  const bool embed_bf16 = w.embed.dtype == DType::BF16;
  w.embed_norm = ld.f32("model.llm.embed_norm.weight", {H});
  w.final_query = ld.f32("model.llm.attn_res.output_query", {1, H});
  w.final_norm = ld.f32("model.llm.norm.weight", {H});
  // Row-sharded unembed: [LOCAL_VOCAB_CHUNK, H] per rank
  const int v0 = tp_rank * LOCAL_VOCAB_CHUNK;
  const int v1 = std::min(UNPADDED_VOCAB, v0 + LOCAL_VOCAB_CHUNK);
  const int v_real = std::max(0, v1 - v0);
  w.vocab_begin = v0;
  w.vocab_rows = v_real;
  DeviceBuffer unembed_tmp, full_unembed_tmp;
  {
    const std::string name = "model.llm.unembed.weight";
    const SafetensorsRef uref = m.at(name);
    if (uref->dtype == "F8_E4M3") {
      void* q = ld.alloc(std::size_t(LOCAL_VOCAB_CHUNK) * H);
      float* sc = static_cast<float*>(ld.alloc(std::size_t(LOCAL_VOCAB_CHUNK) * 4));
      CUDA_CHECK(cudaMemsetAsync(q, 0, std::size_t(LOCAL_VOCAB_CHUNK) * H, stream));
      CUDA_CHECK(cudaMemsetAsync(sc, 0, std::size_t(LOCAL_VOCAB_CHUNK) * 4, stream));
      if (v_real > 0) {
        ld.queue_rows(name, v0, v_real, q);
        ld.queue_rows(name + "_scale", v0, v_real, sc);
      }
      void* fq = nullptr; float* fsc = nullptr;
      if (keep_full_unembed) {
        fq = ld.alloc(std::size_t(UNPADDED_VOCAB) * H);
        fsc = static_cast<float*>(ld.alloc(std::size_t(UNPADDED_VOCAB) * 4));
        ld.queue(name, fq, UNPADDED_VOCAB);
        ld.queue(name + "_scale", fsc, UNPADDED_VOCAB);
      }
      ld.flush();
      if (embed_bf16) { bf16_to_act_inplace(w.embed.data, std::size_t(w.embed.numel()), stream); w.embed.dtype = kActDType; }
      w.unembed = Dense{rm(q, DType::FP8_E4M3FN, {LOCAL_VOCAB_CHUNK, H}), rm(sc, DType::FP32, {LOCAL_VOCAB_CHUNK})};
      if (keep_full_unembed) w.full_unembed = Dense{rm(fq, DType::FP8_E4M3FN, {UNPADDED_VOCAB, H}), rm(fsc, DType::FP32, {UNPADDED_VOCAB})};
      w.dense_fp8 = true;
    } else if (uref->dtype == "BF16") {
      const std::size_t bytes = std::size_t(LOCAL_VOCAB_CHUNK) * H * 2;
      void* d = dense_fp8 ? (unembed_tmp = DeviceBuffer(bytes)).p : ld.alloc(bytes);
      CUDA_CHECK(cudaMemsetAsync(d, 0, bytes, stream));
      if (v_real > 0) ld.queue_rows(name, v0, v_real, d);
      void* fd = nullptr;
      if (keep_full_unembed) {
        const std::size_t fbytes = std::size_t(UNPADDED_VOCAB) * H * 2;
        fd = dense_fp8 ? (full_unembed_tmp = DeviceBuffer(fbytes)).p : ld.alloc(fbytes);
        ld.queue(name, fd, UNPADDED_VOCAB);
      }
      ld.flush();
      if (embed_bf16) { bf16_to_act_inplace(w.embed.data, std::size_t(w.embed.numel()), stream); w.embed.dtype = kActDType; }
      bf16_to_act_inplace(d, std::size_t(LOCAL_VOCAB_CHUNK) * H, stream);
      const Tensor bf = rm(d, kActDType, {LOCAL_VOCAB_CHUNK, H});
      w.unembed = dense_fp8 ? to_fp8(ld, bf, LOCAL_VOCAB_CHUNK, H) : Dense{bf, {}};
      if (keep_full_unembed) {
        bf16_to_act_inplace(fd, std::size_t(UNPADDED_VOCAB) * H, stream);
        const Tensor fbf = rm(fd, kActDType, {UNPADDED_VOCAB, H});
        w.full_unembed = dense_fp8 ? to_fp8(ld, fbf, UNPADDED_VOCAB, H) : Dense{fbf, {}};
      }
      if (dense_fp8) CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
      throw std::runtime_error("weights: unembed must be BF16 or F8_E4M3");
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return w;
}

Layer load_layer(const fs::path& model_dir, int l, bool dense_fp8, cudaStream_t stream,
                 std::vector<DeviceBuffer>& storage) {
  if (l < 0 || l >= LAYERS) throw std::invalid_argument("load_layer: layer out of range");
  const SafetensorsModel m(model_dir);
  DeviceUploader up;
  Loader ld(m, up, stream, storage);
  Layer layer = load_layer_into(ld, l, expert_formats(model_dir)[l], dense_fp8, true);
  ld.flush();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return layer;
}

// File layouts (FP32 or BF16 tensors):
//   attn_dirs, mlp_dirs [48, H]; attn_w, mlp_w [48]                                  plain projection
//   attn_dirs, mlp_dirs [48, H]; attn_weight, mlp_weight, attn_offset, mlp_offset [48]; gate [1];
//       optionally attn_det, mlp_det [48, H] and attn_threshold, mlp_threshold [48]
// The second form removes w * r * (r . y - offset), limited to components above the offset when gate is
// non-zero and, with detectors, to writes with det . y above the threshold. Its weights carry other names so
// that engines that only know the first form reject the file.
Ablation load_ablation(const fs::path& path) {
  Ablation a;
  const int H = HIDDEN;
  const std::size_t nd = std::size_t(NUM_VISITS) * H;
  // rows 0..47: directions, rows 48..95: detectors (zero without them)
  std::vector<float> attn_dirs(2 * nd, 0.f), mlp_dirs(2 * nd, 0.f);
  std::vector<float> attn_w(std::size_t(NUM_VISITS) * 4, 0.f), mlp_w(attn_w.size(), 0.f);
  if (!path.empty()) {
    const SafetensorsModel m(path);
    auto read = [&](const std::string& name, std::size_t n) {
      std::vector<float> v = host_f32(m, name);
      if (v.size() != n) throw std::runtime_error("ablation: " + name + " has the wrong size");
      return v;
    };
    // F.normalize(dirs, dim=-1): x / max(||x||, 1e-12) in FP32
    auto normalize = [&](std::vector<float> v) {
      for (int r = 0; r < NUM_VISITS; ++r) {
        double ss = 0;
        for (int c = 0; c < H; ++c) ss += double(v[std::size_t(r) * H + c]) * v[std::size_t(r) * H + c];
        const float norm = std::max(float(std::sqrt(ss)), 1e-12f);
        for (int c = 0; c < H; ++c) v[std::size_t(r) * H + c] = v[std::size_t(r) * H + c] / norm;
      }
      return v;
    };
    const bool plain = m.contains("attn_w");
    a.detect = !plain && m.contains("attn_det");
    const float floor = !plain && read("gate", 1)[0] != 0.f ? 0.f : -std::numeric_limits<float>::max();
    auto load = [&](const std::string& kind, std::vector<float>& dirs, std::vector<float>& rows) {
      const std::vector<float> r = normalize(read(kind + "_dirs", nd));
      std::copy(r.begin(), r.end(), dirs.begin());
      const std::vector<float> w = read(kind + (plain ? "_w" : "_weight"), NUM_VISITS);
      const std::vector<float> c = plain ? std::vector<float>(NUM_VISITS, 0.f) : read(kind + "_offset", NUM_VISITS);
      std::vector<float> tau(NUM_VISITS, 0.f);
      if (a.detect) {
        const std::vector<float> d = read(kind + "_det", nd);  // as stored: the threshold is in its units
        std::copy(d.begin(), d.end(), dirs.begin() + std::ptrdiff_t(nd));
        tau = read(kind + "_threshold", NUM_VISITS);
      }
      for (int v = 0; v < NUM_VISITS; ++v)
        rows[4 * v] = w[v], rows[4 * v + 1] = c[v], rows[4 * v + 2] = floor, rows[4 * v + 3] = tau[v];
    };
    load("attn", attn_dirs, attn_w);
    load("mlp", mlp_dirs, mlp_w);
    a.active = true;
  }
  auto up = [&](const std::vector<float>& h, std::initializer_list<std::int32_t> shape) {
    a.storage.emplace_back(h.size() * 4);
    CUDA_CHECK(cudaMemcpy(a.storage.back().p, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return rm(a.storage.back().p, DType::FP32, shape);
  };
  a.attn_dirs = up(attn_dirs, {2 * NUM_VISITS, H});
  a.attn_w = up(attn_w, {NUM_VISITS, 4});
  a.mlp_dirs = up(mlp_dirs, {2 * NUM_VISITS, H});
  a.mlp_w = up(mlp_w, {NUM_VISITS, 4});
  return a;
}

}  // namespace ingine::model

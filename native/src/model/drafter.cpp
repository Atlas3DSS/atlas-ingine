#include "model/drafter.h"

#include "core/act.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "kernels/draft/shortlist.h"
#include "model/model_kernels.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ingine::model {

namespace kd = kernels::draft;
namespace kdense = kernels::dense;

namespace {

constexpr int H = kd::H;
constexpr int LOCAL_SLOTS = kd::WINDOW + kd::BLOCK;  // local ring + the drafted block's rows
constexpr int COMMIT_PIECE = 1024;                   // eager commit rows per piece
// The global ring's keys/values are E4M3 (kernels/draft/draft.h): a draft over a long context is bound by
// the bytes of that ring. The local rings are BF16.
constexpr bool GLOBAL_FP8 = true;
constexpr std::size_t GLOBAL_SLOT_BYTES = std::size_t(kd::KV_DIM) * (GLOBAL_FP8 ? 1 : 2);  // of k or of v

bool is_local(int l) { return l != 1; }  // dspark.LOCAL = (0, 2)

std::vector<float> bf16_to_float(const std::vector<std::byte>& raw) {
  std::vector<float> out(raw.size() / 2);
  for (std::size_t i = 0; i < out.size(); ++i) {
    std::uint16_t h;
    std::memcpy(&h, raw.data() + 2 * i, 2);
    const std::uint32_t u = std::uint32_t(h) << 16;
    std::memcpy(&out[i], &u, 4);
  }
  return out;
}

void capture_begin(cudaStream_t s) { CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)); }
cudaGraphExec_t capture_end(cudaStream_t s) {
  cudaGraph_t g = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(s, &g));
  cudaGraphExec_t e = nullptr;
  const cudaError_t err = cudaGraphInstantiate(&e, g, 0);
  cudaGraphDestroy(g);
  CUDA_CHECK(err);
  return e;
}

}  // namespace

void* Drafter::alloc(std::size_t bytes) {
  storage_.emplace_back(bytes);
  return storage_.back().p;
}

Drafter::Drafter(const std::filesystem::path& model_dir, const Weights& w, int max_ctx, cudaStream_t stream,
                 int shortlist_rows, const std::filesystem::path& dflash2_dir)
    : w_(w), stream_(stream), dflash2_(!dflash2_dir.empty()), ctx_(std::min(max_ctx, DRAFT_CTX)) {
  if (max_ctx < 2 * BLOCK) throw std::invalid_argument("Drafter: max_ctx must hold two blocks");
  head_ = {w.unembed.w.data, w.unembed.fp8() ? static_cast<const float*>(w.unembed.scale.data) : nullptr,
           w.unembed.rows()};
  V_ = w.unembed.rows();
  if (dflash2_) load_dflash2(dflash2_dir);
  else load(model_dir);

  auto bf16s = [&](std::size_t n) { return static_cast<ActT*>(alloc(n * 2)); };
  auto f32s = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto i32s = [&](std::size_t n) { return static_cast<std::int32_t*>(alloc(n * 4)); };

  params_ = static_cast<kd::DraftParams*>(alloc(sizeof(kd::DraftParams)));
  CUDA_CHECK(cudaMemsetAsync(params_, 0, sizeof(kd::DraftParams), stream_));
  res_ = bf16s(BLOCK * H);
  xn_ = bf16s(BLOCK * H);
  delta_ = f32s(BLOCK * H);
  act_ = bf16s(std::size_t(BLOCK) * inter_);
  gu_out_ = f32s(std::size_t(BLOCK) * 2 * inter_);
  // the shortlist head's base logits (rows past the ones in use stay -inf)
  base_ = f32s(std::size_t(GAMMA) * std::max(V_, shortlist_rows + 256 + CONTEXT_ROWS));
  picks_ = static_cast<unsigned long long*>(alloc(GAMMA * sizeof(unsigned long long)));
  draft_logits_ = f32s(std::size_t(GAMMA) * V_);
  conf_ = f32s(GAMMA);
  qstats_ = f32s(2 * GAMMA);
  tokens_ = i32s(GAMMA);
  warm_feats_ = bf16s(std::size_t(BLOCK) * 4 * H);
  ctx_c_ = bf16s(std::size_t(COMMIT_PIECE) * H);
  ctx_kv_ = bf16s(std::size_t(COMMIT_PIECE) * 2 * kd::KV_DIM);
  CUDA_CHECK(cudaMemsetAsync(warm_feats_, 0, std::size_t(BLOCK) * 4 * H * 2, stream_));
  kd::fill(draft_logits_, std::size_t(GAMMA) * V_, -INFINITY, stream_);

  // segment tables (row_begin, row_end, input) for the dense GEMV
  auto segments = [&](std::initializer_list<int> rows) {
    std::vector<int> host;
    int b = 0, input = 0;
    for (int r : rows) {
      host.insert(host.end(), {b, b + r, input++});
      b += r;
    }
    int* d = i32s(host.size());
    CUDA_CHECK(cudaMemcpyAsync(d, host.data(), host.size() * 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return d;
  };
  seg_gate_up_ = segments({2 * inter_});

  // the shortlist head: the shortlist_rows most frequent tokens and the specials, by ascending id (so equal
  // base logits break ties towards the lower id, as over the full head), then room for context tokens
  {
    short_in_.assign(V_, 0);
    int taken = 0;
    for (std::uint32_t t : kd::draft_shortlist_ranks())
      if (taken < shortlist_rows && t < std::uint32_t(V_) && !short_in_[t]) short_in_[t] = 1, ++taken;
    for (int t = 199990; t < V_; ++t) short_in_[t] = 1;  // <|unused|>, <|endoftext|> and the added tokens
    std::vector<std::int32_t> list;
    for (int t = 0; t < V_; ++t)
      if (short_in_[t]) list.push_back(t);
    short_K_ = int(list.size());
    const int rows = short_K_ + CONTEXT_ROWS;
    const std::size_t row_bytes = head_.scale ? std::size_t(H) : std::size_t(H) * 2;
    void* q = alloc(std::size_t(rows) * row_bytes);
    auto* qs = head_.scale ? f32s(rows) : nullptr;
    short_map_ = i32s(rows);
    short_ids_ = i32s(rows);
    CUDA_CHECK(cudaMemsetAsync(short_map_, 0, std::size_t(rows) * 4, stream_));
    CUDA_CHECK(cudaMemcpyAsync(short_ids_, list.data(), list.size() * 4, cudaMemcpyHostToDevice, stream_));
    kd::gather_head_rows(head_.w, head_.scale, row_bytes, short_ids_, short_K_, q, qs, short_map_, stream_);
    short_head_ = kdense::Matrix{q, qs, rows};
    seg_short_ = segments({short_K_});
    kd::fill(base_, std::size_t(GAMMA) * rows, -INFINITY, stream_);
  }

  if (dflash2_) {
    df2_q_ = bf16s(BLOCK * kd::DF2_Q_DIM);
    df2_attn_o_ = bf16s(BLOCK * kd::DF2_Q_DIM);
    df2_qkv_out_ = f32s(BLOCK * kd::DF2_QKV_ROWS);
    df2_conv_dyn_ = f32s(BLOCK * kd::DF2_CONV_PROJ_ROWS);
    df2_conv_in_ = bf16s(BLOCK * H);
    df2_commit_taps_ = bf16s(std::size_t(BLOCK) * 4 * H);
    df2_fc_out_act_ = bf16s(std::size_t(COMMIT_PIECE) * H);
    df2_c_small_ = bf16s(BLOCK * H);
    df2_commit_kv_ = f32s(std::size_t(BLOCK) * kd::DF2_LAYERS * 2 * kd::KV_DIM);
    df2_g_out_ = f32s(GAMMA * kd::DF2_RANK);
    attn_ws_ = f32s(kd::dflash2_attention_workspace_floats());
    topk_ws_ = alloc(kd::dflash2_topk16_workspace_bytes(GAMMA));
    cand_val_ = f32s(GAMMA * kd::DF2_TOPK);
    cand_idx_ = i32s(GAMMA * kd::DF2_TOPK);
    prev_cand_ = i32s(GAMMA * kd::DF2_TOPK);
    CUDA_CHECK(cudaMemsetAsync(prev_cand_, 0, GAMMA * kd::DF2_TOPK * 4, stream_));
    df2_seg_qkv_ = segments({kd::DF2_QKV_ROWS});
    df2_seg_conv_ = segments({kd::DF2_CONV_PROJ_ROWS});
    df2_seg_kv_ = segments({kd::DF2_LAYERS * 2 * kd::KV_DIM});
    df2_seg_sel_ = segments({kd::DF2_RANK});
    const std::size_t ring_bytes = std::size_t(kd::DF2_LOCAL_SLOTS) * kd::KV_DIM * 2;
    for (int l = 0; l < kd::DF2_LAYERS; ++l) {
      DFlash2Layer& L = df2_layers_[l];
      L.k_ctx = bf16s(std::size_t(kd::DF2_LOCAL_SLOTS) * kd::KV_DIM);
      L.v_ctx = bf16s(std::size_t(kd::DF2_LOCAL_SLOTS) * kd::KV_DIM);
      CUDA_CHECK(cudaMemsetAsync(L.k_ctx, 0, ring_bytes, stream_));
      CUDA_CHECK(cudaMemsetAsync(L.v_ctx, 0, ring_bytes, stream_));
      L.ctx_pos = i32s(kd::DF2_WINDOW);
      df2_ctx_layers_.k_norm[l] = L.k_norm;
      df2_ctx_layers_.k_cache[l] = L.k_ctx;
      df2_ctx_layers_.v_cache[l] = L.v_ctx;
      df2_ctx_layers_.ctx_pos[l] = L.ctx_pos;
    }
  } else {
    q_ = bf16s(BLOCK * kd::Q_DIM);
    attn_o_ = bf16s(BLOCK * kd::Q_DIM);
    f_rnn_ = bf16s(H);
    state_ = bf16s((GAMMA + 1) * kd::RANK);
    latent_ = bf16s(kd::RANK);
    qkv_out_ = f32s(BLOCK * kd::QKV_ROWS);
    attn_ws_ = f32s(std::max(kd::attention_workspace_floats(LOCAL_SLOTS), kd::attention_workspace_floats(ctx_)));
    topk_ws_ = alloc(kd::topk_workspace_bytes(GAMMA, V_));
    CUDA_CHECK(cudaMemsetAsync(topk_ws_, 0, kd::topk_workspace_bytes(GAMMA, V_), stream_));  // kept zeroed
    cand_val_ = f32s(GAMMA * kd::CANDIDATES);
    cand_logits_ = f32s(GAMMA * kd::CANDIDATES);
    part_ = f32s(GAMMA * kd::RNN_OUT);
    cand_idx_ = i32s(GAMMA * kd::CANDIDATES);
    prev_cand_ = i32s(GAMMA * kd::CANDIDATES);
    commit_xin_ = bf16s(std::size_t(3) * BLOCK * H);
    commit_kv_ = f32s(std::size_t(BLOCK) * 3 * 2 * kd::KV_DIM);
    CUDA_CHECK(cudaMemsetAsync(state_, 0, (GAMMA + 1) * kd::RANK * 2, stream_));  // state row 0 stays zero
    CUDA_CHECK(cudaMemsetAsync(prev_cand_, 0, GAMMA * kd::CANDIDATES * 4, stream_));
    CUDA_CHECK(cudaMemsetAsync(f_rnn_, 0, H * 2, stream_));
    seg_qkv_ = segments({kd::QKV_ROWS});
    seg_kv_ = segments({2 * kd::KV_DIM, 2 * kd::KV_DIM, 2 * kd::KV_DIM});  // layer l reads xin[l]
    for (int l = 0; l < 3; ++l) {
      DraftLayer& L = layers_[l];
      const std::size_t bytes = is_local(l) ? std::size_t(LOCAL_SLOTS) * kd::KV_DIM * 2 : ctx_ * GLOBAL_SLOT_BYTES;
      L.k_ctx = alloc(bytes);
      L.v_ctx = alloc(bytes);
      CUDA_CHECK(cudaMemsetAsync(L.k_ctx, 0, bytes, stream_));
      CUDA_CHECK(cudaMemsetAsync(L.v_ctx, 0, bytes, stream_));
      L.ctx_pos = is_local(l) ? i32s(kd::WINDOW) : nullptr;
      ctx_layers_.ctx_norm[l] = L.ctx_norm;
      ctx_layers_.k_norm_ctx[l] = L.k_norm_ctx;
      ctx_layers_.k_cache[l] = L.k_ctx;
      ctx_layers_.v_cache[l] = L.v_ctx;
      ctx_layers_.fp8[l] = !is_local(l) && GLOBAL_FP8;
      ctx_layers_.ctx_pos[l] = L.ctx_pos;
      ctx_layers_.size[l] = is_local(l) ? kd::WINDOW : ctx_;
    }
  }

  accept_.target = i32s(BLOCK);
  accept_.probs = f32s(std::size_t(BLOCK) * V_);
  accept_.resid = f32s(V_);
  accept_.uniforms = f32s(GAMMA);
  accept_.ws = f32s(kd::sampled_accept_workspace_floats(V_));
  accept_.result = static_cast<kd::AcceptResult*>(alloc(sizeof(kd::AcceptResult)));

  reset();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

Drafter::~Drafter() {
  for (auto& [b, g] : draft_graphs_) cudaGraphExecDestroy(g);
  if (commit_graph_) cudaGraphExecDestroy(commit_graph_);
  for (auto& [b, g] : advance_graphs_) cudaGraphExecDestroy(g);
}

void Drafter::load(const std::filesystem::path& model_dir) {
  SafetensorsFile f(model_dir / "mtp.safetensors");
  // Whole data region on the device, then the persistent (aligned, derived) tensors from it.
  const std::uint64_t base = f.data_offset();
  DeviceBuffer raw(f.file_size() - base);
  {
    std::vector<UploadRequest> reqs;
    for (const SafetensorsTensor& t : f.tensors())
      reqs.push_back(upload_request(f, t, static_cast<std::byte*>(raw.p) + (t.offset - base)));
    DeviceUploader up(UploadOptions{std::size_t(16) << 20, 4, 4});
    up.upload(reqs, stream_);
    for (const SafetensorsTensor& t : f.tensors()) {
      if (t.dtype == "BF16")
        bf16_to_act_inplace(static_cast<std::byte*>(raw.p) + (t.offset - base), std::size_t(t.numel()), stream_);
    }
  }
  auto src = [&](const std::string& name, std::initializer_list<std::int64_t> shape) {
    const SafetensorsTensor& t = f.at(name);
    if (t.dtype != "BF16") throw std::runtime_error("mtp.safetensors: " + name + " is not BF16");
    if (!std::equal(t.shape.begin(), t.shape.end(), shape.begin(), shape.end()))
      throw std::runtime_error("mtp.safetensors: unexpected shape of " + name);
    return static_cast<const std::byte*>(raw.p) + (t.offset - base);
  };
  // Copies tensors (concatenated along rows) into one new aligned device buffer.
  auto persist = [&](std::initializer_list<std::pair<std::string, std::vector<std::int64_t>>> parts) {
    std::size_t bytes = 0;
    for (const auto& [name, shape] : parts) bytes += f.at(name).bytes;
    auto* dst = static_cast<std::byte*>(alloc(bytes));
    std::size_t off = 0;
    for (const auto& [name, shape] : parts) {
      const SafetensorsTensor& t = f.at(name);
      if (t.dtype != "BF16" || t.shape != shape) throw std::runtime_error("mtp.safetensors: unexpected " + name);
      CUDA_CHECK(cudaMemcpyAsync(dst + off, static_cast<const std::byte*>(raw.p) + (t.offset - base), t.bytes,
                                 cudaMemcpyDeviceToDevice, stream_));
      off += t.bytes;
    }
    return reinterpret_cast<ActT*>(dst);
  };
  const std::int64_t h = H;
  inter_ = int(f.at("layers.0.mlp.gate_proj.weight").shape.at(0));
  const std::int64_t I = inter_;

  // alpha = bf16(softmax(fuse_weight.float(), -1))
  {
    const SafetensorsTensor& t = f.at("fuse_weight");
    if (t.dtype != "BF16" || t.shape != std::vector<std::int64_t>{4, 4})
      throw std::runtime_error("mtp.safetensors: unexpected fuse_weight");
    const std::vector<float> fw = bf16_to_float(f.read("fuse_weight"));
    ActT host[16];
    for (int r = 0; r < 4; ++r) {
      float m = -INFINITY;
      for (int c = 0; c < 4; ++c) m = std::max(m, fw[r * 4 + c]);
      float e[4], s = 0.f;
      for (int c = 0; c < 4; ++c) s += (e[c] = std::exp(fw[r * 4 + c] - m));
      for (int c = 0; c < 4; ++c) host[r * 4 + c] = act::host_from_float(e[c] / s);
    }
    auto* d = static_cast<ActT*>(alloc(sizeof(host)));
    CUDA_CHECK(cudaMemcpyAsync(d, host, sizeof(host), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    alpha_ = d;
  }

  // FP8 rows of a row-concatenation of [r, K] tensors (kernels::dense::fp8_rows, as dense.fp8_rows in dspark)
  DeviceBuffer cat(std::size_t(2) * I * h * 2);
  auto fp8 = [&](std::initializer_list<std::pair<std::string, std::int64_t>> parts, std::int64_t K = H) {
    std::int64_t rows = 0;
    for (const auto& [name, r] : parts) {
      const std::byte* s = src(name, {r, K});
      CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(cat.p) + rows * K * 2, s, std::size_t(r * K * 2),
                                 cudaMemcpyDeviceToDevice, stream_));
      rows += r;
    }
    void* q = alloc(std::size_t(rows * K));
    auto* scale = static_cast<float*>(alloc(std::size_t(rows) * 4));
    kdense::fp8_rows(static_cast<const ActT*>(cat.p), int(rows), int(K), q, scale, stream_);
    return kdense::Matrix{q, scale, int(rows)};
  };

  // the three layers' context k/v projections: BF16 for large commits (cuBLASLt), and one FP8-row matrix of
  // all three for a small commit (one dense GEMV)
  {
    auto kv = [](int l, const char* x) { return "layers." + std::to_string(l) + ".self_attn." + x + "_proj_ctx.weight"; };
    const std::vector<std::int64_t> shape{kd::KV_DIM, h};
    const ActT* all = persist({{kv(0, "k"), shape}, {kv(0, "v"), shape}, {kv(1, "k"), shape},
                                        {kv(1, "v"), shape}, {kv(2, "k"), shape}, {kv(2, "v"), shape}});
    for (int l = 0; l < 3; ++l) layers_[l].wkv_ctx = all + std::size_t(l) * 2 * kd::KV_DIM * H;
    wkv_all_ = fp8({{kv(0, "k"), kd::KV_DIM}, {kv(0, "v"), kd::KV_DIM}, {kv(1, "k"), kd::KV_DIM},
                    {kv(1, "v"), kd::KV_DIM}, {kv(2, "k"), kd::KV_DIM}, {kv(2, "v"), kd::KV_DIM}});
  }
  for (int l = 0; l < 3; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    const std::string a = p + "self_attn.";
    DraftLayer& L = layers_[l];
    L.in_norm = persist({{p + "input_layernorm.weight", {h}}});
    L.post_norm = persist({{p + "post_attention_layernorm.weight", {h}}});
    L.ctx_norm = persist({{p + "dflare_norm.weight", {h}}});
    L.q_norm = persist({{a + "q_norm.weight", {kd::HEAD_DIM}}});
    L.k_norm = persist({{a + "k_norm.weight", {kd::HEAD_DIM}}});
    L.k_norm_ctx = persist({{a + "k_norm_ctx.weight", {kd::HEAD_DIM}}});
    L.qkv = fp8({{a + "q_proj.weight", kd::Q_DIM}, {a + "k_proj.weight", kd::KV_DIM}, {a + "v_proj.weight", kd::KV_DIM}});
    L.gate_up = fp8({{p + "mlp.gate_proj.weight", I}, {p + "mlp.up_proj.weight", I}});
    L.wo = fp8({{a + "o_proj.weight", h}}, kd::Q_DIM);
    L.w_down = fp8({{p + "mlp.down_proj.weight", h}}, I);
  }
  final_norm_ = persist({{"norm.weight", {h}}});
  mask_embed_ = persist({{"mask_embed", {h}}});
  const std::string mh = "markov_head.";
  w1_ = persist({{mh + "markov_w1.weight", {f.at(mh + "markov_w1.weight").shape.at(0), kd::RANK}}});
  if (f.at(mh + "markov_w1.weight").shape.at(0) < V_)
    throw std::runtime_error("mtp.safetensors: markov_w1 has fewer rows than the vocabulary");
  rnn_w_ = persist({{mh + "rnn_proj.weight", {kd::RNN_OUT, kd::RNN_IN}}});
  rnn_b_ = persist({{mh + "rnn_proj.bias", {kd::RNN_OUT}}});
  f_norm_ = persist({{mh + "dflare_norm.weight", {h}}});

  // fold_w / fold_b (FP64, from the unembedding as the decoder stores it)
  fold_w_ = static_cast<ActT*>(alloc(std::size_t(V_) * kd::RANK * 2));
  fold_b_ = static_cast<ActT*>(alloc(std::size_t(V_) * 2));
  const ActT* hw = persist({{mh + "hidden_out.weight", {h, kd::RANK}}});
  const ActT* hb = persist({{mh + "hidden_out.bias", {h}}});
  kd::fold_head(w_.unembed.w.data, head_.scale, V_, hw, hb, fold_w_, fold_b_, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));  // raw and cat are freed on return
}

void Drafter::load_dflash2(const std::filesystem::path& dflash2_dir) {
  // Parse and validate config.json against supported Inkling-2-Mini DFlash2DraftModel geometry.
  {
    std::ifstream in(dflash2_dir / "config.json");
    if (!in) throw std::runtime_error("DFlash2: cannot open " + (dflash2_dir / "config.json").string());
    const nlohmann::json cfg = nlohmann::json::parse(in);
    bool is_df2 = false;
    if (auto it = cfg.find("architectures"); it != cfg.end() && it->is_array()) {
      for (const auto& a : *it)
        if (a.is_string() && a.get<std::string>() == "DFlash2DraftModel") is_df2 = true;
    }
    if (!is_df2) throw std::runtime_error("DFlash2: config.json architectures must include DFlash2DraftModel");
    auto req_int = [&](const nlohmann::json& j, const char* k, int expect) {
      if (!j.contains(k) || !j.at(k).is_number_integer() || j.at(k).get<int>() != expect)
        throw std::runtime_error(std::string("DFlash2: expected ") + k + "=" + std::to_string(expect));
    };
    req_int(cfg, "hidden_size", H);
    req_int(cfg, "num_hidden_layers", kd::DF2_LAYERS);
    req_int(cfg, "num_attention_heads", kd::DF2_HEADS);
    req_int(cfg, "num_key_value_heads", kd::KV_HEADS);
    req_int(cfg, "head_dim", kd::HEAD_DIM);
    req_int(cfg, "intermediate_size", kd::DF2_INTER);
    req_int(cfg, "sliding_window", kd::DF2_WINDOW);
    if (auto it = cfg.find("layer_types"); it != cfg.end()) {
      if (!it->is_array() || int(it->size()) != kd::DF2_LAYERS)
        throw std::runtime_error("DFlash2: layer_types must have 5 entries");
      for (const auto& lt : *it)
        if (!lt.is_string() || lt.get<std::string>() != "sliding_attention")
          throw std::runtime_error("DFlash2: all 5 layers must be sliding_attention");
    }
    if (auto rp = cfg.find("rope_parameters"); rp != cfg.end() && rp->is_object() && rp->contains("rope_theta")) {
      df2_rope_theta_ = rp->at("rope_theta").get<float>();
    } else if (cfg.contains("rope_theta") && cfg.at("rope_theta").is_number()) {
      df2_rope_theta_ = cfg.at("rope_theta").get<float>();
    }
    if (!(df2_rope_theta_ > 0.f)) throw std::runtime_error("DFlash2: invalid rope_theta");
    if (!cfg.contains("dflash_config") || !cfg.at("dflash_config").is_object())
      throw std::runtime_error("DFlash2: missing dflash_config object");
    const auto& dc = cfg.at("dflash_config");
    req_int(dc, "block_size", BLOCK);
    req_int(dc, "conv_kernel_size", kd::DF2_CONV_K);
    req_int(dc, "conv_group_size", kd::DF2_CONV_GROUP);
    req_int(dc, "selector_rank", kd::DF2_RANK);
    req_int(dc, "selector_top_k", kd::DF2_TOPK);
    if (!dc.contains("target_layer_ids") ||
        dc.at("target_layer_ids").get<std::vector<int>>() != std::vector<int>({5, 23, 41, 48})) {
      throw std::runtime_error("DFlash2: target_layer_ids must be [5, 23, 41, 48] for Inkling-2-Mini");
    }
    df2_mask_token_id_ = dc.value("mask_token_id", 200056);
    if (df2_mask_token_id_ < 0 || df2_mask_token_id_ >= V_)
      throw std::runtime_error("DFlash2: mask_token_id out of target vocabulary range");
    df2_softcap_ = dc.value("final_logit_softcapping", kd::DF2_SOFTCAP);
    const std::string sc_mode = dc.value("selector_softcap_mode", std::string("after_correction"));
    if (sc_mode == "after_correction") df2_softcap_after_correction_ = true;
    else if (sc_mode == "before_correction") df2_softcap_after_correction_ = false;
    else throw std::runtime_error("DFlash2: unsupported selector_softcap_mode '" + sc_mode + "'");
  }

  SafetensorsFile f(dflash2_dir / "model.safetensors");
  const std::uint64_t base = f.data_offset();
  DeviceBuffer raw(f.file_size() - base);
  {
    std::vector<UploadRequest> reqs;
    for (const SafetensorsTensor& t : f.tensors())
      reqs.push_back(upload_request(f, t, static_cast<std::byte*>(raw.p) + (t.offset - base)));
    DeviceUploader up(UploadOptions{std::size_t(16) << 20, 4, 4});
    up.upload(reqs, stream_);
    for (const SafetensorsTensor& t : f.tensors()) {
      if (t.dtype == "BF16")
        bf16_to_act_inplace(static_cast<std::byte*>(raw.p) + (t.offset - base), std::size_t(t.numel()), stream_);
    }
  }
  auto src = [&](const std::string& name, std::initializer_list<std::int64_t> shape) {
    const SafetensorsTensor& t = f.at(name);
    if (t.dtype != "BF16") throw std::runtime_error("DFlash2 model.safetensors: " + name + " is not BF16");
    if (!std::equal(t.shape.begin(), t.shape.end(), shape.begin(), shape.end()))
      throw std::runtime_error("DFlash2 model.safetensors: unexpected shape of " + name);
    return static_cast<const std::byte*>(raw.p) + (t.offset - base);
  };
  auto persist = [&](std::initializer_list<std::pair<std::string, std::vector<std::int64_t>>> parts) {
    std::size_t bytes = 0;
    for (const auto& [name, shape] : parts) bytes += f.at(name).bytes;
    auto* dst = static_cast<std::byte*>(alloc(bytes));
    std::size_t off = 0;
    for (const auto& [name, shape] : parts) {
      const SafetensorsTensor& t = f.at(name);
      if (t.dtype != "BF16" || t.shape != shape)
        throw std::runtime_error("DFlash2 model.safetensors: unexpected " + name);
      CUDA_CHECK(cudaMemcpyAsync(dst + off, static_cast<const std::byte*>(raw.p) + (t.offset - base), t.bytes,
                                 cudaMemcpyDeviceToDevice, stream_));
      off += t.bytes;
    }
    return reinterpret_cast<ActT*>(dst);
  };

  inter_ = kd::DF2_INTER;
  const std::int64_t h = H, I = inter_;
  const std::size_t cat_bytes = std::max<std::size_t>(
      {std::size_t(2) * I * h * 2, std::size_t(kd::DF2_LAYERS) * 2 * kd::KV_DIM * h * 2, std::size_t(h) * 4 * h * 2});
  DeviceBuffer cat(cat_bytes);
  auto fp8 = [&](std::initializer_list<std::pair<std::string, std::int64_t>> parts, std::int64_t K = H) {
    std::int64_t rows = 0;
    for (const auto& [name, r] : parts) {
      const std::byte* s = src(name, {r, K});
      CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(cat.p) + rows * K * 2, s, std::size_t(r * K * 2),
                                 cudaMemcpyDeviceToDevice, stream_));
      rows += r;
    }
    void* q = alloc(std::size_t(rows * K));
    auto* scale = static_cast<float*>(alloc(std::size_t(rows) * 4));
    kdense::fp8_rows(static_cast<const ActT*>(cat.p), int(rows), int(K), q, scale, stream_);
    return kdense::Matrix{q, scale, int(rows)};
  };

  df2_fc_bf16_ = persist({{"fc.weight", {h, 4 * h}}});
  df2_fc_ = fp8({{"fc.weight", h}}, 4 * h);
  df2_hidden_norm_ = persist({{"hidden_norm.weight", {h}}});
  final_norm_ = persist({{"norm.weight", {h}}});

  // Stack all 5 layers' k_proj | v_proj for both large commits (BF16 per layer) and small commits (1 FP8 GEMV).
  {
    auto kv = [](int l, const char* x) { return "layers." + std::to_string(l) + ".self_attn." + x + "_proj.weight"; };
    const std::vector<std::int64_t> sh{kd::KV_DIM, h};
    const ActT* all = persist({{kv(0, "k"), sh}, {kv(0, "v"), sh}, {kv(1, "k"), sh}, {kv(1, "v"), sh},
                               {kv(2, "k"), sh}, {kv(2, "v"), sh}, {kv(3, "k"), sh}, {kv(3, "v"), sh},
                               {kv(4, "k"), sh}, {kv(4, "v"), sh}});
    for (int l = 0; l < kd::DF2_LAYERS; ++l) df2_layers_[l].wkv_ctx = all + std::size_t(l) * 2 * kd::KV_DIM * H;
    df2_wkv_all_ = fp8({{kv(0, "k"), kd::KV_DIM}, {kv(0, "v"), kd::KV_DIM}, {kv(1, "k"), kd::KV_DIM},
                        {kv(1, "v"), kd::KV_DIM}, {kv(2, "k"), kd::KV_DIM}, {kv(2, "v"), kd::KV_DIM},
                        {kv(3, "k"), kd::KV_DIM}, {kv(3, "v"), kd::KV_DIM}, {kv(4, "k"), kd::KV_DIM},
                        {kv(4, "v"), kd::KV_DIM}});
  }

  for (int l = 0; l < kd::DF2_LAYERS; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    const std::string a = p + "self_attn.";
    DFlash2Layer& L = df2_layers_[l];
    L.in_norm = persist({{p + "input_layernorm.weight", {h}}});
    L.post_norm = persist({{p + "post_attention_layernorm.weight", {h}}});
    L.q_norm = persist({{a + "q_norm.weight", {kd::HEAD_DIM}}});
    L.k_norm = persist({{a + "k_norm.weight", {kd::HEAD_DIM}}});
    L.attn_conv_base = persist({{p + "attention_conv.base_kernel", {2, 2, h}}});
    L.mlp_conv_base = persist({{p + "mlp_conv.base_kernel", {2, 2, h}}});
    L.attn_conv_proj = fp8({{p + "attention_conv.kernel_projection.weight", kd::DF2_CONV_PROJ_ROWS}});
    L.mlp_conv_proj = fp8({{p + "mlp_conv.kernel_projection.weight", kd::DF2_CONV_PROJ_ROWS}});
    L.qkv = fp8({{a + "q_proj.weight", kd::DF2_Q_DIM}, {a + "k_proj.weight", kd::KV_DIM}, {a + "v_proj.weight", kd::KV_DIM}});
    L.wo = fp8({{a + "o_proj.weight", h}}, kd::DF2_Q_DIM);
    L.gate_up = fp8({{p + "mlp.gate_proj.weight", I}, {p + "mlp.up_proj.weight", I}});
    L.w_down = fp8({{p + "mlp.down_proj.weight", h}}, I);
  }

  df2_sel_proj_ = fp8({{"candidate_selector.hidden_projection.weight", kd::DF2_RANK}});
  auto persist_codebook = [&](const std::string& name) {
    const SafetensorsTensor& t = f.at(name);
    if (t.dtype != "BF16" || t.shape.size() != 2 || t.shape[0] < V_ || t.shape[1] != kd::DF2_RANK)
      throw std::runtime_error("DFlash2 model.safetensors: unexpected " + name);
    const std::size_t bytes = std::size_t(V_) * kd::DF2_RANK * 2;
    auto* dst = static_cast<ActT*>(alloc(bytes));
    CUDA_CHECK(cudaMemcpyAsync(dst, static_cast<const std::byte*>(raw.p) + (t.offset - base), bytes,
                               cudaMemcpyDeviceToDevice, stream_));
    return dst;
  };
  df2_pred_cb_ = persist_codebook("candidate_selector.predecessor_codebook");
  df2_succ_cb_ = persist_codebook("candidate_selector.successor_codebook");

  // Precompute normalized mask token embedding using target model's embed + embed_norm.
  auto* mask_buf = static_cast<ActT*>(alloc(std::size_t(H) * 2));
  kd::dflash2_embed_token(static_cast<const ActT*>(w_.embed.data), static_cast<const float*>(w_.embed_norm.data),
                          df2_mask_token_id_, mask_buf, stream_, w_.embed_q4());
  mask_embed_ = mask_buf;
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Drafter::reset() {
  if (dflash2_) {
    for (const DFlash2Layer& L : df2_layers_)
      CUDA_CHECK(cudaMemsetAsync(L.ctx_pos, 0xff, kd::DF2_WINDOW * 4, stream_));
  } else {
    for (const DraftLayer& L : layers_)
      if (L.ctx_pos != nullptr) CUDA_CHECK(cudaMemsetAsync(L.ctx_pos, 0xff, kd::WINDOW * 4, stream_));
  }
  length_ = 0;
  set_context_rows(0);
}

std::size_t Drafter::set_context_rows(int rows) {
  if (rows >= short_E_) return 0;
  for (std::size_t i = std::size_t(rows); i < short_extra_.size(); ++i) short_in_[short_extra_[i]] = 0;
  short_extra_.resize(std::size_t(rows));
  short_E_ = rows;
  const int seg[3] = {0, short_K_ + rows, 0};  // pageable source: copied before cudaMemcpyAsync returns
  CUDA_CHECK(cudaMemcpyAsync(seg_short_, seg, sizeof seg, cudaMemcpyHostToDevice, stream_));
  // the dropped rows' base logits back to -inf (the GEMV only writes the rows in use)
  for (int i = 0; i < GAMMA; ++i)
    kd::fill(base_ + std::size_t(i) * short_head_.rows + short_K_ + rows, std::size_t(CONTEXT_ROWS - rows), -INFINITY,
             stream_);
  return sizeof seg;
}

std::size_t Drafter::note_tokens(std::span<const std::int32_t> ids) {
  std::vector<std::int32_t> add;
  for (std::int32_t t : ids)
    if (t >= 0 && t < V_ && !short_in_[t] && short_E_ + int(add.size()) < CONTEXT_ROWS) {
      short_in_[t] = 1;
      add.push_back(t);
    }
  if (add.empty()) return 0;
  const std::size_t row = std::size_t(short_K_ + short_E_);
  const std::size_t row_bytes = head_.scale ? std::size_t(H) : std::size_t(H) * 2;
  // pageable sources: copied before cudaMemcpyAsync returns; short_ids_ is free once earlier gathers ran
  const std::size_t ids_bytes = add.size() * 4;
  CUDA_CHECK(cudaMemcpyAsync(short_ids_, add.data(), ids_bytes, cudaMemcpyHostToDevice, stream_));
  kd::gather_head_rows(head_.w, head_.scale, row_bytes, short_ids_, int(add.size()),
                       static_cast<std::uint8_t*>(const_cast<void*>(short_head_.w)) + row * row_bytes,
                       head_.scale ? const_cast<float*>(short_head_.scale) + row : nullptr, short_map_ + row, stream_);
  short_extra_.insert(short_extra_.end(), add.begin(), add.end());
  short_E_ += int(add.size());
  const int seg[3] = {0, short_K_ + short_E_, 0};
  CUDA_CHECK(cudaMemcpyAsync(seg_short_, seg, sizeof seg, cudaMemcpyHostToDevice, stream_));
  return ids_bytes + sizeof seg;
}

void Drafter::set_seed(std::uint64_t seed) { kd::set_seed(params_, seed, stream_); }

void Drafter::global_written(int begin, int end) {
  if (dflash2_ || checkpoint_ == nullptr || checkpoint_clobbered_ || end <= begin) return;
  // global slot = position % ctx; the checkpoint holds slots [0, min(L, ctx))
  const int live = std::min(checkpoint_length_, ctx_);
  checkpoint_clobbered_ = end - begin >= ctx_ || begin % ctx_ < live || begin / ctx_ != (end - 1) / ctx_;
}

// layout: local k/v of layers 0 and 2 (ring slots only), their ctx_pos, f_rnn, global k and v [ctx slots]
namespace {
constexpr std::size_t LOCAL_BYTES = std::size_t(kd::WINDOW) * kd::KV_DIM * 2;
constexpr std::size_t DF2_LOCAL_BYTES = std::size_t(kd::DF2_WINDOW) * kd::KV_DIM * 2;
constexpr std::size_t DF2_POS_BYTES = std::size_t(kd::DF2_WINDOW) * sizeof(std::int32_t);
}

std::size_t Drafter::checkpoint_bytes() const {
  if (dflash2_) return kd::DF2_LAYERS * (2 * DF2_LOCAL_BYTES + DF2_POS_BYTES);
  return 4 * LOCAL_BYTES + 2 * kd::WINDOW * sizeof(std::int32_t) + std::size_t(H) * 2 +
         2 * ctx_ * GLOBAL_SLOT_BYTES;
}

std::array<Drafter::Region, 3> Drafter::checkpoint_regions(int length) const {
  if (dflash2_) return {Region{0, checkpoint_bytes()}, Region{0, 0}, Region{0, 0}};
  const std::size_t head = 4 * LOCAL_BYTES + 2 * kd::WINDOW * sizeof(std::int32_t) + std::size_t(H) * 2;
  const std::size_t live = std::size_t(std::min(std::max(length, 0), ctx_)) * GLOBAL_SLOT_BYTES;
  return {Region{0, head}, Region{head, live}, Region{head + ctx_ * GLOBAL_SLOT_BYTES, live}};
}

void Drafter::adopt_checkpoint(const void* src, int length, std::span<const std::int32_t> context) {
  checkpoint_ = src;
  checkpoint_length_ = length;
  checkpoint_clobbered_ = true;
  checkpoint_extra_.assign(context.begin(), context.end());
}

std::size_t Drafter::save_checkpoint(void* dst) {
  auto* d = static_cast<std::byte*>(dst);
  std::size_t copied_bytes = 0;
  auto copy = [&](std::size_t off, const void* src, std::size_t bytes) {
    if (bytes == 0) return;
    CUDA_CHECK(cudaMemcpyAsync(d + off, src, bytes, cudaMemcpyDeviceToDevice, stream_));
    copied_bytes += bytes;
  };
  std::size_t off = 0;
  if (dflash2_) {
    for (int l = 0; l < kd::DF2_LAYERS; ++l) {
      copy(off, df2_layers_[l].k_ctx, DF2_LOCAL_BYTES);
      copy(off + DF2_LOCAL_BYTES, df2_layers_[l].v_ctx, DF2_LOCAL_BYTES);
      off += 2 * DF2_LOCAL_BYTES;
    }
    for (int l = 0; l < kd::DF2_LAYERS; ++l) {
      copy(off, df2_layers_[l].ctx_pos, DF2_POS_BYTES);
      off += DF2_POS_BYTES;
    }
  } else {
    for (int l : {0, 2}) {
      copy(off, layers_[l].k_ctx, LOCAL_BYTES);
      copy(off + LOCAL_BYTES, layers_[l].v_ctx, LOCAL_BYTES);
      off += 2 * LOCAL_BYTES;
    }
    for (int l : {0, 2}) {
      copy(off, layers_[l].ctx_pos, kd::WINDOW * sizeof(std::int32_t));
      off += kd::WINDOW * sizeof(std::int32_t);
    }
    copy(off, f_rnn_, std::size_t(H) * 2);
    off += std::size_t(H) * 2;
    const std::size_t live = std::size_t(std::min(length_, ctx_)) * GLOBAL_SLOT_BYTES;
    copy(off, layers_[1].k_ctx, live);
    copy(off + ctx_ * GLOBAL_SLOT_BYTES, layers_[1].v_ctx, live);
  }
  checkpoint_ = dst;
  checkpoint_length_ = length_;
  checkpoint_clobbered_ = false;
  checkpoint_extra_ = short_extra_;
  return copied_bytes;
}

std::size_t Drafter::restore_checkpoint(const void* src) {
  if (src != checkpoint_) throw std::invalid_argument("Drafter::restore_checkpoint: not the last checkpoint saved");
  const auto* s = static_cast<const std::byte*>(src);
  std::size_t copied_bytes = 0;
  auto copy = [&](void* dst, std::size_t off, std::size_t bytes) {
    if (bytes == 0) return;
    CUDA_CHECK(cudaMemcpyAsync(dst, s + off, bytes, cudaMemcpyDeviceToDevice, stream_));
    copied_bytes += bytes;
  };
  std::size_t off = 0;
  if (dflash2_) {
    for (int l = 0; l < kd::DF2_LAYERS; ++l) {
      copy(df2_layers_[l].k_ctx, off, DF2_LOCAL_BYTES);
      copy(df2_layers_[l].v_ctx, off + DF2_LOCAL_BYTES, DF2_LOCAL_BYTES);
      off += 2 * DF2_LOCAL_BYTES;
    }
    for (int l = 0; l < kd::DF2_LAYERS; ++l) {
      copy(df2_layers_[l].ctx_pos, off, DF2_POS_BYTES);
      off += DF2_POS_BYTES;
    }
  } else {
    for (int l : {0, 2}) {
      copy(layers_[l].k_ctx, off, LOCAL_BYTES);
      copy(layers_[l].v_ctx, off + LOCAL_BYTES, LOCAL_BYTES);
      off += 2 * LOCAL_BYTES;
    }
    for (int l : {0, 2}) {
      copy(layers_[l].ctx_pos, off, kd::WINDOW * sizeof(std::int32_t));
      off += kd::WINDOW * sizeof(std::int32_t);
    }
    copy(f_rnn_, off, std::size_t(H) * 2);
    off += std::size_t(H) * 2;
    if (checkpoint_clobbered_) {
      const std::size_t live = std::size_t(std::min(checkpoint_length_, ctx_)) * GLOBAL_SLOT_BYTES;
      copy(layers_[1].k_ctx, off, live);
      copy(layers_[1].v_ctx, off + ctx_ * GLOBAL_SLOT_BYTES, live);
    }
  }
  length_ = checkpoint_length_;
  checkpoint_clobbered_ = false;
  // the context tokens of the checkpoint: drop the ones added since (rebuild if they were reset meanwhile)
  const bool prefix = short_extra_.size() >= checkpoint_extra_.size() &&
                      std::equal(checkpoint_extra_.begin(), checkpoint_extra_.end(), short_extra_.begin());
  copied_bytes += set_context_rows(prefix ? int(checkpoint_extra_.size()) : 0);
  if (!prefix) copied_bytes += note_tokens(checkpoint_extra_);
  return copied_bytes;
}

void Drafter::run_commit(const ActT* feats, int n, int row0, bool last_piece) {
  for (int l = 0; l < 3; ++l) {
    const DraftLayer& L = layers_[l];
    kd::mix_rmsnorm(feats, n, alpha_ + 4 * l, L.ctx_norm, ctx_c_, stream_);
    gemm_.run(ctx_c_, n, L.wkv_ctx, 2 * kd::KV_DIM, H, ctx_kv_, stream_);
    const int size = is_local(l) ? kd::WINDOW : ctx_;
    kd::ctx_kv_store(ctx_kv_, n, params_, row0, n - std::min(n, size), L.k_norm_ctx, L.k_ctx, L.v_ctx,
                     !is_local(l) && GLOBAL_FP8, size, L.ctx_pos, stream_);
  }
  if (last_piece) kd::mix_rmsnorm(feats + std::size_t(n - 1) * 4 * H, 1, alpha_ + 12, f_norm_, f_rnn_, stream_);
}

void Drafter::run_commit_dflash2(const ActT* feats, int n, int row0) {
  gemm_.run(feats, n, df2_fc_bf16_, H, 4 * H, df2_fc_out_act_, stream_);
  kd::dflash2_rmsnorm(df2_fc_out_act_, n, df2_hidden_norm_, ctx_c_, stream_);
  for (int l = 0; l < kd::DF2_LAYERS; ++l) {
    const DFlash2Layer& L = df2_layers_[l];
    gemm_.run(ctx_c_, n, L.wkv_ctx, 2 * kd::KV_DIM, H, ctx_kv_, stream_);
    kd::dflash2_ctx_kv_store(ctx_kv_, n, params_, row0, n - std::min(n, kd::DF2_WINDOW), L.k_norm, L.k_ctx, L.v_ctx,
                             kd::DF2_WINDOW, L.ctx_pos, df2_rope_theta_, stream_);
  }
}

void Drafter::commit(const Tensor& feats, int start) {
  if (feats.dtype != kActDType || feats.ne[0] != H || feats.ne[1] != 4 || !feats.is_contiguous())
    throw std::invalid_argument("Drafter::commit: feats must be contiguous bf16 [n, 4, 2304]");
  commit(static_cast<const ActT*>(feats.data), feats.ne[2] * feats.ne[3], start);
}

void Drafter::commit(const ActT* feats, int n, int start) {
  if (n <= 0) return;
  if (n <= BLOCK) {
    kd::set_commit(params_, feats, n, start, stream_);
    if (commit_graph_) CUDA_CHECK(cudaGraphLaunch(commit_graph_, stream_));
    else if (dflash2_) run_small_commit_dflash2();
    else run_small_commit();
  } else {
    kd::set_commit_start(params_, start, stream_);
    for (int r0 = 0; r0 < n; r0 += COMMIT_PIECE) {
      const int k = std::min(COMMIT_PIECE, n - r0);
      if (dflash2_) run_commit_dflash2(feats + std::size_t(r0) * 4 * H, k, r0);
      else run_commit(feats + std::size_t(r0) * 4 * H, k, r0, r0 + k == n);
    }
  }
  global_written(start, start + n);
  length_ = start + n;
}

// the k/v context rows of the n = params_->ncommit rows of params_->commit_feats in all three layers
void Drafter::run_small_commit() {
  kd::commit_mix(params_, alpha_, ctx_layers_, f_norm_, commit_xin_, f_rnn_, stream_);
  kdense::proj(commit_xin_, BLOCK, wkv_all_, seg_kv_, 3, commit_kv_, true, stream_);
  kd::commit_store(commit_kv_, params_, ctx_layers_, stream_);
}

void Drafter::run_small_commit_dflash2() {
  kd::dflash2_commit_prep(params_, df2_commit_taps_, stream_);
  kdense::gemv(df2_commit_taps_, BLOCK, 4 * H, df2_fc_, delta_, true, stream_);
  kd::dflash2_rmsnorm_f32(delta_, BLOCK, df2_hidden_norm_, df2_c_small_, stream_);
  kdense::proj(df2_c_small_, BLOCK, df2_wkv_all_, df2_seg_kv_, 1, df2_commit_kv_, true, stream_);
  kd::dflash2_commit_store(df2_commit_kv_, params_, df2_ctx_layers_, df2_rope_theta_, stream_);
}

void Drafter::run_draft(int bucket) {
  const ActT* in0 = dflash2_ ? df2_layers_[0].in_norm : layers_[0].in_norm;
  kd::embed_block(static_cast<const ActT*>(w_.embed.data), static_cast<const float*>(w_.embed_norm.data),
                  mask_embed_, in0, params_, res_, xn_, picks_, stream_, w_.embed_q4());
  if (dflash2_) run_layers_dflash2();
  else run_layers(bucket);
}

void Drafter::run_layers_dflash2() {
  for (int l = 0; l < kd::DF2_LAYERS; ++l) {
    const DFlash2Layer& L = df2_layers_[l];
    kdense::proj(xn_, BLOCK, L.attn_conv_proj, df2_seg_conv_, 1, df2_conv_dyn_, true, stream_);
    kd::dflash2_conv_prepare(xn_, L.attn_conv_base, df2_conv_dyn_, df2_conv_in_, BLOCK, stream_);
    kdense::proj(df2_conv_in_, BLOCK, L.qkv, df2_seg_qkv_, 1, df2_qkv_out_, true, stream_);
    kd::dflash2_qkv_rope(df2_qkv_out_, L.q_norm, L.k_norm, params_, df2_q_, L.k_ctx, L.v_ctx, df2_rope_theta_, stream_);
    kd::dflash2_block_attention(df2_q_, L.k_ctx, L.v_ctx, L.ctx_pos, params_, attn_ws_, df2_attn_o_, stream_);
    kdense::gemv(df2_attn_o_, BLOCK, kd::DF2_Q_DIM, L.wo, delta_, true, stream_);
    kd::dflash2_conv_finish_add_rmsnorm(res_, delta_, L.attn_conv_base, df2_conv_dyn_, L.post_norm, xn_, BLOCK, stream_);

    kdense::proj(xn_, BLOCK, L.mlp_conv_proj, df2_seg_conv_, 1, df2_conv_dyn_, true, stream_);
    kd::dflash2_conv_prepare(xn_, L.mlp_conv_base, df2_conv_dyn_, df2_conv_in_, BLOCK, stream_);
    kdense::proj(df2_conv_in_, BLOCK, L.gate_up, seg_gate_up_, 1, gu_out_, true, stream_);
    kd::silu_mul(gu_out_, BLOCK, inter_, act_, stream_);
    kdense::gemv(act_, BLOCK, inter_, L.w_down, delta_, true, stream_);
    kd::dflash2_conv_finish_add_rmsnorm(res_, delta_, L.mlp_conv_base, df2_conv_dyn_,
                                        l < kd::DF2_LAYERS - 1 ? df2_layers_[l + 1].in_norm : final_norm_, xn_, BLOCK,
                                        stream_);
  }
  const ActT* h = xn_ + H;  // rows 1..7 (GAMMA)
  kdense::proj(h, GAMMA, short_head_, seg_short_, 1, base_, true, stream_);
  kd::dflash2_topk16(base_, GAMMA, short_head_.rows, short_map_, cand_val_, cand_idx_, topk_ws_, stream_);
  kdense::proj(h, GAMMA, df2_sel_proj_, df2_seg_sel_, 1, df2_g_out_, true, stream_);
  kd::dflash2_select_path(cand_val_, cand_idx_, df2_g_out_, df2_pred_cb_, df2_succ_cb_, prev_cand_, draft_logits_, V_,
                          df2_softcap_, df2_softcap_after_correction_, tokens_, conf_, qstats_, params_, stream_);
}

// the draft after the block's input rows (res_, xn_)
void Drafter::run_layers(int bucket) {
  for (int l = 0; l < 3; ++l) {
    const DraftLayer& L = layers_[l];
    const bool local = is_local(l);
    kdense::proj(xn_, BLOCK, L.qkv, seg_qkv_, 1, qkv_out_, true, stream_);
    kd::qkv_rope(qkv_out_, L.q_norm, L.k_norm, params_, q_, L.k_ctx, L.v_ctx, !local && GLOBAL_FP8, local, ctx_,
                 stream_);
    kd::block_attention(q_, L.k_ctx, L.v_ctx, local ? LOCAL_SLOTS : bucket, L.ctx_pos, !local && GLOBAL_FP8, params_,
                        attn_ws_, attn_o_, stream_);
    kdense::gemv(attn_o_, BLOCK, kd::Q_DIM, L.wo, delta_, true, stream_);
    kd::add_rmsnorm(res_, delta_, L.post_norm, xn_, BLOCK, stream_);
    kdense::proj(xn_, BLOCK, L.gate_up, seg_gate_up_, 1, gu_out_, true, stream_);
    kd::silu_mul(gu_out_, BLOCK, inter_, act_, stream_);
    kdense::gemv(act_, BLOCK, inter_, L.w_down, delta_, true, stream_);
    kd::add_rmsnorm(res_, delta_, l < 2 ? layers_[l + 1].in_norm : final_norm_, xn_, BLOCK, stream_);
  }
  const ActT* h = xn_ + H;  // rows 1..7
  kdense::proj(h, GAMMA, short_head_, seg_short_, 1, base_, true, stream_);
  kd::topk_candidates(base_, GAMMA, short_head_.rows, short_map_, cand_val_, cand_idx_, topk_ws_, stream_);
  kd::markov_pre(rnn_w_, h, f_rnn_, part_, stream_);
  for (int i = 0; i < GAMMA; ++i) {
    kd::markov_step(i, rnn_w_, rnn_b_, w1_, part_, picks_, cand_idx_, params_, state_, latent_, stream_);
    kd::markov_logits(i, fold_w_, fold_b_, cand_val_, cand_idx_, latent_, params_, cand_logits_, picks_, stream_);
  }
  kd::markov_finish(cand_idx_, cand_logits_, picks_, prev_cand_, draft_logits_, V_, tokens_, conf_, qstats_, params_,
                    stream_);
}

// the smallest bucket holding the block; the whole ring once the context reaches it (and without graphs)
int Drafter::bucket_for(int P) const {
  if (dflash2_) return kd::DF2_LOCAL_SLOTS;
  for (const auto& [b, g] : draft_graphs_)
    if (b >= P + BLOCK) return b;
  return ctx_;
}

Drafter::Draft Drafter::draft(int anchor, int P, float temperature) {
  if (anchor < 0 || anchor >= V_) throw std::invalid_argument("Drafter::draft: anchor out of range");
  kd::set_draft_inputs(params_, anchor, P, temperature, stream_);
  global_written(P, P + BLOCK);  // the block's own rows
  const int bucket = bucket_for(P);
  if (auto it = draft_graphs_.find(bucket); it != draft_graphs_.end())
    CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  else
    run_draft(bucket);
  return {tokens_, draft_logits_, conf_, qstats_};
}

// the round's parameters, the small commit and the draft, all from device values: the commit's input mix and
// the block's input rows share the first launch
void Drafter::run_advance(int bucket) {
  if (dflash2_) {
    kd::dflash2_round_begin(params_, round_, static_cast<const ActT*>(w_.embed.data),
                            static_cast<const float*>(w_.embed_norm.data), mask_embed_, df2_layers_[0].in_norm,
                            df2_commit_taps_, res_, xn_, stream_, w_.embed_q4());
    kdense::gemv(df2_commit_taps_, BLOCK, 4 * H, df2_fc_, delta_, true, stream_);
    kd::dflash2_rmsnorm_f32(delta_, BLOCK, df2_hidden_norm_, df2_c_small_, stream_);
    kdense::proj(df2_c_small_, BLOCK, df2_wkv_all_, df2_seg_kv_, 1, df2_commit_kv_, true, stream_);
    kd::dflash2_commit_store(df2_commit_kv_, params_, df2_ctx_layers_, df2_rope_theta_, stream_);
    run_layers_dflash2();
    return;
  }
  kd::round_begin(params_, round_, alpha_, ctx_layers_, f_norm_, static_cast<const ActT*>(w_.embed.data),
                  static_cast<const float*>(w_.embed_norm.data), mask_embed_, layers_[0].in_norm, commit_xin_, f_rnn_,
                  res_, xn_, picks_, stream_, w_.embed_q4());
  kdense::proj(commit_xin_, BLOCK, wkv_all_, seg_kv_, 3, commit_kv_, true, stream_);
  kd::commit_store(commit_kv_, params_, ctx_layers_, stream_);
  run_layers(bucket);
}

void Drafter::bind_round(const kd::RoundInputs& in) {
  for (auto& [b, g] : advance_graphs_) cudaGraphExecDestroy(g);
  advance_graphs_.clear();
  round_ = in;
  round_bound_ = true;
  for (const auto& [b, g] : draft_graphs_) {  // the same buckets as the draft graphs (none before capture())
    capture_begin(stream_);
    run_advance(b);
    advance_graphs_[b] = capture_end(stream_);
  }
}

Drafter::Draft Drafter::advance(int start) {
  if (!round_bound_) throw std::logic_error("Drafter::advance: no bind_round()");
  pending_start_ = start;
  global_written(start, start + 2 * BLOCK);  // the commit (<= 8 rows) and the next block (P <= start + 8)
  const int bucket = bucket_for(start + BLOCK);  // P = start + n <= start + 8
  if (auto it = advance_graphs_.find(bucket); it != advance_graphs_.end())
    CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  else
    run_advance(bucket);
  return {tokens_, draft_logits_, conf_, qstats_};
}

void Drafter::capture() {
  std::vector<int> buckets;
  if (dflash2_) {
    buckets.push_back(kd::DF2_LOCAL_SLOTS);
  } else {
    for (int b = 1024; b < ctx_; b *= 2) buckets.push_back(b);
    buckets.push_back(ctx_);
  }
  // warm-up outside capture (cuBLASLt plans), then capture; the scratch context rows are reset afterwards
  kd::set_draft_inputs(params_, 0, 0, 0.f, stream_);
  kd::set_commit(params_, warm_feats_, BLOCK, 0, stream_);
  for (int b : buckets) run_draft(b);
  if (dflash2_) run_small_commit_dflash2();
  else run_small_commit();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (int b : buckets) {
    capture_begin(stream_);
    run_draft(b);
    draft_graphs_[b] = capture_end(stream_);
  }
  capture_begin(stream_);
  if (dflash2_) run_small_commit_dflash2();
  else run_small_commit();
  commit_graph_ = capture_end(stream_);
  if (round_bound_) bind_round(kd::RoundInputs(round_));
  reset();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace ingine::model

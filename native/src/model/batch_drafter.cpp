#include "model/batch_drafter.h"

#include "core/act.h"
#include "core/device.h"
#include "core/gguf.h"
#include "core/safetensors.h"
#include "kernels/draft/shortlist.h"
#include "model/model_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>

namespace ingine::model {

namespace kd = kernels::draft;
namespace kdense = kernels::dense;

namespace {

constexpr int H = kd::H;
constexpr int LOCAL_SLOTS = kd::WINDOW + kd::BLOCK;
constexpr int PREFILL_CHUNK_STRIDE = 32;
constexpr bool GLOBAL_FP8 = true;
constexpr std::size_t GLOBAL_SLOT_BYTES = std::size_t(kd::KV_DIM) * (GLOBAL_FP8 ? 1 : 2);

bool is_local(int l) { return l != 1; }

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

void* BatchDrafter::alloc(std::size_t bytes) {
  storage_.emplace_back(bytes);
  return storage_.back().p;
}

BatchDrafter::BatchDrafter(const std::filesystem::path& model_dir, const Weights& w, int max_slots, int max_ctx,
                           int draft_ctx, cudaStream_t stream, int shortlist_rows, const kd::HeadShards& head_shards,
                           int max_commit, int draft_len)
    : w_(w),
      stream_(stream),
      max_slots_(max_slots),
      max_commit_(max_commit),
      draft_len_(draft_len),
      ctx_(draft_ctx > 0 ? std::min({max_ctx, DRAFT_CTX, draft_ctx}) : std::min(max_ctx, DRAFT_CTX)) {
  if (max_slots_ < 1 || max_slots_ > MAX_SLOTS)
    throw std::invalid_argument("BatchDrafter: max_slots must be in 1..4");
  if (max_commit_ < 1 || max_commit_ > BLOCK)
    throw std::invalid_argument("BatchDrafter: max_commit must be in 1..8");
  if (draft_len_ < 1 || draft_len_ > GAMMA)
    throw std::invalid_argument("BatchDrafter: draft_len must be in 1..7");
  if (ctx_ < 2 * BLOCK) throw std::invalid_argument("BatchDrafter: draft_ctx must hold two blocks");

  if (head_shards.n_shards > 0) {
    head_shards_ = head_shards;
  } else {
    head_shards_.n_shards = 1;
    head_shards_.rows_per_shard = w.unembed.rows();
    head_shards_.total_rows = w.unembed.rows();
    head_shards_.shard[0] = kdense::Matrix{
        w.unembed.w.data,
        w.unembed.fp8() ? static_cast<const float*>(w.unembed.scale.data) : nullptr,
        w.unembed.rows(),
        static_cast<const std::int64_t*>(w.unembed.blk_off.data),
        static_cast<const std::int32_t*>(w.unembed.blk_soff.data),
        static_cast<const std::uint16_t*>(w.unembed.q8_scale.data)};
  }
  head_ = head_shards_.shard[0];
  head_.rows = head_shards_.total_rows;
  V_ = head_shards_.total_rows;
  load(model_dir);

  auto bf16s = [&](std::size_t n) { return static_cast<ActT*>(alloc(n * 2)); };
  auto f32s = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto i32s = [&](std::size_t n) { return static_cast<std::int32_t*>(alloc(n * 4)); };

  const std::size_t max_blk_rows = std::size_t(max_slots_) * BLOCK;
  const std::size_t max_gam_rows = std::size_t(max_slots_) * draft_len_;
  res_ = bf16s(max_blk_rows * H);
  xn_ = bf16s(max_blk_rows * H);
  q_ = bf16s(max_blk_rows * kd::Q_DIM);
  attn_o_ = bf16s(max_blk_rows * kd::Q_DIM);
  delta_ = f32s(max_blk_rows * H);
  act_ = bf16s(max_blk_rows * inter_);
  h_ = bf16s(max_gam_rows * H);
  qkv_out_ = f32s(max_blk_rows * kd::QKV_ROWS);
  gu_out_ = f32s(max_blk_rows * 2 * inter_);

  // Segment tables
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
  seg_qkv_ = segments({kd::QKV_ROWS});
  seg_gate_up_ = segments({2 * inter_});
  seg_kv_ = segments({2 * kd::KV_DIM, 2 * kd::KV_DIM, 2 * kd::KV_DIM});

  // Shared shortlist head (ranked + specials + union of active slots' context tokens)
  {
    short_in_.assign(V_, 0);
    int taken = 0;
    for (std::uint32_t t : kd::draft_shortlist_ranks())
      if (taken < shortlist_rows && t < std::uint32_t(V_) && !short_in_[t]) short_in_[t] = 1, ++taken;
    for (int t = 199990; t < V_; ++t) short_in_[t] = 1;
    std::vector<std::int32_t> list;
    for (int t = 0; t < V_; ++t)
      if (short_in_[t]) list.push_back(t);
    short_K_ = int(list.size());
    const bool blocked_q6 = head_shards_.blocked();
    const int rows = blocked_q6 ? ((short_K_ + CONTEXT_ROWS + 15) & ~15) : (short_K_ + CONTEXT_ROWS);
    short_map_ = i32s(rows);
    short_ids_ = i32s(rows);
    CUDA_CHECK(cudaMemsetAsync(short_map_, 0, std::size_t(rows) * 4, stream_));
    CUDA_CHECK(cudaMemcpyAsync(short_ids_, list.data(), list.size() * 4, cudaMemcpyHostToDevice, stream_));
    if (blocked_q6) {
      void* q = alloc(std::size_t(rows) * H);
      const int n_blks = rows / 16;
      const std::size_t sc_u16_per_row = std::size_t(H / 16) * 2;
      short_q6_scales_ = static_cast<std::uint16_t*>(alloc(std::size_t(rows) * sc_u16_per_row * 2));
      CUDA_CHECK(cudaMemsetAsync(q, 0, std::size_t(rows) * H, stream_));
      CUDA_CHECK(cudaMemsetAsync(short_q6_scales_, 0, std::size_t(rows) * sc_u16_per_row * 2, stream_));
      std::vector<std::int64_t> bo(n_blks);
      std::vector<std::int32_t> bso(n_blks);
      for (int b = 0; b < n_blks; ++b) {
        bo[b] = std::int64_t(b) * 16 * H;
        bso[b] = -2 - std::int32_t(b) * 16 * int(sc_u16_per_row);
      }
      short_blk_off_ = static_cast<std::int64_t*>(alloc(std::size_t(n_blks) * 8));
      short_blk_soff_ = static_cast<std::int32_t*>(alloc(std::size_t(n_blks) * 4));
      CUDA_CHECK(cudaMemcpyAsync(short_blk_off_, bo.data(), std::size_t(n_blks) * 8, cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(short_blk_soff_, bso.data(), std::size_t(n_blks) * 4, cudaMemcpyHostToDevice, stream_));
      kd::gather_head_rows_shards(head_shards_, H, short_ids_, short_K_, 0, q, nullptr, short_q6_scales_, short_map_,
                                  stream_);
      short_head_ = kdense::Matrix{q, nullptr, rows, short_blk_off_, short_blk_soff_, short_q6_scales_};
    } else {
      const std::size_t row_bytes = head_.scale ? std::size_t(H) : std::size_t(H) * 2;
      void* q = alloc(std::size_t(rows) * row_bytes);
      auto* qs = head_.scale ? f32s(rows) : nullptr;
      kd::gather_head_rows_shards(head_shards_, row_bytes, short_ids_, short_K_, 0, q, qs, nullptr, short_map_,
                                  stream_);
      short_head_ = kdense::Matrix{q, qs, rows};
    }
    seg_short_ = segments({short_K_});
    base_ = f32s(max_gam_rows * std::size_t(rows));
    kd::fill(base_, max_gam_rows * std::size_t(rows), -INFINITY, stream_);
  }

  attn_ws_ = f32s(std::max(kd::attention_workspace_floats(LOCAL_SLOTS), kd::attention_workspace_floats(ctx_)));
  topk_ws_ = alloc(kd::topk_workspace_bytes(draft_len_, V_));
  CUDA_CHECK(cudaMemsetAsync(topk_ws_, 0, kd::topk_workspace_bytes(draft_len_, V_), stream_));
  cand_val_ = f32s(max_gam_rows * kd::CANDIDATES);
  cand_logits_ = f32s(max_gam_rows * kd::CANDIDATES);
  part_ = f32s(max_gam_rows * kd::RNN_OUT);
  cand_idx_ = i32s(max_gam_rows * kd::CANDIDATES);

  const std::size_t commit_rows = std::max<std::size_t>(PREFILL_CHUNK_STRIDE, std::size_t(max_slots_) * max_commit_);
  warm_feats_ = bf16s(std::size_t(BLOCK) * 4 * H);
  commit_xin_ = bf16s(std::size_t(3) * commit_rows * H);
  commit_kv_ = f32s(commit_rows * 3 * 2 * kd::KV_DIM);
  CUDA_CHECK(cudaMemsetAsync(warm_feats_, 0, std::size_t(BLOCK) * 4 * H * 2, stream_));

  slots_dev_ = static_cast<kd::BatchSlotDesc*>(alloc(sizeof(kd::BatchSlotDesc)));
  CUDA_CHECK(cudaMemsetAsync(slots_dev_, 0, sizeof(kd::BatchSlotDesc), stream_));

  // Per-slot context rings and Markov head state
  for (int s = 0; s < max_slots_; ++s) {
    SlotState& st = slots_[s];
    st.params = static_cast<kd::DraftParams*>(alloc(sizeof(kd::DraftParams)));
    CUDA_CHECK(cudaMemsetAsync(st.params, 0, sizeof(kd::DraftParams), stream_));
    st.f_rnn = bf16s(H);
    st.state = bf16s((GAMMA + 1) * kd::RANK);
    st.latent = bf16s(kd::RANK);
    st.picks = static_cast<unsigned long long*>(alloc(GAMMA * sizeof(unsigned long long)));
    st.tokens = i32s(GAMMA);
    st.conf = f32s(GAMMA);
    CUDA_CHECK(cudaMemsetAsync(st.f_rnn, 0, H * 2, stream_));
    CUDA_CHECK(cudaMemsetAsync(st.state, 0, (GAMMA + 1) * kd::RANK * 2, stream_));
    CUDA_CHECK(cudaMemsetAsync(st.picks, 0, GAMMA * sizeof(unsigned long long), stream_));
    CUDA_CHECK(cudaMemsetAsync(st.tokens, 0, GAMMA * 4, stream_));
    CUDA_CHECK(cudaMemsetAsync(st.conf, 0, GAMMA * 4, stream_));

    for (int l = 0; l < 3; ++l) {
      const DraftLayerWeights& L = layers_[l];
      const std::size_t bytes = is_local(l) ? std::size_t(LOCAL_SLOTS) * kd::KV_DIM * 2 : ctx_ * GLOBAL_SLOT_BYTES;
      void* kc = alloc(bytes);
      void* vc = alloc(bytes);
      CUDA_CHECK(cudaMemsetAsync(kc, 0, bytes, stream_));
      CUDA_CHECK(cudaMemsetAsync(vc, 0, bytes, stream_));
      std::int32_t* cpos = is_local(l) ? i32s(kd::WINDOW) : nullptr;
      st.ctx.ctx_norm[l] = L.ctx_norm;
      st.ctx.k_norm_ctx[l] = L.k_norm_ctx;
      st.ctx.k_cache[l] = kc;
      st.ctx.v_cache[l] = vc;
      st.ctx.fp8[l] = !is_local(l) && GLOBAL_FP8;
      st.ctx.ctx_pos[l] = cpos;
      st.ctx.size[l] = is_local(l) ? kd::WINDOW : ctx_;
    }
  }

  for (int b = 1024; b < ctx_; b *= 2) buckets_.push_back(b);
  buckets_.push_back(ctx_);

  reset_all();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

BatchDrafter::~BatchDrafter() {
  for (auto& [k, g] : draft_graphs_) cudaGraphExecDestroy(g);
  for (auto& [k, g] : step_graphs_) cudaGraphExecDestroy(g);
}

void BatchDrafter::load(const std::filesystem::path& model_dir) {
  const auto sf = model_dir / "mtp.safetensors";
  const auto gg = model_dir / "dspark.gguf";
  if (std::filesystem::exists(sf)) load_safetensors(sf);
  else if (std::filesystem::exists(gg)) load_gguf(gg);
  else throw std::runtime_error("BatchDrafter::load: neither mtp.safetensors nor dspark.gguf found in " + model_dir.string());
}

void BatchDrafter::load_safetensors(const std::filesystem::path& path) {
  SafetensorsFile f(path);
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

  {
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

  {
    auto kv = [](int l, const char* x) { return "layers." + std::to_string(l) + ".self_attn." + x + "_proj_ctx.weight"; };
    wkv_all_ = fp8({{kv(0, "k"), kd::KV_DIM}, {kv(0, "v"), kd::KV_DIM}, {kv(1, "k"), kd::KV_DIM},
                    {kv(1, "v"), kd::KV_DIM}, {kv(2, "k"), kd::KV_DIM}, {kv(2, "v"), kd::KV_DIM}});
  }
  for (int l = 0; l < 3; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    const std::string a = p + "self_attn.";
    DraftLayerWeights& L = layers_[l];
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

  fold_w_ = static_cast<ActT*>(alloc(std::size_t(V_) * kd::RANK * 2));
  fold_b_ = static_cast<ActT*>(alloc(std::size_t(V_) * 2));
  DeviceBuffer hw_buf(std::size_t(h) * kd::RANK * 2);
  DeviceBuffer hb_buf(std::size_t(h) * 2);
  CUDA_CHECK(cudaMemcpyAsync(hw_buf.p, src(mh + "hidden_out.weight", {h, kd::RANK}), std::size_t(h) * kd::RANK * 2,
                             cudaMemcpyDeviceToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(hb_buf.p, src(mh + "hidden_out.bias", {h}), std::size_t(h) * 2,
                             cudaMemcpyDeviceToDevice, stream_));
  kd::fold_head_shards(head_shards_, static_cast<const ActT*>(hw_buf.p), static_cast<const ActT*>(hb_buf.p), fold_w_,
                       fold_b_, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void BatchDrafter::load_gguf(const std::filesystem::path& path) {
  const GgufFile f(path);
  auto req_str = [&](std::string_view key, std::string_view expect) {
    const auto v = f.get_string(key);
    if (!v || *v != expect)
      throw std::runtime_error("dspark.gguf: " + std::string(key) + " mismatch");
  };
  auto req_i = [&](std::string_view key, std::int64_t expect) {
    const auto v = f.get_int(key);
    if (!v || *v != expect)
      throw std::runtime_error("dspark.gguf: " + std::string(key) + " mismatch");
  };
  auto req_f = [&](std::string_view key, double expect, double tol) {
    const auto v = f.get_float(key);
    if (!v || std::fabs(*v - expect) > tol)
      throw std::runtime_error("dspark.gguf: " + std::string(key) + " mismatch");
  };
  req_str("general.architecture", "dflash");
  if (!f.find("dspark_hidden_out.weight"))
    throw std::runtime_error("dspark.gguf: missing dspark_hidden_out.weight");
  req_i("dflash.block_size", BLOCK);
  req_i("dflash.embedding_length", H);
  req_i("dflash.block_count", 3);
  req_i("dflash.attention.head_count", kd::HEADS);
  req_i("dflash.attention.head_count_kv", kd::KV_HEADS);
  req_i("dflash.attention.key_length", kd::HEAD_DIM);
  req_i("dflash.attention.sliding_window", kd::WINDOW);
  req_f("dflash.rope.freq_base", double(kd::ROPE_THETA), 1.0);
  req_f("dflash.attention.layer_norm_rms_epsilon", double(kd::EPS), 1e-9);
  {
    const auto tl = f.get_int_array("dflash.target_layers");
    if (!tl || *tl != std::vector<std::int64_t>{5, 23, 41, 48})
      throw std::runtime_error("dspark.gguf: unexpected dflash.target_layers");
  }
  {
    const auto pat = f.get_int_array("dflash.attention.sliding_window_pattern");
    if (!pat || *pat != std::vector<std::int64_t>{1, 0, 1})
      throw std::runtime_error("dspark.gguf: unexpected sliding_window_pattern");
  }

  auto gguf_name = [](const std::string& n) -> std::string {
    static const std::map<std::string, std::string> top = {
        {"fuse_weight", "dspark_fuse.weight"},
        {"mask_embed", "mask_embd.weight"},
        {"norm.weight", "output_norm.weight"},
        {"markov_head.dflare_norm.weight", "dspark_norm.weight"},
        {"markov_head.markov_w1.weight", "markov_w1.weight"},
        {"markov_head.rnn_proj.weight", "joint_proj.weight"},
        {"markov_head.rnn_proj.bias", "joint_proj.bias"},
        {"markov_head.hidden_out.weight", "dspark_hidden_out.weight"},
        {"markov_head.hidden_out.bias", "dspark_hidden_out.bias"}};
    static const std::map<std::string, std::string> layer = {
        {"input_layernorm.weight", "attn_norm.weight"},
        {"post_attention_layernorm.weight", "ffn_norm.weight"},
        {"dflare_norm.weight", "dflare_norm.weight"},
        {"self_attn.q_norm.weight", "attn_q_norm.weight"},
        {"self_attn.k_norm.weight", "attn_k_norm.weight"},
        {"self_attn.k_norm_ctx.weight", "attn_k_ctx_norm.weight"},
        {"self_attn.q_proj.weight", "attn_q.weight"},
        {"self_attn.k_proj.weight", "attn_k.weight"},
        {"self_attn.v_proj.weight", "attn_v.weight"},
        {"self_attn.o_proj.weight", "attn_output.weight"},
        {"self_attn.k_proj_ctx.weight", "attn_k_ctx.weight"},
        {"self_attn.v_proj_ctx.weight", "attn_v_ctx.weight"},
        {"mlp.gate_proj.weight", "ffn_gate.weight"},
        {"mlp.up_proj.weight", "ffn_up.weight"},
        {"mlp.down_proj.weight", "ffn_down.weight"}};
    if (const auto it = top.find(n); it != top.end()) return it->second;
    if (n.rfind("layers.", 0) == 0) {
      const std::size_t dot = n.find('.', 7);
      if (dot != std::string::npos) {
        if (const auto it = layer.find(n.substr(dot + 1)); it != layer.end())
          return "blk." + n.substr(7, dot - 7) + "." + it->second;
      }
    }
    throw std::runtime_error("dspark.gguf: no GGUF tensor mapping for " + n);
  };

  auto get_t = [&](const std::string& sf_name, const std::vector<std::int64_t>& sf_shape) -> const GgufTensor& {
    const std::string gn = gguf_name(sf_name);
    const GgufTensor& t = f.at(gn);
    bool ok = t.ne.size() == sf_shape.size();
    for (std::size_t i = 0; ok && i < sf_shape.size(); ++i) ok = t.ne[sf_shape.size() - 1 - i] == sf_shape[i];
    if (!ok) throw std::runtime_error("dspark.gguf: unexpected shape for " + gn);
    return t;
  };

  auto upload_act = [&](const GgufTensor& t, ActT* dst) {
    const std::size_t n = std::size_t(t.numel());
    if (t.type == GgmlType::BF16) {
      CUDA_CHECK(cudaMemcpyAsync(dst, t.data, n * 2, cudaMemcpyHostToDevice, stream_));
      bf16_to_act_inplace(dst, n, stream_);
    } else if (t.type == GgmlType::F32) {
      std::vector<ActT> host(n);
      const auto* fp = reinterpret_cast<const float*>(t.data);
      for (std::size_t i = 0; i < n; ++i) host[i] = act::host_from_float(fp[i]);
      CUDA_CHECK(cudaMemcpyAsync(dst, host.data(), n * 2, cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    } else {
      std::vector<float> tmp(n);
      ggml_dequantize(t.type, t.data, t.numel(), tmp.data());
      std::vector<ActT> host(n);
      for (std::size_t i = 0; i < n; ++i) host[i] = act::host_from_float(tmp[i]);
      CUDA_CHECK(cudaMemcpyAsync(dst, host.data(), n * 2, cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
  };

  auto persist = [&](std::initializer_list<std::pair<std::string, std::vector<std::int64_t>>> parts) {
    std::size_t total_elems = 0;
    for (const auto& [name, shape] : parts) total_elems += std::size_t(get_t(name, shape).numel());
    auto* dst = static_cast<ActT*>(alloc(total_elems * 2));
    std::size_t off = 0;
    for (const auto& [name, shape] : parts) {
      const GgufTensor& t = get_t(name, shape);
      upload_act(t, dst + off);
      off += std::size_t(t.numel());
    }
    return dst;
  };

  const std::int64_t h = H;
  const GgufTensor& g0 = f.at("blk.0.ffn_gate.weight");
  if (g0.ne.size() != 2 || g0.ne[0] != h) throw std::runtime_error("dspark.gguf: unexpected blk.0.ffn_gate.weight");
  inter_ = int(g0.ne[1]);
  req_i("dflash.feed_forward_length", inter_);
  const std::int64_t I = inter_;

  {
    const GgufTensor& t = get_t("fuse_weight", {4, 4});
    std::vector<float> fw(16);
    if (t.type == GgmlType::F32) std::memcpy(fw.data(), t.data, 16 * 4);
    else ggml_dequantize(t.type, t.data, 16, fw.data());
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

  {
    DeviceBuffer cat(std::size_t(2) * I * h * 2);
    auto fp8 = [&](std::initializer_list<std::pair<std::string, std::int64_t>> parts, std::int64_t K = H) {
      std::int64_t rows = 0;
      for (const auto& [name, r] : parts) {
        const GgufTensor& t = get_t(name, {r, K});
        upload_act(t, static_cast<ActT*>(cat.p) + rows * K);
        rows += r;
      }
      void* q = alloc(std::size_t(rows * K));
      auto* scale = static_cast<float*>(alloc(std::size_t(rows) * 4));
      kdense::fp8_rows(static_cast<const ActT*>(cat.p), int(rows), int(K), q, scale, stream_);
      return kdense::Matrix{q, scale, int(rows)};
    };

    auto kv = [](int l, const char* x) { return "layers." + std::to_string(l) + ".self_attn." + x + "_proj_ctx.weight"; };
    wkv_all_ = fp8({{kv(0, "k"), kd::KV_DIM}, {kv(0, "v"), kd::KV_DIM}, {kv(1, "k"), kd::KV_DIM},
                    {kv(1, "v"), kd::KV_DIM}, {kv(2, "k"), kd::KV_DIM}, {kv(2, "v"), kd::KV_DIM}});
    for (int l = 0; l < 3; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".";
      const std::string a = p + "self_attn.";
      DraftLayerWeights& L = layers_[l];
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
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }  // cat freed before allocating remaining weights & fold_w

  final_norm_ = persist({{"norm.weight", {h}}});
  mask_embed_ = persist({{"mask_embed", {h}}});
  const std::string mh = "markov_head.";
  const std::int64_t w1_rows = f.at("markov_w1.weight").ne.at(1);
  if (w1_rows < V_) throw std::runtime_error("dspark.gguf: markov_w1 has fewer rows than the vocabulary");
  w1_ = persist({{mh + "markov_w1.weight", {w1_rows, kd::RANK}}});
  rnn_w_ = persist({{mh + "rnn_proj.weight", {kd::RNN_OUT, kd::RNN_IN}}});
  rnn_b_ = persist({{mh + "rnn_proj.bias", {kd::RNN_OUT}}});
  f_norm_ = persist({{mh + "dflare_norm.weight", {h}}});

  fold_w_ = static_cast<ActT*>(alloc(std::size_t(V_) * kd::RANK * 2));
  fold_b_ = static_cast<ActT*>(alloc(std::size_t(V_) * 2));
  {
    DeviceBuffer hw_buf(std::size_t(h) * kd::RANK * 2);
    DeviceBuffer hb_buf(std::size_t(h) * 2);
    upload_act(get_t(mh + "hidden_out.weight", {h, kd::RANK}), static_cast<ActT*>(hw_buf.p));
    upload_act(get_t(mh + "hidden_out.bias", {h}), static_cast<ActT*>(hb_buf.p));
    kd::fold_head_shards(head_shards_, static_cast<const ActT*>(hw_buf.p), static_cast<const ActT*>(hb_buf.p), fold_w_,
                         fold_b_, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }
}

void BatchDrafter::reset_slot(int slot) {
  if (slot < 0 || slot >= max_slots_) throw std::invalid_argument("BatchDrafter::reset_slot: slot out of range");
  SlotState& st = slots_[slot];
  for (int l = 0; l < 3; ++l) {
    if (st.ctx.ctx_pos[l] != nullptr)
      CUDA_CHECK(cudaMemsetAsync(st.ctx.ctx_pos[l], 0xff, kd::WINDOW * 4, stream_));
  }
  CUDA_CHECK(cudaMemsetAsync(st.f_rnn, 0, H * 2, stream_));
  st.length = 0;
}

void BatchDrafter::reset_all() {
  for (int s = 0; s < max_slots_; ++s) reset_slot(s);
  reset_context_shortlist();
}

void BatchDrafter::reset_context_shortlist() { set_context_rows(0); }

std::size_t BatchDrafter::set_context_rows(int rows) {
  if (rows >= short_E_) return 0;
  for (std::size_t i = std::size_t(rows); i < short_extra_.size(); ++i) short_in_[short_extra_[i]] = 0;
  short_extra_.resize(std::size_t(rows));
  short_E_ = rows;
  const int seg[3] = {0, short_K_ + rows, 0};
  CUDA_CHECK(cudaMemcpyAsync(seg_short_, seg, sizeof seg, cudaMemcpyHostToDevice, stream_));
  const int total_gam = max_slots_ * draft_len_;
  for (int i = 0; i < total_gam; ++i)
    kd::fill(base_ + std::size_t(i) * short_head_.rows + short_K_ + rows,
             std::size_t(short_head_.rows - short_K_ - rows), -INFINITY, stream_);
  return sizeof seg;
}

std::size_t BatchDrafter::note_tokens(std::span<const std::int32_t> ids) {
  std::vector<std::int32_t> add;
  for (std::int32_t t : ids)
    if (t >= 0 && t < V_ && !short_in_[t] && short_E_ + int(add.size()) < CONTEXT_ROWS) {
      short_in_[t] = 1;
      add.push_back(t);
    }
  if (add.empty()) return 0;
  const int row = short_K_ + short_E_;
  const std::size_t row_bytes =
      short_head_.blk_off ? std::size_t(H) : (head_.scale ? std::size_t(H) : std::size_t(H) * 2);
  const std::size_t ids_bytes = add.size() * 4;
  CUDA_CHECK(cudaMemcpyAsync(short_ids_, add.data(), ids_bytes, cudaMemcpyHostToDevice, stream_));
  kd::gather_head_rows_shards(head_shards_, row_bytes, short_ids_, int(add.size()), row,
                              const_cast<void*>(short_head_.w), const_cast<float*>(short_head_.scale),
                              short_q6_scales_, short_map_, stream_);
  short_extra_.insert(short_extra_.end(), add.begin(), add.end());
  short_E_ += int(add.size());
  const int seg[3] = {0, short_K_ + short_E_, 0};
  CUDA_CHECK(cudaMemcpyAsync(seg_short_, seg, sizeof seg, cudaMemcpyHostToDevice, stream_));
  return ids_bytes + sizeof seg;
}

void BatchDrafter::commit_slot(int slot, const ActT* feats, int n, int start) {
  if (slot < 0 || slot >= max_slots_) throw std::invalid_argument("BatchDrafter::commit_slot: slot out of range");
  if (n <= 0) return;
  SlotState& st = slots_[slot];
  for (int r0 = 0; r0 < n; r0 += PREFILL_CHUNK_STRIDE) {
    const int k = std::min(PREFILL_CHUNK_STRIDE, n - r0);
    const int stride = (k <= BLOCK) ? BLOCK : PREFILL_CHUNK_STRIDE;
    const bool last = (r0 + k == n);
    kd::prefill_commit_mix(feats + std::size_t(r0) * 4 * H, k, stride, last, alpha_, st.ctx, f_norm_, commit_xin_,
                           st.f_rnn, stream_);
    kdense::proj(commit_xin_, stride, wkv_all_, seg_kv_, 3, commit_kv_, true, stream_);
    const int keep_from_local = std::max(0, (n - kd::WINDOW) - r0);
    kd::prefill_commit_store(commit_kv_, k, start + r0, keep_from_local, st.ctx, stream_);
  }
  st.length = start + n;
}

void BatchDrafter::upload_active_desc(std::span<const int> active_slots) {
  const int S = int(active_slots.size());
  if (S < 1 || S > max_slots_) throw std::invalid_argument("BatchDrafter::upload_active_desc: bad S");
  kd::BatchSlotDesc desc{};
  desc.count = S;
  for (int i = 0; i < S; ++i) {
    const int phys = active_slots[i];
    if (phys < 0 || phys >= max_slots_)
      throw std::invalid_argument("BatchDrafter::upload_active_desc: slot out of range");
    const SlotState& st = slots_[phys];
    desc.params[i] = st.params;
    desc.ctx[i] = st.ctx;
    desc.f_rnn[i] = st.f_rnn;
    desc.state[i] = st.state;
    desc.latent[i] = st.latent;
    desc.picks[i] = st.picks;
    desc.tokens[i] = st.tokens;
    desc.conf[i] = st.conf;
    desc.qstats[i] = nullptr;
  }
  CUDA_CHECK(cudaMemcpyAsync(slots_dev_, &desc, sizeof(desc), cudaMemcpyHostToDevice, stream_));
}

void BatchDrafter::run_draft(int S, int bucket) {
  const int M_blk = S * BLOCK;
  const int M_gam = S * draft_len_;
  kd::embed_block_batch(static_cast<const ActT*>(w_.embed.data), static_cast<const float*>(w_.embed_norm.data),
                        mask_embed_, layers_[0].in_norm, slots_dev_, S, res_, xn_, stream_, w_.embed_q4(),
                        w_.embed_q8());
  for (int l = 0; l < 3; ++l) {
    const DraftLayerWeights& L = layers_[l];
    const bool local = is_local(l);
    kdense::proj(xn_, M_blk, L.qkv, seg_qkv_, 1, qkv_out_, true, stream_);
    kd::qkv_rope_batch(qkv_out_, L.q_norm, L.k_norm, slots_dev_, S, l, q_, local, ctx_, stream_);
    for (int s = 0; s < S; ++s) {
      kd::block_attention_slot(q_ + std::size_t(s) * BLOCK * kd::Q_DIM, slots_dev_, s, l, local ? LOCAL_SLOTS : bucket,
                               local, !local && GLOBAL_FP8, attn_ws_, attn_o_ + std::size_t(s) * BLOCK * kd::Q_DIM,
                               stream_);
    }
    kdense::gemv(attn_o_, M_blk, kd::Q_DIM, L.wo, delta_, true, stream_);
    kd::add_rmsnorm(res_, delta_, L.post_norm, xn_, M_blk, stream_);
    kdense::proj(xn_, M_blk, L.gate_up, seg_gate_up_, 1, gu_out_, true, stream_);
    kd::silu_mul(gu_out_, M_blk, inter_, act_, stream_);
    kdense::gemv(act_, M_blk, inter_, L.w_down, delta_, true, stream_);
    kd::add_rmsnorm(res_, delta_, l < 2 ? layers_[l + 1].in_norm : final_norm_, xn_, M_blk, stream_);
  }
  kd::gather_draft_h(xn_, S, draft_len_, h_, stream_);
  kdense::proj(h_, M_gam, short_head_, seg_short_, 1, base_, true, stream_);
  for (int s = 0; s < S; ++s) {
    kd::topk_candidates(base_ + std::size_t(s) * draft_len_ * short_head_.rows, draft_len_, short_head_.rows, short_map_,
                        cand_val_ + std::size_t(s) * draft_len_ * kd::CANDIDATES,
                        cand_idx_ + std::size_t(s) * draft_len_ * kd::CANDIDATES, topk_ws_, stream_);
  }
  kd::markov_pre_batch(rnn_w_, h_, slots_dev_, S, draft_len_, part_, stream_);
  for (int i = 0; i < draft_len_; ++i) {
    kd::markov_step_batch(i, rnn_w_, rnn_b_, w1_, part_, cand_idx_, slots_dev_, S, draft_len_, stream_);
    kd::markov_logits_batch(i, fold_w_, fold_b_, cand_val_, cand_idx_, slots_dev_, S, draft_len_, cand_logits_,
                            stream_);
  }
  kd::markov_finish_batch(cand_idx_, cand_logits_, slots_dev_, S, draft_len_, stream_);
}

void BatchDrafter::run_step(int S, int bucket) {
  kd::commit_mix_batch(slots_dev_, S, max_commit_, alpha_, f_norm_, commit_xin_, stream_);
  kdense::proj(commit_xin_, S * max_commit_, wkv_all_, seg_kv_, 3, commit_kv_, true, stream_);
  kd::commit_store_batch(commit_kv_, slots_dev_, S, max_commit_, stream_);
  run_draft(S, bucket);
}

int BatchDrafter::bucket_for(int max_P) const {
  for (int b : buckets_)
    if (b >= max_P + BLOCK) return b;
  return ctx_;
}

void BatchDrafter::capture() {
  for (auto& [k, g] : draft_graphs_) cudaGraphExecDestroy(g);
  for (auto& [k, g] : step_graphs_) cudaGraphExecDestroy(g);
  draft_graphs_.clear();
  step_graphs_.clear();

  std::vector<int> all_slots(max_slots_);
  for (int s = 0; s < max_slots_; ++s) {
    all_slots[s] = s;
    kd::set_commit(slots_[s].params, warm_feats_, 1, 0, stream_);
    kd::set_draft_inputs(slots_[s].params, 0, 0, 0.f, stream_);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  for (int S = 1; S <= max_slots_; ++S) {
    upload_active_desc(std::span<const int>(all_slots.data(), std::size_t(S)));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    for (int b : buckets_) {
      run_draft(S, b);  // warmup
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      capture_begin(stream_);
      run_draft(S, b);
      draft_graphs_[{S, b}] = capture_end(stream_);

      run_step(S, b);  // warmup
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      capture_begin(stream_);
      run_step(S, b);
      step_graphs_[{S, b}] = capture_end(stream_);
    }
  }
  reset_all();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void BatchDrafter::draft_slots(std::span<const int> active_slots, std::span<const std::int32_t> anchors,
                               std::span<const int> positions, float temperature) {
  const int S = int(active_slots.size());
  if (S < 1 || S > max_slots_ || int(anchors.size()) != S || int(positions.size()) != S)
    throw std::invalid_argument("BatchDrafter::draft_slots: mismatched sizes");
  int max_P = 0;
  for (int i = 0; i < S; ++i) {
    if (anchors[i] < 0 || anchors[i] >= V_) throw std::invalid_argument("BatchDrafter::draft_slots: anchor out of range");
    kd::set_draft_inputs(slots_[active_slots[i]].params, anchors[i], positions[i], temperature, stream_);
    max_P = std::max(max_P, positions[i]);
  }
  upload_active_desc(active_slots);
  const int b = bucket_for(max_P);
  if (const auto it = draft_graphs_.find({S, b}); it != draft_graphs_.end()) {
    CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  } else {
    run_draft(S, b);
  }
}

void BatchDrafter::step_commit_and_draft(std::span<const int> active_slots, int max_P, bool desc_already_uploaded) {
  const int S = int(active_slots.size());
  if (S < 1 || S > max_slots_) throw std::invalid_argument("BatchDrafter::step_commit_and_draft: bad S");
  if (!desc_already_uploaded) upload_active_desc(active_slots);
  const int b = bucket_for(max_P);
  if (const auto it = step_graphs_.find({S, b}); it != step_graphs_.end()) {
    CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  } else {
    run_step(S, b);
  }
}

}  // namespace ingine::model

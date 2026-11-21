#include "model/drafter.h"

#include "core/device.h"
#include "core/safetensors.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ingine::model {

namespace kd = kernels::draft;
namespace kdense = kernels::dense;

namespace {

constexpr int H = kd::H;
constexpr int LOCAL_SLOTS = kd::WINDOW + kd::BLOCK;  // local ring + the drafted block's rows
constexpr int COMMIT_PIECE = 1024;                   // eager commit rows per piece

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

Drafter::Drafter(const std::filesystem::path& model_dir, const Weights& w, int max_ctx, cudaStream_t stream)
    : w_(w), stream_(stream), ctx_(std::min(max_ctx, DRAFT_CTX)) {
  if (max_ctx <= 0) throw std::invalid_argument("Drafter: max_ctx must be positive");
  head_ = {w.unembed.w.data, w.unembed.fp8() ? static_cast<const float*>(w.unembed.scale.data) : nullptr,
           w.unembed.rows()};
  V_ = w.unembed.rows();
  load(model_dir);

  auto bf16s = [&](std::size_t n) { return static_cast<__nv_bfloat16*>(alloc(n * 2)); };
  auto f32s = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto i32s = [&](std::size_t n) { return static_cast<std::int32_t*>(alloc(n * 4)); };

  params_ = static_cast<kd::DraftParams*>(alloc(sizeof(kd::DraftParams)));
  CUDA_CHECK(cudaMemsetAsync(params_, 0, sizeof(kd::DraftParams), stream_));
  res_ = bf16s(BLOCK * H);
  xn_ = bf16s(BLOCK * H);
  q_ = bf16s(BLOCK * kd::Q_DIM);
  attn_o_ = bf16s(BLOCK * kd::Q_DIM);
  delta_ = bf16s(BLOCK * H);
  act_ = bf16s(std::size_t(BLOCK) * inter_);
  f_rnn_ = bf16s(H);
  state_ = bf16s((GAMMA + 1) * kd::RANK);
  latent_ = bf16s(kd::RANK);
  qkv_out_ = f32s(BLOCK * kd::QKV_ROWS);
  gu_out_ = f32s(std::size_t(BLOCK) * 2 * inter_);
  base_ = f32s(std::size_t(GAMMA) * V_);
  attn_ws_ = f32s(std::max(kd::attention_workspace_floats(LOCAL_SLOTS), kd::attention_workspace_floats(ctx_)));
  cand_val_ = f32s(GAMMA * kd::CANDIDATES);
  cand_logits_ = f32s(GAMMA * kd::CANDIDATES);
  part_ = f32s(GAMMA * kd::RNN_OUT);
  cand_idx_ = i32s(GAMMA * kd::CANDIDATES);
  prev_cand_ = i32s(GAMMA * kd::CANDIDATES);
  picks_ = static_cast<unsigned long long*>(alloc(GAMMA * sizeof(unsigned long long)));
  draft_logits_ = f32s(std::size_t(GAMMA) * V_);
  conf_ = f32s(GAMMA);
  tokens_ = i32s(GAMMA);
  commit_feats_ = bf16s(std::size_t(BLOCK) * 4 * H);
  ctx_c_ = bf16s(std::size_t(COMMIT_PIECE) * H);
  ctx_kv_ = bf16s(std::size_t(COMMIT_PIECE) * 2 * kd::KV_DIM);
  CUDA_CHECK(cudaMemsetAsync(state_, 0, (GAMMA + 1) * kd::RANK * 2, stream_));  // state row 0 stays zero
  CUDA_CHECK(cudaMemsetAsync(prev_cand_, 0, GAMMA * kd::CANDIDATES * 4, stream_));
  CUDA_CHECK(cudaMemsetAsync(f_rnn_, 0, H * 2, stream_));
  kd::fill(draft_logits_, std::size_t(GAMMA) * V_, -INFINITY, stream_);

  // one-segment tables (rows [0, R) from input 0) for the dense GEMV
  auto segment = [&](int rows) {
    int* d = i32s(3);
    const int host[3] = {0, rows, 0};
    CUDA_CHECK(cudaMemcpyAsync(d, host, sizeof(host), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return d;
  };
  seg_qkv_ = segment(kd::QKV_ROWS);
  seg_gate_up_ = segment(2 * inter_);
  seg_head_ = segment(V_);

  for (int l = 0; l < 3; ++l) {
    DraftLayer& L = layers_[l];
    const int slots = is_local(l) ? LOCAL_SLOTS : ctx_;
    L.k_ctx = bf16s(std::size_t(slots) * kd::KV_DIM);
    L.v_ctx = bf16s(std::size_t(slots) * kd::KV_DIM);
    CUDA_CHECK(cudaMemsetAsync(L.k_ctx, 0, std::size_t(slots) * kd::KV_DIM * 2, stream_));
    CUDA_CHECK(cudaMemsetAsync(L.v_ctx, 0, std::size_t(slots) * kd::KV_DIM * 2, stream_));
    L.ctx_pos = is_local(l) ? i32s(kd::WINDOW) : nullptr;
  }

  accept_.target = i32s(BLOCK);
  accept_.probs = f32s(std::size_t(BLOCK) * V_);
  accept_.resid = f32s(V_);
  accept_.uniforms = f32s(GAMMA);
  accept_.ws = f32s(2 * GAMMA + 1);
  accept_.result = static_cast<kd::AcceptResult*>(alloc(sizeof(kd::AcceptResult)));
  accept_.host = pinned_.data();

  reset();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

Drafter::~Drafter() {
  for (auto& [b, g] : draft_graphs_) cudaGraphExecDestroy(g);
  for (auto& [n, g] : commit_graphs_) cudaGraphExecDestroy(g);
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
    return reinterpret_cast<__nv_bfloat16*>(dst);
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
    __nv_bfloat16 host[16];
    for (int r = 0; r < 4; ++r) {
      float m = -INFINITY;
      for (int c = 0; c < 4; ++c) m = std::max(m, fw[r * 4 + c]);
      float e[4], s = 0.f;
      for (int c = 0; c < 4; ++c) s += (e[c] = std::exp(fw[r * 4 + c] - m));
      for (int c = 0; c < 4; ++c) host[r * 4 + c] = __float2bfloat16_rn(e[c] / s);
    }
    auto* d = static_cast<__nv_bfloat16*>(alloc(sizeof(host)));
    CUDA_CHECK(cudaMemcpyAsync(d, host, sizeof(host), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    alpha_ = d;
  }

  // FP8 rows of a row-concatenation (kernels::dense::fp8_rows, as dense.fp8_rows in dspark)
  DeviceBuffer cat(std::size_t(2) * I * h * 2);
  auto fp8 = [&](std::initializer_list<std::pair<std::string, std::int64_t>> parts) {
    std::int64_t rows = 0;
    for (const auto& [name, r] : parts) {
      const std::byte* s = src(name, {r, h});
      CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(cat.p) + rows * h * 2, s, std::size_t(r * h * 2),
                                 cudaMemcpyDeviceToDevice, stream_));
      rows += r;
    }
    void* q = alloc(std::size_t(rows * h));
    auto* scale = static_cast<float*>(alloc(std::size_t(rows) * 4));
    kdense::fp8_rows(static_cast<const __nv_bfloat16*>(cat.p), int(rows), H, q, scale, stream_);
    return kdense::Matrix{q, scale, int(rows)};
  };

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
    L.wo = persist({{a + "o_proj.weight", {h, kd::Q_DIM}}});
    L.w_down = persist({{p + "mlp.down_proj.weight", {h, I}}});
    L.wkv_ctx = persist({{a + "k_proj_ctx.weight", {kd::KV_DIM, h}}, {a + "v_proj_ctx.weight", {kd::KV_DIM, h}}});
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
  fold_w_ = static_cast<__nv_bfloat16*>(alloc(std::size_t(V_) * kd::RANK * 2));
  fold_b_ = static_cast<__nv_bfloat16*>(alloc(std::size_t(V_) * 2));
  const __nv_bfloat16* hw = persist({{mh + "hidden_out.weight", {h, kd::RANK}}});
  const __nv_bfloat16* hb = persist({{mh + "hidden_out.bias", {h}}});
  kd::fold_head(w_.unembed.w.data, head_.scale, V_, hw, hb, fold_w_, fold_b_, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));  // raw and cat are freed on return
}

void Drafter::reset() {
  for (const DraftLayer& L : layers_)
    if (L.ctx_pos != nullptr) CUDA_CHECK(cudaMemsetAsync(L.ctx_pos, 0xff, kd::WINDOW * 4, stream_));
  length_ = 0;
}

void Drafter::set_seed(std::uint64_t seed) { kd::set_seed(params_, seed, stream_); }

void Drafter::run_commit(const __nv_bfloat16* feats, int n, int row0, bool last_piece) {
  for (int l = 0; l < 3; ++l) {
    const DraftLayer& L = layers_[l];
    kd::mix_rmsnorm(feats, n, alpha_ + 4 * l, L.ctx_norm, ctx_c_, stream_);
    gemm_.run(ctx_c_, n, L.wkv_ctx, 2 * kd::KV_DIM, H, ctx_kv_, stream_);
    const int size = is_local(l) ? kd::WINDOW : ctx_;
    kd::ctx_kv_store(ctx_kv_, n, params_, row0, n - std::min(n, size), L.k_norm_ctx, L.k_ctx, L.v_ctx, size,
                     L.ctx_pos, stream_);
  }
  if (last_piece) kd::mix_rmsnorm(feats + std::size_t(n - 1) * 4 * H, 1, alpha_ + 12, f_norm_, f_rnn_, stream_);
}

void Drafter::commit(const Tensor& feats, int start) {
  if (feats.dtype != DType::BF16 || feats.ne[0] != H || feats.ne[1] != 4 || !feats.is_contiguous())
    throw std::invalid_argument("Drafter::commit: feats must be contiguous bf16 [n, 4, 2304]");
  commit(static_cast<const __nv_bfloat16*>(feats.data), feats.ne[2] * feats.ne[3], start);
}

void Drafter::commit(const __nv_bfloat16* feats, int n, int start) {
  if (n <= 0) return;
  kd::set_commit_start(params_, start, stream_);
  if (auto it = commit_graphs_.find(n); it != commit_graphs_.end()) {
    CUDA_CHECK(cudaMemcpyAsync(commit_feats_, feats, std::size_t(n) * 4 * H * 2, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  } else {
    for (int r0 = 0; r0 < n; r0 += COMMIT_PIECE) {
      const int k = std::min(COMMIT_PIECE, n - r0);
      run_commit(feats + std::size_t(r0) * 4 * H, k, r0, r0 + k == n);
    }
  }
  length_ = start + n;
}

void Drafter::run_draft(int bucket) {
  kd::embed_block(static_cast<const __nv_bfloat16*>(w_.embed.data), static_cast<const float*>(w_.embed_norm.data),
                  mask_embed_, layers_[0].in_norm, params_, res_, xn_, picks_, stream_);
  for (int l = 0; l < 3; ++l) {
    const DraftLayer& L = layers_[l];
    const bool local = is_local(l);
    kdense::proj(xn_, BLOCK, L.qkv, seg_qkv_, 1, qkv_out_, true, stream_);
    kd::qkv_rope(qkv_out_, L.q_norm, L.k_norm, params_, q_, L.k_ctx, L.v_ctx, local, ctx_, stream_);
    kd::block_attention(q_, L.k_ctx, L.v_ctx, local ? LOCAL_SLOTS : bucket, L.ctx_pos, params_, attn_ws_, attn_o_,
                        stream_);
    gemm_.run(attn_o_, BLOCK, L.wo, H, kd::Q_DIM, delta_, stream_);
    kd::add_rmsnorm(res_, delta_, L.post_norm, xn_, BLOCK, stream_);
    kdense::proj(xn_, BLOCK, L.gate_up, seg_gate_up_, 1, gu_out_, true, stream_);
    kd::silu_mul(gu_out_, BLOCK, inter_, act_, stream_);
    gemm_.run(act_, BLOCK, L.w_down, H, inter_, delta_, stream_);
    kd::add_rmsnorm(res_, delta_, l < 2 ? layers_[l + 1].in_norm : final_norm_, xn_, BLOCK, stream_);
  }
  const __nv_bfloat16* h = xn_ + H;  // rows 1..7
  kdense::proj(h, GAMMA, head_, seg_head_, 1, base_, true, stream_);
  kd::topk_candidates(base_, GAMMA, V_, cand_val_, cand_idx_, stream_);
  kd::markov_pre(rnn_w_, h, f_rnn_, part_, stream_);
  for (int i = 0; i < GAMMA; ++i) {
    kd::markov_step(i, rnn_w_, rnn_b_, w1_, part_, picks_, cand_idx_, params_, state_, latent_, stream_);
    kd::markov_logits(i, fold_w_, fold_b_, cand_val_, cand_idx_, latent_, params_, cand_logits_, picks_, stream_);
  }
  kd::markov_finish(cand_idx_, cand_logits_, picks_, prev_cand_, draft_logits_, V_, tokens_, conf_, params_, stream_);
}

// the smallest bucket holding the block; the whole ring once the context reaches it (and without graphs)
int Drafter::bucket_for(int P) const {
  for (const auto& [b, g] : draft_graphs_)
    if (b >= P + BLOCK) return b;
  return ctx_;
}

Drafter::Draft Drafter::draft(int anchor, int P, float temperature) {
  if (anchor < 0 || anchor >= V_) throw std::invalid_argument("Drafter::draft: anchor out of range");
  kd::set_draft_inputs(params_, anchor, P, temperature, stream_);
  const int bucket = bucket_for(P);
  if (auto it = draft_graphs_.find(bucket); it != draft_graphs_.end())
    CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  else
    run_draft(bucket);
  return {tokens_, draft_logits_, conf_};
}

void Drafter::capture() {
  std::vector<int> buckets;
  for (int b = 1024; b < ctx_; b *= 2) buckets.push_back(b);
  buckets.push_back(ctx_);
  // warm-up outside capture (cuBLASLt plans), then capture; the scratch context rows are reset afterwards
  kd::set_draft_inputs(params_, 0, 0, 0.f, stream_);
  kd::set_commit_start(params_, 0, stream_);
  for (int b : buckets) run_draft(b);
  for (int n = 1; n <= BLOCK; ++n) run_commit(commit_feats_, n, 0, true);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (int b : buckets) {
    capture_begin(stream_);
    run_draft(b);
    draft_graphs_[b] = capture_end(stream_);
  }
  for (int n = 1; n <= BLOCK; ++n) {
    capture_begin(stream_);
    run_commit(commit_feats_, n, 0, true);
    commit_graphs_[n] = capture_end(stream_);
  }
  reset();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace ingine::model

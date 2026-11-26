#include "model/batch_drafter.h"

#include "core/act.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "ingine/batch_spec_plan.h"
#include "kernels/draft/shortlist.h"
#include "kernels/prefill/prefill.h"
#include "model/model_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ingine::model {

namespace kd = kernels::draft;
namespace kdense = kernels::dense;
namespace pk = kernels::prefill;

namespace {

constexpr int H = kd::H;
constexpr int LOCAL_SLOTS = kd::WINDOW + kd::BLOCK;
constexpr int COMMIT_PIECE = 1024;
constexpr int kShortChunk = 16384;
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

int choose_draft_ctx(int max_batch, int max_ctx, std::optional<std::size_t> vram_budget_bytes) {
  std::size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  if (vram_budget_bytes.has_value()) free_b = std::min(free_b, *vram_budget_bytes);
  int cap = 4096;
  if (max_batch <= 8 && free_b > (std::size_t(12) << 30)) cap = 65536;
  else if (free_b > (std::size_t(48) << 30)) cap = 16384;
  else if (free_b > (std::size_t(18) << 30)) cap = 8192;
  else if (free_b < (std::size_t(6) << 30)) cap = 2048;
  return std::max(std::min(max_ctx, 1024), std::min(max_ctx, cap));
}

}  // namespace

void* BatchDrafter::alloc(std::size_t bytes) {
  storage_.emplace_back(bytes);
  return storage_.back().p;
}

BatchDrafter::BatchDrafter(const std::filesystem::path& model_dir, const Weights& w, int max_batch, int max_ctx,
                           int max_verify_width, cudaStream_t stream, int shortlist_rows,
                           std::optional<std::size_t> vram_budget_bytes)
    : w_(w),
      stream_(stream),
      max_batch_(max_batch),
      max_verify_width_(max_verify_width),
      ctx_(choose_draft_ctx(max_batch, max_ctx, vram_budget_bytes)),
      draft_meta_host_(std::size_t(max_batch) * sizeof(kd::BatchSlotDraftMeta)),
      commit_meta_host_(std::size_t(max_batch) * sizeof(kd::BatchSlotCommitMeta)) {
  if (max_batch < 1) throw std::invalid_argument("BatchDrafter: max_batch must be >= 1");
  if (max_ctx < 2 * BLOCK) throw std::invalid_argument("BatchDrafter: max_ctx must hold two blocks");
  if (w.embed_q8())
    throw std::invalid_argument("BatchDrafter: Q8_0 embedding tables are not supported (set INGINE_GGUF_Q8_EMBED=0)");
  if (w.unembed.blocked())
    throw std::invalid_argument("BatchDrafter: blocked/Q6_K unembedding tables are not supported (set INGINE_GGUF_Q6_HEAD=0)");
  if (vram_budget_bytes.has_value()) {
    // Header-only inspection (0 GPU bytes — BEFORE pk::Gemm 32 MiB workspace or raw mtp.safetensors upload buffer!)
    // using the exact compile-time production geometry of this build.
    SafetensorsFile mtp_hdr(model_dir / "mtp.safetensors");
    const std::size_t mtp_raw_bytes = std::size_t(mtp_hdr.file_size() - mtp_hdr.data_offset());
    const int inter_hdr = int(mtp_hdr.at("layers.0.mlp.gate_proj.weight").shape.at(0));
    DrafterGeometry geom;
    geom.H = std::size_t(kd::H);
    geom.rank = std::size_t(kd::RANK);
    geom.gamma = std::size_t(kd::GAMMA);
    geom.block = std::size_t(kd::BLOCK);
    geom.q_dim = std::size_t(kd::Q_DIM);
    geom.kv_dim = std::size_t(kd::KV_DIM);
    geom.local_slots = std::size_t(LOCAL_SLOTS);
    geom.global_slot_bytes = GLOBAL_SLOT_BYTES;
    geom.max_slot_extra = std::size_t(kd::kMaxSlotExtra);
    geom.short_chunk = std::size_t(kShortChunk);
    geom.commit_piece = std::size_t(COMMIT_PIECE);
    const DrafterVramPeakEstimate est = estimate_drafter_peak_bytes(
        mtp_raw_bytes, inter_hdr, max_batch, ctx_, w.unembed.rows(), shortlist_rows, max_verify_width, w.unembed.fp8(),
        geom);
    const VramLeaseCheck chk = check_vram_lease_fit(vram_budget_bytes, est.peak_bytes);
    if (!chk.ok) {
      throw std::runtime_error(
          "BatchDrafter: VRAM lease cap exceeded before allocation (peak_need=" +
          std::to_string(est.peak_bytes >> 20) + " MiB [load_peak=" + std::to_string(est.load_peak_bytes >> 20) +
          " MiB, cat=" + std::to_string(est.cat_bytes >> 20) + " MiB, steady=" +
          std::to_string(est.steady_bytes >> 20) + " MiB], remaining=" +
          std::to_string(chk.remaining_bytes >> 20) + " MiB)");
    }
  }
  gemm_ = std::make_unique<pk::Gemm>();
  head_ = {w.unembed.w.data, w.unembed.fp8() ? static_cast<const float*>(w.unembed.scale.data) : nullptr,
           w.unembed.rows()};
  V_ = w.unembed.rows();
  load(model_dir);

  auto bf16s = [&](std::size_t n) { return static_cast<ActT*>(alloc(n * sizeof(ActT))); };
  auto f32s = [&](std::size_t n) { return static_cast<float*>(alloc(n * sizeof(float))); };
  auto i32s = [&](std::size_t n) { return static_cast<std::int32_t*>(alloc(n * sizeof(std::int32_t))); };

  auto segments = [&](std::initializer_list<int> rows) {
    std::vector<int> host;
    int b = 0, input = 0;
    for (int r : rows) {
      host.insert(host.end(), {b, b + r, input++});
      b += r;
    }
    int* d = i32s(host.size());
    CUDA_CHECK(cudaMemcpyAsync(d, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return d;
  };
  seg_qkv_ = segments({kd::QKV_ROWS});
  seg_gate_up_ = segments({2 * inter_});
  seg_kv_ = segments({2 * kd::KV_DIM, 2 * kd::KV_DIM, 2 * kd::KV_DIM});

  // Build IMMUTABLE ranked + special shortlist head [short_K_, H] (shared safely across all tenants/slots).
  {
    short_base_in_.assign(std::size_t(V_), 0);
    int taken = 0;
    for (std::uint32_t t : kd::draft_shortlist_ranks()) {
      if (taken < shortlist_rows && t < std::uint32_t(V_) && !short_base_in_[t]) {
        short_base_in_[t] = 1;
        ++taken;
      }
    }
    for (int t = 199990; t < V_; ++t) short_base_in_[std::size_t(t)] = 1;
    std::vector<std::int32_t> list;
    list.reserve(std::size_t(shortlist_rows) + 256);
    for (int t = 0; t < V_; ++t) {
      if (short_base_in_[std::size_t(t)]) list.push_back(t);
    }
    short_K_ = int(list.size());
    base_stride_ = short_K_ + MAX_SLOT_EXTRA;
    const std::size_t row_bytes = head_.scale ? std::size_t(H) : std::size_t(H) * 2;
    void* q = alloc(std::size_t(short_K_) * row_bytes);
    auto* qs = head_.scale ? f32s(std::size_t(short_K_)) : nullptr;
    short_map_ = i32s(std::size_t(short_K_));
    DeviceBuffer tmp_ids(list.size() * sizeof(std::int32_t));
    CUDA_CHECK(cudaMemcpyAsync(tmp_ids.p, list.data(), list.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream_));
    kd::gather_head_rows(head_.w, head_.scale, row_bytes, static_cast<const std::int32_t*>(tmp_ids.p), short_K_, q, qs,
                         short_map_, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    short_head_ = kdense::Matrix{q, qs, short_K_};
    seg_short_ = segments({short_K_});
  }

  // Per-slot KV caches, ctx_pos, f_rnn, and slot-local extra token ids.
  for (int l = 0; l < 3; ++l) {
    DraftLayer& L = layers_[l];
    L.kv_slot_bytes = is_local(l) ? std::size_t(LOCAL_SLOTS) * kd::KV_DIM * sizeof(ActT)
                                  : std::size_t(ctx_) * GLOBAL_SLOT_BYTES;
    const std::size_t total_bytes = std::size_t(max_batch_) * L.kv_slot_bytes;
    L.k_ctx = alloc(total_bytes);
    L.v_ctx = alloc(total_bytes);
    CUDA_CHECK(cudaMemsetAsync(L.k_ctx, 0, total_bytes, stream_));
    CUDA_CHECK(cudaMemsetAsync(L.v_ctx, 0, total_bytes, stream_));
    L.ctx_pos = is_local(l) ? i32s(std::size_t(max_batch_) * kd::WINDOW) : nullptr;
    if (L.ctx_pos) {
      CUDA_CHECK(cudaMemsetAsync(L.ctx_pos, 0xff, std::size_t(max_batch_) * kd::WINDOW * sizeof(std::int32_t), stream_));
    }
    ctx_layers_.ctx_norm[l] = L.ctx_norm;
    ctx_layers_.k_norm_ctx[l] = L.k_norm_ctx;
    ctx_layers_.k_cache[l] = L.k_ctx;
    ctx_layers_.v_cache[l] = L.v_ctx;
    ctx_layers_.kv_slot_bytes[l] = L.kv_slot_bytes;
    ctx_layers_.fp8[l] = !is_local(l) && GLOBAL_FP8;
    ctx_layers_.ctx_pos[l] = L.ctx_pos;
    ctx_layers_.size[l] = is_local(l) ? kd::WINDOW : ctx_;
  }

  slot_length_.assign(std::size_t(max_batch_), 0);
  slot_seed_.assign(std::size_t(max_batch_), 0);
  slot_draws_.assign(std::size_t(max_batch_), 0);
  slot_extra_ids_host_.assign(std::size_t(max_batch_), {});
  slot_extra_set_.assign(std::size_t(max_batch_), {});
  slot_extra_ids_dev_ = i32s(std::size_t(max_batch_) * MAX_SLOT_EXTRA);
  CUDA_CHECK(cudaMemsetAsync(slot_extra_ids_dev_, 0, std::size_t(max_batch_) * MAX_SLOT_EXTRA * sizeof(std::int32_t), stream_));
  f_rnn_pool_ = bf16s(std::size_t(max_batch_) * H);
  CUDA_CHECK(cudaMemsetAsync(f_rnn_pool_, 0, std::size_t(max_batch_) * H * sizeof(ActT), stream_));

  draft_meta_dev_ = static_cast<kd::BatchSlotDraftMeta*>(alloc(std::size_t(max_batch_) * sizeof(kd::BatchSlotDraftMeta)));
  commit_meta_dev_ =
      static_cast<kd::BatchSlotCommitMeta*>(alloc(std::size_t(max_batch_) * sizeof(kd::BatchSlotCommitMeta)));
  prefill_params_ = static_cast<kd::DraftParams*>(alloc(sizeof(kd::DraftParams)));
  CUDA_CHECK(cudaMemsetAsync(prefill_params_, 0, sizeof(kd::DraftParams), stream_));

  const std::size_t max_rows = std::size_t(max_batch_) * BLOCK;
  const std::size_t max_gamma = std::size_t(max_batch_) * GAMMA;
  const std::size_t max_commit_rows = std::size_t(max_batch_) * std::size_t(max_verify_width_);

  res_ = bf16s(max_rows * H);
  xn_ = bf16s(max_rows * H);
  q_ = bf16s(max_rows * kd::Q_DIM);
  attn_o_ = bf16s(max_rows * kd::Q_DIM);
  act_ = bf16s(max_rows * std::size_t(inter_));
  h_gamma_ = bf16s(max_gamma * H);

  const bool need_gemm = (max_rows > 32);
  qkv_f32_ = f32s(std::min<std::size_t>(max_rows, 32) * kd::QKV_ROWS);
  delta_f32_ = f32s(std::min<std::size_t>(max_rows, 32) * H);
  gu_f32_ = f32s(std::min<std::size_t>(max_rows, 32) * 2 * std::size_t(inter_));
  base_tmp_f32_ = f32s(std::min<std::size_t>(max_gamma, 32) * std::size_t(short_K_));
  if (need_gemm) {
    qkv_act_ = bf16s(max_rows * kd::QKV_ROWS);
    delta_act_ = bf16s(max_rows * H);
    gu_act_ = bf16s(max_rows * 2 * std::size_t(inter_));
    base_act_ = bf16s(max_gamma * kShortChunk);
    const std::size_t wscratch_elems = std::max({std::size_t(kd::QKV_ROWS) * H, std::size_t(2) * inter_ * H,
                                                 std::size_t(H) * kd::Q_DIM, std::size_t(H) * inter_,
                                                 std::size_t(kShortChunk) * H});
    wscratch_ = bf16s(wscratch_elems);
  }
  base_ = f32s(max_gamma * std::size_t(base_stride_));
  kd::fill(base_, max_gamma * std::size_t(base_stride_), -INFINITY, stream_);

  attn_ws_ = f32s(kd::batch_attention_workspace_floats(max_batch_, std::max(LOCAL_SLOTS, ctx_)));
  const std::size_t topk_bytes = kd::batch_topk_workspace_bytes(int(max_gamma));
  topk_ws_ = alloc(topk_bytes);
  CUDA_CHECK(cudaMemsetAsync(topk_ws_, 0, topk_bytes, stream_));

  cand_val_ = f32s(max_gamma * kd::CANDIDATES);
  cand_logits_ = f32s(max_gamma * kd::CANDIDATES);
  cand_idx_ = i32s(max_gamma * kd::CANDIDATES);
  part_ = f32s(max_gamma * kd::RNN_OUT);
  state_ = bf16s(std::size_t(max_batch_) * (GAMMA + 1) * kd::RANK);
  latent_ = bf16s(std::size_t(max_batch_) * kd::RANK);
  picks_ = static_cast<unsigned long long*>(alloc(max_gamma * sizeof(unsigned long long)));
  tokens_ = i32s(max_gamma);
  conf_ = f32s(max_gamma);
  qstats_ = f32s(2 * max_gamma);
  CUDA_CHECK(cudaMemsetAsync(state_, 0, std::size_t(max_batch_) * (GAMMA + 1) * kd::RANK * sizeof(ActT), stream_));

  // Commit scratch
  ctx_c_ = bf16s(std::size_t(COMMIT_PIECE) * H);
  ctx_kv_ = bf16s(std::size_t(COMMIT_PIECE) * 2 * kd::KV_DIM);
  commit_xin_ = bf16s(3 * max_commit_rows * H);
  commit_kv_f32_ = f32s(std::min<std::size_t>(max_commit_rows, 32) * 3 * 2 * kd::KV_DIM);
  if (max_commit_rows > 32) commit_kv_act_ = bf16s(max_commit_rows * 2 * kd::KV_DIM);

  // Acceptance scratch (single reusable [8, V] probs + [7, V] draft_logits for sampled slots!)
  accept_.target_argmax = i32s(std::size_t(max_batch_) * std::size_t(max_verify_width_));
  accept_.accepted = i32s(std::size_t(max_batch_));
  accept_.next = i32s(std::size_t(max_batch_));
  accept_.commit = i32s(std::size_t(max_batch_));
  accept_.probs = f32s(std::size_t(BLOCK) * std::size_t(V_));
  accept_.draft_logits = f32s(std::size_t(GAMMA) * std::size_t(V_));
  kd::fill(accept_.draft_logits, std::size_t(GAMMA) * std::size_t(V_), -INFINITY, stream_);
  accept_.resid = f32s(std::size_t(V_));
  accept_.uniforms = f32s(GAMMA);
  accept_.ws = f32s(kd::sampled_accept_workspace_floats(V_));
  accept_.one_res = static_cast<kd::AcceptResult*>(alloc(sizeof(kd::AcceptResult)));

  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

BatchDrafter::~BatchDrafter() = default;

std::size_t BatchDrafter::device_bytes() const {
  std::size_t total = 0;
  for (const auto& b : storage_) total += b.bytes;
  return total;
}

void BatchDrafter::load(const std::filesystem::path& model_dir) {
  SafetensorsFile f(model_dir / "mtp.safetensors");
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

  DeviceBuffer cat(std::size_t(2) * std::size_t(I) * std::size_t(h) * 2);
  auto fp8 = [&](std::initializer_list<std::pair<std::string, std::int64_t>> parts, std::int64_t K = H) {
    std::int64_t rows = 0;
    for (const auto& [name, r] : parts) {
      const std::byte* s = src(name, {r, K});
      CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(cat.p) + rows * K * 2, s, std::size_t(r * K * 2),
                                 cudaMemcpyDeviceToDevice, stream_));
      rows += r;
    }
    void* q = alloc(std::size_t(rows * K));
    auto* scale = static_cast<float*>(alloc(std::size_t(rows) * sizeof(float)));
    kdense::fp8_rows(static_cast<const ActT*>(cat.p), int(rows), int(K), q, scale, stream_);
    return kdense::Matrix{q, scale, int(rows)};
  };

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

  fold_w_ = static_cast<ActT*>(alloc(std::size_t(V_) * kd::RANK * sizeof(ActT)));
  fold_b_ = static_cast<ActT*>(alloc(std::size_t(V_) * sizeof(ActT)));
  const ActT* hw = persist({{mh + "hidden_out.weight", {h, kd::RANK}}});
  const ActT* hb = persist({{mh + "hidden_out.bias", {h}}});
  kd::fold_head(w_.unembed.w.data, head_.scale, V_, hw, hb, fold_w_, fold_b_, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

const ActT* BatchDrafter::dequant_to_scratch(const kdense::Matrix& m, int K) {
  if (!m.scale) return static_cast<const ActT*>(m.w);
  kdense::dequant_rows(m.w, m.scale, m.rows, K, wscratch_, stream_);
  return wscratch_;
}

void BatchDrafter::reset_slot(int slot) {
  if (slot < 0 || slot >= max_batch_) throw std::invalid_argument("BatchDrafter::reset_slot: bad slot");
  for (int l : {0, 2}) {
    CUDA_CHECK(cudaMemsetAsync(layers_[l].ctx_pos + std::size_t(slot) * kd::WINDOW, 0xff,
                               kd::WINDOW * sizeof(std::int32_t), stream_));
  }
  CUDA_CHECK(cudaMemsetAsync(f_rnn_pool_ + std::size_t(slot) * H, 0, H * sizeof(ActT), stream_));
  slot_length_[std::size_t(slot)] = 0;
  slot_draws_[std::size_t(slot)] = 0;
  slot_extra_ids_host_[std::size_t(slot)].clear();
  slot_extra_set_[std::size_t(slot)].clear();
}

void BatchDrafter::set_slot_seed(int slot, std::uint64_t seed) {
  if (slot < 0 || slot >= max_batch_) throw std::invalid_argument("BatchDrafter::set_slot_seed: bad slot");
  slot_seed_[std::size_t(slot)] = seed;
  slot_draws_[std::size_t(slot)] = 0;
}

void BatchDrafter::note_tokens_slot(int slot, std::span<const std::int32_t> ids) {
  if (slot < 0 || slot >= max_batch_) throw std::invalid_argument("BatchDrafter::note_tokens_slot: bad slot");
  auto& vec = slot_extra_ids_host_[std::size_t(slot)];
  auto& set = slot_extra_set_[std::size_t(slot)];
  if (int(vec.size()) >= MAX_SLOT_EXTRA) return;
  const std::size_t prev = vec.size();
  for (std::int32_t t : ids) {
    if (t >= 0 && t < V_ && !short_base_in_[std::size_t(t)] && set.insert(t).second) {
      vec.push_back(t);
      if (int(vec.size()) >= MAX_SLOT_EXTRA) break;
    }
  }
  const std::size_t added = vec.size() - prev;
  if (added > 0) {
    std::int32_t* dst = slot_extra_ids_dev_ + std::size_t(slot) * MAX_SLOT_EXTRA + prev;
    CUDA_CHECK(cudaMemcpyAsync(dst, vec.data() + prev, added * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream_));
  }
}

void BatchDrafter::commit_slot_prefill(int slot, const ActT* taps, int n, int start) {
  if (n <= 0) return;
  if (slot < 0 || slot >= max_batch_) throw std::invalid_argument("BatchDrafter::commit_slot_prefill: bad slot");
  kd::set_commit_start(prefill_params_, start, stream_);
  for (int r0 = 0; r0 < n; r0 += COMMIT_PIECE) {
    const int k = std::min(COMMIT_PIECE, n - r0);
    const ActT* piece = taps + std::size_t(r0) * 4 * H;
    for (int l = 0; l < 3; ++l) {
      const DraftLayer& L = layers_[l];
      kd::mix_rmsnorm(piece, k, alpha_ + 4 * l, L.ctx_norm, ctx_c_, stream_);
      gemm_->run(ctx_c_, H, L.wkv_ctx, H, ctx_kv_, 2 * kd::KV_DIM, false, k, 2 * kd::KV_DIM, H, stream_);
      const int size = is_local(l) ? kd::WINDOW : ctx_;
      void* kc = static_cast<std::uint8_t*>(L.k_ctx) + std::size_t(slot) * L.kv_slot_bytes;
      void* vc = static_cast<std::uint8_t*>(L.v_ctx) + std::size_t(slot) * L.kv_slot_bytes;
      std::int32_t* cp = is_local(l) ? (L.ctx_pos + std::size_t(slot) * kd::WINDOW) : nullptr;
      kd::ctx_kv_store(ctx_kv_, k, prefill_params_, r0, k - std::min(k, size), L.k_norm_ctx, kc, vc,
                       !is_local(l) && GLOBAL_FP8, size, cp, stream_);
    }
    if (r0 + k == n) {
      kd::mix_rmsnorm(piece + std::size_t(k - 1) * 4 * H, 1, alpha_ + 12, f_norm_,
                      f_rnn_pool_ + std::size_t(slot) * H, stream_);
    }
  }
  slot_length_[std::size_t(slot)] = start + n;
}

void BatchDrafter::commit_batch(std::span<const kd::BatchSlotCommitMeta> commits, const ActT* taps, int W) {
  const int B = int(commits.size());
  if (B <= 0 || W <= 0) return;
  auto* hm = static_cast<kd::BatchSlotCommitMeta*>(commit_meta_host_.data());
  for (int b = 0; b < B; ++b) {
    hm[b] = commits[std::size_t(b)];
    if (hm[b].slot >= 0) slot_length_[std::size_t(hm[b].slot)] = hm[b].start + hm[b].ncommit;
  }
  CUDA_CHECK(cudaMemcpyAsync(commit_meta_dev_, hm, std::size_t(B) * sizeof(kd::BatchSlotCommitMeta),
                             cudaMemcpyHostToDevice, stream_));
  kd::batch_commit_mix(commit_meta_dev_, B, W, taps, alpha_, ctx_layers_, f_norm_, commit_xin_, f_rnn_pool_, stream_);

  const int M = B * W;
  const std::size_t slab = std::size_t(M) * H;
  // Always use resident full-precision layers_[l].wkv_ctx (matching commit_slot_prefill) so drafter KV
  // precision is consistent across all batch sizes M (eliminates FP8-vs-ActT drift between B=1 and B>1).
  ActT* dst_kv = (M <= COMMIT_PIECE) ? ctx_kv_ : commit_kv_act_;
  for (int l = 0; l < 3; ++l) {
    gemm_->run(commit_xin_ + std::size_t(l) * slab, H, layers_[l].wkv_ctx, H, dst_kv, 2 * kd::KV_DIM, false, M,
               2 * kd::KV_DIM, H, stream_);
    kd::batch_commit_store(dst_kv, false, 2 * kd::KV_DIM, 0, l, commit_meta_dev_, B, W, ctx_layers_, stream_);
  }
}

BatchDrafter::BatchDraftOutput BatchDrafter::draft_batch(std::span<const DraftRequest> reqs) {
  const int B = int(reqs.size());
  if (B <= 0 || B > max_batch_) throw std::invalid_argument("BatchDrafter::draft_batch: bad batch size");

  auto* hm = static_cast<kd::BatchSlotDraftMeta*>(draft_meta_host_.data());
  int max_P = 0;
  int max_extra = 0;
  for (int b = 0; b < B; ++b) {
    const int s = reqs[std::size_t(b)].slot;
    kd::BatchSlotDraftMeta m{};
    m.slot = s;
    if (s >= 0) {
      const int P = slot_length_[std::size_t(s)];
      m.anchor = reqs[std::size_t(b)].anchor;
      m.P = P;
      m.temperature = reqs[std::size_t(b)].temperature;
      m.seed = slot_seed_[std::size_t(s)];
      m.draws = slot_draws_[std::size_t(s)]++;
      m.extra_count = int(slot_extra_ids_host_[std::size_t(s)].size());
      max_P = std::max(max_P, P);
      max_extra = std::max(max_extra, m.extra_count);
    }
    hm[b] = m;
  }
  CUDA_CHECK(cudaMemcpyAsync(draft_meta_dev_, hm, std::size_t(B) * sizeof(kd::BatchSlotDraftMeta),
                             cudaMemcpyHostToDevice, stream_));

  const int N = B * BLOCK;
  const int G = B * GAMMA;
  const bool use_gemv = (N <= 32);
  const int global_nkeys = std::min(ctx_, std::max(1024, ((max_P + BLOCK + 1023) / 1024) * 1024));

  kd::batch_embed_block(static_cast<const ActT*>(w_.embed.data), static_cast<const float*>(w_.embed_norm.data),
                        mask_embed_, layers_[0].in_norm, draft_meta_dev_, B, res_, xn_, picks_, stream_,
                        w_.embed_q4());

  for (int l = 0; l < 3; ++l) {
    const DraftLayer& L = layers_[l];
    const bool local = is_local(l);

    if (use_gemv) {
      for (int r0 = 0; r0 < N; r0 += 8) {
        const int cr = std::min(8, N - r0);
        kdense::proj(xn_ + std::size_t(r0) * H, cr, L.qkv, seg_qkv_, 1, qkv_f32_ + std::size_t(r0) * kd::QKV_ROWS,
                     false, stream_);
      }
      kd::batch_qkv_rope(qkv_f32_, true, L.q_norm, L.k_norm, draft_meta_dev_, B, q_, L.k_ctx, L.v_ctx,
                         L.kv_slot_bytes, !local && GLOBAL_FP8, local, ctx_, stream_);
    } else {
      const ActT* wqkv = dequant_to_scratch(L.qkv, H);
      gemm_->run(xn_, H, wqkv, H, qkv_act_, kd::QKV_ROWS, false, N, kd::QKV_ROWS, H, stream_);
      kd::batch_qkv_rope(qkv_act_, false, L.q_norm, L.k_norm, draft_meta_dev_, B, q_, L.k_ctx, L.v_ctx,
                         L.kv_slot_bytes, !local && GLOBAL_FP8, local, ctx_, stream_);
    }

    kd::batch_block_attention(q_, L.k_ctx, L.v_ctx, L.kv_slot_bytes, local ? LOCAL_SLOTS : global_nkeys, L.ctx_pos,
                              !local && GLOBAL_FP8, draft_meta_dev_, B, attn_ws_, attn_o_, stream_);

    if (use_gemv) {
      for (int r0 = 0; r0 < N; r0 += 8) {
        const int cr = std::min(8, N - r0);
        kdense::gemv(attn_o_ + std::size_t(r0) * kd::Q_DIM, cr, kd::Q_DIM, L.wo, delta_f32_ + std::size_t(r0) * H,
                     false, stream_);
      }
      kd::batch_add_rmsnorm(res_, delta_f32_, true, L.post_norm, xn_, N, stream_);
    } else {
      const ActT* wwo = dequant_to_scratch(L.wo, kd::Q_DIM);
      gemm_->run(attn_o_, kd::Q_DIM, wwo, kd::Q_DIM, delta_act_, H, false, N, H, kd::Q_DIM, stream_);
      kd::batch_add_rmsnorm(res_, delta_act_, false, L.post_norm, xn_, N, stream_);
    }

    if (use_gemv) {
      for (int r0 = 0; r0 < N; r0 += 8) {
        const int cr = std::min(8, N - r0);
        kdense::proj(xn_ + std::size_t(r0) * H, cr, L.gate_up, seg_gate_up_, 1,
                     gu_f32_ + std::size_t(r0) * 2 * std::size_t(inter_), false, stream_);
      }
      kd::batch_silu_mul(gu_f32_, true, N, inter_, act_, stream_);
    } else {
      const ActT* wgu = dequant_to_scratch(L.gate_up, H);
      gemm_->run(xn_, H, wgu, H, gu_act_, 2 * inter_, false, N, 2 * inter_, H, stream_);
      kd::batch_silu_mul(gu_act_, false, N, inter_, act_, stream_);
    }

    const ActT* next_norm = (l < 2) ? layers_[l + 1].in_norm : final_norm_;
    if (use_gemv) {
      for (int r0 = 0; r0 < N; r0 += 8) {
        const int cr = std::min(8, N - r0);
        kdense::gemv(act_ + std::size_t(r0) * std::size_t(inter_), cr, inter_, L.w_down,
                     delta_f32_ + std::size_t(r0) * H, false, stream_);
      }
      kd::batch_add_rmsnorm(res_, delta_f32_, true, next_norm, xn_, N, stream_);
    } else {
      const ActT* wd = dequant_to_scratch(L.w_down, inter_);
      gemm_->run(act_, inter_, wd, inter_, delta_act_, H, false, N, H, inter_, stream_);
      kd::batch_add_rmsnorm(res_, delta_act_, false, next_norm, xn_, N, stream_);
    }
  }

  // Extract rows 1..7 of each slot into h_gamma [B * 7, H], score immutable ranked shortlist, then slot-local extras!
  kd::batch_extract_gamma_rows(xn_, B, h_gamma_, stream_);
  if (G <= 32) {
    for (int r0 = 0; r0 < G; r0 += 8) {
      const int cr = std::min(8, G - r0);
      kdense::proj(h_gamma_ + std::size_t(r0) * H, cr, short_head_, seg_short_, 1,
                   base_tmp_f32_ + std::size_t(r0) * std::size_t(short_K_), false, stream_);
    }
    kd::batch_copy_base_rows(base_tmp_f32_, G, short_K_, base_stride_, base_, stream_);
  } else {
    for (int v0 = 0; v0 < short_K_; v0 += kShortChunk) {
      const int cv = std::min(kShortChunk, short_K_ - v0);
      const ActT* wchunk;
      if (short_head_.scale) {
        const auto* q = static_cast<const std::uint8_t*>(short_head_.w) + std::size_t(v0) * H;
        kdense::dequant_rows(q, short_head_.scale + v0, cv, H, wscratch_, stream_);
        wchunk = wscratch_;
      } else {
        wchunk = static_cast<const ActT*>(short_head_.w) + std::size_t(v0) * H;
      }
      gemm_->run(h_gamma_, H, wchunk, H, base_act_, cv, false, G, cv, H, stream_);
      kd::batch_cast_base_chunk(base_act_, G, cv, v0, base_stride_, base_, stream_);
    }
  }
  if (max_extra > 0) {
    kd::batch_score_slot_extra(h_gamma_, B, head_.w, head_.scale, slot_extra_ids_dev_, MAX_SLOT_EXTRA, draft_meta_dev_,
                               short_K_, base_stride_, base_, stream_);
  }

  kd::batch_topk_candidates(base_, B, base_stride_, short_K_, short_map_, slot_extra_ids_dev_, MAX_SLOT_EXTRA,
                            draft_meta_dev_, cand_val_, cand_idx_, topk_ws_, stream_);
  kd::batch_markov_pre(rnn_w_, h_gamma_, f_rnn_pool_, draft_meta_dev_, B, part_, stream_);
  for (int i = 0; i < GAMMA; ++i) {
    kd::batch_markov_step(i, rnn_w_, rnn_b_, w1_, part_, picks_, cand_idx_, draft_meta_dev_, B, state_, latent_,
                          stream_);
    kd::batch_markov_logits(i, fold_w_, fold_b_, cand_val_, cand_idx_, latent_, draft_meta_dev_, B, cand_logits_,
                            picks_, stream_);
  }
  kd::batch_markov_finish(cand_idx_, cand_logits_, picks_, draft_meta_dev_, B, tokens_, conf_, qstats_, stream_);
  return {tokens_, conf_, qstats_, cand_idx_, cand_logits_};
}

std::size_t BatchDrafter::estimate_checkpoint_bytes(int length) const {
  const int local_rows = std::min(std::max(length, 0), kd::WINDOW);
  const int global_rows = std::min(std::max(length, 0), ctx_);
  const std::size_t local_kv = std::size_t(local_rows) * kd::KV_DIM * sizeof(ActT);
  const std::size_t local_pos = std::size_t(local_rows) * sizeof(std::int32_t);
  const std::size_t global_kv = std::size_t(global_rows) * GLOBAL_SLOT_BYTES;
  return 4 * local_kv + 2 * local_pos + 2 * global_kv + std::size_t(H) * sizeof(ActT);
}

std::size_t BatchDrafter::save_slot_checkpoint(int slot, int length, DraftCheckpointGpu& out) {
  const int local_rows = std::min(std::max(length, 0), kd::WINDOW);
  const int global_rows = std::min(std::max(length, 0), ctx_);
  out.length = length;
  out.local_rows = local_rows;
  out.global_rows = global_rows;

  const std::size_t local_kv_bytes = std::size_t(local_rows) * kd::KV_DIM * sizeof(ActT);
  const std::size_t local_pos_bytes = std::size_t(local_rows) * sizeof(std::int32_t);
  const std::size_t global_kv_bytes = std::size_t(global_rows) * GLOBAL_SLOT_BYTES;
  const std::size_t f_rnn_bytes = std::size_t(H) * sizeof(ActT);

  std::size_t copied = 0;
  const int local_layers[2] = {0, 2};
  for (int idx = 0; idx < 2; ++idx) {
    const int l = local_layers[idx];
    const DraftLayer& L = layers_[l];
    if (local_kv_bytes > 0) {
      out.local_k[idx] = DeviceBuffer(local_kv_bytes);
      out.local_v[idx] = DeviceBuffer(local_kv_bytes);
      out.local_pos[idx] = DeviceBuffer(local_pos_bytes);
      const auto* src_k = static_cast<const std::uint8_t*>(L.k_ctx) + std::size_t(slot) * L.kv_slot_bytes;
      const auto* src_v = static_cast<const std::uint8_t*>(L.v_ctx) + std::size_t(slot) * L.kv_slot_bytes;
      const auto* src_p = L.ctx_pos + std::size_t(slot) * kd::WINDOW;
      CUDA_CHECK(cudaMemcpyAsync(out.local_k[idx].p, src_k, local_kv_bytes, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(out.local_v[idx].p, src_v, local_kv_bytes, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(out.local_pos[idx].p, src_p, local_pos_bytes, cudaMemcpyDeviceToDevice, stream_));
      copied += 2 * local_kv_bytes + local_pos_bytes;
    }
  }
  if (global_kv_bytes > 0) {
    out.global_k = DeviceBuffer(global_kv_bytes);
    out.global_v = DeviceBuffer(global_kv_bytes);
    const auto* src_k = static_cast<const std::uint8_t*>(layers_[1].k_ctx) + std::size_t(slot) * layers_[1].kv_slot_bytes;
    const auto* src_v = static_cast<const std::uint8_t*>(layers_[1].v_ctx) + std::size_t(slot) * layers_[1].kv_slot_bytes;
    CUDA_CHECK(cudaMemcpyAsync(out.global_k.p, src_k, global_kv_bytes, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(out.global_v.p, src_v, global_kv_bytes, cudaMemcpyDeviceToDevice, stream_));
    copied += 2 * global_kv_bytes;
  }
  out.f_rnn = DeviceBuffer(f_rnn_bytes);
  CUDA_CHECK(cudaMemcpyAsync(out.f_rnn.p, f_rnn_pool_ + std::size_t(slot) * H, f_rnn_bytes, cudaMemcpyDeviceToDevice,
                             stream_));
  copied += f_rnn_bytes;
  out.extra_ids = slot_extra_ids_host_[std::size_t(slot)];
  return copied;
}

std::size_t BatchDrafter::restore_slot_checkpoint(int slot, const DraftCheckpointGpu& ckpt) {
  std::size_t copied = 0;
  const int local_layers[2] = {0, 2};
  for (int idx = 0; idx < 2; ++idx) {
    const int l = local_layers[idx];
    DraftLayer& L = layers_[l];
    std::int32_t* dst_p = L.ctx_pos + std::size_t(slot) * kd::WINDOW;
    if (ckpt.local_rows < kd::WINDOW) {
      CUDA_CHECK(cudaMemsetAsync(dst_p, 0xff, kd::WINDOW * sizeof(std::int32_t), stream_));
    }
    if (ckpt.local_rows > 0) {
      auto* dst_k = static_cast<std::uint8_t*>(L.k_ctx) + std::size_t(slot) * L.kv_slot_bytes;
      auto* dst_v = static_cast<std::uint8_t*>(L.v_ctx) + std::size_t(slot) * L.kv_slot_bytes;
      CUDA_CHECK(cudaMemcpyAsync(dst_k, ckpt.local_k[idx].p, ckpt.local_k[idx].bytes, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(dst_v, ckpt.local_v[idx].p, ckpt.local_v[idx].bytes, cudaMemcpyDeviceToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(dst_p, ckpt.local_pos[idx].p, ckpt.local_pos[idx].bytes, cudaMemcpyDeviceToDevice, stream_));
      copied += ckpt.local_k[idx].bytes + ckpt.local_v[idx].bytes + ckpt.local_pos[idx].bytes;
    }
  }
  // Unconditionally copy ALL live global KV rows so cross-slot restores are always complete!
  if (ckpt.global_rows > 0) {
    auto* dst_k = static_cast<std::uint8_t*>(layers_[1].k_ctx) + std::size_t(slot) * layers_[1].kv_slot_bytes;
    auto* dst_v = static_cast<std::uint8_t*>(layers_[1].v_ctx) + std::size_t(slot) * layers_[1].kv_slot_bytes;
    CUDA_CHECK(cudaMemcpyAsync(dst_k, ckpt.global_k.p, ckpt.global_k.bytes, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(dst_v, ckpt.global_v.p, ckpt.global_v.bytes, cudaMemcpyDeviceToDevice, stream_));
    copied += ckpt.global_k.bytes + ckpt.global_v.bytes;
  }
  if (ckpt.f_rnn.bytes > 0) {
    CUDA_CHECK(cudaMemcpyAsync(f_rnn_pool_ + std::size_t(slot) * H, ckpt.f_rnn.p, ckpt.f_rnn.bytes,
                               cudaMemcpyDeviceToDevice, stream_));
    copied += ckpt.f_rnn.bytes;
  }
  slot_length_[std::size_t(slot)] = ckpt.length;
  slot_extra_ids_host_[std::size_t(slot)] = ckpt.extra_ids;
  slot_extra_set_[std::size_t(slot)].clear();
  for (std::int32_t id : ckpt.extra_ids) slot_extra_set_[std::size_t(slot)].insert(id);
  if (!ckpt.extra_ids.empty()) {
    const std::size_t eb = ckpt.extra_ids.size() * sizeof(std::int32_t);
    CUDA_CHECK(cudaMemcpyAsync(slot_extra_ids_dev_ + std::size_t(slot) * MAX_SLOT_EXTRA, ckpt.extra_ids.data(), eb,
                               cudaMemcpyHostToDevice, stream_));
    copied += eb;
  }
  return copied;
}

}  // namespace ingine::model

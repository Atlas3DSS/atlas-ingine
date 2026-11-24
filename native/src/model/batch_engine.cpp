// Continuous-batching runtime and generator (B = 1..64+ concurrent sequences) with paged full-attention KV
// cache (page size 64), per-slot sliding rings + 4-tap conv states, bucketed CUDA-graph decode capture, and
// zero-copy Prefiller slot integration.
#include "model/batch_engine.h"

#include "core/act.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "kernels/batch/batch.h"
#include "kernels/dense/dense.h"
#include "kernels/head/head.h"
#include "kernels/prefill/prefill.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "model/vision.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ingine::model {
namespace {

namespace bk = kernels::batch;
namespace dense = kernels::dense;
namespace head = kernels::head;
namespace pk = kernels::prefill;

constexpr int H = HIDDEN;
constexpr int kHeadChunk = 32768;  // tiled FP8/BF16 unembed chunk for B > 8 (32768 * 2304 * 2 = 150.9 MB)

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

const float* f32p(const Tensor& t) { return static_cast<const float*>(t.data); }
const ActT* bf16p(const Tensor& t) { return static_cast<const ActT*>(t.data); }
const ActT* bf16p(const void* p) { return static_cast<const ActT*>(p); }

dense::Matrix matrix(const Dense& d) {
  return dense::Matrix{d.w.data, static_cast<const float*>(d.scale.data), d.rows()};
}

std::vector<int> proj_layout(bool full, bool single_input) {
  std::vector<int> t;
  for (int s = 0; s < num_segs(full); ++s) {
    t.push_back(SEG_BEGIN[s]);
    t.push_back(SEG_BEGIN[s + 1]);
    t.push_back(single_input ? 0 : SEG_INPUT[s]);
  }
  return t;
}

DeviceBuffer upload_ints(const std::vector<int>& v) {
  DeviceBuffer b(v.size() * sizeof(int));
  b.copy_from_host(v.data(), v.size() * sizeof(int));
  return b;
}

std::vector<const float*> blocks_of(int l, int p, const float* B0, const float* B1, float* const* B2, float* const* E) {
  if (l < 1) return {};
  if (l < 7) return {B0};
  if (l < 13) return {B0, B1};
  if (l < 19) return {B0, B1, B2[l < LOOP_END ? p : 2]};
  return {B0, B1, B2[2], E[2]};
}

head::SamplingParams to_head_params(const Sampling& sampling) {
  head::SamplingParams p;
  p.temperature = sampling.temperature;
  p.top_k = sampling.top_k;
  p.top_p = sampling.top_p;
  for (std::int32_t id : sampling.banned_tokens) {
    if (id >= 0 && p.banned.count < head::BannedTokens::kMax) p.banned.ids[p.banned.count++] = id;
  }
  return p;
}

bool same_banned(const head::BannedTokens& a, const head::BannedTokens& b) {
  if (a.count != b.count) return false;
  for (int i = 0; i < a.count; ++i)
    if (a.ids[i] != b.ids[i]) return false;
  return true;
}

// ----------------------------------------------------------------------------- Per-slot state + paged full KV pool

struct BatchStatePool {
  int max_batch = 1;
  int max_ctx = 0;
  bool kv_fp8 = false;
  int max_pages_per_slot = 0;
  int num_pages = 0;

  // Per-slot contiguous slices: slot s starts at `base + s * *_slot_stride`.
  std::size_t ring_slot_stride = std::size_t(NUM_SLIDING_VISITS) * WINDOW * KV_DIM;
  std::size_t ring_pos_slot_stride = std::size_t(NUM_SLIDING_VISITS) * WINDOW;
  std::size_t conv_kv_slot_stride = std::size_t(NUM_VISITS) * (CONV_K - 1) * KV_DIM;
  std::size_t conv_h_slot_stride = std::size_t(NUM_VISITS) * (CONV_K - 1) * HIDDEN;

  DeviceBuffer k_ring, v_ring, ring_pos;
  DeviceBuffer conv_k, conv_v, conv_attn, conv_mlp;
  DeviceBuffer slot_pos;  // int64 [max_batch]

  // Shared paged full-attention KV pool: [num_pages, NUM_FULL_VISITS, kPageSize, KV_HEADS, HEAD_DIM]
  DeviceBuffer k_full_pool, v_full_pool;
  DeviceBuffer page_table;  // int32 [max_batch, max_pages_per_slot]

  // Single scratch full-KV buffer for Prefiller::run (reused across slots, scattered immediately after prefill)
  DeviceBuffer scratch_k_full, scratch_v_full;

  // Host-side allocator state for physical full-KV pages
  std::vector<int> free_pages;
  std::vector<std::vector<int>> slot_pages;
  std::vector<int> slot_pos_host;
  std::vector<bool> slot_in_use;

  void init(int max_b, int ctx, bool fp8, int max_kv_tokens) {
    max_batch = max_b;
    max_ctx = ctx;
    kv_fp8 = fp8;
    max_pages_per_slot = (ctx + bk::kPageSize - 1) / bk::kPageSize;

    // Allocate per-slot sliding rings & conv states
    k_ring = DeviceBuffer(std::size_t(max_b) * ring_slot_stride * sizeof(ActT));
    v_ring = DeviceBuffer(std::size_t(max_b) * ring_slot_stride * sizeof(ActT));
    ring_pos = DeviceBuffer(std::size_t(max_b) * ring_pos_slot_stride * sizeof(std::int64_t));
    conv_k = DeviceBuffer(std::size_t(max_b) * conv_kv_slot_stride * sizeof(float));
    conv_v = DeviceBuffer(std::size_t(max_b) * conv_kv_slot_stride * sizeof(float));
    conv_attn = DeviceBuffer(std::size_t(max_b) * conv_h_slot_stride * sizeof(float));
    conv_mlp = DeviceBuffer(std::size_t(max_b) * conv_h_slot_stride * sizeof(float));
    slot_pos = DeviceBuffer(std::size_t(max_b) * sizeof(std::int64_t));

    k_ring.fill(0);
    v_ring.fill(0);
    ring_pos.fill(0xff);
    conv_k.fill(0);
    conv_v.fill(0);
    conv_attn.fill(0);
    conv_mlp.fill(0);
    slot_pos.fill(0);

    // Scratch full-KV buffer for single-sequence Prefiller
    const std::size_t es = fp8 ? 1 : 2;
    const std::size_t scratch_bytes = std::size_t(NUM_FULL_VISITS) * ctx * KV_DIM * es;
    scratch_k_full = DeviceBuffer(scratch_bytes);
    scratch_v_full = DeviceBuffer(scratch_bytes);
    scratch_k_full.fill(0);
    scratch_v_full.fill(0);

    // Paged full-KV pool sizing
    const std::size_t page_kv_bytes = std::size_t(NUM_FULL_VISITS) * bk::kPageSize * KV_DIM * es;
    const std::size_t bytes_per_page = 2 * page_kv_bytes;  // K + V
    const int max_possible_pages = max_b * max_pages_per_slot;
    if (max_kv_tokens > 0) {
      num_pages = std::min(max_possible_pages, std::max(max_pages_per_slot, (max_kv_tokens + bk::kPageSize - 1) / bk::kPageSize));
    } else {
      std::size_t free_bytes = 0, total_bytes = 0;
      CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
      // Leave headroom for CUDA graphs, cuBLASLt, activation buffers, and workspace (~2.5 GiB or 20% of free).
      const std::size_t reserve = std::min<std::size_t>(std::size_t(2560) << 20, free_bytes / 4);
      const std::size_t avail = free_bytes > reserve ? free_bytes - reserve : 0;
      const int fit_pages = int(avail / bytes_per_page);
      num_pages = std::min(max_possible_pages, std::max(std::min(max_possible_pages, max_pages_per_slot), fit_pages));
      if (num_pages < 1) throw std::runtime_error("BatchStatePool: insufficient VRAM for paged full-KV pool");
    }

    k_full_pool = DeviceBuffer(std::size_t(num_pages) * page_kv_bytes);
    v_full_pool = DeviceBuffer(std::size_t(num_pages) * page_kv_bytes);
    k_full_pool.fill(0);
    v_full_pool.fill(0);

    page_table = DeviceBuffer(std::size_t(max_b) * max_pages_per_slot * sizeof(std::int32_t));
    page_table.fill(0xff);  // -1

    free_pages.resize(num_pages);
    for (int i = 0; i < num_pages; ++i) free_pages[i] = num_pages - 1 - i;  // pop_back gives 0, 1, 2...
    slot_pages.assign(max_b, {});
    slot_pos_host.assign(max_b, 0);
    slot_in_use.assign(max_b, false);
  }

  // Zero-copy State view pointing directly at slot `s`'s sliding rings & conv states, and shared scratch full KV.
  State prefill_view(int s) const {
    State st;
    st.max_ctx = max_ctx;
    st.kv_fp8 = kv_fp8;
    const DType full_dt = kv_fp8 ? DType::FP8_E4M3FN : kActDType;
    st.k_ring = rm(static_cast<ActT*>(k_ring.p) + std::size_t(s) * ring_slot_stride, kActDType,
                   {NUM_SLIDING_VISITS, WINDOW, KV_HEADS, HEAD_DIM});
    st.v_ring = rm(static_cast<ActT*>(v_ring.p) + std::size_t(s) * ring_slot_stride, kActDType,
                   {NUM_SLIDING_VISITS, WINDOW, KV_HEADS, HEAD_DIM});
    st.ring_pos = rm(static_cast<std::int64_t*>(ring_pos.p) + std::size_t(s) * ring_pos_slot_stride, DType::I64,
                     {NUM_SLIDING_VISITS, WINDOW});
    st.k_full = rm(scratch_k_full.p, full_dt, {NUM_FULL_VISITS, max_ctx, KV_HEADS, HEAD_DIM});
    st.v_full = rm(scratch_v_full.p, full_dt, {NUM_FULL_VISITS, max_ctx, KV_HEADS, HEAD_DIM});
    st.conv_k = rm(static_cast<float*>(conv_k.p) + std::size_t(s) * conv_kv_slot_stride, DType::FP32,
                   {NUM_VISITS, CONV_K - 1, KV_DIM});
    st.conv_v = rm(static_cast<float*>(conv_v.p) + std::size_t(s) * conv_kv_slot_stride, DType::FP32,
                   {NUM_VISITS, CONV_K - 1, KV_DIM});
    st.conv_attn = rm(static_cast<float*>(conv_attn.p) + std::size_t(s) * conv_h_slot_stride, DType::FP32,
                      {NUM_VISITS, CONV_K - 1, HIDDEN});
    st.conv_mlp = rm(static_cast<float*>(conv_mlp.p) + std::size_t(s) * conv_h_slot_stride, DType::FP32,
                     {NUM_VISITS, CONV_K - 1, HIDDEN});
    st.length = 0;
    return st;
  }

  int free_slot_count() const {
    int n = 0;
    for (bool used : slot_in_use)
      if (!used) ++n;
    return n;
  }

  int alloc_slot() {
    for (int s = 0; s < max_batch; ++s) {
      if (!slot_in_use[s]) {
        slot_in_use[s] = true;
        slot_pages[s].clear();
        slot_pos_host[s] = 0;
        return s;
      }
    }
    return -1;
  }

  bool ensure_pages(int s, int total_tokens, cudaStream_t stream) {
    const int needed = (total_tokens + bk::kPageSize - 1) / bk::kPageSize;
    if (needed > max_pages_per_slot) return false;
    auto& pages = slot_pages[s];
    const int cur = int(pages.size());
    if (needed <= cur) return true;
    const int add = needed - cur;
    if (int(free_pages.size()) < add) return false;
    for (int i = 0; i < add; ++i) {
      pages.push_back(free_pages.back());
      free_pages.pop_back();
    }
    std::int32_t* dev_row = static_cast<std::int32_t*>(page_table.p) + std::size_t(s) * max_pages_per_slot;
    CUDA_CHECK(cudaMemcpyAsync(dev_row + cur, pages.data() + cur, std::size_t(add) * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    return true;
  }

  void free_slot(int s) {
    if (s < 0 || s >= max_batch || !slot_in_use[s]) return;
    for (int p : slot_pages[s]) free_pages.push_back(p);
    slot_pages[s].clear();
    slot_pos_host[s] = 0;
    slot_in_use[s] = false;
  }

  void set_slot_pos(int s, std::int64_t pos, cudaStream_t stream) {
    slot_pos_host[s] = int(pos);
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int64_t*>(slot_pos.p) + s, &pos, sizeof(std::int64_t),
                               cudaMemcpyHostToDevice, stream));
  }
};

// ----------------------------------------------------------------------------- BatchDecoder (B = 1..max_batch)

class BatchDecoder {
 public:
  BatchDecoder(const Weights& w, const Ablation& abl, BatchStatePool& pool, int max_batch)
      : w_(w),
        abl_(abl),
        pool_(pool),
        max_batch_(max_batch),
        gemm_(std::make_unique<pk::Gemm>()),
        tokens_host_(std::size_t(max_batch) * sizeof(std::int64_t)),
        slots_host_(std::size_t(max_batch) * sizeof(std::int32_t)) {
    auto alloc = [&](std::size_t bytes) {
      storage_.emplace_back(bytes);
      storage_.back().fill(0);
      return storage_.back().p;
    };
    const std::size_t B = std::size_t(max_batch);
    auto f32 = [&](std::size_t n) { return static_cast<float*>(alloc(n * sizeof(float))); };
    auto bf16 = [&](std::size_t n) { return static_cast<ActT*>(alloc(n * sizeof(ActT))); };
    auto i32 = [&](std::size_t n) { return static_cast<int*>(alloc(n * sizeof(int))); };

    tokens_dev_ = static_cast<std::int64_t*>(alloc(B * sizeof(std::int64_t)));
    slot_ids_dev_ = static_cast<std::int32_t*>(alloc(B * sizeof(std::int32_t)));
    B0_ = f32(B * H), B1_ = f32(B * H);
    for (int p = 0; p < PASSES; ++p) B2_[p] = f32(B * H), E_[p] = f32(B * H);
    hid_ = f32(B * H), part_ = f32(B * H), y_ = f32(B * H), y2_ = f32(B * H);
    xin_ = bf16(3 * B * H), xm_ = bf16(B * H), xf_ = bf16(B * H);
    xin_chunk_ = (w.dense_fp8 && B > 8) ? bf16(3 * 8 * H) : nullptr;
    mixf_ = f32(3 * B * H), msgs_ = f32(4 * B * H), mixm_ = f32(B * H);
    proj_fp32_ = f32(std::min<std::size_t>(B, w.dense_fp8 ? 32 : 8) * PROJ_ROWS_FULL);
    proj_act_ = B > 8 ? bf16(B * PROJ_ROWS_FULL) : nullptr;
    attn_out_ = bf16(B * HEADS * HEAD_DIM);
    wo_fp32_ = f32(std::min<std::size_t>(B, w.dense_fp8 ? 32 : 8) * H);
    wo_act_ = B > 8 ? bf16(B * H) : nullptr;
    gin_ = B > 8 ? bf16(3 * B * 2 * H) : nullptr;
    g16_ = B > 8 ? bf16(B * GATE_RANK) : nullptr;
    fpre_ = B > 8 ? f32(B * H) : nullptr;
    gpre_ = B > 8 ? f32(B * H) : nullptr;
    router_logits_ = f32(B * EXPERTS);
    idx_ = i32(B * TOP_K);
    wts_ = f32(B * TOP_K);
    counts_ = i32(EXPERTS);
    tiles_n_ = i32(1);
    pair_token_ = i32(B * TOP_K);
    pair_dst_ = i32(B * TOP_K);
    tiles_ = static_cast<int4*>(alloc(std::size_t(pk::moe_tiles_capacity(max_batch)) * sizeof(int4)));
    act_ = bf16(B * TOP_K * INTER);
    down_ = bf16(B * TOP_K * H);
    moe_sum_ = bf16(B * H);
    logits_ = f32(B * UNPADDED_VOCAB);
    head_act_ = B > 8 ? bf16(B * kHeadChunk) : nullptr;
    wscratch_ = (w.dense_fp8 && B > 32) ? bf16(std::size_t(kHeadChunk) * H) : nullptr;
    attn_ws_ = DeviceBuffer(bk::attn_workspace_bytes(max_batch));
    attn_ws_.fill(0);

    segments_[0] = upload_ints(proj_layout(false, true));
    segments_[1] = upload_ints(proj_layout(false, false));
    full_segments_ = upload_ints(proj_layout(true, false));
  }

  ~BatchDecoder() {
    for (auto& [b, exec] : graphs_)
      if (exec) cudaGraphExecDestroy(exec);
  }

  void capture_buckets(cudaStream_t stream) {
    // Capture power-of-2 buckets up to max_batch_ (plus max_batch_ itself if not power of 2).
    std::vector<int> buckets;
    for (int b = 1; b < max_batch_; b <<= 1) buckets.push_back(b);
    buckets.push_back(max_batch_);

    // Warmup & capture with all slot_ids = -1 so ZERO state is modified.
    std::vector<std::int64_t> ztok(max_batch_, 0);
    std::vector<std::int32_t> nslot(max_batch_, -1);
    CUDA_CHECK(cudaMemcpyAsync(tokens_dev_, ztok.data(), ztok.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_ids_dev_, nslot.data(), nslot.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int B : buckets) {
      for (int w = 0; w < 2; ++w) decode_eager(B, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      cudaGraph_t graph = nullptr;
      CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
      try {
        decode_eager(B, stream);
      } catch (...) {
        cudaStreamEndCapture(stream, &graph);
        if (graph) cudaGraphDestroy(graph);
        throw;
      }
      CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
      cudaGraphExec_t exec = nullptr;
      const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
      cudaGraphDestroy(graph);
      CUDA_CHECK(err);
      CUDA_CHECK(cudaGraphUpload(exec, stream));
      graphs_[B] = exec;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // Runs one batched decode step for `inputs` (1..max_batch_). Returns device logits [B_active, UNPADDED_VOCAB].
  const float* run_step(std::span<const BatchSlotStep> inputs, cudaStream_t stream) {
    const int B_active = int(inputs.size());
    if (B_active < 1 || B_active > max_batch_) throw std::invalid_argument("BatchDecoder::run_step: bad batch size");

    // Pick smallest captured bucket >= B_active.
    auto it = graphs_.lower_bound(B_active);
    const int B_cap = (it != graphs_.end()) ? it->first : B_active;

    auto* ht = static_cast<std::int64_t*>(tokens_host_.data());
    auto* hs = static_cast<std::int32_t*>(slots_host_.data());
    for (int i = 0; i < B_active; ++i) {
      ht[i] = inputs[i].token;
      hs[i] = inputs[i].slot;
    }
    for (int i = B_active; i < B_cap; ++i) {
      ht[i] = 0;
      hs[i] = -1;
    }
    CUDA_CHECK(cudaMemcpyAsync(tokens_dev_, ht, std::size_t(B_cap) * sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(slot_ids_dev_, hs, std::size_t(B_cap) * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream));

    if (it != graphs_.end()) {
      CUDA_CHECK(cudaGraphLaunch(it->second, stream));
    } else {
      decode_eager(B_active, stream);
    }
    return logits_;
  }

 private:
  const float* const* block_table(const std::vector<const float*>& blocks) {
    auto it = tables_.find(blocks);
    if (it == tables_.end()) {
      DeviceBuffer b(std::max<std::size_t>(blocks.size(), 1) * sizeof(void*));
      if (!blocks.empty()) b.copy_from_host(blocks.data(), blocks.size() * sizeof(void*));
      it = tables_.emplace(blocks, std::move(b)).first;
    }
    return static_cast<const float* const*>(it->second.p);
  }

  void decode_eager(int B, cudaStream_t stream) {
    const Weights& w = w_;
    const Ablation& abl = abl_;
    const std::size_t slab = std::size_t(B) * H;
    const bool use_gemv = (B <= 8);

    const float* attn_dirs = f32p(abl.attn_dirs);
    const float* attn_w = f32p(abl.attn_w);
    const float* mlp_dirs = f32p(abl.mlp_dirs);
    const float* mlp_w = f32p(abl.mlp_w);

    auto run_mix = [&](const std::vector<const float*>& blocks, const float* base, const float* write,
                       const float* abl_r, const float* abl_w, float* partial_out, const float* queries, int Q,
                       const float* norm_w, ActT* out_bf16, float* out_f32) {
      dense::MixParams p;
      p.blocks = blocks.empty() ? nullptr : block_table(blocks);
      p.nblocks = int(blocks.size());
      p.base = base;
      p.write = write;
      p.abl_r = write ? abl_r : nullptr;
      p.abl_d = write && abl.detect ? abl_r + std::size_t(NUM_VISITS) * H : nullptr;
      p.abl_w = write ? abl_w : nullptr;
      p.partial_out = partial_out;
      p.queries = queries;
      p.Q = Q;
      p.norm_w = norm_w;
      p.out_bf16 = out_bf16;
      p.out_f32 = out_f32;
      p.M = B;
      dense::mix(p, false, stream);
    };

    auto bf16_rows = [&](const Dense& d) -> const ActT* {
      if (!d.fp8()) return bf16p(d.w);
      dense::dequant_rows(d.w.data, f32p(d.scale), d.rows(), d.cols(), wscratch_, stream);
      return wscratch_;
    };

    auto run_gates_gemm = [&](const pk::GateRows& rows, int G, const Tensor& down, const Tensor& forget16,
                              const Tensor& gain16, const float* norm_w, ActT* out) {
      pk::gate_inputs(rows, G, B, gin_, stream);
      for (int g = 0; g < G; ++g) {
        const ActT* gd = bf16p(down) + std::size_t(g) * GATE_RANK * 2 * H;
        gemm_->run(gin_ + std::size_t(g) * B * 2 * H, 2 * H, gd, 2 * H, g16_, GATE_RANK, false, B, GATE_RANK, 2 * H,
                   stream);
        gemm_->run(g16_, GATE_RANK, bf16p(forget16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, fpre_, H, true, B, H,
                   GATE_RANK, stream);
        gemm_->run(g16_, GATE_RANK, bf16p(gain16) + std::size_t(g) * H * GATE_RANK, GATE_RANK, gpre_, H, true, B, H,
                   GATE_RANK, stream);
        pk::gate_apply(rows.h[g], rows.msg[g], fpre_, gpre_, norm_w + std::size_t(g) * H, B,
                       out + std::size_t(g) * slab, stream);
      }
    };

    dense::embed(tokens_dev_, B, bf16p(w.embed), f32p(w.embed_norm), B0_, false, stream);

    int prev_v = -1;
    for (int v = 0; v < NUM_VISITS; ++v) {
      const int l = VISITS[v].layer, p = VISITS[v].pass;
      const Layer& layer = w.layers[l];
      const bool msgs = is_loop_layer(l) && p > 0;
      const bool boundary = (l % BLOCK == 0);
      const std::vector<const float*> blocks = blocks_of(l, p, B0_, B1_, B2_, E_);

      float* hid;
      const float* base;
      const float* write;
      if (v == 0) {
        hid = B0_, base = B0_, write = nullptr;
      } else if (l == LOOP_START && p > 0) {
        hid = B1_, base = B1_, write = nullptr;
      } else {
        hid = l == 6 ? B1_ : l == 12 ? B2_[p] : l == 18 ? E_[2] : hid_;
        base = part_, write = y2_;
      }

      const int pv = prev_v >= 0 ? prev_v : 0;
      const float* r_prev = mlp_dirs + std::size_t(pv) * H;
      const float* w_prev = mlp_w + 4 * std::size_t(pv);
      if (msgs) {
        const int i = l - LOOP_START;
        const float* src_base = i == 0 ? part_ : E_[p - 1];
        const float* src_write = i == 0 ? y2_ : nullptr;
        run_mix({B2_[p - 1]}, src_base, src_write, r_prev, w_prev, i == 0 ? E_[p - 1] : nullptr,
                f32p(w.message_queries) + std::size_t(i) * 4 * H, 4, nullptr, nullptr, msgs_);
        if (i == 0) write = nullptr;
      }

      // ---- Attention read + message gates
      const float* queries = layer.attn_query.data ? f32p(layer.attn_query) : nullptr;
      const int nq = layer.attn_query.data ? layer.attn_query.ne[1] : 1;
      float* partial_out = hid != base ? hid : nullptr;
      if (msgs) {
        run_mix(blocks, base, write, r_prev, w_prev, partial_out, queries, nq, nullptr, nullptr, mixf_);
        if (use_gemv) {
          dense::message_gates(mixf_, !layer.full, msgs_, 3, B, bf16p(layer.qkv_gate_down),
                               layer.qkv_gate_forget16.data, layer.qkv_gate_gain16.data, false, f32p(layer.attn_norm3),
                               xin_, nullptr, false, stream);
        } else {
          pk::GateRows rows;
          for (int g = 0; g < 3; ++g) rows.h[g] = mixf_ + (layer.full ? g : 0) * slab, rows.msg[g] = msgs_ + g * slab;
          run_gates_gemm(rows, 3, layer.qkv_gate_down, layer.qkv_gate_forget16, layer.qkv_gate_gain16,
                         f32p(layer.attn_norm3), xin_);
        }
      } else {
        run_mix(blocks, base, write, r_prev, w_prev, partial_out, l > 0 ? queries : nullptr, l > 0 ? nq : 1,
                f32p(layer.attn_norm_mix), xin_, nullptr);
      }

      // ---- Attention projections
      // For FP8 dense weights at B <= 32, fused chunk-8 GEMV moves ceil(B/8)x W_fp8 vs 5x W_fp8 for dequant+cuBLASLt.
      const int R = proj_rows(layer.full);
      const bool single = !layer.full && !msgs;
      const bool proj_gemv = use_gemv || (layer.wproj.fp8() && B <= 32);
      if (proj_gemv) {
        const DeviceBuffer& seg = layer.full ? full_segments_ : segments_[single ? 0 : 1];
        if (B <= 8) {
          dense::proj(xin_, B, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full), proj_fp32_,
                      false, stream);
        } else {
          for (int b0 = 0; b0 < B; b0 += 8) {
            const int cb = std::min(8, B - b0);
            const ActT* xptr = xin_ + std::size_t(b0) * H;
            if (!single) {
              for (int g = 0; g < 3; ++g) {
                CUDA_CHECK(cudaMemcpyAsync(xin_chunk_ + std::size_t(g) * cb * H,
                                           xin_ + std::size_t(g) * slab + std::size_t(b0) * H,
                                           std::size_t(cb) * H * sizeof(ActT), cudaMemcpyDeviceToDevice, stream));
              }
              xptr = xin_chunk_;
            }
            dense::proj(xptr, cb, matrix(layer.wproj), static_cast<const int*>(seg.p), num_segs(layer.full),
                        proj_fp32_ + std::size_t(b0) * R, false, stream);
          }
        }
      } else {
        const ActT* wp = bf16_rows(layer.wproj);
        if (single) {
          gemm_->run(xin_, H, wp, H, proj_act_, R, false, B, R, H, stream);
        } else {
          struct Part {
            int begin, end, input;
          };
          const Part parts[] = {{SEG_BEGIN[SEG_Q], SEG_BEGIN[SEG_K], 0},
                                {SEG_BEGIN[SEG_K], SEG_BEGIN[SEG_V], 1},
                                {SEG_BEGIN[SEG_V], SEG_BEGIN[SEG_K_DYN], 2},
                                {SEG_BEGIN[SEG_K_DYN], SEG_BEGIN[SEG_V_DYN], 1},
                                {SEG_BEGIN[SEG_V_DYN], SEG_BEGIN[NUM_SEGS], 2}};
          for (const Part& part : parts) {
            if (part.begin >= R) break;
            gemm_->run(xin_ + std::size_t(part.input) * slab, H, wp + std::size_t(part.begin) * H, H,
                       proj_act_ + part.begin, R, false, B, part.end - part.begin, H, stream);
          }
        }
      }

      // ---- Batched per-slot attention (sliding ring or paged full KV)
      bk::BatchedAttnArgs a;
      a.proj = proj_gemv ? static_cast<const void*>(proj_fp32_) : static_cast<const void*>(proj_act_);
      a.proj_fp32 = proj_gemv;
      a.B = B;
      a.slot_ids = slot_ids_dev_;
      a.slot_pos = static_cast<const std::int64_t*>(pool_.slot_pos.p);
      a.q_scale = f32p(layer.q_scale);
      a.r_scale = f32p(layer.r_scale);
      a.rel_proj = f32p(layer.rel_proj);
      a.conv_k_visit = static_cast<float*>(pool_.conv_k.p) + std::size_t(v) * (CONV_K - 1) * KV_DIM;
      a.conv_v_visit = static_cast<float*>(pool_.conv_v.p) + std::size_t(v) * (CONV_K - 1) * KV_DIM;
      a.conv_slot_stride = pool_.conv_kv_slot_stride;
      a.workspace = attn_ws_.p;
      a.workspace_size = attn_ws_.bytes;
      a.out = attn_out_;
      if (layer.full) {
        a.k_conv = f32p(layer.k_static);
        a.v_conv = f32p(layer.v_static);
        a.k_full_pool = pool_.k_full_pool.p;
        a.v_full_pool = pool_.v_full_pool.p;
        a.page_table = static_cast<const std::int32_t*>(pool_.page_table.p);
        a.max_pages_per_slot = pool_.max_pages_per_slot;
        a.full_visit_idx = FULL_INDEX[v];
        a.max_ctx = pool_.max_ctx;
        a.cache_fp8 = pool_.kv_fp8;
        a.window = 0;
        a.log_scaling = true;
      } else {
        const int ri = RING_INDEX[v];
        a.k_conv = f32p(layer.k_conv);
        a.v_conv = f32p(layer.v_conv);
        a.k_ring_visit = static_cast<ActT*>(pool_.k_ring.p) + std::size_t(ri) * WINDOW * KV_DIM;
        a.v_ring_visit = static_cast<ActT*>(pool_.v_ring.p) + std::size_t(ri) * WINDOW * KV_DIM;
        a.ring_pos_visit = static_cast<std::int64_t*>(pool_.ring_pos.p) + std::size_t(ri) * WINDOW;
        a.ring_slot_stride = pool_.ring_slot_stride;
        a.ring_pos_slot_stride = pool_.ring_pos_slot_stride;
        a.cache_fp8 = false;
        a.window = WINDOW;
        a.log_scaling = false;
      }
      bk::batched_attn_decode(a, stream);

      // ---- Output projection + per-slot post-attention 4-tap conv
      float* conv_attn_v = static_cast<float*>(pool_.conv_attn.p) + std::size_t(v) * (CONV_K - 1) * H;
      const bool wo_gemv = use_gemv || (layer.wo.fp8() && B <= 32);
      if (wo_gemv) {
        for (int b0 = 0; b0 < B; b0 += 8) {
          const int cb = std::min(8, B - b0);
          dense::gemv(attn_out_ + std::size_t(b0) * HEADS * HEAD_DIM, cb, HEADS * HEAD_DIM, matrix(layer.wo),
                      wo_fp32_ + std::size_t(b0) * H, false, stream);
        }
        bk::batched_static_conv_step(wo_fp32_, true, H, B, H, slot_ids_dev_, f32p(layer.attn_conv), conv_attn_v,
                                     pool_.conv_h_slot_stride, y_, stream);
      } else {
        const ActT* wo = bf16_rows(layer.wo);
        gemm_->run(attn_out_, HEADS * HEAD_DIM, wo, HEADS * HEAD_DIM, wo_act_, H, false, B, H, HEADS * HEAD_DIM, stream);
        bk::batched_static_conv_step(wo_act_, false, H, B, H, slot_ids_dev_, f32p(layer.attn_conv), conv_attn_v,
                                     pool_.conv_h_slot_stride, y_, stream);
      }

      // ---- MLP read + message gate
      std::vector<const float*> mlp_blocks = blocks;
      if (boundary) mlp_blocks.push_back(hid);
      const float* mbase = boundary ? nullptr : hid;
      if (msgs) {
        run_mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
                f32p(layer.mlp_query), 1, nullptr, nullptr, mixm_);
        if (use_gemv) {
          dense::message_gates(mixm_, false, msgs_ + 3 * slab, 1, B, bf16p(layer.mlp1_gate_down),
                               layer.mlp1_gate_forget16.data, layer.mlp1_gate_gain16.data, false,
                               f32p(layer.mlp_norm1), xm_, nullptr, false, stream);
        } else {
          pk::GateRows rows;
          rows.h[0] = mixm_, rows.msg[0] = msgs_ + 3 * slab;
          run_gates_gemm(rows, 1, layer.mlp1_gate_down, layer.mlp1_gate_forget16, layer.mlp1_gate_gain16,
                         f32p(layer.mlp_norm1), xm_);
        }
      } else {
        run_mix(mlp_blocks, mbase, y_, attn_dirs + std::size_t(v) * H, attn_w + 4 * std::size_t(v), part_,
                f32p(layer.mlp_query), 1, f32p(layer.mlp_norm1), xm_, nullptr);
      }

      // ---- Router + grouped tensor-core MoE + per-slot post-MLP 4-tap conv
      gemm_->run(xm_, H, bf16p(layer.router_bf16), H, router_logits_, EXPERTS, true, B, EXPERTS, H, stream);
      CUDA_CHECK(cudaMemsetAsync(counts_, 0, EXPERTS * sizeof(int), stream));
      pk::moe_topk(router_logits_, f32p(layer.router_bias), B, idx_, wts_, counts_, stream);
      const pk::MoeWork work{counts_, tiles_n_, tiles_, pair_token_, pair_dst_};
      pk::moe_route(idx_, B, work, stream);
      const Experts& ex = layer.experts;
      const pk::MoeExperts me{static_cast<int>(ex.fmt), ex.w13.data, ex.w13_scale.data, f32p(ex.w13_factor),
                              ex.w2.data, ex.w2_scale.data, f32p(ex.w2_factor)};
      pk::moe_gate_up(me, xm_, B, work, act_, stream);
      pk::moe_down(me, act_, B, work, down_, stream);
      pk::moe_combine(down_, wts_, B, moe_sum_, stream);
      float* conv_mlp_v = static_cast<float*>(pool_.conv_mlp.p) + std::size_t(v) * (CONV_K - 1) * H;
      bk::batched_static_conv_step(moe_sum_, false, H, B, H, slot_ids_dev_, f32p(layer.mlp_conv), conv_mlp_v,
                                   pool_.conv_h_slot_stride, y2_, stream);
      prev_v = v;
    }

    // ---- Final read + LM head
    run_mix({B0_, B1_, B2_[2], E_[2]}, part_, y2_, mlp_dirs + std::size_t(prev_v) * H,
            mlp_w + 4 * std::size_t(prev_v), nullptr, f32p(w.final_query), 1, f32p(w.final_norm), xf_, nullptr);
    const bool head_gemv = use_gemv || (w.unembed.fp8() && B <= 32);
    if (head_gemv) {
      for (int b0 = 0; b0 < B; b0 += 8) {
        const int cb = std::min(8, B - b0);
        dense::head(xf_ + std::size_t(b0) * H, cb, matrix(w.unembed), logits_ + std::size_t(b0) * UNPADDED_VOCAB,
                    false, stream);
      }
    } else {
      for (int v0 = 0; v0 < UNPADDED_VOCAB; v0 += kHeadChunk) {
        const int cv = std::min(kHeadChunk, UNPADDED_VOCAB - v0);
        const ActT* wchunk;
        if (w.unembed.fp8()) {
          const auto* q = static_cast<const std::uint8_t*>(w.unembed.w.data) + std::size_t(v0) * H;
          dense::dequant_rows(q, f32p(w.unembed.scale) + v0, cv, H, wscratch_, stream);
          wchunk = wscratch_;
        } else {
          wchunk = bf16p(w.unembed.w) + std::size_t(v0) * H;
        }
        gemm_->run(xf_, H, wchunk, H, head_act_, cv, false, B, cv, H, stream);
        bk::batched_head_softcap(head_act_, B, cv, v0, UNPADDED_VOCAB, logits_, stream);
      }
    }

    // Advance per-slot position on device for active slots
    bk::batched_advance_pos(slot_ids_dev_, B, static_cast<std::int64_t*>(pool_.slot_pos.p), stream);
  }

  const Weights& w_;
  const Ablation& abl_;
  BatchStatePool& pool_;
  int max_batch_;
  std::unique_ptr<pk::Gemm> gemm_;

  std::vector<DeviceBuffer> storage_;
  std::map<std::vector<const float*>, DeviceBuffer> tables_;
  std::array<DeviceBuffer, 2> segments_;
  DeviceBuffer full_segments_;
  DeviceBuffer attn_ws_;
  std::map<int, cudaGraphExec_t> graphs_;
  PinnedHostBuffer tokens_host_, slots_host_;

  std::int64_t* tokens_dev_ = nullptr;
  std::int32_t* slot_ids_dev_ = nullptr;
  float *B0_ = nullptr, *B1_ = nullptr, *B2_[PASSES] = {}, *E_[PASSES] = {};
  float *hid_ = nullptr, *part_ = nullptr, *y_ = nullptr, *y2_ = nullptr;
  ActT *xin_ = nullptr, *xin_chunk_ = nullptr, *xm_ = nullptr, *xf_ = nullptr;
  float *mixf_ = nullptr, *msgs_ = nullptr, *mixm_ = nullptr;
  float* proj_fp32_ = nullptr;
  ActT* proj_act_ = nullptr;
  ActT* attn_out_ = nullptr;
  float* wo_fp32_ = nullptr;
  ActT* wo_act_ = nullptr;
  ActT *gin_ = nullptr, *g16_ = nullptr;
  float *fpre_ = nullptr, *gpre_ = nullptr;
  float* router_logits_ = nullptr;
  int *idx_ = nullptr, *counts_ = nullptr, *tiles_n_ = nullptr, *pair_token_ = nullptr, *pair_dst_ = nullptr;
  int4* tiles_ = nullptr;
  float* wts_ = nullptr;
  ActT *act_ = nullptr, *down_ = nullptr, *moe_sum_ = nullptr;
  float* logits_ = nullptr;
  ActT *head_act_ = nullptr, *wscratch_ = nullptr;
};

// ----------------------------------------------------------------------------- BatchGenerator (implements Generator)

class BatchGenerator final : public Generator {
 public:
  explicit BatchGenerator(const RuntimeOptions& options)
      : options_(options), sampled_host_(std::size_t(std::max(1, options.max_batch)) * sizeof(std::int32_t)) {
    if (options.max_batch < 1) throw std::invalid_argument("max_batch must be >= 1");
    if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be >= 16");
    if (options.kv_q4 || options.kv_paged || options.kv_vram_mb > 0)
      throw std::invalid_argument("continuous batching (--max-batch > 1) supports BF16 or --kv-fp8 KV cache (not --kv-q4/--kv-paged)");
    if (options.offload_vram_mb > 0)
      throw std::invalid_argument("continuous batching (--max-batch > 1) requires resident weights (not --offload-vram-mb/--atlas-6g)");
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    weights_ = load_weights(options.model_dir, options.dense_fp8, stream_);
    ablation_ = load_ablation(options.ablation);
    init();
  }

  BatchGenerator(const RuntimeOptions& options, Weights weights, Ablation ablation)
      : options_(options),
        weights_(std::move(weights)),
        ablation_(std::move(ablation)),
        sampled_host_(std::size_t(std::max(1, options.max_batch)) * sizeof(std::int32_t)) {
    if (options.max_batch < 1) throw std::invalid_argument("max_batch must be >= 1");
    if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be >= 16");
    if (options.kv_q4 || options.kv_paged || options.kv_vram_mb > 0)
      throw std::invalid_argument("continuous batching (--max-batch > 1) supports BF16 or --kv-fp8 KV cache (not --kv-q4/--kv-paged)");
    if (options.offload_vram_mb > 0)
      throw std::invalid_argument("continuous batching (--max-batch > 1) requires resident weights (not --offload-vram-mb/--atlas-6g)");
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    init();
  }

  ~BatchGenerator() override {
    if (stream_) cudaStreamSynchronize(stream_);
    decoder_.reset();
    prefiller_.reset();
    vision_.reset();
    if (stream_) cudaStreamDestroy(stream_);
  }

  std::string model_id() const override {
    return options_.ablation.empty() ? "inkling-2-mini" : "inkling-2-mini-abliterated";
  }

  int max_model_len() const override { return options_.max_ctx; }
  bool supports_images() const override { return vision_ != nullptr; }
  int max_batch_size() const override { return options_.max_batch; }

  bool can_admit_slot(int prompt_tokens, int /*max_new_tokens*/) const override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pool_.free_slot_count() <= 0) return false;
    const int need_pages = (prompt_tokens + 1 + bk::kPageSize - 1) / bk::kPageSize;
    // Keep at least 1 extra page headroom for currently active sequences.
    return int(pool_.free_pages.size()) >= need_pages;
  }

  int start_slot(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                 std::int32_t& first_token, GenerateStats& stats) override {
    const std::span<const std::int32_t> prompt = full_prompt.ids;
    if (prompt.empty()) throw std::invalid_argument("prompt is empty");
    const int n = int(prompt.size());
    const int image_tokens = check_images(full_prompt);
    if (n >= options_.max_ctx) {
      throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                  std::to_string(options_.max_ctx));
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const int need_pages = (n + 1 + bk::kPageSize - 1) / bk::kPageSize;
    if (pool_.free_slot_count() <= 0 || int(pool_.free_pages.size()) < need_pages) return -1;

    const int s = pool_.alloc_slot();
    if (s < 0) return -1;
    if (!pool_.ensure_pages(s, n + 1, stream_)) {
      pool_.free_slot(s);
      return -1;
    }

    auto t0 = std::chrono::steady_clock::now();
    try {
      State st = pool_.prefill_view(s);
      reset_state(st, stream_);

      const std::size_t chunk = std::size_t(prefill_chunk_);
      struct Encoded {
        const PromptImage* image;
        DeviceBuffer rows;
      };
      std::deque<Encoded> encoded;
      std::size_t next_image = 0;
      std::vector<EmbedRows> rows;

      for (std::size_t start = 0; start < std::size_t(n); start += chunk) {
        const int T = int(std::min(chunk, std::size_t(n) - start));
        for (; next_image < full_prompt.images.size() &&
               std::size_t(full_prompt.images[next_image].pos) < start + std::size_t(T);
             ++next_image) {
          const PromptImage& image = full_prompt.images[next_image];
          const auto tv = std::chrono::steady_clock::now();
          DeviceBuffer out = vision_->encode(image.rgb->data(), image.height, image.width);
          while (!encoded.empty() &&
                 std::size_t(encoded.front().image->pos + encoded.front().image->tokens()) <= start) {
            encoded.pop_front();
          }
          encoded.push_back(Encoded{&image, std::move(out)});
          stats.vision_seconds += seconds_since(tv);
          ++stats.images_encoded;
        }
        rows.clear();
        for (const Encoded& e : encoded) {
          const std::int64_t first = std::max<std::int64_t>(e.image->pos, std::int64_t(start));
          const std::int64_t last =
              std::min<std::int64_t>(std::int64_t(e.image->pos) + e.image->tokens(), std::int64_t(start) + T);
          if (first < last) {
            rows.push_back(EmbedRows{int(first - std::int64_t(start)), int(last - first),
                                     static_cast<const ActT*>(e.rows.p) + std::size_t(first - e.image->pos) * H});
          }
        }
        CUDA_CHECK(cudaMemcpyAsync(chunk_tokens_.p, prompt.data() + start, std::size_t(T) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream_));
        prefiller_->run(st, static_cast<const std::int32_t*>(chunk_tokens_.p), T, int(start),
                        static_cast<float*>(prefill_logits_.p), nullptr, rows);
      }
      if (!encoded.empty()) CUDA_CHECK(cudaStreamSynchronize(stream_));

      // Scatter prefilled full-attention KV rows [0, n) into slot s's physical pages.
      bk::scatter_full_kv(pool_.scratch_k_full.p, pool_.scratch_v_full.p, pool_.max_ctx, 0, n, pool_.kv_fp8, s,
                          static_cast<const std::int32_t*>(pool_.page_table.p), pool_.max_pages_per_slot,
                          pool_.k_full_pool.p, pool_.v_full_pool.p, stream_);
      pool_.set_slot_pos(s, n, stream_);

      // Initialize per-slot sampling state and sample first_token.
      SlotMeta& meta = slot_meta_[s];
      meta.params = to_head_params(sampling);
      meta.rng.seed =
          sampling.seed ? *sampling.seed : (std::uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
      meta.rng.counter = 0;
      meta.forced.assign(sampling.forced.begin(), sampling.forced.end());
      meta.forced_idx = 0;
      meta.max_pos = std::min(options_.max_ctx, n + std::max(0, max_new_tokens));

      auto* sampled = static_cast<std::int32_t*>(sampled_dev_.p);
      head::sample(static_cast<const float*>(prefill_logits_.p), 1, UNPADDED_VOCAB, UNPADDED_VOCAB, meta.params,
                   meta.rng, sampled, stream_);
      CUDA_CHECK(cudaMemcpyAsync(sampled_host_.data(), sampled, sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      first_token = static_cast<const std::int32_t*>(sampled_host_.data())[0];
      if (!meta.forced.empty()) {
        first_token = meta.forced[0];
        meta.forced_idx = 1;
      }

      stats.prompt_tokens = n;
      stats.reused_tokens = 0;
      stats.reuse = Reuse::None;
      stats.image_tokens = image_tokens;
      stats.prefill_seconds = seconds_since(t0);
      return s;
    } catch (...) {
      cudaStreamSynchronize(stream_);
      pool_.free_slot(s);
      throw;
    }
  }

  void step_slots(std::span<const BatchSlotStep> inputs, std::span<std::int32_t> out_tokens) override {
    const int B = int(inputs.size());
    if (B == 0) return;
    if (int(out_tokens.size()) < B) throw std::invalid_argument("step_slots: out_tokens too small");

    std::lock_guard<std::mutex> lock(mutex_);
    // Ensure every active slot has a physical page for its current decode position `slot_pos_host[s]`.
    for (int i = 0; i < B; ++i) {
      const int s = inputs[i].slot;
      if (s < 0 || s >= options_.max_batch || !pool_.slot_in_use[s])
        throw std::invalid_argument("step_slots: invalid or inactive slot");
      const int pos = pool_.slot_pos_host[s];
      if (pos >= options_.max_ctx) throw std::runtime_error("step_slots: slot reached max_ctx");
      if (!pool_.ensure_pages(s, pos + 1, stream_))
        throw std::runtime_error("step_slots: paged KV pool exhausted");
    }

    const float* logits = decoder_->run_step(inputs, stream_);
    for (int i = 0; i < B; ++i) pool_.slot_pos_host[inputs[i].slot] += 1;

    // Fast path: all greedy with identical banned tokens -> single batched argmax launch.
    bool all_greedy = true;
    const head::BannedTokens& b0 = slot_meta_[inputs[0].slot].params.banned;
    for (int i = 0; i < B; ++i) {
      const head::SamplingParams& sp = slot_meta_[inputs[i].slot].params;
      if (sp.temperature > 0.0f || !same_banned(sp.banned, b0)) {
        all_greedy = false;
        break;
      }
    }

    auto* sampled = static_cast<std::int32_t*>(sampled_dev_.p);
    if (all_greedy) {
      head::argmax(logits, B, UNPADDED_VOCAB, UNPADDED_VOCAB, sampled, stream_, b0);
    } else {
      for (int i = 0; i < B; ++i) {
        SlotMeta& meta = slot_meta_[inputs[i].slot];
        head::sample(logits + std::size_t(i) * UNPADDED_VOCAB, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, meta.params,
                     meta.rng, sampled + i, stream_);
      }
    }

    CUDA_CHECK(cudaMemcpyAsync(sampled_host_.data(), sampled, std::size_t(B) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    const auto* host_out = static_cast<const std::int32_t*>(sampled_host_.data());
    for (int i = 0; i < B; ++i) {
      SlotMeta& meta = slot_meta_[inputs[i].slot];
      std::int32_t tok = host_out[i];
      if (meta.forced_idx < meta.forced.size()) tok = meta.forced[meta.forced_idx++];
      out_tokens[i] = tok;
    }
  }

  void release_slot(int slot) override {
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.free_slot(slot);
  }

  GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    return generate(Prompt{prompt, {}}, max_new_tokens, sampling, stop_ids, on_tokens, PrefillCallback{});
  }

  GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override {
    return generate(full_prompt, max_new_tokens, sampling, stop_ids, on_tokens, PrefillCallback{});
  }

  GenerateStats generate(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                         const PrefillCallback& on_prefill) override {
    GenerateStats stats;
    std::int32_t token = 0;
    const int n = int(full_prompt.ids.size());
    if (!sampling.forced.empty()) max_new_tokens = std::min(max_new_tokens, int(sampling.forced.size()));
    if (n + max_new_tokens > options_.max_ctx) {
      max_new_tokens = options_.max_ctx - n;
      if (max_new_tokens <= 0)
        throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                    std::to_string(options_.max_ctx));
    }
    const int s = start_slot(full_prompt, max_new_tokens, sampling, token, stats);
    if (s < 0) throw std::runtime_error("BatchGenerator::generate: no free batch slot");
    if (on_prefill) on_prefill(stats);
    auto is_stop = [&](std::int32_t tok) {
      return std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end();
    };
    const auto t0 = std::chrono::steady_clock::now();
    try {
      for (int i = 0; i < max_new_tokens; ++i) {
        stats.generated_tokens = i + 1;
        const std::int32_t out[1] = {token};
        const bool go_on = on_tokens(out);
        if (i + 1 == max_new_tokens || is_stop(token) || !go_on) break;
        const BatchSlotStep step[1] = {{s, token}};
        std::int32_t next[1] = {0};
        step_slots(step, next);
        token = next[0];
      }
    } catch (...) {
      release_slot(s);
      throw;
    }
    stats.decode_seconds = seconds_since(t0);
    release_slot(s);
    return stats;
  }

 private:
  struct SlotMeta {
    head::SamplingParams params{};
    head::SamplerRng rng{};
    std::vector<std::int32_t> forced;
    std::size_t forced_idx = 0;
    int max_pos = 0;
  };

  void init() {
    prefill_chunk_ = std::min(options_.prefill_chunk, options_.max_ctx);
    if (prefill_chunk_ < 1) throw std::invalid_argument("prefill_chunk must be positive");
    if (options_.vision) vision_ = std::make_unique<VisionEncoder>(options_.model_dir, stream_);
    prefiller_ = std::make_unique<Prefiller>(weights_, ablation_, prefill_chunk_, stream_);
    chunk_tokens_ = DeviceBuffer(std::size_t(prefill_chunk_) * sizeof(std::int32_t));
    prefill_logits_ = DeviceBuffer(std::size_t(UNPADDED_VOCAB) * sizeof(float));
    sampled_dev_ = DeviceBuffer(std::size_t(options_.max_batch) * sizeof(std::int32_t));

    pool_.init(options_.max_batch, options_.max_ctx, options_.kv_fp8, options_.max_kv_tokens);
    slot_meta_.resize(options_.max_batch);

    decoder_ = std::make_unique<BatchDecoder>(weights_, ablation_, pool_, options_.max_batch);
    decoder_->capture_buckets(stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

  int check_images(const Prompt& prompt) const {
    if (!prompt.images.empty() && !vision_) {
      throw std::invalid_argument("the prompt has images but the vision tower is not loaded (RuntimeOptions::vision)");
    }
    const std::int64_t n = std::int64_t(prompt.ids.size());
    std::int64_t end = 0, tokens = 0;
    for (const PromptImage& image : prompt.images) {
      if (!image.rgb || image.height < 1 || image.width < 1 ||
          image.rgb->size() != std::size_t(image.height) * std::size_t(image.width) * 3) {
        throw std::invalid_argument("a prompt image has no pixels or pixels of another size");
      }
      if (image.pos < end || std::int64_t(image.pos) + image.tokens() > n) {
        throw std::invalid_argument("the prompt's images overlap, are out of order or lie outside the prompt");
      }
      end = std::int64_t(image.pos) + image.tokens();
      tokens += image.tokens();
    }
    const std::int64_t negative =
        std::count_if(prompt.ids.begin(), prompt.ids.end(), [](std::int32_t id) { return id < 0; });
    bool covered = negative == tokens;
    for (const PromptImage& image : prompt.images) {
      for (int k = 0; covered && k < image.tokens(); ++k) covered = prompt.ids[std::size_t(image.pos + k)] < 0;
    }
    if (!covered) throw std::invalid_argument("negative token ids of the prompt are not exactly its image tokens");
    return int(tokens);
  }

  RuntimeOptions options_;
  cudaStream_t stream_ = nullptr;
  Weights weights_;
  Ablation ablation_;
  int prefill_chunk_ = 3072;
  std::unique_ptr<VisionEncoder> vision_;
  std::unique_ptr<Prefiller> prefiller_;
  BatchStatePool pool_;
  std::unique_ptr<BatchDecoder> decoder_;
  DeviceBuffer chunk_tokens_, prefill_logits_, sampled_dev_;
  PinnedHostBuffer sampled_host_;
  std::vector<SlotMeta> slot_meta_;
  mutable std::mutex mutex_;
};

}  // namespace

std::unique_ptr<Generator> make_batch_generator(const RuntimeOptions& options) {
  return std::make_unique<BatchGenerator>(options);
}

std::unique_ptr<Generator> make_batch_generator(const RuntimeOptions& options, Weights weights, Ablation ablation) {
  return std::make_unique<BatchGenerator>(options, std::move(weights), std::move(ablation));
}

}  // namespace ingine::model

// Model runtime: weights + ablation + state + decoder, decode graphs per call width (engine.Engine._capture),
// state snapshots and speculative rollback (engine.Engine._snapshot/_restore/_rollback), pipelined width-1
// steps and the switch graphs of device-selected verify widths.
#include "model/runtime.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ingine::model {
namespace {

constexpr int kMaxWidth = 8;
constexpr std::array<int, 3> kVerifyWidths = {2, 4, 8};

DeviceBuffer device_copy(const Tensor& t, cudaStream_t stream) {
  DeviceBuffer b(t.bytes());
  CUDA_CHECK(cudaMemcpyAsync(b.p, t.data, t.bytes(), cudaMemcpyDeviceToDevice, stream));
  return b;
}

// Carves consecutive 256-byte aligned pieces out of one allocation.
struct Carve {
  char* p;
  template <class T>
  T* take(std::size_t count) {
    T* r = reinterpret_cast<T*>(p);
    p += (count * sizeof(T) + 255) / 256 * 256;
    return r;
  }
};
constexpr std::size_t carved(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

constexpr std::size_t kUndoBytes = carved(std::size_t(NUM_VISITS) * CONV_ROW_FLOATS * 4) +
                                   2 * carved(std::size_t(NUM_SLIDING_VISITS) * KV_DIM * 2) +
                                   carved(std::size_t(NUM_SLIDING_VISITS) * 8);
constexpr std::size_t kBackupBytes = 2 * carved(std::size_t(NUM_VISITS) * 3 * KV_DIM * 4) +
                                     2 * carved(std::size_t(NUM_VISITS) * 3 * HIDDEN * 4) +
                                     2 * carved(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth * KV_DIM * 2) +
                                     carved(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth * 8);

}  // namespace

GraphExec::~GraphExec() {
  if (exec_) cudaGraphExecDestroy(exec_);
}

GraphExec& GraphExec::operator=(GraphExec&& o) noexcept {
  if (this != &o) {
    if (exec_) cudaGraphExecDestroy(exec_);
    exec_ = o.exec_;
    o.exec_ = nullptr;
  }
  return *this;
}

void GraphExec::launch(cudaStream_t stream) const {
  if (!exec_) throw std::logic_error("GraphExec::launch: empty graph");
  CUDA_CHECK(cudaGraphLaunch(exec_, stream));
}

Runtime::Runtime(const RuntimeOptions& options) : options_(options) {
  if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be at least 16");
  if (options_.kv_paged || options_.kv_vram_mb > 0) {
    options_.kv_paged = true;
    options_.kv_q4 = true;
    options_.kv_fp8 = false;
    if (options_.kv_vram_mb <= 0) {
      options_.kv_vram_mb = options_.offload_vram_mb > 0 ? 256 : 1536;
    }
  }
  if (options_.offload_vram_mb > 0 && options_.prefill_chunk > 1024) {
    options_.prefill_chunk = 1024;  // aliased Prefiller phase arena fits 1024-token chunks (~315 MB) inside 6 GB caps
  }
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  OffloadConfig offload;
  offload.vram_mb = options_.offload_vram_mb;
  offload.ram_mb = options_.offload_ram_mb;
  offload.mode = options_.offload_mode;
  offload.ranking = options_.expert_ranking;
  offload.dump_router_stats = options_.dump_router_stats;
  offload.max_ctx = options_.max_ctx;
  offload.prefill_chunk = options_.prefill_chunk;
  offload.kv_fp8 = options_.kv_fp8;
  offload.kv_q4 = options_.kv_q4;
  offload.spec = options_.spec;
  offload.kv_vram_mb = options_.kv_paged ? options_.kv_vram_mb : 0;
  weights_ = load_weights(options_.model_dir, options_.dense_fp8, stream_, offload);
  ablation_ = load_ablation(options_.ablation);
  init();
}

Runtime::Runtime(const RuntimeOptions& options, Weights weights, Ablation ablation)
    : options_(options), weights_(std::move(weights)), ablation_(std::move(ablation)) {
  if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be at least 16");
  // Decoder/Prefiller always dereference the ablation buffers (zeroed when inactive); materialize an
  // empty device ablation if the caller passed a default-constructed Ablation{}.
  if (!ablation_.active && (!ablation_.attn_dirs.data || !ablation_.attn_w.data || !ablation_.mlp_dirs.data ||
                            !ablation_.mlp_w.data)) {
    ablation_ = load_ablation({});
  }
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  init();
}

void Runtime::init() {
  const RuntimeOptions& options = options_;
  PagedKvConfig pkv;
  if (options.kv_paged || options.kv_vram_mb > 0) {
    pkv.max_ctx = options.max_ctx;
    pkv.vram_mb = options.kv_vram_mb > 0 ? options.kv_vram_mb : (options.offload_vram_mb > 0 ? 256 : 1536);
    pkv.ram_mb = options.kv_ram_mb;
    pkv.prefill_chunk = options.prefill_chunk;
    pkv.swap_dir = options.kv_swap_dir;
  }
  state_ = make_state(options.max_ctx, options.kv_fp8, options.kv_q4, pkv);
  if (state_.paged_kv) {
    state_.paged_kv->ensure_decode_pages(0, kMaxWidth, stream_);
  }
  decoder_ = std::make_unique<Decoder>(weights_, ablation_, state_, true);
  persona_rt_ = std::make_unique<PersonaRuntime>();
  pos_ = DeviceBuffer(sizeof(std::int64_t));
  pos_.fill(0);
  token_ = DeviceBuffer(kMaxWidth * sizeof(std::int64_t));
  token_.fill(0);
  CUDA_CHECK(cudaStreamCreateWithFlags(&capture_stream_, cudaStreamNonBlocking));
  for (cudaEvent_t& e : chain_done_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  chain_host_ = MappedHostBuffer(2 * sizeof(std::int32_t));
  for (int i = 0; i < 2; ++i) {
    undo_storage_[i] = DeviceBuffer(kUndoBytes);
    Carve c{static_cast<char*>(undo_storage_[i].p)};
    undo_[i].conv = c.take<float>(std::size_t(NUM_VISITS) * CONV_ROW_FLOATS);
    undo_[i].k = c.take<ActT>(std::size_t(NUM_SLIDING_VISITS) * KV_DIM);
    undo_[i].v = c.take<ActT>(std::size_t(NUM_SLIDING_VISITS) * KV_DIM);
    undo_[i].pos = c.take<std::int64_t>(NUM_SLIDING_VISITS);
  }

  buffers_.emplace(1, make_buffers(1));
  history_[1] = false;
  if (options.spec) {
    for (int w : kVerifyWidths) {
      buffers_.emplace(w, make_buffers(w));
      history_[w] = true;
      verify_widths_.push_back(w);
    }
    backup_storage_ = DeviceBuffer(kBackupBytes);
    Carve c{static_cast<char*>(backup_storage_.p)};
    for (int i = 0; i < 4; ++i) backup_.conv[i] = c.take<float>(std::size_t(NUM_VISITS) * 3 * CONV_COLS[i]);
    backup_.k = c.take<ActT>(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth * KV_DIM);
    backup_.v = c.take<ActT>(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth * KV_DIM);
    backup_.pos = c.take<std::int64_t>(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth);
  }
  // every width reads its tokens from the shared buffer (set_inputs, chain_advance and the verify selector
  // write one place)
  for (auto& [m, b] : buffers_) b.token = rm(token_.p, DType::I64, {m});
  const bool eager_offload = weights_.expert_cache && weights_.expert_cache->needs_host_dispatch();
  if (!eager_offload) {
    for (const auto& [m, history] : history_) capture(m, history);
    if (options_.enable_persona_hooks) {
      persona_rt_->upload_zero_table(stream_);
      for (const auto& [m, history] : history_) capture_hooked(m, history);
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

Runtime::~Runtime() {
  if (stream_) cudaStreamSynchronize(stream_);
  for (auto& [m, exec] : graphs_) cudaGraphExecDestroy(exec);
  graphs_.clear();
  for (auto& [m, exec] : hooked_graphs_) cudaGraphExecDestroy(exec);
  hooked_graphs_.clear();
  persona_rt_.reset();
  decoder_.reset();
  for (cudaEvent_t e : chain_done_)
    if (e) cudaEventDestroy(e);
  if (capture_stream_) cudaStreamDestroy(capture_stream_);
  if (stream_) cudaStreamDestroy(stream_);
}

bool Runtime::has_width(int m) const { return buffers_.count(m) != 0; }

std::span<const int> Runtime::verify_widths() const { return verify_widths_; }

bool Runtime::needs_eager_step() const noexcept {
  return (weights_.expert_cache && weights_.expert_cache->needs_host_dispatch()) ||
         (state_.paged_kv && state_.paged_kv->needs_streamed_decode());
}

bool Runtime::has_verify_graph() const noexcept {
  return !(weights_.expert_cache && weights_.expert_cache->needs_host_dispatch());
}

Buffers& Runtime::buffers(int m) {
  auto it = buffers_.find(m);
  if (it == buffers_.end()) throw std::invalid_argument("no decode buffers for width " + std::to_string(m));
  return it->second;
}

StatePtrs Runtime::state_ptrs() const {
  StatePtrs p;
  p.conv[0] = ptr<float>(state_.conv_k);
  p.conv[1] = ptr<float>(state_.conv_v);
  p.conv[2] = ptr<float>(state_.conv_attn);
  p.conv[3] = ptr<float>(state_.conv_mlp);
  p.k_ring = ptr<ActT>(state_.k_ring);
  p.v_ring = ptr<ActT>(state_.v_ring);
  p.ring_pos = ptr<std::int64_t>(state_.ring_pos);
  return p;
}

void Runtime::capture(int m, bool history) {
  // warm up so the pointer tables and kernel attributes are set outside capture; the scratch writes to the
  // conv states and ring positions are undone afterwards (the cache rows it wrote are never valid)
  const Snapshot saved = snapshot();
  Buffers& b = buffers(m);
  for (int i = 0; i < 2; ++i) decoder_->decode(b, pos(), history, stream_);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  try {
    decoder_->decode(b, pos(), history, stream_);
  } catch (...) {
    cudaStreamEndCapture(stream_, &graph);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
  CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
  cudaGraphExec_t exec = nullptr;
  const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  CUDA_CHECK(err);
  CUDA_CHECK(cudaGraphUpload(exec, stream_));
  graphs_[m] = exec;
  restore(saved);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Runtime::capture_hooked(int m, bool history) {
  const Snapshot saved = snapshot();
  Buffers& b = buffers(m);
  const auto* tbl = persona_rt_->device_table();
  for (int i = 0; i < 2; ++i) decoder_->decode(b, pos(), history, stream_, tbl);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  try {
    decoder_->decode(b, pos(), history, stream_, tbl);
  } catch (...) {
    cudaStreamEndCapture(stream_, &graph);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
  CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
  cudaGraphExec_t exec = nullptr;
  const cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
  cudaGraphDestroy(graph);
  CUDA_CHECK(err);
  CUDA_CHECK(cudaGraphUpload(exec, stream_));
  hooked_graphs_[m] = exec;
  restore(saved);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Runtime::ensure_hooked_graphs() {
  if (!hooked_graphs_.empty()) return;
  if ((weights_.expert_cache && weights_.expert_cache->needs_host_dispatch()) ||
      (state_.paged_kv && state_.paged_kv->needs_streamed_decode())) {
    return;
  }
  // Losslessly preserve live state, pos_, token_, and full-attention rows [0..kMaxWidth) so lazy capture
  // mid-session cannot corrupt any cached or parked conversation.
  DeviceBuffer saved_ckpt(checkpoint_bytes());
  DeviceBuffer saved_full;
  if (!state_.paged_kv) saved_full = DeviceBuffer(full_rows_bytes(kMaxWidth));
  DeviceBuffer saved_pos(pos_.bytes);
  DeviceBuffer saved_tok(token_.bytes);
  save_checkpoint(saved_ckpt.p);
  if (saved_full.p) save_full_rows(saved_full.p, kMaxWidth);
  CUDA_CHECK(cudaMemcpyAsync(saved_pos.p, pos_.p, pos_.bytes, cudaMemcpyDeviceToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(saved_tok.p, token_.p, token_.bytes, cudaMemcpyDeviceToDevice, stream_));
  CUDA_CHECK(cudaMemsetAsync(pos_.p, 0, pos_.bytes, stream_));
  CUDA_CHECK(cudaMemsetAsync(token_.p, 0, token_.bytes, stream_));
  persona_rt_->upload_zero_table(stream_);
  for (const auto& [m, history] : history_) capture_hooked(m, history);
  restore_checkpoint(saved_ckpt.p);
  if (saved_full.p) restore_full_rows(saved_full.p, kMaxWidth);
  CUDA_CHECK(cudaMemcpyAsync(pos_.p, saved_pos.p, pos_.bytes, cudaMemcpyDeviceToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(token_.p, saved_tok.p, token_.bytes, cudaMemcpyDeviceToDevice, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

void Runtime::set_persona(const persona::PersonaProfile* profile, const PersonaTapConfig& taps) {
  const bool want_active = (profile != nullptr && !profile->is_unhooked()) || !taps.empty();
  if (want_active && hooked_graphs_.empty()) ensure_hooked_graphs();
  persona_rt_->configure(profile, taps, stream_);
}

void Runtime::set_inputs(std::span<const std::int32_t> tokens, int position) {
  const int m = int(tokens.size());
  buffers(m);  // the width must exist
  if (state_.paged_kv) state_.paged_kv->ensure_decode_pages(position, m, stream_);
  set_step_inputs(tokens.data(), m, position, this->tokens(), pos(), stream_);
}

const float* Runtime::run_step(int m) {
  if ((weights_.expert_cache && weights_.expert_cache->needs_host_dispatch()) ||
      (state_.paged_kv && state_.paged_kv->needs_streamed_decode())) {
    return run_step_eager(m);
  }
  const auto& map = hooked_active() ? hooked_graphs_ : graphs_;
  auto it = map.find(m);
  if (it == map.end()) throw std::invalid_argument("no decode graph for width " + std::to_string(m));
  CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  return static_cast<const float*>(buffers(m).logits.data);
}

const float* Runtime::run_step_eager(int m) {
  Buffers& b = buffers(m);
  decoder_->decode(b, pos(), history_.at(m), stream_, active_hooks());
  return static_cast<const float*>(b.logits.data);
}

const float* Runtime::decode_logits(std::int32_t token, int position) {
  const std::int32_t t[1] = {token};
  set_inputs(t, position);
  return run_step(1);
}

void Runtime::chain_advance(const std::int32_t* sampled, int position) {
  const int k = position & 1;
  if (state_.paged_kv) state_.paged_kv->ensure_decode_pages(position, 1, stream_);
  model::chain_advance(sampled, tokens(), pos(), position, static_cast<std::int32_t*>(chain_host_.device_ptr()) + k,
                       state_ptrs(), undo_[k], stream_);
  CUDA_CHECK(cudaEventRecord(chain_done_[k], stream_));
}

std::int32_t Runtime::chain_token(int position) {
  const int k = position & 1;
  CUDA_CHECK(cudaEventSynchronize(chain_done_[k]));
  return static_cast<const volatile std::int32_t*>(chain_host_.data())[k];
}

void Runtime::chain_undo(int position) { model::chain_undo(state_ptrs(), undo_[position & 1], position, stream_); }

Runtime::Snapshot Runtime::snapshot() {
  Snapshot s;
  for (const Tensor* t : {&state_.ring_pos, &state_.conv_k, &state_.conv_v, &state_.conv_attn, &state_.conv_mlp})
    s.tensors.push_back(device_copy(*t, stream_));
  return s;
}

void Runtime::restore(const Snapshot& snap) {
  const Tensor* ts[5] = {&state_.ring_pos, &state_.conv_k, &state_.conv_v, &state_.conv_attn, &state_.conv_mlp};
  for (int i = 0; i < 5; ++i)
    CUDA_CHECK(cudaMemcpyAsync(ts[i]->data, snap.tensors[i].p, ts[i]->bytes(), cudaMemcpyDeviceToDevice, stream_));
}

namespace {
// the state tensors a prompt checkpoint holds, in checkpoint order
std::array<const Tensor*, 7> checkpoint_tensors(const State& st) {
  return {&st.k_ring, &st.v_ring, &st.ring_pos, &st.conv_k, &st.conv_v, &st.conv_attn, &st.conv_mlp};
}
}  // namespace

std::size_t Runtime::checkpoint_bytes() const {
  std::size_t n = 0;
  for (const Tensor* t : checkpoint_tensors(state_)) n += carved(t->bytes());
  return n;
}

std::size_t Runtime::save_checkpoint(void* dst) {
  char* p = static_cast<char*>(dst);
  std::size_t copied = 0;
  for (const Tensor* t : checkpoint_tensors(state_)) {
    CUDA_CHECK(cudaMemcpyAsync(p, t->data, t->bytes(), cudaMemcpyDefault, stream_));
    copied += t->bytes();
    p += carved(t->bytes());
  }
  return copied;
}

std::size_t Runtime::restore_checkpoint(const void* src) {
  const char* p = static_cast<const char*>(src);
  std::size_t copied = 0;
  for (const Tensor* t : checkpoint_tensors(state_)) {
    CUDA_CHECK(cudaMemcpyAsync(t->data, p, t->bytes(), cudaMemcpyDefault, stream_));
    copied += t->bytes();
    p += carved(t->bytes());
  }
  return copied;
}

std::size_t Runtime::full_rows_bytes(int n) const {
  if (n < 0 || n > options_.max_ctx)
    throw std::invalid_argument("full_rows_bytes: n = " + std::to_string(n) + " outside [0, max_ctx = " +
                                std::to_string(options_.max_ctx) + "]");
  const std::size_t row = state_.k_full.bytes() / NUM_FULL_VISITS / std::size_t(options_.max_ctx);
  return 2 * std::size_t(NUM_FULL_VISITS) * row * std::size_t(n);
}

// Copy NUM_FULL_VISITS visits of n positions packed contiguously (k visits, then v visits).
// Per-visit 1D copies avoid CUDA's 2 GiB cudaMemcpy2DAsync pitch ceiling at >=1M context lengths.
std::size_t Runtime::save_full_rows(void* dst, int n) {
  const std::size_t bytes = full_rows_bytes(n) / 2, width = bytes / NUM_FULL_VISITS;
  if (n == 0) return 0;
  const std::size_t pitch = state_.k_full.bytes() / NUM_FULL_VISITS;
  char* p = static_cast<char*>(dst);
  std::size_t copied = 0;
  for (const Tensor* t : {&state_.k_full, &state_.v_full}) {
    const char* src = static_cast<const char*>(t->data);
    for (int f = 0; f < NUM_FULL_VISITS; ++f) {
      CUDA_CHECK(cudaMemcpyAsync(p + std::size_t(f) * width, src + std::size_t(f) * pitch, width, cudaMemcpyDefault,
                                 stream_));
      copied += width;
    }
    p += bytes;
  }
  return copied;
}

std::size_t Runtime::restore_full_rows(const void* src, int n) {
  const std::size_t total = full_rows_bytes(n);
  const std::size_t bytes = total / 2, width = bytes / NUM_FULL_VISITS;
  if (n == 0) return 0;
  const std::size_t pitch = state_.k_full.bytes() / NUM_FULL_VISITS;
  const char* p = static_cast<const char*>(src);
  std::size_t copied = 0;
  for (const Tensor* t : {&state_.k_full, &state_.v_full}) {
    char* dst = static_cast<char*>(t->data);
    for (int f = 0; f < NUM_FULL_VISITS; ++f) {
      CUDA_CHECK(cudaMemcpyAsync(dst + std::size_t(f) * pitch, p + std::size_t(f) * width, width, cudaMemcpyDefault,
                                 stream_));
      copied += width;
    }
    p += bytes;
  }
  return copied;
}

void Runtime::save_verify(int P, int m) {
  if (verify_widths_.empty()) throw std::logic_error("save_verify needs the speculative widths (options.spec)");
  verify_save(state_ptrs(), backup_, nullptr, P, m, stream_);
}

void Runtime::rollback(int P, int commit, int m) {
  if (commit == m) return;
  if (commit < 1 || commit > m) throw std::invalid_argument("rollback: commit out of range");
  const Buffers& b = buffers(m);
  const float* hist[4] = {ptr<float>(b.hist_k), ptr<float>(b.hist_v), ptr<float>(b.hist_attn), ptr<float>(b.hist_mlp)};
  verify_rollback(state_ptrs(), backup_, hist, nullptr, P, nullptr, commit, m, stream_);
}

void Runtime::record_step(int m, cudaStream_t stream, bool hooked) {
  if (!graphs_.count(m)) throw std::invalid_argument("record_step: no decode graph for width " + std::to_string(m));
  decoder_->decode(buffers(m), pos(), history_.at(m), stream, hooked ? persona_rt_->device_table() : nullptr);
}

void Runtime::record_save_verify(int m, cudaStream_t stream) {
  if (verify_widths_.empty()) throw std::logic_error("record_save_verify needs the speculative widths (options.spec)");
  verify_save(state_ptrs(), backup_, pos(), 0, m, stream);
}

void Runtime::record_rollback(int m, const std::int32_t* accepted, cudaStream_t stream) {
  if (!history_.count(m) || !history_.at(m)) throw std::invalid_argument("record_rollback: width without history");
  const Buffers& b = buffers(m);
  const float* hist[4] = {ptr<float>(b.hist_k), ptr<float>(b.hist_v), ptr<float>(b.hist_attn), ptr<float>(b.hist_mlp)};
  verify_rollback(state_ptrs(), backup_, hist, pos(), 0, accepted, 0, m, stream);
}

GraphExec Runtime::capture_switch(const std::function<void(cudaStream_t, cudaGraphConditionalHandle)>& head,
                                  const std::function<void(cudaStream_t, int)>& branch) {
  if (verify_widths_.empty()) throw std::logic_error("capture_switch needs the speculative widths (options.spec)");
  const cudaStream_t s = capture_stream_;
  cudaGraph_t graph = nullptr;
  CUDA_CHECK(cudaGraphCreate(&graph, 0));
  struct Owned {
    cudaGraph_t g;
    ~Owned() { cudaGraphDestroy(g); }
  } owned{graph};
  // run `body` captured into `g` (an exception ends the capture first)
  auto captured = [&](cudaGraph_t g, const std::function<void()>& body) {
    CUDA_CHECK(cudaStreamBeginCaptureToGraph(s, g, nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
    try {
      body();
    } catch (...) {
      cudaGraph_t out = nullptr;
      cudaStreamEndCapture(s, &out);
      throw;
    }
    cudaGraph_t out = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(s, &out));
  };

  cudaGraphConditionalHandle handle;
  CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, 0));
  cudaGraph_t* bodies = nullptr;
  captured(graph, [&] {
    head(s, handle);
    cudaStreamCaptureStatus status;
    cudaGraph_t g = nullptr;
    const cudaGraphNode_t* deps = nullptr;
    std::size_t n = 0;
#if CUDART_VERSION >= 13000
    const cudaGraphEdgeData* edges = nullptr;
    CUDA_CHECK(cudaStreamGetCaptureInfo(s, &status, nullptr, &g, &deps, &edges, &n));
#else
    CUDA_CHECK(cudaStreamGetCaptureInfo(s, &status, nullptr, &g, &deps, &n));
#endif
    cudaGraphNodeParams p{};
    p.type = cudaGraphNodeTypeConditional;
    p.conditional.handle = handle;
    p.conditional.type = cudaGraphCondTypeSwitch;
    p.conditional.size = unsigned(verify_widths_.size());
    cudaGraphNode_t node = nullptr;
#if CUDART_VERSION >= 13000
    CUDA_CHECK(cudaGraphAddNode(&node, g, deps, edges, n, &p));
    bodies = p.conditional.phGraph_out;
    CUDA_CHECK(cudaStreamUpdateCaptureDependencies(s, &node, nullptr, 1, cudaStreamSetCaptureDependencies));
#else
    CUDA_CHECK(cudaGraphAddNode(&node, g, deps, n, &p));
    bodies = p.conditional.phGraph_out;
    CUDA_CHECK(cudaStreamUpdateCaptureDependencies(s, &node, 1, cudaStreamSetCaptureDependencies));
#endif
  });
  for (std::size_t i = 0; i < verify_widths_.size(); ++i)
    captured(bodies[i], [&] { branch(s, verify_widths_[i]); });

  cudaGraphExec_t exec = nullptr;
  CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
  GraphExec result(exec);
  CUDA_CHECK(cudaGraphUpload(exec, stream_));
  return result;
}

}  // namespace ingine::model

// Model runtime: weights + ablation + state + decoder, decode graphs per call width (engine.Engine._capture),
// state snapshots and speculative rollback (engine.Engine._snapshot/_restore/_rollback).
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

// Contiguous ring-slot runs of the call rows [r0, r1) at positions P + r: (first slot, first row, count).
struct SlotRun {
  int slot, row, count;
};
std::vector<SlotRun> slot_runs(int P, int r0, int r1) {
  std::vector<SlotRun> runs;
  int r = r0;
  while (r < r1) {
    const int slot = (P + r) % WINDOW;
    const int n = std::min(r1 - r, WINDOW - slot);
    runs.push_back({slot, r, n});
    r += n;
  }
  return runs;
}

}  // namespace

Runtime::Runtime(const RuntimeOptions& options) : options_(options) {
  if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be at least 16");
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  weights_ = load_weights(options.model_dir, options.dense_fp8, stream_);
  ablation_ = load_ablation(options.ablation);
  init();
}

Runtime::Runtime(const RuntimeOptions& options, Weights weights, Ablation ablation)
    : options_(options), weights_(std::move(weights)), ablation_(std::move(ablation)) {
  if (options.max_ctx < 16) throw std::invalid_argument("max_ctx must be at least 16");
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  init();
}

void Runtime::init() {
  const RuntimeOptions& options = options_;
  state_ = make_state(options.max_ctx, options.kv_fp8);
  decoder_ = std::make_unique<Decoder>(weights_, ablation_, state_, true);
  pos_ = DeviceBuffer(sizeof(std::int64_t));
  pos_.fill(0);

  buffers_.emplace(1, make_buffers(1));
  history_[1] = false;
  if (options.spec) {
    for (int w : kVerifyWidths) {
      buffers_.emplace(w, make_buffers(w));
      history_[w] = true;
      verify_widths_.push_back(w);
    }
    const std::size_t kv_conv = std::size_t(NUM_VISITS) * 3 * KV_DIM * 4;
    const std::size_t h_conv = std::size_t(NUM_VISITS) * 3 * HIDDEN * 4;
    old_conv_[0] = DeviceBuffer(kv_conv);
    old_conv_[1] = DeviceBuffer(kv_conv);
    old_conv_[2] = DeviceBuffer(h_conv);
    old_conv_[3] = DeviceBuffer(h_conv);
    ring_k_backup_ = DeviceBuffer(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth * KV_DIM * 2);
    ring_v_backup_ = DeviceBuffer(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth * KV_DIM * 2);
    ring_pos_backup_ = DeviceBuffer(std::size_t(NUM_SLIDING_VISITS) * kMaxWidth * sizeof(std::int64_t));
  }
  for (const auto& [m, history] : history_) capture(m, history);
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

Runtime::~Runtime() {
  if (stream_) cudaStreamSynchronize(stream_);
  for (auto& [m, exec] : graphs_) cudaGraphExecDestroy(exec);
  graphs_.clear();
  decoder_.reset();
  if (stream_) cudaStreamDestroy(stream_);
}

bool Runtime::has_width(int m) const { return buffers_.count(m) != 0; }

std::span<const int> Runtime::verify_widths() const { return verify_widths_; }

Buffers& Runtime::buffers(int m) {
  auto it = buffers_.find(m);
  if (it == buffers_.end()) throw std::invalid_argument("no decode buffers for width " + std::to_string(m));
  return it->second;
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

void Runtime::set_inputs(std::span<const std::int32_t> tokens, int position) {
  const int m = int(tokens.size());
  Buffers& b = buffers(m);
  std::array<std::int64_t, kMaxWidth> t{};
  for (int i = 0; i < m; ++i) t[i] = tokens[i];
  const std::int64_t p = position;
  // pageable source: the call returns once the bytes are staged, so the locals may go out of scope
  CUDA_CHECK(cudaMemcpyAsync(b.token.data, t.data(), std::size_t(m) * 8, cudaMemcpyHostToDevice, stream_));
  CUDA_CHECK(cudaMemcpyAsync(pos_.p, &p, 8, cudaMemcpyHostToDevice, stream_));
}

const float* Runtime::run_step(int m) {
  auto it = graphs_.find(m);
  if (it == graphs_.end()) throw std::invalid_argument("no decode graph for width " + std::to_string(m));
  CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  return static_cast<const float*>(buffers(m).logits.data);
}

const float* Runtime::run_step_eager(int m) {
  Buffers& b = buffers(m);
  decoder_->decode(b, pos(), history_.at(m), stream_);
  return static_cast<const float*>(b.logits.data);
}

const float* Runtime::decode_logits(std::int32_t token, int position) {
  const std::int32_t t[1] = {token};
  set_inputs(t, position);
  return run_step(1);
}

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

void Runtime::save_verify(int P, int m) {
  if (verify_widths_.empty()) throw std::logic_error("save_verify needs the speculative widths (options.spec)");
  if (m < 1 || m > kMaxWidth) throw std::invalid_argument("save_verify: width out of range");
  const Tensor* conv[4] = {&state_.conv_k, &state_.conv_v, &state_.conv_attn, &state_.conv_mlp};
  for (int i = 0; i < 4; ++i)
    CUDA_CHECK(cudaMemcpyAsync(old_conv_[i].p, conv[i]->data, conv[i]->bytes(), cudaMemcpyDeviceToDevice, stream_));
  // ring_backup = (k_ring[:, slots], v_ring[:, slots], ring_pos[:, slots]) for slots (P + i) % WINDOW
  const std::size_t row = std::size_t(KV_DIM) * 2;
  for (const SlotRun& r : slot_runs(P, 0, m)) {
    for (auto [ring, backup] : {std::pair{&state_.k_ring, &ring_k_backup_}, std::pair{&state_.v_ring, &ring_v_backup_}})
      CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(backup->p) + std::size_t(r.row) * row, kMaxWidth * row,
                                   static_cast<const char*>(ring->data) + std::size_t(r.slot) * row, WINDOW * row,
                                   std::size_t(r.count) * row, NUM_SLIDING_VISITS, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::int64_t*>(ring_pos_backup_.p) + r.row, kMaxWidth * 8,
                                 static_cast<const std::int64_t*>(state_.ring_pos.data) + r.slot, WINDOW * 8,
                                 std::size_t(r.count) * 8, NUM_SLIDING_VISITS, cudaMemcpyDeviceToDevice, stream_));
  }
}

void Runtime::rollback(int P, int commit, int m) {
  if (commit == m) return;
  if (commit < 1 || commit > m) throw std::invalid_argument("rollback: commit out of range");
  Buffers& b = buffers(m);
  // conv state = cat([old, hist], dim=1)[:, commit:commit + 3]
  struct Conv {
    const Tensor* state;
    const DeviceBuffer* old;
    const Tensor* hist;
    int C;
  };
  const Conv convs[4] = {{&state_.conv_k, &old_conv_[0], &b.hist_k, KV_DIM},
                         {&state_.conv_v, &old_conv_[1], &b.hist_v, KV_DIM},
                         {&state_.conv_attn, &old_conv_[2], &b.hist_attn, HIDDEN},
                         {&state_.conv_mlp, &old_conv_[3], &b.hist_mlp, HIDDEN}};
  for (const Conv& c : convs) {
    const std::size_t row = std::size_t(c.C) * 4;
    for (int j = 0; j < 3; ++j) {
      const int i = commit + j;
      const char* src;
      std::size_t spitch;
      if (i < 3) {
        src = static_cast<const char*>(c.old->p) + std::size_t(i) * row;
        spitch = 3 * row;
      } else {
        src = static_cast<const char*>(c.hist->data) + std::size_t(i - 3) * row;
        spitch = std::size_t(m) * row;
      }
      CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(c.state->data) + std::size_t(j) * row, 3 * row, src, spitch, row,
                                   NUM_VISITS, cudaMemcpyDeviceToDevice, stream_));
    }
  }
  // ring slots of the rejected rows back to their saved contents
  const std::size_t row = std::size_t(KV_DIM) * 2;
  for (const SlotRun& r : slot_runs(P, commit, m)) {
    for (auto [ring, backup] : {std::pair{&state_.k_ring, &ring_k_backup_}, std::pair{&state_.v_ring, &ring_v_backup_}})
      CUDA_CHECK(cudaMemcpy2DAsync(static_cast<char*>(ring->data) + std::size_t(r.slot) * row, WINDOW * row,
                                   static_cast<const char*>(backup->p) + std::size_t(r.row) * row, kMaxWidth * row,
                                   std::size_t(r.count) * row, NUM_SLIDING_VISITS, cudaMemcpyDeviceToDevice, stream_));
    CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::int64_t*>(state_.ring_pos.data) + r.slot, WINDOW * 8,
                                 static_cast<const std::int64_t*>(ring_pos_backup_.p) + r.row, kMaxWidth * 8,
                                 std::size_t(r.count) * 8, NUM_SLIDING_VISITS, cudaMemcpyDeviceToDevice, stream_));
  }
}

}  // namespace ingine::model

#include "backends/qwen4exp/qwen4exp_gpu.h"
#include "backends/qwen4exp/qwen4exp_cpu_ops.h"
#include "backends/qwen4exp/qwen4exp_cuda_ops.h"
#include "backends/qwen4exp/qwen4exp_elastic_safetensors.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <list>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace ingine::backends::qwen4exp {
namespace {

using ingine::model::ActivationDtype;
using ingine::model::AdmissionEstimate;
using ingine::model::BackendCapabilities;
using ingine::model::BackendRegistry;
using ingine::model::BackendRegistration;
using ingine::model::BackendRequest;
using ingine::model::BackendSession;
using ingine::model::BackendStateIdentity;
using ingine::model::GenerateStats;
using ingine::model::LogitCallback;
using ingine::model::LogitStep;
using ingine::model::ModelBackend;
using ingine::model::ModelFamily;
using ingine::model::PrefillCallback;
using ingine::model::PrefixCacheStats;
using ingine::model::Prompt;
using ingine::model::ResidencyReport;
using ingine::model::Reuse;
using ingine::model::RuntimeOptions;
using ingine::model::Sampling;
using ingine::model::SessionLookupQuery;
using ingine::model::SessionLookupResult;

struct DevBuf {
  void* p = nullptr;
  std::size_t bytes = 0;
  DevBuf() = default;
  explicit DevBuf(std::size_t b) : bytes(b) {
    if (bytes > 0) CUDA_CHECK(cudaMalloc(&p, bytes));
  }
  ~DevBuf() { reset(); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : p(o.p), bytes(o.bytes) {
    o.p = nullptr;
    o.bytes = 0;
  }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) {
      reset();
      p = o.p;
      bytes = o.bytes;
      o.p = nullptr;
      o.bytes = 0;
    }
    return *this;
  }
  void reset() noexcept {
    if (p) {
      cudaFree(p);
      p = nullptr;
      bytes = 0;
    }
  }
  void zero(cudaStream_t stream) {
    if (p && bytes > 0) CUDA_CHECK(cudaMemsetAsync(p, 0, bytes, stream));
  }
  float* f32() noexcept { return static_cast<float*>(p); }
  const float* f32() const noexcept { return static_cast<const float*>(p); }
  std::uint8_t* u8() noexcept { return static_cast<std::uint8_t*>(p); }
  const std::uint8_t* u8() const noexcept { return static_cast<const std::uint8_t*>(p); }
  std::int32_t* i32() noexcept { return static_cast<std::int32_t*>(p); }
  const std::int32_t* i32() const noexcept { return static_cast<const std::int32_t*>(p); }
};

struct DevQuantWeight {
  enum class Kind : std::uint8_t { Ggml = 0, Fp8 = 1, Nvfp4 = 2, Atlas3 = 3, Atlas2 = 4 };
  Kind kind = Kind::Ggml;
  DevBuf buf;
  DevBuf scales;
  float scale = 1.0f;
  std::uint32_t type_id = 0;
  int in_features = 0;
  int out_features = 0;
};

struct DevLayerWeights {
  bool is_linear = true;
  bool has_ple = false;
  bool has_qsa = false;

  // Hyper-Connections
  DevBuf hc_attn_norm, hc_ffn_norm;
  DevQuantWeight hc_attn_down, hc_attn_up, hc_attn_inject;
  DevQuantWeight hc_ffn_down, hc_ffn_up, hc_ffn_inject;

  // PLE (layer 1)
  DevBuf ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv1d;
  DevQuantWeight ple_key, ple_value;

  // LinearAttention (GDN)
  DevQuantWeight attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out;
  DevBuf ssm_conv1d, ssm_dt_bias, ssm_a, ssm_norm;

  // FullAttention + QSA
  DevQuantWeight attn_q, attn_k, attn_v, attn_output;
  DevBuf attn_q_norm, attn_k_norm;
  DevQuantWeight idx_q_proj, idx_k_proj;
  DevBuf idx_q_norm, idx_k_norm;

  // MoE router + shared expert
  DevQuantWeight ffn_gate_inp;
  DevBuf ffn_gate_inp_shexp;
  DevQuantWeight ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;

  // Routed MoE experts metadata (GGUF or Elastic Safetensors) + optional resident VRAM tensors
  bool elastic_moe = false;
  ElasticWeightFormat gate_moe_fmt = ElasticWeightFormat::Nvfp4;
  ElasticWeightFormat up_moe_fmt = ElasticWeightFormat::Nvfp4;
  ElasticWeightFormat down_moe_fmt = ElasticWeightFormat::Nvfp4;
  const GgufTensorEntry* gate_exps_entry = nullptr;
  const GgufTensorEntry* up_exps_entry = nullptr;
  const GgufTensorEntry* down_exps_entry = nullptr;
  const SafetensorsEntry* e_gate_w = nullptr;
  const SafetensorsEntry* e_gate_s = nullptr;
  const SafetensorsEntry* e_up_w = nullptr;
  const SafetensorsEntry* e_up_s = nullptr;
  const SafetensorsEntry* e_down_w = nullptr;
  const SafetensorsEntry* e_down_s = nullptr;
  DevBuf gate_exps_dev;
  DevBuf up_exps_dev;
  DevBuf down_exps_dev;
  DevBuf gate_exps_scales_dev;
  DevBuf up_exps_scales_dev;
  DevBuf down_exps_scales_dev;
  std::size_t gate_exp_bytes = 0;
  std::size_t up_exp_bytes = 0;
  std::size_t down_exp_bytes = 0;
  std::size_t gate_code_bytes = 0, gate_scale_bytes = 0;
  std::size_t up_code_bytes = 0, up_scale_bytes = 0;
  std::size_t down_code_bytes = 0, down_scale_bytes = 0;
  std::vector<float> gate_scale2, up_scale2, down_scale2;
};

// Direct-mapped / LRU GPU cache for quantized routed MoE experts (key = (layer << 16) | expert_id).
class GpuExpertLruCache {
 public:
  void init(std::size_t max_slots, std::size_t max_gate_bytes, std::size_t max_up_bytes, std::size_t max_down_bytes) {
    slot_stride_ = max_gate_bytes + max_up_bytes + max_down_bytes;
    gate_off_ = 0;
    up_off_ = max_gate_bytes;
    down_off_ = max_gate_bytes + max_up_bytes;
    capacity_ = max_slots;
    if (capacity_ > 0 && slot_stride_ > 0) {
      pool_ = DevBuf(capacity_ * slot_stride_);
      CUDA_CHECK(cudaMallocHost(&pinned_staging_, slot_stride_));
    }
  }

  ~GpuExpertLruCache() {
    if (pinned_staging_) {
      cudaFreeHost(pinned_staging_);
      pinned_staging_ = nullptr;
    }
  }

  struct ExpertPtrs {
    const void* gate = nullptr;
    const void* up = nullptr;
    const void* down = nullptr;
  };

  ExpertPtrs fetch(const Qwen4ExpGgufSet& gguf, int layer_idx, int expert_id, const DevLayerWeights& L,
                   cudaStream_t stream) {
    const std::uint64_t key = (static_cast<std::uint64_t>(layer_idx) << 16) | static_cast<std::uint64_t>(expert_id);
    if (const auto it = map_.find(key); it != map_.end()) {
      ++hits_;
      lru_.splice(lru_.begin(), lru_, it->second.lru_it);
      return slot_ptrs(it->second.slot);
    }
    ++misses_;

    std::size_t slot = 0;
    if (used_ < capacity_) {
      slot = used_++;
      lru_.push_front(key);
    } else {
      const std::uint64_t evict_key = lru_.back();
      lru_.pop_back();
      slot = map_.at(evict_key).slot;
      map_.erase(evict_key);
      lru_.push_front(key);
    }
    map_.emplace(key, Entry{slot, lru_.begin()});

    // Read only this single expert's gate/up/down quantized rows via cached-fd POSIX pread into pinned staging,
    // then H2D copy directly into the GPU cache slot.
    const auto ff_e = static_cast<std::uint64_t>(L.gate_exps_entry->dims[1]);
    const auto H = static_cast<std::uint64_t>(L.down_exps_entry->dims[1]);
    gguf.read_tensor_rows_raw(*L.gate_exps_entry, static_cast<std::uint64_t>(expert_id) * ff_e, ff_e,
                              std::span<std::uint8_t>(pinned_staging_ + gate_off_, L.gate_exp_bytes));
    gguf.read_tensor_rows_raw(*L.up_exps_entry, static_cast<std::uint64_t>(expert_id) * ff_e, ff_e,
                              std::span<std::uint8_t>(pinned_staging_ + up_off_, L.up_exp_bytes));
    gguf.read_tensor_rows_raw(*L.down_exps_entry, static_cast<std::uint64_t>(expert_id) * H, H,
                              std::span<std::uint8_t>(pinned_staging_ + down_off_, L.down_exp_bytes));

    std::uint8_t* dst = pool_.u8() + slot * slot_stride_;
    CUDA_CHECK(cudaMemcpyAsync(dst + gate_off_, pinned_staging_ + gate_off_, L.gate_exp_bytes, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(dst + up_off_, pinned_staging_ + up_off_, L.up_exp_bytes, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(dst + down_off_, pinned_staging_ + down_off_, L.down_exp_bytes, cudaMemcpyHostToDevice,
                               stream));
    // Synchronize staging buffer before next expert overwrites pinned_staging_.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return slot_ptrs(slot);
  }

  ExpertPtrs fetch_elastic(const Qwen4ExpElasticSet& eset, int layer_idx, int expert_id, const DevLayerWeights& L,
                           cudaStream_t stream) {
    const std::uint64_t key = (static_cast<std::uint64_t>(layer_idx) << 16) | static_cast<std::uint64_t>(expert_id);
    if (const auto it = map_.find(key); it != map_.end()) {
      ++hits_;
      lru_.splice(lru_.begin(), lru_, it->second.lru_it);
      return slot_ptrs(it->second.slot);
    }
    ++misses_;

    std::size_t slot = 0;
    if (used_ < capacity_) {
      slot = used_++;
      lru_.push_front(key);
    } else {
      const std::uint64_t evict_key = lru_.back();
      lru_.pop_back();
      slot = map_.at(evict_key).slot;
      map_.erase(evict_key);
      lru_.push_front(key);
    }
    map_.emplace(key, Entry{slot, lru_.begin()});

    const auto e = static_cast<std::uint64_t>(expert_id);
    eset.read_tensor_bytes(*L.e_gate_w, e * L.gate_code_bytes,
                           std::span<std::uint8_t>(pinned_staging_ + gate_off_, L.gate_code_bytes));
    eset.read_tensor_bytes(*L.e_gate_s, e * L.gate_scale_bytes,
                           std::span<std::uint8_t>(pinned_staging_ + gate_off_ + L.gate_code_bytes, L.gate_scale_bytes));
    eset.read_tensor_bytes(*L.e_up_w, e * L.up_code_bytes,
                           std::span<std::uint8_t>(pinned_staging_ + up_off_, L.up_code_bytes));
    eset.read_tensor_bytes(*L.e_up_s, e * L.up_scale_bytes,
                           std::span<std::uint8_t>(pinned_staging_ + up_off_ + L.up_code_bytes, L.up_scale_bytes));
    eset.read_tensor_bytes(*L.e_down_w, e * L.down_code_bytes,
                           std::span<std::uint8_t>(pinned_staging_ + down_off_, L.down_code_bytes));
    eset.read_tensor_bytes(*L.e_down_s, e * L.down_scale_bytes,
                           std::span<std::uint8_t>(pinned_staging_ + down_off_ + L.down_code_bytes, L.down_scale_bytes));

    std::uint8_t* dst = pool_.u8() + slot * slot_stride_;
    CUDA_CHECK(cudaMemcpyAsync(dst + gate_off_, pinned_staging_ + gate_off_, L.gate_exp_bytes, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(dst + up_off_, pinned_staging_ + up_off_, L.up_exp_bytes, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(dst + down_off_, pinned_staging_ + down_off_, L.down_exp_bytes, cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return slot_ptrs(slot);
  }

  std::uint64_t clear() {
    const std::uint64_t evicted = used_ * slot_stride_;
    map_.clear();
    lru_.clear();
    used_ = 0;
    return evicted;
  }

  [[nodiscard]] std::uint64_t bytes() const noexcept { return pool_.bytes; }
  [[nodiscard]] Qwen4ExpExpertCacheStats stats() const noexcept { return {hits_, misses_, capacity_, used_}; }

 private:
  struct Entry {
    std::size_t slot = 0;
    std::list<std::uint64_t>::iterator lru_it;
  };

  ExpertPtrs slot_ptrs(std::size_t slot) const noexcept {
    const std::uint8_t* base = pool_.u8() + slot * slot_stride_;
    return {base + gate_off_, base + up_off_, base + down_off_};
  }

  DevBuf pool_;
  std::uint8_t* pinned_staging_ = nullptr;
  std::size_t slot_stride_ = 0;
  std::size_t gate_off_ = 0;
  std::size_t up_off_ = 0;
  std::size_t down_off_ = 0;
  std::size_t capacity_ = 0;
  std::size_t used_ = 0;
  std::list<std::uint64_t> lru_;
  std::unordered_map<std::uint64_t, Entry> map_;
  std::uint64_t hits_ = 0;
  std::uint64_t misses_ = 0;
};

std::int32_t sample_logits_cpu(std::span<float> logits, const Sampling& sampling, std::mt19937_64& rng) {
  for (std::int32_t ban : sampling.banned_tokens) {
    if (ban >= 0 && static_cast<std::size_t>(ban) < logits.size()) {
      logits[static_cast<std::size_t>(ban)] = -std::numeric_limits<float>::infinity();
    }
  }
  if (sampling.temperature <= 0.0f || sampling.top_k == 1) {
    return static_cast<std::int32_t>(
        std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
  }

  const float inv_t = 1.0f / sampling.temperature;
  std::vector<int> idx(logits.size());
  std::iota(idx.begin(), idx.end(), 0);
  const std::size_t k = (sampling.top_k > 0 && static_cast<std::size_t>(sampling.top_k) < idx.size())
                            ? static_cast<std::size_t>(sampling.top_k)
                            : idx.size();
  if (k < idx.size()) {
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(k);
  } else {
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return logits[a] > logits[b]; });
  }

  const float max_l = logits[static_cast<std::size_t>(idx[0])];
  std::vector<double> probs(idx.size());
  double sum = 0.0;
  for (std::size_t i = 0; i < idx.size(); ++i) {
    probs[i] = std::exp(static_cast<double>((logits[static_cast<std::size_t>(idx[i])] - max_l) * inv_t));
    sum += probs[i];
  }
  double cum = 0.0;
  std::size_t keep = idx.size();
  if (sampling.top_p > 0.0f && sampling.top_p < 1.0f) {
    for (std::size_t i = 0; i < probs.size(); ++i) {
      cum += probs[i] / sum;
      if (cum >= sampling.top_p) {
        keep = i + 1;
        break;
      }
    }
  }
  double keep_sum = 0.0;
  for (std::size_t i = 0; i < keep; ++i) keep_sum += probs[i];
  std::uniform_real_distribution<double> u(0.0, keep_sum);
  double r = u(rng);
  for (std::size_t i = 0; i < keep; ++i) {
    r -= probs[i];
    if (r <= 0.0) return static_cast<std::int32_t>(idx[i]);
  }
  return static_cast<std::int32_t>(idx[keep - 1]);
}

}  // namespace

struct Qwen4ExpGpuSession::StateImpl {
  BackendStateIdentity identity;
  std::vector<std::int32_t> committed_tokens;
  std::vector<std::int32_t> ple_token_history;

  std::vector<DevBuf> ple_conv_states;
  std::vector<DevBuf> gdn_conv_states;
  std::vector<DevBuf> gdn_recurrent_states;
  std::vector<DevBuf> k_caches;
  std::vector<DevBuf> v_caches;
  std::vector<DevBuf> idx_k_caches;
  std::vector<float> last_logits;
  bool has_last_logits = false;
  std::size_t total_state_bytes = 0;
};

struct Qwen4ExpGpuBackend::Impl {
  Qwen4ExpGpuLoadOptions opts;
  bool is_elastic = false;
  std::optional<Qwen4ExpGgufSet> gguf;
  std::optional<Qwen4ExpElasticSet> elastic;
  cudaStream_t stream = nullptr;

  // Global dense weights
  DevBuf output_hc_norm;
  DevQuantWeight output_hc_down, output_hc_up, output_weight;
  const GgufTensorEntry* tok_embd_entry = nullptr;
  const GgufTensorEntry* ple_table_entry = nullptr;
  std::vector<DevLayerWeights> layers;
  GpuExpertLruCache expert_cache;

  // Reusable GPU activation scratch buffers (single-token forward)
  DevBuf d_emb, d_res_a, d_res_b, d_xn, d_hc_lo, d_hc_gate, d_cur, d_inject, d_block_out;
  DevBuf d_ple_emb, d_ple_key_raw, d_ple_val, d_ple_key, d_ple_q, d_ple_gated, d_ple_norm;
  DevBuf d_qkv, d_z, d_alpha, d_beta, d_conv_tmp, d_gdn_out;
  DevBuf d_iq, d_ik, d_block_scores, d_sel_pos, d_qg, d_kr, d_vr, d_q_sc, d_g_sc, d_attn_out;
  DevBuf d_r_logits, d_r_exp, d_r_w, d_ff_g, d_ff_u, d_ff_act, d_e_out, d_sh_out;
  DevBuf d_logits;

  std::unique_ptr<Qwen4ExpGpuSession> default_session;
  PrefixCacheStats cache_stats{};
  mutable std::mutex mu;

  explicit Impl(const Qwen4ExpGpuLoadOptions& o) : opts(o) {
    if (Qwen4ExpElasticSet::is_elastic_dir(o.model_dir)) {
      is_elastic = true;
      elastic = Qwen4ExpElasticSet::inspect_headers(o.model_dir);
    } else {
      is_elastic = false;
      gguf = (o.verify_pins && o.expected_pins.has_value())
                 ? Qwen4ExpGgufSet::inspect_pinned(o.model_dir, *o.expected_pins)
                 : Qwen4ExpGgufSet::inspect_headers(o.model_dir);
    }
  }

  const Qwen4ExpConfig& cfg() const noexcept { return is_elastic ? elastic->config() : gguf->config(); }
  const ingine::model::CheckpointIdentity& ckpt_id() const noexcept {
    return is_elastic ? elastic->identity() : gguf->identity();
  }
  std::uint64_t total_tensor_bytes() const noexcept {
    return is_elastic ? elastic->total_tensor_bytes() : gguf->total_tensor_bytes();
  }

  ~Impl() {
    if (stream) cudaStreamDestroy(stream);
  }
};

Qwen4ExpGpuSession::Qwen4ExpGpuSession(Qwen4ExpGpuBackend& owner) : owner_(owner), state_(std::make_unique<StateImpl>()) {
  const auto& c = owner_.config();
  const int max_ctx = owner_.impl_->opts.max_ctx;
  state_->identity = owner_.state_identity();
  state_->ple_conv_states.resize(static_cast<std::size_t>(c.num_hidden_layers));
  state_->gdn_conv_states.resize(static_cast<std::size_t>(c.num_hidden_layers));
  state_->gdn_recurrent_states.resize(static_cast<std::size_t>(c.num_hidden_layers));
  state_->k_caches.resize(static_cast<std::size_t>(c.num_hidden_layers));
  state_->v_caches.resize(static_cast<std::size_t>(c.num_hidden_layers));
  state_->idx_k_caches.resize(static_cast<std::size_t>(c.num_hidden_layers));

  std::size_t bytes = 0;
  for (int il = 0; il < c.num_hidden_layers; ++il) {
    if (c.is_ple(il)) {
      const std::size_t b = static_cast<std::size_t>(c.ple_conv_history_cols()) * c.hc_dim() * sizeof(float);
      state_->ple_conv_states[il] = DevBuf(b);
      bytes += b;
    }
    if (c.is_linear(il)) {
      const std::size_t cb = static_cast<std::size_t>(c.ssm_conv_kernel) * c.gdn_conv_dim() * sizeof(float);
      const std::size_t rb =
          static_cast<std::size_t>(c.ssm_time_step_rank) * c.ssm_state_size * c.ssm_state_size * sizeof(float);
      state_->gdn_conv_states[il] = DevBuf(cb);
      state_->gdn_recurrent_states[il] = DevBuf(rb);
      bytes += cb + rb;
    } else {
      const std::size_t kvb = static_cast<std::size_t>(max_ctx) * c.full_kv_out_dim() * sizeof(float);
      state_->k_caches[il] = DevBuf(kvb);
      state_->v_caches[il] = DevBuf(kvb);
      bytes += 2 * kvb;
      if (c.has_qsa(il)) {
        const std::size_t ib = static_cast<std::size_t>(max_ctx) * c.indexer_key_length * sizeof(float);
        state_->idx_k_caches[il] = DevBuf(ib);
        bytes += ib;
      }
    }
  }
  state_->total_state_bytes = bytes;
  reset();
}

Qwen4ExpGpuSession::~Qwen4ExpGpuSession() = default;

void Qwen4ExpGpuSession::reset() {
  const auto& c = owner_.config();
  cudaStream_t s = owner_.impl_->stream;
  state_->committed_tokens.clear();
  state_->has_last_logits = false;
  state_->last_logits.clear();
  if (c.ple_ngram_size >= 2) {
    state_->ple_token_history.assign(static_cast<std::size_t>(c.ple_ngram_size - 1), c.ple_eos_token_id);
  } else {
    state_->ple_token_history.clear();
  }
  for (auto& b : state_->ple_conv_states) b.zero(s);
  for (auto& b : state_->gdn_conv_states) b.zero(s);
  for (auto& b : state_->gdn_recurrent_states) b.zero(s);
  for (auto& b : state_->k_caches) b.zero(s);
  for (auto& b : state_->v_caches) b.zero(s);
  for (auto& b : state_->idx_k_caches) b.zero(s);
  CUDA_CHECK(cudaStreamSynchronize(s));
}

std::size_t Qwen4ExpGpuSession::state_bytes() const noexcept { return state_->total_state_bytes; }
std::span<const std::int32_t> Qwen4ExpGpuSession::committed_tokens() const noexcept { return state_->committed_tokens; }
std::span<const std::uint8_t> Qwen4ExpGpuSession::committed_token_mask() const noexcept { return {}; }
const BackendStateIdentity& Qwen4ExpGpuSession::identity() const noexcept { return state_->identity; }

void Qwen4ExpGpuSession::truncate(std::size_t new_token_count) {
  if (new_token_count == 0) {
    reset();
    return;
  }
  if (new_token_count == state_->committed_tokens.size()) return;
  throw std::logic_error("Qwen4ExpGpuSession: arbitrary mid-sequence truncate not supported without checkpoint");
}

std::size_t Qwen4ExpGpuSession::restorable_prefix_len(std::size_t max_prefix_len) const noexcept {
  const std::size_t tip = state_->committed_tokens.size();
  return (tip > 0 && max_prefix_len >= tip) ? tip : 0;
}

bool Qwen4ExpGpuSession::try_restore_prefix(std::size_t token_count) {
  return token_count > 0 && token_count == state_->committed_tokens.size();
}

Qwen4ExpGpuBackend::Qwen4ExpGpuBackend(const Qwen4ExpGpuLoadOptions& options) {
  const auto t0 = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaSetDevice(options.device_id));
  impl_ = std::make_unique<Impl>(options);
  CUDA_CHECK(cudaStreamCreate(&impl_->stream));

  const auto& c = impl_->cfg();
  const bool is_elastic = impl_->is_elastic;
  load_stats_.pins_verified = impl_->ckpt_id().shard_digests_verified;

  auto is_supported_gpu_ggml_type = [](std::uint32_t tid) noexcept {
    switch (tid) {
      case 0:
      case 1:
      case 2:
      case 7:
      case 8:
      case 12:
      case 13:
      case 14:
      case 30: return true;
      default: return false;
    }
  };
  auto to_elastic_weight_format = [](ElasticTensorKind kind, const std::string& ctx) -> ElasticWeightFormat {
    if (kind == ElasticTensorKind::Nvfp4) return ElasticWeightFormat::Nvfp4;
    if (kind == ElasticTensorKind::Atlas3) return ElasticWeightFormat::Atlas3;
    if (kind == ElasticTensorKind::Atlas2) return ElasticWeightFormat::Atlas2;
    throw std::invalid_argument("Qwen4ExpGpuBackend: unsupported packed Elastic format on " + ctx);
  };

  auto upload_f32 = [&](const std::string& name, std::initializer_list<std::int64_t> expected_shape) -> DevBuf {
    std::vector<std::int64_t> sh(expected_shape.begin(), expected_shape.end());
    std::uint64_t expected_numel = 1;
    for (std::int64_t d : sh) {
      if (d <= 0) throw std::invalid_argument("Qwen4ExpGpuBackend: invalid expected shape on " + name);
      expected_numel *= static_cast<std::uint64_t>(d);
    }
    std::vector<float> v;
    if (is_elastic) {
      const auto& e = impl_->elastic->validate_f32_tensor(name, sh);
      v = impl_->elastic->read_tensor_f32(e);
    } else {
      const auto& e = impl_->gguf->require_tensor(name);
      if (e.numel() != expected_numel) {
        throw std::invalid_argument("Qwen4ExpGpuBackend: GGUF tensor numel mismatch on " + name);
      }
      v = impl_->gguf->read_tensor_f32(e);
    }
    if (v.size() != expected_numel) {
      throw std::invalid_argument("Qwen4ExpGpuBackend: decoded F32 count mismatch on " + name);
    }
    DevBuf b(v.size() * sizeof(float));
    CUDA_CHECK(cudaMemcpy(b.p, v.data(), b.bytes, cudaMemcpyHostToDevice));
    load_stats_.dense_device_bytes += b.bytes;
    return b;
  };
  auto upload_quant = [&](const std::string& name, int expected_out, int expected_in) -> DevQuantWeight {
    if (expected_out <= 0 || expected_in <= 0) {
      throw std::invalid_argument("Qwen4ExpGpuBackend: invalid expected linear shape on " + name);
    }
    if (!is_elastic) {
      const auto& e = impl_->gguf->require_tensor(name);
      if (!is_supported_gpu_ggml_type(e.type_id)) {
        throw std::invalid_argument("Qwen4ExpGpuBackend: unsupported GPU GGML type_id " + std::to_string(e.type_id) +
                                    " on " + name);
      }
      if (e.dims.size() != 2 || e.dims[0] != static_cast<std::uint64_t>(expected_in) ||
          e.dims[1] != static_cast<std::uint64_t>(expected_out)) {
        throw std::invalid_argument("Qwen4ExpGpuBackend: GGUF linear shape mismatch on " + name);
      }
      const std::uint64_t expected_bytes = static_cast<std::uint64_t>(expected_out) * e.row_bytes();
      if (e.byte_size != expected_bytes) {
        throw std::invalid_argument("Qwen4ExpGpuBackend: GGUF linear byte size mismatch on " + name);
      }
      const auto raw = impl_->gguf->read_tensor_raw(e);
      if (raw.size() != expected_bytes) {
        throw std::invalid_argument("Qwen4ExpGpuBackend: GGUF raw payload size mismatch on " + name);
      }
      DevQuantWeight w;
      w.kind = DevQuantWeight::Kind::Ggml;
      w.type_id = e.type_id;
      w.in_features = expected_in;
      w.out_features = expected_out;
      w.buf = DevBuf(raw.size());
      CUDA_CHECK(cudaMemcpy(w.buf.p, raw.data(), raw.size(), cudaMemcpyHostToDevice));
      load_stats_.dense_device_bytes += w.buf.bytes;
      return w;
    }
    const auto& es = *impl_->elastic;
    const ElasticLinearLayout lay = es.validate_dense_linear(name, expected_out, expected_in);
    DevQuantWeight w;
    w.out_features = lay.out_features;
    w.in_features = lay.in_features;
    if (lay.kind == ElasticTensorKind::Fp8) {
      w.kind = DevQuantWeight::Kind::Fp8;
      w.scale = es.read_scalar_f32(*lay.scale);
      const auto raw = es.read_tensor_raw(*lay.weight);
      if (raw.size() != lay.expected_weight_bytes) {
        throw std::invalid_argument("Qwen4ExpGpuBackend: FP8 weight byte size mismatch on " + name);
      }
      w.buf = DevBuf(raw.size());
      CUDA_CHECK(cudaMemcpy(w.buf.p, raw.data(), raw.size(), cudaMemcpyHostToDevice));
      load_stats_.dense_device_bytes += w.buf.bytes;
      return w;
    }
    if (lay.kind == ElasticTensorKind::Nvfp4 || lay.kind == ElasticTensorKind::Atlas3 ||
        lay.kind == ElasticTensorKind::Atlas2) {
      w.kind = (lay.kind == ElasticTensorKind::Nvfp4)
                   ? DevQuantWeight::Kind::Nvfp4
                   : (lay.kind == ElasticTensorKind::Atlas3 ? DevQuantWeight::Kind::Atlas3
                                                            : DevQuantWeight::Kind::Atlas2);
      w.scale = es.read_scalar_f32(*lay.scale_2);
      const auto raw_c = es.read_tensor_raw(*lay.weight);
      const auto raw_s = es.read_tensor_raw(*lay.scale);
      if (raw_c.size() != lay.expected_weight_bytes || raw_s.size() != lay.expected_scale_bytes) {
        throw std::invalid_argument("Qwen4ExpGpuBackend: packed dense byte size mismatch on " + name);
      }
      w.buf = DevBuf(raw_c.size());
      w.scales = DevBuf(raw_s.size());
      CUDA_CHECK(cudaMemcpy(w.buf.p, raw_c.data(), raw_c.size(), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(w.scales.p, raw_s.data(), raw_s.size(), cudaMemcpyHostToDevice));
      load_stats_.dense_device_bytes += w.buf.bytes + w.scales.bytes;
      return w;
    }
    if (lay.kind == ElasticTensorKind::F32 || lay.kind == ElasticTensorKind::Bf16) {
      w.kind = DevQuantWeight::Kind::Ggml;
      w.type_id = (lay.kind == ElasticTensorKind::Bf16) ? 30u : 0u;
      const auto raw = es.read_tensor_raw(*lay.weight);
      if (raw.size() != lay.expected_weight_bytes) {
        throw std::invalid_argument("Qwen4ExpGpuBackend: dense F32/BF16 byte size mismatch on " + name);
      }
      w.buf = DevBuf(raw.size());
      CUDA_CHECK(cudaMemcpy(w.buf.p, raw.data(), raw.size(), cudaMemcpyHostToDevice));
      load_stats_.dense_device_bytes += w.buf.bytes;
      return w;
    }
    throw std::invalid_argument("Qwen4ExpGpuBackend: unsupported Elastic linear format on " + name);
  };

  const int H = c.hidden_size, hc = c.hc_mult, hc_dim = c.hc_dim(), hc_lr = c.hc_low_rank;
  const int conv_dim = c.gdn_conv_dim(), val_dim = c.gdn_value_dim(), v_heads = c.ssm_time_step_rank;
  if (!is_elastic) {
    impl_->tok_embd_entry = &impl_->gguf->require_tensor("token_embd.weight");
    impl_->ple_table_entry = impl_->gguf->find_tensor("per_layer_token_embd.weight");
  }
  impl_->output_hc_norm = upload_f32("output_hc_norm.weight", {hc_dim});
  impl_->output_hc_down = upload_quant("output_hc_down.weight", hc_lr, hc_dim);
  impl_->output_hc_up = upload_quant("output_hc_up.weight", hc_dim, hc_lr);
  const bool has_output_w =
      is_elastic ? (impl_->elastic->find_tensor("output.weight") != nullptr)
                 : (impl_->gguf->find_tensor("output.weight") != nullptr);
  impl_->output_weight = upload_quant(has_output_w ? "output.weight" : "token_embd.weight", c.vocab_size, H);

  impl_->layers.resize(static_cast<std::size_t>(c.num_hidden_layers));
  std::uint64_t total_expert_bytes = 0;
  std::size_t max_gate_exp_b = 0, max_up_exp_b = 0, max_down_exp_b = 0;

  for (int il = 0; il < c.num_hidden_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    auto& L = impl_->layers[static_cast<std::size_t>(il)];
    L.is_linear = c.is_linear(il);
    L.has_ple = c.is_ple(il);
    L.has_qsa = c.has_qsa(il);

    L.hc_attn_norm = upload_f32(p + "hc_attn_norm.weight", {hc_dim});
    L.hc_attn_down = upload_quant(p + "hc_attn_down.weight", hc_lr, hc_dim);
    L.hc_attn_up = upload_quant(p + "hc_attn_up.weight", hc_dim, hc_lr);
    L.hc_attn_inject = upload_quant(p + "hc_attn_inject.weight", hc, hc_dim);
    L.hc_ffn_norm = upload_f32(p + "hc_ffn_norm.weight", {hc_dim});
    L.hc_ffn_down = upload_quant(p + "hc_ffn_down.weight", hc_lr, hc_dim);
    L.hc_ffn_up = upload_quant(p + "hc_ffn_up.weight", hc_dim, hc_lr);
    L.hc_ffn_inject = upload_quant(p + "hc_ffn_inject.weight", hc, hc_dim);

    if (L.has_ple) {
      L.ple_key = upload_quant(p + "ple_key.weight", hc_dim, H);
      L.ple_value = upload_quant(p + "ple_value.weight", H, H);
      L.ple_norm_key = upload_f32(p + "ple_norm_key.weight", {hc_dim});
      L.ple_norm_query = upload_f32(p + "ple_norm_query.weight", {hc_dim});
      L.ple_norm_conv = upload_f32(p + "ple_norm_conv.weight", {hc_dim});
      L.ple_conv1d = upload_f32(p + "ple_conv1d.weight", {hc_dim, c.ple_conv_kernel});
    }

    if (L.is_linear) {
      L.attn_qkv = upload_quant(p + "attn_qkv.weight", conv_dim, H);
      L.attn_gate = upload_quant(p + "attn_gate.weight", val_dim, H);
      L.ssm_alpha = upload_quant(p + "ssm_alpha.weight", v_heads, H);
      L.ssm_beta = upload_quant(p + "ssm_beta.weight", v_heads, H);
      L.ssm_out = upload_quant(p + "ssm_out.weight", H, val_dim);
      L.ssm_conv1d = upload_f32(p + "ssm_conv1d.weight", {conv_dim, c.ssm_conv_kernel});
      L.ssm_dt_bias = upload_f32(p + "ssm_dt.bias", {v_heads});
      L.ssm_a = upload_f32(p + "ssm_a", {v_heads});
      L.ssm_norm = upload_f32(p + "ssm_norm.weight", {c.ssm_state_size});
    } else {
      L.attn_q = upload_quant(p + "attn_q.weight", c.full_q_out_dim(), H);
      L.attn_k = upload_quant(p + "attn_k.weight", c.full_kv_out_dim(), H);
      L.attn_v = upload_quant(p + "attn_v.weight", c.full_kv_out_dim(), H);
      L.attn_output = upload_quant(p + "attn_output.weight", H, c.num_attention_heads * c.head_dim);
      L.attn_q_norm = upload_f32(p + "attn_q_norm.weight", {c.head_dim});
      L.attn_k_norm = upload_f32(p + "attn_k_norm.weight", {c.head_dim});
      if (L.has_qsa) {
        L.idx_q_proj = upload_quant(p + "indexer.q_proj.weight", c.indexer_head_count * c.indexer_key_length, H);
        L.idx_k_proj = upload_quant(p + "indexer.k_proj.weight", c.indexer_key_length, H);
        L.idx_q_norm = upload_f32(p + "indexer.q_norm.weight", {c.indexer_key_length});
        L.idx_k_norm = upload_f32(p + "indexer.k_norm.weight", {c.indexer_key_length});
      }
    }

    L.ffn_gate_inp = upload_quant(p + "ffn_gate_inp.weight", c.expert_count, H);
    if (c.shared_expert_ffn_length > 0) {
      L.ffn_gate_inp_shexp = upload_f32(p + "ffn_gate_inp_shexp.weight", {H});
      L.ffn_gate_shexp = upload_quant(p + "ffn_gate_shexp.weight", c.shared_expert_ffn_length, H);
      L.ffn_up_shexp = upload_quant(p + "ffn_up_shexp.weight", c.shared_expert_ffn_length, H);
      L.ffn_down_shexp = upload_quant(p + "ffn_down_shexp.weight", H, c.shared_expert_ffn_length);
    }

    if (!is_elastic) {
      L.elastic_moe = false;
      auto bind_gguf_exp = [&](const std::string& name, int exp_in, int exp_out) -> std::pair<const GgufTensorEntry*, std::size_t> {
        const auto& e = impl_->gguf->require_tensor(name);
        if (!is_supported_gpu_ggml_type(e.type_id)) {
          throw std::invalid_argument("Qwen4ExpGpuBackend: unsupported GPU GGML expert type_id " +
                                      std::to_string(e.type_id) + " on " + name);
        }
        if (e.dims.size() != 3 || e.dims[0] != static_cast<std::uint64_t>(exp_in) ||
            e.dims[1] != static_cast<std::uint64_t>(exp_out) ||
            e.dims[2] != static_cast<std::uint64_t>(c.expert_count)) {
          throw std::invalid_argument("Qwen4ExpGpuBackend: GGUF expert shape mismatch on " + name);
        }
        const std::uint64_t per_exp = static_cast<std::uint64_t>(exp_out) * e.row_bytes();
        if (e.byte_size != static_cast<std::uint64_t>(c.expert_count) * per_exp) {
          throw std::invalid_argument("Qwen4ExpGpuBackend: GGUF expert byte size mismatch on " + name);
        }
        return {&e, static_cast<std::size_t>(per_exp)};
      };
      std::tie(L.gate_exps_entry, L.gate_exp_bytes) = bind_gguf_exp(p + "ffn_gate_exps.weight", H, c.expert_ffn_length);
      std::tie(L.up_exps_entry, L.up_exp_bytes) = bind_gguf_exp(p + "ffn_up_exps.weight", H, c.expert_ffn_length);
      std::tie(L.down_exps_entry, L.down_exp_bytes) = bind_gguf_exp(p + "ffn_down_exps.weight", c.expert_ffn_length, H);
      total_expert_bytes += L.gate_exps_entry->byte_size + L.up_exps_entry->byte_size + L.down_exps_entry->byte_size;
    } else {
      const auto& es = *impl_->elastic;
      L.elastic_moe = true;
      const ElasticExpertLayout gate_lay = es.validate_moe_experts(p + "ffn_gate_exps", c.expert_count, c.expert_ffn_length, H);
      const ElasticExpertLayout up_lay = es.validate_moe_experts(p + "ffn_up_exps", c.expert_count, c.expert_ffn_length, H);
      const ElasticExpertLayout down_lay = es.validate_moe_experts(p + "ffn_down_exps", c.expert_count, H, c.expert_ffn_length);

      L.gate_moe_fmt = to_elastic_weight_format(gate_lay.kind, p + "ffn_gate_exps");
      L.up_moe_fmt = to_elastic_weight_format(up_lay.kind, p + "ffn_up_exps");
      L.down_moe_fmt = to_elastic_weight_format(down_lay.kind, p + "ffn_down_exps");
      L.e_gate_w = gate_lay.weight;
      L.e_gate_s = gate_lay.scale;
      L.e_up_w = up_lay.weight;
      L.e_up_s = up_lay.scale;
      L.e_down_w = down_lay.weight;
      L.e_down_s = down_lay.scale;

      L.gate_scale2 = es.read_tensor_f32(*gate_lay.scale_2);
      L.up_scale2 = es.read_tensor_f32(*up_lay.scale_2);
      L.down_scale2 = es.read_tensor_f32(*down_lay.scale_2);
      const std::size_t exp_n = static_cast<std::size_t>(c.expert_count);
      if (L.gate_scale2.size() != exp_n || L.up_scale2.size() != exp_n || L.down_scale2.size() != exp_n) {
        throw std::invalid_argument("Qwen4ExpGpuBackend: expert weight_scale_2 count mismatch at layer " +
                                    std::to_string(il));
      }

      L.gate_code_bytes = static_cast<std::size_t>(gate_lay.per_expert_code_bytes);
      L.gate_scale_bytes = static_cast<std::size_t>(gate_lay.per_expert_scale_bytes);
      L.up_code_bytes = static_cast<std::size_t>(up_lay.per_expert_code_bytes);
      L.up_scale_bytes = static_cast<std::size_t>(up_lay.per_expert_scale_bytes);
      L.down_code_bytes = static_cast<std::size_t>(down_lay.per_expert_code_bytes);
      L.down_scale_bytes = static_cast<std::size_t>(down_lay.per_expert_scale_bytes);
      L.gate_exp_bytes = L.gate_code_bytes + L.gate_scale_bytes;
      L.up_exp_bytes = L.up_code_bytes + L.up_scale_bytes;
      L.down_exp_bytes = L.down_code_bytes + L.down_scale_bytes;
      total_expert_bytes += L.e_gate_w->byte_size + L.e_gate_s->byte_size + L.e_up_w->byte_size + L.e_up_s->byte_size +
                            L.e_down_w->byte_size + L.e_down_s->byte_size;
    }
    max_gate_exp_b = std::max(max_gate_exp_b, L.gate_exp_bytes);
    max_up_exp_b = std::max(max_up_exp_b, L.up_exp_bytes);
    max_down_exp_b = std::max(max_down_exp_b, L.down_exp_bytes);
  }

  // Allocate reusable GPU scratch buffers
  auto alloc_f32 = [&](std::size_t n) {
    DevBuf b(n * sizeof(float));
    load_stats_.scratch_device_bytes += b.bytes;
    return b;
  };
  impl_->d_emb = alloc_f32(H);
  impl_->d_res_a = alloc_f32(hc_dim);
  impl_->d_res_b = alloc_f32(hc_dim);
  impl_->d_xn = alloc_f32(hc_dim);
  impl_->d_hc_lo = alloc_f32(hc_lr);
  impl_->d_hc_gate = alloc_f32(hc_dim);
  impl_->d_cur = alloc_f32(H);
  impl_->d_inject = alloc_f32(hc);
  impl_->d_block_out = alloc_f32(H);
  impl_->d_ple_emb = alloc_f32(H);
  impl_->d_ple_key_raw = alloc_f32(hc_dim);
  impl_->d_ple_val = alloc_f32(H);
  impl_->d_ple_key = alloc_f32(hc_dim);
  impl_->d_ple_q = alloc_f32(hc_dim);
  impl_->d_ple_gated = alloc_f32(hc_dim);
  impl_->d_ple_norm = alloc_f32(hc_dim);
  impl_->d_qkv = alloc_f32(c.gdn_conv_dim());
  impl_->d_z = alloc_f32(c.gdn_value_dim());
  impl_->d_alpha = alloc_f32(c.ssm_time_step_rank);
  impl_->d_beta = alloc_f32(c.ssm_time_step_rank);
  impl_->d_conv_tmp = alloc_f32(c.gdn_conv_dim());
  impl_->d_gdn_out = alloc_f32(c.gdn_value_dim());
  impl_->d_iq = alloc_f32(std::max(1, c.indexer_head_count * c.indexer_key_length));
  impl_->d_ik = alloc_f32(std::max(1, c.indexer_key_length));
  impl_->d_block_scores = alloc_f32(static_cast<std::size_t>(options.max_ctx));
  impl_->d_sel_pos = DevBuf(static_cast<std::size_t>(options.max_ctx) * sizeof(std::int32_t));
  load_stats_.scratch_device_bytes += impl_->d_sel_pos.bytes;
  impl_->d_qg = alloc_f32(c.full_q_out_dim());
  impl_->d_kr = alloc_f32(c.full_kv_out_dim());
  impl_->d_vr = alloc_f32(c.full_kv_out_dim());
  impl_->d_q_sc = alloc_f32(c.num_attention_heads * c.head_dim);
  impl_->d_g_sc = alloc_f32(c.num_attention_heads * c.head_dim);
  impl_->d_attn_out = alloc_f32(c.num_attention_heads * c.head_dim);
  impl_->d_r_logits = alloc_f32(c.expert_count);
  impl_->d_r_exp = DevBuf(static_cast<std::size_t>(c.expert_used_count) * sizeof(std::int32_t));
  impl_->d_r_w = alloc_f32(c.expert_used_count);
  const int max_ff = std::max(c.expert_ffn_length, std::max(1, c.shared_expert_ffn_length));
  impl_->d_ff_g = alloc_f32(max_ff);
  impl_->d_ff_u = alloc_f32(max_ff);
  impl_->d_ff_act = alloc_f32(max_ff);
  impl_->d_e_out = alloc_f32(H);
  impl_->d_sh_out = alloc_f32(H);
  impl_->d_logits = alloc_f32(c.vocab_size);

  // Build capabilities & WorkerProfile BEFORE constructing default session
  profile_.family = ModelFamily::Qwen4Exp;
  profile_.model_id = c.model_name.empty() ? "Qwen3.8-Flash-Next" : c.model_name;
  profile_.gear = "gguf_tiered_moe";
  profile_.kv_cache_mode = "per_pass";
  profile_.tokenizer_format = "qwen3_bpe";
  profile_.chat_template_family = "chatml";
  profile_.max_ctx = options.max_ctx;
  profile_.max_batch = 1;
  profile_.capabilities.family = ModelFamily::Qwen4Exp;
  profile_.capabilities.backend_name = "qwen4exp-cuda";
  profile_.capabilities.architecture = c.architecture;
  profile_.capabilities.supports_text = true;
  profile_.capabilities.supports_prefix_cache = options.prefix_cache;
  profile_.capabilities.supports_images = false;
  profile_.capabilities.supports_continuous_batching = false;
  profile_.capabilities.supports_speculation = false;
  profile_.capabilities.supports_persona_hooks = false;
  profile_.capabilities.supports_tiered_residency = true;
  profile_.capabilities.supports_logit_capture = true;
  profile_.capabilities.max_supported_batch = 1;
  profile_.capabilities.vocab_size = c.vocab_size;
  profile_.capabilities.max_position_embeddings = c.max_position_embeddings;

  auto& sid = profile_.state_identity;
  sid.family = ModelFamily::Qwen4Exp;
  sid.backend_name = "qwen4exp-cuda";
  sid.checkpoint = impl_->ckpt_id();
  sid.state_layout_id = "qwen4exp_ple_gdn_qsa_f32_v1";
  sid.act_dtype = ActivationDtype::Fp32;
  sid.numeric_abi = is_elastic ? ("act=f32;weights=atlas_elastic_" + impl_->elastic->profile() + ";gdn=sigmoid_f32;kv=f32")
                               : "act=f32;weights=ggml_direct;gdn=sigmoid_f32;kv=f32";
  sid.weight_precision = is_elastic ? ("atlas_elastic_" + impl_->elastic->profile()) : "gguf_mixed";
  sid.kv_format = "qwen4exp_gdn_qsa_f32";
  sid.kv_cache_mode = "per_pass";
  sid.gear = is_elastic ? "elastic_tiered_moe" : "gguf_tiered_moe";
  profile_.gear = sid.gear;
  sid.max_ctx = options.max_ctx;
  sid.prefill_chunk = 1;
  sid.family_state_fingerprint = c.canonical_fingerprint();

  // Construct default session so KV VRAM is accounted before deciding expert residency!
  impl_->default_session = std::unique_ptr<Qwen4ExpGpuSession>(new Qwen4ExpGpuSession(*this));
  load_stats_.kv_state_device_bytes = impl_->default_session->state_bytes();

  std::size_t free_vram = 0, total_vram = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_vram, &total_vram));
  const std::uint64_t safety_margin = 1024ull * 1024ull * 1024ull;  // leave 1 GiB headroom
  const std::uint64_t avail_for_experts = (free_vram > safety_margin) ? (free_vram - safety_margin) : 0ull;

  if (options.expert_cache_bytes == 0 && total_expert_bytes <= avail_for_experts) {
    // Mode A: All-Resident MoE in GPU VRAM!
    for (int il = 0; il < c.num_hidden_layers; ++il) {
      auto& L = impl_->layers[static_cast<std::size_t>(il)];
      if (!is_elastic) {
        const auto g_raw = impl_->gguf->read_tensor_raw(*L.gate_exps_entry);
        const auto u_raw = impl_->gguf->read_tensor_raw(*L.up_exps_entry);
        const auto d_raw = impl_->gguf->read_tensor_raw(*L.down_exps_entry);
        L.gate_exps_dev = DevBuf(g_raw.size());
        L.up_exps_dev = DevBuf(u_raw.size());
        L.down_exps_dev = DevBuf(d_raw.size());
        CUDA_CHECK(cudaMemcpy(L.gate_exps_dev.p, g_raw.data(), g_raw.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(L.up_exps_dev.p, u_raw.data(), u_raw.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(L.down_exps_dev.p, d_raw.data(), d_raw.size(), cudaMemcpyHostToDevice));
        load_stats_.expert_device_bytes += g_raw.size() + u_raw.size() + d_raw.size();
      } else {
        auto up_dev = [&](const SafetensorsEntry& e, std::size_t expected_total, DevBuf& dst) {
          const auto raw = impl_->elastic->read_tensor_raw(e);
          if (raw.size() != expected_total) {
            throw std::invalid_argument("Qwen4ExpGpuBackend: resident expert payload size mismatch on " + e.name);
          }
          dst = DevBuf(raw.size());
          CUDA_CHECK(cudaMemcpy(dst.p, raw.data(), raw.size(), cudaMemcpyHostToDevice));
          load_stats_.expert_device_bytes += raw.size();
        };
        const std::size_t exp_n = static_cast<std::size_t>(c.expert_count);
        up_dev(*L.e_gate_w, exp_n * L.gate_code_bytes, L.gate_exps_dev);
        up_dev(*L.e_gate_s, exp_n * L.gate_scale_bytes, L.gate_exps_scales_dev);
        up_dev(*L.e_up_w, exp_n * L.up_code_bytes, L.up_exps_dev);
        up_dev(*L.e_up_s, exp_n * L.up_scale_bytes, L.up_exps_scales_dev);
        up_dev(*L.e_down_w, exp_n * L.down_code_bytes, L.down_exps_dev);
        up_dev(*L.e_down_s, exp_n * L.down_scale_bytes, L.down_exps_scales_dev);
      }
    }
    load_stats_.all_experts_resident = true;
    profile_.gear = is_elastic ? "elastic_all_resident" : "gguf_all_resident";
    sid.gear = profile_.gear;
  } else {
    // Mode B: Tiered LRU GPU Expert Cache (works on 16/24/32 GB cards!)
    const std::size_t one_slot = max_gate_exp_b + max_up_exp_b + max_down_exp_b;
    const std::uint64_t budget =
        options.expert_cache_bytes > 0 ? std::min<std::uint64_t>(options.expert_cache_bytes, avail_for_experts)
                                       : avail_for_experts;
    const std::size_t slots = std::max<std::size_t>(static_cast<std::size_t>(c.expert_used_count),
                                                    one_slot > 0 ? static_cast<std::size_t>(budget / one_slot) : 0);
    impl_->expert_cache.init(slots, max_gate_exp_b, max_up_exp_b, max_down_exp_b);
    load_stats_.expert_device_bytes = impl_->expert_cache.bytes();
    load_stats_.expert_cache_slots = slots;
    load_stats_.all_experts_resident = false;
  }

  load_stats_.load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

Qwen4ExpGpuBackend::~Qwen4ExpGpuBackend() = default;

const Qwen4ExpConfig& Qwen4ExpGpuBackend::config() const noexcept { return impl_->cfg(); }
Qwen4ExpExpertCacheStats Qwen4ExpGpuBackend::expert_cache_stats() const noexcept {
  return impl_->expert_cache.stats();
}
std::string Qwen4ExpGpuBackend::model_id() const { return profile_.model_id; }
int Qwen4ExpGpuBackend::max_model_len() const { return impl_->opts.max_ctx; }
PrefixCacheStats Qwen4ExpGpuBackend::cache_stats() const {
  std::lock_guard lock(impl_->mu);
  return impl_->cache_stats;
}
void Qwen4ExpGpuBackend::reset_session() {
  std::lock_guard lock(impl_->mu);
  impl_->default_session->reset();
}
std::unique_ptr<BackendSession> Qwen4ExpGpuBackend::create_session() {
  return std::unique_ptr<BackendSession>(new Qwen4ExpGpuSession(*this));
}
BackendSession* Qwen4ExpGpuBackend::acquire_session(int slot) {
  if (slot != 0) return nullptr;
  return impl_->default_session.get();
}
void Qwen4ExpGpuBackend::release_session(int /*slot*/) {}

ResidencyReport Qwen4ExpGpuBackend::residency_report() const {
  ResidencyReport r;
  r.weights_device_bytes = load_stats_.dense_device_bytes +
                           (load_stats_.all_experts_resident ? load_stats_.expert_device_bytes : 0ull);
  r.cache_device_bytes = load_stats_.all_experts_resident ? 0ull : load_stats_.expert_device_bytes;
  r.state_device_bytes = load_stats_.kv_state_device_bytes;
  r.device_bytes = r.weights_device_bytes + r.cache_device_bytes + r.state_device_bytes +
                   load_stats_.scratch_device_bytes;
  r.mmap_bytes = impl_->total_tensor_bytes();
  return r;
}

AdmissionEstimate Qwen4ExpGpuBackend::admission_estimate(int max_ctx, int max_batch) const {
  AdmissionEstimate est;
  const auto& c = config();
  const std::uint64_t kv_bytes =
      static_cast<std::uint64_t>(c.num_full_layers) *
      (2ull * static_cast<std::uint64_t>(max_ctx) * c.full_kv_out_dim() +
       static_cast<std::uint64_t>(max_ctx) * c.indexer_key_length) *
      sizeof(float);
  est.required_device_bytes = load_stats_.dense_device_bytes + kv_bytes * std::max(1, max_batch) +
                              256ull * 1024ull * 1024ull;
  est.required_mmap_bytes = impl_->total_tensor_bytes();
  std::size_t free_v = 0, total_v = 0;
  if (cudaMemGetInfo(&free_v, &total_v) == cudaSuccess) {
    est.fits_current_budget = (max_batch <= 1) && (est.required_device_bytes <= total_v);
    est.reason = est.fits_current_budget ? "ok" : "insufficient_vram_or_batch_gt_1";
  }
  return est;
}

std::uint64_t Qwen4ExpGpuBackend::evict_caches(std::uint64_t /*target_free_bytes*/) {
  std::lock_guard lock(impl_->mu);
  return impl_->expert_cache.clear();
}

std::vector<float> Qwen4ExpGpuBackend::forward_token_logits_gpu(Qwen4ExpGpuSession& session, std::int32_t token_id,
                                                                bool compute_logits) {
  const auto& c = config();
  const bool is_elastic = impl_->is_elastic;
  auto& st = *session.state_;
  const int pos = static_cast<int>(st.committed_tokens.size());
  if (pos >= impl_->opts.max_ctx) {
    throw std::runtime_error("Qwen4ExpGpuBackend: context length exceeded max_ctx");
  }
  if (token_id < 0 || token_id >= c.vocab_size) {
    throw std::out_of_range("Qwen4ExpGpuBackend: token_id out of range");
  }

  cudaStream_t s = impl_->stream;
  const int H = c.hidden_size, hc = c.hc_mult, hc_dim = c.hc_dim(), hc_lr = c.hc_low_rank;

  // 1. Token embedding row gather on host (O(1) pread) + 10 KB H2D -> repeat across `hc` residual streams
  std::vector<float> h_emb(static_cast<std::size_t>(H));
  if (is_elastic) {
    impl_->elastic->read_embed_row_f32(static_cast<std::uint64_t>(token_id), h_emb);
  } else {
    impl_->gguf->read_tensor_row_f32(*impl_->tok_embd_entry, static_cast<std::uint64_t>(token_id), h_emb);
  }
  CUDA_CHECK(cudaMemcpyAsync(impl_->d_emb.p, h_emb.data(), h_emb.size() * sizeof(float), cudaMemcpyHostToDevice, s));
  hc_init_repeat_f32_cuda(impl_->d_emb.f32(), impl_->d_res_a.f32(), hc, H, s);

  float* res_in = impl_->d_res_a.f32();
  float* res_out = impl_->d_res_b.f32();

  auto run_linear = [&](const DevQuantWeight& w, const float* x, float* y) {
    if (w.kind == DevQuantWeight::Kind::Ggml) {
      ggml_linear_f32_cuda(w.type_id, w.buf.p, x, y, 1, w.out_features, w.in_features, s);
    } else if (w.kind == DevQuantWeight::Kind::Fp8) {
      fp8_linear_f32_cuda(w.buf.u8(), w.scale, x, y, 1, w.out_features, w.in_features, s);
    } else {
      const auto ef = (w.kind == DevQuantWeight::Kind::Nvfp4)
                          ? ElasticWeightFormat::Nvfp4
                          : (w.kind == DevQuantWeight::Kind::Atlas3 ? ElasticWeightFormat::Atlas3
                                                                    : ElasticWeightFormat::Atlas2);
      elastic_linear_f32_cuda(ef, w.buf.u8(), w.scales.u8(), nullptr, w.scale, x, y, 1, w.out_features, w.in_features,
                              s);
    }
  };
  auto run_hc_mix = [&](const float* x, const DevBuf& wn, const DevQuantWeight& wd, const DevQuantWeight& wu,
                        const DevQuantWeight* wi) {
    grouped_rmsnorm_f32_cuda(x, wn.f32(), impl_->d_xn.f32(), hc, H, c.rms_norm_eps, s);
    run_linear(wd, impl_->d_xn.f32(), impl_->d_hc_lo.f32());
    hc_silu_scale_inplace_f32_cuda(impl_->d_hc_lo.f32(), hc_lr, hc, s);
    run_linear(wu, impl_->d_hc_lo.f32(), impl_->d_hc_gate.f32());
    hc_gate_collapse_f32_cuda(impl_->d_xn.f32(), impl_->d_hc_gate.f32(), impl_->d_cur.f32(), hc, H, s);
    if (wi) run_linear(*wi, impl_->d_xn.f32(), impl_->d_inject.f32());
  };

  std::vector<std::int32_t> h_exps(static_cast<std::size_t>(c.expert_used_count));
  std::vector<float> h_exp_w(static_cast<std::size_t>(c.expert_used_count));

  for (int il = 0; il < c.num_hidden_layers; ++il) {
    const auto& L = impl_->layers[static_cast<std::size_t>(il)];

    // Optional PLE module (host O(1) 16-row gather from host-only table -> 10 KB H2D -> GPU step)
    const bool has_ple_table = is_elastic ? impl_->elastic->has_embedded_ple() : (impl_->ple_table_entry != nullptr);
    if (L.has_ple && has_ple_table) {
      std::vector<std::int32_t> ple_rows(static_cast<std::size_t>(c.ple_n_heads()));
      ple_hash_step(c, token_id, st.ple_token_history, ple_rows);
      std::vector<float> h_ple(static_cast<std::size_t>(H));
      for (int h = 0; h < c.ple_n_heads(); ++h) {
        auto dst_span = std::span<float>(h_ple.data() + static_cast<std::size_t>(h) * c.ple_head_dim,
                                         static_cast<std::size_t>(c.ple_head_dim));
        if (is_elastic) {
          impl_->elastic->read_ple_row_f32(static_cast<std::uint64_t>(ple_rows[h]), dst_span);
        } else {
          impl_->gguf->read_tensor_row_f32(*impl_->ple_table_entry, static_cast<std::uint64_t>(ple_rows[h]), dst_span);
        }
      }
      CUDA_CHECK(cudaMemcpyAsync(impl_->d_ple_emb.p, h_ple.data(), h_ple.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, s));
      run_linear(L.ple_key, impl_->d_ple_emb.f32(), impl_->d_ple_key_raw.f32());
      run_linear(L.ple_value, impl_->d_ple_emb.f32(), impl_->d_ple_val.f32());
      grouped_rmsnorm_f32_cuda(impl_->d_ple_key_raw.f32(), L.ple_norm_key.f32(), impl_->d_ple_key.f32(), hc, H,
                               c.rms_norm_eps, s);
      grouped_rmsnorm_f32_cuda(res_in, L.ple_norm_query.f32(), impl_->d_ple_q.f32(), hc, H, c.rms_norm_eps, s);
      ple_gate_value_f32_cuda(impl_->d_ple_key.f32(), impl_->d_ple_q.f32(), impl_->d_ple_val.f32(),
                              impl_->d_ple_gated.f32(), hc, H, s);
      grouped_rmsnorm_f32_cuda(impl_->d_ple_gated.f32(), L.ple_norm_conv.f32(), impl_->d_ple_norm.f32(), hc, H,
                               c.rms_norm_eps, s);
      ple_conv_residual_step_f32_cuda(res_in, impl_->d_ple_gated.f32(), impl_->d_ple_norm.f32(), L.ple_conv1d.f32(),
                                      st.ple_conv_states[il].f32(), res_out, hc_dim, c.ple_conv_kernel,
                                      c.ple_ngram_size, s);
      std::swap(res_in, res_out);
    }

    // Attention / GDN block
    run_hc_mix(res_in, L.hc_attn_norm, L.hc_attn_down, L.hc_attn_up, &L.hc_attn_inject);
    if (L.is_linear) {
      run_linear(L.attn_qkv, impl_->d_cur.f32(), impl_->d_qkv.f32());
      run_linear(L.attn_gate, impl_->d_cur.f32(), impl_->d_z.f32());
      run_linear(L.ssm_alpha, impl_->d_cur.f32(), impl_->d_alpha.f32());
      run_linear(L.ssm_beta, impl_->d_cur.f32(), impl_->d_beta.f32());
      gdn_step_sigmoid_f32_cuda(impl_->d_qkv.f32(), impl_->d_z.f32(), impl_->d_alpha.f32(), impl_->d_beta.f32(),
                                L.ssm_conv1d.f32(), L.ssm_dt_bias.f32(), L.ssm_a.f32(), L.ssm_norm.f32(),
                                c.rms_norm_eps, c.ssm_conv_kernel, c.ssm_group_count, c.ssm_time_step_rank,
                                c.ssm_state_size, st.gdn_conv_states[il].f32(), st.gdn_recurrent_states[il].f32(),
                                impl_->d_conv_tmp.f32(), impl_->d_gdn_out.f32(), s);
      run_linear(L.ssm_out, impl_->d_gdn_out.f32(), impl_->d_block_out.f32());
    } else {
      const std::int32_t* sel_dev = nullptr;
      int n_sel = pos + 1;
      if (L.has_qsa) {
        const int cr = c.compress_ratios[static_cast<std::size_t>(il)];
        const int width = std::min(pos + 1, c.indexer_top_k + cr - 1);
        run_linear(L.idx_q_proj, impl_->d_cur.f32(), impl_->d_iq.f32());
        run_linear(L.idx_k_proj, impl_->d_cur.f32(), impl_->d_ik.f32());
        const bool sparse = (pos + 1 > width);
        qsa_indexer_append_and_score_f32_cuda(impl_->d_iq.f32(), impl_->d_ik.f32(), L.idx_q_norm.f32(),
                                              L.idx_k_norm.f32(), c.rms_norm_eps, c.rope_theta, c.rotary_dim,
                                              c.indexer_head_count, c.indexer_key_length, cr, pos,
                                              st.idx_k_caches[il].f32(), sparse ? impl_->d_block_scores.f32() : nullptr,
                                              s);
        if (sparse) {
          const int n_blocks = (pos + 1) / cr;
          const int tail_start = n_blocks * cr;
          std::vector<float> h_bscores(static_cast<std::size_t>(n_blocks));
          CUDA_CHECK(cudaMemcpyAsync(h_bscores.data(), impl_->d_block_scores.p,
                                     static_cast<std::size_t>(n_blocks) * sizeof(float), cudaMemcpyDeviceToHost, s));
          CUDA_CHECK(cudaStreamSynchronize(s));
          std::vector<std::pair<float, int>> scored;
          scored.reserve(static_cast<std::size_t>(pos + 1));
          for (int p = 0; p <= pos; ++p) {
            scored.emplace_back(p >= tail_start ? 1e9f : h_bscores[static_cast<std::size_t>(p / cr)], p);
          }
          std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
          });
          std::vector<std::int32_t> sel(static_cast<std::size_t>(width));
          for (int i = 0; i < width; ++i) sel[static_cast<std::size_t>(i)] = scored[static_cast<std::size_t>(i)].second;
          std::sort(sel.begin(), sel.end());
          CUDA_CHECK(cudaMemcpyAsync(impl_->d_sel_pos.p, sel.data(), sel.size() * sizeof(std::int32_t),
                                     cudaMemcpyHostToDevice, s));
          sel_dev = impl_->d_sel_pos.i32();
          n_sel = width;
        }
      }
      run_linear(L.attn_q, impl_->d_cur.f32(), impl_->d_qg.f32());
      run_linear(L.attn_k, impl_->d_cur.f32(), impl_->d_kr.f32());
      run_linear(L.attn_v, impl_->d_cur.f32(), impl_->d_vr.f32());
      full_attn_qsa_step_f32_cuda(impl_->d_qg.f32(), impl_->d_kr.f32(), impl_->d_vr.f32(), L.attn_q_norm.f32(),
                                  L.attn_k_norm.f32(), c.rms_norm_eps, c.rope_theta, c.num_attention_heads,
                                  c.num_key_value_heads, c.head_dim, c.rotary_dim, pos, sel_dev, n_sel,
                                  st.k_caches[il].f32(), st.v_caches[il].f32(), impl_->d_q_sc.f32(), impl_->d_g_sc.f32(),
                                  impl_->d_attn_out.f32(), s);
      run_linear(L.attn_output, impl_->d_attn_out.f32(), impl_->d_block_out.f32());
    }
    hc_combine_f32_cuda(res_in, impl_->d_block_out.f32(), impl_->d_inject.f32(), res_out, hc, H, s);
    std::swap(res_in, res_out);

    // MoE + Shared Expert block
    run_hc_mix(res_in, L.hc_ffn_norm, L.hc_ffn_down, L.hc_ffn_up, &L.hc_ffn_inject);
    run_linear(L.ffn_gate_inp, impl_->d_cur.f32(), impl_->d_r_logits.f32());
    route_top_k_softmax_f32_cuda(impl_->d_r_logits.f32(), c.expert_count, c.expert_used_count, c.expert_weights_scale,
                                 impl_->d_r_exp.i32(), impl_->d_r_w.f32(), s);
    CUDA_CHECK(cudaMemcpyAsync(h_exps.data(), impl_->d_r_exp.p, h_exps.size() * sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaMemcpyAsync(h_exp_w.data(), impl_->d_r_w.p, h_exp_w.size() * sizeof(float), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    impl_->d_block_out.zero(s);
    for (int k = 0; k < c.expert_used_count; ++k) {
      const int e = h_exps[static_cast<std::size_t>(k)];
      if (e < 0 || e >= c.expert_count) {
        throw std::out_of_range("Qwen4ExpGpuBackend: routed expert id out of range");
      }
      const float w = h_exp_w[static_cast<std::size_t>(k)];
      if (!L.elastic_moe) {
        const void* gate_ptr = nullptr;
        const void* up_ptr = nullptr;
        const void* down_ptr = nullptr;
        if (load_stats_.all_experts_resident) {
          gate_ptr = L.gate_exps_dev.u8() + static_cast<std::size_t>(e) * L.gate_exp_bytes;
          up_ptr = L.up_exps_dev.u8() + static_cast<std::size_t>(e) * L.up_exp_bytes;
          down_ptr = L.down_exps_dev.u8() + static_cast<std::size_t>(e) * L.down_exp_bytes;
        } else {
          const auto ptrs = impl_->expert_cache.fetch(*impl_->gguf, il, e, L, s);
          gate_ptr = ptrs.gate;
          up_ptr = ptrs.up;
          down_ptr = ptrs.down;
        }
        ggml_linear_f32_cuda(L.gate_exps_entry->type_id, gate_ptr, impl_->d_cur.f32(), impl_->d_ff_g.f32(), 1,
                             c.expert_ffn_length, H, s);
        ggml_linear_f32_cuda(L.up_exps_entry->type_id, up_ptr, impl_->d_cur.f32(), impl_->d_ff_u.f32(), 1,
                             c.expert_ffn_length, H, s);
        swiglu_mul_f32_cuda(impl_->d_ff_g.f32(), impl_->d_ff_u.f32(), impl_->d_ff_act.f32(), c.expert_ffn_length, s);
        ggml_linear_f32_cuda(L.down_exps_entry->type_id, down_ptr, impl_->d_ff_act.f32(), impl_->d_e_out.f32(), 1, H,
                             c.expert_ffn_length, s);
      } else {
        const std::uint8_t* g_c = nullptr;
        const std::uint8_t* g_s = nullptr;
        const std::uint8_t* u_c = nullptr;
        const std::uint8_t* u_s = nullptr;
        const std::uint8_t* d_c = nullptr;
        const std::uint8_t* d_s = nullptr;
        if (load_stats_.all_experts_resident) {
          g_c = L.gate_exps_dev.u8() + static_cast<std::size_t>(e) * L.gate_code_bytes;
          g_s = L.gate_exps_scales_dev.u8() + static_cast<std::size_t>(e) * L.gate_scale_bytes;
          u_c = L.up_exps_dev.u8() + static_cast<std::size_t>(e) * L.up_code_bytes;
          u_s = L.up_exps_scales_dev.u8() + static_cast<std::size_t>(e) * L.up_scale_bytes;
          d_c = L.down_exps_dev.u8() + static_cast<std::size_t>(e) * L.down_code_bytes;
          d_s = L.down_exps_scales_dev.u8() + static_cast<std::size_t>(e) * L.down_scale_bytes;
        } else {
          const auto ptrs = impl_->expert_cache.fetch_elastic(*impl_->elastic, il, e, L, s);
          g_c = static_cast<const std::uint8_t*>(ptrs.gate);
          g_s = g_c + L.gate_code_bytes;
          u_c = static_cast<const std::uint8_t*>(ptrs.up);
          u_s = u_c + L.up_code_bytes;
          d_c = static_cast<const std::uint8_t*>(ptrs.down);
          d_s = d_c + L.down_code_bytes;
        }
        elastic_linear_f32_cuda(L.gate_moe_fmt, g_c, g_s, nullptr, L.gate_scale2.at(static_cast<std::size_t>(e)),
                                impl_->d_cur.f32(), impl_->d_ff_g.f32(), 1, c.expert_ffn_length, H, s);
        elastic_linear_f32_cuda(L.up_moe_fmt, u_c, u_s, nullptr, L.up_scale2.at(static_cast<std::size_t>(e)),
                                impl_->d_cur.f32(), impl_->d_ff_u.f32(), 1, c.expert_ffn_length, H, s);
        swiglu_mul_f32_cuda(impl_->d_ff_g.f32(), impl_->d_ff_u.f32(), impl_->d_ff_act.f32(), c.expert_ffn_length, s);
        elastic_linear_f32_cuda(L.down_moe_fmt, d_c, d_s, nullptr, L.down_scale2.at(static_cast<std::size_t>(e)),
                                impl_->d_ff_act.f32(), impl_->d_e_out.f32(), 1, H, c.expert_ffn_length, s);
      }
      axpy_f32_cuda(impl_->d_block_out.f32(), impl_->d_e_out.f32(), w, H, s);
    }

    if (c.shared_expert_ffn_length > 0) {
      run_linear(L.ffn_gate_shexp, impl_->d_cur.f32(), impl_->d_ff_g.f32());
      run_linear(L.ffn_up_shexp, impl_->d_cur.f32(), impl_->d_ff_u.f32());
      swiglu_mul_f32_cuda(impl_->d_ff_g.f32(), impl_->d_ff_u.f32(), impl_->d_ff_act.f32(), c.shared_expert_ffn_length,
                          s);
      run_linear(L.ffn_down_shexp, impl_->d_ff_act.f32(), impl_->d_sh_out.f32());
      shared_expert_gate_add_f32_cuda(impl_->d_block_out.f32(), impl_->d_sh_out.f32(), L.ffn_gate_inp_shexp.f32(),
                                      impl_->d_cur.f32(), H, s);
    }

    hc_combine_f32_cuda(res_in, impl_->d_block_out.f32(), impl_->d_inject.f32(), res_out, hc, H, s);
    std::swap(res_in, res_out);
  }

  st.committed_tokens.push_back(token_id);

  std::vector<float> logits;
  if (compute_logits) {
    run_hc_mix(res_in, impl_->output_hc_norm, impl_->output_hc_down, impl_->output_hc_up, nullptr);
    run_linear(impl_->output_weight, impl_->d_cur.f32(), impl_->d_logits.f32());
    logits.resize(static_cast<std::size_t>(c.vocab_size));
    CUDA_CHECK(cudaMemcpyAsync(logits.data(), impl_->d_logits.p, logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                               s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    st.last_logits = logits;
    st.has_last_logits = true;
  } else {
    st.has_last_logits = false;
  }
  return logits;
}

GenerateStats Qwen4ExpGpuBackend::generate(std::span<const std::int32_t> prompt, int max_new_tokens,
                                           const Sampling& sampling, std::span<const std::int32_t> stop_ids,
                                           const std::function<bool(std::span<const std::int32_t>)>& on_tokens) {
  Prompt p;
  p.ids = prompt;
  return generate(p, max_new_tokens, sampling, stop_ids, on_tokens, {});
}

GenerateStats Qwen4ExpGpuBackend::generate(const Prompt& prompt, int max_new_tokens, const Sampling& sampling,
                                           std::span<const std::int32_t> stop_ids,
                                           const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                                           const PrefillCallback& on_prefill) {
  BackendRequest req;
  req.prompt = prompt;
  req.cache_namespace = prompt.cache_namespace;
  req.use_prefix_cache = impl_->opts.prefix_cache && !prompt.bypass_prefix_cache;
  return generate_request(req, max_new_tokens, sampling, stop_ids, on_tokens, on_prefill);
}

GenerateStats Qwen4ExpGpuBackend::generate_request(const BackendRequest& req, int max_new_tokens,
                                                   const Sampling& sampling, std::span<const std::int32_t> stop_ids,
                                                   const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                                                   const PrefillCallback& on_prefill) {
  std::lock_guard lock(impl_->mu);
  GenerateStats stats;
  const auto& ids = req.prompt.ids;
  if (ids.empty()) throw std::invalid_argument("Qwen4ExpGpuBackend::generate_request: empty prompt");
  stats.prompt_tokens = static_cast<int>(ids.size());

  auto& sess = *impl_->default_session;
  auto& st = *sess.state_;
  BackendStateIdentity req_id = profile_.state_identity;
  req_id.cache_namespace = req.cache_namespace;

  std::size_t start = 0;
  if (req.use_prefix_cache && st.identity.cache_namespace == req_id.cache_namespace && !st.committed_tokens.empty()) {
    const std::size_t tip = st.committed_tokens.size();
    if (tip <= ids.size() && std::equal(st.committed_tokens.begin(), st.committed_tokens.end(), ids.begin())) {
      start = tip;
      stats.reused_tokens = static_cast<int>(tip);
      stats.reuse = Reuse::Extend;
      stats.restored_bytes = 0;
      stats.restore_seconds = 0.0;
    } else {
      sess.reset();
      st.identity = req_id;
    }
  } else {
    sess.reset();
    st.identity = req_id;
  }

  const auto tp0 = std::chrono::steady_clock::now();
  std::vector<float> logits;
  if (start == ids.size() && st.has_last_logits) {
    logits = st.last_logits;
  } else {
    if (start == ids.size()) {
      sess.reset();
      st.identity = req_id;
      start = 0;
      stats.reused_tokens = 0;
      stats.reuse = Reuse::None;
    }
    for (std::size_t i = start; i < ids.size(); ++i) {
      const bool need_logits = (i + 1 == ids.size()) && (max_new_tokens > 0);
      logits = forward_token_logits_gpu(sess, ids[i], need_logits);
    }
  }
  stats.prefill_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
  if (on_prefill) on_prefill(stats);

  if (max_new_tokens <= 0) return stats;

  std::mt19937_64 rng(sampling.seed.value_or(0x9e3779b97f4a7c15ull));
  const auto td0 = std::chrono::steady_clock::now();
  const int limit = sampling.forced.empty() ? max_new_tokens : std::min(max_new_tokens, int(sampling.forced.size()));
  for (int step = 0; step < limit; ++step) {
    const std::int32_t next_tok =
        sampling.forced.empty() ? sample_logits_cpu(logits, sampling, rng) : sampling.forced[static_cast<std::size_t>(step)];
    ++stats.generated_tokens;
    const std::int32_t emit[1] = {next_tok};
    const bool keep_going = on_tokens(std::span<const std::int32_t>(emit, 1));
    const bool is_stop = std::find(stop_ids.begin(), stop_ids.end(), next_tok) != stop_ids.end();
    if (!keep_going || is_stop || step + 1 == limit) break;
    logits = forward_token_logits_gpu(sess, next_tok, /*compute_logits=*/true);
  }
  stats.decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - td0).count();
  return stats;
}

void Qwen4ExpGpuBackend::evaluate_teacher_forced(std::span<const std::int32_t> token_ids,
                                                 std::span<const int> capture_positions, bool use_prefix_cache,
                                                 const LogitCallback& on_logits,
                                                 std::span<const std::uint8_t> /*token_mask*/) {
  BackendRequest req;
  req.prompt.ids = token_ids;
  req.use_prefix_cache = use_prefix_cache;
  evaluate_teacher_forced(req, capture_positions, on_logits);
}

void Qwen4ExpGpuBackend::evaluate_teacher_forced(const BackendRequest& req, std::span<const int> capture_positions,
                                                 const LogitCallback& on_logits) {
  std::lock_guard lock(impl_->mu);
  auto& sess = *impl_->default_session;
  sess.reset();
  for (std::size_t i = 0; i < req.prompt.ids.size(); ++i) {
    const bool capture =
        std::find(capture_positions.begin(), capture_positions.end(), static_cast<int>(i)) != capture_positions.end();
    const auto logits = forward_token_logits_gpu(sess, req.prompt.ids[i], capture);
    if (capture && on_logits) {
      on_logits(LogitStep{static_cast<int>(i), logits});
    }
  }
}

void register_qwen4exp_cuda_backend() {
  static std::once_flag once;
  std::call_once(once, []() {
    auto& reg = BackendRegistry::instance();
    if (reg.find(ModelFamily::Qwen4Exp).has_value()) return;
    reg.register_backend(BackendRegistration{
        ModelFamily::Qwen4Exp,
        "qwen4exp-cuda",
        [](const RuntimeOptions& rt) -> std::unique_ptr<ModelBackend> {
          Qwen4ExpGpuLoadOptions opts;
          opts.model_dir = rt.model_dir;
          opts.max_ctx = rt.max_ctx > 0 ? rt.max_ctx : 4096;
          opts.prefix_cache = rt.prefix_cache;
          if (rt.offload_vram_mb > 0) {
            opts.expert_cache_bytes = static_cast<std::uint64_t>(rt.offload_vram_mb) * 1024ull * 1024ull;
          }
          return std::make_unique<Qwen4ExpGpuBackend>(opts);
        },
    });
  });
}

}  // namespace ingine::backends::qwen4exp

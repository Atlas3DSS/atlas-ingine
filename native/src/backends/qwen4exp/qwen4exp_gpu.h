// Native CUDA ModelBackend for Qwen3.8-Flash-Next (qwen4exp, #849).
// Architecture:
//   - 4.95B dense backbone (HC, 36x GDN, 12x FullAttn+QSA, L1 PLE proj, MoE routers & shared experts,
//     lm_head/token_embd) 100% resident in GPU VRAM.
//   - 51.20B PLE n-gram table (per_layer_token_embd.weight, 54.4 GB Q8_0) 100% in host mmap/pread
//     (0 GB VRAM; O(1) 16-row gather + 10 KB H2D per token).
//   - 120.80B routed MoE experts (48 layers x 512 experts, top-10 active):
//       * AllResident in VRAM when GPU has headroom (e.g. Elastic NVFP4/ATLAS3/ATLAS2 on 80-96 GB GPUs,
//         or small synthetic checkpoints on any GPU).
//       * TieredExpertCache (LRU GPU cache + pinned-host async staging from mmap/pread) when GPU VRAM
//         is smaller than total expert bytes (enables serving 177B UD-Q6_K_XL on 24-32 GB GPUs!).
#pragma once

#include "backends/qwen4exp/qwen4exp_config.h"
#include "backends/qwen4exp/qwen4exp_gguf.h"
#include "ingine/model_backend.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ingine::backends::qwen4exp {

struct Qwen4ExpGpuLoadOptions {
  std::filesystem::path model_dir;
  int max_ctx = 4096;
  int device_id = 0;
  bool verify_pins = false;
  bool prefix_cache = true;
  // Optional cap in bytes for the GPU expert LRU cache (0 = auto-size from free VRAM).
  std::uint64_t expert_cache_bytes = 0;
  std::optional<ingine::model::CheckpointPinSpec> expected_pins;
};

struct Qwen4ExpGpuLoadStats {
  double load_ms = 0.0;
  std::uint64_t dense_device_bytes = 0;
  std::uint64_t expert_device_bytes = 0;
  std::uint64_t expert_cache_slots = 0;
  std::uint64_t kv_state_device_bytes = 0;
  std::uint64_t scratch_device_bytes = 0;
  bool all_experts_resident = false;
  bool pins_verified = false;
};

struct Qwen4ExpExpertCacheStats {
  std::uint64_t hits = 0;
  std::uint64_t misses = 0;
  std::uint64_t slots_capacity = 0;
  std::uint64_t slots_used = 0;
};

class Qwen4ExpGpuBackend;

class Qwen4ExpGpuSession final : public ingine::model::BackendSession {
 public:
  ~Qwen4ExpGpuSession() override;

  void reset() override;
  std::size_t state_bytes() const noexcept override;
  std::span<const std::int32_t> committed_tokens() const noexcept override;
  std::span<const std::uint8_t> committed_token_mask() const noexcept override;
  const ingine::model::BackendStateIdentity& identity() const noexcept override;
  void truncate(std::size_t new_token_count) override;
  std::size_t restorable_prefix_len(std::size_t max_prefix_len) const noexcept override;
  bool try_restore_prefix(std::size_t token_count) override;

 private:
  friend class Qwen4ExpGpuBackend;
  explicit Qwen4ExpGpuSession(Qwen4ExpGpuBackend& owner);

  struct StateImpl;
  Qwen4ExpGpuBackend& owner_;
  std::unique_ptr<StateImpl> state_;
};

class Qwen4ExpGpuBackend final : public ingine::model::ModelBackend {
 public:
  explicit Qwen4ExpGpuBackend(const Qwen4ExpGpuLoadOptions& options);
  ~Qwen4ExpGpuBackend() override;

  Qwen4ExpGpuBackend(const Qwen4ExpGpuBackend&) = delete;
  Qwen4ExpGpuBackend& operator=(const Qwen4ExpGpuBackend&) = delete;

  [[nodiscard]] const Qwen4ExpConfig& config() const noexcept;
  [[nodiscard]] const Qwen4ExpGpuLoadStats& load_stats() const noexcept { return load_stats_; }
  [[nodiscard]] Qwen4ExpExpertCacheStats expert_cache_stats() const noexcept;

  // ingine::model::Generator
  std::string model_id() const override;
  std::string model_family() const override { return "qwen4exp"; }
  int max_model_len() const override;
  bool supports_logit_masker() const override { return false; }
  std::vector<ingine::model::KvCacheMode> supported_kv_cache_modes() const override {
    return {ingine::model::KvCacheMode::PerPass};
  }
  int loop_passes() const override { return 1; }
  ingine::model::PrefixCacheStats cache_stats() const override;

  using ingine::model::ModelBackend::generate;
  ingine::model::GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens,
                                        const ingine::model::Sampling& sampling,
                                        std::span<const std::int32_t> stop_ids,
                                        const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override;
  ingine::model::GenerateStats generate(const ingine::model::Prompt& prompt, int max_new_tokens,
                                        const ingine::model::Sampling& sampling,
                                        std::span<const std::int32_t> stop_ids,
                                        const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                                        const ingine::model::PrefillCallback& on_prefill = {}) override;

  // ingine::model::ModelBackend
  const ingine::model::BackendCapabilities& capabilities() const override { return profile_.capabilities; }
  const ingine::model::WorkerProfile& worker_profile() const override { return profile_; }
  const ingine::model::BackendStateIdentity& state_identity() const override { return profile_.state_identity; }
  void reset_session() override;

  std::unique_ptr<ingine::model::BackendSession> create_session() override;
  ingine::model::BackendSession* acquire_session(int slot = 0) override;
  void release_session(int slot = 0) override;

  ingine::model::GenerateStats generate_request(const ingine::model::BackendRequest& req, int max_new_tokens,
                                                const ingine::model::Sampling& sampling,
                                                std::span<const std::int32_t> stop_ids,
                                                const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                                                const ingine::model::PrefillCallback& on_prefill = {}) override;

  ingine::model::ResidencyReport residency_report() const override;
  ingine::model::AdmissionEstimate admission_estimate(int max_ctx, int max_batch) const override;
  std::uint64_t evict_caches(std::uint64_t target_free_bytes) override;

  void evaluate_teacher_forced(std::span<const std::int32_t> token_ids, std::span<const int> capture_positions,
                               bool use_prefix_cache, const ingine::model::LogitCallback& on_logits,
                               std::span<const std::uint8_t> token_mask = {}) override;
  void evaluate_teacher_forced(const ingine::model::BackendRequest& req, std::span<const int> capture_positions,
                               const ingine::model::LogitCallback& on_logits) override;

  // Direct single-token step for deterministic GPU-vs-CPU full-model parity tests.
  std::vector<float> forward_token_logits_gpu(Qwen4ExpGpuSession& session, std::int32_t token_id,
                                              bool compute_logits = true);

 private:
  friend class Qwen4ExpGpuSession;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ingine::model::WorkerProfile profile_;
  Qwen4ExpGpuLoadStats load_stats_;
};

// Idempotent registration of ModelFamily::Qwen4Exp into ingine::model::BackendRegistry.
void register_qwen4exp_cuda_backend();

}  // namespace ingine::backends::qwen4exp

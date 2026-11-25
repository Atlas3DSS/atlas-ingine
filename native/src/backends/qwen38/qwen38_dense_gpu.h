#pragma once

#include "backends/qwen38/qwen38_backend.h"
#include "backends/qwen38/qwen38_config.h"
#include "backends/qwen38/qwen38_manifest.h"
#include "ingine/model_backend.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ingine::backends::qwen38 {

struct Qwen38DenseLoadOptions {
  std::filesystem::path model_dir;
  int max_ctx = 4224;
  int prefill_chunk = 256;
  bool verify_pins = true;
  bool use_calibrated_fp8 = true;
  std::string revision = "482ca0f";
  int device_id = 0;
  std::optional<ingine::model::CheckpointPinSpec> expected_pins;
};

struct Qwen38DenseLoadStats {
  double cold_manifest_sha_ms = 0.0;
  double cold_safetensors_load_ms = 0.0;
  std::uint64_t weights_device_bytes = 0;
  std::uint64_t state_device_bytes = 0;
  std::uint64_t scratch_device_bytes = 0;
  std::uint64_t peak_vram_bytes = 0;
  std::uint64_t host_rss_bytes = 0;
  bool pins_verified = false;
};

struct Qwen38GemmParityReport {
  float nvfp4_max_abs = 0.0f;
  float fp8_cal_max_abs = 0.0f;
  float bf16_max_abs = 0.0f;
  float batch_invariance_max_abs = 0.0f;
  float gdn_prefill_vs_step_max_abs = 0.0f;
  float gdn_batch_invariance_max_abs = 0.0f;
  bool passed = false;
};

class Qwen38DenseBackend;

class Qwen38DenseSession final : public ingine::model::BackendSession {
 public:
  ~Qwen38DenseSession() override;

  void reset() override;
  std::size_t state_bytes() const noexcept override;
  std::span<const std::int32_t> committed_tokens() const noexcept override { return committed_tokens_; }
  const ingine::model::BackendStateIdentity& identity() const noexcept override;

 private:
  friend class Qwen38DenseBackend;
  explicit Qwen38DenseSession(Qwen38DenseBackend& owner);

  struct StateImpl;
  Qwen38DenseBackend& owner_;
  std::vector<std::int32_t> committed_tokens_;
  std::unique_ptr<StateImpl> state_;
};

class Qwen38DenseBackend final : public ingine::model::ModelBackend {
 public:
  explicit Qwen38DenseBackend(const Qwen38DenseLoadOptions& options);
  ~Qwen38DenseBackend() override;

  Qwen38DenseBackend(const Qwen38DenseBackend&) = delete;
  Qwen38DenseBackend& operator=(const Qwen38DenseBackend&) = delete;

  // Lightweight unit check comparing batched FP32 GEMM + GDN exact recurrent prefill against M=1 and Gate 1 kernels.
  static Qwen38GemmParityReport self_check_gemm_parity(int device_id = 0);

  [[nodiscard]] const Qwen38Manifest& manifest() const noexcept;
  [[nodiscard]] const Qwen38Config& config() const noexcept { return manifest().config; }
  [[nodiscard]] const Qwen38DenseLoadStats& load_stats() const noexcept { return load_stats_; }

  // ingine::model::Generator
  std::string model_id() const override;
  int max_model_len() const override;
  using ingine::model::ModelBackend::generate;
  ingine::model::GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens,
                                        const ingine::model::Sampling& sampling,
                                        std::span<const std::int32_t> stop_ids,
                                        const std::function<bool(std::span<const std::int32_t>)>& on_tokens) override;

  // ingine::model::ModelBackend
  const ingine::model::BackendCapabilities& capabilities() const override { return profile_.capabilities; }
  const ingine::model::WorkerProfile& worker_profile() const override { return profile_; }
  const ingine::model::BackendStateIdentity& state_identity() const override { return profile_.state_identity; }
  void reset_session() override;

  std::unique_ptr<ingine::model::BackendSession> create_session() override;
  ingine::model::BackendSession* acquire_session(int slot = 0) override;
  void release_session(int slot = 0) override;

  ingine::model::ResidencyReport residency_report() const override;
  ingine::model::AdmissionEstimate admission_estimate(int max_ctx, int max_batch) const override;

  void evaluate_teacher_forced(std::span<const std::int32_t> token_ids, std::span<const int> capture_positions,
                               bool use_prefix_cache, const ingine::model::LogitCallback& on_logits,
                               std::span<const std::uint8_t> token_mask = {}) override;
  void evaluate_teacher_forced(const ingine::model::BackendRequest& req, std::span<const int> capture_positions,
                               const ingine::model::LogitCallback& on_logits) override;

  // Explicit mode entry points for Gate 2 chunked-vs-stepwise parity & telemetry.
  void evaluate_teacher_forced_mode(std::span<const std::int32_t> token_ids, std::span<const int> capture_positions,
                                    bool use_chunk_prefill, const ingine::model::LogitCallback& on_logits,
                                    double* prefill_ms_out = nullptr, Qwen38DenseSession* session = nullptr,
                                    bool reset_before = true);

  void generate_greedy_exact(std::span<const std::int32_t> prompt_ids, int max_new_tokens, bool use_chunk_prefill,
                             bool stop_on_eos, std::vector<std::int32_t>& out_tokens,
                             double* prefill_ms_out = nullptr, double* decode_ms_out = nullptr,
                             Qwen38DenseSession* session = nullptr,
                             const ingine::model::LogitCallback& on_step_logits = {});

 private:
  friend class Qwen38DenseSession;
  struct Impl;
  std::unique_ptr<Impl> impl_;
  ingine::model::WorkerProfile profile_;
  Qwen38DenseLoadStats load_stats_;
};

}  // namespace ingine::backends::qwen38

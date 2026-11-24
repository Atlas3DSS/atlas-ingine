// Device-side manager for mech-interp & persona hooks in ingine_model.
// Owns one fixed-address PersonaDeviceTable on the device so hooked CUDA graphs never need recapture
// when switching persona profiles or tap configurations between requests.
#pragma once

#include "core/arena.h"
#include "ingine/persona.h"
#include "kernels/common/persona_abi.h"
#include "model/model_data.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace ingine::model {

struct DepthTapReq {
  int visit = -1;  // 0..47, or -1 for OutputRes
  persona::DepthSiteKind site = persona::DepthSiteKind::FfnRes;
  bool capture_scores = true;
  bool capture_states = true;
};

struct ExpertTapReq {
  int visit = 0;
  int expert = 0;
  bool per_token = false;
};

struct PersonaTapConfig {
  int max_pos = 0;
  std::vector<int> routing_visits;
  bool routing_logits = false;
  std::vector<int> ffn_inp_visits;
  bool ffn_inp_last_only = false;
  std::vector<DepthTapReq> depth_taps;
  std::vector<ExpertTapReq> expert_taps;
  int expert_tok_cap = 0;

  bool empty() const {
    return routing_visits.empty() && ffn_inp_visits.empty() && depth_taps.empty() && expert_taps.empty();
  }
};

struct NeuronStatReadout {
  int visit = 0;
  int expert = 0;
  int neuron = 0;
  float gain = 1.0f;
  float offset = 0.0f;
  std::uint32_t hits = 0;
  float pre_mean = 0.0f;
  float post_mean = 0.0f;
  std::int32_t last_pos = -1;
  float pre_last = 0.0f;
  float post_last = 0.0f;
};

struct ExpertTapTargetReadout {
  int visit = 0;
  int expert = 0;
  std::uint32_t hits = 0;
  std::int32_t last_pos = -1;
  std::vector<float> mean;  // [INTER]
  std::vector<float> last;  // [INTER]
};

struct ExpertTapTokenRow {
  int tap_idx = 0;
  int visit = 0;
  int expert = 0;
  int pos = 0;
  int slot = 0;
  std::vector<float> act;  // [INTER]
};

struct ExpertTapReadout {
  std::vector<ExpertTapTargetReadout> targets;
  std::vector<ExpertTapTokenRow> tokens;
  int total_tokens_captured = 0;
  bool truncated = false;
};

struct RoutingTapReadout {
  int visit = 0;
  int n_pos = 0;
  std::vector<std::int32_t> idx;  // [n_pos * TOP_K]
  std::vector<float> wts;         // [n_pos * TOP_K]
  std::vector<float> logits;      // [n_pos * EXPERTS] or empty
};

struct DepthTapReadout {
  int visit = -1;
  persona::DepthSiteKind site = persona::DepthSiteKind::FfnRes;
  int n_pos = 0;
  int n_src = 0;
  int n_q = 0;
  std::vector<float> scores;  // [n_pos * n_q * n_src]
  std::vector<float> states;  // [n_pos * n_src * HIDDEN]
};

// Returns (n_src, n_q) for a depth-mix site at chronological visit `visit` (or -1 for OutputRes).
std::pair<int, int> depth_site_shape(int visit, persona::DepthSiteKind site);

class PersonaRuntime {
 public:
  PersonaRuntime();
  ~PersonaRuntime() = default;
  PersonaRuntime(const PersonaRuntime&) = delete;
  PersonaRuntime& operator=(const PersonaRuntime&) = delete;

  const kernels::persona::PersonaDeviceTable* device_table() const {
    return static_cast<const kernels::persona::PersonaDeviceTable*>(table_dev_.p);
  }
  bool active() const { return active_; }
  std::uint64_t active_cache_key() const { return active_cache_key_; }

  // Uploads hook descriptors for `profile` (may be null) and `taps` into the fixed device table.
  // Passing (nullptr, empty taps) disables hooks (fast unhooked path).
  void configure(const persona::PersonaProfile* profile, const PersonaTapConfig& taps, cudaStream_t stream);

  // Temporarily uploads an all-zero table to table_dev_ (used when capturing hooked CUDA graphs).
  void upload_zero_table(cudaStream_t stream);

  // Resets accumulated neuron stats and 3a tap counters/sums in place without reallocating.
  void reset_stats(cudaStream_t stream);

  std::vector<NeuronStatReadout> read_neuron_stats(cudaStream_t stream) const;
  ExpertTapReadout read_expert_taps(cudaStream_t stream) const;
  RoutingTapReadout read_routing_tap(int visit, int n_pos, cudaStream_t stream) const;
  DepthTapReadout read_depth_tap(int visit, persona::DepthSiteKind site, int n_pos, cudaStream_t stream) const;
  std::vector<float> read_ffn_inp_tap(int visit, int n_pos, cudaStream_t stream) const;

 private:
  struct NeuronMeta {
    int visit = 0;
    int local_idx = 0;
    int expert = 0;
    int neuron = 0;
    float gain = 1.0f;
    float offset = 0.0f;
  };
  struct ExpertTapMeta {
    int visit = 0;
    int expert = 0;
    bool per_token = false;
  };
  struct DepthTapMeta {
    int visit = -1;
    persona::DepthSiteKind site = persona::DepthSiteKind::FfnRes;
    int max_pos = 0;
    int n_src = 0;
    int n_q = 0;
    float* scores_dev = nullptr;
    float* states_dev = nullptr;
  };

  DeviceBuffer table_dev_;  // fixed-address PersonaDeviceTable
  kernels::persona::PersonaDeviceTable host_table_{};
  bool active_ = false;
  std::uint64_t active_cache_key_ = 0;
  std::string active_fingerprint_;
  bool active_has_taps_ = false;

  std::vector<DeviceBuffer> backing_;
  std::vector<NeuronMeta> neuron_meta_;
  std::vector<ExpertTapMeta> expert_tap_meta_;
  std::vector<DepthTapMeta> depth_tap_meta_;

  // Direct device pointers inside backing_ for fast reset / readback.
  std::uint32_t* tap_hits_dev_ = nullptr;
  std::int32_t* tap_last_pos_dev_ = nullptr;
  float* tap_sum_dev_ = nullptr;
  float* tap_last_dev_ = nullptr;
  std::int32_t* tap_tok_count_dev_ = nullptr;
  std::int32_t* tap_tok_meta_dev_ = nullptr;
  float* tap_tok_act_dev_ = nullptr;
  int tap_tok_cap_ = 0;
};

}  // namespace ingine::model

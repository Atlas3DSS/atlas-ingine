#include "model/persona_runtime.h"

#include "core/device.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace {

using json = nlohmann::json;
namespace kp = kernels::persona;

int blocks_count_for_layer(int l) {
  if (l < 1) return 0;
  if (l < 7) return 1;
  if (l < 13) return 2;
  if (l < 19) return 3;
  return 4;
}

kp::DepthHookSite& depth_slot(kp::PersonaDeviceTable& tbl, int visit, persona::DepthSiteKind site) {
  if (site == persona::DepthSiteKind::OutputRes) return tbl.depth_output;
  if (visit < 0 || visit >= NUM_VISITS) throw std::invalid_argument("depth hook visit out of range");
  switch (site) {
    case persona::DepthSiteKind::AttnRes: return tbl.depth_attn[visit];
    case persona::DepthSiteKind::FfnRes: return tbl.depth_ffn[visit];
    case persona::DepthSiteKind::LoopMsg: return tbl.depth_msg[visit];
    default: return tbl.depth_output;
  }
}

void load_donor_route_file(const std::string& path, std::vector<std::int32_t>& ids, std::vector<float>& wts) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open route_replay donor_file: " + path);
  std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const json j = json::parse(raw);
  if (!j.is_object() || !j.contains("ids") || !j["ids"].is_array())
    throw std::runtime_error("route_replay donor_file must be JSON object with 'ids' array");
  ids.clear();
  for (const auto& row : j["ids"]) {
    if (row.is_array()) {
      for (const auto& v : row) ids.push_back(v.get<std::int32_t>());
    } else {
      ids.push_back(row.get<std::int32_t>());
    }
  }
  wts.clear();
  if (j.contains("wts") && j["wts"].is_array()) {
    for (const auto& row : j["wts"]) {
      if (row.is_array()) {
        for (const auto& v : row) wts.push_back(v.get<float>());
      } else {
        wts.push_back(row.get<float>());
      }
    }
  }
}

void load_donor_depth_file(const std::string& path, std::vector<float>& scores, std::vector<float>& states, int& n_src,
                           int& n_q) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open depth_replace donor_file: " + path);
  std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  const json j = json::parse(raw);
  if (!j.is_object()) throw std::runtime_error("depth_replace donor_file must be JSON object");
  n_src = j.value("n_src", 0);
  n_q = j.value("n_q", 0);
  scores.clear();
  if (j.contains("scores") && j["scores"].is_array()) {
    scores = j["scores"].get<std::vector<float>>();
  }
  states.clear();
  if (j.contains("states") && j["states"].is_array()) {
    states = j["states"].get<std::vector<float>>();
  }
}

}  // namespace

std::pair<int, int> depth_site_shape(int visit, persona::DepthSiteKind site) {
  if (site == persona::DepthSiteKind::OutputRes) return {5, 1};
  if (visit < 0 || visit >= NUM_VISITS)
    throw std::invalid_argument("depth_site_shape: visit out of range: " + std::to_string(visit));
  const int l = VISITS[visit].layer;
  const int p = VISITS[visit].pass;
  const int nb = blocks_count_for_layer(l);
  switch (site) {
    case persona::DepthSiteKind::AttnRes: {
      const int nq = (l == 0) ? 1 : (is_full_layer(l) ? 3 : 1);
      return {nb + 1, nq};
    }
    case persona::DepthSiteKind::FfnRes: {
      const bool boundary = (l % BLOCK == 0);
      return {nb + (boundary ? 1 : 0) + 1, 1};
    }
    case persona::DepthSiteKind::LoopMsg: {
      if (!(is_loop_layer(l) && p > 0))
        throw std::invalid_argument("depth_site_shape: loop_msg invalid at visit " + std::to_string(visit));
      return {2, 4};
    }
    default: return {5, 1};
  }
}

PersonaRuntime::PersonaRuntime() : table_dev_(sizeof(kp::PersonaDeviceTable)) {
  std::memset(&host_table_, 0, sizeof(host_table_));
  table_dev_.fill(0);
}

void PersonaRuntime::upload_zero_table(cudaStream_t stream) {
  kp::PersonaDeviceTable zero{};
  CUDA_CHECK(cudaMemcpyAsync(table_dev_.p, &zero, sizeof(zero), cudaMemcpyHostToDevice, stream));
}

void PersonaRuntime::reset_stats(cudaStream_t stream) {
  for (int v = 0; v < NUM_VISITS; ++v) {
    const auto& ev = host_table_.expert[v];
    if (ev.neuron_stats && ev.n_neuron_entries > 0) {
      std::vector<kp::NeuronAffineStat> init(static_cast<std::size_t>(ev.n_neuron_entries));
      for (auto& st : init) st.last_pos = -1;
      CUDA_CHECK(cudaMemcpyAsync(ev.neuron_stats, init.data(), init.size() * sizeof(kp::NeuronAffineStat),
                                 cudaMemcpyHostToDevice, stream));
    }
  }
  const std::size_t nt = expert_tap_meta_.size();
  if (nt > 0) {
    CUDA_CHECK(cudaMemsetAsync(tap_hits_dev_, 0, nt * sizeof(std::uint32_t), stream));
    std::vector<std::int32_t> neg1(nt, -1);
    CUDA_CHECK(cudaMemcpyAsync(tap_last_pos_dev_, neg1.data(), nt * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemsetAsync(tap_sum_dev_, 0, nt * INTER * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(tap_last_dev_, 0, nt * INTER * sizeof(float), stream));
    if (tap_tok_count_dev_) CUDA_CHECK(cudaMemsetAsync(tap_tok_count_dev_, 0, sizeof(std::int32_t), stream));
  }
}

void PersonaRuntime::configure(const persona::PersonaProfile* profile, const PersonaTapConfig& taps,
                               cudaStream_t stream) {
  const bool want_active = (profile != nullptr && !profile->is_unhooked()) || !taps.empty();
  const std::string want_fp = (profile && !profile->is_unhooked()) ? profile->fingerprint_sha256 : std::string();
  const bool want_taps = !taps.empty();

  // Fast path: staying unhooked.
  if (!want_active && !active_) return;

  // Fast path: staying on the exact same non-tapped profile (just reset tiny stats buffers if any).
  if (want_active && active_ && !want_taps && !active_has_taps_ && !want_fp.empty() && want_fp == active_fingerprint_) {
    reset_stats(stream);
    return;
  }

  // Synchronize before releasing any previous device backing buffers that an earlier kernel might still read.
  if (!backing_.empty() || active_) CUDA_CHECK(cudaStreamSynchronize(stream));

  backing_.clear();
  neuron_meta_.clear();
  expert_tap_meta_.clear();
  depth_tap_meta_.clear();
  tap_hits_dev_ = nullptr;
  tap_last_pos_dev_ = nullptr;
  tap_sum_dev_ = nullptr;
  tap_last_dev_ = nullptr;
  tap_tok_count_dev_ = nullptr;
  tap_tok_meta_dev_ = nullptr;
  tap_tok_act_dev_ = nullptr;
  tap_tok_cap_ = 0;

  std::memset(&host_table_, 0, sizeof(host_table_));
  for (int v = 0; v < NUM_VISITS; ++v) {
    host_table_.router[v].visit = std::uint32_t(v);
    host_table_.router[v].force_expert = -1;
    host_table_.router[v].force_gain = 1.0f;
    for (int k = 0; k < TOP_K; ++k) host_table_.router[v].slot_perm[k] = std::int8_t(k);
  }

  if (!want_active) {
    active_ = false;
    active_cache_key_ = 0;
    active_fingerprint_.clear();
    active_has_taps_ = false;
    CUDA_CHECK(cudaMemcpyAsync(table_dev_.p, &host_table_, sizeof(host_table_), cudaMemcpyHostToDevice, stream));
    return;
  }

  auto alloc_raw = [&](std::size_t bytes) -> void* {
    backing_.emplace_back(std::max<std::size_t>(bytes, 16));
    return backing_.back().p;
  };
  auto upload_vec = [&](const auto& vec) -> auto* {
    using T = typename std::decay_t<decltype(vec)>::value_type;
    void* p = alloc_raw(vec.size() * sizeof(T));
    if (!vec.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(p, vec.data(), vec.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    return static_cast<T*>(p);
  };

  std::array<std::vector<kp::NeuronAffineEntry>, NUM_VISITS> neurons_by_visit;
  std::map<std::pair<int, int>, bool> expert_tap_map;  // (visit, expert) -> per_token
  std::map<int, bool> routing_tap_map;                 // visit -> capture_logits

  struct SteerAccum {
    std::vector<float> vec;
    int count = 0;
    float single_scale = 0.0f;
    bool has_single_zero = false;
  };
  std::array<SteerAccum, NUM_VISITS> steer_attn_acc, steer_mlp_acc;

  if (profile) {
    for (const persona::InterventionSpec& iv : profile->interventions) {
      switch (iv.type) {
        case persona::InterventionType::RouterAblate: {
          if (iv.visit < 0 || iv.visit >= NUM_VISITS || iv.expert < 0 || iv.expert >= EXPERTS)
            throw std::invalid_argument("router_ablate: visit/expert out of range");
          auto& rv = host_table_.router[iv.visit];
          rv.flags |= kp::kRouterAblate;
          rv.ablate_mask[iv.expert / 64] |= (1ull << (iv.expert % 64));
          break;
        }
        case persona::InterventionType::RouterForce: {
          if (iv.visit < 0 || iv.visit >= NUM_VISITS || iv.expert < 0 || iv.expert >= EXPERTS)
            throw std::invalid_argument("router_force: visit/expert out of range");
          auto& rv = host_table_.router[iv.visit];
          if ((rv.flags & kp::kRouterForce) != 0u)
            throw std::invalid_argument("router_force: multiple force specs at visit " + std::to_string(iv.visit));
          rv.flags |= kp::kRouterForce;
          rv.force_expert = iv.expert;
          rv.force_bias = iv.bias;
          rv.force_gain = iv.gain;
          break;
        }
        case persona::InterventionType::RouterSlotPermute: {
          auto apply_perm = [&](int v) {
            auto& rv = host_table_.router[v];
            rv.flags |= kp::kRouterSlotPermute;
            for (int k = 0; k < TOP_K; ++k) rv.slot_perm[k] = iv.perm[std::size_t(k)];
          };
          if (iv.all_visits) {
            for (int v = 0; v < NUM_VISITS; ++v) apply_perm(v);
          } else {
            if (iv.visit < 0 || iv.visit >= NUM_VISITS)
              throw std::invalid_argument("router_slot_permute: visit out of range");
            apply_perm(iv.visit);
          }
          break;
        }
        case persona::InterventionType::RouteReplay: {
          if (iv.visit < 0 || iv.visit >= NUM_VISITS) throw std::invalid_argument("route_replay: visit out of range");
          std::vector<std::int32_t> ids = iv.inline_replay_ids;
          std::vector<float> wts = iv.inline_replay_wts;
          if (ids.empty() && !iv.donor_file.empty()) load_donor_route_file(iv.donor_file, ids, wts);
          if (ids.empty() || (ids.size() % TOP_K) != 0)
            throw std::invalid_argument("route_replay: donor ids missing or not a multiple of 8");
          const int n_pos = int(ids.size() / TOP_K);
          auto& rv = host_table_.router[iv.visit];
          rv.flags |= kp::kRouterReplayIds;
          rv.replay_ids = upload_vec(ids);
          rv.replay_pos_start = iv.pos_start;
          rv.replay_n_pos = n_pos;
          if (iv.replay_copy_weights) {
            if (int(wts.size()) != n_pos * TOP_K)
              throw std::invalid_argument("route_replay: copy mode requires wts of length n_pos * 8");
            rv.flags |= kp::kRouterReplayWts;
            rv.replay_wts = upload_vec(wts);
          }
          break;
        }
        case persona::InterventionType::NeuronAffine: {
          if (iv.visit < 0 || iv.visit >= NUM_VISITS || iv.expert < 0 || iv.expert >= EXPERTS || iv.neuron < 0 ||
              iv.neuron >= INTER)
            throw std::invalid_argument("neuron_affine: visit/expert/neuron out of range");
          const int local_idx = int(neurons_by_visit[iv.visit].size());
          kp::NeuronAffineEntry e;
          e.global_idx = local_idx;
          e.expert = iv.expert;
          e.neuron = iv.neuron;
          e.gain = iv.gain;
          e.offset = iv.offset;
          neurons_by_visit[iv.visit].push_back(e);
          neuron_meta_.push_back({iv.visit, local_idx, iv.expert, iv.neuron, iv.gain, iv.offset});
          break;
        }
        case persona::InterventionType::DepthReplace: {
          auto [exp_nsrc, exp_nq] = depth_site_shape(iv.visit, iv.depth_site);
          std::vector<float> scores = iv.inline_depth_scores;
          std::vector<float> states = iv.inline_depth_states;
          int n_src = iv.inline_depth_nsrc ? iv.inline_depth_nsrc : exp_nsrc;
          int n_q = iv.inline_depth_nq ? iv.inline_depth_nq : exp_nq;
          if (scores.empty() && states.empty() && !iv.donor_file.empty()) {
            load_donor_depth_file(iv.donor_file, scores, states, n_src, n_q);
            if (n_src == 0) n_src = exp_nsrc;
            if (n_q == 0) n_q = exp_nq;
          }
          if (n_src != exp_nsrc || n_q != exp_nq)
            throw std::invalid_argument("depth_replace: donor (n_src, n_q) does not match site shape");
          int n_pos = 0;
          if ((iv.depth_mode & 1u) != 0u) {
            if (scores.empty() || int(scores.size()) % (n_q * n_src) != 0)
              throw std::invalid_argument("depth_replace: read/both mode requires donor scores");
            n_pos = int(scores.size()) / (n_q * n_src);
          }
          if ((iv.depth_mode & 2u) != 0u) {
            if (states.empty() || int(states.size()) % (n_src * HIDDEN) != 0)
              throw std::invalid_argument("depth_replace: value/both mode requires donor states");
            const int np_st = int(states.size()) / (n_src * HIDDEN);
            if (n_pos != 0 && n_pos != np_st) throw std::invalid_argument("depth_replace: score/state n_pos mismatch");
            n_pos = np_st;
          }
          auto& dh = depth_slot(host_table_, iv.visit, iv.depth_site);
          dh.mode = iv.depth_mode;
          dh.src_mask = (iv.src_idx >= 0) ? (1u << iv.src_idx) : 0u;
          dh.pos_start = iv.pos_start;
          dh.n_pos = n_pos;
          dh.n_src = n_src;
          dh.n_q = n_q;
          if (!scores.empty()) dh.ov_scores = upload_vec(scores);
          if (!states.empty()) dh.ov_states = upload_vec(states);
          break;
        }
        case persona::InterventionType::ResidualSteer: {
          if (iv.visit < 0 || iv.visit >= NUM_VISITS) throw std::invalid_argument("residual_steer: visit out of range");
          if (iv.vector.size() != HIDDEN) throw std::invalid_argument("residual_steer: vector must have length 2304");
          if (iv.vector[persona::kForbiddenCoord] != 0.0f)
            throw std::invalid_argument("residual_steer: forbidden coordinate d_2027 must be 0.0");
          auto& acc = (iv.steer_site == persona::SteerSiteKind::AttnWrite) ? steer_attn_acc[iv.visit]
                                                                           : steer_mlp_acc[iv.visit];
          if (acc.count == 0) {
            acc.vec = iv.vector;
            acc.single_scale = iv.scale;
            acc.has_single_zero = (iv.scale == 0.0f);
            acc.count = 1;
          } else {
            if (acc.count == 1) {
              for (int c = 0; c < HIDDEN; ++c) acc.vec[c] *= acc.single_scale;
              acc.single_scale = 1.0f;
              acc.has_single_zero = false;
            }
            for (int c = 0; c < HIDDEN; ++c) acc.vec[c] += iv.scale * iv.vector[c];
            acc.vec[persona::kForbiddenCoord] = 0.0f;
            acc.count++;
          }
          break;
        }
        case persona::InterventionType::TapExpert: {
          if (iv.visit < 0 || iv.visit >= NUM_VISITS || iv.expert < 0 || iv.expert >= EXPERTS)
            throw std::invalid_argument("tap_expert: visit/expert out of range");
          auto& pt = expert_tap_map[{iv.visit, iv.expert}];
          pt = pt || iv.per_token;
          break;
        }
        case persona::InterventionType::TapRouting: {
          if (iv.all_visits) {
            for (int v = 0; v < NUM_VISITS; ++v) routing_tap_map[v] = routing_tap_map[v] || iv.capture_logits;
          } else {
            if (iv.visit < 0 || iv.visit >= NUM_VISITS) throw std::invalid_argument("tap_routing: visit out of range");
            routing_tap_map[iv.visit] = routing_tap_map[iv.visit] || iv.capture_logits;
          }
          break;
        }
        case persona::InterventionType::RouterLowRankFit:
        case persona::InterventionType::ExpertDownLowRankFit:
          throw std::runtime_error(
              "persona intervention requires Rungs 5-6 low-rank weight loader (schema validated; kernel weights not yet "
              "wired in this build)");
      }
    }
  }

  // Merge programmatic tap requests.
  for (const auto& req : taps.expert_taps) {
    if (req.visit < 0 || req.visit >= NUM_VISITS || req.expert < 0 || req.expert >= EXPERTS)
      throw std::invalid_argument("expert_tap request out of range");
    auto& pt = expert_tap_map[{req.visit, req.expert}];
    pt = pt || req.per_token;
  }
  for (int v : taps.routing_visits) {
    if (v < 0 || v >= NUM_VISITS) throw std::invalid_argument("routing_tap visit out of range");
    routing_tap_map[v] = routing_tap_map[v] || taps.routing_logits;
  }

  // Upload residual steering vectors.
  for (int v = 0; v < NUM_VISITS; ++v) {
    if (steer_attn_acc[v].count > 0) {
      steer_attn_acc[v].vec[persona::kForbiddenCoord] = 0.0f;
      host_table_.steer_attn[v].vec = upload_vec(steer_attn_acc[v].vec);
      host_table_.steer_attn[v].scale = steer_attn_acc[v].single_scale;
    }
    if (steer_mlp_acc[v].count > 0) {
      steer_mlp_acc[v].vec[persona::kForbiddenCoord] = 0.0f;
      host_table_.steer_mlp[v].vec = upload_vec(steer_mlp_acc[v].vec);
      host_table_.steer_mlp[v].scale = steer_mlp_acc[v].single_scale;
    }
  }

  // Upload per-visit 3b neuron affine tables & allocate stats.
  for (int v = 0; v < NUM_VISITS; ++v) {
    if (!neurons_by_visit[v].empty()) {
      const int n = int(neurons_by_visit[v].size());
      host_table_.expert[v].neuron_entries = upload_vec(neurons_by_visit[v]);
      std::vector<kp::NeuronAffineStat> init(static_cast<std::size_t>(n));
      for (auto& st : init) st.last_pos = -1;
      host_table_.expert[v].neuron_stats = upload_vec(init);
      host_table_.expert[v].n_neuron_entries = n;
    }
  }

  // Upload 3a expert taps.
  if (!expert_tap_map.empty()) {
    std::array<std::vector<kp::ExpertTapTarget>, NUM_VISITS> targets_by_visit;
    bool any_per_token = false;
    for (const auto& [key, pt] : expert_tap_map) {
      const int idx = int(expert_tap_meta_.size());
      expert_tap_meta_.push_back({key.first, key.second, pt});
      targets_by_visit[key.first].push_back({key.second, idx, pt ? 1u : 0u});
      any_per_token = any_per_token || pt;
    }
    const std::size_t nt = expert_tap_meta_.size();
    tap_hits_dev_ = static_cast<std::uint32_t*>(alloc_raw(nt * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(tap_hits_dev_, 0, nt * sizeof(std::uint32_t), stream));
    std::vector<std::int32_t> neg1(nt, -1);
    tap_last_pos_dev_ = upload_vec(neg1);
    tap_sum_dev_ = static_cast<float*>(alloc_raw(nt * INTER * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(tap_sum_dev_, 0, nt * INTER * sizeof(float), stream));
    tap_last_dev_ = static_cast<float*>(alloc_raw(nt * INTER * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(tap_last_dev_, 0, nt * INTER * sizeof(float), stream));
    if (any_per_token) {
      tap_tok_cap_ = taps.expert_tok_cap > 0 ? taps.expert_tok_cap : 4096;
      tap_tok_count_dev_ = static_cast<std::int32_t*>(alloc_raw(sizeof(std::int32_t)));
      CUDA_CHECK(cudaMemsetAsync(tap_tok_count_dev_, 0, sizeof(std::int32_t), stream));
      tap_tok_meta_dev_ = static_cast<std::int32_t*>(alloc_raw(std::size_t(tap_tok_cap_) * 3 * sizeof(std::int32_t)));
      tap_tok_act_dev_ = static_cast<float*>(alloc_raw(std::size_t(tap_tok_cap_) * INTER * sizeof(float)));
    }
    for (int v = 0; v < NUM_VISITS; ++v) {
      if (!targets_by_visit[v].empty()) {
        auto& ev = host_table_.expert[v];
        ev.tap_targets = upload_vec(targets_by_visit[v]);
        ev.n_tap_targets = int(targets_by_visit[v].size());
        ev.tap_hits = tap_hits_dev_;
        ev.tap_last_pos = tap_last_pos_dev_;
        ev.tap_sum = tap_sum_dev_;
        ev.tap_last = tap_last_dev_;
        ev.tap_tok_count = tap_tok_count_dev_;
        ev.tap_tok_meta = tap_tok_meta_dev_;
        ev.tap_tok_act = tap_tok_act_dev_;
        ev.tap_tok_cap = tap_tok_cap_;
      }
    }
  }

  const int tap_max_pos = taps.max_pos > 0 ? taps.max_pos : 256;

  // Allocate routing taps.
  for (const auto& [v, cap_logits] : routing_tap_map) {
    auto& rv = host_table_.router[v];
    rv.flags |= kp::kRouterTap;
    rv.tap_max_pos = tap_max_pos;
    std::vector<std::int32_t> neg1(std::size_t(tap_max_pos) * TOP_K, -1);
    rv.tap_idx = upload_vec(neg1);
    rv.tap_wts = static_cast<float*>(alloc_raw(std::size_t(tap_max_pos) * TOP_K * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(rv.tap_wts, 0, std::size_t(tap_max_pos) * TOP_K * sizeof(float), stream));
    if (cap_logits) {
      rv.tap_logits = static_cast<float*>(alloc_raw(std::size_t(tap_max_pos) * EXPERTS * sizeof(float)));
      CUDA_CHECK(cudaMemsetAsync(rv.tap_logits, 0, std::size_t(tap_max_pos) * EXPERTS * sizeof(float), stream));
    }
  }

  // Allocate ffn_moe_inp (xm) taps.
  for (int v : taps.ffn_inp_visits) {
    if (v < 0 || v >= NUM_VISITS) throw std::invalid_argument("ffn_inp_tap visit out of range");
    auto& ft = host_table_.ffn_inp_tap[v];
    ft.enabled = 1u;
    if (taps.ffn_inp_last_only) {
      ft.max_pos = 0;
      ft.last_out = static_cast<float*>(alloc_raw(HIDDEN * sizeof(float)));
      CUDA_CHECK(cudaMemsetAsync(ft.last_out, 0, HIDDEN * sizeof(float), stream));
    } else {
      ft.max_pos = tap_max_pos;
      ft.out = static_cast<float*>(alloc_raw(std::size_t(tap_max_pos) * HIDDEN * sizeof(float)));
      CUDA_CHECK(cudaMemsetAsync(ft.out, 0, std::size_t(tap_max_pos) * HIDDEN * sizeof(float), stream));
    }
  }

  // Deduplicate depth taps by (visit, site) before allocating so repeated requests
  // never overwrite host_table_ pointers while leaving stale zeroed entries in depth_tap_meta_.
  std::map<std::pair<int, int>, std::pair<bool, bool>> depth_tap_map;
  for (const auto& req : taps.depth_taps) {
    auto& flags = depth_tap_map[{req.visit, int(req.site)}];
    flags.first = flags.first || req.capture_scores;
    flags.second = flags.second || req.capture_states;
  }
  for (const auto& [key, flags] : depth_tap_map) {
    const int visit = key.first;
    const auto site = static_cast<persona::DepthSiteKind>(key.second);
    auto [n_src, n_q] = depth_site_shape(visit, site);
    auto& dh = depth_slot(host_table_, visit, site);
    dh.tap_enabled = 1u;
    dh.tap_max_pos = tap_max_pos;
    float* sc = nullptr;
    float* st = nullptr;
    if (flags.first) {
      sc = static_cast<float*>(alloc_raw(std::size_t(tap_max_pos) * n_q * n_src * sizeof(float)));
      CUDA_CHECK(cudaMemsetAsync(sc, 0, std::size_t(tap_max_pos) * n_q * n_src * sizeof(float), stream));
      dh.tap_scores = sc;
    }
    if (flags.second) {
      st = static_cast<float*>(alloc_raw(std::size_t(tap_max_pos) * n_src * HIDDEN * sizeof(float)));
      CUDA_CHECK(cudaMemsetAsync(st, 0, std::size_t(tap_max_pos) * n_src * HIDDEN * sizeof(float), stream));
      dh.tap_states = st;
    }
    depth_tap_meta_.push_back({visit, site, tap_max_pos, n_src, n_q, sc, st});
  }

  host_table_.active = 1u;
  active_ = true;
  active_cache_key_ = (profile && !profile->is_unhooked()) ? profile->cache_key : 0ull;
  active_fingerprint_ = want_fp;
  active_has_taps_ = want_taps;
  CUDA_CHECK(cudaMemcpyAsync(table_dev_.p, &host_table_, sizeof(host_table_), cudaMemcpyHostToDevice, stream));
}

std::vector<NeuronStatReadout> PersonaRuntime::read_neuron_stats(cudaStream_t stream) const {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<NeuronStatReadout> out;
  out.reserve(neuron_meta_.size());
  std::array<std::vector<kp::NeuronAffineStat>, NUM_VISITS> copied;
  for (int v = 0; v < NUM_VISITS; ++v) {
    const auto& ev = host_table_.expert[v];
    if (ev.neuron_stats && ev.n_neuron_entries > 0) {
      copied[v].resize(std::size_t(ev.n_neuron_entries));
      CUDA_CHECK(cudaMemcpy(copied[v].data(), ev.neuron_stats, copied[v].size() * sizeof(kp::NeuronAffineStat),
                            cudaMemcpyDeviceToHost));
    }
  }
  for (const auto& m : neuron_meta_) {
    const auto& st = copied[m.visit][std::size_t(m.local_idx)];
    NeuronStatReadout r;
    r.visit = m.visit;
    r.expert = m.expert;
    r.neuron = m.neuron;
    r.gain = m.gain;
    r.offset = m.offset;
    r.hits = st.hits;
    r.pre_mean = st.hits > 0 ? st.pre_sum / float(st.hits) : 0.0f;
    r.post_mean = st.hits > 0 ? st.post_sum / float(st.hits) : 0.0f;
    r.last_pos = st.last_pos;
    r.pre_last = st.pre_last;
    r.post_last = st.post_last;
    out.push_back(r);
  }
  return out;
}

ExpertTapReadout PersonaRuntime::read_expert_taps(cudaStream_t stream) const {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ExpertTapReadout res;
  const std::size_t nt = expert_tap_meta_.size();
  if (nt == 0) return res;
  std::vector<std::uint32_t> hits(nt);
  std::vector<std::int32_t> last_pos(nt);
  std::vector<float> sums(nt * INTER);
  std::vector<float> lasts(nt * INTER);
  CUDA_CHECK(cudaMemcpy(hits.data(), tap_hits_dev_, nt * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(last_pos.data(), tap_last_pos_dev_, nt * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(sums.data(), tap_sum_dev_, nt * INTER * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(lasts.data(), tap_last_dev_, nt * INTER * sizeof(float), cudaMemcpyDeviceToHost));
  res.targets.resize(nt);
  for (std::size_t i = 0; i < nt; ++i) {
    auto& t = res.targets[i];
    t.visit = expert_tap_meta_[i].visit;
    t.expert = expert_tap_meta_[i].expert;
    t.hits = hits[i];
    t.last_pos = last_pos[i];
    t.mean.resize(INTER);
    t.last.resize(INTER);
    const float inv = hits[i] > 0 ? 1.0f / float(hits[i]) : 0.0f;
    for (int k = 0; k < INTER; ++k) {
      t.mean[k] = sums[i * INTER + k] * inv;
      t.last[k] = lasts[i * INTER + k];
    }
  }
  if (tap_tok_count_dev_) {
    std::int32_t count = 0;
    CUDA_CHECK(cudaMemcpy(&count, tap_tok_count_dev_, sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    res.total_tokens_captured = count;
    res.truncated = count > tap_tok_cap_;
    const int n = std::min(count, tap_tok_cap_);
    if (n > 0) {
      std::vector<std::int32_t> meta(std::size_t(n) * 3);
      std::vector<float> acts(std::size_t(n) * INTER);
      CUDA_CHECK(cudaMemcpy(meta.data(), tap_tok_meta_dev_, meta.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(acts.data(), tap_tok_act_dev_, acts.size() * sizeof(float), cudaMemcpyDeviceToHost));
      res.tokens.resize(std::size_t(n));
      for (int i = 0; i < n; ++i) {
        auto& row = res.tokens[std::size_t(i)];
        row.tap_idx = meta[std::size_t(i) * 3 + 0];
        row.pos = meta[std::size_t(i) * 3 + 1];
        row.slot = meta[std::size_t(i) * 3 + 2];
        if (row.tap_idx >= 0 && std::size_t(row.tap_idx) < expert_tap_meta_.size()) {
          row.visit = expert_tap_meta_[std::size_t(row.tap_idx)].visit;
          row.expert = expert_tap_meta_[std::size_t(row.tap_idx)].expert;
        }
        row.act.assign(acts.begin() + std::ptrdiff_t(i) * INTER, acts.begin() + std::ptrdiff_t(i + 1) * INTER);
      }
    }
  }
  return res;
}

RoutingTapReadout PersonaRuntime::read_routing_tap(int visit, int n_pos, cudaStream_t stream) const {
  if (visit < 0 || visit >= NUM_VISITS) throw std::invalid_argument("read_routing_tap: visit out of range");
  const auto& rv = host_table_.router[visit];
  if ((rv.flags & kp::kRouterTap) == 0u || !rv.tap_idx)
    throw std::invalid_argument("read_routing_tap: routing tap not enabled at visit " + std::to_string(visit));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const int n = std::min(n_pos, rv.tap_max_pos);
  RoutingTapReadout out;
  out.visit = visit;
  out.n_pos = n;
  out.idx.resize(std::size_t(n) * TOP_K);
  out.wts.resize(std::size_t(n) * TOP_K);
  CUDA_CHECK(cudaMemcpy(out.idx.data(), rv.tap_idx, out.idx.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.wts.data(), rv.tap_wts, out.wts.size() * sizeof(float), cudaMemcpyDeviceToHost));
  if (rv.tap_logits) {
    out.logits.resize(std::size_t(n) * EXPERTS);
    CUDA_CHECK(cudaMemcpy(out.logits.data(), rv.tap_logits, out.logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
  }
  return out;
}

DepthTapReadout PersonaRuntime::read_depth_tap(int visit, persona::DepthSiteKind site, int n_pos,
                                               cudaStream_t stream) const {
  for (const auto& m : depth_tap_meta_) {
    if (m.visit == visit && m.site == site) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const int n = std::min(n_pos, m.max_pos);
      DepthTapReadout out;
      out.visit = visit;
      out.site = site;
      out.n_pos = n;
      out.n_src = m.n_src;
      out.n_q = m.n_q;
      if (m.scores_dev) {
        out.scores.resize(std::size_t(n) * m.n_q * m.n_src);
        CUDA_CHECK(cudaMemcpy(out.scores.data(), m.scores_dev, out.scores.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
      }
      if (m.states_dev) {
        out.states.resize(std::size_t(n) * m.n_src * HIDDEN);
        CUDA_CHECK(cudaMemcpy(out.states.data(), m.states_dev, out.states.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
      }
      return out;
    }
  }
  throw std::invalid_argument("read_depth_tap: requested depth tap not enabled");
}

std::vector<float> PersonaRuntime::read_ffn_inp_tap(int visit, int n_pos, cudaStream_t stream) const {
  if (visit < 0 || visit >= NUM_VISITS) throw std::invalid_argument("read_ffn_inp_tap: visit out of range");
  const auto& ft = host_table_.ffn_inp_tap[visit];
  if (!ft.enabled) throw std::invalid_argument("read_ffn_inp_tap: not enabled at visit " + std::to_string(visit));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  if (ft.out) {
    const int n = std::min(n_pos, ft.max_pos);
    std::vector<float> out(std::size_t(n) * HIDDEN);
    CUDA_CHECK(cudaMemcpy(out.data(), ft.out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
    return out;
  }
  std::vector<float> out(HIDDEN);
  CUDA_CHECK(cudaMemcpy(out.data(), ft.last_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  return out;
}

}  // namespace ingine::model

// CUDA-free bounded async GPU<->RAM expert swap queue & residency controller (#888).
//
// Invariants:
//   * Strict opt-in: max_swaps == 0 disables dynamic HOT promotion completely (0-ULP unchanged default).
//   * Strictly NO silent expert dropping (experts_dropped == 0 always); dropped *promotion requests* due
//     to bounded queue saturation are counted separately in dropped_bound().
//   * Only HOT slots with verified immutable backing (WARM pinned RAM or COLD NVMe FileSlice) are marked
//     evictable, ensuring evicted experts can always be re-fetched byte-identically.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ingine::model {

class BoundedSwapQueue {
 public:
  static constexpr int kLayers = 24;
  static constexpr int kExperts = 128;

  struct Entry {
    int layer = -1;
    int expert = -1;
  };

  struct HotSlotState {
    int layer = -1;
    int expert = -1;
    bool evictable = false;
    bool swap_inflight = false;
    std::uint64_t pin_gen = 0;
    std::uint64_t last_use = 0;
  };

  enum class EnqueueStatus : std::uint8_t {
    Disabled = 0,
    Queued = 1,
    AlreadyPending = 2,
    DroppedBound = 3,
  };

  explicit BoundedSwapQueue(int max_swaps = 0, std::uint32_t promote_threshold = 2, std::uint32_t hysteresis = 2,
                            std::uint64_t decay_interval = 4096)
      : max_swaps_(max_swaps),
        promote_threshold_(promote_threshold),
        hysteresis_(hysteresis),
        decay_interval_(decay_interval) {
    if (max_swaps_ < 0) {
      throw std::invalid_argument("BoundedSwapQueue: max_swaps must be >= 0");
    }
    if (max_swaps_ > 0) {
      ring_.resize(std::size_t(max_swaps_));
    }
  }

  bool enabled() const noexcept { return max_swaps_ > 0; }
  int capacity() const noexcept { return max_swaps_; }
  int size() const noexcept { return count_; }
  bool empty() const noexcept { return count_ == 0; }
  bool full() const noexcept { return count_ >= max_swaps_; }

  std::uint32_t live_hits(int l, int e) const noexcept { return live_hits_[l][e]; }
  void seed_hit(int l, int e, std::uint32_t v) noexcept { live_hits_[l][e] = v; }

  // Record one routed access to (l, e). If not already HOT and threshold is reached, enqueue a bounded
  // promotion request. Never drops the expert's computation (all TOP_K experts still execute).
  EnqueueStatus record_access(int l, int e, bool is_hot) noexcept {
    if (l < 0 || l >= kLayers || e < 0 || e >= kExperts) return EnqueueStatus::Disabled;
    if (live_hits_[l][e] < std::numeric_limits<std::uint32_t>::max()) ++live_hits_[l][e];
    ++access_counter_;
    if (decay_interval_ > 0 && (access_counter_ % decay_interval_) == 0) {
      decay_hits();
    }
    if (!enabled() || is_hot) return EnqueueStatus::Disabled;
    if (live_hits_[l][e] < promote_threshold_) return EnqueueStatus::Disabled;
    return try_enqueue(l, e);
  }

  EnqueueStatus try_enqueue(int l, int e) noexcept {
    if (!enabled()) return EnqueueStatus::Disabled;
    if (l < 0 || l >= kLayers || e < 0 || e >= kExperts) return EnqueueStatus::Disabled;
    if (queued_map_[l][e] || inflight_map_[l][e]) return EnqueueStatus::AlreadyPending;
    if (count_ >= max_swaps_) {
      ++dropped_bound_;
      return EnqueueStatus::DroppedBound;
    }
    ring_[std::size_t(tail_)] = Entry{l, e};
    tail_ = (tail_ + 1) % max_swaps_;
    ++count_;
    queued_map_[l][e] = true;
    ++queued_total_;
    return EnqueueStatus::Queued;
  }

  bool pop(Entry& out) noexcept {
    if (count_ == 0) return false;
    out = ring_[std::size_t(head_)];
    head_ = (head_ + 1) % max_swaps_;
    --count_;
    queued_map_[out.layer][out.expert] = false;
    return true;
  }

  void mark_inflight(int l, int e, bool v) noexcept {
    if (l >= 0 && l < kLayers && e >= 0 && e < kExperts) inflight_map_[l][e] = v;
  }
  bool is_inflight(int l, int e) const noexcept {
    return (l >= 0 && l < kLayers && e >= 0 && e < kExperts) ? inflight_map_[l][e] : false;
  }
  bool is_queued(int l, int e) const noexcept {
    return (l >= 0 && l < kLayers && e >= 0 && e < kExperts) ? queued_map_[l][e] : false;
  }

  // Pick the coldest evictable, unpinned, non-inflight HOT victim in `slots` that is strictly colder
  // than the candidate by at least `hysteresis_`. Never evicts an expert in `avoid_layer` (current layer)
  // unless `avoid_layer < 0`. Returns slot index >= 0 or -1 if no eligible victim exists.
  int select_victim(const std::vector<HotSlotState>& slots, int avoid_layer, std::uint64_t cur_pin_gen,
                    std::uint32_t cand_hits) const noexcept {
    int victim = -1;
    std::uint32_t best_hits = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t best_use = std::numeric_limits<std::uint64_t>::max();
    for (int i = 0; i < int(slots.size()); ++i) {
      const HotSlotState& s = slots[std::size_t(i)];
      if (!s.evictable || s.swap_inflight) continue;
      if (s.layer < 0 || s.expert < 0) continue;
      if (s.pin_gen == cur_pin_gen) continue;
      if (avoid_layer >= 0 && s.layer == avoid_layer) continue;
      const std::uint32_t vh = live_hits_[s.layer][s.expert];
      if (vh + hysteresis_ > cand_hits) continue;
      if (vh < best_hits || (vh == best_hits && s.last_use < best_use)) {
        best_hits = vh;
        best_use = s.last_use;
        victim = i;
      }
    }
    return victim;
  }

  void decay_hits() noexcept {
    for (auto& row : live_hits_) {
      for (auto& h : row) h >>= 1;
    }
  }

  void reset_stats() noexcept {
    queued_total_ = 0;
    dropped_bound_ = 0;
  }

  std::uint64_t queued_total() const noexcept { return queued_total_; }
  std::uint64_t dropped_bound() const noexcept { return dropped_bound_; }

 private:
  int max_swaps_ = 0;
  std::uint32_t promote_threshold_ = 2;
  std::uint32_t hysteresis_ = 2;
  std::uint64_t decay_interval_ = 4096;
  std::uint64_t access_counter_ = 0;

  std::vector<Entry> ring_;
  int head_ = 0;
  int tail_ = 0;
  int count_ = 0;

  std::array<std::array<std::uint32_t, kExperts>, kLayers> live_hits_{};
  std::array<std::array<bool, kExperts>, kLayers> queued_map_{};
  std::array<std::array<bool, kExperts>, kLayers> inflight_map_{};

  std::uint64_t queued_total_ = 0;
  std::uint64_t dropped_bound_ = 0;
};

}  // namespace ingine::model

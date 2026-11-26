// Pure-CPU, CUDA-free LRU slot index with pin-generation protection and explicit key erasure.
// Shared by VramLruSlab (core/paged_tier.h) and Gate 0 CPU lifecycle tests so production LRU
// residency semantics are verified directly without duplicating state-machine logic.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ingine {

class LruSlotIndex {
 public:
  static constexpr std::uint64_t kEmptyKey = std::numeric_limits<std::uint64_t>::max();

  struct AcquireResult {
    int slot = -1;
    bool hit = false;
    bool evicted = false;
    std::uint64_t evicted_key = kEmptyKey;
  };

  LruSlotIndex() = default;
  explicit LruSlotIndex(int num_slots) { reset(num_slots); }

  void reset(int num_slots) {
    index_.clear();
    slots_.clear();
    tick_ = 0;
    if (num_slots <= 0) return;
    slots_.resize(std::size_t(num_slots));
    index_.reserve(std::size_t(num_slots) * 2);
  }

  [[nodiscard]] int capacity() const noexcept { return int(slots_.size()); }

  [[nodiscard]] int find(std::uint64_t key) const noexcept {
    auto it = index_.find(key);
    return it == index_.end() ? -1 : it->second;
  }

  // Pin an already-resident key for `pin_gen` so it cannot be evicted by other acquires in the same
  // step/visit. Returns slot index on hit, -1 on miss.
  int touch_and_pin(std::uint64_t key, std::uint64_t pin_gen, bool priority = false) noexcept {
    auto it = index_.find(key);
    if (it == index_.end()) return -1;
    Slot& s = slots_[std::size_t(it->second)];
    s.last_use = ++tick_ + (priority ? std::uint64_t(slots_.size()) : 0ull);
    s.pin_gen = pin_gen;
    return it->second;
  }

  // Remove `key` from the index (e.g. when an expert is promoted into the permanent HOT tier).
  // Preserves `pin_gen` on the vacated slot so acquire() will not reuse the slot during the same
  // pin generation while an in-flight compute kernel may still reference it.
  bool erase(std::uint64_t key) noexcept {
    auto it = index_.find(key);
    if (it == index_.end()) return false;
    Slot& s = slots_[std::size_t(it->second)];
    s.key = kEmptyKey;
    s.last_use = 0;
    index_.erase(it);
    return true;
  }

  // Acquire a slot for `key`, pinning it at `pin_gen`. Hits return hit=true; misses prefer an
  // empty slot whose pin_gen != current pin_gen, else evict the least-recently-used unpinned slot.
  AcquireResult acquire(std::uint64_t key, std::uint64_t pin_gen, bool priority = false) {
    AcquireResult r{};
    if (slots_.empty()) throw std::runtime_error("LruSlotIndex::acquire on empty slab");
    if (auto it = index_.find(key); it != index_.end()) {
      Slot& s = slots_[std::size_t(it->second)];
      s.last_use = ++tick_ + (priority ? std::uint64_t(slots_.size()) : 0ull);
      s.pin_gen = pin_gen;
      r.slot = it->second;
      r.hit = true;
      return r;
    }
    int victim = -1;
    std::uint64_t best_use = std::numeric_limits<std::uint64_t>::max();
    for (int i = 0; i < int(slots_.size()); ++i) {
      if (slots_[std::size_t(i)].pin_gen == pin_gen) continue;
      if (slots_[std::size_t(i)].key == kEmptyKey) {
        victim = i;
        break;
      }
      if (slots_[std::size_t(i)].last_use < best_use) {
        best_use = slots_[std::size_t(i)].last_use;
        victim = i;
      }
    }
    if (victim < 0) {
      throw std::runtime_error("LruSlotIndex::acquire: all " + std::to_string(slots_.size()) +
                               " slots are pinned in the same step");
    }
    Slot& s = slots_[std::size_t(victim)];
    if (s.key != kEmptyKey) {
      r.evicted = true;
      r.evicted_key = s.key;
      index_.erase(s.key);
    }
    s.key = key;
    s.last_use = ++tick_ + (priority ? std::uint64_t(slots_.size()) : 0ull);
    s.pin_gen = pin_gen;
    index_[key] = victim;
    r.slot = victim;
    r.hit = false;
    return r;
  }

  void clear() noexcept {
    index_.clear();
    for (auto& s : slots_) {
      s.key = kEmptyKey;
      s.last_use = 0;
      s.pin_gen = 0;
    }
  }

 private:
  struct Slot {
    std::uint64_t key = kEmptyKey;
    std::uint64_t last_use = 0;
    std::uint64_t pin_gen = 0;
  };

  std::vector<Slot> slots_;
  std::unordered_map<std::uint64_t, int> index_;
  std::uint64_t tick_ = 0;
};

}  // namespace ingine

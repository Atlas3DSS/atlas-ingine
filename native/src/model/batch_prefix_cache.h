// CUDA-free refcounted paged full-KV page pool and exact prefix checkpoint cache for batched (B>1) decoding (#825).
//
// Invariants enforced by this module:
//   1. Only committed immutable full pages (all 64 rows written & stream-synchronized) may be shared across
//      active slots or cached checkpoints. Partial tail pages (<64 rows) are ALWAYS private to one slot.
//   2. Every physical page p tracks active_refs[p] (live slots) and cache_refs[p] (cached checkpoints).
//      A page returns to free_pages ONLY when active_refs[p] == 0 && cache_refs[p] == 0.
//   3. Evicting a checkpoint decrements cache_refs[p] without disturbing live sibling slots (active_refs[p] > 0).
//   4. Evictable cache pages (active_refs == 0 && cache_refs > 0) NEVER starve active admission or decode growth:
//      ensure_free_pages() evicts LRU checkpoints until enough physical pages are free (or cache is empty),
//      skipping an optional protected_entry_id currently being restored.
//   5. Hashes (hash_state_identity, hash_token_prefix) are index accelerators ONLY; every hit verifies exact
//      StateIdentity::operator== and exact std::equal over the full prefix token IDs.
#pragma once

#include "ingine/prefix_cache.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <list>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ingine::model {

class PagedKvPagePool {
 public:
  void init(int num_pages) {
    if (num_pages < 0) throw std::invalid_argument("PagedKvPagePool: negative num_pages");
    num_pages_ = num_pages;
    active_refs_.assign(std::size_t(num_pages), 0);
    cache_refs_.assign(std::size_t(num_pages), 0);
    committed_.assign(std::size_t(num_pages), false);
    free_pages_.resize(std::size_t(num_pages));
    for (int i = 0; i < num_pages; ++i) free_pages_[std::size_t(i)] = num_pages - 1 - i;
  }

  int num_pages() const noexcept { return num_pages_; }
  int free_count() const noexcept { return int(free_pages_.size()); }

  int evictable_count() const noexcept {
    int n = 0;
    for (int p = 0; p < num_pages_; ++p) {
      if (active_refs_[std::size_t(p)] == 0 && cache_refs_[std::size_t(p)] > 0) ++n;
    }
    return n;
  }

  int available_for_admission() const noexcept { return free_count() + evictable_count(); }

  int active_refs(int p) const {
    check_page(p);
    return active_refs_[std::size_t(p)];
  }
  int cache_refs(int p) const {
    check_page(p);
    return cache_refs_[std::size_t(p)];
  }
  bool is_committed(int p) const {
    check_page(p);
    return committed_[std::size_t(p)];
  }

  // Allocates a fresh private page for an active slot (active_refs = 1, cache_refs = 0, committed = false).
  int alloc_private_page() {
    if (free_pages_.empty()) return -1;
    const int p = free_pages_.back();
    free_pages_.pop_back();
    const auto idx = std::size_t(p);
    if (active_refs_[idx] != 0 || cache_refs_[idx] != 0 || committed_[idx])
      throw std::logic_error("PagedKvPagePool: corrupted free page state");
    active_refs_[idx] = 1;
    return p;
  }

  // Retains an already-committed immutable page for a new active slot (zero-copy sharing).
  void retain_committed_page(int p) {
    check_page(p);
    const auto idx = std::size_t(p);
    if (!committed_[idx] || (active_refs_[idx] == 0 && cache_refs_[idx] == 0))
      throw std::logic_error("PagedKvPagePool: cannot retain uncommitted or unowned page");
    ++active_refs_[idx];
  }

  // Releases one active-slot reference; returns page to free_pages iff no active or cache refs remain.
  void release_active_page(int p) {
    check_page(p);
    const auto idx = std::size_t(p);
    if (active_refs_[idx] <= 0) throw std::logic_error("PagedKvPagePool: active_refs underflow");
    --active_refs_[idx];
    if (active_refs_[idx] == 0 && cache_refs_[idx] == 0) {
      committed_[idx] = false;
      free_pages_.push_back(p);
    }
  }

  // Marks an active slot's full page as committed immutable and adds a cache reference.
  // Caller MUST have synchronized the GPU stream that wrote all 64 rows of `p` before calling this.
  void pin_for_cache(int p) {
    check_page(p);
    const auto idx = std::size_t(p);
    if (active_refs_[idx] <= 0 && !committed_[idx])
      throw std::logic_error("PagedKvPagePool: cannot pin unowned uncommitted page");
    committed_[idx] = true;
    ++cache_refs_[idx];
  }

  // Removes one cache reference; returns page to free_pages iff no active or cache refs remain.
  void unpin_from_cache(int p) {
    check_page(p);
    const auto idx = std::size_t(p);
    if (cache_refs_[idx] <= 0) throw std::logic_error("PagedKvPagePool: cache_refs underflow");
    --cache_refs_[idx];
    if (active_refs_[idx] == 0 && cache_refs_[idx] == 0) {
      committed_[idx] = false;
      free_pages_.push_back(p);
    }
  }

 private:
  void check_page(int p) const {
    if (p < 0 || p >= num_pages_) throw std::out_of_range("PagedKvPagePool: page id out of range");
  }

  int num_pages_ = 0;
  std::vector<int> active_refs_;
  std::vector<int> cache_refs_;
  std::vector<bool> committed_;
  std::vector<int> free_pages_;
};

template <typename Payload>
struct BatchPrefixEntry {
  std::uint64_t id = 0;
  StateIdentity identity{};
  std::uint64_t identity_hash = 0;
  std::vector<std::int32_t> tokens;  // exact prefix [0, L)
  std::uint64_t prefix_hash = 0;     // index accelerator ONLY
  std::vector<int> committed_pages;  // floor(L / kPageSize) physical pages pinned in PagedKvPagePool
  int tail_tokens = 0;               // L % kPageSize (0..63), stored privately inside Payload
  std::size_t private_vram_bytes = 0;
  bool is_root = false;
  Payload payload{};
};

template <typename Payload>
class BatchPrefixCache {
 public:
  using Entry = BatchPrefixEntry<Payload>;

  void set_vram_budget(std::size_t budget_bytes) noexcept {
    stats_.vram_budget_bytes = budget_bytes;
  }

  std::size_t vram_budget_bytes() const noexcept { return stats_.vram_budget_bytes; }
  std::size_t size() const noexcept { return entries_.size(); }
  const PrefixCacheStats& stats() const noexcept { return stats_; }

  void record_d2d_restore(std::size_t bytes) noexcept { stats_.d2d_restore_bytes += bytes; }
  void record_d2d_park(std::size_t bytes) noexcept { stats_.d2d_park_bytes += bytes; }

  // Non-mutating longest-prefix lookup for prompt-aware admission checks (does not touch LRU or stats).
  const Entry* find_longest_prefix_const(const StateIdentity& identity,
                                         std::span<const std::int32_t> prompt_ids) const {
    if (prompt_ids.empty() || entries_.empty()) return nullptr;
    const std::uint64_t id_hash = hash_state_identity(identity);
    const Entry* best = nullptr;
    std::size_t best_len = 0;
    for (const Entry& e : entries_) {
      if (e.identity_hash != id_hash) continue;
      const std::size_t len = e.tokens.size();
      if (len == 0 || len > prompt_ids.size() || len <= best_len) continue;
      if (!(e.identity == identity)) continue;
      if (std::equal(e.tokens.begin(), e.tokens.end(), prompt_ids.begin())) {
        best = &e;
        best_len = len;
        if (best_len == prompt_ids.size()) break;
      }
    }
    return best;
  }

  // Counts physical KV pages with active_refs == 0 that can be freed by evicting cached entries other than
  // `protected_entry_id` (0 = none). Used by prompt-aware admission checks so shared committed pages of the
  // matched hit are never counted as evictable while non-overlapping cached pages are.
  int reclaimable_pages_excluding(const PagedKvPagePool& pool, std::uint64_t protected_entry_id = 0) const {
    const int base = pool.evictable_count();
    if (protected_entry_id == 0 || base == 0) return base;
    int held_by_protected = 0;
    for (const Entry& e : entries_) {
      if (e.id != protected_entry_id) continue;
      for (int p : e.committed_pages) {
        if (pool.active_refs(p) == 0 && pool.cache_refs(p) > 0) ++held_by_protected;
      }
      break;
    }
    return base > held_by_protected ? (base - held_by_protected) : 0;
  }

  // Finds the longest exact prefix entry matching both StateIdentity::operator== and exact token IDs.
  // Moves the matched entry to MRU position on hit. Returns nullptr on miss.
  const Entry* find_longest_prefix(const StateIdentity& identity, std::span<const std::int32_t> prompt_ids) {
    if (prompt_ids.empty() || entries_.empty()) return nullptr;
    const std::uint64_t id_hash = hash_state_identity(identity);

    auto best = entries_.end();
    std::size_t best_len = 0;
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->identity_hash != id_hash) continue;
      const std::size_t len = it->tokens.size();
      if (len == 0 || len > prompt_ids.size() || len <= best_len) continue;
      if (!(it->identity == identity)) continue;
      if (std::equal(it->tokens.begin(), it->tokens.end(), prompt_ids.begin())) {
        best = it;
        best_len = len;
        if (best_len == prompt_ids.size()) break;  // cannot beat exact full-prompt match
      }
    }
    if (best == entries_.end()) return nullptr;
    // Promote to MRU (front of list).
    if (best != entries_.begin()) entries_.splice(entries_.begin(), entries_, best);
    return &(*best);
  }

  // Returns true iff an exact (identity, tokens) entry already exists (and promotes it to MRU).
  bool touch_exact(const StateIdentity& identity, std::span<const std::int32_t> tokens) {
    const std::uint64_t id_hash = hash_state_identity(identity);
    const std::uint64_t tok_hash = hash_token_prefix(tokens);
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->identity_hash != id_hash || it->prefix_hash != tok_hash) continue;
      if (it->tokens.size() != tokens.size()) continue;
      if (!(it->identity == identity)) continue;
      if (std::equal(it->tokens.begin(), it->tokens.end(), tokens.begin())) {
        if (it != entries_.begin()) entries_.splice(entries_.begin(), entries_, it);
        return true;
      }
    }
    return false;
  }

  // Evicts LRU entries until pool.free_count() >= needed_free_pages (or no evictable entry remains).
  // Skips `protected_entry_id` so the entry currently being restored is never evicted out from under itself.
  // Returns number of evicted cache entries.
  int ensure_free_pages(PagedKvPagePool& pool, int needed_free_pages, std::uint64_t protected_entry_id = 0) {
    int evicted = 0;
    // First pass: prefer evicting non-root LRU entries; second pass: evict root entries too if still needed.
    for (int pass = 0; pass < 2 && pool.free_count() < needed_free_pages; ++pass) {
      const bool allow_root = (pass == 1);
      auto it = entries_.end();
      while (it != entries_.begin() && pool.free_count() < needed_free_pages) {
        --it;
        if (it->id == protected_entry_id) continue;
        if (!allow_root && it->is_root) continue;
        auto victim = it;
        evict_entry(pool, victim);
        ++evicted;
        it = entries_.end();  // restart from LRU tail after erase
      }
    }
    return evicted;
  }

  // Evicts LRU entries until private VRAM usage + needed_bytes <= vram_budget_bytes (or empty).
  int ensure_private_vram(PagedKvPagePool& pool, std::size_t needed_bytes, std::uint64_t protected_entry_id = 0) {
    int evicted = 0;
    for (int pass = 0; pass < 2 && stats_.vram_used_bytes + needed_bytes > stats_.vram_budget_bytes; ++pass) {
      const bool allow_root = (pass == 1);
      auto it = entries_.end();
      while (it != entries_.begin() && stats_.vram_used_bytes + needed_bytes > stats_.vram_budget_bytes) {
        --it;
        if (it->id == protected_entry_id) continue;
        if (!allow_root && it->is_root) continue;
        evict_entry(pool, it);
        ++evicted;
        it = entries_.end();
      }
    }
    return evicted;
  }

  // Publishes a new entry after stream sync. Caller provides committed page ids (currently owned by an active slot);
  // this function pins them in `pool`. Returns {published, evictions}.
  std::pair<bool, int> publish(PagedKvPagePool& pool, Entry entry) {
    if (entry.tokens.empty() || stats_.vram_budget_bytes == 0 || entry.private_vram_bytes > stats_.vram_budget_bytes) {
      return {false, 0};
    }
    if (touch_exact(entry.identity, entry.tokens)) {
      return {false, 0};
    }
    const int evicted = ensure_private_vram(pool, entry.private_vram_bytes);
    if (stats_.vram_used_bytes + entry.private_vram_bytes > stats_.vram_budget_bytes) {
      return {false, evicted};
    }
    // Transactional publish: allocate list node first (only throwing step), then commit refs/counters + noexcept splice.
    std::list<Entry> node;
    node.push_front(std::move(entry));
    Entry& inserted = node.front();
    inserted.id = ++next_id_;
    inserted.identity_hash = hash_state_identity(inserted.identity);
    inserted.prefix_hash = hash_token_prefix(inserted.tokens);
    for (int p : inserted.committed_pages) pool.pin_for_cache(p);

    stats_.vram_used_bytes += inserted.private_vram_bytes;
    stats_.vram_peak_bytes = std::max(stats_.vram_peak_bytes, stats_.vram_used_bytes);
    ++stats_.vram_entries;
    if (inserted.is_root) ++stats_.root_entries;
    entries_.splice(entries_.begin(), node);
    return {true, evicted};
  }

  // Releases all cached entries (e.g. on VRAM allocation pressure or shutdown).
  int clear(PagedKvPagePool& pool, bool pressure_release = false) {
    int evicted = 0;
    while (!entries_.empty()) {
      evict_entry(pool, std::prev(entries_.end()));
      ++evicted;
    }
    if (pressure_release && evicted > 0) ++stats_.vram_pressure_releases;
    return evicted;
  }

 private:
  void evict_entry(PagedKvPagePool& pool, typename std::list<Entry>::iterator it) {
    for (int p : it->committed_pages) pool.unpin_from_cache(p);
    stats_.vram_used_bytes =
        (stats_.vram_used_bytes >= it->private_vram_bytes) ? (stats_.vram_used_bytes - it->private_vram_bytes) : 0;
    if (stats_.vram_entries > 0) --stats_.vram_entries;
    if (it->is_root && stats_.root_entries > 0) --stats_.root_entries;
    ++stats_.vram_evictions;
    entries_.erase(it);
  }

  std::uint64_t next_id_ = 0;
  std::list<Entry> entries_;  // MRU at front, LRU at back
  PrefixCacheStats stats_{};
};

}  // namespace ingine::model

// Generic 3-tier VRAM / Pinned HostRegion / NVMe mmap paging primitives.
// Shared foundation for both dynamic MoE expert offloading (ExpertCache) and paged long-context KV blocks.
#pragma once

#include "core/arena.h"
#include "core/host_memory.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <unordered_map>
#include <vector>

namespace ingine {

// Read-only mmap backing for a file on NVMe/disk with periodic MADV_DONTNEED page-cache control
// so cold reads do not bloat process RSS beyond the host RAM budget.
class MmapFile {
 public:
  MmapFile() = default;
  explicit MmapFile(const std::filesystem::path& path);
  ~MmapFile();

  MmapFile(const MmapFile&) = delete;
  MmapFile& operator=(const MmapFile&) = delete;
  MmapFile(MmapFile&& other) noexcept;
  MmapFile& operator=(MmapFile&& other) noexcept;

  [[nodiscard]] const std::byte* data() const noexcept { return static_cast<const std::byte*>(addr_); }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] bool valid() const noexcept { return addr_ != nullptr; }

  // Copies `bytes` starting at `offset` into `dst` and tracks cold bytes for page-cache eviction.
  void read_slice(std::uint64_t offset, std::size_t bytes, void* dst) const;
  // Advises the kernel to drop clean page-cache pages for [offset, offset + bytes).
  void advise_dontneed(std::uint64_t offset, std::size_t bytes) const noexcept;
  void advise_dontneed_all() const noexcept;

 private:
  void release() noexcept;

  void* addr_ = nullptr;
  std::size_t size_ = 0;
  int fd_ = -1;
};

// Writable NVMe/disk swap file for Tier-2 cold page spill (e.g. 1M paged KV blocks).
// Uses pread/pwrite + ftruncate + POSIX_FADV_DONTNEED (no mmap RSS bloat) and automatically unlinks
// its backing file on open (default, crash-safe) or on destruction.
class SwapFile {
 public:
  SwapFile() = default;
  // If `path` is empty, creates a unique temp file in the system temp directory.
  // When `unlink_immediately` is true, the directory entry is unlinked right after open() so the OS
  // reclaims disk space even if the process terminates abruptly.
  explicit SwapFile(const std::filesystem::path& path, std::size_t initial_bytes = 0,
                    bool unlink_immediately = true);
  ~SwapFile();

  SwapFile(const SwapFile&) = delete;
  SwapFile& operator=(const SwapFile&) = delete;
  SwapFile(SwapFile&& other) noexcept;
  SwapFile& operator=(SwapFile&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept { return fd_ >= 0; }
  [[nodiscard]] int fd() const noexcept { return fd_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

  void resize(std::size_t bytes);
  void write_slice(std::uint64_t offset, std::size_t bytes, const void* src);
  void read_slice(std::uint64_t offset, std::size_t bytes, void* dst) const;
  void advise_dontneed(std::uint64_t offset, std::size_t bytes) const noexcept;
  void advise_dontneed_all() const noexcept;

 private:
  void release() noexcept;

  int fd_ = -1;
  std::size_t size_ = 0;
  std::filesystem::path path_;
  bool unlink_on_close_ = false;
  mutable std::uint64_t read_since_advise_ = 0;
};

// Fixed-size ring of pinned host bounce buffers for streaming NVMe mmap reads into async PCIe DMA.
// Each slot tracks its own completion event so ring wrap-around never overwrites an in-flight DMA.
class PinnedBounceRing {
 public:
  PinnedBounceRing() = default;
  PinnedBounceRing(int num_slots, std::size_t slot_bytes);
  ~PinnedBounceRing();

  PinnedBounceRing(const PinnedBounceRing&) = delete;
  PinnedBounceRing& operator=(const PinnedBounceRing&) = delete;
  PinnedBounceRing(PinnedBounceRing&& other) noexcept;
  PinnedBounceRing& operator=(PinnedBounceRing&& other) noexcept;

  [[nodiscard]] bool valid() const noexcept { return num_slots_ > 0; }
  [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }
  [[nodiscard]] int num_slots() const noexcept { return num_slots_; }

  struct Slot {
    int index = -1;
    std::uint8_t* ptr = nullptr;
  };

  // Acquires the next ring slot, synchronizing only if that slot's previous DMA is still in flight.
  Slot acquire();
  // Records the slot's DMA completion event on `copy_stream` after enqueueing cudaMemcpyAsync.
  void mark_submitted(int index, cudaStream_t copy_stream);

 private:
  void release() noexcept;

  MappedHostBuffer buf_;
  std::vector<cudaEvent_t> done_;
  std::vector<bool> inflight_;
  std::size_t slot_bytes_ = 0;
  int num_slots_ = 0;
  int cursor_ = 0;
};

// Fixed-capacity VRAM LRU slab of equal-sized slots keyed by an opaque 64-bit object ID
// (e.g. (layer << 8) | expert for MoE, or (visit << 20) | page_id for paged KV).
class VramLruSlab {
 public:
  static constexpr std::uint64_t kEmptyKey = std::numeric_limits<std::uint64_t>::max();

  struct AcquireResult {
    int slot = -1;
    std::uint8_t* vram_ptr = nullptr;
    bool hit = false;
    bool evicted = false;
    std::uint64_t evicted_key = kEmptyKey;
  };

  VramLruSlab() = default;
  VramLruSlab(int num_slots, std::size_t slot_bytes);

  [[nodiscard]] int capacity() const noexcept { return int(slots_.size()); }
  [[nodiscard]] std::size_t slot_bytes() const noexcept { return slot_bytes_; }
  [[nodiscard]] std::size_t total_bytes() const noexcept { return storage_.bytes; }

  // Look up `key` without mutating residency; returns slot index or -1.
  [[nodiscard]] int find(std::uint64_t key) const noexcept;
  [[nodiscard]] std::uint8_t* slot_ptr(int slot) const noexcept { return slots_[slot].vram_ptr; }

  // Pin an already-resident key for the current generation `pin_gen` so it cannot be evicted
  // by other acquires in the same step/visit. When `priority` is true, boosts LRU recency by
  // slab capacity so high-reuse (e.g. 3x looped) entries survive single-pass layer sweeps.
  // Returns slot index on hit, -1 on miss.
  int touch_and_pin(std::uint64_t key, std::uint64_t pin_gen, bool priority = false) noexcept;

  // Acquire a VRAM slot for `key`, pinning it at `pin_gen`. Hits return hit=true without copying;
  // misses evict the least-recently-used slot whose pin_gen != current `pin_gen`.
  AcquireResult acquire(std::uint64_t key, std::uint64_t pin_gen, bool priority = false);

  void clear() noexcept;

 private:
  struct Slot {
    std::uint64_t key = kEmptyKey;
    std::uint64_t last_use = 0;
    std::uint64_t pin_gen = 0;
    std::uint8_t* vram_ptr = nullptr;
  };

  std::size_t slot_bytes_ = 0;
  DeviceBuffer storage_;
  std::vector<Slot> slots_;
  std::unordered_map<std::uint64_t, int> index_;
  std::uint64_t tick_ = 0;
};

// Dedicated non-blocking transfer stream + completion event for overlapping PCIe H2D/D2H copies
// with compute streams.
class TieredDmaStream {
 public:
  TieredDmaStream();
  ~TieredDmaStream();

  TieredDmaStream(const TieredDmaStream&) = delete;
  TieredDmaStream& operator=(const TieredDmaStream&) = delete;
  TieredDmaStream(TieredDmaStream&& other) noexcept;
  TieredDmaStream& operator=(TieredDmaStream&& other) noexcept;

  [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

  void copy_h2d_async(void* dst_device, const void* src_pinned_host, std::size_t bytes);
  void copy_d2h_async(void* dst_pinned_host, const void* src_device, std::size_t bytes);
  void record_and_wait_on(cudaStream_t compute_stream);
  void synchronize() const;

 private:
  void release() noexcept;

  cudaStream_t stream_ = nullptr;
  cudaEvent_t done_ = nullptr;
};

// Current process resident set size (VmRSS) from /proc/self/status in bytes (0 if unreadable).
std::uint64_t process_rss_bytes();

}  // namespace ingine

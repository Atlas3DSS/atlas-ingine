// Three-tier paged Q4_0 full-attention KV cache for 1M-token context on <= 12 GB and <= 6 GB GPUs.
//
// Built on core/paged_tier.h (VramLruSlab, TieredDmaStream, PinnedBounceRing, SwapFile) + chunked
// CUDA-registered HostRegion slabs (lazy 504 MiB slabs so short contexts never pin 9 GiB upfront):
//   - Tier 0 (HOT VRAM):   VramLruSlab of 4096-token Q4_0 pages (4.50 MiB/page: 2.25 MiB K + 2.25 MiB V)
//   - Tier 1 (WARM RAM):   Lazy pinned UVA HostRegion slabs (direct GPU UVA read during decode, 1x-PCIe
//                          double-buffered VRAM staging with FP32 LSE online merge during prefill)
//   - Tier 2 (COLD NVMe):  Crash-safe writable SwapFile (pread/pwrite + POSIX_FADV_DONTNEED) when active
//                          KV exceeds --kv-ram-mb.
#pragma once

#include "core/arena.h"
#include "core/host_memory.h"
#include "core/paged_tier.h"
#include "kernels/prefill/prefill.h"
#include "model/model_data.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ingine::model {

struct PagedKvConfig {
  int max_ctx = 1048576;
  int vram_mb = 0;               // VRAM budget for hot KV pages + staging (<= 0 disables paging)
  int ram_mb = 12288;            // pinned UVA host RAM cap in MiB before spilling to Tier-2 NVMe SwapFile
  int prefill_chunk = 512;       // max T for FP32 online-softmax partial buffers
  int num_full_slots = NUM_FULL_VISITS;  // 8 (per_pass) or 4 (last_full)
  std::filesystem::path swap_dir;  // optional directory for SwapFile (empty -> system temp dir)
};

struct PagedKvStats {
  std::uint64_t vram_pages = 0;
  std::uint64_t ram_pages = 0;
  std::uint64_t swap_pages = 0;
  std::uint64_t evict_to_ram = 0;
  std::uint64_t evict_to_swap = 0;
  std::uint64_t promote_from_ram = 0;
  std::uint64_t promote_from_swap = 0;
  std::uint64_t pcie_h2d_bytes = 0;
  std::uint64_t pcie_d2h_bytes = 0;
  std::uint64_t nvme_write_bytes = 0;
  std::uint64_t nvme_read_bytes = 0;
  std::size_t vram_bytes = 0;
  std::size_t ram_pinned_bytes = 0;
};

class PagedKvCache {
 public:
  static constexpr int kPageShift = 12;
  static constexpr int kPageTokens = 1 << kPageShift;                               // 4096 tokens/page
  static constexpr std::size_t kKPageBytes = std::size_t(kPageTokens) * KV_Q4_ROW_BYTES;  // 2.25 MiB
  static constexpr std::size_t kVPageBytes = kKPageBytes;                                 // 2.25 MiB
  static constexpr std::size_t kPageBytes = kKPageBytes + kVPageBytes;                    // 4.50 MiB

  explicit PagedKvCache(const PagedKvConfig& cfg);
  ~PagedKvCache();

  PagedKvCache(const PagedKvCache&) = delete;
  PagedKvCache& operator=(const PagedKvCache&) = delete;

  int max_ctx() const noexcept { return max_ctx_; }
  int max_pages() const noexcept { return max_pages_; }
  int num_full_slots() const noexcept { return num_full_slots_; }
  bool has_nvme_spill() const noexcept { return swap_page_count_ > 0; }
  // Use bulk-DMA double-buffered streaming decode whenever NVMe swap is active or >=32 pages (144 MiB)
  // are resident in Tier-1 host RAM (saturates PCIe H2D at ~20 GB/s instead of ~2 GB/s scattered UVA).
  bool needs_streamed_decode() const noexcept { return swap_page_count_ > 0 || ram_page_count_ >= 32; }
  std::size_t vram_bytes() const noexcept { return vram_bytes_; }

  // Fixed device pointer tables [max_pages] for full visit `fi` (0..NUM_FULL_VISITS-1).
  void* const* k_table(int fi) const noexcept { return static_cast<void* const*>(dev_k_ptrs_[fi].p); }
  void* const* v_table(int fi) const noexcept { return static_cast<void* const*>(dev_v_ptrs_[fi].p); }

  // Ensures pages covering [pos0, pos0 + count) for full visit `fi` are allocated and pinned in Tier-0 VRAM,
  // flushing the device page table onto `stream`.
  void ensure_write_pages(int fi, int pos0, int count, cudaStream_t stream);

  // Ensures write pages for ALL 8 full visits covering decode positions [pos0, pos0 + m) are resident in VRAM.
  // Fast O(1) no-op on 4095/4096 steps when the tail page is already resident in Tier-0 VRAM.
  void ensure_decode_pages(int pos0, int m, cudaStream_t stream);

  // Runs prefill attention for full visit `fi` over keys [0, base_at.L):
  //   - If all pages [0, num_pages) are in Tier-0 VRAM, launches a single paged attention kernel.
  //   - Otherwise streams batches of pages into double-buffered VRAM staging (1x PCIe traffic, zero
  //     amplification, supports Tier-1 RAM and Tier-2 NVMe SwapFile) using FP32 LSE online merge.
  void prefill_attention(int fi, kernels::prefill::AttnParams base_at, cudaStream_t stream);

  // Tier-2 NVMe decode fallback (called ONLY when has_nvme_spill() == true): executes full-layer decode
  // attention for M tokens (M <= 8) at `pos0` by running attn_prep + rel_logits + prefill_attention + conv update.
  void decode_attention_with_swap(int fi, const Layer& layer, int visit, const float* proj_f32, int M, int pos0,
                                  float* conv_k, float* conv_v, ActT* out, cudaStream_t stream);

  // Resets all logical pages (preserves allocated VRAM/RAM pools for reuse).
  void reset();

  // Streaming disk checkpoint helpers (O(1) host RAM):
  int vram_slot_count() const noexcept { return vram_slab_.capacity(); }
  void export_page_to_host(int fi, int page, std::uint8_t* dst_k, std::uint8_t* dst_v, cudaStream_t compute_stream);
  void import_page_from_host(int fi, int page, const std::uint8_t* src_k, const std::uint8_t* src_v, bool prefer_ram,
                             cudaStream_t compute_stream);
  void flush_all_tables(cudaStream_t stream);

  PagedKvStats stats() const noexcept;
  std::string summary() const;

 private:
  enum class Loc : std::uint8_t { Unalloc = 0, Vram = 1, Ram = 2, Swap = 3 };

  struct PageMeta {
    Loc loc = Loc::Unalloc;
    int slot = -1;  // VramLruSlab slot if Vram; flat RAM slot index if Ram
    std::uint64_t ram_tick = 0;
  };

  struct RamSlot {
    int fi = -1;
    int page = -1;
    std::uint64_t last_use = 0;
  };

  static constexpr std::uint64_t make_key(int fi, int page) noexcept {
    return (static_cast<std::uint64_t>(fi) << 20) | static_cast<std::uint64_t>(page);
  }
  static constexpr int key_fi(std::uint64_t k) noexcept { return static_cast<int>(k >> 20); }
  static constexpr int key_page(std::uint64_t k) noexcept { return static_cast<int>(k & 0xfffffu); }

  std::uint64_t swap_offset(int fi, int page) const noexcept {
    return (static_cast<std::uint64_t>(fi) * std::uint64_t(max_pages_) + std::uint64_t(page)) * kPageBytes;
  }

  std::uint8_t* ram_host_ptr(int ram_slot) const noexcept;
  std::uint8_t* ram_dev_ptr(int ram_slot) const noexcept;
  int alloc_ram_slot(int fi, int page);
  void free_ram_slot(int ram_slot);
  void ensure_swap_open();
  void evict_vram_page(int fi, int page, std::uint8_t* vram_ptr, cudaStream_t compute_stream);
  void set_page_ptrs(int fi, int page, void* k_ptr, void* v_ptr);
  void flush_table(int fi, cudaStream_t stream);
  void copy_page_to_staging(int fi, int page, std::uint8_t* dst_k, std::uint8_t* dst_v);

  int max_ctx_ = 0;
  int max_pages_ = 0;
  int num_full_slots_ = NUM_FULL_VISITS;
  int batch_pages_ = 4;
  int max_ram_slots_ = 0;
  int ram_chunk_slots_ = 112;  // 112 * 4.5 MiB = 504 MiB per lazy HostRegion slab
  std::filesystem::path swap_dir_;
  std::size_t vram_bytes_ = 0;

  // Tier 0: HOT VRAM page pool + double-buffered prefill staging + FP32 LSE partial accumulators
  VramLruSlab vram_slab_;
  std::array<DeviceBuffer, 2> stage_k_, stage_v_;
  std::array<cudaEvent_t, 2> stage_compute_done_{};
  std::array<cudaEvent_t, 2> stage_dma_done_{};
  DeviceBuffer partial_o_, partial_m_, partial_l_;

  // Small scratch buffers for streamed Q4 decode (M <= 8, ~3 MiB total including 16-way split-K workspace)
  static constexpr int kDecodeSplits = 16;
  DeviceBuffer dec_proj16_, dec_q_, dec_rel16_, dec_rel_, decode_ws_;
  MappedHostBuffer dec_pos_host_;

  // Per-visit fixed device page tables [max_pages]
  std::array<DeviceBuffer, NUM_FULL_VISITS> dev_k_ptrs_, dev_v_ptrs_;
  std::array<std::vector<void*>, NUM_FULL_VISITS> host_k_ptrs_, host_v_ptrs_;
  std::array<bool, NUM_FULL_VISITS> table_dirty_{};

  // Tier 1: Lazy chunked pinned UVA HostRegion slabs
  std::vector<HostRegion> ram_chunks_;
  std::vector<RamSlot> ram_slots_;
  std::vector<int> free_ram_slots_;
  std::uint64_t ram_tick_ = 0;

  // Tier 2: Writable crash-safe NVMe SwapFile + pinned bounce ring
  SwapFile swap_;
  PinnedBounceRing bounce_;
  std::uint64_t ram_page_count_ = 0;
  std::uint64_t swap_page_count_ = 0;

  // Page metadata [NUM_FULL_VISITS][max_pages]
  std::array<std::vector<PageMeta>, NUM_FULL_VISITS> meta_;

  TieredDmaStream dma_;
  cudaEvent_t compute_ready_ = nullptr;
  std::uint64_t pin_gen_ = 1;
  PagedKvStats stats_{};
};

}  // namespace ingine::model

// Three-tier dynamic MoE expert cache and homology-guided placement for low-VRAM execution
// (atlas-6g-homology: <= 6 GB VRAM, <= 16 GB system RAM, optional read-only NVMe mmap).
//
// Built atop the generic core/paged_tier.h primitives (MmapFile, PinnedBounceRing, VramLruSlab,
// TieredDmaStream) + HostRegion so the same tiering layer can be reused for paged 1M KV blocks.
#pragma once

#include "core/arena.h"
#include "core/host_memory.h"
#include "core/paged_tier.h"
#include "core/safetensors.h"
#include "ingine/generator.h"
#include "kernels/moe/moe.h"
#include "model/expert_swap_queue.h"
#include "model/model_data.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ingine::model {

enum class OffloadMode {
  DmaLru,  // 3-tier: HOT VRAM + dynamic VRAM LRU + async PCIe DMA from WARM RAM / COLD NVMe mmap
  Uva,     // 2-tier: HOT VRAM + zero-copy UVA from WARM pinned RAM (100% CUDA-graph capturable)
};

struct OffloadConfig {
  int vram_mb = 0;                          // total target VRAM cap in MiB (0 = disabled)
  int ram_mb = 12288;                       // pinned host RAM cap in MiB for WARM experts
  std::string mode = "auto";                // "auto" | "dma-lru" | "uva"
  std::filesystem::path ranking;            // optional JSON per-layer [24][128] homology/frequency scores
  std::filesystem::path dump_router_stats;  // optional JSON output path for observed [24][128] hits
  int max_ctx = 32768;
  int prefill_chunk = 512;
  bool kv_fp8 = true;
  bool kv_q4 = false;
  bool kv_last_full = false;
  bool spec = false;
  int kv_vram_mb = 0;  // when > 0, full KV cache is paged and capped to kv_vram_mb MiB of VRAM
  int loop_passes = PASSES;
  // Opt-in small-card knobs (#888, all default off / 0):
  bool loop_bf16 = false;
  AttnRequantMode attn_requant = AttnRequantMode::Off;
  int expert_swaps = 0;  // bounded async HOT-promotion queue capacity (0 = disabled/default, >0 = opt-in)
};

struct OffloadStats {
  std::uint64_t hot_hits = 0;
  std::uint64_t lru_hits = 0;
  std::uint64_t warm_fetches = 0;
  std::uint64_t cold_fetches = 0;
  std::uint64_t pcie_h2d_bytes = 0;
  std::uint64_t nvme_read_bytes = 0;
  int hot_experts = 0;
  int warm_experts = 0;
  int cold_experts = 0;
  int lru_slots_total = 0;
  int expert_swaps_cap = 0;
  std::size_t hot_vram_bytes = 0;
  std::size_t lru_vram_bytes = 0;
  std::size_t warm_ram_bytes = 0;
  std::size_t cold_nvme_bytes = 0;
  std::uint64_t hot_evictions = 0;
  std::uint64_t async_swaps_queued = 0;
  std::uint64_t async_swaps_completed = 0;
  std::uint64_t async_swaps_dropped_bound = 0;  // dropped promotion requests when bounded queue is full
  std::uint64_t experts_dropped = 0;            // invariant == 0 (strictly NO silent tail-skip)
};

class ExpertCache {
 public:
  static std::shared_ptr<ExpertCache> create(
      const std::filesystem::path& model_dir,
      const std::array<ExpertFormat, LAYERS>& w13_formats,
      const std::array<ExpertFormat, LAYERS>& w2_formats,
      const std::array<std::vector<float>, LAYERS>& router_biases,
      std::size_t expert_vram_budget,
      std::size_t warm_ram_budget,
      const OffloadConfig& cfg,
      cudaStream_t stream);

  ~ExpertCache();
  ExpertCache(const ExpertCache&) = delete;
  ExpertCache& operator=(const ExpertCache&) = delete;

  OffloadMode mode() const noexcept { return mode_; }
  bool needs_host_dispatch() const noexcept { return mode_ == OffloadMode::DmaLru; }

  kernels::moe::ExpertPtrs layer_ptrs(int l) const noexcept;

  void prepare_decode(int l, const int* d_idx, int m, cudaStream_t compute_stream);

  // Run MoE decode (up + down) for m tokens (1..8). Executes in a single batched launch [0, m) when all
  // unique non-HOT experts fit in the layer's LRU slab; otherwise splits [0, m) into maximal contiguous token
  // slices [tok_offset, tok_offset + tok_count) that fit so DSpark verify (M=2..8) stays bit-exact and
  // deadlock-free under tight --atlas-6g VRAM budgets.
  void run_decode_layer(int l, const int* d_idx, int m, cudaStream_t compute_stream,
                        const std::function<void(int tok_offset, int tok_count)>& launch_moe_slice);

  void run_prefill_layer(int l, const int* d_counts, cudaStream_t compute_stream,
                         const std::function<void(const std::uint8_t* d_active_mask)>& launch_moe_pair);

  const OffloadStats& stats() const noexcept { return stats_; }
  void reset_runtime_stats() noexcept;
  void dump_router_stats_if_configured() const;
  std::string summary() const;

 private:
  struct ExpertLayout {
    std::size_t w13_bytes = 0;
    std::size_t w13_scale_bytes = 0;  // block scales or EXL3 w13_rin
    std::size_t w13_rout_bytes = 0;   // EXL3 w13_rout (0 for non-EXL3)
    std::size_t w2_bytes = 0;
    std::size_t w2_scale_bytes = 0;   // block scales or EXL3 w2_rin
    std::size_t w2_rout_bytes = 0;    // EXL3 w2_rout (0 for non-EXL3)
    std::size_t off_w13 = 0;
    std::size_t off_w13_s = 0;
    std::size_t off_w13_rout = 0;
    std::size_t off_w2 = 0;
    std::size_t off_w2_s = 0;
    std::size_t off_w2_rout = 0;
    std::size_t record_bytes = 0;
  };

  enum class Tier : std::uint8_t { HotVram = 0, WarmRam = 1, ColdNvme = 2 };

  struct FileSlice {
    int file_idx = -1;
    std::uint64_t offset = 0;
    std::size_t bytes = 0;
  };

  struct ExpertMeta {
    Tier tier = Tier::ColdNvme;
    bool has_warm_backing = false;
    bool has_cold_backing = false;
    bool evictable_hot = false;
    int hot_slot = -1;       // index into hot_slots_[layer_slab_[l]] when tier == Tier::HotVram
    int inflight_slot = -1;  // index into inflight_[] when async HOT promotion is in flight
    float score = 0.f;
    std::uint8_t* hot_ptr = nullptr;
    std::size_t warm_offset = 0;
    FileSlice cold_w13, cold_w13_s, cold_w13_rout, cold_w2, cold_w2_s, cold_w2_rout;
  };

  struct InflightSwap {
    bool active = false;
    int bucket = -1;
    int hot_slot = -1;
    int new_layer = -1;
    int new_expert = -1;
    std::uint8_t* dst_vram = nullptr;
    cudaEvent_t done = nullptr;
  };
  static constexpr int kMaxInflightSwaps = 2;

  struct LayerDeviceTables {
    DeviceBuffer dev;
    std::uint8_t** d_w13 = nullptr;
    std::uint8_t** d_w13_s = nullptr;
    std::uint8_t** d_w13_rout = nullptr;
    std::uint8_t** d_w2 = nullptr;
    std::uint8_t** d_w2_s = nullptr;
    std::uint8_t** d_w2_rout = nullptr;
    MappedHostBuffer host_mirror;
    std::uint8_t** h_w13 = nullptr;
    std::uint8_t** h_w13_s = nullptr;
    std::uint8_t** h_w13_rout = nullptr;
    std::uint8_t** h_w2 = nullptr;
    std::uint8_t** h_w2_s = nullptr;
    std::uint8_t** h_w2_rout = nullptr;
    bool dirty = false;
  };

  ExpertCache() = default;

  static constexpr std::uint64_t make_key(int l, int e) noexcept {
    return (static_cast<std::uint64_t>(l) << 8) | static_cast<std::uint64_t>(e);
  }
  static constexpr int key_layer(std::uint64_t k) noexcept { return static_cast<int>(k >> 8); }
  static constexpr int key_expert(std::uint64_t k) noexcept { return static_cast<int>(k & 0xffu); }

  static ExpertLayout layout_for(ExpertFormat w13_fmt, ExpertFormat w2_fmt);
  void set_host_ptrs(int l, int e, std::uint8_t* base);
  void clear_host_ptrs(int l, int e);
  bool stage_expert(int l, int e, std::uint64_t pin_gen);
  void flush_layer_table(int l);
  void read_cold_to_host(int l, int e, std::uint8_t* dst);
  void complete_inflight_swap(int inf_idx);
  void poll_inflight_swaps();
  void finalize_all_inflight_swaps(cudaStream_t compute_stream);
  void drain_swap_queue_async(int current_layer, std::uint64_t cur_pin_gen, cudaStream_t compute_stream);

  OffloadMode mode_ = OffloadMode::DmaLru;
  std::filesystem::path dump_router_stats_;
  std::array<ExpertFormat, LAYERS> w13_formats_{};
  std::array<ExpertFormat, LAYERS> w2_formats_{};
  std::array<ExpertLayout, LAYERS> layouts_{};
  std::array<int, LAYERS> layer_slab_{};
  std::array<std::array<ExpertMeta, EXPERTS>, LAYERS> meta_{};
  std::array<LayerDeviceTables, LAYERS> tables_{};

  // Tier 0: permanent HOT VRAM pool + per-record-size dynamic VRAM LRU slabs
  DeviceBuffer hot_vram_;
  std::vector<std::vector<BoundedSwapQueue::HotSlotState>> hot_slots_;
  std::vector<std::vector<std::uint8_t*>> hot_slot_ptrs_;
  std::vector<VramLruSlab> slabs_;
  // Tier 1: WARM pinned host RAM (2 MiB hugepages, CUDA-registered)
  HostRegion warm_ram_;
  // Tier 2: COLD read-only NVMe mmap files + pinned bounce ring
  std::vector<MmapFile> mmap_files_;
  PinnedBounceRing cold_bounce_;
  std::uint64_t cold_since_madvise_ = 0;

  // Exchange & synchronization primitives
  TieredDmaStream dma_;
  cudaEvent_t idx_ready_ = nullptr;
  cudaEvent_t pass0_done_ = nullptr;
  cudaEvent_t victim_release_ev_ = nullptr;
  BoundedSwapQueue swap_queue_{0};
  std::array<InflightSwap, kMaxInflightSwaps> inflight_{};
  std::uint64_t swap_tick_ = 0;
  MappedHostBuffer host_idx_;                  // int32 [8 * TOP_K]
  MappedHostBuffer host_counts_;               // int32 [EXPERTS]
  std::array<MappedHostBuffer, 2> host_mask_;  // uint8 [EXPERTS] x 2 for overlapped HOT/LRU vs WARM passes
  std::array<DeviceBuffer, 2> dev_mask_;       // uint8 [EXPERTS] x 2

  std::uint64_t pin_gen_ = 0;
  OffloadStats stats_{};
  std::array<std::array<std::uint64_t, EXPERTS>, LAYERS> router_hits_{};
};

}  // namespace ingine::model

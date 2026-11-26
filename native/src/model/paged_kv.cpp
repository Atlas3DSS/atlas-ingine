#include "model/paged_kv.h"

#include "core/device.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace ingine::model {
namespace pk = kernels::prefill;

PagedKvCache::PagedKvCache(const PagedKvConfig& cfg)
    : max_ctx_(cfg.max_ctx),
      max_pages_((cfg.max_ctx + kPageTokens - 1) / kPageTokens),
      num_full_slots_(cfg.num_full_slots),
      swap_dir_(cfg.swap_dir) {
  if (max_ctx_ < 1 || max_pages_ < 1) throw std::invalid_argument("PagedKvCache: invalid max_ctx");
  if (cfg.vram_mb <= 0) throw std::invalid_argument("PagedKvCache: vram_mb must be positive");
  if (num_full_slots_ < 1 || num_full_slots_ > NUM_FULL_VISITS) {
    throw std::invalid_argument("PagedKvCache: invalid num_full_slots");
  }

  // Size double-buffered prefill staging (36 MiB in <=6 GB mode, 72 MiB in >=12 GB mode).
  batch_pages_ = cfg.vram_mb >= 1024 ? 8 : 4;
  const std::size_t stage_half_k = std::size_t(batch_pages_) * kKPageBytes;
  const std::size_t stage_half_v = std::size_t(batch_pages_) * kVPageBytes;
  for (int i = 0; i < 2; ++i) {
    stage_k_[i] = DeviceBuffer(stage_half_k);
    stage_v_[i] = DeviceBuffer(stage_half_v);
    CUDA_CHECK(cudaEventCreateWithFlags(&stage_compute_done_[i], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&stage_dma_done_[i], cudaEventDisableTiming));
  }
  CUDA_CHECK(cudaEventCreateWithFlags(&compute_ready_, cudaEventDisableTiming));

  // FP32 online-softmax partials for prefill (and Tier-2 NVMe decode fallback).
  const int max_t = std::max(cfg.prefill_chunk, 8);
  partial_o_ = DeviceBuffer(std::size_t(max_t) * HEADS * HEAD_DIM * sizeof(float));
  partial_m_ = DeviceBuffer(std::size_t(max_t) * HEADS * sizeof(float));
  partial_l_ = DeviceBuffer(std::size_t(max_t) * HEADS * sizeof(float));

  // Small scratch for streamed Q4 decode (M <= 8) + 16-way split-K workspace (~1.52 MiB).
  dec_proj16_ = DeviceBuffer(std::size_t(8) * PROJ_ROWS_FULL * sizeof(ActT));
  dec_q_ = DeviceBuffer(std::size_t(8) * HEADS * HEAD_DIM * sizeof(ActT));
  dec_rel16_ = DeviceBuffer(std::size_t(8) * HEADS * D_REL * sizeof(ActT));
  dec_rel_ = DeviceBuffer(std::size_t(8) * HEADS * REL_EXTENT * sizeof(ActT));
  decode_ws_ = DeviceBuffer(std::size_t(kDecodeSplits) * 8 * HEADS * (HEAD_DIM + 2) * sizeof(float));
  dec_pos_host_ = MappedHostBuffer(sizeof(std::int64_t));

  // Per-slot fixed device page tables.
  const std::size_t table_bytes = std::size_t(max_pages_) * sizeof(void*);
  for (int fi = 0; fi < num_full_slots_; ++fi) {
    dev_k_ptrs_[fi] = DeviceBuffer(table_bytes);
    dev_v_ptrs_[fi] = DeviceBuffer(table_bytes);
    dev_k_ptrs_[fi].fill(0);
    dev_v_ptrs_[fi].fill(0);
    host_k_ptrs_[fi].assign(std::size_t(max_pages_), nullptr);
    host_v_ptrs_[fi].assign(std::size_t(max_pages_), nullptr);
    meta_[fi].resize(std::size_t(max_pages_));
  }

  const std::size_t overhead_bytes =
      2 * (stage_half_k + stage_half_v) + partial_o_.bytes + partial_m_.bytes + partial_l_.bytes +
      dec_proj16_.bytes + dec_q_.bytes + dec_rel16_.bytes + dec_rel_.bytes + decode_ws_.bytes +
      2 * std::size_t(num_full_slots_) * table_bytes;
  const std::size_t budget_bytes = std::size_t(cfg.vram_mb) << 20;
  // Minimum 2 * num_full_slots_ VRAM slots so every active full slot can pin 2 write pages simultaneously.
  const std::size_t avail_for_slab = budget_bytes > overhead_bytes ? (budget_bytes - overhead_bytes) : 0;
  const int max_total_pages = num_full_slots_ * max_pages_;
  int num_vram_slots = std::max(2 * num_full_slots_, int(avail_for_slab / kPageBytes));
  num_vram_slots = std::min(num_vram_slots, max_total_pages);
  vram_slab_ = VramLruSlab(num_vram_slots, kPageBytes);
  vram_bytes_ = overhead_bytes + vram_slab_.total_bytes();

  // Tier-1 pinned UVA host RAM cap (lazy 504 MiB chunks; capped to total possible pages).
  const std::size_t ram_budget = cfg.ram_mb > 0 ? (std::size_t(cfg.ram_mb) << 20) : 0;
  max_ram_slots_ = std::min(max_total_pages, int(ram_budget / kPageBytes));
  ram_chunk_slots_ = std::max(1, std::min(112, std::max(16, max_ram_slots_)));
}

PagedKvCache::~PagedKvCache() {
  for (int i = 0; i < 2; ++i) {
    if (stage_compute_done_[i]) cudaEventDestroy(stage_compute_done_[i]);
    if (stage_dma_done_[i]) cudaEventDestroy(stage_dma_done_[i]);
  }
  if (compute_ready_) cudaEventDestroy(compute_ready_);
}

std::uint8_t* PagedKvCache::ram_host_ptr(int ram_slot) const noexcept {
  const int chunk = ram_slot / ram_chunk_slots_;
  const int off = ram_slot % ram_chunk_slots_;
  return reinterpret_cast<std::uint8_t*>(ram_chunks_[chunk].data()) + std::size_t(off) * kPageBytes;
}

std::uint8_t* PagedKvCache::ram_dev_ptr(int ram_slot) const noexcept {
  const int chunk = ram_slot / ram_chunk_slots_;
  const int off = ram_slot % ram_chunk_slots_;
  void* base = ram_chunks_[chunk].device_ptr();
  return base ? (static_cast<std::uint8_t*>(base) + std::size_t(off) * kPageBytes) : nullptr;
}

void PagedKvCache::ensure_swap_open() {
  if (swap_.valid()) return;
  const std::size_t total_swap_bytes = std::size_t(num_full_slots_) * std::size_t(max_pages_) * kPageBytes;
  std::filesystem::path p;
  if (!swap_dir_.empty()) {
    std::filesystem::create_directories(swap_dir_);
    p = swap_dir_ / "ingine_kv_swap.bin";
  }
  swap_ = SwapFile(p, total_swap_bytes, /*unlink_immediately=*/true);
  bounce_ = PinnedBounceRing(4, kPageBytes);
}

int PagedKvCache::alloc_ram_slot(int fi, int page) {
  if (max_ram_slots_ <= 0) return -1;
  if (!free_ram_slots_.empty()) {
    const int s = free_ram_slots_.back();
    free_ram_slots_.pop_back();
    ram_slots_[s] = RamSlot{fi, page, ++ram_tick_};
    return s;
  }
  if (int(ram_slots_.size()) < max_ram_slots_) {
    const int next_slots = std::min(ram_chunk_slots_, max_ram_slots_ - int(ram_slots_.size()));
    HostRegion reg(std::size_t(next_slots) * kPageBytes);
    if (!reg.register_with_cuda()) {
      throw std::runtime_error("PagedKvCache: cudaHostRegister failed for " +
                               std::to_string((std::size_t(next_slots) * kPageBytes) >> 20) + " MiB KV slab");
    }
    const int base = int(ram_slots_.size());
    ram_chunks_.push_back(std::move(reg));
    ram_slots_.resize(base + next_slots);
    for (int i = next_slots - 1; i >= 1; --i) free_ram_slots_.push_back(base + i);
    ram_slots_[base] = RamSlot{fi, page, ++ram_tick_};
    return base;
  }
  // Tier-1 RAM is full: evict least-recently-used RAM slot to Tier-2 NVMe SwapFile!
  int victim = -1;
  std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
  for (int i = 0; i < int(ram_slots_.size()); ++i) {
    if (ram_slots_[i].fi >= 0 && ram_slots_[i].last_use < best) {
      best = ram_slots_[i].last_use;
      victim = i;
    }
  }
  if (victim < 0) throw std::runtime_error("PagedKvCache: RAM LRU invariant broken");
  const int vfi = ram_slots_[victim].fi;
  const int vpage = ram_slots_[victim].page;
  ensure_swap_open();
  swap_.write_slice(swap_offset(vfi, vpage), kPageBytes, ram_host_ptr(victim));
  meta_[vfi][vpage].loc = Loc::Swap;
  meta_[vfi][vpage].slot = -1;
  set_page_ptrs(vfi, vpage, nullptr, nullptr);
  if (ram_page_count_ > 0) --ram_page_count_;
  ++swap_page_count_;
  ++stats_.evict_to_swap;
  stats_.nvme_write_bytes += kPageBytes;
  ram_slots_[victim] = RamSlot{fi, page, ++ram_tick_};
  return victim;
}

void PagedKvCache::free_ram_slot(int ram_slot) {
  if (ram_slot < 0) return;
  ram_slots_[ram_slot] = RamSlot{-1, -1, 0};
  free_ram_slots_.push_back(ram_slot);
}

void PagedKvCache::set_page_ptrs(int fi, int page, void* k_ptr, void* v_ptr) {
  if (host_k_ptrs_[fi][page] != k_ptr || host_v_ptrs_[fi][page] != v_ptr) {
    host_k_ptrs_[fi][page] = k_ptr;
    host_v_ptrs_[fi][page] = v_ptr;
    table_dirty_[fi] = true;
  }
}

void PagedKvCache::flush_table(int fi, cudaStream_t stream) {
  if (!table_dirty_[fi]) return;
  const std::size_t bytes = std::size_t(max_pages_) * sizeof(void*);
  CUDA_CHECK(cudaMemcpyAsync(dev_k_ptrs_[fi].p, host_k_ptrs_[fi].data(), bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(dev_v_ptrs_[fi].p, host_v_ptrs_[fi].data(), bytes, cudaMemcpyHostToDevice, stream));
  table_dirty_[fi] = false;
}

void PagedKvCache::evict_vram_page(int fi, int page, std::uint8_t* vram_ptr, cudaStream_t compute_stream) {
  // Ensure any kernels on compute_stream writing/reading this VRAM slot finish before D2H copy.
  CUDA_CHECK(cudaEventRecord(compute_ready_, compute_stream));
  CUDA_CHECK(cudaStreamWaitEvent(dma_.stream(), compute_ready_, 0));

  const int rslot = alloc_ram_slot(fi, page);
  if (rslot >= 0) {
    std::uint8_t* dst_host = ram_host_ptr(rslot);
    std::uint8_t* dst_dev = ram_dev_ptr(rslot);
    dma_.copy_d2h_async(dst_host, vram_ptr, kPageBytes);
    dma_.synchronize();  // must complete before caller reuses `vram_ptr`
    meta_[fi][page].loc = Loc::Ram;
    meta_[fi][page].slot = rslot;
    set_page_ptrs(fi, page, dst_dev, dst_dev + kKPageBytes);
    ++ram_page_count_;
    ++stats_.evict_to_ram;
    stats_.pcie_d2h_bytes += kPageBytes;
  } else {
    // --kv-ram-mb 0: spill straight to Tier-2 NVMe SwapFile via pinned bounce ring.
    ensure_swap_open();
    auto bslot = bounce_.acquire();
    dma_.copy_d2h_async(bslot.ptr, vram_ptr, kPageBytes);
    dma_.synchronize();
    swap_.write_slice(swap_offset(fi, page), kPageBytes, bslot.ptr);
    meta_[fi][page].loc = Loc::Swap;
    meta_[fi][page].slot = -1;
    set_page_ptrs(fi, page, nullptr, nullptr);
    ++swap_page_count_;
    ++stats_.evict_to_swap;
    stats_.pcie_d2h_bytes += kPageBytes;
    stats_.nvme_write_bytes += kPageBytes;
  }
}

void PagedKvCache::ensure_write_pages(int fi, int pos0, int count, cudaStream_t stream) {
  if (count <= 0) return;
  const int p_begin = pos0 >> kPageShift;
  const int p_end = (pos0 + count - 1) >> kPageShift;
  if (p_end >= max_pages_) {
    throw std::out_of_range("PagedKvCache: position exceeds max_ctx");
  }
  ++pin_gen_;
  for (int p = p_begin; p <= p_end; ++p) {
    PageMeta& pm = meta_[fi][p];
    if (pm.loc == Loc::Vram) {
      vram_slab_.touch_and_pin(make_key(fi, p), pin_gen_);
      continue;
    }
    const Loc prev_loc = pm.loc;
    const int prev_slot = pm.slot;
    auto acq = vram_slab_.acquire(make_key(fi, p), pin_gen_);
    if (acq.evicted) {
      const int efi = key_fi(acq.evicted_key);
      const int ep = key_page(acq.evicted_key);
      evict_vram_page(efi, ep, acq.vram_ptr, stream);
      // Evicted page may belong to a different visit; keep its table synced before next use.
      flush_table(efi, stream);
    }
    if (prev_loc == Loc::Ram) {
      dma_.copy_h2d_async(acq.vram_ptr, ram_host_ptr(prev_slot), kPageBytes);
      dma_.synchronize();
      free_ram_slot(prev_slot);
      if (ram_page_count_ > 0) --ram_page_count_;
      ++stats_.promote_from_ram;
      stats_.pcie_h2d_bytes += kPageBytes;
    } else if (prev_loc == Loc::Swap) {
      ensure_swap_open();
      auto bslot = bounce_.acquire();
      swap_.read_slice(swap_offset(fi, p), kPageBytes, bslot.ptr);
      dma_.copy_h2d_async(acq.vram_ptr, bslot.ptr, kPageBytes);
      bounce_.mark_submitted(bslot.index, dma_.stream());
      dma_.synchronize();
      if (swap_page_count_ > 0) --swap_page_count_;
      ++stats_.promote_from_swap;
      stats_.nvme_read_bytes += kPageBytes;
      stats_.pcie_h2d_bytes += kPageBytes;
    }
    pm.loc = Loc::Vram;
    pm.slot = acq.slot;
    set_page_ptrs(fi, p, acq.vram_ptr, acq.vram_ptr + kKPageBytes);
  }
  flush_table(fi, stream);
}

void PagedKvCache::ensure_decode_pages(int pos0, int m, cudaStream_t stream) {
  if (m <= 0) return;
  const int p_begin = pos0 >> kPageShift;
  const int p_end = (pos0 + m - 1) >> kPageShift;
  if (p_end >= max_pages_) return;  // attn_decode handles cap early-out cleanly

  // Fast O(1) check: if all active full slots already have their write page(s) in Tier-0 VRAM, zero work!
  bool all_hot = true;
  for (int fi = 0; fi < num_full_slots_ && all_hot; ++fi) {
    for (int p = p_begin; p <= p_end; ++p) {
      if (meta_[fi][p].loc != Loc::Vram || table_dirty_[fi]) {
        all_hot = false;
        break;
      }
    }
  }
  if (all_hot) return;

  ++pin_gen_;
  // Pin already-resident target pages across all active slots first so evictions never thrash another slot's tail page!
  for (int fi = 0; fi < num_full_slots_; ++fi) {
    for (int p = p_begin; p <= p_end; ++p) {
      if (meta_[fi][p].loc == Loc::Vram) vram_slab_.touch_and_pin(make_key(fi, p), pin_gen_);
    }
  }
  for (int fi = 0; fi < num_full_slots_; ++fi) {
    for (int p = p_begin; p <= p_end; ++p) {
      PageMeta& pm = meta_[fi][p];
      if (pm.loc == Loc::Vram) continue;
      const Loc prev_loc = pm.loc;
      const int prev_slot = pm.slot;
      auto acq = vram_slab_.acquire(make_key(fi, p), pin_gen_);
      if (acq.evicted) {
        const int efi = key_fi(acq.evicted_key);
        const int ep = key_page(acq.evicted_key);
        evict_vram_page(efi, ep, acq.vram_ptr, stream);
      }
      if (prev_loc == Loc::Ram) {
        dma_.copy_h2d_async(acq.vram_ptr, ram_host_ptr(prev_slot), kPageBytes);
        dma_.synchronize();
        free_ram_slot(prev_slot);
        if (ram_page_count_ > 0) --ram_page_count_;
        ++stats_.promote_from_ram;
        stats_.pcie_h2d_bytes += kPageBytes;
      } else if (prev_loc == Loc::Swap) {
        ensure_swap_open();
        auto bslot = bounce_.acquire();
        swap_.read_slice(swap_offset(fi, p), kPageBytes, bslot.ptr);
        dma_.copy_h2d_async(acq.vram_ptr, bslot.ptr, kPageBytes);
        bounce_.mark_submitted(bslot.index, dma_.stream());
        dma_.synchronize();
        if (swap_page_count_ > 0) --swap_page_count_;
        ++stats_.promote_from_swap;
        stats_.nvme_read_bytes += kPageBytes;
        stats_.pcie_h2d_bytes += kPageBytes;
      }
      pm.loc = Loc::Vram;
      pm.slot = acq.slot;
      set_page_ptrs(fi, p, acq.vram_ptr, acq.vram_ptr + kKPageBytes);
    }
  }
  for (int fi = 0; fi < num_full_slots_; ++fi) flush_table(fi, stream);
}

void PagedKvCache::copy_page_to_staging(int fi, int page, std::uint8_t* dst_k, std::uint8_t* dst_v) {
  const PageMeta& pm = meta_[fi][page];
  if (pm.loc == Loc::Vram) {
    const std::uint8_t* src = vram_slab_.slot_ptr(pm.slot);
    CUDA_CHECK(cudaMemcpyAsync(dst_k, src, kKPageBytes, cudaMemcpyDeviceToDevice, dma_.stream()));
    CUDA_CHECK(cudaMemcpyAsync(dst_v, src + kKPageBytes, kVPageBytes, cudaMemcpyDeviceToDevice, dma_.stream()));
  } else if (pm.loc == Loc::Ram) {
    const std::uint8_t* src = ram_host_ptr(pm.slot);
    dma_.copy_h2d_async(dst_k, src, kKPageBytes);
    dma_.copy_h2d_async(dst_v, src + kKPageBytes, kVPageBytes);
    stats_.pcie_h2d_bytes += kPageBytes;
  } else if (pm.loc == Loc::Swap) {
    ensure_swap_open();
    auto bslot = bounce_.acquire();
    swap_.read_slice(swap_offset(fi, page), kPageBytes, bslot.ptr);
    dma_.copy_h2d_async(dst_k, bslot.ptr, kKPageBytes);
    dma_.copy_h2d_async(dst_v, bslot.ptr + kKPageBytes, kVPageBytes);
    bounce_.mark_submitted(bslot.index, dma_.stream());
    stats_.nvme_read_bytes += kPageBytes;
    stats_.pcie_h2d_bytes += kPageBytes;
  } else {
    throw std::runtime_error("PagedKvCache: read of unallocated page (" + std::to_string(fi) + ", " +
                             std::to_string(page) + ")");
  }
}

void PagedKvCache::prefill_attention(int fi, pk::AttnParams base_at, cudaStream_t stream) {
  const int total_tokens = base_at.L;
  const int num_pages = (total_tokens + kPageTokens - 1) / kPageTokens;
  if (num_pages <= 0) return;

  // Fast path: every active page for this visit is already resident in Tier-0 VRAM -> 1 kernel launch!
  bool all_vram = true;
  for (int p = 0; p < num_pages; ++p) {
    if (meta_[fi][p].loc != Loc::Vram) {
      all_vram = false;
      break;
    }
  }
  if (all_vram) {
    flush_table(fi, stream);
    pk::AttnParams at = base_at;
    at.k = nullptr;
    at.v = nullptr;
    at.k_pages = k_table(fi);
    at.v_pages = v_table(fi);
    at.page_shift = kPageShift;
    at.k_pos_base = 0;
    at.partial_o = nullptr;
    at.partial_m = nullptr;
    at.partial_l = nullptr;
    at.partial_init = true;
    at.partial_final = true;
    pk::attention(at, stream);
    return;
  }

  // Streamed path: batch `batch_pages_` pages at a time into double-buffered VRAM staging so every
  // Tier-1 RAM / Tier-2 NVMe page crosses PCIe ONCE (zero CTA amplification), overlapped with compute!
  const int num_batches = (num_pages + batch_pages_ - 1) / batch_pages_;
  // Ensure attn_prep on `stream` finished writing the newest page(s) before dma_ reads VRAM pages.
  CUDA_CHECK(cudaEventRecord(compute_ready_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(dma_.stream(), compute_ready_, 0));

  auto stage_batch = [&](int b) {
    const int buf = b & 1;
    if (b >= 2) CUDA_CHECK(cudaStreamWaitEvent(dma_.stream(), stage_compute_done_[buf], 0));
    const int p0 = b * batch_pages_;
    const int p1 = std::min(p0 + batch_pages_, num_pages);
    auto* dk = static_cast<std::uint8_t*>(stage_k_[buf].p);
    auto* dv = static_cast<std::uint8_t*>(stage_v_[buf].p);
    for (int p = p0; p < p1; ++p) {
      copy_page_to_staging(fi, p, dk + std::size_t(p - p0) * kKPageBytes, dv + std::size_t(p - p0) * kVPageBytes);
    }
    CUDA_CHECK(cudaEventRecord(stage_dma_done_[buf], dma_.stream()));
  };

  stage_batch(0);
  for (int b = 0; b < num_batches; ++b) {
    const int buf = b & 1;
    if (b + 1 < num_batches) stage_batch(b + 1);
    CUDA_CHECK(cudaStreamWaitEvent(stream, stage_dma_done_[buf], 0));

    const int p0 = b * batch_pages_;
    const int k_base = p0 * kPageTokens;
    const int batch_toks = std::min(total_tokens - k_base, batch_pages_ * kPageTokens);

    pk::AttnParams at = base_at;
    at.k = stage_k_[buf].p;
    at.v = stage_v_[buf].p;
    at.k_pages = nullptr;
    at.v_pages = nullptr;
    at.page_shift = 0;
    at.k_pos_base = k_base;
    at.L = batch_toks;
    at.partial_o = static_cast<float*>(partial_o_.p);
    at.partial_m = static_cast<float*>(partial_m_.p);
    at.partial_l = static_cast<float*>(partial_l_.p);
    at.partial_init = (b == 0);
    at.partial_final = (b == num_batches - 1);
    pk::attention(at, stream);

    CUDA_CHECK(cudaEventRecord(stage_compute_done_[buf], stream));
  }
}

void PagedKvCache::decode_attention_with_swap(int fi, const Layer& layer, int /*visit*/, const float* proj_f32, int M,
                                              int pos0, float* conv_k, float* conv_v, ActT* out, cudaStream_t stream) {
  const int R = PROJ_ROWS_FULL;
  ensure_write_pages(fi, pos0, M, stream);
  pk::f32_to_act(proj_f32, static_cast<ActT*>(dec_proj16_.p), std::size_t(M) * R, stream);

  pk::AttnPrepParams ap;
  ap.proj = static_cast<const ActT*>(dec_proj16_.p);
  ap.ld = R;
  ap.T = M;
  ap.pos0 = pos0;
  ap.full = true;
  ap.q_scale = static_cast<const float*>(layer.q_scale.data);
  ap.r_scale = static_cast<const float*>(layer.r_scale.data);
  ap.k_w = static_cast<const float*>(layer.k_static.data);
  ap.v_w = static_cast<const float*>(layer.v_static.data);
  ap.conv_k = conv_k;
  ap.conv_v = conv_v;
  ap.q = static_cast<ActT*>(dec_q_.p);
  ap.rel = static_cast<ActT*>(dec_rel16_.p);
  ap.k_pages = k_table(fi);
  ap.v_pages = v_table(fi);
  ap.page_shift = kPageShift;
  ap.out_q4 = true;
  pk::attn_prep(ap, stream);

  pk::conv_state_update(conv_k, static_cast<const ActT*>(dec_proj16_.p) + SEG_BEGIN[SEG_K], R, M, KV_DIM, stream);
  pk::conv_state_update(conv_v, static_cast<const ActT*>(dec_proj16_.p) + SEG_BEGIN[SEG_V], R, M, KV_DIM, stream);
  pk::rel_logits(static_cast<const ActT*>(dec_rel16_.p), static_cast<const float*>(layer.rel_proj.data), M,
                 static_cast<ActT*>(dec_rel_.p), stream);

  pk::AttnParams at;
  at.q = static_cast<const ActT*>(dec_q_.p);
  at.kv_q4 = true;
  at.L = pos0 + M;
  at.T = M;
  at.pos0 = pos0;
  at.window = 0;
  at.rel = static_cast<const ActT*>(dec_rel_.p);
  at.proj = static_cast<const ActT*>(dec_proj16_.p);
  at.ld = R;
  at.out = out;
  at.decode_ws = static_cast<float*>(decode_ws_.p);
  at.decode_splits = kDecodeSplits;
  prefill_attention(fi, at, stream);
}

void PagedKvCache::reset() {
  dma_.synchronize();
  vram_slab_.clear();
  for (int fi = 0; fi < num_full_slots_; ++fi) {
    for (int p = 0; p < max_pages_; ++p) {
      meta_[fi][p] = PageMeta{};
      host_k_ptrs_[fi][p] = nullptr;
      host_v_ptrs_[fi][p] = nullptr;
    }
    table_dirty_[fi] = true;
  }
  free_ram_slots_.clear();
  for (int i = int(ram_slots_.size()) - 1; i >= 0; --i) {
    ram_slots_[i] = RamSlot{-1, -1, 0};
    free_ram_slots_.push_back(i);
  }
  ram_page_count_ = 0;
  swap_page_count_ = 0;
  pin_gen_ = 1;
}

void PagedKvCache::export_page_to_host(int fi, int page, std::uint8_t* dst_k, std::uint8_t* dst_v,
                                       cudaStream_t compute_stream) {
  const PageMeta& pm = meta_[fi][page];
  if (pm.loc == Loc::Vram) {
    const std::uint8_t* src = vram_slab_.slot_ptr(pm.slot);
    if (dst_k) CUDA_CHECK(cudaMemcpyAsync(dst_k, src, kKPageBytes, cudaMemcpyDeviceToHost, compute_stream));
    if (dst_v) CUDA_CHECK(cudaMemcpyAsync(dst_v, src + kKPageBytes, kVPageBytes, cudaMemcpyDeviceToHost, compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
  } else if (pm.loc == Loc::Ram) {
    const std::uint8_t* src = ram_host_ptr(pm.slot);
    if (dst_k) std::memcpy(dst_k, src, kKPageBytes);
    if (dst_v) std::memcpy(dst_v, src + kKPageBytes, kVPageBytes);
  } else if (pm.loc == Loc::Swap) {
    ensure_swap_open();
    if (dst_k) swap_.read_slice(swap_offset(fi, page), kKPageBytes, dst_k);
    if (dst_v) swap_.read_slice(swap_offset(fi, page) + kKPageBytes, kVPageBytes, dst_v);
  } else {
    if (dst_k) std::memset(dst_k, 0, kKPageBytes);
    if (dst_v) std::memset(dst_v, 0, kVPageBytes);
  }
}

void PagedKvCache::import_page_from_host(int fi, int page, const std::uint8_t* src_k, const std::uint8_t* src_v,
                                         bool prefer_ram, cudaStream_t compute_stream) {
  if (prefer_ram && max_ram_slots_ > 0) {
    const int rslot = alloc_ram_slot(fi, page);
    if (rslot >= 0) {
      std::uint8_t* dst_host = ram_host_ptr(rslot);
      std::uint8_t* dst_dev = ram_dev_ptr(rslot);
      std::memcpy(dst_host, src_k, kKPageBytes);
      std::memcpy(dst_host + kKPageBytes, src_v, kVPageBytes);
      meta_[fi][page].loc = Loc::Ram;
      meta_[fi][page].slot = rslot;
      set_page_ptrs(fi, page, dst_dev, dst_dev + kKPageBytes);
      ++ram_page_count_;
      return;
    }
  }
  ensure_write_pages(fi, page * kPageTokens, 1, compute_stream);
  std::uint8_t* dst = vram_slab_.slot_ptr(meta_[fi][page].slot);
  CUDA_CHECK(cudaMemcpyAsync(dst, src_k, kKPageBytes, cudaMemcpyHostToDevice, compute_stream));
  CUDA_CHECK(cudaMemcpyAsync(dst + kKPageBytes, src_v, kVPageBytes, cudaMemcpyHostToDevice, compute_stream));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream));
}

void PagedKvCache::flush_all_tables(cudaStream_t stream) {
  for (int fi = 0; fi < num_full_slots_; ++fi) flush_table(fi, stream);
}

PagedKvStats PagedKvCache::stats() const noexcept {
  PagedKvStats s = stats_;
  s.vram_pages = 0;
  s.ram_pages = 0;
  s.swap_pages = 0;
  for (int fi = 0; fi < num_full_slots_; ++fi) {
    for (int p = 0; p < max_pages_; ++p) {
      if (meta_[fi][p].loc == Loc::Vram) ++s.vram_pages;
      else if (meta_[fi][p].loc == Loc::Ram) ++s.ram_pages;
      else if (meta_[fi][p].loc == Loc::Swap) ++s.swap_pages;
    }
  }
  s.vram_bytes = vram_bytes_;
  s.ram_pinned_bytes = 0;
  for (const auto& c : ram_chunks_) s.ram_pinned_bytes += c.size();
  return s;
}

std::string PagedKvCache::summary() const {
  const auto s = stats();
  int hp_chunks = 0;
  for (const auto& c : ram_chunks_) {
    if (c.hugepage_advised()) ++hp_chunks;
  }
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "[paged-kv] max_ctx=%d page=%d | pages: VRAM=%llu (%.2f GiB cap) RAM=%llu (%.2f GiB pinned, hp=%d/%zu) "
                "SWAP=%llu | evict(ram=%llu swap=%llu) promote(ram=%llu swap=%llu) | "
                "DMA(h2d=%.2f GB d2h=%.2f GB) NVMe(w=%.2f GB r=%.2f GB)",
                max_ctx_, kPageTokens, (unsigned long long)s.vram_pages, double(s.vram_bytes) / double(1ull << 30),
                (unsigned long long)s.ram_pages, double(s.ram_pinned_bytes) / double(1ull << 30),
                hp_chunks, ram_chunks_.size(),
                (unsigned long long)s.swap_pages, (unsigned long long)s.evict_to_ram,
                (unsigned long long)s.evict_to_swap, (unsigned long long)s.promote_from_ram,
                (unsigned long long)s.promote_from_swap, double(s.pcie_h2d_bytes) / 1e9, double(s.pcie_d2h_bytes) / 1e9,
                double(s.nvme_write_bytes) / 1e9, double(s.nvme_read_bytes) / 1e9);
  return std::string(buf);
}

}  // namespace ingine::model

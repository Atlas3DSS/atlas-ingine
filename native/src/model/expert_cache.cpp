// Three-tier dynamic MoE expert cache and homology-guided placement (atlas-6g-homology).
#include "model/expert_cache.h"

#include "core/device.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace ingine::model {
namespace {

constexpr std::size_t align256(std::size_t n) noexcept { return (n + 255) / 256 * 256; }

std::size_t proj_code_bytes(ExpertFormat fmt, int N, int K) noexcept {
  switch (fmt) {
    case ExpertFormat::NVFP4: return std::size_t(N) * (K / 2);
    case ExpertFormat::MXFP8: return std::size_t(N) * K;
    case ExpertFormat::ATLAS2: return std::size_t(N) * (K / 4);
    case ExpertFormat::ATLAS3: return std::size_t(N) * ((K * 3) / 8);
    case ExpertFormat::EXL3_2: return std::size_t(N / 16) * (K / 16) * 64;
    case ExpertFormat::EXL3_3: return std::size_t(N / 16) * (K / 16) * 96;
  }
  return std::size_t(N) * (K / 2);
}

std::size_t proj_scale_bytes(ExpertFormat fmt, int N, int K) noexcept {
  switch (fmt) {
    case ExpertFormat::MXFP8: return std::size_t(N) * (K / 32);
    case ExpertFormat::EXL3_2:
    case ExpertFormat::EXL3_3: return std::size_t(K) * 2;  // rin F16 [K]
    default: return std::size_t(N) * (K / 16);
  }
}

std::size_t proj_rout_bytes(ExpertFormat fmt, int N) noexcept {
  return is_exl3(fmt) ? (std::size_t(N) * 2) : 0;  // rout F16 [N]
}

const char* format_name(ExpertFormat fmt) noexcept {
  switch (fmt) {
    case ExpertFormat::NVFP4: return "nvfp4";
    case ExpertFormat::MXFP8: return "mxfp8";
    case ExpertFormat::ATLAS2: return "atlas2";
    case ExpertFormat::ATLAS3: return "atlas3";
    case ExpertFormat::EXL3_2: return "exl3_2";
    case ExpertFormat::EXL3_3: return "exl3_3";
  }
  return "unknown";
}

// Compute per-expert [LAYERS][EXPERTS] importance scores from optional ranking JSON or built-in homology prior.
std::array<std::array<float, EXPERTS>, LAYERS> load_or_build_scores(
    const std::filesystem::path& ranking_path,
    const std::array<std::vector<float>, LAYERS>& router_biases) {
  std::array<std::array<float, EXPERTS>, LAYERS> scores{};
  bool loaded = false;
  if (!ranking_path.empty()) {
    std::ifstream in(ranking_path);
    if (!in) throw std::runtime_error("expert_cache: cannot open ranking file " + ranking_path.string());
    const nlohmann::json j = nlohmann::json::parse(in);
    const nlohmann::json* arr = nullptr;
    if (j.contains("scores") && j["scores"].is_array()) arr = &j["scores"];
    else if (j.contains("router_hits") && j["router_hits"].is_array()) arr = &j["router_hits"];
    if (arr) {
      if (int(arr->size()) != LAYERS) {
        throw std::runtime_error("expert_cache: ranking array must have " + std::to_string(LAYERS) + " layers");
      }
      for (int l = 0; l < LAYERS; ++l) {
        const auto& row = (*arr)[l];
        if (!row.is_array() || int(row.size()) != EXPERTS) {
          throw std::runtime_error("expert_cache: ranking layer " + std::to_string(l) + " must have 128 entries");
        }
        const float visit_w = is_loop_layer(l) ? 3.0f : (l >= 20 ? 1.25f : 1.0f);
        for (int e = 0; e < EXPERTS; ++e) {
          const float v = row[e].get<float>();
          // For raw router_hits, already weighted by 3x visits on loop layers; add tiny floor so ties break cleanly.
          scores[l][e] = (arr == &j["scores"] ? v : (v + 1e-3f * visit_w));
        }
      }
      loaded = true;
    } else if (j.contains("layers") && j["layers"].is_object()) {
      for (int l = 0; l < LAYERS; ++l) {
        const std::string key = std::to_string(l);
        if (!j["layers"].contains(key)) {
          throw std::runtime_error("expert_cache: ranking layers missing key " + key);
        }
        const auto& row = j["layers"][key];
        for (int e = 0; e < EXPERTS; ++e) scores[l][e] = row.at(e).get<float>();
      }
      loaded = true;
    } else {
      throw std::runtime_error("expert_cache: ranking JSON must contain 'scores', 'router_hits', or 'layers'");
    }
  }
  if (!loaded) {
    // Built-in homology-aware prior: 3x looped core (L6-17) > output tail (L20-23) > single-pass prelude,
    // modulated by the checkpoint's learned router_bias (higher bias correlates with higher selection frequency).
    for (int l = 0; l < LAYERS; ++l) {
      const float visit_w = is_loop_layer(l) ? 3.0f : (l >= 20 ? 1.5f : 1.0f);
      for (int e = 0; e < EXPERTS; ++e) {
        const float b = (e < int(router_biases[l].size())) ? router_biases[l][e] : 0.f;
        const float sig = 1.0f / (1.0f + std::exp(-b));
        scores[l][e] = visit_w * (1.0f + 0.35f * sig);
      }
    }
  }
  return scores;
}

}  // namespace

ExpertCache::ExpertLayout ExpertCache::layout_for(ExpertFormat w13_fmt, ExpertFormat w2_fmt) {
  ExpertLayout lo;
  lo.w13_bytes = proj_code_bytes(w13_fmt, 2 * INTER, HIDDEN);
  lo.w13_scale_bytes = proj_scale_bytes(w13_fmt, 2 * INTER, HIDDEN);
  lo.w13_rout_bytes = proj_rout_bytes(w13_fmt, 2 * INTER);
  lo.w2_bytes = proj_code_bytes(w2_fmt, HIDDEN, INTER);
  lo.w2_scale_bytes = proj_scale_bytes(w2_fmt, HIDDEN, INTER);
  lo.w2_rout_bytes = proj_rout_bytes(w2_fmt, HIDDEN);
  std::size_t off = 0;
  lo.off_w13 = off;
  off = align256(off + lo.w13_bytes);
  lo.off_w13_s = off;
  off = align256(off + lo.w13_scale_bytes);
  lo.off_w13_rout = lo.w13_rout_bytes ? off : 0;
  if (lo.w13_rout_bytes) off = align256(off + lo.w13_rout_bytes);
  lo.off_w2 = off;
  off = align256(off + lo.w2_bytes);
  lo.off_w2_s = off;
  off = align256(off + lo.w2_scale_bytes);
  lo.off_w2_rout = lo.w2_rout_bytes ? off : 0;
  if (lo.w2_rout_bytes) off = align256(off + lo.w2_rout_bytes);
  lo.record_bytes = off;
  return lo;
}

ExpertCache::~ExpertCache() {
  dump_router_stats_if_configured();
  if (idx_ready_) cudaEventDestroy(idx_ready_);
  if (pass0_done_) cudaEventDestroy(pass0_done_);
}

std::shared_ptr<ExpertCache> ExpertCache::create(
    const std::filesystem::path& model_dir,
    const std::array<ExpertFormat, LAYERS>& w13_formats,
    const std::array<ExpertFormat, LAYERS>& w2_formats,
    const std::array<std::vector<float>, LAYERS>& router_biases,
    std::size_t expert_vram_budget,
    std::size_t warm_ram_budget,
    const OffloadConfig& cfg,
    cudaStream_t stream) {
  auto cache = std::shared_ptr<ExpertCache>(new ExpertCache());
  cache->w13_formats_ = w13_formats;
  cache->w2_formats_ = w2_formats;
  cache->dump_router_stats_ = cfg.dump_router_stats;

  // Group layers by distinct ExpertLayout::record_bytes so mixed (w13_fmt, w2_fmt) models
  // (e.g. EXL3_2/EXL3_3) share exact-sized VramLruSlab buckets with zero wasted padding.
  std::vector<std::size_t> bucket_rec_bytes;
  for (int l = 0; l < LAYERS; ++l) {
    cache->layouts_[l] = layout_for(w13_formats[l], w2_formats[l]);
    const std::size_t rec = cache->layouts_[l].record_bytes;
    int bidx = -1;
    for (int i = 0; i < int(bucket_rec_bytes.size()); ++i) {
      if (bucket_rec_bytes[i] == rec) {
        bidx = i;
        break;
      }
    }
    if (bidx < 0) {
      bidx = int(bucket_rec_bytes.size());
      bucket_rec_bytes.push_back(rec);
    }
    cache->layer_slab_[l] = bidx;
  }
  const int num_buckets = int(bucket_rec_bytes.size());

  CUDA_CHECK(cudaEventCreateWithFlags(&cache->idx_ready_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&cache->pass0_done_, cudaEventDisableTiming));
  cache->host_idx_ = MappedHostBuffer(8 * TOP_K * sizeof(int));
  cache->host_counts_ = MappedHostBuffer(EXPERTS * sizeof(int));
  for (int i = 0; i < 2; ++i) {
    cache->host_mask_[i] = MappedHostBuffer(EXPERTS);
    cache->dev_mask_[i] = DeviceBuffer(EXPERTS);
  }

  const auto scores = load_or_build_scores(cfg.ranking, router_biases);

  struct Item {
    int l;
    int e;
    float score;
    std::size_t bytes;
    double density;
  };
  std::vector<Item> items;
  items.reserve(LAYERS * EXPERTS);
  std::size_t total_expert_bytes = 0;
  for (int l = 0; l < LAYERS; ++l) {
    const std::size_t rec = cache->layouts_[l].record_bytes;
    for (int e = 0; e < EXPERTS; ++e) {
      cache->meta_[l][e].score = scores[l][e];
      items.push_back(Item{l, e, scores[l][e], rec, double(scores[l][e]) / double(rec)});
      total_expert_bytes += rec;
    }
  }
  std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.density > b.density; });

  OffloadMode desired = OffloadMode::DmaLru;
  if (cfg.mode == "uva") desired = OffloadMode::Uva;
  else if (cfg.mode != "auto" && cfg.mode != "dma-lru") {
    throw std::invalid_argument("expert_cache: unknown offload_mode '" + cfg.mode + "' (expected auto|dma-lru|uva)");
  }

  // Decide LRU slab sizes and tier assignments.
  auto plan_tiers = [&](OffloadMode mode, std::vector<int>& slab_slots, std::size_t& hot_used,
                        std::size_t& warm_used, std::size_t& cold_used, int& n_hot, int& n_warm, int& n_cold) {
    slab_slots.assign(std::size_t(num_buckets), 0);
    std::size_t lru_reserved = 0;
    if (mode == OffloadMode::DmaLru && expert_vram_budget < total_expert_bytes) {
      // Reserve ~28% of expert VRAM budget (capped at 560 MiB) for dynamic LRU slabs, distributed by visit*bytes.
      std::vector<double> bucket_weight(std::size_t(num_buckets), 0.0);
      double sum_w = 0.0;
      for (int l = 0; l < LAYERS; ++l) {
        const int b = cache->layer_slab_[l];
        const double vw = is_loop_layer(l) ? 3.0 : 1.0;
        const double w = vw * double(bucket_rec_bytes[b]);
        bucket_weight[b] += w;
        sum_w += w;
      }
      const std::size_t target_lru = std::min<std::size_t>(std::size_t(560) << 20, expert_vram_budget * 28 / 100);
      // Guarantee at least 16 slots per active record-size bucket (covers 2 decode visits / clean prefill sub-batching).
      std::size_t min_needed = 0;
      for (int b = 0; b < num_buckets; ++b) min_needed += 16 * bucket_rec_bytes[b];
      const std::size_t lru_budget = std::min(expert_vram_budget, std::max(target_lru, min_needed));
      for (int b = 0; b < num_buckets; ++b) {
        const std::size_t rec = bucket_rec_bytes[b];
        const std::size_t share = std::size_t(double(lru_budget) * (bucket_weight[b] / sum_w));
        int slots = std::max(16, int(share / rec));
        while (slots > 8 && lru_reserved + std::size_t(slots) * rec > expert_vram_budget) --slots;
        if (lru_reserved + std::size_t(slots) * rec > expert_vram_budget) {
          slots = int((expert_vram_budget - lru_reserved) / rec);
        }
        slab_slots[b] = slots;
        lru_reserved += std::size_t(slots) * rec;
      }
    }

    const std::size_t hot_budget = expert_vram_budget > lru_reserved ? (expert_vram_budget - lru_reserved) : 0;
    hot_used = warm_used = cold_used = 0;
    n_hot = n_warm = n_cold = 0;
    for (const Item& it : items) {
      ExpertMeta& em = cache->meta_[it.l][it.e];
      if (hot_used + it.bytes <= hot_budget) {
        em.tier = Tier::HotVram;
        hot_used += it.bytes;
        ++n_hot;
      } else if (warm_used + it.bytes <= warm_ram_budget) {
        em.tier = Tier::WarmRam;
        em.warm_offset = warm_used;
        warm_used += it.bytes;
        ++n_warm;
      } else {
        em.tier = Tier::ColdNvme;
        cold_used += it.bytes;
        ++n_cold;
      }
    }
    // Trim any bucket slab that has zero non-HOT experts (nothing will ever miss into it).
    std::vector<int> non_hot_bucket(std::size_t(num_buckets), 0);
    for (int l = 0; l < LAYERS; ++l) {
      const int b = cache->layer_slab_[l];
      for (int e = 0; e < EXPERTS; ++e) {
        if (cache->meta_[l][e].tier != Tier::HotVram) ++non_hot_bucket[b];
      }
    }
    for (int b = 0; b < num_buckets; ++b) {
      if (non_hot_bucket[b] == 0) slab_slots[b] = 0;
    }
  };

  std::vector<int> slab_slots;
  std::size_t hot_used = 0, warm_used = 0, cold_used = 0;
  int n_hot = 0, n_warm = 0, n_cold = 0;
  plan_tiers(desired, slab_slots, hot_used, warm_used, cold_used, n_hot, n_warm, n_cold);

  if (desired == OffloadMode::Uva && n_cold > 0) {
    std::fprintf(stderr,
                 "expert_cache: uva mode requested but %d experts (%.2f GB) spill to COLD NVMe; "
                 "falling back to dma-lru\n",
                 n_cold, double(cold_used) / 1e9);
    desired = OffloadMode::DmaLru;
    plan_tiers(desired, slab_slots, hot_used, warm_used, cold_used, n_hot, n_warm, n_cold);
  }
  cache->mode_ = desired;

  // Allocate Tier 0 HOT VRAM and Dynamic LRU slabs.
  if (hot_used > 0) cache->hot_vram_ = DeviceBuffer(hot_used);
  cache->slabs_.resize(std::size_t(num_buckets));
  std::size_t lru_vram_bytes = 0;
  int lru_slots_total = 0;
  for (int b = 0; b < num_buckets; ++b) {
    if (slab_slots[b] > 0) {
      cache->slabs_[b] = VramLruSlab(slab_slots[b], bucket_rec_bytes[b]);
      lru_vram_bytes += cache->slabs_[b].total_bytes();
      lru_slots_total += slab_slots[b];
    }
  }

  // Allocate Tier 1 WARM pinned host RAM (2 MiB hugepage HostRegion).
  if (warm_used > 0) cache->warm_ram_ = HostRegion(warm_used);

  // Allocate per-layer device pointer tables + pinned host mirrors (6 tables: w13, w13_s/rin, w13_rout, w2, w2_s/rin, w2_rout).
  for (int l = 0; l < LAYERS; ++l) {
    LayerDeviceTables& tb = cache->tables_[l];
    const std::size_t bytes = 6 * EXPERTS * sizeof(std::uint8_t*);
    tb.dev = DeviceBuffer(bytes);
    tb.host_mirror = MappedHostBuffer(bytes);
    auto* dbase = static_cast<std::uint8_t**>(tb.dev.p);
    auto* hbase = static_cast<std::uint8_t**>(tb.host_mirror.data());
    tb.d_w13 = dbase;
    tb.d_w13_s = dbase + EXPERTS;
    tb.d_w13_rout = dbase + 2 * EXPERTS;
    tb.d_w2 = dbase + 3 * EXPERTS;
    tb.d_w2_s = dbase + 4 * EXPERTS;
    tb.d_w2_rout = dbase + 5 * EXPERTS;
    tb.h_w13 = hbase;
    tb.h_w13_s = hbase + EXPERTS;
    tb.h_w13_rout = hbase + 2 * EXPERTS;
    tb.h_w2 = hbase + 3 * EXPERTS;
    tb.h_w2_s = hbase + 4 * EXPERTS;
    tb.h_w2_rout = hbase + 5 * EXPERTS;
    std::memset(hbase, 0, bytes);
  }

  // Stream expert tensors from safetensors into HOT VRAM / WARM HostRegion / COLD mmap index.
  const SafetensorsModel m(model_dir);
  std::unordered_map<const SafetensorsFile*, int> file_to_mmap;
  auto ensure_mmap = [&](const SafetensorsFile* sf) -> int {
    auto it = file_to_mmap.find(sf);
    if (it != file_to_mmap.end()) return it->second;
    const int idx = int(cache->mmap_files_.size());
    cache->mmap_files_.emplace_back(sf->path());
    file_to_mmap.emplace(sf, idx);
    return idx;
  };

  std::size_t max_rec = 0;
  for (int l = 0; l < LAYERS; ++l) max_rec = std::max(max_rec, cache->layouts_[l].record_bytes);
  std::size_t hot_cursor = 0;
  std::size_t max_cold_rec = 0;

  struct LayerRefs {
    SafetensorsRef w13, w13_s, w13_rout, w2, w2_s, w2_rout;
    bool has_w13_rout = false, has_w2_rout = false;
  };
  std::array<LayerRefs, LAYERS> lrefs{};

  // Pass A (metadata & addresses, <1 ms): validate tensor sizes, assign HOT VRAM addresses & COLD mmap slices.
  for (int l = 0; l < LAYERS; ++l) {
    const std::string p = "model.llm.layers." + std::to_string(l) + ".mlp.experts.";
    const bool exl13 = is_exl3(w13_formats[l]);
    const bool exl2 = is_exl3(w2_formats[l]);
    lrefs[l].w13 = m.at(p + "w13_weight");
    lrefs[l].w13_s = m.at(p + (exl13 ? "w13_rin" : "w13_weight_scale"));
    if (exl13) {
      lrefs[l].w13_rout = m.at(p + "w13_rout");
      lrefs[l].has_w13_rout = true;
    }
    lrefs[l].w2 = m.at(p + "w2_weight");
    lrefs[l].w2_s = m.at(p + (exl2 ? "w2_rin" : "w2_weight_scale"));
    if (exl2) {
      lrefs[l].w2_rout = m.at(p + "w2_rout");
      lrefs[l].has_w2_rout = true;
    }
    const ExpertLayout& lo = cache->layouts_[l];
    if (lrefs[l].w13->bytes != std::uint64_t(EXPERTS) * lo.w13_bytes ||
        lrefs[l].w13_s->bytes != std::uint64_t(EXPERTS) * lo.w13_scale_bytes ||
        (exl13 && lrefs[l].w13_rout->bytes != std::uint64_t(EXPERTS) * lo.w13_rout_bytes) ||
        lrefs[l].w2->bytes != std::uint64_t(EXPERTS) * lo.w2_bytes ||
        lrefs[l].w2_s->bytes != std::uint64_t(EXPERTS) * lo.w2_scale_bytes ||
        (exl2 && lrefs[l].w2_rout->bytes != std::uint64_t(EXPERTS) * lo.w2_rout_bytes)) {
      throw std::runtime_error("expert_cache: layer " + std::to_string(l) +
                               " expert tensor byte sizes do not match format " + format_name(w13_formats[l]) + "/" +
                               format_name(w2_formats[l]));
    }
    for (int e = 0; e < EXPERTS; ++e) {
      ExpertMeta& em = cache->meta_[l][e];
      if (em.tier == Tier::HotVram) {
        auto* dst_gpu = static_cast<std::uint8_t*>(cache->hot_vram_.p) + hot_cursor;
        hot_cursor += lo.record_bytes;
        em.hot_ptr = dst_gpu;
        cache->set_host_ptrs(l, e, dst_gpu);
      } else if (em.tier == Tier::ColdNvme) {
        em.cold_w13 = FileSlice{ensure_mmap(lrefs[l].w13.file),
                                lrefs[l].w13->offset + std::uint64_t(e) * lo.w13_bytes, lo.w13_bytes};
        em.cold_w13_s = FileSlice{ensure_mmap(lrefs[l].w13_s.file),
                                  lrefs[l].w13_s->offset + std::uint64_t(e) * lo.w13_scale_bytes, lo.w13_scale_bytes};
        if (exl13) {
          em.cold_w13_rout = FileSlice{ensure_mmap(lrefs[l].w13_rout.file),
                                       lrefs[l].w13_rout->offset + std::uint64_t(e) * lo.w13_rout_bytes,
                                       lo.w13_rout_bytes};
        }
        em.cold_w2 = FileSlice{ensure_mmap(lrefs[l].w2.file),
                               lrefs[l].w2->offset + std::uint64_t(e) * lo.w2_bytes, lo.w2_bytes};
        em.cold_w2_s = FileSlice{ensure_mmap(lrefs[l].w2_s.file),
                                 lrefs[l].w2_s->offset + std::uint64_t(e) * lo.w2_scale_bytes, lo.w2_scale_bytes};
        if (exl2) {
          em.cold_w2_rout = FileSlice{ensure_mmap(lrefs[l].w2_rout.file),
                                      lrefs[l].w2_rout->offset + std::uint64_t(e) * lo.w2_rout_bytes,
                                      lo.w2_rout_bytes};
        }
        max_cold_rec = std::max(max_cold_rec, lo.record_bytes);
      }
    }
  }

  // Pass B (parallel disk read overlapped with async PCIe H2D into HOT VRAM).
  // Each reader thread owns a private 2-slot PinnedBounceRing so bounce slots are never shared across threads.
  int cur_dev = 0;
  CUDA_CHECK(cudaGetDevice(&cur_dev));
  std::mutex stream_mu, err_mu;
  std::exception_ptr load_err;
  std::atomic<int> next_layer{0};
  const int n_threads = 6;
  std::vector<std::thread> readers;
  readers.reserve(n_threads);
  for (int ti = 0; ti < n_threads; ++ti) {
    readers.emplace_back([&]() {
      try {
        CUDA_CHECK(cudaSetDevice(cur_dev));
        PinnedBounceRing thread_stage(2, max_rec);
        while (true) {
          const int l = next_layer.fetch_add(1);
          if (l >= LAYERS) break;
          const ExpertLayout& lo = cache->layouts_[l];
          const LayerRefs& lr = lrefs[l];
          auto read_record_into = [&](int e, std::uint8_t* dst) {
            lr.w13.file->read_range(lr.w13->offset + std::uint64_t(e) * lo.w13_bytes, lo.w13_bytes, dst + lo.off_w13);
            lr.w13_s.file->read_range(lr.w13_s->offset + std::uint64_t(e) * lo.w13_scale_bytes, lo.w13_scale_bytes,
                                      dst + lo.off_w13_s);
            if (lr.has_w13_rout) {
              lr.w13_rout.file->read_range(lr.w13_rout->offset + std::uint64_t(e) * lo.w13_rout_bytes,
                                           lo.w13_rout_bytes, dst + lo.off_w13_rout);
            }
            lr.w2.file->read_range(lr.w2->offset + std::uint64_t(e) * lo.w2_bytes, lo.w2_bytes, dst + lo.off_w2);
            lr.w2_s.file->read_range(lr.w2_s->offset + std::uint64_t(e) * lo.w2_scale_bytes, lo.w2_scale_bytes,
                                     dst + lo.off_w2_s);
            if (lr.has_w2_rout) {
              lr.w2_rout.file->read_range(lr.w2_rout->offset + std::uint64_t(e) * lo.w2_rout_bytes,
                                          lo.w2_rout_bytes, dst + lo.off_w2_rout);
            }
          };
          for (int e = 0; e < EXPERTS; ++e) {
            const ExpertMeta& em = cache->meta_[l][e];
            if (em.tier == Tier::WarmRam) {
              auto* dst_host = reinterpret_cast<std::uint8_t*>(cache->warm_ram_.data()) + em.warm_offset;
              read_record_into(e, dst_host);
            } else if (em.tier == Tier::HotVram) {
              const auto bslot = thread_stage.acquire();
              read_record_into(e, bslot.ptr);
              std::lock_guard<std::mutex> lk(stream_mu);
              CUDA_CHECK(cudaMemcpyAsync(em.hot_ptr, bslot.ptr, lo.record_bytes, cudaMemcpyHostToDevice, stream));
              thread_stage.mark_submitted(bslot.index, stream);
            }
          }
        }
        std::lock_guard<std::mutex> lk(stream_mu);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      } catch (...) {
        std::lock_guard<std::mutex> lk(err_mu);
        if (!load_err) load_err = std::current_exception();
      }
    });
  }
  for (auto& th : readers) th.join();
  if (load_err) std::rethrow_exception(load_err);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Register WARM HostRegion with CUDA for fast pinned DMA and/or UVA zero-copy reads.
  if (warm_used > 0) {
    const bool reg_ok = cache->warm_ram_.register_with_cuda();
    if (!reg_ok && cache->mode_ == OffloadMode::Uva) {
      throw std::runtime_error(
          "expert_cache: uva mode requires CUDA host registration (cudaHostRegister failed); use --offload-mode dma-lru");
    }
    if (cache->mode_ == OffloadMode::Uva) {
      auto* dev_base = static_cast<std::uint8_t*>(cache->warm_ram_.device_ptr());
      for (int l = 0; l < LAYERS; ++l) {
        for (int e = 0; e < EXPERTS; ++e) {
          if (cache->meta_[l][e].tier == Tier::WarmRam) {
            cache->set_host_ptrs(l, e, dev_base + cache->meta_[l][e].warm_offset);
          }
        }
      }
    }
  }

  // Allocate pinned bounce ring for COLD NVMe mmap reads (only when cold experts exist).
  if (n_cold > 0) {
    cache->cold_bounce_ = PinnedBounceRing(16, max_cold_rec);
  }

  // Upload initial layer device pointer tables.
  for (int l = 0; l < LAYERS; ++l) {
    cache->tables_[l].dirty = true;
    cache->flush_layer_table(l);
  }
  cache->dma_.synchronize();

  cache->stats_.hot_experts = n_hot;
  cache->stats_.warm_experts = n_warm;
  cache->stats_.cold_experts = n_cold;
  cache->stats_.lru_slots_total = lru_slots_total;
  cache->stats_.hot_vram_bytes = hot_used;
  cache->stats_.lru_vram_bytes = lru_vram_bytes;
  cache->stats_.warm_ram_bytes = warm_used;
  cache->stats_.cold_nvme_bytes = cold_used;

  std::fprintf(stderr, "%s\n", cache->summary().c_str());
  return cache;
}

kernels::moe::ExpertPtrs ExpertCache::layer_ptrs(int l) const noexcept {
  const LayerDeviceTables& tb = tables_[l];
  return kernels::moe::ExpertPtrs{tb.d_w13, tb.d_w13_s, tb.d_w2, tb.d_w2_s, tb.d_w13_rout, tb.d_w2_rout};
}

void ExpertCache::set_host_ptrs(int l, int e, std::uint8_t* base) {
  const ExpertLayout& lo = layouts_[l];
  LayerDeviceTables& tb = tables_[l];
  tb.h_w13[e] = base + lo.off_w13;
  tb.h_w13_s[e] = base + lo.off_w13_s;
  tb.h_w13_rout[e] = lo.w13_rout_bytes ? (base + lo.off_w13_rout) : nullptr;
  tb.h_w2[e] = base + lo.off_w2;
  tb.h_w2_s[e] = base + lo.off_w2_s;
  tb.h_w2_rout[e] = lo.w2_rout_bytes ? (base + lo.off_w2_rout) : nullptr;
  tb.dirty = true;
}

void ExpertCache::clear_host_ptrs(int l, int e) {
  LayerDeviceTables& tb = tables_[l];
  tb.h_w13[e] = nullptr;
  tb.h_w13_s[e] = nullptr;
  tb.h_w13_rout[e] = nullptr;
  tb.h_w2[e] = nullptr;
  tb.h_w2_s[e] = nullptr;
  tb.h_w2_rout[e] = nullptr;
  tb.dirty = true;
}

void ExpertCache::flush_layer_table(int l) {
  LayerDeviceTables& tb = tables_[l];
  if (!tb.dirty) return;
  dma_.copy_h2d_async(tb.dev.p, tb.host_mirror.data(), tb.dev.bytes);
  tb.dirty = false;
}

void ExpertCache::read_cold_to_host(int l, int e, std::uint8_t* dst) {
  const ExpertLayout& lo = layouts_[l];
  const ExpertMeta& em = meta_[l][e];
  mmap_files_[em.cold_w13.file_idx].read_slice(em.cold_w13.offset, em.cold_w13.bytes, dst + lo.off_w13);
  mmap_files_[em.cold_w13_s.file_idx].read_slice(em.cold_w13_s.offset, em.cold_w13_s.bytes, dst + lo.off_w13_s);
  if (lo.w13_rout_bytes) {
    mmap_files_[em.cold_w13_rout.file_idx].read_slice(em.cold_w13_rout.offset, em.cold_w13_rout.bytes,
                                                      dst + lo.off_w13_rout);
  }
  mmap_files_[em.cold_w2.file_idx].read_slice(em.cold_w2.offset, em.cold_w2.bytes, dst + lo.off_w2);
  mmap_files_[em.cold_w2_s.file_idx].read_slice(em.cold_w2_s.offset, em.cold_w2_s.bytes, dst + lo.off_w2_s);
  if (lo.w2_rout_bytes) {
    mmap_files_[em.cold_w2_rout.file_idx].read_slice(em.cold_w2_rout.offset, em.cold_w2_rout.bytes,
                                                     dst + lo.off_w2_rout);
  }
  stats_.nvme_read_bytes += lo.record_bytes;
  cold_since_madvise_ += lo.record_bytes;
  if (cold_since_madvise_ >= (std::uint64_t(256) << 20)) {
    for (const auto& mf : mmap_files_) mf.advise_dontneed_all();
    cold_since_madvise_ = 0;
  }
}

bool ExpertCache::stage_expert(int l, int e, std::uint64_t pin_gen) {
  const int b = layer_slab_[l];
  const ExpertLayout& lo = layouts_[l];
  const ExpertMeta& em = meta_[l][e];
  const auto res = slabs_[b].acquire(make_key(l, e), pin_gen, is_loop_layer(l));
  if (res.hit) {
    ++stats_.lru_hits;
    return false;
  }
  if (res.evicted) {
    clear_host_ptrs(key_layer(res.evicted_key), key_expert(res.evicted_key));
  }
  if (em.tier == Tier::WarmRam) {
    const std::uint8_t* src = reinterpret_cast<const std::uint8_t*>(warm_ram_.data()) + em.warm_offset;
    dma_.copy_h2d_async(res.vram_ptr, src, lo.record_bytes);
    ++stats_.warm_fetches;
  } else {
    const auto bslot = cold_bounce_.acquire();
    read_cold_to_host(l, e, bslot.ptr);
    dma_.copy_h2d_async(res.vram_ptr, bslot.ptr, lo.record_bytes);
    cold_bounce_.mark_submitted(bslot.index, dma_.stream());
    ++stats_.cold_fetches;
  }
  stats_.pcie_h2d_bytes += lo.record_bytes;
  set_host_ptrs(l, e, res.vram_ptr);
  return true;
}

void ExpertCache::prepare_decode(int l, const int* d_idx, int m, cudaStream_t compute_stream) {
  if (mode_ == OffloadMode::Uva && dump_router_stats_.empty()) return;

  const int nslots = m * TOP_K;
  CUDA_CHECK(cudaMemcpyAsync(host_idx_.data(), d_idx, std::size_t(nslots) * sizeof(int), cudaMemcpyDeviceToHost,
                             compute_stream));
  CUDA_CHECK(cudaEventRecord(idx_ready_, compute_stream));
  CUDA_CHECK(cudaEventSynchronize(idx_ready_));

  const auto* idx = static_cast<const int*>(host_idx_.data());
  std::array<int, 8 * TOP_K> uniq{};
  int nuniq = 0;
  std::uint64_t seen_lo = 0, seen_hi = 0;
  for (int i = 0; i < nslots; ++i) {
    const int e = idx[i];
    if (e < 0 || e >= EXPERTS) continue;
    ++router_hits_[l][e];
    if (e < 64) {
      if ((seen_lo >> e) & 1ull) continue;
      seen_lo |= (1ull << e);
    } else {
      if ((seen_hi >> (e - 64)) & 1ull) continue;
      seen_hi |= (1ull << (e - 64));
    }
    uniq[nuniq++] = e;
  }

  if (mode_ == OffloadMode::Uva) {
    for (int i = 0; i < nuniq; ++i) {
      if (meta_[l][uniq[i]].tier == Tier::HotVram) ++stats_.hot_hits;
      else ++stats_.warm_fetches;
    }
    return;
  }

  const int b = layer_slab_[l];
  const bool prio = is_loop_layer(l);
  const std::uint64_t gen = ++pin_gen_;
  std::array<int, 8 * TOP_K> misses{};
  int nmiss = 0;
  // Pass 1: count HOT hits and pin already-resident LRU hits so misses cannot evict them.
  for (int i = 0; i < nuniq; ++i) {
    const int e = uniq[i];
    if (meta_[l][e].tier == Tier::HotVram) {
      ++stats_.hot_hits;
    } else if (slabs_[b].touch_and_pin(make_key(l, e), gen, prio) >= 0) {
      ++stats_.lru_hits;
    } else {
      misses[nmiss++] = e;
    }
  }
  // Pass 2: stage misses asynchronously on copy_stream.
  bool need_wait = false;
  for (int i = 0; i < nmiss; ++i) {
    need_wait |= stage_expert(l, misses[i], gen);
  }
  if (tables_[l].dirty) {
    flush_layer_table(l);
    need_wait = true;
  }
  if (need_wait) {
    dma_.record_and_wait_on(compute_stream);
  }
}

void ExpertCache::run_decode_layer(
    int l, const int* d_idx, int m, cudaStream_t compute_stream,
    const std::function<void(int tok_offset, int tok_count)>& launch_moe_slice) {
  if (mode_ == OffloadMode::Uva && dump_router_stats_.empty()) {
    launch_moe_slice(0, m);
    return;
  }

  const int nslots = m * TOP_K;
  CUDA_CHECK(cudaMemcpyAsync(host_idx_.data(), d_idx, std::size_t(nslots) * sizeof(int), cudaMemcpyDeviceToHost,
                             compute_stream));
  CUDA_CHECK(cudaEventRecord(idx_ready_, compute_stream));
  CUDA_CHECK(cudaEventSynchronize(idx_ready_));

  const auto* idx = static_cast<const int*>(host_idx_.data());
  for (int i = 0; i < nslots; ++i) {
    const int e = idx[i];
    if (e >= 0 && e < EXPERTS) ++router_hits_[l][e];
  }

  if (mode_ == OffloadMode::Uva) {
    std::uint64_t seen_lo = 0, seen_hi = 0;
    for (int i = 0; i < nslots; ++i) {
      const int e = idx[i];
      if (e < 0 || e >= EXPERTS) continue;
      const std::uint64_t bit = 1ull << (e & 63);
      std::uint64_t& word = (e < 64) ? seen_lo : seen_hi;
      if (word & bit) continue;
      word |= bit;
      if (meta_[l][e].tier == Tier::HotVram) ++stats_.hot_hits;
      else ++stats_.warm_fetches;
    }
    launch_moe_slice(0, m);
    return;
  }

  const int b = layer_slab_[l];
  const int cap = slabs_[b].capacity();
  const bool prio = is_loop_layer(l);
  if (cap == 0) {
    // All experts in this record-size bucket are HOT VRAM resident.
    std::uint64_t seen_lo = 0, seen_hi = 0;
    for (int i = 0; i < nslots; ++i) {
      const int e = idx[i];
      if (e < 0 || e >= EXPERTS) continue;
      const std::uint64_t bit = 1ull << (e & 63);
      std::uint64_t& word = (e < 64) ? seen_lo : seen_hi;
      if (word & bit) continue;
      word |= bit;
      ++stats_.hot_hits;
    }
    if (tables_[l].dirty) {
      flush_layer_table(l);
      dma_.record_and_wait_on(compute_stream);
    }
    launch_moe_slice(0, m);
    return;
  }

  int t0 = 0;
  while (t0 < m) {
    if (t0 > 0) CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    std::uint64_t seen_lo = 0, seen_hi = 0;
    std::array<int, 8 * TOP_K> slice_non_hot{};
    int n_non_hot = 0;
    int n_hot = 0;
    int t1 = t0;
    while (t1 < m) {
      std::uint64_t cand_lo = seen_lo, cand_hi = seen_hi;
      std::array<int, TOP_K> add_non_hot{};
      int n_add_non_hot = 0;
      int n_add_hot = 0;
      for (int k = 0; k < TOP_K; ++k) {
        const int e = idx[t1 * TOP_K + k];
        if (e < 0 || e >= EXPERTS) continue;
        const std::uint64_t bit = 1ull << (e & 63);
        std::uint64_t& word = (e < 64) ? cand_lo : cand_hi;
        if (word & bit) continue;
        word |= bit;
        if (meta_[l][e].tier == Tier::HotVram) {
          ++n_add_hot;
        } else {
          add_non_hot[n_add_non_hot++] = e;
        }
      }
      if (t1 > t0 && n_non_hot + n_add_non_hot > cap) break;
      seen_lo = cand_lo;
      seen_hi = cand_hi;
      n_hot += n_add_hot;
      for (int i = 0; i < n_add_non_hot; ++i) slice_non_hot[n_non_hot++] = add_non_hot[i];
      ++t1;
    }

    stats_.hot_hits += std::uint64_t(n_hot);
    const std::uint64_t gen = ++pin_gen_;
    std::array<int, 8 * TOP_K> misses{};
    int nmiss = 0;
    for (int i = 0; i < n_non_hot; ++i) {
      const int e = slice_non_hot[i];
      if (slabs_[b].touch_and_pin(make_key(l, e), gen, prio) >= 0) {
        ++stats_.lru_hits;
      } else {
        misses[nmiss++] = e;
      }
    }
    bool need_wait = false;
    for (int i = 0; i < nmiss; ++i) {
      need_wait |= stage_expert(l, misses[i], gen);
    }
    if (tables_[l].dirty) {
      flush_layer_table(l);
      need_wait = true;
    }
    if (need_wait) {
      dma_.record_and_wait_on(compute_stream);
    }
    launch_moe_slice(t0, t1 - t0);
    t0 = t1;
  }
}

void ExpertCache::run_prefill_layer(
    int l, const int* d_counts, cudaStream_t compute_stream,
    const std::function<void(const std::uint8_t* d_active_mask)>& launch_moe_pair) {
  if (mode_ == OffloadMode::Uva && dump_router_stats_.empty()) {
    launch_moe_pair(nullptr);
    return;
  }

  CUDA_CHECK(cudaMemcpyAsync(host_counts_.data(), d_counts, EXPERTS * sizeof(int), cudaMemcpyDeviceToHost,
                             compute_stream));
  CUDA_CHECK(cudaEventRecord(idx_ready_, compute_stream));
  CUDA_CHECK(cudaEventSynchronize(idx_ready_));

  const auto* counts = static_cast<const int*>(host_counts_.data());
  std::vector<int> active_hot;
  std::vector<int> misses;
  active_hot.reserve(EXPERTS);
  misses.reserve(EXPERTS);
  for (int e = 0; e < EXPERTS; ++e) {
    if (counts[e] <= 0) continue;
    router_hits_[l][e] += std::uint64_t(counts[e]);
    if (meta_[l][e].tier == Tier::HotVram) {
      active_hot.push_back(e);
      ++stats_.hot_hits;
    } else {
      misses.push_back(e);
    }
  }

  if (mode_ == OffloadMode::Uva) {
    stats_.warm_fetches += misses.size();
    launch_moe_pair(nullptr);
    return;
  }

  if (misses.empty()) {
    if (tables_[l].dirty) {
      flush_layer_table(l);
      dma_.record_and_wait_on(compute_stream);
    }
    launch_moe_pair(nullptr);
    return;
  }

  const int b = layer_slab_[l];
  const int cap = slabs_[b].capacity();
  const bool prio = is_loop_layer(l);
  if (cap <= 0) {
    throw std::runtime_error("expert_cache: zero LRU slab capacity for format " +
                             std::string(format_name(w13_formats_[l])) + "/" +
                             std::string(format_name(w2_formats_[l])));
  }

  if (int(misses.size()) <= cap) {
    const std::uint64_t gen = ++pin_gen_;
    std::vector<int> lru_hits;
    std::vector<int> to_stage;
    lru_hits.reserve(misses.size());
    to_stage.reserve(misses.size());
    for (int e : misses) {
      if (slabs_[b].touch_and_pin(make_key(l, e), gen, prio) >= 0) {
        ++stats_.lru_hits;
        lru_hits.push_back(e);
      } else {
        to_stage.push_back(e);
      }
    }
    if (to_stage.empty()) {
      if (tables_[l].dirty) {
        flush_layer_table(l);
        dma_.record_and_wait_on(compute_stream);
      }
      launch_moe_pair(nullptr);
      return;
    }
    if (!active_hot.empty() || !lru_hits.empty()) {
      // Overlap Pass 0 (already-resident HOT + pinned LRU hits, valid in tables_[l].dev) on compute_stream
      // with async PCIe H2D staging of `to_stage` into unreferenced LRU slots on dma_.stream().
      auto* hmask0 = static_cast<std::uint8_t*>(host_mask_[0].data());
      std::memset(hmask0, 0, EXPERTS);
      for (int e : active_hot) hmask0[e] = 1;
      for (int e : lru_hits) hmask0[e] = 1;
      CUDA_CHECK(cudaMemcpyAsync(dev_mask_[0].p, hmask0, EXPERTS, cudaMemcpyHostToDevice, compute_stream));
      launch_moe_pair(static_cast<const std::uint8_t*>(dev_mask_[0].p));
      CUDA_CHECK(cudaEventRecord(pass0_done_, compute_stream));

      auto* hmask1 = static_cast<std::uint8_t*>(host_mask_[1].data());
      std::memset(hmask1, 0, EXPERTS);
      for (int e : to_stage) {
        hmask1[e] = 1;
        (void)stage_expert(l, e, gen);
      }
      // Wait for Pass 0 to finish reading tables_[l].dev before flushing the updated pointer table (6 KB).
      CUDA_CHECK(cudaStreamWaitEvent(dma_.stream(), pass0_done_, 0));
      flush_layer_table(l);
      dma_.copy_h2d_async(dev_mask_[1].p, hmask1, EXPERTS);
      dma_.record_and_wait_on(compute_stream);
      launch_moe_pair(static_cast<const std::uint8_t*>(dev_mask_[1].p));
      return;
    }
    for (int e : to_stage) (void)stage_expert(l, e, gen);
    flush_layer_table(l);
    dma_.record_and_wait_on(compute_stream);
    launch_moe_pair(nullptr);
    return;
  }

  // Sub-batched prefill when active misses exceed the bucket's LRU slab capacity.
  // Overlap active_hot compute in Pass 0 with batch 0's PCIe H2D DMA when HOT experts exist.
  bool hot_done = active_hot.empty();
  for (std::size_t off = 0; off < misses.size(); off += std::size_t(cap)) {
    if (off > 0) CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    const std::uint64_t gen = ++pin_gen_;
    const std::size_t end = std::min(misses.size(), off + std::size_t(cap));
    // Touch resident hits in this batch first so they aren't evicted by misses in the same batch.
    std::vector<int> batch_hits;
    std::vector<int> batch_stage;
    batch_hits.reserve(end - off);
    batch_stage.reserve(end - off);
    for (std::size_t j = off; j < end; ++j) {
      const int e = misses[j];
      if (slabs_[b].touch_and_pin(make_key(l, e), gen, prio) >= 0) {
        ++stats_.lru_hits;
        batch_hits.push_back(e);
      } else {
        batch_stage.push_back(e);
      }
    }
    const bool overlap_p0 = (!hot_done || !batch_hits.empty()) && !batch_stage.empty();
    if (overlap_p0) {
      auto* hmask0 = static_cast<std::uint8_t*>(host_mask_[0].data());
      std::memset(hmask0, 0, EXPERTS);
      if (!hot_done) {
        for (int e : active_hot) hmask0[e] = 1;
        hot_done = true;
      }
      for (int e : batch_hits) hmask0[e] = 1;
      CUDA_CHECK(cudaMemcpyAsync(dev_mask_[0].p, hmask0, EXPERTS, cudaMemcpyHostToDevice, compute_stream));
      launch_moe_pair(static_cast<const std::uint8_t*>(dev_mask_[0].p));
      CUDA_CHECK(cudaEventRecord(pass0_done_, compute_stream));

      auto* hmask1 = static_cast<std::uint8_t*>(host_mask_[1].data());
      std::memset(hmask1, 0, EXPERTS);
      for (int e : batch_stage) {
        hmask1[e] = 1;
        (void)stage_expert(l, e, gen);
      }
      CUDA_CHECK(cudaStreamWaitEvent(dma_.stream(), pass0_done_, 0));
      flush_layer_table(l);
      dma_.copy_h2d_async(dev_mask_[1].p, hmask1, EXPERTS);
      dma_.record_and_wait_on(compute_stream);
      launch_moe_pair(static_cast<const std::uint8_t*>(dev_mask_[1].p));
    } else {
      auto* hmask = static_cast<std::uint8_t*>(host_mask_[0].data());
      std::memset(hmask, 0, EXPERTS);
      if (!hot_done) {
        for (int e : active_hot) hmask[e] = 1;
        hot_done = true;
      }
      for (std::size_t j = off; j < end; ++j) hmask[misses[j]] = 1;
      for (int e : batch_stage) (void)stage_expert(l, e, gen);
      flush_layer_table(l);
      dma_.copy_h2d_async(dev_mask_[0].p, hmask, EXPERTS);
      dma_.record_and_wait_on(compute_stream);
      launch_moe_pair(static_cast<const std::uint8_t*>(dev_mask_[0].p));
    }
  }
}

void ExpertCache::reset_runtime_stats() noexcept {
  stats_.hot_hits = 0;
  stats_.lru_hits = 0;
  stats_.warm_fetches = 0;
  stats_.cold_fetches = 0;
  stats_.pcie_h2d_bytes = 0;
  stats_.nvme_read_bytes = 0;
}

void ExpertCache::dump_router_stats_if_configured() const {
  if (dump_router_stats_.empty()) return;
  try {
    nlohmann::json out;
    out["layers"] = LAYERS;
    out["experts"] = EXPERTS;
    nlohmann::json hits = nlohmann::json::array();
    nlohmann::json scores = nlohmann::json::array();
    for (int l = 0; l < LAYERS; ++l) {
      nlohmann::json row_h = nlohmann::json::array();
      nlohmann::json row_s = nlohmann::json::array();
      const float layer_boost = is_loop_layer(l) ? 1.0f : (l >= 20 ? 1.25f : 1.0f);
      for (int e = 0; e < EXPERTS; ++e) {
        row_h.push_back(router_hits_[l][e]);
        row_s.push_back(double(router_hits_[l][e]) * double(layer_boost));
      }
      hits.push_back(std::move(row_h));
      scores.push_back(std::move(row_s));
    }
    out["router_hits"] = std::move(hits);
    out["scores"] = std::move(scores);
    std::ofstream f(dump_router_stats_);
    if (f) f << out.dump(2) << "\n";
  } catch (...) {
  }
}

std::string ExpertCache::summary() const {
  const std::uint64_t total_lookups = stats_.hot_hits + stats_.lru_hits + stats_.warm_fetches + stats_.cold_fetches;
  const double hit_pct =
      total_lookups > 0 ? 100.0 * double(stats_.hot_hits + stats_.lru_hits) / double(total_lookups) : 0.0;
  const double rss_gb = double(process_rss_bytes()) / 1e9;
  char buf[512];
  std::snprintf(
      buf, sizeof(buf),
      "[atlas-6g] mode=%s | experts: HOT=%d (%.2f GB VRAM) + LRU=%d slots (%.2f GB VRAM) | WARM=%d (%.2f GB RAM%s) | "
      "COLD=%d (%.2f GB NVMe mmap) | RSS=%.2f GB | lookups=%llu hit=%.1f%% (hot=%llu lru=%llu warm=%llu cold=%llu "
      "h2d=%.2f GB nvme=%.2f GB)",
      mode_ == OffloadMode::DmaLru ? "dma-lru" : "uva", stats_.hot_experts, double(stats_.hot_vram_bytes) / 1e9,
      stats_.lru_slots_total, double(stats_.lru_vram_bytes) / 1e9, stats_.warm_experts,
      double(stats_.warm_ram_bytes) / 1e9, warm_ram_.registered() ? " pinned" : " pageable", stats_.cold_experts,
      double(stats_.cold_nvme_bytes) / 1e9, rss_gb, static_cast<unsigned long long>(total_lookups), hit_pct,
      static_cast<unsigned long long>(stats_.hot_hits), static_cast<unsigned long long>(stats_.lru_hits),
      static_cast<unsigned long long>(stats_.warm_fetches), static_cast<unsigned long long>(stats_.cold_fetches),
      double(stats_.pcie_h2d_bytes) / 1e9, double(stats_.nvme_read_bytes) / 1e9);
  return std::string(buf);
}

}  // namespace ingine::model

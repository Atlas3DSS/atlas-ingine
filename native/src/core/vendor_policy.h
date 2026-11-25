// Pure CUDA-free C++20 target vendor architecture, tile, and memory-streaming policy helpers.
// Incorporates verified patterns studied from Strata / Project Maya (MIT) and ggml-sycl (MIT) while keeping
// execution 100% pure native atlas-ingine C++. Host-unit-testable with zero vendor SDK headers required.
#pragma once

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace ingine::vendor {

namespace hip {

enum class ArchFamily : std::uint8_t {
  Unknown = 0,
  Rdna1 = 1,
  Rdna2 = 2,
  Rdna3 = 3,
  Rdna35 = 4,
  Rdna4 = 5,
  Cdna1 = 6,
  Cdna2 = 7,
  Cdna3 = 8,
  Cdna4 = 9,
};

struct HipArchInfo {
  ArchFamily family = ArchFamily::Unknown;
  int warp_size = 32;
  std::size_t lds_bytes = 65536;
  bool has_sudot4 = false;
  bool has_sdot4 = false;
  bool has_rdna_wmma = false;
  bool is_wave64 = false;
  bool requires_fp_contract_on = false;
  bool validated_tier = false;
};

constexpr std::string_view strip_target_features(std::string_view arch) noexcept {
  const std::size_t colon = arch.find(':');
  return colon == std::string_view::npos ? arch : arch.substr(0, colon);
}

constexpr HipArchInfo classify_hip_arch(std::string_view raw_arch) noexcept {
  const std::string_view arch = strip_target_features(raw_arch);
  if (arch == "gfx1100" || arch == "gfx1101" || arch == "gfx1102" || arch == "gfx1103") {
    const bool has_wmma = (arch != "gfx1103");
    const bool validated = (arch == "gfx1100" || arch == "gfx1101");
    return HipArchInfo{ArchFamily::Rdna3, 32, 65536, true, false, has_wmma, false, false, validated};
  }
  if (arch == "gfx1150" || arch == "gfx1151") {
    return HipArchInfo{ArchFamily::Rdna35, 32, 65536, true, false, true, false, false, (arch == "gfx1151")};
  }
  if (arch == "gfx1200" || arch == "gfx1201") {
    return HipArchInfo{ArchFamily::Rdna4, 32, 65536, true, false, true, false, false, true};
  }
  if (arch == "gfx1030" || arch == "gfx1031" || arch == "gfx1032" || arch == "gfx1034") {
    return HipArchInfo{ArchFamily::Rdna2, 32, 65536, false, (arch != "gfx1034"), false, false, false, false};
  }
  if (arch == "gfx1010" || arch == "gfx1011" || arch == "gfx1012") {
    return HipArchInfo{ArchFamily::Rdna1, 32, 65536, false, false, false, false, false, false};
  }
  if (arch == "gfx906" || arch == "gfx908") {
    return HipArchInfo{ArchFamily::Cdna1, 64, 65536, false, true, false, true, false, false};
  }
  if (arch == "gfx90a") {
    return HipArchInfo{ArchFamily::Cdna2, 64, 65536, false, true, false, true, false, false};
  }
  if (arch == "gfx942") {
    return HipArchInfo{ArchFamily::Cdna3, 64, 65536, false, true, false, true, true, false};
  }
  if (arch == "gfx950") {
    return HipArchInfo{ArchFamily::Cdna4, 64, 65536, false, true, false, true, true, false};
  }
  return HipArchInfo{};
}

// RDNA3 (gfx11) vs RDNA4 (gfx12) 16x16x16 WMMA output lane row mapping (Strata / Maya rdna_wmma pattern, MIT).
constexpr int wmma_result_row(ArchFamily family, int lane_half, int value_idx) noexcept {
  return (family == ArchFamily::Rdna4) ? (8 * lane_half + value_idx) : (2 * value_idx + lane_half);
}

// Compute compressed-to-LDS tile footprint for wave32 RDNA direct MoE dequantization without FP16 buffers.
constexpr std::size_t compressed_lds_tile_bytes(int tile_rows, int tile_cols, int bits_per_weight) noexcept {
  if (tile_rows <= 0 || tile_cols <= 0 || bits_per_weight <= 0) return 0;
  return (static_cast<std::size_t>(tile_rows) * static_cast<std::size_t>(tile_cols) *
          static_cast<std::size_t>(bits_per_weight) + 7u) / 8u;
}

// Host policy for HIP scalar EXL3 prompt tile row selection (adapted from Strata exl3_hip_scalar.hpp, MIT).
inline int parse_exl3_scalar_mode(std::string_view value) noexcept {
  if (value.empty() || value == "0" || value == "8") return 8;
  if (value == "auto") return -1;
  if (value == "12") return 12;
  if (value == "16") return 16;
  if (value == "24") return 24;
  return 8;
}

inline int parse_positive_cutoff(std::string_view value, int fallback) noexcept {
  if (value.empty()) return fallback;
  unsigned acc = 0;
  for (char c : value) {
    if (c < '0' || c > '9') return fallback;
    const unsigned digit = static_cast<unsigned>(c - '0');
    if (acc > (static_cast<unsigned>(INT_MAX) - digit) / 10u) return fallback;
    acc = acc * 10u + digit;
  }
  return static_cast<int>(acc);
}

constexpr int select_exl3_scalar_rows(int mode, bool gate_up, int max_rows, int gate_up_min, int down_min) noexcept {
  if (mode != -1) return mode;
  return gate_up ? (max_rows < gate_up_min ? 8 : 12) : (max_rows < down_min ? 12 : 24);
}

}  // namespace hip

namespace mlx {

enum class MemoryResidencyMode : std::uint8_t {
  FullUnifiedResident = 0,
  BoundedStreamingRing = 1,
};

inline constexpr std::size_t kCompactPhoneRingCapBytes = 2469606195ull;  // ~2.3 GiB
inline constexpr std::size_t kTabletMacRingCapBytes = 3221225472ull;     // 3.0 GiB
inline constexpr std::size_t kDefaultPreadPieceBytes = 1ull << 20;       // 1 MiB parallel pread pieces
inline constexpr std::size_t kDefaultSlotBytes = 32ull << 20;            // 32 MiB double/ring-buffered slots

struct StreamingBudgetPlan {
  MemoryResidencyMode mode = MemoryResidencyMode::FullUnifiedResident;
  std::size_t total_ram_bytes = 0;
  std::size_t recommended_working_set_bytes = 0;
  std::size_t model_pack_bytes = 0;
  std::size_t staging_ring_bytes = 0;
  std::size_t slot_bytes = kDefaultSlotBytes;
  std::size_t ring_slots = 0;
  std::size_t pread_piece_bytes = kDefaultPreadPieceBytes;
  bool pin_host_memory = false;
};

constexpr std::size_t align_page_up(std::size_t bytes, std::size_t page_size = 16384u) noexcept {
  if (page_size <= 1u || bytes == 0u) return bytes;
  const std::size_t rem = bytes % page_size;
  return rem == 0u ? bytes : (bytes + (page_size - rem));
}

// Plans unified-memory full residency vs fixed 2.3–3.0 GiB streaming staging ring (for 4–8 GB Apple Silicon /
// iPhone / iPad alongside our ~9 GB EXL3 pack and up to ~14 GB 3-bit S quant packs).
constexpr StreamingBudgetPlan plan_streaming_budget(std::size_t total_ram_bytes,
                                                    std::size_t recommended_working_set_bytes,
                                                    std::size_t model_pack_bytes,
                                                    bool compact_mobile_target = false,
                                                    std::size_t slot_bytes = kDefaultSlotBytes) noexcept {
  StreamingBudgetPlan plan;
  plan.total_ram_bytes = total_ram_bytes;
  plan.recommended_working_set_bytes =
      recommended_working_set_bytes > 0 ? recommended_working_set_bytes : (total_ram_bytes * 3u) / 4u;
  plan.model_pack_bytes = model_pack_bytes;
  plan.slot_bytes = slot_bytes > 0 ? slot_bytes : kDefaultSlotBytes;

  constexpr std::size_t kMinHeadroomBytes = 1536ull << 20;  // keep 1.5 GiB headroom for KV + activations + OS
  const bool fits_full =
      plan.recommended_working_set_bytes > model_pack_bytes + kMinHeadroomBytes &&
      model_pack_bytes > 0 && !compact_mobile_target;

  if (fits_full) {
    plan.mode = MemoryResidencyMode::FullUnifiedResident;
    plan.staging_ring_bytes = std::min<std::size_t>(plan.slot_bytes * 4u, 256ull << 20);
    plan.ring_slots = std::max<std::size_t>(2u, plan.staging_ring_bytes / plan.slot_bytes);
    plan.pin_host_memory = total_ram_bytes >= (24ull << 30);
    return plan;
  }

  plan.mode = MemoryResidencyMode::BoundedStreamingRing;
  const std::size_t hard_cap = (total_ram_bytes <= (6ull << 30) || compact_mobile_target)
                                   ? kCompactPhoneRingCapBytes
                                   : kTabletMacRingCapBytes;
  const std::size_t working_cap =
      plan.recommended_working_set_bytes > (512ull << 20)
          ? (plan.recommended_working_set_bytes - (512ull << 20))
          : plan.slot_bytes * 2u;
  const std::size_t raw_budget = std::min(hard_cap, std::min(working_cap, kTabletMacRingCapBytes));
  std::size_t slots = std::max<std::size_t>(2u, raw_budget / plan.slot_bytes);
  // Keep exact byte count <= hard_cap at slot granularity.
  while (slots > 2u && slots * plan.slot_bytes > hard_cap) --slots;
  plan.ring_slots = slots;
  plan.staging_ring_bytes = slots * plan.slot_bytes;
  plan.pin_host_memory = false;  // avoid Jetsam / swap pressure on compact 4-8 GB unified devices
  return plan;
}

}  // namespace mlx

namespace sycl {

enum class IntelArchFamily : std::uint8_t {
  Unknown = 0,
  XeLp = 1,
  XeHpgAlchemist = 2,
  XeHpcPonteVecchio = 3,
  Xe2Battlemage = 4,
  Xe2LunarLake = 5,
};

struct SyclArchInfo {
  IntelArchFamily family = IntelArchFamily::Unknown;
  int min_subgroup = 8;
  int max_subgroup = 16;
  int preferred_subgroup = 16;
  bool has_xmx_dpas = false;
  bool prefer_level_zero_direct_alloc = true;
  bool requires_dnnl_intel_vendor_filter = true;
};

constexpr SyclArchInfo classify_intel_device(std::string_view name_lower, int max_subgroup) noexcept {
  if (name_lower.find("b580") != std::string_view::npos || name_lower.find("b570") != std::string_view::npos ||
      name_lower.find("battlemage") != std::string_view::npos || max_subgroup >= 32) {
    if (name_lower.find("ponte") != std::string_view::npos || name_lower.find("max ") != std::string_view::npos) {
      return SyclArchInfo{IntelArchFamily::XeHpcPonteVecchio, 16, 32, 16, true, true, true};
    }
    if (name_lower.find("lunar") != std::string_view::npos || name_lower.find("140v") != std::string_view::npos) {
      return SyclArchInfo{IntelArchFamily::Xe2LunarLake, 16, 32, 16, true, false, true};
    }
    return SyclArchInfo{IntelArchFamily::Xe2Battlemage, 16, 32, 16, true, true, true};
  }
  if (name_lower.find("arc") != std::string_view::npos || name_lower.find("a770") != std::string_view::npos ||
      name_lower.find("a750") != std::string_view::npos || name_lower.find("alchemist") != std::string_view::npos ||
      max_subgroup == 16) {
    return SyclArchInfo{IntelArchFamily::XeHpgAlchemist, 8, 16, 16, true, true, true};
  }
  return SyclArchInfo{IntelArchFamily::XeLp, 8, 16, 8, false, false, true};
}

constexpr int select_rmsnorm_workgroup(int hidden_size, int subgroup_size) noexcept {
  const int sg = (subgroup_size == 8 || subgroup_size == 16 || subgroup_size == 32) ? subgroup_size : 16;
  const int target = hidden_size >= 2048 ? 256 : 128;
  const int aligned = (target / sg) * sg;
  return aligned >= sg ? aligned : sg;
}

constexpr bool dnnl_vendor_filter_is_intel(std::string_view env_value) noexcept {
  return env_value == "INTEL" || env_value == "intel";
}

}  // namespace sycl

}  // namespace ingine::vendor
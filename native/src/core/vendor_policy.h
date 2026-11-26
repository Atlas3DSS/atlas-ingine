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
#include <vector>

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
  bool is_unified_apu = false;
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
    return HipArchInfo{ArchFamily::Rdna3, 32, 65536, true, false, has_wmma, false, false, validated, false};
  }
  if (arch == "gfx1150" || arch == "gfx1151") {
    return HipArchInfo{ArchFamily::Rdna35, 32, 65536, true, false, true, false, false, (arch == "gfx1151"), true};
  }
  if (arch == "gfx1200" || arch == "gfx1201") {
    return HipArchInfo{ArchFamily::Rdna4, 32, 65536, true, false, true, false, false, true, false};
  }
  if (arch == "gfx1030" || arch == "gfx1031" || arch == "gfx1032" || arch == "gfx1034") {
    return HipArchInfo{ArchFamily::Rdna2, 32, 65536, false, (arch != "gfx1034"), false, false, false, false, false};
  }
  if (arch == "gfx1010" || arch == "gfx1011" || arch == "gfx1012") {
    return HipArchInfo{ArchFamily::Rdna1, 32, 65536, false, false, false, false, false, false, false};
  }
  if (arch == "gfx906" || arch == "gfx908") {
    return HipArchInfo{ArchFamily::Cdna1, 64, 65536, false, true, false, true, false, false, false};
  }
  if (arch == "gfx90a") {
    return HipArchInfo{ArchFamily::Cdna2, 64, 65536, false, true, false, true, false, false, false};
  }
  if (arch == "gfx942") {
    return HipArchInfo{ArchFamily::Cdna3, 64, 65536, false, true, false, true, true, false, false};
  }
  if (arch == "gfx950") {
    return HipArchInfo{ArchFamily::Cdna4, 64, 65536, false, true, false, true, true, false, false};
  }
  return HipArchInfo{};
}

// Owner-pinned benchmark floor numbers for AMD Strix Halo Ryzen AI Max+ 395 128 GB (gfx1151) + Qwen3.8-Flash-Next
// (recorded strictly as numeric performance targets to beat; proprietary reference, strict zero code reuse).
struct StrixHaloMetrics {
  double decode_code_tok_s = 0.0;
  double decode_prose_tok_s = 0.0;
  double decode_edit_tok_s = 0.0;
  double decode_tool_tok_s = 0.0;
  double decode_multilang_tok_s = 0.0;
  double prefill_8k_tok_s = 0.0;
  double prefill_32k_tok_s = 0.0;
  double prefill_131k_tok_s = 0.0;
  double agent_turn_30k_96k_s = 0.0;
  double agent_ttft_30k_96k_s = 0.0;
  double cold_34k_s = 0.0;
};

struct StrixHaloTargetFloor {
  double min_decode_code_tok_s = 100.0;
  double min_decode_prose_tok_s = 70.0;
  double min_decode_edit_tok_s = 121.0;
  double min_decode_tool_tok_s = 88.0;
  double min_decode_multilang_tok_s = 67.0;
  double min_prefill_8k_tok_s = 1983.0;
  double min_prefill_32k_tok_s = 1876.0;
  double min_prefill_131k_tok_s = 1796.0;
  double max_agent_turn_30k_96k_s = 2.1;
  double max_agent_ttft_30k_96k_s = 1.2;
  double max_cold_34k_s = 19.0;

  constexpr bool meets_all(const StrixHaloMetrics& m) const noexcept {
    return m.decode_code_tok_s > min_decode_code_tok_s &&
           m.decode_prose_tok_s > min_decode_prose_tok_s &&
           m.decode_edit_tok_s > min_decode_edit_tok_s &&
           m.decode_tool_tok_s > min_decode_tool_tok_s &&
           m.decode_multilang_tok_s > min_decode_multilang_tok_s &&
           m.prefill_8k_tok_s > min_prefill_8k_tok_s &&
           m.prefill_32k_tok_s > min_prefill_32k_tok_s &&
           m.prefill_131k_tok_s > min_prefill_131k_tok_s &&
           m.agent_turn_30k_96k_s > 0.0 && m.agent_turn_30k_96k_s < max_agent_turn_30k_96k_s &&
           m.agent_ttft_30k_96k_s > 0.0 && m.agent_ttft_30k_96k_s < max_agent_ttft_30k_96k_s &&
           m.cold_34k_s > 0.0 && m.cold_34k_s < max_cold_34k_s;
  }
};

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

constexpr bool can_async_blit_d2d(bool same_buffer, std::size_t src_offset, std::size_t dst_offset,
                                  std::size_t bytes) noexcept {
  if (bytes == 0 || !same_buffer) return true;
  if (src_offset <= dst_offset) return bytes <= (dst_offset - src_offset);
  return bytes <= (src_offset - dst_offset);
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

inline constexpr std::uint32_t kIntelPciVendorId = 0x8086u;

enum class IntelArchFamily : std::uint8_t {
  Unknown = 0,
  XeLp = 1,
  XeHpgAlchemist = 2,
  XeHpcPonteVecchio = 3,
  Xe2Battlemage = 4,
  Xe2LunarLake = 5,
};

struct SyclDeviceCandidate {
  bool is_gpu = false;
  bool is_level_zero = false;
  std::string_view vendor_lower{};
  std::uint32_t vendor_id = 0;
};

struct SyclDeviceProbe {
  bool is_gpu = false;
  bool is_level_zero = false;
  bool host_unified_memory = false;
  std::string_view vendor_lower{};
  std::uint32_t vendor_id = 0;
  std::string_view name_lower{};
  int min_subgroup = 0;
  int max_subgroup = 0;
  bool supports_sg8 = false;
  bool supports_sg16 = false;
  bool supports_sg32 = false;
  bool has_runtime_xmx_dpas = false;
  bool has_runtime_bf16 = false;
};

struct SyclArchInfo {
  IntelArchFamily family = IntelArchFamily::Unknown;
  int min_subgroup = 0;
  int max_subgroup = 0;
  int preferred_subgroup = 0;
  bool has_xmx_dpas = false;
  bool prefer_level_zero_direct_alloc = false;
  bool requires_dnnl_intel_vendor_filter = true;
};

constexpr bool is_intel_vendor(std::string_view vendor_lower, std::uint32_t vendor_id) noexcept {
  return vendor_id == kIntelPciVendorId || vendor_lower.find("intel") != std::string_view::npos;
}

// Filters and prioritizes Intel GPUs only (Level Zero before OpenCL); never accepts CPUs or non-Intel GPUs.
inline std::vector<std::size_t> select_intel_gpu_indices(const std::vector<SyclDeviceCandidate>& candidates) {
  std::vector<std::size_t> out;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto& c = candidates[i];
    if (c.is_gpu && is_intel_vendor(c.vendor_lower, c.vendor_id) && c.is_level_zero) {
      out.push_back(i);
    }
  }
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    const auto& c = candidates[i];
    if (c.is_gpu && is_intel_vendor(c.vendor_lower, c.vendor_id) && !c.is_level_zero) {
      out.push_back(i);
    }
  }
  return out;
}

// Classifies an Intel GPU strictly from verified vendor identity, device name, queried subgroup widths,
// and runtime XMX/matrix aspect capabilities (never inferring XMX or Battlemage from subgroup width).
constexpr SyclArchInfo classify_intel_device(const SyclDeviceProbe& probe) noexcept {
  if (!probe.is_gpu || !is_intel_vendor(probe.vendor_lower, probe.vendor_id)) {
    return SyclArchInfo{IntelArchFamily::Unknown, 0, 0, 0, false, false, false};
  }

  IntelArchFamily family = IntelArchFamily::Unknown;
  if (probe.name_lower.find("ponte") != std::string_view::npos ||
      probe.name_lower.find("data center gpu max") != std::string_view::npos) {
    family = IntelArchFamily::XeHpcPonteVecchio;
  } else if (probe.name_lower.find("battlemage") != std::string_view::npos ||
             probe.name_lower.find("b580") != std::string_view::npos ||
             probe.name_lower.find("b570") != std::string_view::npos ||
             probe.name_lower.find(" bmg") != std::string_view::npos) {
    family = IntelArchFamily::Xe2Battlemage;
  } else if (probe.name_lower.find("lunar lake") != std::string_view::npos ||
             probe.name_lower.find("140v") != std::string_view::npos ||
             probe.name_lower.find("130v") != std::string_view::npos) {
    family = IntelArchFamily::Xe2LunarLake;
  } else if (probe.name_lower.find("alchemist") != std::string_view::npos ||
             probe.name_lower.find("a770") != std::string_view::npos ||
             probe.name_lower.find("a750") != std::string_view::npos ||
             probe.name_lower.find("a580") != std::string_view::npos ||
             probe.name_lower.find("a380") != std::string_view::npos ||
             probe.name_lower.find("a310") != std::string_view::npos ||
             probe.name_lower.find("data center gpu flex") != std::string_view::npos ||
             probe.name_lower.find("dg2") != std::string_view::npos) {
    family = IntelArchFamily::XeHpgAlchemist;
  } else if (probe.name_lower.find("iris") != std::string_view::npos ||
             probe.name_lower.find("uhd") != std::string_view::npos ||
             probe.name_lower.find("xe-lp") != std::string_view::npos ||
             probe.name_lower.find("gen12") != std::string_view::npos ||
             probe.name_lower.find("meteor lake") != std::string_view::npos ||
             probe.name_lower.find("arrow lake") != std::string_view::npos) {
    family = IntelArchFamily::XeLp;
  }

  int preferred = 0;
  if (probe.supports_sg16) preferred = 16;
  else if (probe.supports_sg32) preferred = 32;
  else if (probe.supports_sg8) preferred = 8;
  else preferred = probe.min_subgroup;

  return SyclArchInfo{
      family,
      probe.min_subgroup,
      probe.max_subgroup,
      preferred,
      probe.has_runtime_xmx_dpas,
      probe.is_level_zero && !probe.host_unified_memory,
      true,
  };
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
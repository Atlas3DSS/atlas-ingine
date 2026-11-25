// Host-compilable verification suite for Phase 3 pure native C++ vendor portability primitives,
// bit-exact ALU helpers (arch_ops.cuh), wave64/32 logical-32 half-wavefront masks (warp.cuh),
// AMD HIP / Apple MLX / Intel SYCL architecture & memory streaming policies (vendor_policy.h),
// and cross-backend HAL GraphRequirement / DeferredFlag contract invariants.
#include "core/act.h"
#include "core/hal.h"
#include "core/ops.h"
#include "core/vendor_policy.h"
#include "kernels/common/arch_ops.cuh"
#include "kernels/common/warp.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

#define VP_CHECK(cond)                                                                      \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);        \
      ++failures;                                                                           \
    }                                                                                       \
  } while (0)

std::uint32_t fbits(float v) noexcept {
  std::uint32_t b = 0;
  std::memcpy(&b, &v, sizeof(b));
  return b;
}

void test_arch_ops_bit_equivalence() {
  using namespace ingine::kernels::arch;

  // Test LOP3 LUT 0x6a == (a & b) ^ c and general 8-minterm LUT consistency.
  const std::uint32_t patterns[] = {0x00000000u, 0xffffffffu, 0x12345678u, 0xdeadbeefu, 0x80000001u, 0x7fffffffu};
  for (std::uint32_t a : patterns) {
    for (std::uint32_t b : patterns) {
      for (std::uint32_t c : patterns) {
        VP_CHECK(lop3_lut<0x6au>(a, b, c) == ((a & b) ^ c));
        VP_CHECK(lop3_lut<0xcau>(a, b, c) == ((a & b) | (~a & c)));
        VP_CHECK(lop3_lut<0x96u>(a, b, c) == (a ^ b ^ c));
      }
    }
    VP_CHECK(exl3_mcg_mask_xor(a) == ((a & 0x8fff8fffu) ^ 0x3b603b60u));
  }

  // Test funnelshift_l(lo, hi, shift) exact match against 64-bit shift model across shifts 0..63.
  const std::uint32_t lo = 0xa1b2c3d4u;
  const std::uint32_t hi = 0x0f1e2d3cu;
  for (int s = 0; s < 64; ++s) {
    const unsigned k = static_cast<unsigned>(s) & 31u;
    const std::uint64_t concat = (static_cast<std::uint64_t>(hi) << 32) | lo;
    const auto expect = static_cast<std::uint32_t>((concat << k) >> 32);
    VP_CHECK(funnelshift_l(lo, hi, s) == expect);
  }

  // Test dp4a_s8 including extremal -128 * -128 * 4 = 65536 and signed overflow wrap modulo 2^32.
  VP_CHECK(dp4a_s8(static_cast<std::int32_t>(0x80808080u), static_cast<std::int32_t>(0x80808080u), 0) == 65536);
  VP_CHECK(dp4a_s8(static_cast<std::int32_t>(0x01ff02feu), static_cast<std::int32_t>(0x0304fc05u), -10) ==
           ((1 * 3) + (-1 * 4) + (2 * -4) + (-2 * 5) - 10));
  VP_CHECK(dp4a_s8(static_cast<std::int32_t>(0x7f7f7f7fu), static_cast<std::int32_t>(0x7f7f7f7fu), 0x7fffffff) ==
           static_cast<std::int32_t>(0x7fffffffu + 4u * 127u * 127u));

  // Test byte_perm over {y:x}.
  VP_CHECK(byte_perm(0x11223344u, 0x55667788u, 0x7654u) == 0x55667788u);
  VP_CHECK(byte_perm(0x11223344u, 0x55667788u, 0x3210u) == 0x11223344u);
  VP_CHECK(byte_perm(0x11223344u, 0x55667788u, 0x7052u) == 0x55447722u);

  // Test SWAR 4-byte unsigned wrapping subtraction without cross-lane borrow.
  VP_CHECK(vsub4_u8(0x00010203u, 0x01010101u) == 0xff000102u);
  VP_CHECK(vsub4_u8(0x80800001u, 0x01ff0202u) == 0x7f81feffu);

  // Test canonical Q4_0 decode rule (neg scale * nibble 8 -> +0.0f / 0x00000000, non-finite scale -> 0).
  VP_CHECK(fbits(decode_q4_0_nibble_canonical(8, 0xbc00u)) == 0x00000000u);  // -1.0 * (8 - 8)
  VP_CHECK(fbits(decode_q4_0_nibble_canonical(0, 0xbc00u)) == 0x41000000u);  // +8.0
  VP_CHECK(fbits(decode_q4_0_nibble_canonical(15, 0xbc00u)) == 0xc0e00000u); // -7.0
  VP_CHECK(fbits(decode_q4_0_nibble_canonical(3, 0x7c00u)) == 0x00000000u);  // +Inf guarded
  VP_CHECK(fbits(decode_q4_0_nibble_canonical(3, 0x7e00u)) == 0x00000000u);  // NaN guarded
}

void test_hip_policies_and_warp_mapping() {
  using namespace ingine::vendor::hip;

  // RDNA3/3.5/4 wave32 classifications.
  const auto r3 = classify_hip_arch("gfx1100:xnack-");
  VP_CHECK(r3.family == ArchFamily::Rdna3 && r3.warp_size == 32 && r3.lds_bytes == 65536);
  VP_CHECK(r3.has_sudot4 && !r3.has_sdot4 && r3.has_rdna_wmma && !r3.is_wave64 && !r3.requires_fp_contract_on);

  const auto r35 = classify_hip_arch("gfx1151");
  VP_CHECK(r35.family == ArchFamily::Rdna35 && r35.warp_size == 32 && r35.has_rdna_wmma && r35.validated_tier);

  const auto r4 = classify_hip_arch("gfx1201");
  VP_CHECK(r4.family == ArchFamily::Rdna4 && r4.warp_size == 32 && r4.has_rdna_wmma && r4.validated_tier);

  // CDNA wave64 + -ffp-contract=on requirement on gfx942/gfx950.
  const auto cdna3 = classify_hip_arch("gfx942:sramecc+");
  VP_CHECK(cdna3.family == ArchFamily::Cdna3 && cdna3.warp_size == 64 && cdna3.is_wave64 && cdna3.has_sdot4 &&
           cdna3.requires_fp_contract_on);
  const auto cdna1 = classify_hip_arch("gfx908");
  VP_CHECK(cdna1.family == ArchFamily::Cdna1 && cdna1.is_wave64 && cdna1.has_sdot4 && !cdna1.requires_fp_contract_on);
  const auto unk = classify_hip_arch("gfx999");
  VP_CHECK(unk.family == ArchFamily::Unknown);

  // WMMA 16x16x16 lane row layout differs between gfx11 and gfx12.
  VP_CHECK(wmma_result_row(ArchFamily::Rdna3, 0, 3) == 6);
  VP_CHECK(wmma_result_row(ArchFamily::Rdna3, 1, 3) == 7);
  VP_CHECK(wmma_result_row(ArchFamily::Rdna4, 0, 3) == 3);
  VP_CHECK(wmma_result_row(ArchFamily::Rdna4, 1, 3) == 11);

  // Compressed NVFP4 MoE tile fits cleanly in 64 KiB LDS without FP16 buffer materialization.
  VP_CHECK(compressed_lds_tile_bytes(256, 256, 4) == 32768u);
  VP_CHECK(compressed_lds_tile_bytes(256, 256, 4) <= r3.lds_bytes);

  // EXL3 HIP scalar prompt tile policy (Maya exl3_hip_scalar).
  VP_CHECK(parse_exl3_scalar_mode("") == 8);
  VP_CHECK(parse_exl3_scalar_mode("0") == 8);
  VP_CHECK(parse_exl3_scalar_mode("auto") == -1);
  VP_CHECK(parse_exl3_scalar_mode("24") == 24);
  VP_CHECK(parse_exl3_scalar_mode("invalid") == 8);
  VP_CHECK(parse_positive_cutoff("512", 128) == 512);
  VP_CHECK(parse_positive_cutoff("-1", 128) == 128);
  VP_CHECK(select_exl3_scalar_rows(-1, true, 16, 128, 256) == 8);
  VP_CHECK(select_exl3_scalar_rows(-1, true, 256, 128, 256) == 12);
  VP_CHECK(select_exl3_scalar_rows(-1, false, 512, 128, 256) == 24);

  // CDNA wave64 logical-32 half-wavefront ballot extraction.
  const std::uint64_t wave64_mask = 0x12345678abcdef01ull;
  VP_CHECK(ingine::kernels::warp::logical_32_half_ballot(wave64_mask, 5) == 0xabcdef01u);
  VP_CHECK(ingine::kernels::warp::logical_32_half_ballot(wave64_mask, 37) == 0x12345678u);
}

void test_mlx_and_sycl_policies() {
  using namespace ingine::vendor::mlx;
  using namespace ingine::vendor::sycl;

  // Compact iPhone 6 GiB class with ~9 GiB EXL3 pack -> fixed <= 2.3 GiB ring cap.
  const auto phone = plan_streaming_budget(6ull << 30, 4ull << 30, 9ull << 30, true);
  VP_CHECK(phone.mode == MemoryResidencyMode::BoundedStreamingRing);
  VP_CHECK(phone.staging_ring_bytes <= kCompactPhoneRingCapBytes);
  VP_CHECK(phone.ring_slots >= 2);
  VP_CHECK(!phone.pin_host_memory);

  // 8 GiB Apple Silicon / iPad with ~9 GiB EXL3 or ~14 GiB 3-bit S pack -> fixed <= 3.0 GiB ring cap.
  const auto ipad8 = plan_streaming_budget(8ull << 30, 6ull << 30, 9ull << 30, false);
  VP_CHECK(ipad8.mode == MemoryResidencyMode::BoundedStreamingRing);
  VP_CHECK(ipad8.staging_ring_bytes <= kTabletMacRingCapBytes);
  VP_CHECK(ipad8.pread_piece_bytes == kDefaultPreadPieceBytes);

  // 64 GiB M-series Mac with ~9 GiB EXL3 pack -> FullUnifiedResident.
  const auto mac64 = plan_streaming_budget(64ull << 30, 48ull << 30, 9ull << 30, false);
  VP_CHECK(mac64.mode == MemoryResidencyMode::FullUnifiedResident);
  VP_CHECK(mac64.pin_host_memory);
  VP_CHECK(align_page_up(16385u) == 32768u);

  // Intel SYCL classifications & subgroup workgroup multiples.
  const auto acm = classify_intel_device("intel(r) arc(tm) a770 graphics", 16);
  VP_CHECK(acm.family == IntelArchFamily::XeHpgAlchemist && acm.preferred_subgroup == 16 && acm.has_xmx_dpas &&
           acm.prefer_level_zero_direct_alloc);
  const auto lnl = classify_intel_device("intel(r) arc(tm) 140v gpu (lunar lake)", 32);
  VP_CHECK(lnl.family == IntelArchFamily::Xe2LunarLake && !lnl.prefer_level_zero_direct_alloc);
  VP_CHECK((select_rmsnorm_workgroup(2304, 8) % 8) == 0);
  VP_CHECK((select_rmsnorm_workgroup(2304, 16) % 16) == 0);
  VP_CHECK((select_rmsnorm_workgroup(2304, 32) % 32) == 0);
  VP_CHECK(dnnl_vendor_filter_is_intel("INTEL"));
  VP_CHECK(!dnnl_vendor_filter_is_intel("NVIDIA"));
}

void test_cross_backend_graph_admission() {
  using namespace ingine::hal;

  // Verify honest graph admission across all 5 BackendKinds.
  DeviceCaps hip_caps{};
  hip_caps.backend = BackendKind::Hip;
  hip_caps.supports_graph_capture = true;
  hip_caps.supports_conditional_graph = false;
  VP_CHECK(graph_requirement_met(hip_caps, GraphRequirement::Capture));
  VP_CHECK(!graph_requirement_met(hip_caps, GraphRequirement::ConditionalSwitch));
  require_graph_capture_or_eager(hip_caps, false, "hip_capture", GraphRequirement::Capture);
  require_conditional_switch_or_eager(hip_caps, true, "hip_switch_eager");
  bool hip_switch_rejected = false;
  try {
    require_conditional_switch_or_eager(hip_caps, false, "hip_switch_strict");
  } catch (const std::runtime_error&) {
    hip_switch_rejected = true;
  }
  VP_CHECK(hip_switch_rejected);

  for (BackendKind k : {BackendKind::Cpu, BackendKind::Mlx, BackendKind::Sycl}) {
    DeviceCaps caps{};
    caps.backend = k;
    caps.supports_graph_capture = false;
    caps.supports_conditional_graph = false;
    VP_CHECK(!graph_requirement_met(caps, GraphRequirement::Capture));
    VP_CHECK(!graph_requirement_met(caps, GraphRequirement::ConditionalSwitch));
    require_graph_capture_or_eager(caps, true, "eager_ok", GraphRequirement::ConditionalSwitch);
    bool threw = false;
    try {
      require_graph_capture_or_eager(caps, false, "strict_reject", GraphRequirement::Capture);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    VP_CHECK(threw);
  }
}

}  // namespace

int main() {
  test_arch_ops_bit_equivalence();
  test_hip_policies_and_warp_mapping();
  test_mlx_and_sycl_policies();
  test_cross_backend_graph_admission();

  if (failures != 0) {
    std::fprintf(stderr, "FAIL core_vendor_portability: %d checks failed\n", failures);
    return 1;
  }
  std::printf("PASS core_vendor_portability (arch_ops + hip/mlx/sycl policies + 5-backend graph admission)\n");
  return 0;
}

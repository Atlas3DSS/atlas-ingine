// Host-compilable verification suite for Phase 3 pure native C++ vendor portability primitives,
// bit-exact ALU helpers (arch_ops.cuh), wave64/32 logical-32 half-wavefront masks (warp.cuh),
// AMD HIP / Apple MLX / Intel SYCL architecture & memory streaming policies (vendor_policy.h),
// and cross-backend HAL GraphRequirement / DeferredFlag contract invariants.
#include "core/act.h"
#include "core/backend_contract.h"
#include "core/gpu_compat.h"
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
  VP_CHECK(r3.has_sudot4 && !r3.has_sdot4 && r3.has_rdna_wmma && !r3.is_wave64 && !r3.requires_fp_contract_on &&
           !r3.is_unified_apu);

  const auto r35 = classify_hip_arch("gfx1151");
  VP_CHECK(r35.family == ArchFamily::Rdna35 && r35.warp_size == 32 && r35.has_rdna_wmma && r35.validated_tier &&
           r35.is_unified_apu);

  const auto r4 = classify_hip_arch("gfx1201");
  VP_CHECK(r4.family == ArchFamily::Rdna4 && r4.warp_size == 32 && r4.has_rdna_wmma && r4.validated_tier &&
           !r4.is_unified_apu);

  // CDNA wave64 + -ffp-contract=on requirement on gfx942/gfx950.
  const auto cdna3 = classify_hip_arch("gfx942:sramecc+");
  VP_CHECK(cdna3.family == ArchFamily::Cdna3 && cdna3.warp_size == 64 && cdna3.is_wave64 && cdna3.has_sdot4 &&
           cdna3.requires_fp_contract_on && !cdna3.is_unified_apu);

  // Strix Halo (gfx1151) + Qwen3.8-Flash-Next owner-pinned target floor gate (strict inequalities).
  constexpr StrixHaloTargetFloor strix_floor{};
  const StrixHaloMetrics pass_metrics{105.0, 72.5, 125.0, 90.0, 68.5, 2050.0, 1920.0, 1825.0, 1.95, 1.05, 17.5};
  VP_CHECK(strix_floor.meets_all(pass_metrics));
  const StrixHaloMetrics boundary_metrics{100.0, 70.0, 121.0, 88.0, 67.0, 1983.0, 1876.0, 1796.0, 2.1, 1.2, 19.0};
  VP_CHECK(!strix_floor.meets_all(boundary_metrics));
  auto fail_edit = pass_metrics;
  fail_edit.decode_edit_tok_s = 121.0;
  VP_CHECK(!strix_floor.meets_all(fail_edit));
  auto fail_prefill_131k = pass_metrics;
  fail_prefill_131k.prefill_131k_tok_s = 1796.0;
  VP_CHECK(!strix_floor.meets_all(fail_prefill_131k));
  auto fail_ttft = pass_metrics;
  fail_ttft.agent_ttft_30k_96k_s = 1.2;
  VP_CHECK(!strix_floor.meets_all(fail_ttft));
  auto fail_cold = pass_metrics;
  fail_cold.cold_34k_s = 19.0;
  VP_CHECK(!strix_floor.meets_all(fail_cold));
  auto fail_zero_turn = pass_metrics;
  fail_zero_turn.agent_turn_30k_96k_s = 0.0;
  VP_CHECK(!strix_floor.meets_all(fail_zero_turn));
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

  // Unified-APU zero-copy INGPACK1 policy (default on unified APU, off on discrete, env override).
  VP_CHECK(!should_use_zero_copy_weight_pack(true, false));
  VP_CHECK(!should_use_zero_copy_weight_pack(true, false, "1"));
  VP_CHECK(should_use_zero_copy_weight_pack(true, true));
  VP_CHECK(!should_use_zero_copy_weight_pack(false, true));
  VP_CHECK(!should_use_zero_copy_weight_pack(true, true, "0"));
  VP_CHECK(!should_use_zero_copy_weight_pack(true, true, "false"));
  VP_CHECK(!should_use_zero_copy_weight_pack(true, true, "off"));
  VP_CHECK(should_use_zero_copy_weight_pack(false, true, "1"));
  VP_CHECK(should_use_zero_copy_weight_pack(false, true, "true"));
  VP_CHECK(should_use_zero_copy_weight_pack(false, true, "on"));

  // Portable OCP E4M3FN host conversion invariants (exact midpoints, saturation, NaN, full 254 finite roundtrip).
  using namespace ingine::gpu;
  VP_CHECK(float_to_e4m3fn_sat_portable(0.0f) == 0x00u);
  VP_CHECK(float_to_e4m3fn_sat_portable(-0.0f) == 0x80u);
  VP_CHECK(float_to_e4m3fn_sat_portable(448.0f) == 0x7eu);
  VP_CHECK(float_to_e4m3fn_sat_portable(-448.0f) == 0xfeu);
  VP_CHECK(float_to_e4m3fn_sat_portable(1000.0f) == 0x7eu);
  VP_CHECK(float_to_e4m3fn_sat_portable(-1000.0f) == 0xfeu);
  VP_CHECK(float_to_e4m3fn_sat_portable(INFINITY) == 0x7eu);
  VP_CHECK(float_to_e4m3fn_sat_portable(-INFINITY) == 0xfeu);
  VP_CHECK(float_to_e4m3fn_sat_portable(NAN) == 0x7fu);
  VP_CHECK(float_to_e4m3fn_sat_portable(-NAN) == 0x7fu);
  // Subnormal round-to-even midpoints: 0.5*2^-9 -> 0 (even), 1.5*2^-9 -> 2 (even), 7.5*2^-9 -> 8 (min normal 0x08).
  VP_CHECK(float_to_e4m3fn_sat_portable(0.5f / 512.0f) == 0x00u);
  VP_CHECK(float_to_e4m3fn_sat_portable(1.5f / 512.0f) == 0x02u);
  VP_CHECK(float_to_e4m3fn_sat_portable(7.5f / 512.0f) == 0x08u);
  VP_CHECK(e4m3fn_to_f16_bits_portable(0x7fu) == 0x7fffu);
  VP_CHECK(e4m3fn_to_f16_bits_portable(0xffu) == 0x7fffu);
  for (int code = 0; code < 256; ++code) {
    const std::uint8_t c = static_cast<std::uint8_t>(code);
    const float fv = e4m3fn_to_float_portable(c);
    if ((code & 0x7f) == 0x7f) {
      VP_CHECK(std::isnan(fv));
    } else {
      VP_CHECK(float_to_e4m3fn_sat_portable(fv) == c);
    }
  }
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

  // D2D async blit overlap safety (distinct buffers, disjoint slices, overlapping slices).
  VP_CHECK(can_async_blit_d2d(false, 0u, 0u, 4096u));
  VP_CHECK(can_async_blit_d2d(true, 0u, 4096u, 4096u));
  VP_CHECK(can_async_blit_d2d(true, 4096u, 0u, 4096u));
  VP_CHECK(!can_async_blit_d2d(true, 0u, 2048u, 4096u));
  VP_CHECK(!can_async_blit_d2d(true, 2048u, 0u, 4096u));
  VP_CHECK(can_async_blit_d2d(true, 128u, 128u, 0u));

  // Metal registered-range lookup: distinguish unregistered host pointers, valid offset subspans,
  // and out-of-bounds spans that cross past either end of a registered MTLBuffer.
  const std::vector<RegisteredRange> mtl_spans = {
      {0x100000u, 16384u},
      {0x200000u, 32768u},
  };
  const auto unreg_lo = classify_registered_range(mtl_spans, 0x050000u, 4096u);
  VP_CHECK(unreg_lo.status == BufferRangeMatch::Unregistered);
  const auto unreg_adj = classify_registered_range(mtl_spans, 0x104000u, 16u);  // right after first buffer end
  VP_CHECK(unreg_adj.status == BufferRangeMatch::Unregistered);
  const auto valid_base = classify_registered_range(mtl_spans, 0x100000u, 16384u);
  VP_CHECK(valid_base.status == BufferRangeMatch::Valid && valid_base.index == 0u && valid_base.offset == 0u);
  const auto valid_sub = classify_registered_range(mtl_spans, 0x201000u, 4096u);
  VP_CHECK(valid_sub.status == BufferRangeMatch::Valid && valid_sub.index == 1u && valid_sub.offset == 0x1000u);
  const auto valid_tail = classify_registered_range(mtl_spans, 0x207f00u, 256u);
  VP_CHECK(valid_tail.status == BufferRangeMatch::Valid && valid_tail.index == 1u && valid_tail.offset == 0x7f00u);
  const auto oob_end = classify_registered_range(mtl_spans, 0x207f00u, 257u);  // crosses end by 1 byte
  VP_CHECK(oob_end.status == BufferRangeMatch::OutOfBounds && oob_end.index == 1u && oob_end.offset == 0x7f00u);
  const auto oob_whole = classify_registered_range(mtl_spans, 0x100000u, 16385u);
  VP_CHECK(oob_whole.status == BufferRangeMatch::OutOfBounds && oob_whole.index == 0u && oob_whole.offset == 0u);
  const auto oob_front = classify_registered_range(mtl_spans, 0x0fff00u, 512u);  // starts before buffer, crosses in
  VP_CHECK(oob_front.status == BufferRangeMatch::OutOfBounds && oob_front.index == 0u);

  // Intel SYCL GPU enumeration strictly rejects CPUs and non-Intel GPUs and prefers Level Zero over OpenCL.
  const std::vector<SyclDeviceCandidate> cands = {
      {false, false, "intel(r) corporation", 0x8086u},      // idx 0: Intel CPU -> reject
      {true, false, "nvidia corporation", 0x10deu},         // idx 1: NVIDIA OpenCL GPU -> reject
      {true, false, "advanced micro devices", 0x1002u},     // idx 2: AMD GPU -> reject
      {true, false, "intel(r) corporation", 0x8086u},       // idx 3: Intel OpenCL GPU
      {true, true, "intel(r) corporation", 0x8086u},        // idx 4: Intel Level Zero GPU (preferred)
  };
  const auto chosen = select_intel_gpu_indices(cands);
  VP_CHECK(chosen.size() == 2 && chosen[0] == 4u && chosen[1] == 3u);
  const std::vector<SyclDeviceCandidate> only_cpu_and_nvidia = {cands[0], cands[1], cands[2]};
  VP_CHECK(select_intel_gpu_indices(only_cpu_and_nvidia).empty());

  // Non-Intel GPU with subgroup 32 must NEVER classify as Battlemage or advertise XMX/DPAS.
  SyclDeviceProbe nvidia_probe;
  nvidia_probe.is_gpu = true;
  nvidia_probe.vendor_lower = "nvidia corporation";
  nvidia_probe.vendor_id = 0x10deu;
  nvidia_probe.name_lower = "nvidia geforce rtx 4090";
  nvidia_probe.min_subgroup = 32;
  nvidia_probe.max_subgroup = 32;
  nvidia_probe.supports_sg32 = true;
  nvidia_probe.has_runtime_xmx_dpas = true;
  const auto bad_vendor = classify_intel_device(nvidia_probe);
  VP_CHECK(bad_vendor.family == IntelArchFamily::Unknown && !bad_vendor.has_xmx_dpas &&
           bad_vendor.preferred_subgroup == 0 && !bad_vendor.prefer_level_zero_direct_alloc);

  // CPU device (even Intel CPU) must NEVER classify as GPU or XMX.
  SyclDeviceProbe cpu_probe;
  cpu_probe.is_gpu = false;
  cpu_probe.vendor_lower = "intel(r) corporation";
  cpu_probe.vendor_id = 0x8086u;
  cpu_probe.name_lower = "intel(r) xeon(r) gold";
  cpu_probe.min_subgroup = 8;
  cpu_probe.max_subgroup = 64;
  cpu_probe.supports_sg16 = true;
  cpu_probe.supports_sg32 = true;
  const auto cpu_cls = classify_intel_device(cpu_probe);
  VP_CHECK(cpu_cls.family == IntelArchFamily::Unknown && !cpu_cls.has_xmx_dpas && cpu_cls.preferred_subgroup == 0);

  // Intel GPU without runtime XMX aspect (even if name matches) must not advertise XMX.
  SyclDeviceProbe acm_no_xmx;
  acm_no_xmx.is_gpu = true;
  acm_no_xmx.is_level_zero = true;
  acm_no_xmx.vendor_lower = "intel(r) corporation";
  acm_no_xmx.vendor_id = 0x8086u;
  acm_no_xmx.name_lower = "intel(r) arc(tm) a770 graphics";
  acm_no_xmx.min_subgroup = 8;
  acm_no_xmx.max_subgroup = 32;
  acm_no_xmx.supports_sg8 = true;
  acm_no_xmx.supports_sg16 = true;
  acm_no_xmx.supports_sg32 = true;
  acm_no_xmx.has_runtime_xmx_dpas = false;
  const auto acm_nox = classify_intel_device(acm_no_xmx);
  VP_CHECK(acm_nox.family == IntelArchFamily::XeHpgAlchemist && acm_nox.preferred_subgroup == 16 &&
           !acm_nox.has_xmx_dpas && acm_nox.prefer_level_zero_direct_alloc);

  // Intel Arc A770 + Lunar Lake with confirmed runtime XMX.
  acm_no_xmx.has_runtime_xmx_dpas = true;
  const auto acm = classify_intel_device(acm_no_xmx);
  VP_CHECK(acm.family == IntelArchFamily::XeHpgAlchemist && acm.preferred_subgroup == 16 && acm.has_xmx_dpas &&
           acm.prefer_level_zero_direct_alloc);

  SyclDeviceProbe lnl_probe;
  lnl_probe.is_gpu = true;
  lnl_probe.is_level_zero = true;
  lnl_probe.host_unified_memory = true;
  lnl_probe.vendor_lower = "intel(r) corporation";
  lnl_probe.vendor_id = 0x8086u;
  lnl_probe.name_lower = "intel(r) arc(tm) 140v gpu (lunar lake)";
  lnl_probe.min_subgroup = 16;
  lnl_probe.max_subgroup = 32;
  lnl_probe.supports_sg16 = true;
  lnl_probe.supports_sg32 = true;
  lnl_probe.has_runtime_xmx_dpas = true;
  const auto lnl = classify_intel_device(lnl_probe);
  VP_CHECK(lnl.family == IntelArchFamily::Xe2LunarLake && lnl.preferred_subgroup == 16 && lnl.has_xmx_dpas &&
           !lnl.prefer_level_zero_direct_alloc);

  VP_CHECK((select_rmsnorm_workgroup(2304, 8) % 8) == 0);
  VP_CHECK((select_rmsnorm_workgroup(2304, 16) % 16) == 0);
  VP_CHECK((select_rmsnorm_workgroup(2304, 32) % 32) == 0);
  VP_CHECK(dnnl_vendor_filter_is_intel("INTEL"));
  VP_CHECK(!dnnl_vendor_filter_is_intel("NVIDIA"));
}

void test_per_backend_contract_matrix() {
  using namespace ingine::hal;
  using namespace ingine::test_contract;

  DeviceCaps cpu{};
  cpu.backend = BackendKind::Cpu;
  cpu.primary_space = MemorySpace::Host;
  cpu.subgroup = {1, 1, 1};
  VP_CHECK(check_backend_caps_contract(BackendKind::Cpu, cpu));
  VP_CHECK(!check_backend_caps_contract(BackendKind::Hip, cpu));
  VP_CHECK(!expect_embed_q4_fallback(BackendKind::Cpu, false, 2304, 2368, ingine::ops::ActRounding::Fp16RnSat));
  VP_CHECK(expect_embed_q4_fallback(BackendKind::Cpu, true, 2304, 2304, ingine::ops::ActRounding::Bf16Rn));
  VP_CHECK(!backend_rejects_strided_without_fallback(BackendKind::Cpu));

  DeviceCaps cuda{};
  cuda.backend = BackendKind::Cuda;
  cuda.primary_space = MemorySpace::Device;
  cuda.subgroup = {32, 32, 32};
  cuda.supports_graph_capture = true;
  VP_CHECK(check_backend_caps_contract(BackendKind::Cuda, cuda));
  VP_CHECK(expect_embed_q4_fallback(BackendKind::Cuda, false, 2304, 2368, build_default_act_rounding()));
  VP_CHECK(expect_embed_q4_fallback(
      BackendKind::Cuda, false, 2304, 2304,
      ingine::kActIsFP16 ? ingine::ops::ActRounding::Bf16Rn : ingine::ops::ActRounding::Fp16RnSat));
  VP_CHECK(backend_rejects_strided_without_fallback(BackendKind::Cuda));

  DeviceCaps hip_w32{};
  hip_w32.backend = BackendKind::Hip;
  hip_w32.primary_space = MemorySpace::Device;
  hip_w32.subgroup = {32, 32, 32};
  hip_w32.supports_graph_capture = true;
  VP_CHECK(check_backend_caps_contract(BackendKind::Hip, hip_w32));
  DeviceCaps hip_w64 = hip_w32;
  hip_w64.primary_space = MemorySpace::Unified;
  hip_w64.subgroup = {64, 64, 64};
  VP_CHECK(check_backend_caps_contract(BackendKind::Hip, hip_w64));
  VP_CHECK(!check_backend_caps_contract(BackendKind::Hip, cpu));
  VP_CHECK(!expect_embed_q4_fallback(BackendKind::Hip, false, 2304, 2304, ingine::ops::ActRounding::Fp16RnSat));
  VP_CHECK(expect_embed_q4_fallback(BackendKind::Hip, false, 2304, 2368, ingine::ops::ActRounding::Bf16Rn));
  VP_CHECK(backend_rejects_strided_without_fallback(BackendKind::Hip));
  VP_CHECK(backend_has_native_prevalidated_guard(BackendKind::Hip));

  DeviceCaps mlx{};
  mlx.backend = BackendKind::Mlx;
  mlx.primary_space = MemorySpace::Unified;
  mlx.subgroup = {32, 32, 32};
  VP_CHECK(check_backend_caps_contract(BackendKind::Mlx, mlx));
  VP_CHECK(!check_backend_caps_contract(BackendKind::Mlx, cpu));
  VP_CHECK(expect_embed_q4_fallback(BackendKind::Mlx, false, 2304, 2368, ingine::ops::ActRounding::Bf16Rn));
  VP_CHECK(!expect_embed_q4_fallback(BackendKind::Mlx, false, 2304, 2304, ingine::ops::ActRounding::Fp16RnSat));
  VP_CHECK(backend_rejects_strided_without_fallback(BackendKind::Mlx));
  VP_CHECK(backend_has_native_prevalidated_guard(BackendKind::Mlx));

  DeviceCaps sycl_caps{};
  sycl_caps.backend = BackendKind::Sycl;
  sycl_caps.primary_space = MemorySpace::Device;
  sycl_caps.subgroup = {8, 32, 16};
  VP_CHECK(check_backend_caps_contract(BackendKind::Sycl, sycl_caps));
  VP_CHECK(!check_backend_caps_contract(BackendKind::Sycl, cpu));
  VP_CHECK(expect_embed_q4_fallback(BackendKind::Sycl, false, 2304, 2368, ingine::ops::ActRounding::Bf16Rn));
  VP_CHECK(!expect_embed_q4_fallback(BackendKind::Sycl, false, 2304, 2304, ingine::ops::ActRounding::Fp16RnSat));
  VP_CHECK(backend_rejects_strided_without_fallback(BackendKind::Sycl));
  VP_CHECK(backend_has_native_prevalidated_guard(BackendKind::Sycl));
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
  test_per_backend_contract_matrix();
  test_cross_backend_graph_admission();

  if (failures != 0) {
    std::fprintf(stderr, "FAIL core_vendor_portability: %d checks failed\n", failures);
    return 1;
  }
  std::printf("PASS core_vendor_portability (arch_ops + hip/mlx/sycl policies + 5-backend graph admission)\n");
  return 0;
}

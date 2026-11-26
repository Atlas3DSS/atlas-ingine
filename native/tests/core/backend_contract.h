// Shared pure C++20 per-backend contract expectations for HAL capabilities, Q4_0 embed
// dispatch fallback, PreValidated kernel guards, and no-fallback descriptor rejection.
// CUDA-free so both host portability suites and backend parity binaries use one definition.
#pragma once

#include "core/act.h"
#include "core/hal.h"
#include "core/ops.h"

#include <cmath>

namespace ingine::test_contract {

constexpr ops::ActRounding build_default_act_rounding() noexcept {
  return ingine::kActIsFP16 ? ops::ActRounding::Fp16RnSat : ops::ActRounding::Bf16Rn;
}

constexpr hal::BackendKind expected_compiled_backend() noexcept {
#if INGINE_BACKEND_CUDA
  return hal::BackendKind::Cuda;
#elif defined(INGINE_BACKEND_HIP) && INGINE_BACKEND_HIP
  return hal::BackendKind::Hip;
#elif defined(INGINE_BACKEND_MLX) && INGINE_BACKEND_MLX
  return hal::BackendKind::Mlx;
#elif defined(INGINE_BACKEND_SYCL) && INGINE_BACKEND_SYCL
  return hal::BackendKind::Sycl;
#else
  return hal::BackendKind::Cpu;
#endif
}

inline bool check_backend_caps_contract(hal::BackendKind active, const hal::DeviceCaps& caps) noexcept {
  if (caps.backend != active) return false;
  switch (caps.backend) {
    case hal::BackendKind::Cpu:
      return caps.primary_space == hal::MemorySpace::Host &&
             caps.subgroup.min_width == 1 && caps.subgroup.max_width == 1 &&
             caps.subgroup.preferred_width == 1 &&
             !caps.supports_graph_capture && !caps.supports_conditional_graph;
    case hal::BackendKind::Cuda:
      return caps.primary_space == hal::MemorySpace::Device &&
             caps.subgroup.preferred_width == 32 && caps.subgroup.min_width == 32 &&
             caps.subgroup.max_width == 32 && caps.supports_graph_capture;
    case hal::BackendKind::Hip:
      return (caps.primary_space == hal::MemorySpace::Device ||
              caps.primary_space == hal::MemorySpace::Unified) &&
             (caps.subgroup.preferred_width == 32 || caps.subgroup.preferred_width == 64) &&
             caps.subgroup.min_width == caps.subgroup.preferred_width &&
             caps.subgroup.max_width == caps.subgroup.preferred_width &&
             caps.supports_graph_capture && !caps.supports_conditional_graph;
    case hal::BackendKind::Mlx:
      return caps.primary_space == hal::MemorySpace::Unified &&
             caps.subgroup.preferred_width == 32 && caps.subgroup.min_width == 32 &&
             caps.subgroup.max_width == 32 &&
             !caps.supports_graph_capture && !caps.supports_conditional_graph;
    case hal::BackendKind::Sycl:
      return caps.primary_space == hal::MemorySpace::Device &&
             caps.subgroup.min_width >= 8 &&
             caps.subgroup.min_width <= caps.subgroup.preferred_width &&
             caps.subgroup.preferred_width <= caps.subgroup.max_width &&
             (caps.subgroup.preferred_width == 8 || caps.subgroup.preferred_width == 16 ||
              caps.subgroup.preferred_width == 32) &&
             !caps.supports_graph_capture && !caps.supports_conditional_graph;
  }
  return false;
}

inline bool expect_embed_q4_fallback(hal::BackendKind backend, bool force_fallback, int hidden_size,
                                     int out_stride, ops::ActRounding rounding, float eps = 1e-6f) noexcept {
  if (force_fallback) return true;
  if (backend == hal::BackendKind::Cpu) return false;
  if (hidden_size != 2304 || out_stride != 2304 || std::fabs(eps - 1e-6f) != 0.f) return true;
  if (backend == hal::BackendKind::Cuda && rounding != build_default_act_rounding()) return true;
  return false;
}

constexpr bool backend_rejects_strided_without_fallback(hal::BackendKind backend) noexcept {
  return backend != hal::BackendKind::Cpu;
}

constexpr bool backend_has_native_prevalidated_guard(hal::BackendKind backend) noexcept {
  return backend != hal::BackendKind::Cpu;
}

}  // namespace ingine::test_contract

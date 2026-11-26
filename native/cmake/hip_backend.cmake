# Pure native AMD ROCm/HIP backend configuration for atlas-ingine (INGINE_BACKEND=HIP).
# Incorporates architecture classification and packed-FMA determinism rules studied from
# Strata / Project Maya (MIT) and ggml-hip (MIT). Never defines __CUDA_ARCH__.

if(NOT DEFINED INGINE_HIP_ARCHITECTURES OR INGINE_HIP_ARCHITECTURES STREQUAL "")
  if(DEFINED CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(INGINE_HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
  elseif(DEFINED ENV{GPU_TARGETS} AND NOT "$ENV{GPU_TARGETS}" STREQUAL "")
    set(INGINE_HIP_ARCHITECTURES "$ENV{GPU_TARGETS}")
  else()
    set(INGINE_HIP_ARCHITECTURES "gfx1100;gfx1101;gfx1151;gfx1200;gfx1201;gfx942")
  endif()
endif()

set(_ingine_hip_validated gfx1100 gfx1101 gfx1151 gfx1200 gfx1201)
set(_ingine_hip_unvalidated gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1034 gfx1102 gfx1103 gfx1150)
set(_ingine_hip_wave64 gfx906 gfx908 gfx90a gfx942 gfx950)

string(REPLACE " " ";" _ingine_hip_arch_norm "${INGINE_HIP_ARCHITECTURES}")
list(FILTER _ingine_hip_arch_norm EXCLUDE REGEX "^$")
list(REMOVE_DUPLICATES _ingine_hip_arch_norm)

set(INGINE_HIP_ARCH_LIST "")
set(INGINE_HIP_WAVE64 OFF)
set(INGINE_HIP_FP_CONTRACT_ON OFF)
set(INGINE_HIP_HAS_WMMA OFF)

foreach(_arch IN LISTS _ingine_hip_arch_norm)
  string(REGEX REPLACE ":.*$" "" _base "${_arch}")
  if(_base IN_LIST _ingine_hip_validated)
  elseif(_base IN_LIST _ingine_hip_unvalidated)
    message(WARNING "ingine HIP: architecture '${_base}' builds natively but is unvalidated on hardware")
  elseif(_base IN_LIST _ingine_hip_wave64)
    message(STATUS "ingine HIP: '${_base}' uses CDNA wave64 logical-32 half-wavefront mapping")
    set(INGINE_HIP_WAVE64 ON)
    if(_base STREQUAL "gfx942" OR _base STREQUAL "gfx950")
      # Packed FP32 instructions (v_pk_fma_f32) with clang's default -ffp-contract=fast can break
      # multi-activation vs single-activation bit identity; keep contraction source-scoped.
      set(INGINE_HIP_FP_CONTRACT_ON ON)
    endif()
  else()
    message(FATAL_ERROR
      "Unsupported INGINE_HIP_ARCHITECTURES entry '${_base}' in '${INGINE_HIP_ARCHITECTURES}'")
  endif()
  if(_base MATCHES "^gfx1[12][05][012]$")
    set(INGINE_HIP_HAS_WMMA ON)
  endif()
  list(APPEND INGINE_HIP_ARCH_LIST "${_base}")
endforeach()

if(NOT INGINE_HIP_ARCH_LIST)
  message(FATAL_ERROR "ingine HIP: INGINE_HIP_ARCHITECTURES cannot be empty")
endif()

set(CMAKE_HIP_ARCHITECTURES "${_ingine_hip_arch_norm}" CACHE STRING "HIP target architectures" FORCE)
string(REPLACE ";" "," INGINE_HIP_ARCHS_CSV "${INGINE_HIP_ARCH_LIST}")

if(NOT EXISTS "$ENV{ROCM_PATH}")
  if(EXISTS "/opt/rocm")
    set(ROCM_PATH "/opt/rocm")
  else()
    set(ROCM_PATH "/usr")
  endif()
else()
  set(ROCM_PATH "$ENV{ROCM_PATH}")
endif()
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/lib/cmake" "${ROCM_PATH}/lib64/cmake")

enable_language(HIP)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

string(STRIP "${CMAKE_HIP_FLAGS_RELEASE}" _ingine_hip_rel_flags)
if(_ingine_hip_rel_flags STREQUAL "")
  set(CMAKE_HIP_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

find_package(hip CONFIG REQUIRED)
find_package(hipblas CONFIG REQUIRED)
find_package(hipblaslt CONFIG QUIET)

if(NOT TARGET hip::host)
  message(FATAL_ERROR "ROCm hip package did not provide hip::host")
endif()
if(NOT TARGET roc::hipblas)
  message(FATAL_ERROR "ROCm hipblas package did not provide roc::hipblas")
endif()

set(INGINE_GPU_RUNTIME_TARGET hip::host)
if(TARGET roc::hipblaslt)
  set(INGINE_HIPBLASLT_AVAILABLE ON)
  set(INGINE_GPU_BLAS_TARGETS roc::hipblaslt roc::hipblas)
else()
  set(INGINE_HIPBLASLT_AVAILABLE OFF)
  set(INGINE_GPU_BLAS_TARGETS roc::hipblas)
  message(STATUS "ingine HIP: hipBLASLt not found; falling back to roc::hipblas for dense GEMMs")
endif()

function(ingine_hip_target target)
  target_compile_definitions(${target} PRIVATE
    INGINE_HIP_ARCHS="${INGINE_HIP_ARCHS_CSV}"
    INGINE_HIP_HAS_WMMA=$<BOOL:${INGINE_HIP_HAS_WMMA}>
    INGINE_HIPBLASLT_AVAILABLE=$<BOOL:${INGINE_HIPBLASLT_AVAILABLE}>
  )
  if(INGINE_HIP_WAVE64)
    target_compile_definitions(${target} PUBLIC INGINE_HIP_WAVE64=1)
  endif()
  if(INGINE_HIP_FP_CONTRACT_ON)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=on>)
  endif()
endfunction()

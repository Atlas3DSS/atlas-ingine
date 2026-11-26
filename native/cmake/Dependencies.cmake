# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
find_package(Threads REQUIRED)

if(INGINE_BACKEND STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
  foreach(_cuda_target CUDA::cudart CUDA::cublasLt)
    if(NOT TARGET ${_cuda_target})
      message(FATAL_ERROR "CUDAToolkit does not provide ${_cuda_target}")
    endif()
  endforeach()
  set(INGINE_GPU_RUNTIME_TARGET CUDA::cudart)
  set(INGINE_GPU_BLAS_TARGETS CUDA::cublasLt)
endif()

# Internal headers: public API in include/, implementation in src/, vendored headers in third_party/.
function(ingine_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party)
endfunction()

# Shared GPU compile settings for every target with .cu / .hip kernel sources:
# - CUDA: whole-program compilation, line info for profilers, relaxed constexpr.
# - HIP: marks every .cu/.hip source in `target` as LANGUAGE HIP and applies ingine_hip_target().
function(ingine_cuda_target target)
  if(INGINE_BACKEND STREQUAL "CUDA")
    set_target_properties(${target} PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr>)
  elseif(INGINE_BACKEND STREQUAL "HIP")
    get_target_property(_ingine_target_srcs ${target} SOURCES)
    if(_ingine_target_srcs)
      foreach(_src IN LISTS _ingine_target_srcs)
        if(_src MATCHES "\\.(cu|hip)$")
          set_source_files_properties(${_src} PROPERTIES LANGUAGE HIP)
        endif()
      endforeach()
    endif()
    ingine_hip_target(${target})
  endif()
endfunction()

# Test oracles: no reassociation or contraction.
set(INGINE_ORACLE_FLAGS -fno-fast-math -ffp-contract=off)

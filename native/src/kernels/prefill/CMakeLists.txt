add_library(ingine_kernels_prefill STATIC
  attention.cu
  moe.cu
  wgemm.cu
  rows.cu
  gemm.cpp
)
ingine_cuda_target(ingine_kernels_prefill)
ingine_internal_includes(ingine_kernels_prefill)
target_include_directories(ingine_kernels_prefill PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_kernels_prefill PUBLIC ingine_core ingine_kernels_common ${INGINE_GPU_RUNTIME_TARGET} ${INGINE_GPU_BLAS_TARGETS})

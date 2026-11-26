add_library(ingine_kernels_dense STATIC
  gemv.cu
  mix.cu
  gates.cu
  router.cu
  persona_ops.cu
)
ingine_cuda_target(ingine_kernels_dense)
ingine_internal_includes(ingine_kernels_dense)
target_include_directories(ingine_kernels_dense PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_kernels_dense PUBLIC ingine_core ingine_kernels_common ${INGINE_GPU_RUNTIME_TARGET})

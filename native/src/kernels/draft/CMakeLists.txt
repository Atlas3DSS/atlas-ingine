add_library(ingine_kernels_draft STATIC
  accept.cu
  block.cu
  fold.cu
  gemm.cpp
  markov.cu
)
ingine_cuda_target(ingine_kernels_draft)
ingine_internal_includes(ingine_kernels_draft)
target_link_libraries(ingine_kernels_draft PUBLIC ingine_core CUDA::cudart CUDA::cublasLt)

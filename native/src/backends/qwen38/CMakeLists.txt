add_library(ingine_qwen38_cpu STATIC
  qwen38_config.cpp
  qwen38_manifest.cpp
  qwen38_cpu_ops.cpp
  qwen38_block_cpu.cpp
  qwen38_backend.cpp
  qwen38_gate2_metrics.cpp
  qwen38_gate3_metrics.cpp
)

target_link_libraries(ingine_qwen38_cpu PUBLIC
  ingine_engine
  ingine_text
  nlohmann_json::nlohmann_json
)

ingine_internal_includes(ingine_qwen38_cpu)
target_compile_options(ingine_qwen38_cpu PRIVATE -Wall -Wextra -Wpedantic)

add_library(ingine_qwen38_cuda STATIC
  qwen38_cuda_ops.cu
  qwen38_dense_gpu.cu
)
ingine_cuda_target(ingine_qwen38_cuda)
ingine_internal_includes(ingine_qwen38_cuda)
target_link_libraries(ingine_qwen38_cuda PUBLIC
  ingine_qwen38_cpu
  ingine_core
  CUDA::cudart
)

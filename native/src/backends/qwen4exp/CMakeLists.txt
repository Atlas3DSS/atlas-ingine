add_library(ingine_qwen4exp_cpu STATIC
  qwen4exp_config.cpp
  qwen4exp_gguf.cpp
  qwen4exp_elastic_safetensors.cpp
  qwen4exp_cpu_ops.cpp
)

target_link_libraries(ingine_qwen4exp_cpu PUBLIC
  ingine_core
  ingine_qwen38_cpu
  ingine_engine
  ingine_text
  nlohmann_json::nlohmann_json
)

ingine_internal_includes(ingine_qwen4exp_cpu)
target_compile_options(ingine_qwen4exp_cpu PRIVATE -Wall -Wextra -Wpedantic)

if(INGINE_BACKEND STREQUAL "CUDA")
  add_library(ingine_qwen4exp_cuda STATIC
    qwen4exp_cuda_ops.cu
    qwen4exp_gpu.cu
  )
  ingine_cuda_target(ingine_qwen4exp_cuda)
  ingine_internal_includes(ingine_qwen4exp_cuda)
  target_link_libraries(ingine_qwen4exp_cuda PUBLIC
    ingine_qwen4exp_cpu
    ingine_core
    CUDA::cudart
  )
endif()

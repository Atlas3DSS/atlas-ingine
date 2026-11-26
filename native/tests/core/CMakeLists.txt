ingine_add_test(NAME core_safetensors SOURCES test_safetensors.cpp LIBS ingine_core)
ingine_add_test(NAME core_safetensors_model SOURCES test_safetensors_model.cpp LIBS ingine_core
  LABELS model)
ingine_add_test(NAME core_oracle_precision_bf16 SOURCES test_scalar_oracle_precision.cpp)
target_compile_definitions(core_oracle_precision_bf16 PRIVATE INGINE_BACKEND_CUDA=0 INGINE_BACKEND_CPU=1 INGINE_ACT_FP16=0)
ingine_add_test(NAME core_oracle_precision_fp16 SOURCES test_scalar_oracle_precision.cpp)
target_compile_definitions(core_oracle_precision_fp16 PRIVATE INGINE_BACKEND_CUDA=0 INGINE_BACKEND_CPU=1 INGINE_ACT_FP16=1)
ingine_add_test(NAME core_vendor_portability SOURCES test_vendor_portability.cpp LIBS ingine_core ORACLE)
add_executable(core_mlx_uploader_host
  test_mlx_uploader_host.cpp
  ${PROJECT_SOURCE_DIR}/src/core/safetensors_upload_mlx.cpp
  ${PROJECT_SOURCE_DIR}/src/core/safetensors_io.cpp
  ${PROJECT_SOURCE_DIR}/src/core/weight_pack.cpp
  ${PROJECT_SOURCE_DIR}/src/core/dtype.cpp
  ${PROJECT_SOURCE_DIR}/src/core/tensor.cpp
)
ingine_internal_includes(core_mlx_uploader_host)
target_include_directories(core_mlx_uploader_host PRIVATE ${PROJECT_SOURCE_DIR}/tests)
target_compile_definitions(core_mlx_uploader_host PRIVATE INGINE_BACKEND_CUDA=0 INGINE_BACKEND_MLX=1)
target_link_libraries(core_mlx_uploader_host PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
if(UNIX AND NOT APPLE)
  target_link_options(core_mlx_uploader_host PRIVATE -Wl,--wrap=pread)
endif()
add_test(NAME core_mlx_uploader_host COMMAND core_mlx_uploader_host)
if(INGINE_BACKEND STREQUAL "CUDA")
  ingine_add_test(NAME core_arch_ops_cuda SOURCES test_arch_ops_device.cu LIBS ingine_core CUDA::cudart GPU)
  target_compile_options(core_arch_ops_cuda PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--Werror cross-execution-space-call>")
  ingine_add_test(NAME hal_embed_q4_parity SOURCES test_embed_q4_parity.cpp LIBS ingine_ops ingine_core CUDA::cudart
                  GPU ORACLE)
  ingine_add_test(NAME hal_embed_q4_capture SOURCES test_embed_q4_capture.cpp LIBS ingine_ops ingine_core CUDA::cudart
                  GPU)
  ingine_add_test(NAME core_arena SOURCES test_arena.cpp LIBS ingine_core GPU)
  ingine_add_test(NAME core_paged_tier SOURCES test_paged_tier.cpp LIBS ingine_core GPU)
  ingine_add_test(NAME core_upload SOURCES test_upload.cpp LIBS ingine_core GPU)
  ingine_add_test(NAME core_baremetal_cpu SOURCES test_baremetal_cpu.cpp LIBS ingine_ops ingine_core CUDA::cudart
                  GPU ORACLE)
  ingine_add_test(NAME core_baremetal_cuda SOURCES test_baremetal_cuda.cpp LIBS ingine_ops ingine_core
                  ingine_kernels_moe CUDA::cudart GPU ORACLE)
  ingine_add_test(NAME core_weight_pack SOURCES test_weight_pack.cpp LIBS ingine_core CUDA::cudart GPU)
else()
  ingine_add_test(NAME hal_embed_q4_parity SOURCES test_embed_q4_parity.cpp LIBS ingine_ops ingine_core ORACLE)
  ingine_add_test(NAME core_upload SOURCES test_upload.cpp LIBS ingine_core)
  ingine_add_test(NAME core_baremetal_cpu SOURCES test_baremetal_cpu.cpp LIBS ingine_ops ingine_core ORACLE)
  ingine_add_test(NAME core_weight_pack SOURCES test_weight_pack.cpp LIBS ingine_core)
endif()

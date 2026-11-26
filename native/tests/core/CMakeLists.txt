ingine_add_test(NAME core_safetensors SOURCES test_safetensors.cpp LIBS ingine_core)
ingine_add_test(NAME core_gguf SOURCES test_gguf.cpp LIBS ingine_core)
ingine_add_test(NAME core_safetensors_model SOURCES test_safetensors_model.cpp LIBS ingine_core
  LABELS model)
ingine_add_test(NAME core_oracle_precision_bf16 SOURCES test_scalar_oracle_precision.cpp)
target_compile_definitions(core_oracle_precision_bf16 PRIVATE INGINE_BACKEND_CUDA=0 INGINE_BACKEND_CPU=1 INGINE_ACT_FP16=0)
ingine_add_test(NAME core_oracle_precision_fp16 SOURCES test_scalar_oracle_precision.cpp)
target_compile_definitions(core_oracle_precision_fp16 PRIVATE INGINE_BACKEND_CUDA=0 INGINE_BACKEND_CPU=1 INGINE_ACT_FP16=1)
if(INGINE_BACKEND STREQUAL "CUDA")
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

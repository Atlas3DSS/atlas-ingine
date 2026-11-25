ingine_add_test(NAME core_safetensors SOURCES test_safetensors.cpp LIBS ingine_core)
ingine_add_test(NAME core_safetensors_model SOURCES test_safetensors_model.cpp LIBS ingine_core
  LABELS model)
if(INGINE_BACKEND STREQUAL "CUDA")
  ingine_add_test(NAME hal_embed_q4_parity SOURCES test_embed_q4_parity.cpp LIBS ingine_ops ingine_core CUDA::cudart
                  GPU ORACLE)
  ingine_add_test(NAME core_arena SOURCES test_arena.cpp LIBS ingine_core GPU)
  ingine_add_test(NAME core_paged_tier SOURCES test_paged_tier.cpp LIBS ingine_core GPU)
  ingine_add_test(NAME core_upload SOURCES test_upload.cpp LIBS ingine_core GPU)
else()
  ingine_add_test(NAME hal_embed_q4_parity SOURCES test_embed_q4_parity.cpp LIBS ingine_ops ingine_core ORACLE)
endif()

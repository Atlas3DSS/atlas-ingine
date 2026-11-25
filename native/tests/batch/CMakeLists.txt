# Continuous-batching tests: CPU scheduler, HTTP 64-client server, GPU batched kernels, and full-model B-vs-B=1 parity.
ingine_add_test(NAME batch_scheduler SOURCES test_batch_scheduler.cpp LIBS ingine_engine LABELS batch serve TIMEOUT 60)
ingine_add_test(NAME batch_server SOURCES test_batch_server.cpp LIBS ingine_serve LABELS batch serve TIMEOUT 120)
if(INGINE_BACKEND STREQUAL "CUDA")
  ingine_add_test(NAME batch_kernels SOURCES test_batch_kernels.cpp LIBS ingine_kernels_batch ingine_kernels_attn ingine_core
                  LABELS batch GPU TIMEOUT 300)
  if(TARGET ingine_model)
    ingine_add_test(NAME batch_parity SOURCES test_batch_parity.cpp LIBS ingine_model ingine_engine ingine_text
                    LABELS batch GPU MODEL TIMEOUT 900)
  endif()
endif()

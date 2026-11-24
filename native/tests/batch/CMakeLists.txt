# Continuous-batching tests: CPU scheduler, prefix cache, HTTP 64-client server, GPU batched kernels, and full-model parity.
ingine_add_test(NAME batch_scheduler SOURCES test_batch_scheduler.cpp LIBS ingine_engine LABELS batch serve TIMEOUT 60)
ingine_add_test(NAME batch_prefix_cache SOURCES test_batch_prefix_cache.cpp LIBS ingine_core LABELS batch TIMEOUT 60)
ingine_add_test(NAME batch_server SOURCES test_batch_server.cpp LIBS ingine_serve LABELS batch serve TIMEOUT 120)
ingine_add_test(NAME batch_kernels SOURCES test_batch_kernels.cpp LIBS ingine_kernels_batch ingine_kernels_attn ingine_core
                LABELS batch GPU TIMEOUT 300)
ingine_add_test(NAME batch_parity SOURCES test_batch_parity.cpp LIBS ingine_model ingine_engine ingine_text
                LABELS batch GPU MODEL TIMEOUT 900)
ingine_add_test(NAME batch_prefix_parity SOURCES test_batch_prefix_parity.cpp LIBS ingine_model ingine_engine ingine_text
                LABELS batch GPU MODEL TIMEOUT 900)

ingine_add_test(NAME model_data SOURCES test_model_data.cpp LIBS ingine_core)
# one real layer at a time (~1.1 GB of GPU memory)
ingine_add_test(NAME model_layer SOURCES test_layer.cpp LIBS ingine_model GPU MODEL TIMEOUT 600)
# the decode runtime (graphs, verify widths, rollback) on a synthetic model (< 2 GB of GPU memory)
ingine_add_test(NAME model_runtime SOURCES test_runtime.cpp LIBS ingine_model GPU TIMEOUT 600)
# full-model parity against the Python engine's goldens (tools/goldens_model.py); ~20 GB of GPU memory
if(TARGET ingine-validate)
  add_test(NAME model_parity COMMAND ingine-validate --golden ${PROJECT_SOURCE_DIR}/tests/golden/model)
  # exact greedy continuations only where the goldens have no near ties (long and FP8-cache cases flip at ties)
  set(_greedy_cases --cases kv16_chat,kv16_t300 --min-top1 1)
  add_test(NAME model_parity_generate COMMAND ingine-validate --generate ${_greedy_cases}
    --golden ${PROJECT_SOURCE_DIR}/tests/golden/model)
  add_test(NAME model_parity_spec COMMAND ingine-validate --generate --spec ${_greedy_cases}
    --golden ${PROJECT_SOURCE_DIR}/tests/golden/model)
  set_tests_properties(model_parity model_parity_generate model_parity_spec PROPERTIES
    LABELS "gpu;model" RESOURCE_LOCK gpu TIMEOUT 3600)
endif()

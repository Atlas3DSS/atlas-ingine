if(TARGET ingine_model)
  ingine_add_test(NAME model_data SOURCES test_model_data.cpp LIBS ingine_model)
  # one real layer at a time (~1.1 GB of GPU memory)
  ingine_add_test(NAME model_layer SOURCES test_layer.cpp LIBS ingine_model GPU MODEL TIMEOUT 600)
  # the decode runtime (graphs, verify widths, rollback, pipelined steps, device-selected verify) and the Generator's
  # stop/cancel/budget exactness on a synthetic model (< 2 GB of GPU memory)
  ingine_add_test(NAME model_runtime SOURCES test_runtime.cpp LIBS ingine_model GPU TIMEOUT 600)
  # prompt checkpoints with speculative decoding on the full model (~22 GB of GPU memory)
  ingine_add_test(NAME model_reuse SOURCES test_reuse.cpp LIBS ingine_model GPU MODEL TIMEOUT 900)
  # Stage 3 (#826): verified root-prefix checkpoint, forks, exact-root, byte-capped pinned host-RAM parking,
  # LRU eviction, memory-pressure fallback, and restore-vs-recompute latency crossover
  ingine_add_test(NAME model_cache_stage3 SOURCES test_cache_stage3.cpp LIBS ingine_model ingine_text GPU MODEL TIMEOUT 1200)
  if(UNIX AND NOT APPLE)
    # Test-only deterministic synchronization wrappers for Suite 2h-consumed-weight-toctou (mutates private embed_norm
    # immediately after SafetensorsModel::read returns its payload and records actual GPU norm[0] + content_sha256).
    target_link_options(model_cache_stage3 PRIVATE
      -Wl,--wrap=_ZNK6ingine16SafetensorsModel4readESt17basic_string_viewIcSt11char_traitsIcEE
      -Wl,--wrap=_ZN6ingine5model12load_weightsERKNSt10filesystem7__cxx114pathEbP11CUstream_stRKNS0_13OffloadConfigE)
  endif()
  # full-model parity against the Python engine's goldens (tools/goldens_model.py); ~20 GB of GPU memory
  if(TARGET ingine-validate)
    set(_validate ingine-validate --model-dir ${INGINE_TEST_MODEL_DIR} --golden ${PROJECT_SOURCE_DIR}/tests/golden/model)
    add_test(NAME model_parity COMMAND ${_validate})
    # exact greedy continuations only where the goldens have no near ties (long and FP8-cache cases flip at ties)
    set(_greedy_cases --cases kv16_chat,kv16_t300 --min-top1 1)
    add_test(NAME model_parity_generate COMMAND ${_validate} --generate ${_greedy_cases})
    add_test(NAME model_parity_spec COMMAND ${_validate} --generate --spec ${_greedy_cases})
    # long greedy paths of the Generator teacher-forced through a plain width-1 runtime: plain decoding exact;
    # speculative decoding as a rate over the four FP8-cache paths (12,000 tokens): single positions move by
    # several logits between kernel widths (30,000 tokens over all golden cases: 13-14 above 0.5 logits, 3 above
    # 2, in old and new verify kernels alike), a real bug moves most tokens
    add_test(NAME model_path COMMAND ${_validate} --path 1500 --cases abl_kv8_t300)
    add_test(NAME model_path_spec COMMAND ${_validate} --path 3000 --spec --cases abl_kv8 --max-gap-rate 15)
    set_tests_properties(model_parity model_parity_generate model_parity_spec model_path model_path_spec PROPERTIES
      LABELS "gpu;model" RESOURCE_LOCK gpu TIMEOUT 3600)
  endif()
  # image input on the full model: colours and text of images, prompt caches with images, an image over several
  # prefill chunks (~22 GB of GPU memory)
  if(TARGET ingine_serve AND TARGET ingine_kernels_vision)
    ingine_add_test(NAME model_vision SOURCES test_vision_generate.cpp LIBS ingine_model ingine_serve
                    LABELS vision GPU MODEL TIMEOUT 900)
  endif()
  # Standalone Part 1 (--kv-q4) KL, VRAM, and 64K/128K/245K needle evaluator
  add_executable(eval_kv_q4 eval_kv_q4.cpp)
  target_link_libraries(eval_kv_q4 PRIVATE ingine_model ingine_text nlohmann_json::nlohmann_json)
  target_compile_definitions(eval_kv_q4 PRIVATE INGINE_GOLDEN_DIR="${PROJECT_SOURCE_DIR}/tests/golden")
else()
  ingine_add_test(NAME model_data SOURCES test_model_data.cpp LIBS ingine_core)
endif()

ingine_add_test(
  NAME backend_model_contract
  SOURCES test_model_backend.cpp
  LIBS ingine_qwen38_cpu ingine_engine nlohmann_json::nlohmann_json
  LABELS backend cpu
)

ingine_add_test(
  NAME qwen38_manifest
  SOURCES test_qwen38_manifest.cpp
  LIBS ingine_qwen38_cpu ingine_engine nlohmann_json::nlohmann_json
  LABELS backend qwen38 cpu
)

ingine_add_test(
  NAME qwen38_text
  SOURCES test_qwen38_text.cpp
  LIBS ingine_qwen38_cpu ingine_text nlohmann_json::nlohmann_json
  LABELS backend qwen38 cpu
)

ingine_add_test(
  NAME qwen38_cpu_ops
  SOURCES test_qwen38_cpu_ops.cpp
  LIBS ingine_qwen38_cpu
  LABELS backend qwen38 cpu
  ORACLE
)

ingine_add_test(
  NAME backend_multifamily_tokenizer
  SOURCES test_multifamily_tokenizer.cpp
  LIBS ingine_text nlohmann_json::nlohmann_json
  LABELS backend text cpu
)

ingine_add_test(
  NAME qwen38_phase1a_cpu_prep
  SOURCES test_qwen38_phase1a_cpu_prep.cpp
  LIBS ingine_qwen38_cpu ingine_engine nlohmann_json::nlohmann_json
  LABELS backend qwen38 cpu
  ORACLE
)

ingine_add_test(
  NAME qwen38_phase1b_cpu_prep
  SOURCES test_qwen38_phase1b_cpu_prep.cpp
  LIBS ingine_qwen38_cpu ingine_engine nlohmann_json::nlohmann_json
  LABELS backend qwen38 cpu
)

if(INGINE_BACKEND STREQUAL "CUDA")
  ingine_add_test(
    NAME qwen38_gpu_gate1
    SOURCES test_qwen38_gpu_gate1.cpp
    LIBS ingine_qwen38_cuda ingine_qwen38_cpu ingine_core nlohmann_json::nlohmann_json
    LABELS backend qwen38 phase1a
    GPU
    ORACLE
    TIMEOUT 600
  )

  ingine_add_test(
    NAME qwen38_gpu_gate2
    SOURCES test_qwen38_gpu_gate2.cpp
    LIBS ingine_qwen38_cuda ingine_qwen38_cpu ingine_core nlohmann_json::nlohmann_json
    LABELS backend qwen38 phase1b
    GPU
    ORACLE
    TIMEOUT 900
  )
endif()

ingine_add_test(
  NAME qwen38_phase1c_cpu_prep
  SOURCES test_qwen38_phase1c_cpu_prep.cpp
  LIBS ingine_qwen38_cpu ingine_engine nlohmann_json::nlohmann_json
  LABELS backend qwen38 cpu
)

ingine_add_test(
  NAME qwen38_gate4_dispatch
  SOURCES test_qwen38_gate4_dispatch.cpp
  LIBS ingine_qwen38_cpu ingine_serve ingine_engine ingine_text nlohmann_json::nlohmann_json
  LABELS backend qwen38 serve cpu
  TIMEOUT 120
)

if(INGINE_BACKEND STREQUAL "CUDA")
  ingine_add_test(
    NAME qwen38_gpu_gate3
    SOURCES test_qwen38_gpu_gate3.cpp
    LIBS ingine_qwen38_cuda ingine_qwen38_cpu ingine_core nlohmann_json::nlohmann_json
    LABELS backend qwen38 phase1c
    GPU
    ORACLE
    TIMEOUT 600
  )

  ingine_add_test(
    NAME qwen38_gpu_883_bench
    SOURCES test_qwen38_gpu_883_bench.cpp
    LIBS ingine_qwen38_cuda ingine_qwen38_cpu ingine_core ingine_text nlohmann_json::nlohmann_json
    LABELS backend qwen38
    GPU
    TIMEOUT 600
  )
endif()

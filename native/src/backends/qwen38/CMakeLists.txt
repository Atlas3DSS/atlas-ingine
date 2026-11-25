add_library(ingine_qwen38_cpu STATIC
  qwen38_config.cpp
  qwen38_manifest.cpp
  qwen38_cpu_ops.cpp
  qwen38_backend.cpp
)

target_link_libraries(ingine_qwen38_cpu PUBLIC
  ingine_engine
  ingine_text
  nlohmann_json::nlohmann_json
)

ingine_internal_includes(ingine_qwen38_cpu)
target_compile_options(ingine_qwen38_cpu PRIVATE -Wall -Wextra -Wpedantic)

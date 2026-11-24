# ingine_persona: CUDA-free persona & mech-interp spec parser, SHA-256 provenance validator, and refusal guard.
add_library(ingine_persona STATIC
  persona.cpp
)
ingine_internal_includes(ingine_persona)
target_include_directories(ingine_persona PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_persona PUBLIC nlohmann_json::nlohmann_json)

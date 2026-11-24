add_executable(ingine-persona-probe main.cpp)
ingine_internal_includes(ingine-persona-probe)
target_link_libraries(ingine-persona-probe PRIVATE ingine_model ingine_serve ingine_text ingine_persona nlohmann_json::nlohmann_json)

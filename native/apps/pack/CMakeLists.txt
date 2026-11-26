add_executable(ingine-pack main.cpp)
ingine_internal_includes(ingine-pack)
target_link_libraries(ingine-pack PRIVATE ingine_core)

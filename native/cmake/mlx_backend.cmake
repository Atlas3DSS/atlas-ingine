# Pure native Apple Silicon MLX C++20 + Metal backend configuration (INGINE_BACKEND=MLX).
# 100% pure C++20 + Metal Shading Language + optional libmlx C++ API: zero Python, zero ggml shims.

if(NOT APPLE)
  message(FATAL_ERROR "INGINE_BACKEND=MLX targets macOS / Apple Silicon with Metal (not available on ${CMAKE_SYSTEM_NAME})")
endif()

enable_language(OBJCXX)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

find_library(INGINE_METAL_FRAMEWORK Metal REQUIRED)
find_library(INGINE_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(INGINE_COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)

find_package(MLX CONFIG QUIET)
if(TARGET mlx)
  set(INGINE_HAVE_LIBMLX ON)
  message(STATUS "ingine MLX: found C++20 libmlx (zero-copy MLX array seam enabled)")
else()
  set(INGINE_HAVE_LIBMLX OFF)
  message(STATUS "ingine MLX: libmlx CMake package not found; building native Metal compute + unified ring path")
endif()

# Optional precompiled .metallib alongside runtime-compiled embedded MSL source.
find_program(INGINE_METAL_COMPILER metal)
find_program(INGINE_METALLIB_TOOL metallib)
if(INGINE_METAL_COMPILER AND INGINE_METALLIB_TOOL)
  set(_embed_metal_src "${PROJECT_SOURCE_DIR}/src/kernels/metal/embed_q4.metal")
  set(_embed_metal_air "${CMAKE_CURRENT_BINARY_DIR}/embed_q4.air")
  set(_embed_metallib "${CMAKE_CURRENT_BINARY_DIR}/ingine_kernels.metallib")
  add_custom_command(
    OUTPUT "${_embed_metallib}"
    COMMAND "${INGINE_METAL_COMPILER}" -c "${_embed_metal_src}" -o "${_embed_metal_air}"
    COMMAND "${INGINE_METALLIB_TOOL}" "${_embed_metal_air}" -o "${_embed_metallib}"
    DEPENDS "${_embed_metal_src}"
    COMMENT "Compiling pure Metal shader embed_q4.metal -> ingine_kernels.metallib"
  )
  add_custom_target(ingine_metallib DEPENDS "${_embed_metallib}")
endif()

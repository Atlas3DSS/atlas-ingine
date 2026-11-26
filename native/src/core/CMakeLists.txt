# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
if(INGINE_BACKEND STREQUAL "CUDA")
  add_library(ingine_core STATIC
    arena.cu
    decode_graph.cpp
    device.cu
    dtype.cpp
    hal_cuda.cpp
    host_memory.cpp
    paged_tier.cpp
    safetensors_io.cpp
    safetensors_upload_cuda.cpp
    tensor.cpp
    weight_pack.cpp
  )
  ingine_cuda_target(ingine_core)
  target_compile_definitions(ingine_core PUBLIC
    INGINE_BACKEND_CUDA=1 INGINE_BACKEND_CPU=0 INGINE_BACKEND_HIP=0 INGINE_BACKEND_MLX=0 INGINE_BACKEND_SYCL=0)
  target_link_libraries(ingine_core PUBLIC
    CUDA::cudart CUDA::nvtx3 Threads::Threads spdlog::spdlog nlohmann_json::nlohmann_json)

  add_library(ingine_ops STATIC
    cpu_kernels.cpp
    cpu_pool.cpp
    ops_fallback.cpp
    ops_cuda.cpp
  )
  ingine_internal_includes(ingine_ops)
  target_include_directories(ingine_ops PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(ingine_ops PUBLIC ingine_core Threads::Threads)
elseif(INGINE_BACKEND STREQUAL "HIP")
  add_library(ingine_core STATIC
    arena_hip.cpp
    decode_graph_hip.cpp
    device_hip.cpp
    dtype.cpp
    hal_hip.cpp
    host_memory_hip.cpp
    paged_tier.cpp
    safetensors_io.cpp
    safetensors_upload_hip.cpp
    tensor.cpp
    weight_pack.cpp
  )
  ingine_hip_target(ingine_core)
  target_compile_definitions(ingine_core PUBLIC
    INGINE_BACKEND_CUDA=0 INGINE_BACKEND_CPU=0 INGINE_BACKEND_HIP=1 INGINE_BACKEND_MLX=0 INGINE_BACKEND_SYCL=0)
  target_link_libraries(ingine_core PUBLIC
    hip::host roc::hipblas Threads::Threads spdlog::spdlog nlohmann_json::nlohmann_json)
  if(TARGET roc::hipblaslt)
    target_link_libraries(ingine_core PUBLIC roc::hipblaslt)
  endif()

  add_library(ingine_ops STATIC
    cpu_kernels.cpp
    cpu_pool.cpp
    ops_fallback.cpp
    ops_hip.cpp
    ${PROJECT_SOURCE_DIR}/src/kernels/dense/embed_q4_hip.hip
  )
  ingine_hip_target(ingine_ops)
  ingine_internal_includes(ingine_ops)
  target_include_directories(ingine_ops PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(ingine_ops PUBLIC ingine_core Threads::Threads)
elseif(INGINE_BACKEND STREQUAL "MLX")
  set_source_files_properties(hal_mlx.cpp ops_mlx.cpp PROPERTIES LANGUAGE OBJCXX)
  add_library(ingine_core STATIC
    dtype.cpp
    hal_mlx.cpp
    safetensors_io.cpp
    safetensors_upload_mlx.cpp
    tensor.cpp
    weight_pack.cpp
  )
  target_compile_definitions(ingine_core PUBLIC
    INGINE_BACKEND_CUDA=0 INGINE_BACKEND_CPU=0 INGINE_BACKEND_HIP=0 INGINE_BACKEND_MLX=1 INGINE_BACKEND_SYCL=0)
  target_link_libraries(ingine_core PUBLIC
    ${INGINE_METAL_FRAMEWORK} ${INGINE_FOUNDATION_FRAMEWORK} ${INGINE_COREGRAPHICS_FRAMEWORK}
    Threads::Threads spdlog::spdlog nlohmann_json::nlohmann_json)
  if(INGINE_HAVE_LIBMLX)
    target_compile_definitions(ingine_core PUBLIC INGINE_HAVE_LIBMLX=1)
    target_link_libraries(ingine_core PUBLIC mlx)
  endif()

  add_library(ingine_ops STATIC
    cpu_kernels.cpp
    cpu_pool.cpp
    ops_fallback.cpp
    ops_mlx.cpp
  )
  ingine_internal_includes(ingine_ops)
  target_include_directories(ingine_ops PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(ingine_ops PUBLIC ingine_core Threads::Threads)
elseif(INGINE_BACKEND STREQUAL "SYCL")
  add_library(ingine_core STATIC
    dtype.cpp
    hal_sycl.cpp
    safetensors_io.cpp
    safetensors_upload_sycl.cpp
    tensor.cpp
    weight_pack.cpp
  )
  ingine_sycl_target(ingine_core)
  target_compile_definitions(ingine_core PUBLIC
    INGINE_BACKEND_CUDA=0 INGINE_BACKEND_CPU=0 INGINE_BACKEND_HIP=0 INGINE_BACKEND_MLX=0 INGINE_BACKEND_SYCL=1)
  target_link_libraries(ingine_core PUBLIC
    Threads::Threads spdlog::spdlog nlohmann_json::nlohmann_json)

  add_library(ingine_ops STATIC
    cpu_kernels.cpp
    cpu_pool.cpp
    ops_fallback.cpp
    ops_sycl.cpp
  )
  ingine_sycl_target(ingine_ops)
  ingine_internal_includes(ingine_ops)
  target_include_directories(ingine_ops PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(ingine_ops PUBLIC ingine_core Threads::Threads)
else()
  add_library(ingine_core STATIC
    dtype.cpp
    hal_cpu.cpp
    safetensors_io.cpp
    safetensors_upload_cpu.cpp
    tensor.cpp
    weight_pack.cpp
  )
  target_compile_definitions(ingine_core PUBLIC
    INGINE_BACKEND_CUDA=0 INGINE_BACKEND_CPU=1 INGINE_BACKEND_HIP=0 INGINE_BACKEND_MLX=0 INGINE_BACKEND_SYCL=0)
  target_link_libraries(ingine_core PUBLIC
    Threads::Threads spdlog::spdlog nlohmann_json::nlohmann_json)

  add_library(ingine_ops STATIC
    cpu_kernels.cpp
    cpu_pool.cpp
    ops_fallback.cpp
    ops_cpu.cpp
  )
  ingine_internal_includes(ingine_ops)
  target_include_directories(ingine_ops PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
  target_link_libraries(ingine_ops PUBLIC ingine_core Threads::Threads)
endif()

ingine_internal_includes(ingine_core)
target_include_directories(ingine_core PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
if(INGINE_ACT_FP16)
  target_compile_definitions(ingine_core PUBLIC INGINE_ACT_FP16=1)
else()
  target_compile_definitions(ingine_core PUBLIC INGINE_ACT_FP16=0)
endif()
if(INGINE_FORCE_SIMT_MMA)
  target_compile_definitions(ingine_core PUBLIC INGINE_FORCE_SIMT_MMA=1)
endif()

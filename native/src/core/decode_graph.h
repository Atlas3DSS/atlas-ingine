// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Modified for fngine from cinference:src/core/decode_graph.h @ 383e5db. See NOTICE.
// Modified by satellitedown for Cinference: expose captured CUDA Graph topology signatures.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#ifndef INGINE_BACKEND_CUDA
#if defined(INGINE_BACKEND_CPU) || defined(INGINE_BACKEND_HIP) || defined(INGINE_BACKEND_MLX) || defined(INGINE_BACKEND_SYCL)
#define INGINE_BACKEND_CUDA 0
#else
#define INGINE_BACKEND_CUDA 1
#endif
#endif

#ifndef INGINE_BACKEND_HIP
#define INGINE_BACKEND_HIP 0
#endif

#if INGINE_BACKEND_HIP
#include <hip/hip_runtime.h>
#elif INGINE_BACKEND_CUDA
#include <cuda_runtime.h>
#endif

#include <cstdint>
#include <functional>
#include <utility>

namespace ingine {

#if INGINE_BACKEND_HIP
using GraphStreamT = hipStream_t;
using BackendGraphT = hipGraph_t;
using BackendGraphExecT = hipGraphExec_t;
#elif INGINE_BACKEND_CUDA
using GraphStreamT = cudaStream_t;
using BackendGraphT = cudaGraph_t;
using BackendGraphExecT = cudaGraphExec_t;
#else
using GraphStreamT = void*;
using BackendGraphT = void*;
using BackendGraphExecT = void*;
#endif

class DecodeGraphDefinition {
public:
    DecodeGraphDefinition() = default;
    ~DecodeGraphDefinition();

    DecodeGraphDefinition(const DecodeGraphDefinition&)            = delete;
    DecodeGraphDefinition& operator=(const DecodeGraphDefinition&) = delete;
    DecodeGraphDefinition(DecodeGraphDefinition&& other) noexcept;
    DecodeGraphDefinition& operator=(DecodeGraphDefinition&& other) noexcept;

    void capture(GraphStreamT stream, const std::function<void()>& body);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

    // Node count and an order-sensitive hash of the captured node sequence. Profiles with different
    // signatures capture different node topologies and must not share one executable: a graph
    // executable can be updated across profiles only while the topology is unchanged.
    [[nodiscard]] std::pair<std::size_t, std::uint64_t> topology_signature() const;

private:
    friend class DecodeGraphExecutable;
    BackendGraphT graph_ = nullptr;
};

class DecodeGraphExecutable {
public:
    DecodeGraphExecutable() = default;
    ~DecodeGraphExecutable();

    DecodeGraphExecutable(const DecodeGraphExecutable&)            = delete;
    DecodeGraphExecutable& operator=(const DecodeGraphExecutable&) = delete;
    DecodeGraphExecutable(DecodeGraphExecutable&& other) noexcept;
    DecodeGraphExecutable& operator=(DecodeGraphExecutable&& other) noexcept;

    void instantiate(const DecodeGraphDefinition& definition);
    void update(const DecodeGraphDefinition& definition);
    void upload(GraphStreamT stream);
    void launch(GraphStreamT stream);
    [[nodiscard]] bool ready() const noexcept;
    void reset() noexcept;

private:
    BackendGraphExecT exec_ = nullptr;
};

} // namespace ingine

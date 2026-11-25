// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Modified for fngine from cinference:src/core/device.h @ 383e5db. See NOTICE.
// Modified by satellitedown for Cinference: a concurrent compute stream with fork/join events.
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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace ingine {

#if INGINE_BACKEND_HIP
using BackendStreamT = hipStream_t;
using BackendEventT = hipEvent_t;
using BackendDevicePropT = hipDeviceProp_t;
void hip_check(hipError_t err, const char* expr, const char* file, int line);
#define HIP_CHECK(expr) ::ingine::hip_check((expr), #expr, __FILE__, __LINE__)
#elif INGINE_BACKEND_CUDA
using BackendStreamT = cudaStream_t;
using BackendEventT = cudaEvent_t;
using BackendDevicePropT = cudaDeviceProp;
void cuda_check(cudaError_t err, const char* expr, const char* file, int line);
#define CUDA_CHECK(expr) ::ingine::cuda_check((expr), #expr, __FILE__, __LINE__)
#endif

// Cached compute capability major version for the active device (safe inside graph capture).
inline int current_sm_major() {
#if INGINE_BACKEND_HIP
    static std::array<int, 16> cached{};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16) return 11;
    if (cached[dev] == 0) {
        int maj = 0;
        if (hipDeviceGetAttribute(&maj, hipDeviceAttributeComputeCapabilityMajor, dev) == hipSuccess && maj > 0) {
            cached[dev] = maj;
        } else {
            return 11;
        }
    }
    return cached[dev];
#elif INGINE_BACKEND_CUDA
    static std::array<int, 16> cached{};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 8;
    if (cached[dev] == 0) {
        int maj = 0;
        if (cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && maj > 0) {
            cached[dev] = maj;
        } else {
            return 8;
        }
    }
    return cached[dev];
#else
    return 0;
#endif
}

// Cached maximum opt-in dynamic shared memory per block in bytes (safe inside graph capture).
// Pascal sm_60/61: 49,152 B; Turing sm_75: 65,536 B; Volta sm_70: 98,304 B; Ampere+: >= 101,376 B.
// Honors INGINE_TEST_MAX_SMEM (cap in bytes) for exercising lower-SMEM paths on larger-SMEM GPUs.
inline int current_max_smem_optin() {
    static std::array<int, 16> cached{};
    static const int cap = [] {
        const char* e = std::getenv("INGINE_TEST_MAX_SMEM");
        const int v = e ? std::atoi(e) : 0;
        return v > 0 ? v : 0;
    }();
#if INGINE_BACKEND_HIP
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16) return cap > 0 ? std::min(65536, cap) : 65536;
    if (cached[dev] == 0) {
        int bytes = 0;
        if (hipDeviceGetAttribute(&bytes, hipDeviceAttributeMaxSharedMemoryPerBlock, dev) != hipSuccess || bytes <= 0) {
            bytes = 65536;
        }
        cached[dev] = bytes;
    }
    return cap > 0 ? std::min(cached[dev], cap) : cached[dev];
#elif INGINE_BACKEND_CUDA
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return cap > 0 ? std::min(49152, cap) : 49152;
    if (cached[dev] == 0) {
        int bytes = 0;
        if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess || bytes <= 0) {
            if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlock, dev) != cudaSuccess || bytes <= 0) {
                bytes = 49152;
            }
        }
        cached[dev] = bytes;
    }
    return cap > 0 ? std::min(cached[dev], cap) : cached[dev];
#else
    return cap;
#endif
}

#if INGINE_BACKEND_CUDA || INGINE_BACKEND_HIP

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    BackendStreamT stream             = nullptr;
    std::int32_t multiprocessor_count = 0;
};

// A second compute stream, at the compute stream's priority, for work that a sequence of Ops forks
// from `stream` and joins back into it through `events` (recorded on one stream, waited on the
// other). Every fork is joined before the sequence ends, so no work stays pending on it.
struct ConcurrentStream {
    static constexpr std::size_t kEvents = 3;
    BackendStreamT stream                = nullptr;
    std::array<BackendEventT, kEvents> events{};
};

struct DeviceContext {
    int device                     = 0;
    BackendStreamT stream          = nullptr;
    BackendStreamT transfer_stream = nullptr;
    ConcurrentStream concurrent;
    BackendDevicePropT props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int compute_capability() const noexcept;
    int multiprocessor_count() const noexcept;
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    void synchronize() const;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, BackendStreamT stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    BackendStreamT stream_ = nullptr;
    BackendEventT start_   = nullptr;
    BackendEventT stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(BackendStreamT stream);
    void wait(BackendStreamT stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_          = 0;
    BackendEventT event_ = nullptr;
};

#endif  // INGINE_BACKEND_CUDA || INGINE_BACKEND_HIP

} // namespace ingine

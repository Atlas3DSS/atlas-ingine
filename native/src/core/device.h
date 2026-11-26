// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Modified for fngine from cinference:src/core/device.h @ 383e5db. See NOTICE.
// Modified by satellitedown for Cinference: a concurrent compute stream with fork/join events.
// See NOTICE and upstream-provenance.json for upstream attribution.

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace ingine {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ingine::cuda_check((expr), #expr, __FILE__, __LINE__)

// Cached compute capability major version for the active CUDA device (thread-safe, safe inside CUDA graph capture).
inline int current_sm_major() {
    static std::array<std::atomic<int>, 16> cached{};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 8;
    int v = cached[dev].load(std::memory_order_acquire);
    if (v == 0) {
        int maj = 0;
        if (cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && maj > 0) {
            v = maj;
            cached[dev].store(v, std::memory_order_release);
        } else {
            return 8;
        }
    }
    return v;
}

// Cached compute capability version as major * 10 + minor (e.g. 60, 61, 70, 75, 86, 90, 120).
inline int current_sm_version() {
    static std::array<std::atomic<int>, 16> cached{};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 80;
    int v = cached[dev].load(std::memory_order_acquire);
    if (v == 0) {
        int maj = 0, min = 0;
        if (cudaDeviceGetAttribute(&maj, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&min, cudaDevAttrComputeCapabilityMinor, dev) == cudaSuccess && maj > 0) {
            v = maj * 10 + min;
            cached[dev].store(v, std::memory_order_release);
        } else {
            return 80;
        }
    }
    return v;
}

// Cached maximum opt-in dynamic shared memory per block in bytes (thread-safe, safe inside CUDA graph capture).
// Pascal sm_60/61: 49,152 B; Turing sm_75: 65,536 B; Volta sm_70: 98,304 B; Ampere+: >= 101,376 B.
// Honors INGINE_TEST_MAX_SMEM (cap in bytes) for exercising lower-SMEM paths on larger-SMEM GPUs.
inline int current_max_smem_optin() {
    static std::array<std::atomic<int>, 16> cached{};
    static const int cap = [] {
        const char* e = std::getenv("INGINE_TEST_MAX_SMEM");
        const int v = e ? std::atoi(e) : 0;
        return v > 0 ? v : 0;
    }();
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return cap > 0 ? std::min(49152, cap) : 49152;
    int v = cached[dev].load(std::memory_order_acquire);
    if (v == 0) {
        int bytes = 0;
        if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess || bytes <= 0) {
            if (cudaDeviceGetAttribute(&bytes, cudaDevAttrMaxSharedMemoryPerBlock, dev) != cudaSuccess || bytes <= 0) {
                bytes = 49152;
            }
        }
        v = bytes;
        cached[dev].store(v, std::memory_order_release);
    }
    return cap > 0 ? std::min(v, cap) : v;
}

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;
};

// A second compute stream, at the compute stream's priority, for work that a sequence of Ops forks
// from `stream` and joins back into it through `events` (recorded on one stream, waited on the
// other). Every fork is joined before the sequence ends, so no work stays pending on it.
struct ConcurrentStream {
    static constexpr std::size_t kEvents = 3;
    cudaStream_t stream                  = nullptr;
    std::array<cudaEvent_t, kEvents> events{};
};

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    ConcurrentStream concurrent;
    cudaDeviceProp props{};

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
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
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
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
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

    void record(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

} // namespace ingine

// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
#pragma once

// Large host memory regions (pinned or registered staging) and device-mapped exchange buffers.
#include <cstddef>
#include <cstdint>
#include <memory>

namespace ingine {

class WeightPack;

inline constexpr std::size_t kHugePageBytes = 2ull << 20;

// One 2 MiB-aligned anonymous mapping advised MADV_HUGEPAGE. Callers fill it and then register
// it with CUDA so the GPU can read it
// over PCIe and cudaMemcpyAsync can DMA from it. Registration failure is not fatal: the
// region stays usable by the CPU, registered() returns false and a warning is logged.
class HostRegion {
public:
    HostRegion() noexcept = default;
    // Maps align_up(bytes, 2 MiB) bytes. Pages are not touched (no prefault).
    explicit HostRegion(std::size_t bytes);
    ~HostRegion();

    HostRegion(const HostRegion&)            = delete;
    HostRegion& operator=(const HostRegion&) = delete;
    HostRegion(HostRegion&& other) noexcept;
    HostRegion& operator=(HostRegion&& other) noexcept;

    // cudaHostRegister(portable | mapped). Returns registered(). Idempotent.
    bool register_with_cuda();
    void unregister() noexcept;

    [[nodiscard]] std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool registered() const noexcept { return registered_; }
    [[nodiscard]] bool hugepage_advised() const noexcept { return hugepage_advised_; }
    [[nodiscard]] bool numa_bound() const noexcept { return numa_bound_; }
    [[nodiscard]] int numa_node() const noexcept { return numa_node_; }
    // Device-visible alias of data() when registered (UVA: equal to data()), else nullptr.
    [[nodiscard]] void* device_ptr() const noexcept { return device_; }
    // Seconds the last register_with_cuda() call took.
    [[nodiscard]] double register_seconds() const noexcept { return register_seconds_; }

private:
    void release() noexcept;

    std::byte* data_         = nullptr;
    std::size_t size_        = 0;
    bool registered_         = false;
    bool via_mmap_           = true;
    bool hugepage_advised_   = false;
    bool numa_bound_         = false;
    int numa_node_           = -1;
    void* device_            = nullptr;
    double register_seconds_ = 0;
};

// Pinned host allocation mapped into the device address space (cudaHostAlloc portable|mapped),
// or 2 MiB THP mmap + cudaHostRegister when bytes >= 2 MiB.
// Zero-filled on construction. Throws on failure.
class MappedHostBuffer {
public:
    MappedHostBuffer() noexcept = default;
    explicit MappedHostBuffer(std::size_t bytes);
    ~MappedHostBuffer();

    MappedHostBuffer(const MappedHostBuffer&)            = delete;
    MappedHostBuffer& operator=(const MappedHostBuffer&) = delete;
    MappedHostBuffer(MappedHostBuffer&& other) noexcept;
    MappedHostBuffer& operator=(MappedHostBuffer&& other) noexcept;

    [[nodiscard]] void* data() const noexcept { return data_; }
    [[nodiscard]] void* device_ptr() const noexcept { return device_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool hugepage_advised() const noexcept { return hugepage_advised_; }

private:
    void release() noexcept;

    void* data_                = nullptr;
    void* device_              = nullptr;
    std::size_t size_          = 0;
    std::size_t mapped_bytes_  = 0;
    bool via_hugepage_mmap_    = false;
    bool hugepage_advised_     = false;
};

// RAII zero-copy device registration of an open 2 MiB-aligned INGPACK1 WeightPack mmap for unified-memory
// APUs (e.g., AMD Strix Halo gfx1151 128 GB LPDDR5X). Keeps the WeightPack mapping alive until after
// hipHostUnregister / cudaHostUnregister completes. Registration failure is non-fatal (registered() == false).
class UnifiedWeightPackPin {
public:
    UnifiedWeightPackPin() noexcept = default;
    explicit UnifiedWeightPackPin(std::shared_ptr<const WeightPack> pack);
    ~UnifiedWeightPackPin();

    UnifiedWeightPackPin(const UnifiedWeightPackPin&)            = delete;
    UnifiedWeightPackPin& operator=(const UnifiedWeightPackPin&) = delete;
    UnifiedWeightPackPin(UnifiedWeightPackPin&& other) noexcept;
    UnifiedWeightPackPin& operator=(UnifiedWeightPackPin&& other) noexcept;

    // Registers the WeightPack's mapped address space with the active GPU runtime (portable | mapped | read-only,
    // retrying without read-only if the driver rejects read-only file mappings). Idempotent; returns registered().
    bool pin();
    void unpin() noexcept;

    [[nodiscard]] bool registered() const noexcept { return registered_; }
    [[nodiscard]] bool readonly() const noexcept { return readonly_; }
    [[nodiscard]] const void* host_base() const noexcept { return host_base_; }
    [[nodiscard]] void* device_base() const noexcept { return device_base_; }
    [[nodiscard]] std::size_t mapped_bytes() const noexcept { return size_; }
    [[nodiscard]] double register_seconds() const noexcept { return register_seconds_; }
    [[nodiscard]] const WeightPack* pack() const noexcept { return owner_.get(); }

    // Device pointer for an in-bounds byte range [offset, offset + bytes) within the pinned WeightPack, or nullptr.
    [[nodiscard]] void* tensor_device_ptr(std::uint64_t offset, std::uint64_t bytes) const noexcept;

private:
    void release() noexcept;

    // Declared first so `owner_` is destroyed strictly AFTER release() unregisters host_base_.
    std::shared_ptr<const WeightPack> owner_;
    const void* host_base_   = nullptr;
    void* device_base_       = nullptr;
    std::size_t size_        = 0;
    bool registered_         = false;
    bool readonly_           = false;
    double register_seconds_ = 0.0;
};

// /proc/meminfo MemAvailable in bytes (0 when unreadable).
std::uint64_t mem_available_bytes();

} // namespace ingine

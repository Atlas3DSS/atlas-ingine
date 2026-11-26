// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
#pragma once

// Large host memory regions (pinned or registered staging) and device-mapped exchange buffers.
#include <cstddef>
#include <cstdint>

namespace ingine {

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

// /proc/meminfo MemAvailable in bytes (0 when unreadable).
std::uint64_t mem_available_bytes();

} // namespace ingine

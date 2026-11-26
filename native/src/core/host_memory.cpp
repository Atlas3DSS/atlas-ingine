// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
#include "core/host_memory.h"

#include "core/hugepage.h"
#include "core/weight_pack.h"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <sys/mman.h>

namespace ingine {

HostRegion::HostRegion(std::size_t bytes) {
    if (bytes == 0) { throw std::invalid_argument("HostRegion size must be nonzero"); }
    HugePageMapping m = alloc_hugepage_anon(bytes);
    if (!m.ptr) {
        throw std::runtime_error("HostRegion: allocation of " + std::to_string(bytes) +
                                 " bytes failed: " + std::strerror(errno));
    }
    data_             = static_cast<std::byte*>(m.ptr);
    size_             = m.mapped_bytes;
    via_mmap_         = m.via_mmap;
    hugepage_advised_ = m.hugepage_advised;
    numa_bound_       = m.numa_bound;
    numa_node_        = m.numa_node;
    if (hugepages_enabled() && !hugepage_advised_) {
        spdlog::warn("HostRegion: madvise(MADV_HUGEPAGE) failed or unavailable");
    }
}

HostRegion::~HostRegion() { release(); }

HostRegion::HostRegion(HostRegion&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
      registered_(std::exchange(other.registered_, false)),
      via_mmap_(std::exchange(other.via_mmap_, true)),
      hugepage_advised_(std::exchange(other.hugepage_advised_, false)),
      numa_bound_(std::exchange(other.numa_bound_, false)),
      numa_node_(std::exchange(other.numa_node_, -1)),
      device_(std::exchange(other.device_, nullptr)), register_seconds_(other.register_seconds_) {}

HostRegion& HostRegion::operator=(HostRegion&& other) noexcept {
    if (this != &other) {
        release();
        data_             = std::exchange(other.data_, nullptr);
        size_             = std::exchange(other.size_, 0);
        registered_       = std::exchange(other.registered_, false);
        via_mmap_         = std::exchange(other.via_mmap_, true);
        hugepage_advised_ = std::exchange(other.hugepage_advised_, false);
        numa_bound_       = std::exchange(other.numa_bound_, false);
        numa_node_        = std::exchange(other.numa_node_, -1);
        device_           = std::exchange(other.device_, nullptr);
        register_seconds_ = other.register_seconds_;
    }
    return *this;
}

bool HostRegion::register_with_cuda() {
    if (registered_ || !data_) { return registered_; }
    if (hugepage_advised_) { prefault_hugepage_stride_nondestructive(data_, size_); }
    const auto start = std::chrono::steady_clock::now();
    cudaError_t err  = cudaHostRegister(data_, size_, cudaHostRegisterPortable | cudaHostRegisterMapped);
    register_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        spdlog::warn("HostRegion: cudaHostRegister of {:.2f} GiB failed ({}: {}); the region stays "
                     "pageable; asynchronous transfers require pinned staging",
                     double(size_) / double(1ull << 30), cudaGetErrorName(err), cudaGetErrorString(err));
        return false;
    }
    void* dev = nullptr;
    err       = cudaHostGetDevicePointer(&dev, data_, 0);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        (void)cudaHostUnregister(data_);
        spdlog::warn("HostRegion: cudaHostGetDevicePointer failed ({}); region stays CPU-only",
                     cudaGetErrorString(err));
        return false;
    }
    registered_ = true;
    device_     = dev;
    return true;
}

void HostRegion::unregister() noexcept {
    if (registered_) {
        const cudaError_t err = cudaHostUnregister(data_);
        if (err != cudaSuccess) {
            spdlog::error("HostRegion: cudaHostUnregister failed: {}", cudaGetErrorString(err));
        }
        registered_ = false;
        device_     = nullptr;
    }
}

void HostRegion::release() noexcept {
    unregister();
    if (data_) {
        HugePageMapping m{data_, size_, hugepage_advised_, via_mmap_};
        free_hugepage_anon(m);
    }
    data_             = nullptr;
    size_             = 0;
    via_mmap_         = true;
    hugepage_advised_ = false;
    numa_bound_       = false;
    numa_node_        = -1;
}

MappedHostBuffer::MappedHostBuffer(std::size_t bytes) {
    if (bytes == 0) { throw std::invalid_argument("MappedHostBuffer size must be nonzero"); }
    if (bytes >= kHugePageBytes && hugepages_enabled()) {
        HugePageMapping m = alloc_hugepage_anon(bytes);
        if (m.ptr) {
            std::memset(m.ptr, 0, bytes);
            if (m.hugepage_advised) { prefault_hugepage_stride_nondestructive(m.ptr, m.mapped_bytes); }
            cudaError_t err =
                cudaHostRegister(m.ptr, m.mapped_bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
            if (err == cudaSuccess) {
                void* dev = nullptr;
                err       = cudaHostGetDevicePointer(&dev, m.ptr, 0);
                if (err == cudaSuccess) {
                    data_              = m.ptr;
                    device_            = dev;
                    size_              = bytes;
                    mapped_bytes_      = m.mapped_bytes;
                    via_hugepage_mmap_ = m.via_mmap;
                    hugepage_advised_  = m.hugepage_advised;
                    return;
                }
                (void)cudaGetLastError();
                (void)cudaHostUnregister(m.ptr);
            } else {
                (void)cudaGetLastError();
            }
            free_hugepage_anon(m);
        }
    }

    void* ptr       = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaHostAlloc(mapped) of ") + std::to_string(bytes) +
                                 " bytes failed: " + cudaGetErrorString(err));
    }
    void* dev = nullptr;
    err       = cudaHostGetDevicePointer(&dev, ptr, 0);
    if (err != cudaSuccess) {
        (void)cudaFreeHost(ptr);
        throw std::runtime_error(std::string("cudaHostGetDevicePointer failed: ") +
                                 cudaGetErrorString(err));
    }
    std::memset(ptr, 0, bytes);
    data_   = ptr;
    device_ = dev;
    size_   = bytes;
}

MappedHostBuffer::~MappedHostBuffer() { release(); }

MappedHostBuffer::MappedHostBuffer(MappedHostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), device_(std::exchange(other.device_, nullptr)),
      size_(std::exchange(other.size_, 0)), mapped_bytes_(std::exchange(other.mapped_bytes_, 0)),
      via_hugepage_mmap_(std::exchange(other.via_hugepage_mmap_, false)),
      hugepage_advised_(std::exchange(other.hugepage_advised_, false)) {}

MappedHostBuffer& MappedHostBuffer::operator=(MappedHostBuffer&& other) noexcept {
    if (this != &other) {
        release();
        data_              = std::exchange(other.data_, nullptr);
        device_            = std::exchange(other.device_, nullptr);
        size_              = std::exchange(other.size_, 0);
        mapped_bytes_      = std::exchange(other.mapped_bytes_, 0);
        via_hugepage_mmap_ = std::exchange(other.via_hugepage_mmap_, false);
        hugepage_advised_  = std::exchange(other.hugepage_advised_, false);
    }
    return *this;
}

void MappedHostBuffer::release() noexcept {
    if (data_) {
        if (mapped_bytes_ > 0) {
            const cudaError_t err = cudaHostUnregister(data_);
            if (err != cudaSuccess) {
                spdlog::error("MappedHostBuffer: cudaHostUnregister failed: {}", cudaGetErrorString(err));
            }
            HugePageMapping m{data_, mapped_bytes_, hugepage_advised_, via_hugepage_mmap_};
            free_hugepage_anon(m);
        } else {
            const cudaError_t err = cudaFreeHost(data_);
            if (err != cudaSuccess) {
                spdlog::error("MappedHostBuffer: cudaFreeHost failed: {}", cudaGetErrorString(err));
            }
        }
    }
    data_              = nullptr;
    device_            = nullptr;
    size_              = 0;
    mapped_bytes_      = 0;
    via_hugepage_mmap_ = false;
    hugepage_advised_  = false;
}

UnifiedWeightPackPin::UnifiedWeightPackPin(std::shared_ptr<const WeightPack> pack)
    : owner_(std::move(pack)),
      host_base_(owner_ ? static_cast<const void*>(owner_->mapped_base()) : nullptr),
      size_(owner_ ? static_cast<std::size_t>(owner_->file_size()) : 0) {}

UnifiedWeightPackPin::~UnifiedWeightPackPin() { release(); }

UnifiedWeightPackPin::UnifiedWeightPackPin(UnifiedWeightPackPin&& other) noexcept
    : owner_(std::move(other.owner_)),
      host_base_(std::exchange(other.host_base_, nullptr)),
      device_base_(std::exchange(other.device_base_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      registered_(std::exchange(other.registered_, false)),
      readonly_(std::exchange(other.readonly_, false)),
      register_seconds_(other.register_seconds_) {}

UnifiedWeightPackPin& UnifiedWeightPackPin::operator=(UnifiedWeightPackPin&& other) noexcept {
    if (this != &other) {
        release();
        owner_            = std::move(other.owner_);
        host_base_        = std::exchange(other.host_base_, nullptr);
        device_base_      = std::exchange(other.device_base_, nullptr);
        size_             = std::exchange(other.size_, 0);
        registered_       = std::exchange(other.registered_, false);
        readonly_         = std::exchange(other.readonly_, false);
        register_seconds_ = other.register_seconds_;
    }
    return *this;
}

bool UnifiedWeightPackPin::pin() {
    if (registered_ || !host_base_ || size_ == 0) { return registered_; }
    const auto start = std::chrono::steady_clock::now();
    void* raw        = const_cast<void*>(host_base_);
    bool ro          = true;
    cudaError_t err  = cudaHostRegister(raw, size_, cudaHostRegisterPortable | cudaHostRegisterMapped | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        ro  = false;
        err = cudaHostRegister(raw, size_, cudaHostRegisterPortable | cudaHostRegisterMapped);
    }
    register_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        spdlog::warn("UnifiedWeightPackPin(CUDA): cudaHostRegister({:.2f} GiB) failed ({}: {}); falling back to DeviceUploader",
                     double(size_) / double(1ull << 30), cudaGetErrorName(err), cudaGetErrorString(err));
        return false;
    }
    void* dev = nullptr;
    err       = cudaHostGetDevicePointer(&dev, raw, 0);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        (void)cudaHostUnregister(raw);
        spdlog::warn("UnifiedWeightPackPin(CUDA): cudaHostGetDevicePointer failed ({}); falling back to DeviceUploader",
                     cudaGetErrorString(err));
        return false;
    }
    device_base_ = dev;
    registered_  = true;
    readonly_    = ro;
    return true;
}

void UnifiedWeightPackPin::unpin() noexcept {
    if (registered_ && host_base_) {
        const cudaError_t err = cudaHostUnregister(const_cast<void*>(host_base_));
        if (err != cudaSuccess) {
            spdlog::error("UnifiedWeightPackPin(CUDA): cudaHostUnregister failed: {}", cudaGetErrorString(err));
        }
        registered_  = false;
        readonly_    = false;
        device_base_ = nullptr;
    }
}

void UnifiedWeightPackPin::release() noexcept {
    unpin();
    host_base_ = nullptr;
    size_      = 0;
    owner_.reset();
}

void* UnifiedWeightPackPin::tensor_device_ptr(std::uint64_t offset, std::uint64_t bytes) const noexcept {
    if (!registered_ || !device_base_ || offset > size_ || bytes > size_ - offset) { return nullptr; }
    return static_cast<std::uint8_t*>(device_base_) + offset;
}

std::uint64_t mem_available_bytes() {
    std::ifstream in("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    while (in >> key >> value) {
        std::getline(in, unit);
        if (key == "MemAvailable:") { return value * 1024; }
    }
    return 0;
}

} // namespace ingine

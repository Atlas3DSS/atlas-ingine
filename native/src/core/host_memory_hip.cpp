// Pure native AMD ROCm/HIP HostRegion and MappedHostBuffer implementation.
#include "core/host_memory.h"

#include <hip/hip_runtime.h>
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
namespace {

std::size_t align_huge(std::size_t bytes) {
  return (bytes + kHugePageBytes - 1) / kHugePageBytes * kHugePageBytes;
}

}  // namespace

HostRegion::HostRegion(std::size_t bytes) {
  if (bytes == 0) throw std::invalid_argument("HostRegion size must be nonzero");
  const std::size_t size = align_huge(bytes);
  const std::size_t span = size + kHugePageBytes;
  void* raw = ::mmap(nullptr, span, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (raw == MAP_FAILED) {
    throw std::runtime_error("HostRegion: mmap of " + std::to_string(span) + " bytes failed: " + std::strerror(errno));
  }
  const auto base = reinterpret_cast<std::uintptr_t>(raw);
  const auto aligned = (base + kHugePageBytes - 1) & ~(std::uintptr_t(kHugePageBytes) - 1);
  const std::size_t head = aligned - base;
  const std::size_t tail = span - head - size;
  if (head) ::munmap(raw, head);
  if (tail) ::munmap(reinterpret_cast<void*>(aligned + size), tail);
  data_ = reinterpret_cast<std::byte*>(aligned);
  size_ = size;
  if (::madvise(data_, size_, MADV_HUGEPAGE) != 0) {
    spdlog::warn("HostRegion: madvise(MADV_HUGEPAGE) failed: {}", std::strerror(errno));
  }
}

HostRegion::~HostRegion() { release(); }

HostRegion::HostRegion(HostRegion&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      registered_(std::exchange(other.registered_, false)),
      device_(std::exchange(other.device_, nullptr)),
      register_seconds_(other.register_seconds_) {}

HostRegion& HostRegion::operator=(HostRegion&& other) noexcept {
  if (this != &other) {
    release();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    registered_ = std::exchange(other.registered_, false);
    device_ = std::exchange(other.device_, nullptr);
    register_seconds_ = other.register_seconds_;
  }
  return *this;
}

bool HostRegion::register_with_cuda() {
  if (registered_ || !data_) return registered_;
  const auto start = std::chrono::steady_clock::now();
  hipError_t err = hipHostRegister(data_, size_, hipHostRegisterPortable | hipHostRegisterMapped);
  register_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (err != hipSuccess) {
    (void)hipGetLastError();
    spdlog::warn("HostRegion(HIP): hipHostRegister failed ({}: {}); staying pageable", hipGetErrorName(err),
                 hipGetErrorString(err));
    return false;
  }
  void* dev = nullptr;
  err = hipHostGetDevicePointer(&dev, data_, 0);
  if (err != hipSuccess) {
    (void)hipGetLastError();
    (void)hipHostUnregister(data_);
    return false;
  }
  registered_ = true;
  device_ = dev;
  return true;
}

void HostRegion::unregister() noexcept {
  if (registered_) {
    const hipError_t err = hipHostUnregister(data_);
    if (err != hipSuccess) {
      spdlog::error("HostRegion(HIP): hipHostUnregister failed: {}", hipGetErrorString(err));
    }
    registered_ = false;
    device_ = nullptr;
  }
}

void HostRegion::release() noexcept {
  unregister();
  if (data_) ::munmap(data_, size_);
  data_ = nullptr;
  size_ = 0;
}

MappedHostBuffer::MappedHostBuffer(std::size_t bytes) {
  if (bytes == 0) throw std::invalid_argument("MappedHostBuffer size must be nonzero");
  void* ptr = nullptr;
  hipError_t err = hipHostMalloc(&ptr, bytes, hipHostMallocPortable | hipHostMallocMapped);
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("hipHostMalloc(mapped) failed: ") + hipGetErrorString(err));
  }
  void* dev = nullptr;
  err = hipHostGetDevicePointer(&dev, ptr, 0);
  if (err != hipSuccess) {
    (void)hipHostFree(ptr);
    throw std::runtime_error(std::string("hipHostGetDevicePointer failed: ") + hipGetErrorString(err));
  }
  std::memset(ptr, 0, bytes);
  data_ = ptr;
  device_ = dev;
  size_ = bytes;
}

MappedHostBuffer::~MappedHostBuffer() { release(); }

MappedHostBuffer::MappedHostBuffer(MappedHostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      device_(std::exchange(other.device_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedHostBuffer& MappedHostBuffer::operator=(MappedHostBuffer&& other) noexcept {
  if (this != &other) {
    release();
    data_ = std::exchange(other.data_, nullptr);
    device_ = std::exchange(other.device_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

void MappedHostBuffer::release() noexcept {
  if (data_) {
    const hipError_t err = hipHostFree(data_);
    if (err != hipSuccess) {
      spdlog::error("MappedHostBuffer(HIP): hipHostFree failed: {}", hipGetErrorString(err));
    }
  }
  data_ = nullptr;
  device_ = nullptr;
  size_ = 0;
}

std::uint64_t mem_available_bytes() {
  std::ifstream in("/proc/meminfo");
  std::string key, unit;
  std::uint64_t value = 0;
  while (in >> key >> value) {
    std::getline(in, unit);
    if (key == "MemAvailable:") return value * 1024;
  }
  return 0;
}

}  // namespace ingine

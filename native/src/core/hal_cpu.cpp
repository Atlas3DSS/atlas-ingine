// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// DeviceBuffer, DeviceArena, and PinnedHostBuffer sections adapted from arena.cu:
// Modified for fngine from cinference:src/core/arena.cu @ 383e5db. See NOTICE.
// HostRegion and MappedHostBuffer sections adapted from host_memory.cpp (fngine, Apache-2.0).

#include "core/arena.h"
#include "core/hal.h"
#include "core/host_memory.h"

#include <spdlog/spdlog.h>

#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/mman.h>

namespace ingine {
namespace {

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::uintptr_t checked_add_uintptr(std::uintptr_t a, std::size_t b) {
  if (b > std::numeric_limits<std::uintptr_t>::max() - a) {
    throw std::overflow_error("arena address arithmetic overflow");
  }
  return a + b;
}

std::uintptr_t align_up_addr(std::uintptr_t addr, std::size_t align) {
  const std::size_t mask = align - 1;
  const std::uintptr_t with_mask = checked_add_uintptr(addr, mask);
  return with_mask & ~static_cast<std::uintptr_t>(mask);
}

struct ReleaseHooks {
  std::mutex mutex;
  std::map<int, ReleaseHook> hooks;
  int next_id = 0;
};

ReleaseHooks& release_hooks() {
  static ReleaseHooks h;
  return h;
}

std::atomic<int> g_fail_next{0};

void* alloc_aligned(std::size_t align, std::size_t bytes, const char* what) {
  for (int left = g_fail_next.load(); left > 0;) {
    if (g_fail_next.compare_exchange_weak(left, left - 1)) {
      return nullptr;
    }
  }
  void* ptr = nullptr;
  const int rc = ::posix_memalign(&ptr, align, bytes);
  if (rc != 0) {
    if (rc == ENOMEM) return nullptr;
    throw std::runtime_error(std::string(what) + " failed: " + std::strerror(rc));
  }
  return ptr;
}

std::size_t align_huge(std::size_t bytes) {
  return (bytes + kHugePageBytes - 1) / kHugePageBytes * kHugePageBytes;
}

std::uint64_t read_meminfo_kib(const char* target_key) {
  std::ifstream in("/proc/meminfo");
  std::string key;
  std::uint64_t val = 0;
  std::string unit;
  while (in >> key >> val >> unit) {
    if (key == target_key) return val * 1024ull;
  }
  return 0;
}

struct CpuStreamObj {
  std::uint64_t seq = 1;
};

struct CpuEventObj {
  bool recorded = true;
};

}  // namespace

int add_release_hook(ReleaseHook hook) {
  ReleaseHooks& h = release_hooks();
  std::lock_guard<std::mutex> lock(h.mutex);
  h.hooks.emplace(h.next_id, std::move(hook));
  return h.next_id++;
}

void remove_release_hook(int id) {
  ReleaseHooks& h = release_hooks();
  std::lock_guard<std::mutex> lock(h.mutex);
  h.hooks.erase(id);
}

void fail_next_allocations(int n) { g_fail_next = n; }

DeviceBuffer::DeviceBuffer(std::size_t size_bytes) : bytes(size_bytes) {
  if (bytes == 0) return;
  void* ptr = alloc_aligned(256, bytes, "CPU DeviceBuffer");
  if (!ptr) {
    std::vector<ReleaseHook> hooks;
    {
      ReleaseHooks& h = release_hooks();
      std::lock_guard<std::mutex> lock(h.mutex);
      for (const auto& [id, hook] : h.hooks) hooks.push_back(hook);
    }
    bool freed = false;
    for (const ReleaseHook& hook : hooks) freed |= hook();
    if (freed) ptr = alloc_aligned(256, bytes, "CPU DeviceBuffer");
  }
  if (!ptr) throw std::runtime_error("CPU DeviceBuffer allocation failed");
  p = ptr;
}

DeviceBuffer::~DeviceBuffer() {
  std::free(p);
  p = nullptr;
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), bytes(other.bytes) {
  other.p = nullptr;
  other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this == &other) return *this;
  std::free(p);
  p = other.p;
  bytes = other.bytes;
  other.p = nullptr;
  other.bytes = 0;
  return *this;
}

void DeviceBuffer::fill(int byte_value) {
  if (bytes == 0) return;
  std::memset(p, byte_value, bytes);
}

void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t byte_offset) {
  require_range(byte_offset, count, "host-to-device copy");
  if (count == 0) return;
  std::memcpy(static_cast<std::uint8_t*>(p) + byte_offset, source, count);
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count, std::size_t byte_offset) const {
  require_range(byte_offset, count, "device-to-host copy");
  if (count == 0) return;
  std::memcpy(destination, static_cast<const std::uint8_t*>(p) + byte_offset, count);
}

void DeviceBuffer::require_range(std::size_t byte_offset, std::size_t count, const char* operation) const {
  if (byte_offset > bytes || count > bytes - byte_offset) {
    throw std::out_of_range(std::string(operation) + " exceeds device buffer");
  }
}

DeviceArena::Scope::Scope(DeviceArena& arena) noexcept : arena_(&arena), saved_offset_(arena.off_) {}

DeviceArena::Scope::~Scope() noexcept {
  if (arena_ != nullptr && saved_offset_ <= arena_->off_) arena_->off_ = saved_offset_;
}

DeviceArena::Scope::Scope(Scope&& other) noexcept : arena_(other.arena_), saved_offset_(other.saved_offset_) {
  other.arena_ = nullptr;
}

DeviceArena::DeviceArena(std::size_t capacity_bytes) {
  if (capacity_bytes == 0) throw std::invalid_argument("DeviceArena capacity must be nonzero");
  void* ptr = alloc_aligned(256, capacity_bytes, "CPU DeviceArena");
  if (!ptr) throw std::runtime_error("CPU DeviceArena allocation failed");
  base_ = ptr;
  cap_ = capacity_bytes;
  off_ = 0;
}

DeviceArena::DeviceArena(DeviceSpan storage) : base_(storage.data), cap_(storage.bytes), owns_(false) {
  if (base_ == nullptr || cap_ == 0) throw std::invalid_argument("borrowed DeviceArena storage must be non-empty");
}

DeviceArena::~DeviceArena() {
  if (owns_) std::free(base_);
}

DeviceArena::DeviceArena(DeviceArena&& other) noexcept
    : base_(other.base_), cap_(other.cap_), off_(other.off_), peak_(other.peak_), owns_(other.owns_) {
  other.base_ = nullptr;
  other.cap_ = 0;
  other.off_ = 0;
  other.peak_ = 0;
  other.owns_ = true;
}

DeviceArena& DeviceArena::operator=(DeviceArena&& other) noexcept {
  if (this == &other) return *this;
  if (owns_) std::free(base_);
  base_ = other.base_;
  cap_ = other.cap_;
  off_ = other.off_;
  peak_ = other.peak_;
  owns_ = other.owns_;
  other.base_ = nullptr;
  other.cap_ = 0;
  other.off_ = 0;
  other.peak_ = 0;
  other.owns_ = true;
  return *this;
}

DeviceSpan DeviceArena::alloc_bytes(std::size_t bytes, std::size_t align) {
  if (base_ == nullptr) throw std::runtime_error("DeviceArena has no backing allocation");
  if (!is_power_of_two(align)) throw std::invalid_argument("arena alignment must be a nonzero power of two");
  if (bytes == 0) throw std::invalid_argument("arena allocation must be nonzero");

  const std::uintptr_t base_addr = reinterpret_cast<std::uintptr_t>(base_);
  const std::uintptr_t current_addr = checked_add_uintptr(base_addr, off_);
  const std::uintptr_t aligned_addr = align_up_addr(current_addr, align);
  const std::uintptr_t aligned_offset_addr = aligned_addr - base_addr;
  if (aligned_offset_addr > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error("arena aligned offset overflows size_t");
  }
  const auto aligned_offset = static_cast<std::size_t>(aligned_offset_addr);
  if (bytes > std::numeric_limits<std::size_t>::max() - aligned_offset) {
    throw std::overflow_error("arena allocation end offset overflows size_t");
  }
  const std::size_t end = aligned_offset + bytes;
  if (end > cap_) throw std::bad_alloc();

  auto* ptr = static_cast<unsigned char*>(base_) + aligned_offset;
  off_ = end;
  if (off_ > peak_) peak_ = off_;
  return DeviceSpan{ptr, bytes};
}

Tensor DeviceArena::alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t align) {
  Tensor view(nullptr, dtype, shape);
  const DeviceSpan storage = alloc_bytes(view.bytes(), align);
  return Tensor(storage.data, dtype, shape);
}

DeviceArena::Scope DeviceArena::scope() noexcept { return Scope(*this); }
void DeviceArena::reset() noexcept { off_ = 0; }
void* DeviceArena::base() const noexcept { return base_; }
std::size_t DeviceArena::used() const noexcept { return off_; }
std::size_t DeviceArena::capacity() const noexcept { return cap_; }
std::size_t DeviceArena::peak_used() const noexcept { return peak_; }
void DeviceArena::reset_peak() noexcept { peak_ = off_; }

PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes) {
  if (size_bytes == 0) throw std::invalid_argument("PinnedHostBuffer size must be nonzero");
  void* ptr = alloc_aligned(4096, size_bytes, "CPU PinnedHostBuffer");
  if (!ptr) throw std::runtime_error("CPU PinnedHostBuffer allocation failed");
  data_ = ptr;
  size_ = size_bytes;
}

PinnedHostBuffer::~PinnedHostBuffer() {
  std::free(data_);
  data_ = nullptr;
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
  if (this == &other) return *this;
  std::free(data_);
  data_ = other.data_;
  size_ = other.size_;
  other.data_ = nullptr;
  other.size_ = 0;
  return *this;
}

void* PinnedHostBuffer::data() const noexcept { return data_; }
std::size_t PinnedHostBuffer::size() const noexcept { return size_; }

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

bool HostRegion::register_with_cuda() { return false; }
void HostRegion::unregister() noexcept {
  registered_ = false;
  device_ = nullptr;
}

void HostRegion::release() noexcept {
  unregister();
  if (data_ != nullptr) {
    if (::munmap(data_, size_) != 0) {
      spdlog::error("HostRegion: munmap failed: {}", std::strerror(errno));
    }
    data_ = nullptr;
    size_ = 0;
  }
}

MappedHostBuffer::MappedHostBuffer(std::size_t bytes) {
  if (bytes == 0) throw std::invalid_argument("MappedHostBuffer size must be nonzero");
  void* ptr = alloc_aligned(4096, bytes, "CPU MappedHostBuffer");
  if (!ptr) throw std::runtime_error("CPU MappedHostBuffer allocation failed");
  std::memset(ptr, 0, bytes);
  data_ = ptr;
  device_ = ptr;
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
  if (data_ != nullptr) {
    std::free(data_);
    data_ = nullptr;
    device_ = nullptr;
    size_ = 0;
  }
}

std::uint64_t mem_available_bytes() { return read_meminfo_kib("MemAvailable:"); }

namespace hal {

Stream::Stream(int device_id) : device_id_(device_id) {
  if (device_id != 0) throw std::invalid_argument("CPU HAL only supports device_id 0");
  handle_ = StreamHandle::from_native(new CpuStreamObj{});
}

Stream::~Stream() { release(); }

Stream::Stream(Stream&& other) noexcept
    : device_id_(other.device_id_), handle_(std::exchange(other.handle_, StreamHandle{})) {}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this != &other) {
    release();
    device_id_ = other.device_id_;
    handle_ = std::exchange(other.handle_, StreamHandle{});
  }
  return *this;
}

void Stream::release() noexcept {
  delete static_cast<CpuStreamObj*>(handle_.native);
  handle_ = {};
}

void Stream::synchronize() const {
  if (!handle_) throw std::logic_error("hal::Stream::synchronize on empty stream");
  stream_synchronize(handle_);
}

Event::Event(int device_id) : device_id_(device_id) {
  if (device_id != 0) throw std::invalid_argument("CPU HAL only supports device_id 0");
  handle_ = EventHandle::from_native(new CpuEventObj{});
}

Event::~Event() { release(); }

Event::Event(Event&& other) noexcept
    : device_id_(other.device_id_), handle_(std::exchange(other.handle_, EventHandle{})) {}

Event& Event::operator=(Event&& other) noexcept {
  if (this != &other) {
    release();
    device_id_ = other.device_id_;
    handle_ = std::exchange(other.handle_, EventHandle{});
  }
  return *this;
}

void Event::release() noexcept {
  delete static_cast<CpuEventObj*>(handle_.native);
  handle_ = {};
}

void Event::record(StreamHandle /*stream*/) {
  if (!handle_) throw std::logic_error("hal::Event::record on empty event");
  static_cast<CpuEventObj*>(handle_.native)->recorded = true;
}

void Event::wait(StreamHandle /*stream*/) const {
  if (!handle_) throw std::logic_error("hal::Event::wait on empty event");
}

bool Event::ready() const {
  if (!handle_) throw std::logic_error("hal::Event::ready on empty event");
  return static_cast<const CpuEventObj*>(handle_.native)->recorded;
}

void Event::synchronize() const {
  if (!handle_) throw std::logic_error("hal::Event::synchronize on empty event");
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

BackendKind active_backend() noexcept { return BackendKind::Cpu; }

const char* backend_name(BackendKind kind) noexcept {
  switch (kind) {
    case BackendKind::Cpu: return "cpu";
    case BackendKind::Cuda: return "cuda";
    case BackendKind::Hip: return "hip";
    case BackendKind::Mlx: return "mlx";
    case BackendKind::Sycl: return "sycl";
  }
  return "unknown";
}

bool conditional_graphs_enabled(int /*device_id*/) { return false; }

DeviceCaps query_device_caps(int device_id) {
  if (device_id != 0) throw std::invalid_argument("CPU HAL only supports device_id 0");
  DeviceCaps caps;
  caps.backend = BackendKind::Cpu;
  caps.primary_space = MemorySpace::Host;
  caps.name = "cpu-host";
  caps.device_id = 0;
  caps.arch_major = 0;
  caps.arch_minor = 0;
  caps.subgroup = SubgroupCaps{1, 1, 1};
  caps.max_shared_mem_bytes = 0;
  caps.supports_bf16 = true;
  caps.supports_fp16 = true;
  caps.supports_fp8_e4m3 = false;
  caps.supports_graph_capture = false;
  caps.supports_conditional_graph = false;
  caps.supports_pdl = false;
  caps.unified_memory = true;
  return caps;
}

MemoryInfo query_memory_info() {
  return MemoryInfo{
      static_cast<std::size_t>(read_meminfo_kib("MemAvailable:")),
      static_cast<std::size_t>(read_meminfo_kib("MemTotal:")),
  };
}

bool is_stream_capturing(StreamHandle /*stream*/) { return false; }

bool graph_requirement_met(const DeviceCaps& caps, GraphRequirement requirement) noexcept {
  switch (requirement) {
    case GraphRequirement::Capture: return caps.supports_graph_capture;
    case GraphRequirement::ConditionalSwitch: return caps.supports_graph_capture && caps.supports_conditional_graph;
  }
  return false;
}

void require_graph_capture_or_eager(const DeviceCaps& caps, bool allow_eager, const char* caller,
                                    GraphRequirement requirement) {
  if (allow_eager || graph_requirement_met(caps, requirement)) return;
  const char* tag = caller ? caller : "hal";
  if (!caps.supports_graph_capture) {
    throw std::runtime_error(std::string(tag) + ": graph capture unsupported on backend " +
                             backend_name(caps.backend));
  }
  throw std::runtime_error(std::string(tag) + ": conditional SWITCH graph unsupported on backend " +
                           backend_name(caps.backend));
}

void require_conditional_switch_or_eager(const DeviceCaps& caps, bool allow_eager, const char* caller) {
  require_graph_capture_or_eager(caps, allow_eager, caller, GraphRequirement::ConditionalSwitch);
}

void memcpy_h2d_async(void* dst_dev, const void* src_host, std::size_t bytes, StreamHandle /*stream*/) {
  if (bytes == 0) return;
  if (!dst_dev || !src_host) throw std::invalid_argument("hal::memcpy_h2d_async: null pointer");
  std::memcpy(dst_dev, src_host, bytes);
}

void memcpy_d2h_async(void* dst_host, const void* src_dev, std::size_t bytes, StreamHandle /*stream*/) {
  if (bytes == 0) return;
  if (!dst_host || !src_dev) throw std::invalid_argument("hal::memcpy_d2h_async: null pointer");
  std::memcpy(dst_host, src_dev, bytes);
}

void memcpy_d2d_async(void* dst_dev, const void* src_dev, std::size_t bytes, StreamHandle /*stream*/) {
  if (bytes == 0) return;
  if (!dst_dev || !src_dev) throw std::invalid_argument("hal::memcpy_d2d_async: null pointer");
  std::memcpy(dst_dev, src_dev, bytes);
}

void memset_async(void* dst_dev, int byte_value, std::size_t bytes, StreamHandle /*stream*/) {
  if (bytes == 0) return;
  if (!dst_dev) throw std::invalid_argument("hal::memset_async: null pointer");
  std::memset(dst_dev, byte_value, bytes);
}

void stream_synchronize(StreamHandle /*stream*/) {
  std::atomic_thread_fence(std::memory_order_seq_cst);
}

}  // namespace hal
}  // namespace ingine

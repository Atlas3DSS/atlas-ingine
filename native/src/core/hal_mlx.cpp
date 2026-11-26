// Pure native C++20 + Objective-C++ Apple MLX/Metal Hardware Abstraction Layer & unified memory manager.
// Zero Python runtime shims and zero external ggml runtime dependencies.
// Compiled on macOS/Apple Silicon as Objective-C++20 when INGINE_BACKEND=MLX.
#include "core/arena.h"
#include "core/hal.h"
#include "core/host_memory.h"
#include "core/vendor_policy.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#if defined(INGINE_HAVE_LIBMLX) && INGINE_HAVE_LIBMLX
#include <mlx/mlx.h>
#endif

#include <spdlog/spdlog.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ingine {

id<MTLDevice> shared_mtl_device() {
  static id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) throw std::runtime_error("HAL(MLX/Metal): no Metal GPU device available");
  return dev;
}

namespace {

std::size_t query_apple_physical_ram() noexcept {
  std::uint64_t memsize = 0;
  std::size_t len = sizeof(memsize);
  if (::sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0) {
    return static_cast<std::size_t>(memsize);
  }
  return 0;
}

struct MtlStreamState {
  id<MTLCommandQueue> queue = nil;
};

struct MtlEventState {
  id<MTLSharedEvent> shared_event = nil;
  std::uint64_t value = 0;
};

id<MTLCommandQueue> default_mtl_queue() {
  static std::mutex q_mutex;
  static id<MTLCommandQueue> default_q = nil;
  std::lock_guard<std::mutex> lock(q_mutex);
  if (!default_q) {
    default_q = [shared_mtl_device() newCommandQueue];
    if (!default_q) throw std::runtime_error("HAL(MLX/Metal): failed to create default MTLCommandQueue");
  }
  return default_q;
}

struct AllocTable {
  std::mutex mutex;
  std::unordered_map<const void*, id<MTLBuffer>> buffers;
};

AllocTable& alloc_table() {
  static AllocTable t;
  return t;
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

void* alloc_metal_shared(std::size_t bytes) {
  for (int left = g_fail_next.load(); left > 0;) {
    if (g_fail_next.compare_exchange_weak(left, left - 1)) return nullptr;
  }
  const std::size_t aligned = vendor::mlx::align_page_up(bytes, 16384u);
  @autoreleasepool {
    id<MTLBuffer> buf = [shared_mtl_device() newBufferWithLength:aligned options:MTLResourceStorageModeShared];
    if (!buf) return nullptr;
    void* ptr = [buf contents];
    if (!ptr) return nullptr;
    std::lock_guard<std::mutex> lock(alloc_table().mutex);
    alloc_table().buffers.emplace(ptr, buf);
    return ptr;
  }
}

void free_metal_shared(void*& ptr) noexcept {
  if (!ptr) return;
  @autoreleasepool {
    std::lock_guard<std::mutex> lock(alloc_table().mutex);
    alloc_table().buffers.erase(ptr);
  }
  ptr = nullptr;
}

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

}  // namespace

id<MTLCommandQueue> metal_queue_for_stream(hal::StreamHandle stream) {
  if (!stream.native) return default_mtl_queue();
  auto* st = static_cast<MtlStreamState*>(stream.native);
  if (!st->queue) throw std::invalid_argument("HAL(MLX/Metal): invalid stream handle without MTLCommandQueue");
  return st->queue;
}

id<MTLBuffer> lookup_metal_buffer_for_range(const void* ptr, std::size_t bytes, std::size_t& out_offset,
                                            const char* caller) {
  if (!ptr) return nil;
  std::lock_guard<std::mutex> lock(alloc_table().mutex);
  const auto addr = reinterpret_cast<std::uintptr_t>(ptr);
  std::vector<vendor::mlx::RegisteredRange> spans;
  std::vector<id<MTLBuffer>> bufs;
  spans.reserve(alloc_table().buffers.size());
  bufs.reserve(alloc_table().buffers.size());
  for (const auto& [base_ptr, buf] : alloc_table().buffers) {
    spans.push_back(vendor::mlx::RegisteredRange{reinterpret_cast<std::uintptr_t>(base_ptr),
                                                 static_cast<std::size_t>([buf length])});
    bufs.push_back(buf);
  }
  const vendor::mlx::BufferRangeLookup res = vendor::mlx::classify_registered_range(spans, addr, bytes);
  if (res.status == vendor::mlx::BufferRangeMatch::Valid) {
    out_offset = res.offset;
    return bufs[res.index];
  }
  if (res.status == vendor::mlx::BufferRangeMatch::OutOfBounds) {
    throw std::out_of_range(std::string(caller ? caller : "HAL(MLX/Metal)") +
                            ": requested byte range exceeds registered MTLBuffer bounds (offset=" +
                            std::to_string(res.offset) + ", bytes=" + std::to_string(bytes) +
                            ", length=" + std::to_string(spans[res.index].length) + ")");
  }
  return nil;
}

int add_release_hook(ReleaseHook hook) {
  std::lock_guard<std::mutex> lock(release_hooks().mutex);
  release_hooks().hooks.emplace(release_hooks().next_id, std::move(hook));
  return release_hooks().next_id++;
}

void remove_release_hook(int id) {
  std::lock_guard<std::mutex> lock(release_hooks().mutex);
  release_hooks().hooks.erase(id);
}

void fail_next_allocations(int n) { g_fail_next = n; }

std::atomic<int> g_fail_next_pinned{0};

void fail_next_pinned_allocations(int n) { g_fail_next_pinned = n; }

DeviceBuffer::DeviceBuffer(std::size_t size_bytes) : bytes(size_bytes) {
  if (bytes == 0) return;
  void* ptr = alloc_metal_shared(bytes);
  if (!ptr) {
    std::vector<ReleaseHook> hooks;
    {
      std::lock_guard<std::mutex> lock(release_hooks().mutex);
      for (const auto& [id, hook] : release_hooks().hooks) hooks.push_back(hook);
    }
    bool freed = false;
    for (const ReleaseHook& hook : hooks) freed |= hook();
    if (freed) ptr = alloc_metal_shared(bytes);
  }
  if (!ptr) throw std::runtime_error("HAL(MLX/Metal): shared buffer allocation failed");
  p = ptr;
}

DeviceBuffer::~DeviceBuffer() { free_metal_shared(p); }

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), bytes(other.bytes) {
  other.p = nullptr;
  other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this == &other) return *this;
  free_metal_shared(p);
  p = other.p;
  bytes = other.bytes;
  other.p = nullptr;
  other.bytes = 0;
  return *this;
}

void DeviceBuffer::fill(int byte_value) {
  if (bytes > 0) {
    hal::stream_synchronize(hal::StreamHandle{});
    std::memset(p, byte_value, bytes);
  }
}

void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t byte_offset) {
  require_range(byte_offset, count, "host-to-device copy");
  if (count > 0) {
    hal::stream_synchronize(hal::StreamHandle{});
    std::memcpy(static_cast<std::uint8_t*>(p) + byte_offset, source, count);
  }
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count, std::size_t byte_offset) const {
  require_range(byte_offset, count, "device-to-host copy");
  if (count > 0) {
    hal::stream_synchronize(hal::StreamHandle{});
    std::memcpy(destination, static_cast<const std::uint8_t*>(p) + byte_offset, count);
  }
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
  void* ptr = alloc_metal_shared(capacity_bytes);
  if (!ptr) throw std::runtime_error("HAL(MLX/Metal): DeviceArena allocation failed");
  base_ = ptr;
  cap_ = capacity_bytes;
}

DeviceArena::DeviceArena(DeviceSpan storage) : base_(storage.data), cap_(storage.bytes), owns_(false) {
  if (base_ == nullptr || cap_ == 0) throw std::invalid_argument("borrowed DeviceArena storage must be non-empty");
}

DeviceArena::~DeviceArena() {
  if (owns_) free_metal_shared(base_);
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
  if (owns_) free_metal_shared(base_);
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
  if (!base_) throw std::runtime_error("DeviceArena has no backing allocation");
  if (!is_power_of_two(align)) throw std::invalid_argument("arena alignment must be power of two");
  if (bytes == 0) throw std::invalid_argument("arena allocation must be nonzero");
  const auto base_addr = reinterpret_cast<std::uintptr_t>(base_);
  const std::uintptr_t mask = align - 1;
  const std::uintptr_t aligned = (base_addr + off_ + mask) & ~mask;
  const auto aligned_off = static_cast<std::size_t>(aligned - base_addr);
  const std::size_t end = aligned_off + bytes;
  if (end > cap_) throw std::bad_alloc();
  off_ = end;
  if (off_ > peak_) peak_ = off_;
  return DeviceSpan{static_cast<unsigned char*>(base_) + aligned_off, bytes};
}

Tensor DeviceArena::alloc(DType dtype, std::initializer_list<std::int32_t> shape, std::size_t align) {
  Tensor view(nullptr, dtype, shape);
  return Tensor(alloc_bytes(view.bytes(), align).data, dtype, shape);
}

DeviceArena::Scope DeviceArena::scope() noexcept { return Scope(*this); }
void DeviceArena::reset() noexcept { off_ = 0; }
void* DeviceArena::base() const noexcept { return base_; }
std::size_t DeviceArena::used() const noexcept { return off_; }
std::size_t DeviceArena::capacity() const noexcept { return cap_; }
std::size_t DeviceArena::peak_used() const noexcept { return peak_; }
void DeviceArena::reset_peak() noexcept { peak_ = off_; }

PinnedHostBuffer::PinnedHostBuffer(std::size_t size_bytes) : size_(size_bytes) {
  if (size_bytes == 0) throw std::invalid_argument("PinnedHostBuffer size must be nonzero");
  for (int left = g_fail_next_pinned.load(); left > 0;) {
    if (g_fail_next_pinned.compare_exchange_weak(left, left - 1)) {
      throw std::runtime_error("HAL(MLX/Metal): injected pinned allocation failure");
    }
  }
  void* ptr = alloc_metal_shared(size_bytes);
  if (!ptr) throw std::runtime_error("HAL(MLX/Metal): PinnedHostBuffer allocation failed");
  data_ = ptr;
}

PinnedHostBuffer::~PinnedHostBuffer() { free_metal_shared(data_); }

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
  if (this == &other) return *this;
  free_metal_shared(data_);
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
  const std::size_t size = vendor::mlx::align_page_up(bytes, kHugePageBytes);
  void* raw = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (raw == MAP_FAILED) throw std::runtime_error("HostRegion(MLX): mmap failed");
  data_ = static_cast<std::byte*>(raw);
  size_ = size;
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
  if (!data_) return false;
  registered_ = true;
  device_ = data_;  // Apple Unified Memory shares host/GPU pointer semantics
  return true;
}

void HostRegion::unregister() noexcept {
  registered_ = false;
  device_ = nullptr;
}

void HostRegion::release() noexcept {
  unregister();
  if (data_) {
    ::munmap(data_, size_);
    data_ = nullptr;
    size_ = 0;
  }
}

MappedHostBuffer::MappedHostBuffer(std::size_t bytes) : size_(bytes) {
  if (bytes == 0) throw std::invalid_argument("MappedHostBuffer size must be nonzero");
  void* ptr = alloc_metal_shared(bytes);
  if (!ptr) throw std::runtime_error("HAL(MLX/Metal): MappedHostBuffer allocation failed");
  std::memset(ptr, 0, bytes);
  data_ = ptr;
  device_ = ptr;
}

MappedHostBuffer::~MappedHostBuffer() { free_metal_shared(data_); device_ = nullptr; }

MappedHostBuffer::MappedHostBuffer(MappedHostBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      device_(std::exchange(other.device_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedHostBuffer& MappedHostBuffer::operator=(MappedHostBuffer&& other) noexcept {
  if (this != &other) {
    free_metal_shared(data_);
    data_ = std::exchange(other.data_, nullptr);
    device_ = std::exchange(other.device_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

void MappedHostBuffer::release() noexcept {
  free_metal_shared(data_);
  device_ = nullptr;
  size_ = 0;
}

std::uint64_t mem_available_bytes() {
  return static_cast<std::uint64_t>(query_apple_physical_ram());
}

namespace hal {

Stream::Stream(int device_id) : device_id_(device_id) {
  if (device_id != 0) throw std::invalid_argument("HAL(MLX/Metal): only device_id 0 is supported");
  @autoreleasepool {
    auto* state = new MtlStreamState{};
    state->queue = [shared_mtl_device() newCommandQueue];
    if (!state->queue) {
      delete state;
      throw std::runtime_error("HAL(MLX/Metal): failed to create MTLCommandQueue");
    }
    handle_ = StreamHandle::from_native(state);
  }
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
  delete static_cast<MtlStreamState*>(handle_.native);
  handle_ = {};
}

void Stream::synchronize() const {
  if (!handle_) throw std::logic_error("hal::Stream::synchronize on empty stream");
  stream_synchronize(handle_);
}

Event::Event(int device_id) : device_id_(device_id) {
  if (device_id != 0) throw std::invalid_argument("HAL(MLX/Metal): only device_id 0 is supported");
  @autoreleasepool {
    auto* ev = new MtlEventState{};
    ev->shared_event = [shared_mtl_device() newSharedEvent];
    handle_ = EventHandle::from_native(ev);
  }
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
  delete static_cast<MtlEventState*>(handle_.native);
  handle_ = {};
}

void Event::record(StreamHandle stream) {
  if (!handle_) throw std::logic_error("hal::Event::record on empty event");
  auto* ev = static_cast<MtlEventState*>(handle_.native);
  id<MTLCommandQueue> q = metal_queue_for_stream(stream);
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    if (!cb) throw std::runtime_error("hal::Event::record(MLX): failed to create MTLCommandBuffer");
    const std::uint64_t next_val = ++ev->value;
    [cb encodeSignalEvent:ev->shared_event value:next_val];
    [cb commit];
  }
}

void Event::wait(StreamHandle stream) const {
  if (!handle_) throw std::logic_error("hal::Event::wait on empty event");
  auto* ev = static_cast<const MtlEventState*>(handle_.native);
  id<MTLCommandQueue> q = metal_queue_for_stream(stream);
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    if (!cb) throw std::runtime_error("hal::Event::wait(MLX): failed to create MTLCommandBuffer");
    [cb encodeWaitForEvent:ev->shared_event value:ev->value];
    [cb commit];
  }
}

bool Event::ready() const {
  if (!handle_) throw std::logic_error("hal::Event::ready on empty event");
  const auto* ev = static_cast<const MtlEventState*>(handle_.native);
  return ev->shared_event.signaledValue >= ev->value;
}

void Event::synchronize() const {
  if (!handle_) throw std::logic_error("hal::Event::synchronize on empty event");
  const auto* ev = static_cast<const MtlEventState*>(handle_.native);
  while (ev->shared_event.signaledValue < ev->value) {
    ::usleep(50);
  }
}

BackendKind active_backend() noexcept { return BackendKind::Mlx; }

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
  if (device_id != 0) throw std::invalid_argument("HAL(MLX/Metal): invalid device id");
  @autoreleasepool {
    id<MTLDevice> dev = shared_mtl_device();
    const std::size_t ram = query_apple_physical_ram();
    const std::size_t rec_ws = static_cast<std::size_t>(dev.recommendedMaxWorkingSetSize);
    // Size default streaming budget around our ~9 GiB EXL3 weight pack.
    const auto plan = vendor::mlx::plan_streaming_budget(ram, rec_ws, 9ull << 30, false);

    DeviceCaps caps;
    caps.backend = BackendKind::Mlx;
    caps.primary_space = MemorySpace::Unified;
    caps.name = std::string(dev.name.UTF8String ? dev.name.UTF8String : "apple-metal");
    caps.arch_name = "apple-metal";
    caps.device_id = 0;
    caps.subgroup = SubgroupCaps{32, 32, 32};
    caps.max_shared_mem_bytes = static_cast<std::size_t>(dev.maxThreadgroupMemoryLength);
    caps.recommended_working_set_bytes = rec_ws;
    caps.streaming_ring_budget_bytes = plan.staging_ring_bytes;
    caps.supports_bf16 = [dev supportsFamily:MTLGPUFamilyApple9];
    caps.supports_fp16 = true;
    caps.supports_fp8_e4m3 = false;
    caps.supports_graph_capture = false;
    caps.supports_conditional_graph = false;
    caps.supports_pdl = false;
    caps.supports_wmma_or_xmx = [dev supportsFamily:MTLGPUFamilyApple7];
    caps.prefers_direct_device_alloc = false;
    caps.unified_memory = dev.hasUnifiedMemory;
    return caps;
  }
}

MemoryInfo query_memory_info() {
  const std::size_t ram = query_apple_physical_ram();
  const std::size_t rec = static_cast<std::size_t>(shared_mtl_device().recommendedMaxWorkingSetSize);
  return MemoryInfo{rec > 0 ? rec : ram, ram};
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

void memcpy_h2d_async(void* dst_dev, const void* src_host, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  if (!dst_dev || !src_host) throw std::invalid_argument("hal::memcpy_h2d_async(MLX): null pointer");
  std::size_t dst_off = 0, src_off = 0;
  (void)lookup_metal_buffer_for_range(dst_dev, bytes, dst_off, "hal::memcpy_h2d_async(MLX) dst");
  (void)lookup_metal_buffer_for_range(src_host, bytes, src_off, "hal::memcpy_h2d_async(MLX) src");
  // Drain prior GPU work on `stream` so unified-memory write does not race earlier commands and is visible
  // before any subsequent command committed on `stream`.
  stream_synchronize(stream);
  std::memcpy(dst_dev, src_host, bytes);
}

void memcpy_d2h_async(void* dst_host, const void* src_dev, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  if (!dst_host || !src_dev) throw std::invalid_argument("hal::memcpy_d2h_async(MLX): null pointer");
  std::size_t dst_off = 0, src_off = 0;
  (void)lookup_metal_buffer_for_range(src_dev, bytes, src_off, "hal::memcpy_d2h_async(MLX) src");
  (void)lookup_metal_buffer_for_range(dst_host, bytes, dst_off, "hal::memcpy_d2h_async(MLX) dst");
  stream_synchronize(stream);
  std::memcpy(dst_host, src_dev, bytes);
}

void memcpy_d2d_async(void* dst_dev, const void* src_dev, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  if (!dst_dev || !src_dev) throw std::invalid_argument("hal::memcpy_d2d_async(MLX): null pointer");
  std::size_t src_off = 0, dst_off = 0;
  id<MTLBuffer> b_src = lookup_metal_buffer_for_range(src_dev, bytes, src_off, "hal::memcpy_d2d_async(MLX) src");
  id<MTLBuffer> b_dst = lookup_metal_buffer_for_range(dst_dev, bytes, dst_off, "hal::memcpy_d2d_async(MLX) dst");
  if (b_src && b_dst && vendor::mlx::can_async_blit_d2d(b_src == b_dst, src_off, dst_off, bytes)) {
    id<MTLCommandQueue> q = metal_queue_for_stream(stream);
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit copyFromBuffer:b_src sourceOffset:src_off toBuffer:b_dst destinationOffset:dst_off size:bytes];
      [blit endEncoding];
      [cb commit];
    }
    return;
  }
  stream_synchronize(stream);
  std::memmove(dst_dev, src_dev, bytes);
}

void memset_async(void* dst_dev, int byte_value, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  if (!dst_dev) throw std::invalid_argument("hal::memset_async(MLX): null pointer");
  std::size_t dst_off = 0;
  id<MTLBuffer> b_dst = lookup_metal_buffer_for_range(dst_dev, bytes, dst_off, "hal::memset_async(MLX) dst");
  if (b_dst) {
    id<MTLCommandQueue> q = metal_queue_for_stream(stream);
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
      [blit fillBuffer:b_dst range:NSMakeRange(dst_off, bytes) value:static_cast<std::uint8_t>(byte_value)];
      [blit endEncoding];
      [cb commit];
    }
    return;
  }
  stream_synchronize(stream);
  std::memset(dst_dev, byte_value, bytes);
}

void stream_synchronize(StreamHandle stream) {
  id<MTLCommandQueue> q = metal_queue_for_stream(stream);
  @autoreleasepool {
    id<MTLCommandBuffer> barrier = [q commandBuffer];
    if (!barrier) throw std::runtime_error("hal::stream_synchronize(MLX): failed to create MTLCommandBuffer");
    [barrier commit];
    [barrier waitUntilCompleted];
  }
}

}  // namespace hal
}  // namespace ingine

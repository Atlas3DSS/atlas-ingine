// Pure native C++20 Intel SYCL / oneAPI Hardware Abstraction Layer & USM memory manager.
// Supports in-order sycl::queue streams, USM device/host buffers, Intel Arc/Data Center GPU subgroup
// capability reporting (widths 8/16/32), Level Zero direct-alloc preference, and DNNL_GPU_VENDOR=INTEL guard.
#include "core/arena.h"
#include "core/hal.h"
#include "core/host_memory.h"
#include "core/vendor_policy.h"
#include "core/weight_pack.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
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
#include <sys/mman.h>
#include <utility>
#include <vector>

namespace ingine {
namespace {

std::string to_lower_ascii(std::string s) {
  for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

std::vector<sycl::device> enumerate_sycl_devices() {
  static const bool guard_applied = [] {
    if (std::getenv("DNNL_GPU_VENDOR") == nullptr) {
      ::setenv("DNNL_GPU_VENDOR", "INTEL", 0);
    }
    return true;
  }();
  (void)guard_applied;
  const auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (gpus.empty()) return {};
  std::vector<std::string> lower_vendors;
  lower_vendors.reserve(gpus.size());
  std::vector<vendor::sycl::SyclDeviceCandidate> cands;
  cands.reserve(gpus.size());
  for (const auto& dev : gpus) {
    lower_vendors.push_back(to_lower_ascii(dev.get_info<sycl::info::device::vendor>()));
    const auto vid = static_cast<std::uint32_t>(dev.get_info<sycl::info::device::vendor_id>());
    const bool is_l0 = (dev.get_backend() == sycl::backend::ext_oneapi_level_zero);
    cands.push_back(vendor::sycl::SyclDeviceCandidate{dev.is_gpu(), is_l0, lower_vendors.back(), vid});
  }
  const std::vector<std::size_t> indices = vendor::sycl::select_intel_gpu_indices(cands);
  std::vector<sycl::device> out;
  out.reserve(indices.size());
  for (const std::size_t idx : indices) out.push_back(gpus[idx]);
  return out;
}

const sycl::device& get_sycl_device(int device_id) {
  static const std::vector<sycl::device> cached = enumerate_sycl_devices();
  if (cached.empty()) throw std::runtime_error("HAL(SYCL): no Intel GPU SYCL devices available");
  if (device_id < 0 || static_cast<std::size_t>(device_id) >= cached.size()) {
    throw std::invalid_argument("HAL(SYCL): invalid Intel GPU device id " + std::to_string(device_id));
  }
  return cached[static_cast<std::size_t>(device_id)];
}

sycl::queue& default_device_queue(int device_id = 0) {
  static std::mutex q_mutex;
  static std::map<int, sycl::queue> queues;
  std::lock_guard<std::mutex> lock(q_mutex);
  auto it = queues.find(device_id);
  if (it == queues.end()) {
    it = queues.emplace(device_id, sycl::queue{get_sycl_device(device_id), sycl::property::queue::in_order{}}).first;
  }
  return it->second;
}

sycl::queue* unwrap_queue(hal::StreamHandle stream) {
  return stream.native ? static_cast<sycl::queue*>(stream.native) : &default_device_queue(0);
}

}  // namespace

const sycl::device& sycl_device_for_id(int device_id) { return get_sycl_device(device_id); }

sycl::queue* sycl_queue_for_stream(hal::StreamHandle stream) { return unwrap_queue(stream); }

namespace {

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

void* try_usm_device(std::size_t bytes) {
  for (int left = g_fail_next.load(); left > 0;) {
    if (g_fail_next.compare_exchange_weak(left, left - 1)) return nullptr;
  }
  sycl::queue& q = default_device_queue(0);
  return sycl::aligned_alloc_device(256, bytes, q);
}

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

std::uint64_t read_meminfo_kib(const char* key_target) {
  std::ifstream in("/proc/meminfo");
  std::string key, unit;
  std::uint64_t val = 0;
  while (in >> key >> val >> unit) {
    if (key == key_target) return val * 1024ull;
  }
  return 0;
}

}  // namespace

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
  void* ptr = try_usm_device(bytes);
  if (!ptr) {
    std::vector<ReleaseHook> hooks;
    {
      std::lock_guard<std::mutex> lock(release_hooks().mutex);
      for (const auto& [id, hook] : release_hooks().hooks) hooks.push_back(hook);
    }
    bool freed = false;
    for (const ReleaseHook& hook : hooks) freed |= hook();
    if (freed) ptr = try_usm_device(bytes);
  }
  if (!ptr) throw std::runtime_error("HAL(SYCL): aligned_alloc_device failed");
  p = ptr;
}

DeviceBuffer::~DeviceBuffer() {
  if (p) {
    sycl::free(p, default_device_queue(0));
    p = nullptr;
  }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : p(other.p), bytes(other.bytes) {
  other.p = nullptr;
  other.bytes = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this == &other) return *this;
  if (p) sycl::free(p, default_device_queue(0));
  p = other.p;
  bytes = other.bytes;
  other.p = nullptr;
  other.bytes = 0;
  return *this;
}

void DeviceBuffer::fill(int byte_value) {
  if (bytes == 0) return;
  default_device_queue(0).memset(p, byte_value, bytes).wait_and_throw();
}

void DeviceBuffer::copy_from_host(const void* source, std::size_t count, std::size_t byte_offset) {
  require_range(byte_offset, count, "host-to-device copy");
  if (count == 0) return;
  default_device_queue(0).memcpy(static_cast<std::uint8_t*>(p) + byte_offset, source, count).wait_and_throw();
}

void DeviceBuffer::copy_to_host(void* destination, std::size_t count, std::size_t byte_offset) const {
  require_range(byte_offset, count, "device-to-host copy");
  if (count == 0) return;
  default_device_queue(0).memcpy(destination, static_cast<const std::uint8_t*>(p) + byte_offset, count).wait_and_throw();
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
  void* ptr = sycl::aligned_alloc_device(256, capacity_bytes, default_device_queue(0));
  if (!ptr) throw std::runtime_error("HAL(SYCL): DeviceArena allocation failed");
  base_ = ptr;
  cap_ = capacity_bytes;
}

DeviceArena::DeviceArena(DeviceSpan storage) : base_(storage.data), cap_(storage.bytes), owns_(false) {
  if (base_ == nullptr || cap_ == 0) throw std::invalid_argument("borrowed DeviceArena storage must be non-empty");
}

DeviceArena::~DeviceArena() {
  if (owns_ && base_) sycl::free(base_, default_device_queue(0));
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
  if (owns_ && base_) sycl::free(base_, default_device_queue(0));
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
      throw std::runtime_error("HAL(SYCL): injected pinned allocation failure");
    }
  }
  void* ptr = sycl::aligned_alloc_host(4096, size_bytes, default_device_queue(0));
  if (!ptr) throw std::runtime_error("HAL(SYCL): aligned_alloc_host failed");
  data_ = ptr;
}

PinnedHostBuffer::~PinnedHostBuffer() {
  if (data_) sycl::free(data_, default_device_queue(0));
  data_ = nullptr;
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
  if (this == &other) return *this;
  if (data_) sycl::free(data_, default_device_queue(0));
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
  const std::size_t size = (bytes + kHugePageBytes - 1) / kHugePageBytes * kHugePageBytes;
  void* raw = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (raw == MAP_FAILED) throw std::runtime_error("HostRegion(SYCL): mmap failed");
  data_ = static_cast<std::byte*>(raw);
  size_ = size;
  (void)::madvise(data_, size_, MADV_HUGEPAGE);
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
void HostRegion::unregister() noexcept { registered_ = false; device_ = nullptr; }
void HostRegion::release() noexcept {
  unregister();
  if (data_) ::munmap(data_, size_);
  data_ = nullptr;
  size_ = 0;
}

MappedHostBuffer::MappedHostBuffer(std::size_t bytes) : size_(bytes) {
  if (bytes == 0) throw std::invalid_argument("MappedHostBuffer size must be nonzero");
  void* ptr = sycl::aligned_alloc_shared(4096, bytes, default_device_queue(0));
  if (!ptr) throw std::runtime_error("HAL(SYCL): aligned_alloc_shared failed");
  std::memset(ptr, 0, bytes);
  data_ = ptr;
  device_ = ptr;
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
  if (data_) sycl::free(data_, default_device_queue(0));
  data_ = nullptr;
  device_ = nullptr;
  size_ = 0;
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
bool UnifiedWeightPackPin::pin() { return false; }
void UnifiedWeightPackPin::unpin() noexcept { registered_ = false; readonly_ = false; device_base_ = nullptr; }
void UnifiedWeightPackPin::release() noexcept { unpin(); host_base_ = nullptr; size_ = 0; owner_.reset(); }
void* UnifiedWeightPackPin::tensor_device_ptr(std::uint64_t offset, std::uint64_t bytes) const noexcept {
  if (!registered_ || !device_base_ || offset > size_ || bytes > size_ - offset) return nullptr;
  return static_cast<std::uint8_t*>(device_base_) + offset;
}

std::uint64_t mem_available_bytes() { return read_meminfo_kib("MemAvailable:"); }

namespace hal {

Stream::Stream(int device_id) : device_id_(device_id) {
  auto* q = new sycl::queue{default_device_queue(device_id).get_context(), get_sycl_device(device_id),
                            sycl::property::queue::in_order{}};
  handle_ = StreamHandle::from_native(q);
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
  delete static_cast<sycl::queue*>(handle_.native);
  handle_ = {};
}

void Stream::synchronize() const {
  if (!handle_) throw std::logic_error("hal::Stream::synchronize on empty stream");
  stream_synchronize(handle_);
}

Event::Event(int device_id) : device_id_(device_id) {
  (void)get_sycl_device(device_id);
  handle_ = EventHandle::from_native(new sycl::event{});
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
  delete static_cast<sycl::event*>(handle_.native);
  handle_ = {};
}

void Event::record(StreamHandle stream) {
  if (!handle_) throw std::logic_error("hal::Event::record on empty event");
  auto* ev = static_cast<sycl::event*>(handle_.native);
  *ev = unwrap_queue(stream)->ext_oneapi_submit_barrier();
}

void Event::wait(StreamHandle stream) const {
  if (!handle_) throw std::logic_error("hal::Event::wait on empty event");
  const auto* ev = static_cast<const sycl::event*>(handle_.native);
  unwrap_queue(stream)->ext_oneapi_submit_barrier({*ev});
}

bool Event::ready() const {
  if (!handle_) throw std::logic_error("hal::Event::ready on empty event");
  const auto* ev = static_cast<const sycl::event*>(handle_.native);
  return ev->get_info<sycl::info::event::command_execution_status>() == sycl::info::event_command_status::complete;
}

void Event::synchronize() const {
  if (!handle_) throw std::logic_error("hal::Event::synchronize on empty event");
  static_cast<sycl::event*>(handle_.native)->wait_and_throw();
}

BackendKind active_backend() noexcept { return BackendKind::Sycl; }

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
  const sycl::device& dev = get_sycl_device(device_id);
  const std::string raw_name = dev.get_info<sycl::info::device::name>();
  const std::string lower_name = to_lower_ascii(raw_name);
  const std::string lower_vendor = to_lower_ascii(dev.get_info<sycl::info::device::vendor>());
  const auto vendor_id = static_cast<std::uint32_t>(dev.get_info<sycl::info::device::vendor_id>());
  const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  int min_sg = 0, max_sg = 0;
  bool sg8 = false, sg16 = false, sg32 = false;
  if (!sg_sizes.empty()) {
    min_sg = static_cast<int>(*std::min_element(sg_sizes.begin(), sg_sizes.end()));
    max_sg = static_cast<int>(*std::max_element(sg_sizes.begin(), sg_sizes.end()));
    for (const auto w : sg_sizes) {
      if (w == 8) sg8 = true;
      else if (w == 16) sg16 = true;
      else if (w == 32) sg32 = true;
    }
  }

  vendor::sycl::SyclDeviceProbe probe;
  probe.is_gpu = dev.is_gpu();
  probe.is_level_zero = (dev.get_backend() == sycl::backend::ext_oneapi_level_zero);
  probe.host_unified_memory = dev.get_info<sycl::info::device::host_unified_memory>();
  probe.vendor_lower = lower_vendor;
  probe.vendor_id = vendor_id;
  probe.name_lower = lower_name;
  probe.min_subgroup = min_sg;
  probe.max_subgroup = max_sg;
  probe.supports_sg8 = sg8;
  probe.supports_sg16 = sg16;
  probe.supports_sg32 = sg32;
  probe.has_runtime_xmx_dpas = dev.has(sycl::aspect::ext_intel_matrix);
  probe.has_runtime_bf16 = dev.has(sycl::aspect::ext_oneapi_bfloat16_math_functions);

  const auto arch = vendor::sycl::classify_intel_device(probe);
  if (arch.preferred_subgroup <= 0) {
    throw std::runtime_error("HAL(SYCL): selected device is not a validated Intel GPU with subgroup support: " +
                             raw_name);
  }

  DeviceCaps caps;
  caps.backend = BackendKind::Sycl;
  caps.primary_space = MemorySpace::Device;
  caps.name = raw_name;
  caps.arch_name = arch.has_xmx_dpas ? "intel-xe-xmx" : "intel-xe";
  caps.device_id = device_id;
  caps.subgroup = SubgroupCaps{arch.min_subgroup, arch.max_subgroup, arch.preferred_subgroup};
  caps.max_shared_mem_bytes = static_cast<std::size_t>(dev.get_info<sycl::info::device::local_mem_size>());
  caps.supports_bf16 = probe.has_runtime_bf16;
  caps.supports_fp16 = dev.has(sycl::aspect::fp16);
  caps.supports_fp8_e4m3 = false;
  caps.supports_graph_capture = false;
  caps.supports_conditional_graph = false;
  caps.supports_pdl = false;
  caps.supports_wmma_or_xmx = arch.has_xmx_dpas;
  caps.prefers_direct_device_alloc = arch.prefer_level_zero_direct_alloc;
  caps.unified_memory = probe.host_unified_memory;
  return caps;
}

MemoryInfo query_memory_info() {
  const sycl::device& dev = get_sycl_device(0);
  const std::size_t total = static_cast<std::size_t>(dev.get_info<sycl::info::device::global_mem_size>());
  std::size_t free_b = total;
  if (dev.has(sycl::aspect::ext_intel_free_memory)) {
    free_b = static_cast<std::size_t>(dev.get_info<sycl::ext::intel::info::device::free_memory>());
  }
  return MemoryInfo{free_b, total};
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
  unwrap_queue(stream)->memcpy(dst_dev, src_host, bytes);
}

void memcpy_d2h_async(void* dst_host, const void* src_dev, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  unwrap_queue(stream)->memcpy(dst_host, src_dev, bytes);
}

void memcpy_d2d_async(void* dst_dev, const void* src_dev, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  unwrap_queue(stream)->memcpy(dst_dev, src_dev, bytes);
}

void memset_async(void* dst_dev, int byte_value, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  unwrap_queue(stream)->memset(dst_dev, byte_value, bytes);
}

void stream_synchronize(StreamHandle stream) { unwrap_queue(stream)->wait_and_throw(); }

}  // namespace hal
}  // namespace ingine

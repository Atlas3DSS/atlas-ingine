// Pure native AMD ROCm/HIP Hardware Abstraction Layer (HAL) implementation. Zero CUDA runtime or headers.
#include "core/hal.h"
#include "core/vendor_policy.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace ingine::hal {
namespace {

std::string hip_err_msg(const char* prefix, hipError_t err) {
  return std::string(prefix) + ": " + hipGetErrorName(err) + ": " + hipGetErrorString(err);
}

void throw_if_hip(hipError_t err, const char* what) {
  if (err != hipSuccess) throw std::runtime_error(hip_err_msg(what, err));
}

void destroy_hip_stream(StreamHandle& h, int device_id) noexcept {
  if (!h) return;
  (void)hipSetDevice(device_id);
  const hipError_t err = hipStreamDestroy(static_cast<hipStream_t>(h.native));
  if (err != hipSuccess) {
    std::fprintf(stderr, "hal::Stream(HIP) cleanup failed: %s: %s\n", hipGetErrorName(err), hipGetErrorString(err));
  }
  h = {};
}

void destroy_hip_event(EventHandle& h, int device_id) noexcept {
  if (!h) return;
  (void)hipSetDevice(device_id);
  const hipError_t err = hipEventDestroy(static_cast<hipEvent_t>(h.native));
  if (err != hipSuccess) {
    std::fprintf(stderr, "hal::Event(HIP) cleanup failed: %s: %s\n", hipGetErrorName(err), hipGetErrorString(err));
  }
  h = {};
}

}  // namespace

Stream::Stream(int device_id) : device_id_(device_id) {
  throw_if_hip(hipSetDevice(device_id_), "hal::Stream(HIP) hipSetDevice");
  hipStream_t s = nullptr;
  throw_if_hip(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "hal::Stream(HIP) create");
  handle_ = StreamHandle::from_native(s);
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

void Stream::release() noexcept { destroy_hip_stream(handle_, device_id_); }

void Stream::synchronize() const {
  if (!handle_) throw std::logic_error("hal::Stream::synchronize on empty stream");
  stream_synchronize(handle_);
}

Event::Event(int device_id) : device_id_(device_id) {
  throw_if_hip(hipSetDevice(device_id_), "hal::Event(HIP) hipSetDevice");
  hipEvent_t e = nullptr;
  throw_if_hip(hipEventCreateWithFlags(&e, hipEventDisableTiming), "hal::Event(HIP) create");
  handle_ = EventHandle::from_native(e);
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

void Event::release() noexcept { destroy_hip_event(handle_, device_id_); }

void Event::record(StreamHandle stream) {
  if (!handle_) throw std::logic_error("hal::Event::record on empty event");
  throw_if_hip(hipEventRecord(static_cast<hipEvent_t>(handle_.native), static_cast<hipStream_t>(stream.native)),
               "hal::Event(HIP) record");
}

void Event::wait(StreamHandle stream) const {
  if (!handle_) throw std::logic_error("hal::Event::wait on empty event");
  throw_if_hip(hipStreamWaitEvent(static_cast<hipStream_t>(stream.native), static_cast<hipEvent_t>(handle_.native), 0),
               "hal::Event(HIP) wait");
}

bool Event::ready() const {
  if (!handle_) throw std::logic_error("hal::Event::ready on empty event");
  const hipError_t err = hipEventQuery(static_cast<hipEvent_t>(handle_.native));
  if (err == hipSuccess) return true;
  if (err == hipErrorNotReady) return false;
  throw_if_hip(err, "hal::Event(HIP) query");
  return false;
}

void Event::synchronize() const {
  if (!handle_) throw std::logic_error("hal::Event::synchronize on empty event");
  throw_if_hip(hipEventSynchronize(static_cast<hipEvent_t>(handle_.native)), "hal::Event(HIP) synchronize");
}

BackendKind active_backend() noexcept { return BackendKind::Hip; }

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
  int count = 0;
  throw_if_hip(hipGetDeviceCount(&count), "hipGetDeviceCount");
  if (device_id < 0 || device_id >= count) {
    throw std::invalid_argument("hal::query_device_caps(HIP): invalid device id " + std::to_string(device_id));
  }
  hipDeviceProp_t prop{};
  throw_if_hip(hipGetDeviceProperties(&prop, device_id), "hipGetDeviceProperties");
  const auto arch = vendor::hip::classify_hip_arch(prop.gcnArchName);

  DeviceCaps caps;
  caps.backend = BackendKind::Hip;
  caps.primary_space = prop.integrated ? MemorySpace::Unified : MemorySpace::Device;
  caps.name = prop.name;
  caps.arch_name = std::string(vendor::hip::strip_target_features(prop.gcnArchName));
  caps.device_id = device_id;
  caps.arch_major = prop.major;
  caps.arch_minor = prop.minor;
  caps.subgroup = SubgroupCaps{prop.warpSize, prop.warpSize, prop.warpSize};
  caps.max_shared_mem_bytes = prop.sharedMemPerBlock;
  caps.supports_bf16 = (arch.family >= vendor::hip::ArchFamily::Rdna3) || arch.is_wave64;
  caps.supports_fp16 = true;
  caps.supports_fp8_e4m3 = (arch.family == vendor::hip::ArchFamily::Cdna3 ||
                            arch.family == vendor::hip::ArchFamily::Cdna4 ||
                            arch.family == vendor::hip::ArchFamily::Rdna4);
  caps.supports_graph_capture = true;
  caps.supports_conditional_graph = false;
  caps.supports_pdl = false;
  caps.supports_wmma_or_xmx = arch.has_rdna_wmma || arch.is_wave64;
  caps.prefers_direct_device_alloc = !prop.integrated;
  caps.unified_memory = prop.integrated != 0;
  return caps;
}

MemoryInfo query_memory_info() {
  std::size_t free_b = 0, total_b = 0;
  throw_if_hip(hipMemGetInfo(&free_b, &total_b), "hipMemGetInfo");
  return MemoryInfo{free_b, total_b};
}

bool is_stream_capturing(StreamHandle stream) {
  hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
  throw_if_hip(hipStreamIsCapturing(static_cast<hipStream_t>(stream.native), &status), "hipStreamIsCapturing");
  return status != hipStreamCaptureStatusNone;
}

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
  throw_if_hip(hipMemcpyAsync(dst_dev, src_host, bytes, hipMemcpyHostToDevice, static_cast<hipStream_t>(stream.native)),
               "hal::memcpy_h2d_async(HIP)");
}

void memcpy_d2h_async(void* dst_host, const void* src_dev, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  throw_if_hip(hipMemcpyAsync(dst_host, src_dev, bytes, hipMemcpyDeviceToHost, static_cast<hipStream_t>(stream.native)),
               "hal::memcpy_d2h_async(HIP)");
}

void memcpy_d2d_async(void* dst_dev, const void* src_dev, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  throw_if_hip(
      hipMemcpyAsync(dst_dev, src_dev, bytes, hipMemcpyDeviceToDevice, static_cast<hipStream_t>(stream.native)),
      "hal::memcpy_d2d_async(HIP)");
}

void memset_async(void* dst_dev, int byte_value, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  throw_if_hip(hipMemsetAsync(dst_dev, byte_value, bytes, static_cast<hipStream_t>(stream.native)),
               "hal::memset_async(HIP)");
}

void stream_synchronize(StreamHandle stream) {
  throw_if_hip(hipStreamSynchronize(static_cast<hipStream_t>(stream.native)), "hal::stream_synchronize(HIP)");
}

}  // namespace ingine::hal

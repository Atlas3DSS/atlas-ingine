#include "core/hal.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ingine::hal {
namespace {

std::string cuda_err_msg(const char* prefix, cudaError_t err) {
  return std::string(prefix) + ": " + cudaGetErrorName(err) + ": " + cudaGetErrorString(err);
}

void throw_if_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(cuda_err_msg(what, err));
}

void destroy_cuda_stream(StreamHandle& h, int device_id) noexcept {
  if (!h) return;
  (void)cudaSetDevice(device_id);
  const cudaError_t err = cudaStreamDestroy(static_cast<cudaStream_t>(h));
  if (err != cudaSuccess) {
    std::fprintf(stderr, "hal::Stream cleanup failed: %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
  }
  h = {};
}

void destroy_cuda_event(EventHandle& h, int device_id) noexcept {
  if (!h) return;
  (void)cudaSetDevice(device_id);
  const cudaError_t err = cudaEventDestroy(static_cast<cudaEvent_t>(h));
  if (err != cudaSuccess) {
    std::fprintf(stderr, "hal::Event cleanup failed: %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));
  }
  h = {};
}

}  // namespace

Stream::Stream(int device_id) : device_id_(device_id) {
  throw_if_cuda(cudaSetDevice(device_id_), "hal::Stream cudaSetDevice");
  cudaStream_t s = nullptr;
  throw_if_cuda(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "hal::Stream create");
  handle_ = s;
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

void Stream::release() noexcept { destroy_cuda_stream(handle_, device_id_); }

void Stream::synchronize() const {
  if (!handle_) throw std::logic_error("hal::Stream::synchronize on empty stream");
  stream_synchronize(handle_);
}

Event::Event(int device_id) : device_id_(device_id) {
  throw_if_cuda(cudaSetDevice(device_id_), "hal::Event cudaSetDevice");
  cudaEvent_t e = nullptr;
  throw_if_cuda(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "hal::Event create");
  handle_ = e;
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

void Event::release() noexcept { destroy_cuda_event(handle_, device_id_); }

void Event::record(StreamHandle stream) {
  if (!handle_) throw std::logic_error("hal::Event::record on empty event");
  throw_if_cuda(cudaEventRecord(static_cast<cudaEvent_t>(handle_), static_cast<cudaStream_t>(stream)),
                "hal::Event record");
}

void Event::wait(StreamHandle stream) const {
  if (!handle_) throw std::logic_error("hal::Event::wait on empty event");
  throw_if_cuda(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(handle_), 0),
                "hal::Event wait");
}

bool Event::ready() const {
  if (!handle_) throw std::logic_error("hal::Event::ready on empty event");
  const cudaError_t err = cudaEventQuery(static_cast<cudaEvent_t>(handle_));
  if (err == cudaSuccess) return true;
  if (err == cudaErrorNotReady) return false;
  throw_if_cuda(err, "hal::Event query");
  return false;
}

void Event::synchronize() const {
  if (!handle_) throw std::logic_error("hal::Event::synchronize on empty event");
  throw_if_cuda(cudaEventSynchronize(static_cast<cudaEvent_t>(handle_)), "hal::Event synchronize");
}

BackendKind active_backend() noexcept { return BackendKind::Cuda; }

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

bool conditional_graphs_enabled(int /*device_id*/) {
  const char* disable = std::getenv("INGINE_DISABLE_CONDITIONAL_GRAPHS");
  if (disable != nullptr && *disable != '\0' && std::strcmp(disable, "0") != 0) {
    return false;
  }
#if CUDART_VERSION >= 12040
  return true;
#else
  return false;
#endif
}

DeviceCaps query_device_caps(int device_id) {
  int count = 0;
  throw_if_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (device_id < 0 || device_id >= count) {
    throw std::invalid_argument("hal::query_device_caps: invalid CUDA device id " + std::to_string(device_id));
  }
  cudaDeviceProp prop{};
  throw_if_cuda(cudaGetDeviceProperties(&prop, device_id), "cudaGetDeviceProperties");
  DeviceCaps caps;
  caps.backend = BackendKind::Cuda;
  caps.primary_space = MemorySpace::Device;
  caps.name = prop.name;
  caps.device_id = device_id;
  caps.arch_major = prop.major;
  caps.arch_minor = prop.minor;
  caps.subgroup = SubgroupCaps{prop.warpSize, prop.warpSize, prop.warpSize};
  caps.max_shared_mem_bytes =
      prop.sharedMemPerBlockOptin > 0 ? prop.sharedMemPerBlockOptin : prop.sharedMemPerBlock;
  caps.supports_bf16 = prop.major >= 8;
  caps.supports_fp16 = true;
  caps.supports_fp8_e4m3 = (prop.major > 8) || (prop.major == 8 && prop.minor >= 9);
  caps.supports_graph_capture = true;
  caps.supports_conditional_graph = conditional_graphs_enabled(device_id);
  caps.supports_pdl = prop.major >= 9;
  caps.unified_memory = prop.integrated != 0;
  return caps;
}

MemoryInfo query_memory_info() {
  std::size_t free = 0, total = 0;
  throw_if_cuda(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
  return MemoryInfo{free, total};
}

bool is_stream_capturing(StreamHandle stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  throw_if_cuda(cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &status), "cudaStreamIsCapturing");
  return status != cudaStreamCaptureStatusNone;
}

void require_graph_capture_or_eager(const DeviceCaps& caps, bool allow_eager, const char* caller) {
  if (!caps.supports_graph_capture && !allow_eager) {
    throw std::runtime_error(std::string(caller ? caller : "hal") + ": graph capture unsupported on backend " +
                             backend_name(caps.backend));
  }
}

void memcpy_h2d_async(void* dst_dev, const void* src_host, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  throw_if_cuda(cudaMemcpyAsync(dst_dev, src_host, bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)),
                "hal::memcpy_h2d_async");
}

void memcpy_d2h_async(void* dst_host, const void* src_dev, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  throw_if_cuda(cudaMemcpyAsync(dst_host, src_dev, bytes, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream)),
                "hal::memcpy_d2h_async");
}

void memcpy_d2d_async(void* dst_dev, const void* src_dev, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  throw_if_cuda(cudaMemcpyAsync(dst_dev, src_dev, bytes, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream)),
                "hal::memcpy_d2d_async");
}

void memset_async(void* dst_dev, int byte_value, std::size_t bytes, StreamHandle stream) {
  if (bytes == 0) return;
  throw_if_cuda(cudaMemsetAsync(dst_dev, byte_value, bytes, static_cast<cudaStream_t>(stream)), "hal::memset_async");
}

void stream_synchronize(StreamHandle stream) {
  throw_if_cuda(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)), "hal::stream_synchronize");
}

}  // namespace ingine::hal

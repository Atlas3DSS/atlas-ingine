// Pure native AMD ROCm/HIP DeviceContext, CudaEventTimer (BackendStreamT/EventT), and completion event.
#include "core/device.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace ingine {
namespace {

std::string hip_error_message(const char* prefix, hipError_t err) {
  return std::string(prefix) + ": " + hipGetErrorName(err) + ": " + hipGetErrorString(err);
}

void log_hip_error(const char* op, hipError_t err) noexcept {
  if (err != hipSuccess) {
    std::fprintf(stderr, "HIP cleanup failed during %s: %s: %s\n", op, hipGetErrorName(err), hipGetErrorString(err));
  }
}

void destroy_stream(hipStream_t& stream) noexcept {
  if (stream != nullptr) {
    log_hip_error("hipStreamDestroy", hipStreamDestroy(stream));
    stream = nullptr;
  }
}

void destroy_event(hipEvent_t& event) noexcept {
  if (event != nullptr) {
    log_hip_error("hipEventDestroy", hipEventDestroy(event));
    event = nullptr;
  }
}

void destroy_concurrent(ConcurrentStream& concurrent) noexcept {
  for (hipEvent_t& event : concurrent.events) destroy_event(event);
  destroy_stream(concurrent.stream);
}

}  // namespace

void hip_check(hipError_t err, const char* expr, const char* file, int line) {
  if (err == hipSuccess) return;
  std::fprintf(stderr, "%s:%d: HIP_CHECK(%s) failed: %s: %s\n", file, line, expr, hipGetErrorName(err),
               hipGetErrorString(err));
  std::abort();
}

DeviceContext::DeviceContext(int device_id) : device(device_id) {
  int count = 0;
  hipError_t err = hipGetDeviceCount(&count);
  if (err != hipSuccess) throw std::runtime_error(hip_error_message("hipGetDeviceCount failed", err));
  if (count <= 0) throw std::runtime_error("no HIP devices available");
  if (device_id < 0 || device_id >= count) throw std::runtime_error("invalid HIP device id");

  bind_to_current_thread();
  err = hipGetDeviceProperties(&props, device_id);
  if (err != hipSuccess) throw std::runtime_error(hip_error_message("hipGetDeviceProperties failed", err));

  int least_priority = 0, greatest_priority = 0;
  err = hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
  if (err != hipSuccess) {
    throw std::runtime_error(hip_error_message("hipDeviceGetStreamPriorityRange failed", err));
  }

  hipStream_t compute = nullptr, load = nullptr;
  err = hipStreamCreateWithPriority(&compute, hipStreamNonBlocking, greatest_priority);
  if (err != hipSuccess) throw std::runtime_error(hip_error_message("hipStreamCreateWithPriority failed", err));

  err = hipStreamCreateWithFlags(&load, hipStreamNonBlocking);
  if (err != hipSuccess) {
    destroy_stream(compute);
    throw std::runtime_error(hip_error_message("hipStreamCreateWithFlags(transfer_stream) failed", err));
  }

  ConcurrentStream side;
  err = hipStreamCreateWithPriority(&side.stream, hipStreamNonBlocking, greatest_priority);
  for (hipEvent_t& ev : side.events) {
    if (err == hipSuccess) err = hipEventCreateWithFlags(&ev, hipEventDisableTiming);
  }
  if (err != hipSuccess) {
    destroy_concurrent(side);
    destroy_stream(load);
    destroy_stream(compute);
    throw std::runtime_error(hip_error_message("concurrent HIP stream creation failed", err));
  }

  stream = compute;
  transfer_stream = load;
  concurrent = side;
}

DeviceContext::~DeviceContext() {
  if (stream != nullptr || transfer_stream != nullptr || concurrent.stream != nullptr) {
    bind_to_current_thread_noexcept();
  }
  destroy_concurrent(concurrent);
  destroy_stream(transfer_stream);
  destroy_stream(stream);
}

DeviceContext::DeviceContext(DeviceContext&& other) noexcept
    : device(other.device),
      stream(other.stream),
      transfer_stream(other.transfer_stream),
      concurrent(other.concurrent),
      props(other.props) {
  other.stream = nullptr;
  other.transfer_stream = nullptr;
  other.concurrent = {};
}

DeviceContext& DeviceContext::operator=(DeviceContext&& other) noexcept {
  if (this == &other) return *this;
  if (stream != nullptr || transfer_stream != nullptr || concurrent.stream != nullptr) {
    bind_to_current_thread_noexcept();
  }
  destroy_concurrent(concurrent);
  destroy_stream(transfer_stream);
  destroy_stream(stream);
  device = other.device;
  props = other.props;
  stream = other.stream;
  transfer_stream = other.transfer_stream;
  concurrent = other.concurrent;
  other.stream = nullptr;
  other.transfer_stream = nullptr;
  other.concurrent = {};
  return *this;
}

void DeviceContext::bind_to_current_thread() const {
  const hipError_t err = hipSetDevice(device);
  if (err != hipSuccess) throw std::runtime_error(hip_error_message("hipSetDevice failed", err));
}

void DeviceContext::bind_to_current_thread_noexcept() const noexcept { log_hip_error("hipSetDevice", hipSetDevice(device)); }
int DeviceContext::compute_capability() const noexcept { return props.major * 10 + props.minor; }
int DeviceContext::multiprocessor_count() const noexcept { return props.multiProcessorCount; }
DeviceExecutionView DeviceContext::execution_view() const noexcept { return {.stream = stream, .multiprocessor_count = multiprocessor_count()}; }
std::size_t DeviceContext::total_vram() const noexcept { return props.totalGlobalMem; }
void DeviceContext::synchronize() const { HIP_CHECK(hipStreamSynchronize(stream)); }

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx) : CudaEventTimer(ctx, ctx.stream) {}

CudaEventTimer::CudaEventTimer(const DeviceContext& ctx, BackendStreamT stream) : stream_(stream) {
  if (stream == nullptr) throw std::invalid_argument("HIP timer stream is null");
  ctx.bind_to_current_thread();
  hipEvent_t s = nullptr, e = nullptr;
  hipError_t err = hipEventCreate(&s);
  if (err != hipSuccess) throw std::runtime_error(hip_error_message("hipEventCreate(start) failed", err));
  err = hipEventCreate(&e);
  if (err != hipSuccess) {
    destroy_event(s);
    throw std::runtime_error(hip_error_message("hipEventCreate(stop) failed", err));
  }
  start_ = s;
  stop_ = e;
}

CudaEventTimer::~CudaEventTimer() {
  destroy_event(stop_);
  destroy_event(start_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer&& other) noexcept
    : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
  other.stream_ = nullptr;
  other.start_ = nullptr;
  other.stop_ = nullptr;
}

CudaEventTimer& CudaEventTimer::operator=(CudaEventTimer&& other) noexcept {
  if (this == &other) return *this;
  destroy_event(stop_);
  destroy_event(start_);
  stream_ = other.stream_;
  start_ = other.start_;
  stop_ = other.stop_;
  other.stream_ = nullptr;
  other.start_ = nullptr;
  other.stop_ = nullptr;
  return *this;
}

void CudaEventTimer::start() { HIP_CHECK(hipEventRecord(start_, stream_)); }
void CudaEventTimer::record_stop() { HIP_CHECK(hipEventRecord(stop_, stream_)); }
float CudaEventTimer::elapsed_ms() const {
  float ms = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
  return ms;
}
float CudaEventTimer::stop_ms() {
  record_stop();
  HIP_CHECK(hipEventSynchronize(stop_));
  return elapsed_ms();
}

CudaCompletionEvent::CudaCompletionEvent(const DeviceContext& ctx) : device_(ctx.device) {
  ctx.bind_to_current_thread();
  const hipError_t err = hipEventCreateWithFlags(&event_, hipEventDisableTiming);
  if (err != hipSuccess) throw std::runtime_error(hip_error_message("hipEventCreateWithFlags failed", err));
}

CudaCompletionEvent::~CudaCompletionEvent() { destroy_event(event_); }

CudaCompletionEvent::CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
    : device_(other.device_), event_(std::exchange(other.event_, nullptr)) {}

CudaCompletionEvent& CudaCompletionEvent::operator=(CudaCompletionEvent&& other) noexcept {
  if (this == &other) return *this;
  destroy_event(event_);
  device_ = other.device_;
  event_ = std::exchange(other.event_, nullptr);
  return *this;
}

void CudaCompletionEvent::record(BackendStreamT stream) { HIP_CHECK(hipEventRecord(event_, stream)); }
void CudaCompletionEvent::wait(BackendStreamT stream) const { HIP_CHECK(hipStreamWaitEvent(stream, event_, 0)); }

bool CudaCompletionEvent::ready() const {
  const hipError_t err = hipEventQuery(event_);
  if (err == hipSuccess) return true;
  if (err == hipErrorNotReady) return false;
  HIP_CHECK(err);
  return false;
}

void CudaCompletionEvent::synchronize() const { HIP_CHECK(hipEventSynchronize(event_)); }

}  // namespace ingine

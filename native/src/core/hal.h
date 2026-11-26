// Hardware abstraction layer (HAL) boundary for device placement, streams, events, copies, and capabilities.
//
// CUDA-free on non-CUDA builds (INGINE_BACKEND_CUDA == 0). On CUDA builds, StreamHandle and EventHandle
// convert implicitly to/from cudaStream_t / cudaEvent_t with zero abstraction overhead.
//
// Lifetime, deferred execution & host visibility contract:
//   * `ingine::Tensor` and operation arguments are non-owning views. Backing DeviceBuffer, DeviceArena scopes,
//     PinnedHostBuffer, and MappedHostBuffer objects must remain alive until every async operation and Event
//     referencing them has completed (via stream_synchronize() or Event::synchronize()).
//   * Under graph capture or lazy execution, operations enqueued to a StreamHandle are deferred until the
//     captured graph launches or commits; host-visible readbacks from memcpy_d2h_async or mapped exchange
//     buffers are guaranteed visible to the host only after stream_synchronize() or Event::synchronize().
//   * Backends without ordinary graph capture or conditional SWITCH graph nodes must either execute eagerly via
//     allow_eager=true or fail explicitly via require_graph_capture_or_eager() / require_conditional_switch_or_eager().
#pragma once

#ifndef INGINE_BACKEND_CUDA
#if defined(INGINE_BACKEND_CPU) || defined(INGINE_BACKEND_HIP) || defined(INGINE_BACKEND_MLX) || defined(INGINE_BACKEND_SYCL)
#define INGINE_BACKEND_CUDA 0
#else
#define INGINE_BACKEND_CUDA 1
#endif
#endif

#ifndef INGINE_BACKEND_HIP
#define INGINE_BACKEND_HIP 0
#endif

#if INGINE_BACKEND_CUDA
#include <cuda_runtime.h>
#elif INGINE_BACKEND_HIP
#include <hip/hip_runtime.h>
#endif

#include <cstddef>
#include <cstdint>
#include <string>

namespace ingine::hal {

enum class BackendKind : std::uint8_t {
  Cpu = 0,
  Cuda = 1,
  Hip = 2,
  Mlx = 3,
  Sycl = 4,
};

enum class MemorySpace : std::uint8_t {
  Host = 0,
  Device = 1,
  Unified = 2,
};

struct SubgroupCaps {
  int min_width = 1;
  int max_width = 1;
  int preferred_width = 1;
};

struct DeviceCaps {
  BackendKind backend = BackendKind::Cpu;
  MemorySpace primary_space = MemorySpace::Host;
  std::string name;
  std::string arch_name;
  int device_id = 0;
  int arch_major = 0;
  int arch_minor = 0;
  SubgroupCaps subgroup{};
  std::size_t max_shared_mem_bytes = 0;
  std::size_t recommended_working_set_bytes = 0;
  std::size_t streaming_ring_budget_bytes = 0;
  bool supports_bf16 = false;
  bool supports_fp16 = false;
  bool supports_fp8_e4m3 = false;
  bool supports_graph_capture = false;
  // Conditional SWITCH graph node support (e.g. cudaGraphCondTypeSwitch on CUDA >= 12.8 driver/context).
  bool supports_conditional_graph = false;
  bool supports_pdl = false;
  bool supports_wmma_or_xmx = false;
  bool prefers_direct_device_alloc = true;
  bool unified_memory = false;
};

enum class GraphRequirement : std::uint8_t {
  Capture = 0,            // ordinary linear graph capture
  ConditionalSwitch = 1,  // graph capture + conditional SWITCH node support
};

struct MemoryInfo {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

struct StreamHandle {
  void* native = nullptr;
  constexpr StreamHandle() noexcept = default;
#if INGINE_BACKEND_CUDA
  constexpr StreamHandle(cudaStream_t s) noexcept : native(s) {}
  constexpr operator cudaStream_t() const noexcept { return static_cast<cudaStream_t>(native); }
#elif INGINE_BACKEND_HIP
  constexpr StreamHandle(hipStream_t s) noexcept : native(s) {}
  constexpr operator hipStream_t() const noexcept { return static_cast<hipStream_t>(native); }
#else
  constexpr StreamHandle(std::nullptr_t) noexcept : native(nullptr) {}
  constexpr explicit StreamHandle(void* p) noexcept : native(p) {}
#endif
  constexpr static StreamHandle from_native(void* p) noexcept {
#if INGINE_BACKEND_CUDA
    return StreamHandle(static_cast<cudaStream_t>(p));
#elif INGINE_BACKEND_HIP
    return StreamHandle(static_cast<hipStream_t>(p));
#else
    return StreamHandle(p);
#endif
  }
  [[nodiscard]] explicit operator bool() const noexcept { return native != nullptr; }
  friend bool operator==(StreamHandle a, StreamHandle b) noexcept { return a.native == b.native; }
  friend bool operator!=(StreamHandle a, StreamHandle b) noexcept { return a.native != b.native; }
};

struct EventHandle {
  void* native = nullptr;
  constexpr EventHandle() noexcept = default;
#if INGINE_BACKEND_CUDA
  constexpr EventHandle(cudaEvent_t e) noexcept : native(e) {}
  constexpr operator cudaEvent_t() const noexcept { return static_cast<cudaEvent_t>(native); }
#elif INGINE_BACKEND_HIP
  constexpr EventHandle(hipEvent_t e) noexcept : native(e) {}
  constexpr operator hipEvent_t() const noexcept { return static_cast<hipEvent_t>(native); }
#else
  constexpr EventHandle(std::nullptr_t) noexcept : native(nullptr) {}
  constexpr explicit EventHandle(void* p) noexcept : native(p) {}
#endif
  constexpr static EventHandle from_native(void* p) noexcept {
#if INGINE_BACKEND_CUDA
    return EventHandle(static_cast<cudaEvent_t>(p));
#elif INGINE_BACKEND_HIP
    return EventHandle(static_cast<hipEvent_t>(p));
#else
    return EventHandle(p);
#endif
  }
  [[nodiscard]] explicit operator bool() const noexcept { return native != nullptr; }
  friend bool operator==(EventHandle a, EventHandle b) noexcept { return a.native == b.native; }
  friend bool operator!=(EventHandle a, EventHandle b) noexcept { return a.native != b.native; }
};

class Stream {
 public:
  explicit Stream(int device_id = 0);
  ~Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;

  [[nodiscard]] StreamHandle handle() const noexcept { return handle_; }
  void synchronize() const;

 private:
  void release() noexcept;
  int device_id_ = 0;
  StreamHandle handle_{};
};

class Event {
 public:
  explicit Event(int device_id = 0);
  ~Event();

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& other) noexcept;
  Event& operator=(Event&& other) noexcept;

  [[nodiscard]] EventHandle handle() const noexcept { return handle_; }
  void record(StreamHandle stream);
  void wait(StreamHandle stream) const;
  [[nodiscard]] bool ready() const;
  void synchronize() const;

 private:
  void release() noexcept;
  int device_id_ = 0;
  EventHandle handle_{};
};

BackendKind active_backend() noexcept;
const char* backend_name(BackendKind kind) noexcept;
DeviceCaps query_device_caps(int device_id = 0);
MemoryInfo query_memory_info();
bool conditional_graphs_enabled(int device_id = 0);
bool is_stream_capturing(StreamHandle stream);
bool graph_requirement_met(const DeviceCaps& caps, GraphRequirement requirement) noexcept;
void require_graph_capture_or_eager(const DeviceCaps& caps, bool allow_eager, const char* caller,
                                    GraphRequirement requirement = GraphRequirement::Capture);
void require_conditional_switch_or_eager(const DeviceCaps& caps, bool allow_eager, const char* caller);

void memcpy_h2d_async(void* dst_dev, const void* src_host, std::size_t bytes, StreamHandle stream = {});
void memcpy_d2h_async(void* dst_host, const void* src_dev, std::size_t bytes, StreamHandle stream = {});
void memcpy_d2d_async(void* dst_dev, const void* src_dev, std::size_t bytes, StreamHandle stream = {});
void memset_async(void* dst_dev, int byte_value, std::size_t bytes, StreamHandle stream = {});
void stream_synchronize(StreamHandle stream = {});

}  // namespace ingine::hal

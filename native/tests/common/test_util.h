// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
#pragma once

// Shared helpers for ingine tests: delegates CUDA-free scalar helpers/oracles to common/scalar_oracle.h
// and keeps CUDA device helpers guarded when INGINE_BACKEND_CUDA == 0.
#include "common/scalar_oracle.h"
#include "core/act.h"

#if INGINE_BACKEND_CUDA || INGINE_BACKEND_HIP
#include "core/gpu_compat.h"
#endif

namespace ingine::test {

inline float bf16_to_float(ingine::ActT v) { return ingine::act::host_to_float(v); }

// Round-to-nearest-even FP32 -> ActT (BF16 on SM80+, FP16 clamped on INGINE_ACT_FP16).
inline ingine::ActT float_to_bf16(float f) { return ingine::act::host_from_float(f); }

inline float round_bf16(float f) { return ingine::act::host_round(f); }

#if INGINE_BACKEND_CUDA || INGINE_BACKEND_HIP

#define INGINE_TEST_CUDA(expr)                                                            \
  do {                                                                                    \
    cudaError_t e_ = (expr);                                                              \
    if (e_ != cudaSuccess) {                                                              \
      std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr,           \
                   cudaGetErrorString(e_));                                               \
      std::exit(1);                                                                       \
    }                                                                                     \
  } while (0)

// Returns true (after printing a skip line) when no CUDA device is usable.
inline bool skip_without_gpu() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
    std::printf("skip: no CUDA device\n");
    return true;
  }
  return false;
}

// RAII device buffer for tests. Construction and upload complete their default-stream writes
// before returning, so consumers on nonblocking streams never race initialization.
template <class T>
struct DeviceArray {
  T* ptr = nullptr;
  size_t count = 0;
  DeviceArray() = default;
  explicit DeviceArray(size_t n) : count(n) {
    INGINE_TEST_CUDA(cudaMalloc(&ptr, n * sizeof(T) + 16));
    INGINE_TEST_CUDA(cudaMemset(ptr, 0, n * sizeof(T) + 16));
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  }
  explicit DeviceArray(const std::vector<T>& h) : DeviceArray(h.size()) { upload(h); }
  ~DeviceArray() {
    if (ptr) cudaFree(ptr);
  }
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& o) noexcept : ptr(o.ptr), count(o.count) { o.ptr = nullptr; }
  void upload(const std::vector<T>& h) {
    INGINE_TEST_CUDA(cudaMemcpy(ptr, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
  }
  std::vector<T> download() const {
    std::vector<T> h(count);
    INGINE_TEST_CUDA(cudaMemcpy(h.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  T* get() const { return ptr; }
};

#endif  // INGINE_BACKEND_CUDA || INGINE_BACKEND_HIP

}  // namespace ingine::test

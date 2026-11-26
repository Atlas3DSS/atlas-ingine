// Unified pure-native CUDA / AMD ROCm HIP device and runtime compatibility header.
//
// CRITICAL HIP INVARIANT: Never define __CUDA_ARCH__ under INGINE_BACKEND_HIP. All NVIDIA inline PTX
// stays guarded by `defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)`. When INGINE_BACKEND_CUDA=1,
// this header is a direct pass-through to standard CUDA headers with zero runtime or codegen overhead.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

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

#if defined(INGINE_BACKEND_HIP) && INGINE_BACKEND_HIP

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

// Portable 2-element packed BF16 storage (ABI-identical across GCC host TUs and amdclang++ HIP TUs;
// avoids ROCm 7's Clang-only <hip/hip_bf16.h> builtins under host GCC).
struct alignas(4) ingine_hip_bfloat162 {
  hip_bfloat16 x;
  hip_bfloat16 y;
};

#if !(defined(__clang__) && defined(__HIP__))
inline unsigned short __half_as_ushort(__half x) noexcept {
  return static_cast<__half_raw>(x).x;
}
inline __half __ushort_as_half(unsigned short x) noexcept {
  __half_raw r{x};
  return __half(r);
}
#endif

#ifndef CUDART_INF_F
#define CUDART_INF_F (__builtin_inff())
#endif

// 1:1 CUDA -> HIP runtime type and constant aliases (no __CUDA_ARCH__ defined).
using cudaError_t               = hipError_t;
using cudaStream_t              = hipStream_t;
using cudaEvent_t               = hipEvent_t;
using cudaGraph_t                = hipGraph_t;
using cudaGraphExec_t            = hipGraphExec_t;
using cudaGraphNode_t            = hipGraphNode_t;
using cudaGraphConditionalHandle = unsigned long long;
using cudaDeviceProp             = hipDeviceProp_t;
using cudaPointerAttributes      = hipPointerAttribute_t;
using cudaUUID_t                = hipUUID;
using cudaMemcpyKind            = hipMemcpyKind;
using cudaStreamCaptureMode     = hipStreamCaptureMode;
using cudaStreamCaptureStatus   = hipStreamCaptureStatus;
using cudaFuncAttribute         = hipFuncAttribute;

inline constexpr hipError_t cudaSuccess                       = hipSuccess;
inline constexpr hipError_t cudaErrorNotReady                 = hipErrorNotReady;
inline constexpr hipError_t cudaErrorMemoryAllocation         = hipErrorMemoryAllocation;
inline constexpr hipError_t cudaErrorInvalidValue             = hipErrorInvalidValue;
inline constexpr hipError_t cudaErrorNotSupported             = hipErrorNotSupported;
inline constexpr hipError_t cudaErrorNotPermitted             = hipErrorNotSupported;
inline constexpr hipError_t cudaErrorOperatingSystem          = hipErrorOperatingSystem;
inline constexpr hipError_t cudaErrorInsufficientDriver       = hipErrorInsufficientDriver;
inline constexpr hipError_t cudaErrorCallRequiresNewerDriver  = hipErrorInsufficientDriver;
inline constexpr hipError_t cudaErrorUnsupportedPtxVersion    = hipErrorInvalidKernelFile;
inline constexpr hipError_t cudaErrorIllegalAddress           = hipErrorIllegalAddress;
inline constexpr hipError_t cudaErrorPeerAccessAlreadyEnabled = hipErrorPeerAccessAlreadyEnabled;

inline constexpr hipMemoryType cudaMemoryTypeHost    = hipMemoryTypeHost;
inline constexpr hipMemoryType cudaMemoryTypeDevice  = hipMemoryTypeDevice;
inline constexpr hipMemoryType cudaMemoryTypeManaged = hipMemoryTypeManaged;

inline constexpr hipMemcpyKind cudaMemcpyHostToHost     = hipMemcpyHostToHost;
inline constexpr hipMemcpyKind cudaMemcpyHostToDevice   = hipMemcpyHostToDevice;
inline constexpr hipMemcpyKind cudaMemcpyDeviceToHost   = hipMemcpyDeviceToHost;
inline constexpr hipMemcpyKind cudaMemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
inline constexpr hipMemcpyKind cudaMemcpyDefault        = hipMemcpyDefault;

inline constexpr unsigned int cudaStreamNonBlocking       = hipStreamNonBlocking;
inline constexpr unsigned int cudaEventDisableTiming      = hipEventDisableTiming;
inline constexpr unsigned int cudaHostAllocPortable       = hipHostMallocPortable;
inline constexpr unsigned int cudaHostAllocMapped         = hipHostMallocMapped;
inline constexpr unsigned int cudaHostRegisterPortable    = hipHostRegisterPortable;
inline constexpr unsigned int cudaHostRegisterMapped      = hipHostRegisterMapped;
inline constexpr unsigned int cudaHostRegisterReadOnly    = hipHostRegisterReadOnly;

inline constexpr hipStreamCaptureMode cudaStreamCaptureModeThreadLocal = hipStreamCaptureModeThreadLocal;
inline constexpr hipStreamCaptureMode cudaStreamCaptureModeRelaxed     = hipStreamCaptureModeRelaxed;
inline constexpr hipStreamCaptureStatus cudaStreamCaptureStatusNone    = hipStreamCaptureStatusNone;

inline constexpr hipFuncAttribute cudaFuncAttributeMaxDynamicSharedMemorySize =
    hipFuncAttributeMaxDynamicSharedMemorySize;

inline const char* cudaGetErrorString(hipError_t err) noexcept { return hipGetErrorString(err); }
inline const char* cudaGetErrorName(hipError_t err) noexcept { return hipGetErrorName(err); }
inline hipError_t cudaGetLastError() noexcept { return hipGetLastError(); }
inline hipError_t cudaGetDeviceCount(int* count) noexcept { return hipGetDeviceCount(count); }
inline hipError_t cudaGetDevice(int* dev) noexcept { return hipGetDevice(dev); }
inline hipError_t cudaSetDevice(int dev) noexcept { return hipSetDevice(dev); }
inline hipError_t cudaGetDeviceProperties(hipDeviceProp_t* prop, int dev) noexcept {
  return hipGetDeviceProperties(prop, dev);
}
inline hipError_t cudaDeviceGetPCIBusId(char* pci, int len, int dev) noexcept {
  return hipDeviceGetPCIBusId(pci, len, dev);
}
inline hipError_t cudaDeviceSynchronize() noexcept { return hipDeviceSynchronize(); }
inline hipError_t cudaDeviceCanAccessPeer(int* can, int dev, int peer) noexcept {
  return hipDeviceCanAccessPeer(can, dev, peer);
}
inline hipError_t cudaDeviceEnablePeerAccess(int peer, unsigned int flags) noexcept {
  return hipDeviceEnablePeerAccess(peer, flags);
}
inline hipError_t cudaDeviceDisablePeerAccess(int peer) noexcept { return hipDeviceDisablePeerAccess(peer); }

inline hipError_t cudaMemGetInfo(std::size_t* free_b, std::size_t* total_b) noexcept {
  return hipMemGetInfo(free_b, total_b);
}
inline hipError_t cudaMalloc(void** ptr, std::size_t bytes) noexcept { return hipMalloc(ptr, bytes); }
template <typename T>
inline hipError_t cudaMalloc(T** ptr, std::size_t bytes) noexcept {
  return hipMalloc(reinterpret_cast<void**>(ptr), bytes);
}
inline hipError_t cudaFree(void* ptr) noexcept { return hipFree(ptr); }
inline hipError_t cudaHostAlloc(void** ptr, std::size_t bytes, unsigned int flags) noexcept {
  return hipHostMalloc(ptr, bytes, flags);
}
template <typename T>
inline hipError_t cudaHostAlloc(T** ptr, std::size_t bytes, unsigned int flags) noexcept {
  return hipHostMalloc(reinterpret_cast<void**>(ptr), bytes, flags);
}
inline hipError_t cudaFreeHost(void* ptr) noexcept { return hipHostFree(ptr); }
inline hipError_t cudaHostRegister(void* ptr, std::size_t bytes, unsigned int flags) noexcept {
  return hipHostRegister(ptr, bytes, flags);
}
inline hipError_t cudaHostUnregister(void* ptr) noexcept { return hipHostUnregister(ptr); }
inline hipError_t cudaHostGetDevicePointer(void** dev_ptr, void* host_ptr, unsigned int flags) noexcept {
  return hipHostGetDevicePointer(dev_ptr, host_ptr, flags);
}
inline hipError_t cudaPointerGetAttributes(hipPointerAttribute_t* attr, const void* ptr) noexcept {
  return hipPointerGetAttributes(attr, ptr);
}

inline hipError_t cudaMemcpy(void* dst, const void* src, std::size_t bytes, hipMemcpyKind kind) noexcept {
  return hipMemcpy(dst, src, bytes, kind);
}
inline hipError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes, hipMemcpyKind kind,
                                  hipStream_t stream = nullptr) noexcept {
  return hipMemcpyAsync(dst, src, bytes, kind, stream);
}
inline hipError_t cudaMemcpy2DAsync(void* dst, std::size_t dpitch, const void* src, std::size_t spitch,
                                    std::size_t width, std::size_t height, hipMemcpyKind kind,
                                    hipStream_t stream = nullptr) noexcept {
  return hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream);
}
inline hipError_t cudaMemcpyPeer(void* dst, int dst_dev, const void* src, int src_dev, std::size_t bytes) noexcept {
  return hipMemcpyPeer(dst, dst_dev, src, src_dev, bytes);
}
inline hipError_t cudaMemset(void* dst, int value, std::size_t bytes) noexcept { return hipMemset(dst, value, bytes); }
inline hipError_t cudaMemsetAsync(void* dst, int value, std::size_t bytes, hipStream_t stream = nullptr) noexcept {
  return hipMemsetAsync(dst, value, bytes, stream);
}

inline hipError_t cudaStreamCreate(hipStream_t* s) noexcept { return hipStreamCreate(s); }
inline hipError_t cudaStreamCreateWithFlags(hipStream_t* s, unsigned int flags) noexcept {
  return hipStreamCreateWithFlags(s, flags);
}
inline hipError_t cudaStreamDestroy(hipStream_t s) noexcept { return hipStreamDestroy(s); }
inline hipError_t cudaStreamSynchronize(hipStream_t s) noexcept { return hipStreamSynchronize(s); }
inline hipError_t cudaStreamWaitEvent(hipStream_t s, hipEvent_t e, unsigned int flags = 0) noexcept {
  return hipStreamWaitEvent(s, e, flags);
}
inline hipError_t cudaStreamIsCapturing(hipStream_t s, hipStreamCaptureStatus* st) noexcept {
  return hipStreamIsCapturing(s, st);
}
inline hipError_t cudaStreamBeginCapture(hipStream_t s, hipStreamCaptureMode mode) noexcept {
  return hipStreamBeginCapture(s, mode);
}
inline hipError_t cudaStreamEndCapture(hipStream_t s, hipGraph_t* graph) noexcept {
  return hipStreamEndCapture(s, graph);
}

inline hipError_t cudaEventCreate(hipEvent_t* e) noexcept { return hipEventCreate(e); }
inline hipError_t cudaEventCreateWithFlags(hipEvent_t* e, unsigned int flags) noexcept {
  return hipEventCreateWithFlags(e, flags);
}
inline hipError_t cudaEventDestroy(hipEvent_t e) noexcept { return hipEventDestroy(e); }
inline hipError_t cudaEventRecord(hipEvent_t e, hipStream_t s = nullptr) noexcept { return hipEventRecord(e, s); }
inline hipError_t cudaEventSynchronize(hipEvent_t e) noexcept { return hipEventSynchronize(e); }
inline hipError_t cudaEventQuery(hipEvent_t e) noexcept { return hipEventQuery(e); }
inline hipError_t cudaEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) noexcept {
  return hipEventElapsedTime(ms, start, stop);
}

inline hipError_t cudaGraphCreate(hipGraph_t* g, unsigned int flags) noexcept { return hipGraphCreate(g, flags); }
inline hipError_t cudaGraphDestroy(hipGraph_t g) noexcept { return hipGraphDestroy(g); }
inline hipError_t cudaGraphExecDestroy(hipGraphExec_t ge) noexcept { return hipGraphExecDestroy(ge); }
inline hipError_t cudaGraphLaunch(hipGraphExec_t ge, hipStream_t s) noexcept { return hipGraphLaunch(ge, s); }
inline hipError_t cudaGraphUpload(hipGraphExec_t ge, hipStream_t s) noexcept { return hipGraphUpload(ge, s); }
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* ge, hipGraph_t g, unsigned long long /*flags*/ = 0) noexcept {
  return hipGraphInstantiate(ge, g, nullptr, nullptr, 0);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* ge, hipGraph_t g, hipGraphNode_t* err_node, char* log_buf,
                                       std::size_t buf_sz) noexcept {
  return hipGraphInstantiate(ge, g, err_node, log_buf, buf_sz);
}

template <typename Func>
inline hipError_t cudaFuncSetAttribute(Func fn, hipFuncAttribute attr, int value) noexcept {
  return hipFuncSetAttribute(reinterpret_cast<const void*>(fn), attr, value);
}

template <typename Sym>
inline hipError_t cudaGetSymbolAddress(void** dev_ptr, const Sym& sym) noexcept {
  return hipGetSymbolAddress(dev_ptr, sym);
}

// Logical 32-lane shuffle/ballot/vote/syncwarp/cache-bypass shims for shared kernels under ROCm 6 & 7.
#if defined(__HIPCC__)
template <typename T>
__device__ __forceinline__ T __shfl_xor_sync(unsigned /*mask*/, T val, int lane_mask, int width = 32) {
  return __shfl_xor(val, lane_mask, width);
}
template <typename T>
__device__ __forceinline__ T __shfl_down_sync(unsigned /*mask*/, T val, unsigned delta, int width = 32) {
  return __shfl_down(val, delta, width);
}
template <typename T>
__device__ __forceinline__ T __shfl_up_sync(unsigned /*mask*/, T val, unsigned delta, int width = 32) {
  return __shfl_up(val, delta, width);
}
template <typename T>
__device__ __forceinline__ T __shfl_sync(unsigned /*mask*/, T val, int src_lane, int width = 32) {
  return __shfl(val, src_lane, width);
}
__device__ __forceinline__ unsigned __ballot_sync(unsigned /*mask*/, int pred) {
#if defined(INGINE_HIP_WAVE64)
  return static_cast<unsigned>(static_cast<std::uint64_t>(__ballot(pred)) >> (__lane_id() & 32));
#else
  return static_cast<unsigned>(__ballot(pred));
#endif
}
__device__ __forceinline__ int __any_sync(unsigned mask, int pred) {
  return (__ballot_sync(mask, pred) & mask) != 0u ? 1 : 0;
}
__device__ __forceinline__ int __all_sync(unsigned mask, int pred) {
  return (__ballot_sync(mask, pred) & mask) == mask ? 1 : 0;
}
#if defined(HIP_VERSION_MAJOR) && (HIP_VERSION_MAJOR >= 7)
// ROCm 7 provides 0-arg __syncwarp() and 64-bit-only template __syncwarp(MaskT); add a 32-bit
// overload for shared CUDA/HIP kernels passing 32-bit logical warp masks.
__device__ __forceinline__ void __syncwarp(unsigned /*mask*/) {
#if defined(__HIP_DEVICE_COMPILE__)
  ::__syncwarp();
#endif
}
#else
__device__ __forceinline__ void __syncwarp(unsigned /*mask*/ = 0xffffffffu) {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#endif
}
#endif
template <typename T>
__device__ __forceinline__ T __ldcg(const T* ptr) {
  return *ptr;
}
template <typename T>
__device__ __forceinline__ void __stcg(T* ptr, const T& val) {
  *ptr = val;
}
#endif  // __HIPCC__

#elif defined(INGINE_BACKEND_CUDA) && INGINE_BACKEND_CUDA

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#endif  // backend selection

namespace ingine::gpu {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define INGINE_GPU_HD __host__ __device__ __forceinline__
#else
#define INGINE_GPU_HD inline
#endif

using fp8_e4m3_storage_t   = std::uint8_t;
using fp8_e4m3x2_storage_t = std::uint16_t;

// Bit-exact OCP E4M3FN conversion with round-to-nearest-even and +-448 saturation, matching
// NVIDIA `__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)` on every IEEE-754 float32 input.
INGINE_GPU_HD std::uint8_t float_to_e4m3fn_sat_portable(float x) noexcept {
  std::uint32_t bits = 0;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  bits = __float_as_uint(x);
#else
  std::memcpy(&bits, &x, sizeof(bits));
#endif
  const std::uint32_t sign = (bits >> 24) & 0x80u;
  const std::uint32_t mag  = bits & 0x7fffffffu;
  if (mag > 0x7f800000u) return 0x7fu;  // Canonical quiet NaN (matches cvt.rn.satfinite.e4m3x2.f32 / cuda_fp8.hpp)
  if (mag >= 0x43e00000u) return static_cast<std::uint8_t>(sign | 0x7eu);  // |x| >= 448.0 -> saturate
  if (mag == 0u) return static_cast<std::uint8_t>(sign);

  const int exp = static_cast<int>((mag >> 23) & 0xffu) - 127;
  if (exp < -10) return static_cast<std::uint8_t>(sign);  // |x| < 2^-10 -> rounds to 0

  if (exp < -6) {
    // Subnormal E4M3FN range: exp in [-10, -7], value = k * 2^-9 (k in 0..8; k==8 overflows to min normal 0x08).
    const std::uint32_t mant = (mag & 0x7fffffu) | 0x800000u;
    const int shift          = 14 - exp;  // 21..24
    std::uint32_t q          = mant >> shift;
    const std::uint32_t half = 1u << (shift - 1);
    const std::uint32_t rem  = mant & ((1u << shift) - 1u);
    if (rem > half || (rem == half && (q & 1u) != 0u)) ++q;
    return static_cast<std::uint8_t>(sign | q);
  }

  // Normal E4M3FN range: exp in [-6, 8], biased exponent in [1, 15].
  std::uint32_t q         = (static_cast<std::uint32_t>(exp + 7) << 3) | ((mag >> 20) & 0x7u);
  const std::uint32_t rem = mag & 0xfffffu;
  if (rem > 0x80000u || (rem == 0x80000u && (q & 1u) != 0u)) ++q;
  if (q > 0x7eu) q = 0x7eu;
  return static_cast<std::uint8_t>(sign | q);
}

// Bit-exact OCP E4M3FN -> IEEE-754 binary16 bit pattern (all 256 codes including +-NaN).
INGINE_GPU_HD std::uint16_t e4m3fn_to_f16_bits_portable(std::uint8_t code) noexcept {
  const std::uint32_t b    = code;
  const std::uint32_t s    = (b & 0x80u) << 8;
  const std::uint32_t a    = b & 0x7fu;
  if (a == 0x7fu) return 0x7fffu;  // FP16 canonical NaN, discard sign (matches cvt.rn.f16x2.e4m3x2 / cuda_fp8.hpp)
  const std::uint32_t norm = (a + 64u) << 7;
  const std::uint32_t sub  = a ? (0x1800u + (((0xBA986400u >> (a * 4u)) & 0xfu) << 8)) : 0u;
  return static_cast<std::uint16_t>(s | (a >= 8u ? norm : sub));
}

// Bit-exact OCP E4M3FN -> IEEE-754 binary32 (exact for all 256 codes including +-NaN).
INGINE_GPU_HD float e4m3fn_to_float_portable(std::uint8_t code) noexcept {
  const std::uint32_t h = e4m3fn_to_f16_bits_portable(code);
  const std::uint32_t s = (h & 0x8000u) << 16;
  const std::uint32_t e = (h >> 10) & 0x1fu;
  const std::uint32_t m = h & 0x03ffu;
  std::uint32_t w       = 0;
  if (e == 0u) {
    w = s;  // e4m3fn_to_f16_bits_portable never produces FP16 subnormals (min E4M3 subnormal = 2^-9, normal in FP16)
  } else if (e == 31u) {
    w = s | 0x7fc00000u;  // canonical quiet NaN
  } else {
    w = s | ((e + 112u) << 23) | (m << 13);
  }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float(w);
#else
  float f = 0.0f;
  std::memcpy(&f, &w, sizeof(f));
  return f;
#endif
}

INGINE_GPU_HD std::uint8_t float_to_e4m3fn_sat(float x) noexcept {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && defined(INGINE_BACKEND_CUDA) && INGINE_BACKEND_CUDA
  return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
#else
  return float_to_e4m3fn_sat_portable(x);
#endif
}

INGINE_GPU_HD std::uint16_t float2_to_e4m3fnx2_sat(float a, float b) noexcept {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && defined(INGINE_BACKEND_CUDA) && INGINE_BACKEND_CUDA
  return static_cast<std::uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3));
#else
  return static_cast<std::uint16_t>(float_to_e4m3fn_sat_portable(a)) |
         static_cast<std::uint16_t>(static_cast<std::uint16_t>(float_to_e4m3fn_sat_portable(b)) << 8);
#endif
}

#if defined(__CUDACC__) || defined(__HIPCC__)
__device__ __forceinline__ float e4m3fn_to_float(std::uint8_t code) noexcept {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && defined(INGINE_BACKEND_CUDA) && INGINE_BACKEND_CUDA
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
#else
  return e4m3fn_to_float_portable(code);
#endif
}

__device__ __forceinline__ __half2 e4m3fnx2_to_half2(std::uint16_t pair) noexcept {
#if defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__) && defined(INGINE_BACKEND_CUDA) && INGINE_BACKEND_CUDA
  return __half2(__nv_cvt_fp8x2_to_halfraw2(pair, __NV_E4M3));
#else
  const std::uint32_t lo = e4m3fn_to_f16_bits_portable(static_cast<std::uint8_t>(pair & 0xffu));
  const std::uint32_t hi = e4m3fn_to_f16_bits_portable(static_cast<std::uint8_t>(pair >> 8));
  const std::uint32_t u  = lo | (hi << 16);
  return *reinterpret_cast<const __half2*>(&u);
#endif
}

__device__ __forceinline__ float2 e4m3fnx2_to_float2(std::uint16_t pair) noexcept {
  return __half22float2(e4m3fnx2_to_half2(pair));
}
#endif

// Shared-memory handle abstraction: 32-bit shared-space offset on CUDA (for PTX ldmatrix),
// generic pointer on HIP (avoiding 64-to-32 bit truncation of LDS addresses).
#if defined(INGINE_BACKEND_HIP) && INGINE_BACKEND_HIP
using SmemAddr = const std::uint8_t*;
#if defined(__HIPCC__)
__device__ __forceinline__ SmemAddr make_smem_addr(const void* ptr) noexcept {
  return static_cast<const std::uint8_t*>(ptr);
}
#endif
#else
using SmemAddr = unsigned;
#if defined(__CUDACC__)
__device__ __forceinline__ SmemAddr make_smem_addr(const void* ptr) noexcept {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}
#endif
#endif

#undef INGINE_GPU_HD

}  // namespace ingine::gpu

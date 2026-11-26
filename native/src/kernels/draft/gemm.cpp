#include "kernels/draft/gemm.h"

#include "core/device.h"

#if defined(INGINE_BACKEND_HIP) && INGINE_BACKEND_HIP
#ifndef HIPBLAS_V2
#define HIPBLAS_V2
#endif
#include <hipblas/hipblas.h>
#if defined(INGINE_HIPBLASLT_AVAILABLE) && INGINE_HIPBLASLT_AVAILABLE
#include <hipblaslt/hipblaslt.h>
#endif
#elif defined(INGINE_BACKEND_CUDA) && INGINE_BACKEND_CUDA
#include <cublasLt.h>
#endif

#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace ingine::kernels::draft {
namespace {

#if defined(INGINE_BACKEND_HIP) && INGINE_BACKEND_HIP

void hipblas_check(hipblasStatus_t s, const char* what) {
  if (s != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("hipBLAS ") + what + " failed: " + std::to_string(int(s)));
  }
}

inline constexpr hipDataType kHipActType = kActIsFP16 ? HIP_R_16F : HIP_R_16BF;

#else

void lt_check(cublasStatus_t s, const char* what) {
  if (s != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLASLt ") + what + " failed: " + std::to_string(int(s)));
  }
}

#endif

}  // namespace

#if defined(INGINE_BACKEND_HIP) && INGINE_BACKEND_HIP

struct Bf16Gemm::Impl {
#if defined(INGINE_HIPBLASLT_AVAILABLE) && INGINE_HIPBLASLT_AVAILABLE
  struct Plan {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    bool fallback_hipblas = false;
  };
  hipblasLtHandle_t lt_handle = nullptr;
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  std::map<std::tuple<int, int, int>, Plan> plans;
#endif
  hipblasHandle_t blas_handle = nullptr;
};

Bf16Gemm::Bf16Gemm(std::size_t workspace_bytes) : impl_(new Impl) {
  try {
    hipblas_check(hipblasCreate(&impl_->blas_handle), "create");
#if defined(INGINE_HIPBLASLT_AVAILABLE) && INGINE_HIPBLASLT_AVAILABLE
    impl_->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&impl_->lt_handle) == HIPBLAS_STATUS_SUCCESS) {
      if (hipMalloc(&impl_->workspace, impl_->workspace_bytes) != hipSuccess) {
        hipblasLtDestroy(impl_->lt_handle);
        impl_->lt_handle = nullptr;
        impl_->workspace = nullptr;
      }
    }
#else
    (void)workspace_bytes;
#endif
  } catch (...) {
#if defined(INGINE_HIPBLASLT_AVAILABLE) && INGINE_HIPBLASLT_AVAILABLE
    if (impl_->workspace) hipFree(impl_->workspace);
    if (impl_->lt_handle) hipblasLtDestroy(impl_->lt_handle);
#endif
    if (impl_->blas_handle) hipblasDestroy(impl_->blas_handle);
    delete impl_;
    throw;
  }
}

Bf16Gemm::~Bf16Gemm() {
#if defined(INGINE_HIPBLASLT_AVAILABLE) && INGINE_HIPBLASLT_AVAILABLE
  for (auto& [key, p] : impl_->plans) {
    if (p.a) hipblasLtMatrixLayoutDestroy(p.a);
    if (p.b) hipblasLtMatrixLayoutDestroy(p.b);
    if (p.c) hipblasLtMatrixLayoutDestroy(p.c);
    if (p.op) hipblasLtMatmulDescDestroy(p.op);
  }
  if (impl_->workspace) hipFree(impl_->workspace);
  if (impl_->lt_handle) hipblasLtDestroy(impl_->lt_handle);
#endif
  if (impl_->blas_handle) hipblasDestroy(impl_->blas_handle);
  delete impl_;
}

void Bf16Gemm::run(const ActT* a, int n, const ActT* w, int N, int K, ActT* c, cudaStream_t stream) {
  if (n <= 0) return;
  const float alpha = 1.f, beta = 0.f;

#if defined(INGINE_HIPBLASLT_AVAILABLE) && INGINE_HIPBLASLT_AVAILABLE
  if (impl_->lt_handle != nullptr && impl_->workspace != nullptr) {
    const auto key = std::make_tuple(n, N, K);
    auto it = impl_->plans.find(key);
    if (it == impl_->plans.end()) {
      Impl::Plan p;
      auto cleanup = [&] {
        if (p.a) { hipblasLtMatrixLayoutDestroy(p.a); p.a = nullptr; }
        if (p.b) { hipblasLtMatrixLayoutDestroy(p.b); p.b = nullptr; }
        if (p.c) { hipblasLtMatrixLayoutDestroy(p.c); p.c = nullptr; }
        if (p.op) { hipblasLtMatmulDescDestroy(p.op); p.op = nullptr; }
      };
      hipblasStatus_t s = hipblasLtMatmulDescCreate(&p.op, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
      if (s == HIPBLAS_STATUS_SUCCESS)
        s = hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
      if (s == HIPBLAS_STATUS_SUCCESS)
        s = hipblasLtMatmulDescSetAttribute(p.op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
      if (s == HIPBLAS_STATUS_SUCCESS) s = hipblasLtMatrixLayoutCreate(&p.a, kHipActType, K, N, K);
      if (s == HIPBLAS_STATUS_SUCCESS) s = hipblasLtMatrixLayoutCreate(&p.b, kHipActType, K, n, K);
      if (s == HIPBLAS_STATUS_SUCCESS) s = hipblasLtMatrixLayoutCreate(&p.c, kHipActType, N, n, N);
      hipblasLtMatmulPreference_t pref = nullptr;
      hipblasLtMatmulHeuristicResult_t result{};
      int found = 0;
      if (s == HIPBLAS_STATUS_SUCCESS) s = hipblasLtMatmulPreferenceCreate(&pref);
      if (s == HIPBLAS_STATUS_SUCCESS)
        s = hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &impl_->workspace_bytes, sizeof(impl_->workspace_bytes));
      if (s == HIPBLAS_STATUS_SUCCESS)
        s = hipblasLtMatmulAlgoGetHeuristic(impl_->lt_handle, p.op, p.a, p.b, p.c, p.c, pref, 1, &result, &found);
      if (pref) hipblasLtMatmulPreferenceDestroy(pref);
      if (s == HIPBLAS_STATUS_SUCCESS && found > 0) {
        p.algo = result.algo;
        p.fallback_hipblas = false;
      } else {
        cleanup();
        p.fallback_hipblas = true;
      }
      it = impl_->plans.emplace(key, p).first;
    }
    if (!it->second.fallback_hipblas) {
      const Impl::Plan& p = it->second;
      hipblas_check(hipblasLtMatmul(impl_->lt_handle, p.op, &alpha, w, p.a, a, p.b, &beta, c, p.c, c, p.c, &p.algo,
                                    impl_->workspace, impl_->workspace_bytes, stream),
                    "LtMatmul");
      return;
    }
  }
#endif

  hipblas_check(hipblasSetStream(impl_->blas_handle, stream), "setStream");
  hipblas_check(hipblasGemmEx(impl_->blas_handle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, n, K, &alpha, w, kHipActType, K, a,
                              kHipActType, K, &beta, c, kHipActType, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                "GemmEx");
}

#else

struct Bf16Gemm::Impl {
  struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
  };

  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  std::size_t workspace_bytes = 0;
  std::map<std::tuple<int, int, int>, Plan> plans;

  Plan& plan(int n, int N, int K) {
    const auto key = std::make_tuple(n, N, K);
    if (auto it = plans.find(key); it != plans.end()) return it->second;
    Plan p;
    lt_check(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "matmul desc");
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
    lt_check(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
    lt_check(cublasLtMatrixLayoutCreate(&p.a, kCublasActType, K, N, K), "layout w");
    lt_check(cublasLtMatrixLayoutCreate(&p.b, kCublasActType, K, n, K), "layout a");
    lt_check(cublasLtMatrixLayoutCreate(&p.c, kCublasActType, N, n, N), "layout c");
    cublasLtMatmulPreference_t pref = nullptr;
    lt_check(cublasLtMatmulPreferenceCreate(&pref), "preference");
    lt_check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
                                                  sizeof(workspace_bytes)),
             "workspace preference");
    cublasLtMatmulHeuristicResult_t result{};
    int found = 0;
    const cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(handle, p.op, p.a, p.b, p.c, p.c, pref, 1, &result, &found);
    cublasLtMatmulPreferenceDestroy(pref);
    if (s != CUBLAS_STATUS_SUCCESS || found == 0) {
      cublasLtMatrixLayoutDestroy(p.a);
      cublasLtMatrixLayoutDestroy(p.b);
      cublasLtMatrixLayoutDestroy(p.c);
      cublasLtMatmulDescDestroy(p.op);
      throw std::runtime_error("cuBLASLt: no activation-type algorithm for [" + std::to_string(n) + ", " +
                               std::to_string(K) + "] x [" + std::to_string(K) + ", " + std::to_string(N) + "]");
    }
    p.algo = result.algo;
    return plans.emplace(key, p).first->second;
  }
};

Bf16Gemm::Bf16Gemm(std::size_t workspace_bytes) : impl_(new Impl) {
  impl_->workspace_bytes = workspace_bytes;
  try {
    lt_check(cublasLtCreate(&impl_->handle), "create");
    CUDA_CHECK(cudaMalloc(&impl_->workspace, impl_->workspace_bytes));
  } catch (...) {
    if (impl_->handle != nullptr) cublasLtDestroy(impl_->handle);
    delete impl_;
    throw;
  }
}

Bf16Gemm::~Bf16Gemm() {
  for (auto& [key, p] : impl_->plans) {
    cublasLtMatrixLayoutDestroy(p.a);
    cublasLtMatrixLayoutDestroy(p.b);
    cublasLtMatrixLayoutDestroy(p.c);
    cublasLtMatmulDescDestroy(p.op);
  }
  if (impl_->workspace != nullptr) cudaFree(impl_->workspace);
  if (impl_->handle != nullptr) cublasLtDestroy(impl_->handle);
  delete impl_;
}

// Row-major c [n, N] = a [n, K] w[N, K]^T is column-major c^T [N, n] = op(w) a^T with w seen as a
// column-major [K, N] matrix (transposed) and a as column-major [K, n].
void Bf16Gemm::run(const ActT* a, int n, const ActT* w, int N, int K, ActT* c, cudaStream_t stream) {
  if (n <= 0) return;
  Impl::Plan& p = impl_->plan(n, N, K);
  const float alpha = 1.f, beta = 0.f;
  lt_check(cublasLtMatmul(impl_->handle, p.op, &alpha, w, p.a, a, p.b, &beta, c, p.c, c, p.c, &p.algo,
                          impl_->workspace, impl_->workspace_bytes, stream),
           "matmul");
}

#endif

}  // namespace ingine::kernels::draft

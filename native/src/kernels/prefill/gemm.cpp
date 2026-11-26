// Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
// Dense BF16/FP16 GEMM for prefill projections: FP32 compute and FP32 split-K reductions.
// Uses cuBLASLt on CUDA; uses hipBLASLt (when available) with automatic hipblasGemmEx fallback on HIP.
#include "kernels/prefill/prefill.h"

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

namespace ingine::kernels::prefill {

namespace {

constexpr std::size_t kWorkspaceBytes = 32u << 20;

#if defined(INGINE_BACKEND_HIP) && INGINE_BACKEND_HIP

void hipblas_check(hipblasStatus_t status, const char* what) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("hipBLAS ") + what + " failed: status " +
                             std::to_string(static_cast<int>(status)));
  }
}

inline constexpr hipDataType kHipActType = kActIsFP16 ? HIP_R_16F : HIP_R_16BF;

#else

void cublas_check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLASLt ") + what + " failed: status " +
                             std::to_string(static_cast<int>(status)));
  }
}

#endif

}  // namespace

#if defined(INGINE_BACKEND_HIP) && INGINE_BACKEND_HIP

struct Gemm::Impl {
#if defined(INGINE_HIPBLASLT_AVAILABLE) && INGINE_HIPBLASLT_AVAILABLE
  struct Plan {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t la = nullptr;
    hipblasLtMatrixLayout_t lb = nullptr;
    hipblasLtMatrixLayout_t lc = nullptr;
    hipblasLtMatmulAlgo_t algo{};
    bool fallback_hipblas = false;
  };
  hipblasLtHandle_t lt_handle = nullptr;
  void* workspace = nullptr;
  std::map<std::tuple<int, int, int, int, int, int, bool>, Plan> plans;

  void destroy_plans() {
    for (auto& [k, p] : plans) {
      if (p.la) hipblasLtMatrixLayoutDestroy(p.la);
      if (p.lb) hipblasLtMatrixLayoutDestroy(p.lb);
      if (p.lc) hipblasLtMatrixLayoutDestroy(p.lc);
      if (p.op) hipblasLtMatmulDescDestroy(p.op);
    }
    plans.clear();
  }
#endif
  hipblasHandle_t blas_handle = nullptr;
};

Gemm::Gemm() : impl_(new Impl) {
  try {
    hipblas_check(hipblasCreate(&impl_->blas_handle), "create");
#if defined(INGINE_HIPBLASLT_AVAILABLE) && INGINE_HIPBLASLT_AVAILABLE
    if (hipblasLtCreate(&impl_->lt_handle) == HIPBLAS_STATUS_SUCCESS) {
      if (hipMalloc(&impl_->workspace, kWorkspaceBytes) != hipSuccess) {
        hipblasLtDestroy(impl_->lt_handle);
        impl_->lt_handle = nullptr;
        impl_->workspace = nullptr;
      }
    }
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

Gemm::~Gemm() {
#if defined(INGINE_HIPBLASLT_AVAILABLE) && INGINE_HIPBLASLT_AVAILABLE
  impl_->destroy_plans();
  if (impl_->workspace) hipFree(impl_->workspace);
  if (impl_->lt_handle) hipblasLtDestroy(impl_->lt_handle);
#endif
  if (impl_->blas_handle) hipblasDestroy(impl_->blas_handle);
  delete impl_;
}

void Gemm::run(const ActT* A, int lda, const ActT* B, int ldb, void* C, int ldc, bool c_fp32,
               int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) throw std::invalid_argument("prefill Gemm: empty shape");
  const float alpha = 1.f, beta = 0.f;
  const hipDataType c_type = c_fp32 ? HIP_R_32F : kHipActType;

#if defined(INGINE_HIPBLASLT_AVAILABLE) && INGINE_HIPBLASLT_AVAILABLE
  if (impl_->lt_handle != nullptr && impl_->workspace != nullptr) {
    const auto key = std::make_tuple(M, N, K, lda, ldb, ldc, c_fp32);
    auto it = impl_->plans.find(key);
    if (it == impl_->plans.end()) {
      Impl::Plan plan;
      auto cleanup = [&] {
        if (plan.la) { hipblasLtMatrixLayoutDestroy(plan.la); plan.la = nullptr; }
        if (plan.lb) { hipblasLtMatrixLayoutDestroy(plan.lb); plan.lb = nullptr; }
        if (plan.lc) { hipblasLtMatrixLayoutDestroy(plan.lc); plan.lc = nullptr; }
        if (plan.op) { hipblasLtMatmulDescDestroy(plan.op); plan.op = nullptr; }
      };
      hipblasStatus_t st = hipblasLtMatmulDescCreate(&plan.op, HIPBLAS_COMPUTE_32F, HIP_R_32F);
      const hipblasOperation_t ta = HIPBLAS_OP_T, tb = HIPBLAS_OP_N;
      if (st == HIPBLAS_STATUS_SUCCESS)
        st = hipblasLtMatmulDescSetAttribute(plan.op, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta));
      if (st == HIPBLAS_STATUS_SUCCESS)
        st = hipblasLtMatmulDescSetAttribute(plan.op, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb));
      if (st == HIPBLAS_STATUS_SUCCESS)
        st = hipblasLtMatrixLayoutCreate(&plan.la, kHipActType, K, N, ldb);
      if (st == HIPBLAS_STATUS_SUCCESS)
        st = hipblasLtMatrixLayoutCreate(&plan.lb, kHipActType, K, M, lda);
      if (st == HIPBLAS_STATUS_SUCCESS)
        st = hipblasLtMatrixLayoutCreate(&plan.lc, c_type, N, M, ldc);

      hipblasLtMatmulPreference_t pref = nullptr;
      hipblasLtMatmulHeuristicResult_t result{};
      int found = 0;
      if (st == HIPBLAS_STATUS_SUCCESS) st = hipblasLtMatmulPreferenceCreate(&pref);
      const std::size_t ws = kWorkspaceBytes;
      if (st == HIPBLAS_STATUS_SUCCESS)
        st = hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
      if (st == HIPBLAS_STATUS_SUCCESS)
        st = hipblasLtMatmulAlgoGetHeuristic(impl_->lt_handle, plan.op, plan.la, plan.lb, plan.lc, plan.lc, pref, 1,
                                             &result, &found);
      if (pref) hipblasLtMatmulPreferenceDestroy(pref);
      if (st == HIPBLAS_STATUS_SUCCESS && found > 0) {
        plan.algo = result.algo;
        plan.fallback_hipblas = false;
      } else {
        cleanup();
        plan.fallback_hipblas = true;
      }
      it = impl_->plans.emplace(key, plan).first;
    }
    if (!it->second.fallback_hipblas) {
      const Impl::Plan& plan = it->second;
      hipblas_check(hipblasLtMatmul(impl_->lt_handle, plan.op, &alpha, B, plan.la, A, plan.lb, &beta, C, plan.lc, C,
                                    plan.lc, &plan.algo, impl_->workspace, kWorkspaceBytes, stream),
                    "LtMatmul");
      return;
    }
  }
#endif

  hipblas_check(hipblasSetStream(impl_->blas_handle, stream), "setStream");
  hipblas_check(hipblasGemmEx(impl_->blas_handle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, M, K, &alpha, B, kHipActType, ldb, A,
                              kHipActType, lda, &beta, C, c_type, ldc, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT),
                "GemmEx");
}

#else

struct Gemm::Impl {
  struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr;
    cublasLtMatrixLayout_t lb = nullptr;
    cublasLtMatrixLayout_t lc = nullptr;
    cublasLtMatmulAlgo_t algo{};
  };

  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  // (M, N, K, lda, ldb, ldc, fp32 output) -> cached descriptors + heuristic algorithm
  std::map<std::tuple<int, int, int, int, int, int, bool>, Plan> plans;

  void destroy_plans() {
    for (auto& [k, p] : plans) {
      if (p.la) cublasLtMatrixLayoutDestroy(p.la);
      if (p.lb) cublasLtMatrixLayoutDestroy(p.lb);
      if (p.lc) cublasLtMatrixLayoutDestroy(p.lc);
      if (p.op) cublasLtMatmulDescDestroy(p.op);
    }
    plans.clear();
  }
};

Gemm::Gemm() : impl_(new Impl) {
  try {
    cublas_check(cublasLtCreate(&impl_->handle), "create");
    if (cudaMalloc(&impl_->workspace, kWorkspaceBytes) != cudaSuccess)
      throw std::runtime_error("prefill Gemm: workspace allocation failed");
  } catch (...) {
    if (impl_->handle) cublasLtDestroy(impl_->handle);
    delete impl_;
    throw;
  }
}

Gemm::~Gemm() {
  impl_->destroy_plans();
  if (impl_->workspace) cudaFree(impl_->workspace);
  if (impl_->handle) cublasLtDestroy(impl_->handle);
  delete impl_;
}

// Row-major C[M, N] = A[M, K] . B[N, K]^T is column-major C^T[N, M] = B^T(op T on the K x N matrix B) . A
// (the K x M matrix A).
void Gemm::run(const ActT* A, int lda, const ActT* B, int ldb, void* C, int ldc, bool c_fp32,
               int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0) throw std::invalid_argument("prefill Gemm: empty shape");
  const auto key = std::make_tuple(M, N, K, lda, ldb, ldc, c_fp32);
  auto it = impl_->plans.find(key);
  if (it == impl_->plans.end()) {
    Impl::Plan plan;
    auto cleanup = [&] {
      if (plan.la) cublasLtMatrixLayoutDestroy(plan.la);
      if (plan.lb) cublasLtMatrixLayoutDestroy(plan.lb);
      if (plan.lc) cublasLtMatrixLayoutDestroy(plan.lc);
      if (plan.op) cublasLtMatmulDescDestroy(plan.op);
    };
    try {
      cublas_check(cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F), "desc create");
      const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
      cublas_check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transa");
      cublas_check(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transb");
      cublas_check(cublasLtMatrixLayoutCreate(&plan.la, kCublasActType, K, N, ldb), "layout a");
      cublas_check(cublasLtMatrixLayoutCreate(&plan.lb, kCublasActType, K, M, lda), "layout b");
      cublas_check(cublasLtMatrixLayoutCreate(&plan.lc, c_fp32 ? CUDA_R_32F : kCublasActType, N, M, ldc), "layout c");

      cublasLtMatmulPreference_t pref = nullptr;
      cublas_check(cublasLtMatmulPreferenceCreate(&pref), "preference create");
      const std::size_t ws = kWorkspaceBytes;
      const std::uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
      cublasLtMatmulHeuristicResult_t result{};
      int found = 0;
      cublasStatus_t st =
          cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction,
                                                  sizeof(reduction));
      if (st == CUBLAS_STATUS_SUCCESS)
        st = cublasLtMatmulAlgoGetHeuristic(impl_->handle, plan.op, plan.la, plan.lb, plan.lc, plan.lc, pref, 1,
                                            &result, &found);
#if INGINE_ACT_FP16
      if ((st != CUBLAS_STATUS_SUCCESS || found == 0)) {
        const std::uint32_t any_red = CUBLASLT_REDUCTION_SCHEME_MASK;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &any_red,
                                             sizeof(any_red));
        st = cublasLtMatmulAlgoGetHeuristic(impl_->handle, plan.op, plan.la, plan.lb, plan.lc, plan.lc, pref, 1,
                                            &result, &found);
      }
#endif
      cublasLtMatmulPreferenceDestroy(pref);
      cublas_check(st, "heuristic");
      if (found == 0) throw std::runtime_error("prefill Gemm: no cuBLASLt algorithm");
      plan.algo = result.algo;
      it = impl_->plans.emplace(key, plan).first;
    } catch (...) {
      cleanup();
      throw;
    }
  }
  const Impl::Plan& plan = it->second;
  const float alpha = 1.f, beta = 0.f;
  cublas_check(cublasLtMatmul(impl_->handle, plan.op, &alpha, B, plan.la, A, plan.lb, &beta, C, plan.lc, C, plan.lc,
                              &plan.algo, impl_->workspace, kWorkspaceBytes, stream),
               "matmul");
}

#endif

}  // namespace ingine::kernels::prefill

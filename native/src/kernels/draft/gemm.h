// BF16/FP16 GEMM for the drafter's linear matmuls (o_proj, down_proj, and context k/v projections):
// c [n, N] = act(a [n, K] . w [N, K]^T), FP32 accumulation. Plans are cached per shape; run() is graph-capturable.
#pragma once

#include "core/act.h"
#include "core/gpu_compat.h"

#include <cstddef>

namespace ingine::kernels::draft {

class Bf16Gemm {
 public:
  explicit Bf16Gemm(std::size_t workspace_bytes = std::size_t(32) << 20);
  ~Bf16Gemm();
  Bf16Gemm(const Bf16Gemm&) = delete;
  Bf16Gemm& operator=(const Bf16Gemm&) = delete;

  void run(const ActT* a, int n, const ActT* w, int N, int K, ActT* c, cudaStream_t stream);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace ingine::kernels::draft

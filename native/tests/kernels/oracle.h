// Shared host helpers of the kernel tests: delegates pure scalar oracles/conversions to
// common/scalar_oracle.h and keeps CUDA device helpers guarded when INGINE_BACKEND_CUDA == 0.
#pragma once

#include "common/scalar_oracle.h"
#include "common/test_util.h"

#include <optional>

namespace ingine::test {

inline double rbf(double x) { return round_bf16(float(x)); }  // to FP32, then ActT (nearest even)

inline std::vector<ingine::ActT> to_bf16(const std::vector<float>& v) {
  std::vector<ingine::ActT> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = float_to_bf16(v[i]);
  return o;
}

using Check = OracleCheck;

#if INGINE_BACKEND_CUDA
inline std::vector<float> bf16_download(const DeviceArray<ingine::ActT>& d) {
  const auto h = d.download();
  std::vector<float> o(h.size());
  for (size_t i = 0; i < h.size(); ++i) o[i] = bf16_to_float(h[i]);
  return o;
}
#endif

}  // namespace ingine::test

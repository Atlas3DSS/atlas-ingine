#include "core/ops.h"

#include "kernels/dense/dense.h"

#include <cmath>
#include <stdexcept>

namespace ingine::ops {

OpResolution resolve_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, bool allow_fallback) {
  validate_embed_q4_rmsnorm_desc(desc);
  const bool native_ok = desc.hidden_size == 2304 && desc.out_row_stride == 2304 &&
                         desc.output_rounding == default_act_rounding() &&
                         std::fabs(desc.eps - 1e-6f) == 0.f;
  if (native_ok) {
    return OpResolution{true, hal::BackendKind::Cuda, DispatchPath::Native, "cuda native dense::embed_q4"};
  }
  if (allow_fallback) {
    return OpResolution{true, hal::BackendKind::Cuda, DispatchPath::PortableFallback,
                        "cuda fallback for non-specialized shape, stride, eps, or rounding"};
  }
  return OpResolution{false, hal::BackendKind::Cuda, DispatchPath::PortableFallback,
                      "cuda native dense::embed_q4 requires H=2304, out_row_stride=2304, eps=1e-6, and build ActT rounding"};
}

DispatchPath run_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, const EmbedQ4RmsNormArgs& args,
                                  hal::StreamHandle stream, bool allow_fallback, bool force_fallback) {
  const OpResolution res = resolve_embed_q4_rmsnorm(desc, allow_fallback);
  if (!res.supported) throw std::invalid_argument(res.reason);
  if (!force_fallback && res.path == DispatchPath::Native) {
    kernels::dense::embed_q4(args.token_ids, desc.tokens, args.table_q4, args.norm_weight, args.out,
                             desc.allow_pdl, static_cast<cudaStream_t>(stream));
    return DispatchPath::Native;
  }
  if (!allow_fallback) throw std::invalid_argument("embed_q4_rmsnorm: portable fallback disabled");
  return run_embed_q4_rmsnorm_fallback(desc, args, stream);
}

}  // namespace ingine::ops

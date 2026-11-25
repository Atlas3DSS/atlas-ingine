#include "core/ops.h"

#include "kernels/dense/dense.h"

#include <cmath>
#include <stdexcept>
#include <vector>

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

void decode_q4_0(const void* table_q4, int rows, int hidden_size, float* out_f32, hal::StreamHandle stream) {
  if (!table_q4 || !out_f32) throw std::invalid_argument("decode_q4_0: null pointer");
  if (rows < 1) throw std::invalid_argument("decode_q4_0: rows must be >= 1");
  if (hidden_size <= 0 || (hidden_size % 32) != 0) {
    throw std::invalid_argument("decode_q4_0: hidden_size must be a positive multiple of 32");
  }
  kernels::dense::decode_q4_0(table_q4, rows, hidden_size, out_f32, static_cast<cudaStream_t>(stream));
}

DispatchPath run_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, const EmbedQ4RmsNormArgs& args,
                                  hal::StreamHandle stream, bool allow_fallback, bool force_fallback) {
  const OpResolution res = resolve_embed_q4_rmsnorm(desc, allow_fallback);
  if (!res.supported) throw std::invalid_argument(res.reason);
  if (!args.token_ids || !args.table_q4 || !args.norm_weight || !args.out) {
    throw std::invalid_argument("embed_q4_rmsnorm: null buffer argument");
  }
  if (!force_fallback && res.path == DispatchPath::Native) {
    if (desc.token_bounds == TokenBoundsMode::Validate) {
      if (args.host_token_ids != nullptr) {
        validate_token_ids(args.host_token_ids, desc.tokens, desc.vocab_size);
      } else if (hal::is_stream_capturing(stream)) {
        throw std::logic_error(
            "embed_q4_rmsnorm: TokenBoundsMode::Validate during graph capture requires host_token_ids or "
            "TokenBoundsMode::PreValidated");
      } else {
        std::vector<std::int64_t> host_tokens(static_cast<std::size_t>(desc.tokens));
        hal::memcpy_d2h_async(host_tokens.data(), args.token_ids, host_tokens.size() * sizeof(std::int64_t), stream);
        hal::stream_synchronize(stream);
        try {
          validate_token_ids(host_tokens.data(), desc.tokens, desc.vocab_size);
        } catch (const std::out_of_range&) {
          if (args.invalid_token_flag != nullptr) {
            const std::uint32_t one = 1u;
            hal::memcpy_h2d_async(args.invalid_token_flag, &one, sizeof(one), stream);
            hal::stream_synchronize(stream);
          }
          throw;
        }
      }
    }
    kernels::dense::embed_q4(args.token_ids, desc.tokens, desc.vocab_size, args.table_q4, args.norm_weight, args.out,
                             desc.allow_pdl, static_cast<cudaStream_t>(stream), args.invalid_token_flag);
    return DispatchPath::Native;
  }
  if (!allow_fallback) throw std::invalid_argument("embed_q4_rmsnorm: portable fallback disabled");
  return run_embed_q4_rmsnorm_fallback(desc, args, stream);
}

}  // namespace ingine::ops

// Pure native AMD ROCm/HIP operation dispatch boundary.
#include "core/ops.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::kernels::hip_dense {
void decode_q4_0_hip(const void* table_q4, int rows, int hidden_size, float* out_f32, hipStream_t stream);
void embed_q4_hip(const std::int64_t* token_ids, int tokens, int vocab_size, const void* table_q4,
                  const float* norm_weight, float* out, ops::ActRounding rounding, hipStream_t stream,
                  std::uint32_t* oob_flag);
}  // namespace ingine::kernels::hip_dense

namespace ingine::ops {
namespace {

void clear_flag_async(std::uint32_t* flag, hipStream_t hs) {
  const hipError_t err = hipMemsetAsync(flag, 0, sizeof(std::uint32_t), hs);
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("embed_q4_rmsnorm(HIP): hipMemsetAsync(invalid_token_flag) failed: ") +
                             hipGetErrorString(err));
  }
}

}  // namespace

OpResolution resolve_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, bool allow_fallback) {
  validate_embed_q4_rmsnorm_desc(desc);
  const bool native_ok = desc.hidden_size == 2304 && desc.out_row_stride == 2304 &&
                         std::fabs(desc.eps - 1e-6f) == 0.f;
  if (native_ok) {
    return OpResolution{true, hal::BackendKind::Hip, DispatchPath::Native, "hip native Q4_0 embed + RMSNorm"};
  }
  if (allow_fallback) {
    return OpResolution{true, hal::BackendKind::Hip, DispatchPath::PortableFallback,
                        "hip fallback for non-specialized shape, stride, or eps"};
  }
  return OpResolution{false, hal::BackendKind::Hip, DispatchPath::PortableFallback,
                      "hip native embed_q4 requires H=2304, out_row_stride=2304, and eps=1e-6"};
}

void decode_q4_0(const void* table_q4, int rows, int hidden_size, float* out_f32, hal::StreamHandle stream) {
  if (!table_q4 || !out_f32) throw std::invalid_argument("decode_q4_0(HIP): null pointer");
  if (rows < 1) throw std::invalid_argument("decode_q4_0(HIP): rows must be >= 1");
  if (hidden_size <= 0 || (hidden_size % 32) != 0) {
    throw std::invalid_argument("decode_q4_0(HIP): hidden_size must be a positive multiple of 32");
  }
  kernels::hip_dense::decode_q4_0_hip(table_q4, rows, hidden_size, out_f32, static_cast<hipStream_t>(stream.native));
}

DispatchPath run_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, const EmbedQ4RmsNormArgs& args,
                                  hal::StreamHandle stream, bool allow_fallback, bool force_fallback) {
  const OpResolution res = resolve_embed_q4_rmsnorm(desc, allow_fallback);
  if (!res.supported) throw std::invalid_argument(res.reason);
  if (!args.token_ids || !args.table_q4 || !args.norm_weight || !args.out) {
    throw std::invalid_argument("embed_q4_rmsnorm(HIP): null buffer argument");
  }
  if (!force_fallback && res.path == DispatchPath::Native) {
    const hipStream_t hs = static_cast<hipStream_t>(stream.native);
    switch (desc.token_bounds) {
      case TokenBoundsMode::Validate: {
        if (hal::is_stream_capturing(stream)) {
          throw std::logic_error(
              "embed_q4_rmsnorm(HIP): TokenBoundsMode::Validate cannot run inside graph capture; use "
              "TokenBoundsMode::DeferredFlag or TokenBoundsMode::PreValidated");
        }
        std::vector<std::int64_t> host_tokens(static_cast<std::size_t>(desc.tokens));
        hal::memcpy_d2h_async(host_tokens.data(), args.token_ids, host_tokens.size() * sizeof(std::int64_t), stream);
        hal::stream_synchronize(stream);
        try {
          validate_token_ids(host_tokens.data(), desc.tokens, desc.vocab_size);
          if (args.invalid_token_flag != nullptr) clear_flag_async(args.invalid_token_flag, hs);
        } catch (const std::out_of_range&) {
          if (args.invalid_token_flag != nullptr) {
            const std::uint32_t one = 1u;
            hal::memcpy_h2d_async(args.invalid_token_flag, &one, sizeof(one), stream);
            hal::stream_synchronize(stream);
          }
          throw;
        }
        break;
      }
      case TokenBoundsMode::DeferredFlag:
        if (args.invalid_token_flag == nullptr) {
          throw std::invalid_argument("embed_q4_rmsnorm(HIP): TokenBoundsMode::DeferredFlag requires invalid_token_flag");
        }
        clear_flag_async(args.invalid_token_flag, hs);
        break;
      case TokenBoundsMode::PreValidated:
        if (args.invalid_token_flag != nullptr) clear_flag_async(args.invalid_token_flag, hs);
        break;
    }
    kernels::hip_dense::embed_q4_hip(args.token_ids, desc.tokens, desc.vocab_size, args.table_q4, args.norm_weight,
                                     args.out, desc.output_rounding, hs, args.invalid_token_flag);
    return DispatchPath::Native;
  }
  if (!allow_fallback) throw std::invalid_argument("embed_q4_rmsnorm(HIP): portable fallback disabled");
  return run_embed_q4_rmsnorm_fallback(desc, args, stream);
}

}  // namespace ingine::ops

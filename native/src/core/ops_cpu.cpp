#include "core/ops.h"

#include "core/act.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace ingine::ops {
namespace {

float round_cpu(float x, ActRounding rounding) noexcept {
  switch (rounding) {
    case ActRounding::None: return x;
    case ActRounding::Bf16Rn: return act::host_bf16_bits_to_float(act::host_float_to_bf16_bits(x));
    case ActRounding::Fp16RnSat: return act::host_fp16_bits_to_float(act::host_float_to_fp16_bits(x));
  }
  return x;
}

}  // namespace

OpResolution resolve_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, bool /*allow_fallback*/) {
  validate_embed_q4_rmsnorm_desc(desc);
  return OpResolution{true, hal::BackendKind::Cpu, DispatchPath::Native, "cpu native Q4_0 embed + RMSNorm"};
}

DispatchPath run_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, const EmbedQ4RmsNormArgs& args,
                                  hal::StreamHandle stream, bool allow_fallback, bool force_fallback) {
  const OpResolution res = resolve_embed_q4_rmsnorm(desc, allow_fallback);
  if (!res.supported) throw std::invalid_argument(res.reason);
  if (force_fallback || res.path == DispatchPath::PortableFallback) {
    if (!allow_fallback) throw std::invalid_argument("embed_q4_rmsnorm: portable fallback disabled");
    return run_embed_q4_rmsnorm_fallback(desc, args, stream);
  }
  if (!args.token_ids || !args.table_q4 || !args.norm_weight || !args.out) {
    throw std::invalid_argument("embed_q4_rmsnorm: null buffer argument");
  }

  hal::stream_synchronize(stream);
  const int M = desc.tokens;
  const int H = desc.hidden_size;
  const auto* tbl = static_cast<const std::uint8_t*>(args.table_q4);
  std::vector<float> decoded(static_cast<std::size_t>(H));
  for (int m = 0; m < M; ++m) {
    const std::int64_t tok = args.token_ids[m];
    if (tok < 0 || tok >= desc.vocab_size) {
      throw std::out_of_range("embed_q4_rmsnorm: token id out of vocabulary range");
    }
    const std::uint8_t* row = tbl + static_cast<std::size_t>(tok) * desc.weight_row_bytes;
    decode_q4_0_row(row, H, decoded.data());
    float ss = 0.f;
    for (int j = 0; j < H; ++j) ss += decoded[j] * decoded[j];
    const float rinv = 1.0f / std::sqrt(ss / static_cast<float>(H) + desc.eps);
    float* out_row = args.out + static_cast<std::size_t>(m) * static_cast<std::size_t>(desc.out_row_stride);
    for (int j = 0; j < H; ++j) {
      out_row[j] = round_cpu(decoded[j] * rinv * args.norm_weight[j], desc.output_rounding);
    }
  }
  return DispatchPath::Native;
}

}  // namespace ingine::ops

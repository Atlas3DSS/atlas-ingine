#include "core/ops.h"

#include "core/cpu_kernels.h"
#include "core/cpu_pool.h"

#include <cstdint>
#include <stdexcept>

namespace ingine::ops {

OpResolution resolve_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, bool /*allow_fallback*/) {
  validate_embed_q4_rmsnorm_desc(desc);
  return OpResolution{true, hal::BackendKind::Cpu, DispatchPath::Native, "cpu native Q4_0 embed + RMSNorm (SIMD+pool)"};
}

void decode_q4_0(const void* table_q4, int rows, int hidden_size, float* out_f32, hal::StreamHandle stream) {
  if (!table_q4 || !out_f32) throw std::invalid_argument("decode_q4_0: null pointer");
  if (rows < 1) throw std::invalid_argument("decode_q4_0: rows must be >= 1");
  if (hidden_size <= 0 || (hidden_size % 32) != 0) {
    throw std::invalid_argument("decode_q4_0: hidden_size must be a positive multiple of 32");
  }
  hal::stream_synchronize(stream);
  cpu::decode_q4_0_rows(static_cast<const std::uint8_t*>(table_q4), rows, hidden_size, out_f32,
                        &cpu::CpuThreadPool::global());
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
  if (desc.token_bounds == TokenBoundsMode::DeferredFlag && args.invalid_token_flag == nullptr) {
    throw std::invalid_argument("embed_q4_rmsnorm: TokenBoundsMode::DeferredFlag requires invalid_token_flag");
  }

  hal::stream_synchronize(stream);
  if (args.invalid_token_flag != nullptr) *args.invalid_token_flag = 0u;
  try {
    validate_token_ids(args.token_ids, desc.tokens, desc.vocab_size);
  } catch (const std::out_of_range&) {
    if (args.invalid_token_flag != nullptr) *args.invalid_token_flag = 1u;
    if (desc.token_bounds == TokenBoundsMode::DeferredFlag) return DispatchPath::Native;
    throw;
  }

  cpu::embed_q4_rmsnorm_cpu(desc, args, &cpu::CpuThreadPool::global());
  return DispatchPath::Native;
}

}  // namespace ingine::ops

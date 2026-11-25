// Model-facing operation descriptors, layout/precision contracts, and backend dispatch boundary.
//
// Model code constructs descriptors owning the logical tensor representation (dtype, accumulator, packed
// layout, shape/strides, rounding mode, execution constraints); backends resolve descriptors against their
// capabilities and execute natively or via explicit portable fallback. 100% CUDA-free.
#pragma once

#include "core/dtype.h"
#include "core/hal.h"

#include <cstddef>
#include <cstdint>

namespace ingine::ops {

enum class PackedLayout : std::uint8_t {
  DenseRowMajor = 0,
  DenseColMajor = 1,
  Q4_0_SoA_1296 = 2,  // [hidden/2 nibble bytes | (hidden/32)*2 FP16 scale bytes], 1296 B for H=2304
};

enum class ActRounding : std::uint8_t {
  None = 0,
  Bf16Rn = 1,
  Fp16RnSat = 2,
};

enum class DispatchPath : std::uint8_t {
  Native = 0,
  PortableFallback = 1,
};

// Token-ID bounds enforcement contract for model-facing embedding ops:
//   * Every token ID must satisfy 0 <= id < desc.vocab_size.
//   * Validate (default): synchronously reads back and validates the ACTUAL args.token_ids buffer consumed
//     by execution before writing args.out. Rejects OOB IDs with std::out_of_range on every backend, and
//     throws std::logic_error if invoked during stream capture (where buffer readback is disallowed).
//   * DeferredFlag: capture-safe replay-time bounds check bound to the ACTUAL args.token_ids buffer on
//     every execution/replay. Requires non-null args.invalid_token_flag (throws std::invalid_argument if null),
//     clears it stream-ordered to 0 before every launch/replay, records 1 on OOB without fabricating zero rows,
//     and is enforced after synchronization via ops::check_deferred_token_bounds(flag, desc.vocab_size).
//     Caller contract: whenever *invalid_token_flag != 0, treat the entire output tensor as unusable
//     (CUDA native still computes valid neighboring rows, whereas CPU native/portable leave output unchanged).
//   * PreValidated: for model callers enforcing [0, vocab_size) at host admission before every launch/replay
//     (EngineGenerator, BatchGenerator, Runtime::set_inputs). Preserves zero-sync capture/fast path while
//     device kernels still guard OOB table reads and record invalid_token_flag (cleared stream-ordered).
enum class TokenBoundsMode : std::uint8_t {
  Validate = 0,
  DeferredFlag = 1,
  PreValidated = 2,
};

struct OpResolution {
  bool supported = false;
  hal::BackendKind backend = hal::BackendKind::Cpu;
  DispatchPath path = DispatchPath::Native;
  const char* reason = "";
};

struct EmbedQ4RmsNormDesc {
  int tokens = 0;
  int vocab_size = 0;
  int hidden_size = 2304;
  DType token_dtype = DType::I64;
  DType weight_storage_dtype = DType::U8;
  PackedLayout weight_layout = PackedLayout::Q4_0_SoA_1296;
  std::size_t weight_row_bytes = 1296;
  DType norm_weight_dtype = DType::FP32;
  DType output_dtype = DType::FP32;
  DType accumulator_dtype = DType::FP32;
  ActRounding output_rounding = ActRounding::Bf16Rn;
  std::int64_t out_row_stride = 2304;
  float eps = 1e-6f;
  bool allow_pdl = true;
  TokenBoundsMode token_bounds = TokenBoundsMode::Validate;
};

struct EmbedQ4RmsNormArgs {
  const std::int64_t* token_ids = nullptr;     // device [tokens]
  const void* table_q4 = nullptr;              // device u8 [vocab_size, weight_row_bytes]
  const float* norm_weight = nullptr;          // device fp32 [hidden_size]
  float* out = nullptr;                        // device fp32 [tokens, out_row_stride]
  std::uint32_t* invalid_token_flag = nullptr; // required for DeferredFlag; optional u32 [1] set to 1 on OOB
};

// Descriptor preserving prefill / drafter GEMM strides, output dtype, FP32 accumulation/split-K policy,
// and bounded workspace requirements across backends.
struct DenseGemmDesc {
  int M = 0;
  int N = 0;
  int K = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  DType input_dtype = DType::BF16;
  DType output_dtype = DType::BF16;
  DType accumulator_dtype = DType::FP32;
  bool fp32_splitk_reduction = true;
  std::size_t max_workspace_bytes = 32u << 20;
};

ActRounding default_act_rounding() noexcept;
EmbedQ4RmsNormDesc make_inkling_embed_q4_desc(int tokens, int vocab_size = 201024, bool allow_pdl = true) noexcept;

void validate_embed_q4_rmsnorm_desc(const EmbedQ4RmsNormDesc& desc);
void validate_dense_gemm_desc(const DenseGemmDesc& desc);
void validate_token_ids(const std::int64_t* host_token_ids, int tokens, int vocab_size);
void validate_token_ids_i32(const std::int32_t* host_token_ids, int tokens, int vocab_size);
void check_deferred_token_bounds(std::uint32_t invalid_token_flag, int vocab_size);

// Exact Q4_0 SoA row decoder: [hidden_size/2 nibble bytes | (hidden_size/32)*2 FP16 scale bytes] -> float[hidden_size].
// Canonical rule matching CUDA kernels:
//   d = isfinite(fp16_to_f32(scale)) ? fp16_to_f32(scale) : 0.0f
//   lane = std::fmaf(static_cast<float>(nibble), d, -8.0f * d)  (all zero products canonicalize to +0.0f / 0x00000000).
void decode_q4_0_row(const std::uint8_t* row_bytes, int hidden_size, float* out_f32);

// Backend-dispatched Q4_0 SoA decoder over [rows, row_bytes] -> float[rows * hidden_size] on `stream`.
void decode_q4_0(const void* table_q4, int rows, int hidden_size, float* out_f32, hal::StreamHandle stream);

// Shared backend-neutral fallback using HAL async copies + stream synchronization.
DispatchPath run_embed_q4_rmsnorm_fallback(const EmbedQ4RmsNormDesc& desc, const EmbedQ4RmsNormArgs& args,
                                           hal::StreamHandle stream);

// Backend-resolved operation entry points (implemented by ingine_ops_<backend>).
OpResolution resolve_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, bool allow_fallback = true);
DispatchPath run_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, const EmbedQ4RmsNormArgs& args,
                                  hal::StreamHandle stream, bool allow_fallback = true,
                                  bool force_fallback = false);

}  // namespace ingine::ops

#include "core/ops.h"

#include "core/act.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::ops {

ActRounding default_act_rounding() noexcept {
  return kActIsFP16 ? ActRounding::Fp16RnSat : ActRounding::Bf16Rn;
}

EmbedQ4RmsNormDesc make_inkling_embed_q4_desc(int tokens, int vocab_size, bool allow_pdl) noexcept {
  EmbedQ4RmsNormDesc desc;
  desc.tokens = tokens;
  desc.vocab_size = vocab_size;
  desc.hidden_size = 2304;
  desc.token_dtype = DType::I64;
  desc.weight_storage_dtype = DType::U8;
  desc.weight_layout = PackedLayout::Q4_0_SoA_1296;
  desc.weight_row_bytes = 1296;
  desc.norm_weight_dtype = DType::FP32;
  desc.output_dtype = DType::FP32;
  desc.accumulator_dtype = DType::FP32;
  desc.output_rounding = default_act_rounding();
  desc.out_row_stride = 2304;
  desc.eps = 1e-6f;
  desc.allow_pdl = allow_pdl;
  return desc;
}

void validate_embed_q4_rmsnorm_desc(const EmbedQ4RmsNormDesc& desc) {
  if (desc.tokens < 1) throw std::invalid_argument("embed_q4_rmsnorm: tokens must be >= 1");
  if (desc.vocab_size < 1) throw std::invalid_argument("embed_q4_rmsnorm: vocab_size must be >= 1");
  if (desc.hidden_size <= 0 || (desc.hidden_size % 32) != 0) {
    throw std::invalid_argument("embed_q4_rmsnorm: hidden_size must be positive and a multiple of 32");
  }
  if (desc.token_dtype != DType::I64) throw std::invalid_argument("embed_q4_rmsnorm: token_dtype must be I64");
  if (desc.weight_storage_dtype != DType::U8 || desc.weight_layout != PackedLayout::Q4_0_SoA_1296) {
    throw std::invalid_argument("embed_q4_rmsnorm: weight layout must be U8 Q4_0_SoA_1296");
  }
  const std::size_t expect_row_bytes =
      static_cast<std::size_t>(desc.hidden_size / 2) + static_cast<std::size_t>(desc.hidden_size / 32) * 2;
  if (desc.weight_row_bytes != expect_row_bytes) {
    throw std::invalid_argument("embed_q4_rmsnorm: unexpected weight_row_bytes " +
                                std::to_string(desc.weight_row_bytes));
  }
  if (desc.norm_weight_dtype != DType::FP32) {
    throw std::invalid_argument("embed_q4_rmsnorm: norm_weight_dtype must be FP32");
  }
  if (desc.output_dtype != DType::FP32 || desc.accumulator_dtype != DType::FP32) {
    throw std::invalid_argument("embed_q4_rmsnorm: output and accumulator dtypes must be FP32");
  }
  if (desc.out_row_stride < desc.hidden_size) {
    throw std::invalid_argument("embed_q4_rmsnorm: out_row_stride must be >= hidden_size");
  }
  if (!(desc.eps > 0.f) || !std::isfinite(desc.eps)) {
    throw std::invalid_argument("embed_q4_rmsnorm: eps must be finite and positive");
  }
}

void validate_dense_gemm_desc(const DenseGemmDesc& desc) {
  if (desc.M <= 0 || desc.N <= 0 || desc.K <= 0) {
    throw std::invalid_argument("dense_gemm: M, N, K must be positive");
  }
  if (desc.lda < desc.K || desc.ldb < desc.K || desc.ldc < desc.N) {
    throw std::invalid_argument("dense_gemm: invalid leading dimension stride");
  }
  if (desc.input_dtype != DType::BF16 && desc.input_dtype != DType::FP16) {
    throw std::invalid_argument("dense_gemm: input_dtype must be BF16 or FP16");
  }
  if (desc.output_dtype != desc.input_dtype && desc.output_dtype != DType::FP32) {
    throw std::invalid_argument("dense_gemm: output_dtype must match input_dtype or be FP32");
  }
  if (desc.accumulator_dtype != DType::FP32) {
    throw std::invalid_argument("dense_gemm: accumulator_dtype must be FP32");
  }
  if (desc.max_workspace_bytes == 0) {
    throw std::invalid_argument("dense_gemm: max_workspace_bytes must be positive");
  }
}

void validate_token_ids(const std::int64_t* host_token_ids, int tokens, int vocab_size) {
  if (vocab_size < 1) throw std::invalid_argument("token bounds: vocab_size must be >= 1");
  if (tokens < 0) throw std::invalid_argument("token bounds: tokens must be >= 0");
  if (tokens == 0) return;
  if (!host_token_ids) throw std::invalid_argument("token bounds: null host_token_ids");
  for (int m = 0; m < tokens; ++m) {
    const std::int64_t tok = host_token_ids[m];
    if (tok < 0 || tok >= static_cast<std::int64_t>(vocab_size)) {
      throw std::out_of_range("embed_q4_rmsnorm: token id " + std::to_string(tok) +
                              " out of vocabulary range [0, " + std::to_string(vocab_size) + ")");
    }
  }
}

void validate_token_ids_i32(const std::int32_t* host_token_ids, int tokens, int vocab_size) {
  if (vocab_size < 1) throw std::invalid_argument("token bounds: vocab_size must be >= 1");
  if (tokens < 0) throw std::invalid_argument("token bounds: tokens must be >= 0");
  if (tokens == 0) return;
  if (!host_token_ids) throw std::invalid_argument("token bounds: null host_token_ids");
  for (int m = 0; m < tokens; ++m) {
    const std::int32_t tok = host_token_ids[m];
    if (tok < 0 || tok >= vocab_size) {
      throw std::out_of_range("token id " + std::to_string(tok) + " out of vocabulary range [0, " +
                              std::to_string(vocab_size) + ")");
    }
  }
}

void check_deferred_token_bounds(std::uint32_t invalid_token_flag, int vocab_size) {
  if (invalid_token_flag != 0u) {
    throw std::out_of_range("embed_q4_rmsnorm: deferred token-bounds check failed for vocabulary range [0, " +
                            std::to_string(vocab_size) + ")");
  }
}

void decode_q4_0_row(const std::uint8_t* row_bytes, int hidden_size, float* out_f32) {
  if (!row_bytes || !out_f32) throw std::invalid_argument("decode_q4_0_row: null pointer");
  if (hidden_size <= 0 || (hidden_size % 32) != 0) {
    throw std::invalid_argument("decode_q4_0_row: hidden_size must be a positive multiple of 32");
  }
  const int nib_bytes = hidden_size / 2;
  const int blocks = hidden_size / 32;
  const std::uint8_t* scales = row_bytes + nib_bytes;
  for (int b = 0; b < blocks; ++b) {
    std::uint16_t d_bits = 0;
    std::memcpy(&d_bits, scales + std::size_t(b) * 2, 2);
    const float d_raw = act::host_fp16_bits_to_float(d_bits);
    const float d = std::isfinite(d_raw) ? d_raw : 0.f;
    const float m8d = -8.0f * d;
    const int base = b * 32;
    for (int j = 0; j < 32; ++j) {
      const std::uint8_t byte = row_bytes[(base + j) >> 1];
      const std::uint8_t nib = ((j & 1) != 0) ? (byte >> 4) : (byte & 0x0fu);
      out_f32[base + j] = std::fmaf(static_cast<float>(nib), d, m8d);
    }
  }
}

namespace {

float round_output(float x, ActRounding rounding) noexcept {
  switch (rounding) {
    case ActRounding::None: return x;
    case ActRounding::Bf16Rn: return act::host_bf16_bits_to_float(act::host_float_to_bf16_bits(x));
    case ActRounding::Fp16RnSat: return act::host_fp16_bits_to_float(act::host_float_to_fp16_bits(x));
  }
  return x;
}

}  // namespace

DispatchPath run_embed_q4_rmsnorm_fallback(const EmbedQ4RmsNormDesc& desc, const EmbedQ4RmsNormArgs& args,
                                           hal::StreamHandle stream) {
  validate_embed_q4_rmsnorm_desc(desc);
  if (!args.token_ids || !args.table_q4 || !args.norm_weight || !args.out) {
    throw std::invalid_argument("embed_q4_rmsnorm: null buffer argument");
  }
  if (hal::is_stream_capturing(stream)) {
    throw std::runtime_error("embed_q4_rmsnorm portable fallback cannot run inside graph capture");
  }
  if (desc.token_bounds == TokenBoundsMode::DeferredFlag && args.invalid_token_flag == nullptr) {
    throw std::invalid_argument("embed_q4_rmsnorm: TokenBoundsMode::DeferredFlag requires invalid_token_flag");
  }

  const int M = desc.tokens;
  const int H = desc.hidden_size;
  std::vector<std::int64_t> host_tokens(static_cast<std::size_t>(M));
  std::vector<float> host_norm(static_cast<std::size_t>(H));
  hal::memcpy_d2h_async(host_tokens.data(), args.token_ids, host_tokens.size() * sizeof(std::int64_t), stream);
  hal::memcpy_d2h_async(host_norm.data(), args.norm_weight, host_norm.size() * sizeof(float), stream);
  hal::stream_synchronize(stream);

  auto set_flag = [&](std::uint32_t value) {
    if (args.invalid_token_flag != nullptr) {
      hal::memcpy_h2d_async(args.invalid_token_flag, &value, sizeof(value), stream);
      hal::stream_synchronize(stream);
    }
  };

  try {
    validate_token_ids(host_tokens.data(), M, desc.vocab_size);
    set_flag(0u);
  } catch (const std::out_of_range&) {
    set_flag(1u);
    if (desc.token_bounds == TokenBoundsMode::DeferredFlag) return DispatchPath::PortableFallback;
    throw;
  }

  std::vector<std::uint8_t> host_rows(static_cast<std::size_t>(M) * desc.weight_row_bytes);
  const auto* tbl = static_cast<const std::uint8_t*>(args.table_q4);
  for (int m = 0; m < M; ++m) {
    const std::size_t tok = static_cast<std::size_t>(host_tokens[m]);
    hal::memcpy_d2h_async(host_rows.data() + static_cast<std::size_t>(m) * desc.weight_row_bytes,
                          tbl + tok * desc.weight_row_bytes, desc.weight_row_bytes, stream);
  }
  hal::stream_synchronize(stream);

  std::vector<float> decoded(static_cast<std::size_t>(H));
  std::vector<float> host_out(static_cast<std::size_t>(M) * static_cast<std::size_t>(H));
  for (int m = 0; m < M; ++m) {
    const std::uint8_t* row = host_rows.data() + static_cast<std::size_t>(m) * desc.weight_row_bytes;
    decode_q4_0_row(row, H, decoded.data());
    float ss = 0.f;
    for (int j = 0; j < H; ++j) ss += decoded[j] * decoded[j];
    const float rinv = 1.0f / std::sqrt(ss / static_cast<float>(H) + desc.eps);
    float* out_row = host_out.data() + static_cast<std::size_t>(m) * static_cast<std::size_t>(H);
    for (int j = 0; j < H; ++j) {
      out_row[j] = round_output(decoded[j] * rinv * host_norm[j], desc.output_rounding);
    }
  }

  if (desc.out_row_stride == H) {
    hal::memcpy_h2d_async(args.out, host_out.data(), host_out.size() * sizeof(float), stream);
  } else {
    for (int m = 0; m < M; ++m) {
      hal::memcpy_h2d_async(args.out + static_cast<std::size_t>(m) * static_cast<std::size_t>(desc.out_row_stride),
                            host_out.data() + static_cast<std::size_t>(m) * static_cast<std::size_t>(H),
                            static_cast<std::size_t>(H) * sizeof(float), stream);
    }
  }
  hal::stream_synchronize(stream);
  return DispatchPath::PortableFallback;
}

}  // namespace ingine::ops

// Pure native C++20 Intel SYCL Q4_0 SoA embedding + RMSNorm operation kernels and dispatch.
// Supports subgroup widths 8, 16, and 32 cleanly via sycl::reduce_over_group + SLM reduction.
#include "core/act.h"
#include "core/ops.h"
#include "core/vendor_policy.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ingine {
sycl::queue* sycl_queue_for_stream(hal::StreamHandle stream);
}

namespace ingine::ops {
namespace {

constexpr int kInklingHidden = 2304;
constexpr float kInklingEps = 1e-6f;

inline float sycl_fp16_bits_to_float(std::uint16_t h) noexcept {
  sycl::half hv = sycl::bit_cast<sycl::half>(h);
  return static_cast<float>(hv);
}

inline float sycl_round_act(float x, ActRounding rounding) noexcept {
  if (rounding == ActRounding::Bf16Rn) {
    const std::uint32_t w = sycl::bit_cast<std::uint32_t>(x);
    if ((w & 0x7fffffffu) > 0x7f800000u) {
      return sycl::bit_cast<float>((w & 0xffff0000u) | 0x00400000u);
    }
    const std::uint32_t r = ((w + 0x7fffu + ((w >> 16) & 1u)) >> 16) << 16;
    return sycl::bit_cast<float>(r);
  }
  if (rounding == ActRounding::Fp16RnSat) {
    const float clamped = sycl::isfinite(x) ? sycl::fmin(sycl::fmax(x, -act::kFp16Max), act::kFp16Max) : x;
    return static_cast<float>(static_cast<sycl::half>(clamped));
  }
  return x;
}

inline sycl::float4 decode_q4_0_x4_sycl(std::uint16_t b2, std::uint16_t scale_bits) noexcept {
  const float d_raw = sycl_fp16_bits_to_float(scale_bits);
  const float d = sycl::isfinite(d_raw) ? d_raw : 0.f;
  const float m8d = -8.f * d;
  return sycl::float4{sycl::fma(static_cast<float>(b2 & 0xfu), d, m8d),
                      sycl::fma(static_cast<float>((b2 >> 4) & 0xfu), d, m8d),
                      sycl::fma(static_cast<float>((b2 >> 8) & 0xfu), d, m8d),
                      sycl::fma(static_cast<float>((b2 >> 12) & 0xfu), d, m8d)};
}

}  // namespace

OpResolution resolve_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, bool allow_fallback) {
  validate_embed_q4_rmsnorm_desc(desc);
  const bool native_ok = desc.hidden_size == 2304 && desc.out_row_stride == 2304 &&
                         std::fabs(desc.eps - 1e-6f) == 0.f;
  if (native_ok) {
    return OpResolution{true, hal::BackendKind::Sycl, DispatchPath::Native, "sycl native Q4_0 embed + RMSNorm"};
  }
  if (allow_fallback) {
    return OpResolution{true, hal::BackendKind::Sycl, DispatchPath::PortableFallback,
                        "sycl fallback for non-specialized shape, stride, or eps"};
  }
  return OpResolution{false, hal::BackendKind::Sycl, DispatchPath::PortableFallback,
                      "sycl native embed_q4 requires H=2304, out_row_stride=2304, and eps=1e-6"};
}

void decode_q4_0(const void* table_q4, int rows, int hidden_size, float* out_f32, hal::StreamHandle stream) {
  if (!table_q4 || !out_f32) throw std::invalid_argument("decode_q4_0(SYCL): null pointer");
  if (rows < 1) throw std::invalid_argument("decode_q4_0(SYCL): rows must be >= 1");
  if (hidden_size <= 0 || (hidden_size % 32) != 0) {
    throw std::invalid_argument("decode_q4_0(SYCL): hidden_size must be a positive multiple of 32");
  }
  sycl::queue* q = sycl_queue_for_stream(stream);
  if (!q) throw std::invalid_argument("decode_q4_0(SYCL): null stream queue");
  const auto* tbl = static_cast<const std::uint8_t*>(table_q4);
  const int nv = hidden_size / 4;
  constexpr int wg_size = 256;

  q->submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(static_cast<std::size_t>(rows) * wg_size, wg_size),
                   [=](sycl::nd_item<1> item) {
                     const int row_idx = static_cast<int>(item.get_group(0));
                     const int t = static_cast<int>(item.get_local_id(0));
                     const std::size_t nib_bytes = static_cast<std::size_t>(hidden_size) / 2u;
                     const std::size_t row_bytes = nib_bytes + static_cast<std::size_t>(hidden_size / 32) * 2u;
                     const std::uint8_t* row = tbl + static_cast<std::size_t>(row_idx) * row_bytes;
                     const std::uint8_t* scales = row + nib_bytes;
                     float* out_row = out_f32 + static_cast<std::size_t>(row_idx) * static_cast<std::size_t>(hidden_size);
                     for (int idx = t; idx < nv; idx += wg_size) {
                       const std::uint16_t b2 = static_cast<std::uint16_t>(row[2 * idx]) |
                                                (static_cast<std::uint16_t>(row[2 * idx + 1]) << 8);
                       const std::uint16_t sb = static_cast<std::uint16_t>(scales[2 * (idx >> 3)]) |
                                                (static_cast<std::uint16_t>(scales[2 * (idx >> 3) + 1]) << 8);
                       const sycl::float4 v = decode_q4_0_x4_sycl(b2, sb);
                       out_row[4 * idx + 0] = v.x();
                       out_row[4 * idx + 1] = v.y();
                       out_row[4 * idx + 2] = v.z();
                       out_row[4 * idx + 3] = v.w();
                     }
                   });
  });
}

DispatchPath run_embed_q4_rmsnorm(const EmbedQ4RmsNormDesc& desc, const EmbedQ4RmsNormArgs& args,
                                  hal::StreamHandle stream, bool allow_fallback, bool force_fallback) {
  const OpResolution res = resolve_embed_q4_rmsnorm(desc, allow_fallback);
  if (!res.supported) throw std::invalid_argument(res.reason);
  if (!args.token_ids || !args.table_q4 || !args.norm_weight || !args.out) {
    throw std::invalid_argument("embed_q4_rmsnorm(SYCL): null buffer argument");
  }
  sycl::queue* q = sycl_queue_for_stream(stream);
  if (!force_fallback && res.path == DispatchPath::Native && q != nullptr) {
    if (desc.token_bounds == TokenBoundsMode::Validate) {
      std::vector<std::int64_t> host_tokens(static_cast<std::size_t>(desc.tokens));
      hal::memcpy_d2h_async(host_tokens.data(), args.token_ids, host_tokens.size() * sizeof(std::int64_t), stream);
      hal::stream_synchronize(stream);
      try {
        validate_token_ids(host_tokens.data(), desc.tokens, desc.vocab_size);
        if (args.invalid_token_flag != nullptr) hal::memset_async(args.invalid_token_flag, 0, sizeof(std::uint32_t), stream);
      } catch (const std::out_of_range&) {
        if (args.invalid_token_flag != nullptr) {
          const std::uint32_t one = 1u;
          hal::memcpy_h2d_async(args.invalid_token_flag, &one, sizeof(one), stream);
          hal::stream_synchronize(stream);
        }
        throw;
      }
    } else if (desc.token_bounds == TokenBoundsMode::DeferredFlag) {
      if (args.invalid_token_flag == nullptr) {
        throw std::invalid_argument("embed_q4_rmsnorm(SYCL): TokenBoundsMode::DeferredFlag requires invalid_token_flag");
      }
      q->memset(args.invalid_token_flag, 0, sizeof(std::uint32_t));
    } else if (args.invalid_token_flag != nullptr) {
      q->memset(args.invalid_token_flag, 0, sizeof(std::uint32_t));
    }

    const auto caps = hal::query_device_caps(0);
    const int wg_size = vendor::sycl::select_rmsnorm_workgroup(kInklingHidden, caps.subgroup.preferred_width);
    const std::int64_t* token_ids = args.token_ids;
    const auto* tbl = static_cast<const std::uint8_t*>(args.table_q4);
    const float* norm_w = args.norm_weight;
    float* out = args.out;
    std::uint32_t* oob_flag = args.invalid_token_flag;
    const int vocab_size = desc.vocab_size;
    const ActRounding rounding = desc.output_rounding;

    q->submit([&](sycl::handler& h) {
      sycl::local_accessor<float, 1> sg_part(64, h);
      sycl::local_accessor<float, 1> rinv_s(1, h);
      h.parallel_for(sycl::nd_range<1>(static_cast<std::size_t>(desc.tokens) * static_cast<std::size_t>(wg_size),
                                       static_cast<std::size_t>(wg_size)),
                     [=](sycl::nd_item<1> item) {
                       constexpr int NV = kInklingHidden / 4;
                       constexpr std::size_t NIB_BYTES = kInklingHidden / 2;
                       constexpr std::size_t ROW_BYTES = NIB_BYTES + (kInklingHidden / 32) * 2;
                       const int m = static_cast<int>(item.get_group(0));
                       const int t = static_cast<int>(item.get_local_id(0));

                       const std::int64_t tok = token_ids[m];
                       if (tok < 0 || tok >= vocab_size) {
                         if (t == 0 && oob_flag != nullptr) {
                           sycl::atomic_ref<std::uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                            sycl::access::address_space::global_space>
                               flag_ref(*oob_flag);
                           flag_ref.store(1u);
                         }
                         return;
                       }

                       const std::uint8_t* row = tbl + static_cast<std::size_t>(tok) * ROW_BYTES;
                       const std::uint8_t* scales = row + NIB_BYTES;
                       float ss = 0.f;
                       for (int idx = t; idx < NV; idx += wg_size) {
                         const std::uint16_t b2 = static_cast<std::uint16_t>(row[2 * idx]) |
                                                  (static_cast<std::uint16_t>(row[2 * idx + 1]) << 8);
                         const std::uint16_t sb = static_cast<std::uint16_t>(scales[2 * (idx >> 3)]) |
                                                  (static_cast<std::uint16_t>(scales[2 * (idx >> 3) + 1]) << 8);
                         const sycl::float4 v = decode_q4_0_x4_sycl(b2, sb);
                         ss += v.x() * v.x() + v.y() * v.y() + v.z() * v.z() + v.w() * v.w();
                       }

                       const auto sg = item.get_sub_group();
                       const float sg_sum = sycl::reduce_over_group(sg, ss, sycl::plus<float>{});
                       if (sg.leader()) sg_part[sg.get_group_linear_id()] = sg_sum;
                       item.barrier(sycl::access::fence_space::local_space);

                       if (t == 0) {
                         const int num_sg = static_cast<int>(sg.get_group_linear_range());
                         float total = 0.f;
                         for (int i = 0; i < num_sg; ++i) total += sg_part[i];
                         rinv_s[0] = sycl::rsqrt(total * (1.f / static_cast<float>(kInklingHidden)) + kInklingEps);
                       }
                       item.barrier(sycl::access::fence_space::local_space);

                       const float rinv = rinv_s[0];
                       float* o = out + static_cast<std::size_t>(m) * static_cast<std::size_t>(kInklingHidden);
                       for (int idx = t; idx < NV; idx += wg_size) {
                         const std::uint16_t b2 = static_cast<std::uint16_t>(row[2 * idx]) |
                                                  (static_cast<std::uint16_t>(row[2 * idx + 1]) << 8);
                         const std::uint16_t sb = static_cast<std::uint16_t>(scales[2 * (idx >> 3)]) |
                                                  (static_cast<std::uint16_t>(scales[2 * (idx >> 3) + 1]) << 8);
                         const sycl::float4 v = decode_q4_0_x4_sycl(b2, sb);
                         o[4 * idx + 0] = sycl_round_act(v.x() * rinv * norm_w[4 * idx + 0], rounding);
                         o[4 * idx + 1] = sycl_round_act(v.y() * rinv * norm_w[4 * idx + 1], rounding);
                         o[4 * idx + 2] = sycl_round_act(v.z() * rinv * norm_w[4 * idx + 2], rounding);
                         o[4 * idx + 3] = sycl_round_act(v.w() * rinv * norm_w[4 * idx + 3], rounding);
                       }
                     });
    });
    return DispatchPath::Native;
  }

  if (!allow_fallback) throw std::invalid_argument("embed_q4_rmsnorm(SYCL): portable fallback disabled");
  return run_embed_q4_rmsnorm_fallback(desc, args, stream);
}

}  // namespace ingine::ops

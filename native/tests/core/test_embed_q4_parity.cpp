// HAL + Ops First Parity Case: Inkling Q4_0 embedding lookup (1,296 B/row) + RMSNorm through core/hal.h,
// core/arena.h, and core/ops.h on both CPU and CUDA with zero direct CUDA headers in the test translation unit.
#include "core/act.h"
#include "core/arena.h"
#include "core/backend_contract.h"
#include "core/hal.h"
#include "core/ops.h"
#include "common/scalar_oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ingine;
using namespace ingine::test;

namespace {

constexpr int kInklingHidden = 2304;
constexpr int kInklingRowBytes = 1296;
constexpr int kBlocksPerRow = kInklingHidden / 32;  // 72

std::uint32_t float_bits(float v) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

void write_q4_block(std::uint8_t* row, int blk, std::uint16_t scale_bits, const std::uint8_t* nibbles32) {
  for (int j = 0; j < 32; ++j) {
    const std::uint8_t q = nibbles32[j] & 0x0fu;
    std::uint8_t& byte = row[blk * 16 + (j >> 1)];
    if ((j & 1) == 0) byte = q;
    else byte |= static_cast<std::uint8_t>(q << 4);
  }
  std::memcpy(row + (kInklingHidden / 2) + blk * 2, &scale_bits, 2);
}

bool check_decode_q4_bit_exact(hal::Stream& stream) {
  constexpr int V = 16;
  Rng rng(20261124ull);
  bool ok = true;

  std::vector<std::uint8_t> packed_table(std::size_t(V) * kInklingRowBytes, 0);
  // Row 0: fixed signed-scale & zero fixture covering negative/positive scales, nibble 8, +-0.0, subnormals, Inf/NaN.
  {
    std::uint8_t* row0 = packed_table.data();
    std::uint8_t all8[32]{};
    std::fill_n(all8, 32, 8);
    std::uint8_t ramp[32]{};
    for (int j = 0; j < 32; ++j) ramp[j] = static_cast<std::uint8_t>(j & 0x0f);

    write_q4_block(row0, 0, 0xbc00u, all8);  // scale = -1.0f, nibble = 8 -> +0.0f (0x00000000)
    write_q4_block(row0, 1, 0x3c00u, all8);  // scale = +1.0f, nibble = 8 -> +0.0f (0x00000000)
    write_q4_block(row0, 2, 0xbc00u, ramp);  // scale = -1.0f, nibbles 0..15 -> +8.0f .. -7.0f
    write_q4_block(row0, 3, 0x8000u, ramp);  // scale = -0.0f, nibbles 0..15 -> +0.0f (0x00000000)
    write_q4_block(row0, 4, 0x0000u, ramp);  // scale = +0.0f, nibbles 0..15 -> +0.0f (0x00000000)
    write_q4_block(row0, 5, 0x8001u, ramp);  // scale = subnormal -2^-24, nibbles 0..15
    write_q4_block(row0, 6, 0x7c00u, ramp);  // scale = +Inf -> guarded to 0
    write_q4_block(row0, 7, 0xfc00u, ramp);  // scale = -Inf -> guarded to 0
    write_q4_block(row0, 8, 0x7e00u, ramp);  // scale = NaN  -> guarded to 0
    for (int b = 9; b < kBlocksPerRow; ++b) {
      write_q4_block(row0, b, (b & 1) ? 0x3e00u : 0xbc00u, (b & 1) ? ramp : all8);
    }
  }
  // Rows 1..V-1: randomized Q4_0 rows.
  for (int row = 1; row < V; ++row) {
    const std::vector<float> src = randn(rng, kInklingHidden, 1.5);
    scalar_q4_0_pack_row(src.data(), 1, kInklingHidden, packed_table.data() + std::size_t(row) * kInklingRowBytes);
  }

  // Run backend-dispatched ops::decode_q4_0 on device/host memory and read back actual emitted bits.
  DeviceBuffer d_table(packed_table.size());
  DeviceBuffer d_decoded(std::size_t(V) * kInklingHidden * sizeof(float));
  hal::memcpy_h2d_async(d_table.p, packed_table.data(), packed_table.size(), stream.handle());
  ops::decode_q4_0(d_table.p, V, kInklingHidden, static_cast<float*>(d_decoded.p), stream.handle());

  std::vector<float> dev_got(std::size_t(V) * kInklingHidden, 0.f);
  hal::memcpy_d2h_async(dev_got.data(), d_decoded.p, dev_got.size() * sizeof(float), stream.handle());
  hal::Event done;
  done.record(stream.handle());
  done.synchronize();

  // Explicit bit assertions on the signed-scale & zero fixture (row 0).
  for (int j = 0; j < 32; ++j) {
    ok &= (float_bits(dev_got[0 * 32 + j]) == 0x00000000u);  // scale -1.0, nibble 8
    ok &= (float_bits(dev_got[1 * 32 + j]) == 0x00000000u);  // scale +1.0, nibble 8
    ok &= (float_bits(dev_got[3 * 32 + j]) == 0x00000000u);  // scale -0.0, all nibbles
    ok &= (float_bits(dev_got[4 * 32 + j]) == 0x00000000u);  // scale +0.0, all nibbles
    ok &= (float_bits(dev_got[6 * 32 + j]) == 0x00000000u);  // scale +Inf
    ok &= (float_bits(dev_got[7 * 32 + j]) == 0x00000000u);  // scale -Inf
    ok &= (float_bits(dev_got[8 * 32 + j]) == 0x00000000u);  // scale NaN
  }
  ok &= (float_bits(dev_got[2 * 32 + 0]) == 0x41000000u);   // (0 - 8) * -1 = +8.0f
  ok &= (float_bits(dev_got[2 * 32 + 8]) == 0x00000000u);   // (8 - 8) * -1 = +0.0f
  ok &= (float_bits(dev_got[2 * 32 + 15]) == 0xc0e00000u);  // (15 - 8) * -1 = -7.0f
  ok &= (float_bits(dev_got[5 * 32 + 8]) == 0x00000000u);   // subnormal scale, nibble 8

  for (int row = 0; row < V && ok; ++row) {
    const std::uint8_t* row_ptr = packed_table.data() + std::size_t(row) * kInklingRowBytes;
    std::vector<float> ref(kInklingHidden), host_got(kInklingHidden);
    scalar_q4_0_unpack_row(row_ptr, 1, kInklingHidden, ref.data());
    ops::decode_q4_0_row(row_ptr, kInklingHidden, host_got.data());
    for (int j = 0; j < kInklingHidden; ++j) {
      const std::uint32_t rb = float_bits(ref[j]);
      const std::uint32_t hb = float_bits(host_got[j]);
      const std::uint32_t db = float_bits(dev_got[std::size_t(row) * kInklingHidden + j]);
      if (rb != hb || rb != db) {
        std::fprintf(stderr, "decode_q4_0 mismatch row=%d col=%d scalar=0x%08x host=0x%08x backend=0x%08x\n",
                     row, j, rb, hb, db);
        ok = false;
        break;
      }
    }
  }

  // Verify end-to-end embed_q4_rmsnorm canonical zero bits across Native and PortableFallback.
  {
    std::vector<std::uint8_t> zero_tbl(2 * kInklingRowBytes, 0);
    std::uint8_t all8[32]{};
    std::fill_n(all8, 32, 8);
    std::uint8_t mix_nib[32]{};
    for (int j = 0; j < 32; ++j) mix_nib[j] = static_cast<std::uint8_t>(j & 0x0f);
    for (int b = 0; b < kBlocksPerRow; ++b) {
      write_q4_block(zero_tbl.data(), b, 0xbc00u, all8);                    // row 0: scale -1.0, nibble 8
      write_q4_block(zero_tbl.data() + kInklingRowBytes, b, 0x8000u, mix_nib);  // row 1: scale -0.0, mixed nibbles
    }
    const std::vector<std::int64_t> ids = {0, 1};
    const std::vector<float> norm_pos(kInklingHidden, 1.0f);
    const std::vector<float> norm_neg(kInklingHidden, -1.0f);
    DeviceBuffer d_z_tbl(zero_tbl.size()), d_z_tok(ids.size() * sizeof(std::int64_t)),
        d_z_norm(kInklingHidden * sizeof(float)), d_z_out(ids.size() * kInklingHidden * sizeof(float));
    d_z_tbl.copy_from_host(zero_tbl.data(), zero_tbl.size());
    d_z_tok.copy_from_host(ids.data(), ids.size() * sizeof(std::int64_t));
    const auto desc = ops::make_inkling_embed_q4_desc(2, 2, /*allow_pdl=*/true);

    for (bool fallback : {false, true}) {
      for (const bool neg : {false, true}) {
        const auto& nw = neg ? norm_neg : norm_pos;
        d_z_norm.copy_from_host(nw.data(), nw.size() * sizeof(float));
        const ops::EmbedQ4RmsNormArgs args{
            static_cast<const std::int64_t*>(d_z_tok.p), d_z_tbl.p,
            static_cast<const float*>(d_z_norm.p), static_cast<float*>(d_z_out.p), nullptr};
        const ops::DispatchPath path =
            ops::run_embed_q4_rmsnorm(desc, args, stream.handle(), /*allow_fallback=*/true, fallback);
        std::vector<float> got(ids.size() * kInklingHidden, 0.f);
        hal::memcpy_d2h_async(got.data(), d_z_out.p, got.size() * sizeof(float), stream.handle());
        hal::stream_synchronize(stream.handle());
        const std::uint32_t expected_bits = neg ? 0x80000000u : 0x00000000u;
        ok &= (path == (fallback ? ops::DispatchPath::PortableFallback : ops::DispatchPath::Native));
        for (float v : got) {
          if (float_bits(v) != expected_bits) {
            std::fprintf(stderr, "embed_q4 canonical zero mismatch fallback=%d neg=%d got=0x%08x expect=0x%08x\n",
                         int(fallback), int(neg), float_bits(v), expected_bits);
            ok = false;
            break;
          }
        }
      }
    }
  }

  std::printf("%s decode_q4_0 bit-exact (%s emitted bits + signed scale/zero fixture, 1,296 B/row)\n",
              ok ? "PASS" : "FAIL", hal::backend_name(hal::active_backend()));
  return ok;
}

bool check_caps_and_graph_policy(const hal::DeviceCaps& caps) {
  bool ok = true;
  const hal::BackendKind expected_backend = ingine::test_contract::expected_compiled_backend();
  ok &= (hal::active_backend() == expected_backend);
  ok &= ingine::test_contract::check_backend_caps_contract(hal::active_backend(), caps);
  ok &= (!hal::conditional_graphs_enabled(-1) && !hal::conditional_graphs_enabled(999));
  if (hal::active_backend() != hal::BackendKind::Cuda) {
    ok &= (!caps.supports_conditional_graph && !hal::conditional_graphs_enabled(0));
  }
  ok &= (!hal::is_stream_capturing({}));

  // Env-off toggle check (restores original env state on exit).
  {
    const char* prev_env = std::getenv("INGINE_DISABLE_CONDITIONAL_GRAPHS");
    const std::optional<std::string> saved = prev_env ? std::optional<std::string>(prev_env) : std::nullopt;
    setenv("INGINE_DISABLE_CONDITIONAL_GRAPHS", "1", 1);
    ok &= !hal::conditional_graphs_enabled(0);
    ok &= !hal::query_device_caps(0).supports_conditional_graph;
    if (saved) setenv("INGINE_DISABLE_CONDITIONAL_GRAPHS", saved->c_str(), 1);
    else unsetenv("INGINE_DISABLE_CONDITIONAL_GRAPHS");
    ok &= (hal::query_device_caps(0).supports_conditional_graph == hal::conditional_graphs_enabled(0));
  }

  auto expect_throws = [](auto fn) {
    try {
      fn();
      return false;
    } catch (const std::runtime_error&) {
      return true;
    }
  };

  // Matrix of ordinary capture vs conditional SWITCH capability admission.
  for (bool cap : {false, true}) {
    for (bool cond : {false, true}) {
      hal::DeviceCaps c = caps;
      c.supports_graph_capture = cap;
      c.supports_conditional_graph = cond;

      ok &= (hal::graph_requirement_met(c, hal::GraphRequirement::Capture) == cap);
      ok &= (hal::graph_requirement_met(c, hal::GraphRequirement::ConditionalSwitch) == (cap && cond));

      if (cap) {
        hal::require_graph_capture_or_eager(c, /*allow_eager=*/false, "capture-caller",
                                            hal::GraphRequirement::Capture);
      } else {
        ok &= expect_throws([&] {
          hal::require_graph_capture_or_eager(c, /*allow_eager=*/false, "capture-caller",
                                              hal::GraphRequirement::Capture);
        });
      }
      hal::require_graph_capture_or_eager(c, /*allow_eager=*/true, "capture-caller",
                                          hal::GraphRequirement::Capture);

      if (cap && cond) {
        hal::require_conditional_switch_or_eager(c, /*allow_eager=*/false, "switch-caller");
      } else {
        ok &= expect_throws([&] {
          hal::require_conditional_switch_or_eager(c, /*allow_eager=*/false, "switch-caller");
        });
        ok &= expect_throws([&] {
          hal::require_graph_capture_or_eager(c, /*allow_eager=*/false, "switch-caller",
                                              hal::GraphRequirement::ConditionalSwitch);
        });
      }
      hal::require_conditional_switch_or_eager(c, /*allow_eager=*/true, "switch-caller");
    }
  }

  ops::DenseGemmDesc valid_gemm;
  valid_gemm.M = 8;
  valid_gemm.N = kInklingHidden;
  valid_gemm.K = kInklingHidden;
  valid_gemm.lda = kInklingHidden;
  valid_gemm.ldb = kInklingHidden;
  valid_gemm.ldc = kInklingHidden;
  ops::validate_dense_gemm_desc(valid_gemm);

  ops::DenseGemmDesc bad_gemm = valid_gemm;
  bad_gemm.ldc = 128;
  bool bad_gemm_rejected = false;
  try {
    ops::validate_dense_gemm_desc(bad_gemm);
  } catch (const std::invalid_argument&) {
    bad_gemm_rejected = true;
  }
  ok &= bad_gemm_rejected;
  std::printf("%s HAL capabilities (%s, subgroup=%d, switch=%d), mixed capture/switch admission, and DenseGemmDesc\n",
              ok ? "PASS" : "FAIL", hal::backend_name(caps.backend), caps.subgroup.preferred_width,
              int(caps.supports_conditional_graph));
  return ok;
}

bool check_embed_q4_rmsnorm_case(int M, const std::vector<std::int64_t>& tok_ids,
                                 const std::vector<std::uint8_t>& table, const std::vector<float>& norm_f32,
                                 int vocab_size, bool force_fallback, int out_stride, hal::Stream& stream,
                                 DeviceArena& arena,
                                 std::optional<ops::ActRounding> rounding_override = std::nullopt) {
  constexpr double kEps = 1e-6;
  ops::EmbedQ4RmsNormDesc desc = ops::make_inkling_embed_q4_desc(M, vocab_size, /*allow_pdl=*/true);
  desc.out_row_stride = out_stride;
  if (rounding_override) desc.output_rounding = *rounding_override;

  const bool fp16_round = (desc.output_rounding == ops::ActRounding::Fp16RnSat);
  std::vector<float> ref_decoded;
  std::vector<double> ref_unrounded, ref_rounded;
  scalar_embed_q4_rmsnorm_fp64(tok_ids.data(), M, table.data(), vocab_size, norm_f32.data(), kInklingHidden, kEps,
                               fp16_round, &ref_decoded, &ref_unrounded, &ref_rounded);

  const std::size_t tok_bytes = std::size_t(M) * sizeof(std::int64_t);
  const std::size_t out_elems = std::size_t(M) * std::size_t(out_stride);
  const std::size_t out_bytes = out_elems * sizeof(float);

  arena.reset();
  DeviceBuffer d_table(table.size());
  DeviceBuffer d_norm(norm_f32.size() * sizeof(float));
  const Tensor t_tok = arena.alloc(DType::I64, {M});
  const Tensor t_out = arena.alloc(DType::FP32, {M, out_stride});
  void* d_tok = t_tok.data;
  void* d_out = t_out.data;

  hal::memcpy_h2d_async(d_table.p, table.data(), table.size(), stream.handle());
  hal::memcpy_h2d_async(d_norm.p, norm_f32.data(), norm_f32.size() * sizeof(float), stream.handle());
  hal::memcpy_h2d_async(d_tok, tok_ids.data(), tok_bytes, stream.handle());

  constexpr float kSentinel = 42.5f;
  std::vector<float> init_out(out_elems, kSentinel);
  hal::memcpy_h2d_async(d_out, init_out.data(), out_bytes, stream.handle());

  ops::EmbedQ4RmsNormArgs args{
      static_cast<const std::int64_t*>(d_tok),
      d_table.p,
      static_cast<const float*>(d_norm.p),
      static_cast<float*>(d_out),
      nullptr,
  };

  auto run_once = [&](std::vector<float>& host_out, ops::DispatchPath& path_out) {
    path_out = ops::run_embed_q4_rmsnorm(desc, args, stream.handle(), /*allow_fallback=*/true, force_fallback);
    host_out.assign(out_elems, 0.f);
    hal::memcpy_d2h_async(host_out.data(), d_out, out_bytes, stream.handle());
    hal::Event done;
    done.record(stream.handle());
    done.synchronize();
  };

  ops::DispatchPath path1{}, path2{};
  std::vector<float> out1, out2;
  run_once(out1, path1);
  run_once(out2, path2);

  const bool expect_fallback = ingine::test_contract::expect_embed_q4_fallback(
      hal::active_backend(), force_fallback, desc.hidden_size, out_stride, desc.output_rounding, desc.eps);
  const ops::OpResolution resolved = ops::resolve_embed_q4_rmsnorm(desc, /*allow_fallback=*/true);
  const ops::DispatchPath resolved_effective =
      force_fallback ? ops::DispatchPath::PortableFallback : resolved.path;
  const ops::DispatchPath expected_path =
      expect_fallback ? ops::DispatchPath::PortableFallback : ops::DispatchPath::Native;
  bool ok = (resolved.supported && resolved.backend == hal::active_backend() &&
             resolved_effective == expected_path &&
             path1 == expected_path && path2 == expected_path && out1.size() == out2.size());
  for (std::size_t i = 0; i < out1.size() && ok; ++i) {
    if (std::memcmp(&out1[i], &out2[i], sizeof(float)) != 0) ok = false;
  }

  std::vector<float> got_packed(std::size_t(M) * kInklingHidden);
  bool sentinel_ok = true;
  bool rounding_ok = true;
  for (int m = 0; m < M; ++m) {
    for (int j = 0; j < kInklingHidden; ++j) {
      const float v = out1[std::size_t(m) * out_stride + j];
      if (!std::isfinite(v)) ok = false;
      const float r = round_act_scalar(v, fp16_round);
      if (float_bits(v) != float_bits(r)) {
        if (rounding_ok) {
          std::fprintf(stderr, "activation rounding mismatch m=%d j=%d got=0x%08x (%g) rounded=0x%08x (%g)\n",
                       m, j, float_bits(v), v, float_bits(r), r);
        }
        rounding_ok = false;
      }
      got_packed[std::size_t(m) * kInklingHidden + j] = v;
    }
    for (int j = kInklingHidden; j < out_stride; ++j) {
      if (out1[std::size_t(m) * out_stride + j] != kSentinel) sentinel_ok = false;
    }
  }
  ok &= sentinel_ok && rounding_ok;

  const char* label = expect_fallback ? "PortableFallback" : "Native";
  const char* round_tag = fp16_round ? "FP16" : "BF16";
  char msg[160];
  std::snprintf(msg, sizeof(msg), "embed_q4_rmsnorm[%s %s M=%d stride=%d]", label, round_tag, M, out_stride);
  const OracleCheck chk{msg, 1.0 / 128.0, 1e-6, 0.01};
  ok &= chk(got_packed, ref_unrounded, &ref_rounded);
  return ok;
}

bool check_token_bounds_contract(int vocab_size, const std::vector<std::uint8_t>& table,
                                 const std::vector<float>& norm_f32, hal::Stream& stream) {
  bool ok = true;
  constexpr float kSentinel = -99.25f;
  const auto base_desc = ops::make_inkling_embed_q4_desc(2, vocab_size, /*allow_pdl=*/true);
  const std::vector<float> init_out(2 * kInklingHidden, kSentinel);

  DeviceBuffer d_tok(2 * sizeof(std::int64_t)), d_tbl(table.size()), d_norm(norm_f32.size() * sizeof(float)),
      d_out(2 * kInklingHidden * sizeof(float)), d_oob(sizeof(std::uint32_t));
  d_tbl.copy_from_host(table.data(), table.size());
  d_norm.copy_from_host(norm_f32.data(), norm_f32.size() * sizeof(float));

  auto check_out_untouched = [&]() {
    std::vector<float> got(2 * kInklingHidden, 0.f);
    d_out.copy_to_host(got.data(), got.size() * sizeof(float));
    for (float v : got) {
      if (v != kSentinel) return false;
    }
    return true;
  };
  auto check_invalid_rows_untouched = [&](const std::vector<std::int64_t>& ids) {
    std::vector<float> got(2 * kInklingHidden, 0.f);
    d_out.copy_to_host(got.data(), got.size() * sizeof(float));
    for (int m = 0; m < int(ids.size()); ++m) {
      if (ids[std::size_t(m)] < 0 || ids[std::size_t(m)] >= vocab_size) {
        for (int j = 0; j < kInklingHidden; ++j) {
          if (got[std::size_t(m) * kInklingHidden + j] != kSentinel) return false;
        }
      }
    }
    return true;
  };

  const std::vector<std::vector<std::int64_t>> bad_cases = {
      {0, -1},
      {0, vocab_size},
      {vocab_size + 10, 1},
  };

  for (const auto& bad_ids : bad_cases) {
    // 1. Validate synchronously inspects the actual args.token_ids buffer (both Native and PortableFallback),
    // with and without invalid_token_flag.
    for (bool fallback : {false, true}) {
      d_tok.copy_from_host(bad_ids.data(), bad_ids.size() * sizeof(std::int64_t));
      d_out.copy_from_host(init_out.data(), init_out.size() * sizeof(float));
      const std::uint32_t zero_flag = 0u;
      d_oob.copy_from_host(&zero_flag, sizeof(zero_flag));
      ops::EmbedQ4RmsNormArgs args{
          static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
          static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p),
          static_cast<std::uint32_t*>(d_oob.p)};
      bool threw = false;
      try {
        (void)ops::run_embed_q4_rmsnorm(base_desc, args, stream.handle(), /*allow_fallback=*/true, fallback);
      } catch (const std::out_of_range&) {
        threw = true;
      }
      std::uint32_t flag_val = 0u;
      d_oob.copy_to_host(&flag_val, sizeof(flag_val));
      const bool out_ok1 = check_out_untouched();
      if (!threw || flag_val != 1u || !out_ok1) {
        std::fprintf(stderr, "token-bounds Validate(with flag) failed fallback=%d ids=[%lld,%lld] threw=%d flag=%u out_ok=%d\n",
                     int(fallback), static_cast<long long>(bad_ids[0]), static_cast<long long>(bad_ids[1]),
                     int(threw), flag_val, int(out_ok1));
      }
      ok &= threw && (flag_val == 1u) && out_ok1;

      ops::EmbedQ4RmsNormArgs args_no_flag{
          static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
          static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p), nullptr};
      threw = false;
      try {
        (void)ops::run_embed_q4_rmsnorm(base_desc, args_no_flag, stream.handle(), /*allow_fallback=*/true, fallback);
      } catch (const std::out_of_range&) {
        threw = true;
      }
      const bool out_ok2 = check_out_untouched();
      if (!threw || !out_ok2) {
        std::fprintf(stderr, "token-bounds Validate(no flag) failed fallback=%d ids=[%lld,%lld] threw=%d out_ok=%d\n",
                     int(fallback), static_cast<long long>(bad_ids[0]), static_cast<long long>(bad_ids[1]),
                     int(threw), int(out_ok2));
      }
      ok &= threw && out_ok2;
    }

    // 2. DeferredFlag requires non-null flag, clears flag on valid, sets flag=1 on OOB without touching invalid rows,
    // and enforces via check_deferred_token_bounds.
    {
      auto def_desc = base_desc;
      def_desc.token_bounds = ops::TokenBoundsMode::DeferredFlag;
      bool null_rejected = false;
      try {
        ops::EmbedQ4RmsNormArgs null_args{
            static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
            static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p), nullptr};
        (void)ops::run_embed_q4_rmsnorm(def_desc, null_args, stream.handle(), /*allow_fallback=*/true, false);
      } catch (const std::invalid_argument&) {
        null_rejected = true;
      }
      if (!null_rejected) std::fprintf(stderr, "token-bounds DeferredFlag null flag check failed\n");
      ok &= null_rejected;

      for (bool fallback : {false, true}) {
        d_tok.copy_from_host(bad_ids.data(), bad_ids.size() * sizeof(std::int64_t));
        d_out.copy_from_host(init_out.data(), init_out.size() * sizeof(float));
        const std::uint32_t zero_flag = 0u;
        d_oob.copy_from_host(&zero_flag, sizeof(zero_flag));
        ops::EmbedQ4RmsNormArgs args{
            static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
            static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p),
            static_cast<std::uint32_t*>(d_oob.p)};
        (void)ops::run_embed_q4_rmsnorm(def_desc, args, stream.handle(), /*allow_fallback=*/true, fallback);
        hal::stream_synchronize(stream.handle());
        std::uint32_t flag_val = 0u;
        d_oob.copy_to_host(&flag_val, sizeof(flag_val));
        bool deferred_threw = false;
        try {
          ops::check_deferred_token_bounds(flag_val, vocab_size);
        } catch (const std::out_of_range&) {
          deferred_threw = true;
        }
        const bool rows_ok = (fallback || hal::active_backend() == hal::BackendKind::Cpu)
                                 ? check_out_untouched()
                                 : check_invalid_rows_untouched(bad_ids);
        if (flag_val != 1u || !deferred_threw || !rows_ok) {
          std::fprintf(stderr, "token-bounds DeferredFlag OOB failed fallback=%d ids=[%lld,%lld] flag=%u threw=%d rows_ok=%d\n",
                       int(fallback), static_cast<long long>(bad_ids[0]), static_cast<long long>(bad_ids[1]),
                       flag_val, int(deferred_threw), int(rows_ok));
        }
        ok &= (flag_val == 1u) && deferred_threw && rows_ok;

        // Subsequent valid call on the same flag must reset flag back to 0u.
        const std::vector<std::int64_t> good_ids = {1, 2};
        d_tok.copy_from_host(good_ids.data(), good_ids.size() * sizeof(std::int64_t));
        (void)ops::run_embed_q4_rmsnorm(def_desc, args, stream.handle(), /*allow_fallback=*/true, fallback);
        hal::stream_synchronize(stream.handle());
        d_oob.copy_to_host(&flag_val, sizeof(flag_val));
        ops::check_deferred_token_bounds(flag_val, vocab_size);
        if (flag_val != 0u) {
          std::fprintf(stderr, "token-bounds DeferredFlag reset failed fallback=%d flag=%u\n", int(fallback), flag_val);
        }
        ok &= (flag_val == 0u);
      }
    }

    // 3. Scalar oracle rejects OOB instead of silently substituting zero.
    {
      bool threw = false;
      try {
        std::vector<double> ref_u, ref_r;
        scalar_embed_q4_rmsnorm_fp64(bad_ids.data(), int(bad_ids.size()), table.data(), vocab_size,
                                     norm_f32.data(), kInklingHidden, 1e-6, ingine::kActIsFP16,
                                     nullptr, &ref_u, &ref_r);
      } catch (const std::out_of_range&) {
        threw = true;
      }
      if (!threw) std::fprintf(stderr, "token-bounds scalar oracle check failed\n");
      ok &= threw;
    }
  }

  // 4. Native GPU backends enforce PreValidated kernel OOB guards without fabricating zero rows.
  if (ingine::test_contract::backend_has_native_prevalidated_guard(hal::active_backend())) {
    const std::vector<std::int64_t> bad_ids = {vocab_size + 5, 0};
    d_tok.copy_from_host(bad_ids.data(), bad_ids.size() * sizeof(std::int64_t));
    d_out.copy_from_host(init_out.data(), init_out.size() * sizeof(float));
    const std::uint32_t zero_flag = 0u;
    d_oob.copy_from_host(&zero_flag, sizeof(zero_flag));
    auto pre_desc = base_desc;
    pre_desc.token_bounds = ops::TokenBoundsMode::PreValidated;
    ops::EmbedQ4RmsNormArgs args{
        static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
        static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p),
        static_cast<std::uint32_t*>(d_oob.p)};
    const ops::DispatchPath path =
        ops::run_embed_q4_rmsnorm(pre_desc, args, stream.handle(), /*allow_fallback=*/false, false);
    hal::stream_synchronize(stream.handle());
    std::uint32_t flag_val = 0u;
    d_oob.copy_to_host(&flag_val, sizeof(flag_val));
    std::vector<float> got(2 * kInklingHidden, 0.f);
    d_out.copy_to_host(got.data(), got.size() * sizeof(float));
    bool row0_untouched = true;
    for (int j = 0; j < kInklingHidden; ++j) {
      if (got[j] != kSentinel) row0_untouched = false;
    }
    if (path != ops::DispatchPath::Native || flag_val != 1u || !row0_untouched) {
      std::fprintf(stderr, "token-bounds cuda PreValidated check failed path=%d flag=%u row0_untouched=%d\n",
                   int(path), flag_val, int(row0_untouched));
    }
    ok &= (path == ops::DispatchPath::Native) && (flag_val == 1u) && row0_untouched;
  }

  std::printf("%s unified token-bounds contract checks (actual-buffer Validate, DeferredFlag, fallback, oracle%s)\n",
              ok ? "PASS" : "FAIL",
              ingine::test_contract::backend_has_native_prevalidated_guard(hal::active_backend())
                  ? ", native PreValidated kernel guard"
                  : "");
  return ok;
}

bool check_stream_ordering_copy_kernel_read(const std::vector<std::uint8_t>& table,
                                           const std::vector<float>& norm_f32, int vocab_size) {
  constexpr int M = 2;
  const std::vector<std::int64_t> ids_a = {3, 17};
  const std::vector<std::int64_t> ids_b = {5, 19};
  const std::size_t tok_bytes = std::size_t(M) * sizeof(std::int64_t);
  const std::size_t out_bytes = std::size_t(M) * kInklingHidden * sizeof(float);
  const auto desc = ops::make_inkling_embed_q4_desc(M, vocab_size, /*allow_pdl=*/false);

  DeviceBuffer d_tbl(table.size()), d_norm(norm_f32.size() * sizeof(float));
  d_tbl.copy_from_host(table.data(), table.size());
  d_norm.copy_from_host(norm_f32.data(), norm_f32.size() * sizeof(float));

  // Build synchronized reference outputs first.
  hal::Stream ref_stream(0);
  DeviceBuffer d_ref_tok(tok_bytes), d_ref_out(out_bytes);
  std::vector<float> ref_a(std::size_t(M) * kInklingHidden, 0.f);
  std::vector<float> ref_b(std::size_t(M) * kInklingHidden, 0.f);
  std::vector<float> ref_dec(std::size_t(M) * kInklingHidden, 0.f);

  const ops::EmbedQ4RmsNormArgs ref_args{
      static_cast<const std::int64_t*>(d_ref_tok.p), d_tbl.p,
      static_cast<const float*>(d_norm.p), static_cast<float*>(d_ref_out.p), nullptr};
  hal::memcpy_h2d_async(d_ref_tok.p, ids_a.data(), tok_bytes, ref_stream.handle());
  (void)ops::run_embed_q4_rmsnorm(desc, ref_args, ref_stream.handle(), false, false);
  hal::memcpy_d2h_async(ref_a.data(), d_ref_out.p, out_bytes, ref_stream.handle());
  ref_stream.synchronize();

  hal::memcpy_h2d_async(d_ref_tok.p, ids_b.data(), tok_bytes, ref_stream.handle());
  (void)ops::run_embed_q4_rmsnorm(desc, ref_args, ref_stream.handle(), false, false);
  hal::memcpy_d2h_async(ref_b.data(), d_ref_out.p, out_bytes, ref_stream.handle());
  ops::decode_q4_0(d_tbl.p, M, kInklingHidden, static_cast<float*>(d_ref_out.p), ref_stream.handle());
  hal::memcpy_d2h_async(ref_dec.data(), d_ref_out.p, out_bytes, ref_stream.handle());
  ref_stream.synchronize();

  if (std::memcmp(ref_a.data(), ref_b.data(), out_bytes) == 0) return false;

  hal::Stream explicit_stream(0);
  const std::pair<const char*, hal::StreamHandle> streams[2] = {
      {"default", hal::StreamHandle{}},
      {"explicit", explicit_stream.handle()},
  };

  bool ok = true;
  DeviceBuffer d_tok(tok_bytes), d_out_a(out_bytes), d_copy_a(out_bytes), d_out_b(out_bytes),
      d_dec_src(out_bytes), d_dec_copy(out_bytes);
  for (const auto& [label, s] : streams) {
    const ops::EmbedQ4RmsNormArgs args_a{
        static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
        static_cast<const float*>(d_norm.p), static_cast<float*>(d_out_a.p), nullptr};
    const ops::EmbedQ4RmsNormArgs args_b{
        static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
        static_cast<const float*>(d_norm.p), static_cast<float*>(d_out_b.p), nullptr};

    // Enqueue copy -> kernel -> D2D -> clobber memset -> H2D -> kernel -> decode -> D2D -> memset -> D2H
    // on `s` without intermediate host sync to prove strict in-order execution on default and explicit streams.
    hal::memset_async(d_out_a.p, 0, out_bytes, s);
    hal::memset_async(d_copy_a.p, 0, out_bytes, s);
    hal::memset_async(d_out_b.p, 0, out_bytes, s);
    hal::memset_async(d_dec_src.p, 0, out_bytes, s);
    hal::memset_async(d_dec_copy.p, 0, out_bytes, s);

    hal::memcpy_h2d_async(d_tok.p, ids_a.data(), tok_bytes, s);
    (void)ops::run_embed_q4_rmsnorm(desc, args_a, s, false, false);
    hal::memcpy_d2d_async(d_copy_a.p, d_out_a.p, out_bytes, s);
    hal::memset_async(d_out_a.p, 0x5a, out_bytes, s);

    hal::memcpy_h2d_async(d_tok.p, ids_b.data(), tok_bytes, s);
    (void)ops::run_embed_q4_rmsnorm(desc, args_b, s, false, false);

    ops::decode_q4_0(d_tbl.p, M, kInklingHidden, static_cast<float*>(d_dec_src.p), s);
    hal::memcpy_d2d_async(d_dec_copy.p, d_dec_src.p, out_bytes, s);
    hal::memset_async(d_dec_src.p, 0x3c, out_bytes, s);

    hal::Event done(0);
    done.record(s);

    std::vector<float> got_copy_a(std::size_t(M) * kInklingHidden, 0.f);
    std::vector<std::uint8_t> got_clobber_a(out_bytes, 0u);
    std::vector<float> got_b(std::size_t(M) * kInklingHidden, 0.f);
    std::vector<float> got_dec_copy(std::size_t(M) * kInklingHidden, 0.f);
    std::vector<std::uint8_t> got_dec_clobber(out_bytes, 0u);

    hal::memcpy_d2h_async(got_copy_a.data(), d_copy_a.p, out_bytes, s);
    hal::memcpy_d2h_async(got_clobber_a.data(), d_out_a.p, out_bytes, s);
    hal::memcpy_d2h_async(got_b.data(), d_out_b.p, out_bytes, s);
    hal::memcpy_d2h_async(got_dec_copy.data(), d_dec_copy.p, out_bytes, s);
    hal::memcpy_d2h_async(got_dec_clobber.data(), d_dec_src.p, out_bytes, s);
    hal::stream_synchronize(s);
    done.synchronize();

    const bool copy_a_ok = (std::memcmp(got_copy_a.data(), ref_a.data(), out_bytes) == 0);
    const bool b_ok = (std::memcmp(got_b.data(), ref_b.data(), out_bytes) == 0);
    const bool dec_ok = (std::memcmp(got_dec_copy.data(), ref_dec.data(), out_bytes) == 0);
    bool clobber_ok = true;
    for (std::uint8_t v : got_clobber_a) clobber_ok &= (v == 0x5au);
    for (std::uint8_t v : got_dec_clobber) clobber_ok &= (v == 0x3cu);
    const bool stream_ok = done.ready() && copy_a_ok && b_ok && dec_ok && clobber_ok;
    if (!stream_ok) {
      std::fprintf(stderr, "stream-ordering(%s) failed: copy_a=%d b=%d dec=%d clobber=%d ready=%d\n",
                   label, int(copy_a_ok), int(b_ok), int(dec_ok), int(clobber_ok), int(done.ready()));
    }
    ok &= stream_ok;
  }

  std::printf("%s stream ordering (copy -> kernel -> D2D -> memset -> read on default & explicit streams)\n",
              ok ? "PASS" : "FAIL");
  return ok;
}

}  // namespace

int main() {
  hal::DeviceCaps caps{};
  try {
    caps = hal::query_device_caps(0);
    (void)hal::query_memory_info();
  } catch (const std::exception& e) {
    std::printf("skip: HAL device unavailable (%s)\n", e.what());
    return kSkip;
  }

  hal::Stream stream;
  bool ok = true;
  ok &= check_decode_q4_bit_exact(stream);
  ok &= check_caps_and_graph_policy(caps);

  constexpr int V = 64;
  Rng rng(777);

  std::vector<std::uint8_t> table(std::size_t(V) * kInklingRowBytes);
  for (int row = 0; row < V; ++row) {
    const std::vector<float> x = randn(rng, kInklingHidden, row % 7 == 0 ? 3.5 : 0.6);
    scalar_q4_0_pack_row(x.data(), 1, kInklingHidden, table.data() + std::size_t(row) * kInklingRowBytes);
  }

  std::vector<float> norm_f32(kInklingHidden);
  for (int j = 0; j < kInklingHidden; ++j) {
    const float w = float(0.5 + 1.5 * rng.uniform());
    norm_f32[j] = round_act_scalar(w, ingine::kActIsFP16);
  }

  DeviceArena arena(32ull << 20);  // 32 MiB reusable arena

  const std::vector<std::vector<std::int64_t>> cases = {
      {13},
      {0, 63, 13},
      {5, 5, 19, 0, 62},
      {1, 2, 3, 4, 5, 6, 7, 8},
  };
  for (const auto& ids : cases) {
    ok &= check_embed_q4_rmsnorm_case(int(ids.size()), ids, table, norm_f32, V, /*force_fallback=*/false,
                                      kInklingHidden, stream, arena);
  }

  // Exercise dispatch fallback on exact same M=5 inputs: packed fallback, padded native/fallback, and forced padded fallback.
  ok &= check_embed_q4_rmsnorm_case(int(cases[2].size()), cases[2], table, norm_f32, V, /*force_fallback=*/true,
                                    kInklingHidden, stream, arena);
  ok &= check_embed_q4_rmsnorm_case(int(cases[2].size()), cases[2], table, norm_f32, V, /*force_fallback=*/false,
                                    kInklingHidden + 64, stream, arena);
  ok &= check_embed_q4_rmsnorm_case(int(cases[2].size()), cases[2], table, norm_f32, V, /*force_fallback=*/true,
                                    kInklingHidden + 64, stream, arena);
  // Exercise explicit alternate ActRounding representation.
  const ops::ActRounding alt_rounding =
      ingine::kActIsFP16 ? ops::ActRounding::Bf16Rn : ops::ActRounding::Fp16RnSat;
  ok &= check_embed_q4_rmsnorm_case(int(cases[2].size()), cases[2], table, norm_f32, V, /*force_fallback=*/false,
                                    kInklingHidden, stream, arena, alt_rounding);

  ok &= check_token_bounds_contract(V, table, norm_f32, stream);
  ok &= check_stream_ordering_copy_kernel_read(table, norm_f32, V);

  // Explicit rejection checks: disabled fallback and invalid descriptors.
  {
    const auto packed_desc = ops::make_inkling_embed_q4_desc(1, V, true);
    DeviceBuffer d_tok(sizeof(std::int64_t)), d_tbl(table.size()), d_norm(norm_f32.size() * sizeof(float)),
        d_out(kInklingHidden * sizeof(float));
    const std::int64_t tok0 = 0;
    d_tok.copy_from_host(&tok0, sizeof(tok0));
    d_tbl.copy_from_host(table.data(), table.size());
    d_norm.copy_from_host(norm_f32.data(), norm_f32.size() * sizeof(float));
    ops::EmbedQ4RmsNormArgs dummy{static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
                                 static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p),
                                 nullptr};

    bool disabled_fallback_rejected = false;
    try {
      (void)ops::run_embed_q4_rmsnorm(packed_desc, dummy, stream.handle(), /*allow_fallback=*/false,
                                      /*force_fallback=*/true);
    } catch (const std::invalid_argument&) {
      disabled_fallback_rejected = true;
    }
    ok &= disabled_fallback_rejected;

    bool invalid_stride_rejected = false;
    try {
      auto bad = packed_desc;
      bad.out_row_stride = kInklingHidden - 1;
      (void)ops::resolve_embed_q4_rmsnorm(bad, /*allow_fallback=*/true);
    } catch (const std::invalid_argument&) {
      invalid_stride_rejected = true;
    }
    ok &= invalid_stride_rejected;

    if (ingine::test_contract::backend_rejects_strided_without_fallback(hal::active_backend())) {
      auto strided = packed_desc;
      strided.out_row_stride = kInklingHidden + 64;
      const ops::OpResolution res = ops::resolve_embed_q4_rmsnorm(strided, /*allow_fallback=*/false);
      bool strided_no_fallback_rejected = !res.supported;
      bool threw = false;
      try {
        (void)ops::run_embed_q4_rmsnorm(strided, dummy, stream.handle(), /*allow_fallback=*/false, false);
      } catch (const std::invalid_argument&) {
        threw = true;
      }
      ok &= strided_no_fallback_rejected && threw;
    }
    std::printf("%s descriptor/fallback explicit rejection checks\n",
                (disabled_fallback_rejected && invalid_stride_rejected) ? "PASS" : "FAIL");
  }

  std::printf("\n%s\n", ok ? "ALL PASSED" : "FAILED");
  return ok ? 0 : 1;
}

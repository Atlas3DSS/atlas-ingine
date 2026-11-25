// HAL + Ops First Parity Case: Inkling Q4_0 embedding lookup (1,296 B/row) + RMSNorm through core/hal.h,
// core/arena.h, and core/ops.h on both CPU and CUDA with zero direct CUDA headers in the test translation unit.
#include "core/act.h"
#include "core/arena.h"
#include "core/hal.h"
#include "core/ops.h"
#include "common/scalar_oracle.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ingine;
using namespace ingine::test;

namespace {

constexpr int kInklingHidden = 2304;
constexpr int kInklingRowBytes = 1296;

bool check_decode_q4_bit_exact() {
  constexpr int V = 16;
  Rng rng(20261124ull);
  bool ok = true;

  for (int row = 0; row < V; ++row) {
    const std::vector<float> src = randn(rng, kInklingHidden, 1.5);
    std::vector<std::uint8_t> packed(kInklingRowBytes);
    scalar_q4_0_pack_row(src.data(), 1, kInklingHidden, packed.data());

    std::vector<float> ref(kInklingHidden), got(kInklingHidden);
    scalar_q4_0_unpack_row(packed.data(), 1, kInklingHidden, ref.data());
    ops::decode_q4_0_row(packed.data(), kInklingHidden, got.data());
    for (int j = 0; j < kInklingHidden; ++j) {
      std::uint32_t rb = 0, gb = 0;
      std::memcpy(&rb, &ref[j], 4);
      std::memcpy(&gb, &got[j], 4);
      if (rb != gb) {
        std::fprintf(stderr, "decode_q4_0_row mismatch row=%d col=%d ref=0x%08x got=0x%08x\n", row, j, rb, gb);
        ok = false;
        break;
      }
    }
  }
  std::printf("%s decode_q4_0_row bit-exact vs scalar binary16 unpack (1,296 B/row)\n", ok ? "PASS" : "FAIL");
  return ok;
}

bool check_caps_and_graph_policy(const hal::DeviceCaps& caps) {
  bool ok = true;
#if INGINE_BACKEND_CUDA
  ok &= (hal::active_backend() == hal::BackendKind::Cuda);
  ok &= (caps.backend == hal::BackendKind::Cuda);
  ok &= (caps.subgroup.preferred_width == 32 && caps.subgroup.min_width == 32 && caps.subgroup.max_width == 32);
#else
  ok &= (hal::active_backend() == hal::BackendKind::Cpu);
  ok &= (caps.backend == hal::BackendKind::Cpu);
  ok &= (!caps.supports_conditional_graph && !caps.supports_graph_capture);
  ok &= (caps.subgroup.preferred_width == 1);
#endif
  ok &= (!hal::is_stream_capturing({}));

  hal::DeviceCaps no_graph = caps;
  no_graph.supports_graph_capture = false;
  no_graph.supports_conditional_graph = false;
  bool rejected = false;
  try {
    hal::require_graph_capture_or_eager(no_graph, /*allow_eager=*/false, "hal_embed_q4_parity");
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  ok &= rejected;
  hal::require_graph_capture_or_eager(no_graph, /*allow_eager=*/true, "hal_embed_q4_parity");

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
  std::printf("%s HAL capabilities (%s, subgroup=%d), eager-or-reject policy, and DenseGemmDesc contract\n",
              ok ? "PASS" : "FAIL", hal::backend_name(caps.backend), caps.subgroup.preferred_width);
  return ok;
}

bool check_embed_q4_rmsnorm_case(int M, const std::vector<std::int64_t>& tok_ids,
                                 const std::vector<std::uint8_t>& table, const std::vector<float>& norm_f32,
                                 int vocab_size, bool force_fallback, int out_stride, hal::Stream& stream,
                                 DeviceArena& arena) {
  constexpr double kEps = 1e-6;
  ops::EmbedQ4RmsNormDesc desc = ops::make_inkling_embed_q4_desc(M, vocab_size, /*allow_pdl=*/true);
  desc.out_row_stride = out_stride;

  std::vector<float> ref_decoded;
  std::vector<double> ref_unrounded, ref_rounded;
  scalar_embed_q4_rmsnorm_fp64(tok_ids.data(), M, table.data(), vocab_size, norm_f32.data(), kInklingHidden, kEps,
                               ingine::kActIsFP16, &ref_decoded, &ref_unrounded, &ref_rounded);

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

  const bool expect_fallback =
      force_fallback || (hal::active_backend() == hal::BackendKind::Cuda && out_stride != kInklingHidden);
  const ops::DispatchPath expected_path =
      expect_fallback ? ops::DispatchPath::PortableFallback : ops::DispatchPath::Native;
  bool ok = (path1 == expected_path && path2 == expected_path && out1.size() == out2.size());
  for (std::size_t i = 0; i < out1.size() && ok; ++i) {
    if (std::memcmp(&out1[i], &out2[i], sizeof(float)) != 0) ok = false;
  }

  std::vector<float> got_packed(std::size_t(M) * kInklingHidden);
  bool sentinel_ok = true;
  for (int m = 0; m < M; ++m) {
    for (int j = 0; j < kInklingHidden; ++j) {
      const float v = out1[std::size_t(m) * out_stride + j];
      if (!std::isfinite(v)) ok = false;
      got_packed[std::size_t(m) * kInklingHidden + j] = v;
    }
    for (int j = kInklingHidden; j < out_stride; ++j) {
      if (out1[std::size_t(m) * out_stride + j] != kSentinel) sentinel_ok = false;
    }
  }
  ok &= sentinel_ok;

  const double tol_rel = ingine::kActIsFP16 ? 1e-2 : 2e-2;
  const char* label = expect_fallback ? "PortableFallback" : "Native";
  char msg[128];
  std::snprintf(msg, sizeof(msg), "embed_q4_rmsnorm[%s M=%d stride=%d]", label, M, out_stride);
  const OracleCheck chk{msg, tol_rel, 1e-5, 0.01};
  ok &= chk(got_packed, ref_unrounded, &ref_rounded);
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

  bool ok = true;
  ok &= check_decode_q4_bit_exact();
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

  hal::Stream stream;
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

  // Exercise dispatch fallback on exact same M=5 inputs (packed and padded stride).
  ok &= check_embed_q4_rmsnorm_case(int(cases[2].size()), cases[2], table, norm_f32, V, /*force_fallback=*/true,
                                    kInklingHidden, stream, arena);
  ok &= check_embed_q4_rmsnorm_case(int(cases[2].size()), cases[2], table, norm_f32, V, /*force_fallback=*/false,
                                    kInklingHidden + 64, stream, arena);

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
                                 static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p)};

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

    if (hal::active_backend() == hal::BackendKind::Cuda) {
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

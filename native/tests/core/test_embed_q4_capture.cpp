// CUDA graph capture test for ops::run_embed_q4_rmsnorm:
//   * Stream-ordered producer/readback synchronization (no cross-stream default-stream readback race).
//   * TokenBoundsMode::PreValidated captures and replays bit-identically across mutated valid IDs.
//   * TokenBoundsMode::DeferredFlag captures once, resets and validates actual device IDs on EVERY replay
//     (pass on valid ids1 -> detect OOB mutation & guard table read on bad_ids -> reset & pass on valid ids2).
//   * TokenBoundsMode::Validate is always disallowed inside stream capture (throws std::logic_error).
#include "common/scalar_oracle.h"
#include "common/test_util.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/hal.h"
#include "core/ops.h"

#include "core/gpu_compat.h"

#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ingine;
using namespace ingine::test;

namespace {

constexpr int kInklingHidden = 2304;
constexpr int kInklingRowBytes = 1296;

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  hal::DeviceCaps caps = hal::query_device_caps(0);
  hal::require_graph_capture_or_eager(caps, /*allow_eager=*/false, "hal_embed_q4_capture",
                                      hal::GraphRequirement::Capture);

  constexpr int V = 32;
  constexpr int M = 3;
  Rng rng(20261125ull);
  std::vector<std::uint8_t> table(std::size_t(V) * kInklingRowBytes);
  for (int r = 0; r < V; ++r) {
    const auto x = randn(rng, kInklingHidden, 1.0);
    scalar_q4_0_pack_row(x.data(), 1, kInklingHidden, table.data() + std::size_t(r) * kInklingRowBytes);
  }
  std::vector<float> norm(kInklingHidden);
  for (float& w : norm) w = round_bf16(float(0.5 + rng.uniform()));

  const std::vector<std::int64_t> ids1 = {3, 17, 29};
  const std::vector<std::int64_t> ids2 = {5, 11, 23};
  const std::vector<std::int64_t> bad_ids = {3, V, 29};

  DeviceBuffer d_tbl(table.size()), d_norm(norm.size() * sizeof(float)),
      d_tok(ids1.size() * sizeof(std::int64_t)), d_out(ids1.size() * kInklingHidden * sizeof(float)),
      d_oob(sizeof(std::uint32_t));

  hal::Stream stream(0);
  const cudaStream_t cs = static_cast<cudaStream_t>(stream.handle());

  auto upload = [&](void* dst, const void* src, std::size_t bytes) {
    hal::memcpy_h2d_async(dst, src, bytes, stream.handle());
    stream.synchronize();
  };
  auto download = [&](void* dst, const void* src, std::size_t bytes) {
    hal::memcpy_d2h_async(dst, src, bytes, stream.handle());
    stream.synchronize();
  };
  auto zero_dev = [&](void* dst, std::size_t bytes) {
    INGINE_TEST_CUDA(cudaMemsetAsync(dst, 0, bytes, cs));
    stream.synchronize();
  };
  auto fill_out = [&](float val) {
    const std::vector<float> buf(std::size_t(M) * kInklingHidden, val);
    upload(d_out.p, buf.data(), buf.size() * sizeof(float));
  };

  upload(d_tbl.p, table.data(), table.size());
  upload(d_norm.p, norm.data(), norm.size() * sizeof(float));

  // Synchronized eager references (Validate checks actual d_tok before compute).
  const auto desc_val = ops::make_inkling_embed_q4_desc(M, V, /*allow_pdl=*/false);
  const ops::EmbedQ4RmsNormArgs args_no_flag{
      static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
      static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p), nullptr};
  const ops::EmbedQ4RmsNormArgs args_with_flag{
      static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
      static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p),
      static_cast<std::uint32_t*>(d_oob.p)};

  upload(d_tok.p, ids1.data(), ids1.size() * sizeof(std::int64_t));
  zero_dev(d_out.p, d_out.bytes);
  INGINE_CHECK(ops::run_embed_q4_rmsnorm(desc_val, args_with_flag, stream.handle(), false, false) ==
               ops::DispatchPath::Native);
  std::vector<float> eager_out1(std::size_t(M) * kInklingHidden, 0.f);
  download(eager_out1.data(), d_out.p, eager_out1.size() * sizeof(float));

  upload(d_tok.p, ids2.data(), ids2.size() * sizeof(std::int64_t));
  zero_dev(d_out.p, d_out.bytes);
  INGINE_CHECK(ops::run_embed_q4_rmsnorm(desc_val, args_with_flag, stream.handle(), false, false) ==
               ops::DispatchPath::Native);
  std::vector<float> eager_out2(std::size_t(M) * kInklingHidden, 0.f);
  download(eager_out2.data(), d_out.p, eager_out2.size() * sizeof(float));
  INGINE_CHECK(std::memcmp(eager_out1.data(), eager_out2.data(), eager_out1.size() * sizeof(float)) != 0);

  // Validate rejects OOB synchronously without touching d_out.
  upload(d_tok.p, bad_ids.data(), bad_ids.size() * sizeof(std::int64_t));
  fill_out(-55.25f);
  {
    bool oob_threw = false;
    try {
      (void)ops::run_embed_q4_rmsnorm(desc_val, args_with_flag, stream.handle(), false, false);
    } catch (const std::out_of_range&) {
      oob_threw = true;
    }
    INGINE_CHECK(oob_threw);
    std::uint32_t flag = 0u;
    download(&flag, d_oob.p, sizeof(flag));
    INGINE_CHECK(flag == 1u);
    std::vector<float> untouched(eager_out1.size(), 0.f);
    download(untouched.data(), d_out.p, untouched.size() * sizeof(float));
    for (float v : untouched) INGINE_CHECK(v == -55.25f);
  }

  // 1. PreValidated capture + replays across ids1 -> ids2.
  auto desc_pre = ops::make_inkling_embed_q4_desc(M, V, /*allow_pdl=*/false);
  desc_pre.token_bounds = ops::TokenBoundsMode::PreValidated;
  upload(d_tok.p, ids1.data(), ids1.size() * sizeof(std::int64_t));
  zero_dev(d_out.p, d_out.bytes);
  DecodeGraphDefinition def_pre;
  def_pre.capture(cs, [&] {
    INGINE_CHECK(ops::run_embed_q4_rmsnorm(desc_pre, args_no_flag, stream.handle(), false, false) ==
                 ops::DispatchPath::Native);
  });
  DecodeGraphExecutable exec_pre;
  exec_pre.instantiate(def_pre);
  exec_pre.upload(cs);

  exec_pre.launch(cs);
  std::vector<float> cap_pre_out1(eager_out1.size(), 0.f);
  download(cap_pre_out1.data(), d_out.p, cap_pre_out1.size() * sizeof(float));
  INGINE_CHECK(std::memcmp(eager_out1.data(), cap_pre_out1.data(), eager_out1.size() * sizeof(float)) == 0);

  upload(d_tok.p, ids2.data(), ids2.size() * sizeof(std::int64_t));
  zero_dev(d_out.p, d_out.bytes);
  exec_pre.launch(cs);
  std::vector<float> cap_pre_out2(eager_out2.size(), 0.f);
  download(cap_pre_out2.data(), d_out.p, cap_pre_out2.size() * sizeof(float));
  INGINE_CHECK(std::memcmp(eager_out2.data(), cap_pre_out2.data(), eager_out2.size() * sizeof(float)) == 0);

  // 2. DeferredFlag requires non-null invalid_token_flag (both eager and during capture).
  auto desc_def = ops::make_inkling_embed_q4_desc(M, V, /*allow_pdl=*/false);
  desc_def.token_bounds = ops::TokenBoundsMode::DeferredFlag;
  {
    bool eager_null_threw = false;
    try {
      (void)ops::run_embed_q4_rmsnorm(desc_def, args_no_flag, stream.handle(), false, false);
    } catch (const std::invalid_argument&) {
      eager_null_threw = true;
    }
    INGINE_CHECK(eager_null_threw);

    bool cap_null_threw = false;
    try {
      DecodeGraphDefinition def_null;
      def_null.capture(cs, [&] {
        (void)ops::run_embed_q4_rmsnorm(desc_def, args_no_flag, stream.handle(), false, false);
      });
    } catch (const std::invalid_argument&) {
      cap_null_threw = true;
    }
    INGINE_CHECK(cap_null_threw);
    INGINE_CHECK(!hal::is_stream_capturing(stream.handle()));
  }

  // 3. DeferredFlag captured ONCE and replayed across:
  //    replay 1 (valid ids1, pre-seeded stale flag=1 to prove per-replay reset) ->
  //    replay 2 (mutated OOB bad_ids on device without recapturing -> flag=1 & sentinel row untouched) ->
  //    replay 3 (mutated valid ids2 without external flag clear -> flag=0 & bit-identical eager_out2).
  upload(d_tok.p, ids1.data(), ids1.size() * sizeof(std::int64_t));
  zero_dev(d_out.p, d_out.bytes);
  zero_dev(d_oob.p, sizeof(std::uint32_t));
  DecodeGraphDefinition def_def;
  def_def.capture(cs, [&] {
    INGINE_CHECK(ops::run_embed_q4_rmsnorm(desc_def, args_with_flag, stream.handle(), false, false) ==
                 ops::DispatchPath::Native);
  });
  DecodeGraphExecutable exec_def;
  exec_def.instantiate(def_def);
  exec_def.upload(cs);

  // Replay 1 (with stale flag = 1 beforehand to verify captured reset).
  const std::uint32_t one = 1u;
  upload(d_oob.p, &one, sizeof(one));
  zero_dev(d_out.p, d_out.bytes);
  exec_def.launch(cs);
  std::uint32_t flag = 1u;
  download(&flag, d_oob.p, sizeof(flag));
  INGINE_CHECK(flag == 0u);
  ops::check_deferred_token_bounds(flag, V);
  std::vector<float> cap_def_out1(eager_out1.size(), 0.f);
  download(cap_def_out1.data(), d_out.p, cap_def_out1.size() * sizeof(float));
  INGINE_CHECK(std::memcmp(eager_out1.data(), cap_def_out1.data(), eager_out1.size() * sizeof(float)) == 0);

  // Replay 2 (mutate device token buffer to OOB bad_ids = {3, V, 29} and re-launch SAME captured graph).
  upload(d_tok.p, bad_ids.data(), bad_ids.size() * sizeof(std::int64_t));
  fill_out(-77.5f);
  exec_def.launch(cs);
  download(&flag, d_oob.p, sizeof(flag));
  INGINE_CHECK(flag == 1u);
  {
    bool deferred_threw = false;
    try {
      ops::check_deferred_token_bounds(flag, V);
    } catch (const std::out_of_range&) {
      deferred_threw = true;
    }
    INGINE_CHECK(deferred_threw);
  }
  std::vector<float> cap_def_oob(eager_out1.size(), 0.f);
  download(cap_def_oob.data(), d_out.p, cap_def_oob.size() * sizeof(float));
  for (int j = 0; j < kInklingHidden; ++j) {
    INGINE_CHECK(cap_def_oob[std::size_t(1) * kInklingHidden + j] == -77.5f);
  }
  INGINE_CHECK(std::memcmp(cap_def_oob.data(), eager_out1.data(), kInklingHidden * sizeof(float)) == 0);
  INGINE_CHECK(std::memcmp(cap_def_oob.data() + 2 * kInklingHidden,
                           eager_out1.data() + 2 * kInklingHidden, kInklingHidden * sizeof(float)) == 0);

  // Replay 3 (mutate back to valid ids2 and confirm flag resets to 0 and outputs match eager_out2).
  upload(d_tok.p, ids2.data(), ids2.size() * sizeof(std::int64_t));
  zero_dev(d_out.p, d_out.bytes);
  exec_def.launch(cs);
  download(&flag, d_oob.p, sizeof(flag));
  INGINE_CHECK(flag == 0u);
  ops::check_deferred_token_bounds(flag, V);
  std::vector<float> cap_def_out2(eager_out2.size(), 0.f);
  download(cap_def_out2.data(), d_out.p, cap_def_out2.size() * sizeof(float));
  INGINE_CHECK(std::memcmp(eager_out2.data(), cap_def_out2.data(), eager_out2.size() * sizeof(float)) == 0);

  // 4. Validate inside capture always throws std::logic_error cleanly.
  upload(d_tok.p, ids1.data(), ids1.size() * sizeof(std::int64_t));
  bool validate_cap_threw = false;
  try {
    DecodeGraphDefinition def_sync;
    def_sync.capture(cs, [&] {
      (void)ops::run_embed_q4_rmsnorm(desc_val, args_with_flag, stream.handle(), false, false);
    });
  } catch (const std::logic_error&) {
    validate_cap_threw = true;
  }
  INGINE_CHECK(validate_cap_threw);
  INGINE_CHECK(!hal::is_stream_capturing(stream.handle()));

  std::printf("PASS hal_embed_q4_capture\n");
  return 0;
}

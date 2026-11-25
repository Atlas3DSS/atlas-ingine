// CUDA graph capture test for ops::run_embed_q4_rmsnorm:
//   * TokenBoundsMode::PreValidated captures and replays bit-identically to eager native execution.
//   * TokenBoundsMode::Validate with host_token_ids != nullptr captures and replays cleanly, and rejects OOB
//     host tokens before enqueueing work.
//   * TokenBoundsMode::Validate with host_token_ids == nullptr during capture throws std::logic_error cleanly.
#include "common/scalar_oracle.h"
#include "common/test_util.h"
#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/hal.h"
#include "core/ops.h"

#include <cuda_runtime.h>

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

  const std::vector<std::int64_t> ids = {3, 17, 29};
  DeviceBuffer d_tbl(table.size()), d_norm(norm.size() * sizeof(float)),
      d_tok(ids.size() * sizeof(std::int64_t)), d_out(ids.size() * kInklingHidden * sizeof(float));
  d_tbl.copy_from_host(table.data(), table.size());
  d_norm.copy_from_host(norm.data(), norm.size() * sizeof(float));
  d_tok.copy_from_host(ids.data(), ids.size() * sizeof(std::int64_t));

  hal::Stream stream(0);
  const cudaStream_t cs = static_cast<cudaStream_t>(stream.handle());

  auto desc_pre = ops::make_inkling_embed_q4_desc(M, V, /*allow_pdl=*/false);
  desc_pre.token_bounds = ops::TokenBoundsMode::PreValidated;
  const ops::EmbedQ4RmsNormArgs args_no_host{
      static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
      static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p), nullptr, nullptr};

  // Eager reference.
  INGINE_CHECK(ops::run_embed_q4_rmsnorm(desc_pre, args_no_host, stream.handle(), false, false) ==
               ops::DispatchPath::Native);
  std::vector<float> eager_out(std::size_t(M) * kInklingHidden, 0.f);
  d_out.copy_to_host(eager_out.data(), eager_out.size() * sizeof(float));

  // 1. PreValidated capture + replay.
  d_out.fill(0);
  DecodeGraphDefinition def_pre;
  def_pre.capture(cs, [&] {
    INGINE_CHECK(ops::run_embed_q4_rmsnorm(desc_pre, args_no_host, stream.handle(), false, false) ==
                 ops::DispatchPath::Native);
  });
  DecodeGraphExecutable exec_pre;
  exec_pre.instantiate(def_pre);
  exec_pre.upload(cs);
  exec_pre.launch(cs);
  stream.synchronize();
  std::vector<float> cap_pre_out(eager_out.size(), 0.f);
  d_out.copy_to_host(cap_pre_out.data(), cap_pre_out.size() * sizeof(float));
  INGINE_CHECK(std::memcmp(eager_out.data(), cap_pre_out.data(), eager_out.size() * sizeof(float)) == 0);

  // 2. Validate with host_token_ids != nullptr during capture + replay.
  d_out.fill(0);
  const auto desc_val = ops::make_inkling_embed_q4_desc(M, V, /*allow_pdl=*/false);
  const ops::EmbedQ4RmsNormArgs args_host{
      static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
      static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p), ids.data(), nullptr};
  DecodeGraphDefinition def_val;
  def_val.capture(cs, [&] {
    INGINE_CHECK(ops::run_embed_q4_rmsnorm(desc_val, args_host, stream.handle(), false, false) ==
                 ops::DispatchPath::Native);
  });
  DecodeGraphExecutable exec_val;
  exec_val.instantiate(def_val);
  exec_val.upload(cs);
  exec_val.launch(cs);
  stream.synchronize();
  std::vector<float> cap_val_out(eager_out.size(), 0.f);
  d_out.copy_to_host(cap_val_out.data(), cap_val_out.size() * sizeof(float));
  INGINE_CHECK(std::memcmp(eager_out.data(), cap_val_out.data(), eager_out.size() * sizeof(float)) == 0);

  // 3. Validate with OOB host_token_ids during capture throws std::out_of_range and aborts capture cleanly.
  const std::vector<std::int64_t> bad_ids = {3, V, 29};
  const ops::EmbedQ4RmsNormArgs args_bad_host{
      static_cast<const std::int64_t*>(d_tok.p), d_tbl.p,
      static_cast<const float*>(d_norm.p), static_cast<float*>(d_out.p), bad_ids.data(), nullptr};
  bool oob_threw = false;
  try {
    DecodeGraphDefinition def_bad;
    def_bad.capture(cs, [&] {
      (void)ops::run_embed_q4_rmsnorm(desc_val, args_bad_host, stream.handle(), false, false);
    });
  } catch (const std::out_of_range&) {
    oob_threw = true;
  }
  INGINE_CHECK(oob_threw);
  INGINE_CHECK(!hal::is_stream_capturing(stream.handle()));

  // 4. Validate with host_token_ids == nullptr during capture throws std::logic_error cleanly.
  bool sync_cap_threw = false;
  try {
    DecodeGraphDefinition def_sync;
    def_sync.capture(cs, [&] {
      (void)ops::run_embed_q4_rmsnorm(desc_val, args_no_host, stream.handle(), false, false);
    });
  } catch (const std::logic_error&) {
    sync_cap_threw = true;
  }
  INGINE_CHECK(sync_cap_threw);
  INGINE_CHECK(!hal::is_stream_capturing(stream.handle()));

  std::printf("PASS hal_embed_q4_capture\n");
  return 0;
}

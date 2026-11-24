// model_data.h constants against weights.py / model.py / fast.py (CPU only): the visit plan, ring/full cache
// indices, tap visits and the projection layout.
#include "ingine/generator.h"
#include "model/model_data.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace ingine::model;

int main() {
  bool ok = true;
  auto expect = [&](bool c, const char* what) {
    if (!c) std::printf("FAIL %s\n", what);
    ok &= c;
  };
  // weights.visit_plan: layers 0-5, 6-17 three times (passes 0, 1, 2), 18-23
  int v = 0;
  for (int l = 0; l < 6; ++l, ++v) expect(VISITS[v].layer == l && VISITS[v].pass == 0, "visits 0-5");
  for (int p = 0; p < 3; ++p)
    for (int l = 6; l < 18; ++l, ++v) expect(VISITS[v].layer == l && VISITS[v].pass == p, "loop visits");
  for (int l = 18; l < 24; ++l, ++v) expect(VISITS[v].layer == l && VISITS[v].pass == 0, "visits 18-23");
  expect(v == NUM_VISITS, "48 visits");
  int nf = 0, ns = 0;
  for (int i = 0; i < NUM_VISITS; ++i) {
    const bool full = is_full_layer(VISITS[i].layer);
    expect(full ? FULL_INDEX[i] == nf++ && RING_INDEX[i] == -1 : RING_INDEX[i] == ns++ && FULL_INDEX[i] == -1,
           "cache indices");
  }
  expect(nf == NUM_FULL_VISITS && ns == NUM_SLIDING_VISITS, "8 full / 40 sliding visits");
  // model.TAP_VISITS = (5, 23, 41): layer 5, layer 11 of pass 1, layer 17 of pass 2
  expect(VISITS[TAP_VISITS[0]].layer == 5 && VISITS[TAP_VISITS[1]].layer == 11 && VISITS[TAP_VISITS[1]].pass == 1 &&
             VISITS[TAP_VISITS[2]].layer == 17 && VISITS[TAP_VISITS[2]].pass == 2,
         "tap visits");
  // SPEC row order: q 3072 | gate 3072 | rel 384 | sink 24 | k 1024 | v 1024 | k_dyn 32 | v_dyn 32
  const int begin[] = {0, 3072, 6144, 6528, 6552, 7576, 8600, 8632, 8664};
  for (int s = 0; s <= NUM_SEGS; ++s) expect(SEG_BEGIN[s] == begin[s], "projection layout");
  expect(PROJ_ROWS_SLIDING == 8600 && PROJ_ROWS_FULL == 8664, "projection rows");
  expect(KV_Q4_ROW_BYTES == 576 && NUM_FULL_VISITS * 2 * KV_Q4_ROW_BYTES == 9216, "q4 kv row bytes (9 KiB/tok)");

  // 1M-token (1,048,576) 64-bit byte-size, pitch, and offset overflow checks (no GPU needed)
  constexpr int kOneM = 1 << 20;
  const ingine::Tensor kq4 = rm(nullptr, ingine::DType::U8, {NUM_FULL_VISITS, kOneM, KV_Q4_ROW_BYTES});
  expect(kq4.bytes() == 4831838208ull, "1M q4_0 k_full == 4.50 GiB");
  expect(kq4.bytes() * 2 == 9663676416ull, "1M q4_0 k+v == 9.00 GiB");
  // Visit 7 offset exceeds 32-bit uint32_t max (4,294,967,295): verify 64-bit pointer arithmetic
  const std::size_t q4_visit7 = std::size_t(NUM_FULL_VISITS - 1) * std::size_t(kOneM) * KV_Q4_ROW_BYTES;
  expect(q4_visit7 == 4227858432ull, "1M q4_0 visit 7 byte offset");
  const ingine::Tensor kbf16 = rm(nullptr, ingine::DType::BF16, {NUM_FULL_VISITS, kOneM, KV_HEADS, HEAD_DIM});
  expect(kbf16.bytes() == 17179869184ull, "1M bf16 k_full == 16.00 GiB");
  expect(kbf16.bytes() / NUM_FULL_VISITS == 2147483648ull, "1M bf16 per-visit pitch == 2.00 GiB (> INT32_MAX)");
  // log_scale at 1M: float(pos + 1) exact (< 2^24), tau = 1 + 0.1 * ln(128) ~= 1.485203
  const float pos_f = float(kOneM);
  const float tau_1m = 1.f + 0.1f * std::log(std::fmax(pos_f * (1.f / 8192.f), 1.f));
  expect(pos_f == 1048576.f && std::isfinite(tau_1m) && std::fabs(tau_1m - 1.4852030f) < 1e-5f, "1M log_scale tau");

  // Shared prefix-cache contract checks (#825/#826): StateIdentity, hashes, root validation, Reuse strings.
  {
    StateIdentity a;
    a.model_sha256[0] = 0xAB;
    a.act_dtype = 0;
    a.dense_fp8 = true;
    a.kv_format = 1;
    a.max_ctx = 65536;
    a.prefill_chunk = 3072;
    a.spec_enabled = false;
    a.persona_key = 0;
    a.cache_namespace = "";
    StateIdentity b = a;
    expect(a == b && hash_state_identity(a) == hash_state_identity(b), "StateIdentity equal & hash match");
    b.cache_namespace = "tenant-b";
    expect(!(a == b), "StateIdentity cache_namespace isolation");
    b = a;
    b.persona_key = 42;
    b.persona_fingerprint = "deadbeef";
    expect(!(a == b), "StateIdentity persona isolation");
    b = a;
    b.kv_format = 2;
    expect(!(a == b), "StateIdentity kv_format isolation");
    b = a;
    b.prefill_chunk = 512;
    expect(!(a == b), "StateIdentity prefill_chunk isolation");
    b = a;
    b.spec_enabled = true;
    expect(!(a == b), "StateIdentity spec_enabled isolation");

    const std::vector<std::int32_t> ids = {1, 2, 3, -1, -2, -3, -4, 8, 9};
    PromptImage img;
    img.pos = 3;
    img.height = 80;
    img.width = 80;  // 2x2 = 4 tokens covering [3, 7)
    const PromptImage imgs[1] = {img};
    expect(validate_root_prefix_tokens(ids, imgs, 3) == 3, "root boundary before image is valid");
    expect(validate_root_prefix_tokens(ids, imgs, 7) == 7, "root boundary after image is valid");
    expect(validate_root_prefix_tokens(ids, imgs, 5) == 0, "root boundary straddling image rejected");
    expect(validate_root_prefix_tokens(ids, imgs, -1) == 0, "negative root boundary rejected");
    expect(validate_root_prefix_tokens(ids, imgs, 10) == 0, "out-of-range root boundary rejected");
    expect(std::strcmp(to_string(Reuse::RootCheckpoint), "root_checkpoint") == 0 &&
               std::strcmp(to_string(Reuse::HostParked), "host_parked") == 0 &&
               std::strcmp(to_string(Reuse::PagedPrefix), "paged_prefix") == 0,
           "Reuse enum to_string contract");
    expect(hash_token_prefix(std::span(ids).subspan(0, 3)) != hash_token_prefix(std::span(ids).subspan(0, 4)),
           "hash_token_prefix distinguishes prefix lengths");
  }

  std::printf("%s model_data constants\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

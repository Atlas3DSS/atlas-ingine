// model_data.h constants against weights.py / model.py / fast.py (CPU only): the visit plan, ring/full cache
// indices, tap visits and the projection layout.
#include "core/act.h"
#include "ingine/generator.h"
#if INGINE_BACKEND_CUDA
#include "core/safetensors.h"
#include "core/weight_pack.h"
#include "model/kv_checkpoint.h"
#endif
#include "model/model_data.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
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
    a.kv_cache_mode = 0;
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
    b.kv_cache_mode = 1;
    expect(!(a == b) && hash_state_identity(a) != hash_state_identity(b), "StateIdentity kv_cache_mode isolation");
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

#if INGINE_BACKEND_CUDA
  // Disk KV checkpoint boundary dtype & eligibility checks (BF16=0 vs FP16=6 raw DType vs StateIdentity 0/1).
  {
    const std::uint32_t kDiskBf16 = static_cast<std::uint32_t>(ingine::DType::BF16);
    const std::uint32_t kDiskFp16 = static_cast<std::uint32_t>(ingine::DType::FP16);
    const std::uint32_t kDiskFp32 = static_cast<std::uint32_t>(ingine::DType::FP32);
    expect(kDiskBf16 == 0u && kDiskFp32 == 1u && kDiskFp16 == 6u, "DType raw codes (BF16=0, FP32=1, FP16=6)");

    const std::uint32_t build_id_act = (ingine::kActDType == ingine::DType::FP16) ? 1u : 0u;
    const std::uint32_t build_disk_act = static_cast<std::uint32_t>(ingine::kActDType);
    expect(kv_checkpoint_act_dtype_matches(build_disk_act, build_id_act),
           "active build kActDType matches its own StateIdentity::act_dtype");

    // Same-dtype matches
    expect(kv_checkpoint_act_dtype_matches(kDiskBf16, 0u), "same-BF16 disk(0) == identity(0)");
    expect(kv_checkpoint_act_dtype_matches(kDiskFp16, 1u), "same-FP16 disk(6) == identity(1)");

    // Mixed-dtype & raw-code collision rejections (fail-closed)
    expect(!kv_checkpoint_act_dtype_matches(kDiskFp16, 0u), "mixed BF16 engine rejects FP16 disk(6)");
    expect(!kv_checkpoint_act_dtype_matches(kDiskBf16, 1u), "mixed FP16 engine rejects BF16 disk(0)");
    expect(!kv_checkpoint_act_dtype_matches(kDiskFp32, 1u), "FP16 engine rejects raw FP32 disk(1) collision");
    expect(!kv_checkpoint_act_dtype_matches(kDiskFp32, 0u), "BF16 engine rejects raw FP32 disk(1)");
    expect(!kv_checkpoint_act_dtype_matches(kDiskFp16, 6u), "unknown StateIdentity::act_dtype(6) fails closed");
    expect(!kv_checkpoint_act_dtype_matches(99u, 0u) && !kv_checkpoint_act_dtype_matches(99u, 1u),
           "unknown disk act_dtype(99) fails closed");

    StateIdentity id_bf16;
    id_bf16.model_sha256[0] = 0x42;
    id_bf16.act_dtype = 0u;
    id_bf16.dense_fp8 = true;
    id_bf16.kv_format = 1u;
    id_bf16.max_ctx = 65536;
    id_bf16.prefill_chunk = 3072;
    StateIdentity id_fp16 = id_bf16;
    id_fp16.act_dtype = 1u;

    const auto prod_bf16 = compute_disk_producer_sha256(id_bf16);
    const auto prod_fp16 = compute_disk_producer_sha256(id_fp16);
    expect(prod_bf16 != prod_fp16, "producer SHA distinguishes BF16 vs FP16");

    // Deliberate compatible case: max_ctx differs (pos <= max_ctx) -> identical producer SHA
    StateIdentity id_fp16_bigctx = id_fp16;
    id_fp16_bigctx.max_ctx = 131072;
    expect(compute_disk_producer_sha256(id_fp16_bigctx) == prod_fp16,
           "producer SHA deliberately invariant to max_ctx (pos <= runtime.max_ctx checked at load)");

    // Producer-affecting changes MUST alter producer SHA
    StateIdentity id_fp16_densebf = id_fp16;
    id_fp16_densebf.dense_fp8 = false;
    StateIdentity id_fp16_chunk = id_fp16;
    id_fp16_chunk.prefill_chunk = 1024;
    StateIdentity id_fp16_otherpkg = id_fp16;
    id_fp16_otherpkg.model_sha256[0] = 0x99;
    expect(compute_disk_producer_sha256(id_fp16_densebf) != prod_fp16 &&
               compute_disk_producer_sha256(id_fp16_chunk) != prod_fp16 &&
               compute_disk_producer_sha256(id_fp16_otherpkg) != prod_fp16,
           "producer SHA pins weights/model_sha256, dense_fp8, and prefill_chunk");

    KvCheckpointInfo ckpt_bf16;
    ckpt_bf16.pos = 128;
    ckpt_bf16.max_ctx_at_save = 8192;
    ckpt_bf16.act_dtype = kDiskBf16;
    ckpt_bf16.kv_format = 1u;
    ckpt_bf16.has_logits = true;
    ckpt_bf16.has_producer_identity = true;
    ckpt_bf16.producer_dense_fp8 = true;
    ckpt_bf16.producer_prefill_chunk = 3072;
    ckpt_bf16.producer_sha256 = prod_bf16;

    KvCheckpointInfo ckpt_fp16 = ckpt_bf16;
    ckpt_fp16.act_dtype = kDiskFp16;
    ckpt_fp16.producer_sha256 = prod_fp16;

    KvCheckpointInfo ckpt_fp32_bogus = ckpt_fp16;
    ckpt_fp32_bogus.act_dtype = kDiskFp32;

    // Runtime match & auto-resume eligibility across same-BF16, same-FP16, and mixed-dtype
    expect(kv_checkpoint_runtime_matches(ckpt_bf16, id_bf16), "runtime match same-BF16");
    expect(kv_checkpoint_runtime_matches(ckpt_fp16, id_fp16), "runtime match same-FP16");
    expect(!kv_checkpoint_runtime_matches(ckpt_fp16, id_bf16), "runtime reject BF16 engine vs FP16 ckpt");
    expect(!kv_checkpoint_runtime_matches(ckpt_bf16, id_fp16), "runtime reject FP16 engine vs BF16 ckpt");
    expect(!kv_checkpoint_runtime_matches(ckpt_fp32_bogus, id_fp16), "runtime reject FP16 engine vs FP32(1) ckpt");

    expect(kv_checkpoint_auto_resume_eligible(ckpt_bf16, id_bf16, prod_bf16, 256, 2048, false),
           "auto-resume eligible same-BF16 (different max_ctx_at_save=8192 vs runtime max_ctx=2048)");
    expect(kv_checkpoint_auto_resume_eligible(ckpt_fp16, id_fp16, prod_fp16, 256, 2048, false),
           "auto-resume eligible same-FP16");
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_fp16, id_bf16, prod_bf16, 256, 2048, false),
           "auto-resume declines BF16 engine vs FP16 ckpt");
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_bf16, id_fp16, prod_fp16, 256, 2048, false),
           "auto-resume declines FP16 engine vs BF16 ckpt");
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_fp32_bogus, id_fp16, prod_fp16, 256, 2048, false),
           "auto-resume declines FP16 engine vs raw FP32(1) ckpt");

    // Legacy checkpoint without verified producer trailer MUST be declined for auto-resume
    KvCheckpointInfo ckpt_fp16_legacy = ckpt_fp16;
    ckpt_fp16_legacy.has_producer_identity = false;
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_fp16_legacy, id_fp16, prod_fp16, 256, 2048, false),
           "auto-resume declines legacy config.json-only checkpoint (no verified producer identity)");

    // Verified checkpoint with mismatched producer (different weights/dense_fp8/prefill_chunk) MUST be declined
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_fp16, id_fp16, compute_disk_producer_sha256(id_fp16_densebf),
                                               256, 2048, false),
           "auto-resume declines mismatched producer dense_fp8");
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_fp16, id_fp16, compute_disk_producer_sha256(id_fp16_chunk),
                                               256, 2048, false),
           "auto-resume declines mismatched producer prefill_chunk");

    // Additional eligibility guards on same-FP16
    KvCheckpointInfo ckpt_fp16_q4 = ckpt_fp16;
    ckpt_fp16_q4.kv_format = 2u;
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_fp16_q4, id_fp16, prod_fp16, 256, 2048, false),
           "same-FP16 declines mismatched kv_format");
    KvCheckpointInfo ckpt_fp16_nologits = ckpt_fp16;
    ckpt_fp16_nologits.has_logits = false;
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_fp16_nologits, id_fp16, prod_fp16, 128, 2048, false),
           "same-FP16 declines has_logits=false at exact prompt length");
    expect(kv_checkpoint_auto_resume_eligible(ckpt_fp16_nologits, id_fp16, prod_fp16, 256, 2048, false),
           "same-FP16 allows has_logits=false for strict prefix");
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_fp16, id_fp16, prod_fp16, 256, 2048, true),
           "same-FP16 declines prefix containing media");
    StateIdentity id_fp16_spec = id_fp16;
    id_fp16_spec.spec_enabled = true;
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_fp16, id_fp16_spec, prod_fp16, 256, 2048, false),
           "same-FP16 declines when spec_enabled");

    // #829 + #827 combined kv_cache_mode & INGCKPT1 byte-144/byte-192 disambiguation checks:
    StateIdentity id_bf16_lf = id_bf16;
    id_bf16_lf.kv_cache_mode = 1u;
    const auto prod_bf16_lf = compute_disk_producer_sha256(id_bf16_lf);
    expect(prod_bf16_lf != prod_bf16, "producer SHA distinguishes per_pass vs last_full");

    KvCheckpointInfo ckpt_bf16_lf = ckpt_bf16;
    ckpt_bf16_lf.kv_cache_mode = KvCacheMode::LastFull;
    ckpt_bf16_lf.num_full_slots = NUM_FULL_SLOTS_LAST_FULL;
    ckpt_bf16_lf.producer_sha256 = prod_bf16_lf;
    expect(kv_checkpoint_runtime_matches(ckpt_bf16_lf, id_bf16_lf), "runtime match same-last_full");
    expect(!kv_checkpoint_runtime_matches(ckpt_bf16_lf, id_bf16), "runtime reject per_pass engine vs last_full ckpt");
    expect(!kv_checkpoint_runtime_matches(ckpt_bf16, id_bf16_lf), "runtime reject last_full engine vs per_pass ckpt");
    expect(kv_checkpoint_auto_resume_eligible(ckpt_bf16_lf, id_bf16_lf, prod_bf16_lf, 256, 2048, false),
           "auto-resume eligible same-last_full");
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_bf16_lf, id_bf16, prod_bf16, 256, 2048, false),
           "auto-resume declines per_pass engine vs last_full ckpt");
    expect(!kv_checkpoint_auto_resume_eligible(ckpt_bf16, id_bf16_lf, prod_bf16_lf, 256, 2048, false),
           "auto-resume declines last_full engine vs per_pass ckpt");

    // Binary header parser tests covering all 4 INGCKPT1 header layouts + corrupt cases:
    {
      const std::filesystem::path hdr_dir = std::filesystem::temp_directory_path() / "ingine_test_ckpt_hdr_layouts";
      std::error_code hec;
      std::filesystem::remove_all(hdr_dir, hec);
      std::filesystem::create_directories(hdr_dir);
      auto write_hdr_file = [&](const std::string& fname, std::int32_t nfull, std::uint32_t b144,
                                bool fill_producer_tail, std::uint32_t b192, std::uint32_t b196) {
        const std::filesystem::path p = hdr_dir / fname;
        std::vector<std::uint8_t> page(4096, 0);
        std::memcpy(page.data(), "INGCKPT1", 8);
        const std::uint32_t ver = 1u, hbytes = 4096u, adt = kDiskBf16, kvf = 1u, has_log = 0u;
        const std::int32_t mctx = 2048, pos = 4, nsl = NUM_SLIDING_VISITS, win = WINDOW, nv = NUM_VISITS, ck = CONV_K;
        const std::uint64_t row_b = 1024u, swa_b = 16u;
        const std::uint64_t full_b = 2u * std::uint64_t(nfull) * std::uint64_t(pos) * row_b;
        std::memcpy(page.data() + 8, &ver, 4);
        std::memcpy(page.data() + 12, &hbytes, 4);
        std::memcpy(page.data() + 16, &adt, 4);
        std::memcpy(page.data() + 20, &kvf, 4);
        std::memcpy(page.data() + 24, &mctx, 4);
        std::memcpy(page.data() + 28, &pos, 4);
        std::memcpy(page.data() + 32, &nfull, 4);
        std::memcpy(page.data() + 36, &nsl, 4);
        std::memcpy(page.data() + 40, &win, 4);
        std::memcpy(page.data() + 44, &nv, 4);
        std::memcpy(page.data() + 48, &ck, 4);
        std::memcpy(page.data() + 52, &has_log, 4);
        std::memcpy(page.data() + 56, &row_b, 8);
        std::memcpy(page.data() + 64, &swa_b, 8);
        std::memcpy(page.data() + 72, &full_b, 8);
        std::memcpy(page.data() + 144, &b144, 4);
        if (fill_producer_tail) {
          const std::uint32_t dfp8 = 1u;
          const std::int32_t chunk = 3072;
          std::memcpy(page.data() + 148, &dfp8, 4);
          std::memcpy(page.data() + 152, &chunk, 4);
          std::memcpy(page.data() + 160, prod_bf16.data(), 32);
        }
        std::memcpy(page.data() + 192, &b192, 4);
        std::memcpy(page.data() + 196, &b196, 4);
        std::vector<std::uint8_t> payload(std::size_t(swa_b + full_b), 0);
        std::ofstream out(p, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(page.data()), std::streamsize(page.size()));
        out.write(reinterpret_cast<const char*>(payload.data()), std::streamsize(payload.size()));
        return p;
      };
      // (A) Legacy per_pass (8 slots, bytes 144..4095 zeroed)
      const auto info_leg = inspect_kv_checkpoint(write_hdr_file("legacy_pp.bin", 8, 0u, false, 0u, 0u));
      expect(info_leg.kv_cache_mode == KvCacheMode::PerPass && info_leg.num_full_slots == 8 &&
                 !info_leg.has_producer_identity,
             "INGCKPT1 legacy per_pass parsed without producer identity");
      // (B) #827 producer-tagged per_pass (8 slots, byte 144=1 + trailer, bytes 192..199 zeroed)
      const auto info_827 = inspect_kv_checkpoint(write_hdr_file("prod_827_pp.bin", 8, 1u, true, 0u, 0u));
      expect(info_827.kv_cache_mode == KvCacheMode::PerPass && info_827.num_full_slots == 8 &&
                 info_827.has_producer_identity && info_827.producer_sha256 == prod_bf16,
             "INGCKPT1 #827 producer-tagged per_pass retains producer identity");
      // (C) Fresh combined #827+#829 per_pass & last_full (byte 192=1, byte 196=0|1, producer trailer at 144..191)
      const auto info_c_pp = inspect_kv_checkpoint(write_hdr_file("comb_pp.bin", 8, 1u, true, 1u, 0u));
      const auto info_c_lf = inspect_kv_checkpoint(write_hdr_file("comb_lf.bin", 4, 1u, true, 1u, 1u));
      expect(info_c_pp.kv_cache_mode == KvCacheMode::PerPass && info_c_pp.num_full_slots == 8 &&
                 info_c_pp.has_producer_identity,
             "INGCKPT1 fresh combined per_pass parsed with producer identity");
      expect(info_c_lf.kv_cache_mode == KvCacheMode::LastFull && info_c_lf.num_full_slots == 4 &&
                 info_c_lf.has_producer_identity,
             "INGCKPT1 fresh combined last_full parsed with producer identity");
      // (D) Standalone #829 4-slot file (4 slots, byte 144=1, bytes 148..199 zeroed) MUST NOT get producer identity!
      const auto info_s829 = inspect_kv_checkpoint(write_hdr_file("standalone_829_4slot.bin", 4, 1u, false, 0u, 0u));
      expect(info_s829.kv_cache_mode == KvCacheMode::LastFull && info_s829.num_full_slots == 4 &&
                 !info_s829.has_producer_identity,
             "INGCKPT1 standalone #829 4-slot file never acquires producer identity from byte 144");
      // Fail-closed corrupt/ambiguous combinations:
      auto expect_inspect_throws = [&](const std::filesystem::path& p, const char* msg) {
        bool threw = false;
        try {
          (void)inspect_kv_checkpoint(p);
        } catch (const std::runtime_error&) {
          threw = true;
        }
        expect(threw, msg);
      };
      expect_inspect_throws(write_hdr_file("bad_4slot_zero144.bin", 4, 0u, false, 0u, 0u),
                            "rejects ambiguous 4-slot legacy header with byte 144 == 0");
      expect_inspect_throws(write_hdr_file("bad_4slot_tail.bin", 4, 1u, true, 0u, 0u),
                            "rejects ambiguous 4-slot header with byte 192 == 0 and non-zero producer tail");
      expect_inspect_throws(write_hdr_file("bad_8slot_emptytail.bin", 8, 1u, false, 0u, 0u),
                            "rejects 8-slot header with producer_flags=1 and empty producer tail");
      expect_inspect_throws(write_hdr_file("bad_comb_mismatch.bin", 8, 0u, false, 1u, 1u),
                            "rejects combined header with last_full(1) and num_full_visits=8");
      std::filesystem::remove_all(hdr_dir, hec);
    }

    // Model-package fingerprint binds 100% of actual weight bytes (no reliance on optional SHA256SUMS text):
    // two valid same-layout safetensors with identical config.json/index/headers and weights 1.0 vs 2.0 MUST differ,
    // and empty/weightless directories fail closed to std::nullopt (unverified).
    const std::filesystem::path pkg_base =
        std::filesystem::temp_directory_path() / "ingine_test_model_pkg_sha";
    std::error_code ec;
    std::filesystem::remove_all(pkg_base, ec);
    const auto pkg_rev1 = pkg_base / "rev1_pkg";
    const auto pkg_rev2 = pkg_base / "rev2_pkg";
    const auto pkg_empty = pkg_base / "no_weights_pkg";
    std::filesystem::create_directories(pkg_rev1);
    std::filesystem::create_directories(pkg_rev2);
    std::filesystem::create_directories(pkg_empty);
    auto write_str = [](const std::filesystem::path& p, const std::string& s) {
      std::ofstream out(p, std::ios::binary | std::ios::trunc);
      out.write(s.data(), std::streamsize(s.size()));
    };
    auto write_single_f32_safetensors = [](const std::filesystem::path& p, float val) {
      const std::string hdr = R"({"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
      const std::uint64_t hlen = std::uint64_t(hdr.size());
      std::ofstream out(p, std::ios::binary | std::ios::trunc);
      out.write(reinterpret_cast<const char*>(&hlen), sizeof(hlen));
      out.write(hdr.data(), std::streamsize(hdr.size()));
      out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    };
    const std::string shared_cfg = R"({"architectures":["Inkling2ForCausalLM"],"hidden_size":2304})";
    const std::string shared_idx = R"({"weight_map":{"w":"model-00001-of-00001.safetensors"}})";
    write_str(pkg_rev1 / "config.json", shared_cfg);
    write_str(pkg_rev2 / "config.json", shared_cfg);
    write_str(pkg_empty / "config.json", shared_cfg);
    write_str(pkg_rev1 / "model.safetensors.index.json", shared_idx);
    write_str(pkg_rev2 / "model.safetensors.index.json", shared_idx);
    write_single_f32_safetensors(pkg_rev1 / "model-00001-of-00001.safetensors", 1.0f);
    write_single_f32_safetensors(pkg_rev2 / "model-00001-of-00001.safetensors", 2.0f);

    const auto sha_rev1 = compute_model_package_sha256_opt(pkg_rev1);
    const auto sha_rev2 = compute_model_package_sha256_opt(pkg_rev2);
    expect(sha_rev1.has_value() && sha_rev2.has_value(), "valid safetensors packages produce verified content SHA");
    expect(*sha_rev1 != *sha_rev2,
           "same-layout safetensors with weights 1.0 vs 2.0 (no SHA256SUMS) produce distinct package SHAs");
    StateIdentity id_rev1 = id_fp16, id_rev2 = id_fp16;
    id_rev1.model_sha256 = *sha_rev1;
    id_rev2.model_sha256 = *sha_rev2;
    expect(compute_disk_producer_sha256(id_rev1) != compute_disk_producer_sha256(id_rev2),
           "same-layout safetensors with weights 1.0 vs 2.0 produce distinct disk producer SHAs");
    expect(!compute_model_package_sha256_opt({}).has_value() && !compute_model_package_sha256_opt(pkg_empty).has_value(),
           "empty or weightless model directories fail closed to std::nullopt (unverified)");

    // Verify load_weights-style tensor digest assembly matches standalone compute_model_package_sha256_opt bit-for-bit
    const float v1 = 1.0f;
    const std::array<std::uint8_t, 32> csha = sha256_bytes(&v1, sizeof(v1));
    const std::array<std::int64_t, 1> sh1 = {1};
    const std::array<std::array<std::uint8_t, 32>, 1> cshas = {csha};
    std::map<std::string, std::array<std::uint8_t, 32>> tmap;
    tmap["w"] = sha256_tensor_from_chunk_digests("w", "F32", sh1, sizeof(v1), cshas);
    expect(compute_model_package_sha256_from_tensors(pkg_rev1, tmap) == *sha_rev1,
           "in-flight tensor digest assembly matches standalone compute_model_package_sha256_opt bit-for-bit");

    // Consumed-weights digest (used by load_weights with zero post-load file re-reads) distinguishes value changes
    const float v2 = 2.0f;
    const std::array<std::array<std::uint8_t, 32>, 1> cshas2 = {sha256_bytes(&v2, sizeof(v2))};
    std::map<std::string, std::array<std::uint8_t, 32>> tmap2;
    tmap2["w"] = sha256_tensor_from_chunk_digests("w", "F32", sh1, sizeof(v2), cshas2);
    const auto cfg_sha = sha256_file_or_zero(pkg_rev1 / "config.json");
    const auto q_sha = sha256_bytes("quant", 5);
    expect(compute_consumed_weights_sha256(cfg_sha, q_sha, tmap) !=
               compute_consumed_weights_sha256(cfg_sha, q_sha, tmap2),
           "compute_consumed_weights_sha256 distinguishes consumed tensor byte changes");

    // Pack-only, same-source mixed, and changed-source mixed package SHA-256 checks:
    const auto pkg_pack1 = pkg_base / "pack_only_rev1";
    const auto pkg_pack2 = pkg_base / "pack_only_rev2";
    std::filesystem::create_directories(pkg_pack1);
    std::filesystem::create_directories(pkg_pack2);
    write_str(pkg_pack1 / "config.json", shared_cfg);
    write_str(pkg_pack2 / "config.json", shared_cfg);
    (void)ingine::WeightPackBuilder::pack_from_safetensors(pkg_rev1, pkg_pack1 / "model.ingpack");
    (void)ingine::WeightPackBuilder::pack_from_safetensors(pkg_rev2, pkg_pack2 / "model.ingpack");

    const auto sha_pack1 = compute_model_package_sha256_opt(pkg_pack1);
    const auto sha_pack2 = compute_model_package_sha256_opt(pkg_pack2);
    expect(sha_pack1.has_value() && sha_pack2.has_value(),
           "pack-only directories with model.ingpack produce verified content SHA");
    expect(*sha_pack1 != *sha_pack2,
           "pack-only directories with weights 1.0 vs 2.0 produce distinct package SHAs");
    expect(compute_model_package_sha256_from_tensors(pkg_pack1, tmap) == *sha_pack1,
           "pack-only compute_model_package_sha256_opt matches compute_model_package_sha256_from_tensors");

    // Same-source mixed dir: model.ingpack built from pkg_rev1 matches pkg_rev1 safetensors bit-for-bit.
    (void)ingine::WeightPackBuilder::pack_from_safetensors(pkg_rev1, pkg_rev1 / "model.ingpack");
    expect(ingine::SafetensorsModel(pkg_rev1).is_weight_pack(),
           "same-source mixed directory selects verified model.ingpack");
    const auto sha_mixed_same = compute_model_package_sha256_opt(pkg_rev1);
    expect(sha_mixed_same.has_value() && *sha_mixed_same == *sha_rev1,
           "same-source mixed directory package SHA matches safetensors-only baseline bit-for-bit");

    // Changed-source mixed dir: copying rev1 pack next to rev2 shards rejects stale pack and hashes rev2.
    std::filesystem::copy_file(pkg_rev1 / "model.ingpack", pkg_rev2 / "model.ingpack",
                               std::filesystem::copy_options::overwrite_existing);
    expect(!ingine::SafetensorsModel(pkg_rev2).is_weight_pack(),
           "changed-source mixed directory rejects stale model.ingpack");
    const auto sha_mixed_changed = compute_model_package_sha256_opt(pkg_rev2);
    expect(sha_mixed_changed.has_value() && *sha_mixed_changed == *sha_rev2 && *sha_mixed_changed != *sha_rev1,
           "changed-source mixed directory package SHA hashes updated safetensors, not stale pack");

    std::filesystem::remove_all(pkg_base, ec);
  }
#endif

  // #829 opt-in L11/L17 last_full KV sharing invariants (CPU-verifiable):
  expect(NUM_FULL_SLOTS_LAST_FULL == 4, "last_full has 4 physical full slots");
  expect(full_slot_count(false) == 8 && full_slot_count(true) == 4, "full_slot_count");
  for (int i = 0; i < NUM_VISITS; ++i) {
    const int l = VISITS[i].layer;
    const int expected_lf = (l == 5) ? 0 : (l == 11) ? 1 : (l == 17) ? 2 : (l == 23) ? 3 : -1;
    expect(FULL_INDEX_LAST_FULL[i] == expected_lf, "FULL_INDEX_LAST_FULL mapping");
    expect(full_slot_index(i, false) == FULL_INDEX[i], "full_slot_index per_pass");
    expect(full_slot_index(i, true) == expected_lf, "full_slot_index last_full");
  }
  // Verify exact pass sharing on L11 (visits 11, 23, 35 -> slot 1) and L17 (visits 17, 29, 41 -> slot 2).
  expect(FULL_INDEX_LAST_FULL[5] == 0 && FULL_INDEX_LAST_FULL[47] == 3, "L5=0 and L23=3");
  expect(FULL_INDEX_LAST_FULL[11] == 1 && FULL_INDEX_LAST_FULL[23] == 1 && FULL_INDEX_LAST_FULL[35] == 1,
         "L11 passes 0,1,2 share slot 1");
  expect(FULL_INDEX_LAST_FULL[17] == 2 && FULL_INDEX_LAST_FULL[29] == 2 && FULL_INDEX_LAST_FULL[41] == 2,
         "L17 passes 0,1,2 share slot 2");
  // Verify projected 1M-token Q4_0 full-KV buffer size: 9.00 GiB -> 4.50 GiB (exact 50% of full-KV buffer).
  const ingine::Tensor kq4_lf = rm(nullptr, ingine::DType::U8, {NUM_FULL_SLOTS_LAST_FULL, kOneM, KV_Q4_ROW_BYTES});
  expect(kq4_lf.bytes() == 2415919104ull, "1M q4_0 last_full k_full == 2.25 GiB");
  expect(kq4_lf.bytes() * 2 == 4831838208ull, "1M q4_0 last_full k+v == 4.50 GiB");
  // State helper methods match mode.
  State st_pp;
  st_pp.kv_last_full = false;
  State st_lf;
  st_lf.kv_last_full = true;
  expect(st_pp.num_full_slots() == 8 && st_lf.num_full_slots() == 4, "State::num_full_slots");
  expect(st_pp.full_slot(35) == 5 && st_lf.full_slot(35) == 1, "State::full_slot L11 pass 2");
  expect(st_pp.full_slot(41) == 6 && st_lf.full_slot(41) == 2, "State::full_slot L17 pass 2");
  // KvCacheMode string parsing & default.
  RuntimeOptions default_opts;
  expect(default_opts.kv_cache_mode == KvCacheMode::PerPass, "RuntimeOptions defaults to per_pass");
  expect(std::strcmp(to_string(KvCacheMode::PerPass), "per_pass") == 0, "to_string per_pass");
  expect(std::strcmp(to_string(KvCacheMode::LastFull), "last_full") == 0, "to_string last_full");
  expect(parse_kv_cache_mode("per_pass") == KvCacheMode::PerPass, "parse per_pass");
  expect(parse_kv_cache_mode("last_full") == KvCacheMode::LastFull, "parse last_full");
  bool threw_bad_mode = false;
  try {
    (void)parse_kv_cache_mode("invalid_mode");
  } catch (const std::invalid_argument&) {
    threw_bad_mode = true;
  }
  expect(threw_bad_mode, "parse_kv_cache_mode rejects unknown string");

  // Variable loop depth VisitPlan (L in 1..MAX_PASSES)
  {
    StateIdentity lp3;
    lp3.loop_passes = 3;
    StateIdentity lp5 = lp3;
    lp5.loop_passes = 5;
    expect(!(lp3 == lp5) && hash_state_identity(lp3) != hash_state_identity(lp5),
           "StateIdentity loop_passes isolation");
#if INGINE_BACKEND_CUDA
    expect(compute_disk_producer_sha256(lp3) != compute_disk_producer_sha256(lp5),
           "producer SHA distinguishes loop_passes 3 vs 5");
    KvCheckpointInfo info_l5;
    info_l5.act_dtype = static_cast<std::uint32_t>(ingine::DType::BF16);
    info_l5.kv_format = 0u;
    info_l5.kv_cache_mode = KvCacheMode::PerPass;
    info_l5.loop_passes = 5;
    info_l5.num_full_slots = 12;
    expect(kv_checkpoint_runtime_matches(info_l5, lp5), "runtime match same loop_passes=5");
    expect(!kv_checkpoint_runtime_matches(info_l5, lp3), "runtime reject loop_passes=5 ckpt on L=3 engine");
#endif
  }
  for (int L = 1; L <= MAX_PASSES; ++L) {
    const VisitPlan plan = make_visit_plan(L);
    expect(plan.passes == L && plan.last_pass() == L - 1, "VisitPlan passes/last_pass");
    expect(plan.num_visits == 12 + 12 * L, "VisitPlan num_visits == 12 + 12*L");
    expect(plan.num_full_visits == 2 + 2 * L, "VisitPlan num_full_visits == 2 + 2*L");
    expect(plan.num_sliding_visits == 10 + 10 * L, "VisitPlan num_sliding_visits == 10 + 10*L");
    expect(int(plan.visits.size()) == plan.num_visits, "VisitPlan visits size");
    int vf = 0, vs = 0;
    for (int i = 0; i < plan.num_visits; ++i) {
      const bool full = is_full_layer(plan.visits[std::size_t(i)].layer);
      expect(full ? (plan.full_index[std::size_t(i)] == vf++ && plan.ring_index[std::size_t(i)] == -1)
                  : (plan.ring_index[std::size_t(i)] == vs++ && plan.full_index[std::size_t(i)] == -1),
             "VisitPlan index partitioning");
    }
    expect(vf == plan.num_full_visits && vs == plan.num_sliding_visits, "VisitPlan index counts");
    if (L == PASSES) {
      expect(plan.num_visits == NUM_VISITS && plan.num_full_visits == NUM_FULL_VISITS &&
                 plan.num_sliding_visits == NUM_SLIDING_VISITS,
             "VisitPlan L=3 matches compile-time counts");
      for (int i = 0; i < NUM_VISITS; ++i) {
        expect(plan.visits[std::size_t(i)].layer == VISITS[std::size_t(i)].layer &&
                   plan.visits[std::size_t(i)].pass == VISITS[std::size_t(i)].pass &&
                   plan.full_index[std::size_t(i)] == FULL_INDEX[std::size_t(i)] &&
                   plan.ring_index[std::size_t(i)] == RING_INDEX[std::size_t(i)],
               "VisitPlan L=3 exact parity with VISITS/FULL_INDEX/RING_INDEX");
      }
    }
  }
  auto expect_throws = [&](int bad_L, const char* what) {
    bool threw = false;
    try {
      (void)make_visit_plan(bad_L);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    expect(threw, what);
  };
  expect_throws(0, "VisitPlan rejects L=0");
  expect_throws(-1, "VisitPlan rejects L=-1");
  expect_throws(MAX_PASSES + 1, "VisitPlan rejects L=MAX_PASSES+1");

  // Offload prefill VRAM budgeting tracks pass-dependent B2[p]/E[p] buffers (18 KiB/tok per pass)
  constexpr int kChunk = 1024;
  const std::size_t prefill_l3 = offload_prefill_vram_bytes(kChunk, 3);
  const std::size_t prefill_l8 = offload_prefill_vram_bytes(kChunk, 8);
  const std::size_t prefill_l1 = offload_prefill_vram_bytes(kChunk, 1);
  expect(prefill_l3 == std::size_t(kChunk) * 280 * 1024 + (std::size_t(48) << 20),
         "offload_prefill_vram_bytes L=3 matches baseline");
  expect(prefill_l8 - prefill_l3 == (std::size_t(90) << 20),
         "offload_prefill_vram_bytes L=8 adds +90 MiB vs L=3 at chunk=1024");
  expect(prefill_l3 - prefill_l1 == (std::size_t(36) << 20),
         "offload_prefill_vram_bytes L=1 saves 36 MiB vs L=3 at chunk=1024");

  std::printf("%s model_data constants\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// model_data.h constants against weights.py / model.py / fast.py (CPU only): the visit plan, ring/full cache
// indices, tap visits and the projection layout.
#include "core/act.h"
#include "ingine/generator.h"
#if INGINE_BACKEND_CUDA
#include "model/kv_checkpoint.h"
#endif
#include "model/model_data.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
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
    std::filesystem::remove_all(pkg_base, ec);
  }
#endif

  std::printf("%s model_data constants\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// Paired #829 native evaluator: per_pass (default, 8 full slots) vs opt-in L11/L17 last_full (4 full slots).
//
// Runs both arms sequentially on the same loaded Weights and same physical GPU (B=1, 3 passes, no spec, cold
// prefix) and measures:
//   1. Exact full-KV / SWA-ring bytes and CUDA VRAM deltas across context lengths (64K, 128K, 262K, 1M).
//   2. Multi-chunk prefill (e.g. chunk=512 and chunk=2048) + teacher-forced decode past the 1024-token SWA window:
//      - Next-token cross-entropy loss (NLL), perplexity, relative NLL delta vs per_pass.
//      - Full-vocab KL(per_pass || last_full) mean/p50/p95/max and top-1 agreement (overall, <=1024 SWA, >1024 full).
//      - Exact prefill wall time, decode wall time, tok/s, peak allocated VRAM, and GPU-seconds.
//   3. Disk KV checkpoint mode-identity isolation (per_pass <-> last_full cross-load fails loudly).
#include "core/device.h"
#include "ingine/generator.h"
#include "model/kv_checkpoint.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "text/chat.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace ingine;
using namespace ingine::model;
using Json = nlohmann::ordered_json;

namespace {

std::vector<float> download_f32(const float* d, std::size_t n) {
  std::vector<float> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}

double log_sum_exp(const std::vector<float>& logits, double& max_v) {
  max_v = double(*std::max_element(logits.begin(), logits.end()));
  double z = 0.0;
  for (float x : logits) z += std::exp(double(x) - max_v);
  return max_v + std::log(z);
}

double token_nll(const std::vector<float>& logits, std::int32_t target_id) {
  if (target_id < 0 || std::size_t(target_id) >= logits.size()) {
    throw std::out_of_range("target_id out of vocab range");
  }
  double mv = 0.0;
  const double lse = log_sum_exp(logits, mv);
  return lse - double(logits[std::size_t(target_id)]);
}

double kl_div(const std::vector<float>& ref, const std::vector<float>& test) {
  double ma = 0.0, mb = 0.0;
  const double lsea = log_sum_exp(ref, ma);
  const double lseb = log_sum_exp(test, mb);
  double k = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double la = double(ref[i]) - lsea;
    const double lb = double(test[i]) - lseb;
    k += std::exp(la) * (la - lb);
  }
  return k;
}

int argmax(const std::vector<float>& a) {
  return int(std::max_element(a.begin(), a.end()) - a.begin());
}

std::string sha256_hex(const std::array<std::uint8_t, 32>& h) {
  char out[65];
  for (int i = 0; i < 32; ++i) std::snprintf(out + 2 * i, 3, "%02x", unsigned(h[i]));
  out[64] = '\0';
  return std::string(out);
}

std::string format_uuid(const cudaUUID_t& u) {
  char buf[64];
  const auto* b = reinterpret_cast<const unsigned char*>(u.bytes);
  std::snprintf(buf, sizeof(buf),
                "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
  return std::string(buf);
}

std::vector<int> parse_csv_ints(const std::string& s) {
  std::vector<int> out;
  std::size_t pos = 0;
  while (pos < s.size()) {
    const std::size_t c = s.find(',', pos);
    const std::string tok = s.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
    if (!tok.empty()) out.push_back(std::stoi(tok));
    if (c == std::string::npos) break;
    pos = c + 1;
  }
  return out;
}

std::vector<std::int32_t> build_eval_tokens(const std::string& model_dir, const std::string& prompt_json,
                                            const std::string& text_file, int min_tokens) {
  if (!prompt_json.empty()) {
    std::ifstream in(prompt_json);
    if (!in) throw std::runtime_error("cannot open prompt json: " + prompt_json);
    const auto j = nlohmann::json::parse(in);
    auto ids = j.at("prompt_ids").get<std::vector<std::int32_t>>();
    if (int(ids.size()) < min_tokens) {
      throw std::runtime_error("prompt_json has " + std::to_string(ids.size()) + " tokens, need " +
                               std::to_string(min_tokens));
    }
    return ids;
  }
  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  const auto& tok = fmt.tokenizer();
  std::string corpus;
  if (!text_file.empty()) {
    std::ifstream in(text_file);
    if (!in) throw std::runtime_error("cannot open text file: " + text_file);
    std::ostringstream ss;
    ss << in.rdbuf();
    corpus = ss.str();
  } else {
    // Deterministic multi-domain technical/coding corpus repeated with unique section counters so
    // attention keys across chunks are distinct (never identical degenerate copies).
    std::ostringstream ss;
    for (int i = 0; i < 400; ++i) {
      ss << "### Section " << i << " (module_id=MOD-" << (1000 + i * 7) << ")\n"
         << "In systems verification run #" << i << ", worker node W" << (i % 16)
         << " processes batch checksum 0x" << std::hex << (0x9e3779b9u ^ (unsigned(i) * 2654435761u)) << std::dec
         << " under memory budget " << (4096 + (i % 11) * 512) << " MiB.\n"
         << "def reconcile_ledger_" << i << "(entries, threshold=" << (10 + (i % 23)) << "):\n"
         << "    total = 0\n"
         << "    for idx, val in enumerate(entries):\n"
         << "        if (val + idx) % " << (3 + (i % 7)) << " == 0:\n"
         << "            total += val * " << (2 + (i % 5)) << "\n"
         << "    return total >= threshold\n\n";
    }
    corpus = ss.str();
  }
  std::vector<std::int32_t> ids = tok.encode(corpus, /*special=*/false);
  if (int(ids.size()) < min_tokens) {
    throw std::runtime_error("tokenized text has " + std::to_string(ids.size()) + " tokens, need " +
                             std::to_string(min_tokens));
  }
  return ids;
}

struct ArmOutput {
  KvCacheMode mode = KvCacheMode::PerPass;
  int prefill_chunk = 0;
  int prefix_len = 0;
  int decode_len = 0;
  std::uint64_t full_kv_bytes = 0;
  std::uint64_t swa_ring_bytes = 0;
  std::uint64_t cuda_state_delta_bytes = 0;
  double prefill_seconds = 0.0;
  double decode_seconds = 0.0;
  double mean_nll = 0.0;
  double mean_nll_within_swa = 0.0;
  double mean_nll_beyond_swa = 0.0;
  double ppl = 0.0;
  std::vector<std::vector<float>> logits;  // [decode_len] logits predicting ids[prefix_len + i]
};

Json run_vram_table(bool kv_fp8, bool kv_q4, const std::vector<int>& contexts) {
  Json rows = Json::array();
  for (int ctx : contexts) {
    for (KvCacheMode mode : {KvCacheMode::PerPass, KvCacheMode::LastFull}) {
      const bool lf = (mode == KvCacheMode::LastFull);
      const std::uint64_t row_b = kv_q4 ? std::uint64_t(KV_Q4_ROW_BYTES) : std::uint64_t(KV_DIM) * (kv_fp8 ? 1u : 2u);
      const std::uint64_t expected_full = 2u * std::uint64_t(full_slot_count(lf)) * std::uint64_t(ctx) * row_b;
      const std::uint64_t expected_ring = 2u * std::uint64_t(NUM_SLIDING_VISITS) * std::uint64_t(WINDOW) *
                                          std::uint64_t(KV_DIM) * 2u;
      std::size_t f0 = 0, tot = 0;
      CUDA_CHECK(cudaMemGetInfo(&f0, &tot));
      Json row{{"ctx", ctx},
               {"kv_cache_mode", to_string(mode)},
               {"full_slots", full_slot_count(lf)},
               {"expected_full_kv_bytes", expected_full},
               {"expected_swa_ring_bytes", expected_ring}};
      // Only allocate if free VRAM has >= 512 MiB headroom above expected full+ring+conv state.
      const std::uint64_t need = expected_full + expected_ring + (std::uint64_t(512) << 20);
      if (f0 >= need) {
        State st = make_state(ctx, kv_fp8, kv_q4, {}, lf);
        std::size_t f1 = 0;
        CUDA_CHECK(cudaMemGetInfo(&f1, &tot));
        const std::uint64_t actual_full = st.k_full.bytes() + st.v_full.bytes();
        const std::uint64_t actual_ring = st.k_ring.bytes() + st.v_ring.bytes();
        const std::uint64_t delta = (f0 > f1) ? std::uint64_t(f0 - f1) : 0u;
        row["allocated"] = true;
        row["actual_full_kv_bytes"] = actual_full;
        row["actual_swa_ring_bytes"] = actual_ring;
        row["cuda_mem_delta_bytes"] = delta;
      } else {
        row["allocated"] = false;
        row["skip_reason"] = "insufficient_free_vram_for_resident_alloc";
        row["free_vram_bytes"] = std::uint64_t(f0);
      }
      rows.push_back(std::move(row));
    }
  }
  return rows;
}

Json run_ckpt_isolation_test(const Weights& w, const Ablation& abl, const std::string& model_dir, bool kv_fp8,
                             bool kv_q4, const std::vector<std::int32_t>& ids, const DeviceBuffer& d_ids,
                             DeviceBuffer& d_logits, DeviceBuffer& d_pos, Buffers& buf1, cudaStream_t stream) {
  const int ckpt_prefix = 128;
  const int ckpt_steps = 16;
  std::span<const std::int32_t> prefix(ids.data(), ckpt_prefix);
  const std::filesystem::path dir = std::filesystem::temp_directory_path();
  const std::filesystem::path p_pp = dir / ("eval_kv_lf_pp_" + std::to_string(int(getpid())) + ".ckpt");
  const std::filesystem::path p_lf = dir / ("eval_kv_lf_lf_" + std::to_string(int(getpid())) + ".ckpt");
  bool pp_into_lf_rejected = false;
  bool lf_into_pp_rejected = false;
  bool pp_roundtrip_ok = false;
  bool lf_roundtrip_ok = false;
  bool pp_populated_exact = false;
  bool lf_populated_exact = false;
  int pp_tok_matches = 0, lf_tok_matches = 0;
  int pp_logit_exact_steps = 0, lf_logit_exact_steps = 0;
  double pp_max_abs = 0.0, lf_max_abs = 0.0;
  double pp_max_kl = 0.0, lf_max_kl = 0.0;
  std::string pp_err, lf_err;
  const int ctx = ckpt_prefix + ckpt_steps + 32;

  auto run_populated_mode = [&](bool last_full, const std::filesystem::path& ckpt_path, bool& roundtrip_ok,
                                bool& populated_exact, int& tok_matches, int& logit_exact_steps, double& max_abs_out,
                                double& max_kl_out) -> State {
    Prefiller pre(w, abl, ckpt_prefix, stream);
    State st_orig = make_state(ctx, kv_fp8, kv_q4, {}, last_full);
    reset_state(st_orig, stream);
    pre.run(st_orig, static_cast<const std::int32_t*>(d_ids.p), ckpt_prefix, 0, static_cast<float*>(d_logits.p),
            nullptr);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    save_kv_checkpoint(st_orig, model_dir, prefix, static_cast<const float*>(d_logits.p), ckpt_path, stream);

    // Uninterrupted continuation (step 0 = prefill logits, steps 1..ckpt_steps = decode logits).
    std::vector<std::vector<float>> orig_logits;
    orig_logits.reserve(std::size_t(ckpt_steps + 1));
    orig_logits.push_back(download_f32(static_cast<const float*>(d_logits.p), UNPADDED_VOCAB));
    {
      Decoder dec_orig(w, abl, st_orig, true);
      for (int i = 0; i < ckpt_steps; ++i) {
        const std::int64_t tok = ids[std::size_t(ckpt_prefix + i)];
        const std::int64_t pos = ckpt_prefix + i;
        CUDA_CHECK(cudaMemcpyAsync(buf1.token.data, &tok, sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos.p, &pos, sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
        dec_orig.decode(buf1, static_cast<const std::int64_t*>(d_pos.p), false, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        orig_logits.push_back(download_f32(static_cast<const float*>(buf1.logits.data), UNPADDED_VOCAB));
      }
    }

    // Fresh same-mode state loaded from populated checkpoint.
    State st_res = make_state(ctx, kv_fp8, kv_q4, {}, last_full);
    reset_state(st_res, stream);
    DeviceBuffer d_res_logits(std::size_t(UNPADDED_VOCAB) * sizeof(float));
    CUDA_CHECK(cudaMemsetAsync(d_res_logits.p, 0, d_res_logits.bytes, stream));
    const int loaded =
        load_kv_checkpoint(st_res, model_dir, prefix, static_cast<float*>(d_res_logits.p), ckpt_path, stream);
    roundtrip_ok = (loaded == ckpt_prefix);

    std::vector<std::vector<float>> res_logits;
    res_logits.reserve(std::size_t(ckpt_steps + 1));
    res_logits.push_back(download_f32(static_cast<const float*>(d_res_logits.p), UNPADDED_VOCAB));
    {
      Decoder dec_res(w, abl, st_res, true);
      for (int i = 0; i < ckpt_steps; ++i) {
        const std::int64_t tok = ids[std::size_t(ckpt_prefix + i)];
        const std::int64_t pos = ckpt_prefix + i;
        CUDA_CHECK(cudaMemcpyAsync(buf1.token.data, &tok, sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos.p, &pos, sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
        dec_res.decode(buf1, static_cast<const std::int64_t*>(d_pos.p), false, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        res_logits.push_back(download_f32(static_cast<const float*>(buf1.logits.data), UNPADDED_VOCAB));
      }
    }

    bool all_ok = roundtrip_ok;
    tok_matches = 0;
    logit_exact_steps = 0;
    max_abs_out = 0.0;
    max_kl_out = 0.0;
    for (int s = 0; s <= ckpt_steps; ++s) {
      const auto& a = orig_logits[std::size_t(s)];
      const auto& b = res_logits[std::size_t(s)];
      bool finite = (a.size() == b.size() && !a.empty());
      for (std::size_t v = 0; finite && v < a.size(); ++v) {
        if (!std::isfinite(a[v]) || !std::isfinite(b[v])) finite = false;
        else max_abs_out = std::max(max_abs_out, std::abs(double(a[v]) - double(b[v])));
      }
      const bool bytes_eq = finite && (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
      if (bytes_eq) ++logit_exact_steps;
      if (finite && argmax(a) == argmax(b)) ++tok_matches;
      if (finite) max_kl_out = std::max(max_kl_out, kl_div(a, b));
      if (!finite || !bytes_eq || argmax(a) != argmax(b)) all_ok = false;
    }
    populated_exact = all_ok;
    return st_res;
  };

  try {
    State st_pp = run_populated_mode(false, p_pp, pp_roundtrip_ok, pp_populated_exact, pp_tok_matches,
                                     pp_logit_exact_steps, pp_max_abs, pp_max_kl);
    State st_lf = run_populated_mode(true, p_lf, lf_roundtrip_ok, lf_populated_exact, lf_tok_matches,
                                     lf_logit_exact_steps, lf_max_abs, lf_max_kl);

    const KvCheckpointInfo info_pp = inspect_kv_checkpoint(p_pp);
    const KvCheckpointInfo info_lf = inspect_kv_checkpoint(p_lf);
    if (info_pp.kv_cache_mode != KvCacheMode::PerPass || info_pp.num_full_slots != 8) {
      throw std::runtime_error("inspect_kv_checkpoint per_pass mismatch");
    }
    if (info_lf.kv_cache_mode != KvCacheMode::LastFull || info_lf.num_full_slots != 4) {
      throw std::runtime_error("inspect_kv_checkpoint last_full mismatch");
    }

    try {
      (void)load_kv_checkpoint(st_lf, model_dir, prefix, nullptr, p_pp, stream);
    } catch (const std::exception& e) {
      pp_into_lf_rejected = true;
      pp_err = e.what();
    }
    try {
      (void)load_kv_checkpoint(st_pp, model_dir, prefix, nullptr, p_lf, stream);
    } catch (const std::exception& e) {
      lf_into_pp_rejected = true;
      lf_err = e.what();
    }
  } catch (...) {
    std::error_code ec;
    std::filesystem::remove(p_pp, ec);
    std::filesystem::remove(p_lf, ec);
    throw;
  }
  std::error_code ec;
  std::filesystem::remove(p_pp, ec);
  std::filesystem::remove(p_lf, ec);
  return Json{
      {"passed", pp_roundtrip_ok && lf_roundtrip_ok && pp_populated_exact && lf_populated_exact &&
                     pp_into_lf_rejected && lf_into_pp_rejected},
      {"ckpt_prefix_tokens", ckpt_prefix},
      {"ckpt_verify_steps", ckpt_steps + 1},
      {"per_pass_roundtrip_ok", pp_roundtrip_ok},
      {"last_full_roundtrip_ok", lf_roundtrip_ok},
      {"per_pass_populated_resume_exact", pp_populated_exact},
      {"last_full_populated_resume_exact", lf_populated_exact},
      {"per_pass_tok_matches", pp_tok_matches},
      {"last_full_tok_matches", lf_tok_matches},
      {"per_pass_logit_byte_exact_steps", pp_logit_exact_steps},
      {"last_full_logit_byte_exact_steps", lf_logit_exact_steps},
      {"per_pass_max_abs_logit_diff", pp_max_abs},
      {"last_full_max_abs_logit_diff", lf_max_abs},
      {"per_pass_max_kl_fp64", pp_max_kl},
      {"last_full_max_kl_fp64", lf_max_kl},
      {"per_pass_into_last_full_rejected", pp_into_lf_rejected},
      {"last_full_into_per_pass_rejected", lf_into_pp_rejected},
      {"per_pass_into_last_full_error", pp_err},
      {"last_full_into_per_pass_error", lf_err}};
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir, prompt_json, text_file, out_json;
  std::string kv_format = "q4_0";
  std::string chunks_csv = "512,2048";
  std::string vram_ctxs_csv = "65536,131072,262144,1048576";
  int prefix_len = 4096;
  int decode_len = 1280;  // crosses past WINDOW=1024 so full-attention layers carry >SWA context
  bool dense_fp8 = true;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + flag);
      return argv[++i];
    };
    if (a == "--model-dir") model_dir = next("--model-dir");
    else if (a == "--prompt-json") prompt_json = next("--prompt-json");
    else if (a == "--text-file") text_file = next("--text-file");
    else if (a == "--out-json") out_json = next("--out-json");
    else if (a == "--kv-format") kv_format = next("--kv-format");
    else if (a == "--chunks") chunks_csv = next("--chunks");
    else if (a == "--vram-contexts") vram_ctxs_csv = next("--vram-contexts");
    else if (a == "--prefix-len") prefix_len = std::stoi(next("--prefix-len"));
    else if (a == "--decode-len") decode_len = std::stoi(next("--decode-len"));
    else if (a == "--no-dense-fp8") dense_fp8 = false;
    else if (a == "-h" || a == "--help") {
      std::printf(
          "Usage: eval_kv_last_full --model-dir DIR [--kv-format q4_0|fp8|bf16] [--chunks 512,2048]\n"
          "       [--prefix-len 4096] [--decode-len 1280] [--prompt-json FILE | --text-file FILE]\n"
          "       [--vram-contexts 65536,131072,262144,1048576] [--out-json FILE]\n");
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return 2;
    }
  }
  if (model_dir.empty()) {
    std::fprintf(stderr, "error: --model-dir is required\n");
    return 2;
  }
  const bool kv_fp8 = (kv_format == "fp8");
  const bool kv_q4 = (kv_format == "q4_0");
  if (!kv_fp8 && !kv_q4 && kv_format != "bf16") {
    std::fprintf(stderr, "error: --kv-format must be q4_0, fp8, or bf16\n");
    return 2;
  }

  try {
    cudaFree(nullptr);
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    const std::string gpu_uuid = format_uuid(prop.uuid);

    const std::vector<int> chunks = parse_csv_ints(chunks_csv);
    const std::vector<int> vram_ctxs = parse_csv_ints(vram_ctxs_csv);
    const int need_tokens = prefix_len + decode_len + 1;
    std::vector<std::int32_t> ids = build_eval_tokens(model_dir, prompt_json, text_file, need_tokens);
    ids.resize(std::size_t(need_tokens));
    const std::string prompt_sha = sha256_hex(sha256_bytes(ids.data(), ids.size() * sizeof(std::int32_t)));
    const std::string model_cfg_sha = sha256_hex(sha256_file_or_zero(std::filesystem::path(model_dir) / "config.json"));

    std::printf("[#829] GPU=%s (%s) model=%s kv=%s prefix=%d decode=%d chunks=%s\n", prop.name, gpu_uuid.c_str(),
                model_dir.c_str(), kv_format.c_str(), prefix_len, decode_len, chunks_csv.c_str());
    std::fflush(stdout);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // 1. Standalone VRAM state sizing table before loading weights (so 1M alloc can run when card fits).
    Json vram_table = run_vram_table(kv_fp8, kv_q4, vram_ctxs);

    // 2. Load weights once; run both arms on identical Weights and same physical GPU.
    std::size_t free_before_w = 0, total_vram = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before_w, &total_vram));
    Weights w = load_weights(model_dir, dense_fp8, stream);
    Ablation abl = load_ablation({});
    std::size_t free_after_w = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after_w, &total_vram));

    DeviceBuffer d_ids(ids.size() * sizeof(std::int32_t));
    d_ids.copy_from_host(ids.data(), ids.size() * sizeof(std::int32_t));
    DeviceBuffer d_logits(std::size_t(UNPADDED_VOCAB) * sizeof(float));
    DeviceBuffer d_pos(sizeof(std::int64_t));
    Buffers buf1 = make_buffers(1);
    const int max_ctx = prefix_len + decode_len + 64;

    // 3. Checkpoint populated-prefix same-mode resume + cross-mode isolation check.
    Json ckpt_check = run_ckpt_isolation_test(w, abl, model_dir, kv_fp8, kv_q4, ids, d_ids, d_logits, d_pos, buf1, stream);
    std::printf("[#829] checkpoint populated-resume + mode isolation: %s (pp_exact=%d/%d lf_exact=%d/%d)\n",
                ckpt_check["passed"].get<bool>() ? "PASS" : "FAIL",
                ckpt_check["per_pass_logit_byte_exact_steps"].get<int>(), ckpt_check["ckpt_verify_steps"].get<int>(),
                ckpt_check["last_full_logit_byte_exact_steps"].get<int>(), ckpt_check["ckpt_verify_steps"].get<int>());
    std::fflush(stdout);

    auto run_arm = [&](KvCacheMode mode, int chunk_size) -> ArmOutput {
      using Clock = std::chrono::steady_clock;
      ArmOutput out;
      out.mode = mode;
      out.prefill_chunk = chunk_size;
      out.prefix_len = prefix_len;
      out.decode_len = decode_len;
      Prefiller pre(w, abl, chunk_size, stream);
      std::size_t f0 = 0, f1 = 0, tot = 0;
      CUDA_CHECK(cudaMemGetInfo(&f0, &tot));
      State st = make_state(max_ctx, kv_fp8, kv_q4, {}, mode == KvCacheMode::LastFull);
      CUDA_CHECK(cudaMemGetInfo(&f1, &tot));
      out.full_kv_bytes = st.k_full.bytes() + st.v_full.bytes();
      out.swa_ring_bytes = st.k_ring.bytes() + st.v_ring.bytes();
      out.cuda_state_delta_bytes = (f0 > f1) ? std::uint64_t(f0 - f1) : 0u;

      Decoder dec(w, abl, st, true);
      reset_state(st, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));

      const auto tp0 = Clock::now();
      for (int p0 = 0; p0 < prefix_len; p0 += chunk_size) {
        const int T = std::min(chunk_size, prefix_len - p0);
        pre.run(st, static_cast<const std::int32_t*>(d_ids.p) + p0, T, p0, static_cast<float*>(d_logits.p), nullptr);
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      out.prefill_seconds = std::chrono::duration<double>(Clock::now() - tp0).count();

      out.logits.reserve(std::size_t(decode_len));
      out.logits.push_back(download_f32(static_cast<const float*>(d_logits.p), UNPADDED_VOCAB));

      const auto td0 = Clock::now();
      for (int i = 0; i < decode_len - 1; ++i) {
        const std::int64_t tok = ids[std::size_t(prefix_len + i)];
        const std::int64_t pos = prefix_len + i;
        CUDA_CHECK(cudaMemcpyAsync(buf1.token.data, &tok, sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pos.p, &pos, sizeof(std::int64_t), cudaMemcpyHostToDevice, stream));
        dec.decode(buf1, static_cast<const std::int64_t*>(d_pos.p), false, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        out.logits.push_back(download_f32(static_cast<const float*>(buf1.logits.data), UNPADDED_VOCAB));
      }
      out.decode_seconds = std::chrono::duration<double>(Clock::now() - td0).count();

      double sum_all = 0.0, sum_swa = 0.0, sum_beyond = 0.0;
      int n_swa = 0, n_beyond = 0;
      for (int i = 0; i < decode_len; ++i) {
        const std::int32_t target = ids[std::size_t(prefix_len + i)];
        const double nll = token_nll(out.logits[std::size_t(i)], target);
        sum_all += nll;
        if (i < WINDOW) {
          sum_swa += nll;
          ++n_swa;
        } else {
          sum_beyond += nll;
          ++n_beyond;
        }
      }
      out.mean_nll = sum_all / double(decode_len);
      out.mean_nll_within_swa = n_swa > 0 ? (sum_swa / double(n_swa)) : 0.0;
      out.mean_nll_beyond_swa = n_beyond > 0 ? (sum_beyond / double(n_beyond)) : 0.0;
      out.ppl = std::exp(out.mean_nll);
      std::printf("  [chunk=%4d mode=%-9s] full_kv=%.2f MiB prefill=%.3fs (%.0f tok/s) decode=%.3fs (%.1f tok/s) "
                  "NLL=%.6f (<=1024:%.6f >1024:%.6f) PPL=%.4f\n",
                  chunk_size, to_string(mode), double(out.full_kv_bytes) / double(1ull << 20), out.prefill_seconds,
                  double(prefix_len) / std::max(out.prefill_seconds, 1e-9), out.decode_seconds,
                  double(std::max(1, decode_len - 1)) / std::max(out.decode_seconds, 1e-9), out.mean_nll,
                  out.mean_nll_within_swa, out.mean_nll_beyond_swa, out.ppl);
      std::fflush(stdout);
      return out;
    };

    Json pairs = Json::array();
    for (int chunk_size : chunks) {
      ArmOutput pp = run_arm(KvCacheMode::PerPass, chunk_size);
      ArmOutput lf = run_arm(KvCacheMode::LastFull, chunk_size);

      std::vector<double> kls(static_cast<std::size_t>(decode_len));
      int top1_all = 0, top1_swa = 0, top1_beyond = 0;
      int n_swa = 0, n_beyond = 0;
      double kl_swa = 0.0, kl_beyond = 0.0;
      for (int i = 0; i < decode_len; ++i) {
        const double k = kl_div(pp.logits[std::size_t(i)], lf.logits[std::size_t(i)]);
        kls[std::size_t(i)] = k;
        const bool match = (argmax(pp.logits[std::size_t(i)]) == argmax(lf.logits[std::size_t(i)]));
        if (match) ++top1_all;
        if (i < WINDOW) {
          kl_swa += k;
          ++n_swa;
          if (match) ++top1_swa;
        } else {
          kl_beyond += k;
          ++n_beyond;
          if (match) ++top1_beyond;
        }
      }
      std::vector<double> sorted = kls;
      std::sort(sorted.begin(), sorted.end());
      const double kl_mean = std::accumulate(kls.begin(), kls.end(), 0.0) / double(kls.size());
      const double kl_p50 = sorted[sorted.size() / 2];
      const double kl_p95 = sorted[std::size_t(double(sorted.size() - 1) * 0.95)];
      const double kl_max = sorted.back();
      const double nll_delta = lf.mean_nll - pp.mean_nll;
      const double nll_rel_pct = 100.0 * nll_delta / std::max(pp.mean_nll, 1e-12);
      const double nll_beyond_delta = lf.mean_nll_beyond_swa - pp.mean_nll_beyond_swa;
      const double nll_beyond_rel_pct =
          n_beyond > 0 ? (100.0 * nll_beyond_delta / std::max(pp.mean_nll_beyond_swa, 1e-12)) : 0.0;

      std::printf("  [PAIR chunk=%4d] dNLL=%+.6f (%+.3f%%) | >1024 dNLL=%+.6f (%+.3f%%) | "
                  "KL mean=%.6f p50=%.6f p95=%.6f max=%.6f | top1=%d/%d (%.2f%%, >1024: %d/%d=%.2f%%)\n",
                  chunk_size, nll_delta, nll_rel_pct, nll_beyond_delta, nll_beyond_rel_pct, kl_mean, kl_p50, kl_p95,
                  kl_max, top1_all, decode_len, 100.0 * double(top1_all) / double(decode_len), top1_beyond, n_beyond,
                  n_beyond > 0 ? (100.0 * double(top1_beyond) / double(n_beyond)) : 0.0);
      std::fflush(stdout);

      auto arm_json = [](const ArmOutput& a) {
        return Json{{"kv_cache_mode", to_string(a.mode)},
                    {"full_slots", full_slot_count(a.mode == KvCacheMode::LastFull)},
                    {"full_kv_bytes", a.full_kv_bytes},
                    {"swa_ring_bytes", a.swa_ring_bytes},
                    {"cuda_state_delta_bytes", a.cuda_state_delta_bytes},
                    {"prefill_seconds", a.prefill_seconds},
                    {"prefill_tok_s", double(a.prefix_len) / std::max(a.prefill_seconds, 1e-9)},
                    {"decode_seconds", a.decode_seconds},
                    {"decode_tok_s", double(std::max(1, a.decode_len - 1)) / std::max(a.decode_seconds, 1e-9)},
                    {"mean_nll", a.mean_nll},
                    {"mean_nll_within_swa", a.mean_nll_within_swa},
                    {"mean_nll_beyond_swa", a.mean_nll_beyond_swa},
                    {"ppl", a.ppl}};
      };
      pairs.push_back(Json{
          {"prefill_chunk", chunk_size},
          {"prefix_len", prefix_len},
          {"decode_len", decode_len},
          {"per_pass", arm_json(pp)},
          {"last_full", arm_json(lf)},
          {"delta",
           {{"full_kv_ratio", double(lf.full_kv_bytes) / double(std::max<std::uint64_t>(1, pp.full_kv_bytes))},
            {"nll_delta", nll_delta},
            {"nll_rel_pct", nll_rel_pct},
            {"nll_beyond_swa_delta", nll_beyond_delta},
            {"nll_beyond_swa_rel_pct", nll_beyond_rel_pct},
            {"kl_mean", kl_mean},
            {"kl_within_swa_mean", n_swa > 0 ? kl_swa / double(n_swa) : 0.0},
            {"kl_beyond_swa_mean", n_beyond > 0 ? kl_beyond / double(n_beyond) : 0.0},
            {"kl_p50", kl_p50},
            {"kl_p95", kl_p95},
            {"kl_max", kl_max},
            {"top1_match", top1_all},
            {"top1_total", decode_len},
            {"top1_rate", double(top1_all) / double(decode_len)},
            {"top1_beyond_swa_match", top1_beyond},
            {"top1_beyond_swa_total", n_beyond},
            {"top1_beyond_swa_rate", n_beyond > 0 ? double(top1_beyond) / double(n_beyond) : 0.0}}},
      });
    }

    CUDA_CHECK(cudaStreamDestroy(stream));

    Json report{
        {"experiment", "829_native_last_full_kv_control"},
        {"gpu", {{"name", prop.name}, {"uuid", gpu_uuid}, {"total_vram_bytes", std::uint64_t(total_vram)}}},
        {"model",
         {{"dir", model_dir},
          {"config_sha256", model_cfg_sha},
          {"dense_fp8", dense_fp8},
          {"kv_format", kv_format},
          {"weights_vram_delta_bytes",
           free_before_w > free_after_w ? std::uint64_t(free_before_w - free_after_w) : 0u}}},
        {"eval_tokens",
         {{"prefix_len", prefix_len},
          {"decode_len", decode_len},
          {"swa_window", WINDOW},
          {"prompt_sha256", prompt_sha}}},
        {"vram_table", std::move(vram_table)},
        {"checkpoint_mode_isolation", std::move(ckpt_check)},
        {"pairs", std::move(pairs)},
    };
    if (!out_json.empty()) {
      std::ofstream out(out_json);
      if (!out) throw std::runtime_error("cannot write out_json: " + out_json);
      out << report.dump(2) << "\n";
      std::printf("[#829] wrote %s\n", out_json.c_str());
    }
    return report["checkpoint_mode_isolation"]["passed"].get<bool>() ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "eval_kv_last_full error: %s\n", e.what());
    return 1;
  }
}

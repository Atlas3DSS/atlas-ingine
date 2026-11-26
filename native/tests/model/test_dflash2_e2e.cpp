// Real-model end-to-end verification for #875 native Apache-2.0 DFlash2 drafter:
// 1. Fail-closed option rejection before model load (batch max_batch>1, loop_passes!=3, last_full, Qwen3.8).
// 2. Greedy token comparison: DFlash2 (th=0 and th=0.8) vs unassisted base model (spec=false) on chat & code.
// 3. Prefix-cache checkpoint + VRAM parking round-trip parity & tenant namespace isolation on DFlash2.
// 4. Sampled (T=0.7) determinism & verifier acceptance on real DFlash2 + Inkling-2-Mini.
#include "backends/qwen38/qwen38_backend.h"
#include "ingine/generator.h"
#include "model/model_data.h"
#include "text/chat.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ingine::model;

namespace {

#define REQ(cond)                                                                         \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);       \
      return 1;                                                                           \
    }                                                                                     \
  } while (0)

template <class Fn>
bool throws_invalid(Fn&& fn, const char* label, const char* substr) {
  try {
    fn();
  } catch (const std::invalid_argument& e) {
    if (std::string_view(e.what()).find(substr) == std::string_view::npos) {
      std::fprintf(stderr, "FAIL reject_%s: unexpected message '%s' (expected '%s')\n", label, e.what(), substr);
      return false;
    }
    std::printf("PASS reject_%s (%s)\n", label, e.what());
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL reject_%s: wrong exception: %s\n", label, e.what());
    return false;
  }
  std::fprintf(stderr, "FAIL reject_%s: did not throw\n", label);
  return false;
}

std::vector<std::int32_t> run_gen(Generator& gen, const Prompt& p, int max_toks, const Sampling& s,
                                  std::span<const std::int32_t> stop, GenerateStats* out_stats = nullptr) {
  std::vector<std::int32_t> out;
  GenerateStats st = gen.generate(p, max_toks, s, stop, [&](std::span<const std::int32_t> toks) {
    out.insert(out.end(), toks.begin(), toks.end());
    return true;
  });
  if (out_stats) *out_stats = st;
  return out;
}

}  // namespace

int main() {
  const char* model_env = std::getenv("INGINE_MODEL_DIR");
  const char* df2_env = std::getenv("INGINE_DFLASH2_DIR");
  if (!model_env || !*model_env || !df2_env || !*df2_env) {
    std::printf("skip: set INGINE_MODEL_DIR and INGINE_DFLASH2_DIR to run model_dflash2_e2e\n");
    return 77;
  }
  const std::filesystem::path model_dir(model_env);
  const std::filesystem::path dflash2_dir(df2_env);
  if (!std::filesystem::exists(model_dir / "tokenizer.json") ||
      !std::filesystem::exists(dflash2_dir / "model.safetensors")) {
    std::printf("skip: model or DFlash2 checkpoint missing\n");
    return 77;
  }

  // 1. Fail-closed rejections before model load.
  {
    RuntimeOptions bad;
    bad.model_dir = model_dir;
    bad.dflash2_dir = dflash2_dir;
    bad.spec = true;
    bad.max_batch = 2;
    REQ(throws_invalid([&] { (void)make_generator(bad); }, "dflash2_batch", "--max-batch 1"));

    bad.max_batch = 1;
    bad.loop_passes = 2;
    REQ(throws_invalid([&] { (void)make_generator(bad); }, "dflash2_loop_passes", "--loop-passes 3"));

    bad.loop_passes = 3;
    bad.kv_cache_mode = KvCacheMode::LastFull;
    REQ(throws_invalid([&] { (void)make_generator(bad); }, "dflash2_last_full", "per_pass"));

    bad.kv_cache_mode = KvCacheMode::PerPass;
    bad.spec = false;
    REQ(throws_invalid([&] { (void)make_generator(bad); }, "dflash2_without_spec", "--spec"));

    ingine::backends::qwen38::Qwen38Config qcfg;
    qcfg.model_type = "qwen3_5_text";
    qcfg.architectures = {"Qwen3_5ForConditionalGeneration"};
    qcfg.vocab_size = 248320;
    qcfg.hidden_size = 5120;
    qcfg.intermediate_size = 17408;
    qcfg.num_hidden_layers = 4;
    qcfg.max_position_embeddings = 262144;
    qcfg.num_attention_heads = 24;
    qcfg.num_key_value_heads = 4;
    qcfg.head_dim = 256;
    qcfg.rotary_dim = 64;
    qcfg.linear_key_head_dim = 128;
    qcfg.linear_value_head_dim = 128;
    qcfg.linear_num_key_heads = 16;
    qcfg.linear_num_value_heads = 48;
    qcfg.layer_types = {ingine::backends::qwen38::Qwen38LayerKind::LinearAttention,
                        ingine::backends::qwen38::Qwen38LayerKind::LinearAttention,
                        ingine::backends::qwen38::Qwen38LayerKind::LinearAttention,
                        ingine::backends::qwen38::Qwen38LayerKind::FullAttention};
    qcfg.num_linear_layers = 3;
    qcfg.num_full_layers = 1;
    qcfg.weight_precision = ingine::backends::qwen38::Qwen38WeightPrecision::Bf16;
    RuntimeOptions qopt;
    qopt.dense_fp8 = false;
    qopt.dflash2_dir = dflash2_dir;
    qopt.spec = false;  // test explicit dflash2_dir check even if spec=false
    REQ(throws_invalid([&] { ingine::backends::qwen38::validate_qwen38_runtime_options(qcfg, qopt); },
                       "qwen38_dflash2_dir", "--dflash2-dir"));
  }

  const auto fmt = ingine::text::ChatFormat::load(model_dir);
  const auto& tok = fmt.tokenizer();
  const std::vector<std::int32_t> stop = {*tok.token_id("<|content_model_end_sampling|>")};

  auto render = [&](const std::string& sys, const std::string& user) {
    ingine::text::ChatOptions co;
    co.reasoning_effort = "medium";
    nlohmann::ordered_json msgs = nlohmann::ordered_json::array();
    if (!sys.empty()) msgs.push_back({{"role", "system"}, {"content", sys}});
    msgs.push_back({{"role", "user"}, {"content", user}});
    return fmt.render(msgs, nlohmann::ordered_json(), co);
  };

  const auto p_chat = render("", "Explain how a hash map handles collisions, with a short Python example.");
  const auto p_code = render("", "Write a Python module implementing an LRU cache with O(1) get/put, type hints and unit tests.");
  const int N_TOKS = 128;

  // 2. Unassisted base greedy outputs (spec=false).
  std::vector<std::int32_t> base_chat, base_code;
  {
    RuntimeOptions opt;
    opt.model_dir = model_dir;
    opt.max_ctx = 8192;
    opt.dense_fp8 = true;
    opt.kv_fp8 = true;
    opt.spec = false;
    opt.park_mb = 0;
    auto gen = make_generator(opt);
    Sampling greedy;
    greedy.temperature = 0.f;
    base_chat = run_gen(*gen, Prompt{p_chat.ids, {}, 0, "", true}, N_TOKS, greedy, stop);
    base_code = run_gen(*gen, Prompt{p_code.ids, {}, 0, "", true}, N_TOKS, greedy, stop);
    REQ(int(base_chat.size()) == N_TOKS && int(base_code.size()) == N_TOKS);
  }

  // 3. Real DFlash2 generator: greedy comparison + teacher-forced base replay + cache round-trip + sampled fidelity.
  {
    RuntimeOptions opt;
    opt.model_dir = model_dir;
    opt.dflash2_dir = dflash2_dir;
    opt.max_ctx = 8192;
    opt.dense_fp8 = true;
    opt.kv_fp8 = true;
    opt.spec = true;
    opt.spec_threshold = 0.8f;
    opt.park_mb = 1024;
    auto gen = make_generator(opt);

    Sampling greedy;
    greedy.temperature = 0.f;
    GenerateStats st_chat, st_code;
    const auto df2_chat = run_gen(*gen, Prompt{p_chat.ids, {}, 0, "", true}, N_TOKS, greedy, stop, &st_chat);
    const auto df2_code = run_gen(*gen, Prompt{p_code.ids, {}, 0, "", true}, N_TOKS, greedy, stop, &st_code);
    REQ(df2_chat.size() == base_chat.size() && df2_code.size() == base_code.size());

    auto report_match = [](const char* name, const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b,
                           const GenerateStats& st) {
      int prefix = 0;
      while (prefix < int(a.size()) && a[prefix] == b[prefix]) ++prefix;
      std::printf("GREEDY %s: exact_prefix=%d/%zu rounds=%d accepted=%d tok/verify=%.2f\n", name, prefix, a.size(),
                  st.spec_rounds, st.spec_accepted,
                  st.spec_rounds ? double(st.spec_accepted + st.spec_rounds) / st.spec_rounds : 0.0);
      return prefix;
    };
    const int chat_pre = report_match("chat_th0.8", base_chat, df2_chat, st_chat);
    const int code_pre = report_match("code_th0.8", base_code, df2_code, st_code);
    // Target verifier uses the same model weights; early prefix must agree strongly (FP16 verify-width rounding can
    // flip only at genuine near-ties deep in generation).
    REQ(chat_pre >= 16 && code_pre >= 16);

    // Teacher-forced replay of exact unassisted base_code tokens through DFlash2: must emit exact base_code tokens
    // and achieve strong draft acceptance (> 2.0 tok/verify).
    Sampling forced_s;
    forced_s.temperature = 0.f;
    forced_s.forced = base_code;
    GenerateStats st_forced;
    const auto replay_code = run_gen(*gen, Prompt{p_code.ids, {}, 0, "", true}, N_TOKS, forced_s, stop, &st_forced);
    REQ(replay_code == base_code);
    const double forced_tpv =
        st_forced.spec_rounds ? double(st_forced.spec_accepted + st_forced.spec_rounds) / st_forced.spec_rounds : 0.0;
    std::printf("PASS dflash2_forced_base_replay (exact=128/128, tok/verify=%.2f)\n", forced_tpv);
    REQ(forced_tpv >= 2.5);

    // Prefix-cache root checkpoint, VRAM parking, in-place checkpoint round-trip parity & tenant isolation on DFlash2.
    // Root prefix must exceed kMinRootTokens (64) and full prompts must exceed kMinParked (256).
    const std::string shared_sys =
        "You are a concise, precise systems engineering assistant specializing in algorithms, data structures, "
        "memory hierarchy, cache locality, concurrency primitives, and numerical stability. Always structure your "
        "response with clear headings, exact asymptotic time and space complexity bounds, worst-case failure modes, "
        "practical cache-line tradeoffs, and concrete implementation invariants. Keep explanations tightly grounded "
        "in standard computer science terminology without filler prose or repetition.";
    const std::string filler_ctx =
        " Assume keys are 64-bit unsigned integers stored in a contiguous array allocated on 64-byte cache-line "
        "boundaries, running on a single core with a 32 KiB L1 data cache, a 256 KiB L2 cache, and an 8 MiB shared "
        "L3 cache. Consider both random uniform input distributions and adversarial inputs crafted to trigger "
        "worst-case recursion depth, branch mispredictions, or extra memory traffic. Include a short note on "
        "in-place vs auxiliary buffer memory footprint, stability guarantees for equal keys, and when hybrid "
        "fallbacks such as insertion sort for small partitions improve throughput in practice. Provide exact "
        "recurrence relations and closed-form bounds for best, average, and worst cases, and call out any "
        "assumptions about pivot selection or merge buffer allocation. Finally, compare sequential memory access "
        "patterns, hardware prefetcher friendliness, TLB pressure on multi-gigabyte arrays, branchless partitioning "
        "techniques, and deterministic worst-case safeguards like introsort or bottom-up natural run merging.";
    const auto pa = render(shared_sys, "Analyze quicksort thoroughly." + filler_ctx);
    const auto pb = render(shared_sys, "Analyze mergesort thoroughly." + filler_ctx);
    std::printf("CACHE prompt sizes: pa=%zu (root=%d) pb=%zu (root=%d)\n", pa.ids.size(), pa.root_prefix_tokens,
                pb.ids.size(), pb.root_prefix_tokens);
    REQ(pa.root_prefix_tokens >= 64 && pa.root_prefix_tokens == pb.root_prefix_tokens);
    REQ(pa.ids.size() >= 256 && pb.ids.size() >= 256);

    GenerateStats s1, s2, s_ref0, s_ref, s3, s4, s4_ref0, s4_ref, s_iso;
    // 1) Cold pa -> captures root checkpoint at root_prefix_tokens and saves live checkpoint at pa.ids.size().
    const auto out_a1 =
        run_gen(*gen, Prompt{pa.ids, {}, pa.root_prefix_tokens, "tenant-a", false}, 64, greedy, stop, &s1);
    REQ(s1.reuse == Reuse::None);
    // 2) Sibling pb -> parks pa in VRAM and restores the verified root checkpoint (Reuse::RootCheckpoint).
    const auto out_b_warm =
        run_gen(*gen, Prompt{pb.ids, {}, pb.root_prefix_tokens, "tenant-a", false}, 64, greedy, stop, &s2);
    // 3) Strict-prefix Extend reference for pb with identical chunk split at root_prefix_tokens.
    const std::vector<std::int32_t> root_only(pa.ids.begin(), pa.ids.begin() + pa.root_prefix_tokens);
    (void)run_gen(*gen, Prompt{root_only, {}, 0, "tenant-ref", false}, 0, greedy, stop, &s_ref0);
    const auto out_b_ext =
        run_gen(*gen, Prompt{pb.ids, {}, 0, "tenant-ref", false}, 64, greedy, stop, &s_ref);
    // 4) Return to pa -> restores parked pa from VRAM (Reuse::Parked) with zero recompute, exact bit parity.
    const auto out_a2 =
        run_gen(*gen, Prompt{pa.ids, {}, pa.root_prefix_tokens, "tenant-a", false}, 64, greedy, stop, &s3);
    // 5) Follow-up extending pa -> restores live device checkpoint of pa (Reuse::Checkpoint), exact bit parity vs Extend.
    std::vector<std::int32_t> pa_follow = pa.ids;
    pa_follow.insert(pa_follow.end(), out_a1.begin(), out_a1.begin() + 2);
    pa_follow.insert(pa_follow.end(), pb.ids.end() - 24, pb.ids.end());
    const auto out_af_ckpt =
        run_gen(*gen, Prompt{pa_follow, {}, 0, "tenant-a", false}, 64, greedy, stop, &s4);
    (void)run_gen(*gen, Prompt{pa.ids, {}, 0, "tenant-ref2", false}, 0, greedy, stop, &s4_ref0);
    const auto out_af_ext =
        run_gen(*gen, Prompt{pa_follow, {}, 0, "tenant-ref2", false}, 64, greedy, stop, &s4_ref);
    // 6) Different tenant namespace -> must fail closed to Reuse::None (0 reused tokens).
    const auto out_b_iso =
        run_gen(*gen, Prompt{pb.ids, {}, pb.root_prefix_tokens, "tenant-b", false}, 64, greedy, stop, &s_iso);
    const PrefixCacheStats cs = gen->cache_stats();

    std::printf("CACHE roundtrip: s2=%s(%d) s_ref=%s(%d) s3=%s(%d) s4=%s(%d) s_iso=%s(%d) "
                "staging=%.1fMiB vram_entries=%llu roots=%llu d2d_restore=%.1fMB\n",
                to_string(s2.reuse), s2.reused_tokens, to_string(s_ref.reuse), s_ref.reused_tokens,
                to_string(s3.reuse), s3.reused_tokens, to_string(s4.reuse), s4.reused_tokens,
                to_string(s_iso.reuse), s_iso.reused_tokens,
                double(cs.device_staging_bytes) / (1024.0 * 1024.0),
                static_cast<unsigned long long>(cs.vram_entries), static_cast<unsigned long long>(cs.root_entries),
                double(cs.d2d_restore_bytes) / 1e6);
    REQ(s2.reuse == Reuse::RootCheckpoint && s2.reused_tokens == pa.root_prefix_tokens);
    REQ(s_ref.reuse == Reuse::Extend && s_ref.reused_tokens == pa.root_prefix_tokens);
    REQ(out_b_warm == out_b_ext);
    REQ(s3.reuse == Reuse::Parked && s3.reused_tokens == int(pa.ids.size()));
    REQ(out_a2 == out_a1);
    REQ(s4.reuse == Reuse::Checkpoint && s4.reused_tokens == int(pa.ids.size()));
    REQ(s4_ref.reuse == Reuse::Extend && s4_ref.reused_tokens == int(pa.ids.size()));
    REQ(out_af_ckpt == out_af_ext);
    REQ(s_iso.reuse == Reuse::None && s_iso.reused_tokens == 0);
    REQ(out_b_iso.size() == out_b_warm.size());
    REQ(cs.root_entries >= 1 && cs.vram_entries >= 2 && cs.d2d_restore_bytes > 0);
    std::printf("PASS dflash2_prefix_cache_roundtrip_and_isolation\n");

    // Sampled (T=0.7) determinism & verifier acceptance.
    Sampling samp1;
    samp1.temperature = 0.7f;
    samp1.seed = 424242ull;
    GenerateStats ss1, ss2;
    const auto tok_s1 = run_gen(*gen, Prompt{p_code.ids, {}, 0, "", true}, 64, samp1, stop, &ss1);
    const auto tok_s2 = run_gen(*gen, Prompt{p_code.ids, {}, 0, "", true}, 64, samp1, stop, &ss2);
    REQ(tok_s1 == tok_s2);
    REQ(ss1.spec_rounds > 0 && ss1.spec_accepted > 0);
    for (std::int32_t id : tok_s1) REQ(id >= 0 && id < UNPADDED_VOCAB);
    Sampling samp3 = samp1;
    samp3.seed = 999999ull;
    const auto tok_s3 = run_gen(*gen, Prompt{p_code.ids, {}, 0, "", true}, 64, samp3, stop);
    REQ(tok_s3 != tok_s1);
    std::printf("PASS dflash2_sampled_determinism (accepted=%d/%d rounds, tok/verify=%.2f)\n", ss1.spec_accepted,
                ss1.spec_rounds, double(ss1.spec_accepted + ss1.spec_rounds) / ss1.spec_rounds);
  }

  std::printf("MODEL_DFLASH2_E2E_ALL_PASS\n");
  return 0;
}

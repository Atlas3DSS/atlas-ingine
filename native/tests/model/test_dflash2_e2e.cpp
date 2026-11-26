// Real-model end-to-end verification for #875 native Apache-2.0 DFlash2 drafter:
// 1. Fail-closed option rejection before model load (max_batch>1, loop_passes!=3, last_full, !spec, Qwen3.8).
// 2. Unforced fixed-prefix target verifier audit on chat & code across BOTH th=0.8 and th=0.0 for DFlash2
//    and the existing DSpark verifier at identical geometry:
//      - Asserts 0 wrong accepts against actual verifier-width target logits across all 128 unforced tokens.
//      - Asserts 128/128 bit-exact parity between the audited unforced loop and production EngineGenerator/SpecLoop.
//      - At the first unforced divergence from clean width-1 base output (identical preceding token prefix),
//        records clean width-1, pre-round checkpoint width-1 (state-drift check), and actual verify-width m
//        selected IDs, top-two logits/margins, and drafted vs correction token.
// 3. Prefix-cache root checkpoint, VRAM parking, in-place checkpoint round-trip parity & tenant isolation on DFlash2.
// 4. Sampled (T=0.7) determinism & verifier acceptance on real DFlash2 + Inkling-2-Mini.
#include "backends/qwen38/qwen38_backend.h"
#include "core/device.h"
#include "ingine/generator.h"
#include "kernels/draft/draft.h"
#include "kernels/head/head.h"
#include "model/drafter.h"
#include "model/model_data.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "text/chat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ingine::model;
namespace kd = ingine::kernels::draft;
namespace kh = ingine::kernels::head;

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

struct Top2 {
  int id1 = -1;
  float v1 = -std::numeric_limits<float>::infinity();
  int id2 = -1;
  float v2 = -std::numeric_limits<float>::infinity();
  float margin() const { return v1 - v2; }
};

Top2 compute_top2(const float* row, int V) {
  Top2 r;
  for (int i = 0; i < V; ++i) {
    const float v = row[i];
    if (v > r.v1) {
      r.id2 = r.id1;
      r.v2 = r.v1;
      r.id1 = i;
      r.v1 = v;
    } else if (v > r.v2) {
      r.id2 = i;
      r.v2 = v;
    }
  }
  return r;
}

int select_verify_width(const float conf[kd::GAMMA], float threshold, std::span<const int> widths) {
  int lead = 0;
  while (lead < kd::GAMMA && conf[lead] > threshold) ++lead;
  int m = widths.back();
  for (int k = int(widths.size()) - 1; k >= 0; --k) {
    if (widths[std::size_t(k)] >= lead + 1) m = widths[std::size_t(k)];
  }
  return m;
}

void prefill_prompt(Runtime& rt, Prefiller& prefiller, Drafter* drafter, const std::vector<std::int32_t>& prompt,
                    ingine::DeviceBuffer& chunk_tokens, ingine::DeviceBuffer& logits_buf,
                    ingine::DeviceBuffer& taps_buf) {
  const cudaStream_t stream = rt.stream();
  reset_state(rt.state(), stream);
  if (drafter) drafter->reset();
  const std::size_t n = prompt.size();
  const std::size_t chunk = std::size_t(prefiller.max_chunk());
  for (std::size_t s = 0; s < n; s += chunk) {
    const std::size_t step = std::min(chunk, n - s);
    const int T = int(step);
    CUDA_CHECK(cudaMemcpyAsync(chunk_tokens.p, prompt.data() + s, step * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream));
    ingine::Tensor taps;
    if (drafter) taps = rm(taps_buf.p, ingine::kActDType, {T, NUM_TAPS, HIDDEN});
    prefiller.run(rt.state(), static_cast<const std::int32_t*>(chunk_tokens.p), T, int(s),
                  static_cast<float*>(logits_buf.p), drafter ? &taps : nullptr);
    if (drafter) {
      drafter->note_tokens(std::span<const std::int32_t>(prompt.data() + s, step));
      drafter->commit(taps, int(s));
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

struct BaseRun {
  std::vector<std::int32_t> tokens;
  std::vector<Top2> top2;
};

BaseRun run_clean_width1(Runtime& rt, Prefiller& prefiller, const std::vector<std::int32_t>& prompt, int n_toks,
                         ingine::DeviceBuffer& chunk_tokens, ingine::DeviceBuffer& logits_buf,
                         ingine::DeviceBuffer& taps_buf) {
  const cudaStream_t stream = rt.stream();
  prefill_prompt(rt, prefiller, nullptr, prompt, chunk_tokens, logits_buf, taps_buf);
  std::vector<float> host_logits(UNPADDED_VOCAB);
  CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), logits_buf.p, std::size_t(UNPADDED_VOCAB) * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  BaseRun res;
  res.tokens.reserve(std::size_t(n_toks));
  res.top2.reserve(std::size_t(n_toks));
  const Top2 t0 = compute_top2(host_logits.data(), UNPADDED_VOCAB);
  res.tokens.push_back(t0.id1);
  res.top2.push_back(t0);

  for (int i = 1; i < n_toks; ++i) {
    const std::int32_t tok = res.tokens.back();
    const int pos = int(prompt.size()) + i - 1;
    rt.set_inputs(std::span<const std::int32_t>(&tok, 1), pos);
    const float* d_logits = rt.run_step(1);
    CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), d_logits, std::size_t(UNPADDED_VOCAB) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const Top2 ti = compute_top2(host_logits.data(), UNPADDED_VOCAB);
    res.tokens.push_back(ti.id1);
    res.top2.push_back(ti);
  }
  return res;
}

struct AuditResult {
  std::vector<std::int32_t> tokens;
  int rounds = 0;
  int accepted = 0;
  int exact_prefix = 0;
  int wrong_accepts_vs_target_verify = 0;
  int wrong_corrections_vs_target_verify = 0;
  bool pre_div_state_drift = false;
  bool diverged = false;
  int div_idx = -1;
  bool div_was_top2_swap = false;
  float div_clean_margin = 0.f;
};

AuditResult audit_unforced_spec(Runtime& rt, Prefiller& prefiller, Drafter& drafter,
                                const std::vector<std::int32_t>& prompt, const BaseRun& base, int n_toks,
                                float threshold, const char* label, ingine::DeviceBuffer& chunk_tokens,
                                ingine::DeviceBuffer& logits_buf, ingine::DeviceBuffer& taps_buf,
                                ingine::DeviceBuffer& rt_ckpt) {
  const cudaStream_t stream = rt.stream();
  prefill_prompt(rt, prefiller, &drafter, prompt, chunk_tokens, logits_buf, taps_buf);
  std::vector<float> host_logits(std::size_t(kd::BLOCK) * UNPADDED_VOCAB);
  std::vector<float> w1_logits(UNPADDED_VOCAB);
  CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), logits_buf.p, std::size_t(UNPADDED_VOCAB) * sizeof(float),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  AuditResult ar;
  ar.tokens.reserve(std::size_t(n_toks));
  const Top2 t0 = compute_top2(host_logits.data(), UNPADDED_VOCAB);
  ar.tokens.push_back(t0.id1);
  drafter.note_tokens(std::span<const std::int32_t>(&ar.tokens[0], 1));

  const Drafter::AcceptScratch& scratch = drafter.accept_scratch();
  const auto widths = rt.verify_widths();

  while (int(ar.tokens.size()) < n_toks) {
    const int P = int(prompt.size()) + int(ar.tokens.size()) - 1;
    const std::int32_t anchor = ar.tokens.back();
    const Drafter::Draft d = drafter.draft(anchor, P, 0.f);
    float conf[kd::GAMMA]{};
    std::int32_t d_tok[kd::GAMMA]{};
    CUDA_CHECK(cudaMemcpyAsync(conf, d.conf, sizeof(conf), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_tok, d.tokens, sizeof(d_tok), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int m = select_verify_width(conf, threshold, widths);
    std::vector<std::int32_t> inputs(static_cast<std::size_t>(m));
    inputs[0] = anchor;
    for (int i = 0; i < m - 1; ++i) inputs[static_cast<std::size_t>(i + 1)] = d_tok[i];

    // Save pre-round runtime checkpoint so if this round produces the first divergence vs base, we can measure
    // exact width-1 logits from the identical pre-round state without perturbing the run.
    if (!ar.diverged) {
      rt.save_checkpoint(rt_ckpt.p);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    rt.set_inputs(inputs, P);
    rt.save_verify(P, m);
    const float* d_logits_m = rt.run_step(m);
    kh::argmax(d_logits_m, m, UNPADDED_VOCAB, UNPADDED_VOCAB, scratch.target, stream);
    kd::greedy_accept(scratch.target, d.tokens, m - 1, scratch.result, stream);

    kd::AcceptResult h_acc{};
    CUDA_CHECK(cudaMemcpyAsync(&h_acc, scratch.result, sizeof(h_acc), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_logits.data(), d_logits_m,
                               static_cast<std::size_t>(m) * UNPADDED_VOCAB * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<Top2> wm_top2(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
      wm_top2[std::size_t(i)] = compute_top2(host_logits.data() + std::size_t(i) * UNPADDED_VOCAB, UNPADDED_VOCAB);
    }

    // Hard verification against actual verify-width target logits on EVERY round.
    for (int i = 0; i < h_acc.accepted; ++i) {
      if (d_tok[i] != wm_top2[std::size_t(i)].id1) ++ar.wrong_accepts_vs_target_verify;
    }
    if (h_acc.next != wm_top2[std::size_t(h_acc.accepted)].id1) ++ar.wrong_corrections_vs_target_verify;

    std::vector<std::int32_t> round_out;
    round_out.reserve(std::size_t(h_acc.accepted + 1));
    for (int i = 0; i < h_acc.accepted; ++i) round_out.push_back(d_tok[i]);
    round_out.push_back(h_acc.next);

    // Check for the FIRST unforced divergence from clean width-1 base at identical preceding token prefix.
    if (!ar.diverged) {
      for (int k = 0; k < int(round_out.size()); ++k) {
        const int pos_gen = int(ar.tokens.size()) + k;
        if (pos_gen >= n_toks) break;
        if (round_out[std::size_t(k)] != base.tokens[std::size_t(pos_gen)]) {
          ar.diverged = true;
          ar.div_idx = pos_gen;

          // Restore pre-round checkpoint at P and step width-1 through [anchor, round_out[0..k-1]] (which equals
          // base[out.size()-1 .. pos_gen-1] identically) to inspect exact width-1 logits at pos_gen.
          rt.restore_checkpoint(rt_ckpt.p);
          const float* last_w1 = nullptr;
          for (int step = 0; step <= k; ++step) {
            const std::int32_t tok = (step == 0) ? anchor : round_out[std::size_t(step - 1)];
            rt.set_inputs(std::span<const std::int32_t>(&tok, 1), P + step);
            last_w1 = rt.run_step(1);
          }
          CUDA_CHECK(cudaMemcpyAsync(w1_logits.data(), last_w1, std::size_t(UNPADDED_VOCAB) * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          const Top2 spec_w1 = compute_top2(w1_logits.data(), UNPADDED_VOCAB);
          const Top2& clean_w1 = base.top2[std::size_t(pos_gen)];
          const Top2& vm = wm_top2[std::size_t(k)];
          const float* vm_row = host_logits.data() + std::size_t(k) * UNPADDED_VOCAB;
          const float vm_of_w1_top1 = vm_row[clean_w1.id1];
          const float w1_of_vm_top1 = w1_logits[vm.id1];
          const int draft_at_k = (k < m - 1) ? d_tok[k] : -1;
          const bool was_accepted = (k < h_acc.accepted);
          if (spec_w1.id1 != clean_w1.id1) ar.pre_div_state_drift = true;
          ar.div_was_top2_swap = (round_out[std::size_t(k)] == clean_w1.id2);
          ar.div_clean_margin = clean_w1.margin();

          std::printf(
              "DIV %-16s gen=%d/%d (idx=%d) P=%d m=%d row=%d decision=%s draft=%d emitted=%d accepted_in_round=%d\n"
              "    clean_w1 : top1=%d (%.5f) top2=%d (%.5f) margin=%.5f\n"
              "    spec_w1  : top1=%d (%.5f) top2=%d (%.5f) margin=%.5f logit(vm_top1=%d)=%.5f\n"
              "    verify_m : top1=%d (%.5f) top2=%d (%.5f) margin=%.5f logit(w1_top1=%d)=%.5f "
              "delta_w1_vs_vm=%.5f\n",
              label, pos_gen + 1, n_toks, pos_gen, P, m, k, was_accepted ? "accepted_draft" : "correction", draft_at_k,
              round_out[std::size_t(k)], h_acc.accepted, clean_w1.id1, clean_w1.v1, clean_w1.id2, clean_w1.v2,
              clean_w1.margin(), spec_w1.id1, spec_w1.v1, spec_w1.id2, spec_w1.v2, spec_w1.margin(), vm.id1,
              w1_of_vm_top1, vm.id1, vm.v1, vm.id2, vm.v2, vm.margin(), clean_w1.id1, vm_of_w1_top1,
              std::fabs(vm.v1 - w1_of_vm_top1));

          // Restore pre-round checkpoint and re-execute width m so subsequent rollback/commit uses exact verify state.
          rt.restore_checkpoint(rt_ckpt.p);
          rt.set_inputs(inputs, P);
          rt.save_verify(P, m);
          (void)rt.run_step(m);
          CUDA_CHECK(cudaStreamSynchronize(stream));
          break;
        }
      }
    }

    const int commit = h_acc.accepted + 1;
    rt.rollback(P, commit, m);
    drafter.commit(ptr<const ingine::ActT>(rt.buffers(m).taps), commit, P);
    ++ar.rounds;
    ar.accepted += h_acc.accepted;

    const int remain = n_toks - int(ar.tokens.size());
    const int take = std::min(remain, int(round_out.size()));
    ar.tokens.insert(ar.tokens.end(), round_out.begin(), round_out.begin() + take);
    drafter.note_tokens(std::span<const std::int32_t>(round_out.data(), std::size_t(take)));
  }

  int pre = 0;
  while (pre < n_toks && ar.tokens[std::size_t(pre)] == base.tokens[std::size_t(pre)]) ++pre;
  ar.exact_prefix = pre;
  const double tpv = ar.rounds ? double(ar.accepted + ar.rounds) / ar.rounds : 0.0;
  std::printf("AUDIT %-16s exact_prefix=%d/%d rounds=%d accepted=%d tok/verify=%.2f "
              "wrong_accept=%d wrong_corr=%d pre_div_drift=%s\n",
              label, ar.exact_prefix, n_toks, ar.rounds, ar.accepted, tpv, ar.wrong_accepts_vs_target_verify,
              ar.wrong_corrections_vs_target_verify, ar.pre_div_state_drift ? "YES" : "no");
  return ar;
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
    qopt.spec = false;
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
  const auto p_code =
      render("", "Write a Python module implementing an LRU cache with O(1) get/put, type hints and unit tests.");
  const int N_TOKS = 128;

  // 2. Fixed-prefix unforced target verifier audit:
  //    Single shared Runtime (spec=true, dense_fp8=true, kv_fp8=true, max_ctx=8192) runs clean width-1 base,
  //    then DSpark (th=0.8 & th=0.0) control, then DFlash2 (th=0.8 & th=0.0).
  AuditResult audit_df2_chat_08, audit_df2_code_08, audit_df2_chat_0, audit_df2_code_0;
  BaseRun base_chat, base_code;
  {
    RuntimeOptions rt_opt;
    rt_opt.model_dir = model_dir;
    rt_opt.max_ctx = 8192;
    rt_opt.dense_fp8 = true;
    rt_opt.kv_fp8 = true;
    rt_opt.spec = true;
    rt_opt.park_mb = 0;
    Runtime rt(rt_opt);
    Prefiller prefiller(rt.weights(), rt.ablation(), rt_opt.prefill_chunk, rt.stream(), rt_opt.loop_passes);
    ingine::DeviceBuffer chunk_tokens(std::size_t(rt_opt.prefill_chunk) * sizeof(std::int32_t));
    ingine::DeviceBuffer logits_buf(std::size_t(UNPADDED_VOCAB) * sizeof(float));
    ingine::DeviceBuffer taps_buf(std::size_t(rt_opt.prefill_chunk) * NUM_TAPS * HIDDEN * 2);
    ingine::DeviceBuffer rt_ckpt(rt.checkpoint_bytes());

    base_chat = run_clean_width1(rt, prefiller, p_chat.ids, N_TOKS, chunk_tokens, logits_buf, taps_buf);
    base_code = run_clean_width1(rt, prefiller, p_code.ids, N_TOKS, chunk_tokens, logits_buf, taps_buf);
    REQ(int(base_chat.tokens.size()) == N_TOKS && int(base_code.tokens.size()) == N_TOKS);

    // Control: existing DSpark drafter at identical geometry (th=0.8 and th=0.0).
    {
      Drafter dspark(model_dir, rt.weights(), rt_opt.max_ctx, rt.stream(), Drafter::SHORTLIST_ROWS);
      dspark.capture();
      const auto ds_chat_08 = audit_unforced_spec(rt, prefiller, dspark, p_chat.ids, base_chat, N_TOKS, 0.8f,
                                                  "dspark_chat_0.8", chunk_tokens, logits_buf, taps_buf, rt_ckpt);
      const auto ds_code_08 = audit_unforced_spec(rt, prefiller, dspark, p_code.ids, base_code, N_TOKS, 0.8f,
                                                  "dspark_code_0.8", chunk_tokens, logits_buf, taps_buf, rt_ckpt);
      const auto ds_chat_0 = audit_unforced_spec(rt, prefiller, dspark, p_chat.ids, base_chat, N_TOKS, 0.0f,
                                                 "dspark_chat_0.0", chunk_tokens, logits_buf, taps_buf, rt_ckpt);
      const auto ds_code_0 = audit_unforced_spec(rt, prefiller, dspark, p_code.ids, base_code, N_TOKS, 0.0f,
                                                 "dspark_code_0.0", chunk_tokens, logits_buf, taps_buf, rt_ckpt);
      REQ(ds_chat_08.wrong_accepts_vs_target_verify == 0 && ds_chat_08.wrong_corrections_vs_target_verify == 0);
      REQ(ds_code_08.wrong_accepts_vs_target_verify == 0 && ds_code_08.wrong_corrections_vs_target_verify == 0);
      REQ(ds_chat_0.wrong_accepts_vs_target_verify == 0 && ds_chat_0.wrong_corrections_vs_target_verify == 0);
      REQ(ds_code_0.wrong_accepts_vs_target_verify == 0 && ds_code_0.wrong_corrections_vs_target_verify == 0);
    }

    // DFlash2 drafter at identical geometry (th=0.8 and th=0.0).
    {
      Drafter df2(model_dir, rt.weights(), rt_opt.max_ctx, rt.stream(), Drafter::SHORTLIST_ROWS, dflash2_dir);
      df2.capture();
      audit_df2_chat_08 = audit_unforced_spec(rt, prefiller, df2, p_chat.ids, base_chat, N_TOKS, 0.8f,
                                              "dflash2_chat_0.8", chunk_tokens, logits_buf, taps_buf, rt_ckpt);
      audit_df2_code_08 = audit_unforced_spec(rt, prefiller, df2, p_code.ids, base_code, N_TOKS, 0.8f,
                                              "dflash2_code_0.8", chunk_tokens, logits_buf, taps_buf, rt_ckpt);
      audit_df2_chat_0 = audit_unforced_spec(rt, prefiller, df2, p_chat.ids, base_chat, N_TOKS, 0.0f,
                                             "dflash2_chat_0.0", chunk_tokens, logits_buf, taps_buf, rt_ckpt);
      audit_df2_code_0 = audit_unforced_spec(rt, prefiller, df2, p_code.ids, base_code, N_TOKS, 0.0f,
                                             "dflash2_code_0.0", chunk_tokens, logits_buf, taps_buf, rt_ckpt);
      REQ(audit_df2_chat_08.wrong_accepts_vs_target_verify == 0 &&
          audit_df2_chat_08.wrong_corrections_vs_target_verify == 0);
      REQ(audit_df2_code_08.wrong_accepts_vs_target_verify == 0 &&
          audit_df2_code_08.wrong_corrections_vs_target_verify == 0);
      REQ(audit_df2_chat_0.wrong_accepts_vs_target_verify == 0 &&
          audit_df2_chat_0.wrong_corrections_vs_target_verify == 0);
      REQ(audit_df2_code_0.wrong_accepts_vs_target_verify == 0 &&
          audit_df2_code_0.wrong_corrections_vs_target_verify == 0);
      // Every unforced divergence vs clean width-1 must be a genuine small-margin top1<->top2 flip caused by
      // shared FP16/FP8 verify-width & KV rounding (matching the DSpark control), never an out-of-top2 jump.
      for (const AuditResult* a : {&audit_df2_chat_08, &audit_df2_code_08, &audit_df2_chat_0, &audit_df2_code_0}) {
        REQ(!a->diverged || (a->div_was_top2_swap && a->div_clean_margin < 0.25f));
      }
    }
    std::printf("PASS unforced_fixed_prefix_target_audit (0 wrong accepts/corrections; all diffs are small-margin top1<->top2 width flips)\n");
  }

  // 3. Production EngineGenerator (SpecLoop) bit-exact agreement at th=0.0 and th=0.8 + cache round-trip + sampled.
  {
    RuntimeOptions opt0;
    opt0.model_dir = model_dir;
    opt0.dflash2_dir = dflash2_dir;
    opt0.max_ctx = 8192;
    opt0.dense_fp8 = true;
    opt0.kv_fp8 = true;
    opt0.spec = true;
    opt0.spec_threshold = 0.0f;
    opt0.park_mb = 0;
    auto gen0 = make_generator(opt0);
    Sampling greedy;
    greedy.temperature = 0.f;
    const auto gen_chat_0 = run_gen(*gen0, Prompt{p_chat.ids, {}, 0, "", true}, N_TOKS, greedy, stop);
    const auto gen_code_0 = run_gen(*gen0, Prompt{p_code.ids, {}, 0, "", true}, N_TOKS, greedy, stop);
    REQ(gen_chat_0 == audit_df2_chat_0.tokens);
    REQ(gen_code_0 == audit_df2_code_0.tokens);
    std::printf("PASS dflash2_specloop_exact_match_th0.0 (chat=128/128, code=128/128)\n");
  }

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
    REQ(df2_chat == audit_df2_chat_08.tokens);
    REQ(df2_code == audit_df2_code_08.tokens);
    std::printf("PASS dflash2_specloop_exact_match_th0.8 (chat=128/128, code=128/128)\n");

    // Prefix-cache root checkpoint, VRAM parking, in-place checkpoint round-trip parity & tenant isolation on DFlash2.
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
    const auto out_a1 =
        run_gen(*gen, Prompt{pa.ids, {}, pa.root_prefix_tokens, "tenant-a", false}, 64, greedy, stop, &s1);
    REQ(s1.reuse == Reuse::None);
    const auto out_b_warm =
        run_gen(*gen, Prompt{pb.ids, {}, pb.root_prefix_tokens, "tenant-a", false}, 64, greedy, stop, &s2);
    const std::vector<std::int32_t> root_only(pa.ids.begin(), pa.ids.begin() + pa.root_prefix_tokens);
    (void)run_gen(*gen, Prompt{root_only, {}, 0, "tenant-ref", false}, 0, greedy, stop, &s_ref0);
    const auto out_b_ext = run_gen(*gen, Prompt{pb.ids, {}, 0, "tenant-ref", false}, 64, greedy, stop, &s_ref);
    const auto out_a2 =
        run_gen(*gen, Prompt{pa.ids, {}, pa.root_prefix_tokens, "tenant-a", false}, 64, greedy, stop, &s3);
    std::vector<std::int32_t> pa_follow = pa.ids;
    pa_follow.insert(pa_follow.end(), out_a1.begin(), out_a1.begin() + 2);
    pa_follow.insert(pa_follow.end(), pb.ids.end() - 24, pb.ids.end());
    const auto out_af_ckpt = run_gen(*gen, Prompt{pa_follow, {}, 0, "tenant-a", false}, 64, greedy, stop, &s4);
    (void)run_gen(*gen, Prompt{pa.ids, {}, 0, "tenant-ref2", false}, 0, greedy, stop, &s4_ref0);
    const auto out_af_ext = run_gen(*gen, Prompt{pa_follow, {}, 0, "tenant-ref2", false}, 64, greedy, stop, &s4_ref);
    const auto out_b_iso =
        run_gen(*gen, Prompt{pb.ids, {}, pb.root_prefix_tokens, "tenant-b", false}, 64, greedy, stop, &s_iso);
    const PrefixCacheStats cs = gen->cache_stats();

    std::printf("CACHE roundtrip: s2=%s(%d) s_ref=%s(%d) s3=%s(%d) s4=%s(%d) s_iso=%s(%d) "
                "staging=%.1fMiB vram_entries=%llu roots=%llu d2d_restore=%.1fMB\n",
                to_string(s2.reuse), s2.reused_tokens, to_string(s_ref.reuse), s_ref.reused_tokens,
                to_string(s3.reuse), s3.reused_tokens, to_string(s4.reuse), s4.reused_tokens,
                to_string(s_iso.reuse), s_iso.reused_tokens, double(cs.device_staging_bytes) / (1024.0 * 1024.0),
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

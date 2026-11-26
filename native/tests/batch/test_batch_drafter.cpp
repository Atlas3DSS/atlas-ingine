// Parity + microbenchmark test for BatchDrafter (S = 1..4) against single-stream Drafter.
#include "core/act.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/gguf.h"
#include "kernels/draft/draft.h"
#include "model/batch_drafter.h"
#include "model/drafter.h"
#include "model/model_data.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

using namespace ingine;
namespace kd = kernels::draft;

std::filesystem::path find_model_dir() {
  if (const char* env = std::getenv("INGINE_MODEL_DIR"); env && *env) return std::filesystem::path(env);
  const std::filesystem::path def = "/home/svniak/models/Inkling-2-Mini-atlas-gguf";
  if (std::filesystem::exists(def / "model.gguf") && std::filesystem::exists(def / "dspark.gguf")) return def;
  return {};
}

// Loads only embed (Q8_0), embed_norm (FP32), and unembed (blocked Q6_K) from model.gguf (~1.05 GiB VRAM).
model::Weights load_draft_only_weights(const std::filesystem::path& model_dir, cudaStream_t stream) {
  const GgufFile f(model_dir / "model.gguf");
  model::Weights w;
  auto upload = [&](const void* src, std::size_t bytes) -> void* {
    w.storage.emplace_back(bytes);
    void* p = w.storage.back().p;
    CUDA_CHECK(cudaMemcpyAsync(p, src, bytes, cudaMemcpyHostToDevice, stream));
    return p;
  };
  constexpr int H = model::HIDDEN;
  {
    const GgufTensor& e = f.at("token_embd.weight");
    if (e.type != GgmlType::Q8_0) throw std::runtime_error("expected Q8_0 token_embd.weight");
    w.embed = model::rm(upload(e.data, e.bytes), DType::I8, {model::VOCAB, std::int32_t(e.row_bytes())});
  }
  {
    const GgufTensor& en = f.at("token_embd_norm.weight");
    std::vector<float> buf(H);
    if (en.type == GgmlType::F32) std::memcpy(buf.data(), en.data, H * 4);
    else ggml_dequantize(en.type, en.data, H, buf.data());
    w.embed_norm = model::rm(upload(buf.data(), H * 4), DType::FP32, {H});
  }
  {
    const GgufTensor& ot = f.at("output.weight");
    if (ot.type != GgmlType::Q6_K) throw std::runtime_error("expected Q6_K output.weight");
    const int V_pad = (model::UNPADDED_VOCAB + 15) & ~15;
    const int n_blks = V_pad / 16;
    std::vector<std::uint8_t> qbytes;
    qbytes.reserve(std::size_t(V_pad) * H);
    std::vector<std::int64_t> off(n_blks);
    std::vector<std::int32_t> soff(n_blks);
    std::vector<std::uint16_t> scales;
    scales.reserve(std::size_t(V_pad) * (H / 16) * 2);
    const std::size_t rb = ot.row_bytes();
    for (int blk_i = 0; blk_i < n_blks; ++blk_i) {
      off[blk_i] = std::int64_t(qbytes.size());
      soff[blk_i] = -2 - std::int32_t(scales.size());
      for (int rl = 0; rl < 16; ++rl) {
        const int r = blk_i * 16 + rl;
        std::vector<std::int8_t> q(H, 0);
        std::vector<float> ds(H / 16, 0.f);
        if (r < model::UNPADDED_VOCAB) {
          const auto* row = reinterpret_cast<const std::uint8_t*>(ot.data) + std::size_t(r) * rb;
          for (int sb = 0; sb < H / 256; ++sb) {
            const std::uint8_t* b = row + 210 * sb;
            const std::uint8_t* ql = b;
            const std::uint8_t* qh = b + 128;
            const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(b + 192);
            std::uint16_t dh;
            std::memcpy(&dh, b + 208, 2);
            __half_raw hr;
            hr.x = dh;
            const float d = __half2float(__half(hr));
            std::int8_t* qq = q.data() + 256 * sb;
            for (int n = 0; n < 2; ++n, qq += 128, ql += 64, qh += 32) {
              for (int l = 0; l < 32; ++l) {
                qq[l] = std::int8_t(int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                qq[l + 32] = std::int8_t(int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                qq[l + 64] = std::int8_t(int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                qq[l + 96] = std::int8_t(int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
              }
            }
            for (int j = 0; j < 16; ++j) ds[16 * sb + j] = d * float(sc[j]);
          }
        }
        qbytes.insert(qbytes.end(), reinterpret_cast<const std::uint8_t*>(q.data()),
                      reinterpret_cast<const std::uint8_t*>(q.data()) + H);
        for (float fv : ds) {
          std::uint16_t h2[2];
          std::memcpy(h2, &fv, 4);
          scales.push_back(h2[0]);
          scales.push_back(h2[1]);
        }
      }
    }
    w.unembed.w = model::rm(upload(qbytes.data(), qbytes.size()), DType::U8, {V_pad, H});
    w.unembed.blk_off = model::rm(upload(off.data(), off.size() * 8), DType::I64, {n_blks});
    w.unembed.blk_soff = model::rm(upload(soff.data(), soff.size() * 4), DType::I32, {n_blks});
    w.unembed.q8_scale = model::rm(upload(scales.data(), scales.size() * 2), DType::FP16, {std::int32_t(scales.size())});
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return w;
}

std::vector<ActT> make_synthetic_taps(int n, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 0.25f);
  std::vector<ActT> h(std::size_t(n) * 4 * kd::H);
  for (auto& v : h) v = act::host_from_float(nd(rng));
  return h;
}

}  // namespace

int main() {
  const auto model_dir = find_model_dir();
  if (model_dir.empty()) {
    std::cout << "SKIP: INGINE_MODEL_DIR not set and default Inkling-2-Mini dir missing\n";
    return 0;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  model::Weights w = load_draft_only_weights(model_dir, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Build 4 distinct synthetic contexts of varying lengths (16, 40, 24, 64) + 3 rounds of taps each.
  constexpr int S_MAX = 4;
  const int pref_lens[S_MAX] = {16, 40, 24, 64};
  const std::int32_t anchors0[S_MAX] = {100, 250, 512, 1024};
  const int round_acc[S_MAX] = {0, 1, 2, 3};  // ncommit = {1, 2, 3, 4}
  const std::int32_t round_next[S_MAX] = {77, 88, 99, 111};

  std::vector<DeviceBuffer> pref_taps;
  std::vector<DeviceBuffer> step_taps;
  for (int s = 0; s < S_MAX; ++s) {
    auto hp = make_synthetic_taps(pref_lens[s], 1000u + s);
    pref_taps.emplace_back(hp.size() * sizeof(ActT));
    CUDA_CHECK(cudaMemcpyAsync(pref_taps.back().p, hp.data(), hp.size() * sizeof(ActT), cudaMemcpyHostToDevice, stream));

    auto hs = make_synthetic_taps(4, 2000u + s);
    step_taps.emplace_back(hs.size() * sizeof(ActT));
    CUDA_CHECK(cudaMemcpyAsync(step_taps.back().p, hs.data(), hs.size() * sizeof(ActT), cudaMemcpyHostToDevice, stream));
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // 1) Reference outputs from single-stream Drafter (small commit uses FP8 wkv_all_, prefill uses BF16 wkv_ctx)
  std::vector<std::array<std::int32_t, kd::GAMMA>> ref_tok0(S_MAX), ref_tok1(S_MAX);
  std::vector<std::array<float, kd::GAMMA>> ref_conf0(S_MAX), ref_conf1(S_MAX);
  {
    model::Drafter ref(model_dir, w, 16384, stream);
    ref.capture();
    for (int s = 0; s < S_MAX; ++s) {
      ref.reset();
      // Commit in <= 8 chunks so reference ALSO uses FP8 wkv_all_ (exact apples-to-apples check!)
      for (int r0 = 0; r0 < pref_lens[s]; r0 += 8) {
        const int k = std::min(8, pref_lens[s] - r0);
        ref.commit(static_cast<const ActT*>(pref_taps[s].p) + std::size_t(r0) * 4 * kd::H, k, r0);
      }
      auto d0 = ref.draft(anchors0[s], pref_lens[s], 0.f);
      CUDA_CHECK(cudaMemcpyAsync(ref_tok0[s].data(), d0.tokens, kd::GAMMA * 4, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(ref_conf0[s].data(), d0.conf, kd::GAMMA * 4, cudaMemcpyDeviceToHost, stream));

      const int ncom = round_acc[s] + 1;
      ref.commit(static_cast<const ActT*>(step_taps[s].p), ncom, pref_lens[s]);
      auto d1 = ref.draft(round_next[s], pref_lens[s] + ncom, 0.f);
      CUDA_CHECK(cudaMemcpyAsync(ref_tok1[s].data(), d1.tokens, kd::GAMMA * 4, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(ref_conf1[s].data(), d1.conf, kd::GAMMA * 4, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }  // Drafter destroyed -> frees ~900 MiB VRAM

  // 2) BatchDrafter parity: full draft_len=7 first (all 28 tokens), then production draft_len=3 (first 3 tokens + bench)
  for (int dlen : {7, 3}) {
    model::BatchDrafter bd(model_dir, w, S_MAX, 262144, 16384, stream, model::BatchDrafter::SHORTLIST_ROWS, {}, 4, dlen);
    bd.capture();

    for (int s = 0; s < S_MAX; ++s) {
      bd.commit_slot(s, static_cast<const ActT*>(pref_taps[s].p), pref_lens[s], 0);
    }

    // Test S=1 per slot
    for (int s = 0; s < S_MAX; ++s) {
      const int active[1] = {s};
      const std::int32_t anc[1] = {anchors0[s]};
      const int pos[1] = {pref_lens[s]};
      bd.draft_slots(active, anc, pos, 0.f);
      std::array<std::int32_t, kd::GAMMA> got_tok{};
      std::array<float, kd::GAMMA> got_conf{};
      CUDA_CHECK(cudaMemcpyAsync(got_tok.data(), bd.slot_tokens_dev(s), dlen * 4, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(got_conf.data(), bd.slot_conf_dev(s), dlen * 4, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      for (int i = 0; i < dlen; ++i) {
        if (got_tok[i] != ref_tok0[s][i]) {
          std::cerr << "FAIL dlen=" << dlen << " S=1 slot " << s << " step " << i << ": got " << got_tok[i]
                    << " vs ref " << ref_tok0[s][i] << "\n";
          return 1;
        }
        if (std::fabs(got_conf[i] - ref_conf0[s][i]) > 1e-4f) {
          std::cerr << "FAIL dlen=" << dlen << " S=1 conf slot " << s << " step " << i << "\n";
          return 1;
        }
      }
    }
    std::cout << "PASS: BatchDrafter(draft_len=" << dlen << ") S=1 matches Drafter (" << (4 * dlen) << "/"
              << (4 * dlen) << " tokens bit-identical)\n";

    // Test S=4 simultaneous draft_slots
    {
      const int active[S_MAX] = {0, 1, 2, 3};
      bd.draft_slots(active, anchors0, pref_lens, 0.f);
      for (int s = 0; s < S_MAX; ++s) {
        std::array<std::int32_t, kd::GAMMA> got_tok{};
        CUDA_CHECK(cudaMemcpyAsync(got_tok.data(), bd.slot_tokens_dev(s), dlen * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < dlen; ++i) {
          if (got_tok[i] != ref_tok0[s][i]) {
            std::cerr << "FAIL dlen=" << dlen << " S=4 draft slot " << s << " step " << i << "\n";
            return 1;
          }
        }
      }
    }
    std::cout << "PASS: BatchDrafter(draft_len=" << dlen << ") S=4 draft_slots matches Drafter (" << (4 * dlen) << "/"
              << (4 * dlen) << " tokens bit-identical)\n";

    // Test fused batch_greedy_accept + step_commit_and_draft
    {
      const int slot_row0_h[S_MAX] = {0, 2, 5, 9};
      const int slot_m_h[S_MAX] = {2, 3, 4, 4};
      const int active_h[S_MAX] = {0, 1, 2, 3};
      std::int64_t slot_pos_h[S_MAX];
      for (int s = 0; s < S_MAX; ++s) slot_pos_h[s] = pref_lens[s];

      DeviceBuffer packed_taps(std::size_t(16) * 4 * kd::H * sizeof(ActT));
      for (int s = 0; s < S_MAX; ++s) {
        CUDA_CHECK(cudaMemcpyAsync(static_cast<ActT*>(packed_taps.p) + std::size_t(slot_row0_h[s]) * 4 * kd::H,
                                   step_taps[s].p, std::size_t(slot_m_h[s]) * 4 * kd::H * sizeof(ActT),
                                   cudaMemcpyDeviceToDevice, stream));
      }

      std::vector<std::int32_t> target_h(16, 0);
      for (int s = 0; s < S_MAX; ++s) {
        for (int j = 0; j < round_acc[s]; ++j) target_h[slot_row0_h[s] + j] = ref_tok0[s][j];
        target_h[slot_row0_h[s] + round_acc[s]] = round_next[s];
      }

      DeviceBuffer target_d(16 * 4), active_d(S_MAX * 4), row0_d(S_MAX * 4), m_d(S_MAX * 4), pos_d(S_MAX * 8);
      DeviceBuffer out_acc_d(S_MAX * 4), out_com_d(S_MAX * 4), out_next_d(S_MAX * 4);
      CUDA_CHECK(cudaMemcpyAsync(target_d.p, target_h.data(), 16 * 4, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(active_d.p, active_h, S_MAX * 4, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(row0_d.p, slot_row0_h, S_MAX * 4, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(m_d.p, slot_m_h, S_MAX * 4, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(pos_d.p, slot_pos_h, S_MAX * 8, cudaMemcpyHostToDevice, stream));

      bd.upload_active_desc(active_h);
      kd::batch_greedy_accept(static_cast<const std::int32_t*>(target_d.p), static_cast<const std::int32_t*>(active_d.p),
                              static_cast<const std::int32_t*>(row0_d.p), static_cast<const std::int32_t*>(m_d.p),
                              static_cast<const std::int64_t*>(pos_d.p), static_cast<const ActT*>(packed_taps.p),
                              bd.slots_dev(), S_MAX, static_cast<std::int32_t*>(out_acc_d.p),
                              static_cast<std::int32_t*>(out_com_d.p), static_cast<std::int32_t*>(out_next_d.p), stream);
      bd.step_commit_and_draft(active_h, 70, /*desc_already_uploaded=*/true);

      std::int32_t got_acc[S_MAX], got_com[S_MAX], got_next[S_MAX];
      CUDA_CHECK(cudaMemcpyAsync(got_acc, out_acc_d.p, S_MAX * 4, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(got_com, out_com_d.p, S_MAX * 4, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(got_next, out_next_d.p, S_MAX * 4, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));

      for (int s = 0; s < S_MAX; ++s) {
        if (got_acc[s] != round_acc[s] || got_com[s] != round_acc[s] + 1 || got_next[s] != round_next[s]) {
          std::cerr << "FAIL dlen=" << dlen << " batch_greedy_accept slot " << s << "\n";
          return 1;
        }
        std::array<std::int32_t, kd::GAMMA> got_tok{};
        CUDA_CHECK(cudaMemcpyAsync(got_tok.data(), bd.slot_tokens_dev(s), dlen * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < dlen; ++i) {
          if (got_tok[i] != ref_tok1[s][i]) {
            std::cerr << "FAIL dlen=" << dlen << " step_commit_and_draft slot " << s << " step " << i << "\n";
            return 1;
          }
        }
      }
    }
    std::cout << "PASS: BatchDrafter(draft_len=" << dlen << ") S=4 batch_greedy_accept + step_commit_and_draft bit-identical\n";

    // Permuted subset {3, 1}
    {
      const int sub[2] = {3, 1};
      const std::int32_t anc[2] = {round_next[3], round_next[1]};
      const int pos[2] = {pref_lens[3] + round_acc[3] + 1, pref_lens[1] + round_acc[1] + 1};
      bd.draft_slots(sub, anc, pos, 0.f);
      for (int i = 0; i < 2; ++i) {
        const int s = sub[i];
        std::array<std::int32_t, kd::GAMMA> got_tok{};
        CUDA_CHECK(cudaMemcpyAsync(got_tok.data(), bd.slot_tokens_dev(s), dlen * 4, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int k = 0; k < dlen; ++k) {
          if (got_tok[k] != ref_tok1[s][k]) {
            std::cerr << "FAIL dlen=" << dlen << " permuted subset S=2 slot " << s << " step " << k << "\n";
            return 1;
          }
        }
      }
    }
    std::cout << "PASS: BatchDrafter(draft_len=" << dlen << ") permuted subset {3, 1} bit-identical\n";

    // 3) Microbenchmark T_draft (step_commit_and_draft graph) for S = 1..4 at P = 4000 and P = 16000
    std::cout << "\n--- BatchDrafter(draft_len=" << dlen << ") T_draft microbenchmark (CUDA graph, 50 iters) ---\n";
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    for (int depth : {4000, 16000}) {
      for (int S = 1; S <= S_MAX; ++S) {
        std::vector<int> act(S);
        for (int s = 0; s < S; ++s) {
          act[s] = s;
          kd::set_commit(bd.slot_params_dev(s), static_cast<const ActT*>(step_taps[s].p), 2, depth - 2, stream);
          kd::set_draft_inputs(bd.slot_params_dev(s), anchors0[s], depth, 0.f, stream);
        }
        bd.upload_active_desc(act);
        for (int w_it = 0; w_it < 10; ++w_it) bd.step_commit_and_draft(act, depth, true);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(ev0, stream));
        constexpr int N_IT = 50;
        for (int it = 0; it < N_IT; ++it) bd.step_commit_and_draft(act, depth, true);
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        std::cout << "dlen=" << dlen << " depth=" << depth << " S=" << S
                  << ": step_commit_and_draft = " << (ms / N_IT) << " ms (" << (ms / N_IT / S) << " ms/slot)\n";
      }
    }
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
  }
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}

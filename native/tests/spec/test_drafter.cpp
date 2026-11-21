// Drafter smoke test with the real mtp.safetensors (~1.2 GB of GPU memory at peak) and a small synthetic
// target (embedding/unembedding of 4096 tokens): eager and graph drafts agree bit for bit, commits through
// the per-row-count graphs equal eager commits, the outputs are well formed (tokens are each row's first
// maximum, -inf outside 1024 candidates, conf = 1 / sum exp(l - max)), and sampled drafts are reproducible
// per seed.
#include "common/test_util.h"
#include "kernels/dense/dense.h"
#include "model/drafter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <vector>

using namespace ingine::test;
using ingine::DeviceBuffer;
using ingine::DType;
namespace model = ingine::model;

namespace {

constexpr int H = model::HIDDEN, V = 4096, MAX_CTX = 4096;
int g_failures = 0;

void expect(bool ok, const char* what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) ++g_failures;
}

struct DraftOut {
  std::vector<int32_t> tokens;
  std::vector<float> logits, conf;
  bool operator==(const DraftOut& o) const {
    return tokens == o.tokens && conf == o.conf &&
           std::equal(logits.begin(), logits.end(), o.logits.begin(), o.logits.end(),
                      [](float a, float b) { return a == b || (std::isinf(a) && std::isinf(b) && a == b); });
  }
};

DraftOut fetch(const model::Drafter::Draft& d, cudaStream_t s) {
  INGINE_TEST_CUDA(cudaStreamSynchronize(s));
  DraftOut o{std::vector<int32_t>(7), std::vector<float>(size_t(7) * V), std::vector<float>(7)};
  INGINE_TEST_CUDA(cudaMemcpy(o.tokens.data(), d.tokens, 7 * 4, cudaMemcpyDeviceToHost));
  INGINE_TEST_CUDA(cudaMemcpy(o.logits.data(), d.logits, size_t(7) * V * 4, cudaMemcpyDeviceToHost));
  INGINE_TEST_CUDA(cudaMemcpy(o.conf.data(), d.conf, 7 * 4, cudaMemcpyDeviceToHost));
  return o;
}

bool well_formed(const DraftOut& o, bool greedy) {
  bool ok = true;
  for (int i = 0; i < 7; ++i) {
    const float* row = o.logits.data() + size_t(i) * V;
    int finite = 0, first_max = -1;
    double m = -INFINITY, s = 0;
    for (int x = 0; x < V; ++x)
      if (std::isfinite(row[x])) {
        ++finite;
        if (row[x] > m) {
          m = row[x];
          first_max = x;
        }
      }
    for (int x = 0; x < V; ++x)
      if (std::isfinite(row[x])) s += std::exp(double(row[x]) - m);
    ok &= finite == 1024;
    ok &= o.tokens[i] >= 0 && o.tokens[i] < V && std::isfinite(row[o.tokens[i]]);
    if (greedy) ok &= o.tokens[i] == first_max;
    ok &= std::fabs(o.conf[i] - 1.0 / s) <= 1e-5 / s && o.conf[i] > 0.f && o.conf[i] <= 1.f;
  }
  return ok;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  const std::filesystem::path dir = INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "mtp.safetensors")) {
    std::printf("skip: %s/mtp.safetensors not found\n", dir.c_str());
    return kSkip;
  }
  size_t free_b = 0, total_b = 0;
  INGINE_TEST_CUDA(cudaMemGetInfo(&free_b, &total_b));
  if (free_b < (size_t(2) << 30)) {
    std::printf("skip: %.2f GB of GPU memory free, the test needs 2 GB\n", double(free_b) / (1 << 30));
    return kSkip;
  }
  cudaStream_t stream;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // synthetic target: embedding, its norm, FP8-row unembedding of V tokens
  Rng r(5);
  model::Weights w;
  {
    std::vector<__nv_bfloat16> embed(size_t(V) * H), unembed(size_t(V) * H);
    for (auto& x : embed) x = float_to_bf16(float(r.normal()));
    for (auto& x : unembed) x = float_to_bf16(float(r.normal() * 0.05));
    std::vector<float> en(H);
    for (auto& x : en) x = float(r.uniform(0.5, 1.5));
    w.storage.reserve(4);  // references below stay valid
    auto& e = w.storage.emplace_back(embed.size() * 2);
    e.copy_from_host(embed.data(), embed.size() * 2);
    auto& n = w.storage.emplace_back(en.size() * 4);
    n.copy_from_host(en.data(), en.size() * 4);
    DeviceBuffer u16(unembed.size() * 2);
    u16.copy_from_host(unembed.data(), unembed.size() * 2);
    auto& q = w.storage.emplace_back(unembed.size());
    auto& sc = w.storage.emplace_back(size_t(V) * 4);
    ingine::kernels::dense::fp8_rows(static_cast<const __nv_bfloat16*>(u16.p), V, H, q.p, static_cast<float*>(sc.p),
                                     nullptr);
    INGINE_TEST_CUDA(cudaDeviceSynchronize());
    w.embed = model::rm(w.storage[0].p, DType::BF16, {V, H});
    w.embed_norm = model::rm(w.storage[1].p, DType::FP32, {H});
    w.unembed.w = model::rm(w.storage[2].p, DType::FP8_E4M3FN, {V, H});
    w.unembed.scale = model::rm(w.storage[3].p, DType::FP32, {V});
    w.dense_fp8 = true;
  }

  const int rows = 300;
  std::vector<__nv_bfloat16> feats(size_t(rows) * 4 * H);
  for (auto& x : feats) x = float_to_bf16(float(r.normal()));
  DeviceBuffer d_feats(feats.size() * 2);
  d_feats.copy_from_host(feats.data(), feats.size() * 2);
  const auto* fp = static_cast<const __nv_bfloat16*>(d_feats.p);

  {
    model::Drafter d(dir, w, MAX_CTX, stream);
    const int anchor = 17;
    // A: one eager commit of all rows; B: rows - 5 rows, then 5 rows (a commit graph once captured)
    auto commit_a = [&] { d.commit(fp, rows, 0); };
    auto commit_b = [&] {
      d.commit(fp, rows - 5, 0);
      d.commit(fp + size_t(rows - 5) * 4 * H, 5, rows - 5);
    };
    commit_a();
    expect(d.length() == rows, "length after an eager commit");
    const DraftOut eager_a = fetch(d.draft(anchor, rows, 0.f), stream);
    expect(well_formed(eager_a, true), "eager greedy draft is well formed");
    commit_b();
    const DraftOut eager_b = fetch(d.draft(anchor, rows, 0.f), stream);
    float diff = 0.f;
    for (size_t i = 0; i < eager_a.logits.size(); ++i)
      if (std::isfinite(eager_a.logits[i]) && std::isfinite(eager_b.logits[i]))
        diff = std::max(diff, std::fabs(eager_a.logits[i] - eager_b.logits[i]));
    std::printf("one-shot vs split commit (cuBLASLt shapes differ): max logit diff %.4g, tokens %s\n", diff,
                eager_a.tokens == eager_b.tokens ? "equal" : "differ");

    d.capture();
    expect(d.length() == 0, "capture resets the context");
    commit_a();
    const DraftOut graph_a = fetch(d.draft(anchor, rows, 0.f), stream);
    expect(graph_a == eager_a, "graph draft == eager draft");
    commit_b();
    const DraftOut graph_b = fetch(d.draft(anchor, rows, 0.f), stream);
    expect(graph_b == eager_b, "draft after a commit graph == draft after the eager commit");
    const DraftOut& eager = eager_a;

    d.set_seed(123);
    const DraftOut s1 = fetch(d.draft(anchor, rows, 0.8f), stream);
    d.set_seed(123);
    const DraftOut s2 = fetch(d.draft(anchor, rows, 0.8f), stream);
    const DraftOut s3 = fetch(d.draft(anchor, rows, 0.8f), stream);
    expect(well_formed(s1, false), "sampled draft is well formed");
    expect(s1 == s2, "sampled drafts reproduce per seed");
    expect(std::equal(s1.logits.begin(), s1.logits.begin() + V, s3.logits.begin()),
           "step 0 logits do not depend on the noise");
    std::printf("greedy drafts:");
    for (int t : eager.tokens) std::printf(" %d", t);
    std::printf("\nsampled drafts:");
    for (int t : s1.tokens) std::printf(" %d", t);
    std::printf(" / next draw:");
    for (int t : s3.tokens) std::printf(" %d", t);
    std::printf("\n");
  }
  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}

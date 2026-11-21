// Prefill kernel timings at production sizes (T = 2048 chunk): attention over a 16K / 64K / 255K full cache
// (BF16 and FP8) and a sliding visit, grouped MoE (NVFP4, MXFP8, all 128 experts hit), the dense projection
// GEMMs, and the per-visit row kernels. Random data; < 2 GB of GPU memory. Runs only with INGINE_BENCH=1.
#include "kernels/prefill/prefill.h"
#include "common/test_util.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

using namespace ingine::test;
namespace pk = ingine::kernels::prefill;

namespace {

__global__ void fill_bf16(uint16_t* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
    const float u = (h & 0xffffff) / 16777216.f - 0.5f;
    p[i] = __bfloat16_as_ushort(__float2bfloat16(u * scale));
  }
}
__global__ void fill_bytes(uint8_t* p, size_t n, uint32_t seed, uint8_t mask, uint8_t base) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
    p[i] = uint8_t((h & mask) | base);
  }
}

template <class T>
T* dalloc(size_t n) {
  T* p = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&p, n * sizeof(T) + 16));
  INGINE_TEST_CUDA(cudaMemset(p, 0, n * sizeof(T) + 16));
  return p;
}

float time_ms(const std::function<void()>& f, int reps = 5) {
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  f();
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  std::vector<float> t;
  for (int i = 0; i < reps; ++i) {
    cudaEventRecord(a);
    f();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    t.push_back(ms);
  }
  INGINE_TEST_CUDA(cudaGetLastError());
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

}  // namespace

int main() {
  const char* on = std::getenv("INGINE_BENCH");
  if (!on || std::string(on) != "1") {
    std::printf("skip: set INGINE_BENCH=1 to run the prefill benchmark\n");
    return kSkip;
  }
  if (skip_without_gpu()) return kSkip;
  const int T = 2048, H = 2304;
  const int ld = 8664;
  auto* q = dalloc<uint16_t>(size_t(T) * 3072);
  auto* rel = dalloc<uint16_t>(size_t(T) * 24 * 1024);
  auto* proj = dalloc<uint16_t>(size_t(T) * ld);
  auto* out = dalloc<uint16_t>(size_t(T) * 3072);
  fill_bf16<<<1024, 256>>>(q, size_t(T) * 3072, 1, 4.f);
  fill_bf16<<<1024, 256>>>(rel, size_t(T) * 24 * 1024, 2, 2.f);
  fill_bf16<<<1024, 256>>>(proj, size_t(T) * ld, 3, 2.f);

  // ---- attention, full cache
  const int max_ctx = 262144;
  auto* kc = dalloc<uint8_t>(size_t(max_ctx) * 1024);  // FP8 cache (256 MiB) / BF16 view for <= 128K
  auto* vc = dalloc<uint8_t>(size_t(max_ctx) * 1024);
  fill_bytes<<<4096, 256>>>(kc, size_t(max_ctx) * 1024, 4, 0x3f, 0x10);
  fill_bytes<<<4096, 256>>>(vc, size_t(max_ctx) * 1024, 5, 0x3f, 0x10);
  pk::AttnParams p;
  p.q = reinterpret_cast<const __nv_bfloat16*>(q);
  p.rel = reinterpret_cast<const __nv_bfloat16*>(rel);
  p.proj = reinterpret_cast<const __nv_bfloat16*>(proj), p.ld = ld;
  p.out = reinterpret_cast<__nv_bfloat16*>(out);
  p.T = T;
  for (bool fp8 : {false, true}) {
    for (int ctx : {16384, 65536, 131072, 262144}) {
      if (!fp8 && ctx > 131072) continue;
      if (!fp8) fill_bf16<<<4096, 256>>>(reinterpret_cast<uint16_t*>(kc), size_t(ctx) * 1024, 6, 0.3f),
                fill_bf16<<<4096, 256>>>(reinterpret_cast<uint16_t*>(vc), size_t(ctx) * 1024, 7, 2.f);
      p.k = kc, p.v = vc, p.kv_fp8 = fp8, p.kpos = nullptr, p.bounds = nullptr;
      p.pos0 = ctx - T, p.L = ctx, p.window = 0;
      const float ms = time_ms([&] { pk::attention(p, nullptr); }, 3);
      // causal pairs: T queries over pos0 + t + 1 keys, 24 heads, 4 * 128 flop per pair
      const double flop = 24.0 * 512 * (double(T) * (ctx - T) + 0.5 * double(T) * T);
      std::printf("attention full %s ctx=%6d: %8.3f ms  %6.1f TFLOP/s\n", fp8 ? "fp8 " : "bf16", ctx, ms,
                  flop / ms / 1e9);
    }
  }
  // ---- attention, sliding (ring full + chunk)
  {
    const int L = 1024 + T, kr = (L + 63) / 64 * 64;
    auto* rk = dalloc<uint16_t>(size_t(1024) * 1024);
    auto* rv = dalloc<uint16_t>(size_t(1024) * 1024);
    auto* rp = dalloc<int64_t>(1024);
    std::vector<int64_t> pos(1024);
    const int pos0 = 50000;
    for (int s = 0; s < 1024; ++s) pos[s] = pos0 - 1024 + ((s - pos0 % 1024 + 1024) % 1024);
    INGINE_TEST_CUDA(cudaMemcpy(rp, pos.data(), 1024 * 8, cudaMemcpyHostToDevice));
    auto* krows = dalloc<uint16_t>(size_t(L) * 1024);
    auto* vrows = dalloc<uint16_t>(size_t(L) * 1024);
    fill_bf16<<<4096, 256>>>(krows, size_t(L) * 1024, 8, 0.3f);
    fill_bf16<<<4096, 256>>>(vrows, size_t(L) * 1024, 9, 2.f);
    auto* kp = dalloc<int>(kr);
    auto* bounds = dalloc<int4>(kr / 64);
    const float gather = time_ms([&] {
      pk::ring_gather(reinterpret_cast<const __nv_bfloat16*>(rk), reinterpret_cast<const __nv_bfloat16*>(rv), rp, pos0,
                      T, reinterpret_cast<__nv_bfloat16*>(krows), reinterpret_cast<__nv_bfloat16*>(vrows), kp, kr,
                      nullptr);
      pk::key_tile_bounds(kp, L, bounds, nullptr);
    });
    p.k = krows, p.v = vrows, p.kv_fp8 = false, p.kpos = kp, p.bounds = bounds, p.L = L, p.pos0 = pos0, p.window = 1024;
    const float ms = time_ms([&] { pk::attention(p, nullptr); });
    const double flop = 24.0 * 512 * double(T) * 1024;
    std::printf("attention sliding: %8.3f ms  %6.1f TFLOP/s (window pairs)  gather+bounds %.3f ms\n", ms,
                flop / ms / 1e9, gather);
  }
  cudaFree(kc), cudaFree(vc);

  // ---- MoE
  {
    auto* x = dalloc<uint16_t>(size_t(T) * H);
    fill_bf16<<<1024, 256>>>(x, size_t(T) * H, 10, 3.f);
    std::vector<float> logits(size_t(T) * 128);
    Rng rng(5);
    for (auto& v : logits) v = float(rng.normal());
    auto* dl = dalloc<float>(logits.size());
    INGINE_TEST_CUDA(cudaMemcpy(dl, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice));
    auto* bias = dalloc<float>(128);
    auto* idx = dalloc<int>(size_t(T) * 8);
    auto* wts = dalloc<float>(size_t(T) * 8);
    pk::MoeWork w{dalloc<int>(128), dalloc<int>(1), dalloc<int4>(pk::moe_max_tiles(T)), dalloc<int>(size_t(T) * 8),
                  dalloc<int>(size_t(T) * 8)};
    auto* act = dalloc<uint16_t>(size_t(T) * 8 * 1152);
    auto* down = dalloc<uint16_t>(size_t(T) * 8 * H);
    auto* sum = dalloc<uint16_t>(size_t(T) * H);
    const float route = time_ms([&] {
      cudaMemsetAsync(w.counts, 0, 128 * 4);
      pk::moe_topk(dl, bias, T, idx, wts, w.counts, nullptr);
      pk::moe_route(idx, T, w, nullptr);
    });
    for (int fmt : {0, 1}) {
      const size_t n13 = size_t(128) * 2304 * (fmt == 0 ? 1152 : 2304), n2 = n13 / 2;
      auto* c13 = dalloc<uint8_t>(n13);
      auto* s13 = dalloc<uint8_t>(n13 / (fmt == 0 ? 8 : 32));
      auto* c2 = dalloc<uint8_t>(n2);
      auto* s2 = dalloc<uint8_t>(n2 / (fmt == 0 ? 8 : 32));
      fill_bytes<<<4096, 256>>>(c13, n13, 11, fmt == 0 ? 0xff : 0x3f, fmt == 0 ? 0 : 0x10);
      fill_bytes<<<4096, 256>>>(c2, n2, 12, fmt == 0 ? 0xff : 0x3f, fmt == 0 ? 0 : 0x10);
      fill_bytes<<<4096, 256>>>(s13, n13 / (fmt == 0 ? 8 : 32), 13, 0x07, fmt == 0 ? 0x30 : 0x78);
      fill_bytes<<<4096, 256>>>(s2, n2 / (fmt == 0 ? 8 : 32), 14, 0x07, fmt == 0 ? 0x30 : 0x78);
      std::vector<float> ones(128, 0.02f);
      auto* f = dalloc<float>(128);
      INGINE_TEST_CUDA(cudaMemcpy(f, ones.data(), 512, cudaMemcpyHostToDevice));
      pk::MoeExperts e{fmt, c13, s13, fmt == 0 ? f : nullptr, c2, s2, fmt == 0 ? f : nullptr};
      const float up = time_ms([&] {
        pk::moe_gate_up(e, reinterpret_cast<const __nv_bfloat16*>(x), T, w, reinterpret_cast<__nv_bfloat16*>(act), nullptr);
      });
      const float dn = time_ms([&] {
        pk::moe_down(e, reinterpret_cast<const __nv_bfloat16*>(act), T, w, reinterpret_cast<__nv_bfloat16*>(down), nullptr);
      });
      const float cb = time_ms([&] {
        pk::moe_combine(reinterpret_cast<const __nv_bfloat16*>(down), wts, T, reinterpret_cast<__nv_bfloat16*>(sum), nullptr);
      });
      const double fu = 2.0 * T * 8 * 2304 * 2304, fd = 2.0 * T * 8 * 2304 * 1152;
      std::printf("moe %s: gate_up %.3f ms (%.1f TFLOP/s)  down %.3f ms (%.1f TFLOP/s)  combine %.3f ms  route %.3f ms\n",
                  fmt == 0 ? "nvfp4" : "mxfp8", up, fu / up / 1e9, dn, fd / dn / 1e9, cb, route);
      cudaFree(c13), cudaFree(s13), cudaFree(c2), cudaFree(s2), cudaFree(f);
    }
  }

  // ---- dense GEMMs of one visit (full layer, three inputs) + wo
  {
    auto* xin = dalloc<uint16_t>(size_t(3) * T * H);
    auto* wproj = dalloc<uint16_t>(size_t(ld) * H);
    auto* wo = dalloc<uint16_t>(size_t(H) * 3072);
    auto* pout = dalloc<uint16_t>(size_t(T) * ld);
    auto* oout = dalloc<uint16_t>(size_t(T) * H);
    auto* attn = dalloc<uint16_t>(size_t(T) * 3072);
    fill_bf16<<<4096, 256>>>(xin, size_t(3) * T * H, 15, 2.f);
    fill_bf16<<<4096, 256>>>(wproj, size_t(ld) * H, 16, 0.05f);
    fill_bf16<<<4096, 256>>>(wo, size_t(H) * 3072, 17, 0.05f);
    fill_bf16<<<4096, 256>>>(attn, size_t(T) * 3072, 18, 1.f);
    pk::Gemm gemm;
    auto bf = [](uint16_t* p) { return reinterpret_cast<const __nv_bfloat16*>(p); };
    const float ms_proj = time_ms([&] {
      gemm.run(bf(xin), H, bf(wproj), H, pout, ld, false, T, 6552, H, nullptr);
      gemm.run(bf(xin) + size_t(T) * H, H, bf(wproj) + size_t(6552) * H, H, pout + 6552, ld, false, T, 1024, H, nullptr);
      gemm.run(bf(xin) + size_t(2) * T * H, H, bf(wproj) + size_t(7576) * H, H, pout + 7576, ld, false, T, 1024, H, nullptr);
      gemm.run(bf(xin) + size_t(T) * H, H, bf(wproj) + size_t(8600) * H, H, pout + 8600, ld, false, T, 32, H, nullptr);
      gemm.run(bf(xin) + size_t(2) * T * H, H, bf(wproj) + size_t(8632) * H, H, pout + 8632, ld, false, T, 32, H, nullptr);
    });
    const float ms_single = time_ms([&] { gemm.run(bf(xin), H, bf(wproj), H, pout, 8600, false, T, 8600, H, nullptr); });
    const float ms_wo = time_ms([&] { gemm.run(bf(attn), 3072, bf(wo), 3072, oout, H, false, T, H, 3072, nullptr); });
    std::printf("dense: proj (5 GEMMs) %.3f ms (%.1f TFLOP/s), proj single %.3f ms (%.1f TFLOP/s), wo %.3f ms (%.1f TFLOP/s)\n",
                ms_proj, 2.0 * T * ld * H / ms_proj / 1e9, ms_single, 2.0 * T * 8600 * H / ms_single / 1e9, ms_wo,
                2.0 * T * H * 3072 / ms_wo / 1e9);
  }
  return 0;
}

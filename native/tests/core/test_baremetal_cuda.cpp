// Bare-metal Phase A CUDA verification:
//   1. Non-destructive 2 MiB THP prefault before cudaHostRegister (HostRegion / PinnedHostBuffer / MappedHostBuffer)
//   2. Bit-exact H2D / D2H DMA round-trip and cudaPointerGetAttributes validation
//   3. Cross-backend CPU vs GPU EXL3_2 / EXL3_3 dequantization and GEMV parity on live CUDA device

#include "common/test_util.h"
#include "core/act.h"
#include "core/arena.h"
#include "core/cpu_kernels.h"
#include "core/cpu_pool.h"
#include "core/host_memory.h"
#include "core/hugepage.h"
#include "kernels/moe/moe.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

bool is_2m_aligned(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) % ingine::kHugePageAlignBytes) == 0;
}

std::uint8_t byte_for(std::size_t off, std::uint64_t seed) {
  return static_cast<std::uint8_t>(((off * 1315423911ull + seed) >> 8) & 0xffu);
}

void fill_bytes(void* ptr, std::size_t n, std::uint64_t seed) {
  auto* p = static_cast<std::uint8_t*>(ptr);
  for (std::size_t i = 0; i < n; ++i) {
    p[i] = byte_for(i, seed);
  }
}

void check_bytes(const void* ptr, std::size_t n, std::uint64_t seed) {
  const auto* p = static_cast<const std::uint8_t*>(ptr);
  for (std::size_t i = 0; i < n; ++i) {
    if (p[i] != byte_for(i, seed)) {
      std::fprintf(stderr, "byte mismatch at off=%zu: got=0x%02x expected=0x%02x\n",
                   i, unsigned(p[i]), unsigned(byte_for(i, seed)));
      std::exit(1);
    }
  }
}

void test_cuda_hugepage_host_buffers() {
  std::printf("[1/2] Testing CUDA host-memory 2 MiB THP registration & async DMA round-trip...\n");
  constexpr std::size_t kEightMiB = 8ull << 20;
  constexpr std::size_t kFourMiB = 4ull << 20;
  const bool hp_env = ingine::hugepages_enabled();

  // Pre-filled HostRegion: ExpertCache fills WARM RAM BEFORE calling register_with_cuda().
  // Verify prefault_hugepage_stride_nondestructive preserves every single byte across cudaHostRegister().
  ingine::HostRegion prefilled(kEightMiB);
  INGINE_CHECK(prefilled.data() != nullptr);
  INGINE_CHECK(is_2m_aligned(prefilled.data()));
  INGINE_CHECK(prefilled.hugepage_advised() == hp_env);
  fill_bytes(prefilled.data(), kEightMiB, 0xCAFEBABE1234ull);
  prefilled.register_with_cuda();
  INGINE_CHECK(prefilled.registered());
  check_bytes(prefilled.data(), kEightMiB, 0xCAFEBABE1234ull);

  cudaPointerAttributes attr{};
  INGINE_TEST_CUDA(cudaPointerGetAttributes(&attr, prefilled.data()));
  INGINE_CHECK(attr.type == cudaMemoryTypeHost);
  const auto sm_pre = ingine::query_range_smaps_hugepages(prefilled.data(), kEightMiB);
  INGINE_CHECK(sm_pre.readable && sm_pre.rss_bytes >= kEightMiB);
  std::printf("  HostRegion 8MiB (pre-filled + cudaHostRegister): rss=%.1f MiB AnonHugePages=%.1f MiB\n",
              sm_pre.rss_bytes / 1048576.0, sm_pre.anon_huge_pages_bytes / 1048576.0);

  // Untouched HostRegion + H2D/D2H DMA
  ingine::HostRegion fresh(kFourMiB);
  fresh.register_with_cuda();
  INGINE_CHECK(fresh.registered());
  fill_bytes(fresh.data(), kFourMiB, 0x01020304ull);

  // PinnedHostBuffer + MappedHostBuffer
  ingine::fail_next_pinned_allocations(1);
  bool pinned_threw = false;
  try {
    ingine::PinnedHostBuffer should_fail(kFourMiB);
  } catch (const std::runtime_error&) {
    pinned_threw = true;
  }
  ingine::fail_next_pinned_allocations(0);
  INGINE_CHECK(pinned_threw);

  ingine::PinnedHostBuffer pinned(kFourMiB);
  INGINE_CHECK(pinned.data() != nullptr && is_2m_aligned(pinned.data()));
  INGINE_CHECK(pinned.hugepage_advised() == hp_env);
  fill_bytes(pinned.data(), kFourMiB, 0x77778888ull);
  INGINE_TEST_CUDA(cudaPointerGetAttributes(&attr, pinned.data()));
  INGINE_CHECK(attr.type == cudaMemoryTypeHost);

  ingine::MappedHostBuffer mapped(kFourMiB);
  INGINE_CHECK(mapped.data() != nullptr && mapped.device_ptr() != nullptr);
  INGINE_CHECK(is_2m_aligned(mapped.data()));
  INGINE_CHECK(mapped.hugepage_advised() == hp_env);
  INGINE_TEST_CUDA(cudaPointerGetAttributes(&attr, mapped.data()));
  INGINE_CHECK(attr.type == cudaMemoryTypeHost);
  std::memset(mapped.data(), 0, kFourMiB);

  ingine::DeviceBuffer dev(kFourMiB);
  cudaStream_t stream = nullptr;
  INGINE_TEST_CUDA(cudaStreamCreate(&stream));

  // Round-trip: pinned -> device -> mapped, and fresh -> device -> pinned
  INGINE_TEST_CUDA(cudaMemcpyAsync(dev.p, pinned.data(), kFourMiB, cudaMemcpyHostToDevice, stream));
  INGINE_TEST_CUDA(cudaMemcpyAsync(mapped.data(), dev.p, kFourMiB, cudaMemcpyDeviceToHost, stream));
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  check_bytes(mapped.data(), kFourMiB, 0x77778888ull);

  INGINE_TEST_CUDA(cudaMemcpyAsync(dev.p, fresh.data(), kFourMiB, cudaMemcpyHostToDevice, stream));
  INGINE_TEST_CUDA(cudaMemcpyAsync(pinned.data(), dev.p, kFourMiB, cudaMemcpyDeviceToHost, stream));
  INGINE_TEST_CUDA(cudaStreamSynchronize(stream));
  check_bytes(pinned.data(), kFourMiB, 0x01020304ull);

  INGINE_TEST_CUDA(cudaStreamDestroy(stream));
}

void test_cpu_gpu_exl3_parity() {
  std::printf("[2/2] Testing CPU vs CUDA EXL3_2 and EXL3_3 dequantization + GEMV parity...\n");
  using namespace ingine::ops::cpu;
  using namespace ingine::test;
  namespace moe = ingine::kernels::moe;

  constexpr int kRows = 256;
  constexpr int kCols = 256;
  Rng rng(0xE313BEEFull);
  CpuThreadPool& pool = CpuThreadPool::global();

  for (int bits : {2, 3}) {
    const moe::Format fmt = (bits == 2) ? moe::Format::EXL3_2 : moe::Format::EXL3_3;
    const std::size_t codes_bytes =
        static_cast<std::size_t>(kRows / 16) * static_cast<std::size_t>(kCols / 16) * static_cast<std::size_t>(32 * bits);
    std::vector<std::uint8_t> h_codes(codes_bytes);
    for (std::size_t i = 0; i < codes_bytes; ++i) h_codes[i] = static_cast<std::uint8_t>(rng.next() & 0xffu);

    std::vector<std::uint16_t> h_rin(kCols), h_rout(kRows);
    for (int c = 0; c < kCols; ++c) {
      h_rin[static_cast<std::size_t>(c)] = ingine::act::host_float_to_fp16_bits(static_cast<float>(0.15 + 0.25 * rng.uniform()));
    }
    for (int r = 0; r < kRows; ++r) {
      h_rout[static_cast<std::size_t>(r)] = ingine::act::host_float_to_fp16_bits(static_cast<float>(0.15 + 0.25 * rng.uniform()));
    }

    // CPU dequantization + GEMV
    std::vector<float> cpu_mat(static_cast<std::size_t>(kRows) * kCols);
    dequant_exl3_matrix_cpu(h_codes.data(), h_rin.data(), h_rout.data(), bits, kRows, kCols,
                            cpu_mat.data(), &pool, detect_cpu_isa());

    auto x = randn(rng, kCols, 0.5);
    std::vector<float> cpu_y(kRows);
    gemv_exl3_f32(h_codes.data(), h_rin.data(), h_rout.data(), bits, x.data(), cpu_y.data(),
                  kRows, kCols, &pool, detect_cpu_isa());

    // GPU dequant_experts on device
    DeviceArray<std::uint8_t> d_codes(h_codes);
    DeviceArray<std::uint16_t> d_rin(h_rin);
    DeviceArray<std::uint16_t> d_rout(h_rout);
    DeviceArray<int> d_experts(std::vector<int>{0});
    DeviceArray<ingine::ActT> d_out(static_cast<std::size_t>(kRows) * kCols);

    moe::dequant_experts(d_codes.get(), d_rin.get(), /*factor=*/nullptr, d_experts.get(),
                         /*n=*/1, kRows, kCols, fmt, d_out.get(), /*stream=*/nullptr, d_rout.get());
    INGINE_TEST_CUDA(cudaStreamSynchronize(nullptr));
    auto gpu_act = d_out.download();

    std::vector<float> cpu_mat_act(cpu_mat.size());
    for (std::size_t i = 0; i < cpu_mat.size(); ++i) {
      cpu_mat_act[i] = round_bf16(cpu_mat[i]);
    }
    std::vector<double> gpu_mat_d(static_cast<std::size_t>(kRows) * kCols);
    for (std::size_t i = 0; i < gpu_mat_d.size(); ++i) {
      gpu_mat_d[i] = static_cast<double>(bf16_to_float(gpu_act[i]));
    }
    const std::string name_m = "exl3_" + std::to_string(bits) + "_mat_cpu_vs_cuda";
    INGINE_CHECK(report(name_m.c_str(), compare(cpu_mat_act.data(), gpu_mat_d.data(), gpu_mat_d.size()),
                        5e-3));

    std::vector<double> gpu_y(kRows, 0.0);
    for (int r = 0; r < kRows; ++r) {
      double s = 0.0;
      for (int c = 0; c < kCols; ++c) s += gpu_mat_d[static_cast<std::size_t>(r) * kCols + c] * x[c];
      gpu_y[static_cast<std::size_t>(r)] = s;
    }
    const std::string name_y = "exl3_" + std::to_string(bits) + "_gemv_cpu_vs_cuda";
    INGINE_CHECK(report(name_y.c_str(), compare(cpu_y.data(), gpu_y.data(), kRows),
                        kScalarActIsFP16 ? 5e-3 : 1.5e-2));
  }
}

}  // namespace

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  int dev = 0;
  cudaDeviceProp prop{};
  INGINE_TEST_CUDA(cudaGetDevice(&dev));
  INGINE_TEST_CUDA(cudaGetDeviceProperties(&prop, dev));
  std::printf("core_baremetal_cuda running on device %d: %s (sm_%d%d)\n",
              dev, prop.name, prop.major, prop.minor);

  test_cuda_hugepage_host_buffers();
  test_cpu_gpu_exl3_parity();
  std::printf("core_baremetal_cuda: ALL PASSED\n");
  return 0;
}

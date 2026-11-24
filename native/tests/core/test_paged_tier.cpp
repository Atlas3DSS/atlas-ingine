// Unit tests for generic 3-tier VRAM / Pinned HostRegion / NVMe mmap paging primitives (core/paged_tier.h).
#include "core/device.h"
#include "core/paged_tier.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool ok, const char* msg) {
  if (!ok) throw std::runtime_error(std::string("test_paged_tier: ") + msg);
}

void test_mmap_and_bounce_and_lru() {
  namespace fs = std::filesystem;
  const fs::path tmp = fs::temp_directory_path() / "ingine_test_paged_tier.bin";
  constexpr std::size_t kSlot = 4096;
  constexpr int kObjects = 8;
  std::vector<std::uint8_t> golden(kObjects * kSlot);
  for (int i = 0; i < kObjects; ++i) {
    for (std::size_t b = 0; b < kSlot; ++b) {
      golden[i * kSlot + b] = static_cast<std::uint8_t>((i * 37 + b * 13) & 0xff);
    }
  }
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(golden.data()), std::streamsize(golden.size()));
    require(bool(f), "failed to write temp file");
  }

  ingine::MmapFile mf(tmp);
  require(mf.valid() && mf.size() == golden.size(), "MmapFile size mismatch");

  ingine::VramLruSlab slab(3, kSlot);
  require(slab.capacity() == 3 && slab.slot_bytes() == kSlot, "VramLruSlab geometry mismatch");
  ingine::PinnedBounceRing ring(2, kSlot);
  ingine::TieredDmaStream dma;

  cudaStream_t compute = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));

  // Pin objects 0 and 1 at generation 1, fetch object 2, then verify object 3 evicts object 2 (or 0 if unpinned).
  const std::uint64_t g1 = 1;
  auto r0 = slab.acquire(0, g1);
  auto r1 = slab.acquire(1, g1);
  auto r2 = slab.acquire(2, g1);
  require(!r0.hit && !r1.hit && !r2.hit && !r0.evicted && !r1.evicted && !r2.evicted, "initial fills failed");
  require(slab.touch_and_pin(0, g1) == r0.slot, "touch_and_pin hit failed");

  // Stage object 0 from MmapFile via PinnedBounceRing -> TieredDmaStream -> VRAM slot, verify exact bytes back.
  for (int obj : {0, 1, 2}) {
    const auto bslot = ring.acquire();
    mf.read_slice(std::uint64_t(obj) * kSlot, kSlot, bslot.ptr);
    const int sidx = slab.find(std::uint64_t(obj));
    require(sidx >= 0, "expected resident slot");
    dma.copy_h2d_async(slab.slot_ptr(sidx), bslot.ptr, kSlot);
    ring.mark_submitted(bslot.index, dma.stream());
  }
  dma.record_and_wait_on(compute);
  CUDA_CHECK(cudaStreamSynchronize(compute));

  // Generation 2: pin objects 0 and 2, then acquire object 3 -> MUST evict object 1 (the only unpinned LRU).
  const std::uint64_t g2 = 2;
  require(slab.touch_and_pin(0, g2) >= 0, "touch 0 failed");
  require(slab.touch_and_pin(2, g2) >= 0, "touch 2 failed");
  const auto r3 = slab.acquire(3, g2);
  require(!r3.hit && r3.evicted && r3.evicted_key == 1, "expected eviction of unpinned LRU key 1");

  const auto b3 = ring.acquire();
  mf.read_slice(std::uint64_t(3) * kSlot, kSlot, b3.ptr);
  dma.copy_h2d_async(r3.vram_ptr, b3.ptr, kSlot);
  ring.mark_submitted(b3.index, dma.stream());
  dma.record_and_wait_on(compute);

  std::vector<std::uint8_t> readback(kSlot);
  CUDA_CHECK(cudaMemcpyAsync(readback.data(), r3.vram_ptr, kSlot, cudaMemcpyDeviceToHost, compute));
  CUDA_CHECK(cudaStreamSynchronize(compute));
  require(std::memcmp(readback.data(), golden.data() + 3 * kSlot, kSlot) == 0, "VRAM readback mismatch for obj 3");

  // Also test HostRegion + MADV_DONTNEED + process_rss_bytes.
  mf.advise_dontneed_all();
  require(ingine::process_rss_bytes() > 0, "process_rss_bytes returned 0");

  // Priority LRU eviction shield: a priority (3x loop-layer) key survives single-pass sweeps.
  {
    ingine::VramLruSlab pslab(3, kSlot);
    (void)pslab.acquire(100, 1, /*priority=*/true);   // tick 1 + 3 = 4
    (void)pslab.acquire(200, 2, /*priority=*/false);  // tick 2
    (void)pslab.acquire(300, 3, /*priority=*/false);  // tick 3
    // Acquiring key 400 at gen 4 must evict non-priority key 200 (last_use=2), NOT priority key 100 (last_use=4).
    const auto ev = pslab.acquire(400, 4, /*priority=*/false);
    require(ev.evicted && ev.evicted_key == 200, "priority key should survive non-priority sweep");
    require(pslab.find(100) >= 0, "priority key 100 should remain resident");
  }

  CUDA_CHECK(cudaStreamDestroy(compute));
  fs::remove(tmp);
  std::printf("test_paged_tier: PASS (MmapFile + PinnedBounceRing + VramLruSlab[+priority] + TieredDmaStream)\n");
}

void test_swap_file() {
  namespace fs = std::filesystem;
  constexpr std::size_t kPage = 16384;  // 16 KiB test page
  constexpr int kPages = 4;
  const fs::path swap_path = fs::temp_directory_path() / "ingine_test_paged_tier_swap.bin";
  std::error_code ec;
  fs::remove(swap_path, ec);

  // Test non-immediate unlink so we can verify file creation AND automatic unlink on destruction.
  {
    ingine::SwapFile sf(swap_path, kPage * 2, /*unlink_immediately=*/false);
    require(sf.valid() && sf.size() == kPage * 2, "SwapFile initial size mismatch");
    require(fs::exists(swap_path), "SwapFile should exist when unlink_immediately=false");

    // Grow via resize() and write/read round-trip through GPU D2H -> BounceRing -> SwapFile -> BounceRing -> H2D.
    sf.resize(kPage * kPages);
    require(sf.size() == kPage * kPages, "SwapFile resize mismatch");

    std::vector<std::uint8_t> golden(kPage * kPages);
    for (std::size_t i = 0; i < golden.size(); ++i) golden[i] = static_cast<std::uint8_t>((i * 29 + 7) & 0xff);

    ingine::DeviceBuffer gpu_src(kPage), gpu_dst(kPage);
    ingine::PinnedBounceRing ring(2, kPage);
    ingine::TieredDmaStream dma;

    for (int p = 0; p < kPages; ++p) {
      gpu_src.copy_from_host(golden.data() + std::size_t(p) * kPage, kPage);
      const auto slot = ring.acquire();
      dma.copy_d2h_async(slot.ptr, gpu_src.p, kPage);
      ring.mark_submitted(slot.index, dma.stream());
      dma.synchronize();
      sf.write_slice(std::uint64_t(p) * kPage, kPage, slot.ptr);
    }

    std::vector<std::uint8_t> verify(kPage);
    for (int p = kPages - 1; p >= 0; --p) {
      const auto slot = ring.acquire();
      sf.read_slice(std::uint64_t(p) * kPage, kPage, slot.ptr);
      dma.copy_h2d_async(gpu_dst.p, slot.ptr, kPage);
      ring.mark_submitted(slot.index, dma.stream());
      dma.synchronize();
      CUDA_CHECK(cudaMemcpy(verify.data(), gpu_dst.p, kPage, cudaMemcpyDeviceToHost));
      require(std::memcmp(verify.data(), golden.data() + std::size_t(p) * kPage, kPage) == 0,
              "SwapFile GPU round-trip mismatch");
    }
    sf.advise_dontneed_all();
  }
  require(!fs::exists(swap_path), "SwapFile should be unlinked on close");

  // Also test auto-temp unlinked-immediately mode (crash-safe anonymous swap file).
  {
    ingine::SwapFile anon(fs::path{}, kPage, /*unlink_immediately=*/true);
    require(anon.valid() && anon.size() == kPage, "anonymous SwapFile init failed");
    require(!fs::exists(anon.path()), "anonymous SwapFile should be unlinked immediately");
    std::vector<std::uint8_t> buf(kPage, 0xA5), out(kPage, 0x00);
    anon.write_slice(0, kPage, buf.data());
    anon.read_slice(0, kPage, out.data());
    require(out == buf, "anonymous SwapFile round-trip mismatch");
  }
  std::printf("test_swap_file: PASS (SwapFile pread/pwrite + resize + D2H/H2D round-trip + unlink)\n");
}

}  // namespace

int main() {
  test_mmap_and_bounce_and_lru();
  test_swap_file();
  return 0;
}

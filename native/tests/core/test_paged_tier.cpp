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

  CUDA_CHECK(cudaStreamDestroy(compute));
  fs::remove(tmp);
  std::printf("test_paged_tier: PASS (MmapFile + PinnedBounceRing + VramLruSlab + TieredDmaStream)\n");
}

}  // namespace

int main() {
  test_mmap_and_bounce_and_lru();
  return 0;
}

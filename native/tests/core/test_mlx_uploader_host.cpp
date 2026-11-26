// Host compilation and behavior test for the real Apple MLX/Metal uploader translation unit
// (native/src/core/safetensors_upload_mlx.cpp). Verifies UploadOptions::chunk_bytes is strictly
// honored per pread chunk (64 KiB and 2 MiB on a 3 MiB fixture), hal::stream_synchronize(stream)
// runs prior to reads, and invalid requests / partial worker startup failures recover cleanly.
#include "core/fixture.h"
#include "core/hal.h"
#include "core/safetensors.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <sys/types.h>
#include <unistd.h>

namespace {

std::atomic<bool> g_probe_active{false};
std::atomic<int> g_sync_count{0};
void* g_last_synced_stream = nullptr;
std::atomic<std::size_t> g_pread_calls{0};
std::atomic<std::size_t> g_max_pread_bytes{0};
std::atomic<std::uint64_t> g_total_pread_bytes{0};
std::atomic<bool> g_saw_unsynced_pread{false};

void reset_probes() noexcept {
  g_sync_count.store(0, std::memory_order_release);
  g_last_synced_stream = nullptr;
  g_pread_calls.store(0, std::memory_order_release);
  g_max_pread_bytes.store(0, std::memory_order_release);
  g_total_pread_bytes.store(0, std::memory_order_release);
  g_saw_unsynced_pread.store(false, std::memory_order_release);
}

int g_failures = 0;
#define MLX_UP_CHECK(cond)                                                                   \
  do {                                                                                       \
    if (!(cond)) {                                                                           \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      ++g_failures;                                                                          \
    }                                                                                        \
  } while (0)

}  // namespace

namespace ingine::hal {

void stream_synchronize(StreamHandle stream) {
  g_last_synced_stream = stream.native;
  g_sync_count.fetch_add(1, std::memory_order_acq_rel);
}

}  // namespace ingine::hal

#if defined(__linux__)
extern "C" {
ssize_t __real_pread(int fd, void* buf, std::size_t count, off_t offset);
ssize_t __wrap_pread(int fd, void* buf, std::size_t count, off_t offset) {
  if (g_probe_active.load(std::memory_order_acquire)) {
    if (g_sync_count.load(std::memory_order_acquire) < 1) {
      g_saw_unsynced_pread.store(true, std::memory_order_release);
    }
    g_pread_calls.fetch_add(1, std::memory_order_relaxed);
    g_total_pread_bytes.fetch_add(count, std::memory_order_relaxed);
    std::size_t prev = g_max_pread_bytes.load(std::memory_order_relaxed);
    while (count > prev && !g_max_pread_bytes.compare_exchange_weak(prev, count, std::memory_order_relaxed)) {
    }
  }
  return __real_pread(fd, buf, count, offset);
}
}
#endif

int main() {
  using ingine::DeviceUploader;
  using ingine::SafetensorsFile;
  using ingine::UploadOptions;
  using ingine::UploadRequest;
  using ingine::UploadStats;
  using ingine::test::FixtureTensor;
  using ingine::test::pattern_bytes;
  namespace hal = ingine::hal;

  ingine::test::TempDir tmp;
  constexpr std::size_t kThreeMiB = std::size_t(3) << 20;
  std::vector<FixtureTensor> tensors = {
      {"w.u8", "U8", {static_cast<std::int64_t>(kThreeMiB)}, pattern_bytes(kThreeMiB, 99)},
  };
  const auto path = tmp.path / "mlx_3mib.safetensors";
  ingine::test::write_safetensors(path, tensors, {});
  const SafetensorsFile file(path);
  std::vector<std::uint8_t> dst(kThreeMiB, 0);

  // 1. Upload with chunk_bytes = 64 KiB on an explicit non-null StreamHandle sentinel.
  {
    UploadOptions opts;
    opts.chunk_bytes = 64u << 10;
    opts.staging_buffers = 4;
    opts.reader_threads = 3;
    DeviceUploader uploader(opts);

    std::fill(dst.begin(), dst.end(), 0u);
    const UploadRequest req = ingine::upload_request(file, file.at("w.u8"), dst.data());
    int stream_sentinel = 1;
    const hal::StreamHandle fake_stream = hal::StreamHandle::from_native(&stream_sentinel);

    reset_probes();
    g_probe_active.store(true, std::memory_order_release);
    const UploadStats st = uploader.upload({&req, 1}, fake_stream);
    g_probe_active.store(false, std::memory_order_release);

    MLX_UP_CHECK(st.bytes == kThreeMiB);
    MLX_UP_CHECK(g_sync_count.load() == 1);
    MLX_UP_CHECK(g_last_synced_stream == &stream_sentinel);
    MLX_UP_CHECK(std::memcmp(dst.data(), tensors[0].data.data(), kThreeMiB) == 0);
#if defined(__linux__)
    MLX_UP_CHECK(!g_saw_unsynced_pread.load());
    MLX_UP_CHECK(g_max_pread_bytes.load() <= opts.chunk_bytes);
    MLX_UP_CHECK(g_max_pread_bytes.load() == opts.chunk_bytes);
    MLX_UP_CHECK(g_pread_calls.load() >= kThreeMiB / opts.chunk_bytes);
    MLX_UP_CHECK(g_total_pread_bytes.load() == kThreeMiB);
#endif
  }

  // 2. Upload with chunk_bytes = 2 MiB on default stream.
  {
    UploadOptions opts;
    opts.chunk_bytes = std::size_t(2) << 20;
    opts.staging_buffers = 2;
    opts.reader_threads = 2;
    DeviceUploader uploader(opts);

    std::fill(dst.begin(), dst.end(), 0u);
    const UploadRequest req = ingine::upload_request(file, file.at("w.u8"), dst.data());
    reset_probes();
    g_probe_active.store(true, std::memory_order_release);
    const UploadStats st = uploader.upload({&req, 1}, hal::StreamHandle{});
    g_probe_active.store(false, std::memory_order_release);

    MLX_UP_CHECK(st.bytes == kThreeMiB);
    MLX_UP_CHECK(g_sync_count.load() == 1);
    MLX_UP_CHECK(g_last_synced_stream == nullptr);
    MLX_UP_CHECK(std::memcmp(dst.data(), tensors[0].data.data(), kThreeMiB) == 0);
#if defined(__linux__)
    MLX_UP_CHECK(!g_saw_unsynced_pread.load());
    MLX_UP_CHECK(g_max_pread_bytes.load() <= opts.chunk_bytes);
    MLX_UP_CHECK(g_max_pread_bytes.load() == opts.chunk_bytes);
    MLX_UP_CHECK(g_pread_calls.load() >= 2u);
    MLX_UP_CHECK(g_total_pread_bytes.load() == kThreeMiB);
#endif
  }

  // 3. Invalid requests and partial worker startup failures.
  {
    UploadOptions bad_opts;
    bad_opts.chunk_bytes = 0;
    bool threw_zero = false;
    try {
      DeviceUploader bad(bad_opts);
    } catch (const std::invalid_argument&) {
      threw_zero = true;
    }
    MLX_UP_CHECK(threw_zero);

    UploadOptions opts{4096, 2, 4};
    ingine::fail_next_uploader_worker_start(0);
    bool threw_w0 = false;
    try {
      DeviceUploader failed(opts);
    } catch (const std::system_error&) {
      threw_w0 = true;
    }
    MLX_UP_CHECK(threw_w0);

    ingine::fail_next_uploader_worker_start(2);
    bool threw_w2 = false;
    try {
      DeviceUploader partial(opts);
    } catch (const std::system_error&) {
      threw_w2 = true;
    }
    MLX_UP_CHECK(threw_w2);

    DeviceUploader uploader(opts);
    const UploadRequest past_end{&file, file.file_size() - 2, 4, dst.data()};
    bool threw_oob = false;
    try {
      uploader.upload({&past_end, 1}, {});
    } catch (const std::out_of_range&) {
      threw_oob = true;
    }
    MLX_UP_CHECK(threw_oob);
  }

  if (g_failures != 0) {
    std::fprintf(stderr, "FAIL core_mlx_uploader_host: %d checks failed\n", g_failures);
    return 1;
  }
  std::printf("PASS core_mlx_uploader_host (real safetensors_upload_mlx.cpp honors 64KiB/2MiB chunk_bytes & stream sync)\n");
  return 0;
}

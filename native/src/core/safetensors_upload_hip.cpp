// Pure native AMD ROCm/HIP safetensors weight uploader over pinned host staging and hipMemcpyAsync.
#include "core/safetensors.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace ingine {
namespace {

std::atomic<int> g_fail_worker_index{-1};

std::string hip_message(const char* what, hipError_t err) {
  return std::string(what) + ": " + hipGetErrorName(err) + ": " + hipGetErrorString(err);
}

void hip_throw(hipError_t err, const char* what) {
  if (err != hipSuccess) throw std::runtime_error(hip_message(what, err));
}

}  // namespace

struct DeviceUploader::Impl {
  struct Chunk {
    const SafetensorsFile* file;
    std::uint64_t offset;
    std::uint64_t bytes;
    void* dst;
  };
  struct Slot {
    std::byte* host = nullptr;
    hipEvent_t done = nullptr;
    bool copy_inflight = false;
  };

  explicit Impl(UploadOptions options) : options_(options) {
    if (options_.chunk_bytes == 0 || options_.staging_buffers <= 0 || options_.reader_threads <= 0) {
      throw std::invalid_argument("DeviceUploader: chunk_bytes, staging_buffers and reader_threads must be positive");
    }
    hip_throw(hipGetDevice(&device_), "hipGetDevice");
    const std::size_t n = static_cast<std::size_t>(options_.staging_buffers);
    if (options_.chunk_bytes > std::numeric_limits<std::size_t>::max() / n) {
      throw std::invalid_argument("DeviceUploader: staging size overflows");
    }
    hip_throw(hipHostMalloc(&pinned_, options_.chunk_bytes * n, hipHostMallocPortable),
              "DeviceUploader(HIP): hipHostMalloc");
    slots_.resize(n);
    try {
      for (std::size_t i = 0; i < n; ++i) {
        slots_[i].host = static_cast<std::byte*>(pinned_) + i * options_.chunk_bytes;
        hip_throw(hipEventCreateWithFlags(&slots_[i].done, hipEventDisableTiming),
                  "DeviceUploader(HIP): hipEventCreateWithFlags");
        free_slots_.push_back(static_cast<int>(i));
      }
      const int fail_at = g_fail_worker_index.exchange(-1, std::memory_order_acq_rel);
      workers_.reserve(static_cast<std::size_t>(options_.reader_threads));
      for (int i = 0; i < options_.reader_threads; ++i) {
        if (i == fail_at) {
          throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again),
                                  "injected HIP uploader worker startup failure");
        }
        workers_.emplace_back([this] { worker_main(); });
      }
    } catch (...) {
      {
        std::lock_guard lock(mutex_);
        stopping_ = true;
      }
      work_cv_.notify_all();
      for (std::thread& w : workers_) {
        if (w.joinable()) w.join();
      }
      for (Slot& s : slots_) {
        if (s.done != nullptr) (void)hipEventDestroy(s.done);
      }
      (void)hipHostFree(pinned_);
      throw;
    }
  }

  ~Impl() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    work_cv_.notify_all();
    for (std::thread& w : workers_) w.join();
    for (Slot& s : slots_) {
      if (s.copy_inflight) (void)hipEventSynchronize(s.done);
      if (s.done != nullptr) (void)hipEventDestroy(s.done);
    }
    (void)hipHostFree(pinned_);
  }

  UploadStats upload(std::span<const UploadRequest> requests, hipStream_t stream) {
    const auto start = std::chrono::steady_clock::now();
    int current = 0;
    hip_throw(hipGetDevice(&current), "hipGetDevice");
    if (current != device_) {
      throw std::logic_error("DeviceUploader::upload(HIP): current device " + std::to_string(current) +
                             " differs from construction device " + std::to_string(device_));
    }

    std::vector<Chunk> chunks;
    UploadStats stats;
    for (const UploadRequest& r : requests) {
      if (r.bytes == 0) continue;
      if (r.file == nullptr || r.dst == nullptr) {
        throw std::invalid_argument("DeviceUploader::upload: request without file or destination");
      }
      if (r.offset > r.file->file_size() || r.bytes > r.file->file_size() - r.offset) {
        throw std::out_of_range("DeviceUploader::upload: range exceeds " + r.file->path().string());
      }
      for (std::uint64_t done = 0; done < r.bytes;) {
        const std::uint64_t n = std::min<std::uint64_t>(r.bytes - done, options_.chunk_bytes);
        chunks.push_back(Chunk{r.file, r.offset + done, n, static_cast<std::byte*>(r.dst) + done});
        done += n;
      }
      stats.bytes += r.bytes;
    }
    if (chunks.empty()) return stats;

    {
      std::unique_lock lock(mutex_);
      chunks_ = std::move(chunks);
      next_chunk_ = 0;
      stream_ = stream;
      error_ = nullptr;
      active_workers_ = static_cast<int>(workers_.size());
      ++generation_;
      work_cv_.notify_all();
      done_cv_.wait(lock, [this] { return active_workers_ == 0; });
      chunks_.clear();
    }

    hipError_t sync_err = hipSuccess;
    for (Slot& s : slots_) {
      if (s.copy_inflight) {
        const hipError_t err = hipEventSynchronize(s.done);
        if (sync_err == hipSuccess) sync_err = err;
        s.copy_inflight = false;
      }
    }
    if (error_) std::rethrow_exception(std::exchange(error_, nullptr));
    hip_throw(sync_err, "DeviceUploader(HIP): waiting for copies");
    stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return stats;
  }

  void worker_main() {
    std::uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock lock(mutex_);
        work_cv_.wait(lock, [&] { return stopping_ || generation_ != seen; });
        if (stopping_) return;
        seen = generation_;
      }
      try {
        hip_throw(hipSetDevice(device_), "DeviceUploader(HIP): hipSetDevice");
        run_chunks();
      } catch (...) {
        std::lock_guard lock(mutex_);
        if (!error_) error_ = std::current_exception();
      }
      {
        std::lock_guard lock(mutex_);
        if (--active_workers_ == 0) done_cv_.notify_all();
      }
    }
  }

  void run_chunks() {
    for (;;) {
      Chunk chunk{};
      {
        std::lock_guard lock(mutex_);
        if (error_ || next_chunk_ >= chunks_.size()) return;
        chunk = chunks_[next_chunk_++];
      }
      const int slot_index = acquire_slot();
      Slot& slot = slots_[static_cast<std::size_t>(slot_index)];
      try {
        if (slot.copy_inflight) {
          hip_throw(hipEventSynchronize(slot.done), "DeviceUploader(HIP): staging wait");
          slot.copy_inflight = false;
        }
        chunk.file->read_range(chunk.offset, chunk.bytes, slot.host);
        {
          std::lock_guard lock(stream_mutex_);
          hip_throw(hipMemcpyAsync(chunk.dst, slot.host, chunk.bytes, hipMemcpyHostToDevice, stream_),
                    "DeviceUploader(HIP): hipMemcpyAsync");
          slot.copy_inflight = true;
          hip_throw(hipEventRecord(slot.done, stream_), "DeviceUploader(HIP): hipEventRecord");
        }
      } catch (...) {
        if (slot.copy_inflight) {
          (void)hipStreamSynchronize(stream_);
          slot.copy_inflight = false;
        }
        release_slot(slot_index);
        throw;
      }
      release_slot(slot_index);
    }
  }

  int acquire_slot() {
    std::unique_lock lock(mutex_);
    slot_cv_.wait(lock, [this] { return !free_slots_.empty(); });
    const int slot = free_slots_.front();
    free_slots_.pop_front();
    return slot;
  }

  void release_slot(int slot) {
    {
      std::lock_guard lock(mutex_);
      free_slots_.push_back(slot);
    }
    slot_cv_.notify_one();
  }

  UploadOptions options_;
  void* pinned_ = nullptr;
  std::vector<Slot> slots_;
  std::deque<int> free_slots_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  std::condition_variable slot_cv_;
  std::mutex stream_mutex_;
  std::uint64_t generation_ = 0;
  bool stopping_ = false;
  int active_workers_ = 0;
  std::vector<Chunk> chunks_;
  std::size_t next_chunk_ = 0;
  hipStream_t stream_ = nullptr;
  int device_ = 0;
  std::exception_ptr error_;
};

DeviceUploader::DeviceUploader(UploadOptions options)
    : options_(options), impl_(std::make_unique<Impl>(options)) {}

DeviceUploader::~DeviceUploader() = default;

UploadStats DeviceUploader::upload(std::span<const UploadRequest> requests, hal::StreamHandle stream) {
  return impl_->upload(requests, static_cast<hipStream_t>(stream.native));
}

void fail_next_uploader_worker_start(int worker_index) noexcept {
  g_fail_worker_index.store(worker_index, std::memory_order_release);
}

}  // namespace ingine

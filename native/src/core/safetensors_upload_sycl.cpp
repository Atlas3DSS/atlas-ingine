// Pure native C++20 Intel SYCL safetensors weight uploader over USM host staging and queue::memcpy.
#include "core/safetensors.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace ingine {
const sycl::device& sycl_device_for_id(int device_id);
sycl::queue* sycl_queue_for_stream(hal::StreamHandle stream);

namespace {

std::atomic<int> g_fail_worker_index{-1};

}  // namespace

struct DeviceUploader::Impl {
  struct Chunk {
    const SafetensorsFile* file;
    std::uint64_t offset;
    std::uint64_t bytes;
    void* dst;
    std::size_t req_idx = 0;
    std::size_t chunk_idx = 0;
  };
  struct Slot {
    std::byte* host = nullptr;
    sycl::event done{};
    bool copy_inflight = false;
  };

  explicit Impl(UploadOptions options) : options_(options) {
    if (options_.chunk_bytes == 0 || options_.staging_buffers <= 0 || options_.reader_threads <= 0) {
      throw std::invalid_argument("DeviceUploader(SYCL): chunk_bytes, staging_buffers and reader_threads must be > 0");
    }
    const std::size_t n = static_cast<std::size_t>(options_.staging_buffers);
    if (options_.chunk_bytes > std::numeric_limits<std::size_t>::max() / n) {
      throw std::invalid_argument("DeviceUploader(SYCL): staging size overflows");
    }
    queue_ = sycl_queue_for_stream(hal::StreamHandle{});
    pinned_ = sycl::aligned_alloc_host(4096, options_.chunk_bytes * n, *queue_);
    if (!pinned_) throw std::runtime_error("DeviceUploader(SYCL): aligned_alloc_host failed");
    slots_.resize(n);
    try {
      for (std::size_t i = 0; i < n; ++i) {
        slots_[i].host = static_cast<std::byte*>(pinned_) + i * options_.chunk_bytes;
        free_slots_.push_back(static_cast<int>(i));
      }
      const int fail_at = g_fail_worker_index.exchange(-1, std::memory_order_acq_rel);
      workers_.reserve(static_cast<std::size_t>(options_.reader_threads));
      for (int i = 0; i < options_.reader_threads; ++i) {
        if (i == fail_at) {
          throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again),
                                  "injected SYCL uploader worker startup failure");
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
      sycl::free(pinned_, *queue_);
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
      if (s.copy_inflight) s.done.wait();
    }
    if (pinned_ && queue_ != nullptr) sycl::free(pinned_, *queue_);
  }

  UploadStats upload(std::span<const UploadRequest> requests, hal::StreamHandle stream,
                     UploadChunkObserver observer = {}) {
    const auto start = std::chrono::steady_clock::now();
    sycl::queue* target_q = stream.native ? static_cast<sycl::queue*>(stream.native) : sycl_queue_for_stream(stream);
    std::vector<Chunk> chunks;
    UploadStats stats;
    for (std::size_t ri = 0; ri < requests.size(); ++ri) {
      const UploadRequest& r = requests[ri];
      if (r.bytes == 0) continue;
      if (r.file == nullptr || r.dst == nullptr) {
        throw std::invalid_argument("DeviceUploader::upload(SYCL): request without file or destination");
      }
      if (r.offset > r.file->file_size() || r.bytes > r.file->file_size() - r.offset) {
        throw std::out_of_range("DeviceUploader::upload(SYCL): range exceeds " + r.file->path().string());
      }
      std::size_t ci = 0;
      for (std::uint64_t done = 0; done < r.bytes; ++ci) {
        const std::uint64_t n = std::min<std::uint64_t>(r.bytes - done, options_.chunk_bytes);
        chunks.push_back(Chunk{r.file, r.offset + done, n, static_cast<std::byte*>(r.dst) + done, ri, ci});
        done += n;
      }
      stats.bytes += r.bytes;
    }
    if (chunks.empty()) return stats;

    {
      std::unique_lock lock(mutex_);
      chunks_ = std::move(chunks);
      next_chunk_ = 0;
      active_queue_ = target_q;
      observer_ = std::move(observer);
      error_ = nullptr;
      active_workers_ = static_cast<int>(workers_.size());
      ++generation_;
      work_cv_.notify_all();
      done_cv_.wait(lock, [this] { return active_workers_ == 0; });
      chunks_.clear();
      observer_ = {};
    }

    for (Slot& s : slots_) {
      if (s.copy_inflight) {
        s.done.wait_and_throw();
        s.copy_inflight = false;
      }
    }
    if (error_) std::rethrow_exception(std::exchange(error_, nullptr));
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
        for (;;) {
          Chunk chunk{};
          {
            std::lock_guard lock(mutex_);
            if (error_ || next_chunk_ >= chunks_.size()) break;
            chunk = chunks_[next_chunk_++];
          }
          const int slot_index = acquire_slot();
          Slot& slot = slots_[static_cast<std::size_t>(slot_index)];
          try {
            if (slot.copy_inflight) {
              slot.done.wait_and_throw();
              slot.copy_inflight = false;
            }
            chunk.file->read_range(chunk.offset, chunk.bytes, slot.host);
            if (observer_) {
              observer_(chunk.req_idx, chunk.chunk_idx, slot.host, static_cast<std::size_t>(chunk.bytes));
            }
            {
              std::lock_guard lock(stream_mutex_);
              slot.done = active_queue_->memcpy(chunk.dst, slot.host, chunk.bytes);
              slot.copy_inflight = true;
            }
          } catch (...) {
            if (slot.copy_inflight) {
              slot.done.wait();
              slot.copy_inflight = false;
            }
            release_slot(slot_index);
            throw;
          }
          release_slot(slot_index);
        }
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
  sycl::queue* queue_ = nullptr;
  sycl::queue* active_queue_ = nullptr;
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
  UploadChunkObserver observer_;
  std::exception_ptr error_;
};

DeviceUploader::DeviceUploader(UploadOptions options)
    : options_(options), impl_(std::make_unique<Impl>(options)) {}

DeviceUploader::~DeviceUploader() = default;

UploadStats DeviceUploader::upload(std::span<const UploadRequest> requests, hal::StreamHandle stream,
                                   UploadChunkObserver observer) {
  return impl_->upload(requests, stream, std::move(observer));
}

void fail_next_uploader_worker_start(int worker_index) noexcept {
  g_fail_worker_index.store(worker_index, std::memory_order_release);
}

}  // namespace ingine

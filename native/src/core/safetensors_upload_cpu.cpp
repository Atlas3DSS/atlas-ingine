#include "core/safetensors.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
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

} // namespace

struct DeviceUploader::Impl {
    struct Chunk {
        const SafetensorsFile* file;
        std::uint64_t offset;
        std::uint64_t bytes;
        void* dst;
    };

    explicit Impl(UploadOptions options) : options_(options) {
        if (options_.chunk_bytes == 0 || options_.staging_buffers <= 0 ||
            options_.reader_threads <= 0) {
            throw std::invalid_argument("DeviceUploader: chunk_bytes, staging_buffers and "
                                        "reader_threads must be positive");
        }
        const std::size_t n = static_cast<std::size_t>(options_.staging_buffers);
        if (options_.chunk_bytes > std::numeric_limits<std::size_t>::max() / n) {
            throw std::invalid_argument("DeviceUploader: staging size overflows");
        }
        const int fail_at = g_fail_worker_index.exchange(-1, std::memory_order_acq_rel);
        workers_.reserve(static_cast<std::size_t>(options_.reader_threads));
        try {
            for (int i = 0; i < options_.reader_threads; ++i) {
                if (i == fail_at) {
                    throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again),
                                            "injected CPU uploader worker startup failure");
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
                if (w.joinable()) { w.join(); }
            }
            throw;
        }
    }

    ~Impl() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        work_cv_.notify_all();
        for (std::thread& w : workers_) { w.join(); }
    }

    UploadStats upload(std::span<const UploadRequest> requests, hal::StreamHandle stream) {
        hal::stream_synchronize(stream);
        const auto start = std::chrono::steady_clock::now();
        std::vector<Chunk> chunks;
        UploadStats stats;
        for (const UploadRequest& r : requests) {
            if (r.bytes == 0) { continue; }
            if (r.file == nullptr || r.dst == nullptr) {
                throw std::invalid_argument(
                    "DeviceUploader::upload: request without file or destination");
            }
            if (r.offset > r.file->file_size() || r.bytes > r.file->file_size() - r.offset) {
                throw std::out_of_range("DeviceUploader::upload: range exceeds " +
                                        r.file->path().string());
            }
            for (std::uint64_t done = 0; done < r.bytes;) {
                const std::uint64_t n =
                    std::min<std::uint64_t>(r.bytes - done, options_.chunk_bytes);
                chunks.push_back(
                    Chunk{r.file, r.offset + done, n, static_cast<std::byte*>(r.dst) + done});
                done += n;
            }
            stats.bytes += r.bytes;
        }
        if (chunks.empty()) { return stats; }

        {
            std::unique_lock lock(mutex_);
            chunks_         = std::move(chunks);
            next_chunk_     = 0;
            error_          = nullptr;
            active_workers_ = static_cast<int>(workers_.size());
            ++generation_;
            work_cv_.notify_all();
            done_cv_.wait(lock, [this] { return active_workers_ == 0; });
            chunks_.clear();
        }
        if (error_) { std::rethrow_exception(std::exchange(error_, nullptr)); }
        stats.seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return stats;
    }

    void worker_main() {
        std::uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock lock(mutex_);
                work_cv_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                if (stopping_) { return; }
                seen = generation_;
            }
            try {
                for (;;) {
                    Chunk chunk{};
                    {
                        std::lock_guard lock(mutex_);
                        if (error_ || next_chunk_ >= chunks_.size()) { break; }
                        chunk = chunks_[next_chunk_++];
                    }
                    chunk.file->read_range(chunk.offset, chunk.bytes, chunk.dst);
                }
            } catch (...) {
                std::lock_guard lock(mutex_);
                if (!error_) { error_ = std::current_exception(); }
            }
            {
                std::lock_guard lock(mutex_);
                if (--active_workers_ == 0) { done_cv_.notify_all(); }
            }
        }
    }

    UploadOptions options_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_cv_;
    std::condition_variable done_cv_;
    std::uint64_t generation_ = 0;
    bool stopping_            = false;
    int active_workers_       = 0;
    std::vector<Chunk> chunks_;
    std::size_t next_chunk_ = 0;
    std::exception_ptr error_;
};

DeviceUploader::DeviceUploader(UploadOptions options)
    : options_(options), impl_(std::make_unique<Impl>(options)) {}

DeviceUploader::~DeviceUploader() = default;

UploadStats DeviceUploader::upload(std::span<const UploadRequest> requests,
                                   hal::StreamHandle stream) {
    return impl_->upload(requests, stream);
}

void fail_next_uploader_worker_start(int worker_index) noexcept {
    g_fail_worker_index.store(worker_index, std::memory_order_release);
}

} // namespace ingine

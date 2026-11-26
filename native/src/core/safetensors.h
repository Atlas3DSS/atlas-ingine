#pragma once

// Safetensors checkpoints: header parsing, tensor lookup, host reads (pread) and a fast host-to-device
// upload path (parallel reads into pinned staging buffers overlapped with asynchronous copies).
//
// File format: an 8-byte little-endian header length N, N bytes of JSON
//   {"__metadata__": {str: str}, "<name>": {"dtype": "BF16", "shape": [..], "data_offsets": [b, e]}, ...}
// then the data region; data_offsets are relative to the start of the data region.

#include "core/dtype.h"
#include "core/hal.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ingine {

class WeightPack;

// Element size in bytes of a safetensors dtype string ("BF16", "F8_E4M3", "U8", ...); throws for
// dtypes without a whole-byte element size or unknown names.
std::size_t safetensors_dtype_size(std::string_view dtype);
// The ingine DType for a safetensors dtype string, if ingine has one (F8_E4M3 -> FP8_E4M3FN).
std::optional<DType> safetensors_to_dtype(std::string_view dtype);

struct SafetensorsTensor {
    std::string name;
    std::string dtype;               // safetensors dtype string as stored ("BF16", "F8_E4M3", ...)
    std::vector<std::int64_t> shape; // as stored: outermost dimension first (PyTorch order)
    std::uint64_t offset = 0;        // absolute byte offset of the first element in the file
    std::uint64_t bytes  = 0;

    [[nodiscard]] std::int64_t numel() const noexcept;
    // ingine DType; throws when the stored dtype has no ingine equivalent.
    [[nodiscard]] DType ingine_dtype() const;
    // Non-owning ingine::Tensor over `data` (device or host) with this tensor's dtype and shape:
    // ne[0] is the innermost (last stored) dimension. Throws for rank > 4 or dims beyond int32.
    [[nodiscard]] Tensor view(void* data) const;
};

class SafetensorsFile {
public:
    // Opens the file and parses its header. Throws std::runtime_error on I/O or format errors.
    explicit SafetensorsFile(const std::filesystem::path& path);
    ~SafetensorsFile();

    SafetensorsFile(const SafetensorsFile&)            = delete;
    SafetensorsFile& operator=(const SafetensorsFile&) = delete;
    SafetensorsFile(SafetensorsFile&& other) noexcept;
    SafetensorsFile& operator=(SafetensorsFile&& other) noexcept;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] int fd() const noexcept;
    [[nodiscard]] bool is_weight_pack() const noexcept { return static_cast<bool>(pack_); }
    [[nodiscard]] const WeightPack* weight_pack() const noexcept { return pack_.get(); }
    [[nodiscard]] const std::byte* mapped_data() const noexcept;
    [[nodiscard]] const std::byte* tensor_mapped_ptr(const SafetensorsTensor& tensor) const noexcept;
    [[nodiscard]] std::uint64_t file_size() const noexcept { return file_size_; }
    [[nodiscard]] std::uint64_t data_offset() const noexcept { return data_offset_; }
    // "__metadata__" (string values only, as the format requires); empty when absent.
    [[nodiscard]] const std::map<std::string, std::string>& metadata() const noexcept {
        return metadata_;
    }
    // Every tensor, in data-offset order.
    [[nodiscard]] const std::vector<SafetensorsTensor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] std::vector<std::string> names() const;
    [[nodiscard]] const SafetensorsTensor* find(std::string_view name) const;
    [[nodiscard]] const SafetensorsTensor& at(std::string_view name) const; // throws when absent
    [[nodiscard]] bool contains(std::string_view name) const { return find(name) != nullptr; }

    // Reads `bytes` bytes at absolute file offset `offset` into host memory (pread loop).
    void read_range(std::uint64_t offset, std::uint64_t bytes, void* dst) const;
    void read(const SafetensorsTensor& tensor, void* dst) const;
    [[nodiscard]] std::vector<std::byte> read(std::string_view name) const;

private:
    struct NameHash {
        using is_transparent = void;
        std::size_t operator()(std::string_view s) const noexcept {
            return std::hash<std::string_view>{}(s);
        }
    };

    std::filesystem::path path_;
    int fd_                    = -1;
    std::unique_ptr<WeightPack> pack_;
    std::uint64_t file_size_   = 0;
    std::uint64_t data_offset_ = 0;
    std::map<std::string, std::string> metadata_;
    std::vector<SafetensorsTensor> tensors_;
    std::unordered_map<std::string, std::size_t, NameHash, std::equal_to<>> index_;
};

// A tensor together with the file holding it.
struct SafetensorsRef {
    const SafetensorsFile* file     = nullptr;
    const SafetensorsTensor* tensor = nullptr;

    explicit operator bool() const noexcept { return tensor != nullptr; }
    const SafetensorsTensor* operator->() const noexcept { return tensor; }
    const SafetensorsTensor& operator*() const noexcept { return *tensor; }
};

// A checkpoint made of one or more safetensors files.
class SafetensorsModel {
public:
    // `path` is a .safetensors file, or a directory holding model.safetensors.index.json (every file
    // named in its weight_map; the index decides which file holds which tensor) or model.safetensors.
    explicit SafetensorsModel(const std::filesystem::path& path);

    SafetensorsModel(SafetensorsModel&&) noexcept            = default;
    SafetensorsModel& operator=(SafetensorsModel&&) noexcept = default;

    [[nodiscard]] const std::vector<std::unique_ptr<SafetensorsFile>>& files() const noexcept {
        return files_;
    }
    [[nodiscard]] bool is_weight_pack() const noexcept {
        return !files_.empty() && files_[0]->is_weight_pack();
    }
    [[nodiscard]] std::vector<std::string> names() const; // sorted
    [[nodiscard]] std::size_t size() const noexcept { return index_.size(); }
    [[nodiscard]] SafetensorsRef find(std::string_view name) const;
    [[nodiscard]] SafetensorsRef at(std::string_view name) const; // throws when absent
    [[nodiscard]] bool contains(std::string_view name) const { return static_cast<bool>(find(name)); }
    [[nodiscard]] std::vector<std::byte> read(std::string_view name) const;

private:
    std::vector<std::unique_ptr<SafetensorsFile>> files_;
    std::map<std::string, SafetensorsRef, std::less<>> index_;
};

// One host-to-device copy: `bytes` bytes at absolute offset `offset` of `file` to device `dst`.
struct UploadRequest {
    const SafetensorsFile* file = nullptr;
    std::uint64_t offset        = 0;
    std::uint64_t bytes         = 0;
    void* dst                   = nullptr;
};

[[nodiscard]] UploadRequest upload_request(const SafetensorsFile& file,
                                           const SafetensorsTensor& tensor, void* dst);
[[nodiscard]] UploadRequest upload_request(SafetensorsRef ref, void* dst);

// Canonical chunk size shared by DeviceUploader and chunked safetensors content digests.
inline constexpr std::size_t kSafetensorsHashChunkBytes = std::size_t(32) << 20;

struct UploadOptions {
    std::size_t chunk_bytes = kSafetensorsHashChunkBytes; // staging buffer size and max bytes per copy
    int staging_buffers     = 12;                         // pinned buffers in flight
    int reader_threads      = 6;                          // threads reading file data into staging
};

struct UploadStats {
    std::uint64_t bytes = 0;
    double seconds      = 0;
    [[nodiscard]] double gib_per_second() const noexcept {
        return seconds > 0 ? double(bytes) / double(1ull << 30) / seconds : 0.0;
    }
};

// Optional per-chunk host-buffer observer invoked from a reader thread right after pread, with
// (request_index, chunk_index_within_request, host_data, chunk_bytes). Must be thread-safe.
using UploadChunkObserver =
    std::function<void(std::size_t req_idx, std::size_t chunk_idx, const void* data, std::size_t bytes)>;

// Uploads file ranges to device memory. Reader threads pread chunks into pinned staging buffers and
// enqueue each chunk's cudaMemcpyAsync as soon as it is read, so disk/page-cache reads, the copies of
// other chunks and further reads overlap. Owns the pinned staging memory (chunk_bytes *
// staging_buffers) and the reader threads for its lifetime: create one for a whole model load.
class DeviceUploader {
public:
    explicit DeviceUploader(UploadOptions options = {});
    ~DeviceUploader();

    DeviceUploader(const DeviceUploader&)            = delete;
    DeviceUploader& operator=(const DeviceUploader&) = delete;

    // Copies every request (on `stream`, after the work already queued there) and returns once all
    // bytes have arrived on the device. Destinations must not overlap. On CUDA builds, the calling
    // thread's current device must match the device at construction. One upload() at a time per uploader.
    UploadStats upload(std::span<const UploadRequest> requests, hal::StreamHandle stream,
                       UploadChunkObserver observer = {});

    [[nodiscard]] const UploadOptions& options() const noexcept { return options_; }

private:
    struct Impl;
    UploadOptions options_;
    std::unique_ptr<Impl> impl_;
};

// Test hook: causes the next DeviceUploader worker startup loop to fail with std::system_error when
// launching worker `worker_index` (0-based; negative disables).
void fail_next_uploader_worker_start(int worker_index) noexcept;

} // namespace ingine

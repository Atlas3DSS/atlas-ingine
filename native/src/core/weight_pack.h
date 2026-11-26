// 2 MiB-aligned flat mmap weight pack format (`INGPACK1`) for zero-parse cold boot.
// 100% CUDA-free (lives strictly in ingine_core).
#pragma once

#include "core/dtype.h"
#include "core/hugepage.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ingine {

inline constexpr char kWeightPackMagic[8]                = {'I', 'N', 'G', 'P', 'A', 'C', 'K', '1'};
inline constexpr std::uint32_t kWeightPackVersion        = 1u;
inline constexpr std::uint32_t kWeightPackFlagHasContentSha256 = 1u << 0;
inline constexpr std::uint32_t kWeightPackFlagLargeAligned2M   = 1u << 1;
inline constexpr std::size_t kWeightPackDefaultAlignBytes      = kHugePageAlignBytes;  // 2 MiB
inline constexpr std::size_t kWeightPackDefaultThresholdBytes  = kHugePageAlignBytes;  // 2 MiB
inline constexpr std::size_t kWeightPackSmallAlignBytes        = 256u;
inline constexpr std::size_t kWeightPackHashChunkBytes         = std::size_t(32) << 20;

#pragma pack(push, 1)
struct WeightPackHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t flags;
    std::uint64_t file_size;
    std::uint64_t data_offset;
    std::uint64_t data_size;
    std::uint64_t tensor_alignment;
    std::uint64_t large_tensor_threshold;
    std::uint32_t num_tensors;
    std::uint32_t num_metadata;
    std::uint64_t dir_offset;
    std::uint64_t meta_offset;
    std::uint64_t string_table_offset;
    std::uint64_t string_table_bytes;
    std::uint8_t header_sha256[32];
    std::uint8_t content_sha256[32];
    std::uint8_t reserved[96];
};

struct WeightPackEntry {
    std::uint64_t name_hash;
    std::uint32_t name_offset;
    std::uint32_t name_len;
    std::uint32_t dtype_offset;
    std::uint32_t dtype_len;
    std::uint32_t rank;
    std::uint32_t ingine_dtype_code;
    std::uint64_t data_offset;
    std::uint64_t data_bytes;
    std::int64_t shape[6];
};

struct WeightPackMetaEntry {
    std::uint32_t key_offset;
    std::uint32_t key_len;
    std::uint32_t val_offset;
    std::uint32_t val_len;
};
#pragma pack(pop)

static_assert(sizeof(WeightPackHeader) == 256, "WeightPackHeader must be 256 bytes");
static_assert(std::is_trivially_copyable_v<WeightPackHeader>);
static_assert(sizeof(WeightPackEntry) == 96, "WeightPackEntry must be 96 bytes");
static_assert(std::is_trivially_copyable_v<WeightPackEntry>);
static_assert(sizeof(WeightPackMetaEntry) == 16, "WeightPackMetaEntry must be 16 bytes");
static_assert(std::is_trivially_copyable_v<WeightPackMetaEntry>);

[[nodiscard]] bool weight_pack_enabled() noexcept;
[[nodiscard]] bool is_weight_pack_file(const std::filesystem::path& path) noexcept;
[[nodiscard]] std::uint64_t weight_pack_fnv1a(std::string_view text) noexcept;

struct WeightPackOpenOptions {
    bool advise_hugepage = true;
    bool advise_sequential = false;
    bool advise_willneed = false;
    bool verify_header_sha = true;
    bool verify_content_sha = false;
    int numa_node = -1;
    NumaPolicy numa_policy = NumaPolicy::Default;
};

class WeightPack {
public:
    explicit WeightPack(const std::filesystem::path& path, WeightPackOpenOptions opts = {});
    ~WeightPack();

    WeightPack(const WeightPack&)            = delete;
    WeightPack& operator=(const WeightPack&) = delete;
    WeightPack(WeightPack&& other) noexcept;
    WeightPack& operator=(WeightPack&& other) noexcept;

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }
    [[nodiscard]] int fd() const noexcept { return fd_; }
    [[nodiscard]] std::uint64_t file_size() const noexcept { return file_size_; }
    [[nodiscard]] std::uint64_t data_offset() const noexcept { return header_.data_offset; }
    [[nodiscard]] const WeightPackHeader& header() const noexcept { return header_; }
    [[nodiscard]] std::uint32_t num_tensors() const noexcept { return header_.num_tensors; }
    [[nodiscard]] std::span<const WeightPackEntry> entries() const noexcept { return entries_; }

    [[nodiscard]] const WeightPackEntry* find(std::string_view name) const noexcept;
    [[nodiscard]] const WeightPackEntry& at(std::string_view name) const;
    [[nodiscard]] std::string_view entry_name(const WeightPackEntry& entry) const noexcept;
    [[nodiscard]] std::string_view entry_dtype(const WeightPackEntry& entry) const noexcept;
    [[nodiscard]] const std::byte* tensor_data(const WeightPackEntry& entry) const noexcept;
    [[nodiscard]] const std::byte* tensor_data(std::string_view name) const;
    [[nodiscard]] Tensor tensor_view(std::string_view name) const;
    [[nodiscard]] std::optional<std::string_view> find_metadata(std::string_view key) const noexcept;
    [[nodiscard]] std::map<std::string, std::string> metadata() const;

    void read_range(std::uint64_t offset, std::uint64_t bytes, void* dst) const;
    void prefault(bool parallel = true) const;
    [[nodiscard]] bool verify_content_sha256() const;

    [[nodiscard]] bool is_2m_aligned_base() const noexcept { return aligned_2m_base_; }
    [[nodiscard]] bool hugepage_advised() const noexcept { return hugepage_advised_; }
    [[nodiscard]] bool numa_bound() const noexcept { return numa_bound_; }
    [[nodiscard]] const std::byte* mapped_base() const noexcept { return base_; }

private:
    void release() noexcept;
    [[nodiscard]] std::string_view slice_string(std::uint32_t off, std::uint32_t len) const noexcept;

    std::filesystem::path path_;
    int fd_                              = -1;
    std::byte* base_                     = nullptr;
    std::size_t mapped_bytes_            = 0;
    std::uint64_t file_size_             = 0;
    bool aligned_2m_base_                = false;
    bool hugepage_advised_               = false;
    bool numa_bound_                     = false;
    WeightPackHeader header_{};
    std::span<const WeightPackEntry> entries_;
    std::span<const WeightPackMetaEntry> meta_entries_;
    const char* string_table_            = nullptr;
};

[[nodiscard]] bool weight_pack_matches_safetensors_dir(const WeightPack& pack,
                                                       const std::filesystem::path& dir) noexcept;

struct WeightPackBuildOptions {
    std::size_t tensor_alignment       = kWeightPackDefaultAlignBytes;
    std::size_t large_tensor_threshold = kWeightPackDefaultThresholdBytes;
    bool compute_content_sha256        = true;
    std::size_t io_chunk_bytes         = std::size_t(32) << 20;
};

struct WeightPackBuildStats {
    std::uint64_t num_tensors          = 0;
    std::uint64_t small_tensors        = 0;
    std::uint64_t large_tensors        = 0;
    std::uint64_t file_size            = 0;
    std::uint64_t data_offset          = 0;
    std::uint64_t payload_bytes        = 0;
    std::array<std::uint8_t, 32> content_sha256{};
    double seconds                     = 0.0;
};

class WeightPackBuilder {
public:
    using DataWriter = std::function<void(const std::function<void(const void*, std::size_t)>& emit)>;

    void add_metadata(std::string key, std::string value);
    void add_tensor(std::string name, std::string dtype, std::vector<std::int64_t> shape,
                    std::uint64_t bytes, DataWriter writer);

    [[nodiscard]] WeightPackBuildStats write(const std::filesystem::path& output_path,
                                             WeightPackBuildOptions opts = {});

    [[nodiscard]] static WeightPackBuildStats pack_from_safetensors(
        const std::filesystem::path& src_path, const std::filesystem::path& output_path,
        WeightPackBuildOptions opts = {});

private:
    struct PendingTensor {
        std::string name;
        std::string dtype;
        std::vector<std::int64_t> shape;
        std::uint64_t bytes = 0;
        DataWriter writer;
        std::size_t source_order = 0;
    };

    [[nodiscard]] WeightPackBuildStats write_impl(const std::filesystem::path& output_path,
                                                  WeightPackBuildOptions opts,
                                                  const std::function<void()>& pre_publish_check);

    std::map<std::string, std::string> metadata_;
    std::vector<PendingTensor> tensors_;
};

// Test hooks for deterministic snapshot-consistency verification in pack_from_safetensors.
void set_next_pack_after_open_hook(std::function<void()> hook);
void set_next_pack_before_copy_hook(std::function<void()> hook);

}  // namespace ingine

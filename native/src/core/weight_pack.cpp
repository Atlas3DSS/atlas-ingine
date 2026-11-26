#include "core/weight_pack.h"

#include "core/safetensors.h"
#include "core/sha256.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>

namespace ingine {
namespace {

bool is_power_of_two_u64(std::uint64_t v) noexcept { return v != 0 && (v & (v - 1)) == 0; }

std::uint64_t align_up_u64(std::uint64_t v, std::uint64_t align) {
    if (!is_power_of_two_u64(align)) {
        throw std::invalid_argument("WeightPack: alignment must be a power of two");
    }
    if (v > std::numeric_limits<std::uint64_t>::max() - (align - 1)) {
        throw std::overflow_error("WeightPack: aligned offset overflows");
    }
    return (v + align - 1) & ~(align - 1);
}

[[noreturn]] void fail_pack(const std::filesystem::path& path, const std::string& what) {
    throw std::runtime_error("weight_pack " + path.string() + ": " + what);
}

[[noreturn]] void fail_pack_errno(const std::filesystem::path& path, const std::string& what) {
    fail_pack(path, what + ": " + std::strerror(errno));
}

void hash_tagged(Sha256& ctx, const char* tag, const void* data, std::uint64_t len) noexcept {
    const std::uint64_t tlen = static_cast<std::uint64_t>(std::strlen(tag));
    ctx.update(&tlen, sizeof(tlen));
    ctx.update(tag, static_cast<std::size_t>(tlen));
    ctx.update(&len, sizeof(len));
    if (len > 0) { ctx.update(data, static_cast<std::size_t>(len)); }
}

std::array<std::uint8_t, 32> tensor_digest_from_chunks(
    std::string_view name, std::string_view dtype, std::span<const std::int64_t> shape,
    std::uint64_t bytes, std::span<const std::array<std::uint8_t, 32>> chunk_shas) noexcept {
    Sha256 ctx;
    static constexpr char kDomain[] = "INGINE_TENSOR_V1";
    ctx.update(kDomain, sizeof(kDomain) - 1);
    hash_tagged(ctx, "name", name.data(), name.size());
    hash_tagged(ctx, "dtype", dtype.data(), dtype.size());
    const std::uint64_t rank = static_cast<std::uint64_t>(shape.size());
    hash_tagged(ctx, "rank", &rank, sizeof(rank));
    if (!shape.empty()) { hash_tagged(ctx, "shape", shape.data(), shape.size_bytes()); }
    hash_tagged(ctx, "bytes", &bytes, sizeof(bytes));
    const std::uint64_t n_chunks = static_cast<std::uint64_t>(chunk_shas.size());
    hash_tagged(ctx, "n_chunks", &n_chunks, sizeof(n_chunks));
    for (const auto& c : chunk_shas) { ctx.update(c.data(), c.size()); }
    return ctx.finish();
}

std::array<std::uint8_t, 32> finalize_pack_content_sha256(
    std::span<const std::pair<std::string_view, std::array<std::uint8_t, 32>>> sorted_tensors) noexcept {
    Sha256 ctx;
    static constexpr char kDomain[] = "INGPACK_CONTENT_V1";
    ctx.update(kDomain, sizeof(kDomain) - 1);
    const std::uint64_t count = static_cast<std::uint64_t>(sorted_tensors.size());
    hash_tagged(ctx, "num_tensors", &count, sizeof(count));
    for (const auto& [name, sha] : sorted_tensors) {
        hash_tagged(ctx, "tensor_name", name.data(), name.size());
        ctx.update(sha.data(), sha.size());
    }
    return ctx.finish();
}

std::array<std::uint8_t, 32> compute_header_sha256(const std::byte* base,
                                                   std::uint64_t header_end) noexcept {
    constexpr std::size_t kHashOff = offsetof(WeightPackHeader, header_sha256);
    constexpr std::size_t kHashLen = sizeof(WeightPackHeader::header_sha256);
    static constexpr std::uint8_t kZeros[kHashLen]{};
    Sha256 ctx;
    ctx.update(base, kHashOff);
    ctx.update(kZeros, kHashLen);
    if (header_end > kHashOff + kHashLen) {
        ctx.update(base + kHashOff + kHashLen, static_cast<std::size_t>(header_end - (kHashOff + kHashLen)));
    }
    return ctx.finish();
}

void write_all_fd(int fd, const void* buf, std::size_t len, const std::filesystem::path& path) {
    const auto* p = static_cast<const std::uint8_t*>(buf);
    while (len > 0) {
        const ssize_t n = ::write(fd, p, len);
        if (n < 0) {
            if (errno == EINTR) { continue; }
            fail_pack_errno(path, "write failed");
        }
        p += n;
        len -= static_cast<std::size_t>(n);
    }
}

void write_zeros_fd(int fd, std::uint64_t count, const std::filesystem::path& path) {
    static constexpr std::uint8_t kZeros[65536]{};
    while (count > 0) {
        const std::size_t chunk = static_cast<std::size_t>(std::min<std::uint64_t>(count, sizeof(kZeros)));
        write_all_fd(fd, kZeros, chunk, path);
        count -= chunk;
    }
}

void sync_parent_dir(const std::filesystem::path& file_path) noexcept {
    std::filesystem::path dir = file_path.has_parent_path() ? file_path.parent_path() : std::filesystem::path(".");
    if (dir.empty()) { dir = "."; }
    const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (dfd >= 0) {
        (void)::fsync(dfd);
        ::close(dfd);
    }
}

struct ScopedFd {
    int fd = -1;
    ~ScopedFd() {
        if (fd >= 0) { ::close(fd); }
    }
};

inline constexpr std::uint64_t kFullFileShaThresholdBytes = 32ull << 20;

struct ShardFingerprint {
    std::uint64_t file_size  = 0;
    std::uint64_t dev        = 0;
    std::uint64_t ino        = 0;
    std::uint64_t nlink      = 0;
    std::int64_t mtime_sec   = 0;
    std::int64_t mtime_nsec  = 0;
    std::int64_t ctime_sec   = 0;
    std::int64_t ctime_nsec  = 0;
    bool has_full_sha256     = false;
    std::array<std::uint8_t, 32> full_sha256{};
};

bool timespec_strictly_less(std::int64_t a_sec, std::int64_t a_nsec,
                            std::int64_t b_sec, std::int64_t b_nsec) noexcept {
    return a_sec < b_sec || (a_sec == b_sec && a_nsec < b_nsec);
}

std::mutex g_pack_hook_mu;
std::function<void()> g_next_after_open_hook;
std::function<void()> g_next_before_copy_hook;

std::function<void()> take_hook(std::function<void()>& slot) {
    std::lock_guard<std::mutex> lk(g_pack_hook_mu);
    return std::exchange(slot, nullptr);
}

bool fill_fingerprint_from_stat(const struct stat& st, ShardFingerprint& out) noexcept {
    if (!S_ISREG(st.st_mode) || st.st_size < 0) { return false; }
    out.file_size       = static_cast<std::uint64_t>(st.st_size);
    out.dev             = static_cast<std::uint64_t>(st.st_dev);
    out.ino             = static_cast<std::uint64_t>(st.st_ino);
    out.nlink           = static_cast<std::uint64_t>(st.st_nlink);
    out.mtime_sec       = static_cast<std::int64_t>(st.st_mtim.tv_sec);
    out.mtime_nsec      = static_cast<std::int64_t>(st.st_mtim.tv_nsec);
    out.ctime_sec       = static_cast<std::int64_t>(st.st_ctim.tv_sec);
    out.ctime_nsec      = static_cast<std::int64_t>(st.st_ctim.tv_nsec);
    out.has_full_sha256 = false;
    out.full_sha256     = {};
    return true;
}

bool stat_fd(int fd, ShardFingerprint& out) noexcept {
    struct stat st {};
    if (fd < 0 || ::fstat(fd, &st) != 0) { return false; }
    return fill_fingerprint_from_stat(st, out);
}

bool stat_regular_file(const std::filesystem::path& path, ShardFingerprint& out) noexcept {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) { return false; }
    return fill_fingerprint_from_stat(st, out);
}

bool pread_exact(int fd, std::uint64_t offset, void* dst, std::size_t len) noexcept {
    auto* out = static_cast<std::uint8_t*>(dst);
    std::size_t done = 0;
    while (done < len) {
        const ssize_t got = ::pread(fd, out + done, len - done, static_cast<off_t>(offset + done));
        if (got < 0) {
            if (errno == EINTR) { continue; }
            return false;
        }
        if (got == 0) { return false; }
        done += static_cast<std::size_t>(got);
    }
    return true;
}

bool sha256_fd(int fd, std::uint64_t fsz, std::array<std::uint8_t, 32>& out) noexcept {
    Sha256 ctx;
    std::uint8_t buf[65536];
    std::uint64_t off = 0;
    while (off < fsz) {
        const std::size_t take = static_cast<std::size_t>(std::min<std::uint64_t>(fsz - off, sizeof(buf)));
        if (!pread_exact(fd, off, buf, take)) { return false; }
        ctx.update(buf, take);
        off += take;
    }
    out = ctx.finish();
    return true;
}

bool sha256_regular_file(const std::filesystem::path& path, std::array<std::uint8_t, 32>& out) noexcept {
    ScopedFd sfd;
    sfd.fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (sfd.fd < 0) { return false; }
    struct stat st {};
    if (::fstat(sfd.fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0) { return false; }
    return sha256_fd(sfd.fd, static_cast<std::uint64_t>(st.st_size), out);
}

bool fingerprint_safetensors_fd(int fd, ShardFingerprint& out) noexcept {
    if (!stat_fd(fd, out) || out.file_size < 8u) { return false; }
    if (out.file_size <= kFullFileShaThresholdBytes) {
        if (!sha256_fd(fd, out.file_size, out.full_sha256)) { return false; }
        out.has_full_sha256 = true;
    }
    return true;
}

bool shard_stat_matches_open_stat(const ShardFingerprint& fp, const SafetensorsFileStat& st) noexcept {
    return st.valid && fp.file_size == st.file_size && fp.dev == st.dev && fp.ino == st.ino &&
           fp.nlink == st.nlink && fp.mtime_sec == st.mtime_sec && fp.mtime_nsec == st.mtime_nsec &&
           fp.ctime_sec == st.ctime_sec && fp.ctime_nsec == st.ctime_nsec;
}

bool shard_stats_match(const ShardFingerprint& a, const ShardFingerprint& b) noexcept {
    return a.file_size == b.file_size && a.dev == b.dev && a.ino == b.ino && a.nlink == b.nlink &&
           a.mtime_sec == b.mtime_sec && a.mtime_nsec == b.mtime_nsec &&
           a.ctime_sec == b.ctime_sec && a.ctime_nsec == b.ctime_nsec;
}

std::string format_shard_fingerprint_line(std::string_view rel_path, const ShardFingerprint& fp) {
    std::string line;
    line.reserve(rel_path.size() + 144);
    line.append(rel_path);
    line.push_back('\t');
    line.append(std::to_string(fp.file_size));
    line.push_back('\t');
    line.append(std::to_string(fp.dev));
    line.push_back('\t');
    line.append(std::to_string(fp.ino));
    line.push_back('\t');
    line.append(std::to_string(fp.mtime_sec));
    line.push_back('\t');
    line.append(std::to_string(fp.mtime_nsec));
    line.push_back('\t');
    line.append(std::to_string(fp.ctime_sec));
    line.push_back('\t');
    line.append(std::to_string(fp.ctime_nsec));
    line.push_back('\t');
    if (fp.has_full_sha256) {
        line.append(sha256_hex(fp.full_sha256));
    } else {
        line.push_back('-');
    }
    return line;
}

template <typename IntT>
bool parse_exact_int(std::string_view sv, IntT& out) noexcept {
    if (sv.empty()) { return false; }
    const char* begin = sv.data();
    const char* end   = sv.data() + sv.size();
    auto res          = std::from_chars(begin, end, out, 10);
    return res.ec == std::errc{} && res.ptr == end;
}

}  // namespace

bool weight_pack_enabled() noexcept {
    const char* env = std::getenv("INGINE_DISABLE_WEIGHT_PACK");
    if (!env || env[0] == '\0') { return true; }
    if (std::strcmp(env, "0") == 0) { return true; }
    if (std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0) { return true; }
    return false;
}

bool is_weight_pack_file(const std::filesystem::path& path) noexcept {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) { return false; }
    char magic[sizeof(kWeightPackMagic)]{};
    const ssize_t n = ::pread(fd, magic, sizeof(magic), 0);
    ::close(fd);
    return n == static_cast<ssize_t>(sizeof(magic)) &&
           std::memcmp(magic, kWeightPackMagic, sizeof(kWeightPackMagic)) == 0;
}

std::uint64_t weight_pack_fnv1a(std::string_view text) noexcept {
    std::uint64_t h = 14695981039346656037ull;
    for (unsigned char c : text) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

WeightPack::WeightPack(const std::filesystem::path& path, WeightPackOpenOptions opts) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail_pack_errno(path_, "open failed"); }

    try {
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { fail_pack_errno(path_, "fstat failed"); }
        if (st.st_size < static_cast<off_t>(sizeof(WeightPackHeader))) {
            fail_pack(path_, "file shorter than 256-byte WeightPackHeader");
        }
        file_size_ = static_cast<std::uint64_t>(st.st_size);
        if (file_size_ > std::numeric_limits<std::size_t>::max()) {
            fail_pack(path_, "file exceeds address space");
        }
        const std::size_t fsz = static_cast<std::size_t>(file_size_);

        const long ps = ::sysconf(_SC_PAGESIZE);
        const std::size_t page = ps > 0 ? static_cast<std::size_t>(ps) : 4096ull;
        const std::size_t reserve_body = ((fsz + page - 1) / page) * page;
        if (reserve_body >= fsz && reserve_body <= std::numeric_limits<std::size_t>::max() - kHugePageAlignBytes) {
            const std::size_t span = reserve_body + kHugePageAlignBytes;
            void* raw = ::mmap(nullptr, span, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (raw != MAP_FAILED) {
                const auto raw_addr = reinterpret_cast<std::uintptr_t>(raw);
                const auto aligned  = (raw_addr + kHugePageAlignBytes - 1) & ~(std::uintptr_t(kHugePageAlignBytes) - 1);
                const std::size_t head = aligned - raw_addr;
                const std::size_t tail = span - head - reserve_body;
                if (head > 0) { ::munmap(raw, head); }
                if (tail > 0) { ::munmap(reinterpret_cast<void*>(aligned + reserve_body), tail); }
                void* mapped = ::mmap(reinterpret_cast<void*>(aligned), fsz, PROT_READ,
                                      MAP_PRIVATE | MAP_FIXED, fd_, 0);
                if (mapped != MAP_FAILED) {
                    base_            = static_cast<std::byte*>(mapped);
                    mapped_bytes_    = reserve_body;
                    aligned_2m_base_ = true;
                } else {
                    ::munmap(reinterpret_cast<void*>(aligned), reserve_body);
                }
            }
        }
        if (base_ == nullptr) {
            void* mapped = ::mmap(nullptr, fsz, PROT_READ, MAP_PRIVATE, fd_, 0);
            if (mapped == MAP_FAILED) { fail_pack_errno(path_, "mmap failed"); }
            base_            = static_cast<std::byte*>(mapped);
            mapped_bytes_    = fsz;
            aligned_2m_base_ = ((reinterpret_cast<std::uintptr_t>(base_) % kHugePageAlignBytes) == 0);
        }

        std::memcpy(&header_, base_, sizeof(WeightPackHeader));
        if (std::memcmp(header_.magic, kWeightPackMagic, sizeof(kWeightPackMagic)) != 0) {
            fail_pack(path_, "invalid magic");
        }
        if (header_.version != kWeightPackVersion) {
            fail_pack(path_, "unsupported version " + std::to_string(header_.version));
        }
        if ((header_.flags & ~(kWeightPackFlagHasContentSha256 | kWeightPackFlagLargeAligned2M)) != 0u) {
            fail_pack(path_, "unsupported flags " + std::to_string(header_.flags));
        }
        if (header_.file_size != file_size_) {
            fail_pack(path_, "file_size mismatch: header=" + std::to_string(header_.file_size) +
                                 " disk=" + std::to_string(file_size_));
        }
        if (header_.data_offset < sizeof(WeightPackHeader) || header_.data_offset > file_size_ ||
            (header_.data_offset % kHugePageAlignBytes) != 0) {
            fail_pack(path_, "invalid or non-2MiB-aligned data_offset " + std::to_string(header_.data_offset));
        }
        if (header_.data_size != file_size_ - header_.data_offset) {
            fail_pack(path_, "data_size mismatch");
        }
        if (header_.tensor_alignment < kWeightPackSmallAlignBytes ||
            !is_power_of_two_u64(header_.tensor_alignment)) {
            fail_pack(path_, "invalid tensor_alignment");
        }
        if ((header_.flags & kWeightPackFlagLargeAligned2M) != 0u &&
            header_.tensor_alignment < kHugePageAlignBytes) {
            fail_pack(path_, "kWeightPackFlagLargeAligned2M requires tensor_alignment >= 2 MiB");
        }

        const std::uint64_t dir_bytes =
            static_cast<std::uint64_t>(header_.num_tensors) * sizeof(WeightPackEntry);
        const std::uint64_t meta_bytes =
            static_cast<std::uint64_t>(header_.num_metadata) * sizeof(WeightPackMetaEntry);
        if (header_.dir_offset < sizeof(WeightPackHeader) || header_.dir_offset > header_.data_offset ||
            dir_bytes > header_.data_offset - header_.dir_offset) {
            fail_pack(path_, "directory table outside header region");
        }
        const std::uint64_t dir_end = header_.dir_offset + dir_bytes;
        if (header_.meta_offset < dir_end || header_.meta_offset > header_.data_offset ||
            meta_bytes > header_.data_offset - header_.meta_offset) {
            fail_pack(path_, "metadata table outside header region");
        }
        const std::uint64_t meta_end = header_.meta_offset + meta_bytes;
        if (header_.string_table_offset < meta_end || header_.string_table_offset > header_.data_offset ||
            header_.string_table_bytes > header_.data_offset - header_.string_table_offset) {
            fail_pack(path_, "string table outside header region");
        }
        const std::uint64_t header_end = header_.string_table_offset + header_.string_table_bytes;

        if (opts.verify_header_sha) {
            const auto expected_sha = compute_header_sha256(base_, header_end);
            if (std::memcmp(expected_sha.data(), header_.header_sha256, 32) != 0) {
                fail_pack(path_, "header_sha256 mismatch");
            }
        }

        entries_ = std::span<const WeightPackEntry>(
            reinterpret_cast<const WeightPackEntry*>(base_ + header_.dir_offset), header_.num_tensors);
        meta_entries_ = std::span<const WeightPackMetaEntry>(
            reinterpret_cast<const WeightPackMetaEntry*>(base_ + header_.meta_offset), header_.num_metadata);
        string_table_ = reinterpret_cast<const char*>(base_ + header_.string_table_offset);

        // Validate every metadata slice.
        for (std::size_t i = 0; i < meta_entries_.size(); ++i) {
            const WeightPackMetaEntry& m = meta_entries_[i];
            if (static_cast<std::uint64_t>(m.key_offset) + m.key_len > header_.string_table_bytes ||
                static_cast<std::uint64_t>(m.val_offset) + m.val_len > header_.string_table_bytes) {
                fail_pack(path_, "metadata string slice out of bounds");
            }
        }

        // Validate every directory entry: string bounds, FNV-1a, strict sort order, rank/shape/dtype/data bounds.
        std::vector<std::pair<std::uint64_t, std::uint64_t>> occupied;
        occupied.reserve(entries_.size());
        std::string_view prev_name;
        for (std::size_t i = 0; i < entries_.size(); ++i) {
            const WeightPackEntry& e = entries_[i];
            if (e.name_len == 0 ||
                static_cast<std::uint64_t>(e.name_offset) + e.name_len > header_.string_table_bytes ||
                e.dtype_len == 0 ||
                static_cast<std::uint64_t>(e.dtype_offset) + e.dtype_len > header_.string_table_bytes) {
                fail_pack(path_, "entry " + std::to_string(i) + " string slice out of bounds");
            }
            const std::string_view name = entry_name(e);
            const std::string_view dt   = entry_dtype(e);
            if (e.name_hash != weight_pack_fnv1a(name)) {
                fail_pack(path_, "entry " + std::string(name) + " hash mismatch");
            }
            if (i > 0 && !(prev_name < name)) {
                fail_pack(path_, "directory entries are not strictly sorted by name at " + std::string(name));
            }
            prev_name = name;

            if (e.rank > 6u) { fail_pack(path_, "tensor " + std::string(name) + " rank > 6"); }
            for (std::uint32_t d = e.rank; d < 6u; ++d) {
                if (e.shape[d] != 0) {
                    fail_pack(path_, "tensor " + std::string(name) + " non-zero unused shape dim");
                }
            }
            std::uint64_t elem_count = 1;
            for (std::uint32_t d = 0; d < e.rank; ++d) {
                if (e.shape[d] < 0) {
                    fail_pack(path_, "tensor " + std::string(name) + " has negative dimension");
                }
                const std::uint64_t dim = static_cast<std::uint64_t>(e.shape[d]);
                if (dim != 0 && elem_count > std::numeric_limits<std::uint64_t>::max() / dim) {
                    fail_pack(path_, "tensor " + std::string(name) + " numel overflows");
                }
                elem_count *= dim;
            }
            const auto opt_dt = safetensors_to_dtype(dt);
            const std::uint32_t expected_code = opt_dt ? static_cast<std::uint32_t>(*opt_dt) : 0xFFFFFFFFu;
            if (e.ingine_dtype_code != expected_code) {
                fail_pack(path_, "tensor " + std::string(name) + " ingine_dtype_code mismatch");
            }
            std::size_t elem_bytes = 0;
            try {
                elem_bytes = safetensors_dtype_size(dt);
            } catch (const std::invalid_argument&) {
                elem_bytes = 0;
            }
            if (elem_bytes != 0) {
                if (elem_count > std::numeric_limits<std::uint64_t>::max() / elem_bytes ||
                    elem_count * elem_bytes != e.data_bytes) {
                    fail_pack(path_, "tensor " + std::string(name) + " data_bytes mismatch with dtype/shape");
                }
            }

            if (e.data_offset < header_.data_offset || e.data_offset > file_size_ ||
                e.data_bytes > file_size_ - e.data_offset) {
                fail_pack(path_, "tensor " + std::string(name) + " payload range out of bounds");
            }
            if (e.data_bytes >= header_.large_tensor_threshold &&
                (e.data_offset % header_.tensor_alignment) != 0) {
                fail_pack(path_, "large tensor " + std::string(name) + " violates tensor_alignment");
            }
            if (e.data_bytes > 0) {
                occupied.emplace_back(e.data_offset, e.data_offset + e.data_bytes);
            }
        }

        std::sort(occupied.begin(), occupied.end());
        for (std::size_t i = 1; i < occupied.size(); ++i) {
            if (occupied[i].first < occupied[i - 1].second) {
                fail_pack(path_, "overlapping tensor payload ranges");
            }
        }

        if (header_.data_size > 0) {
            std::byte* data_ptr = base_ + header_.data_offset;
            const std::size_t data_len = static_cast<std::size_t>(header_.data_size);
#ifdef MADV_HUGEPAGE
            if (opts.advise_hugepage && hugepages_enabled()) {
                hugepage_advised_ = (::madvise(data_ptr, data_len, MADV_HUGEPAGE) == 0);
            }
#endif
            if (opts.advise_sequential) { (void)::madvise(data_ptr, data_len, MADV_SEQUENTIAL); }
            if (opts.advise_willneed) { (void)::madvise(data_ptr, data_len, MADV_WILLNEED); }
            if (opts.numa_policy != NumaPolicy::Default) {
                numa_bound_ = bind_memory_numa(data_ptr, data_len, opts.numa_node, opts.numa_policy, false);
            }
        }

        if (opts.verify_content_sha && !verify_content_sha256()) {
            fail_pack(path_, "content_sha256 verification failed");
        }
    } catch (...) {
        release();
        throw;
    }
}

WeightPack::~WeightPack() { release(); }

WeightPack::WeightPack(WeightPack&& other) noexcept
    : path_(std::move(other.path_)),
      fd_(std::exchange(other.fd_, -1)),
      base_(std::exchange(other.base_, nullptr)),
      mapped_bytes_(std::exchange(other.mapped_bytes_, 0)),
      file_size_(std::exchange(other.file_size_, 0)),
      aligned_2m_base_(std::exchange(other.aligned_2m_base_, false)),
      hugepage_advised_(std::exchange(other.hugepage_advised_, false)),
      numa_bound_(std::exchange(other.numa_bound_, false)),
      header_(other.header_),
      entries_(other.entries_),
      meta_entries_(other.meta_entries_),
      string_table_(std::exchange(other.string_table_, nullptr)) {
    other.header_       = WeightPackHeader{};
    other.entries_      = {};
    other.meta_entries_ = {};
}

WeightPack& WeightPack::operator=(WeightPack&& other) noexcept {
    if (this != &other) {
        release();
        path_             = std::move(other.path_);
        fd_               = std::exchange(other.fd_, -1);
        base_             = std::exchange(other.base_, nullptr);
        mapped_bytes_     = std::exchange(other.mapped_bytes_, 0);
        file_size_        = std::exchange(other.file_size_, 0);
        aligned_2m_base_  = std::exchange(other.aligned_2m_base_, false);
        hugepage_advised_ = std::exchange(other.hugepage_advised_, false);
        numa_bound_       = std::exchange(other.numa_bound_, false);
        header_           = other.header_;
        entries_          = other.entries_;
        meta_entries_     = other.meta_entries_;
        string_table_     = std::exchange(other.string_table_, nullptr);
        other.header_       = WeightPackHeader{};
        other.entries_      = {};
        other.meta_entries_ = {};
    }
    return *this;
}

void WeightPack::release() noexcept {
    if (base_ != nullptr && mapped_bytes_ > 0) {
        ::munmap(base_, mapped_bytes_);
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    fd_               = -1;
    base_             = nullptr;
    mapped_bytes_     = 0;
    file_size_        = 0;
    aligned_2m_base_  = false;
    hugepage_advised_ = false;
    numa_bound_       = false;
    header_           = WeightPackHeader{};
    entries_          = {};
    meta_entries_     = {};
    string_table_     = nullptr;
}

std::string_view WeightPack::slice_string(std::uint32_t off, std::uint32_t len) const noexcept {
    if (!string_table_ || len == 0) { return {}; }
    return std::string_view(string_table_ + off, len);
}

std::string_view WeightPack::entry_name(const WeightPackEntry& entry) const noexcept {
    return slice_string(entry.name_offset, entry.name_len);
}

std::string_view WeightPack::entry_dtype(const WeightPackEntry& entry) const noexcept {
    return slice_string(entry.dtype_offset, entry.dtype_len);
}

const WeightPackEntry* WeightPack::find(std::string_view name) const noexcept {
    const auto it = std::lower_bound(
        entries_.begin(), entries_.end(), name,
        [this](const WeightPackEntry& entry, std::string_view key) { return entry_name(entry) < key; });
    if (it != entries_.end() && entry_name(*it) == name) { return &(*it); }
    return nullptr;
}

const WeightPackEntry& WeightPack::at(std::string_view name) const {
    const WeightPackEntry* e = find(name);
    if (e == nullptr) { fail_pack(path_, "no tensor named " + std::string(name)); }
    return *e;
}

const std::byte* WeightPack::tensor_data(const WeightPackEntry& entry) const noexcept {
    return base_ + entry.data_offset;
}

const std::byte* WeightPack::tensor_data(std::string_view name) const {
    return tensor_data(at(name));
}

Tensor WeightPack::tensor_view(std::string_view name) const {
    const WeightPackEntry& e = at(name);
    if (e.rank > 4u) {
        throw std::invalid_argument("tensor " + std::string(name) + ": rank exceeds ingine::Tensor rank 4");
    }
    const auto dt = safetensors_to_dtype(entry_dtype(e));
    if (!dt) {
        throw std::invalid_argument("tensor " + std::string(name) + ": no ingine DType for " +
                                    std::string(entry_dtype(e)));
    }
    Tensor t;
    t.data  = const_cast<std::byte*>(tensor_data(e));
    t.dtype = *dt;
    for (std::uint32_t i = 0; i < e.rank; ++i) {
        const std::int64_t d = e.shape[e.rank - 1u - i];
        if (d <= 0 || d > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("tensor " + std::string(name) + ": dimension " + std::to_string(d) +
                                        " not representable in ingine::Tensor");
        }
        t.ne[i] = static_cast<std::int32_t>(d);
    }
    t.nb[0] = static_cast<std::int64_t>(dtype_size(t.dtype));
    for (int i = 1; i < 4; ++i) { t.nb[i] = t.nb[i - 1] * t.ne[i - 1]; }
    return t;
}

std::optional<std::string_view> WeightPack::find_metadata(std::string_view key) const noexcept {
    for (const WeightPackMetaEntry& m : meta_entries_) {
        if (slice_string(m.key_offset, m.key_len) == key) {
            return slice_string(m.val_offset, m.val_len);
        }
    }
    return std::nullopt;
}

std::map<std::string, std::string> WeightPack::metadata() const {
    std::map<std::string, std::string> out;
    for (const WeightPackMetaEntry& m : meta_entries_) {
        out.emplace(std::string(slice_string(m.key_offset, m.key_len)),
                    std::string(slice_string(m.val_offset, m.val_len)));
    }
    return out;
}

bool weight_pack_matches_safetensors_dir(const WeightPack& pack, const std::filesystem::path& dir) noexcept {
    try {
        namespace fs = std::filesystem;
        const auto src_ver = pack.find_metadata("ingpack.source_version");
        if (!src_ver || *src_ver != "1") { return false; }

        std::error_code ec;
        const bool has_index = fs::exists(dir / "model.safetensors.index.json", ec);
        if (ec) { return false; }
        const bool has_single = fs::exists(dir / "model.safetensors", ec);
        if (ec) { return false; }
        const std::string_view dir_kind = has_index ? "index" : (has_single ? "single" : "");
        if (dir_kind.empty()) { return false; }

        const auto src_kind = pack.find_metadata("ingpack.source_kind");
        if (!src_kind || *src_kind != dir_kind) { return false; }

        if (dir_kind == "index") {
            const auto expected_idx = pack.find_metadata("ingpack.source_index_sha256");
            if (!expected_idx || expected_idx->size() != 64u) { return false; }
            std::array<std::uint8_t, 32> actual_idx{};
            if (!sha256_regular_file(dir / "model.safetensors.index.json", actual_idx)) { return false; }
            if (sha256_hex(actual_idx) != *expected_idx) { return false; }
        }

        const auto shards_meta = pack.find_metadata("ingpack.source_shards");
        if (!shards_meta) { return false; }

        struct stat pack_st {};
        if (pack.fd() < 0 || ::fstat(pack.fd(), &pack_st) != 0) { return false; }
        const std::int64_t pack_csec  = static_cast<std::int64_t>(pack_st.st_ctim.tv_sec);
        const std::int64_t pack_cnsec = static_cast<std::int64_t>(pack_st.st_ctim.tv_nsec);

        std::size_t num_records = 0;
        std::size_t pos         = 0;
        while (pos < shards_meta->size()) {
            const std::size_t nl       = shards_meta->find('\n', pos);
            const std::string_view line = (nl == std::string_view::npos)
                                              ? shards_meta->substr(pos)
                                              : shards_meta->substr(pos, nl - pos);
            pos = (nl == std::string_view::npos) ? shards_meta->size() : (nl + 1);
            if (line.empty()) { continue; }
            ++num_records;

            std::string_view fields[9]{};
            std::size_t fpos = 0;
            for (int f = 0; f < 8; ++f) {
                const std::size_t tab = line.find('\t', fpos);
                if (tab == std::string_view::npos) { return false; }
                fields[f] = line.substr(fpos, tab - fpos);
                fpos      = tab + 1;
            }
            if (line.find('\t', fpos) != std::string_view::npos) { return false; }
            fields[8] = line.substr(fpos);

            const std::string_view rel_path = fields[0];
            if (rel_path.empty()) { return false; }
            const fs::path rel(rel_path);
            if (rel.is_absolute()) { return false; }
            for (const fs::path& comp : rel) {
                if (comp == ".." || comp == ".") { return false; }
            }
            if (dir_kind == "single" && rel_path != "model.safetensors") { return false; }

            std::uint64_t exp_size  = 0;
            std::uint64_t exp_dev   = 0;
            std::uint64_t exp_ino   = 0;
            std::int64_t exp_msec   = 0;
            std::int64_t exp_mnsec  = 0;
            std::int64_t exp_csec   = 0;
            std::int64_t exp_cnsec  = 0;
            if (!parse_exact_int(fields[1], exp_size) || !parse_exact_int(fields[2], exp_dev) ||
                !parse_exact_int(fields[3], exp_ino) || !parse_exact_int(fields[4], exp_msec) ||
                !parse_exact_int(fields[5], exp_mnsec) || !parse_exact_int(fields[6], exp_csec) ||
                !parse_exact_int(fields[7], exp_cnsec)) {
                return false;
            }
            const std::string_view exp_sha_field = fields[8];
            if (exp_sha_field != "-" && exp_sha_field.size() != 64u) { return false; }

            ShardFingerprint actual{};
            if (!stat_regular_file(dir / rel, actual)) { return false; }
            if (actual.file_size != exp_size) { return false; }

            const bool identity_matches =
                actual.dev == exp_dev && actual.ino == exp_ino &&
                actual.mtime_sec == exp_msec && actual.mtime_nsec == exp_mnsec &&
                actual.ctime_sec == exp_csec && actual.ctime_nsec == exp_cnsec;
            const bool strictly_older_than_pack =
                timespec_strictly_less(actual.ctime_sec, actual.ctime_nsec, pack_csec, pack_cnsec);
            if (identity_matches && strictly_older_than_pack) { continue; }

            if (exp_sha_field == "-") { return false; }
            std::array<std::uint8_t, 32> actual_sha{};
            if (!sha256_regular_file(dir / rel, actual_sha)) { return false; }
            if (sha256_hex(actual_sha) != exp_sha_field) { return false; }
        }

        if (dir_kind == "single") {
            return num_records == 1;
        }
        return num_records >= 1 || pack.num_tensors() == 0;
    } catch (...) {
        return false;
    }
}

void WeightPack::read_range(std::uint64_t offset, std::uint64_t bytes, void* dst) const {
    if (offset > file_size_ || bytes > file_size_ - offset) {
        fail_pack(path_, "read of " + std::to_string(bytes) + " bytes at " + std::to_string(offset) +
                             " exceeds the file");
    }
    if (bytes > 0) {
        std::memcpy(dst, base_ + offset, static_cast<std::size_t>(bytes));
    }
}

void WeightPack::prefault(bool parallel) const {
    if (!base_ || file_size_ <= header_.data_offset) { return; }
    const long ps = ::sysconf(_SC_PAGESIZE);
    const std::size_t stride = ps > 0 ? static_cast<std::size_t>(ps) : 4096ull;
    const auto* start = reinterpret_cast<const volatile std::uint8_t*>(base_ + header_.data_offset);
    const std::size_t total = static_cast<std::size_t>(file_size_ - header_.data_offset);
    auto touch_slice = [&](std::size_t begin, std::size_t end) noexcept {
        std::uint8_t acc = 0;
        for (std::size_t off = begin; off < end; off += stride) { acc ^= start[off]; }
        if (end > begin) { acc ^= start[end - 1]; }
        volatile std::uint8_t sink = acc;
        (void)sink;
    };
    if (!parallel || total < (8ull << 20)) {
        touch_slice(0, total);
        return;
    }
    const int n_threads = 4;
    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    const std::size_t chunk = (total + n_threads - 1) / n_threads;
    try {
        for (int t = 0; t < n_threads; ++t) {
            const std::size_t b = static_cast<std::size_t>(t) * chunk;
            if (b >= total) { break; }
            const std::size_t e = std::min(total, b + chunk);
            threads.emplace_back([&, b, e]() { touch_slice(b, e); });
        }
    } catch (...) {
        for (auto& th : threads) {
            if (th.joinable()) { th.join(); }
        }
        throw;
    }
    for (auto& th : threads) { th.join(); }
}

bool WeightPack::verify_content_sha256() const {
    if ((header_.flags & kWeightPackFlagHasContentSha256) == 0u) { return false; }
    std::vector<std::pair<std::string_view, std::array<std::uint8_t, 32>>> sorted_digests;
    sorted_digests.reserve(entries_.size());
    for (const WeightPackEntry& e : entries_) {
        std::vector<std::array<std::uint8_t, 32>> cshas;
        if (e.data_bytes > 0) {
            const std::size_t n_chunks =
                static_cast<std::size_t>((e.data_bytes + kWeightPackHashChunkBytes - 1) / kWeightPackHashChunkBytes);
            cshas.reserve(n_chunks);
            const std::byte* ptr = tensor_data(e);
            for (std::uint64_t off = 0; off < e.data_bytes; off += kWeightPackHashChunkBytes) {
                const std::size_t take =
                    static_cast<std::size_t>(std::min<std::uint64_t>(e.data_bytes - off, kWeightPackHashChunkBytes));
                cshas.push_back(sha256_bytes(ptr + off, take));
            }
        }
        std::span<const std::int64_t> shape_span(e.shape, e.rank);
        const auto td = tensor_digest_from_chunks(entry_name(e), entry_dtype(e), shape_span, e.data_bytes, cshas);
        sorted_digests.emplace_back(entry_name(e), td);
    }
    const auto computed = finalize_pack_content_sha256(sorted_digests);
    return std::memcmp(computed.data(), header_.content_sha256, 32) == 0;
}

void WeightPackBuilder::add_metadata(std::string key, std::string value) {
    metadata_.insert_or_assign(std::move(key), std::move(value));
}

void WeightPackBuilder::add_tensor(std::string name, std::string dtype, std::vector<std::int64_t> shape,
                                   std::uint64_t bytes, DataWriter writer) {
    if (name.empty() || dtype.empty()) {
        throw std::invalid_argument("WeightPackBuilder: tensor name and dtype must be non-empty");
    }
    if (shape.size() > 6) {
        throw std::invalid_argument("WeightPackBuilder: tensor rank > 6 for " + name);
    }
    PendingTensor pt;
    pt.name         = std::move(name);
    pt.dtype        = std::move(dtype);
    pt.shape        = std::move(shape);
    pt.bytes        = bytes;
    pt.writer       = std::move(writer);
    pt.source_order = tensors_.size();
    tensors_.push_back(std::move(pt));
}

WeightPackBuildStats WeightPackBuilder::write(const std::filesystem::path& output_path,
                                              WeightPackBuildOptions opts) {
    return write_impl(output_path, opts, {});
}

WeightPackBuildStats WeightPackBuilder::write_impl(const std::filesystem::path& output_path,
                                                   WeightPackBuildOptions opts,
                                                   const std::function<void()>& pre_publish_check) {
    const auto t0 = std::chrono::steady_clock::now();
    if (opts.tensor_alignment < kWeightPackSmallAlignBytes || !is_power_of_two_u64(opts.tensor_alignment)) {
        throw std::invalid_argument("WeightPackBuilder: invalid tensor_alignment");
    }
    if (opts.tensor_alignment < kHugePageAlignBytes) {
        throw std::invalid_argument("WeightPackBuilder: tensor_alignment must be >= 2 MiB for Phase B");
    }

    // Build sorted view indices and check for duplicate tensor names.
    std::vector<std::size_t> sorted_idx(tensors_.size());
    for (std::size_t i = 0; i < tensors_.size(); ++i) { sorted_idx[i] = i; }
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](std::size_t a, std::size_t b) { return tensors_[a].name < tensors_[b].name; });
    for (std::size_t i = 1; i < sorted_idx.size(); ++i) {
        if (tensors_[sorted_idx[i - 1]].name == tensors_[sorted_idx[i]].name) {
            throw std::invalid_argument("WeightPackBuilder: duplicate tensor " + tensors_[sorted_idx[i]].name);
        }
    }

    // Build string table and metadata/directory offsets.
    std::string string_table;
    auto intern = [&](std::string_view sv) -> std::pair<std::uint32_t, std::uint32_t> {
        if (sv.empty()) { return {0u, 0u}; }
        if (string_table.size() + sv.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("WeightPackBuilder: string table exceeds 4 GiB");
        }
        const std::uint32_t off = static_cast<std::uint32_t>(string_table.size());
        const std::uint32_t len = static_cast<std::uint32_t>(sv.size());
        string_table.append(sv.data(), sv.size());
        return {off, len};
    };

    std::vector<WeightPackEntry> dir(sorted_idx.size());
    std::vector<std::size_t> orig_to_sorted(tensors_.size());
    for (std::size_t si = 0; si < sorted_idx.size(); ++si) {
        const PendingTensor& pt = tensors_[sorted_idx[si]];
        orig_to_sorted[sorted_idx[si]] = si;
        WeightPackEntry& e = dir[si];
        e = WeightPackEntry{};
        e.name_hash = weight_pack_fnv1a(pt.name);
        auto [no, nl] = intern(pt.name);
        auto [doff, dl] = intern(pt.dtype);
        e.name_offset = no;
        e.name_len = nl;
        e.dtype_offset = doff;
        e.dtype_len = dl;
        e.rank = static_cast<std::uint32_t>(pt.shape.size());
        const auto dt = safetensors_to_dtype(pt.dtype);
        e.ingine_dtype_code = dt ? static_cast<std::uint32_t>(*dt) : 0xFFFFFFFFu;
        e.data_bytes = pt.bytes;
        for (std::size_t d = 0; d < pt.shape.size(); ++d) { e.shape[d] = pt.shape[d]; }
    }

    std::vector<WeightPackMetaEntry> meta_dir;
    meta_dir.reserve(metadata_.size());
    for (const auto& [k, v] : metadata_) {
        WeightPackMetaEntry me{};
        auto [ko, kl] = intern(k);
        auto [vo, vl] = intern(v);
        me.key_offset = ko;
        me.key_len = kl;
        me.val_offset = vo;
        me.val_len = vl;
        meta_dir.push_back(me);
    }

    const std::uint64_t dir_offset = sizeof(WeightPackHeader);
    const std::uint64_t meta_offset = dir_offset + static_cast<std::uint64_t>(dir.size()) * sizeof(WeightPackEntry);
    const std::uint64_t string_table_offset =
        meta_offset + static_cast<std::uint64_t>(meta_dir.size()) * sizeof(WeightPackMetaEntry);
    const std::uint64_t header_end = string_table_offset + static_cast<std::uint64_t>(string_table.size());
    const std::uint64_t data_offset = align_up_u64(header_end, kHugePageAlignBytes);

    // Two-zone data layout: Zone 1 small (< threshold) @ 256B, Zone 2 large (>= threshold) @ tensor_alignment.
    std::vector<std::size_t> write_order;
    write_order.reserve(tensors_.size());
    std::uint64_t cursor = data_offset;
    std::uint64_t small_count = 0;
    std::uint64_t large_count = 0;
    std::uint64_t payload_bytes = 0;

    for (std::size_t i = 0; i < tensors_.size(); ++i) {
        if (tensors_[i].bytes < opts.large_tensor_threshold) {
            cursor = align_up_u64(cursor, kWeightPackSmallAlignBytes);
            dir[orig_to_sorted[i]].data_offset = cursor;
            cursor += tensors_[i].bytes;
            payload_bytes += tensors_[i].bytes;
            write_order.push_back(i);
            ++small_count;
        }
    }
    for (std::size_t i = 0; i < tensors_.size(); ++i) {
        if (tensors_[i].bytes >= opts.large_tensor_threshold) {
            cursor = align_up_u64(cursor, opts.tensor_alignment);
            dir[orig_to_sorted[i]].data_offset = cursor;
            cursor += tensors_[i].bytes;
            payload_bytes += tensors_[i].bytes;
            write_order.push_back(i);
            ++large_count;
        }
    }
    const std::uint64_t file_size = cursor;

    if (output_path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(output_path.parent_path(), ec);
    }
    const std::filesystem::path tmp_path =
        output_path.string() + ".tmp." + std::to_string(static_cast<long>(::getpid()));
    int fd = ::open(tmp_path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) { fail_pack_errno(tmp_path, "open temp output failed"); }

    auto cleanup_tmp = [&]() noexcept {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
    };

    try {
        WeightPackHeader hdr{};
        std::memcpy(hdr.magic, kWeightPackMagic, sizeof(kWeightPackMagic));
        hdr.version = kWeightPackVersion;
        hdr.flags = kWeightPackFlagLargeAligned2M |
                    (opts.compute_content_sha256 ? kWeightPackFlagHasContentSha256 : 0u);
        hdr.file_size = file_size;
        hdr.data_offset = data_offset;
        hdr.data_size = file_size - data_offset;
        hdr.tensor_alignment = opts.tensor_alignment;
        hdr.large_tensor_threshold = opts.large_tensor_threshold;
        hdr.num_tensors = static_cast<std::uint32_t>(dir.size());
        hdr.num_metadata = static_cast<std::uint32_t>(meta_dir.size());
        hdr.dir_offset = dir_offset;
        hdr.meta_offset = meta_offset;
        hdr.string_table_offset = string_table_offset;
        hdr.string_table_bytes = static_cast<std::uint64_t>(string_table.size());

        // Write preliminary header + directory + string table + pad to data_offset, then stream payload.
        write_all_fd(fd, &hdr, sizeof(hdr), tmp_path);
        if (!dir.empty()) {
            write_all_fd(fd, dir.data(), dir.size() * sizeof(WeightPackEntry), tmp_path);
        }
        if (!meta_dir.empty()) {
            write_all_fd(fd, meta_dir.data(), meta_dir.size() * sizeof(WeightPackMetaEntry), tmp_path);
        }
        if (!string_table.empty()) {
            write_all_fd(fd, string_table.data(), string_table.size(), tmp_path);
        }
        write_zeros_fd(fd, data_offset - header_end, tmp_path);

        std::vector<std::array<std::uint8_t, 32>> per_sorted_tensor_sha(dir.size());
        std::uint64_t file_pos = data_offset;

        for (std::size_t idx : write_order) {
            const PendingTensor& pt = tensors_[idx];
            const std::size_t si = orig_to_sorted[idx];
            const std::uint64_t target_off = dir[si].data_offset;
            if (target_off > file_pos) {
                write_zeros_fd(fd, target_off - file_pos, tmp_path);
                file_pos = target_off;
            }
            std::vector<std::array<std::uint8_t, 32>> chunk_shas;
            Sha256 chunk_ctx;
            std::size_t chunk_filled = 0;
            std::uint64_t written_tensor = 0;

            auto emit = [&](const void* data, std::size_t len) {
                if (len == 0) { return; }
                write_all_fd(fd, data, len, tmp_path);
                file_pos += len;
                written_tensor += len;
                if (opts.compute_content_sha256) {
                    const auto* p = static_cast<const std::uint8_t*>(data);
                    std::size_t rem = len;
                    while (rem > 0) {
                        const std::size_t take = std::min(rem, kWeightPackHashChunkBytes - chunk_filled);
                        chunk_ctx.update(p, take);
                        chunk_filled += take;
                        p += take;
                        rem -= take;
                        if (chunk_filled == kWeightPackHashChunkBytes) {
                            chunk_shas.push_back(chunk_ctx.finish());
                            chunk_ctx = Sha256{};
                            chunk_filled = 0;
                        }
                    }
                }
            };
            if (pt.bytes > 0) {
                if (!pt.writer) {
                    throw std::invalid_argument("WeightPackBuilder: null writer for non-empty tensor " + pt.name);
                }
                pt.writer(emit);
            }
            if (written_tensor != pt.bytes) {
                throw std::runtime_error("WeightPackBuilder: tensor " + pt.name + " emitted " +
                                         std::to_string(written_tensor) + " bytes, expected " +
                                         std::to_string(pt.bytes));
            }
            if (opts.compute_content_sha256) {
                if (chunk_filled > 0) { chunk_shas.push_back(chunk_ctx.finish()); }
                per_sorted_tensor_sha[si] = tensor_digest_from_chunks(pt.name, pt.dtype, pt.shape, pt.bytes, chunk_shas);
            }
        }

        if (opts.compute_content_sha256) {
            std::vector<std::pair<std::string_view, std::array<std::uint8_t, 32>>> sorted_pairs;
            sorted_pairs.reserve(dir.size());
            for (std::size_t si = 0; si < sorted_idx.size(); ++si) {
                sorted_pairs.emplace_back(tensors_[sorted_idx[si]].name, per_sorted_tensor_sha[si]);
            }
            const auto csha = finalize_pack_content_sha256(sorted_pairs);
            std::memcpy(hdr.content_sha256, csha.data(), 32);
        }

        // Compute header_sha256 over [0, header_end) with header_sha256 zeroed.
        {
            constexpr std::size_t kHashOff = offsetof(WeightPackHeader, header_sha256);
            constexpr std::size_t kHashLen = sizeof(WeightPackHeader::header_sha256);
            static constexpr std::uint8_t kZeros[kHashLen]{};
            Sha256 hctx;
            hctx.update(&hdr, kHashOff);
            hctx.update(kZeros, kHashLen);
            hctx.update(reinterpret_cast<const std::uint8_t*>(&hdr) + kHashOff + kHashLen,
                        sizeof(WeightPackHeader) - (kHashOff + kHashLen));
            if (!dir.empty()) {
                hctx.update(dir.data(), dir.size() * sizeof(WeightPackEntry));
            }
            if (!meta_dir.empty()) {
                hctx.update(meta_dir.data(), meta_dir.size() * sizeof(WeightPackMetaEntry));
            }
            if (!string_table.empty()) {
                hctx.update(string_table.data(), string_table.size());
            }
            const auto hsha = hctx.finish();
            std::memcpy(hdr.header_sha256, hsha.data(), 32);
        }

        // Rewrite finalized header at offset 0.
        if (::lseek(fd, 0, SEEK_SET) < 0) { fail_pack_errno(tmp_path, "lseek(0) failed"); }
        write_all_fd(fd, &hdr, sizeof(hdr), tmp_path);
        if (::fsync(fd) != 0) { fail_pack_errno(tmp_path, "fsync failed"); }
        const int close_fd = fd;
        fd = -1;
        if (::close(close_fd) != 0) { fail_pack_errno(tmp_path, "close failed"); }

        if (pre_publish_check) { pre_publish_check(); }

        if (::rename(tmp_path.c_str(), output_path.c_str()) != 0) {
            const int e = errno;
            std::error_code ec;
            std::filesystem::remove(tmp_path, ec);
            throw std::runtime_error(std::string("weight_pack rename failed: ") + std::strerror(e));
        }
        sync_parent_dir(output_path);

        WeightPackBuildStats stats;
        stats.num_tensors   = dir.size();
        stats.small_tensors = small_count;
        stats.large_tensors = large_count;
        stats.file_size     = file_size;
        stats.data_offset   = data_offset;
        stats.payload_bytes = payload_bytes;
        std::memcpy(stats.content_sha256.data(), hdr.content_sha256, 32);
        stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return stats;
    } catch (...) {
        cleanup_tmp();
        throw;
    }
}

WeightPackBuildStats WeightPackBuilder::pack_from_safetensors(const std::filesystem::path& src_path,
                                                              const std::filesystem::path& output_path,
                                                              WeightPackBuildOptions opts) {
    namespace fs = std::filesystem;
    fs::path authoritative_src = src_path;
    fs::path base_dir;
    std::string source_kind;
    if (fs::is_directory(src_path)) {
        if (fs::exists(src_path / "model.safetensors.index.json")) {
            source_kind       = "index";
            authoritative_src = src_path / "model.safetensors.index.json";
            base_dir          = src_path;
        } else if (fs::exists(src_path / "model.safetensors")) {
            source_kind       = "single";
            authoritative_src = src_path / "model.safetensors";
            base_dir          = src_path;
        } else {
            throw std::runtime_error("pack_from_safetensors " + src_path.string() +
                                     ": no model.safetensors.index.json or model.safetensors");
        }
    } else if (src_path.extension() == ".json") {
        source_kind       = (src_path.filename() == "model.safetensors.index.json") ? "index" : "custom_index";
        authoritative_src = src_path;
        base_dir          = src_path.parent_path();
    } else {
        source_kind       = (src_path.filename() == "model.safetensors") ? "single" : "file";
        authoritative_src = src_path;
        base_dir          = src_path.parent_path();
    }

    const SafetensorsModel model(authoritative_src, SafetensorsOpenMode::SafetensorsOnly);
    if (auto hook = take_hook(g_next_after_open_hook)) { hook(); }

    WeightPackBuilder builder;
    if (model.files().size() == 1) {
        for (const auto& [k, v] : model.files()[0]->metadata()) {
            builder.add_metadata(k, v);
        }
    }
    builder.add_metadata("ingpack.source_version", "1");
    builder.add_metadata("ingpack.source_kind", source_kind);
    if (source_kind == "index" || source_kind == "custom_index") {
        if (!model.index_sha256()) {
            fail_pack(authoritative_src, "missing parsed index sha256 for provenance");
        }
        builder.add_metadata("ingpack.source_index_sha256", sha256_hex(*model.index_sha256()));
    }
    struct CapturedShard {
        const SafetensorsFile* file = nullptr;
        ShardFingerprint fp{};
    };
    std::vector<CapturedShard> captured_shards;
    bool any_large_shard_without_sha = false;
    bool has_max_ctime               = false;
    std::int64_t max_csec            = 0;
    std::int64_t max_cnsec           = 0;
    {
        struct ShardRef {
            std::string rel_path;
            const SafetensorsFile* file = nullptr;
        };
        std::vector<ShardRef> shard_refs;
        shard_refs.reserve(model.files().size());
        const fs::path norm_base = base_dir.lexically_normal();
        for (const auto& file_ptr : model.files()) {
            std::string rel_str;
            if (source_kind == "index" || source_kind == "custom_index") {
                if (norm_base.empty() || norm_base == ".") {
                    rel_str = file_ptr->path().lexically_normal().generic_string();
                } else {
                    rel_str = file_ptr->path().lexically_normal().lexically_relative(norm_base).generic_string();
                }
            } else {
                rel_str = file_ptr->path().filename().string();
            }
            shard_refs.push_back(ShardRef{std::move(rel_str), file_ptr.get()});
        }
        std::sort(shard_refs.begin(), shard_refs.end(),
                  [](const ShardRef& a, const ShardRef& b) { return a.rel_path < b.rel_path; });
        shard_refs.erase(std::unique(shard_refs.begin(), shard_refs.end(),
                                     [](const ShardRef& a, const ShardRef& b) { return a.rel_path == b.rel_path; }),
                         shard_refs.end());

        std::string shards_manifest;
        captured_shards.reserve(shard_refs.size());
        for (const ShardRef& sr : shard_refs) {
            ShardFingerprint fp{};
            if (!fingerprint_safetensors_fd(sr.file->fd(), fp)) {
                fail_pack(sr.file->path(), "cannot fingerprint open source shard fd for provenance");
            }
            if (!shard_stat_matches_open_stat(fp, sr.file->open_stat())) {
                fail_pack(sr.file->path(), "source shard changed in-place after open");
            }
            if (!fp.has_full_sha256) { any_large_shard_without_sha = true; }
            if (!has_max_ctime || timespec_strictly_less(max_csec, max_cnsec, fp.ctime_sec, fp.ctime_nsec)) {
                has_max_ctime = true;
                max_csec      = fp.ctime_sec;
                max_cnsec     = fp.ctime_nsec;
            }
            if (!shards_manifest.empty()) { shards_manifest.push_back('\n'); }
            shards_manifest.append(format_shard_fingerprint_line(sr.rel_path, fp));
            captured_shards.push_back(CapturedShard{sr.file, fp});
        }
        builder.add_metadata("ingpack.source_shards", std::move(shards_manifest));
    }
    if (auto hook = take_hook(g_next_before_copy_hook)) { hook(); }
    const std::size_t io_chunk = std::max<std::size_t>(opts.io_chunk_bytes, std::size_t(1) << 20);
    auto scratch = std::make_shared<std::vector<std::byte>>();

    // Preserve physical file / tensor order across files so sequential NVMe pread stays friendly.
    for (const auto& file_ptr : model.files()) {
        for (const SafetensorsTensor& t : file_ptr->tensors()) {
            if (!model.contains(t.name)) { continue; }
            const SafetensorsRef ref = model.at(t.name);
            if (ref.file != file_ptr.get()) { continue; }
            const SafetensorsFile* sf = ref.file;
            const std::uint64_t off   = ref->offset;
            const std::uint64_t len   = ref->bytes;
            builder.add_tensor(
                ref->name, ref->dtype, ref->shape, len,
                [sf, off, len, io_chunk, scratch](const std::function<void(const void*, std::size_t)>& emit) {
                    if (scratch->size() < std::min<std::uint64_t>(len, io_chunk)) {
                        scratch->resize(static_cast<std::size_t>(std::min<std::uint64_t>(len, io_chunk)));
                    }
                    std::uint64_t done = 0;
                    while (done < len) {
                        const std::size_t take =
                            static_cast<std::size_t>(std::min<std::uint64_t>(len - done, io_chunk));
                        sf->read_range(off + done, take, scratch->data());
                        emit(scratch->data(), take);
                        done += take;
                    }
                });
        }
    }
    const bool is_indexed = (source_kind == "index" || source_kind == "custom_index");
    auto pre_publish_check = [&captured_shards, &authoritative_src, is_indexed, &model]() {
        if (is_indexed) {
            std::array<std::uint8_t, 32> cur_idx_sha{};
            if (!model.index_sha256() || !sha256_regular_file(authoritative_src, cur_idx_sha) ||
                cur_idx_sha != *model.index_sha256()) {
                fail_pack(authoritative_src, "source snapshot changed during pack build");
            }
        }
        for (const CapturedShard& cs : captured_shards) {
            ShardFingerprint post_fp{};
            if (!stat_fd(cs.file->fd(), post_fp) || !shard_stats_match(cs.fp, post_fp)) {
                fail_pack(cs.file->path(), "source snapshot changed during pack build (open shard fd mutated)");
            }
            ShardFingerprint path_fp{};
            if (!stat_regular_file(cs.file->path(), path_fp) || !shard_stats_match(cs.fp, path_fp)) {
                fail_pack(cs.file->path(), "source snapshot changed during pack build (shard path replaced)");
            }
            if (cs.fp.has_full_sha256) {
                std::array<std::uint8_t, 32> post_sha{};
                if (!sha256_fd(cs.file->fd(), post_fp.file_size, post_sha) ||
                    post_sha != cs.fp.full_sha256) {
                    fail_pack(cs.file->path(), "source snapshot changed during pack build (shard content mutated)");
                }
            }
        }
    };
    WeightPackBuildStats stats = builder.write_impl(output_path, opts, pre_publish_check);
    if (any_large_shard_without_sha && has_max_ctime) {
        for (int attempt = 0; attempt < 10; ++attempt) {
            struct stat pst {};
            if (::stat(output_path.c_str(), &pst) == 0 &&
                timespec_strictly_less(max_csec, max_cnsec,
                                       static_cast<std::int64_t>(pst.st_ctim.tv_sec),
                                       static_cast<std::int64_t>(pst.st_ctim.tv_nsec))) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            struct timespec ts[2] = {{0, UTIME_OMIT}, {0, UTIME_NOW}};
            (void)::utimensat(AT_FDCWD, output_path.c_str(), ts, 0);
        }
    }
    return stats;
}

void set_next_pack_after_open_hook(std::function<void()> hook) {
    std::lock_guard<std::mutex> lk(g_pack_hook_mu);
    g_next_after_open_hook = std::move(hook);
}

void set_next_pack_before_copy_hook(std::function<void()> hook) {
    std::lock_guard<std::mutex> lk(g_pack_hook_mu);
    g_next_before_copy_hook = std::move(hook);
}

}  // namespace ingine

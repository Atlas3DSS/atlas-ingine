// 2 MiB huge-page host allocation & /proc/self/smaps observability (adapted from alesha-pro/llm.asm 02-baremetal-os).
// 100% CUDA-free; safe fallback to standard 4 KiB pages when THP/mmap is disabled or unavailable.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

#include <sys/mman.h>

namespace ingine {

inline constexpr std::size_t kHugePageAlignBytes = 2ull << 20;

inline bool hugepages_enabled() noexcept {
    const char* env = std::getenv("INGINE_DISABLE_HUGEPAGES");
    if (!env || env[0] == '\0') { return true; }
    if (std::strcmp(env, "0") == 0) { return true; }
    if (std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0) { return true; }
    return false;
}

struct HugePageMapping {
    void* ptr                = nullptr;
    std::size_t mapped_bytes = 0;
    bool hugepage_advised    = false;
    bool via_mmap            = false;
};

inline std::size_t align_up_hugepage(std::size_t bytes) noexcept {
    if (bytes == 0) { return kHugePageAlignBytes; }
    if (bytes > std::numeric_limits<std::size_t>::max() - (kHugePageAlignBytes - 1)) { return 0; }
    return (bytes + kHugePageAlignBytes - 1) & ~(kHugePageAlignBytes - 1);
}

// Allocates `align_up(bytes, 2 MiB)` (when via_mmap) aligned strictly on a 2 MiB virtual boundary
// and advises MADV_HUGEPAGE (unless allow_hugepage is false or INGINE_DISABLE_HUGEPAGES is set).
// Never throws; falls back to posix_memalign(4096) on mmap failure, or returns {nullptr} on OOM.
inline HugePageMapping alloc_hugepage_anon(std::size_t bytes, bool allow_hugepage = true) noexcept {
    HugePageMapping out;
    if (bytes == 0) { return out; }

    const std::size_t aligned_size = align_up_hugepage(bytes);
    if (aligned_size != 0 && aligned_size <= std::numeric_limits<std::size_t>::max() - kHugePageAlignBytes) {
        const std::size_t span = aligned_size + kHugePageAlignBytes;
        void* raw = ::mmap(nullptr, span, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (raw != MAP_FAILED) {
            const auto base    = reinterpret_cast<std::uintptr_t>(raw);
            const auto aligned = (base + kHugePageAlignBytes - 1) & ~(std::uintptr_t(kHugePageAlignBytes) - 1);
            const std::size_t head = aligned - base;
            const std::size_t tail = span - head - aligned_size;
            if (head) { ::munmap(raw, head); }
            if (tail) { ::munmap(reinterpret_cast<void*>(aligned + aligned_size), tail); }
            out.ptr          = reinterpret_cast<void*>(aligned);
            out.mapped_bytes = aligned_size;
            out.via_mmap     = true;
#ifdef MADV_HUGEPAGE
            if (allow_hugepage && hugepages_enabled()) {
                out.hugepage_advised = (::madvise(out.ptr, out.mapped_bytes, MADV_HUGEPAGE) == 0);
            }
#else
            (void)allow_hugepage;
#endif
            return out;
        }
    }

    void* fallback = nullptr;
    if (::posix_memalign(&fallback, 4096, bytes) == 0 && fallback != nullptr) {
        out.ptr          = fallback;
        out.mapped_bytes = bytes;
        out.via_mmap     = false;
#ifdef MADV_HUGEPAGE
        if (allow_hugepage && hugepages_enabled()) {
            out.hugepage_advised = (::madvise(out.ptr, out.mapped_bytes, MADV_HUGEPAGE) == 0);
        }
#endif
    }
    return out;
}

inline void free_hugepage_anon(HugePageMapping& m) noexcept {
    if (m.ptr != nullptr) {
        if (m.via_mmap) {
            ::munmap(m.ptr, m.mapped_bytes);
        } else {
            std::free(m.ptr);
        }
    }
    m = HugePageMapping{};
}

// Non-destructive write-fault touch at 2 MiB stride so Linux backs anonymous THP pages BEFORE
// cudaHostRegister() pins them, without altering any bytes already written (e.g. ExpertCache WARM RAM).
inline void prefault_hugepage_stride_nondestructive(void* ptr, std::size_t bytes) noexcept {
    if (!ptr || bytes == 0) { return; }
    volatile std::uint8_t* p = static_cast<volatile std::uint8_t*>(ptr);
    for (std::size_t off = 0; off < bytes; off += kHugePageAlignBytes) {
        const std::uint8_t v = p[off];
        p[off]               = v;
    }
}

struct SmapsHugePageInfo {
    std::uint64_t anon_huge_pages_bytes = 0;
    std::uint64_t rss_bytes             = 0;
    bool readable                       = false;
};

// Inspects /proc/self/smaps for VMAs overlapping [ptr, ptr + bytes) and sums AnonHugePages and Rss.
inline SmapsHugePageInfo query_range_smaps_hugepages(const void* ptr, std::size_t bytes) {
    SmapsHugePageInfo info;
    if (!ptr || bytes == 0) { return info; }
    std::ifstream in("/proc/self/smaps");
    if (!in) { return info; }
    info.readable = true;

    const auto target_lo = reinterpret_cast<std::uintptr_t>(ptr);
    const auto target_hi = target_lo + bytes;
    bool in_overlap      = false;
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t dash = line.find('-');
        const std::size_t sp   = line.find(' ');
        if (dash != std::string::npos && sp != std::string::npos && dash < sp &&
            std::isxdigit(static_cast<unsigned char>(line[0]))) {
            std::uintptr_t vma_lo = 0;
            std::uintptr_t vma_hi = 0;
            std::istringstream hdr(line.substr(0, sp));
            char c = 0;
            if ((hdr >> std::hex >> vma_lo >> c >> std::hex >> vma_hi) && c == '-') {
                in_overlap = (vma_lo < target_hi) && (vma_hi > target_lo);
            } else {
                in_overlap = false;
            }
            continue;
        }
        if (!in_overlap) { continue; }
        auto parse_kb_field = [&](const char* prefix, std::uint64_t& dst) {
            const std::size_t plen = std::strlen(prefix);
            if (line.compare(0, plen, prefix) == 0) {
                std::istringstream vs(line.substr(plen));
                std::uint64_t kb = 0;
                if (vs >> kb) { dst += kb * 1024ull; }
            }
        };
        parse_kb_field("AnonHugePages:", info.anon_huge_pages_bytes);
        parse_kb_field("Rss:", info.rss_bytes);
    }
    return info;
}

} // namespace ingine

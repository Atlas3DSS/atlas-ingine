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
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sys/mman.h>
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace ingine {

inline constexpr std::size_t kHugePageAlignBytes = 2ull << 20;

enum class NumaPolicy : int {
    Default    = 0,
    Preferred  = 1,
    Bind       = 2,
    Interleave = 3,
};

inline constexpr int kMpolPreferred  = 1;
inline constexpr int kMpolBind       = 2;
inline constexpr int kMpolInterleave = 3;
inline constexpr int kMpolMfMove     = 2;

inline bool hugepages_enabled() noexcept {
    const char* env = std::getenv("INGINE_DISABLE_HUGEPAGES");
    if (!env || env[0] == '\0') { return true; }
    if (std::strcmp(env, "0") == 0) { return true; }
    if (std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0) { return true; }
    return false;
}

inline bool numa_bind_enabled() noexcept {
    const char* env = std::getenv("INGINE_DISABLE_NUMA_BIND");
    if (!env || env[0] == '\0') { return true; }
    if (std::strcmp(env, "0") == 0) { return true; }
    if (std::strcmp(env, "false") == 0 || std::strcmp(env, "FALSE") == 0) { return true; }
    return false;
}

inline int current_numa_node() noexcept {
#if defined(__linux__) && defined(SYS_getcpu)
    unsigned cpu  = 0;
    unsigned node = 0;
    if (::syscall(SYS_getcpu, &cpu, &node, nullptr) == 0) {
        return static_cast<int>(node);
    }
#endif
    return -1;
}

// Binds virtual range [ptr, ptr + bytes) to `target_node` via Linux SYS_mbind (no libnuma dependency).
// Never throws; returns false silently on unsupported kernels or invalid inputs.
inline bool bind_memory_numa(void* ptr, std::size_t bytes, int target_node = -1,
                             NumaPolicy policy = NumaPolicy::Preferred,
                             bool move_existing = false) noexcept {
    if (!ptr || bytes == 0 || !numa_bind_enabled() || policy == NumaPolicy::Default) { return false; }
#if defined(__linux__) && defined(SYS_mbind)
    const int node = (target_node < 0) ? current_numa_node() : target_node;
    constexpr int kMaskWords = 16;
    constexpr int kBitsPerWord = static_cast<int>(sizeof(unsigned long) * 8);
    if (node < 0 || node >= kMaskWords * kBitsPerWord - 1) { return false; }

    const long ps = ::sysconf(_SC_PAGESIZE);
    const std::uintptr_t page = (ps > 0) ? static_cast<std::uintptr_t>(ps) : 4096ull;
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - addr) { return false; }
    const std::uintptr_t lo = addr & ~(page - 1);
    const std::uintptr_t raw_hi = addr + bytes;
    const std::uintptr_t hi = (raw_hi <= std::numeric_limits<std::uintptr_t>::max() - (page - 1))
                                  ? ((raw_hi + page - 1) & ~(page - 1))
                                  : (std::numeric_limits<std::uintptr_t>::max() & ~(page - 1));
    if (hi <= lo) { return false; }

    unsigned long mask[kMaskWords]{};
    mask[node / kBitsPerWord] = 1ul << (node % kBitsPerWord);
    // Linux mbind(2) reads (maxnode - 1) bits from nodemask; node 0 requires maxnode >= 2.
    const unsigned long maxnode = static_cast<unsigned long>(node) + 2ul;

    int mode = kMpolPreferred;
    switch (policy) {
        case NumaPolicy::Preferred: mode = kMpolPreferred; break;
        case NumaPolicy::Bind: mode = kMpolBind; break;
        case NumaPolicy::Interleave: mode = kMpolInterleave; break;
        case NumaPolicy::Default: return false;
    }
    const unsigned long flags = move_existing ? static_cast<unsigned long>(kMpolMfMove) : 0ul;
    const long rc = ::syscall(SYS_mbind, reinterpret_cast<void*>(lo), static_cast<unsigned long>(hi - lo),
                              mode, mask, maxnode, flags);
    return rc == 0;
#else
    (void)target_node;
    (void)policy;
    (void)move_existing;
    return false;
#endif
}

struct HugePageMapping {
    void* ptr                = nullptr;
    std::size_t mapped_bytes = 0;
    bool hugepage_advised    = false;
    bool via_mmap            = false;
    int numa_node            = -1;
    bool numa_bound          = false;
};

inline std::size_t align_up_hugepage(std::size_t bytes) noexcept {
    if (bytes == 0) { return kHugePageAlignBytes; }
    if (bytes > std::numeric_limits<std::size_t>::max() - (kHugePageAlignBytes - 1)) { return 0; }
    return (bytes + kHugePageAlignBytes - 1) & ~(kHugePageAlignBytes - 1);
}

// Allocates `align_up(bytes, 2 MiB)` (when via_mmap) aligned strictly on a 2 MiB virtual boundary,
// advises MADV_HUGEPAGE (unless allow_hugepage is false or INGINE_DISABLE_HUGEPAGES is set),
// and applies Preferred/Bind NUMA placement before faulting pages.
// Never throws; falls back to posix_memalign(4096) on mmap failure, or returns {nullptr} on OOM.
inline HugePageMapping alloc_hugepage_anon(std::size_t bytes, bool allow_hugepage = true,
                                           int numa_node = -1,
                                           NumaPolicy numa_policy = NumaPolicy::Preferred) noexcept {
    HugePageMapping out;
    if (bytes == 0) { return out; }

    auto apply_numa = [&](HugePageMapping& m) noexcept {
        if (m.ptr != nullptr && numa_bind_enabled() && numa_policy != NumaPolicy::Default) {
            const int target = (numa_node >= 0) ? numa_node : current_numa_node();
            if (target >= 0) {
                m.numa_bound = bind_memory_numa(m.ptr, m.mapped_bytes, target, numa_policy, false);
                if (m.numa_bound) { m.numa_node = target; }
            }
        }
    };

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
            apply_numa(out);
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
        apply_numa(out);
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

struct NumaMapsInfo {
    bool readable = false;
    std::map<int, std::uint64_t> node_bytes;

    [[nodiscard]] std::uint64_t total_bytes() const noexcept {
        std::uint64_t sum = 0;
        for (const auto& [node, bytes] : node_bytes) {
            (void)node;
            sum += bytes;
        }
        return sum;
    }
};

// Inspects /proc/self/maps + /proc/self/numa_maps for VMAs overlapping [ptr, ptr + bytes)
// and returns resident bytes broken down by NUMA node.
inline NumaMapsInfo query_range_numa_nodes(const void* ptr, std::size_t bytes) {
    NumaMapsInfo info;
    if (!ptr || bytes == 0) { return info; }

    const auto target_lo = reinterpret_cast<std::uintptr_t>(ptr);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - target_lo) { return info; }
    const auto target_hi = target_lo + bytes;

    std::vector<std::uintptr_t> vma_starts;
    {
        std::ifstream maps("/proc/self/maps");
        if (!maps) { return info; }
        std::string line;
        while (std::getline(maps, line)) {
            const std::size_t dash = line.find('-');
            const std::size_t sp   = line.find(' ');
            if (dash == std::string::npos || sp == std::string::npos || dash >= sp ||
                !std::isxdigit(static_cast<unsigned char>(line[0]))) {
                continue;
            }
            std::uintptr_t vma_lo = 0;
            std::uintptr_t vma_hi = 0;
            std::istringstream hdr(line.substr(0, sp));
            char c = 0;
            if ((hdr >> std::hex >> vma_lo >> c >> std::hex >> vma_hi) && c == '-') {
                if ((vma_lo < target_hi) && (vma_hi > target_lo)) {
                    vma_starts.push_back(vma_lo);
                }
            }
        }
    }
    std::sort(vma_starts.begin(), vma_starts.end());

    std::ifstream numa("/proc/self/numa_maps");
    if (!numa) { return info; }
    info.readable = true;
    if (vma_starts.empty()) { return info; }

    std::string line;
    while (std::getline(numa, line)) {
        std::istringstream iss(line);
        std::uintptr_t vma_lo = 0;
        if (!(iss >> std::hex >> vma_lo)) { continue; }
        if (!std::binary_search(vma_starts.begin(), vma_starts.end(), vma_lo)) { continue; }

        std::uint64_t page_bytes = 4096ull;
        std::vector<std::pair<int, std::uint64_t>> node_pages;
        std::string tok;
        while (iss >> tok) {
            if (tok.rfind("kernelpagesize_kB=", 0) == 0) {
                char* end = nullptr;
                const unsigned long long kb = std::strtoull(tok.c_str() + 18, &end, 10);
                if (end != tok.c_str() + 18 && *end == '\0' && kb > 0) {
                    page_bytes = static_cast<std::uint64_t>(kb) * 1024ull;
                }
            } else if (tok.size() >= 4 && tok[0] == 'N' &&
                       std::isdigit(static_cast<unsigned char>(tok[1]))) {
                const std::size_t eq = tok.find('=');
                if (eq != std::string::npos && eq > 1 && eq + 1 < tok.size()) {
                    bool all_digits = true;
                    for (std::size_t k = 1; k < eq; ++k) {
                        if (!std::isdigit(static_cast<unsigned char>(tok[k]))) {
                            all_digits = false;
                            break;
                        }
                    }
                    if (all_digits) {
                        char* end_n = nullptr;
                        char* end_p = nullptr;
                        const long node = std::strtol(tok.c_str() + 1, &end_n, 10);
                        const unsigned long long pages = std::strtoull(tok.c_str() + eq + 1, &end_p, 10);
                        if (end_n == tok.c_str() + eq && end_p != tok.c_str() + eq + 1 && *end_p == '\0' &&
                            node >= 0 && pages > 0) {
                            node_pages.emplace_back(static_cast<int>(node), static_cast<std::uint64_t>(pages));
                        }
                    }
                }
            }
        }
        for (const auto& [node, pages] : node_pages) {
            info.node_bytes[node] += pages * page_bytes;
        }
    }
    return info;
}

} // namespace ingine

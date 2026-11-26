// ingine-pack: build, inspect, verify, and benchmark 2 MiB-aligned INGPACK1 flat mmap weight packs.
// 100% CUDA-free (links only ingine_core).
#include "core/safetensors.h"
#include "core/sha256.h"
#include "core/weight_pack.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void print_usage() {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  ingine-pack build <src_safetensors_or_dir> <out.ingpack> [--align-mib N] "
                 "[--threshold-mib N] [--no-sha256]\n"
                 "  ingine-pack inspect <file.ingpack> [--verify]\n"
                 "  ingine-pack bench <src_safetensors_or_dir> <pack.ingpack> [--iters N]\n");
}

long parse_pos_long(const char* s, const char* flag) {
    char* end = nullptr;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0) {
        throw std::invalid_argument(std::string("invalid value for ") + flag + ": " + s);
    }
    return v;
}

int cmd_build(int argc, char** argv) {
    if (argc < 4) {
        print_usage();
        return 2;
    }
    const std::filesystem::path src = argv[2];
    const std::filesystem::path out = argv[3];
    ingine::WeightPackBuildOptions opts;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--align-mib") == 0 && i + 1 < argc) {
            opts.tensor_alignment = static_cast<std::size_t>(parse_pos_long(argv[++i], "--align-mib")) << 20;
        } else if (std::strcmp(argv[i], "--threshold-mib") == 0 && i + 1 < argc) {
            opts.large_tensor_threshold =
                static_cast<std::size_t>(parse_pos_long(argv[++i], "--threshold-mib")) << 20;
        } else if (std::strcmp(argv[i], "--no-sha256") == 0) {
            opts.compute_content_sha256 = false;
        } else {
            throw std::invalid_argument(std::string("unknown build option: ") + argv[i]);
        }
    }
    const ingine::WeightPackBuildStats st =
        ingine::WeightPackBuilder::pack_from_safetensors(src, out, opts);
    std::printf("ingine-pack build: %s -> %s\n", src.c_str(), out.c_str());
    std::printf("  tensors=%llu (small=%llu, large=%llu)\n",
                static_cast<unsigned long long>(st.num_tensors),
                static_cast<unsigned long long>(st.small_tensors),
                static_cast<unsigned long long>(st.large_tensors));
    std::printf("  data_offset=%llu file_size=%llu payload=%llu (%.2f MiB) in %.3f s (%.2f GiB/s)\n",
                static_cast<unsigned long long>(st.data_offset),
                static_cast<unsigned long long>(st.file_size),
                static_cast<unsigned long long>(st.payload_bytes),
                double(st.payload_bytes) / 1048576.0,
                st.seconds,
                st.seconds > 0.0 ? (double(st.payload_bytes) / double(1ull << 30)) / st.seconds : 0.0);
    if (opts.compute_content_sha256) {
        std::printf("  content_sha256=%s\n", ingine::sha256_hex(st.content_sha256).c_str());
    }
    return 0;
}

int cmd_inspect(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 2;
    }
    const std::filesystem::path file = argv[2];
    bool verify = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--verify") == 0) {
            verify = true;
        } else {
            throw std::invalid_argument(std::string("unknown inspect option: ") + argv[i]);
        }
    }
    const auto t0 = Clock::now();
    ingine::WeightPackOpenOptions o;
    o.verify_header_sha  = true;
    o.verify_content_sha = false;
    ingine::WeightPack pack(file, o);
    const double open_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    std::uint64_t small_cnt = 0, large_cnt = 0, large_aligned = 0, payload = 0;
    for (const ingine::WeightPackEntry& e : pack.entries()) {
        payload += e.data_bytes;
        if (e.data_bytes >= pack.header().large_tensor_threshold) {
            ++large_cnt;
            if ((reinterpret_cast<std::uintptr_t>(pack.tensor_data(e)) % pack.header().tensor_alignment) == 0) {
                ++large_aligned;
            }
        } else {
            ++small_cnt;
        }
    }
    std::printf("ingine-pack inspect: %s (open=%.3f ms)\n", file.c_str(), open_ms);
    std::printf("  file_size=%llu data_offset=%llu aligned_2m_base=%s hugepage_advised=%s\n",
                static_cast<unsigned long long>(pack.file_size()),
                static_cast<unsigned long long>(pack.data_offset()),
                pack.is_2m_aligned_base() ? "yes" : "no",
                pack.hugepage_advised() ? "yes" : "no");
    std::printf("  tensors=%u (small=%llu, large=%llu, large_aligned=%llu/%llu) metadata=%u payload=%.2f MiB\n",
                pack.num_tensors(),
                static_cast<unsigned long long>(small_cnt),
                static_cast<unsigned long long>(large_cnt),
                static_cast<unsigned long long>(large_aligned),
                static_cast<unsigned long long>(large_cnt),
                pack.header().num_metadata,
                double(payload) / 1048576.0);
    if (verify) {
        const auto tv0 = Clock::now();
        const bool ok = pack.verify_content_sha256();
        const double vsec = std::chrono::duration<double>(Clock::now() - tv0).count();
        std::array<std::uint8_t, 32> csha{};
        std::memcpy(csha.data(), pack.header().content_sha256, 32);
        std::printf("  verify_content_sha256=%s (%.3f s) sha256=%s\n",
                    ok ? "PASS" : "FAIL", vsec, ingine::sha256_hex(csha).c_str());
        if (!ok) { return 1; }
    }
    return 0;
}

int cmd_bench(int argc, char** argv) {
    if (argc < 4) {
        print_usage();
        return 2;
    }
    const std::filesystem::path st_src   = argv[2];
    const std::filesystem::path pack_src = argv[3];
    int iters = 5;
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = static_cast<int>(parse_pos_long(argv[++i], "--iters"));
        } else {
            throw std::invalid_argument(std::string("unknown bench option: ") + argv[i]);
        }
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    std::vector<double> st_open_ms, pk_open_ms;
    std::uint64_t total_bytes = 0;
    for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        ingine::SafetensorsModel sm(st_src);
        const auto t1 = Clock::now();
        st_open_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (i == 0) {
            for (const std::string& n : sm.names()) { total_bytes += sm.at(n)->bytes; }
        }
    }
    for (int i = 0; i < iters; ++i) {
        const auto t0 = Clock::now();
        ingine::WeightPack wp(pack_src);
        const auto t1 = Clock::now();
        pk_open_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::vector<std::byte> sink_buf(std::size_t(32) << 20);
    auto read_all_st = [&]() {
        ingine::SafetensorsModel sm(st_src);
        for (const std::string& n : sm.names()) {
            const ingine::SafetensorsRef r = sm.at(n);
            std::uint64_t off = 0;
            while (off < r->bytes) {
                const std::size_t take = static_cast<std::size_t>(std::min<std::uint64_t>(r->bytes - off, sink_buf.size()));
                r.file->read_range(r->offset + off, take, sink_buf.data());
                off += take;
            }
        }
    };
    auto read_all_pk = [&]() {
        ingine::WeightPack wp(pack_src);
        for (const ingine::WeightPackEntry& e : wp.entries()) {
            std::uint64_t off = 0;
            while (off < e.data_bytes) {
                const std::size_t take = static_cast<std::size_t>(std::min<std::uint64_t>(e.data_bytes - off, sink_buf.size()));
                std::memcpy(sink_buf.data(), wp.tensor_data(e) + off, take);
                off += take;
            }
        }
    };
    read_all_st();
    read_all_pk();
    std::vector<double> st_read_ms, pk_read_ms;
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        read_all_st();
        auto t1 = Clock::now();
        st_read_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        t0 = Clock::now();
        read_all_pk();
        t1 = Clock::now();
        pk_read_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    const double open_st = median(st_open_ms);
    const double open_pk = median(pk_open_ms);
    const double rd_st   = median(st_read_ms);
    const double rd_pk   = median(pk_read_ms);
    const double gib     = double(total_bytes) / double(1ull << 30);
    std::printf("=== ingine-pack bench (%.2f MiB, %d iters) ===\n",
                double(total_bytes) / 1048576.0, iters);
    std::printf("  open/parse : safetensors=%.3f ms | ingpack=%.3f ms (%.2fx)\n",
                open_st, open_pk, open_pk > 0.0 ? open_st / open_pk : 0.0);
    std::printf("  read all   : safetensors=%.3f ms (%.2f GiB/s) | ingpack=%.3f ms (%.2f GiB/s, %.2fx)\n",
                rd_st, rd_st > 0.0 ? gib / (rd_st * 1e-3) : 0.0,
                rd_pk, rd_pk > 0.0 ? gib / (rd_pk * 1e-3) : 0.0,
                rd_pk > 0.0 ? rd_st / rd_pk : 0.0);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }
    try {
        if (std::strcmp(argv[1], "build") == 0) { return cmd_build(argc, argv); }
        if (std::strcmp(argv[1], "inspect") == 0) { return cmd_inspect(argc, argv); }
        if (std::strcmp(argv[1], "bench") == 0) { return cmd_bench(argc, argv); }
        print_usage();
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ingine-pack error: %s\n", e.what());
        return 1;
    }
}

// Bare-metal Phase B: 2 MiB-aligned INGPACK1 weight pack verification, SafetensorsModel integration,
// adversarial corruption checks, DeviceUploader + UploadChunkObserver parity, and optional --bench.
// 100% CUDA-free when INGINE_BACKEND_CUDA == 0 (guarded HAL/GPU upload check when CUDA is active).

#include "common/test_util.h"
#include "core/arena.h"
#include "core/fixture.h"
#include "core/hal.h"
#include "core/hugepage.h"
#include "core/safetensors.h"
#include "core/sha256.h"
#include "core/weight_pack.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using ingine::DeviceBuffer;
using ingine::DeviceUploader;
using ingine::DType;
using ingine::SafetensorsFile;
using ingine::SafetensorsModel;
using ingine::UploadRequest;
using ingine::WeightPack;
using ingine::WeightPackBuilder;
using ingine::WeightPackBuildOptions;
using ingine::WeightPackBuildStats;
using ingine::WeightPackEntry;
using ingine::WeightPackHeader;
using ingine::test::FixtureTensor;
using ingine::test::pattern_bytes;
using ingine::test::write_safetensors;
namespace hal = ingine::hal;

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        const char* prev = ::getenv(name_);
        if (prev != nullptr) {
            had_prev_ = true;
            prev_     = prev;
        }
        if (value != nullptr) {
            ::setenv(name_, value, 1);
        } else {
            ::unsetenv(name_);
        }
    }
    ~ScopedEnvVar() {
        if (had_prev_) {
            ::setenv(name_, prev_.c_str(), 1);
        } else {
            ::unsetenv(name_);
        }
    }
    ScopedEnvVar(const ScopedEnvVar&)            = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

private:
    const char* name_;
    bool had_prev_ = false;
    std::string prev_;
};

bool throws_runtime(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) { return true; }
    return false;
}

bool is_2m_aligned(const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) % ingine::kHugePageAlignBytes) == 0;
}

void recompute_header_sha(std::vector<std::uint8_t>& buf) {
    INGINE_CHECK(buf.size() >= sizeof(WeightPackHeader));
    WeightPackHeader hdr{};
    std::memcpy(&hdr, buf.data(), sizeof(hdr));
    const std::uint64_t hend = hdr.string_table_offset + hdr.string_table_bytes;
    INGINE_CHECK(hend <= buf.size());
    std::memset(buf.data() + offsetof(WeightPackHeader, header_sha256), 0, 32);
    const auto sha = ingine::sha256_bytes(buf.data(), static_cast<std::size_t>(hend));
    std::memcpy(buf.data() + offsetof(WeightPackHeader, header_sha256), sha.data(), 32);
}

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    INGINE_CHECK(static_cast<bool>(in));
    return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void write_file_bytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    INGINE_CHECK(static_cast<bool>(out));
    if (!data.empty()) {
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    INGINE_CHECK(static_cast<bool>(out));
}

std::vector<FixtureTensor> mixed_fixture_tensors() {
    return {
        {"model.norm.scale", "F32", {}, pattern_bytes(4, 10)},
        {"model.layers.0.scales", "U8", {5}, pattern_bytes(5, 11)},
        {"model.empty", "F16", {0, 3}, {}},
        {"model.layers.0.small_bf16", "BF16", {3, 4}, pattern_bytes(24, 12)},
        {"model.layers.0.r4", "F32", {2, 2, 3, 4}, pattern_bytes(48 * 4, 13)},
        {"model.layers.0.large_bf16", "BF16", {1024, 1536}, pattern_bytes(1024 * 1536 * 2, 14)},
        {"model.layers.1.exact_2m", "U8", {2 * 1024 * 1024}, pattern_bytes(2 * 1024 * 1024, 15)},
        {"model.layers.1.large_f32", "F32", {768, 1024}, pattern_bytes(768 * 1024 * 4, 16)},
    };
}

void test_builder_and_roundtrip(const std::filesystem::path& dir) {
    std::printf("[1/4] Testing WeightPackBuilder, 2 MiB alignment, zero-copy views, and round-trip...\n");
    const auto tensors = mixed_fixture_tensors();

    // Build directly with WeightPackBuilder using small I/O chunk size to test streaming SHA-256 invariance.
    const auto direct_path = dir / "direct.ingpack";
    {
        WeightPackBuilder b;
        b.add_metadata("format", "pt");
        b.add_metadata("note", "phaseB direct fixture");
        for (const FixtureTensor& t : tensors) {
            b.add_tensor(t.name, t.dtype, t.shape, t.data.size(),
                         [&](const std::function<void(const void*, std::size_t)>& emit) {
                             std::size_t off = 0;
                             while (off < t.data.size()) {
                                 const std::size_t chunk = std::min<std::size_t>(t.data.size() - off, 65537u);
                                 emit(t.data.data() + off, chunk);
                                 off += chunk;
                             }
                         });
        }
        WeightPackBuildOptions opts;
        opts.io_chunk_bytes = 1u << 20;
        const WeightPackBuildStats st = b.write(direct_path, opts);
        INGINE_CHECK(st.num_tensors == tensors.size());
        INGINE_CHECK(st.small_tensors == 5);
        INGINE_CHECK(st.large_tensors == 3);
        INGINE_CHECK((st.data_offset % ingine::kHugePageAlignBytes) == 0);
    }

    INGINE_CHECK(ingine::is_weight_pack_file(direct_path));

    WeightPack pack(direct_path);
    INGINE_CHECK(pack.num_tensors() == tensors.size());
    INGINE_CHECK((pack.data_offset() % ingine::kHugePageAlignBytes) == 0);
    INGINE_CHECK(pack.is_2m_aligned_base());
    INGINE_CHECK(is_2m_aligned(pack.mapped_base()));
    INGINE_CHECK(pack.verify_content_sha256());

    const auto meta = pack.metadata();
    INGINE_CHECK(meta.size() == 2);
    INGINE_CHECK(meta.at("format") == "pt");
    INGINE_CHECK(meta.at("note") == "phaseB direct fixture");

    for (const FixtureTensor& t : tensors) {
        const WeightPackEntry* e = pack.find(t.name);
        INGINE_CHECK(e != nullptr);
        INGINE_CHECK(pack.entry_name(*e) == t.name);
        INGINE_CHECK(pack.entry_dtype(*e) == t.dtype);
        INGINE_CHECK(e->data_bytes == t.data.size());
        if (t.data.size() >= ingine::kWeightPackDefaultThresholdBytes) {
            INGINE_CHECK((e->data_offset % ingine::kHugePageAlignBytes) == 0);
            INGINE_CHECK(is_2m_aligned(pack.tensor_data(*e)));
        }
        if (!t.data.empty()) {
            INGINE_CHECK(std::memcmp(pack.tensor_data(*e), t.data.data(), t.data.size()) == 0);
        }
    }
    INGINE_CHECK(pack.find("does.not.exist") == nullptr);
    INGINE_CHECK(throws_runtime([&] { (void)pack.at("does.not.exist"); }));

    // Zero-copy Tensor view check (ne[0] innermost).
    const ingine::Tensor tv = pack.tensor_view("model.layers.0.large_bf16");
    INGINE_CHECK(tv.data == pack.tensor_data("model.layers.0.large_bf16"));
    INGINE_CHECK(tv.dtype == DType::BF16);
    INGINE_CHECK(tv.ne[0] == 1536 && tv.ne[1] == 1024 && tv.ne[2] == 1 && tv.ne[3] == 1);
    INGINE_CHECK(tv.nb[0] == 2 && tv.nb[1] == 3072);
    INGINE_CHECK(throws_runtime([&] { (void)pack.tensor_view("model.empty"); }));

    const ingine::Tensor rv = pack.tensor_view("model.layers.0.r4");
    INGINE_CHECK(rv.dtype == DType::FP32);
    INGINE_CHECK(rv.ne[0] == 4 && rv.ne[1] == 3 && rv.ne[2] == 2 && rv.ne[3] == 2);

    pack.prefault(true);
    INGINE_CHECK(pack.verify_content_sha256());
}

void test_safetensors_and_uploader_integration(const std::filesystem::path& dir) {
    std::printf("[2/4] Testing SafetensorsModel integration, toggle, and DeviceUploader SHA-256 parity...\n");
    const auto tensors = mixed_fixture_tensors();
    const auto model_dir = dir / "multi_shard";
    std::filesystem::create_directories(model_dir);

    std::vector<FixtureTensor> shard_a(tensors.begin(), tensors.begin() + 5);
    std::vector<FixtureTensor> shard_b(tensors.begin() + 5, tensors.end());
    shard_b.push_back({"unindexed.ignored", "U8", {4}, pattern_bytes(4, 99)});
    write_safetensors(model_dir / "model-00001-of-00002.safetensors", shard_a);
    write_safetensors(model_dir / "model-00002-of-00002.safetensors", shard_b);

    nlohmann::json idx = {{"metadata", {{"total_size", 0}}}, {"weight_map", nlohmann::json::object()}};
    for (const auto& t : shard_a) { idx["weight_map"][t.name] = "model-00001-of-00002.safetensors"; }
    for (std::size_t i = 5; i < tensors.size(); ++i) {
        idx["weight_map"][tensors[i].name] = "model-00002-of-00002.safetensors";
    }
    std::ofstream(model_dir / "model.safetensors.index.json") << idx.dump(2);

    const auto pack_path = model_dir / "model.ingpack";
    const WeightPackBuildStats st = WeightPackBuilder::pack_from_safetensors(model_dir, pack_path);
    INGINE_CHECK(st.num_tensors == tensors.size());

    // Default: SafetensorsModel prefers model.ingpack cleanly.
    SafetensorsModel pack_model(model_dir);
    INGINE_CHECK(pack_model.is_weight_pack());
    INGINE_CHECK(pack_model.files().size() == 1);
    INGINE_CHECK(pack_model.size() == tensors.size());
    INGINE_CHECK(!pack_model.contains("unindexed.ignored"));
    INGINE_CHECK(pack_model.files()[0]->mapped_data() != nullptr);
    INGINE_CHECK(is_2m_aligned(
        pack_model.files()[0]->tensor_mapped_ptr(*pack_model.at("model.layers.0.large_bf16").tensor)));

    // Sort invariant in SafetensorsFile::tensors()
    const auto& fl = *pack_model.files()[0];
    for (std::size_t i = 1; i < fl.tensors().size(); ++i) {
        INGINE_CHECK(fl.tensors()[i - 1].offset <= fl.tensors()[i].offset);
    }

    // INGINE_DISABLE_WEIGHT_PACK=1 falls back to model.safetensors.index.json.
    {
        ScopedEnvVar off("INGINE_DISABLE_WEIGHT_PACK", "1");
        INGINE_CHECK(!ingine::weight_pack_enabled());
        SafetensorsModel st_model(model_dir);
        INGINE_CHECK(!st_model.is_weight_pack());
        INGINE_CHECK(st_model.files().size() == 2);
        INGINE_CHECK(st_model.size() == tensors.size());

        for (const FixtureTensor& t : tensors) {
            const auto got_pk = pack_model.read(t.name);
            const auto got_st = st_model.read(t.name);
            INGINE_CHECK(got_pk == got_st);
            INGINE_CHECK(got_pk.size() == t.data.size());
            if (!t.data.empty()) {
                INGINE_CHECK(std::memcmp(got_pk.data(), t.data.data(), t.data.size()) == 0);
            }
        }

        // DeviceUploader + UploadChunkObserver SHA-256 parity across .ingpack vs .safetensors.
        bool can_run_hal = (hal::active_backend() == hal::BackendKind::Cpu);
        if (!can_run_hal) {
            try {
                (void)hal::query_device_caps(0);
                can_run_hal = true;
            } catch (...) { can_run_hal = false; }
        }
        if (can_run_hal) {
            hal::Stream stream(0);
            auto upload_and_hash = [&](const SafetensorsModel& m) {
                std::size_t total = 0;
                for (const FixtureTensor& t : tensors) { total += t.data.size(); }
                DeviceBuffer dev(total);
                dev.fill(0);

                std::vector<UploadRequest> reqs;
                std::size_t cursor = 0;
                for (const FixtureTensor& t : tensors) {
                    reqs.push_back(ingine::upload_request(m.at(t.name), static_cast<std::byte*>(dev.p) + cursor));
                    cursor += t.data.size();
                }

                ingine::UploadOptions up_opts;
                up_opts.chunk_bytes     = 1u << 20;
                up_opts.staging_buffers = 3;
                up_opts.reader_threads  = 2;
                DeviceUploader uploader(up_opts);

                std::map<std::pair<std::size_t, std::size_t>, std::array<std::uint8_t, 32>> digests;
                std::mutex mu;
                const auto ustats = uploader.upload(
                    reqs, stream.handle(),
                    [&](std::size_t ri, std::size_t ci, const void* data, std::size_t bytes) {
                        const auto sha = ingine::sha256_bytes(data, bytes);
                        std::lock_guard<std::mutex> lk(mu);
                        digests.emplace(std::make_pair(ri, ci), sha);
                    });
                INGINE_CHECK(ustats.bytes == total);

                std::vector<std::uint8_t> back(total);
                dev.copy_to_host(back.data(), back.size());
                return std::make_pair(std::move(back), std::move(digests));
            };

            auto [bytes_pk, sha_pk] = upload_and_hash(pack_model);
            auto [bytes_st, sha_st] = upload_and_hash(st_model);
            INGINE_CHECK(bytes_pk == bytes_st);
            INGINE_CHECK(sha_pk == sha_st);
        }
    }
}

void test_adversarial_corruption(const std::filesystem::path& dir) {
    std::printf("[3/4] Testing adversarial WeightPack header/body corruption rejection...\n");
    const auto base_pack = dir / "clean.ingpack";
    {
        WeightPackBuilder b;
        b.add_metadata("k", "v");
        const auto a = pattern_bytes(64, 1);
        const auto c = pattern_bytes(2 * 1024 * 1024 + 128, 2);
        b.add_tensor("alpha", "U8", {64}, a.size(),
                     [&](const auto& emit) { emit(a.data(), a.size()); });
        b.add_tensor("beta", "U8", {static_cast<std::int64_t>(c.size())}, c.size(),
                     [&](const auto& emit) { emit(c.data(), c.size()); });
        (void)b.write(base_pack);
    }

    const std::vector<std::uint8_t> clean = read_file_bytes(base_pack);
    const auto bad_path = dir / "bad.ingpack";

    // 1. Bad magic
    {
        auto mut = clean;
        mut[0]   = 'X';
        write_file_bytes(bad_path, mut);
        INGINE_CHECK(throws_runtime([&] { WeightPack p(bad_path); }));
    }
    // 2. Bad version
    {
        auto mut = clean;
        WeightPackHeader hdr{};
        std::memcpy(&hdr, mut.data(), sizeof(hdr));
        hdr.version = 2;
        std::memcpy(mut.data(), &hdr, sizeof(hdr));
        recompute_header_sha(mut);
        write_file_bytes(bad_path, mut);
        INGINE_CHECK(throws_runtime([&] { WeightPack p(bad_path); }));
    }
    // 3. Truncated file (<256 bytes and minus 64 bytes)
    {
        std::vector<std::uint8_t> tiny(clean.begin(), clean.begin() + 128);
        write_file_bytes(bad_path, tiny);
        INGINE_CHECK(throws_runtime([&] { WeightPack p(bad_path); }));

        std::vector<std::uint8_t> trunc(clean.begin(), clean.end() - 64);
        write_file_bytes(bad_path, trunc);
        INGINE_CHECK(throws_runtime([&] { WeightPack p(bad_path); }));
    }
    // 4. Appended trailing bytes
    {
        auto mut = clean;
        mut.insert(mut.end(), 16, 0xAA);
        write_file_bytes(bad_path, mut);
        INGINE_CHECK(throws_runtime([&] { WeightPack p(bad_path); }));
    }
    // 5. Unaligned data_offset (with header SHA recomputed)
    {
        auto mut = clean;
        WeightPackHeader hdr{};
        std::memcpy(&hdr, mut.data(), sizeof(hdr));
        hdr.data_offset += 4096;
        hdr.data_size = hdr.file_size - hdr.data_offset;
        std::memcpy(mut.data(), &hdr, sizeof(hdr));
        recompute_header_sha(mut);
        write_file_bytes(bad_path, mut);
        INGINE_CHECK(throws_runtime([&] { WeightPack p(bad_path); }));
    }
    // 6. Corrupted header byte without recomputing header_sha256
    {
        auto mut = clean;
        mut[12] ^= 0x01u;
        write_file_bytes(bad_path, mut);
        INGINE_CHECK(throws_runtime([&] { WeightPack p(bad_path); }));
    }
    // 7. Corrupted payload byte -> verify_content_sha256() == false and open with verify_content_sha throws
    {
        auto mut = clean;
        mut.back() ^= 0x55u;
        write_file_bytes(bad_path, mut);
        WeightPack p(bad_path);
        INGINE_CHECK(!p.verify_content_sha256());

        ingine::WeightPackOpenOptions o;
        o.verify_content_sha = true;
        INGINE_CHECK(throws_runtime([&] { WeightPack strict(bad_path, o); }));
    }
    // 8. Unsorted / duplicate directory entries (with header SHA recomputed)
    {
        auto mut = clean;
        WeightPackHeader hdr{};
        std::memcpy(&hdr, mut.data(), sizeof(hdr));
        WeightPackEntry e0{}, e1{};
        std::memcpy(&e0, mut.data() + hdr.dir_offset, sizeof(e0));
        std::memcpy(&e1, mut.data() + hdr.dir_offset + sizeof(WeightPackEntry), sizeof(e1));
        std::memcpy(mut.data() + hdr.dir_offset, &e1, sizeof(e1));
        std::memcpy(mut.data() + hdr.dir_offset + sizeof(WeightPackEntry), &e0, sizeof(e0));
        recompute_header_sha(mut);
        write_file_bytes(bad_path, mut);
        INGINE_CHECK(throws_runtime([&] { WeightPack p(bad_path); }));
    }
    // 9. OOB tensor offset (with header SHA recomputed)
    {
        auto mut = clean;
        WeightPackHeader hdr{};
        std::memcpy(&hdr, mut.data(), sizeof(hdr));
        WeightPackEntry e1{};
        std::memcpy(&e1, mut.data() + hdr.dir_offset + sizeof(WeightPackEntry), sizeof(e1));
        e1.data_offset = hdr.file_size - 8;
        std::memcpy(mut.data() + hdr.dir_offset + sizeof(WeightPackEntry), &e1, sizeof(e1));
        recompute_header_sha(mut);
        write_file_bytes(bad_path, mut);
        INGINE_CHECK(throws_runtime([&] { WeightPack p(bad_path); }));
    }
    // 10. shape * dtype mismatch (with header SHA recomputed)
    {
        auto mut = clean;
        WeightPackHeader hdr{};
        std::memcpy(&hdr, mut.data(), sizeof(hdr));
        WeightPackEntry e0{};
        std::memcpy(&e0, mut.data() + hdr.dir_offset, sizeof(e0));
        e0.shape[0] += 1;
        std::memcpy(mut.data() + hdr.dir_offset, &e0, sizeof(e0));
        recompute_header_sha(mut);
        write_file_bytes(bad_path, mut);
        INGINE_CHECK(throws_runtime([&] { WeightPack p(bad_path); }));
    }
}

void run_weight_pack_bench(const std::filesystem::path& dir) {
    std::printf("[4/4] Running 128 MiB safetensors vs INGPACK1 open & read microbenchmark...\n");
    const auto bench_dir = dir / "bench_128m";
    std::filesystem::create_directories(bench_dir);

    std::vector<FixtureTensor> tensors;
    const auto small_pat = pattern_bytes(16 * 1024, 21);
    const auto large_pat = pattern_bytes(8 * 1024 * 1024, 22);
    for (int i = 0; i < 32; ++i) {
        tensors.push_back({"layer.small." + std::to_string(i), "BF16", {128, 64}, small_pat});
    }
    for (int i = 0; i < 16; ++i) {
        tensors.push_back({"layer.large." + std::to_string(i), "BF16", {2048, 2048}, large_pat});
    }
    const auto st_path   = bench_dir / "model.safetensors";
    const auto pack_path = bench_dir / "model.ingpack";
    write_safetensors(st_path, tensors, {{"format", "pt"}});
    const WeightPackBuildStats st = WeightPackBuilder::pack_from_safetensors(st_path, pack_path);

    auto median_ms = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    constexpr int kOpenIters = 25;
    std::vector<double> st_open, pk_open;
    for (int i = 0; i < kOpenIters; ++i) {
        auto t0 = Clock::now();
        SafetensorsFile sf(st_path);
        auto t1 = Clock::now();
        st_open.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        t0 = Clock::now();
        WeightPack wp(pack_path);
        t1 = Clock::now();
        pk_open.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    std::vector<std::byte> scratch(8 * 1024 * 1024);
    auto sweep_st = [&]() {
        SafetensorsFile sf(st_path);
        for (const auto& t : sf.tensors()) { sf.read_range(t.offset, t.bytes, scratch.data()); }
    };
    auto sweep_pk = [&]() {
        WeightPack wp(pack_path);
        for (const auto& e : wp.entries()) {
            std::memcpy(scratch.data(), wp.tensor_data(e), static_cast<std::size_t>(e.data_bytes));
        }
    };
    sweep_st();
    sweep_pk();
    std::vector<double> st_rd, pk_rd;
    for (int i = 0; i < 5; ++i) {
        auto t0 = Clock::now();
        sweep_st();
        auto t1 = Clock::now();
        st_rd.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        t0 = Clock::now();
        sweep_pk();
        t1 = Clock::now();
        pk_rd.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    const double o_st = median_ms(st_open);
    const double o_pk = median_ms(pk_open);
    const double r_st = median_ms(st_rd);
    const double r_pk = median_ms(pk_rd);
    const double gib  = double(st.payload_bytes) / double(1ull << 30);
    std::printf("  128 MiB fixture (48 tensors): payload=%.1f MiB, aligned_2m_base=yes\n",
                double(st.payload_bytes) / 1048576.0);
    std::printf("  open/parse : safetensors=%.3f ms | ingpack=%.3f ms (%.2fx)\n",
                o_st, o_pk, o_pk > 0.0 ? o_st / o_pk : 0.0);
    std::printf("  sweep read : safetensors=%.3f ms (%.2f GiB/s) | ingpack=%.3f ms (%.2f GiB/s, %.2fx)\n",
                r_st, gib / (r_st * 1e-3), r_pk, gib / (r_pk * 1e-3), r_pk > 0.0 ? r_st / r_pk : 0.0);
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0) { bench = true; }
    }
    ingine::test::TempDir tmp;
    test_builder_and_roundtrip(tmp.path);
    test_safetensors_and_uploader_integration(tmp.path);
    test_adversarial_corruption(tmp.path);
    if (bench) { run_weight_pack_bench(tmp.path); }
    std::printf("core_weight_pack: ALL PASSED\n");
    return 0;
}

// Bare-metal Phase B: 2 MiB-aligned INGPACK1 weight pack verification, SafetensorsModel integration,
// adversarial corruption checks, DeviceUploader + UploadChunkObserver parity, and optional --bench.
// 100% CUDA-free when INGINE_BACKEND_CUDA == 0 (guarded HAL/GPU upload check when CUDA is active).

#include "common/test_util.h"
#include "core/arena.h"
#include "core/fixture.h"
#include "core/hal.h"
#include "core/host_memory.h"
#include "core/hugepage.h"
#include "core/safetensors.h"
#include "core/sha256.h"
#include "core/weight_pack.h"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

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
using ingine::SafetensorsOpenMode;
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

void test_weight_source_consistency(const std::filesystem::path& dir) {
    std::printf("[4/5] Testing same-source, changed-source, and explicit pack-only weight consistency...\n");
    const auto consistency_root = dir / "source_consistency";
    std::filesystem::create_directories(consistency_root);

    // 1. Same-source multi-shard and single-file directories + explicit SafetensorsOpenMode overrides.
    {
        const auto single_dir = consistency_root / "same_single_dir";
        std::filesystem::create_directories(single_dir);
        const std::vector<FixtureTensor> t_v1 = {
            {"w.embed", "F32", {4}, pattern_bytes(16, 41)},
            {"w.block", "BF16", {32, 64}, pattern_bytes(32 * 64 * 2, 42)},
        };
        write_safetensors(single_dir / "model.safetensors", t_v1, {{"format", "pt"}});
        (void)WeightPackBuilder::pack_from_safetensors(single_dir / "model.safetensors",
                                                       single_dir / "model.ingpack");

        SafetensorsModel auto_m(single_dir);
        INGINE_CHECK(auto_m.is_weight_pack());
        INGINE_CHECK(ingine::weight_pack_matches_safetensors_dir(*auto_m.files()[0]->weight_pack(), single_dir));
        SafetensorsModel st_only(single_dir, SafetensorsOpenMode::SafetensorsOnly);
        INGINE_CHECK(!st_only.is_weight_pack());
        INGINE_CHECK(auto_m.read("w.embed") == st_only.read("w.embed"));
        INGINE_CHECK(auto_m.read("w.block") == st_only.read("w.block"));

        ScopedEnvVar off("INGINE_DISABLE_WEIGHT_PACK", "1");
        SafetensorsModel wp_only(single_dir, SafetensorsOpenMode::WeightPackOnly);
        INGINE_CHECK(wp_only.is_weight_pack());
    }

    // 2. Changed-source subcase A: --no-vision subset change (src 25 tensors w/ vision -> dst 24 tensors w/o vision + copied pack).
    const auto src_25 = consistency_root / "src_25";
    const auto dst_24 = consistency_root / "dst_24";
    std::filesystem::create_directories(src_25);
    std::filesystem::create_directories(dst_24);
    std::vector<FixtureTensor> t_25;
    for (int i = 0; i < 24; ++i) {
        t_25.push_back({"model.llm.layers." + std::to_string(i) + ".mlp.experts.w13_weight",
                        "U8",
                        {1},
                        {static_cast<std::uint8_t>(i)}});
    }
    std::vector<FixtureTensor> t_24 = t_25;
    t_25.push_back({"model.visual.review", "U8", {1}, {std::uint8_t{7}}});
    write_safetensors(src_25 / "model-00001-of-00001.safetensors", t_25);
    write_safetensors(dst_24 / "model-00001-of-00001.safetensors", t_24);
    auto write_idx = [](const std::filesystem::path& p, const std::vector<FixtureTensor>& ts) {
        nlohmann::json idx = {{"weight_map", nlohmann::json::object()}};
        for (const auto& t : ts) { idx["weight_map"][t.name] = "model-00001-of-00001.safetensors"; }
        std::ofstream(p / "model.safetensors.index.json") << idx.dump(2);
    };
    write_idx(src_25, t_25);
    write_idx(dst_24, t_24);
    (void)WeightPackBuilder::pack_from_safetensors(src_25, src_25 / "model.ingpack");
    std::filesystem::copy_file(src_25 / "model.ingpack", dst_24 / "model.ingpack",
                               std::filesystem::copy_options::overwrite_existing);

    {
        SafetensorsModel dst_auto(dst_24);
        INGINE_CHECK(!dst_auto.is_weight_pack());
        INGINE_CHECK(dst_auto.size() == 24);
        INGINE_CHECK(!dst_auto.contains("model.visual.review"));

        SafetensorsModel dst_wp(dst_24, SafetensorsOpenMode::WeightPackOnly);
        INGINE_CHECK(dst_wp.is_weight_pack());
        INGINE_CHECK(dst_wp.size() == 25);
        INGINE_CHECK(dst_wp.contains("model.visual.review"));
    }

    // 3. Changed-source subcase B1 (<= 1 MiB full sample): identical layout, header, size, and newer pack mtime.
    {
        const auto mut_dir = consistency_root / "same_shape_mutated";
        std::filesystem::create_directories(mut_dir);
        const std::filesystem::path shard_p = mut_dir / "model-00001-of-00001.safetensors";
        const std::filesystem::path pack_p  = mut_dir / "model.ingpack";
        const std::vector<FixtureTensor> v1 = {
            {"w.large", "U8", {512 * 1024}, pattern_bytes(512 * 1024, 51)},
        };
        write_safetensors(shard_p, v1);
        write_idx(mut_dir, v1);
        (void)WeightPackBuilder::pack_from_safetensors(mut_dir, pack_p);
        INGINE_CHECK(SafetensorsModel(mut_dir).is_weight_pack());

        // Overwrite shard with identical layout/size and mutate only interior payload bytes.
        std::vector<FixtureTensor> v2 = v1;
        v2[0].data[200000] ^= 0xFFu;
        write_safetensors(shard_p, v2);
        // Set pack mtime cleanly ahead of shard mtime so timestamp-only checks would be fooled.
        const auto shard_mtime = std::filesystem::last_write_time(shard_p);
        std::filesystem::last_write_time(pack_p, shard_mtime + std::chrono::seconds(10));

        SafetensorsModel refreshed(mut_dir);
        INGINE_CHECK(!refreshed.is_weight_pack());
        const auto got = refreshed.read("w.large");
        INGINE_CHECK(got.size() == v2[0].data.size() &&
                     std::memcmp(got.data(), v2[0].data.data(), got.size()) == 0);
    }

    // 3b. Changed-source subcase B2 (> 1 MiB sampled shard): mutated shard mtime +1s with pack mtime +10s.
    {
        const auto mut_big = consistency_root / "same_shape_mutated_big";
        std::filesystem::create_directories(mut_big);
        const std::filesystem::path shard_p = mut_big / "model-00001-of-00001.safetensors";
        const std::filesystem::path pack_p  = mut_big / "model.ingpack";
        const std::size_t big_sz            = 2560u * 1024u;  // 2.5 MiB
        const std::vector<FixtureTensor> v1 = {
            {"w.big", "U8", {static_cast<std::int64_t>(big_sz)}, pattern_bytes(big_sz, 55)},
        };
        write_safetensors(shard_p, v1);
        write_idx(mut_big, v1);
        (void)WeightPackBuilder::pack_from_safetensors(mut_big, pack_p);
        INGINE_CHECK(SafetensorsModel(mut_big).is_weight_pack());

        std::vector<FixtureTensor> v2 = v1;
        v2[0].data[0] ^= 0xFFu;
        const auto orig_mtime = std::filesystem::last_write_time(shard_p);
        write_safetensors(shard_p, v2);
        std::filesystem::last_write_time(shard_p, orig_mtime + std::chrono::seconds(1));
        std::filesystem::last_write_time(pack_p, orig_mtime + std::chrono::seconds(10));

        SafetensorsModel refreshed(mut_big);
        INGINE_CHECK(!refreshed.is_weight_pack());
        const auto got = refreshed.read("w.big");
        INGINE_CHECK(got.size() == v2[0].data.size() &&
                     std::memcmp(got.data(), v2[0].data.data(), got.size()) == 0);
    }

    // 4. Changed-source subcases C, D, E: unprovenanced pack, auxiliary file packed as model.ingpack, corrupt pack.
    {
        const auto unprov_dir = consistency_root / "unprovenanced_mixed";
        std::filesystem::create_directories(unprov_dir);
        const std::vector<FixtureTensor> st_ts = {{"w.a", "U8", {8}, pattern_bytes(8, 61)}};
        write_safetensors(unprov_dir / "model.safetensors", st_ts);
        {
            WeightPackBuilder b;
            const auto alt = pattern_bytes(8, 62);
            b.add_tensor("w.wrong", "U8", {8}, alt.size(),
                         [&](const auto& emit) { emit(alt.data(), alt.size()); });
            (void)b.write(unprov_dir / "model.ingpack");
        }
        SafetensorsModel mixed_unprov(unprov_dir);
        INGINE_CHECK(!mixed_unprov.is_weight_pack());
        INGINE_CHECK(mixed_unprov.contains("w.a"));
        INGINE_CHECK(!mixed_unprov.contains("w.wrong"));

        // Explicit pack-only directory with the exact same unprovenanced pack MUST still load cleanly!
        const auto pack_only_unprov = consistency_root / "pack_only_unprov";
        std::filesystem::create_directories(pack_only_unprov);
        std::filesystem::copy_file(unprov_dir / "model.ingpack", pack_only_unprov / "model.ingpack");
        {
            ScopedEnvVar off("INGINE_DISABLE_WEIGHT_PACK", "1");
            SafetensorsModel only_m(pack_only_unprov);
            INGINE_CHECK(only_m.is_weight_pack());
            INGINE_CHECK(only_m.contains("w.wrong"));
        }

        // Auxiliary mtp.safetensors packed as model.ingpack next to model.safetensors -> rejected in Auto.
        const auto aux_mixed = consistency_root / "aux_packed_mixed";
        std::filesystem::create_directories(aux_mixed);
        write_safetensors(aux_mixed / "model.safetensors", st_ts);
        const std::vector<FixtureTensor> aux_ts = {{"mtp.w", "U8", {8}, pattern_bytes(8, 63)}};
        write_safetensors(aux_mixed / "mtp.safetensors", aux_ts);
        (void)WeightPackBuilder::pack_from_safetensors(aux_mixed / "mtp.safetensors",
                                                       aux_mixed / "model.ingpack");
        SafetensorsModel aux_auto(aux_mixed);
        INGINE_CHECK(!aux_auto.is_weight_pack());
        INGINE_CHECK(aux_auto.contains("w.a"));
        INGINE_CHECK(!aux_auto.contains("mtp.w"));

        // Corrupt model.ingpack next to valid safetensors -> Auto falls back cleanly, WeightPackOnly throws.
        const auto corrupt_mixed = consistency_root / "corrupt_mixed";
        std::filesystem::create_directories(corrupt_mixed);
        write_safetensors(corrupt_mixed / "model.safetensors", st_ts);
        write_file_bytes(corrupt_mixed / "model.ingpack", {0x00, 0x01, 0x02, 0x03});
        SafetensorsModel corrupt_auto(corrupt_mixed);
        INGINE_CHECK(!corrupt_auto.is_weight_pack());
        INGINE_CHECK(corrupt_auto.contains("w.a"));
        INGINE_CHECK(throws_runtime([&] { SafetensorsModel bad_wp(corrupt_mixed, SafetensorsOpenMode::WeightPackOnly); }));
    }

    // 5. Explicit pack-only reads (tagged directory + explicit file path + mode rejections).
    {
        const auto pack_only_dir = consistency_root / "pack_only_25";
        std::filesystem::create_directories(pack_only_dir);
        const auto explicit_pack = pack_only_dir / "model.ingpack";
        std::filesystem::copy_file(src_25 / "model.ingpack", explicit_pack);

        SafetensorsModel dir_auto(pack_only_dir);
        INGINE_CHECK(dir_auto.is_weight_pack());
        INGINE_CHECK(dir_auto.size() == 25);
        INGINE_CHECK(dir_auto.contains("model.visual.review"));

        ScopedEnvVar off("INGINE_DISABLE_WEIGHT_PACK", "1");
        SafetensorsModel dir_disabled(pack_only_dir);
        INGINE_CHECK(dir_disabled.is_weight_pack());
        INGINE_CHECK(dir_disabled.size() == 25);

        SafetensorsModel file_explicit(explicit_pack);
        INGINE_CHECK(file_explicit.is_weight_pack());
        INGINE_CHECK(file_explicit.size() == 25);

        INGINE_CHECK(throws_runtime([&] {
            SafetensorsModel bad1(explicit_pack, SafetensorsOpenMode::SafetensorsOnly);
        }));
        INGINE_CHECK(throws_runtime([&] {
            SafetensorsModel bad2(pack_only_dir, SafetensorsOpenMode::SafetensorsOnly);
        }));
        INGINE_CHECK(throws_runtime([&] {
            SafetensorsModel bad3(src_25 / "model-00001-of-00001.safetensors", SafetensorsOpenMode::WeightPackOnly);
        }));
        INGINE_CHECK(throws_runtime([&] {
            SafetensorsModel bad4(src_25 / "model.safetensors.index.json", SafetensorsOpenMode::WeightPackOnly);
        }));
    }

    // 6. Timestamp-preserving copy of same-shape release (shutil.copy2 / fixed mtime bypass check).
    auto set_fixed_mtime = [](const std::filesystem::path& p, std::time_t sec) {
        struct timespec ts[2] = {{sec, 0}, {sec, 0}};
        INGINE_CHECK(::utimensat(AT_FDCWD, p.c_str(), ts, 0) == 0);
    };
    {
        const auto rel_dir = consistency_root / "published_release_2m";
        std::filesystem::create_directories(rel_dir);
        const std::filesystem::path cur_st   = rel_dir / "model.safetensors";
        const std::filesystem::path next_st  = consistency_root / "next_release_2m.safetensors";
        const std::filesystem::path pack_out = rel_dir / "model.ingpack";

        const std::size_t elems = 512u * 1024u;  // 2 MiB F32
        std::vector<std::uint8_t> d1(elems * sizeof(float), 0u);
        std::vector<std::uint8_t> d2 = d1;
        const float one              = 1.0f;
        const std::size_t mid_off    = (256u * 1024u) * sizeof(float);
        std::memcpy(d2.data() + mid_off, &one, sizeof(one));

        write_safetensors(cur_st, {{"matrix", "F32", {512, 1024}, std::move(d1)}});
        set_fixed_mtime(cur_st, 1700000000);
        write_safetensors(next_st, {{"matrix", "F32", {512, 1024}, std::move(d2)}});
        set_fixed_mtime(next_st, 1700000000);

        (void)WeightPackBuilder::pack_from_safetensors(rel_dir, pack_out);
        {
            SafetensorsModel orig(rel_dir);
            INGINE_CHECK(orig.is_weight_pack());
            const auto b = orig.read("matrix");
            float v      = -1.0f;
            std::memcpy(&v, b.data() + mid_off, sizeof(v));
            INGINE_CHECK(v == 0.0f);
        }

        // Replace in-place with identical mtime/layout/header/size (single 1.0f interior edit).
        std::filesystem::copy_file(next_st, cur_st, std::filesystem::copy_options::overwrite_existing);
        set_fixed_mtime(cur_st, 1700000000);
        {
            SafetensorsModel updated(rel_dir);
            SafetensorsModel source_only(rel_dir, SafetensorsOpenMode::SafetensorsOnly);
            INGINE_CHECK(!updated.is_weight_pack());
            INGINE_CHECK(updated.read("matrix") == source_only.read("matrix"));
            float v = -1.0f;
            std::memcpy(&v, updated.read("matrix").data() + mid_off, sizeof(v));
            INGINE_CHECK(v == 1.0f);
        }
    }

    // 7. Large (> 32 MiB) shard without full-file SHA: fast path on unchanged inode+ctime, fail-closed on copy2 overwrite.
    {
        const auto big_dir = consistency_root / "published_release_36m";
        std::filesystem::create_directories(big_dir);
        const std::filesystem::path cur_st   = big_dir / "model.safetensors";
        const std::filesystem::path next_st  = consistency_root / "next_release_36m.safetensors";
        const std::filesystem::path pack_out = big_dir / "model.ingpack";

        const std::size_t rows    = 9216u;  // 9216 * 1024 * 4 = 36 MiB (> 32 MiB)
        const std::size_t cols    = 1024u;
        const std::size_t mid_off = (rows / 2u) * cols * sizeof(float);
        std::vector<std::uint8_t> d1(rows * cols * sizeof(float), 0u);
        std::vector<std::uint8_t> d2 = d1;
        const float one              = 1.0f;
        std::memcpy(d2.data() + mid_off, &one, sizeof(one));

        write_safetensors(cur_st, {{"matrix", "F32", {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(cols)}, std::move(d1)}});
        set_fixed_mtime(cur_st, 1700000000);
        write_safetensors(next_st, {{"matrix", "F32", {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(cols)}, std::move(d2)}});
        set_fixed_mtime(next_st, 1700000000);

        (void)WeightPackBuilder::pack_from_safetensors(big_dir, pack_out);
        {
            SafetensorsModel orig(big_dir);
            INGINE_CHECK(orig.is_weight_pack());
            const auto wp = orig.files()[0]->weight_pack();
            INGINE_CHECK(wp != nullptr);
            INGINE_CHECK(ingine::weight_pack_matches_safetensors_dir(*wp, big_dir));
        }

        // Timestamp-preserving overwrite (> 32 MiB has '-' SHA field -> ctime change MUST fail closed to source).
        std::filesystem::copy_file(next_st, cur_st, std::filesystem::copy_options::overwrite_existing);
        set_fixed_mtime(cur_st, 1700000000);
        {
            SafetensorsModel updated(big_dir);
            SafetensorsModel source_only(big_dir, SafetensorsOpenMode::SafetensorsOnly);
            INGINE_CHECK(!updated.is_weight_pack());
            const auto b = updated.read("matrix");
            INGINE_CHECK(b == source_only.read("matrix"));
            float v = -1.0f;
            std::memcpy(&v, b.data() + mid_off, sizeof(v));
            INGINE_CHECK(v == 1.0f);
        }
    }

    // 8a. Atomic release rename of model.safetensors during pack_from_safetensors (2 MiB & 36 MiB, after_open & before_copy)
    //     MUST decline publication cleanly (no model.ingpack published, no .tmp.* left), leaving rev2 source active.
    for (bool at_before_copy : {false, true}) {
        for (std::size_t rows : {std::size_t(512), std::size_t(9216)}) {
            const auto race_dir = consistency_root /
                                  ("race_atomic_single_" + std::string(at_before_copy ? "bc_" : "ao_") +
                                   std::to_string(rows));
            std::filesystem::create_directories(race_dir);
            const std::filesystem::path cur_st = race_dir / "model.safetensors";
            const std::filesystem::path next_st =
                consistency_root /
                ("race_next_single_" + std::string(at_before_copy ? "bc_" : "ao_") + std::to_string(rows) +
                 ".safetensors");
            const std::filesystem::path pack_out = race_dir / "model.ingpack";

            const std::size_t cols    = 1024u;
            const std::size_t mid_off = (rows / 2u) * cols * sizeof(float);
            std::vector<std::uint8_t> d1(rows * cols * sizeof(float), 0u);
            std::vector<std::uint8_t> d2 = d1;
            const float one              = 1.0f;
            std::memcpy(d2.data() + mid_off, &one, sizeof(one));

            write_safetensors(cur_st, {{"matrix", "F32", {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(cols)}, std::move(d1)}});
            set_fixed_mtime(cur_st, 1700000000);
            write_safetensors(next_st, {{"matrix", "F32", {static_cast<std::int64_t>(rows), static_cast<std::int64_t>(cols)}, std::move(d2)}});
            set_fixed_mtime(next_st, 1700000000);

            bool hook_ran = false;
            auto do_rename = [&]() {
                std::filesystem::rename(next_st, cur_st);
                hook_ran = true;
            };
            if (at_before_copy) {
                ingine::set_next_pack_before_copy_hook(do_rename);
            } else {
                ingine::set_next_pack_after_open_hook(do_rename);
            }
            INGINE_CHECK(throws_runtime([&] { (void)WeightPackBuilder::pack_from_safetensors(race_dir, pack_out); }));
            INGINE_CHECK(hook_ran);
            ingine::set_next_pack_after_open_hook({});
            ingine::set_next_pack_before_copy_hook({});

            INGINE_CHECK(!std::filesystem::exists(pack_out));
            for (const auto& entry : std::filesystem::directory_iterator(race_dir)) {
                INGINE_CHECK(entry.path().string().find(".tmp.") == std::string::npos);
            }
            SafetensorsModel auto_m(race_dir);
            SafetensorsModel src_m(race_dir, SafetensorsOpenMode::SafetensorsOnly);
            INGINE_CHECK(!auto_m.is_weight_pack());
            INGINE_CHECK(auto_m.read("matrix") == src_m.read("matrix"));
            float v = -1.0f;
            std::memcpy(&v, auto_m.read("matrix").data() + mid_off, sizeof(v));
            INGINE_CHECK(v == 1.0f);
        }
    }

    // 8b. Atomic release rename of model.safetensors.index.json (+ shard, or index-only in before_copy)
    //     MUST decline publication cleanly, leaving rev2 source active.
    for (bool index_only_before_copy : {false, true}) {
        const auto idx_race_dir =
            consistency_root / (index_only_before_copy ? "race_atomic_index_only_bc" : "race_atomic_index_ao");
        std::filesystem::create_directories(idx_race_dir);
        const std::filesystem::path shard_p  = idx_race_dir / "model-00001-of-00001.safetensors";
        const std::filesystem::path idx_p    = idx_race_dir / "model.safetensors.index.json";
        const std::filesystem::path staged_s = consistency_root / "race_staged_shard.safetensors";
        const std::filesystem::path staged_i = consistency_root / "race_staged_index.json";
        const std::filesystem::path pack_out = idx_race_dir / "model.ingpack";

        write_safetensors(shard_p, t_25);
        write_idx(idx_race_dir, t_25);

        nlohmann::json idx2 = {{"weight_map", nlohmann::json::object()}};
        for (const auto& t : t_24) { idx2["weight_map"][t.name] = "model-00001-of-00001.safetensors"; }
        std::ofstream(staged_i) << idx2.dump(2);
        if (!index_only_before_copy) { write_safetensors(staged_s, t_24); }

        bool hook_ran = false;
        auto do_idx_rename = [&]() {
            if (!index_only_before_copy) { std::filesystem::rename(staged_s, shard_p); }
            std::filesystem::rename(staged_i, idx_p);
            hook_ran = true;
        };
        if (index_only_before_copy) {
            ingine::set_next_pack_before_copy_hook(do_idx_rename);
        } else {
            ingine::set_next_pack_after_open_hook(do_idx_rename);
        }
        INGINE_CHECK(throws_runtime([&] { (void)WeightPackBuilder::pack_from_safetensors(idx_race_dir, pack_out); }));
        INGINE_CHECK(hook_ran);
        ingine::set_next_pack_after_open_hook({});
        ingine::set_next_pack_before_copy_hook({});

        INGINE_CHECK(!std::filesystem::exists(pack_out));
        for (const auto& entry : std::filesystem::directory_iterator(idx_race_dir)) {
            INGINE_CHECK(entry.path().string().find(".tmp.") == std::string::npos);
        }
        SafetensorsModel auto_m(idx_race_dir);
        INGINE_CHECK(!auto_m.is_weight_pack());
        INGINE_CHECK(auto_m.size() == 24);
        INGINE_CHECK(!auto_m.contains("model.visual.review"));
    }

    // 8c. In-place overwrite of the open source shard inode during pack_from_safetensors MUST decline publication.
    for (bool at_before_copy : {false, true}) {
        const auto inplace_dir = consistency_root / (at_before_copy ? "race_inplace_before_copy" : "race_inplace_after_open");
        std::filesystem::create_directories(inplace_dir);
        const std::filesystem::path cur_st   = inplace_dir / "model.safetensors";
        const std::filesystem::path next_st  = consistency_root / (at_before_copy ? "inplace_next1.safetensors" : "inplace_next2.safetensors");
        const std::filesystem::path pack_out = inplace_dir / "model.ingpack";

        std::vector<std::uint8_t> d1(512u * 1024u * sizeof(float), 0u);
        std::vector<std::uint8_t> d2 = d1;
        const float one              = 1.0f;
        std::memcpy(d2.data() + (256u * 1024u) * sizeof(float), &one, sizeof(one));

        write_safetensors(cur_st, {{"matrix", "F32", {512, 1024}, std::move(d1)}});
        set_fixed_mtime(cur_st, 1700000000);
        write_safetensors(next_st, {{"matrix", "F32", {512, 1024}, std::move(d2)}});
        const auto next_bytes = read_file_bytes(next_st);

        bool hook_ran = false;
        auto overwrite_inplace = [&]() {
            const int wfd = ::open(cur_st.c_str(), O_WRONLY | O_TRUNC | O_CLOEXEC);
            INGINE_CHECK(wfd >= 0);
            std::size_t off = 0;
            while (off < next_bytes.size()) {
                const ssize_t n = ::write(wfd, next_bytes.data() + off, next_bytes.size() - off);
                INGINE_CHECK(n > 0);
                off += static_cast<std::size_t>(n);
            }
            ::close(wfd);
            set_fixed_mtime(cur_st, 1700000000);
            hook_ran = true;
        };
        if (at_before_copy) {
            ingine::set_next_pack_before_copy_hook(overwrite_inplace);
        } else {
            ingine::set_next_pack_after_open_hook(overwrite_inplace);
        }
        INGINE_CHECK(throws_runtime([&] { (void)WeightPackBuilder::pack_from_safetensors(inplace_dir, pack_out); }));
        INGINE_CHECK(hook_ran);
        ingine::set_next_pack_after_open_hook({});
        ingine::set_next_pack_before_copy_hook({});

        INGINE_CHECK(!std::filesystem::exists(pack_out));
        for (const auto& entry : std::filesystem::directory_iterator(inplace_dir)) {
            INGINE_CHECK(entry.path().string().find(".tmp.") == std::string::npos);
        }
    }

    // 8d. Post-build atomic rename release after clean pack_from_safetensors on rev1:
    //     rev1 pack builds cleanly and passes verify_content_sha256(), while post-rename Auto open loads rev2.
    {
        const auto post_dir = consistency_root / "post_build_atomic_rename";
        std::filesystem::create_directories(post_dir);
        const std::filesystem::path cur_st   = post_dir / "model.safetensors";
        const std::filesystem::path next_st  = consistency_root / "post_build_next.safetensors";
        const std::filesystem::path pack_out = post_dir / "model.ingpack";
        const std::size_t mid_off            = (256u * 1024u) * sizeof(float);

        std::vector<std::uint8_t> d1(512u * 1024u * sizeof(float), 0u);
        std::vector<std::uint8_t> d2 = d1;
        const float one              = 1.0f;
        std::memcpy(d2.data() + mid_off, &one, sizeof(one));

        write_safetensors(cur_st, {{"matrix", "F32", {512, 1024}, std::move(d1)}});
        set_fixed_mtime(cur_st, 1700000000);
        write_safetensors(next_st, {{"matrix", "F32", {512, 1024}, std::move(d2)}});
        set_fixed_mtime(next_st, 1700000000);

        (void)WeightPackBuilder::pack_from_safetensors(post_dir, pack_out);
        std::filesystem::rename(next_st, cur_st);

        SafetensorsModel pack_only_m(post_dir, SafetensorsOpenMode::WeightPackOnly);
        INGINE_CHECK(pack_only_m.files()[0]->weight_pack()->verify_content_sha256());
        float pv = -1.0f;
        std::memcpy(&pv, pack_only_m.read("matrix").data() + mid_off, sizeof(pv));
        INGINE_CHECK(pv == 0.0f);

        SafetensorsModel auto_m(post_dir);
        INGINE_CHECK(!auto_m.is_weight_pack());
        float av = -1.0f;
        std::memcpy(&av, auto_m.read("matrix").data() + mid_off, sizeof(av));
        INGINE_CHECK(av == 1.0f);
    }
}

void run_weight_pack_bench(const std::filesystem::path& dir) {
    std::printf("[5/5] Running 128 MiB safetensors vs INGPACK1 open & read microbenchmark...\n");
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

void test_unified_weight_pack_pin(const std::filesystem::path& dir) {
    const auto pin_dir = dir / "unified_pin";
    std::filesystem::create_directories(pin_dir);
    const auto st_path   = pin_dir / "model.safetensors";
    const auto pack_path = pin_dir / "model.ingpack";
    const auto pat_a     = pattern_bytes(65536, 71);
    const auto pat_b     = pattern_bytes(131072, 93);
    write_safetensors(st_path, {
                                   {"layer.0.w", "U8", {256, 256}, pat_a},
                                   {"layer.1.w", "BF16", {256, 256}, pat_b},
                               });
    (void)WeightPackBuilder::pack_from_safetensors(st_path, pack_path);

    SafetensorsFile sf(pack_path);
    INGINE_CHECK(sf.is_weight_pack());
    auto shared_pack = sf.shared_weight_pack();
    INGINE_CHECK(static_cast<bool>(shared_pack));

    ingine::UnifiedWeightPackPin pin(shared_pack);
    INGINE_CHECK(!pin.registered());
    INGINE_CHECK(pin.device_base() == nullptr);
    const auto& ta = sf.at("layer.0.w");
    const auto& tb = sf.at("layer.1.w");
    INGINE_CHECK(pin.tensor_device_ptr(ta.offset, ta.bytes) == nullptr);

    const bool ok = pin.pin();
#if defined(INGINE_BACKEND_CUDA) && INGINE_BACKEND_CUDA
    if (ok) {
        INGINE_CHECK(pin.registered());
        INGINE_CHECK(pin.device_base() != nullptr);
        INGINE_CHECK(pin.mapped_bytes() == shared_pack->file_size());
        void* dev_a = pin.tensor_device_ptr(ta.offset, ta.bytes);
        void* dev_b = pin.tensor_device_ptr(tb.offset, tb.bytes);
        INGINE_CHECK(dev_a != nullptr && dev_b != nullptr);
        // Out-of-bounds offset or length must return nullptr.
        INGINE_CHECK(pin.tensor_device_ptr(pin.mapped_bytes(), 1) == nullptr);
        INGINE_CHECK(pin.tensor_device_ptr(ta.offset, pin.mapped_bytes() + 1) == nullptr);
        INGINE_CHECK(pin.tensor_device_ptr(UINT64_MAX - 8, 16) == nullptr);

        std::vector<std::uint8_t> got_a(ta.bytes), got_b(tb.bytes);
        INGINE_TEST_CUDA(cudaMemcpy(got_a.data(), dev_a, ta.bytes, cudaMemcpyDeviceToHost));
        INGINE_TEST_CUDA(cudaMemcpy(got_b.data(), dev_b, tb.bytes, cudaMemcpyDeviceToHost));
        INGINE_CHECK(got_a == pat_a);
        INGINE_CHECK(got_b == pat_b);
    }
#elif defined(INGINE_BACKEND_HIP) && INGINE_BACKEND_HIP
    if (ok) {
        INGINE_CHECK(pin.registered());
        INGINE_CHECK(pin.device_base() != nullptr);
        void* dev_a = pin.tensor_device_ptr(ta.offset, ta.bytes);
        INGINE_CHECK(dev_a != nullptr);
        INGINE_CHECK(pin.tensor_device_ptr(pin.mapped_bytes(), 1) == nullptr);
        std::vector<std::uint8_t> got_a(ta.bytes);
        INGINE_TEST_CUDA(cudaMemcpy(got_a.data(), dev_a, ta.bytes, cudaMemcpyDeviceToHost));
        INGINE_CHECK(got_a == pat_a);
    }
#else
    INGINE_CHECK(ok);
    INGINE_CHECK(pin.registered());
    INGINE_CHECK(pin.readonly());
    INGINE_CHECK(pin.device_base() != nullptr);
    INGINE_CHECK(pin.device_base() == pin.host_base());
    INGINE_CHECK(pin.mapped_bytes() == shared_pack->file_size());
    void* dev_a = pin.tensor_device_ptr(ta.offset, ta.bytes);
    void* dev_b = pin.tensor_device_ptr(tb.offset, tb.bytes);
    INGINE_CHECK(dev_a != nullptr && dev_b != nullptr);
    INGINE_CHECK(pin.tensor_device_ptr(pin.mapped_bytes(), 1) == nullptr);
    INGINE_CHECK(pin.tensor_device_ptr(ta.offset, pin.mapped_bytes() + 1) == nullptr);
    INGINE_CHECK(pin.tensor_device_ptr(UINT64_MAX - 8, 16) == nullptr);
    INGINE_CHECK(std::memcmp(dev_a, pat_a.data(), ta.bytes) == 0);
    INGINE_CHECK(std::memcmp(dev_b, pat_b.data(), tb.bytes) == 0);
#endif

    // Unpin clears device pointers cleanly and is idempotent.
    pin.unpin();
    pin.unpin();
    INGINE_CHECK(!pin.registered());
    INGINE_CHECK(pin.device_base() == nullptr);
    INGINE_CHECK(pin.tensor_device_ptr(ta.offset, ta.bytes) == nullptr);
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
    test_weight_source_consistency(tmp.path);
    test_unified_weight_pack_pin(tmp.path);
    if (bench) { run_weight_pack_bench(tmp.path); }
    std::printf("core_weight_pack: ALL PASSED\n");
    return 0;
}

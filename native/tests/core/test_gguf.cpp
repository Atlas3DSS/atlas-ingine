// CPU unit tests for GgufFile parser hardening: exception-safe mmap cleanup, metadata/alignment validation,
// tensor rank/dimension/block-alignment checks, checked arithmetic, and subtraction-based range bounds.
#include "core/gguf.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ingine::GgmlType;
using ingine::GgufFile;

struct Buf {
  std::vector<std::uint8_t> b;
  template <class T>
  void put(T v) {
    const std::size_t off = b.size();
    b.resize(off + sizeof(T));
    std::memcpy(b.data() + off, &v, sizeof(T));
  }
  void put_str(const std::string& s) {
    put<std::uint64_t>(s.size());
    b.insert(b.end(), s.begin(), s.end());
  }
  void put_hdr(std::uint32_t magic, std::uint32_t ver, std::uint64_t nt, std::uint64_t nkv) {
    put<std::uint32_t>(magic);
    put<std::uint32_t>(ver);
    put<std::uint64_t>(nt);
    put<std::uint64_t>(nkv);
  }
  void pad_to(std::size_t align) {
    const std::size_t rem = b.size() % align;
    if (rem) b.resize(b.size() + (align - rem), 0);
  }
};

std::filesystem::path write_tmp(const std::filesystem::path& dir, const std::string& name, const std::vector<std::uint8_t>& bytes) {
  const auto p = dir / name;
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
  out.close();
  return p;
}

bool maps_contain(const std::string& needle) {
  std::ifstream in("/proc/self/maps");
  std::string line;
  while (std::getline(in, line)) {
    if (line.find(needle) != std::string::npos) return true;
  }
  return false;
}

}  // namespace

int main() {
  bool ok = true;
  auto expect = [&](bool cond, const char* what) {
    if (!cond) std::printf("FAIL %s\n", what);
    ok &= cond;
  };
  auto expect_throws_unmapped = [&](const std::filesystem::path& p, const char* what) {
    bool threw = false;
    try {
      GgufFile gf(p);
      (void)gf.version();
    } catch (const std::runtime_error&) {
      threw = true;
    }
    const bool leaked = maps_contain(p.filename().string());
    expect(threw, what);
    expect(!leaked, (std::string("no leaked mmap after throw: ") + what).c_str());
  };

  const auto dir = std::filesystem::temp_directory_path() / "ingine_test_core_gguf";
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir);

  // 1. Valid minimal GGUF v3 file with 1 F32 tensor + 1 Q8_0 tensor and general.alignment=64.
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 2u, 2u);
    // KV 0: general.architecture = "inkling2_loop" (T_STR=8)
    f.put_str("general.architecture");
    f.put<std::uint32_t>(8u);
    f.put_str("inkling2_loop");
    // KV 1: general.alignment = 64 (T_U32=4)
    f.put_str("general.alignment");
    f.put<std::uint32_t>(4u);
    f.put<std::uint32_t>(64u);
    // Tensor 0: F32 [4, 2] at offset 0 (32 bytes)
    f.put_str("t_f32");
    f.put<std::uint32_t>(2u);
    f.put<std::uint64_t>(4u);
    f.put<std::uint64_t>(2u);
    f.put<std::uint32_t>(std::uint32_t(GgmlType::F32));
    f.put<std::uint64_t>(0u);
    // Tensor 1: Q8_0 [32, 2] at offset 64 (2 * 34 = 68 bytes)
    f.put_str("t_q8");
    f.put<std::uint32_t>(2u);
    f.put<std::uint64_t>(32u);
    f.put<std::uint64_t>(2u);
    f.put<std::uint32_t>(std::uint32_t(GgmlType::Q8_0));
    f.put<std::uint64_t>(64u);
    f.pad_to(64u);
    const std::size_t d0 = f.b.size();
    f.b.resize(d0 + 64u + 68u, 0);
    const float val = 3.25f;
    std::memcpy(f.b.data() + d0, &val, sizeof(val));

    const auto p = write_tmp(dir, "valid_ok.gguf", f.b);
    {
      GgufFile gf(p);
      expect(maps_contain("valid_ok.gguf"), "valid GgufFile maps file while alive");
      expect(gf.version() == 3u, "valid version");
      expect(gf.get_string("general.architecture") == "inkling2_loop", "valid string KV");
      expect(gf.get_int("general.alignment") == 64, "valid alignment KV");
      const auto& t0 = gf.at("t_f32");
      expect(t0.bytes == 32u && t0.numel() == 8 && t0.row_len() == 4 && t0.rows() == 2, "valid f32 tensor");
      float got = 0.f;
      std::memcpy(&got, t0.data, sizeof(got));
      expect(got == 3.25f, "valid tensor data pointer");
      const auto& t1 = gf.at("t_q8");
      expect(t1.bytes == 68u && t1.row_bytes() == 34u, "valid q8_0 tensor");
    }
    expect(!maps_contain("valid_ok.gguf"), "valid GgufFile unmaps cleanly on destruction");
  }

  // 2. Empty file & bad magic & unsupported version -> throws ordinary runtime_error, zero leaked mappings.
  expect_throws_unmapped(write_tmp(dir, "empty_err.gguf", {}), "empty file rejected");
  {
    Buf f;
    f.put_hdr(0xDEADBEEFu, 3u, 0u, 0u);
    expect_throws_unmapped(write_tmp(dir, "bad_magic.gguf", f.b), "bad magic rejected and unmapped");
  }
  {
    Buf f;
    f.put_hdr(0x46554747u, 99u, 0u, 0u);
    expect_throws_unmapped(write_tmp(dir, "bad_ver.gguf", f.b), "bad version rejected and unmapped");
  }

  // 3. Truncated / huge crafted header counts & metadata array counts (must throw runtime_error, not bad_alloc).
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 1000000000ull, 0u);
    expect_throws_unmapped(write_tmp(dir, "huge_nt.gguf", f.b), "huge n_tensors rejected before reserve");
  }
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 0u, 1000000000ull);
    expect_throws_unmapped(write_tmp(dir, "huge_nkv.gguf", f.b), "huge n_kv rejected");
  }
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 0u, 1u);
    f.put_str("arr");
    f.put<std::uint32_t>(9u);  // T_ARR
    f.put<std::uint32_t>(4u);  // T_U32
    f.put<std::uint64_t>(std::numeric_limits<std::uint64_t>::max() / 2);
    expect_throws_unmapped(write_tmp(dir, "huge_arr.gguf", f.b), "huge metadata array rejected");
  }
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 0u, 1u);
    f.put_str("nested");
    f.put<std::uint32_t>(9u);  // T_ARR
    f.put<std::uint32_t>(9u);  // nested T_ARR
    f.put<std::uint64_t>(1u);
    expect_throws_unmapped(write_tmp(dir, "nested_arr.gguf", f.b), "nested array type rejected");
  }
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 0u, 1u);
    f.put<std::uint64_t>(999999ull);  // truncated key string
    expect_throws_unmapped(write_tmp(dir, "trunc_str.gguf", f.b), "truncated string rejected");
  }

  // 4. Invalid general.alignment: 0, negative, non-power-of-two, string, huge exceeding file size.
  auto test_bad_align_i32 = [&](std::int32_t a, const char* fn, const char* msg) {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 0u, 1u);
    f.put_str("general.alignment");
    f.put<std::uint32_t>(5u);  // T_I32
    f.put<std::int32_t>(a);
    f.b.resize(f.b.size() + 64, 0);
    expect_throws_unmapped(write_tmp(dir, fn, f.b), msg);
  };
  test_bad_align_i32(0, "align_zero.gguf", "alignment 0 rejected");
  test_bad_align_i32(-32, "align_neg.gguf", "negative alignment rejected");
  test_bad_align_i32(48, "align_nonpow2.gguf", "non-power-of-two alignment rejected");
  test_bad_align_i32(1 << 20, "align_past_eof.gguf", "alignment past EOF rejected");
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 0u, 1u);
    f.put_str("general.alignment");
    f.put<std::uint32_t>(8u);  // T_STR
    f.put_str("32");
    f.b.resize(f.b.size() + 64, 0);
    expect_throws_unmapped(write_tmp(dir, "align_str.gguf", f.b), "string general.alignment rejected");
  }

  // 5. Invalid tensor rank (0 and 5), invalid dimension (0 and > INT64_MAX), non-block-aligned ne[0], dim overflow.
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 1u, 0u);
    f.put_str("rank0");
    f.put<std::uint32_t>(0u);
    f.put<std::uint32_t>(0u);
    f.put<std::uint64_t>(0u);
    f.b.resize(f.b.size() + 64, 0);
    expect_throws_unmapped(write_tmp(dir, "rank0.gguf", f.b), "rank 0 rejected");
  }
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 1u, 0u);
    f.put_str("rank5");
    f.put<std::uint32_t>(5u);
    for (int i = 0; i < 5; ++i) f.put<std::uint64_t>(1u);
    f.put<std::uint32_t>(0u);
    f.put<std::uint64_t>(0u);
    f.b.resize(f.b.size() + 64, 0);
    expect_throws_unmapped(write_tmp(dir, "rank5.gguf", f.b), "rank > 4 rejected");
  }
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 1u, 0u);
    f.put_str("dim0");
    f.put<std::uint32_t>(2u);
    f.put<std::uint64_t>(32u);
    f.put<std::uint64_t>(0u);
    f.put<std::uint32_t>(0u);
    f.put<std::uint64_t>(0u);
    f.b.resize(f.b.size() + 64, 0);
    expect_throws_unmapped(write_tmp(dir, "dim0.gguf", f.b), "zero dimension rejected");
  }
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 1u, 0u);
    f.put_str("unaligned_q8");
    f.put<std::uint32_t>(1u);
    f.put<std::uint64_t>(33u);  // not a multiple of Q8_0 block_values=32
    f.put<std::uint32_t>(std::uint32_t(GgmlType::Q8_0));
    f.put<std::uint64_t>(0u);
    f.b.resize(f.b.size() + 128, 0);
    expect_throws_unmapped(write_tmp(dir, "unaligned_q8.gguf", f.b), "non-block-aligned ne[0] rejected");
  }
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 1u, 0u);
    f.put_str("mul_overflow");
    f.put<std::uint32_t>(2u);
    f.put<std::uint64_t>(std::uint64_t(1) << 40);
    f.put<std::uint64_t>(std::uint64_t(1) << 40);
    f.put<std::uint32_t>(std::uint32_t(GgmlType::F32));
    f.put<std::uint64_t>(0u);
    f.b.resize(f.b.size() + 128, 0);
    expect_throws_unmapped(write_tmp(dir, "mul_overflow.gguf", f.b), "dimension multiply overflow rejected");
  }

  // 6. Subtraction-based range check catches wrap-around offset (UINT64_MAX - 8) and past-end tensor.
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 1u, 0u);
    f.put_str("wrap_off");
    f.put<std::uint32_t>(1u);
    f.put<std::uint64_t>(4u);
    f.put<std::uint32_t>(std::uint32_t(GgmlType::F32));
    f.put<std::uint64_t>(std::numeric_limits<std::uint64_t>::max() - 8u);
    f.b.resize(f.b.size() + 128, 0);
    expect_throws_unmapped(write_tmp(dir, "wrap_off.gguf", f.b), "wrap-around offset rejected");
  }
  {
    Buf f;
    f.put_hdr(0x46554747u, 3u, 1u, 0u);
    f.put_str("past_eof");
    f.put<std::uint32_t>(1u);
    f.put<std::uint64_t>(64u);  // 256 bytes
    f.put<std::uint32_t>(std::uint32_t(GgmlType::F32));
    f.put<std::uint64_t>(0u);
    f.pad_to(32u);
    f.b.resize(f.b.size() + 64u, 0);  // only 64B available < 256B
    expect_throws_unmapped(write_tmp(dir, "past_eof.gguf", f.b), "tensor past EOF rejected");
  }

  // 7. ggml_row_bytes rejects negative and non-block-aligned lengths.
  {
    bool threw_neg = false, threw_unaligned = false;
    try { (void)ingine::ggml_row_bytes(GgmlType::Q8_0, -32); } catch (const std::runtime_error&) { threw_neg = true; }
    try { (void)ingine::ggml_row_bytes(GgmlType::Q8_0, 33); } catch (const std::runtime_error&) { threw_unaligned = true; }
    expect(threw_neg && threw_unaligned, "ggml_row_bytes rejects negative and unaligned widths");
  }

  std::filesystem::remove_all(dir, ec);
  std::printf("%s core_gguf\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

#include "backends/qwen4exp/qwen4exp_gguf.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <stdexcept>

namespace ingine::backends::qwen4exp {
namespace {

class Sha256 {
 public:
  Sha256() { reset(); }

  void reset() noexcept {
    state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    total_bits_ = 0;
    buf_len_ = 0;
  }

  void update(const void* data, std::size_t len) noexcept {
    const auto* p = static_cast<const std::uint8_t*>(data);
    total_bits_ += static_cast<std::uint64_t>(len) * 8ull;
    while (len > 0) {
      const std::size_t take = std::min<std::size_t>(len, 64 - buf_len_);
      std::memcpy(buf_.data() + buf_len_, p, take);
      buf_len_ += take;
      p += take;
      len -= take;
      if (buf_len_ == 64) {
        compress(buf_.data());
        buf_len_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> finalize() noexcept {
    buf_[buf_len_++] = 0x80u;
    if (buf_len_ > 56) {
      while (buf_len_ < 64) buf_[buf_len_++] = 0;
      compress(buf_.data());
      buf_len_ = 0;
    }
    while (buf_len_ < 56) buf_[buf_len_++] = 0;
    for (int i = 0; i < 8; ++i) {
      buf_[56 + i] = static_cast<std::uint8_t>((total_bits_ >> (56 - 8 * i)) & 0xffu);
    }
    compress(buf_.data());
    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
      out[4 * i + 0] = static_cast<std::uint8_t>((state_[i] >> 24) & 0xffu);
      out[4 * i + 1] = static_cast<std::uint8_t>((state_[i] >> 16) & 0xffu);
      out[4 * i + 2] = static_cast<std::uint8_t>((state_[i] >> 8) & 0xffu);
      out[4 * i + 3] = static_cast<std::uint8_t>(state_[i] & 0xffu);
    }
    return out;
  }

 private:
  static inline std::uint32_t rotr(std::uint32_t x, std::uint32_t n) noexcept { return (x >> n) | (x << (32 - n)); }

  void compress(const std::uint8_t* block) noexcept {
    static constexpr std::array<std::uint32_t, 64> kK = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(block[4 * i + 0]) << 24) |
             (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
             (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) |
             static_cast<std::uint32_t>(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    auto [a, b, c, d, e, f, g, h] = state_;
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = h + s1 + ch + kK[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{};
  std::uint64_t total_bits_ = 0;
  std::array<std::uint8_t, 64> buf_{};
  std::size_t buf_len_ = 0;
};

float fp16_to_f32(std::uint16_t h) noexcept {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1fu;
  const std::uint32_t mant = h & 0x03ffu;
  std::uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Subnormal FP16 -> normalized FP32.
      float v = std::ldexp(static_cast<float>(mant), -24);
      return (h & 0x8000u) ? -v : v;
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

float bf16_to_f32(std::uint16_t b) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

std::uint16_t read_le16(const std::uint8_t* p) noexcept {
  return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}

std::uint32_t read_le32(const std::uint8_t* p) noexcept {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::shared_ptr<int> open_cached_fd(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("cannot open shard " + path.string() + ": " + std::strerror(errno));
  }
  return std::shared_ptr<int>(new int(fd), [](int* p) {
    if (p) {
      if (*p >= 0) ::close(*p);
      delete p;
    }
  });
}

void pread_exact(int cached_fd, const std::filesystem::path& path, std::uint64_t offset, std::uint8_t* dst,
                 std::size_t len) {
  int owned_fd = -1;
  int fd = cached_fd;
  if (fd < 0) {
    owned_fd = ::open(path.c_str(), O_RDONLY);
    if (owned_fd < 0) {
      throw std::runtime_error("cannot open shard " + path.string() + ": " + std::strerror(errno));
    }
    fd = owned_fd;
  }
  struct FdGuard {
    int fd;
    ~FdGuard() {
      if (fd >= 0) ::close(fd);
    }
  } guard{owned_fd};

  std::size_t done = 0;
  while (done < len) {
    const ssize_t n = ::pread(fd, dst + done, len - done, static_cast<off_t>(offset + done));
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("pread failed on " + path.string() + ": " + std::strerror(errno));
    }
    if (n == 0) {
      throw std::runtime_error("unexpected EOF in " + path.string() + " at offset " +
                               std::to_string(offset + done));
    }
    done += static_cast<std::size_t>(n);
  }
}

struct StreamReader {
  std::ifstream& in;
  std::uint64_t pos = 0;
  std::uint64_t size = 0;

  void read_bytes(void* dst, std::size_t n) {
    if (n > size - pos) throw std::invalid_argument("truncated GGUF file");
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
    if (!in) throw std::invalid_argument("GGUF read error");
    pos += n;
  }

  void skip_bytes(std::uint64_t n) {
    if (n > size - pos) throw std::invalid_argument("truncated GGUF payload");
    in.seekg(static_cast<std::streamoff>(n), std::ios::cur);
    if (!in) throw std::invalid_argument("GGUF seek error");
    pos += n;
  }

  template <typename T>
  T read_le() {
    std::array<std::uint8_t, sizeof(T)> b{};
    read_bytes(b.data(), sizeof(T));
    T v{};
    std::memcpy(&v, b.data(), sizeof(T));
    return v;
  }

  std::string read_str() {
    const std::uint64_t len = read_le<std::uint64_t>();
    if (len > size - pos || len > 32ull * 1024ull * 1024ull) {
      throw std::invalid_argument("invalid GGUF string length");
    }
    std::string s(len, '\0');
    if (len > 0) read_bytes(s.data(), len);
    return s;
  }
};

std::size_t scalar_size(std::uint32_t vtype) noexcept {
  switch (vtype) {
    case 0:
    case 1:
    case 7: return 1;
    case 2:
    case 3: return 2;
    case 4:
    case 5:
    case 6: return 4;
    case 10:
    case 11:
    case 12: return 8;
    default: return 0;
  }
}

void skip_gguf_value(StreamReader& r, std::uint32_t vtype, int depth) {
  if (depth > 8) throw std::invalid_argument("GGUF metadata nesting too deep");
  if (const std::size_t sz = scalar_size(vtype); sz > 0) {
    r.skip_bytes(sz);
    return;
  }
  if (vtype == 8) {
    const std::uint64_t len = r.read_le<std::uint64_t>();
    r.skip_bytes(len);
    return;
  }
  if (vtype == 9) {
    const std::uint32_t et = r.read_le<std::uint32_t>();
    const std::uint64_t cnt = r.read_le<std::uint64_t>();
    if (const std::size_t esz = scalar_size(et); esz > 0) {
      if (cnt > (r.size - r.pos) / esz) throw std::invalid_argument("GGUF array overflow");
      r.skip_bytes(cnt * esz);
      return;
    }
    if (cnt > 10000000ull) throw std::invalid_argument("GGUF array count exceeds cap");
    for (std::uint64_t i = 0; i < cnt; ++i) skip_gguf_value(r, et, depth + 1);
    return;
  }
  throw std::invalid_argument("unknown GGUF value type " + std::to_string(vtype));
}

bool should_store_kv_key(std::string_view key) noexcept {
  if (key == "tokenizer.ggml.tokens" || key == "tokenizer.ggml.merges" || key == "tokenizer.ggml.token_type" ||
      key == "tokenizer.ggml.scores") {
    return false;
  }
  return true;
}

GgufValue read_gguf_value(StreamReader& r, std::uint32_t vtype) {
  switch (vtype) {
    case 0: return GgufValue{std::uint64_t(r.read_le<std::uint8_t>())};
    case 1: return GgufValue{std::int64_t(r.read_le<std::int8_t>())};
    case 2: return GgufValue{std::uint64_t(r.read_le<std::uint16_t>())};
    case 3: return GgufValue{std::int64_t(r.read_le<std::int16_t>())};
    case 4: return GgufValue{std::uint64_t(r.read_le<std::uint32_t>())};
    case 5: return GgufValue{std::int64_t(r.read_le<std::int32_t>())};
    case 6: return GgufValue{double(r.read_le<float>())};
    case 7: return GgufValue{r.read_le<std::uint8_t>() != 0};
    case 8: return GgufValue{r.read_str()};
    case 10: return GgufValue{r.read_le<std::uint64_t>()};
    case 11: return GgufValue{r.read_le<std::int64_t>()};
    case 12: return GgufValue{r.read_le<double>()};
    case 9: {
      const std::uint32_t et = r.read_le<std::uint32_t>();
      const std::uint64_t cnt = r.read_le<std::uint64_t>();
      if (cnt > 1000000ull) throw std::invalid_argument("stored GGUF array exceeds limit");
      if (et == 0 || et == 2 || et == 4 || et == 10) {
        std::vector<std::uint64_t> out;
        out.reserve(cnt);
        for (std::uint64_t i = 0; i < cnt; ++i) {
          if (et == 0) out.push_back(r.read_le<std::uint8_t>());
          else if (et == 2) out.push_back(r.read_le<std::uint16_t>());
          else if (et == 4) out.push_back(r.read_le<std::uint32_t>());
          else out.push_back(r.read_le<std::uint64_t>());
        }
        return GgufValue{std::move(out)};
      }
      if (et == 1 || et == 3 || et == 5 || et == 11) {
        std::vector<std::int64_t> out;
        out.reserve(cnt);
        for (std::uint64_t i = 0; i < cnt; ++i) {
          if (et == 1) out.push_back(r.read_le<std::int8_t>());
          else if (et == 3) out.push_back(r.read_le<std::int16_t>());
          else if (et == 5) out.push_back(r.read_le<std::int32_t>());
          else out.push_back(r.read_le<std::int64_t>());
        }
        return GgufValue{std::move(out)};
      }
      if (et == 6 || et == 12) {
        std::vector<double> out;
        out.reserve(cnt);
        for (std::uint64_t i = 0; i < cnt; ++i) {
          out.push_back(et == 6 ? double(r.read_le<float>()) : r.read_le<double>());
        }
        return GgufValue{std::move(out)};
      }
      if (et == 7) {
        std::vector<bool> out;
        out.reserve(cnt);
        for (std::uint64_t i = 0; i < cnt; ++i) out.push_back(r.read_le<std::uint8_t>() != 0);
        return GgufValue{std::move(out)};
      }
      if (et == 8) {
        std::vector<std::string> out;
        out.reserve(cnt);
        for (std::uint64_t i = 0; i < cnt; ++i) out.push_back(r.read_str());
        return GgufValue{std::move(out)};
      }
      throw std::invalid_argument("unsupported GGUF array element type " + std::to_string(et));
    }
    default: throw std::invalid_argument("unsupported GGUF value type " + std::to_string(vtype));
  }
}

bool is_auxiliary_filename(std::string_view fname) noexcept {
  std::string lower(fname);
  for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return lower.find("mmproj") != std::string::npos || lower.find("imatrix") != std::string::npos ||
         lower.find("dflash") != std::string::npos || lower.find("mtp") != std::string::npos;
}

std::vector<std::filesystem::path> discover_gguf_shards(const std::filesystem::path& input_path) {
  static const std::regex split_re(R"(^(.*)-([0-9]{5})-of-([0-9]{5})\.gguf$)");

  auto collect_matching_splits = [&](const std::filesystem::path& dir, const std::string& stem, int total) {
    if (total <= 0 || total > 1000) throw std::invalid_argument("invalid GGUF split count");
    std::vector<std::filesystem::path> shards;
    shards.reserve(static_cast<std::size_t>(total));
    for (int i = 1; i <= total; ++i) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "-%05d-of-%05d.gguf", i, total);
      const std::filesystem::path p = dir / (stem + buf);
      if (!std::filesystem::is_regular_file(p)) {
        throw std::invalid_argument("missing contiguous GGUF shard: " + p.string());
      }
      shards.push_back(p);
    }
    return shards;
  };

  if (std::filesystem::is_regular_file(input_path)) {
    if (input_path.extension() != ".gguf") {
      throw std::invalid_argument("expected .gguf file, got: " + input_path.string());
    }
    const std::string fname = input_path.filename().string();
    std::smatch m;
    if (std::regex_match(fname, m, split_re)) {
      return collect_matching_splits(input_path.parent_path(), m[1].str(), std::stoi(m[3].str()));
    }
    return {input_path};
  }

  if (!std::filesystem::is_directory(input_path)) {
    throw std::invalid_argument("GGUF checkpoint path not found: " + input_path.string());
  }

  std::vector<std::filesystem::path> all_ggufs;
  for (const auto& entry : std::filesystem::directory_iterator(input_path)) {
    if (!entry.is_regular_file()) continue;
    if (entry.path().extension() != ".gguf") continue;
    if (is_auxiliary_filename(entry.path().filename().string())) continue;
    all_ggufs.push_back(entry.path());
  }
  if (all_ggufs.empty()) {
    throw std::invalid_argument("no backbone .gguf shards found in " + input_path.string());
  }
  std::sort(all_ggufs.begin(), all_ggufs.end());

  if (all_ggufs.size() == 1) {
    const std::string fname = all_ggufs[0].filename().string();
    std::smatch m;
    if (std::regex_match(fname, m, split_re)) {
      return collect_matching_splits(input_path, m[1].str(), std::stoi(m[3].str()));
    }
    return all_ggufs;
  }

  // Require all backbone GGUF files in directory to belong to ONE contiguous split set.
  std::string expected_stem;
  int expected_total = 0;
  std::set<int> seen_idx;
  for (const auto& p : all_ggufs) {
    const std::string fname = p.filename().string();
    std::smatch m;
    if (!std::regex_match(fname, m, split_re)) {
      throw std::invalid_argument("ambiguous multi-GGUF directory contains non-split file: " + fname);
    }
    const std::string stem = m[1].str();
    const int idx = std::stoi(m[2].str());
    const int total = std::stoi(m[3].str());
    if (expected_stem.empty()) {
      expected_stem = stem;
      expected_total = total;
    } else if (stem != expected_stem || total != expected_total) {
      throw std::invalid_argument("ambiguous GGUF directory contains multiple split sets (" + expected_stem + " vs " +
                                  stem + ")");
    }
    if (idx < 1 || idx > total || !seen_idx.insert(idx).second) {
      throw std::invalid_argument("invalid or duplicate GGUF split index in " + fname);
    }
  }
  if (static_cast<int>(seen_idx.size()) != expected_total) {
    throw std::invalid_argument("incomplete GGUF split set in " + input_path.string() + ": found " +
                                std::to_string(seen_idx.size()) + " of " + std::to_string(expected_total));
  }
  return collect_matching_splits(input_path, expected_stem, expected_total);
}

void get_scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) noexcept {
  if (j < 4) {
    d = q[j] & 63u;
    m = q[j + 4] & 63u;
  } else {
    d = (q[j + 4] & 0x0fu) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

}  // namespace

std::array<std::uint8_t, 32> sha256_bytes(const void* data, std::size_t len) noexcept {
  Sha256 h;
  h.update(data, len);
  return h.finalize();
}

std::array<std::uint8_t, 32> sha256_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open file for sha256: " + path.string());
  Sha256 h;
  std::vector<char> buf(4 * 1024 * 1024);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize n = in.gcount();
    if (n > 0) h.update(buf.data(), static_cast<std::size_t>(n));
  }
  return h.finalize();
}

std::string sha256_to_hex(const std::array<std::uint8_t, 32>& digest) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out(64, '0');
  for (std::size_t i = 0; i < 32; ++i) {
    out[2 * i + 0] = kHex[digest[i] >> 4];
    out[2 * i + 1] = kHex[digest[i] & 0x0fu];
  }
  return out;
}

const char* ggml_type_name(std::uint32_t type_id) noexcept {
  switch (type_id) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "Q4_0";
    case 7: return "Q5_1";
    case 8: return "Q8_0";
    case 12: return "Q4_K";
    case 13: return "Q5_K";
    case 14: return "Q6_K";
    case 18: return "IQ3_XXS";
    case 20: return "IQ4_NL";
    case 23: return "IQ4_XS";
    case 30: return "BF16";
    default: return "UNKNOWN";
  }
}

bool ggml_block_geometry(std::uint32_t type_id, GgmlBlockGeometry& out) noexcept {
  switch (type_id) {
    case 0: out = {1, 4}; return true;     // F32
    case 1: out = {1, 2}; return true;     // F16
    case 2: out = {32, 18}; return true;   // Q4_0
    case 7: out = {32, 24}; return true;   // Q5_1
    case 8: out = {32, 34}; return true;   // Q8_0
    case 12: out = {256, 144}; return true;  // Q4_K
    case 13: out = {256, 176}; return true;  // Q5_K
    case 14: out = {256, 210}; return true;  // Q6_K
    case 18: out = {256, 98}; return true;   // IQ3_XXS
    case 20: out = {32, 18}; return true;    // IQ4_NL
    case 23: out = {256, 136}; return true;  // IQ4_XS
    case 30: out = {1, 2}; return true;      // BF16
    default: return false;
  }
}

void dequantize_row_to_f32(std::uint32_t type_id, std::span<const std::uint8_t> raw, std::span<float> out) {
  GgmlBlockGeometry geom{};
  if (!ggml_block_geometry(type_id, geom)) {
    throw std::invalid_argument("unsupported GGML dequantization type " + std::to_string(type_id));
  }
  const std::size_t n = out.size();
  if (n % geom.block_elems != 0) {
    throw std::invalid_argument("row length not divisible by GGML block size");
  }
  const std::size_t nb = n / geom.block_elems;
  if (raw.size() < nb * geom.block_bytes) {
    throw std::invalid_argument("raw GGML row buffer too small");
  }

  const std::uint8_t* src = raw.data();
  switch (type_id) {
    case 0:  // F32
      std::memcpy(out.data(), src, n * sizeof(float));
      return;
    case 1:  // F16
      for (std::size_t i = 0; i < n; ++i) out[i] = fp16_to_f32(read_le16(src + 2 * i));
      return;
    case 30:  // BF16
      for (std::size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(read_le16(src + 2 * i));
      return;
    case 2: {  // Q4_0: fp16 d (2B) + 16 packed bytes (lo 0..15, hi 16..31)
      for (std::size_t b = 0; b < nb; ++b) {
        const std::uint8_t* blk = src + b * 18;
        const float d = fp16_to_f32(read_le16(blk));
        const std::uint8_t* qs = blk + 2;
        float* dst = out.data() + b * 32;
        for (int j = 0; j < 16; ++j) {
          dst[j] = (static_cast<int>(qs[j] & 0x0fu) - 8) * d;
          dst[j + 16] = (static_cast<int>(qs[j] >> 4) - 8) * d;
        }
      }
      return;
    }
    case 7: {  // Q5_1: fp16 d (2B) + fp16 m (2B) + u32 qh (4B) + 16 packed bytes
      for (std::size_t b = 0; b < nb; ++b) {
        const std::uint8_t* blk = src + b * 24;
        const float d = fp16_to_f32(read_le16(blk));
        const float m = fp16_to_f32(read_le16(blk + 2));
        const std::uint32_t qh = read_le32(blk + 4);
        const std::uint8_t* qs = blk + 8;
        float* dst = out.data() + b * 32;
        for (int j = 0; j < 16; ++j) {
          const int xh0 = static_cast<int>((qh >> j) & 1u) << 4;
          const int xh1 = static_cast<int>((qh >> (j + 16)) & 1u) << 4;
          dst[j] = static_cast<float>((qs[j] & 0x0fu) | xh0) * d + m;
          dst[j + 16] = static_cast<float>((qs[j] >> 4) | xh1) * d + m;
        }
      }
      return;
    }
    case 8: {  // Q8_0: fp16 d (2B) + 32 int8 qs
      for (std::size_t b = 0; b < nb; ++b) {
        const std::uint8_t* blk = src + b * 34;
        const float d = fp16_to_f32(read_le16(blk));
        const auto* qs = reinterpret_cast<const std::int8_t*>(blk + 2);
        float* dst = out.data() + b * 32;
        for (int j = 0; j < 32; ++j) dst[j] = static_cast<float>(qs[j]) * d;
      }
      return;
    }
    case 12: {  // Q4_K: fp16 d (2B) + fp16 dmin (2B) + scales[12] + qs[128]
      for (std::size_t b = 0; b < nb; ++b) {
        const std::uint8_t* blk = src + b * 144;
        const float d = fp16_to_f32(read_le16(blk));
        const float min = fp16_to_f32(read_le16(blk + 2));
        const std::uint8_t* scales = blk + 4;
        const std::uint8_t* q = blk + 16;
        float* y = out.data() + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
          std::uint8_t sc = 0, m = 0;
          get_scale_min_k4(is + 0, scales, sc, m);
          const float d1 = d * static_cast<float>(sc);
          const float m1 = min * static_cast<float>(m);
          get_scale_min_k4(is + 1, scales, sc, m);
          const float d2 = d * static_cast<float>(sc);
          const float m2 = min * static_cast<float>(m);
          for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>(q[l] & 0x0fu) - m1;
          for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>(q[l] >> 4) - m2;
          q += 32;
          is += 2;
        }
      }
      return;
    }
    case 13: {  // Q5_K: fp16 d (2B) + fp16 dmin (2B) + scales[12] + qh[32] + qs[128]
      for (std::size_t b = 0; b < nb; ++b) {
        const std::uint8_t* blk = src + b * 176;
        const float d = fp16_to_f32(read_le16(blk));
        const float min = fp16_to_f32(read_le16(blk + 2));
        const std::uint8_t* scales = blk + 4;
        const std::uint8_t* qh = blk + 16;
        const std::uint8_t* ql = blk + 48;
        float* y = out.data() + b * 256;
        int is = 0;
        std::uint8_t u1 = 1u, u2 = 2u;
        for (int j = 0; j < 256; j += 64) {
          std::uint8_t sc = 0, m = 0;
          get_scale_min_k4(is + 0, scales, sc, m);
          const float d1 = d * static_cast<float>(sc);
          const float m1 = min * static_cast<float>(m);
          get_scale_min_k4(is + 1, scales, sc, m);
          const float d2 = d * static_cast<float>(sc);
          const float m2 = min * static_cast<float>(m);
          for (int l = 0; l < 32; ++l) {
            *y++ = d1 * static_cast<float>((ql[l] & 0x0fu) + ((qh[l] & u1) ? 16 : 0)) - m1;
          }
          for (int l = 0; l < 32; ++l) {
            *y++ = d2 * static_cast<float>((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
          }
          ql += 32;
          is += 2;
          u1 <<= 2;
          u2 <<= 2;
        }
      }
      return;
    }
    case 14: {  // Q6_K: ql[128] + qh[64] + scales[16] (int8) + fp16 d (2B) = 210 bytes
      for (std::size_t b = 0; b < nb; ++b) {
        const std::uint8_t* blk = src + b * 210;
        const std::uint8_t* ql = blk;
        const std::uint8_t* qh = blk + 128;
        const auto* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
        const float d = fp16_to_f32(read_le16(blk + 208));
        float* y = out.data() + b * 256;
        for (int n128 = 0; n128 < 256; n128 += 128) {
          for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int q1 = static_cast<int>((ql[l + 0] & 0x0fu) | (((qh[l] >> 0) & 3u) << 4)) - 32;
            const int q2 = static_cast<int>((ql[l + 32] & 0x0fu) | (((qh[l] >> 2) & 3u) << 4)) - 32;
            const int q3 = static_cast<int>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
            const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;
            y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
            y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
            y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
            y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
          }
          y += 128;
          ql += 64;
          qh += 32;
          sc += 8;
        }
      }
      return;
    }
    default:
      throw std::invalid_argument(std::string("dequantize_row_to_f32 not implemented for GGML type ") +
                                  ggml_type_name(type_id));
  }
}

std::int64_t GgufValue::as_i64() const {
  if (const auto* v = std::get_if<std::int64_t>(&value)) return *v;
  if (const auto* u = std::get_if<std::uint64_t>(&value)) {
    if (*u > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::invalid_argument("GGUF uint64 overflows int64");
    }
    return static_cast<std::int64_t>(*u);
  }
  throw std::invalid_argument("GGUF value is not integer");
}

std::uint64_t GgufValue::as_u64() const {
  if (const auto* u = std::get_if<std::uint64_t>(&value)) return *u;
  if (const auto* v = std::get_if<std::int64_t>(&value)) {
    if (*v < 0) throw std::invalid_argument("GGUF negative int64 for uint64");
    return static_cast<std::uint64_t>(*v);
  }
  throw std::invalid_argument("GGUF value is not unsigned integer");
}

double GgufValue::as_f64() const {
  if (const auto* d = std::get_if<double>(&value)) return *d;
  if (const auto* v = std::get_if<std::int64_t>(&value)) return static_cast<double>(*v);
  if (const auto* u = std::get_if<std::uint64_t>(&value)) return static_cast<double>(*u);
  throw std::invalid_argument("GGUF value is not numeric");
}

bool GgufValue::as_bool() const {
  if (const auto* b = std::get_if<bool>(&value)) return *b;
  if (const auto* u = std::get_if<std::uint64_t>(&value)) return *u != 0;
  throw std::invalid_argument("GGUF value is not bool");
}

const std::string& GgufValue::as_string() const {
  if (const auto* s = std::get_if<std::string>(&value)) return *s;
  throw std::invalid_argument("GGUF value is not string");
}

std::vector<std::int64_t> GgufValue::as_i64_array() const {
  if (const auto* a = std::get_if<std::vector<std::int64_t>>(&value)) return *a;
  if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&value)) {
    std::vector<std::int64_t> out;
    out.reserve(u->size());
    for (std::uint64_t x : *u) out.push_back(static_cast<std::int64_t>(x));
    return out;
  }
  throw std::invalid_argument("GGUF value is not int array");
}

std::vector<std::uint64_t> GgufValue::as_u64_array() const {
  if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&value)) return *u;
  if (const auto* a = std::get_if<std::vector<std::int64_t>>(&value)) {
    std::vector<std::uint64_t> out;
    out.reserve(a->size());
    for (std::int64_t x : *a) {
      if (x < 0) throw std::invalid_argument("negative entry in uint64 array");
      out.push_back(static_cast<std::uint64_t>(x));
    }
    return out;
  }
  throw std::invalid_argument("GGUF value is not uint array");
}

std::uint64_t GgufTensorEntry::numel() const noexcept {
  if (dims.empty()) return 0;
  std::uint64_t n = 1;
  for (std::uint64_t d : dims) n *= d;
  return n;
}

std::uint64_t GgufTensorEntry::row_bytes() const noexcept {
  GgmlBlockGeometry g{};
  if (dims.empty() || !ggml_block_geometry(type_id, g) || dims[0] % g.block_elems != 0) return 0;
  return (dims[0] / g.block_elems) * g.block_bytes;
}

std::uint64_t GgufTensorEntry::num_rows() const noexcept {
  if (dims.empty()) return 0;
  std::uint64_t r = 1;
  for (std::size_t i = 1; i < dims.size(); ++i) r *= dims[i];
  return r;
}

Qwen4ExpGgufSet Qwen4ExpGgufSet::inspect_headers(const std::filesystem::path& path) {
  return load_impl(path, nullptr);
}

Qwen4ExpGgufSet Qwen4ExpGgufSet::inspect_pinned(const std::filesystem::path& path,
                                                const ingine::model::CheckpointPinSpec& pin) {
  return load_impl(path, &pin);
}

Qwen4ExpGgufSet Qwen4ExpGgufSet::load_impl(const std::filesystem::path& path,
                                           const ingine::model::CheckpointPinSpec* pin) {
  Qwen4ExpGgufSet set;
  const auto shard_paths = discover_gguf_shards(path);
  set.shards_.reserve(shard_paths.size());

  std::unordered_map<std::string, const ingine::model::ExpectedShardDigest*> expected_by_name;
  if (pin) {
    if (pin->shards.size() != shard_paths.size()) {
      throw std::invalid_argument("Qwen4ExpGgufSet pin shard count mismatch: expected " +
                                  std::to_string(pin->shards.size()) + ", found " +
                                  std::to_string(shard_paths.size()));
    }
    for (const auto& exp : pin->shards) {
      expected_by_name.emplace(exp.filename, &exp);
    }
  }

  for (std::size_t sidx = 0; sidx < shard_paths.size(); ++sidx) {
    const auto& sp = shard_paths[sidx];
    std::ifstream in(sp, std::ios::binary | std::ios::ate);
    if (!in) throw std::invalid_argument("cannot open GGUF shard: " + sp.string());
    const std::uint64_t fsize = static_cast<std::uint64_t>(in.tellg());
    if (fsize < 24) throw std::invalid_argument("truncated GGUF shard: " + sp.string());
    in.seekg(0, std::ios::beg);

    StreamReader r{in, 0, fsize};
    char magic[4]{};
    r.read_bytes(magic, 4);
    if (std::string_view(magic, 4) != "GGUF") {
      throw std::invalid_argument("invalid GGUF magic in " + sp.string());
    }
    const std::uint32_t ver = r.read_le<std::uint32_t>();
    if (ver != 2 && ver != 3) {
      throw std::invalid_argument("unsupported GGUF version " + std::to_string(ver) + " in " + sp.string());
    }
    const std::uint64_t n_tensors = r.read_le<std::uint64_t>();
    const std::uint64_t n_kv = r.read_le<std::uint64_t>();

    GgufShardInfo sinfo;
    sinfo.path = sp;
    sinfo.filename = sp.filename().string();
    sinfo.file_size = fsize;
    sinfo.tensor_count = n_tensors;
    sinfo.kv_count = n_kv;
    sinfo.alignment = 32;

    for (std::uint64_t i = 0; i < n_kv; ++i) {
      const std::string key = r.read_str();
      const std::uint32_t vtype = r.read_le<std::uint32_t>();
      if (sidx == 0 && should_store_kv_key(key)) {
        GgufValue val = read_gguf_value(r, vtype);
        if (key == "general.alignment") sinfo.alignment = static_cast<std::uint32_t>(val.as_u64());
        set.kv_.insert_or_assign(key, std::move(val));
      } else {
        if (key == "general.alignment") {
          GgufValue val = read_gguf_value(r, vtype);
          sinfo.alignment = static_cast<std::uint32_t>(val.as_u64());
        } else {
          skip_gguf_value(r, vtype, 0);
        }
      }
    }
    if (sinfo.alignment == 0 || (sinfo.alignment & (sinfo.alignment - 1)) != 0) {
      throw std::invalid_argument("invalid GGUF alignment in " + sp.string());
    }

    struct RawTensorMeta {
      std::string name;
      std::vector<std::uint64_t> dims;
      std::uint32_t type_id = 0;
      std::uint64_t rel_offset = 0;
    };
    std::vector<RawTensorMeta> metas;
    metas.reserve(n_tensors);
    for (std::uint64_t i = 0; i < n_tensors; ++i) {
      RawTensorMeta m;
      m.name = r.read_str();
      const std::uint32_t nd = r.read_le<std::uint32_t>();
      if (nd == 0 || nd > 4) throw std::invalid_argument("invalid tensor ndims in " + sp.string());
      m.dims.resize(nd);
      for (std::uint32_t d = 0; d < nd; ++d) m.dims[d] = r.read_le<std::uint64_t>();
      m.type_id = r.read_le<std::uint32_t>();
      m.rel_offset = r.read_le<std::uint64_t>();
      metas.push_back(std::move(m));
    }

    const std::uint64_t align = sinfo.alignment;
    sinfo.data_start = (r.pos + align - 1) & ~(align - 1);
    if (sinfo.data_start > fsize) {
      throw std::invalid_argument("GGUF data_start past EOF in " + sp.string());
    }

    for (auto& m : metas) {
      GgmlBlockGeometry geom{};
      if (!ggml_block_geometry(m.type_id, geom)) {
        throw std::invalid_argument("unknown GGML type " + std::to_string(m.type_id) + " on tensor " + m.name);
      }
      if (m.dims[0] == 0 || (m.dims[0] % geom.block_elems) != 0) {
        throw std::invalid_argument("tensor inner dim not aligned to GGML block on " + m.name);
      }
      std::uint64_t numel = 1;
      for (std::uint64_t d : m.dims) numel *= d;
      const std::uint64_t byte_sz = (numel / geom.block_elems) * geom.block_bytes;
      const std::uint64_t abs_off = sinfo.data_start + m.rel_offset;
      if (abs_off > fsize || byte_sz > fsize - abs_off) {
        throw std::invalid_argument("GGUF tensor out of bounds: " + m.name + " in " + sp.string());
      }
      GgufTensorEntry entry;
      entry.name = std::move(m.name);
      entry.dims = std::move(m.dims);
      entry.type_id = m.type_id;
      entry.data_offset = abs_off;
      entry.byte_size = byte_sz;
      entry.shard_index = sidx;
      const std::string key_name = entry.name;
      if (!set.tensors_.emplace(key_name, std::move(entry)).second) {
        throw std::invalid_argument("duplicate GGUF tensor across shards: " + key_name);
      }
      set.total_tensor_bytes_ += byte_sz;
    }

    if (pin) {
      const auto it = expected_by_name.find(sinfo.filename);
      if (it == expected_by_name.end()) {
        throw std::invalid_argument("unexpected GGUF shard filename not in CheckpointPinSpec: " + sinfo.filename);
      }
      const auto& exp = *it->second;
      if (exp.byte_size != sinfo.file_size) {
        throw std::invalid_argument("GGUF shard size mismatch on " + sinfo.filename + ": expected " +
                                    std::to_string(exp.byte_size) + ", actual " + std::to_string(sinfo.file_size));
      }
      sinfo.sha256 = sha256_file(sp);
      if (sinfo.sha256 != exp.sha256) {
        throw std::invalid_argument("GGUF shard SHA-256 mismatch on " + sinfo.filename);
      }
      sinfo.sha256_verified = true;
    }

    sinfo.cached_fd = open_cached_fd(sp);
    set.shards_.push_back(std::move(sinfo));
  }

  set.parse_config_from_kv();
  set.validate_qwen4exp_tensor_schema();
  set.compute_manifest_digest();

  set.identity_.format = ingine::model::CheckpointFormat::Gguf;
  set.identity_.schema_variant = "qwen4exp_gguf_v1";
  set.identity_.model_id = set.config_.model_name.empty() ? "Qwen3.8-Flash-Next" : set.config_.model_name;
  set.identity_.shard_count = static_cast<std::uint32_t>(set.shards_.size());
  set.identity_.weight_manifest_verified = true;
  if (pin) {
    set.identity_.verification_level = ingine::model::VerificationLevel::PinnedVerified;
    set.identity_.shard_digests_verified = true;
    Sha256 sh;
    for (const auto& s : set.shards_) {
      sh.update(s.filename.data(), s.filename.size());
      sh.update(&s.file_size, sizeof(s.file_size));
      sh.update(s.sha256.data(), s.sha256.size());
    }
    set.identity_.shard_set_sha256 = sh.finalize();
  } else {
    set.identity_.verification_level = ingine::model::VerificationLevel::HeadersOnly;
    set.identity_.shard_digests_verified = false;
  }
  return set;
}

void Qwen4ExpGgufSet::parse_config_from_kv() {
  auto req = [&](const std::string& k) -> const GgufValue& {
    const auto it = kv_.find(k);
    if (it == kv_.end()) throw std::invalid_argument("missing required GGUF metadata key: " + k);
    return it->second;
  };
  auto opt = [&](const std::string& k) -> const GgufValue* {
    const auto it = kv_.find(k);
    return it == kv_.end() ? nullptr : &it->second;
  };

  config_.architecture = req("general.architecture").as_string();
  if (const auto* n = opt("general.name")) config_.model_name = n->as_string();

  const std::string p = config_.architecture + ".";
  config_.num_hidden_layers = static_cast<int>(req(p + "block_count").as_i64());
  config_.hidden_size = static_cast<int>(req(p + "embedding_length").as_i64());
  config_.max_position_embeddings = static_cast<int>(req(p + "context_length").as_i64());
  config_.rms_norm_eps = static_cast<float>(req(p + "attention.layer_norm_rms_epsilon").as_f64());

  config_.hc_mult = static_cast<int>(req(p + "hyper_connection.count").as_i64());
  config_.hc_low_rank = static_cast<int>(req(p + "hyper_connection.low_rank").as_i64());

  config_.num_attention_heads = static_cast<int>(req(p + "attention.head_count").as_i64());
  config_.num_key_value_heads = static_cast<int>(req(p + "attention.head_count_kv").as_i64());
  config_.head_dim = static_cast<int>(req(p + "attention.key_length").as_i64());
  if (const auto* vl = opt(p + "attention.value_length")) {
    if (static_cast<int>(vl->as_i64()) != config_.head_dim) {
      throw std::invalid_argument("qwen4exp requires attention.value_length == attention.key_length");
    }
  }
  if (const auto* fai = opt(p + "full_attention_interval")) {
    config_.full_attention_interval = static_cast<int>(fai->as_i64());
  }
  if (const auto* rd = opt(p + "rope.dimension_count")) {
    config_.rotary_dim = static_cast<int>(rd->as_i64());
  }
  if (const auto* rf = opt(p + "rope.freq_base")) {
    config_.rope_theta = static_cast<float>(rf->as_f64());
  }
  if (const auto* rs = opt(p + "rope.dimension_sections")) {
    const auto arr = rs->as_i64_array();
    if (arr.size() >= 4) {
      for (int i = 0; i < 4; ++i) config_.rope_sections[static_cast<std::size_t>(i)] = static_cast<int>(arr[i]);
    }
  }

  if (const auto* ih = opt(p + "attention.indexer.head_count")) {
    config_.indexer_head_count = static_cast<int>(ih->as_i64());
  }
  if (const auto* ik = opt(p + "attention.indexer.key_length")) {
    config_.indexer_key_length = static_cast<int>(ik->as_i64());
  }
  if (const auto* itk = opt(p + "attention.indexer.top_k")) {
    config_.indexer_top_k = static_cast<int>(itk->as_i64());
  }
  if (const auto* cr = opt(p + "attention.compress_ratios")) {
    const auto arr = cr->as_i64_array();
    config_.compress_ratios.clear();
    config_.compress_ratios.reserve(arr.size());
    for (std::int64_t x : arr) config_.compress_ratios.push_back(static_cast<int>(x));
  }

  config_.ssm_conv_kernel = static_cast<int>(req(p + "ssm.conv_kernel").as_i64());
  config_.ssm_state_size = static_cast<int>(req(p + "ssm.state_size").as_i64());
  config_.ssm_group_count = static_cast<int>(req(p + "ssm.group_count").as_i64());
  config_.ssm_time_step_rank = static_cast<int>(req(p + "ssm.time_step_rank").as_i64());
  config_.ssm_inner_size = static_cast<int>(req(p + "ssm.inner_size").as_i64());

  config_.expert_count = static_cast<int>(req(p + "expert_count").as_i64());
  config_.expert_used_count = static_cast<int>(req(p + "expert_used_count").as_i64());
  config_.expert_ffn_length = static_cast<int>(req(p + "expert_feed_forward_length").as_i64());
  if (const auto* sh = opt(p + "expert_shared_feed_forward_length")) {
    config_.shared_expert_ffn_length = static_cast<int>(sh->as_i64());
  }
  if (const auto* ws = opt(p + "expert_weights_scale")) {
    config_.expert_weights_scale = static_cast<float>(ws->as_f64());
  }

  if (const auto* pl = opt(p + "ple.layers")) {
    const auto arr = pl->as_i64_array();
    config_.ple_layers.clear();
    for (std::int64_t x : arr) config_.ple_layers.push_back(static_cast<int>(x));
    if (!config_.ple_layers.empty()) {
      config_.ple_ngram_size = static_cast<int>(req(p + "ple.ngram_size").as_i64());
      config_.ple_heads_per_ngram = static_cast<int>(req(p + "ple.heads_per_ngram").as_i64());
      config_.ple_conv_kernel = static_cast<int>(req(p + "ple.conv_kernel").as_i64());
      config_.ple_eos_token_id = static_cast<std::int32_t>(req(p + "ple.eos_token_id").as_i64());
      if (const auto* im = opt(p + "ple.image_token_id")) {
        config_.ple_image_token_id = static_cast<std::int32_t>(im->as_i64());
      }
      config_.ple_head_dim = static_cast<int>(req(p + "embedding_length_per_layer_input").as_i64());
      config_.ple_layer_multipliers = req(p + "ple.layer_multipliers").as_u64_array();
      config_.ple_head_offsets = req(p + "ple.head_offsets").as_u64_array();
      config_.ple_head_vocab_sizes = req(p + "ple.head_vocab_sizes").as_u64_array();
    }
  }

  // Derive vocab_size from token_embd.weight dims[1].
  const auto& tok = require_tensor("token_embd.weight");
  if (tok.dims.size() != 2) throw std::invalid_argument("token_embd.weight must be 2D");
  config_.vocab_size = static_cast<int>(tok.dims[1]);

  config_.finalize_and_validate();
}

void Qwen4ExpGgufSet::validate_qwen4exp_tensor_schema() {
  const auto check_dims = [&](const std::string& name, std::initializer_list<std::uint64_t> expected) {
    const auto& t = require_tensor(name);
    if (t.dims.size() != expected.size()) {
      throw std::invalid_argument("tensor " + name + " rank mismatch");
    }
    std::size_t i = 0;
    for (std::uint64_t d : expected) {
      if (t.dims[i] != d) {
        throw std::invalid_argument("tensor " + name + " dim[" + std::to_string(i) + "] mismatch: expected " +
                                    std::to_string(d) + ", got " + std::to_string(t.dims[i]));
      }
      ++i;
    }
  };

  const std::uint64_t H = static_cast<std::uint64_t>(config_.hidden_size);
  const std::uint64_t V = static_cast<std::uint64_t>(config_.vocab_size);
  const std::uint64_t hc = static_cast<std::uint64_t>(config_.hc_mult);
  const std::uint64_t hc_dim = static_cast<std::uint64_t>(config_.hc_dim());
  const std::uint64_t hc_lr = static_cast<std::uint64_t>(config_.hc_low_rank);

  check_dims("token_embd.weight", {H, V});
  check_dims("output_hc_norm.weight", {hc_dim});
  check_dims("output_hc_down.weight", {hc_dim, hc_lr});
  check_dims("output_hc_up.weight", {hc_lr, hc_dim});
  if (find_tensor("output.weight") != nullptr) {
    check_dims("output.weight", {H, V});
  }

  if (config_.ple_n_heads() > 0) {
    const auto& ple_t = require_tensor("per_layer_token_embd.weight");
    if (ple_t.dims.size() != 2 || ple_t.dims[0] != static_cast<std::uint64_t>(config_.ple_head_dim)) {
      throw std::invalid_argument("per_layer_token_embd.weight shape mismatch");
    }
    ple_table_rows_ = ple_t.dims[1];
    for (int h = 0; h < config_.ple_n_heads(); ++h) {
      const std::uint64_t end_row =
          config_.ple_head_offsets[static_cast<std::size_t>(h)] + config_.ple_head_vocab_sizes[static_cast<std::size_t>(h)];
      if (end_row > ple_table_rows_) {
        throw std::invalid_argument("PLE head vocab exceeds per_layer_token_embd.weight row count");
      }
    }
  }

  const std::uint64_t E = static_cast<std::uint64_t>(config_.expert_count);
  const std::uint64_t ff_e = static_cast<std::uint64_t>(config_.expert_ffn_length);
  const std::uint64_t ff_s = static_cast<std::uint64_t>(config_.shared_expert_ffn_length);
  const std::uint64_t conv_dim = static_cast<std::uint64_t>(config_.gdn_conv_dim());
  const std::uint64_t val_dim = static_cast<std::uint64_t>(config_.gdn_value_dim());
  const std::uint64_t v_heads = static_cast<std::uint64_t>(config_.ssm_time_step_rank);

  for (int il = 0; il < config_.num_hidden_layers; ++il) {
    const std::string p = "blk." + std::to_string(il) + ".";
    check_dims(p + "hc_attn_norm.weight", {hc_dim});
    check_dims(p + "hc_attn_down.weight", {hc_dim, hc_lr});
    check_dims(p + "hc_attn_up.weight", {hc_lr, hc_dim});
    check_dims(p + "hc_attn_inject.weight", {hc_dim, hc});
    check_dims(p + "hc_ffn_norm.weight", {hc_dim});
    check_dims(p + "hc_ffn_down.weight", {hc_dim, hc_lr});
    check_dims(p + "hc_ffn_up.weight", {hc_lr, hc_dim});
    check_dims(p + "hc_ffn_inject.weight", {hc_dim, hc});

    if (config_.is_linear(il)) {
      check_dims(p + "attn_qkv.weight", {H, conv_dim});
      check_dims(p + "attn_gate.weight", {H, val_dim});
      check_dims(p + "ssm_conv1d.weight", {static_cast<std::uint64_t>(config_.ssm_conv_kernel), conv_dim});
      check_dims(p + "ssm_dt.bias", {v_heads});
      check_dims(p + "ssm_a", {v_heads});
      check_dims(p + "ssm_beta.weight", {H, v_heads});
      check_dims(p + "ssm_alpha.weight", {H, v_heads});
      check_dims(p + "ssm_norm.weight", {static_cast<std::uint64_t>(config_.ssm_state_size)});
      check_dims(p + "ssm_out.weight", {val_dim, H});
    } else {
      const std::uint64_t q_dim = static_cast<std::uint64_t>(config_.full_q_out_dim());
      const std::uint64_t kv_dim = static_cast<std::uint64_t>(config_.full_kv_out_dim());
      const std::uint64_t hd = static_cast<std::uint64_t>(config_.head_dim);
      check_dims(p + "attn_q.weight", {H, q_dim});
      check_dims(p + "attn_k.weight", {H, kv_dim});
      check_dims(p + "attn_v.weight", {H, kv_dim});
      check_dims(p + "attn_output.weight", {static_cast<std::uint64_t>(config_.num_attention_heads) * hd, H});
      check_dims(p + "attn_q_norm.weight", {hd});
      check_dims(p + "attn_k_norm.weight", {hd});
      if (config_.has_qsa(il)) {
        const std::uint64_t idim = static_cast<std::uint64_t>(config_.indexer_key_length);
        check_dims(p + "indexer.q_proj.weight", {H, static_cast<std::uint64_t>(config_.indexer_head_count) * idim});
        check_dims(p + "indexer.k_proj.weight", {H, idim});
        check_dims(p + "indexer.q_norm.weight", {idim});
        check_dims(p + "indexer.k_norm.weight", {idim});
      }
    }

    if (config_.is_ple(il)) {
      check_dims(p + "ple_key.weight", {H, hc_dim});
      check_dims(p + "ple_value.weight", {H, H});
      check_dims(p + "ple_norm_key.weight", {hc_dim});
      check_dims(p + "ple_norm_query.weight", {hc_dim});
      check_dims(p + "ple_norm_conv.weight", {hc_dim});
      check_dims(p + "ple_conv1d.weight", {static_cast<std::uint64_t>(config_.ple_conv_kernel), hc_dim});
    }

    check_dims(p + "ffn_gate_inp.weight", {H, E});
    check_dims(p + "ffn_gate_exps.weight", {H, ff_e, E});
    check_dims(p + "ffn_up_exps.weight", {H, ff_e, E});
    check_dims(p + "ffn_down_exps.weight", {ff_e, H, E});

    if (ff_s > 0) {
      check_dims(p + "ffn_gate_inp_shexp.weight", {H});
      check_dims(p + "ffn_gate_shexp.weight", {H, ff_s});
      check_dims(p + "ffn_up_shexp.weight", {H, ff_s});
      check_dims(p + "ffn_down_shexp.weight", {ff_s, H});
    }
  }
}

void Qwen4ExpGgufSet::compute_manifest_digest() {
  std::vector<const GgufTensorEntry*> sorted;
  sorted.reserve(tensors_.size());
  for (const auto& [_, e] : tensors_) sorted.push_back(&e);
  std::sort(sorted.begin(), sorted.end(), [](const GgufTensorEntry* a, const GgufTensorEntry* b) {
    return a->name < b->name;
  });
  Sha256 h;
  const std::string fp = config_.canonical_fingerprint();
  h.update(fp.data(), fp.size());
  for (const auto* e : sorted) {
    h.update(e->name.data(), e->name.size());
    h.update(&e->type_id, sizeof(e->type_id));
    for (std::uint64_t d : e->dims) h.update(&d, sizeof(d));
    h.update(&e->byte_size, sizeof(e->byte_size));
  }
  identity_.weight_manifest_sha256 = h.finalize();
}

const GgufTensorEntry* Qwen4ExpGgufSet::find_tensor(std::string_view name) const noexcept {
  const auto it = tensors_.find(std::string(name));
  return it == tensors_.end() ? nullptr : &it->second;
}

const GgufTensorEntry& Qwen4ExpGgufSet::require_tensor(std::string_view name) const {
  const auto* e = find_tensor(name);
  if (!e) throw std::invalid_argument("missing required qwen4exp GGUF tensor: " + std::string(name));
  return *e;
}

std::vector<std::uint8_t> Qwen4ExpGgufSet::read_tensor_raw(const GgufTensorEntry& entry) const {
  std::vector<std::uint8_t> buf(entry.byte_size);
  if (entry.byte_size > 0) {
    const auto& s = shards_.at(entry.shard_index);
    pread_exact(s.cached_fd ? *s.cached_fd : -1, s.path, entry.data_offset, buf.data(), entry.byte_size);
  }
  return buf;
}

std::vector<float> Qwen4ExpGgufSet::read_tensor_f32(const GgufTensorEntry& entry) const {
  const std::uint64_t rows = entry.num_rows();
  return read_tensor_rows_f32(entry, 0, rows);
}

void Qwen4ExpGgufSet::read_tensor_row_f32(const GgufTensorEntry& entry, std::uint64_t row_index,
                                          std::span<float> out_f32) const {
  const std::uint64_t total_rows = entry.num_rows();
  if (row_index >= total_rows) {
    throw std::out_of_range("row_index out of range for tensor " + entry.name);
  }
  const std::uint64_t cols = entry.dims[0];
  if (out_f32.size() != cols) {
    throw std::invalid_argument("output span mismatch for row read on " + entry.name);
  }
  const std::uint64_t rb = entry.row_bytes();
  std::vector<std::uint8_t> raw(rb);
  const auto& s = shards_.at(entry.shard_index);
  pread_exact(s.cached_fd ? *s.cached_fd : -1, s.path, entry.data_offset + row_index * rb, raw.data(), rb);
  dequantize_row_to_f32(entry.type_id, raw, out_f32);
}

void Qwen4ExpGgufSet::read_tensor_rows_raw(const GgufTensorEntry& entry, std::uint64_t first_row, std::uint64_t count,
                                           std::span<std::uint8_t> out_raw) const {
  const std::uint64_t total_rows = entry.num_rows();
  if (first_row > total_rows || count > total_rows - first_row) {
    throw std::out_of_range("row slice out of range for tensor " + entry.name);
  }
  const std::uint64_t rb = entry.row_bytes();
  const std::uint64_t need = count * rb;
  if (out_raw.size() < need) {
    throw std::invalid_argument("output raw span too small for tensor " + entry.name);
  }
  if (need > 0) {
    const auto& s = shards_.at(entry.shard_index);
    pread_exact(s.cached_fd ? *s.cached_fd : -1, s.path, entry.data_offset + first_row * rb, out_raw.data(), need);
  }
}

std::vector<float> Qwen4ExpGgufSet::read_tensor_rows_f32(const GgufTensorEntry& entry, std::uint64_t first_row,
                                                         std::uint64_t count) const {
  const std::uint64_t total_rows = entry.num_rows();
  if (first_row > total_rows || count > total_rows - first_row) {
    throw std::out_of_range("row slice out of range for tensor " + entry.name);
  }
  const std::uint64_t cols = entry.dims[0];
  const std::uint64_t rb = entry.row_bytes();
  std::vector<std::uint8_t> raw(count * rb);
  if (count > 0 && rb > 0) {
    read_tensor_rows_raw(entry, first_row, count, raw);
  }
  std::vector<float> out(count * cols);
  for (std::uint64_t r = 0; r < count; ++r) {
    dequantize_row_to_f32(entry.type_id, std::span<const std::uint8_t>(raw.data() + r * rb, rb),
                          std::span<float>(out.data() + r * cols, cols));
  }
  return out;
}

ingine::model::CheckpointPinSpec qwen4exp_ud_q6_k_xl_pin_spec() {
  using ingine::model::ExpectedShardDigest;
  using ingine::model::parse_sha256_hex;
  ingine::model::CheckpointPinSpec spec;
  spec.shards = {
      ExpectedShardDigest{"Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf", 10946624ull,
                          parse_sha256_hex("8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7")},
      ExpectedShardDigest{"Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf", 682434912ull,
                          parse_sha256_hex("494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e")},
      ExpectedShardDigest{"Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf", 54400261312ull,
                          parse_sha256_hex("34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20")},
      ExpectedShardDigest{"Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf", 49814108512ull,
                          parse_sha256_hex("0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997")},
      ExpectedShardDigest{"Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf", 49419317344ull,
                          parse_sha256_hex("9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9")},
      ExpectedShardDigest{"Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf", 14838313984ull,
                          parse_sha256_hex("e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f")},
  };
  return spec;
}

}  // namespace ingine::backends::qwen4exp

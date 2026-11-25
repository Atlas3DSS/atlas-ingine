#include "backends/qwen38/qwen38_manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>

namespace ingine::backends::qwen38 {
namespace {

bool is_zero_digest(const std::array<std::uint8_t, 32>& d) noexcept {
  for (std::uint8_t b : d) {
    if (b != 0) return false;
  }
  return true;
}

class Sha256 {
 public:
  Sha256() { reset(); }

  void reset() {
    state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    total_bits_ = 0;
    buf_len_ = 0;
  }

  void update(const void* data, std::size_t len) {
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

  std::array<std::uint8_t, 32> finalize() {
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
  static inline std::uint32_t rotr(std::uint32_t x, std::uint32_t n) noexcept {
    return (x >> n) | (x << (32 - n));
  }

  void compress(const std::uint8_t* block) {
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

std::string to_hex(const std::array<std::uint8_t, 32>& d) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out(64, '0');
  for (std::size_t i = 0; i < 32; ++i) {
    out[2 * i] = kHex[(d[i] >> 4) & 0xf];
    out[2 * i + 1] = kHex[d[i] & 0xf];
  }
  return out;
}

struct ScopedFd {
  int fd = -1;
  ScopedFd() = default;
  explicit ScopedFd(int f) noexcept : fd(f) {}
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  ScopedFd(ScopedFd&& other) noexcept : fd(other.fd) { other.fd = -1; }
  ScopedFd& operator=(ScopedFd&& other) noexcept {
    if (this != &other) {
      reset();
      fd = other.fd;
      other.fd = -1;
    }
    return *this;
  }
  ~ScopedFd() { reset(); }
  void reset() noexcept {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }
  [[nodiscard]] int get() const noexcept { return fd; }
};

int open_readonly_fd(const std::filesystem::path& path, const char* what) {
  int fd = -1;
  do {
    fd = ::open(path.c_str(), O_RDONLY);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    throw std::invalid_argument(std::string("cannot open ") + what + ": " + path.string() + " (" +
                                std::strerror(errno) + ")");
  }
  return fd;
}

std::optional<ScopedFd> try_open_readonly_fd(const std::filesystem::path& path, const char* what) {
  int fd = -1;
  do {
    fd = ::open(path.c_str(), O_RDONLY);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    if (errno == ENOENT) return std::nullopt;
    throw std::invalid_argument(std::string("cannot open ") + what + ": " + path.string() + " (" +
                                std::strerror(errno) + ")");
  }
  return ScopedFd(fd);
}

struct stat fstat_regular(int fd, const std::filesystem::path& path) {
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    throw std::invalid_argument("fstat failed for " + path.string() + ": " + std::strerror(errno));
  }
  if (!S_ISREG(st.st_mode)) {
    throw std::invalid_argument("expected regular file, got non-regular path: " + path.string());
  }
  if (st.st_size < 0) {
    throw std::invalid_argument("negative file size reported for " + path.string());
  }
  return st;
}

void read_exact_fd(int fd, void* dst, std::size_t n, const std::filesystem::path& path) {
  auto* out = static_cast<std::uint8_t*>(dst);
  std::size_t off = 0;
  while (off < n) {
    const ssize_t got = ::read(fd, out + off, n - off);
    if (got < 0) {
      if (errno == EINTR) continue;
      throw std::invalid_argument("read error on " + path.string() + ": " + std::strerror(errno));
    }
    if (got == 0) {
      throw std::invalid_argument("unexpected EOF (concurrent truncation) in " + path.string());
    }
    off += static_cast<std::size_t>(got);
  }
}

struct VerifiedSmallFile {
  std::string bytes;
  Qwen38VerifiedFile identity;
};

VerifiedSmallFile consume_and_hash_small_fd(int fd, const std::filesystem::path& path, std::string_view filename) {
  const struct stat st = fstat_regular(fd, path);
  const std::uint64_t fsize = static_cast<std::uint64_t>(st.st_size);
  constexpr std::uint64_t kMaxSmallFileBytes = 256ull * 1024ull * 1024ull;
  if (fsize > kMaxSmallFileBytes) {
    throw std::invalid_argument("metadata file exceeds 256 MiB limit: " + path.string());
  }
  VerifiedSmallFile out;
  out.bytes.reserve(static_cast<std::size_t>(fsize));
  Sha256 h;
  std::array<char, 65536> buf{};
  while (true) {
    const ssize_t got = ::read(fd, buf.data(), buf.size());
    if (got < 0) {
      if (errno == EINTR) continue;
      throw std::invalid_argument("read error on " + path.string() + ": " + std::strerror(errno));
    }
    if (got == 0) break;
    const std::size_t chunk = static_cast<std::size_t>(got);
    if (out.bytes.size() + chunk > fsize) {
      throw std::invalid_argument("concurrent file growth detected while reading " + path.string());
    }
    out.bytes.append(buf.data(), chunk);
    h.update(buf.data(), chunk);
  }
  if (out.bytes.size() != fsize) {
    throw std::invalid_argument("concurrent file truncation detected while reading " + path.string());
  }
  out.identity.filename = std::string(filename);
  out.identity.dev = static_cast<std::uint64_t>(st.st_dev);
  out.identity.ino = static_cast<std::uint64_t>(st.st_ino);
  out.identity.byte_size = fsize;
  out.identity.sha256 = h.finalize();
  out.identity.payload_hashed = true;
  return out;
}

std::optional<VerifiedSmallFile> read_and_hash_small_file_fd(const std::filesystem::path& path,
                                                             std::string_view filename, bool optional) {
  auto fd_opt = try_open_readonly_fd(path, filename.data());
  if (!fd_opt.has_value()) {
    if (optional) return std::nullopt;
    throw std::invalid_argument("missing " + std::string(filename) + ": " + path.string());
  }
  return consume_and_hash_small_fd(fd_opt->get(), path, filename);
}

std::vector<std::int32_t> collect_media_token_ids_from_tokenizer(std::string_view json_text) {
  nlohmann::json root;
  try {
    root = nlohmann::json::parse(json_text);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument(std::string("invalid tokenizer.json: ") + e.what());
  }
  if (!root.is_object() || !root.contains("added_tokens") || !root.at("added_tokens").is_array()) {
    return {};
  }
  static constexpr std::array<std::string_view, 8> kMediaTokens = {
      "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>",
      "<|video_pad|>",    "<|audio_start|>", "<|audio_end|>",  "<|audio_pad|>",
  };
  std::vector<std::int32_t> ids;
  for (const auto& entry : root.at("added_tokens")) {
    if (!entry.is_object() || !entry.contains("id") || !entry.at("id").is_number_integer() ||
        !entry.contains("content") || !entry.at("content").is_string()) {
      continue;
    }
    const std::string content = entry.at("content").get<std::string>();
    if (std::find(kMediaTokens.begin(), kMediaTokens.end(), content) != kMediaTokens.end()) {
      ids.push_back(entry.at("id").get<std::int32_t>());
    }
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

struct ParsedSafetensorHeader {
  std::uint64_t data_base_offset = 0;
  std::uint64_t file_size = 0;
  std::unordered_map<std::string, Qwen38TensorInfo> tensors;
  Qwen38VerifiedFile file_info;
};

ParsedSafetensorHeader inspect_safetensors_shard_fd(const std::filesystem::path& path, const std::string& shard_name,
                                                    bool hash_payload) {
  ScopedFd fd(open_readonly_fd(path, "safetensors shard"));
  const struct stat st = fstat_regular(fd.get(), path);
  const std::uint64_t file_size = static_cast<std::uint64_t>(st.st_size);
  if (file_size < 8) {
    throw std::invalid_argument("safetensors shard truncated (< 8 bytes): " + path.string());
  }

  Sha256 hasher;
  std::array<std::uint8_t, 8> len_bytes{};
  read_exact_fd(fd.get(), len_bytes.data(), 8, path);
  if (hash_payload) hasher.update(len_bytes.data(), 8);

  std::uint64_t header_len = 0;
  for (int i = 0; i < 8; ++i) {
    header_len |= (static_cast<std::uint64_t>(len_bytes[i]) << (8 * i));
  }
  if (header_len == 0 || header_len > (64ull * 1024ull * 1024ull) || 8ull + header_len > file_size) {
    throw std::invalid_argument("invalid safetensors header length in " + path.string());
  }

  std::string header_json(header_len, '\0');
  read_exact_fd(fd.get(), header_json.data(), static_cast<std::size_t>(header_len), path);
  if (hash_payload) hasher.update(header_json.data(), header_json.size());

  nlohmann::json hdr;
  try {
    hdr = nlohmann::json::parse(header_json);
  } catch (const nlohmann::json::exception& e) {
    throw std::invalid_argument("invalid safetensors JSON header in " + path.string() + ": " + e.what());
  }
  if (!hdr.is_object()) {
    throw std::invalid_argument("safetensors header is not a JSON object: " + path.string());
  }

  ParsedSafetensorHeader parsed;
  parsed.data_base_offset = 8ull + header_len;
  parsed.file_size = file_size;
  const std::uint64_t payload_bytes = file_size - parsed.data_base_offset;

  for (auto it = hdr.begin(); it != hdr.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const auto& obj = it.value();
    if (!obj.is_object() || !obj.contains("dtype") || !obj.contains("shape") || !obj.contains("data_offsets")) {
      throw std::invalid_argument("malformed safetensors entry '" + it.key() + "' in " + path.string());
    }
    Qwen38TensorInfo info;
    info.name = it.key();
    info.shard_file = shard_name;
    info.dtype = obj.at("dtype").get<std::string>();
    for (const auto& d : obj.at("shape")) {
      info.shape.push_back(d.get<std::int64_t>());
    }
    const auto& offs = obj.at("data_offsets");
    if (!offs.is_array() || offs.size() != 2) {
      throw std::invalid_argument("invalid data_offsets for tensor '" + info.name + "'");
    }
    info.data_offset_begin = offs[0].get<std::uint64_t>();
    info.data_offset_end = offs[1].get<std::uint64_t>();
    if (info.data_offset_begin > info.data_offset_end || info.data_offset_end > payload_bytes) {
      throw std::invalid_argument("out-of-bounds data_offsets for tensor '" + info.name + "' in " + shard_name);
    }
    parsed.tensors.emplace(info.name, std::move(info));
  }

  parsed.file_info.filename = shard_name;
  parsed.file_info.dev = static_cast<std::uint64_t>(st.st_dev);
  parsed.file_info.ino = static_cast<std::uint64_t>(st.st_ino);
  parsed.file_info.byte_size = file_size;
  parsed.file_info.payload_hashed = hash_payload;

  if (hash_payload) {
    std::uint64_t total_read = parsed.data_base_offset;
    std::array<char, 65536> buf{};
    while (true) {
      const ssize_t got = ::read(fd.get(), buf.data(), buf.size());
      if (got < 0) {
        if (errno == EINTR) continue;
        throw std::invalid_argument("read error while hashing shard " + path.string() + ": " + std::strerror(errno));
      }
      if (got == 0) break;
      const std::uint64_t chunk = static_cast<std::uint64_t>(got);
      if (total_read + chunk > file_size) {
        throw std::invalid_argument("concurrent shard growth detected while hashing " + path.string());
      }
      hasher.update(buf.data(), static_cast<std::size_t>(chunk));
      total_read += chunk;
    }
    if (total_read != file_size) {
      throw std::invalid_argument("concurrent shard truncation detected while hashing " + path.string());
    }
    parsed.file_info.sha256 = hasher.finalize();
  }
  return parsed;
}

void hash_u32(Sha256& h, std::uint32_t v) {
  std::array<std::uint8_t, 4> b{static_cast<std::uint8_t>(v & 0xffu), static_cast<std::uint8_t>((v >> 8) & 0xffu),
                                static_cast<std::uint8_t>((v >> 16) & 0xffu),
                                static_cast<std::uint8_t>((v >> 24) & 0xffu)};
  h.update(b.data(), b.size());
}

void hash_i32(Sha256& h, std::int32_t v) { hash_u32(h, static_cast<std::uint32_t>(v)); }

void hash_f32_bits(Sha256& h, float v) {
  static_assert(sizeof(float) == sizeof(std::uint32_t));
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  hash_u32(h, bits);
}

void hash_str(Sha256& h, std::string_view s) {
  hash_u32(h, static_cast<std::uint32_t>(s.size()));
  if (!s.empty()) h.update(s.data(), s.size());
}

bool is_excluded_tensor(std::string_view name, bool& is_visual, bool& is_mtp) {
  is_visual = name.starts_with("model.visual.");
  is_mtp = name.starts_with("mtp.");
  return is_visual || is_mtp;
}

void assert_tensor(const std::unordered_map<std::string, Qwen38TensorInfo>& tensors, const std::string& name,
                   std::initializer_list<std::string_view> allowed_dtypes,
                   std::initializer_list<std::int64_t> expected_shape, std::unordered_set<std::string>& seen) {
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    throw std::invalid_argument("qwen38 manifest missing required tensor '" + name + "'");
  }
  const Qwen38TensorInfo& t = it->second;
  bool dtype_ok = false;
  for (std::string_view dt : allowed_dtypes) {
    if (t.dtype == dt) {
      dtype_ok = true;
      break;
    }
  }
  if (!dtype_ok) {
    throw std::invalid_argument("qwen38 tensor '" + name + "' has unexpected dtype '" + t.dtype + "'");
  }
  if (t.shape.size() != expected_shape.size() ||
      !std::equal(t.shape.begin(), t.shape.end(), expected_shape.begin())) {
    std::ostringstream ss;
    ss << "qwen38 tensor '" << name << "' has unexpected shape [";
    for (std::size_t i = 0; i < t.shape.size(); ++i) ss << (i ? "," : "") << t.shape[i];
    ss << "]";
    throw std::invalid_argument(ss.str());
  }
  seen.insert(name);
}

void assert_linear_proj(const std::unordered_map<std::string, Qwen38TensorInfo>& tensors, const std::string& prefix,
                        int out_f, int in_f, Qwen38LinearQuantKind quant, std::unordered_set<std::string>& seen) {
  if (quant == Qwen38LinearQuantKind::Bf16) {
    assert_tensor(tensors, prefix + ".weight", {"BF16"}, {out_f, in_f}, seen);
    return;
  }
  if (quant == Qwen38LinearQuantKind::Fp8E4M3) {
    assert_tensor(tensors, prefix + ".weight", {"F8_E4M3"}, {out_f, in_f}, seen);
    assert_tensor(tensors, prefix + ".weight_scale", {"F32"}, {}, seen);
    assert_tensor(tensors, prefix + ".input_scale", {"F32"}, {}, seen);
    return;
  }
  if (in_f % 16 != 0) {
    throw std::invalid_argument("NVFP4 tensor '" + prefix + "' requires in_features divisible by 16");
  }
  assert_tensor(tensors, prefix + ".weight", {"U8"}, {out_f, in_f / 2}, seen);
  assert_tensor(tensors, prefix + ".weight_scale", {"F8_E4M3"}, {out_f, in_f / 16}, seen);
  assert_tensor(tensors, prefix + ".weight_scale_2", {"F32"}, {}, seen);
  assert_tensor(tensors, prefix + ".input_scale", {"F32"}, {}, seen);
}

}  // namespace

std::array<std::uint8_t, 32> sha256_bytes_digest(const void* data, std::size_t size) {
  Sha256 h;
  h.update(data, size);
  return h.finalize();
}

std::string sha256_bytes_hex(const void* data, std::size_t size) { return to_hex(sha256_bytes_digest(data, size)); }

std::array<std::uint8_t, 32> sha256_file_digest(const std::filesystem::path& path) {
  ScopedFd fd(open_readonly_fd(path, "file for sha256"));
  const struct stat st = fstat_regular(fd.get(), path);
  const std::uint64_t fsize = static_cast<std::uint64_t>(st.st_size);
  Sha256 h;
  std::uint64_t total_read = 0;
  std::array<char, 65536> buf{};
  while (true) {
    const ssize_t got = ::read(fd.get(), buf.data(), buf.size());
    if (got < 0) {
      if (errno == EINTR) continue;
      throw std::invalid_argument("read error while hashing file: " + path.string() + " (" + std::strerror(errno) + ")");
    }
    if (got == 0) break;
    const std::uint64_t chunk = static_cast<std::uint64_t>(got);
    if (total_read + chunk > fsize) {
      throw std::invalid_argument("concurrent file growth detected while hashing " + path.string());
    }
    h.update(buf.data(), static_cast<std::size_t>(chunk));
    total_read += chunk;
  }
  if (total_read != fsize) {
    throw std::invalid_argument("concurrent file truncation detected while hashing " + path.string());
  }
  return h.finalize();
}

std::string sha256_file_hex_streaming(const std::filesystem::path& path) {
  return to_hex(sha256_file_digest(path));
}

std::array<std::uint8_t, 32> qwen38_config_fingerprint(const Qwen38Config& c) {
  c.validate();
  Sha256 h;
  static constexpr std::string_view kTag = "qwen38_cfg_v1";
  hash_str(h, kTag);
  hash_str(h, c.model_type);
  hash_u32(h, static_cast<std::uint32_t>(c.architectures.size()));
  for (const auto& arch : c.architectures) hash_str(h, arch);
  hash_i32(h, c.vocab_size);
  hash_i32(h, c.hidden_size);
  hash_i32(h, c.intermediate_size);
  hash_i32(h, c.num_hidden_layers);
  hash_i32(h, c.max_position_embeddings);
  hash_f32_bits(h, c.rms_norm_eps);
  hash_u32(h, c.tie_word_embeddings ? 1u : 0u);
  hash_i32(h, c.full_attention_interval);
  hash_i32(h, c.num_attention_heads);
  hash_i32(h, c.num_key_value_heads);
  hash_i32(h, c.head_dim);
  hash_u32(h, c.attention_bias ? 1u : 0u);
  hash_u32(h, c.attn_output_gate ? 1u : 0u);
  hash_f32_bits(h, c.rope_theta);
  hash_f32_bits(h, c.partial_rotary_factor);
  hash_i32(h, c.rotary_dim);
  for (int m : c.mrope_section) hash_i32(h, m);
  hash_u32(h, c.mrope_interleaved ? 1u : 0u);
  hash_i32(h, c.linear_conv_kernel_dim);
  hash_i32(h, c.linear_key_head_dim);
  hash_i32(h, c.linear_value_head_dim);
  hash_i32(h, c.linear_num_key_heads);
  hash_i32(h, c.linear_num_value_heads);
  hash_str(h, c.mamba_ssm_dtype);
  hash_u32(h, static_cast<std::uint32_t>(c.layer_types.size()));
  for (Qwen38LayerKind lk : c.layer_types) hash_u32(h, static_cast<std::uint32_t>(lk));
  hash_u32(h, static_cast<std::uint32_t>(c.weight_precision));
  hash_str(h, c.quant_method);
  hash_str(h, c.quant_algo);
  hash_i32(h, c.nvfp4_group_size);
  return h.finalize();
}

std::string qwen38_config_fingerprint_hex(const Qwen38Config& config) {
  return to_hex(qwen38_config_fingerprint(config));
}

Qwen38StreamResult Qwen38Manifest::stream_verified_open_fd(
    int fd, std::string_view filename,
    const std::function<void(std::uint64_t offset, std::span<const std::uint8_t> chunk)>& on_chunk,
    const std::function<void(const Qwen38StreamResult&)>& on_admit,
    const std::function<void()>& on_abort,
    std::size_t chunk_bytes) const {
  static_assert(sizeof(off_t) >= 8, "64-bit off_t required for >2 GiB shard streaming");
  constexpr std::size_t kMaxChunkBytes = 16ull * 1024ull * 1024ull;

  Qwen38StreamResult res;
  try {
    if (chunk_bytes == 0 || chunk_bytes > kMaxChunkBytes) {
      throw std::invalid_argument("bounded streaming chunk_bytes must be in (0, 16 MiB]");
    }
    if (checkpoint.verification_level != ingine::model::VerificationLevel::PinnedVerified ||
        !checkpoint.shard_digests_verified) {
      throw std::invalid_argument("verified stream consumption requires VerificationLevel::PinnedVerified manifest");
    }
    const auto it = verified_files.find(std::string(filename));
    if (it == verified_files.end()) {
      throw std::invalid_argument("file not present in verified manifest inventory: " + std::string(filename));
    }
    const Qwen38VerifiedFile& v = it->second;
    if (!v.payload_hashed || is_zero_digest(v.sha256)) {
      throw std::invalid_argument("file lacks non-zero verified payload pin: " + std::string(filename));
    }
    if (v.byte_size > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      throw std::invalid_argument("verified file size overflows off_t for '" + std::string(filename) + "'");
    }

    const struct stat st_pre = fstat_regular(fd, filename);
    if (static_cast<std::uint64_t>(st_pre.st_dev) != v.dev || static_cast<std::uint64_t>(st_pre.st_ino) != v.ino ||
        static_cast<std::uint64_t>(st_pre.st_size) != v.byte_size) {
      throw std::invalid_argument("open FD identity mismatch (file replaced or resized since inspection) for '" +
                                  std::string(filename) + "'");
    }

    std::vector<std::uint8_t> buf(chunk_bytes);
    Sha256 h;
    std::uint64_t offset = 0;
    while (true) {
      const ssize_t got = ::pread(fd, buf.data(), buf.size(), static_cast<off_t>(offset));
      if (got < 0) {
        if (errno == EINTR) continue;
        throw std::invalid_argument("pread error while streaming '" + std::string(filename) +
                                    "': " + std::strerror(errno));
      }
      if (got == 0) break;
      const std::uint64_t chunk = static_cast<std::uint64_t>(got);
      if (chunk > v.byte_size - offset) {
        throw std::invalid_argument("concurrent file growth detected while streaming '" + std::string(filename) + "'");
      }
      h.update(buf.data(), static_cast<std::size_t>(chunk));
      if (on_chunk) {
        on_chunk(offset, std::span<const std::uint8_t>(buf.data(), static_cast<std::size_t>(chunk)));
      }
      offset += chunk;
    }

    const struct stat st_post = fstat_regular(fd, filename);
    if (static_cast<std::uint64_t>(st_post.st_dev) != v.dev ||
        static_cast<std::uint64_t>(st_post.st_ino) != v.ino ||
        static_cast<std::uint64_t>(st_post.st_size) != v.byte_size || offset != v.byte_size) {
      throw std::invalid_argument("concurrent file modification/truncation during verified stream for '" +
                                  std::string(filename) + "'");
    }

    const std::array<std::uint8_t, 32> digest = h.finalize();
    if (digest != v.sha256) {
      throw std::invalid_argument("SHA-256 mismatch on streamed consumption (in-place mutation or corruption) for '" +
                                  std::string(filename) + "'");
    }

    res.bytes_delivered = offset;
    res.sha256 = digest;
    res.admitted = true;
    if (on_admit) {
      on_admit(res);
    }
    return res;
  } catch (...) {
    if (on_abort) {
      try {
        on_abort();
      } catch (...) {
      }
    }
    throw;
  }
}

Qwen38StreamResult Qwen38Manifest::stream_verified_file(
    const std::filesystem::path& model_dir, std::string_view filename,
    const std::function<void(std::uint64_t offset, std::span<const std::uint8_t> chunk)>& on_chunk,
    const std::function<void(const Qwen38StreamResult&)>& on_admit,
    const std::function<void()>& on_abort,
    std::size_t chunk_bytes) const {
  int raw_fd = -1;
  try {
    raw_fd = open_readonly_fd(model_dir / std::string(filename), filename.data());
  } catch (...) {
    if (on_abort) {
      try {
        on_abort();
      } catch (...) {
      }
    }
    throw;
  }
  ScopedFd fd(raw_fd);
  return stream_verified_open_fd(fd.get(), filename, on_chunk, on_admit, on_abort, chunk_bytes);
}

void Qwen38Manifest::verify_open_fd(int fd, std::string_view filename, std::size_t chunk_bytes) const {
  (void)stream_verified_open_fd(fd, filename, {}, {}, {}, chunk_bytes);
}

std::string Qwen38Manifest::read_verified_file_bytes(const std::filesystem::path& model_dir,
                                                     std::string_view filename) const {
  constexpr std::uint64_t kMaxMetadataBytes = 256ull * 1024ull * 1024ull;
  const auto it = verified_files.find(std::string(filename));
  if (it == verified_files.end()) {
    throw std::invalid_argument("file not present in verified manifest inventory: " + std::string(filename));
  }
  if (it->second.byte_size > kMaxMetadataBytes) {
    throw std::invalid_argument("file '" + std::string(filename) + "' (" + std::to_string(it->second.byte_size) +
                                " bytes) exceeds 256 MiB metadata buffer cap; use stream_verified_file()");
  }
  std::string staged;
  staged.reserve(static_cast<std::size_t>(it->second.byte_size));
  std::string admitted_bytes;
  (void)stream_verified_file(
      model_dir, filename,
      [&](std::uint64_t /*offset*/, std::span<const std::uint8_t> chunk) {
        if (staged.size() + chunk.size() > kMaxMetadataBytes) {
          throw std::invalid_argument("metadata file grew beyond 256 MiB during verified read: " +
                                      std::string(filename));
        }
        staged.append(reinterpret_cast<const char*>(chunk.data()), chunk.size());
      },
      [&](const Qwen38StreamResult& /*res*/) { admitted_bytes = std::move(staged); },
      [&]() {
        staged.clear();
        admitted_bytes.clear();
      });
  return admitted_bytes;
}

Qwen38Manifest Qwen38Manifest::inspect(const std::filesystem::path& model_dir, bool hash_shards,
                                        const std::string& revision) {
  if (!std::filesystem::is_directory(model_dir)) {
    throw std::invalid_argument("qwen38 checkpoint directory does not exist: " + model_dir.string());
  }
  const std::filesystem::path cfg_path = model_dir / "config.json";
  const std::filesystem::path tok_path = model_dir / "tokenizer.json";
  const std::filesystem::path tmpl_path = model_dir / "chat_template.jinja";

  // Single-pass open-FD read+hash for metadata files (never parse one path and reopen to hash).
  const VerifiedSmallFile cfg_file = *read_and_hash_small_file_fd(cfg_path, "config.json", false);
  const VerifiedSmallFile tok_file = *read_and_hash_small_file_fd(tok_path, "tokenizer.json", false);
  const auto tmpl_file = read_and_hash_small_file_fd(tmpl_path, "chat_template.jinja", true);

  Qwen38Manifest manifest;
  manifest.config = Qwen38Config::from_json_str(cfg_file.bytes);
  manifest.parsed_config_fingerprint = qwen38_config_fingerprint(manifest.config);
  manifest.checkpoint.format = ingine::model::CheckpointFormat::Safetensors;
  manifest.checkpoint.verification_level =
      hash_shards ? ingine::model::VerificationLevel::SelfHashed : ingine::model::VerificationLevel::HeadersOnly;
  manifest.checkpoint.schema_variant =
      manifest.config.weight_precision == Qwen38WeightPrecision::MixedFp8Nvfp4 ? "qwen3_5_mixed_fp8_nvfp4"
                                                                              : "qwen3_5_bf16";
  manifest.checkpoint.model_id = model_dir.filename().string();
  manifest.checkpoint.revision = revision;
  manifest.checkpoint.config_sha256 = cfg_file.identity.sha256;
  manifest.checkpoint.tokenizer_sha256 = tok_file.identity.sha256;
  manifest.media_token_ids = collect_media_token_ids_from_tokenizer(tok_file.bytes);
  manifest.verified_files.emplace("config.json", cfg_file.identity);
  manifest.verified_files.emplace("tokenizer.json", tok_file.identity);
  if (tmpl_file.has_value()) {
    manifest.checkpoint.chat_template_sha256 = tmpl_file->identity.sha256;
    manifest.verified_files.emplace("chat_template.jinja", tmpl_file->identity);
  }

  const std::filesystem::path index_path = model_dir / "model.safetensors.index.json";
  const auto idx_file = read_and_hash_small_file_fd(index_path, "model.safetensors.index.json", true);

  std::vector<std::pair<std::string, std::string>> index_entries;  // (tensor_name, shard_file)
  std::unordered_set<std::string> shard_set;
  if (idx_file.has_value()) {
    manifest.verified_files.emplace("model.safetensors.index.json", idx_file->identity);
    nlohmann::json idx;
    try {
      idx = nlohmann::json::parse(idx_file->bytes);
    } catch (const nlohmann::json::exception& e) {
      throw std::invalid_argument("invalid model.safetensors.index.json: " + std::string(e.what()));
    }
    if (!idx.is_object() || !idx.contains("weight_map") || !idx.at("weight_map").is_object()) {
      throw std::invalid_argument("model.safetensors.index.json missing weight_map object");
    }
    const auto& wm = idx.at("weight_map");
    for (auto it = wm.begin(); it != wm.end(); ++it) {
      if (!it.value().is_string()) {
        throw std::invalid_argument("weight_map value must be string for tensor '" + it.key() + "'");
      }
      const std::string shard = it.value().get<std::string>();
      index_entries.emplace_back(it.key(), shard);
      shard_set.insert(shard);
    }
    manifest.shard_files.assign(shard_set.begin(), shard_set.end());
    std::sort(manifest.shard_files.begin(), manifest.shard_files.end());
  } else if (std::filesystem::is_regular_file(model_dir / "model.safetensors")) {
    manifest.shard_files.push_back("model.safetensors");
  } else {
    throw std::invalid_argument("neither model.safetensors.index.json nor model.safetensors found in " +
                                model_dir.string());
  }

  std::unordered_map<std::string, ParsedSafetensorHeader> parsed_shards;
  Sha256 shard_set_hash;
  for (const std::string& shard : manifest.shard_files) {
    const std::filesystem::path sp = model_dir / shard;
    auto hdr = inspect_safetensors_shard_fd(sp, shard, hash_shards);
    manifest.shard_sizes.emplace(shard, hdr.file_size);
    std::string digest_note = "header_verified";
    if (hash_shards) {
      digest_note = to_hex(hdr.file_info.sha256);
      manifest.shard_sha256.emplace(shard, digest_note);
    }
    manifest.verified_files.emplace(shard, hdr.file_info);
    const std::string line = shard + '|' + std::to_string(hdr.file_size) + '|' + digest_note + '\n';
    shard_set_hash.update(line.data(), line.size());
    parsed_shards.emplace(shard, std::move(hdr));
  }
  manifest.checkpoint.shard_set_sha256 = shard_set_hash.finalize();
  manifest.shard_payloads_verified = hash_shards;

  // Reconcile index weight_map against shard headers, partitioning active text vs visual/MTP.
  if (!index_entries.empty()) {
    manifest.total_index_tensors = index_entries.size();
    for (const auto& [name, shard] : index_entries) {
      const auto& hdr = parsed_shards.at(shard);
      auto tit = hdr.tensors.find(name);
      if (tit == hdr.tensors.end()) {
        throw std::invalid_argument("index references tensor '" + name + "' missing from shard '" + shard + "'");
      }
      bool vis = false;
      bool mtp = false;
      if (is_excluded_tensor(name, vis, mtp)) {
        if (vis) {
          ++manifest.excluded_visual_tensor_count;
          manifest.memory_budget.excluded_visual_bytes += tit->second.byte_size();
        } else if (mtp) {
          ++manifest.excluded_mtp_tensor_count;
          manifest.memory_budget.excluded_mtp_bytes += tit->second.byte_size();
        }
        continue;
      }
      manifest.memory_budget.text_weight_bytes += tit->second.byte_size();
      manifest.active_tensors.emplace(name, tit->second);
    }
  } else {
    const auto& hdr = parsed_shards.at("model.safetensors");
    manifest.total_index_tensors = hdr.tensors.size();
    for (const auto& [name, info] : hdr.tensors) {
      bool vis = false;
      bool mtp = false;
      if (is_excluded_tensor(name, vis, mtp)) {
        if (vis) {
          ++manifest.excluded_visual_tensor_count;
          manifest.memory_budget.excluded_visual_bytes += info.byte_size();
        } else if (mtp) {
          ++manifest.excluded_mtp_tensor_count;
          manifest.memory_budget.excluded_mtp_bytes += info.byte_size();
        }
        continue;
      }
      manifest.memory_budget.text_weight_bytes += info.byte_size();
      manifest.active_tensors.emplace(name, info);
    }
  }
  manifest.active_text_tensor_count = manifest.active_tensors.size();

  // Verify exact expected text tensor inventory against Qwen38Config.
  const Qwen38Config& c = manifest.config;
  std::unordered_set<std::string> seen;
  seen.reserve(manifest.active_tensors.size());

  assert_tensor(manifest.active_tensors, "model.language_model.embed_tokens.weight", {"BF16"},
                {c.vocab_size, c.hidden_size}, seen);
  assert_tensor(manifest.active_tensors, "model.language_model.norm.weight", {"BF16"}, {c.hidden_size}, seen);
  if (!c.tie_word_embeddings) {
    assert_linear_proj(manifest.active_tensors, "lm_head", c.vocab_size, c.hidden_size, c.lm_head_quant(), seen);
  }

  for (int i = 0; i < c.num_hidden_layers; ++i) {
    const std::string lp = "model.language_model.layers." + std::to_string(i);
    assert_tensor(manifest.active_tensors, lp + ".input_layernorm.weight", {"BF16"}, {c.hidden_size}, seen);
    assert_tensor(manifest.active_tensors, lp + ".post_attention_layernorm.weight", {"BF16"}, {c.hidden_size}, seen);

    if (c.layer_types[static_cast<std::size_t>(i)] == Qwen38LayerKind::LinearAttention) {
      const std::string attn = lp + ".linear_attn";
      assert_tensor(manifest.active_tensors, attn + ".A_log", {"BF16", "F32"}, {c.linear_num_value_heads}, seen);
      assert_tensor(manifest.active_tensors, attn + ".dt_bias", {"BF16", "F32"}, {c.linear_num_value_heads}, seen);
      assert_tensor(manifest.active_tensors, attn + ".conv1d.weight", {"BF16", "F32"},
                    {c.linear_conv_dim(), 1, c.linear_conv_kernel_dim}, seen);
      assert_tensor(manifest.active_tensors, attn + ".in_proj_a.weight", {"BF16", "F32"},
                    {c.linear_num_value_heads, c.hidden_size}, seen);
      assert_tensor(manifest.active_tensors, attn + ".in_proj_b.weight", {"BF16", "F32"},
                    {c.linear_num_value_heads, c.hidden_size}, seen);
      assert_tensor(manifest.active_tensors, attn + ".norm.weight", {"BF16", "F32"}, {c.linear_value_head_dim}, seen);

      assert_linear_proj(manifest.active_tensors, attn + ".in_proj_qkv", c.linear_conv_dim(), c.hidden_size,
                         c.linear_attn_quant(), seen);
      assert_linear_proj(manifest.active_tensors, attn + ".in_proj_z", c.linear_value_dim(), c.hidden_size,
                         c.linear_attn_quant(), seen);
      assert_linear_proj(manifest.active_tensors, attn + ".out_proj", c.hidden_size, c.linear_value_dim(),
                         c.linear_attn_quant(), seen);
    } else {
      const std::string attn = lp + ".self_attn";
      assert_tensor(manifest.active_tensors, attn + ".q_norm.weight", {"BF16", "F32"}, {c.head_dim}, seen);
      assert_tensor(manifest.active_tensors, attn + ".k_norm.weight", {"BF16", "F32"}, {c.head_dim}, seen);

      assert_linear_proj(manifest.active_tensors, attn + ".q_proj", c.full_q_proj_out_dim(), c.hidden_size,
                         c.full_attn_quant(), seen);
      assert_linear_proj(manifest.active_tensors, attn + ".k_proj", c.full_kv_proj_out_dim(), c.hidden_size,
                         c.full_attn_quant(), seen);
      assert_linear_proj(manifest.active_tensors, attn + ".v_proj", c.full_kv_proj_out_dim(), c.hidden_size,
                         c.full_attn_quant(), seen);
      assert_linear_proj(manifest.active_tensors, attn + ".o_proj", c.hidden_size, c.num_attention_heads * c.head_dim,
                         c.full_attn_quant(), seen);
    }

    assert_linear_proj(manifest.active_tensors, lp + ".mlp.gate_proj", c.intermediate_size, c.hidden_size,
                       c.mlp_quant(), seen);
    assert_linear_proj(manifest.active_tensors, lp + ".mlp.up_proj", c.intermediate_size, c.hidden_size,
                       c.mlp_quant(), seen);
    assert_linear_proj(manifest.active_tensors, lp + ".mlp.down_proj", c.hidden_size, c.intermediate_size,
                       c.mlp_quant(), seen);
  }

  if (seen.size() != manifest.active_tensors.size()) {
    for (const auto& [name, _] : manifest.active_tensors) {
      if (seen.find(name) == seen.end()) {
        throw std::invalid_argument("unexpected active text tensor in qwen38 manifest: '" + name + "'");
      }
    }
  }

  // Canonical digest over active tensor inventory (deterministic regardless of JSON iteration order).
  std::vector<std::string> sorted_names;
  sorted_names.reserve(manifest.active_tensors.size());
  for (const auto& [name, _] : manifest.active_tensors) sorted_names.push_back(name);
  std::sort(sorted_names.begin(), sorted_names.end());

  Sha256 mh;
  for (const std::string& name : sorted_names) {
    const Qwen38TensorInfo& t = manifest.active_tensors.at(name);
    std::ostringstream line;
    line << name << '|' << t.shard_file << '|' << t.dtype << '|';
    for (std::size_t i = 0; i < t.shape.size(); ++i) line << (i ? "," : "") << t.shape[i];
    line << '|' << t.data_offset_begin << ':' << t.data_offset_end << '\n';
    const std::string s = line.str();
    mh.update(s.data(), s.size());
  }
  manifest.checkpoint.weight_manifest_sha256 = mh.finalize();
  manifest.checkpoint.weight_manifest_verified = true;
  manifest.checkpoint.shard_count = static_cast<std::uint32_t>(manifest.shard_files.size());
  // Self-computed hashes do NOT verify expected pins; only inspect_pinned() sets shard_digests_verified=true.
  manifest.checkpoint.shard_digests_verified = false;

  manifest.memory_budget.gdn_recurrent_bytes =
      static_cast<std::uint64_t>(c.num_linear_layers) * static_cast<std::uint64_t>(c.linear_num_value_heads) *
      static_cast<std::uint64_t>(c.linear_key_head_dim) * static_cast<std::uint64_t>(c.linear_value_head_dim) * 4ull;
  manifest.memory_budget.gdn_conv_bytes =
      static_cast<std::uint64_t>(c.num_linear_layers) * static_cast<std::uint64_t>(c.linear_conv_dim()) *
      static_cast<std::uint64_t>(c.linear_conv_kernel_dim) * 2ull;
  manifest.memory_budget.kv_bytes_per_token =
      static_cast<std::uint64_t>(c.num_full_layers) * 2ull * static_cast<std::uint64_t>(c.num_key_value_heads) *
      static_cast<std::uint64_t>(c.head_dim) * 2ull;

  return manifest;
}

Qwen38Manifest Qwen38Manifest::inspect_pinned(const std::filesystem::path& model_dir,
                                               const ingine::model::CheckpointPinSpec& expected_pins,
                                               const std::string& revision) {
  if (expected_pins.shards.empty()) {
    throw std::invalid_argument("inspect_pinned requires non-empty expected shard pins");
  }
  Qwen38Manifest m = inspect(model_dir, true, revision);
  if (expected_pins.config_sha256.has_value() && m.checkpoint.config_sha256 != *expected_pins.config_sha256) {
    throw std::invalid_argument("config.json SHA-256 mismatch against expected pin");
  }
  if (expected_pins.tokenizer_sha256.has_value() && m.checkpoint.tokenizer_sha256 != *expected_pins.tokenizer_sha256) {
    throw std::invalid_argument("tokenizer.json SHA-256 mismatch against expected pin");
  }
  if (expected_pins.chat_template_sha256.has_value() &&
      m.checkpoint.chat_template_sha256 != *expected_pins.chat_template_sha256) {
    throw std::invalid_argument("chat_template.jinja SHA-256 mismatch against expected pin");
  }
  if (expected_pins.shards.size() != m.shard_files.size()) {
    throw std::invalid_argument("shard count mismatch against expected pin set");
  }

  std::unordered_map<std::string, const ingine::model::ExpectedShardDigest*> pin_by_name;
  for (const auto& sp : expected_pins.shards) {
    bool zero = true;
    for (std::uint8_t b : sp.sha256) {
      if (b != 0) {
        zero = false;
        break;
      }
    }
    if (sp.filename.empty() || sp.byte_size == 0 || zero) {
      throw std::invalid_argument("expected shard pin requires non-empty filename, positive size, and non-zero sha256");
    }
    if (!pin_by_name.emplace(sp.filename, &sp).second) {
      throw std::invalid_argument("duplicate expected shard filename in pin spec: " + sp.filename);
    }
  }
  for (const std::string& shard : m.shard_files) {
    const auto it = pin_by_name.find(shard);
    if (it == pin_by_name.end()) {
      throw std::invalid_argument("unexpected shard missing from pin spec: " + shard);
    }
    const auto& exp = *it->second;
    if (m.shard_sizes.at(shard) != exp.byte_size) {
      throw std::invalid_argument("shard byte size mismatch for '" + shard + "'");
    }
    if (m.shard_sha256.at(shard) != to_hex(exp.sha256)) {
      throw std::invalid_argument("shard SHA-256 mismatch for '" + shard + "' (corrupted or unpinned weights)");
    }
  }
  m.checkpoint.verification_level = ingine::model::VerificationLevel::PinnedVerified;
  m.checkpoint.shard_digests_verified = true;
  return m;
}

ingine::model::CheckpointPinSpec qwen38_27b_nvfp4_pin_spec() {
  ingine::model::CheckpointPinSpec spec;
  spec.config_sha256 =
      ingine::model::parse_sha256_hex("71fa01bb64de20971045cd82a82c0ebb25a54e1b0a3f09ef23cf5479e9c54df8");
  spec.tokenizer_sha256 =
      ingine::model::parse_sha256_hex("0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3");
  spec.chat_template_sha256 =
      ingine::model::parse_sha256_hex("c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041");
  spec.shards = {
      {"model-00001-of-00003.safetensors", 9965652544ull,
       ingine::model::parse_sha256_hex("7d0fd155118901373eb0fd13ed3aae68f95be747bc205be72ebc41739c25ee80")},
      {"model-00002-of-00003.safetensors", 9985757064ull,
       ingine::model::parse_sha256_hex("98a7e9486baa860c792c9463a770cb9d017696bdd17606300a5b9334149f4c27")},
      {"model-00003-of-00003.safetensors", 1970287672ull,
       ingine::model::parse_sha256_hex("0506ad35dc21469708e7813bd76c592cef08bd0317b4a1fe899745ebe2435271")},
  };
  return spec;
}

}  // namespace ingine::backends::qwen38

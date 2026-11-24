#include "model/kv_checkpoint.h"

#include "core/act.h"
#include "core/device.h"
#include "model/paged_kv.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::model {
namespace {

std::atomic<bool> g_sigterm_requested{false};

void kv_ckpt_sigterm_handler(int) { g_sigterm_requested.store(true, std::memory_order_relaxed); }

// ---- Self-contained FIPS 180-4 SHA-256 ----
struct Sha256 {
  std::uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::uint64_t total_bits = 0;
  std::uint8_t buf[64]{};
  std::size_t buf_len = 0;

  static constexpr std::uint32_t rotr(std::uint32_t x, unsigned n) noexcept { return (x >> n) | (x << (32u - n)); }

  void transform(const std::uint8_t block[64]) noexcept {
    static constexpr std::uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (std::uint32_t(block[4 * i]) << 24) | (std::uint32_t(block[4 * i + 1]) << 16) |
             (std::uint32_t(block[4 * i + 2]) << 8) | std::uint32_t(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t t1 = hh + S1 + ch + k[i] + w[i];
      const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = S0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }

  void update(const void* data, std::size_t len) noexcept {
    const auto* p = static_cast<const std::uint8_t*>(data);
    total_bits += std::uint64_t(len) * 8u;
    if (buf_len > 0) {
      const std::size_t take = std::min(len, std::size_t(64) - buf_len);
      std::memcpy(buf + buf_len, p, take);
      buf_len += take;
      p += take;
      len -= take;
      if (buf_len == 64) {
        transform(buf);
        buf_len = 0;
      }
    }
    while (len >= 64) {
      transform(p);
      p += 64;
      len -= 64;
    }
    if (len > 0) {
      std::memcpy(buf, p, len);
      buf_len = len;
    }
  }

  std::array<std::uint8_t, 32> finish() noexcept {
    buf[buf_len++] = 0x80u;
    if (buf_len > 56) {
      std::memset(buf + buf_len, 0, 64 - buf_len);
      transform(buf);
      buf_len = 0;
    }
    std::memset(buf + buf_len, 0, 56 - buf_len);
    for (int i = 0; i < 8; ++i) buf[56 + i] = static_cast<std::uint8_t>((total_bits >> (56 - 8 * i)) & 0xffu);
    transform(buf);
    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
      out[4 * i] = static_cast<std::uint8_t>((h[i] >> 24) & 0xffu);
      out[4 * i + 1] = static_cast<std::uint8_t>((h[i] >> 16) & 0xffu);
      out[4 * i + 2] = static_cast<std::uint8_t>((h[i] >> 8) & 0xffu);
      out[4 * i + 3] = static_cast<std::uint8_t>(h[i] & 0xffu);
    }
    return out;
  }
};

constexpr char kMagic[8] = {'I', 'N', 'G', 'C', 'K', 'P', 'T', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kHeaderBytes = 4096;

struct alignas(8) RawHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t header_bytes;
  std::uint32_t act_dtype;
  std::uint32_t kv_format;
  std::int32_t max_ctx_at_save;
  std::int32_t pos;
  std::int32_t num_full_visits;
  std::int32_t num_sliding_visits;
  std::int32_t window;
  std::int32_t num_visits;
  std::int32_t conv_k;
  std::uint32_t has_logits;
  std::uint64_t row_bytes;
  std::uint64_t swa_bytes;
  std::uint64_t full_kv_bytes;
  std::uint8_t prompt_sha256[32];
  std::uint8_t model_sha256[32];
};

static_assert(sizeof(RawHeader) <= kHeaderBytes, "RawHeader exceeds 4 KiB page");

std::uint32_t state_kv_format(const State& st) noexcept {
  if (st.kv_q4) return 2u;
  if (st.kv_fp8) return 1u;
  return 0u;
}

std::uint64_t state_row_bytes(const State& st) noexcept {
  if (st.kv_q4) return std::uint64_t(KV_Q4_ROW_BYTES);
  if (st.kv_fp8) return std::uint64_t(KV_DIM);
  return std::uint64_t(KV_DIM) * 2u;
}

std::array<const Tensor*, 7> swa_tensors(const State& st) noexcept {
  return {&st.k_ring, &st.v_ring, &st.ring_pos, &st.conv_k, &st.conv_v, &st.conv_attn, &st.conv_mlp};
}

std::array<Tensor*, 7> swa_tensors_mut(State& st) noexcept {
  return {&st.k_ring, &st.v_ring, &st.ring_pos, &st.conv_k, &st.conv_v, &st.conv_attn, &st.conv_mlp};
}

std::uint64_t swa_total_bytes(const State& st) noexcept {
  std::uint64_t n = 0;
  for (const Tensor* t : swa_tensors(st)) n += t->bytes();
  return n;
}

void write_all(int fd, const void* buf, std::size_t n) {
  const auto* p = static_cast<const std::uint8_t*>(buf);
  while (n > 0) {
    const ssize_t w = ::write(fd, p, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("kv_checkpoint: write failed: " + std::string(std::strerror(errno)));
    }
    p += std::size_t(w);
    n -= std::size_t(w);
  }
}

void pread_all(int fd, void* buf, std::size_t n, std::uint64_t off) {
  auto* p = static_cast<std::uint8_t*>(buf);
  while (n > 0) {
    const ssize_t r = ::pread(fd, p, n, static_cast<off_t>(off));
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("kv_checkpoint: pread failed: " + std::string(std::strerror(errno)));
    }
    if (r == 0) throw std::runtime_error("kv_checkpoint: unexpected EOF reading checkpoint file");
    p += std::size_t(r);
    off += std::uint64_t(r);
    n -= std::size_t(r);
  }
}

std::string hex_prefix(const std::uint8_t h[32]) {
  char out[17];
  for (int i = 0; i < 8; ++i) std::snprintf(out + 2 * i, 3, "%02x", unsigned(h[i]));
  out[16] = '\0';
  return std::string(out);
}

void fsync_parent_dir(const std::filesystem::path& path) {
  std::filesystem::path dir = path.parent_path();
  if (dir.empty()) dir = ".";
  const int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
  if (dfd >= 0) {
    (void)::fsync(dfd);
    ::close(dfd);
  }
}

RawHeader read_and_validate_raw_header(int fd, std::uint64_t file_size) {
  if (file_size < kHeaderBytes) {
    throw std::runtime_error("kv_checkpoint: file smaller than 4096-byte header");
  }
  std::uint8_t page[kHeaderBytes]{};
  pread_all(fd, page, kHeaderBytes, 0);
  RawHeader hdr{};
  std::memcpy(&hdr, page, sizeof(RawHeader));
  if (std::memcmp(hdr.magic, kMagic, 8) != 0) {
    throw std::runtime_error("kv_checkpoint: invalid magic (not an INGCKPT1 file)");
  }
  if (hdr.version != kVersion || hdr.header_bytes != kHeaderBytes) {
    throw std::runtime_error("kv_checkpoint: unsupported version " + std::to_string(hdr.version));
  }
  const std::uint64_t logits_bytes = hdr.has_logits ? std::uint64_t(UNPADDED_VOCAB) * sizeof(float) : 0u;
  const std::uint64_t expected = std::uint64_t(hdr.header_bytes) + hdr.swa_bytes + hdr.full_kv_bytes + logits_bytes;
  if (file_size != expected) {
    throw std::runtime_error("kv_checkpoint: truncated or corrupted file (got " + std::to_string(file_size) +
                             " bytes, expected " + std::to_string(expected) + ")");
  }
  return hdr;
}

}  // namespace

void install_kv_checkpoint_sigterm_handler() noexcept { std::signal(SIGTERM, kv_ckpt_sigterm_handler); }
bool kv_checkpoint_sigterm_requested() noexcept { return g_sigterm_requested.load(std::memory_order_relaxed); }
void clear_kv_checkpoint_sigterm() noexcept { g_sigterm_requested.store(false, std::memory_order_relaxed); }

std::array<std::uint8_t, 32> sha256_bytes(const void* data, std::size_t len) noexcept {
  Sha256 ctx;
  ctx.update(data, len);
  return ctx.finish();
}

std::array<std::uint8_t, 32> sha256_file_or_zero(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return {};
  Sha256 ctx;
  char buf[65536];
  while (in.good()) {
    in.read(buf, sizeof(buf));
    const std::streamsize n = in.gcount();
    if (n > 0) ctx.update(buf, std::size_t(n));
  }
  return ctx.finish();
}

KvCheckpointInfo inspect_kv_checkpoint(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("kv_checkpoint: cannot open " + path.string() + ": " + std::strerror(errno));
  }
  struct stat st_buf{};
  if (::fstat(fd, &st_buf) != 0) {
    ::close(fd);
    throw std::runtime_error("kv_checkpoint: fstat failed for " + path.string());
  }
  RawHeader hdr{};
  try {
    hdr = read_and_validate_raw_header(fd, std::uint64_t(st_buf.st_size));
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);
  KvCheckpointInfo info;
  info.pos = hdr.pos;
  info.max_ctx_at_save = hdr.max_ctx_at_save;
  info.act_dtype = hdr.act_dtype;
  info.kv_format = hdr.kv_format;
  info.has_logits = hdr.has_logits != 0;
  std::memcpy(info.prompt_sha256.data(), hdr.prompt_sha256, 32);
  std::memcpy(info.model_sha256.data(), hdr.model_sha256, 32);
  return info;
}

void save_kv_checkpoint(State& st, const std::filesystem::path& model_dir,
                        std::span<const std::int32_t> prompt_prefix, const float* d_logits,
                        const std::filesystem::path& out_path, cudaStream_t stream) {
  const int pos = int(prompt_prefix.size());
  if (pos < 0 || pos > st.max_ctx) {
    throw std::invalid_argument("kv_checkpoint: save pos=" + std::to_string(pos) + " outside [0, " +
                                std::to_string(st.max_ctx) + "]");
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (!out_path.parent_path().empty()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  const std::filesystem::path tmp_path = out_path.string() + ".tmp." + std::to_string(int(getpid()));
  const int fd = ::open(tmp_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  if (fd < 0) {
    throw std::runtime_error("kv_checkpoint: cannot create temp file " + tmp_path.string() + ": " +
                             std::strerror(errno));
  }

  const std::uint64_t row_b = state_row_bytes(st);
  const std::uint64_t swa_b = swa_total_bytes(st);
  const std::uint64_t per_visit_half = std::uint64_t(pos) * row_b;
  const std::uint64_t full_b = 2u * std::uint64_t(NUM_FULL_VISITS) * per_visit_half;

  RawHeader hdr{};
  std::memcpy(hdr.magic, kMagic, 8);
  hdr.version = kVersion;
  hdr.header_bytes = kHeaderBytes;
  hdr.act_dtype = static_cast<std::uint32_t>(kActDType);
  hdr.kv_format = state_kv_format(st);
  hdr.max_ctx_at_save = st.max_ctx;
  hdr.pos = pos;
  hdr.num_full_visits = NUM_FULL_VISITS;
  hdr.num_sliding_visits = NUM_SLIDING_VISITS;
  hdr.window = WINDOW;
  hdr.num_visits = NUM_VISITS;
  hdr.conv_k = CONV_K;
  hdr.has_logits = d_logits ? 1u : 0u;
  hdr.row_bytes = row_b;
  hdr.swa_bytes = swa_b;
  hdr.full_kv_bytes = full_b;

  const auto psha = sha256_bytes(prompt_prefix.data(), prompt_prefix.size_bytes());
  const auto msha = sha256_file_or_zero(model_dir / "config.json");
  std::memcpy(hdr.prompt_sha256, psha.data(), 32);
  std::memcpy(hdr.model_sha256, msha.data(), 32);

  try {
    // 1. 4 KiB header
    std::uint8_t page[kHeaderBytes]{};
    std::memcpy(page, &hdr, sizeof(RawHeader));
    write_all(fd, page, kHeaderBytes);

    // Single reusable 4.50 MiB streaming buffer (O(1) extra host RAM even for 1M tokens!)
    constexpr std::size_t kChunkBytes = PagedKvCache::kPageBytes;
    std::vector<std::uint8_t> buf(kChunkBytes);

    // 2. SWA rings + ring_pos + conv states
    for (const Tensor* t : swa_tensors(st)) {
      const auto* dev = static_cast<const std::uint8_t*>(t->data);
      std::size_t rem = t->bytes();
      std::size_t off = 0;
      while (rem > 0) {
        const std::size_t take = std::min(rem, kChunkBytes);
        CUDA_CHECK(cudaMemcpy(buf.data(), dev + off, take, cudaMemcpyDeviceToHost));
        write_all(fd, buf.data(), take);
        off += take;
        rem -= take;
      }
    }

    // 3. Full-attention KV [fi = 0..7][K rows 0..pos-1][V rows 0..pos-1]
    if (pos > 0) {
      if (st.paged_kv) {
        const int num_pages = (pos + PagedKvCache::kPageTokens - 1) / PagedKvCache::kPageTokens;
        auto* k_buf = buf.data();
        auto* v_buf = buf.data() + PagedKvCache::kKPageBytes;
        for (int fi = 0; fi < NUM_FULL_VISITS; ++fi) {
          for (int p = 0; p < num_pages; ++p) {
            const int rows = std::min(PagedKvCache::kPageTokens, pos - p * PagedKvCache::kPageTokens);
            st.paged_kv->export_page_to_host(fi, p, k_buf, nullptr, stream);
            write_all(fd, k_buf, std::size_t(rows) * KV_Q4_ROW_BYTES);
          }
          for (int p = 0; p < num_pages; ++p) {
            const int rows = std::min(PagedKvCache::kPageTokens, pos - p * PagedKvCache::kPageTokens);
            st.paged_kv->export_page_to_host(fi, p, nullptr, v_buf, stream);
            write_all(fd, v_buf, std::size_t(rows) * KV_Q4_ROW_BYTES);
          }
        }
      } else {
        for (int fi = 0; fi < NUM_FULL_VISITS; ++fi) {
          for (const Tensor& half : {st.k_full_at(fi), st.v_full_at(fi)}) {
            const auto* dev = static_cast<const std::uint8_t*>(half.data);
            std::size_t rem = std::size_t(per_visit_half);
            std::size_t off = 0;
            while (rem > 0) {
              const std::size_t take = std::min(rem, kChunkBytes);
              CUDA_CHECK(cudaMemcpy(buf.data(), dev + off, take, cudaMemcpyDeviceToHost));
              write_all(fd, buf.data(), take);
              off += take;
              rem -= take;
            }
          }
        }
      }
    }

    // 4. Optional last prefill logits
    if (d_logits) {
      const std::size_t lbytes = std::size_t(UNPADDED_VOCAB) * sizeof(float);
      std::vector<std::uint8_t> lbuf(lbytes);
      CUDA_CHECK(cudaMemcpy(lbuf.data(), d_logits, lbytes, cudaMemcpyDeviceToHost));
      write_all(fd, lbuf.data(), lbytes);
    }

    if (::fsync(fd) != 0) {
      throw std::runtime_error("kv_checkpoint: fsync failed: " + std::string(std::strerror(errno)));
    }
    if (::close(fd) != 0) {
      throw std::runtime_error("kv_checkpoint: close failed: " + std::string(std::strerror(errno)));
    }
    if (::rename(tmp_path.c_str(), out_path.c_str()) != 0) {
      throw std::runtime_error("kv_checkpoint: atomic rename failed: " + std::string(std::strerror(errno)));
    }
    fsync_parent_dir(out_path);
  } catch (...) {
    ::close(fd);
    std::error_code ec;
    std::filesystem::remove(tmp_path, ec);
    throw;
  }
}

int load_kv_checkpoint(State& st, const std::filesystem::path& model_dir,
                       std::span<const std::int32_t> full_prompt, float* d_logits,
                       const std::filesystem::path& path, cudaStream_t stream) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("kv_checkpoint: cannot open " + path.string() + ": " + std::strerror(errno));
  }
  try {
    struct stat st_buf{};
    if (::fstat(fd, &st_buf) != 0) {
      throw std::runtime_error("kv_checkpoint: fstat failed for " + path.string());
    }
    const RawHeader hdr = read_and_validate_raw_header(fd, std::uint64_t(st_buf.st_size));

    if (hdr.act_dtype != static_cast<std::uint32_t>(kActDType)) {
      throw std::runtime_error("kv_checkpoint: activation dtype mismatch (ckpt=" + std::to_string(hdr.act_dtype) +
                               ", engine=" + std::to_string(static_cast<std::uint32_t>(kActDType)) + ")");
    }
    if (hdr.kv_format != state_kv_format(st) || hdr.row_bytes != state_row_bytes(st)) {
      throw std::runtime_error("kv_checkpoint: KV format mismatch (ckpt fmt=" + std::to_string(hdr.kv_format) +
                               " row=" + std::to_string(hdr.row_bytes) + ", engine fmt=" +
                               std::to_string(state_kv_format(st)) + " row=" + std::to_string(state_row_bytes(st)) + ")");
    }
    if (hdr.num_full_visits != NUM_FULL_VISITS || hdr.num_sliding_visits != NUM_SLIDING_VISITS ||
        hdr.window != WINDOW || hdr.num_visits != NUM_VISITS || hdr.conv_k != CONV_K) {
      throw std::runtime_error("kv_checkpoint: model architecture geometry mismatch");
    }
    if (hdr.swa_bytes != swa_total_bytes(st)) {
      throw std::runtime_error("kv_checkpoint: SWA/conv section byte size mismatch");
    }
    if (hdr.pos < 0 || hdr.pos > st.max_ctx) {
      throw std::runtime_error("kv_checkpoint: checkpoint pos=" + std::to_string(hdr.pos) +
                               " exceeds engine max_ctx=" + std::to_string(st.max_ctx));
    }
    if (std::size_t(hdr.pos) > full_prompt.size()) {
      throw std::runtime_error("kv_checkpoint: checkpoint pos=" + std::to_string(hdr.pos) +
                               " exceeds prompt length=" + std::to_string(full_prompt.size()));
    }
    const auto expected_msha = sha256_file_or_zero(model_dir / "config.json");
    if (std::memcmp(hdr.model_sha256, expected_msha.data(), 32) != 0) {
      throw std::runtime_error("kv_checkpoint: model config.json SHA-256 mismatch (ckpt=" +
                               hex_prefix(hdr.model_sha256) + "..., current=" + hex_prefix(expected_msha.data()) + "...)");
    }
    const auto actual_psha = sha256_bytes(full_prompt.data(), std::size_t(hdr.pos) * sizeof(std::int32_t));
    if (std::memcmp(hdr.prompt_sha256, actual_psha.data(), 32) != 0) {
      throw std::runtime_error("kv_checkpoint: prompt prefix SHA-256 mismatch at pos=" + std::to_string(hdr.pos) +
                               " (ckpt=" + hex_prefix(hdr.prompt_sha256) + "..., prompt=" +
                               hex_prefix(actual_psha.data()) + "...)");
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    constexpr std::size_t kChunkBytes = PagedKvCache::kPageBytes;
    std::vector<std::uint8_t> buf(kChunkBytes);
    std::uint64_t off = hdr.header_bytes;

    // 1. Restore SWA rings + ring_pos + conv states
    for (Tensor* t : swa_tensors_mut(st)) {
      auto* dev = static_cast<std::uint8_t*>(t->data);
      std::size_t rem = t->bytes();
      std::size_t dst_off = 0;
      while (rem > 0) {
        const std::size_t take = std::min(rem, kChunkBytes);
        pread_all(fd, buf.data(), take, off);
        CUDA_CHECK(cudaMemcpy(dev + dst_off, buf.data(), take, cudaMemcpyHostToDevice));
        off += take;
        dst_off += take;
        rem -= take;
      }
    }

    // 2. Restore full-attention KV
    const std::uint64_t per_visit_half = std::uint64_t(hdr.pos) * hdr.row_bytes;
    if (hdr.pos > 0) {
      if (st.paged_kv) {
        st.paged_kv->reset();
        const int num_pages = (hdr.pos + PagedKvCache::kPageTokens - 1) / PagedKvCache::kPageTokens;
        const int hot_per_visit = std::max(1, st.paged_kv->vram_slot_count() / NUM_FULL_VISITS);
        auto* k_buf = buf.data();
        auto* v_buf = buf.data() + PagedKvCache::kKPageBytes;
        for (int fi = 0; fi < NUM_FULL_VISITS; ++fi) {
          const std::uint64_t k_base = off + std::uint64_t(fi) * 2u * per_visit_half;
          const std::uint64_t v_base = k_base + per_visit_half;
          for (int p = 0; p < num_pages; ++p) {
            const int rows = std::min(PagedKvCache::kPageTokens, hdr.pos - p * PagedKvCache::kPageTokens);
            const std::size_t bytes = std::size_t(rows) * KV_Q4_ROW_BYTES;
            if (bytes < PagedKvCache::kKPageBytes) {
              std::memset(k_buf + bytes, 0, PagedKvCache::kKPageBytes - bytes);
              std::memset(v_buf + bytes, 0, PagedKvCache::kVPageBytes - bytes);
            }
            pread_all(fd, k_buf, bytes, k_base + std::uint64_t(p) * PagedKvCache::kKPageBytes);
            pread_all(fd, v_buf, bytes, v_base + std::uint64_t(p) * PagedKvCache::kVPageBytes);
            const bool prefer_ram = (p < num_pages - hot_per_visit);
            st.paged_kv->import_page_from_host(fi, p, k_buf, v_buf, prefer_ram, stream);
          }
        }
        st.paged_kv->flush_all_tables(stream);
        off += hdr.full_kv_bytes;
      } else {
        for (int fi = 0; fi < NUM_FULL_VISITS; ++fi) {
          for (Tensor half : {st.k_full_at(fi), st.v_full_at(fi)}) {
            auto* dev = static_cast<std::uint8_t*>(half.data);
            std::size_t rem = std::size_t(per_visit_half);
            std::size_t dst_off = 0;
            while (rem > 0) {
              const std::size_t take = std::min(rem, kChunkBytes);
              pread_all(fd, buf.data(), take, off);
              CUDA_CHECK(cudaMemcpy(dev + dst_off, buf.data(), take, cudaMemcpyHostToDevice));
              off += take;
              dst_off += take;
              rem -= take;
            }
          }
        }
      }
    }

    // 3. Optional last prefill logits
    if (hdr.has_logits && d_logits) {
      const std::size_t lbytes = std::size_t(UNPADDED_VOCAB) * sizeof(float);
      std::vector<std::uint8_t> lbuf(lbytes);
      pread_all(fd, lbuf.data(), lbytes, off);
      CUDA_CHECK(cudaMemcpy(d_logits, lbuf.data(), lbytes, cudaMemcpyHostToDevice));
    }

    st.length = hdr.pos;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ::close(fd);
    return hdr.pos;
  } catch (...) {
    ::close(fd);
    throw;
  }
}

}  // namespace ingine::model

#include "model/kv_checkpoint.h"

#include "core/act.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "model/paged_kv.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
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
constexpr std::uint32_t kProducerFlagValid = 1u;

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
  // Bytes 144..191: Optional v1 extended producer trailer (#827) inside the 4096-byte header page.
  // Legacy INGCKPT1 files have bytes 144..4095 zeroed (producer_flags == 0).
  // Standalone #829 4-slot files wrote kv_cache_mode=1 at byte 144 with bytes 148..4095 zeroed and num_full_visits=4.
  std::uint32_t producer_flags;
  std::uint32_t producer_dense_fp8;
  std::int32_t producer_prefill_chunk;
  std::uint32_t producer_reserved;
  std::uint8_t producer_sha256[32];
  // Bytes 192..199: Unambiguous combined #829 v2 KV cache mode extension (disjoint from producer trailer).
  std::uint32_t kv_mode_flags;     // bit 0 (kKvModeFlagValid = 1): kv_cache_mode_v2 is explicitly encoded
  std::uint32_t kv_cache_mode_v2;  // 0 = per_pass (8 full slots), 1 = last_full (4 full slots)
};

constexpr std::uint32_t kKvModeFlagValid = 1u;

static_assert(offsetof(RawHeader, producer_flags) == 144, "producer_flags must remain at byte 144");
static_assert(offsetof(RawHeader, kv_mode_flags) == 192, "kv_mode_flags must sit at byte 192");
static_assert(sizeof(RawHeader) <= kHeaderBytes, "RawHeader exceeds 4 KiB page");

bool producer_trailer_tail_is_zero(const RawHeader& hdr) noexcept {
  if (hdr.producer_dense_fp8 != 0u || hdr.producer_prefill_chunk != 0 || hdr.producer_reserved != 0u) return false;
  for (std::uint8_t b : hdr.producer_sha256) {
    if (b != 0u) return false;
  }
  return true;
}

KvCheckpointInfo info_from_raw_header(const RawHeader& hdr) {
  if (hdr.num_full_visits != NUM_FULL_VISITS && hdr.num_full_visits != NUM_FULL_SLOTS_LAST_FULL) {
    throw std::runtime_error("kv_checkpoint: unsupported num_full_visits " + std::to_string(hdr.num_full_visits));
  }
  if (hdr.pos < 0) {
    throw std::runtime_error("kv_checkpoint: negative checkpoint pos " + std::to_string(hdr.pos));
  }
  const std::uint64_t expected_full_b =
      2u * std::uint64_t(hdr.num_full_visits) * std::uint64_t(hdr.pos) * hdr.row_bytes;
  if (hdr.full_kv_bytes != expected_full_b) {
    throw std::runtime_error("kv_checkpoint: full_kv_bytes (" + std::to_string(hdr.full_kv_bytes) +
                             ") disagrees with num_full_visits=" + std::to_string(hdr.num_full_visits) +
                             " pos=" + std::to_string(hdr.pos) + " row_bytes=" + std::to_string(hdr.row_bytes));
  }

  KvCheckpointInfo info;
  info.pos = hdr.pos;
  info.max_ctx_at_save = hdr.max_ctx_at_save;
  info.act_dtype = hdr.act_dtype;
  info.kv_format = hdr.kv_format;
  info.num_full_slots = hdr.num_full_visits;
  info.has_logits = hdr.has_logits != 0;

  if ((hdr.kv_mode_flags & ~kKvModeFlagValid) != 0u) {
    throw std::runtime_error("kv_checkpoint: unsupported kv_mode_flags " + std::to_string(hdr.kv_mode_flags));
  }
  if ((hdr.kv_mode_flags & kKvModeFlagValid) != 0u) {
    // Fresh combined #827+#829 header: mode at byte 196, producer trailer at bytes 144..191.
    if (hdr.kv_cache_mode_v2 > 1u) {
      throw std::runtime_error("kv_checkpoint: unsupported kv_cache_mode " + std::to_string(hdr.kv_cache_mode_v2));
    }
    info.kv_cache_mode = (hdr.kv_cache_mode_v2 == 1u) ? KvCacheMode::LastFull : KvCacheMode::PerPass;
    const int want_slots = (info.kv_cache_mode == KvCacheMode::LastFull) ? NUM_FULL_SLOTS_LAST_FULL : NUM_FULL_VISITS;
    if (hdr.num_full_visits != want_slots) {
      throw std::runtime_error("kv_checkpoint: kv_cache_mode=" + std::string(to_string(info.kv_cache_mode)) +
                               " disagrees with num_full_visits=" + std::to_string(hdr.num_full_visits));
    }
    if ((hdr.producer_flags & ~kProducerFlagValid) != 0u) {
      throw std::runtime_error("kv_checkpoint: unsupported producer_flags " + std::to_string(hdr.producer_flags));
    }
    if ((hdr.producer_flags & kProducerFlagValid) != 0u) {
      if (producer_trailer_tail_is_zero(hdr)) {
        throw std::runtime_error("kv_checkpoint: producer_flags valid bit set with empty producer trailer");
      }
      info.has_producer_identity = true;
      info.producer_dense_fp8 = hdr.producer_dense_fp8 != 0;
      info.producer_prefill_chunk = hdr.producer_prefill_chunk;
      std::memcpy(info.producer_sha256.data(), hdr.producer_sha256, 32);
    }
  } else if (hdr.num_full_visits == NUM_FULL_VISITS) {
    // Legacy or #827 producer-tagged per_pass header (8 full slots, bytes 192..199 zeroed).
    info.kv_cache_mode = KvCacheMode::PerPass;
    if ((hdr.producer_flags & ~kProducerFlagValid) != 0u) {
      throw std::runtime_error("kv_checkpoint: unsupported producer_flags " + std::to_string(hdr.producer_flags));
    }
    if ((hdr.producer_flags & kProducerFlagValid) != 0u) {
      if (producer_trailer_tail_is_zero(hdr)) {
        throw std::runtime_error("kv_checkpoint: producer_flags valid bit set with empty producer trailer");
      }
      info.has_producer_identity = true;
      info.producer_dense_fp8 = hdr.producer_dense_fp8 != 0;
      info.producer_prefill_chunk = hdr.producer_prefill_chunk;
      std::memcpy(info.producer_sha256.data(), hdr.producer_sha256, 32);
    }
  } else {
    // Standalone #829 4-slot file: wrote kv_cache_mode=1 at byte 144 with bytes 148..191 zeroed and NO producer
    // trailer. NEVER promote byte 144 to has_producer_identity; fail closed on any other 4-slot legacy layout.
    if (hdr.producer_flags != 1u || !producer_trailer_tail_is_zero(hdr)) {
      throw std::runtime_error("kv_checkpoint: corrupt or ambiguous 4-slot legacy header");
    }
    info.kv_cache_mode = KvCacheMode::LastFull;
    info.has_producer_identity = false;
  }

  std::memcpy(info.prompt_sha256.data(), hdr.prompt_sha256, 32);
  std::memcpy(info.model_sha256.data(), hdr.model_sha256, 32);
  return info;
}

void reset_state_noexcept(State& st, cudaStream_t stream) noexcept {
  try {
    (void)cudaMemsetAsync(st.ring_pos.data, 0xff, st.ring_pos.bytes(), stream);
    for (Tensor* t : {&st.conv_k, &st.conv_v, &st.conv_attn, &st.conv_mlp}) {
      (void)cudaMemsetAsync(t->data, 0, t->bytes(), stream);
    }
    if (st.paged_kv) st.paged_kv->reset();
    st.length = 0;
    (void)cudaStreamSynchronize(stream);
  } catch (...) {
    st.length = 0;
  }
}

void hash_tagged_bytes(Sha256& ctx, const char* tag, const void* data, std::uint64_t len) noexcept {
  const std::uint64_t tag_len = std::uint64_t(std::strlen(tag));
  ctx.update(&tag_len, sizeof(tag_len));
  ctx.update(tag, std::size_t(tag_len));
  ctx.update(&len, sizeof(len));
  if (len > 0) ctx.update(data, std::size_t(len));
}

void hash_optional_file(Sha256& ctx, const char* tag, const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    const std::uint8_t absent = 0;
    hash_tagged_bytes(ctx, tag, &absent, 1);
    return;
  }
  const std::uint8_t present = 1;
  hash_tagged_bytes(ctx, tag, &present, 1);
  const auto file_sha = sha256_file_or_zero(path);
  hash_tagged_bytes(ctx, "sha256", file_sha.data(), file_sha.size());
}

void pread_all(int fd, void* buf, std::size_t n, std::uint64_t off);

std::array<std::uint8_t, 32> hash_safetensors_tensor_file(const SafetensorsFile& file,
                                                          const SafetensorsTensor& tensor,
                                                          std::vector<std::byte>& scratch) {
  std::vector<std::array<std::uint8_t, 32>> chunk_shas;
  if (tensor.bytes > 0) {
    const std::size_t n_chunks =
        std::size_t((tensor.bytes + kSafetensorsHashChunkBytes - 1) / kSafetensorsHashChunkBytes);
    chunk_shas.reserve(n_chunks);
    std::uint64_t done = 0;
    while (done < tensor.bytes) {
      const std::size_t take =
          std::size_t(std::min<std::uint64_t>(tensor.bytes - done, kSafetensorsHashChunkBytes));
      if (scratch.size() < take) scratch.resize(take);
      file.read_range(tensor.offset + done, take, scratch.data());
      chunk_shas.push_back(sha256_bytes(scratch.data(), take));
      done += take;
    }
  }
  return sha256_tensor_from_chunk_digests(tensor.name, tensor.dtype, tensor.shape, tensor.bytes, chunk_shas);
}

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

std::array<std::uint8_t, 32> sha256_tensor_from_chunk_digests(
    std::string_view name, std::string_view dtype, std::span<const std::int64_t> shape, std::uint64_t bytes,
    std::span<const std::array<std::uint8_t, 32>> chunk_shas) noexcept {
  Sha256 ctx;
  static constexpr char kDomain[] = "INGINE_TENSOR_V1";
  ctx.update(kDomain, sizeof(kDomain) - 1);
  hash_tagged_bytes(ctx, "name", name.data(), name.size());
  hash_tagged_bytes(ctx, "dtype", dtype.data(), dtype.size());
  const std::uint64_t rank = std::uint64_t(shape.size());
  hash_tagged_bytes(ctx, "rank", &rank, sizeof(rank));
  if (!shape.empty()) hash_tagged_bytes(ctx, "shape", shape.data(), shape.size_bytes());
  hash_tagged_bytes(ctx, "bytes", &bytes, sizeof(bytes));
  const std::uint64_t n_chunks = std::uint64_t(chunk_shas.size());
  hash_tagged_bytes(ctx, "n_chunks", &n_chunks, sizeof(n_chunks));
  for (const auto& c : chunk_shas) ctx.update(c.data(), c.size());
  return ctx.finish();
}

std::array<std::uint8_t, 32> compute_model_package_sha256_from_tensors(
    const std::filesystem::path& model_dir,
    const std::map<std::string, std::array<std::uint8_t, 32>>& tensor_shas) {
  Sha256 ctx;
  static constexpr char kDomain[] = "INGINE_MODEL_PKG_V2";
  ctx.update(kDomain, sizeof(kDomain) - 1);
  // Hash authoritative model config/quant/index metadata + 100% of verified weight tensor contents.
  // Never rely on optional SHA256SUMS text files as a proxy for weight bytes.
  hash_optional_file(ctx, "config.json", model_dir / "config.json");
  hash_optional_file(ctx, "hf_quant_config.json", model_dir / "hf_quant_config.json");
  hash_optional_file(ctx, "model.safetensors.index.json", model_dir / "model.safetensors.index.json");
  const std::uint64_t n_tensors = std::uint64_t(tensor_shas.size());
  hash_tagged_bytes(ctx, "num_tensors", &n_tensors, sizeof(n_tensors));
  for (const auto& [name, tsha] : tensor_shas) {
    hash_tagged_bytes(ctx, "tensor_name", name.data(), name.size());
    ctx.update(tsha.data(), tsha.size());
  }
  return ctx.finish();
}

std::array<std::uint8_t, 32> compute_consumed_weights_sha256(
    const std::array<std::uint8_t, 32>& config_sha, const std::array<std::uint8_t, 32>& quant_config_sha,
    const std::map<std::string, std::array<std::uint8_t, 32>>& tensor_shas) noexcept {
  Sha256 ctx;
  static constexpr char kDomain[] = "INGINE_CONSUMED_WEIGHTS_V1";
  ctx.update(kDomain, sizeof(kDomain) - 1);
  hash_tagged_bytes(ctx, "config_sha256", config_sha.data(), config_sha.size());
  hash_tagged_bytes(ctx, "quant_config_sha256", quant_config_sha.data(), quant_config_sha.size());
  const std::uint64_t n_tensors = std::uint64_t(tensor_shas.size());
  hash_tagged_bytes(ctx, "num_tensors", &n_tensors, sizeof(n_tensors));
  for (const auto& [name, tsha] : tensor_shas) {
    hash_tagged_bytes(ctx, "tensor_name", name.data(), name.size());
    ctx.update(tsha.data(), tsha.size());
  }
  return ctx.finish();
}

std::optional<std::array<std::uint8_t, 32>> compute_model_package_sha256_opt(const std::filesystem::path& model_dir) {
  if (model_dir.empty()) return std::nullopt;
  std::error_code ec;
  if (!std::filesystem::is_directory(model_dir, ec) || ec) return std::nullopt;
  try {
    std::map<std::string, std::array<std::uint8_t, 32>> tensor_shas;
    std::vector<std::byte> scratch;
    if (std::filesystem::exists(model_dir / "model.safetensors.index.json", ec) ||
        std::filesystem::exists(model_dir / "model.safetensors", ec) ||
        std::filesystem::exists(model_dir / "model.ingpack", ec)) {
      const SafetensorsModel m(model_dir);
      for (const std::string& name : m.names()) {
        const SafetensorsRef ref = m.at(name);
        tensor_shas.emplace(name, hash_safetensors_tensor_file(*ref.file, *ref.tensor, scratch));
      }
    } else {
      std::vector<std::filesystem::path> shards;
      for (const auto& entry : std::filesystem::directory_iterator(model_dir, ec)) {
        if (ec) return std::nullopt;
        if (!entry.is_regular_file(ec)) continue;
        const auto p = entry.path();
        if (p.extension() != ".safetensors") continue;
        const std::string fname = p.filename().string();
        if (fname == "mtp.safetensors" || fname.find("abliterated") != std::string::npos) continue;
        shards.push_back(p);
      }
      std::sort(shards.begin(), shards.end(), [](const std::filesystem::path& a, const std::filesystem::path& b) {
        return a.filename() < b.filename();
      });
      for (const auto& shard : shards) {
        const SafetensorsFile sf(shard);
        for (const SafetensorsTensor& t : sf.tensors()) {
          tensor_shas.emplace(t.name, hash_safetensors_tensor_file(sf, t, scratch));
        }
      }
    }
    if (tensor_shas.empty()) return std::nullopt;
    return compute_model_package_sha256_from_tensors(model_dir, tensor_shas);
  } catch (...) {
    return std::nullopt;
  }
}

std::array<std::uint8_t, 32> compute_model_package_sha256(const std::filesystem::path& model_dir) {
  return compute_model_package_sha256_opt(model_dir).value_or(std::array<std::uint8_t, 32>{});
}

std::array<std::uint8_t, 32> compute_disk_producer_sha256(const StateIdentity& identity) noexcept {
  Sha256 ctx;
  static constexpr char kDomain[] = "INGCKPT1_PRODUCER_V1";
  ctx.update(kDomain, sizeof(kDomain) - 1);
  ctx.update(identity.model_sha256.data(), identity.model_sha256.size());
  ctx.update(&identity.act_dtype, sizeof(identity.act_dtype));
  const std::uint8_t dense_fp8 = identity.dense_fp8 ? 1u : 0u;
  ctx.update(&dense_fp8, sizeof(dense_fp8));
  ctx.update(&identity.kv_format, sizeof(identity.kv_format));
  ctx.update(&identity.prefill_chunk, sizeof(identity.prefill_chunk));
  if (identity.kv_cache_mode != 0u) {
    static constexpr char kKvModeDomain[] = "KVMODE";
    ctx.update(kKvModeDomain, sizeof(kKvModeDomain) - 1);
    ctx.update(&identity.kv_cache_mode, sizeof(identity.kv_cache_mode));
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
  return info_from_raw_header(hdr);
}

void save_kv_checkpoint(State& st, const std::filesystem::path& model_dir,
                        std::span<const std::int32_t> prompt_prefix, const float* d_logits,
                        const std::filesystem::path& out_path, cudaStream_t stream,
                        const KvCheckpointProducer* producer) {
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

  const int nfull = st.num_full_slots();
  const std::uint64_t row_b = state_row_bytes(st);
  const std::uint64_t swa_b = swa_total_bytes(st);
  const std::uint64_t per_visit_half = std::uint64_t(pos) * row_b;
  const std::uint64_t full_b = 2u * std::uint64_t(nfull) * per_visit_half;

  RawHeader hdr{};
  std::memcpy(hdr.magic, kMagic, 8);
  hdr.version = kVersion;
  hdr.header_bytes = kHeaderBytes;
  hdr.act_dtype = static_cast<std::uint32_t>(kActDType);
  hdr.kv_format = state_kv_format(st);
  hdr.max_ctx_at_save = st.max_ctx;
  hdr.pos = pos;
  hdr.num_full_visits = nfull;
  hdr.num_sliding_visits = NUM_SLIDING_VISITS;
  hdr.window = WINDOW;
  hdr.num_visits = NUM_VISITS;
  hdr.conv_k = CONV_K;
  hdr.has_logits = d_logits ? 1u : 0u;
  hdr.row_bytes = row_b;
  hdr.swa_bytes = swa_b;
  hdr.full_kv_bytes = full_b;
  hdr.kv_mode_flags = kKvModeFlagValid;
  hdr.kv_cache_mode_v2 = static_cast<std::uint32_t>(st.kv_last_full ? KvCacheMode::LastFull : KvCacheMode::PerPass);

  const auto psha = sha256_bytes(prompt_prefix.data(), prompt_prefix.size_bytes());
  const auto msha = sha256_file_or_zero(model_dir / "config.json");
  std::memcpy(hdr.prompt_sha256, psha.data(), 32);
  std::memcpy(hdr.model_sha256, msha.data(), 32);
  if (producer && producer->valid) {
    hdr.producer_flags = kProducerFlagValid;
    hdr.producer_dense_fp8 = producer->dense_fp8 ? 1u : 0u;
    hdr.producer_prefill_chunk = producer->prefill_chunk;
    hdr.producer_reserved = 0u;
    std::memcpy(hdr.producer_sha256, producer->producer_sha256.data(), 32);
  }

  try {
    // 1. 4 KiB header
    std::uint8_t page[kHeaderBytes]{};
    std::memcpy(page, &hdr, sizeof(RawHeader));
    write_all(fd, page, kHeaderBytes);

    // Single reusable 4.50 MiB streaming buffer (O(1) extra host RAM even for 1M tokens!)
    constexpr std::size_t kChunkBytes = PagedKvCache::kPageBytes;
    std::vector<std::uint8_t> buf(kChunkBytes);

    // 2. SWA rings + ring_pos + conv states (strictly stream-ordered; caller stream may be cudaStreamNonBlocking)
    for (const Tensor* t : swa_tensors(st)) {
      const auto* dev = static_cast<const std::uint8_t*>(t->data);
      std::size_t rem = t->bytes();
      std::size_t off = 0;
      while (rem > 0) {
        const std::size_t take = std::min(rem, kChunkBytes);
        CUDA_CHECK(cudaMemcpyAsync(buf.data(), dev + off, take, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        write_all(fd, buf.data(), take);
        off += take;
        rem -= take;
      }
    }

    // 3. Full-attention KV [fi = 0..nfull-1][K rows 0..pos-1][V rows 0..pos-1]
    if (pos > 0) {
      if (st.paged_kv) {
        const int num_pages = (pos + PagedKvCache::kPageTokens - 1) / PagedKvCache::kPageTokens;
        auto* k_buf = buf.data();
        auto* v_buf = buf.data() + PagedKvCache::kKPageBytes;
        for (int fi = 0; fi < nfull; ++fi) {
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
        for (int fi = 0; fi < nfull; ++fi) {
          for (const Tensor& half : {st.k_full_at(fi), st.v_full_at(fi)}) {
            const auto* dev = static_cast<const std::uint8_t*>(half.data);
            std::size_t rem = std::size_t(per_visit_half);
            std::size_t off = 0;
            while (rem > 0) {
              const std::size_t take = std::min(rem, kChunkBytes);
              CUDA_CHECK(cudaMemcpyAsync(buf.data(), dev + off, take, cudaMemcpyDeviceToHost, stream));
              CUDA_CHECK(cudaStreamSynchronize(stream));
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
      CUDA_CHECK(cudaMemcpyAsync(lbuf.data(), d_logits, lbytes, cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
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

KvCheckpointLoadResult load_kv_checkpoint_checked(
    State& st, const std::filesystem::path& model_dir, std::span<const std::int32_t> full_prompt, float* d_logits,
    const std::filesystem::path& path, cudaStream_t stream,
    const std::function<bool(const KvCheckpointInfo&)>& eligibility) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("kv_checkpoint: cannot open " + path.string() + ": " + std::strerror(errno));
  }
  bool state_mutated = false;
  try {
    struct stat st_buf{};
    if (::fstat(fd, &st_buf) != 0) {
      throw std::runtime_error("kv_checkpoint: fstat failed for " + path.string());
    }
    const RawHeader hdr = read_and_validate_raw_header(fd, std::uint64_t(st_buf.st_size));
    const KvCheckpointInfo info = info_from_raw_header(hdr);
    if (eligibility && !eligibility(info)) {
      ::close(fd);
      return KvCheckpointLoadResult{false, info};
    }

    if (hdr.act_dtype != static_cast<std::uint32_t>(kActDType)) {
      throw std::runtime_error("kv_checkpoint: activation dtype mismatch (ckpt=" + std::to_string(hdr.act_dtype) +
                               ", engine=" + std::to_string(static_cast<std::uint32_t>(kActDType)) + ")");
    }
    if (hdr.kv_format != state_kv_format(st) || hdr.row_bytes != state_row_bytes(st)) {
      throw std::runtime_error("kv_checkpoint: KV format mismatch (ckpt fmt=" + std::to_string(hdr.kv_format) +
                               " row=" + std::to_string(hdr.row_bytes) + ", engine fmt=" +
                               std::to_string(state_kv_format(st)) + " row=" + std::to_string(state_row_bytes(st)) + ")");
    }
    const int expected_nfull = st.num_full_slots();
    const KvCacheMode expected_mode = st.kv_last_full ? KvCacheMode::LastFull : KvCacheMode::PerPass;
    if (info.kv_cache_mode != expected_mode || info.num_full_slots != expected_nfull) {
      throw std::runtime_error("kv_checkpoint: kv_cache_mode mismatch (ckpt=" + std::string(to_string(info.kv_cache_mode)) +
                               " full_slots=" + std::to_string(info.num_full_slots) + ", engine=" +
                               std::string(to_string(expected_mode)) + " full_slots=" +
                               std::to_string(expected_nfull) + ")");
    }
    if (hdr.num_sliding_visits != NUM_SLIDING_VISITS || hdr.window != WINDOW || hdr.num_visits != NUM_VISITS ||
        hdr.conv_k != CONV_K) {
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

    // 1. Restore SWA rings + ring_pos + conv states.
    // All H2D copies MUST be queued on `stream` (never stream-0 cudaMemcpy): Runtime::stream() is
    // cudaStreamNonBlocking and does not synchronize with stream 0, so a pageable H2D cudaMemcpy can return
    // while DMA is still in flight and race against subsequent kernels on `stream`.
    state_mutated = true;
    for (Tensor* t : swa_tensors_mut(st)) {
      auto* dev = static_cast<std::uint8_t*>(t->data);
      std::size_t rem = t->bytes();
      std::size_t dst_off = 0;
      while (rem > 0) {
        const std::size_t take = std::min(rem, kChunkBytes);
        pread_all(fd, buf.data(), take, off);
        CUDA_CHECK(cudaMemcpyAsync(dev + dst_off, buf.data(), take, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
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
        const int hot_per_visit = std::max(1, st.paged_kv->vram_slot_count() / expected_nfull);
        auto* k_buf = buf.data();
        auto* v_buf = buf.data() + PagedKvCache::kKPageBytes;
        for (int fi = 0; fi < expected_nfull; ++fi) {
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
        for (int fi = 0; fi < expected_nfull; ++fi) {
          for (Tensor half : {st.k_full_at(fi), st.v_full_at(fi)}) {
            auto* dev = static_cast<std::uint8_t*>(half.data);
            std::size_t rem = std::size_t(per_visit_half);
            std::size_t dst_off = 0;
            while (rem > 0) {
              const std::size_t take = std::min(rem, kChunkBytes);
              pread_all(fd, buf.data(), take, off);
              CUDA_CHECK(cudaMemcpyAsync(dev + dst_off, buf.data(), take, cudaMemcpyHostToDevice, stream));
              CUDA_CHECK(cudaStreamSynchronize(stream));
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
      CUDA_CHECK(cudaMemcpyAsync(d_logits, lbuf.data(), lbytes, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    st.length = hdr.pos;
    CUDA_CHECK(cudaStreamSynchronize(stream));
    ::close(fd);
    return KvCheckpointLoadResult{true, info};
  } catch (...) {
    if (state_mutated) reset_state_noexcept(st, stream);
    ::close(fd);
    throw;
  }
}

int load_kv_checkpoint(State& st, const std::filesystem::path& model_dir,
                       std::span<const std::int32_t> full_prompt, float* d_logits,
                       const std::filesystem::path& path, cudaStream_t stream) {
  return load_kv_checkpoint_checked(st, model_dir, full_prompt, d_logits, path, stream, nullptr).info.pos;
}

}  // namespace ingine::model

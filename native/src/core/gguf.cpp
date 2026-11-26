// GGUF reader and host dequantizers (see gguf.h). Block layouts and dequantization follow ggml
// (ggml-common.h / ggml-quants.c, dequantize_row_*): every value is formed in FP32 exactly as there.
#include "core/gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ingine {
namespace {

enum : std::uint32_t {
    T_U8 = 0, T_I8 = 1, T_U16 = 2, T_I16 = 3, T_U32 = 4, T_I32 = 5, T_F32 = 6, T_BOOL = 7, T_STR = 8, T_ARR = 9,
    T_U64 = 10, T_I64 = 11, T_F64 = 12,
};

struct FdGuard {
    int fd = -1;
    explicit FdGuard(int f) noexcept : fd(f) {}
    ~FdGuard() {
        if (fd >= 0) ::close(fd);
    }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
};

std::uint64_t checked_mul_u64(std::uint64_t a, std::uint64_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) throw std::runtime_error(what);
    return a * b;
}

std::size_t gguf_scalar_min_bytes(std::uint32_t t) noexcept {
    switch (t) {
        case T_U8: case T_I8: case T_BOOL: return 1;
        case T_U16: case T_I16: return 2;
        case T_U32: case T_I32: case T_F32: return 4;
        case T_U64: case T_I64: case T_F64: case T_STR: return 8;
        default: return 0;
    }
}

struct Reader {
    const std::byte* p;
    const std::byte* end;
    std::size_t remaining() const noexcept { return std::size_t(end - p); }
    template <class T>
    T get() {
        if (remaining() < sizeof(T)) throw std::runtime_error("gguf: truncated header");
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const auto n = get<std::uint64_t>();
        if (n > remaining()) throw std::runtime_error("gguf: truncated string");
        std::string s(reinterpret_cast<const char*>(p), std::size_t(n));
        p += std::size_t(n);
        return s;
    }
};

float fp16_to_f32(std::uint16_t h) {
    const std::uint32_t s = std::uint32_t(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    std::uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {  // subnormal: m * 2^-24
            const float f = float(m) * 0x1p-24f;
            bits = std::bit_cast<std::uint32_t>(f) | s;
        }
    } else if (e == 31) {
        bits = s | 0x7f800000u | (m << 13);
    } else {
        bits = s | ((e + 112u) << 23) | (m << 13);
    }
    return std::bit_cast<float>(bits);
}
float bf16_to_f32(std::uint16_t h) { return std::bit_cast<float>(std::uint32_t(h) << 16); }
std::uint16_t rd16(const std::byte* p) {
    std::uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
// UE4M3 block scale of NVFP4 (ggml_ue4m3_to_fp32): 4 exponent bits (bias 7), 3 mantissa bits; 0x7F is NaN, read 0.
float ue4m3_to_f32(std::uint8_t x) {
    x &= 0x7f;
    if (x == 0x7f) return 0.0f;
    const int e = x >> 3, m = x & 7;
    return e ? std::ldexp(1.0f + float(m) / 8.0f, e - 7) : std::ldexp(float(m) / 8.0f, -6);
}
constexpr float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
constexpr std::int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

void scale_min_k4(int j, const std::uint8_t* q, std::uint8_t& d, std::uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63, m = q[j + 4] & 63;
    } else {
        d = std::uint8_t((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = std::uint8_t((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

}  // namespace

int ggml_block_values(GgmlType t) noexcept {
    switch (t) {
        case GgmlType::F32: case GgmlType::F16: case GgmlType::BF16: case GgmlType::I8: case GgmlType::I16:
        case GgmlType::I32: case GgmlType::I64: case GgmlType::F64: return 1;
        case GgmlType::Q4_0: case GgmlType::Q4_1: case GgmlType::Q5_0: case GgmlType::Q5_1: case GgmlType::Q8_0:
        case GgmlType::Q8_1: case GgmlType::IQ4_NL: case GgmlType::MXFP4: return 32;
        case GgmlType::NVFP4: return 64;
        case GgmlType::Q2_K: case GgmlType::Q3_K: case GgmlType::Q4_K: case GgmlType::Q5_K: case GgmlType::Q6_K:
        case GgmlType::Q8_K: case GgmlType::IQ2_XXS: case GgmlType::IQ2_XS: case GgmlType::IQ3_XXS:
        case GgmlType::IQ1_S: case GgmlType::IQ3_S: case GgmlType::IQ2_S: case GgmlType::IQ4_XS:
        case GgmlType::IQ1_M: return 256;
    }
    return 0;
}

int ggml_block_bytes(GgmlType t) noexcept {
    switch (t) {
        case GgmlType::F32: case GgmlType::I32: return 4;
        case GgmlType::F16: case GgmlType::BF16: case GgmlType::I16: return 2;
        case GgmlType::I8: return 1;
        case GgmlType::I64: case GgmlType::F64: return 8;
        case GgmlType::Q4_0: return 18;
        case GgmlType::Q4_1: return 20;
        case GgmlType::Q5_0: return 22;
        case GgmlType::Q5_1: return 24;
        case GgmlType::Q8_0: return 34;
        case GgmlType::Q8_1: return 36;
        case GgmlType::IQ4_NL: return 18;
        case GgmlType::MXFP4: return 17;
        case GgmlType::NVFP4: return 36;
        case GgmlType::Q2_K: return 84;
        case GgmlType::Q3_K: return 110;
        case GgmlType::Q4_K: return 144;
        case GgmlType::Q5_K: return 176;
        case GgmlType::Q6_K: return 210;
        case GgmlType::Q8_K: return 292;
        case GgmlType::IQ2_XXS: return 66;
        case GgmlType::IQ2_XS: return 74;
        case GgmlType::IQ3_XXS: return 98;
        case GgmlType::IQ1_S: return 50;
        case GgmlType::IQ3_S: return 110;
        case GgmlType::IQ2_S: return 82;
        case GgmlType::IQ4_XS: return 136;
        case GgmlType::IQ1_M: return 56;
    }
    return 0;
}

const char* ggml_type_name(GgmlType t) noexcept {
    switch (t) {
        case GgmlType::F32: return "F32"; case GgmlType::F16: return "F16"; case GgmlType::BF16: return "BF16";
        case GgmlType::Q4_0: return "Q4_0"; case GgmlType::Q4_1: return "Q4_1"; case GgmlType::Q5_0: return "Q5_0";
        case GgmlType::Q5_1: return "Q5_1"; case GgmlType::Q8_0: return "Q8_0"; case GgmlType::Q8_1: return "Q8_1";
        case GgmlType::Q2_K: return "Q2_K"; case GgmlType::Q3_K: return "Q3_K"; case GgmlType::Q4_K: return "Q4_K";
        case GgmlType::Q5_K: return "Q5_K"; case GgmlType::Q6_K: return "Q6_K"; case GgmlType::Q8_K: return "Q8_K";
        case GgmlType::IQ4_NL: return "IQ4_NL"; case GgmlType::IQ4_XS: return "IQ4_XS";
        case GgmlType::MXFP4: return "MXFP4"; case GgmlType::NVFP4: return "NVFP4";
        default: return "?";
    }
}

std::size_t ggml_row_bytes(GgmlType t, std::int64_t n) {
    const int bv = ggml_block_values(t), bb = ggml_block_bytes(t);
    if (!bv || !bb) throw std::runtime_error(std::string("gguf: unsupported ggml type ") + ggml_type_name(t));
    if (n < 0 || n % bv != 0) throw std::runtime_error("gguf: row length is not a whole number of blocks");
    const std::uint64_t bytes =
        checked_mul_u64(std::uint64_t(n / bv), std::uint64_t(bb), "gguf: row byte size overflow");
    if (bytes > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("gguf: row byte size overflow");
    return std::size_t(bytes);
}

std::int64_t GgufTensor::numel() const noexcept {
    std::int64_t n = 1;
    for (auto d : ne) n *= d;
    return n;
}

void GgufFile::Mapping::reset() noexcept {
    if (data && bytes > 0) ::munmap(data, bytes);
    data = nullptr;
    bytes = 0;
}

GgufFile::Mapping::~Mapping() { reset(); }

GgufFile::GgufFile(const std::filesystem::path& path) : path_(path) {
    FdGuard fd(::open(path.c_str(), O_RDONLY));
    if (fd.fd < 0) throw std::runtime_error("gguf: cannot open " + path.string());
    struct stat st {};
    if (::fstat(fd.fd, &st) != 0) throw std::runtime_error("gguf: cannot stat " + path.string());
    if (!S_ISREG(st.st_mode)) throw std::runtime_error("gguf: not a regular file: " + path.string());
    if (st.st_size <= 0) throw std::runtime_error("gguf: empty or invalid file size: " + path.string());
    const std::size_t map_bytes = std::size_t(st.st_size);
    void* raw = ::mmap(nullptr, map_bytes, PROT_READ, MAP_SHARED, fd.fd, 0);
    if (raw == MAP_FAILED) throw std::runtime_error("gguf: cannot map " + path.string());
    mapping_ = Mapping(raw, map_bytes);

    const auto* base = static_cast<const std::byte*>(mapping_.data);
    Reader rd{base, base + mapping_.bytes};
    if (rd.get<std::uint32_t>() != 0x46554747u) throw std::runtime_error("gguf: bad magic in " + path.string());
    version_ = rd.get<std::uint32_t>();
    if (version_ < 2 || version_ > 3) throw std::runtime_error("gguf: unsupported version");
    const auto n_tensors = rd.get<std::uint64_t>(), n_kv = rd.get<std::uint64_t>();
    // Minimum serialized sizes: KV entry >= 12B (8B key len + 4B value type); tensor >= 24B (8B name + 4B nd + 4B type + 8B off).
    if (n_kv > rd.remaining() / 12u) throw std::runtime_error("gguf: metadata count exceeds file size");
    if (n_tensors > rd.remaining() / 24u) throw std::runtime_error("gguf: tensor count exceeds file size");

    auto scalar = [&](std::uint32_t t, Value& v) {
        switch (t) {
            case T_U8: v.i = rd.get<std::uint8_t>(); break;
            case T_I8: v.i = rd.get<std::int8_t>(); break;
            case T_U16: v.i = rd.get<std::uint16_t>(); break;
            case T_I16: v.i = rd.get<std::int16_t>(); break;
            case T_U32: v.i = rd.get<std::uint32_t>(); break;
            case T_I32: v.i = rd.get<std::int32_t>(); break;
            case T_U64: v.i = std::int64_t(rd.get<std::uint64_t>()); break;
            case T_I64: v.i = rd.get<std::int64_t>(); break;
            case T_BOOL: v.i = rd.get<std::uint8_t>() != 0; break;
            case T_F32: v.f = rd.get<float>(); break;
            case T_F64: v.f = rd.get<double>(); break;
            case T_STR: v.s = rd.str(); break;
            default: throw std::runtime_error("gguf: bad value type");
        }
    };
    for (std::uint64_t i = 0; i < n_kv; ++i) {
        std::string key = rd.str();
        Value v;
        v.type = rd.get<std::uint32_t>();
        if (v.type == T_ARR) {
            const auto at = rd.get<std::uint32_t>();
            const auto n = rd.get<std::uint64_t>();
            const std::size_t emin = gguf_scalar_min_bytes(at);
            if (emin == 0) throw std::runtime_error("gguf: bad array value type");
            if (n > rd.remaining() / emin) throw std::runtime_error("gguf: truncated metadata array");
            if (at == T_F32 || at == T_F64) v.fa.reserve(std::size_t(n));
            else if (at != T_STR) v.ia.reserve(std::size_t(n));
            for (std::uint64_t j = 0; j < n; ++j) {
                Value e;
                scalar(at, e);
                if (at == T_STR) ++v.n_strings;  // (vocabulary arrays: not kept)
                else if (at == T_F32 || at == T_F64) v.fa.push_back(e.f);
                else v.ia.push_back(e.i);
            }
        } else {
            scalar(v.type, v);
        }
        kv_.emplace(std::move(key), std::move(v));
    }
    if (n_tensors > rd.remaining() / 24u) throw std::runtime_error("gguf: tensor count exceeds remaining file size");
    tensors_.reserve(std::size_t(n_tensors));
    for (std::uint64_t i = 0; i < n_tensors; ++i) {
        GgufTensor t;
        t.name = rd.str();
        const auto nd = rd.get<std::uint32_t>();
        if (nd == 0 || nd > 4)
            throw std::runtime_error("gguf: tensor " + t.name + " has invalid rank " + std::to_string(nd));
        if (std::size_t(nd) * sizeof(std::uint64_t) > rd.remaining())
            throw std::runtime_error("gguf: truncated tensor dimensions for " + t.name);
        t.ne.reserve(nd);
        for (std::uint32_t d = 0; d < nd; ++d) {
            const auto dim_u = rd.get<std::uint64_t>();
            if (dim_u == 0 || dim_u > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
                throw std::runtime_error("gguf: tensor " + t.name + " has invalid dimension");
            t.ne.push_back(std::int64_t(dim_u));
        }
        t.type = static_cast<GgmlType>(rd.get<std::uint32_t>());
        t.offset = rd.get<std::uint64_t>();
        const int bv = ggml_block_values(t.type), bb = ggml_block_bytes(t.type);
        if (!bv || !bb) throw std::runtime_error("gguf: tensor " + t.name + " has unsupported type " + std::to_string(int(t.type)));
        const std::uint64_t row_len_u = std::uint64_t(t.ne[0]);
        if (row_len_u % std::uint64_t(bv) != 0)
            throw std::runtime_error("gguf: tensor " + t.name + " row width " + std::to_string(row_len_u) +
                                     " is not block-aligned to " + std::to_string(bv));
        std::uint64_t rows_u = 1;
        for (std::size_t d = 1; d < t.ne.size(); ++d)
            rows_u = checked_mul_u64(rows_u, std::uint64_t(t.ne[d]), "gguf: tensor row count overflow");
        const std::uint64_t numel_u = checked_mul_u64(row_len_u, rows_u, "gguf: tensor element count overflow");
        if (numel_u > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
            throw std::runtime_error("gguf: tensor element count exceeds int64");
        const std::uint64_t row_bytes_u =
            checked_mul_u64(row_len_u / std::uint64_t(bv), std::uint64_t(bb), "gguf: tensor row byte size overflow");
        t.bytes = checked_mul_u64(row_bytes_u, rows_u, "gguf: tensor byte size overflow");
        if (!index_.emplace(t.name, tensors_.size()).second)
            throw std::runtime_error("gguf: duplicate tensor " + t.name);
        tensors_.push_back(std::move(t));
    }
    std::uint64_t align_u = 32;
    if (has("general.alignment")) {
        const auto align_opt = get_int("general.alignment");
        if (!align_opt || *align_opt <= 0)
            throw std::runtime_error("gguf: invalid general.alignment");
        align_u = std::uint64_t(*align_opt);
        if ((align_u & (align_u - 1)) != 0 || align_u > (std::uint64_t(1) << 20))
            throw std::runtime_error("gguf: general.alignment must be a power of two <= 1 MiB");
    }
    const std::size_t hdr = std::size_t(rd.p - base);
    const std::size_t align = std::size_t(align_u);
    const std::size_t rem = hdr % align;
    const std::size_t pad = rem ? (align - rem) : 0;
    if (pad > mapping_.bytes - hdr) throw std::runtime_error("gguf: aligned data section past end of file");
    const std::size_t data0 = hdr + pad;
    const std::uint64_t avail = std::uint64_t(mapping_.bytes - data0);
    for (auto& t : tensors_) {
        if (t.offset > avail || t.bytes > avail - t.offset)
            throw std::runtime_error("gguf: tensor " + t.name + " past the end");
        t.data = base + data0 + std::size_t(t.offset);
    }
}

GgufFile::~GgufFile() = default;

const GgufTensor* GgufFile::find(std::string_view name) const noexcept {
    const auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}
const GgufTensor& GgufFile::at(std::string_view name) const {
    const GgufTensor* t = find(name);
    if (!t) throw std::runtime_error("gguf: no tensor " + std::string(name) + " in " + path_.string());
    return *t;
}
bool GgufFile::has(std::string_view key) const noexcept { return kv_.find(key) != kv_.end(); }
std::optional<std::int64_t> GgufFile::get_int(std::string_view key) const {
    const auto it = kv_.find(key);
    if (it == kv_.end() || it->second.type == T_STR || it->second.type == T_ARR || it->second.type == T_F32 ||
        it->second.type == T_F64)
        return std::nullopt;
    return it->second.i;
}
std::optional<double> GgufFile::get_float(std::string_view key) const {
    const auto it = kv_.find(key);
    if (it == kv_.end()) return std::nullopt;
    if (it->second.type == T_F32 || it->second.type == T_F64) return it->second.f;
    if (it->second.type == T_STR || it->second.type == T_ARR) return std::nullopt;
    return double(it->second.i);
}
std::optional<std::string> GgufFile::get_string(std::string_view key) const {
    const auto it = kv_.find(key);
    if (it == kv_.end() || it->second.type != T_STR) return std::nullopt;
    return it->second.s;
}
std::optional<std::vector<std::int64_t>> GgufFile::get_int_array(std::string_view key) const {
    const auto it = kv_.find(key);
    if (it == kv_.end() || it->second.type != T_ARR || it->second.n_strings) return std::nullopt;
    return it->second.ia;
}
std::int64_t GgufFile::require_int(std::string_view key) const {
    const auto v = get_int(key);
    if (!v) throw std::runtime_error("gguf: missing integer key " + std::string(key));
    return *v;
}

void ggml_dequantize(GgmlType t, const std::byte* src, std::int64_t n, float* y) {
    const auto* b = reinterpret_cast<const std::uint8_t*>(src);
    const int bv = ggml_block_values(t), bb = ggml_block_bytes(t);
    if (!bv || n % bv) throw std::runtime_error("ggml_dequantize: bad length");
    const std::int64_t nb = n / bv;
    switch (t) {
        case GgmlType::F32: std::memcpy(y, src, std::size_t(n) * 4); return;
        case GgmlType::F16:
            for (std::int64_t i = 0; i < n; ++i) y[i] = fp16_to_f32(rd16(src + 2 * i));
            return;
        case GgmlType::BF16:
            for (std::int64_t i = 0; i < n; ++i) y[i] = bf16_to_f32(rd16(src + 2 * i));
            return;
        default: break;
    }
    for (std::int64_t i = 0; i < nb; ++i, b += bb, y += bv) {
        const auto* blk = reinterpret_cast<const std::byte*>(b);
        switch (t) {
            case GgmlType::Q8_0: {
                const float d = fp16_to_f32(rd16(blk));
                for (int j = 0; j < 32; ++j) y[j] = float(std::int8_t(b[2 + j])) * d;
                break;
            }
            case GgmlType::Q4_0: {
                const float d = fp16_to_f32(rd16(blk));
                for (int j = 0; j < 16; ++j) {
                    y[j] = float((b[2 + j] & 0xF) - 8) * d;
                    y[j + 16] = float((b[2 + j] >> 4) - 8) * d;
                }
                break;
            }
            case GgmlType::Q5_0: {
                const float d = fp16_to_f32(rd16(blk));
                std::uint32_t qh;
                std::memcpy(&qh, b + 2, 4);
                for (int j = 0; j < 16; ++j) {
                    const std::uint8_t h0 = ((qh >> j) << 4) & 0x10, h1 = (qh >> (j + 12)) & 0x10;
                    y[j] = float(((b[6 + j] & 0x0F) | h0) - 16) * d;
                    y[j + 16] = float(((b[6 + j] >> 4) | h1) - 16) * d;
                }
                break;
            }
            case GgmlType::IQ4_NL: {
                const float d = fp16_to_f32(rd16(blk));
                for (int j = 0; j < 16; ++j) {
                    y[j] = d * float(kIq4nl[b[2 + j] & 0xf]);
                    y[j + 16] = d * float(kIq4nl[b[2 + j] >> 4]);
                }
                break;
            }
            case GgmlType::Q4_K:
            case GgmlType::Q5_K: {
                const bool q5 = t == GgmlType::Q5_K;
                const float d = fp16_to_f32(rd16(blk)), dmin = fp16_to_f32(rd16(blk + 2));
                const std::uint8_t* sc = b + 4;
                const std::uint8_t* qh = q5 ? b + 16 : nullptr;
                const std::uint8_t* ql = q5 ? b + 48 : b + 16;
                std::uint8_t u1 = 1, u2 = 2;
                for (int j = 0, is = 0; j < 256; j += 64, is += 2, ql += 32, u1 <<= 2, u2 <<= 2) {
                    std::uint8_t s, m;
                    scale_min_k4(is, sc, s, m);
                    const float d1 = d * s, m1 = dmin * m;
                    scale_min_k4(is + 1, sc, s, m);
                    const float d2 = d * s, m2 = dmin * m;
                    for (int l = 0; l < 32; ++l)
                        y[j + l] = d1 * float((ql[l] & 0xF) + (q5 && (qh[l] & u1) ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; ++l)
                        y[j + 32 + l] = d2 * float((ql[l] >> 4) + (q5 && (qh[l] & u2) ? 16 : 0)) - m2;
                }
                break;
            }
            case GgmlType::Q6_K: {
                const std::uint8_t* ql = b;
                const std::uint8_t* qh = b + 128;
                const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(b + 192);
                const float d = fp16_to_f32(rd16(blk + 208));
                float* yy = y;
                for (int n128 = 0; n128 < 256; n128 += 128, yy += 128, ql += 64, qh += 32, sc += 8) {
                    for (int l = 0; l < 32; ++l) {
                        const int is = l / 16;
                        const int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        yy[l] = d * float(sc[is]) * float(q1);
                        yy[l + 32] = d * float(sc[is + 2]) * float(q2);
                        yy[l + 64] = d * float(sc[is + 4]) * float(q3);
                        yy[l + 96] = d * float(sc[is + 6]) * float(q4);
                    }
                }
                break;
            }
            case GgmlType::IQ4_XS: {
                const float d = fp16_to_f32(rd16(blk));
                const std::uint16_t sh = rd16(blk + 2);
                const std::uint8_t* sl = b + 4;
                const std::uint8_t* qs = b + 8;
                for (int ib = 0; ib < 8; ++ib, qs += 16) {
                    const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
                    const float dl = d * float(ls - 32);
                    for (int j = 0; j < 16; ++j) {
                        y[32 * ib + j] = dl * float(kIq4nl[qs[j] & 0xf]);
                        y[32 * ib + j + 16] = dl * float(kIq4nl[qs[j] >> 4]);
                    }
                }
                break;
            }
            case GgmlType::NVFP4: {  // 4 UE4M3 sub-block scales, then per sub-block s: byte k = values 16s+k, 16s+k+8
                for (int s = 0; s < 4; ++s) {
                    const float d = ue4m3_to_f32(b[s]);
                    for (int k = 0; k < 8; ++k) {
                        const std::uint8_t q = b[4 + 8 * s + k];
                        y[16 * s + k] = kE2M1[q & 0xf] * d;
                        y[16 * s + k + 8] = kE2M1[q >> 4] * d;
                    }
                }
                break;
            }
            default:
                throw std::runtime_error(std::string("ggml_dequantize: type not supported: ") + ggml_type_name(t));
        }
    }
}

}  // namespace ingine

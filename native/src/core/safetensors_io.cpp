#include "core/safetensors.h"

#include "core/weight_pack.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ingine {
namespace {

constexpr std::uint64_t kMaxHeaderBytes = 100ull << 20; // the safetensors reference limit

[[noreturn]] void fail(const std::filesystem::path& path, const std::string& what) {
    throw std::runtime_error("safetensors " + path.string() + ": " + what);
}

[[noreturn]] void fail_errno(const std::filesystem::path& path, const std::string& what) {
    fail(path, what + ": " + std::strerror(errno));
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const std::filesystem::path& path) {
    if (b != 0 && a > std::numeric_limits<std::uint64_t>::max() / b) {
        fail(path, "tensor size overflows");
    }
    return a * b;
}

} // namespace

std::size_t safetensors_dtype_size(std::string_view dtype) {
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3" ||
        dtype == "F8_E5M2" || dtype == "F8_E8M0") {
        return 1;
    }
    if (dtype == "I16" || dtype == "U16" || dtype == "F16" || dtype == "BF16") { return 2; }
    if (dtype == "I32" || dtype == "U32" || dtype == "F32") { return 4; }
    if (dtype == "I64" || dtype == "U64" || dtype == "F64") { return 8; }
    throw std::invalid_argument("unsupported safetensors dtype " + std::string(dtype));
}

std::optional<DType> safetensors_to_dtype(std::string_view dtype) {
    if (dtype == "BF16") { return DType::BF16; }
    if (dtype == "F32") { return DType::FP32; }
    if (dtype == "I32") { return DType::I32; }
    if (dtype == "U8") { return DType::U8; }
    if (dtype == "I64") { return DType::I64; }
    if (dtype == "I8") { return DType::I8; }
    if (dtype == "F16") { return DType::FP16; }
    if (dtype == "F8_E4M3") { return DType::FP8_E4M3FN; }
    return std::nullopt;
}

std::int64_t SafetensorsTensor::numel() const noexcept {
    std::int64_t n = 1;
    for (std::int64_t d : shape) { n *= d; }
    return n;
}

DType SafetensorsTensor::ingine_dtype() const {
    const std::optional<DType> d = safetensors_to_dtype(dtype);
    if (!d) { throw std::invalid_argument("tensor " + name + ": no ingine DType for " + dtype); }
    return *d;
}

Tensor SafetensorsTensor::view(void* data) const {
    if (shape.size() > 4) {
        throw std::invalid_argument("tensor " + name + ": rank " + std::to_string(shape.size()) +
                                    " exceeds ingine::Tensor rank 4");
    }
    Tensor t;
    t.data  = data;
    t.dtype = ingine_dtype();
    const std::size_t rank = shape.size();
    for (std::size_t i = 0; i < rank; ++i) {
        const std::int64_t d = shape[rank - 1 - i];
        if (d <= 0 || d > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("tensor " + name + ": dimension " + std::to_string(d) +
                                        " not representable in ingine::Tensor");
        }
        t.ne[i] = static_cast<std::int32_t>(d);
    }
    t.nb[0] = static_cast<std::int64_t>(dtype_size(t.dtype));
    for (int i = 1; i < 4; ++i) { t.nb[i] = t.nb[i - 1] * t.ne[i - 1]; }
    return t;
}

SafetensorsFile::SafetensorsFile(const std::filesystem::path& path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) { fail_errno(path_, "open failed"); }
    try {
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { fail_errno(path_, "fstat failed"); }
        file_size_ = static_cast<std::uint64_t>(st.st_size);
        if (file_size_ >= sizeof(kWeightPackMagic)) {
            char magic[sizeof(kWeightPackMagic)]{};
            read_range(0, sizeof(magic), magic);
            if (std::memcmp(magic, kWeightPackMagic, sizeof(kWeightPackMagic)) == 0) {
                ::close(fd_);
                fd_ = -1;
                pack_        = std::make_unique<WeightPack>(path_);
                file_size_   = pack_->file_size();
                data_offset_ = pack_->data_offset();
                metadata_    = pack_->metadata();
                tensors_.reserve(pack_->entries().size());
                for (const WeightPackEntry& e : pack_->entries()) {
                    SafetensorsTensor t;
                    t.name  = std::string(pack_->entry_name(e));
                    t.dtype = std::string(pack_->entry_dtype(e));
                    t.shape.assign(e.shape, e.shape + e.rank);
                    t.offset = e.data_offset;
                    t.bytes  = e.data_bytes;
                    tensors_.push_back(std::move(t));
                }
                std::sort(tensors_.begin(), tensors_.end(),
                          [](const SafetensorsTensor& a, const SafetensorsTensor& b) {
                              return a.offset != b.offset ? a.offset < b.offset : a.name < b.name;
                          });
                index_.reserve(tensors_.size());
                for (std::size_t i = 0; i < tensors_.size(); ++i) { index_.emplace(tensors_[i].name, i); }
                return;
            }
        }
        (void)::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);

        if (file_size_ < 8) { fail(path_, "file shorter than the 8-byte header length"); }
        unsigned char len_bytes[8];
        read_range(0, 8, len_bytes);
        std::uint64_t header_len = 0;
        for (int i = 7; i >= 0; --i) { header_len = (header_len << 8) | len_bytes[i]; }
        if (header_len > kMaxHeaderBytes || header_len > file_size_ - 8) {
            fail(path_, "invalid header length " + std::to_string(header_len));
        }
        data_offset_ = 8 + header_len;
        std::string header(header_len, '\0');
        read_range(8, header_len, header.data());

        nlohmann::json j;
        try {
            j = nlohmann::json::parse(header);
        } catch (const nlohmann::json::exception& e) {
            fail(path_, std::string("header is not valid JSON: ") + e.what());
        }
        if (!j.is_object()) { fail(path_, "header is not a JSON object"); }

        const std::uint64_t data_size = file_size_ - data_offset_;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (it.key() == "__metadata__") {
                if (!it.value().is_object()) { fail(path_, "__metadata__ is not an object"); }
                for (auto m = it.value().begin(); m != it.value().end(); ++m) {
                    if (!m.value().is_string()) {
                        fail(path_, "__metadata__ value for " + m.key() + " is not a string");
                    }
                    metadata_.emplace(m.key(), m.value().get<std::string>());
                }
                continue;
            }
            const nlohmann::json& e = it.value();
            if (!e.is_object() || !e.contains("dtype") || !e.contains("shape") ||
                !e.contains("data_offsets") || !e["dtype"].is_string() || !e["shape"].is_array() ||
                !e["data_offsets"].is_array() || e["data_offsets"].size() != 2) {
                fail(path_, "malformed entry for tensor " + it.key());
            }
            SafetensorsTensor t;
            t.name  = it.key();
            t.dtype = e["dtype"].get<std::string>();
            for (const nlohmann::json& d : e["shape"]) {
                if (!d.is_number_unsigned()) {
                    fail(path_, "tensor " + t.name + " has a non-integer or negative dimension");
                }
                t.shape.push_back(static_cast<std::int64_t>(d.get<std::uint64_t>()));
            }
            const nlohmann::json& offs = e["data_offsets"];
            if (!offs[0].is_number_unsigned() || !offs[1].is_number_unsigned()) {
                fail(path_, "tensor " + t.name + " has invalid data_offsets");
            }
            const auto begin = offs[0].get<std::uint64_t>();
            const auto end   = offs[1].get<std::uint64_t>();
            if (begin > end || end > data_size) {
                fail(path_, "tensor " + t.name + " data_offsets [" + std::to_string(begin) + ", " +
                                std::to_string(end) + ") outside the data region of " +
                                std::to_string(data_size) + " bytes");
            }
            t.offset = data_offset_ + begin;
            t.bytes  = end - begin;
            std::size_t elem = 0;
            try {
                elem = safetensors_dtype_size(t.dtype);
            } catch (const std::invalid_argument&) {
                elem = 0; // dtype without a whole-byte element: no size check
            }
            if (elem != 0) {
                std::uint64_t expect = elem;
                for (std::int64_t d : t.shape) {
                    expect = checked_mul(expect, static_cast<std::uint64_t>(d), path_);
                }
                if (expect != t.bytes) {
                    fail(path_, "tensor " + t.name + " holds " + std::to_string(t.bytes) +
                                    " bytes, its dtype and shape need " + std::to_string(expect));
                }
            }
            tensors_.push_back(std::move(t));
        }
        std::sort(tensors_.begin(), tensors_.end(),
                  [](const SafetensorsTensor& a, const SafetensorsTensor& b) {
                      return a.offset != b.offset ? a.offset < b.offset : a.name < b.name;
                  });
        index_.reserve(tensors_.size());
        for (std::size_t i = 0; i < tensors_.size(); ++i) { index_.emplace(tensors_[i].name, i); }
    } catch (...) {
        if (fd_ >= 0) { ::close(fd_); }
        fd_ = -1;
        throw;
    }
}

SafetensorsFile::~SafetensorsFile() {
    if (fd_ >= 0) { ::close(fd_); }
}

SafetensorsFile::SafetensorsFile(SafetensorsFile&& other) noexcept
    : path_(std::move(other.path_)), fd_(std::exchange(other.fd_, -1)),
      pack_(std::move(other.pack_)),
      file_size_(other.file_size_), data_offset_(other.data_offset_),
      metadata_(std::move(other.metadata_)), tensors_(std::move(other.tensors_)),
      index_(std::move(other.index_)) {}

SafetensorsFile& SafetensorsFile::operator=(SafetensorsFile&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) { ::close(fd_); }
        path_        = std::move(other.path_);
        fd_          = std::exchange(other.fd_, -1);
        pack_        = std::move(other.pack_);
        file_size_   = other.file_size_;
        data_offset_ = other.data_offset_;
        metadata_    = std::move(other.metadata_);
        tensors_     = std::move(other.tensors_);
        index_       = std::move(other.index_);
    }
    return *this;
}

int SafetensorsFile::fd() const noexcept { return pack_ ? pack_->fd() : fd_; }

const std::byte* SafetensorsFile::mapped_data() const noexcept {
    return pack_ ? pack_->mapped_base() : nullptr;
}

const std::byte* SafetensorsFile::tensor_mapped_ptr(const SafetensorsTensor& tensor) const noexcept {
    if (!pack_ || tensor.offset > file_size_ || tensor.bytes > file_size_ - tensor.offset) {
        return nullptr;
    }
    return pack_->mapped_base() + tensor.offset;
}

std::vector<std::string> SafetensorsFile::names() const {
    std::vector<std::string> out;
    out.reserve(tensors_.size());
    for (const SafetensorsTensor& t : tensors_) { out.push_back(t.name); }
    return out;
}

const SafetensorsTensor* SafetensorsFile::find(std::string_view name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const SafetensorsTensor& SafetensorsFile::at(std::string_view name) const {
    const SafetensorsTensor* t = find(name);
    if (t == nullptr) { fail(path_, "no tensor named " + std::string(name)); }
    return *t;
}

void SafetensorsFile::read_range(std::uint64_t offset, std::uint64_t bytes, void* dst) const {
    if (pack_) {
        pack_->read_range(offset, bytes, dst);
        return;
    }
    if (offset > file_size_ || bytes > file_size_ - offset) {
        fail(path_, "read of " + std::to_string(bytes) + " bytes at " + std::to_string(offset) +
                        " exceeds the file");
    }
    auto* out = static_cast<char*>(dst);
    while (bytes > 0) {
        const std::size_t want =
            static_cast<std::size_t>(std::min<std::uint64_t>(bytes, std::uint64_t(1) << 30));
        const ssize_t got = ::pread(fd_, out, want, static_cast<off_t>(offset));
        if (got < 0) {
            if (errno == EINTR) { continue; }
            fail_errno(path_, "pread failed");
        }
        if (got == 0) { fail(path_, "unexpected end of file"); }
        out += got;
        offset += static_cast<std::uint64_t>(got);
        bytes -= static_cast<std::uint64_t>(got);
    }
}

void SafetensorsFile::read(const SafetensorsTensor& tensor, void* dst) const {
    read_range(tensor.offset, tensor.bytes, dst);
}

std::vector<std::byte> SafetensorsFile::read(std::string_view name) const {
    const SafetensorsTensor& t = at(name);
    std::vector<std::byte> out(t.bytes);
    read(t, out.data());
    return out;
}

SafetensorsModel::SafetensorsModel(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    fs::path index_path;
    fs::path single;
    if (fs::is_directory(path)) {
        if (weight_pack_enabled() && fs::exists(path / "model.ingpack")) {
            single = path / "model.ingpack";
        } else if (fs::exists(path / "model.safetensors.index.json")) {
            index_path = path / "model.safetensors.index.json";
        } else if (fs::exists(path / "model.safetensors")) {
            single = path / "model.safetensors";
        } else if (fs::exists(path / "model.ingpack")) {
            single = path / "model.ingpack";
        } else {
            throw std::runtime_error("safetensors model " + path.string() +
                                     ": neither model.ingpack, model.safetensors.index.json nor model.safetensors");
        }
    } else if (path.extension() == ".json") {
        index_path = path;
    } else {
        single = path;
    }

    if (!single.empty()) {
        files_.push_back(std::make_unique<SafetensorsFile>(single));
        const SafetensorsFile& f = *files_.back();
        for (const SafetensorsTensor& t : f.tensors()) { index_.emplace(t.name, SafetensorsRef{&f, &t}); }
        return;
    }

    std::ifstream in(index_path);
    if (!in) { throw std::runtime_error("cannot open " + index_path.string()); }
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(in);
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(index_path.string() + ": invalid JSON: " + e.what());
    }
    if (!j.is_object() || !j.contains("weight_map") || !j["weight_map"].is_object()) {
        throw std::runtime_error(index_path.string() + ": no weight_map object");
    }
    const fs::path dir = index_path.parent_path();
    std::map<std::string, const SafetensorsFile*, std::less<>> by_name;
    for (auto it = j["weight_map"].begin(); it != j["weight_map"].end(); ++it) {
        if (!it.value().is_string()) {
            throw std::runtime_error(index_path.string() + ": weight_map entry " + it.key() +
                                     " is not a file name");
        }
        const std::string file_name = it.value().get<std::string>();
        auto f                      = by_name.find(file_name);
        if (f == by_name.end()) {
            files_.push_back(std::make_unique<SafetensorsFile>(dir / file_name));
            f = by_name.emplace(file_name, files_.back().get()).first;
        }
        const SafetensorsTensor* t = f->second->find(it.key());
        if (t == nullptr) {
            throw std::runtime_error(index_path.string() + ": " + it.key() + " is not in " +
                                     file_name);
        }
        index_.emplace(it.key(), SafetensorsRef{f->second, t});
    }
}

std::vector<std::string> SafetensorsModel::names() const {
    std::vector<std::string> out;
    out.reserve(index_.size());
    for (const auto& [name, ref] : index_) { out.push_back(name); }
    return out;
}

SafetensorsRef SafetensorsModel::find(std::string_view name) const {
    const auto it = index_.find(name);
    return it == index_.end() ? SafetensorsRef{} : it->second;
}

SafetensorsRef SafetensorsModel::at(std::string_view name) const {
    const SafetensorsRef ref = find(name);
    if (!ref) { throw std::runtime_error("safetensors model: no tensor named " + std::string(name)); }
    return ref;
}

std::vector<std::byte> SafetensorsModel::read(std::string_view name) const {
    const SafetensorsRef ref = at(name);
    std::vector<std::byte> out(ref->bytes);
    ref.file->read(*ref.tensor, out.data());
    return out;
}

UploadRequest upload_request(const SafetensorsFile& file, const SafetensorsTensor& tensor,
                             void* dst) {
    return UploadRequest{&file, tensor.offset, tensor.bytes, dst};
}

UploadRequest upload_request(SafetensorsRef ref, void* dst) {
    if (!ref) { throw std::invalid_argument("upload_request: empty tensor reference"); }
    return upload_request(*ref.file, *ref.tensor, dst);
}

} // namespace ingine

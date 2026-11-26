// GGUF (v2/v3) model file reader: metadata key/values and the tensor table over a read-only mmap.
//
// ingine's own loaders read Hugging Face safetensors; this reader lets `load_weights_gguf` (src/model) serve a
// llama.cpp-format Inkling-2-Mini file (e.g. unsloth's Inkling-2-Mini-UD-IQ4_XS.gguf: NVFP4 experts in layers 0-19,
// K-quant / legacy-quant experts in 20-23, Q8_0 / K-quant dense projections) with no conversion step.
//
// Shapes follow ggml: `ne[0]` is the innermost (contiguous) dimension, so a row-major [rows, cols] matrix has
// ne = {cols, rows}. Quantized types store whole blocks of `ggml_block_values(type)` values along ne[0].
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ingine {

// ggml_type ids (ggml.h), the subset Inkling-2-Mini GGUF files use plus their common siblings.
enum class GgmlType : std::uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30,
    MXFP4 = 39, NVFP4 = 40,
};

// Values per block and bytes per block (0 if the type is not supported here).
int ggml_block_values(GgmlType t) noexcept;
int ggml_block_bytes(GgmlType t) noexcept;
const char* ggml_type_name(GgmlType t) noexcept;
// Bytes of a row of n values (n a multiple of the block size; throws otherwise).
std::size_t ggml_row_bytes(GgmlType t, std::int64_t n);

struct GgufTensor {
    std::string name;
    GgmlType type = GgmlType::F32;
    std::vector<std::int64_t> ne;  // ggml order, innermost first
    std::uint64_t offset = 0;      // from the start of the data section
    std::uint64_t bytes = 0;
    const std::byte* data = nullptr;  // into the mapping

    std::int64_t numel() const noexcept;
    std::int64_t row_len() const noexcept { return ne.empty() ? 1 : ne[0]; }   // values per row
    std::int64_t rows() const noexcept { return row_len() ? numel() / row_len() : 0; }
    std::size_t row_bytes() const { return ggml_row_bytes(type, row_len()); }
    std::span<const std::byte> bytes_span() const noexcept { return {data, static_cast<std::size_t>(bytes)}; }
    const std::byte* row(std::int64_t r) const { return data + std::size_t(r) * row_bytes(); }
};

class GgufFile {
public:
    explicit GgufFile(const std::filesystem::path& path);
    ~GgufFile();
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    const std::filesystem::path& path() const noexcept { return path_; }
    std::uint32_t version() const noexcept { return version_; }
    const std::vector<GgufTensor>& tensors() const noexcept { return tensors_; }
    const GgufTensor* find(std::string_view name) const noexcept;
    const GgufTensor& at(std::string_view name) const;  // throws std::runtime_error naming the tensor

    // Metadata. Integer getters accept any integer/bool type; arrays of integers are widened to int64.
    bool has(std::string_view key) const noexcept;
    std::optional<std::int64_t> get_int(std::string_view key) const;
    std::optional<double> get_float(std::string_view key) const;
    std::optional<std::string> get_string(std::string_view key) const;
    std::optional<std::vector<std::int64_t>> get_int_array(std::string_view key) const;
    std::int64_t require_int(std::string_view key) const;

private:
    struct Value {
        std::uint32_t type = 0;  // gguf_type
        std::int64_t i = 0;
        double f = 0;
        std::string s;
        std::vector<std::int64_t> ia;
        std::vector<double> fa;
        std::size_t n_strings = 0;
    };
    std::filesystem::path path_;
    std::uint32_t version_ = 0;
    void* map_ = nullptr;
    std::size_t map_bytes_ = 0;
    std::map<std::string, Value, std::less<>> kv_;
    std::vector<GgufTensor> tensors_;
    std::map<std::string, std::size_t, std::less<>> index_;
};

// Host-side dequantization of `n` values starting at a block boundary of `src` (a row or a run of rows) into FP32,
// exactly as ggml's dequantize_row_* (the products in FP32). Supports F32, F16, BF16, Q8_0, Q4_0, Q5_0, IQ4_NL,
// Q4_K, Q5_K, Q6_K, IQ4_XS and NVFP4 (per-block UE4M3 scales; the tensor's ".scale" factor is not applied).
void ggml_dequantize(GgmlType t, const std::byte* src, std::int64_t n, float* dst);

}  // namespace ingine

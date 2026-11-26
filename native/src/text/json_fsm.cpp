#include "text/json_fsm.h"

#include "text/chat.h"

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ingine::text {
namespace {

using Json = nlohmann::ordered_json;
using Int128 = __int128;
using UInt128 = unsigned __int128;

constexpr int kMaxStackDepth = 64;
constexpr std::size_t kMaxUnionBranches = 16;
constexpr std::size_t kMaxNumLen = 128;
constexpr std::size_t kMaxOneOfBuf = 1024 * 1024;
constexpr double kMaxExactDoubleInt = 9007199254740992.0;  // 2^53

inline bool is_json_ws(std::uint8_t c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

inline bool is_hex(std::uint8_t c) noexcept {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

inline std::uint16_t hex_val(std::uint8_t c) noexcept {
  if (c >= '0' && c <= '9') return std::uint16_t(c - '0');
  if (c >= 'a' && c <= 'f') return std::uint16_t(10 + (c - 'a'));
  return std::uint16_t(10 + (c - 'A'));
}

inline bool is_digit(std::uint8_t c) noexcept { return c >= '0' && c <= '9'; }

inline void set_bit(std::uint32_t* words, std::size_t word_count, std::int32_t id) noexcept {
  if (id < 0) return;
  const std::size_t w = std::size_t(id) >> 5u;
  if (w < word_count) words[w] |= (std::uint32_t(1u) << (std::uint32_t(id) & 31u));
}

// Encode a Unicode scalar value (not a surrogate, <= 0x10FFFF) as UTF-8.
void append_utf8(std::string& out, std::uint32_t cp) {
  if (cp <= 0x7Fu) {
    out.push_back(char(cp));
  } else if (cp <= 0x7FFu) {
    out.push_back(char(0xC0u | (cp >> 6u)));
    out.push_back(char(0x80u | (cp & 0x3Fu)));
  } else if (cp <= 0xFFFFu) {
    out.push_back(char(0xE0u | (cp >> 12u)));
    out.push_back(char(0x80u | ((cp >> 6u) & 0x3Fu)));
    out.push_back(char(0x80u | (cp & 0x3Fu)));
  } else {
    out.push_back(char(0xF0u | (cp >> 18u)));
    out.push_back(char(0x80u | ((cp >> 12u) & 0x3Fu)));
    out.push_back(char(0x80u | ((cp >> 6u) & 0x3Fu)));
    out.push_back(char(0x80u | (cp & 0x3Fu)));
  }
}

// Count Unicode scalar values in a valid UTF-8 string (for JSON Schema minLength / maxLength).
std::size_t utf8_codepoint_length(std::string_view s) noexcept {
  std::size_t count = 0;
  for (char ch : s) {
    if ((static_cast<std::uint8_t>(ch) & 0xC0u) != 0x80u) ++count;
  }
  return count;
}

// Decode raw JSON string content (between quotes) to canonical UTF-8; rejects invalid escapes or lone surrogates.
bool unescape_json_string(std::string_view raw, std::string& out) {
  out.clear();
  out.reserve(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const auto c = static_cast<std::uint8_t>(raw[i]);
    if (c != '\\') {
      out.push_back(char(c));
      continue;
    }
    if (i + 1 >= raw.size()) return false;
    const auto esc = static_cast<std::uint8_t>(raw[++i]);
    switch (esc) {
      case '"': out.push_back('"'); break;
      case '\\': out.push_back('\\'); break;
      case '/': out.push_back('/'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      case 'u': {
        if (i + 4 >= raw.size()) return false;
        std::uint16_t u1 = 0;
        for (int k = 0; k < 4; ++k) {
          const auto hc = static_cast<std::uint8_t>(raw[++i]);
          if (!is_hex(hc)) return false;
          u1 = std::uint16_t((u1 << 4u) | hex_val(hc));
        }
        if (u1 >= 0xDC00u && u1 <= 0xDFFFu) return false;
        if (u1 >= 0xD800u && u1 <= 0xDBFFu) {
          if (i + 6 >= raw.size() || raw[i + 1] != '\\' || raw[i + 2] != 'u') return false;
          i += 2;
          std::uint16_t u2 = 0;
          for (int k = 0; k < 4; ++k) {
            const auto hc = static_cast<std::uint8_t>(raw[++i]);
            if (!is_hex(hc)) return false;
            u2 = std::uint16_t((u2 << 4u) | hex_val(hc));
          }
          if (u2 < 0xDC00u || u2 > 0xDFFFu) return false;
          const std::uint32_t cp = 0x10000u + ((std::uint32_t(u1 - 0xD800u) << 10u) | std::uint32_t(u2 - 0xDC00u));
          append_utf8(out, cp);
        } else {
          append_utf8(out, u1);
        }
        break;
      }
      default:
        return false;
    }
  }
  return true;
}

// Check whether `b` is a non-empty, complete, valid UTF-8 sequence containing no JSON control chars, '"', or '\\'.
bool is_complete_string_safe_utf8(std::string_view b) noexcept {
  if (b.empty()) return false;
  std::uint8_t rem = 0;
  std::uint8_t min_next = 0x80;
  std::uint8_t max_next = 0xBF;
  for (char ch : b) {
    const auto u = static_cast<std::uint8_t>(ch);
    if (rem > 0) {
      if (u < min_next || u > max_next) return false;
      --rem;
      min_next = 0x80;
      max_next = 0xBF;
      continue;
    }
    if (u < 0x20 || u == '"' || u == '\\') return false;
    if (u < 0x80) continue;
    if (u >= 0xC2 && u <= 0xDF) {
      rem = 1;
      min_next = 0x80;
      max_next = 0xBF;
    } else if (u == 0xE0) {
      rem = 2;
      min_next = 0xA0;
      max_next = 0xBF;
    } else if ((u >= 0xE1 && u <= 0xEC) || (u >= 0xEE && u <= 0xEF)) {
      rem = 2;
      min_next = 0x80;
      max_next = 0xBF;
    } else if (u == 0xED) {
      rem = 2;
      min_next = 0x80;
      max_next = 0x9F;
    } else if (u == 0xF0) {
      rem = 3;
      min_next = 0x90;
      max_next = 0xBF;
    } else if (u >= 0xF1 && u <= 0xF3) {
      rem = 3;
      min_next = 0x80;
      max_next = 0xBF;
    } else if (u == 0xF4) {
      rem = 3;
      min_next = 0x80;
      max_next = 0x8F;
    } else {
      return false;
    }
  }
  return rem == 0;
}

// ---- Exact Arbitrary-Precision Decimal Comparison & JSON Schema 4.2.2 Equality ----------------

// Normalized decimal: value == (-1)^neg * sig * 10^scale10 (sig has no leading or trailing zeros).
struct ParsedDecimal {
  bool neg = false;
  bool is_zero = false;
  std::string sig;
  std::int64_t scale10 = 0;
};

bool parse_json_decimal(std::string_view s, ParsedDecimal& out) noexcept {
  out = ParsedDecimal{};
  if (s.empty()) return false;
  std::size_t pos = 0;
  if (s[pos] == '-') {
    out.neg = true;
    ++pos;
  }
  if (pos >= s.size()) return false;
  std::string raw_digits;
  raw_digits.reserve(s.size());
  while (pos < s.size() && is_digit(static_cast<std::uint8_t>(s[pos]))) {
    raw_digits.push_back(s[pos++]);
  }
  if (raw_digits.empty()) return false;
  std::int64_t frac_len = 0;
  if (pos < s.size() && s[pos] == '.') {
    ++pos;
    std::size_t fstart = pos;
    while (pos < s.size() && is_digit(static_cast<std::uint8_t>(s[pos]))) {
      raw_digits.push_back(s[pos++]);
    }
    if (pos == fstart) return false;
    frac_len = std::int64_t(pos - fstart);
  }
  std::int64_t exp_val = 0;
  if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
    ++pos;
    bool exp_neg = false;
    if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) {
      exp_neg = (s[pos] == '-');
      ++pos;
    }
    if (pos >= s.size() || !is_digit(static_cast<std::uint8_t>(s[pos]))) return false;
    std::int64_t mag = 0;
    while (pos < s.size() && is_digit(static_cast<std::uint8_t>(s[pos]))) {
      if (mag < 1000000000LL) mag = mag * 10 + (s[pos] - '0');
      ++pos;
    }
    exp_val = exp_neg ? -mag : mag;
  }
  if (pos != s.size()) return false;

  std::size_t first_nz = 0;
  while (first_nz < raw_digits.size() && raw_digits[first_nz] == '0') ++first_nz;
  if (first_nz == raw_digits.size()) {
    out.is_zero = true;
    out.neg = false;
    return true;
  }
  std::size_t last_nz = raw_digits.size();
  while (last_nz > first_nz && raw_digits[last_nz - 1] == '0') --last_nz;
  const std::int64_t stripped_trailing = std::int64_t(raw_digits.size() - last_nz);
  out.sig = raw_digits.substr(first_nz, last_nz - first_nz);
  out.scale10 = exp_val - frac_len + stripped_trailing;
  return true;
}

// Exact comparison of two normalized decimals (-1, 0, +1): zero floating-point rounding!
int compare_decimals(const ParsedDecimal& a, const ParsedDecimal& b) noexcept {
  if (a.is_zero && b.is_zero) return 0;
  if (a.is_zero) return b.neg ? 1 : -1;
  if (b.is_zero) return a.neg ? -1 : 1;
  if (a.neg != b.neg) return a.neg ? -1 : 1;

  const std::int64_t ea = std::int64_t(a.sig.size()) - 1 + a.scale10;
  const std::int64_t eb = std::int64_t(b.sig.size()) - 1 + b.scale10;
  int mag_cmp = 0;
  if (ea != eb) {
    mag_cmp = (ea < eb) ? -1 : 1;
  } else {
    const int c = a.sig.compare(b.sig);
    mag_cmp = (c < 0) ? -1 : (c > 0 ? 1 : 0);
  }
  return a.neg ? -mag_cmp : mag_cmp;
}

// Per JSON Schema 2020-12 section 4.2.2: 3, 3.0, 3e0, 30e-1, -0.0 are all mathematically integers.
bool json_decimal_is_integral(std::string_view s) noexcept {
  ParsedDecimal dec;
  if (!parse_json_decimal(s, dec)) return false;
  return dec.is_zero || dec.scale10 >= 0;
}

// Parse pure integer text (no '.' or 'e') into Int128; sets overflow_dir (-1 or +1) if outside Int128.
bool parse_pure_int128(std::string_view buf, Int128& val_out, int& overflow_dir) noexcept {
  overflow_dir = 0;
  if (buf.empty()) return false;
  std::size_t pos = 0;
  bool neg = false;
  if (buf[pos] == '-') {
    neg = true;
    ++pos;
  }
  if (pos >= buf.size()) return false;
  UInt128 acc = 0;
  constexpr UInt128 kLimitPos = (UInt128(1) << 126u) - 1u + (UInt128(1) << 126u);
  constexpr UInt128 kLimitNeg = (UInt128(1) << 127u);
  const UInt128 limit = neg ? kLimitNeg : kLimitPos;
  for (; pos < buf.size(); ++pos) {
    if (!is_digit(static_cast<std::uint8_t>(buf[pos]))) return false;
    const unsigned int d = unsigned(buf[pos] - '0');
    if (acc > (limit - d) / 10u) {
      overflow_dir = neg ? -1 : 1;
      return true;
    }
    acc = acc * 10u + d;
  }
  val_out = neg ? -Int128(acc) : Int128(acc);
  return true;
}

// Exact floor(d) or ceil(d) of a ParsedDecimal into Int128 (returns nullopt if outside Int128 range).
std::optional<Int128> decimal_to_int128_round(const ParsedDecimal& dec, bool round_up) noexcept {
  if (dec.is_zero) return Int128(0);
  const std::int64_t int_digits = std::int64_t(dec.sig.size()) + dec.scale10;
  if (int_digits <= 0) {
    // 0 < |dec| < 1
    if (dec.neg) return round_up ? Int128(0) : Int128(-1);
    return round_up ? Int128(1) : Int128(0);
  }
  if (int_digits > 37) return std::nullopt;
  const std::size_t nint = std::size_t(int_digits);
  std::string ibuf;
  ibuf.reserve(nint + 1);
  if (dec.neg) ibuf.push_back('-');
  for (std::size_t i = 0; i < nint; ++i) {
    ibuf.push_back(i < dec.sig.size() ? dec.sig[i] : '0');
  }
  Int128 base = 0;
  int ov = 0;
  if (!parse_pure_int128(ibuf, base, ov) || ov != 0) return std::nullopt;
  const bool has_frac = (dec.sig.size() > nint);
  if (has_frac) {
    if (!dec.neg && round_up) base += 1;
    if (dec.neg && !round_up) base -= 1;
  }
  return base;
}

bool compare_num_text_to_bound(std::string_view buf, const ParsedDecimal& b, int& cmp_out) noexcept {
  ParsedDecimal dec;
  if (!parse_json_decimal(buf, dec)) return false;
  cmp_out = compare_decimals(dec, b);
  return true;
}

bool json_to_decimal(const Json& j, ParsedDecimal& out) noexcept {
  if (j.is_boolean() || !j.is_number()) return false;
  return parse_json_decimal(j.dump(), out);
}

bool json_numbers_equal(const Json& a, const Json& b) noexcept {
  ParsedDecimal da, db;
  return json_to_decimal(a, da) && json_to_decimal(b, db) && compare_decimals(da, db) == 0;
}

// JSON Schema 2020-12 section 4.2.2 mathematical / structural equality (key-order-independent objects,
// unescaped strings, mathematical number equality, distinct bool vs number).
bool json_schema_equal(const Json& a, const Json& b) noexcept {
  if (a.is_null() || b.is_null()) return a.is_null() && b.is_null();
  if (a.is_boolean() || b.is_boolean()) return a.is_boolean() && b.is_boolean() && a.get<bool>() == b.get<bool>();
  if (a.is_number() || b.is_number()) return a.is_number() && b.is_number() && json_numbers_equal(a, b);
  if (a.is_string() || b.is_string()) {
    return a.is_string() && b.is_string() && a.get_ref<const std::string&>() == b.get_ref<const std::string&>();
  }
  if (a.is_array() || b.is_array()) {
    if (!a.is_array() || !b.is_array() || a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (!json_schema_equal(a[i], b[i])) return false;
    }
    return true;
  }
  if (a.is_object() || b.is_object()) {
    if (!a.is_object() || !b.is_object() || a.size() != b.size()) return false;
    for (auto it = a.begin(); it != a.end(); ++it) {
      auto bit = b.find(it.key());
      if (bit == b.end() || !json_schema_equal(it.value(), bit.value())) return false;
    }
    return true;
  }
  return false;
}

// ---- Compiled JSON Schema graph ---------------------------------------------------------------

enum class NodeKind : std::uint8_t {
  Any,
  Object,
  Array,
  String,
  Integer,
  Number,
  Boolean,
  Null,
  Enum,
  Union,
};

struct PropDef {
  std::string name;
  std::string json_escaped;  // content between quotes when JSON-escaped (without outer quotes)
  int node_id = 0;
  bool required = false;
};

struct SchemaNode {
  NodeKind kind = NodeKind::Any;
  // Object
  std::vector<PropDef> props;
  bool additional_properties = true;
  int additional_node_id = 0;  // Any
  int min_properties = 0;
  int max_properties = -1;  // -1 = unbounded
  // Array
  int item_node_id = 0;  // Any
  int min_items = 0;
  int max_items = -1;  // -1 = unbounded
  // String
  int min_length = 0;
  int max_length = -1;  // -1 = unbounded
  // Integer / Number (exact arbitrary-precision decimal bounds — zero double rounding)
  std::optional<ParsedDecimal> minimum;
  std::optional<ParsedDecimal> maximum;
  std::optional<ParsedDecimal> exclusive_minimum;
  std::optional<ParsedDecimal> exclusive_maximum;
  // Enum / Const (compact encodings for PDA + exact parsed JSON values for semantic equality)
  std::vector<std::string> enum_literals;
  std::vector<Json> enum_values;
  // Union (anyOf / oneOf / type array)
  std::vector<int> union_nodes;
  bool is_one_of = false;
};

std::string json_escape_inner(const std::string& raw) {
  const std::string dumped = Json(raw).dump();
  if (dumped.size() >= 2 && dumped.front() == '"' && dumped.back() == '"') {
    return dumped.substr(1, dumped.size() - 2);
  }
  return dumped;
}

const Json* resolve_json_pointer(const Json& root, const std::string& ref) {
  if (ref == "#") return &root;
  if (ref.size() < 2 || ref[0] != '#' || ref[1] != '/') return nullptr;
  const Json* cur = &root;
  std::size_t pos = 2;
  while (pos <= ref.size()) {
    std::size_t slash = ref.find('/', pos);
    std::string seg = (slash == std::string::npos) ? ref.substr(pos) : ref.substr(pos, slash - pos);
    std::string unescaped;
    unescaped.reserve(seg.size());
    for (std::size_t i = 0; i < seg.size(); ++i) {
      if (seg[i] == '~' && i + 1 < seg.size()) {
        if (seg[i + 1] == '1') {
          unescaped.push_back('/');
          ++i;
          continue;
        }
        if (seg[i + 1] == '0') {
          unescaped.push_back('~');
          ++i;
          continue;
        }
      }
      unescaped.push_back(seg[i]);
    }
    if (!cur->is_object()) return nullptr;
    auto it = cur->find(unescaped);
    if (it == cur->end()) return nullptr;
    cur = &(*it);
    if (slash == std::string::npos) break;
    pos = slash + 1;
  }
  return cur;
}

bool is_annotation_keyword(std::string_view key) noexcept {
  return key == "$schema" || key == "$id" || key == "id" || key == "$defs" || key == "definitions" ||
         key == "$comment" || key == "$vocabulary" || key == "$anchor" || key == "title" || key == "description" ||
         key == "default" || key == "examples" || key == "readOnly" || key == "writeOnly" || key == "deprecated";
}

bool is_supported_validation_keyword(std::string_view key) noexcept {
  return key == "$ref" || key == "type" || key == "const" || key == "enum" || key == "anyOf" || key == "oneOf" ||
         key == "properties" || key == "required" || key == "additionalProperties" || key == "items" ||
         key == "minimum" || key == "maximum" || key == "exclusiveMinimum" || key == "exclusiveMaximum" ||
         key == "minLength" || key == "maxLength" || key == "minItems" || key == "maxItems" ||
         key == "minProperties" || key == "maxProperties";
}

bool is_valid_json_type_name(std::string_view t) noexcept {
  return t == "object" || t == "array" || t == "string" || t == "integer" || t == "number" || t == "boolean" ||
         t == "null";
}

bool json_matches_type_name(const Json& v, std::string_view t) noexcept {
  if (t == "null") return v.is_null();
  if (t == "boolean") return v.is_boolean();
  if (t == "string") return v.is_string();
  if (t == "array") return v.is_array();
  if (t == "object") return v.is_object();
  if (t == "number") return !v.is_boolean() && v.is_number();
  if (t == "integer") {
    if (v.is_boolean() || !v.is_number()) return false;
    if (v.is_number_integer() || v.is_number_unsigned()) return true;
    const double d = v.get<double>();
    return std::isfinite(d) && std::trunc(d) == d;
  }
  return false;
}

int parse_nonneg_int_kw(const Json& obj, const char* key, int default_val) {
  auto it = obj.find(key);
  if (it == obj.end()) return default_val;
  if (it->is_boolean() || !it->is_number()) {
    throw std::invalid_argument(std::string("JSON schema '") + key + "' must be a non-negative integer");
  }
  if (it->is_number_unsigned()) {
    const std::uint64_t v = it->get<std::uint64_t>();
    if (v > std::uint64_t(std::numeric_limits<int>::max())) {
      throw std::invalid_argument(std::string("JSON schema '") + key + "' out of supported integer range");
    }
    return int(v);
  }
  if (it->is_number_integer()) {
    const std::int64_t v = it->get<std::int64_t>();
    if (v < 0 || v > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::string("JSON schema '") + key + "' out of supported integer range");
    }
    return int(v);
  }
  const double dv = it->get<double>();
  if (!std::isfinite(dv) || std::trunc(dv) != dv || dv < 0.0 || dv > double(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string("JSON schema '") + key + "' must be a non-negative integer");
  }
  return int(dv);
}

std::optional<ParsedDecimal> parse_exact_bound_kw(const Json& obj, const char* key) {
  auto it = obj.find(key);
  if (it == obj.end()) return std::nullopt;
  if (it->is_boolean() || !it->is_number()) {
    throw std::invalid_argument(std::string("JSON schema '") + key + "' must be a number");
  }
  if (it->is_number_float()) {
    const double d = it->get<double>();
    if (!std::isfinite(d)) {
      throw std::invalid_argument(std::string("JSON schema '") + key + "' must be finite");
    }
    if (std::abs(d) >= kMaxExactDoubleInt) {
      throw std::invalid_argument(std::string("JSON schema '") + key +
                                  "' floating-point bound exceeds exact IEEE-754 integer precision (>= 2^53)");
    }
  }
  ParsedDecimal dec;
  if (!json_to_decimal(*it, dec)) {
    throw std::invalid_argument(std::string("JSON schema '") + key + "' could not be parsed as an exact decimal");
  }
  return dec;
}

struct NumBounds {
  std::optional<ParsedDecimal> minimum;
  std::optional<ParsedDecimal> maximum;
  std::optional<ParsedDecimal> exclusive_minimum;
  std::optional<ParsedDecimal> exclusive_maximum;
};

NumBounds parse_num_bounds(const Json& obj) {
  NumBounds b;
  b.minimum = parse_exact_bound_kw(obj, "minimum");
  b.maximum = parse_exact_bound_kw(obj, "maximum");
  b.exclusive_minimum = parse_exact_bound_kw(obj, "exclusiveMinimum");
  b.exclusive_maximum = parse_exact_bound_kw(obj, "exclusiveMaximum");
  return b;
}

// Compute exact tight Int128 [lo, hi] bounds for integer values (or nullopt if unbounded on that side).
void compute_integer_bounds(const std::optional<ParsedDecimal>& minimum, const std::optional<ParsedDecimal>& maximum,
                            const std::optional<ParsedDecimal>& exclusive_minimum,
                            const std::optional<ParsedDecimal>& exclusive_maximum, std::optional<Int128>& eff_lo,
                            std::optional<Int128>& eff_hi) noexcept {
  eff_lo.reset();
  eff_hi.reset();
  if (minimum) {
    if (auto v = decimal_to_int128_round(*minimum, /*round_up=*/true)) {
      eff_lo = eff_lo ? std::max(*eff_lo, *v) : *v;
    }
  }
  if (exclusive_minimum) {
    if (auto f = decimal_to_int128_round(*exclusive_minimum, /*round_up=*/false)) {
      const Int128 v = *f + 1;
      eff_lo = eff_lo ? std::max(*eff_lo, v) : v;
    }
  }
  if (maximum) {
    if (auto v = decimal_to_int128_round(*maximum, /*round_up=*/false)) {
      eff_hi = eff_hi ? std::min(*eff_hi, *v) : *v;
    }
  }
  if (exclusive_maximum) {
    if (auto c = decimal_to_int128_round(*exclusive_maximum, /*round_up=*/true)) {
      const Int128 v = *c - 1;
      eff_hi = eff_hi ? std::min(*eff_hi, v) : v;
    }
  }
}

void validate_num_bounds_satisfiable(const NumBounds& b, bool integer_only) {
  std::optional<ParsedDecimal> lo;
  bool lo_exc = false;
  if (b.minimum) {
    lo = *b.minimum;
    lo_exc = false;
  }
  if (b.exclusive_minimum) {
    if (!lo || compare_decimals(*b.exclusive_minimum, *lo) >= 0) {
      lo = *b.exclusive_minimum;
      lo_exc = true;
    }
  }
  std::optional<ParsedDecimal> hi;
  bool hi_exc = false;
  if (b.maximum) {
    hi = *b.maximum;
    hi_exc = false;
  }
  if (b.exclusive_maximum) {
    if (!hi || compare_decimals(*b.exclusive_maximum, *hi) <= 0) {
      hi = *b.exclusive_maximum;
      hi_exc = true;
    }
  }
  if (lo && hi) {
    const int c = compare_decimals(*lo, *hi);
    if (c > 0 || (c == 0 && (lo_exc || hi_exc))) {
      throw std::invalid_argument("unsatisfiable JSON schema numeric bounds");
    }
  }
  if (integer_only) {
    std::optional<Int128> ilo, ihi;
    compute_integer_bounds(b.minimum, b.maximum, b.exclusive_minimum, b.exclusive_maximum, ilo, ihi);
    if (ilo && ihi && *ilo > *ihi) {
      throw std::invalid_argument("unsatisfiable JSON schema integer bounds");
    }
  }
}

bool num_value_satisfies(const SchemaNode& node, std::string_view buf) noexcept {
  if (!node.minimum && !node.maximum && !node.exclusive_minimum && !node.exclusive_maximum) return true;
  if (buf.empty()) return false;
  int cmp = 0;
  if (node.minimum) {
    if (!compare_num_text_to_bound(buf, *node.minimum, cmp) || cmp < 0) return false;
  }
  if (node.exclusive_minimum) {
    if (!compare_num_text_to_bound(buf, *node.exclusive_minimum, cmp) || cmp <= 0) return false;
  }
  if (node.maximum) {
    if (!compare_num_text_to_bound(buf, *node.maximum, cmp) || cmp > 0) return false;
  }
  if (node.exclusive_maximum) {
    if (!compare_num_text_to_bound(buf, *node.exclusive_maximum, cmp) || cmp >= 0) return false;
  }
  return true;
}

bool json_number_satisfies_bounds(const Json& v, const SchemaNode& node) noexcept {
  if (!node.minimum && !node.maximum && !node.exclusive_minimum && !node.exclusive_maximum) return true;
  ParsedDecimal vb;
  if (!json_to_decimal(v, vb)) return false;
  if (node.minimum && compare_decimals(vb, *node.minimum) < 0) return false;
  if (node.exclusive_minimum && compare_decimals(vb, *node.exclusive_minimum) <= 0) return false;
  if (node.maximum && compare_decimals(vb, *node.maximum) > 0) return false;
  if (node.exclusive_maximum && compare_decimals(vb, *node.exclusive_maximum) >= 0) return false;
  return true;
}

// Sound prefix check for numbers/integers: never rejects a prefix with any valid completion.
bool num_prefix_can_satisfy(const SchemaNode& node, const std::string& buf) noexcept {
  if (!node.minimum && !node.maximum && !node.exclusive_minimum && !node.exclusive_maximum) return true;
  if (buf.empty()) return true;

  if (node.kind == NodeKind::Integer) {
    std::optional<Int128> eff_lo, eff_hi;
    compute_integer_bounds(node.minimum, node.maximum, node.exclusive_minimum, node.exclusive_maximum, eff_lo, eff_hi);
    if (buf == "-") return !eff_lo || *eff_lo <= -1;
    if (buf == "-0" || buf == "0") return (!eff_lo || 0 >= *eff_lo) && (!eff_hi || 0 <= *eff_hi);

    Int128 v = 0;
    int ov = 0;
    if (!parse_pure_int128(buf, v, ov)) return false;
    if (ov > 0) return !eff_hi;
    if (ov < 0) return !eff_lo;

    constexpr Int128 kMaxMul10 = (Int128(1) << 120);
    if (v > 0) {
      if (eff_hi && v > *eff_hi) return false;
      if (!eff_lo || v >= *eff_lo) return true;
      if (!eff_hi) return true;
      Int128 p_lo = v;
      Int128 p_hi = v;
      for (int step = 0; step < 38 && p_lo <= *eff_hi; ++step) {
        if (p_hi >= *eff_lo) return true;
        if (p_lo > kMaxMul10) break;
        p_lo = p_lo * 10;
        p_hi = (p_hi > kMaxMul10) ? (Int128(1) << 125) : (p_hi * 10 + 9);
      }
      return false;
    } else {
      if (eff_lo && v < *eff_lo) return false;
      if (!eff_hi || v <= *eff_hi) return true;
      if (!eff_lo) return true;
      Int128 p_lo = v;
      Int128 p_hi = v;
      for (int step = 0; step < 38 && p_hi >= *eff_lo; ++step) {
        if (p_lo <= *eff_hi) return true;
        if (p_hi < -kMaxMul10) break;
        p_hi = p_hi * 10;
        p_lo = (p_lo < -kMaxMul10) ? -(Int128(1) << 125) : (p_lo * 10 - 9);
      }
      return false;
    }
  }

  // Floating-point Number: sign feasibility check.
  ParsedDecimal zero;
  zero.is_zero = true;
  if (buf[0] == '-') {
    if (node.minimum && compare_decimals(*node.minimum, zero) > 0) return false;
    if (node.exclusive_minimum && compare_decimals(*node.exclusive_minimum, zero) >= 0) return false;
    if (buf.size() >= 2 && buf[1] >= '1' && buf[1] <= '9') {
      if (node.minimum && compare_decimals(*node.minimum, zero) >= 0) return false;
    }
  } else {
    if (node.maximum && compare_decimals(*node.maximum, zero) < 0) return false;
    if (node.exclusive_maximum && compare_decimals(*node.exclusive_maximum, zero) <= 0) return false;
    if (buf[0] >= '1' && buf[0] <= '9') {
      if (node.maximum && compare_decimals(*node.maximum, zero) <= 0) return false;
    }
  }
  return true;
}

struct SchemaGraph {
  std::vector<SchemaNode> nodes;
  int root_id = 0;

  SchemaGraph() {
    nodes.push_back(SchemaNode{NodeKind::Any});
  }

  int add_node(SchemaNode n) {
    const int id = int(nodes.size());
    nodes.push_back(std::move(n));
    return id;
  }

  int compile_type_name(const std::string& t, const Json& obj, const Json& root, int depth) {
    if (t == "object") {
      SchemaNode n;
      n.kind = NodeKind::Object;
      n.min_properties = parse_nonneg_int_kw(obj, "minProperties", 0);
      n.max_properties = parse_nonneg_int_kw(obj, "maxProperties", -1);
      if (n.max_properties >= 0 && n.min_properties > n.max_properties) {
        throw std::invalid_argument("unsatisfiable JSON schema: minProperties > maxProperties");
      }

      std::unordered_set<std::string> req_set;
      if (auto it = obj.find("required"); it != obj.end()) {
        if (!it->is_array()) throw std::invalid_argument("JSON schema 'required' must be an array");
        for (const Json& r : *it) {
          if (!r.is_string()) throw std::invalid_argument("JSON schema 'required' entries must be strings");
          req_set.insert(r.get<std::string>());
        }
      }
      if (auto it = obj.find("additionalProperties"); it != obj.end()) {
        if (it->is_boolean()) {
          n.additional_properties = it->get<bool>();
          n.additional_node_id = 0;
        } else if (it->is_object()) {
          n.additional_properties = true;
          n.additional_node_id = compile(*it, root, depth + 1);
        } else {
          throw std::invalid_argument("JSON schema 'additionalProperties' must be a boolean or object");
        }
      }
      if (auto it = obj.find("properties"); it != obj.end()) {
        if (!it->is_object()) throw std::invalid_argument("JSON schema 'properties' must be an object");
        for (auto pit = it->begin(); pit != it->end(); ++pit) {
          PropDef pd;
          pd.name = pit.key();
          pd.json_escaped = json_escape_inner(pd.name);
          pd.node_id = compile(pit.value(), root, depth + 1);
          pd.required = req_set.count(pd.name) > 0;
          n.props.push_back(std::move(pd));
        }
      }
      for (const std::string& req : req_set) {
        bool found = false;
        for (const PropDef& pd : n.props) {
          if (pd.name == req) {
            found = true;
            break;
          }
        }
        if (!found) {
          if (!n.additional_properties) {
            throw std::invalid_argument("unsatisfiable JSON schema: required property '" + req +
                                        "' is disallowed by additionalProperties=false");
          }
          n.props.push_back(PropDef{req, json_escape_inner(req), n.additional_node_id, true});
        }
      }
      if (n.max_properties >= 0 && int(req_set.size()) > n.max_properties) {
        throw std::invalid_argument("unsatisfiable JSON schema: required count exceeds maxProperties");
      }
      if (!n.additional_properties && n.min_properties > int(n.props.size())) {
        throw std::invalid_argument("unsatisfiable JSON schema: minProperties exceeds declared properties");
      }
      return add_node(std::move(n));
    }
    if (t == "array") {
      SchemaNode n;
      n.kind = NodeKind::Array;
      n.min_items = parse_nonneg_int_kw(obj, "minItems", 0);
      n.max_items = parse_nonneg_int_kw(obj, "maxItems", -1);
      if (auto it = obj.find("items"); it != obj.end()) {
        if (it->is_object()) {
          n.item_node_id = compile(*it, root, depth + 1);
        } else if (it->is_boolean()) {
          if (it->get<bool>()) {
            n.item_node_id = 0;
          } else {
            n.item_node_id = 0;
            n.max_items = (n.max_items < 0) ? 0 : std::min(n.max_items, 0);
          }
        } else {
          throw std::invalid_argument("JSON schema tuple/array 'items' is not supported");
        }
      } else {
        n.item_node_id = 0;
      }
      if (n.max_items >= 0 && n.min_items > n.max_items) {
        throw std::invalid_argument("unsatisfiable JSON schema: minItems > maxItems");
      }
      return add_node(std::move(n));
    }
    if (t == "string") {
      SchemaNode n;
      n.kind = NodeKind::String;
      n.min_length = parse_nonneg_int_kw(obj, "minLength", 0);
      n.max_length = parse_nonneg_int_kw(obj, "maxLength", -1);
      if (n.max_length >= 0 && n.min_length > n.max_length) {
        throw std::invalid_argument("unsatisfiable JSON schema: minLength > maxLength");
      }
      return add_node(std::move(n));
    }
    if (t == "integer" || t == "number") {
      SchemaNode n;
      n.kind = (t == "integer") ? NodeKind::Integer : NodeKind::Number;
      const NumBounds b = parse_num_bounds(obj);
      validate_num_bounds_satisfiable(b, n.kind == NodeKind::Integer);
      n.minimum = b.minimum;
      n.maximum = b.maximum;
      n.exclusive_minimum = b.exclusive_minimum;
      n.exclusive_maximum = b.exclusive_maximum;
      return add_node(std::move(n));
    }
    if (t == "boolean") return add_node(SchemaNode{NodeKind::Boolean});
    if (t == "null") return add_node(SchemaNode{NodeKind::Null});
    throw std::invalid_argument("unsupported JSON schema type: '" + t + "'");
  }

  int compile(const Json& schema, const Json& root, int depth = 0) {
    if (depth > 24) {
      throw std::invalid_argument("JSON schema exceeds maximum nesting/recursion depth");
    }
    if (schema.is_boolean()) {
      if (schema.get<bool>()) return 0;
      throw std::invalid_argument("unsatisfiable boolean false JSON schema");
    }
    if (!schema.is_object()) {
      throw std::invalid_argument("JSON schema node must be an object or boolean");
    }

    int validation_kw_count = 0;
    for (auto it = schema.begin(); it != schema.end(); ++it) {
      const std::string& k = it.key();
      if (is_annotation_keyword(k)) continue;
      if (!is_supported_validation_keyword(k)) {
        throw std::invalid_argument("unsupported JSON schema keyword: '" + k + "'");
      }
      ++validation_kw_count;
    }

    if (auto it = schema.find("$ref"); it != schema.end()) {
      if (validation_kw_count > 1) {
        throw std::invalid_argument("JSON schema '$ref' with sibling validation keywords is not supported");
      }
      if (!it->is_string()) {
        throw std::invalid_argument("JSON schema '$ref' must be a string");
      }
      const std::string ref = it->get<std::string>();
      const Json* target = resolve_json_pointer(root, ref);
      if (!target) {
        throw std::invalid_argument("unresolved or external JSON schema $ref: '" + ref + "'");
      }
      return compile(*target, root, depth + 1);
    }

    const bool has_any_of = schema.contains("anyOf");
    const bool has_one_of = schema.contains("oneOf");
    if (has_any_of || has_one_of) {
      if (has_any_of && has_one_of) {
        throw std::invalid_argument("combining 'anyOf' and 'oneOf' in one schema node is not supported");
      }
      if (validation_kw_count > 1) {
        throw std::invalid_argument("combining 'anyOf'/'oneOf' with sibling validation keywords is not supported");
      }
      const char* ukey = has_one_of ? "oneOf" : "anyOf";
      const Json& arr = schema.at(ukey);
      if (!arr.is_array() || arr.empty()) {
        throw std::invalid_argument(std::string("JSON schema '") + ukey + "' must be a non-empty array");
      }
      SchemaNode n;
      n.kind = NodeKind::Union;
      n.is_one_of = has_one_of;
      for (const Json& sub : arr) {
        n.union_nodes.push_back(compile(sub, root, depth + 1));
      }
      if (n.union_nodes.size() == 1) return n.union_nodes[0];
      return add_node(std::move(n));
    }

    std::vector<std::string> allowed_types;
    if (auto it = schema.find("type"); it != schema.end()) {
      if (it->is_string()) {
        const std::string t = it->get<std::string>();
        if (!is_valid_json_type_name(t)) {
          throw std::invalid_argument("unsupported JSON schema type: '" + t + "'");
        }
        allowed_types.push_back(t);
      } else if (it->is_array() && !it->empty()) {
        for (const Json& t : *it) {
          if (!t.is_string() || !is_valid_json_type_name(t.get<std::string>())) {
            throw std::invalid_argument("JSON schema 'type' array contains invalid type name");
          }
          allowed_types.push_back(t.get<std::string>());
        }
      } else {
        throw std::invalid_argument("JSON schema 'type' must be a valid type string or non-empty array");
      }
    }

    const bool has_const = schema.contains("const");
    const bool has_enum = schema.contains("enum");
    if (has_const || has_enum) {
      if (schema.contains("properties") || schema.contains("required") || schema.contains("additionalProperties") ||
          schema.contains("items")) {
        throw std::invalid_argument("combining 'const'/'enum' with 'properties'/'required'/'items' is not supported");
      }
      std::vector<Json> candidates;
      if (has_const) {
        candidates.push_back(schema.at("const"));
      }
      if (has_enum) {
        const Json& earr = schema.at("enum");
        if (!earr.is_array() || earr.empty()) {
          throw std::invalid_argument("JSON schema 'enum' must be a non-empty array");
        }
        if (has_const) {
          bool found = false;
          for (const Json& ev : earr) {
            if (json_schema_equal(ev, candidates[0])) {
              found = true;
              break;
            }
          }
          if (!found) candidates.clear();
        } else {
          for (const Json& ev : earr) candidates.push_back(ev);
        }
      }

      const NumBounds nb = parse_num_bounds(schema);
      SchemaNode bound_probe;
      bound_probe.minimum = nb.minimum;
      bound_probe.maximum = nb.maximum;
      bound_probe.exclusive_minimum = nb.exclusive_minimum;
      bound_probe.exclusive_maximum = nb.exclusive_maximum;
      const int min_len = parse_nonneg_int_kw(schema, "minLength", 0);
      const int max_len = parse_nonneg_int_kw(schema, "maxLength", -1);
      const int min_items = parse_nonneg_int_kw(schema, "minItems", 0);
      const int max_items = parse_nonneg_int_kw(schema, "maxItems", -1);
      const int min_props = parse_nonneg_int_kw(schema, "minProperties", 0);
      const int max_props = parse_nonneg_int_kw(schema, "maxProperties", -1);

      SchemaNode n;
      n.kind = NodeKind::Enum;
      std::unordered_set<std::string> dedup;
      for (const Json& v : candidates) {
        if (!allowed_types.empty()) {
          bool ok_type = false;
          for (const std::string& t : allowed_types) {
            if (json_matches_type_name(v, t)) {
              ok_type = true;
              break;
            }
          }
          if (!ok_type) continue;
        }
        if (!v.is_boolean() && v.is_number() && !json_number_satisfies_bounds(v, bound_probe)) continue;
        if (v.is_string()) {
          const std::size_t cp_len = utf8_codepoint_length(v.get_ref<const std::string&>());
          if (cp_len < std::size_t(min_len)) continue;
          if (max_len >= 0 && cp_len > std::size_t(max_len)) continue;
        }
        if (v.is_array()) {
          if (int(v.size()) < min_items) continue;
          if (max_items >= 0 && int(v.size()) > max_items) continue;
        }
        if (v.is_object()) {
          if (int(v.size()) < min_props) continue;
          if (max_props >= 0 && int(v.size()) > max_props) continue;
        }
        std::string dumped = v.dump();
        if (dedup.insert(dumped).second) {
          n.enum_literals.push_back(std::move(dumped));
          n.enum_values.push_back(v);
        }
      }
      if (n.enum_literals.empty()) {
        throw std::invalid_argument("unsatisfiable JSON schema: no const/enum value matches constraints");
      }
      return add_node(std::move(n));
    }

    if (!allowed_types.empty()) {
      if (allowed_types.size() == 1) {
        return compile_type_name(allowed_types[0], schema, root, depth);
      }
      SchemaNode n;
      n.kind = NodeKind::Union;
      n.is_one_of = false;
      for (const std::string& t : allowed_types) {
        n.union_nodes.push_back(compile_type_name(t, schema, root, depth));
      }
      return add_node(std::move(n));
    }

    const bool has_obj = schema.contains("properties") || schema.contains("required") ||
                         schema.contains("additionalProperties") || schema.contains("minProperties") ||
                         schema.contains("maxProperties");
    const bool has_arr = schema.contains("items") || schema.contains("minItems") || schema.contains("maxItems");
    const bool has_str = schema.contains("minLength") || schema.contains("maxLength");
    const bool has_num = schema.contains("minimum") || schema.contains("maximum") ||
                         schema.contains("exclusiveMinimum") || schema.contains("exclusiveMaximum");
    if (!has_obj && !has_arr && !has_str && !has_num) return 0;

    if (has_obj && !has_arr && !has_str && !has_num) {
      return compile_type_name("object", schema, root, depth);
    }
    if (has_arr && !has_obj && !has_str && !has_num) {
      return compile_type_name("array", schema, root, depth);
    }
    SchemaNode u;
    u.kind = NodeKind::Union;
    u.is_one_of = false;
    for (const char* t : {"object", "array", "string", "number", "boolean", "null"}) {
      u.union_nodes.push_back(compile_type_name(t, schema, root, depth));
    }
    return add_node(std::move(u));
  }
};

// Exact JSON Schema 2020-12 semantic validator used at OneOf completion boundaries so value-equality
// cases (3.0 / 3e0 integral, "\u0061" == "a", {"b":2,"a":1} == {"a":1,"b":2}) match exact oneOf semantics.
bool json_value_matches_node(const SchemaGraph& g, int node_id, const Json& val, std::string_view raw_text = {}) {
  const SchemaNode& node = g.nodes[std::size_t(node_id)];
  switch (node.kind) {
    case NodeKind::Any:
      return true;
    case NodeKind::Null:
      return val.is_null();
    case NodeKind::Boolean:
      return val.is_boolean();
    case NodeKind::String: {
      if (!val.is_string()) return false;
      const std::size_t len = utf8_codepoint_length(val.get_ref<const std::string&>());
      if (len < std::size_t(node.min_length)) return false;
      if (node.max_length >= 0 && len > std::size_t(node.max_length)) return false;
      return true;
    }
    case NodeKind::Integer: {
      if (val.is_boolean() || !val.is_number()) return false;
      if (!raw_text.empty()) {
        if (!json_decimal_is_integral(raw_text)) return false;
        return num_value_satisfies(node, raw_text);
      }
      if (!json_matches_type_name(val, "integer")) return false;
      return json_number_satisfies_bounds(val, node);
    }
    case NodeKind::Number: {
      if (val.is_boolean() || !val.is_number()) return false;
      if (!raw_text.empty()) return num_value_satisfies(node, raw_text);
      return json_number_satisfies_bounds(val, node);
    }
    case NodeKind::Array: {
      if (!val.is_array()) return false;
      if (int(val.size()) < node.min_items) return false;
      if (node.max_items >= 0 && int(val.size()) > node.max_items) return false;
      for (const Json& elem : val) {
        if (!json_value_matches_node(g, node.item_node_id, elem)) return false;
      }
      return true;
    }
    case NodeKind::Object: {
      if (!val.is_object()) return false;
      if (int(val.size()) < node.min_properties) return false;
      if (node.max_properties >= 0 && int(val.size()) > node.max_properties) return false;
      for (const PropDef& pd : node.props) {
        if (pd.required && !val.contains(pd.name)) return false;
      }
      for (auto it = val.begin(); it != val.end(); ++it) {
        bool matched_prop = false;
        for (const PropDef& pd : node.props) {
          if (pd.name == it.key()) {
            matched_prop = true;
            if (!json_value_matches_node(g, pd.node_id, it.value())) return false;
            break;
          }
        }
        if (!matched_prop) {
          if (!node.additional_properties) return false;
          if (!json_value_matches_node(g, node.additional_node_id, it.value())) return false;
        }
      }
      return true;
    }
    case NodeKind::Enum: {
      for (const Json& ev : node.enum_values) {
        if (json_schema_equal(val, ev)) return true;
      }
      return false;
    }
    case NodeKind::Union: {
      int matched = 0;
      for (int sub_id : node.union_nodes) {
        if (json_value_matches_node(g, sub_id, val, raw_text)) ++matched;
      }
      return node.is_one_of ? (matched == 1) : (matched >= 1);
    }
  }
  return false;
}

bool oneof_raw_matches_exactly_one(const SchemaGraph& g, int oneof_node_id, std::string_view raw_text) noexcept {
  if (raw_text.empty()) return false;
  try {
    const Json parsed = Json::parse(raw_text.begin(), raw_text.end());
    const SchemaNode& node = g.nodes[std::size_t(oneof_node_id)];
    const std::string_view num_raw = (!parsed.is_boolean() && parsed.is_number()) ? raw_text : std::string_view{};
    int matched = 0;
    for (int sub_id : node.union_nodes) {
      if (json_value_matches_node(g, sub_id, parsed, num_raw)) {
        if (++matched > 1) return false;
      }
    }
    return matched == 1;
  } catch (...) {
    return false;
  }
}

// ---- Byte-level Pushdown Automaton ------------------------------------------------------------

enum class FrameKind : std::uint8_t {
  Value,          // expecting start of value for node_id
  ObjFirstOrEnd,  // just saw '{'
  ObjNextKey,     // just saw ',' inside object; must see '"' (no trailing comma)
  ObjKeyStr,      // reading object key string content (after opening '"')
  ObjColon,       // expecting ':' then Value(val_node_id)
  ObjCommaOrEnd,  // after object value; expecting ',' or '}'
  ArrFirstOrEnd,  // just saw '['
  ArrNextVal,     // just saw ',' inside array; must see value
  ArrCommaOrEnd,  // after array value; expecting ',' or ']'
  StrBody,        // inside JSON string value
  Num,            // inside JSON number (integer_only flag + exact bounds)
  Literal,        // matching fixed ASCII literal ("true", "false", "null")
  EnumLit,        // matching one of node.enum_literals
  OneOf,          // parallel exact-one evaluation of oneOf alternatives
};

enum class NumPhase : std::uint8_t {
  AfterMinus,  // saw '-', need '0' or '1'..'9'
  Zero,        // saw '0', can end or '.' or 'e'/'E'
  IntDigits,   // saw [1-9][0-9]*, can digit, '.', 'e'/'E', or end
  AfterDot,    // saw '.', need [0-9]
  FracDigits,  // saw .[0-9]+, can digit, 'e'/'E', or end
  AfterExp,    // saw 'e'/'E', need '+', '-', or [0-9]
  AfterExpSign,// saw 'e[+-]', need [0-9]
  ExpDigits,   // saw e[+-]?[0-9]+, can digit or end
};

struct OneOfState;

struct Frame {
  FrameKind kind = FrameKind::Value;
  int node_id = 0;
  int val_node_id = 0;
  // Object tracking: arbitrary-width bitset for declared properties + canonical sorted key list for open objects
  std::vector<std::uint64_t> seen_prop_words;
  std::vector<std::string> seen_keys;
  int seen_prop_count = 0;
  std::string key_buf;
  // Array tracking
  int arr_count = 0;
  // String / Escape / UTF-8 / Surrogate tracking
  bool in_escape = false;
  bool expect_low_surrogate = false;
  std::uint8_t uni_rem = 0;
  std::uint16_t uni_val = 0;
  std::uint8_t utf8_rem = 0;
  std::uint8_t utf8_min_next = 0x80;
  std::uint8_t utf8_max_next = 0xBF;
  std::size_t str_len = 0;
  // Number tracking
  bool integer_only = false;
  NumPhase num_phase = NumPhase::IntDigits;
  std::string num_buf;
  // Literal / Enum tracking
  const char* lit_ptr = nullptr;
  std::uint16_t lit_pos = 0;
  std::string enum_buf;
  // OneOf tracking (immutable shared sub-automaton vector + raw JSON bytes for exact semantic check)
  std::shared_ptr<const OneOfState> oneof;
  std::string oneof_raw;
};

struct PdaState {
  std::vector<Frame> stack;

  bool is_done() const noexcept { return stack.empty(); }
};

struct OneOfState {
  std::vector<std::vector<PdaState>> alts;
};

inline bool prop_is_seen(const std::vector<std::uint64_t>& words, std::size_t idx) noexcept {
  const std::size_t w = idx >> 6u;
  return w < words.size() && ((words[w] >> (idx & 63u)) & 1ull) != 0ull;
}

inline void prop_mark_seen(std::vector<std::uint64_t>& words, std::size_t idx) {
  const std::size_t w = idx >> 6u;
  if (w >= words.size()) words.resize(w + 1, 0ull);
  words[w] |= (1ull << (idx & 63u));
}

bool obj_required_satisfied(const SchemaNode& node, const std::vector<std::uint64_t>& seen_words,
                            int seen_prop_count) noexcept {
  if (seen_prop_count < node.min_properties) return false;
  for (std::size_t i = 0; i < node.props.size(); ++i) {
    if (node.props[i].required && !prop_is_seen(seen_words, i)) return false;
  }
  return true;
}

bool obj_has_remaining_key(const SchemaNode& node, const std::vector<std::uint64_t>& seen_words,
                           int seen_prop_count) noexcept {
  if (node.max_properties >= 0 && seen_prop_count >= node.max_properties) return false;
  if (node.additional_properties) return true;
  for (std::size_t i = 0; i < node.props.size(); ++i) {
    if (!prop_is_seen(seen_words, i)) return true;
  }
  return false;
}

bool num_can_terminate(NumPhase ph) noexcept {
  return ph == NumPhase::Zero || ph == NumPhase::IntDigits || ph == NumPhase::FracDigits || ph == NumPhase::ExpDigits;
}

bool step_string_byte(Frame& top, std::uint8_t c, int max_len, bool& completed_char, bool& closed_quote) noexcept {
  completed_char = false;
  closed_quote = false;

  if (top.uni_rem > 0) {
    if (!is_hex(c)) return false;
    top.uni_val = std::uint16_t((top.uni_val << 4u) | hex_val(c));
    --top.uni_rem;
    if (top.uni_rem == 0) {
      if (top.expect_low_surrogate) {
        if (top.uni_val < 0xDC00u || top.uni_val > 0xDFFFu) return false;
        top.expect_low_surrogate = false;
        completed_char = true;
      } else {
        if (top.uni_val >= 0xDC00u && top.uni_val <= 0xDFFFu) return false;
        if (top.uni_val >= 0xD800u && top.uni_val <= 0xDBFFu) {
          top.expect_low_surrogate = true;
        } else {
          completed_char = true;
        }
      }
    }
    return true;
  }

  if (top.expect_low_surrogate) {
    if (!top.in_escape) {
      if (c != '\\') return false;
      top.in_escape = true;
      return true;
    }
    if (c != 'u') return false;
    top.in_escape = false;
    top.uni_rem = 4;
    top.uni_val = 0;
    return true;
  }

  if (top.in_escape) {
    if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') {
      top.in_escape = false;
      completed_char = true;
      return true;
    }
    if (c == 'u') {
      top.in_escape = false;
      top.uni_rem = 4;
      top.uni_val = 0;
      return true;
    }
    return false;
  }

  if (top.utf8_rem > 0) {
    if (c < top.utf8_min_next || c > top.utf8_max_next) return false;
    --top.utf8_rem;
    top.utf8_min_next = 0x80;
    top.utf8_max_next = 0xBF;
    if (top.utf8_rem == 0) completed_char = true;
    return true;
  }

  if (c == '"') {
    closed_quote = true;
    return true;
  }
  if (max_len >= 0 && top.str_len >= std::size_t(max_len)) return false;
  if (c == '\\') {
    top.in_escape = true;
    return true;
  }
  if (c < 0x20) return false;
  if (c < 0x80) {
    completed_char = true;
    return true;
  }
  if (c >= 0xC2 && c <= 0xDF) {
    top.utf8_rem = 1;
    top.utf8_min_next = 0x80;
    top.utf8_max_next = 0xBF;
    return true;
  }
  if (c == 0xE0) {
    top.utf8_rem = 2;
    top.utf8_min_next = 0xA0;
    top.utf8_max_next = 0xBF;
    return true;
  }
  if ((c >= 0xE1 && c <= 0xEC) || (c >= 0xEE && c <= 0xEF)) {
    top.utf8_rem = 2;
    top.utf8_min_next = 0x80;
    top.utf8_max_next = 0xBF;
    return true;
  }
  if (c == 0xED) {
    top.utf8_rem = 2;
    top.utf8_min_next = 0x80;
    top.utf8_max_next = 0x9F;
    return true;
  }
  if (c == 0xF0) {
    top.utf8_rem = 3;
    top.utf8_min_next = 0x90;
    top.utf8_max_next = 0xBF;
    return true;
  }
  if (c >= 0xF1 && c <= 0xF3) {
    top.utf8_rem = 3;
    top.utf8_min_next = 0x80;
    top.utf8_max_next = 0xBF;
    return true;
  }
  if (c == 0xF4) {
    top.utf8_rem = 3;
    top.utf8_min_next = 0x80;
    top.utf8_max_next = 0x8F;
    return true;
  }
  return false;
}

bool pda_is_accepting(const SchemaGraph& g, const PdaState& st) noexcept;
bool step_byte_single(const SchemaGraph& g, PdaState& st, std::uint8_t c, std::vector<PdaState>& out_branches,
                      bool allow_empty_ws = true);

bool start_value_on_byte(const SchemaGraph& g, PdaState& st, int node_id, std::uint8_t c,
                         std::vector<PdaState>& out_branches, bool allow_empty_ws = true) {
  if (int(st.stack.size()) > kMaxStackDepth) return false;
  const SchemaNode& node = g.nodes[std::size_t(node_id)];

  if (node.kind == NodeKind::Union) {
    if (node.is_one_of) {
      auto next_oneof = std::make_shared<OneOfState>();
      next_oneof->alts.resize(node.union_nodes.size());
      int alive_alts = 0;
      bool all_alive_done = true;
      for (std::size_t j = 0; j < node.union_nodes.size(); ++j) {
        PdaState sub;
        Frame vf;
        vf.kind = FrameKind::Value;
        vf.node_id = node.union_nodes[j];
        sub.stack.push_back(std::move(vf));
        start_value_on_byte(g, sub, node.union_nodes[j], c, next_oneof->alts[j], /*allow_empty_ws=*/false);
        if (!next_oneof->alts[j].empty()) {
          ++alive_alts;
          for (const PdaState& s : next_oneof->alts[j]) {
            if (!s.is_done()) all_alive_done = false;
          }
        }
      }
      if (alive_alts == 0) return false;
      std::string raw(1, char(c));
      st.stack.pop_back();
      if (all_alive_done) {
        if (!oneof_raw_matches_exactly_one(g, node_id, raw)) return false;
        out_branches.push_back(std::move(st));
        return true;
      }
      Frame f;
      f.kind = FrameKind::OneOf;
      f.node_id = node_id;
      f.oneof = std::move(next_oneof);
      f.oneof_raw = std::move(raw);
      st.stack.push_back(std::move(f));
      out_branches.push_back(std::move(st));
      return true;
    }

    bool any_ok = false;
    for (int sub_id : node.union_nodes) {
      if (out_branches.size() >= kMaxUnionBranches) break;
      PdaState copy = st;
      if (start_value_on_byte(g, copy, sub_id, c, out_branches, allow_empty_ws)) {
        any_ok = true;
      }
    }
    return any_ok;
  }

  if (node.kind == NodeKind::Enum) {
    std::string prefix(1, char(c));
    bool exact_done = false;
    bool longer_prefix = false;
    bool numeric_exact = false;
    for (const std::string& lit : node.enum_literals) {
      if (lit == prefix) {
        exact_done = true;
        if (!lit.empty() && (lit[0] == '-' || is_digit(static_cast<std::uint8_t>(lit[0])))) numeric_exact = true;
      } else if (lit.size() > 1 && lit[0] == char(c)) {
        longer_prefix = true;
      }
    }
    if (!exact_done && !longer_prefix) return false;
    st.stack.pop_back();
    if (exact_done && !longer_prefix && !numeric_exact) {
      out_branches.push_back(std::move(st));
      return true;
    }
    Frame f;
    f.kind = FrameKind::EnumLit;
    f.node_id = node_id;
    f.enum_buf = std::move(prefix);
    st.stack.push_back(std::move(f));
    out_branches.push_back(std::move(st));
    return true;
  }

  auto allows_kind = [&](NodeKind k) { return node.kind == NodeKind::Any || node.kind == k; };

  if (c == '{' && allows_kind(NodeKind::Object)) {
    st.stack.pop_back();
    Frame f;
    f.kind = FrameKind::ObjFirstOrEnd;
    f.node_id = (node.kind == NodeKind::Object) ? node_id : 0;
    const SchemaNode& obj_node = g.nodes[std::size_t(f.node_id)];
    f.seen_prop_words.assign((obj_node.props.size() + 63u) / 64u, 0ull);
    st.stack.push_back(std::move(f));
    out_branches.push_back(std::move(st));
    return true;
  }
  if (c == '[' && allows_kind(NodeKind::Array)) {
    st.stack.pop_back();
    Frame f;
    f.kind = FrameKind::ArrFirstOrEnd;
    f.node_id = (node.kind == NodeKind::Array) ? node_id : 0;
    f.arr_count = 0;
    st.stack.push_back(std::move(f));
    out_branches.push_back(std::move(st));
    return true;
  }
  if (c == '"' && allows_kind(NodeKind::String)) {
    st.stack.pop_back();
    Frame f;
    f.kind = FrameKind::StrBody;
    f.node_id = (node.kind == NodeKind::String) ? node_id : 0;
    st.stack.push_back(std::move(f));
    out_branches.push_back(std::move(st));
    return true;
  }
  if ((c == '-' || is_digit(c)) && (allows_kind(NodeKind::Number) || allows_kind(NodeKind::Integer))) {
    const int num_node_id = (node.kind == NodeKind::Number || node.kind == NodeKind::Integer) ? node_id : 0;
    std::string buf(1, char(c));
    if (!num_prefix_can_satisfy(g.nodes[std::size_t(num_node_id)], buf)) return false;
    st.stack.pop_back();
    Frame f;
    f.kind = FrameKind::Num;
    f.node_id = num_node_id;
    f.integer_only = (node.kind == NodeKind::Integer);
    f.num_buf = std::move(buf);
    if (c == '-') {
      f.num_phase = NumPhase::AfterMinus;
    } else if (c == '0') {
      f.num_phase = NumPhase::Zero;
    } else {
      f.num_phase = NumPhase::IntDigits;
    }
    st.stack.push_back(std::move(f));
    out_branches.push_back(std::move(st));
    return true;
  }
  if (c == 't' && allows_kind(NodeKind::Boolean)) {
    st.stack.pop_back();
    Frame f;
    f.kind = FrameKind::Literal;
    f.lit_ptr = "true";
    f.lit_pos = 1;
    st.stack.push_back(std::move(f));
    out_branches.push_back(std::move(st));
    return true;
  }
  if (c == 'f' && allows_kind(NodeKind::Boolean)) {
    st.stack.pop_back();
    Frame f;
    f.kind = FrameKind::Literal;
    f.lit_ptr = "false";
    f.lit_pos = 1;
    st.stack.push_back(std::move(f));
    out_branches.push_back(std::move(st));
    return true;
  }
  if (c == 'n' && allows_kind(NodeKind::Null)) {
    st.stack.pop_back();
    Frame f;
    f.kind = FrameKind::Literal;
    f.lit_ptr = "null";
    f.lit_pos = 1;
    st.stack.push_back(std::move(f));
    out_branches.push_back(std::move(st));
    return true;
  }
  return false;
}

bool step_byte_single(const SchemaGraph& g, PdaState& st, std::uint8_t c, std::vector<PdaState>& out_branches,
                      bool allow_empty_ws) {
  if (st.stack.empty()) {
    if (allow_empty_ws && is_json_ws(c)) {
      out_branches.push_back(std::move(st));
      return true;
    }
    return false;
  }

  Frame& top = st.stack.back();
  switch (top.kind) {
    case FrameKind::Value: {
      if (is_json_ws(c)) {
        out_branches.push_back(std::move(st));
        return true;
      }
      return start_value_on_byte(g, st, top.node_id, c, out_branches, allow_empty_ws);
    }

    case FrameKind::OneOf: {
      const OneOfState& cur = *top.oneof;
      const int oneof_node_id = top.node_id;
      bool cur_any_accepting = false;
      for (const auto& alt_branches : cur.alts) {
        for (const PdaState& sub : alt_branches) {
          if (pda_is_accepting(g, sub)) {
            cur_any_accepting = true;
            break;
          }
        }
        if (cur_any_accepting) break;
      }

      auto next_oneof = std::make_shared<OneOfState>();
      next_oneof->alts.resize(cur.alts.size());
      int alive_alts = 0;
      bool all_alive_done = true;
      for (std::size_t j = 0; j < cur.alts.size(); ++j) {
        for (const PdaState& sub : cur.alts[j]) {
          if (sub.is_done()) continue;
          PdaState copy = sub;
          step_byte_single(g, copy, c, next_oneof->alts[j], /*allow_empty_ws=*/false);
          if (next_oneof->alts[j].size() >= kMaxUnionBranches) break;
        }
        if (!next_oneof->alts[j].empty()) {
          ++alive_alts;
          for (const PdaState& s : next_oneof->alts[j]) {
            if (!s.is_done()) all_alive_done = false;
          }
        }
      }

      bool any_ok = false;
      if (alive_alts > 0 && top.oneof_raw.size() < kMaxOneOfBuf) {
        std::string next_raw = top.oneof_raw;
        next_raw.push_back(char(c));
        if (all_alive_done) {
          if (oneof_raw_matches_exactly_one(g, oneof_node_id, next_raw)) {
            PdaState done_st = st;
            done_st.stack.pop_back();
            out_branches.push_back(std::move(done_st));
            any_ok = true;
          }
        } else {
          PdaState cont_st = st;
          cont_st.stack.back().oneof = std::move(next_oneof);
          cont_st.stack.back().oneof_raw = std::move(next_raw);
          out_branches.push_back(std::move(cont_st));
          any_ok = true;
        }
      }

      if (cur_any_accepting && oneof_raw_matches_exactly_one(g, oneof_node_id, top.oneof_raw)) {
        PdaState exit_st = st;
        exit_st.stack.pop_back();
        if (step_byte_single(g, exit_st, c, out_branches, allow_empty_ws)) {
          any_ok = true;
        }
      }
      return any_ok;
    }

    case FrameKind::ObjFirstOrEnd: {
      if (is_json_ws(c)) {
        out_branches.push_back(std::move(st));
        return true;
      }
      const SchemaNode& node = g.nodes[std::size_t(top.node_id)];
      if (c == '}') {
        if (!obj_required_satisfied(node, top.seen_prop_words, top.seen_prop_count)) return false;
        st.stack.pop_back();
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c == '"') {
        if (!obj_has_remaining_key(node, top.seen_prop_words, top.seen_prop_count)) return false;
        top.kind = FrameKind::ObjKeyStr;
        top.key_buf.clear();
        top.in_escape = false;
        top.expect_low_surrogate = false;
        top.uni_rem = 0;
        top.utf8_rem = 0;
        top.str_len = 0;
        out_branches.push_back(std::move(st));
        return true;
      }
      return false;
    }

    case FrameKind::ObjNextKey: {
      if (is_json_ws(c)) {
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c == '"') {
        const SchemaNode& node = g.nodes[std::size_t(top.node_id)];
        if (!obj_has_remaining_key(node, top.seen_prop_words, top.seen_prop_count)) return false;
        top.kind = FrameKind::ObjKeyStr;
        top.key_buf.clear();
        top.in_escape = false;
        top.expect_low_surrogate = false;
        top.uni_rem = 0;
        top.utf8_rem = 0;
        top.str_len = 0;
        out_branches.push_back(std::move(st));
        return true;
      }
      return false;
    }

    case FrameKind::ObjKeyStr: {
      const SchemaNode& node = g.nodes[std::size_t(top.node_id)];
      if (!node.additional_properties) {
        if (c == '"') {
          for (std::size_t i = 0; i < node.props.size(); ++i) {
            if (prop_is_seen(top.seen_prop_words, i)) continue;
            if (node.props[i].json_escaped == top.key_buf) {
              prop_mark_seen(top.seen_prop_words, i);
              ++top.seen_prop_count;
              top.val_node_id = node.props[i].node_id;
              top.kind = FrameKind::ObjColon;
              top.key_buf.clear();
              out_branches.push_back(std::move(st));
              return true;
            }
          }
          return false;
        }
        top.key_buf.push_back(char(c));
        bool any_prefix = false;
        for (std::size_t i = 0; i < node.props.size(); ++i) {
          if (prop_is_seen(top.seen_prop_words, i)) continue;
          const std::string& target = node.props[i].json_escaped;
          if (target.size() >= top.key_buf.size() &&
              target.compare(0, top.key_buf.size(), top.key_buf) == 0) {
            any_prefix = true;
            break;
          }
        }
        if (!any_prefix) return false;
        out_branches.push_back(std::move(st));
        return true;
      }

      bool completed_char = false;
      bool closed_quote = false;
      if (!step_string_byte(top, c, /*max_len=*/-1, completed_char, closed_quote)) return false;
      if (closed_quote) {
        std::string decoded_key;
        if (!unescape_json_string(top.key_buf, decoded_key)) return false;
        auto pos = std::lower_bound(top.seen_keys.begin(), top.seen_keys.end(), decoded_key);
        if (pos != top.seen_keys.end() && *pos == decoded_key) return false;
        top.seen_keys.insert(pos, decoded_key);

        int val_node = node.additional_node_id;
        for (std::size_t i = 0; i < node.props.size(); ++i) {
          if (node.props[i].name == decoded_key) {
            prop_mark_seen(top.seen_prop_words, i);
            val_node = node.props[i].node_id;
            break;
          }
        }
        ++top.seen_prop_count;
        top.val_node_id = val_node;
        top.kind = FrameKind::ObjColon;
        top.key_buf.clear();
        top.in_escape = false;
        top.expect_low_surrogate = false;
        top.uni_rem = 0;
        top.utf8_rem = 0;
        out_branches.push_back(std::move(st));
        return true;
      }
      if (completed_char) ++top.str_len;
      top.key_buf.push_back(char(c));
      out_branches.push_back(std::move(st));
      return true;
    }

    case FrameKind::ObjColon: {
      if (is_json_ws(c)) {
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c == ':') {
        const int vnode = top.val_node_id;
        top.kind = FrameKind::ObjCommaOrEnd;
        Frame vf;
        vf.kind = FrameKind::Value;
        vf.node_id = vnode;
        st.stack.push_back(std::move(vf));
        out_branches.push_back(std::move(st));
        return true;
      }
      return false;
    }

    case FrameKind::ObjCommaOrEnd: {
      if (is_json_ws(c)) {
        out_branches.push_back(std::move(st));
        return true;
      }
      const SchemaNode& node = g.nodes[std::size_t(top.node_id)];
      if (c == '}') {
        if (!obj_required_satisfied(node, top.seen_prop_words, top.seen_prop_count)) return false;
        st.stack.pop_back();
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c == ',') {
        if (!obj_has_remaining_key(node, top.seen_prop_words, top.seen_prop_count)) return false;
        top.kind = FrameKind::ObjNextKey;
        out_branches.push_back(std::move(st));
        return true;
      }
      return false;
    }

    case FrameKind::ArrFirstOrEnd: {
      if (is_json_ws(c)) {
        out_branches.push_back(std::move(st));
        return true;
      }
      const SchemaNode& arr_node = g.nodes[std::size_t(top.node_id)];
      if (c == ']') {
        if (arr_node.min_items > 0) return false;
        st.stack.pop_back();
        out_branches.push_back(std::move(st));
        return true;
      }
      if (arr_node.max_items >= 0 && arr_node.max_items == 0) return false;
      const int item_node = arr_node.item_node_id;
      top.arr_count = 1;
      top.kind = FrameKind::ArrCommaOrEnd;
      Frame vf;
      vf.kind = FrameKind::Value;
      vf.node_id = item_node;
      st.stack.push_back(std::move(vf));
      return start_value_on_byte(g, st, item_node, c, out_branches, allow_empty_ws);
    }

    case FrameKind::ArrNextVal: {
      if (is_json_ws(c)) {
        out_branches.push_back(std::move(st));
        return true;
      }
      const SchemaNode& arr_node = g.nodes[std::size_t(top.node_id)];
      const int item_node = arr_node.item_node_id;
      ++top.arr_count;
      top.kind = FrameKind::ArrCommaOrEnd;
      Frame vf;
      vf.kind = FrameKind::Value;
      vf.node_id = item_node;
      st.stack.push_back(std::move(vf));
      return start_value_on_byte(g, st, item_node, c, out_branches, allow_empty_ws);
    }

    case FrameKind::ArrCommaOrEnd: {
      if (is_json_ws(c)) {
        out_branches.push_back(std::move(st));
        return true;
      }
      const SchemaNode& arr_node = g.nodes[std::size_t(top.node_id)];
      if (c == ']') {
        if (top.arr_count < arr_node.min_items) return false;
        st.stack.pop_back();
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c == ',') {
        if (arr_node.max_items >= 0 && top.arr_count >= arr_node.max_items) return false;
        top.kind = FrameKind::ArrNextVal;
        out_branches.push_back(std::move(st));
        return true;
      }
      return false;
    }

    case FrameKind::StrBody: {
      const SchemaNode& node = g.nodes[std::size_t(top.node_id)];
      bool completed_char = false;
      bool closed_quote = false;
      if (!step_string_byte(top, c, node.max_length, completed_char, closed_quote)) return false;
      if (closed_quote) {
        if (top.str_len < std::size_t(node.min_length)) return false;
        st.stack.pop_back();
        out_branches.push_back(std::move(st));
        return true;
      }
      if (completed_char) ++top.str_len;
      out_branches.push_back(std::move(st));
      return true;
    }

    case FrameKind::Num: {
      const SchemaNode& node = g.nodes[std::size_t(top.node_id)];
      auto push_num_char = [&](NumPhase next_ph) -> bool {
        if (top.num_buf.size() >= kMaxNumLen) return false;
        top.num_buf.push_back(char(c));
        if (!num_prefix_can_satisfy(node, top.num_buf)) return false;
        top.num_phase = next_ph;
        out_branches.push_back(std::move(st));
        return true;
      };
      auto finish_num_and_redispatch = [&]() -> bool {
        if (!num_value_satisfies(node, top.num_buf)) return false;
        st.stack.pop_back();
        return step_byte_single(g, st, c, out_branches, allow_empty_ws);
      };

      switch (top.num_phase) {
        case NumPhase::AfterMinus:
          if (c == '0') return push_num_char(NumPhase::Zero);
          if (c >= '1' && c <= '9') return push_num_char(NumPhase::IntDigits);
          return false;
        case NumPhase::Zero:
          if (!top.integer_only && c == '.') return push_num_char(NumPhase::AfterDot);
          if (!top.integer_only && (c == 'e' || c == 'E')) return push_num_char(NumPhase::AfterExp);
          return finish_num_and_redispatch();
        case NumPhase::IntDigits:
          if (is_digit(c)) return push_num_char(NumPhase::IntDigits);
          if (!top.integer_only && c == '.') return push_num_char(NumPhase::AfterDot);
          if (!top.integer_only && (c == 'e' || c == 'E')) return push_num_char(NumPhase::AfterExp);
          return finish_num_and_redispatch();
        case NumPhase::AfterDot:
          if (is_digit(c)) return push_num_char(NumPhase::FracDigits);
          return false;
        case NumPhase::FracDigits:
          if (is_digit(c)) return push_num_char(NumPhase::FracDigits);
          if (c == 'e' || c == 'E') return push_num_char(NumPhase::AfterExp);
          return finish_num_and_redispatch();
        case NumPhase::AfterExp:
          if (c == '+' || c == '-') return push_num_char(NumPhase::AfterExpSign);
          if (is_digit(c)) return push_num_char(NumPhase::ExpDigits);
          return false;
        case NumPhase::AfterExpSign:
          if (is_digit(c)) return push_num_char(NumPhase::ExpDigits);
          return false;
        case NumPhase::ExpDigits:
          if (is_digit(c)) return push_num_char(NumPhase::ExpDigits);
          return finish_num_and_redispatch();
      }
      return false;
    }

    case FrameKind::Literal: {
      if (top.lit_ptr[top.lit_pos] != char(c)) return false;
      ++top.lit_pos;
      if (top.lit_ptr[top.lit_pos] == '\0') st.stack.pop_back();
      out_branches.push_back(std::move(st));
      return true;
    }

    case FrameKind::EnumLit: {
      const SchemaNode& node = g.nodes[std::size_t(top.node_id)];
      std::string next = top.enum_buf;
      next.push_back(char(c));
      bool matched_exact = false;
      bool matched_longer = false;
      bool numeric_exact = false;
      for (const std::string& lit : node.enum_literals) {
        if (lit == next) {
          matched_exact = true;
          if (!lit.empty() && (lit[0] == '-' || is_digit(static_cast<std::uint8_t>(lit[0])))) numeric_exact = true;
        } else if (lit.size() > next.size() && lit.compare(0, next.size(), next) == 0) {
          matched_longer = true;
        }
      }
      if (matched_exact || matched_longer) {
        if (matched_exact && !matched_longer && !numeric_exact) {
          st.stack.pop_back();
        } else {
          top.enum_buf = std::move(next);
        }
        out_branches.push_back(std::move(st));
        return true;
      }
      bool cur_exact = false;
      for (const std::string& lit : node.enum_literals) {
        if (lit == top.enum_buf) {
          cur_exact = true;
          break;
        }
      }
      if (cur_exact) {
        st.stack.pop_back();
        return step_byte_single(g, st, c, out_branches, allow_empty_ws);
      }
      return false;
    }
  }
  return false;
}

bool pda_is_accepting(const SchemaGraph& g, const PdaState& st) noexcept {
  if (st.stack.empty()) return true;
  if (st.stack.size() == 1) {
    const Frame& f = st.stack.back();
    if (f.kind == FrameKind::Num) {
      return num_can_terminate(f.num_phase) && num_value_satisfies(g.nodes[std::size_t(f.node_id)], f.num_buf);
    }
    if (f.kind == FrameKind::EnumLit) {
      const SchemaNode& node = g.nodes[std::size_t(f.node_id)];
      for (const std::string& lit : node.enum_literals) {
        if (lit == f.enum_buf) return true;
      }
    }
    if (f.kind == FrameKind::OneOf && f.oneof) {
      bool any_alt_accepting = false;
      for (const auto& alt_branches : f.oneof->alts) {
        for (const PdaState& sub : alt_branches) {
          if (pda_is_accepting(g, sub)) {
            any_alt_accepting = true;
            break;
          }
        }
        if (any_alt_accepting) break;
      }
      if (!any_alt_accepting) return false;
      return oneof_raw_matches_exactly_one(g, f.node_id, f.oneof_raw);
    }
  }
  return false;
}

bool advance_branches(const SchemaGraph& g, const std::vector<PdaState>& in, std::string_view bytes,
                      std::vector<PdaState>& out) {
  std::vector<PdaState> cur = in;
  std::vector<PdaState> next;
  for (char ch : bytes) {
    next.clear();
    const auto c = static_cast<std::uint8_t>(ch);
    for (PdaState& st : cur) {
      step_byte_single(g, st, c, next);
      if (next.size() >= kMaxUnionBranches) break;
    }
    if (next.empty()) return false;
    cur.swap(next);
  }
  out = std::move(cur);
  return true;
}

bool can_advance_any(const SchemaGraph& g, const std::vector<PdaState>& in, std::string_view bytes) {
  if (bytes.empty()) return false;
  std::vector<PdaState> tmp;
  return advance_branches(g, in, bytes, tmp);
}

std::bitset<256> allowed_first_bytes(const SchemaGraph& g, const std::vector<PdaState>& branches) {
  std::bitset<256> bits;
  std::vector<PdaState> tmp;
  for (int b = 0; b < 256; ++b) {
    const auto c = static_cast<std::uint8_t>(b);
    for (const PdaState& st : branches) {
      PdaState copy = st;
      tmp.clear();
      if (step_byte_single(g, copy, c, tmp)) {
        bits.set(std::size_t(b));
        break;
      }
    }
  }
  return bits;
}

class JsonFsmImpl final : public model::TokenMasker {
 public:
  enum class Channel : std::uint8_t {
    PreContent,      // thinking / header before <|content_text|>; unconstrained!
    InContent,       // actively constraining JSON / JSON-Schema bytes
    AfterEndMessage, // saw <|end_message|> after valid JSON; allow only EOS/stop tokens
  };

  JsonFsmImpl(std::shared_ptr<const VocabByteIndex> vocab, std::shared_ptr<const SchemaGraph> graph,
              std::span<const std::int32_t> prefill_ids)
      : vocab_(std::move(vocab)), graph_(std::move(graph)) {
    PdaState init;
    Frame root;
    root.kind = FrameKind::Value;
    root.node_id = graph_->root_id;
    init.stack.push_back(std::move(root));
    branches_.push_back(std::move(init));

    channel_ = Channel::PreContent;
    const std::int32_t ct_id = vocab_->content_text_id();
    const std::int32_t em_id = vocab_->end_message_id();
    for (std::int32_t id : prefill_ids) {
      if (ct_id >= 0 && id == ct_id) {
        channel_ = Channel::InContent;
      } else if (em_id >= 0 && id == em_id) {
        channel_ = Channel::PreContent;
      }
    }
    if (ct_id < 0) channel_ = Channel::InContent;
  }

  int vocab_size() const override { return vocab_->vocab_size(); }

  bool is_active() const override { return channel_ != Channel::PreContent; }

  bool fill_mask(std::uint32_t* words, std::size_t word_count) const override {
    if (channel_ == Channel::PreContent) return false;
    const std::size_t nwords = std::min(word_count, vocab_->mask_words());
    std::memset(words, 0, nwords * sizeof(std::uint32_t));

    auto allow_stops = [&] {
      set_bit(words, nwords, vocab_->end_sampling_id());
      set_bit(words, nwords, vocab_->endoftext_id());
      for (std::int32_t sid : vocab_->stop_ids()) set_bit(words, nwords, sid);
    };

    if (channel_ == Channel::AfterEndMessage) {
      allow_stops();
      return true;
    }

    bool any_accepting = false;
    bool all_done = !branches_.empty();
    for (const PdaState& st : branches_) {
      if (pda_is_accepting(*graph_, st)) any_accepting = true;
      if (!st.is_done()) all_done = false;
    }

    if (all_done) {
      const auto& ws = vocab_->ws_mask();
      std::memcpy(words, ws.data(), std::min(nwords, ws.size()) * sizeof(std::uint32_t));
      set_bit(words, nwords, vocab_->end_message_id());
      allow_stops();
      return true;
    }

    // Fast path: single branch inside unbounded JSON string value with clean UTF-8/escape state.
    if (branches_.size() == 1 && !branches_[0].stack.empty()) {
      const Frame& top = branches_[0].stack.back();
      if (top.kind == FrameKind::StrBody && !top.in_escape && !top.expect_low_surrogate && top.uni_rem == 0 &&
          top.utf8_rem == 0) {
        const SchemaNode& snode = graph_->nodes[std::size_t(top.node_id)];
        if (snode.min_length == 0 && snode.max_length < 0) {
          const auto& safe = vocab_->string_safe_mask();
          std::memcpy(words, safe.data(), std::min(nwords, safe.size()) * sizeof(std::uint32_t));
          for (std::int32_t id : vocab_->string_special_candidates()) {
            if (can_advance_any(*graph_, branches_, vocab_->token_bytes(id))) {
              set_bit(words, nwords, id);
            }
          }
          return true;
        }
      }
    }

    const std::bitset<256> first = allowed_first_bytes(*graph_, branches_);
    for (int b = 0; b < 256; ++b) {
      if (!first.test(std::size_t(b))) continue;
      for (std::int32_t id : vocab_->bucket(std::uint8_t(b))) {
        if (can_advance_any(*graph_, branches_, vocab_->token_bytes(id))) {
          set_bit(words, nwords, id);
        }
      }
    }

    if (any_accepting) {
      set_bit(words, nwords, vocab_->end_message_id());
      allow_stops();
    }
    return true;
  }

  void advance(std::int32_t token_id) override {
    if (token_id < 0 || token_id >= vocab_->vocab_size()) return;
    const std::int32_t ct_id = vocab_->content_text_id();
    const std::int32_t em_id = vocab_->end_message_id();

    if (channel_ == Channel::PreContent) {
      if (ct_id >= 0 && token_id == ct_id) {
        channel_ = Channel::InContent;
      }
      return;
    }

    if (channel_ == Channel::AfterEndMessage) return;

    if (em_id >= 0 && token_id == em_id) {
      channel_ = Channel::AfterEndMessage;
      return;
    }
    if (vocab_->is_special(token_id)) return;

    std::vector<PdaState> next;
    if (advance_branches(*graph_, branches_, vocab_->token_bytes(token_id), next)) {
      branches_ = std::move(next);
    }
  }

 private:
  std::shared_ptr<const VocabByteIndex> vocab_;
  std::shared_ptr<const SchemaGraph> graph_;
  Channel channel_ = Channel::InContent;
  std::vector<PdaState> branches_;
};

}  // namespace

std::shared_ptr<const VocabByteIndex> VocabByteIndex::build(const Tokenizer& tokenizer,
                                                            std::span<const std::int32_t> stop_ids) {
  auto idx = std::make_shared<VocabByteIndex>();
  idx->vocab_size_ = int(tokenizer.vocab_size());
  idx->mask_words_ = (std::size_t(idx->vocab_size_) + 31u) / 32u;
  idx->stop_ids_.assign(stop_ids.begin(), stop_ids.end());

  if (auto id = tokenizer.token_id(CONTENT_TEXT)) idx->content_text_id_ = *id;
  if (auto id = tokenizer.token_id(END_MESSAGE)) idx->end_message_id_ = *id;
  if (auto id = tokenizer.token_id(END_SAMPLING)) idx->end_sampling_id_ = *id;
  if (auto id = tokenizer.token_id(ENDOFTEXT)) idx->endoftext_id_ = *id;

  idx->bytes_.resize(std::size_t(idx->vocab_size_));
  idx->is_special_.assign(std::size_t(idx->vocab_size_), 0);
  idx->ws_mask_.assign(idx->mask_words_, 0u);
  idx->string_safe_mask_.assign(idx->mask_words_, 0u);

  for (int id = 0; id < idx->vocab_size_; ++id) {
    const bool spec = tokenizer.is_special(id);
    idx->is_special_[std::size_t(id)] = spec ? 1 : 0;
    std::string b = tokenizer.token_bytes(id);
    idx->bytes_[std::size_t(id)] = b;
    if (spec || b.empty()) continue;

    idx->by_first_byte_[static_cast<std::uint8_t>(b[0])].push_back(id);

    bool all_ws = true;
    for (char ch : b) {
      if (!is_json_ws(static_cast<std::uint8_t>(ch))) {
        all_ws = false;
        break;
      }
    }
    if (all_ws) set_bit(idx->ws_mask_.data(), idx->mask_words_, id);
    if (is_complete_string_safe_utf8(b)) {
      set_bit(idx->string_safe_mask_.data(), idx->mask_words_, id);
    } else {
      idx->string_special_candidates_.push_back(id);
    }
  }
  return idx;
}

std::shared_ptr<model::TokenMasker> make_json_fsm(std::shared_ptr<const VocabByteIndex> vocab,
                                                  const JsonFormatSpec& spec,
                                                  std::span<const std::int32_t> prefill_ids) {
  if (!vocab || spec.mode == JsonFormatMode::None) return nullptr;
  auto graph = std::make_shared<SchemaGraph>();
  if (spec.mode == JsonFormatMode::JsonObject) {
    SchemaNode root;
    root.kind = NodeKind::Object;
    root.additional_properties = true;
    root.additional_node_id = 0;
    graph->root_id = graph->add_node(std::move(root));
  } else {
    graph->root_id = graph->compile(spec.schema, spec.schema, 0);
  }
  return std::make_shared<JsonFsmImpl>(std::move(vocab), std::move(graph), prefill_ids);
}

}  // namespace ingine::text

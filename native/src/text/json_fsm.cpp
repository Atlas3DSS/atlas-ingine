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

std::size_t utf8_codepoint_length(std::string_view s) noexcept {
  std::size_t count = 0;
  for (char ch : s) {
    if ((static_cast<std::uint8_t>(ch) & 0xC0u) != 0x80u) ++count;
  }
  return count;
}

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

// ---- Exact Arbitrary-Precision Decimal & Exact-Decimal JSON DOM -------------------------------

struct ParsedDecimal {
  bool neg = false;
  bool is_zero = false;
  std::string sig;  // digits with no leading or trailing zeros (when !is_zero)
  std::int64_t scale10 = 0;

  bool is_integral() const noexcept { return is_zero || scale10 >= 0; }
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

std::optional<Int128> decimal_to_int128_round(const ParsedDecimal& dec, bool round_up) noexcept {
  if (dec.is_zero) return Int128(0);
  const std::int64_t int_digits = std::int64_t(dec.sig.size()) + dec.scale10;
  if (int_digits <= 0) {
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

// Lossless exact-decimal JSON value (never normalizes numbers through binary64 double).
struct ExactJson {
  enum class Kind : std::uint8_t { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool b = false;
  std::string str;  // decoded UTF-8 for String; verbatim raw JSON number text for Number
  ParsedDecimal dec;
  std::vector<ExactJson> arr;
  std::vector<std::pair<std::string, ExactJson>> obj;

  bool is_null() const noexcept { return kind == Kind::Null; }
  bool is_bool() const noexcept { return kind == Kind::Bool; }
  bool is_number() const noexcept { return kind == Kind::Number; }
  bool is_string() const noexcept { return kind == Kind::String; }
  bool is_array() const noexcept { return kind == Kind::Array; }
  bool is_object() const noexcept { return kind == Kind::Object; }

  const ExactJson* find(std::string_view key) const noexcept {
    if (!is_object()) return nullptr;
    for (const auto& [k, v] : obj) {
      if (k == key) return &v;
    }
    return nullptr;
  }
  bool contains(std::string_view key) const noexcept { return find(key) != nullptr; }
};

struct ExactJsonParser {
  std::string_view s;
  std::size_t pos = 0;

  void skip_ws() noexcept {
    while (pos < s.size() && is_json_ws(static_cast<std::uint8_t>(s[pos]))) ++pos;
  }

  bool scan_raw_string(std::string_view& raw_inner) noexcept {
    if (pos >= s.size() || s[pos] != '"') return false;
    ++pos;
    const std::size_t start = pos;
    while (pos < s.size()) {
      const auto c = static_cast<std::uint8_t>(s[pos]);
      if (c == '"') {
        raw_inner = s.substr(start, pos - start);
        ++pos;
        return true;
      }
      if (c == '\\') {
        if (pos + 1 >= s.size()) return false;
        pos += 2;
        continue;
      }
      if (c < 0x20) return false;
      ++pos;
    }
    return false;
  }

  bool scan_number(std::string_view& raw_num) noexcept {
    const std::size_t start = pos;
    if (pos < s.size() && s[pos] == '-') ++pos;
    if (pos >= s.size()) return false;
    if (s[pos] == '0') {
      ++pos;
    } else if (s[pos] >= '1' && s[pos] <= '9') {
      while (pos < s.size() && is_digit(static_cast<std::uint8_t>(s[pos]))) ++pos;
    } else {
      return false;
    }
    if (pos < s.size() && s[pos] == '.') {
      ++pos;
      if (pos >= s.size() || !is_digit(static_cast<std::uint8_t>(s[pos]))) return false;
      while (pos < s.size() && is_digit(static_cast<std::uint8_t>(s[pos]))) ++pos;
    }
    if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
      ++pos;
      if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
      if (pos >= s.size() || !is_digit(static_cast<std::uint8_t>(s[pos]))) return false;
      while (pos < s.size() && is_digit(static_cast<std::uint8_t>(s[pos]))) ++pos;
    }
    raw_num = s.substr(start, pos - start);
    return true;
  }

  bool skip_value(int depth = 0) noexcept {
    if (depth > 64) return false;
    skip_ws();
    if (pos >= s.size()) return false;
    const char c = s[pos];
    if (c == '"') {
      std::string_view tmp;
      return scan_raw_string(tmp);
    }
    if (c == '{') {
      ++pos;
      skip_ws();
      if (pos < s.size() && s[pos] == '}') {
        ++pos;
        return true;
      }
      while (true) {
        skip_ws();
        std::string_view key;
        if (!scan_raw_string(key)) return false;
        skip_ws();
        if (pos >= s.size() || s[pos] != ':') return false;
        ++pos;
        if (!skip_value(depth + 1)) return false;
        skip_ws();
        if (pos >= s.size()) return false;
        if (s[pos] == '}') {
          ++pos;
          return true;
        }
        if (s[pos] != ',') return false;
        ++pos;
      }
    }
    if (c == '[') {
      ++pos;
      skip_ws();
      if (pos < s.size() && s[pos] == ']') {
        ++pos;
        return true;
      }
      while (true) {
        if (!skip_value(depth + 1)) return false;
        skip_ws();
        if (pos >= s.size()) return false;
        if (s[pos] == ']') {
          ++pos;
          return true;
        }
        if (s[pos] != ',') return false;
        ++pos;
      }
    }
    if (c == 't') {
      if (s.substr(pos, 4) != "true") return false;
      pos += 4;
      return true;
    }
    if (c == 'f') {
      if (s.substr(pos, 5) != "false") return false;
      pos += 5;
      return true;
    }
    if (c == 'n') {
      if (s.substr(pos, 4) != "null") return false;
      pos += 4;
      return true;
    }
    std::string_view tmp;
    return scan_number(tmp);
  }

  bool parse_value(ExactJson& out, int depth = 0) {
    if (depth > 64) return false;
    skip_ws();
    if (pos >= s.size()) return false;
    const char c = s[pos];
    if (c == 'n') {
      if (s.substr(pos, 4) != "null") return false;
      pos += 4;
      out = ExactJson{};
      out.kind = ExactJson::Kind::Null;
      return true;
    }
    if (c == 't') {
      if (s.substr(pos, 4) != "true") return false;
      pos += 4;
      out = ExactJson{};
      out.kind = ExactJson::Kind::Bool;
      out.b = true;
      return true;
    }
    if (c == 'f') {
      if (s.substr(pos, 5) != "false") return false;
      pos += 5;
      out = ExactJson{};
      out.kind = ExactJson::Kind::Bool;
      out.b = false;
      return true;
    }
    if (c == '"') {
      std::string_view raw_inner;
      if (!scan_raw_string(raw_inner)) return false;
      out = ExactJson{};
      out.kind = ExactJson::Kind::String;
      return unescape_json_string(raw_inner, out.str);
    }
    if (c == '[') {
      ++pos;
      out = ExactJson{};
      out.kind = ExactJson::Kind::Array;
      skip_ws();
      if (pos < s.size() && s[pos] == ']') {
        ++pos;
        return true;
      }
      while (true) {
        ExactJson elem;
        if (!parse_value(elem, depth + 1)) return false;
        out.arr.push_back(std::move(elem));
        skip_ws();
        if (pos >= s.size()) return false;
        if (s[pos] == ']') {
          ++pos;
          return true;
        }
        if (s[pos] != ',') return false;
        ++pos;
      }
    }
    if (c == '{') {
      ++pos;
      out = ExactJson{};
      out.kind = ExactJson::Kind::Object;
      skip_ws();
      if (pos < s.size() && s[pos] == '}') {
        ++pos;
        return true;
      }
      while (true) {
        skip_ws();
        std::string_view raw_key;
        if (!scan_raw_string(raw_key)) return false;
        std::string key;
        if (!unescape_json_string(raw_key, key)) return false;
        skip_ws();
        if (pos >= s.size() || s[pos] != ':') return false;
        ++pos;
        ExactJson val;
        if (!parse_value(val, depth + 1)) return false;
        // Last key wins (matches nlohmann::json behavior).
        bool replaced = false;
        for (auto& kv : out.obj) {
          if (kv.first == key) {
            kv.second = std::move(val);
            replaced = true;
            break;
          }
        }
        if (!replaced) out.obj.emplace_back(std::move(key), std::move(val));
        skip_ws();
        if (pos >= s.size()) return false;
        if (s[pos] == '}') {
          ++pos;
          return true;
        }
        if (s[pos] != ',') return false;
        ++pos;
      }
    }
    std::string_view raw_num;
    if (!scan_number(raw_num)) return false;
    out = ExactJson{};
    out.kind = ExactJson::Kind::Number;
    out.str = std::string(raw_num);
    return parse_json_decimal(raw_num, out.dec);
  }
};

bool parse_exact_json(std::string_view raw, ExactJson& out) {
  ExactJsonParser p{raw, 0};
  if (!p.parse_value(out, 0)) return false;
  p.skip_ws();
  return p.pos == raw.size();
}

// Compact lossless dump of an ExactJson (preserves exact decimal spellings without double rounding).
std::string exact_json_dump(const ExactJson& v) {
  switch (v.kind) {
    case ExactJson::Kind::Null: return "null";
    case ExactJson::Kind::Bool: return v.b ? "true" : "false";
    case ExactJson::Kind::Number: return v.str;
    case ExactJson::Kind::String: return Json(v.str).dump();
    case ExactJson::Kind::Array: {
      std::string out = "[";
      for (std::size_t i = 0; i < v.arr.size(); ++i) {
        if (i > 0) out.push_back(',');
        out += exact_json_dump(v.arr[i]);
      }
      out.push_back(']');
      return out;
    }
    case ExactJson::Kind::Object: {
      std::string out = "{";
      for (std::size_t i = 0; i < v.obj.size(); ++i) {
        if (i > 0) out.push_back(',');
        out += Json(v.obj[i].first).dump();
        out.push_back(':');
        out += exact_json_dump(v.obj[i].second);
      }
      out.push_back('}');
      return out;
    }
  }
  return "null";
}

// JSON Schema 2020-12 section 4.2.2 mathematical/structural equality on ExactJson.
bool exact_json_equal(const ExactJson& a, const ExactJson& b) noexcept {
  if (a.kind != b.kind) return false;
  switch (a.kind) {
    case ExactJson::Kind::Null: return true;
    case ExactJson::Kind::Bool: return a.b == b.b;
    case ExactJson::Kind::Number: return compare_decimals(a.dec, b.dec) == 0;
    case ExactJson::Kind::String: return a.str == b.str;
    case ExactJson::Kind::Array: {
      if (a.arr.size() != b.arr.size()) return false;
      for (std::size_t i = 0; i < a.arr.size(); ++i) {
        if (!exact_json_equal(a.arr[i], b.arr[i])) return false;
      }
      return true;
    }
    case ExactJson::Kind::Object: {
      if (a.obj.size() != b.obj.size()) return false;
      for (const auto& [k, av] : a.obj) {
        const ExactJson* bv = b.find(k);
        if (!bv || !exact_json_equal(av, *bv)) return false;
      }
      return true;
    }
  }
  return false;
}

// Rejects any binary64-rounded float inside numeric bounds or const/enum when raw wire JSON was not supplied.
bool contains_json_float(const Json& j) noexcept {
  if (j.is_number_float()) return true;
  if (j.is_array()) {
    for (const Json& e : j) {
      if (contains_json_float(e)) return true;
    }
  } else if (j.is_object()) {
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (contains_json_float(it.value())) return true;
    }
  }
  return false;
}

void reject_binary64_rounded_constraints(const Json& schema, int depth = 0) {
  if (depth > 32 || !schema.is_object()) return;
  for (const char* kw : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}) {
    if (auto it = schema.find(kw); it != schema.end() && it->is_number_float()) {
      throw std::invalid_argument(std::string("JSON schema decimal '") + kw +
                                  "' requires raw_schema_json (wire JSON) to preserve exact decimal precision");
    }
  }
  for (const char* kw : {"const", "enum"}) {
    if (auto it = schema.find(kw); it != schema.end() && contains_json_float(*it)) {
      throw std::invalid_argument(std::string("JSON schema decimal '") + kw +
                                  "' requires raw_schema_json (wire JSON) to preserve exact decimal precision");
    }
  }
  for (const char* map_kw : {"properties", "$defs", "definitions"}) {
    if (auto it = schema.find(map_kw); it != schema.end() && it->is_object()) {
      for (auto pit = it->begin(); pit != it->end(); ++pit) {
        reject_binary64_rounded_constraints(pit.value(), depth + 1);
      }
    }
  }
  for (const char* sub_kw : {"additionalProperties", "items"}) {
    if (auto it = schema.find(sub_kw); it != schema.end() && it->is_object()) {
      reject_binary64_rounded_constraints(*it, depth + 1);
    }
  }
  for (const char* arr_kw : {"anyOf", "oneOf"}) {
    if (auto it = schema.find(arr_kw); it != schema.end() && it->is_array()) {
      for (const Json& elem : *it) reject_binary64_rounded_constraints(elem, depth + 1);
    }
  }
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
  std::string json_escaped;
  int node_id = 0;
  bool required = false;
};

struct SchemaNode {
  NodeKind kind = NodeKind::Any;
  std::vector<PropDef> props;
  bool additional_properties = true;
  int additional_node_id = 0;
  int min_properties = 0;
  int max_properties = -1;
  int item_node_id = 0;
  int min_items = 0;
  int max_items = -1;
  int min_length = 0;
  int max_length = -1;
  std::optional<ParsedDecimal> minimum;
  std::optional<ParsedDecimal> maximum;
  std::optional<ParsedDecimal> exclusive_minimum;
  std::optional<ParsedDecimal> exclusive_maximum;
  std::vector<std::string> enum_literals;
  std::vector<ExactJson> enum_values;
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

const ExactJson* resolve_json_pointer(const ExactJson& root, const std::string& ref) {
  if (ref == "#") return &root;
  if (ref.size() < 2 || ref[0] != '#' || ref[1] != '/') return nullptr;
  const ExactJson* cur = &root;
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
    cur = cur->find(unescaped);
    if (!cur) return nullptr;
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

bool exact_json_matches_type_name(const ExactJson& v, std::string_view t) noexcept {
  if (t == "null") return v.is_null();
  if (t == "boolean") return v.is_bool();
  if (t == "string") return v.is_string();
  if (t == "array") return v.is_array();
  if (t == "object") return v.is_object();
  if (t == "number") return v.is_number();
  if (t == "integer") return v.is_number() && v.dec.is_integral();
  return false;
}

int parse_nonneg_int_kw(const ExactJson& obj, const char* key, int default_val) {
  const ExactJson* v = obj.find(key);
  if (!v) return default_val;
  if (!v->is_number() || !v->dec.is_integral()) {
    throw std::invalid_argument(std::string("JSON schema '") + key + "' must be a non-negative integer");
  }
  const auto iv = decimal_to_int128_round(v->dec, false);
  if (!iv || *iv < 0 || *iv > Int128(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string("JSON schema '") + key + "' out of supported integer range");
  }
  return int(*iv);
}

std::optional<ParsedDecimal> parse_exact_bound_kw(const ExactJson& obj, const char* key) {
  const ExactJson* v = obj.find(key);
  if (!v) return std::nullopt;
  if (!v->is_number()) {
    throw std::invalid_argument(std::string("JSON schema '") + key + "' must be a number");
  }
  return v->dec;
}

struct NumBounds {
  std::optional<ParsedDecimal> minimum;
  std::optional<ParsedDecimal> maximum;
  std::optional<ParsedDecimal> exclusive_minimum;
  std::optional<ParsedDecimal> exclusive_maximum;
};

NumBounds parse_num_bounds(const ExactJson& obj) {
  NumBounds b;
  b.minimum = parse_exact_bound_kw(obj, "minimum");
  b.maximum = parse_exact_bound_kw(obj, "maximum");
  b.exclusive_minimum = parse_exact_bound_kw(obj, "exclusiveMinimum");
  b.exclusive_maximum = parse_exact_bound_kw(obj, "exclusiveMaximum");
  return b;
}

void compute_integer_bounds(const std::optional<ParsedDecimal>& minimum, const std::optional<ParsedDecimal>& maximum,
                            const std::optional<ParsedDecimal>& exclusive_minimum,
                            const std::optional<ParsedDecimal>& exclusive_maximum, std::optional<Int128>& eff_lo,
                            std::optional<Int128>& eff_hi) noexcept {
  eff_lo.reset();
  eff_hi.reset();
  if (minimum) {
    if (auto v = decimal_to_int128_round(*minimum, true)) eff_lo = eff_lo ? std::max(*eff_lo, *v) : *v;
  }
  if (exclusive_minimum) {
    if (auto f = decimal_to_int128_round(*exclusive_minimum, false)) {
      const Int128 v = *f + 1;
      eff_lo = eff_lo ? std::max(*eff_lo, v) : v;
    }
  }
  if (maximum) {
    if (auto v = decimal_to_int128_round(*maximum, false)) eff_hi = eff_hi ? std::min(*eff_hi, *v) : *v;
  }
  if (exclusive_maximum) {
    if (auto c = decimal_to_int128_round(*exclusive_maximum, true)) {
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

bool decimal_satisfies_bounds(const ParsedDecimal& val, const SchemaNode& node) noexcept {
  if (node.minimum && compare_decimals(val, *node.minimum) < 0) return false;
  if (node.exclusive_minimum && compare_decimals(val, *node.exclusive_minimum) <= 0) return false;
  if (node.maximum && compare_decimals(val, *node.maximum) > 0) return false;
  if (node.exclusive_maximum && compare_decimals(val, *node.exclusive_maximum) >= 0) return false;
  return true;
}

bool num_value_satisfies(const SchemaNode& node, std::string_view buf) noexcept {
  if (!node.minimum && !node.maximum && !node.exclusive_minimum && !node.exclusive_maximum) return true;
  if (buf.empty()) return false;
  int cmp = 0;
  if (node.minimum && (!compare_num_text_to_bound(buf, *node.minimum, cmp) || cmp < 0)) return false;
  if (node.exclusive_minimum && (!compare_num_text_to_bound(buf, *node.exclusive_minimum, cmp) || cmp <= 0)) return false;
  if (node.maximum && (!compare_num_text_to_bound(buf, *node.maximum, cmp) || cmp > 0)) return false;
  if (node.exclusive_maximum && (!compare_num_text_to_bound(buf, *node.exclusive_maximum, cmp) || cmp >= 0)) return false;
  return true;
}

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

  SchemaGraph() { nodes.push_back(SchemaNode{NodeKind::Any}); }

  int add_node(SchemaNode n) {
    const int id = int(nodes.size());
    nodes.push_back(std::move(n));
    return id;
  }

  int compile_type_name(const std::string& t, const ExactJson& obj, const ExactJson& root, int depth) {
    if (t == "object") {
      SchemaNode n;
      n.kind = NodeKind::Object;
      n.min_properties = parse_nonneg_int_kw(obj, "minProperties", 0);
      n.max_properties = parse_nonneg_int_kw(obj, "maxProperties", -1);
      if (n.max_properties >= 0 && n.min_properties > n.max_properties) {
        throw std::invalid_argument("unsatisfiable JSON schema: minProperties > maxProperties");
      }

      std::unordered_set<std::string> req_set;
      if (const ExactJson* req = obj.find("required")) {
        if (!req->is_array()) throw std::invalid_argument("JSON schema 'required' must be an array");
        for (const ExactJson& r : req->arr) {
          if (!r.is_string()) throw std::invalid_argument("JSON schema 'required' entries must be strings");
          req_set.insert(r.str);
        }
      }
      if (const ExactJson* ap = obj.find("additionalProperties")) {
        if (ap->is_bool()) {
          n.additional_properties = ap->b;
          n.additional_node_id = 0;
        } else if (ap->is_object()) {
          n.additional_properties = true;
          n.additional_node_id = compile(*ap, root, depth + 1);
        } else {
          throw std::invalid_argument("JSON schema 'additionalProperties' must be a boolean or object");
        }
      }
      if (const ExactJson* props = obj.find("properties")) {
        if (!props->is_object()) throw std::invalid_argument("JSON schema 'properties' must be an object");
        for (const auto& [pname, pschema] : props->obj) {
          PropDef pd;
          pd.name = pname;
          pd.json_escaped = json_escape_inner(pd.name);
          pd.node_id = compile(pschema, root, depth + 1);
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
      if (const ExactJson* items = obj.find("items")) {
        if (items->is_object()) {
          n.item_node_id = compile(*items, root, depth + 1);
        } else if (items->is_bool()) {
          if (items->b) {
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

  int compile(const ExactJson& schema, const ExactJson& root, int depth = 0) {
    if (depth > 24) {
      throw std::invalid_argument("JSON schema exceeds maximum nesting/recursion depth");
    }
    if (schema.is_bool()) {
      if (schema.b) return 0;
      throw std::invalid_argument("unsatisfiable boolean false JSON schema");
    }
    if (!schema.is_object()) {
      throw std::invalid_argument("JSON schema node must be an object or boolean");
    }

    int validation_kw_count = 0;
    for (const auto& [k, _] : schema.obj) {
      if (is_annotation_keyword(k)) continue;
      if (!is_supported_validation_keyword(k)) {
        throw std::invalid_argument("unsupported JSON schema keyword: '" + k + "'");
      }
      ++validation_kw_count;
    }

    if (const ExactJson* ref = schema.find("$ref")) {
      if (validation_kw_count > 1) {
        throw std::invalid_argument("JSON schema '$ref' with sibling validation keywords is not supported");
      }
      if (!ref->is_string()) {
        throw std::invalid_argument("JSON schema '$ref' must be a string");
      }
      const ExactJson* target = resolve_json_pointer(root, ref->str);
      if (!target) {
        throw std::invalid_argument("unresolved or external JSON schema $ref: '" + ref->str + "'");
      }
      return compile(*target, root, depth + 1);
    }

    const ExactJson* any_of = schema.find("anyOf");
    const ExactJson* one_of = schema.find("oneOf");
    if (any_of || one_of) {
      if (any_of && one_of) {
        throw std::invalid_argument("combining 'anyOf' and 'oneOf' in one schema node is not supported");
      }
      if (validation_kw_count > 1) {
        throw std::invalid_argument("combining 'anyOf'/'oneOf' with sibling validation keywords is not supported");
      }
      const char* ukey = one_of ? "oneOf" : "anyOf";
      const ExactJson* arr = one_of ? one_of : any_of;
      if (!arr->is_array() || arr->arr.empty()) {
        throw std::invalid_argument(std::string("JSON schema '") + ukey + "' must be a non-empty array");
      }
      SchemaNode n;
      n.kind = NodeKind::Union;
      n.is_one_of = (one_of != nullptr);
      for (const ExactJson& sub : arr->arr) {
        n.union_nodes.push_back(compile(sub, root, depth + 1));
      }
      if (n.union_nodes.size() == 1) return n.union_nodes[0];
      return add_node(std::move(n));
    }

    std::vector<std::string> allowed_types;
    if (const ExactJson* ty = schema.find("type")) {
      if (ty->is_string()) {
        if (!is_valid_json_type_name(ty->str)) {
          throw std::invalid_argument("unsupported JSON schema type: '" + ty->str + "'");
        }
        allowed_types.push_back(ty->str);
      } else if (ty->is_array() && !ty->arr.empty()) {
        for (const ExactJson& t : ty->arr) {
          if (!t.is_string() || !is_valid_json_type_name(t.str)) {
            throw std::invalid_argument("JSON schema 'type' array contains invalid type name");
          }
          allowed_types.push_back(t.str);
        }
      } else {
        throw std::invalid_argument("JSON schema 'type' must be a valid type string or non-empty array");
      }
    }

    const ExactJson* c_kw = schema.find("const");
    const ExactJson* e_kw = schema.find("enum");
    if (c_kw || e_kw) {
      if (schema.contains("properties") || schema.contains("required") || schema.contains("additionalProperties") ||
          schema.contains("items")) {
        throw std::invalid_argument("combining 'const'/'enum' with 'properties'/'required'/'items' is not supported");
      }
      std::vector<ExactJson> candidates;
      if (c_kw) candidates.push_back(*c_kw);
      if (e_kw) {
        if (!e_kw->is_array() || e_kw->arr.empty()) {
          throw std::invalid_argument("JSON schema 'enum' must be a non-empty array");
        }
        if (c_kw) {
          bool found = false;
          for (const ExactJson& ev : e_kw->arr) {
            if (exact_json_equal(ev, candidates[0])) {
              found = true;
              break;
            }
          }
          if (!found) candidates.clear();
        } else {
          for (const ExactJson& ev : e_kw->arr) candidates.push_back(ev);
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
      for (const ExactJson& v : candidates) {
        if (!allowed_types.empty()) {
          bool ok_type = false;
          for (const std::string& t : allowed_types) {
            if (exact_json_matches_type_name(v, t)) {
              ok_type = true;
              break;
            }
          }
          if (!ok_type) continue;
        }
        if (v.is_number() && !decimal_satisfies_bounds(v.dec, bound_probe)) continue;
        if (v.is_string()) {
          const std::size_t cp_len = utf8_codepoint_length(v.str);
          if (cp_len < std::size_t(min_len)) continue;
          if (max_len >= 0 && cp_len > std::size_t(max_len)) continue;
        }
        if (v.is_array()) {
          if (int(v.arr.size()) < min_items) continue;
          if (max_items >= 0 && int(v.arr.size()) > max_items) continue;
        }
        if (v.is_object()) {
          if (int(v.obj.size()) < min_props) continue;
          if (max_props >= 0 && int(v.obj.size()) > max_props) continue;
        }
        std::string dumped = exact_json_dump(v);
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

// Exact JSON Schema 2020-12 semantic validator on ExactJson (preserves nested arbitrary-precision decimals).
bool exact_json_matches_node(const SchemaGraph& g, int node_id, const ExactJson& val) {
  const SchemaNode& node = g.nodes[std::size_t(node_id)];
  switch (node.kind) {
    case NodeKind::Any:
      return true;
    case NodeKind::Null:
      return val.is_null();
    case NodeKind::Boolean:
      return val.is_bool();
    case NodeKind::String: {
      if (!val.is_string()) return false;
      const std::size_t len = utf8_codepoint_length(val.str);
      if (len < std::size_t(node.min_length)) return false;
      if (node.max_length >= 0 && len > std::size_t(node.max_length)) return false;
      return true;
    }
    case NodeKind::Integer:
      return val.is_number() && val.dec.is_integral() && decimal_satisfies_bounds(val.dec, node);
    case NodeKind::Number:
      return val.is_number() && decimal_satisfies_bounds(val.dec, node);
    case NodeKind::Array: {
      if (!val.is_array()) return false;
      if (int(val.arr.size()) < node.min_items) return false;
      if (node.max_items >= 0 && int(val.arr.size()) > node.max_items) return false;
      for (const ExactJson& elem : val.arr) {
        if (!exact_json_matches_node(g, node.item_node_id, elem)) return false;
      }
      return true;
    }
    case NodeKind::Object: {
      if (!val.is_object()) return false;
      if (int(val.obj.size()) < node.min_properties) return false;
      if (node.max_properties >= 0 && int(val.obj.size()) > node.max_properties) return false;
      for (const PropDef& pd : node.props) {
        if (pd.required && !val.contains(pd.name)) return false;
      }
      for (const auto& [k, v] : val.obj) {
        bool matched_prop = false;
        for (const PropDef& pd : node.props) {
          if (pd.name == k) {
            matched_prop = true;
            if (!exact_json_matches_node(g, pd.node_id, v)) return false;
            break;
          }
        }
        if (!matched_prop) {
          if (!node.additional_properties) return false;
          if (!exact_json_matches_node(g, node.additional_node_id, v)) return false;
        }
      }
      return true;
    }
    case NodeKind::Enum: {
      for (const ExactJson& ev : node.enum_values) {
        if (exact_json_equal(val, ev)) return true;
      }
      return false;
    }
    case NodeKind::Union: {
      int matched = 0;
      for (int sub_id : node.union_nodes) {
        if (exact_json_matches_node(g, sub_id, val)) ++matched;
      }
      return node.is_one_of ? (matched == 1) : (matched >= 1);
    }
  }
  return false;
}

bool oneof_raw_matches_exactly_one(const SchemaGraph& g, int oneof_node_id, std::string_view raw_text) noexcept {
  if (raw_text.empty()) return false;
  try {
    ExactJson parsed;
    if (!parse_exact_json(raw_text, parsed)) return false;
    const SchemaNode& node = g.nodes[std::size_t(oneof_node_id)];
    int matched = 0;
    for (int sub_id : node.union_nodes) {
      if (exact_json_matches_node(g, sub_id, parsed)) {
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
  Value,
  ObjFirstOrEnd,
  ObjNextKey,
  ObjKeyStr,
  ObjColon,
  ObjCommaOrEnd,
  ArrFirstOrEnd,
  ArrNextVal,
  ArrCommaOrEnd,
  StrBody,
  Num,
  Literal,
  EnumLit,
  OneOf,
};

enum class NumPhase : std::uint8_t {
  AfterMinus,
  Zero,
  IntDigits,
  AfterDot,
  FracDigits,
  AfterExp,
  AfterExpSign,
  ExpDigits,
};

struct OneOfState;

struct Frame {
  FrameKind kind = FrameKind::Value;
  int node_id = 0;
  int val_node_id = 0;
  std::vector<std::uint64_t> seen_prop_words;
  std::vector<std::string> seen_keys;
  int seen_prop_count = 0;
  std::string key_buf;
  int arr_count = 0;
  bool in_escape = false;
  bool expect_low_surrogate = false;
  std::uint8_t uni_rem = 0;
  std::uint16_t uni_val = 0;
  std::uint8_t utf8_rem = 0;
  std::uint8_t utf8_min_next = 0x80;
  std::uint8_t utf8_max_next = 0xBF;
  std::size_t str_len = 0;
  bool integer_only = false;
  NumPhase num_phase = NumPhase::IntDigits;
  std::string num_buf;
  const char* lit_ptr = nullptr;
  std::uint16_t lit_pos = 0;
  std::string enum_buf;
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
        start_value_on_byte(g, sub, node.union_nodes[j], c, next_oneof->alts[j], false);
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
      if (start_value_on_byte(g, copy, sub_id, c, out_branches, allow_empty_ws)) any_ok = true;
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
          step_byte_single(g, copy, c, next_oneof->alts[j], false);
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
        if (step_byte_single(g, exit_st, c, out_branches, allow_empty_ws)) any_ok = true;
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
      if (!step_string_byte(top, c, -1, completed_char, closed_quote)) return false;
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
    PreContent,
    InContent,
    AfterEndMessage,
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

    if (branches_.size() == 1 && !branches_[0].stack.empty()) {
      const Frame& top = branches_[0].stack.back();
      if (top.kind == FrameKind::StrBody && !top.in_escape && !top.expect_low_surrogate && top.uni_rem == 0 &&
          top.utf8_rem == 0) {
        const SchemaNode& snode = graph_->nodes[std::size_t(top.node_id)];
        if (snode.min_length == 0 && snode.max_length < 0) {
          const auto& safe = vocab_->string_safe_mask();
          std::memcpy(words, safe.data(), std::min(nwords, safe.size()) * sizeof(std::uint32_t));
          for (std::int32_t id : vocab_->string_special_candidates()) {
            if (can_advance_any(*graph_, branches_, vocab_->token_bytes(id))) set_bit(words, nwords, id);
          }
          return true;
        }
      }
    }

    const std::bitset<256> first = allowed_first_bytes(*graph_, branches_);
    for (int b = 0; b < 256; ++b) {
      if (!first.test(std::size_t(b))) continue;
      for (std::int32_t id : vocab_->bucket(std::uint8_t(b))) {
        if (can_advance_any(*graph_, branches_, vocab_->token_bytes(id))) set_bit(words, nwords, id);
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
      if (ct_id >= 0 && token_id == ct_id) channel_ = Channel::InContent;
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

std::optional<std::string> extract_raw_json_field_path(std::string_view raw_json,
                                                       std::initializer_list<std::string_view> path) {
  std::string_view cur = raw_json;
  for (std::string_view target_key : path) {
    ExactJsonParser p{cur, 0};
    p.skip_ws();
    if (p.pos >= cur.size() || cur[p.pos] != '{') return std::nullopt;
    ++p.pos;
    p.skip_ws();
    if (p.pos < cur.size() && cur[p.pos] == '}') return std::nullopt;
    std::optional<std::string_view> found;
    while (true) {
      p.skip_ws();
      std::string_view raw_key;
      if (!p.scan_raw_string(raw_key)) return std::nullopt;
      std::string decoded_key;
      if (!unescape_json_string(raw_key, decoded_key)) return std::nullopt;
      p.skip_ws();
      if (p.pos >= cur.size() || cur[p.pos] != ':') return std::nullopt;
      ++p.pos;
      p.skip_ws();
      const std::size_t val_start = p.pos;
      if (!p.skip_value(0)) return std::nullopt;
      const std::size_t val_end = p.pos;
      if (decoded_key == target_key) found = cur.substr(val_start, val_end - val_start);
      p.skip_ws();
      if (p.pos >= cur.size()) return std::nullopt;
      if (cur[p.pos] == '}') {
        ++p.pos;
        break;
      }
      if (cur[p.pos] != ',') return std::nullopt;
      ++p.pos;
    }
    if (!found) return std::nullopt;
    cur = *found;
  }
  return std::string(cur);
}

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
    ExactJson exact_root;
    if (!spec.raw_schema_json.empty()) {
      if (!parse_exact_json(spec.raw_schema_json, exact_root)) {
        throw std::invalid_argument("invalid raw JSON schema syntax");
      }
    } else {
      reject_binary64_rounded_constraints(spec.schema);
      const std::string dumped = spec.schema.dump();
      if (!parse_exact_json(dumped, exact_root)) {
        throw std::invalid_argument("invalid JSON schema syntax");
      }
    }
    graph->root_id = graph->compile(exact_root, exact_root, 0);
  }
  return std::make_shared<JsonFsmImpl>(std::move(vocab), std::move(graph), prefill_ids);
}

}  // namespace ingine::text

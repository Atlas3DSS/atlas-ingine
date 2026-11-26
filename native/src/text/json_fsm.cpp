#include "text/json_fsm.h"

#include "text/chat.h"

#include <algorithm>
#include <bitset>
#include <cctype>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace ingine::text {
namespace {

using Json = nlohmann::ordered_json;

constexpr int kMaxStackDepth = 64;
constexpr std::size_t kMaxUnionBranches = 16;

inline bool is_json_ws(std::uint8_t c) noexcept {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

inline bool is_hex(std::uint8_t c) noexcept {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

inline bool is_digit(std::uint8_t c) noexcept { return c >= '0' && c <= '9'; }

inline void set_bit(std::uint32_t* words, std::size_t word_count, std::int32_t id) noexcept {
  if (id < 0) return;
  const std::size_t w = std::size_t(id) >> 5u;
  if (w < word_count) words[w] |= (std::uint32_t(1u) << (std::uint32_t(id) & 31u));
}

inline bool test_bit(const std::uint32_t* words, std::size_t word_count, std::int32_t id) noexcept {
  if (id < 0) return false;
  const std::size_t w = std::size_t(id) >> 5u;
  return w < word_count && ((words[w] >> (std::uint32_t(id) & 31u)) & 1u) != 0u;
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
  // Array
  int item_node_id = 0;  // Any
  // Enum / Const (exact compact JSON encodings, e.g. "\"red\"", "42", "true", "null")
  std::vector<std::string> enum_literals;
  // Union (anyOf / oneOf / type array)
  std::vector<int> union_nodes;
};

std::string json_escape_inner(const std::string& raw) {
  const std::string dumped = Json(raw).dump();
  if (dumped.size() >= 2 && dumped.front() == '"' && dumped.back() == '"') {
    return dumped.substr(1, dumped.size() - 2);
  }
  return dumped;
}

const Json* resolve_json_pointer(const Json& root, const std::string& ref) {
  if (ref.size() < 2 || ref[0] != '#' || ref[1] != '/') return nullptr;
  const Json* cur = &root;
  std::size_t pos = 2;
  while (pos <= ref.size()) {
    std::size_t slash = ref.find('/', pos);
    std::string seg = (slash == std::string::npos) ? ref.substr(pos) : ref.substr(pos, slash - pos);
    // Unescape RFC 6901 (~1 -> /, ~0 -> ~)
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

struct SchemaGraph {
  std::vector<SchemaNode> nodes;
  int root_id = 0;

  SchemaGraph() {
    // Node 0 is always Any.
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
      std::unordered_set<std::string> req_set;
      if (auto it = obj.find("required"); it != obj.end() && it->is_array()) {
        for (const Json& r : *it) {
          if (r.is_string()) req_set.insert(r.get<std::string>());
        }
      }
      if (auto it = obj.find("additionalProperties"); it != obj.end()) {
        if (it->is_boolean()) {
          n.additional_properties = it->get<bool>();
          n.additional_node_id = 0;
        } else if (it->is_object()) {
          n.additional_properties = true;
          n.additional_node_id = compile(*it, root, depth + 1);
        }
      }
      if (auto it = obj.find("properties"); it != obj.end() && it->is_object()) {
        for (auto pit = it->begin(); pit != it->end(); ++pit) {
          PropDef pd;
          pd.name = pit.key();
          pd.json_escaped = json_escape_inner(pd.name);
          pd.node_id = compile(pit.value(), root, depth + 1);
          pd.required = req_set.count(pd.name) > 0;
          n.props.push_back(std::move(pd));
        }
      }
      // If a required key wasn't listed under properties, add it as Any so schema stays satisfiable.
      for (const std::string& req : req_set) {
        bool found = false;
        for (const PropDef& pd : n.props) {
          if (pd.name == req) {
            found = true;
            break;
          }
        }
        if (!found) {
          n.props.push_back(PropDef{req, json_escape_inner(req), 0, true});
        }
      }
      return add_node(std::move(n));
    }
    if (t == "array") {
      SchemaNode n;
      n.kind = NodeKind::Array;
      if (auto it = obj.find("items"); it != obj.end() && it->is_object()) {
        n.item_node_id = compile(*it, root, depth + 1);
      } else {
        n.item_node_id = 0;
      }
      return add_node(std::move(n));
    }
    if (t == "string") return add_node(SchemaNode{NodeKind::String});
    if (t == "integer") return add_node(SchemaNode{NodeKind::Integer});
    if (t == "number") return add_node(SchemaNode{NodeKind::Number});
    if (t == "boolean") return add_node(SchemaNode{NodeKind::Boolean});
    if (t == "null") return add_node(SchemaNode{NodeKind::Null});
    return 0;
  }

  int compile(const Json& schema, const Json& root, int depth = 0) {
    if (depth > 24 || !schema.is_object()) return 0;

    if (auto it = schema.find("$ref"); it != schema.end() && it->is_string()) {
      if (const Json* target = resolve_json_pointer(root, it->get<std::string>())) {
        return compile(*target, root, depth + 1);
      }
      return 0;
    }

    if (auto it = schema.find("const"); it != schema.end()) {
      SchemaNode n;
      n.kind = NodeKind::Enum;
      n.enum_literals.push_back(it->dump());
      return add_node(std::move(n));
    }

    if (auto it = schema.find("enum"); it != schema.end() && it->is_array() && !it->empty()) {
      SchemaNode n;
      n.kind = NodeKind::Enum;
      for (const Json& v : *it) {
        n.enum_literals.push_back(v.dump());
      }
      return add_node(std::move(n));
    }

    for (const char* ukey : {"anyOf", "oneOf"}) {
      if (auto it = schema.find(ukey); it != schema.end() && it->is_array() && !it->empty()) {
        SchemaNode n;
        n.kind = NodeKind::Union;
        for (const Json& sub : *it) {
          n.union_nodes.push_back(compile(sub, root, depth + 1));
        }
        return add_node(std::move(n));
      }
    }

    if (auto it = schema.find("type"); it != schema.end()) {
      if (it->is_string()) {
        return compile_type_name(it->get<std::string>(), schema, root, depth);
      }
      if (it->is_array() && !it->empty()) {
        SchemaNode n;
        n.kind = NodeKind::Union;
        for (const Json& t : *it) {
          if (t.is_string()) {
            n.union_nodes.push_back(compile_type_name(t.get<std::string>(), schema, root, depth));
          }
        }
        if (n.union_nodes.size() == 1) return n.union_nodes[0];
        if (!n.union_nodes.empty()) return add_node(std::move(n));
      }
    }

    if (schema.contains("properties") || schema.contains("required")) {
      return compile_type_name("object", schema, root, depth);
    }
    if (schema.contains("items")) {
      return compile_type_name("array", schema, root, depth);
    }
    return 0;
  }
};

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
  Num,            // inside JSON number (integer_only flag)
  Literal,        // matching fixed ASCII literal ("true", "false", "null")
  EnumLit,        // matching one of node.enum_literals
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

struct Frame {
  FrameKind kind = FrameKind::Value;
  int node_id = 0;
  int val_node_id = 0;
  // Object tracking (supports up to 64 schema properties via bitset + open set for extra keys)
  std::uint64_t seen_prop_mask = 0;
  std::string key_buf;  // partial escaped key while in ObjKeyStr
  // String / Escape tracking
  bool in_escape = false;
  std::uint8_t uni_rem = 0;
  // Number tracking
  bool integer_only = false;
  NumPhase num_phase = NumPhase::IntDigits;
  // Literal / Enum tracking
  const char* lit_ptr = nullptr;
  std::uint16_t lit_pos = 0;
  std::string enum_buf;
};

struct PdaState {
  std::vector<Frame> stack;

  bool is_done() const noexcept { return stack.empty(); }
};

bool obj_required_satisfied(const SchemaNode& node, std::uint64_t seen_mask) noexcept {
  for (std::size_t i = 0; i < node.props.size() && i < 64; ++i) {
    if (node.props[i].required && ((seen_mask >> i) & 1ull) == 0ull) return false;
  }
  return true;
}

bool obj_has_remaining_key(const SchemaNode& node, std::uint64_t seen_mask) noexcept {
  if (node.additional_properties) return true;
  for (std::size_t i = 0; i < node.props.size() && i < 64; ++i) {
    if (((seen_mask >> i) & 1ull) == 0ull) return true;
  }
  return false;
}

bool num_can_terminate(NumPhase ph) noexcept {
  return ph == NumPhase::Zero || ph == NumPhase::IntDigits || ph == NumPhase::FracDigits || ph == NumPhase::ExpDigits;
}

bool step_byte_single(const SchemaGraph& g, PdaState& st, std::uint8_t c, std::vector<PdaState>& out_branches);

// Expands Value(Union) or starts a concrete value on byte `c`.
bool start_value_on_byte(const SchemaGraph& g, PdaState& st, int node_id, std::uint8_t c,
                         std::vector<PdaState>& out_branches) {
  if (int(st.stack.size()) > kMaxStackDepth) return false;
  const SchemaNode& node = g.nodes[std::size_t(node_id)];

  if (node.kind == NodeKind::Union) {
    bool any_ok = false;
    for (int sub_id : node.union_nodes) {
      if (out_branches.size() >= kMaxUnionBranches) break;
      PdaState copy = st;
      if (start_value_on_byte(g, copy, sub_id, c, out_branches)) {
        any_ok = true;
      }
    }
    return any_ok;
  }

  if (node.kind == NodeKind::Enum) {
    // Check if any enum literal starts with byte c.
    std::string prefix(1, char(c));
    bool exact_done = false;
    bool longer_prefix = false;
    bool non_string_exact = false;
    for (const std::string& lit : node.enum_literals) {
      if (lit == prefix) {
        exact_done = true;
        if (!lit.empty() && lit[0] != '"') non_string_exact = true;
      } else if (lit.size() > 1 && lit[0] == char(c)) {
        longer_prefix = true;
      }
    }
    if (!exact_done && !longer_prefix) return false;
    st.stack.pop_back();  // pop Value
    if (exact_done && !longer_prefix && !non_string_exact) {
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
    st.stack.push_back(std::move(f));
    out_branches.push_back(std::move(st));
    return true;
  }
  if (c == '[' && allows_kind(NodeKind::Array)) {
    st.stack.pop_back();
    Frame f;
    f.kind = FrameKind::ArrFirstOrEnd;
    f.node_id = (node.kind == NodeKind::Array) ? node.item_node_id : 0;
    st.stack.push_back(std::move(f));
    out_branches.push_back(std::move(st));
    return true;
  }
  if (c == '"' && allows_kind(NodeKind::String)) {
    st.stack.pop_back();
    Frame f;
    f.kind = FrameKind::StrBody;
    st.stack.push_back(std::move(f));
    out_branches.push_back(std::move(st));
    return true;
  }
  if ((c == '-' || is_digit(c)) && (allows_kind(NodeKind::Number) || allows_kind(NodeKind::Integer))) {
    st.stack.pop_back();
    Frame f;
    f.kind = FrameKind::Num;
    f.integer_only = (node.kind == NodeKind::Integer);
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

bool step_byte_single(const SchemaGraph& g, PdaState& st, std::uint8_t c, std::vector<PdaState>& out_branches) {
  if (st.stack.empty()) {
    if (is_json_ws(c)) {
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
      return start_value_on_byte(g, st, top.node_id, c, out_branches);
    }

    case FrameKind::ObjFirstOrEnd: {
      if (is_json_ws(c)) {
        out_branches.push_back(std::move(st));
        return true;
      }
      const SchemaNode& node = g.nodes[std::size_t(top.node_id)];
      if (c == '}') {
        if (!obj_required_satisfied(node, top.seen_prop_mask)) return false;
        st.stack.pop_back();
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c == '"') {
        if (!obj_has_remaining_key(node, top.seen_prop_mask)) return false;
        top.kind = FrameKind::ObjKeyStr;
        top.key_buf.clear();
        top.in_escape = false;
        top.uni_rem = 0;
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
        if (!obj_has_remaining_key(node, top.seen_prop_mask)) return false;
        top.kind = FrameKind::ObjKeyStr;
        top.key_buf.clear();
        top.in_escape = false;
        top.uni_rem = 0;
        out_branches.push_back(std::move(st));
        return true;
      }
      return false;
    }

    case FrameKind::ObjKeyStr: {
      const SchemaNode& node = g.nodes[std::size_t(top.node_id)];
      if (!node.additional_properties) {
        // Strict schema keys: match exact JSON-escaped property names among unused properties.
        if (c == '"') {
          for (std::size_t i = 0; i < node.props.size(); ++i) {
            if (i < 64 && ((top.seen_prop_mask >> i) & 1ull) != 0ull) continue;
            if (node.props[i].json_escaped == top.key_buf) {
              if (i < 64) top.seen_prop_mask |= (1ull << i);
              top.val_node_id = node.props[i].node_id;
              top.kind = FrameKind::ObjColon;
              top.key_buf.clear();
              out_branches.push_back(std::move(st));
              return true;
            }
          }
          return false;
        }
        // Check if appending byte c is a valid prefix of at least one unused property key.
        top.key_buf.push_back(char(c));
        bool any_prefix = false;
        for (std::size_t i = 0; i < node.props.size(); ++i) {
          if (i < 64 && ((top.seen_prop_mask >> i) & 1ull) != 0ull) continue;
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

      // Open object key string (additionalProperties = true or Any object).
      if (top.uni_rem > 0) {
        if (!is_hex(c)) return false;
        --top.uni_rem;
        top.key_buf.push_back(char(c));
        out_branches.push_back(std::move(st));
        return true;
      }
      if (top.in_escape) {
        if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') {
          top.in_escape = false;
          top.key_buf.push_back(char(c));
          out_branches.push_back(std::move(st));
          return true;
        }
        if (c == 'u') {
          top.in_escape = false;
          top.uni_rem = 4;
          top.key_buf.push_back(char(c));
          out_branches.push_back(std::move(st));
          return true;
        }
        return false;
      }
      if (c == '\\') {
        top.in_escape = true;
        top.key_buf.push_back(char(c));
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c == '"') {
        int val_node = node.additional_node_id;
        for (std::size_t i = 0; i < node.props.size(); ++i) {
          if (node.props[i].json_escaped == top.key_buf) {
            if (i < 64) {
              if (((top.seen_prop_mask >> i) & 1ull) != 0ull) return false;  // reject duplicate schema key
              top.seen_prop_mask |= (1ull << i);
            }
            val_node = node.props[i].node_id;
            break;
          }
        }
        top.val_node_id = val_node;
        top.kind = FrameKind::ObjColon;
        top.key_buf.clear();
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c < 0x20) return false;
      if (top.key_buf.size() < 256) top.key_buf.push_back(char(c));
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
        if (!obj_required_satisfied(node, top.seen_prop_mask)) return false;
        st.stack.pop_back();
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c == ',') {
        if (!obj_has_remaining_key(node, top.seen_prop_mask)) return false;
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
      if (c == ']') {
        st.stack.pop_back();
        out_branches.push_back(std::move(st));
        return true;
      }
      const int item_node = top.node_id;
      top.kind = FrameKind::ArrCommaOrEnd;
      Frame vf;
      vf.kind = FrameKind::Value;
      vf.node_id = item_node;
      st.stack.push_back(std::move(vf));
      return start_value_on_byte(g, st, item_node, c, out_branches);
    }

    case FrameKind::ArrNextVal: {
      if (is_json_ws(c)) {
        out_branches.push_back(std::move(st));
        return true;
      }
      const int item_node = top.node_id;
      top.kind = FrameKind::ArrCommaOrEnd;
      Frame vf;
      vf.kind = FrameKind::Value;
      vf.node_id = item_node;
      st.stack.push_back(std::move(vf));
      return start_value_on_byte(g, st, item_node, c, out_branches);
    }

    case FrameKind::ArrCommaOrEnd: {
      if (is_json_ws(c)) {
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c == ']') {
        st.stack.pop_back();
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c == ',') {
        top.kind = FrameKind::ArrNextVal;
        out_branches.push_back(std::move(st));
        return true;
      }
      return false;
    }

    case FrameKind::StrBody: {
      if (top.uni_rem > 0) {
        if (!is_hex(c)) return false;
        --top.uni_rem;
        out_branches.push_back(std::move(st));
        return true;
      }
      if (top.in_escape) {
        if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' || c == 'n' || c == 'r' || c == 't') {
          top.in_escape = false;
          out_branches.push_back(std::move(st));
          return true;
        }
        if (c == 'u') {
          top.in_escape = false;
          top.uni_rem = 4;
          out_branches.push_back(std::move(st));
          return true;
        }
        return false;
      }
      if (c == '\\') {
        top.in_escape = true;
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c == '"') {
        st.stack.pop_back();
        out_branches.push_back(std::move(st));
        return true;
      }
      if (c < 0x20) return false;
      out_branches.push_back(std::move(st));
      return true;
    }

    case FrameKind::Num: {
      switch (top.num_phase) {
        case NumPhase::AfterMinus:
          if (c == '0') {
            top.num_phase = NumPhase::Zero;
            out_branches.push_back(std::move(st));
            return true;
          }
          if (c >= '1' && c <= '9') {
            top.num_phase = NumPhase::IntDigits;
            out_branches.push_back(std::move(st));
            return true;
          }
          return false;
        case NumPhase::Zero:
          if (!top.integer_only && c == '.') {
            top.num_phase = NumPhase::AfterDot;
            out_branches.push_back(std::move(st));
            return true;
          }
          if (!top.integer_only && (c == 'e' || c == 'E')) {
            top.num_phase = NumPhase::AfterExp;
            out_branches.push_back(std::move(st));
            return true;
          }
          // Delimiter terminates number; re-dispatch byte to parent frame!
          st.stack.pop_back();
          return step_byte_single(g, st, c, out_branches);
        case NumPhase::IntDigits:
          if (is_digit(c)) {
            out_branches.push_back(std::move(st));
            return true;
          }
          if (!top.integer_only && c == '.') {
            top.num_phase = NumPhase::AfterDot;
            out_branches.push_back(std::move(st));
            return true;
          }
          if (!top.integer_only && (c == 'e' || c == 'E')) {
            top.num_phase = NumPhase::AfterExp;
            out_branches.push_back(std::move(st));
            return true;
          }
          st.stack.pop_back();
          return step_byte_single(g, st, c, out_branches);
        case NumPhase::AfterDot:
          if (is_digit(c)) {
            top.num_phase = NumPhase::FracDigits;
            out_branches.push_back(std::move(st));
            return true;
          }
          return false;
        case NumPhase::FracDigits:
          if (is_digit(c)) {
            out_branches.push_back(std::move(st));
            return true;
          }
          if (c == 'e' || c == 'E') {
            top.num_phase = NumPhase::AfterExp;
            out_branches.push_back(std::move(st));
            return true;
          }
          st.stack.pop_back();
          return step_byte_single(g, st, c, out_branches);
        case NumPhase::AfterExp:
          if (c == '+' || c == '-') {
            top.num_phase = NumPhase::AfterExpSign;
            out_branches.push_back(std::move(st));
            return true;
          }
          if (is_digit(c)) {
            top.num_phase = NumPhase::ExpDigits;
            out_branches.push_back(std::move(st));
            return true;
          }
          return false;
        case NumPhase::AfterExpSign:
          if (is_digit(c)) {
            top.num_phase = NumPhase::ExpDigits;
            out_branches.push_back(std::move(st));
            return true;
          }
          return false;
        case NumPhase::ExpDigits:
          if (is_digit(c)) {
            out_branches.push_back(std::move(st));
            return true;
          }
          st.stack.pop_back();
          return step_byte_single(g, st, c, out_branches);
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
      bool non_string_exact = false;
      for (const std::string& lit : node.enum_literals) {
        if (lit == next) {
          matched_exact = true;
          if (!lit.empty() && lit[0] != '"') non_string_exact = true;
        } else if (lit.size() > next.size() && lit.compare(0, next.size(), next) == 0) {
          matched_longer = true;
        }
      }
      if (matched_exact || matched_longer) {
        if (matched_exact && !matched_longer && !non_string_exact) {
          st.stack.pop_back();
        } else {
          top.enum_buf = std::move(next);
        }
        out_branches.push_back(std::move(st));
        return true;
      }
      // If current enum_buf is already a complete literal (e.g. number/bool/null), pop and re-dispatch delimiter c.
      bool cur_exact = false;
      for (const std::string& lit : node.enum_literals) {
        if (lit == top.enum_buf) {
          cur_exact = true;
          break;
        }
      }
      if (cur_exact) {
        st.stack.pop_back();
        return step_byte_single(g, st, c, out_branches);
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
    if (f.kind == FrameKind::Num) return num_can_terminate(f.num_phase);
    if (f.kind == FrameKind::EnumLit) {
      const SchemaNode& node = g.nodes[std::size_t(f.node_id)];
      for (const std::string& lit : node.enum_literals) {
        if (lit == f.enum_buf) return true;
      }
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

// Compute exact bitset of allowed first bytes for fast bucket filtering.
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

    // Check if prefill_ids already placed the model inside <|content_text|>.
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
    // Fallback for synthetic/minimal tokenizers without Inkling special tokens: constrain immediately.
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

    // Channel::InContent
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

    // Fast path: single branch inside normal JSON string value (no pending escape/unicode).
    if (branches_.size() == 1 && !branches_[0].stack.empty()) {
      const Frame& top = branches_[0].stack.back();
      if (top.kind == FrameKind::StrBody && !top.in_escape && top.uni_rem == 0) {
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

    // General path: filter vocabulary by exact allowed first bytes.
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

    // Safety net: never return an all-zero mask (prevents NaN / deadlock on pathological schemas).
    bool any_set = false;
    for (std::size_t i = 0; i < nwords; ++i) {
      if (words[i] != 0u) {
        any_set = true;
        break;
      }
    }
    if (!any_set) {
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

    // Channel::InContent
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
    bool str_safe = true;
    for (char ch : b) {
      const auto u = static_cast<std::uint8_t>(ch);
      if (!is_json_ws(u)) all_ws = false;
      if (u < 0x20 || u == '"' || u == '\\') str_safe = false;
    }
    if (all_ws) set_bit(idx->ws_mask_.data(), idx->mask_words_, id);
    if (str_safe) {
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

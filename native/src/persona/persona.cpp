#include "ingine/persona.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ingine::persona {
namespace {

using Json = nlohmann::ordered_json;

// ---- Self-contained SHA-256 (FIPS 180-4) --------------------------------------------------------------------

struct Sha256 {
  std::uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  std::uint8_t buf[64] = {};
  std::size_t buf_len = 0;
  std::uint64_t total_bits = 0;

  static std::uint32_t rotr(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32u - n)); }

  void block(const std::uint8_t b[64]) {
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
      w[i] = (std::uint32_t(b[4 * i]) << 24) | (std::uint32_t(b[4 * i + 1]) << 16) |
             (std::uint32_t(b[4 * i + 2]) << 8) | std::uint32_t(b[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = hh + S1 + ch + k[i] + w[i];
      const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & bb) ^ (a & c) ^ (bb & c);
      const std::uint32_t temp2 = S0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = bb;
      bb = a;
      a = temp1 + temp2;
    }
    h[0] += a;
    h[1] += bb;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }

  void update(const void* data, std::size_t n) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    total_bits += std::uint64_t(n) * 8u;
    while (n > 0) {
      const std::size_t take = std::min(n, 64 - buf_len);
      std::memcpy(buf + buf_len, p, take);
      buf_len += take;
      p += take;
      n -= take;
      if (buf_len == 64) {
        block(buf);
        buf_len = 0;
      }
    }
  }

  std::string finish_hex() {
    buf[buf_len++] = 0x80u;
    if (buf_len > 56) {
      std::memset(buf + buf_len, 0, 64 - buf_len);
      block(buf);
      buf_len = 0;
    }
    std::memset(buf + buf_len, 0, 56 - buf_len);
    for (int i = 0; i < 8; ++i) buf[56 + i] = std::uint8_t((total_bits >> (56 - 8 * i)) & 0xffu);
    block(buf);
    char out[65];
    for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
    out[64] = '\0';
    return std::string(out, 64);
  }
};

// Recursively canonicalise JSON (keys sorted lexicographically, compact encoding) for deterministic SHA-256.
nlohmann::json to_canonical(const Json& j) {
  if (j.is_object()) {
    nlohmann::json out = nlohmann::json::object();
    for (auto it = j.begin(); it != j.end(); ++it) out[it.key()] = to_canonical(it.value());
    return out;
  }
  if (j.is_array()) {
    nlohmann::json out = nlohmann::json::array();
    for (const auto& el : j) out.push_back(to_canonical(el));
    return out;
  }
  if (j.is_null()) return nullptr;
  if (j.is_boolean()) return j.get<bool>();
  if (j.is_number_integer()) return j.get<std::int64_t>();
  if (j.is_number_unsigned()) return j.get<std::uint64_t>();
  if (j.is_number_float()) return j.get<double>();
  if (j.is_string()) return j.get<std::string>();
  return nullptr;
}

bool is_sha256_hex(const std::string& s) {
  if (s.size() != 64) return false;
  for (char c : s) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
  }
  return true;
}

bool is_denied_sealed_path(const std::string& path) {
  std::string lower = path;
  for (char& c : lower) c = char(std::tolower(static_cast<unsigned char>(c)));
  static const char* kDenied[] = {"persona_battery", "heldout", "confirmatory_v2", "sealed", "seal.json"};
  for (const char* needle : kDenied) {
    if (lower.find(needle) != std::string::npos) return true;
  }
  return false;
}

PersonaStatus parse_status(const std::string& s, const std::string& where) {
  if (s == "probe") return PersonaStatus::Probe;
  if (s == "validated") return PersonaStatus::Validated;
  if (s == "product") return PersonaStatus::Product;
  throw std::invalid_argument(where + ": invalid status '" + s + "' (expected probe|validated|product)");
}

PersonaStatus min_status(PersonaStatus a, PersonaStatus b) {
  return static_cast<int>(a) <= static_cast<int>(b) ? a : b;
}

std::string req_str(const Json& obj, const char* key, const std::string& where) {
  auto it = obj.find(key);
  if (it == obj.end() || !it->is_string() || it->get_ref<const std::string&>().empty()) {
    throw std::invalid_argument(where + ": missing or empty string '" + key + "'");
  }
  return it->get<std::string>();
}

int req_int(const Json& obj, const char* key, int lo, int hi, const std::string& where) {
  auto it = obj.find(key);
  if (it == obj.end() || !it->is_number_integer()) {
    throw std::invalid_argument(where + ": missing integer '" + key + "'");
  }
  const int v = it->get<int>();
  if (v < lo || v > hi) {
    throw std::invalid_argument(where + ": '" + key + "' = " + std::to_string(v) + " outside [" +
                                std::to_string(lo) + ", " + std::to_string(hi) + "]");
  }
  return v;
}

double req_num(const Json& obj, const char* key, const std::string& where) {
  auto it = obj.find(key);
  if (it == obj.end() || !it->is_number()) {
    throw std::invalid_argument(where + ": missing number '" + key + "'");
  }
  const double v = it->get<double>();
  if (!std::isfinite(v)) throw std::invalid_argument(where + ": '" + key + "' must be finite");
  return v;
}

bool validate_noop_check(const Json& nc, bool strict_zero, std::string& reason) {
  if (!nc.is_object()) {
    reason = "noop_check must be an object";
    return false;
  }
  for (const char* k : {"flag", "fused_path", "hardware"}) {
    if (!nc.contains(k) || !nc[k].is_string() || nc[k].get_ref<const std::string&>().empty()) {
      reason = std::string("noop_check missing required string '") + k + "'";
      return false;
    }
  }
  if (!nc.contains("visits") || !nc["visits"].is_array()) {
    reason = "noop_check must specify 'visits' array containing {3, 6, 10, 18, 29, 44}";
    return false;
  }
  std::set<int> got_visits;
  for (const auto& v : nc["visits"]) {
    if (!v.is_number_integer()) {
      reason = "noop_check.visits entries must be integers in 0..47";
      return false;
    }
    const int iv = v.get<int>();
    if (iv < 0 || iv >= kVisits) {
      reason = "noop_check.visits entry outside 0..47";
      return false;
    }
    got_visits.insert(iv);
  }
  static constexpr int kRequiredNoOpVisits[6] = {3, 6, 10, 18, 29, 44};
  for (int req_v : kRequiredNoOpVisits) {
    if (!got_visits.contains(req_v)) {
      reason = "noop_check.visits missing required boundary/pass visit " + std::to_string(req_v) +
               " (must contain {3, 6, 10, 18, 29, 44})";
      return false;
    }
  }
  if (!nc.contains("kl") || !nc["kl"].is_number() || !nc.contains("delta_nll") || !nc["delta_nll"].is_number()) {
    reason = "noop_check missing numeric kl / delta_nll";
    return false;
  }
  if (strict_zero && (nc["kl"].get<double>() != 0.0 || nc["delta_nll"].get<double>() != 0.0)) {
    reason = "noop_check kl and delta_nll must be exactly 0.0";
    return false;
  }
  return true;
}

bool validate_measured_effect(const Json& me, std::string& reason) {
  if (!me.is_object()) {
    reason = "measured_effect must be an object";
    return false;
  }
  for (const char* k : {"summary", "metric", "engine", "weights_sha256", "floor_kind"}) {
    if (!me.contains(k) || !me[k].is_string() || me[k].get_ref<const std::string&>().empty()) {
      reason = std::string("measured_effect missing '") + k + "'";
      return false;
    }
  }
  const std::string fk = me["floor_kind"].get<std::string>();
  if (fk != "slot_permute_null" && fk != "sample_to_sample" && fk != "self_swap" &&
      fk != "same_character_other_question") {
    reason = "measured_effect.floor_kind invalid: '" + fk + "'";
    return false;
  }
  if (!is_sha256_hex(me["weights_sha256"].get<std::string>())) {
    reason = "measured_effect.weights_sha256 must be 64 lowercase hex chars";
    return false;
  }
  if (!me.contains("value") || !me["value"].is_number() || !me.contains("floor") || !me["floor"].is_number()) {
    reason = "measured_effect missing numeric value / floor";
    return false;
  }
  if (!me.contains("floor_n") || !me["floor_n"].is_number_integer() || me["floor_n"].get<int>() < 1 ||
      !me.contains("sample_count") || !me["sample_count"].is_number_integer() || me["sample_count"].get<int>() < 1) {
    reason = "measured_effect requires positive integer floor_n and sample_count";
    return false;
  }
  const double val = me["value"].get<double>();
  const double flr = me["floor"].get<double>();
  if (!std::isfinite(val) || !std::isfinite(flr) || flr < 0.0) {
    reason = "measured_effect value/floor non-finite or negative floor";
    return false;
  }
  if (std::fabs(val) <= flr) {
    reason = "measured_effect |value| does not exceed floor";
    return false;
  }
  return true;
}

bool validate_refusal_check(const Json& rc, std::string& reason) {
  if (!rc.is_object()) {
    reason = "refusal_checked must be an object";
    return false;
  }
  if (!rc.contains("passed") || !rc["passed"].is_boolean() || !rc["passed"].get<bool>()) {
    reason = "refusal_checked.passed must be true";
    return false;
  }
  if (!rc.contains("evidence_file") || !rc["evidence_file"].is_string() ||
      rc["evidence_file"].get_ref<const std::string&>().empty()) {
    reason = "refusal_checked.evidence_file required";
    return false;
  }
  if (!rc.contains("rate_before") || !rc["rate_before"].is_number() || !rc.contains("rate_after") ||
      !rc["rate_after"].is_number()) {
    reason = "refusal_checked requires numeric rate_before and rate_after";
    return false;
  }
  return true;
}

InterventionType parse_type(const std::string& s, const std::string& where) {
  if (s == "router_ablate") return InterventionType::RouterAblate;
  if (s == "router_force") return InterventionType::RouterForce;
  if (s == "router_slot_permute") return InterventionType::RouterSlotPermute;
  if (s == "route_replay") return InterventionType::RouteReplay;
  if (s == "neuron_affine") return InterventionType::NeuronAffine;
  if (s == "depth_replace") return InterventionType::DepthReplace;
  if (s == "residual_steer") return InterventionType::ResidualSteer;
  if (s == "tap_expert") return InterventionType::TapExpert;
  if (s == "tap_routing") return InterventionType::TapRouting;
  if (s == "router_lowrank_fit") return InterventionType::RouterLowRankFit;
  if (s == "expert_down_lowrank_fit") return InterventionType::ExpertDownLowRankFit;
  throw std::invalid_argument(where + ": unknown intervention type '" + s + "'");
}

void cross_check_visit(const InterventionSpec& sp, const std::string& where) {
  if (sp.all_visits || sp.visit < 0) return;
  const VisitCoord vc = visit_coord(sp.visit);
  if (sp.physical_block && *sp.physical_block != vc.physical_block) {
    throw std::invalid_argument(where + ": visit " + std::to_string(sp.visit) + " maps to physical_block " +
                                std::to_string(vc.physical_block) + ", not " + std::to_string(*sp.physical_block));
  }
  if (sp.loop_pass && *sp.loop_pass != vc.loop_pass) {
    throw std::invalid_argument(where + ": visit " + std::to_string(sp.visit) + " maps to loop_pass " +
                                std::to_string(vc.loop_pass) + ", not " + std::to_string(*sp.loop_pass));
  }
}

bool unit_matches_refusal(const RefusalUnit& ru, int visit, int expert, int neuron) {
  if (ru.visit >= 0 && visit >= 0 && ru.visit != visit) return false;
  if (ru.physical_block >= 0 && visit >= 0 && visit_coord(visit).physical_block != ru.physical_block) return false;
  if (ru.expert >= 0 && expert >= 0 && ru.expert != expert) return false;
  if (ru.neuron >= 0 && neuron >= 0 && ru.neuron != neuron) return false;
  return true;
}

}  // namespace

VisitCoord visit_coord(int visit) {
  if (visit < 0 || visit >= kVisits) {
    throw std::invalid_argument("visit " + std::to_string(visit) + " outside 0..47");
  }
  if (visit < 6) return {visit, 0};
  if (visit < 18) return {visit, 0};
  if (visit < 30) return {6 + (visit - 18), 1};
  if (visit < 42) return {6 + (visit - 30), 2};
  return {18 + (visit - 42), 0};
}

std::string sha256_hex(const void* data, std::size_t bytes) {
  Sha256 s;
  s.update(data, bytes);
  return s.finish_hex();
}

std::string sha256_hex(const std::string& s) { return sha256_hex(s.data(), s.size()); }

std::string sha256_file_hex(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open file for sha256: " + path.string());
  Sha256 s;
  char buf[65536];
  while (in.good()) {
    in.read(buf, sizeof(buf));
    const std::streamsize got = in.gcount();
    if (got > 0) s.update(buf, std::size_t(got));
  }
  return s.finish_hex();
}

std::uint64_t cache_key_from_hex(const std::string& hex) {
  std::uint64_t v = 0;
  const std::size_t n = std::min<std::size_t>(16, hex.size());
  for (std::size_t i = 0; i < n; ++i) {
    const char c = hex[i];
    std::uint64_t d = 0;
    if (c >= '0' && c <= '9') d = std::uint64_t(c - '0');
    else if (c >= 'a' && c <= 'f') d = std::uint64_t(10 + c - 'a');
    else if (c >= 'A' && c <= 'F') d = std::uint64_t(10 + c - 'A');
    v = (v << 4) | d;
  }
  return v == 0 ? 1ull : v;
}

RefusalMap RefusalMap::default_map() {
  RefusalMap m;
  m.protected_coords.push_back(kForbiddenCoord);
  return m;
}

RefusalMap RefusalMap::load(const std::filesystem::path& path) {
  RefusalMap m = default_map();
  if (path.empty()) return m;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open refusal map: " + path.string());
  const Json j = Json::parse(in);
  if (!j.is_object()) throw std::invalid_argument("refusal map must be a JSON object");
  if (j.contains("protected_units")) {
    if (!j["protected_units"].is_array()) throw std::invalid_argument("refusal map 'protected_units' must be an array");
    for (const auto& u : j["protected_units"]) {
      if (!u.is_object()) throw std::invalid_argument("refusal map unit must be an object");
      RefusalUnit ru;
      if (u.contains("visit") && !u["visit"].is_null()) ru.visit = req_int(u, "visit", 0, kVisits - 1, "refusal_map");
      if (u.contains("physical_block") && !u["physical_block"].is_null())
        ru.physical_block = req_int(u, "physical_block", 0, kLayers - 1, "refusal_map");
      if (u.contains("expert") && !u["expert"].is_null()) ru.expert = req_int(u, "expert", 0, kExperts - 1, "refusal_map");
      if (u.contains("neuron") && !u["neuron"].is_null()) ru.neuron = req_int(u, "neuron", 0, kInter - 1, "refusal_map");
      if (ru.visit < 0 && ru.physical_block < 0 && ru.expert < 0) {
        throw std::invalid_argument("refusal map unit must specify at least visit, physical_block, or expert");
      }
      m.protected_units.push_back(ru);
    }
  }
  if (j.contains("protected_coords")) {
    if (!j["protected_coords"].is_array())
      throw std::invalid_argument("refusal map 'protected_coords' must be an array");
    for (const auto& c : j["protected_coords"]) {
      if (!c.is_number_integer()) throw std::invalid_argument("refusal map coord must be integer");
      const int coord = c.get<int>();
      if (coord < 0 || coord >= kHidden) throw std::invalid_argument("refusal map coord out of range 0..2303");
      if (std::find(m.protected_coords.begin(), m.protected_coords.end(), coord) == m.protected_coords.end()) {
        m.protected_coords.push_back(coord);
      }
    }
  }
  return m;
}

bool PersonaProfile::modifies_model() const {
  return std::any_of(interventions.begin(), interventions.end(),
                     [](const InterventionSpec& sp) { return !is_read_only_tap(sp.type); });
}

bool PersonaProfile::bypasses_prefix_reuse() const {
  for (const auto& sp : interventions) {
    if (sp.type == InterventionType::RouteReplay || sp.type == InterventionType::DepthReplace ||
        is_read_only_tap(sp.type)) {
      return true;
    }
  }
  return false;
}

void validate_profile_safety(const PersonaProfile& profile, const RefusalMap& refusal_map) {
  std::set<std::tuple<int, int, int>> seen_neurons;
  for (const auto& sp : profile.interventions) {
    const std::string where = "profile '" + profile.name + "' intervention '" + sp.id + "'";
    cross_check_visit(sp, where);
    if (is_read_only_tap(sp.type)) continue;  // read-only taps never modify the model

    // Coordinate 2027 + protected_coords guard for residual steering
    if (sp.type == InterventionType::ResidualSteer && !sp.vector.empty()) {
      if (int(sp.vector.size()) != kHidden) {
        throw std::invalid_argument(where + ": residual_steer vector length must be 2304");
      }
      if (sp.vector[kForbiddenCoord] != 0.0f) {
        throw std::invalid_argument(where + ": forbidden perturbation of coordinate d_2027");
      }
      for (int c : refusal_map.protected_coords) {
        if (c >= 0 && c < kHidden && sp.vector[std::size_t(c)] != 0.0f) {
          throw std::invalid_argument(where + ": residual_steer touches protected coordinate d_" + std::to_string(c));
        }
      }
    }

    // Duplicate (visit, expert, neuron) check for 3b
    if (sp.type == InterventionType::NeuronAffine) {
      const auto key = std::make_tuple(sp.visit, sp.expert, sp.neuron);
      if (!seen_neurons.insert(key).second) {
        throw std::invalid_argument(where + ": duplicate neuron_affine target (visit, expert, neuron)");
      }
    }

    // Refusal-linked unit deny-list guard
    if (sp.type == InterventionType::RouterAblate || sp.type == InterventionType::RouterForce ||
        sp.type == InterventionType::NeuronAffine) {
      for (const auto& ru : refusal_map.protected_units) {
        if (unit_matches_refusal(ru, sp.visit, sp.expert, sp.type == InterventionType::NeuronAffine ? sp.neuron : -1)) {
          throw std::invalid_argument(where + ": targets refusal-protected unit (visit=" + std::to_string(sp.visit) +
                                      ", expert=" + std::to_string(sp.expert) + ")");
        }
      }
    }
    if (sp.type == InterventionType::ExpertDownLowRankFit && sp.physical_block) {
      for (int e : sp.experts) {
        for (const auto& ru : refusal_map.protected_units) {
          if (ru.physical_block >= 0 && ru.physical_block != *sp.physical_block) continue;
          if (ru.visit >= 0 && visit_coord(ru.visit).physical_block != *sp.physical_block) continue;
          if (ru.expert >= 0 && ru.expert != e) continue;
          throw std::invalid_argument(where + ": expert_down_lowrank_fit targets refusal-protected expert " +
                                      std::to_string(e) + " in physical_block " + std::to_string(*sp.physical_block));
        }
      }
    }
  }
}

std::shared_ptr<const PersonaProfile> PersonaCatalog::resolve(const std::string& requested) const {
  const std::string name = requested.empty() ? default_profile : requested;
  if (name.empty() || name == "none") return nullptr;
  auto it = profiles.find(name);
  if (it == profiles.end()) {
    throw std::invalid_argument("unknown persona profile '" + name + "'");
  }
  return it->second;
}

PersonaCatalog PersonaCatalog::load(const std::filesystem::path& path, bool probe_mode, const RefusalMap& refusal_map,
                                    const ModelRuntimeIdentity& runtime_id) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open persona spec: " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return parse_json(ss.str(), probe_mode, refusal_map, runtime_id, path.parent_path());
}

PersonaCatalog PersonaCatalog::parse_json(const std::string& raw_json, bool probe_mode, const RefusalMap& refusal_map,
                                          const ModelRuntimeIdentity& runtime_id,
                                          const std::filesystem::path& base_dir) {
  const Json root = Json::parse(raw_json);
  if (!root.is_object()) throw std::invalid_argument("persona spec must be a JSON object");
  if (!root.contains("schema_version") || root["schema_version"] != "atlas-persona-spec/v1") {
    throw std::invalid_argument("persona spec requires schema_version == 'atlas-persona-spec/v1'");
  }
  PersonaCatalog cat;
  cat.spec_sha256 = sha256_hex(raw_json);
  cat.probe_mode = probe_mode;

  if (!root.contains("spec_metadata") || !root["spec_metadata"].is_object()) {
    throw std::invalid_argument("persona spec requires 'spec_metadata' object (created, author)");
  }
  const Json& meta = root["spec_metadata"];
  cat.created = req_str(meta, "created", "spec_metadata");
  cat.author = req_str(meta, "author", "spec_metadata");
  if (meta.contains("spec_id") && meta["spec_id"].is_string()) cat.spec_id = meta["spec_id"].get<std::string>();

  if (!root.contains("profiles") || !root["profiles"].is_object()) {
    throw std::invalid_argument("persona spec requires 'profiles' object");
  }
  if (meta.contains("profiles_sha256") && !meta["profiles_sha256"].is_null()) {
    const std::string want = req_str(meta, "profiles_sha256", "spec_metadata");
    if (!is_sha256_hex(want)) throw std::invalid_argument("spec_metadata.profiles_sha256 must be 64 lowercase hex");
    const std::string got = sha256_hex(to_canonical(root["profiles"]).dump());
    if (got != want) {
      throw std::invalid_argument("spec_metadata.profiles_sha256 mismatch: expected " + want + ", got " + got);
    }
  }
  cat.default_profile = root.contains("default_profile") && root["default_profile"].is_string()
                            ? root["default_profile"].get<std::string>()
                            : "none";

  for (auto it = root["profiles"].begin(); it != root["profiles"].end(); ++it) {
    const std::string pname = it.key();
    const Json& pjson = it.value();
    const std::string pwhere = "profile '" + pname + "'";
    if (!pjson.is_object()) throw std::invalid_argument(pwhere + " must be a JSON object");

    auto prof = std::make_shared<PersonaProfile>();
    prof->name = pname;
    prof->spec_sha256 = cat.spec_sha256;
    if (pjson.contains("description") && pjson["description"].is_string()) {
      prof->description = pjson["description"].get<std::string>();
    }
    if (pjson.contains("category") && pjson["category"].is_string()) {
      prof->declared_category = parse_status(pjson["category"].get<std::string>(), pwhere);
    }

    // Compute canonical profile fingerprint excluding 'persona_fingerprint' itself.
    Json p_for_hash = pjson;
    p_for_hash.erase("persona_fingerprint");
    prof->fingerprint_sha256 = sha256_hex(to_canonical(p_for_hash).dump());
    prof->cache_key = pname == "none" ? 0ull : cache_key_from_hex(prof->fingerprint_sha256);
    if (pjson.contains("persona_fingerprint") && !pjson["persona_fingerprint"].is_null()) {
      const std::string want = req_str(pjson, "persona_fingerprint", pwhere);
      if (want != prof->fingerprint_sha256) {
        throw std::invalid_argument(pwhere + ": persona_fingerprint mismatch (expected " + want + ", computed " +
                                    prof->fingerprint_sha256 + ")");
      }
    }

    if (!pjson.contains("interventions") || !pjson["interventions"].is_array()) {
      throw std::invalid_argument(pwhere + ": missing 'interventions' array");
    }
    if (pname == "none" && !pjson["interventions"].empty()) {
      throw std::invalid_argument("profile 'none' must have an empty interventions array");
    }

    PersonaStatus eff = prof->declared_category;
    std::string downgrade;
    auto note_downgrade = [&](const std::string& r) {
      eff = PersonaStatus::Probe;
      if (downgrade.empty()) downgrade = r;
    };

    std::set<std::string> seen_ids;
    int modifying_count = 0;

    for (std::size_t idx = 0; idx < pjson["interventions"].size(); ++idx) {
      const Json& ij = pjson["interventions"][idx];
      const std::string iwhere = pwhere + ".interventions[" + std::to_string(idx) + "]";
      if (!ij.is_object()) throw std::invalid_argument(iwhere + " must be an object");

      InterventionSpec sp;
      sp.id = req_str(ij, "id", iwhere);
      if (!seen_ids.insert(sp.id).second) {
        throw std::invalid_argument(iwhere + ": duplicate intervention id '" + sp.id + "'");
      }
      sp.type = parse_type(req_str(ij, "type", iwhere), iwhere);
      if (ij.contains("role") && ij["role"].is_string()) sp.role = ij["role"].get<std::string>();
      if (ij.contains("fit_method") && !ij["fit_method"].is_null()) {
        const std::string fm = req_str(ij, "fit_method", iwhere);
        if (fm != "closed_form" && fm != "analytic_projection") {
          throw std::invalid_argument(iwhere + ": forbidden fit_method '" + fm +
                                      "' (only closed_form|analytic_projection allowed; trained adapters are rejected)");
        }
        sp.fit_method = fm;
      }

      // Optional physical_block / loop_pass cross-check fields
      if (ij.contains("physical_block") && !ij["physical_block"].is_null()) {
        sp.physical_block = req_int(ij, "physical_block", 0, kLayers - 1, iwhere);
      }
      if (ij.contains("loop_pass") && !ij["loop_pass"].is_null()) {
        sp.loop_pass = req_int(ij, "loop_pass", 0, 2, iwhere);
      }

      // Parse type-specific parameters
      switch (sp.type) {
        case InterventionType::RouterAblate:
          sp.visit = req_int(ij, "visit", 0, kVisits - 1, iwhere);
          sp.expert = req_int(ij, "expert", 0, kExperts - 1, iwhere);
          break;
        case InterventionType::RouterForce:
          sp.visit = req_int(ij, "visit", 0, kVisits - 1, iwhere);
          sp.expert = req_int(ij, "expert", 0, kExperts - 1, iwhere);
          sp.bias = ij.contains("bias") ? float(req_num(ij, "bias", iwhere)) : 0.0f;
          sp.gain = ij.contains("gain") ? float(req_num(ij, "gain", iwhere)) : 1.0f;
          break;
        case InterventionType::RouterSlotPermute: {
          if (!ij.contains("visit")) throw std::invalid_argument(iwhere + ": missing 'visit'");
          if (ij["visit"].is_string() && ij["visit"] == "all") {
            sp.all_visits = true;
            sp.visit = -1;
          } else {
            sp.visit = req_int(ij, "visit", 0, kVisits - 1, iwhere);
          }
          if (!ij.contains("perm") || !ij["perm"].is_array() || ij["perm"].size() != kTopK) {
            throw std::invalid_argument(iwhere + ": 'perm' must be an array of 8 unique ints 0..7");
          }
          std::array<bool, kTopK> used{};
          for (int k = 0; k < kTopK; ++k) {
            if (!ij["perm"][k].is_number_integer()) throw std::invalid_argument(iwhere + ": perm entry must be int");
            const int dst = ij["perm"][k].get<int>();
            if (dst < 0 || dst >= kTopK || used[dst]) {
              throw std::invalid_argument(iwhere + ": perm must be a permutation of 0..7");
            }
            used[dst] = true;
            sp.perm[k] = std::int8_t(dst);
          }
          break;
        }
        case InterventionType::RouteReplay: {
          sp.visit = req_int(ij, "visit", 0, kVisits - 1, iwhere);
          const std::string mode = ij.contains("mode") ? req_str(ij, "mode", iwhere) : "recompute";
          if (mode == "recompute") sp.replay_copy_weights = false;
          else if (mode == "copy") sp.replay_copy_weights = true;
          else throw std::invalid_argument(iwhere + ": route_replay mode must be 'recompute' or 'copy'");
          sp.pos_start = ij.contains("pos_start") ? req_int(ij, "pos_start", 0, 1 << 24, iwhere) : 0;
          if (!ij.contains("donor") || !ij["donor"].is_object()) {
            throw std::invalid_argument(iwhere + ": route_replay requires 'donor' object (manifest_sha256, row_id, split='train', raw_order=true)");
          }
          const Json& d = ij["donor"];
          sp.donor_manifest_sha256 = req_str(d, "manifest_sha256", iwhere + ".donor");
          if (!is_sha256_hex(sp.donor_manifest_sha256)) {
            throw std::invalid_argument(iwhere + ".donor.manifest_sha256 must be 64 lowercase hex");
          }
          sp.donor_row_id = req_str(d, "row_id", iwhere + ".donor");
          sp.donor_split = req_str(d, "split", iwhere + ".donor");
          if (sp.donor_split != "train") {
            throw std::invalid_argument(iwhere + ".donor.split must be 'train' (sealed/heldout donors rejected)");
          }
          if (!d.contains("raw_order") || !d["raw_order"].is_boolean() || !d["raw_order"].get<bool>()) {
            throw std::invalid_argument(iwhere + ".donor.raw_order must be true");
          }
          sp.donor_raw_order = true;
          if (d.contains("donor_file") && d["donor_file"].is_string()) {
            sp.donor_file = d["donor_file"].get<std::string>();
            if (is_denied_sealed_path(sp.donor_file)) {
              throw std::invalid_argument(iwhere + ".donor.donor_file references sealed/heldout path");
            }
          }
          break;
        }
        case InterventionType::NeuronAffine:
          sp.visit = req_int(ij, "visit", 0, kVisits - 1, iwhere);
          sp.expert = req_int(ij, "expert", 0, kExperts - 1, iwhere);
          sp.neuron = req_int(ij, "neuron", 0, kInter - 1, iwhere);
          sp.gain = ij.contains("gain") ? float(req_num(ij, "gain", iwhere)) : 1.0f;
          sp.offset = ij.contains("offset") ? float(req_num(ij, "offset", iwhere)) : 0.0f;
          break;
        case InterventionType::DepthReplace: {
          const std::string site = req_str(ij, "site", iwhere);
          if (site == "attn_res") sp.depth_site = DepthSiteKind::AttnRes;
          else if (site == "ffn_res") sp.depth_site = DepthSiteKind::FfnRes;
          else if (site == "loop_msg") sp.depth_site = DepthSiteKind::LoopMsg;
          else if (site == "output_res") sp.depth_site = DepthSiteKind::OutputRes;
          else throw std::invalid_argument(iwhere + ": depth_replace site must be attn_res|ffn_res|loop_msg|output_res");
          sp.visit = (sp.depth_site == DepthSiteKind::OutputRes) ? -1 : req_int(ij, "visit", 0, kVisits - 1, iwhere);
          const std::string mode = req_str(ij, "mode", iwhere);
          if (mode == "read") sp.depth_mode = 1;
          else if (mode == "value") sp.depth_mode = 2;
          else if (mode == "both") sp.depth_mode = 3;
          else throw std::invalid_argument(iwhere + ": depth_replace mode must be read|value|both");
          sp.src_idx = ij.contains("src_idx") ? req_int(ij, "src_idx", -1, 5, iwhere) : -1;
          sp.pos_start = ij.contains("pos_start") ? req_int(ij, "pos_start", 0, 1 << 24, iwhere) : 0;
          if (!ij.contains("donor") || !ij["donor"].is_object()) {
            throw std::invalid_argument(iwhere + ": depth_replace requires 'donor' object");
          }
          const Json& d = ij["donor"];
          sp.donor_manifest_sha256 = req_str(d, "manifest_sha256", iwhere + ".donor");
          if (!is_sha256_hex(sp.donor_manifest_sha256)) {
            throw std::invalid_argument(iwhere + ".donor.manifest_sha256 must be 64 lowercase hex");
          }
          sp.donor_row_id = req_str(d, "row_id", iwhere + ".donor");
          sp.donor_split = req_str(d, "split", iwhere + ".donor");
          if (sp.donor_split != "train") {
            throw std::invalid_argument(iwhere + ".donor.split must be 'train'");
          }
          if (!d.contains("raw_order") || !d["raw_order"].is_boolean() || !d["raw_order"].get<bool>()) {
            throw std::invalid_argument(iwhere + ".donor.raw_order must be true");
          }
          sp.donor_raw_order = true;
          if (d.contains("donor_file") && d["donor_file"].is_string()) {
            sp.donor_file = d["donor_file"].get<std::string>();
            if (is_denied_sealed_path(sp.donor_file)) {
              throw std::invalid_argument(iwhere + ".donor.donor_file references sealed/heldout path");
            }
          }
          break;
        }
        case InterventionType::ResidualSteer: {
          sp.visit = req_int(ij, "visit", 0, kVisits - 1, iwhere);
          const std::string site = ij.contains("site") ? req_str(ij, "site", iwhere) : "mlp_write";
          if (site == "attn_write") sp.steer_site = SteerSiteKind::AttnWrite;
          else if (site == "mlp_write") sp.steer_site = SteerSiteKind::MlpWrite;
          else throw std::invalid_argument(iwhere + ": residual_steer site must be attn_write|mlp_write");
          sp.scale = ij.contains("scale") ? float(req_num(ij, "scale", iwhere)) : 1.0f;
          sp.vector.assign(kHidden, 0.0f);
          if (ij.contains("vector")) {
            if (!ij["vector"].is_array() || int(ij["vector"].size()) != kHidden) {
              throw std::invalid_argument(iwhere + ": residual_steer 'vector' must have length 2304");
            }
            for (int c = 0; c < kHidden; ++c) {
              const double v = ij["vector"][c].get<double>();
              if (!std::isfinite(v)) throw std::invalid_argument(iwhere + ": non-finite value in vector");
              sp.vector[std::size_t(c)] = float(v);
            }
          } else if (ij.contains("sparse_vector")) {
            if (!ij["sparse_vector"].is_object()) {
              throw std::invalid_argument(iwhere + ": 'sparse_vector' must be an object");
            }
            for (auto sit = ij["sparse_vector"].begin(); sit != ij["sparse_vector"].end(); ++sit) {
              const int c = std::stoi(sit.key());
              if (c < 0 || c >= kHidden) throw std::invalid_argument(iwhere + ": sparse_vector index out of 0..2303");
              if (c == kForbiddenCoord) {
                throw std::invalid_argument(iwhere + ": sparse_vector forbids coordinate 2027");
              }
              const double v = sit.value().get<double>();
              if (!std::isfinite(v)) throw std::invalid_argument(iwhere + ": non-finite value in sparse_vector");
              sp.vector[std::size_t(c)] = float(v);
            }
          } else {
            throw std::invalid_argument(iwhere + ": residual_steer requires 'vector' or 'sparse_vector'");
          }
          break;
        }
        case InterventionType::TapExpert:
          sp.visit = req_int(ij, "visit", 0, kVisits - 1, iwhere);
          sp.expert = req_int(ij, "expert", 0, kExperts - 1, iwhere);
          sp.per_token = ij.contains("per_token") && ij["per_token"].is_boolean() && ij["per_token"].get<bool>();
          break;
        case InterventionType::TapRouting:
          if (ij.contains("visit") && ij["visit"].is_string() && ij["visit"] == "all") {
            sp.all_visits = true;
            sp.visit = -1;
          } else {
            sp.visit = req_int(ij, "visit", 0, kVisits - 1, iwhere);
          }
          sp.capture_logits = ij.contains("logits") && ij["logits"].is_boolean() && ij["logits"].get<bool>();
          break;
        case InterventionType::RouterLowRankFit:
        case InterventionType::ExpertDownLowRankFit: {
          sp.physical_block = req_int(ij, "physical_block", 0, kLayers - 1, iwhere);
          sp.rank = req_int(ij, "rank", 1, 1024, iwhere);
          sp.fit_method = req_str(ij, "fit_method", iwhere);
          if (sp.fit_method != "closed_form" && sp.fit_method != "analytic_projection") {
            throw std::invalid_argument(iwhere + ": forbidden fit_method '" + sp.fit_method +
                                        "' (only closed_form|analytic_projection allowed; trained adapters are rejected)");
          }
          sp.weights_file = req_str(ij, "weights_file", iwhere);
          sp.weights_sha256 = req_str(ij, "weights_sha256", iwhere);
          if (!is_sha256_hex(sp.weights_sha256)) {
            throw std::invalid_argument(iwhere + ": weights_sha256 must be 64 lowercase hex chars");
          }
          const std::filesystem::path wp =
              std::filesystem::path(sp.weights_file).is_absolute() || base_dir.empty()
                  ? std::filesystem::path(sp.weights_file)
                  : base_dir / sp.weights_file;
          const std::string actual_sha = sha256_file_hex(wp);
          if (actual_sha != sp.weights_sha256) {
            throw std::invalid_argument(iwhere + ": weights_file sha256 mismatch (expected " + sp.weights_sha256 +
                                        ", got " + actual_sha + ")");
          }
          if (ij.contains("experts") && ij["experts"].is_array()) {
            for (const auto& ex : ij["experts"]) sp.experts.push_back(ex.get<int>());
          }
          break;
        }
      }

      cross_check_visit(sp, iwhere);

      // Provenance evaluation for every model-modifying entry
      if (!is_read_only_tap(sp.type)) {
        ++modifying_count;
        if (!ij.contains("provenance") || ij["provenance"].is_null()) {
          sp.has_provenance = false;
          sp.effective_status = PersonaStatus::Probe;
          sp.downgrade_reason = "missing provenance";
          note_downgrade(iwhere + ": missing provenance");
        } else {
          const Json& pv = ij["provenance"];
          if (!pv.is_object()) throw std::invalid_argument(iwhere + ".provenance must be an object");
          sp.has_provenance = true;
          const PersonaStatus decl = parse_status(req_str(pv, "status", iwhere + ".provenance"), iwhere + ".provenance");
          sp.effective_status = decl;

          // Constant pushes are probes, never product (PERSONA-OPUS.md §1).
          if (is_constant_probe(sp.type) && decl == PersonaStatus::Product) {
            throw std::invalid_argument(iwhere + ": constant push '" + req_str(ij, "type", iwhere) +
                                        "' cannot carry status 'product' (probes are never a product persona)");
          }

          if (decl != PersonaStatus::Probe) {
            std::string why;
            if (!pv.contains("purpose") || !pv["purpose"].is_string() || pv["purpose"].get_ref<const std::string&>().empty() ||
                !pv.contains("experiment") || !pv["experiment"].is_string() || pv["experiment"].get_ref<const std::string&>().empty() ||
                !pv.contains("evidence_file") || !pv["evidence_file"].is_string() ||
                pv["evidence_file"].get_ref<const std::string&>().empty()) {
              sp.effective_status = PersonaStatus::Probe;
              sp.downgrade_reason = "incomplete provenance metadata";
            } else if (!pv.contains("evidence_split") || pv["evidence_split"] != "confirmation") {
              sp.effective_status = PersonaStatus::Probe;
              sp.downgrade_reason = "evidence_split is not 'confirmation'";
            } else if (!pv.contains("reaches") || !pv["reaches"].is_string() ||
                       pv["reaches"].get_ref<const std::string&>().empty()) {
              sp.effective_status = PersonaStatus::Probe;
              sp.downgrade_reason = "missing provenance.reaches";
            } else if (!pv.contains("origin") || !pv["origin"].is_object() ||
                       !pv["origin"].contains("character") || !pv["origin"]["character"].is_string() ||
                       pv["origin"]["character"].get_ref<const std::string&>().empty() ||
                       !pv["origin"].contains("source_text") || !pv["origin"]["source_text"].is_string() ||
                       pv["origin"]["source_text"].get_ref<const std::string&>().empty()) {
              sp.effective_status = PersonaStatus::Probe;
              sp.downgrade_reason = "missing provenance.origin (character, source_text)";
            } else if (!pv.contains("noop_check") || !validate_noop_check(pv["noop_check"], true, why)) {
              sp.effective_status = PersonaStatus::Probe;
              sp.downgrade_reason = why.empty() ? "missing noop_check" : why;
            } else if (!pv.contains("measured_effect") || !validate_measured_effect(pv["measured_effect"], why)) {
              sp.effective_status = PersonaStatus::Probe;
              sp.downgrade_reason = why.empty() ? "missing measured_effect" : why;
            } else if (!pv.contains("refusal_checked") || !validate_refusal_check(pv["refusal_checked"], why)) {
              sp.effective_status = PersonaStatus::Probe;
              sp.downgrade_reason = why.empty() ? "missing refusal_checked" : why;
            }
            if (is_denied_sealed_path(pv.value("evidence_file", std::string()))) {
              throw std::invalid_argument(iwhere + ": evidence_file references sealed/heldout path");
            }
          }
          if (sp.effective_status == PersonaStatus::Probe && !sp.downgrade_reason.empty()) {
            note_downgrade(iwhere + ": " + sp.downgrade_reason);
          }
          eff = min_status(eff, sp.effective_status);
        }
      }

      prof->interventions.push_back(std::move(sp));
    }

    // Profile-level gates for validated / product
    if (modifying_count > 0 && eff != PersonaStatus::Probe) {
      std::string why;
      // 1) Model binding check (weights, quantisation, commit, prompt_renderer, chat_template_sha256)
      if (!pjson.contains("model_binding") || !pjson["model_binding"].is_object()) {
        note_downgrade(pwhere + ": missing model_binding");
      } else {
        const Json& mb = pjson["model_binding"];
        const std::string wsha = mb.value("weights_sha256", std::string());
        const std::string quant = mb.value("quantisation", std::string());
        const std::string commit = mb.value("engine_commit", std::string());
        const std::string renderer = mb.value("prompt_renderer", std::string());
        const std::string tsha = mb.value("chat_template_sha256", std::string());
        if (!is_sha256_hex(wsha) || quant.empty() || commit.size() < 7 || renderer.empty() || !is_sha256_hex(tsha)) {
          note_downgrade(pwhere + ": invalid model_binding fields (requires weights_sha256, quantisation, engine_commit, prompt_renderer, chat_template_sha256)");
        } else if (runtime_id.populated()) {
          if (!runtime_id.weights_sha256.empty() && wsha != runtime_id.weights_sha256) {
            note_downgrade(pwhere + ": model_binding.weights_sha256 mismatch against live runtime");
          } else if (!runtime_id.quantisation.empty() && quant != runtime_id.quantisation) {
            note_downgrade(pwhere + ": model_binding.quantisation mismatch against live runtime");
          } else if (!runtime_id.engine_commit.empty() && commit != runtime_id.engine_commit) {
            note_downgrade(pwhere + ": model_binding.engine_commit mismatch against live runtime");
          } else if (!runtime_id.prompt_renderer.empty() && renderer != runtime_id.prompt_renderer) {
            note_downgrade(pwhere + ": model_binding.prompt_renderer mismatch ('" + renderer + "' vs live '" +
                           runtime_id.prompt_renderer + "')");
          } else if (!runtime_id.chat_template_sha256.empty() && tsha != runtime_id.chat_template_sha256) {
            note_downgrade(pwhere + ": model_binding.chat_template_sha256 mismatch against live chat_template.jinja");
          }
        }
      }
      // 2) Leakage check required for validated / product
      if (eff != PersonaStatus::Probe) {
        if (!pjson.contains("leakage_check") || !pjson["leakage_check"].is_object() ||
            !pjson["leakage_check"].value("passed", false) ||
            pjson["leakage_check"].value("evidence_file", std::string()).empty()) {
          note_downgrade(pwhere + ": missing passed leakage_check");
        }
      }
      // 3) Multi-intervention composition evidence
      if (eff != PersonaStatus::Probe && modifying_count > 1) {
        if (!pjson.contains("composition_evidence") || !pjson["composition_evidence"].is_object()) {
          note_downgrade(pwhere + ": multi-intervention profile requires composition_evidence");
        } else {
          const Json& ce = pjson["composition_evidence"];
          if (ce.value("evidence_split", std::string()) != "confirmation" ||
              !ce.contains("noop_check") || !validate_noop_check(ce["noop_check"], true, why) ||
              !ce.contains("measured_effect") || !validate_measured_effect(ce["measured_effect"], why)) {
            note_downgrade(pwhere + ": invalid composition_evidence (" + why + ")");
          }
        }
      }
      // 4) Product-only gates: owner_grade, reasoning_check, refusal_checked
      if (eff == PersonaStatus::Product) {
        if (!pjson.contains("owner_grade") || !pjson["owner_grade"].is_object() ||
            !pjson["owner_grade"].value("passed", false) ||
            pjson["owner_grade"].value("evidence_file", std::string()).empty()) {
          note_downgrade(pwhere + ": product requires passed owner_grade");
        } else if (!pjson.contains("reasoning_check") || !pjson["reasoning_check"].is_object() ||
                   !pjson["reasoning_check"].value("passed", false) ||
                   pjson["reasoning_check"].value("evidence_file", std::string()).empty() ||
                   !pjson["reasoning_check"].contains("plain_value") ||
                   !pjson["reasoning_check"].contains("profile_value")) {
          note_downgrade(pwhere + ": product requires passed reasoning_check");
        } else if (!pjson.contains("refusal_checked") || !validate_refusal_check(pjson["refusal_checked"], why)) {
          note_downgrade(pwhere + ": product requires passed refusal_checked (" + why + ")");
        }
      }
    }

    prof->effective_status = (modifying_count == 0) ? PersonaStatus::Product : eff;
    prof->downgrade_reason = downgrade;

    // Unconditional safety checks (d_2027, refusal_map, duplicate 3b neurons) even in probe mode!
    validate_profile_safety(*prof, refusal_map);

    // Strict mode enforcement: profiles with effective_status == Probe (or declared category > effective_status
    // when outside probe_mode) are rejected outside --persona-probe-mode.
    if (!probe_mode && modifying_count > 0) {
      if (prof->effective_status == PersonaStatus::Probe) {
        throw std::invalid_argument(pwhere + ": rejected outside --persona-probe-mode (effective status is 'probe'" +
                                    (prof->downgrade_reason.empty() ? "" : ": " + prof->downgrade_reason) + ")");
      }
      if (static_cast<int>(prof->declared_category) > static_cast<int>(prof->effective_status)) {
        throw std::invalid_argument(pwhere + ": declared category '" + to_string(prof->declared_category) +
                                    "' exceeds effective status '" + to_string(prof->effective_status) + "'");
      }
    }

    cat.profiles.emplace(pname, std::move(prof));
  }

  // Enforce default_profile rule: MUST be "none" unless it names a "product" profile (outside probe mode).
  if (cat.default_profile != "none") {
    auto it = cat.profiles.find(cat.default_profile);
    if (it == cat.profiles.end()) {
      throw std::invalid_argument("default_profile '" + cat.default_profile + "' not found in profiles");
    }
    if (!probe_mode && it->second->effective_status != PersonaStatus::Product) {
      throw std::invalid_argument("default_profile '" + cat.default_profile + "' has status '" +
                                  to_string(it->second->effective_status) +
                                  "'; default_profile must be 'none' unless it names a 'product' profile (or run with --persona-probe-mode)");
    }
  }

  return cat;
}

}  // namespace ingine::persona

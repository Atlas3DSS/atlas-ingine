// CUDA-free public persona & mech-interp specification ABI, loader, provenance validator, and refusal guard.
// Implements schemas/persona-spec.schema.json (atlas-persona-spec/v1).
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ingine::persona {

inline constexpr int kVisits = 48;
inline constexpr int kLayers = 24;
inline constexpr int kExperts = 128;
inline constexpr int kTopK = 8;
inline constexpr int kHidden = 2304;
inline constexpr int kInter = 1152;
inline constexpr int kForbiddenCoord = 2027;  // standing invariant: hidden coordinate 2027 is never perturbed

enum class PersonaStatus : int {
  Probe = 0,
  Validated = 1,
  Product = 2,
};

inline const char* to_string(PersonaStatus s) {
  switch (s) {
    case PersonaStatus::Product: return "product";
    case PersonaStatus::Validated: return "validated";
    default: return "probe";
  }
}

enum class InterventionType : int {
  RouterAblate = 0,
  RouterForce,
  RouterSlotPermute,
  RouteReplay,
  NeuronAffine,
  DepthReplace,
  ResidualSteer,
  TapExpert,
  TapRouting,
  RouterLowRankFit,
  ExpertDownLowRankFit,
};

inline bool is_read_only_tap(InterventionType t) {
  return t == InterventionType::TapExpert || t == InterventionType::TapRouting;
}

inline bool is_constant_probe(InterventionType t) {
  switch (t) {
    case InterventionType::RouterAblate:
    case InterventionType::RouterForce:
    case InterventionType::RouterSlotPermute:
    case InterventionType::RouteReplay:
    case InterventionType::NeuronAffine:
    case InterventionType::DepthReplace:
    case InterventionType::ResidualSteer: return true;
    default: return false;
  }
}

enum class DepthSiteKind : int {
  AttnRes = 0,
  FfnRes = 1,
  LoopMsg = 2,
  OutputRes = 3,
};

enum class SteerSiteKind : int {
  AttnWrite = 0,
  MlpWrite = 1,
};

// Chronological visit -> (physical_block, loop_pass) table matching model_data.h VISITS.
struct VisitCoord {
  int physical_block;
  int loop_pass;
};
VisitCoord visit_coord(int visit);

// Refusal-linked unit deny-list (loaded via --refusal-map; coordinate 2027 always protected).
struct RefusalUnit {
  int visit = -1;   // -1 = all visits of physical_block, else 0..47
  int physical_block = -1;
  int expert = -1;  // -1 = all experts at visit, else 0..127
  int neuron = -1;  // -1 = whole expert unit, else 0..1151
};

struct RefusalMap {
  std::vector<RefusalUnit> protected_units;
  std::vector<int> protected_coords;

  static RefusalMap default_map();
  static RefusalMap load(const std::filesystem::path& path);
};

// Live runtime identity for checking profile model_binding (mismatches downgrade profile to probe).
struct ModelRuntimeIdentity {
  std::string weights_sha256;
  std::string quantisation;
  std::string engine_commit;
  std::string prompt_renderer;
  std::string chat_template_sha256;
  bool populated() const {
    return !weights_sha256.empty() || !quantisation.empty() || !engine_commit.empty() || !prompt_renderer.empty() ||
           !chat_template_sha256.empty();
  }
};

struct InterventionSpec {
  std::string id;
  InterventionType type = InterventionType::RouterAblate;
  std::string role;        // "probe" | "tap" | "measured_fit"
  std::string fit_method;  // "closed_form" | "analytic_projection"

  // Address: visit in 0..47, or -1 for all visits (slot_permute) or output_res (depth_replace).
  int visit = -1;
  bool all_visits = false;
  std::optional<int> physical_block;
  std::optional<int> loop_pass;

  // Router ablate / force / neuron affine / tap_expert parameters
  int expert = -1;
  int neuron = -1;
  float bias = 0.0f;
  float gain = 1.0f;
  float offset = 0.0f;

  // Slot permute
  std::array<std::int8_t, kTopK> perm = {0, 1, 2, 3, 4, 5, 6, 7};

  // Route replay / depth replace
  DepthSiteKind depth_site = DepthSiteKind::FfnRes;
  std::uint32_t depth_mode = 0;  // 1=read, 2=value, 3=both
  int src_idx = -1;              // -1 = all sources
  int pos_start = 0;
  bool replay_copy_weights = false;
  std::string donor_manifest_sha256;
  std::string donor_row_id;
  std::string donor_split;
  bool donor_raw_order = false;
  std::string donor_file;

  // Residual steering (probe only; vec[2027] == 0.0f strictly enforced)
  SteerSiteKind steer_site = SteerSiteKind::MlpWrite;
  float scale = 1.0f;
  std::vector<float> vector;  // length 0 or kHidden

  // Read-only tap options
  bool per_token = false;
  bool capture_logits = false;

  // Low-rank measured fit metadata (Rungs 5-6)
  int rank = 0;
  std::vector<int> experts;
  std::string weights_file;
  std::string weights_sha256;

  // Provenance status evaluated at load time
  bool has_provenance = false;
  PersonaStatus effective_status = PersonaStatus::Probe;
  std::string downgrade_reason;

  // Programmatic in-memory donor arrays (used by offline probe / self-replay / self-depth-replace verification;
  // never loaded from inline sealed text in JSON specs).
  std::vector<std::int32_t> inline_replay_ids;  // [n_pos * 8]
  std::vector<float> inline_replay_wts;         // [n_pos * 8] (empty for recompute mode)
  std::vector<float> inline_depth_scores;       // [n_pos * n_q * n_src]
  std::vector<float> inline_depth_states;       // [n_pos * n_src * kHidden]
  int inline_depth_nsrc = 0;
  int inline_depth_nq = 0;
};

struct PersonaProfile {
  std::string name;
  std::string description;
  PersonaStatus declared_category = PersonaStatus::Probe;
  PersonaStatus effective_status = PersonaStatus::Probe;
  std::string downgrade_reason;
  std::string fingerprint_sha256;  // 64 lowercase hex chars of canonicalised profile JSON
  std::uint64_t cache_key = 0;     // non-zero hash for cache isolation (0 reserved for unhooked baseline)
  std::string spec_sha256;         // raw spec file SHA-256 (for response metadata)
  std::vector<InterventionSpec> interventions;

  // True when this profile represents the unhooked baseline fast path.
  bool is_unhooked() const { return interventions.empty(); }
  // True when any intervention modifies the model (false if empty or read-only taps only).
  bool modifies_model() const;
  // True when position-addressed hooks or taps require bypassing prefix cache reuse.
  bool bypasses_prefix_reuse() const;
};

struct PersonaResponseMeta {
  std::string profile;
  std::string persona_fingerprint;
  std::string status;
  std::string spec_sha256;
};

struct PersonaCatalog {
  std::string spec_sha256;
  std::string created;
  std::string author;
  std::string spec_id;
  std::string default_profile = "none";
  bool probe_mode = false;
  std::map<std::string, std::shared_ptr<const PersonaProfile>> profiles;

  bool empty() const { return profiles.empty(); }

  // Resolves a requested profile name (empty => default_profile; "none" => nullptr for unhooked fast path).
  // Throws std::invalid_argument on unknown profile name.
  std::shared_ptr<const PersonaProfile> resolve(const std::string& requested) const;

  static PersonaCatalog load(const std::filesystem::path& path, bool probe_mode,
                             const RefusalMap& refusal_map = RefusalMap::default_map(),
                             const ModelRuntimeIdentity& runtime_id = {});

  static PersonaCatalog parse_json(const std::string& raw_json, bool probe_mode,
                                   const RefusalMap& refusal_map = RefusalMap::default_map(),
                                   const ModelRuntimeIdentity& runtime_id = {},
                                   const std::filesystem::path& base_dir = {});
};

// Pure C++20 SHA-256 hex digest (64 lowercase hex chars).
std::string sha256_hex(const void* data, std::size_t bytes);
std::string sha256_hex(const std::string& s);
std::string sha256_file_hex(const std::filesystem::path& path);

// Returns non-zero 64-bit key derived from a 64-char hex digest (0 reserved for unhooked baseline).
std::uint64_t cache_key_from_hex(const std::string& sha256_hex_str);

// Validates a programmatic PersonaProfile against a RefusalMap and d_2027 invariant (throws std::invalid_argument).
void validate_profile_safety(const PersonaProfile& profile, const RefusalMap& refusal_map);

}  // namespace ingine::persona

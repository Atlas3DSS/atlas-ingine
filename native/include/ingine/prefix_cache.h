// Shared CUDA-free prefix-checkpoint contract for B=1 EngineGenerator (#826) and B>1 BatchEngine (#825).
//
// Invariants (fail-closed):
//   * Hashes (hash_state_identity, hash_token_prefix) are index accelerators ONLY; every cache hit MUST
//     verify exact StateIdentity::operator== and exact token-ID prefix equality.
//   * root_prefix_tokens is strictly a snapshot-boundary hint (never a trust key); validate_root_prefix_tokens
//     ensures [0, ids.size()] range and that no PromptImage straddles the boundary.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace ingine::model {

struct PromptImage;

// Complete runtime/model/isolation identity for prefix-cache compatibility. Any mismatch MUST fail closed to miss.
struct StateIdentity {
  std::array<std::uint8_t, 32> model_sha256{};  // config.json + optional ablation file digest
  std::uint32_t act_dtype = 0;                  // 0 = BF16, 1 = FP16
  bool dense_fp8 = true;
  std::uint32_t kv_format = 0;                  // 0 = ACT (BF16/FP16), 1 = FP8, 2 = Q4_0
  std::uint32_t kv_cache_mode = 0;              // 0 = per_pass (2+2*L full slots), 1 = last_full (4 full slots, L=3)
  int loop_passes = 3;                          // recurrent loop passes (1..8, default 3)
  int max_ctx = 0;
  int prefill_chunk = 0;
  bool spec_enabled = false;
  std::uint64_t persona_key = 0;                // 0 = unhooked baseline; PersonaProfile::cache_key when hooked
  std::string persona_fingerprint;              // SHA-256 hex digest when hooked (empty when unhooked)
  std::string cache_namespace;                  // optional tenant/branch isolation namespace

  bool operator==(const StateIdentity&) const = default;
};

namespace detail {
inline std::uint64_t fnv1a_mix(std::uint64_t h, const void* data, std::size_t bytes) noexcept {
  const auto* p = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < bytes; ++i) {
    h ^= std::uint64_t(p[i]);
    h *= 1099511628211ull;
  }
  return h;
}
}  // namespace detail

// Index hash ONLY (never used for equality without full StateIdentity::operator==).
inline std::uint64_t hash_state_identity(const StateIdentity& id) noexcept {
  std::uint64_t h = 14695981039346656037ull;
  h = detail::fnv1a_mix(h, id.model_sha256.data(), id.model_sha256.size());
  h = detail::fnv1a_mix(h, &id.act_dtype, sizeof(id.act_dtype));
  h = detail::fnv1a_mix(h, &id.dense_fp8, sizeof(id.dense_fp8));
  h = detail::fnv1a_mix(h, &id.kv_format, sizeof(id.kv_format));
  h = detail::fnv1a_mix(h, &id.kv_cache_mode, sizeof(id.kv_cache_mode));
  h = detail::fnv1a_mix(h, &id.loop_passes, sizeof(id.loop_passes));
  h = detail::fnv1a_mix(h, &id.max_ctx, sizeof(id.max_ctx));
  h = detail::fnv1a_mix(h, &id.prefill_chunk, sizeof(id.prefill_chunk));
  h = detail::fnv1a_mix(h, &id.spec_enabled, sizeof(id.spec_enabled));
  h = detail::fnv1a_mix(h, &id.persona_key, sizeof(id.persona_key));
  const std::size_t pf_len = id.persona_fingerprint.size();
  h = detail::fnv1a_mix(h, &pf_len, sizeof(pf_len));
  if (pf_len > 0) h = detail::fnv1a_mix(h, id.persona_fingerprint.data(), pf_len);
  const std::size_t ns_len = id.cache_namespace.size();
  h = detail::fnv1a_mix(h, &ns_len, sizeof(ns_len));
  if (ns_len > 0) h = detail::fnv1a_mix(h, id.cache_namespace.data(), ns_len);
  return h;
}

// Index hash ONLY over an exact token-ID prefix (never used for equality without exact std::equal).
inline std::uint64_t hash_token_prefix(std::span<const std::int32_t> ids) noexcept {
  std::uint64_t h = 14695981039346656037ull;
  const std::size_t n = ids.size();
  h = detail::fnv1a_mix(h, &n, sizeof(n));
  if (!ids.empty()) h = detail::fnv1a_mix(h, ids.data(), ids.size_bytes());
  return h;
}

// Exact runtime accounting for prefix-cache / conversation-parking memory, transfers, and evictions.
struct PrefixCacheStats {
  std::size_t vram_budget_bytes = 0;
  std::size_t vram_used_bytes = 0;
  std::size_t vram_peak_bytes = 0;
  std::size_t device_staging_bytes = 0;  // fixed pre-allocated checkpoint staging buffer in VRAM
  std::size_t host_budget_bytes = 0;
  std::size_t host_used_bytes = 0;
  std::size_t host_peak_bytes = 0;
  std::uint64_t vram_entries = 0;
  std::uint64_t host_entries = 0;
  std::uint64_t root_entries = 0;
  std::uint64_t vram_evictions = 0;
  std::uint64_t host_evictions = 0;
  std::uint64_t host_demotions = 0;       // Tier-0 VRAM -> Tier-1 pinned host RAM demotions
  std::uint64_t host_alloc_failures = 0;  // graceful fallbacks under host memory pressure
  std::uint64_t vram_pressure_releases = 0;
  std::uint64_t h2d_restore_bytes = 0;
  std::uint64_t d2h_park_bytes = 0;
  std::uint64_t d2d_restore_bytes = 0;
  std::uint64_t d2d_park_bytes = 0;
};

}  // namespace ingine::model

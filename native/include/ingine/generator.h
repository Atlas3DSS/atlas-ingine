// The token generator interface the server drives (mirror of engine.Engine.generate). CUDA-free so the
// scheduler, the HTTP layer and their tests build without the model; ingine_model implements make_generator.
#pragma once

#include "ingine/persona.h"
#include "ingine/prefix_cache.h"
#include "ingine/token_masker.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ingine::model {

// Full-attention KV cache sharing mode across loop passes (layers 11 and 17).
// PerPass (default): all 8 full-attention visits (L5, L11x3, L17x3, L23) have independent KV slots.
// LastFull (opt-in): L11 passes 0..2 share one slot and L17 passes 0..2 share one slot (4 full slots total);
// prior tokens use the final pass's persisted KV while sliding rings (40) and conv states (48) stay pass-local.
enum class KvCacheMode : std::uint8_t {
  PerPass = 0,
  LastFull = 1,
};

inline const char* to_string(KvCacheMode mode) {
  switch (mode) {
    case KvCacheMode::LastFull: return "last_full";
    default: return "per_pass";
  }
}

inline KvCacheMode parse_kv_cache_mode(std::string_view s) {
  if (s == "per_pass") return KvCacheMode::PerPass;
  if (s == "last_full") return KvCacheMode::LastFull;
  throw std::invalid_argument("invalid kv_cache_mode '" + std::string(s) + "' (expected 'per_pass' or 'last_full')");
}

struct Sampling {
  float temperature = 1.0f;  // 0 => greedy
  float top_p = 1.0f;
  int top_k = 0;  // 0 => disabled
  std::optional<uint64_t> seed;
  // Token ids masked to -inf before argmax/sampling (e.g. <|content_invoke_tool_json|> when no tools are provided).
  std::vector<int32_t> banned_tokens;
  // Teacher-forced replay (test/benchmark only; empty = off): generate() emits exactly these tokens
  // (max_new_tokens is capped at their count), running its real loop along them: plain decoding samples as
  // usual and then takes the forced token; speculative decoding accepts a draft iff it equals the forced token
  // and continues with the forced token after the accepted run. Fills GenerateStats::spec_trace.
  std::span<const int32_t> forced;
  // Optional resolved persona/mech-interp profile for this request (nullptr => unhooked fast path).
  std::shared_ptr<const persona::PersonaProfile> persona;
  // Optional token-level constrained decoding masker (e.g. JSON / JSON-Schema FSM; nullptr => zero overhead).
  std::shared_ptr<TokenMasker> logit_masker;
};

// The token whose embedding an image token replaces (<|unused_200054|>, the chat template's image placeholder).
inline constexpr std::int32_t IMAGE_PLACEHOLDER_ID = 200054;

// One image of a prompt. The chat template marks an image with one placeholder token; the prompt holds
// tokens() ids in its place, one per 40x40-pixel tile in raster order, whose embeddings the vision tower
// computes from the pixels. These ids are negative (never a vocabulary id) and a function of the image
// content and the tile index (image::token_id), so prompts compare equal exactly when their images do.
struct PromptImage {
  int pos = 0;                                           // prompt position of the image's first token
  int height = 0, width = 0;                             // pixels
  std::shared_ptr<const std::vector<std::uint8_t>> rgb;  // [height, width, 3]
  [[nodiscard]] int rows() const { return (height + 39) / 40; }
  [[nodiscard]] int cols() const { return (width + 39) / 40; }
  [[nodiscard]] int tokens() const { return rows() * cols(); }
};

// A prompt: token ids, images in position order (without overlap), optional verified root-prefix snapshot
// boundary hint (0 = none; never a trust key), optional cache isolation namespace, and per-request bypass.
struct Prompt {
  std::span<const std::int32_t> ids;
  std::span<const PromptImage> images;
  int root_prefix_tokens = 0;
  std::string cache_namespace;
  bool bypass_prefix_cache = false;
};

// Validates that `root_prefix_tokens` lies inside [0, ids.size()] and does not straddle any PromptImage.
// Returns the validated boundary (or 0 if invalid/out-of-range).
inline int validate_root_prefix_tokens(std::span<const std::int32_t> ids, std::span<const PromptImage> images,
                                       int root_prefix_tokens) noexcept {
  if (root_prefix_tokens <= 0 || std::size_t(root_prefix_tokens) > ids.size()) return 0;
  for (const PromptImage& img : images) {
    if (img.pos < root_prefix_tokens && img.pos + img.tokens() > root_prefix_tokens) return 0;
  }
  return root_prefix_tokens;
}

// How the prompt reused the state of earlier requests (reused_tokens were not prefilled again).
enum class Reuse {
  None,            // empty state, the whole prompt prefilled
  Extend,          // the prompt extends the tokens in the state (previous prompt + processed generated tokens)
  Checkpoint,      // restored the checkpoint taken after the previous prompt's prefill (a prefix of this prompt)
  Parked,          // restored a parked conversation's checkpoint in VRAM (a prefix of this prompt; RuntimeOptions::park_mb)
  RootCheckpoint,  // restored a verified root-prefix (system+tools) checkpoint
  HostParked,      // restored a parked conversation's checkpoint from pinned host RAM
  PagedPrefix,     // restored a shared paged full-KV + recurrent prefix checkpoint (batched B>1)
};
inline const char* to_string(Reuse reuse) {
  switch (reuse) {
    case Reuse::Extend: return "extend";
    case Reuse::Checkpoint: return "checkpoint";
    case Reuse::Parked: return "parked";
    case Reuse::RootCheckpoint: return "root_checkpoint";
    case Reuse::HostParked: return "host_parked";
    case Reuse::PagedPrefix: return "paged_prefix";
    default: return "none";
  }
}

// One speculative round of a forced replay: verify width, accepted drafts, the drafts that would have been
// accepted had all 7 been verified, and GPU time of the round's verify (+ acceptance) and of its draft.
struct SpecRoundRecord {
  int width = 0, accepted = 0, accepted_all = 0;
  float verify_ms = 0, draft_ms = 0;
};

struct GenerateStats {
  int prompt_tokens = 0, reused_tokens = 0, generated_tokens = 0;
  Reuse reuse = Reuse::None;
  // Actual bytes copied to restore reused model state, restore time, and cache evictions during this request.
  // std::nullopt means the backend/path did not measure the metric (never reported as a synthetic zero).
  std::optional<std::uint64_t> restored_bytes;
  std::optional<double> restore_seconds;
  std::optional<int> cache_evictions;
  double prefill_seconds = 0, decode_seconds = 0;
  int spec_rounds = 0, spec_accepted = 0;
  std::vector<SpecRoundRecord> spec_trace;  // forced replay only
  // Images: tokens of the prompt's images, how many images the vision tower encoded for this request (those
  // not inside the reused prefix) and its time (part of prefill_seconds).
  int image_tokens = 0, images_encoded = 0;
  double vision_seconds = 0;
};

using PrefillCallback = std::function<void(const GenerateStats&)>;

struct RuntimeOptions {
  std::filesystem::path model_dir, ablation;  // empty ablation: original model
  int max_ctx = 65536;
  // Tokens per prefill chunk. 3072 (the largest the sliding attention takes) reads a long prompt 6-10% faster
  // than 2048 for 0.38 GB more scratch; results differ between chunk sizes only by rounding.
  int prefill_chunk = 3072;
  bool dense_fp8 = true, kv_fp8 = false, kv_q4 = false, spec = false;
  // FP4 (E2M1) full-attention values with FP8 keys in the TP batch pool (needs kv_fp8): 0.75 of the FP8 pool.
  bool kv_fp4v = false;
  KvCacheMode kv_cache_mode = KvCacheMode::PerPass;
  float spec_threshold = 0.8f;
  // Batched DSpark drafter options (used when spec && max_batch > 1):
  //   draft_ctx:      global-layer E4M3 ring slots per concurrent stream (default 16384; 0 = full 65536).
  //   spec_max_m:     maximum verify width per slot in a batched step (2..4, default 4).
  //   spec_shortlist: ranked shortlist rows in the shared drafter head (default 32768).
  int draft_ctx = 16384;
  int spec_max_m = 4;
  int spec_shortlist = 32768;
  // Conversation parking: device memory (MiB) for the checkpoints of conversations displaced by another
  // one, least recently used evicted, all released when a device allocation fails; 0 disables; negative:
  // min(3072, free memory after start-up - 4096), at least 0.
  int park_mb = -1;
  // Bounded pinned host-RAM parking tier (MiB) for checkpoints evicted from VRAM or parked when park_mb=0
  // (LRU eviction, graceful fallback under host memory pressure; default 0 = opt-in).
  int park_host_mb = 0;
  // Symmetric prefix-cache toggle for both B=1 and B>1 (false = cache-off parity baseline, zero reuse).
  bool prefix_cache = true;
  // Image input: load the vision tower (636 MB of the checkpoint, resident) and accept prompts with images.
  bool vision = false;
  // Dynamic 3-tier MoE expert offloading (atlas-6g-homology):
  //   offload_vram_mb == 0: disabled (all experts resident in VRAM).
  //   offload_vram_mb > 0:  target VRAM cap in MiB (e.g. 6144 for 6 GB GPUs); non-expert weights, KV cache,
  //                         and static homology-pinned HOT experts + dynamic LRU pool stay inside this cap.
  //   offload_ram_mb:       pinned host RAM cap in MiB for WARM experts (default 12288 inside 16 GB RAM);
  //                         any remaining COLD experts stream on demand from read-only NVMe mmap.
  //   offload_mode:         "auto" (dma-lru), "dma-lru" (3-tier async PCIe DMA + dynamic VRAM LRU), or
  //                         "uva" (2-tier HOT VRAM + zero-copy UVA pinned RAM, CUDA-graph capturable).
  int offload_vram_mb = 0;
  int offload_ram_mb = 12288;
  std::string offload_mode = "auto";
  std::filesystem::path expert_ranking;     // optional JSON per-layer [24][128] homology/frequency scores
  std::filesystem::path dump_router_stats;  // optional JSON output path for observed [24][128] router hits
  // 3-tier paged Q4_0 full-attention KV cache (for 1M context on <= 12 GB and <= 6 GB GPUs):
  //   kv_paged / kv_vram_mb > 0: keeps hot 4096-token KV pages in VRAM, overflows warm pages to lazy
  //   pinned UVA HostRegion slabs (<= kv_ram_mb), and spills cold pages to a crash-safe NVMe SwapFile.
  bool kv_paged = false;
  int kv_vram_mb = 0;               // 0 = auto when kv_paged (256 MiB with offload, 1536 MiB otherwise)
  int kv_ram_mb = 12288;            // pinned UVA host RAM cap in MiB before spilling to NVMe SwapFile
  std::filesystem::path kv_swap_dir;  // optional directory for Tier-2 NVMe KV swap file
  // Atomic streaming disk checkpoint & resume for long-context (1M+) prefill runs:
  std::filesystem::path kv_ckpt_path;    // path for periodic & SIGTERM atomic KV checkpoints (auto-resumes if prefix matches)
  int kv_ckpt_every = 0;                 // save atomic checkpoint every N tokens during prefill (e.g. 131072; 0 = SIGTERM/manual only)
  std::filesystem::path kv_ckpt_resume;  // explicit checkpoint path to resume from (fails loudly if missing or mismatched)
  // Continuous batching: max concurrent sequences (1 = single-request EngineGenerator, zero B=1 regression).
  int max_batch = 1;
  // Shared paged full-attention KV pool capacity in tokens (0 = auto-size from free VRAM).
  int max_kv_tokens = 0;
  // Interleaved chunked prefill when other slots are actively decoding: max prefill tokens per
  // slice during mixed read+decode (0 disables interleaving; full prefill_chunk is used when idle).
  int prefill_interleave_chunk = 512;
  // Target per-stream decode speed floor during mixed read+decode, as a fraction of full-batch
  // (max_batch) decode speed (e.g. 0.80 => keep each active decoder >= 80% of 4-stream decode).
  float mixed_decode_ratio = 0.80f;
  // Number of recurrent passes over layers 6..17 (default 3 = 48 visits; 1..8 supported for varloop models).
  int loop_passes = 3;
  // Mech-interp / persona hook options: pre-capture hooked CUDA graphs when enabled, and enforce refusal map.
  bool enable_persona_hooks = false;
  std::filesystem::path refusal_map;
};

// One active sequence entering a batched decode step (continuous batching).
struct BatchSlotStep {
  int slot = -1;
  int32_t token = 0;
};

// Output of one active sequence from a multi-token batched step (batched speculative decoding).
struct BatchSlotOutput {
  static constexpr int kMaxTokens = 8;
  int32_t tokens[kMaxTokens] = {};
  int count = 0;         // emitted tokens in this step (1..kMaxTokens)
  int verify_width = 1;  // target verify width m used for this slot
  int accepted = 0;      // accepted draft tokens (count - 1)
};

class Generator {
 public:
  virtual ~Generator() = default;
  virtual std::string model_id() const = 0;  // "inkling-2-mini-abliterated" with an ablation, else "inkling-2-mini"
  virtual int max_model_len() const = 0;     // = max_ctx
  // One request at a time. Emits sampled tokens in order through on_tokens (a speculative round emits several);
  // stops after a token in stop_ids, after max_new_tokens, or when on_tokens returns false (cancel).
  // Throws std::invalid_argument when the prompt does not fit max_ctx (reserve as engine.py does).
  virtual GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) = 0;
  // The same for a prompt with images. Generators without image input take prompts without images only.
  virtual GenerateStats generate(const Prompt& prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens) {
    if (!prompt.images.empty()) throw std::invalid_argument("this generator does not take images");
    return generate(prompt.ids, max_new_tokens, sampling, stop_ids, on_tokens);
  }
  // Variant that delivers completed prefill stats through `on_prefill` strictly before the first token is
  // emitted via `on_tokens`, so request threads that stop early (e.g. stop strings) still observe exact
  // prefill/cache accounting. Subclasses that do not override this fall back to firing `on_prefill` on return.
  virtual GenerateStats generate(const Prompt& prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens,
                                 const PrefillCallback& on_prefill) {
    GenerateStats stats = generate(prompt, max_new_tokens, sampling, stop_ids, on_tokens);
    if (on_prefill) on_prefill(stats);
    return stats;
  }
  virtual bool supports_images() const { return false; }
  virtual KvCacheMode kv_cache_mode() const { return KvCacheMode::PerPass; }
  // Modes supported by this worker's execution profile (e.g. {"per_pass", "last_full"} for B=1 non-spec TP=1 L=3,
  // or {"per_pass"} for speculative, continuous-batch B>1, TP>1, or --loop-passes != 3 workers). Active mode is fixed
  // at worker init.
  virtual std::vector<KvCacheMode> supported_kv_cache_modes() const { return {kv_cache_mode()}; }
  virtual int loop_passes() const { return 3; }
  virtual std::string offload_summary() const { return {}; }
  virtual PrefixCacheStats cache_stats() const { return {}; }

  // ---- Optional continuous-batching interface (used by Scheduler and ingine-bench when max_batch_size() > 1)
  virtual int max_batch_size() const { return 1; }
  virtual bool can_admit_slot(int /*prompt_tokens*/, int /*max_new_tokens*/) const { return false; }
  // Prompt-aware admission check accounting for reusable shared committed KV pages (defaults to cold estimate).
  virtual bool can_admit_slot(const Prompt& prompt, int max_new_tokens, const Sampling& /*sampling*/) const {
    return can_admit_slot(int(prompt.ids.size()), max_new_tokens);
  }
  // Allocates a free slot, prefills `prompt`, samples `first_token` and fills `stats`. Returns slot id >= 0,
  // or -1 if no slot / KV pages are currently available. Throws std::invalid_argument on invalid prompt.
  virtual int start_slot(const Prompt& /*prompt*/, int /*max_new_tokens*/, const Sampling& /*sampling*/,
                         int32_t& /*first_token*/, GenerateStats& /*stats*/) {
    throw std::logic_error("this generator does not support batched decoding");
  }
  // Optional incremental chunked prefill interface so Scheduler::loop_batched() can interleave
  // bounded prefill slices with active decode steps (preventing multi-second decode stalls).
  virtual bool supports_chunked_prefill() const { return false; }
  virtual int prefill_interleave_chunk() const { return 0; }
  virtual float mixed_decode_ratio() const { return 0.80f; }
  // Allocates a slot, reserves KV pages, resets slot state, and queues `prompt` for chunked prefill.
  // Returns slot id >= 0, or -1 if no slot / KV pages are available. Fast (< 1 ms, no prefill compute).
  virtual int begin_prefill_slot(const Prompt& /*prompt*/, int /*max_new_tokens*/, const Sampling& /*sampling*/) {
    throw std::logic_error("this generator does not support chunked prefill");
  }
  // Executes one prefill chunk of up to `max_chunk_tokens` (<= 0 => full prefill_chunk).
  // Returns false while prompt tokens remain; returns true on the final chunk with `first_token` and `stats` filled.
  virtual bool step_prefill_slot(int /*slot*/, int /*max_chunk_tokens*/, int32_t& /*first_token*/,
                                 GenerateStats& /*stats*/) {
    throw std::logic_error("this generator does not support chunked prefill");
  }
  // Returns remaining unprefilled prompt tokens for an in-flight chunked-prefill slot (0 if none).
  virtual int prefill_remaining_tokens(int /*slot*/) const { return 0; }
  // Runs one batched decode step across `inputs` (1..max_batch_size() active slots), sampling each slot's
  // next token into `out_tokens[i]` and advancing each slot's position.
  virtual void step_slots(std::span<const BatchSlotStep> /*inputs*/, std::span<int32_t> /*out_tokens*/) {
    throw std::logic_error("this generator does not support batched decoding");
  }
  // Multi-token variant for batched speculative decoding: emits 1..kMaxTokens tokens per active slot.
  // Default falls back to single-token step_slots().
  virtual bool supports_spec_batch() const { return false; }
  virtual void step_slots_multi(std::span<const BatchSlotStep> inputs, std::span<BatchSlotOutput> outputs) {
    std::vector<int32_t> next(inputs.size(), 0);
    step_slots(inputs, next);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      outputs[i] = BatchSlotOutput{};
      outputs[i].tokens[0] = next[i];
      outputs[i].count = 1;
      outputs[i].verify_width = 1;
      outputs[i].accepted = 0;
    }
  }
  // Optional diagnostic summary of batched speculative decoding counters (tok/verify, m histogram, timings).
  virtual void reset_spec_metrics() {}
  virtual std::string spec_metrics_summary() const { return {}; }
  // Releases `slot` and returns its KV pages immediately. Safe to call on any active slot.
  virtual void release_slot(int /*slot*/) {}

  // Shared single-request lifecycle over the continuous-batching slot API (used by BatchGenerator::generate()
  // and unit tests so slot/KV-page exception cleanup is tested directly through the production path).
  GenerateStats generate_from_slots(const Prompt& full_prompt, int max_new_tokens, const Sampling& sampling,
                                    std::span<const int32_t> stop_ids,
                                    const std::function<bool(std::span<const int32_t>)>& on_tokens,
                                    const PrefillCallback& on_prefill) {
    GenerateStats stats;
    std::int32_t token = 0;
    const int n = int(full_prompt.ids.size());
    if (!sampling.forced.empty()) max_new_tokens = std::min(max_new_tokens, int(sampling.forced.size()));
    const int max_len = max_model_len();
    if (n + max_new_tokens > max_len) {
      max_new_tokens = max_len - n;
      if (max_new_tokens <= 0)
        throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                    std::to_string(max_len));
    }
    const int s = start_slot(full_prompt, max_new_tokens, sampling, token, stats);
    if (s < 0) throw std::runtime_error("BatchGenerator::generate: no free batch slot");
    auto is_stop = [&](std::int32_t tok) {
      return std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end();
    };
    const auto t0 = std::chrono::steady_clock::now();
    try {
      if (on_prefill) on_prefill(stats);
      int emitted = 0;
      auto emit_one = [&](std::int32_t tok) -> bool {
        ++emitted;
        stats.generated_tokens = emitted;
        const std::int32_t out[1] = {tok};
        const bool go_on = on_tokens(out);
        return emitted < max_new_tokens && !is_stop(tok) && go_on;
      };
      if (emit_one(token)) {
        for (;;) {
          const BatchSlotStep step[1] = {{s, token}};
          BatchSlotOutput out[1];
          step_slots_multi(step, out);
          if (out[0].verify_width > 1 || supports_spec_batch()) {
            stats.spec_rounds += 1;
            stats.spec_accepted += out[0].accepted;
          }
          bool cont = true;
          for (int k = 0; k < out[0].count; ++k) {
            token = out[0].tokens[k];
            if (!emit_one(token)) {
              cont = false;
              break;
            }
          }
          if (!cont) break;
        }
      }
    } catch (...) {
      release_slot(s);
      throw;
    }
    stats.decode_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    release_slot(s);
    return stats;
  }
};

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options);  // the model (ingine_model)

}  // namespace ingine::model

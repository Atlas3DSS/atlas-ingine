// The token generator interface the server drives (mirror of engine.Engine.generate). CUDA-free so the
// scheduler, the HTTP layer and their tests build without the model; ingine_model implements make_generator.
#pragma once

#include "ingine/persona.h"
#include "ingine/prefix_cache.h"

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
#include <vector>

namespace ingine::model {

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
  float spec_threshold = 0.8f;
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
  // Mech-interp / persona hook options: pre-capture hooked CUDA graphs when enabled, and enforce refusal map.
  bool enable_persona_hooks = false;
  std::filesystem::path refusal_map;
};

// One active sequence entering a batched decode step (continuous batching).
struct BatchSlotStep {
  int slot = -1;
  int32_t token = 0;
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
  // Runs one batched decode step across `inputs` (1..max_batch_size() active slots), sampling each slot's
  // next token into `out_tokens[i]` and advancing each slot's position.
  virtual void step_slots(std::span<const BatchSlotStep> /*inputs*/, std::span<int32_t> /*out_tokens*/) {
    throw std::logic_error("this generator does not support batched decoding");
  }
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
      for (int i = 0; i < max_new_tokens; ++i) {
        stats.generated_tokens = i + 1;
        const std::int32_t out[1] = {token};
        const bool go_on = on_tokens(out);
        if (i + 1 == max_new_tokens || is_stop(token) || !go_on) break;
        const BatchSlotStep step[1] = {{s, token}};
        std::int32_t next[1] = {0};
        step_slots(step, next);
        token = next[0];
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

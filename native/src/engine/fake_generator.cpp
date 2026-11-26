#include "engine/fake_generator.h"

#include "text/chat.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <optional>
#include <cstring>
#include <thread>
#include <unordered_set>
#include <utility>

namespace ingine::engine {
namespace {

constexpr std::string_view kMessageModel = text::MESSAGE_MODEL;
constexpr std::string_view kMessageUser = text::MESSAGE_USER;
constexpr std::string_view kContentText = text::CONTENT_TEXT;
constexpr std::string_view kContentThinking = text::CONTENT_THINKING;
constexpr std::string_view kContentInvokeToolJson = text::CONTENT_INVOKE_TOOL_JSON;
constexpr std::string_view kEndMessage = text::END_MESSAGE;
constexpr std::string_view kEndSampling = text::END_SAMPLING;

// Python's str \s (ASCII part, plus the \x1c-\x1f separators).
bool is_space(unsigned char c) { return c == ' ' || (c >= '\t' && c <= '\r') || (c >= 0x1c && c <= 0x1f); }
// Python's str \w: ASCII letters, digits and '_'; every non-ASCII byte counts as part of a word character.
bool is_word(unsigned char c) { return std::isalnum(c) || c == '_' || c >= 0x80; }

// Number of Unicode code points (Python len) of UTF-8 text.
size_t code_points(std::string_view s) {
  return std::count_if(s.begin(), s.end(), [](char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; });
}

// re.fullmatch(r"\s*call\s+(\w+)\s*(\{.*\})?\s*", s, re.S): {name, args-or-empty}.
std::optional<std::pair<std::string, std::string>> match_call(std::string_view s) {
  size_t i = 0;
  while (i < s.size() && is_space(s[i])) ++i;
  if (s.substr(i, 4) != "call") return std::nullopt;
  i += 4;
  const size_t ws = i;
  while (i < s.size() && is_space(s[i])) ++i;
  if (i == ws) return std::nullopt;
  const size_t name0 = i;
  while (i < s.size() && is_word(s[i])) ++i;
  if (i == name0) return std::nullopt;
  std::string name(s.substr(name0, i - name0));
  while (i < s.size() && is_space(s[i])) ++i;
  size_t end = s.size();
  while (end > i && is_space(s[end - 1])) --end;
  std::string_view rest = s.substr(i, end - i);
  if (rest.empty()) return std::pair{std::move(name), std::string()};
  if (rest.size() >= 2 && rest.front() == '{' && rest.back() == '}') return std::pair{std::move(name), std::string(rest)};
  return std::nullopt;
}

}  // namespace

FakeGenerator::FakeGenerator(const text::Tokenizer& tokenizer, std::string model_id, int max_model_len,
                             Responder responder, double token_delay, int loop_passes,
                             model::KvCacheMode kv_cache_mode, bool allow_last_full)
    : tokenizer_(tokenizer),
      model_id_(std::move(model_id)),
      max_model_len_(max_model_len),
      loop_passes_(loop_passes),
      responder_(std::move(responder)),
      token_delay_(token_delay),
      kv_cache_mode_(kv_cache_mode) {
  if (loop_passes_ < 1 || loop_passes_ > 8) {
    throw std::invalid_argument("loop_passes must be in 1..8");
  }
  const bool can_last_full = allow_last_full && (loop_passes_ == 3);
  if (kv_cache_mode_ == model::KvCacheMode::LastFull && !can_last_full) {
    if (loop_passes_ != 3) {
      throw std::invalid_argument("--loop-passes != 3 is not supported with --kv-cache-mode last_full");
    }
    throw std::invalid_argument("--kv-cache-mode last_full is not supported with this worker configuration");
  }
  if (can_last_full) {
    supported_kv_cache_modes_ = {model::KvCacheMode::PerPass, model::KvCacheMode::LastFull};
  } else {
    supported_kv_cache_modes_ = {model::KvCacheMode::PerPass};
  }
}

void FakeGenerator::set_kv_cache_mode(model::KvCacheMode mode) {
  if (mode == model::KvCacheMode::LastFull && loop_passes_ != 3) {
    throw std::invalid_argument("--loop-passes != 3 is not supported with --kv-cache-mode last_full");
  }
  kv_cache_mode_ = mode;
}

std::string FakeGenerator::echo(std::span<const int32_t> prompt_ids) const {
  const std::string text = tokenizer_.decode(prompt_ids);
  // re.findall(r"<\|message_user\|><\|content_text\|>(.*?)<\|end_message\|>", text, re.S)[-1]
  std::string open = std::string(kMessageUser) + std::string(kContentText);
  std::string last;
  for (size_t pos = 0;;) {
    const size_t a = text.find(open, pos);
    if (a == std::string::npos) break;
    const size_t b = text.find(kEndMessage, a + open.size());
    if (b == std::string::npos) break;
    last = text.substr(a + open.size(), b - a - open.size());
    pos = b + kEndMessage.size();
  }
  std::string out;
  if (auto call = match_call(last)) {
    auto& [name, args_text] = *call;
    nlohmann::ordered_json body = {{"name", name},
                                   {"args", nlohmann::ordered_json::parse(args_text.empty() ? "{}" : args_text)}};
    out.append(kMessageModel).append(kContentThinking).append("Calling ").append(name).append(".");
    out.append(kEndMessage).append(kMessageModel).append(name).append(kContentInvokeToolJson);
    out.append(body.dump(-1, ' ', false, nlohmann::ordered_json::error_handler_t::replace));
    out.append(kEndMessage).append(kEndSampling);
    return out;
  }
  std::string answer = "You said: " + last;
  answer.append(kEndMessage).append(kEndSampling);
  const std::string no_thinking = std::string(kMessageModel) + std::string(kContentText);
  if (text.ends_with(no_thinking)) return answer;  // thinking disabled by prefill
  out.append(kMessageModel).append(kContentThinking).append("The user wrote ");
  out.append(std::to_string(code_points(last))).append(" characters.").append(kEndMessage);
  out.append(kMessageModel).append(kContentText).append(answer);
  return out;
}

void FakeGenerator::set_prefill_stats_hook(PrefillStatsFn hook) {
  std::lock_guard lock(mutex_);
  prefill_stats_hook_ = std::move(hook);
}

model::GenerateStats FakeGenerator::generate(const model::Prompt& full_prompt, int max_new_tokens,
                                             const model::Sampling& sampling, std::span<const int32_t> stop_ids,
                                             const std::function<bool(std::span<const int32_t>)>& on_tokens,
                                             const model::PrefillCallback& on_prefill) {
  const std::span<const int32_t> prompt = full_prompt.ids;
  int image_tokens = 0;
  for (const model::PromptImage& image : full_prompt.images) image_tokens += image.tokens();
  Output reply = responder_ ? responder_(prompt) : Output(echo(prompt));
  std::vector<int32_t> out = std::holds_alternative<std::string>(reply)
                                 ? tokenizer_.encode(std::get<std::string>(reply), /*special=*/true)
                                 : std::get<std::vector<int32_t>>(std::move(reply));
  size_t index;
  PrefillStatsFn hook;
  {
    std::lock_guard lock(mutex_);
    index = calls_.size();
    calls_.push_back(Call{std::vector<int32_t>(prompt.begin(), prompt.end()), max_new_tokens, sampling,
                          int(full_prompt.images.size()), image_tokens, full_prompt.root_prefix_tokens,
                          full_prompt.cache_namespace, full_prompt.bypass_prefix_cache});
    hook = prefill_stats_hook_;
  }
  model::GenerateStats stats;
  stats.prompt_tokens = int(prompt.size());
  stats.image_tokens = image_tokens;
  if (hook) hook(prompt, stats);
  if (on_prefill) on_prefill(stats);

  const std::unordered_set<int32_t> stops(stop_ids.begin(), stop_ids.end());
  const size_t limit = size_t(std::max(max_new_tokens, 0));
  bool cancelled = false;
  int yielded = 0;

  if (sampling.logit_masker) {
    // Honor token-level constrained decoding mask on FakeGenerator too (enables CPU E2E HTTP tests).
    model::TokenMasker& masker = *sampling.logit_masker;
    const int V = std::min(masker.vocab_size(), int(tokenizer_.vocab_size()));
    const size_t words = (size_t(V) + 31u) / 32u;
    std::vector<std::uint32_t> mask(words, 0u);
    auto bit_ok = [&](int32_t id) {
      return id >= 0 && id < V && ((mask[size_t(id) >> 5u] >> (uint32_t(id) & 31u)) & 1u) != 0u;
    };
    const int32_t end_msg = tokenizer_.token_id(kEndMessage).value_or(-1);
    const int32_t end_samp = tokenizer_.token_id(kEndSampling).value_or(-1);
    size_t src_i = 0;
    for (size_t step = 0; step < limit; ++step) {
      int32_t chosen = -1;
      if (!masker.is_active()) {
        if (src_i >= out.size()) break;
        chosen = out[src_i++];
      } else {
        std::fill(mask.begin(), mask.end(), 0u);
        const bool masked = masker.fill_mask(mask.data(), words);
        if (!masked) {
          if (src_i >= out.size()) break;
          chosen = out[src_i++];
        } else if (src_i < out.size() && bit_ok(out[src_i])) {
          chosen = out[src_i++];
        } else {
          // Deterministically pick a valid minimal token allowed by the FSM bitmask.
          if (end_samp >= 0 && bit_ok(end_samp)) {
            chosen = end_samp;
          } else if (end_msg >= 0 && bit_ok(end_msg)) {
            chosen = end_msg;
          } else {
            // Prefer closing quotes before structural delimiters so open strings terminate immediately.
            static constexpr const char* kPrefs[] = {"\"", "}", "]", ":", ",", "{", "[", "0", "true", "false", "null", "a"};
            for (const char* p : kPrefs) {
              const auto enc = tokenizer_.encode(p, false);
              if (enc.size() == 1 && bit_ok(enc[0])) {
                chosen = enc[0];
                break;
              }
            }
            if (chosen < 0) {
              for (int32_t id = 0; id < V; ++id) {
                if (bit_ok(id)) {
                  const std::string b = tokenizer_.token_bytes(id);
                  if (!b.empty() && b.size() == 1 && static_cast<unsigned char>(b[0]) >= 0x20 &&
                      static_cast<unsigned char>(b[0]) < 0x7f) {
                    chosen = id;
                    break;
                  }
                }
              }
            }
            if (chosen < 0) {
              for (int32_t id = 0; id < V; ++id) {
                if (bit_ok(id)) {
                  chosen = id;
                  break;
                }
              }
            }
          }
        }
      }
      if (chosen < 0) break;
      masker.advance(chosen);
      if (token_delay_ > 0) std::this_thread::sleep_for(std::chrono::duration<double>(token_delay_));
      {
        std::lock_guard lock(mutex_);
        calls_[index].yielded = ++yielded;
      }
      const bool more = on_tokens(std::span<const int32_t>(&chosen, 1));
      if (stops.contains(chosen)) break;
      if (!more) {
        cancelled = int(step) + 1 < max_new_tokens;
        break;
      }
    }
  } else {
    const size_t n_emit = std::min(out.size(), limit);
    for (size_t i = 0; i < n_emit; ++i) {
      if (token_delay_ > 0) std::this_thread::sleep_for(std::chrono::duration<double>(token_delay_));
      {
        std::lock_guard lock(mutex_);
        calls_[index].yielded = ++yielded;
      }
      const bool more = on_tokens(std::span<const int32_t>(&out[i], 1));
      if (stops.contains(out[i])) break;
      if (!more) {
        // Stop ids and max_new_tokens are handled here as well, so a refusal otherwise means cancellation.
        cancelled = int(i) + 1 < max_new_tokens;
        break;
      }
    }
  }
  {
    std::lock_guard lock(mutex_);
    calls_[index].cancelled = cancelled;
    calls_[index].done = true;
  }
  stats.generated_tokens = yielded;
  return stats;
}

std::vector<FakeGenerator::Call> FakeGenerator::calls() const {
  std::lock_guard lock(mutex_);
  return calls_;
}

}  // namespace ingine::engine

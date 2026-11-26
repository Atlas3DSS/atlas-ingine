// Anthropic Messages API (/v1/messages and /v1/messages/count_tokens) request translation and response formatting
// over ingine's existing ChatFormat, StreamParser, and Scheduler (for Claude Code and Omarchy agents).
#pragma once

#include "serve/openai.h"

#include <optional>
#include <string>
#include <string_view>

namespace ingine::serve {

// Translates an Anthropic Messages API JSON request body into a ChatRequest via OpenAI request parsing.
// When `require_max_tokens` is false (used by /v1/messages/count_tokens), omitted max_tokens defaults to 1.
ChatRequest parse_anthropic_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                                    const model::Sampling& defaults = {}, const ImageOptions& images = {},
                                    bool require_max_tokens = true);

// {"type": "error", "error": {"type": "...", "message": "..."}}
Json anthropic_error_body(int status, std::string_view message);

// Maps StreamParser finish_reason ("stop" | "length" | "tool_calls") and optional matched stop_sequence to
// Anthropic stop_reason ("end_turn" | "max_tokens" | "tool_use" | "stop_sequence").
std::optional<std::string> anthropic_stop_reason(const std::optional<std::string>& finish_reason,
                                                 const std::optional<std::string>& stop_sequence = std::nullopt);

// Formats a completed turn as an Anthropic Messages API JSON response body.
Json anthropic_message_response(const std::string& msg_id, const std::string& model_id,
                                const text::Completion& result, int input_tokens, int output_tokens,
                                int cache_read_input_tokens);

// Formats one Anthropic Server-Sent Event ("event: <name>\ndata: <json>\n\n").
std::string anthropic_sse(std::string_view event_name, const Json& payload);

}  // namespace ingine::serve

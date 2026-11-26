#include "serve/anthropic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ingine::serve {
namespace {

const Json* get(const Json& obj, const char* key) {
  const auto it = obj.find(key);
  return (it == obj.end() || it->is_null()) ? nullptr : &*it;
}

const std::string* string_at(const Json& obj, const char* key) {
  const auto it = obj.find(key);
  return (it != obj.end() && it->is_string()) ? &it->get_ref<const std::string&>() : nullptr;
}

std::string join_text_blocks(const Json& value, const char* field_name) {
  if (value.is_string()) return value.get<std::string>();
  if (!value.is_array()) throw BadRequest(std::string("'") + field_name + "' must be a string or list of blocks", field_name);
  std::string out;
  for (std::size_t i = 0; i < value.size(); ++i) {
    const Json& blk = value[i];
    if (!blk.is_object()) {
      throw BadRequest(std::string("'") + field_name + "[" + std::to_string(i) + "]' must be an object", field_name);
    }
    const std::string* type = string_at(blk, "type");
    if (!type || *type != "text") {
      throw BadRequest(std::string("'") + field_name + "[" + std::to_string(i) + "]' only supports type 'text'", field_name);
    }
    const std::string* text = string_at(blk, "text");
    if (!text) {
      throw BadRequest(std::string("'") + field_name + "[" + std::to_string(i) + "].text' must be a string", field_name);
    }
    if (!out.empty() && !text->empty()) out += "\n\n";
    out += *text;
  }
  return out;
}

Json convert_tool_result_content(const Json& content) {
  if (content.is_null()) return "";
  if (content.is_string()) return content;
  if (!content.is_array()) return dump(content);
  bool has_image = false;
  for (const Json& blk : content) {
    if (blk.is_object()) {
      const std::string* t = string_at(blk, "type");
      if (t && *t == "image") has_image = true;
    }
  }
  if (!has_image) {
    std::string text;
    for (const Json& blk : content) {
      if (blk.is_string()) {
        if (!text.empty()) text += "\n";
        text += blk.get_ref<const std::string&>();
      } else if (blk.is_object()) {
        const std::string* t = string_at(blk, "type");
        if (t && *t == "text") {
          if (const std::string* s = string_at(blk, "text")) {
            if (!text.empty() && !s->empty()) text += "\n";
            text += *s;
          }
        } else {
          if (!text.empty()) text += "\n";
          text += dump(blk);
        }
      }
    }
    return text;
  }
  // Preserve mixed text + image content parts so openai.cpp::collect_images expands Anthropic image blocks.
  Json parts = Json::array();
  for (const Json& blk : content) {
    if (blk.is_object()) {
      const std::string* t = string_at(blk, "type");
      if (t && (*t == "text" || *t == "image")) {
        parts.push_back(blk);
      } else {
        parts.push_back(Json{{"type", "text"}, {"text", dump(blk)}});
      }
    }
  }
  return parts;
}

Json translate_messages(const Json& body) {
  const Json* msgs = get(body, "messages");
  if (!msgs || !msgs->is_array() || msgs->empty()) {
    throw BadRequest("'messages' must be a non-empty list", "messages");
  }
  Json out = Json::array();
  if (const Json* sys = get(body, "system")) {
    const std::string sys_text = join_text_blocks(*sys, "system");
    if (!sys_text.empty()) {
      out.push_back(Json{{"role", "system"}, {"content", sys_text}});
    }
  }

  for (std::size_t i = 0; i < msgs->size(); ++i) {
    const Json& m = (*msgs)[i];
    if (!m.is_object()) {
      throw BadRequest("messages[" + std::to_string(i) + "] must be an object", "messages");
    }
    const std::string* role = string_at(m, "role");
    if (!role || (*role != "user" && *role != "assistant")) {
      throw BadRequest("messages[" + std::to_string(i) + "].role must be 'user' or 'assistant'", "messages");
    }
    const auto cit = m.find("content");
    if (cit == m.end() || cit->is_null()) {
      throw BadRequest("messages[" + std::to_string(i) + "].content is required", "messages");
    }
    if (cit->is_string()) {
      out.push_back(Json{{"role", *role}, {"content", *cit}});
      continue;
    }
    if (!cit->is_array()) {
      throw BadRequest("messages[" + std::to_string(i) + "].content must be a string or list of blocks", "messages");
    }

    if (*role == "user") {
      Json user_parts = Json::array();
      auto flush_user = [&] {
        if (user_parts.empty()) return;
        bool all_text = true;
        for (const Json& p : user_parts) {
          const std::string* t = string_at(p, "type");
          if (!t || *t != "text") {
            all_text = false;
            break;
          }
        }
        if (all_text) {
          std::string joined;
          for (const Json& p : user_parts) {
            const std::string& s = p.at("text").get_ref<const std::string&>();
            if (!joined.empty() && !s.empty()) joined += "\n";
            joined += s;
          }
          out.push_back(Json{{"role", "user"}, {"content", std::move(joined)}});
        } else {
          out.push_back(Json{{"role", "user"}, {"content", user_parts}});
        }
        user_parts = Json::array();
      };

      for (std::size_t j = 0; j < cit->size(); ++j) {
        const Json& blk = (*cit)[j];
        if (!blk.is_object()) {
          throw BadRequest("messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "] must be an object",
                           "messages");
        }
        const std::string* type = string_at(blk, "type");
        if (!type) {
          throw BadRequest("messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "].type is required",
                           "messages");
        }
        if (*type == "tool_result") {
          flush_user();
          const std::string* tid = string_at(blk, "tool_use_id");
          if (!tid || tid->empty()) {
            throw BadRequest("messages[" + std::to_string(i) + "].content[" + std::to_string(j) +
                                 "].tool_use_id must be a non-empty string",
                             "messages");
          }
          const auto rc_it = blk.find("content");
          Json tool_content = (rc_it == blk.end()) ? Json("") : convert_tool_result_content(*rc_it);
          if (const auto err_it = blk.find("is_error"); err_it != blk.end() && err_it->is_boolean() && err_it->get<bool>()) {
            if (tool_content.is_string()) {
              tool_content = std::string("[tool_error] ") + tool_content.get_ref<const std::string&>();
            }
          }
          out.push_back(Json{{"role", "tool"}, {"tool_call_id", *tid}, {"content", std::move(tool_content)}});
        } else if (*type == "text") {
          const std::string* text = string_at(blk, "text");
          if (!text) {
            throw BadRequest("messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "].text must be a string",
                             "messages");
          }
          user_parts.push_back(Json{{"type", "text"}, {"text", *text}});
        } else if (*type == "image") {
          user_parts.push_back(blk);
        } else {
          throw BadRequest("unsupported user content block type '" + *type + "'", "messages");
        }
      }
      flush_user();
      continue;
    }

    // Assistant turn: thinking + text + tool_use blocks.
    std::string reasoning, text;
    Json tool_calls = Json::array();
    for (std::size_t j = 0; j < cit->size(); ++j) {
      const Json& blk = (*cit)[j];
      if (!blk.is_object()) {
        throw BadRequest("messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "] must be an object",
                         "messages");
      }
      const std::string* type = string_at(blk, "type");
      if (!type) {
        throw BadRequest("messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "].type is required",
                         "messages");
      }
      if (*type == "thinking") {
        if (const std::string* t = string_at(blk, "thinking")) reasoning += *t;
      } else if (*type == "redacted_thinking") {
        // Opaque Anthropic signature block; safely ignored.
      } else if (*type == "text") {
        const std::string* t = string_at(blk, "text");
        if (!t) {
          throw BadRequest("messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "].text must be a string",
                           "messages");
        }
        text += *t;
      } else if (*type == "tool_use") {
        const std::string* id = string_at(blk, "id");
        const std::string* name = string_at(blk, "name");
        if (!id || id->empty() || !name || name->empty()) {
          throw BadRequest("messages[" + std::to_string(i) + "].content[" + std::to_string(j) +
                               "]: tool_use requires non-empty 'id' and 'name'",
                           "messages");
        }
        const auto in_it = blk.find("input");
        const std::string args =
            (in_it == blk.end() || in_it->is_null()) ? std::string("{}") : (in_it->is_string() ? in_it->get<std::string>() : dump(*in_it));
        tool_calls.push_back(
            Json{{"id", *id}, {"type", "function"}, {"function", {{"name", *name}, {"arguments", args}}}});
      } else {
        throw BadRequest("unsupported assistant content block type '" + *type + "'", "messages");
      }
    }
    Json amsg{{"role", "assistant"}, {"content", text}};
    if (!reasoning.empty()) amsg["reasoning_content"] = std::move(reasoning);
    if (!tool_calls.empty()) amsg["tool_calls"] = std::move(tool_calls);
    out.push_back(std::move(amsg));
  }
  return out;
}

Json translate_tools(const Json& tools_in) {
  if (!tools_in.is_array()) throw BadRequest("'tools' must be a list", "tools");
  Json out = Json::array();
  for (std::size_t i = 0; i < tools_in.size(); ++i) {
    const Json& t = tools_in[i];
    if (!t.is_object()) throw BadRequest("tools[" + std::to_string(i) + "] must be an object", "tools");
    if (const std::string* kind = string_at(t, "type"); kind && *kind == "function" && t.contains("function")) {
      out.push_back(t);
      continue;
    }
    const std::string* name = string_at(t, "name");
    if (!name || name->empty()) {
      throw BadRequest("tools[" + std::to_string(i) + "].name must be a non-empty string", "tools");
    }
    Json fn{{"name", *name}};
    if (const std::string* desc = string_at(t, "description")) fn["description"] = *desc;
    if (const Json* schema = get(t, "input_schema")) {
      if (!schema->is_object()) {
        throw BadRequest("tools[" + std::to_string(i) + "].input_schema must be an object", "tools");
      }
      fn["parameters"] = *schema;
    } else {
      fn["parameters"] = Json{{"type", "object"}, {"properties", Json::object()}};
    }
    out.push_back(Json{{"type", "function"}, {"function", std::move(fn)}});
  }
  return out;
}

}  // namespace

ChatRequest parse_anthropic_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                                    const model::Sampling& defaults, const ImageOptions& images,
                                    bool require_max_tokens) {
  if (!body.is_object()) throw BadRequest("request body must be a JSON object");

  Json openai = Json::object();

  // max_tokens is required on POST /v1/messages (optional on /v1/messages/count_tokens).
  const Json* max_tok = get(body, "max_tokens");
  if (!max_tok) {
    if (require_max_tokens) throw BadRequest("'max_tokens' is required", "max_tokens");
    openai["max_tokens"] = 1;
  } else {
    openai["max_tokens"] = *max_tok;
  }

  openai["messages"] = translate_messages(body);

  bool disable_tools = false;
  if (const Json* tc = get(body, "tool_choice")) {
    if (!tc->is_object()) throw BadRequest("'tool_choice' must be an object", "tool_choice");
    const std::string* tc_type = string_at(*tc, "type");
    if (!tc_type || (*tc_type != "auto" && *tc_type != "any" && *tc_type != "none" && *tc_type != "tool")) {
      throw BadRequest("'tool_choice.type' must be 'auto', 'any', 'none', or 'tool'", "tool_choice");
    }
    if (*tc_type == "none") disable_tools = true;
    if (*tc_type == "tool") {
      const std::string* name = string_at(*tc, "name");
      if (!name || name->empty()) throw BadRequest("'tool_choice.name' is required when type is 'tool'", "tool_choice");
    }
  }
  if (!disable_tools) {
    if (const Json* tools = get(body, "tools")) {
      Json translated = translate_tools(*tools);
      if (!translated.empty()) openai["tools"] = std::move(translated);
    }
  }

  // Anthropic thinking config: {"type": "enabled", "budget_tokens": ...} | {"type": "adaptive"} | {"type": "disabled"}.
  if (const Json* thinking = get(body, "thinking")) {
    if (!thinking->is_object()) throw BadRequest("'thinking' must be an object", "thinking");
    const std::string* ttype = string_at(*thinking, "type");
    if (!ttype || (*ttype != "enabled" && *ttype != "disabled" && *ttype != "adaptive")) {
      throw BadRequest("'thinking.type' must be 'enabled', 'adaptive', or 'disabled'", "thinking");
    }
    if (*ttype == "disabled") {
      openai["enable_thinking"] = false;
      openai["reasoning_effort"] = "none";
    } else {
      openai["enable_thinking"] = true;
    }
  }
  if (const Json* effort = get(body, "reasoning_effort")) openai["reasoning_effort"] = *effort;

  if (const Json* stops = get(body, "stop_sequences")) {
    if (!stops->is_array() ||
        !std::all_of(stops->begin(), stops->end(), [](const Json& s) { return s.is_string(); })) {
      throw BadRequest("'stop_sequences' must be a list of strings", "stop_sequences");
    }
    openai["stop"] = *stops;
  }

  for (const char* key : {"temperature", "top_p", "top_k", "stream", "persona", "seed"}) {
    if (auto it = body.find(key); it != body.end() && !it->is_null()) openai[key] = *it;
  }

  if (const Json* meta = get(body, "metadata")) {
    if (!meta->is_object()) throw BadRequest("'metadata' must be an object", "metadata");
    if (const Json* uid = get(*meta, "user_id")) {
      if (!uid->is_string()) throw BadRequest("'metadata.user_id' must be a string", "metadata");
      if (!uid->get_ref<const std::string&>().empty()) openai["user"] = *uid;
    }
  }

  return parse_request(openai, fmt, max_model_len, defaults, images);
}

Json anthropic_error_body(int status, std::string_view message) {
  const char* kind = status == 400 ? "invalid_request_error"
                     : status == 401 ? "authentication_error"
                     : status == 403 ? "permission_error"
                     : status == 404 ? "not_found_error"
                     : status == 429 ? "rate_limit_error"
                     : status == 503 ? "overloaded_error"
                                     : "api_error";
  return Json{{"type", "error"}, {"error", {{"type", kind}, {"message", message}}}};
}

std::optional<std::string> anthropic_stop_reason(const std::optional<std::string>& finish_reason) {
  if (!finish_reason) return std::nullopt;
  if (*finish_reason == "tool_calls") return std::string("tool_use");
  if (*finish_reason == "length") return std::string("max_tokens");
  return std::string("end_turn");
}

Json anthropic_message_response(const std::string& msg_id, const std::string& model_id,
                                const text::Completion& result, int input_tokens, int output_tokens,
                                int cache_read_input_tokens) {
  Json content = Json::array();
  if (!result.reasoning_content.empty()) {
    content.push_back(Json{{"type", "thinking"}, {"thinking", result.reasoning_content}});
  }
  if (!result.content.empty() || (result.reasoning_content.empty() && result.tool_calls.empty())) {
    content.push_back(Json{{"type", "text"}, {"text", result.content}});
  }
  for (const text::ToolCall& tc : result.tool_calls) {
    Json input = Json::object();
    if (!tc.arguments.empty()) {
      try {
        Json parsed = Json::parse(tc.arguments);
        if (parsed.is_object()) input = std::move(parsed);
        else input = Json{{"value", std::move(parsed)}};
      } catch (const Json::parse_error&) {
        input = Json{{"raw", tc.arguments}};
      }
    }
    content.push_back(Json{{"type", "tool_use"}, {"id", tc.id}, {"name", tc.name}, {"input", std::move(input)}});
  }
  const auto stop = anthropic_stop_reason(result.finish_reason);
  return Json{{"id", msg_id},
              {"type", "message"},
              {"role", "assistant"},
              {"model", model_id},
              {"content", std::move(content)},
              {"stop_reason", stop ? Json(*stop) : Json()},
              {"stop_sequence", nullptr},
              {"usage",
               {{"input_tokens", input_tokens},
                {"output_tokens", output_tokens},
                {"cache_creation_input_tokens", 0},
                {"cache_read_input_tokens", std::max(0, cache_read_input_tokens)}}}};
}

std::string anthropic_sse(std::string_view event_name, const Json& payload) {
  const std::string data = dump(payload);
  std::string out;
  out.reserve(event_name.size() + data.size() + 18);
  out.append("event: ").append(event_name).append("\ndata: ").append(data).append("\n\n");
  return out;
}

}  // namespace ingine::serve

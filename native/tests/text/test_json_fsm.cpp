// Unit tests for token-level RFC 8259 JSON and JSON-Schema FSM constrained decoding (#830).
#include "text/chat.h"
#include "text/json_fsm.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using namespace ingine::text;

// Minimal synthetic byte-level BPE tokenizer JSON (valid for Tokenizer::from_json) covering ASCII +
// multi-byte JSON chunks + Inkling-2 special tokens.
std::string make_test_tokenizer_json() {
  Json vocab = Json::object();
  int next_id = 0;
  // Emit all 256 GPT-2 ByteLevel single-byte tokens (required by Tokenizer::from_json).
  auto encode_cp_utf8 = [](int cp) -> std::string {
    std::string s;
    if (cp < 0x80) {
      s.push_back(char(cp));
    } else {
      s.push_back(char(0xC0 | (cp >> 6)));
      s.push_back(char(0x80 | (cp & 0x3F)));
    }
    return s;
  };
  int next_cp = 256;
  for (int b = 0; b < 256; ++b) {
    const bool visible = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    const int cp = visible ? b : next_cp++;
    vocab[encode_cp_utf8(cp)] = next_id++;
  }
  // Multi-byte tokens (to verify multi-byte delimiter transitions & BPE chunk soundness)
  const char* chunks[] = {
      "true", "false", "null", "hello", "world", "status", "score", "tags", "ok", "fail",
      "42,", "123}", "\"ok\"", "\",", "\":", "},", "]}", "01", "```json", "You",
  };
  for (const char* ck : chunks) {
    if (!vocab.contains(ck)) vocab[ck] = next_id++;
  }

  Json added = Json::array();
  const char* specials[] = {
      "<|message_model|>", "<|message_user|>", "<|message_system|>", "<|message_tool|>",
      "<|content_text|>", "<|content_thinking|>", "<|content_invoke_tool_json|>",
      "<|content_invoke_tool_text|>", "<|content_tool_error|>",
      "<|end_message|>", "<|content_model_end_sampling|>", "<|endoftext|>",
  };
  for (const char* sp : specials) {
    added.push_back(Json{{"id", next_id++}, {"content", sp}, {"special", true}});
  }

  Json root{
      {"version", "1.0"},
      {"normalizer", nullptr},
      {"pre_tokenizer",
       {{"type", "Sequence"},
        {"pretokenizers",
         Json::array({
             Json{{"type", "Split"},
                  {"pattern",
                   {{"Regex",
                     "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*"
                     "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
                     "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+"
                     "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
                     "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}}},
                  {"behavior", "Isolated"},
                  {"invert", false}},
             Json{{"type", "ByteLevel"}, {"add_prefix_space", false}, {"use_regex", false}},
         })}}},
      {"decoder", {{"type", "ByteLevel"}}},
      {"added_tokens", added},
      {"model",
       {{"type", "BPE"},
        {"ignore_merges", true},
        {"byte_fallback", false},
        {"vocab", vocab},
        {"merges", Json::array()}}},
  };
  return root.dump();
}

bool is_allowed(const ingine::model::TokenMasker& masker, std::int32_t id) {
  std::vector<std::uint32_t> words(masker.mask_words(), 0u);
  if (!masker.fill_mask(words.data(), words.size())) return true;
  if (id < 0 || id >= masker.vocab_size()) return false;
  return ((words[std::size_t(id) >> 5u] >> (std::uint32_t(id) & 31u)) & 1u) != 0u;
}

std::int32_t tok_id(const Tokenizer& tok, std::string_view piece) {
  for (int i = 0; i < int(tok.vocab_size()); ++i) {
    if (tok.token_bytes(i) == piece) return i;
  }
  std::cerr << "missing token piece: " << piece << "\n";
  std::abort();
}

void feed_pieces(ingine::model::TokenMasker& fsm, const Tokenizer& tok, const std::vector<std::string_view>& pieces) {
  for (std::string_view p : pieces) {
    const std::int32_t id = tok_id(tok, p);
    assert(is_allowed(fsm, id));
    fsm.advance(id);
  }
}

void test_thinking_channel_unconstrained_then_json_object() {
  const Tokenizer tok = Tokenizer::from_json(make_test_tokenizer_json());
  const std::vector<std::int32_t> stops = {
      *tok.token_id(END_SAMPLING), *tok.token_id(ENDOFTEXT), *tok.token_id(MESSAGE_USER)};
  auto vocab = VocabByteIndex::build(tok, stops);

  // Prefill ends in <|content_thinking|>: FSM must stay inactive during thinking!
  const std::vector<std::int32_t> prefill = {*tok.token_id(MESSAGE_MODEL), *tok.token_id(CONTENT_THINKING)};
  JsonFormatSpec spec{JsonFormatMode::JsonObject, Json()};
  auto fsm = make_json_fsm(vocab, spec, prefill);
  assert(fsm != nullptr);
  assert(!fsm->is_active());

  // Thinking prose + transition into <|content_text|>
  fsm->advance(tok_id(tok, "You"));
  assert(!fsm->is_active());
  fsm->advance(*tok.token_id(END_MESSAGE));
  fsm->advance(*tok.token_id(MESSAGE_MODEL));
  assert(!fsm->is_active());
  fsm->advance(*tok.token_id(CONTENT_TEXT));
  assert(fsm->is_active());

  // At start of JSON object: '{' allowed; markdown fence / prose / '[' / '0' disallowed!
  assert(is_allowed(*fsm, tok_id(tok, "{")));
  assert(!is_allowed(*fsm, tok_id(tok, "```json")));
  assert(!is_allowed(*fsm, tok_id(tok, "You")));
  assert(!is_allowed(*fsm, tok_id(tok, "[")));
  assert(!is_allowed(*fsm, tok_id(tok, "0")));

  // Feed valid JSON with multi-byte tokens ("42,", "123}", escapes, nested array)
  feed_pieces(*fsm, tok, {"{", "\"", "score", "\":", "42,", "\"", "tags", "\":", "[", "\"", "a", "\\", "n", "\"", ",", "true", "]", ",", "\"", "ok", "\":", "123}"});

  // Leading-zero check on separate fresh FSM
  const std::vector<std::int32_t> content_prefill = {*tok.token_id(CONTENT_TEXT)};
  auto fsm2 = make_json_fsm(vocab, spec, content_prefill);
  feed_pieces(*fsm2, tok, {"{", "\"", "a", "\":"});
  assert(!is_allowed(*fsm2, tok_id(tok, "01")));  // RFC 8259 forbids leading zeros!
  assert(is_allowed(*fsm2, tok_id(tok, "0")));

  // Top-level object complete on fsm: non-ws disallowed, <|end_message|> allowed!
  assert(!is_allowed(*fsm, tok_id(tok, "{")));
  assert(!is_allowed(*fsm, tok_id(tok, "You")));
  assert(is_allowed(*fsm, *tok.token_id(END_MESSAGE)));
  fsm->advance(*tok.token_id(END_MESSAGE));

  // After <|end_message|>: only stop/EOS tokens allowed
  assert(!is_allowed(*fsm, tok_id(tok, "{")));
  assert(is_allowed(*fsm, *tok.token_id(END_SAMPLING)));
  std::cout << "[PASS] test_thinking_channel_unconstrained_then_json_object\n";
}

void test_strict_json_schema_required_enum_integer_and_no_duplicates() {
  const Tokenizer tok = Tokenizer::from_json(make_test_tokenizer_json());
  const std::vector<std::int32_t> stops = {*tok.token_id(END_SAMPLING), *tok.token_id(ENDOFTEXT)};
  auto vocab = VocabByteIndex::build(tok, stops);

  const Json schema = Json::parse(R"({
    "type": "object",
    "properties": {
      "status": {"type": "string", "enum": ["ok", "fail"]},
      "score": {"type": "integer"},
      "tags": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["status", "score"],
    "additionalProperties": false
  })");

  JsonFormatSpec spec{JsonFormatMode::JsonSchema, schema};
  const std::vector<std::int32_t> content_prefill = {*tok.token_id(CONTENT_TEXT)};
  auto fsm = make_json_fsm(vocab, spec, content_prefill);
  assert(fsm->is_active());

  feed_pieces(*fsm, tok, {"{"});
  // Cannot close '}' while required keys are missing!
  assert(!is_allowed(*fsm, tok_id(tok, "}")));
  feed_pieces(*fsm, tok, {"\""});
  // Only keys matching unused schema properties ("status", "score", "tags") are allowed!
  assert(is_allowed(*fsm, tok_id(tok, "status")));
  assert(is_allowed(*fsm, tok_id(tok, "score")));
  assert(!is_allowed(*fsm, tok_id(tok, "hello")));  // disallowed extra property!

  feed_pieces(*fsm, tok, {"status", "\":"});
  // Enum allows ONLY "\"ok\"" or "\"fail\"" (not "\"hello\"" or number!)
  assert(is_allowed(*fsm, tok_id(tok, "\"ok\"")));
  assert(!is_allowed(*fsm, tok_id(tok, "true")));
  assert(!is_allowed(*fsm, tok_id(tok, "0")));
  feed_pieces(*fsm, tok, {"\"ok\""});

  // Still missing required "score": '}' MUST be disallowed, ',' MUST be allowed!
  assert(!is_allowed(*fsm, tok_id(tok, "}")));
  assert(is_allowed(*fsm, tok_id(tok, ",")));
  feed_pieces(*fsm, tok, {",", "\""});

  // "status" already used -> duplicate key MUST be disallowed!
  assert(!is_allowed(*fsm, tok_id(tok, "status")));
  assert(is_allowed(*fsm, tok_id(tok, "score")));
  feed_pieces(*fsm, tok, {"score", "\":", "4"});
  // Integer forbids decimal point '.' and exponent 'e'!
  assert(!is_allowed(*fsm, tok_id(tok, ".")));
  assert(!is_allowed(*fsm, tok_id(tok, "e")));
  assert(is_allowed(*fsm, tok_id(tok, "2")));
  feed_pieces(*fsm, tok, {"2"});

  // Now both required keys ("status", "score") are satisfied: '}' AND ',' (for optional "tags") both allowed!
  assert(is_allowed(*fsm, tok_id(tok, "}")));
  assert(is_allowed(*fsm, tok_id(tok, ",")));
  feed_pieces(*fsm, tok, {",", "\"", "tags", "\":", "[", "\"", "hello", "\"", "]", "}"});

  // Complete -> end_message allowed, extra '{' disallowed.
  assert(!is_allowed(*fsm, tok_id(tok, "{")));
  assert(is_allowed(*fsm, *tok.token_id(END_MESSAGE)));
  std::cout << "[PASS] test_strict_json_schema_required_enum_integer_and_no_duplicates\n";
}

void test_ref_and_nullable_union() {
  const Tokenizer tok = Tokenizer::from_json(make_test_tokenizer_json());
  const std::vector<std::int32_t> stops = {*tok.token_id(END_SAMPLING)};
  auto vocab = VocabByteIndex::build(tok, stops);

  const Json schema = Json::parse(R"({
    "$defs": {
      "Note": {"type": ["string", "null"]}
    },
    "type": "object",
    "properties": {
      "ok": {"$ref": "#/$defs/Note"}
    },
    "required": ["ok"],
    "additionalProperties": false
  })");
  JsonFormatSpec spec{JsonFormatMode::JsonSchema, schema};
  const std::vector<std::int32_t> content_prefill = {*tok.token_id(CONTENT_TEXT)};
  auto fsm_null = make_json_fsm(vocab, spec, content_prefill);
  feed_pieces(*fsm_null, tok, {"{", "\"", "ok", "\":"});
  assert(is_allowed(*fsm_null, tok_id(tok, "null")));
  assert(is_allowed(*fsm_null, tok_id(tok, "\"")));
  assert(!is_allowed(*fsm_null, tok_id(tok, "0")));
  feed_pieces(*fsm_null, tok, {"null", "}"});
  assert(is_allowed(*fsm_null, *tok.token_id(END_MESSAGE)));
  std::cout << "[PASS] test_ref_and_nullable_union\n";
}

}  // namespace

int main() {
  test_thinking_channel_unconstrained_then_json_object();
  test_strict_json_schema_required_enum_integer_and_no_duplicates();
  test_ref_and_nullable_union();
  std::cout << "ALL test_json_fsm TESTS PASSED\n";
  return 0;
}

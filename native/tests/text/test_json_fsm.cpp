// Unit tests for token-level RFC 8259 JSON and JSON-Schema FSM constrained decoding (#830).
#include "text/chat.h"
#include "text/json_fsm.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Active runtime check (never compiled out by -DNDEBUG).
#define CHECK(cond)                                                                           \
  do {                                                                                        \
    if (!(cond)) {                                                                            \
      std::cerr << "CHECK FAILED at " << __FILE__ << ":" << __LINE__ << ": " #cond << "\n";  \
      std::exit(1);                                                                           \
    }                                                                                         \
  } while (0)

namespace {

using Json = nlohmann::ordered_json;
using namespace ingine::text;

// Minimal synthetic byte-level BPE tokenizer JSON (valid for Tokenizer::from_json) covering all 256
// single-byte tokens + multi-byte JSON chunks + Inkling-2 special tokens.
std::string make_test_tokenizer_json() {
  Json vocab = Json::object();
  int next_id = 0;
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
    CHECK(is_allowed(fsm, id));
    fsm.advance(id);
  }
}

bool accepts_byte_payload(const Tokenizer& tok, const std::shared_ptr<const VocabByteIndex>& vocab,
                          const std::vector<std::int32_t>& prompt, std::int32_t end_id, const Json& schema,
                          std::string_view payload) {
  const auto mode = schema.is_null() ? JsonFormatMode::JsonObject : JsonFormatMode::JsonSchema;
  auto masker = make_json_fsm(vocab, {mode, schema}, prompt);
  for (char byte : payload) {
    const std::int32_t id = tok_id(tok, std::string_view(&byte, 1));
    if (!is_allowed(*masker, id)) return false;
    masker->advance(id);
  }
  return is_allowed(*masker, end_id);
}

bool accepts_raw_schema_payload(const Tokenizer& tok, const std::shared_ptr<const VocabByteIndex>& vocab,
                                const std::vector<std::int32_t>& prompt, std::int32_t end_id,
                                std::string_view raw_schema, std::string_view payload) {
  JsonFormatSpec spec;
  spec.mode = JsonFormatMode::JsonSchema;
  spec.raw_schema_json = std::string(raw_schema);
  auto masker = make_json_fsm(vocab, spec, prompt);
  for (char byte : payload) {
    const std::int32_t id = tok_id(tok, std::string_view(&byte, 1));
    if (!is_allowed(*masker, id)) return false;
    masker->advance(id);
  }
  return is_allowed(*masker, end_id);
}

bool schema_is_rejected(const std::shared_ptr<const VocabByteIndex>& vocab, const std::vector<std::int32_t>& prompt,
                        const Json& schema) {
  try {
    (void)make_json_fsm(vocab, {JsonFormatMode::JsonSchema, schema}, prompt);
    return false;
  } catch (const std::invalid_argument&) {
    return true;
  }
}

void test_thinking_channel_unconstrained_then_json_object() {
  const Tokenizer tok = Tokenizer::from_json(make_test_tokenizer_json());
  const std::vector<std::int32_t> stops = {
      *tok.token_id(END_SAMPLING), *tok.token_id(ENDOFTEXT), *tok.token_id(MESSAGE_USER)};
  auto vocab = VocabByteIndex::build(tok, stops);

  const std::vector<std::int32_t> prefill = {*tok.token_id(MESSAGE_MODEL), *tok.token_id(CONTENT_THINKING)};
  JsonFormatSpec spec{JsonFormatMode::JsonObject, Json()};
  auto fsm = make_json_fsm(vocab, spec, prefill);
  CHECK(fsm != nullptr);
  CHECK(!fsm->is_active());

  fsm->advance(tok_id(tok, "You"));
  CHECK(!fsm->is_active());
  fsm->advance(*tok.token_id(END_MESSAGE));
  fsm->advance(*tok.token_id(MESSAGE_MODEL));
  CHECK(!fsm->is_active());
  fsm->advance(*tok.token_id(CONTENT_TEXT));
  CHECK(fsm->is_active());

  CHECK(is_allowed(*fsm, tok_id(tok, "{")));
  CHECK(!is_allowed(*fsm, tok_id(tok, "```json")));
  CHECK(!is_allowed(*fsm, tok_id(tok, "You")));
  CHECK(!is_allowed(*fsm, tok_id(tok, "[")));
  CHECK(!is_allowed(*fsm, tok_id(tok, "0")));

  feed_pieces(*fsm, tok, {"{", "\"", "score", "\":", "42,", "\"", "tags", "\":", "[", "\"", "a", "\\", "n", "\"", ",", "true", "]", ",", "\"", "ok", "\":", "123}"});

  const std::vector<std::int32_t> content_prefill = {*tok.token_id(CONTENT_TEXT)};
  auto fsm2 = make_json_fsm(vocab, spec, content_prefill);
  feed_pieces(*fsm2, tok, {"{", "\"", "a", "\":"});
  CHECK(!is_allowed(*fsm2, tok_id(tok, "01")));
  CHECK(is_allowed(*fsm2, tok_id(tok, "0")));

  CHECK(!is_allowed(*fsm, tok_id(tok, "{")));
  CHECK(!is_allowed(*fsm, tok_id(tok, "You")));
  CHECK(is_allowed(*fsm, *tok.token_id(END_MESSAGE)));
  fsm->advance(*tok.token_id(END_MESSAGE));

  CHECK(!is_allowed(*fsm, tok_id(tok, "{")));
  CHECK(is_allowed(*fsm, *tok.token_id(END_SAMPLING)));
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
  CHECK(fsm->is_active());

  feed_pieces(*fsm, tok, {"{"});
  CHECK(!is_allowed(*fsm, tok_id(tok, "}")));
  feed_pieces(*fsm, tok, {"\""});
  CHECK(is_allowed(*fsm, tok_id(tok, "status")));
  CHECK(is_allowed(*fsm, tok_id(tok, "score")));
  CHECK(!is_allowed(*fsm, tok_id(tok, "hello")));

  feed_pieces(*fsm, tok, {"status", "\":"});
  CHECK(is_allowed(*fsm, tok_id(tok, "\"ok\"")));
  CHECK(!is_allowed(*fsm, tok_id(tok, "true")));
  CHECK(!is_allowed(*fsm, tok_id(tok, "0")));
  feed_pieces(*fsm, tok, {"\"ok\""});

  CHECK(!is_allowed(*fsm, tok_id(tok, "}")));
  CHECK(is_allowed(*fsm, tok_id(tok, ",")));
  feed_pieces(*fsm, tok, {",", "\""});

  CHECK(!is_allowed(*fsm, tok_id(tok, "status")));
  CHECK(is_allowed(*fsm, tok_id(tok, "score")));
  feed_pieces(*fsm, tok, {"score", "\":", "4"});
  CHECK(!is_allowed(*fsm, tok_id(tok, ".")));
  CHECK(!is_allowed(*fsm, tok_id(tok, "e")));
  CHECK(is_allowed(*fsm, tok_id(tok, "2")));
  feed_pieces(*fsm, tok, {"2"});

  CHECK(is_allowed(*fsm, tok_id(tok, "}")));
  CHECK(is_allowed(*fsm, tok_id(tok, ",")));
  feed_pieces(*fsm, tok, {",", "\"", "tags", "\":", "[", "\"", "hello", "\"", "]", "}"});

  CHECK(!is_allowed(*fsm, tok_id(tok, "{")));
  CHECK(is_allowed(*fsm, *tok.token_id(END_MESSAGE)));
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
  CHECK(is_allowed(*fsm_null, tok_id(tok, "null")));
  CHECK(is_allowed(*fsm_null, tok_id(tok, "\"")));
  CHECK(!is_allowed(*fsm_null, tok_id(tok, "0")));
  feed_pieces(*fsm_null, tok, {"null", "}"});
  CHECK(is_allowed(*fsm_null, *tok.token_id(END_MESSAGE)));
  std::cout << "[PASS] test_ref_and_nullable_union\n";
}

void test_oneof_exact_semantics() {
  const Tokenizer tok = Tokenizer::from_json(make_test_tokenizer_json());
  const std::int32_t end = *tok.token_id(END_SAMPLING);
  const std::vector<std::int32_t> stops{end}, prompt{*tok.token_id(CONTENT_TEXT)};
  auto vocab = VocabByteIndex::build(tok, stops);

  const Json overlap = Json::parse(R"({"oneOf":[{"type":"integer"},{"type":"number"}]})");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, overlap, "3"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, overlap, "3 "));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, overlap, "3.0"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, overlap, "3e0"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, overlap, "3.5"));

  const Json literals = Json::parse(R"({"oneOf":[{"type":"string"},{"const":"a"}]})");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, literals, R"("\u0061")"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, literals, R"("\u0062")"));

  const Json bools = Json::parse(R"({"oneOf":[{"type":"boolean"},{"const":true}]})");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, bools, "tru"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, bools, "true"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, bools, "false"));

  const Json reordered = Json::parse(R"({"oneOf":[{"type":"object"},{"const":{"a":1,"b":2}}]})");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, reordered, R"({"b":2,"a":1})"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, reordered, R"({"b":1,"a":2})"));

  const Json wrapped = Json::parse(R"({
    "type": "object",
    "properties": {"v": {"oneOf": [{"type": "integer"}, {"type": "number"}]}},
    "required": ["v"],
    "additionalProperties": false
  })");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, wrapped, R"({"v":3})"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, wrapped, R"({"v":3.0})"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, wrapped, R"({"v":3.5})"));

  const Json disjoint = Json::parse(R"({"oneOf":[{"type":"boolean"},{"type":"null"}]})");
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, disjoint, "null"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, disjoint, "true"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, disjoint, "1"));

  const Json obj_oneof = Json::parse(R"({
    "oneOf": [
      {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
      {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]}
    ]
  })");
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, obj_oneof, R"({"a":1})"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, obj_oneof, R"({"b":2})"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, obj_oneof, R"({"a":1,"b":2})"));
  std::cout << "[PASS] test_oneof_exact_semantics\n";
}

void test_many_properties_and_escaped_open_keys() {
  const Tokenizer tok = Tokenizer::from_json(make_test_tokenizer_json());
  const std::int32_t end = *tok.token_id(END_SAMPLING);
  const std::vector<std::int32_t> stops{end}, prompt{*tok.token_id(CONTENT_TEXT)};
  auto vocab = VocabByteIndex::build(tok, stops);

  // >64 properties: 65th property ("k64") is required and deduplicated
  Json many = Json::parse(R"({"type":"object","properties":{},"required":["k64"],"additionalProperties":false})");
  for (int i = 0; i < 70; ++i) many["properties"]["k" + std::to_string(i)] = {{"type", "integer"}};
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, many, "{}"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, many, R"({"k64":7})"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, many, R"({"k64":7,"k64":8})"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, many, R"({"k0":1,"k64":7,"k69":9})"));

  // Escaped key spelling in open object must preserve declared property type and reject duplicates
  const Json note = Json::parse(R"({"type":"object","properties":{"a":{"type":"integer"}}})");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, note, R"({"\u0061":false})"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, note, R"({"\u0061":1})"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, note, R"({"a":1,"\u0061":2})"));

  // Open object duplicate undeclared keys rejected (plain and escaped)
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, nullptr, R"({"v":1,"v":2})"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, nullptr, R"({"v":1,"\u0076":2})"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, nullptr, R"({"v":1,"w":2})"));
  std::cout << "[PASS] test_many_properties_and_escaped_open_keys\n";
}

void test_utf8_and_surrogates() {
  const Tokenizer tok = Tokenizer::from_json(make_test_tokenizer_json());
  const std::int32_t end = *tok.token_id(END_SAMPLING);
  const std::vector<std::int32_t> stops{end}, prompt{*tok.token_id(CONTENT_TEXT)};
  auto vocab = VocabByteIndex::build(tok, stops);

  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, nullptr, std::string("{\"v\":\"\x80\"}")));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, nullptr, std::string("{\"v\":\"\xe2\"}")));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, nullptr, std::string("{\"v\":\"\xc0\xaf\"}")));  // overlong
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, nullptr, std::string("{\"v\":\"\xed\xa0\x80\"}")));  // UTF-8 surrogate
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, nullptr, std::string("{\"\x80\":1}")));  // invalid UTF-8 in key
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, nullptr, std::string("{\"v\":\"\xe2\x82\xac\"}")));  // €
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, nullptr, std::string("{\"v\":\"\xf0\x9f\x98\x80\"}")));  // 😀
  // UTF-16 surrogate pair in \uXXXX
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, nullptr, R"({"v":"\uD83D\uDE00"})"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, nullptr, R"({"v":"\uD83D"})"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, nullptr, R"({"v":"\uDE00"})"));
  std::cout << "[PASS] test_utf8_and_surrogates\n";
}

void test_assertions_conjunctions_and_rejections() {
  const Tokenizer tok = Tokenizer::from_json(make_test_tokenizer_json());
  const std::int32_t end = *tok.token_id(END_SAMPLING);
  const std::vector<std::int32_t> stops{end}, prompt{*tok.token_id(CONTENT_TEXT)};
  auto vocab = VocabByteIndex::build(tok, stops);

  // Numeric bounds
  const Json min2 = Json::parse(R"({"type":"integer","minimum":2})");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, min2, "1"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, min2, "-5"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, min2, "2"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, min2, "10"));

  const std::string_view range_num = R"({"type":"number","minimum":0.0,"maximum":50.0})";
  CHECK(accepts_raw_schema_payload(tok, vocab, prompt, end, range_num, "0"));
  CHECK(accepts_raw_schema_payload(tok, vocab, prompt, end, range_num, "49.5"));
  CHECK(!accepts_raw_schema_payload(tok, vocab, prompt, end, range_num, "-1"));
  CHECK(!accepts_raw_schema_payload(tok, vocab, prompt, end, range_num, "50.5"));

  // String length (in Unicode codepoints) & Array item bounds
  const Json str_len = Json::parse(R"({"type":"string","minLength":2,"maxLength":3})");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, str_len, R"("a")"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, str_len, R"("ab")"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, str_len, std::string("\"a\xe2\x82\xac\"")));  // 2 codepoints
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, str_len, R"("abcd")"));

  const Json arr_len = Json::parse(R"({"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3})");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, arr_len, "[]"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, arr_len, "[1]"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, arr_len, "[1,2]"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, arr_len, "[1,2,3]"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, arr_len, "[1,2,3,4]"));

  // Type + enum conjunction
  const Json str_enum = Json::parse(R"({"type":"string","enum":[42,"ok"]})");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, str_enum, "42"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, str_enum, R"("ok")"));

  // Exact >2^53 64-bit signed & unsigned integer bounds + exact arbitrary-precision fractional bounds
  const Json large_min = Json::parse(R"({"type":"integer","minimum":9007199254740993})");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, large_min, "9007199254740992"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, large_min, "9007199254740993"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, large_min, "9007199254740994"));

  const Json min_uint = Json::parse(R"({"type":"integer","minimum":18446744073709551615})");
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, min_uint, "0"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, min_uint, "18446744073709551615"));

  const Json max_uint = Json::parse(R"({"type":"integer","maximum":9223372036854775808})");
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, max_uint, "0"));
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, max_uint, "9223372036854775808"));
  CHECK(!accepts_byte_payload(tok, vocab, prompt, end, max_uint, "9223372036854775809"));

  // Exact arbitrary-precision wire-decimal bounds, const, and nested oneOf (via raw_schema_json)
  CHECK(!accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","minimum":0.5})", "0.49999999999999999"));
  CHECK(accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","minimum":0.5})", "0.5"));
  CHECK(!accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","maximum":0.5})", "0.50000000000000001"));
  CHECK(accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","maximum":0.5})", "0.5"));
  CHECK(!accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","exclusiveMinimum":0.5})", "0.5"));
  CHECK(accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","exclusiveMinimum":0.5})", "0.50000000000000001"));

  CHECK(!accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","minimum":0.50000000000000001})", "0.5"));
  CHECK(accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","minimum":0.50000000000000001})", "0.50000000000000001"));
  CHECK(!accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","maximum":0.49999999999999999})", "0.5"));
  CHECK(accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","maximum":0.49999999999999999})", "0.49999999999999999"));
  CHECK(!accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"const":0.50000000000000001})", "0.5"));
  CHECK(accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"const":0.50000000000000001})", "0.50000000000000001"));
  CHECK(!accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","minimum":9007199254740993.0})", "9007199254740992"));
  CHECK(accepts_raw_schema_payload(tok, vocab, prompt, end, R"({"type":"number","minimum":9007199254740993.0})", "9007199254740993"));

  const std::string_view nested_oneof =
      R"({"oneOf":[{"type":"object"},{"type":"object","required":["x"],"properties":{"x":{"type":"number","exclusiveMinimum":0.5}}}]})";
  CHECK(!accepts_raw_schema_payload(tok, vocab, prompt, end, nested_oneof, R"({"x":0.50000000000000001})"));
  CHECK(accepts_raw_schema_payload(tok, vocab, prompt, end, nested_oneof, R"({"x":0.5})"));

  const Json typed_oneof = Json::parse(R"({"oneOf":[{"type":"integer"},{"type":"number"}]})");
  CHECK(accepts_byte_payload(tok, vocab, prompt, end, typed_oneof, "1.0000000000000000001"));

  // Pre-parsed binary64 float constraints WITHOUT raw_schema_json must be explicitly rejected (no silent rounding!)
  CHECK(schema_is_rejected(vocab, prompt, Json::parse(R"({"type":"number","minimum":0.50000000000000001})")));
  CHECK(schema_is_rejected(vocab, prompt, Json::parse(R"({"type":"number","maximum":0.49999999999999999})")));
  CHECK(schema_is_rejected(vocab, prompt, Json::parse(R"({"const":0.50000000000000001})")));
  CHECK(schema_is_rejected(vocab, prompt, Json::parse(R"({"type":"number","minimum":9007199254740993.0})")));
  CHECK(schema_is_rejected(vocab, prompt, Json::parse(R"({"type":"integer","minimum":1e30})")));

  // Explicit rejection of unresolved $ref, unsupported keywords, and unsatisfiable schemas
  CHECK(schema_is_rejected(vocab, prompt, Json::parse(R"({"$ref":"#/$defs/absent"})")));
  CHECK(schema_is_rejected(vocab, prompt, Json::parse(R"({"type":"string","pattern":"^[a-z]+$"})")));
  CHECK(schema_is_rejected(vocab, prompt, Json::parse(R"({"allOf":[{"type":"string"}]})")));
  CHECK(schema_is_rejected(vocab, prompt, Json::parse(R"({"type":"integer","minimum":10,"maximum":5})")));
  CHECK(schema_is_rejected(vocab, prompt, Json::parse(R"({"type":"string","enum":[42]})")));
  CHECK(schema_is_rejected(vocab, prompt, Json::parse(R"({"type":"integer","const":-1,"enum":[18446744073709551615]})")));
  std::cout << "[PASS] test_assertions_conjunctions_and_rejections\n";
}

}  // namespace

int main() {
  test_thinking_channel_unconstrained_then_json_object();
  test_strict_json_schema_required_enum_integer_and_no_duplicates();
  test_ref_and_nullable_union();
  test_oneof_exact_semantics();
  test_many_properties_and_escaped_open_keys();
  test_utf8_and_surrogates();
  test_assertions_conjunctions_and_rejections();
  std::cout << "ALL test_json_fsm TESTS PASSED\n";
  return 0;
}

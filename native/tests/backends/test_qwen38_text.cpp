#include "text/chat.h"
#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

namespace {

using namespace ingine::text;

std::string read_text(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  CHECK(in.good());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void test_tokenizer_goldens(const Tokenizer& tok) {
  CHECK(tok.vocab_size() <= 248320);

  struct Case {
    const char* name;
    const char* text;
    bool special;
    std::vector<std::int32_t> expected;
  };
  const std::vector<Case> cases = {
      {
          "ascii_code",
          "def fib(n: int) -> int:\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
          false,
          {727, 15336, 1393, 25, 514, 8, 1411, 514, 25, 198, 262, 460, 307, 413, 307, 361,
           220, 17, 745, 15336, 1393, 12, 16, 8, 478, 15336, 1393, 12, 17, 8, 198},
      },
      {
          "cjk_emoji",
          "你好世界！Qwen3.8 🧪🔬 café Café",
          false,
          {109266, 96748, 6115, 48, 16451, 18, 13, 23, 10838, 100, 103, 9008, 242, 105, 50203, 63457},
      },
      {
          "nfc_decomposed",
          "Cafe\xcc\x81 e\xcc\x81 \xc3\xa9",
          false,
          {34, 2492, 933, 3825, 3825},
      },
      {
          "digits",
          "2026 123456 9 007",
          false,
          {17, 15, 17, 21, 220, 16, 17, 18, 19, 20, 21, 220, 24, 220, 15, 15, 22},
      },
      {
          "contractions",
          "I'm you're They'LL can't he'd she's we've",
          false,
          {40, 2688, 488, 2224, 2302, 6, 3950, 628, 914, 551, 4035, 1292, 579, 567, 2908},
      },
      {
          "punct_ws",
          "Hello, world!  \n\n  foo/bar\r\n  ",
          false,
          {9419, 11, 1814, 0, 18050, 220, 14785, 47849, 317, 256},
      },
      {
          "special",
          "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\nok</think>\n",
          true,
          {248045, 846, 198, 5834, 248046, 198, 248045, 74455, 198, 248068, 198, 547, 248069, 198},
      },
  };

  for (const auto& c : cases) {
    const auto got = tok.encode(c.text, c.special);
    if (got != c.expected) {
      std::fprintf(stderr, "tokenizer case '%s' mismatch: got len=%zu vs expected=%zu\n", c.name, got.size(),
                   c.expected.size());
      std::exit(1);
    }
  }

  const std::string utf8_rt = "Hello, 世界! 2026 🧪\n";
  const auto rt_ids = tok.encode(utf8_rt, false);
  CHECK(tok.decode(rt_ids) == utf8_rt);
}

void test_chat_render_and_stream_parser(const ChatFormat& fmt) {
  CHECK(fmt.kind() == ChatFormatKind::Qwen3_5);
  CHECK(fmt.content_invoke_tool_json_id() == -1);
  CHECK(fmt.content_thinking_id() == -1);
  CHECK((fmt.stop_ids() == std::vector<std::int32_t>{248044, 248046}));
  CHECK(fmt.no_thinking_prefill().empty());

  const nlohmann::ordered_json msgs = nlohmann::ordered_json::array({{{"role", "user"}, {"content", "What is 2+2?"}}});

  // 1. think_default
  {
    ChatOptions opts;
    opts.enable_thinking = true;
    const auto r = fmt.render(msgs, nullptr, opts);
    const std::vector<std::int32_t> expected = {
        248045, 8678,  198,   24342, 286,   4879,  369,   716,   310,   830,   11553, 13,
        5044,   1683,  15060, 1472,  279,   3274,  11,    9307,  1328,  30800, 11,    2814,
        47675,  25605, 11,    321,   60445, 55404, 11,    27224, 11,    321,   30246, 303,
        279,    1534,  4087,  13,    248046, 198,  248045, 846,  198,   3710,  369,   220,
        17,     10,    17,    30,    248046, 198,  248045, 74455, 198,  248068, 198};
    CHECK(r.ids == expected);
    CHECK((r.prefill_ids == std::vector<std::int32_t>{248068}));
  }

  // 2. think_medium
  {
    ChatOptions opts;
    opts.enable_thinking = true;
    opts.reasoning_effort = "medium";
    const auto r = fmt.render(msgs, nullptr, opts);
    const std::vector<std::int32_t> expected = {248045, 846, 198, 3710, 369, 220, 17, 10, 17,
                                                30, 248046, 198, 248045, 74455, 198, 248068, 198};
    CHECK(r.ids == expected);
    CHECK(r.text == "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n");
  }

  // 3. nothink
  {
    ChatOptions opts;
    opts.enable_thinking = false;
    const auto r = fmt.render(msgs, nullptr, opts);
    const std::vector<std::int32_t> expected = {248045, 846, 198, 3710, 369, 220, 17, 10, 17, 30,
                                                248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271};
    CHECK(r.ids == expected);
    CHECK(r.prefill_ids.empty());
    CHECK(r.text == "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
  }

  // 4. tool_nothink + tool history
  const nlohmann::ordered_json tools = nlohmann::ordered_json::array({{{
      "type", "function"},
      {"function",
       {{"name", "get_weather"},
        {"description", "Get weather for a city"},
        {"parameters",
         {{"type", "object"},
          {"properties", {{"city", {{"type", "string"}}}}},
          {"required", nlohmann::ordered_json::array({"city"})}}}}}}});
  {
    ChatOptions opts;
    opts.enable_thinking = false;
    const auto r = fmt.render(msgs, tools, opts);
    CHECK(r.ids.size() == 276);
    CHECK(r.text.find("<function=example_function_name>") != std::string::npos);
    CHECK(r.text.find("\"name\": \"get_weather\"") != std::string::npos);
  }
  {
    const nlohmann::ordered_json hist = nlohmann::ordered_json::array({
        {{"role", "system"}, {"content", "Be concise."}},
        {{"role", "user"}, {"content", "Check weather in Paris"}},
        {{"role", "assistant"},
         {"content", ""},
         {"tool_calls",
          nlohmann::ordered_json::array(
              {{{"id", "c1"}, {"type", "function"}, {"function", {{"name", "get_weather"}, {"arguments", {{"city", "Paris"}}}}}}})}},
        {{"role", "tool"}, {"content", "{\"temp_c\": 18}"}},
    });
    ChatOptions opts;
    opts.enable_thinking = false;
    const auto r = fmt.render(hist, tools, opts);
    CHECK(r.ids.size() == 328);
    CHECK(r.text.find("<tool_response>\n{\"temp_c\": 18}\n</tool_response>") != std::string::npos);
  }

  // 5. StreamParser thinking + answer (starting from prefill <think>)
  {
    const auto think_id = fmt.tokenizer().token_id("<think>");
    CHECK(think_id.has_value());
    const std::vector<std::int32_t> prefill = {*think_id};
    const auto gen = fmt.tokenizer().encode("let me check\n</think>\n\nThe answer is 4.<|im_end|>", true);
    auto parser = fmt.parser(prefill);
    for (std::int32_t t : gen) parser.feed(t);
    parser.finish();
    const auto comp = parser.result();
    CHECK(comp.reasoning_content == "let me check\n");
    CHECK(comp.content == "The answer is 4.");
    CHECK(comp.tool_calls.empty());
    CHECK(comp.finish_reason == "stop");
  }

  // 6. StreamParser XML tool call
  {
    const auto gen = fmt.tokenizer().encode(
        "<think>\nneed weather\n</think>\n\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n"
        "</function>\n</tool_call><|im_end|>",
        true);
    auto parser = fmt.parser({}, {}, [] { return std::string("call_fixed_0"); });
    for (std::int32_t t : gen) parser.feed(t);
    parser.finish();
    const auto comp = parser.result();
    CHECK(comp.tool_calls.size() == 1);
    CHECK(comp.tool_calls[0].id == "call_fixed_0");
    CHECK(comp.tool_calls[0].name == "get_weather");
    const auto args = nlohmann::json::parse(comp.tool_calls[0].arguments);
    CHECK(args.at("city") == "Paris");
    CHECK(args.at("days") == 3);
    CHECK(comp.finish_reason == "tool_calls");
  }
}

void test_release_reference_prompts(const Tokenizer& tok, const ChatFormat& fmt) {
  const std::filesystem::path ref_dir("/mnt/storage/flare/release-references");
  const std::filesystem::path ref_json = ref_dir / "reference.json";
  const std::filesystem::path lref_json = ref_dir / "long-code-reference.json";
  const std::filesystem::path prompts_jsonl = ref_dir / "prompts.jsonl";
  if (!std::filesystem::is_regular_file(ref_json) || !std::filesystem::is_regular_file(lref_json) ||
      !std::filesystem::is_regular_file(prompts_jsonl)) {
    return;
  }

  // Verify reference.json prompt_text -> prompt_ids exact match across all 3 short prompts.
  {
    const auto j = nlohmann::ordered_json::parse(read_text(ref_json));
    const auto& prompts = j.at("prompts");
    CHECK(prompts.size() == 3);
    for (const auto& p : prompts) {
      const std::string text = p.at("prompt_text").get<std::string>();
      const auto expected = p.at("prompt_ids").get<std::vector<std::int32_t>>();
      const auto got = tok.encode(text, false);
      CHECK(got == expected);
    }
    std::printf("  reference.json exact token parity: 3/3 PASS\n");
  }

  // Verify prompts.jsonl -> ChatFormat::render(enable_thinking=false) == long-code-reference.json prompt_ids.
  {
    std::unordered_map<std::string, nlohmann::ordered_json> msg_by_id;
    std::ifstream in(prompts_jsonl, std::ios::binary);
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      const auto obj = nlohmann::ordered_json::parse(line);
      msg_by_id.emplace(obj.at("id").get<std::string>(), obj);
    }
    const auto lref = nlohmann::ordered_json::parse(read_text(lref_json));
    const auto& prompts = lref.at("prompts");
    CHECK(prompts.size() == 6);
    for (const auto& p : prompts) {
      const std::string id = p.at("id").get<std::string>();
      const auto& src = msg_by_id.at(id);
      ChatOptions opts;
      opts.enable_thinking = src.at("thinking");
      const auto rendered = fmt.render(src.at("messages"), nullptr, opts);
      const auto expected = p.at("prompt_ids").get<std::vector<std::int32_t>>();
      CHECK(rendered.ids == expected);
    }
    std::printf("  long-code-reference.json ChatFormat+Tokenizer exact parity: 6/6 PASS\n");
  }
}

}  // namespace

int main() {
  const std::filesystem::path model_dir("/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f");
  if (!std::filesystem::is_directory(model_dir)) {
    std::printf("skip: Qwen3.8 checkpoint not present at %s\n", model_dir.c_str());
    return 77;
  }
  const auto tok = Tokenizer::load(model_dir);
  const auto fmt = ChatFormat::load(model_dir);
  test_tokenizer_goldens(tok);
  test_chat_render_and_stream_parser(fmt);
  test_release_reference_prompts(tok, fmt);
  std::printf("qwen38_text: PASS\n");
  return 0;
}

// Chat render goldens: chat.ChatFormat.render ids/text/prefill or ChatError message and param; concurrent renders
// from several threads must agree with the serial ones (ChatFormat is shared by request threads).
#include "text/chat.h"
#include "text_test.h"

#include <algorithm>
#include <atomic>
#include <thread>

using namespace ingine::text;
using Json = nlohmann::ordered_json;

int main() {
  const auto dir = test::model_dir();
  if (dir.empty()) return test::kSkip;
  const Json golden = test::load_golden("chat.json");
  const ChatFormat fmt = ChatFormat::load(dir);
  test::Checker c;
  for (const Json& req : golden) {
    const std::string note = req["note"].get<std::string>();
    ChatOptions options;
    options.reasoning_effort = req["reasoning_effort"];
    options.enable_thinking = req["enable_thinking"];
    options.template_kwargs = req["kwargs"];
    try {
      const RenderedPrompt p = fmt.render(req["messages"], req["tools"], options);
      if (!c.expect(req.contains("result"), note + ": rendered, expected error " + req.value("error", Json()).dump())) {
        continue;
      }
      const Json& r = req["result"];
      c.expect(p.text == r["text"].get<std::string>(),
               note + ": text\n  got  " + test::show(p.text) + "\n  want " + test::show(r["text"].get<std::string>()));
      c.expect(p.ids == r["ids"].get<std::vector<int32_t>>(), note + ": ids");
      c.expect(p.prefill_ids == r["prefill_ids"].get<std::vector<int32_t>>(), note + ": prefill_ids");
    } catch (const ChatError& e) {
      if (!c.expect(req.contains("error"), note + ": ChatError " + test::show(e.what()) + " param " + e.param)) continue;
      const Json& err = req["error"];
      c.expect(e.what() == err["message"].get<std::string>(),
               note + ": message\n  got  " + test::show(e.what()) + "\n  want " +
                   test::show(err["message"].get<std::string>()));
      c.expect(e.param == err["param"].get<std::string>(), note + ": param " + e.param);
    }
  }
  // Image parts (not in the goldens of the text-only Python engine): the template renders each as its own
  // message segment with one placeholder token, in the order of the parts, for user and tool messages only.
  {
    const Json image = {{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,AAAA"}}}};
    const Json messages = Json::array(
        {{{"role", "user"}, {"content", Json::array({{{"type", "text"}, {"text", "Compare"}}, image,
                                                     {{"type", "input_image"}, {"image_url", "x"}},
                                                     {{"type", "image"}, {"image", "y"}}})}}});
    const RenderedPrompt p = fmt.render(messages, Json(), ChatOptions{});
    const std::string segment = "<|message_user|><|content_image|><|unused_200054|><|end_message|>";
    c.expect(p.text.find("<|message_user|><|content_text|>Compare<|end_message|>" + segment + segment + segment) !=
                 std::string::npos,
             "image parts: text " + test::show(p.text));
    c.expect(std::count(p.ids.begin(), p.ids.end(), 200054) == 3, "image parts: three placeholder ids");
    for (std::size_t i = 0; i < p.ids.size(); ++i) {
      if (p.ids[i] == 200054) {
        c.expect(i > 0 && p.ids[i - 1] == 200005 && i + 1 < p.ids.size() && p.ids[i + 1] == 200010,
                 "image parts: <|content_image|> placeholder <|end_message|>");
      }
    }
    bool rejected = false;
    try {
      (void)fmt.render(Json::array({{{"role", "user"}, {"content", "hi"}},
                                    {{"role", "assistant"}, {"content", Json::array({image})}}}),
                       Json(), ChatOptions{});
    } catch (const ChatError& e) {
      rejected = std::string(e.what()).find("user and tool") != std::string::npos;
    }
    c.expect(rejected, "image parts: an image in an assistant message is a ChatError");
  }
  // Concurrency: every thread renders every request and compares text and ids with a serial render (the
  // render cache is shared by the threads).
  const auto fingerprint = [](const RenderedPrompt& p) {
    std::string s = p.text + "|";
    for (const int32_t id : p.ids) s += std::to_string(id) + ",";
    return s;
  };
  std::vector<std::string> serial;
  for (const Json& req : golden) {
    ChatOptions options{req["reasoning_effort"], req["enable_thinking"], req["kwargs"]};
    try {
      serial.push_back(fingerprint(fmt.render(req["messages"], req["tools"], options)));
    } catch (const ChatError& e) {
      serial.push_back(std::string("error:") + e.what());
    }
  }
  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      for (int round = 0; round < 3; ++round) {
        for (std::size_t k = 0; k < golden.size(); ++k) {
          const std::size_t i = (k + t * 7) % golden.size();
          const Json& req = golden[i];
          ChatOptions options{req["reasoning_effort"], req["enable_thinking"], req["kwargs"]};
          std::string got;
          try {
            got = fingerprint(fmt.render(req["messages"], req["tools"], options));
          } catch (const ChatError& e) {
            got = std::string("error:") + e.what();
          }
          if (got != serial[i]) ++mismatches;
        }
      }
    });
  }
  for (auto& th : threads) th.join();
  c.expect(mismatches.load() == 0, "concurrent renders differ: " + std::to_string(mismatches.load()));
  // Verified root-prefix token boundary detection on rendered prompts (#826).
  {
    const auto sys_id = *fmt.tokenizer().token_id(MESSAGE_SYSTEM);
    const auto user_id = *fmt.tokenizer().token_id(MESSAGE_USER);
    const auto model_id = *fmt.tokenizer().token_id(MESSAGE_MODEL);
    const auto end_id = *fmt.tokenizer().token_id(END_MESSAGE);
    const Json tools = Json::array({{{"type", "function"},
                                     {"function",
                                      {{"name", "read_file"},
                                       {"description", "Read a file"},
                                       {"parameters", {{"type", "object"}, {"properties", {{"path", {{"type", "string"}}}}}}}}}}});
    const Json fork_a = Json::array(
        {{{"role", "system"}, {"content", "You are a precise coding assistant."}},
         {{"role", "user"}, {"content", "Inspect alpha.cpp"}}});
    const Json fork_b = Json::array(
        {{{"role", "system"}, {"content", "You are a precise coding assistant."}},
         {{"role", "user"}, {"content", "Inspect beta.cpp"}}});
    const RenderedPrompt pa = fmt.render(fork_a, tools, ChatOptions{});
    const RenderedPrompt pb = fmt.render(fork_b, tools, ChatOptions{});
    c.expect(pa.root_prefix_tokens > 0 && std::size_t(pa.root_prefix_tokens) < pa.ids.size(),
             "root_prefix: tools+system+user has strict prefix boundary");
    c.expect(pa.root_prefix_tokens == pb.root_prefix_tokens,
             "root_prefix: forks after system+tools share identical boundary");
    c.expect(pa.ids[0] == sys_id && pa.ids[std::size_t(pa.root_prefix_tokens - 1)] == end_id &&
                 pa.ids[std::size_t(pa.root_prefix_tokens)] == user_id,
             "root_prefix: starts at <|message_system|>, ends at <|end_message|> right before <|message_user|>");
    c.expect(std::equal(pa.ids.begin(), pa.ids.begin() + pa.root_prefix_tokens, pb.ids.begin()),
             "root_prefix: forks match token-for-token up to boundary");
    // Exact-root system-only prompt: root_prefix_tokens == ids.size().
    const Json sys_only = Json::array({{{"role", "system"}, {"content", "You are a precise coding assistant."}}});
    const RenderedPrompt ps = fmt.render(sys_only, tools, ChatOptions{});
    c.expect(ps.root_prefix_tokens == int(ps.ids.size()),
             "root_prefix: exact-root system-only prompt equals full ids length");
    c.expect(ps.ids.size() == std::size_t(pa.root_prefix_tokens) &&
                 std::equal(ps.ids.begin(), ps.ids.end(), pa.ids.begin()),
             "root_prefix: exact-root render matches fork prefix token-for-token");
    // enable_thinking=false appends <|message_model|><|content_text|> after root; boundary stops before it.
    ChatOptions nothink;
    nothink.enable_thinking = false;
    const RenderedPrompt pnt = fmt.render(sys_only, tools, nothink);
    c.expect(pnt.root_prefix_tokens == ps.root_prefix_tokens &&
                 pnt.ids[std::size_t(pnt.root_prefix_tokens)] == model_id,
             "root_prefix: enable_thinking=false stops before <|message_model|>");
    // Mid-conversation system message after user turn must NOT extend root_prefix_tokens.
    const Json mid_sys = Json::array(
        {{{"role", "system"}, {"content", "You are a precise coding assistant."}},
         {{"role", "user"}, {"content", "First question"}},
         {{"role", "system"}, {"content", "Mid-turn delta"}}});
    const RenderedPrompt pm = fmt.render(mid_sys, tools, ChatOptions{});
    c.expect(pm.root_prefix_tokens == pa.root_prefix_tokens,
             "root_prefix: mid-conversation system message does not move root boundary");

    // Startup warm-start system prefix (#887): matches Turn-1 regardless of reasoning_effort or enable_thinking.
    const std::string sys_text = "You are a precise coding assistant.";
    const RenderedPrompt ws = fmt.render_system_prefix(sys_text);
    c.expect(!ws.ids.empty() && ws.root_prefix_tokens == int(ws.ids.size()),
             "warm_start_prefix: non-empty with exact root_prefix_tokens");
    c.expect(ws.ids.front() == sys_id && ws.ids.back() == end_id,
             "warm_start_prefix: spans <|message_system|> ... <|end_message|>");
    const Json turn1 = Json::array(
        {{{"role", "system"}, {"content", sys_text}}, {{"role", "user"}, {"content", "Write a binary search."}}});
    ChatOptions opt_default, opt_med, opt_nothink;
    opt_med.reasoning_effort = "medium";
    opt_nothink.enable_thinking = false;
    for (const ChatOptions& o : {opt_default, opt_med, opt_nothink}) {
      const RenderedPrompt p = fmt.render(turn1, nullptr, o);
      c.expect(p.ids.size() > ws.ids.size() && std::equal(ws.ids.begin(), ws.ids.end(), p.ids.begin()),
               "warm_start_prefix: exact token prefix of Turn-1 across effort/thinking options");
    }
    bool rejected_ws = false;
    try {
      (void)fmt.render_system_prefix("   \n\t  ");
    } catch (const ChatError&) {
      rejected_ws = true;
    }
    c.expect(rejected_ws, "warm_start_prefix: whitespace-only prompt rejected fail-closed");
  }
  return c.finish("text_chat");
}

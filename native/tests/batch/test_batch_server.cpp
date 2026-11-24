// End-to-end HTTP continuous-batching test: hammers ChatServer (max_batch=64) with 64 concurrent
// streaming SSE and non-streaming OpenAI /v1/chat/completions requests simultaneously over localhost.
#include "serve/chat_server.h"
#include "text/chat.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ingine::model::BatchSlotStep;
using ingine::model::GenerateStats;
using ingine::model::Prompt;
using ingine::model::Sampling;
using ingine::serve::ChatServer;
using ingine::text::ChatFormat;

int g_failures = 0;
#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                                   \
    }                                                                                 \
  } while (0)

class FakeHttpBatchGenerator final : public ingine::model::Generator {
 public:
  FakeHttpBatchGenerator(const ingine::text::Tokenizer& tok, int max_batch)
      : tok_(tok), max_batch_(max_batch), slots_(static_cast<std::size_t>(max_batch)) {}

  std::string model_id() const override { return "inkling-batch-test"; }
  int max_model_len() const override { return 8192; }
  int max_batch_size() const override { return max_batch_; }
  bool can_admit_slot(int, int) const override {
    std::lock_guard lock(mu_);
    for (const auto& s : slots_)
      if (!s.used) return true;
    return false;
  }

  int start_slot(const Prompt& prompt, int, const Sampling&, int32_t& first_token, GenerateStats& stats) override {
    std::lock_guard lock(mu_);
    for (int i = 0; i < max_batch_; ++i) {
      auto& s = slots_[static_cast<std::size_t>(i)];
      if (!s.used) {
        const std::string prompt_txt = tok_.decode_lossy(prompt.ids);
        std::string tag = "ok";
        const auto p = prompt_txt.find("REQ_");
        if (p != std::string::npos) tag = prompt_txt.substr(p, 7);
        const std::string reply = "<|message_model|><|content_text|>Reply for " + tag +
                                  " done.<|end_message|><|content_model_end_sampling|>";
        s.tokens = tok_.encode(reply, /*special=*/true);
        s.used = true;
        s.idx = 1;
        first_token = s.tokens[0];
        stats.prompt_tokens = int(prompt.ids.size());
        ++active_;
        peak_active_ = std::max(peak_active_, active_);
        return i;
      }
    }
    return -1;
  }

  void step_slots(std::span<const BatchSlotStep> inputs, std::span<int32_t> out_tokens) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    std::lock_guard lock(mu_);
    peak_step_batch_ = std::max(peak_step_batch_, int(inputs.size()));
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      auto& s = slots_[static_cast<std::size_t>(inputs[i].slot)];
      const int32_t tok = s.idx < s.tokens.size() ? s.tokens[s.idx++] : s.tokens.back();
      out_tokens[i] = tok;
    }
  }

  void release_slot(int slot) override {
    std::lock_guard lock(mu_);
    auto& s = slots_[static_cast<std::size_t>(slot)];
    if (!s.used) return;
    s.used = false;
    s.tokens.clear();
    --active_;
  }

  GenerateStats generate(std::span<const int32_t>, int, const Sampling&, std::span<const int32_t>,
                         const std::function<bool(std::span<const int32_t>)>&) override {
    throw std::logic_error("unexpected single-request generate()");
  }

  int peak_active() const {
    std::lock_guard lock(mu_);
    return peak_active_;
  }
  int active() const {
    std::lock_guard lock(mu_);
    return active_;
  }

 private:
  struct Slot {
    bool used = false;
    std::vector<int32_t> tokens;
    std::size_t idx = 0;
  };
  const ingine::text::Tokenizer& tok_;
  int max_batch_;
  mutable std::mutex mu_;
  std::vector<Slot> slots_;
  int active_ = 0;
  int peak_active_ = 0;
  int peak_step_batch_ = 0;
};

}  // namespace

int main() {
  std::string model_dir = INGINE_MODEL_DIR;
  if (const char* env = std::getenv("INGINE_MODEL_DIR")) model_dir = env;
  if (!std::filesystem::exists(std::filesystem::path(model_dir) / "tokenizer.json")) {
    std::fprintf(stderr, "batch_server: model dir '%s' missing, skipping\n", model_dir.c_str());
    return 77;
  }

  const ChatFormat fmt = ChatFormat::load(model_dir);
  FakeHttpBatchGenerator gen(fmt.tokenizer(), 64);
  ingine::serve::ServerOptions opts;
  opts.max_queue = 64;
  ChatServer server(gen, fmt, opts);
  CHECK(server.bind("127.0.0.1", 0));
  server.start();
  const int port = server.port();

  // Verify /health exposes max_batch=64, active=0, running=false
  {
    httplib::Client cli("127.0.0.1", port);
    auto res = cli.Get("/health");
    CHECK(res && res->status == 200);
    const Json h = Json::parse(res->body);
    CHECK(h["max_batch"] == 64 && h["active"] == 0 && h["running"] == false);
  }

  // Hammer server with 64 simultaneous HTTP clients (32 non-streaming + 32 SSE streaming)
  const int N = 64;
  std::atomic<int> ok_count{0};
  std::vector<std::thread> clients;
  clients.reserve(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    clients.emplace_back([&, i] {
      char tag[16];
      std::snprintf(tag, sizeof tag, "REQ_%03d", i);
      const bool stream = (i % 2 == 1);
      Json body{{"model", "inkling-batch-test"},
                {"messages", Json::array({{{"role", "user"}, {"content", std::string("Ping ") + tag}}})},
                {"max_tokens", 64},
                {"stream", stream}};
      httplib::Client cli("127.0.0.1", port);
      cli.set_read_timeout(30, 0);
      auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
      if (!res || res->status != 200) return;
      const std::string expected = std::string("Reply for ") + tag + " done.";
      if (!stream) {
        const Json j = Json::parse(res->body);
        const std::string content = j["choices"][0]["message"]["content"].get<std::string>();
        if (content == expected && j["choices"][0]["finish_reason"] == "stop") ++ok_count;
      } else {
        std::string assembled;
        bool got_stop = false;
        std::size_t pos = 0;
        while (pos < res->body.size()) {
          const auto nl = res->body.find('\n', pos);
          const std::string line = res->body.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
          pos = (nl == std::string::npos) ? res->body.size() : nl + 1;
          if (line.rfind("data: ", 0) != 0) continue;
          const std::string payload = line.substr(6);
          if (payload == "[DONE]") break;
          const Json chunk = Json::parse(payload);
          if (chunk.contains("choices") && !chunk["choices"].empty()) {
            const auto& c = chunk["choices"][0];
            if (c.contains("delta") && c["delta"].contains("content") && c["delta"]["content"].is_string()) {
              assembled += c["delta"]["content"].get<std::string>();
            }
            if (c.contains("finish_reason") && c["finish_reason"].is_string() && c["finish_reason"] == "stop") {
              got_stop = true;
            }
          }
        }
        if (assembled == expected && got_stop) ++ok_count;
      }
    });
  }
  for (auto& t : clients) t.join();
  server.stop();

  CHECK(ok_count.load() == N);
  CHECK(gen.peak_active() >= 16);  // proves concurrent HTTP serving well above the old 8-thread cap!
  CHECK(gen.active() == 0);
  std::printf("batch_server: ok=%d/%d peak_active=%d\n", ok_count.load(), N, gen.peak_active());
  std::printf(g_failures ? "FAILED (%d checks)\n" : "batch_server ok\n", g_failures);
  return g_failures ? 1 : 0;
}

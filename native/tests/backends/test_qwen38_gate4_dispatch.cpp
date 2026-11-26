#include "backends/qwen38/qwen38_backend.h"
#include "engine/fake_generator.h"
#include "ingine/model_backend.h"
#include "qwen38_883_bench_checks.h"
#include "serve/chat_server.h"
#include "text/chat.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ingine::backends::qwen38::find_qwen35_pre_suffix_checkpoint;
using ingine::backends::qwen38::resolve_qwen38_runtime_options;
using ingine::engine::FakeGenerator;
using ingine::model::KvCacheMode;
using ingine::model::ModelFamily;
using ingine::model::RuntimeOptions;
using ingine::model::detect_model_family;
using ingine::model::prefix_matches_exact;
using ingine::serve::ChatServer;
using ingine::serve::ServerOptions;
using ingine::text::ChatFormat;
using ingine::text::ChatFormatKind;
using ingine::text::ChatOptions;
using ingine::text::RenderedPrompt;

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

#define CHECK_THROWS(expr, ex_type)                                                         \
  do {                                                                                      \
    bool threw_ = false;                                                                    \
    try {                                                                                   \
      (void)(expr);                                                                         \
    } catch (const ex_type&) {                                                              \
      threw_ = true;                                                                        \
    }                                                                                       \
    if (!threw_) {                                                                          \
      std::fprintf(stderr, "%s:%d: expected exception %s: %s\n", __FILE__, __LINE__,        \
                   #ex_type, #expr);                                                        \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

struct TempDir {
  std::filesystem::path path;
  explicit TempDir(const char* tag) {
    path = std::filesystem::temp_directory_path() /
           (std::string("ingine_test_g4_") + tag + "_" + std::to_string(::getpid()));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

void write_text_file(const std::filesystem::path& p, const std::string& body) {
  std::ofstream out(p, std::ios::binary);
  CHECK(out.good());
  out.write(body.data(), static_cast<std::streamsize>(body.size()));
}

std::string make_synthetic_qwen_config(bool mixed_fp8_nvfp4) {
  nlohmann::json root = {
      {"architectures", {"Qwen3_5ForConditionalGeneration"}},
      {"model_type", "qwen3_5"},
      {"text_config",
       {
           {"model_type", "qwen3_5_text"},
           {"vocab_size", 248320},
           {"hidden_size", 512},
           {"intermediate_size", 1024},
           {"num_hidden_layers", 4},
           {"max_position_embeddings", 65536},
           {"rms_norm_eps", 1e-6},
           {"full_attention_interval", 4},
           {"num_attention_heads", 4},
           {"num_key_value_heads", 2},
           {"head_dim", 128},
           {"rope_theta", 10000000.0},
           {"partial_rotary_factor", 0.25},
           {"linear_conv_kernel_dim", 4},
           {"linear_key_head_dim", 64},
           {"linear_value_head_dim", 64},
           {"linear_num_key_heads", 2},
           {"linear_num_value_heads", 4},
           {"mamba_ssm_dtype", "float32"},
       }},
  };
  if (mixed_fp8_nvfp4) {
    root["quantization_config"] = {
        {"quant_method", "modelopt"},
        {"quant_algo", "MIXED_PRECISION"},
    };
  }
  return root.dump();
}

void test_inkling_loop_and_qwen_detection() {
  TempDir ink_loop("ink_loop");
  write_text_file(ink_loop.path / "config.json", R"({
    "architectures": ["Inkling2LoopForConditionalGeneration"],
    "model_type": "inkling2_loop",
    "text_config": {"model_type": "inkling2_loop_text"}
  })");
  CHECK(detect_model_family(ink_loop.path) == ModelFamily::Inkling2);

  const std::filesystem::path ink_real("/mnt/storage/models/Inkling-2-Mini-ingine");
  if (std::filesystem::is_directory(ink_real)) {
    CHECK(detect_model_family(ink_real) == ModelFamily::Inkling2);
  }
  const std::filesystem::path qwen_real("/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f");
  if (std::filesystem::is_directory(qwen_real)) {
    CHECK(detect_model_family(qwen_real) == ModelFamily::Qwen3_5);
  }
}

void test_resolve_qwen38_runtime_options() {
  TempDir nvfp4_synth("nvfp4");
  write_text_file(nvfp4_synth.path / "config.json", make_synthetic_qwen_config(/*mixed_fp8_nvfp4=*/true));

  const std::filesystem::path qwen_real("/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f");
  const std::filesystem::path nvfp4_dir = std::filesystem::is_directory(qwen_real) ? qwen_real : nvfp4_synth.path;

  // Defaults (dense_fp8 omitted / false, park_mb = -1, vision = false, kv_cache_mode = PerPass) normalize cleanly.
  {
    RuntimeOptions opts;
    opts.model_dir = nvfp4_dir;
    opts.max_ctx = 8192;
    opts.prefill_chunk = 256;
    opts.dense_fp8 = false;
    opts.vision = false;
    opts.prefix_cache = true;
    const auto plan = resolve_qwen38_runtime_options(opts);
    CHECK(plan.normalized_options.dense_fp8);
    CHECK(plan.use_calibrated_fp8);
    CHECK(plan.verify_pins);
    CHECK(plan.prefix_cache);
    CHECK(plan.max_ctx == 8192);
    CHECK(plan.prefill_chunk == 256);
  }
  {
    RuntimeOptions opts;
    opts.model_dir = nvfp4_dir;
    opts.max_ctx = 4096;
    opts.prefill_chunk = 256;
    opts.dense_fp8 = false;
    opts.prefix_cache = false;
    const auto plan = resolve_qwen38_runtime_options(opts);
    CHECK(!plan.prefix_cache);
    CHECK(!plan.normalized_options.prefix_cache);
  }

  // Every unsupported Inkling flag must fail closed with std::invalid_argument.
  const std::vector<std::function<void(RuntimeOptions&)>> invalid_mutators = {
      [](auto& o) { o.kv_cache_mode = KvCacheMode::LastFull; },
      [](auto& o) { o.loop_passes = 1; },
      [](auto& o) { o.loop_passes = 6; },
      [](auto& o) { o.loop_passes_explicit = true; },
      [](auto& o) { o.park_host_mb = 512; },
      [](auto& o) { o.park_mb = 512; },
      [](auto& o) { o.vision = true; },
      [](auto& o) { o.spec = true; },
      [](auto& o) { o.max_batch = 2; },
      [](auto& o) { o.kv_fp8 = true; },
      [](auto& o) { o.kv_q4 = true; },
      [](auto& o) { o.kv_paged = true; },
      [](auto& o) { o.kv_vram_mb = 512; },
      [](auto& o) { o.kv_swap_dir = "/tmp"; },
      [](auto& o) { o.offload_vram_mb = 5888; },
      [](auto& o) { o.max_kv_tokens = 4096; },
      [](auto& o) { o.ablation = "abl.safetensors"; },
      [](auto& o) { o.enable_persona_hooks = true; },
      [](auto& o) { o.refusal_map = "refusal.json"; },
      [](auto& o) { o.kv_ckpt_path = "ckpt.bin"; },
      [](auto& o) { o.kv_ckpt_resume = "ckpt.bin"; },
      [](auto& o) { o.kv_ckpt_every = 128; },
      [](auto& o) { o.expert_ranking = "rank.json"; },
      [](auto& o) { o.dump_router_stats = "router.json"; },
  };
  for (const auto& mut : invalid_mutators) {
    RuntimeOptions bad;
    bad.model_dir = nvfp4_dir;
    bad.max_ctx = 8192;
    bad.prefill_chunk = 256;
    bad.dense_fp8 = false;
    mut(bad);
    CHECK_THROWS(resolve_qwen38_runtime_options(bad), std::invalid_argument);
  }

  // BF16 Qwen3.8 checkpoint rejects dense_fp8=true fail-closed and allows dense_fp8=false.
  TempDir bf16_synth("bf16");
  write_text_file(bf16_synth.path / "config.json", make_synthetic_qwen_config(/*mixed_fp8_nvfp4=*/false));
  {
    RuntimeOptions ok_bf16;
    ok_bf16.model_dir = bf16_synth.path;
    ok_bf16.max_ctx = 4096;
    ok_bf16.prefill_chunk = 256;
    ok_bf16.dense_fp8 = false;
    const auto plan = resolve_qwen38_runtime_options(ok_bf16);
    CHECK(!plan.use_calibrated_fp8);
    CHECK(!plan.normalized_options.dense_fp8);

    RuntimeOptions bad_bf16 = ok_bf16;
    bad_bf16.dense_fp8 = true;
    CHECK_THROWS(resolve_qwen38_runtime_options(bad_bf16), std::invalid_argument);
  }
}

void test_root_prefix_tokens_safety(const std::filesystem::path& qwen_dir) {
  const ChatFormat qfmt = ChatFormat::load(qwen_dir);
  CHECK(qfmt.kind() == ChatFormatKind::Qwen3_5);
  const Json msgs = Json::array({
      {{"role", "system"}, {"content", "Be concise."}},
      {{"role", "user"}, {"content", "Hello world"}},
  });
  const Json tools = Json::array({{{
      "type", "function"},
      {"function",
       {{"name", "ping"},
        {"description", "Ping target"},
        {"parameters", {{"type", "object"}, {"properties", {{"host", {{"type", "string"}}}}}}}}}}});
  ChatOptions opts;
  opts.enable_thinking = true;
  const auto r1 = qfmt.render(msgs, Json(), opts);
  CHECK(r1.root_prefix_tokens == 0 && !r1.ids.empty());
  const auto r2 = qfmt.render(msgs, tools, opts);
  CHECK(r2.root_prefix_tokens == 0 && !r2.ids.empty());

  const std::filesystem::path ink_dir("/mnt/storage/models/Inkling-2-Mini-ingine");
  if (std::filesystem::is_regular_file(ink_dir / "tokenizer.json") &&
      std::filesystem::is_regular_file(ink_dir / "chat_template.jinja")) {
    const ChatFormat ifmt = ChatFormat::load(ink_dir);
    CHECK(ifmt.kind() == ChatFormatKind::Inkling2);
    const auto ri = ifmt.render(msgs, tools, opts);
    CHECK(ri.root_prefix_tokens > 0 && static_cast<std::size_t>(ri.root_prefix_tokens) < ri.ids.size());
  }
}

std::vector<Json> parse_sse(const std::string& data) {
  std::vector<Json> events;
  std::size_t pos = 0;
  while (pos < data.size()) {
    std::size_t end = data.find("\n\n", pos);
    if (end == std::string::npos) end = data.size();
    const std::string block = data.substr(pos, end - pos);
    pos = end + 2;
    if (block.find_first_not_of(" \t\r\n") == std::string::npos || block.starts_with(":")) continue;
    CHECK(block.starts_with("data: "));
    const std::string payload = block.substr(6);
    events.push_back(payload == "[DONE]" ? Json("[DONE]") : Json::parse(payload));
  }
  return events;
}

void test_qwen_fake_http_server(const std::filesystem::path& qwen_dir) {
  const ChatFormat fmt = ChatFormat::load(qwen_dir);
  FakeGenerator gen(fmt.tokenizer(), "Qwen3.8-27B-NVFP4-fake", 262144, {}, 0.0, /*loop_passes=*/1,
                    KvCacheMode::PerPass, /*allow_last_full=*/false);
  gen.set_model_family("qwen3_5");
  gen.set_supports_logit_masker(false);
  gen.set_supports_images(false);
  ServerOptions so;
  so.images.enabled = false;
  ChatServer srv(gen, fmt, so);
  CHECK(srv.bind("127.0.0.1", 0));
  srv.start();

  httplib::Client cli("127.0.0.1", srv.port());
  cli.set_read_timeout(20, 0);

  // 1. /health and /v1/models advertise accurate non-looped qwen3_5 capabilities
  {
    const auto r_health = cli.Get("/health");
    CHECK(r_health && r_health->status == 200);
    const Json jh = Json::parse(r_health->body);
    CHECK(jh["status"] == "ok");
    CHECK(jh["model"] == "Qwen3.8-27B-NVFP4-fake");
    CHECK(jh["model_family"] == "qwen3_5");
    CHECK(jh["kv_cache_mode"] == "per_pass");
    CHECK(jh["supported_kv_cache_modes"] == Json::array({"per_pass"}));
    CHECK(jh["loop_passes"] == 1);
    CHECK(jh["supports_structured_output"] == false);
    CHECK(r_health->get_header_value("X-Ingine-Loop-Passes") == "1");

    const auto r_models = cli.Get("/v1/models");
    CHECK(r_models && r_models->status == 200);
    const Json jm = Json::parse(r_models->body);
    CHECK(jm["data"].size() == 1);
    CHECK(jm["data"][0]["id"] == "Qwen3.8-27B-NVFP4-fake");
    CHECK(jm["data"][0]["model_family"] == "qwen3_5");
    CHECK(jm["data"][0]["supported_kv_cache_modes"] == Json::array({"per_pass"}));
    CHECK(jm["data"][0]["loop_passes"] == 1);
    CHECK(jm["data"][0]["supports_structured_output"] == false);
    CHECK(jm["data"][0]["input_modalities"] == Json::array({"text"}));
  }

  // 1b. Stale-depth precondition returns HTTP 409 before generation; matching depth 1 succeeds.
  {
    const std::size_t before_calls = gen.calls().size();
    const Json body{{"messages", Json::array({{{"role", "user"}, {"content", "depth check"}}})}};
    const auto r_409 = cli.Post("/v1/chat/completions", httplib::Headers{{"X-Ingine-Expected-Loop-Passes", "3"}},
                                body.dump(), "application/json");
    CHECK(r_409 && r_409->status == 409);
    CHECK(gen.calls().size() == before_calls);

    const auto r_ok = cli.Post("/v1/chat/completions", httplib::Headers{{"X-Ingine-Expected-Loop-Passes", "1"}},
                               body.dump(), "application/json");
    CHECK(r_ok && r_ok->status == 200);
    CHECK(gen.calls().size() == before_calls + 1);
  }

  // 1c. Structured JSON response_format is rejected explicitly with HTTP 400 BEFORE generation; type="text" passes.
  {
    const std::size_t before_calls = gen.calls().size();
    const Json j_obj{{"messages", Json::array({{{"role", "user"}, {"content", "return json"}}})},
                     {"response_format", {{"type", "json_object"}}}};
    const auto r_obj = cli.Post("/v1/chat/completions", j_obj.dump(), "application/json");
    CHECK(r_obj && r_obj->status == 400);
    const Json e_obj = Json::parse(r_obj->body);
    CHECK(e_obj["error"]["param"] == "response_format");
    CHECK(gen.calls().size() == before_calls);

    const Json j_schema{
        {"messages", Json::array({{{"role", "user"}, {"content", "return schema json"}}})},
        {"response_format",
         {{"type", "json_schema"},
          {"json_schema",
           {{"name", "s"},
            {"schema", {{"type", "object"}, {"properties", {{"ok", {{"type", "boolean"}}}}}, {"required", {"ok"}}}}}}}}};
    const auto r_schema = cli.Post("/v1/chat/completions", j_schema.dump(), "application/json");
    CHECK(r_schema && r_schema->status == 400);
    const Json e_schema = Json::parse(r_schema->body);
    CHECK(e_schema["error"]["param"] == "response_format");
    CHECK(gen.calls().size() == before_calls);

    const Json j_text{{"messages", Json::array({{{"role", "user"}, {"content", "plain text ok"}}})},
                      {"response_format", {{"type", "text"}}}};
    const auto r_text = cli.Post("/v1/chat/completions", j_text.dump(), "application/json");
    CHECK(r_text && r_text->status == 200);
    CHECK(gen.calls().size() == before_calls + 1);
  }

  // 2. Non-stream thinking enabled + zero negative banned_tokens
  {
    const Json body{{"messages", Json::array({{{"role", "user"}, {"content", "hello qwen"}}})}};
    const auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    CHECK(res && res->status == 200);
    const Json j = Json::parse(res->body);
    const Json& msg = j["choices"][0]["message"];
    CHECK(j["choices"][0]["finish_reason"] == "stop");
    CHECK(msg["reasoning_content"].is_string() && !msg["reasoning_content"].get<std::string>().empty());
    CHECK(msg["content"] == "You said: hello qwen");
    const auto calls = gen.calls();
    CHECK(!calls.empty());
    for (std::int32_t bid : calls.back().sampling.banned_tokens) CHECK(bid >= 0);
  }

  // 3. Non-stream thinking disabled
  {
    const Json body{{"messages", Json::array({{{"role", "user"}, {"content", "hello nothink"}}})},
                    {"chat_template_kwargs", {{"enable_thinking", false}}}};
    const auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    CHECK(res && res->status == 200);
    const Json j = Json::parse(res->body);
    const Json& msg = j["choices"][0]["message"];
    CHECK(j["choices"][0]["finish_reason"] == "stop");
    CHECK(msg["reasoning_content"].is_null());
    CHECK(msg["content"] == "You said: hello nothink");
  }

  // 4. Streaming SSE
  {
    const Json body{{"messages", Json::array({{{"role", "user"}, {"content", "hello stream"}}})}, {"stream", true}};
    const auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    CHECK(res && res->status == 200);
    const auto ev = parse_sse(res->body);
    CHECK(!ev.empty() && ev.back() == "[DONE]");
    std::string rc, ct, finish;
    for (std::size_t i = 0; i + 1 < ev.size(); ++i) {
      if (!ev[i].contains("choices") || ev[i]["choices"].empty()) continue;
      const Json& c0 = ev[i]["choices"][0];
      if (!c0["finish_reason"].is_null()) finish = c0["finish_reason"].get<std::string>();
      const Json& d = c0["delta"];
      if (d.contains("reasoning_content") && d["reasoning_content"].is_string()) {
        rc += d["reasoning_content"].get<std::string>();
      }
      if (d.contains("content") && d["content"].is_string()) {
        ct += d["content"].get<std::string>();
      }
    }
    CHECK(finish == "stop");
    CHECK(!rc.empty());
    CHECK(ct == "You said: hello stream");
  }

  // 5. XML tool call parsing round-trip
  {
    const Json tools = Json::array({{{
        "type", "function"},
        {"function",
         {{"name", "get_weather"},
          {"description", "Get weather"},
          {"parameters", {{"type", "object"}, {"properties", {{"city", {{"type", "string"}}}}}}}}}}});
    const Json body{
        {"messages", Json::array({{{"role", "user"}, {"content", "call get_weather {\"city\":\"Paris\"}"}}})},
        {"tools", tools}};
    const auto res = cli.Post("/v1/chat/completions", body.dump(), "application/json");
    CHECK(res && res->status == 200);
    const Json j = Json::parse(res->body);
    CHECK(j["choices"][0]["finish_reason"] == "tool_calls");
    const Json& tc = j["choices"][0]["message"]["tool_calls"];
    CHECK(tc.size() == 1);
    CHECK(tc[0]["function"]["name"] == "get_weather");
    const Json parsed_args = Json::parse(tc[0]["function"]["arguments"].get<std::string>());
    CHECK(parsed_args["city"] == "Paris");
  }

  // 6. Authorization header forces fail-closed bypass_prefix_cache
  {
    const Json body{{"messages", Json::array({{{"role", "user"}, {"content", "isolated auth"}}})}};
    const httplib::Headers hdrs{{"Authorization", "Bearer secret"}};
    const auto res = cli.Post("/v1/chat/completions", hdrs, body.dump(), "application/json");
    CHECK(res && res->status == 200);
    const auto calls = gen.calls();
    CHECK(calls.back().bypass_prefix_cache);
  }
  srv.stop();

  // 7. Cache namespace header scoping vs fail-closed bypass
  {
    FakeGenerator ns_gen(fmt.tokenizer(), "Qwen3.8-27B-NVFP4-fake", 262144, {}, 0.0);
    ServerOptions ns_so;
    ns_so.images.enabled = false;
    ns_so.cache_namespace_header = "X-Tenant";
    ChatServer ns_srv(ns_gen, fmt, ns_so);
    CHECK(ns_srv.bind("127.0.0.1", 0));
    ns_srv.start();

    httplib::Client ns_cli("127.0.0.1", ns_srv.port());
    ns_cli.set_read_timeout(20, 0);
    const Json body{{"messages", Json::array({{{"role", "user"}, {"content", "tenant test"}}})}};

    const auto r_with = ns_cli.Post("/v1/chat/completions",
                                    httplib::Headers{{"X-Tenant", "team-a"}, {"Authorization", "Bearer none"}},
                                    body.dump(), "application/json");
    CHECK(r_with && r_with->status == 200);
    CHECK(ns_gen.calls().back().cache_namespace == "team-a");
    CHECK(!ns_gen.calls().back().bypass_prefix_cache);

    const auto r_without = ns_cli.Post("/v1/chat/completions", body.dump(), "application/json");
    CHECK(r_without && r_without->status == 200);
    CHECK(ns_gen.calls().back().bypass_prefix_cache);
    ns_srv.stop();
  }
}

void test_qwen35_pre_suffix_checkpoint(const std::filesystem::path& qwen_dir) {
  // Synthetic non-chat prompts must never trigger a pre-suffix checkpoint.
  {
    const std::vector<std::int32_t> synthetic = {10, 20, 30, 40, 50, 60, 70, 80};
    CHECK(find_qwen35_pre_suffix_checkpoint(synthetic) == 0);
    // Suffix without leading <|im_end|> boundary must also return 0.
    const std::vector<std::int32_t> no_im_end = {10, 20, 248045, 74455, 198, 248068, 198};
    CHECK(find_qwen35_pre_suffix_checkpoint(no_im_end) == 0);
  }

  const ChatFormat fmt = ChatFormat::load(qwen_dir);
  const Json tools = Json::array({{{"type", "function"},
                                   {"function",
                                    {{"name", "get_weather"},
                                     {"description", "Get weather"},
                                     {"parameters",
                                      {{"type", "object"},
                                       {"properties", {{"city", {{"type", "string"}}}}},
                                       {"required", Json::array({"city"})}}}}}}});

  // 1. enable_thinking=true (BFCL default): Turn 1 ends with <|im_end|>\n<|im_start|>assistant\n<think>\n (5 tok).
  ChatOptions think_opts;
  think_opts.enable_thinking = true;
  think_opts.reasoning_effort = "medium";
  const Json m1 = Json::array({{{"role", "user"}, {"content", "Check Tokyo weather with get_weather."}}});
  const RenderedPrompt p1 = fmt.render(m1, tools, think_opts);
  const std::size_t pre1 = find_qwen35_pre_suffix_checkpoint(p1.ids);
  CHECK(pre1 > 0 && pre1 + 5 == p1.ids.size());
  CHECK(p1.ids[pre1] == 248045);  // <|im_start|>

  // Turn 2: OpenAI BFCL client omits reasoning_content -> template renders <think>\n\n</think>\n\n,
  // diverging at p1.ids.size() - 1 (1 token before Turn 1's end), while matching all tokens up to pre1!
  const Json m2 = Json::array({
      {{"role", "user"}, {"content", "Check Tokyo weather with get_weather."}},
      {{"role", "assistant"},
       {"content", nullptr},
       {"tool_calls",
        Json::array({{{"id", "call_1"},
                      {"type", "function"},
                      {"function", {{"name", "get_weather"}, {"arguments", "{\"city\":\"Tokyo\"}"}}}}})}},
      {{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", "{\"temp_c\":19}"}},
  });
  const RenderedPrompt p2 = fmt.render(m2, tools, think_opts);
  const std::size_t pre2 = find_qwen35_pre_suffix_checkpoint(p2.ids);
  CHECK(pre2 > pre1 && pre2 + 5 == p2.ids.size());
  // Verify Turn 1 and Turn 2 diverge strictly before p1.ids.size(), and match 100% through pre1!
  std::size_t lcp = 0;
  while (lcp < p1.ids.size() && lcp < p2.ids.size() && p1.ids[lcp] == p2.ids[lcp]) ++lcp;
  CHECK(lcp < p1.ids.size());
  CHECK(lcp >= pre1);
  CHECK(prefix_matches_exact(p1.ids, p2.ids, pre1));

  // 2. enable_thinking=false: Turn 1 ends with <|im_start|>assistant\n<think>\n\n</think>\n\n (7 tok).
  ChatOptions nothink_opts;
  nothink_opts.enable_thinking = false;
  const RenderedPrompt p1_nt = fmt.render(m1, tools, nothink_opts);
  const std::size_t pre1_nt = find_qwen35_pre_suffix_checkpoint(p1_nt.ids);
  CHECK(pre1_nt == pre1 && pre1_nt + 7 == p1_nt.ids.size());
  CHECK(prefix_matches_exact(p1.ids, p1_nt.ids, pre1));
}

void test_883_followup_restore_predicate() {
  using ingine::backends::qwen38::test_883::check_multiturn_presuffix_restores;
  using ingine::backends::qwen38::test_883::check_presuffix_followup_restore;
  using ingine::model::GenerateStats;
  using ingine::model::Reuse;

  auto mk = [](Reuse r, int reused, std::optional<std::uint64_t> bytes) {
    GenerateStats st;
    st.reuse = r;
    st.reused_tokens = reused;
    st.restored_bytes = bytes;
    return st;
  };

  // Positive: every follow-up turn restores the prior turn's exact pre-suffix boundary via Checkpoint + nonzero bytes.
  const std::vector<std::size_t> bounds = {4204, 4281, 4369, 4416};
  const std::vector<GenerateStats> good = {
      mk(Reuse::None, 0, 0ull),
      mk(Reuse::Checkpoint, 4204, 158859264ull),
      mk(Reuse::Checkpoint, 4281, 158859264ull),
      mk(Reuse::Checkpoint, 4369, 158859264ull),
  };
  CHECK(check_multiturn_presuffix_restores(bounds, good));
  CHECK(check_presuffix_followup_restore(4204, good[1]).passed);

  // Negative 1: silent reuse=none recompute on any follow-up must fail.
  auto bad_none = good;
  bad_none[1] = mk(Reuse::None, 0, 0ull);
  CHECK(!check_multiturn_presuffix_restores(bounds, bad_none));

  // Negative 2: Reuse::Extend instead of Reuse::Checkpoint must fail.
  CHECK(!check_presuffix_followup_restore(4204, mk(Reuse::Extend, 4204, 158859264ull)).passed);

  // Negative 3: wrong reused token count (off by 1) must fail.
  CHECK(!check_presuffix_followup_restore(4204, mk(Reuse::Checkpoint, 4203, 158859264ull)).passed);
  CHECK(!check_presuffix_followup_restore(4204, mk(Reuse::Checkpoint, 4209, 158859264ull)).passed);

  // Negative 4: zero or missing restored_bytes must fail.
  CHECK(!check_presuffix_followup_restore(4204, mk(Reuse::Checkpoint, 4204, 0ull)).passed);
  CHECK(!check_presuffix_followup_restore(4204, mk(Reuse::Checkpoint, 4204, std::nullopt)).passed);

  // Negative 5: missing prior pre-suffix boundary or < 2 turns must fail.
  CHECK(!check_presuffix_followup_restore(0, mk(Reuse::Checkpoint, 0, 158859264ull)).passed);
  CHECK(!check_multiturn_presuffix_restores(std::span<const std::size_t>(bounds.data(), 1),
                                            std::span<const GenerateStats>(good.data(), 1)));
}

}  // namespace

int main() {
  test_inkling_loop_and_qwen_detection();
  test_resolve_qwen38_runtime_options();
  test_883_followup_restore_predicate();

  const std::filesystem::path qwen_real("/home/orwel/dev_genius/models/Qwen3.8-27B-NVFP4-482ca0f");
  if (!std::filesystem::is_directory(qwen_real)) {
    std::printf("skip: Qwen3.8-27B-NVFP4-482ca0f not found at %s\n", qwen_real.c_str());
    return 77;
  }
  test_root_prefix_tokens_safety(qwen_real);
  test_qwen35_pre_suffix_checkpoint(qwen_real);
  test_qwen_fake_http_server(qwen_real);

  std::printf("qwen38_gate4_dispatch: PASS\n");
  return 0;
}

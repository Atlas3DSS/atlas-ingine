#include "text/tokenizer.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#define CHECK(cond)                                                                         \
  do {                                                                                      \
    if (!(cond)) {                                                                          \
      std::fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);         \
      std::exit(1);                                                                         \
    }                                                                                       \
  } while (0)

namespace {

using ingine::text::Tokenizer;
using Json = nlohmann::json;

std::string utf8_from_cp(std::int32_t cp) {
  std::string out;
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else {
    out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

std::string byte_to_symbol(int b) {
  static const auto map = [] {
    std::array<int, 256> m{};
    int next = 256;
    for (int i = 0; i < 256; ++i) {
      const bool vis = (i >= 33 && i <= 126) || (i >= 161 && i <= 172) || (i >= 174 && i <= 255);
      m[static_cast<std::size_t>(i)] = vis ? i : next++;
    }
    return m;
  }();
  return utf8_from_cp(map[static_cast<std::size_t>(b)]);
}

std::string bytes_to_gpt2_symbols(std::string_view s) {
  std::string out;
  for (unsigned char b : s) out += byte_to_symbol(b);
  return out;
}

std::string make_test_tokenizer_json(std::string_view regex, bool nfc_normalizer, bool ignore_merges,
                                     const std::vector<std::string>& extra_words,
                                     const std::vector<std::string>& merges = {}) {
  Json vocab = Json::object();
  int next_id = 0;
  for (int b = 0; b < 256; ++b) vocab[byte_to_symbol(b)] = next_id++;
  for (const std::string& w : extra_words) {
    const std::string sym = bytes_to_gpt2_symbols(w);
    if (!vocab.contains(sym)) vocab[sym] = next_id++;
  }

  Json added = Json::array();
  for (const auto& [content, is_sp] : std::vector<std::pair<std::string, bool>>{
           {"<|endoftext|>", true}, {"<|user|>", true}, {"<|im_start|>", true}, {"<think>", false}}) {
    added.push_back({{"id", next_id++},
                     {"content", content},
                     {"special", is_sp},
                     {"single_word", false},
                     {"lstrip", false},
                     {"rstrip", false},
                     {"normalized", false}});
  }

  Json root = {
      {"version", "1.0"},
      {"normalizer", nfc_normalizer ? Json({{"type", "NFC"}}) : Json(nullptr)},
      {"pre_tokenizer",
       {{"type", "Sequence"},
        {"pretokenizers",
         Json::array(
             {{{"type", "Split"}, {"pattern", {{"Regex", regex}}}, {"behavior", "Isolated"}, {"invert", false}},
              {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", false}}})}}},
      {"decoder", {{"type", "ByteLevel"}, {"add_prefix_space", true}, {"trim_offsets", true}, {"use_regex", true}}},
      {"added_tokens", added},
      {"model",
       {{"type", "BPE"},
        {"dropout", nullptr},
        {"unk_token", nullptr},
        {"continuing_subword_prefix", nullptr},
        {"end_of_word_suffix", nullptr},
        {"fuse_unk", false},
        {"byte_fallback", false},
        {"ignore_merges", ignore_merges},
        {"vocab", vocab},
        {"merges", merges}}},
  };
  return root.dump();
}

}  // namespace

int main() {
  constexpr std::string_view kO200kRegex =
      "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
      "|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
      "|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  constexpr std::string_view kQwen2Regex =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
      "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+"
      "|\\p{N}"
      "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*"
      "|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  constexpr std::string_view kChatGlm4Regex =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}"
      "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

  const std::vector<std::string> words = {"hello", "987", "654", "321", "'s", "fo", "foo", "a",
                                          "\xCC\x81" "b", "a\xCC\x81" "b", "!\n", "/a", "!\n/",
                                          "é", "Café"};
  const std::vector<std::string> qwen2_merges = {
      bytes_to_gpt2_symbols("'") + " " + bytes_to_gpt2_symbols("s"),
      bytes_to_gpt2_symbols("f") + " " + bytes_to_gpt2_symbols("o"),
      bytes_to_gpt2_symbols("fo") + " " + bytes_to_gpt2_symbols("o"),
  };

  // 1. O200k (Inkling): ignore_merges=true, Normalizer=None, slashes folded, 3-digit grouping.
  const Tokenizer o200k = Tokenizer::from_json(make_test_tokenizer_json(kO200kRegex, false, true, words));
  CHECK(o200k.split_mode() == Tokenizer::SplitMode::O200k);
  CHECK(o200k.normalizer_kind() == Tokenizer::NormalizerKind::None);
  CHECK(o200k.encode("hello", false).size() == 1);
  CHECK(o200k.encode("9876543210", false).size() == 4);
  {
    const auto slash = o200k.encode("!\n/a", false);
    CHECK(slash.size() == 2 && o200k.token_bytes(slash[0]) == "!\n/" && o200k.token_bytes(slash[1]) == "a");
  }

  // 2. Qwen2 + independent NFC normalizer: ignore_merges=false + explicit BPE merges, single-digit grouping,
  // contraction-first, NFC composition.
  const Tokenizer qwen2 =
      Tokenizer::from_json(make_test_tokenizer_json(kQwen2Regex, true, false, words, qwen2_merges));
  CHECK(qwen2.split_mode() == Tokenizer::SplitMode::Qwen2);
  CHECK(qwen2.normalizer_kind() == Tokenizer::NormalizerKind::Nfc);
  CHECK(qwen2.encode("hello", false).size() == 5);
  CHECK(qwen2.encode("9876", false).size() == 4);
  CHECK(qwen2.encode("Cafe\xCC\x81", false) == qwen2.encode("Caf\xC3\xA9", false));
  {
    const auto cs = qwen2.encode("'sfoo", false);
    CHECK(cs.size() == 2 && qwen2.token_bytes(cs[0]) == "'s" && qwen2.token_bytes(cs[1]) == "foo");
  }

  // 3. ChatGlm4 (GLM-5.3-Flash): 3-digit grouping, contraction-first, excludes \p{M} from \p{L}+, no '/' fold.
  const Tokenizer glm4 = Tokenizer::from_json(make_test_tokenizer_json(kChatGlm4Regex, false, true, words));
  CHECK(glm4.split_mode() == Tokenizer::SplitMode::ChatGlm4);
  CHECK(glm4.normalizer_kind() == Tokenizer::NormalizerKind::None);
  CHECK(glm4.encode("9876543210", false).size() == 4);
  {
    const std::string comb = "a\xCC\x81" "b";
    CHECK(glm4.encode(comb, false).size() == 2);
    CHECK(o200k.encode(comb, false).size() == 1);
  }
  {
    const auto slash = glm4.encode("!\n/a", false);
    CHECK(slash.size() == 2 && glm4.token_bytes(slash[0]) == "!\n" && glm4.token_bytes(slash[1]) == "/a");
  }
  {
    const std::string rt = "<|user|>Café 987654 'sfoo 你好 🧪\n";
    CHECK(glm4.decode(glm4.encode(rt, true)) == rt);
    CHECK(qwen2.decode(qwen2.encode(rt, true)) == rt);
    CHECK(qwen2.decode(qwen2.encode("<|user|>Cafe\xCC\x81 987654 'sfoo 你好 🧪\n", true)) == rt);
  }

  // 4. Reject unknown split regex or unsupported normalizer fail-closed.
  for (const auto& [regex, nfc] : std::vector<std::pair<std::string_view, bool>>{{".*", false}}) {
    bool rejected = false;
    try {
      (void)Tokenizer::from_json(make_test_tokenizer_json(regex, nfc, true, words));
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    CHECK(rejected);
  }

  std::printf("backend_multifamily_tokenizer: PASS\n");
  return 0;
}

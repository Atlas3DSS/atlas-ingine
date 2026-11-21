#!/usr/bin/env python3
"""Text goldens for the C++ text stack (tokenizer, chat rendering, StreamParser).

Run with the reference venv, CPU only:

    tools/.venv/bin/python native/tools/goldens_text.py [--model /mnt/data/inkling/hf]

Writes native/tests/golden/text/{tokenizer,chat,stream}.json. Every expected value comes from the Python engine
(``engine/ingine/chat.py``) and the Hugging Face tokenizer it uses.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine"))

from ingine import chat  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OUT = ROOT / "native" / "tests" / "golden" / "text"


# ---------------------------------------------------------------------------------------------------------------
# Tokenizer strings
# ---------------------------------------------------------------------------------------------------------------

ATOMS = [
    "a", "A", "ab", "AB", "Ab", "aB", "hello", "Hello", "HELLO", "hELLO", "x", "Z",
    "1", "12", "123", "1234", "12345", "0", "007",
    " ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", "\r", " \n", "\n ", "\t\n", " \r\n ",
    "\u00a0", "\u3000", "\u2028", "\u2029", "\x0b", "\x0c", "\x1c", "\x1f", "\x85", "\u180e", "\u200b", "\ufeff",
    "\u2009", "\u202f",
    "'s", "'S", "'ll", "'LL", "'Ll", "'re", "'RE", "'ve", "'m", "'M", "'d", "'D", "'t", "'T", "'x", "'", "\u017f",
    "'\u017f", "\u2019s",
    "!", "?!", "...", ".", ",", ";", ":", "/", "//", "/\n", "\\", "-", "--", "_", "__init__", "(", ")", "[]",
    "{", "}", "\"", "$", "%", "&&", "||", "->", "=>", "#", "@", "`", "```", "~", "^", "*", "+", "=", "<", ">",
    "é", "É", "ß", "ǅ", "ǈ", "ʰ", "ʼ", "ᵃ", "中", "文", "日本", "한국어", "ا", "ب", "שלום", "ñ", "Ω", "ω",
    "e\u0301", "E\u0301", "\u0301", "\u0903", "\u20dd", "\u0300\u0301", "a\u0308\u0304",
    "٣", "٤٥", "Ⅻ", "ⅻ", "½", "²", "³", "१२३", "໑",
    "😀", "👍🏽", "🇺🇸", "👨\u200d👩\u200d👧", "\u200d", "\ufe0f", "❤️", "€", "™", "©", "°", "→", "∑",
    "𝐀", "𝐚", "𝟏", "\U0001f600\U0001f600",
    "<|end_message|>", "<|message_user|>", "<|content_text|>", "<|", "|>", "<|end_message", "end_message|>",
    "<|unused|>", "<|unused_200007|>", "<|endoftext|>",
    "\ufffd", "\\x80", "\\u0000", "\x00", "\x01", "\x7f", "\u00ad", "\U0010ffff", "\ue000",
]

HANDCRAFTED = [
    "", " ", "  ", "a", "Hello, world!", "Hello world", "hello world", "HELLO WORLD",
    "I'm here, you're there, we've been, they'll go, he'd say, it's fine, don't.",
    "I'M HERE, YOU'RE THERE, WE'VE BEEN, THEY'LL GO, HE'D SAY, IT'S FINE, DON'T.",
    "IT'S", "it's", "It'S", "Itſ'ſ", "can't won't shan't", "rock'n'roll", "'s", "''s", "a's's", "ab'sx",
    "the cat's hat's color", "O'Reilly", "MacDonald's", "CamelCaseWord", "camelCaseWord", "snake_case_word",
    "SCREAMING_SNAKE_CASE", "kebab-case-word", "XMLHttpRequest", "iPhone", "eBay", "McDonald", "HTTPServer2",
    "1", "12", "123", "1234", "12345", "123456", "1234567", "3.14159", "1,000,000", "-42", "+7", "1e10", "0x1F",
    "v1.2.3", "2026-11-20", "12:34:56", "$100.00", "50%", "1st 2nd 3rd 4th", "a1b2c3", "abc123def",
    "\n", "\n\n", "\n\n\n", "\r\n", "\r\n\r\n", "\r", " \n", "  \n", "\n  ", "\n  x", "x\n  ", "x  ", "x   y",
    "x \ty", "\t\t", "\t\tx", "    def f():\n        return 1\n", "a\n\nb", "a \n b", "a\r\nb", "a\rb",
    "  leading", "trailing  ", "   ", "\u00a0x", "x\u00a0", "x\u00a0\u00a0y", "\u3000\u3000中文",
    "def main():\n    print(\"hello\")\n\nif __name__ == \"__main__\":\n    main()\n",
    "for (int i = 0; i < n; ++i) { sum += a[i] * b[i]; }",
    "#include <vector>\nint main() { std::vector<int> v{1, 2, 3}; return v.size(); }\n",
    "const x = () => { return {a: 1, b: [2, 3]}; };",
    "SELECT * FROM users WHERE id = 42 AND name LIKE '%bob%';",
    "<div class=\"main\"><p>Hello &amp; welcome</p></div>",
    "{\"name\": \"get_weather\", \"args\": {\"city\": \"Paris\", \"days\": 3}}",
    "https://example.com/path/to/page?query=1&b=2#frag",
    "path/to/file.txt", "C:\\Users\\name\\file.txt", "a/b/c/", "//comment", "/*\n * block\n */",
    "x // y\n", "a\\\nb", "===", "!==", "<<=", ">>>", "...\n", "?!?!", "--\n--", "**bold**", "__init__",
    "# Heading\n\n- item 1\n- item 2\n\n> quote\n", "| a | b |\n|---|---|\n| 1 | 2 |\n",
    "Γειά σου Κόσμε", "Привет, мир! КАК ДЕЛА?", "שלום עולם", "مرحبا بالعالم", "नमस्ते दुनिया",
    "สวัสดีชาวโลก", "こんにちは世界", "カタカナ", "你好，世界！", "안녕하세요 세계", "வணக்கம் உலகம்",
    "გამარჯობა მსოფლიო", "Բարեւ աշխարհ", "Ἀθῆναι", "Straße", "ǅemal ǈubljana", "ʰello ʼtwas",
    "Z\u0351\u036b\u0343\u036a\u0302\u036b\u033d\u034f\u0334\u0319\u0324\u031e\u0349\u035a\u032f\u031e\u0320\u034dA\u036b\u0357\u0334\u0362\u0335\u031c\u0330\u0354L\u0368\u0367\u0369\u0358\u0320G\u0311\u0357\u030e\u0305\u035b\u0341\u0334\u033b\u0348\u034d\u0354\u0339O\u0342\u030c\u030c\u0358\u0328\u0335\u0339\u033b\u031d\u0333",
    "e\u0301e\u0301e\u0301", "\u0301abc", "\u0301ABC", " \u0301", "1\u0301", "a\u0301B", "A\u0301b",
    "😀😃😄😁", "👍🏽 nice", "🇺🇸🇫🇷🇩🇪", "👨\u200d👩\u200d👧\u200d👦 family", "I ❤️ NY", "→ ← ↑ ↓", "∑∫∂√∞",
    "𝐇𝐞𝐥𝐥𝐨", "𝕳𝖊𝖑𝖑𝖔", "𝟏𝟐𝟑", "Ⅻ ⅻ ½ ² ³", "١٢٣٤٥", "१२३४", "٣٤", "໑໒໓",
    "\x00\x01\x02", "tab\x0bvt\x0cff", "\x1c\x1d\x1e\x1f", "nel\x85x", "zw\u200bsp", "bom\ufeffx", "soft\u00adhy",
    "\ufffd\ufffd", "\\x80\\xff", "\U0010ffff\ue000", "\u2028line\u2029para",
    "<|message_user|>Hello<|end_message|>", "<|message_model|><|content_text|>Hi there<|end_message|>",
    "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>",
    "text<|end_message|>", "<|end_message|>text", "<|end_message|><|end_message|>", "<|<|end_message|>|>",
    "<|end_message", "<|end_message|", "end_message|>", "<|unknown_token|>", "<|unused|><|unused_200007|>",
    "<|endoftext|>", "a<|content_model_end_sampling|>b", "  <|content_text|>  ", "\n<|end_message|>\n",
    "<|message_tool|>get_weather<|content_text|>{\"temp\": 21}<|end_message|>",
    "<|message_model|>search<|content_invoke_tool_json|>{\"name\":\"search\",\"args\":{\"q\":\"x\"}}<|end_message|>",
    "<|begin_of_text|>", "<|content_xml|>", "<|content_tool_error|>", "<|content_invoke_tool_text|>",
    "<|content_image|><|unused_200054|>", "<|MESSAGE_USER|>", "< |end_message|>",
    "a" * 300, "A" * 300, "1" * 100, " " * 100, "\n" * 50, "ab " * 100, "é" * 100, "中" * 100, "😀" * 30,
    "-" * 80, "=" * 80, "\t" * 20 + "x", " " * 33 + "x", "\u0301" * 50, ("x" * 7 + " ") * 40,
]


def corpus_chunks() -> list[str]:
    """Prose and code from the repository, cut into chunks of 40..400 characters."""
    rng = random.Random(1234)
    out = []
    for rel in ("engine/ingine/chat.py", "native/DESIGN.md", "engine/ingine/server.py", "README.md"):
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for _ in range(25):
            n = rng.randint(40, 400)
            i = rng.randint(0, max(0, len(text) - n))
            out.append(text[i:i + n])
    return out


def random_strings(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return ["".join(rng.choice(ATOMS) for _ in range(rng.randint(1, 7))) for _ in range(n)]


def random_codepoint_strings(n: int, seed: int) -> list[str]:
    """Random assigned codepoints from many scripts and categories (regex classes, BPE fallbacks)."""
    import unicodedata
    rng = random.Random(seed)
    ranges = [(0x20, 0x7E), (0xA0, 0x24F), (0x250, 0x2FF), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x52F),
              (0x590, 0x6FF), (0x900, 0x97F), (0xE00, 0xE7F), (0x1100, 0x11FF), (0x1E00, 0x1FFF), (0x2000, 0x206F),
              (0x2070, 0x218F), (0x2190, 0x2BFF), (0x3000, 0x30FF), (0x4E00, 0x4FFF), (0xAC00, 0xAD00),
              (0xFB00, 0xFFFD), (0x1D400, 0x1D7FF), (0x1F300, 0x1FAFF), (0x10000, 0x1007F), (0xE0000, 0xE007F)]
    out = []
    while len(out) < n:
        chars = []
        for _ in range(rng.randint(1, 12)):
            lo, hi = rng.choice(ranges)
            cp = rng.randint(lo, hi)
            if unicodedata.category(chr(cp)) in ("Cn", "Cs"):
                continue
            chars.append(chr(cp))
        if chars:
            out.append("".join(chars))
    return out


def tokenizer_goldens(tok, special_ids: list[int]) -> dict:
    backend = tok.backend_tokenizer
    strings = (HANDCRAFTED + corpus_chunks() + random_strings(1200, 7) + list(ATOMS) + random_codepoint_strings(400, 11))
    seen, uniq = set(), []
    for s in strings:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    encodes = []
    for s in uniq:
        ids = tok.encode(s, add_special_tokens=False)
        backend.encode_special_tokens = True
        try:
            plain = backend.encode(s, add_special_tokens=False).ids
        finally:
            backend.encode_special_tokens = False
        dec = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        encodes.append({"text": s, "ids": ids, "plain_ids": plain, "decode": dec})

    # Decodes of arbitrary id sequences: partial UTF-8 (slices of byte-level tokens), special and unknown ids.
    rng = random.Random(99)
    pool = [i for e in encodes for i in e["ids"]]
    decodes = []
    for k in range(200):
        if k < 60:
            e = rng.choice(encodes)
            ids = e["plain_ids"]
            if not ids:
                continue
            a = rng.randint(0, len(ids) - 1)
            b = rng.randint(a + 1, len(ids))
            ids = ids[a:b]
        elif k < 120:
            ids = [rng.choice(pool) for _ in range(rng.randint(1, 8))]
        elif k < 170:
            ids = [rng.randrange(0, 199998) for _ in range(rng.randint(1, 6))]
        else:
            ids = [rng.choice(pool + special_ids + [199998, 200058, 200063, 210000]) for _ in range(rng.randint(1, 6))]
        decodes.append({"ids": ids, "decode": tok.decode(ids, skip_special_tokens=False,
                                                         clean_up_tokenization_spaces=False)})
    added = [{"id": i, "content": t.content, "special": t.special} for i, t in sorted(tok.added_tokens_decoder.items())]
    return {"encodes": encodes, "decodes": decodes, "added_tokens": added}


# ---------------------------------------------------------------------------------------------------------------
# Chat renders
# ---------------------------------------------------------------------------------------------------------------

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
                                                     "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}
SEARCH_TOOL = {"type": "function", "function": {
    "name": "search", "description": "Search the web — returns snippets.",
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer", "default": 5},
                                                     "z_score": {"type": "number", "minimum": -1.5}},
                   "required": ["q"]}}}
BARE_TOOL = {"name": "ping"}
NOTYPE_TOOL = {"function": {"name": "noop", "description": "", "parameters": {}}}


def U(text):
    return {"role": "user", "content": text}


def S(text):
    return {"role": "system", "content": text}


def A(text=None, **kw):
    m = {"role": "assistant"}
    if text is not None:
        m["content"] = text
    m.update(kw)
    return m


def TC(name, args, call_id="call_1"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def chat_requests() -> list[dict]:
    R = []

    def add(messages, tools=None, effort=None, enable_thinking=True, kwargs=None, note=""):
        R.append({"note": note, "messages": messages, "tools": tools, "reasoning_effort": effort,
                  "enable_thinking": enable_thinking, "kwargs": kwargs or {}})

    add([U("Hello!")], note="single user")
    add([S("You are helpful."), U("Hi")], note="system + user")
    add([{"role": "developer", "content": "Be terse."}, U("Hi")], note="developer -> system")
    add([U("Hi")], enable_thinking=False, note="no thinking prefill")
    for lvl in chat.EFFORT_LEVELS:
        add([U("Explain gravity.")], effort=lvl, note=f"effort {lvl}")
    for lvl in (" HIGH ", "Max", "\tlow\n"):
        add([U("x")], effort=lvl, note="effort string normalisation")
    for num in (0, 0.0, 0.5, 0.99, 0.25, 1e-05, 0.0001, 0.3, 0.1 + 0.2, 1, "0.5", " 0.75 ", "1e-3", "0"):
        add([U("Compute.")], effort=num, note=f"effort numeric {num!r}")
    for bad in (1.5, -0.1, "extreme", True, False, "nan", "inf", "", [], {"a": 1}, 2, "1.0", "-0.0"):
        add([U("x")], effort=bad, note=f"effort invalid {bad!r}")
    add([S("sys"), S("sys2"), U("q")], note="two system messages")
    add([S("only system")], note="system only: effort at end")
    add([U("Q1"), A("A1"), U("Q2"), A("A2"), U("Q3")], note="multi-turn")
    add([U("Q1"), A("A1", reasoning_content="thought 1"), U("Q2")], note="assistant reasoning_content")
    add([U("Q1"), A("A1", reasoning="via alias"), U("Q2")], note="assistant reasoning alias")
    add([U("Q1"), A("A1", reasoning_content="", reasoning="alias wins over empty"), U("Q2")], note="empty rc + alias")
    add([U("Q1"), A("A1", thinking="thinking field"), U("Q2")], note="thinking field alias (transformers)")
    add([U("Q1"), A("", reasoning_content="only thoughts"), U("Q2")], note="empty content with reasoning")
    add([U("Q1"), A(None, reasoning_content="only thoughts"), U("Q2")], note="no content with reasoning")
    add([U("Q1"), A(""), U("Q2")], note="empty assistant content")
    add([U("What's the weather in Paris?"),
         A(None, tool_calls=[TC("get_weather", "{\"city\": \"Paris\", \"unit\": \"c\"}")]),
         {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"}, A("It is 21C.")],
        tools=[WEATHER_TOOL], note="tool round trip, string args")
    add([U("Search"), A("", tool_calls=[TC("search", {"q": "ünïcödé", "n": 3, "z_score": -0.5, "nested": {"b": 1, "a": [1, 2.5, None, True]}})])],
        tools=[SEARCH_TOOL, WEATHER_TOOL], note="dict args, nested, sorted top-level")
    add([U("x"), A("Let me check.", reasoning_content="need tool",
                   tool_calls=[TC("search", "{\"q\":\"a\"}", "c1"), TC("get_weather", "{\"city\":\"B\"}", "c2")]),
         {"role": "tool", "tool_call_id": "c2", "content": "sunny"},
         {"role": "tool", "name": "search", "content": "results"}],
        tools=[SEARCH_TOOL, WEATHER_TOOL], note="parallel tool calls, tool name by id and by name")
    add([U("x"), A(None, tool_calls=[TC("f", "")])], note="empty string args -> {}")
    add([U("x"), A(None, tool_calls=[TC("f", None)])], note="null args -> {}")
    add([U("x"), A(None, tool_calls=[TC("f", "{\"b\": 1, \"a\": 2, \"B\": 3, \"_\": 4}")])], note="dictsort case sensitive")
    add([U("x"), A(None, tool_calls=[TC("f", "{\"x\": 1.0, \"y\": 1e20, \"z\": -0.0, \"w\": 12345678901234, \"v\": 1.5e-7}")])],
        note="numeric args formatting")
    add([U("x"), A(None, tool_calls=[TC("f", "{\"a\": 1, \"a\": 2}")])], note="duplicate keys")
    add([U("x"), A(None, tool_calls=[TC("f", "{\"s\": \"quote\\\" backslash\\\\ nl\\n tab\\t ctrl\\u0001 é 😀 \\u00e9\"}")])],
        note="string escapes in args")
    add([U("x"), A(None, tool_calls=[TC("f", "{\"a\": NaN, \"b\": Infinity}")])], note="NaN/Infinity args")
    add([U("x"), A(None, name="f", tool_calls=[TC("g", "{}")])], note="assistant name mismatch -> error")
    add([U("x"), A("hi", name="bot")], note="assistant name in header")
    add([U("x"), {"role": "tool", "content": "r"}], note="tool without name or id")
    add([U([{"type": "text", "text": "part one"}, {"type": "input_text", "text": "part two"}, "raw string part", {"text": "no type"}])],
        note="content as parts")
    add([U([{"type": "text"}])], note="text part without text")
    add([U([{"type": "text", "text": 5}])], note="non-string text part")
    add([U([])], note="empty parts list")
    add([U([{"type": None, "text": "t"}])], note="part type null -> template error")
    add([U([{"type": "image_url", "image_url": {"url": "x"}}])], note="image part -> ChatError")
    add([U(["a", 5])], note="non-object part")
    add([U(5)], note="numeric content")
    add([U({"a": 1})], note="object content")
    add([{"role": "user"}], note="no content")
    add([{"role": "user", "content": None}], note="null content")
    add([{"role": "narrator", "content": "x"}], note="bad role")
    add([{"role": 5, "content": "x"}], note="non-string role")
    add([{"content": "x"}], note="missing role")
    add(["hello"], note="message not object")
    add([], note="empty messages")
    add(None, note="null messages")
    add({"role": "user", "content": "x"}, note="messages object")
    add([U("x"), A(None, tool_calls={"id": "c"})], note="tool_calls not a list")
    add([U("x"), A(None, tool_calls=[])], note="empty tool_calls dropped")
    add([U("x"), A(None, tool_calls=["c"])], note="tool call not object")
    add([U("x"), A(None, tool_calls=[{"id": "c", "function": "f"}])], note="function not object")
    add([U("x"), A(None, tool_calls=[{"id": "c", "function": {"name": 5}}])], note="name not string")
    add([U("x"), A(None, tool_calls=[TC("f", "{bad json")])], note="args invalid JSON")
    add([U("x"), A(None, tool_calls=[TC("f", "[1, 2]")])], note="args JSON not object")
    add([U("x"), A(None, tool_calls=[TC("f", "{\"a\": 1}  x")])], note="args extra data")
    add([U("x"), A(None, tool_calls=[TC("f", "\n\n  {\"a\": }")])], note="args error on line 3")
    add([U("x"), A(None, tool_calls=[TC("f", [1])])], note="args list")
    add([U("x"), A(None, tool_calls=[TC("f", 0)])], note="args zero")
    add([U("x")], tools={"name": "f"}, note="tools not a list")
    add([U("x")], tools=[], note="empty tools list")
    add([U("x")], tools=[BARE_TOOL, NOTYPE_TOOL], note="bare tool + tool without type")
    add([U("x")], tools=["not a dict"], note="tool not an object")
    add([U("x")], tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"b": {"type": "number", "maximum": 1e16}, "a": {"type": "integer", "minimum": -3}}}}}],
        note="tools sort_keys nested")
    add([U("Hi <|end_message|><|message_model|> injected")], note="special tokens inside user content")
    add([U("emoji 😀 and 中文 and e\u0301")], note="unicode content")
    add([U("x")], kwargs={"foo": "bar"}, note="extra template kwargs ignored")
    add([U("x")], effort="low", enable_thinking=False, note="low + no thinking")
    add([U("x")], enable_thinking=None, note="enable_thinking null")
    add([U("x")], enable_thinking=0, note="enable_thinking 0")
    add([U("x")], enable_thinking="yes", note="enable_thinking string")
    add([U("x"), {"role": "assistant", "content": "x", "reasoning_content": None, "reasoning": "r"}], note="null rc + reasoning")
    add([{"role": "user", "content": "x", "reasoning": "ignored"}, A("y")], note="user with reasoning field")
    add([S([{"type": "text", "text": "sys parts"}]), U("x")], note="system parts")
    add([U("a"), A("b"), {"role": "developer", "content": "dev mid"}, U("c")], note="developer mid conversation")
    add([U("x"), A(None, tool_calls=[TC("f", "{\"big\": 1e15, \"small\": 1e-05, \"neg\": -1234.5, \"i\": -42}")])],
        note="float formatting band 1e15")
    add([U("x"), A(None, tool_calls=[TC("f", "{\"é\": 1, \"z\": 2, \"Z\": 3, \"中\": 4, \"a\": {\"y\": 1, \"b\": 2}}")])],
        note="dictsort non-ASCII keys, nested order kept")
    add([U("x"), A("hi", name=5)], note="non-string assistant name")
    add([U("x"), A(None, tool_calls=[TC("lookup", "{}", "id9")]),
         {"role": "tool", "name": None, "tool_call_id": "id9", "content": "r"}], note="tool name null -> id lookup")
    add([U("x"), A(None, tool_calls=[TC("lookup", "{}", "id9")]),
         {"role": "tool", "tool_call_id": "nope", "content": "r"}], note="tool id not found")
    add([U("x"), {"role": "assistant", "content": None, "tool_calls": None}, U("y")], note="assistant all null")
    add([U("x"), A("y", reasoning_content="r1", thinking="t1")], note="reasoning_content and thinking")
    add([U("x"), A(None, tool_calls=[TC("f", "{\"s\": \"<|end_message|>\"}")])], note="special token inside args")
    add([U("x")], tools=[{"type": "function", "function": {"name": "f", "description": None, "parameters": None}}],
        note="tool with null description/parameters")
    add([U("x")], tools=[{"type": "", "function": {"name": "f", "description": "d"}}], note="tool empty type")
    return R


def chat_goldens(fmt: chat.ChatFormat) -> list[dict]:
    out = []
    for req in chat_requests():
        rec = dict(req)
        try:
            p = fmt.render(req["messages"], tools=req["tools"], reasoning_effort=req["reasoning_effort"],
                           enable_thinking=req["enable_thinking"], **req["kwargs"])
            rec["result"] = {"ids": p.ids, "text": p.text, "prefill_ids": p.prefill_ids}
        except chat.ChatError as e:
            rec["error"] = {"message": str(e), "param": e.param}
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------------------------
# StreamParser
# ---------------------------------------------------------------------------------------------------------------

def delta_json(d: chat.Delta) -> dict:
    tc = None
    if d.tool_call is not None:
        tc = {"index": d.tool_call.index, "id": d.tool_call.id, "name": d.tool_call.name,
              "arguments": d.tool_call.arguments}
    return {"reasoning_content": d.reasoning_content, "content": d.content, "tool_call": tc}


def stream_goldens(fmt: chat.ChatFormat, tok) -> list[dict]:
    backend = tok.backend_tokenizer
    vocab = backend.get_vocab(with_added_tokens=False)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_tok = {b: vocab[chr(c)] for b, c in zip(bs, cs)}

    def T(s: str) -> list[int]:  # tokens with added tokens recognised
        return tok.encode(s, add_special_tokens=False)

    def B(s: str) -> list[int]:  # one token per UTF-8 byte (forces partial characters)
        return [byte_tok[b] for b in s.encode("utf-8")]

    I = fmt.ids
    MM, CT, CTH, EM, ES = (I[chat.MESSAGE_MODEL], I[chat.CONTENT_TEXT], I[chat.CONTENT_THINKING],
                           I[chat.END_MESSAGE], I[chat.END_SAMPLING])
    TJ, TT, TE = I[chat.CONTENT_INVOKE_TOOL_JSON], I[chat.CONTENT_INVOKE_TOOL_TEXT], I[chat.CONTENT_TOOL_ERROR]
    XML = tok.convert_tokens_to_ids("<|content_xml|>")
    IMG = tok.convert_tokens_to_ids("<|content_image|>")

    def tool(name, body):
        return [MM] + T(name) + [TJ] + T(body) + [EM]

    cases = []

    def add(note, tokens, prefill=(), stop=None):
        cases.append({"note": note, "tokens": list(tokens), "prefill": list(prefill), "stop": stop})

    think = [MM, CTH] + T("Let me think about this carefully. 2+2=4.") + [EM]
    answer = [MM, CT] + T("The answer is 4.") + [EM]
    add("thinking then text", think + answer + [ES])
    add("text only", answer + [ES])
    add("text only, no end_message", [MM, CT] + T("Hello there") + [ES])
    add("prefill no-thinking", T("Direct answer.") + [EM, ES], prefill=fmt.no_thinking_prefill)
    add("prefill then stop string", T("one two three four") + [EM, ES], prefill=fmt.no_thinking_prefill, stop=["three"])
    add("tool call", tool("get_weather", '{"name":"get_weather","args":{"city":"Paris","unit":"c"}}') + [ES])
    add("thinking + tool call", think + tool("search", '{"name":"search","args":{"q":"weather today"}}') + [ES])
    add("two tool calls", tool("a", '{"name":"a","args":{"x":1}}') + tool("b", '{"name":"b","args":{"y":[1,2,{"z":"}"}]}}') + [ES])
    add("tool call args bytewise", [MM] + B("f") + [TJ] + B('{"name":"f","args":{"k":"värde 😀","n":-1.5e3}}') + [EM, ES])
    add("tool call spaced JSON", tool("f", ' { "name" : "f" , "args" : { "a" : 1 } } ') + [ES])
    add("tool call header fallback (empty name)", tool("hdr", '{"name":"","args":{"a":1}}') + [ES])
    add("tool call noncanonical body", tool("hdr", '{"args":{"a":1},"name":"real"}') + [ES])
    add("tool call noncanonical args list", tool("hdr", '{"args":[1, 2.50, 1e400],"name":"real"}') + [ES])
    add("tool call invalid JSON body", tool("hdr", '{"name": broken') + [ES])
    add("tool call invalid JSON, no header", [MM, TJ] + T("not json at all") + [EM, ES])
    add("tool call name escape", tool("x", '{"name":"we\\u00efrd\\"n","args":{}}') + [ES])
    add("tool call bad name escape", tool("hdr", '{"name":"bad\\q","args":{"a":2}}') + [ES])
    add("tool call truncated (length)", [MM] + T("f") + [TJ] + T('{"name":"f","args":{"city":"Par'))
    add("tool call, missing end_message", [MM] + T("f") + [TJ] + T('{"name":"f","args":{"a":1}}') + [ES])
    add("tool args trailing whitespace", tool("f", '{"name":"f","args":{"a":1}  }  \n') + [ES])
    add("tool then text", tool("f", '{"name":"f","args":{}}') + answer + [ES])
    add("text then tool", answer + tool("f", '{"name":"f","args":{"q":"x"}}') + [ES])
    add("stop string in content", answer + [ES], stop=["is 4"])
    add("stop string not in thinking", think + answer + [ES], stop=["think"])
    add("stop string across tokens", [MM, CT] + B("abcdefgh") + [EM, ES], stop=["cdef"])
    add("multiple stop strings, earliest wins", [MM, CT] + T("alpha beta gamma delta") + [EM, ES], stop=["gamma", "beta", ""])
    add("stop string never hit (held tail flushed)", [MM, CT] + T("short text") + [EM, ES], stop=["xyz123"])
    add("stop string never hit, length cut", [MM, CT] + T("short text"), stop=["textual"])
    add("unicode stop string", [MM, CT] + B("naïve café 😀 end") + [EM, ES], stop=["😀"])
    add("stop string at very start", [MM, CT] + T("STOP here") + [EM, ES], stop=["STOP"])
    add("max length cut mid thinking", [MM, CTH] + T("thinking without end"))
    add("max length cut mid text", think + [MM, CT] + T("partial answ"))
    add("partial UTF-8 bytewise text", [MM, CT] + B("é中😀👍🏽 done") + [EM, ES])
    add("partial UTF-8 cut", [MM, CT] + B("ok 😀")[:-2])
    add("invalid UTF-8 bytes", [MM, CT] + [byte_tok[0xff], byte_tok[0x41], byte_tok[0xe4], byte_tok[0xb8]] + B("x") + [EM, ES])
    add("lone continuation byte mid text", [MM, CT] + B("a") + [byte_tok[0x80]] + B("b") + [EM, ES])
    add("tool_error and invoke_tool_text bodies", [MM, TE] + T("error body") + [EM, MM, TT] + T("tool text") + [EM, ES])
    add("xml and image bodies dropped", [MM, XML] + T("<x/>") + [EM, MM, IMG] + T("img") + [EM] + answer + [ES])
    add("text outside message ignored", T("stray text") + answer + [ES])
    add("header text before content", [MM] + T("  analysis  ") + [CT] + T("body") + [EM, ES])
    add("missing end_message between bodies", [MM, CTH] + T("t1") + [CT] + T("a1") + [EM, ES])
    add("new message_model mid body", [MM, CT] + T("first") + [MM, CT] + T("second") + [EM, ES])
    add("user token stops", answer + [I[chat.MESSAGE_USER]] + T("more"))
    add("endoftext stops", [MM, CT] + T("abc") + [I[chat.ENDOFTEXT]])
    add("system token stops mid body", [MM, CT] + T("abc") + [I[chat.MESSAGE_SYSTEM]] + T("x") + [EM])
    add("tool token stops", tool("f", '{"name":"f","args":{"a":1}}') + [I[chat.MESSAGE_TOOL]])
    add("unused special ignored", [MM, CT] + T("a") + [tok.convert_tokens_to_ids("<|unused_200007|>")] + T("b") + [EM, ES])
    add("unknown ids", [MM, CT] + T("a") + [200060, 210000] + T("b") + [EM, ES])
    add("empty generation", [])
    add("immediate end", [ES])
    add("empty bodies", [MM, CT, EM, MM, CTH, EM, ES])
    add("thinking repeated", [MM, CTH] + T("a") + [EM, MM, CTH] + T("b") + [EM] + answer + [ES])
    add("tokens after end ignored", answer + [ES] + answer)
    add("tool args with nested braces split", [MM] + T("f") + [TJ] + B('{"name":"f","args":{"o":{"p":"}"}}}') + [EM, ES])
    add("tool name only header, json no name", tool("hdrname", '{"args":{"a":1}}') + [ES])
    add("tool call body with leading ws before prefix", tool("f", '\n{"name":"f","args":{"a":"b"}}') + [ES])
    add("stop string inside tool args ignored", tool("f", '{"name":"f","args":{"s":"STOP"}}') + [ES], stop=["STOP"])
    add("stop then tool not emitted", [MM, CT] + T("x STOP y") + [EM] + tool("f", '{"name":"f","args":{}}') + [ES], stop=["STOP"])
    add("prefill thinking", T("pre-thought") + [EM] + answer + [ES], prefill=[MM, CTH])
    return cases


def run_stream(fmt: chat.ChatFormat, case: dict) -> dict:
    counter = iter(range(1000))
    p = fmt.parser(case["prefill"], stop=case["stop"], id_factory=lambda: f"call_{next(counter)}")
    steps = []
    all_deltas = []
    for t in case["tokens"]:
        ds = p.feed(t)
        all_deltas.extend(ds)
        steps.append({"deltas": [delta_json(d) for d in ds], "finished": p.finished})
    fin = p.finish()
    all_deltas.extend(fin)
    r = p.result()
    return {
        "steps": steps,
        "finish": [delta_json(d) for d in fin],
        "finish_again": [delta_json(d) for d in p.finish()],
        "num_tokens": p.num_tokens,
        "finished": p.finished,
        "result": {"reasoning_content": r.reasoning_content, "content": r.content,
                   "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in r.tool_calls],
                   "finish_reason": r.finish_reason},
        "merged": [delta_json(d) for d in chat.merge_deltas(all_deltas)],
    }


def pyjson_goldens() -> dict:
    """json.loads/json.dumps/repr/float() behaviour the C++ chat layer reproduces."""
    docs = ['', ' ', 'x', '{', '{"a"', '{"a":', '{"a":1', '{"a":1,', '{"a":1,}', '{"a" 1}', '{1:2}', '[', '[1',
            '[1,', '[1,]', '[1 2]', '"abc', '"a\\', '"a\\x"', '"a\\u12"', '"a\\u12g4"', '"a\nb"', '{"a":1}x',
            '{"a":1} ', 'nul', 'null', 'true', 'false', 'NaN', 'Infinity', '-Infinity', '-', '-x', '01', '1.', '1e',
            '1.5e+', '1.5e+3', 'tru', '{"a":tru}', '[,]', '{,}', '{"a":1 "b":2}', '{"a":}', ' \n {"a":\n x}',
            '"\\uD83D\\uDE00"', '{"a":1,"a":2,"b":3}', '1e400', '-1e400', '1e-400', '-0', '-0.0', '0.1', '1E5',
            '12345678901234567', '9223372036854775807', '-9223372036854775808', '18446744073709551615',
            '[1,2,]', '"\x7f"', '"\t"', '\ufeff{}', '{"a":[1,{"b":}]}', '"é\\q"', 'é', '[1]]', '{"a" : 1 , }',
            '"\\u00e9\\n\\t\\"\\\\\\/\\b\\f\\r"', '"\\u0001\\u001f"', '{"k": [1, 2.5, {"z": null}], "é": "ü"}',
            '"\\u1234', '"\\u12', '  [ ]  ', ' { } ', '[[[[]]]]', '{"a":{"b":{"c":[true,false,null]}}}',
            '1.0', '100.0', '1e15', '1e16', '123456789.123456789', '0.000001', '0.0001', '-2.5e-7', '3.14e+100',
            '"\U0001F600 emoji"', '{"name": "x", "args": {"q": "a\\u0000b"}}', '"a"  "b"', '\n', '\t1\t']
    loads = []
    for d in docs:
        try:
            v = json.loads(d)
            loads.append({"doc": d, "dumps": json.dumps(v, ensure_ascii=False, separators=(",", ":")),
                          "repr": repr(v)})
        except json.JSONDecodeError as e:
            loads.append({"doc": d, "error": str(e)})
    floats = [0.0, -0.0, 1.0, 0.1, 0.5, 0.99, 1e-05, 0.0001, 1e15, 1e16, 9999999999999998.0, 123.456, 1 / 3,
              -2.5e20, 1.5e-07, 5e-324, 1.7976931348623157e308, 12345678901234567890.0, 100.0, 0.30000000000000004]
    strs = ["", "a", "it's", 'say "hi"', "both ' and \"", "back\\slash", "nl\ncr\rtab\t", "\x00\x1f\x7f\x80\x9f",
            "é ü 中 😀", "\u00a0\u00ad\u2028\u200b\ufeff", "\ue000\U0010ffff", "\u0301", "'", '"']
    float_strs = ["0.5", " 0.75 ", "1e-3", "1E-3", "-0.0", "+.5", "5.", "1_0.5", "1__0", "_1", "1_", "nan", "-inf",
                  "Infinity", "iNf", "0x10", "", ".", "e5", "1e", "١٫٥", "١.٥", "\u0661", "1 2", "0.99", "1.0",
                  "infinity ", "１２"]
    return {
        "loads": loads,
        "float_repr": [{"value": f, "repr": repr(f)} for f in floats],
        "str_repr": [{"value": s, "repr": repr(s)} for s in strs],
        "parse_float": [{"text": s, "value": _try_float(s)} for s in float_strs],
    }


def _try_float(s: str):
    try:
        return repr(float(s))
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/data/inkling/hf")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    fmt = chat.ChatFormat(tok)
    OUT.mkdir(parents=True, exist_ok=True)

    special_ids = sorted(fmt.special_ids)
    tg = tokenizer_goldens(tok, special_ids)
    tg["stop_ids"] = sorted(fmt.stop_ids)
    tg["special_ids"] = special_ids
    (OUT / "tokenizer.json").write_text(json.dumps(tg, ensure_ascii=False, indent=1), encoding="utf-8")

    cg = chat_goldens(fmt)
    (OUT / "chat.json").write_text(json.dumps(cg, ensure_ascii=False, indent=1), encoding="utf-8")

    sg = []
    for case in stream_goldens(fmt, tok):
        sg.append({**case, **run_stream(fmt, case)})
    (OUT / "stream.json").write_text(json.dumps(sg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"tokenizer: {len(tg['encodes'])} encodes, {len(tg['decodes'])} decodes; chat: {len(cg)} "
          f"({sum('error' in c for c in cg)} errors); stream: {len(sg)}")
    pj = pyjson_goldens()
    (OUT / "pyjson.json").write_text(json.dumps(pj, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

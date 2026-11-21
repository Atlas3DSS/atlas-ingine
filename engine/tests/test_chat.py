"""Prompt rendering and output parsing for Inkling's chat format (CPU only).

Oracles: the checkpoint's chat template (via ``apply_chat_template``), a hand-written expected prompt,
and transformers' ``parse_response`` driven by the checkpoint's ``response_template``.

    PYTHONPATH=engine tools/.venv/bin/python -m pytest -q engine/tests/test_chat.py
"""
from __future__ import annotations

import copy
import itertools
import json
import os
from pathlib import Path

import pytest

from ingine.chat import (NO_THINKING_PREFILL, ChatError, ChatFormat, Delta, merge_deltas, normalize_messages)

MODEL_DIR = os.environ.get("INGINE_MODEL_DIR", str(Path(__file__).resolve().parents[2] / "models" / "Inkling-2-Mini"))

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {
        "q": {"type": "string"}}}}},
]

CONVERSATIONS = {
    "simple": [{"role": "user", "content": "Hello there"}],
    "system_unicode": [{"role": "system", "content": "Answer briefly."},
                       {"role": "user", "content": "Héllo 你好 🎉 𓀀"}],
    "multi_turn_reasoning": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "Simple arithmetic."},
        {"role": "user", "content": "And 3+3?"}],
    "tools": [
        {"role": "system", "content": "You can call tools."},
        {"role": "user", "content": "Weather in Paris and Tokyo?"},
        {"role": "assistant", "content": None, "reasoning_content": "Need two lookups.",
         "tool_calls": [
             {"id": "call_1", "type": "function",
              "function": {"name": "get_weather", "arguments": '{"city": "Paris", "days": 2}'}},
             {"id": "call_2", "type": "function",
              "function": {"name": "get_weather", "arguments": '{"city":"Tokyo"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 21C"},
        {"role": "tool", "tool_call_id": "call_2", "content": "rain"},
        {"role": "assistant", "content": "Paris is sunny; Tokyo is rainy."},
        {"role": "user", "content": [{"type": "text", "text": "Thanks!"}, {"type": "text", "text": "More?"}]}],
    "developer_and_reasoning_alias": [
        {"role": "developer", "content": "Be terse."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "reasoning": "greeting",
         "tool_calls": [{"id": "c", "type": "function", "function": {"name": "search", "arguments": ""}}]},
        {"role": "tool", "name": "search", "content": "nothing"},
        {"role": "user", "content": "ok"}],
}


@pytest.fixture(scope="module")
def fmt() -> ChatFormat:
    return ChatFormat.from_pretrained(MODEL_DIR)


def _canonical(messages):
    """Independent re-statement of the OpenAI -> template normalisation (dict args, system role)."""
    out = []
    for m in copy.deepcopy(messages):
        if m["role"] == "developer":
            m["role"] = "system"
        if m.get("content") is None:
            m.pop("content", None)
        if "reasoning" in m:
            m["reasoning_content"] = m.pop("reasoning")
        for tc in m.get("tool_calls", []):
            a = tc["function"]["arguments"]
            tc["function"]["arguments"] = json.loads(a) if a else {}
        out.append(m)
    return out


# ---- rendering ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(CONVERSATIONS))
@pytest.mark.parametrize("effort", [None, "none", "low", "max", 0.5, "0.25", 0, 0.99])
def test_render_matches_template(fmt, name, effort):
    msgs = CONVERSATIONS[name]
    numeric = isinstance(effort, (int, float)) or (isinstance(effort, str) and effort[0].isdigit())
    ref_effort = float(effort) if numeric else effort
    tools = TOOLS if name in ("tools", "developer_and_reasoning_alias") else None
    r = fmt.render(msgs, tools=tools, reasoning_effort=effort)
    ref_effort = float(effort) if isinstance(effort, (int, str)) and str(effort)[0].isdigit() else effort
    ref = fmt.tokenizer.apply_chat_template(_canonical(msgs), tools=tools, tokenize=False,
                                            reasoning_effort=ref_effort)
    assert r.text == ref
    assert r.ids == fmt.tokenizer.encode(ref, add_special_tokens=False)
    assert r.prefill_ids == []
    assert not r.text.endswith("<|message_model|>")  # the model opens its own message


def test_render_hand_written(fmt):
    r = fmt.render(CONVERSATIONS["tools"][:5], tools=TOOLS[:1], reasoning_effort=0.5)
    expected = (
        '<|message_system|>tool_declare<|content_xml|>[{"description":"Get the weather for a city",'
        '"name":"get_weather","parameters":{"properties":{"city":{"type":"string"},"days":{"type":"integer"}},'
        '"required":["city"],"type":"object"},"type":"function"}]<|end_message|>'
        "<|message_system|><|content_text|>You can call tools.<|end_message|>"
        "<|message_system|><|content_text|>Thinking effort level: 0.5<|end_message|>"
        "<|message_user|><|content_text|>Weather in Paris and Tokyo?<|end_message|>"
        "<|message_model|><|content_thinking|>Need two lookups.<|end_message|>"
        '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Paris",'
        '"days":2}}<|end_message|>'
        '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Tokyo"}}'
        "<|end_message|><|content_model_end_sampling|>"
        "<|message_tool|>get_weather<|content_text|>sunny, 21C<|end_message|>"
        "<|message_tool|>get_weather<|content_text|>rain<|end_message|>")
    assert r.text == expected
    ids = r.ids
    assert ids[0] == fmt.ids["<|message_system|>"] and ids[-1] == fmt.ids["<|end_message|>"]


def test_render_effort_levels(fmt):
    def level(e):
        t = fmt.render([{"role": "user", "content": "x"}], reasoning_effort=e).text
        return t.split("Thinking effort level: ")[1].split("<|")[0]
    assert [level(e) for e in (None, "none", "minimal", "low", "medium", "high", "xhigh", "max", " High ",
                               0, 0.0, 0.3, "0.7")] == \
        ["0.9", "0", "0.1", "0.2", "0.7", "0.9", "0.99", "0.99", "0.9", "0", "0", "0.3", "0.7"]


def test_render_no_thinking_prefill(fmt):
    r = fmt.render([{"role": "user", "content": "hi"}], enable_thinking=False, reasoning_effort="low")
    base = fmt.render([{"role": "user", "content": "hi"}], reasoning_effort="low")
    assert r.text == base.text + NO_THINKING_PREFILL
    assert r.ids == base.ids + r.prefill_ids
    assert r.prefill_ids == [fmt.ids["<|message_model|>"], fmt.ids["<|content_text|>"]]


@pytest.mark.parametrize("kwargs,param", [
    ({"reasoning_effort": "bogus"}, "reasoning_effort"),
    ({"reasoning_effort": 1.0}, "reasoning_effort"),
    ({"reasoning_effort": -0.1}, "reasoning_effort"),
    ({"reasoning_effort": True}, "reasoning_effort"),
    ({"tools": {"not": "a list"}}, "tools"),
])
def test_render_rejects_bad_options(fmt, kwargs, param):
    with pytest.raises(ChatError) as e:
        fmt.render([{"role": "user", "content": "hi"}], **kwargs)
    assert e.value.param == param


@pytest.mark.parametrize("messages", [
    [],
    [{"role": "robot", "content": "x"}],
    [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:"}}]}],
    [{"role": "user", "content": 5}],
    [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]}],
    [{"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": "[1, 2]"}}]}],
    [{"role": "assistant", "tool_calls": [{"function": {"arguments": "{}"}}]}],
])
def test_render_rejects_bad_messages(fmt, messages):
    with pytest.raises(ChatError):
        fmt.render(messages)


def test_normalize_does_not_mutate_input():
    msgs = copy.deepcopy(CONVERSATIONS["tools"])
    normalize_messages(msgs)
    assert msgs == CONVERSATIONS["tools"]


# ---- parsing ------------------------------------------------------------------------------------------

def _ids(fmt, text):
    return fmt.tokenizer.encode(text, add_special_tokens=False)


def _stream(fmt, ids, prefill=(), stop=None):
    p = fmt.parser(prefill, stop=stop, id_factory=(f"call_{i}" for i in itertools.count()).__next__)
    deltas = []
    for t in ids:
        if p.finished:
            break
        deltas += p.feed(t)
    deltas += p.finish()
    return p, deltas


def _aggregate(deltas: list[Delta]):
    reasoning = "".join(d.reasoning_content for d in deltas)
    content = "".join(d.content for d in deltas)
    calls: dict[int, dict] = {}
    for d in deltas:
        tc = d.tool_call
        if tc is None:
            continue
        if tc.id is not None:
            assert tc.index not in calls and tc.index == len(calls), "first delta of each call must carry the id"
            calls[tc.index] = {"id": tc.id, "name": tc.name, "arguments": ""}
        assert tc.index in calls
        calls[tc.index]["arguments"] += tc.arguments
    return reasoning, content, [calls[i] for i in sorted(calls)]


THINK_TEXT = ("<|message_model|><|content_thinking|>Let me think… 2+2 = 4 🤔 𝔘ꙮ<|end_message|>"
              "<|message_model|><|content_text|>The answer is **4**. 你好 𓀀ࠀ𐍈<|end_message|>"
              "<|content_model_end_sampling|>")
TOOL_CALLS = ("<|message_model|><|content_thinking|>Two lookups.<|end_message|>"
              "<|message_model|><|content_text|>Checking…<|end_message|>"
              '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":'
              '{"city":"Pa}ris 🌧","days":3,"nested":{"a":[1,{"b":"}}"}]}}}<|end_message|>'
              '<|message_model|>search<|content_invoke_tool_json|>{"name":"search","args":{}}<|end_message|>'
              "<|content_model_end_sampling|>")
CANNED = {
    "thinking_text": THINK_TEXT,
    "tool_calls": TOOL_CALLS,
    "repeated_fields": ("<|message_model|><|content_thinking|>a<|end_message|>"
                        "<|message_model|><|content_thinking|>b<|end_message|>"
                        "<|message_model|><|content_text|>x<|end_message|>"
                        "<|message_model|><|content_text|>y<|end_message|><|content_model_end_sampling|>"),
    "tool_error_and_text_invoke": ("<|message_model|><|content_tool_error|>oops<|end_message|>"
                                   "<|message_model|>t<|content_invoke_tool_text|>raw<|end_message|>"
                                   "<|content_model_end_sampling|>"),
    "headerless_tool_whitespace": ("<|message_model|><|content_invoke_tool_json|>"
                                   '{ "name" : "search", "args" : { "q" : "x y" } }<|end_message|>'
                                   "<|content_model_end_sampling|>"),
    "noncanonical_key_order": ("<|message_model|>search<|content_invoke_tool_json|>"
                               '{"args":{"q":"z"},"name":"search"}<|end_message|><|content_model_end_sampling|>'),
    "dropped_kinds": ("<|message_model|><|content_xml|><a/><|end_message|>"
                      "<|message_model|><|content_text|>kept<|end_message|><|content_model_end_sampling|>"),
}


@pytest.mark.parametrize("name", list(CANNED))
def test_parser_matches_response_template(fmt, name):
    text = CANNED[name]
    oracle = fmt.tokenizer.parse_response(text, prefix="")
    p, deltas = _stream(fmt, _ids(fmt, text))
    res = p.result()
    assert res.reasoning_content == oracle.get("reasoning_content", "")
    assert res.content == oracle.get("content", "")
    want_calls = [(c["function"]["name"], c["function"]["arguments"]) for c in oracle.get("tool_calls", [])]
    assert [(c.name, json.loads(c.arguments)) for c in res.tool_calls] == want_calls
    assert res.finish_reason == ("tool_calls" if want_calls else "stop")
    # streamed deltas reassemble exactly into the aggregate, with no replacement characters
    reasoning, content, calls = _aggregate(deltas)
    assert (reasoning, content) == (res.reasoning_content, res.content)
    assert calls == [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in res.tool_calls]
    assert all("\ufffd" not in (d.content + d.reasoning_content) for d in deltas)
    # the non-streaming helper agrees
    assert fmt.parse(_ids(fmt, text), id_factory=(f"call_{i}" for i in itertools.count()).__next__) == res


def test_parser_streams_tool_arguments_incrementally(fmt):
    p, deltas = _stream(fmt, _ids(fmt, TOOL_CALLS))
    first = [d.tool_call for d in deltas if d.tool_call is not None and d.tool_call.index == 0]
    assert first[0].id == "call_0" and first[0].name == "get_weather" and first[0].arguments == ""
    assert len(first) > 5 and all(tc.id is None and tc.name is None for tc in first[1:])
    assert json.loads("".join(tc.arguments for tc in first)) == {
        "city": "Pa}ris 🌧", "days": 3, "nested": {"a": [1, {"b": "}}"}]}}
    assert p.result().tool_calls[1].arguments == "{}"


def test_parser_utf8_split_across_tokens(fmt):
    s = "𝔘ꙮ𓀀ࠀ𐍈"
    body = _ids(fmt, s)
    assert len(body) > len(s)  # these characters really are split into byte tokens
    ids = _ids(fmt, "<|message_model|><|content_text|>") + body + _ids(fmt, "<|end_message|>")
    p = fmt.parser()
    pieces = []
    for t in ids:
        pieces += [d.content for d in p.feed(t) if d.content]
    assert "".join(pieces) == s
    assert all("\ufffd" not in x for x in pieces) and len(pieces) == len(s)


def test_parser_invalid_utf8_flushed_at_end(fmt):
    body = _ids(fmt, "𓀀")[:2]  # half a character, then the message ends
    ids = _ids(fmt, "<|message_model|><|content_text|>ok") + body + _ids(fmt, "<|end_message|>")
    res = fmt.parse(ids)
    assert res.content.startswith("ok") and res.content[2:] and set(res.content[2:]) == {"\ufffd"}


@pytest.mark.parametrize("cut_at", ["Let me", "4 🤔", "The answer", "**4**."])
def test_parser_truncated_text(fmt, cut_at):
    text = THINK_TEXT[:THINK_TEXT.index(cut_at) + len(cut_at)]
    p, deltas = _stream(fmt, _ids(fmt, text))
    oracle = fmt.tokenizer.parse_response(text, prefix="")
    res = p.result()
    assert res.finish_reason == "length"
    assert (res.reasoning_content, res.content) == (oracle.get("reasoning_content", ""), oracle.get("content", ""))
    assert _aggregate(deltas)[:2] == (res.reasoning_content, res.content)


def test_parser_truncated_tool_call(fmt):
    full = fmt.parse(_ids(fmt, TOOL_CALLS)).tool_calls[0].arguments
    ids = _ids(fmt, TOOL_CALLS)
    end = _ids(fmt, TOOL_CALLS[:TOOL_CALLS.index('"days"')])
    p, deltas = _stream(fmt, ids[:len(end)])
    res = p.result()
    assert res.finish_reason == "length"
    assert res.tool_calls[0].name == "get_weather"
    assert full.startswith(res.tool_calls[0].arguments) and res.tool_calls[0].arguments.startswith('{"city"')
    # truncated before the args prefix completes: the fallback still reports the call by header name
    cut = _ids(fmt, TOOL_CALLS[:TOOL_CALLS.index('"args"')])
    res = fmt.parse(ids[:len(cut)])
    assert res.finish_reason == "length" and [c.name for c in res.tool_calls] == ["get_weather"]


@pytest.mark.parametrize("stop_tok", ["<|content_model_end_sampling|>", "<|message_user|>", "<|endoftext|>"])
def test_parser_stop_tokens(fmt, stop_tok):
    ids = _ids(fmt, "<|message_model|><|content_text|>hi<|end_message|>" + stop_tok + "garbage")
    p, deltas = _stream(fmt, ids)
    assert p.saw_stop_token and p.result().content == "hi" and p.finish_reason == "stop"
    assert p.num_tokens == len(ids) - len(_ids(fmt, "garbage"))
    assert stop_tok in fmt.tokenizer.convert_ids_to_tokens(sorted(fmt.stop_ids))


def test_parser_prefill(fmt):
    r = fmt.render([{"role": "user", "content": "hi"}], enable_thinking=False)
    out = _ids(fmt, "Hello!<|end_message|><|content_model_end_sampling|>")
    p, deltas = _stream(fmt, out, prefill=r.prefill_ids)
    assert p.result().content == "Hello!" and p.result().reasoning_content == ""
    assert p.num_tokens == len(out)


@pytest.mark.parametrize("stop,expect", [
    (["STOP"], "Answer: a b "),
    (["b S", "zzz"], "Answer: a "),
    (["never"], "Answer: a b STOP c"),
    (["Answer"], ""),
    (["c"], "Answer: a b STOP "),
])
def test_parser_stop_strings(fmt, stop, expect):
    text = ("<|message_model|><|content_thinking|>STOP in reasoning is ignored<|end_message|>"
            "<|message_model|><|content_text|>Answer: a b STOP c<|end_message|><|content_model_end_sampling|>")
    ids = _ids(fmt, text)
    p, deltas = _stream(fmt, ids, stop=stop)
    res = p.result()
    assert res.content == expect and res.reasoning_content == "STOP in reasoning is ignored"
    assert _aggregate(deltas)[1] == expect
    assert res.finish_reason == "stop"
    assert p.stopped_by_string == (stop != ["never"])
    if p.stopped_by_string:
        assert p.num_tokens < len(ids)


def test_parser_stop_string_split_across_tokens(fmt):
    s = "abc<END>def"
    ids = _ids(fmt, "<|message_model|><|content_text|>") + [t for ch in s for t in _ids(fmt, ch)]
    p, deltas = _stream(fmt, ids, stop=["<END>"])
    assert p.result().content == "abc" and p.finish_reason == "stop"
    assert all("<" not in d.content for d in deltas)


def test_merge_deltas(fmt):
    _, deltas = _stream(fmt, _ids(fmt, TOOL_CALLS))
    merged = merge_deltas(deltas)
    assert _aggregate(merged) == _aggregate(deltas)
    kinds = [("r" if d.reasoning_content else "c" if d.content else f"t{d.tool_call.index}") for d in merged]
    assert kinds == ["r", "c", "t0", "t1"]


def test_feeding_after_finish_is_ignored(fmt):
    p = fmt.parser()
    for t in _ids(fmt, THINK_TEXT):
        p.feed(t)
    p.finish()
    n = p.num_tokens
    assert p.feed(_ids(fmt, "x")[0]) == [] and p.num_tokens == n

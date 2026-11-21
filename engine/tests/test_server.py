"""OpenAI-compatible server end to end against FakeGenerator (CPU only, real tokenizer/template).

    PYTHONPATH=engine tools/.venv/bin/python -m pytest -q engine/tests/test_server.py
"""
from __future__ import annotations

import contextlib
import http.client
import json
import socket
import threading
import time

import pytest

from ingine.chat import ChatFormat, Sampling
from ingine.server import App, ChatServer, FakeGenerator

MODEL_DIR = "/mnt/data/inkling/hf"
MSGS = [{"role": "user", "content": "Hi 🎉"}]
THINK_TEXT = ("<|message_model|><|content_thinking|>Let me think… 𓀀 ok<|end_message|>"
              "<|message_model|><|content_text|>Hello, wörld 𝔘! STOP here<|end_message|>"
              "<|content_model_end_sampling|>")
TOOL_TEXT = ("<|message_model|><|content_thinking|>Need the weather.<|end_message|>"
             '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":'
             '{"city":"São Paulo","days":3,"opts":{"units":"metric"}}}<|end_message|>'
             '<|message_model|>search<|content_invoke_tool_json|>{"name":"search","args":{"q":"rain"}}'
             "<|end_message|><|content_model_end_sampling|>")
LONG_TEXT = "<|message_model|><|content_text|>" + "word " * 3000 + "<|end_message|><|content_model_end_sampling|>"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
         {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]
FINISH = {"stop", "length", "tool_calls"}


@pytest.fixture(scope="module")
def fmt() -> ChatFormat:
    return ChatFormat.from_pretrained(MODEL_DIR)


@contextlib.contextmanager
def serve(fmt, text=THINK_TEXT, delay=0.0, max_queue=4, max_model_len=None, responder=None):
    gen = FakeGenerator(fmt.tokenizer, model_id="inkling-test", responder=responder or (lambda ids: text),
                        token_delay=delay)
    srv = ChatServer(App(gen, fmt, max_queue=max_queue, max_model_len=max_model_len), "127.0.0.1", 0)
    srv.start()
    try:
        yield srv, gen
    finally:
        srv.close()


def _port(srv) -> int:
    return srv.server_address[1]


def request(srv, method, path, body=None, raw: bytes | None = None, timeout=20):
    conn = http.client.HTTPConnection("127.0.0.1", _port(srv), timeout=timeout)
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    out = resp.status, dict(resp.getheaders()), resp.read()
    conn.close()
    return out


def chat(srv, body, **kw):
    status, headers, data = request(srv, "POST", "/v1/chat/completions", {"messages": MSGS, **body}, **kw)
    return status, headers, json.loads(data) if not body.get("stream") or status != 200 else _sse(data)


def _sse(data: bytes) -> list:
    events = []
    for block in data.decode().split("\n\n"):
        if not block.strip():
            continue
        assert block.startswith("data: "), block
        payload = block[len("data: "):]
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


def read_raw_response(s: socket.socket) -> tuple[bytes, bytes]:
    s.settimeout(10)
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += s.recv(65536)
    head, body = buf.split(b"\r\n\r\n", 1)
    length = int(next(line.split(b":")[1] for line in head.split(b"\r\n")
                      if line.lower().startswith(b"content-length")))
    while len(body) < length:
        body += s.recv(65536)
    return head, body


def raw_request(srv, body: dict) -> socket.socket:
    data = json.dumps({"messages": MSGS, **body}).encode()
    s = socket.create_connection(("127.0.0.1", _port(srv)))
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
              b"Content-Length: %d\r\n\r\n%s" % (len(data), data))
    return s


def wait_for(cond, timeout=10.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def health(srv) -> dict:
    return json.loads(request(srv, "GET", "/health")[2])


# ---- OpenAI schema checks -----------------------------------------------------------------------------

def check_usage(u):
    assert set(u) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert all(isinstance(v, int) and v >= 0 for v in u.values())
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


def check_completion(r):
    assert isinstance(r["id"], str) and r["id"].startswith("chatcmpl-")
    assert r["object"] == "chat.completion" and isinstance(r["created"], int) and r["model"] == "inkling-test"
    assert len(r["choices"]) == 1
    c = r["choices"][0]
    assert c["index"] == 0 and c["finish_reason"] in FINISH and c["logprobs"] is None
    m = c["message"]
    assert m["role"] == "assistant"
    assert m["content"] is None or isinstance(m["content"], str)
    assert m["reasoning_content"] is None or isinstance(m["reasoning_content"], str)
    for tc in m.get("tool_calls", []):
        assert set(tc) == {"id", "type", "function"} and tc["type"] == "function" and isinstance(tc["id"], str)
        assert set(tc["function"]) == {"name", "arguments"}
        assert isinstance(tc["function"]["name"], str) and isinstance(tc["function"]["arguments"], str)
    if "tool_calls" in m:
        assert len(m["tool_calls"]) > 0
    check_usage(r["usage"])


def check_chunks(events, include_usage=False):
    """Validate a chunk stream and reassemble it into (reasoning, content, tool_calls, finish, usage)."""
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    ids = {c["id"] for c in chunks}
    assert len(ids) == 1 and next(iter(ids)).startswith("chatcmpl-")
    usage = None
    if include_usage:
        last = chunks.pop()
        assert last["choices"] == [] and last["object"] == "chat.completion.chunk"
        check_usage(last["usage"])
        usage = last["usage"]
    reasoning, content, calls, finish = "", "", {}, None
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    for i, c in enumerate(chunks):
        assert c["object"] == "chat.completion.chunk" and isinstance(c["created"], int)
        assert c["model"] == "inkling-test"
        assert ("usage" in c and c["usage"] is None) if include_usage else "usage" not in c
        (ch,) = c["choices"]
        assert ch["index"] == 0 and ch["logprobs"] is None
        d = ch["delta"]
        assert set(d) <= {"role", "content", "reasoning_content", "tool_calls"}
        if ch["finish_reason"] is not None:
            assert i == len(chunks) - 1 and ch["finish_reason"] in FINISH and d == {}
            finish = ch["finish_reason"]
            continue
        if i > 0:
            assert len(d) == 1 and "role" not in d
        reasoning += d.get("reasoning_content", "")
        content += d.get("content") or ""
        for tc in d.get("tool_calls", []):
            assert isinstance(tc["index"], int)
            if "id" in tc:
                assert tc["index"] not in calls and tc["type"] == "function"
                assert isinstance(tc["function"]["name"], str) and tc["function"]["name"]
                calls[tc["index"]] = {"id": tc["id"], "name": tc["function"]["name"], "arguments": ""}
            else:
                assert set(tc) == {"index", "function"} and set(tc["function"]) == {"arguments"}
            calls[tc["index"]]["arguments"] += tc["function"]["arguments"]
    assert finish is not None
    assert "\ufffd" not in reasoning + content
    return reasoning, content, [calls[k] for k in sorted(calls)], finish, usage


# ---- tests --------------------------------------------------------------------------------------------

def test_models_and_health(fmt):
    with serve(fmt) as (srv, _):
        status, _, data = request(srv, "GET", "/v1/models")
        body = json.loads(data)
        assert status == 200 and body["object"] == "list"
        (m,) = body["data"]
        assert m["id"] == "inkling-test" and m["object"] == "model" and isinstance(m["created"], int)
        assert isinstance(m["owned_by"], str)
        h = health(srv)
        assert h["status"] == "ok" and h["waiting"] == 0 and h["running"] is False
        assert request(srv, "GET", "/nope")[0] == 404
        assert request(srv, "POST", "/v1/completions", {})[0] == 404


def test_non_streaming(fmt):
    with serve(fmt) as (srv, gen):
        status, _, r = chat(srv, {"model": "whatever"})
        assert status == 200
        check_completion(r)
        m = r["choices"][0]["message"]
        assert m["reasoning_content"] == "Let me think… 𓀀 ok"
        assert m["content"] == "Hello, wörld 𝔘! STOP here" and "tool_calls" not in m
        assert r["choices"][0]["finish_reason"] == "stop"
        n_out = len(fmt.tokenizer.encode(THINK_TEXT, add_special_tokens=False))
        assert r["usage"]["prompt_tokens"] == len(fmt.render(MSGS).ids) == len(gen.calls[0]["prompt_ids"])
        assert r["usage"]["completion_tokens"] == n_out
        assert gen.calls[0]["prompt_ids"] == fmt.render(MSGS).ids


@pytest.mark.parametrize("include_usage", [False, True])
def test_streaming_matches_non_streaming(fmt, include_usage):
    with serve(fmt) as (srv, _):
        _, _, ref = chat(srv, {})
        status, headers, events = chat(srv, {"stream": True, "stream_options": {"include_usage": include_usage}})
        assert status == 200 and headers["Content-Type"].startswith("text/event-stream")
        reasoning, content, calls, finish, usage = check_chunks(events, include_usage)
        m = ref["choices"][0]["message"]
        assert (reasoning, content, calls, finish) == (m["reasoning_content"], m["content"], [], "stop")
        if include_usage:
            assert usage == ref["usage"]


def test_tool_call_streaming(fmt):
    with serve(fmt, TOOL_TEXT, delay=0.003) as (srv, _):  # paced so deltas are not coalesced into one chunk
        status, _, r = chat(srv, {"tools": TOOLS})
        check_completion(r)
        m = r["choices"][0]["message"]
        assert r["choices"][0]["finish_reason"] == "tool_calls" and m["content"] is None
        assert [(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in m["tool_calls"]] == [
            ("get_weather", {"city": "São Paulo", "days": 3, "opts": {"units": "metric"}}),
            ("search", {"q": "rain"})]
        _, _, events = chat(srv, {"tools": TOOLS, "stream": True, "stream_options": {"include_usage": True}})
        reasoning, content, calls, finish, usage = check_chunks(events, include_usage=True)
        assert finish == "tool_calls" and reasoning == "Need the weather." and content == ""
        assert [(c["name"], c["arguments"]) for c in calls] == \
            [(t["function"]["name"], t["function"]["arguments"]) for t in m["tool_calls"]]
        assert len({c["id"] for c in calls}) == 2 and usage == r["usage"]
        arg_chunks = [tc for e in events[:-2] for tc in e["choices"][0]["delta"].get("tool_calls", [])
                      if "id" not in tc and tc["index"] == 0]
        assert len(arg_chunks) > 1  # arguments really stream


def test_tool_round_trip_prompt(fmt):
    """A follow-up turn with OpenAI-shaped tool calls (string arguments) and tool results renders."""
    msgs = MSGS + [
        {"role": "assistant", "content": None, "reasoning_content": "r",
         "tool_calls": [{"id": "call_a", "type": "function",
                         "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]},
        {"role": "tool", "tool_call_id": "call_a", "content": "sunny"}]
    with serve(fmt) as (srv, gen):
        status, _, r = chat(srv, {"messages": msgs, "tools": TOOLS})
        assert status == 200
        text = fmt.tokenizer.decode(gen.calls[0]["prompt_ids"], skip_special_tokens=False)
        assert ('<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":'
                '{"city":"Paris"}}<|end_message|><|content_model_end_sampling|>'
                "<|message_tool|>get_weather<|content_text|>sunny<|end_message|>") in text


def test_thinking_controls(fmt):
    with serve(fmt, "Plain answer.<|end_message|><|content_model_end_sampling|>") as (srv, gen):
        for body in ({"chat_template_kwargs": {"enable_thinking": False}}, {"enable_thinking": False}):
            status, _, r = chat(srv, {**body, "reasoning_effort": 0.3})
            assert status == 200
            m = r["choices"][0]["message"]
            assert m["content"] == "Plain answer." and m["reasoning_content"] is None
            want = fmt.render(MSGS, reasoning_effort=0.3, enable_thinking=False)
            assert gen.calls[-1]["prompt_ids"] == want.ids
            assert r["usage"]["prompt_tokens"] == len(want.ids)
        chat(srv, {"reasoning_effort": "max"})
        assert gen.calls[-1]["prompt_ids"] == fmt.render(MSGS, reasoning_effort="max").ids
        chat(srv, {"chat_template_kwargs": {"reasoning_effort": "low"}})
        assert gen.calls[-1]["prompt_ids"] == fmt.render(MSGS, reasoning_effort="low").ids


def test_sampling_and_max_tokens(fmt):
    with serve(fmt) as (srv, gen):
        status, _, r = chat(srv, {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 7, "max_tokens": 5})
        assert status == 200 and r["choices"][0]["finish_reason"] == "length"
        assert r["usage"]["completion_tokens"] == 5
        assert gen.calls[-1]["sampling"] == Sampling(temperature=0.6, top_p=0.95, top_k=20, seed=7)
        assert gen.calls[-1]["max_new_tokens"] == 5
        chat(srv, {"max_completion_tokens": 3, "max_tokens": 9, "top_k": -1})
        assert gen.calls[-1]["max_new_tokens"] == 3 and gen.calls[-1]["sampling"].top_k == 0
        _, _, events = chat(srv, {"stream": True, "max_tokens": 4})
        assert check_chunks(events)[3] == "length"


def test_context_limit(fmt):
    n = len(fmt.render(MSGS).ids)
    with serve(fmt, max_model_len=n + 6) as (srv, gen):
        status, _, r = chat(srv, {})
        assert r["choices"][0]["finish_reason"] == "length" and gen.calls[-1]["max_new_tokens"] == 6
        chat(srv, {"max_tokens": 100})
        assert gen.calls[-1]["max_new_tokens"] == 6
    with serve(fmt, max_model_len=n) as (srv, _):
        status, _, r = chat(srv, {})
        assert status == 400 and r["error"]["param"] == "messages"


@pytest.mark.parametrize("stream", [False, True])
def test_stop_strings_cancel_generation(fmt, stream):
    text = "<|message_model|><|content_text|>alpha beta<END> gamma" + " delta" * 200 + "<|end_message|>"
    with serve(fmt, text, delay=0.002) as (srv, gen):
        status, _, r = chat(srv, {"stop": ["<END>", "never"], "stream": stream,
                                  "stream_options": {"include_usage": True}})
        assert status == 200
        if stream:
            _, content, _, finish, usage = check_chunks(r, include_usage=True)
        else:
            check_completion(r)
            content, finish, usage = r["choices"][0]["message"]["content"], r["choices"][0]["finish_reason"], \
                r["usage"]
        assert content == "alpha beta" and finish == "stop"
        assert usage["completion_tokens"] < 20
        wait_for(lambda: gen.calls[-1]["done"])
        assert gen.calls[-1]["cancelled"] and gen.calls[-1]["yielded"] < 50
        status, _, r = chat(srv, {"stop": "beta"})
        assert r["choices"][0]["message"]["content"] == "alpha "


@pytest.mark.parametrize("stream", [False, True])
def test_disconnect_cancels_generation(fmt, stream):
    with serve(fmt, LONG_TEXT, delay=0.002) as (srv, gen):
        s = raw_request(srv, {"stream": stream})
        wait_for(lambda: gen.calls and gen.calls[-1]["yielded"] > 20)
        if stream:
            assert s.recv(4096).startswith(b"HTTP/1.1 200")
        s.close()
        wait_for(lambda: gen.calls[-1]["done"])
        assert gen.calls[-1]["cancelled"] and gen.calls[-1]["yielded"] < 2000
        wait_for(lambda: not health(srv)["running"])
        status, _, r = chat(srv, {"max_tokens": 3})  # server still healthy
        assert status == 200 and r["usage"]["completion_tokens"] == 3


def test_queue_full_503_and_withdraw(fmt):
    with serve(fmt, LONG_TEXT, delay=0.005, max_queue=1) as (srv, gen):
        a = raw_request(srv, {"stream": True})
        wait_for(lambda: health(srv)["running"])
        b = raw_request(srv, {})
        wait_for(lambda: health(srv)["waiting"] == 1)
        status, headers, r = chat(srv, {})
        assert status == 503 and r["error"]["code"] == 503 and isinstance(r["error"]["message"], str)
        assert headers.get("Retry-After") == "1"
        b.close()  # a queued request whose client leaves is withdrawn without ever running
        wait_for(lambda: health(srv)["waiting"] == 0)
        c = raw_request(srv, {"max_tokens": 2})
        wait_for(lambda: health(srv)["waiting"] == 1)
        a.close()
        head, body = read_raw_response(c)
        c.close()
        assert head.startswith(b"HTTP/1.1 200")
        assert json.loads(body)["usage"]["completion_tokens"] == 2
        assert len(gen.calls) == 2  # a and c ran; b never reached the generator
        assert gen.calls[0]["cancelled"] and gen.calls[0]["yielded"] < 2000


def test_fifo_order(fmt):
    with serve(fmt, delay=0.001, max_queue=8) as (srv, gen):
        order = []

        def go(i):
            st, _, r = chat(srv, {"messages": [{"role": "user", "content": f"req {i}"}], "max_tokens": 4})
            order.append((i, st))

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=go, args=(i,)))
            threads[-1].start()
            wait_for(lambda: len(gen.calls) + health(srv)["waiting"] >= i + 1)
        for t in threads:
            t.join()
        assert sorted(order) == [(i, 200) for i in range(5)]
        prompts = [fmt.tokenizer.decode(c["prompt_ids"]) for c in gen.calls]
        assert [f"req {i}" in p for i, p in enumerate(prompts)] == [True] * 5


def test_generator_failure(fmt):
    def boom(ids):
        raise RuntimeError("kaboom")
    with serve(fmt, responder=boom) as (srv, _):
        status, _, r = chat(srv, {})
        assert status == 500 and "kaboom" in r["error"]["message"]
        status, _, events = chat(srv, {"stream": True})
        assert status == 200 and events[-1] == "[DONE]" and "kaboom" in events[-2]["error"]["message"]
        assert not health(srv)["running"]


@pytest.mark.parametrize("body,param", [
    ({"n": 2}, "n"),
    ({"temperature": -1}, "temperature"),
    ({"temperature": "hot"}, "temperature"),
    ({"top_p": 0}, "top_p"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"max_tokens": 1.5}, "max_tokens"),
    ({"seed": True}, "seed"),
    ({"stream": "yes"}, "stream"),
    ({"stop": 5}, "stop"),
    ({"messages": []}, "messages"),
    ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}, "messages"),
    ({"reasoning_effort": 2}, "reasoning_effort"),
    ({"chat_template_kwargs": {"enable_thinking": "no"}}, "enable_thinking"),
])
def test_bad_requests(fmt, body, param):
    with serve(fmt) as (srv, gen):
        status, _, r = chat(srv, body)
        assert status == 400 and r["error"]["param"] == param and r["error"]["type"] == "invalid_request_error"
        assert gen.calls == []


def test_invalid_json(fmt):
    with serve(fmt) as (srv, _):
        status, _, data = request(srv, "POST", "/v1/chat/completions", raw=b"{nope")
        assert status == 400 and "invalid JSON" in json.loads(data)["error"]["message"]

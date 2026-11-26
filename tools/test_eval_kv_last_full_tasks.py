#!/usr/bin/env python3
"""CPU unit tests for tools/eval_kv_last_full_tasks.py (mock OpenAI server, zero GPU needed)."""
from __future__ import annotations

import http.server
import json
import threading
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_kv_last_full_tasks as ev  # noqa: E402


GOOD_CODE = {
    "code_lru_cache": (
        "```python\n"
        "from collections import OrderedDict\n"
        "class LRUCache:\n"
        "    def __init__(self, capacity: int):\n"
        "        self.cap = capacity\n"
        "        self.d = OrderedDict()\n"
        "    def get(self, key: int) -> int:\n"
        "        if key not in self.d:\n"
        "            return -1\n"
        "        self.d.move_to_end(key)\n"
        "        return self.d[key]\n"
        "    def put(self, key: int, value: int) -> None:\n"
        "        if key in self.d:\n"
        "            self.d.move_to_end(key)\n"
        "        self.d[key] = value\n"
        "        if len(self.d) > self.cap:\n"
        "            self.d.popitem(last=False)\n"
        "```"
    ),
    "code_interval_merge": (
        "```python\n"
        "def merge_intervals(intervals):\n"
        "    if not intervals:\n"
        "        return []\n"
        "    intervals = sorted(intervals)\n"
        "    out = [intervals[0][:]]\n"
        "    for s, e in intervals[1:]:\n"
        "        if s <= out[-1][1]:\n"
        "            out[-1][1] = max(out[-1][1], e)\n"
        "        else:\n"
        "            out.append([s, e])\n"
        "    return out\n"
        "```"
    ),
    "code_topo_sort": (
        "```python\n"
        "import heapq\n"
        "def lex_topo_sort(n, edges):\n"
        "    adj = [[] for _ in range(n)]\n"
        "    indeg = [0] * n\n"
        "    for u, v in edges:\n"
        "        adj[u].append(v)\n"
        "        indeg[v] += 1\n"
        "    h = [i for i in range(n) if indeg[i] == 0]\n"
        "    heapq.heapify(h)\n"
        "    out = []\n"
        "    while h:\n"
        "        u = heapq.heappop(h)\n"
        "        out.append(u)\n"
        "        for v in adj[u]:\n"
        "            indeg[v] -= 1\n"
        "            if indeg[v] == 0:\n"
        "                heapq.heappush(h, v)\n"
        "    return out if len(out) == n else []\n"
        "```"
    ),
    "code_rpn_eval": (
        "```python\n"
        "def eval_rpn(tokens):\n"
        "    st = []\n"
        "    for t in tokens:\n"
        "        if t in ('+', '-', '*', '/'):\n"
        "            b = st.pop(); a = st.pop()\n"
        "            if t == '+': st.append(a + b)\n"
        "            elif t == '-': st.append(a - b)\n"
        "            elif t == '*': st.append(a * b)\n"
        "            else: st.append(int(a / b))\n"
        "        else:\n"
        "            st.append(int(t))\n"
        "    return st[0]\n"
        "```"
    ),
}


def start_mock_server(mode: str, cached_tokens: int = 0, fail_task_id: str | None = None):
    tasks = ev.build_task_suite()
    by_prompt = {t["messages"][0]["content"]: t for t in tasks}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def _send(self, code: int, obj: dict):
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": "inkling-mock", "kv_cache_mode": mode})
            elif self.path == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": "inkling-mock", "kv_cache_mode": mode}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            task = by_prompt[body["messages"][0]["content"]]
            tid = task["id"]
            msg: dict = {"role": "assistant"}
            if tid == fail_task_id:
                msg["content"] = "WRONG"
            elif task["category"] == "long_context_retrieval":
                msg["content"] = task["expected_substring"]
            elif task["category"] == "coding":
                msg["content"] = GOOD_CODE[tid]
            elif task["category"] == "tool_use":
                msg["content"] = None
                msg["tool_calls"] = [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": task["expected_tool"],
                            "arguments": json.dumps(task["expected_args"]),
                        },
                    }
                ]
            self._send(
                200,
                {
                    "id": f"chatcmpl-{tid}",
                    "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 2048,
                        "completion_tokens": 32,
                        "total_tokens": 2080,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                    },
                },
            )

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, f"http://127.0.0.1:{port}"


class TestEvalKvLastFullTasks(unittest.TestCase):
    def test_paired_arms_and_regression_detection(self):
        tasks = ev.build_task_suite()
        self.assertEqual(len(tasks), 10)
        srv_pp, url_pp = start_mock_server("per_pass", cached_tokens=0, fail_task_id=None)
        srv_lf, url_lf = start_mock_server("last_full", cached_tokens=0, fail_task_id="lc_needle_deep_12k")
        try:
            arm_pp = ev.run_arm_against_url(url_pp, "per_pass", tasks, gpu_uuid="unknown")
            arm_lf = ev.run_arm_against_url(url_lf, "last_full", tasks, gpu_uuid="unknown")
            self.assertEqual(arm_pp["tasks_passed"], 10)
            self.assertEqual(arm_lf["tasks_passed"], 9)
            cmp = ev.compare_arms(arm_pp, arm_lf, hourly_usd=0.65)
            self.assertEqual(cmp["regressions"], ["lc_needle_deep_12k"])
            self.assertEqual(cmp["wins"], [])
            self.assertEqual(len(cmp["both_pass"]), 9)
        finally:
            srv_pp.shutdown()
            srv_pp.server_close()
            srv_lf.shutdown()
            srv_lf.server_close()

    def test_health_mode_mismatch_fails_loudly(self):
        tasks = ev.build_task_suite()[:1]
        srv, url = start_mock_server("per_pass")
        try:
            with self.assertRaisesRegex(RuntimeError, "kv_cache_mode mismatch"):
                ev.run_arm_against_url(url, "last_full", tasks, gpu_uuid="unknown")
        finally:
            srv.shutdown()
            srv.server_close()

    def test_cold_prefix_violation_fails_loudly(self):
        tasks = ev.build_task_suite()[:1]
        srv, url = start_mock_server("per_pass", cached_tokens=512)
        try:
            with self.assertRaisesRegex(RuntimeError, "cold-prefix violation"):
                ev.run_arm_against_url(url, "per_pass", tasks, gpu_uuid="unknown", require_cold=True)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()

# Building and Updating Your Own Verified Ledger

[`verified_ledger.py`](verified_ledger.py) is a zero-dependency Python module (`sqlite3` + stdlib) for pairing
`atlas-ingine` (or `atlas-engine`, or any OpenAI-compatible server) with a **host-verified continual memory ledger**.

Instead of letting a model write free-form notes into memory (which accumulates hallucinations and stale assumptions),
a Verified Ledger enforces one rule: **the host verifier is the sole write gate.**

## The 4 banks

| Bank | What enters | Who writes it | Invalidated on drift? |
|---|---|---|---|
| `facts` | Schema, API signatures, GPU specs, render rules | Host inspection only (`set_fact`) | Yes (`status='stale'`) |
| `winners` | Passing solutions / SQL / code / configs with scores | Admitted only when `VerifyResult.ok == True` | Yes (`status='stale'`) |
| `repairs` | Host error signature -> actionable fix rule | Auto-linked when a failed task later passes, or added via CLI | Kept active unless replaced |
| `drift` | Version transition notes (`v1 -> v2`) | Recorded via `bump_env_version` | Active for current `env_version` |

## 1. Write a host verifier for your domain

Any deterministic check works: `sqlite3`, `pytest`, `nvcc` + PyTorch correctness/timing, headless Three.js/Playwright
render checks, Blender frame validation, or a linter/compiler. Return a `VerifyResult`:

```python
from ledger.verified_ledger import VerifiedLedger, VerifyResult, call_openai_chat, run_verified_loop

def my_verifier(candidate: str) -> VerifyResult:
    # Run your real host check here (tests, SQL query, compiler, renderer, etc.)
    ok, score, err_sig, err_detail = run_my_real_check(candidate)
    return VerifyResult(
        ok=ok,
        score=score,                  # higher is better (e.g. speedup, tests passed, quality score)
        error_signature=err_sig,      # short canonical failure key, e.g. "CUDA_SHARED_MEM_OOB"
        error_detail=err_detail,      # verbatim host stderr/traceback shown on retry
        summary=f"Passed with score {score:.2f}" if ok else "",
        tags=["my_domain"],
    )
```

## 2. Run the verified loop against `atlas-ingine`

```python
with VerifiedLedger("my_domain.sqlite", env_version="v1", max_context_chars=4000) as ledger:
    # Optional: stamp verified environment facts from host inspection
    ledger.set_fact("gpu", "RTX 4090 (sm_89, 24 GB)", category="hardware")

    def generate(messages):
        return call_openai_chat(
            base_url="http://127.0.0.1:8003/v1",
            model="inkling-2-mini",
            messages=messages,
            max_tokens=16384,          # leave headroom for reasoning_effort + full code
            temperature=0.2,
            reasoning_effort="medium",
        )

    summary = run_verified_loop(
        ledger=ledger,
        task_id="task_01",
        task_text="Write a fast verified solution for ...",
        verifier=my_verifier,
        generate=generate,
        max_iters=5,
    )
    print(summary.solved, summary.best_score)
```

On every iteration:
- If `VerifyResult.ok` is `False`, the failure is logged in `failure_log` and the exact host error is fed back for retry.
- If `VerifyResult.ok` is `True`, the solution is admitted to `winners` (upgrading any prior winner for that `task_id`
  only if `score` improves), and if the previous iteration failed with `error_signature`, a repair rule is linked in
  `repairs` automatically.
- On the next task, `ledger.render_context(task_text)` retrieves the active `facts`, highest-overlap `repairs`, and top
  `winners` under a strict character budget (`max_context_chars`) without mid-line truncation.

## 3. Update, inspect, and migrate a ledger from the CLI

```bash
# Inspect counts (active/stale facts, winners, repairs, failures, drift events)
python3 ledger/verified_ledger.py --db my_domain.sqlite inspect

# Stamp a verified host/schema/API fact
python3 ledger/verified_ledger.py --db my_domain.sqlite add-fact \
  --category api --key threejs_version --value "r170; use WebGLRenderer with ACESFilmicToneMapping"

# Add or refine a repair rule from a recurring host failure
python3 ledger/verified_ledger.py --db my_domain.sqlite add-repair \
  --sig "NameError: name 'torch' is not defined" \
  --rule "Always include top-level 'import torch' before inline CUDA extension wrappers."

# Preview the exact context injected for a new task
python3 ledger/verified_ledger.py --db my_domain.sqlite render --query "optimize matmul kernel"

# When your schema, database, or library version changes, invalidate stale winners/facts cleanly
python3 ledger/verified_ledger.py --db my_domain.sqlite bump-env \
  --version v2 --note "Table 'orders' renamed to 'sales_orders'; column 'amt' -> 'amount_cents'."
```

## 4. Practical tips for `atlas-ingine`

- **Keep `max_context_chars` compact (`3000–4500` chars):** injecting huge raw playbooks increases thinking length and
  can cause `finish_reason="length"` truncations if `max_tokens` is too small.
- **Give reasoning models generous `max_tokens` (`8192–16384`):** especially with `reasoning_effort="medium"` or `"high"`.
- **Use native JSON mode when your verifier expects structured output:** pass
  `response_format={"type": "json_object"}` or `{"type": "json_schema", ...}` to `call_openai_chat(...)`.
- **Run the bundled example and unit tests:**
  ```bash
  python3 -m unittest ledger/test_verified_ledger.py
  python3 ledger/examples/code_hillclimb_example.py --dry-run
  # Or against a live atlas-ingine server:
  python3 ledger/examples/code_hillclimb_example.py --base-url http://127.0.0.1:8003/v1 --model inkling-2-mini
  ```

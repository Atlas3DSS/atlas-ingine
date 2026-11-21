"""Whole-model checks of the engine (needs the checkpoint and ~24 GB of free GPU memory; opt in with
INGINE_FULL_TESTS=1).

* The fused decode step (kernels, FP8 dense, PDL, CUDA graph) against the PyTorch model's decode.
* A follow-up prompt prefilled on top of the previous turn's state against a fresh prefill.
* Greedy speculative decoding (DSpark drafts verified 1, 3 or 7 at a time, with rollback of rejected rows)
  against plain greedy decoding: the token streams must agree; a divergence is only acceptable at a near-tie
  of the plain decode's logits (the wide verify kernels round differently from the 1-wide ones).
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

MODEL_DIR = "/mnt/data/inkling/hf"
pytestmark = pytest.mark.skipif(os.environ.get("INGINE_FULL_TESTS") != "1", reason="set INGINE_FULL_TESTS=1")

PROMPTS = [
    "Explain how a hash map handles collisions, with a short Python example.",
    "Write a haiku about GPUs, then explain its imagery in two sentences.",
    "List three differences between TCP and UDP.",
]


@pytest.fixture(scope="module")
def engine():
    from ingine.engine import Engine
    return Engine(MODEL_DIR, max_ctx=8192, backend="fast", dense_fp8=True, spec=True)


def chat_ids(eng, prompt):
    text = eng.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             reasoning_effort="low")
    return eng.tokenizer(text, add_special_tokens=False)["input_ids"]


@pytest.mark.parametrize("prompt", PROMPTS)
def test_fused_decode_matches_pytorch_decode(engine, prompt):
    """Teacher-forced on the PyTorch decode's own greedy continuation. (On arbitrary token sequences the
    model sits far off its training distribution, where the two paths' 1-ulp BF16 rounding differences
    flip expert choices and diverge; that measures sensitivity, not correctness.)"""
    ids = chat_ids(engine, prompt)
    engine.tokens_in_state = []
    logits, _ = engine._prefill(ids)
    st = engine.state
    snap = st.snapshot(upto=len(ids))
    ref, tokens = [], []
    with torch.inference_mode():
        for i in range(32):
            tokens.append(int(logits.argmax()))
            logits = engine.model.decode(st, torch.tensor(tokens[-1:], device="cuda"),
                                         torch.tensor([len(ids) + i], device="cuda")).clone()
            ref.append(logits)
    st.restore(snap)
    fused = [engine._decode_logits(tok, len(ids) + i).clone() for i, tok in enumerate(tokens)]
    ref, fused = torch.stack(ref), torch.stack(fused)
    lp, lq = F.log_softmax(ref.float(), -1), F.log_softmax(fused.float(), -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    print(f"fused vs PyTorch decode: mean KL {kl.mean():.2e}, max {kl.max():.2e}")
    # one step from the same state agrees to rounding; later steps carry FP8-dense and rounding differences
    # through each path's own cache writes (measured mean ~2e-3 over 32 steps)
    assert kl[0] < 1e-3, kl
    assert kl.mean() < 1e-2, kl
    assert (ref.argmax(-1) == fused.argmax(-1)).float().mean() >= 0.9


def test_prefix_reuse_matches_fresh_prefill(engine):
    """A follow-up whose prompt extends the previous prompt + reply only prefills the new suffix, on top of
    the state the decode steps left; its logits must match prefilling the whole conversation afresh."""
    from ingine.chat import Sampling
    ids = chat_ids(engine, PROMPTS[2])
    engine.tokens_in_state = []
    list(engine.generate(ids, 40, Sampling(temperature=0.0), set()))
    follow = engine.tokens_in_state + engine.tokenizer(" And how does QUIC compare?", add_special_tokens=False)["input_ids"]
    reused, start = engine._prefill(follow)
    assert start > len(ids)
    engine.tokens_in_state = []
    fresh, start = engine._prefill(follow)
    assert start == 0
    lp, lq = F.log_softmax(fresh.float(), -1), F.log_softmax(reused.float(), -1)
    assert float((lp.exp() * (lp - lq)).sum()) < 0.05
    assert int(fresh.argmax()) == int(reused.argmax())


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_speculation_matches_plain_greedy(engine, prompt):
    from ingine.chat import Sampling
    ids = chat_ids(engine, prompt)
    n = 160
    greedy = Sampling(temperature=0.0)
    engine.tokens_in_state = []
    spec = list(engine.generate(ids, n, greedy, set()))
    # plain greedy decode with the 1-wide graph, recording each step's top-2 margin
    engine.tokens_in_state = []
    logits, _ = engine._prefill(ids)
    plain, margins = [], []
    pos = len(ids)
    for _ in range(n):
        top2 = logits.float().topk(2).values
        margins.append(float(top2[0] - top2[1]))
        tok = int(logits.argmax())
        plain.append(tok)
        logits = engine._decode_logits(tok, pos)
        pos += 1
    first = next((i for i, (a, b) in enumerate(zip(spec, plain)) if a != b), None)
    if first is not None:
        assert margins[first] < 0.25, f"diverged at {first} with plain top-2 margin {margins[first]:.3f}"

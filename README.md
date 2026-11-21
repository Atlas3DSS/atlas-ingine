# Inkling-2-Mini: abliteration and ingine

Two things for [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(looped MoE, 24 physical layers run as 48 layer visits, NVFP4/MXFP8 routed experts) on one RTX 5090:

* **Abliteration** by directional ablation of the residual-stream writes, found with a Heretic-style
  search that counts refusals both when the model answers directly and when it reasons first.
* **ingine**, a single-request inference engine: custom CUDA kernels for every decode operation, the whole
  48-visit decode step in one CUDA graph, DSpark speculative decoding with the checkpoint's own drafter,
  and an OpenAI-compatible server.

## Results

### Abliteration

`ablation/inkling-2-mini-abliterated.safetensors` (trial 13 of the reasoning-aware search, 885 KB: one
direction and weight per attention and MLP write of each of the 48 visits). Refusals counted by
`tools/ablate.py verify` (transformers reference model, greedy) on prompts the search never saw:

| | original | abliterated |
|---|---|---|
| `harmful_test` (104), direct answer | 104/104 | 1/104 |
| JailbreakBench harmful behaviours (100), direct answer | 100/100 | 5/100 |
| `harmful_test` (48), reasoning first (effort low, 768-token budget) | 47/48 | 0/48 |

Damage against the original model: KL divergence 0.043 on its own answers to 64 held-out harmless prompts
(the search objective); on 8192 tokens of technical prose (Cinference's README and docs) mean KL 0.034 and
91.2% top-1 agreement.

Through ingine (`tools/eval.sh`: the 32 borderline prompts and capability set adapted from the heretic
eval tools, greedy):

| | original | abliterated |
|---|---|---|
| refusals, answering directly | 15/32 | 0/32 |
| refusals, reasoning first | 10/32 | 0/32 |
| math (20, reasoning, effort low) | 20/20 | 20/20 |
| code (12, tests executed) | 12/12 | 12/12 |
| multiple choice (12) | 12/12 | 12/12 |

Lighter ablations from the same runs, for comparison (`results/verify-*.json`):

| ablation | direct refusals (harmful_test, JBB) | reasoning refusals | general-text KL, top-1 |
|---|---|---|---|
| trial 13 (shipped) | 1/104, 5/100 | 0/48 | 0.034, 91.2% |
| trial 37 | 2/104, 15/100 | 26/48 | 0.006, 96.4% |
| first search, direct answers only (trial 44) | 0/104, 1/100 | 43/48 | 0.017, 93.3% |

### Speed (RTX 5090, one request)

`engine/bench/bench_e2e.py --dense-fp8`, 1024 generated tokens after a short chat prompt, median of 3 runs,
tokens per second; "chat" explains hash-map collisions (reasoning, then prose and a snippet), "code" writes an
LRU-cache module with tests:

| | chat | code |
|---|---|---|
| decode, greedy | 261 | 265 |
| decode + DSpark, greedy | 340 | 372 |
| decode, temperature 1 | 259 | 260 |
| decode + DSpark, temperature 1 | 313 | 356 |

DSpark's gain depends on how predictable the text is: on "Explain TCP congestion control." drafts average
about 2 tokens per verify call and speculation runs ~10% *slower* than plain greedy decoding (236 vs 263).

Prefill (2048-token chunks, FlexAttention compiled at startup): about 3000 tokens/s for 5K-16K-token prompts
and 2700 tokens/s at 30K. A short prompt still takes ~0.17 s to the first token (prefill runs eagerly).
A follow-up turn only prefills the new suffix on top of the cached conversation.

GPU memory with `--spec --max-ctx 65536`: 25.5 GB resident (19 GB checkpoint weights, 1.8 GB FP8 copies of the
dense matrices, 2.3 GB caches, 2 GB of prefill expert scratch, the drafter), 26.7 GB peak during a 30K prefill.

One decode step (3.8 ms, 483 kernels in one CUDA graph) reads ~3.8 GB: routed experts ~1.9 GB, FP8 dense
projections and head ~1.8 GB, router, gates and KV. At the card's ~1.6 TB/s of achievable bandwidth that is
~2.4 ms; the rest is per-kernel ramp-up and tail that programmatic dependent launch only partly hides.

## How the abliteration works

Inkling-2-Mini has no plain residual stream: each layer reads a softmax mix of RMS-normalised *block
states* (attention residuals, one block per 6 layers), and every write into those blocks passes through a
per-channel short convolution (`attn_sconv`/`mlp_sconv`) after the output projection. Removing a direction
`r` from `wo`/`w2` therefore does not remove it from what is written: the convolution's per-channel taps
mix it back in. ingine instead ablates the writes themselves, after the convolution:

    y <- y - w_v * r_v * (r_v . y)        for every attention and MLP write of visit v (48 visits)

The block states and every read are then free of `r_v`'s component up to `w_v`. The routed experts stay
untouched in their NVFP4/MXFP8 encoding (no requantisation), exactly like fngine's `ablation.bin` for
Qwen3.8-Flash-Next.

* **Directions.** Difference of means between 400 harmful (`mlabonne/harmful_behaviors`) and 400 harmless
  (`mlabonne/harmless_alpaca`) prompts of each visit's attention and MLP *reads* (96 read points), taken
  at two positions: the token that opens a direct answer (`<|message_model|><|content_text|>`) and the
  token that opens reasoning (`<|message_model|><|content_thinking|>`), each projected orthogonal to the
  harmless mean (projected abliteration). A visit's attention write uses the direction read right after it
  (its MLP read), its MLP write the next visit's attention read; `dir_mix` blends the two positions.
* **Weights.** Heretic's kernel over the 48 visits (peak weight at a position, linear decay to a minimum
  weight at a distance, zero beyond), separately for attention and MLP writes.
* **Search.** Optuna TPE, two objectives: refusals (held-out harmful prompts answered directly, plus
  reasoning traces whose first 56 tokens plan a refusal) and KL divergence from the original model on its
  own direct answers to held-out harmless prompts.
* **Ablating only the direct-answer position is not enough.** The first search (direct answers only)
  reached 0/104 refusals at KLD 0.019, but with reasoning enabled the model still refused 43/48: it
  decides in its chain of thought ("This is a request to enable spam/abuse, which I should refuse").
* **The 56-token reasoning check undercounts.** Some trials plan to comply and refuse later: trial 37 scored
  3/48 in the search but refused 26/48 full reasoning runs. Trial 13, at 0/48 in the search, also refuses
  0/48 full runs, at three times trial 37's KL on harmless answers; it is the shipped one.

## ingine

PyTorch hosts the engine; the decode step is a fixed sequence of custom kernels captured in one CUDA graph
(one graph per call width: 1 token, and 2, 4 or 8 tokens for speculative verification).

| Area | Kernels (`engine/csrc/`) |
|---|---|
| `dense` | multi-segment projection GEMV (all of a visit's q/gate/rel/sink/k/v rows in one pass, BF16 or FP8 rows), output projection + post-attention conv, router (sigmoid, biased top-8), attention-residual mix with the pending write's ablation, loop message gates, embedding, unembedding with logit softcap |
| `moe` | routed experts decoded in registers from NVFP4/MXFP8, BF16 tensor-core products, fused softcapped SwiGLU; down projection fused with the weighted sum and the post-MLP conv |
| `attn` | one kernel per visit: q/rel post-processing, key/value short convolutions (static or dynamic), KV write, split-K attention with relative bias, sink and output gate; tensor cores for verify widths |
| `prefill` | expert decode to BF16 for the grouped prefill GEMMs |

Prefill runs in PyTorch: grouped expert GEMMs (`torch._grouped_mm`) and FlexAttention with the relative
bias as a score modifier (TF32 dots on BF16 values; IEEE FP32 is 30x slower, BF16 operands round the
probabilities enough to move logits by KL ~0.005). Every kernel is qualified against an FP64 oracle built
from `ingine/model.py`, which itself matches the transformers reference implementation. Kernels use
programmatic dependent launch, so each one prefetches its weights while the previous kernel finishes.

### DSpark speculative decoding

The checkpoint ships a drafter (`mtp.safetensors`): three bidirectional transformer layers that read the
target's hidden states (taps entering visits 5, 23, 41 and the final norm) and draft 7 tokens per round,
followed by a sequential Markov head that corrects each draft with the previous one. ingine runs it as in
vLLM, with these changes for one request on one GPU:

* Draft projections run as FP8 rows through the dense GEMV kernel, and the base logits reuse the target's
  FP8 unembedding. The Markov correction is evaluated on each step's top 1024 base logits instead of the
  whole vocabulary (64 candidates lost up to a quarter of the accepted drafts; 512 came within 3% of 4096).
* One draft graph per context bucket (1K, 2K, ... max_ctx): the global draft layer attends only to the
  bucket, so short conversations do not pay for a 64K cache. Committing verified rows is graphed too.
* Verification width follows the drafter's confidence: only the leading drafts with probability above
  `--spec-threshold` (default 0.8) are verified, in a 2-, 4- or 8-wide call. A wide call is expensive here:
  8 tokens select more distinct experts, so the routed-expert kernels take 3.9x as long as for one token
  (8.6 ms vs 3.8 ms per call; 2-wide 5.1 ms, 4-wide 6.7 ms). The drafter itself takes 1.9 ms.
* Acceptance is exact: greedy takes the longest prefix equal to the target's argmax; sampling uses
  rejection sampling against the drafts' own distributions, so the output distribution is the target's.
  Rejected rows are rolled back (attention ring slots and short-convolution states).

## Commands

Setup (Python 3.12, torch 2.14 cu130, the transformers PR that adds Inkling2Loop):

```sh
uv venv --python 3.12 tools/.venv
uv pip install --python tools/.venv/bin/python -r tools/requirements.txt
uvx --from 'huggingface_hub[hf_xet]' hf download thinkingmachines/Inkling-2-Mini --local-dir /mnt/data/inkling/hf
```

Serve (OpenAI-compatible, port 8003; the CUDA extensions build on first start):

```sh
cd engine
../tools/.venv/bin/python -m ingine.server --engine --model-dir /mnt/data/inkling/hf \
    --ablation ../ablation/inkling-2-mini-abliterated.safetensors --dense-fp8 --spec --max-ctx 65536
```

Without `--ablation` it serves the original model. Requests take `reasoning_effort` (`none` .. `max` or
0-0.99), `enable_thinking: false` (answer without reasoning), tools, and streaming with
`reasoning_content` deltas.

Reproduce the abliteration (`tools/inkling_ref.py` is the transformers model with the routed experts kept
packed, so it fits in 32 GB):

```sh
W=/mnt/data/inkling/work
tools/.venv/bin/python tools/ablate.py directions --out $W/directions.pt
tools/.venv/bin/python tools/ablate.py directions --out $W/directions_think.pt --suffix '<|message_model|><|content_thinking|>'
tools/.venv/bin/python tools/ablate.py baseline --out $W/baseline.pt
tools/.venv/bin/python tools/ablate.py search --work $W --study think --trials 70 --harmful 24 \
    --think-directions directions_think.pt --think-prompts 24 --think-tokens 56
tools/.venv/bin/python tools/ablate.py show --work $W --study think
tools/.venv/bin/python tools/ablate.py export --work $W --study think --think-directions directions_think.pt \
    --trial N --out ablation/inkling-2-mini-abliterated.safetensors
tools/.venv/bin/python tools/ablate.py verify --ablation ablation/inkling-2-mini-abliterated.safetensors --out $W/verify.json
ABLATION=ablation/inkling-2-mini-abliterated.safetensors tools/eval.sh   # refusals + capability through ingine
```

Checks and benchmarks:

```sh
PYTHONPATH=engine tools/.venv/bin/python -m pytest engine/tests        # kernels vs FP64 oracles, chat, server
INGINE_FULL_TESTS=1 PYTHONPATH=engine tools/.venv/bin/python -m pytest engine/tests/test_engine.py  # whole model, ~24 GB
tools/.venv/bin/python tools/check_engine.py reference --out $W/ref_check.pt
tools/.venv/bin/python tools/check_engine.py engine --ref $W/ref_check.pt --fast kernels --fp8
PYTHONPATH=engine tools/.venv/bin/python engine/bench/bench_e2e.py --backend fast --dense-fp8 [--spec]
```

Layout: `tools/` (reference model, abliteration, evals), `engine/ingine/` (weights, PyTorch model and prefill,
fused decode wiring, DSpark drafter, engine, chat format, server), `engine/csrc/` (CUDA), `engine/SPEC.md`
(kernel contracts), `engine/tests/`, `engine/bench/`.

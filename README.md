# Inkling-2-Mini abliterated: ingine

An abliterated [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) served
by **ingine**, a CUDA inference engine written for this model on one RTX 5090. The abliteration removes refusal
directions from every attention and MLP write of the model's 48 layer visits, found with a search that counts
refusals both when the model answers directly and when it reasons first, so it refuses neither way (0/48 harmful
prompts with reasoning, 1/104 answering directly; KL 0.034 from the original on general text). ingine runs the
whole decode step as custom kernels in one CUDA graph: about 255-260 tokens/s plain and 290-380 tokens/s with DSpark
speculative decoding using the checkpoint's own drafter, with a 256K-token context and an OpenAI-compatible API.

This repository holds ingine, the abliteration (`ablation/inkling-2-mini-abliterated.safetensors`, 885 KB, applied
by ingine at run time), the tools that made and verified it, and an installer. The installer downloads the
ready-to-run model from Hugging Face,
[satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine):
the unchanged Inkling-2-Mini files (20.2 GB) together with the abliteration file and its model card, pinned to one
revision with the size and SHA-256 of every file.

## Run

You need **Linux x86_64, an NVIDIA GPU with compute capability 12.0 and at least 32 GB of memory (RTX 5090), an
NVIDIA driver for CUDA 13 (580 or newer), and about 30 GB of free disk** (20.2 GB of model files; the Python
environment with PyTorch's CUDA libraries and the CUDA compiler take the rest). The RTX PRO 6000, 5000 and 4500
Blackwell cards are also compute capability 12.0 but untested. Other GPUs are not supported: the kernels are
built for `sm_120a` and decode the routed experts with Blackwell's FP4 conversion instructions. The installer does
not install or change GPU drivers.

Install with one command. It clones this repository to `~/inkling-2-mini-abliterated-ingine` and opens its menu:

```bash
curl -fsSL https://raw.githubusercontent.com/satellitedown/inkling-2-mini-abliterated-ingine/main/get.sh | bash
```

Choose **1** to install (Python environment, CUDA toolkit, kernel build and model download), then **5** to add the
`ingine` command and `ingine start` to run the server. Run the same command again any time to get back to the menu.
To install somewhere else, set the folder:
`curl -fsSL https://raw.githubusercontent.com/satellitedown/inkling-2-mini-abliterated-ingine/main/get.sh | INGINE_DIR=/path/to/folder bash`.

Or do the same by hand:

```bash
git clone https://github.com/satellitedown/inkling-2-mini-abliterated-ingine
cd inkling-2-mini-abliterated-ingine
bash setup.sh
```

The menu:

1. **Install everything:** check the GPU, driver, tools and disk space, create the Python environment, install a
   private CUDA toolkit, compile ingine's CUDA kernels once (so the first start is fast) and download the model.
2. **Download / resume model:** download the pinned model files, or resume an interrupted download, and verify them.
3. **Start the server (foreground):** run `ingine serve` in this terminal; **Ctrl+C** stops it.
4. **Update:** pull this repository and rerun the install steps (see below). `bash setup.sh --update` does the same
   without the menu.
5. **Install the `ingine` command:** link `~/.local/bin/ingine` to `scripts/ingine` (make sure `~/.local/bin` is on
   your `PATH`).

Choose **0** to exit. Every step can be repeated: an interrupted install continues with option 1, an interrupted
download with option 2. `bash setup.sh --help` lists the options without installing anything.

**Updating** (`ingine update`, menu option 4 or `bash setup.sh --update`) fast-forwards this repository from
GitHub, then reruns the idempotent install steps: Python packages, CUDA toolkit, kernel build, and the model check.
The model is downloaded again only when the pinned revision in [runtime-manifest.json](runtime-manifest.json)
changed; files that no longer match are moved aside to `models/Inkling-2-Mini.previous-<time>/`, never deleted.
Delete that folder once the new version works. A running server keeps the old code until you restart it:
`ingine stop && ingine start`.

Everything lives inside the install folder:

| Path | Contents |
|---|---|
| `.venv/` | Python 3.12 with PyTorch 2.14 (CUDA 13.0 build) and ingine's dependencies ([requirements-engine.txt](requirements-engine.txt)) |
| `.cuda-toolkit/` | private CUDA toolkit used only to compile ingine's kernels; the system CUDA is not used or changed |
| `engine/build/` | ingine's compiled CUDA extensions |
| `models/Inkling-2-Mini/` | the model from Hugging Face: Inkling-2-Mini's files and the abliteration file (set `INGINE_MODEL_DIR` to use a download elsewhere) |
| `.run/` | pid file and server log, used only when there is no systemd user session (`ingine start` then runs the server with `nohup`) |

## Use

```bash
ingine start      # start the server in the background (systemd user unit inkling-ingine), wait until it is ready
ingine stop       # stop it and free the GPU
ingine status     # unit state, health, served model, GPU memory; exit code 0 ready, 1 starting, 3 stopped
ingine logs       # server log; ingine logs -f follows it
ingine serve      # run in the foreground instead; extra arguments go to the server, e.g. --max-ctx 131072
ingine update     # same as bash setup.sh --update
```

Without the symlink from option 5, run `scripts/ingine` from the install folder. Only one copy of the model fits
on a 32 GB card; stop other programs that hold a lot of GPU memory first. `ingine start` and `ingine stop` are
non-interactive, return when the server is ready or stopped, and send a desktop notification when `notify-send`
is available; if you like, bind them to keys in your desktop environment.

Settings come from the profile in [runtime-manifest.json](runtime-manifest.json) and can be overridden in
`~/.config/ingine/env` (shell assignments, read by every `ingine` command) or in the environment:

| Variable | Default | Meaning |
|---|---|---|
| `INGINE_MODEL_DIR` | `models/Inkling-2-Mini` in the install folder | model folder; point it at an existing download of `satellitedown/Inkling-2-Mini-abliterated-ingine` to reuse it |
| `INGINE_ABLATION` | `$INGINE_MODEL_DIR/inkling-2-mini-abliterated.safetensors` | abliteration file; a relative path is taken relative to the install folder; empty serves the original model (model id `inkling-2-mini`) |
| `INGINE_CTX` | 262144 | context length; lower it to leave GPU memory to other programs |
| `HOST`, `PORT` | 127.0.0.1, 8003 | bind address and port |
| `INGINE_EXTRA_ARGS` | empty | further server arguments, e.g. `--spec-threshold 0.9` |
| `INGINE_START_TIMEOUT` | 300 | seconds `ingine start` waits for the server to report ready |

```bash
# ~/.config/ingine/env
INGINE_MODEL_DIR=/data/models/Inkling-2-Mini
INGINE_CTX=131072
```

`INGINE_MODEL_DIR` must hold the pinned revision; this checks it without downloading:
`.venv/bin/python scripts/download_model.py --model-dir "$INGINE_MODEL_DIR" --verify-only`.

## Connect an application

Use the OpenAI-compatible API at **`http://127.0.0.1:8003/v1`**, model **`inkling-2-mini-abliterated`**
(`inkling-2-mini` when serving without the ablation). No API key is required, and **there is no
authentication**: keep the server on loopback. Requests run one at a time; up to 4 more wait in a queue and
further ones get HTTP 503, so set agent frameworks to one concurrent request. `GET /health` and `GET /v1/models`
report the state.

`POST /v1/chat/completions` takes:

* `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` (the chat template maps them to
  0, 0.1, 0.2, 0.7, 0.9, 0.99, 0.99) or a number from 0 to 0.99; the template's default is 0.9.
* `enable_thinking: false` to answer without reasoning (also accepted in `chat_template_kwargs`).
* `tools` (function calling through the model's chat template), `stop`, `max_tokens` / `max_completion_tokens`,
  `temperature` (default 1; 0 is greedy), `top_p`, `top_k`, `seed`.
* `stream: true` for server-sent events; reasoning arrives as `reasoning_content` deltas before the answer's
  `content` (non-streaming responses carry `message.reasoning_content`). `stream_options.include_usage` adds usage.

```bash
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini-abliterated",
  "messages": [{"role": "user", "content": "Explain how hash maps handle collisions."}],
  "reasoning_effort": "low",
  "max_tokens": 4096
}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8003/v1", api_key="none")
stream = client.chat.completions.create(
    model="inkling-2-mini-abliterated",
    messages=[{"role": "user", "content": "Write a Python LRU cache with tests."}],
    reasoning_effort="medium",
    stream=True,
    # extra_body={"enable_thinking": False},  # answer without reasoning
)
for chunk in stream:
    delta = chunk.choices[0].delta
    print(getattr(delta, "reasoning_content", None) or delta.content or "", end="", flush=True)
```

## Oh My Pi (OMP)

Merge this provider into `~/.omp/agent/models.yml`, keeping existing providers:

```yaml
providers:
  ingine:
    baseUrl: http://127.0.0.1:8003/v1
    auth: none
    api: openai-completions
    models:
      - id: inkling-2-mini-abliterated
        name: Inkling 2 Mini abliterated 256K (ingine)
        contextWindow: 262144
        maxTokens: 32768
        reasoning: true
        thinking:
          mode: effort
          efforts: [minimal, low, medium, high, xhigh, max]
          defaultLevel: medium
          requiresEffort: false
        input: [text]
        cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0}
        compat:
          supportsDeveloperRole: true
          supportsStore: false
          supportsStrictMode: false
          supportsToolChoice: false
          supportsReasoningEffort: true
          reasoningEffortMap: {minimal: none}
          reasoningContentField: reasoning_content
          maxTokensField: max_tokens
          streamIdleTimeoutMs: 600000
```

Start the server first (`ingine start`), then open OMP:

```bash
omp --model ingine/inkling-2-mini-abliterated
```

OMP's `minimal` level is sent as `none` (thinking effort 0). If you lower `INGINE_CTX`, lower `contextWindow` to
match.

## Context and memory

The default profile serves **262,144 tokens** of context (the checkpoint supports 1M). Only 8 of the 48 layer
visits use full attention; the other 40 attend to a 1024-token sliding window held in small ring buffers. The
full-attention caches grow with the context: 8 visits x 8 KV heads x 128 dimensions for keys and values is 32 KB
per token in BF16, 8 GiB at 256K. To fit that beside the 19 GB of checkpoint weights, the default profile stores
them as FP8 E4M3 (`--kv-fp8`: no scale, values rounded to nearest and saturated to ±448; a token attends to its own
key and value in their stored FP8 form), 4 GiB at 256K, and keeps a single FP8 copy of the dense projections and
head (`--dense-fp8`, per-row scales), which prefill dequantizes on the fly. The sliding-window caches stay BF16.

Measured on the RTX 5090 with the default profile (abliterated, DSpark on; a code haystack prompt):

| | 134,289-token prompt | 254,774-token prompt |
|---|---:|---:|
| prefill | 86 s (1,555 tokens/s) | 241 s (1,056 tokens/s) |
| decode afterwards (DSpark, greedy) | 149 tokens/s | 156 tokens/s |
| peak GPU memory allocated by PyTorch | 27.1 GB | 28.1 GB |

The server starts in 13-20 s with 25.4 GB allocated. At 254,774 tokens the fused decode matches the PyTorch
reference decode over the same FP8 caches to a mean KL of 1.5e-3. A passphrase hidden in a haystack of Python
source was retrieved at 134,289 tokens (middle), 199,774 tokens (middle; also with BF16 caches, which still fit
there) and 249,773 tokens (a quarter in). At 254,774 tokens with the passphrase in the middle the model kept
searching in its reasoning and had not answered after 400 tokens; BF16 caches do not fit at that length, so that
run could not be compared. With BF16 caches (`--max-ctx` up to about 200,000 without `--kv-fp8`) decode at
199,774 tokens ran at 91 tokens/s against 121 tokens/s with FP8 caches (DSpark off), as the full-attention
visits read half the bytes.

A follow-up turn only prefills the new suffix on top of the cached conversation, so a long conversation pays for
its prefill once.

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
(the search objective); on 8192 tokens of technical prose ([Cinference](https://github.com/satellitedown/cinference)'s
README and docs) mean KL 0.034 and 91.2% top-1 agreement.

Through ingine (`tools/eval.sh`: the 32 borderline prompts and capability set adapted from the Heretic
eval tools, greedy):

| | original | abliterated |
|---|---|---|
| refusals, answering directly | 15/32 | 0/32 |
| refusals, reasoning first | 10/32 | 0/32 |
| math (20, reasoning, effort low) | 20/20 | 20/20 |
| code (12, tests executed) | 12/12 | 12/12 |
| multiple choice (12) | 12/12 | 12/12 |

Lighter ablations from the same runs, for comparison ([`results/verify-*.json`](results)):

| ablation | direct refusals (harmful_test, JBB) | reasoning refusals | general-text KL, top-1 |
|---|---|---|---|
| trial 13 (shipped) | 1/104, 5/100 | 0/48 | 0.034, 91.2% |
| trial 37 | 2/104, 15/100 | 26/48 | 0.006, 96.4% |
| first search, direct answers only (trial 44) | 0/104, 1/100 | 43/48 | 0.017, 93.3% |

### Speed (RTX 5090, one request)

`engine/bench/bench_e2e.py` with the default profile (`--dense-fp8 --kv-fp8 --max-ctx 262144`), 1024 generated
tokens after a short chat prompt, median of 3 runs, tokens per second; "chat" explains hash-map collisions
(reasoning, then prose and a snippet), "code" writes an LRU-cache module with tests:

| | chat | code |
|---|---|---|
| decode, greedy | 262 | 254 |
| decode + DSpark, greedy | 289 | 365 |
| decode, temperature 1 | 257 | 254 |
| decode + DSpark, temperature 1 | 307 | 382 |

DSpark's gain depends on how predictable the text is (2.4 accepted tokens per verify call for the greedy chat
answer, 3.3 for the code): on "Explain TCP congestion control." drafts average about 2 tokens per verify call
and speculation runs ~10% *slower* than plain greedy decoding (236 vs 263).

Prefill (2048-token chunks, FlexAttention compiled at startup): about 2,800-2,900 tokens/s for 5K-16K-token
prompts, 2,600 tokens/s at 30K, 1,555 tokens/s averaged over a 134K-token prompt and 1,056 tokens/s over 255K. A
short prompt still takes ~0.17 s to the first token (prefill runs eagerly).

GPU memory with the default profile: 25.4 GB allocated after startup (17.8 GB of weights with the single FP8
copy of the dense matrices, 4.3 GB of FP8 full-attention caches for 262,144 tokens, 2 GB of prefill expert
scratch, 0.9 GB for the drafter, buffers), 28.1 GB at the peak of a 254,774-token prefill.

One decode step (3.8 ms, 483 kernels in one CUDA graph) reads ~3.8 GB: routed experts ~1.9 GB, FP8 dense
projections and head ~1.8 GB, router, gates and KV. At the card's ~1.6 TB/s of achievable bandwidth that is
~2.4 ms; the rest is per-kernel ramp-up and tail that programmatic dependent launch only partly hides.

## How the abliteration works

Inkling-2-Mini is a looped mixture-of-experts model: 24 physical layers run as 48 layer visits, with NVFP4/MXFP8
routed experts. It has no plain residual stream: each layer reads a softmax mix of RMS-normalised *block
states* (attention residuals, one block per 6 layers), and every write into those blocks passes through a
per-channel short convolution (`attn_sconv`/`mlp_sconv`) after the output projection. Removing a direction
`r` from `wo`/`w2` therefore does not remove it from what is written: the convolution's per-channel taps
mix it back in. ingine instead ablates the writes themselves, after the convolution:

    y <- y - w_v * r_v * (r_v . y)        for every attention and MLP write of visit v (48 visits)

The block states and every read are then free of `r_v`'s component up to `w_v`. The routed experts stay
untouched in their NVFP4/MXFP8 encoding (no requantisation).

* **Directions.** Difference of means between 400 harmful (`mlabonne/harmful_behaviors`) and 400 harmless
  (`mlabonne/harmless_alpaca`) prompts of each visit's attention and MLP *reads* (96 read points), taken
  at two positions: the token that opens a direct answer (`<|message_model|><|content_text|>`) and the
  token that opens reasoning (`<|message_model|><|content_thinking|>`), each projected orthogonal to the
  harmless mean (projected abliteration). A visit's attention write uses the direction read right after it
  (its MLP read), its MLP write the next visit's attention read; `dir_mix` blends the two positions.
* **Weights.** [Heretic](https://github.com/p-e-w/heretic)'s kernel over the 48 visits (peak weight at a
  position, linear decay to a minimum weight at a distance, zero beyond), separately for attention and MLP writes.
* **Search.** Optuna TPE, two objectives: refusals (held-out harmful prompts answered directly, plus
  reasoning traces whose first 56 tokens plan a refusal) and KL divergence from the original model on its
  own direct answers to held-out harmless prompts.
* **Ablating only the direct-answer position is not enough.** The first search (direct answers only)
  reached 0/104 refusals at KLD 0.019, but with reasoning enabled the model still refused 43/48: it
  decides in its chain of thought ("This is a request to enable spam/abuse, which I should refuse").
* **The 56-token reasoning check undercounts.** Some trials plan to comply and refuse later: trial 37 scored
  3/48 in the search but refused 26/48 full reasoning runs. Trial 13, at 0/48 in the search, also refuses
  0/48 full runs, at three times trial 37's KL on harmless answers; it is the shipped one.

The file holds `attn_dirs` and `mlp_dirs` (FP32, [48, 2304], unit directions) and `attn_w`, `mlp_w` (FP32, [48]),
with the search parameters in its metadata.

## ingine

PyTorch hosts the engine; the decode step is a fixed sequence of custom kernels captured in one CUDA graph
(one graph per call width: 1 token, and 2, 4 or 8 tokens for speculative verification).

| Area | Kernels (`engine/csrc/`) |
|---|---|
| `dense` | multi-segment projection GEMV (all of a visit's q/gate/rel/sink/k/v rows in one pass, BF16 or FP8 rows), output projection + post-attention conv, router (sigmoid, biased top-8), attention-residual mix with the pending write's ablation, loop message gates, embedding, unembedding with logit softcap |
| `moe` | routed experts decoded in registers from NVFP4/MXFP8, BF16 tensor-core products, fused softcapped SwiGLU; down projection fused with the weighted sum and the post-MLP conv |
| `attn` | one kernel per visit: q/rel post-processing, key/value short convolutions (static or dynamic), KV write, split-K attention with relative bias, sink and output gate, BF16 or FP8 full-attention KV; tensor cores for verify widths |
| `prefill` | expert decode to BF16 for the grouped prefill GEMMs |

Prefill runs in PyTorch: grouped expert GEMMs (`torch._grouped_mm`) and FlexAttention with the relative
bias as a score modifier (TF32 dots on BF16 values; IEEE FP32 is 30x slower, BF16 operands round the
probabilities enough to move logits by KL ~0.005). Every kernel is qualified against an FP64 oracle built
from `ingine/model.py`, which itself matches the transformers reference implementation. Kernels use
programmatic dependent launch, so each one prefetches its weights while the previous kernel finishes.
The kernels are compiled on the installing machine by `torch.utils.cpp_extension`; their contracts are in
[engine/SPEC.md](engine/SPEC.md).

### DSpark speculative decoding

The checkpoint ships a drafter (`mtp.safetensors`): three bidirectional transformer layers that read the
target's hidden states (taps entering visits 5, 23, 41 and the final norm) and draft 7 tokens per round,
followed by a sequential Markov head that corrects each draft with the previous one. ingine runs it as in
vLLM, with these changes for one request on one GPU:

* Draft projections run as FP8 rows through the dense GEMV kernel, and the base logits reuse the target's
  FP8 unembedding. The Markov correction is evaluated on each step's top 1024 base logits instead of the
  whole vocabulary (64 candidates lost up to a quarter of the accepted drafts; 512 came within 3% of 4096).
* One draft graph per context bucket (1K, 2K, ... max_ctx): the global draft layer attends only to the
  bucket, so short conversations do not pay for the full context. Committing verified rows is graphed too.
* Verification width follows the drafter's confidence: only the leading drafts with probability above
  `--spec-threshold` (default 0.8) are verified, in a 2-, 4- or 8-wide call. A wide call is expensive here:
  8 tokens select more distinct experts, so the routed-expert kernels take 3.9x as long as for one token
  (8.6 ms vs 3.8 ms per call; 2-wide 5.1 ms, 4-wide 6.7 ms). The drafter itself takes 1.9 ms.
* Acceptance is exact: greedy takes the longest prefix equal to the target's argmax; sampling uses
  rejection sampling against the drafts' own distributions, so the output distribution is the target's.
  Rejected rows are rolled back (attention ring slots and short-convolution states).

The server directly, without the `ingine` wrapper (from `engine/`, with the install's `.venv` and `CUDA_HOME`
pointing at a CUDA 13 toolkit for the first kernel build):

```bash
cd engine
../.venv/bin/python -m ingine.server --engine --model-dir "$INGINE_MODEL_DIR" \
    --ablation ../ablation/inkling-2-mini-abliterated.safetensors \
    --max-ctx 262144 --dense-fp8 --kv-fp8 --spec --spec-threshold 0.8
```

Without `--ablation` it serves the original model. `--host`, `--port` (default 8003), `--model-id` and
`--max-queue` are also accepted; `python -m ingine.server --help` lists everything.

## Reproduce the abliteration

The tools use their own environment with the transformers pull request that adds Inkling2Loop
([huggingface/transformers#51050](https://github.com/huggingface/transformers/pull/51050)) and the search
dependencies (`tools/requirements.txt`: Python 3.12, PyTorch 2.14 for CUDA 13.0, Optuna, datasets):

```bash
uv venv --python 3.12 tools/.venv
uv pip install --python tools/.venv/bin/python -r tools/requirements.txt
```

`tools/inkling_ref.py` is the transformers model with the routed experts kept packed, so it fits in 32 GB. The
tools, tests and benchmarks read the checkpoint from `INGINE_MODEL_DIR` (default `models/Inkling-2-Mini` in this
folder, where the installer puts it; `tools/eval.sh` takes `MODEL`, same default) and the prompt datasets
(`mlabonne/harmful_behaviors`, `mlabonne/harmless_alpaca`, `JailbreakBench/JBB-Behaviors`) from the Hugging Face
cache. `ablate.py verify` measures the general-text KL on the Markdown files matched by the glob in
`INGINE_KL_TEXT` (colon-separated globs; default `~/Projects/cinference/README.md:~/Projects/cinference/docs/*.md`,
the README and docs of a [Cinference](https://github.com/satellitedown/cinference) clone, which the numbers above
used). `W` is any work folder:

```bash
W=$HOME/inkling-work
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

### Publishing

`tools/publish_hf.py` stages the Hugging Face repository (Inkling-2-Mini's files, the abliteration file,
`hf/README.md` as the model card, `LICENSE`, `NOTICE`, `SHA256SUMS`, and an `abliteration/` folder with the tools
and `results/`) and checks it; with `--publish` it uploads it with the token in `HF_TOKEN` (`--private` for a
private repository). `tools/pin.py` then re-pins [runtime-manifest.json](runtime-manifest.json) to the latest
Hugging Face revision (or `--revision REV`), taking every file's size and SHA-256 from the Hub (`abliteration/`
and `.gitattributes` are not pinned or downloaded); commit the result so installs and updates fetch that revision.
The staging folder defaults to `work/hf-release`; pass `--stage` on the model's filesystem so the 20 GB of weights
are reflinked rather than copied.

```bash
tools/.venv/bin/python tools/publish_hf.py                              # stage and check only
HF_TOKEN=... tools/.venv/bin/python tools/publish_hf.py --publish
HF_TOKEN=... tools/.venv/bin/python tools/pin.py
```

## Checks and benchmarks

With the tools environment above and the server stopped (the whole-model checks need the GPU to themselves):

```bash
PYTHONPATH=engine tools/.venv/bin/python -m pytest engine/tests        # kernels vs FP64 oracles, chat, server
INGINE_FULL_TESTS=1 PYTHONPATH=engine tools/.venv/bin/python -m pytest engine/tests/test_engine.py  # whole model, ~24 GB
tools/.venv/bin/python tools/check_engine.py reference --out $W/ref_check.pt
tools/.venv/bin/python tools/check_engine.py engine --ref $W/ref_check.pt --fast kernels --fp8
PYTHONPATH=engine tools/.venv/bin/python engine/bench/bench_e2e.py --backend fast --dense-fp8 [--spec]
```

## Layout

| Path | Contents |
|---|---|
| `get.sh` | one-line bootstrap: clones the repository and opens `setup.sh` |
| `setup.sh` | menu installer and updater |
| `scripts/install.sh` | install steps: checks, `.venv`, CUDA toolkit, kernel build, model download |
| `scripts/ingine` | the `ingine` command (serve, start, stop, status, logs, update) |
| `scripts/download_model.py` | resumable, verified download of the pinned model files |
| `runtime-manifest.json` | pinned Hugging Face revision with every file's size and SHA-256, serving profile |
| `requirements-engine.txt` | ingine's Python dependencies |
| `ablation/` | the abliteration file |
| `results/` | verification, refusal and capability results |
| `tools/` | reference model, abliteration search, evals, engine checks, Hugging Face publishing (`publish_hf.py`, `pin.py`) |
| `engine/ingine/` | weights, PyTorch model and prefill, fused decode wiring, DSpark drafter, engine, chat format, server |
| `engine/csrc/` | CUDA kernels; `engine/SPEC.md` their contracts |
| `engine/tests/`, `engine/bench/` | kernel and engine tests, benchmarks |
| `hf/README.md` | the Hugging Face model card |

## License and credits

This repository (ingine, tools, installer and the abliteration file) is licensed under **Apache-2.0**, see
[LICENSE](LICENSE). The base model, [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by
Thinking Machines, is also Apache-2.0; its files are redistributed unchanged in the Hugging Face repository
[satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine),
not in this Git repository. CUDA components keep NVIDIA's license terms.

The search follows [Heretic](https://github.com/p-e-w/heretic) by p-e-w and the abliteration literature; the
reference model is built on the transformers Inkling2Loop pull request. See [NOTICE](NOTICE). The abliteration
removes the model's safety refusals: you are responsible for how you use it. This is an unofficial community
project, not endorsed by Thinking Machines.

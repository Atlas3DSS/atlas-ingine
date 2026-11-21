---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: finetune
library_name: ingine
pipeline_tag: text-generation
tags:
- abliterated
- uncensored
- decensored
- inkling
- mixture-of-experts
- nvfp4
- fp8
- cuda
- ingine
---

# Inkling-2-Mini · abliterated · ingine

An **abliterated** build of [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
for **[ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine)**, a single-GPU CUDA inference
engine written for this model. It refuses neither when it answers directly nor when it reasons first, with the
original's scores on the tested math, code and knowledge sets. On one RTX 5090 ingine decodes about 255-260
tokens/s, 290-380 tokens/s with DSpark speculative decoding, with 262,144 tokens of context and an OpenAI-compatible
API.

This repository is ready to run: Inkling-2-Mini's files unchanged, plus `inkling-2-mini-abliterated.safetensors`,
which ingine applies at every layer visit at run time. No weights were trained or modified.

> **Reduced safety filtering.** Refusal behaviour has been largely removed. The model can produce content
> that the original would decline. You are responsible for how you use it and for complying with
> the law and the license.

## Measured against the original model

Greedy decoding on prompts the abliteration search never saw; raw data in `abliteration/results/`.

| | Original | **This release** |
|---|---:|---:|
| Refusals, `mlabonne/harmful_behaviors` test (104), direct answer | 104 | **1** |
| Refusals, JailbreakBench harmful behaviours (100), direct answer | 100 | **5** |
| Refusals, `harmful_behaviors` test (48), reasoning first (effort low, 768 tokens) | 47 | **0** |
| Refusals, 32 mild borderline prompts, direct / reasoning first (ingine) | 15 / 10 | **0 / 0** |
| Math (20, reasoning) / Python (12, tests executed) / multiple choice (12) (ingine) | 20/20, 12/12, 12/12 | **20/20, 12/12, 12/12** |
| Agreement with the original on 8,192 tokens of technical docs: top-1 / mean KLD | 1 / 0 | **0.912 / 0.034** |
| KL divergence on the original's own answers to 64 held-out harmless prompts | 0 | **0.043** |

The harmful-prompt and agreement rows use the transformers reference model (`abliteration/inkling_ref.py`, the
Inkling2Loop pull request with the routed experts kept packed); the borderline and capability rows run through
ingine (`abliteration/eval.sh`). Lighter ablations from the same search trade refusals for agreement: trial 37
(KLD 0.006, top-1 0.964) still refused 26/48 prompts once it reasoned, and a search on direct answers only
(trial 44: 0/104 direct) refused 43/48 with reasoning. This release is trial 13.

Speed with ingine on an RTX 5090 with the default profile (262,144-token context, FP8 KV cache), one request,
1024 generated tokens after a short prompt, tokens per second ("chat" explains hash-map collisions, "code" writes
an LRU-cache module with tests):

| | chat | code |
|---|---:|---:|
| decode, greedy | 262 | 254 |
| decode + DSpark, greedy | 289 | 365 |
| decode, temperature 1 | 257 | 254 |
| decode + DSpark, temperature 1 | 307 | 382 |

DSpark's gain depends on how predictable the text is; on loosely predictable prose it can be ~10% slower than
plain greedy decoding. Prefill runs at about 2,800-2,900 tokens/s for 5K-16K-token prompts and 1,056 tokens/s
averaged over a 255K-token prompt; decode after 255K tokens of context runs at about 156 tokens/s.

## What is in this repository

| Path | Contents | Size |
|---|---|---:|
| `model-00001-of-00001.safetensors`, `model.safetensors.index.json` | Inkling-2-Mini's weights (BF16 dense, NVFP4/MXFP8 routed experts), unchanged | 19.5 GB |
| `mtp.safetensors` | Inkling-2-Mini's DSpark drafter, unchanged | 636 MB |
| `inkling-2-mini-abliterated.safetensors` | the abliteration: one direction and weight per attention and MLP write of each of the 48 visits | 885 KB |
| `config.json`, `generation_config.json`, `hf_quant_config.json`, `processor_config.json`, tokenizer files, `tiktoken/`, `chat_template.jinja` | Inkling-2-Mini's configuration, tokenizer and chat template, unchanged | small |
| `README.md`, `LICENSE`, `NOTICE`, `SHA256SUMS` | this card, Apache-2.0, attributions, checksums of every file | small |
| `abliteration/` | how the abliteration was made and measured: `ablate.py`, `inkling_ref.py`, `refusal_eval.py`, `capability_eval.py`, `eval.sh`, `requirements.txt`, `results/*.json` | small |

The abliteration file is safetensors, FP32, with the search parameters (`trial`, `kld`, `params`) in its metadata:

| Tensor | Shape | Meaning |
|---|---|---|
| `attn_dirs` | [48, 2304] | unit direction `r` for the attention write of each visit |
| `attn_w` | [48] | its weight `w` |
| `mlp_dirs` | [48, 2304] | unit direction for the MLP write of each visit |
| `mlp_w` | [48] | its weight |

## How the abliteration works

Inkling-2-Mini is a looped mixture-of-experts model: 24 physical layers run as 48 visits in execution order
(layers 0-5, layers 6-17 three times, then layers 18-23). It has no plain residual stream: each layer reads a
softmax mix of RMS-normalised block states, and every attention and MLP output passes through a per-channel short
convolution (`attn_sconv`/`mlp_sconv`, 4 taps) after its output projection before it is written. Each channel has
its own taps, so a direction removed from `wo`/`w2` is mixed back into the convolved write; baking the edit into
the weights would be inexact. ingine instead edits each write `y` of visit `v` after the convolution, with `r` and
`w` from row `v`:

    y <- y - w * r * (r . y)

The block states and every later read are then free of `r`'s component up to `w`. The NVFP4/MXFP8 routed experts
stay byte-identical.

* **Directions.** Difference of means between 400 harmful (`mlabonne/harmful_behaviors`) and 400 harmless
  (`mlabonne/harmless_alpaca`) prompts at each visit's attention and MLP reads, taken at two positions: the token
  that opens a direct answer and the token that opens reasoning, each projected orthogonal to the harmless mean.
  A visit's attention write uses the direction read right after it, its MLP write the next visit's attention
  read; the search blends the two positions (`dir_mix`).
* **Weights.** [Heretic](https://github.com/p-e-w/heretic)'s kernel over the 48 visits (peak weight at a position,
  linear decay to a minimum at a distance, zero beyond), separately for attention and MLP writes.
* **Search.** Optuna TPE over two objectives: refusals (held-out harmful prompts answered directly, plus reasoning
  traces whose first 56 tokens plan a refusal) and KL divergence from the original model on its own answers to
  held-out harmless prompts. Ablating at the direct-answer position alone was not enough: that model still
  refused 43/48 prompts once it reasoned first, deciding in its chain of thought.

## Requirements

* an NVIDIA GPU with compute capability 12.0 and at least 32 GB: RTX 5090 (RTX PRO 6000/5000/4500 Blackwell
  untested); ingine's kernels are built for `sm_120a` and use Blackwell's FP4 conversion instructions
* Linux x86_64 with an NVIDIA driver for CUDA 13
* ~30 GB of disk: 20.2 GB for this repository, the rest for PyTorch and the CUDA compiler

## Usage

The **[installer](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine)** sets up ingine with a
private CUDA toolkit, downloads and verifies this repository at a pinned revision, starts the server (`ingine
start` / `ingine stop`) and updates both:

```sh
curl -fsSL https://raw.githubusercontent.com/satellitedown/inkling-2-mini-abliterated-ingine/main/get.sh | bash
```

Manually (Python 3.12, a CUDA 13 toolkit in `CUDA_HOME` for the kernels' first build):

```sh
hf download satellitedown/Inkling-2-Mini-abliterated-ingine --local-dir inkling-2-mini-abliterated
git clone https://github.com/satellitedown/inkling-2-mini-abliterated-ingine ingine
cd ingine && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-engine.txt
cd engine && ../.venv/bin/python -m ingine.server --engine --model-dir ../../inkling-2-mini-abliterated \
    --ablation ../../inkling-2-mini-abliterated/inkling-2-mini-abliterated.safetensors \
    --max-ctx 262144 --dense-fp8 --kv-fp8 --spec --host 127.0.0.1 --port 8003
```

OpenAI-compatible Chat Completions at `http://127.0.0.1:8003/v1`, model `inkling-2-mini-abliterated` (streaming,
tools, `reasoning_content`, `reasoning_effort` `none` .. `max` or 0-0.99, `enable_thinking: false`):

```sh
curl -N http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini-abliterated","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
```

The server runs one request at a time with a short FIFO queue; cap agent frameworks to one concurrent request.
Without `--ablation` it serves the original model, for comparison. See the
[GitHub README](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine#readme) for every option,
context and memory notes, agent setup (including Oh My Pi) and the engine's design.

Other inference engines can use the abliteration file only if they apply the edit after the short convolutions
as described above; folding it into `wo`/`w2` gives a different model, which was not measured.

## Credits and licenses

* **Inkling-2-Mini**: Thinking Machines, Apache-2.0
  ([thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), revision
  `6e7c8248f75265bdd4fbab5273304243b5d8c2cb`). Its files are redistributed here unchanged.
* **Abliteration** (`inkling-2-mini-abliterated.safetensors`, `abliteration/`): Apache-2.0. The search follows
  [Heretic](https://github.com/p-e-w/heretic) by p-e-w and the abliteration literature (refusal directions by
  difference of means, Arditi et al. 2024; projected abliteration).
* **Reference model**: the transformers Inkling2Loop pull request
  ([huggingface/transformers#51050](https://github.com/huggingface/transformers/pull/51050)), Apache-2.0.
* **[ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine)** (not included here):
  Apache-2.0. See `NOTICE`.

Provided as is, without warranty. This is an unofficial community release, not endorsed by Thinking Machines. Do
not expose the server unauthenticated to untrusted networks.

## Claim boundaries

Measured: the refusal counts above (greedy decoding; English prompt sets; the harmful sets with the transformers
reference model, the borderline set through ingine), the 44 capability items, agreement with the original on
harmless answers and on 8,192 tokens of technical prose, ingine's speed on one RTX 5090 with one request, and
long-context retrieval with the FP8 KV cache: a passphrase hidden in Python source was found at 134K, 200K and
250K tokens; at 255K, with it in the middle, the model was still searching in its reasoning after 400 tokens.

Not measured: refusals with sampling, in other languages or on other prompt sets; broad benchmarks (knowledge,
long-form reasoning, coding suites); long-context quality beyond that retrieval test; answer quality and
accuracy of answers to harmful requests; image input (ingine serves text only); other GPUs; other inference
engines, or the abliteration folded into the weights. Some prompts may still be refused.

# atlas-ingine

Multi-architecture Apache-2.0 CUDA inference engine for
[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), forked from
[satellitedown/ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine).

Runs on **Ampere (`sm_86`, RTX 3090)**, **Ada (`sm_89`, RTX 4090)**, and **Blackwell (`sm_120a`, RTX 5090 / RTX PRO 6000)**
with **native Elastic sub-4-bit MoE weights (`ATLAS2` / `ATLAS3`)**, **DSpark speculative decoding**, **native vision
input**, and **OpenAI-compatible structured outputs (`response_format`: `json_object` / `json_schema`) + automatic
tool-ban**. Blackwell (`sm_120a`) outputs remain bit-identical to upstream (`kernels_bitlock` 48/48 PASS).

## Measured performance (64K context)

Measured with `ingine-bench` (`--max-ctx 65536 --dense-fp8 --kv-fp8 --spec --spec-threshold 0.8 --effort medium --tokens 512 --runs 2 --long-prompt 64000`):

| GPU | Arch | Checkpoint | VRAM @ 64K ctx | Chat decode | Code decode | 64K prefill |
|---|---|---|---:|---:|---:|---:|
| RTX 3090 24 GB | `sm_86` | Native `NVFP4+MXFP8` | `21.82 GB` | `154.9 tok/s` | `213.2 tok/s` | `2,980 tok/s` |
| RTX 3090 24 GB | `sm_86` | Elastic `atlas-12g` | `16.37 GB` | `132.4 tok/s` | `161.2 tok/s` | `2,970 tok/s` |
| RTX 3090 24 GB | `sm_86` | Elastic `atlas-10g` | `14.33 GB` | `145.6 tok/s` | `196.1 tok/s` | `3,012 tok/s` |
| RTX 4090 24 GB | `sm_89` | Elastic `atlas-12g` | `16.36 GB` | `236.7 tok/s` | `278.3 tok/s` | `7,031 tok/s` |
| RTX PRO 6000 96 GB | `sm_120a` | Elastic `atlas-12g` | `16.26 GB` | `411.0 tok/s` | `519.9 tok/s` | `11,728 tok/s` |
| RTX PRO 6000 96 GB | `sm_120a` | Elastic `atlas-10g` | `14.18 GB` | `410.7 tok/s` | `554.3 tok/s` | `11,806 tok/s` |

Additional verified live-serving footprints (`ingine-serve`):
- **RTX 3090 (`sm_86`, 12 GB class):** `atlas-10g` at 16K ctx with `--no-vision --dense-fp8 --kv-fp8 --spec --park-mb 0` uses **`12,984 MiB`** (`<12 GiB` without `--spec`).
- **RTX 4090 (`sm_89`, 24 GB):** `atlas-12g` at 128K ctx with `--no-vision` uses **`17,776 MiB`**; live long-output serving with vision enabled at 64K ctx runs at **~257 tok/s**.
- **RTX 5090 / RTX PRO 6000 (`sm_120a`, 32+ GB):** native `NVFP4+MXFP8` serves up to **262,144 tokens** of context in `24.4 GiB` (`490–560 tok/s` decode, `6,200–10,900 tok/s` prefill; see [Upstream RTX 5090 results](#upstream-rtx-5090-and-abliteration-results)).

## What this fork adds over upstream `satellitedown/ingine`

1. **Multi-architecture CUDA kernels (`sm_86`, `sm_89`, `sm_120a`):**
   - Native Ampere (`sm_86`) and Ada (`sm_89`) codepaths on CUDA 12.8+ and CUDA 13.x without degrading Blackwell (`sm_120a`).
   - Software fallbacks for Hopper/Blackwell-only primitives (PDL, DSMEM/cluster reductions, hardware FP4/FP8 conversions) and branchless FP8/E2M1/ATLAS2/ATLAS3 unpackers for `sm_86`/`sm_89`.
2. **Native Elastic sub-4-bit MoE weights (`ATLAS2` 2.50 b/w, `ATLAS3` 3.50 b/w):**
   - Mixed-precision MoE tier converter ([`tools/quantize_elastic_ingine.py`](tools/quantize_elastic_ingine.py)) that compresses the 20.2 GB base checkpoint to **`atlas-12g` (`14.05 GB` on disk incl. vision + `mtp.safetensors`)** or **`atlas-10g` (`12.01 GB` on disk incl. vision + `mtp.safetensors`)**.
   - Native fused decode (`native/src/kernels/moe/moe.cu`) and prefill (`native/src/kernels/prefill/wgemm.cu`) kernels verified against FP64 oracles (`kernels_moe`, `prefill_moe`).
3. **Structured outputs and tool-ban:**
   - Native OpenAI `response_format` (`text`, `json_object`, `json_schema`) constrained decoding in `ingine-serve` (`serve_server` 22/22 PASS).
   - Automatic `<|content_invoke_tool_json|>` token banning when `tools` is absent from the request.

## Requirements

- **OS:** Linux x86_64
- **GPU:** NVIDIA GPU with compute capability **8.6 (Ampere, e.g. RTX 3090)**, **8.9 (Ada, e.g. RTX 4090)**, or **12.0 (Blackwell, e.g. RTX 5090 / RTX PRO 6000)**
  - **24 GB VRAM (RTX 3090 / 4090):** runs the native 20.2 GB checkpoint at 64K context (`~21.8 GB` VRAM) or Elastic `atlas-12g` up to 128K context (`~17.8 GB` VRAM).
  - **16 GB / 12 GB class VRAM:** runs Elastic `atlas-12g` (`~16.3 GB` @ 64K) or `atlas-10g` (`~13.0 GB` @ 16K with `--spec --no-vision`, `<12 GiB` without `--spec`).
  - **32+ GB VRAM (RTX 5090 / RTX PRO 6000):** runs the native checkpoint up to 262,144-token context (`~24.4 GB` VRAM).
- **Driver & Toolchain:** NVIDIA driver for CUDA 12.8+ / CUDA 13 (580+ for the CUDA 13.4 private toolkit installer), C++20 compiler (GCC 12–16), CMake >= 3.28, Ninja.
- **Runtime:** `ingine-serve` is a standalone C++/CUDA binary; it needs no Python or PyTorch to run.

## Build and run

### Option A: Build directly with CMake (existing CUDA 12.8+ / 13.x toolkit)

```bash
git clone https://github.com/Atlas3DSS/atlas-ingine
cd atlas-ingine

# Build multi-arch (sm_86 + sm_89 + sm_120a) or pick your single GPU arch (e.g. "89" for RTX 4090, "86" for RTX 3090)
cmake -S native -B native/build-multiarch -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="86;89;120a" \
  -DINGINE_BUILD_APPS=ON \
  -DINGINE_BUILD_TESTS=ON
cmake --build native/build-multiarch -j

# Run unit tests (optional)
ctest --test-dir native/build-multiarch --output-on-failure
```

Start the server on a 24 GB card (RTX 3090 / RTX 4090, 64K context with vision + DSpark speculative decoding):

```bash
./native/build-multiarch/apps/serve/ingine-serve \
  --engine \
  --model-dir /path/to/Inkling-2-Mini \
  --host 127.0.0.1 --port 8003 \
  --max-ctx 65536 --dense-fp8 --kv-fp8 \
  --spec --spec-threshold 0.8 --park-mb 0
```

Add `--no-vision` to save ~0.6 GiB VRAM on text-only workloads, or `--ablation /path/to/inkling-2-mini-abliterated.safetensors` to apply the optional abliteration patch (omitting `--ablation` serves the original model as `inkling-2-mini`).

### Option B: One-line interactive installer (private CUDA 13.4 toolkit + model download)

```bash
curl -fsSL https://raw.githubusercontent.com/Atlas3DSS/atlas-ingine/main/get.sh | bash
```

Or manually:

```bash
git clone https://github.com/Atlas3DSS/atlas-ingine
cd atlas-ingine
bash setup.sh
```

Choose **1** to install build tools, a private CUDA 13.4 toolkit, build `ingine-serve` (`sm_86;89;120a`), and download the
pinned model from [satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine)
(20.2 GB), then **5** to link `~/.local/bin/ingine`. On 24 GB GPUs (RTX 3090 / 4090), set `INGINE_CTX=65536` in
`~/.config/ingine/env` (the default `262144` targets 32+ GB cards):

```bash
mkdir -p ~/.config/ingine
cat <<'EOF' > ~/.config/ingine/env
INGINE_CTX=65536
INGINE_EXTRA_ARGS="--park-mb 0"
EOF
ingine start
```

### Creating Elastic sub-4-bit MoE checkpoints (`atlas-12g` / `atlas-10g`)

Convert a downloaded base `Inkling-2-Mini` directory into a smaller mixed-precision `ATLAS2`/`ATLAS3` checkpoint using
[`tools/quantize_elastic_ingine.py`](tools/quantize_elastic_ingine.py) (links/copies unchanged dense, vision, tokenizer,
and `mtp.safetensors` files):

```bash
python3 tools/quantize_elastic_ingine.py \
  --src /path/to/Inkling-2-Mini \
  --dst /path/to/Inkling-2-Mini-atlas-12g \
  --profile atlas-12g   # or atlas-10g
```

Then pass `--model-dir /path/to/Inkling-2-Mini-atlas-12g` to `ingine-serve` (or set `INGINE_MODEL_DIR` in `~/.config/ingine/env`).

## Service management (`ingine` CLI)

```bash
ingine start      # start the server in the background (systemd user unit inkling-ingine), wait until it is ready
ingine stop       # stop it and free the GPU
ingine status     # unit state, health, served model, GPU memory; exit code 0 ready, 1 starting, 3 stopped
ingine logs       # server log; ingine logs -f follows it
ingine serve      # run in the foreground instead; extra arguments go to the server, e.g. --max-ctx 65536
ingine update     # pull updates and rebuild incrementally
```

Environment overrides (`~/.config/ingine/env` or shell environment):

| Variable | Default | Meaning |
|---|---|---|
| `INGINE_MODEL_DIR` | `models/Inkling-2-Mini` in the install folder | model folder (native checkpoint or Elastic `atlas-12g` / `atlas-10g` directory) |
| `INGINE_ABLATION` | `$INGINE_MODEL_DIR/inkling-2-mini-abliterated.safetensors` | optional abliteration file; empty serves the original model (`inkling-2-mini`) |
| `INGINE_CTX` | 262144 | context length (`65536` recommended on 24 GB GPUs with native weights) |
| `HOST`, `PORT` | 127.0.0.1, 8003 | bind address and port |
| `INGINE_EXTRA_ARGS` | empty | further server arguments (e.g. `--no-vision`, `--park-mb 0`, `--temperature`, `--top-p`, `--top-k`, `--spec-threshold 0.8`) |
| `INGINE_START_TIMEOUT` | 300 | seconds `ingine start` waits for `/health` |

## OpenAI-compatible API

Use **`http://127.0.0.1:8003/v1`**, model **`inkling-2-mini`** (or `inkling-2-mini-abliterated` when `--ablation` is set).
No API key is required and there is no authentication; keep the server bound to loopback. `GET /health` and
`GET /v1/models` report server status.

`POST /v1/chat/completions` supports:

* `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` (mapped to `0, 0.1, 0.2, 0.7, 0.9, 0.99, 0.99`) or a float in `[0, 0.99]`.
* `enable_thinking: false` (also accepted in `chat_template_kwargs`) to answer without reasoning.
* `response_format`: `{"type": "text"}`, `{"type": "json_object"}`, or `{"type": "json_schema", "json_schema": {"name": "...", "schema": {...}}}` for grammar-constrained JSON output.
* `tools` (function calling through the chat template). When `tools` is omitted, `<|content_invoke_tool_json|>` is automatically banned.
* `stop`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`.
* `stream: true` (SSE with `reasoning_content` and `content` deltas; `stream_options.include_usage` adds usage).
* Vision inputs as base64 `data:` URIs (`image_url`, `input_image`, or Anthropic `image` blocks; PNG, JPEG, WebP, GIF, BMP). Remote URLs are not fetched.

```bash
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Explain how hash maps handle collisions."}],
  "reasoning_effort": "low",
  "max_tokens": 4096
}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8003/v1", api_key="none")
resp = client.chat.completions.create(
    model="inkling-2-mini",
    messages=[{"role": "user", "content": "Return a JSON object with keys city and country for Tokyo."}],
    response_format={"type": "json_object"},
    extra_body={"reasoning_effort": "none"},
)
print(resp.choices[0].message.content)
```

### Images

The model reads images through its built-in vision tower (part of the checkpoint, no extra download). Each image costs
**one token per 40×40 pixels**, `ceil(height / 40) * ceil(width / 40)` (432 tokens for 960×720, 1,296 for 1920×1080).
Images above 4,096 tokens are resized preserving aspect ratio (`--image-max-tokens N`, up to 10,609 for 4096×4096).
Pass `--no-vision` to serve text only and save ~0.6 GiB VRAM.

## Upstream RTX 5090 and abliteration results

The measurements below are from upstream `satellitedown/ingine` on a single RTX 5090 (32 GB, `sm_120a`, default 262K
profile, native `NVFP4+MXFP8` weights, DSpark speculative decoding):

| RTX 5090 (upstream) | chat | code | coding agent |
|---|---:|---:|---:|
| temperature 1 (default) | 487 tok/s | 554 tok/s | 501 tok/s |
| greedy (temperature 0) | 501 tok/s | 561 tok/s | 510 tok/s |

- Prefill on RTX 5090: `10,900 tok/s` (20K), `9,600 tok/s` (64K), `8,100 tok/s` (131K), `7,000 tok/s` (200K); `24.4 GiB` VRAM (`23.8 GiB` with `--no-vision`).
- Long-context needle retrieval on RTX 5090: passes at `131,443` tokens (`16.4 s` prefill, `354 tok/s` decode) and `254,019` tokens (`40.9 s` prefill, `258 tok/s` decode).
- Optional abliteration eval data from upstream is retained in [`results/`](results).

## License and credits

This repository is licensed under **Apache-2.0**; see [LICENSE](LICENSE) and [NOTICE](NOTICE).

- Upstream engine, installer, and abliteration tooling: [satellitedown/ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine) (Apache-2.0), with upstream kernel inspirations credited in [NOTICE](NOTICE) (`fngine`, `cinference`, `NInfer`).
- Base model: [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by Thinking Machines Lab (Apache-2.0; model weights are not stored in this Git repository).
- Multi-architecture (`sm_86`/`sm_89`/`sm_120a`) port, Elastic `ATLAS2`/`ATLAS3` sub-4-bit MoE kernels and converter, and native OpenAI `response_format` / tool-ban: Atlas 3DSS.
- CUDA runtime/toolkit components remain subject to NVIDIA's license terms. Unofficial community project, not endorsed by Thinking Machines.

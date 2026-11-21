#!/usr/bin/env python3
"""Model goldens for the C++ engine: prefill and greedy-decode logits from the Python fast engine.

Runs engine.Engine (backend "fast", dense FP8, decode graphs, prefill chunk 2048) on a fixed set of prompts and
writes native/tests/golden/model/:

    manifest.json        {"cases": [{"name", "file", "ablated", "kv_fp8", "prompt_tokens"}], ...}
    <case>.json          prompt ids; the last prefill logits (top-256 ids/values + logsumexp of the full
                         [200058] row); then 32 greedy decode steps, each with its input token and position and the
                         resulting logits (top-256 + logsumexp); greedy tokens = argmax of every logits row

Prompts: a short chat turn (chat template), and raw-text prompts of 300 tokens, 2100 tokens (crosses the
2048-token prefill chunk) and 5000 tokens (sliding-window ring wrap, three chunks). Configurations: abliterated
model with BF16 and with FP8 full-attention KV caches (all prompts), and the original model (short chat and
2100 tokens) with BF16 caches. One engine (~20 GB of GPU memory) is resident at a time.

    PYTHONPATH=engine tools/.venv/bin/python native/tools/goldens_model.py \
        --model-dir /mnt/data/inkling/hf --ablation ablation/inkling-2-mini-abliterated.safetensors
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "native" / "tests" / "golden" / "model"
TOPK = 256
STEPS = 32
MAX_CTX = 8192
CHUNK = 2048

CHAT = [{"role": "user", "content": "Explain how a hash map handles collisions, with a short Python example."}]
# raw-text source for the long prompts: repository text, repeated (deterministic)
TEXT_SOURCES = ["README.md", "engine/SPEC.md", "engine/ingine/model.py", "engine/ingine/fast.py",
                "engine/ingine/engine.py", "native/DESIGN.md"]


def text_ids(tokenizer, n: int) -> list[int]:
    text = "\n\n".join((ROOT / p).read_text() for p in TEXT_SOURCES if (ROOT / p).exists())
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def summary(logits: torch.Tensor) -> dict:
    z = logits.float()
    vals, idx = torch.topk(z, TOPK)
    return {"top_ids": idx.tolist(), "top_logits": [float(v) for v in vals.tolist()],
            "logsumexp": float(torch.logsumexp(z.double(), -1))}


def run_case(eng, name: str, ids: list[int], ablated: bool, kv_fp8: bool) -> dict:
    eng.tokens_in_state = []  # fresh state: no prefix reuse across cases
    logits, reused = eng._prefill(ids)
    assert reused == 0
    case = {"name": name, "ablated": ablated, "kv_fp8": kv_fp8, "dense_fp8": True, "prefill_chunk": CHUNK,
            "prompt_ids": list(ids), "prefill": summary(logits), "decode": []}
    pos = len(ids)
    token = int(logits.argmax())
    greedy = [token]
    for _ in range(STEPS):
        logits = eng._decode_logits(token, pos)
        eng.tokens_in_state.append(token)
        step = summary(logits)
        step.update(input_token=token, pos=pos)
        case["decode"].append(step)
        pos += 1
        token = int(logits.argmax())
        greedy.append(token)
    case["greedy_tokens"] = greedy
    torch.cuda.synchronize()
    return case


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="/mnt/data/inkling/hf")
    ap.add_argument("--ablation", default=str(ROOT / "ablation" / "inkling-2-mini-abliterated.safetensors"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from ingine.chat import ChatFormat
    from ingine.engine import Engine

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fmt = ChatFormat.from_pretrained(args.model_dir)
    prompts = {
        "chat": fmt.render(CHAT).ids,
        "t300": text_ids(fmt.tokenizer, 300),
        "t2100": text_ids(fmt.tokenizer, 2100),
        "t5000": text_ids(fmt.tokenizer, 5000),
    }
    configs = [  # (ablated, kv_fp8, prompts)
        (True, False, ["chat", "t300", "t2100", "t5000"]),
        (True, True, ["chat", "t300", "t2100", "t5000"]),
        (False, False, ["chat", "t2100"]),
    ]
    manifest = {"topk": TOPK, "steps": STEPS, "max_ctx": MAX_CTX, "prefill_chunk": CHUNK, "dense_fp8": True,
                "model_dir": args.model_dir, "cases": []}
    for ablated, kv_fp8, names in configs:
        eng = Engine(args.model_dir, max_ctx=MAX_CTX, ablation=args.ablation if ablated else None, backend="fast",
                     graph=True, prefill_chunk=CHUNK, dense_fp8=True, kv_fp8=kv_fp8)
        for p in names:
            name = f"{'abl' if ablated else 'orig'}_{'kv8' if kv_fp8 else 'kv16'}_{p}"
            case = run_case(eng, name, prompts[p], ablated, kv_fp8)
            file = f"{name}.json"
            (out / file).write_text(json.dumps(case))
            manifest["cases"].append({"name": name, "file": file, "ablated": ablated, "kv_fp8": kv_fp8,
                                      "prompt_tokens": len(prompts[p])})
            top1 = case["greedy_tokens"][:8]
            print(f"{name}: {len(prompts[p])} prompt tokens, greedy {top1}...", flush=True)
        del eng
        gc.collect()
        torch.cuda.empty_cache()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest['cases'])} cases to {out}")


if __name__ == "__main__":
    main()

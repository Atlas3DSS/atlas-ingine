#!/usr/bin/env python3
"""Agreement of ingine (PyTorch model or fast decode) with the transformers reference.

    check_engine.py reference --out W/ref_check.pt     # reference logits for a fixed token sequence
    check_engine.py engine --ref W/ref_check.pt        # ingine prefill/decode against them

The sequence is a chat prompt plus the reference's own greedy answer, so every position is in
distribution. Reported per position: KL(reference || engine), top-1 agreement.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "engine"))
MODEL_DIR = os.environ.get("INGINE_MODEL_DIR", str(ROOT / "models" / "Inkling-2-Mini"))

PROMPT = ("Write a Python function that merges overlapping intervals, then explain its time complexity "
          "and give two edge cases.")


def cmd_reference(args):
    import inkling_ref as ref
    model, tok = ref.load(MODEL_DIR)
    text = ref.chat_prompt(tok, PROMPT, effort="low")
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
    with torch.inference_mode():
        out = model.generate(input_ids=ids, max_new_tokens=args.tokens, do_sample=False)
        logits = model(input_ids=out, use_cache=False).logits[0].float()
    torch.save({"ids": out[0].cpu(), "prompt_len": ids.shape[1], "logits": logits.cpu().to(torch.bfloat16)}, args.out)
    print(f"saved {out.shape[1]} tokens ({ids.shape[1]} prompt)")
    print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)[:600])


def compare(name, ref_logits, logits):
    lp_ref = F.log_softmax(ref_logits.float(), -1)
    lp = F.log_softmax(logits.float(), -1)
    kl = (lp_ref.exp() * (lp_ref - lp)).sum(-1)
    top1 = (ref_logits.argmax(-1) == logits.argmax(-1)).float()
    print(f"{name}: positions {len(kl)}  mean KL {kl.mean():.5f}  max KL {kl.max():.4f}  top-1 {top1.mean():.4f}")
    return kl


def cmd_engine(args):
    from ingine import model as M, weights as W
    data = torch.load(args.ref)
    ids = data["ids"].cuda()
    ref_logits = data["logits"].cuda().float()
    t = time.time()
    w = W.load(MODEL_DIR)
    print(f"weights {time.time() - t:.1f}s, {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    m = M.Model(w)
    st = M.State(max_ctx=args.max_ctx)
    with torch.inference_mode():
        t = time.time()
        logits = m.prefill(st, ids, all_logits=True)
        torch.cuda.synchronize()
        print(f"prefill {len(ids)} tokens in {time.time() - t:.2f}s")
        compare("prefill (one chunk)", ref_logits, logits)
        # chunked prefill, then token-by-token decode for the tail
        st.reset()
        split = len(ids) - args.decode_tokens
        chunks = []
        for s in range(0, split, args.chunk):
            chunks.append(m.prefill(st, ids[s:min(split, s + args.chunk)], all_logits=True))
        snap = st.snapshot(upto=split)
        dec = []
        t = time.time()
        for p in range(split, len(ids)):
            dec.append(m.decode(st, ids[p:p + 1], torch.tensor([p], device="cuda")))
        torch.cuda.synchronize()
        print(f"decode {len(dec)} tokens in {time.time() - t:.2f}s")
        logits2 = torch.cat(chunks + [torch.stack(dec)])
        compare(f"prefill chunks of {args.chunk} + decode {args.decode_tokens}", ref_logits, logits2)
        compare("decode vs one-chunk prefill", logits[split:], logits2[split:])
        if args.fast:
            from ingine import fast as FD
            ops = None if args.fast == "torch" else FD.KernelOps()
            dec_fast = FD.FastDecoder(w, m.ablation, st, ops=ops, dense_fp8=args.fp8)
            for graph in (False, True):
                st.restore(snap)
                tok = torch.zeros(1, dtype=torch.long, device="cuda")
                pos = torch.zeros(1, dtype=torch.long, device="cuda")
                if graph:
                    saved = st.snapshot(upto=split)
                    g = torch.cuda.CUDAGraph()
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        dec_fast.decode(tok, pos)
                    torch.cuda.current_stream().wait_stream(s)
                    with torch.cuda.graph(g):
                        out_buf = dec_fast.decode(tok, pos)
                    st.restore(saved)
                outs = []
                t = time.time()
                for p in range(split, len(ids)):
                    tok.fill_(int(ids[p]))
                    pos.fill_(p)
                    if graph:
                        g.replay()
                        outs.append(out_buf[0].clone())
                    else:
                        outs.append(dec_fast.decode(tok, pos)[0].clone())
                torch.cuda.synchronize()
                dt = time.time() - t
                print(f"fast decode ({args.fast}, graph={graph}) {len(outs)} tokens in {dt:.2f}s "
                      f"({len(outs) / dt:.1f} tok/s)")
                compare(f"fast ({args.fast}, graph={graph}) vs PyTorch decode", logits2[split:], torch.stack(outs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reference"); p.add_argument("--out", required=True); p.add_argument("--tokens", type=int, default=320)
    p = sub.add_parser("engine"); p.add_argument("--ref", required=True); p.add_argument("--max-ctx", type=int, default=4096)
    p.add_argument("--chunk", type=int, default=100); p.add_argument("--decode-tokens", type=int, default=24)
    p.add_argument("--fast", choices=("torch", "kernels"), help="also run FastDecoder with TorchOps or KernelOps")
    p.add_argument("--fp8", action="store_true", help="FastDecoder with FP8 dense projections and head")
    args = ap.parse_args()
    {"reference": cmd_reference, "engine": cmd_engine}[args.cmd](args)


if __name__ == "__main__":
    main()

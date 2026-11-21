"""DSpark drafter for Inkling-2-Mini (mtp.safetensors), in PyTorch.

Follows vLLM `vllm/models/inkling2/nvidia/dspark.py` + `speculator.py` (and SGLang `inkling2_mtp.py`):

* Target taps per processed row: the partial attention-residual block entering visits 5, 23 and 41 and the
  final normed hidden (index 48), each BF16 -> `feats [rows, 4, 2304]`.
* Committed rows become draft context: per draft layer l, ``c = mix_rmsnorm(feats, alpha[l])``,
  ``k = rope(rmsnorm_128(c Wk_ctx))``, ``v = c Wv_ctx``. Layers 0 and 2 attend to the last 1024 committed
  positions, layer 1 to the last `DRAFT_CTX` = 65536 (the reference: all of them; the cap keeps the cache
  size independent of max_ctx).
* A round drafts a block of 8 rows at positions P..P+7 (row 0 = the anchor token's normed embedding,
  rows 1-7 = `mask_embed`) through 3 bidirectional transformer layers, then a sequential Markov head turns
  rows 1-7 into 7 draft tokens: ``state <- GRU-like cell([state, W1[prev], h_i, f_rnn])`` and
  ``logits_i = softcap30(h_i U^T + fold_b + tanh(out_i) fold_w^T)``.
* Greedy verification accepts the longest prefix of drafts equal to the target's argmax.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from . import weights as W

D = W.HIDDEN
HEADS, KV_HEADS, HEAD_DIM = 24, 8, 128
BLOCK = 8
GAMMA = BLOCK - 1
RANK = 256
LOCAL = (0, 2)
WINDOW = 1024
DRAFT_CTX = 65536  # positions (block rows included) the global layer attends to: a ring of this many slots
TAPS = (5, 23, 41)  # visits whose entering partial block is tapped; tap 3 is the final normed hidden
ROPE_THETA = 1e6
EPS = 1e-6


def bf(x):
    return x.to(torch.bfloat16)


def rms(x, w=None):
    x = x.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return y * w.float() if w is not None else y


def mix_rmsnorm(feats, alpha_row, w):
    """feats [n, 4, D] BF16, alpha_row [4] BF16 -> BF16 [n, D] (FP32 mix, one rounding)."""
    m = (feats.float() * alpha_row.float()[None, :, None]).sum(1)
    return bf(rms(m, w))


def rope(x, positions):
    """Neox rotary over the full head dim. x [n, heads, 128] BF16, positions [n] int64."""
    half = HEAD_DIM // 2
    inv = ROPE_THETA ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) * 2 / HEAD_DIM)
    ang = positions.float()[:, None] * inv[None]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return bf(torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1))


class Drafter:
    """`head` = the target's unembedding as the fused decoder stores it, (weight, row scales or None): the
    drafter's base logits use the same matrix (FP8 rows when the decoder runs FP8 dense). The drafter's own
    q/k/v and gate/up projections run as FP8 rows through the dense GEMV kernel; drafts only have to be
    good guesses, verification keeps the output exact.

    Global-layer cache: slot = position % `ctx` (ctx = min(max_ctx, DRAFT_CTX)). Every position before the
    block is committed before a draft, so below `ctx` positions slot i holds position i, and beyond it every
    slot holds one of the last `ctx` positions: either way the valid slots are those below P + 8."""

    CANDIDATES = 1024  # Markov-head vocabulary per step: the top base logits (q is supported on them)

    def __init__(self, model_dir: str | Path, target: W.Weights, max_ctx: int, head=None, device="cuda"):
        from . import ext
        self.dense = ext.load("dense")
        fp8 = self.dense.fp8_rows
        path = Path(model_dir) / "mtp.safetensors"
        with safe_open(str(path), framework="pt", device=device) as f:
            t = {k: f.get_tensor(k) for k in f.keys()}
        self.alpha = bf(torch.softmax(t["fuse_weight"].float(), -1))  # [4 rows, 4 taps]
        self.embed, self.embed_norm = target.embed, target.embed_norm
        if head is None:
            head = ((target.unembed, target.unembed_scale) if target.unembed_scale is not None
                    else tuple(fp8(target.unembed)))
        self.head = head
        self.layers = []
        hw, hb = t["markov_head.hidden_out.weight"].double(), t["markov_head.hidden_out.bias"].double()
        fw, fb = [], []
        u_w, u_s = target.unembed, target.unembed_scale
        for s in range(0, u_w.shape[0], 16384):  # FP64 fold, chunked to bound memory (FP8 rows dequantized)
            u = W.dequant(u_w[s:s + 16384], None if u_s is None else u_s[s:s + 16384]).double()
            fw.append(bf(u @ hw))
            fb.append(bf(u @ hb))
        self.fold_w, self.fold_b = torch.cat(fw), torch.cat(fb)  # [V, 256], [V]
        for l in range(3):
            p = f"layers.{l}."
            a = p + "self_attn."
            qkv = torch.cat([t[a + "q_proj.weight"], t[a + "k_proj.weight"], t[a + "v_proj.weight"]])
            gate_up = torch.cat([t[p + "mlp.gate_proj.weight"], t[p + "mlp.up_proj.weight"]])
            self.layers.append({
                "in_norm": t[p + "input_layernorm.weight"], "post_norm": t[p + "post_attention_layernorm.weight"],
                "ctx_norm": t[p + "dflare_norm.weight"],
                "qkv": tuple(fp8(qkv)), "wo": t[a + "o_proj.weight"],
                "q_norm": t[a + "q_norm.weight"], "k_norm": t[a + "k_norm.weight"],
                "wk_ctx": t[a + "k_proj_ctx.weight"], "wv_ctx": t[a + "v_proj_ctx.weight"],
                "k_norm_ctx": t[a + "k_norm_ctx.weight"],
                "gate_up": tuple(fp8(gate_up)), "w_down": t[p + "mlp.down_proj.weight"],
            })
            del qkv, gate_up
        self.final_norm = t["norm.weight"]
        self.mask_embed = t["mask_embed"]
        mh = "markov_head."
        self.w1 = t[mh + "markov_w1.weight"]
        self.rnn_w, self.rnn_b = t[mh + "rnn_proj.weight"], t[mh + "rnn_proj.bias"]
        self.f_norm = t[mh + "dflare_norm.weight"]
        # GEMV kernel operands: three input slots (only slot 0 used), one segment over all rows
        V = self.head[0].shape[0]
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=device)  # noqa: E731
        self.xin8 = z(3, BLOCK, D, dt=torch.bfloat16)
        self.xin7 = z(3, GAMMA, D, dt=torch.bfloat16)
        self.qkv_out = z(BLOCK, (HEADS + 2 * KV_HEADS) * HEAD_DIM)
        self.gu_out = z(BLOCK, self.layers[0]["gate_up"][0].shape[0])
        self.base_out = z(GAMMA, V)
        self.draft_logits = torch.full((GAMMA, V), float("-inf"), device=device)
        self.segments = {r: torch.tensor([[0, r, 0]], dtype=torch.int32, device=device)
                         for r in (self.qkv_out.shape[1], self.gu_out.shape[1], V)}
        self.none = torch.empty(0, device=device)
        self.ctx = min(max_ctx, DRAFT_CTX)
        kv = dict(dtype=torch.bfloat16, device=device)
        # Context caches. Local layers: a 1024-slot ring plus 8 rows for the block being drafted. Global layer:
        # a `ctx`-slot ring; the block's rows are written at P..P+7 (not yet committed, overwritten by commit).
        self.k_ctx = {l: torch.zeros((WINDOW + BLOCK) if l in LOCAL else self.ctx, KV_HEADS, HEAD_DIM, **kv)
                      for l in range(3)}
        self.v_ctx = {l: torch.zeros_like(self.k_ctx[l]) for l in range(3)}
        self.ctx_pos = {l: torch.full((WINDOW,), -1, dtype=torch.long, device=device) for l in LOCAL}
        self.f_rnn = torch.zeros(D, **kv)
        self.length = 0
        # draft inputs/outputs for graph replay
        self.anchor = torch.zeros(1, dtype=torch.long, device=device)
        self.P = torch.zeros(1, dtype=torch.long, device=device)
        self.temperature = torch.zeros(1, device=device)
        self.draft_tokens = torch.zeros(GAMMA, dtype=torch.long, device=device)
        self.bucket = self.ctx  # global-layer slots attended to (each draft graph has its own)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}  # context bucket -> draft graph, ascending
        # commit inputs for graph replay: the accepted rows of a verify call
        self.commit_feats = torch.zeros(BLOCK, 4, D, **kv)
        self.commit_start = torch.zeros(1, dtype=torch.long, device=device)
        self.commit_graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def reset(self):
        for l in LOCAL:
            self.ctx_pos[l].fill_(-1)
        self.length = 0

    @torch.inference_mode()
    def commit(self, feats: torch.Tensor, start: int):
        """Committed target rows (feats [n, 4, D] BF16 at positions start..start+n-1) become draft context."""
        n = feats.shape[0]
        if n in self.commit_graphs:
            self.commit_feats[:n].copy_(feats)
            self.commit_start.fill_(start)
            self.commit_graphs[n].replay()
        else:
            self._commit(feats, torch.arange(start, start + n, device=feats.device))
        self.length = start + n

    @torch.inference_mode()
    def _commit(self, feats: torch.Tensor, positions: torch.Tensor):
        n = feats.shape[0]
        for l, L in enumerate(self.layers):
            c = mix_rmsnorm(feats, self.alpha[l], L["ctx_norm"])
            k = bf(c @ L["wk_ctx"].T).view(-1, KV_HEADS, HEAD_DIM)
            k = rope(bf(rms(k, L["k_norm_ctx"])), positions)
            v = bf(c @ L["wv_ctx"].T).view(-1, KV_HEADS, HEAD_DIM)
            size = WINDOW if l in LOCAL else self.ctx
            keep = min(n, size)
            slots = positions[-keep:] % size
            self.k_ctx[l][slots] = k[-keep:]
            self.v_ctx[l][slots] = v[-keep:]
            if l in LOCAL:
                self.ctx_pos[l][slots] = positions[-keep:]
        self.f_rnn.copy_(mix_rmsnorm(feats[-1:], self.alpha[3], self.f_norm)[0])

    def _gemv(self, x, w_scale, xin, out):
        """out [n, R] = bf16(W x) for x [n, D] BF16 through the dense GEMV kernel (FP8 or BF16 rows)."""
        w, scale = w_scale
        xin[0].copy_(x)
        self.dense.proj(xin, w, scale if scale is not None else self.none, self.segments[out.shape[1]], out, False)
        return out

    def _attention(self, l, L, xn, positions):
        """Block rows attend bidirectionally to each other and to the committed context before P."""
        n = xn.shape[0]
        P = positions[:1]
        qkv = self._gemv(xn, L["qkv"], self.xin8, self.qkv_out)
        q, k, v = qkv.split([HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM], dim=-1)
        q = rope(bf(rms(bf(q).view(n, HEADS, HEAD_DIM), L["q_norm"])), positions)
        k = rope(bf(rms(bf(k).view(n, KV_HEADS, HEAD_DIM), L["k_norm"])), positions)
        v = bf(v).view(n, KV_HEADS, HEAD_DIM)
        keys, vals = self.k_ctx[l], self.v_ctx[l]
        if l in LOCAL:
            keys[WINDOW:] = k
            vals[WINDOW:] = v
            cpos = self.ctx_pos[l]
            ring_ok = (cpos >= 0) & (cpos < P) & (cpos >= P - WINDOW)
            valid = torch.cat([ring_ok, torch.ones(n, dtype=torch.bool, device=xn.device)])
        else:
            slots = positions % self.ctx
            keys.index_copy_(0, slots, k)
            vals.index_copy_(0, slots, v)
            keys, vals = keys[:self.bucket], vals[:self.bucket]
            valid = torch.arange(self.bucket, device=xn.device) < P + n  # all slots once P + n >= ctx
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], keys.transpose(0, 1)[None],
                                           vals.transpose(0, 1)[None], attn_mask=valid[None, None, None, :],
                                           enable_gqa=True)
        o = o[0].transpose(0, 1).reshape(n, HEADS * HEAD_DIM)
        return bf(bf(o) @ L["wo"].T)

    @torch.inference_mode()
    def _draft(self):
        """Graph body: drafts for positions P+1..P+7 from the anchor at P. Greedy when temperature is 0,
        otherwise Gumbel-max at the temperature (vLLM's probabilistic draft sampling). Returns
        (tokens int64 [7], logits FP32 [7, V]); the logits are the corrected, softcapped ones the verifier
        divides by the temperature, -inf outside each step's candidates."""
        dev = self.mask_embed.device
        x0 = torch.cat([bf(rms(self.embed[self.anchor].float(), self.embed_norm)),
                        self.mask_embed[None].expand(GAMMA, -1)])
        positions = self.P + torch.arange(BLOCK, device=dev)
        res = x0
        xn = bf(rms(res, self.layers[0]["in_norm"]))
        for l, L in enumerate(self.layers):
            res = bf(res.float() + self._attention(l, L, xn, positions).float())
            xn = bf(rms(res, L["post_norm"]))
            gu = self._gemv(xn, L["gate_up"], self.xin8, self.gu_out)
            gate, up = gu.chunk(2, dim=-1)
            m = bf(bf(F.silu(gate) * up) @ L["w_down"].T)
            res = bf(res.float() + m.float())
            nxt = self.layers[l + 1]["in_norm"] if l < 2 else self.final_norm
            xn = bf(rms(res, nxt))
        h = xn[1:]  # [7, D]
        base = self._gemv(h, self.head, self.xin7, self.base_out)  # [7, V], BF16 values
        # The Markov correction only reorders near the top: evaluate it on each step's top candidates.
        cand_base, cand = base.topk(self.CANDIDATES, dim=-1)  # [7, K]
        cand_w, cand_b = self.fold_w[cand], self.fold_b[cand]  # [7, K, 256], [7, K]
        state = torch.zeros(RANK, dtype=torch.bfloat16, device=dev)
        prev = self.anchor
        sampled = self.temperature > 0
        tokens, all_logits = [], []
        for i in range(GAMMA):
            z = torch.cat([state, self.w1[prev][0], h[i], self.f_rnn])
            raw = bf(z @ self.rnn_w.T + self.rnn_b).float()
            gate = bf(torch.sigmoid(raw[:RANK]))
            cell = bf(torch.tanh(raw[RANK:2 * RANK]))
            state = bf(bf(gate.float() * state.float()).float() + bf(bf(1 - gate.float()).float() * cell.float()).float())
            latent = bf(torch.tanh(raw[2 * RANK:]))
            bias = bf(cand_b[i].float() + (cand_w[i] @ latent).float())
            logits = bf(30 * torch.tanh(bf(cand_base[i] + bias.float()).float() / 30)).float()  # [K]
            u = torch.rand(logits.shape, device=dev).clamp_(1e-20, 1.0)
            noisy = logits / self.temperature.clamp(min=1e-6) - torch.log(-torch.log(u))
            pick = torch.where(sampled, noisy.argmax(), logits.argmax())
            prev = cand[i, pick].view(1)
            tokens.append(prev)
            all_logits.append(logits)
        self.draft_logits.fill_(float("-inf"))
        self.draft_logits.scatter_(1, cand, torch.stack(all_logits))
        self.draft_tokens.copy_(torch.cat(tokens))
        return self.draft_tokens, self.draft_logits

    def capture(self):
        """CUDA graphs: the draft step once per context bucket (the global layer attends to the first
        `bucket` slots, so short contexts do not pay for the whole ring), and the commit of 1..8 verified rows.
        All share one memory pool; they never run concurrently and their outputs are persistent buffers."""
        dev = self.mask_embed.device
        buckets, b = [], 1024
        while b < self.ctx:
            buckets.append(b)
            b *= 2
        buckets.append(self.ctx)
        commit = {n: (lambda n=n: self._commit(self.commit_feats[:n], self.commit_start + torch.arange(n, device=dev)))
                  for n in range(1, BLOCK + 1)}
        pool = torch.cuda.graph_pool_handle()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):  # warm-up: lazy initialisation stays outside capture
            for B in buckets:
                self.bucket = B
                self._draft()
            for fn in commit.values():
                fn()
        torch.cuda.current_stream().wait_stream(s)
        for B in buckets:
            self.bucket = B
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._draft()
            self.graphs[B] = g
        self.bucket = self.ctx
        for n, fn in commit.items():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                fn()
            self.commit_graphs[n] = g
        self.reset()  # warm-up and capture wrote scratch context rows

    def draft(self, anchor: int, P: int, temperature: float):
        """(tokens [7], logits [7, V]) for the anchor token at position P (fixed buffers, overwritten by the
        next draft)."""
        self.anchor.fill_(anchor)
        self.P.fill_(P)
        self.temperature.fill_(temperature)
        if not self.graphs:
            return self._draft()
        # the smallest bucket holding the block; the whole ring once the context reaches it
        self.graphs[next((b for b in self.graphs if b >= P + BLOCK), self.ctx)].replay()
        return self.draft_tokens, self.draft_logits
